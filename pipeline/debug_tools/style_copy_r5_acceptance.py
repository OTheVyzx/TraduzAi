"""Create authenticated R5 runs and compose their fail-closed acceptance evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Mapping


SOURCE_EXCLUSIONS = (
    "pipeline/tests/**", "pipeline/venv/**", "pipeline/.pytest_cache/**",
    "pipeline/.tmp/**", "pipeline/dummy-debug/**", "**/__pycache__/**",
)


class AcceptanceBundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class InitRunArgs:
    repo_root: Path
    runs_root: Path
    run_prefix: str
    source_manifest: Path
    benchmark_spec: Path
    matrix: Path
    matrix_inputs: Path
    seed: int
    active_manifest: Path


@dataclass(frozen=True)
class InitRunResult:
    run_root: Path
    acceptance_bundle_path: Path
    acceptance_bundle_id: str
    active_manifest_path: Path
    acceptance_bundle: Mapping[str, Any]


def _canonical_sha256(value: Any, *, omit: str | None = None) -> str:
    payload = {key: item for key, item in value.items() if key != omit} if omit and isinstance(value, Mapping) else value
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_input(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(row for row in path.rglob("*") if row.is_file()):
        digest.update(item.relative_to(path).as_posix().encode()); digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item))); digest.update(b"\n")
    return digest.hexdigest()


def verify_inputs(path: Path, *, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    findings: list[dict[str, Any]] = []
    resolved: dict[str, Any] = {}
    if payload.get("hash_algorithm") != "sha256-tree-v1" or not isinstance(payload.get("inputs"), dict):
        findings.append({"code": "input_manifest_schema_invalid"})
    for key, row in sorted((payload.get("inputs") or {}).items()):
        env_name = str((row or {}).get("environment_variable") or "")
        raw = (environ or os.environ).get(env_name, "")
        candidate = Path(raw).expanduser().resolve() if raw else None
        expected = str((row or {}).get("expected_sha256") or "").lower()
        if candidate is None or not candidate.exists():
            findings.append({"code": "input_missing", "input_key": key}); continue
        input_type = str((row or {}).get("input_type") or "")
        if (input_type == "file" and not candidate.is_file()) or (input_type == "directory" and not candidate.is_dir()):
            findings.append({"code": "input_type_mismatch", "input_key": key}); continue
        actual = _sha256_input(candidate)
        if actual != expected:
            findings.append({"code": "input_hash_mismatch", "input_key": key, "expected": expected, "actual": actual})
        resolved[key] = {"path": str(candidate), "content_sha256": actual, "environment_variable": env_name}
    return {"schema_version": 1, "status": "BLOCK" if findings else "PASS", "inputs": resolved, "findings": findings}


def _excluded(relative: str, exclusions: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in exclusions)


def _expand_execution_sources(repo_root: Path, manifest: Mapping[str, Any]) -> tuple[list[str], tuple[str, ...]]:
    if int(manifest.get("schema_version") or 0) != 1:
        raise AcceptanceBundleError("source manifest schema mismatch")
    exclusions = tuple(str(item) for item in manifest.get("exclusions") or ())
    if exclusions != SOURCE_EXCLUSIONS:
        raise AcceptanceBundleError("source manifest exclusion allowlist mismatch")
    explicit = [str(item) for item in manifest.get("sources") or ()]
    if len(explicit) != len(set(explicit)):
        raise AcceptanceBundleError("duplicate_source")
    explicit_set = set(explicit)
    expanded = set(explicit)
    for raw_root in manifest.get("closure_roots") or ():
        root_relative = str(raw_root).replace("\\", "/").strip("/")
        root = (repo_root / root_relative).resolve()
        try:
            root.relative_to(repo_root)
        except ValueError as exc:
            raise AcceptanceBundleError("outside_repo") from exc
        for item in root.rglob("*.py"):
            relative = item.resolve().relative_to(repo_root).as_posix()
            if not _excluded(relative, exclusions):
                expanded.add(relative)
    normalized: list[str] = []
    for relative in sorted(expanded):
        if Path(relative).is_absolute() or ".." in Path(relative).parts or (
            _excluded(relative, exclusions) and relative not in explicit_set
        ):
            raise AcceptanceBundleError(f"outside_repo_or_excluded:{relative}")
        resolved = (repo_root / relative).resolve()
        try:
            resolved.relative_to(repo_root)
        except ValueError as exc:
            raise AcceptanceBundleError(f"outside_repo:{relative}") from exc
        if not resolved.is_file():
            raise AcceptanceBundleError(f"missing_required_source:{relative}")
        normalized.append(relative)
    return normalized, exclusions


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _source_snapshot(repo_root: Path, paths: list[str]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for relative in paths:
        path = repo_root / relative
        status = _git(repo_root, "status", "--porcelain", "--", relative)
        blob_line = _git(repo_root, "ls-files", "-s", "--", relative).split()
        snapshot[relative] = {
            "content_sha256": sha256_file(path),
            "git_blob_sha256": blob_line[1] if len(blob_line) >= 2 else None,
            "dirty": bool(status),
        }
    return snapshot


def _runtime_contract(repo_root: Path) -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "cv2", "matplotlib", "fontTools"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "unavailable"
    font_map = repo_root / "fonts" / "font-map.json"
    catalog_rows = []
    if font_map.is_file():
        for row in json.loads(font_map.read_text(encoding="utf-8-sig")).get("available") or []:
            name = str(row.get("arquivo") or "")
            matches = sorted(repo_root.joinpath("fonts").rglob(name))
            if matches:
                catalog_rows.append({"name": name, "sha256": sha256_file(matches[0])})
    versions["font_catalog_sha256"] = _canonical_sha256(catalog_rows)
    return versions


def init_run(args: InitRunArgs) -> InitRunResult:
    repo_root = Path(args.repo_root).resolve(); runs_root = Path(args.runs_root).resolve()
    try:
        runs_root.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise AcceptanceBundleError("runs_root must be outside repo")
    input_report = verify_inputs(args.matrix_inputs)
    if input_report["status"] != "PASS":
        raise AcceptanceBundleError("matrix inputs failed authentication")
    source_manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8-sig"))
    paths, exclusions = _expand_execution_sources(repo_root, source_manifest)
    sources = _source_snapshot(repo_root, paths)
    git_head = _git(repo_root, "rev-parse", "HEAD")
    config_hashes = {
        "source_manifest_sha256": _canonical_sha256(source_manifest),
        "benchmark_spec_sha256": _canonical_sha256(json.loads(Path(args.benchmark_spec).read_text(encoding="utf-8-sig"))),
        "matrix_sha256": _canonical_sha256(json.loads(Path(args.matrix).read_text(encoding="utf-8-sig"))),
        "matrix_inputs_sha256": _canonical_sha256(json.loads(Path(args.matrix_inputs).read_text(encoding="utf-8-sig"))),
    }
    revision_sha256 = _canonical_sha256({"git_head": git_head, "execution_sources": sources, **config_hashes})
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    run_name = f"{args.run_prefix}_{attempt}"
    runs_root.mkdir(parents=True, exist_ok=True)
    final_root = runs_root / run_name
    staging = Path(tempfile.mkdtemp(prefix=f".{run_name}-", dir=runs_root))
    try:
        bundle = {
            "schema_version": 1, "attempt_id": attempt, "repo_root": str(repo_root),
            "runs_root": str(runs_root), "run_root": str(final_root), "seed": int(args.seed),
            "git_head": git_head, "revision_sha256": revision_sha256,
            "execution_sources": sources, "source_exclusions": list(exclusions),
            "resolved_inputs": input_report["inputs"], "runtime": _runtime_contract(repo_root),
            **config_hashes,
        }
        bundle["acceptance_bundle_id"] = _canonical_sha256(bundle)
        bundle_path = staging / "acceptance_bundle.json"
        bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, final_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    bundle_path = final_root / "acceptance_bundle.json"
    active = {
        "schema_version": 1, "attempt_id": attempt, "run_root": str(final_root),
        "acceptance_bundle_path": str(bundle_path), "acceptance_bundle_sha256": sha256_file(bundle_path),
        "acceptance_bundle_id": bundle["acceptance_bundle_id"], "runs_root": str(runs_root),
    }
    active["active_manifest_sha256"] = _canonical_sha256(active)
    active_path = Path(args.active_manifest); active_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = active_path.with_suffix(active_path.suffix + ".tmp")
    temporary.write_text(json.dumps(active, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, active_path)
    return InitRunResult(final_root, bundle_path, bundle["acceptance_bundle_id"], active_path, bundle)


def load_active_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    claimed = str(payload.get("active_manifest_sha256") or "")
    if claimed != _canonical_sha256(payload, omit="active_manifest_sha256"):
        raise AcceptanceBundleError("active manifest hash mismatch")
    bundle_path = Path(str(payload.get("acceptance_bundle_path") or "")).resolve()
    runs_root = Path(str(payload.get("runs_root") or "")).resolve()
    try:
        bundle_path.relative_to(runs_root)
    except ValueError as exc:
        raise AcceptanceBundleError("active bundle path escapes runs root") from exc
    if not bundle_path.is_file() or sha256_file(bundle_path) != payload.get("acceptance_bundle_sha256"):
        raise AcceptanceBundleError("active bundle bytes mismatch")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
    if bundle.get("acceptance_bundle_id") != payload.get("acceptance_bundle_id"):
        raise AcceptanceBundleError("active bundle identity mismatch")
    return payload


def _finding(findings: list[dict[str, Any]], code: str, **detail: Any) -> None:
    findings.append({"code": code, **detail})


def evaluate_acceptance(*, acceptance_bundle: Mapping[str, Any], synthetic: Mapping[str, Any], owner_qa: Mapping[str, Any], inspection: Mapping[str, Any], matrix_execution: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    bundle_id = str(acceptance_bundle.get("acceptance_bundle_id") or "")
    revision = str(acceptance_bundle.get("revision_sha256") or "")
    source_manifest = str(acceptance_bundle.get("source_manifest_sha256") or "")
    producers = {"synthetic": synthetic, "owner_qa": owner_qa, "inspection": inspection, "matrix_execution": matrix_execution}
    run_ids: set[str] = set()
    for name, payload in producers.items():
        if payload.get("acceptance_bundle_id") != bundle_id or payload.get("revision_sha256") != revision or payload.get("source_manifest_sha256") != source_manifest:
            _finding(findings, "producer_bundle_binding_mismatch", producer=name)
        run_id = str(payload.get("producer_run_id") or "")
        if not run_id or run_id in run_ids:
            _finding(findings, "producer_run_id_invalid", producer=name)
        run_ids.add(run_id)
    policy = thresholds.get("validation_thresholds") if isinstance(thresholds.get("validation_thresholds"), Mapping) else thresholds
    synthetic_policy = policy.get("synthetic") if isinstance(policy.get("synthetic"), Mapping) else policy
    score = synthetic.get("score") if isinstance(synthetic.get("score"), Mapping) else {}
    for section, rule in (("font_top1", "font_top1_min"), ("font_top3", "font_top3_min")):
        metric = score.get(section) if isinstance(score.get(section), Mapping) else None
        if not metric or int(metric.get("evaluated") or 0) == 0:
            _finding(findings, "required_metric_missing", metric=section)
        elif float(metric.get("rate") or 0) < float(synthetic_policy.get(rule, 1.0)):
            _finding(findings, "threshold_breach", metric=section)
    fill = score.get("fill_delta_e_2000") if isinstance(score.get("fill_delta_e_2000"), Mapping) else None
    if not fill or int(fill.get("count") or 0) == 0:
        _finding(findings, "required_metric_missing", metric="fill_delta_e_2000")
    else:
        if float(fill.get("median") or 999) > float(synthetic_policy.get("fill_delta_e_2000_median_max", 8)):
            _finding(findings, "threshold_breach", metric="fill_delta_e_2000.median")
        if float(fill.get("p95") or 999) > float(synthetic_policy.get("fill_delta_e_2000_p95_max", 12)):
            _finding(findings, "threshold_breach", metric="fill_delta_e_2000.p95")
    owner_policy = policy.get("owner_qa") if isinstance(policy.get("owner_qa"), Mapping) else {}
    for section, threshold_name in (("safe_containment", "safe_containment_min"), ("effect_containment", "effect_containment_min")):
        metric = owner_qa.get(section) if isinstance(owner_qa.get(section), Mapping) else None
        if not metric or int(metric.get("evaluated") or 0) == 0:
            _finding(findings, "zero_metric_denominator", metric=section)
        elif float(metric.get("rate") or 0) < float(owner_policy.get(threshold_name, 1.0)):
            _finding(findings, "threshold_breach", metric=section)
    catastrophic = owner_qa.get("catastrophic_mismatches") or {}
    if int(catastrophic.get("evaluated") or 0) == 0 or int(catastrophic.get("count") or 0) > int(owner_policy.get("catastrophic_mismatch_max", 0)):
        _finding(findings, "catastrophic_metric_block")
    if str(owner_qa.get("status") or "BLOCK") != "PASS" or float(owner_qa.get("materialization_observation_coverage") or 0) < 1.0:
        _finding(findings, "owner_qa_block")
    inspection_policy = policy.get("inspection") if isinstance(policy.get("inspection"), Mapping) else {}
    if str(inspection.get("status") or "BLOCK") != "PASS":
        _finding(findings, "inspection_block")
    for section, threshold_name in (("owners", "owner_go_rate_min"), ("speech", "speech_go_rate_min")):
        metric = inspection.get(section) if isinstance(inspection.get(section), Mapping) else None
        if not metric or int(metric.get("evaluated") or 0) == 0:
            _finding(findings, "zero_metric_denominator", metric=section)
        elif float(metric.get("go_rate") or 0) < float(inspection_policy.get(threshold_name, 1.0)):
            _finding(findings, "threshold_breach", metric=section)
    for category in inspection.get("required_holdout_categories") or []:
        metric = (inspection.get("categories") or {}).get(category, {})
        if int(metric.get("evaluated") or 0) == 0 or float(metric.get("go_rate") or 0) < float(inspection_policy.get("category_go_rate_min", 1.0)):
            _finding(findings, "holdout_category_block", category=category)
    coverage = inspection.get("inspection_coverage") or {}
    if int(coverage.get("evaluated") or 0) == 0 or float(coverage.get("rate") or 0) < float(inspection_policy.get("coverage_min", 1.0)):
        _finding(findings, "inspection_coverage_block")
    functional_go = str(matrix_execution.get("functional_status") or "BLOCK") == "GO"
    if not functional_go or not matrix_execution.get("entries"):
        _finding(findings, "functional_matrix_block")
    for entry in matrix_execution.get("entries") or []:
        if str(entry.get("export_gate_status") or "BLOCK") != "PASS" or len(str(entry.get("export_gate_sha256") or "")) != 64:
            _finding(findings, "entry_export_gate_block", entry_id=entry.get("entry_id"))
    style_go = not any(row["code"] in {"owner_qa_block", "catastrophic_metric_block", "threshold_breach", "required_metric_missing", "zero_metric_denominator"} for row in findings)
    inspection_go = not any(row["code"].startswith("inspection") or row["code"] == "holdout_category_block" for row in findings)
    status = "PASS" if not findings else "BLOCK"
    return {
        "schema_version": 1, "status": status,
        "acceptance_bundle_id": bundle_id, "revision_sha256": revision,
        "producer_run_ids": sorted(run_ids), "findings": findings,
        "verdicts": {"style": "GO" if style_go else "NO-GO", "functional": "GO" if functional_go else "NO-GO", "inspection": "GO" if inspection_go else "NO-GO", "overall": "GO" if status == "PASS" else "NO-GO"},
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-run")
    for flag in ("repo-root", "runs-root", "run-prefix", "source-manifest", "benchmark-spec", "matrix", "matrix-inputs", "active-manifest"):
        init.add_argument(f"--{flag}", required=True)
    init.add_argument("--seed", required=True, type=int)
    active = sub.add_parser("validate-active"); active.add_argument("--active-manifest", required=True, type=Path)
    verify = sub.add_parser("verify-inputs"); verify.add_argument("--matrix-inputs", required=True, type=Path)
    evaluate = sub.add_parser("evaluate")
    for flag in ("acceptance-bundle", "benchmark-summary", "owner-qa-summary", "inspection-summary", "matrix-execution-summary", "thresholds", "output"):
        evaluate.add_argument(f"--{flag}", required=True, type=Path)
    evaluate.add_argument("--mode", choices=("shadow", "enforce"), default="enforce")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-inputs":
        report = verify_inputs(args.matrix_inputs); print(json.dumps(report)); return 0 if report["status"] == "PASS" else 2
    if args.command == "validate-active":
        print(json.dumps(load_active_manifest(args.active_manifest))); return 0
    if args.command == "init-run":
        result = init_run(InitRunArgs(Path(args.repo_root), Path(args.runs_root), args.run_prefix, Path(args.source_manifest), Path(args.benchmark_spec), Path(args.matrix), Path(args.matrix_inputs), args.seed, Path(args.active_manifest)))
        print(result.run_root); return 0
    result = evaluate_acceptance(
        acceptance_bundle=_read(args.acceptance_bundle), synthetic=_read(args.benchmark_summary),
        owner_qa=_read(args.owner_qa_summary), inspection=_read(args.inspection_summary),
        matrix_execution=_read(args.matrix_execution_summary), thresholds=_read(args.thresholds),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 2 if args.mode == "enforce" and result["status"] == "BLOCK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
