"""Canonical, hash-bound Consumer Fast V1 full-chapter entrypoint."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


RUNTIME_ID = "consumer-fast-v1"
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = PIPELINE_ROOT.parent
RUNTIME_ROOT = PIPELINE_ROOT / "consumer_fast" / "runtime"
for _path in (PIPELINE_ROOT, RUNTIME_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


class ConsumerFastPreflightError(RuntimeError):
    """The selected runtime cannot be proven before physical processing."""


class ConsumerFastIntegrityError(RuntimeError):
    """The terminal project is incomplete or references invalid artifacts."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_materialized_base(worktree: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str], list[dict[str, str]]]:
    """Check full content; accept only declared CRLF-to-LF equivalence."""
    files = manifest["files"]
    overrides = manifest.get("materialized_overrides", {})
    equivalent = manifest.get("line_ending_equivalent_paths", [])
    if (not isinstance(equivalent, list) or len(equivalent) != len(set(equivalent))
            or any(path not in files for path in equivalent)):
        raise ConsumerFastPreflightError("invalid line-ending equivalence manifest")
    allowed = set(equivalent)
    mismatches: list[dict[str, str]] = []
    raw_hashes: dict[str, str] = {}
    accepted: list[dict[str, str]] = []
    for relative, source_hash in files.items():
        materialized = worktree / relative
        expected = overrides.get(relative, source_hash)
        if not materialized.is_file():
            mismatches.append({"path": relative, "expected": expected, "actual": "missing"})
            continue
        data = materialized.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        raw_hashes[relative] = actual
        if actual == expected:
            continue
        if relative in allowed and b"\r\n" in data:
            normalized = data.replace(b"\r\n", b"\n")
            if b"\r" not in normalized and hashlib.sha256(normalized).hexdigest() == expected:
                accepted.append({"path": relative, "expected_lf_sha256": expected, "raw_sha256": actual})
                continue
        mismatches.append({"path": relative, "expected": expected, "actual": actual})
    return mismatches, raw_hashes, accepted


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _models_hash(models_dir: Path) -> str:
    if not models_dir.is_dir():
        raise ConsumerFastPreflightError(f"models_dir inválido: {models_dir}")
    inventory = [
        {"path": str(path.relative_to(models_dir)).replace("\\", "/"), "size": path.stat().st_size}
        for path in sorted(models_dir.rglob("*"))
        if path.is_file()
    ]
    return _canonical_hash(inventory)


def _git_value(worktree: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(worktree), *args],
            text=True,
            encoding="utf-8",
            errors="strict",
        ).rstrip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConsumerFastPreflightError(f"identidade Git indisponível: {exc}") from exc


def build_preflight(config: dict[str, Any], *, pipeline_root: Path | None = None) -> dict[str, Any]:
    """Resolve the canonical runtime, verify its materialized base and freeze a plan."""
    root = (pipeline_root or PIPELINE_ROOT).resolve()
    worktree = root.parent.resolve()
    if config.get("runtime_id") != RUNTIME_ID:
        raise ConsumerFastPreflightError(f"runtime_id deve ser {RUNTIME_ID}")
    if config.get("translation_backend") == "hy_mt2_gguf_local":
        from urllib.parse import urlparse
        endpoint = urlparse(str(config.get("hy_mt2_url") or ""))
        if (endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost"}
                or endpoint.username or endpoint.password or not config.get("offline_context_only")):
            raise ConsumerFastPreflightError("HY-MT2 Consumer Fast requires an offline loopback endpoint")
    if config.get("owner_graph_mode") != "enforce" or config.get("style_copy_mode") != "shadow":
        raise ConsumerFastPreflightError("owner_graph_mode=enforce e style_copy_mode=shadow são obrigatórios")

    source = Path(str(config.get("source_path", ""))).resolve()
    if not source.is_file():
        raise ConsumerFastPreflightError(f"source_path inválido: {source}")
    models_dir = Path(str(config.get("models_dir", ""))).resolve()

    manifest_path = root / "consumer_fast" / "BASE_SOURCES.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches, materialized_raw_hashes, line_ending_equivalent = _verify_materialized_base(worktree, manifest)
    if mismatches:
        raise ConsumerFastPreflightError(f"BASE_SOURCES divergente: {mismatches}")

    module_names = (
        "consumer_fast_core",
        "main",
        "consumer_operational_recovery",
        "consumer_operational_publish",
        "consumer_fast_project",
        "integration_v1.contracts",
        "integration_v1.providers",
        "vision_runtime.analysis_payload",
        "vision_runtime.cache_key",
        "vision_runtime.cache_reader",
        "vision_runtime.page_record",
        "vision_runtime.stage_cache",
        "consumer_fast.vision_cache",
        "ownership.coverage",
        "ownership.ocr_contract",
        "ownership.consensus_v2",
        "vision_stack.ocr",
        "qa.partial_delivery",
        "typesetter.renderer",
        "typesetter.fixed_font_family",
        "typesetter.recipe_contract",
        "consumer_fast.provider_adapter",
        "consumer_fast.physical_executor",
        "extractor.extractor",
        "inpainter",
        "ownership.chapter_contract",
        "strip.run",
        "strip.process_bands",
        "strip.page_pipeline",
        "strip.detect_balloons",
        "strip.experimental_mayo",
        "inpainter.experimental_fastfill",
        "ownership.render_geometry",
        "ownership.experimental_ocr_region",
        "translator.translate",
        "translator.hy_mt2_local",
        "vision_stack.runtime",
        "vision_stack.engine_presets",
        "project_writer",
        "qa.export_gate",
    )
    resolved_modules: list[dict[str, Any]] = []
    for name in module_names:
        module = importlib.import_module(name)
        module_path = Path(str(module.__file__)).resolve()
        inside = module_path.is_relative_to(worktree)
        if not inside:
            raise ConsumerFastPreflightError(f"módulo fora do worktree: {name} -> {module_path}")
        resolved_modules.append(
            {"name": name, "path": str(module_path), "sha256": _sha256_file(module_path), "inside_worktree": inside}
        )

    from integration_v1.orchestrator import build_execution_plan

    provider_hash = _canonical_hash(
        {row["name"]: row["sha256"] for row in resolved_modules if row["name"].startswith("integration_v1")}
    )
    runtime_binding = {
        "base_sources_manifest": _sha256_file(manifest_path),
        "base_materialized_raw_sha256": _canonical_hash(materialized_raw_hashes),
        "chapter_runner": _sha256_file(Path(__file__).resolve()),
        "provider_adapter": next(
            row["sha256"] for row in resolved_modules if row["name"] == "consumer_fast.provider_adapter"
        ),
        "integrated_modules": {
            row["name"]: row["sha256"] for row in resolved_modules
            if row["name"] in {
                "strip.run", "strip.process_bands", "strip.page_pipeline",
                "strip.detect_balloons", "strip.experimental_mayo",
                "inpainter.experimental_fastfill", "ownership.render_geometry",
                "ownership.experimental_ocr_region", "translator.translate",
                "translator.hy_mt2_local", "typesetter.fixed_font_family",
            }
        },
    }
    from typesetter.fixed_font_family import fixed_font_path
    runtime_binding["fixed_font_asset_sha256"] = _sha256_file(fixed_font_path())
    dependency_hashes = {
        "source": _sha256_file(source),
        "config": _canonical_hash(config),
        "glossary": _canonical_hash(config.get("glossario", {})),
        "context": _canonical_hash(config.get("contexto", {})),
        "providers": provider_hash,
        "models": _models_hash(models_dir),
        "runtime_recipe": _canonical_hash(runtime_binding),
    }
    execution_plan = build_execution_plan(
        {"analysis": True, "continuity": True, "narration_multiline": True, "operational_recovery": True},
        policy_versions={"consumer_fast": "v1", "owner_graph": "enforce", "style_copy": "shadow"},
        dependency_hashes=dependency_hashes,
    )
    dirty_paths = [
        line[3:] for line in _git_value(worktree, "status", "--porcelain=v1").splitlines() if line
    ]
    absolute_import_pattern = re.compile(
        r"(?:sys\.path|spec_from_file_location|SourceFileLoader).{0,160}(?:[A-Za-z]:[\\/]|/home/)",
        re.IGNORECASE,
    )
    experimental_absolute_imports = []
    for row in resolved_modules:
        source_text = Path(row["path"]).read_text(encoding="utf-8", errors="replace")
        if absolute_import_pattern.search(source_text):
            experimental_absolute_imports.append(row["path"])
    if experimental_absolute_imports:
        raise ConsumerFastPreflightError(
            f"imports experimentais absolutos detectados: {experimental_absolute_imports}"
        )
    by_name = {row["name"]: row for row in resolved_modules}
    return {
        "status": "PASS",
        "runtime_id": RUNTIME_ID,
        "worktree": str(worktree),
        "git": {
            "branch": _git_value(worktree, "branch", "--show-current"),
            "head": _git_value(worktree, "rev-parse", "HEAD"),
            "tree_state": "clean" if not dirty_paths else "dirty_justified",
            "changed_paths": dirty_paths,
            "dirty_justification": (
                None if not dirty_paths else "implementação Consumer Fast V1 ainda não commitada"
            ),
        },
        "legacy_fallback_allowed": False,
        "translation_provider_policy": "consumer-fast-bounded-owner-v1",
        "execution_plan": execution_plan,
        "resolved_modules": resolved_modules,
        "runtime_binding": runtime_binding,
        "contract_origins": {
            "vision": {
                "module": "vision_runtime.analysis_payload",
                "path": by_name["vision_runtime.analysis_payload"]["path"],
                "sha256": by_name["vision_runtime.analysis_payload"]["sha256"],
            },
            "renderer": {
                "module": "typesetter.renderer",
                "path": by_name["typesetter.renderer"]["path"],
                "sha256": by_name["typesetter.renderer"]["sha256"],
            },
        },
        "experimental_absolute_imports": experimental_absolute_imports,
        "base_sources": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "checked": len(manifest["files"]),
            "mismatches": mismatches,
            "line_ending_equivalent": line_ending_equivalent,
            "materialized_raw_sha256": materialized_raw_hashes,
        },
    }


def _resolve_artifact(project_root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConsumerFastIntegrityError(f"{field} ausente")
    candidate = (project_root / value).resolve()
    if not candidate.is_relative_to(project_root.resolve()):
        raise ConsumerFastIntegrityError(f"{field} escapa do projeto: {value}")
    if not candidate.is_file():
        raise ConsumerFastIntegrityError(f"{field} não encontrado: {value}")
    return candidate


def _review_blockers(project: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for page in project.get("paginas", []):
        for layer in page.get("text_layers", page.get("textos", [])):
            action = str(layer.get("route_action", "")).lower()
            if action in {"review_required", "block", "blocked"} or layer.get("review_required") is True:
                blockers.append(
                    {
                        "page": page.get("numero"),
                        "owner_id": layer.get("owner_id") or layer.get("id"),
                        "reason": layer.get("review_reason") or action,
                    }
                )
    return blockers


def finalize_project(project_path: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    """Validate referenced rasters and atomically publish a reopenable terminal project."""
    from consumer_fast_core import atomic_json

    path = Path(project_path).resolve()
    project = json.loads(path.read_text(encoding="utf-8"))
    pages = project.get("paginas")
    if not isinstance(pages, list) or not pages:
        raise ConsumerFastIntegrityError("paginas ausentes no projeto terminal")
    originals: list[dict[str, Any]] = []
    for page in pages:
        original = _resolve_artifact(path.parent, page.get("arquivo_original"), "arquivo_original")
        _resolve_artifact(path.parent, page.get("arquivo_traduzido"), "arquivo_traduzido")
        originals.append(
            {"page": page.get("numero"), "path": str(original), "sha256": _sha256_file(original)}
        )

    blockers = _review_blockers(project)
    qa = project.setdefault("qa", {})
    gate = qa.setdefault("export_gate", {})
    criticals = int(gate.get("critical_issue_count", gate.get("critical_flag_count", 0)) or 0)
    legitimately_approved = (
        gate.get("status") == "PASS"
        and gate.get("allowed") is True
        and project.get("verified") is True
        and not blockers
        and criticals == 0
    )
    hard_block = (
        gate.get("status") == "BLOCK"
        or criticals > 0
        or any(str(item.get("reason") or "").lower() in {"block", "blocked"} for item in blockers)
    )
    if hard_block:
        gate["status"] = "BLOCK"
        gate["allowed"] = False
    elif not legitimately_approved:
        # Keep the reviewed gate's evidence and warnings. A review state is not
        # quality approval, but it must not be silently rewritten to BLOCK.
        gate["status"] = "REVIEW"
        gate["allowed"] = False
        gate["needs_review"] = True
    gate["critical_issue_count"] = criticals
    gate["critical_flag_count"] = int(gate.get("critical_flag_count", criticals) or criticals)
    gate["blocker_count"] = len(blockers)

    project["runtime_id"] = RUNTIME_ID
    project["runtime_profile"] = "consumer_fast"
    project["consumer_fast"] = {
        "execution_plan": preflight["execution_plan"],
        "preflight": {
            "status": preflight["status"],
            "base_sources": preflight["base_sources"],
            "resolved_modules": preflight["resolved_modules"],
            "legacy_fallback_allowed": False,
        },
        "originals": originals,
        "blockers": blockers,
    }
    project["blocker_count"] = len(blockers)
    project["critical_issue_count"] = criticals
    if legitimately_approved:
        project["completion_status"] = "complete"
        project["output_review_state"] = "approved"
    else:
        project["completion_status"] = project.get("completion_status") or "complete_with_review"
        project["output_review_state"] = "awaiting_review"

    atomic_json(path, project)
    reopened = json.loads(path.read_text(encoding="utf-8"))
    round_trip = reopened == project
    if not round_trip:
        raise ConsumerFastIntegrityError("round-trip do project.json divergiu")
    return {
        "status": reopened["output_review_state"],
        "runtime_id": RUNTIME_ID,
        "project_path": str(path),
        "project_sha256": _sha256_file(path),
        "round_trip": round_trip,
        "originals": originals,
        "blocker_count": len(blockers),
        "critical_issue_count": criticals,
        "qa": reopened["qa"],
    }


def _emit_json(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _physical_pipeline(config_path: Path) -> Path:
    """Run Consumer Fast's explicit physical executor; there is no general-pipeline fallback."""
    from consumer_fast.physical_executor import execute_config

    return execute_config(config_path)


def run_chapter(
    config_path: Path,
    *,
    physical_executor: Callable[[Path], Path] | None = None,
    emit_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute preflight -> physical pipeline -> terminal validation and receipt."""
    from consumer_fast_core import atomic_json

    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    preflight = build_preflight(config)
    work_dir = Path(config["work_dir"]).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(work_dir / "consumer_fast_v1_preflight.json", preflight)
    config["consumer_fast_execution_plan"] = preflight["execution_plan"]
    config["translation_provider_policy"] = "consumer-fast-bounded-owner-v1"
    config["legacy_translation_fallback_allowed"] = False
    atomic_json(config_file, config)
    project_path = Path((physical_executor or _physical_pipeline)(config_file)).resolve()
    result = finalize_project(project_path, preflight)
    physical_evidence = project_path.parent / "consumer_fast_physical_execution.json"
    if physical_evidence.is_file():
        from project_writer import validate_project_consistency

        validate_project_consistency(json.loads(project_path.read_text(encoding="utf-8")))
        result["physical_execution"] = json.loads(physical_evidence.read_text(encoding="utf-8"))
        result["physical_execution"]["project_sha256_terminal"] = result["project_sha256"]
    receipt_path = work_dir / "consumer_fast_v1_receipt.json"
    result["receipt_path"] = str(receipt_path)
    atomic_json(receipt_path, result)
    event = {"type": "complete", "output_path": str(project_path), **result}
    (emit_event or _emit_json)(event)
    return result


def main() -> int:
    if len(sys.argv) != 2:
        raise ConsumerFastPreflightError("uso: chapter_runner.py <config.json>")
    run_chapter(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _emit_json({"type": "error", "runtime_id": RUNTIME_ID, "message": str(exc)})
        raise
