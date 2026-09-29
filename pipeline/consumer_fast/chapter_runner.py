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
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConsumerFastPreflightError(f"identidade Git indisponível: {exc}") from exc


def build_preflight(config: dict[str, Any], *, pipeline_root: Path | None = None) -> dict[str, Any]:
    """Resolve the canonical runtime, verify its materialized base and freeze a plan."""
    root = (pipeline_root or PIPELINE_ROOT).resolve()
    worktree = root.parent.resolve()
    if config.get("runtime_id") != RUNTIME_ID:
        raise ConsumerFastPreflightError(f"runtime_id deve ser {RUNTIME_ID}")
    if config.get("owner_graph_mode") != "enforce" or config.get("style_copy_mode") != "shadow":
        raise ConsumerFastPreflightError("owner_graph_mode=enforce e style_copy_mode=shadow são obrigatórios")

    source = Path(str(config.get("source_path", ""))).resolve()
    if not source.is_file():
        raise ConsumerFastPreflightError(f"source_path inválido: {source}")
    models_dir = Path(str(config.get("models_dir", ""))).resolve()

    manifest_path = root / "consumer_fast" / "BASE_SOURCES.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overrides = manifest.get("materialized_overrides", {})
    mismatches: list[dict[str, str]] = []
    for relative, source_hash in manifest["files"].items():
        materialized = worktree / relative
        expected = overrides.get(relative, source_hash)
        actual = _sha256_file(materialized) if materialized.is_file() else "missing"
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    if mismatches:
        raise ConsumerFastPreflightError(f"BASE_SOURCES divergente: {mismatches}")

    module_names = (
        "consumer_fast_core",
        "consumer_operational_recovery",
        "consumer_operational_publish",
        "consumer_fast_project",
        "integration_v1.contracts",
        "integration_v1.providers",
        "vision_runtime.analysis_payload",
        "typesetter.renderer",
        "consumer_fast.provider_adapter",
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
        "chapter_runner": _sha256_file(Path(__file__).resolve()),
        "provider_adapter": next(
            row["sha256"] for row in resolved_modules if row["name"] == "consumer_fast.provider_adapter"
        ),
    }
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
    if not legitimately_approved:
        gate["status"] = "BLOCK"
        gate["allowed"] = False
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
    """Run the declared physical V1 stage in-process; never select a fallback executable."""
    import main as physical_runtime

    physical_runtime._run_pipeline(str(config_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return Path(config["work_dir"]) / "project.json"


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
