"""Persist one Vision analysis record from an authenticated owner page result."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_page_analysis_record(
    output_dir: Path,
    root: Path,
    page: Any,
    config: dict[str, Any],
    *,
    source_sha256: str,
    project_id: str,
    project_revision: int = 1,
    capability_version: str = "vision-page-v1",
) -> dict[str, Any]:
    """Write page evidence and a non-publishable record bound to that evidence."""
    from integration_v1.contracts import AnalysisRecord
    from vision_runtime.analysis_payload import build_analysis_payload
    from vision_runtime.structure import build_structural_analysis

    page_result = getattr(page, "owner_page_result", None)
    if page_result is None:
        raise RuntimeError("Vision PageExecutionResult is missing; AnalysisRecord cannot be complete")
    coverage = getattr(page_result, "coverage", None)
    if coverage is None or not getattr(coverage, "canonical_json_bytes", None):
        raise RuntimeError("Vision PageCoverageResult canonical bytes are missing")
    ledger = getattr(coverage, "ledger", None)
    if ledger is None or not getattr(ledger, "sha256", None):
        raise RuntimeError("Vision coverage ledger identity is missing")
    if not _SHA256.fullmatch(source_sha256):
        raise ValueError("Vision source SHA-256 is missing")

    image = getattr(page, "original_image", None)
    height, width = (image.shape[:2] if image is not None else (1, 1))
    root.mkdir(parents=True, exist_ok=True)
    journal = page_result.to_canonical_dict()
    if str(journal.get("page_id")) != str(page_result.page_id):
        raise RuntimeError("Vision page journal identity differs from the page result")
    if str(journal.get("run_id")) != str(page_result.request.run_id) or str(journal.get("execution_id")) != str(page_result.request.execution_id):
        raise RuntimeError("Vision page journal execution identity differs from the page result")
    journal_original = journal.get("original_page")
    if isinstance(journal_original, dict) and journal_original.get("source_file_sha256") != source_sha256:
        raise RuntimeError("Vision source identity differs from the page journal")
    journal_path = root / "page_execution_evidence.json"
    _atomic_json(journal_path, journal)
    journal_sha = _sha256(journal_path)
    coverage_path = root / "coverage_result.json"
    coverage_path.write_bytes(coverage.canonical_json_bytes)
    coverage_sha = _sha256(coverage_path)
    observations = [
        {
            "observation_id": str(item.observation_id), "text": str(item.text),
            "bbox_page": list(item.bbox_page), "page_id": str(item.page_id),
            "provider": str(item.provider), "provider_family": str(item.provider_family),
            "run_id": str(item.run_id), "execution_id": str(item.origin_execution_id),
            "attempt_id": str(item.attempt_id), "invocation_id": str(item.invocation_id),
            "page_source_sha256": str(item.page_source_sha256),
            "payload_sha256": str(item.payload_sha256), "rejection_reason": item.rejection_reason,
        }
        for item in coverage.observations
    ]
    if any(item["page_id"] != str(page_result.page_id) for item in observations):
        raise RuntimeError("Vision OCR observation belongs to a different page")
    if any(item["page_source_sha256"] != str(journal.get("page_source_sha256")) for item in observations):
        raise RuntimeError("Vision OCR observation belongs to a different source image")
    observations_path = root / "ocr_observations.json"
    _atomic_json(observations_path, observations)
    observations_sha = _sha256(observations_path)
    owner_graph_snapshot = getattr(page_result, "owner_graph", None)
    owner_graph = owner_graph_snapshot.read() if callable(getattr(owner_graph_snapshot, "read", None)) else None
    selected_observation_ids = sorted({
        str(observation_id)
        for owner in (getattr(owner_graph, "owners", ()) or ())
        for observation_id in (getattr(owner, "selected_observation_ids", ()) or ())
        if str(observation_id)
    })
    if not set(selected_observation_ids).issubset({item["observation_id"] for item in observations}):
        raise RuntimeError("Vision owner selection references an observation outside the same page evidence")
    structural = build_structural_analysis(
        observations=[
            {
                "observation_id": item["observation_id"], "text": item["text"],
                "bbox_page": item["bbox_page"],
                "selection_state": "eligible" if item["observation_id"] in selected_observation_ids else "unresolved",
                "uncertainty_reasons": [] if item["observation_id"] in selected_observation_ids else ["owner_selection_not_authorized_by_analysis_adapter"],
            }
            for item in observations
        ],
        containers=[],
    )
    structural_payload = {
        "schema": "traduzai.vision-structural-analysis.v1",
        "logical_units": list(structural.logical_units),
        "physical_subblocks": list(structural.physical_subblocks),
        "relations": list(structural.relations),
        "reading_order": list(structural.reading_order),
        "source_observation_sha256": observations_sha,
    }
    structure_path = root / "structural_analysis.json"
    _atomic_json(structure_path, structural_payload)
    structure_sha = _sha256(structure_path)
    artifact_refs = [
        {"kind": kind, "path": path.relative_to(output_dir).as_posix(), "sha256": _sha256(path)}
        for kind, path in (
            ("vision_page_execution_evidence", journal_path),
            ("vision_coverage_result", coverage_path),
            ("vision_ocr_observations", observations_path),
            ("vision_structural_analysis", structure_path),
        )
    ]
    from vision_runtime.cache_key import visual_config_sha256

    config_sha = visual_config_sha256(config)
    transform_sha = hashlib.sha256(b"identity-transform-v1").hexdigest()
    relative_observations = observations_path.relative_to(output_dir).as_posix()
    relative_structure = structure_path.relative_to(output_dir).as_posix()
    payload = build_analysis_payload(
        status="complete",
        identity={
            "source_sha256": source_sha256,
            "authenticated_neighbor_sha256s": [],
            "region": {"bbox": [0, 0, int(width), int(height)], "coordinate_space": "logical_page"},
            "coordinate_space": "logical_page", "transform_sha256": transform_sha,
            "source_language": str(config.get("idioma_origem", "en")),
            "analysis_config_sha256": config_sha,
            "provider_family": "vision_v6", "provider_name": "vision_stack.runtime",
            "provider_model": str(config.get("engine_preset_id") or "max"),
            "provider_version": str(getattr(coverage.observations[0], "provider_family", "vision-v6") if coverage.observations else "vision-v6"),
            "capability_version": capability_version,
        },
        references={
            "artifact_refs": artifact_refs,
            "transform_ref": {"kind": "identity", "sha256": transform_sha, "inverse_sha256": transform_sha},
            "ocr_observations": {"artifact_ref": relative_observations, "sha256": observations_sha, "count": len(observations)},
            "selected_observation_id": selected_observation_ids[0] if len(selected_observation_ids) == 1 else None,
            "selection_provenance": {"kind": "vision_page_execution_and_owner_graph", "page_evidence_sha256": journal_sha, "coverage_sha256": coverage_sha, "coverage_ledger_sha256": ledger.sha256, "owner_graph_sha256": str(getattr(owner_graph_snapshot, "sha256", journal.get("owner_graph_sha256") or "")), "selected_observation_ids": selected_observation_ids, "selection_status": "selected" if selected_observation_ids else "unresolved"},
            "logical_units": {"artifact_ref": relative_structure, "sha256": structure_sha, "count": len(structural.logical_units)},
            "physical_subblocks": {"artifact_ref": relative_structure, "sha256": structure_sha, "count": len(structural.physical_subblocks)},
            "relations": {"artifact_ref": relative_structure, "sha256": structure_sha, "count": len(structural.relations)},
            "reading_order": {"artifact_ref": relative_structure, "sha256": structure_sha, "count": len(structural.reading_order)},
            "container_contour_ref": None, "writing_body_ref": None, "tail_ref": None,
            "dependency_hashes": {"source": source_sha256, "analysis_config": config_sha, "ocr_observations": observations_sha, "vision_page_evidence": journal_sha, "coverage_result": coverage_sha, "coverage_ledger": ledger.sha256, "structural_analysis": structure_sha},
        },
    )
    payload.update({
        "project_id": project_id, "project_revision": project_revision,
        "run_id": str(page_result.request.run_id),
        "execution_id": str(page_result.request.execution_id),
        "page_id": str(page_result.page_id),
        "coverage_ledger_sha256": ledger.sha256,
        "publishable": False,
    })
    record = AnalysisRecord.build(payload).to_dict()
    path = root / "analysis_record.json"
    _atomic_json(path, record)
    return {
        "status": "complete", "publishable": False, "path": str(path), "sha256": _sha256(path),
        "project_id": project_id, "project_revision": project_revision,
        "run_id": record["run_id"], "execution_id": record["execution_id"],
        "page_id": record["page_id"], "coverage_ledger_sha256": ledger.sha256,
        "analysis_record_sha256": record["analysis_record_sha256"],
    }


def write_chapter_analysis_records(
    work_dir: Path,
    pages: list[Any],
    config: dict[str, Any],
    *,
    source_manifest_sha256: str,
    source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Persist one record per verified owner page and a lookup index."""
    if not pages:
        raise ValueError("Vision chapter has no output pages")
    if not _SHA256.fullmatch(source_manifest_sha256):
        raise ValueError("Vision chapter source manifest SHA-256 is missing")
    if source_tree_sha256 is not None and not _SHA256.fullmatch(source_tree_sha256):
        raise ValueError("Vision chapter source tree SHA-256 is invalid")
    root = work_dir / "vision"
    project_id = f"owner-source-{(source_tree_sha256 or source_manifest_sha256)[:24]}"
    seen: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for page in pages:
        result = getattr(page, "owner_page_result", None)
        if result is None:
            raise RuntimeError("Vision owner page result is missing")
        page_id = str(result.page_id)
        if page_id in seen or not re.fullmatch(r"[A-Za-z0-9_-]+", page_id):
            raise ValueError("Vision page identity is duplicate or unsafe")
        seen.add(page_id)
        original_page = getattr(result.request, "original_page", None)
        source_sha = str(getattr(original_page, "source_file_sha256", ""))
        receipts.append(write_page_analysis_record(
            work_dir,
            root / "pages" / page_id,
            page,
            config,
            source_sha256=source_sha,
            project_id=project_id,
        ))
    index = {
        "schema": "traduzai.vision-chapter-analysis-index.v1",
        "source_manifest_sha256": source_manifest_sha256,
        "source_tree_sha256": source_tree_sha256,
        "project_id": project_id,
        "publishable": False,
        "pages": [],
    }
    for receipt in receipts:
        page_id = receipt["page_id"]
        entry = {
            "page_id": page_id,
            "analysis_record_ref": Path(receipt["path"]).relative_to(work_dir).as_posix(),
            "analysis_record_file_sha256": receipt["sha256"],
            "analysis_record_sha256": receipt["analysis_record_sha256"],
            "coverage_ledger_sha256": receipt["coverage_ledger_sha256"],
        }
        for suffix in ("json", "npz"):
            path = root / "pre_ocr" / f"{page_id}.{suffix}"
            if path.is_file():
                entry[f"pre_ocr_{suffix}_ref"] = path.relative_to(work_dir).as_posix()
                entry[f"pre_ocr_{suffix}_sha256"] = _sha256(path)
        index["pages"].append(entry)
    index_path = root / "index.json"
    _atomic_json(index_path, index)
    return {"path": str(index_path), "sha256": _sha256(index_path), "page_count": len(receipts)}
