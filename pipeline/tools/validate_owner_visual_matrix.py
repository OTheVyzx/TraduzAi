"""Run and validate a fresh cross-work owner visual matrix.

This harness is intentionally outside the production pipeline.  It groups
failures by invariant/contract, never by title, chapter, page, or band.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_style_owner_target_manifest import (
    OwnerTargetError,
    build_effective_style_config,
    selected_owner_target,
    source_crop_contract,
)


REQUIRED_VISUAL_CATEGORIES = frozenset(
    {
        "white_balloon",
        "translucent_balloon",
        "burst",
        "dark_panel",
        "colored_card",
        "text_over_art",
        "table_ranking",
        "cross_tile_owner",
        "primary_ocr_omission",
    }
)
REQUIRED_HOLDOUT_CATEGORIES = (
    "white_balloon",
    "burst",
    "cross_tile_owner",
    "dark_panel",
    "text_over_art",
)

REQUIRED_FINAL_PIXEL_CONTRACTS = frozenset(
    {
        "source_coverage_contract",
        "owner_graph_contract",
        "route_state_contract",
        "pixel_ownership_contract",
        "final_language_contract",
        "layout_legibility_contract",
        "residual_cleanup_contract",
        "protected_art_contract",
        "qa_integrity_contract",
    }
)

ACCEPTANCE_METRIC_FIELDS = (
    "english_dialogue_residual_count",
    "translatable_components_without_owner",
    "material_components_without_ocr_attempt",
    "owners_without_valid_pt_br",
    "owners_without_atomic_cleanup_render",
    "owners_without_target_materialization",
    "material_components_without_terminal_lifecycle",
)


class MatrixContractError(ValueError):
    """Raised when the validation corpus itself is not systemic or fresh."""


class InspectionEvidenceError(MatrixContractError):
    """Raised when a visual verdict is not bound to a real native-pixel review."""


def _canonical_entry(entry: Any, index: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise MatrixContractError(f"matrix entry {index} must be an object")
    required = (
        "entry_id",
        "work_id",
        "chapter_id",
        "split",
        "config_path",
        "config_sha256",
        "input_key",
        "expected_input_sha256",
        "work_dir",
    )
    missing = [key for key in required if not str(entry.get(key) or "").strip()]
    if missing:
        raise MatrixContractError(
            f"matrix entry {index} is missing fields: {', '.join(missing)}"
        )
    targets = entry.get("targets")
    generic_categories = entry.get("categories")
    generic_pages = entry.get("category_pages")
    generic_entry = (
        targets is None
        and isinstance(generic_categories, list)
        and bool(generic_categories)
        and isinstance(generic_pages, dict)
        and all(
            isinstance(generic_pages.get(str(category)), list)
            and bool(generic_pages.get(str(category)))
            for category in generic_categories
        )
    )
    if generic_entry:
        normalized = dict(entry)
        normalized["categories"] = sorted({str(value) for value in generic_categories})
        normalized["category_pages"] = {
            str(category): sorted({int(page) for page in pages})
            for category, pages in generic_pages.items()
        }
        normalized["targets"] = []
        normalized["generic_functional_entry"] = True
        if normalized["split"] not in {"calibration", "holdout"}:
            raise MatrixContractError(f"matrix entry {index} has invalid split")
        return normalized
    if not isinstance(targets, list) or not targets:
        raise MatrixContractError(f"matrix entry {index} requires exact owner targets")
    normalized_targets: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise MatrixContractError(f"matrix entry {index} target {target_index} must be an object")
        target_required = ("page_id", "owner_id", "category")
        missing_target = [key for key in target_required if not str(target.get(key) or "").strip()]
        component_ids = target.get("component_ids")
        source_crop = target.get("source_crop")
        crop_valid = (
            isinstance(source_crop, dict)
            and isinstance(source_crop.get("bbox_page"), list)
            and len(source_crop["bbox_page"]) == 4
            and int(source_crop.get("width") or 0) > 0
            and int(source_crop.get("height") or 0) > 0
            and len(str(source_crop.get("sha256") or "")) == 64
        )
        if str(source_crop.get("coordinate_space") or ""):
            crop_valid = crop_valid and source_crop.get("coordinate_space") == "logical_page"
            crop_valid = crop_valid and target.get("expected_artifact_space") == "framed_page"
            crop_valid = crop_valid and bool(str(target.get("semantic_role") or "").strip())
            crop_valid = crop_valid and isinstance(target.get("is_speech"), bool)
        if missing_target or not isinstance(component_ids, list) or not component_ids or not crop_valid:
            raise MatrixContractError(f"matrix entry {index} target {target_index} is incomplete")
        normalized_target = dict(target)
        if str(target.get("split") or "") != str(entry.get("split") or ""):
            raise MatrixContractError(f"matrix entry {index} target {target_index} split mismatch")
        normalized_target["component_ids"] = sorted(set(str(value) for value in component_ids))
        normalized_targets.append(normalized_target)
    categories = sorted({str(target["category"]) for target in normalized_targets})
    normalized = dict(entry)
    normalized["categories"] = categories
    normalized["targets"] = sorted(
        normalized_targets,
        key=lambda row: (str(row["category"]), str(row["page_id"]), str(row["owner_id"])),
    )
    if normalized["split"] not in {"calibration", "holdout"}:
        raise MatrixContractError(f"matrix entry {index} has invalid split")
    return normalized


def validate_manifest(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
        raise MatrixContractError("matrix manifest must contain entries[]")
    has_logical_targets = any(
        str(((target.get("source_crop") or {}).get("coordinate_space") or ""))
        == "logical_page"
        for entry in manifest.get("entries") or []
        if isinstance(entry, dict)
        for target in entry.get("targets") or []
        if isinstance(target, dict)
    )
    if has_logical_targets and manifest.get("schema_version") != 3:
        raise MatrixContractError("matrix schema v3 is required for logical page targets")
    entries = [_canonical_entry(entry, index) for index, entry in enumerate(manifest["entries"])]
    works = {entry["work_id"] for entry in entries}
    if len(works) < 3:
        raise MatrixContractError("matrix requires at least three distinct works")
    if {entry["split"] for entry in entries} != {"calibration", "holdout"}:
        raise MatrixContractError("matrix requires calibration and holdout entries")
    provenance = [(entry["work_id"], entry["chapter_id"]) for entry in entries]
    if len(provenance) != len(set(provenance)):
        raise MatrixContractError("work/chapter provenance cannot cross matrix splits")
    output_dirs = [Path(entry["work_dir"]).as_posix().casefold() for entry in entries]
    if len(output_dirs) != len(set(output_dirs)):
        raise MatrixContractError("every matrix entry requires a unique work_dir")
    categories = {
        category for entry in entries for category in entry.get("categories", [])
    }
    missing = sorted(REQUIRED_VISUAL_CATEGORIES - categories)
    if missing:
        raise MatrixContractError(
            "matrix is missing visual categories: " + ", ".join(missing)
        )
    return entries


def _page_surface_geometry(page: dict[str, Any]):
    from strip.page_surface_geometry import PageSurfaceGeometry

    payload = page.get("page_surface_geometry")
    if not isinstance(payload, dict):
        raise MatrixContractError("missing_page_surface_geometry")
    try:
        geometry = PageSurfaceGeometry.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise MatrixContractError("invalid_page_surface_geometry") from exc
    if str(page.get("page_surface_geometry_sha256") or "") != geometry.geometry_sha256:
        raise MatrixContractError("page_surface_geometry_hash_mismatch")
    return geometry


def _project_page_by_id(project: dict[str, Any], page_id: str) -> dict[str, Any]:
    for index, page in enumerate(project.get("paginas") or [], start=1):
        if not isinstance(page, dict):
            continue
        candidate = str(page.get("page_id") or f"page_{int(page.get('numero') or index):03d}")
        if candidate == page_id:
            return page
    raise MatrixContractError(f"page_target_not_found:{page_id}")


def _target_artifact_bbox_frame(
    target: dict[str, Any], page: dict[str, Any]
) -> tuple[list[int], Any | None]:
    source_crop = target.get("source_crop") or {}
    bbox = [int(value) for value in source_crop.get("bbox_page") or []]
    if len(bbox) != 4:
        raise MatrixContractError("source_crop_bbox_invalid")
    if str(source_crop.get("coordinate_space") or "") == "logical_page":
        geometry = _page_surface_geometry(page)
        if str(target.get("expected_artifact_space") or "") != "framed_page":
            raise MatrixContractError("expected_artifact_space_invalid")
        return list(geometry.logical_bbox_to_frame(tuple(bbox))), geometry
    return bbox, None


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_payload_sha256(payload: Any) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256_input(path: Path) -> str:
    """Hash a file or a directory tree with stable relative-path ordering."""
    path = path.resolve()
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise MatrixContractError(f"input missing: {path}")
    digest = sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(item)))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def resolve_entry_runtime(
    entry: dict[str, Any],
    manifest_path: Path,
    inputs_manifest: dict[str, Any],
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve only versioned configs and hash-pinned external inputs."""
    fixture_root = manifest_path.resolve().parent
    fixtures_root = fixture_root.parent
    raw_config = Path(str(entry.get("config_path") or ""))
    if raw_config.is_absolute():
        raise MatrixContractError("config_path must be relative to the matrix fixture")
    config_path = (fixture_root / raw_config).resolve()
    try:
        config_path.relative_to(fixtures_root)
    except ValueError as exc:
        raise MatrixContractError("config_path escapes the versioned fixtures root") from exc
    if ".codex-tmp" in {part.casefold() for part in config_path.parts}:
        raise MatrixContractError("config_path cannot depend on .codex-tmp")
    if not config_path.is_file():
        raise MatrixContractError(f"versioned config missing: {config_path}")
    expected_config_hash = str(entry.get("config_sha256") or "").lower()
    if _canonical_json_sha256(config_path) != expected_config_hash:
        raise MatrixContractError(f"config hash mismatch: {entry.get('entry_id', 'entry')}")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if str(config.get("input_key") or "") != str(entry.get("input_key") or ""):
        raise MatrixContractError(f"config input_key mismatch: {entry.get('entry_id', 'entry')}")

    rows = inputs_manifest.get("inputs") if isinstance(inputs_manifest, dict) else None
    input_key = str(entry.get("input_key") or "").strip()
    row = rows.get(input_key) if isinstance(rows, dict) else None
    if not isinstance(row, dict):
        raise MatrixContractError(f"input_key missing from inputs manifest: {input_key}")
    expected_hash = str(row.get("expected_sha256") or "").strip().lower()
    if expected_hash != str(entry.get("expected_input_sha256") or "").strip().lower():
        raise MatrixContractError(f"input hash contract mismatch: {input_key}")
    env_name = str(row.get("environment_variable") or "").strip()
    raw_source = (environ or os.environ).get(env_name, "")
    if not raw_source:
        raise MatrixContractError(f"input environment variable missing: {env_name}")
    source_path = Path(raw_source).expanduser().resolve()
    if not source_path.exists():
        raise MatrixContractError(f"input missing: {source_path}")
    input_type = str(row.get("input_type") or "")
    if input_type == "file" and not source_path.is_file():
        raise MatrixContractError(f"input type mismatch for {input_key}: expected file")
    if input_type == "directory" and not source_path.is_dir():
        raise MatrixContractError(f"input type mismatch for {input_key}: expected directory")
    actual_hash = _sha256_input(source_path)
    if actual_hash != expected_hash:
        raise MatrixContractError(f"input hash mismatch: {input_key}")
    return {
        "config_path": config_path,
        "source_path": source_path,
        "input_sha256": actual_hash,
        "input_key": input_key,
    }


def _contract_name(issue: Any) -> str:
    if not isinstance(issue, dict):
        return "malformed_gate_issue"
    for key in ("contract", "code", "reason", "flag", "issue_id"):
        value = str(issue.get(key) or "").strip()
        if value:
            return value
    return "unclassified_gate_issue"


def validate_entry_result(entry: dict[str, Any], output_root: Path) -> dict[str, Any]:
    work_dir = Path(entry["work_dir"])
    if not work_dir.is_absolute():
        work_dir = output_root / work_dir
    project_path = work_dir / "project.json"
    contracts: set[str] = set()
    details: list[str] = []
    style_fallback_owner_ids: set[str] = set()
    for blocker in entry.get("_preflight_contracts") or []:
        contracts.add(str(blocker))
    details.extend(str(value) for value in entry.get("_preflight_details") or [])
    if not project_path.is_file():
        contracts.add("project_result_missing")
        return {
            "entry_id": entry["entry_id"],
            "work_dir": str(work_dir.resolve()),
            "status": "BLOCK",
            "contracts": sorted(contracts),
            "details": details,
        }
    try:
        project = json.loads(project_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        contracts.add("project_result_invalid")
        details.append(str(exc))
        project = {}
    if project.get("owner_graph_status") != "verified":
        contracts.add("owner_graph_unverified")
    category_metrics: dict[str, dict[str, int]] = {
        str(category): {"owner_count": 0, "minimum_count": 0}
        for category in entry.get("categories") or []
    }
    for target in entry.get("targets") or []:
        category = str(target.get("category") or "")
        metrics = category_metrics.setdefault(category, {"owner_count": 0, "minimum_count": 0})
        metrics["minimum_count"] = max(metrics["minimum_count"], int(target.get("minimum_count") or 1))
        try:
            resolved_target = selected_owner_target(target, project)
        except OwnerTargetError as exc:
            detail = str(exc)
            if "page target not found" in detail:
                contracts.add("page_target_not_found")
            elif "component target mismatch" in detail:
                contracts.add("owner_component_target_mismatch")
            else:
                contracts.add("owner_target_not_found")
            details.append(detail)
        else:
            metrics["owner_count"] += 1
            base_path = ((resolved_target["page"].get("image_layers") or {}).get("base") or {}).get("path")
            try:
                artifact_bbox, geometry = _target_artifact_bbox_frame(
                    target, resolved_target["page"]
                )
                actual_crop = source_crop_contract(
                    (work_dir / str(base_path)).resolve(),
                    list(target["source_crop"]["bbox_page"]),
                    artifact_bbox_frame=(artifact_bbox if geometry is not None else None),
                    coordinate_space=("logical_page" if geometry is not None else None),
                )
            except MatrixContractError as exc:
                contracts.add(str(exc))
            except (OSError, ValueError, TypeError):
                contracts.add("source_crop_missing_or_invalid")
            else:
                expected_crop = dict(target["source_crop"])
                actual_crop.pop("artifact_bbox_frame", None)
                if actual_crop != expected_crop:
                    contracts.add("source_crop_hash_or_dimension_mismatch")
    for category, metrics in category_metrics.items():
        if metrics["owner_count"] < metrics["minimum_count"]:
            contracts.add(f"style_category_zero_or_below_minimum:{category}")
    expected_pages: set[str] = set()
    for page_index, page in enumerate(project.get("paginas") or [], start=1):
        if not isinstance(page, dict):
            continue
        page_id = str(page.get("page_id") or f"page_{int(page.get('numero') or page_index):03d}")
        expected_pages.add(page_id)
        for layer in page.get("text_layers") or page.get("textos") or []:
            if not isinstance(layer, dict):
                continue
            if not str(layer.get("owner_id") or "").strip() or layer.get("render_completed") is not True:
                continue
            owner_id = str(layer.get("owner_id") or "").strip()
            profile = layer.get("visual_profile_v2")
            raster = layer.get("style_v2_raster_contract")
            if isinstance(profile, dict) or isinstance(raster, dict):
                if not isinstance(profile, dict) or str(profile.get("owner_id") or "") != owner_id:
                    contracts.add("style_owner_pair_mismatch")
                elif not str(profile.get("source_sha256") or "").strip():
                    contracts.add("style_source_hash_missing")
                if not isinstance(raster, dict):
                    contracts.add("style_final_raster_contract_missing")
                if str((profile or {}).get("status") or "") == "fallback" or str((raster or {}).get("status") or "") == "fallback":
                    style_fallback_owner_ids.add(owner_id)
            quality = layer.get("owner_render_quality")
            if not isinstance(quality, dict):
                layout = layer.get("render_layout_contract")
                quality = layout.get("owner_render_quality") if isinstance(layout, dict) else None
            if not isinstance(quality, dict):
                contracts.add("missing_owner_render_quality_contract")
            else:
                quality_status = str(quality.get("status") or "").strip()
                if quality_status != "ok":
                    contracts.add(quality_status or "invalid_owner_render_quality_contract")
                try:
                    source_ratio = float(quality.get("source_scale_ratio"))
                except (TypeError, ValueError):
                    source_ratio = None
                if source_ratio is not None and source_ratio < 0.75:
                    contracts.add("under_source_scale")
                if int(quality.get("outside_safe_pixels", 0) or 0) > 0:
                    contracts.add("core_pixels_outside_safe_polygon")
                if not quality.get("rendered_line_core_heights_px"):
                    contracts.add("missing_rendered_line_core_metrics")
            if str(layer.get("route_action") or "") in {
                "translate_inpaint_render",
                "translate_sfx_inpaint_render",
            }:
                residual = layer.get("residual_cleanup_contract")
                if not isinstance(residual, dict) or residual.get("residual_verified") is not True:
                    contracts.add("unverified_owner_residual")
                else:
                    try:
                        if float(residual.get("residual_score")) > float(residual.get("residual_threshold")):
                            contracts.add("owner_residual_above_threshold")
                    except (TypeError, ValueError):
                        contracts.add("invalid_owner_residual_contract")
                protected = layer.get("protected_art_contract")
                if not isinstance(protected, dict):
                    contracts.add("missing_protected_art_contract")
                elif (
                    int(protected.get("protected_art_changed_pixels", 0) or 0) > 0
                    or int(protected.get("action_protected_overlap_pixels", 0) or 0) > 0
                ):
                    contracts.add("protected_art_contract_violation")
    qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
    gate = qa.get("export_gate") if isinstance(qa.get("export_gate"), dict) else {}
    if gate.get("status") != "PASS":
        contracts.add("export_gate_blocked")
    for issue in gate.get("issues") or []:
        if isinstance(issue, dict) and (
            str(issue.get("severity") or "").lower() in {"critical", "blocker"}
            or bool(issue.get("blocking"))
            or bool(issue.get("blocks_export"))
        ):
            contracts.add(_contract_name(issue))
    style_fidelity = qa.get("style_fidelity") if isinstance(qa.get("style_fidelity"), dict) else None
    style_status = "PASS" if not entry.get("targets") else "BLOCK"
    if style_fidelity is None:
        if entry.get("targets"):
            contracts.add("style_fidelity_report_missing")
    else:
        style_gate = style_fidelity.get("gate") if isinstance(style_fidelity.get("gate"), dict) else {}
        style_status = str(style_gate.get("status") or "BLOCK").upper()
        if str(style_gate.get("status") or "MISSING").upper() == "BLOCK":
            contracts.add("style_fidelity_gate_blocked")
        audited_owner_ids = {
            str(row.get("owner_id") or "")
            for row in style_fidelity.get("owners") or []
            if isinstance(row, dict) and str(row.get("owner_id") or "")
        }
        rendered_profile_ids = {
            str(layer.get("owner_id") or "")
            for page in project.get("paginas") or [] if isinstance(page, dict)
            for layer in page.get("text_layers") or [] if isinstance(layer, dict)
            if layer.get("render_completed") is True and isinstance(layer.get("visual_profile_v2"), dict)
        }
        if audited_owner_ids and not rendered_profile_ids.issubset(audited_owner_ids):
            contracts.add("style_fidelity_owner_coverage_incomplete")
        for category, metrics in (style_fidelity.get("category_metrics") or {}).items():
            if not isinstance(metrics, dict):
                continue
            try:
                go_rate = float(metrics.get("go_rate"))
            except (TypeError, ValueError):
                continue
            threshold = 0.95 if str(category) in {"speech", "white_balloon"} else 0.85
            if go_rate < threshold:
                contracts.add(f"style_category_below_threshold:{category}")
    reports = qa.get("final_pixel_reports")
    if not isinstance(reports, list) or not reports:
        contracts.add("final_pixel_report_missing")
        reports = []
    seen_pages: set[str] = set()
    for report in reports:
        if not isinstance(report, dict):
            contracts.add("final_pixel_report_malformed")
            continue
        page_id = str(report.get("page_id") or "").strip()
        if not page_id or page_id in seen_pages:
            contracts.add("final_pixel_page_coverage_invalid")
        seen_pages.add(page_id)
        if report.get("observer_available") is not True:
            contracts.add("final_pixel_observer_unavailable")
        if report.get("observation_complete") is not True:
            contracts.add("final_pixel_observation_incomplete")
        if report.get("coverage_complete") is not True:
            failures = [str(value) for value in report.get("coverage_failures") or [] if str(value)]
            contracts.update(failures or ["final_pixel_coverage_incomplete"])
        try:
            expected_challenges = int(report.get("expected_source_challenge_count", 0) or 0)
            completed_challenges = int(report.get("completed_source_challenge_count", 0) or 0)
        except (TypeError, ValueError):
            contracts.add("source_challenge_counts_invalid")
        else:
            if completed_challenges < expected_challenges:
                contracts.add("source_challenge_inconclusive")
        report_contracts = report.get("contracts")
        if not isinstance(report_contracts, dict) or not REQUIRED_FINAL_PIXEL_CONTRACTS.issubset(report_contracts):
            contracts.add("final_pixel_contracts_incomplete")
        else:
            for contract_name in REQUIRED_FINAL_PIXEL_CONTRACTS:
                status = str(report_contracts.get(contract_name) or "").upper()
                if status != "PASS":
                    contracts.add(contract_name if status == "BLOCK" else "final_pixel_contract_status_invalid")
        raw_artifact = str(report.get("artifact_path") or "").strip()
        artifact = Path(raw_artifact) if raw_artifact else Path()
        if raw_artifact and not artifact.is_absolute():
            artifact = work_dir / artifact
        expected_hash = str(report.get("persisted_sha256") or "").strip().lower()
        if not raw_artifact or not artifact.is_file() or len(expected_hash) != 64:
            contracts.add("final_artifact_missing_or_unhashed")
        elif _sha256_file(artifact) != expected_hash:
            contracts.add("final_artifact_hash_mismatch")
        for issue in report.get("issues") or []:
            contracts.add(_contract_name(issue))
    if expected_pages and seen_pages != expected_pages:
        contracts.add("final_pixel_page_coverage_invalid")
    functional_gate = qa.get("functional_export_gate") if isinstance(qa.get("functional_export_gate"), dict) else gate
    functional_status = str(functional_gate.get("status") or "BLOCK").upper()
    return {
        "entry_id": entry["entry_id"],
        "work_id": entry.get("work_id"),
        "chapter_id": entry.get("chapter_id"),
        "categories": list(entry.get("categories") or []),
        "work_dir": str(work_dir.resolve()),
        "project_path": str(project_path.resolve()),
        "status": "PASS" if not contracts else "BLOCK",
        "functional_status": "PASS" if functional_status in {"PASS", "REVIEW"} else "BLOCK",
        "style_status": "PASS" if style_status == "PASS" else "BLOCK",
        "contracts": sorted(contracts),
        "details": details,
        "page_count": len(seen_pages),
        "export_gate": gate.get("status") or "MISSING",
        "style_fallback_owner_ids": sorted(style_fallback_owner_ids),
        "category_metrics": category_metrics,
    }


def group_failures_by_contract(results: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for result in results:
        entry_id = str(result.get("entry_id") or "unknown_entry")
        for contract in sorted(set(result.get("contracts") or [])):
            grouped.setdefault(str(contract), []).append(entry_id)
    return {
        contract: sorted(set(entry_ids))
        for contract, entry_ids in sorted(grouped.items())
    }


def _resolve_from_manifest(manifest_path: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _persist_runner_logs(
    output_root: Path,
    entry_id: str,
    *,
    stdout: str,
    stderr: str,
) -> dict[str, str]:
    log_root = output_root / "runner_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{entry_id}.stdout.log"
    stderr_path = log_root / f"{entry_id}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
    }


def _canonical_payload_sha256(payload: Any, *, omit: str | None = None) -> str:
    canonical = {
        str(key): value for key, value in dict(payload).items() if key != omit
    } if isinstance(payload, dict) else payload
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_runner_evidence(
    *,
    entry: dict[str, Any],
    command: list[str],
    completed: Any,
    runtime: dict[str, Any],
    effective_config_path: Path,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    """Build a self-hashed provenance record for fresh-run and validate-only parity."""

    pipeline_root = Path(__file__).resolve().parents[2]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=pipeline_root, capture_output=True, text=True, check=False
    ).stdout.strip()
    scoped_diff = subprocess.run(
        ["git", "diff", "--binary", "--", "pipeline"],
        cwd=pipeline_root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.encode("utf-8")
    stdout = str(getattr(completed, "stdout", "") or "")
    stderr = str(getattr(completed, "stderr", "") or "")
    target = Path(entry["work_dir"])
    project_path = target / "project.json"
    final_hashes = {
        path.relative_to(target).as_posix(): _sha256_file(path)
        for path in sorted((target / "translated").glob("*"))
        if path.is_file()
    } if target.is_absolute() else {}
    evidence = {
        "schema_version": 1,
        "entry_id": entry["entry_id"],
        "git_head": git_head,
        "scoped_diff_sha256": sha256(scoped_diff).hexdigest(),
        "matrix_entry_sha256": _canonical_payload_sha256(entry),
        "input_sha256": runtime.get("input_sha256"),
        "config_sha256": _canonical_json_sha256(Path(runtime["config_path"])),
        "effective_config_sha256": _canonical_json_sha256(effective_config_path),
        "tool_sha256": _sha256_file(Path(__file__)),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "seed": int(entry.get("seed") or 0),
        "command": list(command),
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": int(getattr(completed, "returncode", 2)),
        "stdout_sha256": sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": sha256(stderr.encode("utf-8")).hexdigest(),
        "project_sha256": _sha256_file(project_path) if project_path.is_file() else None,
        "final_artifact_sha256": final_hashes,
    }
    evidence["runner_evidence_sha256"] = _canonical_payload_sha256(evidence, omit="runner_evidence_sha256")
    return evidence


def validate_runner_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Reject any changed run-manifest field before validate-only reuse."""

    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
        raise MatrixContractError("runner manifest schema is invalid")
    expected = str(evidence.get("runner_evidence_sha256") or "")
    actual = _canonical_payload_sha256(evidence, omit="runner_evidence_sha256")
    if expected != actual:
        raise MatrixContractError("runner manifest hash mismatch")
    return dict(evidence)


def _validate_external_audit_payload(payload: Any, entry_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise MatrixContractError(f"external audit schema invalid: {entry_id}")
    if str(payload.get("external_audit_gate_status") or "") != "PASS":
        raise MatrixContractError(f"external audit did not pass: {entry_id}")
    if any(int(payload.get(field) or 0) != 0 for field in ACCEPTANCE_METRIC_FIELDS):
        raise MatrixContractError(f"external audit acceptance metrics nonzero: {entry_id}")
    page_ids = [str(value) for value in payload.get("source_manifest_page_ids") or ()]
    audits = [row for row in payload.get("external_page_audits") or () if isinstance(row, dict)]
    if [str(row.get("page_id") or "") for row in audits] != page_ids:
        raise MatrixContractError(f"external page audit cardinality mismatch: {entry_id}")
    if not audits or not all(
        row.get("status") == "PASS"
        and str(row.get("auditor_invocation_id") or "")
        and str(row.get("auditor_execution_id") or "")
        and str(row.get("auditor_process_nonce") or "")
        for row in audits
    ):
        raise MatrixContractError(f"external page audit identity incomplete: {entry_id}")
    expected = str(payload.get("report_sha256") or "")
    if expected and expected != _canonical_payload_sha256(payload, omit="report_sha256"):
        raise MatrixContractError(f"external audit report hash mismatch: {entry_id}")
    return dict(payload)


def _run_external_audits(
    entries: list[dict[str, Any]],
    runtimes: dict[str, dict[str, Any]],
    output_root: Path,
    audit_dir: Path,
    *,
    external_auditor_command: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    auditor = Path(__file__).resolve().with_name("audit_owner_chapter_output.py")
    records: dict[str, dict[str, Any]] = {}
    for entry in entries:
        entry_id = str(entry["entry_id"])
        runtime = runtimes.get(entry_id)
        if runtime is None:
            raise MatrixContractError(f"external audit source runtime missing: {entry_id}")
        work_dir = Path(entry["work_dir"])
        if not work_dir.is_absolute():
            work_dir = output_root / work_dir
        report_path = audit_dir / f"{entry_id}.json"
        review_dir = audit_dir / f"{entry_id}-review"
        config = json.loads(Path(runtime["config_path"]).read_text(encoding="utf-8-sig"))
        command = [
            *(tuple(external_auditor_command) if external_auditor_command is not None else (sys.executable, str(auditor))),
            "--source", str(Path(runtime["source_path"]).resolve()),
            "--run", str(work_dir.resolve()),
            "--report", str(report_path.resolve()),
            "--review-dir", str(review_dir.resolve()),
            "--source-lang", str(config.get("idioma_origem") or "en"),
            "--target-lang", str(config.get("idioma_destino") or "pt-BR"),
            "--require-final-verified",
        ]
        child_env = dict(os.environ)
        child_env["TRADUZAI_MATRIX_ENTRY_ID"] = entry_id
        completed = subprocess.run(
            command,
            cwd=str(auditor.parents[1]),
            env=child_env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0 or not report_path.is_file():
            detail = completed.stderr.strip() or completed.stdout.strip() or "report missing"
            raise MatrixContractError(f"external audit process failed: {entry_id}: {detail}")
        records[entry_id] = _validate_external_audit_payload(
            json.loads(report_path.read_text(encoding="utf-8-sig")),
            entry_id,
        )
    return records


def _run_entry(
    entry: dict[str, Any],
    manifest_path: Path,
    output_root: Path,
    runtime: dict[str, Any],
    acceptance_bundle_path: Path | None = None,
) -> dict[str, Any]:
    target = Path(entry["work_dir"])
    if not target.is_absolute():
        target = output_root / target
    if target.exists():
        raise MatrixContractError(f"fresh matrix work_dir already exists: {target}")
    config_path = Path(runtime["config_path"])
    shared_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    config = build_effective_style_config(
        shared_config,
        required_categories=list(entry.get("categories") or []),
    )
    config.update(
        {
            "work_dir": str(target.resolve()),
            "source_path": str(Path(runtime["source_path"]).resolve()),
            "models_dir": str((Path(__file__).resolve().parents[1] / "models").resolve()),
            "owner_graph_mode": "enforce",
            "allow_p0_export_override": False,
        }
    )
    config_dir = output_root / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    effective_config = config_dir / f"{entry['entry_id']}.json"
    effective_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pipeline_main = Path(__file__).resolve().parents[1] / "main.py"
    command = [sys.executable, str(pipeline_main), str(effective_config)]
    started_at = datetime.now(timezone.utc).isoformat()
    child_env = dict(os.environ)
    if acceptance_bundle_path is not None:
        child_env["TRADUZAI_ACCEPTANCE_BUNDLE"] = str(acceptance_bundle_path.resolve())
    completed = subprocess.run(
        command,
        cwd=str(pipeline_main.parent),
        check=False,
        text=True,
        capture_output=True,
        env=child_env,
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    log_paths = _persist_runner_logs(
        output_root,
        entry["entry_id"],
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    evidence_entry = dict(entry)
    evidence_entry["work_dir"] = str(target.resolve())
    runner_evidence = build_runner_evidence(
        entry=evidence_entry,
        command=command,
        completed=completed,
        runtime=runtime,
        effective_config_path=effective_config,
        started_at=started_at,
        finished_at=finished_at,
    )
    runner_manifest_path = target / "run_manifest.json"
    runner_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    runner_manifest_path.write_text(
        json.dumps(runner_evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "entry_id": entry["entry_id"],
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        **log_paths,
        "runner_manifest_path": str(runner_manifest_path.resolve()),
        "runner_evidence": runner_evidence,
    }


def _selected_final_paths(
    entry: dict[str, Any],
    category: str,
    work_dir: Path,
) -> list[Path]:
    project_path = work_dir / "project.json"
    if not project_path.is_file():
        return []
    project = json.loads(project_path.read_text(encoding="utf-8-sig"))
    selected: list[Path] = []
    for target in entry.get("targets") or []:
        if str(target.get("category") or "") != category:
            continue
        resolved = selected_owner_target(target, project)
        rendered = ((resolved["page"].get("image_layers") or {}).get("rendered") or {}).get("path")
        if not isinstance(rendered, str) or not rendered.strip():
            raise MatrixContractError(f"final artifact missing for target: {target.get('page_id')}")
        candidate = (work_dir / rendered).resolve()
        if not candidate.is_file():
            raise MatrixContractError(f"final artifact missing for target: {candidate}")
        if candidate not in selected:
            selected.append(candidate)
    return selected


def _selected_target_finals(
    entry: dict[str, Any], category: str, work_dir: Path
) -> list[tuple[dict[str, Any], Path]]:
    project_path = work_dir / "project.json"
    if not project_path.is_file():
        return []
    project = json.loads(project_path.read_text(encoding="utf-8-sig"))
    selected: list[tuple[dict[str, Any], Path]] = []
    for target in entry.get("targets") or []:
        if str(target.get("category") or "") != category:
            continue
        resolved = selected_owner_target(target, project)
        rendered = ((resolved["page"].get("image_layers") or {}).get("rendered") or {}).get("path")
        if not isinstance(rendered, str) or not rendered.strip():
            raise MatrixContractError(f"final artifact missing for target: {target.get('page_id')}")
        final_path = (work_dir / rendered).resolve()
        if not final_path.is_file():
            raise MatrixContractError(f"final artifact missing for target: {final_path}")
        selected.append((target, final_path))
    return selected


def _selected_final_path(
    entry: dict[str, Any], category: str, work_dir: Path
) -> Path | None:
    selected = _selected_final_paths(entry, category, work_dir)
    return selected[0] if selected else None


def _difference_mask_panel(before, inpaint):
    from PIL import Image, ImageChops, ImageOps

    if before.size != inpaint.size:
        inpaint = inpaint.resize(before.size)
    difference = ImageChops.difference(before, inpaint).convert("L")
    binary = difference.point(lambda value: 255 if value > 12 else 0)
    return ImageOps.colorize(binary, black="#111820", white="#ff3b30").convert("RGB")


def _resolved_intent_panel(page, target, size, geometry):
    from PIL import Image, ImageDraw

    owner_id = str(target.get("owner_id") or "")
    layer = next(
        (
            row for row in page.get("text_layers") or page.get("textos") or []
            if isinstance(row, dict) and str(row.get("owner_id") or "") == owner_id
        ),
        None,
    )
    if layer is None:
        raise MatrixContractError("missing_owner_materialization_intent")
    # Prefer the committed render layout. A review owner has no committed layout,
    # but its independent geometry still has to remain inspectable so the matrix
    # can report the functional NO-GO instead of crashing while building sheets.
    render_plan = layer.get("render_layout_contract")
    plan = render_plan or layer.get("owner_render_geometry")
    if not isinstance(plan, dict):
        raise MatrixContractError("missing_owner_materialization_plan")
    bbox = None
    bbox_keys = (
        ("safe_text_box", "render_bbox", "target_bbox", "bbox")
        if isinstance(render_plan, dict)
        else (
            "layout_container_bbox_page",
            "semantic_body_bbox_page",
            "source_replacement_bbox_page",
        )
    )
    for key in bbox_keys:
        raw = plan.get(key) or layer.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            bbox = tuple(int(value) for value in raw)
            break
    if bbox is None:
        raise MatrixContractError("missing_owner_materialization_bbox")
    coordinate_space = str(
        plan.get("coordinate_space")
        or plan.get("logical_space")
        or ""
    )
    if coordinate_space == "logical_page":
        bbox = geometry.logical_bbox_to_frame(bbox)
    elif coordinate_space != "framed_page":
        raise MatrixContractError("owner_materialization_coordinate_space_invalid")
    panel = Image.new("RGB", size, "#111820")
    draw = ImageDraw.Draw(panel)
    draw.rectangle(bbox, outline="#34c759", width=3)
    payload = str(layer.get("translated") or layer.get("text") or "").strip()
    if payload:
        draw.multiline_text((bbox[0] + 4, bbox[1] + 4), payload, fill="white", spacing=3)
    return panel, {
        "plan_sha256": str(
            layer.get("materialization_plan_sha256")
            or plan.get("geometry_sha256")
            or plan.get("sha256")
            or _canonical_payload_sha256(plan)
        ),
        "render_bbox_frame": list(bbox),
    }


def _owner_map_panel(
    work_dir: Path,
    final_path: Path,
    fallback,
    *,
    return_metadata: bool = False,
):
    from PIL import Image, ImageDraw

    try:
        project = json.loads((work_dir / "project.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        result = fallback.copy()
        return (result, {}) if return_metadata else result
    pages = list(project.get("paginas") or [])
    page_index = 0
    for index, page in enumerate(pages):
        rendered = ((page.get("image_layers") or {}).get("rendered") or {}).get("path")
        if rendered and Path(rendered).name == final_path.name:
            page_index = index
            break
    page = pages[page_index] if page_index < len(pages) else {}
    page_id = str(page.get("page_id") or f"page_{page_index + 1:03d}")
    graph = next(
        (
            item
            for item in project.get("page_owner_graphs") or []
            if isinstance(item, dict) and item.get("page_id") == page_id
        ),
        None,
    )
    if not graph:
        result = fallback.copy()
        return (result, {}) if return_metadata else result
    try:
        geometry = _page_surface_geometry(page)
    except MatrixContractError:
        geometry = None
    try:
        canvas = Image.open(work_dir / "originals" / final_path.name).convert("RGBA")
    except OSError:
        canvas = fallback.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    owner_by_component = {
        component_id: owner.get("owner_id")
        for owner in graph.get("owners") or []
        for component_id in owner.get("component_ids") or []
    }
    palette = (
        (255, 59, 48, 110),
        (52, 199, 89, 110),
        (0, 122, 255, 110),
        (255, 149, 0, 110),
        (175, 82, 222, 110),
    )
    owner_colors: dict[str, tuple[int, int, int, int]] = {}
    polygons_frame: list[list[list[int]]] = []
    for component in graph.get("components") or []:
        component_id = str(component.get("component_id") or "")
        owner_id = str(owner_by_component.get(component_id) or component_id)
        color = owner_colors.setdefault(
            owner_id,
            palette[len(owner_colors) % len(palette)],
        )
        polygon = component.get("polygon_page") or []
        if geometry is not None and polygon:
            polygon = geometry.logical_polygon_to_frame(polygon)
        points = [tuple(map(int, point[:2])) for point in polygon if len(point) >= 2]
        if len(points) >= 3:
            polygons_frame.append([[int(x), int(y)] for x, y in points])
            draw.polygon(points, fill=color, outline=color[:3] + (255,), width=3)
        else:
            bbox = component.get("bbox_page") or []
            if len(bbox) >= 4:
                if geometry is not None:
                    bbox = geometry.logical_bbox_to_frame(tuple(int(value) for value in bbox[:4]))
                draw.rectangle(tuple(map(int, bbox[:4])), fill=color, outline=color[:3] + (255,), width=3)
    result = Image.alpha_composite(canvas, overlay).convert("RGB")
    metadata = {
        "page_id": page_id,
        "page_surface_geometry_sha256": (
            geometry.geometry_sha256 if geometry is not None else None
        ),
        "coordinate_space": "framed_page" if geometry is not None else "page",
        "polygons_frame": polygons_frame,
    }
    return (result, metadata) if return_metadata else result


def _write_contact_sheets(
    entries: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, list[str]]:
    from PIL import Image, ImageDraw

    sheets: dict[str, list[str]] = {}
    cached_pages: dict[tuple[str, str], list[str]] = {}
    sheet_root = output_root / "contact_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    declared_categories = {
        str(category)
        for entry in entries
        for category in entry.get("categories", [])
        if str(category)
    }
    for category in sorted(REQUIRED_VISUAL_CATEGORIES | declared_categories):
        category_paths: list[str] = []
        for entry in entries:
            if category not in entry.get("categories", []):
                continue
            work_dir = Path(entry["work_dir"])
            if not work_dir.is_absolute():
                work_dir = output_root / work_dir
            for target, final_path in _selected_target_finals(entry, category, work_dir):
                cache_key = (str(entry["entry_id"]), str(target.get("owner_id") or ""))
                if cache_key in cached_pages:
                    category_paths.extend(cached_pages[cache_key])
                    continue
                page_paths: list[str] = []
                try:
                    project = json.loads(
                        (work_dir / "project.json").read_text(encoding="utf-8-sig")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise MatrixContractError("project_result_invalid") from exc
                page = _project_page_by_id(project, str(target.get("page_id") or ""))
                strict_geometry = str(
                    ((target.get("source_crop") or {}).get("coordinate_space") or "")
                ) == "logical_page"
                artifact_bbox = None
                surface_geometry = None
                if strict_geometry:
                    artifact_bbox, surface_geometry = _target_artifact_bbox_frame(
                        target, page
                    )
                candidates = [
                    work_dir / "originals" / final_path.name,
                    work_dir / "images" / final_path.name,
                    final_path,
                ]
                base_panels = []
                for candidate in candidates:
                    try:
                        panel = Image.open(candidate).convert("RGB")
                    except OSError:
                        if strict_geometry:
                            raise MatrixContractError(
                                f"inspection_artifact_missing:{candidate}"
                            )
                        panel = Image.new("RGB", (320, 180), "#20252a")
                    base_panels.append(panel)
                canonical_size = base_panels[0].size
                if strict_geometry and any(panel.size != canonical_size for panel in base_panels):
                    raise MatrixContractError("inspection_artifact_dimension_mismatch")
                if not strict_geometry:
                    base_panels = [
                        panel if panel.size == canonical_size else panel.resize(canonical_size)
                        for panel in base_panels
                    ]
                owner_panel, owner_metadata = _owner_map_panel(
                    work_dir,
                    final_path,
                    base_panels[0],
                    return_metadata=True,
                )
                if owner_panel.size != canonical_size:
                    if strict_geometry:
                        raise MatrixContractError("owner_map_dimension_mismatch")
                    owner_panel = owner_panel.resize(canonical_size)
                raw_bbox = ((target.get("source_crop") or {}).get("bbox_page"))
                if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                    bbox = tuple(
                        artifact_bbox
                        if artifact_bbox is not None
                        else (int(value) for value in raw_bbox)
                    )
                    base_panels = [panel.crop(bbox) for panel in base_panels]
                    owner_panel = owner_panel.crop(bbox)
                evidence_panel = _difference_mask_panel(base_panels[0], base_panels[1])
                panel_contracts: dict[str, dict[str, Any]] = {}
                if strict_geometry:
                    requested_panel, intent_metadata = _resolved_intent_panel(
                        page, target, canonical_size, surface_geometry
                    )
                    if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                        requested_panel = requested_panel.crop(bbox)
                    observed_panel = _difference_mask_panel(base_panels[1], base_panels[2])
                    artifact_hashes = [_sha256_file(path) for path in candidates]
                    observation_sha = _canonical_payload_sha256({
                        "inpaint_sha256": artifact_hashes[1],
                        "final_sha256": artifact_hashes[2],
                        "page_surface_geometry_sha256": surface_geometry.geometry_sha256,
                    })
                    panel_contracts = {
                        "source": {
                            "artifact_path": str(candidates[0].resolve()),
                            "source_sha256": artifact_hashes[0],
                        },
                        "masks_evidence": {
                            "artifact_path": str(candidates[1].resolve()),
                            "source_sha256": artifact_hashes[1],
                        },
                        "requested_resolved": {
                            "artifact_path": "derived:owner_materialization_intent",
                            "source_sha256": intent_metadata["plan_sha256"],
                            **intent_metadata,
                        },
                        "observed_raster": {
                            "artifact_path": str(candidates[2].resolve()),
                            "source_sha256": artifact_hashes[2],
                            "observation_sha256": observation_sha,
                        },
                        "final": {
                            "artifact_path": str(candidates[2].resolve()),
                            "source_sha256": artifact_hashes[2],
                        },
                        "safe_geometry": {
                            "artifact_path": "derived:owner_graph_geometry",
                            "source_sha256": _canonical_payload_sha256(owner_metadata),
                        },
                    }
                else:
                    requested_panel = base_panels[0].copy()
                    observed_panel = base_panels[1]
                panels = [
                    base_panels[0],
                    evidence_panel,
                    requested_panel,
                    observed_panel,
                    base_panels[2],
                    owner_panel,
                ]
                labels = ("source", "masks/evidence", "requested/resolved", "observed raster", "final", "safe geometry")
                canonical_size = panels[0].size
                panel_width, page_height = canonical_size
                segment_height = 900
                safe_entry_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(entry["entry_id"]))
                safe_page_id = re.sub(
                    r"[^a-zA-Z0-9_.-]+",
                    "_",
                    f"{target.get('page_id')}__{target.get('owner_id')}",
                )
                for segment_index, y_top in enumerate(
                    range(0, page_height, segment_height),
                    start=1,
                ):
                    y_bottom = min(page_height, y_top + segment_height)
                    visible_height = y_bottom - y_top
                    sheet = Image.new(
                        "RGB",
                        (panel_width * len(panels), visible_height + 44),
                        "white",
                    )
                    draw = ImageDraw.Draw(sheet)
                    for panel_index, (label, panel) in enumerate(zip(labels, panels)):
                        crop = panel.crop((0, y_top, panel_width, y_bottom))
                        sheet.paste(crop, (panel_index * panel_width, 28))
                        draw.text((panel_index * panel_width + 6, 6), label, fill="black")
                    draw.text(
                        (6, visible_height + 30),
                        f"{entry['entry_id']} page={final_path.name} y={y_top}:{y_bottom}",
                        fill="black",
                    )
                    path = sheet_root / (
                        f"{category}__{safe_entry_id}__{safe_page_id}__{segment_index:03d}.png"
                    )
                    sheet.save(path)
                    metadata = {
                        "schema_version": 1,
                        "entry_id": str(entry["entry_id"]),
                        "page_id": str(target.get("page_id") or ""),
                        "owner_id": str(target.get("owner_id") or ""),
                        "bbox_logical": list(raw_bbox) if isinstance(raw_bbox, list) else None,
                        "artifact_bbox_frame": list(artifact_bbox) if artifact_bbox is not None else None,
                        "page_surface_geometry_sha256": (
                            surface_geometry.geometry_sha256
                            if surface_geometry is not None
                            else None
                        ),
                        "crop_dimensions": [panel_width, page_height],
                        "owner_map": owner_metadata,
                        "panels": panel_contracts,
                        "sheet_sha256": _sha256_file(path),
                    }
                    path.with_suffix(".metadata.json").write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    resolved_path = str(path.resolve())
                    category_paths.append(resolved_path)
                    page_paths.append(resolved_path)
                cached_pages[cache_key] = page_paths
        if category_paths:
            sheets[category] = category_paths
    return sheets


def build_inspection_template(
    sheets: dict[str, list[str]], output_root: Path,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    by_hash: dict[str, dict[str, Any]] = {}
    root = output_root.resolve()
    targets = {
        (str(entry.get("entry_id") or ""), str(target.get("page_id") or ""), str(target.get("owner_id") or "")): target
        for entry in entries or []
        for target in entry.get("targets") or []
        if isinstance(entry, dict) and isinstance(target, dict)
    }
    for category, paths in sorted(sheets.items()):
        for raw_path in paths:
            artifact = Path(raw_path).resolve()
            try:
                relative = artifact.relative_to(root).as_posix()
            except ValueError as exc:
                raise MatrixContractError(f"inspection artifact outside output root: {artifact}") from exc
            if not artifact.is_file():
                raise MatrixContractError(f"inspection artifact missing: {relative}")
            digest = _sha256_file(artifact)
            from PIL import Image

            with Image.open(artifact) as image:
                width, height = image.size
            if digest in by_hash:
                if relative != by_hash[digest]["artifact_path"]:
                    raise MatrixContractError(
                        "distinct output artifacts cannot claim identical pixels: "
                        f"{by_hash[digest]['artifact_path']} and {relative}"
                    )
                categories = by_hash[digest]["categories"]
                if category not in categories:
                    categories.append(category)
                continue
            row = {
                "artifact_path": relative,
                "sha256": digest,
                "required_scale": "native",
                "category": category,
                "categories": [category],
                "segment": artifact.stem,
                "width": width,
                "height": height,
                "panels": [
                    "source",
                    "masks_evidence",
                    "requested_resolved",
                    "observed_raster",
                    "final",
                    "safe_geometry",
                ],
            }
            metadata_path = artifact.with_suffix(".metadata.json")
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                identity = (
                    str(metadata.get("entry_id") or ""),
                    str(metadata.get("page_id") or ""),
                    str(metadata.get("owner_id") or ""),
                )
                target = targets.get(identity, {})
                row.update({
                    "entry_id": identity[0], "page_id": identity[1], "owner_id": identity[2],
                    "split": str(target.get("split") or ""),
                    "semantic_role": str(target.get("semantic_role") or ""),
                    "is_speech": bool(target.get("is_speech")),
                    "source_sha256": str((target.get("source_crop") or {}).get("sha256") or ""),
                    "final_sha256": str((((metadata.get("panels") or {}).get("final") or {}).get("source_sha256") or "")),
                    "page_surface_geometry_sha256": str(metadata.get("page_surface_geometry_sha256") or ""),
                })
            artifacts.append(row)
            by_hash[digest] = row
    return {"schema_version": 2, "artifacts": artifacts}


def validate_inspection_manifest(
    template: dict[str, Any],
    manifest: dict[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    template_schema = template.get("schema_version")
    manifest_schema = manifest.get("schema_version")
    if template_schema not in {2, 3} or manifest_schema != template_schema:
        raise MatrixContractError("inspection schemas must match and be supported")
    template_rows = template.get("artifacts") if isinstance(template, dict) else None
    inspection_rows = manifest.get("inspections") if isinstance(manifest, dict) else None
    if not isinstance(template_rows, list) or not isinstance(inspection_rows, list):
        raise MatrixContractError("inspection template and manifest require artifact lists")
    expected = {
        str(row.get("artifact_path") or ""): row
        for row in template_rows
        if isinstance(row, dict) and str(row.get("artifact_path") or "")
    }
    verified: list[dict[str, Any]] = []
    required_fields = {
        "artifact_path",
        "sha256",
        "scale",
        "timestamp",
        "category",
        "owner_or_segment",
        "functional_verdict",
        "style_verdict",
        "functional_note",
        "style_note",
    }
    if manifest_schema == 3:
        required_fields.update({
            "reviewed_at", "reviewer", "source_path", "output_path",
            "source_sha256", "output_sha256", "width", "height",
        })
    seen: set[str] = set()
    for index, row in enumerate(inspection_rows):
        if not isinstance(row, dict) or any(not str(row.get(key) or "").strip() for key in required_fields):
            raise MatrixContractError(f"inspection fields missing at row {index}")
        raw_path = str(row["artifact_path"])
        path = Path(raw_path)
        if path.is_absolute() or raw_path not in expected:
            raise MatrixContractError(f"unverified inspection claim: {raw_path}")
        if raw_path in seen:
            raise MatrixContractError(f"duplicate inspection claim: {raw_path}")
        seen.add(raw_path)
        expected_row = expected[raw_path]
        artifact = (output_root.resolve() / path).resolve()
        try:
            artifact.relative_to(output_root.resolve())
        except ValueError as exc:
            raise MatrixContractError(f"inspection path escapes output root: {raw_path}") from exc
        actual_hash = _sha256_file(artifact) if artifact.is_file() else ""
        claimed_hash = str(row.get("sha256") or "").lower()
        if claimed_hash != str(expected_row.get("sha256") or "").lower() or claimed_hash != actual_hash:
            raise MatrixContractError(f"inspection hash mismatch: {raw_path}")
        if str(row.get("scale")) != "native":
            raise MatrixContractError(f"inspection scale must be native: {raw_path}")
        if str(row.get("category")) != str(expected_row.get("category")):
            raise MatrixContractError(f"inspection category mismatch: {raw_path}")
        if str(row.get("owner_or_segment")) != str(expected_row.get("segment")):
            raise MatrixContractError(f"inspection segment mismatch: {raw_path}")
        if manifest_schema == 3:
            if str(row.get("output_path")) != raw_path or str(row.get("output_sha256") or "").lower() != actual_hash:
                raise InspectionEvidenceError(f"inspection output binding mismatch: {raw_path}")
            if str(row.get("reviewed_at")) != str(row.get("timestamp")):
                raise InspectionEvidenceError(f"inspection review timestamp mismatch: {raw_path}")
            if int(row.get("width") or 0) != int(expected_row.get("width") or 0) or int(
                row.get("height") or 0
            ) != int(expected_row.get("height") or 0):
                raise InspectionEvidenceError(f"inspection dimensions mismatch: {raw_path}")
            source_path = Path(str(row.get("source_path") or ""))
            if source_path.is_absolute():
                source_artifact = source_path.resolve()
            else:
                source_artifact = (output_root.resolve() / source_path).resolve()
            if not source_artifact.is_file() or _sha256_file(source_artifact) != str(
                row.get("source_sha256") or ""
            ).lower():
                raise InspectionEvidenceError(f"inspection source binding mismatch: {raw_path}")
            notes = {
                str(row.get("functional_note") or "").strip().casefold(),
                str(row.get("style_note") or "").strip().casefold(),
            }
            templated = {"clean", "ok", "go", "inspection pending", "template"}
            if notes & templated or len(notes) < 2:
                raise InspectionEvidenceError(
                    f"inspection notes do not prove an independent review: {raw_path}"
                )
        functional = str(row.get("functional_verdict")).upper()
        style = str(row.get("style_verdict")).upper()
        if functional not in {"GO", "NO-GO"} or style not in {"GO", "NO-GO"}:
            raise MatrixContractError(f"inspection verdict invalid: {raw_path}")
        overall = "GO" if functional == style == "GO" else "NO-GO"
        verified.append({**expected_row, **row, "artifact_path": raw_path, "sha256": claimed_hash, "overall_verdict": overall})
    missing = sorted(set(expected) - seen)
    for raw_path in missing:
        expected_row = expected[raw_path]
        verified.append(
            {
                "artifact_path": raw_path,
                "sha256": expected_row.get("sha256"),
                "category": expected_row.get("category"),
                "owner_or_segment": expected_row.get("segment"),
                "overall_verdict": "PENDING",
                "functional_verdict": "PENDING",
                "style_verdict": "PENDING",
                "functional_note": "inspection pending",
                "style_note": "inspection pending",
                **{
                    key: expected_row.get(key)
                    for key in (
                        "entry_id", "page_id", "owner_id", "split", "semantic_role", "is_speech",
                        "source_sha256", "final_sha256", "page_surface_geometry_sha256",
                    )
                },
            }
        )
    return verified


def _matrix_targets(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in matrix.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("entry_id") or "").strip()
        for target in entry.get("targets") or []:
            if not isinstance(target, dict):
                continue
            row = {
                **target,
                "entry_id": entry_id,
                "split": str(target.get("split") or entry.get("split") or ""),
            }
            key = (entry_id, str(row.get("page_id") or ""), str(row.get("owner_id") or ""))
            if not all(key) or key in seen:
                raise MatrixContractError(f"duplicate or invalid matrix target: {key}")
            seen.add(key)
            targets.append(row)
    return targets


def build_style_holdout_summary(
    *,
    matrix: dict[str, Any],
    inspection: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build GO rates from authenticated holdout owner targets only."""

    findings: list[dict[str, Any]] = []
    try:
        targets = _matrix_targets(matrix)
    except MatrixContractError as exc:
        targets = []
        findings.append({"code": "matrix_target_invalid", "detail": str(exc)})
    target_by_key = {
        (str(row["entry_id"]), str(row["page_id"]), str(row["owner_id"])): row
        for row in targets
    }
    rows_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    seen_artifacts: set[tuple[tuple[str, str, str], str]] = set()
    for row in inspection if isinstance(inspection, list) else []:
        if not isinstance(row, dict):
            findings.append({"code": "inspection_row_invalid"})
            continue
        key = (str(row.get("entry_id") or ""), str(row.get("page_id") or ""), str(row.get("owner_id") or ""))
        target = target_by_key.get(key)
        if target is None:
            findings.append({"code": "inspection_target_unbound", "target_key": ":".join(key)})
            continue
        artifact_key = (key, str(row.get("artifact_path") or ""))
        if not artifact_key[1] or artifact_key in seen_artifacts:
            findings.append({"code": "inspection_target_duplicate", "target_key": ":".join(key)})
            continue
        seen_artifacts.add(artifact_key)
        expected_source = str((target.get("source_crop") or {}).get("sha256") or "").lower()
        bindings = {
            "category": str(target.get("category") or ""),
            "split": str(target.get("split") or ""),
            "source_sha256": expected_source,
        }
        mismatch = [name for name, expected in bindings.items() if str(row.get(name) or "").lower() != expected.lower()]
        if mismatch or len(str(row.get("final_sha256") or "")) != 64 or len(str(row.get("page_surface_geometry_sha256") or "")) != 64:
            findings.append({"code": "inspection_binding_mismatch", "target_key": ":".join(key), "fields": mismatch})
            continue
        rows_by_key.setdefault(key, []).append(row)

    holdout = [row for row in targets if row.get("split") == "holdout"]
    categories: dict[str, dict[str, Any]] = {}
    owner_go = 0
    speech_evaluated = speech_go = 0
    inspected_count = 0
    for target in holdout:
        key = (str(target["entry_id"]), str(target["page_id"]), str(target["owner_id"]))
        rows = rows_by_key.get(key, [])
        complete = bool(rows)
        go = complete and all(
            str(row.get("overall_verdict") or "PENDING").upper() == "GO"
            and str(row.get("functional_verdict") or "PENDING").upper() == "GO"
            and str(row.get("style_verdict") or "PENDING").upper() == "GO"
            for row in rows
        )
        inspected_count += int(complete)
        owner_go += int(go)
        if not complete:
            findings.append({"code": "inspection_target_missing", "target_key": ":".join(key)})
        elif not go:
            findings.append({"code": "inspection_target_not_go", "target_key": ":".join(key)})
        if bool(target.get("is_speech")):
            speech_evaluated += 1
            speech_go += int(go)
        category = str(target.get("category") or "")
        metric = categories.setdefault(category, {"evaluated": 0, "go": 0, "go_rate": 0.0})
        metric["evaluated"] += 1
        metric["go"] += int(go)
    for metric in categories.values():
        metric["go_rate"] = metric["go"] / metric["evaluated"] if metric["evaluated"] else 0.0
    for category in REQUIRED_HOLDOUT_CATEGORIES:
        metric = categories.setdefault(category, {"evaluated": 0, "go": 0, "go_rate": 0.0})
        if metric["evaluated"] == 0:
            findings.append({"code": "required_holdout_category_empty", "category": category})
    owners_evaluated = len(holdout)
    summary = {
        "schema_version": 1,
        "status": "BLOCK" if findings else "PASS",
        "required_holdout_categories": list(REQUIRED_HOLDOUT_CATEGORIES),
        "calibration_owner_count": sum(row.get("split") == "calibration" for row in targets),
        "owners": {"evaluated": owners_evaluated, "go": owner_go, "go_rate": owner_go / owners_evaluated if owners_evaluated else 0.0},
        "speech": {"evaluated": speech_evaluated, "go": speech_go, "go_rate": speech_go / speech_evaluated if speech_evaluated else 0.0},
        "categories": categories,
        "inspection_coverage": {"evaluated": owners_evaluated, "inspected": inspected_count, "rate": inspected_count / owners_evaluated if owners_evaluated else 0.0},
        "matrix_sha256": _canonical_payload_sha256(matrix),
        "inspection_sha256": _canonical_payload_sha256(inspection),
        "findings": findings,
    }
    return summary


def _verified_report(report: dict[str, Any]) -> bool:
    if int(report.get("schema_version") or 0) != 3:
        return False
    claimed = str(report.get("report_sha256") or "")
    return len(claimed) == 64 and claimed == _canonical_payload_sha256(report, omit="report_sha256")


def build_owner_qa_summary(
    *,
    matrix: dict[str, Any],
    entry_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate only self-hashed style QA reports bound to matrix entries."""

    findings: list[dict[str, Any]] = []
    targets = _matrix_targets(matrix)
    expected_entries = {str(entry.get("entry_id") or "") for entry in matrix.get("entries") or [] if isinstance(entry, dict)}
    if set(entry_reports) != expected_entries:
        findings.append({"code": "entry_style_report_set_mismatch"})
    eligible = rendered = observation_count = 0
    safe_evaluated = safe_contained = 0
    effect_evaluated = effect_contained = 0
    catastrophic_evaluated = catastrophic_count = 0
    seen_owner_keys: set[tuple[str, str, str]] = set()
    seen_run_ids: set[str] = set()
    embedded: list[dict[str, Any]] = []
    for entry_id in sorted(expected_entries):
        report = entry_reports.get(entry_id)
        if not isinstance(report, dict) or not _verified_report(report):
            findings.append({"code": "style_report_hash_mismatch", "entry_id": entry_id})
            continue
        embedded.append({"entry_id": entry_id, "report": report})
        run_id = str(report.get("run_id") or "").strip()
        if not run_id or run_id in seen_run_ids:
            findings.append({"code": "style_report_run_identity_invalid", "entry_id": entry_id})
        seen_run_ids.add(run_id)
        if str((report.get("gate") or {}).get("status") or "BLOCK").upper() != "PASS":
            findings.append({"code": "style_report_blocked", "entry_id": entry_id})
        summary = report.get("summary") or {}
        eligible += int(summary.get("eligible_owner_count") or 0)
        rendered += int(summary.get("rendered_owner_count") or 0)
        owners = report.get("owners") or []
        if len(owners) != int(summary.get("eligible_owner_count") or 0):
            findings.append({"code": "style_report_owner_denominator_mismatch", "entry_id": entry_id})
        for owner in owners:
            if not isinstance(owner, dict):
                continue
            key = (entry_id, str(owner.get("page_id") or ""), str(owner.get("owner_id") or ""))
            if key in seen_owner_keys:
                findings.append({"code": "duplicate_owner_style_report", "target_key": ":".join(key)})
            seen_owner_keys.add(key)
            observation_count += int(len(str(owner.get("materialization_observation_sha256") or "")) == 64)
        metrics = report.get("metrics") or {}
        safe = metrics.get("safe_containment") or {}
        effect = metrics.get("effect_containment") or {}
        catastrophic = metrics.get("catastrophic_mismatches") or {}
        safe_evaluated += int(safe.get("evaluated") or 0); safe_contained += int(safe.get("contained") or 0)
        effect_evaluated += int(effect.get("evaluated") or 0); effect_contained += int(effect.get("contained") or 0)
        catastrophic_evaluated += int(catastrophic.get("evaluated") or 0); catastrophic_count += int(catastrophic.get("count") or 0)
    expected_target_keys = {(str(row["entry_id"]), str(row["page_id"]), str(row["owner_id"])) for row in targets}
    if not expected_target_keys.issubset(seen_owner_keys):
        findings.append({"code": "matrix_target_missing_from_style_reports"})
    coverage = observation_count / eligible if eligible else 0.0
    if not eligible or coverage < 1.0:
        findings.append({"code": "materialization_observation_coverage_incomplete"})
    return {
        "schema_version": 1,
        "status": "BLOCK" if findings else "PASS",
        "eligible_owner_count": eligible,
        "rendered_owner_count": rendered,
        "materialization_observation_coverage": coverage,
        "safe_containment": {"evaluated": safe_evaluated, "contained": safe_contained, "rate": safe_contained / safe_evaluated if safe_evaluated else 0.0},
        "effect_containment": {"evaluated": effect_evaluated, "contained": effect_contained, "rate": effect_contained / effect_evaluated if effect_evaluated else 0.0},
        "catastrophic_mismatches": {"evaluated": catastrophic_evaluated, "count": catastrophic_count},
        "source_reports": embedded,
        "source_reports_sha256": _canonical_payload_sha256(embedded),
        "findings": findings,
    }


def _write_report(
    path: Path,
    results: list[dict[str, Any]],
    run_results: list[dict[str, Any]],
    sheets: dict[str, list[str]],
    *,
    inspected: list[dict[str, Any]] | None = None,
) -> None:
    grouped = group_failures_by_contract(results)
    functional_go = bool(results) and all(item.get("functional_status", item["status"]) == "PASS" for item in results)
    style_go = bool(results) and all(item.get("style_status", item["status"]) == "PASS" for item in results)
    inspection_status = "PENDING" if inspected is None or any(
        str(item.get("overall_verdict") or "PENDING").upper() == "PENDING" for item in inspected
    ) else "GO" if inspected and all(str(item.get("overall_verdict")).upper() == "GO" for item in inspected) else "NO-GO"
    overall = "NO-GO" if not functional_go or not style_go or inspection_status == "NO-GO" else "PENDING" if inspection_status == "PENDING" else "GO"
    lines = [
        "# Page-owner systemic validation",
        "",
        f"Verdict: **{overall}**",
        f"Functional: **{'GO' if functional_go else 'NO-GO'}**",
        f"Style: **{'GO' if style_go else 'NO-GO'}**",
        f"Inspection: **{inspection_status}**",
        "",
        "## Matrix results",
        "",
        "| Entry | Categories | Pages | Gate | Status | Contracts |",
        "|---|---|---:|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| {entry_id} | {categories} | {page_count} | {export_gate} | {status} | {contracts} |".format(
                entry_id=result["entry_id"],
                categories=", ".join(result.get("categories") or []),
                page_count=result.get("page_count", 0),
                export_gate=result.get("export_gate", "MISSING"),
                status=result["status"],
                contracts=", ".join(result.get("contracts") or []) or "none",
            )
        )
    lines.extend(["", "## Failures grouped by contract", ""])
    if grouped:
        for contract, entry_ids in grouped.items():
            lines.append(f"- `{contract}`: {', '.join(entry_ids)}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Generated artifacts", ""])
    for category, paths in sorted(sheets.items()):
        lines.append(f"- {category}: " + ", ".join(paths))
    lines.extend(["", "## Automatically checked", ""])
    for result in results:
        lines.append(
            f"- {result['entry_id']}: project={result.get('project_path', 'n/a')}; "
            f"gate={result.get('export_gate', 'MISSING')}; status={result['status']}"
        )
    lines.extend(["", "## Visually inspected", ""])
    if inspected:
        for row in inspected:
            lines.append(
                f"- {row['artifact_path']}: sha256={row['sha256']}; "
                f"functional={row.get('functional_verdict')}; style={row.get('style_verdict')}; "
                f"overall={row.get('overall_verdict')}; functional_note={row.get('functional_note')}; "
                f"style_note={row.get('style_note')}"
            )
    else:
        lines.append("- None verified.")
    lines.extend(["", "## Runner exit evidence", ""])
    for result in run_results:
        lines.append(
            f"- {result['entry_id']}: returncode={result['returncode']}; "
            f"stdout={result.get('stdout_path', 'n/a')}; "
            f"stderr={result.get('stderr_path', 'n/a')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_matrix_cli(args: argparse.Namespace) -> int:
    run_id = str(args.run_id or "").strip()
    if not run_id or any(character.isspace() for character in run_id):
        raise MatrixContractError("run_id must be a non-empty token")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entries = validate_manifest(manifest)
    acceptance_bundle: dict[str, Any] | None = None
    if args.acceptance_bundle:
        acceptance_bundle = json.loads(args.acceptance_bundle.resolve().read_text(encoding="utf-8-sig"))
        if str(acceptance_bundle.get("matrix_sha256") or "") != _canonical_payload_sha256(manifest):
            raise MatrixContractError("matrix hash does not match acceptance bundle")
    raw_inputs_path = Path(str(manifest.get("inputs_path") or "inputs.json"))
    if raw_inputs_path.is_absolute():
        raise MatrixContractError("inputs_path must be relative to the matrix fixture")
    inputs_path = (manifest_path.parent / raw_inputs_path).resolve()
    try:
        inputs_path.relative_to(manifest_path.parent.parent)
    except ValueError as exc:
        raise MatrixContractError("inputs_path escapes the versioned fixtures root") from exc
    if not inputs_path.is_file():
        raise MatrixContractError(f"inputs manifest missing: {inputs_path}")
    inputs_manifest = json.loads(inputs_path.read_text(encoding="utf-8-sig"))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_results = []
    runtimes: dict[str, dict[str, Any]] = {}
    for entry in entries:
        try:
            runtimes[entry["entry_id"]] = resolve_entry_runtime(entry, manifest_path, inputs_manifest)
        except MatrixContractError as exc:
            entry.setdefault("_preflight_contracts", []).append("matrix_input_preflight_failed")
            entry.setdefault("_preflight_details", []).append(str(exc))
    if not args.validate_only:
        for entry in entries:
            runtime = runtimes.get(entry["entry_id"])
            if runtime is None:
                run_results.append({"entry_id": entry["entry_id"], "returncode": 2})
                continue
            run_results.append(_run_entry(entry, manifest_path, output_root, runtime, args.acceptance_bundle))
    else:
        for entry in entries:
            work_dir = Path(entry["work_dir"])
            if not work_dir.is_absolute():
                work_dir = output_root / work_dir
            runner_path = work_dir / "run_manifest.json"
            try:
                runner_evidence = validate_runner_evidence(
                    json.loads(runner_path.read_text(encoding="utf-8-sig"))
                )
            except (OSError, json.JSONDecodeError, MatrixContractError) as exc:
                entry.setdefault("_preflight_contracts", []).append("runner_manifest_invalid")
                entry.setdefault("_preflight_details", []).append(str(exc))
                run_results.append({"entry_id": entry["entry_id"], "returncode": 2})
            else:
                run_results.append(
                    {"entry_id": entry["entry_id"], "returncode": runner_evidence["returncode"], "runner_evidence": runner_evidence}
                )
    results = [validate_entry_result(entry, output_root) for entry in entries]
    external_audits: dict[str, dict[str, Any]] = {}
    if args.require_external_audit:
        if args.audit_dir is None:
            raise MatrixContractError("--require-external-audit requires --audit-dir")
        try:
            external_audits = _run_external_audits(
                entries,
                runtimes,
                output_root,
                args.audit_dir.resolve(),
            )
        except MatrixContractError as exc:
            for result in results:
                result["status"] = "BLOCK"
                result["functional_status"] = "BLOCK"
                result.setdefault("contracts", []).append("external_audit_failed")
                result.setdefault("details", []).append(str(exc))
    sheets = _write_contact_sheets(entries, output_root)
    inspection_template = build_inspection_template(sheets, output_root, entries)
    if args.inspection_template:
        args.inspection_template.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.inspection_template.resolve().write_text(
            json.dumps(inspection_template, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    inspected: list[dict[str, Any]] | None = None
    if args.inspection_manifest:
        inspection_manifest = json.loads(args.inspection_manifest.resolve().read_text(encoding="utf-8-sig"))
        inspected = validate_inspection_manifest(inspection_template, inspection_manifest, output_root)
    entry_reports: dict[str, dict[str, Any]] = {}
    for entry in entries:
        work_dir = Path(entry["work_dir"])
        if not work_dir.is_absolute():
            work_dir = output_root / work_dir
        try:
            project = json.loads((work_dir / "project.json").read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
        report = qa.get("style_fidelity") if isinstance(qa.get("style_fidelity"), dict) else None
        if report is not None:
            entry_reports[str(entry["entry_id"])] = report
    owner_qa_summary = build_owner_qa_summary(matrix=manifest, entry_reports=entry_reports)
    holdout_summary = build_style_holdout_summary(matrix=manifest, inspection=inspected or [])
    binding = {
        "acceptance_bundle_id": acceptance_bundle.get("acceptance_bundle_id") if acceptance_bundle else None,
        "revision_sha256": acceptance_bundle.get("revision_sha256") if acceptance_bundle else None,
        "source_manifest_sha256": acceptance_bundle.get("source_manifest_sha256") if acceptance_bundle else None,
    }
    owner_qa_summary.update({**binding, "producer_run_id": f"owner-qa:{run_id}"})
    holdout_summary.update({**binding, "producer_run_id": f"inspection:{run_id}"})
    (output_root / "style_owner_qa_summary.json").write_text(
        json.dumps(owner_qa_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "style_holdout_summary.json").write_text(
        json.dumps(holdout_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    functional_go = bool(results) and all(item.get("functional_status", item["status"]) == "PASS" for item in results)
    execution_findings = [
        {"code": "functional_matrix_blocked"} for _ in [0] if not functional_go
    ] + [
        {"code": "owner_style_qa_blocked"} for _ in [0] if owner_qa_summary["status"] != "PASS"
    ] + list(holdout_summary["findings"])
    execution_entries: list[dict[str, Any]] = []
    for entry in entries:
        work_dir = Path(entry["work_dir"])
        if not work_dir.is_absolute():
            work_dir = output_root / work_dir
        project_path = work_dir / "project.json"
        project = json.loads(project_path.read_text(encoding="utf-8-sig")) if project_path.is_file() else {}
        qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
        gate = qa.get("export_gate") if isinstance(qa.get("export_gate"), dict) else {}
        reports = [row for row in qa.get("final_pixel_reports") or [] if isinstance(row, dict)]
        contract_status = lambda name: "PASS" if reports and all(str((row.get("contracts") or {}).get(name) or "BLOCK") == "PASS" for row in reports) else "BLOCK"
        ledger_path = work_dir / "execution_source_ledger.json"
        execution_entries.append({
            "entry_id": entry["entry_id"],
            "project_json_path": str(project_path.resolve()),
            "project_json_sha256": _sha256_file(project_path) if project_path.is_file() else None,
            "export_gate_path": f"{project_path.resolve()}#qa.export_gate",
            "export_gate_sha256": _canonical_payload_sha256(gate) if gate else None,
            "export_gate_status": str(gate.get("status") or "BLOCK"),
            "route_gate_status": contract_status("route_state_contract"),
            "final_language_gate_status": contract_status("final_language_contract"),
            "inpaint_residual_gate_status": contract_status("residual_cleanup_contract"),
            "pipeline_child_ledger_path": str(ledger_path.resolve()),
            "pipeline_child_ledger_sha256": _sha256_file(ledger_path) if ledger_path.is_file() else None,
            "acceptance_bundle_id": binding["acceptance_bundle_id"],
        })
    if any(
        row["export_gate_status"] != "PASS"
        or row["route_gate_status"] != "PASS"
        or row["final_language_gate_status"] != "PASS"
        or row["inpaint_residual_gate_status"] != "PASS"
        or (acceptance_bundle is not None and not row["pipeline_child_ledger_sha256"])
        for row in execution_entries
    ):
        functional_go = False
    if not functional_go and not any(row.get("code") == "functional_matrix_blocked" for row in execution_findings):
        execution_findings.append({"code": "functional_matrix_blocked"})
    execution_summary = {
        "schema_version": 1,
        **binding,
        "producer_run_id": run_id,
        "functional_status": "GO" if functional_go else "BLOCK",
        "style_status": "GO" if owner_qa_summary["status"] == "PASS" else "BLOCK",
        "inspection_status": "GO" if holdout_summary["status"] == "PASS" else "PENDING" if all(
            row.get("code") in {"inspection_target_missing", "inspection_target_not_go"}
            for row in holdout_summary["findings"]
        ) else "BLOCK",
        "findings": execution_findings,
        "entries": execution_entries,
        "external_audits": {
            entry_id: {
                "report_sha256": payload.get("report_sha256")
                or _canonical_payload_sha256(payload),
                "status": payload.get("external_audit_gate_status"),
                "page_count": payload.get("external_page_audits_count"),
            }
            for entry_id, payload in sorted(external_audits.items())
        },
    }
    (output_root / "matrix_execution_summary.json").write_text(
        json.dumps(execution_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(args.report.resolve(), results, run_results, sheets, inspected=inspected)
    inspection_go = bool(inspected) and all(
        str(item.get("overall_verdict") or "PENDING").upper() == "GO" for item in inspected
    )
    if args.require_external_audit:
        return 0 if functional_go else 2
    return 0 if functional_go and owner_qa_summary["status"] == "PASS" and holdout_summary["status"] == "PASS" and inspection_go else 2


def validate_matrix_cli_request(
    *,
    manifest: Path,
    output_root: Path,
    report: Path,
    validate_only: bool,
    inspection_template: Path | None,
    inspection_manifest: Path | None,
    acceptance_bundle: Path | None,
    audit_dir: Path | None,
    require_external_audit: bool,
    run_id: str,
) -> int:
    """Execute one fully resolved CLI request without reparsing process state."""

    return _run_matrix_cli(argparse.Namespace(
        manifest=manifest,
        output_root=output_root,
        report=report,
        validate_only=validate_only,
        inspection_template=inspection_template,
        inspection_manifest=inspection_manifest,
        acceptance_bundle=acceptance_bundle,
        audit_dir=audit_dir,
        require_external_audit=require_external_audit,
        run_id=run_id,
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--inspection-template", type=Path)
    parser.add_argument("--inspection-manifest", type=Path)
    parser.add_argument("--acceptance-bundle", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--require-external-audit", action="store_true")
    parser.add_argument("--run-id", required=True)
    try:
        args = parser.parse_args(argv)
        result = validate_matrix_cli_request(
            manifest=args.manifest.resolve(),
            output_root=args.output_root.resolve(),
            report=args.report.resolve(),
            validate_only=bool(args.validate_only),
            inspection_template=(args.inspection_template.resolve() if args.inspection_template else None),
            inspection_manifest=(args.inspection_manifest.resolve() if args.inspection_manifest else None),
            acceptance_bundle=(args.acceptance_bundle.resolve() if args.acceptance_bundle else None),
            audit_dir=(args.audit_dir.resolve() if args.audit_dir else None),
            require_external_audit=bool(args.require_external_audit),
            run_id=str(args.run_id),
        )
        status = getattr(result, "status", None)
        return (0 if str(status).upper() == "PASS" else 2) if status is not None else int(result)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"owner visual matrix error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
