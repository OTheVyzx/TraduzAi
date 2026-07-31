"""Run and validate a fresh cross-work owner visual matrix.

This harness is intentionally outside the production pipeline.  It groups
failures by invariant/contract, never by title, chapter, page, or band.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


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


class MatrixContractError(ValueError):
    """Raised when the validation corpus itself is not systemic or fresh."""


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
    categories = entry.get("categories")
    if not isinstance(categories, list) or not all(
        isinstance(value, str) and value.strip() for value in categories
    ):
        raise MatrixContractError(f"matrix entry {index} has invalid categories")
    normalized = dict(entry)
    normalized["categories"] = sorted(set(categories))
    if normalized["split"] not in {"calibration", "holdout"}:
        raise MatrixContractError(f"matrix entry {index} has invalid split")
    return normalized


def validate_manifest(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
        raise MatrixContractError("matrix manifest must contain entries[]")
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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if style_fidelity is not None:
        style_gate = style_fidelity.get("gate") if isinstance(style_fidelity.get("gate"), dict) else {}
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
    return {
        "entry_id": entry["entry_id"],
        "work_id": entry.get("work_id"),
        "chapter_id": entry.get("chapter_id"),
        "categories": list(entry.get("categories") or []),
        "work_dir": str(work_dir.resolve()),
        "project_path": str(project_path.resolve()),
        "status": "PASS" if not contracts else "BLOCK",
        "contracts": sorted(contracts),
        "details": details,
        "page_count": len(seen_pages),
        "export_gate": gate.get("status") or "MISSING",
        "style_fallback_owner_ids": sorted(style_fallback_owner_ids),
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


def _run_entry(
    entry: dict[str, Any],
    manifest_path: Path,
    output_root: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    target = Path(entry["work_dir"])
    if not target.is_absolute():
        target = output_root / target
    if target.exists():
        raise MatrixContractError(f"fresh matrix work_dir already exists: {target}")
    config_path = Path(runtime["config_path"])
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
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
    completed = subprocess.run(
        [sys.executable, str(pipeline_main), str(effective_config)],
        cwd=str(pipeline_main.parent),
        check=False,
        text=True,
        capture_output=True,
    )
    log_paths = _persist_runner_logs(
        output_root,
        entry["entry_id"],
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    return {
        "entry_id": entry["entry_id"],
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        **log_paths,
    }


def _selected_final_paths(
    entry: dict[str, Any],
    category: str,
    work_dir: Path,
) -> list[Path]:
    final_files = sorted((work_dir / "translated").glob("*"))
    if not final_files:
        return []
    raw_page = (entry.get("category_pages") or {}).get(category, 1)
    raw_pages = raw_page if isinstance(raw_page, list) else [raw_page]
    selected: list[Path] = []
    for value in raw_pages:
        try:
            page_index = max(1, int(value)) - 1
        except (TypeError, ValueError):
            page_index = 0
        candidate = final_files[min(page_index, len(final_files) - 1)]
        if candidate not in selected:
            selected.append(candidate)
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


def _owner_map_panel(work_dir: Path, final_path: Path, fallback):
    from PIL import Image, ImageDraw

    try:
        project = json.loads((work_dir / "project.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return fallback.copy()
    pages = list(project.get("paginas") or [])
    page_index = 0
    for index, page in enumerate(pages):
        rendered = ((page.get("image_layers") or {}).get("rendered") or {}).get("path")
        if rendered and Path(rendered).name == final_path.name:
            page_index = index
            break
    page_id = f"page_{page_index + 1:03d}"
    graph = next(
        (
            item
            for item in project.get("page_owner_graphs") or []
            if isinstance(item, dict) and item.get("page_id") == page_id
        ),
        None,
    )
    if not graph:
        return fallback.copy()
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
    for component in graph.get("components") or []:
        component_id = str(component.get("component_id") or "")
        owner_id = str(owner_by_component.get(component_id) or component_id)
        color = owner_colors.setdefault(
            owner_id,
            palette[len(owner_colors) % len(palette)],
        )
        polygon = component.get("polygon_page") or []
        points = [tuple(map(int, point[:2])) for point in polygon if len(point) >= 2]
        if len(points) >= 3:
            draw.polygon(points, fill=color, outline=color[:3] + (255,), width=3)
        else:
            bbox = component.get("bbox_page") or []
            if len(bbox) >= 4:
                draw.rectangle(tuple(map(int, bbox[:4])), fill=color, outline=color[:3] + (255,), width=3)
    return Image.alpha_composite(canvas, overlay).convert("RGB")


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
            for final_path in _selected_final_paths(entry, category, work_dir):
                cache_key = (str(entry["entry_id"]), str(final_path.resolve()))
                if cache_key in cached_pages:
                    category_paths.extend(cached_pages[cache_key])
                    continue
                page_paths: list[str] = []
                candidates = [
                    work_dir / "originals" / final_path.name,
                    work_dir / "images" / final_path.name,
                    final_path,
                ]
                panels = []
                for candidate in candidates:
                    try:
                        panel = Image.open(candidate).convert("RGB")
                    except OSError:
                        panel = Image.new("RGB", (320, 180), "#20252a")
                    panels.append(panel)
                canonical_size = panels[0].size
                panels = [
                    panel if panel.size == canonical_size else panel.resize(canonical_size)
                    for panel in panels
                ]
                mask_panel = _difference_mask_panel(panels[0], panels[1])
                owner_panel = _owner_map_panel(work_dir, final_path, panels[0])
                if owner_panel.size != canonical_size:
                    owner_panel = owner_panel.resize(canonical_size)
                panels.extend((mask_panel, owner_panel))
                labels = ("before", "inpaint", "final", "masks", "owner map")
                panel_width, page_height = canonical_size
                segment_height = 900
                safe_entry_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(entry["entry_id"]))
                safe_page_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", final_path.stem)
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
                    resolved_path = str(path.resolve())
                    category_paths.append(resolved_path)
                    page_paths.append(resolved_path)
                cached_pages[cache_key] = page_paths
        if category_paths:
            sheets[category] = category_paths
    return sheets


def build_inspection_template(
    sheets: dict[str, list[str]], output_root: Path
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    by_hash: dict[str, dict[str, Any]] = {}
    root = output_root.resolve()
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
            if digest in by_hash:
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
            }
            artifacts.append(row)
            by_hash[digest] = row
    return {"schema_version": 1, "artifacts": artifacts}


def validate_inspection_manifest(
    template: dict[str, Any],
    manifest: dict[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
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
        "verdict",
        "note",
    }
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
        if str(row.get("verdict")).upper() not in {"GO", "NO-GO"}:
            raise MatrixContractError(f"inspection verdict invalid: {raw_path}")
        verified.append({**row, "artifact_path": raw_path, "sha256": claimed_hash})
    missing = sorted(set(expected) - seen)
    if missing:
        raise MatrixContractError(
            "inspection manifest missing required artifacts: " + ", ".join(missing)
        )
    return verified


def _write_report(
    path: Path,
    results: list[dict[str, Any]],
    run_results: list[dict[str, Any]],
    sheets: dict[str, list[str]],
    *,
    inspected: list[dict[str, Any]] | None = None,
) -> None:
    grouped = group_failures_by_contract(results)
    functional_go = bool(results) and all(item["status"] == "PASS" for item in results)
    inspection_go = inspected is None or (
        bool(inspected) and all(str(item.get("verdict")).upper() == "GO" for item in inspected)
    )
    overall = "GO" if functional_go and inspection_go else "NO-GO"
    lines = [
        "# Page-owner systemic validation",
        "",
        f"Verdict: **{overall}**",
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
                f"verdict={row['verdict']}; note={row['note']}"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--inspection-template", type=Path)
    parser.add_argument("--inspection-manifest", type=Path)
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entries = validate_manifest(manifest)
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
            run_results.append(_run_entry(entry, manifest_path, output_root, runtime))
    results = [validate_entry_result(entry, output_root) for entry in entries]
    sheets = _write_contact_sheets(entries, output_root)
    inspection_template = build_inspection_template(sheets, output_root)
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
    _write_report(args.report.resolve(), results, run_results, sheets, inspected=inspected)
    functional_go = bool(results) and all(item["status"] == "PASS" for item in results)
    inspection_go = inspected is None or (
        bool(inspected) and all(str(item.get("verdict")).upper() == "GO" for item in inspected)
    )
    return 0 if functional_go and inspection_go else 2


if __name__ == "__main__":
    raise SystemExit(main())
