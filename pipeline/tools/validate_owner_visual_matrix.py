"""Run and validate a fresh cross-work owner visual matrix.

This harness is intentionally outside the production pipeline.  It groups
failures by invariant/contract, never by title, chapter, page, or band.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
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


class MatrixContractError(ValueError):
    """Raised when the validation corpus itself is not systemic or fresh."""


def _canonical_entry(entry: Any, index: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise MatrixContractError(f"matrix entry {index} must be an object")
    required = ("entry_id", "work_id", "chapter_id", "config_path", "work_dir")
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
    return normalized


def validate_manifest(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
        raise MatrixContractError("matrix manifest must contain entries[]")
    entries = [_canonical_entry(entry, index) for index, entry in enumerate(manifest["entries"])]
    works = {entry["work_id"] for entry in entries}
    if len(works) < 3:
        raise MatrixContractError("matrix requires at least three distinct works")
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
    qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
    gate = qa.get("export_gate") if isinstance(qa.get("export_gate"), dict) else {}
    if gate.get("status") != "PASS":
        contracts.add("export_gate_blocked")
    for issue in gate.get("issues") or []:
        if isinstance(issue, dict) and (
            str(issue.get("severity") or "").lower() in {"critical", "blocker"}
            or bool(issue.get("blocking"))
        ):
            contracts.add(_contract_name(issue))
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


def _run_entry(entry: dict[str, Any], manifest_path: Path, output_root: Path) -> dict[str, Any]:
    target = Path(entry["work_dir"])
    if not target.is_absolute():
        target = output_root / target
    if target.exists():
        raise MatrixContractError(f"fresh matrix work_dir already exists: {target}")
    config_path = _resolve_from_manifest(manifest_path, entry["config_path"])
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    config.update(
        {
            "work_dir": str(target.resolve()),
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


def _selected_final_path(
    entry: dict[str, Any],
    category: str,
    work_dir: Path,
) -> Path | None:
    final_files = sorted((work_dir / "translated").glob("*"))
    if not final_files:
        return None
    raw_page = (entry.get("category_pages") or {}).get(category, 1)
    try:
        page_index = max(1, int(raw_page)) - 1
    except (TypeError, ValueError):
        page_index = 0
    return final_files[min(page_index, len(final_files) - 1)]


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
    sheet_root = output_root / "contact_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    for category in sorted(REQUIRED_VISUAL_CATEGORIES):
        category_paths: list[str] = []
        for entry in entries:
            if category not in entry.get("categories", []):
                continue
            work_dir = Path(entry["work_dir"])
            if not work_dir.is_absolute():
                work_dir = output_root / work_dir
            final_path = _selected_final_path(entry, category, work_dir)
            if final_path is None:
                continue
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
                    f"{entry['entry_id']} y={y_top}:{y_bottom}",
                    fill="black",
                )
                path = sheet_root / (
                    f"{category}__{safe_entry_id}__{segment_index:03d}.png"
                )
                sheet.save(path)
                category_paths.append(str(path.resolve()))
        if category_paths:
            sheets[category] = category_paths
    return sheets


def _write_report(
    path: Path,
    results: list[dict[str, Any]],
    run_results: list[dict[str, Any]],
    sheets: dict[str, list[str]],
) -> None:
    grouped = group_failures_by_contract(results)
    overall = "GO" if results and all(item["status"] == "PASS" for item in results) else "NO-GO"
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
    lines.extend(["", "## Contact sheets", ""])
    for category, paths in sorted(sheets.items()):
        lines.append(f"- {category}: " + ", ".join(paths))
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
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    entries = validate_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_results = []
    if not args.validate_only:
        for entry in entries:
            run_results.append(_run_entry(entry, manifest_path, output_root))
    results = [validate_entry_result(entry, output_root) for entry in entries]
    sheets = _write_contact_sheets(entries, output_root)
    _write_report(args.report.resolve(), results, run_results, sheets)
    return 0 if results and all(item["status"] == "PASS" for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
