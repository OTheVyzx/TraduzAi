"""Render a single-run vision debug sheet for problematic TraduzAi pages."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_FILTERS = {"P0", "fallback", "sfx", "inpaint", "typesetting", "glossary", "rerun"}


def render_vision_debug_sheet(
    output_dir: str | Path,
    out_path: str | Path,
    *,
    filters: list[str] | None = None,
    max_pages: int = 20,
) -> dict[str, Any]:
    output = Path(output_dir)
    project_path = output / "project.json"
    target = Path(out_path)
    if not project_path.exists():
        result = {"status": "BLOCK", "reasons": ["missing project.json"], "selected_pages": []}
        _write_summary(target, result)
        return result

    project = json.loads(project_path.read_text(encoding="utf-8"))
    active_filters = {str(item) for item in (filters or DEFAULT_FILTERS)}
    reports = _select_pages(project, active_filters, max_pages=max_pages)
    assets = target.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    rows = []
    for report in reports:
        page = report["page"]
        page_number = report["page_number"]
        original = _copy_asset(_resolve_page_image(output, page, "base", "originals", page_number), assets, f"original_{page_number:03d}")
        inpaint = _copy_asset(_resolve_page_image(output, page, "inpaint", "images", page_number), assets, f"inpaint_{page_number:03d}")
        translated = _copy_asset(_resolve_page_image(output, page, "rendered", "translated", page_number), assets, f"translated_{page_number:03d}")
        rows.append(_render_row(report, original, inpaint, translated))

    html_text = _render_html(rows, output, active_filters)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_text, encoding="utf-8")
    result = {
        "status": "PASS" if reports else "BLOCK",
        "reasons": ["debug sheet generated"] if reports else ["no matching pages"],
        "selected_pages": [report["page_number"] for report in reports],
        "output_html": str(target),
    }
    _write_summary(target, result)
    return result


def _select_pages(project: dict[str, Any], filters: set[str], *, max_pages: int) -> list[dict[str, Any]]:
    reports = []
    for index, page in enumerate(project.get("paginas") or [], start=1):
        page_number = int(page.get("numero") or index)
        issues = _page_issues(page, filters)
        if not issues:
            continue
        reports.append({"page": page, "page_number": page_number, "issues": issues})
    return reports[: max(1, int(max_pages))]


def _page_issues(page: dict[str, Any], filters: set[str]) -> list[dict[str, Any]]:
    issues = []
    for layer_index, layer in enumerate(page.get("text_layers") or page.get("textos") or [], start=1):
        flags = {str(flag) for flag in layer.get("qa_flags") or [] if flag}
        categories = set()
        if flags & {"translation_fallback_phrase", "literal_ocr_translation"}:
            categories.add("fallback")
        if flags & {"glossary_violation", "placeholder_lost", "forbidden_translation", "glossary_locked"}:
            categories.add("glossary")
        if flags & {"text_overflow", "text_overflow_high"}:
            categories.add("typesetting")
        if flags & {"inpaint_artifact", "outline_damage_high"}:
            categories.add("inpaint")
        if flags & {
            "sfx_candidate",
            "sfx_preserved",
            "sfx_render_missing",
            "sfx_render_outside_source_region",
            "sfx_inpaint_damaged_art_risk",
            "sfx_translation_unknown",
            "sfx_style_low_confidence",
            "sfx_visual_candidate",
            "sfx_script_unknown",
            "sfx_mask_density_high",
        } or layer.get("tipo") == "sfx":
            categories.add("sfx")
        if flags & {"source_script_leak", "vlm_failure_phrase", "speech_cjk_preserved_inside_balloon"}:
            categories.add("P0")
        if layer.get("ocr_second_pass") or flags & {"ocr_run_on_suspect", "partial_ocr"}:
            categories.add("rerun")
        if categories & filters:
            issues.append(
                {
                    "layer": layer.get("id") or f"t{layer_index}",
                    "categories": sorted(categories),
                    "flags": sorted(flags),
                    "original": layer.get("original") or layer.get("text") or "",
                    "translated": layer.get("translated") or layer.get("traduzido") or "",
                    "bbox": layer.get("bbox") or layer.get("layout_bbox") or layer.get("source_bbox"),
                    "glossary_hits": layer.get("glossary_hits") or [],
                }
            )
    return issues


def _resolve_page_image(output: Path, page: dict[str, Any], layer_key: str, fallback_dir: str, page_number: int) -> Path | None:
    image_layers = page.get("image_layers") or {}
    layer = image_layers.get(layer_key) if isinstance(image_layers, dict) else None
    if isinstance(layer, dict) and isinstance(layer.get("path"), str):
        candidate = output / layer["path"]
        if candidate.exists():
            return candidate
    for suffix in (".jpg", ".png", ".jpeg", ".webp"):
        candidate = output / fallback_dir / f"{page_number:03d}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _copy_asset(source: Path | None, assets_dir: Path, stem: str) -> str:
    if source is None or not source.exists():
        return ""
    target = assets_dir / f"{stem}{source.suffix.lower() or '.jpg'}"
    shutil.copy2(source, target)
    return f"assets/{target.name}"


def _render_row(report: dict[str, Any], original: str, inpaint: str, translated: str) -> str:
    issues_html = "".join(
        "<li>"
        f"<strong>{html.escape(str(issue['layer']))}</strong> "
        f"{html.escape(', '.join(issue['categories']))}: "
        f"{html.escape(str(issue.get('translated') or issue.get('original') or ''))}"
        f"<pre>{html.escape(json.dumps(issue, ensure_ascii=False, indent=2))}</pre>"
        "</li>"
        for issue in report["issues"]
    )
    return (
        f"<section><h2>Pagina {report['page_number']}</h2>"
        "<div class='grid'>"
        f"{_img_cell('Original', original)}"
        f"{_img_cell('Inpaint', inpaint)}"
        f"{_img_cell('Translated', translated)}"
        "</div>"
        f"<ul>{issues_html}</ul>"
        "</section>"
    )


def _img_cell(label: str, path: str) -> str:
    if not path:
        return f"<div><h3>{html.escape(label)}</h3><p>missing</p></div>"
    return f"<div><h3>{html.escape(label)}</h3><img src='{html.escape(path)}' alt='{html.escape(label)}'></div>"


def _render_html(rows: list[str], output: Path, filters: set[str]) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}"
        "section{border-top:1px solid #444;padding:18px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}"
        "img{max-width:100%;background:#fff}pre{white-space:pre-wrap;background:#1e1e27;padding:8px;border-radius:4px}</style>"
        f"<title>Vision Debug - {html.escape(str(output))}</title></head><body>"
        f"<h1>Vision Debug</h1><p>Filters: {html.escape(', '.join(sorted(filters)))}</p>"
        + "\n".join(rows)
        + "</body></html>"
    )


def _write_summary(out_path: Path, result: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.with_suffix(".summary.json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--filter", action="append", default=[])
    args = parser.parse_args(argv)
    result = render_vision_debug_sheet(args.output, args.out, filters=args.filter or None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
