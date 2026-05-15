"""Analyze a CJK TraduzAi run for the failure classes tracked in the CJK plan."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SOURCE_SCRIPT_RE = re.compile(r"[\u1100-\u11FF\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")
HANGUL_RE = re.compile(r"[\u1100-\u11FF\uAC00-\uD7AF]")

VLM_FAILURE_PATTERNS = (
    "image is too blurry",
    "recognize any text content",
    "cannot recognize any text",
    "unable to recognize text",
)

TRANSLATION_FALLBACK_PATTERNS = (
    "nao consigo encontrar o texto original",
    "não consigo encontrar o texto original",
    "i cannot find the original text",
    "cannot translate this source text",
)

INPAINT_FLAGS = {
    "inpaint_artifact",
    "outline_damage",
    "outline_damage_high",
    "residual_source_text",
    "source_script_leak",
}

TYPESETTING_FLAGS = {
    "text_overflow",
    "text_overflow_high",
    "text_clipped",
    "text_too_small",
    "text_top_aligned",
    "underwrapped",
}

SFX_FLAGS = {
    "cjk_sfx_preserved",
    "sfx_preserved",
    "speech_cjk_preserved_inside_balloon",
}


def analyze_cjk_quality_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    include_pages: list[int] | None = None,
    max_visual_pages: int = 24,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    out_path = Path(out_dir) if out_dir is not None else run_path / "cjk_quality_analysis"
    project_path = run_path / "project.json"
    qa_report_path = run_path / "qa_report.json"

    if not project_path.exists():
        result = {
            "status": "BLOCK",
            "run_dir": str(run_path),
            "reasons": ["missing project.json"],
            "selected_pages": [],
            "issue_counts": {},
            "pages": [],
        }
        _write_outputs(out_path, result, run_path, max_visual_pages=max_visual_pages)
        return result

    project = _load_json(project_path)
    qa_report = _load_json(qa_report_path) if qa_report_path.exists() else {}
    pages = _project_pages(project)
    reports: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    forced_pages = {int(page) for page in include_pages or []}

    for index, page in enumerate(pages, start=1):
        page_number = _page_number(page, index)
        issues = _page_issues(page)
        issues.extend(_qa_report_page_issues(qa_report, page_number))
        if not issues and page_number not in forced_pages:
            continue
        if not issues:
            issues.append({"type": "manual_review", "source": "include_pages"})
        for issue in issues:
            issue_counts[str(issue["type"])] += 1
        reports.append(
            {
                "page_number": page_number,
                "issue_count": len(issues),
                "issues": issues,
                "route_history": page.get("route_history") or page.get("_route_history") or [],
            }
        )

    result = {
        "status": "PASS",
        "run_dir": str(run_path),
        "project_path": str(project_path),
        "qa_report_path": str(qa_report_path) if qa_report_path.exists() else None,
        "page_count": len(pages),
        "selected_pages": [report["page_number"] for report in reports],
        "issue_counts": dict(sorted(issue_counts.items())),
        "pages": reports,
    }
    _write_outputs(out_path, result, run_path, max_visual_pages=max_visual_pages)
    return result


def _project_pages(project: dict[str, Any]) -> list[dict[str, Any]]:
    pages = project.get("paginas") or project.get("pages") or []
    return [page for page in pages if isinstance(page, dict)]


def _page_number(page: dict[str, Any], fallback: int) -> int:
    for key in ("numero", "number", "page_number", "index"):
        value = page.get(key)
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    return fallback


def _page_issues(page: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    layers = _page_text_layers(page)
    for layer_index, layer in enumerate(layers, start=1):
        flags = _layer_flags(layer)
        texts = _layer_text_values(layer)
        translated = str(layer.get("translated") or layer.get("traduzido") or layer.get("translation") or "")
        layer_type = str(layer.get("tipo") or layer.get("type") or "").lower()

        if flags & {"vlm_failure_phrase"} or any(_contains_any(text, VLM_FAILURE_PATTERNS) for text in texts):
            issues.append(_issue("vlm_failure_phrase", layer_index, layer, flags))

        if flags & {"translation_fallback_phrase", "translation_failed", "translation_render_blocked"} or _contains_any(
            translated,
            TRANSLATION_FALLBACK_PATTERNS,
        ):
            issues.append(_issue("translation_fallback_phrase", layer_index, layer, flags))

        if flags & SFX_FLAGS or str(layer.get("ignored_reason") or "") == "cjk_sfx_preserved":
            issues.append(_issue("cjk_sfx_preserved", layer_index, layer, flags))

        if _is_speech_layer(layer_type, flags) and HANGUL_RE.search(translated):
            issues.append(_issue("hangul_residual_in_speech", layer_index, layer, flags))

        if flags & INPAINT_FLAGS:
            issues.append(_issue("inpaint_geometry_or_residual", layer_index, layer, flags & INPAINT_FLAGS))

        typesetting_issue = _typesetting_issue(layer, flags)
        if typesetting_issue:
            issues.append(_issue(typesetting_issue, layer_index, layer, flags & TYPESETTING_FLAGS))

    return _dedupe_issues(issues)


def _page_text_layers(page: dict[str, Any]) -> list[dict[str, Any]]:
    layers = page.get("text_layers") or page.get("textos") or page.get("texts") or []
    return [layer for layer in layers if isinstance(layer, dict)]


def _layer_flags(layer: dict[str, Any]) -> set[str]:
    flags = {str(flag) for flag in layer.get("qa_flags") or [] if flag}
    ignored = layer.get("ignored_reason")
    if ignored:
        flags.add(str(ignored))
    return flags


def _layer_text_values(layer: dict[str, Any]) -> list[str]:
    values = []
    for key in ("text", "original", "raw_text", "translated", "traduzido", "translation"):
        value = layer.get(key)
        if value is not None:
            values.append(str(value))
    return values


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = str(text or "").strip().lower()
    return any(pattern in normalized for pattern in patterns)


def _is_speech_layer(layer_type: str, flags: set[str]) -> bool:
    if layer_type in {"sfx", "sound_effect", "onomatopoeia"}:
        return False
    if flags & {"sfx_preserved", "sfx_candidate"}:
        return False
    return True


def _typesetting_issue(layer: dict[str, Any], flags: set[str]) -> str | None:
    if flags & TYPESETTING_FLAGS:
        return "typesetting_layout"
    vertical_offset = _as_float(layer.get("vertical_offset_ratio"))
    if vertical_offset is not None and abs(vertical_offset) > 0.42:
        return "typesetting_layout"
    fill_ratio = _as_float(layer.get("fill_ratio"))
    if fill_ratio is not None and fill_ratio < 0.08:
        return "typesetting_layout"
    text_aspect = _as_float(layer.get("text_aspect_ratio"))
    if text_aspect is not None and text_aspect > 8.0:
        return "typesetting_layout"
    return None


def _qa_report_page_issues(qa_report: dict[str, Any], page_number: int) -> list[dict[str, Any]]:
    if not isinstance(qa_report, dict):
        return []
    candidates = qa_report.get("pages") or qa_report.get("paginas") or []
    issues: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if _page_number(item, -1) != page_number:
            continue
        for flag in item.get("qa_flags") or item.get("flags") or []:
            flag_text = str(flag)
            if flag_text in {"vlm_failure_phrase", "translation_fallback_phrase", "source_script_leak"}:
                issues.append({"type": flag_text, "source": "qa_report"})
    return issues


def _issue(issue_type: str, layer_index: int, layer: dict[str, Any], flags: set[str]) -> dict[str, Any]:
    return {
        "type": issue_type,
        "layer": layer.get("id") or layer.get("layer_id") or f"t{layer_index}",
        "flags": sorted(flags),
        "bbox": layer.get("bbox") or layer.get("layout_bbox") or layer.get("source_bbox"),
        "original": _short(layer.get("original") or layer.get("text") or layer.get("raw_text") or ""),
        "translated": _short(layer.get("translated") or layer.get("traduzido") or layer.get("translation") or ""),
    }


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for issue in issues:
        key = (issue.get("type"), issue.get("layer"), tuple(issue.get("flags") or []))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _short(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(out_dir: Path, result: dict[str, Any], run_dir: Path, *, max_visual_pages: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_visual_sheet(out_dir / "visual_sheet.html", result, run_dir, max_pages=max_visual_pages)


def _write_visual_sheet(target: Path, result: dict[str, Any], run_dir: Path, *, max_pages: int) -> None:
    rows = []
    for report in result.get("pages", [])[: max(1, int(max_pages))]:
        page_number = int(report["page_number"])
        rows.append(_visual_row(run_dir, page_number, report))
    html_text = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}"
        "section{border-top:1px solid #444;padding:18px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}"
        "img{max-width:100%;background:#fff}pre{white-space:pre-wrap;background:#1e1e27;padding:8px;border-radius:4px}</style>"
        "<title>CJK Quality Analysis</title></head><body>"
        f"<h1>CJK Quality Analysis</h1><p>{html.escape(str(run_dir))}</p>"
        + "\n".join(rows)
        + "</body></html>"
    )
    target.write_text(html_text, encoding="utf-8")


def _visual_row(run_dir: Path, page_number: int, report: dict[str, Any]) -> str:
    issues = html.escape(json.dumps(report.get("issues") or [], ensure_ascii=False, indent=2))
    return (
        f"<section><h2>Page {page_number:03d}</h2><div class='grid'>"
        f"{_image_cell('Original', _find_image(run_dir, 'originals', page_number))}"
        f"{_image_cell('Inpaint', _find_image(run_dir, 'images', page_number))}"
        f"{_image_cell('Translated', _find_image(run_dir, 'translated', page_number))}"
        f"</div><pre>{issues}</pre></section>"
    )


def _find_image(run_dir: Path, folder: str, page_number: int) -> str:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = run_dir / folder / f"{page_number:03d}{suffix}"
        if candidate.exists():
            return candidate.resolve().as_uri()
    return ""


def _image_cell(label: str, uri: str) -> str:
    if not uri:
        return f"<div><h3>{html.escape(label)}</h3><p>missing</p></div>"
    return f"<div><h3>{html.escape(label)}</h3><img src='{html.escape(uri)}' alt='{html.escape(label)}'></div>"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--include-page", action="append", type=int, default=[])
    parser.add_argument("--max-visual-pages", type=int, default=24)
    args = parser.parse_args(argv)
    result = analyze_cjk_quality_run(
        args.run_dir,
        args.out_dir,
        include_pages=args.include_page,
        max_visual_pages=args.max_visual_pages,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
