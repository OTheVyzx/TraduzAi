"""Analyze a CJK TraduzAi run for the failure classes tracked in the CJK plan."""

from __future__ import annotations

import argparse
import contextlib
import html
import io
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
    "sfx_render_missing",
    "sfx_render_outside_source_region",
    "sfx_inpaint_damaged_art_risk",
    "sfx_translation_unknown",
    "sfx_style_low_confidence",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


def analyze_sfx_benchmark(benchmark_dir: str | Path) -> dict[str, Any]:
    """Summarize optional local manhwa SFX benchmark assets without requiring them."""

    benchmark_path = Path(benchmark_dir)
    visual_path = benchmark_path / "sfx_benchmark_visual"
    if not benchmark_path.exists():
        return {
            "status": "SKIP",
            "benchmark": "sfx_manhwa",
            "path": str(benchmark_path),
            "reason": "folder_not_found",
            "items": [],
        }

    image_paths = sorted(
        path
        for path in benchmark_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        and not _is_generated_sfx_benchmark_artifact(path, benchmark_path)
    )
    items: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    mask_reject_counts: Counter[str] = Counter()
    gate_counts: Counter[str] = Counter()
    expected_count = 0
    detected_count = 0
    matched_count = 0
    false_positive_count = 0
    missed_count = 0
    ious: list[float] = []
    for image_path in image_paths:
        manifest = _load_optional_manifest(image_path)
        detections = _detect_sfx_benchmark_candidates(image_path, visual_path)
        detected_count += len(detections)
        expected = [item for item in manifest.get("expected_sfx") or [] if isinstance(item, dict)]
        expected_count += len(expected)
        matches = _match_expected_sfx(expected, detections)
        overlay_path = _write_sfx_benchmark_overlay(image_path, expected, detections, matches, visual_path)
        ocr_report_path = _write_sfx_ocr_candidate_report(image_path, detections, visual_path)
        matched_count += len(matches["matches"])
        false_positive_count += len(matches["false_positives"])
        missed_count += len(matches["missed"])
        ious.extend(float(match["iou"]) for match in matches["matches"])
        route = str(manifest.get("expected_route") or manifest.get("route") or "unlabeled")
        mask_status = str(manifest.get("mask_status") or "unmeasured")
        gate_status = str(manifest.get("gate_status") or "unmeasured")
        route_counts[route] += 1
        mask_reject_counts[mask_status] += 1
        gate_counts[gate_status] += 1
        items.append(
            {
                "path": str(image_path),
                "manifest": str(_manifest_path_for_image(image_path)) if _manifest_path_for_image(image_path).exists() else None,
                "route": route,
                "mask_status": mask_status,
                "gate_status": gate_status,
                "expected_count": len(expected),
                "detected_count": len(detections),
                "matched_count": len(matches["matches"]),
                "missed_count": len(matches["missed"]),
                "false_positive_count": len(matches["false_positives"]),
                "overlay_path": str(overlay_path) if overlay_path is not None else None,
                "ocr_report_path": str(ocr_report_path) if ocr_report_path is not None else None,
                "detections": detections,
                "matches": matches["matches"],
            }
        )

    return {
        "status": "PASS",
        "benchmark": "sfx_manhwa",
        "path": str(benchmark_path),
        "visual_dir": str(visual_path),
        "item_count": len(image_paths),
        "processed_count": len(items),
        "route_counts": dict(sorted(route_counts.items())),
        "mask_reject_counts": dict(sorted(mask_reject_counts.items())),
        "gate_counts": dict(sorted(gate_counts.items())),
        "expected_count": expected_count,
        "detected_count": detected_count,
        "matched_count": matched_count,
        "missed_count": missed_count,
        "false_positive_count": false_positive_count,
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
        "items": items,
    }


def _project_pages(project: dict[str, Any]) -> list[dict[str, Any]]:
    pages = project.get("paginas") or project.get("pages") or []
    return [page for page in pages if isinstance(page, dict)]


def _manifest_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def _is_generated_sfx_benchmark_artifact(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except Exception:
        rel_parts = path.parts
    if any(str(part).startswith(("manual_probe", "sfx_benchmark_visual")) for part in rel_parts[:-1]):
        return True
    stem = path.stem.lower()
    return stem.endswith(("_crop", "_mask", "_render"))


def _load_optional_manifest(image_path: Path) -> dict[str, Any]:
    manifest_path = _manifest_path_for_image(image_path)
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {"manifest_error": "invalid_json"}
    return payload if isinstance(payload, dict) else {"manifest_error": "not_object"}


def _detect_sfx_benchmark_candidates(image_path: Path, visual_dir: Path | None = None) -> list[dict[str, Any]]:
    try:
        import cv2
        from sfx.ocr_probe import probe_sfx_candidate_ocr
        from sfx.script_probe import probe_sfx_candidate_script
        from vision_stack.sfx_detector import detect_sfx_candidates
    except Exception:
        return []
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return []
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    candidates = detect_sfx_candidates(image_rgb)
    crop_dir = None
    if visual_dir is not None:
        crop_dir = visual_dir / f"{image_path.stem}_sfx_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
    detections: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            probed = probe_sfx_candidate_ocr(candidate, image_rgb)
        routed = probe_sfx_candidate_script(probed, str(probed.get("recognized_text") or ""))
        crop_path = _write_sfx_candidate_crop(image_rgb, candidate.get("bbox"), crop_dir, index) if crop_dir else None
        detections.append(
            {
            "bbox": candidate.get("bbox"),
            "confidence": candidate.get("confidence"),
            "detector": candidate.get("detector"),
            "route_action": routed.get("route_action"),
            "recognized_text": routed.get("recognized_text") or routed.get("text") or "",
            "ocr_confidence": probed.get("ocr_confidence"),
            "script": routed.get("script"),
            "visual_source": (candidate.get("sfx") or {}).get("visual_source"),
            "crop_path": str(crop_path) if crop_path is not None else None,
            "sfx_ocr": probed.get("sfx_ocr"),
        }
        )
    return detections


def _write_sfx_candidate_crop(
    image_rgb: Any,
    bbox: Any,
    crop_dir: Path,
    index: int,
) -> Path | None:
    try:
        import cv2
        from sfx.ocr_probe import build_sfx_ocr_crop
    except Exception:
        return None
    candidate = {"bbox": bbox}
    crop_rgb = build_sfx_ocr_crop(candidate, image_rgb)
    if not hasattr(crop_rgb, "size") or crop_rgb.size == 0:
        return None
    out_path = crop_dir / f"detection_{index:03d}_crop.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    return out_path


def _match_expected_sfx(expected: list[dict[str, Any]], detections: list[dict[str, Any]], *, threshold: float = 0.35) -> dict[str, Any]:
    unmatched = set(range(len(detections)))
    matches: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    for expected_index, item in enumerate(expected):
        expected_bbox = item.get("bbox")
        best: tuple[float, int] | None = None
        for detection_index in list(unmatched):
            iou = _bbox_iou(expected_bbox, detections[detection_index].get("bbox"))
            if best is None or iou > best[0]:
                best = (iou, detection_index)
        if best is not None and best[0] >= threshold:
            iou, detection_index = best
            unmatched.remove(detection_index)
            matches.append(
                {
                    "expected_index": expected_index,
                    "detection_index": detection_index,
                    "iou": round(float(iou), 4),
                    "expected_bbox": item.get("bbox"),
                    "detected_bbox": detections[detection_index].get("bbox"),
                    "label": item.get("label"),
                }
            )
        else:
            missed.append(item)
    return {
        "matches": matches,
        "missed": missed,
        "false_positives": [detections[index] for index in sorted(unmatched)],
    }


def _write_sfx_benchmark_overlay(
    image_path: Path,
    expected: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    matches: dict[str, Any],
    out_dir: Path,
) -> Path | None:
    try:
        import cv2
    except Exception:
        return None

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    overlay = image_bgr.copy()
    canvas = overlay.copy()

    matched_detection_indexes = {
        int(match["detection_index"])
        for match in matches.get("matches") or []
        if isinstance(match, dict) and match.get("detection_index") is not None
    }
    matched_expected_indexes = {
        int(match["expected_index"])
        for match in matches.get("matches") or []
        if isinstance(match, dict) and match.get("expected_index") is not None
    }

    for index, item in enumerate(expected):
        if index in matched_expected_indexes:
            continue
        _draw_labeled_bbox(canvas, item.get("bbox"), f"E{index + 1}:{item.get('label') or 'sfx'}", (40, 40, 245))

    for index, detection in enumerate(detections):
        color = (60, 220, 60) if index in matched_detection_indexes else (0, 165, 255)
        confidence = detection.get("confidence")
        label = f"D{index + 1}"
        if confidence is not None:
            label += f" {float(confidence):.2f}"
        _draw_labeled_bbox(canvas, detection.get("bbox"), label, color, thickness=2)

    legend_rows = [
        ("green = match", (60, 220, 60)),
        ("red = missed expected only", (40, 40, 245)),
        ("orange = false positive", (0, 165, 255)),
    ]
    cv2.rectangle(canvas, (8, 8), (270, 88), (20, 20, 20), -1)
    for row, (text, color) in enumerate(legend_rows):
        y = 30 + row * 22
        cv2.rectangle(canvas, (18, y - 12), (34, y + 3), color, -1)
        cv2.putText(canvas, text, (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)

    overlay = cv2.addWeighted(canvas, 0.92, overlay, 0.08, 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_sfx_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    return out_path


def _write_sfx_ocr_candidate_report(
    image_path: Path,
    detections: list[dict[str, Any]],
    out_dir: Path,
) -> Path | None:
    if not detections:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, detection in enumerate(detections, start=1):
        crop_path = Path(str(detection.get("crop_path") or ""))
        crop_uri = crop_path.resolve().as_uri() if crop_path.exists() else ""
        sfx_ocr = detection.get("sfx_ocr") if isinstance(detection.get("sfx_ocr"), dict) else {}
        attempts = sfx_ocr.get("attempts") if isinstance(sfx_ocr.get("attempts"), list) else []
        attempt_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(str(attempt.get('lang') or ''))}</td>"
            f"<td>{html.escape(str(attempt.get('text') or ''))}</td>"
            f"<td>{html.escape(str(attempt.get('confidence') if attempt.get('confidence') is not None else ''))}</td>"
            f"<td>{html.escape(str(attempt.get('status') or ''))}</td>"
            f"<td>{html.escape(str(attempt.get('error') or ''))}</td>"
            "</tr>"
            for attempt in attempts
            if isinstance(attempt, dict)
        )
        if not attempt_rows:
            attempt_rows = "<tr><td colspan='5'>sem tentativas registradas</td></tr>"
        crop_html = f"<img src='{html.escape(crop_uri)}' alt='crop D{index}'>" if crop_uri else "<p>crop ausente</p>"
        rows.append(
            "<section>"
            f"<h2>D{index:02d}</h2>"
            "<div class='card'>"
            f"<div>{crop_html}</div>"
            "<dl>"
            f"<dt>bbox</dt><dd>{html.escape(json.dumps(detection.get('bbox') or [], ensure_ascii=False))}</dd>"
            f"<dt>visual</dt><dd>{html.escape(str(detection.get('confidence')))} / {html.escape(str(detection.get('visual_source') or ''))}</dd>"
            f"<dt>OCR status</dt><dd>{html.escape(str(sfx_ocr.get('status') or ''))}</dd>"
            f"<dt>texto</dt><dd>{html.escape(str(detection.get('recognized_text') or ''))}</dd>"
            f"<dt>confiança OCR</dt><dd>{html.escape(str(detection.get('ocr_confidence') or ''))}</dd>"
            f"<dt>script</dt><dd>{html.escape(str(detection.get('script') or ''))}</dd>"
            f"<dt>rota</dt><dd>{html.escape(str(detection.get('route_action') or ''))}</dd>"
            "</dl>"
            "</div>"
            "<table><thead><tr><th>lang</th><th>texto</th><th>conf</th><th>status</th><th>erro</th></tr></thead>"
            f"<tbody>{attempt_rows}</tbody></table>"
            "</section>"
        )
    html_text = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}"
        "section{border-top:1px solid #444;padding:18px 0}.card{display:grid;grid-template-columns:minmax(120px,260px) 1fr;gap:16px;align-items:start}"
        "img{max-width:100%;background:#fff;border:1px solid #555}dt{font-weight:bold;color:#bbb}dd{margin:0 0 8px 0}"
        "table{width:100%;border-collapse:collapse;margin-top:12px}td,th{border:1px solid #444;padding:6px;text-align:left;vertical-align:top}"
        "th{background:#222}</style>"
        f"<title>SFX OCR Report - {html.escape(image_path.name)}</title></head><body>"
        f"<h1>SFX OCR Report</h1><p>{html.escape(str(image_path))}</p>"
        + "\n".join(rows)
        + "</body></html>"
    )
    report_path = out_dir / f"{image_path.stem}_sfx_ocr_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def _draw_labeled_bbox(
    image: Any,
    bbox: Any,
    label: str,
    color: tuple[int, int, int],
    *,
    thickness: int = 3,
) -> None:
    try:
        import cv2
    except Exception:
        return
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    except Exception:
        return
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    label_y = max(18, y1 - 6)
    cv2.putText(image, str(label), (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def _bbox_iou(a: Any, b: Any) -> float:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) < 4 or len(b) < 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    except Exception:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1.0, a_area + b_area - inter)


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
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--include-page", action="append", type=int, default=[])
    parser.add_argument("--max-visual-pages", type=int, default=24)
    parser.add_argument("--sfx-benchmark", type=Path)
    args = parser.parse_args(argv)
    if args.sfx_benchmark is not None:
        result = analyze_sfx_benchmark(args.sfx_benchmark)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.run_dir is None:
        parser.error("--run-dir is required unless --sfx-benchmark is used")
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
