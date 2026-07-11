"""Generate a visual style-audit report from an existing TraduzAI run.

This is a debug helper: it reads ``project.json`` and original page images,
runs the lightweight text style extractor on each text layer crop, and writes
JSONL plus contact-sheet images for manual inspection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from typesetter.style_extractor import extract_text_style_evidence
from typesetter.style_contract import style_evidence_v2_from_v1
from typesetter.style_policy import style_evidence_v2_shadow_policy


CARD_W = 360
CARD_H = 250
CROP_H = 155
MARGIN = 14
FONT = cv2.FONT_HERSHEY_SIMPLEX
STYLE_COPY_CANDIDATE_CONFIDENCE_THRESHOLD = 0.70
STYLE_COPY_SFX_PROMOTION_THRESHOLD = 0.66


def _bbox4(value: object) -> list[int] | None:
    if not isinstance(value, list | tuple) or len(value) < 4:
        return None
    try:
        return [int(round(float(v))) for v in value[:4]]
    except (TypeError, ValueError):
        return None


def _layer_style(layer: dict) -> dict:
    style = layer.get("style") if isinstance(layer.get("style"), dict) else layer.get("estilo")
    return style if isinstance(style, dict) else {}


def _non_empty_gradient(value: object) -> bool:
    return isinstance(value, list | tuple) and len(value) >= 2 and all(str(item).strip() for item in value[:2])


def _applied_style_fields(layer: dict) -> dict:
    style = _layer_style(layer)
    return {
        "style_origin": str(layer.get("style_origin") or style.get("style_origin") or ""),
        "style_confidence": float(layer.get("style_confidence") or style.get("style_confidence") or 0.0),
        "style_source": str(layer.get("style_source") or style.get("style_source") or ""),
        "render_policy": layer.get("render_policy"),
        "route_action": layer.get("route_action"),
        "content_class": layer.get("content_class"),
        "applied_font_name": str(style.get("fonte") or ""),
        "applied_text_color": str(style.get("cor") or ""),
        "applied_stroke_color": str(style.get("contorno") or ""),
        "applied_stroke_width_px": int(style.get("contorno_px") or 0),
        "applied_gradient": _non_empty_gradient(style.get("cor_gradiente")),
        "applied_gradient_colors": list(style.get("cor_gradiente") or [])[:2]
        if _non_empty_gradient(style.get("cor_gradiente"))
        else [],
        "applied_glow": bool(style.get("glow")),
        "applied_glow_color": str(style.get("glow_cor") or ""),
        "applied_glow_px": int(style.get("glow_px") or 0),
        "applied_shadow": bool(style.get("sombra")),
        "applied_shadow_color": str(style.get("sombra_cor") or ""),
        "applied_shadow_offset": list(style.get("sombra_offset") or [0, 0])[:2],
    }


def _has_applied_style_effect(fields: dict) -> bool:
    return bool(
        fields.get("applied_stroke_color")
        or fields.get("applied_glow")
        or fields.get("applied_shadow")
        or fields.get("applied_gradient")
    )


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_confidence_from_fields(layer: dict, fields: tuple[str, ...]) -> float | None:
    for field in fields:
        value = _float_or_none(layer.get(field))
        if value is not None:
            return value
    return None


def _primary_text_style_candidate_confident(layer: dict) -> bool:
    confidence = _candidate_confidence_from_fields(
        layer,
        ("confidence", "ocr_confidence", "confianca_ocr"),
    )
    if confidence is None:
        return True
    return confidence >= STYLE_COPY_CANDIDATE_CONFIDENCE_THRESHOLD


def _sfx_style_candidate_confident(layer: dict) -> bool:
    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    sfx_ocr = layer.get("sfx_ocr") if isinstance(layer.get("sfx_ocr"), dict) else {}
    confidence_values = [
        _float_or_none(layer.get("sfx_promotion_score")),
        _float_or_none(sfx.get("promotion_score")),
        _float_or_none(layer.get("confidence")),
        _float_or_none(layer.get("ocr_confidence")),
        _float_or_none(sfx.get("visual_confidence")),
        _float_or_none(sfx_ocr.get("confidence")),
        _float_or_none(sfx_ocr.get("ocr_confidence")),
    ]
    confidence_values = [value for value in confidence_values if value is not None]
    if not confidence_values:
        return True
    promotion_score = _float_or_none(layer.get("sfx_promotion_score"))
    if promotion_score is None:
        promotion_score = _float_or_none(sfx.get("promotion_score"))
    if promotion_score is not None and promotion_score >= STYLE_COPY_SFX_PROMOTION_THRESHOLD:
        return True
    return max(confidence_values) >= STYLE_COPY_CANDIDATE_CONFIDENCE_THRESHOLD


def _style_scan_allowed_for_layer(layer: dict, bbox: list[int]) -> bool:
    applied = _applied_style_fields(layer)
    if _has_applied_style_effect(applied):
        return True
    if str(applied.get("style_origin") or "").strip().lower() == "source_detected":
        return True

    content_class = str(layer.get("content_class") or "").strip().lower()
    route_action = str(layer.get("route_action") or "").strip().lower()
    render_policy = str(layer.get("render_policy") or "").strip().lower()
    detector = str(layer.get("detector") or "").strip().lower()

    if route_action == "translate_sfx_inpaint_render" and render_policy != "review_required":
        return _sfx_style_candidate_confident(layer)
    if route_action == "review_required" or render_policy == "review_required":
        return False
    if content_class == "sfx" and detector == "sfx_visual":
        return False
    if content_class and content_class != "sfx":
        return _primary_text_style_candidate_confident(layer)
    if route_action == "translate_inpaint_render":
        return _primary_text_style_candidate_confident(layer)
    text = str(layer.get("text") or layer.get("original") or layer.get("raw_ocr") or "").strip()
    return bool(text) and bool(bbox) and _primary_text_style_candidate_confident(layer)


def _style_scan_skip_reason(layer: dict) -> str:
    route_action = str(layer.get("route_action") or "").strip().lower()
    content_class = str(layer.get("content_class") or "").strip().lower()
    if route_action == "translate_sfx_inpaint_render" or content_class == "sfx":
        if not _sfx_style_candidate_confident(layer):
            return "low_candidate_confidence"
    elif not _primary_text_style_candidate_confident(layer):
        return "low_candidate_confidence"
    return "not_style_copy_candidate"


def _empty_style_evidence(*, skipped: bool = False, reason: str = "") -> dict:
    return {
        "style_scan_skipped": bool(skipped),
        "style_scan_skip_reason": reason,
        "source": None,
        "text_color": "",
        "text_color_confidence": 0.0,
        "stroke_color": "",
        "stroke_width_px": 0,
        "stroke_confidence": 0.0,
        "gradient": False,
        "gradient_colors": [],
        "gradient_confidence": 0.0,
        "shadow": False,
        "shadow_color": "",
        "shadow_offset": [0, 0],
        "shadow_confidence": 0.0,
        "glow": False,
        "glow_color": "",
        "glow_px": 0,
        "glow_confidence": 0.0,
        "curved": False,
        "curve_direction": "",
        "curve_amount": 0.0,
        "curve_confidence": 0.0,
        "font_name": "",
        "font_confidence": 0.0,
    }


def _read_project_records(run_dir: Path, originals_dir: Path) -> list[dict]:
    project_path = run_dir / "project.json"
    data = json.loads(project_path.read_text(encoding="utf-8"))
    pages = data.get("paginas") or data.get("pages") or []
    records: list[dict] = []

    for page_index, page in enumerate(pages, start=1):
        image_path = originals_dir / f"{page_index:03d}.jpg"
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]

        for layer in page.get("text_layers") or []:
            bbox = _bbox4(layer.get("text_pixel_bbox") or layer.get("bbox") or layer.get("balloon_bbox"))
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            pad = 2
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue

            applied_fields = _applied_style_fields(layer)
            if _style_scan_allowed_for_layer(layer, [x1, y1, x2, y2]):
                evidence = extract_text_style_evidence(img_rgb[y1:y2, x1:x2, :3]).to_dict()
                evidence["style_scan_skipped"] = False
                evidence["style_scan_skip_reason"] = ""
            else:
                evidence = _empty_style_evidence(
                    skipped=True,
                    reason=_style_scan_skip_reason(layer),
                )
            evidence_v2 = style_evidence_v2_from_v1(evidence)
            records.append(
                {
                    "page": page_index,
                    "id": layer.get("id") or layer.get("text_id"),
                    "tipo": layer.get("tipo"),
                    "text": str(layer.get("text") or "")[:120],
                    "bbox": [x1, y1, x2, y2],
                    **applied_fields,
                    **evidence,
                    "style_evidence_v2": evidence_v2.to_dict(),
                    "style_evidence_v2_shadow_policy": style_evidence_v2_shadow_policy(evidence_v2),
                }
            )
    return records


def _read_crop(rec: dict, originals_dir: Path) -> np.ndarray:
    img = cv2.imread(str(originals_dir / f"{int(rec['page']):03d}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        return np.full((80, 160, 3), 245, np.uint8)
    x1, y1, x2, y2 = [int(v) for v in rec["bbox"]]
    height, width = img.shape[:2]
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.full((80, 160, 3), 245, np.uint8)
    return img[y1:y2, x1:x2]


def _fit_crop(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    scale = min((CARD_W - 20) / max(1, width), (CROP_H - 14) / max(1, height), 3.0)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((CROP_H, CARD_W, 3), 250, dtype=np.uint8)
    x = (CARD_W - new_w) // 2
    y = (CROP_H - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    cv2.rectangle(canvas, (x, y), (x + new_w - 1, y + new_h - 1), (180, 180, 180), 1)
    return canvas


def _make_card(rec: dict, originals_dir: Path) -> np.ndarray:
    card = np.full((CARD_H, CARD_W, 3), 255, dtype=np.uint8)
    card[:CROP_H] = _fit_crop(_read_crop(rec, originals_dir))

    effects: list[str] = []
    if rec.get("style_scan_skipped"):
        effects.append("skipped")
    if rec.get("stroke_color"):
        effects.append(f"stroke {rec.get('stroke_color')}:{rec.get('stroke_width_px')}")
    if rec.get("shadow"):
        effects.append(f"shadow {rec.get('shadow_color')} {rec.get('shadow_offset')}")
    if rec.get("glow"):
        effects.append(f"glow {rec.get('glow_color')}:{rec.get('glow_px')}")
    if rec.get("gradient"):
        effects.append(f"grad {rec.get('gradient_colors')}")
    if rec.get("curved"):
        effects.append(
            f"curve {rec.get('curve_direction')}:{float(rec.get('curve_amount') or 0.0):.2f}"
        )
    if not effects:
        effects.append("no fx")
    applied_effects: list[str] = []
    if rec.get("applied_stroke_color"):
        applied_effects.append(f"stroke {rec.get('applied_stroke_color')}:{rec.get('applied_stroke_width_px')}")
    if rec.get("applied_shadow"):
        applied_effects.append(f"shadow {rec.get('applied_shadow_color')} {rec.get('applied_shadow_offset')}")
    if rec.get("applied_glow"):
        applied_effects.append(f"glow {rec.get('applied_glow_color')}:{rec.get('applied_glow_px')}")
    if rec.get("applied_gradient"):
        applied_effects.append(f"grad {rec.get('applied_gradient_colors')}")
    if not applied_effects:
        applied_effects.append("no applied fx")

    lines = [
        f"p{int(rec['page']):02d} {rec.get('id') or ''} {rec.get('tipo') or ''}",
        f"fill {rec.get('text_color') or '-'} conf {float(rec.get('text_color_confidence') or 0):.2f}",
        "det " + " | ".join(effects),
        "app " + " | ".join(applied_effects),
        f"font {rec.get('font_name') or '-'} {float(rec.get('font_confidence') or 0):.2f}",
        str(rec.get("text") or "")[:44],
    ]
    y = CROP_H + 18
    for i, line in enumerate(lines[:5]):
        color = (20, 20, 20) if i < 3 else (80, 80, 80)
        cv2.putText(card, line[:48], (10, y), FONT, 0.42, color, 1, cv2.LINE_AA)
        y += 18
    return card


def _contact_sheet(
    output_path: Path,
    records: Iterable[dict],
    originals_dir: Path,
    *,
    cols: int = 3,
) -> None:
    subset = list(records)
    rows = max(1, (len(subset) + cols - 1) // cols)
    sheet = np.full(
        (rows * CARD_H + (rows + 1) * MARGIN, cols * CARD_W + (cols + 1) * MARGIN, 3),
        236,
        dtype=np.uint8,
    )
    for idx, rec in enumerate(subset):
        row, col = divmod(idx, cols)
        y = MARGIN + row * (CARD_H + MARGIN)
        x = MARGIN + col * (CARD_W + MARGIN)
        sheet[y : y + CARD_H, x : x + CARD_W] = _make_card(rec, originals_dir)
    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def _write_visual_report(records: list[dict], run_dir: Path, originals_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "style_audit_records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    high_stroke = [
        rec
        for rec in records
        if rec.get("stroke_color") and float(rec.get("stroke_confidence") or 0) >= 0.7
    ]
    effects = [rec for rec in records if rec.get("glow") or rec.get("shadow")]
    gradients = [rec for rec in records if rec.get("gradient")]
    curved = [rec for rec in records if rec.get("curved")]
    impact = [rec for rec in records if rec.get("font_name") == "KOMIKAX_.ttf"]
    common_stroke = [
        rec
        for rec in records
        if rec.get("font_name") == "ComicNeue-Bold.ttf"
        and rec.get("stroke_color")
        and float(rec.get("stroke_confidence") or 0) >= 0.7
    ]
    no_fx = [
        rec
        for rec in records
        if not rec.get("stroke_color") and not rec.get("glow") and not rec.get("shadow")
    ]
    applied_effects = [
        rec
        for rec in records
        if rec.get("applied_stroke_color")
        or rec.get("applied_glow")
        or rec.get("applied_shadow")
        or rec.get("applied_gradient")
    ]
    detected_not_applied = [
        rec
        for rec in records
        if (
            rec.get("stroke_color")
            or rec.get("glow")
            or rec.get("shadow")
            or rec.get("gradient")
        )
        and not (
            rec.get("applied_stroke_color")
            or rec.get("applied_glow")
            or rec.get("applied_shadow")
            or rec.get("applied_gradient")
        )
    ]
    skipped_scan = [rec for rec in records if rec.get("style_scan_skipped")]

    sheets = {
        "01_high_conf_strokes.jpg": high_stroke[:45],
        "02_effects_glow_shadow.jpg": effects[:30],
        "03_impact_font_candidates.jpg": impact[:45],
        "04_common_font_with_stroke.jpg": common_stroke[:45],
        "05_no_effect_baseline.jpg": no_fx[:30],
        "06_gradients.jpg": gradients[:45],
        "07_curved.jpg": curved[:45],
        "08_applied_effects.jpg": applied_effects[:45],
        "09_detected_not_applied.jpg": detected_not_applied[:45],
        "10_style_scan_skipped.jpg": skipped_scan[:45],
    }
    for filename, subset in sheets.items():
        _contact_sheet(output_dir / filename, subset, originals_dir)

    counts = Counter()
    for rec in records:
        counts["layers"] += 1
        if rec.get("style_scan_skipped"):
            counts["style_scan_skipped"] += 1
        if rec.get("stroke_color"):
            counts["stroke"] += 1
        if rec.get("shadow"):
            counts["shadow"] += 1
        if rec.get("glow"):
            counts["glow"] += 1
        if rec.get("gradient"):
            counts["gradient"] += 1
        if rec.get("curved"):
            counts["curved"] += 1
        if rec.get("font_name") == "ComicNeue-Bold.ttf":
            counts["font_default"] += 1
        elif rec.get("font_name"):
            counts["font_other"] += 1
        if rec.get("applied_stroke_color"):
            counts["applied_stroke"] += 1
        if rec.get("applied_shadow"):
            counts["applied_shadow"] += 1
        if rec.get("applied_glow"):
            counts["applied_glow"] += 1
        if rec.get("applied_gradient"):
            counts["applied_gradient"] += 1

    summary = {
        "run": str(run_dir),
        "output_dir": str(output_dir),
        "records": len(records),
        "counts": dict(counts),
        "fonts": Counter(rec.get("font_name") or "" for rec in records).most_common(10),
        "strokes": Counter(rec.get("stroke_color") or "" for rec in records).most_common(10),
        "sheets": list(sheets.keys()),
        "records_file": str(records_path),
    }
    (output_dir / "style_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    html = [
        "<!doctype html><meta charset=\"utf-8\"><title>TraduzAI Style Audit</title>",
        "<style>body{font-family:Arial,sans-serif;background:#111;color:#eee;margin:24px} img{max-width:100%;border:1px solid #444;margin:12px 0 32px} code{color:#9fe}</style>",
        "<h1>TraduzAI Style Audit Visual Report</h1>",
        f"<p>Run: <code>{run_dir}</code></p>",
        f"<p>Total: {len(records)}. Stroke: {counts['stroke']}. Glow: {counts['glow']}. Shadow: {counts['shadow']}. Gradient: {counts['gradient']}. Curved: {counts['curved']}. Default font: {counts['font_default']}.</p>",
        f"<p>Applied: Stroke: {counts['applied_stroke']}. Glow: {counts['applied_glow']}. Shadow: {counts['applied_shadow']}. Gradient: {counts['applied_gradient']}.</p>",
    ]
    for filename in sheets:
        html.append(f"<h2>{Path(filename).stem}</h2><img src=\"{filename}\" alt=\"{filename}\">")
    (output_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Existing run directory containing project.json")
    parser.add_argument("--originals", type=Path, help="Directory with original page images")
    parser.add_argument("--output", type=Path, help="Output directory for report")
    args = parser.parse_args()

    run_dir = args.run
    originals_dir = args.originals or (run_dir / "originals")
    output_dir = args.output or (run_dir / "debug" / "codex_style_audit" / "visual_report")
    records = _read_project_records(run_dir, originals_dir)
    summary = _write_visual_report(records, run_dir, originals_dir, output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
