"""
Builds expanded mask regions from OCR boxes.
This is the first step toward page-level inpainting that respects balloons
instead of removing each OCR box in isolation.
"""

from __future__ import annotations

from collections import Counter
import os
import re

import cv2
import numpy as np

try:
    from ocr.text_router import ROUTE_ACTIONS, route_action_requires_inpaint
except ImportError:
    from ..ocr.text_router import ROUTE_ACTIONS, route_action_requires_inpaint

try:
    from inpainter.notanother_adapter import build_notanother_text_mask
except ImportError:
    from .notanother_adapter import build_notanother_text_mask

try:
    from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask
except ImportError:
    from ..vision_stack.bubble_shape_refiner import refine_bubble_shape_mask

OVERREACH_RATIO_FLAG = 2.5
OVERREACH_RATIO_CRITICAL = 4.0
MASK_DENSITY_HIGH = 0.12
MASK_OUTSIDE_BALLOON_FLAG = 0.08
MASK_OUTSIDE_BALLOON_CRITICAL = 0.18
DARK_PANEL_RECT_MAX_HALF_WIDTH_FROM_TEXT_CENTER = 116
DARK_PANEL_RECT_MAX_HALF_HEIGHT_FROM_TEXT_CENTER = 64
SPECIAL_MASK_CLASSES = {
    "sfx",
    "sound_effect",
    "watermark",
    "scanlator_credit",
    "tn_note",
    "noise",
}
HANGUL_PATTERN = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
TEXTURED_STATUS_PANEL_TERMS = {
    "BEAST",
    "COLLECTION",
    "CONFIRM",
    "ERROR",
    "LEVEL",
    "MISSION",
    "NEWS",
    "POWER",
    "PROMOTION",
    "RECORD",
    "RECORDS",
    "REQUIRED",
    "SOURCE",
    "STATUS",
    "SYSTEM",
    "TAMED",
    "TAMEABLE",
    "TRIAL",
}
FAST_FILL_MASK_EVIDENCE_KINDS = {
    "ocr_pixels",
    "glyph_segmentation",
    "cjk_segmentation",
    "sfx_glyph_mask",
    "component_bubble_cleaner",
    "verified_rect_sign",
}
DIALOGUE_MASK_CONTENT_CLASSES = {
    "dialogue",
    "speech",
    "fala",
    "pensamento",
    "thought",
    "narration",
    "narrative",
    "narracao",
    "caption",
}
MASK_EVIDENCE_MIN_SCORE = 0.18
AUTOMATIC_MASK_EVIDENCE_REJECT_REASONS = {
    "raw_mask_pixels_zero",
    "coverage_too_low",
    "mask_kind_not_fast_fill_allowed",
}


def _image_hw(image_shape: tuple[int, ...]) -> tuple[int, int]:
    return int(image_shape[0]), int(image_shape[1])


def _normalize_bbox(value, width: int, height: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _is_translator_note_block(block: dict) -> bool:
    text = str(
        block.get("translated")
        or block.get("text")
        or block.get("original")
        or ""
    ).strip().lower()
    return text.startswith("t/n:") or text.startswith("tn:") or text.startswith("n/t:")


def _translator_note_uses_text_only_mask(
    block: dict,
    image_rgb: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> bool:
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    flags = {str(flag).strip() for flag in block.get("qa_flags") or [] if str(flag).strip()}
    if source == "translator_note_text_mask" or "translator_note_text_only_mask" in flags:
        return True
    if not _is_translator_note_block(block):
        return False
    background = _rgb_luma_chroma(block.get("background_rgb"))
    if background is not None:
        luma, chroma = background
        if luma <= 150.0 and chroma <= 80:
            return True
        if luma >= 210.0 and chroma <= 45:
            return False
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return False
    height, width = _image_hw(image_shape)
    anchor = _normalize_bbox(
        block.get("text_pixel_bbox") or block.get("bbox") or block.get("source_bbox"),
        width,
        height,
    )
    if anchor is None:
        return False
    x1, y1, x2, y2 = anchor
    pad_x = max(8, int(round((x2 - x1) * 0.20)))
    pad_y = max(6, int(round((y2 - y1) * 0.35)))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return False
    crop = image_rgb[y1:y2, x1:x2, :3].astype(np.float32)
    if crop.size == 0:
        return False
    luma = crop[:, :, 0] * 0.299 + crop[:, :, 1] * 0.587 + crop[:, :, 2] * 0.114
    dark_ratio = float(np.count_nonzero(luma <= 96.0)) / float(max(1, luma.size))
    light_ratio = float(np.count_nonzero(luma >= 180.0)) / float(max(1, luma.size))
    return bool(dark_ratio >= 0.45 and light_ratio <= 0.35)


def _translator_note_bubble_clip_misses_text(
    block: dict,
    geometry_mask: np.ndarray | None,
    bubble_mask: np.ndarray | None,
) -> bool:
    if not _is_translator_note_block(block):
        return False
    if not isinstance(geometry_mask, np.ndarray) or not np.any(geometry_mask):
        return False
    if not isinstance(bubble_mask, np.ndarray) or not np.any(bubble_mask):
        return False
    if bubble_mask.shape[:2] != geometry_mask.shape[:2]:
        page_mask, _reason = _page_space_bubble_mask(block, geometry_mask.shape[:2])
        if page_mask is None or not np.any(page_mask):
            return False
        bubble_mask = page_mask
    if bubble_mask.shape[:2] != geometry_mask.shape[:2]:
        return False
    geometry_pixels = int(np.count_nonzero(geometry_mask > 0))
    overlap = int(np.count_nonzero((geometry_mask > 0) & (bubble_mask > 0)))
    return overlap < max(8, int(round(geometry_pixels * 0.20)))


def _normalize_polygon(value, width: int, height: int) -> list[list[int]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    points: list[list[int]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
        except Exception:
            return None
        points.append([max(0, min(width - 1, x)), max(0, min(height - 1, y))])
    return points if len(points) >= 3 else None


def _normalize_polygons(value, width: int, height: int) -> list[list[list[int]]]:
    if not isinstance(value, (list, tuple)) or not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2 and not (
        first and isinstance(first[0], (list, tuple))
    ):
        polygon = _normalize_polygon(value, width, height)
        return [polygon] if polygon else []

    polygons: list[list[list[int]]] = []
    for item in value:
        polygon = _normalize_polygon(item, width, height)
        if polygon:
            polygons.append(polygon)
    return polygons


def _polygon_bbox(polygon: list[list[int]]) -> list[int] | None:
    if not polygon:
        return None
    xs = [int(point[0]) for point in polygon if isinstance(point, (list, tuple)) and len(point) >= 2]
    ys = [int(point[1]) for point in polygon if isinstance(point, (list, tuple)) and len(point) >= 2]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


def _bbox_area(bbox: list[int] | tuple[int, int, int, int] | None) -> int:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    except Exception:
        return 0
    return max(0, x2 - x1) * max(0, y2 - y1)


def _filter_dark_bubble_connected_lobe_line_polygons(block: dict, width: int, height: int) -> dict:
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    if source != "image_dark_bubble_mask" or not block.get("line_polygons"):
        return block
    lobe_bbox = _normalize_bbox(block.get("bubble_mask_bbox") or block.get("bubbleMaskBbox"), width, height)
    balloon_bbox = _normalize_bbox(block.get("balloon_bbox") or block.get("balloonBbox"), width, height)
    if lobe_bbox is None or balloon_bbox is None:
        return block
    flags = {str(flag).strip() for flag in block.get("qa_flags") or [] if str(flag).strip()}
    connected_lobe = bool(
        "dark_bubble_connected_lobes_promoted" in flags
        or "dark_bubble_lobe_mask_bbox_preferred" in flags
    )
    if not connected_lobe:
        return block

    polygons = _normalize_polygons(block.get("line_polygons"), width, height)
    if len(polygons) <= 1:
        return block
    lx1, ly1, lx2, ly2 = lobe_bbox
    keep: list[list[list[int]]] = []
    removed = 0
    for polygon in polygons:
        poly_bbox = _polygon_bbox(polygon)
        if poly_bbox is None:
            continue
        px1, py1, px2, py2 = poly_bbox
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        overlap_w = max(0, min(px2, lx2) - max(px1, lx1))
        overlap_h = max(0, min(py2, ly2) - max(py1, ly1))
        overlap_area = overlap_w * overlap_h
        poly_area = max(1, _bbox_area(poly_bbox))
        inside_center = lx1 <= pcx <= lx2 and ly1 <= pcy <= ly2
        inside_enough = overlap_area >= int(round(poly_area * 0.35))
        if inside_center or inside_enough:
            keep.append(polygon)
        else:
            removed += 1
    if removed <= 0 or not keep:
        return block
    cleaned = dict(block)
    cleaned["line_polygons"] = keep
    metrics = cleaned.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["dark_bubble_connected_lobe_line_polygons_filtered"] = {
            "removed": int(removed),
            "kept": int(len(keep)),
            "lobe_bbox": list(lobe_bbox),
        }
    flags_list = cleaned.setdefault("qa_flags", [])
    if isinstance(flags_list, list) and "dark_bubble_connected_lobe_line_polygons_filtered" not in flags_list:
        flags_list.append("dark_bubble_connected_lobe_line_polygons_filtered")
    return cleaned


def _detach_mixed_sfx_from_white_bubble_block(block: dict, width: int, height: int) -> dict:
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    if source != "image_dark_bubble_mask" or not block.get("line_polygons"):
        return block
    metrics = block.get("qa_metrics") if isinstance(block.get("qa_metrics"), dict) else {}
    white_metric = metrics.get("derived_white_bubble_mask") if isinstance(metrics, dict) else None
    if not isinstance(white_metric, dict):
        return block
    white_bbox = _normalize_bbox(white_metric.get("mask_bbox"), width, height)
    if white_bbox is None:
        return block
    polygons = _normalize_polygons(block.get("line_polygons"), width, height)
    if len(polygons) <= 1:
        return block

    keep: list[list[list[int]]] = []
    removed = 0
    for polygon in polygons:
        bbox = _polygon_bbox(polygon)
        if bbox is None:
            continue
        bx1, by1, bx2, by2 = bbox
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        overlap = _bbox_intersection_area(bbox, white_bbox)
        area = max(1, _bbox_area(bbox))
        if (white_bbox[0] <= cx <= white_bbox[2] and white_bbox[1] <= cy <= white_bbox[3]) or overlap >= int(area * 0.25):
            keep.append(polygon)
        else:
            removed += 1
    if removed <= 0 or not keep:
        return block

    keep_bbox: list[int] | None = None
    for polygon in keep:
        bbox = _polygon_bbox(polygon)
        if bbox is not None:
            keep_bbox = bbox if keep_bbox is None else union_bbox(keep_bbox, bbox)
    if keep_bbox is None:
        return block

    cleaned = dict(block)
    cleaned["line_polygons"] = keep
    cleaned["bbox"] = list(keep_bbox)
    cleaned["source_bbox"] = list(keep_bbox)
    cleaned["text_pixel_bbox"] = list(keep_bbox)
    cleaned["bubble_mask_source"] = "image_white_bubble_mask"
    cleaned["bubbleMaskSource"] = "image_white_bubble_mask"
    cleaned["bubble_mask_bbox"] = list(white_bbox)
    cleaned["balloon_bbox"] = list(white_bbox)
    cleaned["block_profile"] = "white_balloon"
    cleaned["layout_profile"] = "white_balloon"
    for key in ("text", "original", "raw_ocr", "normalized_ocr", "translated", "traduzido"):
        value = cleaned.get(key)
        if isinstance(value, str):
            cleaned[key] = re.sub(r"^\s*\d+\s*/\s*", "", value).strip() or value
    metrics = cleaned.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["mixed_sfx_detached_from_white_bubble"] = {
            "white_bbox": list(white_bbox),
            "kept_bbox": list(keep_bbox),
            "removed_polygons": int(removed),
        }
    flags = cleaned.setdefault("qa_flags", [])
    if isinstance(flags, list):
        for flag in (
            "mixed_sfx_detached_from_white_bubble",
            "false_light_dark_bubble_promoted_to_white",
            "false_dark_white_style_neutralized",
        ):
            if flag not in flags:
                flags.append(flag)
        for stale in (
            "dark_bubble_oval_reocr",
            "dark_bubble_ellipse_bbox_mask",
            "dark_bubble_recovered_text_bbox_floor",
            "dark_bubble_visual_glyph_mask_replaced_geometry",
        ):
            while stale in flags:
                flags.remove(stale)
    return cleaned


def _normalize_rotation_degrees(value) -> float:
    try:
        numeric = float(value or 0)
    except Exception:
        return 0.0
    normalized = numeric % 360.0
    if normalized > 180.0:
        normalized -= 360.0
    if normalized <= -180.0:
        normalized += 360.0
    if abs(normalized) < 0.01:
        return 0.0
    return round(normalized, 2)


def _block_text_angle_degrees(block: dict) -> float:
    if block.get("text_angle_degrees") is not None:
        return _normalize_rotation_degrees(block.get("text_angle_degrees"))
    return _normalize_rotation_degrees(block.get("rotation_deg"))


def _rotated_text_polygons(block: dict, width: int, height: int) -> list[list[list[int]]]:
    if abs(_block_text_angle_degrees(block)) <= 5.0:
        return []
    return _normalize_polygons(block.get("rotated_polygon"), width, height)


def _text_geometry_polygons(block: dict, width: int, height: int) -> list[list[list[int]]]:
    return _rotated_text_polygons(block, width, height) or _normalize_polygons(
        block.get("line_polygons"),
        width,
        height,
    )


def _bbox_to_polygon(bbox: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]]


def bbox_to_octagon_polygon(bbox: list[int], cut_ratio: float = 0.18) -> list[list[int]]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    max_cut = max(0, (min(width, height) - 1) // 2)
    cut = min(max_cut, max(0, int(round(min(width, height) * cut_ratio))))
    if cut <= 0:
        return _bbox_to_polygon(bbox)
    right = x2 - 1
    bottom = y2 - 1
    return [
        [x1 + cut, y1],
        [right - cut, y1],
        [right, y1 + cut],
        [right, bottom - cut],
        [right - cut, bottom],
        [x1 + cut, bottom],
        [x1, bottom - cut],
        [x1, y1 + cut],
    ]


def bbox_to_octagon_mask(
    width: int,
    height: int,
    bbox: list[int],
    padding: int = 0,
    cut_ratio: float = 0.18,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    normalized = _normalize_bbox(
        [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
        width,
        height,
    )
    if normalized is None:
        return mask
    cv2.fillPoly(
        mask,
        [np.asarray(bbox_to_octagon_polygon(normalized, cut_ratio=cut_ratio), dtype=np.int32)],
        255,
    )
    return mask


def _font_size_from_block(block: dict) -> int | None:
    for key in ("font_size_px", "font_size", "tamanho_fonte"):
        value = block.get(key)
        try:
            if value is not None:
                return int(round(float(value)))
        except Exception:
            pass
    estilo = block.get("estilo")
    if isinstance(estilo, dict):
        for key in ("tamanho", "font_size", "font_size_px"):
            try:
                value = estilo.get(key)
                if value is not None:
                    return int(round(float(value)))
            except Exception:
                pass
    return None


def glyph_padding(font_size_px: int | float | None) -> int:
    try:
        font_size = int(round(float(font_size_px)))
    except Exception:
        font_size = 16
    return max(3, int(font_size * 0.06))


def polygon_to_mask(points, image_shape: tuple[int, ...]) -> np.ndarray:
    height, width = _image_hw(image_shape)
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = _normalize_polygon(points, width, height)
    if not polygon:
        return mask
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return mask


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_area(bbox: list[int]) -> int:
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _bbox_intersection_area(a: list[int], b: list[int]) -> int:
    ix1 = max(int(a[0]), int(b[0]))
    iy1 = max(int(a[1]), int(b[1]))
    ix2 = min(int(a[2]), int(b[2]))
    iy2 = min(int(a[3]), int(b[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def _bbox_iou(a: list[int], b: list[int]) -> float:
    inter = _bbox_intersection_area(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / float(max(1, union))


def _bbox_min_overlap_ratio(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / float(max(1, min(_bbox_area(a), _bbox_area(b))))


def _append_qa_flag(block: dict, flag: str) -> None:
    flags = block.setdefault("qa_flags", [])
    if not isinstance(flags, list):
        flags = [str(flags)]
        block["qa_flags"] = flags
    if flag not in flags:
        flags.append(flag)


def _remove_qa_flag(block: dict, flag: str) -> None:
    flags = block.get("qa_flags")
    if not isinstance(flags, list):
        return
    block["qa_flags"] = [item for item in flags if item != flag]


def _remove_qa_flags(block: dict, flags_to_remove: set[str]) -> None:
    flags = block.get("qa_flags")
    if not isinstance(flags, list):
        return
    block["qa_flags"] = [item for item in flags if str(item) not in flags_to_remove]


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(round(float(value))))
    except Exception:
        return default


def _coerce_score(value: object, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def _stable_reject_reasons(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    reasons: list[str] = []
    for item in value:
        reason = str(item or "").strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons


def _mask_evidence_content_class(region: dict) -> str:
    return "text"


def consolidate_mask_evidence(
    region: dict,
    *,
    kind: str | None = None,
    raw_mask_pixels: int | None = None,
    expanded_mask_pixels: int | None = None,
    evidence_score: float | None = None,
    fast_fill_reject_reasons: list[str] | None = None,
    min_evidence_score: float = MASK_EVIDENCE_MIN_SCORE,
) -> dict:
    """Persist the single mask-evidence contract used by fast fill and QA."""

    existing = region.get("mask_evidence") if isinstance(region.get("mask_evidence"), dict) else {}
    existing_kind = str(existing.get("kind") or "").strip()
    normalized_kind = str(kind or existing_kind or "none").strip()
    if normalized_kind not in FAST_FILL_MASK_EVIDENCE_KINDS and normalized_kind not in {
        "clipped_line_polygon",
        "none",
    }:
        normalized_kind = "none"

    raw_pixels = _coerce_int(
        raw_mask_pixels if raw_mask_pixels is not None else existing.get("raw_mask_pixels"),
        0,
    )
    expanded_pixels = _coerce_int(
        expanded_mask_pixels if expanded_mask_pixels is not None else existing.get("expanded_mask_pixels"),
        raw_pixels,
    )
    score = _coerce_score(
        evidence_score if evidence_score is not None else existing.get("evidence_score"),
        1.0 if raw_pixels > 0 else 0.0,
    )
    reasons = [
        reason
        for reason in _stable_reject_reasons(existing.get("fast_fill_reject_reasons"))
        if reason not in AUTOMATIC_MASK_EVIDENCE_REJECT_REASONS
    ]
    for reason in _stable_reject_reasons(fast_fill_reject_reasons):
        if reason not in reasons:
            reasons.append(reason)

    if normalized_kind not in FAST_FILL_MASK_EVIDENCE_KINDS and "mask_kind_not_fast_fill_allowed" not in reasons:
        reasons.append("mask_kind_not_fast_fill_allowed")
    if raw_pixels <= 0 and "raw_mask_pixels_zero" not in reasons:
        reasons.append("raw_mask_pixels_zero")
    if score < min_evidence_score and "coverage_too_low" not in reasons:
        reasons.append("coverage_too_low")

    fast_fill_allowed = not reasons
    evidence = {
        "kind": normalized_kind,
        "raw_mask_pixels": raw_pixels,
        "expanded_mask_pixels": expanded_pixels,
        "evidence_score": round(float(score), 6),
        "fast_fill_allowed": bool(fast_fill_allowed),
        "fast_fill_reject_reasons": reasons,
    }
    region["mask_evidence"] = evidence
    route_action = str(region.get("route_action") or "").strip().lower()
    fast_fill_route = route_action in {"", "translate_inpaint_render", "inpaint_only"}
    if (
        not fast_fill_allowed
        and fast_fill_route
        and "raw_mask_pixels_zero" in reasons
    ):
        _append_qa_flag(region, "fast_fill_no_glyph_evidence")
    elif "raw_mask_pixels_zero" not in reasons:
        _remove_qa_flag(region, "fast_fill_no_glyph_evidence")
    elif not fast_fill_route:
        _remove_qa_flag(region, "fast_fill_no_glyph_evidence")
    return evidence


def _has_line_polygon_geometry(block: dict, width: int, height: int) -> bool:
    return bool(_text_geometry_polygons(block, width, height))


def _block_special_class(block: dict) -> str:
    content_class = str(block.get("content_class") or "").strip().lower()
    tipo = str(block.get("tipo") or "").strip().lower()
    route_action = str(block.get("route_action") or "").strip().lower()
    text = " ".join(str(block.get(key) or "") for key in ("text", "original", "raw_ocr", "normalized_ocr"))
    sfx = block.get("sfx")
    if isinstance(sfx, dict):
        text = f"{text} {sfx.get('source_text') or ''}"
    has_hangul = bool(HANGUL_PATTERN.search(text))
    if route_action == "translate_sfx_inpaint_render" or content_class == "sfx":
        return "sfx"
    if tipo in {"sfx", "sound_effect", "sound"} and has_hangul:
        return "sfx"
    if content_class in SPECIAL_MASK_CLASSES:
        return content_class
    if tipo in SPECIAL_MASK_CLASSES:
        return tipo
    return ""


def _is_special_mask_class(block: dict) -> bool:
    return bool(_block_special_class(block))


def _rgb_luma_chroma(rgb: object) -> tuple[float, int] | None:
    if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
        return None
    try:
        channels = [int(round(float(value))) for value in rgb[:3]]
    except Exception:
        return None
    luma = (channels[0] * 0.299) + (channels[1] * 0.587) + (channels[2] * 0.114)
    return float(luma), int(max(channels) - min(channels))


def _text_words_upper(block: dict) -> set[str]:
    source = " ".join(
        str(block.get(key) or "")
        for key in ("text", "original", "raw_ocr", "translated", "traduzido")
    )
    return set(re.findall(r"[A-Z]+", source.upper()))


def _candidate_text_search_mask(block: dict, width: int, height: int) -> tuple[np.ndarray, list[int]] | None:
    candidate = np.zeros((height, width), dtype=np.uint8)
    polygons = _text_geometry_polygons(block, width, height)
    if polygons:
        for polygon in polygons:
            cv2.fillPoly(candidate, [np.asarray(polygon, dtype=np.int32)], 255)
        bbox = _bbox_from_mask(candidate)
        if bbox:
            return candidate, bbox

    bbox = (
        _normalize_bbox(block.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(block.get("bbox"), width, height)
    )
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    candidate[y1:y2, x1:x2] = 255
    return candidate, bbox


def _merged_source_bbox_candidates(block: dict, width: int, height: int) -> list[list[int]]:
    candidates: list[list[int]] = []
    for key in ("_merged_source_bboxes", "merged_source_bboxes", "source_bboxes"):
        raw_items = block.get(key)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            bbox = _normalize_bbox(item, width, height)
            if not bbox:
                continue
            if _bbox_area(bbox) < 32:
                continue
            if any(_bbox_iou(bbox, existing) >= 0.92 for existing in candidates):
                continue
            candidates.append(bbox)
    return candidates


def _has_explicit_text_geometry(block: dict, width: int, height: int) -> bool:
    if _text_geometry_polygons(block, width, height):
        return True
    return _normalize_bbox(block.get("text_pixel_bbox"), width, height) is not None


def bbox_overreach_ratio(block: dict, image_shape: tuple[int, ...]) -> float:
    """Return how much broader bbox is than line/text-pixel geometry."""

    height, width = _image_hw(image_shape)
    broad_bbox = _normalize_bbox(block.get("bbox"), width, height)
    if not broad_bbox:
        return 0.0
    geometry_bbox = _text_geometry_bbox(block, image_shape)
    if not geometry_bbox:
        return 0.0
    broad_area = _bbox_area(broad_bbox)
    geometry_area = _bbox_area(geometry_bbox)
    if geometry_area <= 0:
        return 0.0
    return max(0.0, (broad_area - geometry_area) / float(geometry_area))


def _record_overreach_qa(block: dict, image_shape: tuple[int, ...]) -> float:
    height, width = _image_hw(image_shape)
    ratio = bbox_overreach_ratio(block, image_shape)
    broad_bbox = _normalize_bbox(block.get("bbox"), width, height)
    geometry_bbox = _text_geometry_bbox(block, image_shape)
    # If we have OCR line polygons, the broad bbox is only a search/balloon
    # envelope. It is still useful to review, but it should not block export
    # unless the broad box is the geometry that will actually drive masking.
    reliable_line_geometry = _has_line_polygon_geometry(block, width, height)
    broad_bbox_drives_mask = bool(block.get("allow_broad_bbox_text_search")) or not reliable_line_geometry
    if ratio >= OVERREACH_RATIO_FLAG:
        qa_metrics = block.setdefault("qa_metrics", {})
        if isinstance(qa_metrics, dict):
            qa_metrics["bbox_overreach"] = {
                "ratio": round(float(ratio), 6),
                "bbox": list(broad_bbox) if broad_bbox else None,
                "text_geometry_bbox": list(geometry_bbox) if geometry_bbox else None,
                "has_line_polygon_geometry": bool(reliable_line_geometry),
                "broad_bbox_drives_mask": bool(broad_bbox_drives_mask),
            }
    if ratio >= OVERREACH_RATIO_CRITICAL and broad_bbox_drives_mask:
        _append_qa_flag(block, "bbox_overreach_critical")
    elif ratio >= OVERREACH_RATIO_FLAG and broad_bbox_drives_mask:
        _append_qa_flag(block, "bbox_overreach")
    return ratio


def _add_broad_bbox_search_candidate(
    block: dict,
    width: int,
    height: int,
    candidate_infos: list[tuple[np.ndarray, list[int], str]],
    *,
    force: bool = False,
) -> None:
    bbox = _normalize_bbox(block.get("bbox"), width, height)
    if not bbox:
        return
    if candidate_infos and not force:
        existing = np.zeros((height, width), dtype=np.uint8)
        for candidate_mask, _bbox, _kind in candidate_infos:
            existing = np.maximum(existing, candidate_mask)
        existing_bbox = _bbox_from_mask(existing)
        if existing_bbox:
            bx1, by1, bx2, by2 = bbox
            ex1, ey1, ex2, ey2 = existing_bbox
            bbox_h = max(1, by2 - by1)
            existing_h = max(1, ey2 - ey1)
            bbox_w = max(1, bx2 - bx1)
            existing_w = max(1, ex2 - ex1)
            if bbox_h < existing_h * 1.65 and bbox_w < existing_w * 1.65:
                return
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(_bbox_to_polygon(bbox), dtype=np.int32)], 255)
    candidate_infos.append((mask, bbox, "forced_broad" if force else "broad"))


def _source_reference_bbox(block: dict, width: int, height: int) -> list[int] | None:
    for key in ("source_bbox", "layout_bbox", "bbox"):
        bbox = _normalize_bbox(block.get(key), width, height)
        if bbox:
            return bbox
    return None


def _should_add_tight_white_source_search(block: dict, image_shape: tuple[int, ...]) -> bool:
    height, width = _image_hw(image_shape)
    if not _text_geometry_polygons(block, width, height):
        return False
    source_bbox = _source_reference_bbox(block, width, height)
    geometry_bbox = _text_geometry_bbox(block, image_shape)
    if not source_bbox or not geometry_bbox:
        return False
    compat_balloon = _normalize_bbox(block.get("balloon_bbox"), width, height)
    if source_bbox == compat_balloon and not _balloon_bbox_is_tight_text_anchor(block, image_shape):
        return False
    source_area = max(1, _bbox_area(source_bbox))
    geometry_area = max(1, _bbox_area(geometry_bbox))
    if source_area / float(geometry_area) > 2.35:
        return False
    sx1, sy1, sx2, sy2 = source_bbox
    gx1, gy1, gx2, gy2 = geometry_bbox
    lateral_extra = max(0, gx1 - sx1, sx2 - gx2)
    vertical_extra = max(0, gy1 - sy1, sy2 - gy2)
    return lateral_extra >= 4 or vertical_extra >= 4


def _add_tight_white_source_search_candidate(
    block: dict,
    width: int,
    height: int,
    image_shape: tuple[int, ...],
    candidate_infos: list[tuple[np.ndarray, list[int], str]],
) -> None:
    if not _should_add_tight_white_source_search(block, image_shape):
        return
    bbox = _source_reference_bbox(block, width, height)
    if not bbox:
        return
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 255
    candidate_infos.append((mask, bbox, "tight_white_source_bbox"))


def _filter_tight_white_source_components(
    cleaned: np.ndarray,
    *,
    candidate_bbox: list[int],
    block: dict,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(cleaned, np.ndarray) or not np.any(cleaned):
        return cleaned
    geometry_bbox = _text_geometry_bbox(block, image_shape)
    geometry_mask = mask_from_text_geometry(block, image_shape)
    if geometry_bbox is None or geometry_mask is None or not np.any(geometry_mask):
        return cleaned

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8),
        connectivity=8,
    )
    if labels_count <= 1:
        return cleaned

    kept = np.zeros_like(cleaned, dtype=np.uint8)
    gx1, gy1, gx2, gy2 = geometry_bbox
    geometry_w = max(1, gx2 - gx1)
    geometry_h = max(1, gy2 - gy1)
    cx1, cy1, cx2, cy2 = candidate_bbox
    for label in range(1, labels_count):
        x, y, comp_w, comp_h, area = [int(v) for v in stats[label]]
        if area < 2 or comp_w <= 0 or comp_h <= 0:
            continue
        component = labels == label
        if np.any((geometry_mask > 0) & component):
            kept[component] = 255
            continue
        y_overlap = max(0, min(y + comp_h, gy2) - max(y, gy1))
        y_overlap_ratio = y_overlap / float(max(1, min(comp_h, geometry_h)))
        horizontal_gap = max(0, gx1 - (x + comp_w), x - gx2)
        close_to_geometry = y_overlap_ratio >= 0.35 and horizontal_gap <= max(24, int(round(geometry_w * 0.30)))
        glyph_sized = comp_w <= max(34, int(round(geometry_w * 0.28))) and comp_h <= max(
            48,
            int(round(geometry_h * 0.95)),
        )
        touches_source_edge = x <= cx1 + 1 or y <= cy1 + 1 or x + comp_w >= cx2 - 1 or y + comp_h >= cy2 - 1
        if touches_source_edge and not (close_to_geometry and glyph_sized and area <= 420):
            continue
        if close_to_geometry and glyph_sized:
            kept[component] = 255
    return kept


def _filter_unanchored_raw_text_components(
    raw_mask: np.ndarray,
    block: dict,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(raw_mask, np.ndarray) or not np.any(raw_mask):
        return raw_mask
    if not block.get("line_polygons"):
        return raw_mask
    profile = str(block.get("layout_profile") or block.get("block_profile") or "").strip().lower()
    balloon_type = str(block.get("balloon_type") or "").strip().lower()
    bubble_source = str(block.get("bubble_mask_source") or "").strip().lower()
    if profile != "white_balloon" and balloon_type != "white" and bubble_source != "image_white_bubble_mask":
        return raw_mask

    height, width = _image_hw(image_shape)
    geometry_mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in _text_geometry_polygons(block, width, height):
        cv2.fillPoly(geometry_mask, [np.asarray(polygon, dtype=np.int32)], 255)
    if not isinstance(geometry_mask, np.ndarray) or not np.any(geometry_mask):
        return raw_mask
    source_support = _raw_floor_support_mask_for_block(block, (height, width))

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (raw_mask > 0).astype(np.uint8),
        connectivity=8,
    )
    if labels_count <= 1:
        return raw_mask

    kept = np.zeros_like(raw_mask, dtype=np.uint8)
    removed: list[dict] = []
    for label in range(1, labels_count):
        component = labels == label
        if np.any(component & (geometry_mask > 0)):
            kept[component] = 255
            continue
        if isinstance(source_support, np.ndarray) and np.any(component & (source_support > 0)):
            kept[component] = 255
            continue
        x, y, comp_w, comp_h, area = [int(v) for v in stats[label]]
        removed.append({"bbox": [x, y, x + comp_w, y + comp_h], "area": area})

    if not removed or not np.any(kept):
        return raw_mask

    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["unanchored_raw_text_components_removed"] = removed
    flags = block.setdefault("qa_flags", [])
    if isinstance(flags, list) and "unanchored_raw_text_components_removed" not in flags:
        flags.append("unanchored_raw_text_components_removed")
    return kept


def _should_force_broad_text_search(block: dict, image_shape: tuple[int, ...]) -> bool:
    height, width = _image_hw(image_shape)
    if not _text_geometry_polygons(block, width, height):
        return False
    try:
        confidence = float(block.get("confidence", 1.0) or 0.0)
    except Exception:
        confidence = 1.0
    flags = {str(flag).strip().upper() for flag in block.get("qa_flags") or []}
    try:
        rotation_abs = abs(float(block.get("rotation_deg") or 0.0))
    except Exception:
        rotation_abs = 0.0
    if confidence >= 0.55 and "TEXT_CLIPPED" not in flags and rotation_abs < 45.0:
        return False

    broad_bbox = _normalize_bbox(block.get("bbox"), width, height)
    geometry_bbox = _text_geometry_bbox(block, image_shape)
    if not broad_bbox or not geometry_bbox:
        return False
    gx1, gy1, gx2, gy2 = geometry_bbox
    bx1, by1, bx2, by2 = broad_bbox
    geometry_w = max(1, gx2 - gx1)
    geometry_h = max(1, gy2 - gy1)
    lateral_extra = max(0, gx1 - bx1, bx2 - gx2)
    vertical_extra = max(0, gy1 - by1, by2 - gy2)
    return (
        lateral_extra >= max(18, int(geometry_w * 0.18))
        or vertical_extra >= max(18, int(geometry_h * 0.18))
    )


def _odd_kernel_size(value: int, maximum: int = 31) -> int:
    value = max(3, min(maximum, int(value)))
    return value if value % 2 else value - 1


def _raw_text_search_expand_px(block: dict) -> int:
    style = block.get("estilo") if isinstance(block.get("estilo"), dict) else {}
    profile = str(block.get("layout_profile") or block.get("block_profile") or "").strip().lower()
    balloon_type = str(block.get("balloon_type") or "").strip().lower()
    expand = 0
    background = _rgb_luma_chroma(block.get("background_rgb"))
    non_plain_background = bool(background is not None and (background[0] < 228.0 or background[1] > 18.0))
    if style.get("italico") or profile == "sfx" or non_plain_background:
        expand = 7
    if block.get("line_polygons") and (balloon_type == "white" or profile == "white_balloon"):
        expand = max(expand, 6)
    font_size = _font_size_from_block(block)
    if font_size is not None and font_size >= 40:
        expand = max(expand, 6)
    return expand


RAW_TEXT_EVIDENCE_MISSING_FLAG = "raw_text_evidence_missing"
RAW_TEXT_EVIDENCE_REJECTED_FLAG = "raw_text_evidence_rejected"


def raw_text_evidence_rejected(block: dict) -> bool:
    flags = {str(flag).strip().lower() for flag in block.get("qa_flags") or []}
    return RAW_TEXT_EVIDENCE_MISSING_FLAG in flags or RAW_TEXT_EVIDENCE_REJECTED_FLAG in flags


def _requires_segment_like_raw_text_evidence(block: dict, image_shape: tuple[int, ...]) -> bool:
    height, width = _image_hw(image_shape)
    if _has_line_polygon_geometry(block, width, height) or _is_special_mask_class(block):
        return False
    support_bbox = _text_geometry_bbox(block, image_shape) or _normalize_bbox(block.get("bbox"), width, height)
    if not support_bbox:
        return False
    support_area = _bbox_area(support_bbox)
    image_area = max(1, width * height)
    support_w = max(1, support_bbox[2] - support_bbox[0])
    support_h = max(1, support_bbox[3] - support_bbox[1])
    width_ratio = support_w / float(max(1, width))
    height_ratio = support_h / float(max(1, height))
    large_support = support_area >= max(45_000, int(image_area * 0.12))
    return (
        support_area >= max(80_000, int(image_area * 0.18))
        or (large_support and max(width_ratio, height_ratio) >= 0.60)
        or (large_support and width_ratio >= 0.35 and height_ratio >= 0.45)
    )


def _record_raw_text_evidence_metric(block: dict, reason: str, metrics: dict) -> None:
    qa_metrics = block.setdefault("qa_metrics", {})
    if isinstance(qa_metrics, dict):
        qa_metrics["raw_text_evidence"] = {"reason": reason, **metrics}


def _append_unique_bbox_metadata(block: dict, field: str, bbox: list[int] | None) -> None:
    if not bbox:
        return
    normalized = [int(v) for v in bbox[:4]]
    values = block.setdefault(field, [])
    if not isinstance(values, list):
        values = []
        block[field] = values
    duplicate = any(
        _bbox_iou(normalized, existing) >= 0.94
        for existing in values
        if isinstance(existing, list) and len(existing) >= 4
    )
    if not duplicate:
        values.append(normalized)


def _record_raw_text_validation_result(block: dict, mask: np.ndarray | None) -> None:
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        block["validated_by_segment_mask"] = False
        return
    raw_bbox = _bbox_from_mask(mask)
    block["validated_by_segment_mask"] = True
    if raw_bbox:
        block["_raw_text_evidence_bbox"] = raw_bbox
    block["_raw_text_evidence_pixels"] = int(np.count_nonzero(mask))


def _reject_raw_text_mask_without_koharu_evidence(
    block: dict,
    text_mask: np.ndarray,
    image_shape: tuple[int, ...],
) -> bool:
    if not _requires_segment_like_raw_text_evidence(block, image_shape):
        return False
    if not isinstance(text_mask, np.ndarray) or not np.any(text_mask):
        _record_raw_text_validation_result(block, None)
        _append_qa_flag(block, RAW_TEXT_EVIDENCE_MISSING_FLAG)
        _record_raw_text_evidence_metric(block, "missing_raw_text_pixels", {})
        return True

    height, width = _image_hw(image_shape)
    support_bbox = _text_geometry_bbox(block, image_shape) or _normalize_bbox(block.get("bbox"), width, height)
    if not support_bbox:
        return False
    support_area = max(1, _bbox_area(support_bbox))
    support_w = max(1, support_bbox[2] - support_bbox[0])
    support_h = max(1, support_bbox[3] - support_bbox[1])
    raw_bbox = _bbox_from_mask(text_mask)
    if not raw_bbox:
        _record_raw_text_validation_result(block, None)
        _append_qa_flag(block, RAW_TEXT_EVIDENCE_MISSING_FLAG)
        _record_raw_text_evidence_metric(block, "missing_raw_text_bbox", {})
        return True

    raw_area = int(np.count_nonzero(text_mask))
    labels_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (text_mask > 0).astype(np.uint8),
        connectivity=8,
    )
    component_count = max(0, int(labels_count) - 1)
    if component_count <= 0:
        _record_raw_text_validation_result(block, None)
        _append_qa_flag(block, RAW_TEXT_EVIDENCE_MISSING_FLAG)
        _record_raw_text_evidence_metric(block, "missing_raw_components", {})
        return True

    component_areas = stats[1:, cv2.CC_STAT_AREA] if labels_count > 1 else np.array([], dtype=np.int32)
    largest_component = int(np.max(component_areas)) if component_areas.size else 0
    largest_ratio = largest_component / float(max(1, raw_area))
    raw_w = max(1, raw_bbox[2] - raw_bbox[0])
    raw_h = max(1, raw_bbox[3] - raw_bbox[1])
    raw_bbox_area = max(1, _bbox_area(raw_bbox))
    raw_bbox_support_ratio = raw_bbox_area / float(support_area)
    raw_h_support_ratio = raw_h / float(support_h)
    raw_w_support_ratio = raw_w / float(support_w)

    giant_component = (
        largest_ratio >= 0.62
        and largest_component >= max(5_000, int(support_area * 0.03))
        and (raw_h_support_ratio >= 0.22 or raw_bbox_support_ratio >= 0.12)
    )
    few_tall_components = (
        component_count <= 4
        and raw_area >= int(support_area * 0.035)
        and raw_h_support_ratio >= 0.36
        and raw_w_support_ratio >= 0.35
    )
    if not (giant_component or few_tall_components):
        return False

    _record_raw_text_validation_result(block, None)
    _append_qa_flag(block, RAW_TEXT_EVIDENCE_REJECTED_FLAG)
    _record_raw_text_evidence_metric(
        block,
        "giant_component_without_line_geometry" if giant_component else "few_tall_components_without_line_geometry",
        {
            "component_count": component_count,
            "largest_component": largest_component,
            "largest_component_ratio": round(float(largest_ratio), 6),
            "raw_area": raw_area,
            "raw_bbox": raw_bbox,
            "support_bbox": support_bbox,
            "raw_bbox_support_ratio": round(float(raw_bbox_support_ratio), 6),
            "raw_h_support_ratio": round(float(raw_h_support_ratio), 6),
            "raw_w_support_ratio": round(float(raw_w_support_ratio), 6),
        },
    )
    return True


def build_raw_text_mask_from_image(
    block: dict,
    image_rgb: np.ndarray,
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    height, width = _image_hw(image_shape)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.shape[0] < height or image_rgb.shape[1] < width:
        return None

    candidate_infos: list[tuple[np.ndarray, list[int], str]] = []
    polygons = _text_geometry_polygons(block, width, height)
    has_explicit_geometry = bool(polygons) or _normalize_bbox(block.get("text_pixel_bbox"), width, height) is not None
    overreach_ratio = _record_overreach_qa(block, image_shape)
    allow_broad_bbox_search = bool(block.get("allow_broad_bbox_text_search"))
    force_broad_text_search = _should_force_broad_text_search(block, image_shape)
    if polygons:
        for polygon in polygons:
            candidate = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(candidate, [np.asarray(polygon, dtype=np.int32)], 255)
            bbox = _bbox_from_mask(candidate)
            if bbox:
                candidate_infos.append((candidate, bbox, "polygon"))
        _add_tight_white_source_search_candidate(
            block,
            width,
            height,
            image_shape,
            candidate_infos,
        )
        if allow_broad_bbox_search or force_broad_text_search:
            _add_broad_bbox_search_candidate(
                block,
                width,
                height,
                candidate_infos,
                force=force_broad_text_search,
            )
    else:
        source_bboxes = _merged_source_bbox_candidates(block, width, height)
        for source_bbox in source_bboxes:
            candidate = np.zeros((height, width), dtype=np.uint8)
            sx1, sy1, sx2, sy2 = source_bbox
            candidate[sy1:sy2, sx1:sx2] = 255
            candidate_infos.append((candidate, source_bbox, "source_bbox"))
        candidate_info = _candidate_text_search_mask(block, width, height)
        if candidate_info is not None and not source_bboxes:
            candidate_infos.append((candidate_info[0], candidate_info[1], "text_bbox"))
        if allow_broad_bbox_search or not has_explicit_geometry:
            _add_broad_bbox_search_candidate(block, width, height, candidate_infos)

    if not candidate_infos:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    search_expand_px = _raw_text_search_expand_px(block)
    search_kernel = None
    if search_expand_px > 0:
        search_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (search_expand_px * 2 + 1, search_expand_px * 2 + 1),
        )

    for candidate_mask, bbox, candidate_kind in candidate_infos:
        forced_broad_candidate = candidate_kind == "forced_broad"
        record_bbox = list(bbox)
        if search_kernel is not None and not forced_broad_candidate:
            candidate_mask = cv2.dilate(candidate_mask, search_kernel, iterations=1)
            expanded_bbox = _bbox_from_mask(candidate_mask)
            if expanded_bbox:
                bbox = expanded_bbox
        candidate_area = int(np.count_nonzero(candidate_mask))
        if candidate_area < 8:
            continue

        x1, y1, x2, y2 = bbox
        pad = 2
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = image_rgb[y1:y2, x1:x2]
        roi_candidate = candidate_mask[y1:y2, x1:x2] > 0
        if roi.size == 0 or int(np.count_nonzero(roi_candidate)) < 8:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi.astype(np.uint8)
        kernel_size = _odd_kernel_size(max(7, min(gray.shape[:2]) // 2), maximum=31)
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0).astype(np.float32)
        gray_f = gray.astype(np.float32)
        candidate_gray = gray_f[roi_candidate]
        if candidate_gray.size == 0:
            continue
        low_luma = float(np.percentile(candidate_gray, 8))
        median_luma = float(np.median(candidate_gray))
        high_luma = float(np.percentile(candidate_gray, 92))
        dark_span = median_luma - low_luma
        light_span = high_luma - median_luma
        dark_cutoff = min(median_luma - max(16.0, dark_span * 0.35), low_luma + 22.0)
        light_cutoff = max(median_luma + max(16.0, light_span * 0.35), high_luma - 22.0)

        side_candidates: list[tuple[float, np.ndarray]] = []
        if dark_span >= 24.0:
            dark_like = gray_f <= dark_cutoff
            dark_area = int(np.count_nonzero(dark_like & roi_candidate))
            dark_ratio = dark_area / max(1, candidate_area)
            if 0.001 <= dark_ratio <= 0.45:
                side_candidates.append((dark_ratio, dark_like))
        if light_span >= 24.0:
            light_like = gray_f >= light_cutoff
            light_area = int(np.count_nonzero(light_like & roi_candidate))
            light_ratio = light_area / max(1, candidate_area)
            if 0.001 <= light_ratio <= 0.45:
                side_candidates.append((light_ratio, light_like))

        text_like = np.zeros_like(gray_f, dtype=bool)
        if side_candidates:
            _, text_like = min(side_candidates, key=lambda item: item[0])

        if not np.any(text_like & roi_candidate):
            dark_contrast = background - gray_f
            light_contrast = gray_f - background
            candidate_values = dark_contrast[roi_candidate]
            light_values = light_contrast[roi_candidate]
            if candidate_values.size == 0 or light_values.size == 0:
                continue
            threshold = max(18.0, min(50.0, float(np.percentile(candidate_values, 95)) * 0.5))
            light_threshold = max(18.0, min(50.0, float(np.percentile(light_values, 95)) * 0.5))
            text_like = (dark_contrast >= threshold) | (light_contrast >= light_threshold)
        raw_roi = (text_like & roi_candidate).astype(np.uint8) * 255
        if not np.any(raw_roi):
            continue

        cleaned = np.zeros_like(raw_roi)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((raw_roi > 0).astype(np.uint8), connectivity=8)
        min_area = max(2, int(candidate_area * 0.0005))
        max_component_area = max(8, int(candidate_area * 0.45))
        for label in range(1, num_labels):
            x, y, comp_w, comp_h, area = stats[label].tolist()
            if forced_broad_candidate:
                border_margin = 3
                touches_border = (
                    x <= border_margin
                    or y <= border_margin
                    or x + comp_w >= raw_roi.shape[1] - border_margin
                    or y + comp_h >= raw_roi.shape[0] - border_margin
                )
                border_like = touches_border and (
                    comp_w >= int(raw_roi.shape[1] * 0.22)
                    or comp_h >= int(raw_roi.shape[0] * 0.22)
                    or area >= int(candidate_area * 0.015)
                )
                if border_like:
                    continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            if min_area <= area <= max_component_area:
                cleaned[labels == label] = 255
        if not np.any(cleaned):
            continue
        if candidate_kind == "tight_white_source_bbox":
            candidate_canvas = np.zeros((height, width), dtype=np.uint8)
            candidate_canvas[y1:y2, x1:x2] = cleaned
            candidate_canvas = _filter_tight_white_source_components(
                candidate_canvas,
                candidate_bbox=record_bbox,
                block=block,
                image_shape=image_shape,
            )
            cleaned = candidate_canvas[y1:y2, x1:x2]
            if not np.any(cleaned):
                continue

        raw_area = int(np.count_nonzero(cleaned))
        if raw_area < max(8, int(candidate_area * 0.001)):
            continue
        if raw_area > int(candidate_area * 0.45):
            continue
        if candidate_kind in {"source_bbox", "text_bbox", "broad", "forced_broad"} and not polygons:
            candidate_canvas = np.zeros((height, width), dtype=np.uint8)
            candidate_canvas[y1:y2, x1:x2] = cleaned
            candidate_block = dict(block)
            candidate_block["qa_flags"] = list(block.get("qa_flags") or [])
            candidate_block["qa_metrics"] = dict(block.get("qa_metrics") or {})
            candidate_block["bbox"] = list(bbox)
            candidate_block["text_pixel_bbox"] = list(bbox)
            candidate_block.pop("_merged_source_bboxes", None)
            candidate_block.pop("merged_source_bboxes", None)
            candidate_block.pop("source_bboxes", None)
            if _reject_raw_text_mask_without_koharu_evidence(candidate_block, candidate_canvas, image_shape):
                if candidate_kind == "source_bbox":
                    _append_unique_bbox_metadata(block, "_rejected_text_source_bboxes", record_bbox)
                continue
        if candidate_kind == "source_bbox":
            _append_unique_bbox_metadata(block, "_validated_text_source_bboxes", record_bbox)
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], cleaned)

    if not np.any(mask):
        _record_raw_text_validation_result(block, None)
        if _requires_segment_like_raw_text_evidence(block, image_shape):
            _append_qa_flag(block, RAW_TEXT_EVIDENCE_MISSING_FLAG)
            _record_raw_text_evidence_metric(block, "missing_raw_text_pixels", {})
        return None
    filtered_mask = _filter_unanchored_raw_text_components(mask, block, image_shape)
    if isinstance(filtered_mask, np.ndarray) and np.any(filtered_mask):
        mask = filtered_mask.astype(np.uint8)
    if _reject_raw_text_mask_without_koharu_evidence(block, mask, image_shape):
        for source_bbox in block.pop("_validated_text_source_bboxes", []) or []:
            _append_unique_bbox_metadata(block, "_rejected_text_source_bboxes", source_bbox)
        return None
    _record_raw_text_validation_result(block, mask)
    return mask


def expand_text_mask(mask: np.ndarray, expand_px: int = 5) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        return mask
    if expand_px <= 0 or not np.any(mask):
        return mask.copy()
    kernel_size = int(expand_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def expand_text_mask_monotonic(
    mask: np.ndarray,
    *,
    limit_mask: np.ndarray | None = None,
    radius: int = 2,
) -> np.ndarray:
    """Dilate text pixels without ever dropping raw pixels."""

    if not isinstance(mask, np.ndarray):
        return mask
    raw = np.where(mask > 0, 255, 0).astype(np.uint8)
    if radius <= 0 or not np.any(raw):
        return raw.copy()
    kernel_size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(raw, kernel, iterations=1).astype(np.uint8)
    added = cv2.bitwise_and(dilated, cv2.bitwise_not(raw))
    if isinstance(limit_mask, np.ndarray) and limit_mask.shape[:2] == raw.shape[:2] and np.any(limit_mask):
        limit = np.where(limit_mask > 0, 255, 0).astype(np.uint8)
        added = cv2.bitwise_and(added, limit)
    return cv2.bitwise_or(raw, added).astype(np.uint8)


def _block_luma_chroma(block: dict) -> tuple[float, float] | None:
    value = block.get("background_rgb")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        r, g, b = [float(v) for v in value[:3]]
    except Exception:
        return None
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    chroma = max(r, g, b) - min(r, g, b)
    return luma, chroma


def _dark_or_colored_text_card_context(block: dict) -> bool:
    background = _block_luma_chroma(block)
    if background is not None:
        luma, chroma = background
        if luma < 135.0 or chroma > 45.0:
            return True
    profile = {
        str(block.get("layout_profile") or "").strip().lower(),
        str(block.get("block_profile") or "").strip().lower(),
        str(block.get("render_profile") or "").strip().lower(),
    }
    return bool(profile & {"dark_panel", "colored_status_panel", "status_panel", "card", "title_card"})


def _infer_dark_or_colored_card_from_mask(gray: np.ndarray, rgb: np.ndarray, base: np.ndarray) -> bool:
    if not isinstance(gray, np.ndarray) or not isinstance(rgb, np.ndarray) or not isinstance(base, np.ndarray):
        return False
    if gray.shape[:2] != base.shape[:2] or rgb.shape[:2] != base.shape[:2] or not np.any(base):
        return False
    active = base > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    near = cv2.dilate(base.astype(np.uint8), kernel, iterations=1) > 0
    ring = near & ~active
    if int(np.count_nonzero(ring)) < 24:
        return False
    ring_rgb = rgb[ring].astype(np.float32)
    ring_gray = gray[ring].astype(np.float32)
    ring_luma = float(np.median(ring_gray))
    ring_chroma = float(np.median(np.max(ring_rgb, axis=1) - np.min(ring_rgb, axis=1)))
    return ring_luma < 150.0 or ring_chroma > 42.0


def _merged_fragment_without_glyph_evidence(block: dict) -> bool:
    flags = {str(flag).strip() for flag in block.get("qa_flags") or []}
    return (
        "same_balloon_fragment_merged" in flags
        and (
            "raw_text_evidence_missing" in flags
            or "fast_fill_no_glyph_evidence" in flags
            or bool(block.get("raw_text_evidence_missing"))
        )
    )


def expand_light_halo_mask_for_dark_card(
    mask: np.ndarray,
    image_rgb: np.ndarray | None,
    block: dict | None = None,
    *,
    radius: int = 8,
) -> np.ndarray:
    """Include bright glow/anti-alias pixels around light text on dark/color cards."""

    if not isinstance(mask, np.ndarray):
        return mask
    base = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(base) or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return base
    if image_rgb.shape[:2] != base.shape[:2]:
        return base
    block = block or {}
    rgb = image_rgb.astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb.astype(np.uint8)
    inferred_card_context = _infer_dark_or_colored_card_from_mask(gray, rgb, base)
    if not _dark_or_colored_text_card_context(block) and not inferred_card_context:
        return base

    active = base > 0
    active_rgb = rgb[active].astype(np.float32) if rgb.ndim == 3 else None
    active_chroma = (
        float(np.median(np.max(active_rgb, axis=1) - np.min(active_rgb, axis=1)))
        if active_rgb is not None and active_rgb.size
        else 0.0
    )
    text_luma = float(np.median(gray[active]))
    block_background = _block_luma_chroma(block)
    inferred_background_luma = (
        float(block_background[0])
        if block_background is not None
        else float(np.percentile(gray[~active], 35)) if np.any(~active) else text_luma
    )
    background_chroma = float(block_background[1]) if block_background is not None else 0.0
    if abs(text_luma - inferred_background_luma) < 12.0 and abs(active_chroma - background_chroma) < 18.0:
        return base
    colored_light_text = active_chroma >= 48.0 and text_luma >= inferred_background_luma + 18.0
    colored_dark_effect = (
        inferred_background_luma <= 80.0
        and text_luma >= inferred_background_luma + 18.0
        and active_chroma >= 24.0
    )
    if text_luma < 150.0 and not colored_light_text and not colored_dark_effect:
        return base
    effective_radius = max(
        int(radius),
        14 if colored_light_text or colored_dark_effect else (10 if background_chroma > 45.0 else int(radius)),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (effective_radius * 2 + 1, effective_radius * 2 + 1))
    near = cv2.dilate(base, kernel, iterations=1) > 0
    ring = near & ~active
    if int(np.count_nonzero(ring)) < 8:
        return base
    background_luma = float(block_background[0]) if block_background is not None else float(np.median(gray[ring]))
    threshold = min(max(background_luma + 16.0, 48.0), max(48.0, text_luma - 18.0))
    halo = ring & (gray >= threshold)
    if inferred_card_context or colored_light_text or colored_dark_effect:
        ring_rgb = rgb.astype(np.float32)
        chroma = np.max(ring_rgb, axis=2) - np.min(ring_rgb, axis=2)
        colored_halo = ring & (chroma >= 28.0) & (gray >= max(38.0, background_luma - 14.0))
        halo |= colored_halo
    if colored_light_text:
        context_radius = effective_radius + 10
        context_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (context_radius * 2 + 1, context_radius * 2 + 1),
        )
        context_near = cv2.dilate(base, context_kernel, iterations=1) > 0
        context_ring = context_near & ~near
        if int(np.count_nonzero(context_ring)) >= 32 and rgb.ndim == 3:
            context_sample = rgb[context_ring].astype(np.float32)
            bg_rgb = np.median(context_sample, axis=0)
            bg_luma = float(np.median(gray[context_ring]))
            dist = np.sqrt(np.sum(np.square(rgb.astype(np.float32) - bg_rgb[None, None, :]), axis=2))
            color_effect = ring & (dist >= 26.0) & (gray >= max(18.0, bg_luma - 8.0))
            halo |= color_effect
    if int(np.count_nonzero(halo)) < 8:
        return base
    expanded = base.copy()
    expanded[halo] = 255
    return expanded.astype(np.uint8)


def safe_bubble_interior_mask(bubble_mask: np.ndarray, erode_px: int = 2) -> np.ndarray:
    """Return the usable interior of a real BubbleMask, excluding outline pixels."""

    if not isinstance(bubble_mask, np.ndarray):
        return np.zeros((0, 0), dtype=np.uint8)
    binary = np.where(bubble_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros(binary.shape[:2], dtype=np.uint8)
    erode_px = max(0, int(erode_px))
    if erode_px <= 0:
        return binary
    kernel_size = erode_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    interior = cv2.erode(binary, kernel, iterations=1)
    return interior.astype(np.uint8) if np.any(interior) else np.zeros(binary.shape[:2], dtype=np.uint8)


def build_bubble_limited_glyph_mask(
    glyph_mask: np.ndarray,
    bubble_mask: np.ndarray,
    *,
    glyph_expand_px: int = 2,
    bubble_erode_px: int = 2,
) -> np.ndarray:
    """Expand glyph/text pixels, then clip them to a safe real BubbleMask interior."""

    if not isinstance(glyph_mask, np.ndarray):
        return np.zeros((0, 0), dtype=np.uint8)
    glyph = np.where(glyph_mask > 0, 255, 0).astype(np.uint8)
    if not isinstance(bubble_mask, np.ndarray) or bubble_mask.shape[:2] != glyph.shape[:2]:
        return np.zeros(glyph.shape[:2], dtype=np.uint8)
    if not np.any(glyph) or not np.any(bubble_mask):
        return np.zeros(glyph.shape[:2], dtype=np.uint8)
    expanded = expand_text_mask(glyph, expand_px=max(0, int(glyph_expand_px)))
    interior = safe_bubble_interior_mask(bubble_mask, erode_px=bubble_erode_px)
    if interior.shape[:2] != glyph.shape[:2] or not np.any(interior):
        return np.zeros(glyph.shape[:2], dtype=np.uint8)
    return cv2.bitwise_and(expanded.astype(np.uint8), interior.astype(np.uint8))


def _final_text_mask_expand_px(block: dict) -> int:
    expand_px = 5
    flags = {str(flag).strip().upper() for flag in block.get("qa_flags") or []}
    try:
        rotation_abs = abs(float(block.get("rotation_deg") or 0.0))
    except Exception:
        rotation_abs = 0.0
    rotated_recovery = "ROTATED_TEXT_RECOVERY" in flags or str(block.get("detector") or "").strip().lower() == "rotated_full_page_recovery"
    if rotated_recovery and rotation_abs >= 35.0:
        expand_px = max(expand_px, 14)
    if rotated_recovery and "TEXT_CLIPPED" in flags and rotation_abs >= 35.0:
        expand_px = max(expand_px, 34)
    return expand_px


def clip_text_mask_to_balloon_interior(
    text_mask: np.ndarray,
    balloon_mask: np.ndarray | None,
    erode_px: int = 2,
) -> np.ndarray:
    if not isinstance(balloon_mask, np.ndarray) or not np.any(balloon_mask):
        return text_mask
    interior = safe_bubble_interior_mask(balloon_mask, erode_px=erode_px)
    if not np.any(interior):
        return np.zeros(text_mask.shape[:2], dtype=np.uint8)
    clipped = cv2.bitwise_and(text_mask.astype(np.uint8), interior)
    return clipped.astype(np.uint8)


def _raw_floor_support_mask_for_block(block: dict | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(block, dict):
        return None
    height, width = int(shape[0]), int(shape[1])
    support = np.zeros((height, width), dtype=np.uint8)
    for key in (
        "_raw_text_evidence_bbox",
        "raw_text_evidence_bbox",
        "raw_text_bbox",
        "ocr_text_bbox",
        "source_bbox",
    ):
        bbox = _normalize_bbox(block.get(key), width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = _expand_bbox_px(bbox, width, height, pad_x=4, pad_y=4) or bbox
        support[y1:y2, x1:x2] = 255
    if not np.any(support):
        return None
    return cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)


def _source_bbox_support_mask_for_block(block: dict | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(block, dict):
        return None
    height, width = int(shape[0]), int(shape[1])
    bbox = _normalize_bbox(block.get("source_bbox"), width, height)
    if bbox is None:
        return None
    support = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    support[y1:y2, x1:x2] = 255
    return support if np.any(support) else None


def _anchored_raw_floor_components(
    raw_mask: np.ndarray,
    support_mask: np.ndarray | None,
) -> np.ndarray:
    if not isinstance(support_mask, np.ndarray) or support_mask.shape[:2] != raw_mask.shape[:2] or not np.any(support_mask):
        return np.zeros(raw_mask.shape[:2], dtype=np.uint8)
    raw = np.where(raw_mask > 0, 255, 0).astype(np.uint8)
    kept = np.zeros(raw.shape[:2], dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == label
        overlap = int(np.count_nonzero(component & (support_mask > 0)))
        if overlap <= 0:
            continue
        kept[component] = 255
    return kept


def _block_uses_labeled_bubble_component(block: dict | None) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("bubble_id") is None and block.get("bubbleId") is None:
        return False
    bubble_mask = block.get("bubble_mask")
    if not isinstance(bubble_mask, np.ndarray) or bubble_mask.size == 0:
        return False
    try:
        labels = [int(value) for value in np.unique(bubble_mask) if int(value) != 0]
    except Exception:
        return False
    return len(set(labels)) > 1


def _drop_unanchored_balloon_outline_components(
    text_mask: np.ndarray,
    balloon_mask: np.ndarray | None,
    block: dict | None,
    *,
    erode_px: int,
) -> np.ndarray:
    if _block_uses_labeled_bubble_component(block):
        return text_mask
    if not isinstance(balloon_mask, np.ndarray) or balloon_mask.shape[:2] != text_mask.shape[:2] or not np.any(balloon_mask):
        return text_mask
    balloon_binary = np.where(balloon_mask > 0, 255, 0).astype(np.uint8)
    deep_interior = safe_bubble_interior_mask(balloon_binary, erode_px=max(int(erode_px) + 3, 5))
    if not np.any(deep_interior):
        return text_mask
    outline_zone = np.where((balloon_binary > 0) & (deep_interior == 0), 255, 0).astype(np.uint8)
    if not np.any(outline_zone):
        return text_mask
    narrow_support = _raw_floor_support_mask_for_block(block, text_mask.shape[:2])
    mask = np.where(text_mask > 0, 255, 0).astype(np.uint8)
    kept = np.zeros(mask.shape[:2], dtype=np.uint8)
    removed = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == label
        touches_outline = bool(np.any(component & (outline_zone > 0)))
        anchored = bool(
            isinstance(narrow_support, np.ndarray)
            and narrow_support.shape[:2] == mask.shape[:2]
            and np.any(component & (narrow_support > 0))
        )
        if touches_outline:
            interior_component = component & (deep_interior > 0)
            interior_area = int(np.count_nonzero(interior_component))
            if interior_area >= max(96, int(round(area * 0.35))) or (anchored and interior_area > 0):
                kept[interior_component] = 255
                removed += max(0, area - interior_area)
                continue
            removed += area
            continue
        kept[component] = 255
    if removed and isinstance(block, dict):
        metrics = block.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["balloon_outline_components_removed"] = int(removed)
        _append_qa_flag(block, "balloon_outline_components_removed")
    return kept


def clip_text_mask_to_balloon_interior_preserving_raw(
    text_mask: np.ndarray,
    balloon_mask: np.ndarray | None,
    raw_floor_mask: np.ndarray | None,
    *,
    erode_px: int = 2,
    block: dict | None = None,
    source: str = "bubble_mask",
    min_raw_overlap: float = 0.92,
) -> np.ndarray:
    clipped = clip_text_mask_to_balloon_interior(text_mask, balloon_mask, erode_px=erode_px)
    if not (source == "derived_bubble_mask" and _dark_or_colored_text_card_context(block or {})):
        clipped = _drop_unanchored_balloon_outline_components(
            clipped,
            balloon_mask,
            block,
            erode_px=erode_px,
        )
    if not isinstance(raw_floor_mask, np.ndarray) or raw_floor_mask.shape[:2] != clipped.shape[:2]:
        return clipped
    raw_floor = np.where(raw_floor_mask > 0, 255, 0).astype(np.uint8)
    preserve_reference = raw_floor
    dropped_outline_pixels = 0
    if not (source == "derived_bubble_mask" and _dark_or_colored_text_card_context(block or {})):
        if isinstance(balloon_mask, np.ndarray) and balloon_mask.shape[:2] == clipped.shape[:2] and np.any(balloon_mask):
            balloon_binary = np.where(balloon_mask > 0, 255, 0).astype(np.uint8)
            safe_interior = safe_bubble_interior_mask(balloon_binary, erode_px=erode_px)
            if not np.any(safe_interior):
                safe_interior = balloon_binary
            candidate_preserve = cv2.bitwise_and(
                raw_floor,
                safe_interior.astype(np.uint8),
            )
            clipped_support = cv2.dilate(
                np.where(clipped > 0, 255, 0).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            narrow_support = _raw_floor_support_mask_for_block(block, raw_floor.shape[:2])
            if isinstance(narrow_support, np.ndarray) and np.any(narrow_support):
                clipped_support = cv2.bitwise_or(clipped_support, narrow_support.astype(np.uint8))
            preserve_reference = _anchored_raw_floor_components(candidate_preserve, clipped_support)
            raw_inside_balloon = cv2.bitwise_and(
                raw_floor,
                balloon_binary,
            )
            raw_outline = np.where((raw_inside_balloon > 0) & (safe_interior == 0), 255, 0).astype(np.uint8)
            anchored_outline = _anchored_raw_floor_components(
                raw_outline,
                _raw_floor_support_mask_for_block(block, raw_floor.shape[:2]),
            )
            if np.any(anchored_outline):
                preserve_reference = cv2.bitwise_or(preserve_reference, anchored_outline)
            dropped_outline_pixels = int(
                np.count_nonzero((raw_inside_balloon > 0) & (preserve_reference == 0))
            )
    raw_pixels = int(np.count_nonzero(preserve_reference))
    if raw_pixels <= 0:
        return clipped
    overlap = int(np.count_nonzero((clipped > 0) & (preserve_reference > 0)))
    overlap_ratio = overlap / float(max(1, raw_pixels))
    if overlap >= raw_pixels:
        return clipped

    preserved = cv2.bitwise_or(clipped.astype(np.uint8), preserve_reference)
    if not (source == "derived_bubble_mask" and _dark_or_colored_text_card_context(block or {})):
        preserved = _drop_unanchored_balloon_outline_components(
            preserved,
            balloon_mask,
            block,
            erode_px=erode_px,
        )
    if isinstance(block, dict):
        metrics = block.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["balloon_clip_raw_preservation"] = {
                "source": source,
                "raw_pixels": raw_pixels,
                "clipped_raw_overlap_pixels": overlap,
                "clipped_raw_overlap_ratio": round(float(overlap_ratio), 6),
                "min_raw_overlap": round(float(min_raw_overlap), 6),
                "clipped_pixels": int(np.count_nonzero(clipped)),
                "preserved_pixels": int(np.count_nonzero(preserved)),
                "dropped_outline_raw_pixels": dropped_outline_pixels,
            }
        _append_qa_flag(block, "bubble_clip_preserved_raw_text")
    return preserved.astype(np.uint8)


def choose_balloon_component(
    *,
    text_bbox: list[int],
    candidates: list[list[int]],
    page_size: tuple[int, int],
    min_margin_px: int = 8,
) -> list[int] | None:
    """Pick a candidate component only when it actually surrounds the text anchor."""

    del page_size
    text = _normalize_bbox(text_bbox, 10**9, 10**9)
    if not text:
        return None
    tx1, ty1, tx2, ty2 = text
    tcx = (tx1 + tx2) / 2.0
    tcy = (ty1 + ty2) / 2.0
    best: tuple[float, list[int]] | None = None
    for candidate in candidates or []:
        bbox = _normalize_bbox(candidate, 10**9, 10**9)
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        contains_center = x1 <= tcx <= x2 and y1 <= tcy <= y2
        overlap = _bbox_intersection_area(text, bbox)
        text_area = max(1, _bbox_area(text))
        if not contains_center or overlap / float(text_area) < 0.55:
            continue
        if (x2 - x1) < max(1, (tx2 - tx1) * 0.50):
            continue
        if (y2 - y1) < max(1, (ty2 - ty1) * 0.50):
            continue
        area = _bbox_area(bbox)
        margin_bonus = 0
        if x1 <= tx1 - min_margin_px and x2 >= tx2 + min_margin_px:
            margin_bonus += 1
        if y1 <= ty1 - min_margin_px and y2 >= ty2 + min_margin_px:
            margin_bonus += 1
        score = overlap / float(max(1, area)) + margin_bonus * 0.05
        if best is None or score > best[0]:
            best = (score, bbox)
    return list(best[1]) if best else None


def _text_geometry_bbox(block: dict, image_shape: tuple[int, ...]) -> list[int] | None:
    height, width = _image_hw(image_shape)
    polygons = _text_geometry_polygons(block, width, height)
    if polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        bbox = _bbox_from_mask(mask)
        if bbox:
            return bbox
    return _normalize_bbox(block.get("text_pixel_bbox"), width, height)


def _expand_bbox_px(bbox: list[int], width: int, height: int, pad_x: int, pad_y: int) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    return _normalize_bbox([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def _text_geometry_has_white_context(
    block: dict,
    image_rgb: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> bool:
    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return False
    height, width = _image_hw(image_shape)
    if image_rgb.shape[0] < height or image_rgb.shape[1] < width:
        return False
    geometry_bbox = _text_geometry_bbox(block, image_shape)
    if not geometry_bbox:
        return False
    expanded = _expand_bbox_px(geometry_bbox, width, height, pad_x=8, pad_y=8)
    if not expanded:
        return False
    x1, y1, x2, y2 = expanded
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.astype(np.uint8)
    if crop.ndim == 3:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        bright = (gray >= 220) & (value >= 220) & (saturation <= 70)
    else:
        bright = gray >= 220
    bright_ratio = float(np.mean(bright)) if bright.size else 0.0
    if bright_ratio < 0.48:
        return False
    bright_pixels = gray[bright]
    if bright_pixels.size < 24:
        return False
    return float(np.percentile(bright_pixels, 70)) >= 228.0


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return mask
    binary = (mask > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = padded.shape[:2]
    flood = padded.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def _edge_contact_ratio(mask: np.ndarray) -> float:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return 0.0
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0
    contacts = [
        float(np.count_nonzero(mask[0, :] > 0)) / float(max(1, w)),
        float(np.count_nonzero(mask[-1, :] > 0)) / float(max(1, w)),
        float(np.count_nonzero(mask[:, 0] > 0)) / float(max(1, h)),
        float(np.count_nonzero(mask[:, -1] > 0)) / float(max(1, h)),
    ]
    return max(contacts)


def _dark_border_evidence(crop_rgb: np.ndarray, local_mask: np.ndarray) -> bool:
    if not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0:
        return False
    if not isinstance(local_mask, np.ndarray) or local_mask.size == 0 or not np.any(local_mask):
        return False
    gray = cv2.cvtColor(crop_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb.astype(np.uint8)
    ys, xs = np.where(local_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return False
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = 3
    bands: list[np.ndarray] = []
    if y1 > 0:
        bands.append(gray[max(0, y1 - pad) : y1 + 1, max(0, x1 - pad) : min(gray.shape[1], x2 + pad)])
    else:
        bands.append(gray[: min(gray.shape[0], pad + 1), :])
    if y2 < gray.shape[0]:
        bands.append(gray[max(0, y2 - 1) : min(gray.shape[0], y2 + pad), max(0, x1 - pad) : min(gray.shape[1], x2 + pad)])
    else:
        bands.append(gray[max(0, gray.shape[0] - pad - 1) :, :])
    if x1 > 0:
        bands.append(gray[max(0, y1 - pad) : min(gray.shape[0], y2 + pad), max(0, x1 - pad) : x1 + 1])
    else:
        bands.append(gray[:, : min(gray.shape[1], pad + 1)])
    if x2 < gray.shape[1]:
        bands.append(gray[max(0, y1 - pad) : min(gray.shape[0], y2 + pad), max(0, x2 - 1) : min(gray.shape[1], x2 + pad)])
    else:
        bands.append(gray[:, max(0, gray.shape[1] - pad - 1) :])
    evidence = 0
    for band in bands:
        if band.size == 0:
            continue
        dark_ratio = float(np.count_nonzero(band < 96)) / float(band.size)
        if dark_ratio >= 0.04:
            evidence += 1
    return evidence >= 2


def _classify_derived_bubble_mask(crop_rgb: np.ndarray, local_mask: np.ndarray) -> tuple[str | None, str | None]:
    if not isinstance(local_mask, np.ndarray) or local_mask.size == 0 or not np.any(local_mask):
        return None, "empty_derived_mask"
    h, w = local_mask.shape[:2]
    area = int(np.count_nonzero(local_mask > 0))
    bbox_area = max(1, h * w)
    density = float(area) / float(bbox_area)
    ys, xs = np.where(local_mask > 0)
    component_density = 0.0
    if len(xs) > 0 and len(ys) > 0:
        component_area = max(1, (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1))
        component_density = float(area) / float(component_area)
    if component_density >= 0.92 and _dark_border_evidence(crop_rgb, local_mask):
        return "derived_rectangular_balloon", None
    edge_ratio = _edge_contact_ratio(local_mask)
    if density >= 0.92 and edge_ratio >= 0.92:
        return None, "rejected_rectangular_crop"
    return "image_white_region", None


def _expanded_bbox(
    bbox: list[int],
    width: int,
    height: int,
    scale: float = 1.0,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    base_pad = max(6, int(min(w, h) * 0.35))
    x_pad = max(6, int(base_pad * max(1.0, float(scale))))
    y_pad = max(6, int(base_pad * max(1.0, float(scale))))
    return [
        max(0, x1 - x_pad),
        max(0, y1 - y_pad),
        min(width, x2 + x_pad),
        min(height, y2 + y_pad),
    ]


def _component_fill_ratio(mask: np.ndarray) -> float:
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return 0.0
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    area = int(np.count_nonzero(mask))
    component_box_area = max(1, (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1))
    return area / float(component_box_area)


def _mask_bbox_touches_crop_edge(mask: np.ndarray, edge_px: int = 2) -> bool:
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return False
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox
    return x1 <= edge_px or y1 <= edge_px or x2 >= w - edge_px or y2 >= h - edge_px


def _bbox_from_bool_mask(mask: np.ndarray) -> list[int] | None:
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return None
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _ellipse_model_for_crop(crop_shape: tuple[int, ...], local_support: np.ndarray | None) -> np.ndarray | None:
    h, w = crop_shape[:2]
    if h < 24 or w < 24:
        return None
    model = np.zeros((h, w), dtype=np.uint8)
    axes = (max(1, int(round(w * 0.47))), max(1, int(round(h * 0.47))))
    cv2.ellipse(model, (w // 2, h // 2), axes, 0, 0, 360, 255, -1)
    if isinstance(local_support, np.ndarray) and local_support.shape[:2] == (h, w) and np.any(local_support):
        support_bbox = _bbox_from_bool_mask(local_support > 0)
        if support_bbox:
            sx1, sy1, sx2, sy2 = support_bbox
            support_area = max(1, (sx2 - sx1) * (sy2 - sy1))
            support_hit = int(np.count_nonzero(model[sy1:sy2, sx1:sx2] > 0))
            if support_hit / float(support_area) < 0.45:
                return None
    return model if np.any(model) else None


def _derive_outline_contour_bubble_mask(
    crop_rgb: np.ndarray,
    local_support: np.ndarray | None,
) -> np.ndarray | None:
    if not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0:
        return None
    if crop_rgb.ndim == 3:
        gray = cv2.cvtColor(crop_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = crop_rgb.astype(np.uint8)
    h, w = gray.shape[:2]
    if h < 18 or w < 18:
        return None

    dark = (gray <= 88).astype(np.uint8) * 255
    if isinstance(local_support, np.ndarray) and local_support.shape[:2] == dark.shape[:2]:
        support_bbox = _bbox_from_bool_mask(local_support > 0)
        if support_bbox:
            sx1, sy1, sx2, sy2 = support_bbox
            # OCR glyphs are dark too. Suppress them so the contour stage sees
            # the balloon outline, not the text itself.
            dark[max(0, sy1 - 2) : min(h, sy2 + 2), max(0, sx1 - 2) : min(w, sx2 + 2)] = 0
    else:
        support_bbox = None

    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    contours, _hierarchy = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    support_cx = support_cy = None
    if support_bbox:
        support_cx = (support_bbox[0] + support_bbox[2]) / 2.0
        support_cy = (support_bbox[1] + support_bbox[3]) / 2.0

    best: tuple[float, np.ndarray] | None = None
    crop_area = max(1, h * w)
    for contour in contours:
        if len(contour) < 5:
            continue
        area = float(cv2.contourArea(contour))
        if area < max(160.0, crop_area * 0.015):
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 24 or ch < 18:
            continue
        if support_cx is not None and support_cy is not None:
            if not (x - 8 <= support_cx <= x + cw + 8 and y - 8 <= support_cy <= y + ch + 8):
                continue
        score = area
        if support_bbox:
            sx1, sy1, sx2, sy2 = support_bbox
            overlap = max(0, min(x + cw, sx2) - max(x, sx1)) * max(0, min(y + ch, sy2) - max(y, sy1))
            score += overlap * 20.0
        if best is None or score > best[0]:
            best = (score, contour)
    if best is None:
        return None

    contour = best[1]
    model = np.zeros((h, w), dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (axis_a, axis_b), angle = ellipse
    axes = (
        max(1, int(round(axis_a * 0.54))),
        max(1, int(round(axis_b * 0.54))),
    )
    cv2.ellipse(
        model,
        (int(round(cx)), int(round(cy))),
        axes,
        float(angle),
        0,
        360,
        255,
        -1,
    )
    if support_bbox:
        sx1, sy1, sx2, sy2 = support_bbox
        support_area = max(1, (sx2 - sx1) * (sy2 - sy1))
        support_hit = int(np.count_nonzero(model[sy1:sy2, sx1:sx2] > 0))
        if support_hit / float(support_area) < 0.35:
            return None
    return model if np.any(model) else None


def _should_model_connected_white_crop_as_ellipse(
    block: dict,
    crop_rgb: np.ndarray,
    local_mask: np.ndarray,
    local_support: np.ndarray | None,
    shape_kind: str,
    source: str | None,
) -> bool:
    if source != "image_white_region" or shape_kind != "rectangle":
        return False
    if not _mask_bbox_touches_crop_edge(local_mask, edge_px=3):
        return False
    if _dark_border_evidence(crop_rgb, local_mask):
        return False
    support_bbox = _bbox_from_bool_mask(local_support > 0) if isinstance(local_support, np.ndarray) and np.any(local_support) else None
    if support_bbox is None:
        return False
    crop_area = max(1, int(crop_rgb.shape[0]) * int(crop_rgb.shape[1]))
    support_area = max(1, (support_bbox[2] - support_bbox[0]) * (support_bbox[3] - support_bbox[1]))
    if crop_area / float(support_area) > 16.0:
        return False
    background = _rgb_luma_chroma(block.get("background_rgb"))
    if background is not None:
        luma, chroma = background
        if luma < 238.0 or chroma > 28.0:
            return False
    profile = {
        str(block.get("layout_profile") or "").strip().lower(),
        str(block.get("block_profile") or "").strip().lower(),
        str(block.get("balloon_type") or "").strip().lower(),
    }
    return bool(profile & {"white", "white_balloon", "dialogue_balloon", ""})


def _card_like_dark_context(block: dict) -> bool:
    if bool(block.get("card_panel_text_context")):
        return True
    background = _rgb_luma_chroma(block.get("background_rgb"))
    if background is not None:
        luma, chroma = background
        if luma < 125.0 or chroma > 42.0:
            return True
    profile = {
        str(block.get("layout_profile") or "").strip().lower(),
        str(block.get("block_profile") or "").strip().lower(),
        str(block.get("render_profile") or "").strip().lower(),
    }
    return bool(profile & {"dark_panel", "colored_status_panel", "status_panel", "card", "title_card"})


def _dark_text_without_balloon_context(block: dict) -> bool:
    evidence = block.get("dark_light_text_evidence")
    if not isinstance(evidence, dict) or evidence.get("has_balloon") is not False:
        return False
    background = _rgb_luma_chroma(block.get("background_rgb"))
    if background is not None and float(background[0]) < 80.0:
        return True
    return str(block.get("background_polarity") or "").strip().lower() == "dark"


def _dark_bubble_source_should_remain_rect_panel(block: dict) -> bool:
    if not isinstance(block, dict):
        return False
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    if source != "image_dark_bubble_mask":
        return False
    flags = {str(flag).strip() for flag in block.get("qa_flags") or [] if str(flag).strip()}
    lower_flags = {flag.lower() for flag in flags}
    if bool(block.get("card_panel_text_context")):
        return True
    profiles = {
        str(block.get("layout_profile") or "").strip().lower(),
        str(block.get("block_profile") or "").strip().lower(),
        str(block.get("render_profile") or "").strip().lower(),
    }
    if profiles & {"dark_panel", "colored_status_panel", "status_panel", "card", "title_card"}:
        return True
    if "connected_layout_disabled_dark_panel_visual_mask" in flags:
        return True
    metrics = block.get("qa_metrics") if isinstance(block.get("qa_metrics"), dict) else {}
    rejected = metrics.get("image_dark_panel_mask_rejected") if isinstance(metrics, dict) else None
    if isinstance(rejected, list):
        for item in rejected:
            if isinstance(item, dict) and str(item.get("reason") or "") == "overbroad_against_balloon_bbox":
                return True
    if (
        "dark_bubble_ellipse_bbox_mask" in lower_flags
        or "dark_bubble_visual_bbox_refined" in lower_flags
        or isinstance(block.get("bubble_mask_ellipse") or block.get("bubbleMaskEllipse"), dict)
    ):
        return False
    return False


def _centered_capped_dark_panel_bbox(
    block: dict,
    *,
    width: int,
    height: int,
) -> list[int] | None:
    panel_bbox = _normalize_bbox(block.get("bubble_mask_bbox") or block.get("balloon_bbox"), width, height)
    anchor_bbox = _normalize_bbox(block.get("text_pixel_bbox") or block.get("source_bbox") or block.get("bbox"), width, height)
    if not panel_bbox or not anchor_bbox:
        return panel_bbox
    px1, py1, px2, py2 = panel_bbox
    ax1, ay1, ax2, ay2 = anchor_bbox
    cx = (ax1 + ax2) / 2.0
    cy = (ay1 + ay2) / 2.0
    text_half_w = max(1.0, (ax2 - ax1) / 2.0)
    text_half_h = max(1.0, (ay2 - ay1) / 2.0)
    current_half_w = max(abs(cx - px1), abs(px2 - cx))
    current_half_h = max(abs(cy - py1), abs(py2 - cy))
    max_half_w = max(float(DARK_PANEL_RECT_MAX_HALF_WIDTH_FROM_TEXT_CENTER), text_half_w + 8.0)
    max_half_h = max(float(DARK_PANEL_RECT_MAX_HALF_HEIGHT_FROM_TEXT_CENTER), text_half_h + 8.0)
    half_w = min(current_half_w, max_half_w)
    half_h = min(current_half_h, max_half_h)
    return _normalize_bbox(
        [
            int(np.floor(cx - half_w)),
            int(np.floor(cy - half_h)),
            int(np.ceil(cx + half_w)),
            int(np.ceil(cy + half_h)),
        ],
        width,
        height,
    )


def _promote_rect_dark_panel_contract(block: dict, *, width: int, height: int) -> None:
    if not isinstance(block, dict):
        return
    block["bubble_mask_source"] = "image_dark_panel_mask"
    block.pop("bubbleMaskSource", None)
    block.pop("bubble_mask_shape", None)
    block.pop("bubble_mask_ellipse", None)
    block.pop("bubbleMaskEllipse", None)
    block["block_profile"] = "dark_panel"
    block["layout_profile"] = "dark_panel"
    flags = [
        str(flag)
        for flag in block.get("qa_flags") or []
        if str(flag) and str(flag) != "dark_bubble_ellipse_bbox_mask"
    ]
    if "dark_panel_rect_from_dark_bubble_bbox" not in flags:
        flags.append("dark_panel_rect_from_dark_bubble_bbox")
    block["qa_flags"] = flags
    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        bbox = _centered_capped_dark_panel_bbox(block, width=width, height=height)
        if bbox is not None:
            block["bubble_mask_bbox"] = list(bbox)
            block["balloon_bbox"] = list(bbox)
        anchor = _normalize_bbox(block.get("text_pixel_bbox") or block.get("source_bbox") or block.get("bbox"), width, height)
        metrics["image_dark_panel_mask"] = {
            "source": "image_dark_panel_mask",
            "detection_space": "dark_bubble_bbox_rect_panel_contract",
            "mask_bbox": list(bbox) if bbox is not None else None,
            "anchor_bbox": list(anchor) if anchor is not None else None,
            "centered_on_text": bool(bbox is not None and anchor is not None),
            "max_half_width_from_text_center": DARK_PANEL_RECT_MAX_HALF_WIDTH_FROM_TEXT_CENTER,
            "max_half_height_from_text_center": DARK_PANEL_RECT_MAX_HALF_HEIGHT_FROM_TEXT_CENTER,
        }


def _dark_text_without_balloon_fast_fill_reject_reasons(block: dict, image_shape: tuple[int, ...]) -> list[str]:
    if not _dark_text_without_balloon_context(block):
        return []
    height, width = _image_hw(image_shape)
    bbox = _normalize_bbox(block.get("text_pixel_bbox") or block.get("source_bbox") or block.get("bbox"), width, height)
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    if x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height:
        return ["dark_text_without_balloon_touches_art_border"]
    return []


def _derive_card_panel_mask_from_image(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None,
    support_mask: np.ndarray | None = None,
    *,
    allow_inferred_card: bool = False,
    ) -> np.ndarray | None:
    if "short_dark_text_full_panel_bbox_rejected" in {str(flag).strip() for flag in block.get("qa_flags") or []}:
        metrics = block.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["derived_card_panel_mask_rejected"] = {
                "reason": "short_dark_text_full_panel_bbox_rejected",
            }
        return None
    if not _card_like_dark_context(block) and not allow_inferred_card:
        return None
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None
    height, width = _image_hw(image_shape)
    bbox = (
        _normalize_bbox(block.get("balloon_bbox"), width, height)
        or _normalize_bbox(block.get("target_bbox"), width, height)
        or _normalize_bbox(block.get("safe_text_box"), width, height)
    )
    anchor = _text_geometry_bbox(block, image_shape) or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if not bbox or not anchor:
        return None
    x1, y1, x2, y2 = bbox
    ax1, ay1, ax2, ay2 = anchor
    overflow = max(
        0,
        x1 - ax1,
        y1 - ay1,
        ax2 - x2,
        ay2 - y2,
    )
    if overflow:
        max_small_overflow = max(12, int(round(max(ax2 - ax1, ay2 - ay1) * 0.18)))
        if overflow > max_small_overflow:
            return None
        x1 = max(0, min(x1, ax1 - 6))
        y1 = max(0, min(y1, ay1 - 6))
        x2 = min(width, max(x2, ax2 + 6))
        y2 = min(height, max(y2, ay2 + 6))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    if bw < max(aw + 8, 28) or bh < max(ah + 8, 22):
        return None
    if _bbox_min_overlap_ratio(bbox, anchor) < 0.70:
        return None
    page_area = max(1, width * height)
    bbox_area = max(1, bw * bh)
    anchor_area = max(1, aw * ah)
    profile = {
        str(block.get("layout_profile") or "").strip().lower(),
        str(block.get("block_profile") or "").strip().lower(),
        str(block.get("render_profile") or "").strip().lower(),
    }
    if bbox_area > int(page_area * 0.45) and not (profile & {"dark_panel", "colored_status_panel", "status_panel", "card", "title_card"}):
        return None
    if bbox_area > max(int(page_area * 0.18), anchor_area * 10):
        return None
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.ndim == 3:
        crop_rgb = crop.astype(np.uint8)
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        chroma = np.max(crop_rgb.astype(np.int16), axis=2) - np.min(crop_rgb.astype(np.int16), axis=2)
    else:
        gray = crop.astype(np.uint8)
        chroma = np.zeros_like(gray, dtype=np.int16)
    local_support = None
    if isinstance(support_mask, np.ndarray) and support_mask.shape[:2] == (height, width) and np.any(support_mask):
        local_support = support_mask[y1:y2, x1:x2] > 0
    if local_support is None or not np.any(local_support):
        local_support = np.zeros((bh, bw), dtype=bool)
        sx1 = max(0, ax1 - x1)
        sy1 = max(0, ay1 - y1)
        sx2 = min(bw, ax2 - x1)
        sy2 = min(bh, ay2 - y1)
        if sx2 > sx1 and sy2 > sy1:
            local_support[sy1:sy2, sx1:sx2] = True
    if local_support is None or not np.any(local_support):
        return None
    active = local_support
    background = _rgb_luma_chroma(block.get("background_rgb"))
    crop_luma = float(np.median(gray))
    crop_chroma = float(np.median(chroma))
    if background is not None:
        bg_luma, bg_chroma = background
    else:
        bg_luma, bg_chroma = crop_luma, crop_chroma
    card_like_pixels = bool(bg_luma < 210.0 or bg_chroma > 38.0 or crop_luma < 210.0 or crop_chroma > 30.0)
    if not card_like_pixels:
        return None
    support_pixels = int(np.count_nonzero(active))
    if support_pixels <= 0:
        return None
    support_bbox = _bbox_from_bool_mask(active)
    if not support_bbox:
        return None
    sx1, sy1, sx2, sy2 = support_bbox
    margin_ok = (
        sx1 >= 2
        and sy1 >= 2
        and sx2 <= bw - 2
        and sy2 <= bh - 2
    )
    if not margin_ok:
        return None
    mask_x1, mask_y1, mask_x2, mask_y2 = x1, y1, x2, y2
    if background is not None and (bg_luma < 210.0 or bg_chroma > 38.0):
        panel_pad = 6
        mask_x1 = max(0, mask_x1 - panel_pad)
        mask_y1 = max(0, mask_y1 - panel_pad)
        mask_x2 = min(width, mask_x2 + panel_pad)
        mask_y2 = min(height, mask_y2 + panel_pad)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        derived_metrics = {
            "source": "derived_card_panel_mask",
            "mask_bbox": [mask_x1, mask_y1, mask_x2, mask_y2],
            "mask_pixels": int(np.count_nonzero(mask)),
            "anchor_bbox": [ax1, ay1, ax2, ay2],
            "background_luma": round(float(bg_luma), 3),
            "background_chroma": round(float(bg_chroma), 3),
        }
        derived_metrics.update(_sample_dark_panel_effect_colors(image_rgb.astype(np.uint8), [mask_x1, mask_y1, mask_x2, mask_y2], support_mask))
        metrics["derived_card_panel_mask"] = derived_metrics
    block["bubble_mask_source"] = "derived_card_panel_mask"
    block["bubble_mask_bbox"] = [mask_x1, mask_y1, mask_x2, mask_y2]
    block["dark_panel_effect_colors"] = _sample_dark_panel_effect_colors(
        image_rgb.astype(np.uint8),
        [mask_x1, mask_y1, mask_x2, mask_y2],
        support_mask,
    )
    _remove_qa_flags(
        block,
        {
            "bbox_fallback_bubble_mask",
            "debug_derived_bubble_mask_rejected",
            "derived_bubble_mask_rejected",
            "missing_real_bubble_mask",
            "rejected_derived_bubble_mask",
        },
    )
    return mask


def _median_rgb_from_mask(image_rgb: np.ndarray, mask: np.ndarray) -> list[int] | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        return None
    if not isinstance(mask, np.ndarray) or mask.shape[:2] != image_rgb.shape[:2] or not np.any(mask):
        return None
    pixels = image_rgb[mask > 0]
    if pixels.size == 0:
        return None
    return [int(v) for v in np.median(pixels, axis=0).astype(int).tolist()]


def _sample_dark_panel_effect_colors(
    image_rgb: np.ndarray,
    panel_bbox: list[int],
    support_mask: np.ndarray | None,
) -> dict:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in panel_bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return {}
    luma = (
        image_rgb[:, :, 0].astype(np.float32) * 0.299
        + image_rgb[:, :, 1].astype(np.float32) * 0.587
        + image_rgb[:, :, 2].astype(np.float32) * 0.114
    )
    panel = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(panel, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
    inner_pad = max(4, min(14, int(round(min(x2 - x1, y2 - y1) * 0.04))))
    inner = np.zeros_like(panel)
    cv2.rectangle(
        inner,
        (min(width - 1, x1 + inner_pad), min(height - 1, y1 + inner_pad)),
        (max(0, x2 - inner_pad - 1), max(0, y2 - inner_pad - 1)),
        255,
        -1,
    )
    dark_inner = np.where((inner > 0) & (luma <= 28.0), 255, 0).astype(np.uint8)
    border = np.zeros_like(panel)
    cv2.rectangle(border, (x1, y1), (x2 - 1, y2 - 1), 255, 3)
    border_pixels = image_rgb[border > 0]
    border_rgb = None
    if border_pixels.size:
        border_luma = (
            border_pixels[:, 0].astype(np.float32) * 0.299
            + border_pixels[:, 1].astype(np.float32) * 0.587
            + border_pixels[:, 2].astype(np.float32) * 0.114
        )
        selected = border_pixels[border_luma >= np.percentile(border_luma, 70)]
        border_rgb = [int(v) for v in np.median(selected if len(selected) else border_pixels, axis=0).astype(int).tolist()]
    ring_outer = np.zeros_like(panel)
    cv2.rectangle(
        ring_outer,
        (max(0, x1 - 45), max(0, y1 - 55)),
        (min(width - 1, x2 + 45), min(height - 1, y2 + 22)),
        255,
        -1,
    )
    ring_inner = np.zeros_like(panel)
    cv2.rectangle(
        ring_inner,
        (max(0, x1 - 4), max(0, y1 - 4)),
        (min(width - 1, x2 + 4), min(height - 1, y2 + 4)),
        255,
        -1,
    )
    yy = np.indices((height, width))[0]
    glow_mask = np.where((ring_outer > 0) & (ring_inner == 0) & (yy < y2 + 18), 255, 0).astype(np.uint8)
    glow_pixels = image_rgb[glow_mask > 0]
    glow_rgb = None
    if glow_pixels.size:
        gp = glow_pixels.astype(np.float32)
        glow_luma = gp[:, 0] * 0.299 + gp[:, 1] * 0.587 + gp[:, 2] * 0.114
        warm_score = gp[:, 0] + gp[:, 1] * 0.95 - gp[:, 2] * 0.55 + glow_luma * 0.35
        warm = (glow_luma > 18.0) & (gp[:, 0] > gp[:, 2] * 0.90) & (gp[:, 1] > gp[:, 2] * 0.75)
        if np.any(warm):
            selected = glow_pixels[warm & (warm_score >= np.percentile(warm_score[warm], 70))]
        else:
            selected = glow_pixels[warm_score >= np.percentile(warm_score, 90)]
        glow_rgb = [int(v) for v in np.median(selected if len(selected) else glow_pixels, axis=0).astype(int).tolist()]

    support = None
    if isinstance(support_mask, np.ndarray) and support_mask.shape[:2] == (height, width) and np.any(support_mask):
        support = np.where((support_mask > 0) & (panel > 0), 255, 0).astype(np.uint8)
    text_fill_rgb = None
    text_glow_rgb = None
    bad_negative_text_glow_rgb = None
    if isinstance(support, np.ndarray) and np.any(support):
        support_dilated = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        text_pixels = image_rgb[(support > 0) & (luma >= 185.0)]
        if text_pixels.size:
            text_fill_rgb = [int(v) for v in np.median(text_pixels, axis=0).astype(int).tolist()]
        glow_support = (support_dilated > 0) & (support == 0) & (panel > 0) & (luma > 35.0) & (luma < 185.0)
        glow_support &= image_rgb[:, :, 0] >= image_rgb[:, :, 2] * 0.85
        glow_support &= image_rgb[:, :, 1] >= image_rgb[:, :, 2] * 0.70
        glow_text_pixels = image_rgb[glow_support]
        if glow_text_pixels.size:
            text_glow_rgb = [int(v) for v in np.median(glow_text_pixels, axis=0).astype(int).tolist()]
            negative = 255 - image_rgb
            bad_negative = negative[glow_support]
            bad_negative_text_glow_rgb = [int(v) for v in np.median(bad_negative, axis=0).astype(int).tolist()]
    sampled: dict[str, object] = {
        "color_sample_space": "original_image",
        "panel_fill_rgb": _median_rgb_from_mask(image_rgb, dark_inner) or _median_rgb_from_mask(image_rgb, inner),
        "border_rgb": border_rgb,
        "panel_glow_rgb": glow_rgb,
        "text_fill_rgb": text_fill_rgb,
        "text_glow_rgb": text_glow_rgb,
        "bad_negative_text_glow_rgb": bad_negative_text_glow_rgb,
    }
    return {key: value for key, value in sampled.items() if value is not None}


def _dark_panel_negative_bbox_overbroad_reason(
    bbox: list[int],
    anchor: list[int],
    reference_bbox: list[int] | None,
) -> str | None:
    if not bbox or not anchor or reference_bbox is None:
        return None
    bbox_area = max(1, _bbox_area(bbox))
    ref_area = max(1, _bbox_area(reference_bbox))
    anchor_area = max(1, _bbox_area(anchor))
    if ref_area < max(int(anchor_area * 1.25), anchor_area + 1800):
        return None
    bw = max(1, int(bbox[2]) - int(bbox[0]))
    bh = max(1, int(bbox[3]) - int(bbox[1]))
    rw = max(1, int(reference_bbox[2]) - int(reference_bbox[0]))
    rh = max(1, int(reference_bbox[3]) - int(reference_bbox[1]))
    aw = max(1, int(anchor[2]) - int(anchor[0]))
    ah = max(1, int(anchor[3]) - int(anchor[1]))
    ref_overlap = _bbox_intersection_area(bbox, reference_bbox) / float(ref_area)
    anchor_overlap = _bbox_intersection_area(bbox, anchor) / float(anchor_area)
    if ref_overlap < 0.80 or anchor_overlap < 0.90:
        return None

    area_vs_ref = bbox_area / float(ref_area)
    area_vs_anchor = bbox_area / float(anchor_area)
    width_vs_ref = bw / float(rw)
    height_vs_ref = bh / float(rh)
    width_vs_anchor = bw / float(aw)
    height_vs_anchor = bh / float(ah)
    side_overflow = max(
        0,
        int(reference_bbox[0]) - int(bbox[0]),
        int(reference_bbox[1]) - int(bbox[1]),
        int(bbox[2]) - int(reference_bbox[2]),
        int(bbox[3]) - int(reference_bbox[3]),
    )
    if (
        area_vs_ref >= 4.2
        or (area_vs_ref >= 3.2 and width_vs_ref >= 2.35)
        or (area_vs_anchor >= 9.0 and width_vs_anchor >= 3.2 and height_vs_anchor >= 1.55)
    ) and side_overflow >= max(48, int(round(max(rw, rh) * 0.20))):
        return "overbroad_against_balloon_bbox"
    return None


def _derive_dark_panel_mask_from_negative_geometry(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None,
    support_mask: np.ndarray | None = None,
    *,
    allow_inferred_card: bool = False,
) -> np.ndarray | None:
    inferred_from_dark_bubble = bool(allow_inferred_card and not _card_like_dark_context(block))
    if not _card_like_dark_context(block) and not allow_inferred_card:
        return None
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return None
    height, width = _image_hw(image_shape)
    if image_rgb.shape[:2] != (height, width):
        return None
    anchor = _text_geometry_bbox(block, image_shape) or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if not anchor:
        return None
    ax1, ay1, ax2, ay2 = anchor
    anchor_cx = (ax1 + ax2) / 2.0
    anchor_cy = (ay1 + ay2) / 2.0
    gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    negative_gray = 255 - gray
    edges = cv2.bitwise_or(cv2.Canny(gray, 40, 120), cv2.Canny(negative_gray, 40, 120))
    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, list[int], dict] | None = None
    anchor_area = max(1, _bbox_area(anchor))
    page_area = max(1, width * height)
    reference_bbox = _normalize_bbox(block.get("balloon_bbox"), width, height)
    rejected: list[dict] = []
    for contour in contours:
        x, y, cw, ch = [int(v) for v in cv2.boundingRect(contour)]
        if cw <= 0 or ch <= 0:
            continue
        bbox = [x, y, x + cw, y + ch]
        area = cw * ch
        if area < max(2500, int(anchor_area * 1.8)):
            continue
        if area > int(page_area * 0.55):
            continue
        aspect = cw / float(max(1, ch))
        if aspect < 1.05 or aspect > 4.2:
            continue
        if not (x - 12 <= anchor_cx <= x + cw + 12 and y - 12 <= anchor_cy <= y + ch + 12):
            continue
        if _bbox_min_overlap_ratio(bbox, anchor) < 0.55:
            continue
        roi = gray[y : y + ch, x : x + cw]
        if roi.size == 0 or float(np.median(roi)) > 92.0:
            continue
        border = np.zeros((ch, cw), dtype=bool)
        t = max(2, min(8, int(round(min(cw, ch) * 0.025))))
        border[:t, :] = True
        border[-t:, :] = True
        border[:, :t] = True
        border[:, -t:] = True
        border_luma = float(np.mean(roi[border])) if np.any(border) else 0.0
        inner = roi[t : ch - t, t : cw - t] if ch > 2 * t and cw > 2 * t else roi
        inner_luma = float(np.median(inner)) if inner.size else 0.0
        contrast = border_luma - inner_luma
        if contrast < 18.0:
            continue
        overbroad_reason = _dark_panel_negative_bbox_overbroad_reason(bbox, anchor, reference_bbox)
        if overbroad_reason:
            rejected.append(
                {
                    "reason": overbroad_reason,
                    "mask_bbox": bbox,
                    "anchor_bbox": [ax1, ay1, ax2, ay2],
                    "reference_bbox": list(reference_bbox) if reference_bbox is not None else None,
                    "mask_pixels": int(area),
                    "border_inner_luma_delta": round(contrast, 3),
                }
            )
            continue
        center_penalty = abs((x + cw / 2.0) - anchor_cx) + abs((y + ch / 2.0) - anchor_cy)
        score = area + contrast * 5000.0 - center_penalty * 50.0
        metrics = {
            "source": "image_dark_panel_mask",
            "detection_space": "negative_geometry_inferred_dark_rect_panel"
            if inferred_from_dark_bubble
            else "negative_geometry",
            "mask_bbox": bbox,
            "mask_pixels": int(area),
            "anchor_bbox": [ax1, ay1, ax2, ay2],
            "border_luma": round(border_luma, 3),
            "inner_luma": round(inner_luma, 3),
            "border_inner_luma_delta": round(contrast, 3),
            "inferred_from_dark_bubble_candidate": inferred_from_dark_bubble,
        }
        if best is None or score > best[0]:
            best = (score, bbox, metrics)
    if rejected:
        qa_metrics = block.setdefault("qa_metrics", {})
        if isinstance(qa_metrics, dict):
            qa_metrics["image_dark_panel_mask_rejected"] = rejected[-6:]
    if best is None:
        return None
    _score, bbox, metrics = best
    x1, y1, x2, y2 = bbox
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    color_metrics = _sample_dark_panel_effect_colors(image_rgb.astype(np.uint8), bbox, support_mask)
    metrics.update(color_metrics)
    qa_metrics = block.setdefault("qa_metrics", {})
    if isinstance(qa_metrics, dict):
        qa_metrics["image_dark_panel_mask"] = metrics
    block["bubble_mask_source"] = "image_dark_panel_mask"
    block["bubble_mask_bbox"] = [x1, y1, x2, y2]
    block["dark_panel_effect_colors"] = color_metrics
    if inferred_from_dark_bubble:
        block["layout_profile"] = "dark_panel"
        block["block_profile"] = "dark_panel"
        _append_qa_flag(block, "dark_panel_rect_inferred_before_dark_ellipse")
    _remove_qa_flags(
        block,
        {
            "bbox_fallback_bubble_mask",
            "debug_derived_bubble_mask_rejected",
            "derived_bubble_mask_rejected",
            "missing_real_bubble_mask",
            "rejected_derived_bubble_mask",
            "dark_bubble_ellipse_bbox_mask",
        },
    )
    block.pop("bubble_mask_error", None)
    block.pop("bubbleMaskError", None)
    return mask


def _derive_dark_panel_mask_from_border_lines(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None,
    support_mask: np.ndarray | None = None,
    *,
    allow_inferred_card: bool = False,
) -> np.ndarray | None:
    inferred_from_dark_bubble = bool(allow_inferred_card and not _card_like_dark_context(block))
    if not _card_like_dark_context(block) and not allow_inferred_card:
        return None
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return None
    height, width = _image_hw(image_shape)
    if image_rgb.shape[:2] != (height, width):
        return None
    anchor = _text_geometry_bbox(block, image_shape) or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if not anchor:
        return None
    ax1, ay1, ax2, ay2 = anchor
    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    anchor_cx = (ax1 + ax2) / 2.0
    anchor_cy = (ay1 + ay2) / 2.0
    sx1 = max(0, ax1 - max(90, int(round(aw * 0.75))))
    sx2 = min(width, ax2 + max(90, int(round(aw * 0.75))))
    sy1 = max(0, ay1 - max(72, int(round(ah * 1.75))))
    sy2 = min(height, ay2 + max(72, int(round(ah * 1.75))))
    if sx2 - sx1 < max(96, aw + 32) or sy2 - sy1 < max(42, ah + 20):
        return None
    crop = image_rgb[sy1:sy2, sx1:sx2].astype(np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    median_luma = float(np.median(gray)) if gray.size else 0.0
    bright_threshold = max(78.0, median_luma + 36.0)
    rgb_crop = crop.astype(np.int16)
    chroma_span = np.max(rgb_crop, axis=2) - np.min(rgb_crop, axis=2)
    bright = ((gray.astype(np.float32) >= bright_threshold) & (chroma_span <= 96)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)),
        iterations=1,
    )
    min_run = max(96, int(round(aw * 1.28)))
    candidates: list[tuple[int, int, int, int]] = []
    for row_idx in range(bright.shape[0]):
        xs = np.where(bright[row_idx] > 0)[0]
        if xs.size == 0:
            continue
        start = int(xs[0])
        prev = int(xs[0])
        for value in [int(v) for v in xs[1:]] + [-1]:
            if value == prev + 1:
                prev = value
                continue
            run_w = prev - start + 1
            if run_w >= min_run:
                candidates.append((sy1 + row_idx, sx1 + start, sx1 + prev + 1, run_w))
            start = value
            prev = value
    if len(candidates) < 2:
        return None
    top_options = [c for c in candidates if c[0] <= anchor_cy - max(10, ah * 0.20)]
    bottom_options = [c for c in candidates if c[0] >= anchor_cy + max(10, ah * 0.20)]
    if not top_options or not bottom_options:
        return None

    best: tuple[float, list[int], dict] | None = None
    reference_bbox = _normalize_bbox(block.get("balloon_bbox"), width, height)
    for top in top_options:
        for bottom in bottom_options:
            ty, tx1, tx2, tw = top
            by, bx1, bx2, bw = bottom
            if by <= ty:
                continue
            panel_h = by - ty + 1
            if panel_h < max(34, int(round(ah * 1.28))) or panel_h > max(220, int(round(ah * 6.2))):
                continue
            overlap_x1 = max(tx1, bx1)
            overlap_x2 = min(tx2, bx2)
            if overlap_x2 - overlap_x1 < min_run:
                continue
            px1 = max(0, min(tx1, bx1) - 2)
            px2 = min(width, max(tx2, bx2) + 2)
            py1 = max(0, ty - 2)
            py2 = min(height, by + 3)
            bbox = [px1, py1, px2, py2]
            if not (px1 - 8 <= anchor_cx <= px2 + 8 and py1 - 8 <= anchor_cy <= py2 + 8):
                continue
            if _bbox_min_overlap_ratio(bbox, anchor) < 0.55:
                continue
            overbroad_reason = _dark_panel_negative_bbox_overbroad_reason(bbox, anchor, reference_bbox)
            if overbroad_reason and not allow_inferred_card:
                continue
            roi = cv2.cvtColor(image_rgb[py1:py2, px1:px2].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            if roi.size == 0:
                continue
            inset = max(3, min(10, int(round(min(px2 - px1, py2 - py1) * 0.06))))
            inner = roi[inset : roi.shape[0] - inset, inset : roi.shape[1] - inset] if roi.shape[0] > inset * 2 and roi.shape[1] > inset * 2 else roi
            inner_luma = float(np.median(inner)) if inner.size else float(np.median(roi))
            border_luma = float(np.median(np.concatenate([roi[:inset, :].ravel(), roi[-inset:, :].ravel()])))
            contrast = border_luma - inner_luma
            if inner_luma > 96.0 or contrast < 14.0:
                continue
            score = (px2 - px1) * (py2 - py1) + contrast * 4000.0 - abs(((py1 + py2) / 2.0) - anchor_cy) * 120.0
            metrics = {
                "source": "image_dark_panel_mask",
                "detection_space": "border_line_geometry_inferred_dark_rect_panel"
                if inferred_from_dark_bubble
                else "border_line_geometry",
                "mask_bbox": bbox,
                "mask_pixels": int((px2 - px1) * (py2 - py1)),
                "anchor_bbox": [ax1, ay1, ax2, ay2],
                "top_border_y": int(ty),
                "bottom_border_y": int(by),
                "border_luma": round(border_luma, 3),
                "inner_luma": round(inner_luma, 3),
                "border_inner_luma_delta": round(contrast, 3),
                "inferred_from_dark_bubble_candidate": bool(allow_inferred_card),
            }
            if best is None or score > best[0]:
                best = (score, bbox, metrics)
    if best is None:
        return None
    _score, bbox, metrics = best
    x1, y1, x2, y2 = bbox
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    color_metrics = _sample_dark_panel_effect_colors(image_rgb.astype(np.uint8), bbox, support_mask)
    metrics.update(color_metrics)
    qa_metrics = block.setdefault("qa_metrics", {})
    if isinstance(qa_metrics, dict):
        qa_metrics["image_dark_panel_mask"] = metrics
    block["bubble_mask_source"] = "image_dark_panel_mask"
    block["bubble_mask_bbox"] = [x1, y1, x2, y2]
    block["balloon_bbox"] = [x1, y1, x2, y2]
    block["dark_panel_effect_colors"] = color_metrics
    block["layout_profile"] = "dark_panel"
    block["block_profile"] = "dark_panel"
    _append_qa_flag(block, "dark_panel_rect_from_border_lines")
    if allow_inferred_card:
        _append_qa_flag(block, "dark_panel_rect_inferred_before_dark_ellipse")
    _remove_qa_flags(
        block,
        {
            "bbox_fallback_bubble_mask",
            "debug_derived_bubble_mask_rejected",
            "derived_bubble_mask_rejected",
            "missing_real_bubble_mask",
            "rejected_derived_bubble_mask",
            "dark_bubble_ellipse_bbox_mask",
        },
    )
    block.pop("bubble_mask_error", None)
    block.pop("bubbleMaskError", None)
    return mask


def _derive_dark_panel_mask_from_balloon_bbox(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None,
    support_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    if not _card_like_dark_context(block):
        return None
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return None
    height, width = _image_hw(image_shape)
    if image_rgb.shape[:2] != (height, width):
        return None
    bbox = _normalize_bbox(block.get("balloon_bbox") or block.get("bubble_mask_bbox"), width, height)
    anchor = _text_geometry_bbox(block, image_shape) or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if bbox is None or anchor is None:
        return None
    if _bbox_min_overlap_ratio(bbox, anchor) < 0.55:
        return None
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(bh)
    if aspect < 1.15 or aspect > 6.5:
        return None
    if _bbox_area(bbox) < int(max(1, _bbox_area(anchor)) * 1.12):
        return None
    rgb = image_rgb.astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    inner_pad = max(2, min(10, int(round(min(bw, bh) * 0.08))))
    inner = roi[inner_pad : bh - inner_pad, inner_pad : bw - inner_pad] if bw > inner_pad * 2 and bh > inner_pad * 2 else roi
    inner_luma = float(np.median(inner)) if inner.size else float(np.median(roi))
    if inner_luma > 118.0:
        return None
    border = np.zeros((bh, bw), dtype=bool)
    border_pad = max(2, min(8, int(round(min(bw, bh) * 0.04))))
    border[:border_pad, :] = True
    border[-border_pad:, :] = True
    border[:, :border_pad] = True
    border[:, -border_pad:] = True
    border_luma = float(np.median(roi[border])) if np.any(border) else inner_luma
    contrast = border_luma - inner_luma
    if contrast < 8.0 and not bool(block.get("card_panel_text_context")):
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    metrics = {
        "source": "image_dark_panel_mask",
        "detection_space": "balloon_bbox_dark_panel",
        "mask_bbox": [x1, y1, x2, y2],
        "mask_pixels": int(np.count_nonzero(mask)),
        "anchor_bbox": list(anchor),
        "border_luma": round(border_luma, 3),
        "inner_luma": round(inner_luma, 3),
        "border_inner_luma_delta": round(contrast, 3),
    }
    color_metrics = _sample_dark_panel_effect_colors(rgb, [x1, y1, x2, y2], support_mask)
    metrics.update(color_metrics)
    qa_metrics = block.setdefault("qa_metrics", {})
    if isinstance(qa_metrics, dict):
        qa_metrics["image_dark_panel_mask"] = metrics
    block["bubble_mask_source"] = "image_dark_panel_mask"
    block["bubble_mask_bbox"] = [x1, y1, x2, y2]
    block["dark_panel_effect_colors"] = color_metrics
    _remove_qa_flags(
        block,
        {
            "bbox_fallback_bubble_mask",
            "debug_derived_bubble_mask_rejected",
            "derived_bubble_mask_rejected",
            "missing_real_bubble_mask",
            "rejected_derived_bubble_mask",
            "dark_bubble_ellipse_bbox_mask",
        },
    )
    block.pop("bubble_mask_error", None)
    block.pop("bubbleMaskError", None)
    return mask


def _derive_dark_oval_bubble_mask_from_negative_geometry(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None,
    support_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    if not _card_like_dark_context(block):
        return None
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return None
    height, width = _image_hw(image_shape)
    if image_rgb.shape[:2] != (height, width):
        return None
    anchor = _text_geometry_bbox(block, image_shape) or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if not anchor:
        return None
    ax1, ay1, ax2, ay2 = anchor
    anchor_cx = (ax1 + ax2) / 2.0
    anchor_cy = (ay1 + ay2) / 2.0
    rgb = image_rgb.astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luma = gray.astype(np.float32)
    warm = (
        (rgb[:, :, 0].astype(np.float32) + rgb[:, :, 1].astype(np.float32) * 0.95 - rgb[:, :, 2].astype(np.float32) * 0.55)
        + luma * 0.20
    )
    warm_threshold = max(40.0, float(np.percentile(warm, 92)))
    warm_mask = np.where(
        (warm >= warm_threshold)
        & (luma >= 18.0)
        & (rgb[:, :, 0] >= rgb[:, :, 2] * 0.80)
        & (rgb[:, :, 1] >= rgb[:, :, 2] * 0.62),
        255,
        0,
    ).astype(np.uint8)
    edge = cv2.bitwise_or(cv2.Canny(gray, 28, 100), cv2.Canny(255 - gray, 28, 100))
    candidate_edges = cv2.bitwise_or(edge, warm_mask)
    ref_bbox = _normalize_bbox(block.get("balloon_bbox"), width, height) or anchor
    rx1, ry1, rx2, ry2 = ref_bbox
    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)
    search_x1 = max(0, min(rx1, ax1) - max(72, int(round(max(aw, rw) * 0.75))))
    search_x2 = min(width, max(rx2, ax2) + max(72, int(round(max(aw, rw) * 0.75))))
    search_y1 = max(0, min(ry1, ay1) - max(56, int(round(max(ah, rh) * 0.55))))
    search_y2 = min(height, max(ry2, ay2) + max(52, int(round(max(ah, rh) * 0.42))))
    window = np.zeros((height, width), dtype=np.uint8)
    window[search_y1:search_y2, search_x1:search_x2] = 255
    candidate_edges = cv2.bitwise_and(candidate_edges, window)
    candidate_edges = cv2.morphologyEx(
        candidate_edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=2,
    )
    contours, _ = cv2.findContours(candidate_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, list[int], tuple[tuple[float, float], tuple[float, float], float], dict] | None = None
    anchor_area = max(1, _bbox_area(anchor))
    page_area = max(1, width * height)
    for contour in contours:
        if len(contour) < 16:
            continue
        x, y, cw, ch = [int(v) for v in cv2.boundingRect(contour)]
        area = cw * ch
        if area < max(3500, int(anchor_area * 2.0)) or area > int(page_area * 0.58):
            continue
        aspect = cw / float(max(1, ch))
        if aspect < 1.10 or aspect > 3.4:
            continue
        bbox = [x, y, x + cw, y + ch]
        if not (x - 18 <= anchor_cx <= x + cw + 18 and y - 18 <= anchor_cy <= y + ch + 18):
            continue
        if _bbox_min_overlap_ratio(bbox, anchor) < 0.55:
            continue
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (axis_a, axis_b), angle = ellipse
        if axis_a <= 0 or axis_b <= 0:
            continue
        major = max(axis_a, axis_b)
        minor = min(axis_a, axis_b)
        if major < max(96.0, (ax2 - ax1) * 1.25) or minor < max(56.0, (ay2 - ay1) * 1.25):
            continue
        if not (x <= cx <= x + cw and y <= cy <= y + ch):
            continue
        probe = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            probe,
            (int(round(cx)), int(round(cy))),
            (max(1, int(round(axis_a / 2.0))), max(1, int(round(axis_b / 2.0)))),
            float(angle),
            0,
            360,
            255,
            -1,
        )
        if int(np.count_nonzero(probe[ay1:ay2, ax1:ax2] > 0)) / float(max(1, _bbox_area(anchor))) < 0.72:
            continue
        panel_luma = float(np.median(gray[probe > 0])) if np.any(probe) else 255.0
        if panel_luma > 72.0:
            continue
        border_pixels = int(np.count_nonzero((candidate_edges > 0) & (probe > 0)))
        score = border_pixels + area * 0.20 - (abs(cx - anchor_cx) + abs(cy - anchor_cy)) * 4.0
        metrics = {
            "source": "image_dark_bubble_mask",
            "shape_kind": "ellipse",
            "detection_space": "negative_geometry",
            "mask_bbox": bbox,
            "mask_pixels": int(np.count_nonzero(probe)),
            "anchor_bbox": [ax1, ay1, ax2, ay2],
            "ellipse_center": [round(float(cx), 3), round(float(cy), 3)],
            "ellipse_axes": [round(float(axis_a), 3), round(float(axis_b), 3)],
            "ellipse_angle": round(float(angle), 3),
            "panel_luma": round(panel_luma, 3),
        }
        if best is None or score > best[0]:
            best = (score, bbox, ellipse, metrics)
    if best is None:
        ref_w = max(1, rx2 - rx1)
        ref_h = max(1, ry2 - ry1)
        inferred_bbox = [
            max(0, int(round(rx1 - ref_w * 0.75))),
            max(0, int(round(ry1 - ref_h * 0.65))),
            min(width, int(round(rx2 + ref_w * 0.35))),
            min(height, int(round(ry2 + ref_h * 0.22))),
        ]
        ix1, iy1, ix2, iy2 = inferred_bbox
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        inferred_probe = np.zeros((height, width), dtype=np.uint8)
        cx = (ix1 + ix2) / 2.0
        cy = (iy1 + iy2) / 2.0
        axis_a = float(ix2 - ix1)
        axis_b = float(iy2 - iy1)
        angle = 0.0
        cv2.ellipse(
            inferred_probe,
            (int(round(cx)), int(round(cy))),
            (max(1, int(round(axis_a / 2.0))), max(1, int(round(axis_b / 2.0)))),
            0.0,
            0,
            360,
            255,
            -1,
        )
        anchor_coverage = int(np.count_nonzero(inferred_probe[ay1:ay2, ax1:ax2] > 0)) / float(max(1, _bbox_area(anchor)))
        inferred_luma = float(np.median(gray[inferred_probe > 0])) if np.any(inferred_probe) else 255.0
        warm_edge_pixels = int(np.count_nonzero((candidate_edges > 0) & (inferred_probe > 0)))
        if anchor_coverage < 0.68 or inferred_luma > 80.0 or warm_edge_pixels < max(180, int(_bbox_area(anchor) * 0.025)):
            return None
        bbox = inferred_bbox
        ellipse = ((cx, cy), (axis_a, axis_b), angle)
        metrics = {
            "source": "image_dark_bubble_mask",
            "shape_kind": "ellipse",
            "detection_space": "negative_geometry",
            "mask_bbox": bbox,
            "mask_pixels": int(np.count_nonzero(inferred_probe)),
            "anchor_bbox": [ax1, ay1, ax2, ay2],
            "ellipse_center": [round(float(cx), 3), round(float(cy), 3)],
            "ellipse_axes": [round(float(axis_a), 3), round(float(axis_b), 3)],
            "ellipse_angle": 0.0,
            "panel_luma": round(inferred_luma, 3),
            "fallback": "expanded_balloon_bbox_dark_ellipse",
            "warm_edge_pixels": warm_edge_pixels,
        }
    else:
        _score, bbox, ellipse, metrics = best
    (cx, cy), (axis_a, axis_b), angle = ellipse
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(axis_a / 2.0))), max(1, int(round(axis_b / 2.0)))),
        float(angle),
        0,
        360,
        255,
        -1,
    )
    metrics.update(_sample_dark_panel_effect_colors(rgb, bbox, support_mask))
    mask_bbox = _bbox_from_mask(mask) or bbox
    metrics["mask_bbox"] = mask_bbox
    metrics["mask_pixels"] = int(np.count_nonzero(mask))
    qa_metrics = block.setdefault("qa_metrics", {})
    if isinstance(qa_metrics, dict):
        qa_metrics["image_dark_bubble_mask"] = metrics
    block["bubble_mask_source"] = "image_dark_bubble_mask"
    block["bubble_mask_bbox"] = mask_bbox
    block["bubble_mask_shape"] = "ellipse"
    block["bubble_mask_ellipse"] = {
        "center": [round(float(cx), 3), round(float(cy), 3)],
        "axes": [round(float(axis_a), 3), round(float(axis_b), 3)],
        "angle": round(float(angle), 3),
    }
    block["dark_panel_effect_colors"] = {
        key: value
        for key, value in metrics.items()
        if key
        in {
            "color_sample_space",
            "panel_fill_rgb",
            "border_rgb",
            "panel_glow_rgb",
            "text_fill_rgb",
            "text_glow_rgb",
            "bad_negative_text_glow_rgb",
        }
    }
    _remove_qa_flags(
        block,
        {
            "bbox_fallback_bubble_mask",
            "debug_derived_bubble_mask_rejected",
            "derived_bubble_mask_rejected",
            "missing_real_bubble_mask",
            "rejected_derived_bubble_mask",
        },
    )
    block.pop("bubble_mask_error", None)
    block.pop("bubbleMaskError", None)
    return mask


def _derive_dark_ellipse_mask_from_balloon_bbox(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None,
    support_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    height, width = _image_hw(image_shape)
    bbox = _normalize_bbox(block.get("balloon_bbox") or block.get("bubble_mask_bbox"), width, height)
    anchor = _text_geometry_bbox(block, image_shape) or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if bbox is None or anchor is None:
        return None
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(bh)
    if aspect < 1.05 or aspect > 4.2:
        return None
    if _bbox_area(bbox) < int(max(1, _bbox_area(anchor)) * 1.18):
        return None
    if _bbox_min_overlap_ratio(bbox, anchor) < 0.55:
        return None
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    if source not in {"", "bbox_fallback", "derived_white_crop_rejected", "rejected_derived_bubble_mask", "image_dark_bubble_mask"} and not _card_like_dark_context(block):
        return None
    roi_median = None
    if isinstance(image_rgb, np.ndarray) and image_rgb.ndim == 3 and image_rgb.shape[:2] == (height, width):
        gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        roi = gray[y1:y2, x1:x2]
        if roi.size:
            roi_median = float(np.median(roi))
        if roi_median is not None and roi_median > 120.0:
            return None
    elif not _card_like_dark_context(block):
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    axis_a = float(bw)
    axis_b = float(bh)
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(axis_a / 2.0))), max(1, int(round(axis_b / 2.0)))),
        0.0,
        0,
        360,
        255,
        -1,
    )
    metrics = {
        "source": "image_dark_bubble_mask",
        "shape_kind": "ellipse",
        "detection_space": "balloon_bbox_dark_context",
        "mask_bbox": bbox,
        "mask_pixels": int(np.count_nonzero(mask)),
        "anchor_bbox": list(anchor),
        "ellipse_center": [round(float(cx), 3), round(float(cy), 3)],
        "ellipse_axes": [round(axis_a, 3), round(axis_b, 3)],
        "ellipse_angle": 0.0,
    }
    if isinstance(image_rgb, np.ndarray) and image_rgb.ndim == 3 and image_rgb.shape[:2] == (height, width):
        metrics.update(_sample_dark_panel_effect_colors(image_rgb.astype(np.uint8), bbox, support_mask))
    qa_metrics = block.setdefault("qa_metrics", {})
    if isinstance(qa_metrics, dict):
        qa_metrics["image_dark_bubble_mask"] = metrics
    block["bubble_mask_source"] = "image_dark_bubble_mask"
    block["bubble_mask_bbox"] = bbox
    block["bubble_mask_shape"] = "ellipse"
    block["bubble_mask_ellipse"] = {
        "center": [round(float(cx), 3), round(float(cy), 3)],
        "axes": [round(axis_a, 3), round(axis_b, 3)],
        "angle": 0.0,
    }
    block["dark_panel_effect_colors"] = {
        key: value
        for key, value in metrics.items()
        if key
        in {
            "color_sample_space",
            "panel_fill_rgb",
            "border_rgb",
            "panel_glow_rgb",
            "text_fill_rgb",
            "text_glow_rgb",
            "bad_negative_text_glow_rgb",
        }
    }
    _remove_qa_flags(
        block,
        {
            "bbox_fallback_bubble_mask",
            "debug_derived_bubble_mask_rejected",
            "derived_bubble_mask_rejected",
            "missing_real_bubble_mask",
            "rejected_derived_bubble_mask",
        },
    )
    _append_qa_flag(block, "dark_bubble_ellipse_bbox_mask")
    block.pop("bubble_mask_error", None)
    block.pop("bubbleMaskError", None)
    return mask


def _explicit_bubble_mask_should_yield_to_card_panel(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None,
    bubble_mask: np.ndarray | None,
    raw_floor_mask: np.ndarray | None,
    geometry_mask: np.ndarray | None,
) -> bool:
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    if source not in {"image_white_bubble_mask", "image_rect_bubble_mask", "image_contour_bubble_mask"}:
        return False
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return False
    height, width = _image_hw(image_shape)
    if image_rgb.shape[:2] != (height, width):
        return False
    if not isinstance(bubble_mask, np.ndarray) or bubble_mask.shape[:2] != (height, width) or not np.any(bubble_mask):
        return False
    support = None
    if isinstance(raw_floor_mask, np.ndarray) and raw_floor_mask.shape[:2] == (height, width) and np.any(raw_floor_mask):
        support = np.where(raw_floor_mask > 0, 255, 0).astype(np.uint8)
    elif isinstance(geometry_mask, np.ndarray) and geometry_mask.shape[:2] == (height, width) and np.any(geometry_mask):
        support = np.where(geometry_mask > 0, 255, 0).astype(np.uint8)
    if support is None:
        return False
    support_pixels = int(np.count_nonzero(support))
    if support_pixels < 64:
        return False
    gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    inferred_card = _infer_dark_or_colored_card_from_mask(gray, image_rgb.astype(np.uint8), support)
    if not _card_like_dark_context(block) and not inferred_card:
        return False
    bubble = np.where(bubble_mask > 0, 255, 0).astype(np.uint8)
    overlap = int(np.count_nonzero((bubble > 0) & (support > 0)))
    bubble_pixels = int(np.count_nonzero(bubble))
    support_coverage = overlap / float(max(1, support_pixels))
    bubble_vs_support = bubble_pixels / float(max(1, support_pixels))
    if support_coverage >= 0.82 and bubble_vs_support >= 0.90:
        return False
    panel_bbox = _normalize_bbox(block.get("balloon_bbox"), width, height) or _normalize_bbox(
        block.get("target_bbox"),
        width,
        height,
    )
    support_bbox = _bbox_from_mask(support)
    if not panel_bbox or not support_bbox:
        return False
    if _bbox_area(panel_bbox) < int(_bbox_area(support_bbox) * 1.35):
        return False
    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["explicit_bubble_mask_rejected_for_card_panel"] = {
            "source": source,
            "bubble_pixels": bubble_pixels,
            "support_pixels": support_pixels,
            "support_coverage": round(float(support_coverage), 6),
            "bubble_vs_support": round(float(bubble_vs_support), 6),
            "inferred_card_context": bool(inferred_card),
        }
    return True


def _derive_candidate_white_mask(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray,
    bbox: list[int],
    support_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str | None, str | None, float]:
    height, width = _image_hw(image_shape)
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, "empty_crop", 0.0
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.astype(np.uint8)
    if crop.ndim == 3:
        hsv = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        white = ((gray >= 218) & (val >= 218) & (sat <= 82)).astype(np.uint8) * 255
    else:
        white = (gray >= 218).astype(np.uint8) * 255
    if int(np.count_nonzero(white)) < max(64, int(white.size * 0.10)):
        return None, None, "insufficient_white_pixels", 0.0
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )
    white = _fill_binary_holes(white)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats((white > 0).astype(np.uint8), 8)
    if labels_count <= 1:
        return None, None, "no_white_components", 0.0

    local_support = None
    if isinstance(support_mask, np.ndarray) and support_mask.shape[:2] == (height, width) and np.any(support_mask):
        local_support = support_mask[y1:y2, x1:x2] > 0
    if local_support is None or not np.any(local_support):
        geometry_bbox = _text_geometry_bbox(block, image_shape) or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
        if geometry_bbox:
            local_support = np.zeros_like(white, dtype=bool)
            sx1 = max(0, geometry_bbox[0] - x1)
            sy1 = max(0, geometry_bbox[1] - y1)
            sx2 = min(white.shape[1], geometry_bbox[2] - x1)
            sy2 = min(white.shape[0], geometry_bbox[3] - y1)
            if sx2 > sx1 and sy2 > sy1:
                local_support[sy1:sy2, sx1:sx2] = True

    best_label = 0
    best_score = -1
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(64, int(white.size * 0.08)):
            continue
        component = labels == label
        support_overlap = int(np.count_nonzero(component & local_support)) if local_support is not None else 0
        if local_support is not None and np.any(local_support) and support_overlap <= 0:
            continue
        score = support_overlap * 1000 + area
        if score > best_score:
            best_score = score
            best_label = label
    if best_label <= 0:
        return None, None, "no_supported_component", 0.0

    local_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    local_mask = cv2.morphologyEx(
        local_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    local_mask = _fill_binary_holes(local_mask)
    shape_refine = refine_bubble_shape_mask(local_mask)
    local_mask = shape_refine.mask
    local_bbox = _bbox_from_mask(local_mask)
    geometry_bbox = _text_geometry_bbox(block, image_shape) or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if local_bbox and geometry_bbox:
        absolute_local_bbox = [local_bbox[0] + x1, local_bbox[1] + y1, local_bbox[2] + x1, local_bbox[3] + y1]
        anchored = choose_balloon_component(
            text_bbox=geometry_bbox,
            candidates=[absolute_local_bbox],
            page_size=(width, height),
            min_margin_px=3,
        )
        if anchored is None:
            return None, None, "unanchored_component", 0.0
    block["_derived_shape_refiner"] = {
        "shape_kind": shape_refine.shape_kind,
        "removed_pixels": int(shape_refine.removed_pixels),
        "added_pixels": int(shape_refine.added_pixels),
        "accepted": bool(shape_refine.accepted),
    }
    source, error = _classify_derived_bubble_mask(crop, local_mask)
    if source is None or (
        source == "image_white_region"
        and _edge_contact_ratio(local_mask) >= 0.80
        and _component_fill_ratio(local_mask) >= 0.82
    ):
        outline_mask = _derive_outline_contour_bubble_mask(crop, local_support)
        if outline_mask is not None:
            block["_derived_shape_refiner"] = {
                "shape_kind": "outline_seeded_contour",
                "removed_pixels": int(np.count_nonzero((local_mask > 0) & (outline_mask == 0))),
                "added_pixels": int(np.count_nonzero((outline_mask > 0) & (local_mask == 0))),
                "accepted": True,
            }
            return outline_mask.astype(np.uint8), "outline_seeded_contour", None, _component_fill_ratio(outline_mask)
    if _should_model_connected_white_crop_as_ellipse(
        block,
        crop,
        local_mask,
        local_support,
        shape_refine.shape_kind,
        source,
    ):
        ellipse_mask = _ellipse_model_for_crop(local_mask.shape, local_support)
        if ellipse_mask is not None:
            block["_derived_shape_refiner"] = {
                "shape_kind": "ellipse_from_connected_white_crop",
                "removed_pixels": int(np.count_nonzero((local_mask > 0) & (ellipse_mask == 0))),
                "added_pixels": int(np.count_nonzero((ellipse_mask > 0) & (local_mask == 0))),
                "accepted": True,
            }
            return ellipse_mask.astype(np.uint8), "outline_seeded_contour", None, _component_fill_ratio(ellipse_mask)
    if source is None:
        return None, None, error or "rejected_derived_bubble_mask", 0.0
    return local_mask.astype(np.uint8), source, None, _component_fill_ratio(local_mask)


def _derive_white_bubble_mask_from_image(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None,
    support_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None
    height, width = _image_hw(image_shape)
    if image_rgb.shape[0] < height or image_rgb.shape[1] < width:
        return None
    bbox = _normalize_bbox(block.get("balloon_bbox"), width, height)
    if not bbox:
        return None
    background = _rgb_luma_chroma(block.get("background_rgb"))
    profile = {
        str(block.get("balloon_type") or "").strip().lower(),
        str(block.get("layout_profile") or "").strip().lower(),
        str(block.get("block_profile") or "").strip().lower(),
    }
    if background is not None:
        luma, chroma = background
        if luma < 215.0 or chroma > 42:
            return None
    if _card_like_dark_context(block):
        return None
    elif profile and not (profile & {"white", "white_balloon", "dialogue_balloon"}):
        # Without a white hint, only derive from image if the crop itself is strongly white.
        pass

    x1, y1, x2, y2 = bbox
    local_mask, source, error, initial_density = _derive_candidate_white_mask(
        block,
        image_shape,
        image_rgb,
        [x1, y1, x2, y2],
        support_mask=support_mask,
    )
    should_retry_expanded = bool(
        error == "rejected_rectangular_crop"
        or error == "unanchored_component"
        or (
            local_mask is not None
            and source == "derived_rectangular_balloon"
            and initial_density >= 0.92
            and _edge_contact_ratio(local_mask) >= 0.80
        )
    )
    if should_retry_expanded:
        original_mask = local_mask
        original_source = source
        for scale in (1.0, 2.0, 4.0, 6.0):
            expanded = _expanded_bbox([x1, y1, x2, y2], width, height, scale=scale)
            if expanded == [x1, y1, x2, y2]:
                continue
            expanded_mask, expanded_source, expanded_error, expanded_density = _derive_candidate_white_mask(
                block,
                image_shape,
                image_rgb,
                expanded,
                support_mask=support_mask,
            )
            if (
                expanded_source is not None
                and expanded_source == "image_white_region"
                and expanded_density < 0.92
            ):
                local_mask = expanded_mask
                source = expanded_source
                x1, y1, x2, y2 = expanded
                bbox = [x1, y1, x2, y2]
                break
            if error is None:
                error = expanded_error
        else:
            local_mask = original_mask
            source = original_source
    if local_mask is None or source is None:
        metrics = block.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["derived_white_bubble_mask"] = {
                "source": "image_white_region",
                "balloon_bbox": list(bbox),
                "mask_bbox": None,
                "mask_pixels": 0,
                "reason": error or "rejected_derived_bubble_mask",
            }
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = local_mask
    if not np.any(mask):
        return None

    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        mask_bbox = _bbox_from_mask(mask)
        metrics["derived_white_bubble_mask"] = {
            "source": source,
            "balloon_bbox": list(bbox),
            "mask_bbox": mask_bbox,
            "mask_pixels": int(np.count_nonzero(mask)),
        }
        shape_refiner = block.pop("_derived_shape_refiner", None)
        if isinstance(shape_refiner, dict):
            metrics["derived_white_bubble_mask"]["shape_refiner"] = shape_refiner
    return mask


def balloon_mask_from_block(block: dict, image_shape: tuple[int, ...]) -> np.ndarray | None:
    height, width = _image_hw(image_shape)
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    dark_bubble_rect_panel = _dark_bubble_source_should_remain_rect_panel(block)
    if dark_bubble_rect_panel:
        _promote_rect_dark_panel_contract(block, width=width, height=height)
        source = "image_dark_panel_mask"
    if source in {"text_rect_fallback", "image_dark_panel_mask", "image_dark_bubble_mask", "derived_card_panel_mask"}:
        ellipse = block.get("bubble_mask_ellipse") or block.get("bubbleMaskEllipse")
        if source == "image_dark_bubble_mask" and not dark_bubble_rect_panel and isinstance(ellipse, dict):
            try:
                center = ellipse.get("center") or []
                axes = ellipse.get("axes") or []
                angle = float(ellipse.get("angle") or 0.0)
                cx, cy = float(center[0]), float(center[1])
                axis_a, axis_b = float(axes[0]), float(axes[1])
            except Exception:
                cx = cy = axis_a = axis_b = 0.0
            if axis_a > 0 and axis_b > 0:
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.ellipse(
                    mask,
                    (int(round(cx)), int(round(cy))),
                    (max(1, int(round(axis_a / 2.0))), max(1, int(round(axis_b / 2.0)))),
                    angle,
                    0,
                    360,
                    255,
                    -1,
                )
                return mask if np.any(mask) else None
        bbox = _normalize_bbox(block.get("bubble_mask_bbox") or block.get("balloon_bbox"), width, height)
        if bbox:
            mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            if source == "image_dark_bubble_mask" and not dark_bubble_rect_panel:
                cx = int(round((x1 + x2) / 2.0))
                cy = int(round((y1 + y2) / 2.0))
                axis_x = max(1, int(round((x2 - x1) / 2.0)))
                axis_y = max(1, int(round((y2 - y1) / 2.0)))
                cv2.ellipse(mask, (cx, cy), (axis_x, axis_y), 0.0, 0, 360, 255, -1)
            else:
                mask[y1:y2, x1:x2] = 255
            return mask if np.any(mask) else None
        explicit_mask, _reason = _page_space_bubble_mask(block, image_shape)
        if explicit_mask is not None and np.any(explicit_mask):
            return np.where(explicit_mask > 0, 255, 0).astype(np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = _normalize_polygons(block.get("balloon_polygon"), width, height)
    polygons.extend(_normalize_polygons(block.get("connected_lobe_polygons"), width, height))
    if not polygons:
        for key in ("balloon_subregions", "connected_lobe_bboxes"):
            for bbox_value in block.get(key) or []:
                bbox = _normalize_bbox(bbox_value, width, height)
                if bbox:
                    cv2.fillPoly(mask, [np.asarray(_bbox_to_polygon(bbox), dtype=np.int32)], 255)
    else:
        for polygon in polygons:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return mask if np.any(mask) else None


def _polygon_bbox(polygon: list[list[int]]) -> list[int] | None:
    if not polygon:
        return None
    xs = [int(point[0]) for point in polygon]
    ys = [int(point[1]) for point in polygon]
    return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


def _drop_isolated_side_note_line_polygons(block: dict) -> dict:
    if not isinstance(block, dict):
        return block
    polygons = _normalize_polygons(
        block.get("line_polygons"),
        1_000_000,
        1_000_000,
    )
    if len(polygons) < 3:
        return block
    entries: list[tuple[int, list[int], float, int]] = []
    for index, polygon in enumerate(polygons):
        bbox = _polygon_bbox(polygon)
        if bbox is None:
            continue
        x1, _y1, x2, _y2 = bbox
        entries.append((index, bbox, (x1 + x2) / 2.0, max(1, x2 - x1)))
    if len(entries) < 3:
        return block
    centers = np.asarray([entry[2] for entry in entries], dtype=np.float32)
    widths = np.asarray([entry[3] for entry in entries], dtype=np.float32)
    median_center = float(np.median(centers))
    main_width = float(np.median(widths))
    kept_indices: set[int] = set()
    removed: list[dict] = []
    for index, bbox, center, poly_width in entries:
        center_gap = abs(center - median_center)
        short_line = poly_width <= max(92.0, main_width * 0.62)
        far_side = center_gap >= max(120.0, main_width * 0.82)
        if short_line and far_side:
            removed.append({"index": index, "bbox": bbox, "center_gap": round(center_gap, 3)})
            continue
        kept_indices.add(index)
    if not removed or len(kept_indices) < 2:
        return block
    cleaned = dict(block)
    cleaned["line_polygons"] = [polygon for index, polygon in enumerate(polygons) if index in kept_indices]
    metrics = cleaned.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["isolated_side_note_line_polygons_removed"] = removed
    flags = cleaned.setdefault("qa_flags", [])
    if isinstance(flags, list) and "isolated_side_note_line_polygons_removed" not in flags:
        flags.append("isolated_side_note_line_polygons_removed")
    return cleaned


def _bridge_status_line_fragments(
    block: dict,
    mask: np.ndarray,
    polygons: list[list[list[int]]],
    image_shape: tuple[int, ...],
) -> np.ndarray:
    if not polygons or not _non_white_textured_geometry_recovery_candidate(block, image_shape):
        return mask
    height, width = _image_hw(image_shape)
    bboxes = [bbox for bbox in (_polygon_bbox(polygon) for polygon in polygons) if bbox]
    if len(bboxes) < 2:
        return mask

    def vertical_overlap_ratio(a: list[int], b: list[int]) -> float:
        overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return overlap / float(max(1, min(a[3] - a[1], b[3] - b[1])))

    groups: list[list[list[int]]] = []
    for bbox in sorted(bboxes, key=lambda item: ((item[1] + item[3]) / 2.0, item[0])):
        placed = False
        cy = (bbox[1] + bbox[3]) / 2.0
        h = max(1, bbox[3] - bbox[1])
        for group in groups:
            gy1 = min(item[1] for item in group)
            gy2 = max(item[3] for item in group)
            group_h = max(1, gy2 - gy1)
            gcy = (gy1 + gy2) / 2.0
            if abs(cy - gcy) <= max(8.0, min(h, group_h) * 0.6) or any(
                vertical_overlap_ratio(bbox, item) >= 0.42 for item in group
            ):
                group.append(bbox)
                placed = True
                break
        if not placed:
            groups.append([bbox])

    bridged = mask.astype(np.uint8).copy()
    bridges: list[dict] = []
    for group in groups:
        if len(group) < 2:
            continue
        x1 = max(0, min(item[0] for item in group))
        y1 = max(0, min(item[1] for item in group) - 1)
        x2 = min(width, max(item[2] for item in group))
        y2 = min(height, max(item[3] for item in group) + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        span_area = (x2 - x1) * (y2 - y1)
        original_area = sum(max(0, item[2] - item[0]) * max(0, item[3] - item[1]) for item in group)
        if span_area > max(original_area * 2.8, original_area + 24_000):
            continue
        bridged[y1:y2, x1:x2] = 255
        bridges.append({"bbox": [x1, y1, x2, y2], "fragments": len(group)})

    if bridges:
        qa_metrics = block.setdefault("qa_metrics", {})
        if isinstance(qa_metrics, dict):
            qa_metrics["status_line_fragment_bridge"] = bridges
    return bridged


def mask_from_text_geometry(block: dict, image_shape: tuple[int, ...]) -> np.ndarray | None:
    height, width = _image_hw(image_shape)
    mask = np.zeros((height, width), dtype=np.uint8)
    pad = glyph_padding(_font_size_from_block(block))
    polygons = _text_geometry_polygons(block, width, height)
    if polygons:
        for polygon in polygons:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        mask = _bridge_status_line_fragments(block, mask, polygons, image_shape)
        if np.any(mask):
            kernel_size = max(3, pad * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            return cv2.dilate(mask, kernel, iterations=1)

    bbox = (
        _normalize_bbox(block.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(block.get("bbox"), width, height)
    )
    if not bbox:
        return None
    mask = bbox_to_octagon_mask(width, height, bbox, padding=pad)
    return mask if np.any(mask) else None


def _mask_is_overbroad_against_geometry(text_mask: np.ndarray, geometry_mask: np.ndarray) -> bool:
    text_area = int(np.count_nonzero(text_mask))
    geometry_area = int(np.count_nonzero(geometry_mask))
    if text_area <= 0 or geometry_area <= 0:
        return False
    extra_area = int(np.count_nonzero((text_mask > 0) & (geometry_mask == 0)))
    extra_ratio = extra_area / float(max(1, text_area))
    area_limit = max(int(geometry_area * 2.8), geometry_area + 256)
    return text_area >= area_limit and extra_ratio >= 0.45


def _merge_missing_geometry_components(
    text_mask: np.ndarray,
    geometry_mask: np.ndarray,
    *,
    min_coverage: float = 0.24,
) -> tuple[np.ndarray, list[dict]]:
    if not isinstance(text_mask, np.ndarray) or not isinstance(geometry_mask, np.ndarray):
        return text_mask, []
    if text_mask.shape != geometry_mask.shape or not np.any(text_mask) or not np.any(geometry_mask):
        return text_mask, []

    geometry_bin = (geometry_mask > 0).astype(np.uint8)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(geometry_bin, connectivity=8)
    if labels_count <= 1:
        return text_mask, []

    merged = text_mask.astype(np.uint8).copy()
    text_bin = merged > 0
    geometry_area = int(np.count_nonzero(geometry_bin))
    min_component_area = max(18, int(geometry_area * 0.015))
    added: list[dict] = []

    for label in range(1, labels_count):
        x, y, comp_w, comp_h, area = [int(v) for v in stats[label]]
        if area < min_component_area or comp_w <= 0 or comp_h <= 0:
            continue
        component = labels == label
        covered = int(np.count_nonzero(text_bin & component))
        coverage = covered / float(max(1, area))
        if coverage >= min_coverage:
            continue
        merged[component] = 255
        added.append(
            {
                "bbox": [x, y, x + comp_w, y + comp_h],
                "area": area,
                "coverage": round(float(coverage), 6),
            }
        )

    return merged, added


def _non_white_textured_geometry_recovery_candidate(block: dict, image_shape: tuple[int, ...]) -> bool:
    height, width = _image_hw(image_shape)
    if not _text_geometry_polygons(block, width, height):
        return False
    background = _rgb_luma_chroma(block.get("background_rgb"))
    if background is not None:
        luma, chroma = background
        if chroma < 18 and luma > 215.0:
            return False
    terms = _text_words_upper(block)
    status_like = bool(terms & TEXTURED_STATUS_PANEL_TERMS)
    if not status_like and background is not None:
        luma, chroma = background
        status_like = 45.0 <= luma <= 215.0 and chroma >= 18
    if not status_like:
        return False
    source_bbox = (
        _normalize_bbox(block.get("balloon_bbox"), width, height)
        or _normalize_bbox(block.get("bbox"), width, height)
    )
    geometry_bbox = _text_geometry_bbox(block, image_shape)
    if not source_bbox or not geometry_bbox:
        return False
    source_area = max(1, _bbox_area(source_bbox))
    geometry_area = max(1, _bbox_area(geometry_bbox))
    if source_area / float(geometry_area) > 1.85:
        return False
    return _bbox_min_overlap_ratio(source_bbox, geometry_bbox) >= 0.72


def _fragmented_mask_geometry_union(
    block: dict,
    text_mask: np.ndarray,
    geometry_mask: np.ndarray,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict | None]:
    if not _non_white_textured_geometry_recovery_candidate(block, image_shape):
        return text_mask, None
    if not isinstance(text_mask, np.ndarray) or not isinstance(geometry_mask, np.ndarray):
        return text_mask, None
    if text_mask.shape != geometry_mask.shape or not np.any(text_mask) or not np.any(geometry_mask):
        return text_mask, None
    raw_is_overbroad = _mask_is_overbroad_against_geometry(text_mask, geometry_mask)
    if raw_is_overbroad:
        return text_mask, None

    text_bin = text_mask > 0
    geometry_bin = geometry_mask > 0
    mask_area = int(np.count_nonzero(text_bin))
    geometry_area = int(np.count_nonzero(geometry_bin))
    if mask_area <= 0 or geometry_area <= 0:
        return text_mask, None
    covered = int(np.count_nonzero(text_bin & geometry_bin))
    geometry_coverage = covered / float(max(1, geometry_area))
    mask_geometry_ratio = mask_area / float(max(1, geometry_area))

    labels_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        text_bin.astype(np.uint8),
        connectivity=8,
    )
    component_count = max(0, int(labels_count) - 1)
    largest_component = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if component_count else 0
    largest_component_ratio = largest_component / float(max(1, mask_area))
    fragment_candidate = (
        component_count >= 3
        and largest_component_ratio <= 0.82
        and mask_geometry_ratio <= 0.62
        and geometry_coverage <= 0.58
    )
    if not fragment_candidate:
        return text_mask, None

    union_mask = np.maximum(text_mask.astype(np.uint8), geometry_mask.astype(np.uint8))
    metrics = {
        "component_count": component_count,
        "coverage": round(float(geometry_coverage), 6),
        "largest_component_ratio": round(float(largest_component_ratio), 6),
        "mask_area": mask_area,
        "geometry_area": geometry_area,
        "mask_geometry_ratio": round(float(mask_geometry_ratio), 6),
    }
    return union_mask, metrics


def _white_balloon_geometry_floor(
    block: dict,
    text_mask: np.ndarray,
    geometry_mask: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict | None]:
    if not isinstance(text_mask, np.ndarray) or not isinstance(geometry_mask, np.ndarray):
        return text_mask, None
    if text_mask.shape != geometry_mask.shape or not np.any(text_mask) or not np.any(geometry_mask):
        return text_mask, None
    profile = str(block.get("layout_profile") or block.get("block_profile") or "").strip().lower()
    balloon_type = str(block.get("balloon_type") or "").strip().lower()
    bubble_source = str(block.get("bubble_mask_source") or "").strip().lower()
    derived_metrics = block.get("qa_metrics", {}).get("derived_white_bubble_mask") if isinstance(block.get("qa_metrics"), dict) else None
    derived_source = str((derived_metrics or {}).get("source") or "").strip().lower() if isinstance(derived_metrics, dict) else ""
    white_source = bubble_source in {
        "image_white_bubble_mask",
        "derived_white_bubble_mask",
        "derived_white_crop",
    } or derived_source in {"image_white_region", "derived_rectangular_balloon", "outline_seeded_contour"}
    if balloon_type != "white" and profile != "white_balloon" and not white_source:
        return text_mask, None
    text_pixels = int(np.count_nonzero(text_mask))
    geometry_pixels = int(np.count_nonzero(geometry_mask))
    if geometry_pixels <= 0:
        return text_mask, None
    coverage = text_pixels / float(max(1, geometry_pixels))
    if coverage >= 0.68:
        return text_mask, None
    image_h, image_w = _image_hw(image_shape)
    if geometry_pixels > int(image_h * image_w * 0.18):
        return text_mask, None
    if _raw_floor_support_mask_for_block(block, text_mask.shape[:2]) is None:
        return text_mask, None
    merged, added = _merge_missing_geometry_components(text_mask, geometry_mask, min_coverage=0.18)
    if not added or not np.any(merged):
        return text_mask, None
    metrics = {
        "raw_pixels": text_pixels,
        "geometry_pixels": geometry_pixels,
        "coverage": round(float(coverage), 6),
        "added_components": added,
    }
    return merged.astype(np.uint8), metrics


def _preserve_colored_card_geometry_when_bubble_clip_is_invalid(
    block: dict,
    text_mask: np.ndarray,
    geometry_mask: np.ndarray | None,
    clip_mask: np.ndarray | None,
    *,
    source: str,
    image_rgb: np.ndarray | None = None,
) -> np.ndarray:
    profile = str(block.get("layout_profile") or block.get("block_profile") or "").strip().lower()
    balloon_type = str(block.get("balloon_type") or "").strip().lower()
    bubble_source = str(block.get("bubble_mask_source") or "").strip().lower()
    explicit_white_balloon = bool(
        profile == "white_balloon"
        or balloon_type == "white"
        or bubble_source in {"image_white_bubble_mask", "image_rect_bubble_mask", "image_contour_bubble_mask"}
    )
    if explicit_white_balloon and not _dark_or_colored_text_card_context(block):
        return text_mask
    inferred_card_context = False
    if (
        isinstance(image_rgb, np.ndarray)
        and image_rgb.ndim == 3
        and isinstance(geometry_mask, np.ndarray)
        and image_rgb.shape[:2] == geometry_mask.shape[:2]
        and np.any(geometry_mask)
    ):
        gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        inferred_card_context = _infer_dark_or_colored_card_from_mask(gray, image_rgb.astype(np.uint8), geometry_mask)
    if not _dark_or_colored_text_card_context(block) and not inferred_card_context:
        return text_mask
    if not isinstance(geometry_mask, np.ndarray) or not np.any(geometry_mask):
        return text_mask
    if not isinstance(clip_mask, np.ndarray) or not np.any(clip_mask):
        return text_mask
    geometry_pixels = int(np.count_nonzero(geometry_mask))
    text_pixels = int(np.count_nonzero(text_mask)) if isinstance(text_mask, np.ndarray) else 0
    if geometry_pixels <= 0:
        return text_mask
    clip_overlap = int(np.count_nonzero((clip_mask > 0) & (geometry_mask > 0)))
    clip_area = int(np.count_nonzero(clip_mask))
    clipped_too_much = text_pixels < max(32, int(round(geometry_pixels * 0.35)))
    clip_misses_text = clip_overlap < max(24, int(round(geometry_pixels * 0.20)))
    clip_too_small = clip_area < max(64, int(round(geometry_pixels * 0.45)))
    if not (clipped_too_much and (clip_misses_text or clip_too_small)):
        return text_mask
    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["colored_card_geometry_preserved"] = {
            "source": source,
            "text_pixels_after_clip": text_pixels,
            "geometry_pixels": geometry_pixels,
            "clip_pixels": clip_area,
            "clip_geometry_overlap": clip_overlap,
        }
    return geometry_mask.astype(np.uint8)


def _balloon_bbox_is_tight_text_anchor(block: dict, image_shape: tuple[int, ...]) -> bool:
    height, width = _image_hw(image_shape)
    balloon_bbox = _normalize_bbox(block.get("balloon_bbox"), width, height)
    if not balloon_bbox:
        return False
    anchor_bbox = (
        _text_geometry_bbox(block, image_shape)
        or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(block.get("bbox"), width, height)
    )
    if not anchor_bbox:
        return False

    balloon_area = max(1, _bbox_area(balloon_bbox))
    anchor_area = max(1, _bbox_area(anchor_bbox))
    area_ratio = balloon_area / float(anchor_area)
    overlap_ratio = _bbox_min_overlap_ratio(balloon_bbox, anchor_bbox)
    balloon_cx = (balloon_bbox[0] + balloon_bbox[2]) / 2.0
    balloon_cy = (balloon_bbox[1] + balloon_bbox[3]) / 2.0
    anchor_cx = (anchor_bbox[0] + anchor_bbox[2]) / 2.0
    anchor_cy = (anchor_bbox[1] + anchor_bbox[3]) / 2.0
    anchor_w = max(1, anchor_bbox[2] - anchor_bbox[0])
    anchor_h = max(1, anchor_bbox[3] - anchor_bbox[1])
    center_offset_x = abs(balloon_cx - anchor_cx) / float(anchor_w)
    center_offset_y = abs(balloon_cy - anchor_cy) / float(anchor_h)
    return 0.72 <= area_ratio <= 1.55 and overlap_ratio >= 0.82 and center_offset_x <= 0.16 and center_offset_y <= 0.16


def _balloon_bbox_clips_line_geometry_tail(block: dict, image_shape: tuple[int, ...]) -> bool:
    height, width = _image_hw(image_shape)
    if not _has_line_polygon_geometry(block, width, height):
        return False
    balloon_bbox = _normalize_bbox(block.get("balloon_bbox"), width, height)
    line_bbox = _text_geometry_bbox(block, image_shape)
    if not balloon_bbox or not line_bbox:
        return False
    line_w = max(1, line_bbox[2] - line_bbox[0])
    line_h = max(1, line_bbox[3] - line_bbox[1])
    balloon_h = max(1, balloon_bbox[3] - balloon_bbox[1])
    if line_w < 180 or line_h > max(56, int(balloon_h * 0.9)):
        return False
    vertical_overlap = max(0, min(balloon_bbox[3], line_bbox[3]) - max(balloon_bbox[1], line_bbox[1]))
    if vertical_overlap / float(line_h) < 0.72:
        return False
    right_tail = max(0, line_bbox[2] - balloon_bbox[2])
    left_tail = max(0, balloon_bbox[0] - line_bbox[0])
    return max(right_tail, left_tail) >= max(32, int(line_w * 0.20))


def _synthetic_tight_bubble_reference(block: dict, image_shape: tuple[int, ...] | None) -> bool:
    if image_shape is None:
        return False
    height, width = _image_hw(image_shape)
    bubble_bbox = _normalize_bbox(block.get("bubble_mask_bbox"), width, height)
    if not bubble_bbox:
        return False

    bubble_area = max(1, _bbox_area(bubble_bbox))
    for key in ("bbox", "source_bbox", "text_pixel_bbox", "layout_bbox"):
        candidate = _normalize_bbox(block.get(key), width, height)
        if candidate and _bbox_iou(bubble_bbox, candidate) >= 0.72:
            return True

    geometry_bbox = _text_geometry_bbox(block, image_shape)
    if geometry_bbox:
        geometry_area = max(1, _bbox_area(geometry_bbox))
        overlap = _bbox_intersection_area(bubble_bbox, geometry_bbox)
        if overlap > 0 and bubble_area <= max(512, int(geometry_area * 0.45)):
            return True

    inner_bbox = _normalize_bbox(block.get("bubble_inner_bbox"), width, height)
    if inner_bbox and _bbox_area(inner_bbox) <= max(16, int(bubble_area * 0.12)):
        fallback = _normalize_bbox(block.get("bbox"), width, height)
        if fallback and _bbox_iou(bubble_bbox, fallback) >= 0.55:
            return True
    return False


def _synthetic_tight_card_bubble_reference(block: dict, image_shape: tuple[int, ...] | None) -> bool:
    if not _card_like_dark_context(block) or image_shape is None:
        return False
    if _synthetic_tight_bubble_reference(block, image_shape):
        return True
    height, width = _image_hw(image_shape)
    bubble_bbox = _normalize_bbox(block.get("bubble_mask_bbox"), width, height)
    balloon_bbox = _normalize_bbox(block.get("balloon_bbox"), width, height) or _normalize_bbox(
        block.get("target_bbox"),
        width,
        height,
    )
    if not bubble_bbox or not balloon_bbox:
        return False
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    weak_sources = {
        "derived_white_crop",
        "derived_white_crop_rejected",
        "image_white_bubble_mask",
        "image_white_region",
        "rejected_derived_bubble_mask",
    }
    if source not in weak_sources:
        return False
    bubble_area = max(1, _bbox_area(bubble_bbox))
    balloon_area = max(1, _bbox_area(balloon_bbox))
    if bubble_area / float(balloon_area) > 0.45:
        return False
    if _bbox_intersection_area(bubble_bbox, balloon_bbox) / float(bubble_area) < 0.80:
        return False
    return True


def _has_real_balloon_reference(block: dict, image_shape: tuple[int, ...] | None = None) -> bool:
    if _synthetic_tight_bubble_reference(block, image_shape):
        return False
    for key in ("bubble_mask", "bubbleMask", "balloon_mask", "balloonMask", "segmentation_mask"):
        mask = block.get(key)
        if isinstance(mask, np.ndarray) and mask.size > 0 and np.any(mask):
            return True
    return bool(
        block.get("balloon_polygon")
        or block.get("connected_lobe_polygons")
        or block.get("balloon_subregions")
        or block.get("connected_lobe_bboxes")
    )


def _skip_density_guard_for_block(block: dict) -> bool:
    flags = {str(flag).strip().upper() for flag in block.get("qa_flags") or []}
    if "TEXT_CLIPPED" in flags:
        return True
    return abs(_block_text_angle_degrees(block)) > 5.0


def _density_guard_text_mask(
    block: dict,
    text_mask: np.ndarray,
    raw_text_mask: np.ndarray | None,
    balloon_area: int,
    *,
    flag_reliable_reference: bool,
    threshold: float = MASK_DENSITY_HIGH,
    erode_px: int = 2,
) -> np.ndarray:
    text_pixels = int(np.count_nonzero(text_mask))
    raw_pixels = int(np.count_nonzero(raw_text_mask)) if isinstance(raw_text_mask, np.ndarray) else 0
    density = text_pixels / float(max(1, balloon_area))
    if density < threshold:
        return text_mask

    guarded = text_mask
    source = "expanded_mask"
    height, width = _image_hw(text_mask.shape)
    raw_bbox = _bbox_from_mask(raw_text_mask) if isinstance(raw_text_mask, np.ndarray) else None
    geometry_bbox = _text_geometry_bbox(block, text_mask.shape)
    raw_misses_line_tail = False
    if raw_bbox and geometry_bbox and _has_line_polygon_geometry(block, width, height):
        raw_w = max(1, raw_bbox[2] - raw_bbox[0])
        raw_h = max(1, raw_bbox[3] - raw_bbox[1])
        geometry_w = max(1, geometry_bbox[2] - geometry_bbox[0])
        geometry_h = max(1, geometry_bbox[3] - geometry_bbox[1])
        overlap = _bbox_intersection_area(raw_bbox, geometry_bbox)
        geometry_area = max(1, _bbox_area(geometry_bbox))
        raw_misses_line_tail = bool(
            geometry_w >= 180
            and raw_w < int(geometry_w * 0.60)
            and raw_h <= max(geometry_h * 2, geometry_h + 12)
            and overlap / float(geometry_area) < 0.72
        )
    if raw_misses_line_tail:
        source = "line_polygon_sparse_raw_union"
    elif isinstance(raw_text_mask, np.ndarray) and raw_pixels > 0 and raw_pixels / float(max(1, balloon_area)) <= threshold:
        guarded = raw_text_mask.astype(np.uint8)
        source = "raw_mask"
    elif erode_px > 0:
        kernel_size = int(erode_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(text_mask.astype(np.uint8), kernel, iterations=1)
        if np.any(eroded) and int(np.count_nonzero(eroded)) < text_pixels:
            guarded = eroded.astype(np.uint8)
            source = "eroded_expanded_mask"
    elif isinstance(raw_text_mask, np.ndarray) and raw_pixels > 0:
        guarded = raw_text_mask.astype(np.uint8)
        source = "raw_mask"

    qa_metrics = block.setdefault("qa_metrics", {})
    if isinstance(qa_metrics, dict):
        qa_metrics["mask_density_guard"] = {
            "density": round(float(density), 6),
            "threshold": round(float(threshold), 6),
            "source": source,
            "raw_mask_pixels": int(raw_pixels),
            "expanded_mask_pixels": int(text_pixels),
            "guarded_mask_pixels": int(np.count_nonzero(guarded)),
            "reliable_balloon_reference": bool(flag_reliable_reference),
        }
    return guarded.astype(np.uint8)

def build_glyph_text_mask(block: dict, image_shape: tuple[int, ...]) -> np.ndarray | None:
    height, width = _image_hw(image_shape)
    mask = np.zeros((height, width), dtype=np.uint8)
    used_polygons = False
    for polygon in _text_geometry_polygons(block, width, height):
        points = np.asarray(polygon, dtype=np.int32)
        if points.shape[0] >= 3:
            cv2.fillPoly(mask, [points], 255)
            used_polygons = True

    text_bbox = _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if text_bbox and not used_polygons:
        x1, y1, x2, y2 = text_bbox
        mask[y1:y2, x1:x2] = 255

    cleanup_bbox, cleanup = _clipped_overlap_fragment_cleanup_bbox_for_block(block, image_shape)
    if cleanup_bbox is not None:
        x1, y1, x2, y2 = cleanup_bbox
        mask[y1:y2, x1:x2] = 255
        if isinstance(cleanup, dict):
            cleanup["applied_to_glyph_mask"] = True

    return mask if np.any(mask) else None


def _clipped_overlap_fragment_cleanup_bbox_for_block(
    block: dict,
    image_shape: tuple[int, ...],
) -> tuple[list[int] | None, dict | None]:
    height, width = _image_hw(image_shape)
    metrics = block.get("qa_metrics") if isinstance(block.get("qa_metrics"), dict) else {}
    cleanup = block.get("clipped_overlap_fragment_cleanup_bbox")
    if not isinstance(cleanup, dict):
        cleanup = metrics.get("clipped_overlap_fragment_cleanup_bbox") if isinstance(metrics, dict) else None
    cleanup_bbox = cleanup.get("bbox") if isinstance(cleanup, dict) else None
    cleanup_bbox = _normalize_bbox(cleanup_bbox, width, height)
    if cleanup_bbox is not None:
        return cleanup_bbox, cleanup if isinstance(cleanup, dict) else None

    flags = {str(flag).strip() for flag in block.get("qa_flags") or [] if str(flag).strip()}
    if "false_dark_bubble_trailing_clipped_fragment_removed" not in flags:
        return None, None
    text_bbox = (
        _normalize_bbox(block.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(block.get("bbox"), width, height)
        or _normalize_bbox(block.get("source_bbox"), width, height)
    )
    if text_bbox is None:
        return None, None
    bubble_bbox = (
        _normalize_bbox(block.get("bubble_mask_bbox"), width, height)
        or _normalize_bbox(block.get("balloon_bbox"), width, height)
        or text_bbox
    )
    text_w = max(1, text_bbox[2] - text_bbox[0])
    text_h = max(1, text_bbox[3] - text_bbox[1])
    x1 = max(0, text_bbox[2] - max(96, int(round(text_w * 0.50))))
    x2 = min(width, text_bbox[2] + max(180, text_w))
    y1 = max(text_bbox[3] + max(18, int(round(text_h * 0.75))), bubble_bbox[3] - max(36, int(round(text_h * 0.50))))
    y2 = min(height, max(y1 + max(32, int(round(text_h * 0.70))), bubble_bbox[3] + max(48, int(round(text_h * 0.80)))))
    fallback_bbox = _normalize_bbox([x1, y1, x2, y2], width, height)
    if fallback_bbox is None:
        return None, None
    cleanup_payload = {
        "bbox": fallback_bbox,
        "source": "fallback_from_removed_trailing_fragment_flag",
    }
    block["clipped_overlap_fragment_cleanup_bbox"] = cleanup_payload
    if isinstance(metrics, dict):
        metrics["clipped_overlap_fragment_cleanup_bbox"] = dict(cleanup_payload)
    return fallback_bbox, cleanup_payload


def _apply_clipped_overlap_fragment_cleanup_mask(
    block: dict,
    text_mask: np.ndarray,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    height, width = _image_hw(image_shape)
    cleanup_bbox, cleanup = _clipped_overlap_fragment_cleanup_bbox_for_block(block, image_shape)
    if cleanup_bbox is None:
        return text_mask
    x1, y1, x2, y2 = cleanup_bbox
    cleanup_mask = np.zeros((height, width), dtype=np.uint8)
    cleanup_mask[y1:y2, x1:x2] = 255
    merged = cv2.bitwise_or(text_mask.astype(np.uint8), cleanup_mask)
    if isinstance(cleanup, dict):
        cleanup["applied_to_final_text_mask"] = True
        cleanup["final_text_mask_pixels_before"] = int(np.count_nonzero(text_mask))
        cleanup["final_text_mask_pixels_after"] = int(np.count_nonzero(merged))
    return merged.astype(np.uint8)


def _dark_bubble_recovered_text_bbox_floor(
    block: dict,
    text_mask: np.ndarray,
    image_shape: tuple[int, ...],
    bubble_mask: np.ndarray | None,
) -> np.ndarray:
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    if source != "image_dark_bubble_mask":
        return text_mask
    flags = {str(flag).strip() for flag in block.get("qa_flags") or []}
    if not (
        "partial_dark_bubble_lobe_reocr" in flags
        or "detected_dark_bubble_without_text_reocr" in flags
        or "candidate_crop_direct_paddle_reocr" in flags
    ):
        return text_mask
    if not isinstance(text_mask, np.ndarray) or not np.any(text_mask):
        return text_mask
    height, width = _image_hw(image_shape)
    text_bbox = _normalize_bbox(block.get("text_pixel_bbox"), width, height) or _normalize_bbox(block.get("bbox"), width, height)
    if text_bbox is None:
        return text_mask
    current_bbox = _bbox_from_mask(text_mask)
    if current_bbox is not None:
        text_area = max(1, _bbox_area(text_bbox))
        overlap = _bbox_intersection_area(current_bbox, text_bbox)
        current_w = max(1, current_bbox[2] - current_bbox[0])
        text_w = max(1, text_bbox[2] - text_bbox[0])
        if overlap / float(text_area) >= 0.72 and current_w >= int(text_w * 0.72):
            return text_mask
    floor_mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = text_bbox
    floor_mask[y1:y2, x1:x2] = 255
    if isinstance(bubble_mask, np.ndarray) and bubble_mask.shape[:2] == (height, width) and np.any(bubble_mask):
        floor_mask = np.where((floor_mask > 0) & (bubble_mask > 0), 255, 0).astype(np.uint8)
        if not np.any(floor_mask):
            return text_mask
    merged = cv2.bitwise_or(text_mask.astype(np.uint8), floor_mask.astype(np.uint8))
    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["dark_bubble_recovered_text_bbox_floor"] = {
            "before_pixels": int(np.count_nonzero(text_mask)),
            "floor_pixels": int(np.count_nonzero(floor_mask)),
            "after_pixels": int(np.count_nonzero(merged)),
        }
    _append_qa_flag(block, "dark_bubble_recovered_text_bbox_floor")
    return merged.astype(np.uint8)


def _has_explicit_tight_source_reference(block: dict, image_shape: tuple[int, ...]) -> bool:
    height, width = _image_hw(image_shape)
    explicit_source = _normalize_bbox(block.get("source_bbox"), width, height) or _normalize_bbox(
        block.get("layout_bbox"),
        width,
        height,
    )
    if not explicit_source:
        return False
    return _should_add_tight_white_source_search(block, image_shape)


def _dark_glyph_pixels_in_text_bbox(
    block: dict,
    image_rgb: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None
    height, width = _image_hw(image_shape)
    bbox = _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if _has_explicit_tight_source_reference(block, image_shape):
        source_bbox = _source_reference_bbox(block, width, height)
        if source_bbox:
            bbox = source_bbox
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    if area > max(12000, int(width * height * 0.08)):
        return None
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = crop.astype(np.uint8)
    if float(np.median(gray)) < 150.0:
        return None
    dark = gray <= 96
    if not np.any(dark):
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = np.where(dark, 255, 0).astype(np.uint8)
    return mask


def _colored_card_visual_glyph_mask(
    block: dict,
    image_rgb: np.ndarray | None,
    image_shape: tuple[int, ...],
    support_mask: np.ndarray | None,
) -> np.ndarray | None:
    if not _dark_or_colored_text_card_context(block):
        return None
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0 or image_rgb.ndim != 3:
        return None
    height, width = _image_hw(image_shape)
    if image_rgb.shape[:2] != (height, width):
        return None
    if not isinstance(support_mask, np.ndarray) or support_mask.shape[:2] != (height, width) or not np.any(support_mask):
        return None
    background = block.get("background_rgb")
    if not isinstance(background, (list, tuple)) or len(background) < 3:
        return None
    try:
        bg = np.asarray([int(round(float(v))) for v in background[:3]], dtype=np.int16)
    except Exception:
        return None
    bg_luma_chroma = _rgb_luma_chroma(background)
    if bg_luma_chroma is None:
        return None
    refined_local_background = False
    region = support_mask > 0
    region_pixels = int(np.count_nonzero(region))
    if region_pixels < 24:
        return None
    region_u8 = region.astype(np.uint8) * 255
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    near = cv2.dilate(region_u8, ring_kernel, iterations=1) > 0
    ring = near & ~region
    if int(np.count_nonzero(ring)) >= 48:
        local_bg = np.median(image_rgb[ring].astype(np.float32), axis=0)
        if float(np.mean(np.abs(local_bg - bg.astype(np.float32)))) >= 18.0:
            bg = np.asarray([int(max(0, min(255, round(float(v))))) for v in local_bg[:3]], dtype=np.int16)
            bg_luma_chroma = _rgb_luma_chroma(bg.tolist())
            if bg_luma_chroma is None:
                return None
            refined_local_background = True
    bg_luma, _bg_chroma = bg_luma_chroma
    rgb_i = image_rgb.astype(np.int16)
    rgb_f = image_rgb.astype(np.float32)
    luma = (rgb_f[:, :, 0] * 0.299) + (rgb_f[:, :, 1] * 0.587) + (rgb_f[:, :, 2] * 0.114)
    chroma = np.max(rgb_i, axis=2) - np.min(rgb_i, axis=2)
    delta = np.mean(np.abs(rgb_i - bg[None, None, :]), axis=2)
    candidate = region & (delta >= 24.0) & (
        ((luma >= float(bg_luma) + 18.0) & (luma >= 145.0))
        | ((luma <= float(bg_luma) - 20.0) & (float(bg_luma) >= 88.0))
        | ((np.abs(luma - float(bg_luma)) >= 4.0) & (chroma >= 38.0) & (delta >= 28.0))
    )
    candidate_pixels = int(np.count_nonzero(candidate))
    refined_by_local_glyph = False
    if refined_local_background or candidate_pixels > int(region_pixels * 0.45):
        local_luma = luma[region]
        local_chroma = chroma[region]
        if local_luma.size >= 24:
            median_luma = float(np.percentile(local_luma, 50))
            chroma_cutoff = max(30.0, min(44.0, float(np.percentile(local_chroma, 35))))
            luma_cutoff = max(205.0, min(232.0, median_luma + 18.0 if refined_local_background else median_luma + 4.0))
            local_candidate = region & (luma >= luma_cutoff) & (chroma <= chroma_cutoff)
            local_pixels = int(np.count_nonzero(local_candidate))
            if local_pixels >= max(24, min(160, int(round(region_pixels * 0.006)))) and local_pixels < candidate_pixels:
                candidate = local_candidate
                candidate_pixels = local_pixels
                refined_by_local_glyph = True
    if candidate_pixels < max(24, min(160, int(round(region_pixels * 0.006)))):
        return None
    if candidate_pixels > int(region_pixels * 0.72):
        return None
    kernel_side = 5 if refined_by_local_glyph else 9
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_side, kernel_side))
    visual = cv2.dilate(candidate.astype(np.uint8) * 255, kernel, iterations=1)
    visual_pixels = int(np.count_nonzero(visual))
    if visual_pixels <= 0 or visual_pixels > int(region_pixels * 0.95):
        return None
    if refined_by_local_glyph or refined_local_background:
        metrics = block.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["colored_card_visual_glyph_mask_refined"] = {
                "candidate_pixels": int(candidate_pixels),
                "mask_pixels": int(visual_pixels),
                "region_pixels": int(region_pixels),
                "local_background": bool(refined_local_background),
            }
    return visual.astype(np.uint8)


def _dark_bubble_visual_glyph_mask(
    block: dict,
    image_rgb: np.ndarray | None,
    image_shape: tuple[int, ...],
    bubble_mask: np.ndarray | None,
    support_mask: np.ndarray | None,
) -> np.ndarray | None:
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    if source != "image_dark_bubble_mask":
        return None
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0 or image_rgb.ndim != 3:
        return None
    height, width = _image_hw(image_shape)
    if image_rgb.shape[:2] != (height, width):
        return None
    if not isinstance(bubble_mask, np.ndarray) or bubble_mask.shape[:2] != (height, width) or not np.any(bubble_mask):
        return None

    bubble = np.where(bubble_mask > 0, 255, 0).astype(np.uint8)
    interior = safe_bubble_interior_mask(bubble, erode_px=5)
    if not np.any(interior):
        interior = bubble
    region = interior > 0
    rgb = image_rgb.astype(np.uint8)
    rgb_f = rgb.astype(np.float32)
    luma = (rgb_f[:, :, 0] * 0.299) + (rgb_f[:, :, 1] * 0.587) + (rgb_f[:, :, 2] * 0.114)
    bg_luma = float(np.percentile(luma[region], 20)) if np.any(region) else 0.0

    bright = region & (luma >= max(188.0, bg_luma + 118.0))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
    kept = np.zeros((height, width), dtype=np.uint8)
    support_anchor = None
    if isinstance(support_mask, np.ndarray) and support_mask.shape[:2] == (height, width) and np.any(support_mask):
        anchor_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))
        support_anchor = cv2.dilate(np.where(support_mask > 0, 255, 0).astype(np.uint8), anchor_kernel, iterations=1) > 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 5 or area > 4200:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if h < 3 or w < 2 or h > 120 or w > 320:
            continue
        component = labels == label
        if support_anchor is not None and not np.any(component & support_anchor):
            continue
        kept[component] = 255
    if not np.any(kept):
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    visual = cv2.dilate(kept, kernel, iterations=1)
    visual = cv2.bitwise_and(visual, interior)
    pixels = int(np.count_nonzero(visual))
    bubble_pixels = int(np.count_nonzero(bubble))
    if pixels < 48 or pixels > int(bubble_pixels * 0.42):
        return None
    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["dark_bubble_visual_glyph_mask"] = {
            "mask_pixels": pixels,
            "bubble_pixels": bubble_pixels,
            "background_luma_p20": round(bg_luma, 3),
        }
    return visual.astype(np.uint8)


def _filter_mask_components_by_geometry(
    text_mask: np.ndarray,
    geometry_mask: np.ndarray | None,
    *,
    margin_px: int = 6,
) -> np.ndarray:
    if not isinstance(geometry_mask, np.ndarray) or text_mask.shape[:2] != geometry_mask.shape[:2]:
        return text_mask
    if not np.any(text_mask) or not np.any(geometry_mask):
        return text_mask

    margin = max(0, int(margin_px))
    geometry_bin = (geometry_mask > 0).astype(np.uint8)
    if margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
        geometry_bin = cv2.dilate(geometry_bin, kernel, iterations=1)

    source = (text_mask > 0).astype(np.uint8)
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(source, connectivity=8)
    kept = np.zeros_like(text_mask, dtype=np.uint8)
    for label in range(1, count):
        component = labels == label
        if np.any(component & (geometry_bin > 0)):
            kept[component] = 255
    return kept


def _filter_white_balloon_components_by_text_anchor(
    text_mask: np.ndarray,
    geometry_mask: np.ndarray,
    source_support: np.ndarray | None,
    bubble_mask: np.ndarray | None,
) -> np.ndarray:
    if not isinstance(text_mask, np.ndarray) or not isinstance(geometry_mask, np.ndarray):
        return text_mask
    if text_mask.shape[:2] != geometry_mask.shape[:2] or not np.any(text_mask) or not np.any(geometry_mask):
        return text_mask
    source_anchor = (
        source_support > 0
        if isinstance(source_support, np.ndarray) and source_support.shape[:2] == text_mask.shape[:2] and np.any(source_support)
        else None
    )
    outline_zone = None
    if isinstance(bubble_mask, np.ndarray) and bubble_mask.shape[:2] == text_mask.shape[:2] and np.any(bubble_mask):
        bubble_binary = np.where(bubble_mask > 0, 255, 0).astype(np.uint8)
        interior = safe_bubble_interior_mask(bubble_binary, erode_px=4)
        if np.any(interior):
            outline_zone = (bubble_binary > 0) & (interior == 0)

    source = (text_mask > 0).astype(np.uint8)
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(source, connectivity=8)
    kept = np.zeros_like(text_mask, dtype=np.uint8)
    geometry_anchor = geometry_mask > 0
    for label in range(1, count):
        component = labels == label
        area = int(np.count_nonzero(component))
        if area < 16:
            continue
        touches_outline = bool(outline_zone is not None and np.any(component & outline_zone))
        if touches_outline and area < 160:
            continue
        if np.any(component & geometry_anchor):
            kept[component] = 255
            continue
        if source_anchor is not None and area >= 16 and not touches_outline and np.any(component & source_anchor):
            kept[component] = 255
    return kept


def _numeric_bubble_id(value: object) -> int | None:
    try:
        numeric_id = int(value)
    except Exception:
        return None
    return numeric_id if numeric_id > 0 else None


def clip_mask_to_bubble_id(
    text_mask: np.ndarray,
    bubble_mask: np.ndarray | None,
    bubble_id: object,
) -> np.ndarray:
    if not isinstance(bubble_mask, np.ndarray) or bubble_mask.shape[:2] != text_mask.shape[:2]:
        return text_mask
    numeric_id = _numeric_bubble_id(bubble_id)
    if numeric_id is None:
        return text_mask
    clipped = np.where((text_mask > 0) & (bubble_mask == numeric_id), 255, 0).astype(np.uint8)
    return clipped if np.any(clipped) else text_mask


_REAL_BUBBLE_MASK_KEYS = (
    "bubble_mask",
    "bubbleMask",
    "balloon_mask",
    "balloonMask",
    "segmentation_mask",
)


def _explicit_bubble_mask_array(block: dict) -> tuple[np.ndarray | None, str | None]:
    for key in _REAL_BUBBLE_MASK_KEYS:
        value = block.get(key)
        if isinstance(value, np.ndarray):
            return value, key
    return None, None


def _page_space_bubble_mask(
    block: dict,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray | None, str | None]:
    height, width = _image_hw(image_shape)
    raw_mask, _key = _explicit_bubble_mask_array(block)
    if raw_mask is None:
        return None, "missing_bubble_mask"
    mask = raw_mask.astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] == (height, width):
        return mask, None

    bbox = _normalize_bbox(block.get("bubble_mask_bbox"), width, height)
    if not bbox:
        return None, "shape_mismatch"
    x1, y1, x2, y2 = bbox
    if mask.shape[:2] != (y2 - y1, x2 - x1):
        return None, "shape_mismatch"
    page_mask = np.zeros((height, width), dtype=np.uint8)
    page_mask[y1:y2, x1:x2] = mask
    return page_mask, None


def _real_bubble_mask_for_block(
    block: dict,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray | None, str]:
    bubble_mask, reason = _page_space_bubble_mask(block, image_shape)
    if bubble_mask is None:
        return None, reason or "missing_bubble_mask"
    labels = [int(value) for value in np.unique(bubble_mask) if int(value) > 0]
    if not labels:
        return None, "empty_bubble_mask"

    numeric_id = _numeric_bubble_id(block.get("bubble_id") or block.get("bubbleId"))
    if numeric_id is not None:
        if numeric_id in labels:
            resolved = np.where(bubble_mask == numeric_id, 255, 0).astype(np.uint8)
        elif len(labels) == 1:
            resolved = np.where(bubble_mask > 0, 255, 0).astype(np.uint8)
        else:
            return None, "bubble_id_not_found"
        return (resolved, "resolved_numeric_id") if np.any(resolved) else (None, "empty_bubble_mask")

    if len(labels) > 1:
        return None, "ambiguous_bubble_id"
    resolved = np.where(bubble_mask > 0, 255, 0).astype(np.uint8)
    return (resolved, "single_label_mask") if np.any(resolved) else (None, "empty_bubble_mask")


def _record_component_bubble_metrics(
    block: dict,
    *,
    reason: str,
    debug: dict | None = None,
) -> None:
    qa_metrics = block.setdefault("qa_metrics", {})
    if not isinstance(qa_metrics, dict):
        return
    payload = dict(debug or {})
    payload["reason"] = reason
    qa_metrics["component_bubble_cleaner"] = payload


def _build_component_bubble_cleaner_mask(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None,
    support_mask: np.ndarray | None,
    bubble_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str, dict | None]:
    real_bubble = None
    reason = "missing_bubble_mask"
    if isinstance(bubble_mask, np.ndarray) and np.any(bubble_mask):
        height, width = _image_hw(image_shape)
        candidate = bubble_mask.astype(np.uint8)
        if candidate.shape[:2] != (height, width):
            bbox = _normalize_bbox(block.get("bubble_mask_bbox"), width, height)
            if not bbox:
                return None, "derived_bubble_mask_shape_mismatch", None
            x1, y1, x2, y2 = bbox
            if candidate.shape[:2] != (y2 - y1, x2 - x1):
                return None, "derived_bubble_mask_shape_mismatch", None
            page_candidate = np.zeros((height, width), dtype=np.uint8)
            page_candidate[y1:y2, x1:x2] = candidate
            candidate = page_candidate
        real_bubble = np.where(candidate > 0, 255, 0).astype(np.uint8)
        reason = "derived_bubble_mask"
    else:
        real_bubble, reason = _real_bubble_mask_for_block(block, image_shape)
    if real_bubble is None:
        return None, reason, None
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None, "missing_image_rgb", None
    if not isinstance(support_mask, np.ndarray) or support_mask.shape[:2] != real_bubble.shape[:2] or not np.any(support_mask):
        return None, "missing_support_mask", None
    try:
        component_mask, debug = build_notanother_text_mask(image_rgb, real_bubble, support_mask)
    except ValueError as exc:
        return None, str(exc) or "component_cleaner_error", None
    if not np.any(component_mask):
        return None, "no_accepted_components", debug
    return component_mask.astype(np.uint8), "component_bubble_cleaner", debug


def _koharu_glyph_dilate_radius(block: dict) -> int:
    font_size = block.get("detected_font_size_px") or block.get("font_size")
    try:
        numeric = float(font_size)
    except Exception:
        numeric = 18.0
    return int(max(2, min(8, round(numeric * 0.16))))


def _original_text_scale_experiment_enabled() -> bool:
    return str(os.getenv("TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _text_bbox_contract_mask(block: dict, shape: tuple[int, int]) -> tuple[np.ndarray | None, list[int] | None]:
    height, width = int(shape[0]), int(shape[1])
    chosen: list[int] | None = None
    if _dark_connected_bubble_broad_bbox_risk(block):
        text_bbox = _normalize_bbox(block.get("text_pixel_bbox"), width, height)
        own_bbox = _normalize_bbox(block.get("bbox") or block.get("source_bbox"), width, height)
        if text_bbox is not None and own_bbox is not None and _bbox_area(own_bbox) >= 16:
            text_area = max(1, _bbox_area(text_bbox))
            own_area = max(1, _bbox_area(own_bbox))
            own_overlap = _bbox_intersection_area(text_bbox, own_bbox)
            if text_area > max(own_area * 1.9, own_area + 12000) and own_overlap < int(own_area * 0.70):
                chosen = own_bbox
                metrics = block.setdefault("qa_metrics", {})
                if isinstance(metrics, dict):
                    metrics["dark_connected_text_pixel_bbox_replaced_by_lobe_bbox"] = {
                        "text_pixel_bbox": list(text_bbox),
                        "lobe_bbox": list(own_bbox),
                        "text_area": text_area,
                        "lobe_area": own_area,
                        "overlap": own_overlap,
                    }
                _append_qa_flag(block, "dark_connected_text_pixel_bbox_replaced_by_lobe_bbox")
    key_order = ("text_pixel_bbox", "ocr_text_bbox", "source_bbox", "bbox")
    if chosen is None:
        for key in key_order:
            bbox = _normalize_bbox(block.get(key), width, height)
            if bbox is not None and _bbox_area(bbox) >= 16:
                chosen = bbox
                break
    if chosen is None:
        return None, None
    reference = _dark_connected_text_reference_bbox(block, width, height)
    if reference is not None:
        chosen_area = max(1, _bbox_area(chosen))
        reference_area = max(1, _bbox_area(reference))
        chosen_width = max(1, chosen[2] - chosen[0])
        reference_width = max(1, reference[2] - reference[0])
        undercovers_reference = (
            chosen_area < int(reference_area * 0.72)
            or chosen_width < int(reference_width * 0.72)
        ) and _bbox_min_overlap_ratio(chosen, reference) >= 0.15
        if undercovers_reference:
            metrics = block.setdefault("qa_metrics", {})
            if isinstance(metrics, dict):
                metrics["dark_connected_compact_text_bbox_rejected_undercoverage"] = {
                    "chosen_bbox": list(chosen),
                    "reference_bbox": list(reference),
                    "chosen_area": chosen_area,
                    "reference_area": reference_area,
                }
            _append_qa_flag(block, "dark_connected_compact_text_bbox_rejected_undercoverage")
            chosen = reference
    x1, y1, x2, y2 = [int(v) for v in chosen]
    pad_x = max(2, min(8, int(round((x2 - x1) * 0.04))))
    pad_y = max(2, min(8, int(round((y2 - y1) * 0.06))))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None, None
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask, [x1, y1, x2, y2]


def _visual_text_only_inpaint_source_required(block: dict) -> bool:
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    if source in {"image_dark_panel_mask", "image_dark_bubble_mask", "derived_card_panel_mask"}:
        return True
    profile = {
        str(block.get("layout_profile") or "").strip().lower(),
        str(block.get("block_profile") or "").strip().lower(),
        str(block.get("render_profile") or "").strip().lower(),
    }
    return bool(profile & {"dark_panel", "dark_bubble", "black_bubble", "colored_status_panel", "status_panel", "card", "title_card"})


def _dark_connected_bubble_broad_bbox_risk(block: dict) -> bool:
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    if source != "image_dark_bubble_mask":
        return False
    flags = {str(flag).strip() for flag in block.get("qa_flags") or [] if str(flag).strip()}
    connected_or_reocr_flags = {
        "dark_bubble_connected_lobes_promoted",
        "dark_bubble_lobe_mask_bbox_preferred",
        "dark_bubble_connected_lobe_passthrough",
        "detected_dark_bubble_without_text_reocr",
        "dark_bubble_full_crop_reocr_replaced",
        "leading_dark_lobe_duplicate_fragment_removed",
        "dark_bubble_visual_glyph_mask_replaced_geometry",
    }
    if flags & connected_or_reocr_flags:
        return True
    lobe_bboxes = block.get("connected_lobe_bboxes") or block.get("connectedLobeBboxes")
    return isinstance(lobe_bboxes, list) and len(lobe_bboxes) >= 2


def _dark_connected_text_reference_bbox(block: dict, width: int, height: int) -> list[int] | None:
    if not _dark_connected_bubble_broad_bbox_risk(block):
        return None

    candidates: list[list[int]] = []
    own_bbox = _normalize_bbox(block.get("bbox") or block.get("source_bbox"), width, height)
    text_bbox = _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if own_bbox is not None and text_bbox is not None and _bbox_area(own_bbox) >= 16:
        text_area = max(1, _bbox_area(text_bbox))
        own_area = max(1, _bbox_area(own_bbox))
        own_overlap = _bbox_intersection_area(text_bbox, own_bbox)
        if text_area > max(own_area * 1.9, own_area + 12000) and own_overlap < int(own_area * 0.70):
            return own_bbox
    for key in ("text_pixel_bbox", "ocr_text_bbox"):
        bbox = _normalize_bbox(block.get(key), width, height)
        if bbox is not None and _bbox_area(bbox) >= 16:
            candidates.append(bbox)

    polygon_bbox: list[int] | None = None
    for polygon in _normalize_polygons(block.get("line_polygons"), width, height):
        bbox = _polygon_bbox(polygon)
        if bbox is not None and _bbox_area(bbox) >= 16:
            polygon_bbox = bbox if polygon_bbox is None else union_bbox(polygon_bbox, bbox)
    if polygon_bbox is not None:
        candidates.append(polygon_bbox)

    metrics = block.get("qa_metrics") if isinstance(block.get("qa_metrics"), dict) else {}
    for item in metrics.get("negative_evidence") or []:
        if not isinstance(item, dict):
            continue
        bbox = _normalize_bbox(item.get("bbox"), width, height)
        if bbox is not None and _bbox_area(bbox) >= 16:
            candidates.append(bbox)
    for key in (
        "image_dark_bubble_mask",
        "dark_text_contract_fill_mask",
        "source_text_anchor_bbox",
        "source_text_mask_bbox",
    ):
        value = metrics.get(key)
        bbox_value = value.get("anchor_bbox") or value.get("bbox") if isinstance(value, dict) else value
        bbox = _normalize_bbox(bbox_value, width, height)
        if bbox is not None and _bbox_area(bbox) >= 16:
            candidates.append(bbox)

    if not candidates:
        return None

    reference = candidates[0]
    for candidate in candidates[1:]:
        if _bbox_min_overlap_ratio(reference, candidate) >= 0.25:
            reference = union_bbox(reference, candidate)
    return reference


def _mask_bbox(mask: np.ndarray | None) -> list[int] | None:
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _close_visual_text_source_mask_gaps(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return mask
    source = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel, iterations=1)
    filled = _fill_binary_holes(closed)
    if not isinstance(filled, np.ndarray) or not np.any(filled):
        filled = closed

    line_filled = _fill_visual_text_line_bands(closed)
    if isinstance(line_filled, np.ndarray) and np.any(line_filled):
        candidate = cv2.bitwise_or(filled.astype(np.uint8), line_filled.astype(np.uint8))
        source_bbox = _mask_bbox(source)
        candidate_bbox = _mask_bbox(candidate)
        source_bbox_area = max(1, _bbox_area(source_bbox))
        candidate_bbox_area = max(1, _bbox_area(candidate_bbox))
        source_pixels = max(1, int(np.count_nonzero(source)))
        candidate_pixels = int(np.count_nonzero(candidate))
        if (
            candidate_bbox_area <= int(source_bbox_area * 1.12) + 96
            and candidate_pixels <= max(source_pixels + 36000, int(source_pixels * 3.35))
        ):
            filled = candidate

    source_bbox = _mask_bbox(source)
    filled_bbox = _mask_bbox(filled)
    source_bbox_area = max(1, _bbox_area(source_bbox))
    filled_bbox_area = max(1, _bbox_area(filled_bbox))
    if filled_bbox_area > int(source_bbox_area * 1.12) + 96:
        return cv2.bitwise_or(source, closed).astype(np.uint8)
    filled_pixels = int(np.count_nonzero(filled))
    source_pixels = max(1, int(np.count_nonzero(source)))
    if filled_pixels > max(source_pixels + 36000, int(source_pixels * 3.35)):
        return cv2.bitwise_or(source, closed).astype(np.uint8)
    return cv2.bitwise_or(source, filled).astype(np.uint8)


def _fill_visual_text_line_bands(mask: np.ndarray) -> np.ndarray | None:
    """Fill line-local text bands when glyph masks outline letters instead of covering them."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or not np.any(mask):
        return None
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    height, width = binary.shape[:2]
    bridge = cv2.dilate(
        binary,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)),
        iterations=1,
    )
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(bridge, 8)
    line_mask = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, component_count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < 24 or w < 8 or h < 3:
            continue
        component = (labels[y:y + h, x:x + w] == label) & (binary[y:y + h, x:x + w] > 0)
        ys, xs = np.where(component)
        if xs.size == 0 or ys.size == 0:
            continue
        x1 = x + int(xs.min())
        x2 = x + int(xs.max()) + 1
        y1 = y + int(ys.min())
        y2 = y + int(ys.max()) + 1
        line_width = x2 - x1
        line_height = y2 - y1
        if line_width < 8 or line_height < 3:
            continue
        x_pad = 0
        y_pad = 0
        xx1 = max(0, x1 - x_pad)
        xx2 = min(width, x2 + x_pad)
        yy1 = max(0, y1 - y_pad)
        yy2 = min(height, y2 + y_pad)
        line_mask[yy1:yy2, xx1:xx2] = 255

    if not np.any(line_mask):
        return None
    return line_mask


def _dark_connected_bubble_mask_undercovers_contract(
    block: dict,
    bubble_mask: np.ndarray | None,
    contract_bbox: list[int],
) -> bool:
    if not _dark_connected_bubble_broad_bbox_risk(block):
        return False
    if not isinstance(bubble_mask, np.ndarray) or not np.any(bubble_mask):
        return False
    height, width = bubble_mask.shape[:2]
    bubble_bbox = _mask_bbox(bubble_mask) or _normalize_bbox(block.get("bubble_mask_bbox"), width, height)
    if bubble_bbox is None:
        return False
    contract_area = max(1, _bbox_area(contract_bbox))
    bubble_area = max(1, _bbox_area(bubble_bbox))
    overlap = _bbox_intersection_area(contract_bbox, bubble_bbox)
    contract_w = max(1, contract_bbox[2] - contract_bbox[0])
    contract_h = max(1, contract_bbox[3] - contract_bbox[1])
    bubble_w = max(1, bubble_bbox[2] - bubble_bbox[0])
    bubble_h = max(1, bubble_bbox[3] - bubble_bbox[1])
    width_coverage = bubble_w / float(contract_w)
    height_coverage = bubble_h / float(contract_h)
    overlap_coverage = overlap / float(contract_area)
    undercovers = bool(
        overlap_coverage < 0.55
        or width_coverage < 0.72
        or height_coverage < 0.45
        or bubble_area < int(contract_area * 0.55)
    )
    if not undercovers:
        return False
    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["dark_connected_bubble_mask_undercovered_text_contract"] = {
            "bubble_bbox": list(bubble_bbox),
            "contract_bbox": list(contract_bbox),
            "bubble_area": int(bubble_area),
            "contract_area": int(contract_area),
            "overlap_coverage": round(float(overlap_coverage), 6),
            "width_coverage": round(float(width_coverage), 6),
            "height_coverage": round(float(height_coverage), 6),
        }
    _append_qa_flag(block, "dark_connected_bubble_mask_undercovered_text_contract")
    return True


def _replace_undercovered_text_source_with_ocr_contract(
    block: dict,
    source: np.ndarray,
    *,
    bubble_mask: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    if not isinstance(block, dict) or not isinstance(source, np.ndarray) or not np.any(source):
        return source, False
    block_source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    if block_source not in {
        "image_dark_bubble_mask",
        "image_dark_panel_mask",
        "derived_card_panel_mask",
        "image_white_bubble_mask",
        "image_rect_bubble_mask",
        "image_contour_bubble_mask",
    }:
        return source, False
    height, width = source.shape[:2]
    contract_mask, contract_bbox = _text_bbox_contract_mask(block, (height, width))
    if not isinstance(contract_mask, np.ndarray) or not np.any(contract_mask) or contract_bbox is None:
        return source, False
    contract_area = _bbox_area(contract_bbox)
    if contract_area < 48 or contract_area > min(width * height * 0.42, 90000):
        return source, False
    source_bbox = _mask_bbox(source)
    if source_bbox is None:
        return source, False
    contract_w = max(1, contract_bbox[2] - contract_bbox[0])
    contract_h = max(1, contract_bbox[3] - contract_bbox[1])
    source_w = max(1, source_bbox[2] - source_bbox[0])
    source_h = max(1, source_bbox[3] - source_bbox[1])
    overlap = _bbox_intersection_area(source_bbox, contract_bbox)
    source_area = max(1, _bbox_area(source_bbox))
    contract_center = ((contract_bbox[0] + contract_bbox[2]) / 2.0, (contract_bbox[1] + contract_bbox[3]) / 2.0)
    source_center = ((source_bbox[0] + source_bbox[2]) / 2.0, (source_bbox[1] + source_bbox[3]) / 2.0)
    dx = abs(source_center[0] - contract_center[0])
    dy = abs(source_center[1] - contract_center[1])
    width_coverage = source_w / float(contract_w)
    height_coverage = source_h / float(contract_h)
    area_coverage = overlap / float(max(1, contract_area))
    source_inside_contract = overlap / float(source_area) >= 0.35
    undercovered = bool(
        source_inside_contract
        and (
            width_coverage < 0.68
            or dx > max(28.0, contract_w * 0.22)
            or (dy > max(24.0, contract_h * 0.28) and width_coverage < 0.82)
        )
    )
    if not undercovered:
        return source, False
    replacement = np.where(contract_mask > 0, 255, 0).astype(np.uint8)
    if (
        isinstance(bubble_mask, np.ndarray)
        and np.any(bubble_mask)
        and bubble_mask.shape[:2] == replacement.shape[:2]
        and not _dark_connected_bubble_mask_undercovers_contract(block, bubble_mask, contract_bbox)
    ):
        limit = safe_bubble_interior_mask(bubble_mask, erode_px=1)
        if isinstance(limit, np.ndarray) and np.any(limit):
            clipped = np.where((replacement > 0) & (limit > 0), 255, 0).astype(np.uint8)
            if int(np.count_nonzero(clipped)) >= max(32, int(np.count_nonzero(replacement) * 0.45)):
                replacement = clipped
    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["text_mask_undercovered_ocr_text"] = {
            "source_bbox": list(source_bbox),
            "contract_bbox": list(contract_bbox),
            "width_coverage": round(float(width_coverage), 6),
            "height_coverage": round(float(height_coverage), 6),
            "area_coverage": round(float(area_coverage), 6),
            "source_pixels": int(np.count_nonzero(source)),
            "contract_pixels": int(np.count_nonzero(replacement)),
        }
    _append_qa_flag(block, "text_mask_undercovered_ocr_text")
    return replacement.astype(np.uint8), True


def _limit_dark_connected_mask_overreach(
    block: dict,
    candidate: np.ndarray,
    source: np.ndarray,
    geometry: np.ndarray | None,
    limit_mask: np.ndarray | None,
    *,
    radius: int,
) -> np.ndarray:
    if not _dark_connected_bubble_broad_bbox_risk(block):
        return candidate
    if not isinstance(candidate, np.ndarray) or not np.any(candidate):
        return candidate
    if not isinstance(source, np.ndarray) or not np.any(source):
        return candidate

    anchor = np.where(source > 0, 255, 0).astype(np.uint8)
    source_pixels = int(np.count_nonzero(anchor))
    if isinstance(geometry, np.ndarray) and np.any(geometry):
        geometry_u8 = np.where(geometry > 0, 255, 0).astype(np.uint8)
        geometry_pixels = int(np.count_nonzero(geometry_u8))
        if geometry_pixels <= max(source_pixels + 2400, int(round(source_pixels * 1.65))):
            anchor = cv2.bitwise_or(anchor, geometry_u8)
            source_pixels = int(np.count_nonzero(anchor))

    height, width = anchor.shape[:2]
    target_bbox = (
        _normalize_bbox(block.get("bbox"), width, height)
        or _normalize_bbox(block.get("source_bbox"), width, height)
        or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    )
    if target_bbox is not None:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats((anchor > 0).astype(np.uint8), connectivity=8)
        if count > 2:
            tx = (target_bbox[0] + target_bbox[2]) / 2.0
            ty = (target_bbox[1] + target_bbox[3]) / 2.0
            components: list[tuple[float, int, int]] = []
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < 12:
                    continue
                cx, cy = centroids[label]
                dist = ((float(cx) - tx) ** 2 + (float(cy) - ty) ** 2) ** 0.5
                components.append((dist, label, area))
            if components:
                components.sort(key=lambda item: item[0])
                best_dist = max(1.0, float(components[0][0]))
                keep_labels = {
                    label
                    for dist, label, area in components
                    if dist <= max(best_dist * 1.35, best_dist + 48.0)
                    or (
                        target_bbox[0] <= float(centroids[label][0]) <= target_bbox[2]
                        and target_bbox[1] <= float(centroids[label][1]) <= target_bbox[3]
                        and dist <= best_dist + 96.0
                    )
                }
                if keep_labels and len(keep_labels) < len(components):
                    filtered = np.where(np.isin(labels, list(keep_labels)), 255, 0).astype(np.uint8)
                    if np.any(filtered):
                        metrics = block.setdefault("qa_metrics", {})
                        if isinstance(metrics, dict):
                            metrics["dark_connected_bubble_anchor_component_filter"] = {
                                "components": int(len(components)),
                                "kept": int(len(keep_labels)),
                                "target_bbox": list(target_bbox),
                                "before_pixels": int(np.count_nonzero(anchor)),
                                "after_pixels": int(np.count_nonzero(filtered)),
                            }
                        _append_qa_flag(block, "dark_connected_lobe_anchor_component_filtered")
                        anchor = filtered

    anchor_pixels = int(np.count_nonzero(anchor))
    candidate_pixels = int(np.count_nonzero(candidate))
    if anchor_pixels <= 0:
        return candidate

    anchor_bbox = _mask_bbox(anchor)
    candidate_bbox = _mask_bbox(candidate)
    anchor_bbox_area = max(1, _bbox_area(anchor_bbox))
    candidate_bbox_area = max(1, _bbox_area(candidate_bbox))
    broad_by_pixels = candidate_pixels > max(anchor_pixels + 2400, int(round(anchor_pixels * 2.75)))
    broad_by_bbox = candidate_bbox_area > max(anchor_bbox_area + 4800, int(round(anchor_bbox_area * 2.25)))
    if not (broad_by_pixels or broad_by_bbox):
        return candidate

    rebuilt_limit = limit_mask if isinstance(limit_mask, np.ndarray) and limit_mask.shape[:2] == anchor.shape[:2] else None
    rebuilt = expand_text_mask_monotonic(anchor, limit_mask=rebuilt_limit, radius=max(2, min(7, int(radius or 0))))
    if not isinstance(rebuilt, np.ndarray) or not np.any(rebuilt):
        return candidate
    rebuilt_pixels = int(np.count_nonzero(rebuilt))
    if rebuilt_pixels >= candidate_pixels:
        return candidate

    metrics = block.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["dark_connected_bubble_broad_mask_rejected"] = {
            "candidate_pixels": candidate_pixels,
            "anchor_pixels": anchor_pixels,
            "rebuilt_pixels": rebuilt_pixels,
            "candidate_bbox": candidate_bbox,
            "anchor_bbox": anchor_bbox,
            "candidate_bbox_area": candidate_bbox_area,
            "anchor_bbox_area": anchor_bbox_area,
        }
    _append_qa_flag(block, "broad_connected_bubble_mask_rejected")
    _append_qa_flag(block, "dark_connected_lobe_mask_rebuilt_from_glyphs")
    return rebuilt.astype(np.uint8)


def _enforce_visual_text_only_inpaint_contract(
    block: dict,
    text_mask: np.ndarray,
    source_mask: np.ndarray | None,
    bubble_mask: np.ndarray | None,
    *,
    radius: int,
) -> np.ndarray | None:
    if not _visual_text_only_inpaint_source_required(block):
        return text_mask
    metrics = block.setdefault("qa_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        block["qa_metrics"] = metrics
    if not isinstance(source_mask, np.ndarray) or not np.any(source_mask):
        if _dark_connected_bubble_broad_bbox_risk(block):
            bbox_mask, bbox_used = _text_bbox_contract_mask(block, text_mask.shape[:2])
            if isinstance(bbox_mask, np.ndarray) and np.any(bbox_mask):
                source = np.where(bbox_mask > 0, 255, 0).astype(np.uint8)
                limit_mask = None
                if (
                    isinstance(bubble_mask, np.ndarray)
                    and np.any(bubble_mask)
                    and bubble_mask.shape[:2] == source.shape[:2]
                ):
                    limit_mask = safe_bubble_interior_mask(bubble_mask, erode_px=1)
                expand_radius = max(5, min(11, int(radius or 0) + 3))
                expanded = expand_text_mask_monotonic(source, limit_mask=limit_mask, radius=expand_radius)
                if not isinstance(expanded, np.ndarray) or not np.any(expanded):
                    expanded = source
                expanded = _limit_dark_connected_mask_overreach(
                    block,
                    expanded,
                    source,
                    None,
                    limit_mask,
                    radius=expand_radius,
                )
                source_pixels = int(np.count_nonzero(source))
                expanded_pixels = int(np.count_nonzero(expanded))
                metrics["inpaint_mask_contract"] = {
                    "contract": "expanded_text_mask",
                    "source": "missing_glyph_lobe_bbox",
                    "source_bbox": bbox_used,
                    "source_pixels": source_pixels,
                    "expanded_pixels": expanded_pixels,
                    "previous_pixels": int(np.count_nonzero(text_mask)) if isinstance(text_mask, np.ndarray) else 0,
                    "area_fallback_used": False,
                    "experimental_original_text_scale_expand": True,
                }
                _append_qa_flag(block, "visual_text_only_inpaint_contract")
                _append_qa_flag(block, "dark_connected_missing_glyph_bbox_contract")
                consolidate_mask_evidence(
                    block,
                    kind="bbox_contract",
                    raw_mask_pixels=source_pixels,
                    expanded_mask_pixels=expanded_pixels,
                    evidence_score=0.72,
                    fast_fill_reject_reasons=[],
                )
                return expanded.astype(np.uint8)
        metrics["inpaint_mask_contract"] = {
            "contract": "expanded_text_mask",
            "source_pixels": 0,
            "expanded_pixels": 0,
            "area_fallback_used": False,
            "rejected_reason": "missing_text_glyph_source",
        }
        _append_qa_flag(block, "visual_text_only_inpaint_missing_glyph_source")
        consolidate_mask_evidence(
            block,
            kind="none",
            raw_mask_pixels=0,
            expanded_mask_pixels=0,
            evidence_score=0.0,
            fast_fill_reject_reasons=["missing_text_glyph_source"],
        )
        return None
    source_before_gap_close_pixels = int(np.count_nonzero(source_mask))
    source = _close_visual_text_source_mask_gaps(source_mask)
    source_after_gap_close_pixels = int(np.count_nonzero(source))
    if source_after_gap_close_pixels != source_before_gap_close_pixels:
        metrics["visual_text_source_gap_close"] = {
            "source_pixels_before": source_before_gap_close_pixels,
            "source_pixels_after": source_after_gap_close_pixels,
        }
    geometry = build_glyph_text_mask(block, source.shape[:2])
    line_geometry_pixels = 0
    skip_bubble_id_clip = False
    compact_dark_connected_source_replaced = False
    if block.get("line_polygons"):
        for polygon in block.get("line_polygons") or []:
            try:
                xs = [int(round(float(point[0]))) for point in polygon]
                ys = [int(round(float(point[1]))) for point in polygon]
            except Exception:
                continue
            if xs and ys:
                line_geometry_pixels += max(0, max(xs) - min(xs)) * max(0, max(ys) - min(ys))
    if _original_text_scale_experiment_enabled():
        bbox_mask, bbox_used = _text_bbox_contract_mask(block, source.shape[:2])
        if isinstance(bbox_mask, np.ndarray) and np.any(bbox_mask):
            bbox_area = max(1, int(np.count_nonzero(bbox_mask)))
            current_overlap = int(np.count_nonzero((source > 0) & (bbox_mask > 0)))
            source_pixels_before_union = int(np.count_nonzero(source))
            source_bbox_area = max(1, _bbox_area(_mask_bbox(source)))
            compact_dark_connected_bbox = (
                _dark_connected_bubble_broad_bbox_risk(block)
                and bbox_area < int(source_bbox_area * 0.55)
                and current_overlap < int(bbox_area * 0.72)
            )
            undercovered_dark_connected_bbox = (
                _dark_connected_bubble_broad_bbox_risk(block)
                and source_pixels_before_union < int(bbox_area * 0.72)
                and bbox_area <= max(source_pixels_before_union + 12000, int(source_pixels_before_union * 2.25))
                and bbox_area <= 60000
            )
            text_reference_bbox = _dark_connected_text_reference_bbox(block, source.shape[1], source.shape[0])
            reliable_dark_connected_text_bbox = (
                _dark_connected_bubble_broad_bbox_risk(block)
                and text_reference_bbox is not None
                and bbox_used is not None
                and _bbox_min_overlap_ratio(bbox_used, text_reference_bbox) >= 0.82
                and bbox_area <= max(_bbox_area(text_reference_bbox) + 4200, int(_bbox_area(text_reference_bbox) * 1.45))
                and bbox_area <= 60000
            )
            tiny_source_dark_connected_bbox = (
                reliable_dark_connected_text_bbox
                and source_pixels_before_union < int(bbox_area * 0.42)
            )
            broad_dark_connected_bbox = (
                _dark_connected_bubble_broad_bbox_risk(block)
                and bbox_area > max(source_pixels_before_union + 4800, int(source_pixels_before_union * 2.25))
                and not tiny_source_dark_connected_bbox
            )
            if compact_dark_connected_bbox or undercovered_dark_connected_bbox or tiny_source_dark_connected_bbox:
                source = bbox_mask
                compact_dark_connected_source_replaced = True
                metrics["inpaint_mask_contract_text_bbox_replaced_aggregate_source"] = {
                    "bbox": bbox_used,
                    "bbox_pixels": bbox_area,
                    "source_bbox_area_before": source_bbox_area,
                    "source_overlap_before": current_overlap,
                    "source_pixels_before_replace": source_pixels_before_union,
                    "source_pixels_after_replace": int(np.count_nonzero(source)),
                    "reason": (
                        "dark_connected_bubble_tiny_source_text_reference_bbox"
                        if tiny_source_dark_connected_bbox
                        else
                        "dark_connected_bubble_undercovered_compact_bbox"
                        if undercovered_dark_connected_bbox
                        else "dark_connected_bubble_compact_bbox"
                    ),
                }
                _append_qa_flag(block, "dark_connected_bubble_compact_bbox_replaced_aggregate_source")
            elif broad_dark_connected_bbox:
                metrics["inpaint_mask_contract_text_bbox_union_skipped"] = {
                    "bbox": bbox_used,
                    "bbox_pixels": bbox_area,
                    "source_overlap_before": current_overlap,
                    "source_pixels_before_union": source_pixels_before_union,
                    "reason": "dark_connected_bubble_broad_bbox",
                }
                _append_qa_flag(block, "dark_connected_bubble_broad_bbox_union_skipped")
            elif current_overlap < int(bbox_area * 0.42) or bbox_area > int(source_pixels_before_union * 1.35):
                source = cv2.bitwise_or(source, bbox_mask)
                skip_bubble_id_clip = True
                metrics["inpaint_mask_contract_text_bbox_union"] = {
                    "bbox": bbox_used,
                    "bbox_pixels": bbox_area,
                    "source_overlap_before": current_overlap,
                    "source_pixels_before_union": source_pixels_before_union,
                    "source_pixels_after_union": int(np.count_nonzero(source)),
                }
    if _original_text_scale_experiment_enabled() and isinstance(geometry, np.ndarray) and np.any(geometry):
        geometry_u8 = np.where(geometry > 0, 255, 0).astype(np.uint8)
        geometry_pixels = int(np.count_nonzero(geometry_u8))
        source_pixels_before_geometry = int(np.count_nonzero(source))
        broad_dark_connected_geometry = (
            _dark_connected_bubble_broad_bbox_risk(block)
            and geometry_pixels > max(
                source_pixels_before_geometry + 4800,
                int(source_pixels_before_geometry * 2.25),
            )
        )
        compact_replacement_geometry_mismatch = (
            compact_dark_connected_source_replaced
            and geometry_pixels > max(
                source_pixels_before_geometry + 1200,
                int(source_pixels_before_geometry * 1.25),
            )
        )
        if broad_dark_connected_geometry or compact_replacement_geometry_mismatch:
            metrics["inpaint_mask_contract_geometry_union_skipped"] = {
                "geometry_pixels": geometry_pixels,
                "source_pixels_before_union": source_pixels_before_geometry,
                "reason": (
                    "dark_connected_bubble_compact_bbox_geometry_mismatch"
                    if compact_replacement_geometry_mismatch
                    else "dark_connected_bubble_broad_geometry"
                ),
            }
            _append_qa_flag(block, "dark_connected_bubble_broad_geometry_union_skipped")
        else:
            source = cv2.bitwise_or(source, geometry_u8)
            metrics["inpaint_mask_contract_geometry_union"] = {
                "source_pixels_after_union": int(np.count_nonzero(source)),
                "geometry_pixels": geometry_pixels,
            }
    source, ocr_contract_replaced_source = _replace_undercovered_text_source_with_ocr_contract(
        block,
        source,
        bubble_mask=bubble_mask,
    )
    if ocr_contract_replaced_source:
        skip_bubble_id_clip = True
    if isinstance(geometry, np.ndarray) and np.any(geometry):
        source_pixels_before = int(np.count_nonzero(source))
        geometry_pixels = line_geometry_pixels or int(np.count_nonzero(geometry))
        if (
            not _original_text_scale_experiment_enabled()
            and geometry_pixels > 0
            and source_pixels_before >= int(geometry_pixels * 0.80)
        ):
            eroded = cv2.erode(
                source,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            eroded_pixels = int(np.count_nonzero(eroded))
            if eroded_pixels >= max(64, int(source_pixels_before * 0.55)):
                source = eroded.astype(np.uint8)
                metrics["inpaint_mask_contract_source_eroded"] = {
                    "source_pixels_before": source_pixels_before,
                    "source_pixels_after": eroded_pixels,
                    "geometry_pixels": geometry_pixels,
                }
    limit_mask = None
    if isinstance(bubble_mask, np.ndarray) and np.any(bubble_mask) and bubble_mask.shape[:2] == source.shape[:2]:
        limit_mask = safe_bubble_interior_mask(bubble_mask, erode_px=1)
    expand_radius = max(3, min(7, int(radius or 0)))
    if _original_text_scale_experiment_enabled():
        expand_radius = max(expand_radius, min(11, int(expand_radius) + 3))
    expanded = expand_text_mask_monotonic(source, limit_mask=limit_mask, radius=expand_radius)
    if not isinstance(expanded, np.ndarray) or not np.any(expanded):
        expanded = source
    expanded = _limit_dark_connected_mask_overreach(
        block,
        expanded,
        source,
        geometry,
        limit_mask,
        radius=expand_radius,
    )
    if (
        not skip_bubble_id_clip
        and isinstance(bubble_mask, np.ndarray)
        and np.any(bubble_mask)
        and bubble_mask.shape[:2] == expanded.shape[:2]
    ):
        expanded = clip_mask_to_bubble_id(expanded, bubble_mask, block.get("bubble_id") or block.get("bubbleId"))
    if (
        not _original_text_scale_experiment_enabled()
        and line_geometry_pixels > 0
        and int(np.count_nonzero(expanded)) >= int(line_geometry_pixels * 0.82)
    ):
        capped = source
        minimal_expanded = expand_text_mask_monotonic(capped, limit_mask=limit_mask, radius=2)
        expanded = minimal_expanded.astype(np.uint8) if isinstance(minimal_expanded, np.ndarray) and np.any(minimal_expanded) else capped.astype(np.uint8)
    source_pixels = int(np.count_nonzero(source))
    expanded_pixels = int(np.count_nonzero(expanded))
    current_pixels = int(np.count_nonzero(text_mask)) if isinstance(text_mask, np.ndarray) else 0
    metrics["inpaint_mask_contract"] = {
        "contract": "expanded_text_mask",
        "source": "ocr_bbox_undercoverage_contract" if ocr_contract_replaced_source else "visual_text_source",
        "source_pixels": source_pixels,
        "expanded_pixels": expanded_pixels,
        "previous_pixels": current_pixels,
        "area_fallback_used": False,
        "experimental_original_text_scale_expand": bool(_original_text_scale_experiment_enabled()),
    }
    _append_qa_flag(block, "visual_text_only_inpaint_contract")
    return expanded.astype(np.uint8)


def build_inpaint_mask(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None = None,
) -> np.ndarray | None:
    block = _drop_isolated_side_note_line_polygons(block)
    height, width = _image_hw(image_shape)
    block = _detach_mixed_sfx_from_white_bubble_block(block, width, height)
    block = _filter_dark_bubble_connected_lobe_line_polygons(block, width, height)
    if "ocr_gibberish" in {str(flag).strip() for flag in block.get("qa_flags") or []}:
        consolidate_mask_evidence(
            block,
            kind="none",
            raw_mask_pixels=0,
            expanded_mask_pixels=0,
            evidence_score=0.0,
            fast_fill_reject_reasons=["ocr_gibberish_suppressed"],
        )
        return None

    if _merged_fragment_without_glyph_evidence(block):
        consolidate_mask_evidence(
            block,
            kind="none",
            raw_mask_pixels=0,
            expanded_mask_pixels=0,
            evidence_score=0.0,
            fast_fill_reject_reasons=["same_balloon_fragment_without_glyph_evidence"],
        )
        return None

    if (
        isinstance(image_rgb, np.ndarray)
        and image_rgb.size
        and _block_special_class(block) == "sfx"
    ):
        try:
            from sfx.mask import build_sfx_glyph_mask
        except ImportError:
            from ..sfx.mask import build_sfx_glyph_mask

        sfx_result = build_sfx_glyph_mask(image_rgb, block)
        sfx_evidence = dict(sfx_result.evidence)
        reject_reason = sfx_evidence.get("reject_reason")
        evidence = consolidate_mask_evidence(
            block,
            kind="sfx_glyph_mask",
            raw_mask_pixels=int(sfx_evidence.get("raw_mask_pixels") or 0),
            expanded_mask_pixels=int(sfx_evidence.get("expanded_mask_pixels") or 0),
            evidence_score=1.0 if sfx_result.mask is not None else 0.0,
            fast_fill_reject_reasons=[str(reject_reason)] if reject_reason else None,
        )
        evidence.update(
            {
                "bbox_fill_ratio": float(sfx_evidence.get("bbox_fill_ratio") or 0.0),
                "component_count": int(sfx_evidence.get("component_count") or 0),
            }
        )
        if reject_reason:
            evidence["reject_reason"] = str(reject_reason)
        block["mask_evidence"] = evidence
        return sfx_result.mask

    text_mask = None
    raw_text_mask = None
    visual_text_only_source_mask = None
    used_component_bubble_cleaner = False
    explicit_bubble_mask, _bubble_mask_key = _explicit_bubble_mask_array(block)
    forced_rect_panel_mask = None
    if _dark_bubble_source_should_remain_rect_panel(block):
        _promote_rect_dark_panel_contract(block, width=width, height=height)
        forced_rect_panel_mask = balloon_mask_from_block(block, image_shape)
        explicit_bubble_mask = None
        block.pop("bubble_mask", None)
        block.pop("bubbleMask", None)
    translator_note_text_only_mask = _translator_note_uses_text_only_mask(block, image_rgb, image_shape)
    if translator_note_text_only_mask:
        explicit_bubble_mask = None
        block["bubble_mask_source"] = "translator_note_text_mask"
        block["bubbleMaskSource"] = "translator_note_text_mask"
        _append_qa_flag(block, "translator_note_text_only_mask")
    real_bubble_mask = None
    synthetic_card_bubble_reference = bool(
        explicit_bubble_mask is not None
        and _synthetic_tight_card_bubble_reference(block, image_shape)
    )
    if explicit_bubble_mask is not None and not synthetic_card_bubble_reference:
        real_bubble, bubble_reason = _real_bubble_mask_for_block(block, image_shape)
        if real_bubble is None:
            _record_component_bubble_metrics(block, reason=bubble_reason)
            consolidate_mask_evidence(
                block,
                kind="none",
                raw_mask_pixels=0,
                expanded_mask_pixels=0,
                evidence_score=0.0,
            )
            return None
        real_bubble_mask = real_bubble

    geometry_mask = build_glyph_text_mask(block, image_shape)
    if image_rgb is not None and image_rgb.size:
        raw_text_mask = build_raw_text_mask_from_image(block, image_rgb, image_shape)
        if isinstance(raw_text_mask, np.ndarray) and np.any(raw_text_mask):
            if isinstance(geometry_mask, np.ndarray) and np.any(geometry_mask):
                filtered_raw = _filter_mask_components_by_geometry(
                    raw_text_mask,
                    geometry_mask,
                    margin_px=max(18, _koharu_glyph_dilate_radius(block) + 8),
                )
                if np.any(filtered_raw):
                    if int(np.count_nonzero(filtered_raw)) < int(np.count_nonzero(raw_text_mask)):
                        metrics = block.setdefault("qa_metrics", {})
                        if isinstance(metrics, dict):
                            metrics["raw_text_component_filter"] = {
                                "source_pixels": int(np.count_nonzero(raw_text_mask)),
                                "filtered_pixels": int(np.count_nonzero(filtered_raw)),
                            }
                    raw_text_mask = filtered_raw.astype(np.uint8)
            text_mask = raw_text_mask.astype(np.uint8)
            visual_text_only_source_mask = text_mask.astype(np.uint8)
            if (
                isinstance(geometry_mask, np.ndarray)
                and np.any(geometry_mask)
                and _mask_is_overbroad_against_geometry(text_mask, geometry_mask)
            ):
                text_mask = geometry_mask.astype(np.uint8)
        elif raw_text_evidence_rejected(block):
            if (
                (translator_note_text_only_mask or _card_like_dark_context(block))
                and isinstance(geometry_mask, np.ndarray)
                and np.any(geometry_mask)
            ):
                text_mask = geometry_mask.astype(np.uint8)
                metrics = block.setdefault("qa_metrics", {})
                if isinstance(metrics, dict):
                    metrics["raw_text_evidence_geometry_fallback"] = {
                        "geometry_pixels": int(np.count_nonzero(geometry_mask)),
                        "reason": "translator_note_text_only_mask" if translator_note_text_only_mask else "card_like_dark_context",
                    }
                _remove_qa_flags(
                    block,
                    {
                        RAW_TEXT_EVIDENCE_MISSING_FLAG,
                        RAW_TEXT_EVIDENCE_REJECTED_FLAG,
                    },
                )
            else:
                consolidate_mask_evidence(
                    block,
                    kind="none",
                    raw_mask_pixels=0,
                    expanded_mask_pixels=0,
                    evidence_score=0.0,
                )
                return None

    if text_mask is None:
        text_mask = geometry_mask
    if text_mask is None:
        consolidate_mask_evidence(
            block,
            kind="none",
            raw_mask_pixels=0,
            expanded_mask_pixels=0,
            evidence_score=0.0,
        )
        return None
    if translator_note_text_only_mask and isinstance(geometry_mask, np.ndarray) and np.any(geometry_mask):
        text_mask = cv2.bitwise_or(text_mask.astype(np.uint8), geometry_mask.astype(np.uint8))
        if visual_text_only_source_mask is None or not np.any(visual_text_only_source_mask):
            visual_text_only_source_mask = text_mask.astype(np.uint8)
        metrics = block.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["translator_note_text_only_contract"] = {
                "source": "raw_or_geometry_text_mask",
                "text_pixels": int(np.count_nonzero(text_mask)),
                "geometry_pixels": int(np.count_nonzero(geometry_mask)),
            }
    if isinstance(geometry_mask, np.ndarray) and np.any(geometry_mask):
        floored_mask, floor_metrics = _white_balloon_geometry_floor(block, text_mask, geometry_mask, image_shape)
        if floor_metrics is not None:
            text_mask = floored_mask
            metrics = block.setdefault("qa_metrics", {})
            if isinstance(metrics, dict):
                metrics["white_balloon_geometry_floor"] = floor_metrics
    raw_floor_mask = raw_text_mask if isinstance(raw_text_mask, np.ndarray) and np.any(raw_text_mask) else text_mask.astype(np.uint8)
    visual_card_text_mask = _colored_card_visual_glyph_mask(
        block,
        image_rgb,
        image_shape,
        geometry_mask if isinstance(geometry_mask, np.ndarray) and np.any(geometry_mask) else text_mask,
    )
    if (
        isinstance(visual_card_text_mask, np.ndarray)
        and np.any(visual_card_text_mask)
        and not translator_note_text_only_mask
    ):
        visual_pixels = int(np.count_nonzero(visual_card_text_mask))
        raw_pixels = int(np.count_nonzero(raw_floor_mask))
        if visual_pixels >= max(64, int(round(raw_pixels * 0.32))):
            text_mask = visual_card_text_mask
            raw_floor_mask = visual_card_text_mask
            visual_text_only_source_mask = visual_card_text_mask.astype(np.uint8)
            qa_metrics = block.setdefault("qa_metrics", {})
            if isinstance(qa_metrics, dict):
                qa_metrics["colored_card_visual_glyph_mask"] = {
                    "mask_pixels": visual_pixels,
                }
        else:
            visual_card_text_mask = None

    component_support = geometry_mask if isinstance(geometry_mask, np.ndarray) and np.any(geometry_mask) else text_mask
    if (
        isinstance(visual_card_text_mask, np.ndarray)
        and np.any(visual_card_text_mask)
        and not translator_note_text_only_mask
    ):
        component_support = visual_card_text_mask
    derived_bubble_mask = forced_rect_panel_mask
    if (
        isinstance(real_bubble_mask, np.ndarray)
        and np.any(real_bubble_mask)
        and _explicit_bubble_mask_should_yield_to_card_panel(
            block,
            image_shape,
            image_rgb,
            real_bubble_mask,
            raw_floor_mask,
            geometry_mask,
        )
    ):
        replacement_mask = _derive_card_panel_mask_from_image(
            block,
            image_shape,
            image_rgb,
            component_support,
            allow_inferred_card=True,
        )
        if isinstance(replacement_mask, np.ndarray) and np.any(replacement_mask):
            derived_bubble_mask = replacement_mask
            real_bubble_mask = None
            explicit_bubble_mask = None
    dark_text_without_balloon = _dark_text_without_balloon_context(block)
    dark_profile_values = {
        str(block.get("layout_profile") or "").strip().lower(),
        str(block.get("block_profile") or "").strip().lower(),
        str(block.get("render_profile") or "").strip().lower(),
    }
    prefer_dark_panel_mask = bool(
        block.get("card_panel_text_context")
        or dark_profile_values & {"dark_panel", "colored_status_panel", "status_panel", "card", "title_card"}
    )
    prefer_dark_bubble_mask = bool(dark_profile_values & {"dark_bubble", "black_bubble"})
    allow_derived_card_context = not _is_translator_note_block(block) and not dark_text_without_balloon
    allow_derived_white_context = not translator_note_text_only_mask
    if explicit_bubble_mask is None or synthetic_card_bubble_reference:
        if allow_derived_card_context and _card_like_dark_context(block):
            derived_bubble_mask = _derive_dark_panel_mask_from_border_lines(
                block,
                image_shape,
                image_rgb,
                component_support,
            )
        if (
            allow_derived_card_context
            and not (isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask))
            and _card_like_dark_context(block)
        ):
            derived_bubble_mask = _derive_dark_panel_mask_from_negative_geometry(
                block,
                image_shape,
                image_rgb,
                component_support,
            )
        if (
            allow_derived_card_context
            and not (isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask))
            and _card_like_dark_context(block)
            and prefer_dark_panel_mask
            and not prefer_dark_bubble_mask
        ):
            derived_bubble_mask = _derive_dark_panel_mask_from_balloon_bbox(
                block,
                image_shape,
                image_rgb,
                component_support,
            )
        if (
            allow_derived_card_context
            and not (isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask))
            and _card_like_dark_context(block)
            and prefer_dark_panel_mask
            and not prefer_dark_bubble_mask
        ):
            derived_bubble_mask = _derive_card_panel_mask_from_image(
                block,
                image_shape,
                image_rgb,
                component_support,
                allow_inferred_card=True,
            )
        if (
            allow_derived_card_context
            and not (isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask))
            and not prefer_dark_panel_mask
            and prefer_dark_bubble_mask
        ):
            derived_bubble_mask = _derive_dark_panel_mask_from_border_lines(
                block,
                image_shape,
                image_rgb,
                component_support,
                allow_inferred_card=True,
            )
        if (
            allow_derived_card_context
            and not (isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask))
            and not prefer_dark_panel_mask
            and prefer_dark_bubble_mask
        ):
            derived_bubble_mask = _derive_dark_panel_mask_from_negative_geometry(
                block,
                image_shape,
                image_rgb,
                component_support,
                allow_inferred_card=True,
            )
        if (
            allow_derived_card_context
            and not (isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask))
            and _card_like_dark_context(block)
            and not prefer_dark_panel_mask
        ):
            derived_bubble_mask = _derive_dark_oval_bubble_mask_from_negative_geometry(
                block,
                image_shape,
                image_rgb,
                component_support,
            )
        if (
            allow_derived_card_context
            and not (isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask))
            and not prefer_dark_panel_mask
        ):
            derived_bubble_mask = _derive_dark_ellipse_mask_from_balloon_bbox(
                block,
                image_shape,
                image_rgb,
                component_support,
            )
        if (
            allow_derived_card_context
            and not (isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask))
            and _card_like_dark_context(block)
        ):
            derived_bubble_mask = _derive_card_panel_mask_from_image(
                block,
                image_shape,
                image_rgb,
                component_support,
            )
        if allow_derived_white_context and not (isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask)):
            derived_bubble_mask = _derive_white_bubble_mask_from_image(
                block,
                image_shape,
                image_rgb,
                component_support,
            )
        if allow_derived_card_context and not (isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask)):
            derived_bubble_mask = _derive_card_panel_mask_from_image(
                block,
                image_shape,
                image_rgb,
                component_support,
            )
        if isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask) and isinstance(geometry_mask, np.ndarray) and np.any(geometry_mask):
            floored_mask, floor_metrics = _white_balloon_geometry_floor(block, text_mask, geometry_mask, image_shape)
            if floor_metrics is not None:
                text_mask = floored_mask
                raw_floor_mask = text_mask.astype(np.uint8)
                qa_metrics = block.setdefault("qa_metrics", {})
                if isinstance(qa_metrics, dict):
                    qa_metrics["white_balloon_geometry_floor_after_image_derivation"] = floor_metrics
    dark_visual_bubble_mask = (
        real_bubble_mask
        if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask)
        else derived_bubble_mask
    )
    dark_bubble_visual_mask = _dark_bubble_visual_glyph_mask(
        block,
        image_rgb,
        image_shape,
        dark_visual_bubble_mask,
        component_support,
    )
    if isinstance(dark_bubble_visual_mask, np.ndarray) and np.any(dark_bubble_visual_mask):
        visual_pixels = int(np.count_nonzero(dark_bubble_visual_mask))
        current_pixels = int(np.count_nonzero(text_mask))
        geometry_pixels = int(np.count_nonzero(geometry_mask)) if isinstance(geometry_mask, np.ndarray) else 0
        visual_replaces_geometry = bool(
            current_pixels > 0
            and geometry_pixels > 0
            and visual_pixels >= max(48, int(round(geometry_pixels * 0.08)))
            and visual_pixels <= int(max(geometry_pixels * 0.82, geometry_pixels - 96))
        )
        if visual_replaces_geometry:
            text_mask = dark_bubble_visual_mask.astype(np.uint8)
            raw_floor_mask = dark_bubble_visual_mask.astype(np.uint8)
            visual_text_only_source_mask = dark_bubble_visual_mask.astype(np.uint8)
            qa_metrics = block.setdefault("qa_metrics", {})
            if isinstance(qa_metrics, dict):
                qa_metrics["dark_bubble_visual_glyph_mask_replaced_geometry"] = {
                    "visual_pixels": visual_pixels,
                    "geometry_pixels": geometry_pixels,
                    "previous_pixels": current_pixels,
                }
            _append_qa_flag(block, "dark_bubble_visual_glyph_mask_replaced_geometry")
        else:
            text_mask = cv2.bitwise_or(text_mask.astype(np.uint8), dark_bubble_visual_mask.astype(np.uint8))
            raw_floor_mask = cv2.bitwise_or(raw_floor_mask.astype(np.uint8), dark_bubble_visual_mask.astype(np.uint8))
            visual_text_only_source_mask = (
                cv2.bitwise_or(visual_text_only_source_mask.astype(np.uint8), dark_bubble_visual_mask.astype(np.uint8))
                if isinstance(visual_text_only_source_mask, np.ndarray) and np.any(visual_text_only_source_mask)
                else dark_bubble_visual_mask.astype(np.uint8)
            )
        visual_card_text_mask = (
            cv2.bitwise_or(visual_card_text_mask.astype(np.uint8), dark_bubble_visual_mask.astype(np.uint8))
            if isinstance(visual_card_text_mask, np.ndarray) and np.any(visual_card_text_mask)
            else dark_bubble_visual_mask.astype(np.uint8)
        )
    skip_component_bubble_cleaner = translator_note_text_only_mask or (
        isinstance(visual_card_text_mask, np.ndarray) and np.any(visual_card_text_mask)
    )
    component_mask = None
    component_reason = "visual_card_glyph_mask"
    component_debug = None
    if not skip_component_bubble_cleaner:
        component_mask, component_reason, component_debug = _build_component_bubble_cleaner_mask(
            block,
            image_shape,
            image_rgb,
            component_support,
            bubble_mask=real_bubble_mask if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask) else derived_bubble_mask,
        )
    elif isinstance(block, dict):
        qa_metrics = block.setdefault("qa_metrics", {})
        if isinstance(qa_metrics, dict):
            if translator_note_text_only_mask:
                qa_metrics["component_bubble_cleaner_skipped_for_translator_note_text_mask"] = {
                    "text_pixels": int(np.count_nonzero(text_mask)),
                }
            else:
                qa_metrics["component_bubble_cleaner_skipped_for_visual_card_mask"] = {
                    "visual_pixels": int(np.count_nonzero(visual_card_text_mask)),
                }
    if component_debug is not None or component_reason not in {"missing_bubble_mask", "missing_image_rgb", "missing_support_mask"}:
        _record_component_bubble_metrics(block, reason=component_reason, debug=component_debug)
    if isinstance(component_mask, np.ndarray) and np.any(component_mask):
        component_mask = np.where(component_mask > 0, 255, 0).astype(np.uint8)
        raw_floor = np.where(raw_floor_mask > 0, 255, 0).astype(np.uint8)
        if (
            isinstance(visual_card_text_mask, np.ndarray)
            and np.any(visual_card_text_mask)
            and int(np.count_nonzero(component_mask)) > int(np.count_nonzero(visual_card_text_mask) * 1.45)
        ):
            text_mask = visual_card_text_mask
            qa_metrics = block.setdefault("qa_metrics", {})
            if isinstance(qa_metrics, dict):
                qa_metrics["colored_card_visual_glyph_mask_preserved"] = {
                    "component_pixels": int(np.count_nonzero(component_mask)),
                    "visual_pixels": int(np.count_nonzero(visual_card_text_mask)),
                }
        elif _dark_or_colored_text_card_context(block) and int(np.count_nonzero(component_mask)) < int(
            np.count_nonzero(raw_floor)
        ):
            text_mask = cv2.bitwise_or(component_mask, raw_floor)
            qa_metrics = block.setdefault("qa_metrics", {})
            if isinstance(qa_metrics, dict):
                qa_metrics["component_bubble_cleaner_raw_floor_preserved"] = {
                    "component_pixels": int(np.count_nonzero(component_mask)),
                    "raw_floor_pixels": int(np.count_nonzero(raw_floor)),
                }
        else:
            text_mask = component_mask
            raw_floor_mask = component_mask
        used_component_bubble_cleaner = True
        if isinstance(geometry_mask, np.ndarray) and np.any(geometry_mask):
            floored_mask, floor_metrics = _white_balloon_geometry_floor(block, text_mask, geometry_mask, image_shape)
            if floor_metrics is not None:
                text_mask = floored_mask
                qa_metrics = block.setdefault("qa_metrics", {})
                if isinstance(qa_metrics, dict):
                    qa_metrics["white_balloon_geometry_floor_after_component_cleaner"] = floor_metrics

    dark_bbox_mask = _dark_glyph_pixels_in_text_bbox(block, image_rgb, image_shape)
    if isinstance(dark_bbox_mask, np.ndarray) and np.any(dark_bbox_mask):
        should_merge_dark_bbox = (not used_component_bubble_cleaner) or _has_explicit_tight_source_reference(
            block,
            image_shape,
        )
        if should_merge_dark_bbox and not used_component_bubble_cleaner and isinstance(geometry_mask, np.ndarray) and np.any(geometry_mask):
            dark_bbox_mask = _filter_mask_components_by_geometry(
                dark_bbox_mask,
                geometry_mask,
                margin_px=max(6, _koharu_glyph_dilate_radius(block) + 3),
            )
        if should_merge_dark_bbox:
            text_mask = cv2.bitwise_or(text_mask.astype(np.uint8), dark_bbox_mask.astype(np.uint8))
            visual_text_only_source_mask = (
                cv2.bitwise_or(visual_text_only_source_mask.astype(np.uint8), dark_bbox_mask.astype(np.uint8))
                if isinstance(visual_text_only_source_mask, np.ndarray) and np.any(visual_text_only_source_mask)
                else dark_bbox_mask.astype(np.uint8)
            )

    if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask):
        bubble_mask = real_bubble_mask
    elif isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask):
        bubble_mask = derived_bubble_mask
    else:
        bubble_mask = block.get("bubble_mask")
    note_bubble_mask = bubble_mask
    if (
        isinstance(geometry_mask, np.ndarray)
        and isinstance(note_bubble_mask, np.ndarray)
        and note_bubble_mask.shape[:2] != geometry_mask.shape[:2]
    ):
        page_note_bubble_mask, _reason = _page_space_bubble_mask(block, geometry_mask.shape[:2])
        if page_note_bubble_mask is not None and np.any(page_note_bubble_mask):
            note_bubble_mask = page_note_bubble_mask
    if _translator_note_bubble_clip_misses_text(block, geometry_mask, note_bubble_mask):
        metrics = block.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["translator_note_bubble_clip_skipped"] = {
                "geometry_pixels": int(np.count_nonzero(geometry_mask > 0)),
                "bubble_pixels": int(np.count_nonzero(np.asarray(note_bubble_mask) > 0)),
                "overlap_pixels": int(np.count_nonzero((geometry_mask > 0) & (np.asarray(note_bubble_mask) > 0))),
            }
        _append_qa_flag(block, "translator_note_bubble_clip_skipped")
        if isinstance(geometry_mask, np.ndarray) and np.any(geometry_mask):
            text_mask = cv2.bitwise_or(text_mask.astype(np.uint8), geometry_mask.astype(np.uint8))
            raw_floor_mask = cv2.bitwise_or(raw_floor_mask.astype(np.uint8), geometry_mask.astype(np.uint8))
        bubble_mask = None
        real_bubble_mask = None
        derived_bubble_mask = None
    bubble_id = block.get("bubble_id") or block.get("bubbleId")
    if isinstance(bubble_mask, np.ndarray) and bubble_mask.shape[:2] != text_mask.shape[:2]:
        page_bubble_mask, _reason = _page_space_bubble_mask(block, text_mask.shape[:2])
        if page_bubble_mask is not None and np.any(page_bubble_mask):
            bubble_mask = page_bubble_mask
    text_mask = _dark_bubble_recovered_text_bbox_floor(block, text_mask, image_shape, bubble_mask)
    if isinstance(raw_floor_mask, np.ndarray) and np.any(raw_floor_mask):
        raw_floor_mask = _dark_bubble_recovered_text_bbox_floor(block, raw_floor_mask, image_shape, bubble_mask)
    text_mask = clip_mask_to_bubble_id(text_mask, bubble_mask, bubble_id)
    if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask):
        text_mask = clip_text_mask_to_balloon_interior_preserving_raw(
            text_mask,
            real_bubble_mask,
            raw_floor_mask,
            erode_px=2,
            block=block,
            source="real_bubble_mask",
        )
        text_mask = _preserve_colored_card_geometry_when_bubble_clip_is_invalid(
            block,
            text_mask,
            geometry_mask,
            real_bubble_mask,
            source="real_bubble_mask",
            image_rgb=image_rgb,
        )
    elif isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask):
        text_mask = clip_text_mask_to_balloon_interior_preserving_raw(
            text_mask,
            derived_bubble_mask,
            raw_floor_mask,
            erode_px=2,
            block=block,
            source="derived_bubble_mask",
        )
        text_mask = _preserve_colored_card_geometry_when_bubble_clip_is_invalid(
            block,
            text_mask,
            geometry_mask,
            derived_bubble_mask,
            source="derived_bubble_mask",
            image_rgb=image_rgb,
        )

    radius = _koharu_glyph_dilate_radius(block)
    if radius > 0:
        radius = max(2, int(radius) - 1)
        pre_dilate_text_mask = text_mask.astype(np.uint8)
        limit_mask = None
        if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask):
            limit_mask = safe_bubble_interior_mask(real_bubble_mask, erode_px=2)
        elif isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask):
            limit_mask = safe_bubble_interior_mask(derived_bubble_mask, erode_px=2)
        text_mask = expand_text_mask_monotonic(pre_dilate_text_mask, limit_mask=limit_mask, radius=radius)
        text_mask = clip_mask_to_bubble_id(text_mask, bubble_mask, bubble_id)
        if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask):
            text_mask = clip_text_mask_to_balloon_interior_preserving_raw(
                text_mask,
                real_bubble_mask,
                raw_floor_mask,
                erode_px=2,
                block=block,
                source="real_bubble_mask",
            )
            text_mask = _preserve_colored_card_geometry_when_bubble_clip_is_invalid(
                block,
                text_mask,
                geometry_mask,
                real_bubble_mask,
                source="real_bubble_mask",
                image_rgb=image_rgb,
            )
        elif isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask):
            text_mask = clip_text_mask_to_balloon_interior_preserving_raw(
                text_mask,
                derived_bubble_mask,
                raw_floor_mask,
                erode_px=2,
                block=block,
                source="derived_bubble_mask",
            )
            text_mask = _preserve_colored_card_geometry_when_bubble_clip_is_invalid(
                block,
                text_mask,
                geometry_mask,
                derived_bubble_mask,
                source="derived_bubble_mask",
                image_rgb=image_rgb,
            )

    if isinstance(visual_card_text_mask, np.ndarray) and np.any(visual_card_text_mask):
        replacement_mask = visual_card_text_mask.astype(np.uint8)
        if isinstance(bubble_mask, np.ndarray) and np.any(bubble_mask):
            replacement_mask = clip_mask_to_bubble_id(replacement_mask, bubble_mask, bubble_id)
            clip_source = real_bubble_mask if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask) else derived_bubble_mask
            if isinstance(clip_source, np.ndarray) and np.any(clip_source):
                replacement_mask = clip_text_mask_to_balloon_interior_preserving_raw(
                    replacement_mask,
                    clip_source,
                    raw_floor_mask,
                    erode_px=1,
                    block=block,
                    source="visual_card_text_mask",
                )
        if (
            np.any(replacement_mask)
            and int(np.count_nonzero(text_mask)) > int(np.count_nonzero(replacement_mask) * 1.65)
        ):
            text_mask = replacement_mask
    else:
        text_mask = expand_light_halo_mask_for_dark_card(text_mask, image_rgb, block, radius=max(6, radius + 3))
    text_mask = clip_mask_to_bubble_id(text_mask, bubble_mask, bubble_id)
    if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask):
        text_mask = clip_text_mask_to_balloon_interior_preserving_raw(
            text_mask,
            real_bubble_mask,
            raw_floor_mask,
            erode_px=1,
            block=block,
            source="real_bubble_mask",
        )
        text_mask = _preserve_colored_card_geometry_when_bubble_clip_is_invalid(
            block,
            text_mask,
            geometry_mask,
            real_bubble_mask,
            source="real_bubble_mask",
            image_rgb=image_rgb,
        )
    elif isinstance(derived_bubble_mask, np.ndarray) and np.any(derived_bubble_mask):
        text_mask = clip_text_mask_to_balloon_interior_preserving_raw(
            text_mask,
            derived_bubble_mask,
            raw_floor_mask,
            erode_px=1,
            block=block,
            source="derived_bubble_mask",
        )
        text_mask = _preserve_colored_card_geometry_when_bubble_clip_is_invalid(
            block,
            text_mask,
            geometry_mask,
            derived_bubble_mask,
            source="derived_bubble_mask",
            image_rgb=image_rgb,
        )

    white_bubble_context = bool(
        str(block.get("layout_profile") or block.get("block_profile") or "").strip().lower() == "white_balloon"
        or str(block.get("balloon_type") or "").strip().lower() == "white"
        or isinstance(real_bubble_mask, np.ndarray)
        or isinstance(derived_bubble_mask, np.ndarray)
    )
    if (
        white_bubble_context
        and not _dark_or_colored_text_card_context(block)
        and isinstance(geometry_mask, np.ndarray)
        and np.any(geometry_mask)
    ):
        source_support = _source_bbox_support_mask_for_block(block, text_mask.shape[:2])
        geometry_filtered = _filter_white_balloon_components_by_text_anchor(
            text_mask,
            geometry_mask,
            source_support,
            bubble_mask if isinstance(bubble_mask, np.ndarray) and np.any(bubble_mask) else None,
        )
        if np.any(geometry_filtered) and int(np.count_nonzero(geometry_filtered)) < int(np.count_nonzero(text_mask)):
            dropped_pixels = int(np.count_nonzero(text_mask)) - int(np.count_nonzero(geometry_filtered))
            text_mask = geometry_filtered.astype(np.uint8)
            metrics = block.setdefault("qa_metrics", {})
            if isinstance(metrics, dict):
                metrics["white_balloon_unanchored_component_filter"] = {
                    "dropped_pixels": dropped_pixels,
                    "remaining_pixels": int(np.count_nonzero(text_mask)),
                }

    text_mask = text_mask.astype(np.uint8)
    text_mask = _enforce_visual_text_only_inpaint_contract(
        block,
        text_mask,
        visual_text_only_source_mask,
        bubble_mask if isinstance(bubble_mask, np.ndarray) and np.any(bubble_mask) else None,
        radius=radius,
    )
    if text_mask is None:
        return None
    text_mask = text_mask.astype(np.uint8)
    text_mask = _apply_clipped_overlap_fragment_cleanup_mask(block, text_mask, image_shape)
    if _mask_bbox_touches_crop_edge(text_mask, edge_px=2):
        _append_qa_flag(block, "band_edge_clipped_text_mask")
        metrics = block.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["band_edge_clipped_text_mask"] = {
                "bbox": _mask_bbox(text_mask),
                "reason": "text_mask_touches_band_edge",
            }
    if isinstance(bubble_mask, np.ndarray) and np.any(bubble_mask) and _mask_bbox_touches_crop_edge(bubble_mask, edge_px=2):
        _append_qa_flag(block, "band_edge_clipped_balloon_mask")
        metrics = block.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["band_edge_clipped_balloon_mask"] = {
                "bbox": _mask_bbox(bubble_mask),
                "reason": "balloon_mask_touches_band_edge",
            }
    consolidate_mask_evidence(
        block,
        kind=(
            "component_bubble_cleaner"
            if used_component_bubble_cleaner
            else "ocr_pixels" if isinstance(raw_text_mask, np.ndarray) and np.any(raw_text_mask) else "clipped_line_polygon"
        ),
        raw_mask_pixels=int(np.count_nonzero(raw_text_mask)) if isinstance(raw_text_mask, np.ndarray) else int(np.count_nonzero(text_mask)),
        expanded_mask_pixels=int(np.count_nonzero(text_mask)),
        evidence_score=1.0,
        fast_fill_reject_reasons=_dark_text_without_balloon_fast_fill_reject_reasons(block, image_shape),
    )
    return text_mask


def build_mask_regions(texts: list[dict], image_shape: tuple[int, int, int]) -> list[dict]:
    image_height, image_width = image_shape[:2]
    seeds = []

    for text in texts:
        if _merged_fragment_without_glyph_evidence(text):
            continue
        bbox = _mask_region_seed_bbox(text, image_shape)
        if not bbox:
            continue
        balloon_bbox = _normalize_bbox(text.get("balloon_bbox"), image_width, image_height)
        seeds.append(
            {
                "bbox": bbox,
                "balloon_bbox": balloon_bbox,
                "tipo": text.get("tipo", "fala"),
                "text": text,
            }
        )

    clusters: list[dict] = []
    for seed in seeds:
        merged = False
        for cluster in clusters:
            if should_merge_text_blocks(cluster, seed):
                cluster["bbox"] = union_bbox(cluster["bbox"], seed["bbox"])
                cluster["balloon_bbox"] = _union_optional_bbox(cluster.get("balloon_bbox"), seed.get("balloon_bbox"))
                cluster["texts"].append(seed["text"])
                cluster["tipos"].append(seed["tipo"])
                merged = True
                break
        if not merged:
            clusters.append(
                {
                    "bbox": seed["bbox"],
                    "balloon_bbox": seed.get("balloon_bbox"),
                    "texts": [seed["text"]],
                    "tipos": [seed["tipo"]],
                }
            )

    # Second pass: merge clusters that now overlap after growth.
    # The greedy single-pass above is order-dependent — a block arriving
    # before its eventual neighbours can start its own cluster that never
    # gets re-checked.  This convergence loop fixes that.
    changed = True
    while changed:
        changed = False
        new_clusters: list[dict] = []
        skip: set[int] = set()
        for i in range(len(clusters)):
            if i in skip:
                continue
            current = clusters[i]
            for j in range(i + 1, len(clusters)):
                if j in skip:
                    continue
                if should_merge_text_blocks(current, clusters[j]):
                    current["bbox"] = union_bbox(current["bbox"], clusters[j]["bbox"])
                    current["balloon_bbox"] = _union_optional_bbox(current.get("balloon_bbox"), clusters[j].get("balloon_bbox"))
                    current["texts"].extend(clusters[j]["texts"])
                    current["tipos"].extend(clusters[j]["tipos"])
                    skip.add(j)
                    changed = True
            new_clusters.append(current)
        clusters = new_clusters

    regions = []
    for cluster in clusters:
        regions.append(
            {
                "bbox": cluster["bbox"],
                "balloon_bbox": cluster.get("balloon_bbox"),
                "texts": cluster["texts"],
                "tipo": Counter(cluster["tipos"]).most_common(1)[0][0],
                "kind": "cluster" if len(cluster["texts"]) > 1 else "single",
            }
        )
    return regions


def _mask_region_seed_bbox(text: dict, image_shape: tuple[int, ...]) -> list[int] | None:
    image_height, image_width = image_shape[:2]
    geometry_bbox = _text_geometry_bbox(text, image_shape)
    if geometry_bbox is not None:
        if _dark_connected_bubble_broad_bbox_risk(text):
            text_bbox = _normalize_bbox(text.get("text_pixel_bbox"), image_width, image_height)
            if text_bbox is not None:
                geometry_area = max(1, _bbox_area(geometry_bbox))
                text_area = max(1, _bbox_area(text_bbox))
                geometry_w = max(1, geometry_bbox[2] - geometry_bbox[0])
                text_w = max(1, text_bbox[2] - text_bbox[0])
                overlap = _bbox_intersection_area(geometry_bbox, text_bbox)
                undercovers_text = bool(
                    overlap >= int(geometry_area * 0.35)
                    and (
                        geometry_area < int(text_area * 0.72)
                        or geometry_w < int(text_w * 0.72)
                    )
                )
                if undercovers_text:
                    metrics = text.setdefault("qa_metrics", {})
                    if isinstance(metrics, dict):
                        metrics["dark_connected_mask_region_seed_bbox_replaced"] = {
                            "geometry_bbox": list(geometry_bbox),
                            "text_pixel_bbox": list(text_bbox),
                            "geometry_area": int(geometry_area),
                            "text_area": int(text_area),
                        }
                    _append_qa_flag(text, "dark_connected_mask_region_seed_bbox_replaced")
                    return _expand_bbox_px(text_bbox, image_width, image_height, pad_x=4, pad_y=4) or text_bbox
        return _expand_bbox_px(geometry_bbox, image_width, image_height, pad_x=4, pad_y=4) or geometry_bbox
    bbox = _normalize_bbox(text.get("bbox"), image_width, image_height)
    if bbox is None:
        return None
    return _expand_bbox_px(bbox, image_width, image_height, pad_x=4, pad_y=4) or bbox


def _union_optional_bbox(a: list[int] | None, b: list[int] | None) -> list[int] | None:
    if a and b:
        return union_bbox(a, b)
    return a or b


def _cluster_has_special_class(item: dict) -> bool:
    texts = item.get("texts")
    if isinstance(texts, list):
        return any(isinstance(text, dict) and _is_special_mask_class(text) for text in texts)
    text = item.get("text")
    if isinstance(text, dict):
        return _is_special_mask_class(text)
    return _is_special_mask_class(item)


def should_merge_text_blocks(a: dict, b: dict) -> bool:
    if _cluster_has_special_class(a) or _cluster_has_special_class(b):
        return False
    return should_merge(a["bbox"], b["bbox"])


def _is_vertical_text(bbox: list[int]) -> bool:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return h > w * 2.5


def expand_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    tipo: str = "fala",
    confidence: float = 1.0,
    estilo: dict | None = None,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    if _is_vertical_text(bbox):
        margin_x = max(2, int(width * 0.08))
        margin_y = max(2, int(height * 0.06))
    else:
        margin_x = max(4, int(width * 0.10))
        margin_y = max(4, int(height * 0.12))

    if confidence < 0.65:
        margin_x += 2
        margin_y += 2

    if estilo:
        contorno_px = int(estilo.get("contorno_px", 0))
        glow_px = int(estilo.get("glow_px", 0))
        sombra = estilo.get("sombra_offset", [0, 0])
        sombra_w = max(0, abs(int(sombra[0])))
        sombra_h = max(0, abs(int(sombra[1])))
        
        extra_x = contorno_px + glow_px + sombra_w
        extra_y = contorno_px + glow_px + sombra_h
        
        margin_x += min(18, extra_x)
        margin_y += min(18, extra_y)

    return [
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(image_width, x2 + margin_x),
        min(image_height, y2 + margin_y),
    ]


def should_merge(a: list[int], b: list[int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    horizontal_gap = max(0, max(ax1, bx1) - min(ax2, bx2), max(bx1, ax1) - min(bx2, ax2))
    vertical_gap = max(0, max(ay1, by1) - min(ay2, by2), max(by1, ay1) - min(by2, ay2))
    width = min(ax2 - ax1, bx2 - bx1)
    height = min(ay2 - ay1, by2 - by1)
    overlaps = not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    # Se os bboxes se sobrepõem, merge (estão no mesmo balão)
    if overlaps:
        return True

    # Gap moderado: próximos o bastante para ser o mesmo balão,
    # mas não tão largo que junte balões separados
    return horizontal_gap <= max(8, int(width * 0.15)) and vertical_gap <= max(12, int(height * 0.25))


def union_bbox(a: list[int], b: list[int]) -> list[int]:
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def build_region_pixel_mask(image_shape: tuple[int, int], region: dict) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for text in region.get("texts", []):
        if not isinstance(text, dict):
            continue
        text_mask = _region_text_pixel_mask(text, width, height)
        if text_mask is not None and np.any(text_mask):
            mask = np.maximum(mask, text_mask)

    x1, y1, x2, y2 = region["bbox"]
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def _region_text_pixel_mask(text: dict, width: int, height: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = _text_geometry_polygons(text, width, height)
    if polygons:
        for polygon in polygons:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    else:
        bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        if bbox is None:
            bbox = _normalize_bbox(text.get("bbox"), width, height)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255

    if not np.any(mask):
        return None
    pad = glyph_padding(_font_size_from_block(text))
    if pad > 0:
        kernel_size = max(3, pad * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    mask = _clip_region_mask_to_real_bubble(mask, text)
    return mask if np.any(mask) else None


def _clip_region_mask_to_real_bubble(text_mask: np.ndarray, text: dict) -> np.ndarray:
    bubble_mask = text.get("bubble_mask")
    if not isinstance(bubble_mask, np.ndarray):
        return text_mask
    if bubble_mask.ndim == 3:
        bubble_mask = bubble_mask[:, :, 0]
    if bubble_mask.shape[:2] != text_mask.shape[:2]:
        return text_mask
    numeric_id = _numeric_bubble_id(text.get("bubble_id") or text.get("bubbleId"))
    allowed = (bubble_mask == numeric_id) if numeric_id is not None else (bubble_mask > 0)
    return np.where((text_mask > 0) & allowed, 255, 0).astype(np.uint8)
