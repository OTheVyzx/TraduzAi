from __future__ import annotations

from collections import Counter
from typing import Any

import cv2
import numpy as np

MASK_DENSITY_WARN = 0.12
MASK_DENSITY_BORDERLINE_WARN = 0.15
MASK_DENSITY_STRONG_WARN = 0.30
EXPANDED_RAW_WARN = 2.5
SOURCE_GLYPH_REVIEW = 1.5
SOURCE_GLYPH_CRITICAL = 8.0
OUTSIDE_BALLOON_WARN_RATIO = 0.08
CLEAN_OUTSIDE_BALLOON_RATIO = 0.01
CLEAN_EXPANDED_RAW_RATIO = 1.5
CLEAN_SOURCE_GLYPH_RATIO = 1.2
OUTSIDE_BALLOON_CRITICAL_PIXELS = 50
OUTSIDE_BALLOON_CRITICAL_RATIO = 0.18
DARK_PANEL_RECT_MAX_HALF_WIDTH_FROM_TEXT_CENTER = 116
DARK_PANEL_RECT_MAX_HALF_HEIGHT_FROM_TEXT_CENTER = 64

_MASK_SUMMARY_STATE: dict[int, dict[str, dict[str, Any]]] = {}


def _image_hw(image: np.ndarray) -> tuple[int, int]:
    return int(image.shape[0]), int(image.shape[1])


def _blank_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _as_mask(mask: np.ndarray | None, height: int, width: int) -> np.ndarray:
    if mask is None:
        return _blank_mask(height, width)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape[:2] != (height, width):
        arr = cv2.resize(arr.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _normalize_bbox(value: Any, width: int, height: int) -> list[int] | None:
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


def _normalize_polygon(value: Any, width: int, height: int) -> list[list[int]] | None:
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


def _normalize_polygons(value: Any, width: int, height: int) -> list[list[list[int]]]:
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


def _fill_polygons(mask: np.ndarray, polygons: list[list[list[int]]]) -> None:
    for polygon in polygons:
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)


def _mask_from_bbox(width: int, height: int, bbox: list[int] | None) -> np.ndarray:
    mask = _blank_mask(height, width)
    if bbox:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
    return mask


def _mask_to_canvas_for_bbox(
    value: Any,
    bbox: list[int] | None,
    width: int,
    height: int,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return None
    if arr.shape[:2] == (height, width):
        mask = np.where(arr > 0, 255, 0).astype(np.uint8)
        return mask if np.any(mask) else None
    normalized = _normalize_bbox(bbox, width, height)
    if normalized is None:
        return None
    x1, y1, x2, y2 = normalized
    target_w = x2 - x1
    target_h = y2 - y1
    if target_w <= 0 or target_h <= 0:
        return None
    patch = arr.astype(np.uint8)
    if patch.shape[:2] != (target_h, target_w):
        patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    canvas = _blank_mask(height, width)
    canvas[y1:y2, x1:x2] = np.where(patch[:target_h, :target_w] > 0, 255, 0).astype(np.uint8)
    return canvas if np.any(canvas) else None


_DERIVED_BUBBLE_MASK_SOURCES = {
    "derived_white_crop",
    "derived_rectangular_balloon",
    "derived_card_panel_mask",
    "image_white_region",
    "outline_seeded_contour",
    "text_rect_fallback",
}

_IMAGE_BUBBLE_MASK_SOURCES = {
    "image_white_bubble_mask",
    "image_rect_bubble_mask",
    "image_contour_bubble_mask",
    "image_dark_panel_mask",
    "image_dark_bubble_mask",
}

_REAL_BUBBLE_MASK_SOURCES = {
    "real",
    "real_bubble_mask",
}

_DERIVED_BUBBLE_REJECTION_FLAGS = {
    "debug_derived_bubble_mask_rejected",
    "derived_bubble_mask_rejected",
    "rejected_derived_bubble_mask",
}


def _rgb_luma_chroma(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        r, g, b = [float(v) for v in value[:3]]
    except Exception:
        return None
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    chroma = max(r, g, b) - min(r, g, b)
    return luma, chroma


def _style_has_glow(text: dict[str, Any]) -> bool:
    for key in ("style", "estilo"):
        style = text.get(key)
        if isinstance(style, dict) and bool(style.get("glow")):
            return True
    return False


def _text_card_panel_context(text: dict[str, Any]) -> bool:
    flags = {str(flag).strip() for flag in (text.get("qa_flags") or []) if str(flag).strip()}
    if any(flag.startswith("translator_note") for flag in flags):
        return True
    text_value = str(
        text.get("text")
        or text.get("original")
        or text.get("raw_ocr")
        or text.get("normalized_ocr")
        or ""
    ).strip().upper()
    if text_value.startswith(("T/N:", "TL/N:", "TN:", "TL:")):
        return True
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source in {"derived_card_panel_mask", "image_dark_panel_mask", "image_dark_bubble_mask"}:
        return True
    profiles = {
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("render_profile") or "").strip().lower(),
    }
    if profiles & {"dark_panel", "colored_status_panel", "status_panel", "card", "title_card"}:
        return True
    background = _rgb_luma_chroma(text.get("background_rgb"))
    if background is not None:
        luma, chroma = background
        if luma < 135.0 or chroma > 45.0:
            return True
    return _style_has_glow(text)


def _texts_card_panel_context(texts: list[dict[str, Any]]) -> bool:
    return bool(texts) and all(isinstance(text, dict) and _text_card_panel_context(text) for text in texts)


def _text_has_rejected_bubble_without_glyph_evidence(text: dict[str, Any]) -> bool:
    if not isinstance(text, dict):
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    error = str(text.get("bubble_mask_error") or text.get("bubbleMaskError") or "").strip()
    flags = {str(flag).strip() for flag in (text.get("qa_flags") or []) if str(flag).strip()}
    rejected = (
        source in {"derived_white_crop_rejected", "rejected_derived_bubble_mask"}
        or bool(error)
        or bool(flags & _DERIVED_BUBBLE_REJECTION_FLAGS)
    )
    if not rejected:
        return False
    missing_glyph_evidence = (
        "raw_text_evidence_missing" in flags
        or "fast_fill_no_glyph_evidence" in flags
        or bool(text.get("raw_text_evidence_missing"))
        or not bool(text.get("line_polygons"))
    )
    merged_fragment = "same_balloon_fragment_merged" in flags or bool(text.get("merged_into_trace_id"))
    has_line_geometry = bool(text.get("line_polygons"))
    return bool(missing_glyph_evidence and (merged_fragment or not has_line_geometry))


def _derived_bubble_mask_rejection_reason(text: dict[str, Any]) -> str | None:
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    error = str(text.get("bubble_mask_error") or text.get("bubbleMaskError") or "").strip()
    if source == "derived_card_panel_mask":
        return None
    if error:
        return error
    flags = {str(flag).strip() for flag in (text.get("qa_flags") or []) if str(flag).strip()}
    for flag in sorted(_DERIVED_BUBBLE_REJECTION_FLAGS):
        if flag in flags:
            return flag
    if source not in _DERIVED_BUBBLE_MASK_SOURCES and source != "derived_white_crop_rejected":
        return None
    if source == "derived_white_crop_rejected":
        return "derived_white_crop_rejected"
    return None


def _bubble_mask_source_kind(text: dict[str, Any]) -> str:
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if _derived_bubble_mask_rejection_reason(text):
        return "derived_white_crop_rejected"
    if source in _REAL_BUBBLE_MASK_SOURCES:
        return "real_bubble_mask"
    if source in _IMAGE_BUBBLE_MASK_SOURCES:
        return source
    if source in _DERIVED_BUBBLE_MASK_SOURCES:
        return source
    if source in {"bbox_fallback", "derived_white_crop_rejected"}:
        return source
    return "missing"


def _dark_bubble_should_debug_as_rect_panel(text: dict[str, Any]) -> bool:
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source != "image_dark_bubble_mask":
        return False
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if bool(text.get("card_panel_text_context")):
        return True
    profiles = {
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("render_profile") or "").strip().lower(),
    }
    if profiles & {"dark_panel", "colored_status_panel", "status_panel", "card", "title_card"}:
        return True
    if "connected_layout_disabled_dark_panel_visual_mask" in flags:
        return True
    metrics = text.get("qa_metrics") if isinstance(text.get("qa_metrics"), dict) else {}
    rejected = metrics.get("image_dark_panel_mask_rejected") if isinstance(metrics, dict) else None
    if isinstance(rejected, list):
        if any(
            isinstance(item, dict)
            and str(item.get("reason") or "").strip().lower() == "overbroad_against_balloon_bbox"
            for item in rejected
        ):
            return True
    if (
        "dark_bubble_ellipse_bbox_mask" in flags
        or "dark_bubble_visual_bbox_refined" in flags
        or isinstance(text.get("bubble_mask_ellipse") or text.get("bubbleMaskEllipse"), dict)
    ):
        return False
    return False


def _centered_capped_dark_panel_bbox(
    text: dict[str, Any],
    *,
    width: int,
    height: int,
    bbox: list[int] | None,
) -> list[int] | None:
    panel_bbox = _normalize_bbox(bbox, width, height)
    anchor_bbox = (
        _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(text.get("source_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
    )
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


def _real_bubble_mask_from_text(text: dict[str, Any], width: int, height: int) -> tuple[np.ndarray | None, str | None]:
    if _derived_bubble_mask_rejection_reason(text):
        return None, "derived_white_crop_rejected"
    source_kind = _bubble_mask_source_kind(text)
    qa_flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    recovered_dark_bubble = source_kind == "image_dark_bubble_mask" and bool(
        {
            "partial_dark_bubble_lobe_reocr",
            "detected_dark_bubble_without_text_reocr",
            "candidate_crop_direct_paddle_reocr",
        }
        & qa_flags
    )
    if recovered_dark_bubble:
        bbox = (
            _normalize_bbox(text.get("balloon_bbox"), width, height)
            or _normalize_bbox(text.get("bubble_mask_bbox"), width, height)
            or _normalize_bbox(text.get("bbox"), width, height)
        )
    else:
        bbox = (
            _normalize_bbox(text.get("bubble_mask_bbox"), width, height)
            or _normalize_bbox(text.get("balloon_bbox"), width, height)
            or _normalize_bbox(text.get("bbox"), width, height)
        )
    if _dark_bubble_should_debug_as_rect_panel(text):
        bbox = _centered_capped_dark_panel_bbox(text, width=width, height=height, bbox=bbox)
        bbox_mask = _mask_from_bbox(width, height, bbox)
        if np.any(bbox_mask):
            return bbox_mask, "image_dark_panel_mask"
    ellipse = text.get("bubble_mask_ellipse") or text.get("bubbleMaskEllipse")
    if source_kind == "image_dark_bubble_mask" and isinstance(ellipse, dict):
        try:
            center = ellipse.get("center") or []
            axes = ellipse.get("axes") or []
            angle = float(ellipse.get("angle") or 0.0)
            cx, cy = float(center[0]), float(center[1])
            axis_a, axis_b = float(axes[0]), float(axes[1])
        except Exception:
            cx = cy = axis_a = axis_b = 0.0
        if axis_a > 0 and axis_b > 0:
            mask = _blank_mask(height, width)
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
            if np.any(mask):
                return mask, source_kind
    if source_kind == "image_dark_bubble_mask" and bbox is not None:
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            mask = _blank_mask(height, width)
            cv2.ellipse(
                mask,
                (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
                (max(1, int(round((x2 - x1) / 2.0))), max(1, int(round((y2 - y1) / 2.0)))),
                0.0,
                0,
                360,
                255,
                -1,
            )
            if np.any(mask):
                return mask, source_kind
    if not recovered_dark_bubble:
        for key in ("bubble_mask", "bubbleMask", "balloon_mask", "balloonMask", "segmentation_mask"):
            mask = _mask_to_canvas_for_bbox(text.get(key), bbox, width, height)
            if mask is not None and np.any(mask):
                return mask, _bubble_mask_source_kind(text)
    if _bubble_mask_source_kind(text) == "derived_card_panel_mask":
        bbox_mask = _mask_from_bbox(width, height, bbox)
        if np.any(bbox_mask):
            return bbox_mask, "derived_card_panel_mask"
    if _bubble_mask_source_kind(text) == "image_dark_panel_mask":
        bbox_mask = _mask_from_bbox(width, height, bbox)
        if np.any(bbox_mask):
            return bbox_mask, "image_dark_panel_mask"
    return None, None


def _mask_from_polygons(width: int, height: int, polygons: list[list[list[int]]]) -> np.ndarray:
    mask = _blank_mask(height, width)
    _fill_polygons(mask, polygons)
    return mask


def _balloon_mask_from_texts(texts: list[dict[str, Any]], width: int, height: int) -> tuple[np.ndarray, dict[str, Any]]:
    mask = _blank_mask(height, width)
    used_real = False
    used_image = False
    used_derived = False
    source_names: set[str] = set()
    rejection_reasons: set[str] = set()
    used_polygon = False
    used_bbox = False
    for text in texts:
        rejection_reason = _derived_bubble_mask_rejection_reason(text)
        if rejection_reason:
            rejection_reasons.add(rejection_reason)
            continue
        real_mask, source = _real_bubble_mask_from_text(text, width, height)
        if real_mask is not None and np.any(real_mask):
            if source == "missing":
                continue
            mask |= real_mask
            source_names.add(source or "real_bubble_mask")
            if source in _DERIVED_BUBBLE_MASK_SOURCES:
                used_derived = True
            elif source in _IMAGE_BUBBLE_MASK_SOURCES:
                used_image = True
            else:
                used_real = True
            continue
        polygons = _normalize_polygons(text.get("balloon_polygon"), width, height)
        if polygons:
            _fill_polygons(mask, polygons)
            used_polygon = True
            continue
        if _bubble_mask_source_kind(text) == "derived_white_crop_rejected":
            continue
        bbox_mask = _mask_from_bbox(width, height, _normalize_bbox(text.get("balloon_bbox"), width, height))
        if np.any(bbox_mask):
            mask |= bbox_mask
            used_bbox = True
    return mask, {
        "used_real_bubble_mask": used_real,
        "used_image_bubble_mask": used_image,
        "used_derived_bubble_mask": used_derived,
        "used_balloon_polygon": used_polygon,
        "used_balloon_bbox_fallback": used_bbox and not used_real and not used_polygon,
        "bubble_mask_source": (
            sorted(source_names)[0]
            if source_names
            else "balloon_polygon"
            if used_polygon
            else "balloon_bbox_fallback"
            if used_bbox
            else "rejected_derived_bubble_mask"
            if rejection_reasons
            else "missing"
        ),
        "bubble_mask_rejection_reason": sorted(rejection_reasons)[0] if rejection_reasons else None,
    }


def _union_mask_from_texts(texts: list[dict[str, Any]], width: int, height: int, kind: str) -> np.ndarray:
    mask = _blank_mask(height, width)
    for text in texts:
        if kind in {"glyph", "line"}:
            _fill_polygons(mask, _normalize_polygons(text.get("line_polygons"), width, height))
        elif kind == "detected":
            bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height) or _normalize_bbox(text.get("bbox"), width, height)
            mask |= _mask_from_bbox(width, height, bbox)
        elif kind == "balloon":
            mask |= _balloon_mask_from_texts([text], width, height)[0]
    return mask


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_area(bbox: list[int] | None) -> int:
    if not bbox:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _bbox_intersection_area(first: list[int] | None, second: list[int] | None) -> int:
    if not first or not second:
        return 0
    x1 = max(int(first[0]), int(second[0]))
    y1 = max(int(first[1]), int(second[1]))
    x2 = min(int(first[2]), int(second[2]))
    y2 = min(int(first[3]), int(second[3]))
    return max(0, x2 - x1) * max(0, y2 - y1)


def _source_bbox_area(texts: list[dict[str, Any]], width: int, height: int) -> int:
    mask = _blank_mask(height, width)
    for text in texts:
        mask |= _mask_from_bbox(width, height, _normalize_bbox(text.get("bbox"), width, height))
    return int(np.count_nonzero(mask))


def _bboxes_nearly_equal(first: list[int] | None, second: list[int] | None, tolerance: int = 2) -> bool:
    if first is None or second is None:
        return False
    return all(abs(int(a) - int(b)) <= tolerance for a, b in zip(first, second))


def _has_explicit_balloon_shape(text: dict[str, Any], width: int, height: int) -> bool:
    if _normalize_polygons(text.get("balloon_polygon"), width, height):
        return True
    for key in (
        "balloon_subregions",
        "connected_lobe_bboxes",
        "connected_lobe_polygons",
        "connected_position_bboxes",
    ):
        raw = text.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            return True
    return False


def _reference_bboxes_for_text(text: dict[str, Any], width: int, height: int) -> list[list[int]]:
    refs: list[list[int]] = []
    for key in ("bbox", "source_bbox", "text_pixel_bbox", "ocr_text_bbox"):
        bbox = _normalize_bbox(text.get(key), width, height)
        if bbox and bbox not in refs:
            refs.append(bbox)
    return refs


def _weak_image_bubble_mask_text_ids(texts: list[dict[str, Any]], width: int, height: int) -> list[str]:
    weak_ids: list[str] = []
    for index, text in enumerate(texts, start=1):
        source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
        if source not in {"image_rect_bubble_mask", "image_white_bubble_mask"}:
            continue
        if _text_card_panel_context(text):
            continue
        text_id = str(text.get("id") or text.get("text_id") or f"text_{index:03d}")
        if source == "image_rect_bubble_mask":
            weak_ids.append(text_id)
            continue
        bubble_bbox = _normalize_bbox(text.get("bubble_mask_bbox") or text.get("balloon_bbox"), width, height)
        text_bbox = (
            _normalize_bbox(text.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(text.get("bbox"), width, height)
        )
        if not bubble_bbox or not text_bbox:
            continue
        bubble_area = max(1, _bbox_area(bubble_bbox))
        text_area = max(1, _bbox_area(text_bbox))
        overlap = _bbox_intersection_area(bubble_bbox, text_bbox)
        if bubble_area <= int(text_area * 1.55) and overlap / float(text_area) >= 0.72:
            weak_ids.append(text_id)
    return weak_ids


def _uses_synthetic_tight_balloon_reference(
    texts: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    mask_source: str,
    used_balloon_clip: bool,
    source_glyph_area_ratio: float,
    mask_balloon_ratio: float,
) -> bool:
    if not used_balloon_clip or mask_source != "line_polygons" or not texts:
        return False
    if any(_has_explicit_balloon_shape(text, width, height) for text in texts):
        return False

    checked = 0
    for text in texts:
        balloon = _normalize_bbox(text.get("balloon_bbox"), width, height)
        if not balloon:
            continue
        refs = _reference_bboxes_for_text(text, width, height)
        if not refs:
            return False
        checked += 1
        if any(_bboxes_nearly_equal(balloon, ref) for ref in refs):
            continue

        balloon_area = _bbox_area(balloon)
        ref_area = max(_bbox_area(ref) for ref in refs)
        if (
            ref_area > 0
            and balloon_area <= int(ref_area * 1.35)
            and source_glyph_area_ratio <= 1.40
            and mask_balloon_ratio <= 1.65
        ):
            continue
        return False
    return checked > 0


def _uses_edge_clipped_text_bbox_reference(
    texts: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    mask_source: str,
    used_balloon_clip: bool,
    outside_balloon_ratio: float,
    mask_balloon_ratio: float,
    expanded_raw_ratio: float,
    has_bbox_overreach: bool,
) -> bool:
    if not used_balloon_clip or mask_source not in {"text_pixel_bbox", "bbox"} or not texts:
        return False
    if has_bbox_overreach:
        return False
    if outside_balloon_ratio > 0.24 or mask_balloon_ratio > 1.35 or expanded_raw_ratio > 1.35:
        return False
    if any(_has_explicit_balloon_shape(text, width, height) for text in texts):
        return False

    checked = 0
    for text in texts:
        balloon = _normalize_bbox(text.get("balloon_bbox"), width, height)
        if not balloon:
            return False
        refs = _reference_bboxes_for_text(text, width, height)
        if not refs:
            return False
        checked += 1
        if any(_bboxes_nearly_equal(balloon, ref) for ref in refs):
            continue

        balloon_area = _bbox_area(balloon)
        ref_area = max(_bbox_area(ref) for ref in refs)
        if balloon_area <= 0 or ref_area <= 0:
            return False
        area_ratio = max(balloon_area, ref_area) / float(max(1, min(balloon_area, ref_area)))
        if area_ratio <= 1.35:
            continue
        return False
    return checked > 0


def _text_ids(texts: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for index, text in enumerate(texts, start=1):
        raw_id = text.get("id") or text.get("text_id") or text.get("_id")
        ids.append(str(raw_id or f"text_{index:03d}"))
    return ids


def _trace_ids(texts: list[dict[str, Any]], text_ids: list[str], band_id: str) -> list[str]:
    ids: list[str] = []
    for index, text in enumerate(texts):
        text_id = text_ids[index] if index < len(text_ids) else str(text.get("id") or text.get("text_id") or "")
        raw_id = text.get("trace_id") or (f"{text_id}@{band_id}" if text_id and band_id else None)
        trace_id = str(raw_id or "").strip()
        if trace_id and trace_id not in ids:
            ids.append(trace_id)
    return ids


def _text_instance_ids(texts: list[dict[str, Any]], text_ids: list[str], band_id: str) -> list[str]:
    ids: list[str] = []
    for index, text in enumerate(texts):
        text_id = text_ids[index] if index < len(text_ids) else str(text.get("id") or text.get("text_id") or "")
        raw_id = (
            text.get("text_instance_id")
            or text.get("instance_id")
            or (f"{band_id}_{text_id}" if text_id and band_id else None)
        )
        instance_id = str(raw_id or "").strip()
        if instance_id and instance_id not in ids:
            ids.append(instance_id)
    return ids


def _band_id(ocr_page: dict[str, Any]) -> str:
    for text in [item for item in ocr_page.get("texts", []) if isinstance(item, dict)]:
        for key in ("band_id", "_band_id"):
            raw_band_id = str(text.get(key) or "").strip()
            if raw_band_id:
                return raw_band_id
        trace_id = str(text.get("trace_id") or "").strip()
        if "@" in trace_id:
            trace_band_id = trace_id.rsplit("@", 1)[-1].strip()
            if trace_band_id:
                return trace_band_id
    for key in ("_band_id", "band_id"):
        raw_band_id = str(ocr_page.get(key) or "").strip()
        if raw_band_id:
            return raw_band_id
    try:
        page_number = int(ocr_page.get("_source_page_number") or ocr_page.get("numero") or 0)
    except Exception:
        page_number = 0
    try:
        band_index = int(ocr_page.get("_band_index") or 0)
    except Exception:
        band_index = 0
    return f"page_{page_number:03d}_band_{band_index:03d}"


def _mask_source(texts: list[dict[str, Any]]) -> str:
    if any(_normalize_polygons(text.get("line_polygons"), 1_000_000, 1_000_000) for text in texts):
        return "line_polygons"
    if any(text.get("text_pixel_bbox") for text in texts):
        return "text_pixel_bbox"
    if any(text.get("bbox") for text in texts):
        return "bbox"
    return "fallback"


def _text_profile_values(texts: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for text in texts:
        for key in ("content_class", "tipo", "layout_profile", "block_profile", "balloon_type"):
            raw = str(text.get(key) or "").strip().lower()
            if raw:
                values.add(raw)
    return values


def _clean_line_polygon_mask(
    *,
    mask_source: str,
    outside_balloon_ratio: float,
    expanded_raw_ratio: float,
) -> bool:
    return (
        mask_source == "line_polygons"
        and outside_balloon_ratio <= CLEAN_OUTSIDE_BALLOON_RATIO
        and expanded_raw_ratio <= CLEAN_EXPANDED_RAW_RATIO
    )


def _mask_density_high_gate(
    texts: list[dict[str, Any]],
    *,
    mask_source: str,
    mask_density: float,
    outside_balloon_ratio: float,
    expanded_raw_ratio: float,
    source_glyph_area_ratio: float,
    has_bbox_overreach: bool,
    synthetic_tight_balloon_reference: bool = False,
) -> bool:
    if mask_density <= MASK_DENSITY_WARN:
        return False
    if (
        synthetic_tight_balloon_reference
        and mask_source == "line_polygons"
        and source_glyph_area_ratio <= CLEAN_SOURCE_GLYPH_RATIO
        and expanded_raw_ratio <= CLEAN_EXPANDED_RAW_RATIO
    ):
        return False
    if outside_balloon_ratio >= OUTSIDE_BALLOON_WARN_RATIO:
        return True
    if expanded_raw_ratio > EXPANDED_RAW_WARN or source_glyph_area_ratio >= SOURCE_GLYPH_CRITICAL:
        return True
    if has_bbox_overreach:
        return True
    if mask_source != "line_polygons":
        return True

    profiles = _text_profile_values(texts)
    clean_line_mask = _clean_line_polygon_mask(
        mask_source=mask_source,
        outside_balloon_ratio=outside_balloon_ratio,
        expanded_raw_ratio=expanded_raw_ratio,
    )
    if clean_line_mask and profiles & {"narracao", "narration", "top_narration", "dark", "textured"}:
        if source_glyph_area_ratio <= CLEAN_SOURCE_GLYPH_RATIO:
            return False
    if clean_line_mask and mask_density < MASK_DENSITY_BORDERLINE_WARN and source_glyph_area_ratio <= CLEAN_SOURCE_GLYPH_RATIO:
        return False
    if mask_density >= MASK_DENSITY_STRONG_WARN and source_glyph_area_ratio >= SOURCE_GLYPH_REVIEW:
        return True
    return False


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.astype(np.uint8).copy()
    if overlay.ndim != 3 or overlay.shape[2] < 3:
        overlay = np.repeat(_as_mask(mask, *_image_hw(image_rgb))[:, :, None], 3, axis=2)
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    active = mask > 0
    overlay[active] = (overlay[active].astype(np.float32) * 0.6 + red[active].astype(np.float32) * 0.4).astype(np.uint8)
    return overlay


def build_mask_chain_debug_payload(
    ocr_page: dict[str, Any],
    *,
    image_rgb: np.ndarray,
    raw_mask: np.ndarray | None,
    expanded_mask: np.ndarray | None,
    final_mask: np.ndarray | None = None,
    protection_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    height, width = _image_hw(image_rgb)
    texts = [
        text
        for text in ocr_page.get("texts", [])
        if isinstance(text, dict) and not _text_has_rejected_bubble_without_glyph_evidence(text)
    ]
    glyph_mask = _union_mask_from_texts(texts, width, height, "glyph")
    line_polygon_mask = _union_mask_from_texts(texts, width, height, "line")
    detected_text_mask = _union_mask_from_texts(texts, width, height, "detected")
    balloon_mask, balloon_metadata = _balloon_mask_from_texts(texts, width, height)
    if np.any(balloon_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        balloon_inner_mask = cv2.erode(balloon_mask, kernel, iterations=1)
    else:
        balloon_inner_mask = _blank_mask(height, width)
    raw = _as_mask(raw_mask, height, width)
    expanded = _as_mask(expanded_mask, height, width)
    protection = _as_mask(protection_mask, height, width)
    final = _as_mask(final_mask, height, width) if final_mask is not None else expanded.copy()
    if np.any(raw):
        raw_preserve = raw
        if np.any(balloon_inner_mask):
            raw_preserve = np.where((raw > 0) & (balloon_inner_mask > 0), 255, 0).astype(np.uint8)
        elif np.any(balloon_mask):
            raw_preserve = np.where((raw > 0) & (balloon_mask > 0), 255, 0).astype(np.uint8)
        if np.any(raw_preserve):
            expanded = cv2.bitwise_or(expanded.astype(np.uint8), raw_preserve)
            final = cv2.bitwise_or(final.astype(np.uint8), raw_preserve)
    if np.any(protection):
        final = np.where((final > 0) & (protection == 0), 255, 0).astype(np.uint8)

    outside_balloon_pixels = 0
    outside_reference = final if final_mask is not None or np.any(protection) else expanded
    used_balloon_clip = bool(np.any(balloon_mask))
    if used_balloon_clip:
        outside_balloon_pixels = int(np.count_nonzero((outside_reference > 0) & (balloon_mask == 0)))

    raw_pixels = int(np.count_nonzero(raw))
    expanded_pixels = int(np.count_nonzero(expanded))
    final_pixels = int(np.count_nonzero(final))
    balloon_pixels = int(np.count_nonzero(balloon_mask))
    balloon_outline_pixels = int(
        np.count_nonzero((final > 0) & (balloon_mask > 0) & (balloon_inner_mask == 0))
    )
    outside_reference_pixels = int(np.count_nonzero(outside_reference))
    outside_balloon_ratio = round(outside_balloon_pixels / float(max(1, outside_reference_pixels)), 6)
    band_pixels = max(1, int(height * width))
    mask_density = round(expanded_pixels / float(band_pixels), 6)
    mask_balloon_ratio = round(final_pixels / float(max(1, balloon_pixels)), 6)
    expanded_raw_ratio = round(expanded_pixels / float(max(1, raw_pixels)), 6)
    source_area = _source_bbox_area(texts, width, height)
    glyph_bbox_area = _bbox_area(_bbox_from_mask(glyph_mask))
    source_glyph_area_ratio = round(source_area / float(max(1, glyph_bbox_area)), 6)
    mask_source = _mask_source(texts)
    has_bbox_overreach = any("bbox_overreach" in (text.get("qa_flags") or []) for text in texts)
    has_bbox_overreach_critical = any("bbox_overreach_critical" in (text.get("qa_flags") or []) for text in texts)
    synthetic_tight_balloon_reference = _uses_synthetic_tight_balloon_reference(
        texts,
        width,
        height,
        mask_source=mask_source,
        used_balloon_clip=used_balloon_clip,
        source_glyph_area_ratio=source_glyph_area_ratio,
        mask_balloon_ratio=mask_balloon_ratio,
    )
    edge_clipped_text_bbox_reference = _uses_edge_clipped_text_bbox_reference(
        texts,
        width,
        height,
        mask_source=mask_source,
        used_balloon_clip=used_balloon_clip,
        outside_balloon_ratio=outside_balloon_ratio,
        mask_balloon_ratio=mask_balloon_ratio,
        expanded_raw_ratio=expanded_raw_ratio,
        has_bbox_overreach=has_bbox_overreach or has_bbox_overreach_critical,
    )
    mask_density_high = _mask_density_high_gate(
        texts,
        mask_source=mask_source,
        mask_density=mask_density,
        outside_balloon_ratio=outside_balloon_ratio,
        expanded_raw_ratio=expanded_raw_ratio,
        source_glyph_area_ratio=source_glyph_area_ratio,
        has_bbox_overreach=has_bbox_overreach or has_bbox_overreach_critical,
        synthetic_tight_balloon_reference=synthetic_tight_balloon_reference,
    )
    source_glyph_area_critical = bool(
        balloon_metadata.get("used_derived_bubble_mask")
        and source_glyph_area_ratio >= SOURCE_GLYPH_CRITICAL
        and not synthetic_tight_balloon_reference
    )
    card_panel_text_context = _texts_card_panel_context(texts)
    weak_image_bubble_ids = _weak_image_bubble_mask_text_ids(texts, width, height)
    outside_balloon_critical = (
        outside_balloon_pixels > OUTSIDE_BALLOON_CRITICAL_PIXELS
        and outside_balloon_ratio >= OUTSIDE_BALLOON_CRITICAL_RATIO
        and not synthetic_tight_balloon_reference
        and not edge_clipped_text_bbox_reference
        and not card_panel_text_context
    )
    ids = _text_ids(texts)
    has_any_bubble_reference = bool(
        balloon_metadata.get("used_real_bubble_mask")
        or balloon_metadata.get("used_image_bubble_mask")
        or balloon_metadata.get("used_derived_bubble_mask")
        or balloon_metadata.get("used_balloon_bbox_fallback")
        or balloon_metadata.get("used_balloon_polygon")
    )
    gates = {
        "mask_density_high": mask_density_high,
        "mask_outside_balloon": (
            outside_balloon_pixels > 0
            and outside_balloon_ratio >= OUTSIDE_BALLOON_WARN_RATIO
        ),
        "mask_outside_balloon_critical": outside_balloon_critical,
        "bbox_overreach": has_bbox_overreach,
        "bbox_overreach_critical": has_bbox_overreach_critical,
        "expanded_ratio_review": expanded_raw_ratio > EXPANDED_RAW_WARN,
        "source_glyph_area_ratio_critical": source_glyph_area_critical,
        "bbox_fallback_bubble_mask": bool(
            (
                balloon_metadata.get("used_balloon_bbox_fallback")
                or balloon_metadata.get("used_balloon_polygon")
                or (
                    balloon_metadata.get("used_derived_bubble_mask")
                    and balloon_metadata.get("bubble_mask_source") != "derived_card_panel_mask"
                )
            )
            and not balloon_metadata.get("used_real_bubble_mask")
        ),
        "missing_real_bubble_mask": bool(ids) and not has_any_bubble_reference and not card_panel_text_context,
        "weak_image_bubble_mask_reference": bool(weak_image_bubble_ids),
    }
    flags = [name for name, enabled in gates.items() if enabled]
    band_id = _band_id(ocr_page)
    trace_ids = _trace_ids(texts, ids, band_id)
    text_instance_ids = _text_instance_ids(texts, ids, band_id)
    decision = {
        "schema_version": 1,
        "band_id": band_id,
        "text_id": ids[0] if len(ids) == 1 else None,
        "text_ids": ids,
        "trace_ids": trace_ids,
        "trace_ids_in_band": trace_ids,
        "text_instance_ids": text_instance_ids,
        "mask_source": mask_source,
        "used_balloon_clip": used_balloon_clip,
        "used_real_bubble_mask": bool(balloon_metadata["used_real_bubble_mask"]),
        "used_image_bubble_mask": bool(balloon_metadata.get("used_image_bubble_mask")),
        "used_derived_bubble_mask": bool(balloon_metadata.get("used_derived_bubble_mask")),
        "used_balloon_polygon": bool(balloon_metadata["used_balloon_polygon"]),
        "used_balloon_bbox_fallback": bool(balloon_metadata["used_balloon_bbox_fallback"]),
        "bubble_mask_source": str(balloon_metadata["bubble_mask_source"]),
        "bubble_mask_rejection_reason": balloon_metadata.get("bubble_mask_rejection_reason"),
        "weak_image_bubble_mask_text_ids": weak_image_bubble_ids,
        "synthetic_tight_balloon_reference": synthetic_tight_balloon_reference,
        "edge_clipped_text_bbox_reference": edge_clipped_text_bbox_reference,
        "card_panel_text_context": card_panel_text_context,
        "used_protection_mask": bool(np.any(protection)),
        "raw_mask_pixels": raw_pixels,
        "expanded_mask_pixels": expanded_pixels,
        "final_mask_pixels": final_pixels,
        "balloon_mask_pixels": balloon_pixels,
        "balloon_inner_mask_pixels": int(np.count_nonzero(balloon_inner_mask)),
        "balloon_outline_mask_pixels": balloon_outline_pixels,
        "mask_balloon_ratio": mask_balloon_ratio,
        "outside_balloon_pixels": outside_balloon_pixels,
        "outside_balloon_ratio": outside_balloon_ratio,
        "outside_balloon_reference": "final_mask" if outside_reference is final else "expanded_mask",
        "expanded_raw_ratio": expanded_raw_ratio,
        "mask_density_in_band": mask_density,
        "source_bbox_area": source_area,
        "glyph_bbox_area": glyph_bbox_area,
        "source_glyph_area_ratio": source_glyph_area_ratio,
        "flags": flags,
        "gates": gates,
        "thresholds": {
            "mask_density_warn": MASK_DENSITY_WARN,
            "mask_density_borderline_warn": MASK_DENSITY_BORDERLINE_WARN,
            "mask_density_strong_warn": MASK_DENSITY_STRONG_WARN,
            "expanded_raw_warn": EXPANDED_RAW_WARN,
            "source_glyph_review": SOURCE_GLYPH_REVIEW,
            "source_glyph_critical": SOURCE_GLYPH_CRITICAL,
            "outside_balloon_warn_ratio": OUTSIDE_BALLOON_WARN_RATIO,
            "clean_outside_balloon_ratio": CLEAN_OUTSIDE_BALLOON_RATIO,
            "clean_expanded_raw_ratio": CLEAN_EXPANDED_RAW_RATIO,
            "clean_source_glyph_ratio": CLEAN_SOURCE_GLYPH_RATIO,
            "outside_balloon_critical_pixels": OUTSIDE_BALLOON_CRITICAL_PIXELS,
            "outside_balloon_critical_ratio": OUTSIDE_BALLOON_CRITICAL_RATIO,
        },
    }
    images = {
        "01_glyph_mask.png": glyph_mask,
        "02_line_polygon_mask.png": line_polygon_mask,
        "03_detected_text_mask.png": detected_text_mask,
        "04_balloon_mask.png": balloon_mask,
        "05_balloon_inner_mask.png": balloon_inner_mask,
        "06_protection_mask.png": protection,
        "07_raw_text_mask.png": raw,
        "08_expanded_text_mask.png": expanded,
        "09_final_inpaint_mask.png": final,
        "10_mask_overlay.jpg": _overlay_mask(image_rgb, final),
    }
    return decision, images


def _summary_from_decisions(decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_source: Counter[str] = Counter()
    flagged_bands: list[str] = []
    totals = {
        "raw_mask_pixels": 0,
        "expanded_mask_pixels": 0,
        "outside_balloon_pixels": 0,
    }
    bands_with_mask = 0
    for band_id, decision in sorted(decisions.items()):
        if int(decision.get("expanded_mask_pixels") or 0) > 0:
            bands_with_mask += 1
        if decision.get("flags"):
            flagged_bands.append(band_id)
        by_source[str(decision.get("mask_source") or "unknown")] += 1
        for key in totals:
            totals[key] += int(decision.get(key) or 0)
    return {
        "schema_version": 1,
        "band_count": len(decisions),
        "bands_with_mask": bands_with_mask,
        "bands_with_flags": len(flagged_bands),
        "totals": totals,
        "by_source": dict(sorted(by_source.items())),
        "flagged_bands": flagged_bands,
    }


def _recorder_key(recorder: Any) -> int:
    return id(recorder)


def _write_recorder_image(recorder: Any, rel_path: str, image: np.ndarray) -> None:
    output = image
    if output.ndim == 3 and output.shape[2] >= 3:
        output = cv2.cvtColor(output[:, :, :3], cv2.COLOR_RGB2BGR)
    recorder.write_image(rel_path, output)


def _safe_path_segment(value: Any) -> str:
    raw = str(value or "").strip() or "text"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return safe.strip("._") or "text"


def _per_text_action_mask(mask: np.ndarray | None, text: dict[str, Any], width: int, height: int) -> np.ndarray | None:
    base = _as_mask(mask, height, width)
    if not np.any(base):
        return base
    balloon_mask, _metadata = _balloon_mask_from_texts([text], width, height)
    if np.any(balloon_mask):
        clipped = np.where((base > 0) & (balloon_mask > 0), 255, 0).astype(np.uint8)
        if np.any(clipped):
            return clipped
    reference = _union_mask_from_texts([text], width, height, "glyph")
    reference |= _union_mask_from_texts([text], width, height, "detected")
    if np.any(reference):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        reference = cv2.dilate(reference, kernel, iterations=1)
        clipped = np.where((base > 0) & (reference > 0), 255, 0).astype(np.uint8)
        if np.any(clipped):
            return clipped
    return _blank_mask(height, width)


def _write_per_text_mask_chain_debug_artifacts(
    recorder: Any,
    band_id: str,
    texts: list[dict[str, Any]],
    *,
    image_rgb: np.ndarray,
    raw_mask: np.ndarray | None,
    expanded_mask: np.ndarray | None,
    final_mask: np.ndarray | None,
    protection_mask: np.ndarray | None,
) -> None:
    if len(texts) <= 1:
        return
    height, width = _image_hw(image_rgb)
    for index, text in enumerate(texts, start=1):
        text_id = str(text.get("id") or text.get("text_id") or f"text_{index:03d}")
        text_ocr_page = {"band_id": band_id, "texts": [text]}
        text_raw = _per_text_action_mask(raw_mask, text, width, height)
        text_expanded = _per_text_action_mask(expanded_mask, text, width, height)
        text_final = _per_text_action_mask(final_mask, text, width, height) if final_mask is not None else text_expanded
        text_protection = _per_text_action_mask(protection_mask, text, width, height) if protection_mask is not None else None
        decision, images = build_mask_chain_debug_payload(
            text_ocr_page,
            image_rgb=image_rgb,
            raw_mask=text_raw,
            expanded_mask=text_expanded,
            final_mask=text_final,
            protection_mask=text_protection,
        )
        base = f"06_mask_segmentation/{band_id}/per_text/{_safe_path_segment(text_id)}"
        for filename, image in images.items():
            _write_recorder_image(recorder, f"{base}/{filename}", image)
        recorder.write_json(f"{base}/mask_decision.json", decision)


def write_mask_chain_debug_artifacts(
    recorder: Any,
    ocr_page: dict[str, Any],
    *,
    image_rgb: np.ndarray,
    raw_mask: np.ndarray | None,
    expanded_mask: np.ndarray | None,
    final_mask: np.ndarray | None = None,
    protection_mask: np.ndarray | None = None,
) -> dict[str, Any] | None:
    try:
        decision, images = build_mask_chain_debug_payload(
            ocr_page,
            image_rgb=image_rgb,
            raw_mask=raw_mask,
            expanded_mask=expanded_mask,
            final_mask=final_mask,
            protection_mask=protection_mask,
        )
        band_id = str(decision["band_id"])
        base = f"06_mask_segmentation/{band_id}"
        texts = [
            text
            for text in ocr_page.get("texts", [])
            if isinstance(text, dict) and not _text_has_rejected_bubble_without_glyph_evidence(text)
        ]
        if len(texts) > 1:
            ids = _text_ids(texts)
            decision["decision_scope"] = "band_aggregate_debug_only"
            decision["actionable"] = False
            decision["layout_inpaint_decision_source"] = "per_text"
            decision["per_text_decision_paths"] = [
                f"per_text/{_safe_path_segment(text_id)}/mask_decision.json"
                for text_id in ids
            ]
        else:
            decision["decision_scope"] = "text_actionable"
            decision["actionable"] = True
        for filename, image in images.items():
            _write_recorder_image(recorder, f"{base}/{filename}", image)
        recorder.write_json(f"{base}/mask_decision.json", decision)
        _write_per_text_mask_chain_debug_artifacts(
            recorder,
            band_id,
            texts,
            image_rgb=image_rgb,
            raw_mask=raw_mask,
            expanded_mask=expanded_mask,
            final_mask=final_mask,
            protection_mask=protection_mask,
        )
        state = _MASK_SUMMARY_STATE.setdefault(_recorder_key(recorder), {})
        state[band_id] = decision
        recorder.write_json("06_mask_segmentation/mask_chain_summary.json", _summary_from_decisions(state))
        return decision
    except Exception as exc:
        try:
            recorder.event(
                "mask_segmentation",
                "mask_chain_debug_failed",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        except Exception:
            pass
        return None
