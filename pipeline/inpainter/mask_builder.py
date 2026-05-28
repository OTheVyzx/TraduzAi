"""
Builds expanded mask regions from OCR boxes.
This is the first step toward page-level inpainting that respects balloons
instead of removing each OCR box in isolation.
"""

from __future__ import annotations

from collections import Counter
import re

import cv2
import numpy as np

try:
    from ocr.text_router import ROUTE_ACTIONS, route_action_requires_inpaint
except ImportError:
    from ..ocr.text_router import ROUTE_ACTIONS, route_action_requires_inpaint

OVERREACH_RATIO_FLAG = 2.5
OVERREACH_RATIO_CRITICAL = 4.0
MASK_DENSITY_HIGH = 0.12
MASK_OUTSIDE_BALLOON_FLAG = 0.08
MASK_OUTSIDE_BALLOON_CRITICAL = 0.18
SPECIAL_MASK_CLASSES = {
    "sfx",
    "sound_effect",
    "watermark",
    "scanlator_credit",
    "tn_note",
    "noise",
}
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
    expand = 0
    background = _rgb_luma_chroma(block.get("background_rgb"))
    non_plain_background = bool(background is not None and (background[0] < 228.0 or background[1] > 18.0))
    if style.get("italico") or profile == "sfx" or non_plain_background:
        expand = 7
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
    if erode_px > 0:
        kernel_size = int(erode_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        interior = cv2.erode(balloon_mask.astype(np.uint8), kernel, iterations=1)
    else:
        interior = balloon_mask.astype(np.uint8)
    if not np.any(interior):
        interior = balloon_mask.astype(np.uint8)
    clipped = cv2.bitwise_and(text_mask.astype(np.uint8), interior)
    return clipped if np.any(clipped) else text_mask


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


def balloon_mask_from_block(block: dict, image_shape: tuple[int, ...]) -> np.ndarray | None:
    height, width = _image_hw(image_shape)
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

    return mask if np.any(mask) else None


def _dark_glyph_pixels_in_text_bbox(
    block: dict,
    image_rgb: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None
    height, width = _image_hw(image_shape)
    bbox = _normalize_bbox(block.get("text_pixel_bbox"), width, height)
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


def _koharu_glyph_dilate_radius(block: dict) -> int:
    font_size = block.get("detected_font_size_px") or block.get("font_size")
    try:
        numeric = float(font_size)
    except Exception:
        numeric = 18.0
    return int(max(2, min(8, round(numeric * 0.16))))


def build_inpaint_mask(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None = None,
) -> np.ndarray | None:
    text_mask = None
    raw_text_mask = None
    geometry_mask = build_glyph_text_mask(block, image_shape)
    if image_rgb is not None and image_rgb.size:
        raw_text_mask = build_raw_text_mask_from_image(block, image_rgb, image_shape)
        if isinstance(raw_text_mask, np.ndarray) and np.any(raw_text_mask):
            text_mask = raw_text_mask.astype(np.uint8)
            if (
                isinstance(geometry_mask, np.ndarray)
                and np.any(geometry_mask)
                and _mask_is_overbroad_against_geometry(text_mask, geometry_mask)
            ):
                text_mask = geometry_mask.astype(np.uint8)
        elif raw_text_evidence_rejected(block):
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

    dark_bbox_mask = _dark_glyph_pixels_in_text_bbox(block, image_rgb, image_shape)
    if isinstance(dark_bbox_mask, np.ndarray) and np.any(dark_bbox_mask):
        text_mask = cv2.bitwise_or(text_mask.astype(np.uint8), dark_bbox_mask.astype(np.uint8))

    bubble_mask = block.get("bubble_mask")
    bubble_id = block.get("bubble_id") or block.get("bubbleId")
    text_mask = clip_mask_to_bubble_id(text_mask, bubble_mask, bubble_id)

    radius = _koharu_glyph_dilate_radius(block)
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1))
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        text_mask = clip_mask_to_bubble_id(text_mask, bubble_mask, bubble_id)

    text_mask = text_mask.astype(np.uint8)
    consolidate_mask_evidence(
        block,
        kind="ocr_pixels" if isinstance(raw_text_mask, np.ndarray) and np.any(raw_text_mask) else "clipped_line_polygon",
        raw_mask_pixels=int(np.count_nonzero(raw_text_mask)) if isinstance(raw_text_mask, np.ndarray) else 0,
        expanded_mask_pixels=int(np.count_nonzero(text_mask)),
        evidence_score=1.0,
    )
    return text_mask


def build_mask_regions(texts: list[dict], image_shape: tuple[int, int, int]) -> list[dict]:
    image_height, image_width = image_shape[:2]
    seeds = []

    for text in texts:
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
    if not isinstance(bubble_mask, np.ndarray) or bubble_mask.shape[:2] != text_mask.shape[:2]:
        return text_mask
    numeric_id = _numeric_bubble_id(text.get("bubble_id") or text.get("bubbleId"))
    allowed = (bubble_mask == numeric_id) if numeric_id is not None else (bubble_mask > 0)
    return np.where((text_mask > 0) & allowed, 255, 0).astype(np.uint8)
