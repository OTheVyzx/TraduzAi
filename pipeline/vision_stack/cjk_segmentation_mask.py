from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import cv2
import numpy as np

try:
    from inpainter.mask_builder import build_inpaint_mask
    from ocr.text_router import ROUTE_ACTIONS, route_action_requires_inpaint
except ImportError:
    from ..inpainter.mask_builder import build_inpaint_mask
    from ..ocr.text_router import ROUTE_ACTIONS, route_action_requires_inpaint

logger = logging.getLogger(__name__)


def expand_cjk_glyph_mask_for_inpaint(
    mask: np.ndarray,
    *,
    min_radius: int = 1,
    max_radius: int = 6,
    component_ratio: float = 0.16,
) -> np.ndarray:
    """Expand glyph pixels for inpainting without filling the enclosing text region."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    base = (mask > 0).astype(np.uint8) * 255
    if not np.any(base):
        return base

    count, labels, stats, _ = cv2.connectedComponentsWithStats(base, connectivity=8)
    expanded = base.copy()
    height, width = base.shape[:2]
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w_box = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w_box <= 0 or h_box <= 0:
            continue
        radius = int(round(min(w_box, h_box) * float(component_ratio)))
        radius = max(int(min_radius), min(int(max_radius), radius))
        if radius <= 0:
            continue
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(width, x + w_box + radius)
        y2 = min(height, y + h_box + radius)
        local = (labels[y1:y2, x1:x2] == label).astype(np.uint8) * 255
        dilated = cv2.dilate(
            local,
            cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1)),
            iterations=1,
        )
        expanded[y1:y2, x1:x2] = np.maximum(expanded[y1:y2, x1:x2], dilated)
    return expanded.astype(np.uint8)


def _image_hw(image_rgb: np.ndarray) -> tuple[int, int]:
    return int(image_rgb.shape[0]), int(image_rgb.shape[1])


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


def _bbox_union(a: list[int] | None, b: list[int] | None) -> list[int] | None:
    if a is None:
        return b
    if b is None:
        return a
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _expand_bbox(bbox: list[int], width: int, height: int, pad: int) -> list[int] | None:
    return _normalize_bbox([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], width, height)


def _bbox_area(bbox: list[int]) -> int:
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _to_binary_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if isinstance(mask, dict):
        for key in ("mask", "segmentation_mask", "probability_map", "text_mask"):
            if key in mask:
                mask = mask[key]
                break
    if isinstance(mask, (list, tuple)) and mask and isinstance(mask[0], np.ndarray):
        mask = mask[0]
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if np.issubdtype(mask.dtype, np.floating):
        binary = (mask > 0.35).astype(np.uint8) * 255
    else:
        binary = (mask > 0).astype(np.uint8) * 255
    return binary if np.any(binary) else None


def _run_segmenter(segmenter: Any, crop: np.ndarray) -> np.ndarray | None:
    if segmenter is None:
        return None
    try:
        output = None
        if callable(segmenter):
            output = segmenter(crop)
        elif hasattr(segmenter, "segment"):
            output = segmenter.segment(crop)
        elif hasattr(segmenter, "predict"):
            output = segmenter.predict(crop)
    except Exception as exc:
        logger.warning("Segmentador CJK falhou em crop %s: %s", crop.shape[:2], exc)
        return None
    return _to_binary_mask(output, crop.shape[:2])


def _candidate_is_acceptable(
    local_mask: np.ndarray,
    *,
    roi_area: int,
    reference_mask: np.ndarray | None = None,
) -> bool:
    area = int(np.count_nonzero(local_mask))
    if area < max(8, int(roi_area * 0.00035)):
        return False
    if area > int(roi_area * 0.58):
        return False

    bbox = _bbox_from_mask(local_mask)
    if bbox is None:
        return False
    mask_bbox_area = max(1, _bbox_area(bbox))
    fill_ratio = area / float(mask_bbox_area)
    if fill_ratio >= 0.86 and area >= 64:
        return False

    if reference_mask is not None and np.any(reference_mask):
        ref_area = int(np.count_nonzero(reference_mask))
        overlap = int(np.count_nonzero((local_mask > 0) & (reference_mask > 0)))
        if overlap <= 0:
            return False
        if area > max(32, ref_area * 5) and overlap < int(ref_area * 0.35):
            return False
    return True


def _geometry_fallback_mask(block: dict, image_rgb: np.ndarray) -> np.ndarray | None:
    try:
        return build_inpaint_mask(block, image_rgb.shape, image_rgb=image_rgb)
    except Exception:
        return None


def _reference_mask_for_block(block: dict, image_rgb: np.ndarray, roi: list[int]) -> np.ndarray | None:
    full = _geometry_fallback_mask(block, image_rgb)
    if full is None or not np.any(full):
        return None
    x1, y1, x2, y2 = roi
    local = full[y1:y2, x1:x2]
    return local if np.any(local) else None


def _clip_to_bubble_when_dominant(mask: np.ndarray, bubble_mask: np.ndarray | None) -> np.ndarray:
    if bubble_mask is None or not np.any(mask) or not np.any(bubble_mask):
        return mask
    overlap = cv2.bitwise_and(mask, bubble_mask)
    overlap_area = int(np.count_nonzero(overlap))
    mask_area = int(np.count_nonzero(mask))
    if mask_area <= 0:
        return mask
    return overlap if overlap_area >= int(mask_area * 0.72) else mask


def _absorb_dark_text_core(
    mask: np.ndarray | None,
    image_rgb: np.ndarray,
    support_mask: np.ndarray | None = None,
    *,
    aggressive: bool = False,
) -> np.ndarray | None:
    if mask is None or not np.any(mask):
        return mask
    if image_rgb.shape[:2] != mask.shape[:2]:
        return mask

    base = (mask > 0).astype(np.uint8) * 255
    base_area = int(np.count_nonzero(base))
    kernel_size = 17 if aggressive else 9
    search = cv2.dilate(
        base,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=2,
    )
    if support_mask is not None and support_mask.shape[:2] == mask.shape[:2]:
        search = cv2.bitwise_and(search, (support_mask > 0).astype(np.uint8) * 255)
    if not np.any(search):
        return mask

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dark = ((gray <= 118) & (search > 0)).astype(np.uint8)
    if not np.any(dark):
        return mask
    if aggressive:
        return np.maximum(base, dark * 255).astype(np.uint8)

    nearby = cv2.dilate(
        base,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=2,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    recovered = np.zeros_like(base, dtype=np.uint8)
    image_area = max(1, int(base.shape[0]) * int(base.shape[1]))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w_box = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_area = max(1, w_box * h_box)
        fill_ratio = area / float(bbox_area)
        aspect_ratio = max(w_box, h_box) / float(max(1, min(w_box, h_box)))
        if aspect_ratio >= 14.0 and fill_ratio < 0.28:
            continue
        if area < 18 and fill_ratio < 0.22:
            continue
        if area > max(2500, int(base_area * 0.75)) and fill_ratio > 0.42:
            continue
        if area > int(image_area * 0.018) and fill_ratio > 0.34:
            continue

        component = labels == label
        if not np.any(component & (nearby > 0)):
            continue
        recovered[component] = 255

    if not np.any(recovered):
        return mask
    return np.maximum(base, recovered).astype(np.uint8)


def _full_segmentation_from_blocks(image_rgb: np.ndarray, blocks: list[dict]) -> np.ndarray | None:
    height, width = _image_hw(image_rgb)
    full = np.zeros((height, width), dtype=np.uint8)
    for block in blocks:
        if not isinstance(block, dict):
            continue
        bbox = _normalize_bbox(block.get("bbox"), width, height)
        if bbox is None:
            continue
        local = _to_binary_mask(block.get("mask"), (bbox[3] - bbox[1], bbox[2] - bbox[0]))
        if local is None:
            continue
        x1, y1, x2, y2 = bbox
        full[y1:y2, x1:x2] = np.maximum(full[y1:y2, x1:x2], local)
    return full if np.any(full) else None


def _expanded_text_support_mask(
    shape: tuple[int, int],
    blocks: list[dict],
    ocr_texts: list[dict] | None,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    support = np.zeros(shape, dtype=np.uint8)
    items = [item for item in list(blocks or []) + list(ocr_texts or []) if isinstance(item, dict)]
    for item in items:
        if _is_preserved_item(item):
            continue
        bbox = None
        for key in ("bbox", "text_pixel_bbox", "balloon_bbox"):
            bbox = _bbox_union(bbox, _normalize_bbox(item.get(key), width, height))
        if bbox is None:
            continue
        box_w = max(1, bbox[2] - bbox[0])
        box_h = max(1, bbox[3] - bbox[1])
        pad = max(36, int(round(max(box_w, box_h) * 0.35)))
        expanded = _expand_bbox(bbox, width, height, pad)
        if expanded is None:
            continue
        x1, y1, x2, y2 = expanded
        support[y1:y2, x1:x2] = 255
    return support


def _recover_textlike_strokes_from_candidate(
    candidate_mask: np.ndarray,
    image_rgb: np.ndarray,
    *,
    existing_mask: np.ndarray | None = None,
    allow_cool_saturated: bool = False,
) -> np.ndarray:
    if not np.any(candidate_mask) or image_rgb.shape[:2] != candidate_mask.shape[:2]:
        return np.zeros(candidate_mask.shape[:2], dtype=np.uint8)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    candidate = candidate_mask > 0
    dark = gray <= 92
    warm_or_purple = (hue <= 24) | (hue >= 145) | ((hue >= 25) & (hue <= 45) & (saturation >= 105))
    if allow_cool_saturated:
        warm_or_purple = warm_or_purple | ((hue >= 82) & (hue <= 135) & (saturation >= 95))
    value_ceiling = 255 if allow_cool_saturated else 252
    saturated = (saturation >= 70) & (value >= 42) & (value <= value_ceiling) & warm_or_purple
    candidate_gray = gray[candidate]
    local_median = float(np.median(candidate_gray)) if candidate_gray.size else 255.0
    local_p75 = float(np.percentile(candidate_gray, 75)) if candidate_gray.size else 255.0
    bright = (gray >= 218) & (local_median <= 176.0) & (local_p75 <= 210.0)

    raw = ((dark | saturated | bright) & candidate).astype(np.uint8) * 255
    if existing_mask is not None and existing_mask.shape[:2] == raw.shape[:2]:
        raw = cv2.bitwise_and(raw, cv2.bitwise_not((existing_mask > 0).astype(np.uint8) * 255))
    if not np.any(raw):
        return np.zeros_like(raw, dtype=np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    recovered = np.zeros_like(raw, dtype=np.uint8)
    image_area = max(1, int(raw.shape[0]) * int(raw.shape[1]))
    if allow_cool_saturated:
        max_component_area = max(9000, int(image_area * 0.05))
    else:
        max_component_area = max(2500, int(image_area * 0.018))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 10 or area > max_component_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w_box = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w_box <= 0 or h_box <= 0:
            continue
        bbox_area = max(1, w_box * h_box)
        fill_ratio = area / float(bbox_area)
        aspect_ratio = max(w_box, h_box) / float(max(1, min(w_box, h_box)))
        if fill_ratio < 0.075:
            continue
        if fill_ratio > 0.92 and area > 128:
            continue
        if aspect_ratio > 18.0 and fill_ratio < 0.24:
            continue
        if aspect_ratio > 24.0 and min(w_box, h_box) <= 10:
            continue
        if min(w_box, h_box) <= 2 and max(w_box, h_box) > 20:
            continue
        if max(w_box, h_box) > 90 and fill_ratio < 0.12:
            continue
        if area < 18 and max(w_box, h_box) < 8:
            continue
        recovered[labels == label] = 255

    if not np.any(recovered):
        return recovered

    recovered = cv2.dilate(
        recovered,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    allowed = cv2.dilate(
        (candidate_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return cv2.bitwise_and(recovered, allowed).astype(np.uint8)


def _recover_orphan_segmentation_components(
    current_mask: np.ndarray,
    image_rgb: np.ndarray,
    blocks: list[dict],
    ocr_texts: list[dict] | None,
    segmenter: Callable[[np.ndarray], Any] | Any | None,
) -> np.ndarray:
    if segmenter is None:
        return current_mask
    if not np.any(current_mask):
        return current_mask
    height, width = _image_hw(image_rgb)
    page_mask = _run_segmenter(segmenter, image_rgb)
    if page_mask is None or not np.any(page_mask):
        return current_mask

    support = _expanded_text_support_mask(
        current_mask.shape,
        blocks,
        ocr_texts,
        width=width,
        height=height,
    )
    if not np.any(support):
        return current_mask

    guard = cv2.dilate(
        (current_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    orphan_candidates = cv2.bitwise_and(page_mask.astype(np.uint8), cv2.bitwise_not(guard))
    global_stroke_recovery = _recover_textlike_strokes_from_candidate(
        orphan_candidates,
        image_rgb,
        existing_mask=current_mask,
    )
    orphan_seed = cv2.bitwise_and(orphan_candidates, support)
    if not np.any(orphan_seed):
        if np.any(global_stroke_recovery):
            image_area = max(1, int(height) * int(width))
            if int(np.count_nonzero(global_stroke_recovery)) <= int(image_area * 0.05):
                return np.maximum(current_mask, global_stroke_recovery.astype(np.uint8))
        return current_mask
    chain_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    orphan = orphan_seed
    for _ in range(3):
        chained_support = cv2.dilate(orphan, chain_kernel, iterations=1)
        expanded = cv2.bitwise_and(orphan_candidates, cv2.bitwise_or(support, chained_support))
        if int(np.count_nonzero(expanded)) == int(np.count_nonzero(orphan)):
            break
        orphan = expanded

    stroke_recovery = _recover_textlike_strokes_from_candidate(orphan, image_rgb, existing_mask=current_mask)
    if np.any(global_stroke_recovery):
        image_area = max(1, int(height) * int(width))
        if int(np.count_nonzero(global_stroke_recovery)) <= int(image_area * 0.05):
            stroke_recovery = np.maximum(stroke_recovery, global_stroke_recovery)
    recovered = np.zeros_like(current_mask, dtype=np.uint8)
    image_area = max(1, int(height) * int(width))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((orphan > 0).astype(np.uint8), connectivity=8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w_box = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_area = max(1, int(w_box) * int(h_box))
        if area > int(image_area * 0.035):
            continue
        fill_ratio = area / float(bbox_area)
        if area > max(4500, int(image_area * 0.008)):
            continue
        if area > 900 and fill_ratio > 0.34:
            continue
        if area > 1800 and max(w_box, h_box) > 70:
            continue
        if fill_ratio >= 0.92 and area >= 96:
            continue
        recovered[labels == label] = 255

    if np.any(stroke_recovery):
        recovered = np.maximum(recovered, stroke_recovery)
    if not np.any(recovered):
        return current_mask
    recovered = _absorb_dark_text_core(recovered.astype(np.uint8), image_rgb, support, aggressive=False)
    if recovered is None or not np.any(recovered):
        return current_mask
    return np.maximum(current_mask, recovered.astype(np.uint8))


def _is_preserved_item(item: dict) -> bool:
    route_action = str(item.get("route_action") or "").strip().lower()
    if route_action in ROUTE_ACTIONS:
        return not route_action_requires_inpaint(route_action)
    return bool(item.get("preserve_original") or item.get("skip_processing") or item.get("ignored_reason"))


def _active_items(items: list[dict] | None) -> list[dict]:
    return [item for item in items or [] if isinstance(item, dict) and not _is_preserved_item(item)]


def _block_has_only_preserved_texts(block: dict, ocr_texts: list[dict], width: int, height: int) -> bool:
    if _is_preserved_item(block):
        return True
    matches = _matching_ocr_texts(block, ocr_texts, width, height)
    return bool(matches) and all(_is_preserved_item(text) for text in matches)


def build_manga_segmentation_mask(
    image_rgb: np.ndarray,
    blocks: list[dict],
    segmentation_mask: np.ndarray | None,
    bubble_regions: list[dict] | None = None,
    ocr_texts: list[dict] | None = None,
    segmenter: Callable[[np.ndarray], Any] | Any | None = None,
    bubble_segmenter: Callable[[np.ndarray], Any] | Any | None = None,
) -> np.ndarray:
    del bubble_regions
    height, width = _image_hw(image_rgb)
    full = np.zeros((height, width), dtype=np.uint8)
    page_texts = [text for text in ocr_texts or [] if isinstance(text, dict)]
    segmentation = _to_binary_mask(segmentation_mask, (height, width)) if segmentation_mask is not None else None
    if segmentation is None:
        segmentation = _full_segmentation_from_blocks(image_rgb, blocks)

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if _block_has_only_preserved_texts(block, page_texts, width, height):
            continue
        bbox = _normalize_bbox(block.get("bbox") or block.get("text_pixel_bbox"), width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = image_rgb[y1:y2, x1:x2]
        candidate = segmentation[y1:y2, x1:x2] if segmentation is not None else None
        if candidate is None and segmenter is not None:
            if crop.size:
                candidate = _run_segmenter(segmenter, crop)
                bubble_mask = _run_segmenter(bubble_segmenter, crop)
                if candidate is not None:
                    candidate = _clip_to_bubble_when_dominant(candidate, bubble_mask)
        reference = _reference_mask_for_block(block, image_rgb, bbox)
        if (
            candidate is not None
            and np.any(candidate)
            and _candidate_is_acceptable(candidate, roi_area=_bbox_area(bbox), reference_mask=reference)
        ):
            candidate = _absorb_dark_text_core(candidate.astype(np.uint8), crop)
            if candidate is None:
                continue
            full[y1:y2, x1:x2] = np.maximum(full[y1:y2, x1:x2], candidate.astype(np.uint8))
            continue
        if candidate is not None and np.any(candidate):
            stroke_candidate = _recover_textlike_strokes_from_candidate(candidate, crop)
            if (
                np.any(stroke_candidate)
                and _candidate_is_acceptable(stroke_candidate, roi_area=_bbox_area(bbox), reference_mask=reference)
            ):
                full[y1:y2, x1:x2] = np.maximum(
                    full[y1:y2, x1:x2],
                    stroke_candidate.astype(np.uint8),
                )
                continue

        fallback = _geometry_fallback_mask(block, image_rgb)
        if fallback is not None and np.any(fallback):
            full = np.maximum(full, fallback.astype(np.uint8))

    return _recover_orphan_segmentation_components(full, image_rgb, blocks, page_texts, segmenter)


def _matching_ocr_texts(block: dict, ocr_texts: list[dict], width: int, height: int) -> list[dict]:
    block_bbox = _normalize_bbox(block.get("bbox") or block.get("balloon_bbox"), width, height)
    if block_bbox is None:
        return []
    matched: list[dict] = []
    for text in ocr_texts or []:
        if not isinstance(text, dict):
            continue
        bbox = _normalize_bbox(text.get("text_pixel_bbox") or text.get("bbox"), width, height)
        if bbox is None:
            continue
        ix1 = max(block_bbox[0], bbox[0])
        iy1 = max(block_bbox[1], bbox[1])
        ix2 = min(block_bbox[2], bbox[2])
        iy2 = min(block_bbox[3], bbox[3])
        if ix2 > ix1 and iy2 > iy1:
            matched.append(text)
    return matched


def _roi_for_block(block: dict, ocr_texts: list[dict], width: int, height: int) -> list[int] | None:
    roi = None
    for key in ("balloon_bbox", "bbox", "text_pixel_bbox"):
        roi = _bbox_union(roi, _normalize_bbox(block.get(key), width, height))
    for text in _matching_ocr_texts(block, ocr_texts, width, height):
        roi = _bbox_union(roi, _normalize_bbox(text.get("text_pixel_bbox") or text.get("bbox"), width, height))
    if roi is None:
        return None
    pad = max(32, int(round(max(roi[2] - roi[0], roi[3] - roi[1]) * 0.20)))
    return _expand_bbox(roi, width, height, pad)


def _split_tall_roi(roi: list[int], max_height: int = 1400) -> list[list[int]]:
    x1, y1, x2, y2 = roi
    if y2 - y1 <= max_height:
        return [roi]
    windows: list[list[int]] = []
    overlap = 64
    start = y1
    while start < y2:
        end = min(y2, start + max_height)
        windows.append([x1, start, x2, end])
        if end >= y2:
            break
        start = max(start + 1, end - overlap)
    return windows


def _fallback_from_block_and_texts(
    image_rgb: np.ndarray,
    block: dict,
    matched_texts: list[dict],
) -> np.ndarray | None:
    fallback = _geometry_fallback_mask(block, image_rgb)
    for text in matched_texts:
        text_fallback = _geometry_fallback_mask(text, image_rgb)
        if text_fallback is not None and np.any(text_fallback):
            fallback = text_fallback if fallback is None else np.maximum(fallback, text_fallback)
    return fallback


def _prefer_textlike_fallback_mask(fallback: np.ndarray | None, image_rgb: np.ndarray) -> np.ndarray | None:
    if fallback is None or not np.any(fallback):
        return fallback
    stroke_fallback = _recover_textlike_strokes_from_candidate(
        fallback,
        image_rgb,
        allow_cool_saturated=True,
    )
    if not np.any(stroke_fallback):
        return fallback
    bbox = _bbox_from_mask(fallback)
    roi_area = _bbox_area(bbox) if bbox is not None else int(fallback.shape[0]) * int(fallback.shape[1])
    if _candidate_is_acceptable(stroke_fallback, roi_area=max(1, roi_area), reference_mask=fallback):
        return stroke_fallback.astype(np.uint8)
    return fallback


def build_manhwa_manhua_roi_segmentation_mask(
    image_rgb: np.ndarray,
    detector_blocks: list[dict],
    ocr_texts: list[dict],
    *,
    segmenter: Callable[[np.ndarray], Any] | Any | None,
    bubble_segmenter: Callable[[np.ndarray], Any] | Any | None = None,
) -> np.ndarray:
    height, width = _image_hw(image_rgb)
    full = np.zeros((height, width), dtype=np.uint8)
    page_texts = [text for text in ocr_texts or [] if isinstance(text, dict)]
    active_texts = _active_items(page_texts)

    for block in detector_blocks:
        if not isinstance(block, dict):
            continue
        if _block_has_only_preserved_texts(block, page_texts, width, height):
            continue
        matched_texts = _active_items(_matching_ocr_texts(block, active_texts, width, height))
        roi = _roi_for_block(block, active_texts, width, height)
        if roi is None:
            continue
        block_added = False
        for window in _split_tall_roi(roi):
            x1, y1, x2, y2 = window
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            local_mask = _run_segmenter(segmenter, crop)
            if local_mask is None:
                continue
            bubble_mask = _run_segmenter(bubble_segmenter, crop)
            local_mask = _clip_to_bubble_when_dominant(local_mask, bubble_mask)
            reference = _reference_mask_for_block(block, image_rgb, window)
            if not _candidate_is_acceptable(local_mask, roi_area=_bbox_area(window), reference_mask=reference):
                stroke_candidate = _recover_textlike_strokes_from_candidate(
                    local_mask,
                    crop,
                    allow_cool_saturated=True,
                )
                if (
                    np.any(stroke_candidate)
                    and _candidate_is_acceptable(stroke_candidate, roi_area=_bbox_area(window), reference_mask=reference)
                ):
                    full[y1:y2, x1:x2] = np.maximum(
                        full[y1:y2, x1:x2],
                        stroke_candidate.astype(np.uint8),
                    )
                    block_added = True
                continue
            stroke_candidate = _recover_textlike_strokes_from_candidate(
                local_mask,
                crop,
                allow_cool_saturated=True,
            )
            if (
                np.any(stroke_candidate)
                and _candidate_is_acceptable(stroke_candidate, roi_area=_bbox_area(window), reference_mask=local_mask)
                and int(np.count_nonzero(local_mask)) > int(np.count_nonzero(stroke_candidate) * 1.45)
            ):
                local_mask = stroke_candidate.astype(np.uint8)
            local_mask = _absorb_dark_text_core(local_mask.astype(np.uint8), crop)
            if local_mask is None:
                continue
            full[y1:y2, x1:x2] = np.maximum(full[y1:y2, x1:x2], local_mask.astype(np.uint8))
            block_added = True
        if block_added:
            continue

        fallback = _fallback_from_block_and_texts(image_rgb, block, matched_texts)
        fallback = _prefer_textlike_fallback_mask(fallback, image_rgb)
        if fallback is not None and np.any(fallback):
            full = np.maximum(full, fallback.astype(np.uint8))

    if height <= 4500:
        return _recover_orphan_segmentation_components(full, image_rgb, detector_blocks, page_texts, segmenter)
    return full
