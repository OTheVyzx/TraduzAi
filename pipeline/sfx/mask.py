from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import cv2
import numpy as np

try:
    from vision_stack.cjk_segmentation_mask import expand_cjk_glyph_mask_for_inpaint
except ImportError:
    from ..vision_stack.cjk_segmentation_mask import expand_cjk_glyph_mask_for_inpaint


HANGUL_RE = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
SFX_ROUTE_ACTION = "translate_sfx_inpaint_render"
SFX_CLASSES = {"sfx", "sound_effect", "sound"}


@dataclass(frozen=True)
class SfxGlyphMaskResult:
    mask: np.ndarray | None
    evidence: dict[str, Any]


def build_sfx_glyph_mask(image_rgb: np.ndarray, layer: dict) -> SfxGlyphMaskResult:
    """Build a conservative SFX glyph mask from local crop strokes."""

    evidence = {
        "kind": "sfx_glyph_mask",
        "raw_mask_pixels": 0,
        "expanded_mask_pixels": 0,
        "bbox_fill_ratio": 0.0,
        "component_count": 0,
    }
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0 or image_rgb.ndim < 2:
        return _reject(evidence, "empty_image")
    if not isinstance(layer, dict):
        return _reject(evidence, "invalid_layer")
    if not _has_sfx_evidence(layer):
        return _reject(evidence, "missing_hangul_sfx_evidence")

    height, width = image_rgb.shape[:2]
    bbox = _normalize_bbox(layer.get("text_pixel_bbox") or layer.get("bbox"), width, height)
    if bbox is None:
        return _reject(evidence, "invalid_bbox")
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return _reject(evidence, "empty_image")
    bbox_area = max(1, (x2 - x1) * (y2 - y1))

    raw = _extract_local_strokes(crop)
    color_raw = _extract_colored_strokes(crop)
    raw = _merge_layer_mask(raw, layer, crop.shape[:2])
    pre_filter_pixels = int(np.count_nonzero(raw))
    pre_filter_ratio = pre_filter_pixels / float(bbox_area)
    if pre_filter_ratio >= 0.52:
        color_pixels = int(np.count_nonzero(color_raw))
        color_fill_ratio = color_pixels / float(bbox_area)
        if color_pixels > 0 and 0.01 <= color_fill_ratio <= 0.42:
            raw = color_raw
            evidence["mask_source"] = "color_chroma"
            evidence["pre_filter_bbox_fill_ratio"] = round(float(pre_filter_ratio), 6)
        else:
            evidence["raw_mask_pixels"] = pre_filter_pixels
            evidence["bbox_fill_ratio"] = round(float(pre_filter_ratio), 6)
            return _reject(evidence, "density_too_high")
    raw, component_count = _filter_components(raw)
    raw_pixels = int(np.count_nonzero(raw))
    evidence["raw_mask_pixels"] = raw_pixels
    evidence["component_count"] = int(component_count)
    if raw_pixels <= 0:
        return _reject(evidence, "raw_mask_empty")

    raw_fill_ratio = raw_pixels / float(bbox_area)
    if raw_fill_ratio >= 0.52:
        evidence["bbox_fill_ratio"] = round(float(raw_fill_ratio), 6)
        return _reject(evidence, "density_too_high")
    if _touches_most_crop_border(raw):
        evidence["bbox_fill_ratio"] = round(float(raw_fill_ratio), 6)
        return _reject(evidence, "touches_most_crop_border")

    expanded = expand_cjk_glyph_mask_for_inpaint(raw, min_radius=1, max_radius=2, component_ratio=0.06)
    expanded = _clip_to_crop(expanded, raw.shape)
    expanded_pixels = int(np.count_nonzero(expanded))
    expanded_fill_ratio = expanded_pixels / float(bbox_area)
    evidence["expanded_mask_pixels"] = expanded_pixels
    evidence["bbox_fill_ratio"] = round(float(expanded_fill_ratio), 6)
    if expanded_pixels <= 0:
        return _reject(evidence, "expanded_mask_empty")
    if expanded_fill_ratio >= 0.62:
        return _reject(evidence, "area_near_full_bbox")
    if _touches_most_crop_border(expanded):
        return _reject(evidence, "touches_most_crop_border")

    page_mask = np.zeros((height, width), dtype=np.uint8)
    page_mask[y1:y2, x1:x2] = expanded.astype(np.uint8)
    return SfxGlyphMaskResult(mask=page_mask, evidence=evidence)


def _reject(evidence: dict[str, Any], reason: str) -> SfxGlyphMaskResult:
    evidence["reject_reason"] = reason
    return SfxGlyphMaskResult(mask=None, evidence=evidence)


def _has_sfx_evidence(layer: dict) -> bool:
    route_action = str(layer.get("route_action") or "").strip().lower()
    content_class = str(layer.get("content_class") or "").strip().lower()
    tipo = str(layer.get("tipo") or "").strip().lower()
    script = str(layer.get("script") or "").strip().lower()
    text = " ".join(
        str(layer.get(key) or "")
        for key in ("text", "source_text", "original", "raw_ocr", "translated")
    )
    sfx = layer.get("sfx")
    if isinstance(sfx, dict):
        text = f"{text} {sfx.get('source_text') or ''} {sfx.get('adapted_text') or ''}"
    return (
        route_action == SFX_ROUTE_ACTION
        or content_class in SFX_CLASSES
        or tipo in SFX_CLASSES
        or script == "hangul"
        or bool(HANGUL_RE.search(text))
    )


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


def _extract_local_strokes(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.astype(np.uint8)
    if gray.size == 0:
        return np.zeros(gray.shape[:2], dtype=np.uint8)
    blur_size = max(7, min(31, (min(gray.shape[:2]) // 2) | 1))
    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0).astype(np.float32)
    gray_f = gray.astype(np.float32)
    median = float(np.median(gray_f))
    dark = (background - gray_f) >= max(18.0, float(np.std(gray_f)) * 0.45)
    light = (gray_f - background) >= max(18.0, float(np.std(gray_f)) * 0.45)
    if median >= 150.0:
        extreme = gray_f <= max(96.0, median - 38.0)
    else:
        extreme = gray_f >= min(230.0, median + 38.0)
    raw = ((dark | light | extreme)).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return raw.astype(np.uint8)


def _extract_colored_strokes(crop: np.ndarray) -> np.ndarray:
    if crop.ndim != 3 or crop.size == 0:
        return np.zeros(crop.shape[:2], dtype=np.uint8)
    crop_u8 = crop.astype(np.uint8)
    hsv = cv2.cvtColor(crop_u8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    rgb_f = crop_u8.astype(np.float32)
    chroma = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    colored = (saturation >= 35.0) & (chroma >= 22.0) & (value <= 235.0)
    raw = colored.astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return raw.astype(np.uint8)


def _merge_layer_mask(raw: np.ndarray, layer: dict, shape: tuple[int, int]) -> np.ndarray:
    mask = None
    for key in ("mask", "segmentation_mask", "text_mask", "probability_map"):
        candidate = layer.get(key)
        if isinstance(candidate, np.ndarray) and candidate.size:
            mask = candidate
            break
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return raw
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        return raw
    return np.maximum(raw, mask)


def _filter_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if not np.any(mask):
        return mask.astype(np.uint8), 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    kept = np.zeros_like(mask, dtype=np.uint8)
    kept_count = 0
    image_area = max(1, mask.shape[0] * mask.shape[1])
    for label in range(1, count):
        x, y, w_box, h_box, area = [int(v) for v in stats[label].tolist()]
        if area < 6 or area > int(image_area * 0.42):
            continue
        bbox_area = max(1, w_box * h_box)
        fill = area / float(bbox_area)
        aspect = max(w_box, h_box) / float(max(1, min(w_box, h_box)))
        if fill >= 0.94 and area > 96:
            continue
        if aspect >= 22.0 and fill < 0.30:
            continue
        kept[labels == label] = 255
        kept_count += 1
    return kept, kept_count


def _touches_most_crop_border(mask: np.ndarray) -> bool:
    if mask.size == 0 or mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    border_hits = 0
    border_hits += int(np.count_nonzero(mask[0, :] > 0) >= mask.shape[1] * 0.45)
    border_hits += int(np.count_nonzero(mask[-1, :] > 0) >= mask.shape[1] * 0.45)
    border_hits += int(np.count_nonzero(mask[:, 0] > 0) >= mask.shape[0] * 0.45)
    border_hits += int(np.count_nonzero(mask[:, -1] > 0) >= mask.shape[0] * 0.45)
    return border_hits >= 3


def _clip_to_crop(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape:
        return np.where(mask > 0, 255, 0).astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
