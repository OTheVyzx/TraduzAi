"""Owned component filter inspired by NotAnotherBubbleCleaner.

The external project is not a production dependency here. This module keeps the
contract local: dark glyph components are accepted only when OCR geometry and a
real BubbleMask interior agree.
"""

from __future__ import annotations

import cv2
import numpy as np


def _as_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.shape[:2] != shape:
        raise ValueError("mask shape must match image")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _threshold_dark_components(
    image_rgb: np.ndarray,
    bubble_mask: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    gray = (
        cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if image_rgb.ndim == 3
        else image_rgb.astype(np.uint8)
    )
    support_bool = support_mask > 0
    sample_bool = (bubble_mask > 0) & support_bool
    sample = gray[sample_bool]
    if sample.size == 0:
        return np.zeros_like(bubble_mask, dtype=np.uint8), 0
    cutoff = min(210, max(72, int(np.percentile(sample, 35)) - 8))
    glyphs = np.where((gray <= cutoff) & support_bool, 255, 0).astype(np.uint8)
    return glyphs, int(cutoff)


def build_notanother_text_mask(
    image_rgb: np.ndarray,
    bubble_mask: np.ndarray,
    support_mask: np.ndarray,
    *,
    min_component_area: int = 4,
    min_bubble_overlap: float = 0.80,
    min_support_overlap: float = 0.45,
    erode_outline_px: int = 2,
) -> tuple[np.ndarray, dict]:
    """Build a conservative glyph mask for one real BubbleMask."""

    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        raise ValueError("image_rgb is required")
    shape = image_rgb.shape[:2]
    bubble = _as_binary_mask(bubble_mask, shape)
    support = _as_binary_mask(support_mask, shape)

    safe_bubble = bubble
    if erode_outline_px > 0:
        kernel_size = int(erode_outline_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(bubble, kernel, iterations=1)
        if np.any(eroded):
            safe_bubble = eroded.astype(np.uint8)

    glyphs, threshold = _threshold_dark_components(image_rgb, bubble, support)
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (glyphs > 0).astype(np.uint8),
        connectivity=8,
    )
    accepted = np.zeros_like(glyphs, dtype=np.uint8)
    debug = {
        "component_total": max(0, int(labels_count) - 1),
        "component_accepted": 0,
        "component_rejected_small": 0,
        "component_rejected_outside_bubble": 0,
        "component_rejected_low_overlap": 0,
        "threshold": int(threshold),
        "safe_bubble_pixels": int(np.count_nonzero(safe_bubble)),
        "support_mask_pixels": int(np.count_nonzero(support)),
        "raw_glyph_pixels": int(np.count_nonzero(glyphs)),
        "hole_fill_pixels": 0,
        "final_mask_pixels": 0,
    }

    safe_bool = safe_bubble > 0
    support_bool = support > 0
    for label in range(1, int(labels_count)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            debug["component_rejected_small"] += 1
            continue
        cx, cy = centroids[label]
        ix, iy = int(round(cx)), int(round(cy))
        if iy < 0 or iy >= shape[0] or ix < 0 or ix >= shape[1] or not safe_bool[iy, ix]:
            debug["component_rejected_outside_bubble"] += 1
            continue
        component = labels == label
        bubble_overlap = np.count_nonzero(component & safe_bool) / float(max(1, area))
        support_overlap = np.count_nonzero(component & support_bool) / float(max(1, area))
        if bubble_overlap < float(min_bubble_overlap) or support_overlap < float(min_support_overlap):
            debug["component_rejected_low_overlap"] += 1
            continue
        accepted[component] = 255
        debug["component_accepted"] += 1

    if np.any(accepted):
        before_fill = int(np.count_nonzero(accepted))
        flood = accepted.copy()
        height, width = flood.shape[:2]
        flood_canvas = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_canvas, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        accepted = cv2.bitwise_or(accepted, holes)
        accepted[(safe_bubble == 0) | (support == 0)] = 0
        debug["hole_fill_pixels"] = max(0, int(np.count_nonzero(accepted)) - before_fill)

    debug["final_mask_pixels"] = int(np.count_nonzero(accepted))
    return accepted.astype(np.uint8), debug
