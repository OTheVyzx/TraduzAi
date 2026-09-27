"""Build page-space writing authority from authenticated Vision evidence."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def polygon_mask(
    shape: Sequence[int],
    polygon_page: Sequence[Sequence[int]],
) -> np.ndarray:
    """Rasterize one explicit page-space polygon as a canonical binary mask."""

    if len(shape) != 2:
        raise ValueError("page mask shape must contain height and width")
    height, width = (int(value) for value in shape)
    if height <= 0 or width <= 0:
        raise ValueError("page mask shape must be positive")
    points = np.asarray(polygon_page, dtype=np.int32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("page-space polygon must contain at least three x/y points")
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= height)
    ):
        raise ValueError("page-space polygon lies outside the page")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return np.ascontiguousarray(mask)


def preserved_art_mask(
    original_rgb: np.ndarray,
    clean_rgb: np.ndarray,
    scope_mask: np.ndarray,
    *,
    luma_max: int = 245,
) -> np.ndarray:
    """Protect directly observed dark pixels preserved across original and clean.

    Exact RGB equality makes this producer deliberately conservative: removed
    source glyphs cannot be relabelled as artwork, while artwork copied through
    cleaning remains protected. The explicit scope prevents unrelated page art
    from entering an owner's mask.
    """

    original = np.asarray(original_rgb)
    clean = np.asarray(clean_rgb)
    scope = np.asarray(scope_mask)
    if (
        original.ndim != 3
        or original.shape[2] != 3
        or original.dtype != np.uint8
        or clean.shape != original.shape
        or clean.dtype != np.uint8
    ):
        raise ValueError("original and clean must be equal-sized RGB uint8 arrays")
    if scope.ndim != 2 or scope.shape != original.shape[:2]:
        raise ValueError("scope mask must match the page")
    if not 0 <= int(luma_max) <= 255:
        raise ValueError("luma_max must be between 0 and 255")

    # RGB integer luma, rounded down, avoids platform-dependent float output.
    luma = (
        clean[:, :, 0].astype(np.uint32) * 299
        + clean[:, :, 1].astype(np.uint32) * 587
        + clean[:, :, 2].astype(np.uint32) * 114
    ) // 1000
    unchanged = np.all(original == clean, axis=2)
    protected = (scope > 0) & unchanged & (luma <= int(luma_max))
    return np.ascontiguousarray(np.where(protected, 255, 0), dtype=np.uint8)
