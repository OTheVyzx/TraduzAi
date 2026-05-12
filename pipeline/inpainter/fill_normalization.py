from __future__ import annotations

import cv2
import numpy as np


def _looks_like_flat_white_sample(pixels: np.ndarray) -> bool:
    if pixels.size == 0:
        return False
    gray = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).reshape(-1)
    if float(np.percentile(gray, 50)) < 220.0:
        return False
    if float(np.std(pixels.astype(np.float32), axis=0).max()) > 10.0:
        return False
    return True


def normalize_white_balloon_fill(
    image_rgb: np.ndarray,
    target_mask: np.ndarray,
    block: dict | None = None,
) -> np.ndarray:
    """Normalize small inpaint patches inside flat white balloons.

    The function intentionally acts only when the surrounding pixels are
    already flat and white; textured/narration regions are left untouched.
    """

    if image_rgb.size == 0 or target_mask.size == 0 or not np.any(target_mask):
        return image_rgb
    if target_mask.shape[:2] != image_rgb.shape[:2]:
        return image_rgb

    block = block or {}
    balloon_type = str(block.get("balloon_type") or "").strip().lower()
    if balloon_type and balloon_type != "white":
        return image_rgb

    mask = (target_mask > 0).astype(np.uint8) * 255
    sample_ring = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    sample_ring = cv2.subtract(sample_ring, mask)
    if not np.any(sample_ring):
        return image_rgb

    sample = image_rgb[sample_ring > 0]
    if not _looks_like_flat_white_sample(sample):
        return image_rgb

    fill = np.median(sample.astype(np.float32), axis=0).clip(0, 255).astype(np.uint8)
    result = image_rgb.copy()
    result[mask > 0] = fill
    return result
