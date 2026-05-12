from __future__ import annotations

import cv2
import numpy as np


def cleanup_white_balloon_residuals(
    image_rgb: np.ndarray,
    balloon_mask: np.ndarray,
    protected_mask: np.ndarray | None = None,
    *,
    max_cluster_area: int = 1800,
) -> np.ndarray:
    """Remove small dark residual text clusters inside a white balloon mask."""

    if image_rgb.size == 0 or balloon_mask.size == 0 or not np.any(balloon_mask):
        return image_rgb
    if balloon_mask.shape[:2] != image_rgb.shape[:2]:
        return image_rgb

    mask = (balloon_mask > 0).astype(np.uint8) * 255
    interior = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    if not np.any(interior):
        return image_rgb

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    pixels = gray[interior > 0]
    if pixels.size == 0 or float(np.percentile(pixels, 60)) < 205.0:
        return image_rgb

    threshold = min(205.0, float(np.percentile(pixels, 55)) - 24.0)
    threshold = max(120.0, threshold)
    candidate = ((gray.astype(np.float32) <= threshold) & (interior > 0)).astype(np.uint8) * 255
    if protected_mask is not None and protected_mask.shape[:2] == candidate.shape:
        candidate[protected_mask > 0] = 0

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    cleanup = np.zeros_like(candidate)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 4 or area > max_cluster_area:
            continue
        if comp_w > image_rgb.shape[1] * 0.45 or comp_h > image_rgb.shape[0] * 0.25:
            continue
        cleanup[label_map == label] = 255

    if not np.any(cleanup):
        return image_rgb

    cleanup = cv2.dilate(
        cleanup,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    sample_mask = cv2.bitwise_and(interior, cv2.bitwise_not(cleanup))
    sample = image_rgb[sample_mask > 0]
    if sample.size == 0:
        return image_rgb
    fill = np.median(sample.astype(np.float32), axis=0).clip(0, 255).astype(np.uint8)
    result = image_rgb.copy()
    result[cleanup > 0] = fill
    return result
