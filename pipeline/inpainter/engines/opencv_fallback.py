from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import normalize_mask, normalize_roi_bbox, pixel_lock_composite


class OpenCVFallbackInpaintEngine:
    name = "opencv_fallback"

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        *,
        quality: str,
        roi_bbox: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del metadata
        if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
            return image_rgb
        normalized_mask = normalize_mask(mask, image_rgb.shape)
        if not np.any(normalized_mask):
            return image_rgb.copy()

        bbox = normalize_roi_bbox(roi_bbox, normalized_mask, image_rgb.shape)
        if bbox is None:
            return image_rgb.copy()
        x1, y1, x2, y2 = bbox
        roi = image_rgb[y1:y2, x1:x2]
        roi_mask = normalized_mask[y1:y2, x1:x2]
        if roi.size == 0 or not np.any(roi_mask):
            return image_rgb.copy()

        radius = 5 if str(quality).strip().lower() == "ultra" else 3
        inpainted_roi = cv2.inpaint(roi.astype(np.uint8), roi_mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)
        inpainted = image_rgb.copy()
        inpainted[y1:y2, x1:x2] = inpainted_roi
        return pixel_lock_composite(image_rgb, inpainted, normalized_mask)
