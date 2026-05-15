from __future__ import annotations

from typing import Any, Protocol

import cv2
import numpy as np


class InpaintEngine(Protocol):
    name: str

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        *,
        quality: str,
        roi_bbox: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        ...


def normalize_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    normalized = mask
    if normalized.ndim == 3:
        normalized = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if normalized.shape[:2] != (height, width):
        normalized = cv2.resize(normalized.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return ((normalized > 0).astype(np.uint8) * 255)


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def normalize_roi_bbox(roi_bbox: list[int] | None, mask: np.ndarray, shape: tuple[int, ...]) -> list[int] | None:
    height, width = int(shape[0]), int(shape[1])
    raw_bbox = roi_bbox or mask_bbox(mask)
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox[:4]]
    except Exception:
        return None
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def pixel_lock_composite(original_rgb: np.ndarray, inpainted_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    locked_mask = normalize_mask(mask, original_rgb.shape)
    result = original_rgb.copy()
    if inpainted_rgb.shape != original_rgb.shape:
        inpainted_rgb = cv2.resize(inpainted_rgb, (original_rgb.shape[1], original_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    result[locked_mask > 0] = inpainted_rgb[locked_mask > 0]
    return result
