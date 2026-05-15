from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _normalize_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    normalized = mask
    if normalized.ndim == 3:
        normalized = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if normalized.shape[:2] != (height, width):
        normalized = cv2.resize(normalized.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return (normalized > 0).astype(np.uint8) * 255


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


class ResidualValidationEngine:
    name = "residual_validator"

    def __init__(self, *, quality: str = "normal") -> None:
        self.quality = "ultra" if str(quality).strip().lower() == "ultra" else "normal"

    def validate(
        self,
        original_roi: np.ndarray,
        cleaned_roi: np.ndarray,
        mask: np.ndarray,
        text_entries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del original_roi, text_entries
        if not isinstance(cleaned_roi, np.ndarray) or cleaned_roi.size == 0:
            return self._clean_result(confidence=0.0, reason="empty_image")

        normalized_mask = _normalize_mask(mask, cleaned_roi.shape)
        bbox = _bbox_from_mask(normalized_mask)
        if bbox is None:
            return self._clean_result(confidence=1.0, reason="empty_mask")

        x1, y1, x2, y2 = bbox
        crop = cleaned_roi[y1:y2, x1:x2]
        mask_crop = normalized_mask[y1:y2, x1:x2] > 0
        if crop.size == 0 or not np.any(mask_crop):
            return self._clean_result(confidence=1.0, reason="empty_roi")

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.astype(np.uint8)
        values = gray[mask_crop]
        if values.size < 8:
            return self._clean_result(confidence=0.9, reason="small_mask")

        median = float(np.median(values))
        dark_residual = gray <= max(0.0, median - 42.0)
        bright_residual = gray >= min(255.0, median + 46.0)
        residual = (dark_residual | bright_residual) & mask_crop
        if not np.any(residual):
            return self._clean_result(confidence=0.93, reason="low_contrast")

        residual_u8 = residual.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(residual_u8, connectivity=8)
        mask_area = int(np.count_nonzero(mask_crop))
        min_area = max(4, int(mask_area * 0.015))
        residual_bboxes: list[list[int]] = []
        residual_area = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            lx = int(stats[label, cv2.CC_STAT_LEFT])
            ly = int(stats[label, cv2.CC_STAT_TOP])
            lw = int(stats[label, cv2.CC_STAT_WIDTH])
            lh = int(stats[label, cv2.CC_STAT_HEIGHT])
            residual_bboxes.append([x1 + lx, y1 + ly, x1 + lx + lw, y1 + ly + lh])
            residual_area += area

        if not residual_bboxes:
            return self._clean_result(confidence=0.88, reason="component_filtered")

        area_ratio = residual_area / float(max(1, mask_area))
        confidence = round(max(0.0, min(0.99, 0.72 + area_ratio)), 3)
        return {
            "status": "residual",
            "residual_bboxes": residual_bboxes,
            "retry_recommended": self.quality == "ultra",
            "confidence": confidence,
            "residual_area": int(residual_area),
            "mask_area": int(mask_area),
            "engine": self.name,
        }

    def _clean_result(self, *, confidence: float, reason: str) -> dict[str, Any]:
        return {
            "status": "clean",
            "residual_bboxes": [],
            "retry_recommended": False,
            "confidence": round(float(confidence), 3),
            "reason": reason,
            "engine": self.name,
        }
