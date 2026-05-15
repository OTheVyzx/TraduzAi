from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import time

from .base import mask_bbox, normalize_mask, pixel_lock_composite
from .lama_onnx import LamaOnnxInpaintEngine


class CompositeBandInpaintEngine:
    name = "lama_onnx_composite"

    def __init__(
        self,
        *,
        quality: str = "normal",
        models_dir: str | Path | None = None,
        validator: Any | None = None,
    ) -> None:
        self.quality = "ultra" if str(quality).strip().lower() == "ultra" else "normal"
        self.primary = LamaOnnxInpaintEngine(models_dir=models_dir)
        self.validator = validator

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray, *, quality: str | None = None) -> np.ndarray:
        active_quality = quality or self.quality
        metadata: dict[str, Any] = {}
        cleaned = self.primary.inpaint(
            image_rgb,
            mask,
            quality=active_quality,
            roi_bbox=mask_bbox(normalize_mask(mask, image_rgb.shape)),
            metadata=metadata,
        )
        return pixel_lock_composite(image_rgb, cleaned, mask)

    def inpaint_band_image(self, band_rgb: np.ndarray, ocr_page: dict) -> np.ndarray:
        if not isinstance(band_rgb, np.ndarray) or band_rgb.size == 0:
            return band_rgb
        vision_blocks = [block for block in list((ocr_page or {}).get("_vision_blocks") or []) if isinstance(block, dict)]
        if not vision_blocks:
            return band_rgb.copy()

        from vision_stack.runtime import vision_blocks_to_mask

        mask = vision_blocks_to_mask(band_rgb.shape, vision_blocks, image_rgb=band_rgb, expand_mask=True)
        if not np.any(mask):
            return band_rgb.copy()
        started = time.perf_counter()
        cleaned = self.inpaint(band_rgb, mask, quality=self.quality)
        validation = self._validate_and_retry(band_rgb, cleaned, mask, vision_blocks)
        if isinstance(ocr_page, dict):
            ocr_page["_inpaint_engine"] = self.name
            ocr_page["_strip_used_real_inpaint"] = True
            ocr_page["_strip_raw_limit_mask_pixels"] = int(np.count_nonzero(mask))
            ocr_page["_t_inpaint_engine_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            if validation is not None:
                cleaned = validation["cleaned"]
                ocr_page["_residual_validation"] = validation["result"]
                if validation.get("retry_result") is not None:
                    ocr_page["_residual_validation_retry"] = validation["retry_result"]
                    if validation["retry_result"].get("status") == "residual":
                        ocr_page["needs_review"] = True
                        ocr_page["reason"] = "residual_text_after_retry"
            fallback_reason = getattr(self.primary, "last_fallback_reason", "")
            if fallback_reason:
                ocr_page["_inpaint_fallback"] = fallback_reason
        return cleaned

    def _validate_and_retry(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        mask: np.ndarray,
        vision_blocks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        validator = self.validator
        if validator is None or not hasattr(validator, "validate"):
            return None
        started = time.perf_counter()
        result = validator.validate(original, cleaned, mask, vision_blocks)
        if isinstance(result, dict):
            result["validation_time_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        payload: dict[str, Any] = {"cleaned": cleaned, "result": result, "retry_result": None}
        if not isinstance(result, dict) or not result.get("retry_recommended"):
            return payload

        expanded = cv2.dilate(
            normalize_mask(mask, original.shape),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        retried = self.inpaint(original, expanded, quality="ultra")
        retry_started = time.perf_counter()
        retry_result = validator.validate(original, retried, expanded, vision_blocks)
        if isinstance(retry_result, dict):
            retry_result["validation_time_ms"] = round((time.perf_counter() - retry_started) * 1000.0, 3)
        payload["retry_result"] = retry_result
        if isinstance(retry_result, dict) and retry_result.get("status") == "clean":
            payload["cleaned"] = retried
        return payload
