from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import normalize_mask, normalize_roi_bbox, pixel_lock_composite
from .opencv_fallback import OpenCVFallbackInpaintEngine


class LamaOnnxInpaintEngine:
    name = "lama_onnx"

    def __init__(self, *, models_dir: str | Path | None = None, fallback: OpenCVFallbackInpaintEngine | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else None
        self.fallback = fallback or OpenCVFallbackInpaintEngine()
        self.last_fallback_reason = ""

    def _model_path(self) -> Path | None:
        if self.models_dir is None:
            return None
        candidates = [
            self.models_dir / "lama-manga-dynamic.onnx",
            self.models_dir / "lama-manga.onnx",
            self.models_dir / "lama_manga_onnx_dynamic" / "lama-manga-dynamic.onnx",
            self.models_dir / "lama_manga_onnx" / "lama-manga.onnx",
            self.models_dir / "inpaint" / "lama_onnx" / "lama-manga-dynamic.onnx",
            self.models_dir / "inpaint" / "lama_onnx" / "lama-manga.onnx",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        *,
        quality: str,
        roi_bbox: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        model_path = self._model_path()
        if model_path is None:
            self.last_fallback_reason = "model_missing"
            if metadata is not None:
                metadata["inpaint_fallback"] = self.last_fallback_reason
            return self.fallback.inpaint(image_rgb, mask, quality=quality, roi_bbox=roi_bbox, metadata=metadata)

        normalized_mask = normalize_mask(mask, image_rgb.shape)
        bbox = normalize_roi_bbox(roi_bbox, normalized_mask, image_rgb.shape)
        if bbox is None or not np.any(normalized_mask):
            return image_rgb.copy()
        x1, y1, x2, y2 = bbox
        crop = image_rgb[y1:y2, x1:x2]
        crop_mask = normalized_mask[y1:y2, x1:x2]
        if crop.size == 0 or not np.any(crop_mask):
            return image_rgb.copy()
        if min(crop.shape[:2]) < 64:
            self.last_fallback_reason = "roi_too_small"
            if metadata is not None:
                metadata["inpaint_fallback"] = self.last_fallback_reason
            return self.fallback.inpaint(image_rgb, normalized_mask, quality=quality, roi_bbox=bbox, metadata=metadata)

        try:
            from inpainter.lama_onnx import get_lama_session, inpaint_region_with_lama

            session = get_lama_session(model_path.parent)
            inpainted_crop = inpaint_region_with_lama(session, crop, crop_mask)
            inpainted = image_rgb.copy()
            if inpainted_crop.shape[:2] != crop.shape[:2]:
                inpainted_crop = cv2.resize(inpainted_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
            inpainted[y1:y2, x1:x2] = inpainted_crop
            self.last_fallback_reason = ""
            if metadata is not None:
                metadata["inpaint_model_path"] = str(model_path)
            return pixel_lock_composite(image_rgb, inpainted, normalized_mask)
        except Exception as exc:
            self.last_fallback_reason = f"lama_failed:{exc.__class__.__name__}"
            if metadata is not None:
                metadata["inpaint_fallback"] = self.last_fallback_reason
            return self.fallback.inpaint(image_rgb, normalized_mask, quality=quality, roi_bbox=bbox, metadata=metadata)
