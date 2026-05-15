from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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
        return self.models_dir / "inpaint" / "lama_onnx" / "model.onnx"

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
        if model_path is None or not model_path.exists():
            self.last_fallback_reason = "model_missing"
            if metadata is not None:
                metadata["inpaint_fallback"] = self.last_fallback_reason
            return self.fallback.inpaint(image_rgb, mask, quality=quality, roi_bbox=roi_bbox, metadata=metadata)

        # The ONNX session is intentionally not guessed here. Until model IO is
        # pinned by a fixture, preserve correctness with the deterministic fallback.
        self.last_fallback_reason = "onnx_adapter_pending"
        if metadata is not None:
            metadata["inpaint_fallback"] = self.last_fallback_reason
        return self.fallback.inpaint(image_rgb, mask, quality=quality, roi_bbox=roi_bbox, metadata=metadata)
