from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TextMaskRefinement:
    mask: np.ndarray
    source: str
    confidence: float = 0.0


class HiSamTextRefiner:
    def __init__(self, model: Any | None = None) -> None:
        self._model = model

    def enabled(self) -> bool:
        return os.getenv("TRADUZAI_HISAM_TEXT_REFINE", "0").strip().lower() in {"1", "true", "yes", "on"}

    def refine(
        self,
        image_rgb: np.ndarray,
        roi_bbox: list[int],
        seed_mask: np.ndarray,
        *,
        evidence: list[Any] | None = None,
    ) -> TextMaskRefinement | None:
        del evidence
        if not self.enabled() or self._model is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in roi_bbox[:4]]
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if callable(self._model):
            local = self._model(crop, seed_mask[y1:y2, x1:x2])
        elif hasattr(self._model, "refine"):
            local = self._model.refine(crop, seed_mask[y1:y2, x1:x2])
        else:
            return None
        if not isinstance(local, np.ndarray) or local.size == 0:
            return None
        if local.shape[:2] != crop.shape[:2]:
            return None
        full = np.zeros(seed_mask.shape[:2], dtype=np.uint8)
        full[y1:y2, x1:x2] = (local > 0).astype(np.uint8) * 255
        return TextMaskRefinement(mask=full, source="hi-sam", confidence=1.0)
