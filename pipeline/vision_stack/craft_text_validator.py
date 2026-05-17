from __future__ import annotations

import os

import numpy as np


def craft_validation_enabled() -> bool:
    return os.getenv("TRADUZAI_CRAFT_VALIDATE", "0").strip().lower() in {"1", "true", "yes", "on"}


def measure_craft_coverage(mask: np.ndarray, char_heatmap: np.ndarray, *, threshold: float = 0.45) -> dict[str, int | float]:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if char_heatmap.shape[:2] != mask.shape[:2]:
        return {"heatmap_pixels": 0, "covered_pixels": 0, "undercovered_pixels": 0, "coverage_ratio": 1.0}
    text = char_heatmap > threshold
    heatmap_pixels = int(np.count_nonzero(text))
    covered = int(np.count_nonzero(text & (mask > 0)))
    undercovered = max(0, heatmap_pixels - covered)
    return {
        "heatmap_pixels": heatmap_pixels,
        "covered_pixels": covered,
        "undercovered_pixels": undercovered,
        "coverage_ratio": float(covered / heatmap_pixels) if heatmap_pixels else 1.0,
    }
