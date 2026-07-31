"""Shadow typographic-mask decomposition for source-style analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _background_rgb(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    border_width = max(1, min(4, height // 8, width // 8))
    border = np.concatenate(
        (
            image_rgb[:border_width].reshape(-1, 3),
            image_rgb[-border_width:].reshape(-1, 3),
            image_rgb[:, :border_width].reshape(-1, 3),
            image_rgb[:, -border_width:].reshape(-1, 3),
        ),
        axis=0,
    )
    return np.median(border.astype(np.float32), axis=0)


def _remove_tiny_components(mask: np.ndarray, *, minimum_area: int) -> np.ndarray:
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            kept[labels == label] = 255
    return kept


def build_typographic_mask_layers(image_rgb: np.ndarray) -> dict[str, Any]:
    """Return explicit foreground/core/stroke-ring/background masks for shadow analysis."""
    if image_rgb is None or image_rgb.ndim < 3 or image_rgb.shape[2] < 3:
        raise ValueError("image_rgb must be a non-empty RGB image")
    rgb = image_rgb[:, :, :3].astype(np.uint8, copy=False)
    background = _background_rgb(rgb)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    threshold = max(26.0, min(72.0, float(np.percentile(distance, 88)) * 0.55))
    foreground = _remove_tiny_components((distance >= threshold).astype(np.uint8) * 255, minimum_area=6)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(foreground, kernel, iterations=1)
    core = eroded if int(np.count_nonzero(eroded)) >= 16 else foreground.copy()
    stroke_ring = cv2.bitwise_and(foreground, cv2.bitwise_not(core))
    background_mask = cv2.bitwise_not(foreground)
    contour = cv2.morphologyEx(core, cv2.MORPH_GRADIENT, kernel)
    normalized_stroke_width = round(
        float(np.count_nonzero(stroke_ring)) / max(1, int(np.count_nonzero(contour))),
        4,
    )
    return {
        "background_mask": background_mask,
        "core_mask": core,
        "foreground_mask": foreground,
        "metrics": {
            "background_rgb": [int(round(value)) for value in background],
            "foreground_pixels": int(np.count_nonzero(foreground)),
            "normalized_stroke_width": normalized_stroke_width,
            "threshold": round(threshold, 4),
        },
        "stroke_ring_mask": stroke_ring,
    }


def write_typographic_mask_debug(output_root: Path, case_id: str, layers: dict[str, Any]) -> dict[str, Path]:
    """Write only explicit debug artifacts for a precomputed shadow mask decomposition."""
    root = Path(output_root) / str(case_id)
    root.mkdir(parents=True, exist_ok=False)
    outputs = {"metrics": root / "metrics.json"}
    outputs["metrics"].write_text(
        json.dumps(layers.get("metrics") or {}, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in ("foreground_mask", "core_mask", "stroke_ring_mask", "background_mask"):
        target = root / f"{name}.png"
        mask = layers.get(name)
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise ValueError(f"missing 2D mask: {name}")
        if not cv2.imwrite(str(target), mask):
            raise RuntimeError(f"failed to write mask debug: {target}")
        outputs[name] = target
    return outputs
