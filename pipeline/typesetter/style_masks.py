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


def build_mask_backed_typographic_layers(
    image_rgb: np.ndarray,
    glyph_mask: np.ndarray,
    context_mask: np.ndarray,
) -> dict[str, Any]:
    """Decompose typography using an authoritative owner glyph mask.

    Unlike :func:`build_typographic_mask_layers`, this path never discovers
    text from the surrounding crop.  Card art and coloured panels therefore
    cannot become glyph fill evidence.
    """

    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError("image_rgb must be a non-empty RGB image")
    rgb = np.ascontiguousarray(image_rgb[:, :, :3], dtype=np.uint8)
    shape = rgb.shape[:2]
    raw_glyph = np.asarray(glyph_mask)
    raw_context = np.asarray(context_mask)
    if raw_glyph.shape != shape or raw_context.shape != shape:
        raise ValueError("glyph and context masks must match image dimensions")
    glyph = np.where(raw_glyph > 0, 255, 0).astype(np.uint8)
    context = np.where(raw_context > 0, 255, 0).astype(np.uint8)
    glyph[context == 0] = 0
    if int(np.count_nonzero(glyph)) < 8:
        raise ValueError("glyph mask has insufficient source text evidence")

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(glyph, 8)
    component_heights = sorted(
        int(stats[label, cv2.CC_STAT_HEIGHT])
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= 4
    )
    if not component_heights:
        raise ValueError("glyph mask has no measurable components")
    source_x_height_px = float(np.median(component_heights))
    search_radius = max(3, int(round(source_x_height_px * 0.34)))
    kernel_size = search_radius * 2 + 1
    neighborhood = cv2.dilate(
        glyph,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    )
    far_background = (context > 0) & (neighborhood == 0)
    if int(np.count_nonzero(far_background)) < 12:
        far_background = (context > 0) & (glyph == 0)
    background_rgb = np.median(rgb[far_background].astype(np.float32), axis=0)
    background_distance = np.linalg.norm(
        rgb.astype(np.float32) - background_rgb,
        axis=2,
    )
    inner_radius = max(2, int(round(source_x_height_px * 0.15)))
    inner_neighborhood = cv2.dilate(
        glyph,
        np.ones((inner_radius * 2 + 1, inner_radius * 2 + 1), dtype=np.uint8),
    )
    ring_search = (inner_neighborhood > 0) & (glyph == 0) & (context > 0)
    stroke_ring = np.where(ring_search & (background_distance >= 24.0), 255, 0).astype(np.uint8)
    stroke_ring = _remove_tiny_components(stroke_ring, minimum_area=4)
    effect_search = (
        (neighborhood > 0)
        & (inner_neighborhood == 0)
        & (context > 0)
    )
    effect_ring = np.where(
        effect_search & (background_distance >= 12.0), 255, 0
    ).astype(np.uint8)
    effect_ring = _remove_tiny_components(effect_ring, minimum_area=6)
    foreground = np.maximum(np.maximum(glyph, stroke_ring), effect_ring)
    background_mask = np.where(
        (context > 0) & (foreground == 0),
        255,
        0,
    ).astype(np.uint8)
    contour = cv2.morphologyEx(glyph, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    normalized_stroke_width = round(
        float(np.count_nonzero(stroke_ring))
        / max(1.0, float(np.count_nonzero(contour)) * source_x_height_px),
        6,
    )
    x, y, width, height = cv2.boundingRect(glyph)
    occupancy = float(np.count_nonzero(glyph)) / max(1, width * height)
    return {
        "background_mask": background_mask,
        "context_mask": context,
        "core_mask": glyph,
        "effect_ring_mask": effect_ring,
        "foreground_mask": foreground,
        "metrics": {
            "background_rgb": [int(round(value)) for value in background_rgb],
            "foreground_pixels": int(np.count_nonzero(foreground)),
            "glyph_bbox": [x, y, x + width, y + height],
            "glyph_occupancy": round(occupancy, 6),
            "normalization_unit": "source_x_height",
            "normalized_stroke_width": normalized_stroke_width,
            "source_x_height_px": round(source_x_height_px, 4),
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
