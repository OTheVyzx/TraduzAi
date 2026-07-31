"""Mask-backed colour evidence for the shadow style engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _hex_color(rgb: np.ndarray) -> str:
    red, green, blue = (int(round(float(value))) for value in rgb[:3])
    return f"#{red:02X}{green:02X}{blue:02X}"


def _color_spread(pixels: np.ndarray, center: np.ndarray) -> float:
    if pixels.size == 0:
        return 255.0
    return float(np.median(np.linalg.norm(pixels.astype(np.float32) - center.astype(np.float32), axis=1)))


def _confidence(*, pixels: np.ndarray, color: np.ndarray, reference: np.ndarray) -> float:
    if len(pixels) < 12:
        return 0.0
    contrast = float(np.linalg.norm(color.astype(np.float32) - reference.astype(np.float32)))
    spread = _color_spread(pixels, color)
    coverage = min(1.0, len(pixels) / 160.0)
    return round(max(0.0, min(1.0, coverage * min(1.0, contrast / 72.0) * (1.0 - min(0.8, spread / 96.0)))), 4)


@dataclass(frozen=True)
class MaskedColorEvidence:
    fill_color: str
    fill_confidence: float
    stroke_color: str
    stroke_confidence: float
    stroke_detected: bool
    stroke_abstention_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_masked_color_evidence(image_rgb: np.ndarray, mask_layers: dict[str, Any]) -> MaskedColorEvidence:
    """Measure fill and outline colours without assigning a runtime style.

    A stroke is only observed when the ring differs materially from both the
    glyph core and the local background. Plain text therefore abstains instead
    of being misclassified as text with an outline.
    """
    if image_rgb is None or image_rgb.ndim < 3 or image_rgb.shape[2] < 3:
        raise ValueError("image_rgb must be a non-empty RGB image")
    rgb = image_rgb[:, :, :3].astype(np.uint8, copy=False)
    core_mask = np.asarray(mask_layers.get("core_mask")) > 0
    ring_mask = np.asarray(mask_layers.get("stroke_ring_mask")) > 0
    background_mask = np.asarray(mask_layers.get("background_mask")) > 0
    if core_mask.shape != rgb.shape[:2] or ring_mask.shape != rgb.shape[:2] or background_mask.shape != rgb.shape[:2]:
        raise ValueError("style masks must match image dimensions")

    core_pixels = rgb[core_mask]
    ring_pixels = rgb[ring_mask]
    background_pixels = rgb[background_mask]
    if len(core_pixels) < 12 or len(background_pixels) < 12:
        return MaskedColorEvidence("", 0.0, "", 0.0, False, "insufficient_mask_pixels")

    fill_rgb = np.median(core_pixels.astype(np.float32), axis=0)
    background_rgb = np.median(background_pixels.astype(np.float32), axis=0)
    fill_confidence = _confidence(pixels=core_pixels, color=fill_rgb, reference=background_rgb)
    if len(ring_pixels) < 12:
        return MaskedColorEvidence(
            _hex_color(fill_rgb),
            fill_confidence,
            "",
            0.0,
            False,
            "insufficient_stroke_ring_pixels",
        )

    if len(ring_pixels) < len(core_pixels) * 0.9:
        return MaskedColorEvidence(
            _hex_color(fill_rgb),
            fill_confidence,
            "",
            0.0,
            False,
            "stroke_ring_not_dominant",
        )

    stroke_rgb = np.median(ring_pixels.astype(np.float32), axis=0)
    core_delta = float(np.linalg.norm(stroke_rgb - fill_rgb))
    background_delta = float(np.linalg.norm(stroke_rgb - background_rgb))
    if core_delta < 28.0:
        return MaskedColorEvidence(_hex_color(fill_rgb), fill_confidence, "", 0.0, False, "stroke_ring_matches_fill")
    if background_delta < 20.0:
        return MaskedColorEvidence(_hex_color(fill_rgb), fill_confidence, "", 0.0, False, "stroke_ring_matches_background")
    stroke_confidence = _confidence(pixels=ring_pixels, color=stroke_rgb, reference=fill_rgb)
    if stroke_confidence < 0.2:
        return MaskedColorEvidence(_hex_color(fill_rgb), fill_confidence, "", stroke_confidence, False, "low_stroke_confidence")
    return MaskedColorEvidence(
        _hex_color(fill_rgb),
        fill_confidence,
        _hex_color(stroke_rgb),
        stroke_confidence,
        True,
        "",
    )
