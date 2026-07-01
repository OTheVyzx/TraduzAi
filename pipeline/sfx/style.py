"""Style extraction for manhwa SFX layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from typesetter.style_extractor import extract_text_style_evidence


@dataclass(frozen=True)
class SfxStyle:
    fill_color: str
    stroke_color: str
    stroke_width_px: int
    glow_color: str
    glow_width_px: int
    rotation_deg: float
    scale_x: float
    scale_y: float
    confidence: float
    qa_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_manhwa_sfx_style(
    crop_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    layer: dict[str, Any] | None = None,
    font_detector: object | None = None,
) -> SfxStyle:
    """Extract renderable style metadata from a SFX crop and optional glyph mask."""

    if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.ndim < 3:
        return _empty_style(["sfx_style_empty_crop"])

    rgb_crop = crop_rgb[:, :, :3].astype(np.uint8)
    glyph_mask = _coerce_mask(mask, rgb_crop.shape[:2])
    if glyph_mask is None and isinstance(layer, dict):
        glyph_mask = _coerce_mask(layer.get("mask"), rgb_crop.shape[:2])
    qa_flags: list[str] = []
    if glyph_mask is None or not np.any(glyph_mask):
        qa_flags.append("sfx_style_missing_mask")

    text_evidence = extract_text_style_evidence(rgb_crop, font_detector=font_detector)
    fill_color, fill_confidence = _fill_color_from_mask(rgb_crop, glyph_mask)
    if not fill_color:
        fill_color = text_evidence.text_color
        fill_confidence = text_evidence.text_color_confidence
    if not fill_color:
        fill_color = "#000000"
        qa_flags.append("sfx_style_low_confidence")

    stroke_color, stroke_width, stroke_confidence = _stroke_from_mask(rgb_crop, glyph_mask)
    if not stroke_color:
        stroke_color = text_evidence.stroke_color
        stroke_width = int(text_evidence.stroke_width_px)
        stroke_confidence = float(text_evidence.stroke_confidence)

    glow_color, glow_width, glow_confidence = _glow_from_mask(rgb_crop, glyph_mask)
    if text_evidence.glow:
        glow_color = glow_color or fill_color
        glow_width = max(glow_width, 8)
        glow_confidence = max(glow_confidence, text_evidence.glow_confidence)

    rotation_deg, scale_x, scale_y, geometry_confidence = _geometry_from_mask(glyph_mask)
    if geometry_confidence < 0.35:
        qa_flags.append("sfx_style_geometry_low_confidence")

    confidence = float(
        np.clip(
            0.45 * fill_confidence
            + 0.25 * max(stroke_confidence, glow_confidence)
            + 0.30 * geometry_confidence,
            0.0,
            1.0,
        )
    )
    if confidence < 0.55 and "sfx_style_low_confidence" not in qa_flags:
        qa_flags.append("sfx_style_low_confidence")

    return SfxStyle(
        fill_color=fill_color,
        stroke_color=stroke_color,
        stroke_width_px=max(0, int(stroke_width)),
        glow_color=glow_color,
        glow_width_px=glow_width,
        rotation_deg=round(rotation_deg, 3),
        scale_x=round(scale_x, 3),
        scale_y=round(scale_y, 3),
        confidence=round(confidence, 3),
        qa_flags=qa_flags,
    )


def _coerce_mask(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(value, np.ndarray) or value.size == 0:
        return None
    mask = value[:, :, 0] if value.ndim == 3 else value
    if mask.ndim != 2:
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    return binary if np.any(binary) else None


def _fill_color_from_mask(crop_rgb: np.ndarray, mask: np.ndarray | None) -> tuple[str, float]:
    if mask is None or not np.any(mask):
        return "", 0.0
    fill_mask = mask > 0
    if int(np.count_nonzero(fill_mask)) >= 20:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
        if int(np.count_nonzero(eroded)) >= 8:
            fill_mask = eroded
    pixels = crop_rgb[fill_mask]
    if pixels.size == 0:
        return "", 0.0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    masked_gray = gray[fill_mask]
    high = masked_gray >= np.percentile(masked_gray, 70)
    low = masked_gray <= np.percentile(masked_gray, 30)
    selected = pixels
    if np.count_nonzero(high) >= 8 and np.count_nonzero(low) >= 8:
        high_pixels = pixels[high]
        low_pixels = pixels[low]
        selected = high_pixels if np.median(high_pixels) > np.median(low_pixels) else low_pixels
    rgb = tuple(int(round(float(v))) for v in np.median(selected, axis=0))
    confidence = min(1.0, max(0.45, float(selected.shape[0]) / max(1.0, float(pixels.shape[0]))))
    return _hex_color(rgb), confidence


def _stroke_from_mask(crop_rgb: np.ndarray, mask: np.ndarray | None) -> tuple[str, int, float]:
    if mask is None or not np.any(mask):
        return "", 0, 0.0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    foreground = mask > 0
    values = gray[foreground]
    if values.size < 16:
        return "", 0, 0.0
    if float(np.percentile(values, 75) - np.percentile(values, 25)) < 18.0:
        return "", 0, 0.0
    dark = foreground & (gray <= np.percentile(values, 25))
    light = foreground & (gray >= np.percentile(values, 75))
    if int(np.count_nonzero(dark)) < 8 or int(np.count_nonzero(light)) < 8:
        return "", 0, 0.0
    stroke_mask = dark if float(np.median(gray[dark])) < float(np.median(gray[light])) else light
    stroke_pixels = crop_rgb[stroke_mask]
    if stroke_pixels.size == 0:
        return "", 0, 0.0
    distances = cv2.distanceTransform(stroke_mask.astype(np.uint8), cv2.DIST_L2, 3)
    nonzero = distances[distances > 0]
    width = int(round(float(np.percentile(nonzero, 90)))) if nonzero.size else 1
    contrast = abs(float(np.median(gray[dark])) - float(np.median(gray[light])))
    rgb = tuple(int(round(float(v))) for v in np.median(stroke_pixels, axis=0))
    return _hex_color(rgb), max(1, min(12, width)), min(1.0, max(0.0, contrast / 160.0))


def _glow_from_mask(crop_rgb: np.ndarray, mask: np.ndarray | None) -> tuple[str, int, float]:
    if mask is None or not np.any(mask):
        return "", 0, 0.0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    outer = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    inner = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    ring = outer & ~(mask > 0)
    near_ring = ring & ~inner
    if int(np.count_nonzero(ring)) < 16:
        return "", 0, 0.0
    ring_gray = gray[ring]
    outside = ~(outer)
    outside_gray = gray[outside] if np.any(outside) else gray.reshape(-1)
    if outside_gray.size == 0:
        return "", 0, 0.0
    ring_luma = float(np.percentile(ring_gray, 80))
    outside_luma = float(np.median(outside_gray))
    if ring_luma - outside_luma < 24.0:
        return "", 0, 0.0
    bright_ring = ring & (gray >= outside_luma + 18.0)
    if int(np.count_nonzero(bright_ring)) < 12:
        bright_ring = near_ring
    pixels = crop_rgb[bright_ring]
    if pixels.size == 0:
        return "", 0, 0.0
    rgb = tuple(int(round(float(v))) for v in np.median(pixels, axis=0))
    confidence = min(1.0, max(0.0, (ring_luma - outside_luma) / 120.0))
    return _hex_color(rgb), 8, confidence


def _geometry_from_mask(mask: np.ndarray | None) -> tuple[float, float, float, float]:
    if mask is None or not np.any(mask):
        return 0.0, 1.0, 1.0, 0.0
    ys, xs = np.where(mask > 0)
    if xs.size < 8:
        return 0.0, 1.0, 1.0, 0.1
    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    principal = eigvecs[:, order[0]]
    rotation = float(np.degrees(np.arctan2(principal[1], principal[0])))
    if rotation > 90.0:
        rotation -= 180.0
    if rotation < -90.0:
        rotation += 180.0
    width = max(1.0, float(xs.max() - xs.min() + 1))
    height = max(1.0, float(ys.max() - ys.min() + 1))
    major = float(np.sqrt(max(0.0, eigvals[order[0]]))) * 2.0
    minor = float(np.sqrt(max(0.0, eigvals[order[1]]))) * 2.0
    scale_x = max(0.1, major / width)
    scale_y = max(0.1, minor / height)
    confidence = min(1.0, max(0.2, xs.size / float(mask.size) * 8.0))
    return rotation, scale_x, scale_y, confidence


def _hex_color(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _empty_style(flags: list[str]) -> SfxStyle:
    return SfxStyle(
        fill_color="#000000",
        stroke_color="",
        stroke_width_px=0,
        glow_color="",
        glow_width_px=0,
        rotation_deg=0.0,
        scale_x=1.0,
        scale_y=1.0,
        confidence=0.0,
        qa_flags=flags,
    )
