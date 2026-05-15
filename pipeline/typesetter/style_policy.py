"""Style policy for automatic typesetting output.

The automatic pipeline should produce conservative, readable defaults. Manual
editor choices are handled outside this module and must not be normalized here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


CANONICAL_AUTO_FONT = "ComicNeue-Bold.ttf"


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    def channel(value: int) -> float:
        value = max(0, min(255, int(value))) / 255.0
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def auto_text_color_for_background(background_rgb: tuple[int, int, int]) -> str:
    # Conservative for manga/manhwa: if contrast is ambiguous, use black.
    return "#000000" if relative_luminance(background_rgb) >= 0.25 else "#FFFFFF"


def normalize_auto_typesetting_style(
    style: dict | None,
    background_rgb: tuple[int, int, int],
    *,
    force_black_text: bool = False,
) -> dict:
    normalized = dict(style or {})
    normalized["fonte"] = CANONICAL_AUTO_FONT
    normalized["cor"] = "#000000" if force_black_text else auto_text_color_for_background(background_rgb)
    normalized["cor_gradiente"] = []
    normalized["contorno"] = ""
    normalized["contorno_px"] = 0
    normalized["glow"] = False
    normalized["glow_cor"] = ""
    normalized["glow_px"] = 0
    normalized["sombra"] = False
    normalized["sombra_cor"] = ""
    normalized["sombra_offset"] = [0, 0]
    normalized["bold"] = True
    normalized.setdefault("italico", False)
    normalized.setdefault("rotacao", 0)
    normalized.setdefault("alinhamento", "center")
    normalized.setdefault("force_upper", False)
    return normalized


def _coerce_bbox(bbox: Sequence[int | float] | None) -> tuple[int, int, int, int] | None:
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def sample_text_background_rgb(
    image_rgb: np.ndarray,
    bbox: Sequence[int | float] | None,
) -> tuple[int, int, int]:
    if image_rgb is None or image_rgb.ndim < 3 or image_rgb.shape[2] < 3:
        return (255, 255, 255)
    coerced = _coerce_bbox(bbox)
    if coerced is None:
        return (255, 255, 255)

    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = coerced
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return (255, 255, 255)

    box_w = x2 - x1
    box_h = y2 - y1
    margin = max(2, int(round(min(box_w, box_h) * 0.08)))
    if box_w > margin * 2 + 2 and box_h > margin * 2 + 2:
        x1 += margin
        x2 -= margin
        y1 += margin
        y2 -= margin

    crop = image_rgb[y1:y2, x1:x2, :3]
    if crop.size == 0:
        return (255, 255, 255)

    pixels = crop.reshape(-1, 3).astype(np.float32)
    if pixels.shape[0] >= 32:
        luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
        low = np.percentile(luminance, 10)
        high = np.percentile(luminance, 95)
        filtered = pixels[(luminance >= low) & (luminance <= high)]
        if filtered.shape[0] >= max(8, pixels.shape[0] // 8):
            pixels = filtered

    rgb = np.median(pixels, axis=0)
    return tuple(int(max(0, min(255, round(float(value))))) for value in rgb)  # type: ignore[return-value]
