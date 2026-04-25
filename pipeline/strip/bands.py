"""Agrupamento de balões em bandas horizontais full-width."""

from __future__ import annotations

import numpy as np

from strip.types import Balloon, Band, VerticalStrip


def group_balloons_into_bands(
    balloons: list[Balloon],
    gap_threshold: int = 64,
    margin: int = 16,
) -> list[Band]:
    """Agrupa balões em bandas horizontais."""
    if not balloons:
        return []

    sorted_balloons = sorted(balloons, key=lambda b: (b.strip_bbox.y1, b.strip_bbox.x1))

    bands: list[Band] = []
    current_balloons: list[Balloon] = [sorted_balloons[0]]
    current_y_bottom = sorted_balloons[0].strip_bbox.y2

    for b in sorted_balloons[1:]:
        gap = b.strip_bbox.y1 - current_y_bottom
        if gap < gap_threshold:
            current_balloons.append(b)
            current_y_bottom = max(current_y_bottom, b.strip_bbox.y2)
        else:
            y_top = min(x.strip_bbox.y1 for x in current_balloons) - margin
            y_bottom = max(x.strip_bbox.y2 for x in current_balloons) + margin
            bands.append(Band(y_top=max(0, y_top), y_bottom=y_bottom, balloons=current_balloons))
            current_balloons = [b]
            current_y_bottom = b.strip_bbox.y2

    y_top = min(x.strip_bbox.y1 for x in current_balloons) - margin
    y_bottom = max(x.strip_bbox.y2 for x in current_balloons) + margin
    bands.append(Band(y_top=max(0, y_top), y_bottom=y_bottom, balloons=current_balloons))

    return bands


def attach_band_slices(strip: VerticalStrip, bands: list[Band]) -> None:
    """Popula strip_slice e original_slice em cada banda (in-place)."""
    for band in bands:
        y0 = max(0, band.y_top)
        y1 = min(strip.height, band.y_bottom)
        view = strip.image[y0:y1, :, :]
        band.strip_slice = view.copy()
        band.original_slice = view.copy()
