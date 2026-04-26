"""Agrupamento de balões em bandas horizontais full-width."""

from __future__ import annotations

import numpy as np

from strip.types import Balloon, Band, VerticalStrip


def _flush_band(
    current_balloons: list[Balloon],
    margin: int,
) -> Band:
    """Cria uma Band a partir de uma lista de balões, aplicando margem."""
    y_top = min(x.strip_bbox.y1 for x in current_balloons) - margin
    y_bottom = max(x.strip_bbox.y2 for x in current_balloons) + margin
    return Band(y_top=max(0, y_top), y_bottom=y_bottom, balloons=list(current_balloons))


def group_balloons_into_bands(
    balloons: list[Balloon],
    gap_threshold: int = 64,
    margin: int = 16,
    max_band_height: int = 4000,
) -> list[Band]:
    """Agrupa balões em bandas horizontais.

    - Balloons dentro de `gap_threshold` px são agrupados na mesma banda.
    - Balloons com gap > `gap_threshold` iniciam nova banda.
    - Bandas com height > `max_band_height` não são formadas — balões com gap
      grande que causariam isso já serão separados pelo gap_threshold.
    """
    if not balloons:
        return []

    sorted_balloons = sorted(balloons, key=lambda b: (b.strip_bbox.y1, b.strip_bbox.x1))

    bands: list[Band] = []
    current_balloons: list[Balloon] = [sorted_balloons[0]]
    current_y_bottom = sorted_balloons[0].strip_bbox.y2

    for b in sorted_balloons[1:]:
        gap = b.strip_bbox.y1 - current_y_bottom
        # Calcular a altura que a banda teria se adicionarmos este balão
        prospective_y_top = min(x.strip_bbox.y1 for x in current_balloons) - margin
        prospective_y_bottom = max(b.strip_bbox.y2, current_y_bottom) + margin
        prospective_height = prospective_y_bottom - max(0, prospective_y_top)

        if gap < gap_threshold and prospective_height <= max_band_height:
            current_balloons.append(b)
            current_y_bottom = max(current_y_bottom, b.strip_bbox.y2)
        else:
            bands.append(_flush_band(current_balloons, margin))
            current_balloons = [b]
            current_y_bottom = b.strip_bbox.y2

    bands.append(_flush_band(current_balloons, margin))
    return bands


def attach_band_slices(strip: VerticalStrip, bands: list[Band]) -> None:
    """Popula strip_slice e original_slice em cada banda (in-place)."""
    for band in bands:
        y0 = max(0, band.y_top)
        y1 = min(strip.height, band.y_bottom)
        view = strip.image[y0:y1, :, :]
        band.strip_slice = view.copy()
        band.original_slice = view.copy()
