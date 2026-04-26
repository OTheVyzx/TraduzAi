"""Reassembly do strip processado em ~60 páginas alvo.

Algoritmo de split (decisão §2.1):
  1. Calcula split-points ideais: y = i * strip_height / target_count
  2. Para cada split, se ele cair dentro de um balão, anda PRA CIMA até cair num gap.
  3. Garantia: nenhum balão é cortado entre páginas.
"""

from __future__ import annotations

import numpy as np

from strip.types import Balloon, OutputPage, VerticalStrip


_MIN_PAGE_HEIGHT = 50  # mínimo de px por página output (evita páginas degeneradas)


def _is_inside_any_balloon(
    y: int,
    balloons: list[Balloon],
    padding: int = 20,
) -> bool:
    """Retorna True se y está dentro de qualquer balão (com padding visual para spikes)."""
    for b in balloons:
        if (b.strip_bbox.y1 - padding) <= y <= (b.strip_bbox.y2 + padding):
            return True
    return False


def _balloon_top_above(
    y: int,
    balloons: list[Balloon],
    safety_margin: int = 12,
    padding: int = 20,
) -> int | None:
    """Retorna a posição segura acima do balão que contém y.

    O `padding` representa a extensão visual do balão (spikes, burst effects)
    além do bbox do detector. O `safety_margin` é a distância extra antes do
    início da zona visual. Posição retornada = top - padding - safety_margin.
    """
    candidates = [
        b for b in balloons
        if (b.strip_bbox.y1 - padding) <= y <= (b.strip_bbox.y2 + padding)
    ]
    if not candidates:
        return None
    top = min(b.strip_bbox.y1 for b in candidates)
    # Subtrair tanto o padding quanto a margem de segurança
    return max(0, top - padding - safety_margin)


def _compute_split_points(
    strip_height: int,
    balloons: list[Balloon],
    target_count: int = 60,
    safety_margin: int = 12,
    balloon_visual_padding: int = 20,
) -> list[int]:
    """Calcula pontos de corte que nunca atravessam um balão.

    Parâmetros:
    - safety_margin: pixels de distância mínima entre o split e o topo do balão
    - balloon_visual_padding: pixels extras considerados ao redor do bbox detector
      para proteger spikes e burst effects
    """
    if strip_height <= 0:
        return [0, 0]
    if target_count < 1:
        target_count = 1

    ideal_step = strip_height / target_count
    points = [0]
    for i in range(1, target_count):
        candidate = int(round(i * ideal_step))
        # Se candidato cai dentro de balão (com padding), sobe até primeiro gap
        guard = 0
        while _is_inside_any_balloon(candidate, balloons, padding=balloon_visual_padding) and guard < 1000:
            new_y = _balloon_top_above(
                candidate, balloons,
                safety_margin=safety_margin,
                padding=balloon_visual_padding,
            )
            if new_y is None:
                break
            candidate = new_y
            guard += 1
        # Descartar splits que gerariam páginas muito pequenas
        if candidate - points[-1] < _MIN_PAGE_HEIGHT:
            continue
        points.append(candidate)
    points.append(strip_height)
    return points


def assemble_output_pages(
    strip: VerticalStrip,
    balloons: list[Balloon],
    target_count: int = 60,
) -> list[OutputPage]:
    """Re-fatia o strip em páginas alvo evitando cortar balões."""
    points = _compute_split_points(strip.height, balloons, target_count)
    pages: list[OutputPage] = []
    for i in range(len(points) - 1):
        y0, y1 = points[i], points[i + 1]
        if y1 <= y0:
            continue
        pages.append(OutputPage(
            y_top=y0,
            y_bottom=y1,
            image=strip.image[y0:y1, :, :].copy(),
        ))
    return pages


def paste_bands_into_strip(strip: VerticalStrip, bands: list) -> None:
    """Cola cada `band.rendered_slice` no strip nas coordenadas (y_top:y_bottom)."""
    for band in bands:
        if band.rendered_slice is None:
            continue
        y0 = max(0, band.y_top)
        y1 = min(strip.height, band.y_bottom)
        h_avail = y1 - y0
        if h_avail <= 0:
            continue
        strip.image[y0:y1, :, :] = band.rendered_slice[:h_avail, :, :]

