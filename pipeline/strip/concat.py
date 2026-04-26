"""Concatenação vertical de páginas em um strip.

Decisões aplicáveis (ver design doc):
  §4 — Letterbox em branco para páginas mais estreitas que max(width).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from strip.types import VerticalStrip


def build_strip(
    page_paths: list[Path],
    progress_callback=None,
) -> VerticalStrip:
    """Concatena páginas verticalmente sem separadores."""
    if not page_paths:
        return VerticalStrip(
            image=np.zeros((0, 0, 3), dtype=np.uint8),
            width=0,
            height=0,
            source_page_breaks=[0],
        )

    # Primeira passada: ler imagens
    images: list[np.ndarray] = []
    for i, path in enumerate(page_paths):
        if progress_callback:
            progress_callback("concat", i, len(page_paths))
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Não consegui ler {path}")
        images.append(img)

    max_width = max(img.shape[1] for img in images)
    total_height = sum(img.shape[0] for img in images)

    # Aloca strip preenchido com branco (letterbox default)
    strip = np.full((total_height, max_width, 3), 255, dtype=np.uint8)

    page_breaks: list[int] = [0]
    page_x_offsets: list[int] = []
    cursor = 0
    for img in images:
        h, w = img.shape[:2]
        x_offset = (max_width - w) // 2  # centraliza (letterbox)
        strip[cursor:cursor + h, x_offset:x_offset + w] = img
        page_x_offsets.append(x_offset)
        cursor += h
        page_breaks.append(cursor)

    return VerticalStrip(
        image=strip,
        width=max_width,
        height=total_height,
        source_page_breaks=page_breaks,
        page_x_offsets=page_x_offsets,
    )


def split_strip_back(strip: VerticalStrip) -> list[np.ndarray]:
    """Inverso de build_strip — útil para validação do round-trip.

    NÃO é usado no pipeline real (lá usamos assemble_output_pages da Fase 5
    com split-points adaptativos). Este helper só serve para testes.
    """
    pages = []
    breaks = strip.source_page_breaks
    for i in range(len(breaks) - 1):
        y0, y1 = breaks[i], breaks[i + 1]
        pages.append(strip.image[y0:y1].copy())
    return pages
