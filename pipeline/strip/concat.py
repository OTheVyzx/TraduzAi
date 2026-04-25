"""Concatenação vertical de páginas em um strip.

Decisões aplicáveis (ver design doc):
  §4 — Letterbox em branco para páginas mais estreitas que max(width).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from strip.types import VerticalStrip


def build_strip(page_paths: list[Path]) -> VerticalStrip:
    """Concatena páginas verticalmente sem separadores.

    Páginas mais estreitas que a maior são centralizadas com letterbox branco
    para que o detector veja branco puro nas laterais e não dispare ali.
    """
    if not page_paths:
        return VerticalStrip(
            image=np.zeros((0, 0, 3), dtype=np.uint8),
            width=0,
            height=0,
            source_page_breaks=[0],
        )

    # Primeira passada: ler imagens
    images: list[np.ndarray] = []
    for path in page_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Não consegui ler {path}")
        images.append(img)

    max_width = max(img.shape[1] for img in images)
    total_height = sum(img.shape[0] for img in images)

    # Aloca strip preenchido com branco (letterbox default)
    strip = np.full((total_height, max_width, 3), 255, dtype=np.uint8)

    page_breaks: list[int] = [0]
    cursor = 0
    for img in images:
        h, w = img.shape[:2]
        x_offset = (max_width - w) // 2  # centraliza
        strip[cursor:cursor + h, x_offset:x_offset + w] = img
        cursor += h
        page_breaks.append(cursor)

    return VerticalStrip(
        image=strip,
        width=max_width,
        height=total_height,
        source_page_breaks=page_breaks,
    )
