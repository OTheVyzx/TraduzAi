"""Tipos compartilhados pelo pipeline strip-based.

Domain model: ver docs/plans/2026-04-25-arquitetura-strip-pipeline-design.md §4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BBox:
    """Bounding box (x1, y1, x2, y2) em coordenadas absolutas do strip."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass
class Page:
    """Página de entrada extraída do CBZ/PDF."""
    path: Path
    width: int
    height: int
    y_offset_in_strip: int = 0


@dataclass
class VerticalStrip:
    """Strip vertical contendo todas as páginas concatenadas."""
    image: np.ndarray  # uint8, shape (H, W, 3)
    width: int
    height: int
    source_page_breaks: list[int] = field(default_factory=list)
    page_x_offsets: list[int] = field(default_factory=list)  # letterbox offset por página


@dataclass
class Balloon:
    """Balão detectado no strip."""
    strip_bbox: BBox
    confidence: float
    lobe_count: int = 1


@dataclass
class Band:
    """Fatia horizontal full-width contendo 1+ balões."""
    y_top: int
    y_bottom: int
    balloons: list[Balloon] = field(default_factory=list)
    strip_slice: Optional[np.ndarray] = None
    original_slice: Optional[np.ndarray] = None
    cleaned_slice: Optional[np.ndarray] = None
    rendered_slice: Optional[np.ndarray] = None
    ocr_result: Optional[dict] = None


    @property
    def height(self) -> int:
        return max(0, self.y_bottom - self.y_top)


@dataclass
class OutputPage:
    """Página final que sai do reassembly."""
    y_top: int
    y_bottom: int
    image: np.ndarray
    original_image: Optional[np.ndarray] = None
    inpainted_image: Optional[np.ndarray] = None
    path: Optional[Path] = None
    ocr_result: dict = field(default_factory=dict)
    text_layers: dict = field(default_factory=dict)
    page_profile: Optional[dict] = None
    inpaint_blocks: Optional[list] = None
