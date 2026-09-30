"""TextPath raster with a face/size baseline independent of line content.

Only the quality overlay imports this module. The vertical canvas reserves a
fixed Portuguese repertoire for each face and size; individual ink bounds are
still measured for clipping and whole-block alignment.
"""

from __future__ import annotations

import math
import unicodedata
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


PORTUGUESE_EXTENTS = "ÁÂÃÀÉÊÍÓÔÕÚÜÇáâãàéêíóôõúüçgjpqy!?.,"
BASELINE_FACTORS = {"standard": 1.0, "compact_5_unadorned": 0.95}


@lru_cache(maxsize=256)
def vertical_metrics(font_path: str, size: int) -> tuple[int, int, int]:
    """Return fixed (baseline offset, cell height, minimum line advance)."""
    from matplotlib.ft2font import FT2Font

    ft = FT2Font(str(Path(font_path).resolve()))
    units = max(1, int(ft.units_per_EM))
    ascent = ft.ascender * size / units
    descent = -ft.descender * size / units
    sample = TextPath(
        (0, 0), PORTUGUESE_EXTENTS,
        prop=FontProperties(fname=str(font_path), size=size), usetex=False,
    ).get_extents()
    top = max(ascent, float(sample.y1))
    bottom = max(descent, -float(sample.y0))
    pad = max(2, int(math.ceil(size * 0.08)))
    baseline = int(math.ceil(top)) + pad
    height = baseline + int(math.ceil(bottom)) + pad
    gap = max(5, int(round(size * 0.18)))
    advance = max(int(math.ceil(float(sample.y1 - sample.y0))) + gap,
                  int(math.ceil(size * 1.10)))
    return baseline, height, advance


def line_advance(font_path: str, size: int, profile: str = "standard") -> int:
    """Apply a declared, face-independent factor to a fixed face/size advance."""
    if profile not in BASELINE_FACTORS:
        raise ValueError("unknown baseline profile")
    base = vertical_metrics(font_path, size)[2]
    return int(math.floor(base * BASELINE_FACTORS[profile] + 0.5))


def rasterize(font_path: str, size: int, text: str) -> np.ndarray:
    """Paint a complete Unicode run at the same baseline for every line.

    TextPath retains kerning and mark placement. Its bounds determine only the
    horizontal ink extent; they never shift the vertical origin.
    """
    text = unicodedata.normalize("NFC", text)
    if not text or not text.strip():
        return np.zeros((1, 1), dtype=np.uint8)
    prop = FontProperties(fname=str(font_path), size=int(size))
    path = TextPath((0, 0), text, prop=prop, usetex=False)
    bbox = path.get_extents()
    baseline, height, _ = vertical_metrics(str(font_path), int(size))
    if baseline - bbox.y1 < 0 or baseline - bbox.y0 > height:
        raise ValueError("glyph exceeds fixed font cell; review required")
    width = max(1, int(math.ceil(bbox.width)) + 1)
    oversample = 3
    canvas = np.zeros((height * oversample, width * oversample), dtype=np.uint8)
    transform = (Affine2D().scale(oversample, -oversample)
                 .translate((0.5 - bbox.x0) * oversample, baseline * oversample))
    polygons = [np.round(polygon).astype(np.int32)
                for polygon in path.transformed(transform).to_polygons()
                if len(polygon) >= 3]
    if polygons:
        cv2.fillPoly(canvas, polygons, 255)
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
