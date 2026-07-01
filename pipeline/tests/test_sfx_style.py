import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfx.style import SfxStyle, extract_manhwa_sfx_style


def test_extracts_white_fill_black_stroke_for_sfx():
    crop = np.full((100, 180, 3), 245, dtype=np.uint8)
    mask = np.zeros((100, 180), dtype=np.uint8)
    cv2.rectangle(mask, (40, 25), (140, 75), 255, -1)
    crop[mask > 0] = [0, 0, 0]
    cv2.rectangle(crop, (52, 36), (128, 64), (255, 255, 255), -1)

    style = extract_manhwa_sfx_style(crop, mask)

    assert isinstance(style, SfxStyle)
    assert style.fill_color == "#FFFFFF"
    assert style.stroke_color == "#000000"
    assert style.stroke_width_px >= 1
    assert style.confidence >= 0.5


def test_extracts_black_fill_white_glow_for_sfx():
    crop = np.full((120, 200, 3), 18, dtype=np.uint8)
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[48:78, 76:128] = 255
    glow = cv2.GaussianBlur(mask, (0, 0), sigmaX=5.0)
    crop = np.maximum(crop, np.dstack([glow, glow, glow]).astype(np.uint8))
    crop[mask > 0] = [0, 0, 0]

    style = extract_manhwa_sfx_style(crop, mask)

    assert style.fill_color == "#000000"
    assert style.glow_color != ""
    assert style.glow_width_px >= 4
    assert style.confidence >= 0.45


def test_extracts_colored_fill_from_sfx_mask():
    crop = np.full((80, 160, 3), 240, dtype=np.uint8)
    mask = np.zeros((80, 160), dtype=np.uint8)
    mask[20:60, 40:120] = 255
    crop[mask > 0] = [224, 36, 48]

    style = extract_manhwa_sfx_style(crop, mask)

    assert style.fill_color == "#E02430"
    assert style.stroke_color == ""
    assert style.confidence >= 0.4


def test_extracts_rotation_from_sfx_mask_pixels():
    crop = np.full((120, 180, 3), 245, dtype=np.uint8)
    mask = np.zeros((120, 180), dtype=np.uint8)
    rect = ((90, 60), (90, 18), 28)
    points = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    crop[mask > 0] = [20, 20, 20]

    style = extract_manhwa_sfx_style(crop, mask)

    assert math.isclose(style.rotation_deg, 28.0, abs_tol=6.0)
    assert style.scale_x > style.scale_y
    assert style.confidence >= 0.4


def test_missing_mask_marks_low_confidence_geometry():
    crop = np.full((80, 160, 3), 255, dtype=np.uint8)
    crop[25:55, 40:120] = 0

    style = extract_manhwa_sfx_style(crop)

    assert "sfx_style_missing_mask" in style.qa_flags
    assert "sfx_style_geometry_low_confidence" in style.qa_flags
