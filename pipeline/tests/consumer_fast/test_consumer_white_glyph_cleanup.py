"""Regression checks for bounded cleanup of authenticated residual glyphs."""

import numpy as np
import pytest

from consumer_white_glyph_cleanup import clear_residual_white_glyphs


def test_only_dark_neutral_residual_inside_observation_is_cleared():
    clean = np.full((80, 100, 3), 255, np.uint8)
    clean[30:34, 39:43] = (12, 12, 12)     # authenticated residual glyph
    clean[30:34, 45:49] = (20, 60, 130)    # colored art, never cleanup
    clean[30:34, 59:63] = (0, 0, 0)        # independent neighbor
    clean[30:34, 35:37] = (0, 0, 0)        # near source contour
    distance = np.full((80, 100), 20.0, np.float32)
    distance[30:34, 35:37] = 1.0
    prepared = dict(source_roi_bbox=[0, 0, 100, 80], distance=distance,
                    minimum_px=4)
    source = clean.copy()
    result, evidence = clear_residual_white_glyphs(
        clean, source, prepared, [36, 28, 52, 38], pad_px=2)
    assert np.all(result[30:34, 39:43] == 255)
    assert np.array_equal(result[30:34, 45:49], clean[30:34, 45:49])
    assert np.array_equal(result[30:34, 59:63], clean[30:34, 59:63])
    assert np.array_equal(result[30:34, 35:37], clean[30:34, 35:37])
    assert evidence["changed_pixels"] == 16
    assert evidence["residual_dark_pixels"] == 0


def test_absent_residual_does_not_authorize_arbitrary_white_cleanup():
    clean = np.full((40, 40, 3), 255, np.uint8)
    prepared = dict(source_roi_bbox=[0, 0, 40, 40],
                    distance=np.full((40, 40), 20.0, np.float32), minimum_px=4)
    with pytest.raises(ValueError, match="no source-bound residual"):
        clear_residual_white_glyphs(clean, clean, prepared, [10, 10, 30, 30])


def test_dark_clean_pixel_without_source_ink_is_preserved():
    clean = np.full((40, 40, 3), 255, np.uint8)
    clean[20:24, 20:24] = 0
    source = np.full_like(clean, 255)
    prepared = dict(source_roi_bbox=[0, 0, 40, 40],
                    distance=np.full((40, 40), 20.0, np.float32), minimum_px=4)
    with pytest.raises(ValueError, match="no source-bound residual"):
        clear_residual_white_glyphs(clean, source, prepared, [10, 10, 30, 30])


def test_black_decoration_inside_white_body_outside_ocr_line_is_preserved():
    source = np.full((80, 100, 3), 255, np.uint8)
    source[32:36, 32:40] = 0  # selected source glyphs
    source[49:56, 54:62] = 0  # separate black decoration in same white body
    clean = source.copy()
    prepared = dict(source_roi_bbox=[0, 0, 100, 80],
                    distance=np.full((80, 100), 20.0, np.float32), minimum_px=4)
    result, evidence = clear_residual_white_glyphs(
        clean, source, prepared, [30, 30, 42, 38], pad_px=2)
    assert evidence["changed_pixels"] == 32
    assert np.array_equal(result[49:56, 54:62], source[49:56, 54:62])


def test_connected_antialias_fringe_beyond_old_pad_is_cleared_without_decoration():
    source = np.full((60, 70, 3), 255, np.uint8)
    source[20:28, 14:24] = 0             # source glyph, OCR starts at x=18
    source[21:23, 14] = (207, 207, 207)  # antialias 4 px left of OCR box
    source[20:28, 32:35] = 0             # independent black decoration
    clean = source.copy()
    prepared = dict(source_roi_bbox=[0, 0, 70, 60],
                    distance=np.full((60, 70), 20.0, np.float32), minimum_px=4)
    result, evidence = clear_residual_white_glyphs(clean, source, prepared,
                                                    [18, 18, 30, 30])
    assert np.all(result[20:28, 14:24] == 255)
    assert np.array_equal(result[20:28, 32:35], source[20:28, 32:35])
    assert evidence["extension_pad_px"] == 6
