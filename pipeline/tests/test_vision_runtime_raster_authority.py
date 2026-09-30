from __future__ import annotations

import numpy as np


def test_polygon_authority_uses_page_space_geometry() -> None:
    from vision_runtime.raster_authority import polygon_mask

    mask = polygon_mask((12, 16), [[3, 2], [10, 2], [10, 8], [3, 8]])

    assert mask.dtype == np.uint8
    assert mask.shape == (12, 16)
    assert mask[2, 3] == 255
    assert mask[8, 10] == 255
    assert mask[1, 3] == 0
    assert mask[9, 10] == 0


def test_preserved_art_requires_dark_pixels_unchanged_between_original_and_clean() -> None:
    from vision_runtime.raster_authority import preserved_art_mask

    original = np.full((8, 10, 3), 255, dtype=np.uint8)
    clean = original.copy()
    scope = np.full((8, 10), 255, dtype=np.uint8)

    # Preserved artwork is identical in both authenticated rasters.
    original[3, 6:9] = [10, 10, 10]
    clean[3, 6:9] = [10, 10, 10]
    # Source glyph ink was removed by cleaning and must not become protected art.
    original[5, 2:5] = [0, 0, 0]

    mask = preserved_art_mask(original, clean, scope, luma_max=245)

    assert np.all(mask[3, 6:9] == 255)
    assert np.all(mask[5, 2:5] == 0)


def test_preserved_art_is_clipped_to_explicit_scope() -> None:
    from vision_runtime.raster_authority import preserved_art_mask

    original = np.full((6, 6, 3), 255, dtype=np.uint8)
    original[1, 1] = [0, 0, 0]
    original[4, 4] = [0, 0, 0]
    clean = original.copy()
    scope = np.zeros((6, 6), dtype=np.uint8)
    scope[:3, :3] = 255

    mask = preserved_art_mask(original, clean, scope)

    assert mask[1, 1] == 255
    assert mask[4, 4] == 0
