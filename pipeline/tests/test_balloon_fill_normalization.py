import numpy as np

from inpainter.fill_normalization import normalize_white_balloon_fill


def test_normalize_white_balloon_fill_replaces_gray_patch_on_flat_white():
    image = np.full((60, 80, 3), 246, dtype=np.uint8)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[24:36, 30:50] = 255
    image[mask > 0] = [205, 205, 205]

    result = normalize_white_balloon_fill(image, mask, {"balloon_type": "white"})

    assert int(result[30, 40, 0]) >= 240


def test_normalize_white_balloon_fill_skips_textured_samples():
    image = np.full((60, 80, 3), 246, dtype=np.uint8)
    for x in range(80):
        image[:, x] = [180 + (x % 40), 180 + (x % 30), 180 + (x % 20)]
    original = image.copy()
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[24:36, 30:50] = 255

    result = normalize_white_balloon_fill(image, mask, {"balloon_type": "white"})

    assert np.array_equal(result, original)
