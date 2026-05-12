import numpy as np

from inpainter.residual_cleanup import cleanup_white_balloon_residuals


def test_cleanup_white_balloon_residuals_removes_small_dark_cluster():
    image = np.full((80, 100, 3), 246, dtype=np.uint8)
    balloon_mask = np.zeros((80, 100), dtype=np.uint8)
    balloon_mask[10:70, 15:85] = 255
    image[36:42, 45:52] = [20, 20, 20]

    result = cleanup_white_balloon_residuals(image, balloon_mask)

    assert int(result[38, 48, 0]) >= 230


def test_cleanup_white_balloon_residuals_keeps_outline_outside_eroded_interior():
    image = np.full((80, 100, 3), 246, dtype=np.uint8)
    balloon_mask = np.zeros((80, 100), dtype=np.uint8)
    balloon_mask[10:70, 15:85] = 255
    image[10, 40:60] = [0, 0, 0]

    result = cleanup_white_balloon_residuals(image, balloon_mask)

    assert int(result[10, 45, 0]) == 0
