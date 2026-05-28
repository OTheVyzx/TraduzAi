import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_check_render_inside_balloon_flags_low_containment():
    from qa.render_geometry import check_render_inside_balloon

    result = check_render_inside_balloon(
        render_bbox=[0, 0, 100, 100],
        balloon_bbox=[0, 0, 80, 80],
    )

    assert result["containment"] == 0.64
    assert result["flags"] == ["render_outside_balloon"]


def test_check_render_background_flags_dark_sample_in_white_balloon():
    from qa.render_geometry import check_render_background

    image = np.full((80, 120, 3), 245, dtype=np.uint8)
    image[20:40, 30:70] = 42

    result = check_render_background(
        image,
        render_bbox=[30, 20, 70, 40],
        balloon_bbox=[10, 10, 100, 70],
        balloon_type="white",
    )

    assert result["background_luma"] == 42.0
    assert result["flags"] == ["render_on_art_suspected"]
