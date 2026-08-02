from __future__ import annotations

import cv2
import numpy as np


def test_stroke_expansion_uses_stroke_thickness_for_connected_script_text():
    from inpainter.owner_mask import _stroke_expansion_radii

    mask = np.zeros((100, 260), dtype=np.uint8)
    cv2.putText(
        mask,
        "Thin script",
        (5, 35),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        1.0,
        255,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        mask,
        "crosses art",
        (5, 72),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        1.0,
        255,
        2,
        cv2.LINE_AA,
    )

    assert _stroke_expansion_radii(mask) == (3,)
