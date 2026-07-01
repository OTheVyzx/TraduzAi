import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inpainter.notanother_adapter import build_notanother_text_mask


def test_component_mask_keeps_only_glyphs_inside_safe_bubble():
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    bubble = np.zeros((80, 100), dtype=np.uint8)
    bubble[10:64, 12:88] = 255
    support = np.zeros((80, 100), dtype=np.uint8)
    support[18:54, 20:78] = 255
    image[28:36, 32:46] = 12
    image[66:72, 32:46] = 12

    mask, debug = build_notanother_text_mask(image, bubble, support)

    assert int(np.count_nonzero(mask[28:36, 32:46])) > 0
    assert int(np.count_nonzero(mask[66:72, 32:46])) == 0
    assert debug["component_accepted"] == 1
    assert debug["component_rejected_outside_bubble"] >= 0


def test_component_mask_rejects_balloon_outline_components():
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    bubble = np.zeros((80, 100), dtype=np.uint8)
    bubble[10:64, 12:88] = 255
    support = bubble.copy()
    image[10:15, 36:52] = 10
    image[32:40, 44:56] = 10

    mask, debug = build_notanother_text_mask(image, bubble, support, erode_outline_px=3)

    assert int(np.count_nonzero(mask[10:15, 36:52])) == 0
    assert int(np.count_nonzero(mask[32:40, 44:56])) > 0
    assert debug["component_rejected_outside_bubble"] >= 1


def test_component_mask_fills_holes_after_acceptance_but_clips_to_support():
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    bubble = np.zeros((80, 100), dtype=np.uint8)
    bubble[8:70, 8:92] = 255
    support = np.zeros((80, 100), dtype=np.uint8)
    support[22:50, 25:58] = 255
    cv2.rectangle(image, (28, 24), (54, 46), (8, 8, 8), 2)

    mask, debug = build_notanother_text_mask(image, bubble, support)

    assert int(mask[35, 40]) == 255
    assert int(mask[20, 40]) == 0
    assert debug["hole_fill_pixels"] > 0


def test_component_mask_rejects_components_outside_bubble_even_inside_support():
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    bubble = np.zeros((80, 100), dtype=np.uint8)
    bubble[10:54, 10:58] = 255
    support = np.zeros((80, 100), dtype=np.uint8)
    support[16:58, 16:88] = 255
    image[28:36, 24:38] = 12
    image[28:36, 72:84] = 12

    mask, debug = build_notanother_text_mask(image, bubble, support)

    assert int(np.count_nonzero(mask[28:36, 24:38])) > 0
    assert int(np.count_nonzero(mask[28:36, 72:84])) == 0
    assert debug["component_rejected_outside_bubble"] >= 1
