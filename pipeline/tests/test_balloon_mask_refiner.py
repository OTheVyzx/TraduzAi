import unittest

import cv2
import numpy as np

from inpainter.lama_onnx import (
    refine_crop_mask_with_balloon_fill,
    segment_text_pixels_from_mask,
)


class BalloonMaskRefinerTests(unittest.TestCase):
    def test_segment_text_pixels_extracts_dark_letters_from_light_balloon(self):
        image = np.full((120, 180, 3), 245, dtype=np.uint8)
        cv2.rectangle(image, (48, 40), (60, 82), (12, 12, 12), thickness=-1)
        cv2.rectangle(image, (70, 40), (82, 82), (12, 12, 12), thickness=-1)
        cv2.rectangle(image, (92, 40), (126, 50), (12, 12, 12), thickness=-1)
        cv2.rectangle(image, (92, 57), (126, 67), (12, 12, 12), thickness=-1)
        cv2.rectangle(image, (92, 74), (126, 84), (12, 12, 12), thickness=-1)

        mask = np.zeros((120, 180), dtype=np.uint8)
        mask[30:92, 40:136] = 255

        segmented = segment_text_pixels_from_mask(image, mask)

        self.assertGreater(int(segmented[55, 54]), 0)
        self.assertEqual(int(segmented[55, 66]), 0)
        self.assertLess(np.count_nonzero(segmented), np.count_nonzero(mask) * 0.6)

    def test_segment_text_pixels_extracts_light_letters_from_dark_balloon(self):
        image = np.full((120, 180, 3), (120, 22, 32), dtype=np.uint8)
        cv2.rectangle(image, (48, 40), (60, 82), (250, 248, 248), thickness=-1)
        cv2.rectangle(image, (70, 40), (82, 82), (250, 248, 248), thickness=-1)
        cv2.rectangle(image, (92, 40), (126, 50), (250, 248, 248), thickness=-1)
        cv2.rectangle(image, (92, 57), (126, 67), (250, 248, 248), thickness=-1)
        cv2.rectangle(image, (92, 74), (126, 84), (250, 248, 248), thickness=-1)

        mask = np.zeros((120, 180), dtype=np.uint8)
        mask[30:92, 40:136] = 255

        segmented = segment_text_pixels_from_mask(image, mask)

        self.assertGreater(int(segmented[55, 54]), 0)
        self.assertEqual(int(segmented[55, 66]), 0)
        self.assertLess(np.count_nonzero(segmented), np.count_nonzero(mask) * 0.6)

    def test_refiner_keeps_mask_inside_white_balloon(self):
        image = np.full((140, 180, 3), 220, dtype=np.uint8)
        yy, xx = np.ogrid[:140, :180]
        cy, cx = 70, 90
        ellipse = (((yy - cy) / 35) ** 2 + ((xx - cx) / 70) ** 2) <= 1.0
        image[ellipse] = [245, 245, 245]
        image[ellipse ^ ((((yy - cy) / 32) ** 2 + ((xx - cx) / 67) ** 2) <= 1.0)] = [20, 20, 20]

        mask = np.zeros((140, 180), dtype=np.uint8)
        mask[58:82, 55:125] = 255

        refined = refine_crop_mask_with_balloon_fill(image, mask)

        self.assertTrue(np.any(refined > 0))
        self.assertEqual(int(refined[20, 20]), 0)
        self.assertGreater(int(refined[70, 90]), 0)

    def test_refiner_limits_growth_when_fill_leaks(self):
        image = np.full((100, 140, 3), 235, dtype=np.uint8)
        mask = np.zeros((100, 140), dtype=np.uint8)
        mask[42:58, 45:95] = 255

        refined = refine_crop_mask_with_balloon_fill(image, mask)

        self.assertLess(int(np.count_nonzero(refined)), 100 * 140)


if __name__ == "__main__":
    unittest.main()
