import unittest

import cv2
import numpy as np

from layout.balloon_layout import refine_balloon_bbox_from_image


class BalloonRefinerTests(unittest.TestCase):
    def test_expands_to_light_balloon_area(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        image[:] = 25
        cv2.ellipse(image, (150, 150), (80, 60), 0, 0, 360, (245, 245, 245), -1)

        cluster_bbox = [125, 135, 175, 165]
        refined = refine_balloon_bbox_from_image(image, cluster_bbox, "fala")

        self.assertLessEqual(refined[0], 90)
        self.assertLessEqual(refined[1], 95)
        self.assertGreaterEqual(refined[2], 210)
        self.assertGreaterEqual(refined[3], 205)

    def test_keeps_original_when_no_clear_balloon_is_found(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        image[:] = 40
        cv2.rectangle(image, (120, 120), (180, 180), (70, 70, 70), -1)

        cluster_bbox = [125, 135, 175, 165]
        refined = refine_balloon_bbox_from_image(image, cluster_bbox, "fala")

        self.assertEqual(refined, cluster_bbox)


if __name__ == "__main__":
    unittest.main()
