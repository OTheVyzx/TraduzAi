import unittest
from unittest.mock import patch

import numpy as np

from ocr.detector import _run_primary_regions, run_ocr


class PrimaryOcrRoutingTests(unittest.TestCase):
    def test_uses_paddle_primary_when_available_and_nonempty(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        paddle_runs = [{"text": "HELLO", "confidence": 0.9, "bbox_pts": [[0, 0]], "source": "primary-paddle"}]

        with patch("ocr.detector.is_paddle_available", return_value=True), patch(
            "ocr.detector.run_paddle_primary_recognition", return_value=paddle_runs
        ), patch("ocr.detector.run_primary_recognition", return_value=[]):
            result = _run_primary_regions(reader=object(), image_bgr=image, preprocessed_image=image)

        self.assertEqual(result, paddle_runs)

    def test_falls_back_to_easyocr_when_paddle_returns_empty(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        easy_runs = [{"text": "FALLBACK", "confidence": 0.8, "bbox_pts": [[0, 0]], "source": "primary"}]

        with patch("ocr.detector.is_paddle_available", return_value=True), patch(
            "ocr.detector.run_paddle_primary_recognition", return_value=[]
        ), patch("ocr.detector.run_primary_recognition", return_value=easy_runs):
            result = _run_primary_regions(reader=object(), image_bgr=image, preprocessed_image=image)

        self.assertEqual(result, easy_runs)

    def test_run_ocr_returns_new_stack_result_when_available(self):
        expected = {"image": "page.jpg", "width": 100, "height": 100, "texts": [{"bbox": [10, 20, 50, 40]}]}

        with patch("ocr.detector.run_detect_ocr", return_value=expected):
            result = run_ocr("page.jpg", profile="quality")

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
