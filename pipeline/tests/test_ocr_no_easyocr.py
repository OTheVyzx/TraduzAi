import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import ocr.detector as active_detector
import ocr_legacy.detector as legacy_detector
from ocr_legacy.recognizer_paddle import choose_primary_ocr_engine


class OcrNoEasyOCRTests(unittest.TestCase):
    def test_active_detector_failure_does_not_fallback_to_legacy_easyocr(self):
        with patch.object(
            active_detector,
            "run_detect_ocr",
            side_effect=RuntimeError("PaddleOCR unavailable"),
        ), patch.object(active_detector, "run_legacy_ocr", create=True) as legacy_ocr:
            with self.assertRaisesRegex(RuntimeError, "PaddleOCR unavailable"):
                active_detector.run_ocr("missing.png")

        legacy_ocr.assert_not_called()

    def test_legacy_primary_engine_fails_closed_when_paddle_is_unavailable(self):
        image = np.full((24, 32, 3), 255, dtype=np.uint8)

        with patch.object(legacy_detector, "is_paddle_available", return_value=False), patch.object(
            legacy_detector,
            "run_primary_recognition",
            create=True,
        ) as easyocr_primary:
            with self.assertRaisesRegex(RuntimeError, "PaddleOCR"):
                legacy_detector._run_primary_regions(
                    reader=None,
                    image_bgr=image,
                    preprocessed_image=image,
                )

        easyocr_primary.assert_not_called()

    def test_legacy_recognizer_never_selects_easyocr_as_primary_engine(self):
        with self.assertRaisesRegex(RuntimeError, "PaddleOCR"):
            choose_primary_ocr_engine(False)

    def test_legacy_run_ocr_does_not_request_easyocr_reader_for_paddle_primary(self):
        image = np.full((48, 96, 3), 255, dtype=np.uint8)
        bbox_pts = [[8, 8], [72, 8], [72, 28], [8, 28]]

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            cv2.imwrite(str(image_path), image)

            with patch.object(
                legacy_detector,
                "_get_reader",
                side_effect=AssertionError("EasyOCR reader requested"),
            ), patch.object(
                legacy_detector,
                "_run_primary_regions",
                return_value=[
                    {
                        "bbox_pts": bbox_pts,
                        "text": "HELLO THERE",
                        "confidence": 0.93,
                        "source": "primary-paddle",
                    }
                ],
            ), patch.object(
                legacy_detector,
                "run_fallback_recognition",
                create=True,
            ) as fallback_recognition, patch.object(
                legacy_detector,
                "_get_font_detector_legacy",
                return_value=None,
            ):
                page = legacy_detector.run_ocr(str(image_path), idioma_origem="en")

        fallback_recognition.assert_not_called()
        self.assertEqual([item["text"] for item in page["texts"]], ["HELLO THERE"])
        self.assertEqual(page["texts"][0]["ocr_mode"], "paddleocr")


if __name__ == "__main__":
    unittest.main()
