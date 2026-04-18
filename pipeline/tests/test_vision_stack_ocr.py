import unittest
import builtins
import types
from unittest.mock import patch

import numpy as np
import cv2

from vision_stack.ocr import (
    OCREngine,
    normalize_easyocr_languages,
    normalize_paddleocr_language,
)


class VisionStackOCRTests(unittest.TestCase):
    def test_normalize_paddleocr_language_handles_regions_and_common_languages(self):
        self.assertEqual(normalize_paddleocr_language("en-GB"), "en")
        self.assertEqual(normalize_paddleocr_language("pt-BR"), "pt")
        self.assertEqual(normalize_paddleocr_language("zh-TW"), "chinese_cht")
        self.assertEqual(normalize_paddleocr_language("ja"), "japan")
        self.assertEqual(normalize_paddleocr_language("ko"), "korean")
        self.assertEqual(normalize_paddleocr_language("ru"), "ru")

    def test_normalize_easyocr_languages_handles_regions_and_fallbacks(self):
        self.assertEqual(normalize_easyocr_languages("en-GB"), ["en"])
        self.assertEqual(normalize_easyocr_languages("pt-BR"), ["pt", "en"])
        self.assertEqual(normalize_easyocr_languages("zh-TW"), ["ch_tra", "en"])
        self.assertEqual(normalize_easyocr_languages("ru"), ["ru", "en"])

    def test_manga_ocr_falls_back_to_paddle_when_model_load_breaks(self):
        engine = OCREngine.__new__(OCREngine)
        engine.model_name = "manga-ocr"
        engine.device = type("Device", (), {"type": "cpu"})()
        engine.half = False
        engine.batch_size = 8
        engine._model = None
        engine._processor = None
        original_import = builtins.__import__

        transformers_stub = types.ModuleType("transformers")

        class _BrokenAutoFeatureExtractor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                del args, kwargs
                raise ValueError("broken hf metadata")

        class _UnusedModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                del args, kwargs
                raise AssertionError("nao deveria chegar aqui")

        transformers_stub.AutoFeatureExtractor = _BrokenAutoFeatureExtractor
        transformers_stub.VisionEncoderDecoderModel = _UnusedModel
        transformers_stub.AutoTokenizer = _UnusedModel

        with patch("vision_stack.ocr.OCREngine._load_paddle_ocr") as load_paddle, patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: transformers_stub
            if name == "transformers"
            else original_import(name, *args, **kwargs),
        ):
            OCREngine._load_manga_ocr(engine)

        self.assertEqual(engine.model_name, "paddleocr")
        load_paddle.assert_called_once()

    def test_paddle_ocr_retries_empty_result_with_upscaled_variants(self):
        engine = OCREngine.__new__(OCREngine)

        class FakeModel:
            def ocr(self, crop, det=True, rec=True, cls=True):
                h, w = crop.shape[:2]
                if h < 60:
                    return [[]]
                return [[[[0, 0], ("A SINGLE STRIKE, SO I NEVER", 0.99)]]]

        engine._model = FakeModel()
        crop = np.zeros((36, 452, 3), dtype=np.uint8)

        texts = OCREngine._paddle_ocr_batch(engine, [crop])

        self.assertEqual(len(texts), 1)
        self.assertIn("A SINGLE STRIKE", texts[0])

    def test_paddle_ocr_detects_dot_run_when_ocr_returns_empty(self):
        engine = OCREngine.__new__(OCREngine)

        class FakeModel:
            def ocr(self, crop, det=True, rec=True, cls=True):
                return [[]]

        engine._model = FakeModel()
        crop = np.full((32, 84, 3), 255, dtype=np.uint8)
        for x in (13, 25, 37, 49, 61, 73):
            cv2.circle(crop, (x, 20), 3, (0, 0, 0), thickness=-1)

        texts = OCREngine._paddle_ocr_batch(engine, [crop])

        self.assertEqual(texts, ["......"])

    def test_load_paddle_ocr_falls_back_to_easyocr_when_paddle_is_unavailable(self):
        engine = OCREngine.__new__(OCREngine)
        engine.model_name = "paddleocr"
        engine.lang = "en"
        engine.device = type("Device", (), {"type": "cpu"})()
        engine.half = False
        engine.batch_size = 8
        engine._model = None
        engine._processor = None
        original_import = builtins.__import__

        with patch("vision_stack.ocr.OCREngine._load_easyocr") as load_easyocr, patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError(name))
            if name == "paddleocr"
            else original_import(name, *args, **kwargs),
        ):
            OCREngine._load_paddle_ocr(engine)

        self.assertEqual(engine.model_name, "easyocr")
        load_easyocr.assert_called_once()


if __name__ == "__main__":
    unittest.main()
