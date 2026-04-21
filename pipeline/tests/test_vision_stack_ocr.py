import unittest
import builtins
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2

import vision_stack.ocr as ocr_mod
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

    def test_derive_text_pixel_bbox_tracks_tight_text_extent_on_synthetic_text(self):
        page = np.full((180, 320, 3), 255, dtype=np.uint8)
        cv2.putText(
            page,
            "ABC",
            (56, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            2,
            cv2.LINE_8,
        )

        bbox = ocr_mod._derive_text_pixel_bbox(page, [24, 48, 230, 140])

        gray = cv2.cvtColor(page[48:140, 24:230], cv2.COLOR_RGB2GRAY)
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(binary_inv > 0)
        expected = [24 + int(xs.min()), 48 + int(ys.min()), 24 + int(xs.max()) + 1, 48 + int(ys.max()) + 1]

        self.assertIsNotNone(bbox)
        self.assertLessEqual(abs(int(bbox[0]) - expected[0]), 2)
        self.assertLessEqual(abs(int(bbox[1]) - expected[1]), 2)
        self.assertLessEqual(abs(int(bbox[2]) - expected[2]), 2)
        self.assertLessEqual(abs(int(bbox[3]) - expected[3]), 2)

    def test_paddle_full_page_blocks_preserve_line_polygons_in_rich_records(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"

        class FakeModel:
            def ocr(self, page_bgr, det=True, rec=True, cls=False):
                del det, rec, cls
                return [[
                    [
                        [[10, 10], [70, 10], [70, 28], [10, 28]],
                        ("HELLO", 0.99),
                    ],
                    [
                        [[12, 34], [88, 34], [88, 50], [12, 50]],
                        ("WORLD", 0.97),
                    ],
                ]]

        engine._model = FakeModel()
        page = np.full((80, 120, 3), 255, dtype=np.uint8)

        records = OCREngine._paddle_ocr_full_page_to_blocks(engine, cv2.cvtColor(page, cv2.COLOR_RGB2BGR), [SimpleNamespace(x1=0, y1=0, x2=120, y2=80)])

        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], "HELLO WORLD")
        self.assertIn("line_polygons", records[0])
        self.assertGreaterEqual(len(records[0]["line_polygons"]), 2)
        self.assertIn("text_pixel_bbox", records[0])
        self.assertGreater(records[0]["text_pixel_bbox"][2], records[0]["text_pixel_bbox"][0])

    def test_recognize_blocks_from_page_crop_fallback_updates_rich_record_text(self):
        engine = OCREngine.__new__(OCREngine)
        engine._backend = "paddleocr"
        block = type("Block", (), {"xyxy": (10, 10, 70, 34), "confidence": 0.91})()

        with patch.object(
            engine,
            "_paddle_ocr_full_page_to_blocks",
            return_value=[{"text": "", "source_bbox": [10, 10, 70, 34], "line_polygons": []}],
        ), patch.object(
            engine,
            "_crop_block_from_page",
            return_value=np.full((24, 60, 3), 255, dtype=np.uint8),
        ), patch.object(
            engine,
            "_crop_might_have_text",
            return_value=True,
        ), patch.object(
            engine,
            "_recognize_single_paddle_with_retry",
            return_value="HELLO",
        ):
            records = engine.recognize_blocks_from_page(np.full((80, 120, 3), 255, dtype=np.uint8), [block])

        self.assertEqual(records[0]["text"], "HELLO")
        self.assertEqual(records[0]["source_bbox"], [10, 10, 70, 34])

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
