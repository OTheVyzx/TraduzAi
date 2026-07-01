import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ocr.postprocess import apply_language_guards
from vision_stack.runtime import build_page_result, _drop_suppressed_ocr_pairs
from main import _drop_suppressed_ocr_texts


class OcrLanguageFilterTests(unittest.TestCase):
    def test_english_source_drops_korean_only_artifact_and_keeps_english_dialogue(self):
        image = np.full((120, 180, 3), 255, dtype=np.uint8)
        blocks = [
            SimpleNamespace(xyxy=(16, 16, 58, 42), mask=None, confidence=0.91),
            SimpleNamespace(xyxy=(70, 18, 150, 48), mask=None, confidence=0.94),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
                page = build_page_result(
                    image_path="page.jpg",
                    image_rgb=image,
                    blocks=blocks,
                    texts=["쾅", "HELLO THERE"],
                    idioma_origem="en",
                )

            decision_log.finalize_decision_trace()

            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual([item["text"] for item in page["texts"]], ["HELLO THERE"])
        self.assertEqual([item["text"] for item in page["_vision_blocks"]], ["HELLO THERE"])
        self.assertTrue(any(item.get("reason") == "korean_text_in_english_source" for item in payloads))

    def test_english_source_suppresses_ocr_gibberish_before_render(self):
        guarded = apply_language_guards(
            [
                {"text": "3 TI2]2H", "qa_flags": ["ocr_gibberish"]},
                {"text": "HELLO THERE", "qa_flags": []},
            ],
            source_language="en",
        )

        self.assertEqual(guarded[0]["route"], "suppress")
        self.assertEqual(guarded[0]["route_reason"], "english_ocr_gibberish_suppressed")
        self.assertEqual(guarded[1].get("route"), None)

    def test_english_source_suppresses_scanlator_text_caption_before_render(self):
        guarded = apply_language_guards(
            [
                {"text": "TEXT: DARLING KARAOKE", "qa_flags": []},
                {"text": "I REALLY DON'T HAVE ANY MONEY RIGHT NOW!", "qa_flags": []},
            ],
            source_language="en",
        )

        self.assertEqual(guarded[0]["route"], "suppress")
        self.assertEqual(guarded[0]["route_reason"], "scanlator_text_caption_suppressed")
        self.assertIn("scanlator_text_caption_suppressed", guarded[0]["qa_flags"])
        self.assertIsNone(guarded[1].get("route"))

    def test_main_final_filter_drops_scanlator_text_caption(self):
        filtered = _drop_suppressed_ocr_texts(
            [
                {"text": "TEXT: DARLING KARAOKE", "qa_flags": []},
                {"text": "I REALLY DON'T HAVE ANY MONEY RIGHT NOW!", "qa_flags": []},
            ],
            source_language="en",
        )

        self.assertEqual([item["text"] for item in filtered], ["I REALLY DON'T HAVE ANY MONEY RIGHT NOW!"])

    def test_english_source_suppresses_cjk_text_before_render(self):
        guarded = apply_language_guards(
            [
                {"text": "달링 가라오케", "qa_flags": []},
                {"text": "PLEASE WAIT", "qa_flags": []},
            ],
            source_language="en",
        )

        self.assertEqual(guarded[0]["route"], "suppress")
        self.assertEqual(guarded[0]["route_reason"], "source_language_cjk_text_suppressed")
        self.assertIn("source_language_cjk_text_suppressed", guarded[0]["qa_flags"])
        self.assertIsNone(guarded[1].get("route"))

    def test_runtime_pair_filter_drops_scanlator_and_cjk_without_misalignment(self):
        texts, blocks = _drop_suppressed_ocr_pairs(
            [
                {"text": "TEXT: DARLING KARAOKE", "bbox": [1, 1, 20, 8], "qa_flags": []},
                {"text": "달링 가라오케", "bbox": [3, 12, 40, 24], "qa_flags": []},
                {"text": "I REALLY DON'T HAVE ANY MONEY RIGHT NOW!", "bbox": [10, 30, 90, 60], "qa_flags": []},
            ],
            [
                {"text": "TEXT: DARLING KARAOKE", "bbox": [1, 1, 20, 8]},
                {"text": "달링 가라오케", "bbox": [3, 12, 40, 24]},
                {"text": "I REALLY DON'T HAVE ANY MONEY RIGHT NOW!", "bbox": [10, 30, 90, 60]},
            ],
            source_language="en",
            page_number=1,
        )

        self.assertEqual([item["text"] for item in texts], ["I REALLY DON'T HAVE ANY MONEY RIGHT NOW!"])
        self.assertEqual([item["text"] for item in blocks], ["I REALLY DON'T HAVE ANY MONEY RIGHT NOW!"])


if __name__ == "__main__":
    unittest.main()
