import unittest
import os
import json
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from vision_stack.runtime import (
    _apply_inpainting_round,
    _apply_textured_balloon_band_artifact_cleanup,
    _apply_textured_balloon_seam_cleanup,
    _apply_white_balloon_artifact_cleanup,
    _apply_white_balloon_line_artifact_cleanup,
    _apply_white_balloon_micro_artifact_cleanup,
    _apply_white_balloon_text_box_cleanup,
    _apply_bright_zone_line_cleanup,
    _apply_mask_boundary_seam_cleanup,
    _apply_white_balloon_fill,
    _apply_letter_white_boxes,
    _apply_white_text_overlay,
    _build_koharu_worker_page_result,
    _build_bright_zone_line_mask,
    _build_mask_boundary_seam_mask,
    _build_residual_cleanup_mask,
    _expand_bbox,
    _extract_white_balloon_fill_mask,
    _extract_white_balloon_text_boxes,
    _integrate_recovery_page,
    _is_white_balloon_region,
    _should_use_base_white_balloon_font,
    _merge_text_fragments,
    _merge_nearby_bboxes,
    _enlarge_koharu_window,
    _profile_to_ocr_model,
    _quick_text_presence_check,
    _run_koharu_blockwise_inpaint_page,
    _run_masked_inpaint_passes,
    _try_koharu_balloon_fill,
    build_page_result,
    run_inpaint_pages,
    run_detect_ocr,
    warmup_visual_stack,
    vision_blocks_to_mask,
)


class VisionStackRuntimeTests(unittest.TestCase):
    @staticmethod
    def _fixture_image_path(name: str) -> Path:
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "testes" / name,
            root / "testes" / "debug_pipeline" / "originals" / name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(name)

    def test_profile_to_ocr_model_defaults_to_paddleocr(self):
        original = os.environ.pop("MANGATL_ENABLE_MANGA_OCR", None)
        try:
            self.assertEqual(_profile_to_ocr_model("quality"), "paddleocr")
            self.assertEqual(_profile_to_ocr_model("alta"), "paddleocr")
        finally:
            if original is not None:
                os.environ["MANGATL_ENABLE_MANGA_OCR"] = original

    def test_profile_to_ocr_model_can_reenable_manga_ocr_by_flag(self):
        original = os.environ.get("MANGATL_ENABLE_MANGA_OCR")
        os.environ["MANGATL_ENABLE_MANGA_OCR"] = "1"
        try:
            self.assertEqual(_profile_to_ocr_model("quality"), "manga-ocr")
            self.assertEqual(_profile_to_ocr_model("rapida"), "paddleocr")
        finally:
            if original is None:
                os.environ.pop("MANGATL_ENABLE_MANGA_OCR", None)
            else:
                os.environ["MANGATL_ENABLE_MANGA_OCR"] = original

    def test_build_page_result_keeps_vision_blocks_for_later_inpainting(self):
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[18:42, 30:78] = 255
        block = SimpleNamespace(
            xyxy=(30, 18, 78, 42),
            mask=mask,
            confidence=0.91,
        )

        page = build_page_result(
            image_path="page.jpg",
            image_rgb=np.full((80, 120, 3), 255, dtype=np.uint8),
            blocks=[block],
            texts=["HELLO"],
        )

        self.assertEqual(page["texts"][0]["bbox"], [30, 18, 78, 42])
        self.assertEqual(page["texts"][0]["text"], "HELLO")
        self.assertEqual(len(page["_vision_blocks"]), 1)
        self.assertEqual(page["_vision_blocks"][0]["bbox"], [30, 18, 78, 42])
        self.assertEqual(int(np.count_nonzero(page["_vision_blocks"][0]["mask"])), int(np.count_nonzero(mask)))

    def test_build_page_result_accepts_rich_ocr_items_and_preserves_metadata(self):
        block = SimpleNamespace(
            xyxy=(30, 18, 78, 42),
            mask=None,
            confidence=0.91,
        )
        rich_item = {
            "text": "HELLO",
            "line_polygons": [
                [[34, 24], [56, 24], [56, 34], [34, 34]],
                [[34, 34], [62, 34], [62, 42], [34, 42]],
            ],
            "text_pixel_bbox": [35, 24, 61, 42],
            "bbox": [30, 18, 78, 42],
        }

        page = build_page_result(
            image_path="page.jpg",
            image_rgb=np.full((80, 120, 3), 255, dtype=np.uint8),
            blocks=[block],
            texts=[rich_item],
        )

        text = page["texts"][0]
        self.assertEqual(text["text"], "HELLO")
        self.assertEqual(text["bbox"], [30, 18, 78, 42])
        self.assertEqual(text["confidence"], 0.91)
        self.assertIn("tipo", text)
        self.assertFalse(text["skip_processing"])
        self.assertEqual(text["line_polygons"], rich_item["line_polygons"])
        self.assertEqual(text["text_pixel_bbox"], rich_item["text_pixel_bbox"])

    def test_build_page_result_merges_clustered_line_fragments_before_translation(self):
        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            blocks = [
                SimpleNamespace(xyxy=(80, 70, 320, 102), mask=None, confidence=0.93),
                SimpleNamespace(xyxy=(84, 110, 316, 142), mask=None, confidence=0.94),
                SimpleNamespace(xyxy=(88, 150, 312, 182), mask=None, confidence=0.92),
                SimpleNamespace(xyxy=(92, 190, 308, 222), mask=None, confidence=0.95),
            ]

            page = build_page_result(
                image_path="005.jpg",
                image_rgb=np.full((320, 420, 3), 255, dtype=np.uint8),
                blocks=blocks,
                texts=[
                    "HE BROKE",
                    "THE MANA-INFUSED",
                    "BLADE WITH SHEER",
                    "GRIP STRENGTH.",
                ],
            )

            decision_log.finalize_decision_trace()

            self.assertEqual(len(page["texts"]), 1)
            self.assertEqual(len(page["_vision_blocks"]), 1)
            self.assertIn("HE BROKE", page["texts"][0]["text"])
            self.assertIn("GRIP STRENGTH.", page["texts"][0]["text"])
            self.assertEqual(page["texts"][0]["ocr_merged_source_count"], 4)
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]
            self.assertTrue(any(item["action"] == "merge_blocks" for item in payloads))

    def test_build_page_result_marks_balloon_type_for_white_and_textured_regions(self):
        blocks = [
            SimpleNamespace(xyxy=(10, 10, 70, 34), mask=None, confidence=0.88),
            SimpleNamespace(xyxy=(10, 40, 70, 64), mask=None, confidence=0.89),
        ]

        with patch("vision_stack.runtime._should_use_base_white_balloon_font", side_effect=[True, False]), patch(
            "vision_stack.runtime._is_white_balloon_region",
            side_effect=[True, False],
        ):
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=np.full((100, 100, 3), 255, dtype=np.uint8),
                blocks=blocks,
                texts=["HELLO", "WORLD"],
            )

        self.assertEqual(page["texts"][0]["balloon_type"], "white")
        self.assertEqual(page["texts"][1]["balloon_type"], "textured")

    def test_build_page_result_skips_watermark_and_credit_noise(self):
        blocks = [
            SimpleNamespace(xyxy=(10, 10, 70, 30), mask=None, confidence=0.43),
            SimpleNamespace(xyxy=(10, 40, 90, 62), mask=None, confidence=0.46),
            SimpleNamespace(xyxy=(10, 64, 100, 86), mask=None, confidence=0.47),
            SimpleNamespace(xyxy=(10, 70, 110, 110), mask=None, confidence=0.91),
        ]

        page = build_page_result(
            image_path="page.jpg",
            image_rgb=np.full((140, 140, 3), 255, dtype=np.uint8),
            blocks=blocks,
            texts=["ASURASCANS.COM", "QC MED", "NIGHTTOONS", "GET OUT OF HERE!"],
        )

        self.assertEqual([item["text"] for item in page["texts"]], ["GET OUT OF HERE!"])

    def test_build_page_result_skips_structured_payload_and_records_reason(self):
        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            block = SimpleNamespace(
                xyxy=(30, 18, 78, 42),
                mask=None,
                confidence=0.73,
            )

            page = build_page_result(
                image_path="058.jpg",
                image_rgb=np.full((80, 120, 3), 255, dtype=np.uint8),
                blocks=[block],
                texts=[{"text": "", "source_bbox": [], "line_polygons": [], "text_pixel_bbox": []}],
            )

            decision_log.finalize_decision_trace()

            self.assertEqual(page["texts"], [])
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]
            self.assertTrue(any(item["reason"] == "structured_payload" for item in payloads))

    def test_build_page_result_skips_punctuation_only_noise_and_records_reason(self):
        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            block = SimpleNamespace(
                xyxy=(30, 18, 78, 42),
                mask=None,
                confidence=0.91,
            )

            page = build_page_result(
                image_path="027.jpg",
                image_rgb=np.full((80, 120, 3), 255, dtype=np.uint8),
                blocks=[block],
                texts=["-"],
            )

            decision_log.finalize_decision_trace()

            self.assertEqual(page["texts"], [])
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]
            self.assertTrue(any(item["reason"] == "punctuation_only" for item in payloads))

    def test_build_page_result_skips_short_ornamental_cover_noise_and_records_reason(self):
        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            block = SimpleNamespace(
                xyxy=(640, 490, 705, 520),
                mask=None,
                confidence=0.59,
            )

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=False), patch(
                "vision_stack.runtime.classify_text_type",
                return_value="narracao",
            ):
                page = build_page_result(
                    image_path="001.jpg",
                    image_rgb=np.full((800, 1200, 3), 80, dtype=np.uint8),
                    blocks=[block],
                    texts=["KIRO"],
                )

            decision_log.finalize_decision_trace()

            self.assertEqual(page["texts"], [])
            self.assertEqual(page["page_profile"], "cover_opening")
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]
            self.assertEqual(payloads[0]["action"], "classify_page_profile")
            self.assertEqual(payloads[0]["reason"], "cover_opening")
            self.assertTrue(any(item["reason"] == "ornamental_cover_noise" for item in payloads))

    def test_build_page_result_keeps_substantive_text_on_cover_opening_page(self):
        block = SimpleNamespace(
            xyxy=(120, 180, 780, 320),
            mask=None,
            confidence=0.94,
        )

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=False), patch(
            "vision_stack.runtime.classify_text_type",
            return_value="narracao",
        ):
            page = build_page_result(
                image_path="001.jpg",
                image_rgb=np.full((1600, 1100, 3), 90, dtype=np.uint8),
                blocks=[block],
                texts=["The battle for the northern wall had already begun."],
            )

        self.assertEqual(page["page_profile"], "cover_opening")
        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["page_profile"], "cover_opening")
        self.assertEqual(page["texts"][0]["text"], "The battle for the northern wall had already begun.")

    def test_build_page_result_skips_cover_title_logo_noise_even_when_confident(self):
        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            block = SimpleNamespace(
                xyxy=(560, 320, 1120, 620),
                mask=None,
                confidence=0.91,
            )

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=False), patch(
                "vision_stack.runtime.classify_text_type",
                return_value="narracao",
            ):
                page = build_page_result(
                    image_path="001.jpg",
                    image_rgb=np.full((800, 1200, 3), 70, dtype=np.uint8),
                    blocks=[block],
                    texts=["THE REGRESSED MERCENARYS MACHINATIONS KIRO SHOUNEN"],
                )

            decision_log.finalize_decision_trace()

            self.assertEqual(page["texts"], [])
            self.assertEqual(page["page_profile"], "cover_opening")
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]
            self.assertTrue(any(item["reason"] == "cover_title_logo" for item in payloads))

    def test_build_page_result_skips_white_background_cover_title_logo(self):
        block = SimpleNamespace(
            xyxy=(100, 320, 660, 640),
            mask=None,
            confidence=0.69,
        )

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=True), patch(
            "vision_stack.runtime.classify_text_type",
            return_value="fala",
        ):
            page = build_page_result(
                image_path="002__001.jpg",
                image_rgb=np.full((1600, 800, 3), 250, dtype=np.uint8),
                blocks=[block],
                texts=["Theregressed Mercenarys"],
            )

        self.assertEqual(page["page_profile"], "cover_opening")
        self.assertEqual(page["texts"], [])

    def test_build_page_result_assigns_top_narration_block_profile(self):
        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            block = SimpleNamespace(
                xyxy=(180, 40, 620, 118),
                mask=None,
                confidence=0.93,
            )

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=False), patch(
                "vision_stack.runtime.classify_text_type",
                return_value="narracao",
            ):
                page = build_page_result(
                    image_path="007.jpg",
                    image_rgb=np.full((1800, 800, 3), 220, dtype=np.uint8),
                    blocks=[block],
                    texts=["Three days later, the northern wall had already fallen."],
                )

            decision_log.finalize_decision_trace()

            self.assertEqual(page["texts"][0]["block_profile"], "top_narration")
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]
            self.assertTrue(
                any(
                    item["action"] == "classify_block_profile" and item["reason"] == "top_narration"
                    for item in payloads
                )
            )

    def test_build_page_result_skips_font_detector_by_default(self):
        block = SimpleNamespace(
            xyxy=(20, 16, 84, 40),
            mask=None,
            confidence=0.88,
        )

        with patch("vision_stack.runtime._get_font_detector") as get_font_detector:
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=np.full((80, 120, 3), 255, dtype=np.uint8),
                blocks=[block],
                texts=["HELLO"],
            )

        get_font_detector.assert_not_called()
        self.assertEqual(page["texts"][0]["text"], "HELLO")

    def test_build_page_result_treats_english_regions_as_english_for_non_latin_filter(self):
        block = SimpleNamespace(
            xyxy=(20, 16, 84, 40),
            mask=None,
            confidence=0.88,
        )

        page = build_page_result(
            image_path="page.jpg",
            image_rgb=np.full((80, 120, 3), 255, dtype=np.uint8),
            blocks=[block],
            texts=["Привет"],
            idioma_origem="en-GB",
        )

        self.assertEqual(page["texts"], [])

    def test_build_page_result_keeps_textured_font_when_font_detection_is_enabled(self):
        block = SimpleNamespace(
            xyxy=(20, 16, 84, 40),
            mask=None,
            confidence=0.88,
        )

        with patch("vision_stack.runtime._should_use_base_white_balloon_font", return_value=False), patch(
            "vision_stack.runtime._get_font_detector",
        ) as get_font_detector:
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=np.full((80, 120, 3), 255, dtype=np.uint8),
                blocks=[block],
                texts=["HELLO"],
                enable_font_detection=True,
            )

        get_font_detector.assert_not_called()
        self.assertEqual(page["texts"][0]["estilo"].get("fonte"), "Newrotic.ttf")
        self.assertEqual(page["texts"][0]["estilo"].get("cor"), "#FFFFFF")

    def test_build_koharu_worker_page_result_passes_rich_text_blocks_to_builder(self):
        worker_payload = {
            "text_blocks": [
                {
                    "bbox": [30, 18, 78, 42],
                    "text": "HELLO",
                    "line_polygons": [
                        [[34, 24], [56, 24], [56, 34], [34, 34]],
                    ],
                    "text_pixel_bbox": [35, 24, 61, 42],
                    "confidence": 0.88,
                    "detector": "paddleocr",
                }
            ],
            "bubble_regions": [],
        }

        with patch("vision_stack.runtime.build_page_result", return_value={"texts": []}) as build_mock:
            _build_koharu_worker_page_result(
                image_rgb=np.full((80, 120, 3), 255, dtype=np.uint8),
                image_label="page.jpg",
                worker_payload=worker_payload,
            )

        build_kwargs = build_mock.call_args.kwargs
        self.assertEqual(build_kwargs["texts"][0]["text"], "HELLO")
        self.assertEqual(build_kwargs["texts"][0]["line_polygons"], worker_payload["text_blocks"][0]["line_polygons"])
        self.assertEqual(build_kwargs["texts"][0]["text_pixel_bbox"], worker_payload["text_blocks"][0]["text_pixel_bbox"])

    def test_should_use_base_white_balloon_font_detects_real_012_bottom_balloon(self):
        image_path = self._fixture_image_path("012__001.jpg")
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

        self.assertTrue(_should_use_base_white_balloon_font(image, [206, 2172, 610, 2301]))
        self.assertFalse(_should_use_base_white_balloon_font(image, [206, 1427, 580, 1550]))

    def test_build_page_result_white_balloon_uses_comicneue_uppercase_without_detector(self):
        block = SimpleNamespace(
            xyxy=(20, 16, 84, 40),
            mask=None,
            confidence=0.88,
        )

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=True), patch(
            "vision_stack.runtime._get_font_detector"
        ) as get_font_detector:
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=np.full((80, 120, 3), 255, dtype=np.uint8),
                blocks=[block],
                texts=["HELLO"],
                enable_font_detection=True,
            )

        get_font_detector.assert_not_called()
        self.assertEqual(page["texts"][0]["estilo"].get("fonte"), "ComicNeue-Bold.ttf")
        self.assertTrue(page["texts"][0]["estilo"].get("force_upper"))

    def test_build_page_result_textured_balloon_uses_fixed_font_without_detector(self):
        block = SimpleNamespace(
            xyxy=(20, 16, 84, 40),
            mask=None,
            confidence=0.88,
        )

        with patch("vision_stack.runtime._should_use_base_white_balloon_font", return_value=False), patch(
            "vision_stack.runtime._get_font_detector",
        ) as get_font_detector:
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=np.full((80, 120, 3), 200, dtype=np.uint8),
                blocks=[block],
                texts=["HELLO"],
                enable_font_detection=True,
            )

        get_font_detector.assert_not_called()
        self.assertEqual(page["texts"][0]["estilo"].get("fonte"), "Newrotic.ttf")
        self.assertEqual(page["texts"][0]["estilo"].get("cor"), "#FFFFFF")

    def test_vision_blocks_to_mask_prefers_precise_mask_and_falls_back_to_bbox(self):
        precise = np.zeros((90, 140), dtype=np.uint8)
        precise[20:40, 28:70] = 255
        blocks = [
            {"bbox": [28, 20, 70, 40], "mask": precise},
            {"bbox": [90, 50, 118, 76], "mask": None},
        ]

        mask = vision_blocks_to_mask((90, 140, 3), blocks)

        self.assertGreater(int(mask[25, 35]), 0)
        self.assertEqual(int(mask[10, 10]), 0)
        self.assertGreater(int(mask[60, 100]), 0)

    def test_vision_blocks_to_mask_uses_refined_shape_when_image_is_available(self):
        image = np.full((90, 140, 3), 245, dtype=np.uint8)
        image[28:32, 40:100] = 20
        image[36:40, 48:92] = 20
        blocks = [
            {"bbox": [30, 20, 110, 50], "mask": None},
        ]

        mask = vision_blocks_to_mask(image.shape, blocks, image_rgb=image, expand_mask=False)

        self.assertGreater(int(mask[30, 60]), 0)
        self.assertEqual(int(mask[21, 34]), 0)
        self.assertEqual(int(mask[46, 106]), 0)

    def test_white_balloon_text_box_cleanup_uses_balloon_bbox_when_available(self):
        original = np.full((120, 160, 3), 245, dtype=np.uint8)
        cleaned = original.copy()
        balloon_mask = np.zeros((120, 160), dtype=np.uint8)
        balloon_mask[20:92, 28:132] = 255
        text = {
            "bbox": [58, 48, 104, 64],
            "balloon_bbox": [28, 20, 132, 92],
            "tipo": "fala",
        }

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=True), patch(
            "vision_stack.runtime._extract_white_balloon_fill_mask",
            return_value=balloon_mask,
        ) as extract_fill_mask, patch(
            "vision_stack.runtime._extract_white_balloon_text_boxes",
            return_value=[[58, 48, 104, 64]],
        ):
            _apply_white_balloon_text_box_cleanup(original, cleaned, [text])

        self.assertEqual(list(extract_fill_mask.call_args.args[1]), [28, 20, 132, 92])

    def test_vision_blocks_to_mask_falls_back_to_full_bbox_without_image(self):
        blocks = [
            {"bbox": [30, 20, 110, 50], "mask": None},
        ]

        mask = vision_blocks_to_mask((90, 140, 3), blocks, image_rgb=None)

        self.assertGreater(int(mask[30, 60]), 0)
        self.assertGreater(int(mask[24, 34]), 0)
        self.assertGreater(int(mask[46, 106]), 0)

    def test_vision_blocks_to_mask_white_balloon_falls_back_when_exact_boxes_are_too_sparse(self):
        image = np.full((120, 160, 3), 245, dtype=np.uint8)
        refined_patch = np.zeros((40, 80), dtype=np.uint8)
        refined_patch[8:32, 12:68] = 255

        with patch('vision_stack.runtime._is_white_balloon_region', return_value=True), \
             patch('vision_stack.runtime._extract_white_balloon_text_boxes', return_value=[[50, 42, 56, 46]]), \
             patch('vision_stack.runtime._build_refined_bbox_mask', return_value=(40, 30, refined_patch)):
            mask = vision_blocks_to_mask(
                image.shape,
                [{"bbox": [40, 30, 120, 70], "mask": None}],
                image_rgb=image,
                expand_mask=False,
            )

        self.assertGreater(int(mask[50, 80]), 0)
        self.assertEqual(int(mask[10, 10]), 0)

    def test_vision_blocks_to_mask_white_balloon_avoids_overbroad_refined_mask(self):
        image = np.full((120, 160, 3), 245, dtype=np.uint8)
        refined_patch = np.zeros((50, 120), dtype=np.uint8)
        refined_patch[5:45, 10:68] = 255
        full_balloon = np.full((120, 160), 255, dtype=np.uint8)

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=True), patch(
            "vision_stack.runtime._extract_white_balloon_text_boxes",
            return_value=[[50, 42, 56, 46]],
        ), patch(
            "vision_stack.runtime._build_refined_bbox_mask",
            return_value=(20, 15, refined_patch),
        ), patch(
            "vision_stack.runtime._extract_white_balloon_fill_mask",
            return_value=full_balloon,
        ):
            mask = vision_blocks_to_mask(
                image.shape,
                [{"bbox": [40, 30, 120, 70], "mask": None}],
                image_rgb=image,
                expand_mask=False,
            )

        self.assertGreater(int(mask[50, 80]), 0)
        self.assertEqual(int(mask[25, 35]), 0)

    def test_vision_blocks_to_mask_splits_real_009_white_balloon_mask_components(self):
        image_path = self._fixture_image_path("009__001.jpg")
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

        mask = vision_blocks_to_mask(
            image.shape,
            [{"bbox": [113, 1514, 705, 1767], "mask": None}],
            image_rgb=image,
            expand_mask=False,
        )
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        self.assertGreaterEqual(num_labels - 1, 2)

    def test_run_detect_ocr_keeps_detector_bbox_without_rescaling(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(xyxy=(10, 20, 50, 40), mask=None, confidence=0.93)

        with patch("vision_stack.runtime.cv2.imread", return_value=image), patch(
            "vision_stack.runtime._get_detector"
        ) as get_detector, patch("vision_stack.runtime._get_ocr_engine") as get_ocr:
            get_detector.return_value.detect.return_value = [block]
            get_detector.return_value.crop.return_value = image[20:40, 10:50]
            get_ocr.return_value._backend = "paddleocr"
            get_ocr.return_value.recognize_blocks_from_page.return_value = ["HELLO"]

            result = run_detect_ocr("page.jpg", profile="quality")

        self.assertEqual(result["texts"][0]["bbox"], [10, 20, 50, 40])

    def test_run_detect_ocr_enables_font_detector_in_default_vision_flow(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(xyxy=(10, 20, 50, 40), mask=None, confidence=0.93)
        captured: dict[str, bool] = {}

        def fake_build_page_result(*args, **kwargs):
            captured["enable_font_detection"] = bool(kwargs.get("enable_font_detection"))
            return {
                "image": "page.jpg",
                "width": 100,
                "height": 100,
                "texts": [],
                "_vision_blocks": [],
            }

        with patch("vision_stack.runtime.cv2.imread", return_value=image), patch(
            "vision_stack.runtime._get_detector"
        ) as get_detector, patch("vision_stack.runtime._get_ocr_engine") as get_ocr, patch(
            "vision_stack.runtime.build_page_result",
            side_effect=fake_build_page_result,
        ):
            get_detector.return_value.detect.return_value = [block]
            get_detector.return_value.crop.return_value = image[20:40, 10:50]
            get_ocr.return_value.recognize_batch.return_value = ["HELLO"]
            get_ocr.return_value._backend = "paddleocr"

            run_detect_ocr("page.jpg", profile="quality")

        self.assertTrue(captured.get("enable_font_detection", False))

    def test_quick_text_presence_check_returns_false_for_blank_page(self):
        image = np.full((1600, 1100, 3), 248, dtype=np.uint8)

        self.assertFalse(_quick_text_presence_check(image))

    def test_quick_text_presence_check_detects_dark_text_on_light_bg(self):
        image = np.full((900, 700, 3), 245, dtype=np.uint8)
        cv2.putText(
            image,
            "HELLO",
            (180, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.2,
            (15, 15, 15),
            8,
            cv2.LINE_AA,
        )

        self.assertTrue(_quick_text_presence_check(image))

    def test_quick_text_presence_check_detects_light_text_on_dark_bg(self):
        image = np.full((900, 700, 3), 40, dtype=np.uint8)
        cv2.putText(
            image,
            "NO WAY",
            (120, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.6,
            (245, 245, 245),
            7,
            cv2.LINE_AA,
        )

        self.assertTrue(_quick_text_presence_check(image))

    def test_run_detect_ocr_skips_detector_and_ocr_when_quick_scan_finds_no_text(self):
        image = np.full((120, 120, 3), 250, dtype=np.uint8)
        events: list[tuple[str, float, str]] = []

        with patch("vision_stack.runtime.cv2.imread", return_value=image), patch(
            "vision_stack.runtime._quick_text_presence_check",
            return_value=False,
        ) as quick_check, patch(
            "vision_stack.runtime._get_detector",
            side_effect=AssertionError("detector nao deveria carregar"),
        ), patch(
            "vision_stack.runtime._get_ocr_engine",
            side_effect=AssertionError("ocr nao deveria carregar"),
        ):
            result = run_detect_ocr(
                "page.jpg",
                profile="quality",
                progress_callback=lambda stage, progress, message: events.append((stage, progress, message)),
            )

        quick_check.assert_called_once()
        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])
        self.assertTrue(bool(result.get("quick_skipped_no_text")))
        self.assertTrue(bool(result.get("sem_texto_detectado")))
        self.assertEqual(events[0][0], "prepare_image")
        self.assertEqual(events[-1][0], "complete")
        self.assertIn("sem texto", events[-1][2].lower())

    def test_run_detect_ocr_reports_granular_progress(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(xyxy=(10, 20, 50, 40), mask=None, confidence=0.93)
        events: list[tuple[str, float, str]] = []

        def fake_build_page_result(*args, **kwargs):
            callback = kwargs.get("progress_callback")
            if callback:
                callback("font_detection", 0.92, "Analisando fonte")
                callback("finalize_blocks", 0.98, "Finalizando blocos")
            return {
                "image": "page.jpg",
                "width": 100,
                "height": 100,
                "texts": [{"bbox": [10, 20, 50, 40], "text": "HELLO"}],
                "_vision_blocks": [],
            }

        with patch("vision_stack.runtime.cv2.imread", return_value=image), patch(
            "vision_stack.runtime._get_detector"
        ) as get_detector, patch("vision_stack.runtime._get_ocr_engine") as get_ocr, patch(
            "vision_stack.runtime.build_page_result",
            side_effect=fake_build_page_result,
        ):
            get_detector.return_value.detect.return_value = [block]
            get_detector.return_value.crop.return_value = image[20:40, 10:50]
            get_ocr.return_value.recognize_batch.return_value = ["HELLO"]
            get_ocr.return_value._backend = "paddleocr"

            run_detect_ocr(
                "page.jpg",
                profile="quality",
                progress_callback=lambda stage, progress, message: events.append((stage, progress, message)),
            )

        self.assertEqual(
            [stage for stage, _, _ in events[:5]],
            [
                "prepare_image",
                "load_detector",
                "load_ocr_engine",
                "detect_text",
                "recognize_text",
            ],
        )
        self.assertEqual(events[-1][0], "complete")
        self.assertGreaterEqual(events[-1][1], 1.0)
        self.assertIn("Finalizando", events[-2][2])

    def test_run_detect_ocr_prefers_koharu_worker_when_path_present(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        worker_page = {
            "image": "page.jpg",
            "width": 100,
            "height": 100,
            "texts": [{"bbox": [10, 20, 50, 40], "text": "HELLO"}],
            "_vision_blocks": [{"bbox": [10, 20, 50, 40], "mask": None, "confidence": 0.93}],
        }

        with patch("vision_stack.runtime.cv2.imread", return_value=image), patch(
            "vision_stack.runtime._run_koharu_worker_detect_ocr",
            return_value=worker_page,
            create=True,
        ) as run_worker, patch(
            "vision_stack.runtime._run_detect_ocr_on_image",
            side_effect=AssertionError("nao deveria usar o stack antigo quando o worker novo estiver disponivel"),
        ):
            result = run_detect_ocr(
                "page.jpg",
                profile="quality",
                vision_worker_path="D:/mangatl/vision-worker/target/debug/traduzai-vision.exe",
            )

        run_worker.assert_called_once()
        self.assertEqual(result["texts"][0]["text"], "HELLO")

    def test_run_detect_ocr_falls_back_to_current_stack_when_koharu_worker_fails(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        fallback_page = {
            "image": "page.jpg",
            "width": 100,
            "height": 100,
            "texts": [{"bbox": [12, 18, 54, 42], "text": "FALLBACK"}],
            "_vision_blocks": [{"bbox": [12, 18, 54, 42], "mask": None, "confidence": 0.81}],
        }

        with patch("vision_stack.runtime.cv2.imread", return_value=image), patch(
            "vision_stack.runtime._run_koharu_worker_detect_ocr",
            side_effect=RuntimeError("worker falhou"),
            create=True,
        ) as run_worker, patch(
            "vision_stack.runtime._run_detect_ocr_on_image",
            return_value=fallback_page,
        ) as run_current_stack:
            result = run_detect_ocr(
                "page.jpg",
                profile="quality",
                vision_worker_path="D:/mangatl/vision-worker/target/debug/traduzai-vision.exe",
            )

        run_worker.assert_called_once()
        run_current_stack.assert_called_once()
        self.assertEqual(result["texts"][0]["text"], "FALLBACK")

    def test_build_koharu_worker_page_result_accepts_camel_case_payload(self):
        image = np.full((120, 160, 3), 255, dtype=np.uint8)

        page = _build_koharu_worker_page_result(
            image_rgb=image,
            image_label="page.jpg",
            worker_payload={
                "textBlocks": [
                    {
                        "bbox": [20, 30, 80, 54],
                        "confidence": 0.91,
                        "text": "HELLO THERE",
                        "detector": "comic-text-bubble-detector",
                    }
                ],
                "bubbleRegions": [
                    {
                        "bbox": [12, 18, 96, 70],
                        "confidence": 0.82,
                    }
                ],
            },
            profile="quality",
        )

        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["text"], "HELLO THERE")
        self.assertEqual(page["_bubble_regions"][0]["bbox"], [12, 18, 96, 70])

    def test_warmup_visual_stack_primes_detector_ocr_and_font_detector(self):
        detector_calls: list[str] = []
        ocr_calls: list[int] = []
        font_calls: list[bool] = []

        class FakeDetector:
            def detect(self, image_rgb, conf_threshold=0.5):
                detector_calls.append(f"detect:{image_rgb.shape[1]}x{image_rgb.shape[0]}")
                return []

        class FakeOcr:
            def recognize_batch(self, crops):
                ocr_calls.append(len(crops))
                return ["HELLO"]

        class FakeFontDetector:
            def detect(self, region, allow_default=True):
                font_calls.append(bool(allow_default))
                return "DK Full Blast.otf"

        with patch("vision_stack.runtime._configure_model_roots") as configure_roots, patch(
            "vision_stack.runtime._get_detector",
            return_value=FakeDetector(),
        ), patch(
            "vision_stack.runtime._get_ocr_engine",
            return_value=FakeOcr(),
        ), patch(
            "vision_stack.runtime._get_font_detector",
            return_value=FakeFontDetector(),
        ):
            warmup_visual_stack(models_dir="models", profile="quality")

        configure_roots.assert_called_once_with("models")
        self.assertEqual(detector_calls, ["detect:256x256"])
        self.assertEqual(ocr_calls, [1])
        self.assertEqual(font_calls, [False])

    def test_is_white_balloon_region_detects_clean_bright_area(self):
        image = np.full((120, 140, 3), 250, dtype=np.uint8)
        image[44:76, 35:105] = 246

        self.assertTrue(_is_white_balloon_region(image, [40, 46, 100, 74]))

    def test_apply_white_text_overlay_covers_text_bbox_only(self):
        image = np.full((90, 140, 3), 240, dtype=np.uint8)
        image[32:48, 50:90] = 10

        overlaid = _apply_white_text_overlay(image, [50, 32, 90, 48])

        self.assertGreater(int(np.mean(overlaid[38, 70])), 235)
        self.assertEqual(int(overlaid[10, 10, 0]), 240)

    def test_apply_white_text_overlay_rounds_patch_corners(self):
        image = np.full((90, 140, 3), 240, dtype=np.uint8)
        image[32:48, 50:90] = 10

        overlaid = _apply_white_text_overlay(image, [50, 32, 90, 48])

        self.assertEqual(int(overlaid[27, 46, 0]), 240)
        self.assertGreater(int(overlaid[38, 70, 0]), 235)

    def test_apply_letter_white_boxes_targets_bright_text_region(self):
        image = np.full((100, 180, 3), 245, dtype=np.uint8)
        image[35:55, 55:125] = 20

        overlaid = _apply_letter_white_boxes(
            image,
            {"text": "HELLO", "bbox": [55, 35, 125, 55]},
        )

        self.assertGreater(int(np.mean(overlaid[44, 90])), 235)
        self.assertEqual(int(overlaid[10, 10, 0]), 245)

    def test_apply_white_balloon_fill_stays_local_to_balloon_shape(self):
        image = np.full((220, 180, 3), 230, dtype=np.uint8)
        cv2.ellipse(image, (90, 110), (48, 28), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(image, (90, 110), (48, 28), 0, 0, 360, (30, 30, 30), 2)
        image[96:104, 60:120] = 10
        image[109:117, 55:125] = 10

        filled = _apply_white_balloon_fill(image, [55, 96, 125, 117])

        self.assertGreater(int(filled[110, 90, 0]), 240)
        self.assertEqual(int(filled[30, 30, 0]), 230)
        self.assertEqual(int(filled[150, 90, 0]), 230)
        self.assertLess(int(filled[110, 43, 0]), 120)

    def test_apply_white_balloon_fill_preserves_lower_outline(self):
        image = np.full((220, 180, 3), 235, dtype=np.uint8)
        cv2.ellipse(image, (90, 110), (56, 34), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(image, (90, 110), (56, 34), 0, 0, 360, (20, 20, 20), 2)
        image[100:118, 55:125] = 15

        filled = _apply_white_balloon_fill(image, [52, 96, 128, 121])

        self.assertGreater(int(filled[110, 90, 0]), 242)
        self.assertLess(int(filled[143, 90, 0]), 90)

    def test_apply_white_balloon_artifact_cleanup_removes_internal_dark_residue(self):
        original = np.full((220, 220, 3), 230, dtype=np.uint8)
        cv2.ellipse(original, (110, 110), (62, 38), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(original, (110, 110), (62, 38), 0, 0, 360, (20, 20, 20), 2)
        cleaned = original.copy()
        cleaned[106:110, 78:142] = 55
        cleaned[112:114, 88:132] = 65

        result = _apply_white_balloon_artifact_cleanup(
            original,
            cleaned,
            [{"bbox": [74, 96, 146, 124], "skip_processing": False}],
        )

        self.assertGreater(int(result[108, 110, 0]), 220)
        self.assertLessEqual(abs(int(result[110, 48, 0]) - int(original[110, 48, 0])), 8)
        self.assertEqual(int(result[20, 20, 0]), 230)

    def test_apply_white_balloon_artifact_cleanup_preserves_connected_balloon_neck(self):
        original = np.full((260, 240, 3), 228, dtype=np.uint8)
        cv2.ellipse(original, (120, 92), (58, 32), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(original, (92, 148), (84, 46), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(original, (120, 92), (58, 32), 0, 0, 360, (22, 22, 22), 2)
        cv2.ellipse(original, (92, 148), (84, 46), 0, 0, 360, (22, 22, 22), 2)
        cleaned = original.copy()
        cleaned[88:92, 90:150] = 48
        cleaned[140:144, 54:146] = 52

        result = _apply_white_balloon_artifact_cleanup(
            original,
            cleaned,
            [
                {"bbox": [84, 80, 156, 108], "skip_processing": False},
                {"bbox": [52, 132, 150, 168], "skip_processing": False},
            ],
        )

        self.assertGreater(int(result[90, 120, 0]), 220)
        self.assertGreater(int(result[142, 100, 0]), 220)
        self.assertLessEqual(abs(int(result[120, 166, 0]) - int(original[120, 166, 0])), 8)

    def test_apply_white_balloon_line_artifact_cleanup_removes_internal_horizontal_line(self):
        original = np.full((260, 260, 3), 230, dtype=np.uint8)
        cv2.ellipse(original, (130, 130), (76, 44), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(original, (130, 130), (76, 44), 0, 0, 360, (20, 20, 20), 2)
        cleaned = original.copy()
        cleaned[127:131, 72:188] = 150

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
            result = _apply_white_balloon_line_artifact_cleanup(
                original,
                cleaned,
                [{"bbox": [82, 108, 178, 148], "skip_processing": False}],
            )

        self.assertGreater(int(result[129, 130, 0]), 220)
        self.assertLessEqual(abs(int(result[130, 53, 0]) - int(original[130, 53, 0])), 8)

    def test_apply_white_balloon_line_artifact_cleanup_preserves_connected_balloon_border(self):
        original = np.full((300, 280, 3), 230, dtype=np.uint8)
        cv2.ellipse(original, (150, 104), (62, 34), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(original, (116, 172), (98, 52), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(original, (150, 104), (62, 34), 0, 0, 360, (20, 20, 20), 2)
        cv2.ellipse(original, (116, 172), (98, 52), 0, 0, 360, (20, 20, 20), 2)
        cleaned = original.copy()
        cleaned[102:106, 102:198] = 145

        result = _apply_white_balloon_line_artifact_cleanup(
            original,
            cleaned,
            [
                {"bbox": [100, 92, 200, 118], "skip_processing": False},
                {"bbox": [58, 158, 176, 186], "skip_processing": False},
            ],
        )

        self.assertGreater(int(result[104, 150, 0]), 220)
        self.assertLessEqual(abs(int(result[142, 207, 0]) - int(original[142, 207, 0])), 8)

    def test_extract_white_balloon_text_boxes_splits_multiline_text(self):
        image = np.full((220, 220, 3), 250, dtype=np.uint8)
        cv2.ellipse(image, (110, 110), (72, 48), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(image, (110, 110), (72, 48), 0, 0, 360, (25, 25, 25), 2)
        image[84:92, 78:146] = 20
        image[102:110, 60:160] = 20
        image[120:128, 88:140] = 20

        boxes = _extract_white_balloon_text_boxes(image, [58, 80, 162, 130])

        self.assertEqual(len(boxes), 3)
        self.assertLessEqual(sum(abs(a - b) for a, b in zip(boxes[0], [78, 84, 146, 92])), 4)
        self.assertLessEqual(sum(abs(a - b) for a, b in zip(boxes[1], [60, 102, 160, 110])), 4)
        self.assertLessEqual(sum(abs(a - b) for a, b in zip(boxes[2], [88, 120, 140, 128])), 4)

    def test_extract_white_balloon_text_boxes_splits_real_009_balloon_lines(self):
        image_path = self._fixture_image_path("009__001.jpg")
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

        boxes = _extract_white_balloon_text_boxes(image, [113, 1514, 705, 1767])

        self.assertGreaterEqual(len(boxes), 2)

    def test_apply_white_balloon_text_box_cleanup_uses_exact_boxes_without_border_leak(self):
        original = np.full((220, 220, 3), 250, dtype=np.uint8)
        cv2.ellipse(original, (110, 110), (72, 48), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(original, (110, 110), (72, 48), 0, 0, 360, (25, 25, 25), 2)
        original[84:92, 78:146] = 20
        original[102:110, 60:160] = 20
        original[120:128, 88:140] = 20
        cleaned = original.copy()
        cleaned[84:92, 78:146] = 160
        cleaned[102:110, 60:160] = 150
        cleaned[120:128, 88:140] = 155

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
            result = _apply_white_balloon_text_box_cleanup(
                original,
                cleaned,
                [{"bbox": [58, 80, 162, 130], "skip_processing": False}],
            )

        self.assertGreater(int(result[88, 110, 0]), 245)
        self.assertGreater(int(result[106, 110, 0]), 245)
        self.assertGreater(int(result[124, 110, 0]), 245)
        self.assertEqual(int(result[110, 37, 0]), int(original[110, 37, 0]))

    def test_apply_white_balloon_text_box_cleanup_clips_box_to_balloon_interior(self):
        original = np.full((220, 220, 3), 250, dtype=np.uint8)
        cleaned = original.copy()
        cleaned[70:95, 70:160] = 180
        fake_balloon_mask = np.zeros((220, 220), dtype=np.uint8)
        cv2.ellipse(fake_balloon_mask, (116, 88), (20, 8), 0, 0, 360, 255, -1)

        with patch(
            "vision_stack.runtime._extract_white_balloon_text_boxes",
            return_value=[[70, 70, 160, 95]],
        ), patch(
            "vision_stack.runtime._extract_white_balloon_fill_mask",
            return_value=fake_balloon_mask,
        ), patch(
            "vision_stack.runtime._is_white_balloon_region",
            return_value=True,
        ):
            result = _apply_white_balloon_text_box_cleanup(
                original,
                cleaned,
                [{"bbox": [86, 88, 144, 116], "skip_processing": False}],
            )

        self.assertEqual(int(result[75, 75, 0]), 180)
        self.assertGreater(int(result[88, 110, 0]), 240)
        self.assertEqual(int(result[80, 97, 0]), 180)

    def test_apply_white_balloon_text_box_cleanup_rounds_box_corners(self):
        original = np.full((220, 220, 3), 250, dtype=np.uint8)
        cleaned = original.copy()
        cleaned[70:95, 70:160] = 180
        fake_balloon_mask = np.zeros((220, 220), dtype=np.uint8)
        fake_balloon_mask[70:95, 70:160] = 255

        with patch(
            "vision_stack.runtime._extract_white_balloon_text_boxes",
            return_value=[[70, 70, 160, 95]],
        ), patch(
            "vision_stack.runtime._extract_white_balloon_fill_mask",
            return_value=fake_balloon_mask,
        ), patch(
            "vision_stack.runtime._is_white_balloon_region",
            return_value=True,
        ):
            result = _apply_white_balloon_text_box_cleanup(
                original,
                cleaned,
                [{"bbox": [86, 88, 144, 116], "skip_processing": False}],
            )

        self.assertEqual(int(result[71, 71, 0]), 180)
        self.assertGreater(int(result[82, 110, 0]), 240)

    def test_white_balloon_postprocess_skips_textured_regions(self):
        original = np.full((220, 220, 3), (130, 45, 45), dtype=np.uint8)
        cleaned = original.copy()
        cleaned[92:104, 84:136] = (220, 220, 220)

        bbox = [78, 88, 142, 108]
        texts = [{"bbox": bbox, "skip_processing": False}]

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=False):
            text_box = _apply_white_balloon_text_box_cleanup(original, cleaned, texts)
            micro = _apply_white_balloon_micro_artifact_cleanup(original, cleaned, texts)
            line = _apply_white_balloon_line_artifact_cleanup(original, cleaned, texts)

        self.assertTrue(np.array_equal(text_box, cleaned))
        self.assertTrue(np.array_equal(micro, cleaned))
        self.assertTrue(np.array_equal(line, cleaned))

    def test_apply_white_balloon_micro_artifact_cleanup_removes_tiny_dark_traces_inside_balloon(self):
        original = np.full((260, 260, 3), 248, dtype=np.uint8)
        cv2.ellipse(original, (130, 130), (82, 56), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(original, (130, 130), (82, 56), 0, 0, 360, (25, 25, 25), 2)
        cleaned = original.copy()
        cleaned[112:126, 78:84] = 70
        cleaned[150:158, 168:174] = 80

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
            result = _apply_white_balloon_micro_artifact_cleanup(
                original,
                cleaned,
                [{"bbox": [58, 88, 204, 172], "skip_processing": False}],
            )

        self.assertGreater(int(result[118, 81, 0]), 220)
        self.assertGreater(int(result[154, 171, 0]), 220)
        self.assertLessEqual(abs(int(result[130, 48, 0]) - int(original[130, 48, 0])), 8)

    def test_merge_text_fragments_inserts_residual_word_in_middle(self):
        merged = _merge_text_fragments(
            "o que ha nisso",
            "demais",
            [100, 100, 300, 130],
            [210, 102, 250, 128],
        )

        self.assertEqual(merged, "o que ha demais nisso")

    def test_extract_white_balloon_fill_mask_expands_beyond_partial_text_bbox(self):
        image = np.full((180, 180, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (90, 90), (55, 35), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(image, (90, 90), (55, 35), 0, 0, 360, (30, 30, 30), 2)
        image[82:96, 75:105] = 10

        mask = _extract_white_balloon_fill_mask(image, [78, 80, 108, 98])

        self.assertGreater(int(mask[90, 90]), 0)
        self.assertGreater(int(mask[90, 70]), 0)
        self.assertEqual(int(mask[30, 30]), 0)

    def test_extract_white_balloon_fill_mask_closes_internal_holes_on_real_010_balloon(self):
        image_path = self._fixture_image_path("010__001.jpg")
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

        mask = _extract_white_balloon_fill_mask(image, [248, 165, 541, 269])

        self.assertGreater(int(mask[98, 338]), 0)
        self.assertGreater(int(mask[95, 438]), 0)
        self.assertEqual(int(mask[340, 80]), 0)

    def test_run_masked_inpaint_passes_prefers_single_full_image_pass_by_default(self):
        calls = []

        class FakeInpainter:
            def inpaint(self, image_np, mask, batch_size=4, debug=None, force_no_tiling=False):
                calls.append(
                    {
                        "shape": image_np.shape,
                        "mask_nonzero": int(np.count_nonzero(mask)),
                        "batch_size": batch_size,
                        "force_no_tiling": force_no_tiling,
                    }
                )
                return image_np

        image = np.full((80, 100, 3), 127, dtype=np.uint8)
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:40, 30:70] = 255

        result = _run_masked_inpaint_passes(FakeInpainter(), image, mask, batch_size=4)

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["force_no_tiling"])
        self.assertEqual(result["final_output"].shape, image.shape)

    def test_run_masked_inpaint_passes_can_still_use_multi_pass_when_requested(self):
        calls = []

        class FakeInpainter:
            def inpaint(self, image_np, mask, batch_size=4, debug=None, force_no_tiling=False):
                calls.append(force_no_tiling)
                return image_np

        image = np.full((80, 100, 3), 127, dtype=np.uint8)
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:40, 30:70] = 255

        _run_masked_inpaint_passes(
            FakeInpainter(),
            image,
            mask,
            batch_size=4,
            multi_pass=True,
            force_no_tiling=False,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls, [False, False])

    def test_build_residual_cleanup_mask_targets_dark_rectangular_seams(self):
        image = np.full((120, 160, 3), 120, dtype=np.uint8)
        base_mask = np.zeros((120, 160), dtype=np.uint8)
        base_mask[35:85, 40:120] = 255
        image[34:36, 40:120] = 5
        image[84:86, 40:120] = 5

        cleanup = _build_residual_cleanup_mask(image, base_mask)

        self.assertGreater(int(cleanup[35, 70]), 0)
        self.assertGreater(int(cleanup[84, 70]), 0)
        self.assertEqual(int(cleanup[10, 10]), 0)

    def test_build_bright_zone_line_mask_detects_horizontal_dark_line(self):
        image = np.full((120, 180, 3), 245, dtype=np.uint8)
        image[58:60, 25:155] = 40
        mask = _build_bright_zone_line_mask(image)

        self.assertGreater(int(mask[59, 90]), 0)
        self.assertEqual(int(mask[20, 20]), 0)

    def test_apply_bright_zone_line_cleanup_removes_faint_horizontal_seam(self):
        image = np.full((140, 200, 3), 246, dtype=np.uint8)
        image[68:70, 18:182] = 150

        cleaned = _apply_bright_zone_line_cleanup(image)

        self.assertGreater(int(np.mean(cleaned[69, 100])), 235)
        self.assertEqual(int(cleaned[20, 20, 0]), 246)

    def test_build_mask_boundary_seam_mask_detects_top_and_bottom_seams(self):
        image = np.full((180, 220, 3), 120, dtype=np.uint8)
        base_mask = np.zeros((180, 220), dtype=np.uint8)
        base_mask[60:120, 40:180] = 255
        image[60:62, 52:168] = 35
        image[118:120, 48:172] = 35

        seam_mask = _build_mask_boundary_seam_mask(image, base_mask)

        self.assertGreater(int(seam_mask[61, 110]), 0)
        self.assertGreater(int(seam_mask[119, 110]), 0)
        self.assertEqual(int(seam_mask[20, 20]), 0)

    def test_apply_mask_boundary_seam_cleanup_removes_boundary_lines(self):
        image = np.full((180, 220, 3), 122, dtype=np.uint8)
        base_mask = np.zeros((180, 220), dtype=np.uint8)
        base_mask[60:120, 40:180] = 255
        image[60:62, 52:168] = 28
        image[118:120, 48:172] = 28

        cleaned = _apply_mask_boundary_seam_cleanup(image, base_mask)

        self.assertGreater(int(np.mean(cleaned[61, 110])), 80)
        self.assertGreater(int(np.mean(cleaned[119, 110])), 80)
        self.assertEqual(int(cleaned[20, 20, 0]), 122)

    def test_apply_textured_balloon_seam_cleanup_removes_bbox_edge_seam(self):
        original = np.zeros((260, 360, 3), dtype=np.uint8)
        for x in range(original.shape[1]):
            tone = 72 + (x % 18)
            original[:, x] = [tone + 42, 8, 12]
        for y in range(original.shape[0]):
            original[y] = np.clip(original[y].astype(np.int16) + (y % 9), 0, 255).astype(np.uint8)

        bbox = [74, 92, 286, 176]
        expanded = _expand_bbox(
            bbox,
            original.shape,
            pad_x_ratio=0.06,
            pad_y_ratio=0.10,
            min_pad_x=14,
            min_pad_y=12,
        )
        seam_top = expanded[1] + 1
        seam_bottom = expanded[3] - 2

        cleaned = original.copy()
        cleaned[seam_top : seam_top + 2, expanded[0] + 18 : expanded[2] - 18] = [16, 0, 0]
        cleaned[seam_bottom : seam_bottom + 2, expanded[0] + 14 : expanded[2] - 14] = [18, 0, 0]

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=False):
            result = _apply_textured_balloon_seam_cleanup(
                original,
                cleaned,
                [{"bbox": bbox, "skip_processing": False}],
            )

        self.assertGreater(int(np.mean(result[seam_top + 1, 180])), 40)
        self.assertGreater(int(np.mean(result[seam_bottom + 1, 180])), 40)
        self.assertLessEqual(abs(int(result[30, 30, 0]) - int(original[30, 30, 0])), 4)

    def test_apply_textured_balloon_seam_cleanup_preserves_panel_border_outside_balloon(self):
        original = np.full((220, 260, 3), 252, dtype=np.uint8)
        original[100:102, :] = [18, 18, 18]
        for x in range(0, original.shape[1], 10):
            original[102:, x : x + 3] = [40, 70, 150]

        balloon_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cv2.ellipse(balloon_mask, (130, 74), (60, 36), 0, 0, 360, 255, -1)
        original[balloon_mask > 0] = [135, 20, 32]
        original[:, :, 0] = np.where(balloon_mask > 0, np.clip(original[:, :, 0] + (np.indices(original.shape[:2])[1] % 11), 0, 255), original[:, :, 0])

        bbox = [95, 58, 165, 88]
        expanded = _expand_bbox(
            bbox,
            original.shape,
            pad_x_ratio=0.06,
            pad_y_ratio=0.10,
            min_pad_x=14,
            min_pad_y=12,
        )
        seam_y = expanded[3] - 2

        cleaned = original.copy()
        cleaned[seam_y : seam_y + 2, expanded[0] + 4 : expanded[2] - 4] = [12, 0, 0]

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=False):
            result = _apply_textured_balloon_seam_cleanup(
                original,
                cleaned,
                [{"bbox": bbox, "skip_processing": False, "tipo": "fala", "confidence": 0.92}],
            )

        self.assertGreater(int(np.mean(result[seam_y + 1, 130])), 40)
        self.assertLessEqual(abs(int(result[seam_y + 1, 84, 0]) - int(original[seam_y + 1, 84, 0])), 10)
        self.assertLessEqual(abs(int(result[seam_y + 1, 84, 1]) - int(original[seam_y + 1, 84, 1])), 10)

    def test_apply_textured_balloon_band_artifact_cleanup_softens_internal_dark_band(self):
        original = np.full((260, 320, 3), 248, dtype=np.uint8)
        balloon_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cv2.ellipse(balloon_mask, (160, 126), (98, 64), 0, 0, 360, 255, -1)

        yy = np.indices(original.shape[:2], dtype=np.float32)[0]
        top_color = np.array([178, 34, 58], dtype=np.float32)
        bottom_color = np.array([30, 5, 7], dtype=np.float32)
        ratio = np.clip((yy - 62.0) / max(1.0, 188.0 - 62.0), 0.0, 1.0)[..., None]
        gradient = (top_color * (1.0 - ratio) + bottom_color * ratio).astype(np.uint8)
        original[balloon_mask > 0] = gradient[balloon_mask > 0]

        cleaned = original.copy()
        cleaned[114:166, 90:232] = [48, 6, 7]

        text = {
            "bbox": [98, 96, 222, 146],
            "balloon_bbox": [62, 62, 258, 190],
            "confidence": 0.93,
            "tipo": "fala",
            "skip_processing": False,
        }

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=False):
            result = _apply_textured_balloon_band_artifact_cleanup(
                original,
                cleaned,
                [text],
            )

        self.assertGreater(int(result[132, 160, 0]), int(cleaned[132, 160, 0]) + 20)
        self.assertGreater(int(result[132, 160, 1]), int(cleaned[132, 160, 1]) + 5)
        self.assertLessEqual(abs(int(result[132, 28, 0]) - int(original[132, 28, 0])), 3)
        self.assertLessEqual(abs(int(result[132, 28, 1]) - int(original[132, 28, 1])), 3)

    def test_build_residual_cleanup_mask_catches_internal_dark_trace(self):
        image = np.full((120, 160, 3), 140, dtype=np.uint8)
        base_mask = np.zeros((120, 160), dtype=np.uint8)
        base_mask[35:85, 40:120] = 255
        image[58:60, 48:112] = 8

        cleanup = _build_residual_cleanup_mask(image, base_mask)

        self.assertGreater(int(cleanup[59, 80]), 0)
        self.assertEqual(int(cleanup[12, 12]), 0)

    def test_build_residual_cleanup_mask_catches_relative_dark_trace_on_midtone_bg(self):
        image = np.full((120, 180, 3), 142, dtype=np.uint8)
        base_mask = np.zeros((120, 180), dtype=np.uint8)
        base_mask[30:92, 42:138] = 255
        image[59:61, 50:130] = 103

        cleanup = _build_residual_cleanup_mask(image, base_mask)

        self.assertGreater(int(cleanup[60, 88]), 0)
        self.assertEqual(int(cleanup[8, 8]), 0)

    def test_merge_nearby_bboxes_combines_multiline_balloon_text(self):
        boxes = [
            [267, 2212, 550, 2250],
            [172, 2250, 647, 2285],
            [322, 2321, 492, 2351],
        ]

        merged = _merge_nearby_bboxes(boxes, gap_x=80, gap_y=46)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0], [172, 2212, 647, 2351])

    def test_enlarge_koharu_window_matches_reference_ratio(self):
        enlarged = _enlarge_koharu_window([10, 20, 50, 60], 200, 150)

        self.assertEqual(enlarged, [4, 14, 56, 66])

    def test_try_koharu_balloon_fill_fills_simple_flat_balloon(self):
        image = np.full((96, 120, 3), 245, dtype=np.uint8)
        balloon_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.ellipse(balloon_mask, (60, 48), (34, 20), 0, 0, 360, 255, -1)
        image[balloon_mask > 0] = [166, 28, 40]
        cv2.ellipse(image, (60, 48), (34, 20), 0, 0, 360, (28, 6, 6), 2)

        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        text_mask[42:54, 38:82] = 255

        filled = _try_koharu_balloon_fill(image, text_mask)

        self.assertIsNotNone(filled)
        self.assertGreater(int(filled[48, 60, 0]), 140)
        self.assertLessEqual(abs(int(filled[10, 10, 0]) - int(image[10, 10, 0])), 2)

    def test_run_koharu_blockwise_inpaint_page_skips_model_for_simple_balloon(self):
        image = np.full((180, 180, 3), 245, dtype=np.uint8)
        balloon_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.ellipse(balloon_mask, (90, 84), (54, 30), 0, 0, 360, 255, -1)
        image[balloon_mask > 0] = [170, 34, 46]
        cv2.ellipse(image, (90, 84), (54, 30), 0, 0, 360, (18, 4, 4), 2)

        ocr_data = {
            "texts": [
                {"bbox": [60, 72, 120, 96], "skip_processing": False, "tipo": "fala"},
            ],
            "_vision_blocks": [
                {"bbox": [60, 72, 120, 96], "mask": None, "confidence": 0.93},
            ],
        }

        class FakeInpainter:
            def __init__(self):
                self.calls = []

            def inpaint(self, image_np, mask, batch_size=4, debug=None, force_no_tiling=False):
                self.calls.append((tuple(image_np.shape), int(np.count_nonzero(mask))))
                return image_np

        inpainter = FakeInpainter()

        result = _run_koharu_blockwise_inpaint_page(image, ocr_data, inpainter)

        self.assertEqual(inpainter.calls, [])
        self.assertGreater(int(result[84, 90, 0]), 140)
        self.assertLessEqual(abs(int(result[15, 15, 0]) - int(image[15, 15, 0])), 2)

    def test_run_koharu_blockwise_inpaint_page_uses_cropped_model_window_for_textured_balloon(self):
        image = np.full((220, 240, 3), 252, dtype=np.uint8)
        balloon_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.ellipse(balloon_mask, (122, 110), (70, 42), 0, 0, 360, 255, -1)
        yy, xx = np.indices(image.shape[:2])
        image[balloon_mask > 0] = np.stack(
            [
                110 + ((xx[balloon_mask > 0] * 3 + yy[balloon_mask > 0]) % 35),
                18 + ((yy[balloon_mask > 0] * 2) % 12),
                24 + (xx[balloon_mask > 0] % 10),
            ],
            axis=1,
        )
        cv2.ellipse(image, (122, 110), (70, 42), 0, 0, 360, (18, 6, 6), 2)

        ocr_data = {
            "texts": [
                {"bbox": [92, 96, 152, 124], "skip_processing": False, "tipo": "fala"},
            ],
            "_vision_blocks": [
                {"bbox": [92, 96, 152, 124], "mask": None, "confidence": 0.92},
            ],
        }

        class FakeInpainter:
            def __init__(self):
                self.calls = []

            def inpaint(self, image_np, mask, batch_size=4, debug=None, force_no_tiling=False):
                self.calls.append((tuple(image_np.shape), int(np.count_nonzero(mask)), bool(force_no_tiling)))
                result = image_np.copy()
                result[mask > 0] = [210, 36, 42]
                return result

        inpainter = FakeInpainter()

        result = _run_koharu_blockwise_inpaint_page(image, ocr_data, inpainter)

        self.assertEqual(len(inpainter.calls), 1)
        self.assertLess(inpainter.calls[0][0][0], image.shape[0])
        self.assertLess(inpainter.calls[0][0][1], image.shape[1])
        self.assertTrue(inpainter.calls[0][2])
        self.assertGreater(int(result[108, 122, 0]), int(image[108, 122, 0]))
        self.assertLessEqual(abs(int(result[30, 30, 0]) - int(image[30, 30, 0])), 2)

    def test_run_inpaint_pages_applies_white_balloon_cleanup_stack_after_lama(self):
        image = np.full((180, 180, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (90, 90), (55, 35), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(image, (90, 90), (55, 35), 0, 0, 360, (25, 25, 25), 2)
        image[72:80, 65:120] = 10
        image[87:95, 60:125] = 10
        image[102:108, 72:112] = 10

        ocr_data = {
            "texts": [
                {"bbox": [65, 72, 120, 80], "skip_processing": False},
                {"bbox": [60, 87, 125, 95], "skip_processing": False},
                {"bbox": [72, 102, 112, 108], "skip_processing": False},
            ],
            "_vision_blocks": [
                {"bbox": [65, 72, 120, 80], "mask": None},
                {"bbox": [60, 87, 125, 95], "mask": None},
                {"bbox": [72, 102, 112, 108], "mask": None},
            ],
        }

        class FakeInpainter:
            def inpaint(self, image_np, mask, batch_size=4):
                return image_np.copy()

        def fake_line_artifact_cleanup(original_rgb, cleaned_rgb, texts):
            result = cleaned_rgb.copy()
            result[90, 90] = [210, 210, 210]
            result[105, 90] = [205, 205, 205]
            return result

        def fake_text_box_cleanup(original_rgb, cleaned_rgb, texts):
            result = cleaned_rgb.copy()
            result[90, 90] = [252, 252, 252]
            result[105, 90] = [248, 248, 248]
            return result

        def fake_micro_cleanup(original_rgb, cleaned_rgb, texts):
            result = cleaned_rgb.copy()
            result[90, 90] = [255, 255, 255]
            result[105, 90] = [255, 255, 255]
            return result

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "page.jpg"
            output_dir = Path(tmpdir) / "out"
            Image.fromarray(image).save(image_path)
            with patch("vision_stack.runtime._get_inpainter", return_value=FakeInpainter()), patch(
                "vision_stack.runtime._apply_white_balloon_line_artifact_cleanup",
              side_effect=fake_line_artifact_cleanup,
            ), patch(
                "vision_stack.runtime._apply_white_balloon_text_box_cleanup",
                side_effect=fake_text_box_cleanup,
            ), patch(
                "vision_stack.runtime._apply_white_balloon_micro_artifact_cleanup",
                side_effect=fake_micro_cleanup,
            ):
                outputs = run_inpaint_pages([image_path], [ocr_data], str(output_dir))

            result = np.array(Image.open(outputs[0]).convert("RGB"))

        self.assertGreater(int(result[90, 90, 0]), 220)
        self.assertGreater(int(result[105, 90, 0]), 220)

    def test_run_inpaint_pages_can_disable_white_balloon_whitening_temporarily(self):
        image = np.full((180, 180, 3), 255, dtype=np.uint8)
        ocr_data = {
            "texts": [
                {"bbox": [65, 72, 120, 80], "skip_processing": False},
            ],
            "_vision_blocks": [
                {"bbox": [65, 72, 120, 80], "mask": None},
            ],
        }

        class FakeInpainter:
            def inpaint(self, image_np, mask, batch_size=4):
                result = image_np.copy()
                cy = image_np.shape[0] // 2
                cx = image_np.shape[1] // 2
                result[cy, cx] = [77, 77, 77]
                return result

        def fake_line_artifact_cleanup(original_rgb, cleaned_rgb, texts):
            result = cleaned_rgb.copy()
            result[0, 0] = [111, 111, 111]
            return result

        def fake_micro_cleanup(original_rgb, cleaned_rgb, texts):
            result = cleaned_rgb.copy()
            result[0, 0] = [123, 123, 123]
            return result

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "page.jpg"
            output_dir = Path(tmpdir) / "out"
            Image.fromarray(image).save(image_path)
            with patch.dict(os.environ, {"MANGATL_DISABLE_WHITE_BALLOON_WHITENING": "1"}), patch(
                "vision_stack.runtime._get_inpainter",
                return_value=FakeInpainter(),
            ), patch(
                "vision_stack.runtime._apply_white_balloon_line_artifact_cleanup",
                side_effect=fake_line_artifact_cleanup,
            ), patch(
                "vision_stack.runtime._apply_white_balloon_text_box_cleanup",
                side_effect=AssertionError("text box cleanup nao deveria rodar"),
            ), patch(
                "vision_stack.runtime._apply_white_balloon_micro_artifact_cleanup",
                side_effect=fake_micro_cleanup,
            ):
                outputs = run_inpaint_pages([image_path], [ocr_data], str(output_dir))

            result = np.array(Image.open(outputs[0]).convert("RGB"))

        self.assertGreater(int(result[0, 0, 0]), 105)
        self.assertLess(int(result[0, 0, 0]), 145)

    def test_apply_inpainting_round_applies_white_balloon_cleanup_stack_after_lama(self):
        image = np.full((120, 160, 3), 230, dtype=np.uint8)
        ocr_data = {
            "texts": [
                {"bbox": [40, 40, 110, 70], "text": "HELLO", "skip_processing": False},
            ],
            "_vision_blocks": [
                {"bbox": [40, 40, 110, 70], "mask": None, "confidence": 0.9},
            ],
        }
        calls = []

        def fake_run_masked_inpaint_passes(inpainter, image_np, mask, batch_size=4):
            calls.append("lama")
            result = image_np.copy()
            result[55, 80] = [77, 77, 77]
            return {
                "expanded_mask": mask.copy(),
                "raw_output": result.copy(),
                "after_roi_paste": result.copy(),
                "after_seam_cleanup": result.copy(),
                "final_output": result.copy(),
                "cleanup_base_mask": mask.copy(),
                "fallback_to_legacy": False,
                "fallback_error": "",
            }

        def fake_line_artifact_cleanup(original_rgb, cleaned_rgb, texts):
            calls.append("line_cleanup")
            result = cleaned_rgb.copy()
            result[55, 80] = [220, 220, 220]
            return result

        def fake_text_box_cleanup(original_rgb, cleaned_rgb, texts):
            calls.append("text_box_cleanup")
            result = cleaned_rgb.copy()
            result[55, 80] = [241, 241, 241]
            return result

        def fake_micro_cleanup(original_rgb, cleaned_rgb, texts):
            calls.append("micro_cleanup")
            result = cleaned_rgb.copy()
            result[55, 80] = [248, 248, 248]
            return result

        def fake_apply_white_balloon_fill(image_np, bbox):
            calls.append("white_fill")
            result = image_np.copy()
            result[55, 80] = [255, 255, 255]
            return result

        def fake_line_cleanup(image_np):
            calls.append("line_cleanup")
            return image_np

        with patch("vision_stack.runtime._run_masked_inpaint_passes", side_effect=fake_run_masked_inpaint_passes), patch(
          "vision_stack.runtime._apply_white_balloon_line_artifact_cleanup", side_effect=fake_line_artifact_cleanup
        ), patch(
            "vision_stack.runtime._apply_white_balloon_text_box_cleanup", side_effect=fake_text_box_cleanup
        ), patch(
            "vision_stack.runtime._apply_white_balloon_micro_artifact_cleanup", side_effect=fake_micro_cleanup
        ), patch(
            "vision_stack.runtime._apply_white_balloon_fill", side_effect=fake_apply_white_balloon_fill
        ), patch("vision_stack.runtime._apply_bright_zone_line_cleanup", side_effect=fake_line_cleanup), patch(
            "vision_stack.runtime._is_white_balloon_region", return_value=True
        ):
            result = _apply_inpainting_round(image, ocr_data, inpainter=object())

        self.assertEqual(calls, ["lama", "line_cleanup", "text_box_cleanup", "micro_cleanup"])
        self.assertEqual(int(result[55, 80, 0]), 248)

    def test_run_inpaint_pages_skips_page_without_detected_blocks(self):
        image = np.full((120, 160, 3), 210, dtype=np.uint8)
        ocr_data = {
            "texts": [],
            "_vision_blocks": [],
        }

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "page.jpg"
            output_dir = Path(tmpdir) / "out"
            Image.fromarray(image).save(image_path)

            with patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
                "vision_stack.runtime._apply_inpainting_round",
                side_effect=AssertionError("nao deveria chamar inpaint em pagina sem deteccao"),
            ), patch(
                "vision_stack.runtime._run_detect_ocr_on_image",
                side_effect=AssertionError("nao deveria reler detect/ocr em pagina sem deteccao"),
            ):
                outputs = run_inpaint_pages([image_path], [ocr_data], str(output_dir))

            result = np.array(Image.open(outputs[0]).convert("RGB"))

        self.assertTrue(np.array_equal(result, image))
        self.assertTrue(ocr_data.get("sem_texto_detectado"))

    def test_run_inpaint_pages_does_not_run_recovery_detect_after_inpaint(self):
        image = np.full((120, 160, 3), 200, dtype=np.uint8)
        cleaned = image.copy()
        cleaned[42:54, 60:96] = 123
        ocr_data = {
            "texts": [
                {"bbox": [58, 40, 98, 56], "text": "HELLO", "skip_processing": False},
            ],
            "_vision_blocks": [
                {"bbox": [58, 40, 98, 56], "mask": None, "confidence": 0.9},
            ],
        }

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "page.jpg"
            output_dir = Path(tmpdir) / "out"
            Image.fromarray(image).save(image_path)

            with patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
                "vision_stack.runtime._run_koharu_blockwise_inpaint_page",
                return_value=cleaned,
            ) as blockwise_round, patch(
                "vision_stack.runtime._run_detect_ocr_on_image",
                side_effect=AssertionError("o passo 5 foi removido e nao deve mais rodar"),
            ), patch(
                "vision_stack.runtime._integrate_recovery_page",
                side_effect=AssertionError("o passo 5 foi removido e nao deve mais rodar"),
            ):
                outputs = run_inpaint_pages([image_path], [ocr_data], str(output_dir))

            result = np.array(Image.open(outputs[0]).convert("RGB"))

        self.assertEqual(blockwise_round.call_count, 1)
        self.assertEqual(int(result[45, 70, 0]), 123)

    def test_integrate_recovery_page_merges_residual_into_existing_text(self):
        base_page = {
            "image": "page.jpg",
            "width": 800,
            "height": 1200,
            "texts": [
                {
                    "text": "o que ha nisso",
                    "bbox": [200, 300, 520, 352],
                    "confidence": 0.91,
                    "tipo": "fala",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {"bbox": [200, 300, 520, 352], "mask": None, "confidence": 0.91},
            ],
        }
        recovered_page = {
            "image": "page.jpg",
            "width": 800,
            "height": 1200,
            "texts": [
                {
                    "text": "demais",
                    "bbox": [360, 306, 430, 346],
                    "confidence": 0.88,
                    "tipo": "fala",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {"bbox": [360, 306, 430, 346], "mask": None, "confidence": 0.88},
            ],
        }

        updated_page, recovery_page = _integrate_recovery_page(base_page, recovered_page)

        self.assertEqual(updated_page["texts"][0]["text"], "o que ha demais nisso")
        self.assertEqual(updated_page["texts"][0]["bbox"], [200, 300, 520, 352])
        self.assertEqual(len(recovery_page["texts"]), 1)
        self.assertEqual(recovery_page["texts"][0]["bbox"], [200, 300, 520, 352])
        self.assertEqual(recovery_page["_vision_blocks"][0]["bbox"], [200, 300, 520, 352])

    def test_integrate_recovery_page_uses_balloon_cluster_fallback(self):
        base_page = {
            "image": "page.jpg",
            "width": 800,
            "height": 1200,
            "texts": [
                {
                    "text": "THIS IS STILL",
                    "bbox": [288, 265, 525, 299],
                    "confidence": 0.91,
                    "tipo": "fala",
                    "skip_processing": False,
                },
                {
                    "text": "YOUR HANDS",
                    "bbox": [291, 346, 519, 383],
                    "confidence": 0.90,
                    "tipo": "fala",
                    "skip_processing": False,
                },
            ],
            "_vision_blocks": [
                {"bbox": [288, 265, 525, 299], "mask": None, "confidence": 0.91},
                {"bbox": [291, 346, 519, 383], "mask": None, "confidence": 0.90},
            ],
        }
        recovered_page = {
            "image": "page.jpg",
            "width": 800,
            "height": 1200,
            "texts": [
                {
                    "text": "BETTER",
                    "bbox": [308, 304, 412, 338],
                    "confidence": 0.88,
                    "tipo": "fala",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {"bbox": [308, 304, 412, 338], "mask": None, "confidence": 0.88},
            ],
        }

        updated_page, recovery_page = _integrate_recovery_page(base_page, recovered_page)

        self.assertEqual(len(recovery_page["texts"]), 1)
        self.assertTrue(any("BETTER" in text["text"] for text in updated_page["texts"]))

    def test_integrate_recovery_page_ignores_unmatched_noise(self):
        base_page = {
            "image": "page.jpg",
            "width": 800,
            "height": 1200,
            "texts": [
                {
                    "text": "fala original",
                    "bbox": [200, 300, 520, 352],
                    "confidence": 0.91,
                    "tipo": "fala",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {"bbox": [200, 300, 520, 352], "mask": None, "confidence": 0.91},
            ],
        }
        recovered_page = {
            "image": "page.jpg",
            "width": 800,
            "height": 1200,
            "texts": [
                {
                    "text": "marca d'agua",
                    "bbox": [20, 30, 140, 70],
                    "confidence": 0.88,
                    "tipo": "fala",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {"bbox": [20, 30, 140, 70], "mask": None, "confidence": 0.88},
            ],
        }

        updated_page, recovery_page = _integrate_recovery_page(base_page, recovered_page)

        self.assertEqual(updated_page["texts"][0]["text"], "fala original")
        self.assertEqual(len(updated_page["texts"]), 1)
        self.assertEqual(len(recovery_page["texts"]), 0)


if __name__ == "__main__":
    unittest.main()
