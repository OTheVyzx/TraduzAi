import unittest
import os
import json
import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image

from vision_stack.runtime import (
    _apply_inpainting_round,
    _apply_textured_balloon_band_artifact_cleanup,
    _apply_textured_light_text_residual_cleanup,
    _apply_textured_balloon_seam_cleanup,
    _apply_geometry_white_balloon_cleanup,
    _apply_glyph_residual_cleanup_for_texts,
    _apply_white_balloon_artifact_cleanup,
    _apply_white_balloon_line_artifact_cleanup,
    _apply_white_balloon_micro_artifact_cleanup,
    _apply_white_balloon_near_text_residual_cleanup,
    _apply_white_balloon_text_box_cleanup,
    _apply_bright_zone_line_cleanup,
    _apply_cjk_mask_residual_cleanup,
    _apply_ui_panel_text_cleanup_after_inpaint,
    _apply_mask_boundary_seam_cleanup,
    _apply_post_inpaint_cleanup_timed,
    _apply_white_balloon_fill,
    _apply_white_balloon_residual_force_fill,
    _apply_letter_white_boxes,
    _add_uied_layout_candidate_blocks,
    _attach_sfx_visual_candidates,
    _apply_white_text_overlay,
    _build_refined_bbox_mask,
    _build_post_cleanup_limit_mask,
    _clustered_inpaint_crop_windows,
    _build_koharu_worker_page_result,
    _extract_koharu_scene_text_blocks,
    _build_bright_zone_line_mask,
    _build_glyph_residual_cleanup_mask,
    _build_mask_boundary_seam_mask,
    _clamp_image_to_limit_mask,
    _build_residual_cleanup_mask,
    _expand_bbox,
    _extract_white_balloon_fill_mask,
    _extract_white_balloon_text_boxes,
    _has_white_balloon_text_residual,
    _integrate_recovery_page,
    _is_white_balloon_region,
    _looks_like_cover_editorial_band,
    _looks_like_cover_merged_visual_art_text,
    _looks_like_short_latin_cjk_visual_misread,
    _accepted_ocr_system_ui_rescue_allowed,
    _drop_contained_duplicate_ocr_texts,
    _drop_suppressed_ocr_pairs,
    _drop_normal_ocr_blocks_overlapping_sfx_candidates,
    _finalize_page_ocr_texts,
    _should_use_base_white_balloon_font,
    _merge_text_fragments,
    _merge_nearby_bboxes,
    _merge_ocr_clusters,
    _enlarge_koharu_window,
    _ocr_pre_translation_skip_policy,
    _profile_to_ocr_model,
    _prepare_pre_ocr_sfx_visual_candidates,
    _propagate_scanlation_credit_band_policy,
    _quick_text_presence_check,
    _remap_orientation_recovery_page,
    _run_orientation_recovery,
    _should_use_koharu_cjk_ocr,
    _run_koharu_blockwise_inpaint_page,
    _run_koharu_worker_detect_ocr_batch,
    _run_masked_inpaint_passes,
    _restore_dark_line_art_outside_text_geometry,
    _scan_orphan_white_balloon_blocks,
    _should_merge_ocr_cluster,
    _text_background_looks_translucent_or_textured,
    _text_is_white_cleanup_safe,
    _white_cleanup_texts,
    _strict_cjk_aot_crop_windows,
    _append_rotated_recovery_page,
    _should_run_rotated_text_recovery,
    _split_uied_form_label_texts,
    _try_koharu_balloon_fill,
    build_page_result,
    run_inpaint_pages,
    run_detect_ocr,
    run_ocr_stage,
    warmup_visual_stack,
    vision_blocks_to_mask,
)


class VisionStackRuntimeTests(unittest.TestCase):
    def test_drop_suppressed_ocr_pairs_removes_visual_sfx_overlap_before_masks(self):
        texts = [
            {
                "id": "ocr_cjk_sign",
                "text": "TEXTO:QUERIDO KARAOKE",
                "bbox": [12, 88, 178, 104],
                "route_action": "translate_inpaint_render",
                "route_reason": "visual_sfx_overlap_suppressed",
                "skip_processing": False,
            },
            {
                "id": "ocr_dialogue",
                "text": "PLEASE!",
                "bbox": [40, 20, 140, 64],
                "route_action": "translate_inpaint_render",
            },
        ]
        blocks = [
            {"bbox": [12, 88, 178, 104], "text": "TEXTO:QUERIDO KARAOKE"},
            {"bbox": [40, 20, 140, 64], "text": "PLEASE!"},
        ]

        kept_texts, kept_blocks = _drop_suppressed_ocr_pairs(
            texts,
            blocks,
            source_language="en",
            page_number=4,
        )

        self.assertEqual([text["id"] for text in kept_texts], ["ocr_dialogue"])
        self.assertEqual([block["text"] for block in kept_blocks], ["PLEASE!"])

    def test_drop_suppressed_ocr_pairs_removes_visual_cjk_flag_before_masks(self):
        texts = [
            {
                "id": "ocr_cjk_visual",
                "text": "달링 가라오케",
                "bbox": [10, 10, 160, 48],
                "route_action": "translate_inpaint_render",
                "qa_flags": ["visual_cjk_suppressed"],
            }
        ]
        blocks = [{"bbox": [10, 10, 160, 48], "text": "달링 가라오케"}]

        kept_texts, kept_blocks = _drop_suppressed_ocr_pairs(
            texts,
            blocks,
            source_language="en",
            page_number=4,
        )

        self.assertEqual(kept_texts, [])
        self.assertEqual(kept_blocks, [])

    def test_vision_blocks_to_mask_reapplies_suppressed_ocr_pair_guard(self):
        image = np.full((120, 260, 3), 255, dtype=np.uint8)
        texts = [
            {
                "id": "ocr_scanlator",
                "text": "TEXT: DARLING KARAOKE",
                "bbox": [30, 80, 217, 102],
                "line_polygons": [[[30, 80], [217, 80], [217, 102], [30, 102]]],
                "route_action": "translate_inpaint_render",
            }
        ]
        blocks = [dict(texts[0])]

        mask = vision_blocks_to_mask(
            image.shape,
            blocks,
            image_rgb=image,
            expand_mask=True,
            ocr_texts=texts,
        )

        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_vision_blocks_to_mask_drops_isolated_side_note_line_polygon(self):
        image = np.full((120, 520, 3), 255, dtype=np.uint8)
        block = {
            "text": "DON'T HIT MY MOM!",
            "bbox": [30, 34, 450, 104],
            "text_pixel_bbox": [30, 34, 450, 104],
            "line_polygons": [
                [[272, 36], [445, 36], [445, 62], [272, 62]],
                [[38, 54], [102, 54], [102, 71], [38, 71]],
                [[268, 78], [439, 78], [439, 106], [268, 106]],
            ],
        }

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertEqual(int(np.count_nonzero(mask[54:72, 38:103])), 0)
        self.assertGreater(int(np.count_nonzero(mask[36:63, 272:446])), 100)
        self.assertGreater(int(np.count_nonzero(mask[78:107, 268:440])), 100)

    def test_pre_ocr_sfx_skip_drops_stylized_sfx_block(self):
        image = np.full((180, 220, 3), 245, dtype=np.uint8)
        cv2.line(image, (42, 38), (122, 134), (128, 22, 34), 10, cv2.LINE_AA)
        cv2.line(image, (96, 38), (44, 132), (128, 22, 34), 10, cv2.LINE_AA)
        block = SimpleNamespace(xyxy=(32, 26, 138, 148), confidence=0.82, detector="comic-text-detector")

        candidates = _prepare_pre_ocr_sfx_visual_candidates(
            image,
            [block],
            detector_backend="comic-text-detector",
        )
        kept, skipped = _drop_normal_ocr_blocks_overlapping_sfx_candidates(image, [block], candidates)

        self.assertEqual(kept, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "english_sfx_pre_ocr_skip")

    def test_pre_ocr_sfx_skip_preserves_stylized_vertical_sfx(self):
        image = np.full((520, 220, 3), [8, 18, 28], dtype=np.uint8)
        cv2.rectangle(image, (78, 30), (156, 470), [12, 34, 54], -1)
        for offset in (0, 135, 270):
            cv2.putText(
                image,
                "K",
                (48, 108 + offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.7,
                (18, 235, 248),
                7,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "K",
                (48, 108 + offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.7,
                (230, 255, 255),
                2,
                cv2.LINE_AA,
            )
        block = SimpleNamespace(xyxy=(34, 54, 174, 420), confidence=0.74, detector="comic-text-detector")
        candidate = {
            "id": "sfx_visual_001",
            "bbox": [34, 54, 174, 420],
            "text_pixel_bbox": [34, 54, 174, 420],
            "confidence": 0.78,
            "detector": "sfx_visual",
            "content_class": "sfx",
            "sfx": {"visual_source": "comic_text_detector_fallback", "inpaint_allowed": False},
        }

        kept, skipped = _drop_normal_ocr_blocks_overlapping_sfx_candidates(image, [block], [candidate])

        self.assertEqual(kept, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "english_sfx_pre_ocr_skip")

    def test_pre_ocr_sfx_skip_preserves_plain_horizontal_dialogue_block(self):
        image = np.full((160, 260, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (130, 80), (110, 48), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (130, 80), (110, 48), 0, 0, 360, (30, 30, 30), 2)
        cv2.putText(image, "HELLO", (72, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2, cv2.LINE_AA)
        block = SimpleNamespace(xyxy=(66, 58, 174, 102), confidence=0.91, detector="comic-text-detector")

        candidates = _prepare_pre_ocr_sfx_visual_candidates(
            image,
            [block],
            detector_backend="comic-text-detector",
        )
        kept, skipped = _drop_normal_ocr_blocks_overlapping_sfx_candidates(image, [block], candidates)

        self.assertEqual(kept, [block])
        self.assertEqual(skipped, [])

    def test_attach_sfx_visual_candidates_keeps_parallel_runtime_field(self):
        image = np.full((240, 320, 3), 236, dtype=np.uint8)
        for x in range(0, 320, 7):
            cv2.line(image, (x, 20), (max(0, x - 120), 210), (38, 42, 52), 1)
        cv2.rectangle(image, (62, 48), (82, 172), (116, 32, 38), -1)
        cv2.rectangle(image, (62, 142), (164, 166), (116, 32, 38), -1)
        cv2.rectangle(image, (202, 46), (226, 178), (116, 32, 38), -1)
        cv2.rectangle(image, (202, 46), (282, 70), (116, 32, 38), -1)
        cv2.rectangle(image, (246, 104), (282, 178), (116, 32, 38), -1)
        page = {"texts": [], "_vision_blocks": []}

        with patch.dict(os.environ, {"TRADUZAI_SFX_TEXT_DETECTOR_RESCUE": "0"}), patch(
            "sfx.ocr_probe.probe_sfx_candidate_ocr",
            side_effect=lambda candidate, image_rgb: candidate,
        ):
            result = _attach_sfx_visual_candidates(page, image)

        self.assertIn("_sfx_visual_candidates", result)
        self.assertGreaterEqual(len(result["_sfx_visual_candidates"]), 1)
        self.assertEqual(result["_sfx_visual_candidates"][0]["content_class"], "sfx")
        self.assertEqual(result.get("texts"), [])

    def test_attach_sfx_visual_candidates_adds_text_detector_rescue_candidates(self):
        image = np.full((600, 320, 3), 248, dtype=np.uint8)
        cv2.putText(image, "o!", (82, 520), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (5, 5, 5), 8, cv2.LINE_AA)
        page = {"texts": [], "_vision_blocks": []}

        class FakeDetector:
            def __init__(self, blocks):
                self.blocks = blocks

            def detect(self, image_rgb, conf_threshold=0.5):
                return self.blocks

        anime_block = SimpleNamespace(xyxy=(73, 440, 177, 541), confidence=0.01079)

        def fake_get_detector(profile="quality", model="comic-text-detector"):
            if model == "anime-text-yolo-n":
                return FakeDetector([anime_block])
            return FakeDetector([])

        with patch("vision_stack.sfx_detector.detect_sfx_candidates", return_value=[]), patch(
            "vision_stack.runtime._get_detector",
            side_effect=fake_get_detector,
        ), patch(
            "sfx.ocr_probe.probe_sfx_candidate_ocr",
            side_effect=lambda candidate, image_rgb: candidate,
        ):
            result = _attach_sfx_visual_candidates(page, image)

        candidates = result["_sfx_visual_candidates"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["bbox"], [73, 440, 177, 541])
        self.assertEqual(candidates[0]["detector"], "sfx_text_detector")
        self.assertIn("sfx_text_detector_candidate", candidates[0]["qa_flags"])

    def test_attach_sfx_visual_candidates_filters_grid_after_failed_cjk_ocr(self):
        image = np.full((220, 180, 3), [30, 42, 54], dtype=np.uint8)
        for offset in range(0, 58, 8):
            cv2.line(image, (8 + offset, 70), (0 + offset, 118), (170, 180, 190), 1)
            cv2.line(image, (10 + offset, 70), (2 + offset, 118), (18, 24, 30), 1)
        candidate = {
            "id": "sfx_visual_001",
            "text_id": "sfx_visual_001",
            "bbox": [3, 68, 45, 124],
            "text_pixel_bbox": [3, 68, 45, 124],
            "content_class": "sfx",
            "tipo": "sfx",
            "detector": "sfx_text_detector",
            "confidence": 0.11,
            "qa_flags": ["sfx_visual_candidate", "sfx_text_detector_candidate"],
            "sfx": {"visual_source": "comic_text_detector_fallback", "inpaint_allowed": False},
        }

        with patch.dict(os.environ, {"TRADUZAI_SFX_TEXT_DETECTOR_RESCUE": "0"}), patch(
            "vision_stack.sfx_detector.detect_sfx_candidates",
            return_value=[candidate],
        ), patch(
            "sfx.ocr_probe.probe_sfx_candidate_ocr",
            side_effect=lambda item, image_rgb: {
                **item,
                "sfx_ocr": {"status": "no_confident_cjk", "source": "sfx_cjk_crop_probe"},
            },
        ):
            result = _attach_sfx_visual_candidates({"texts": [], "_vision_blocks": []}, image)

        self.assertEqual(result["_sfx_visual_candidates"], [])
        self.assertEqual(result.get("texts"), [])
        self.assertEqual(result.get("_vision_blocks"), [])

    def test_attach_sfx_visual_candidates_drops_pre_ocr_candidate_over_normal_text(self):
        image = np.full((220, 220, 3), 248, dtype=np.uint8)
        candidate = {
            "id": "sfx_visual_001",
            "text_id": "sfx_visual_001",
            "bbox": [42, 42, 178, 114],
            "text_pixel_bbox": [42, 42, 178, 114],
            "content_class": "sfx",
            "tipo": "sfx",
            "detector": "sfx_text_detector",
            "confidence": 0.10,
            "qa_flags": ["sfx_visual_candidate", "sfx_text_detector_candidate"],
            "sfx": {"visual_source": "comic_text_detector_fallback", "inpaint_allowed": False},
        }
        text = {"bbox": [58, 54, 162, 92], "text": "THERE WAS NO PRINCIPLE"}

        with patch.dict(os.environ, {"TRADUZAI_SFX_TEXT_DETECTOR_RESCUE": "0"}), patch(
            "vision_stack.sfx_detector.detect_sfx_candidates",
            return_value=[],
        ), patch(
            "sfx.ocr_probe.probe_sfx_candidate_ocr",
            side_effect=lambda item, image_rgb: {
                **item,
                "sfx_ocr": {"status": "no_confident_cjk", "source": "sfx_cjk_crop_probe"},
            },
        ):
            result = _attach_sfx_visual_candidates(
                {"texts": [text], "_vision_blocks": [text], "_sfx_visual_candidates": [candidate]},
                image,
            )

        self.assertEqual(result["_sfx_visual_candidates"], [])

    def test_attach_sfx_visual_candidates_keeps_recognized_cjk_overlapping_normal_text(self):
        image = np.full((220, 220, 3), 248, dtype=np.uint8)
        candidate = {
            "id": "sfx_visual_001",
            "text_id": "sfx_visual_001",
            "bbox": [42, 42, 178, 114],
            "text_pixel_bbox": [42, 42, 178, 114],
            "content_class": "sfx",
            "tipo": "sfx",
            "detector": "sfx_text_detector",
            "confidence": 0.10,
            "qa_flags": ["sfx_visual_candidate", "sfx_text_detector_candidate"],
            "sfx": {"visual_source": "comic_text_detector_fallback", "inpaint_allowed": False},
            "sfx_ocr": {"status": "recognized", "source": "sfx_cjk_crop_probe"},
            "recognized_text": "쿵",
        }
        text = {"bbox": [58, 54, 162, 92], "text": "THERE WAS NO PRINCIPLE"}

        with patch.dict(os.environ, {"TRADUZAI_SFX_TEXT_DETECTOR_RESCUE": "0"}), patch(
            "vision_stack.sfx_detector.detect_sfx_candidates",
            return_value=[],
        ), patch(
            "sfx.ocr_probe.probe_sfx_candidate_ocr",
            side_effect=lambda item, image_rgb: item,
        ):
            result = _attach_sfx_visual_candidates(
                {"texts": [text], "_vision_blocks": [text], "_sfx_visual_candidates": [candidate]},
                image,
            )

        self.assertEqual(len(result["_sfx_visual_candidates"]), 1)
        self.assertEqual(result["_sfx_visual_candidates"][0]["recognized_text"], "쿵")

    def test_attach_sfx_visual_candidates_keeps_stylized_sfx_after_failed_cjk_ocr(self):
        image = np.full((220, 180, 3), 248, dtype=np.uint8)
        cv2.putText(image, "o!", (40, 154), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (5, 5, 5), 6, cv2.LINE_AA)
        candidate = {
            "id": "sfx_visual_001",
            "text_id": "sfx_visual_001",
            "bbox": [30, 70, 128, 182],
            "text_pixel_bbox": [30, 70, 128, 182],
            "content_class": "sfx",
            "tipo": "sfx",
            "detector": "sfx_text_detector",
            "confidence": 0.064,
            "qa_flags": ["sfx_visual_candidate", "sfx_text_detector_candidate"],
            "sfx": {"visual_source": "comic_text_detector_fallback", "inpaint_allowed": False},
        }

        with patch.dict(os.environ, {"TRADUZAI_SFX_TEXT_DETECTOR_RESCUE": "0"}), patch(
            "vision_stack.sfx_detector.detect_sfx_candidates",
            return_value=[candidate],
        ), patch(
            "sfx.ocr_probe.probe_sfx_candidate_ocr",
            side_effect=lambda item, image_rgb: {
                **item,
                "sfx_ocr": {"status": "no_confident_cjk", "source": "sfx_cjk_crop_probe"},
            },
        ):
            result = _attach_sfx_visual_candidates({"texts": [], "_vision_blocks": []}, image)

        self.assertEqual(len(result["_sfx_visual_candidates"]), 1)
        self.assertEqual(result["_sfx_visual_candidates"][0]["bbox"], [30, 70, 128, 182])
        self.assertFalse(result["_sfx_visual_candidates"][0]["sfx"]["inpaint_allowed"])

    def test_attach_sfx_visual_candidates_never_filters_recognized_cjk(self):
        image = np.full((220, 180, 3), [30, 42, 54], dtype=np.uint8)
        for offset in range(0, 58, 8):
            cv2.line(image, (8 + offset, 70), (0 + offset, 118), (170, 180, 190), 1)
            cv2.line(image, (10 + offset, 70), (2 + offset, 118), (18, 24, 30), 1)
        candidate = {
            "id": "sfx_visual_001",
            "text_id": "sfx_visual_001",
            "bbox": [3, 68, 45, 124],
            "text_pixel_bbox": [3, 68, 45, 124],
            "content_class": "sfx",
            "tipo": "sfx",
            "detector": "sfx_text_detector",
            "confidence": 0.11,
            "qa_flags": ["sfx_visual_candidate", "sfx_text_detector_candidate"],
            "sfx": {"visual_source": "comic_text_detector_fallback", "inpaint_allowed": False},
        }

        with patch.dict(os.environ, {"TRADUZAI_SFX_TEXT_DETECTOR_RESCUE": "0"}), patch(
            "vision_stack.sfx_detector.detect_sfx_candidates",
            return_value=[candidate],
        ), patch(
            "sfx.ocr_probe.probe_sfx_candidate_ocr",
            side_effect=lambda item, image_rgb: {
                **item,
                "recognized_text": "쿵",
                "text": "쿵",
                "sfx_ocr": {"status": "recognized", "source": "sfx_cjk_crop_probe"},
            },
        ):
            result = _attach_sfx_visual_candidates({"texts": [], "_vision_blocks": []}, image)

        self.assertEqual(len(result["_sfx_visual_candidates"]), 1)
        self.assertEqual(result["_sfx_visual_candidates"][0]["recognized_text"], "쿵")

    def test_post_cleanup_limit_uses_line_polygons_without_filling_gap_sfx(self):
        limit_mask = np.zeros((260, 260), dtype=np.uint8)
        limit_mask[32:52, 172:234] = 255
        limit_mask[208:234, 38:132] = 255
        text = {
            "bbox": [38, 32, 234, 234],
            "text_pixel_bbox": [38, 32, 234, 234],
            "line_polygons": [
                [[172, 32], [234, 32], [234, 52], [172, 52]],
                [[38, 208], [132, 208], [132, 234], [38, 234]],
            ],
        }

        result = _build_post_cleanup_limit_mask(
            limit_mask,
            [text],
            limit_mask.shape,
        )

        self.assertIsNotNone(result)
        self.assertGreater(int(result[42, 190]), 0)
        self.assertGreater(int(result[220, 72]), 0)
        self.assertEqual(int(result[128, 128]), 0)

    def test_white_balloon_residual_force_fill_clears_text_line_mask_only(self):
        original = np.full((120, 180, 3), 248, dtype=np.uint8)
        cv2.ellipse(original, (90, 60), (70, 42), 0, 0, 360, (255, 255, 255), -1)
        original[54:62, 58:122] = [120, 132, 150]
        cleaned = original.copy()
        cleaned[54:62, 58:122] = [190, 205, 224]
        text = {
            "bbox": [54, 48, 126, 68],
            "text_pixel_bbox": [54, 48, 126, 68],
            "balloon_bbox": [20, 18, 160, 102],
        }

        result = _apply_white_balloon_residual_force_fill(original, cleaned, [text])

        self.assertGreater(int(np.mean(cleaned[56:60, 64:116])), 180)
        self.assertGreater(int(np.mean(result[56:60, 64:116])), 235)
        self.assertTrue(np.array_equal(result[12, 90], cleaned[12, 90]))

    def test_white_balloon_residual_force_fill_clears_lower_colored_ghost(self):
        original = np.full((140, 220, 3), 248, dtype=np.uint8)
        cv2.ellipse(original, (110, 70), (86, 50), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(original, (110, 70), (86, 50), 0, 0, 360, (8, 8, 8), 2)
        original[50:60, 64:156] = [70, 82, 98]
        cleaned = original.copy()
        cleaned[50:60, 64:156] = [254, 254, 254]
        cleaned[64:70, 68:152] = [188, 198, 214]
        text = {
            "bbox": [60, 44, 160, 72],
            "text_pixel_bbox": [64, 50, 156, 60],
            "balloon_bbox": [24, 20, 196, 120],
        }

        result = _apply_white_balloon_residual_force_fill(original, cleaned, [text])

        self.assertGreater(int(np.mean(result[65:69, 74:146])), 235)
        self.assertLessEqual(int(np.min(result[20:24, 100:120])), 16)

    def test_white_balloon_residual_force_fill_prefers_source_bbox_over_render_bbox(self):
        original = np.full((150, 240, 3), 248, dtype=np.uint8)
        cv2.ellipse(original, (120, 78), (92, 54), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(original, (120, 78), (92, 54), 0, 0, 360, (8, 8, 8), 2)
        original[84:96, 58:182] = [80, 92, 112]
        cleaned = original.copy()
        cleaned[84:96, 58:182] = [184, 198, 218]
        text = {
            "bbox": [82, 48, 158, 68],
            "text_pixel_bbox": [82, 48, 158, 68],
            "source_bbox": [52, 78, 188, 104],
            "balloon_bbox": [28, 24, 212, 132],
        }

        result = _apply_white_balloon_residual_force_fill(original, cleaned, [text])

        self.assertGreater(int(np.mean(result[86:94, 64:176])), 235)
        self.assertTrue(np.array_equal(result[44, 120], cleaned[44, 120]))

    def test_white_balloon_residual_force_fill_uses_bubble_mask_when_source_bbox_missing(self):
        original = np.full((150, 240, 3), 248, dtype=np.uint8)
        cv2.ellipse(original, (120, 78), (92, 54), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(original, (120, 78), (92, 54), 0, 0, 360, (8, 8, 8), 2)
        original[88:100, 54:186] = [82, 94, 112]
        cleaned = original.copy()
        cleaned[88:100, 54:186] = [184, 198, 218]
        text = {
            "bbox": [82, 48, 158, 68],
            "text_pixel_bbox": [82, 48, 158, 68],
            "bubble_mask_bbox": [28, 24, 212, 132],
            "balloon_bbox": [28, 24, 212, 132],
        }

        result = _apply_white_balloon_residual_force_fill(original, cleaned, [text])

        self.assertGreater(int(np.mean(result[90:98, 60:180])), 235)
        self.assertTrue(np.array_equal(result[44, 120], cleaned[44, 120]))

    def test_white_balloon_force_fill_clamp_keeps_text_bbox_outside_expanded_mask(self):
        base = np.full((90, 180, 3), 255, dtype=np.uint8)
        candidate = base.copy()
        candidate[54:64, 70:130] = 248
        expanded_mask = np.zeros(base.shape[:2], dtype=np.uint8)
        expanded_mask[30:45, 70:130] = 255
        text = {"text_pixel_bbox": [64, 28, 136, 68], "bbox": [64, 28, 136, 68]}

        clamped, _, outside = _clamp_image_to_limit_mask(
            base,
            candidate,
            expanded_mask,
            [text],
            include_text_bboxes=True,
        )

        self.assertEqual(outside, 0)
        self.assertTrue(np.all(clamped[56:62, 76:124] == 248))

    def test_ui_panel_text_cleanup_runs_after_full_page_inpaint(self):
        cleaned = np.full((140, 260, 3), 255, dtype=np.uint8)
        cleaned[42:72, 24:236] = [184, 196, 224]
        cleaned[54:62, 60:198] = 255
        ocr_data = {
            "texts": [
                {
                    "id": "ocr_header",
                    "text": "Successful candidate inquiry",
                    "bbox": [60, 54, 198, 62],
                    "text_pixel_bbox": [60, 54, 198, 62],
                    "line_polygons": [[[60, 54], [198, 54], [198, 62], [60, 62]]],
                    "route_action": "translate_inpaint_render",
                }
            ]
        }

        result = _apply_ui_panel_text_cleanup_after_inpaint(cleaned, ocr_data)

        self.assertTrue(ocr_data["_inpaint_used_ui_panel_text_cleanup"])
        self.assertEqual(ocr_data["_inpaint_ui_panel_text_cleanup_count"], 1)
        self.assertTrue(np.all(result[55:61, 64:194] == [184, 196, 224]))

    def test_pre_translation_skip_preserves_textured_logo_or_emblem(self):
        result = _ocr_pre_translation_skip_policy(
            "FIRE",
            [10, 24, 510, 245],
            0.56,
            tipo="fala",
            page_profile="story",
            block_profile="standard",
            is_white_balloon=False,
            image_shape=(360, 800, 3),
            line_polygons=[],
            run_on_suspect=False,
            pre_semantic_run_on=False,
            source_lang="en",
            background_rgb=[44, 48, 52],
        )

        self.assertIsNone(result)

    def test_pre_translation_skip_keeps_line_geometry_phrase_inside_broad_textured_region(self):
        result = _ocr_pre_translation_skip_policy(
            "THE PRINCIPAL",
            [12, 130, 656, 785],
            0.56,
            tipo="fala",
            page_profile="story",
            block_profile="standard",
            is_white_balloon=False,
            image_shape=(801, 800, 3),
            line_polygons=[[[544, 700], [675, 700], [675, 717], [544, 717]]],
            run_on_suspect=False,
            pre_semantic_run_on=False,
            source_lang="en",
            background_rgb=[120, 134, 176],
        )

        self.assertIsNone(result)

    def test_pre_translation_skip_does_not_preserve_system_ui_message_as_logo(self):
        result = _ocr_pre_translation_skip_policy(
            "SYSTEM ERROR",
            [218, 115, 426, 160],
            0.838,
            tipo="fala",
            page_profile="story",
            block_profile="standard",
            is_white_balloon=False,
            image_shape=(720, 540, 3),
            line_polygons=[[[218, 115], [426, 115], [426, 160], [218, 160]]],
            run_on_suspect=False,
            pre_semantic_run_on=False,
            source_lang="en",
            background_rgb=[2, 2, 137],
        )

        self.assertIsNone(result)

    def test_pre_translation_skip_does_not_preserve_system_ui_message_without_geometry(self):
        result = _ocr_pre_translation_skip_policy(
            "SYSTEM ERROR",
            [218, 115, 426, 160],
            0.838,
            tipo="fala",
            page_profile="story",
            block_profile="standard",
            is_white_balloon=False,
            image_shape=(720, 540, 3),
            line_polygons=[],
            run_on_suspect=False,
            pre_semantic_run_on=False,
            source_lang="en",
            background_rgb=[2, 2, 137],
        )

        self.assertIsNone(result)

    def test_pre_translation_skip_does_not_preserve_white_balloon_word_as_logo(self):
        result = _ocr_pre_translation_skip_policy(
            "FIRE",
            [120, 84, 210, 132],
            0.78,
            tipo="fala",
            page_profile="story",
            block_profile="white_balloon",
            is_white_balloon=True,
            image_shape=(360, 800, 3),
            line_polygons=[[[120, 84], [210, 84], [210, 132], [120, 132]]],
            run_on_suspect=False,
            pre_semantic_run_on=False,
            source_lang="en",
            background_rgb=[250, 250, 250],
        )

        self.assertIsNone(result)

    def test_pre_translation_skip_preserves_textured_art_label_with_catalog_tokens(self):
        samples = [
            (
                "Side Stereo SFB2BT LSPATO2 BGBSOBGA FIVEYEARS SOUL LOVE",
                [
                    [[328, 2108], [532, 2108], [532, 2128], [328, 2128]],
                    [[336, 2130], [512, 2130], [512, 2150], [336, 2150]],
                    [[340, 2152], [518, 2152], [518, 2172], [340, 2172]],
                    [[344, 2174], [506, 2174], [506, 2194], [344, 2194]],
                    [[348, 2196], [514, 2196], [514, 2216], [348, 2216]],
                    [[352, 2218], [500, 2218], [500, 2238], [352, 2238]],
                ],
            ),
            (
                '(Lse ao2i Side " Stereoi Sf b2bt Eaes Ave = :YEARS SOUL ~Love',
                [],
            ),
        ]
        for sample, line_polygons in samples:
            with self.subTest(sample=sample):
                result = _ocr_pre_translation_skip_policy(
                    sample,
                    [328, 2108, 532, 2253],
                    0.872,
                    tipo="fala",
                    page_profile="story",
                    block_profile="standard",
                    is_white_balloon=False,
                    image_shape=(2800, 690, 3),
                    line_polygons=line_polygons,
                    run_on_suspect=False,
                    pre_semantic_run_on=False,
                    source_lang="en",
                    background_rgb=[58, 139, 200],
                )

                self.assertIsNone(result)

    def test_pre_translation_skip_does_not_preserve_numbered_system_ui_as_art_label(self):
        samples = [
            "LEVEL 2 CLEAR",
            "RECORD A1B2 C3D4 CLEAR",
            "PLAYER X9Y8Z7 RECORD A1B2C3 STATUS",
            "TITLE A1B2 C3D4 COMPLETE",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                result = _ocr_pre_translation_skip_policy(
                    sample,
                    [120, 84, 360, 150],
                    0.88,
                    tipo="fala",
                    page_profile="story",
                    block_profile="standard",
                    is_white_balloon=False,
                    image_shape=(720, 540, 3),
                    line_polygons=[
                        [[120, 84], [360, 84], [360, 150], [120, 150]],
                        [[130, 152], [350, 152], [350, 180], [130, 180]],
                        [[135, 182], [345, 182], [345, 210], [135, 210]],
                    ],
                    run_on_suspect=False,
                    pre_semantic_run_on=False,
                    source_lang="en",
                    background_rgb=[12, 70, 120],
                )

                self.assertIsNone(result)

    def test_scanlation_credit_policy_preserves_promotional_titles_in_same_band(self):
        page_texts = [
            {
                "id": "ocr_credit",
                "text_id": "ocr_credit",
                "text": "Join our Discord and support our Patreon.",
                "bbox": [40, 80, 720, 180],
                "content_class": "url_watermark",
                "skip_processing": True,
                "qa_flags": ["scanlation_credit"],
                "balloon_type": "textured",
            },
            {
                "id": "ocr_title",
                "text_id": "ocr_title",
                "text": "SUPER CUBE",
                "bbox": [260, 230, 360, 262],
                "content_class": "dialogue",
                "skip_processing": False,
                "qa_flags": [],
                "balloon_type": "textured",
            },
        ]
        vision_blocks = [{"text_id": item["text_id"], "bbox": item["bbox"]} for item in page_texts]

        result_texts, result_blocks = _propagate_scanlation_credit_band_policy(
            page_texts,
            vision_blocks,
            (420, 800, 3),
        )

        title = result_texts[1]
        self.assertFalse(title["skip_processing"])
        self.assertEqual(title["content_class"], "dialogue")
        self.assertNotIn("scanlation_credit", title["qa_flags"])
        self.assertFalse(result_blocks[1].get("skip_processing", False))

    def test_scanlation_credit_policy_preserves_aligned_promotional_title_strip(self):
        page_texts = [
            {
                "id": f"ocr_{idx}",
                "text_id": f"ocr_{idx}",
                "text": text,
                "bbox": bbox,
                "content_class": content_class,
                "skip_processing": skip_processing,
                "preserve_original": skip_processing,
                "qa_flags": [],
                "balloon_type": "textured",
            }
            for idx, (text, bbox, content_class, skip_processing) in enumerate(
                [
                    ("REBORN bo,ooo YEARS", [40, 200, 210, 230], "narration", False),
                    ("THE STRONGEST GOD KING", [260, 203, 470, 232], "logo", True),
                    ("GLOBAL MARTIAL ARTS", [520, 201, 720, 230], "logo", True),
                ],
                start=1,
            )
        ]
        vision_blocks = [{"text_id": item["text_id"], "bbox": item["bbox"]} for item in page_texts]

        result_texts, result_blocks = _propagate_scanlation_credit_band_policy(
            page_texts,
            vision_blocks,
            (360, 800, 3),
        )

        self.assertFalse(result_texts[0]["skip_processing"])
        self.assertEqual(result_texts[0]["content_class"], "narration")
        self.assertFalse(result_blocks[0].get("skip_processing", False))

    def test_should_merge_ocr_cluster_vetoes_band_043_dominant_partial_overlap(self):
        texts = [
            {"text": "HEY, LET'S GO!", "bbox": [63, 16, 635, 625], "balloon_type": "white"},
            {"text": "WHO'S PAYING TODAY?", "bbox": [338, 562, 546, 668], "balloon_type": "white"},
        ]

        self.assertFalse(_should_merge_ocr_cluster(texts, [63, 16, 635, 668]))

    def test_should_merge_ocr_cluster_vetoes_band_017_dominant_partial_overlap(self):
        texts = [
            {"text": "SO PLEASE WAIT A LITTLE LONGER.", "bbox": [121, 88, 350, 214], "balloon_type": "white"},
            {"text": "WHEN I GET BACK TO WORK...", "bbox": [137, 156, 726, 625], "balloon_type": "white"},
        ]

        self.assertFalse(_should_merge_ocr_cluster(texts, [121, 88, 726, 625]))

    def test_should_merge_ocr_cluster_keeps_close_stacked_lines(self):
        texts = [
            {"text": "FIRST LINE", "bbox": [100, 100, 220, 132], "balloon_type": "white"},
            {"text": "SECOND LINE", "bbox": [104, 136, 224, 168], "balloon_type": "white"},
        ]

        self.assertTrue(_should_merge_ocr_cluster(texts, [90, 90, 240, 180]))

    def test_merge_ocr_clusters_keeps_dominant_partial_overlap_separate(self):
        page_texts = [
            {"text": "HEY, LET'S GO!", "bbox": [63, 16, 635, 625], "balloon_type": "white", "confidence": 0.91},
            {"text": "WHO'S PAYING TODAY?", "bbox": [338, 562, 546, 668], "balloon_type": "white", "confidence": 0.88},
        ]
        vision_blocks = [{"bbox": text["bbox"], "confidence": text["confidence"]} for text in page_texts]

        merged_texts, merged_blocks = _merge_ocr_clusters(page_texts, vision_blocks, (700, 700, 3), 3)

        self.assertEqual(len(merged_texts), 2)
        self.assertEqual(len(merged_blocks), 2)

    def test_merge_ocr_clusters_keeps_p23_broad_container_and_lower_fragments_separate(self):
        page_texts = [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_025_band_067",
                "text": "What is it...? Why is he getting",
                "bbox": [148, 96, 667, 439],
                "confidence": 0.56,
            },
            {
                "id": "ocr_002",
                "text_id": "ocr_002",
                "trace_id": "ocr_002@page_025_band_067",
                "text": "What is it...?",
                "bbox": [581, 311, 663, 350],
                "confidence": 0.875,
            },
            {
                "id": "ocr_003",
                "text_id": "ocr_003",
                "trace_id": "ocr_003@page_025_band_067",
                "text": "scared alone?",
                "bbox": [463, 374, 629, 445],
                "confidence": 0.935,
            },
        ]
        vision_blocks = [{"bbox": text["bbox"], "confidence": text["confidence"]} for text in page_texts]

        merged_texts, merged_blocks = _merge_ocr_clusters(page_texts, vision_blocks, (500, 800, 3), page_number=23)

        self.assertEqual([item["text"] for item in merged_texts], [item["text"] for item in page_texts])
        self.assertEqual(len(merged_blocks), 3)
        self.assertTrue(all("source_text_ids" not in item for item in merged_texts))
        self.assertTrue(all("source_text_ids" not in item for item in merged_blocks))

    def test_merge_ocr_clusters_vetoes_p23_geometry_without_text_duplication(self):
        page_texts = [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_025_band_067",
                "text": "Why is he getting",
                "bbox": [148, 96, 667, 439],
                "confidence": 0.56,
            },
            {
                "id": "ocr_002",
                "text_id": "ocr_002",
                "trace_id": "ocr_002@page_025_band_067",
                "text": "What is it...?",
                "bbox": [581, 311, 663, 350],
                "confidence": 0.875,
            },
            {
                "id": "ocr_003",
                "text_id": "ocr_003",
                "trace_id": "ocr_003@page_025_band_067",
                "text": "scared alone?",
                "bbox": [463, 374, 629, 445],
                "confidence": 0.935,
            },
        ]
        vision_blocks = [{"bbox": text["bbox"], "confidence": text["confidence"]} for text in page_texts]

        merged_texts, merged_blocks = _merge_ocr_clusters(page_texts, vision_blocks, (500, 800, 3), page_number=23)

        self.assertEqual([item["text"] for item in merged_texts], [item["text"] for item in page_texts])
        self.assertEqual(len(merged_blocks), 3)

    def test_vision_blocks_to_mask_skips_preserved_noise_regions(self):
        image = np.full((220, 260, 3), 245, dtype=np.uint8)
        image[120:176, 112:220] = [220, 184, 164]
        image[34:46, 36:104] = 8
        blocks = [
            {
                "bbox": [110, 118, 224, 180],
                "text_pixel_bbox": [112, 120, 220, 176],
                "skip_processing": True,
                "skip_reason": "suspicious_art_ocr_low_confidence",
                "content_class": "noise",
            },
            {
                "bbox": [34, 32, 108, 50],
                "text_pixel_bbox": [36, 34, 104, 46],
                "line_polygons": [[[36, 34], [104, 34], [104, 46], [36, 46]]],
                "balloon_type": "white",
                "layout_profile": "white_balloon",
            },
        ]

        mask = vision_blocks_to_mask(image.shape, blocks, image_rgb=image, expand_mask=False)

        self.assertGreater(int(np.count_nonzero(mask[34:47, 36:105])), 0)
        self.assertGreater(int(np.count_nonzero(mask[120:176, 112:220])), 0)

    def test_vision_blocks_to_mask_skips_review_required_art_fragments(self):
        image = np.full((220, 260, 3), 245, dtype=np.uint8)
        image[70:120, 80:180] = [80, 140, 128]
        block = {
            "bbox": [70, 60, 190, 130],
            "text_pixel_bbox": [80, 70, 180, 120],
            "line_polygons": [[[80, 70], [180, 70], [180, 120], [80, 120]]],
            "route_action": "review_required",
            "route_reason": "ocr_art_fragment_suspected",
            "qa_flags": ["ocr_art_fragment_suspected"],
        }

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False, ocr_texts=[block])

        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_glyph_cleanup_removes_residual_text_without_box_fill(self):
        original = np.full((120, 220, 3), 255, dtype=np.uint8)
        cv2.putText(
            original,
            "CANCER",
            (48, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (18, 18, 18),
            2,
            cv2.LINE_AA,
        )
        cleaned = original.copy()
        text = {
            "bbox": [42, 42, 164, 76],
            "text_pixel_bbox": [42, 42, 164, 76],
            "line_polygons": [[[42, 42], [164, 42], [164, 76], [42, 76]]],
            "source_bbox": [18, 24, 198, 96],
            "balloon_bbox": [42, 42, 164, 76],
            "balloon_type": "textured",
        }

        result = _apply_glyph_residual_cleanup_for_texts(original, cleaned, [text])

        before_dark = int(np.count_nonzero(cv2.cvtColor(cleaned[42:76, 42:164], cv2.COLOR_RGB2GRAY) < 180))
        after_dark = int(np.count_nonzero(cv2.cvtColor(result[42:76, 42:164], cv2.COLOR_RGB2GRAY) < 180))
        self.assertGreater(before_dark, 40)
        self.assertEqual(after_dark, 0)
        self.assertGreater(int(result[28, 22, 0]), 240)

    def test_glyph_cleanup_mask_does_not_target_connected_balloon_outline(self):
        original = np.full((170, 330, 3), 255, dtype=np.uint8)
        outline = np.zeros(original.shape[:2], dtype=np.uint8)
        cv2.ellipse(original, (102, 74), (78, 50), 0, 0, 360, (8, 8, 8), 2, cv2.LINE_AA)
        cv2.ellipse(original, (228, 84), (82, 54), 0, 0, 360, (8, 8, 8), 2, cv2.LINE_AA)
        cv2.ellipse(outline, (102, 74), (78, 50), 0, 0, 360, 255, 2, cv2.LINE_AA)
        cv2.ellipse(outline, (228, 84), (82, 54), 0, 0, 360, 255, 2, cv2.LINE_AA)
        cv2.putText(original, "WHY", (74, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (12, 12, 12), 2, cv2.LINE_AA)
        cv2.putText(original, "NEED", (196, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (12, 12, 12), 2, cv2.LINE_AA)
        text = {
            "bbox": [22, 18, 310, 140],
            "text_pixel_bbox": [22, 18, 310, 140],
            "line_polygons": [
                [[66, 42], [140, 42], [140, 70], [66, 70]],
                [[188, 68], [276, 68], [276, 98], [188, 98]],
            ],
            "balloon_bbox": [10, 10, 320, 150],
            "balloon_type": "textured",
            "layout_profile": "connected_balloon",
        }

        glyph_mask = _build_glyph_residual_cleanup_mask(original, text, original.shape[:2])

        self.assertIsNotNone(glyph_mask)
        assert glyph_mask is not None
        self.assertGreater(int(np.count_nonzero(glyph_mask)), 20)
        self.assertEqual(int(np.count_nonzero((glyph_mask > 0) & (outline > 0))), 0)

    def test_textured_cleanup_requires_line_geometry(self):
        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        image[48:88, 74:146] = 20
        text = {
            "bbox": [70, 44, 150, 92],
            "text_pixel_bbox": [70, 44, 150, 92],
            "balloon_type": "textured",
        }

        self.assertTrue(_text_is_white_cleanup_safe(image, text))

    def test_textured_light_residual_cleanup_removes_white_ghost_text(self):
        original = np.full((120, 260, 3), (72, 136, 214), dtype=np.uint8)
        cv2.putText(
            original,
            "TITLE",
            (82, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (248, 248, 248),
            2,
            cv2.LINE_AA,
        )
        cleaned = original.copy()
        text = {
            "bbox": [76, 46, 194, 82],
            "text_pixel_bbox": [76, 46, 194, 82],
            "line_polygons": [[[76, 46], [194, 46], [194, 82], [76, 82]]],
            "balloon_bbox": [72, 40, 198, 88],
            "balloon_type": "textured",
        }

        result = _apply_textured_light_text_residual_cleanup(original, cleaned, [text])

        before_gray = cv2.cvtColor(cleaned[46:82, 76:194], cv2.COLOR_RGB2GRAY)
        after_gray = cv2.cvtColor(result[46:82, 76:194], cv2.COLOR_RGB2GRAY)
        self.assertGreater(int(np.count_nonzero(before_gray > 225)), 30)
        self.assertLess(int(np.count_nonzero(after_gray > 225)), 8)
        self.assertTrue(np.array_equal(result[4, 4], cleaned[4, 4]))

    def test_geometry_white_cleanup_preserves_outline_outside_text_geometry(self):
        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        image[76:90, 112:148] = 20
        cv2.line(image, (58, 138), (144, 138), (8, 8, 8), 2)
        text = {
            "bbox": [108, 72, 152, 94],
            "text_pixel_bbox": [108, 72, 152, 94],
            "line_polygons": [[[108, 72], [152, 72], [152, 94], [108, 94]]],
            "balloon_bbox": [34, 34, 226, 154],
            "balloon_type": "white",
            "block_profile": "white_balloon",
        }

        result = _apply_geometry_white_balloon_cleanup(image, image.copy(), [text])

        self.assertLess(int(result[138, 80, 0]), 80)

    def test_line_art_restore_does_not_reintroduce_white_balloon_text_residual_below_bbox(self):
        original = np.full((160, 260, 3), 255, dtype=np.uint8)
        cv2.ellipse(original, (130, 85), (95, 55), 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
        original[118:120, 90:112] = 10
        original[119:121, 130:146] = 10
        original[118:120, 165:184] = 10
        cleaned = original.copy()
        cleaned[70:130, 60:205] = 255
        cv2.ellipse(cleaned, (130, 85), (95, 55), 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
        text = {
            "bbox": [70, 70, 190, 116],
            "text_pixel_bbox": [75, 75, 185, 112],
            "source_bbox": [70, 70, 190, 116],
            "balloon_bbox": [35, 30, 225, 140],
            "bubble_mask_bbox": [35, 30, 225, 140],
            "layout_profile": "white_balloon",
            "background_rgb": [255, 255, 255],
        }

        result = _restore_dark_line_art_outside_text_geometry(original, cleaned, [text])

        self.assertGreater(int(result[119, 100, 0]), 230)
        self.assertGreater(int(result[119, 138, 0]), 230)
        self.assertGreater(int(result[119, 175, 0]), 230)
        self.assertLess(int(result[85, 35, 0]), 80)

    def test_post_cleanup_removes_small_text_edge_specks_without_erasing_outline(self):
        original = np.full((140, 220, 3), 255, dtype=np.uint8)
        cleaned = original.copy()
        original[58:72, 88:132] = 20
        cv2.line(original, (48, 32), (172, 32), (8, 8, 8), 2)
        cleaned[76:79, 82:85] = 18
        cleaned[81, 92] = 12
        cv2.line(cleaned, (54, 112), (166, 112), (8, 8, 8), 2)
        text = {
            "bbox": [84, 54, 136, 76],
            "text_pixel_bbox": [84, 54, 136, 76],
            "line_polygons": [[[84, 54], [136, 54], [136, 76], [84, 76]]],
            "balloon_bbox": [34, 26, 186, 122],
            "balloon_type": "white",
            "block_profile": "white_balloon",
        }
        limit_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        limit_mask[46:88, 76:144] = 255

        result, stats = _apply_post_inpaint_cleanup_timed(original, cleaned, [text], limit_mask=limit_mask)

        self.assertGreater(stats["_t_cleanup_near_text_residual_ms"], 0.0)
        self.assertGreater(int(result[77, 83, 0]), 230)
        self.assertGreater(int(result[81, 92, 0]), 230)
        self.assertLess(int(result[32, 90, 0]), 80)
        self.assertLess(int(result[112, 90, 0]), 80)

    def test_vision_mask_keeps_near_text_residual_but_protects_distant_outline(self):
        image = np.full((120, 180, 3), 255, dtype=np.uint8)
        image[50:70, 70:110] = 15
        image[47:50, 111:114] = 15
        cv2.ellipse(image, (92, 43), (42, 14), 0, 190, 350, (8, 8, 8), 2)
        cv2.line(image, (40, 94), (142, 94), (8, 8, 8), 2)
        block = {
            "bbox": [68, 48, 112, 72],
            "text_pixel_bbox": [70, 50, 110, 70],
            "line_polygons": [[[70, 50], [110, 50], [110, 70], [70, 70]]],
            "balloon_type": "white",
        }

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=True)

        self.assertGreater(int(mask[48, 112]), 0)
        self.assertEqual(int(mask[28, 92]), 0)
        self.assertEqual(int(mask[94, 82]), 0)

    def test_vision_mask_merges_tight_reference_glyph_missing_from_local_mask(self):
        from inpainter.mask_builder import build_inpaint_mask

        image = np.full((110, 180, 3), 255, dtype=np.uint8)
        image[22:44, 37:43] = 0
        image[22:26, 34:47] = 0
        image[40:44, 34:47] = 0
        image[26:44, 60:126] = 0
        block = {
            "bbox": [58, 20, 142, 54],
            "text_pixel_bbox": [58, 20, 142, 54],
            "source_bbox": [34, 18, 142, 54],
            "balloon_bbox": [34, 18, 142, 54],
            "line_polygons": [[[58, 20], [142, 20], [142, 54], [58, 54]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "texto",
        }
        local_mask = build_inpaint_mask(dict(block), image.shape, image_rgb=image)
        self.assertIsNotNone(local_mask)
        local_mask = local_mask.copy()
        local_mask[18:48, 34:52] = 0
        block["mask"] = local_mask

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertGreater(int(mask[32, 40]), 0)
        self.assertGreater(int(mask[34, 92]), 0)
        self.assertEqual(int(mask[8, 40]), 0)
        self.assertIn("tight_reference_geometry_extra_pixels", block.get("qa_metrics", {}))

        expanded_mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=True)
        self.assertGreater(int(expanded_mask[32, 40]), 0)

    def test_vision_mask_keeps_tight_reference_glyph_when_local_mask_is_larger(self):
        from inpainter.mask_builder import build_inpaint_mask

        image = np.full((110, 180, 3), 255, dtype=np.uint8)
        image[22:44, 38:46] = 0
        image[22:26, 35:49] = 0
        image[40:44, 35:49] = 0
        image[26:44, 55:126] = 0
        block = {
            "bbox": [34, 18, 142, 54],
            "text_pixel_bbox": [55, 20, 142, 54],
            "source_bbox": [34, 18, 142, 54],
            "balloon_bbox": [34, 18, 142, 54],
            "line_polygons": [[[55, 20], [142, 20], [142, 54], [55, 54]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "texto",
        }
        local_mask = build_inpaint_mask(dict(block), image.shape, image_rgb=image)
        self.assertIsNotNone(local_mask)
        local_mask = local_mask.copy()
        local_mask[14:50, 30:52] = 0
        local_mask[10:80, 130:150] = 255
        block["mask"] = local_mask

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertGreater(int(mask[32, 42]), 0)
        self.assertIn("tight_reference_geometry_extra_pixels", block.get("qa_metrics", {}))

    def test_near_text_cleanup_uses_local_roi_for_balloon_distance_transform(self):
        original = np.full((200, 300, 3), 255, dtype=np.uint8)
        cleaned = original.copy()
        cleaned[86:94, 108:118] = [0, 0, 0]
        text = {
            "bbox": [96, 74, 136, 106],
            "text_pixel_bbox": [100, 80, 130, 100],
            "balloon_bbox": [80, 60, 160, 120],
            "balloon_type": "white",
            "block_profile": "white_balloon",
        }
        balloon_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        balloon_mask[60:120, 80:160] = 255

        with patch(
            "vision_stack.runtime._resolve_white_balloon_bbox",
            return_value=[80, 60, 160, 120],
        ), patch(
            "vision_stack.runtime._extract_white_balloon_fill_mask",
            return_value=balloon_mask,
        ), patch(
            "vision_stack.runtime.cv2.distanceTransform",
            wraps=cv2.distanceTransform,
        ) as distance_transform:
            _apply_white_balloon_near_text_residual_cleanup(original, cleaned, [text])

        self.assertTrue(distance_transform.called)
        distance_shape = distance_transform.call_args.args[0].shape
        self.assertLess(distance_shape[0], original.shape[0])
        self.assertLess(distance_shape[1], original.shape[1])
        self.assertEqual(distance_shape, (36, 46))

    def test_serialized_vision_block_keeps_text_geometry_without_connected_fields(self):
        from vision_stack.runtime import _apply_text_geometry_to_serialized_block

        block = _apply_text_geometry_to_serialized_block(
            {
                "bbox": [0, 0, 300, 180],
                "confidence": 0.9,
                "connected_lobe_bboxes": [[0, 0, 150, 180], [150, 0, 300, 180]],
            },
            {
                "text": "HELLO",
                "bbox": [20, 30, 220, 110],
                "source_bbox": [20, 30, 220, 110],
                "text_pixel_bbox": [60, 50, 160, 88],
                "line_polygons": [[[60, 50], [160, 50], [160, 88], [60, 88]]],
                "balloon_subregions": [[0, 0, 150, 180], [150, 0, 300, 180]],
                "connected_lobe_bboxes": [[0, 0, 150, 180], [150, 0, 300, 180]],
                "connected_lobe_polygons": [],
                "tipo": "fala",
            },
        )

        self.assertEqual(block["bbox"], [0, 0, 300, 180])
        self.assertEqual(block["source_bbox"], [0, 0, 300, 180])
        self.assertEqual(block["text_pixel_bbox"], [60, 50, 160, 88])
        self.assertEqual(block["text_pixel_bbox"], [60, 50, 160, 88])
        self.assertEqual(block["line_polygons"], [[[60, 50], [160, 50], [160, 88], [60, 88]]])
        self.assertEqual(block["balloon_subregions"], [[0, 0, 150, 180], [150, 0, 300, 180]])
        self.assertEqual(block["connected_lobe_bboxes"], [[0, 0, 150, 180], [150, 0, 300, 180]])
        self.assertNotIn("connected_lobe_polygons", block)

    def test_strip_inpaint_debug_writes_masks_and_overlay(self):
        from inpainter import inpaint_band_image

        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        image[30:42, 45:75] = 10
        ocr_page = {
            "_band_index": 7,
            "_source_page_number": 3,
            "texts": [
                {
                    "text": "YES",
                    "translated": "SIM",
                    "bbox": [45, 30, 75, 42],
                    "text_pixel_bbox": [45, 30, 75, 42],
                    "tipo": "fala",
                    "confidence": 0.95,
                    "balloon_type": "white",
                }
            ],
            "_vision_blocks": [{"bbox": [45, 30, 75, 42], "mask": None, "confidence": 0.95}],
        }

        class FakeInpainter:
            def inpaint(self, image_np, mask, batch_size=4, debug=None, force_no_tiling=False):
                result = image_np.copy()
                result[mask > 0] = 245
                return result

        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "TRADUZAI_INPAINT_DEBUG_DIR": tmp,
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
        ), patch("vision_stack.runtime._get_inpainter", return_value=FakeInpainter()):
            inpaint_band_image(image, ocr_page)
            debug_dir = Path(tmp) / "page_003_band_007"
            raw_mask_path = debug_dir / "01_inpaint_mask_raw.png"

            self.assertTrue(raw_mask_path.exists())
            self.assertTrue((debug_dir / "02_inpaint_mask_expanded.png").exists())
            self.assertTrue((debug_dir / "03_inpaint_mask_overlay.jpg").exists())
            metadata = json.loads((debug_dir / "metadata.json").read_text(encoding="utf-8"))
            raw_mask = np.array(Image.open(raw_mask_path).convert("L"))

        self.assertGreater(int(np.count_nonzero(raw_mask)), 0)
        self.assertEqual(metadata["remaining_inpaint_blocks"], 1)
        self.assertEqual(metadata["text_count"], 1)

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
        raise unittest.SkipTest(f"fixture image missing: {name}")

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

    def test_orphan_white_balloon_scan_adds_uncovered_small_bubble(self):
        image = np.zeros((260, 360, 3), dtype=np.uint8)
        cv2.ellipse(image, (245, 130), (78, 42), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (245, 130), (78, 42), 0, 0, 360, (0, 0, 0), 3)
        cv2.putText(image, "NONE.", (202, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
        existing = [SimpleNamespace(xyxy=(32, 32, 96, 82), confidence=0.9)]

        blocks = _scan_orphan_white_balloon_blocks(image, existing)

        self.assertGreaterEqual(len(blocks), 2)
        orphan = blocks[-1]
        x1, y1, x2, y2 = [int(v) for v in orphan.xyxy]
        self.assertLessEqual(x1, 220)
        self.assertGreaterEqual(x2, 260)
        self.assertLessEqual(y1, 128)
        self.assertGreaterEqual(y2, 145)
        self.assertEqual(orphan.detector, "white_balloon_orphan_scan")

    def test_orphan_white_balloon_scan_adds_light_translucent_small_bubble_without_existing_blocks(self):
        image = np.zeros((220, 260, 3), dtype=np.uint8)
        cv2.ellipse(image, (130, 90), (54, 34), 0, 0, 360, (232, 232, 232), -1)
        cv2.ellipse(image, (130, 90), (54, 34), 0, 0, 360, (0, 0, 0), 2)
        cv2.putText(image, "MOM...", (100, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)

        blocks = _scan_orphan_white_balloon_blocks(image, [])

        self.assertGreaterEqual(len(blocks), 1)
        orphan = blocks[0]
        x1, y1, x2, y2 = [int(v) for v in orphan.xyxy]
        self.assertLessEqual(x1, 105)
        self.assertGreaterEqual(x2, 145)
        self.assertLessEqual(y1, 86)
        self.assertGreaterEqual(y2, 102)
        self.assertEqual(orphan.detector, "white_balloon_orphan_scan")

    def test_orphan_white_balloon_scan_adds_uncovered_line_when_white_regions_are_connected(self):
        image = np.full((240, 360, 3), 255, dtype=np.uint8)
        cv2.putText(image, "WHAT?", (70, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "SHE HID THIS MUCH.", (126, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        existing = [SimpleNamespace(xyxy=(66, 58, 154, 86), confidence=0.9)]

        blocks = _scan_orphan_white_balloon_blocks(image, existing)

        added = [block for block in blocks if getattr(block, "detector", "") == "white_text_line_orphan_scan"]
        self.assertTrue(added)
        x1, y1, x2, y2 = [int(v) for v in added[0].xyxy]
        self.assertLessEqual(x1, 132)
        self.assertGreaterEqual(x2, 300)
        self.assertLessEqual(y1, 150)
        self.assertGreaterEqual(y2, 165)

    def test_get_inpainter_is_thread_safe_during_prewarm(self):
        import vision_stack.runtime as runtime

        previous = runtime._inpainter
        runtime._inpainter = None
        constructor_calls = []
        gate = threading.Barrier(2)

        class FakeInpainter:
            def __init__(self, **kwargs):
                constructor_calls.append(kwargs)
                time.sleep(0.05)

        def call_get():
            gate.wait(timeout=2)
            return runtime._get_inpainter("quality")

        try:
            with patch("vision_stack.inpainter.Inpainter", FakeInpainter), patch(
                "vision_stack.runtime._profile_to_device",
                return_value="cpu",
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _idx: call_get(), range(2)))
        finally:
            runtime._inpainter = previous

        self.assertEqual(len(constructor_calls), 1)
        self.assertIs(results[0], results[1])

    def test_get_detector_is_thread_safe_during_prewarm(self):
        import vision_stack.runtime as runtime

        previous = runtime._detector
        previous_model = runtime._detector_model
        runtime._detector = None
        runtime._detector_model = ""
        constructor_calls = []
        gate = threading.Barrier(2)

        class FakeDetector:
            def __init__(self, **kwargs):
                constructor_calls.append(kwargs)
                time.sleep(0.05)

        def call_get():
            gate.wait(timeout=2)
            return runtime._get_detector("quality")

        try:
            with patch("vision_stack.detector.TextDetector", FakeDetector), patch(
                "vision_stack.runtime._profile_to_device",
                return_value="cpu",
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _idx: call_get(), range(2)))
        finally:
            runtime._detector = previous
            runtime._detector_model = previous_model

        self.assertEqual(len(constructor_calls), 1)
        self.assertIs(results[0], results[1])

    def test_detector_model_for_manga_preset_uses_comic_text_detector(self):
        from vision_stack.engine_presets import resolve_engine_preset
        from vision_stack.runtime import _detector_model_for_preset

        self.assertEqual(
            _detector_model_for_preset(resolve_engine_preset({"engine_preset_id": "manga"})),
            "comic-text-detector",
        )
        self.assertEqual(
            _detector_model_for_preset(resolve_engine_preset({"engine_preset_id": "manhwa_manhua"})),
            "comic-text-detector",
        )

    def test_engine_preset_metadata_records_detector_loader_and_candidates(self):
        from vision_stack.engine_presets import resolve_engine_preset
        from vision_stack.runtime import _attach_engine_preset_metadata

        preset = resolve_engine_preset({"engine_preset_id": "manga"})
        page = {
            "texts": [],
            "_vision_blocks": [
                {"bbox": [10, 12, 60, 34]},
                {
                    "bbox": [70, 12, 120, 34],
                    "candidate_kind": "ocr_recovered_block",
                    "validated_by_segment_mask": True,
                },
            ],
        }

        result = _attach_engine_preset_metadata(page, preset)

        self.assertEqual(result["_engine_preset"]["detector_engine_id"], "comic-text-bubble-detector")
        self.assertEqual(result["_engine_preset"]["detector_loader"], "comic-text-detector")
        self.assertEqual(result["_pipeline_artifacts"]["TextBoxes"]["producer"], "comic-text-bubble-detector")
        self.assertEqual(result["_pipeline_artifacts"]["SegmentMask"]["producer"], "comic-text-detector-seg")
        self.assertEqual(result["_pipeline_artifacts"]["BubbleMask"]["producer"], "speech-bubble-segmentation")
        self.assertEqual(result["_pipeline_artifacts"]["BubbleMask"]["status"], "pending")
        self.assertEqual(result["_pipeline_artifacts"]["OcrText"]["producer"], "paddle-ocr-vl-1.5")
        first, second = result["_vision_blocks"]
        self.assertEqual(first["detector_preset_id"], "manga")
        self.assertEqual(first["detector_engine_id"], "comic-text-bubble-detector")
        self.assertEqual(first["detector_loader"], "comic-text-detector")
        self.assertEqual(first["candidate_kind"], "detector_block")
        self.assertFalse(first["validated_by_segment_mask"])
        self.assertEqual(second["candidate_kind"], "ocr_recovered_block")
        self.assertTrue(second["validated_by_segment_mask"])

    def test_reconcile_ocr_with_validated_sources_clamps_broad_bbox(self):
        from vision_stack.runtime import _reconcile_ocr_with_validated_sources

        page = {
            "texts": [
                {
                    "text": "wide full page ocr block",
                    "bbox": [20, 20, 380, 290],
                    "text_pixel_bbox": [20, 20, 380, 290],
                    "tipo": "narracao",
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [20, 20, 380, 290],
                    "_validated_text_source_bboxes": [[40, 20, 360, 120]],
                    "_rejected_text_source_bboxes": [[20, 80, 380, 290]],
                    "validated_by_segment_mask": True,
                    "detector_preset_id": "manhwa_manhua",
                    "detector_engine_id": "comic-text-bubble-detector",
                }
            ],
        }

        result = _reconcile_ocr_with_validated_sources(page)
        text = result["texts"][0]

        self.assertEqual(text["layout_bbox"], [40, 20, 360, 120])
        self.assertEqual(text["text_pixel_bbox"], [40, 20, 360, 120])
        self.assertEqual(text["_validated_text_source_bboxes"], [[40, 20, 360, 120]])
        self.assertEqual(text["_rejected_text_source_bboxes"], [[20, 80, 380, 290]])
        self.assertTrue(text["validated_by_segment_mask"])
        self.assertIn("ocr_overmerged_validated_sources", text["qa_flags"])
        self.assertEqual(text["detector_preset_id"], "manhwa_manhua")

    def test_reconcile_ocr_with_multiple_validated_sources_keeps_single_item_without_split_evidence(self):
        from vision_stack.runtime import _reconcile_ocr_with_validated_sources

        page = {
            "texts": [
                {
                    "text": "wide full page ocr block",
                    "bbox": [20, 20, 380, 300],
                    "text_pixel_bbox": [20, 20, 380, 300],
                    "tipo": "narracao",
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [20, 20, 380, 300],
                    "_validated_text_source_bboxes": [[40, 24, 350, 74], [48, 210, 340, 260]],
                    "validated_by_segment_mask": True,
                }
            ],
        }

        result = _reconcile_ocr_with_validated_sources(page)

        self.assertEqual(len(result["texts"]), 1)
        text = result["texts"][0]
        self.assertEqual(text["layout_bbox"], [40, 24, 350, 260])
        self.assertEqual(text["text_pixel_bbox"], [40, 24, 350, 260])
        self.assertEqual(text["_render_target_source"], "validated_text_source")
        self.assertIn("ocr_multiple_validated_sources", text["qa_flags"])

    def test_reconcile_ocr_removes_inline_sfx_line_from_dialogue_geometry(self):
        from vision_stack.runtime import _reconcile_ocr_with_validated_sources

        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "DON'T HIT SFXKICK My MOM!",
                    "original": "DON'T HIT SFXKICK My MOM!",
                    "bbox": [0, 96, 555, 755],
                    "text_pixel_bbox": [38, 605, 445, 677],
                    "line_polygons": [
                        [[38, 605], [108, 605], [108, 629], [38, 629]],
                        [[220, 610], [396, 610], [396, 638], [220, 638]],
                        [[230, 646], [430, 646], [430, 677], [230, 677]],
                    ],
                    "tipo": "fala",
                }
            ],
            "_vision_blocks": [],
        }

        result = _reconcile_ocr_with_validated_sources(page)
        text = result["texts"][0]

        self.assertEqual(text["text"], "DON'T HIT My MOM!")
        self.assertEqual(text["original"], "DON'T HIT My MOM!")
        self.assertEqual(text["line_polygons"], [
            [[220, 610], [396, 610], [396, 638], [220, 638]],
            [[230, 646], [430, 646], [430, 677], [230, 677]],
        ])
        self.assertEqual(text["bbox"], [220, 610, 431, 678])
        self.assertEqual(text["text_pixel_bbox"], [220, 610, 431, 678])
        self.assertIn("inline_sfx_geometry_removed", text["qa_flags"])

    def test_reconcile_ocr_with_multiple_validated_sources_splits_when_lines_map_to_sources(self):
        from vision_stack.runtime import _reconcile_ocr_with_validated_sources

        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "first line\nsecond line",
                    "bbox": [20, 20, 380, 300],
                    "text_pixel_bbox": [20, 20, 380, 300],
                    "line_polygons": [
                        [[48, 34], [320, 34], [320, 58], [48, 58]],
                        [[52, 220], [316, 220], [316, 244], [52, 244]],
                    ],
                    "tipo": "narracao",
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [20, 20, 380, 300],
                    "_validated_text_source_bboxes": [[40, 24, 350, 74], [48, 210, 340, 260]],
                    "validated_by_segment_mask": True,
                }
            ],
        }

        result = _reconcile_ocr_with_validated_sources(page)

        self.assertEqual(len(result["texts"]), 2)
        first, second = result["texts"]
        self.assertEqual(first["text"], "first line")
        self.assertEqual(second["text"], "second line")
        self.assertEqual(first["_validated_text_source_bboxes"], [[40, 24, 350, 74]])
        self.assertEqual(second["_validated_text_source_bboxes"], [[48, 210, 340, 260]])
        self.assertEqual(first["layout_bbox"], [40, 24, 350, 74])
        self.assertEqual(second["layout_bbox"], [48, 210, 340, 260])
        self.assertEqual(first["_render_target_source"], "validated_text_source")
        self.assertIn("ocr_split_validated_sources", first["qa_flags"])

    def test_get_ocr_engine_is_thread_safe_during_prewarm(self):
        import vision_stack.runtime as runtime

        previous = runtime._ocr_engine
        runtime._ocr_engine = None
        constructor_calls = []
        gate = threading.Barrier(2)

        class FakeOCR:
            def __init__(self, **kwargs):
                constructor_calls.append(kwargs)
                self._requested_model = kwargs.get("model")
                self.model_name = kwargs.get("model")
                self.lang = kwargs.get("lang", "en")
                time.sleep(0.05)

        def call_get():
            gate.wait(timeout=2)
            return runtime._get_ocr_engine("quality", lang="en")

        try:
            with patch("vision_stack.ocr.OCREngine", FakeOCR), patch(
                "vision_stack.runtime._profile_to_device",
                return_value="cpu",
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _idx: call_get(), range(2)))
        finally:
            runtime._ocr_engine = previous

        self.assertEqual(len(constructor_calls), 1)
        self.assertIs(results[0], results[1])

    def test_selective_cleanup_skips_unneeded_stages_but_keeps_micro_cleanup(self):
        image = np.full((80, 120, 3), 240, dtype=np.uint8)
        calls = []

        def fake_micro(original, cleaned, texts):
            calls.append("micro")
            return cleaned

        with patch.dict(os.environ, {"TRADUZAI_CLEANUP_SELECTIVE": "1"}, clear=False), patch(
            "vision_stack.runtime._apply_textured_balloon_seam_cleanup",
            side_effect=AssertionError("textured cleanup should be skipped"),
        ), patch(
            "vision_stack.runtime._apply_textured_balloon_band_artifact_cleanup",
            side_effect=AssertionError("band cleanup should be skipped"),
        ), patch(
            "vision_stack.runtime._apply_white_balloon_line_artifact_cleanup",
            side_effect=AssertionError("white cleanup should be skipped"),
        ), patch(
            "vision_stack.runtime._apply_white_balloon_text_box_cleanup",
            side_effect=AssertionError("white box cleanup should be skipped"),
        ), patch(
            "vision_stack.runtime._apply_white_balloon_micro_artifact_cleanup",
            side_effect=fake_micro,
        ):
            cleaned, stats = _apply_post_inpaint_cleanup_timed(image, image.copy(), [{"balloon_type": "dark"}])

        self.assertEqual(calls, ["micro"])
        self.assertEqual(cleaned.shape, image.shape)
        self.assertEqual(stats["cleanup_reason"], "micro_only")
        self.assertTrue(stats["cleanup_skipped_seam"])
        self.assertTrue(stats["cleanup_skipped_white_line"])

    def test_post_inpaint_cleanup_filters_textured_art_out_of_white_cleanup(self):
        image = np.full((90, 130, 3), [238, 246, 249], dtype=np.uint8)
        image[:, 55:59] = [160, 188, 202]
        cleaned = image.copy()
        texts = [
            {
                "text": "HELLO",
                "bbox": [10, 12, 38, 32],
                "text_pixel_bbox": [10, 12, 38, 32],
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
            {
                "text": "SFX",
                "bbox": [50, 8, 68, 78],
                "text_pixel_bbox": [50, 8, 68, 78],
                "balloon_type": "textured",
                "block_profile": "sfx",
                "background_type": "textured_background",
                "background_rgb": [238, 246, 249],
            },
        ]
        seen = {}

        def capture(name):
            def fake_cleanup(original, current, cleanup_texts):
                seen[name] = [text.get("text") for text in cleanup_texts]
                return current

            return fake_cleanup

        with patch("vision_stack.runtime._apply_textured_balloon_seam_cleanup", side_effect=capture("seam")), patch(
            "vision_stack.runtime._apply_textured_balloon_band_artifact_cleanup",
            side_effect=capture("band"),
        ), patch(
            "vision_stack.runtime._apply_textured_light_text_residual_cleanup",
            side_effect=capture("textured_light"),
        ), patch(
            "vision_stack.runtime._apply_white_balloon_line_artifact_cleanup",
            side_effect=capture("white_line"),
        ), patch(
            "vision_stack.runtime._apply_white_balloon_text_box_cleanup",
            side_effect=capture("white_box"),
        ), patch(
            "vision_stack.runtime._apply_geometry_white_balloon_cleanup",
            side_effect=capture("geometry_white"),
        ), patch(
            "vision_stack.runtime._apply_white_balloon_micro_artifact_cleanup",
            side_effect=capture("micro"),
        ), patch(
            "vision_stack.runtime._apply_white_balloon_near_text_residual_cleanup",
            side_effect=capture("near_text"),
        ), patch(
            "vision_stack.runtime._apply_glyph_residual_cleanup_for_texts",
            side_effect=capture("glyph"),
        ):
            result, stats = _apply_post_inpaint_cleanup_timed(image, cleaned, texts, selective=False)

        self.assertEqual(seen["white_line"], ["HELLO", "SFX"])
        self.assertNotIn("white_box", seen)
        self.assertEqual(seen["geometry_white"], ["HELLO", "SFX"])
        self.assertEqual(seen["micro"], ["HELLO", "SFX"])
        self.assertEqual(seen["seam"], ["HELLO", "SFX"])
        self.assertEqual(seen["band"], ["HELLO", "SFX"])
        self.assertEqual(seen["textured_light"], ["HELLO", "SFX"])
        self.assertEqual(seen["near_text"], ["HELLO", "SFX"])
        self.assertEqual(seen["glyph"], ["HELLO", "SFX"])
        self.assertEqual(stats["cleanup_reason"], "white_only")
        self.assertTrue(stats["cleanup_skipped_white_box"])
        self.assertTrue(np.array_equal(result, cleaned))

    def test_white_cleanup_accepts_text_anchor_inside_white_balloon_even_when_detector_bbox_is_textured(self):
        image = np.full((140, 220, 3), [90, 110, 130], dtype=np.uint8)
        cv2.ellipse(image, (150, 70), (54, 34), 0, 0, 360, (252, 252, 252), -1)
        image[60:74, 118:184] = 18
        text = {
            "text": "THE CHILD'S",
            "bbox": [20, 12, 204, 112],
            "text_pixel_bbox": [116, 58, 186, 76],
            "line_polygons": [[[116, 58], [186, 58], [186, 76], [116, 76]]],
            "balloon_type": "textured",
            "block_profile": "standard",
            "skip_processing": False,
        }

        self.assertTrue(_text_is_white_cleanup_safe(image, text))

    def test_geometry_white_cleanup_removes_residual_for_textured_detector_inside_white_balloon(self):
        original = np.full((140, 220, 3), [90, 110, 130], dtype=np.uint8)
        cv2.ellipse(original, (150, 70), (54, 34), 0, 0, 360, (252, 252, 252), -1)
        cv2.ellipse(original, (150, 70), (54, 34), 0, 0, 360, (20, 20, 20), 2)
        original[60:74, 118:184] = 18
        cleaned = original.copy()
        cleaned[60:74, 118:184] = 32
        text = {
            "text": "THE CHILD'S",
            "bbox": [20, 12, 204, 112],
            "text_pixel_bbox": [116, 58, 186, 76],
            "line_polygons": [[[116, 58], [186, 58], [186, 76], [116, 76]]],
            "balloon_type": "textured",
            "block_profile": "standard",
            "skip_processing": False,
        }

        result = _apply_geometry_white_balloon_cleanup(original, cleaned, [text])

        self.assertGreaterEqual(int(np.percentile(result[61:73, 122:180, 0], 10)), 225)
        self.assertLessEqual(int(result[36, 150, 0]), 30)

    def test_post_inpaint_cleanup_clamps_white_cleanup_to_limit_mask(self):
        image = np.full((80, 120, 3), 230, dtype=np.uint8)
        cleaned = image.copy()
        limit_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        limit_mask[10:25, 20:45] = 255
        text = {
            "text": "HELLO",
            "bbox": [20, 10, 45, 25],
            "text_pixel_bbox": [20, 10, 45, 25],
            "balloon_type": "white",
            "block_profile": "white_balloon",
        }

        def fake_white_cleanup(original, current, cleanup_texts):
            result = current.copy()
            result[15, 30] = [255, 255, 255]
            result[55, 95] = [255, 255, 255]
            return result

        with patch(
            "vision_stack.runtime._apply_textured_balloon_seam_cleanup",
            side_effect=lambda _original, current, _texts: current,
        ), patch(
            "vision_stack.runtime._apply_textured_balloon_band_artifact_cleanup",
            side_effect=lambda _original, current, _texts: current,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_line_artifact_cleanup",
            side_effect=fake_white_cleanup,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_text_box_cleanup",
            side_effect=lambda _original, current, _texts: current,
        ), patch(
            "vision_stack.runtime._apply_geometry_white_balloon_cleanup",
            side_effect=lambda _original, current, _texts: current,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_micro_artifact_cleanup",
            side_effect=lambda _original, current, _texts: current,
        ):
            result, stats = _apply_post_inpaint_cleanup_timed(
                image,
                cleaned,
                [text],
                selective=False,
                limit_mask=limit_mask,
            )

        self.assertEqual(result[15, 30].tolist(), [255, 255, 255])
        self.assertEqual(result[55, 95].tolist(), cleaned[55, 95].tolist())
        self.assertEqual(stats["cleanup_limit_mask_pixels"], int(np.count_nonzero(limit_mask)))
        self.assertEqual(stats["cleanup_changed_outside_limit_mask"], 1)

    def test_run_masked_inpaint_passes_reports_roi_metrics(self):
        class FakeInpainter:
            def inpaint(self, image_np, mask, **kwargs):
                del mask, kwargs
                return image_np.copy()

        image = np.full((180, 240, 3), 255, dtype=np.uint8)
        mask = np.zeros((180, 240), dtype=np.uint8)
        mask[70:95, 100:130] = 255

        result = _run_masked_inpaint_passes(FakeInpainter(), image, mask, texts=[{"balloon_type": "white"}])

        self.assertIn("_t_roi_select_ms", result)
        self.assertIn("_t_lama_ms", result)
        self.assertIn("roi_area_ratio", result)
        self.assertIsInstance(result["used_roi_crop"], bool)

    def test_run_masked_inpaint_passes_can_skip_internal_mask_expansion(self):
        calls = []

        class FakeInpainter:
            def inpaint(self, image_np, mask, **kwargs):
                del kwargs
                calls.append(mask.copy())
                return image_np.copy()

        image = np.full((40, 40, 3), 127, dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[20, 20] = 255

        result = _run_masked_inpaint_passes(
            FakeInpainter(),
            image,
            mask,
            expand_mask=False,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(int(np.count_nonzero(calls[0])), 1)
        self.assertEqual(int(np.count_nonzero(result["expanded_mask"])), 1)

    def test_run_masked_inpaint_passes_with_crop_windows_pastes_mask_only(self):
        calls = []

        class FakeInpainter:
            def inpaint(self, image_np, mask, **kwargs):
                del kwargs
                calls.append((image_np.shape, int(np.count_nonzero(mask))))
                return np.full_like(image_np, 9)

        image = np.full((120, 160, 3), 127, dtype=np.uint8)
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[50:56, 70:78] = 255

        result = _run_masked_inpaint_passes(
            FakeInpainter(),
            image,
            mask,
            expand_mask=False,
            crop_windows=[[40, 40, 100, 80]],
        )

        self.assertEqual(calls, [((40, 60, 3), 48)])
        self.assertEqual(int(result["final_output"][52, 72, 0]), 9)
        self.assertEqual(int(result["final_output"][42, 42, 0]), 127)
        self.assertEqual(int(result["final_output"][10, 10, 0]), 127)
        self.assertEqual(result["crop_windows_used"], 1)

    def test_clustered_inpaint_crop_windows_splits_sparse_components_when_it_saves_area(self):
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[60:70, 60:72] = 255
        mask[300:312, 310:324] = 255

        with patch.dict(
            os.environ,
            {
                "TRADUZAI_INPAINT_CLUSTERED_CROP_WINDOWS": "1",
                "TRADUZAI_INPAINT_CLUSTERED_CROP_MARGIN": "12",
            },
            clear=False,
        ):
            windows = _clustered_inpaint_crop_windows(mask, (400, 400, 3))

        self.assertIsNotNone(windows)
        self.assertEqual(len(windows), 2)
        crop_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in windows or [])
        self.assertLess(crop_area, 400 * 400 * 0.25)

    def test_clustered_inpaint_crop_windows_are_opt_in_by_default(self):
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[60:70, 60:72] = 255
        mask[300:312, 310:324] = 255

        with patch.dict(os.environ, {}, clear=True):
            windows = _clustered_inpaint_crop_windows(mask, (400, 400, 3))

        self.assertIsNone(windows)

    def test_run_masked_inpaint_passes_uses_clustered_windows_for_sparse_components(self):
        calls = []

        class FakeInpainter:
            def inpaint(self, image_np, mask, **kwargs):
                del kwargs
                calls.append((image_np.shape[:2], int(np.count_nonzero(mask))))
                result = image_np.copy()
                result[mask > 0] = [9, 9, 9]
                return result

        image = np.full((400, 400, 3), 127, dtype=np.uint8)
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[60:70, 60:72] = 255
        mask[300:312, 310:324] = 255

        with patch.dict(
            os.environ,
            {
                "TRADUZAI_INPAINT_CLUSTERED_CROP_WINDOWS": "1",
                "TRADUZAI_INPAINT_CLUSTERED_CROP_MARGIN": "12",
            },
            clear=False,
        ):
            result = _run_masked_inpaint_passes(
                FakeInpainter(),
                image,
                mask,
                expand_mask=False,
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(result["crop_windows_used"], 2)
        self.assertTrue(result["used_roi_crop"])
        self.assertLess(result["roi_area_ratio"], 0.25)
        self.assertEqual(int(result["final_output"][65, 65, 0]), 9)
        self.assertEqual(int(result["final_output"][306, 316, 0]), 9)
        self.assertEqual(int(result["final_output"][200, 200, 0]), 127)

    def test_clustered_windows_do_not_bypass_requested_seam_cleanup(self):
        calls = []

        class FakeInpainter:
            def inpaint(self, image_np, mask, **kwargs):
                del mask, kwargs
                calls.append(image_np.shape[:2])
                return image_np.copy()

        image = np.full((400, 400, 3), 127, dtype=np.uint8)
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[60:70, 60:72] = 255
        mask[300:312, 310:324] = 255

        with patch.dict(
            os.environ,
            {
                "TRADUZAI_INPAINT_CLUSTERED_CROP_WINDOWS": "1",
                "TRADUZAI_INPAINT_CLUSTERED_CROP_MARGIN": "12",
            },
            clear=False,
        ), patch(
            "vision_stack.runtime._apply_mask_boundary_seam_cleanup",
            side_effect=lambda output, _mask, debug=None: output,
        ) as seam_cleanup:
            result = _run_masked_inpaint_passes(
                FakeInpainter(),
                image,
                mask,
                expand_mask=False,
                seam_cleanup=True,
            )

        self.assertEqual(len(calls), 1)
        self.assertNotIn("crop_windows_used", result)
        seam_cleanup.assert_called_once()

    def test_cjk_mask_residual_cleanup_removes_unchanged_text_inside_mask_only(self):
        original = np.zeros((80, 140, 3), dtype=np.uint8)
        cleaned = original.copy()
        cv2.putText(original, "...", (42, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
        cleaned[:] = original
        mask = np.zeros((80, 140), dtype=np.uint8)
        mask[20:56, 35:104] = 255
        mask[20:56, 108:130] = 0
        untouched = cleaned.copy()

        result = _apply_cjk_mask_residual_cleanup(original, cleaned, mask)

        self.assertLess(int(np.mean(result[30:50, 42:96])), int(np.mean(untouched[30:50, 42:96])))
        self.assertTrue(np.array_equal(result[:, 110:130], untouched[:, 110:130]))

    def test_strict_cjk_aot_crop_windows_follow_mask_inside_blocks(self):
        mask = np.zeros((160, 220), dtype=np.uint8)
        mask[70:76, 100:112] = 255
        blocks = [{"bbox": [20, 30, 180, 130]}]

        windows = _strict_cjk_aot_crop_windows(mask, blocks, (160, 220, 3), margin=12)

        self.assertEqual(windows, [[88, 58, 124, 88]])

    def test_strict_cjk_aot_crop_windows_include_orphan_mask_outside_blocks(self):
        mask = np.zeros((160, 220), dtype=np.uint8)
        mask[70:76, 100:112] = 255
        mask[18:25, 30:44] = 255
        blocks = [{"bbox": [20, 30, 180, 130]}]

        windows = _strict_cjk_aot_crop_windows(mask, blocks, (160, 220, 3), margin=12)

        self.assertIn([88, 58, 124, 88], windows)
        self.assertIn([18, 6, 56, 37], windows)

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

    def test_build_page_result_keeps_repaired_age_number_after_semantic_review(self):
        block = SimpleNamespace(
            xyxy=(30, 18, 130, 56),
            mask=None,
            confidence=0.73,
            line_polygons=[],
        )

        page = build_page_result(
            image_path="page.jpg",
            image_rgb=np.full((90, 180, 3), 255, dtype=np.uint8),
            blocks=[block],
            texts=["Hosu 2a years old Unemployed"],
            idioma_origem="en",
        )

        self.assertEqual(page["texts"][0]["text"], "Hosu 24 years old Unemployed")
        self.assertEqual(page["_vision_blocks"][0]["text"], "Hosu 24 years old Unemployed")

    def test_build_page_result_attaches_uied_layout_evidence_to_ui_text(self):
        image = np.full((90, 180, 3), 255, dtype=np.uint8)
        image[22:52, 14:166] = [184, 196, 224]
        line_polygons = [[[42, 32], [138, 32], [138, 42], [42, 42]]]
        block = SimpleNamespace(
            xyxy=(42, 32, 138, 42),
            mask=None,
            confidence=0.91,
            line_polygons=line_polygons,
        )

        with patch.dict(os.environ, {"TRADUZAI_UIED_LAYOUT": "1"}):
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[{"text": "Open status window", "line_polygons": line_polygons}],
            )

        text = page["texts"][0]
        vision_block = page["_vision_blocks"][0]
        self.assertEqual(text["ui_layout_evidence"]["source"], "uied_cv")
        self.assertEqual(text["ui_layout_evidence"]["role"], "text_inside_component")
        self.assertEqual(text["background_rgb"], [184, 196, 224])
        self.assertEqual(text["layout_profile"], "ui_form")
        self.assertEqual(vision_block["ui_layout_evidence"]["source"], "uied_cv")
        self.assertGreaterEqual(len(page["_ui_layout_components"]), 1)

    def test_build_page_result_does_not_attach_uied_layout_evidence_by_default(self):
        image = np.full((90, 180, 3), 255, dtype=np.uint8)
        image[22:52, 14:166] = [184, 196, 224]
        line_polygons = [[[42, 32], [138, 32], [138, 42], [42, 42]]]
        block = SimpleNamespace(
            xyxy=(42, 32, 138, 42),
            mask=None,
            confidence=0.91,
            line_polygons=line_polygons,
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRADUZAI_UIED_LAYOUT", None)
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[{"text": "Open status window", "line_polygons": line_polygons}],
            )

        text = page["texts"][0]
        vision_block = page["_vision_blocks"][0]
        self.assertIsNone(text.get("ui_layout_evidence"))
        self.assertNotEqual(text.get("layout_profile"), "ui_form")
        self.assertIsNone(vision_block.get("ui_layout_evidence"))
        self.assertEqual(page["_ui_layout_components"], [])

    def test_build_page_result_preserves_uied_candidate_metadata_from_block(self):
        image = np.full((90, 180, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(
            xyxy=(18, 20, 160, 44),
            mask=None,
            confidence=0.62,
            detector="uied_cv",
            ui_layout_evidence={
                "source": "uied_cv",
                "role": "header_near_component",
                "confidence": 0.62,
                "component_bbox": [18, 48, 160, 72],
            },
            background_rgb=[255, 255, 255],
            layout_profile="ui_form",
            block_profile="ui_form",
            layout_safe_reason="uied_cv_candidate",
        )

        page = build_page_result(
            image_path="page.jpg",
            image_rgb=image,
            blocks=[block],
            texts=["20** regional fireman"],
        )

        text = page["texts"][0]
        vision_block = page["_vision_blocks"][0]
        self.assertEqual(text["ui_layout_evidence"]["role"], "header_near_component")
        self.assertEqual(text["layout_profile"], "ui_form")
        self.assertEqual(text["block_profile"], "ui_form")
        self.assertEqual(vision_block["ui_layout_evidence"]["role"], "header_near_component")

    def test_build_page_result_splits_uied_form_labels_by_nearby_field_rows(self):
        image = np.full((130, 360, 3), 255, dtype=np.uint8)
        image[42:64, 170:330] = [220, 220, 220]
        image[72:102, 170:330] = [220, 220, 220]
        line_polygons = [
            [[42, 44], [86, 44], [86, 58], [42, 58]],
            [[34, 72], [112, 72], [112, 86], [34, 86]],
            [[18, 88], [154, 88], [154, 102], [18, 102]],
        ]
        block = SimpleNamespace(
            xyxy=(18, 40, 154, 104),
            mask=None,
            confidence=0.91,
            line_polygons=line_polygons,
        )

        with patch.dict(os.environ, {"TRADUZAI_UIED_LAYOUT": "1"}):
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[
                    {
                        "text": "Name Resident registration number",
                        "line_texts": ["Name", "Resident", "registration number"],
                        "line_polygons": line_polygons,
                    }
                ],
            )

        self.assertEqual([text["text"] for text in page["texts"]], ["Name", "Resident registration number"])
        self.assertEqual(len(page["_vision_blocks"]), 2)
        self.assertTrue(all(text["layout_profile"] == "ui_form" for text in page["texts"]))
        self.assertTrue(all(text["ui_layout_evidence"]["source"] == "uied_cv" for text in page["texts"]))

    def test_add_uied_layout_candidate_blocks_adds_ui_bar_and_header_band(self):
        image = np.full((150, 300, 3), 255, dtype=np.uint8)
        cv2.putText(image, "20** regional fireman", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (18, 18, 18), 1)
        image[58:84, 22:278] = [184, 196, 224]
        cv2.putText(image, "Successful candidate inquiry", (62, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (18, 18, 18), 1)

        with patch.dict(os.environ, {"TRADUZAI_UIED_LAYOUT": "1"}):
            blocks = _add_uied_layout_candidate_blocks(image, [])

        roles = [getattr(block, "ui_layout_role", "") for block in blocks]
        bboxes = [[int(v) for v in getattr(block, "xyxy")] for block in blocks]
        self.assertIn("text_inside_component", roles)
        self.assertIn("header_near_component", roles)
        self.assertTrue(any(bbox[0] <= 24 and bbox[1] <= 58 and bbox[2] >= 278 and bbox[3] <= 58 for bbox in bboxes))

    def test_add_uied_layout_candidate_blocks_does_not_duplicate_existing_overlap(self):
        image = np.full((120, 260, 3), 255, dtype=np.uint8)
        image[40:66, 20:240] = [184, 196, 224]
        existing = [SimpleNamespace(xyxy=(22, 42, 238, 64), mask=None, confidence=0.93)]

        with patch.dict(os.environ, {"TRADUZAI_UIED_LAYOUT": "1"}):
            blocks = _add_uied_layout_candidate_blocks(image, existing)

        self.assertEqual(len(blocks), 1)
        self.assertIs(blocks[0], existing[0])

    def test_build_page_result_propagates_rotation_metadata_from_line_polygons(self):
        line_polygons = [
            [[40, 20], [56, 16], [96, 136], [80, 140]],
        ]
        block = SimpleNamespace(
            xyxy=(20, 0, 130, 160),
            mask=None,
            confidence=0.91,
            line_polygons=line_polygons,
        )

        page = build_page_result(
            image_path="page.jpg",
            image_rgb=np.full((180, 160, 3), 210, dtype=np.uint8),
            blocks=[block],
            texts=[{"text": "TILTED", "line_polygons": line_polygons}],
        )

        text = page["texts"][0]
        vision_block = page["_vision_blocks"][0]
        self.assertEqual(text["rotation_source"], "line_polygons")
        self.assertEqual(vision_block["rotation_source"], "line_polygons")
        self.assertGreater(text["rotation_deg"], 65.0)
        self.assertLess(text["rotation_deg"], 80.0)
        self.assertEqual(vision_block["rotation_deg"], text["rotation_deg"])

    def test_rotated_recovery_runs_for_clipped_existing_rotated_text(self):
        page = {
            "texts": [
                {
                    "text": "karma and destiny to theirs.",
                    "bbox": [334, 1097, 646, 1631],
                    "text_pixel_bbox": [333, 1123, 559, 1596],
                    "rotation_deg": 74.31,
                    "confidence": 0.34,
                    "qa_flags": ["TEXT_CLIPPED"],
                }
            ],
        }
        ocr = SimpleNamespace(recognize_rotated_full_page_lines=lambda _image: [])

        self.assertTrue(_should_run_rotated_text_recovery(page, [object()], "paddleocr", ocr))

    def test_rotated_recovery_merges_better_overlapping_record(self):
        base_page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text_id": "ocr_002",
                    "text": "karma and destiny to theirs.",
                    "bbox": [334, 1097, 646, 1631],
                    "text_pixel_bbox": [333, 1123, 559, 1596],
                    "line_polygons": [
                        [[333, 1136], [380, 1123], [512, 1583], [465, 1596]],
                    ],
                    "rotation_deg": 74.31,
                    "confidence": 0.34,
                    "qa_flags": ["TEXT_CLIPPED"],
                    "balloon_type": "textured",
                }
            ],
            "_vision_blocks": [
                {
                    "text_id": "ocr_002",
                    "bbox": [334, 1097, 646, 1631],
                    "text_pixel_bbox": [333, 1123, 559, 1596],
                    "rotation_deg": 74.31,
                    "confidence": 0.34,
                    "qa_flags": ["TEXT_CLIPPED"],
                    "balloon_type": "textured",
                }
            ],
        }
        recovered_page = {
            "texts": [
                {
                    "text": "Appoint Guardian (Unique) You appointed a guardian, intrinsically linking your karma and destiny to theirs.",
                    "bbox": [257, 1085, 800, 1673],
                    "source_bbox": [257, 1085, 800, 1673],
                    "text_pixel_bbox": [338, 1085, 759, 1592],
                    "line_polygons": [
                        [[627, 1085], [759, 1536], [704, 1551], [572, 1094]],
                        [[499, 1114], [614, 1531], [571, 1543], [456, 1126]],
                        [[448, 1141], [552, 1535], [509, 1546], [405, 1153]],
                        [[378, 1127], [507, 1581], [466, 1592], [338, 1138]],
                    ],
                    "rotation_deg": 74.42,
                    "rotation_source": "rotated_page_ocr",
                    "confidence": 0.96,
                    "qa_flags": ["TEXT_CLIPPED"],
                    "balloon_type": "textured",
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [257, 1085, 800, 1673],
                    "source_bbox": [257, 1085, 800, 1673],
                    "text_pixel_bbox": [338, 1085, 759, 1592],
                    "line_polygons": [
                        [[627, 1085], [759, 1536], [704, 1551], [572, 1094]],
                        [[499, 1114], [614, 1531], [571, 1543], [456, 1126]],
                    ],
                    "rotation_deg": 74.42,
                    "rotation_source": "rotated_page_ocr",
                    "confidence": 0.96,
                    "qa_flags": ["TEXT_CLIPPED"],
                    "balloon_type": "textured",
                }
            ],
        }

        updated, count = _append_rotated_recovery_page(base_page, recovered_page)

        self.assertEqual(count, 1)
        self.assertEqual(len(updated["texts"]), 1)
        self.assertEqual(updated["texts"][0]["id"], "ocr_002")
        self.assertIn("Appoint Guardian", updated["texts"][0]["text"])
        self.assertTrue(updated["texts"][0]["allow_broad_bbox_text_search"])
        self.assertEqual(updated["texts"][0]["rotation_source"], "rotated_page_ocr")
        self.assertEqual(len(updated["texts"][0]["line_polygons"]), 4)
        self.assertTrue(updated["_vision_blocks"][0]["allow_broad_bbox_text_search"])

    def test_build_page_result_can_disable_cjk_sfx_preservation_for_ocr_guided_cleanup(self):
        block = SimpleNamespace(
            xyxy=(30, 18, 98, 54),
            mask=None,
            confidence=0.91,
        )
        image = np.full((90, 130, 3), 42, dtype=np.uint8)

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False):
            preserved = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["반짝"],
                idioma_origem="ko",
            )
            cleaned = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["반짝"],
                idioma_origem="ko",
                preserve_cjk_sfx=False,
            )

        self.assertEqual(preserved["texts"][0]["ignored_reason"], "cjk_sfx_preserved")
        self.assertEqual(preserved["texts"][0]["route_action"], "review_required")
        self.assertEqual(preserved["texts"][0]["render_policy"], "preserve_original")
        self.assertEqual(preserved["texts"][0]["translate_policy"], "skip_translation")
        self.assertFalse(preserved["texts"][0]["sfx"]["inpaint_allowed"])
        self.assertFalse(preserved["texts"][0]["skip_processing"])
        self.assertTrue(preserved["texts"][0].get("preserve_original", False))
        self.assertFalse(cleaned["texts"][0]["skip_processing"])
        self.assertEqual(cleaned["texts"][0]["route_action"], "translate_inpaint_render")
        self.assertEqual(cleaned["texts"][0]["text"], "반짝")

    def test_build_page_result_routes_watermark_as_normal_text(self):
        block = SimpleNamespace(
            xyxy=(20, 12, 180, 42),
            mask=None,
            confidence=0.97,
        )
        image = np.full((80, 220, 3), 255, dtype=np.uint8)

        result = build_page_result(
            image_path="page.jpg",
            image_rgb=image,
            blocks=[block],
            texts=["Read at ASURACOMIC.NET"],
            idioma_origem="en",
        )

        self.assertEqual(len(result["texts"]), 1)
        text = result["texts"][0]
        self.assertEqual(text["route_action"], "translate_inpaint_render")
        self.assertNotEqual(text.get("route_reason"), "watermark_detected")
        self.assertFalse(text.get("is_watermark", False))
        self.assertFalse(text["skip_processing"])
        self.assertEqual(text["content_class"], "text")

    def test_build_page_result_drops_scanlation_credit_on_art_background(self):
        block = SimpleNamespace(
            xyxy=(52, 742, 260, 782),
            mask=None,
            confidence=0.69,
            line_polygons=[],
        )
        image = np.full((900, 360, 3), [38, 42, 49], dtype=np.uint8)

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["HIVETOON. COM"],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_short_unknown_punctuated_white_art_fragment(self):
        block = SimpleNamespace(
            xyxy=(191, 1502, 600, 1655),
            mask=None,
            confidence=0.56,
            line_polygons=[],
        )
        image = np.full((1751, 690, 3), 255, dtype=np.uint8)

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["? doy"],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_numeric_visual_fragment_in_large_white_region(self):
        block = SimpleNamespace(
            xyxy=(442, 1233, 552, 1499),
            mask=None,
            confidence=0.56,
            line_polygons=[],
        )
        image = np.full((1751, 690, 3), 255, dtype=np.uint8)

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["400"],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_uppercase_short_unknown_large_white_fragment(self):
        block = SimpleNamespace(
            xyxy=(174, 380, 526, 529),
            mask=None,
            confidence=0.56,
            line_polygons=[],
        )
        image = np.full((1751, 690, 3), 255, dtype=np.uint8)

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["CNG"],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_routes_short_english_dialogue_for_full_pipeline(self):
        block = SimpleNamespace(
            xyxy=(20, 12, 160, 58),
            mask=None,
            confidence=0.96,
        )
        image = np.full((90, 220, 3), 255, dtype=np.uint8)

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True), patch(
            "vision_stack.runtime.classify_text_type", return_value="fala"
        ):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["I am here."],
                idioma_origem="en",
            )

        self.assertEqual(len(result["texts"]), 1)
        text = result["texts"][0]
        self.assertEqual(text["route_action"], "translate_inpaint_render")
        self.assertEqual(text["route_reason"], "dialogue_balloon_with_english_text")
        self.assertFalse(text["skip_processing"])
        self.assertEqual(text["content_class"], "text")

    def test_build_page_result_drops_non_english_sfx_for_english_source(self):
        block = SimpleNamespace(
            xyxy=(30, 18, 98, 54),
            mask=None,
            confidence=0.91,
        )
        image = np.full((90, 130, 3), 42, dtype=np.uint8)

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["쿵쿵"],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_short_latin_misread_from_large_cjk_visual_region_for_english_source(self):
        block = SimpleNamespace(
            xyxy=(42, 28, 222, 132),
            mask=None,
            confidence=0.56,
        )
        image = np.full((170, 280, 3), 255, dtype=np.uint8)
        image[54:92, 78:94] = 8
        image[88:100, 68:112] = 8
        image[64:112, 146:158] = 8
        image[108:122, 132:178] = 8
        image[70:116, 202:216] = 8

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["Un"],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_short_unknown_line_geometry_art_fragment_for_english_source(self):
        block = SimpleNamespace(
            xyxy=(130, 660, 571, 802),
            mask=None,
            confidence=0.86,
            line_polygons=[[[130, 663], [570, 660], [571, 802], [131, 805]]],
        )
        image = np.full((16383, 800, 3), 24, dtype=np.uint8)

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["SIE"],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_tahey_line_polygon_misread_from_cjk_sfx_for_english_source(self):
        block = SimpleNamespace(
            xyxy=(42, 28, 242, 148),
            mask=None,
            confidence=0.62,
            line_polygons=[[[42, 28], [242, 28], [242, 148], [42, 148]]],
        )
        image = np.full((190, 300, 3), 88, dtype=np.uint8)
        for x1, y1, x2, y2 in (
            (76, 52, 88, 122),
            (102, 78, 148, 92),
            (150, 52, 162, 124),
            (184, 48, 206, 120),
            (222, 62, 232, 136),
        ):
            cv2.line(image, (x1, y1), (x2, y2), (238, 238, 238), 8, cv2.LINE_AA)

        raw_text = {
            "text": "TAHEY",
            "line_polygons": [[[42, 28], [242, 28], [242, 148], [42, 148]]],
        }

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[raw_text],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_short_latin_cjk_visual_misread_drops_narrow_tall_hol(self):
        image = np.full((220, 140, 3), 255, dtype=np.uint8)
        for x1, x2 in (
            (28, 33),
            (39, 44),
            (50, 55),
        ):
            cv2.rectangle(image, (x1, 20), (x2, 175), (8, 8, 8), -1)

        self.assertTrue(
            _looks_like_short_latin_cjk_visual_misread(
                image,
                [20, 18, 79, 178],
                "Hol",
                raw_record={
                    "line_polygons": [
                        [[20, 18], [79, 18], [79, 178], [20, 178]],
                    ],
                },
                block=None,
                is_white_balloon_context=False,
            )
        )

    def test_short_latin_cjk_visual_misread_keeps_whitelist_without_drop(self):
        image = np.full((220, 140, 3), 255, dtype=np.uint8)
        for x1, x2 in (
            (28, 33),
            (39, 44),
            (50, 55),
        ):
            cv2.rectangle(image, (x1, 20), (x2, 175), (8, 8, 8), -1)

        for token in ("OK", "NO", "YES", "HUH"):
            with self.subTest(token=token):
                self.assertFalse(
                    _looks_like_short_latin_cjk_visual_misread(
                        image,
                        [20, 18, 79, 178],
                        token,
                        raw_record={
                            "line_polygons": [
                                [[20, 18], [79, 18], [79, 178], [20, 178]],
                            ],
                        },
                        block=None,
                        is_white_balloon_context=False,
                    )
                )

    def test_build_page_result_keeps_known_english_sfx_token_in_visual_region(self):
        block = SimpleNamespace(
            xyxy=(42, 28, 242, 148),
            mask=None,
            confidence=0.62,
            line_polygons=[[[42, 28], [242, 28], [242, 148], [42, 148]]],
        )
        image = np.full((190, 300, 3), 88, dtype=np.uint8)
        for x1, y1, x2, y2 in (
            (76, 52, 88, 122),
            (102, 78, 148, 92),
            (150, 52, 162, 124),
            (184, 48, 206, 120),
            (222, 62, 232, 136),
        ):
            cv2.line(image, (x1, y1), (x2, y2), (238, 238, 238), 8, cv2.LINE_AA)

        raw_text = {
            "text": "CLICK",
            "line_polygons": [[[42, 28], [242, 28], [242, 148], [42, 148]]],
        }

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[raw_text],
                idioma_origem="en",
            )

        self.assertEqual(len(result["texts"]), 1)
        self.assertEqual(result["texts"][0]["text"], "CLICK")
        self.assertEqual(result["texts"][0]["route_action"], "translate_inpaint_render")

    def test_build_page_result_drops_split_latin_sfx_misread_from_large_white_region(self):
        block = SimpleNamespace(
            xyxy=(66, 96, 544, 482),
            mask=None,
            confidence=0.284,
            line_polygons=[
                [[197, 194], [414, 159], [455, 426], [238, 461]],
                [[93, 171], [222, 171], [222, 316], [93, 316]],
            ],
        )
        image = np.full((620, 700, 3), 255, dtype=np.uint8)
        raw_text = {
            "text": "of n",
            "line_polygons": block.line_polygons,
            "text_pixel_bbox": [93, 159, 455, 461],
        }

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[raw_text],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_single_letter_misread_from_huge_white_region(self):
        block = SimpleNamespace(
            xyxy=(165, 579, 983, 956),
            mask=None,
            confidence=0.56,
            line_polygons=[],
        )
        image = np.full((1100, 1250, 3), 232, dtype=np.uint8)
        raw_text = {
            "text": "A",
            "text_pixel_bbox": [242, 579, 983, 903],
        }

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[raw_text],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_scene_art_word_without_glyph_evidence(self):
        block = SimpleNamespace(
            xyxy=(851, 96, 1367, 473),
            mask=None,
            confidence=0.56,
            line_polygons=[],
        )
        image = np.full((620, 1600, 3), [158, 167, 155], dtype=np.uint8)
        for x in range(900, 1330, 42):
            image[20:560, x:x + 8] = [70, 78, 72]
            image[80:140, x + 12:x + 28] = [205, 213, 205]
            image[330:390, x + 12:x + 28] = [205, 213, 205]

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[{"text": "PICTURES"}],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_cover_credit_or_ui_art_text_without_title_gate(self):
        block = SimpleNamespace(
            xyxy=(34, 44, 445, 157),
            mask=None,
            confidence=0.475,
            line_polygons=[],
        )
        image = np.full((900, 1600, 3), [16, 10, 31], dtype=np.uint8)

        with patch("vision_stack.runtime.infer_page_profile", return_value="cover_opening"), patch(
            "vision_stack.runtime._is_white_balloon_context_for_text", return_value=False
        ):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[{"text": "HIVE Scan"}],
                idioma_origem="en",
                work_title="",
                work_title_user_provided=False,
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_cover_site_credit_without_title_gate(self):
        block = SimpleNamespace(
            xyxy=(672, 717, 978, 784),
            mask=None,
            confidence=0.708,
            line_polygons=[],
        )
        image = np.full((7824, 1600, 3), [128, 23, 23], dtype=np.uint8)

        with patch("vision_stack.runtime.infer_page_profile", return_value="cover_opening"), patch(
            "vision_stack.runtime._is_white_balloon_context_for_text", return_value=False
        ):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[{"text": "HIVETOONCOM"}],
                idioma_origem="en",
                work_title="",
                work_title_user_provided=False,
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_cover_gibberish_ui_art_ocr(self):
        block = SimpleNamespace(
            xyxy=(571, 227, 1285, 663),
            mask=None,
            confidence=0.817,
            line_polygons=[
                [[564, 218], [994, 247], [986, 361], [556, 332]],
                [[724, 360], [1289, 341], [1293, 455], [728, 474]],
                [[808, 489], [1083, 489], [1083, 570], [808, 570]],
            ],
        )
        image = np.full((900, 1600, 3), [185, 73, 79], dtype=np.uint8)

        with patch("vision_stack.runtime.infer_page_profile", return_value="cover_opening"), patch(
            "vision_stack.runtime._is_white_balloon_context_for_text", return_value=False
        ):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[
                    {
                        "text": "GCHKO DRANGER ELMOGY RDA ELMOGYTS",
                        "line_polygons": block.line_polygons,
                        "ui_layout_evidence": {
                            "source": "uied_cv",
                            "role": "label_near_components",
                            "confidence": 0.52,
                        },
                    }
                ],
                idioma_origem="en",
                work_title="",
                work_title_user_provided=False,
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_build_page_result_drops_cover_single_word_gibberish_art_ocr(self):
        block = SimpleNamespace(
            xyxy=(571, 227, 994, 361),
            mask=None,
            confidence=0.817,
            line_polygons=[],
        )
        image = np.full((7824, 1600, 3), [185, 73, 79], dtype=np.uint8)

        with patch("vision_stack.runtime.infer_page_profile", return_value="cover_opening"), patch(
            "vision_stack.runtime._is_white_balloon_context_for_text", return_value=False
        ):
            result = build_page_result(
                image_path="page.jpg",
                image_rgb=image,
                blocks=[block],
                texts=[{"text": "GCHKO"}],
                idioma_origem="en",
            )

        self.assertEqual(result["texts"], [])
        self.assertEqual(result["_vision_blocks"], [])

    def test_cover_merged_visual_art_text_drops_gibberish_after_cluster_merge(self):
        text = {
            "text": "GCHKO DRANGER ELMOGY RDA ELMOGYTS",
            "confidence": 0.817,
            "page_profile": "cover_opening",
            "merge_reason": "clustered_line_fragments",
            "bbox": [571, 227, 1285, 663],
            "text_pixel_bbox": [571, 227, 1285, 663],
        }

        self.assertTrue(_looks_like_cover_merged_visual_art_text(text, (7824, 1600, 3)))

    def test_uied_form_label_split_does_not_split_real_balloon_text(self):
        line_polygons = [
            [[755, 5961], [1189, 5961], [1189, 6008], [755, 6008]],
            [[755, 6008], [1189, 6008], [1189, 6051], [755, 6051]],
            [[755, 6051], [1189, 6051], [1189, 6151], [755, 6151]],
        ]
        text = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "text": "BUT I'VE COME ALL THE WAY HERE! I CAN'T LOOK LIKE AN AMATEUR!!",
            "line_texts": ["BUT I'VE COME", "ALL THE WAY HERE!", "I CAN'T LOOK LIKE AN AMATEUR!!"],
            "line_polygons": line_polygons,
            "bbox": [755, 5961, 1189, 6151],
            "source_bbox": [755, 5961, 1189, 6151],
            "balloon_bbox": [523, 5902, 1421, 6110],
            "bubble_id": "page_004_band_043_bubble_001",
            "bubble_mask_bbox": [523, 5902, 1421, 6110],
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "ui_layout_evidence": {"source": "uied_cv", "role": "label_near_components", "confidence": 0.7},
            "qa_flags": [],
        }
        block = dict(text)
        components = [
            {"bbox": [1200, 5940, 1400, 6008], "component_type": "ui_panel", "confidence": 0.6},
            {"bbox": [1200, 6009, 1400, 6051], "component_type": "ui_panel", "confidence": 0.6},
            {"bbox": [1200, 6052, 1400, 6160], "component_type": "ui_panel", "confidence": 0.6},
        ]

        texts, blocks = _split_uied_form_label_texts([text], [block], components, page_number=4)

        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0]["text_id"], "ocr_001")
        self.assertNotIn("uied_form_label_split", texts[0].get("qa_flags", []))

    def test_uied_form_label_split_does_not_split_connected_balloon_evidence(self):
        line_polygons = [
            [[653, 5171], [763, 5194], [748, 5255], [639, 5232]],
            [[583, 5291], [808, 5291], [808, 5346], [583, 5346]],
        ]
        text = {
            "id": "ocr_007",
            "text_id": "ocr_007",
            "text": "THEY APANESE",
            "line_texts": ["THEY", "APANESE"],
            "line_polygons": line_polygons,
            "bbox": [583, 5171, 809, 5347],
            "source_bbox": [583, 5171, 809, 5347],
            "balloon_bbox": [463, 5058, 929, 5460],
            "connected_lobe_bboxes": [[534, 5133, 858, 5254], [534, 5268, 858, 5385]],
            "balloon_subregions": [[534, 5133, 858, 5254], [534, 5268, 858, 5385]],
            "ui_layout_evidence": {"source": "uied_cv", "role": "label_near_components", "confidence": 0.7},
            "qa_flags": [],
        }
        block = dict(text)
        components = [
            {"bbox": [921, 5228, 998, 5250], "component_type": "ui_input", "confidence": 0.445},
            {"bbox": [919, 5288, 992, 5298], "component_type": "ui_input", "confidence": 0.557},
        ]

        texts, blocks = _split_uied_form_label_texts([text], [block], components, page_number=5)

        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0]["text_id"], "ocr_007")
        self.assertNotIn("uied_form_label_split", texts[0].get("qa_flags", []))

    def test_uied_form_label_split_does_not_split_large_balloon_bbox(self):
        line_polygons = [
            [[540, 258], [851, 258], [851, 304], [540, 304]],
            [[540, 312], [851, 312], [851, 364], [540, 364]],
            [[540, 372], [851, 372], [851, 422], [540, 422]],
        ]
        text = {
            "id": "ocr_002",
            "text_id": "ocr_002",
            "text": "THEY DON'T SEEM JAPANESE.",
            "line_texts": ["THEY", "DON'T SEEM", "JAPANESE."],
            "line_polygons": line_polygons,
            "bbox": [540, 258, 851, 422],
            "source_bbox": [540, 258, 851, 422],
            "balloon_bbox": [463, 180, 929, 500],
            "ui_layout_evidence": {"source": "uied_cv", "role": "label_near_components", "confidence": 0.7},
            "qa_flags": [],
        }
        block = dict(text)
        components = [
            {"bbox": [921, 300, 998, 325], "component_type": "ui_input", "confidence": 0.445},
            {"bbox": [919, 360, 992, 385], "component_type": "ui_input", "confidence": 0.557},
            {"bbox": [919, 420, 992, 445], "component_type": "ui_input", "confidence": 0.557},
        ]

        texts, blocks = _split_uied_form_label_texts([text], [block], components, page_number=5)

        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0]["text_id"], "ocr_002")
        self.assertNotIn("uied_form_label_split", texts[0].get("qa_flags", []))

    def test_uied_form_label_split_does_not_split_punctuated_dialogue(self):
        line_polygons = [
            [[540, 258], [851, 258], [851, 304], [540, 304]],
            [[540, 312], [851, 312], [851, 364], [540, 364]],
            [[540, 372], [851, 372], [851, 422], [540, 422]],
        ]
        text = {
            "id": "ocr_002",
            "text_id": "ocr_002",
            "text": "THEY DON'T SEEM JAPANESE.",
            "line_texts": ["THEY", "DON'T SEEM", "JAPANESE."],
            "line_polygons": line_polygons,
            "bbox": [540, 258, 851, 422],
            "ui_layout_evidence": {"source": "uied_cv", "role": "label_near_components", "confidence": 0.7},
            "qa_flags": [],
        }
        block = dict(text)
        components = [
            {"bbox": [921, 300, 998, 325], "component_type": "ui_input", "confidence": 0.445},
            {"bbox": [919, 360, 992, 385], "component_type": "ui_input", "confidence": 0.557},
            {"bbox": [919, 420, 992, 445], "component_type": "ui_input", "confidence": 0.557},
        ]

        texts, blocks = _split_uied_form_label_texts([text], [block], components, page_number=5)

        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0]["text_id"], "ocr_002")
        self.assertNotIn("uied_form_label_split", texts[0].get("qa_flags", []))

    def test_uied_form_label_split_does_not_split_non_form_dialogue_words(self):
        line_polygons = [
            [[639, 5171], [764, 5171], [764, 5256], [639, 5256]],
            [[583, 5291], [809, 5291], [809, 5347], [583, 5347]],
        ]
        text = {
            "id": "ocr_007",
            "text_id": "ocr_007",
            "text": "THEY APANESE",
            "line_texts": ["THEY", "APANESE"],
            "line_polygons": line_polygons,
            "bbox": [583, 5171, 809, 5347],
            "ui_layout_evidence": {"source": "uied_cv", "role": "label_near_components", "confidence": 0.7},
            "qa_flags": [],
        }
        block = dict(text)
        components = [
            {"bbox": [921, 5228, 998, 5250], "component_type": "ui_input", "confidence": 0.445},
            {"bbox": [919, 5288, 992, 5298], "component_type": "ui_input", "confidence": 0.557},
        ]

        texts, blocks = _split_uied_form_label_texts([text], [block], components, page_number=5)

        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0]["text_id"], "ocr_007")
        self.assertNotIn("uied_form_label_split", texts[0].get("qa_flags", []))

    def test_ocr_guided_mask_absorbs_dark_outline_around_light_cjk_text(self):
        image = np.full((100, 150, 3), [58, 82, 150], dtype=np.uint8)
        image[28:62, 34:96] = [8, 8, 12]
        image[34:56, 42:88] = [245, 245, 245]
        block = {"bbox": [28, 22, 104, 70], "confidence": 0.91}
        text = {"bbox": [28, 22, 104, 70], "text": "CJK"}

        def segmenter(crop):
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            return (gray > 220).astype(np.uint8) * 255

        mask = vision_blocks_to_mask(
            image.shape,
            [block],
            image_rgb=image,
            expand_mask=True,
            mask_strategy="ocr_guided_roi_segmentation",
            ocr_texts=[text],
            text_segmenter=segmenter,
        )

        self.assertEqual(mask[36, 44], 255)
        self.assertEqual(mask[30, 36], 255)
        self.assertEqual(mask[80, 120], 0)

    def test_remap_orientation_recovery_page_restores_texts_blocks_and_masks(self):
        rotated_mask = np.zeros((100, 200), dtype=np.uint8)
        rotated_mask[30:70, 120:160] = 255
        page = {
            "image": "page.png",
            "width": 200,
            "height": 100,
            "texts": [
                {
                    "text": "HELLO",
                    "bbox": [120, 30, 160, 70],
                    "source_bbox": [120, 30, 160, 70],
                    "text_pixel_bbox": [122, 32, 158, 68],
                    "line_polygons": [[[120, 30], [160, 30], [160, 70], [120, 70]]],
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [120, 30, 160, 70],
                    "mask": rotated_mask,
                    "confidence": 0.91,
                }
            ],
        }

        remapped = _remap_orientation_recovery_page(
            page,
            rotation_deg=180,
            original_shape=(100, 200),
            rotated_shape=(100, 200),
        )

        self.assertEqual(remapped["width"], 200)
        self.assertEqual(remapped["height"], 100)
        self.assertEqual(remapped["orientation_recovery_deg"], 180)
        self.assertEqual(remapped["texts"][0]["bbox"], [40, 30, 80, 70])
        self.assertEqual(remapped["texts"][0]["source_bbox"], [40, 30, 80, 70])
        self.assertEqual(remapped["texts"][0]["text_pixel_bbox"], [42, 32, 78, 68])
        self.assertEqual(remapped["texts"][0]["orientation_recovery_deg"], 180)
        self.assertEqual(remapped["_vision_blocks"][0]["bbox"], [40, 30, 80, 70])
        self.assertEqual(remapped["_vision_blocks"][0]["orientation_recovery_deg"], 180)
        self.assertEqual(remapped["_vision_blocks"][0]["mask"].shape, (100, 200))
        self.assertEqual(
            int(np.count_nonzero(remapped["_vision_blocks"][0]["mask"])),
            int(np.count_nonzero(rotated_mask)),
        )

    def test_orientation_recovery_selects_better_rotated_ocr_result(self):
        image = np.full((40, 80, 3), 255, dtype=np.uint8)
        baseline = {
            "image": "page.png",
            "width": 80,
            "height": 40,
            "texts": [],
            "_vision_blocks": [],
            "sem_texto_detectado": True,
        }

        def fake_run(rotated_image, image_label, **_kwargs):
            if image_label.endswith("#rot90"):
                return {
                    "image": image_label,
                    "width": rotated_image.shape[1],
                    "height": rotated_image.shape[0],
                    "texts": [{"text": "HELLO", "bbox": [5, 10, 25, 30]}],
                    "_vision_blocks": [{"bbox": [5, 10, 25, 30], "confidence": 0.9}],
                }
            return {
                "image": image_label,
                "width": rotated_image.shape[1],
                "height": rotated_image.shape[0],
                "texts": [],
                "_vision_blocks": [],
            }

        with patch("vision_stack.runtime._run_detect_ocr_on_image", side_effect=fake_run):
            recovered = _run_orientation_recovery(
                image,
                "page.png",
                baseline,
                profile="rapida",
                progress_callback=None,
                idioma_origem="en",
            )

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered["image"], "page.png")
        self.assertEqual(recovered["orientation_recovery_deg"], 90)
        self.assertEqual(recovered["texts"][0]["bbox"], [10, 15, 30, 35])
        self.assertEqual(recovered["_vision_blocks"][0]["bbox"], [10, 15, 30, 35])
        self.assertFalse(recovered["sem_texto_detectado"])

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
        vision_block = page["_vision_blocks"][0]
        self.assertEqual(vision_block["line_polygons"], rich_item["line_polygons"])
        self.assertEqual(vision_block["text_pixel_bbox"], rich_item["text_pixel_bbox"])
        self.assertEqual(vision_block.get("balloon_type", ""), "")

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

    def test_build_page_result_keeps_connected_dark_lobes_separate(self):
        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            common = {
                "bubble_mask_source": "image_dark_bubble_mask",
                "bubble_mask_bbox": [56, 28, 713, 593],
                "layout_profile": "dark_bubble",
                "qa_flags": ["dark_bubble_oval_reocr"],
            }
            blocks = [
                SimpleNamespace(xyxy=(123, 96, 318, 239), mask=None, confidence=0.88),
                SimpleNamespace(xyxy=(373, 75, 698, 455), mask=None, confidence=0.89),
            ]
            page = build_page_result(
                image_path="005.jpg",
                image_rgb=np.zeros((620, 800, 3), dtype=np.uint8),
                blocks=blocks,
                texts=[
                    {
                        **common,
                        "id": "ocr_001",
                        "text": "The subspace retention is only five minutes",
                        "bbox": [123, 96, 318, 239],
                        "layout_bbox": [131, 112, 312, 231],
                        "text_pixel_bbox": [131, 112, 312, 231],
                        "bubble_id": "page_005_band_078_partial_dark_lobe_1000",
                    },
                    {
                        **common,
                        "id": "ocr_001_002",
                        "text": "space is only utes. If you exceed that time, you will return to your original world!",
                        "bbox": [373, 75, 698, 455],
                        "layout_bbox": [399, 204, 675, 335],
                        "text_pixel_bbox": [131, 112, 312, 231],
                        "line_polygons": [[[131, 112], [312, 112], [312, 231], [131, 231]]],
                        "bubble_id": "page_005_band_078_partial_dark_lobe_1001",
                    },
                ],
            )

            decision_log.finalize_decision_trace()

        self.assertEqual(len(page["texts"]), 2)
        self.assertNotIn("merge_reason", page["texts"][0])
        self.assertNotIn("merge_reason", page["texts"][1])
        self.assertTrue(
            any(
                "original world" in str(text.get("text") or "")
                for text in page["texts"]
            )
        )
        right_lobe = next(
            text
            for text in page["texts"]
            if "original world" in str(text.get("text") or "")
        )
        self.assertEqual(
            right_lobe["text"],
            "If you exceed that time, you will return to your original world!",
        )
        self.assertEqual(right_lobe["text_pixel_bbox"], [399, 204, 675, 335])
        self.assertEqual(right_lobe.get("line_polygons"), [])
        self.assertIn("leading_dark_lobe_duplicate_fragment_removed", right_lobe["qa_flags"])
        self.assertIn("stale_text_pixel_bbox_repaired", right_lobe["qa_flags"])
        right_block = next(
            block
            for block in page["_vision_blocks"]
            if "original world" in str(block.get("text") or "")
        )
        self.assertEqual(right_block["text_pixel_bbox"], [399, 204, 675, 335])

    def test_build_page_result_semantically_repairs_merged_cluster_text(self):
        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            blocks = [
                SimpleNamespace(xyxy=(198, 16, 654, 352), mask=None, confidence=0.634),
                SimpleNamespace(xyxy=(192, 48, 426, 160), mask=None, confidence=0.279),
                SimpleNamespace(xyxy=(360, 220, 661, 363), mask=None, confidence=0.83),
            ]

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
                page = build_page_result(
                    image_path="010.jpg",
                    image_rgb=np.full((420, 720, 3), 255, dtype=np.uint8),
                    blocks=blocks,
                    texts=[
                        "WHAT'S WITHE TONE? ARE YOU ACCUSING ME OF AND DESTROYING OUR",
                        "THAT ARROGANT",
                        "BETRAYING OUR MASTER LINEAGE?",
                    ],
                )

            decision_log.finalize_decision_trace()

        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(
            page["texts"][0]["text"],
            "WHAT'S WITH THAT ARROGANT TONE? ARE YOU ACCUSING ME OF BETRAYING OUR MASTER AND DESTROYING OUR LINEAGE?",
        )

    def test_build_page_result_keeps_white_orphan_line_on_opening_strip_band(self):
        image = np.full((300, 800, 3), 255, dtype=np.uint8)
        cv2.putText(image, "SHE HID THIS MUCH.", (436, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        block = SimpleNamespace(
            xyxy=(436, 112, 713, 142),
            mask=None,
            confidence=0.61,
            detector="white_text_line_orphan_scan",
        )

        page = build_page_result(
            image_path="band_003.jpg",
            image_rgb=image,
            blocks=[block],
            texts=["SHE HID THISMUCH"],
        )

        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["text"], "SHE HID THISMUCH")

    def test_cover_opening_keeps_short_white_balloon_line_with_geometry(self):
        image = np.full((300, 800, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(
            xyxy=(436, 112, 713, 142),
            mask=None,
            confidence=0.611,
            line_polygons=[[[445, 117], [707, 117], [707, 139], [445, 139]]],
        )

        with patch("vision_stack.runtime.infer_page_profile", return_value="cover_opening"), patch(
            "vision_stack.runtime._is_white_balloon_context_for_text",
            return_value=True,
        ):
            page = build_page_result(
                image_path="003.jpg",
                image_rgb=image,
                blocks=[block],
                texts=["SHE HID THISMUCH"],
            )

        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertFalse(text["skip_processing"])
        self.assertNotEqual(text.get("skip_reason"), "cover_logo_or_art_ocr")

    def test_drop_contained_duplicate_ocr_texts_keeps_tight_block(self):
        page_texts = [
            {
                "text": "DON'T HIT SFX: KICK My MOM!",
                "bbox": [80, 60, 300, 160],
                "text_pixel_bbox": [80, 60, 300, 160],
            },
            {
                "text": "DON'T HIT My MOM!",
                "bbox": [122, 84, 244, 126],
                "text_pixel_bbox": [122, 84, 244, 126],
            },
        ]
        vision_blocks = [{"bbox": text["bbox"]} for text in page_texts]

        kept_texts, kept_blocks = _drop_contained_duplicate_ocr_texts(page_texts, vision_blocks, page_number=2)

        self.assertEqual([item["text"] for item in kept_texts], ["DON'T HIT My MOM!"])
        self.assertEqual(kept_blocks, [{"bbox": [122, 84, 244, 126]}])

    def test_drop_contained_duplicate_ocr_texts_removes_near_identical_overlap(self):
        page_texts = [
            {"text": "PLEASE!", "bbox": [473, 2764, 644, 2797], "text_pixel_bbox": [473, 2764, 644, 2797], "confidence": 0.82},
            {"text": "PLEASE!", "bbox": [473, 2765, 641, 2796], "text_pixel_bbox": [473, 2765, 641, 2796], "confidence": 0.81},
        ]
        vision_blocks = [{"bbox": text["bbox"]} for text in page_texts]

        kept_texts, kept_blocks = _drop_contained_duplicate_ocr_texts(page_texts, vision_blocks, page_number=1)

        self.assertEqual(len(kept_texts), 1)
        self.assertEqual(kept_texts[0]["bbox"], [473, 2764, 644, 2797])
        self.assertEqual(kept_blocks, [{"bbox": [473, 2764, 644, 2797]}])

    def test_drop_contained_duplicate_ocr_texts_removes_overmerged_container_with_real_children(self):
        page_texts = [
            {
                "text": "What is it...? Why is he getting What is it...? scared alone?",
                "bbox": [148, 655, 667, 1004],
                "text_pixel_bbox": [148, 655, 667, 1004],
                "confidence": 0.58,
            },
            {
                "text": "What is it...?",
                "bbox": [581, 870, 663, 909],
                "text_pixel_bbox": [581, 870, 663, 909],
                "confidence": 0.92,
            },
            {
                "text": "What is it...? scared alone?",
                "bbox": [463, 933, 629, 1004],
                "text_pixel_bbox": [463, 933, 629, 1004],
                "confidence": 0.89,
            },
        ]
        vision_blocks = [{"bbox": text["bbox"]} for text in page_texts]

        kept_texts, kept_blocks = _drop_contained_duplicate_ocr_texts(page_texts, vision_blocks, page_number=23)

        self.assertEqual([item["text"] for item in kept_texts], ["What is it...?", "What is it...? scared alone?"])
        self.assertEqual(kept_blocks, [{"bbox": [581, 870, 663, 909]}, {"bbox": [463, 933, 629, 1004]}])

    def test_looks_like_cover_editorial_band_detects_cover_credit_layout(self):
        image = np.full((401, 1200, 3), 230, dtype=np.uint8)
        cv2.rectangle(image, (500, 92), (775, 104), (255, 255, 255), -1)
        cv2.rectangle(image, (790, 52), (1120, 70), (255, 255, 255), -1)
        cv2.rectangle(image, (810, 16), (1040, 22), (255, 255, 255), -1)
        blocks = [
            SimpleNamespace(xyxy=(590, 21, 979, 100)),
            SimpleNamespace(xyxy=(958, 144, 1096, 179)),
            SimpleNamespace(xyxy=(648, 160, 814, 194)),
            SimpleNamespace(xyxy=(188, 190, 393, 233)),
            SimpleNamespace(xyxy=(880, 229, 1029, 261)),
            SimpleNamespace(xyxy=(637, 253, 726, 280)),
            SimpleNamespace(xyxy=(286, 284, 504, 380)),
            SimpleNamespace(xyxy=(0, 267, 174, 294)),
            SimpleNamespace(xyxy=(945, 312, 1119, 347)),
            SimpleNamespace(xyxy=(652, 328, 827, 363)),
        ]

        self.assertTrue(_looks_like_cover_editorial_band(image, blocks, source_page_number=1))
        self.assertFalse(_looks_like_cover_editorial_band(image, blocks, source_page_number=3))

    def test_should_merge_ocr_cluster_keeps_separate_stacked_white_balloons(self):
        texts = [
            {
                "text": "THERE ARE WAY MORE OF THEM THAN WAS REPORTED TOO!",
                "bbox": [406, 2079, 740, 2196],
                "balloon_type": "white",
            },
            {
                "text": "AT THIS RATE, WE'RE ALL GONNA DIE-",
                "bbox": [673, 2220, 905, 2338],
                "balloon_type": "white",
            },
        ]

        should_merge = _should_merge_ocr_cluster(texts, [277, 1969, 1032, 2436])

        self.assertFalse(should_merge)

    def test_should_merge_ocr_cluster_keeps_separate_diagonal_worker_white_balloons(self):
        texts = [
            {
                "text": "AT THIS RATE, WE'RE ALL GONNA DIE-",
                "bbox": [656, 2201, 917, 2349],
                "balloon_type": "white",
            },
            {
                "text": "THERE ARE WAY MORE OF THEM THAN WAS REPORTED TOO!",
                "bbox": [392, 2056, 751, 2209],
                "balloon_type": "white",
            },
        ]

        should_merge = _should_merge_ocr_cluster(texts, [392, 2056, 917, 2349])

        self.assertFalse(should_merge)

    def test_should_merge_ocr_cluster_keeps_distinct_bubble_masks_separate(self):
        texts = [
            {
                "id": "ocr_001",
                "text": "HEY, LET'S GO! I'M STARVING",
                "bbox": [148, 7248, 310, 7268],
                "text_pixel_bbox": [148, 7248, 310, 7268],
                "bubble_mask_bbox": [29, 7109, 642, 7722],
                "balloon_bbox": [0, 7013, 800, 7967],
                "balloon_type": "white",
            },
            {
                "id": "ocr_003",
                "text": "WHO'S PAYING TODAY?",
                "bbox": [344, 7702, 540, 7761],
                "text_pixel_bbox": [344, 7702, 540, 7761],
                "bubble_mask_bbox": [276, 7612, 598, 7799],
                "balloon_bbox": [276, 7612, 598, 7799],
                "balloon_type": "white",
            },
        ]

        should_merge = _should_merge_ocr_cluster(texts, [29, 7109, 642, 7799])

        self.assertFalse(should_merge)

    def test_merge_ocr_clusters_keeps_distinct_bubble_masks_separate(self):
        texts = [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_035",
                "text": "HEY, LET'S GO! I'M STARVING",
                "bbox": [148, 7248, 310, 7268],
                "source_bbox": [148, 7248, 310, 7268],
                "text_pixel_bbox": [148, 7248, 310, 7268],
                "bubble_mask_bbox": [29, 7109, 642, 7722],
                "balloon_bbox": [0, 7013, 800, 7967],
                "confidence": 0.92,
                "balloon_type": "white",
            },
            {
                "id": "ocr_003",
                "text_id": "ocr_003",
                "trace_id": "ocr_003@page_003_band_035",
                "text": "WHO'S PAYING TODAY?",
                "bbox": [344, 7702, 540, 7761],
                "source_bbox": [325, 7647, 549, 7764],
                "text_pixel_bbox": [344, 7702, 540, 7761],
                "bubble_mask_bbox": [276, 7612, 598, 7799],
                "balloon_bbox": [276, 7612, 598, 7799],
                "confidence": 0.94,
                "balloon_type": "white",
            },
        ]
        blocks = [
            {"bbox": [29, 7109, 642, 7722], "confidence": 0.92, "balloon_type": "white"},
            {"bbox": [276, 7612, 598, 7799], "confidence": 0.94, "balloon_type": "white"},
        ]

        merged_texts, merged_blocks = _merge_ocr_clusters(texts, blocks, (8000, 800, 3), page_number=3)

        self.assertEqual([text["id"] for text in merged_texts], ["ocr_001", "ocr_003"])
        self.assertEqual(len(merged_blocks), 2)

    def test_should_merge_ocr_cluster_merges_touching_sparse_narration_lines(self):
        texts = [
            {
                "text": "ONCE THAT FORCE",
                "bbox": [171, 16, 545, 89],
                "balloon_type": "textured",
            },
            {
                "text": "IS RELEASED",
                "bbox": [201, 89, 513, 194],
                "balloon_type": "textured",
            },
        ]

        should_merge = _should_merge_ocr_cluster(texts, [118, 0, 598, 210])

        self.assertTrue(should_merge)

    def test_merge_ocr_clusters_merges_mixed_same_balloon_tail_line(self):
        texts = [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_007",
                "text": "PLEASE, FOR THE CHILD'S",
                "bbox": [498, 5655, 656, 5707],
                "source_bbox": [25, 5436, 667, 5708],
                "text_pixel_bbox": [498, 5655, 656, 5707],
                "line_polygons": [
                    [[499, 5655], [655, 5655], [655, 5681], [499, 5681]],
                    [[506, 5684], [647, 5684], [647, 5707], [506, 5707]],
                ],
                "confidence": 0.93,
                "tipo": "fala",
                "balloon_type": "textured",
                "block_profile": "standard",
            },
            {
                "id": "ocr_002",
                "text_id": "ocr_002",
                "trace_id": "ocr_002@page_001_band_007",
                "text": "SAKE.",
                "bbox": [546, 5720, 612, 5740],
                "source_bbox": [497, 5648, 660, 5741],
                "text_pixel_bbox": [546, 5720, 612, 5740],
                "line_polygons": [[[546, 5720], [612, 5720], [612, 5740], [546, 5740]]],
                "confidence": 0.94,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
        ]
        blocks = [
            {"bbox": text["source_bbox"], "confidence": text["confidence"], "balloon_type": text["balloon_type"]}
            for text in texts
        ]

        merged_texts, merged_blocks = _merge_ocr_clusters(texts, blocks, (6200, 760, 3), page_number=1)

        self.assertEqual(len(merged_texts), 2)
        self.assertEqual([text["text"] for text in merged_texts], ["PLEASE, FOR THE CHILD'S", "SAKE."])
        self.assertEqual(len(merged_blocks), 2)

    def test_merge_ocr_clusters_keeps_white_card_separate_from_textured_news_title(self):
        texts = [
            {
                "id": "ocr_002",
                "text_id": "ocr_002",
                "trace_id": "ocr_002@page_020_band_108",
                "text": "WHEN I'M WATCHING A HORROR MOVIE, I GET SCARED.",
                "bbox": [17, 16, 624, 735],
                "source_bbox": [0, 0, 684, 3267],
                "text_pixel_bbox": [17, 16, 624, 735],
                "line_polygons": [
                    [[30, 30], [610, 30], [610, 88], [30, 88]],
                    [[30, 110], [610, 110], [610, 168], [30, 168]],
                ],
                "confidence": 0.91,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
                "background_rgb": [255, 255, 252],
            },
            {
                "id": "ocr_004",
                "text_id": "ocr_004",
                "trace_id": "ocr_004@page_020_band_108",
                "text": "RDAY'S HOT NEWS COLLECTION",
                "bbox": [5, 766, 351, 941],
                "source_bbox": [0, 740, 437, 984],
                "text_pixel_bbox": [5, 766, 351, 941],
                "line_polygons": [[[5, 766], [351, 766], [351, 941], [5, 941]]],
                "confidence": 0.82,
                "tipo": "fala",
                "balloon_type": "textured",
                "block_profile": "standard",
                "background_rgb": [188, 136, 55],
            },
        ]
        blocks = [
            {"bbox": text["source_bbox"], "confidence": text["confidence"], "balloon_type": text["balloon_type"]}
            for text in texts
        ]

        merged_texts, merged_blocks = _merge_ocr_clusters(texts, blocks, (3300, 684, 3), page_number=20)

        self.assertEqual([text["id"] for text in merged_texts], ["ocr_002", "ocr_004"])
        self.assertEqual(merged_texts[1]["text"], "RDAY'S HOT NEWS COLLECTION")
        self.assertEqual(len(merged_blocks), 2)

    def test_finalize_page_ocr_texts_drops_short_art_noise_before_tail_merge(self):
        texts = [
            {
                "text": "PLEASE, FOR THE CHILD'S",
                "translated": "POR FAVOR, PELO BEM",
                "bbox": [498, 5655, 656, 5707],
                "source_bbox": [25, 5436, 667, 5708],
                "text_pixel_bbox": [498, 5655, 656, 5707],
                "line_polygons": [
                    [[498, 5652], [658, 5654], [658, 5678], [498, 5676]],
                    [[507, 5686], [648, 5686], [648, 5707], [507, 5707]],
                ],
                "confidence": 0.93,
                "tipo": "fala",
                "balloon_type": "textured",
                "block_profile": "standard",
            },
            {
                "text": "SAKE.",
                "translated": "DA CRIANCA.",
                "bbox": [546, 5720, 612, 5740],
                "source_bbox": [497, 5648, 660, 5741],
                "text_pixel_bbox": [546, 5720, 612, 5740],
                "line_polygons": [[[543, 5718], [615, 5718], [615, 5743], [543, 5743]]],
                "confidence": 0.94,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
            {
                "text": "THE SA",
                "translated": "O SA",
                "bbox": [320, 5699, 562, 5949],
                "source_bbox": [320, 5699, 562, 5949],
                "text_pixel_bbox": [320, 5699, 562, 5949],
                "line_polygons": [],
                "confidence": 0.80,
                "tipo": "fala",
                "balloon_type": "textured",
                "block_profile": "standard",
            },
        ]
        blocks = [
            {"bbox": text["source_bbox"], "text_pixel_bbox": text["text_pixel_bbox"], "line_polygons": text["line_polygons"], "confidence": text["confidence"], "balloon_type": text["balloon_type"]}
            for text in texts
        ]

        final_texts, final_blocks = _finalize_page_ocr_texts(texts, blocks, (6200, 760, 3), page_number=1)

        self.assertEqual([text["text"] for text in final_texts], ["PLEASE, FOR THE CHILD'S", "THE SA", "SAKE."])
        self.assertEqual(len(final_blocks), 3)

    def test_accepted_ocr_system_ui_rescue_allows_short_status_sentence(self):
        text = {
            "text": "Main Quest will be shown shortly",
            "bbox": [0, 0, 202, 209],
            "source_bbox": [0, 0, 202, 209],
            "text_pixel_bbox": [34, 72, 178, 146],
            "confidence": 0.86,
            "route_action": "translate_inpaint_render",
            "skip_processing": False,
        }

        self.assertTrue(_accepted_ocr_system_ui_rescue_allowed(text, (261, 800, 3)))

    def test_finalize_page_ocr_texts_rescues_accepted_system_ui_if_finalizer_drops_it(self):
        texts = [
            {
                "id": "ocr_001",
                "text": "Main Quest will be shown shortly",
                "bbox": [0, 0, 202, 209],
                "source_bbox": [0, 0, 202, 209],
                "text_pixel_bbox": [34, 72, 178, 146],
                "line_polygons": [
                    [[34, 72], [178, 72], [178, 98], [34, 98]],
                    [[52, 112], [160, 112], [160, 146], [52, 146]],
                ],
                "confidence": 0.854,
                "tipo": "text",
                "balloon_type": "",
                "block_profile": "standard",
                "page_profile": "standard",
                "route_action": "translate_inpaint_render",
                "skip_processing": False,
            }
        ]
        blocks = [
            {
                "text_id": "ocr_001",
                "bbox": [0, 0, 202, 209],
                "source_bbox": [0, 0, 202, 209],
                "text_pixel_bbox": [34, 72, 178, 146],
                "line_polygons": texts[0]["line_polygons"],
                "confidence": 0.854,
                "text": "Main Quest will be shown shortly",
            }
        ]

        with patch("vision_stack.runtime._filter_page_ocr_noise", return_value=([], [])):
            final_texts, final_blocks = _finalize_page_ocr_texts(texts, blocks, (261, 800, 3), page_number=2)

        self.assertEqual(len(final_texts), 1)
        self.assertEqual(final_texts[0]["text"], "Main Quest will be shown shortly")
        self.assertIn("accepted_ocr_finalizer_rescue", final_texts[0]["qa_flags"])
        self.assertEqual(final_blocks[0]["text"], "Main Quest will be shown shortly")

    def test_finalize_page_ocr_texts_does_not_rescue_scanlation_or_short_noise(self):
        texts = [
            {
                "id": "ocr_001",
                "text": "SECRET SCANS PRESENTS",
                "bbox": [20, 30, 360, 90],
                "source_bbox": [20, 30, 360, 90],
                "text_pixel_bbox": [20, 30, 360, 90],
                "line_polygons": [],
                "confidence": 0.93,
                "tipo": "text",
                "block_profile": "standard",
                "route_action": "translate_inpaint_render",
                "skip_processing": False,
            },
            {
                "id": "ocr_002",
                "text": "002",
                "bbox": [334, 96, 475, 165],
                "source_bbox": [334, 96, 475, 165],
                "text_pixel_bbox": [334, 96, 475, 165],
                "line_polygons": [],
                "confidence": 0.86,
                "tipo": "text",
                "block_profile": "standard",
                "route_action": "translate_inpaint_render",
                "skip_processing": False,
            },
        ]
        blocks = [
            {
                "text_id": text["id"],
                "bbox": text["source_bbox"],
                "source_bbox": text["source_bbox"],
                "text_pixel_bbox": text["text_pixel_bbox"],
                "line_polygons": [],
                "confidence": text["confidence"],
                "text": text["text"],
            }
            for text in texts
        ]

        with patch("vision_stack.runtime._filter_page_ocr_noise", return_value=([], [])):
            final_texts, final_blocks = _finalize_page_ocr_texts(texts, blocks, (620, 800, 3), page_number=1)

        self.assertEqual(final_texts, [])
        self.assertEqual(final_blocks, [])

    def test_finalize_page_ocr_texts_keeps_traced_translated_short_project_layer(self):
        texts = [
            {
                "text": "A normal speech line.",
                "translated": "UMA FALA NORMAL.",
                "bbox": [120, 820, 360, 890],
                "source_bbox": [116, 816, 364, 894],
                "text_pixel_bbox": [124, 826, 354, 880],
                "line_polygons": [[[124, 826], [354, 826], [354, 880], [124, 880]]],
                "confidence": 0.91,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
            {
                "text": "FIRE",
                "translated": "FOGO",
                "bbox": [17, 13502, 516, 13747],
                "source_bbox": [17, 13502, 516, 13747],
                "text_pixel_bbox": [98, 13502, 432, 13691],
                "line_polygons": [],
                "confidence": 0.56,
                "tipo": "fala",
                "balloon_type": "textured",
                "block_profile": "standard",
                "trace_id": "ocr_002@page_005_band_116",
                "source_trace_ids": ["ocr_002@page_005_band_116"],
                "band_id": "page_005_band_116",
            },
        ]
        blocks = [
            {
                "bbox": text["source_bbox"],
                "text_pixel_bbox": text["text_pixel_bbox"],
                "line_polygons": text["line_polygons"],
                "confidence": text["confidence"],
                "balloon_type": text["balloon_type"],
            }
            for text in texts
        ]

        final_texts, _final_blocks = _finalize_page_ocr_texts(
            texts,
            blocks,
            (13845, 800, 3),
            page_number=6,
            total_pages=7,
        )

        self.assertIn("FIRE", [text["text"] for text in final_texts])
        fire = next(text for text in final_texts if text["text"] == "FIRE")
        self.assertEqual(fire["translated"], "FOGO")
        self.assertEqual(fire["trace_id"], "ocr_002@page_005_band_116")

    def test_finalize_page_ocr_texts_drops_partial_duplicate_without_line_polygons(self):
        texts = [
            {
                "text": "AISH! WHY DOYOUKEEP MAKINGUS THE BAD GUYS.",
                "translated": "AISH! POR QUE VOCE CONTINUA NOS TRANSFORMANDO EM BANDIDOS?",
                "bbox": [132, 7619, 322, 7737],
                "source_bbox": [130, 7615, 328, 7739],
                "text_pixel_bbox": [132, 7619, 322, 7737],
                "line_polygons": [
                    [[165, 7619], [291, 7619], [291, 7639], [165, 7639]],
                    [[146, 7651], [309, 7651], [309, 7672], [146, 7672]],
                    [[132, 7685], [324, 7685], [324, 7704], [132, 7704]],
                    [[163, 7715], [289, 7715], [289, 7739], [163, 7739]],
                ],
                "confidence": 0.93,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
            {
                "text": "AISH! WHY DO YOU KEEP",
                "translated": "AISH! POR QUE VOCE CONTINUA",
                "bbox": [146, 7619, 309, 7672],
                "source_bbox": [142, 7616, 313, 7674],
                "text_pixel_bbox": [146, 7619, 309, 7672],
                "line_polygons": [],
                "confidence": 0.80,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
        ]
        blocks = [
            {"bbox": text["source_bbox"], "text_pixel_bbox": text["text_pixel_bbox"], "line_polygons": text["line_polygons"], "confidence": text["confidence"], "balloon_type": text["balloon_type"]}
            for text in texts
        ]

        final_texts, final_blocks = _finalize_page_ocr_texts(texts, blocks, (8200, 760, 3), page_number=1)

        self.assertEqual(len(final_texts), 1)
        self.assertEqual(final_texts[0]["text"], "AISH! WHY DOYOUKEEP MAKINGUS THE BAD GUYS.")
        self.assertEqual(final_texts[0]["translated"], "AISH! POR QUE VOCE CONTINUA NOS TRANSFORMANDO EM BANDIDOS?")
        self.assertEqual(final_blocks[0]["bbox"], [130, 7615, 328, 7739])

    def test_finalize_page_ocr_texts_uses_preserved_bbox_when_balloon_bbox_was_sanitized(self):
        texts = [
            {
                "text": "AFTER ALL, IT'S CANCER, WHY BOTHER USING A PRIVATE LOAN FOR A PATIENT? YOUR LIFE IS SO",
                "translated": "AFINAL, E CANCER, POR QUE SE PREOCUPAR EM USAR UM EMPRESTIMO PRIVADO PARA UM PACIENTE? SUA VIDA E TAO",
                "bbox": [8, 12873, 749, 13652],
                "balloon_bbox": [303, 13497, 691, 13630],
                "text_pixel_bbox": [303, 13497, 691, 13630],
                "line_polygons": [
                    [[304, 13497], [690, 13497], [690, 13528], [304, 13528]],
                    [[314, 13531], [681, 13531], [681, 13562], [314, 13562]],
                    [[330, 13566], [665, 13566], [665, 13596], [330, 13596]],
                    [[359, 13600], [636, 13600], [636, 13630], [359, 13630]],
                ],
                "confidence": 0.91,
                "tipo": "fala",
                "balloon_type": "textured",
                "block_profile": "standard",
            },
            {
                "text": "FRUSTRATING TOO",
                "translated": "FRUSTRANTE TAMBEM",
                "bbox": [296, 13496, 693, 13672],
                "balloon_bbox": [365, 13644, 627, 13667],
                "text_pixel_bbox": [365, 13644, 627, 13667],
                "line_polygons": [[[365, 13644], [627, 13644], [627, 13667], [365, 13667]]],
                "confidence": 0.94,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
        ]
        blocks = [
            {"bbox": text["bbox"], "text_pixel_bbox": text["text_pixel_bbox"], "line_polygons": text["line_polygons"], "confidence": text["confidence"], "balloon_type": text["balloon_type"]}
            for text in texts
        ]

        final_texts, _final_blocks = _finalize_page_ocr_texts(texts, blocks, (14000, 760, 3), page_number=1)

        self.assertEqual(len(final_texts), 2)
        self.assertEqual(final_texts[0]["text"], "AFTER ALL, IT'S CANCER, WHY BOTHER USING A PRIVATE LOAN FOR A PATIENT? YOUR LIFE IS SO")
        self.assertEqual(final_texts[1]["text"], "FRUSTRATING TOO")

    def test_finalize_page_ocr_texts_repairs_overmerged_container_before_nearby_balloon_merge(self):
        texts = [
            {
                "text": "What is it...? Why is he getting",
                "bbox": [148, 96, 667, 439],
                "text_pixel_bbox": [577, 311, 665, 350],
                "line_polygons": [
                    [[577, 311], [665, 311], [665, 328], [577, 328]],
                    [[597, 334], [647, 334], [647, 352], [597, 352]],
                ],
                "confidence": 0.56,
            },
            {
                "text": "What is it...?",
                "bbox": [581, 311, 663, 350],
                "text_pixel_bbox": [581, 311, 663, 350],
                "line_polygons": [[[581, 311], [663, 311], [663, 350], [581, 350]]],
                "confidence": 0.875,
            },
            {
                "text": "scared alone?",
                "bbox": [463, 374, 629, 445],
                "text_pixel_bbox": [463, 374, 629, 445],
                "line_polygons": [
                    [[487, 374], [605, 374], [605, 395], [487, 395]],
                    [[500, 401], [593, 399], [593, 420], [500, 422]],
                    [[463, 428], [630, 428], [630, 445], [463, 445]],
                ],
                "confidence": 0.89,
            },
        ]
        blocks = [
            {"bbox": text["bbox"], "text_pixel_bbox": text["text_pixel_bbox"], "line_polygons": text["line_polygons"], "confidence": text["confidence"]}
            for text in texts
        ]

        final_texts, _final_blocks = _finalize_page_ocr_texts(texts, blocks, (900, 700, 3), page_number=23)

        self.assertEqual([item["text"] for item in final_texts], ["What is it...?", "scared alone?"])

    def test_finalize_page_ocr_texts_merges_valid_pair_inside_larger_mask_region(self):
        texts = [
            {
                "text": "AISH IT'S NOT LIKE WE'RE FOOLS",
                "translated": "AISH, NAO SOMOS TOLOS",
                "bbox": [88, 22, 394, 158],
                "text_pixel_bbox": [88, 22, 394, 158],
                "line_polygons": [],
                "confidence": 0.94,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
            {
                "text": "AFTER ALL, IT'S CANCER, WHY BOTHER USING A PRIVATE LOAN FOR A PATIENT? YOUR LIFE IS SO",
                "translated": "AFINAL, E CANCER, POR QUE SE PREOCUPAR EM USAR UM EMPRESTIMO PRIVADO PARA UM PACIENTE? SUA VIDA E TAO",
                "bbox": [8, 49, 749, 828],
                "balloon_bbox": [303, 673, 691, 806],
                "text_pixel_bbox": [303, 673, 691, 806],
                "line_polygons": [
                    [[304, 673], [690, 673], [690, 704], [304, 704]],
                    [[314, 707], [681, 707], [681, 738], [314, 738]],
                    [[330, 742], [665, 742], [665, 772], [330, 772]],
                    [[359, 776], [636, 776], [636, 806], [359, 806]],
                ],
                "confidence": 0.56,
                "tipo": "fala",
                "balloon_type": "textured",
                "block_profile": "standard",
            },
            {
                "text": "FRUSTRATING TOO",
                "translated": "FRUSTRANTE TAMBEM",
                "bbox": [296, 672, 693, 848],
                "balloon_bbox": [365, 820, 627, 843],
                "text_pixel_bbox": [365, 820, 627, 843],
                "line_polygons": [[[365, 820], [627, 820], [627, 843], [365, 843]]],
                "confidence": 0.94,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
        ]
        blocks = [
            {"bbox": text["bbox"], "text_pixel_bbox": text["text_pixel_bbox"], "line_polygons": text["line_polygons"], "confidence": text["confidence"], "balloon_type": text["balloon_type"]}
            for text in texts
        ]

        final_texts, _final_blocks = _finalize_page_ocr_texts(texts, blocks, (900, 800, 3), page_number=1)

        self.assertEqual(len(final_texts), 3)
        self.assertEqual(final_texts[0]["text"], "AISH IT'S NOT LIKE WE'RE FOOLS")
        self.assertEqual(final_texts[1]["text"], "AFTER ALL, IT'S CANCER, WHY BOTHER USING A PRIVATE LOAN FOR A PATIENT? YOUR LIFE IS SO")
        self.assertEqual(final_texts[2]["text"], "FRUSTRATING TOO")

    def test_finalize_page_ocr_texts_keeps_cover_title_overlay_without_title_gate(self):
        texts = [
            {
                "text": "The God ofdeath Shadow Erian Shadow NTEEM",
                "translated": "O deus da morte sombra erian shadow NTEEM",
                "bbox": [116, 167, 541, 430],
                "text_pixel_bbox": [132, 174, 534, 372],
                "line_polygons": [[[132, 174], [534, 174], [534, 372], [132, 372]]],
                "confidence": 0.77,
                "tipo": "narracao",
                "balloon_type": "white",
                "block_profile": "white_balloon",
                "page_profile": "cover_opening",
            },
            {
                "text": "I'M SORRY... MOM IS VERY SORRY.",
                "translated": "SINTO MUITO... MAMAE SENTE MUITO.",
                "bbox": [217, 64, 406, 170],
                "text_pixel_bbox": [218, 70, 401, 166],
                "line_polygons": [[[218, 70], [401, 70], [401, 166], [218, 166]]],
                "confidence": 0.67,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
                "page_profile": "cover_opening",
            },
        ]
        blocks = [
            {"bbox": text["bbox"], "text_pixel_bbox": text["text_pixel_bbox"], "line_polygons": text["line_polygons"], "confidence": text["confidence"], "balloon_type": text["balloon_type"]}
            for text in texts
        ]

        final_texts, _final_blocks = _finalize_page_ocr_texts(texts, blocks, (1400, 760, 3), page_number=1)

        self.assertEqual(
            [text["text"] for text in final_texts],
            ["I'M SORRY... MOM IS VERY SORRY.", "The God ofdeath Shadow Erian Shadow NTEEM"],
        )

    def test_finalize_page_ocr_texts_keeps_cover_title_like_text_on_middle_page(self):
        texts = [
            {
                "text": "ILIVE LIKETHIS BUTI DON'T HAVE THE COURAGE TOCHANGE MY LIFE",
                "translated": "EU VIVO ASSIM MAS NAO TENHO CORAGEM DE MUDAR MINHA VIDA",
                "bbox": [373, 6075, 671, 6182],
                "text_pixel_bbox": [378, 6085, 665, 6169],
                "line_polygons": [[[378, 6085], [665, 6085], [665, 6169], [378, 6169]]],
                "confidence": 0.819,
                "tipo": "narracao",
                "balloon_type": "white",
                "block_profile": "white_balloon",
                "page_profile": "cover_opening",
            },
            {
                "text": "A regular speech line.",
                "translated": "UMA FALA NORMAL.",
                "bbox": [120, 8200, 360, 8270],
                "text_pixel_bbox": [124, 8210, 354, 8260],
                "line_polygons": [[[124, 8210], [354, 8210], [354, 8260], [124, 8260]]],
                "confidence": 0.91,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
                "page_profile": "cover_opening",
            }
        ]
        blocks = [
            {
                "bbox": text["bbox"],
                "text_pixel_bbox": text["text_pixel_bbox"],
                "line_polygons": text["line_polygons"],
                "confidence": text["confidence"],
                "balloon_type": text["balloon_type"],
            }
            for text in texts
        ]

        final_texts, _final_blocks = _finalize_page_ocr_texts(
            texts,
            blocks,
            (13845, 800, 3),
            page_number=3,
            total_pages=6,
        )

        self.assertEqual([text["text"] for text in final_texts], [texts[0]["text"], texts[1]["text"]])

    def test_finalize_page_ocr_texts_drops_cover_title_like_footer_only_on_last_page(self):
        texts = [
            {
                "text": "SPECIAL THANKS PRODUCTION GROUP STAFF CREDIT",
                "translated": "AGRADECIMENTOS ESPECIAIS",
                "bbox": [80, 12800, 700, 13680],
                "text_pixel_bbox": [100, 12840, 680, 13620],
                "line_polygons": [[[100, 12840], [680, 12840], [680, 13620], [100, 13620]]],
                "confidence": 0.82,
                "tipo": "narracao",
                "balloon_type": "white",
                "block_profile": "white_balloon",
                "page_profile": "cover_opening",
            },
            {
                "text": "END.",
                "translated": "FIM.",
                "bbox": [300, 300, 420, 350],
                "text_pixel_bbox": [304, 304, 416, 346],
                "line_polygons": [[[304, 304], [416, 304], [416, 346], [304, 346]]],
                "confidence": 0.91,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
                "page_profile": "cover_opening",
            },
        ]
        blocks = [
            {
                "bbox": text["bbox"],
                "text_pixel_bbox": text["text_pixel_bbox"],
                "line_polygons": text["line_polygons"],
                "confidence": text["confidence"],
                "balloon_type": text["balloon_type"],
            }
            for text in texts
        ]

        middle_texts, _ = _finalize_page_ocr_texts(texts, blocks, (13845, 800, 3), page_number=5, total_pages=6)
        last_texts, _ = _finalize_page_ocr_texts(texts, blocks, (13845, 800, 3), page_number=6, total_pages=6)

        self.assertEqual([text["text"] for text in middle_texts], [texts[1]["text"], texts[0]["text"]])
        self.assertEqual([text["text"] for text in last_texts], [texts[1]["text"], texts[0]["text"]])

    def test_merge_ocr_clusters_merges_mixed_same_balloon_short_bottom_line(self):
        texts = [
            {
                "text": "AFTER ALL, IT'S CANCER, WHY BOTHER USING A PRIVATE LOAN FOR A PATIENT? YOUR LIFE IS SO",
                "bbox": [303, 13497, 691, 13630],
                "source_bbox": [8, 12873, 749, 13652],
                "text_pixel_bbox": [303, 13497, 691, 13630],
                "line_polygons": [
                    [[304, 13497], [690, 13497], [690, 13528], [304, 13528]],
                    [[314, 13531], [681, 13531], [681, 13562], [314, 13562]],
                    [[330, 13566], [665, 13566], [665, 13596], [330, 13596]],
                    [[359, 13600], [636, 13600], [636, 13630], [359, 13630]],
                ],
                "confidence": 0.91,
                "tipo": "fala",
                "balloon_type": "textured",
                "block_profile": "standard",
            },
            {
                "text": "FRUSTRATING TOO",
                "bbox": [365, 13644, 627, 13667],
                "source_bbox": [296, 13496, 693, 13672],
                "text_pixel_bbox": [365, 13644, 627, 13667],
                "line_polygons": [[[365, 13644], [627, 13644], [627, 13667], [365, 13667]]],
                "confidence": 0.94,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
        ]
        blocks = [
            {"bbox": text["source_bbox"], "confidence": text["confidence"], "balloon_type": text["balloon_type"]}
            for text in texts
        ]

        merged_texts, _merged_blocks = _merge_ocr_clusters(texts, blocks, (14000, 760, 3), page_number=1)

        self.assertEqual(len(merged_texts), 2)
        self.assertEqual([text["text"] for text in merged_texts], [
            "AFTER ALL, IT'S CANCER, WHY BOTHER USING A PRIVATE LOAN FOR A PATIENT? YOUR LIFE IS SO",
            "FRUSTRATING TOO",
        ])

    def test_merge_ocr_clusters_keeps_mixed_diagonal_balloons_separate(self):
        texts = [
            {
                "text": "FIRST BALLOON",
                "bbox": [100, 100, 260, 160],
                "source_bbox": [80, 80, 280, 180],
                "confidence": 0.92,
                "tipo": "fala",
                "balloon_type": "textured",
                "block_profile": "standard",
            },
            {
                "text": "SECOND BALLOON",
                "bbox": [300, 170, 460, 230],
                "source_bbox": [282, 150, 482, 250],
                "confidence": 0.93,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
        ]
        blocks = [
            {"bbox": text["source_bbox"], "confidence": text["confidence"], "balloon_type": text["balloon_type"]}
            for text in texts
        ]

        merged_texts, merged_blocks = _merge_ocr_clusters(texts, blocks, (400, 620, 3), page_number=1)

        self.assertEqual([text["text"] for text in merged_texts], ["FIRST BALLOON", "SECOND BALLOON"])
        self.assertEqual(len(merged_blocks), 2)

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

        self.assertTrue(page["texts"])
        self.assertTrue(all(item.get("balloon_type", "") == "" for item in page["texts"]))

    def test_build_page_result_marks_cover_logo_noise_skip_processing_before_translation(self):
        block = SimpleNamespace(xyxy=(100, 500, 360, 630), mask=None, confidence=0.94, line_polygons=[])

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False):
            page = build_page_result(
                image_path="001.jpg",
                image_rgb=np.full((1000, 700, 3), 42, dtype=np.uint8),
                blocks=[block],
                texts=["Shadow Erian Shadow"],
            )

        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertEqual(text["text"], "Shadow Erian Shadow")
        self.assertFalse(text["skip_processing"])
        self.assertNotEqual(text.get("skip_reason"), "cover_repeated_words_noise")
        self.assertEqual(text["translate_policy"], "translate")

    def test_cover_opening_keeps_candidate_inquiry_ui_text_for_translation(self):
        block = SimpleNamespace(
            xyxy=(212, 16, 435, 68),
            mask=None,
            confidence=0.891,
            line_polygons=[],
        )

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False), patch(
            "vision_stack.runtime.classify_text_type",
            return_value="narracao",
        ):
            page = build_page_result(
                image_path="003.jpg",
                image_rgb=np.full((180, 690, 3), 235, dtype=np.uint8),
                blocks=[block],
                texts=["Go to successful candidate inquiry"],
            )

        self.assertEqual(page["page_profile"], "cover_opening")
        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertFalse(text["skip_processing"])
        self.assertNotEqual(text.get("skip_reason"), "cover_logo_or_art_ocr")
        self.assertEqual(text["translate_policy"], "translate")

    def test_cover_opening_repeated_words_keeps_sentence_white_balloon_speech(self):
        block = SimpleNamespace(
            xyxy=(253, 10560, 570, 10752),
            mask=None,
            confidence=0.71,
            line_polygons=[[[253, 10560], [570, 10560], [570, 10752], [253, 10752]]],
        )

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            page = build_page_result(
                image_path="001.jpg",
                image_rgb=np.full((13832, 800, 3), 246, dtype=np.uint8),
                blocks=[block],
                texts=["YOU KNOW, REAL-LIFEINSURANCE, STUFF LIKE THAT? IF YOU DON'T HAVE MONEY, YOU HAVE TOSHOWYOUR SINCERITY."],
            )

        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertFalse(text["skip_processing"])
        self.assertEqual(text["content_class"], "text")
        self.assertNotEqual(text.get("skip_reason"), "cover_repeated_words_noise")

    def test_cover_opening_repeated_words_keeps_sentence_on_plain_bright_balloon(self):
        block = SimpleNamespace(
            xyxy=(88, 21626, 445, 21797),
            mask=None,
            confidence=0.95,
            line_polygons=[
                [[88, 21626], [445, 21626], [445, 21650], [88, 21650]],
            ],
        )

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False):
            page = build_page_result(
                image_path="001.jpg",
                image_rgb=np.full((24000, 800, 3), 255, dtype=np.uint8),
                blocks=[block],
                texts=[
                    "AJUMMAYOU DON'T KNOW ME, DO YOU? THERE'SNO HOUSE THAT I CAN'T GETMONEYFROM. DIDYOU REALLY THINK YOUCOULD FOOLME?"
                ],
            )

        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertFalse(text["skip_processing"])
        self.assertEqual(text["content_class"], "text")
        self.assertNotEqual(text.get("skip_reason"), "cover_repeated_words_noise")

    def test_cover_opening_keeps_short_punctuated_white_balloon_speech(self):
        block = SimpleNamespace(
            xyxy=(194, 17952, 285, 17978),
            mask=None,
            confidence=0.78,
            line_polygons=[],
        )

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            page = build_page_result(
                image_path="001.jpg",
                image_rgb=np.full((22000, 800, 3), 255, dtype=np.uint8),
                blocks=[block],
                texts=["WHAT?"],
            )

        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertFalse(text["skip_processing"])
        self.assertNotEqual(text["content_class"], "noise")
        self.assertNotEqual(text.get("skip_reason"), "cover_short_ornamental_noise")

    def test_cover_opening_keeps_long_plain_sentence_without_terminal_punctuation(self):
        block = SimpleNamespace(
            xyxy=(376, 33732, 669, 33825),
            mask=None,
            confidence=0.93,
            line_polygons=[
                [[376, 33732], [669, 33732], [669, 33825], [376, 33825]],
            ],
        )

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            page = build_page_result(
                image_path="001.jpg",
                image_rgb=np.full((36000, 800, 3), 255, dtype=np.uint8),
                blocks=[block],
                texts=["ILIVE LIKETHIS BUTI DON'T HAVE THE COURAGE TOCHANGE MY LIFE"],
            )

        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertFalse(text["skip_processing"])
        self.assertNotEqual(text["content_class"], "noise")
        self.assertNotEqual(text.get("skip_reason"), "cover_logo_or_art_ocr")

    def test_build_page_result_marks_low_confidence_art_run_on_ocr_for_review(self):
        block = SimpleNamespace(xyxy=(80, 260, 540, 430), mask=None, confidence=0.56, line_polygons=[])

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False):
            page = build_page_result(
                image_path="002.jpg",
                image_rgb=np.full((1000, 700, 3), 58, dtype=np.uint8),
                blocks=[block],
                texts=["VHEN IGETBACK TOWORK..."],
            )

        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertFalse(text["skip_processing"])
        self.assertFalse(text["needs_review"])
        self.assertNotEqual(text.get("skip_reason"), "suspicious_art_ocr_low_confidence")
        self.assertEqual(text["translate_policy"], "translate")

    def test_large_joined_word_art_ocr_skips_even_when_detector_marks_white_balloon(self):
        block = SimpleNamespace(xyxy=(137, 11390, 726, 11859), mask=None, confidence=0.56, line_polygons=[])

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            page = build_page_result(
                image_path="002.jpg",
                image_rgb=np.full((13832, 800, 3), [212, 223, 240], dtype=np.uint8),
                blocks=[block],
                texts=["VHEN IGETBACK TOWORK..."],
            )

        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertFalse(text["skip_processing"])
        self.assertFalse(text["needs_review"])
        self.assertNotEqual(text.get("skip_reason"), "suspicious_art_ocr_low_confidence")
        self.assertEqual(text["translate_policy"], "translate")

    def test_short_low_signal_art_ocr_skips_even_when_detector_marks_white_balloon(self):
        block = SimpleNamespace(xyxy=(140, 45, 260, 229), mask=None, confidence=0.56, line_polygons=[])

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
            page = build_page_result(
                image_path="051.jpg",
                image_rgb=np.full((320, 720, 3), [214, 206, 198], dtype=np.uint8),
                blocks=[block],
                texts=["Lns"],
            )

        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertFalse(text["skip_processing"])
        self.assertFalse(text.get("preserve_original", False))
        self.assertFalse(text["needs_review"])
        self.assertNotEqual(text.get("skip_reason"), "suspicious_art_ocr_low_confidence")
        self.assertEqual(text["translate_policy"], "translate")

    def test_build_page_result_keeps_watermark_and_credit_noise_as_text(self):
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

        self.assertEqual([item["text"] for item in page["texts"]], ["ASURASCANS. COM QC MED NIGHTTOONS GET OUT OF HERE!"])
        self.assertTrue(all(item.get("route_action") == "translate_inpaint_render" for item in page["texts"]))

    def test_build_page_result_routes_hyphenated_all_caps_name_list_as_credit(self):
        block = SimpleNamespace(
            xyxy=(18, 44, 590, 96),
            mask=None,
            confidence=0.61,
            line_polygons=[],
        )

        with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=False), patch(
            "vision_stack.runtime.classify_text_type",
            return_value="narracao",
        ):
            page = build_page_result(
                image_path="page_006_band_133.jpg",
                image_rgb=np.full((160, 640, 3), 245, dtype=np.uint8),
                blocks=[block],
                texts=["-KANJI2E2 -NEONNIGHTMARE -DRAGON EMPRYEAN SHADOWLESS"],
            )

        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertEqual(text["content_class"], "text")
        self.assertEqual(text["route_action"], "translate_inpaint_render")
        self.assertFalse(text.get("needs_review", False))
        self.assertNotIn("ocr_truncated_or_joined", text.get("qa_flags", []))
        self.assertNotIn("scanlation_credit", text.get("qa_flags", []))

    def test_build_page_result_keeps_scanlation_staff_page_roles_as_text(self):
        blocks = [
            SimpleNamespace(xyxy=(10, 10, 86, 34), mask=None, confidence=0.92),
            SimpleNamespace(xyxy=(110, 10, 186, 34), mask=None, confidence=0.92),
            SimpleNamespace(xyxy=(210, 10, 330, 34), mask=None, confidence=0.92),
            SimpleNamespace(xyxy=(10, 60, 170, 92), mask=None, confidence=0.92),
            SimpleNamespace(xyxy=(210, 60, 440, 92), mask=None, confidence=0.92),
            SimpleNamespace(xyxy=(10, 120, 260, 152), mask=None, confidence=0.92),
            SimpleNamespace(xyxy=(210, 120, 440, 152), mask=None, confidence=0.92),
            SimpleNamespace(xyxy=(10, 190, 260, 230), mask=None, confidence=0.91),
        ]

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=False):
            page = build_page_result(
                image_path="credits.jpg",
                image_rgb=np.full((300, 480, 3), 245, dtype=np.uint8),
                blocks=blocks,
                texts=[
                    "STAFF",
                    "EDITOR",
                    "REDRAWER",
                    "TYPESETTER STAFF",
                    "QUALITYCHECKER SLAYER",
                    "HELPUS WITH Donations",
                    "/ROUGHSYUDIO E",
                    "GET OUT OF HERE!",
                ],
            )

        self.assertEqual(
            [item["text"] for item in page["texts"]],
            [
                "STAFF",
                "EDITOR",
                "REDRAWER",
                "TYPESETTER STAFF",
                "QUALITYCHECKER SLAYER",
                "HELPUS WITH Donations",
                "/ROUGHSYUDIO E",
                "GET OUT OF HERE!",
            ],
        )
        self.assertTrue(all(item.get("route_action") == "translate_inpaint_render" for item in page["texts"]))

    def test_build_page_result_keeps_staff_word_when_used_as_story_text(self):
        blocks = [
            SimpleNamespace(xyxy=(20, 20, 190, 58), mask=None, confidence=0.91),
        ]

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=False):
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=np.full((120, 240, 3), 245, dtype=np.uint8),
                blocks=blocks,
                texts=["STAFF OF POWER"],
            )

        self.assertEqual([item["text"] for item in page["texts"]], ["STAFF OF POWER"])

    def test_build_page_result_skips_short_textured_sfx_and_noise(self):
        blocks = [
            SimpleNamespace(xyxy=(10, 10, 80, 40), mask=None, confidence=0.81),
            SimpleNamespace(xyxy=(10, 50, 96, 84), mask=None, confidence=0.87),
            SimpleNamespace(xyxy=(10, 96, 128, 132), mask=None, confidence=0.91),
            SimpleNamespace(xyxy=(10, 144, 166, 220), mask=None, confidence=0.71),
            SimpleNamespace(xyxy=(20, 228, 94, 276), mask=None, confidence=0.61),
            SimpleNamespace(xyxy=(24, 286, 188, 328), mask=None, confidence=0.94),
        ]

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=False):
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=np.full((360, 220, 3), 48, dtype=np.uint8),
                blocks=blocks,
                texts=["XEV", "Mo", "HMPH", "t", "iia", "THE ORCS"],
            )

        self.assertIn("THE ORCS", " ".join(item["text"] for item in page["texts"]))
        self.assertTrue(all(not item.get("skip_processing") for item in page["texts"]))

    def test_build_page_result_keeps_clean_low_confidence_sparse_narration(self):
        blocks = [
            SimpleNamespace(xyxy=(171, 16, 545, 89), mask=None, confidence=0.422),
            SimpleNamespace(xyxy=(201, 89, 513, 194), mask=None, confidence=0.476),
        ]

        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=False), patch(
                "vision_stack.runtime.classify_text_type",
                return_value="narracao",
            ):
                page = build_page_result(
                    image_path="007.jpg",
                    image_rgb=np.full((320, 720, 3), 255, dtype=np.uint8),
                    blocks=blocks,
                    texts=["ONCE THAT FORCE", "IS RELEASED"],
                )

            decision_log.finalize_decision_trace()

            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual([item["text"] for item in page["texts"]], ["ONCE THAT FORCE IS RELEASED"])
        self.assertFalse(any(item.get("reason") == "suspicious_low_confidence" for item in payloads))

    def test_build_page_result_keeps_clean_low_confidence_single_word_dialogue(self):
        block = SimpleNamespace(xyxy=(110, 16, 204, 45), mask=None, confidence=0.378)

        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
                page = build_page_result(
                    image_path="001.jpg",
                    image_rgb=np.full((120, 240, 3), 255, dtype=np.uint8),
                    blocks=[block],
                    texts=["BECAUSE."],
                )

            decision_log.finalize_decision_trace()

            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual([item["text"] for item in page["texts"]], ["BECAUSE."])
        self.assertFalse(any(item.get("reason") == "suspicious_low_confidence" for item in payloads))

    def test_build_page_result_keeps_low_confidence_visual_noise_as_text_signal(self):
        block = SimpleNamespace(xyxy=(217, 147, 1322, 388), mask=None, confidence=0.38)

        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True), patch(
                "vision_stack.runtime.is_short_textured_sfx_or_noise",
                return_value=False,
            ):
                page = build_page_result(
                    image_path="001.jpg",
                    image_rgb=np.full((5000, 1600, 3), 255, dtype=np.uint8),
                    blocks=[block],
                    texts=["W I KO"],
                )

            decision_log.finalize_decision_trace()

            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual([item["text"] for item in page["texts"]], ["W I KO"])
        self.assertFalse(page["texts"][0].get("skip_processing"))
        self.assertNotEqual(page["texts"][0].get("route_action"), "review_required")
        self.assertNotIn("low_confidence_visual_noise", page["texts"][0].get("qa_flags", []))
        self.assertFalse(any(item.get("reason") == "low_confidence_visual_noise" for item in payloads))

    def test_build_page_result_reviews_partial_white_balloon_fragment_instead_of_rendering_prefix(self):
        blocks = [
            SimpleNamespace(xyxy=(161, 39, 309, 65), mask=None, confidence=0.54),
            SimpleNamespace(xyxy=(164, 91, 280, 119), mask=None, confidence=0.54),
        ]

        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            with patch("vision_stack.runtime._is_white_balloon_context_for_text", return_value=True):
                page = build_page_result(
                    image_path="004.jpg",
                    image_rgb=np.full((180, 420, 3), 255, dtype=np.uint8),
                    blocks=blocks,
                    texts=["Hello,is this", "Hospital's"],
                )

            decision_log.finalize_decision_trace()

            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertTrue(any("Hospital" in item["text"] for item in page["texts"]))
        affected = [
            item
            for item in page["texts"]
            if "Hello" in item["text"] or "Hospital" in item["text"]
        ]
        self.assertTrue(affected)
        for item in affected:
            self.assertEqual(item["route_action"], "translate_inpaint_render")
            self.assertNotIn("ocr_truncated_or_joined", item.get("qa_flags", []))
        self.assertFalse(any(item.get("reason") == "suspicious_low_confidence" for item in payloads))
        self.assertTrue(any(item.get("reason") == "ocr_partial_low_confidence_fragment" for item in payloads))

    def test_build_page_result_keeps_clean_low_confidence_short_phrase(self):
        block = SimpleNamespace(xyxy=(254, 16, 449, 56), mask=None, confidence=0.519)

        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=False):
                page = build_page_result(
                    image_path="009.jpg",
                    image_rgb=np.full((120, 520, 3), 245, dtype=np.uint8),
                    blocks=[block],
                    texts=["FOR NOW."],
                )

            decision_log.finalize_decision_trace()

            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual([item["text"] for item in page["texts"]], ["FOR NOW."])
        self.assertFalse(any(item.get("reason") == "suspicious_low_confidence" for item in payloads))

    def test_build_page_result_keeps_known_clean_very_low_confidence_dialogue_phrase(self):
        block = SimpleNamespace(xyxy=(192, 48, 426, 160), mask=None, confidence=0.279)

        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
                page = build_page_result(
                    image_path="010.jpg",
                    image_rgb=np.full((220, 640, 3), 255, dtype=np.uint8),
                    blocks=[block],
                    texts=["THAT ARROGANT"],
                )

            decision_log.finalize_decision_trace()

            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual([item["text"] for item in page["texts"]], ["THAT ARROGANT"])
        self.assertFalse(any(item.get("reason") == "suspicious_low_confidence" for item in payloads))

    def test_build_page_result_keeps_clean_caps_dialogue_near_very_low_cutoff(self):
        block = SimpleNamespace(xyxy=(378, 16, 840, 276), mask=None, confidence=0.346)

        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=False):
                page = build_page_result(
                    image_path="001.jpg",
                    image_rgb=np.full((292, 1200, 3), 255, dtype=np.uint8),
                    blocks=[block],
                    texts=["THE COMMANDER RIGHTS"],
                )

            decision_log.finalize_decision_trace()

            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual([item["text"] for item in page["texts"]], ["THE COMMANDER RIGHTS"])
        self.assertFalse(any(item.get("reason") == "suspicious_low_confidence" for item in payloads))

    def test_build_page_result_keeps_clean_connected_balloon_phrase_at_low_confidence(self):
        blocks = [
            SimpleNamespace(xyxy=(384, 16, 866, 362), mask=None, confidence=0.309),
            SimpleNamespace(xyxy=(542, 234, 878, 376), mask=None, confidence=0.286),
        ]

        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
                page = build_page_result(
                    image_path="032.jpg",
                    image_rgb=np.full((430, 980, 3), 255, dtype=np.uint8),
                    blocks=blocks,
                    texts=[
                        "I KNOW. I WORKED MY TRYING TO TRACK",
                        "Ass OFF Long AGO YOU DOWN.",
                    ],
                )

            decision_log.finalize_decision_trace()

            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual(
            [item["text"] for item in page["texts"]],
            ["I KNOW. I WORKED MY TRYING TO TRACK Ass OFF Long AGO YOU DOWN."],
        )
        self.assertFalse(any(item.get("reason") == "suspicious_low_confidence" for item in payloads))

    def test_build_page_result_keeps_long_clean_dialogue_below_short_phrase_cutoff(self):
        block = SimpleNamespace(xyxy=(304, 16, 872, 296), mask=None, confidence=0.256)

        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
                page = build_page_result(
                    image_path="088.jpg",
                    image_rgb=np.full((360, 980, 3), 255, dtype=np.uint8),
                    blocks=[block],
                    texts=["Tch.I Guess I Can't USE MANA. I WISH I HAD RETURNED TO THE Past earlier."],
                )

            decision_log.finalize_decision_trace()

            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual(
            [item["text"] for item in page["texts"]],
            ["Tch.I Guess I Can't USE MANA. I WISH I HAD RETURNED TO THE Past earlier."],
        )
        self.assertFalse(any(item.get("reason") == "suspicious_low_confidence" for item in payloads))

    def test_build_page_result_keeps_short_white_balloon_dialogue(self):
        block = SimpleNamespace(xyxy=(20, 20, 120, 72), mask=None, confidence=0.88)

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
            page = build_page_result(
                image_path="page.jpg",
                image_rgb=np.full((140, 160, 3), 255, dtype=np.uint8),
                blocks=[block],
                texts=["NO"],
            )

        self.assertEqual([item["text"] for item in page["texts"]], ["NO"])

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

            self.assertEqual([item["text"] for item in page["texts"]], ["KIRO"])
            self.assertEqual(page["page_profile"], "cover_opening")
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]
            self.assertEqual(payloads[0]["action"], "classify_page_profile")
            self.assertEqual(payloads[0]["reason"], "cover_opening")
            self.assertFalse(any(item["reason"] == "ornamental_cover_noise" for item in payloads))

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

    def test_build_page_result_keeps_cover_title_logo_heuristic_without_explicit_title(self):
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
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["text"], "THE REGRESSED MERCENARYS MACHINATIONS KIRO SHOUNEN")
        self.assertEqual(page["texts"][0]["block_profile"], "standard")
        self.assertEqual(page["page_profile"], "cover_opening")
        self.assertFalse(any(item["reason"] == "cover_title_logo" for item in payloads))

    def test_build_page_result_skips_cover_title_logo_only_when_title_was_provided(self):
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
                    texts=["THE REGRESSED MERCENARYS MACHINATIONS"],
                    work_title="The Regressed Mercenary's Machinations",
                    work_title_user_provided=True,
                )

            decision_log.finalize_decision_trace()
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual([item["text"] for item in page["texts"]], ["THE REGRESSED MERCENARYS MACHINATIONS"])
        self.assertEqual(page["texts"][0]["block_profile"], "cover_title_logo")
        self.assertFalse(page["texts"][0]["skip_processing"])
        self.assertFalse(page["texts"][0].get("preserve_original", False))
        self.assertEqual(page["texts"][0]["route_action"], "translate_inpaint_render")
        self.assertEqual(page["page_profile"], "cover_opening")
        self.assertFalse(any(item["action"] == "drop_block" and item["reason"] == "cover_title_logo" for item in payloads))

    def test_build_page_result_keeps_low_confidence_cover_opening_narration_without_top_profile(self):
        block = SimpleNamespace(
            xyxy=(121, 16, 574, 75),
            mask=None,
            confidence=0.431,
        )

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=False), patch(
            "vision_stack.runtime.classify_text_type",
            return_value="narracao",
        ):
            page = build_page_result(
                image_path="003.jpg",
                image_rgb=np.full((520, 720, 3), 70, dtype=np.uint8),
                blocks=[block],
                texts=["STOPS ALL MOVEMENT AND"],
            )

        self.assertEqual(page["page_profile"], "cover_opening")
        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["text"], "STOPS ALL MOVEMENT AND")
        self.assertEqual(page["texts"][0]["block_profile"], "standard")

    def test_build_page_result_keeps_textured_cover_title_logo_without_explicit_title(self):
        with TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")
            decision_log.configure_decision_trace(tmp)

            block = SimpleNamespace(
                xyxy=(590, 346, 982, 429),
                mask=None,
                confidence=0.907,
            )

            with patch("vision_stack.runtime._is_white_balloon_region", return_value=False), patch(
                "vision_stack.runtime.classify_text_type",
                return_value="narracao",
            ):
                page = build_page_result(
                    image_path="001.jpg",
                    image_rgb=np.full((2444, 1200, 3), 70, dtype=np.uint8),
                    blocks=[block],
                    texts=["THE REGRESSED MERCENARYS MACHINATIONS"],
                )

            decision_log.finalize_decision_trace()
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]

        self.assertEqual(page["page_profile"], "cover_opening")
        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["text"], "THE REGRESSED MERCENARYS MACHINATIONS")
        self.assertEqual(page["texts"][0]["block_profile"], "standard")
        self.assertFalse(any(item["reason"] == "cover_title_logo" for item in payloads))

    def test_build_page_result_keeps_white_background_cover_title_logo_without_explicit_title(self):
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
        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["text"], "Theregressed Mercenarys")
        self.assertNotEqual(page["texts"][0]["block_profile"], "cover_title_logo")

    def test_white_cleanup_texts_ignores_legacy_skip_processing_marker(self):
        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        text = {
            "text": "HELLO",
            "bbox": [40, 32, 128, 74],
            "text_pixel_bbox": [46, 38, 120, 68],
            "block_profile": "white_balloon",
            "balloon_type": "white",
            "skip_processing": True,
            "preserve_original": True,
            "content_class": "noise",
        }

        self.assertEqual(_white_cleanup_texts(image, [text]), [text])

    def test_build_page_result_does_not_assign_top_narration_block_profile(self):
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

            self.assertEqual(page["texts"][0]["block_profile"], "standard")
            trace_lines = (Path(tmp) / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in trace_lines if line.strip()]
            self.assertFalse(
                any(item["action"] == "classify_block_profile" and item["reason"] == "top_narration" for item in payloads)
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

        self.assertEqual(len(page["texts"]), 1)
        self.assertFalse(page["texts"][0]["skip_processing"])
        self.assertFalse(page["texts"][0].get("preserve_original", False))

    def test_build_page_result_keeps_canonical_font_when_font_detection_is_enabled(self):
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
        self.assertEqual(page["texts"][0]["estilo"].get("fonte"), "ComicNeue-Bold.ttf")
        self.assertEqual(page["texts"][0]["estilo"].get("cor"), "#000000")

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

    def test_build_page_result_textured_balloon_uses_canonical_font_without_detector(self):
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
                image_rgb=np.full((80, 120, 3), 30, dtype=np.uint8),
                blocks=[block],
                texts=["HELLO"],
                enable_font_detection=True,
            )

        get_font_detector.assert_not_called()
        self.assertEqual(page["texts"][0]["estilo"].get("fonte"), "ComicNeue-Bold.ttf")
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

    def test_vision_blocks_to_mask_uses_line_polygons_for_textured_blocks(self):
        image = np.full((260, 260, 3), 45, dtype=np.uint8)
        block = {
            "bbox": [30, 40, 230, 210],
            "mask": None,
            "balloon_type": "textured",
            "text_pixel_bbox": [40, 55, 220, 195],
            "line_polygons": [
                [[50, 60], [210, 60], [210, 82], [50, 82]],
                [[45, 105], [220, 105], [220, 127], [45, 127]],
                [[70, 150], [190, 150], [190, 172], [70, 172]],
            ],
        }

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=False), patch(
            "vision_stack.runtime._build_refined_bbox_mask",
            side_effect=AssertionError("line geometry should prevent full bbox refinement"),
        ):
            mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertGreater(int(mask[70, 100]), 0)
        self.assertGreater(int(mask[116, 100]), 0)
        self.assertEqual(int(mask[94, 100]), 0)
        self.assertEqual(int(mask[45, 35]), 0)
        self.assertLess(int(np.count_nonzero(mask)), 22000)

    def test_vision_blocks_to_mask_prefers_explicit_geometry_over_overbroad_local_mask(self):
        image = np.full((160, 160, 3), 245, dtype=np.uint8)
        broad_local_mask = np.ones((140, 140), dtype=np.uint8) * 255
        block = {
            "bbox": [10, 10, 150, 150],
            "mask": broad_local_mask,
            "balloon_bbox": [10, 10, 150, 150],
            "balloon_type": "white",
            "tipo": "fala",
            "text_pixel_bbox": [58, 58, 102, 104],
            "line_polygons": [
                [[60, 60], [100, 60], [100, 78], [60, 78]],
                [[65, 84], [95, 84], [95, 102], [65, 102]],
            ],
        }

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertGreater(int(mask[66, 70]), 0)
        self.assertGreater(int(mask[92, 80]), 0)
        self.assertEqual(int(mask[20, 20]), 0)
        self.assertEqual(int(mask[132, 132]), 0)
        self.assertLess(int(np.count_nonzero(mask)), 4500)

    def test_vision_blocks_to_mask_drops_white_balloon_outline_sliver_from_local_mask(self):
        image = np.full((90, 160, 3), 255, dtype=np.uint8)
        local_mask = np.zeros((90, 160), dtype=np.uint8)
        local_mask[44:56, 50:112] = 255
        local_mask[20:36, 136:144] = 255
        bubble_mask = np.zeros((90, 160), dtype=np.uint8)
        cv2.ellipse(bubble_mask, (80, 44), (64, 34), 0, 0, 360, 255, -1)
        block = {
            "bbox": [44, 20, 140, 60],
            "text_pixel_bbox": [44, 20, 140, 60],
            "source_bbox": [44, 20, 140, 60],
            "mask": local_mask,
            "bubble_mask": bubble_mask,
            "bubble_mask_source": "image_contour_bubble_mask",
            "balloon_bbox": [16, 10, 148, 78],
            "layout_profile": "white_balloon",
        }

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertEqual(int(np.count_nonzero(mask[20:36, 136:144])), 0)
        self.assertGreater(int(np.count_nonzero(mask[44:56, 50:112])), 0)

    def test_vision_blocks_to_mask_recovers_line_polygon_component_missing_from_local_mask(self):
        image = np.full((128, 192, 3), 245, dtype=np.uint8)
        local_mask = np.zeros((128, 192), dtype=np.uint8)
        local_mask[23:67, 38:134] = 255
        block = {
            "bbox": [24, 16, 150, 104],
            "mask": local_mask,
            "balloon_bbox": [10, 8, 170, 112],
            "balloon_type": "white",
            "tipo": "fala",
            "text_pixel_bbox": [42, 24, 132, 98],
            "line_polygons": [
                [[40, 24], [132, 24], [132, 40], [40, 40]],
                [[46, 50], [128, 50], [128, 66], [46, 66]],
                [[70, 82], [112, 82], [112, 98], [70, 98]],
            ],
        }

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertGreater(int(mask[90, 88]), 0)
        self.assertEqual(int(mask[8, 8]), 0)
        self.assertIn("local_mask_missing_geometry_components", block.get("qa_metrics", {}))
        recovered = block["qa_metrics"]["local_mask_missing_geometry_components"]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["bbox"], [68, 80, 115, 101])

    def test_vision_blocks_to_mask_floors_recovered_dark_bubble_to_text_bbox(self):
        image = np.zeros((120, 260, 3), dtype=np.uint8)
        local_mask = np.zeros((120, 260), dtype=np.uint8)
        local_mask[48:74, 104:166] = 255
        bubble_mask = np.zeros((120, 260), dtype=np.uint8)
        cv2.ellipse(bubble_mask, (145, 62), (110, 46), 0, 0, 360, 255, -1)
        block = {
            "bbox": [42, 38, 220, 86],
            "text_pixel_bbox": [42, 38, 220, 86],
            "mask": local_mask,
            "bubble_mask": bubble_mask,
            "bubble_mask_bbox": [20, 12, 250, 110],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["candidate_crop_direct_paddle_reocr", "partial_dark_bubble_lobe_reocr"],
        }

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertGreater(int(np.count_nonzero(mask[42:82, 42:96])), 0)
        self.assertGreater(int(np.count_nonzero(mask[42:82, 172:220])), 0)
        self.assertIn("dark_bubble_recovered_text_bbox_floor", block.get("qa_flags", []))

    def test_vision_blocks_to_mask_does_not_or_line_polygons_over_component_mask(self):
        image = np.full((96, 140, 3), 245, dtype=np.uint8)
        refined_mask = np.zeros((96, 140), dtype=np.uint8)
        refined_mask[42:50, 64:78] = 255
        block = {
            "bbox": [20, 20, 120, 76],
            "mask": None,
            "balloon_bbox": [20, 20, 120, 76],
            "balloon_type": "white",
            "text_pixel_bbox": [40, 34, 96, 58],
            "line_polygons": [[[40, 34], [96, 34], [96, 58], [40, 58]]],
        }

        def fake_build_inpaint_mask(target_block, _image_shape, image_rgb=None):
            target_block["mask_evidence"] = {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": int(np.count_nonzero(refined_mask)),
                "expanded_mask_pixels": int(np.count_nonzero(refined_mask)),
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
                "evidence_score": 1.0,
            }
            return refined_mask.copy()

        with patch("inpainter.mask_builder.build_inpaint_mask", side_effect=fake_build_inpaint_mask):
            mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertGreater(int(mask[45, 70]), 0)
        self.assertEqual(int(mask[36, 44]), 0)
        self.assertEqual(int(mask[56, 94]), 0)
        self.assertLess(int(np.count_nonzero(mask)), 140)

    def test_vision_blocks_to_mask_does_not_or_line_polygons_over_dark_visual_glyph_mask(self):
        image = np.zeros((120, 220, 3), dtype=np.uint8)
        refined_mask = np.zeros((120, 220), dtype=np.uint8)
        refined_mask[48:56, 78:118] = 255
        refined_mask[70:78, 84:126] = 255
        block = {
            "bbox": [60, 40, 150, 88],
            "mask": None,
            "text_pixel_bbox": [60, 40, 150, 88],
            "line_polygons": [
                [[56, 38], [154, 38], [154, 62], [56, 62]],
                [[56, 66], [154, 66], [154, 90], [56, 90]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
        }

        def fake_build_inpaint_mask(target_block, _image_shape, image_rgb=None):
            target_block.setdefault("qa_flags", []).append("dark_bubble_visual_glyph_mask_replaced_geometry")
            target_block.setdefault("qa_metrics", {})["dark_bubble_visual_glyph_mask_replaced_geometry"] = {
                "visual_pixels": int(np.count_nonzero(refined_mask)),
                "geometry_pixels": 4704,
            }
            target_block["mask_evidence"] = {
                "kind": "ocr_pixels",
                "raw_mask_pixels": int(np.count_nonzero(refined_mask)),
                "expanded_mask_pixels": int(np.count_nonzero(refined_mask)),
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
                "evidence_score": 1.0,
            }
            return refined_mask.copy()

        with patch("inpainter.mask_builder.build_inpaint_mask", side_effect=fake_build_inpaint_mask):
            mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertGreater(int(mask[52, 90]), 0)
        self.assertEqual(int(mask[42, 60]), 0)
        self.assertEqual(int(mask[86, 150]), 0)
        self.assertLess(int(np.count_nonzero(mask)), 760)

    def test_vision_blocks_to_mask_uses_text_pixel_bbox_without_overbroad_refined_fallback(self):
        image = np.full((120, 180, 3), 245, dtype=np.uint8)
        block = {
            "bbox": [64, 42, 116, 62],
            "text_pixel_bbox": [64, 42, 116, 62],
            "balloon_bbox": [28, 16, 150, 92],
            "balloon_type": "white",
            "tipo": "fala",
            "mask": None,
        }

        with patch(
            "vision_stack.runtime._build_refined_bbox_mask",
            side_effect=AssertionError("text geometry should prevent broad balloon refinement"),
        ):
            mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=False)

        self.assertGreater(int(mask[50, 80]), 0)
        self.assertEqual(int(mask[20, 36]), 0)
        self.assertEqual(int(mask[88, 144]), 0)
        self.assertLess(int(np.count_nonzero(mask)), 3000)

    def test_vision_blocks_to_mask_protects_balloon_outline_near_text_geometry(self):
        image = np.full((80, 140, 3), 250, dtype=np.uint8)
        image[34:36, 18:122] = 8
        block = {
            "bbox": [42, 18, 98, 29],
            "text_pixel_bbox": [42, 18, 98, 29],
            "line_polygons": [[[42, 18], [98, 18], [98, 29], [42, 29]]],
            "balloon_type": "white",
            "tipo": "fala",
            "mask": None,
        }

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=True)

        self.assertGreater(int(mask[23, 70]), 0)
        self.assertEqual(int(mask[34, 70]), 0)
        self.assertEqual(int(mask[35, 70]), 0)

    def test_build_refined_bbox_mask_expands_light_text_on_dark_background_beyond_seed(self):
        image = np.full((180, 260, 3), 18, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (220, 140), (20, 20, 20), -1)
        cv2.putText(image, "TEST", (78, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 80, 80), 10, cv2.LINE_AA)
        cv2.putText(image, "TEST", (78, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (245, 245, 245), 3, cv2.LINE_AA)

        bbox = [85, 70, 180, 112]
        refined = _build_refined_bbox_mask(image, bbox)

        self.assertIsNotNone(refined)
        rx1, ry1, patch = refined
        ys, xs = np.where(patch > 0)
        global_x1 = rx1 + int(xs.min())
        global_x2 = rx1 + int(xs.max()) + 1

        self.assertLess(global_x1, bbox[0])
        self.assertGreater(global_x2, bbox[2])

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

    def test_vision_blocks_to_mask_falls_back_to_octagon_bbox_without_image(self):
        blocks = [
            {"bbox": [30, 20, 110, 50], "mask": None},
        ]

        mask = vision_blocks_to_mask((90, 140, 3), blocks, image_rgb=None, expand_mask=False)

        self.assertGreater(int(mask[30, 60]), 0)
        self.assertGreater(int(mask[24, 34]), 0)
        self.assertGreater(int(mask[46, 106]), 0)
        self.assertEqual(int(mask[20, 30]), 0)
        self.assertEqual(int(mask[49, 109]), 0)

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

    def test_vision_blocks_to_mask_white_balloon_rejects_top_half_exact_boxes_and_uses_refined_mask(self):
        image = np.full((260, 220, 3), 245, dtype=np.uint8)
        refined_patch = np.zeros((120, 120), dtype=np.uint8)
        refined_patch[16:112, 12:108] = 255

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=True), patch(
            "vision_stack.runtime._extract_white_balloon_text_boxes",
            return_value=[[62, 70, 158, 94]],
        ), patch(
            "vision_stack.runtime._build_refined_bbox_mask",
            return_value=(50, 60, refined_patch),
        ):
            mask = vision_blocks_to_mask(
                image.shape,
                [{"bbox": [50, 60, 170, 180], "mask": None}],
                image_rgb=image,
                expand_mask=False,
            )

        self.assertGreater(int(mask[160, 100]), 0)
        self.assertEqual(int(mask[20, 20]), 0)

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

    def test_run_ocr_stage_skips_orphan_lobe_scan_by_default_for_strip_bands(self):
        image = np.full((120, 180, 3), 255, dtype=np.uint8)
        page_dict = {
            "numero": 1,
            "_vision_blocks": [{"bbox": [40, 30, 120, 80], "confidence": 0.9}],
        }
        fake_ocr = SimpleNamespace(
            _backend="paddleocr",
            recognize_blocks_from_page=lambda _image, _blocks: [],
        )

        with patch("vision_stack.runtime._get_ocr_engine", return_value=fake_ocr), patch(
            "vision_stack.runtime._scan_orphan_lobe_blocks",
            side_effect=AssertionError("strip OCR should not scan orphan lobes by default"),
        ), patch("vision_stack.runtime.build_page_result", return_value={"texts": [], "_vision_blocks": []}):
            result = run_ocr_stage(image, page_dict)

        self.assertEqual(result["texts"], [])

    def test_run_ocr_stage_uses_zero_padded_source_page_label_for_strip_bands(self):
        image = np.full((120, 180, 3), 255, dtype=np.uint8)
        page_dict = {
            "numero": 2,
            "_source_page_number": 2,
            "_vision_blocks": [{"bbox": [40, 30, 120, 80], "confidence": 0.9}],
        }
        fake_ocr = SimpleNamespace(
            _backend="paddleocr",
            recognize_blocks_from_page=lambda _image, _blocks, **_kw: [],
        )
        captured = {}

        def fake_build_page_result(*, image_path, **_kwargs):
            captured["image_path"] = image_path
            return {"texts": [], "_vision_blocks": []}

        with patch.dict(os.environ, {"TRADUZAI_STRIP_QUICK_TEXT_SKIP": "0"}, clear=False), patch(
            "vision_stack.runtime._get_ocr_engine",
            return_value=fake_ocr,
        ), patch("vision_stack.runtime.build_page_result", side_effect=fake_build_page_result):
            result = run_ocr_stage(image, page_dict)

        self.assertEqual(result["texts"], [])
        self.assertEqual(captured["image_path"], "band_002")

    def test_negative_pass_does_not_mutate_page_texts_until_promoted(self):
        image = np.full((80, 120, 3), 10, dtype=np.uint8)
        image[20:40, 30:90] = 245
        page_dict = {
            "numero": 1,
            "_vision_blocks": [{"bbox": [20, 16, 96, 48], "confidence": 0.9}],
        }
        normal_text = {"id": "normal", "text": "NORMAL", "bbox": [20, 16, 96, 48]}
        negative_text = {"id": "neg", "text": "NEGATIVE", "bbox": [30, 20, 90, 40]}
        captured = {}

        class FakeDetector:
            def detect(self, img, conf_threshold=0.5):
                captured["negative_image_sample"] = int(img[0, 0, 0])
                return [SimpleNamespace(x1=30, y1=20, x2=90, y2=40, confidence=0.88)]

            def crop(self, img, block):
                return img[int(block.y1):int(block.y2), int(block.x1):int(block.x2)]

        fake_ocr = SimpleNamespace(
            _backend="mockocr",
            recognize_blocks_from_page=lambda img, blocks, **_kw: [negative_text],
            recognize_batch=lambda crops: [negative_text],
        )

        def fake_build_page_result(*, image_path, **_kwargs):
            return {"image": image_path, "texts": [dict(normal_text)], "_vision_blocks": [dict(page_dict["_vision_blocks"][0])]}

        with patch.dict(os.environ, {"TRADUZAI_STRIP_QUICK_TEXT_SKIP": "0", "TRADUZAI_NEGATIVE_EVIDENCE_PASS": "1"}), patch(
            "vision_stack.runtime._get_ocr_engine",
            return_value=fake_ocr,
        ), patch("vision_stack.runtime._get_detector", return_value=FakeDetector()), patch(
            "vision_stack.runtime.build_page_result",
            side_effect=fake_build_page_result,
        ):
            result = run_ocr_stage(image, page_dict)

        self.assertEqual([text["id"] for text in result["texts"]], ["normal"])
        self.assertEqual(captured["negative_image_sample"], 245)
        evidence = result.get("_negative_evidence")
        self.assertIsInstance(evidence, dict)
        self.assertFalse(evidence.get("eligible_for_promotion"))
        self.assertEqual(evidence.get("source"), "negative_detect_ocr")
        self.assertEqual(evidence.get("texts", [])[0]["id"], "neg")

    def test_negative_pass_does_not_mutate_vision_blocks_until_promoted(self):
        image = np.full((80, 120, 3), 8, dtype=np.uint8)
        page_dict = {
            "numero": 1,
            "_vision_blocks": [{"bbox": [10, 10, 50, 36], "confidence": 0.91, "detector": "strip-detector"}],
        }
        negative_block = SimpleNamespace(x1=60, y1=20, x2=104, y2=48, confidence=0.86)

        class FakeDetector:
            def detect(self, img, conf_threshold=0.5):
                return [negative_block]

            def crop(self, img, block):
                return img[int(block.y1):int(block.y2), int(block.x1):int(block.x2)]

        fake_ocr = SimpleNamespace(
            _backend="mockocr",
            recognize_blocks_from_page=lambda img, blocks, **_kw: [{"text": "SHADOW", "bbox": [60, 20, 104, 48]}],
            recognize_batch=lambda crops: [{"text": "SHADOW", "bbox": [60, 20, 104, 48]}],
        )

        with patch.dict(os.environ, {"TRADUZAI_STRIP_QUICK_TEXT_SKIP": "0", "TRADUZAI_NEGATIVE_EVIDENCE_PASS": "1"}), patch(
            "vision_stack.runtime._get_ocr_engine",
            return_value=fake_ocr,
        ), patch("vision_stack.runtime._get_detector", return_value=FakeDetector()), patch(
            "vision_stack.runtime.build_page_result",
            return_value={"texts": [], "_vision_blocks": [dict(page_dict["_vision_blocks"][0])]},
        ):
            result = run_ocr_stage(image, page_dict)

        self.assertEqual(result["_vision_blocks"][0]["bbox"], page_dict["_vision_blocks"][0]["bbox"])
        self.assertEqual(result["_vision_blocks"][0]["detector"], page_dict["_vision_blocks"][0]["detector"])
        self.assertEqual(len(result["_vision_blocks"]), 1)
        self.assertEqual(result["_negative_evidence"]["blocks"][0]["bbox"], [60, 20, 104, 48])

    def test_run_ocr_stage_can_opt_into_orphan_lobe_scan_for_strip_bands(self):
        image = np.full((120, 180, 3), 255, dtype=np.uint8)
        page_dict = {
            "numero": 1,
            "_enable_orphan_lobe_scan": True,
            "_vision_blocks": [{"bbox": [40, 30, 120, 80], "confidence": 0.9}],
        }
        fake_ocr = SimpleNamespace(
            _backend="paddleocr",
            recognize_blocks_from_page=lambda _image, _blocks: [],
        )

        with patch("vision_stack.runtime._get_ocr_engine", return_value=fake_ocr), patch(
            "vision_stack.runtime._scan_orphan_lobe_blocks",
            side_effect=lambda _image, blocks, _ocr: blocks,
        ) as scan, patch("vision_stack.runtime.build_page_result", return_value={"texts": [], "_vision_blocks": []}):
            result = run_ocr_stage(image, page_dict)

        scan.assert_called_once()
        self.assertEqual(result["texts"], [])

    def test_run_ocr_stage_quick_skips_large_blank_strip_band(self):
        image = np.full((520, 800, 3), 248, dtype=np.uint8)
        page_dict = {
            "numero": 1,
            "_vision_blocks": [{"bbox": [80, 120, 260, 240], "confidence": 0.9}],
        }

        fake_ocr = SimpleNamespace(
            _backend="paddleocr",
            recognize_blocks_from_page=lambda _image, _blocks, **_kwargs: [],
        )

        with patch("vision_stack.runtime._get_ocr_engine", return_value=fake_ocr) as get_ocr:
            result = run_ocr_stage(image, page_dict)

        get_ocr.assert_called_once()
        self.assertEqual(result["texts"], [])
        self.assertFalse(result.get("quick_skipped_no_text", False))
        self.assertFalse(result.get("sem_texto_detectado", False))

    def test_run_ocr_stage_quick_skip_can_be_disabled(self):
        image = np.full((520, 800, 3), 248, dtype=np.uint8)
        page_dict = {
            "numero": 1,
            "_vision_blocks": [{"bbox": [80, 120, 260, 240], "confidence": 0.9}],
        }
        fake_ocr = SimpleNamespace(
            _backend="paddleocr",
            recognize_blocks_from_page=lambda _image, _blocks: [],
        )

        with patch.dict(os.environ, {"TRADUZAI_STRIP_QUICK_TEXT_SKIP": "0"}), patch(
            "vision_stack.runtime._get_ocr_engine",
            return_value=fake_ocr,
        ) as get_ocr, patch("vision_stack.runtime.build_page_result", return_value={"texts": [], "_vision_blocks": []}):
            result = run_ocr_stage(image, page_dict)

        get_ocr.assert_called_once()
        self.assertEqual(result["texts"], [])

    def test_run_ocr_stage_skips_scanlation_credit_band_before_ocr(self):
        image = np.full((520, 800, 3), 24, dtype=np.uint8)
        for y in (60, 150, 240, 330):
            cv2.line(image, (120, y), (680, y), (245, 245, 245), 3)
            cv2.line(image, (160, y + 42), (640, y + 42), (245, 245, 245), 3)
            cv2.putText(
                image,
                "STAFF",
                (300, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

        blocks = []
        for y in (52, 84, 142, 174, 232, 264, 322, 354):
            blocks.append({"bbox": [180, y, 620, y + 30], "confidence": 0.9})
        blocks.extend(
            [
                {"bbox": [86, 52, 130, 96], "confidence": 0.9},
                {"bbox": [670, 142, 720, 190], "confidence": 0.9},
                {"bbox": [98, 322, 150, 372], "confidence": 0.9},
                {"bbox": [250, 440, 560, 482], "confidence": 0.9},
            ]
        )
        page_dict = {"numero": 1, "_vision_blocks": blocks}

        fake_ocr = SimpleNamespace(
            _backend="paddleocr",
            recognize_blocks_from_page=lambda _image, _blocks, **_kwargs: [],
        )

        with patch("vision_stack.runtime._get_ocr_engine", return_value=fake_ocr) as get_ocr:
            result = run_ocr_stage(image, page_dict)

        get_ocr.assert_called_once()
        self.assertEqual(result["texts"], [])
        self.assertFalse(result.get("scanlation_credit_skipped", False))
        self.assertFalse(result.get("sem_texto_detectado", False))
        self.assertFalse(result["_ocr_stats"].get("scanlation_credit_skipped", False))

    def test_run_ocr_stage_keeps_dense_story_band_without_credit_lines(self):
        image = np.full((520, 800, 3), 36, dtype=np.uint8)
        blocks = []
        for index in range(12):
            x = 80 + (index % 3) * 220
            y = 60 + (index // 3) * 90
            cv2.putText(
                image,
                "WAIT",
                (x, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            blocks.append({"bbox": [x, y, x + 120, y + 42], "confidence": 0.9})

        class FakeOcr:
            _backend = "paddleocr"

            def recognize_blocks_from_page(self, _image, _blocks, **_kwargs):
                return []

        with patch.dict(os.environ, {"TRADUZAI_STRIP_QUICK_TEXT_SKIP": "0"}, clear=False), patch(
            "vision_stack.runtime._get_ocr_engine",
            return_value=FakeOcr(),
        ) as get_ocr, patch(
            "vision_stack.runtime.build_page_result",
            return_value={"texts": [], "_vision_blocks": []},
        ):
            result = run_ocr_stage(image, {"numero": 1, "_vision_blocks": blocks})

        get_ocr.assert_called_once()
        self.assertEqual(result["texts"], [])

    def test_run_ocr_stage_enables_small_crop_fallback_by_default_for_strip(self):
        image = np.full((520, 800, 3), 248, dtype=np.uint8)
        page_dict = {
            "numero": 1,
            "_vision_blocks": [{"bbox": [80, 120, 260, 240], "confidence": 0.9}],
        }
        seen_kwargs = {}

        class FakeOcr:
            _backend = "paddleocr"

            def recognize_blocks_from_page(self, _image, _blocks, **kwargs):
                seen_kwargs.update(kwargs)
                return []

        with patch.dict(os.environ, {"TRADUZAI_STRIP_QUICK_TEXT_SKIP": "0"}, clear=False):
            os.environ.pop("TRADUZAI_STRIP_PADDLE_CROP_FALLBACK_MAX", None)
            os.environ.pop("TRADUZAI_PADDLE_CROP_FALLBACK_MAX", None)
            os.environ.pop("TRADUZAI_STRIP_PADDLE_SPARSE_CROP_FALLBACK_MAX", None)
            os.environ.pop("TRADUZAI_PADDLE_SPARSE_CROP_FALLBACK_MAX", None)
            with patch("vision_stack.runtime._get_ocr_engine", return_value=FakeOcr()), patch(
                "vision_stack.runtime.build_page_result",
                return_value={"texts": [], "_vision_blocks": []},
            ):
                result = run_ocr_stage(image, page_dict)

        self.assertEqual(result["texts"], [])
        self.assertEqual(seen_kwargs["crop_fallback_max"], 3)
        self.assertEqual(seen_kwargs["sparse_crop_fallback_max"], 3)

    def test_run_ocr_stage_can_disable_sparse_mapping_for_strip_bands(self):
        image = np.full((520, 800, 3), 248, dtype=np.uint8)
        page_dict = {
            "numero": 1,
            "_disable_sparse_ocr_mapping": True,
            "_vision_blocks": [{"bbox": [80, 120, 260, 240], "confidence": 0.9}],
        }
        seen_kwargs = {}

        class FakeOcr:
            _backend = "paddleocr"

            def recognize_blocks_from_page(self, _image, _blocks, **kwargs):
                seen_kwargs.update(kwargs)
                return []

        with patch.dict(os.environ, {"TRADUZAI_STRIP_QUICK_TEXT_SKIP": "0"}, clear=False):
            with patch("vision_stack.runtime._get_ocr_engine", return_value=FakeOcr()), patch(
                "vision_stack.runtime.build_page_result",
                return_value={"texts": [], "_vision_blocks": []},
            ):
                run_ocr_stage(image, page_dict)

        self.assertIs(seen_kwargs["allow_sparse_mapping"], False)

    def test_run_ocr_stage_allows_strip_crop_fallback_override(self):
        image = np.full((520, 800, 3), 248, dtype=np.uint8)
        page_dict = {
            "numero": 1,
            "_vision_blocks": [{"bbox": [80, 120, 260, 240], "confidence": 0.9}],
        }
        seen_kwargs = {}

        class FakeOcr:
            _backend = "paddleocr"

            def recognize_blocks_from_page(self, _image, _blocks, **kwargs):
                seen_kwargs.update(kwargs)
                return []

        with patch.dict(
            os.environ,
            {
                "TRADUZAI_STRIP_QUICK_TEXT_SKIP": "0",
                "TRADUZAI_STRIP_PADDLE_CROP_FALLBACK_MAX": "2",
                "TRADUZAI_STRIP_PADDLE_SPARSE_CROP_FALLBACK_MAX": "1",
            },
            clear=False,
        ):
            with patch("vision_stack.runtime._get_ocr_engine", return_value=FakeOcr()), patch(
                "vision_stack.runtime.build_page_result",
                return_value={"texts": [], "_vision_blocks": []},
            ):
                result = run_ocr_stage(image, page_dict)

        self.assertEqual(result["texts"], [])
        self.assertEqual(seen_kwargs["crop_fallback_max"], 2)
        self.assertEqual(seen_kwargs["sparse_crop_fallback_max"], 1)

    def test_run_ocr_stage_rescues_raw_system_ui_when_page_result_drops_all_texts(self):
        image = np.full((261, 800, 3), 12, dtype=np.uint8)
        page_dict = {
            "numero": 2,
            "_vision_blocks": [{"bbox": [0, 0, 202, 209], "confidence": 0.854}],
        }

        class FakeOcr:
            _backend = "paddleocr"
            _last_recognize_blocks_stats = {"full_page_mapped": 1}

            def recognize_blocks_from_page(self, _image, _blocks, **_kwargs):
                return [
                    {
                        "text": "Main Quest will be shown shortly",
                        "confidence": 0.854,
                        "bbox": [0, 0, 202, 209],
                        "text_pixel_bbox": [34, 72, 178, 146],
                        "line_polygons": [
                            [[34, 72], [178, 72], [178, 98], [34, 98]],
                            [[52, 112], [160, 112], [160, 146], [52, 146]],
                        ],
                    }
                ]

        def fake_build_page_result(*, image_path, blocks, texts, **kwargs):
            if not getattr(fake_build_page_result, "called", False):
                fake_build_page_result.called = True
                return {"image": image_path, "width": 800, "height": 261, "texts": [], "_vision_blocks": []}
            return {
                "image": image_path,
                "width": 800,
                "height": 261,
                "texts": [
                    {
                        "id": "ocr_001",
                        "text": texts[0]["text"],
                        "bbox": list(texts[0]["bbox"]),
                        "source_bbox": list(texts[0]["bbox"]),
                        "text_pixel_bbox": list(texts[0]["text_pixel_bbox"]),
                        "confidence": texts[0]["confidence"],
                        "qa_flags": ["accepted_ocr_raw_system_ui_rescue"],
                    }
                ],
                "_vision_blocks": [
                    {
                        "text_id": "ocr_001",
                        "text": texts[0]["text"],
                        "bbox": list(texts[0]["bbox"]),
                        "text_pixel_bbox": list(texts[0]["text_pixel_bbox"]),
                        "confidence": texts[0]["confidence"],
                    }
                ],
            }

        with patch.dict(os.environ, {"TRADUZAI_STRIP_QUICK_TEXT_SKIP": "0"}, clear=False), patch(
            "vision_stack.runtime._get_ocr_engine",
            return_value=FakeOcr(),
        ), patch("vision_stack.runtime.build_page_result", side_effect=fake_build_page_result):
            result = run_ocr_stage(image, page_dict)

        self.assertEqual(len(result["texts"]), 1)
        self.assertEqual(result["texts"][0]["text"], "Main Quest will be shown shortly")
        self.assertIn("accepted_ocr_raw_system_ui_rescue", result["texts"][0]["qa_flags"])
        self.assertEqual(result["_ocr_stats"]["raw_system_ui_rescue_count"], 1)

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

    def test_run_detect_ocr_adds_uied_layout_blocks_before_recognition(self):
        image = np.full((150, 300, 3), 255, dtype=np.uint8)
        cv2.putText(image, "20** regional fireman", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (18, 18, 18), 1)
        image[58:84, 22:278] = [184, 196, 224]
        cv2.putText(image, "Successful candidate inquiry", (62, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (18, 18, 18), 1)
        captured: dict[str, list] = {}

        def fake_build_page_result(*args, **kwargs):
            captured.setdefault("blocks", list(kwargs.get("blocks") or []))
            captured.setdefault("texts", list(kwargs.get("texts") or []))
            return {
                "image": "page.jpg",
                "width": 300,
                "height": 150,
                "texts": [{"bbox": [22, 58, 278, 84], "text": "Successful candidate inquiry"}],
                "_vision_blocks": [{"bbox": [22, 58, 278, 84], "text": "Successful candidate inquiry"}],
            }

        with patch.dict(os.environ, {"TRADUZAI_UIED_LAYOUT": "1"}), patch(
            "vision_stack.runtime.cv2.imread", return_value=image
        ), patch("vision_stack.runtime._get_detector") as get_detector, patch(
            "vision_stack.runtime._get_ocr_engine"
        ) as get_ocr, patch(
            "vision_stack.runtime.build_page_result",
            side_effect=fake_build_page_result,
        ):
            get_detector.return_value.detect.return_value = []
            get_ocr.return_value._backend = "paddleocr"
            get_ocr.return_value.recognize_blocks_from_page.return_value = [
                {"text": "Successful candidate inquiry"},
                {"text": "20** regional fireman"},
            ]

            run_detect_ocr("page.jpg", profile="quality")

        self.assertTrue(any(getattr(block, "detector", "") == "uied_cv" for block in captured["blocks"]))
        get_ocr.return_value.recognize_blocks_from_page.assert_called_once()

    def test_run_ocr_stage_adds_uied_layout_blocks_before_recognition(self):
        image = np.full((150, 300, 3), 255, dtype=np.uint8)
        cv2.putText(image, "20** regional fireman", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (18, 18, 18), 1)
        image[58:84, 22:278] = [184, 196, 224]
        cv2.putText(image, "Successful candidate inquiry", (62, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (18, 18, 18), 1)
        captured: dict[str, list] = {}

        def fake_build_page_result(*args, **kwargs):
            captured["blocks"] = list(kwargs.get("blocks") or [])
            captured["texts"] = list(kwargs.get("texts") or [])
            return {
                "image": "band_003",
                "width": 300,
                "height": 150,
                "texts": [{"bbox": [22, 58, 278, 84], "text": "Successful candidate inquiry"}],
                "_vision_blocks": [{"bbox": [22, 58, 278, 84], "text": "Successful candidate inquiry"}],
            }

        with patch.dict(os.environ, {"TRADUZAI_UIED_LAYOUT": "1"}), patch(
            "vision_stack.runtime._get_ocr_engine"
        ) as get_ocr, patch(
            "vision_stack.runtime.build_page_result",
            side_effect=fake_build_page_result,
        ):
            get_ocr.return_value._backend = "mock-ocr"
            get_ocr.return_value.recognize_batch.return_value = [
                "20** regional fireman",
                "Successful candidate inquiry",
            ]

            run_ocr_stage(image, {"numero": 3, "_vision_blocks": []}, profile="quality")

        self.assertTrue(any(getattr(block, "detector", "") == "uied_cv" for block in captured["blocks"]))
        get_ocr.return_value.recognize_batch.assert_called_once()

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

    def test_koharu_worker_batch_invokes_worker_once_and_maps_responses(self):
        image_a = np.full((60, 80, 3), 255, dtype=np.uint8)
        image_b = np.full((70, 90, 3), 255, dtype=np.uint8)

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            worker_path = tmp_path / "traduzai-vision.exe"
            worker_path.write_text("", encoding="utf-8")
            page_a = tmp_path / "roi-a.jpg"
            page_b = tmp_path / "roi-b.jpg"
            page_a.write_text("", encoding="utf-8")
            page_b.write_text("", encoding="utf-8")

            def fake_run(args, **kwargs):
                self.assertEqual(args[1], "--batch-request-file")
                request_path = Path(args[2])
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                self.assertEqual(len(payload["requests"]), 2)
                self.assertEqual(payload["requests"][0]["imagePath"], str(page_a))
                self.assertEqual(payload["requests"][0]["mode"], "ocrOnly")
                self.assertEqual(payload["requests"][0]["knownTextBBoxes"], [[8, 9, 54, 40]])
                self.assertLessEqual(payload["requests"][0]["maxNewTokens"], 96)
                self.assertEqual(payload["requests"][0]["enginePresetId"], "manhwa_manhua")
                self.assertEqual(
                    payload["requests"][0]["engineSteps"],
                    [
                        "comic-text-bubble-detector",
                        "comic-text-detector-seg",
                        "speech-bubble-segmentation",
                        "paddle-ocr-vl-1.5",
                        "aot-inpainting",
                    ],
                )
                self.assertEqual(payload["requests"][0]["maskStrategy"], "roi_segmentation_assisted")
                self.assertEqual(payload["requests"][1]["imagePath"], str(page_b))
                self.assertEqual(payload["requests"][1]["mode"], "page")
                self.assertTrue(kwargs.get("capture_output"))
                return SimpleNamespace(
                    returncode=0,
                    stderr="",
                    stdout=json.dumps(
                        {
                            "status": "ok",
                            "responses": [
                                {
                                    "index": 0,
                                    "status": "ok",
                                    "response": {
                                        "status": "ok",
                                        "imageWidth": 80,
                                        "imageHeight": 60,
                                        "textBlocks": [
                                            {
                                                "bbox": [10, 12, 50, 34],
                                                "confidence": 0.94,
                                                "text": "\ud55c\uad6d\uc5b4",
                                                "detector": "koharu",
                                                "sourceDirection": "horizontal",
                                                "linePolygons": [
                                                    [[10, 12], [50, 12], [50, 34], [10, 34]]
                                                ],
                                            }
                                        ],
                                        "bubbleRegions": [],
                                        "timingsMs": {"detect": 3, "ocr": 4},
                                        "warnings": [],
                                    },
                                    "error": None,
                                },
                                {
                                    "index": 1,
                                    "status": "ok",
                                    "response": {
                                        "status": "ok",
                                        "imageWidth": 90,
                                        "imageHeight": 70,
                                        "textBlocks": [],
                                        "bubbleRegions": [],
                                        "timingsMs": {"detect": 1, "ocr": 0},
                                        "warnings": [],
                                    },
                                    "error": None,
                                },
                            ],
                            "timingsMs": {"prepare": 100, "total": 120},
                            "warnings": [],
                        },
                        ensure_ascii=False,
                    ),
                )

            with patch("vision_stack.runtime.subprocess.run", side_effect=fake_run) as run_mock, patch.dict(
                "os.environ",
                {"TRADUZAI_KOHARU_WORKER_PERSISTENT": "0"},
                clear=False,
            ):
                pages = _run_koharu_worker_detect_ocr_batch(
                    [
                        {
                            "image_path": str(page_a),
                            "image_rgb": image_a,
                            "mode": "roi",
                            "known_text_bboxes": [[8, 9, 54, 40]],
                        },
                        {"image_path": str(page_b), "image_rgb": image_b, "mode": "roi"},
                    ],
                    vision_worker_path=str(worker_path),
                    models_dir=str(tmp_path),
                    profile="max",
                    idioma_origem="ko",
                )

        run_mock.assert_called_once()
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["_vision_backend"], "koharu-worker-batch")
        self.assertEqual(pages[0]["texts"][0]["text"], "\ud55c\uad6d\uc5b4")
        self.assertEqual(pages[0]["texts"][0]["line_polygons"][0][0], [10, 12])
        self.assertEqual(pages[0]["_koharu_worker_batch"]["timings_ms"]["detect"], 3)
        self.assertEqual(pages[0]["_koharu_worker_batch"]["batch_timings_ms"]["prepare"], 100)
        self.assertEqual(pages[0]["_koharu_worker_batch"]["ocr_only_job_count"], 1)
        self.assertEqual(pages[1]["texts"], [])

    def test_run_detect_ocr_uses_koharu_http_for_cjk_when_available(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        koharu_page = {
            "image": "page.jpg",
            "width": 100,
            "height": 100,
            "texts": [{"bbox": [10, 20, 50, 40], "text": "\ud658\uc0dd\ucc9c\ub9c8"}],
            "_vision_blocks": [{"bbox": [10, 20, 50, 40], "mask": None, "confidence": 0.93}],
        }

        with patch("vision_stack.runtime.cv2.imread", return_value=image), patch(
            "vision_stack.runtime._should_use_koharu_cjk_ocr",
            return_value=True,
        ), patch(
            "vision_stack.runtime._run_koharu_cjk_http_detect_ocr",
            return_value=koharu_page,
        ) as run_koharu, patch(
            "vision_stack.runtime._quick_text_presence_check",
            side_effect=AssertionError("quick scan nao deve bloquear OCR CJK Koharu"),
        ), patch(
            "vision_stack.runtime._run_detect_ocr_on_image",
            side_effect=AssertionError("stack atual nao deveria rodar quando Koharu CJK funciona"),
        ):
            result = run_detect_ocr("page.jpg", profile="quality", idioma_origem="ko")

        run_koharu.assert_called_once()
        self.assertEqual(run_koharu.call_args.kwargs["engine_preset_id"], "manhwa_manhua")
        self.assertEqual(result["texts"][0]["text"], "\ud658\uc0dd\ucc9c\ub9c8")

    def test_koharu_http_client_runs_batch_import_and_pipeline_once(self):
        from pathlib import Path

        from vision_stack.runtime import _KoharuHttpOcrClient

        client = _KoharuHttpOcrClient(Path("N:/TraduzAI/koharu/koharu.exe"))
        client.start = MagicMock()
        client._ensure_project = MagicMock()
        client._wait_operation = MagicMock(return_value={"status": "completed"})
        requests = []

        def fake_request(method, path, payload=None, timeout=120):
            requests.append((method, path, payload))
            if path == "/pages/from-paths":
                return {"pages": ["page-a", "page-b"]}
            if path == "/pipelines":
                return {"operationId": "op-1"}
            if path == "/scene.json":
                return {
                    "scene": {
                        "pages": {
                            "page-a": {
                                "nodes": {
                                    "n1": {
                                        "transform": {"x": 10, "y": 20, "width": 30, "height": 10},
                                        "kind": {"text": {"text": "도저히", "confidence": 0.9}},
                                    }
                                }
                            },
                            "page-b": {"nodes": {}},
                        }
                    }
                }
            raise AssertionError(f"unexpected request {method} {path}")

        client.request_json = MagicMock(side_effect=fake_request)
        jobs = [
            {"image_path": "a.jpg", "image_rgb": np.full((80, 120, 3), 255, dtype=np.uint8)},
            {"image_path": "b.jpg", "image_rgb": np.full((90, 130, 3), 255, dtype=np.uint8)},
        ]

        pages = client.run_ocr_batch(jobs, profile="max", idioma_origem="ko")

        self.assertEqual([page["image"] for page in pages], ["a.jpg", "b.jpg"])
        self.assertEqual(pages[0]["texts"][0]["text"], "도저히")
        self.assertEqual(pages[1]["texts"], [])
        self.assertEqual(requests[0][1], "/pages/from-paths")
        self.assertEqual(requests[0][2]["paths"], [str(Path("a.jpg").resolve()), str(Path("b.jpg").resolve())])
        self.assertEqual(requests[0][2]["replace"], True)
        self.assertEqual(requests[1][1], "/pipelines")
        self.assertEqual(requests[1][2]["pages"], ["page-a", "page-b"])
        self.assertEqual(
            requests[1][2]["steps"],
            [
                "comic-text-bubble-detector",
                "comic-text-detector-seg",
                "speech-bubble-segmentation",
                "paddle-ocr-vl-1.5",
                "aot-inpainting",
            ],
        )
        self.assertEqual(pages[0]["_engine_preset"]["engine_preset_id"], "manhwa_manhua")
        self.assertEqual(pages[0]["_engine_preset"]["mask_strategy"], "roi_segmentation_assisted")

    def test_koharu_http_client_uses_manga_engine_steps_when_requested(self):
        from pathlib import Path

        from vision_stack.runtime import _KoharuHttpOcrClient

        client = _KoharuHttpOcrClient(Path("N:/TraduzAI/koharu/koharu.exe"))
        client.start = MagicMock()
        client._ensure_project = MagicMock()
        client._wait_operation = MagicMock(return_value={"status": "completed"})
        requests = []

        def fake_request(method, path, payload=None, timeout=120):
            requests.append((method, path, payload))
            if path == "/pages/from-paths":
                return {"pages": ["page-a"]}
            if path == "/pipelines":
                return {"operationId": "op-1"}
            if path == "/scene.json":
                return {"scene": {"pages": {"page-a": {"nodes": {}}}}}
            raise AssertionError(f"unexpected request {method} {path}")

        client.request_json = MagicMock(side_effect=fake_request)

        page = client.run_ocr(
            "a.jpg",
            np.full((80, 120, 3), 255, dtype=np.uint8),
            profile="max",
            idioma_origem="ja",
            engine_preset_id="manga",
        )

        self.assertEqual(
            requests[1][2]["steps"],
            [
                "comic-text-bubble-detector",
                "yuzumarker-font-detection",
                "comic-text-detector-seg",
                "speech-bubble-segmentation",
                "paddle-ocr-vl-1.5",
                "aot-inpainting",
            ],
        )
        self.assertEqual(page["_engine_preset"]["engine_preset_id"], "manga")
        self.assertEqual(page["_engine_preset"]["mask_strategy"], "segmentation_assisted")

    def test_run_detect_ocr_cjk_koharu_failure_falls_back_to_quick_skip(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)

        with patch("vision_stack.runtime.cv2.imread", return_value=image), patch(
            "vision_stack.runtime._should_use_koharu_cjk_ocr",
            return_value=True,
        ), patch(
            "vision_stack.runtime._run_koharu_cjk_http_detect_ocr",
            side_effect=RuntimeError("koharu offline"),
        ), patch(
            "vision_stack.runtime._quick_text_presence_check",
            return_value=False,
        ) as quick_check, patch(
            "vision_stack.runtime._run_detect_ocr_on_image",
            side_effect=AssertionError("stack atual nao deveria rodar quando quick skip confirma pagina vazia"),
        ):
            result = run_detect_ocr("page.jpg", profile="quality", idioma_origem="ja")

        quick_check.assert_called_once()
        self.assertEqual(result["texts"], [])
        self.assertTrue(result["quick_skipped_no_text"])
        self.assertEqual(result["koharu_cjk_fallback"], "quick_skip")

    def test_run_detect_ocr_recovers_sparse_page_with_full_page_lines_when_primary_result_is_empty(self):
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        block = SimpleNamespace(xyxy=(10, 20, 50, 40), mask=None, confidence=0.42)
        empty_page = {
            "image": "page.jpg",
            "width": 120,
            "height": 120,
            "texts": [],
            "_vision_blocks": [],
        }
        recovered_page = {
            "image": "page.jpg",
            "width": 120,
            "height": 120,
            "texts": [{"bbox": [18, 70, 104, 92], "text": "SO THIS IS HOW IT ENDS"}],
            "_vision_blocks": [{"bbox": [18, 70, 104, 92], "mask": None, "confidence": 0.88}],
        }

        with patch("vision_stack.runtime.cv2.imread", return_value=image), patch(
            "vision_stack.runtime._get_detector"
        ) as get_detector, patch("vision_stack.runtime._get_ocr_engine") as get_ocr, patch(
            "vision_stack.runtime.build_page_result",
            side_effect=[empty_page, recovered_page],
        ) as build_page_result:
            get_detector.return_value.detect.return_value = [block]
            get_detector.return_value.crop.return_value = image[20:40, 10:50]
            get_ocr.return_value._backend = "paddleocr"
            get_ocr.return_value.recognize_blocks_from_page.return_value = [{"text": "IL"}]
            get_ocr.return_value.recognize_full_page_lines.return_value = [
                {
                    "text": "SO THIS IS HOW IT ENDS",
                    "source_bbox": [18, 70, 104, 92],
                    "line_polygons": [],
                    "text_pixel_bbox": [18, 70, 104, 92],
                    "confidence": 0.88,
                }
            ]

            result = run_detect_ocr("page.jpg", profile="quality")

        self.assertEqual(result["texts"][0]["text"], "SO THIS IS HOW IT ENDS")
        get_ocr.return_value.recognize_full_page_lines.assert_called_once()
        self.assertEqual(build_page_result.call_count, 2)

    def test_vision_blocks_to_mask_uses_roi_strategy_without_full_bbox_fill(self):
        image = np.full((120, 160, 3), 248, dtype=np.uint8)
        block = {
            "bbox": [20, 20, 130, 90],
            "text_pixel_bbox": [55, 42, 95, 62],
        }

        mask = vision_blocks_to_mask(
            image.shape,
            [block],
            image_rgb=image,
            expand_mask=False,
            mask_strategy="roi_segmentation_assisted",
            ocr_texts=[block],
        )

        self.assertEqual(mask[52, 72], 255)
        self.assertEqual(mask[25, 25], 0)

    def test_vision_blocks_to_mask_uses_text_segmenter_for_manga_strategy(self):
        image = np.full((120, 160, 3), 248, dtype=np.uint8)
        block = {
            "bbox": [20, 20, 130, 90],
            "text_pixel_bbox": [55, 42, 95, 62],
        }
        seen_shapes = []

        def segmenter(crop):
            seen_shapes.append(crop.shape[:2])
            local = np.zeros(crop.shape[:2], dtype=np.uint8)
            local[24:30, 42:46] = 255
            local[24:30, 64:68] = 255
            return local

        mask = vision_blocks_to_mask(
            image.shape,
            [block],
            image_rgb=image,
            expand_mask=False,
            mask_strategy="segmentation_assisted",
            ocr_texts=[block],
            text_segmenter=segmenter,
        )

        self.assertEqual(seen_shapes, [(70, 110), (120, 160)])
        self.assertEqual(mask[46, 64], 255)
        self.assertEqual(mask[46, 86], 255)
        self.assertEqual(mask[25, 25], 0)

    def test_vision_blocks_to_mask_keeps_cjk_orphan_sfx_after_expansion(self):
        image = np.full((120, 160, 3), 248, dtype=np.uint8)
        image[78:84, 88:94] = 20
        block = {
            "bbox": [20, 20, 60, 50],
            "text_pixel_bbox": [30, 30, 45, 42],
        }

        def segmenter(crop):
            local = np.zeros(crop.shape[:2], dtype=np.uint8)
            if crop.shape[:2] == image.shape[:2]:
                local[78:84, 88:94] = 255
            else:
                local[10:16, 12:18] = 255
            return local

        mask = vision_blocks_to_mask(
            image.shape,
            [block],
            image_rgb=image,
            expand_mask=True,
            mask_strategy="segmentation_assisted",
            ocr_texts=[block],
            text_segmenter=segmenter,
        )

        self.assertEqual(mask[81, 91], 255)
        self.assertEqual(mask[110, 145], 0)

    def test_apply_inpainting_round_uses_page_text_segmenter_for_cjk_preset(self):
        image = np.full((120, 160, 3), 248, dtype=np.uint8)
        block = {
            "bbox": [20, 20, 130, 90],
            "text_pixel_bbox": [55, 42, 95, 62],
        }
        ocr_data = {
            "_vision_blocks": [block],
            "texts": [block],
            "engine_preset": {"segmenter": "manga-text-segmentation-2025"},
            "_engine_preset": {"mask_strategy": "segmentation_assisted"},
        }
        captured = {}

        def segmenter(crop):
            local = np.zeros(crop.shape[:2], dtype=np.uint8)
            local[24:30, 42:46] = 255
            local[24:30, 64:68] = 255
            return local

        def fake_inpaint(_inpainter, image_arg, mask_arg, **_kwargs):
            captured["mask"] = mask_arg.copy()
            return image_arg.copy()

        with patch("vision_stack.runtime._get_text_segmenter_for_page", return_value=segmenter), patch(
            "vision_stack.runtime._run_masked_inpaint_passes",
            side_effect=fake_inpaint,
        ):
            result = _apply_inpainting_round(image, ocr_data, object())

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(captured["mask"][46, 64], 255)
        self.assertEqual(captured["mask"][25, 25], 0)

    def test_apply_inpainting_round_passes_page_bubble_segmenter_for_assisted_preset(self):
        image = np.full((120, 160, 3), 248, dtype=np.uint8)
        block = {
            "bbox": [20, 20, 130, 90],
            "text_pixel_bbox": [55, 42, 95, 62],
        }
        ocr_data = {
            "_vision_blocks": [block],
            "texts": [block],
            "engine_preset": {
                "segmenter": "manga-text-segmentation-2025",
                "bubble_segmenter": "speech-bubble-segmentation",
            },
            "_engine_preset": {"mask_strategy": "segmentation_assisted"},
        }
        captured = {}

        def segmenter(crop):
            local = np.zeros(crop.shape[:2], dtype=np.uint8)
            local[24:36, 42:72] = 255
            return local

        def bubble_segmenter(crop):
            captured["bubble_called"] = True
            local = np.zeros(crop.shape[:2], dtype=np.uint8)
            local[20:42, 36:80] = 255
            return local

        def fake_inpaint(_inpainter, image_arg, mask_arg, **_kwargs):
            captured["mask"] = mask_arg.copy()
            return image_arg.copy()

        with patch("vision_stack.runtime._get_text_segmenter_for_page", return_value=segmenter), patch(
            "vision_stack.runtime._get_bubble_segmenter_for_page",
            return_value=bubble_segmenter,
        ), patch(
            "vision_stack.runtime._run_masked_inpaint_passes",
            side_effect=fake_inpaint,
        ):
            result = _apply_inpainting_round(image, ocr_data, object())

        self.assertEqual(result.shape, image.shape)
        self.assertTrue(captured["bubble_called"])
        self.assertEqual(captured["mask"][46, 64], 255)
        self.assertEqual(captured["mask"][25, 25], 0)

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
                        "bubbleId": "worker_bubble_001",
                        "bubbleInnerBbox": [18, 24, 90, 64],
                    }
                ],
            },
            profile="quality",
        )

        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["text"], "HELLO THERE")
        self.assertEqual(page["_bubble_regions"][0]["bbox"], [12, 18, 96, 70])
        self.assertEqual(page["_bubble_regions"][0]["bubble_id"], "worker_bubble_001")
        self.assertEqual(page["texts"][0]["bubble_id"], "worker_bubble_001")
        self.assertEqual(page["texts"][0]["bubble_mask_bbox"], [12, 18, 96, 70])
        self.assertEqual(page["texts"][0]["bubble_inner_bbox"], [18, 24, 90, 64])

    def test_build_koharu_worker_page_result_preserves_cjk_source_language(self):
        image = np.full((120, 180, 3), 255, dtype=np.uint8)

        page = _build_koharu_worker_page_result(
            image_rgb=image,
            image_label="page.jpg",
            worker_payload={
                "text_blocks": [
                    {
                        "bbox": [20, 30, 150, 64],
                        "confidence": 0.82,
                        "text": "\ud658\uc0dd\ucc9c\ub9c8",
                        "detector": "paddle-ocr-vl-1.5",
                    }
                ]
            },
            profile="quality",
            idioma_origem="ko",
        )

        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["text"], "\ud658\uc0dd\ucc9c\ub9c8")
        self.assertEqual(page["texts"][0]["ocr_mode"], "koharu-paddle-ocr-vl-1.5")

    def test_extract_koharu_scene_text_blocks_converts_transform_to_xyxy(self):
        page_id = "page-1"
        scene = {
            "scene": {
                "pages": {
                    page_id: {
                        "nodes": {
                            "node-1": {
                                "transform": {"x": 10, "y": 20, "width": 70, "height": 24},
                                "kind": {
                                    "text": {
                                        "text": "\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059",
                                        "confidence": 0.67,
                                        "linePolygons": [],
                                    }
                                },
                            }
                        }
                    }
                }
            }
        }

        blocks = _extract_koharu_scene_text_blocks(scene, page_id)

        self.assertEqual(blocks[0]["bbox"], [10, 20, 80, 44])
        self.assertEqual(blocks[0]["text"], "\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059")

    def test_should_use_koharu_cjk_ocr_is_cjk_only_and_can_be_disabled(self):
        with patch.dict(os.environ, {"TRADUZAI_KOHARU_CJK_OCR": "1"}, clear=False), patch(
            "vision_stack.runtime._resolve_koharu_exe",
            return_value=Path("N:/TraduzAI/koharu/koharu.exe"),
        ):
            self.assertTrue(_should_use_koharu_cjk_ocr("ko"))
            self.assertTrue(_should_use_koharu_cjk_ocr("ja"))
            self.assertTrue(_should_use_koharu_cjk_ocr("zh-CN"))
            self.assertFalse(_should_use_koharu_cjk_ocr("en"))

        with patch.dict(os.environ, {"TRADUZAI_KOHARU_CJK_OCR": "0"}, clear=False), patch(
            "vision_stack.runtime._resolve_koharu_exe",
            return_value=Path("N:/TraduzAI/koharu/koharu.exe"),
        ):
            self.assertFalse(_should_use_koharu_cjk_ocr("ko"))

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
        ) as get_ocr, patch(
            "vision_stack.runtime._get_font_detector",
            return_value=FakeFontDetector(),
        ):
            warmup_visual_stack(models_dir="models", profile="quality", lang="ko")

        configure_roots.assert_called_once_with("models")
        get_ocr.assert_called_once_with("quality", lang="ko")
        self.assertEqual(detector_calls, ["detect:256x256"])
        self.assertEqual(ocr_calls, [1])
        self.assertEqual(font_calls, [False])

    def test_warmup_visual_stack_can_load_models_without_sample_inference(self):
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
        ) as get_detector, patch(
            "vision_stack.runtime._get_ocr_engine",
            return_value=FakeOcr(),
        ) as get_ocr, patch(
            "vision_stack.runtime._get_font_detector",
            return_value=FakeFontDetector(),
        ) as get_font:
            warmup_visual_stack(models_dir="models", profile="quality", run_sample=False, lang="ja")

        configure_roots.assert_called_once_with("models")
        get_detector.assert_called_once_with("quality")
        get_ocr.assert_called_once_with("quality", lang="ja")
        get_font.assert_called_once_with()
        self.assertEqual(detector_calls, [])
        self.assertEqual(ocr_calls, [])
        self.assertEqual(font_calls, [])

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

    def test_apply_white_balloon_fill_preserves_outline_outside_text_bbox(self):
        image = np.full((220, 260, 3), 250, dtype=np.uint8)
        cv2.ellipse(image, (130, 112), (96, 62), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (130, 112), (96, 62), 0, 0, 360, (20, 20, 20), 2)
        image[104:114, 92:168] = 12
        image[120:130, 96:164] = 12

        filled = _apply_white_balloon_fill(
            image,
            [34, 50, 226, 174],
            text_bbox=[92, 104, 168, 130],
        )

        self.assertGreater(int(filled[110, 130, 0]), 242)
        self.assertGreater(int(filled[124, 130, 0]), 242)
        self.assertLess(int(filled[112, 34, 0]), 80)
        self.assertLess(int(filled[50, 130, 0]), 80)

    def test_apply_white_balloon_fill_preserves_antialiased_outline_outside_text_bbox(self):
        image = np.full((220, 260, 3), 250, dtype=np.uint8)
        cv2.ellipse(image, (130, 112), (96, 62), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (130, 112), (96, 62), 0, 0, 360, (165, 165, 165), 2, cv2.LINE_AA)
        image[104:114, 92:168] = 12
        image[120:130, 96:164] = 12

        top_outline = image[50, 130].copy()
        left_outline = image[112, 34].copy()
        filled = _apply_white_balloon_fill(
            image,
            [34, 50, 226, 174],
            text_bbox=[92, 104, 168, 130],
        )

        self.assertGreater(int(filled[110, 130, 0]), 242)
        self.assertGreater(int(filled[124, 130, 0]), 242)
        self.assertTrue(np.array_equal(filled[50, 130], top_outline))
        self.assertTrue(np.array_equal(filled[112, 34], left_outline))

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

    def test_white_balloon_cleanup_expands_when_ocr_bbox_misses_upper_line(self):
        original = np.full((260, 240, 3), 250, dtype=np.uint8)
        cv2.ellipse(original, (120, 130), (86, 68), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(original, (120, 130), (86, 68), 0, 0, 360, (20, 20, 20), 2)
        original[76:88, 84:156] = 18
        original[105:117, 62:178] = 18
        original[132:144, 74:166] = 18
        cleaned = original.copy()
        cleaned[76:88, 84:156] = 30
        cleaned[105:117, 62:178] = 80
        cleaned[132:144, 74:166] = 80
        text = {
            "bbox": [62, 105, 178, 144],
            "text_pixel_bbox": [62, 105, 178, 144],
            "block_profile": "top_narration",
            "translated": "AGORA EU SEI.",
            "original": "source",
            "skip_processing": False,
        }

        with patch("vision_stack.runtime._is_white_balloon_region", return_value=True):
            self.assertTrue(_has_white_balloon_text_residual(original, cleaned, [text]))
            result = _apply_white_balloon_text_box_cleanup(original, cleaned, [text])

        self.assertGreater(int(result[82, 120, 0]), 245)
        self.assertGreater(int(result[111, 120, 0]), 245)
        self.assertGreater(int(result[138, 120, 0]), 245)

    def test_white_balloon_residual_force_fill_removes_leftover_text_without_erasing_outline(self):
        original = np.full((120, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(original, (40, 30), (180, 90), (0, 0, 0), 2)
        cleaned = original.copy()
        cleaned[52:68, 82:138] = [0, 0, 0]
        text = {
            "bbox": [76, 48, 144, 74],
            "text_pixel_bbox": [82, 52, 138, 68],
            "balloon_bbox": [40, 30, 180, 90],
            "balloon_type": "white",
            "skip_processing": False,
        }

        forced = _apply_white_balloon_residual_force_fill(original, cleaned, [text])

        self.assertEqual(int(np.min(forced[52:68, 82:138])), 0)
        self.assertLessEqual(int(np.max(forced[30:32, 40:180])), 12)

    def test_geometry_white_cleanup_uses_full_fill_mask_when_bbox_misses_upper_text(self):
        original = np.full((230, 240, 3), 255, dtype=np.uint8)
        cv2.ellipse(original, (120, 112), (82, 62), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(original, (120, 112), (82, 62), 0, 0, 360, (15, 15, 15), 2)
        original[82:94, 92:148] = 20
        original[120:136, 70:170] = 20
        cleaned = original.copy()
        cleaned[120:136, 70:170] = 255
        text = {
            "bbox": [70, 110, 170, 150],
            "text_pixel_bbox": [70, 120, 170, 136],
            "balloon_bbox": [70, 110, 170, 150],
            "balloon_type": "white",
            "layout_profile": "top_narration",
            "skip_processing": False,
        }

        result = _apply_geometry_white_balloon_cleanup(original, cleaned, [text])

        self.assertGreaterEqual(int(result[88, 120, 0]), 245)
        self.assertLess(
            int(np.count_nonzero(result[82:94, 92:148, 0] < 80)),
            int(np.count_nonzero(cleaned[82:94, 92:148, 0] < 80)),
        )
        self.assertLessEqual(int(result[112, 38, 0]), 25)

    def test_geometry_white_cleanup_rejects_bright_textured_misclassification(self):
        original = np.full((180, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(original, (40, 28), (180, 132), (255, 255, 255), -1)
        cv2.rectangle(original, (40, 28), (180, 132), (12, 12, 12), 2)
        original[68:80, 90:140] = 20
        original[88:102, 76:158] = 20
        cleaned = original.copy()
        cleaned[88:102, 76:158] = 255
        text = {
            "bbox": [72, 82, 162, 108],
            "text_pixel_bbox": [76, 88, 158, 102],
            "balloon_bbox": [72, 82, 162, 108],
            "balloon_type": "textured",
            "layout_profile": "standard",
            "skip_processing": False,
        }

        result = _apply_geometry_white_balloon_cleanup(original, cleaned, [text])

        self.assertFalse(np.array_equal(result, cleaned))
        self.assertLess(
            int(np.count_nonzero(result[68:80, 90:140, 0] < 80)),
            int(np.count_nonzero(cleaned[68:80, 90:140, 0] < 80)),
        )
        self.assertLessEqual(int(result[28, 100, 0]), 25)

    def test_translucent_white_balloon_is_not_white_cleanup_safe(self):
        image = np.full((150, 220, 3), 244, dtype=np.uint8)
        for x in range(40, 180):
            image[36:118, x, :] = 232 + ((x - 40) % 24)
        image[70:86, 82:138] = 18
        text = {
            "bbox": [76, 66, 144, 92],
            "text_pixel_bbox": [82, 70, 138, 86],
            "balloon_bbox": [40, 36, 180, 118],
            "balloon_type": "white",
            "skip_processing": False,
        }

        self.assertTrue(_text_background_looks_translucent_or_textured(image, text))
        self.assertFalse(_text_is_white_cleanup_safe(image, text))

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

    def test_apply_inpainting_round_falls_back_from_overbroad_balloon_bbox_in_white_balloon_cleanup(self):
        original = np.full((220, 240, 3), 252, dtype=np.uint8)
        cv2.ellipse(original, (120, 110), (78, 48), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(original, (120, 110), (78, 48), 0, 0, 360, (20, 20, 20), 2)
        original[84:96, 78:162] = 10
        original[104:118, 62:178] = 10
        original[128:138, 86:154] = 10

        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        mask[84:138, 62:178] = 255

        ocr_data = {
            "texts": [
                {
                    "bbox": [62, 84, 178, 138],
                    "text_pixel_bbox": [60, 80, 186, 142],
                    "balloon_bbox": [42, 58, 198, 162],
                    "skip_processing": False,
                    "text": "QUEBROU O TEXTO",
                }
            ],
            "_vision_blocks": [
                {"bbox": [62, 84, 178, 138], "mask": None, "confidence": 0.95},
            ],
        }

        class FakeInpainter:
            def inpaint(self, image_np, mask, batch_size=4, debug=None, force_no_tiling=False):
                result = image_np.copy()
                # Simula resíduo cinza escuro mais largo que as linhas originais.
                result[80:100, 72:168] = [92, 92, 92]
                result[100:122, 54:186] = [98, 98, 98]
                result[124:142, 80:160] = [100, 100, 100]
                return result

        with patch("vision_stack.runtime.vision_blocks_to_mask", return_value=mask):
            cleaned = _apply_inpainting_round(original, ocr_data, FakeInpainter())

        cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY)
        x1, y1, x2, y2 = ocr_data["texts"][0]["text_pixel_bbox"]
        residue = int(np.count_nonzero(cleaned_gray[y1:y2, x1:x2] <= 120))
        self.assertEqual(residue, 0)

    def test_apply_inpainting_round_clamps_model_changes_outside_mask(self):
        original = np.full((90, 140, 3), 240, dtype=np.uint8)
        original[34:46, 46:94] = 10
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        mask[34:46, 46:94] = 255
        ocr_data = {
            "texts": [
                {
                    "bbox": [44, 32, 96, 48],
                    "text_pixel_bbox": [46, 34, 94, 46],
                    "line_polygons": [[[46, 34], [94, 34], [94, 46], [46, 46]]],
                    "balloon_type": "white",
                    "layout_profile": "white_balloon",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [{"bbox": [44, 32, 96, 48], "text_pixel_bbox": [46, 34, 94, 46]}],
        }

        class WideChangingInpainter:
            def inpaint(self, image_np, mask, batch_size=4, debug=None, force_no_tiling=False):
                result = image_np.copy()
                result[:, :] = [30, 30, 30]
                result[mask > 0] = [245, 245, 245]
                return result

        with patch("vision_stack.runtime.vision_blocks_to_mask", return_value=mask), patch(
            "vision_stack.runtime._select_inpaint_roi",
            return_value=([0, 0, original.shape[1], original.shape[0]], False),
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda base, candidate, texts, **kwargs: (candidate, {}),
        ), patch("vision_stack.runtime._has_white_balloon_text_residual", return_value=False):
            cleaned = _apply_inpainting_round(original, ocr_data, WideChangingInpainter())

        self.assertTrue(np.array_equal(cleaned[8, 8], original[8, 8]))
        self.assertTrue(np.all(cleaned[36:44, 50:90] >= 240))

    def test_apply_inpainting_round_uses_precomputed_mask_when_present(self):
        original = np.full((60, 80, 3), 240, dtype=np.uint8)
        precomputed = np.zeros(original.shape[:2], dtype=np.uint8)
        precomputed[20:32, 24:52] = 255
        ocr_data = {
            "texts": [{"bbox": [24, 20, 52, 32], "skip_processing": False}],
            "_vision_blocks": [{"bbox": [24, 20, 52, 32]}],
            "_precomputed_inpaint_mask": precomputed,
        }

        def fake_run(_inpainter, image_np, mask, **_kwargs):
            result = image_np.copy()
            result[mask > 0] = [250, 250, 250]
            return result

        with patch("vision_stack.runtime.vision_blocks_to_mask", side_effect=AssertionError("should not rebuild")), patch(
            "vision_stack.runtime._run_masked_inpaint_passes",
            side_effect=fake_run,
        ) as mocked_run:
            cleaned = _apply_inpainting_round(original, ocr_data, object())

        used_mask = mocked_run.call_args.args[2]
        self.assertTrue(np.array_equal(used_mask, precomputed))
        self.assertTrue(np.all(cleaned[22:30, 28:48] >= 240))

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

    def test_run_masked_inpaint_passes_uses_full_image_when_roi_would_not_save_work(self):
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
        self.assertEqual(calls[0]["shape"], image.shape)
        self.assertEqual(result["final_output"].shape, image.shape)

    def test_run_masked_inpaint_passes_crops_sparse_large_masks_by_default(self):
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
                result = np.full_like(image_np, 23)
                result[mask > 0] = [220, 220, 220]
                return result

        image = np.full((900, 1200, 3), 127, dtype=np.uint8)
        mask = np.zeros((900, 1200), dtype=np.uint8)
        mask[380:430, 510:590] = 255

        result = _run_masked_inpaint_passes(FakeInpainter(), image, mask, batch_size=4)

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["force_no_tiling"])
        self.assertLess(calls[0]["shape"][0], image.shape[0])
        self.assertLess(calls[0]["shape"][1], image.shape[1])
        self.assertEqual(result["final_output"].shape, image.shape)
        self.assertTrue(np.array_equal(result["final_output"][20, 20], image[20, 20]))
        self.assertGreaterEqual(int(result["final_output"][405, 550, 0]), 220)

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

        self.assertGreaterEqual(int(result[90, 90, 0]), 240)
        self.assertGreaterEqual(int(result[105, 90, 0]), 240)

    def test_koharu_blockwise_falls_back_to_full_page_when_white_balloon_residual_survives_cleanup(self):
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        bbox = [48, 42, 112, 78]
        ocr_data = {
            "texts": [
                {
                    "bbox": bbox,
                    "text_pixel_bbox": [48, 8, 112, 36],
                    "balloon_bbox": [36, 28, 124, 92],
                    "skip_processing": False,
                    "text": "HELLO",
                },
            ],
            "_vision_blocks": [
                {"bbox": bbox, "mask": None, "confidence": 0.95},
            ],
        }

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[42:78, 48:112] = 255

        residual = image.copy()
        residual[52:68, 64:96] = [0, 0, 0]

        recovered = image.copy()

        class FakeInpainter:
            pass

        def fake_run_masked_inpaint_passes(inpainter, crop_image, crop_mask, **kwargs):
            crop_residual = crop_image.copy()
            crop_residual[10:26, 16:48] = [0, 0, 0]
            return {
                "expanded_mask": mask.copy(),
                "raw_output": crop_residual.copy(),
                "after_roi_paste": crop_residual.copy(),
                "after_seam_cleanup": crop_residual.copy(),
                "final_output": crop_residual.copy(),
                "cleanup_base_mask": mask.copy(),
                "fallback_to_legacy": False,
                "fallback_error": "",
            }

        balloon_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        balloon_mask[34:90, 40:120] = 255

        with patch("vision_stack.runtime.vision_blocks_to_mask", return_value=mask.copy()), patch(
            "vision_stack.runtime._try_koharu_balloon_fill",
            return_value=None,
        ), patch(
            "vision_stack.runtime._run_masked_inpaint_passes",
            side_effect=fake_run_masked_inpaint_passes,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup",
            return_value=residual.copy(),
        ), patch(
            "vision_stack.runtime._is_white_balloon_region",
            return_value=True,
        ), patch(
            "vision_stack.runtime._extract_white_balloon_fill_mask",
            return_value=balloon_mask,
        ), patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=recovered.copy(),
        ) as fallback_round:
            result = _run_koharu_blockwise_inpaint_page(image, ocr_data, FakeInpainter())

        fallback_round.assert_called_once()
        self.assertEqual(int(result[60, 80, 0]), 255)

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

        self.assertGreaterEqual(int(result[0, 0, 0]), 245)

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

        def fake_run_masked_inpaint_passes(inpainter, image_np, mask, batch_size=4, **kwargs):
            del kwargs
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

        self.assertEqual(calls, ["lama", "line_cleanup", "micro_cleanup"])
        self.assertEqual(int(result[55, 80, 0]), 248)

    def test_apply_inpainting_round_skips_cjk_aot_force_fill_after_masked_crop(self):
        image = np.full((120, 160, 3), 210, dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[52:62, 72:84] = 255
        ocr_data = {
            "texts": [
                {
                    "bbox": [30, 30, 130, 90],
                    "text_pixel_bbox": [30, 30, 130, 90],
                    "balloon_bbox": [24, 24, 136, 96],
                    "balloon_type": "white",
                    "block_profile": "white_balloon",
                    "skip_processing": False,
                    "text": "SFX",
                }
            ],
            "_vision_blocks": [{"bbox": [30, 30, 130, 90], "mask": None, "confidence": 0.9}],
            "engine_preset": {"inpainter": "aot-inpainting"},
            "_engine_preset": {"mask_strategy": "segmentation_assisted"},
        }
        pass_kwargs = {}

        def fake_run_masked_inpaint_passes(inpainter, image_np, mask_arg, **kwargs):
            del inpainter
            pass_kwargs.update(kwargs)
            result = image_np.copy()
            result[mask_arg > 0] = [77, 77, 77]
            return {
                "expanded_mask": mask_arg.copy(),
                "raw_output": result.copy(),
                "after_roi_paste": result.copy(),
                "after_seam_cleanup": result.copy(),
                "final_output": result.copy(),
                "cleanup_base_mask": mask_arg.copy(),
                "fallback_to_legacy": False,
                "fallback_error": "",
            }

        def fake_cleanup(original_rgb, cleaned_rgb, texts, **kwargs):
            del original_rgb, texts, kwargs
            return cleaned_rgb.copy(), {"cleanup_changed_outside_limit_mask": 0, "cleanup_limit_mask_pixels": 0}

        with patch("vision_stack.runtime.vision_blocks_to_mask", return_value=mask), patch(
            "vision_stack.runtime._run_masked_inpaint_passes",
            side_effect=fake_run_masked_inpaint_passes,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=fake_cleanup,
        ), patch(
            "vision_stack.runtime._has_white_balloon_text_residual",
            return_value=True,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_residual_force_fill",
            side_effect=AssertionError("force-fill nao deve rodar no modo CJK+AOT estrito"),
        ):
            result = _apply_inpainting_round(image, ocr_data, inpainter=object())

        self.assertIs(pass_kwargs.get("expand_mask"), False)
        self.assertEqual(pass_kwargs.get("crop_windows"), [[0, 0, 160, 120]])
        self.assertEqual(int(result[55, 75, 0]), 77)
        self.assertEqual(int(result[35, 35, 0]), 210)
        self.assertEqual(ocr_data.get("_inpaint_white_residual_force_fill"), False)
        self.assertEqual(ocr_data.get("_inpaint_white_residual_force_fill_skipped"), "strict_cjk_aot")

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
                "vision_stack.runtime._apply_inpainting_round",
                return_value=cleaned,
            ) as full_page_round, patch(
                "vision_stack.runtime._run_detect_ocr_on_image",
                side_effect=AssertionError("o passo 5 foi removido e nao deve mais rodar"),
            ), patch(
                "vision_stack.runtime._integrate_recovery_page",
                side_effect=AssertionError("o passo 5 foi removido e nao deve mais rodar"),
            ):
                outputs = run_inpaint_pages([image_path], [ocr_data], str(output_dir))

            result = np.array(Image.open(outputs[0]).convert("RGB"))

        self.assertEqual(full_page_round.call_count, 1)
        self.assertEqual(int(result[45, 70, 0]), 123)

    def test_run_inpaint_pages_does_not_use_blockwise_path_anymore(self):
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
                side_effect=AssertionError("blockwise nao deveria mais ser usado no inpaint"),
            ), patch(
                "vision_stack.runtime._apply_inpainting_round",
                return_value=cleaned,
            ) as full_page_round:
                outputs = run_inpaint_pages([image_path], [ocr_data], str(output_dir))

            result = np.array(Image.open(outputs[0]).convert("RGB"))

        self.assertEqual(full_page_round.call_count, 1)
        self.assertEqual(int(result[45, 70, 0]), 123)

    def test_run_inpaint_pages_selects_aot_from_engine_preset(self):
        image = np.full((120, 160, 3), 200, dtype=np.uint8)
        cleaned = image.copy()
        cleaned[42:54, 60:96] = 123
        ocr_data = {
            "engine_preset": {
                "id": "manhwa_manhua",
                "inpainter": "aot-inpainting",
            },
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

            with patch("vision_stack.runtime._get_inpainter", return_value=object()) as get_inpainter, patch(
                "vision_stack.runtime._apply_inpainting_round",
                return_value=cleaned,
            ):
                outputs = run_inpaint_pages([image_path], [ocr_data], str(output_dir))

            result = np.array(Image.open(outputs[0]).convert("RGB"))

        get_inpainter.assert_called_once_with("quality", model="aot-inpainting")
        self.assertEqual(ocr_data["_inpaint_engine"], "aot-inpainting")
        self.assertEqual(int(result[45, 70, 0]), 123)

    def test_inpainter_model_for_page_uses_aot_for_default_preset(self):
        from vision_stack.runtime import _inpainter_model_for_page

        self.assertEqual(_inpainter_model_for_page({"engine_preset": {"inpainter": "default"}}), "aot-inpainting")

    def test_page_engine_preset_dict_forces_koharu_visual_engines(self):
        from vision_stack.runtime import _page_engine_preset_dict

        preset = _page_engine_preset_dict(
            {
                "engine_preset": {
                    "id": "legacy",
                    "segmenter": "disabled",
                    "bubble_segmenter": "default",
                    "inpainter": "default",
                    "mask_strategy": "default",
                }
            }
        )

        self.assertEqual(preset["segmenter"], "comic-text-detector-seg")
        self.assertEqual(preset["bubble_segmenter"], "speech-bubble-segmentation")
        self.assertEqual(preset["inpainter"], "aot-inpainting")

    def test_inpaint_band_image_uses_only_koharu_fast_fill_before_aot(self):
        from inpainter import inpaint_band_image

        image = np.full((64, 96, 3), 240, dtype=np.uint8)
        image[30:34, 36:42] = 10
        image[30:34, 52:58] = 10
        bubble_mask = np.zeros((64, 96), dtype=np.uint8)
        bubble_mask[12:52, 16:80] = 3
        text = {
            "text": "SIM, NAO FUNCIONA",
            "bbox": [0, 0, 96, 64],
            "text_pixel_bbox": [32, 28, 64, 36],
            "line_polygons": [[[32, 28], [64, 28], [64, 36], [32, 36]]],
            "balloon_bbox": [0, 0, 96, 64],
            "bubble_mask": bubble_mask,
            "bubble_id": 3,
            "content_class": "noise",
            "tipo": "sfx",
            "balloon_type": "white",
            "skip_processing": True,
            "preserve_original": True,
        }
        ocr_data = {
            "texts": [dict(text)],
            "_vision_blocks": [dict(text)],
        }

        with patch("inpainter._apply_fast_solid_balloon_fill", side_effect=AssertionError("legacy solid fill")), patch(
            "inpainter._apply_fast_white_balloon_fill",
            side_effect=AssertionError("legacy white fill"),
        ), patch(
            "inpainter._apply_connected_white_geometry_fill",
            side_effect=AssertionError("legacy connected fill"),
        ), patch(
            "inpainter._apply_fast_dark_panel_text_fill",
            side_effect=AssertionError("legacy dark fill"),
        ), patch(
            "inpainter._apply_fast_local_balloon_fill",
            side_effect=AssertionError("legacy local fill"),
        ), patch(
            "vision_stack.runtime._get_inpainter",
            side_effect=AssertionError("AOT should not run when Koharu fast fill covers mask"),
        ):
            result = inpaint_band_image(image, ocr_data)

        self.assertTrue(ocr_data["_strip_used_koharu_fast_fill"])
        self.assertEqual(ocr_data["_strip_remaining_inpaint_blocks"], 0)
        self.assertTrue(np.all(result[30:34, 36:42] == 240))
        self.assertTrue(np.all(result[30:34, 52:58] == 240))

    def test_run_inpaint_pages_can_use_koharu_blockwise_path_by_flag(self):
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

            with patch.dict(os.environ, {"TRADUZAI_KOHARU_BLOCKWISE_INPAINT": "1"}), patch(
                "vision_stack.runtime._get_inpainter",
                return_value=object(),
            ), patch(
                "vision_stack.runtime._run_koharu_blockwise_inpaint_page",
                return_value=cleaned,
            ) as blockwise_round, patch(
                "vision_stack.runtime._apply_inpainting_round",
                side_effect=AssertionError("full-page nao deveria rodar com blockwise habilitado"),
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
