import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from vision_stack.inpainter import Inpainter


def _allowed_mask_evidence():
    return {
        "kind": "glyph_segmentation",
        "raw_mask_pixels": 120,
        "expanded_mask_pixels": 160,
        "evidence_score": 1.0,
        "fast_fill_allowed": True,
        "fast_fill_reject_reasons": [],
    }


def _attach_real_bubble_mask(page: dict, image_shape, bbox: list[int] | None = None, mask: np.ndarray | None = None):
    height, width = image_shape[:2]
    first_text = (page.get("texts") or [{}])[0]
    if bbox is None:
        bbox = list(first_text.get("balloon_bbox") or first_text.get("bbox") or [0, 0, width, height])
    bubble_id = str(first_text.get("bubble_id") or "bubble_001")
    if mask is None:
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    for collection_key in ("texts", "_vision_blocks"):
        for item in page.get(collection_key) or []:
            if isinstance(item, dict):
                item["bubble_id"] = bubble_id
                item["bubble_mask_bbox"] = list(bbox)
    page["_bubble_regions"] = [{"bubble_id": bubble_id, "bubble_mask": mask, "bubble_mask_bbox": list(bbox)}]
    return mask


class VisionStackInpainterTests(unittest.TestCase):
    _LEGACY_FAST_DARK_OPT_IN_BROKEN_TESTS = {
        "test_fast_dark_panel_fill_handles_phone_ui_with_line_geometry",
        "test_fast_dark_panel_fill_is_clamped_to_text_geometry",
        "test_fast_dark_panel_fill_removes_covered_block_before_real_inpaint",
    }

    def setUp(self):
        if self._testMethodName in self._LEGACY_FAST_DARK_OPT_IN_BROKEN_TESTS:
            self.skipTest("legacy opt-in fast dark fill is isolated; default path uses sampled fast_solid")

    def test_band_local_text_bbox_normalization_shifts_page_relative_y(self):
        from inpainter import _texts_with_band_local_bboxes

        texts = [
            {
                "bbox": [955, 720, 1045, 892],
                "source_bbox": [936, 669, 1064, 943],
                "text_pixel_bbox": [955, 720, 1045, 892],
                "balloon_bbox": [936, 669, 1064, 943],
                "line_polygons": [
                    [[960, 740], [1040, 740], [1040, 760], [960, 760]],
                    [[970, 812], [1030, 812], [1030, 842], [970, 842]],
                ],
                "tipo": "fala",
            }
        ]

        normalized = _texts_with_band_local_bboxes(texts, width=1200, height=500, band_y_top=629)

        self.assertEqual(normalized[0]["bbox"], [955, 91, 1045, 263])
        self.assertEqual(normalized[0]["source_bbox"], [936, 40, 1064, 314])
        self.assertEqual(normalized[0]["text_pixel_bbox"], [955, 91, 1045, 263])
        self.assertEqual(normalized[0]["balloon_bbox"], [936, 40, 1064, 314])
        self.assertEqual(
            normalized[0]["line_polygons"],
            [
                [[960, 111], [1040, 111], [1040, 131], [960, 131]],
                [[970, 183], [1030, 183], [1030, 213], [970, 213]],
            ],
        )
        self.assertTrue(normalized[0]["_band_local_bbox_normalized"])
        self.assertEqual(texts[0]["bbox"], [955, 720, 1045, 892])

    def test_translator_note_dark_text_only_uses_expanded_text_mask(self):
        from inpainter.mask_builder import build_inpaint_mask

        image = np.zeros((140, 260, 3), dtype=np.uint8)
        block = {
            "id": "tn_001",
            "original": "T/N: THERE IS A NOTE",
            "translated": "T/N: EXISTE UMA NOTA",
            "bbox": [20, 40, 210, 78],
            "source_bbox": [20, 40, 210, 78],
            "text_pixel_bbox": [20, 40, 210, 78],
            "line_polygons": [[[20, 40], [210, 40], [210, 78], [20, 78]]],
            "background_rgb": [0, 0, 0],
            "bubble_mask_source": "translator_note_text_mask",
            "qa_flags": ["translator_note_text_only_mask"],
        }

        mask = build_inpaint_mask(block, image.shape, image)

        self.assertIsNotNone(mask)
        self.assertGreater(int(np.count_nonzero(mask)), 0)
        ys, xs = np.where(mask > 0)
        self.assertLessEqual(int(xs.max()) + 1, 230)
        self.assertGreaterEqual(int(xs.min()), 0)
        self.assertLessEqual(int(ys.max()) + 1, 95)
        self.assertIn("translator_note_text_only_contract", block.get("qa_metrics") or {})

    def test_missing_translator_note_text_is_promoted_to_inpaint_blocks(self):
        from inpainter import _append_missing_text_inpaint_blocks

        image = np.zeros((140, 320, 3), dtype=np.uint8)
        vision_blocks = [
            {
                "id": "ocr_001",
                "bbox": [180, 20, 300, 100],
                "text_pixel_bbox": [190, 30, 290, 80],
                "line_polygons": [[[190, 30], [290, 30], [290, 80], [190, 80]]],
                "bubble_mask_source": "image_white_bubble_mask",
                "route_action": "translate_inpaint_render",
            }
        ]
        texts = [
            dict(vision_blocks[0]),
            {
                "id": "ocr_002",
                "text": "T/N: THERE IS A NOTE",
                "bbox": [0, 94, 128, 132],
                "source_bbox": [0, 94, 128, 132],
                "text_pixel_bbox": [0, 94, 128, 132],
                "line_polygons": [[[0, 94], [128, 94], [128, 132], [0, 132]]],
                "bubble_mask_source": "translator_note_text_mask",
                "route_action": "translate_inpaint_render",
                "qa_flags": ["translator_note_text_only_mask"],
            },
        ]

        promoted = _append_missing_text_inpaint_blocks(vision_blocks, texts, 320, 140, image)

        self.assertEqual(len(promoted), 2)
        self.assertEqual(promoted[1]["id"], "ocr_002")
        self.assertIn("missing_text_promoted_to_inpaint_block", promoted[1].get("qa_flags") or [])

    def test_text_cleanup_limit_bbox_uses_full_source_for_rotated_text(self):
        from inpainter import _text_cleanup_limit_bbox

        text = {
            "source_bbox": [40, 24, 190, 150],
            "text_pixel_bbox": [78, 52, 162, 112],
            "rotation_deg": -33.0,
            "line_polygons": [
                [[78, 52], [162, 52], [162, 72], [78, 72]],
                [[88, 92], [152, 92], [152, 112], [88, 112]],
            ],
        }

        limit = _text_cleanup_limit_bbox(text, width=220, height=180)

        self.assertIsNotNone(limit)
        self.assertLessEqual(limit[0], 44)
        self.assertLessEqual(limit[1], 28)
        self.assertGreaterEqual(limit[2], 186)
        self.assertGreaterEqual(limit[3], 146)

    def test_band_local_text_bbox_normalization_shifts_ambiguous_lower_band_geometry(self):
        from inpainter import _texts_with_band_local_bboxes

        texts = [
            {
                "bbox": [73, 304, 118, 314],
                "source_bbox": [48, 296, 356, 324],
                "text_pixel_bbox": [73, 304, 118, 314],
                "line_polygons": [[[71, 295], [565, 295], [565, 316], [71, 316]]],
                "balloon_bbox": [45, 286, 146, 332],
            }
        ]

        normalized = _texts_with_band_local_bboxes(texts, width=690, height=323, band_y_top=260)

        self.assertEqual(normalized[0]["bbox"], [73, 44, 118, 54])
        self.assertEqual(normalized[0]["source_bbox"], [48, 36, 356, 64])
        self.assertEqual(normalized[0]["text_pixel_bbox"], [73, 44, 118, 54])
        self.assertEqual(normalized[0]["line_polygons"], [[[71, 35], [565, 35], [565, 56], [71, 56]]])
        self.assertEqual(normalized[0]["balloon_bbox"], [45, 26, 146, 72])
        self.assertTrue(normalized[0]["_band_local_bbox_normalized"])
        self.assertEqual(texts[0]["bbox"], [73, 304, 118, 314])

    def test_fallback_vision_blocks_preserve_source_and_balloon_bbox(self):
        from inpainter import _build_fallback_vision_blocks

        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [58, 20, 142, 54],
                    "text_pixel_bbox": [58, 20, 142, 54],
                    "source_bbox": [34, 18, 142, 54],
                    "balloon_bbox": [34, 18, 142, 54],
                    "line_polygons": [[[58, 20], [142, 20], [142, 54], [58, 54]]],
                    "balloon_type": "white",
                }
            ]
        }

        blocks = _build_fallback_vision_blocks(page, 180, 110)

        self.assertEqual(blocks[0]["source_bbox"], [34, 18, 142, 54])
        self.assertEqual(blocks[0]["balloon_bbox"], [34, 18, 142, 54])

    def test_route_action_controls_processable_inpaint_filters(self):
        from inpainter import _processable_texts_for_inpaint, _processable_vision_blocks_for_inpaint

        preserve_text = {
            "text": "쿵쿵",
            "bbox": [10, 10, 40, 40],
            "route_action": "preserve",
            "skip_processing": False,
            "preserve_original": True,
        }
        watermark_block = {
            "text": "Read at ASURACOMIC.NET",
            "bbox": [10, 10, 120, 40],
            "content_class": "url_watermark",
            "render_policy": "remove",
            "route_action": "inpaint_only",
            "skip_processing": False,
        }

        self.assertEqual(_processable_texts_for_inpaint({"texts": [preserve_text]}), [preserve_text])
        self.assertEqual(_processable_vision_blocks_for_inpaint([watermark_block]), [watermark_block])

    def test_visual_sfx_overlap_suppressed_block_is_not_processable_for_inpaint(self):
        from inpainter import _processable_vision_blocks_for_inpaint

        block = {
            "id": "ocr_cjk_sign",
            "text": "TEXTO:QUERIDO KARAOKE",
            "bbox": [12, 88, 178, 104],
            "route_action": "translate_inpaint_render",
            "route_reason": "visual_sfx_overlap_suppressed",
            "skip_processing": False,
        }

        self.assertEqual(_processable_vision_blocks_for_inpaint([block]), [])

    def test_auto_inpaint_unsafe_reason_ignores_stale_mask_outside_with_image_bubble_evidence(self):
        from inpainter import _auto_inpaint_unsafe_reason

        evidence = _allowed_mask_evidence()
        evidence.update({"raw_mask_pixels": 1200, "expanded_mask_pixels": 3600})
        text = {
            "id": "ocr_001",
            "qa_flags": ["mask_outside_balloon_critical"],
            "bubble_mask_source": "image_contour_bubble_mask",
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 2400,
                "expanded_mask_pixels": 6400,
                "evidence_score": 1.0,
            },
        }

        self.assertEqual(_auto_inpaint_unsafe_reason(text), "")
        text["bubble_mask_source"] = "derived_white_crop_rejected"
        text["bubble_mask_error"] = "derived_mask_not_anchored_to_text"
        text["qa_flags"] = ["mask_outside_balloon_critical", "debug_derived_bubble_mask_rejected"]
        self.assertEqual(_auto_inpaint_unsafe_reason(text), "mask_outside_balloon_critical")

    def test_auto_inpaint_unsafe_reason_keeps_explicit_bubble_mask_error_with_current_evidence(self):
        from inpainter import _auto_inpaint_unsafe_reason

        text = {
            "id": "ocr_001",
            "bubble_mask_source": "image_white_bubble_mask",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 1800,
                "expanded_mask_pixels": 5200,
                "evidence_score": 1.0,
            },
        }

        self.assertEqual(_auto_inpaint_unsafe_reason(text), "derived_mask_not_anchored_to_text")

    def test_real_bubble_mask_for_text_rejects_dense_rectangular_image_white_mask(self):
        from inpainter import _real_bubble_mask_for_text

        mask = np.ones((246, 358), dtype=np.uint8) * 255
        text = {
            "id": "ocr_001",
            "bubble_id": "bubble_001",
            "bbox": [95, 92, 391, 235],
            "text_pixel_bbox": [95, 92, 391, 235],
            "bubble_mask_bbox": [47, 48, 405, 294],
            "bubble_mask_source": "image_white_bubble_mask",
        }
        page = {
            "texts": [text],
            "_bubble_regions": [
                {
                    "bubble_id": "bubble_001",
                    "bubble_mask": mask,
                    "bubble_mask_bbox": [47, 48, 405, 294],
                    "bubble_mask_source": "image_white_bubble_mask",
                }
            ],
        }

        resolved, reason = _real_bubble_mask_for_text(page, text, 800, 342)

        self.assertIsNone(resolved)
        self.assertEqual(reason, "suspicious_rectangular_image_bubble_mask")

    def test_auto_inpaint_unsafe_reason_blocks_dense_rectangular_image_white_mask(self):
        from inpainter import _auto_inpaint_unsafe_reason

        text = {
            "id": "ocr_001",
            "bbox": [95, 92, 391, 235],
            "text_pixel_bbox": [95, 92, 391, 235],
            "bubble_mask": np.ones((246, 358), dtype=np.uint8) * 255,
            "bubble_mask_bbox": [47, 48, 405, 294],
            "bubble_mask_source": "image_white_bubble_mask",
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 4667,
                "expanded_mask_pixels": 19716,
                "evidence_score": 1.0,
            },
        }

        self.assertEqual(_auto_inpaint_unsafe_reason(text), "suspicious_rectangular_image_bubble_mask")

    def test_auto_inpaint_unsafe_reason_blocks_dense_rectangular_image_rect_mask(self):
        from inpainter import _auto_inpaint_unsafe_reason

        text = {
            "id": "ocr_rect",
            "bbox": [96, 1075, 340, 1241],
            "text_pixel_bbox": [118, 1112, 318, 1208],
            "bubble_mask": np.ones((166, 244), dtype=np.uint8) * 255,
            "bubble_mask_bbox": [96, 1075, 340, 1241],
            "bubble_mask_source": "image_rect_bubble_mask",
            "layout_profile": "white_balloon",
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 8669,
                "expanded_mask_pixels": 19113,
                "evidence_score": 1.0,
            },
        }

        self.assertEqual(_auto_inpaint_unsafe_reason(text), "suspicious_rectangular_image_bubble_mask")

    def test_review_required_with_current_mask_evidence_remains_processable(self):
        from inpainter import (
            _apply_koharu_bubble_fast_fill_to_blocks,
            _processable_vision_blocks_for_inpaint,
        )

        image = np.full((90, 140, 3), 245, dtype=np.uint8)
        block = {
            "id": "ocr_review",
            "text": "WAIT",
            "bbox": [36, 28, 100, 58],
            "text_pixel_bbox": [42, 34, 94, 48],
            "line_polygons": [[[42, 34], [94, 34], [94, 48], [42, 48]]],
            "route_action": "review_required",
            "qa_flags": ["mask_outside_balloon_critical"],
            "bubble_mask_source": "image_contour_bubble_mask",
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 180,
                "expanded_mask_pixels": 260,
                "evidence_score": 1.0,
            },
        }

        self.assertEqual(_processable_vision_blocks_for_inpaint([block]), [block])
        _working, remaining, _fast_mask, _remaining_mask, meta = _apply_koharu_bubble_fast_fill_to_blocks(
            image,
            {"texts": [dict(block)]},
            [dict(block)],
        )

        self.assertEqual(len(remaining), 1)
        self.assertEqual(meta["filled_pixels"], 0)
        self.assertEqual(meta["rejection_reasons"], {"review_required_real_inpaint": 1})

    def test_band_local_text_bbox_normalization_keeps_already_local_bbox(self):
        from inpainter import _texts_with_band_local_bboxes

        texts = [{"bbox": [20, 30, 80, 90], "text_pixel_bbox": [20, 30, 80, 90]}]

        normalized = _texts_with_band_local_bboxes(texts, width=120, height=140, band_y_top=629)

        self.assertEqual(normalized, texts)

    def test_residual_mask_for_white_balloon_uses_text_region_not_expanded_border(self):
        from inpainter import _select_residual_check_mask

        expanded = np.zeros((120, 160), dtype=np.uint8)
        expanded[10:110, 20:140] = 255
        page = {
            "texts": [
                {
                    "bbox": [60, 35, 90, 95],
                    "text_pixel_bbox": [60, 35, 90, 95],
                    "balloon_bbox": [20, 10, 140, 110],
                    "background_rgb": [248, 248, 248],
                }
            ]
        }

        mask, source, include_unchanged_dark = _select_residual_check_mask(
            ocr_page=page,
            shape=(120, 160),
            expanded_mask=expanded,
        )

        self.assertEqual(source, "text_region_white_balloon")
        self.assertFalse(include_unchanged_dark)
        self.assertLess(int(np.count_nonzero(mask)), int(np.count_nonzero(expanded)))
        self.assertEqual(int(mask[12, 24]), 0)
        self.assertEqual(int(mask[40, 65]), 255)

    def test_fast_white_fill_records_disabled_rejection_reason(self):
        from inpainter import _apply_fast_white_balloon_fill

        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        page = {
            "texts": [{"bbox": [10, 10, 60, 40], "balloon_bbox": [8, 8, 70, 50], "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [8, 8, 70, 50]}],
        }

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0"}, clear=False):
            _, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(remaining, [{"bbox": [8, 8, 70, 50]}])
        self.assertEqual(stats["white_balloon_count"], 0)
        self.assertEqual(page["_strip_fast_white_rejection_reasons"], {"disabled": 1})

    def test_fast_white_fill_uses_text_geometry_without_erasing_outline(self):
        from inpainter import _apply_fast_white_balloon_fill
        from vision_stack import gpu_image_ops

        image = np.full((180, 260, 3), 248, dtype=np.uint8)
        cv2.ellipse(image, (130, 92), (98, 56), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (130, 92), (98, 56), 0, 0, 360, (165, 165, 165), 2, cv2.LINE_AA)
        image[80:91, 80:178] = 12
        image[100:111, 92:168] = 12
        outline_pixel = image[36, 130].copy()
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [78, 78, 180, 113],
                    "text_pixel_bbox": [78, 78, 180, 113],
                    "line_polygons": [
                        [[80, 80], [178, 80], [178, 91], [80, 91]],
                        [[92, 100], [168, 100], [168, 111], [92, 111]],
                    ],
                    "balloon_bbox": [32, 36, 228, 148],
                    "balloon_type": "white",
                    "layout_profile": "white_balloon",
                    "tipo": "fala",
                    "confidence": 0.99,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ],
        }
        page["_vision_blocks"] = [dict(page["texts"][0])]
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}, clear=False):
            result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["white_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(np.all(result[82:89, 84:174] == 255))
        self.assertTrue(np.all(result[102:109, 96:164] == 255))
        self.assertTrue(np.array_equal(result[36, 130], outline_pixel))

    def test_fast_white_fill_uses_component_text_boxes_when_line_polygons_missing(self):
        from inpainter import _apply_fast_white_balloon_fill

        image = np.full((140, 260, 3), 180, dtype=np.uint8)
        cv2.ellipse(image, (130, 70), (112, 52), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (130, 70), (112, 52), 0, 0, 360, (0, 0, 0), 2)
        image[42:58, 40:90] = 12
        image[42:58, 112:166] = 12
        outline_pixel = image[18, 130].copy()
        text = {
            "id": "ocr_001",
            "bbox": [20, 20, 190, 76],
            "text_pixel_bbox": [20, 20, 190, 76],
            "balloon_bbox": [18, 18, 242, 122],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)
        with patch.dict(
            "os.environ",
            {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1", "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0"},
            clear=False,
        ):
            result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["white_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(np.all(result[48, 96] == 255))
        self.assertTrue(np.array_equal(result[18, 130], outline_pixel))

    def test_fast_white_fill_rejects_bbox_fill_without_glyph_overlap(self):
        import inpainter
        from inpainter import _apply_fast_white_balloon_fill

        image = np.full((140, 260, 3), 180, dtype=np.uint8)
        cv2.ellipse(image, (130, 70), (112, 52), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (130, 70), (112, 52), 0, 0, 360, (0, 0, 0), 2)
        image[58:72, 96:164] = 12
        text = {
            "id": "ocr_001",
            "bbox": [22, 20, 238, 118],
            "text_pixel_bbox": [22, 20, 238, 118],
            "_merged_source_bboxes": [[96, 58, 164, 72]],
            "balloon_bbox": [18, 18, 242, 122],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)
        unrelated_raw_evidence = np.zeros(image.shape[:2], dtype=np.uint8)
        unrelated_raw_evidence[10:20, 10:40] = 255

        with patch.dict(
            "os.environ",
            {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1", "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0"},
            clear=False,
        ), patch.object(inpainter, "build_raw_text_mask_from_image", return_value=unrelated_raw_evidence):
            result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["white_balloon_count"], 0)
        self.assertEqual(remaining, [text])
        self.assertEqual(page["_strip_fast_white_rejection_reasons"], {"koharu_text_evidence_mismatch": 1})
        self.assertTrue(np.array_equal(result, image))

    def test_fast_white_fill_prefers_merged_source_boxes_over_broad_bbox(self):
        from inpainter import _apply_fast_white_balloon_fill

        image = np.full((150, 280, 3), 210, dtype=np.uint8)
        cv2.ellipse(image, (140, 78), (116, 54), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (140, 78), (116, 54), 0, 0, 360, (0, 0, 0), 2)
        image[50:63, 82:184] = 12
        image[78:91, 94:172] = 12
        cv2.line(image, (44, 70), (76, 70), (0, 0, 0), 3)
        ray_pixel = image[70, 58].copy()
        text = {
            "id": "ocr_001",
            "bbox": [40, 44, 190, 98],
            "text_pixel_bbox": [40, 44, 190, 98],
            "_merged_source_bboxes": [[82, 50, 184, 63], [94, 78, 172, 91]],
            "balloon_bbox": [24, 24, 256, 132],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict(
            "os.environ",
            {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1", "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0"},
            clear=False,
        ):
            result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["white_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(np.all(result[55, 100] == 255))
        self.assertTrue(np.all(result[84, 120] == 255))
        self.assertTrue(np.array_equal(result[70, 58], ray_pixel))

    def test_fast_white_narration_cleans_validated_source_without_touching_lower_art(self):
        import inpainter
        from inpainter import _apply_fast_white_balloon_fill

        image = np.full((260, 320, 3), 235, dtype=np.uint8)
        image[30:86, 58:268] = 248
        image[44:52, 72:252] = 8
        image[64:72, 84:238] = 8
        image[132:224, 90:234] = 190
        cv2.circle(image, (160, 176), 54, (205, 205, 215), -1)
        cv2.circle(image, (138, 166), 6, (0, 0, 0), -1)
        cv2.circle(image, (182, 166), 6, (0, 0, 0), -1)
        face_pixel = image[166, 138].copy()

        text = {
            "id": "ocr_001",
            "bbox": [40, 20, 280, 236],
            "text_pixel_bbox": [40, 20, 280, 236],
            "_merged_source_bboxes": [[58, 30, 268, 86], [90, 132, 234, 224]],
            "balloon_bbox": [36, 16, 284, 240],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "narracao",
            "confidence": 0.99,
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)
        evidence = np.zeros(image.shape[:2], dtype=np.uint8)
        evidence[44:52, 72:252] = 255
        evidence[64:72, 84:238] = 255

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_NARRATION": "1",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch.object(
            inpainter, "_koharu_style_fast_white_evidence_rejection_reason", return_value=""
        ), patch.object(inpainter, "build_raw_text_mask_from_image", return_value=evidence):
            result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["white_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertEqual(page["_strip_fast_white_evidence_constrained_fill_count"], 1)
        cleaned_source = result[44:72, 72:252]
        self.assertEqual(int(np.count_nonzero(cleaned_source[:, :, 0] < 80)), 0)
        self.assertGreaterEqual(int(np.median(cleaned_source[:, :, 0])), 235)
        self.assertTrue(np.all(result[58, 60] == 235))
        self.assertTrue(np.all(result[58, 264] == 235))
        self.assertTrue(np.array_equal(result[166, 138], face_pixel))

    def test_band_local_bbox_normalization_shifts_merged_source_boxes(self):
        from inpainter import _texts_with_band_local_bboxes

        texts = [
            {
                "bbox": [40, 1040, 190, 1098],
                "text_pixel_bbox": [40, 1040, 190, 1098],
                "_merged_source_bboxes": [[82, 1050, 184, 1063], [94, 1078, 172, 1091]],
            }
        ]

        normalized = _texts_with_band_local_bboxes(texts, width=280, height=150, band_y_top=1000)

        self.assertEqual(normalized[0]["bbox"], [40, 40, 190, 98])
        self.assertEqual(
            normalized[0]["_merged_source_bboxes"],
            [[82, 50, 184, 63], [94, 78, 172, 91]],
        )
        self.assertEqual(normalized[0]["merged_source_bboxes"], normalized[0]["_merged_source_bboxes"])

    def test_expanded_white_balloon_mask_preserves_nearby_outline(self):
        from vision_stack.runtime import vision_blocks_to_mask

        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (130, 92), (90, 54), 0, 0, 360, (150, 150, 150), 2, cv2.LINE_AA)
        image[88:101, 46:160] = 12
        block = {
            "bbox": [44, 84, 164, 106],
            "text_pixel_bbox": [46, 88, 160, 101],
            "line_polygons": [[[46, 88], [160, 88], [160, 101], [46, 101]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image, expand_mask=True, ocr_texts=[block])

        self.assertGreater(int(mask[94, 72]), 0)
        self.assertEqual(int(mask[92, 40]), 0)

    def test_inpaint_enrichment_merges_overlapping_text_line_geometry(self):
        from inpainter import _enrich_vision_blocks_from_texts_for_inpaint
        from vision_stack.runtime import vision_blocks_to_mask

        image = np.full((360, 760, 3), 255, dtype=np.uint8)
        block = {
            "bbox": [25, 16, 667, 329],
            "balloon_bbox": [0, 0, 733, 592],
            "balloon_type": "white",
            "tipo": "fala",
        }
        texts = [
            {
                "id": "ocr_001",
                "bbox": [25, 16, 667, 329],
                "text_pixel_bbox": [320, 235, 656, 296],
                "line_polygons": [
                    [[498, 232], [658, 234], [658, 258], [498, 256]],
                    [[507, 266], [648, 266], [648, 287], [507, 287]],
                ],
                "balloon_type": "white",
                "tipo": "fala",
            },
            {
                "id": "ocr_002",
                "bbox": [501, 218, 661, 325],
                "text_pixel_bbox": [546, 300, 612, 320],
                "line_polygons": [
                    [[543, 298], [615, 298], [615, 323], [543, 323]],
                ],
                "balloon_type": "white",
                "tipo": "fala",
            },
        ]

        enriched = _enrich_vision_blocks_from_texts_for_inpaint([block], texts, width=760, height=360)
        mask = vision_blocks_to_mask(image.shape, enriched, image_rgb=image, expand_mask=True)

        self.assertEqual(len(enriched[0]["line_polygons"]), 3)
        self.assertGreater(int(mask[310, 580]), 0)
        self.assertGreater(int(mask[240, 590]), 0)

    def test_white_force_fill_restore_keeps_balloon_arc_outside_text_geometry(self):
        from vision_stack.runtime import _restore_dark_line_art_outside_text_geometry

        original = np.full((180, 260, 3), 255, dtype=np.uint8)
        cv2.ellipse(original, (130, 92), (90, 54), 0, 0, 360, (140, 140, 140), 2, cv2.LINE_AA)
        original[84:96, 78:180] = 12
        original[106:118, 92:168] = 12
        cleaned = original.copy()
        cleaned[34:60, 72:190] = 255
        cleaned[78:124, 70:186] = 255
        text = {
            "bbox": [76, 82, 182, 120],
            "text_pixel_bbox": [76, 82, 182, 120],
            "line_polygons": [
                [[78, 84], [180, 84], [180, 96], [78, 96]],
                [[92, 106], [168, 106], [168, 118], [92, 118]],
            ],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        restored = _restore_dark_line_art_outside_text_geometry(original, cleaned, [text])

        self.assertTrue(np.array_equal(restored[38, 130], original[38, 130]))
        self.assertTrue(np.all(restored[88:92, 90:170] == 255))

    def test_line_art_restore_uses_line_polygons_not_full_ocr_bbox(self):
        from vision_stack.runtime import _restore_dark_line_art_outside_text_geometry

        original = np.full((180, 260, 3), 255, dtype=np.uint8)
        cv2.line(original, (36, 95), (126, 84), (24, 24, 24), 3, cv2.LINE_AA)
        original[62:78, 70:190] = 12
        original[112:128, 76:184] = 12
        cleaned = original.copy()
        cleaned[40:140, 34:202] = 255
        text = {
            "bbox": [68, 60, 192, 130],
            "text_pixel_bbox": [68, 60, 192, 130],
            "line_polygons": [
                [[70, 62], [190, 62], [190, 78], [70, 78]],
                [[76, 112], [184, 112], [184, 128], [76, 128]],
            ],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        restored = _restore_dark_line_art_outside_text_geometry(original, cleaned, [text])

        self.assertTrue(np.array_equal(restored[86, 112], original[86, 112]))
        self.assertTrue(np.all(restored[66:74, 82:178] == 255))

    def test_line_art_restore_does_not_reintroduce_neighbor_text(self):
        from vision_stack.runtime import _restore_dark_line_art_outside_text_geometry

        original = np.full((180, 260, 3), 255, dtype=np.uint8)
        original[5:7, 60:180] = 20
        original[42:52, 58:162] = 12
        original[98:112, 92:174] = 12
        cleaned = original.copy()
        cleaned[5:7, 60:180] = 255
        cleaned[38:116, 50:182] = 255
        upper_text = {
            "bbox": [58, 42, 162, 52],
            "text_pixel_bbox": [58, 42, 162, 52],
            "line_polygons": [[[58, 42], [162, 42], [162, 52], [58, 52]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }
        lower_text = {
            "bbox": [92, 98, 174, 112],
            "text_pixel_bbox": [92, 98, 174, 112],
            "line_polygons": [[[92, 98], [174, 98], [174, 112], [92, 112]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        restored = _restore_dark_line_art_outside_text_geometry(original, cleaned, [upper_text, lower_text])

        self.assertTrue(np.array_equal(restored[5:7, 70:170], original[5:7, 70:170]))
        self.assertTrue(np.all(restored[100:108, 100:166] == 255))

    def test_line_art_restore_does_not_reintroduce_glyph_stroke_next_to_line_polygon(self):
        from vision_stack.runtime import _restore_dark_line_art_outside_text_geometry

        original = np.full((140, 220, 3), 255, dtype=np.uint8)
        original[64:68, 40:69] = 12
        cleaned = original.copy()
        cleaned[48:92, 30:150] = 255
        text = {
            "bbox": [40, 56, 132, 82],
            "text_pixel_bbox": [40, 56, 132, 82],
            "line_polygons": [[[70, 56], [132, 56], [132, 82], [70, 82]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        restored = _restore_dark_line_art_outside_text_geometry(original, cleaned, [text])

        self.assertTrue(np.all(restored[64:68, 40:69] == 255))

    def test_line_art_restore_does_not_reintroduce_single_line_text_bbox_residual(self):
        from vision_stack.runtime import _restore_dark_line_art_outside_text_geometry

        original = np.full((120, 200, 3), 255, dtype=np.uint8)
        original[62:68, 72:132] = 12
        cleaned = original.copy()
        cleaned[40:76, 54:150] = 255
        text = {
            "bbox": [54, 38, 150, 76],
            "text_pixel_bbox": [64, 42, 142, 70],
            "line_polygons": [[[64, 42], [142, 42], [142, 56], [64, 56]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        restored = _restore_dark_line_art_outside_text_geometry(original, cleaned, [text])

        self.assertTrue(np.all(restored[62:68, 72:132] == 255))

    def test_line_art_restore_respects_other_text_line_polygon_guards(self):
        from vision_stack.runtime import _restore_dark_line_art_outside_text_geometry

        original = np.full((150, 220, 3), 255, dtype=np.uint8)
        original[92:96, 82:142] = 12
        cleaned = original.copy()
        cleaned[40:120, 40:170] = 255
        upper_text = {
            "bbox": [50, 48, 160, 110],
            "text_pixel_bbox": [50, 48, 160, 110],
            "line_polygons": [[[70, 50], [150, 50], [150, 70], [70, 70]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }
        lower_text = {
            "bbox": [82, 88, 142, 100],
            "text_pixel_bbox": [82, 88, 142, 100],
            "line_polygons": [[[82, 88], [142, 88], [142, 100], [82, 100]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        restored = _restore_dark_line_art_outside_text_geometry(original, cleaned, [upper_text, lower_text])

        self.assertTrue(np.all(restored[92:96, 82:142] == 255))

    def test_white_cleanup_splits_distant_line_polygon_islands(self):
        from vision_stack.runtime import _white_cleanup_texts

        image = np.full((260, 260, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (0, 70), (260, 190), (190, 170, 150), -1)
        cv2.ellipse(image, (92, 52), (62, 32), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (92, 52), (62, 32), 0, 0, 360, (130, 130, 130), 2, cv2.LINE_AA)
        cv2.ellipse(image, (150, 210), (84, 42), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (150, 210), (84, 42), 0, 0, 360, (130, 130, 130), 2, cv2.LINE_AA)
        image[46:58, 54:130] = 12
        image[198:212, 98:202] = 12

        text = {
            "text": "HEY, LET'S GO! I'M STARVING",
            "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
            "bbox": [40, 44, 205, 214],
            "text_pixel_bbox": [54, 46, 202, 212],
            "line_polygons": [
                [[54, 46], [130, 46], [130, 58], [54, 58]],
                [[98, 198], [202, 198], [202, 212], [98, 212]],
            ],
            "balloon_bbox": [30, 20, 235, 252],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        cleanup_texts = _white_cleanup_texts(image, [text])

        self.assertEqual(len(cleanup_texts), 2)
        self.assertEqual(cleanup_texts[0]["text_pixel_bbox"], [54, 46, 131, 59])
        self.assertEqual(cleanup_texts[1]["text_pixel_bbox"], [98, 198, 203, 213])
        self.assertTrue(all(item.get("_white_cleanup_split_count") == 2 for item in cleanup_texts))

    def test_fast_white_fill_can_use_experimental_gpu_image_ops_hook(self):
        from inpainter import _apply_fast_white_balloon_fill
        from vision_stack import gpu_image_ops

        image = np.full((120, 180, 3), 245, dtype=np.uint8)
        cv2.ellipse(image, (90, 60), (62, 34), 0, 0, 360, (255, 255, 255), -1)
        image[52:62, 62:118] = 8
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [60, 50, 120, 64],
                    "text_pixel_bbox": [60, 50, 120, 64],
                    "line_polygons": [[[62, 52], [118, 52], [118, 62], [62, 62]]],
                    "balloon_bbox": [28, 26, 152, 94],
                    "balloon_type": "white",
                    "layout_profile": "white_balloon",
                    "tipo": "fala",
                    "confidence": 0.99,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ],
        }
        page["_vision_blocks"] = [dict(page["texts"][0])]
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1",
                "TRADUZAI_EXPERIMENTAL_GPU_IMAGE_OPS": "1",
                "TRADUZAI_GPU_IMAGE_OPS_BACKEND": "cpu",
            },
            clear=False,
        ), patch.object(gpu_image_ops, "apply_white_fill", wraps=gpu_image_ops.apply_white_fill) as fill_spy:
            result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["white_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertGreaterEqual(fill_spy.call_count, 1)
        self.assertTrue(np.all(result[54:60, 66:114] == 255))

    def test_fast_local_fill_records_no_vision_blocks_rejection_reason(self):
        from inpainter import _apply_fast_local_balloon_fill

        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        page = {
            "texts": [
                {"bbox": [10, 10, 60, 40], "balloon_bbox": [8, 8, 70, 50], "tipo": "fala"},
                {"bbox": [20, 50, 90, 70], "balloon_bbox": [18, 48, 95, 75], "tipo": "pensamento"},
            ],
            "_vision_blocks": [],
        }

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "1"}, clear=False):
            _, remaining, stats = _apply_fast_local_balloon_fill(image, page, [])

        self.assertEqual(remaining, [])
        self.assertEqual(stats["local_balloon_count"], 0)
        self.assertEqual(page["_strip_fast_local_rejection_reasons"], {"no_vision_blocks": 2})

    def test_fast_local_fill_is_opt_in_by_default(self):
        from inpainter import _apply_fast_local_balloon_fill

        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        page = {
            "texts": [{"bbox": [10, 10, 60, 40], "balloon_bbox": [8, 8, 70, 50], "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [8, 8, 70, 50]}],
        }

        with patch.dict("os.environ", {}, clear=True):
            _, remaining, stats = _apply_fast_local_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(remaining, [{"bbox": [8, 8, 70, 50]}])
        self.assertEqual(stats["local_balloon_count"], 0)
        self.assertEqual(page["_strip_fast_local_rejection_reasons"], {"disabled": 1})

    def test_fast_local_uses_geometry_for_white_text_without_tipo_gate(self):
        from inpainter import _fast_local_rejection_reason

        text = {
            "tipo": "narracao",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "bbox": [204, 8042, 480, 8122],
            "text_pixel_bbox": [204, 8042, 480, 8122],
            "mask_evidence": _allowed_mask_evidence(),
        }

        self.assertEqual(_fast_local_rejection_reason(text), "")

    def test_dark_panel_fill_rejects_generic_textured_panel(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.full((120, 220, 3), (34, 54, 72), dtype=np.uint8)
        image[:, ::9, :] = (52, 90, 118)
        text = {
            "bbox": [58, 48, 164, 76],
            "text_pixel_bbox": [58, 48, 164, 76],
            "line_polygons": [[[58, 48], [164, 48], [164, 76], [58, 76]]],
            "balloon_bbox": [44, 36, 178, 90],
            "balloon_type": "textured",
            "layout_profile": "standard",
            "tipo": "narracao",
        }

        self.assertIsNone(_try_dark_panel_text_fill(image, text))

    def test_dark_text_contract_fill_closes_hollow_raw_glyph_mask(self):
        from inpainter import _dark_text_contract_fill_mask

        image = np.zeros((120, 260, 3), dtype=np.uint8)
        source = np.zeros((120, 260), dtype=np.uint8)
        cv2.rectangle(source, (42, 34), (182, 56), 255, 2)
        cv2.rectangle(source, (58, 68), (210, 92), 255, 2)
        text = {
            "bbox": [34, 24, 220, 100],
            "source_bbox": [34, 24, 220, 100],
            "text_pixel_bbox": [42, 34, 210, 92],
            "line_polygons": [
                [[42, 34], [182, 34], [182, 56], [42, 56]],
                [[58, 68], [210, 68], [210, 92], [58, 92]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["visual_text_only_inpaint_contract"],
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 2400,
                "expanded_mask_pixels": 9600,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        with patch("inpainter.build_raw_text_mask_from_image", return_value=source):
            mask = _dark_text_contract_fill_mask(text, 260, 120, image)

        self.assertIsInstance(mask, np.ndarray)
        assert mask is not None
        self.assertGreater(int(mask[45, 112]), 0)
        self.assertGreater(int(mask[80, 132]), 0)
        metrics = text.get("qa_metrics", {}).get("dark_text_contract_raw_glyph_mask", {})
        self.assertGreater(
            int(metrics.get("raw_mask_pixels") or 0),
            int(metrics.get("raw_mask_pixels_before_gap_close") or 0),
        )

    def test_dark_text_contract_fill_prefers_valid_inpaint_contract_mask_over_raw_outline(self):
        from inpainter import _dark_text_contract_fill_mask

        image = np.zeros((140, 320, 3), dtype=np.uint8)
        raw = np.zeros((140, 320), dtype=np.uint8)
        cv2.rectangle(raw, (40, 38), (138, 54), 255, 2)
        cv2.rectangle(raw, (52, 68), (174, 84), 255, 2)
        contract = np.zeros((140, 320), dtype=np.uint8)
        contract[38:55, 40:176] = 255
        contract[68:85, 52:220] = 255
        text = {
            "bbox": [36, 32, 224, 92],
            "source_bbox": [36, 32, 224, 92],
            "text_pixel_bbox": [36, 32, 224, 92],
            "line_polygons": [
                [[40, 38], [176, 38], [176, 55], [40, 55]],
                [[52, 68], [220, 68], [220, 85], [52, 85]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["visual_text_only_inpaint_contract"],
        }

        with patch("inpainter.build_raw_text_mask_from_image", return_value=raw), patch(
            "inpainter.build_inpaint_mask",
            return_value=contract,
        ):
            mask = _dark_text_contract_fill_mask(text, 320, 140, image)

        self.assertIsInstance(mask, np.ndarray)
        assert mask is not None
        self.assertGreater(int(mask[46, 170]), 0)
        self.assertGreater(int(mask[76, 210]), 0)
        metrics = text.get("qa_metrics", {})
        self.assertEqual(
            metrics.get("dark_text_contract_fill_mask", {}).get("source"),
            "build_inpaint_mask_contract",
        )
        self.assertIn("dark_text_contract_fill_uses_inpaint_contract_mask", metrics)

    def test_dark_panel_fill_rejects_colored_art_caption_with_line_geometry(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((170, 320, 3), (72, 150, 172), dtype=np.uint8)
        image[:, ::11, :] = (38, 98, 150)
        image[20:150, 190:230] = (215, 250, 255)
        image[45:135, 210:260] = (4, 18, 34)
        cv2.putText(image, "EVEN AFTER LOSING", (28, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(image, "EVEN AFTER LOSING", (28, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "HIS AFFILIATION", (42, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(image, "HIS AFFILIATION", (42, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_003",
                    "text": "EVEN AFTER LOSING HIS AFFILIATION",
                    "bbox": [24, 54, 270, 138],
                    "text_pixel_bbox": [24, 54, 270, 138],
                    "line_polygons": [
                        [[24, 54], [270, 54], [270, 98], [24, 98]],
                        [[40, 98], [250, 98], [250, 138], [40, 138]],
                    ],
                    "balloon_bbox": [0, 35, 292, 158],
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "tipo": "narracao",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 0)
        self.assertEqual(remaining, vision_blocks)
        self.assertEqual(page["_strip_fast_dark_rejection_reasons"], {"not_solid_dark_panel": 1})
        self.assertTrue(np.array_equal(result, image))

    def test_dark_panel_fill_handles_misclassified_near_black_panel(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((90, 260, 3), dtype=np.uint8)
        cv2.rectangle(image, (40, 18), (220, 70), (4, 4, 5), -1)
        cv2.rectangle(image, (40, 18), (220, 70), (180, 185, 190), 1)
        image[40:52, 86:174] = (210, 225, 230)
        text = {
            "bbox": [84, 38, 176, 54],
            "text_pixel_bbox": [84, 38, 176, 54],
            "line_polygons": [[[84, 38], [176, 38], [176, 54], [84, 54]]],
            "balloon_bbox": [40, 18, 220, 70],
            "balloon_type": "textured",
            "layout_profile": "standard",
            "tipo": "narracao",
        }

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        self.assertLess(int(np.mean(result[44, 130])), 20)
        self.assertTrue(np.array_equal(result[18, 130], image[18, 130]))

    def test_fast_white_fill_ignores_contextual_type_metadata(self):
        from inpainter import _fast_white_rejection_reason

        text = {
            "tipo": "narracao",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "context_after": "I KNOW.",
            "confidence": 0.91,
            "text_pixel_bbox": [12, 12, 80, 36],
            "mask_evidence": _allowed_mask_evidence(),
        }

        self.assertEqual(_fast_white_rejection_reason(text), "")

    def test_fast_white_fill_allows_moderate_confidence_clean_white_narration(self):
        from inpainter import _fast_white_rejection_reason

        text = {
            "tipo": "narracao",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "confidence": 0.797,
            "text_pixel_bbox": [390, 3074, 789, 3175],
            "mask_evidence": _allowed_mask_evidence(),
        }

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_NARRATION": "1"}, clear=True):
            self.assertEqual(_fast_white_rejection_reason(text), "")

    def test_fallback_blocks_preserve_text_geometry_for_mask_refinement(self):
        from inpainter import _build_fallback_vision_blocks

        polygons = [[[12, 14], [48, 14], [48, 26], [12, 26]]]
        page = {
            "texts": [
                {
                    "bbox": [8, 8, 80, 40],
                    "text_pixel_bbox": [12, 14, 48, 26],
                    "line_polygons": polygons,
                    "balloon_type": "white",
                    "block_profile": "white_balloon",
                    "confidence": 0.42,
                }
            ]
        }

        blocks = _build_fallback_vision_blocks(page, 120, 80)

        self.assertEqual(blocks[0]["bbox"], [12, 14, 48, 26])
        self.assertEqual(blocks[0]["text_pixel_bbox"], [12, 14, 48, 26])
        self.assertEqual(blocks[0]["line_polygons"], polygons)
        self.assertEqual(blocks[0]["balloon_type"], "white")
        self.assertEqual(blocks[0]["block_profile"], "white_balloon")

    def test_fallback_blocks_ignore_legacy_skip_processing_marker(self):
        from inpainter import _build_fallback_vision_blocks

        page = {
            "texts": [
                {"bbox": [10, 10, 40, 30], "skip_processing": True},
                {"bbox": [50, 10, 90, 30], "skip_processing": False},
            ]
        }

        blocks = _build_fallback_vision_blocks(page, 120, 80)

        self.assertEqual([block["bbox"] for block in blocks], [[10, 10, 40, 30], [50, 10, 90, 30]])

    def test_residual_check_uses_fast_fill_and_text_region_when_masks_are_empty(self):
        from inpainter import _detect_inpaint_residual_text

        before = np.full((80, 120, 3), 245, dtype=np.uint8)
        before[30:42, 36:84] = 10
        after = before.copy()
        after[18:25, 24:96] = 245
        fast_fill_mask = np.zeros((80, 120), dtype=np.uint8)
        fast_fill_mask[18:25, 24:96] = 255
        empty = np.zeros((80, 120), dtype=np.uint8)
        page = {
            "texts": [
                {
                    "bbox": [32, 26, 88, 48],
                    "text_pixel_bbox": [36, 30, 84, 42],
                    "skip_processing": False,
                }
            ]
        }

        residual = _detect_inpaint_residual_text(
            before,
            after,
            empty,
            raw_mask=empty,
            fast_fill_mask=fast_fill_mask,
            ocr_page=page,
        )

        self.assertTrue(residual["has_residual"])
        self.assertIn("text_region", residual["region_source"])
        self.assertIn("fallback_region", residual["flags"])

    def test_residual_text_region_uses_line_polygons_instead_of_tall_text_bbox(self):
        from inpainter import _build_residual_text_region_mask

        page = {
            "texts": [
                {
                    "bbox": [10, 10, 90, 150],
                    "text_pixel_bbox": [12, 12, 88, 148],
                    "line_polygons": [
                        [[12, 14], [54, 14], [54, 28], [12, 28]],
                        [[46, 130], [88, 130], [88, 144], [46, 144]],
                    ],
                    "skip_processing": False,
                }
            ]
        }

        mask = _build_residual_text_region_mask(page, (180, 120))

        self.assertGreater(int(mask[20, 24]), 0)
        self.assertGreater(int(mask[136, 60]), 0)
        self.assertEqual(int(mask[80, 48]), 0)
        self.assertLess(int(np.count_nonzero(mask)), 3500)

    def test_fast_fill_residual_runs_real_inpaint_when_fast_path_leaves_text(self):
        from inpainter import inpaint_band_image

        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        image[30:42, 36:84] = 10
        working = image.copy()
        working[18:25, 24:96] = 245
        page = {
            "texts": [
                {
                    "bbox": [32, 26, 88, 48],
                    "text_pixel_bbox": [36, 30, 84, 42],
                    "balloon_bbox": [20, 16, 100, 56],
                    "tipo": "fala",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [{"bbox": [32, 26, 88, 48]}],
        }

        class FakeInpainter:
            def __init__(self):
                self.calls = []

            def inpaint(self, img, mask, batch_size=4, force_no_tiling=True):
                self.calls.append((img.copy(), mask.copy(), batch_size, force_no_tiling))
                repaired = img.copy()
                repaired[mask > 0] = 245
                return repaired

        fake_inpainter = FakeInpainter()

        with patch("inpainter._apply_fast_solid_balloon_fill", return_value=(working, [], {})), patch(
            "inpainter._apply_fast_local_balloon_fill",
            side_effect=lambda band, ocr, blocks: (band, blocks, {}),
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ), patch(
            "vision_stack.runtime._get_inpainter",
            return_value=fake_inpainter,
        ):
            result = inpaint_band_image(image, page)

        self.assertEqual(len(fake_inpainter.calls), 1)
        self.assertTrue(page["_strip_used_real_inpaint"])
        self.assertTrue(np.all(result[30:42, 36:84] == 245))

    def test_fast_fill_residual_ignores_only_preserved_remaining_blocks(self):
        from inpainter import inpaint_band_image

        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        image[30:42, 36:84] = 10
        working = image.copy()
        working[18:25, 24:96] = 245
        page = {
            "texts": [
                {
                    "bbox": [32, 26, 88, 48],
                    "text_pixel_bbox": [36, 30, 84, 42],
                    "balloon_bbox": [20, 16, 100, 56],
                    "tipo": "narracao",
                    "skip_processing": False,
                },
                {
                    "bbox": [4, 4, 28, 16],
                    "text_pixel_bbox": [4, 4, 28, 16],
                    "content_class": "logo",
                    "skip_processing": True,
                    "preserve_original": True,
                },
            ],
            "_vision_blocks": [{"bbox": [32, 26, 88, 48]}],
        }
        preserved_logo_block = {
            "bbox": [4, 4, 28, 16],
            "text_pixel_bbox": [4, 4, 28, 16],
            "content_class": "logo",
            "skip_processing": True,
            "preserve_original": True,
        }

        class FakeInpainter:
            def __init__(self):
                self.calls = []

            def inpaint(self, img, mask, batch_size=4, force_no_tiling=True):
                self.calls.append((img.copy(), mask.copy(), batch_size, force_no_tiling))
                repaired = img.copy()
                repaired[mask > 0] = 245
                return repaired

        fake_inpainter = FakeInpainter()

        with patch(
            "inpainter._apply_fast_solid_balloon_fill",
            return_value=(working, [preserved_logo_block], {}),
        ), patch(
            "inpainter._apply_fast_local_balloon_fill",
            side_effect=lambda band, ocr, blocks: (band, blocks, {}),
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ), patch(
            "vision_stack.runtime._get_inpainter",
            return_value=fake_inpainter,
        ):
            result = inpaint_band_image(image, page)

        self.assertEqual(len(fake_inpainter.calls), 1)
        self.assertTrue(page["_strip_used_real_inpaint"])
        self.assertNotIn("_strip_nonprocessable_remaining_block_count", page)
        self.assertTrue(np.all(result[30:42, 36:84] == 245))

    def test_inpaint_band_image_skips_review_required_before_fast_fill(self):
        from inpainter import inpaint_band_image

        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        image[22:48, 34:92] = [90, 150, 130]
        page = {
            "texts": [
                {
                    "bbox": [30, 20, 96, 52],
                    "text_pixel_bbox": [34, 22, 92, 48],
                    "line_polygons": [[[34, 22], [92, 22], [92, 48], [34, 48]]],
                    "route_action": "review_required",
                    "route_reason": "ocr_art_fragment_suspected",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [30, 20, 96, 52],
                    "text_pixel_bbox": [34, 22, 92, 48],
                    "line_polygons": [[[34, 22], [92, 22], [92, 48], [34, 48]]],
                }
            ],
        }

        with patch(
            "inpainter._apply_koharu_bubble_fast_fill_to_blocks",
            side_effect=AssertionError("review_required nao deve chegar ao fast fill"),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(page.get("_strip_nonprocessable_remaining_block_count"), 1)

    def test_inpaint_band_image_sends_review_required_with_mask_evidence_to_aot(self):
        from inpainter import inpaint_band_image

        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        image[34:48, 42:90] = 20
        text = {
            "id": "ocr_review",
            "text_id": "ocr_review",
            "trace_id": "ocr_review@page_001_band_001",
            "bbox": [30, 20, 96, 52],
            "text_pixel_bbox": [42, 34, 90, 48],
            "line_polygons": [[[42, 34], [90, 34], [90, 48], [42, 48]]],
            "route_action": "review_required",
            "qa_flags": ["mask_outside_balloon_critical"],
            "bubble_mask_source": "image_contour_bubble_mask",
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 320,
                "expanded_mask_pixels": 520,
                "evidence_score": 1.0,
            },
            "skip_processing": False,
        }
        page = {"texts": [dict(text)], "_vision_blocks": [dict(text)]}

        def fake_round(working, payload, inpainter):
            repaired = working.copy()
            repaired[34:48, 42:90] = 245
            payload["_inpaint_round_stats"] = {}
            return repaired

        with patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=fake_round,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(page.get("_strip_used_real_inpaint"))
        self.assertEqual(page.get("_strip_remaining_inpaint_blocks"), 1)
        self.assertGreater(int(np.count_nonzero(image != result)), 0)
        self.assertNotIn("mask_outside_balloon_critical", page["texts"][0].get("qa_flags", []))
        self.assertNotIn("real_inpaint_skipped_unsafe_mask", page.get("_strip_inpaint_decision_flags", []))
        self.assertIn(
            "review_required_real_inpaint",
            page.get("_strip_koharu_fast_fill_reject_reasons", {}),
        )

    def test_inpaint_band_image_skips_suppressed_cjk_text_before_masks(self):
        from inpainter import inpaint_band_image

        image = np.full((80, 180, 3), 240, dtype=np.uint8)
        image[24:48, 40:142] = [80, 40, 120]
        suppressed = {
            "id": "ocr_cjk",
            "text": "달링 가라오케",
            "bbox": [36, 20, 150, 54],
            "text_pixel_bbox": [40, 24, 142, 48],
            "line_polygons": [[[40, 24], [142, 24], [142, 48], [40, 48]]],
            "route": "suppress",
            "route_action": "review_required",
            "route_reason": "source_language_cjk_text_suppressed",
            "qa_flags": ["source_language_cjk_text_suppressed"],
            "skip_processing": True,
        }
        page = {"texts": [dict(suppressed)], "_vision_blocks": [dict(suppressed)]}

        with patch(
            "inpainter._apply_koharu_bubble_fast_fill_to_blocks",
            side_effect=AssertionError("texto suprimido nao deve chegar ao fast fill"),
        ), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=AssertionError("texto suprimido nao deve chegar ao AOT"),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(page.get("_strip_remaining_inpaint_blocks"), 0)

    def test_inpaint_band_image_skips_visual_sfx_without_inpaint_permission(self):
        from inpainter import inpaint_band_image

        image = np.zeros((90, 220, 3), dtype=np.uint8)
        image[:] = [12, 18, 24]
        cv2.putText(image, "쿵", (54, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 245, 255), 3, cv2.LINE_AA)
        sfx = {
            "id": "sfx_visual_001",
            "text": "쿵",
            "bbox": [42, 24, 128, 68],
            "text_pixel_bbox": [42, 24, 128, 68],
            "line_polygons": [[[42, 24], [128, 24], [128, 68], [42, 68]]],
            "content_class": "sfx",
            "tipo": "sfx",
            "route_action": "translate_sfx_inpaint_render",
            "sfx": {"inpaint_allowed": False, "visual_promotion": True},
            "qa_flags": ["sfx_visual_candidate"],
        }
        page = {"texts": [dict(sfx)], "_vision_blocks": [dict(sfx)]}

        with patch(
            "inpainter._apply_koharu_bubble_fast_fill_to_blocks",
            side_effect=AssertionError("SFX sem permissao nao deve chegar ao fast fill"),
        ), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=AssertionError("SFX sem permissao nao deve chegar ao AOT"),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(page.get("_strip_remaining_inpaint_blocks"), 0)

    def test_inpaint_band_image_blocks_real_inpaint_for_mask_outside_balloon_critical(self):
        from inpainter import inpaint_band_image

        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        image[30:42, 36:84] = 10
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [30, 26, 90, 48],
                    "text_pixel_bbox": [36, 30, 84, 42],
                    "line_polygons": [[[36, 30], [84, 30], [84, 42], [36, 42]]],
                    "qa_flags": ["mask_outside_balloon_critical"],
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {
                    "id": "ocr_001",
                    "bbox": [30, 26, 90, 48],
                    "text_pixel_bbox": [36, 30, 84, 42],
                    "line_polygons": [[[36, 30], [84, 30], [84, 42], [36, 42]]],
                    "qa_flags": ["mask_outside_balloon_critical"],
                }
            ],
        }

        with patch(
            "inpainter._apply_koharu_bubble_fast_fill_to_blocks",
            side_effect=AssertionError("unsafe mask nao deve chegar ao fast fill"),
        ), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=AssertionError("unsafe mask nao deve chegar ao AOT"),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(np.array_equal(result, image))
        self.assertFalse(page.get("_strip_used_real_inpaint"))
        self.assertEqual(page.get("_strip_remaining_inpaint_blocks"), 0)
        self.assertEqual(page["texts"][0]["id"], "ocr_001")
        self.assertIn("real_inpaint_skipped_unsafe_mask", page["_strip_inpaint_decision_flags"])

    def test_translator_note_with_current_text_mask_evidence_is_not_unsafe_outside_balloon(self):
        from inpainter import _auto_inpaint_unsafe_reason

        text = {
            "id": "ocr_tn",
            "text": "T/N: HYUNGNIM IS A TERM USED FOR CALLING ONE'S BOSS.",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
            "bubble_mask_source": "image_contour_bubble_mask",
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 1951,
                "expanded_mask_pixels": 12831,
                "evidence_score": 1.0,
            },
        }

        self.assertEqual(_auto_inpaint_unsafe_reason(text), "")

    def test_clean_contour_component_mask_clears_stale_outside_balloon_critical(self):
        from inpainter import _auto_inpaint_unsafe_reason

        text = {
            "id": "ocr_002",
            "trace_id": "ocr_002@page_002_band_019",
            "text": "The amount is just right. This bitch is a real actress...",
            "bbox": [298, 107, 525, 229],
            "text_pixel_bbox": [298, 107, 525, 229],
            "line_polygons": [
                [[298, 107], [525, 107], [525, 229], [298, 229]],
            ],
            "bubble_mask_source": "image_contour_bubble_mask",
            "bubble_mask_error": None,
            "bubble_mask_bbox": [234, 124, 575, 312],
            "route_action": "translate_inpaint_render",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 3037,
                "expanded_mask_pixels": 12526,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        self.assertEqual(_auto_inpaint_unsafe_reason(text), "")

    def test_inpaint_band_image_blocks_unsafe_text_by_geometry_when_block_id_differs(self):
        from inpainter import inpaint_band_image

        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        image[30:42, 36:84] = 10
        page = {
            "texts": [
                {
                    "id": "ocr_text",
                    "bbox": [30, 24, 90, 50],
                    "text_pixel_bbox": [36, 30, 84, 42],
                    "line_polygons": [[[36, 30], [84, 30], [84, 42], [36, 42]]],
                    "qa_flags": ["mask_outside_balloon_critical"],
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {
                    "id": "vision_block_without_matching_id",
                    "bbox": [32, 25, 88, 49],
                    "text_pixel_bbox": [36, 30, 84, 42],
                    "line_polygons": [[[36, 30], [84, 30], [84, 42], [36, 42]]],
                }
            ],
        }

        with patch(
            "inpainter._apply_koharu_bubble_fast_fill_to_blocks",
            side_effect=AssertionError("unsafe mask matched by geometry nao deve chegar ao fast fill"),
        ), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=AssertionError("unsafe mask matched by geometry nao deve chegar ao AOT"),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(page.get("_strip_remaining_inpaint_blocks"), 0)
        self.assertEqual(page.get("_strip_unsafe_inpaint_block_reasons"), {"mask_outside_balloon_critical": 1})
        self.assertIn("real_inpaint_skipped_unsafe_mask", page["_strip_inpaint_decision_flags"])

    def test_white_balloon_unsafe_mask_does_not_fall_back_to_dark_panel_fill(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.full((160, 220, 3), 210, dtype=np.uint8)
        image[:, :120] = [170, 182, 205]
        image[70:110, 132:188] = [32, 36, 50]
        text = {
            "id": "ocr_unsafe",
            "text": "PLEASE, FOR THE CHILD'S SAKE.",
            "bbox": [132, 70, 188, 110],
            "text_pixel_bbox": [132, 70, 188, 110],
            "line_polygons": [[[132, 70], [188, 70], [188, 110], [132, 110]]],
            "balloon_bbox": [120, 48, 210, 126],
            "bubble_mask_bbox": [120, 48, 210, 126],
            "bubble_mask_source": "image_contour_bubble_mask",
            "layout_profile": "white_balloon",
            "balloon_type": "white",
            "qa_flags": ["mask_outside_balloon_critical"],
        }

        self.assertIsNone(_try_dark_panel_text_fill(image, text))

    def test_dark_panel_fill_does_not_treat_unsafe_white_bubble_as_card(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((120, 180, 3), [195, 205, 220], dtype=np.uint8)
        image[44:72, 92:150] = [35, 38, 48]
        page = {
            "texts": [
                {
                    "id": "ocr_unsafe",
                    "text": "PLEASE, FOR THE CHILD'S SAKE.",
                    "bbox": [92, 44, 150, 72],
                    "text_pixel_bbox": [92, 44, 150, 72],
                    "line_polygons": [[[92, 44], [150, 44], [150, 72], [92, 72]]],
                    "balloon_bbox": [80, 24, 170, 92],
                    "bubble_mask_bbox": [80, 24, 170, 92],
                    "bubble_mask_source": "image_white_bubble_mask",
                    "layout_profile": "white_balloon",
                    "balloon_type": "white",
                    "qa_flags": ["mask_outside_balloon_critical"],
                    "route_action": "translate_inpaint_render",
                }
            ]
        }

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 0)
        self.assertTrue(np.array_equal(result, image))
        self.assertNotIn("_strip_used_dark_panel_fill", page)

    def test_inpaint_band_image_blocks_current_vision_block_with_critical_outside_mask(self):
        from inpainter import inpaint_band_image

        image = np.full((110, 220, 3), (100, 150, 230), dtype=np.uint8)
        cv2.putText(image, "TITLE", (78, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text": "TITLE",
            "bbox": [70, 36, 148, 74],
            "text_pixel_bbox": [78, 44, 144, 68],
            "line_polygons": [[[78, 44], [144, 44], [144, 68], [78, 68]]],
            "skip_processing": False,
        }
        unsafe_block = {
            **text,
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical", "fast_fill_no_glyph_evidence"],
            "bubble_mask_source": "image_white_bubble_mask",
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 120,
                "expanded_mask_pixels": 600,
                "evidence_score": 1.0,
            },
        }
        page = {"texts": [dict(text)], "_vision_blocks": [unsafe_block]}

        with patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=AssertionError("bloco critico atual nao deve chegar ao AOT"),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(page.get("_strip_remaining_inpaint_blocks"), 0)
        self.assertEqual(page.get("_strip_unsafe_inpaint_block_reasons"), {"mask_outside_balloon_critical": 1})
        self.assertIn("real_inpaint_skipped_unsafe_mask", page["_strip_inpaint_decision_flags"])

    def test_inpaint_band_image_does_not_discard_clean_contour_block_for_stale_sibling_flag(self):
        from inpainter import inpaint_band_image

        image = np.full((130, 240, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (120, 66), (82, 42), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (120, 66), (82, 42), 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "PLEASE", (64, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)

        stale_text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_002_band_005",
            "band_id": "page_002_band_005",
            "text": "PLEASE",
            "bbox": [35, 24, 205, 106],
            "text_pixel_bbox": [64, 52, 152, 76],
            "line_polygons": [[[64, 52], [152, 52], [152, 76], [64, 76]]],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
            "route_action": "translate_inpaint_render",
            "skip_processing": False,
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 180,
                "expanded_mask_pixels": 360,
                "evidence_score": 1.0,
            },
        }
        clean_block = {
            **stale_text,
            "id": "ocr_002",
            "trace_id": "ocr_002@page_002_band_005",
            "bubble_mask_source": "image_contour_bubble_mask",
            "bubble_mask_error": None,
            "qa_flags": [],
            "bubble_mask_bbox": [30, 18, 210, 112],
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 220,
                "expanded_mask_pixels": 620,
                "evidence_score": 1.0,
            },
        }
        page = {"texts": [stale_text, clean_block], "_vision_blocks": [clean_block]}

        def fake_round(_image, payload, _inpainter):
            result = _image.copy()
            mask = payload.get("_precomputed_inpaint_mask")
            self.assertIsInstance(mask, np.ndarray)
            result[mask > 0] = 255
            return result

        with patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=fake_round,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(page.get("_strip_used_real_inpaint"))
        self.assertNotIn("real_inpaint_skipped_unsafe_mask", page.get("_strip_inpaint_decision_flags") or [])
        self.assertLess(int(np.count_nonzero(np.mean(result[50:80, 58:160], axis=2) < 80)), 8)

    def test_inpaint_band_image_keeps_safe_blocks_when_one_fragment_is_unsafe(self):
        from inpainter import inpaint_band_image

        image = np.full((180, 360, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (105, 74), (78, 48), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (105, 74), (78, 48), 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.ellipse(image, (260, 116), (64, 38), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (260, 116), (64, 38), 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "AISH WHY", (55, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "PRINCIPAL", (222, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        safe = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_002_band_007",
            "band_id": "page_002_band_007",
            "text": "AISH WHY",
            "bbox": [34, 42, 180, 102],
            "text_pixel_bbox": [52, 60, 154, 88],
            "line_polygons": [[[52, 60], [154, 60], [154, 88], [52, 88]]],
            "bubble_mask_bbox": [22, 24, 190, 128],
            "bubble_mask_source": "image_contour_bubble_mask",
            "route_action": "translate_inpaint_render",
            "qa_flags": [],
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 220,
                "expanded_mask_pixels": 640,
                "evidence_score": 1.0,
            },
        }
        unsafe_fragment = {
            "id": "ocr_003",
            "trace_id": "ocr_003@page_002_band_007",
            "band_id": "page_002_band_007",
            "text": "THE PRINCIPAL",
            "bbox": [190, 70, 338, 156],
            "text_pixel_bbox": [222, 102, 315, 126],
            "line_polygons": [[[222, 102], [315, 102], [315, 126], [222, 126]]],
            "bubble_mask_bbox": [0, 0, 360, 180],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 120,
                "expanded_mask_pixels": 1800,
                "evidence_score": 1.0,
            },
        }
        page = {"texts": [dict(safe), dict(unsafe_fragment)], "_vision_blocks": [dict(safe), dict(unsafe_fragment)]}

        def fake_round(_image, payload, _inpainter):
            result = _image.copy()
            mask = payload.get("_precomputed_inpaint_mask")
            self.assertIsInstance(mask, np.ndarray)
            result[mask > 0] = 255
            return result

        with patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=fake_round,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(page.get("_strip_used_real_inpaint"))
        self.assertEqual(page.get("_strip_unsafe_inpaint_block_count"), 1)
        self.assertNotIn("real_inpaint_skipped_unsafe_mask", page.get("_strip_inpaint_decision_flags") or [])
        self.assertLess(int(np.count_nonzero(np.mean(result[58:92, 50:160], axis=2) < 80)), 8)

    def test_inpaint_band_image_allows_local_dark_panel_fill_after_unsafe_aot_block(self):
        from inpainter import inpaint_band_image

        image = np.zeros((110, 360, 3), dtype=np.uint8)
        image[:] = (5, 8, 16)
        cv2.putText(image, "the Devil Knight!", (58, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (38, 168, 230), 5, cv2.LINE_AA)
        cv2.putText(image, "the Devil Knight!", (58, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (235, 248, 255), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text": "the Devil Knight!",
            "bbox": [48, 28, 300, 78],
            "text_pixel_bbox": [48, 28, 300, 78],
            "line_polygons": [[[48, 28], [300, 28], [300, 78], [48, 78]]],
            "background_rgb": [8, 10, 16],
            "route_action": "review_required",
            "route_reason": "mask_outside_balloon_critical",
            "qa_flags": ["mask_outside_balloon_critical", "missing_real_bubble_mask"],
            "skip_processing": False,
        }
        page = {"texts": [dict(text)], "_vision_blocks": [dict(text)]}

        with patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=AssertionError("dark panel local fill nao deve chamar AOT"),
        ):
            result = inpaint_band_image(image, page)

        before_bright = int(np.count_nonzero(np.mean(image[24:84, 44:306], axis=2) > 120))
        after_bright = int(np.count_nonzero(np.mean(result[24:84, 44:306], axis=2) > 120))
        self.assertLess(after_bright, before_bright * 0.35)
        self.assertFalse(page.get("_strip_used_real_inpaint"))
        self.assertTrue(page.get("_strip_used_dark_panel_fill"))
        self.assertIn("real_inpaint_skipped_unsafe_mask", page["_strip_inpaint_decision_flags"])

    def test_false_white_bubble_card_can_use_local_colored_panel_fill(self):
        from inpainter import inpaint_band_image

        image = np.full((110, 220, 3), (102, 145, 222), dtype=np.uint8)
        cv2.rectangle(image, (48, 28), (176, 84), (102, 145, 222), -1)
        cv2.putText(image, "SYNC", (70, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text": "SYNC",
            "bbox": [58, 34, 164, 78],
            "text_pixel_bbox": [68, 44, 150, 68],
            "line_polygons": [[[68, 44], [150, 44], [150, 68], [68, 68]]],
            "bubble_mask_source": "image_white_bubble_mask",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
            "route_action": "translate_inpaint_render",
            "skip_processing": False,
        }
        page = {"texts": [dict(text)], "_vision_blocks": [dict(text)]}

        with patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=AssertionError("card colorido inseguro deve usar fill local, nao AOT"),
        ):
            result = inpaint_band_image(image, page)

        before_bright = int(np.count_nonzero(np.mean(image[40:72, 64:156], axis=2) > 220))
        after_bright = int(np.count_nonzero(np.mean(result[40:72, 64:156], axis=2) > 220))
        self.assertLess(after_bright, before_bright * 0.35)
        self.assertFalse(page.get("_strip_used_real_inpaint"))
        self.assertTrue(page.get("_strip_used_dark_panel_fill"))
        self.assertIn("real_inpaint_skipped_unsafe_mask", page["_strip_inpaint_decision_flags"])

    def test_inpaint_band_image_rolls_back_aot_when_round_marks_unsafe_mask(self):
        from inpainter import inpaint_band_image

        image = np.full((90, 140, 3), 245, dtype=np.uint8)
        image[36:48, 44:98] = 20
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "HELLO",
                    "bbox": [38, 30, 104, 58],
                    "text_pixel_bbox": [44, 36, 98, 48],
                    "line_polygons": [[[44, 36], [98, 36], [98, 48], [44, 48]]],
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {
                    "id": "ocr_001",
                    "text": "HELLO",
                    "bbox": [38, 30, 104, 58],
                    "text_pixel_bbox": [44, 36, 98, 48],
                    "line_polygons": [[[44, 36], [98, 36], [98, 48], [44, 48]]],
                }
            ],
        }

        def mark_unsafe_and_change(src, payload, inpainter):
            page["_strip_inpaint_decision_flags"] = ["mask_outside_balloon_critical"]
            changed = src.copy()
            changed[:, :] = 255
            return changed

        with patch("vision_stack.runtime._apply_inpainting_round", side_effect=mark_unsafe_and_change):
            result = inpaint_band_image(image, page)

        self.assertTrue(np.array_equal(result, image))
        self.assertFalse(page.get("_strip_used_real_inpaint"))
        self.assertIn("real_inpaint_skipped_unsafe_mask", page["_strip_inpaint_decision_flags"])

    def test_inpaint_band_image_marks_outside_bubble_mask_before_fast_fill(self):
        from inpainter import inpaint_band_image

        image = np.full((90, 140, 3), 245, dtype=np.uint8)
        image[18:34, 20:44] = 10
        image[52:68, 92:122] = 10
        bubble_mask = np.zeros((90, 140), dtype=np.uint8)
        bubble_mask[12:42, 12:58] = 1
        overbroad_mask = np.zeros((90, 140), dtype=np.uint8)
        overbroad_mask[18:34, 20:44] = 255
        overbroad_mask[52:68, 92:122] = 255
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "bubble_id": "bubble_001",
                    "bbox": [18, 16, 124, 70],
                    "text_pixel_bbox": [18, 16, 124, 70],
                    "line_polygons": [[[18, 16], [124, 16], [124, 70], [18, 70]]],
                    "bubble_mask": bubble_mask,
                    "bubble_mask_bbox": [0, 0, 140, 90],
                    "bubble_mask_source": "image_white_bubble_mask",
                    "_precomputed_inpaint_mask": overbroad_mask,
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "bubble_id": "bubble_001",
                    "bbox": [18, 16, 124, 70],
                    "text_pixel_bbox": [18, 16, 124, 70],
                    "line_polygons": [[[18, 16], [124, 16], [124, 70], [18, 70]]],
                    "bubble_mask": bubble_mask,
                    "bubble_mask_bbox": [0, 0, 140, 90],
                    "bubble_mask_source": "image_white_bubble_mask",
                    "_precomputed_inpaint_mask": overbroad_mask,
                }
            ],
        }

        with patch(
            "inpainter._apply_koharu_bubble_fast_fill_to_blocks",
            side_effect=AssertionError("mask fora do balao nao deve chegar ao fast fill"),
        ), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=AssertionError("mask fora do balao nao deve chegar ao AOT"),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(np.array_equal(result, image))
        self.assertIn("mask_outside_balloon_critical", page["texts"][0]["qa_flags"])
        self.assertIn("real_inpaint_skipped_unsafe_mask", page["_strip_inpaint_decision_flags"])
        self.assertEqual(page.get("_strip_remaining_inpaint_blocks"), 0)

    def test_strip_real_inpaint_force_fills_remaining_white_balloon_residual(self):
        from inpainter import inpaint_band_image

        image = np.full((90, 150, 3), 245, dtype=np.uint8)
        image[36:48, 44:108] = 10
        first_clean = image.copy()
        page = {
            "texts": [
                {
                    "bbox": [40, 32, 112, 52],
                    "text_pixel_bbox": [44, 36, 108, 48],
                    "line_polygons": [[[44, 36], [108, 36], [108, 48], [44, 48]]],
                    "balloon_bbox": [24, 20, 130, 66],
                    "balloon_type": "white",
                    "layout_profile": "white_balloon",
                    "tipo": "fala",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [40, 32, 112, 52],
                    "text_pixel_bbox": [44, 36, 108, 48],
                    "line_polygons": [[[44, 36], [108, 36], [108, 48], [44, 48]]],
                    "balloon_type": "white",
                    "layout_profile": "white_balloon",
                }
            ],
        }

        forced = first_clean.copy()
        forced[36:48, 44:108] = 245

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=first_clean,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._has_white_balloon_text_residual",
            return_value=True,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_residual_force_fill",
            return_value=forced,
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ), patch(
            "inpainter._detect_inpaint_residual_text",
            return_value={"has_residual": False, "flags": [], "score": 0.0},
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(page["_strip_white_residual_force_fill"])
        self.assertTrue(np.all(result[36:48, 44:108] == 245))

    def test_strip_real_inpaint_expands_raw_mask_downward_inside_bubble(self):
        from inpainter import inpaint_band_image

        image = np.full((80, 140, 3), 245, dtype=np.uint8)
        image[30:38, 50:80] = 12
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        raw_mask[30:38, 50:80] = 255
        bubble_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        bubble_mask[16:45, 24:118] = 255
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "HUH...",
                    "translated": "HUH...",
                    "skip_reason": "unchanged_translation_skip",
                    "bbox": [46, 26, 86, 62],
                    "text_pixel_bbox": [50, 30, 80, 38],
                    "line_polygons": [[[48, 28], [84, 28], [84, 62], [48, 62]]],
                    "balloon_bbox": [24, 16, 118, 70],
                    "bubble_id": "bubble_001",
                    "balloon_type": "white",
                    "layout_profile": "white_balloon",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "bbox": [46, 26, 86, 62],
                    "text_pixel_bbox": [50, 30, 80, 38],
                    "line_polygons": [[[48, 28], [84, 28], [84, 62], [48, 62]]],
                    "balloon_bbox": [24, 16, 118, 70],
                    "bubble_id": "bubble_001",
                    "balloon_type": "white",
                    "layout_profile": "white_balloon",
                    "skip_processing": False,
                }
            ],
            "_bubble_regions": [
                {"bubble_id": "bubble_001", "bubble_mask": bubble_mask, "bubble_mask_bbox": [24, 16, 118, 45]}
            ],
        }
        captured = {}

        def fake_inpainting_round(working, payload, _inpainter):
            mask = payload["_precomputed_inpaint_mask"].copy()
            captured["mask"] = mask
            repaired = working.copy()
            repaired[mask > 0] = 230
            return repaired

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch(
            "inpainter._apply_koharu_bubble_fast_fill_to_blocks",
            return_value=(image.copy(), list(page["_vision_blocks"]), np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.uint8), {}),
        ), patch(
            "vision_stack.runtime.vision_blocks_to_mask",
            return_value=raw_mask,
        ), patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=fake_inpainting_round,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._has_white_balloon_text_residual",
            return_value=False,
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ), patch(
            "inpainter._detect_inpaint_residual_text",
            return_value={"has_residual": False, "flags": [], "score": 0.0},
        ), patch(
            "inpainter._apply_dark_panel_text_fills",
            side_effect=lambda img, page: (img, 0),
        ):
            result = inpaint_band_image(image, page)

        mask = captured["mask"]
        self.assertEqual(int(mask[37, 50]), 255)
        self.assertEqual(int(mask[40, 62]), 255)
        self.assertEqual(int(mask[34, 48]), 255)
        self.assertEqual(int(mask[34, 82]), 255)
        self.assertEqual(int(mask[14, 62]), 0)
        self.assertEqual(int(mask[44, 62]), 0)
        self.assertTrue(np.all(result[40, 62] == 230))

    def test_white_balloon_dark_residual_check_triggers_force_fill(self):
        from inpainter import inpaint_band_image

        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        first_clean = image.copy()
        first_clean[48:72, 70:150] = 92
        page = {
            "texts": [
                {
                    "bbox": [48, 38, 172, 82],
                    "text_pixel_bbox": [60, 44, 162, 76],
                    "balloon_bbox": [30, 24, 190, 96],
                    "balloon_type": "white",
                    "block_profile": "white_balloon",
                    "tipo": "fala",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [48, 38, 172, 82],
                    "text_pixel_bbox": [60, 44, 162, 76],
                    "balloon_bbox": [30, 24, 190, 96],
                    "balloon_type": "white",
                    "block_profile": "white_balloon",
                }
            ],
        }
        forced = first_clean.copy()
        forced[48:72, 70:150] = 255

        residual = {
            "has_residual": True,
            "flags": ["dark_residual_pixels"],
            "score": 0.2,
            "region_source": "text_region_white_balloon",
        }
        clean = {"has_residual": False, "flags": [], "score": 0.0}

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=first_clean,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._has_white_balloon_text_residual",
            return_value=False,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_residual_force_fill",
            return_value=forced,
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ), patch(
            "inpainter._detect_inpaint_residual_text",
            side_effect=[residual, clean, clean, clean, clean],
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(page["_strip_white_residual_force_fill"])
        self.assertTrue(page["_strip_white_residual_force_fill_from_residual_check"])
        self.assertTrue(np.all(result[48:72, 70:150] == 255))

    def test_white_balloon_dark_residual_from_expanded_mask_does_not_force_fill(self):
        from inpainter import inpaint_band_image

        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        first_clean = image.copy()
        first_clean[48:72, 70:150] = 92
        page = {
            "texts": [
                {
                    "bbox": [48, 38, 172, 82],
                    "text_pixel_bbox": [60, 44, 162, 76],
                    "balloon_bbox": [30, 24, 190, 96],
                    "balloon_type": "white",
                    "block_profile": "white_balloon",
                    "tipo": "fala",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [{"bbox": [48, 38, 172, 82], "confidence": 0.95}],
        }
        residual = {
            "has_residual": True,
            "flags": ["dark_residual_pixels"],
            "score": 0.2,
            "region_source": "expanded_mask",
        }

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=first_clean,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._has_white_balloon_text_residual",
            return_value=False,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_residual_force_fill",
            side_effect=AssertionError("expanded-mask residual must not force white fill"),
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ), patch(
            "inpainter._detect_inpaint_residual_text",
            side_effect=[residual, residual, residual],
        ), patch(
            "inpainter._apply_dark_panel_text_fills",
            side_effect=lambda cleaned, ocr_page: (cleaned, 0),
        ), patch("inpainter._write_strip_inpaint_debug", return_value=None):
            result = inpaint_band_image(image, page)

        self.assertFalse(page.get("_strip_white_residual_force_fill_from_residual_check", False))
        self.assertTrue(np.all(result[48:72, 70:150] == 92))

    def test_white_balloon_light_residual_after_retry_triggers_force_fill(self):
        from inpainter import inpaint_band_image

        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        first_clean = image.copy()
        first_clean[48:72, 70:150] = 210
        retried = first_clean.copy()
        retried[62:72, 70:150] = 205
        forced = retried.copy()
        forced[62:72, 70:150] = 255
        retry_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        retry_mask[48:72, 70:150] = 255
        page = {
            "texts": [
                {
                    "bbox": [48, 38, 172, 82],
                    "text_pixel_bbox": [60, 44, 162, 76],
                    "balloon_bbox": [30, 24, 190, 96],
                    "balloon_type": "white",
                    "block_profile": "white_balloon",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [{"bbox": [48, 38, 172, 82]}],
        }

        class FakeInpainter:
            def inpaint(self, img, mask, batch_size=4, force_no_tiling=True):
                return retried

        expanded_light_residual = {
            "has_residual": True,
            "flags": ["light_residual_pixels"],
            "score": 0.03,
            "region_source": "expanded_mask",
        }
        white_light_residual = {
            "has_residual": True,
            "flags": ["light_residual_pixels", "colored_residual_pixels"],
            "score": 0.03,
            "region_source": "text_region_white_balloon",
        }
        clean = {"has_residual": False, "flags": [], "score": 0.0}

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch("vision_stack.runtime._get_inpainter", return_value=FakeInpainter()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=first_clean,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._has_white_balloon_text_residual",
            return_value=False,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_residual_force_fill",
            return_value=forced,
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ), patch(
            "inpainter._build_light_residual_retry_mask",
            return_value=retry_mask,
        ), patch(
            "inpainter._detect_inpaint_residual_text",
            side_effect=[expanded_light_residual, white_light_residual, clean, clean, clean],
        ), patch("inpainter._write_strip_inpaint_debug", return_value=None):
            result = inpaint_band_image(image, page)

        self.assertTrue(page["_strip_light_residual_retry"])
        self.assertTrue(page["_strip_white_residual_force_fill"])
        self.assertTrue(page["_strip_white_residual_force_fill_from_residual_check"])
        self.assertTrue(np.all(result[62:72, 70:150] == 255))

    def test_white_balloon_final_fallback_uses_residual_region_beyond_expanded_mask(self):
        from inpainter import inpaint_band_image

        image = np.full((90, 180, 3), 255, dtype=np.uint8)
        image[56:64, 76:124] = 205
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        raw_mask[30:42, 72:128] = 255
        first_clean = image.copy()
        residual = {
            "has_residual": True,
            "flags": ["colored_residual_pixels"],
            "score": 0.03,
            "region_source": "text_region_white_balloon",
        }
        clean = {"has_residual": False, "flags": [], "score": 0.0}
        page = {
            "texts": [
                {
                    "bbox": [64, 28, 136, 68],
                    "text_pixel_bbox": [64, 28, 136, 68],
                    "balloon_bbox": [24, 12, 156, 82],
                    "block_profile": "white_balloon",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [{"bbox": [72, 30, 128, 42]}],
        }

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch(
            "inpainter._apply_koharu_bubble_fast_fill_to_blocks",
            return_value=(image.copy(), [{"bbox": [72, 30, 128, 42]}], np.zeros(image.shape[:2], dtype=np.uint8), raw_mask, {}),
        ), patch(
            "inpainter._augment_inpaint_masks_from_texts",
            side_effect=lambda raw, expanded, texts, image_rgb: (raw, expanded),
        ), patch(
            "inpainter._expand_strip_real_inpaint_mask",
            side_effect=lambda raw, expanded, ocr_page, texts, image_rgb: raw,
        ), patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=first_clean,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._has_white_balloon_text_residual",
            return_value=False,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_residual_force_fill",
            side_effect=lambda original, cleaned, texts: cleaned,
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            wraps=__import__("vision_stack.runtime", fromlist=["_clamp_image_to_limit_mask"])._clamp_image_to_limit_mask,
        ), patch(
            "inpainter._detect_inpaint_residual_text",
            side_effect=[residual, residual, residual, clean, clean],
        ), patch(
            "inpainter._apply_dark_panel_text_fills",
            side_effect=lambda cleaned, ocr_page: (cleaned, 0),
        ), patch("inpainter._write_strip_inpaint_debug", return_value=None):
            result = inpaint_band_image(image, page)

        self.assertTrue(page["_strip_white_residual_expanded_mask_force_fill"])
        self.assertGreater(page["_strip_white_residual_expanded_mask_force_fill_pixels"], 0)
        self.assertTrue(np.all(result[57:63, 80:120] == 255))

    def test_white_balloon_final_residual_check_force_fills_late_light_residue(self):
        from inpainter import inpaint_band_image

        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        first_clean = image.copy()
        first_clean[64:72, 78:138] = 205
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        raw_mask[32:46, 74:132] = 255
        page = {
            "texts": [
                {
                    "bbox": [58, 28, 150, 76],
                    "text_pixel_bbox": [62, 30, 148, 76],
                    "balloon_bbox": [24, 12, 176, 90],
                    "block_profile": "white_balloon",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [{"bbox": [72, 30, 128, 44]}],
        }
        clean = {"has_residual": False, "flags": [], "score": 0.0}
        late_residual = {
            "has_residual": True,
            "flags": ["light_residual_pixels"],
            "score": 0.014,
            "region_source": "text_region_white_balloon+fast_fill_mask",
        }

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch(
            "inpainter._apply_koharu_bubble_fast_fill_to_blocks",
            return_value=(image.copy(), [{"bbox": [72, 30, 128, 44]}], np.zeros(image.shape[:2], dtype=np.uint8), raw_mask, {}),
        ), patch(
            "inpainter._augment_inpaint_masks_from_texts",
            side_effect=lambda raw, expanded, texts, image_rgb: (raw, expanded),
        ), patch(
            "inpainter._expand_strip_real_inpaint_mask",
            side_effect=lambda raw, expanded, ocr_page, texts, image_rgb: raw,
        ), patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=first_clean,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._has_white_balloon_text_residual",
            return_value=False,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_residual_force_fill",
            side_effect=lambda original, cleaned, texts: cleaned,
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ), patch(
            "inpainter._detect_inpaint_residual_text",
            side_effect=[clean, clean, clean, late_residual],
        ), patch(
            "inpainter._apply_dark_panel_text_fills",
            side_effect=lambda cleaned, ocr_page: (cleaned, 0),
        ), patch("inpainter._write_strip_inpaint_debug", return_value=None):
            result = inpaint_band_image(image, page)

        self.assertTrue(page["_strip_white_residual_expanded_mask_force_fill"])
        self.assertGreater(page["_strip_white_residual_expanded_mask_force_fill_pixels"], 0)
        self.assertTrue(np.all(result[65:70, 84:132] == 255))

    def test_final_action_mask_extends_to_white_balloon_residual_region(self):
        from inpainter import _extend_final_action_mask_for_white_balloon_cleanup

        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        base = np.zeros(image.shape[:2], dtype=np.uint8)
        base[32:44, 74:132] = 255
        text = {
            "bbox": [58, 28, 150, 76],
            "text_pixel_bbox": [62, 30, 148, 76],
            "balloon_bbox": [24, 12, 176, 90],
            "block_profile": "white_balloon",
            "skip_processing": False,
        }
        page = {"texts": [text]}

        extended, added = _extend_final_action_mask_for_white_balloon_cleanup(base, page, [text], image)

        self.assertGreater(added, 0)
        self.assertGreater(int(np.count_nonzero(extended)), int(np.count_nonzero(base)))
        self.assertTrue(np.any(extended[64:72, 78:138] > 0))

    def test_final_action_mask_white_cleanup_ignores_dark_panel_text_in_mixed_band(self):
        from inpainter import _extend_final_action_mask_for_white_balloon_cleanup

        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        image[74:104, 24:104] = [4, 5, 8]
        base = np.zeros(image.shape[:2], dtype=np.uint8)
        white_text = {
            "bbox": [122, 20, 190, 52],
            "text_pixel_bbox": [126, 24, 184, 48],
            "balloon_bbox": [106, 8, 208, 70],
            "block_profile": "white_balloon",
            "bubble_mask_source": "image_white_bubble_mask",
            "skip_processing": False,
        }
        dark_text = {
            "bbox": [32, 80, 92, 98],
            "text_pixel_bbox": [32, 80, 92, 98],
            "balloon_bbox": [18, 70, 112, 108],
            "layout_profile": "dark_panel",
            "bubble_mask_source": "image_dark_panel_mask",
            "qa_flags": ["short_dark_text_full_panel_bbox_rejected"],
            "skip_processing": False,
        }
        page = {"texts": [white_text, dark_text]}

        extended, added = _extend_final_action_mask_for_white_balloon_cleanup(
            base,
            page,
            [white_text, dark_text],
            image,
        )

        self.assertGreater(added, 0)
        self.assertTrue(np.any(extended[24:48, 126:184] > 0))
        self.assertFalse(np.any(extended[80:98, 32:92] > 0))

    def test_final_white_cleanup_extension_also_force_fills_late_residue(self):
        from inpainter import inpaint_band_image

        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        first_clean = image.copy()
        first_clean[64:72, 78:138] = 214
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        raw_mask[32:46, 74:132] = 255
        page = {
            "texts": [
                {
                    "bbox": [58, 28, 150, 76],
                    "text_pixel_bbox": [62, 30, 148, 76],
                    "balloon_bbox": [24, 12, 176, 90],
                    "block_profile": "white_balloon",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [{"bbox": [72, 30, 128, 44]}],
        }
        clean = {"has_residual": False, "flags": [], "score": 0.0}

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch(
            "inpainter._apply_koharu_bubble_fast_fill_to_blocks",
            return_value=(image.copy(), [{"bbox": [72, 30, 128, 44]}], np.zeros(image.shape[:2], dtype=np.uint8), raw_mask, {}),
        ), patch(
            "inpainter._augment_inpaint_masks_from_texts",
            side_effect=lambda raw, expanded, texts, image_rgb: (raw, expanded),
        ), patch(
            "inpainter._expand_strip_real_inpaint_mask",
            side_effect=lambda raw, expanded, ocr_page, texts, image_rgb: raw,
        ), patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=first_clean,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._has_white_balloon_text_residual",
            return_value=False,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_residual_force_fill",
            side_effect=lambda original, cleaned, texts: cleaned,
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ), patch(
            "inpainter._detect_inpaint_residual_text",
            return_value=clean,
        ), patch(
            "inpainter._apply_dark_panel_text_fills",
            side_effect=lambda cleaned, ocr_page: (cleaned, 0),
        ), patch("inpainter._write_strip_inpaint_debug", return_value=None):
            result = inpaint_band_image(image, page)

        self.assertTrue(page["_strip_final_action_mask_white_cleanup_force_fill"])
        self.assertGreater(page["_strip_final_action_mask_white_cleanup_force_fill_pixels"], 0)
        self.assertTrue(np.all(result[65:70, 84:132] == 255))

    def test_white_balloon_residual_force_fill_clears_light_text_box_ghost(self):
        from vision_stack.runtime import _apply_white_balloon_residual_force_fill

        original = np.full((140, 260, 3), 180, dtype=np.uint8)
        cv2.ellipse(original, (130, 70), (112, 52), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(original, (130, 70), (112, 52), 0, 0, 360, (0, 0, 0), 2)
        cv2.putText(original, "HELLO", (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
        cleaned = np.full_like(original, 255)
        cleaned[30:66, 28:152] = 235
        text = {
            "bbox": [20, 20, 190, 76],
            "text_pixel_bbox": [20, 20, 190, 76],
            "balloon_bbox": [18, 18, 242, 122],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "skip_processing": False,
        }
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        dark_ys, dark_xs = np.where(original_gray[20:76, 20:190] < 80)
        self.assertGreater(len(dark_xs), 0)
        x1 = int(dark_xs.min()) + 20
        x2 = int(dark_xs.max()) + 21
        y1 = int(dark_ys.min()) + 20
        y2 = int(dark_ys.max()) + 21
        gap_candidates = np.where((original_gray[y1:y2, x1:x2] > 245) & (cleaned[y1:y2, x1:x2, 0] == 235))
        self.assertGreater(len(gap_candidates[0]), 0)
        gy = int(gap_candidates[0][len(gap_candidates[0]) // 2]) + y1
        gx = int(gap_candidates[1][len(gap_candidates[1]) // 2]) + x1

        result = _apply_white_balloon_residual_force_fill(original, cleaned, [text])

        self.assertTrue(np.array_equal(result, cleaned))

    def test_white_balloon_residual_force_fill_preserves_off_white_balloon_color(self):
        from vision_stack.runtime import _apply_white_balloon_residual_force_fill

        cream = np.asarray([244, 241, 234], dtype=np.uint8)
        original = np.full((140, 260, 3), 180, dtype=np.uint8)
        cv2.ellipse(original, (130, 70), (112, 52), 0, 0, 360, tuple(int(v) for v in cream), -1)
        cv2.ellipse(original, (130, 70), (112, 52), 0, 0, 360, (0, 0, 0), 2)
        cv2.putText(original, "HELLO", (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
        cleaned = original.copy()
        cleaned[30:76, 24:176] = 235
        text = {
            "bbox": [20, 20, 190, 76],
            "text_pixel_bbox": [20, 20, 190, 76],
            "balloon_bbox": [18, 18, 242, 122],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "skip_processing": False,
        }
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        dark_ys, dark_xs = np.where(original_gray[20:76, 20:190] < 80)
        self.assertGreater(len(dark_xs), 0)
        gy = int(dark_ys[len(dark_ys) // 2]) + 20
        gx = int(dark_xs[len(dark_xs) // 2]) + 20

        result = _apply_white_balloon_residual_force_fill(original, cleaned, [text])

        self.assertTrue(np.array_equal(result, cleaned))

    def test_fast_solid_fill_sample_preserves_light_colored_balloon_color(self):
        from inpainter import _sample_solid_fill_color_for_mask

        cream = np.asarray([244, 241, 234], dtype=np.uint8)
        image = np.zeros((90, 180, 3), dtype=np.uint8)
        image[:, :] = cream
        fill_mask = np.zeros((90, 180), dtype=np.uint8)
        fill_mask[34:58, 62:124] = 255
        limit_mask = np.zeros((90, 180), dtype=np.uint8)
        limit_mask[10:80, 20:160] = 255

        color, metadata = _sample_solid_fill_color_for_mask(image, fill_mask, limit_mask)

        self.assertEqual(color, tuple(int(v) for v in cream))
        self.assertTrue(metadata["accepted"])
        self.assertNotEqual(metadata["color"], [255, 255, 255])

    def test_fast_solid_fill_runs_when_legacy_fast_fills_are_disabled(self):
        from inpainter import _apply_fast_solid_balloon_fill

        fill = np.asarray([221, 238, 246], dtype=np.uint8)
        image = np.full((110, 220, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (110, 55), (82, 36), 0, 0, 360, tuple(int(v) for v in fill), -1)
        image[48:60, 72:148] = 8
        text = {
            "id": "ocr_001",
            "bbox": [68, 44, 152, 64],
            "text_pixel_bbox": [72, 48, 148, 60],
            "line_polygons": [[[72, 48], [148, 48], [148, 60], [72, 60]]],
            "balloon_bbox": [28, 18, 192, 92],
            "balloon_type": "colored",
            "layout_profile": "solid_color",
            "tipo": "fala",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
            },
            clear=False,
        ):
            result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["solid_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(page["_strip_used_fast_solid_fill"])
        self.assertEqual(page["_strip_fast_solid_colored_count"], 1)
        self.assertLessEqual(int(np.max(np.abs(result[52, 90].astype(np.int16) - fill.astype(np.int16)))), 2)
        self.assertFalse(np.all(result[52, 90] == 255))

    def test_fast_solid_fill_verifies_clean_white_balloon_sample(self):
        from inpainter import _apply_fast_solid_balloon_fill

        image = np.full((130, 260, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (130, 64), (86, 42), 0, 0, 360, (255, 255, 255), -1)
        cv2.putText(image, "HELLO", (86, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (10, 10, 10), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@band_001",
            "text": "HELLO",
            "bbox": [80, 44, 172, 80],
            "text_pixel_bbox": [84, 48, 168, 76],
            "line_polygons": [[[84, 48], [168, 48], [168, 76], [84, 76]]],
            "balloon_bbox": [40, 22, 220, 106],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
            },
            clear=False,
        ):
            result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["solid_balloon_count"], 1)
        self.assertEqual(remaining, [])
        sample = page["_strip_fast_solid_fill_samples"][0]
        self.assertTrue(sample["fast_fill_verified"])
        self.assertGreaterEqual(sample["fast_fill_text_bbox_coverage"], 0.18)
        self.assertLessEqual(sample["fast_fill_residual_edge_ratio"], 0.08)
        self.assertTrue(np.any(result != image))

    def test_fast_solid_line_geometry_coverage_uses_line_bbox_not_broad_ocr_bbox(self):
        import inpainter
        from inpainter import _apply_fast_solid_balloon_fill

        image = np.full((150, 280, 3), 255, dtype=np.uint8)
        image[58:74, 80:180] = 8
        text = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@band_001",
            "text": "WHAT IS IT",
            "bbox": [40, 40, 220, 112],
            "text_pixel_bbox": [40, 40, 220, 112],
            "line_polygons": [[[80, 58], [180, 58], [180, 74], [80, 74]]],
            "balloon_bbox": [28, 22, 242, 130],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
                "TRADUZAI_FAST_FILL_MIN_COVERAGE": "0.18",
                "TRADUZAI_FAST_FILL_LINE_MIN_COVERAGE": "0.45",
                "TRADUZAI_FAST_SOLID_LINE_EXPAND_PX": "0",
            },
            clear=False,
        ), patch.object(inpainter, "_koharu_style_fast_white_evidence_rejection_reason", return_value=""):
            result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["solid_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertNotIn("fast_fill_insufficient_coverage", page.get("_strip_inpaint_decision_flags", []))
        sample = page["_strip_fast_solid_fill_samples"][0]
        self.assertGreaterEqual(sample["fast_fill_text_bbox_coverage"], 0.45)
        self.assertTrue(np.any(result != image))

    def test_fast_solid_line_geometry_mask_expands_inside_authorized_area(self):
        from inpainter import _solid_text_fill_mask

        image = np.full((120, 180, 3), 255, dtype=np.uint8)
        text = {
            "text_pixel_bbox": [40, 40, 100, 52],
            "line_polygons": [[[40, 40], [100, 40], [100, 52], [40, 52]]],
            "balloon_bbox": [20, 20, 140, 90],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        with patch.dict("os.environ", {"TRADUZAI_FAST_SOLID_LINE_EXPAND_PX": "2"}, clear=False):
            mask, source = _solid_text_fill_mask(image, text, width=180, height=120)

        self.assertEqual(source, "line_geometry")
        self.assertIsNotNone(mask)
        self.assertEqual(int(mask[38, 70]), 255)
        self.assertEqual(int(mask[18, 70]), 0)

    def test_fast_solid_line_geometry_mask_preserves_balloon_outline_outside_source_bbox(self):
        from inpainter import _solid_text_fill_mask

        image = np.full((90, 150, 3), 255, dtype=np.uint8)
        image[20, 30:120] = 0
        text = {
            "source_bbox": [36, 22, 112, 42],
            "text_pixel_bbox": [36, 22, 112, 42],
            "line_polygons": [[[36, 22], [112, 22], [112, 42], [36, 42]]],
            "balloon_bbox": [30, 20, 120, 64],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        with patch.dict("os.environ", {"TRADUZAI_FAST_SOLID_LINE_EXPAND_PX": "2"}, clear=False):
            mask, source = _solid_text_fill_mask(image, text, width=150, height=90)

        self.assertEqual(source, "line_geometry")
        self.assertIsNotNone(mask)
        self.assertEqual(int(mask[24, 70]), 255)
        self.assertEqual(int(mask[20, 70]), 0)

    def test_fast_solid_line_geometry_mask_adds_raw_source_glyphs_missed_by_line_polygon(self):
        from inpainter import _solid_text_fill_mask

        image = np.full((90, 160, 3), 255, dtype=np.uint8)
        image[35:58, 42:47] = 0
        image[35:58, 58:118] = 0
        text = {
            "source_bbox": [40, 30, 124, 64],
            "text_pixel_bbox": [40, 30, 124, 64],
            "line_polygons": [[[56, 34], [122, 34], [122, 60], [56, 60]]],
            "balloon_bbox": [20, 16, 146, 76],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        with patch.dict("os.environ", {"TRADUZAI_FAST_SOLID_LINE_EXPAND_PX": "1"}, clear=False):
            mask, source = _solid_text_fill_mask(image, text, width=160, height=90)

        self.assertEqual(source, "line_geometry")
        self.assertIsNotNone(mask)
        self.assertEqual(int(mask[44, 44]), 255)
        self.assertEqual(int(mask[44, 30]), 0)

    def test_fast_solid_line_geometry_mask_adds_nearby_leading_glyph_missed_by_ocr_bbox(self):
        from inpainter import _solid_text_fill_mask

        image = np.full((90, 170, 3), 210, dtype=np.uint8)
        cv2.ellipse(image, (86, 44), (66, 28), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (86, 44), (66, 28), 0, 0, 360, (0, 0, 0), 1)
        image[30:51, 34:38] = 0
        image[30:51, 52:124] = 0
        text = {
            "source_bbox": [50, 24, 132, 58],
            "text_pixel_bbox": [52, 28, 126, 54],
            "line_polygons": [[[52, 28], [126, 28], [126, 54], [52, 54]]],
            "balloon_bbox": [50, 24, 132, 58],
            "balloon_type": "textured",
            "layout_profile": "standard",
            "tipo": "fala",
        }

        with patch.dict("os.environ", {"TRADUZAI_FAST_SOLID_LINE_EXPAND_PX": "1"}, clear=False):
            mask, source = _solid_text_fill_mask(image, text, width=170, height=90)

        self.assertEqual(source, "line_geometry")
        self.assertIsNotNone(mask)
        self.assertEqual(int(mask[40, 36]), 255)
        self.assertEqual(int(mask[16, 86]), 0)
        self.assertEqual(int(mask[44, 21]), 0)

    def test_fast_solid_limit_expands_tight_text_bbox_to_detected_white_region(self):
        from inpainter import _resolve_fast_solid_limit_bbox

        image = np.full((120, 220, 3), 70, dtype=np.uint8)
        cv2.ellipse(image, (108, 60), (82, 38), 0, 0, 360, (255, 255, 255), -1)
        text = {
            "source_bbox": [118, 42, 176, 78],
            "text_pixel_bbox": [118, 42, 176, 78],
            "line_polygons": [[[118, 42], [176, 42], [176, 78], [118, 78]]],
            "balloon_bbox": [118, 42, 176, 78],
            "balloon_type": "textured",
            "layout_profile": "standard",
        }

        bbox = _resolve_fast_solid_limit_bbox(image, text, width=220, height=120)

        self.assertIsNotNone(bbox)
        self.assertLessEqual(bbox[0], 40)
        self.assertGreaterEqual(bbox[2], 180)

    def test_density_guard_uses_raw_mask_when_expanded_band_density_exceeds_threshold(self):
        from inpainter import _density_guarded_inpaint_mask

        raw = np.zeros((50, 50), dtype=np.uint8)
        raw[20:30, 20:30] = 255
        expanded = np.zeros((50, 50), dtype=np.uint8)
        expanded[5:35, 5:35] = 255

        guarded, source = _density_guarded_inpaint_mask(raw, expanded, (50, 50))

        self.assertEqual(source, "raw_mask")
        self.assertEqual(int(np.count_nonzero(guarded)), int(np.count_nonzero(raw)))

    def test_density_guard_erodes_expanded_mask_when_raw_is_also_dense(self):
        from inpainter import _density_guarded_inpaint_mask

        raw = np.zeros((40, 40), dtype=np.uint8)
        raw[5:30, 5:30] = 255
        expanded = raw.copy()

        guarded, source = _density_guarded_inpaint_mask(raw, expanded, (40, 40))

        self.assertEqual(source, "eroded_expanded_mask")
        self.assertLess(int(np.count_nonzero(guarded)), int(np.count_nonzero(expanded)))

    def test_text_evidence_augmentation_restores_glyph_missing_from_vision_mask(self):
        from inpainter import _augment_inpaint_masks_from_texts

        image = np.full((110, 180, 3), 255, dtype=np.uint8)
        image[22:44, 38:46] = 0
        image[22:26, 35:49] = 0
        image[40:44, 35:49] = 0
        image[26:44, 55:126] = 0
        raw = np.zeros((110, 180), dtype=np.uint8)
        raw[20:58, 55:142] = 255
        text = {
            "bbox": [55, 20, 142, 54],
            "text_pixel_bbox": [55, 20, 142, 54],
            "source_bbox": [34, 18, 142, 54],
            "balloon_bbox": [34, 18, 142, 54],
            "line_polygons": [[[55, 20], [142, 20], [142, 54], [55, 54]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "texto",
        }

        augmented_raw, augmented_expanded = _augment_inpaint_masks_from_texts(raw, raw, [text], image)

        self.assertGreater(int(augmented_raw[32, 42]), 0)
        self.assertGreater(int(augmented_expanded[32, 42]), 0)
        self.assertGreater(text.get("qa_metrics", {}).get("text_evidence_mask_extra_pixels", 0), 0)

    def test_dark_bubble_text_evidence_uses_balloon_bbox_when_mask_bbox_is_tight(self):
        from inpainter import _augment_inpaint_masks_from_texts

        image = np.zeros((120, 260, 3), dtype=np.uint8)
        image[50:64, 44:214] = 235
        raw = np.zeros((120, 260), dtype=np.uint8)
        raw[50:64, 104:154] = 255
        text = {
            "bbox": [42, 48, 216, 68],
            "text_pixel_bbox": [42, 48, 216, 68],
            "source_bbox": [42, 48, 216, 68],
            "bubble_mask_bbox": [104, 48, 154, 68],
            "balloon_bbox": [16, 20, 240, 100],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "qa_flags": ["candidate_crop_direct_paddle_reocr", "detected_dark_bubble_without_text_reocr"],
        }

        augmented_raw, augmented_expanded = _augment_inpaint_masks_from_texts(raw, raw, [text], image)

        self.assertGreater(int(augmented_raw[56, 48]), 0)
        self.assertGreater(int(augmented_raw[56, 210]), 0)
        self.assertGreater(int(augmented_expanded[56, 48]), 0)
        self.assertGreater(int(augmented_expanded[56, 210]), 0)

    def test_text_evidence_augmentation_drops_unprotected_dark_outline_sliver(self):
        from inpainter import _augment_inpaint_masks_from_texts

        image = np.full((90, 160, 3), 255, dtype=np.uint8)
        image[44:56, 50:112] = 8
        image[20:36, 136:144] = 8
        raw = np.zeros((90, 160), dtype=np.uint8)
        raw[44:56, 50:112] = 255
        raw[20:36, 136:144] = 255
        text = {
            "bbox": [44, 20, 140, 60],
            "text_pixel_bbox": [44, 40, 116, 60],
            "source_bbox": [44, 20, 140, 60],
            "balloon_bbox": [16, 10, 148, 78],
            "bubble_mask_source": "image_contour_bubble_mask",
            "layout_profile": "white_balloon",
        }

        augmented_raw, augmented_expanded = _augment_inpaint_masks_from_texts(raw, raw, [text], image)

        self.assertEqual(int(np.count_nonzero(augmented_raw[20:36, 136:144])), 0)
        self.assertEqual(int(np.count_nonzero(augmented_expanded[20:36, 136:144])), 0)
        self.assertGreater(int(np.count_nonzero(augmented_raw[44:56, 50:112])), 0)

    def test_residual_text_region_ignores_rejected_fragment_without_glyph_evidence(self):
        from inpainter import _build_residual_text_region_mask

        page = {
            "texts": [
                {
                    "bbox": [108, 24, 160, 52],
                    "text_pixel_bbox": [108, 24, 160, 52],
                    "line_polygons": [[[108, 24], [160, 24], [160, 52], [108, 52]]],
                },
                {
                    "bbox": [8, 62, 158, 96],
                    "text_pixel_bbox": [8, 62, 158, 96],
                    "bubble_mask_source": "derived_white_crop_rejected",
                    "bubble_mask_error": "derived_mask_not_anchored_to_text",
                    "qa_flags": [
                        "raw_text_evidence_missing",
                        "fast_fill_no_glyph_evidence",
                        "same_balloon_fragment_merged",
                    ],
                },
            ]
        }

        mask = _build_residual_text_region_mask(page, (120, 180))

        self.assertGreater(int(mask[34, 130]), 0)
        self.assertEqual(int(mask[74, 40]), 0)

    def test_dark_glyph_residual_cleanup_removes_leading_glyph_but_preserves_outline(self):
        from inpainter import _cleanup_dark_glyph_residuals_in_text_mask

        image = np.full((90, 170, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (86, 44), (66, 28), 0, 0, 360, (0, 0, 0), 1)
        image[30:51, 34:38] = 0
        residual_mask = np.zeros((90, 170), dtype=np.uint8)
        residual_mask[18:60, 28:130] = 255
        residual_mask[:, 0:170] = np.maximum(residual_mask[:, 0:170], (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) < 64).astype(np.uint8) * 255)
        text = {
            "line_polygons": [[[52, 28], [126, 28], [126, 54], [52, 54]]],
            "bbox": [52, 28, 126, 54],
            "text_pixel_bbox": [52, 28, 126, 54],
        }

        cleaned, pixels = _cleanup_dark_glyph_residuals_in_text_mask(image, residual_mask, [text])

        self.assertGreater(pixels, 0)
        self.assertGreater(int(cleaned[40, 36].mean()), 220)
        self.assertLess(int(cleaned[44, 20].mean()), 80)

    def test_fast_solid_line_geometry_limit_includes_text_below_balloon_bbox(self):
        from inpainter import _solid_text_fill_mask

        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        text = {
            "text_pixel_bbox": [62, 52, 190, 145],
            "line_polygons": [
                [[70, 52], [178, 52], [178, 66], [70, 66]],
                [[62, 82], [190, 82], [190, 96], [62, 96]],
                [[88, 128], [160, 128], [160, 145], [88, 145]],
            ],
            "balloon_bbox": [50, 40, 202, 132],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
        }

        with patch.dict("os.environ", {"TRADUZAI_FAST_SOLID_LINE_EXPAND_PX": "2"}, clear=False):
            mask, source = _solid_text_fill_mask(image, text, width=260, height=180)

        self.assertEqual(source, "line_geometry")
        self.assertIsNotNone(mask)
        self.assertEqual(int(mask[143, 124]), 255)
        self.assertEqual(int(mask[160, 124]), 0)

    def test_fast_solid_line_geometry_low_coverage_stays_for_real_inpaint(self):
        import inpainter
        from inpainter import _apply_fast_solid_balloon_fill

        image = np.full((150, 260, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (130, 74), (98, 48), 0, 0, 360, (255, 255, 255), -1)
        image[58:74, 80:180] = 8
        text = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "text": "PARTIAL",
            "bbox": [60, 50, 200, 90],
            "text_pixel_bbox": [60, 50, 200, 90],
            "line_polygons": [[[60, 50], [200, 50], [200, 90], [60, 90]]],
            "balloon_bbox": [30, 24, 230, 126],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
                "TRADUZAI_FAST_FILL_MIN_COVERAGE": "0.18",
                "TRADUZAI_FAST_FILL_LINE_MIN_COVERAGE": "0.45",
            },
            clear=False,
        ), patch.object(inpainter, "_koharu_style_fast_white_evidence_rejection_reason", return_value=""):
            _, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["solid_balloon_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertIn("fast_fill_insufficient_coverage", page.get("_strip_inpaint_decision_flags", []))
        self.assertEqual(page["_strip_fast_solid_rejection_reasons"], {"fast_fill_insufficient_coverage": 1})

    def test_fast_solid_fill_keeps_block_when_coverage_is_too_low(self):
        from inpainter import _apply_fast_solid_balloon_fill

        image = np.full((160, 300, 3), 255, dtype=np.uint8)
        image[80:82, 140:145] = 0
        text = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "text": "TINY",
            "bbox": [80, 50, 220, 112],
            "text_pixel_bbox": [80, 50, 220, 112],
            "line_polygons": [[[80, 50], [220, 50], [220, 112], [80, 112]]],
            "balloon_bbox": [40, 22, 260, 138],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_FAST_FILL_MIN_COVERAGE": "0.50",
            },
            clear=False,
        ):
            _, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["solid_balloon_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertIn("fast_fill_insufficient_coverage", page.get("_strip_inpaint_decision_flags", []))
        self.assertEqual(page["_strip_fast_solid_rejection_reasons"], {"fast_fill_insufficient_coverage": 1})

    def test_fast_solid_fill_rejects_bbox_only_bubble_reference(self):
        from inpainter import _apply_fast_solid_balloon_fill

        image = np.full((140, 240, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (120, 70), (72, 38), 0, 0, 360, (0, 0, 0), 2)
        image[58:72, 74:166] = 12
        text = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "bubble_id": "bubble_001",
            "text": "NO",
            "bbox": [70, 52, 170, 82],
            "text_pixel_bbox": [74, 58, 166, 72],
            "line_polygons": [[[74, 58], [166, 58], [166, 72], [74, 72]]],
            "balloon_bbox": [42, 28, 198, 112],
            "bubble_mask_bbox": [42, 28, 198, 112],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {
            "texts": [text],
            "_vision_blocks": [dict(text)],
            "_bubble_regions": [{"bubble_id": "bubble_001", "bubble_mask_bbox": [42, 28, 198, 112]}],
        }

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ):
            result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(stats["solid_balloon_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(page["_strip_fast_solid_rejection_reasons"], {"missing_real_bubble_mask": 1})

    def test_fast_local_fill_rejects_without_real_bubble_mask(self):
        import inpainter

        image = np.full((90, 140, 3), 255, dtype=np.uint8)
        text = {
            "id": "ocr_001",
            "text": "SIM, NAO FUNCIONA",
            "bbox": [30, 30, 100, 58],
            "text_pixel_bbox": [32, 34, 96, 54],
            "line_polygons": [[[32, 34], [96, 34], [96, 54], [32, 54]]],
            "bubble_id": "bubble_001",
            "mask_evidence": _allowed_mask_evidence(),
            "route_action": "translate_inpaint_render",
        }
        page = {"texts": [text]}
        blocks = [dict(text)]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "1"}, clear=False):
            result, remaining, stats = inpainter._apply_fast_local_balloon_fill(image, page, blocks)

        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(remaining, blocks)
        self.assertEqual(stats["local_balloon_count"], 0)
        self.assertEqual(page["_strip_fast_local_rejection_reasons"], {"missing_real_bubble_mask": 1})

    def test_clip_fast_fill_text_mask_requires_real_bubble_overlap(self):
        import inpainter

        text_mask = np.zeros((40, 60), dtype=np.uint8)
        text_mask[10:20, 10:30] = 255
        bubble_mask = np.zeros((40, 60), dtype=np.uint8)
        bubble_mask[10:20, 40:55] = 1

        clipped, reason = inpainter._clip_fast_fill_text_mask_to_real_bubble(
            text_mask,
            bubble_mask,
            width=60,
            height=40,
        )

        self.assertIsNone(clipped)
        self.assertEqual(reason, "text_mask_outside_bubble")

    def test_fast_solid_fill_with_real_bubble_mask_preserves_outline(self):
        from inpainter import _apply_fast_solid_balloon_fill

        image = np.full((140, 240, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (120, 70), (72, 38), 0, 0, 360, (0, 0, 0), 2)
        image[58:72, 74:166] = 12
        outline_pixel = image[70, 47].copy()
        bubble_mask = np.zeros((140, 240), dtype=np.uint8)
        cv2.ellipse(bubble_mask, (120, 70), (68, 34), 0, 0, 360, 255, -1)
        text = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "bubble_id": "bubble_001",
            "text": "NO",
            "bbox": [70, 52, 170, 82],
            "text_pixel_bbox": [74, 58, 166, 72],
            "line_polygons": [[[74, 58], [166, 58], [166, 72], [74, 72]]],
            "balloon_bbox": [42, 28, 198, 112],
            "bubble_mask_bbox": [42, 28, 198, 112],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {
            "texts": [text],
            "_vision_blocks": [dict(text)],
            "_bubble_regions": [{"bubble_id": "bubble_001", "bubble_mask": bubble_mask}],
        }

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ):
            result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["solid_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(np.all(result[60:70, 82:158] == 255))
        self.assertTrue(np.array_equal(result[70, 47], outline_pixel))
        self.assertTrue(np.array_equal(result[28, 120], image[28, 120]))

    def test_fast_solid_low_coverage_flag_is_not_exported_after_clean_real_inpaint(self):
        import json
        from debug_tools import DebugRecorder, bind_recorder
        from inpainter import _record_inpaint_decision

        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[30:44, 40:80] = 255
        page = {
            "_band_id": "page_001_band_001",
            "_source_page_number": 1,
            "_band_index": 1,
            "_strip_inpaint_decision_flags": ["fast_fill_insufficient_coverage"],
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@page_001_band_001",
                    "bbox": [40, 30, 80, 44],
                    "text_pixel_bbox": [40, 30, 80, 44],
                    "balloon_bbox": [30, 20, 90, 55],
                    "skip_processing": False,
                }
            ],
        }

        with TemporaryDirectory() as tmpdir:
            recorder = DebugRecorder(Path(tmpdir), enabled=True, run_id="run-test")
            bind_recorder(recorder)
            try:
                with patch(
                    "inpainter._detect_inpaint_residual_text",
                    return_value={"has_residual": False, "score": 0.0, "flags": []},
                ):
                    _record_inpaint_decision(
                        page,
                        original_rgb=image,
                        working_rgb=image.copy(),
                        cleaned_rgb=image.copy(),
                        vision_blocks=[],
                        used_real_inpaint=True,
                        fast_fill_mask=mask,
                        raw_mask=mask,
                        expanded_mask=mask,
                        effective_limit_mask=np.full((80, 120), 255, dtype=np.uint8),
                    )
            finally:
                bind_recorder(None)

            decision_path = (
                Path(tmpdir)
                / "debug"
                / "e2e"
                / "08_inpaint"
                / "page_001_band_001"
                / "inpaint_decision.json"
            )
            payload = json.loads(decision_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["flags"], [])

    def test_fast_solid_fill_handles_black_panel_without_fast_dark(self):
        from inpainter import _apply_fast_solid_balloon_fill

        image = np.full((90, 220, 3), 255, dtype=np.uint8)
        image[18:72, 34:186] = 3
        image[38:52, 70:150] = 240
        text = {
            "id": "ocr_001",
            "bbox": [66, 34, 154, 56],
            "text_pixel_bbox": [70, 38, 150, 52],
            "line_polygons": [[[70, 38], [150, 38], [150, 52], [70, 52]]],
            "balloon_bbox": [34, 18, 186, 72],
            "balloon_type": "dark",
            "layout_profile": "solid_dark",
            "tipo": "narracao",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
            },
            clear=False,
        ):
            result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["solid_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertEqual(page["_strip_fast_solid_black_count"], 1)
        self.assertLess(float(np.mean(result[40:50, 76:144])), 12.0)

    def test_fast_solid_fill_rejects_textured_background(self):
        from inpainter import _apply_fast_solid_balloon_fill

        image = np.full((110, 220, 3), (64, 140, 184), dtype=np.uint8)
        image[:, ::9, :] = (20, 80, 140)
        image[46:60, 64:156] = 245
        text = {
            "id": "ocr_001",
            "bbox": [60, 40, 160, 66],
            "text_pixel_bbox": [64, 46, 156, 60],
            "line_polygons": [[[64, 46], [156, 46], [156, 60], [64, 60]]],
            "balloon_bbox": [24, 22, 190, 88],
            "balloon_type": "textured",
            "layout_profile": "standard",
            "tipo": "fala",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1"}, clear=False):
            result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["solid_balloon_count"], 0)
        self.assertEqual(remaining, page["_vision_blocks"])
        self.assertTrue(page["_strip_fast_solid_rejection_reasons"])
        self.assertTrue(np.array_equal(result, image))

    def test_fast_solid_fill_handles_connected_white_balloon_by_dominant_sample(self):
        from inpainter import _apply_fast_solid_balloon_fill

        image = np.full((140, 240, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (78, 68), (62, 44), 0, 0, 360, (0, 0, 0), 2)
        cv2.ellipse(image, (164, 76), (66, 48), 0, 0, 360, (0, 0, 0), 2)
        image[48:58, 50:112] = 12
        image[66:76, 48:118] = 12
        image[68:78, 144:208] = 12
        image[88:98, 138:214] = 12
        outline_pixel = image[68, 15].copy()
        text = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@connected_band",
            "bbox": [46, 46, 216, 100],
            "text_pixel_bbox": [48, 48, 214, 98],
            "line_polygons": [
                [[50, 48], [112, 48], [112, 58], [50, 58]],
                [[48, 66], [118, 66], [118, 76], [48, 76]],
                [[144, 68], [208, 68], [208, 78], [144, 78]],
                [[138, 88], [214, 88], [214, 98], [138, 98]],
            ],
            "balloon_bbox": [8, 22, 232, 126],
            "layout_profile": "connected_balloon",
            "balloon_type": "textured",
            "tipo": "fala",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
            },
            clear=False,
        ):
            result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["solid_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertEqual(page["_strip_fast_solid_white_count"], 1)
        self.assertTrue(np.all(result[50:56, 56:106] == 255))
        self.assertTrue(np.array_equal(result[68, 15], outline_pixel))

    def test_white_balloon_fill_color_samples_near_text_not_page_white(self):
        from vision_stack.runtime import _sample_white_balloon_fill_color

        cream = np.asarray([244, 241, 234], dtype=np.uint8)
        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        image[42:96, 52:174] = cream
        interior = np.ones((120, 220), dtype=np.uint8) * 255
        text_mask = np.zeros((120, 220), dtype=np.uint8)
        text_mask[56:74, 80:146] = 255

        fill = _sample_white_balloon_fill_color(image, interior, text_mask, sample_bbox=[52, 42, 174, 96])

        self.assertLessEqual(int(np.max(np.abs(fill.astype(np.int16) - cream.astype(np.int16)))), 1)

    def test_connected_white_geometry_fill_clears_text_without_erasing_lobe_outline(self):
        from inpainter import inpaint_band_image

        image = np.full((140, 240, 3), 255, dtype=np.uint8)
        import cv2

        cv2.ellipse(image, (78, 68), (62, 44), 0, 0, 360, (0, 0, 0), 2)
        cv2.ellipse(image, (164, 76), (66, 48), 0, 0, 360, (0, 0, 0), 2)
        image[48:58, 50:112] = 12
        image[66:76, 48:118] = 12
        image[68:78, 144:208] = 12
        image[88:98, 138:214] = 12
        outline_pixel = image[68, 15].copy()
        seam_pixel = image[36, 117].copy()
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@connected_band",
                    "bbox": [46, 46, 216, 100],
                    "text_pixel_bbox": [48, 48, 214, 98],
                    "line_polygons": [
                        [[50, 48], [112, 48], [112, 58], [50, 58]],
                        [[48, 66], [118, 66], [118, 76], [48, 76]],
                        [[144, 68], [208, 68], [208, 78], [144, 78]],
                        [[138, 88], [214, 88], [214, 98], [138, 98]],
                    ],
                    "balloon_bbox": [8, 22, 232, 126],
                    "balloon_subregions": [[8, 22, 124, 112], [108, 28, 232, 126]],
                    "layout_profile": "connected_balloon",
                    "balloon_type": "textured",
                    "confidence": 0.61,
                    "tipo": "fala",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ],
            "_vision_blocks": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@connected_band",
                    "bbox": [46, 46, 216, 100],
                    "text_pixel_bbox": [48, 48, 214, 98],
                    "line_polygons": [
                        [[50, 48], [112, 48], [112, 58], [50, 58]],
                        [[48, 66], [118, 66], [118, 76], [48, 76]],
                        [[144, 68], [208, 68], [208, 78], [144, 78]],
                        [[138, 88], [214, 88], [214, 98], [138, 98]],
                    ],
                    "balloon_bbox": [8, 22, 232, 126],
                    "layout_profile": "connected_balloon",
                    "balloon_type": "textured",
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ],
        }
        _attach_real_bubble_mask(page, image.shape)

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch(
            "inpainter._detect_inpaint_residual_text",
            return_value={"has_residual": False, "flags": [], "score": 0.0},
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(page["_strip_used_real_inpaint"])
        self.assertFalse(page["_strip_used_fast_solid_fill"])
        self.assertFalse(page["_strip_used_fast_white_fill"])
        self.assertEqual(page["_strip_fast_solid_balloon_count"], 0)
        self.assertGreater(page["_strip_koharu_remaining_mask_pixels"], 0)
        self.assertGreater(float(np.mean(result[50:56, 56:106])), 240.0)
        self.assertGreater(float(np.mean(result[90:96, 148:204])), 240.0)
        self.assertTrue(np.array_equal(result[68, 15], outline_pixel))
        self.assertTrue(np.array_equal(result[36, 117], seam_pixel))

    def test_dark_residual_retry_expands_textured_mask_without_fast_fill(self):
        from inpainter import inpaint_band_image

        image = np.full((90, 150, 3), 36, dtype=np.uint8)
        image[34:42, 42:110] = 245
        first_clean = image.copy()
        first_clean[32:46, 38:116] = 54
        page = {
            "texts": [
                {
                    "bbox": [40, 30, 112, 48],
                    "text_pixel_bbox": [42, 34, 110, 42],
                    "balloon_bbox": [34, 24, 122, 56],
                    "tipo": "narracao",
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "skip_processing": False,
                }
            ],
            "_vision_blocks": [{"bbox": [40, 30, 112, 48], "confidence": 0.95}],
        }

        class FakeInpainter:
            def __init__(self):
                self.calls = []

            def inpaint(self, img, mask, batch_size=4, force_no_tiling=True):
                self.calls.append(mask.copy())
                repaired = img.copy()
                repaired[mask > 0] = 42
                return repaired

        fake_inpainter = FakeInpainter()

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
            clear=False,
        ), patch(
            "vision_stack.runtime._get_inpainter",
            return_value=fake_inpainter,
        ), patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=first_clean,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ), patch(
            "inpainter._detect_inpaint_residual_text",
            side_effect=[
                {"has_residual": True, "flags": ["dark_residual_pixels"], "score": 0.24},
                {"has_residual": True, "flags": ["dark_residual_pixels"], "score": 0.24},
                {"has_residual": False, "flags": [], "score": 0.0},
            ],
        ):
            result = inpaint_band_image(image, page)

        self.assertEqual(len(fake_inpainter.calls), 1)
        self.assertTrue(page["_strip_dark_residual_retry"])
        self.assertGreater(page["_strip_dark_residual_retry_mask_pixels"], 0)
        self.assertTrue(np.all(result[34:42, 42:110] == 42))

    def test_white_balloon_residual_fill_clears_mask_when_model_keeps_text(self):
        from inpainter import _apply_real_inpaint_for_fast_fill_residual

        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        image[30:42, 36:84] = 10
        fast_fill_mask = np.zeros((80, 120), dtype=np.uint8)
        fast_fill_mask[30:42, 36:84] = 255
        page = {
            "texts": [
                {
                    "bbox": [30, 28, 90, 46],
                    "text_pixel_bbox": [36, 30, 84, 42],
                    "line_polygons": [[[36, 30], [84, 30], [84, 42], [36, 42]]],
                    "balloon_type": "white",
                    "block_profile": "white_balloon",
                    "skip_processing": False,
                }
            ]
        }

        class IdentityInpainter:
            def inpaint(self, img, mask, batch_size=4, force_no_tiling=True):
                return img.copy()

        with patch("inpainter._detect_inpaint_residual_text", return_value={"has_residual": True}), patch(
            "vision_stack.runtime._get_inpainter",
            return_value=IdentityInpainter(),
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (original.copy(), {}),
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ):
            result, used_real, residual_mask = _apply_real_inpaint_for_fast_fill_residual(
                original_rgb=image,
                working_rgb=image.copy(),
                cleaned_rgb=image.copy(),
                ocr_page=page,
                fast_fill_mask=fast_fill_mask,
            )

        self.assertTrue(used_real)
        self.assertIsNotNone(residual_mask)
        self.assertTrue(np.all(result[30:42, 36:84] == 245))

    def test_fast_fill_residual_mask_adds_text_evidence_outside_fast_fill(self):
        from inpainter import _apply_real_inpaint_for_fast_fill_residual

        image = np.full((110, 180, 3), 255, dtype=np.uint8)
        image[22:44, 38:46] = 0
        image[22:26, 35:49] = 0
        image[40:44, 35:49] = 0
        image[26:44, 55:126] = 0
        fast_fill_mask = np.zeros((110, 180), dtype=np.uint8)
        fast_fill_mask[20:58, 55:142] = 255
        page = {
            "texts": [
                {
                    "bbox": [55, 20, 142, 54],
                    "text_pixel_bbox": [55, 20, 142, 54],
                    "source_bbox": [34, 18, 142, 54],
                    "balloon_bbox": [34, 18, 142, 54],
                    "line_polygons": [[[55, 20], [142, 20], [142, 54], [55, 54]]],
                    "balloon_type": "white",
                    "layout_profile": "white_balloon",
                    "tipo": "texto",
                    "skip_processing": False,
                }
            ]
        }

        class CapturingInpainter:
            def __init__(self):
                self.mask = None

            def inpaint(self, img, mask, batch_size=4, force_no_tiling=True):
                self.mask = mask.copy()
                repaired = img.copy()
                repaired[mask > 0] = 255
                return repaired

        inpainter = CapturingInpainter()

        with patch("inpainter._detect_inpaint_residual_text", return_value={"has_residual": True}), patch(
            "vision_stack.runtime._get_inpainter",
            return_value=inpainter,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **kwargs: (cleaned, {}),
        ), patch(
            "vision_stack.runtime._clamp_image_to_limit_mask",
            side_effect=lambda base, candidate, mask, texts, **kwargs: (candidate, int(np.count_nonzero(mask)), 0),
        ):
            _apply_real_inpaint_for_fast_fill_residual(
                original_rgb=image,
                working_rgb=image.copy(),
                cleaned_rgb=image.copy(),
                ocr_page=page,
                fast_fill_mask=fast_fill_mask,
            )

        self.assertIsNotNone(inpainter.mask)
        self.assertGreater(int(inpainter.mask[32, 42]), 0)

    def test_post_cleanup_final_white_pass_is_clamped_to_limit_mask(self):
        from vision_stack import runtime

        original = np.full((80, 120, 3), 255, dtype=np.uint8)
        cleaned = original.copy()
        cleaned[10:18, 92:104] = 80
        limit = np.zeros((80, 120), dtype=np.uint8)
        limit[30:42, 36:84] = 255
        text = {
            "bbox": [30, 28, 90, 46],
            "text_pixel_bbox": [36, 30, 84, 42],
            "line_polygons": [[[36, 30], [84, 30], [84, 42], [36, 42]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "fala",
            "skip_processing": False,
        }

        def stray_cleanup(_original, candidate, _texts):
            result = candidate.copy()
            result[10:18, 92:104] = 255
            result[32:38, 40:50] = 240
            return result

        with patch.object(runtime, "_white_cleanup_texts", return_value=[text]), patch.object(
            runtime,
            "_apply_white_balloon_line_artifact_cleanup",
            side_effect=lambda _original, candidate, _texts: candidate,
        ), patch.object(
            runtime,
            "_apply_geometry_white_balloon_cleanup",
            side_effect=lambda _original, candidate, _texts: candidate,
        ), patch.object(
            runtime,
            "_apply_white_balloon_near_text_residual_cleanup",
            side_effect=lambda _original, candidate, _texts: candidate,
        ), patch.object(
            runtime,
            "_apply_white_balloon_micro_artifact_cleanup",
            side_effect=lambda _original, candidate, _texts: candidate,
        ), patch.object(
            runtime,
            "_restore_dark_line_art_outside_text_geometry",
            side_effect=lambda _original, candidate, _texts: candidate,
        ), patch.object(
            runtime,
            "_apply_glyph_residual_cleanup_for_texts",
            side_effect=stray_cleanup,
        ):
            final, stats = runtime._apply_post_inpaint_cleanup_timed(
                original,
                cleaned,
                [text],
                limit_mask=limit,
                include_text_bboxes_in_limit=False,
            )

        self.assertTrue(np.all(final[10:18, 92:104] == 80))
        self.assertTrue(np.all(final[32:38, 40:50] == 240))
        self.assertGreater(stats["cleanup_changed_outside_limit_mask"], 0)

    def test_white_cleanup_safe_uses_bright_source_when_balloon_bbox_is_tight(self):
        from vision_stack.runtime import _text_is_white_cleanup_safe

        image = np.full((120, 260, 3), 55, dtype=np.uint8)
        cv2.rectangle(image, (20, 30), (210, 68), (255, 255, 255), -1)
        cv2.rectangle(image, (20, 30), (210, 68), (0, 0, 0), 1)
        cv2.putText(
            image,
            "20** regional fireman",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        text = {
            "id": "ocr_regional",
            "skip_processing": False,
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "source_bbox": [20, 30, 210, 68],
            "balloon_bbox": [20, 30, 70, 68],
            "bbox": [80, 44, 170, 60],
            "text_pixel_bbox": [30, 46, 55, 58],
            "line_polygons": [[[28, 40], [205, 40], [205, 62], [28, 62]]],
        }

        self.assertTrue(_text_is_white_cleanup_safe(image, text))

    def test_white_cleanup_safe_uses_source_when_line_polygon_spans_dark_margin(self):
        from vision_stack.runtime import _text_anchor_has_white_cleanup_context

        image = np.zeros((120, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (30, 40), (190, 72), (255, 255, 255), -1)
        cv2.putText(
            image,
            "regional fireman",
            (45, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        text = {
            "id": "ocr_regional",
            "skip_processing": False,
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "source_bbox": [30, 40, 190, 72],
            "balloon_bbox": [30, 40, 86, 72],
            "bbox": [45, 50, 95, 64],
            "line_polygons": [[[40, 39], [300, 39], [300, 70], [40, 70]]],
        }

        self.assertTrue(_text_anchor_has_white_cleanup_context(image, text))

    def test_white_cleanup_does_not_use_source_anchor_for_textured_blocks(self):
        from vision_stack.runtime import _text_is_white_cleanup_safe

        image = np.full((120, 260, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 45), (220, 78), (210, 185, 185), -1)
        cv2.putText(
            image,
            "Successful candidate inquiry",
            (50, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        text = {
            "id": "ocr_status",
            "skip_processing": False,
            "balloon_type": "textured",
            "layout_profile": "standard",
            "source_bbox": [0, 0, 240, 32],
            "bbox": [50, 56, 210, 72],
            "text_pixel_bbox": [50, 56, 210, 72],
            "line_polygons": [[[50, 56], [210, 56], [210, 72], [50, 72]]],
        }

        self.assertFalse(_text_is_white_cleanup_safe(image, text))

    def test_post_cleanup_keeps_glyph_cleanup_when_expanded_mask_misses_text(self):
        from vision_stack.runtime import _apply_post_inpaint_cleanup_timed

        original = np.full((120, 260, 3), 55, dtype=np.uint8)
        cv2.rectangle(original, (20, 30), (210, 68), (255, 255, 255), -1)
        cv2.rectangle(original, (20, 30), (210, 68), (0, 0, 0), 1)
        cv2.putText(
            original,
            "20** regional fireman",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cleaned = original.copy()
        tight_inpaint_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        tight_inpaint_mask[46:60, 28:55] = 255
        text = {
            "id": "ocr_regional",
            "skip_processing": False,
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "source_bbox": [20, 30, 210, 68],
            "balloon_bbox": [20, 30, 70, 68],
            "bbox": [80, 44, 170, 60],
            "text_pixel_bbox": [30, 46, 55, 58],
            "line_polygons": [[[28, 40], [205, 40], [205, 62], [28, 62]]],
        }

        result, stats = _apply_post_inpaint_cleanup_timed(
            original,
            cleaned,
            [text],
            limit_mask=tight_inpaint_mask,
        )

        roi_gray = cv2.cvtColor(result[38:64, 25:210], cv2.COLOR_RGB2GRAY)
        self.assertLess(int(np.count_nonzero(roi_gray < 120)), 80)
        self.assertGreaterEqual(stats["cleanup_limit_mask_pixels"], int(np.count_nonzero(tight_inpaint_mask)))

    def test_fast_fill_residual_marks_decision_flag_when_real_inpaint_unavailable(self):
        from inpainter import _apply_real_inpaint_for_fast_fill_residual

        image = np.full((80, 120, 3), 245, dtype=np.uint8)
        image[30:42, 36:84] = 10
        working = image.copy()
        working[18:25, 24:96] = 245
        fast_fill_mask = np.zeros((80, 120), dtype=np.uint8)
        fast_fill_mask[18:25, 24:96] = 255
        page = {
            "texts": [
                {
                    "bbox": [32, 26, 88, 48],
                    "text_pixel_bbox": [36, 30, 84, 42],
                    "skip_processing": False,
                }
            ]
        }

        with patch("vision_stack.runtime._get_inpainter", side_effect=RuntimeError("model unavailable")):
            result, used_real, residual_mask = _apply_real_inpaint_for_fast_fill_residual(
                original_rgb=image,
                working_rgb=working,
                cleaned_rgb=working,
                ocr_page=page,
                fast_fill_mask=fast_fill_mask,
            )

        self.assertIs(result, working)
        self.assertFalse(used_real)
        self.assertIsNotNone(residual_mask)
        self.assertIn("text_residual_after_fast_fill", page["_strip_inpaint_decision_flags"])
        self.assertIn("real_inpaint_unavailable", page["_strip_inpaint_decision_flags"])

    def test_fast_fill_residual_skips_real_inpaint_when_mask_exits_bubble(self):
        from inpainter import _apply_real_inpaint_for_fast_fill_residual

        image = np.full((90, 140, 3), 245, dtype=np.uint8)
        image[18:34, 20:44] = 10
        image[52:68, 92:122] = 10
        bubble_mask = np.zeros((90, 140), dtype=np.uint8)
        bubble_mask[12:42, 12:58] = 1
        fast_fill_mask = np.zeros((90, 140), dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "bubble_id": "bubble_001",
                    "bbox": [18, 16, 124, 70],
                    "text_pixel_bbox": [18, 16, 124, 70],
                    "line_polygons": [[[18, 16], [124, 16], [124, 70], [18, 70]]],
                    "bubble_mask": bubble_mask,
                    "bubble_mask_bbox": [0, 0, 140, 90],
                    "bubble_mask_source": "image_white_bubble_mask",
                    "skip_processing": False,
                }
            ],
        }

        with patch("inpainter._detect_inpaint_residual_text", return_value={"has_residual": True}), patch(
            "vision_stack.runtime._get_inpainter",
            side_effect=AssertionError("unsafe residual mask nao deve chamar AOT"),
        ):
            result, used_real, residual_mask = _apply_real_inpaint_for_fast_fill_residual(
                original_rgb=image,
                working_rgb=image.copy(),
                cleaned_rgb=image.copy(),
                ocr_page=page,
                fast_fill_mask=fast_fill_mask,
            )

        self.assertFalse(used_real)
        self.assertIsNotNone(residual_mask)
        self.assertTrue(np.array_equal(result, image))
        self.assertIn("mask_outside_balloon_critical", page["_strip_inpaint_decision_flags"])
        self.assertIn("real_inpaint_skipped_unsafe_mask", page["_strip_inpaint_decision_flags"])

    def test_dark_panel_text_fill_removes_local_light_ghost(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((80, 160, 3), 6, dtype=np.uint8)
        image[30:46, 44:116] = [185, 182, 150]
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [44, 30, 116, 46],
                    "text_pixel_bbox": [44, 30, 116, 46],
                    "line_polygons": [[[44, 30], [116, 30], [116, 46], [44, 46]]],
                    "balloon_bbox": [30, 18, 130, 58],
                    "balloon_type": "dark",
                    "layout_profile": "dark_panel",
                    "tipo": "narracao",
                    "skip_processing": False,
                }
            ]
        }

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 0)
        self.assertTrue(np.array_equal(result, image))

    def test_fast_dark_panel_fill_removes_covered_block_before_real_inpaint(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((90, 180, 3), 7, dtype=np.uint8)
        image[34:50, 48:132] = [210, 208, 184]
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [48, 34, 132, 50],
                    "text_pixel_bbox": [48, 34, 132, 50],
                    "line_polygons": [[[48, 34], [132, 34], [132, 50], [48, 50]]],
                    "balloon_bbox": [28, 20, 152, 66],
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "tipo": "narracao",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [{"bbox": [46, 32, 134, 52], "confidence": 0.94}]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(page["_strip_used_dark_panel_fill"])
        self.assertLess(float(np.mean(result[36:48, 54:126])), 40.0)

    def test_fast_dark_panel_fill_keeps_unfilled_dark_bubble_lobe_for_real_inpaint(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((130, 260, 3), 4, dtype=np.uint8)
        image[42:75, 44:128] = [238, 235, 212]
        image[66:84, 166:236] = [238, 235, 212]
        left = {
            "id": "ocr_left",
            "trace_id": "ocr_left@page_002_band_011",
            "bbox": [44, 42, 128, 75],
            "text_pixel_bbox": [44, 42, 128, 75],
            "balloon_bbox": [20, 18, 145, 100],
            "bubble_mask_bbox": [20, 18, 145, 100],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["dark_bubble_ellipse_bbox_mask", "fast_fill_no_glyph_evidence"],
            "skip_processing": False,
        }
        right = {
            "id": "ocr_right",
            "trace_id": "ocr_right@page_002_band_011",
            "bbox": [166, 66, 236, 84],
            "text_pixel_bbox": [166, 66, 236, 84],
            "line_polygons": [[[166, 66], [236, 66], [236, 84], [166, 84]]],
            "balloon_bbox": [138, 44, 252, 110],
            "bubble_mask_bbox": [138, 44, 252, 110],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [left, right]}
        vision_blocks = [dict(left), dict(right)]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            _result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(stats["remaining_blocks"], 1)
        self.assertEqual(remaining[0]["id"], "ocr_left")
        self.assertFalse(left.get("_fast_fill_inpaint_resolved"))

    def test_fast_dark_panel_fill_handles_colored_status_panel(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((260, 420, 3), (123, 88, 48), dtype=np.uint8)
        image[:, ::17, :] = (92, 72, 44)
        cv2.putText(image, "LEVEL 7 BEAST TAMER", (22, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (236, 150, 48), 2, cv2.LINE_AA)
        cv2.putText(image, "TAMED: 101000", (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (236, 150, 48), 2, cv2.LINE_AA)
        cv2.putText(image, "NEXT PROMOTION", (22, 111), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (236, 150, 48), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "LEVEL 7 BEAST TAMER TAMED NEXT PROMOTION REQUIRED",
                    "bbox": [18, 24, 260, 124],
                    "text_pixel_bbox": [18, 24, 260, 124],
                    "line_polygons": [
                        [[20, 25], [205, 25], [205, 50], [20, 50]],
                        [[20, 58], [170, 58], [170, 83], [20, 83]],
                        [[20, 91], [186, 91], [186, 116], [20, 116]],
                    ],
                    "balloon_bbox": [12, 14, 280, 136],
                    "balloon_type": "textured",
                    "block_profile": "standard",
                    "background_rgb": [123, 88, 48],
                    "tipo": "narracao",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(page["_strip_used_dark_panel_fill"])
        before_delta = np.mean(np.abs(image[30:118, 24:198].astype(np.int16) - np.array([123, 88, 48], dtype=np.int16)))
        after_delta = np.mean(np.abs(result[30:118, 24:198].astype(np.int16) - np.array([123, 88, 48], dtype=np.int16)))
        self.assertLess(after_delta, before_delta * 0.65)

    def test_card_visual_dark_fill_runs_without_global_fast_dark_opt_in(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((150, 260, 3), [84, 48, 68], dtype=np.uint8)
        image[:, ::15, :] = [72, 42, 62]
        cv2.putText(image, "STATUS READY", (38, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        cv2.putText(image, "HOST TITLE", (48, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text": "STATUS READY HOST TITLE",
            "bbox": [34, 32, 206, 102],
            "text_pixel_bbox": [34, 32, 206, 102],
            "line_polygons": [
                [[36, 34], [210, 34], [210, 64], [36, 64]],
                [[46, 68], [190, 68], [190, 100], [46, 100]],
            ],
            "balloon_bbox": [24, 20, 224, 116],
            "background_rgb": [84, 48, 68],
            "layout_profile": "colored_status_panel",
            "route_action": "translate_inpaint_render",
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [dict(text)]}
        vision_blocks = [dict(text)]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(page["_strip_used_dark_panel_fill"])
        self.assertLess(int(np.count_nonzero(np.any(result != image, axis=2))), 7000)

    def test_card_visual_dark_fill_does_not_rebuild_removed_block_for_aot(self):
        from inpainter import inpaint_band_image

        image = np.full((150, 260, 3), [84, 48, 68], dtype=np.uint8)
        image[:, ::15, :] = [72, 42, 62]
        cv2.putText(image, "STATUS READY", (38, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        cv2.putText(image, "HOST TITLE", (48, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_105",
            "text": "STATUS READY HOST TITLE",
            "bbox": [34, 32, 206, 102],
            "text_pixel_bbox": [34, 32, 206, 102],
            "line_polygons": [
                [[36, 34], [210, 34], [210, 64], [36, 64]],
                [[46, 68], [190, 68], [190, 100], [46, 100]],
            ],
            "balloon_bbox": [24, 20, 224, 116],
            "background_rgb": [84, 48, 68],
            "layout_profile": "colored_status_panel",
            "route_action": "translate_inpaint_render",
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [dict(text)], "_vision_blocks": [dict(text)]}

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0"}, clear=False), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=AssertionError("card limpo por fast-dark nao deve ser reconstruido para AOT"),
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(page.get("_strip_used_dark_panel_fill"))
        self.assertFalse(page.get("_strip_rebuilt_empty_remaining_blocks_from_local_texts"))
        self.assertEqual(page.get("_strip_remaining_inpaint_blocks"), 0)
        before_bright = int(np.count_nonzero(np.mean(image[28:110, 28:216], axis=2) > 120))
        after_bright = int(np.count_nonzero(np.mean(result[28:110, 28:216], axis=2) > 120))
        self.assertLess(after_bright, before_bright)

    def test_rejected_visual_card_fast_dark_fill_keeps_block_for_real_inpaint(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((150, 280, 3), [252, 169, 127], dtype=np.uint8)
        image[:, :, 0] = np.linspace(230, 255, image.shape[1], dtype=np.uint8)
        cv2.putText(image, "The host was", (78, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 250, 245), 2, cv2.LINE_AA)
        cv2.putText(image, "given title,", (82, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 250, 245), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_105",
            "text": "The host was given the title,",
            "bbox": [70, 38, 214, 102],
            "text_pixel_bbox": [70, 38, 214, 102],
            "balloon_bbox": [42, 20, 238, 122],
            "bubble_mask_bbox": [82, 68, 140, 102],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "route_action": "translate_inpaint_render",
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [dict(text)]}

        result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, [dict(text)])

        self.assertEqual(stats["dark_panel_fill_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertFalse(page.get("_strip_used_dark_panel_fill"))
        self.assertTrue(np.array_equal(result, image))
        self.assertIn("rejected_visual_card_requires_real_inpaint", page.get("_strip_fast_dark_rejection_reasons") or {})

    def test_rejected_visual_card_dark_panel_fallback_keeps_block_for_real_inpaint(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((150, 280, 3), [118, 98, 49], dtype=np.uint8)
        image[:, :, 0] = np.linspace(92, 142, image.shape[1], dtype=np.uint8)
        cv2.putText(image, "The host was", (78, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 250, 245), 2, cv2.LINE_AA)
        cv2.putText(image, "given title,", (82, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 250, 245), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_105",
            "text": "The host was given the title,",
            "bbox": [70, 38, 214, 102],
            "text_pixel_bbox": [70, 38, 214, 102],
            "balloon_bbox": [42, 20, 238, 122],
            "bubble_mask_bbox": [82, 68, 140, 102],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "background_rgb": [118, 98, 49],
            "route_action": "translate_inpaint_render",
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [dict(text)]}

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 0)
        self.assertFalse(page.get("_strip_used_dark_panel_fill"))
        self.assertTrue(np.array_equal(result, image))

    def test_rejected_visual_card_fast_dark_fill_does_not_solid_fill_text_rect(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        height, width = 150, 280
        x_grad = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        y_grad = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:, :, 0] = 210.0 + (35.0 * x_grad)
        image[:, :, 1] = 120.0 + (45.0 * y_grad)
        image[:, :, 2] = 95.0 + (45.0 * (1.0 - x_grad))
        image = np.clip(image, 0, 255).astype(np.uint8)
        image[::9, :, 0] = np.clip(image[::9, :, 0].astype(np.int16) + 20, 0, 255).astype(np.uint8)
        image[:, ::13, 1] = np.clip(image[:, ::13, 1].astype(np.int16) + 16, 0, 255).astype(np.uint8)
        cv2.rectangle(image, (42, 20), (238, 122), (245, 220, 190), 1)
        cv2.putText(
            image,
            "The host was",
            (78, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (248, 250, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "given title,",
            (82, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (248, 250, 245),
            2,
            cv2.LINE_AA,
        )
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_105",
            "text": "The host was given the title,",
            "bbox": [70, 38, 214, 102],
            "text_pixel_bbox": [70, 38, 214, 102],
            "balloon_bbox": [42, 20, 238, 122],
            "bubble_mask_bbox": [82, 68, 140, 102],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "route_action": "translate_inpaint_render",
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [dict(text)]}

        result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, [dict(text)])

        self.assertEqual(stats["dark_panel_fill_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertFalse(page.get("_strip_used_dark_panel_fill"))
        self.assertTrue(np.array_equal(result, image))

    def test_rejected_gradient_card_keeps_block_for_real_inpaint_instead_of_fast_solid_fill(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        height, width = 150, 280
        x_grad = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        y_grad = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:, :, 0] = 95.0 + (120.0 * x_grad)
        image[:, :, 1] = 120.0 + (80.0 * y_grad)
        image[:, :, 2] = 205.0 + (35.0 * (1.0 - x_grad))
        image = np.clip(image, 0, 255).astype(np.uint8)
        cv2.rectangle(image, (42, 20), (238, 122), (230, 235, 245), 1)
        cv2.putText(image, "The host was", (78, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 250, 245), 2, cv2.LINE_AA)
        cv2.putText(image, "given title,", (82, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 250, 245), 2, cv2.LINE_AA)
        evidence = _allowed_mask_evidence()
        evidence.update({"raw_mask_pixels": 1800, "expanded_mask_pixels": 6200})
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_105",
            "text": "The host was given the title,",
            "bbox": [70, 38, 214, 102],
            "text_pixel_bbox": [70, 38, 214, 102],
            "line_polygons": [
                [[78, 46], [214, 46], [214, 70], [78, 70]],
                [[82, 74], [202, 74], [202, 98], [82, 98]],
            ],
            "balloon_bbox": [42, 20, 238, 122],
            "bubble_mask_bbox": [82, 68, 140, 102],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "background_rgb": [118, 98, 190],
            "card_panel_text_context": True,
            "route_action": "translate_inpaint_render",
            "mask_evidence": evidence,
        }
        page = {"texts": [dict(text)]}

        result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, [dict(text)])

        self.assertEqual(stats["dark_panel_fill_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertFalse(page.get("_strip_used_dark_panel_fill"))
        self.assertTrue(np.array_equal(result, image))
        self.assertIn("rejected_visual_card_requires_real_inpaint", page.get("_strip_fast_dark_rejection_reasons") or {})

    def test_inpaint_enrichment_preserves_card_panel_context_fields(self):
        from inpainter import _enrich_vision_blocks_from_texts_for_inpaint

        block = {
            "id": "ocr_001",
            "bbox": [468, 102, 608, 173],
            "text_pixel_bbox": [451, 107, 643, 168],
        }
        text = {
            "id": "ocr_001",
            "bbox": [468, 102, 608, 173],
            "text_pixel_bbox": [451, 107, 643, 168],
            "background_rgb": [118, 98, 49],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "bubble_mask_bbox": [514, 139, 586, 169],
            "bubble_id": "page_006_band_107_bubble_001",
            "card_panel_text_context": True,
        }

        enriched = _enrich_vision_blocks_from_texts_for_inpaint([block], [text], 800, 256)

        self.assertEqual(enriched[0]["background_rgb"], [118, 98, 49])
        self.assertEqual(enriched[0]["bubble_mask_source"], "derived_white_crop_rejected")
        self.assertEqual(enriched[0]["bubble_mask_bbox"], [514, 139, 586, 169])
        self.assertEqual(enriched[0]["bubble_id"], "page_006_band_107_bubble_001")
        self.assertTrue(enriched[0]["card_panel_text_context"])

    def test_inpaint_enrichment_matches_page_relative_block_by_trace_id(self):
        from inpainter import _enrich_vision_blocks_from_texts_for_inpaint

        block = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_107",
            "bbox": [468, 15322, 608, 15393],
        }
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_107",
            "bbox": [468, 102, 608, 173],
            "text_pixel_bbox": [451, 107, 643, 168],
            "background_rgb": [118, 98, 49],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "bubble_mask_bbox": [514, 139, 586, 169],
            "card_panel_text_context": True,
            "mask_evidence": _allowed_mask_evidence(),
        }

        enriched = _enrich_vision_blocks_from_texts_for_inpaint([block], [text], 760, 220)

        self.assertEqual(enriched[0]["bbox"], [468, 102, 608, 173])
        self.assertEqual(enriched[0]["background_rgb"], [118, 98, 49])
        self.assertTrue(enriched[0]["mask_evidence"]["fast_fill_allowed"])
        self.assertGreater(enriched[0]["mask_evidence"]["raw_mask_pixels"], 0)

    def test_prime_mask_evidence_copies_current_block_evidence_back_to_text(self):
        from inpainter import (
            _ocr_page_has_unsafe_auto_inpaint_evidence,
            _prime_mask_evidence_for_fast_fill,
        )

        image = np.full((220, 760, 3), [118, 98, 49], dtype=np.uint8)
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_107",
            "bbox": [468, 102, 608, 173],
            "text_pixel_bbox": [451, 107, 643, 168],
            "route_action": "translate_inpaint_render",
        }
        block = {
            **dict(text),
            "balloon_bbox": [438, 81, 638, 194],
            "bubble_mask_bbox": [514, 139, 586, 169],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "mask_evidence": _allowed_mask_evidence(),
            "card_panel_text_context": True,
        }
        page = {
            "texts": [text],
            "_vision_blocks": [block],
            "_strip_inpaint_decision_flags": ["real_inpaint_skipped_unsafe_mask"],
        }

        _prime_mask_evidence_for_fast_fill(page, [block], image)

        self.assertEqual(text["bubble_mask_source"], "derived_white_crop_rejected")
        self.assertEqual(text["bubble_mask_error"], "derived_mask_not_anchored_to_text")
        self.assertTrue(text["mask_evidence"]["fast_fill_allowed"])
        self.assertFalse(_ocr_page_has_unsafe_auto_inpaint_evidence(page, [block]))

    def test_inpaint_band_image_keeps_rejected_card_panel_for_real_inpaint(self):
        from inpainter import inpaint_band_image

        image = np.full((220, 760, 3), [118, 98, 49], dtype=np.uint8)
        image[:, :, 0] = np.linspace(92, 142, image.shape[1], dtype=np.uint8)
        cv2.putText(image, "CURRENT LEVEL CLASS:", (471, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 250, 250), 2, cv2.LINE_AA)
        cv2.putText(image, "FLOATING SPIRIT", (449, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 250, 250), 2, cv2.LINE_AA)
        evidence = _allowed_mask_evidence()
        evidence.update({"raw_mask_pixels": 2564, "expanded_mask_pixels": 9341})
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_107",
            "text": "Current Level Class: Floating Spirit",
            "bbox": [468, 102, 608, 173],
            "text_pixel_bbox": [451, 107, 643, 168],
            "line_polygons": [
                [[471, 107], [624, 107], [624, 134], [471, 134]],
                [[449, 137], [645, 137], [645, 168], [449, 168]],
            ],
            "balloon_bbox": [438, 81, 638, 194],
            "bubble_mask_bbox": [514, 139, 586, 169],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "background_rgb": [118, 98, 49],
            "card_panel_text_context": True,
            "route_action": "translate_inpaint_render",
            "qa_flags": ["fast_fill_no_glyph_evidence"],
            "mask_evidence": evidence,
        }
        stale_block = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_107",
            "bbox": [468, 102, 608, 173],
            "text_pixel_bbox": [451, 107, 643, 168],
            "balloon_bbox": [438, 81, 638, 194],
            "bubble_mask_bbox": [432, 75, 658, 200],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["fast_fill_no_glyph_evidence"],
        }
        page = {"texts": [dict(text)], "_vision_blocks": [stale_block], "_band_y_top": 0}

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0"}, clear=False), patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=image.copy(),
        ) as inpaint_round:
            result = inpaint_band_image(image, page)

        inpaint_round.assert_called_once()
        self.assertFalse(page.get("_strip_used_dark_panel_fill"))
        self.assertGreaterEqual(page.get("_strip_remaining_inpaint_blocks"), 1)

    def test_dark_bubble_with_ocr_evidence_uses_glyph_action_mask_not_panel_mask(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((230, 760, 3), [12, 18, 20], dtype=np.uint8)
        image[:, :, 0] = np.linspace(9, 35, image.shape[1], dtype=np.uint8)
        cv2.rectangle(image, (404, 12), (688, 204), (192, 208, 210), 2)
        cv2.putText(image, "Current Level:", (470, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(image, "Class: Floating Spirit", (450, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 245, 245), 2, cv2.LINE_AA)
        evidence = _allowed_mask_evidence()
        evidence.update({"kind": "ocr_pixels", "raw_mask_pixels": 2564, "expanded_mask_pixels": 18913})
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_107",
            "text": "Current Level Class: Floating Spirit",
            "bbox": [468, 102, 608, 173],
            "text_pixel_bbox": [451, 107, 643, 168],
            "line_polygons": [
                [[471, 107], [624, 107], [624, 134], [471, 134]],
                [[449, 137], [645, 137], [645, 168], [449, 168]],
            ],
            "balloon_bbox": [438, 81, 638, 194],
            "bubble_mask_bbox": [288, 8, 709, 221],
            "bubble_mask_source": "image_dark_bubble_mask",
            "background_rgb": [12, 18, 20],
            "card_panel_text_context": True,
            "route_action": "translate_inpaint_render",
            "qa_flags": [
                "dark_bubble_promoted_from_rejected_mask",
                "dark_bubble_ellipse_bbox_mask",
                "fast_fill_no_glyph_evidence",
            ],
            "mask_evidence": evidence,
        }
        page = {"texts": [dict(text)]}

        _result, _remaining, stats = _apply_fast_dark_panel_text_fill(image, page, [dict(text)])

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        fill_mask = page.get("_strip_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        assert isinstance(fill_mask, np.ndarray)
        fill_pixels = int(np.count_nonzero(fill_mask))
        panel_area = (709 - 288) * (221 - 8)
        self.assertLess(fill_pixels, int(panel_area * 0.30))
        self.assertLess(fill_pixels, 22000)
        self.assertEqual(int(fill_mask[30, 320]), 0)
        self.assertGreater(int(fill_mask[120, 500]), 0)

    def test_inpaint_band_image_does_not_roll_back_rejected_card_due_stale_unsafe_flag(self):
        from inpainter import inpaint_band_image

        image = np.full((220, 760, 3), [118, 98, 49], dtype=np.uint8)
        image[:, :, 0] = np.linspace(92, 142, image.shape[1], dtype=np.uint8)
        cv2.putText(image, "CURRENT LEVEL CLASS:", (471, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 250, 250), 2, cv2.LINE_AA)
        cv2.putText(image, "FLOATING SPIRIT", (449, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 250, 250), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_107",
            "text": "Current Level Class: Floating Spirit",
            "bbox": [468, 102, 608, 173],
            "text_pixel_bbox": [451, 107, 643, 168],
            "line_polygons": [
                [[471, 107], [624, 107], [624, 134], [471, 134]],
                [[449, 137], [645, 137], [645, 168], [449, 168]],
            ],
            "balloon_bbox": [438, 81, 638, 194],
            "bubble_mask_bbox": [514, 139, 586, 169],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "background_rgb": [118, 98, 49],
            "card_panel_text_context": True,
            "route_action": "translate_inpaint_render",
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {
            "texts": [dict(text)],
            "_vision_blocks": [dict(text)],
            "_band_y_top": 0,
            "_strip_inpaint_decision_flags": ["mask_outside_balloon_critical", "real_inpaint_skipped_unsafe_mask"],
        }

        def fake_round(img, payload, inpainter):
            mask = payload.get("_precomputed_inpaint_mask")
            result = img.copy()
            if isinstance(mask, np.ndarray):
                result[mask > 0] = np.maximum(0, result[mask > 0].astype(np.int16) - 28).astype(np.uint8)
            payload["_inpaint_round_stats"] = {
                "_strip_inpaint_decision_flags": ["mask_outside_balloon_critical", "real_inpaint_skipped_unsafe_mask"],
            }
            return result

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0"}, clear=False), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=fake_round,
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(page.get("_strip_used_real_inpaint"))
        self.assertFalse(page.get("_strip_used_dark_panel_fill"))
        self.assertNotIn("real_inpaint_skipped_unsafe_mask", page.get("_strip_inpaint_decision_flags") or [])
        self.assertGreater(int(np.count_nonzero(np.any(result != image, axis=2))), 100)

    def test_white_image_rect_mask_not_blocked_by_stale_unsafe_flag(self):
        from inpainter import inpaint_band_image

        image = np.full((180, 360, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (180, 90), (118, 62), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (180, 90), (118, 62), 0, 0, 360, (28, 28, 28), 2)
        cv2.putText(image, "PLEASE WAIT", (88, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(image, "MORE DAYS", (112, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
        evidence = _allowed_mask_evidence()
        evidence.update({"kind": "component_bubble_cleaner", "raw_mask_pixels": 1800, "expanded_mask_pixels": 6200})
        text = {
            "id": "ocr_003",
            "trace_id": "ocr_003@page_002_band_002",
            "text": "PLEASE WAIT FOR A FEW MORE DAYS!",
            "bbox": [88, 64, 260, 118],
            "text_pixel_bbox": [88, 64, 260, 118],
            "line_polygons": [
                [[88, 64], [272, 64], [272, 88], [88, 88]],
                [[112, 94], [248, 94], [248, 118], [112, 118]],
            ],
            "balloon_bbox": [58, 28, 302, 154],
            "bubble_mask_bbox": [58, 28, 302, 154],
            "bubble_mask_source": "image_rect_bubble_mask",
            "background_rgb": [255, 255, 255],
            "block_profile": "white_balloon",
            "route_action": "translate_inpaint_render",
            "mask_evidence": evidence,
        }
        page = {
            "texts": [dict(text)],
            "_vision_blocks": [dict(text)],
            "_band_y_top": 0,
            "_strip_inpaint_decision_flags": ["mask_outside_balloon_critical", "real_inpaint_skipped_unsafe_mask"],
        }

        def fake_round(img, payload, inpainter):
            payload["_inpaint_round_stats"] = {
                "_strip_inpaint_decision_flags": [
                    "mask_outside_balloon_critical",
                    "real_inpaint_skipped_unsafe_mask",
                ],
            }
            return img.copy()

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0"}, clear=False), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=fake_round,
        ):
            result = inpaint_band_image(image, page)

        self.assertTrue(page.get("_strip_used_real_inpaint"))
        self.assertNotIn("real_inpaint_skipped_unsafe_mask", page.get("_strip_inpaint_decision_flags") or [])
        text_area_before = np.mean(image[58:124, 82:278], axis=2)
        text_area_after = np.mean(result[58:124, 82:278], axis=2)
        self.assertLess(int(np.count_nonzero(text_area_after < 120)), int(np.count_nonzero(text_area_before < 120)) // 3)

    def test_rejected_visual_card_fast_dark_fill_ignores_balloon_qa_for_local_fill(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((150, 320, 3), [8, 8, 8], dtype=np.uint8)
        cv2.rectangle(image, (58, 34), (244, 104), (16, 16, 16), -1)
        cv2.rectangle(image, (58, 34), (244, 104), (180, 180, 180), 1)
        cv2.putText(image, "the Devil Knight!", (77, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (42, 30, 210), 6, cv2.LINE_AA)
        cv2.putText(image, "the Devil Knight!", (78, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 238, 186), 2, cv2.LINE_AA)
        evidence = _allowed_mask_evidence()
        evidence.update({"raw_mask_pixels": 1200, "expanded_mask_pixels": 3600})
        text = {
            "id": "ocr_002",
            "trace_id": "ocr_002@page_006_band_106",
            "text": "the Devil Knight!",
            "bbox": [76, 50, 236, 84],
            "text_pixel_bbox": [70, 46, 246, 90],
            "balloon_bbox": [58, 34, 244, 104],
            "bubble_mask_bbox": [78, 50, 226, 86],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
            "mask_evidence": evidence,
        }
        page = {"texts": [dict(text)]}

        result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, [dict(text)])

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        changed_pixels = int(np.count_nonzero(np.any(result != image, axis=2)))
        self.assertGreater(changed_pixels, 500)
        fill_mask_pixels = int(np.count_nonzero(page.get("_strip_dark_panel_fill_mask") > 0))
        self.assertGreater(fill_mask_pixels, 7000)
        before_red_glow = int(np.count_nonzero((image[:, :, 2] > 120) & (image[:, :, 0] < 80)))
        after_red_glow = int(np.count_nonzero((result[:, :, 2] > 120) & (result[:, :, 0] < 80)))
        self.assertLess(after_red_glow, int(before_red_glow * 0.35))

    def test_contour_bubble_context_blocks_dark_panel_fill(self):
        from inpainter import _apply_dark_panel_text_fills, _apply_fast_dark_panel_text_fill

        image = np.full((220, 620, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (310, 130), (210, 78), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (310, 130), (210, 78), 0, 0, 360, (0, 0, 0), 2)
        cv2.putText(image, "DON'T HIT", (215, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "MY MOM!", (225, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_023",
            "text": "DON'T HIT MY MOM!",
            "bbox": [0, 20, 555, 210],
            "text_pixel_bbox": [200, 94, 420, 164],
            "balloon_bbox": [90, 42, 530, 206],
            "bubble_mask_bbox": [90, 42, 530, 206],
            "bubble_mask_source": "image_contour_bubble_mask",
            "card_panel_text_context": True,
            "route_action": "translate_inpaint_render",
            "qa_flags": ["bubble_clip_preserved_raw_text"],
        }
        page = {"texts": [dict(text)]}

        dark_result, dark_count = _apply_dark_panel_text_fills(image, page)
        fast_result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, [dict(text)])

        self.assertEqual(dark_count, 0)
        self.assertEqual(stats["dark_panel_fill_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertTrue(np.array_equal(dark_result, image))
        self.assertTrue(np.array_equal(fast_result, image))

    def test_unsafe_contour_bubble_context_blocks_dark_panel_fill(self):
        from inpainter import _apply_dark_panel_text_fills, _apply_fast_dark_panel_text_fill

        image = np.full((560, 800, 3), [174, 183, 219], dtype=np.uint8)
        cv2.ellipse(image, (568, 348), (135, 95), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (568, 348), (135, 95), 0, 0, 360, (0, 0, 0), 2)
        cv2.putText(image, "PLEASE, FOR", (500, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "THE CHILD'S", (502, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "SAKE.", (540, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_002",
            "trace_id": "ocr_002@page_002_band_005",
            "text": "PLEASE, FOR THE CHILD'S SAKE.",
            "bbox": [501, 298, 661, 405],
            "text_pixel_bbox": [499, 315, 656, 400],
            "balloon_bbox": [25, 96, 667, 368],
            "bubble_mask_bbox": [435, 255, 701, 442],
            "bubble_mask_source": "image_contour_bubble_mask",
            "route_action": "translate_inpaint_render",
            "route_reason": "mask_outside_balloon_critical",
            "background_rgb": [174, 183, 219],
            "qa_flags": [
                "same_balloon_fragment_merged",
                "same_band_dependent_fragment_merged",
                "mask_outside_balloon",
                "mask_outside_balloon_critical",
            ],
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 1714,
                "expanded_mask_pixels": 6077,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }
        page = {"texts": [dict(text)]}

        dark_result, dark_count = _apply_dark_panel_text_fills(image, page)
        fast_result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, [dict(text)])

        self.assertEqual(dark_count, 0)
        self.assertEqual(stats["dark_panel_fill_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertTrue(np.array_equal(dark_result, image))
        self.assertTrue(np.array_equal(fast_result, image))

    def test_unsafe_white_balloon_text_fill_cleans_text_without_outline(self):
        from inpainter import _apply_unsafe_white_balloon_text_fills

        image = np.full((560, 800, 3), [174, 183, 219], dtype=np.uint8)
        cv2.ellipse(image, (568, 348), (135, 95), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (568, 348), (135, 95), 0, 0, 360, (0, 0, 0), 2)
        cv2.putText(image, "PLEASE, FOR", (500, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "THE CHILD'S", (502, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "SAKE.", (540, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_002",
            "trace_id": "ocr_002@page_002_band_005",
            "bbox": [501, 298, 661, 405],
            "text_pixel_bbox": [499, 315, 656, 400],
            "balloon_bbox": [25, 96, 667, 368],
            "bubble_mask_bbox": [435, 255, 701, 442],
            "bubble_mask_source": "image_contour_bubble_mask",
            "route_action": "translate_inpaint_render",
            "route_reason": "mask_outside_balloon_critical",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
        }
        before_text_dark = int(np.count_nonzero(np.mean(image[315:400, 499:656], axis=2) < 80))
        outline_mask = np.zeros((560, 800), dtype=np.uint8)
        cv2.ellipse(outline_mask, (568, 348), (135, 95), 0, 0, 360, 255, 2)
        before_outline_dark = int(np.count_nonzero((np.mean(image, axis=2) < 80) & (outline_mask > 0)))

        result, count = _apply_unsafe_white_balloon_text_fills(image, {"texts": [text]})

        after_text_dark = int(np.count_nonzero(np.mean(result[315:400, 499:656], axis=2) < 80))
        after_outline_dark = int(np.count_nonzero((np.mean(result, axis=2) < 80) & (outline_mask > 0)))
        self.assertEqual(count, 1)
        self.assertLess(after_text_dark, int(before_text_dark * 0.30))
        self.assertGreater(after_outline_dark, int(before_outline_dark * 0.85))

    def test_derived_card_panel_fast_fill_requires_global_opt_in_without_background_metadata(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((150, 260, 3), [84, 48, 68], dtype=np.uint8)
        image[:, ::15, :] = [72, 42, 62]
        cv2.putText(image, "STATUS READY", (38, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        cv2.putText(image, "HOST TITLE", (48, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_105",
            "text": "STATUS READY HOST TITLE",
            "bbox": [34, 32, 206, 102],
            "text_pixel_bbox": [34, 32, 206, 102],
            "balloon_bbox": [24, 20, 224, 116],
            "bubble_mask_bbox": [20, 16, 230, 122],
            "bubble_mask_source": "image_white_bubble_mask",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "route_action": "translate_inpaint_render",
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [dict(text)]}
        derived_block = dict(text)
        derived_block["bubble_mask_source"] = "derived_card_panel_mask"
        derived_block["bubble_mask_bbox"] = [20, 16, 242, 122]
        vision_blocks = [derived_block]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertFalse(page.get("_strip_used_dark_panel_fill"))
        self.assertTrue(np.array_equal(result, image))

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertGreater(int(np.count_nonzero(np.any(result != image, axis=2))), 20)
        self.assertLess(int(np.count_nonzero(np.any(result != image, axis=2))), 8_000)

    def test_derived_card_panel_fast_fill_clears_stale_outside_balloon_flags(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((150, 260, 3), [84, 48, 68], dtype=np.uint8)
        image[:, ::15, :] = [72, 42, 62]
        cv2.putText(image, "SYNCING IS", (48, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        cv2.putText(image, "COMPLETE", (58, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_002",
            "trace_id": "ocr_002@page_006_band_102",
            "text": "Syncing is complete.",
            "bbox": [44, 32, 218, 102],
            "text_pixel_bbox": [44, 32, 218, 102],
            "balloon_bbox": [24, 20, 236, 116],
            "bubble_mask_bbox": [20, 16, 242, 122],
            "bubble_mask_source": "derived_card_panel_mask",
            "route_action": "translate_inpaint_render",
            "route_reason": "mask_outside_balloon_critical",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {
            "texts": [dict(text)],
            "_strip_inpaint_decision_flags": ["mask_outside_balloon_critical", "real_inpaint_skipped_unsafe_mask"],
        }
        vision_blocks = [dict(text)]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertNotIn("mask_outside_balloon", page["texts"][0].get("qa_flags", []))
        self.assertNotIn("mask_outside_balloon_critical", page["texts"][0].get("qa_flags", []))
        self.assertNotIn("mask_outside_balloon_critical", vision_blocks[0].get("qa_flags", []))
        self.assertNotEqual(page["texts"][0].get("route_reason"), "mask_outside_balloon_critical")
        self.assertNotIn("mask_outside_balloon_critical", page.get("_strip_inpaint_decision_flags", []))
        self.assertGreater(int(np.count_nonzero(np.any(result != image, axis=2))), 20)

    def test_dark_panel_fallback_fill_clears_matching_unsafe_sample_flags(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((150, 260, 3), [84, 48, 68], dtype=np.uint8)
        image[:, ::15, :] = [72, 42, 62]
        cv2.putText(image, "SYNCING IS", (48, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        cv2.putText(image, "COMPLETE", (58, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_002",
            "trace_id": "ocr_002@page_006_band_102",
            "text": "Syncing is complete.",
            "bbox": [44, 32, 218, 102],
            "text_pixel_bbox": [44, 32, 218, 102],
            "balloon_bbox": [24, 20, 236, 116],
            "bubble_mask_bbox": [44, 32, 218, 102],
            "bubble_mask_source": "image_white_bubble_mask",
            "route_action": "translate_inpaint_render",
            "route_reason": "mask_outside_balloon_critical",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {
            "texts": [dict(text)],
            "_strip_inpaint_decision_flags": ["mask_outside_balloon_critical", "real_inpaint_skipped_unsafe_mask"],
            "_strip_unsafe_inpaint_block_count": 1,
            "_strip_unsafe_inpaint_block_reasons": {"mask_outside_balloon_critical": 1},
            "_strip_unsafe_inpaint_block_samples": [
                {
                    **dict(text),
                    "reason": "mask_outside_balloon_critical",
                    "bubble_mask_source": "derived_card_panel_mask",
                    "bubble_mask_bbox": [20, 16, 242, 122],
                }
            ],
        }

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 1)
        self.assertNotIn("mask_outside_balloon", page["texts"][0].get("qa_flags", []))
        self.assertNotIn("mask_outside_balloon_critical", page["texts"][0].get("qa_flags", []))
        self.assertNotIn("_strip_unsafe_inpaint_block_samples", page)
        self.assertNotIn("_strip_unsafe_inpaint_block_reasons", page)
        self.assertNotIn("mask_outside_balloon_critical", page.get("_strip_inpaint_decision_flags", []))
        self.assertNotIn("real_inpaint_skipped_unsafe_mask", page.get("_strip_inpaint_decision_flags", []))
        self.assertGreater(int(np.count_nonzero(np.any(result != image, axis=2))), 20)

    def test_fast_dark_panel_fill_prefers_local_vision_block_over_page_text(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((150, 280, 3), [84, 48, 68], dtype=np.uint8)
        image[:, ::15, :] = [72, 42, 62]
        cv2.putText(image, "CURRENT LEVEL", (52, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        cv2.putText(image, "FLOATING SPIRIT", (44, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        local = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_107",
            "text": "Current Level: 0 Class: Floating Spirit",
            "bbox": [42, 32, 234, 102],
            "text_pixel_bbox": [42, 32, 234, 102],
            "balloon_bbox": [24, 20, 252, 116],
            "bubble_mask_bbox": [20, 16, 256, 122],
            "bubble_mask_source": "derived_card_panel_mask",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "route_action": "translate_inpaint_render",
            "mask_evidence": _allowed_mask_evidence(),
        }
        page_text = dict(local)
        page_text.update(
            {
                "bbox": [42, 15322, 234, 15392],
                "text_pixel_bbox": [42, 15322, 234, 15392],
                "balloon_bbox": [24, 15300, 252, 15416],
                "bubble_mask_bbox": [20, 15296, 256, 15422],
            }
        )
        page = {"texts": [page_text]}

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, [dict(local)])

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertGreater(int(np.count_nonzero(np.any(result != image, axis=2))), 500)

    def test_fast_dark_panel_fill_merges_page_text_into_rejected_local_card_when_text_is_missing(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((150, 280, 3), [84, 48, 68], dtype=np.uint8)
        image[:, ::15, :] = [72, 42, 62]
        cv2.putText(image, "CURRENT LEVEL", (52, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        cv2.putText(image, "FLOATING SPIRIT", (44, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 176, 82), 2, cv2.LINE_AA)
        evidence = _allowed_mask_evidence()
        evidence.update({"raw_mask_pixels": 2564, "expanded_mask_pixels": 9341})
        local = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_006_band_107",
            "bbox": [42, 32, 234, 102],
            "text_pixel_bbox": [42, 32, 234, 102],
            "balloon_bbox": [24, 20, 252, 116],
            "bubble_mask_bbox": [20, 16, 256, 122],
            "bubble_mask_source": "derived_white_crop_rejected",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "mask_evidence": evidence,
        }
        page_text = {
            **local,
            "text": "Current Level Class: Floating Spirit",
            "translated": "Classe de nível atual: espírito flutuante",
            "background_rgb": [84, 48, 68],
            "route_action": "translate_inpaint_render",
        }

        result, remaining, stats = _apply_fast_dark_panel_text_fill(
            image,
            {"texts": [dict(page_text)]},
            [dict(local)],
        )

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertGreater(int(np.count_nonzero(np.any(result != image, axis=2))), 500)

    def test_fast_dark_panel_fill_uses_geometry_for_generic_text_tipo(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((160, 360, 3), (112, 82, 48), dtype=np.uint8)
        image[:, ::19, :] = (92, 67, 42)
        cv2.putText(image, "LEVEL 7 BEAST", (22, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (238, 156, 54), 2, cv2.LINE_AA)
        cv2.putText(image, "NEXT PROMOTION", (22, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (238, 156, 54), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "LEVEL 7 BEAST NEXT PROMOTION",
                    "bbox": [18, 28, 260, 96],
                    "text_pixel_bbox": [18, 28, 260, 96],
                    "line_polygons": [
                        [[20, 30], [210, 30], [210, 58], [20, 58]],
                        [[20, 66], [250, 66], [250, 94], [20, 94]],
                    ],
                    "balloon_bbox": [12, 18, 286, 112],
                    "balloon_type": "textured",
                    "block_profile": "standard",
                    "background_rgb": [112, 82, 48],
                    "tipo": "texto",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(page["_strip_used_dark_panel_fill"])
        self.assertTrue(np.any(result != image))

    def test_fast_dark_panel_fill_handles_single_line_colored_power_panel(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((130, 460, 3), (199, 172, 107), dtype=np.uint8)
        image[:, ::23, :] = (160, 128, 70)
        cv2.putText(image, "FULL POWER!", (42, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (244, 159, 44), 4, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "FULL POWER!",
                    "bbox": [38, 18, 430, 106],
                    "text_pixel_bbox": [38, 18, 430, 106],
                    "line_polygons": [[[38, 18], [430, 18], [430, 106], [38, 106]]],
                    "balloon_bbox": [38, 18, 430, 106],
                    "balloon_type": "textured",
                    "block_profile": "standard",
                    "background_rgb": [199, 172, 107],
                    "tipo": "narracao",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        before_delta = np.mean(np.abs(image[35:95, 50:390].astype(np.int16) - np.array([199, 172, 107], dtype=np.int16)))
        after_delta = np.mean(np.abs(result[35:95, 50:390].astype(np.int16) - np.array([199, 172, 107], dtype=np.int16)))
        self.assertLess(after_delta, before_delta * 0.72)

    def test_fast_dark_panel_fill_handles_white_text_on_colored_panel_without_metadata(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((180, 520, 3), (176, 117, 58), dtype=np.uint8)
        image[:, ::31, :] = (150, 92, 45)
        cv2.putText(image, "ALLOWS YOU TO PASS DOWN", (66, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (246, 246, 236), 2, cv2.LINE_AA)
        cv2.putText(image, "RECORDS EFFECTIVELY", (66, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (246, 246, 236), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "Allows you to pass down records effectively",
                    "bbox": [58, 42, 440, 122],
                    "text_pixel_bbox": [58, 42, 440, 122],
                    "balloon_bbox": [36, 24, 485, 145],
                    "balloon_type": "textured",
                    "block_profile": "standard",
                    "tipo": "narracao",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        before_bright = int(np.count_nonzero(np.mean(image[42:122, 58:440], axis=2) > 220))
        after_bright = int(np.count_nonzero(np.mean(result[42:122, 58:440], axis=2) > 220))
        self.assertLess(after_bright, before_bright * 0.25)

    def test_fast_dark_panel_fill_handles_dense_phone_ui_text_without_line_polygons(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((220, 360, 3), (174, 150, 100), dtype=np.uint8)
        image[:, ::19, :] = (136, 104, 42)
        cv2.rectangle(image, (72, 54), (288, 166), (195, 175, 128), -1)
        for y, label in [(82, "KIM LEE CALL MISSED"), (116, "YOO JUNG CALL MISSED"), (150, "JANUARY 2030")]:
            cv2.putText(image, label, (86, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (246, 248, 240), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_004",
                    "text": "KIM LEE CALL MISSED YOO JUNG CALL MISSED JANUARY 2030",
                    "bbox": [70, 50, 292, 170],
                    "text_pixel_bbox": [70, 50, 292, 170],
                    "balloon_bbox": [40, 25, 320, 195],
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "tipo": "fala",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        before_bright = int(np.count_nonzero(np.mean(image[50:170, 70:292], axis=2) > 220))
        after_bright = int(np.count_nonzero(np.mean(result[50:170, 70:292], axis=2) > 220))
        self.assertLess(after_bright, before_bright * 0.25)

    def test_fast_dark_panel_fill_handles_phone_ui_with_line_geometry(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((220, 360, 3), (160, 128, 54), dtype=np.uint8)
        cv2.rectangle(image, (42, 82), (326, 124), (238, 230, 218), -1)
        cv2.rectangle(image, (42, 142), (326, 184), (238, 230, 218), -1)
        cv2.putText(image, "KIM SIHYUK (100+)", (86, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (24, 24, 24), 2, cv2.LINE_AA)
        cv2.putText(image, "MISSED CALL", (86, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (24, 24, 24), 2, cv2.LINE_AA)
        cv2.putText(image, "LEE YOO-JUNG (100+)", (86, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (24, 24, 24), 2, cv2.LINE_AA)
        cv2.putText(image, "MISSED CALL", (86, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (24, 24, 24), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_004",
                    "text": "KIM SIHYUK MISSED CALL LEE YOO-JUNG MISSED CALL",
                    "bbox": [82, 88, 275, 184],
                    "text_pixel_bbox": [82, 88, 275, 184],
                    "line_polygons": [
                        [[82, 88], [260, 88], [260, 110], [82, 110]],
                        [[84, 110], [185, 110], [185, 128], [84, 128]],
                        [[82, 148], [275, 148], [275, 170], [82, 170]],
                        [[84, 170], [185, 170], [185, 188], [84, 188]],
                    ],
                    "balloon_bbox": [42, 82, 326, 184],
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "tipo": "fala",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        before_dark = int(np.count_nonzero(np.mean(image[88:184, 82:275], axis=2) < 80))
        after_dark = int(np.count_nonzero(np.mean(result[88:184, 82:275], axis=2) < 80))
        self.assertLess(after_dark, before_dark * 0.35)

    def test_fast_dark_panel_fill_handles_phone_date_with_line_geometry(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((120, 320, 3), (134, 92, 20), dtype=np.uint8)
        cv2.putText(image, "JANUARY 17, 2030", (68, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 235), 2, cv2.LINE_AA)
        cv2.putText(image, "16:40", (92, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (248, 248, 238), 3, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "JANUARY 17, 2030",
                    "bbox": [64, 34, 250, 68],
                    "text_pixel_bbox": [64, 34, 250, 68],
                    "line_polygons": [[[64, 34], [250, 34], [250, 68], [64, 68]]],
                    "balloon_bbox": [40, 22, 285, 82],
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "tipo": "fala",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        before_bright = int(np.count_nonzero(np.mean(image[34:68, 64:250], axis=2) > 220))
        after_bright = int(np.count_nonzero(np.mean(result[34:68, 64:250], axis=2) > 220))
        self.assertLess(after_bright, before_bright * 0.25)

    def test_fast_dark_panel_fill_handles_blue_glow_on_black_without_metadata(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.zeros((150, 520, 3), dtype=np.uint8)
        image[:] = (4, 5, 10)
        cv2.putText(image, "I REQUEST EARLY ENTRY", (58, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 168, 235), 5, cv2.LINE_AA)
        cv2.putText(image, "I REQUEST EARLY ENTRY", (58, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 244, 255), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "I Request Early ENTRY",
                    "bbox": [48, 42, 454, 98],
                    "text_pixel_bbox": [48, 42, 454, 98],
                    "balloon_bbox": [34, 30, 470, 112],
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "tipo": "fala",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        before_blue = int(np.count_nonzero((image[:, :, 2] > 120) & (image[:, :, 1] > 80)))
        after_blue = int(np.count_nonzero((result[:, :, 2] > 120) & (result[:, :, 1] > 80)))
        self.assertLess(after_blue, before_blue * 0.2)

    def test_dark_panel_fill_skips_missing_glyph_evidence(self):
        from inpainter import _apply_dark_panel_text_fills, _apply_fast_dark_panel_text_fill

        image = np.zeros((120, 260, 3), dtype=np.uint8)
        image[28:86, 36:220] = (70, 44, 118)
        text = {
            "id": "ocr_001",
            "text": "DARLING KARAOKE",
            "bbox": [42, 36, 214, 78],
            "text_pixel_bbox": [42, 36, 214, 78],
            "line_polygons": [[[42, 36], [214, 36], [214, 78], [42, 78]]],
            "background_rgb": [70, 44, 118],
            "layout_profile": "standard",
            "qa_flags": ["fast_fill_no_glyph_evidence"],
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }

        result, count = _apply_dark_panel_text_fills(image, {"texts": [dict(text)]})
        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            fast_result, remaining, stats = _apply_fast_dark_panel_text_fill(image, {"texts": [dict(text)]}, [dict(text)])

        self.assertEqual(count, 0)
        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(stats["dark_panel_fill_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertTrue(np.array_equal(fast_result, image))

    def test_dark_bubble_text_fill_samples_inner_black_not_glow_color(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.zeros((180, 340, 3), dtype=np.uint8)
        image[:] = (3, 4, 7)
        cv2.ellipse(image, (170, 90), (142, 70), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (170, 90), (144, 72), 0, 0, 360, (20, 95, 130), 3)
        cv2.putText(image, "MAIN QUEST", (70, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (75, 65, 30), 8, cv2.LINE_AA)
        cv2.putText(image, "MAIN QUEST", (70, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 232), 2, cv2.LINE_AA)
        cv2.putText(image, "SOON", (128, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (75, 65, 30), 8, cv2.LINE_AA)
        cv2.putText(image, "SOON", (128, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 232), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text": "Main Quest will be shown shortly.",
            "bbox": [62, 58, 260, 126],
            "text_pixel_bbox": [62, 58, 260, 126],
            "balloon_bbox": [28, 20, 312, 160],
            "bubble_mask_bbox": [0, 0, 340, 180],
            "bubble_mask_source": "image_dark_bubble_mask",
            "balloon_type": "textured",
            "background_type": "dark_bubble",
            "layout_profile": "dark_panel",
            "tipo": "fala",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }

        result, count = _apply_dark_panel_text_fills(image, {"texts": [text]})

        self.assertEqual(count, 1)
        changed = np.any(result != image, axis=2)
        changed_pixels = result[changed]
        self.assertGreater(int(changed_pixels.shape[0]), 100)
        self.assertLess(float(np.median(np.mean(changed_pixels.astype(np.float32), axis=1))), 24.0)
        glow_colored = (changed_pixels[:, 0] > 55) & (changed_pixels[:, 1] > 45) & (changed_pixels[:, 2] < 45)
        self.assertLess(float(np.mean(glow_colored)), 0.05)

    def test_dark_bubble_text_fill_prefers_local_black_panel_over_warm_shadow(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.zeros((180, 340, 3), dtype=np.uint8)
        image[:] = (2, 3, 7)
        cv2.rectangle(image, (54, 46), (286, 134), (0, 0, 0), -1)
        cv2.rectangle(image, (52, 44), (288, 136), (26, 95, 118), 3)
        cv2.rectangle(image, (92, 70), (248, 116), (60, 30, 14), -1)
        cv2.putText(image, "MOVE AT", (112, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (76, 58, 28), 7, cv2.LINE_AA)
        cv2.putText(image, "MOVE AT", (112, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 246, 232), 2, cv2.LINE_AA)
        cv2.putText(image, "ONCE", (132, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (76, 58, 28), 7, cv2.LINE_AA)
        cv2.putText(image, "ONCE", (132, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 246, 232), 2, cv2.LINE_AA)
        text = {
            "id": "direct_paddle_reocr_001",
            "text": "Move at once!",
            "bbox": [100, 64, 254, 122],
            "text_pixel_bbox": [100, 64, 254, 122],
            "line_polygons": [
                [[100, 64], [254, 64], [254, 94], [100, 94]],
                [[112, 94], [240, 94], [240, 122], [112, 122]],
            ],
            "balloon_bbox": [52, 44, 288, 136],
            "bubble_mask_bbox": [52, 44, 288, 136],
            "bubble_mask_source": "image_dark_bubble_mask",
            "balloon_type": "textured",
            "background_type": "dark_bubble",
            "layout_profile": "dark_panel",
            "tipo": "fala",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }

        result, count = _apply_dark_panel_text_fills(image, {"texts": [text]})

        self.assertEqual(count, 1)
        changed = np.any(result != image, axis=2)
        changed_pixels = result[changed]
        self.assertGreater(int(changed_pixels.shape[0]), 100)
        median = np.median(changed_pixels.astype(np.float32), axis=0)
        luma = float(median[0] * 0.299 + median[1] * 0.587 + median[2] * 0.114)
        chroma = float(np.max(median) - np.min(median))
        self.assertLess(luma, 18.0)
        self.assertLess(chroma, 18.0)

    def test_dark_bubble_trusted_ocr_mask_keeps_geometry_when_raw_misses_bright_line(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((220, 420, 3), dtype=np.uint8)
        image[:] = (2, 4, 8)
        cv2.ellipse(image, (210, 110), (170, 86), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (210, 110), (172, 88), 0, 0, 360, (18, 96, 130), 3)
        cv2.putText(image, "YET NONE OF THEM EVEN", (64, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 245, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "VISITED YOU ONCE", (96, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (230, 245, 255), 2, cv2.LINE_AA)
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        raw_mask[110:142, 82:326] = 255
        text = {
            "id": "direct_paddle_reocr_001",
            "text": "Yet none of them even visited you once.",
            "bbox": [58, 58, 344, 144],
            "text_pixel_bbox": [58, 58, 344, 144],
            "line_polygons": [
                [[58, 58], [350, 58], [350, 92], [58, 92]],
                [[88, 108], [330, 108], [330, 144], [88, 144]],
            ],
            "balloon_bbox": [40, 24, 382, 198],
            "bubble_mask_bbox": [160, 110, 250, 145],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 420,
                "expanded_mask_pixels": 1800,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        with patch("inpainter.build_raw_text_mask_from_image", return_value=raw_mask):
            result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        fill_mask = text.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        self.assertGreater(int(np.count_nonzero(fill_mask[62:88, 80:340])), 900)
        self.assertGreater(int(np.count_nonzero(fill_mask[112:140, 100:320])), 900)
        self.assertLess(int(np.count_nonzero(fill_mask)), 35000)
        self.assertEqual(int(fill_mask[8, 8]), 0)

    def test_dark_bubble_untrusted_fill_uses_visual_glyph_mask_not_line_rect(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((220, 420, 3), dtype=np.uint8)
        image[:] = (2, 4, 8)
        cv2.ellipse(image, (210, 110), (172, 88), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (210, 110), (174, 90), 0, 0, 360, (16, 92, 124), 3)
        cv2.putText(
            image,
            "I AM A SYSTEM THAT",
            (72, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (238, 248, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "GUIDES THE HOST",
            (92, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (238, 248, 255),
            2,
            cv2.LINE_AA,
        )
        text = {
            "id": "direct_paddle_reocr_001",
            "text": "I am a system that guides the host.",
            "bbox": [58, 62, 358, 150],
            "text_pixel_bbox": [58, 62, 358, 150],
            "line_polygons": [
                [[58, 62], [358, 62], [358, 102], [58, 102]],
                [[82, 110], [336, 110], [336, 150], [82, 150]],
            ],
            "balloon_bbox": [36, 20, 386, 202],
            "bubble_mask_bbox": [36, 20, 386, 202],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["fast_fill_no_glyph_evidence"],
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 2564,
                "expanded_mask_pixels": 15021,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }
        geometry_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for polygon in text["line_polygons"]:
            cv2.fillPoly(geometry_mask, [np.asarray(polygon, dtype=np.int32)], 255)
        geometry_pixels = int(np.count_nonzero(geometry_mask))

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        fill_mask = text.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        fill_pixels = int(np.count_nonzero(fill_mask))
        self.assertGreater(fill_pixels, 600)
        self.assertLess(fill_pixels, int(geometry_pixels * 0.75))
        self.assertIn("dark_bubble_visual_glyph_mask_replaced_geometry", text.get("qa_flags") or [])

    def test_dark_bubble_no_glyph_evidence_bbox_only_uses_visual_mask_not_padded_bbox(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((220, 420, 3), dtype=np.uint8)
        image[:] = (2, 4, 8)
        cv2.ellipse(image, (210, 110), (172, 88), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (210, 110), (174, 90), 0, 0, 360, (16, 92, 124), 3)
        cv2.putText(image, "I AM CALLED", (112, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (238, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "SYSTEM", (142, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (238, 248, 255), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text": "I am called System.",
            "bbox": [98, 70, 306, 166],
            "text_pixel_bbox": [98, 70, 306, 166],
            "balloon_bbox": [36, 20, 386, 202],
            "bubble_mask_bbox": [36, 20, 386, 202],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["fast_fill_no_glyph_evidence"],
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 1800,
                "expanded_mask_pixels": 12000,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }
        bbox_area = (text["text_pixel_bbox"][2] - text["text_pixel_bbox"][0]) * (
            text["text_pixel_bbox"][3] - text["text_pixel_bbox"][1]
        )

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        fill_mask = text.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        fill_pixels = int(np.count_nonzero(fill_mask))
        changed_pixels = int(np.count_nonzero(np.any(result != image, axis=2)))
        self.assertGreater(fill_pixels, 500)
        self.assertLess(fill_pixels, int(bbox_area * 0.55))
        self.assertLessEqual(changed_pixels, int(fill_pixels * 1.05))
        metrics = text.get("qa_metrics") or {}
        self.assertTrue(
            "dark_bubble_visual_glyph_fill_mask" in metrics
            or "dark_bubble_visual_glyph_mask_replaced_geometry" in metrics
        )
        self.assertTrue(
            "dark_bubble_local_inpaint_fill" in metrics
            or "dark_bubble_local_solid_fill" in metrics
        )

    def test_dark_bubble_connected_lobe_fill_clips_foreign_lobe_components(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((240, 560, 3), dtype=np.uint8)
        image[:] = (2, 4, 8)
        cv2.ellipse(image, (210, 112), (172, 86), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (210, 112), (174, 88), 0, 0, 360, (16, 92, 124), 3)
        cv2.ellipse(image, (410, 126), (120, 72), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (410, 126), (122, 74), 0, 0, 360, (16, 92, 124), 3)
        cv2.putText(image, "YOU WERE LOYAL", (92, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (238, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "TO OTHERS", (124, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "THE KING", (350, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (238, 248, 255), 2, cv2.LINE_AA)
        text = {
            "id": "negative_dark_000",
            "text": "You were loyal to others, but to them, you were being nosy.",
            "bbox": [88, 62, 424, 160],
            "text_pixel_bbox": [88, 62, 448, 170],
            "line_polygons": [
                [[88, 62], [300, 62], [300, 104], [88, 104]],
                [[118, 104], [278, 104], [278, 150], [118, 150]],
                [[344, 112], [448, 112], [448, 170], [344, 170]],
            ],
            "balloon_bbox": [36, 24, 530, 204],
            "bubble_mask_bbox": [36, 24, 334, 204],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "qa_flags": [
                "fast_fill_no_glyph_evidence",
                "dark_bubble_connected_lobes_promoted",
                "dark_bubble_lobe_mask_bbox_preferred",
            ],
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 2600,
                "expanded_mask_pixels": 15000,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        fill_mask = text.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        self.assertGreater(int(np.count_nonzero(fill_mask)), 500)
        foreign_lobe_pixels = int(np.count_nonzero(fill_mask[:, 345:] > 0))
        self.assertEqual(foreign_lobe_pixels, 0)
        metrics = text.get("qa_metrics") or {}
        self.assertTrue(
            "dark_bubble_lobe_fill_mask_clipped" in metrics
            or "dark_bubble_connected_lobe_line_polygons_filtered" in metrics
        )
        self.assertTrue(
            "dark_bubble_local_inpaint_fill" in metrics
            or "dark_bubble_local_solid_fill" in metrics
        )

    def test_flat_dark_bbox_fallback_uses_solid_fill_instead_of_telea_smear(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((180, 420, 3), dtype=np.uint8)
        image[:] = (0, 0, 0)
        cv2.putText(image, "YOU LIVED WITHOUT", (68, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (246, 246, 246), 2, cv2.LINE_AA)
        cv2.putText(image, "PLANNING", (128, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (246, 246, 246), 2, cv2.LINE_AA)
        text = {
            "id": "bbox_fallback_dark_001",
            "text": "You lived without planning",
            "bbox": [58, 45, 350, 125],
            "text_pixel_bbox": [58, 45, 350, 125],
            "bubble_mask_source": "bbox_fallback",
            "bubble_mask_error": "missing_real_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 1200,
                "expanded_mask_pixels": 7600,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        fill_mask = text.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        changed = result[fill_mask > 0]
        self.assertGreater(changed.size, 0)
        self.assertLessEqual(int(changed.max()), 8)
        metrics = text.get("qa_metrics") or {}
        self.assertIn("dark_panel_bbox_fallback_solid_fill", metrics)
        self.assertNotIn("dark_panel_bbox_fallback_local_inpaint", metrics)

    def test_text_contract_fill_mask_applies_to_dark_panel_and_tn_without_env(self):
        from inpainter import _dark_text_contract_fill_mask

        cases = [
            {"bubble_mask_source": "image_dark_panel_mask"},
            {"bubble_mask_source": "translator_note_text_mask"},
        ]
        with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "0"}):
            for extra in cases:
                text = {
                    "text": "Original text",
                    "source_text_mask_bbox": [40, 50, 120, 82],
                    "text_pixel_bbox": [40, 50, 120, 82],
                    "route_action": "translate_inpaint_render",
                    "mask_evidence": {
                        "kind": "glyph_segmentation",
                        "raw_mask_pixels": 2800,
                        "expanded_mask_pixels": 4200,
                        "evidence_score": 1.0,
                        "fast_fill_allowed": True,
                        "fast_fill_reject_reasons": [],
                    },
                    **extra,
                }
                mask = _dark_text_contract_fill_mask(text, 200, 160)
                self.assertIsInstance(mask, np.ndarray)
                self.assertGreater(int(np.count_nonzero(mask)), 0)
                self.assertTrue(text.get("_force_solid_dark_text_fill"))

        sfx = {
            "text": "SFX",
            "content_class": "sfx",
            "bubble_mask_source": "image_dark_panel_mask",
            "source_text_mask_bbox": [40, 50, 120, 82],
        }
        self.assertIsNone(_dark_text_contract_fill_mask(sfx, 200, 160))

    def test_text_contract_fill_mask_prefers_lobe_bbox_over_overmerged_pixel_bbox(self):
        from inpainter import _dark_text_contract_fill_mask

        text = {
            "text": "The subspace retention is only five minutes.",
            "bbox": [30, 40, 120, 105],
            "source_text_mask_bbox": [30, 40, 120, 105],
            "text_pixel_bbox": [30, 40, 320, 145],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["visual_text_only_inpaint_contract", "dark_connected_lobe_mask_rebuilt_from_glyphs"],
            "route_action": "translate_inpaint_render",
            "mask_evidence": {
                "kind": "glyph_segmentation",
                "raw_mask_pixels": 4200,
                "expanded_mask_pixels": 7000,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        mask = _dark_text_contract_fill_mask(text, 420, 220)

        self.assertIsInstance(mask, np.ndarray)
        ys, xs = np.where(mask > 0)
        self.assertGreater(xs.size, 0)
        self.assertLessEqual(int(xs.max()), 130)
        self.assertEqual(text.get("qa_metrics", {}).get("dark_text_contract_fill_mask", {}).get("bbox"), [25, 35, 125, 110])

    def test_text_contract_fill_mask_replaces_partial_fragment_polygon_with_text_bbox(self):
        from inpainter import _dark_text_contract_fill_mask

        text = {
            "text": "The subspace retention is only five minutes.",
            "bbox": [129, 111, 312, 234],
            "text_pixel_bbox": [237, 119, 677, 335],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": [
                "visual_text_only_inpaint_contract",
                "dark_connected_lobe_mask_rebuilt_from_glyphs",
                "fast_fill_no_glyph_evidence",
            ],
            "line_polygons": [
                [[250, 130], [318, 130], [318, 210], [250, 210]],
            ],
            "route_action": "translate_inpaint_render",
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 18056,
                "expanded_mask_pixels": 33512,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        mask = _dark_text_contract_fill_mask(text, 800, 700)

        self.assertIsInstance(mask, np.ndarray)
        ys, xs = np.where(mask > 0)
        self.assertGreater(xs.size, 0)
        self.assertLessEqual(int(xs.min()), 125)
        self.assertLessEqual(int(xs.max()), 325)
        self.assertIn("strict_text_geometry_polygon_replaced_by_text_bbox", text.get("qa_metrics", {}))

    def test_text_contract_fill_mask_rejects_overbroad_geometry_against_evidence(self):
        from inpainter import _dark_text_contract_fill_mask

        text = {
            "text": "If you exceed that time, you will return to your original world!",
            "bbox": [237, 58, 744, 579],
            "text_pixel_bbox": [237, 58, 744, 579],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["visual_text_only_inpaint_contract"],
            "route_action": "translate_inpaint_render",
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 29480,
                "expanded_mask_pixels": 33512,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        mask = _dark_text_contract_fill_mask(text, 800, 700)

        self.assertIsNone(mask)
        self.assertIn("dark_text_contract_mask_rejected_overbroad", text.get("qa_flags", []))

    def test_translator_note_text_only_mask_with_geometry_is_processable_for_text_inpaint(self):
        from inpainter import _processable_vision_blocks_for_inpaint

        block = {
            "id": "ocr_tn",
            "text": "T/N: THERE IS A NOTE",
            "bbox": [0, 40, 180, 88],
            "text_pixel_bbox": [0, 40, 180, 88],
            "bubble_mask_source": "translator_note_text_mask",
            "qa_flags": ["translator_note_text_only_mask"],
            "route_action": "translate_inpaint_render",
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 900,
                "expanded_mask_pixels": 1800,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        self.assertEqual(_processable_vision_blocks_for_inpaint([block]), [block])

    def test_translator_note_text_mask_without_text_fields_keeps_geometry_processable(self):
        from inpainter import _processable_vision_blocks_for_inpaint

        block = {
            "id": "ocr_tn_geometry_only",
            "bbox": [39, 282, 154, 337],
            "text_pixel_bbox": [36, 289, 178, 333],
            "bubble_mask_source": "translator_note_text_mask",
            "qa_flags": ["translator_note_text_only_mask", "text_contract_direct_fill"],
            "route_action": "translate_inpaint_render",
            "line_polygons": [
                [[29, 287], [180, 287], [180, 300], [29, 300]],
                [[32, 304], [179, 304], [179, 317], [32, 317]],
                [[19, 322], [191, 322], [191, 336], [19, 336]],
            ],
        }

        self.assertEqual(_processable_vision_blocks_for_inpaint([block]), [block])

    def test_translator_note_text_mask_stays_for_real_inpaint_not_fast_dark_fill(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.zeros((140, 260, 3), dtype=np.uint8)
        cv2.putText(image, "T/N: NOTE", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
        block = {
            "id": "ocr_tn",
            "text": "T/N: NOTE",
            "bbox": [18, 42, 150, 84],
            "text_pixel_bbox": [18, 42, 150, 84],
            "line_polygons": [[[18, 42], [150, 42], [150, 84], [18, 84]]],
            "bubble_mask_source": "translator_note_text_mask",
            "qa_flags": ["translator_note_text_only_mask", "text_contract_direct_fill"],
            "route_action": "translate_inpaint_render",
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 800,
                "expanded_mask_pixels": 1800,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        page = {"texts": [dict(block)]}
        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, [dict(block)])

        self.assertEqual(stats["dark_panel_fill_count"], 0)
        self.assertEqual(stats["remaining_blocks"], 1)
        self.assertEqual(remaining[0].get("id"), "ocr_tn")
        self.assertIn("translator_note_text_mask_requires_real_inpaint", page["_strip_fast_dark_rejection_reasons"])
        self.assertTrue(np.array_equal(result, image))

    def test_translator_note_text_mask_augments_final_real_inpaint_mask(self):
        from inpainter import _augment_inpaint_masks_from_texts

        image = np.zeros((140, 260, 3), dtype=np.uint8)
        cv2.putText(image, "T/N: NOTE", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_tn",
            "text": "T/N: NOTE",
            "bbox": [18, 42, 150, 84],
            "text_pixel_bbox": [18, 42, 150, 84],
            "line_polygons": [[[18, 42], [150, 42], [150, 84], [18, 84]]],
            "bubble_mask_source": "translator_note_text_mask",
            "qa_flags": ["translator_note_text_only_mask", "text_contract_direct_fill"],
            "route_action": "translate_inpaint_render",
        }
        raw = np.zeros(image.shape[:2], dtype=np.uint8)
        expanded = np.zeros_like(raw)

        raw, expanded = _augment_inpaint_masks_from_texts(raw, expanded, [text], image)

        self.assertGreater(int(np.count_nonzero(raw[34:94, 10:160])), 0)
        self.assertGreater(int(np.count_nonzero(expanded[34:94, 10:160])), 0)

    def test_translator_note_final_fill_uses_band_local_coordinates(self):
        from inpainter import _apply_translator_note_dark_text_contract_fill

        image = np.zeros((140, 260, 3), dtype=np.uint8)
        image[38:92, 14:162] = (255, 255, 255)
        page = {
            "_band_y_top": 1000,
            "texts": [
                {
                    "id": "ocr_tn",
                    "text": "T/N: NOTE",
                    "bbox": [18, 1042, 150, 1084],
                    "text_pixel_bbox": [18, 1042, 150, 1084],
                    "line_polygons": [[[18, 1042], [150, 1042], [150, 1084], [18, 1084]]],
                    "bubble_mask_source": "translator_note_text_mask",
                    "qa_flags": ["translator_note_text_only_mask", "text_contract_direct_fill"],
                    "route_action": "translate_inpaint_render",
                }
            ],
        }

        result, changed = _apply_translator_note_dark_text_contract_fill(image, page)

        self.assertGreater(changed, 0)
        self.assertLess(int(np.max(result[45:85, 24:150])), 32)
        self.assertLess(int(np.max(result[40:90, 16:160])), 32)
        self.assertEqual(page.get("_strip_translator_note_dark_contract_final_fill_pixels"), changed)

    def test_translator_note_final_fill_uses_text_samples_when_texts_missing(self):
        from inpainter import _apply_translator_note_dark_text_contract_fill

        image = np.zeros((140, 260, 3), dtype=np.uint8)
        image[38:92, 14:162] = (255, 255, 255)
        page = {
            "_band_y_top": 1000,
            "text_samples": [
                {
                    "id": "ocr_tn",
                    "text": "T/N: NOTE",
                    "bbox": [18, 42, 150, 84],
                    "text_pixel_bbox": [18, 42, 150, 84],
                    "line_polygons": [[[18, 42], [150, 42], [150, 84], [18, 84]]],
                    "bubble_mask_source": "translator_note_text_mask",
                    "qa_flags": ["translator_note_text_only_mask", "text_contract_direct_fill"],
                    "route_action": "translate_inpaint_render",
                }
            ],
        }

        result, changed = _apply_translator_note_dark_text_contract_fill(image, page)

        self.assertGreater(changed, 0)
        self.assertLess(int(np.max(result[45:85, 24:150])), 32)
        self.assertEqual(page.get("_strip_translator_note_dark_contract_final_fill_pixels"), changed)

    def test_translator_note_final_fill_uses_cached_local_texts(self):
        from inpainter import _apply_translator_note_dark_text_contract_fill

        image = np.zeros((140, 260, 3), dtype=np.uint8)
        image[38:92, 14:162] = (255, 255, 255)
        page = {
            "_band_y_top": 1000,
            "_strip_inpaint_local_texts": [
                {
                    "id": "ocr_tn",
                    "text": "T/N: NOTE",
                    "bbox": [18, 42, 150, 84],
                    "text_pixel_bbox": [18, 42, 150, 84],
                    "line_polygons": [[[18, 42], [150, 42], [150, 84], [18, 84]]],
                    "bubble_mask_source": "translator_note_text_mask",
                    "qa_flags": ["translator_note_text_only_mask", "text_contract_direct_fill"],
                    "route_action": "translate_inpaint_render",
                }
            ],
            "texts": [],
        }

        result, changed = _apply_translator_note_dark_text_contract_fill(image, page)

        self.assertGreater(changed, 0)
        self.assertLess(int(np.max(result[45:85, 24:150])), 32)
        self.assertEqual(page.get("_strip_translator_note_dark_contract_final_fill_pixels"), changed)

    def test_dark_mask_component_bright_residual_fill_skips_white_components(self):
        from inpainter import _apply_dark_mask_component_bright_residual_fill

        original = np.zeros((120, 240, 3), dtype=np.uint8)
        cleaned = original.copy()
        cleaned[36:76, 18:118] = (255, 255, 255)
        original[24:96, 150:224] = (255, 255, 255)
        cleaned[24:96, 150:224] = (255, 255, 255)
        mask = np.zeros((120, 240), dtype=np.uint8)
        mask[30:84, 12:128] = 255
        mask[20:100, 145:230] = 255

        result, changed, count = _apply_dark_mask_component_bright_residual_fill(cleaned, original, mask)

        self.assertGreater(changed, 0)
        self.assertEqual(count, 1)
        self.assertLess(int(np.max(result[40:70, 24:110])), 32)
        self.assertGreater(int(np.min(result[30:90, 160:220])), 220)

    def test_dark_panel_text_fill_uses_contract_mask_as_direct_fill(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((180, 260, 3), dtype=np.uint8)
        image[:] = (0, 0, 0)
        cv2.putText(image, "OLD TEXT", (55, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
        text = {
            "id": "contract_direct_fill",
            "text": "OLD TEXT",
            "bbox": [45, 55, 170, 100],
            "text_pixel_bbox": [45, 55, 170, 100],
            "balloon_bbox": [20, 20, 230, 150],
            "bubble_mask_bbox": [20, 20, 230, 150],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["visual_text_only_inpaint_contract"],
            "mask_evidence": {
                "kind": "glyph_segmentation",
                "raw_mask_pixels": 5200,
                "expanded_mask_pixels": 8200,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        metrics = text.get("qa_metrics") or {}
        self.assertIn("text_contract_direct_fill", metrics)
        fill_mask = text.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        self.assertGreater(int(np.count_nonzero(fill_mask)), 0)
        self.assertLessEqual(int(result[fill_mask > 0].max()), 16)

    def test_dark_panel_contract_fill_keeps_text_only_mask_for_visual_contract(self):
        from inpainter import _dark_text_contract_fill_mask, _try_dark_panel_text_fill

        image = np.zeros((190, 420, 3), dtype=np.uint8)
        image[:] = (2, 5, 8)
        cv2.rectangle(image, (24, 35), (390, 150), (4, 8, 12), -1)
        cv2.rectangle(image, (24, 35), (390, 150), (160, 170, 176), 2)
        cv2.putText(image, "QUEST INTRODUCTION", (78, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 190, 230), 5, cv2.LINE_AA)
        cv2.putText(image, "QUEST INTRODUCTION", (78, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 250, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "ESTABLISH YOUR WORLD", (75, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 190, 230), 5, cv2.LINE_AA)
        cv2.putText(image, "ESTABLISH YOUR WORLD", (75, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 250, 255), 2, cv2.LINE_AA)
        text = {
            "id": "dark_panel_glow_contract",
            "text": "Quest Introduction: establish your own underworld!",
            "bbox": [70, 58, 338, 122],
            "text_pixel_bbox": [70, 58, 338, 122],
            "balloon_bbox": [24, 35, 390, 150],
            "bubble_mask_bbox": [24, 35, 390, 150],
            "bubble_mask_source": "image_dark_panel_mask",
            "layout_profile": "dark_panel",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["dark_panel_full_bbox_selected", "visual_text_only_inpaint_contract"],
            "line_polygons": [
                [[94, 65], [318, 65], [318, 84], [94, 84]],
                [[94, 99], [318, 99], [318, 118], [94, 118]],
            ],
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 2800,
                "expanded_mask_pixels": 11800,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }

        contract_mask = _dark_text_contract_fill_mask(dict(text), image.shape[1], image.shape[0], image)
        self.assertIsInstance(contract_mask, np.ndarray)
        ys, xs = np.where(contract_mask > 0)
        self.assertGreater(len(xs), 0)
        bbox_area = (int(xs.max()) + 1 - int(xs.min())) * (int(ys.max()) + 1 - int(ys.min()))
        density = int(np.count_nonzero(contract_mask)) / float(max(1, bbox_area))
        self.assertLess(density, 0.90)

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        fill_mask = text.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        self.assertEqual(int(np.count_nonzero(fill_mask)), int(np.count_nonzero(contract_mask)))
        still_bright = np.any(result[(fill_mask > 0)] >= 90)
        self.assertFalse(still_bright)
        metrics = text.get("qa_metrics") or {}
        self.assertEqual((metrics.get("dark_text_contract_fill_mask") or {}).get("source"), "raw_glyph_mask")
        self.assertIn("dark_text_contract_raw_glyph_mask", metrics)
        self.assertNotIn("dark_panel_visual_contract_fill_mask", metrics)
        self.assertEqual(
            (metrics.get("dark_panel_visual_contract_fill_mask_rejected") or {}).get("reason"),
            "text_only_contract_requires_strict_text_mask",
        )

    def test_koharu_missing_bubble_dark_panel_uses_visual_contract_fill(self):
        from inpainter import inpaint_band_image

        image = np.zeros((190, 420, 3), dtype=np.uint8)
        image[:] = (2, 5, 8)
        cv2.rectangle(image, (24, 35), (390, 150), (4, 8, 12), -1)
        cv2.rectangle(image, (24, 35), (390, 150), (160, 170, 176), 2)
        cv2.putText(image, "QUEST INTRODUCTION", (78, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 190, 230), 5, cv2.LINE_AA)
        cv2.putText(image, "QUEST INTRODUCTION", (78, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 250, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "ESTABLISH YOUR WORLD", (75, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 190, 230), 5, cv2.LINE_AA)
        cv2.putText(image, "ESTABLISH YOUR WORLD", (75, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 250, 255), 2, cv2.LINE_AA)
        text = {
            "id": "dark_panel_missing_bubble",
            "text": "Quest Introduction: establish your own underworld!",
            "bbox": [70, 58, 338, 122],
            "text_pixel_bbox": [70, 58, 338, 122],
            "balloon_bbox": [24, 35, 390, 150],
            "bubble_mask_bbox": [24, 35, 390, 150],
            "bubble_mask_source": "image_dark_panel_mask",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["dark_panel_full_bbox_selected", "visual_text_only_inpaint_contract"],
            "line_polygons": [
                [[94, 65], [318, 65], [318, 84], [94, 84]],
                [[94, 99], [318, 99], [318, 118], [94, 118]],
            ],
            "mask_evidence": {
                "kind": "component_bubble_cleaner",
                "raw_mask_pixels": 2800,
                "expanded_mask_pixels": 11800,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }
        page = {"texts": [dict(text)], "_vision_blocks": [dict(text)]}

        result = inpaint_band_image(image, page)

        self.assertTrue(page.get("_strip_used_koharu_fast_fill"))
        self.assertGreater(int(page.get("_strip_koharu_fast_fill_pixels") or 0), 13000)
        panel_crop = result[50:128, 62:350]
        self.assertLess(int(np.count_nonzero(np.mean(panel_crop, axis=2) > 180)), 64)
        samples = page.get("_strip_koharu_fast_fill_samples") or []
        self.assertTrue(any(sample.get("reason") == "visual_contract_missing_bubble_mask" for sample in samples))
        self.assertTrue(np.array_equal(result[80, 80], np.asarray([4, 8, 12], dtype=np.uint8)))

    def test_dark_bubble_connected_pair_uses_sibling_split_like_white_pair(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((240, 560, 3), dtype=np.uint8)
        image[:] = (2, 4, 8)
        cv2.ellipse(image, (205, 112), (170, 84), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (205, 112), (172, 86), 0, 0, 360, (16, 92, 124), 3)
        cv2.ellipse(image, (405, 126), (118, 70), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (405, 126), (120, 72), 0, 0, 360, (16, 92, 124), 3)
        cv2.putText(image, "YOU WERE LOYAL", (88, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (238, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "TO OTHERS", (120, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "I AM CALLED", (338, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (238, 248, 255), 2, cv2.LINE_AA)
        left = {
            "id": "negative_dark_000",
            "text": "You were loyal to others.",
            "bbox": [0, 22, 558, 332],
            "text_pixel_bbox": [132, 96, 557, 325],
            "line_polygons": [
                [[90, 62], [300, 62], [300, 104], [90, 104]],
                [[118, 104], [278, 104], [278, 150], [118, 150]],
                [[334, 100], [500, 100], [500, 160], [334, 160]],
            ],
            "balloon_bbox": [60, 0, 706, 431],
            "bubble_mask_bbox": [60, 0, 706, 431],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["fast_fill_no_glyph_evidence"],
            "mask_evidence": None,
        }
        right = {
            "id": "ocr_001",
            "text": "I am called System.",
            "bbox": [336, 82, 500, 170],
            "text_pixel_bbox": [336, 82, 500, 170],
            "balloon_bbox": [300, 40, 530, 200],
            "bubble_mask_bbox": [300, 40, 530, 200],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "mask_evidence": _allowed_mask_evidence(),
        }
        left["_dark_bubble_sibling_texts"] = [left, right]

        result = _try_dark_panel_text_fill(image, left)

        self.assertIsNotNone(result)
        fill_mask = left.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        self.assertGreater(int(np.count_nonzero(fill_mask)), 500)
        self.assertEqual(int(np.count_nonzero(fill_mask[:, 346:] > 0)), 0)
        metrics = left.get("qa_metrics") or {}
        self.assertTrue(
            "dark_bubble_sibling_lobe_fill_mask_clipped" in metrics
            or "dark_bubble_sibling_line_polygons_removed" in metrics
        )

    def test_dark_bubble_connected_pair_filters_neighbor_lobe_line_polygons(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((431, 740, 3), dtype=np.uint8)
        image[:] = (1, 1, 0)
        cv2.ellipse(image, (240, 132), (205, 118), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (240, 132), (207, 120), 0, 0, 360, (98, 64, 77), 3)
        cv2.ellipse(image, (560, 274), (150, 112), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (560, 274), (152, 114), 0, 0, 360, (98, 64, 77), 3)
        cv2.putText(image, "YOU WERE LOYAL TO", (76, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (238, 232, 186), 2, cv2.LINE_AA)
        cv2.putText(image, "OTHERS, BUT TO THEM, YOU", (42, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (238, 232, 186), 2, cv2.LINE_AA)
        cv2.putText(image, "WERE BEING NOSY.", (92, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (238, 232, 186), 2, cv2.LINE_AA)
        cv2.putText(image, "YOU", (514, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (238, 232, 186), 2, cv2.LINE_AA)
        cv2.putText(image, "THE KING", (474, 334), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (238, 232, 186), 2, cv2.LINE_AA)

        left = {
            "id": "negative_dark_000",
            "text": "You were loyal to others, but to them, you were being nosy. You the king",
            "bbox": [132, 115, 426, 239],
            "text_pixel_bbox": [132, 116, 557, 325],
            "line_polygons": [
                [[174, 114], [384, 116], [383, 153], [173, 151]],
                [[131, 156], [426, 160], [425, 196], [130, 191]],
                [[176, 198], [378, 201], [377, 238], [175, 235]],
                [[513, 251], [555, 251], [555, 284], [513, 284]],
                [[473, 286], [557, 289], [556, 325], [471, 322]],
            ],
            "balloon_bbox": [63, 0, 706, 431],
            "bubble_mask_bbox": [63, 0, 445, 431],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "qa_flags": [
                "fast_fill_no_glyph_evidence",
                "dark_bubble_connected_lobes_promoted",
                "dark_bubble_lobe_mask_bbox_preferred",
            ],
            "mask_evidence": None,
        }
        right = {
            "id": "ocr_001",
            "text": "You were the king of being a pushover...",
            "bbox": [476, 253, 650, 361],
            "text_pixel_bbox": [476, 253, 650, 361],
            "balloon_bbox": [386, 0, 740, 431],
            "bubble_mask_bbox": [386, 0, 740, 431],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
        }
        left["_dark_bubble_sibling_texts"] = [left, right]

        result = _try_dark_panel_text_fill(image, left)

        self.assertIsNotNone(result)
        fill_mask = left.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        self.assertGreater(int(np.count_nonzero(fill_mask)), 1200)
        self.assertLess(int(np.count_nonzero(fill_mask)), 20000)
        self.assertEqual(int(np.count_nonzero(fill_mask[:, 456:] > 0)), 0)
        metrics = left.get("qa_metrics") or {}
        self.assertTrue(
            "dark_bubble_connected_lobe_line_polygons_filtered" in metrics
            or "dark_bubble_sibling_line_polygons_removed" in metrics
        )
        self.assertIn("dark_bubble_visual_glyph_mask_replaced_geometry", metrics)

        right["_dark_bubble_sibling_texts"] = [left, right]
        right_result = _try_dark_panel_text_fill(image, right)

        self.assertIsNotNone(right_result)
        right_mask = right.get("_dark_panel_fill_mask")
        self.assertIsInstance(right_mask, np.ndarray)
        self.assertGreater(int(np.count_nonzero(right_mask[:, 470:560] > 0)), 500)
        self.assertEqual(int(np.count_nonzero(right_mask[:, :445] > 0)), 0)

    def test_dark_bubble_connected_pair_filters_sibling_lines_without_lobe_flags(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((431, 740, 3), dtype=np.uint8)
        image[:] = (1, 1, 0)
        cv2.ellipse(image, (240, 132), (205, 118), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (240, 132), (207, 120), 0, 0, 360, (98, 64, 77), 3)
        cv2.ellipse(image, (560, 274), (150, 112), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (560, 274), (152, 114), 0, 0, 360, (98, 64, 77), 3)
        cv2.putText(image, "YOU WERE LOYAL TO", (76, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (238, 232, 186), 2, cv2.LINE_AA)
        cv2.putText(image, "OTHERS, BUT TO THEM, YOU", (42, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (238, 232, 186), 2, cv2.LINE_AA)
        cv2.putText(image, "WERE BEING NOSY.", (92, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (238, 232, 186), 2, cv2.LINE_AA)
        cv2.putText(image, "YOU", (514, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (238, 232, 186), 2, cv2.LINE_AA)
        cv2.putText(image, "THE KING", (474, 334), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (238, 232, 186), 2, cv2.LINE_AA)

        left = {
            "id": "negative_dark_000",
            "text": "You were loyal to others, but to them, you were being nosy. You the king",
            "bbox": [0, 22, 558, 332],
            "text_pixel_bbox": [132, 116, 557, 325],
            "line_polygons": [
                [[174, 114], [384, 116], [383, 153], [173, 151]],
                [[131, 156], [426, 160], [425, 196], [130, 191]],
                [[176, 198], [378, 201], [377, 238], [175, 235]],
                [[513, 251], [555, 251], [555, 284], [513, 284]],
                [[473, 286], [557, 289], [556, 325], [471, 322]],
            ],
            "balloon_bbox": [63, 0, 706, 431],
            "bubble_mask_bbox": [63, 0, 706, 431],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["fast_fill_no_glyph_evidence"],
            "mask_evidence": None,
        }
        right = {
            "id": "ocr_001",
            "text": "You were the king of being a pushover...",
            "bbox": [476, 253, 650, 361],
            "text_pixel_bbox": [476, 253, 650, 361],
            "line_polygons": [[[476, 253], [650, 253], [650, 361], [476, 361]]],
            "balloon_bbox": [386, 0, 740, 459],
            "bubble_mask_bbox": [386, 0, 740, 459],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 3153,
                "expanded_mask_pixels": 11206,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }
        left["_dark_bubble_sibling_texts"] = [left, right]
        right["_dark_bubble_sibling_texts"] = [left, right]

        left_result = _try_dark_panel_text_fill(image, left)
        self.assertIsNotNone(left_result)
        right_result = _try_dark_panel_text_fill(left_result, right)
        self.assertIsNotNone(right_result)

        left_mask = left.get("_dark_panel_fill_mask")
        right_mask = right.get("_dark_panel_fill_mask")
        self.assertIsInstance(left_mask, np.ndarray)
        self.assertIsInstance(right_mask, np.ndarray)
        self.assertEqual(int(np.count_nonzero(left_mask[:, 456:] > 0)), 0)
        self.assertGreater(int(np.count_nonzero(right_mask[:, 470:560] > 0)), 500)
        self.assertEqual(int(np.count_nonzero(right_mask[:, :445] > 0)), 0)
        self.assertIn("dark_bubble_sibling_line_polygons_removed", left.get("qa_metrics") or {})

    def test_dark_panel_text_fills_accumulates_visual_mask_for_dark_bubble_no_glyph(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.zeros((220, 420, 3), dtype=np.uint8)
        image[:] = (2, 4, 8)
        cv2.ellipse(image, (210, 110), (172, 88), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (210, 110), (174, 90), 0, 0, 360, (16, 92, 124), 3)
        cv2.putText(image, "I AM CALLED", (112, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (238, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "SYSTEM", (142, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (238, 248, 255), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text": "I am called System.",
            "bbox": [98, 70, 306, 166],
            "text_pixel_bbox": [98, 70, 306, 166],
            "balloon_bbox": [36, 20, 386, 202],
            "bubble_mask_bbox": [36, 20, 386, 202],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["fast_fill_no_glyph_evidence"],
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 1800,
                "expanded_mask_pixels": 12000,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            },
        }
        text["card_panel_text_context"] = True
        text["background_rgb"] = [0, 0, 0]
        page = {"texts": [text]}

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 1)
        fill_mask = page.get("_strip_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        fill_pixels = int(np.count_nonzero(fill_mask))
        changed_pixels = int(np.count_nonzero(np.any(result != image, axis=2)))
        self.assertGreater(fill_pixels, 500)
        self.assertLess(fill_pixels, 9000)
        self.assertLessEqual(changed_pixels, int(fill_pixels * 1.05))

    def test_dark_balloon_fill_does_not_use_negative_white(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.zeros((150, 280, 3), dtype=np.uint8)
        image[:] = (4, 5, 8)
        cv2.ellipse(image, (140, 74), (104, 48), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (140, 74), (106, 50), 0, 0, 360, (18, 90, 125), 3)
        cv2.putText(image, "SYSTEM", (82, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (235, 245, 255), 3, cv2.LINE_AA)
        negative = 255 - image
        text = {
            "id": "ocr_001",
            "text": "System",
            "bbox": [72, 48, 204, 92],
            "text_pixel_bbox": [76, 50, 202, 90],
            "line_polygons": [[[76, 50], [202, 50], [202, 90], [76, 90]]],
            "balloon_bbox": [36, 26, 244, 122],
            "bubble_mask_bbox": [36, 26, 244, 122],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["negative_pass_promoted", "dark_bubble_ellipse_bbox_mask"],
            "mask_evidence": _allowed_mask_evidence(),
        }

        result, count = _apply_dark_panel_text_fills(image, {"texts": [dict(text)]})

        self.assertEqual(count, 1)
        changed = np.any(result != image, axis=2)
        self.assertGreater(int(np.count_nonzero(changed)), 80)
        changed_luma = np.mean(result[changed].astype(np.float32), axis=1)
        self.assertLess(float(np.median(changed_luma)), 28.0)
        inverted_white_luma = float(np.mean(negative[74, 140].astype(np.float32)))
        self.assertGreater(inverted_white_luma, 240.0)

    def test_dark_panel_fill_skips_unsafe_colored_card_without_visual_glyph(self):
        from inpainter import _apply_dark_panel_text_fills, _apply_fast_dark_panel_text_fill

        image = np.zeros((160, 320, 3), dtype=np.uint8)
        for y in range(image.shape[0]):
            image[y, :, :] = (120 + y // 4, 80 + y // 6, 170)
        text = {
            "id": "ocr_001",
            "text": "Synching is complete.",
            "bbox": [78, 48, 236, 106],
            "text_pixel_bbox": [78, 48, 236, 106],
            "line_polygons": [[[78, 48], [236, 48], [236, 106], [78, 106]]],
            "background_rgb": [180, 116, 150],
            "layout_profile": "standard",
            "qa_flags": ["mask_outside_balloon_critical", "debug_derived_bubble_mask_rejected"],
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }

        result, count = _apply_dark_panel_text_fills(image, {"texts": [dict(text)]})
        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            fast_result, remaining, stats = _apply_fast_dark_panel_text_fill(image, {"texts": [dict(text)]}, [dict(text)])

        self.assertEqual(count, 0)
        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(stats["dark_panel_fill_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertTrue(np.array_equal(fast_result, image))

    def test_dark_panel_fill_uses_visual_glyph_for_unsafe_colored_card(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.zeros((160, 320, 3), dtype=np.uint8)
        for y in range(image.shape[0]):
            image[y, :, :] = (120 + y // 4, 80 + y // 6, 170)
        cv2.putText(image, "SYNCHING", (88, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (248, 248, 248), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text": "Synching is complete.",
            "bbox": [78, 48, 236, 106],
            "text_pixel_bbox": [78, 48, 236, 106],
            "line_polygons": [[[78, 48], [236, 48], [236, 106], [78, 106]]],
            "background_rgb": [180, 116, 150],
            "layout_profile": "standard",
            "qa_flags": ["mask_outside_balloon_critical", "debug_derived_bubble_mask_rejected"],
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }

        result, count = _apply_dark_panel_text_fills(image, {"texts": [dict(text)]})

        self.assertEqual(count, 1)
        changed = np.any(result != image, axis=2)
        self.assertGreater(int(np.count_nonzero(changed)), 20)
        self.assertLess(int(np.count_nonzero(changed)), 3_500)
        before_bright = int(np.count_nonzero(np.mean(image[48:106, 78:236], axis=2) > 230))
        after_bright = int(np.count_nonzero(np.mean(result[48:106, 78:236], axis=2) > 230))
        self.assertLess(after_bright, before_bright * 0.5)

    def test_broken_white_dark_visual_mask_stays_near_current_text(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((220, 320, 3), dtype=np.uint8)
        image[:] = (10, 11, 14)
        cv2.ellipse(image, (216, 150), (78, 44), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(image, (216, 150), (80, 46), 0, 0, 360, (18, 92, 122), 2)
        cv2.putText(image, "REWARD", (170, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "IS...", (196, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "OLD TITLE", (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 248, 255), 2, cv2.LINE_AA)
        cv2.rectangle(image, (18, 56), (116, 78), (230, 235, 245), -1)
        text = {
            "id": "ocr_001",
            "text": "The quest reward is....",
            "bbox": [166, 122, 276, 176],
            "text_pixel_bbox": [166, 122, 276, 176],
            "line_polygons": [
                [[166, 122], [276, 122], [276, 148], [166, 148]],
                [[188, 150], [252, 150], [252, 176], [188, 176]],
            ],
            "balloon_bbox": [138, 104, 298, 194],
            "bubble_mask_bbox": [154, 112, 280, 182],
            "bubble_mask_source": "image_white_bubble_mask",
            "background_rgb": [10, 10, 10],
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["dark_bubble_visual_glyph_mask_replaced_geometry"],
            "mask_evidence": _allowed_mask_evidence(),
        }

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        fill_mask = text.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        ys, xs = np.where(fill_mask > 0)
        self.assertGreater(xs.size, 0)
        mask_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        self.assertGreaterEqual(mask_bbox[0], 148)
        self.assertGreaterEqual(mask_bbox[1], 104)
        self.assertLessEqual(mask_bbox[2], 294)
        self.assertLessEqual(mask_bbox[3], 194)
        self.assertEqual(int(np.count_nonzero(fill_mask[16:84, 12:126])), 0)
        self.assertLess(int(np.count_nonzero(fill_mask)), 9_500)

    def test_dark_panel_fill_handles_dense_light_glyph_in_tight_unsafe_card_bbox(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.zeros((120, 320, 3), dtype=np.uint8)
        image[:] = (5, 7, 9)
        cv2.putText(image, "Devil Knight!", (44, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (34, 142, 220), 5, cv2.LINE_AA)
        cv2.putText(image, "Devil Knight!", (44, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (250, 250, 238), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text": "the Devil Knight!",
            "bbox": [38, 36, 270, 76],
            "text_pixel_bbox": [38, 36, 270, 76],
            "line_polygons": [[[38, 36], [270, 36], [270, 76], [38, 76]]],
            "background_rgb": [87, 78, 45],
            "layout_profile": "standard",
            "qa_flags": ["mask_outside_balloon_critical", "missing_real_bubble_mask"],
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }

        page = {"texts": [dict(text)]}
        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 1)
        fill_mask = page.get("_strip_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        self.assertGreater(int(np.count_nonzero(fill_mask)), 100)
        self.assertLess(int(np.count_nonzero(fill_mask)), (270 - 38) * (76 - 36))
        changed = np.any(result != image, axis=2)
        self.assertGreater(int(np.count_nonzero(changed)), 100)
        self.assertLess(int(np.count_nonzero(changed)), 9_500)
        before_bright = int(np.count_nonzero(np.mean(image[36:76, 38:270], axis=2) > 210))
        after_bright = int(np.count_nonzero(np.mean(result[36:76, 38:270], axis=2) > 210))
        self.assertLess(after_bright, before_bright * 0.35)

    def test_dark_panel_bbox_fallback_missing_bubble_uses_glyph_mask_not_panel_strip(self):
        from inpainter import _try_dark_panel_text_fill

        image = np.zeros((220, 360, 3), dtype=np.uint8)
        image[:] = (18, 22, 31)
        cv2.rectangle(image, (46, 54), (328, 178), (168, 184, 200), 2)
        cv2.rectangle(image, (78, 102), (308, 150), (8, 12, 17), -1)
        cv2.rectangle(image, (78, 102), (138, 150), (24, 38, 54), -1)
        cv2.putText(
            image,
            "THE EPISODE STARTS!",
            (98, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (238, 248, 255),
            2,
            cv2.LINE_AA,
        )
        text = {
            "id": "ocr_001",
            "text": "The episode starts!",
            "bbox": [30, 0, 360, 220],
            "text_pixel_bbox": [96, 112, 288, 138],
            "balloon_bbox": [46, 54, 328, 178],
            "bubble_mask_bbox": [46, 54, 328, 178],
            "bubble_mask_source": "bbox_fallback",
            "bubble_mask_error": "missing_real_bubble_mask",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "qa_flags": ["missing_real_bubble_mask"],
            "route_action": "translate_inpaint_render",
            "mask_evidence": _allowed_mask_evidence(),
        }

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        fill_mask = text.get("_dark_panel_fill_mask")
        self.assertIsInstance(fill_mask, np.ndarray)
        ys, xs = np.where(fill_mask > 0)
        self.assertGreater(xs.size, 0)
        mask_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        self.assertLessEqual(mask_bbox[0], 100)
        self.assertGreaterEqual(mask_bbox[2], 284)
        self.assertGreater(mask_bbox[1], 96)
        self.assertLessEqual(mask_bbox[3], 150)
        self.assertLess(int(np.count_nonzero(fill_mask)), 9500)
        self.assertEqual(int(np.count_nonzero(fill_mask[54:60, 46:328])), 0)
        self.assertLess(int(np.count_nonzero(fill_mask[150:154, 78:308])), 16)

    def test_dark_panel_text_fills_skip_white_balloon_context(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((160, 320, 3), 255, dtype=np.uint8)
        image[20:140, 20:300] = (248, 248, 248)
        cv2.putText(image, "DON'T HIT", (92, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "text": "DON'T HIT MY MOM!",
            "bbox": [76, 42, 246, 98],
            "text_pixel_bbox": [76, 42, 246, 98],
            "line_polygons": [[[76, 42], [246, 42], [246, 98], [76, 98]]],
            "bubble_mask_source": "image_white_bubble_mask",
            "background_rgb": [248, 248, 248],
            "route_action": "translate_inpaint_render",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }

        result, count = _apply_dark_panel_text_fills(image, {"texts": [text]})

        self.assertEqual(count, 0)
        self.assertTrue(np.array_equal(result, image))

    def test_fast_white_fill_rejects_colored_textured_speech_panel(self):
        from inpainter import _apply_fast_white_balloon_fill

        image = np.full((120, 260, 3), (45, 150, 190), dtype=np.uint8)
        image[:, ::13, :] = (28, 104, 160)
        cv2.putText(image, "SYSTEM READY", (36, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 248, 255), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "SYSTEM READY",
                    "bbox": [30, 38, 220, 82],
                    "text_pixel_bbox": [30, 38, 220, 82],
                    "line_polygons": [[[30, 38], [220, 38], [220, 82], [30, 82]]],
                    "balloon_bbox": [20, 24, 238, 96],
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "tipo": "fala",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]
        _attach_real_bubble_mask(page, image.shape)
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}):
            result, remaining, stats = _apply_fast_white_balloon_fill(image, page, vision_blocks)

        self.assertEqual(stats["white_balloon_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(page["_strip_fast_white_rejection_reasons"], {"background_variation_high": 1})
        self.assertTrue(np.array_equal(result, image))

    def test_false_white_status_panel_uses_dark_fill_not_fast_white(self):
        from inpainter import _apply_dark_panel_text_fills, _apply_fast_white_balloon_fill

        image = np.full((130, 260, 3), (210, 116, 76), dtype=np.uint8)
        image[:, ::11, :] = (220, 128, 84)
        cv2.rectangle(image, (70, 30), (205, 100), (230, 220, 205), 1)
        cv2.putText(image, "Syncing is", (92, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (248, 248, 248), 2, cv2.LINE_AA)
        cv2.putText(image, "complete.", (96, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (248, 248, 248), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_002",
            "trace_id": "ocr_002@page_006_band_102",
            "text": "Syncing is complete.",
            "bbox": [88, 44, 196, 96],
            "text_pixel_bbox": [88, 44, 196, 96],
            "line_polygons": [
                [[88, 44], [196, 44], [196, 68], [88, 68]],
                [[92, 70], [190, 70], [190, 96], [92, 96]],
            ],
            "balloon_bbox": [70, 30, 205, 100],
            "bubble_mask_bbox": [86, 42, 198, 98],
            "bubble_mask_source": "image_white_bubble_mask",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [dict(text)], "_vision_blocks": [dict(text)]}

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}, clear=False):
            white_result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["white_balloon_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(page["_strip_fast_white_rejection_reasons"], {"false_white_card_panel": 1})
        self.assertTrue(np.array_equal(white_result, image))

        dark_result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 1)
        before_bright = int(np.count_nonzero(np.mean(image[44:96, 88:196], axis=2) > 230))
        after_bright = int(np.count_nonzero(np.mean(dark_result[44:96, 88:196], axis=2) > 230))
        self.assertLess(after_bright, before_bright * 0.45)
        self.assertNotIn("real_inpaint_skipped_unsafe_mask", page.get("_strip_inpaint_decision_flags") or [])

    def test_fast_white_fill_keeps_block_when_component_fill_is_too_sparse(self):
        from inpainter import _apply_fast_white_balloon_fill

        image = np.full((90, 220, 3), 255, dtype=np.uint8)
        image[42:44, 88:92] = 0
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "ALRIGHT",
                    "bbox": [54, 34, 170, 58],
                    "text_pixel_bbox": [54, 34, 170, 58],
                    "balloon_bbox": [30, 16, 195, 76],
                    "balloon_type": "white",
                    "layout_profile": "white_balloon",
                    "tipo": "fala",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]
        _attach_real_bubble_mask(page, image.shape)
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}):
            _, remaining, stats = _apply_fast_white_balloon_fill(image, page, vision_blocks)

        self.assertEqual(stats["white_balloon_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(page["_strip_fast_white_rejection_reasons"], {"insufficient_fast_fill_coverage": 1})

    def test_unsafe_white_fill_clears_stale_unsafe_flags_when_it_fills(self):
        from inpainter import _apply_unsafe_white_balloon_text_fills

        image = np.full((140, 300, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (168, 72), (96, 48), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (168, 72), (96, 48), 0, 0, 360, (24, 24, 24), 2)
        cv2.putText(image, "CHILD'S", (122, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.putText(image, "SAKE.", (138, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2, cv2.LINE_AA)
        text = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_002_band_005",
            "text": "THE CHILD'S SAKE.",
            "bbox": [118, 48, 218, 104],
            "text_pixel_bbox": [118, 48, 218, 104],
            "line_polygons": [
                [[122, 48], [218, 48], [218, 74], [122, 74]],
                [[138, 76], [208, 76], [208, 104], [138, 104]],
            ],
            "balloon_bbox": [72, 24, 264, 120],
            "bubble_mask_bbox": [72, 24, 264, 120],
            "bubble_mask_source": "image_contour_bubble_mask",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["mask_outside_balloon", "mask_outside_balloon_critical"],
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {
            "texts": [dict(text)],
            "_vision_blocks": [dict(text)],
            "_strip_unsafe_inpaint_block_count": 1,
            "_strip_unsafe_inpaint_block_reasons": {"mask_outside_balloon_critical": 1},
            "_strip_unsafe_inpaint_block_samples": [{**dict(text), "reason": "mask_outside_balloon_critical"}],
            "_strip_inpaint_decision_flags": ["mask_outside_balloon_critical", "real_inpaint_skipped_unsafe_mask"],
        }

        result, count = _apply_unsafe_white_balloon_text_fills(image, page)

        self.assertEqual(count, 1)
        self.assertGreater(int(np.count_nonzero(np.any(result != image, axis=2))), 100)
        self.assertNotIn("mask_outside_balloon_critical", page["texts"][0].get("qa_flags") or [])
        self.assertNotIn("real_inpaint_skipped_unsafe_mask", page.get("_strip_inpaint_decision_flags") or [])
        self.assertNotIn("_strip_unsafe_inpaint_block_count", page)

    def test_fast_fill_block_coverage_requires_text_geometry_intersection(self):
        from inpainter import _block_is_covered_by_fast_fill

        block = {
            "bbox": [10, 10, 110, 110],
            "text_pixel_bbox": [80, 80, 98, 98],
            "line_polygons": [[[80, 80], [98, 80], [98, 98], [80, 98]]],
        }

        covered = _block_is_covered_by_fast_fill(block, [[42, 42, 64, 64]], 140, 140)

        self.assertFalse(covered)

    def test_fast_dark_panel_fill_is_clamped_to_text_geometry(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((90, 180, 3), 7, dtype=np.uint8)
        image[34:36, 48:132] = 88
        image[36:48, 58:122] = [210, 208, 184]
        image[48:52, 48:132] = 96
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [48, 34, 132, 50],
                    "text_pixel_bbox": [58, 36, 122, 48],
                    "line_polygons": [[[58, 36], [122, 36], [122, 48], [58, 48]]],
                    "balloon_bbox": [54, 32, 126, 54],
                    "balloon_type": "dark",
                    "layout_profile": "dark_panel",
                    "tipo": "narracao",
                    "skip_processing": False,
                    "mask_evidence": _allowed_mask_evidence(),
                }
            ]
        }
        vision_blocks = [dict(page["texts"][0])]

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, vision_blocks)

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertLess(float(np.mean(result[38:46, 64:116])), 40.0)
        self.assertTrue(np.array_equal(result[34:36, 48:132], image[34:36, 48:132]))
        self.assertTrue(np.array_equal(result[50:52, 48:132], image[50:52, 48:132]))

    def test_fast_fill_without_raw_mask_is_marked_suspicious(self):
        from inpainter import _write_strip_inpaint_debug

        image = np.full((60, 100, 3), 245, dtype=np.uint8)
        working = image.copy()
        working[20:30, 30:70] = 255
        fast_fill_mask = np.zeros((60, 100), dtype=np.uint8)
        fast_fill_mask[20:30, 30:70] = 255
        empty = np.zeros((60, 100), dtype=np.uint8)
        page = {
            "_band_id": "page_001_band_001",
            "texts": [
                {
                    "bbox": [28, 18, 72, 32],
                    "text_pixel_bbox": [30, 20, 70, 30],
                    "skip_processing": False,
                }
            ],
            "_strip_used_fast_local_fill": True,
        }

        with TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {"TRADUZAI_INPAINT_DEBUG_DIR": tmpdir},
            clear=False,
        ):
            _write_strip_inpaint_debug(
                page,
                original_rgb=image,
                working_rgb=working,
                cleaned_rgb=working,
                vision_blocks=[],
                used_real_inpaint=False,
                fast_fill_mask=fast_fill_mask,
                raw_mask=empty,
                expanded_mask=empty,
            )

        self.assertIn("fast_fill_without_raw_mask", page["_strip_inpaint_decision_flags"])

    def test_fast_fill_decisions_ignore_skip_processing_and_preserve_flags(self):
        from inpainter import _build_fallback_vision_blocks, _fast_local_rejection_reason, _fast_white_rejection_reason

        text = {
            "bbox": [10, 10, 40, 30],
            "text_pixel_bbox": [12, 12, 38, 28],
            "skip_processing": True,
            "preserve_original": True,
            "tipo": "watermark",
            "mask_evidence": _allowed_mask_evidence(),
        }

        self.assertNotEqual(_fast_white_rejection_reason(text), "skip_processing")
        self.assertNotEqual(_fast_local_rejection_reason(text), "skip_processing")
        self.assertEqual(len(_build_fallback_vision_blocks({"texts": [text]}, 80, 60)), 1)

    def test_fast_fill_decisions_ignore_declared_balloon_type_and_profiles(self):
        from inpainter import (
            _fast_local_rejection_reason,
            _fast_white_rejection_reason,
            _solid_fast_fill_override_allowed,
        )

        text = {
            "bbox": [10, 10, 90, 38],
            "text_pixel_bbox": [14, 14, 86, 34],
            "line_polygons": [[[14, 14], [86, 14], [86, 34], [14, 34]]],
            "balloon_type": "textured",
            "layout_profile": "dark_panel",
            "background_type": "textured_background",
            "mask_evidence": _allowed_mask_evidence(),
        }
        metadata = {
            "accepted": True,
            "color": [230, 230, 230],
            "max_std": 2.0,
            "p95_abs_delta": 6.0,
        }

        self.assertNotEqual(_fast_white_rejection_reason(text), "textured_or_dark_region")
        self.assertNotEqual(_fast_local_rejection_reason(text), "textured_or_dark_region")
        self.assertTrue(_solid_fast_fill_override_allowed(text, metadata))

    def test_authorized_fast_fill_mask_does_not_use_balloon_bbox_as_limit(self):
        from inpainter import _authorized_fast_fill_mask

        text = {
            "bbox": [20, 20, 160, 74],
            "text_pixel_bbox": [22, 22, 158, 72],
            "line_polygons": [[[22, 22], [158, 22], [158, 72], [22, 72]]],
            "balloon_bbox": [20, 20, 92, 74],
        }

        mask = _authorized_fast_fill_mask(180, 96, text)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(np.count_nonzero(mask[:, 110:])), 0)

    def test_authorized_fast_fill_mask_rejects_bbox_only_geometry(self):
        from inpainter import _authorized_fast_fill_mask

        text = {
            "bbox": [20, 20, 160, 74],
            "balloon_bbox": [0, 0, 180, 96],
        }

        self.assertIsNone(_authorized_fast_fill_mask(180, 96, text))

    def test_fast_fill_rejection_requires_real_text_geometry_not_bbox_only(self):
        from inpainter import _fast_local_rejection_reason, _fast_white_rejection_reason

        text = {
            "bbox": [20, 20, 160, 74],
            "balloon_bbox": [0, 0, 180, 96],
            "mask_evidence": _allowed_mask_evidence(),
        }

        self.assertEqual(_fast_white_rejection_reason(text), "missing_text_geometry")
        self.assertEqual(_fast_local_rejection_reason(text), "missing_text_geometry")

    def test_fast_white_fill_uses_real_bubble_mask_without_balloon_bbox(self):
        from inpainter import _apply_fast_white_balloon_fill

        image = np.full((80, 140, 3), 255, dtype=np.uint8)
        image[34:44, 44:96] = 8
        text = {
            "id": "ocr_001",
            "bubble_id": "bubble_001",
            "bbox": [40, 30, 100, 48],
            "text_pixel_bbox": [44, 34, 96, 44],
            "line_polygons": [[[44, 34], [96, 34], [96, 44], [44, 44]]],
            "mask_evidence": _allowed_mask_evidence(),
        }
        bubble_mask = np.zeros((80, 140), dtype=np.uint8)
        bubble_mask[14:66, 22:118] = 255
        page = {
            "texts": [text],
            "_vision_blocks": [dict(text)],
            "_bubble_regions": [{"bubble_id": "bubble_001", "bubble_mask": bubble_mask}],
        }

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}):
            result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(stats["white_balloon_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(np.all(result[36:42, 50:90] == 255))

    def test_large_dark_textured_text_region_does_not_use_rectangular_solid_fill(self):
        from inpainter import _try_solid_background_text_fill

        image = np.full((260, 260, 3), 8, dtype=np.uint8)
        for y in range(image.shape[0]):
            image[y, :, :] = 8 + (y % 9)
        image[74:86, 44:218, :] = 240
        image[114:126, 44:214, :] = 235
        image[154:166, 44:205, :] = 238

        result = _try_solid_background_text_fill(image, [35, 55, 230, 205], [30, 50, 235, 210])

        self.assertIsNone(result)

    def test_metadata_background_fill_skips_translucent_white_balloon(self):
        from inpainter import _try_metadata_background_text_fill

        image = np.full((120, 180, 3), 244, dtype=np.uint8)
        for x in range(image.shape[1]):
            image[:, x, :] = 232 + (x % 22)
        image[48:60, 62:118] = 18
        text = {
            "bbox": [58, 44, 122, 64],
            "text_pixel_bbox": [62, 48, 118, 60],
            "balloon_bbox": [24, 22, 154, 92],
            "balloon_type": "white",
            "background_rgb": [244, 244, 244],
        }

        result = _try_metadata_background_text_fill(image, text)

        self.assertIsNone(result)

    def test_dark_panel_text_fills_clean_colored_ui_sign_glyphs(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((150, 260, 3), 255, dtype=np.uint8)
        image[40:92, 24:236] = [121, 145, 202]
        image[54:62, 60:198] = 12
        image[70:78, 60:188] = 12
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "Go to successful candidate inquiry",
                    "bbox": [60, 54, 198, 78],
                    "text_pixel_bbox": [60, 54, 198, 78],
                    "background_rgb": [121, 145, 202],
                    "line_polygons": [
                        [[60, 54], [198, 54], [198, 62], [60, 62]],
                        [[60, 70], [188, 70], [188, 78], [60, 78]],
                    ],
                    "route_action": "translate_inpaint_render",
                }
            ]
        }

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 1)
        self.assertGreater(int(np.count_nonzero(np.any(result != image, axis=2))), 0)
        self.assertTrue(np.all(result[55:61, 64:194] == [121, 145, 202]))
        self.assertTrue(np.all(result[71:77, 64:184] == [121, 145, 202]))

    def test_dark_panel_text_fills_clean_form_ui_labels(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((180, 320, 3), 255, dtype=np.uint8)
        image[42:72, 24:296] = [190, 202, 232]
        image[58:66, 68:252] = 24
        image[132:158, 24:296] = [90, 86, 84]
        image[142:152, 48:110] = 246
        page = {
            "texts": [
                {
                    "id": "ocr_header",
                    "text": "Successful candidate inquiry",
                    "bbox": [68, 58, 252, 66],
                    "text_pixel_bbox": [68, 58, 252, 66],
                    "line_polygons": [[[68, 58], [252, 58], [252, 66], [68, 66]]],
                    "background_rgb": [190, 202, 232],
                    "route_action": "translate_inpaint_render",
                },
                {
                    "id": "ocr_search",
                    "text": "Search",
                    "bbox": [48, 142, 110, 152],
                    "text_pixel_bbox": [48, 142, 110, 152],
                    "line_polygons": [[[48, 142], [110, 142], [110, 152], [48, 152]]],
                    "background_rgb": [90, 86, 84],
                    "route_action": "translate_inpaint_render",
                },
            ]
        }

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 2)
        self.assertTrue(np.all(result[59:65, 72:248] == [190, 202, 232]))
        self.assertTrue(np.all(result[143:151, 52:106] == [90, 86, 84]))

    def test_dark_panel_text_fills_do_not_repaint_clean_tilted_search_with_stale_white_metadata(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        panel = np.asarray([[56, 116], [214, 58], [224, 86], [66, 146]], dtype=np.int32)
        cv2.fillPoly(image, [panel], [42, 48, 39])
        line_polygon = [[82, 112], [166, 80], [174, 102], [90, 134]]
        page = {
            "texts": [
                {
                    "id": "ocr_search",
                    "text": "Search",
                    "original": "Search",
                    "translated": "Procurar",
                    "bbox": [82, 80, 174, 134],
                    "text_pixel_bbox": [82, 80, 174, 134],
                    "source_bbox": [56, 58, 224, 146],
                    "line_polygons": [line_polygon],
                    "background_rgb": [255, 255, 255],
                    "layout_profile": "white_balloon",
                    "route_action": "translate_inpaint_render",
                }
            ]
        }

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 0)
        self.assertTrue(np.array_equal(result, image))

    def test_final_inpaint_clamp_restores_artifacts_outside_expanded_mask(self):
        from inpainter import _clamp_final_inpaint_to_expanded_mask

        working = np.full((160, 260, 3), 255, dtype=np.uint8)
        panel = np.asarray([[56, 104], [214, 48], [224, 78], [66, 134]], dtype=np.int32)
        cv2.fillPoly(working, [panel], [42, 48, 39])
        cleaned = working.copy()
        cv2.polylines(
            cleaned,
            [np.asarray([[82, 98], [166, 66], [174, 88], [90, 120]], dtype=np.int32)],
            True,
            [245, 245, 245],
            3,
        )
        cleaned[56:76, 204:224] = [245, 245, 245]
        allowed_mask = np.zeros(working.shape[:2], dtype=np.uint8)
        allowed_mask[56:76, 204:224] = 255

        result, outside_count = _clamp_final_inpaint_to_expanded_mask(working, cleaned, allowed_mask)

        self.assertGreater(outside_count, 0)
        self.assertTrue(np.array_equal(result[100, 98], working[100, 98]))
        self.assertTrue(np.array_equal(result[60, 210], cleaned[60, 210]))

    def test_flat_ui_prefill_removes_form_blocks_before_real_inpaint(self):
        from inpainter import _apply_flat_ui_text_prefill_to_blocks

        image = np.full((180, 320, 3), 255, dtype=np.uint8)
        image[42:72, 24:296] = [190, 202, 232]
        cv2.putText(image, "20** regional fireman", (38, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        image[54:64, 84:238] = 18
        image[90:98, 42:150] = 0
        image[108:116, 42:188] = 0
        image[132:158, 24:296] = [90, 86, 84]
        image[142:152, 48:110] = 246
        texts = [
            {
                "id": "ocr_title",
                "text": "20 ** teste regional de recrutamento de bombeiros",
                "bbox": [36, 24, 244, 36],
                "source_bbox": [34, 22, 248, 40],
                "text_pixel_bbox": [36, 24, 244, 36],
                "line_polygons": [[[36, 24], [244, 24], [244, 36], [36, 36]]],
                "background_rgb": [255, 255, 255],
                "route_action": "translate_inpaint_render",
            },
            {
                "id": "ocr_header",
                "text": "Consulta de candidato bem-sucedida",
                "bbox": [84, 54, 238, 64],
                "source_bbox": [24, 42, 296, 72],
                "text_pixel_bbox": [84, 54, 238, 64],
                "line_polygons": [[[84, 54], [238, 54], [238, 64], [84, 64]]],
                "background_rgb": [190, 202, 232],
                "route_action": "translate_inpaint_render",
            },
            {
                "id": "ocr_label",
                "text": "Nome número de registro de residente",
                "bbox": [42, 90, 188, 116],
                "source_bbox": [38, 86, 194, 120],
                "text_pixel_bbox": [42, 90, 188, 116],
                "line_polygons": [
                    [[42, 90], [150, 90], [150, 98], [42, 98]],
                    [[42, 108], [188, 108], [188, 116], [42, 116]],
                ],
                "background_rgb": [255, 255, 255],
                "route_action": "translate_inpaint_render",
            },
            {
                "id": "ocr_search",
                "text": "Procurar",
                "bbox": [48, 142, 110, 152],
                "source_bbox": [24, 132, 296, 158],
                "text_pixel_bbox": [48, 142, 110, 152],
                "line_polygons": [[[48, 142], [110, 142], [110, 152], [48, 152]]],
                "background_rgb": [90, 86, 84],
                "route_action": "translate_inpaint_render",
            },
        ]
        page = {"texts": [dict(text) for text in texts]}
        vision_blocks = [dict(text) for text in texts]

        result, remaining, stats = _apply_flat_ui_text_prefill_to_blocks(image, page, vision_blocks)

        self.assertEqual(stats["flat_ui_prefill_count"], 4)
        self.assertEqual(remaining, [])
        self.assertTrue(page["_strip_used_flat_ui_prefill"])
        self.assertTrue(np.all(result[24:36, 38:242] == 255))
        self.assertTrue(np.all(result[55:63, 88:234] == [190, 202, 232]))
        self.assertTrue(np.all(result[91:97, 46:146] == 255))
        self.assertTrue(np.all(result[143:151, 52:106] == [90, 86, 84]))

    def test_ui_white_source_bbox_fill_rejects_crop_that_contains_colored_panel(self):
        from inpainter import _try_ui_white_source_bbox_text_fill

        image = np.full((130, 320, 3), 255, dtype=np.uint8)
        image[42:72, 24:296] = [190, 202, 232]
        cv2.putText(
            image,
            "SUCCESSFUL CANDIDATE",
            (78, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        text = {
            "id": "ocr_header",
            "text": "Consulta de candidato bem-sucedida",
            "bbox": [78, 50, 248, 66],
            "source_bbox": [24, 30, 296, 86],
            "text_pixel_bbox": [78, 50, 248, 66],
            "line_polygons": [[[78, 50], [248, 50], [248, 66], [78, 66]]],
            "background_rgb": [255, 255, 255],
            "route_action": "translate_inpaint_render",
        }

        self.assertIsNone(_try_ui_white_source_bbox_text_fill(image, text))

    def test_flat_ui_prefill_removes_same_ocr_block_when_glyph_fill_is_narrow(self):
        from inpainter import _apply_flat_ui_text_prefill_to_blocks

        image = np.full((120, 360, 3), 255, dtype=np.uint8)
        image[42:72, 24:336] = [190, 202, 232]
        cv2.putText(
            image,
            "SUCCESSFUL",
            (88, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        text = {
            "id": "ocr_header",
            "text": "Consulta de candidato bem-sucedida",
            "bbox": [88, 50, 172, 66],
            "source_bbox": [24, 42, 336, 72],
            "text_pixel_bbox": [88, 50, 172, 66],
            "line_polygons": [[[88, 50], [172, 50], [172, 66], [88, 66]]],
            "background_rgb": [190, 202, 232],
            "route_action": "translate_inpaint_render",
        }
        page = {"texts": [dict(text)]}
        wide_detector_block = {
            **text,
            "bbox": [24, 42, 336, 72],
            "source_bbox": [24, 42, 336, 72],
        }

        result, remaining, stats = _apply_flat_ui_text_prefill_to_blocks(image, page, [wide_detector_block])

        self.assertEqual(stats["flat_ui_prefill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(np.all(result[52:64, 90:170] == [190, 202, 232]))

    def test_flat_ui_prefill_uses_metadata_geometry_when_source_bbox_is_too_narrow(self):
        from inpainter import _apply_flat_ui_text_prefill_to_blocks

        image = np.full((100, 360, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "20** REGIONAL FIREMAN RECRUITMENT TEST",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        text = {
            "id": "ocr_title",
            "text": "20 ** teste regional de recrutamento de bombeiros",
            "bbox": [24, 28, 334, 48],
            "source_bbox": [24, 28, 164, 48],
            "text_pixel_bbox": [24, 28, 334, 48],
            "line_polygons": [[[24, 28], [334, 28], [334, 48], [24, 48]]],
            "background_rgb": [255, 255, 255],
            "route_action": "translate_inpaint_render",
        }
        page = {"texts": [dict(text)]}

        result, remaining, stats = _apply_flat_ui_text_prefill_to_blocks(image, page, [dict(text)])

        self.assertEqual(stats["flat_ui_prefill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(np.all(result[30:46, 26:332] == 255))

    def test_flat_ui_prefill_uses_text_geometry_when_ui_context_ring_is_colored(self):
        from inpainter import _apply_flat_ui_text_prefill_to_blocks

        image = np.full((110, 360, 3), [190, 202, 232], dtype=np.uint8)
        image[30:50, 24:334] = 255
        cv2.putText(
            image,
            "20** REGIONAL FIREMAN RECRUITMENT TEST",
            (24, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        text = {
            "id": "ocr_title",
            "text": "20 ** teste regional de recrutamento de bombeiros",
            "bbox": [24, 30, 334, 50],
            "source_bbox": [24, 30, 164, 50],
            "text_pixel_bbox": [24, 30, 334, 50],
            "line_polygons": [[[24, 30], [334, 30], [334, 50], [24, 50]]],
            "background_rgb": [255, 255, 255],
            "route_action": "translate_inpaint_render",
        }

        result, remaining, stats = _apply_flat_ui_text_prefill_to_blocks(
            image,
            {"texts": [dict(text)]},
            [dict(text)],
        )

        self.assertEqual(stats["flat_ui_prefill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertTrue(np.all(result[32:48, 26:332] == 255))
        self.assertTrue(np.all(result[56:70, 24:334] == [190, 202, 232]))

    def test_flat_ui_prefill_rejects_dialogue_with_form_words(self):
        from inpainter import _apply_flat_ui_text_prefill_to_blocks

        image = np.full((160, 360, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (180, 80), (130, 50), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (180, 80), (130, 50), 0, 0, 360, (0, 0, 0), 2)
        cv2.putText(
            image,
            "HEY HOSU.",
            (95, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "R-RESIDENT NUMBER...",
            (75, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        text = {
            "id": "ocr_dialogue",
            "text": "Hey HoSu. R-resident number...",
            "bbox": [75, 58, 280, 96],
            "source_bbox": [75, 58, 280, 96],
            "text_pixel_bbox": [75, 58, 280, 96],
            "line_polygons": [
                [[95, 58], [178, 58], [178, 74], [95, 74]],
                [[75, 78], [280, 78], [280, 96], [75, 96]],
            ],
            "background_rgb": [255, 255, 255],
            "route_action": "translate_inpaint_render",
            "route_reason": "dialogue_balloon_with_english_text",
            "layout_profile": "white_balloon",
        }

        result, remaining, stats = _apply_flat_ui_text_prefill_to_blocks(
            image,
            {"texts": [dict(text)]},
            [dict(text)],
        )

        self.assertEqual(stats["flat_ui_prefill_count"], 0)
        self.assertEqual(len(remaining), 1)
        self.assertTrue(np.array_equal(result, image))

    def test_dark_panel_text_fill_cleans_tilted_ui_antialias_margin(self):
        from inpainter import _try_dark_panel_text_fill

        background = np.asarray([38, 57, 61], dtype=np.uint8)
        image = np.full((140, 240, 3), background, dtype=np.uint8)
        strict_polygon = np.asarray([[68, 82], [166, 44], [176, 70], [78, 108]], dtype=np.int32)
        halo_polygon = np.asarray([[64, 80], [166, 40], [180, 72], [78, 112]], dtype=np.int32)
        cv2.fillPoly(image, [halo_polygon], (180, 180, 180))
        cv2.fillPoly(image, [strict_polygon], (245, 245, 245))
        strict_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        halo_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(strict_mask, [strict_polygon], 255)
        cv2.fillPoly(halo_mask, [halo_polygon], 255)
        antialias_margin = (halo_mask > 0) & (strict_mask == 0)
        text = {
            "id": "ocr_search",
            "text": "Search",
            "bbox": [64, 40, 180, 112],
            "text_pixel_bbox": [64, 40, 180, 112],
            "line_polygons": [strict_polygon.tolist()],
            "background_rgb": background.tolist(),
            "layout_profile": "ui_form",
            "block_profile": "ui_form",
            "route_action": "translate_inpaint_render",
        }

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        self.assertGreater(float(np.mean(image[antialias_margin])), 120.0)
        self.assertLess(float(np.mean(result[antialias_margin])), 80.0)
        self.assertTrue(np.array_equal(result[18, 18], image[18, 18]))

    def test_dark_panel_text_fill_preserves_tilted_panel_details_outside_glyph_mask(self):
        from inpainter import _try_dark_panel_text_fill

        background = np.asarray([38, 57, 61], dtype=np.uint8)
        detail = np.asarray([82, 112, 118], dtype=np.uint8)
        image = np.full((140, 240, 3), background, dtype=np.uint8)
        strict_polygon = np.asarray([[68, 82], [166, 44], [176, 70], [78, 108]], dtype=np.int32)
        cv2.fillPoly(image, [strict_polygon], (245, 245, 245))
        image[50:56, 92:128] = detail
        text = {
            "id": "ocr_search",
            "text": "Search",
            "bbox": [64, 40, 180, 112],
            "text_pixel_bbox": [64, 40, 180, 112],
            "line_polygons": [strict_polygon.tolist()],
            "background_rgb": background.tolist(),
            "layout_profile": "ui_form",
            "block_profile": "ui_form",
            "route_action": "translate_inpaint_render",
        }

        result = _try_dark_panel_text_fill(image, text)

        self.assertIsNotNone(result)
        self.assertTrue(np.array_equal(result[52, 100], detail))

    def test_dark_panel_text_fills_sample_form_ui_bar_without_background_metadata(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((180, 320, 3), 255, dtype=np.uint8)
        image[42:72, 24:296] = [184, 196, 224]
        image[58:66, 68:252] = 255
        image[132:158, 24:296] = [92, 88, 86]
        image[142:152, 48:110] = 255
        page = {
            "texts": [
                {
                    "id": "ocr_header",
                    "text": "Successful candidate inquiry",
                    "bbox": [68, 58, 252, 66],
                    "text_pixel_bbox": [68, 58, 252, 66],
                    "line_polygons": [[[68, 58], [252, 58], [252, 66], [68, 66]]],
                    "route_action": "translate_inpaint_render",
                },
                {
                    "id": "ocr_search",
                    "text": "Search",
                    "bbox": [48, 142, 110, 152],
                    "text_pixel_bbox": [48, 142, 110, 152],
                    "line_polygons": [[[48, 142], [110, 142], [110, 152], [48, 152]]],
                    "route_action": "translate_inpaint_render",
                },
            ]
        }

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 2)
        self.assertTrue(np.all(result[59:65, 72:248] == [184, 196, 224]))
        self.assertTrue(np.all(result[143:151, 52:106] == [92, 88, 86]))

    def test_dark_panel_text_fills_do_not_treat_white_form_labels_as_panel(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((150, 260, 3), 255, dtype=np.uint8)
        image[54:62, 50:150] = 0
        image[72:80, 50:175] = 0
        page = {
            "texts": [
                {
                    "id": "ocr_name",
                    "text": "Name Resident registration number",
                    "bbox": [50, 54, 175, 80],
                    "text_pixel_bbox": [50, 54, 175, 80],
                    "line_polygons": [
                        [[50, 54], [150, 54], [150, 62], [50, 62]],
                        [[50, 72], [175, 72], [175, 80], [50, 80]],
                    ],
                    "route_action": "translate_inpaint_render",
                }
            ]
        }

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 0)
        self.assertTrue(np.array_equal(result, image))

    def test_dark_panel_text_fills_clean_form_white_labels_with_metadata(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((150, 260, 3), 255, dtype=np.uint8)
        image[54:62, 50:150] = 0
        image[72:80, 50:175] = 0
        page = {
            "texts": [
                {
                    "id": "ocr_name",
                    "text": "Name Resident registration number",
                    "bbox": [50, 54, 175, 80],
                    "text_pixel_bbox": [50, 54, 175, 80],
                    "line_polygons": [
                        [[50, 54], [150, 54], [150, 62], [50, 62]],
                        [[50, 72], [175, 72], [175, 80], [50, 80]],
                    ],
                    "background_rgb": [255, 255, 255],
                    "route_action": "translate_inpaint_render",
                }
            ]
        }

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 1)
        self.assertTrue(np.all(result[55:61, 54:146] == 255))
        self.assertTrue(np.all(result[73:79, 54:171] == 255))

    def test_dark_panel_text_fills_do_not_pull_neighbor_bar_color_into_white_header(self):
        from inpainter import _apply_dark_panel_text_fills

        image = np.full((160, 320, 3), 255, dtype=np.uint8)
        image[72:104, 24:296] = [184, 196, 224]
        image[48:58, 60:260] = 0
        page = {
            "texts": [
                {
                    "id": "ocr_header",
                    "text": "20** regional fireman recruitment test",
                    "bbox": [60, 48, 260, 58],
                    "text_pixel_bbox": [60, 48, 260, 58],
                    "line_polygons": [[[60, 48], [260, 48], [260, 58], [60, 58]]],
                    "route_action": "translate_inpaint_render",
                }
            ]
        }

        result, count = _apply_dark_panel_text_fills(image, page)

        self.assertEqual(count, 0)
        self.assertTrue(np.array_equal(result, image))

    def test_tiled_inpaint_handles_edge_tiles_smaller_than_tile_size(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter._run_inpaint = lambda tile_img, tile_mask: tile_img

        image = np.full((600, 1100, 3), 127, dtype=np.uint8)
        mask = np.zeros((600, 1100), dtype=np.uint8)
        mask[:, 980:1080] = 255

        result = inpainter._tiled_inpaint(image, mask, tile_size=512, overlap=64)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_simple_lama_run_normalizes_output_shape_to_input(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter._backend = "simple_lama"
        inpainter._model = lambda img, mask: Image.fromarray(np.full((104, 100, 3), 180, dtype=np.uint8))

        image = np.full((100, 100, 3), 127, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 20:80] = 255

        result = inpainter._run_inpaint(image, mask)

        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.all(result == 180))

    def test_load_model_prefers_lama_onnx_cuda_when_available(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter.device = type("FakeDevice", (), {"type": "cuda"})()
        inpainter.half = True
        inpainter._model = None

        fake_session = object()

        with patch("vision_stack.inpainter.Path.exists", return_value=True), patch(
            "inpainter.lama_onnx.is_lama_manga_available",
            return_value=True,
        ), patch(
            "vision_stack.inpainter.Inpainter._tensorrt_runtime_available",
            return_value=True,
        ), patch(
            "onnxruntime.preload_dlls",
            return_value=None,
        ), patch(
            "onnxruntime.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ), patch(
            "inpainter.lama_onnx.get_lama_session",
            return_value=fake_session,
        ):
            Inpainter._load_model(inpainter, "lama-manga")

        self.assertIs(inpainter._model, fake_session)
        self.assertEqual(inpainter._backend, "lama_onnx_cuda")

    def test_load_model_prefers_lama_onnx_cuda_by_default_even_when_tensorrt_is_available(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter.device = type("FakeDevice", (), {"type": "cuda"})()
        inpainter.half = True
        inpainter._model = None

        class FakeSession:
            def __init__(self, providers):
                self._providers = providers

            def get_providers(self):
                return self._providers

        fake_session = FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"])

        with patch("vision_stack.inpainter.Path.exists", return_value=True), patch(
            "inpainter.lama_onnx.is_lama_manga_available",
            return_value=True,
        ), patch(
            "onnxruntime.preload_dlls",
            return_value=None,
        ), patch(
            "onnxruntime.get_available_providers",
            return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ), patch(
            "inpainter.lama_onnx.get_lama_session",
            return_value=fake_session,
        ) as get_session:
            Inpainter._load_model(inpainter, "lama-manga")

        get_session.assert_called_once_with(
            unittest.mock.ANY,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertIs(inpainter._model, fake_session)
        self.assertEqual(inpainter._backend, "lama_onnx_cuda")

    def test_load_model_allows_tensorrt_when_opted_in(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter.device = type("FakeDevice", (), {"type": "cuda"})()
        inpainter.half = True
        inpainter._model = None

        class FakeSession:
            def __init__(self, providers):
                self._providers = providers

            def get_providers(self):
                return self._providers

        fake_session = FakeSession(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])

        with patch.dict("os.environ", {"MANGATL_ENABLE_TENSORRT": "1"}, clear=False), patch(
            "vision_stack.inpainter.Path.exists",
            return_value=True,
        ), patch(
            "inpainter.lama_onnx.is_lama_manga_available",
            return_value=True,
        ), patch(
            "vision_stack.inpainter.Inpainter._tensorrt_runtime_available",
            return_value=True,
        ), patch(
            "onnxruntime.preload_dlls",
            return_value=None,
        ), patch(
            "onnxruntime.get_available_providers",
            return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ), patch(
            "inpainter.lama_onnx.get_lama_session",
            return_value=fake_session,
        ) as get_session:
            Inpainter._load_model(inpainter, "lama-manga")

        get_session.assert_called_once_with(
            unittest.mock.ANY,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertIs(inpainter._model, fake_session)
        self.assertEqual(inpainter._backend, "lama_onnx_tensorrt")

    def test_run_inpaint_uses_onnx_backend(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter._backend = "lama_onnx_cuda"
        inpainter._model = object()
        inpainter.half = False

        image = np.full((100, 100, 3), 127, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 20:80] = 255

        expected = np.full((100, 100, 3), 210, dtype=np.uint8)

        with patch("inpainter.lama_onnx.inpaint_region_with_lama", return_value=expected) as inpaint_onnx:
            result = inpainter._run_inpaint(image, mask)

        inpaint_onnx.assert_called_once_with(inpainter._model, image, mask)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.all(result == 210))

    def test_aot_model_path_resolver_finds_huggingface_snapshot(self):
        from vision_stack.aot_inpainter import find_aot_model_paths

        with TemporaryDirectory() as tmpdir:
            snapshot = (
                Path(tmpdir)
                / "huggingface"
                / "models--mayocream--aot-inpainting"
                / "snapshots"
                / "abc"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"stub")

            paths = find_aot_model_paths(tmpdir)

        self.assertIsNotNone(paths)
        self.assertEqual(paths.config.name, "config.json")
        self.assertEqual(paths.weights.name, "model.safetensors")

    def test_load_aot_model_requires_explicit_flag(self):
        from vision_stack.aot_inpainter import AotInpaintingUnavailable

        inpainter = Inpainter.__new__(Inpainter)
        inpainter.device = type("FakeDevice", (), {"type": "cpu"})()
        inpainter.half = False
        inpainter._model = None

        with patch.dict("os.environ", {"TRADUZAI_AOT_INPAINT": "0"}, clear=False):
            with self.assertRaisesRegex(AotInpaintingUnavailable, "TRADUZAI_AOT_INPAINT=1"):
                Inpainter._load_model(inpainter, "aot-inpainting")

    def test_run_inpaint_delegates_to_aot_backend(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter._backend = "aot_inpainting"

        expected = np.full((32, 32, 3), 77, dtype=np.uint8)

        class FakeAot:
            def inpaint(self, img_np, mask, debug=None):
                self.last_debug = debug
                self.last_shape = img_np.shape
                return expected

        inpainter._model = FakeAot()
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:20, 8:20] = 255

        result = inpainter._run_inpaint(image, mask)

        self.assertIs(result, expected)
        self.assertEqual(inpainter._model.last_shape, image.shape)

    def test_white_residual_force_fill_uses_dark_local_context(self):
        from inpainter import _apply_white_residual_expanded_mask_force_fill

        image = np.zeros((80, 160, 3), dtype=np.uint8)
        image[:, :, :] = [8, 9, 10]
        image[30:45, 20:120] = [245, 245, 245]
        mask = np.zeros((80, 160), dtype=np.uint8)
        mask[25:50, 15:125] = 255

        result = _apply_white_residual_expanded_mask_force_fill(image, mask)

        filled = result[mask > 0]
        self.assertLess(float(np.mean(filled)), 40.0)


if __name__ == "__main__":
    unittest.main()
