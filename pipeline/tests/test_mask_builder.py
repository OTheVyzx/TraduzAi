import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inpainter.mask_builder import (
    _derive_card_panel_mask_from_image,
    _detach_mixed_sfx_from_white_bubble_block,
    _apply_clipped_overlap_fragment_cleanup_mask,
    _close_visual_text_source_mask_gaps,
    _enforce_visual_text_only_inpaint_contract,
    _text_bbox_contract_mask,
    balloon_mask_from_block,
    bbox_overreach_ratio,
    build_raw_text_mask_from_image,
    build_bubble_limited_glyph_mask,
    build_glyph_text_mask,
    build_inpaint_mask,
    build_mask_regions,
    build_region_pixel_mask,
    clip_text_mask_to_balloon_interior_preserving_raw,
    safe_bubble_interior_mask,
)


class MaskBuilderTests(unittest.TestCase):
    def test_fill_binary_holes_does_not_turn_foreground_touching_corner_into_full_rect(self):
        from inpainter.mask_builder import _fill_binary_holes

        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[:18, :24] = 255
        mask[6:10, 6:10] = 0

        filled = _fill_binary_holes(mask)

        self.assertGreater(int(filled[7, 7]), 0)
        self.assertEqual(int(filled[25, 35]), 0)

    def test_final_text_cleanup_mask_clips_expanded_glyphs_to_eroded_bubble(self):
        bubble = np.zeros((80, 120), dtype=np.uint8)
        cv2.rectangle(bubble, (10, 10), (110, 70), 255, -1)
        glyph = np.zeros_like(bubble)
        glyph[12:20, 54:66] = 255

        mask = build_bubble_limited_glyph_mask(
            glyph,
            bubble,
            glyph_expand_px=8,
            bubble_erode_px=4,
        )

        self.assertGreater(int(np.count_nonzero(mask)), int(np.count_nonzero(glyph)))
        self.assertEqual(int(mask[10, 60]), 0)
        self.assertEqual(int(mask[12, 60]), 0)
        self.assertGreater(int(mask[24, 60]), 0)

    def test_bubble_mask_is_not_used_as_full_erase_mask_by_default(self):
        bubble = np.zeros((100, 140), dtype=np.uint8)
        cv2.ellipse(bubble, (70, 50), (52, 32), 0, 0, 360, 255, -1)
        glyph = np.zeros_like(bubble)
        glyph[46:54, 62:78] = 255

        interior = safe_bubble_interior_mask(bubble, erode_px=3)
        mask = build_bubble_limited_glyph_mask(
            glyph,
            bubble,
            glyph_expand_px=3,
            bubble_erode_px=3,
        )

        self.assertGreater(int(np.count_nonzero(interior)), int(np.count_nonzero(mask)) * 4)
        self.assertEqual(int(mask[50, 24]), 0)
        self.assertGreater(int(mask[50, 70]), 0)

    def test_clipped_overlap_fragment_cleanup_bbox_is_added_to_glyph_mask(self):
        block = {
            "text_pixel_bbox": [20, 20, 80, 40],
            "line_polygons": [[[20, 20], [80, 20], [80, 40], [20, 40]]],
            "qa_metrics": {
                "clipped_overlap_fragment_cleanup_bbox": {
                    "bbox": [100, 70, 170, 95],
                    "removed_tail": "Th",
                }
            },
        }

        mask = build_glyph_text_mask(block, (120, 200, 3))

        self.assertIsNotNone(mask)
        self.assertGreater(int(mask[30, 30]), 0)
        self.assertGreater(int(mask[80, 120]), 0)
        self.assertTrue(block["qa_metrics"]["clipped_overlap_fragment_cleanup_bbox"]["applied_to_glyph_mask"])

    def test_clipped_overlap_fragment_cleanup_survives_final_balloon_clips(self):
        base_mask = np.zeros((720, 640), dtype=np.uint8)
        base_mask[451:533, 164:406] = 255
        block = {
            "text": "That's the power of this underworld!",
            "bbox": [164, 451, 406, 533],
            "source_bbox": [164, 451, 406, 533],
            "text_pixel_bbox": [164, 451, 406, 533],
            "line_polygons": [
                [[179, 451], [397, 457], [397, 494], [177, 488]],
                [[164, 499], [406, 499], [406, 533], [164, 533]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "block_profile": "dark_bubble",
            "layout_profile": "dark_bubble",
            "qa_flags": ["candidate_crop_direct_paddle_reocr", "dark_bubble_oval_reocr"],
            "clipped_overlap_fragment_cleanup_bbox": {
                "bbox": [321, 650, 563, 716],
                "removed_tail": "Th",
            },
            "qa_metrics": {},
        }

        mask = _apply_clipped_overlap_fragment_cleanup_mask(block, base_mask, (720, 640, 3))

        self.assertIsNotNone(mask)
        self.assertGreater(int(mask[470, 190]), 0)
        self.assertGreater(int(mask[670, 350]), 0)
        cleanup = block["clipped_overlap_fragment_cleanup_bbox"]
        self.assertTrue(cleanup["applied_to_final_text_mask"])
        self.assertGreater(cleanup["final_text_mask_pixels_after"], cleanup["final_text_mask_pixels_before"])

    def test_clipped_overlap_fragment_cleanup_falls_back_from_flag_when_bbox_was_dropped(self):
        base_mask = np.zeros((720, 800), dtype=np.uint8)
        base_mask[451:533, 164:406] = 255
        block = {
            "bbox": [164, 451, 406, 533],
            "source_bbox": [164, 451, 406, 533],
            "text_pixel_bbox": [164, 451, 406, 533],
            "bubble_mask_bbox": [88, 221, 496, 668],
            "qa_flags": ["false_dark_bubble_trailing_clipped_fragment_removed"],
            "qa_metrics": {},
        }

        mask = _apply_clipped_overlap_fragment_cleanup_mask(block, base_mask, (720, 800, 3))

        self.assertGreater(int(mask[470, 190]), 0)
        self.assertGreater(int(mask[650, 350]), 0)
        cleanup = block["clipped_overlap_fragment_cleanup_bbox"]
        self.assertEqual(cleanup["source"], "fallback_from_removed_trailing_fragment_flag")
        self.assertTrue(cleanup["applied_to_final_text_mask"])

    def test_clipped_overlap_fragment_cleanup_fill_accumulates_dark_panel_mask(self):
        from inpainter import _apply_clipped_overlap_fragment_cleanup_fill

        image = np.full((720, 800, 3), 24, dtype=np.uint8)
        image[650:700, 330:560] = 220
        ocr_page = {
            "texts": [
                {
                    "bbox": [164, 451, 406, 533],
                    "source_bbox": [164, 451, 406, 533],
                    "text_pixel_bbox": [164, 451, 406, 533],
                    "bubble_mask_bbox": [88, 221, 496, 668],
                    "qa_flags": ["false_dark_bubble_trailing_clipped_fragment_removed"],
                }
            ]
        }

        count = _apply_clipped_overlap_fragment_cleanup_fill(image, ocr_page)

        self.assertEqual(count, 1)
        self.assertLess(int(image[650, 350, 0]), 8)
        fill_mask = ocr_page["_strip_dark_panel_fill_mask"]
        self.assertGreater(int(fill_mask[650, 350]), 0)
        self.assertIn("clipped_overlap_fragment_cleanup_fill", ocr_page["_strip_inpaint_decision_flags"])

    def test_fast_dark_panel_fill_applies_clipped_overlap_cleanup_before_return(self):
        from inpainter import _apply_fast_dark_panel_text_fill

        image = np.full((720, 800, 3), 24, dtype=np.uint8)
        text = {
            "text_id": "direct_paddle_reocr_001",
            "bbox": [164, 451, 406, 533],
            "source_bbox": [164, 451, 406, 533],
            "text_pixel_bbox": [164, 451, 406, 533],
            "bubble_mask_bbox": [88, 221, 496, 668],
            "bubble_mask_source": "image_dark_bubble_mask",
            "line_polygons": [[[164, 451], [406, 451], [406, 533], [164, 533]]],
            "qa_flags": ["false_dark_bubble_trailing_clipped_fragment_removed"],
            "mask_evidence": {
                "fast_fill_allowed": True,
                "kind": "ocr_pixels",
                "raw_mask_pixels": 120,
                "expanded_mask_pixels": 160,
            },
        }
        page = {"texts": [text]}

        def fake_fill(source, _text):
            filled = source.copy()
            filled[451:533, 164:406] = 0
            return filled

        with patch("inpainter._fast_dark_panel_fill_enabled", return_value=True), patch(
            "inpainter._image_dark_bubble_is_visually_light", return_value=False
        ), patch("inpainter._fast_fill_mask_evidence_rejection_reason", return_value=None), patch(
            "inpainter._fast_fill_blocking_qa_reason", return_value=None
        ), patch("inpainter._try_dark_panel_text_fill", side_effect=fake_fill), patch(
            "inpainter._dark_fill_mask_is_overbroad_for_text", return_value=False
        ), patch("inpainter._block_is_covered_by_fast_fill", return_value=True):
            result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, [text])

        self.assertEqual(stats["dark_panel_fill_count"], 1)
        self.assertEqual(remaining, [])
        self.assertLess(int(result[650, 350, 0]), 8)
        self.assertGreater(int(page["_strip_dark_panel_fill_mask"][650, 350]), 0)
        self.assertIn("clipped_overlap_fragment_cleanup_fill", page["_strip_inpaint_decision_flags"])

    def test_dark_connected_text_bbox_contract_uses_lobe_bbox_when_text_pixel_is_shared(self):
        block = {
            "bbox": [129, 111, 312, 234],
            "text_pixel_bbox": [237, 119, 677, 335],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["dark_bubble_connected_lobe_passthrough"],
        }

        mask, bbox = _text_bbox_contract_mask(block, (700, 800))

        self.assertIsNotNone(mask)
        self.assertEqual(bbox, [122, 104, 319, 241])
        self.assertIn("dark_connected_text_pixel_bbox_replaced_by_lobe_bbox", block.get("qa_flags") or [])

    def test_short_dark_text_rejected_full_panel_does_not_derive_card_panel_mask(self):
        image = np.zeros((640, 260, 3), dtype=np.uint8)
        image[:, :] = [8, 9, 12]
        support = np.zeros((640, 260), dtype=np.uint8)
        support[553:590, 63:183] = 255
        block = {
            "text_pixel_bbox": [63, 553, 183, 590],
            "balloon_bbox": [3, 519, 243, 624],
            "background_rgb": [8, 9, 12],
            "qa_flags": ["short_dark_text_full_panel_bbox_rejected"],
            "layout_profile": "dark_panel",
        }

        mask = _derive_card_panel_mask_from_image(block, image.shape, image, support)

        self.assertIsNone(mask)
        self.assertNotEqual(block.get("bubble_mask_source"), "derived_card_panel_mask")
        self.assertEqual(
            block.get("qa_metrics", {}).get("derived_card_panel_mask_rejected", {}).get("reason"),
            "short_dark_text_full_panel_bbox_rejected",
        )

    @patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "1"})
    def test_dark_connected_missing_glyph_uses_lobe_bbox_contract(self):
        block = {
            "bbox": [129, 111, 312, 234],
            "text_pixel_bbox": [237, 119, 677, 335],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["dark_bubble_connected_lobe_passthrough"],
        }
        text_mask = np.zeros((700, 800), dtype=np.uint8)
        text_mask[119:335, 237:677] = 255
        bubble_mask = np.zeros((700, 800), dtype=np.uint8)
        bubble_mask[80:280, 90:350] = 1

        mask = _enforce_visual_text_only_inpaint_contract(
            block,
            text_mask,
            None,
            bubble_mask,
            radius=5,
        )

        self.assertIsNotNone(mask)
        self.assertGreater(int(np.count_nonzero(mask)), 0)
        bbox = block.get("qa_metrics", {}).get("inpaint_mask_contract", {}).get("source_bbox")
        self.assertEqual(bbox, [122, 104, 319, 241])
        self.assertIn("dark_connected_missing_glyph_bbox_contract", block.get("qa_flags") or [])
        self.assertNotIn("visual_text_only_inpaint_missing_glyph_source", block.get("qa_flags") or [])

    def test_balloon_mask_from_block_uses_text_rect_fallback_bubble_mask(self):
        bubble = np.ones((40, 80), dtype=np.uint8) * 255
        block = {
            "bubble_mask": bubble,
            "bubble_mask_bbox": [10, 12, 90, 52],
            "bubble_mask_source": "text_rect_fallback",
        }

        mask = balloon_mask_from_block(block, (80, 120, 3))

        self.assertIsNotNone(mask)
        self.assertEqual(int(np.count_nonzero(mask)), 40 * 80)
        self.assertEqual(int(mask[11, 50]), 0)
        self.assertEqual(int(mask[12, 10]), 255)
        self.assertEqual(int(mask[51, 89]), 255)
        self.assertEqual(int(mask[52, 50]), 0)

    def test_balloon_mask_from_block_uses_ellipse_for_dark_bubble_bbox(self):
        block = {
            "bubble_mask_bbox": [10, 12, 90, 52],
            "bubble_mask_source": "image_dark_bubble_mask",
        }

        mask = balloon_mask_from_block(block, (80, 120, 3))

        self.assertIsNotNone(mask)
        self.assertEqual(int(mask[12, 10]), 0)
        self.assertEqual(int(mask[32, 50]), 255)
        self.assertLess(int(np.count_nonzero(mask)), 40 * 80)

    def test_balloon_mask_from_block_keeps_dark_panel_rect_when_bubble_fallback_was_overbroad(self):
        block = {
            "bubble_mask_bbox": [10, 12, 90, 52],
            "bubble_mask_source": "image_dark_bubble_mask",
            "card_panel_text_context": True,
            "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
            "qa_metrics": {
                "image_dark_panel_mask_rejected": [
                    {"reason": "overbroad_against_balloon_bbox", "mask_bbox": [8, 10, 118, 52]}
                ]
            },
        }

        mask = balloon_mask_from_block(block, (80, 120, 3))

        self.assertIsNotNone(mask)
        self.assertEqual(int(np.count_nonzero(mask)), 40 * 80)
        self.assertEqual(int(mask[12, 10]), 255)
        self.assertEqual(int(mask[51, 89]), 255)
        self.assertEqual(int(mask[11, 50]), 0)
        self.assertEqual(block.get("bubble_mask_source"), "image_dark_panel_mask")
        self.assertNotIn("dark_bubble_ellipse_bbox_mask", block.get("qa_flags") or [])
        self.assertIn("dark_panel_rect_from_dark_bubble_bbox", block.get("qa_flags") or [])

    def test_text_rect_fallback_clips_inpaint_mask_to_rect(self):
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        image[42:58, 20:86] = 0
        bubble = np.ones((40, 80), dtype=np.uint8) * 255
        block = {
            "text": "TL/N: NOTE",
            "bbox": [14, 14, 92, 48],
            "text_pixel_bbox": [20, 42, 86, 58],
            "line_polygons": [[[20, 42], [86, 42], [86, 58], [20, 58]]],
            "bubble_mask": bubble,
            "bubble_mask_bbox": [10, 10, 90, 50],
            "bubble_mask_source": "text_rect_fallback",
            "balloon_bbox": [10, 10, 90, 50],
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertGreater(int(np.count_nonzero(mask)), 0)
        ys, xs = np.where(mask > 0)
        self.assertGreater(len(xs), 0)
        self.assertGreaterEqual(int(xs.min()), 10)
        self.assertLessEqual(int(xs.max()), 89)
        self.assertGreaterEqual(int(ys.min()), 10)
        self.assertLessEqual(int(ys.max()), 49)

    def test_dark_translator_note_does_not_derive_dark_bubble_mask(self):
        image = np.full((180, 260, 3), 2, dtype=np.uint8)
        cv2.putText(
            image,
            "T/N: THERE IS A NOVEL",
            (34, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (244, 244, 244),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "CALLED DEMI-GODS.",
            (34, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (244, 244, 244),
            1,
            cv2.LINE_AA,
        )
        block = {
            "text": "T/N: THERE IS A NOVEL CALLED DEMI-GODS.",
            "bbox": [30, 54, 212, 112],
            "text_pixel_bbox": [34, 62, 196, 104],
            "line_polygons": [
                [[34, 62], [196, 62], [196, 78], [34, 78]],
                [[34, 86], [178, 86], [178, 104], [34, 104]],
            ],
            "background_rgb": [2, 2, 2],
            "balloon_bbox": [0, 38, 238, 132],
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertEqual(block.get("bubble_mask_source"), "translator_note_text_mask")
        self.assertIn("translator_note_text_only_mask", block.get("qa_flags") or [])
        self.assertNotEqual(block.get("bubble_mask_source"), "image_dark_bubble_mask")
        self.assertNotEqual(block.get("bubble_mask_source"), "image_white_bubble_mask")
        self.assertNotIn("image_dark_bubble_mask", block.get("qa_metrics", {}))
        ys, xs = np.where(mask > 0)
        self.assertGreater(len(xs), 0)
        self.assertGreaterEqual(int(xs.min()), 28)
        self.assertLessEqual(int(xs.max()), 210)
        self.assertLessEqual((int(xs.max()) - int(xs.min())) * (int(ys.max()) - int(ys.min())), 16000)

    def test_dark_bubble_build_inpaint_mask_derives_image_dark_bubble_mask(self):
        image = np.zeros((160, 280, 3), dtype=np.uint8)
        cv2.ellipse(image, (140, 78), (96, 48), 0, 0, 360, (5, 7, 10), -1)
        cv2.ellipse(image, (140, 78), (96, 48), 0, 0, 360, (45, 148, 190), 4)
        cv2.putText(image, "DARK", (92, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (246, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "OVAL", (94, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (246, 248, 255), 2, cv2.LINE_AA)
        block = {
            "text": "DARK OVAL",
            "bbox": [76, 44, 204, 112],
            "source_bbox": [76, 44, 204, 112],
            "text_pixel_bbox": [88, 58, 196, 108],
            "line_polygons": [
                [[88, 58], [194, 58], [194, 82], [88, 82]],
                [[88, 86], [190, 86], [190, 110], [88, 110]],
            ],
            "balloon_bbox": [44, 30, 236, 126],
            "background_rgb": [5, 7, 10],
            "background_polarity": "dark",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "dark_light_text_evidence": {"useful": True},
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(block.get("bubble_mask_source"), "image_dark_bubble_mask")
        self.assertEqual(block.get("bubble_mask_shape"), "ellipse")
        self.assertGreater(int(mask[78, 140]), 0)
        self.assertEqual(int(mask[30, 44]), 0)

    def test_dark_bubble_visual_glyph_mask_replaces_rectangular_line_geometry(self):
        image = np.full((180, 280, 3), 8, dtype=np.uint8)
        bubble = np.zeros((180, 280), dtype=np.uint8)
        cv2.ellipse(bubble, (140, 90), (110, 66), 0, 0, 360, 255, -1)
        image[bubble > 0] = (0, 0, 0)
        cv2.putText(image, "CURRENT LEVEL", (72, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(image, "CLASS FLOATING", (70, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)
        block = {
            "id": "ocr_001",
            "text": "Current Level Class Floating",
            "translated": "Classe de nivel atual: espirito flutuante",
            "bbox": [68, 62, 214, 112],
            "source_bbox": [68, 62, 214, 112],
            "text_pixel_bbox": [68, 62, 214, 112],
            "line_polygons": [
                [[66, 60], [218, 60], [218, 86], [66, 86]],
                [[66, 88], [218, 88], [218, 114], [66, 114]],
            ],
            "bubble_mask": bubble,
            "bubble_mask_bbox": [30, 24, 250, 156],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "background_rgb": [0, 0, 0],
            "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        geometry_area = (218 - 66) * (86 - 60) + (218 - 66) * (114 - 88)
        self.assertLess(int(np.count_nonzero(mask)), int(geometry_area * 0.82))
        self.assertGreater(int(np.count_nonzero(mask)), 120)
        self.assertIn("dark_bubble_visual_glyph_mask_replaced_geometry", block.get("qa_flags") or [])
        self.assertIn("dark_bubble_visual_glyph_mask", block.get("qa_metrics", {}))

    def test_dark_panel_build_inpaint_mask_derives_image_dark_panel_mask(self):
        image = np.full((150, 320, 3), 232, dtype=np.uint8)
        cv2.rectangle(image, (48, 42), (272, 112), (8, 8, 10), -1)
        cv2.rectangle(image, (48, 42), (272, 112), (92, 180, 210), 3)
        cv2.putText(image, "DARK PANEL", (82, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (248, 248, 255), 2, cv2.LINE_AA)
        block = {
            "text": "DARK PANEL",
            "bbox": [74, 58, 246, 96],
            "source_bbox": [74, 58, 246, 96],
            "text_pixel_bbox": [82, 64, 238, 88],
            "line_polygons": [[[82, 64], [238, 64], [238, 88], [82, 88]]],
            "balloon_bbox": [48, 42, 272, 112],
            "background_rgb": [8, 8, 10],
            "background_polarity": "dark",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "dark_light_text_evidence": {"useful": True},
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertIn(block.get("bubble_mask_source"), {"image_dark_panel_mask", "derived_card_panel_mask"})
        self.assertTrue(block.get("mask_evidence", {}).get("fast_fill_allowed"))

    def test_dark_ui_panel_prefers_rect_panel_mask_over_expanded_ellipse(self):
        image = np.zeros((120, 360, 3), dtype=np.uint8)
        image[:, :] = (8, 10, 12)
        cv2.rectangle(image, (92, 28), (318, 92), (12, 9, 5), -1)
        cv2.rectangle(image, (92, 28), (318, 92), (190, 178, 150), 2)
        cv2.putText(image, "Current Level: 0", (142, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (248, 246, 232), 2, cv2.LINE_AA)
        cv2.putText(image, "Class: Floating Spirit", (122, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (248, 246, 232), 2, cv2.LINE_AA)
        block = {
            "text": "Current Level: 0 Class: Floating Spirit",
            "bbox": [122, 40, 300, 88],
            "source_bbox": [122, 40, 300, 88],
            "text_pixel_bbox": [122, 40, 300, 88],
            "line_polygons": [
                [[142, 40], [292, 40], [292, 62], [142, 62]],
                [[122, 66], [300, 66], [300, 88], [122, 88]],
            ],
            "balloon_bbox": [92, 28, 318, 92],
            "background_rgb": [12, 9, 5],
            "background_polarity": "dark",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "card_panel_text_context": True,
            "dark_light_text_evidence": {"useful": True},
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertIn(block.get("bubble_mask_source"), {"image_dark_panel_mask", "derived_card_panel_mask"})
        self.assertNotEqual(block.get("bubble_mask_source"), "image_dark_bubble_mask")
        self.assertNotIn("dark_bubble_ellipse_bbox_mask", block.get("qa_flags") or [])
        self.assertLess(int(np.count_nonzero(mask)), 14000)

    def test_dark_bubble_candidate_with_rect_border_promotes_to_dark_panel(self):
        image = np.zeros((180, 520, 3), dtype=np.uint8)
        image[:, :] = (2, 3, 5)
        cv2.rectangle(image, (156, 54), (420, 126), (1, 1, 1), -1)
        cv2.rectangle(image, (156, 54), (420, 126), (178, 190, 196), 3)
        cv2.putText(image, "MOVE", (226, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 246), 2, cv2.LINE_AA)
        cv2.putText(image, "NOW", (236, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 246), 2, cv2.LINE_AA)
        block = {
            "text": "MOVE NOW",
            "bbox": [218, 66, 326, 118],
            "source_bbox": [218, 66, 326, 118],
            "text_pixel_bbox": [218, 66, 326, 118],
            "line_polygons": [
                [[226, 66], [318, 66], [318, 90], [226, 90]],
                [[236, 92], [308, 92], [308, 118], [236, 118]],
            ],
            "balloon_bbox": [110, 18, 470, 164],
            "background_rgb": [2, 3, 5],
            "background_polarity": "dark",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "dark_light_text_evidence": {"useful": True},
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertEqual(block.get("bubble_mask_source"), "image_dark_panel_mask")
        bx1, by1, bx2, by2 = block.get("bubble_mask_bbox") or [0, 0, 0, 0]
        self.assertLessEqual(abs(bx1 - 156), 4)
        self.assertLessEqual(abs(by1 - 54), 4)
        self.assertLessEqual(abs(bx2 - 421), 12)
        self.assertLessEqual(abs(by2 - 127), 4)
        self.assertIn("dark_panel_rect_from_border_lines", block.get("qa_flags") or [])
        self.assertNotIn("dark_bubble_ellipse_bbox_mask", block.get("qa_flags") or [])
        metrics = block.get("qa_metrics", {}).get("image_dark_panel_mask", {})
        self.assertIn(metrics.get("detection_space"), {"border_line_geometry", "border_line_geometry_inferred_dark_rect_panel"})
        self.assertGreater(metrics.get("border_inner_luma_delta", 0), 14.0)

    def test_negative_white_text_without_balloon_uses_fast_fill_only_when_not_touching_art_border(self):
        image = np.full((120, 260, 3), 5, dtype=np.uint8)
        cv2.putText(image, "SYSTEM", (76, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (246, 246, 250), 2, cv2.LINE_AA)
        block = {
            "text": "SYSTEM",
            "bbox": [68, 42, 188, 76],
            "source_bbox": [68, 42, 188, 76],
            "text_pixel_bbox": [76, 48, 178, 70],
            "line_polygons": [[[76, 48], [178, 48], [178, 70], [76, 70]]],
            "background_rgb": [5, 5, 5],
            "background_polarity": "dark",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "dark_light_text_evidence": {"useful": True, "has_balloon": False},
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertNotEqual(block.get("bubble_mask_source"), "image_dark_bubble_mask")
        self.assertTrue(block.get("mask_evidence", {}).get("fast_fill_allowed"))

        border_block = dict(block)
        border_block["bbox"] = [0, 42, 188, 76]
        border_block["source_bbox"] = [0, 42, 188, 76]
        border_block["text_pixel_bbox"] = [0, 48, 178, 70]
        border_block["line_polygons"] = [[[0, 48], [178, 48], [178, 70], [0, 70]]]
        border_mask = build_inpaint_mask(border_block, image.shape, image_rgb=image)

        self.assertIsNotNone(border_mask)
        self.assertFalse(border_block.get("mask_evidence", {}).get("fast_fill_allowed"))

    def test_balloon_clip_does_not_preserve_raw_glyphs_on_balloon_outline(self):
        bubble = np.zeros((80, 160), dtype=np.uint8)
        cv2.rectangle(bubble, (20, 20), (140, 60), 255, -1)
        raw = np.zeros_like(bubble)
        raw[21:28, 128:138] = 255
        text_mask = raw.copy()
        text_mask[34:42, 68:92] = 255

        clipped = clip_text_mask_to_balloon_interior_preserving_raw(
            text_mask,
            bubble,
            raw,
            erode_px=4,
            block={"layout_profile": "white_balloon", "block_profile": "white_balloon"},
            source="image_white_bubble_mask",
        )

        safe = safe_bubble_interior_mask(bubble, erode_px=4)
        raw_on_outline = (raw > 0) & (safe == 0)
        raw_inside = (raw > 0) & (safe > 0)
        self.assertGreater(int(np.count_nonzero(raw_on_outline)), 0)
        self.assertGreater(int(np.count_nonzero(raw_inside)), 0)
        self.assertEqual(int(np.count_nonzero((clipped > 0) & raw_on_outline)), 0)
        self.assertEqual(int(np.count_nonzero((clipped > 0) & raw_inside)), 0)
        self.assertGreater(int(np.count_nonzero(clipped[34:42, 68:92])), 0)

    def test_raw_text_mask_drops_unanchored_balloon_outline_fragments(self):
        image = np.full((90, 140, 3), 255, dtype=np.uint8)
        image[42:48, 50:92] = 0
        image[35:38, 92:100] = 0
        image[61:68, 96:106] = 0
        block = {
            "text": "I REALLY",
            "bbox": [48, 38, 94, 52],
            "text_pixel_bbox": [48, 38, 94, 52],
            "line_polygons": [[[48, 40], [94, 40], [94, 52], [48, 52]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "bubble_mask_source": "image_white_bubble_mask",
        }

        mask = build_raw_text_mask_from_image(block, image, image.shape)

        self.assertIsNotNone(mask)
        self.assertGreater(int(np.count_nonzero(mask[42:48, 50:92])), 0)
        self.assertEqual(int(np.count_nonzero(mask[35:38, 92:100])), 0)
        self.assertEqual(int(np.count_nonzero(mask[61:68, 96:106])), 0)

    def test_build_inpaint_mask_drops_isolated_sfx_side_note_polygon_from_dialogue(self):
        image = np.full((180, 240, 3), 255, dtype=np.uint8)
        image[118:132, 20:82] = 0
        image[100:122, 130:220] = 0
        image[140:162, 130:220] = 0
        block = {
            "text": "DON'T HIT MY MOM!",
            "translated": "NÃO APERTE MINHA MÃE!",
            "bbox": [20, 95, 220, 165],
            "text_pixel_bbox": [20, 100, 220, 162],
            "line_polygons": [
                [[130, 100], [220, 100], [220, 122], [130, 122]],
                [[20, 118], [82, 118], [82, 132], [20, 132]],
                [[130, 140], [220, 140], [220, 162], [130, 162]],
            ],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertEqual(int(np.count_nonzero(mask[116:134, 18:86])), 0)
        self.assertGreater(int(np.count_nonzero(mask[98:124, 128:222])), 0)
        self.assertGreater(int(np.count_nonzero(mask[138:164, 128:222])), 0)

    def test_ocr_gibberish_does_not_build_inpaint_mask(self):
        image = np.full((90, 140, 3), 255, dtype=np.uint8)
        image[30:50, 40:90] = 0
        text = {
            "text": "3 TI2]2H",
            "bbox": [30, 20, 100, 60],
            "text_pixel_bbox": [30, 20, 100, 60],
            "line_polygons": [[[30, 20], [100, 20], [100, 60], [30, 60]]],
            "qa_flags": ["ocr_gibberish"],
        }

        mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNone(mask)
        self.assertIn(
            "ocr_gibberish_suppressed",
            text.get("mask_evidence", {}).get("fast_fill_reject_reasons", []),
        )

    def test_translator_note_keeps_glyph_mask_when_bubble_clip_misses_note(self):
        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        image[62:66, 126:204] = 0
        image[70:74, 126:196] = 0
        image[78:82, 126:188] = 0
        bubble = np.zeros((120, 220), dtype=np.uint8)
        cv2.ellipse(bubble, (60, 24), (45, 18), 0, 0, 360, 255, -1)
        text = {
            "text": "T/N: HYUNGNIM IS A TERM USED FOR CALLING ONE'S BOSS.",
            "translated": "T/N: HYUNGNIM É UM TERMO USADO PARA CHAMAR O CHEFE.",
            "bbox": [126, 60, 204, 84],
            "text_pixel_bbox": [126, 60, 204, 84],
            "line_polygons": [
                [[126, 62], [204, 62], [204, 66], [126, 66]],
                [[126, 70], [196, 70], [196, 74], [126, 74]],
                [[126, 78], [188, 78], [188, 82], [126, 82]],
            ],
            "balloon_bbox": [15, 6, 105, 42],
            "bubble_mask": bubble,
            "bubble_mask_source": "image_contour_bubble_mask",
            "layout_profile": "white_balloon",
        }

        mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertGreater(int(np.count_nonzero(mask[:, 120:])), 0)
        self.assertIn("translator_note_bubble_clip_skipped", text.get("qa_flags") or [])

    def test_translator_note_with_local_bubble_mask_does_not_crash_clip_check(self):
        from inpainter.mask_builder import _translator_note_bubble_clip_misses_text

        image = np.full((342, 800, 3), 245, dtype=np.uint8)
        local_bubble = np.ones((63, 123), dtype=np.uint8) * 255
        text = {
            "text": "T/N: AJUMMA IS A TERM USED FOR MARRIED WOMEN.",
            "bbox": [423, 228, 506, 277],
            "text_pixel_bbox": [212, 233, 585, 273],
            "line_polygons": [
                [[212, 233], [585, 233], [585, 273], [212, 273]],
            ],
            "balloon_bbox": [405, 214, 524, 291],
            "bubble_mask": local_bubble,
            "bubble_mask_bbox": [407, 234, 530, 297],
            "bubble_mask_source": "derived_white_crop_rejected",
        }

        mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, image.shape[:2])

        geometry = np.zeros(image.shape[:2], dtype=np.uint8)
        geometry[233:273, 212:585] = 255
        self.assertIsInstance(
            _translator_note_bubble_clip_misses_text(text, geometry, local_bubble),
            bool,
        )

    def test_merges_nearby_text_boxes_from_same_balloon(self):
        texts = [
            {"bbox": [100, 100, 180, 140], "tipo": "fala", "confidence": 0.91},
            {"bbox": [110, 148, 176, 186], "tipo": "fala", "confidence": 0.88},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(400, 300, 3))

        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertLessEqual(region["bbox"][0], 96)
        self.assertLessEqual(region["bbox"][1], 96)
        self.assertGreaterEqual(region["bbox"][2], 180)
        self.assertGreaterEqual(region["bbox"][3], 186)
        self.assertEqual(region["kind"], "cluster")

    def test_keeps_far_apart_regions_separate(self):
        texts = [
            {"bbox": [10, 20, 60, 55], "tipo": "fala", "confidence": 0.82},
            {"bbox": [220, 260, 280, 320], "tipo": "fala", "confidence": 0.79},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(400, 300, 3))

        self.assertEqual(len(regions), 2)

    def test_pixel_mask_is_smaller_than_full_union_box(self):
        texts = [
            {"bbox": [100, 100, 180, 140], "tipo": "fala", "confidence": 0.91},
            {"bbox": [110, 150, 176, 186], "tipo": "fala", "confidence": 0.88},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(400, 300, 3))
        region = regions[0]
        pixel_mask = build_region_pixel_mask((400, 300), region)

        x1, y1, x2, y2 = region["bbox"]
        full_union_area = (x2 - x1) * (y2 - y1)
        masked_pixels = int(pixel_mask.sum() // 255)

        self.assertGreater(masked_pixels, 0)
        self.assertLess(masked_pixels, full_union_area)

    def test_build_mask_regions_prefers_text_geometry_over_broad_bbox(self):
        texts = [
            {
                "bbox": [0, 0, 140, 80],
                "text_pixel_bbox": [58, 28, 96, 44],
                "line_polygons": [[[58, 28], [96, 28], [96, 44], [58, 44]]],
                "balloon_bbox": [0, 0, 140, 80],
                "tipo": "sfx",
                "content_class": "noise",
                "balloon_type": "white",
            }
        ]

        regions = build_mask_regions(texts=texts, image_shape=(80, 140, 3))

        self.assertEqual(len(regions), 1)
        self.assertLessEqual(regions[0]["bbox"][0], 58)
        self.assertGreaterEqual(regions[0]["bbox"][2], 96)
        self.assertGreater(regions[0]["bbox"][0], 10)
        self.assertLess(regions[0]["bbox"][2], 130)

    def test_region_pixel_mask_prefers_line_geometry_and_clips_real_bubble_id(self):
        bubble_mask = np.zeros((80, 140), dtype=np.uint8)
        bubble_mask[:, :70] = 3
        bubble_mask[:, 70:] = 4
        text = {
            "bbox": [0, 0, 140, 80],
            "text_pixel_bbox": [56, 28, 96, 44],
            "line_polygons": [[[56, 28], [96, 28], [96, 44], [56, 44]]],
            "bubble_mask": bubble_mask,
            "bubble_id": 3,
            "tipo": "sfx",
            "content_class": "noise",
            "balloon_type": "white",
            "skip_processing": True,
            "preserve_original": True,
        }
        region = {"bbox": [0, 0, 140, 80], "texts": [text], "tipo": "fala"}

        mask = build_region_pixel_mask((80, 140), region)

        self.assertGreater(int(np.count_nonzero(mask[:, :70])), 0)
        self.assertEqual(int(np.count_nonzero(mask[:, 70:])), 0)
        self.assertLess(int(np.count_nonzero(mask)), 1400)

    def test_region_pixel_mask_normalizes_3d_bubble_mask(self):
        bubble_mask = np.zeros((80, 140, 3), dtype=np.uint8)
        bubble_mask[:, :70, :] = 3
        bubble_mask[:, 70:, :] = 4
        text = {
            "bbox": [0, 0, 140, 80],
            "text_pixel_bbox": [56, 28, 96, 44],
            "line_polygons": [[[56, 28], [96, 28], [96, 44], [56, 44]]],
            "bubble_mask": bubble_mask,
            "bubble_id": 3,
            "tipo": "sfx",
            "content_class": "noise",
            "balloon_type": "white",
            "skip_processing": True,
            "preserve_original": True,
        }
        region = {"bbox": [0, 0, 140, 80], "texts": [text], "tipo": "fala"}

        mask = build_region_pixel_mask((80, 140), region)

        self.assertEqual(mask.ndim, 2)
        self.assertGreater(int(np.count_nonzero(mask[:, :70])), 0)
        self.assertEqual(int(np.count_nonzero(mask[:, 70:])), 0)
        self.assertLess(int(np.count_nonzero(mask)), 1400)

    def test_balloon_mask_from_block_does_not_synthesize_bbox_masks(self):
        block = {
            "bbox": [20, 20, 80, 60],
            "balloon_bbox": [10, 10, 100, 70],
        }

        self.assertIsNone(balloon_mask_from_block(block, (100, 140, 3)))

    def test_bbox_overreach_ratio_uses_text_geometry_bbox(self):
        text = {
            "bbox": [10, 10, 110, 110],
            "text_pixel_bbox": [45, 45, 65, 65],
        }

        ratio = bbox_overreach_ratio(text, (160, 160, 3))

        self.assertGreater(ratio, 20.0)

    def test_build_inpaint_mask_records_overreach_and_mask_diagnostics(self):
        text = {
            "bbox": [0, 0, 160, 160],
            "text_pixel_bbox": [70, 70, 90, 90],
            "balloon_polygon": [[75, 75], [80, 75], [80, 80], [75, 80]],
        }

        mask = build_inpaint_mask(text, (160, 160, 3), image_rgb=np.full((160, 160, 3), 255, dtype=np.uint8))

        self.assertIsNotNone(mask)
        self.assertIn("bbox_overreach_critical", text["qa_flags"])
        self.assertNotIn("mask_density_high", text.get("qa_flags", []))
        self.assertNotIn("mask_outside_balloon_critical", text.get("qa_flags", []))

    def test_line_polygons_downgrade_broad_bbox_overreach(self):
        text = {
            "bbox": [0, 0, 160, 160],
            "text_pixel_bbox": [70, 70, 90, 90],
            "line_polygons": [[[70, 70], [90, 70], [90, 90], [70, 90]]],
            "balloon_bbox": [40, 40, 120, 120],
        }

        mask = build_inpaint_mask(text, (160, 160, 3), image_rgb=np.full((160, 160, 3), 255, dtype=np.uint8))

        self.assertIsNotNone(mask)
        self.assertNotIn("bbox_overreach", text.get("qa_flags", []))
        self.assertNotIn("bbox_overreach_critical", text.get("qa_flags", []))
        overreach = text["qa_metrics"]["bbox_overreach"]
        self.assertEqual(overreach["bbox"], [0, 0, 160, 160])
        self.assertEqual(overreach["text_geometry_bbox"], [70, 70, 91, 91])
        self.assertTrue(overreach["has_line_polygon_geometry"])
        self.assertFalse(overreach["broad_bbox_drives_mask"])

    def test_build_inpaint_mask_prefers_line_geometry_when_raw_image_mask_is_overbroad(self):
        text = {
            "bbox": [10, 10, 150, 150],
            "text_pixel_bbox": [58, 58, 102, 104],
            "line_polygons": [
                [[60, 60], [100, 60], [100, 78], [60, 78]],
                [[65, 84], [95, 84], [95, 102], [65, 102]],
            ],
            "balloon_bbox": [10, 10, 150, 150],
        }
        raw = np.zeros((160, 160), dtype=np.uint8)
        raw[10:150, 10:150] = 255

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, (160, 160, 3), image_rgb=np.full((160, 160, 3), 245, dtype=np.uint8))

        self.assertIsNotNone(mask)
        self.assertGreater(int(mask[66, 70]), 0)
        self.assertGreater(int(mask[92, 80]), 0)
        self.assertEqual(int(mask[20, 20]), 0)
        self.assertEqual(int(mask[132, 132]), 0)
        self.assertLess(int(np.count_nonzero(mask)), 4500)
        self.assertNotIn("mask_density_high", text.get("qa_flags", []))

    def test_build_inpaint_mask_does_not_raise_density_for_tight_white_text_anchor(self):
        text = {
            "bbox": [48, 24, 138, 72],
            "text_pixel_bbox": [48, 24, 138, 72],
            "line_polygons": [
                [[50, 26], [136, 26], [136, 42], [50, 42]],
                [[58, 50], [128, 50], [128, 68], [58, 68]],
            ],
            "balloon_bbox": [44, 20, 142, 76],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }
        raw = np.zeros((100, 180), dtype=np.uint8)
        raw[20:76, 44:142] = 255

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, (100, 180, 3), image_rgb=np.full((100, 180, 3), 255, dtype=np.uint8))

        self.assertIsNotNone(mask)
        self.assertNotIn("mask_density_high", text.get("qa_flags", []))

    def test_build_inpaint_mask_density_guard_uses_raw_or_erodes_before_balloon_clip(self):
        text = {
            "bbox": [42, 34, 77, 69],
            "text_pixel_bbox": [42, 34, 77, 69],
            "balloon_bbox": [0, 0, 120, 100],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "texto",
        }
        raw = np.zeros((110, 140), dtype=np.uint8)
        raw[34:69, 42:77] = 255
        image = np.full((110, 140, 3), 255, dtype=np.uint8)

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertGreaterEqual(int(np.count_nonzero(mask)), int(np.count_nonzero(raw)))
        self.assertNotIn("mask_density_high", text.get("qa_flags", []))
        self.assertNotIn("mask_density_guard", text.get("qa_metrics", {}))

    def test_density_guard_keeps_line_polygon_when_raw_mask_misses_long_text_tail(self):
        text = {
            "text": "2o ** regional fireman recruitment test",
            "bbox": [48, 36, 356, 64],
            "source_bbox": [48, 36, 356, 64],
            "text_pixel_bbox": [73, 44, 118, 54],
            "line_polygons": [[[71, 35], [566, 35], [566, 57], [71, 57]]],
            "balloon_bbox": [45, 28, 356, 70],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "bubble_id": "bubble-status-bar",
            "tipo": "texto",
        }
        raw = np.zeros((100, 600), dtype=np.uint8)
        raw[44:55, 73:119] = 255
        image = np.full((100, 600, 3), 255, dtype=np.uint8)

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(mask[48, 520]), 0)
        self.assertEqual(int(mask[12, 520]), 0)
        self.assertNotIn("mask_density_guard", text.get("qa_metrics", {}))

    def test_tight_synthetic_bubble_reference_does_not_raise_density_or_critical_outside(self):
        text = {
            "text": "What happened?",
            "translated": "O que aconteceu?",
            "bbox": [258, 29, 287, 57],
            "text_pixel_bbox": [228, 39, 237, 52],
            "line_polygons": [[[225, 32], [407, 32], [407, 52], [225, 52]]],
            "balloon_bbox": [216, 27, 249, 64],
            "bubble_id": "page_004_band_078_bubble_001",
            "bubble_mask_bbox": [258, 29, 287, 57],
            "bubble_inner_bbox": [270, 41, 275, 45],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "texto",
        }
        raw = np.zeros((90, 430), dtype=np.uint8)
        raw[39:53, 228:238] = 255
        image = np.full((90, 430, 3), 255, dtype=np.uint8)

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertNotIn("mask_density_high", text.get("qa_flags", []))
        self.assertNotIn("mask_outside_balloon_critical", text.get("qa_flags", []))
        self.assertNotIn("mask_density_guard", text.get("qa_metrics", {}))

    def test_build_inpaint_mask_recovers_nearby_glyph_inside_tight_white_reference(self):
        image = np.full((110, 180, 3), 255, dtype=np.uint8)
        image[22:44, 37:43] = 0
        image[22:26, 34:47] = 0
        image[40:44, 34:47] = 0
        image[26:44, 60:126] = 0
        text = {
            "text": "will reach",
            "bbox": [58, 20, 142, 54],
            "text_pixel_bbox": [58, 20, 142, 54],
            "source_bbox": [34, 18, 142, 54],
            "balloon_bbox": [34, 18, 142, 54],
            "line_polygons": [[[58, 20], [142, 20], [142, 54], [58, 54]]],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "texto",
        }

        mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(mask[32, 40]), 0)
        self.assertGreater(int(mask[34, 92]), 0)
        self.assertEqual(int(mask[8, 40]), 0)

    def test_build_inpaint_mask_recovers_line_polygon_component_missing_from_raw_mask(self):
        text = {
            "bbox": [24, 16, 150, 104],
            "text_pixel_bbox": [42, 24, 132, 98],
            "line_polygons": [
                [[40, 24], [132, 24], [132, 40], [40, 40]],
                [[46, 50], [128, 50], [128, 66], [46, 66]],
                [[70, 82], [112, 82], [112, 98], [70, 98]],
            ],
            "balloon_bbox": [10, 8, 170, 112],
            "balloon_type": "white",
        }
        raw = np.zeros((128, 192), dtype=np.uint8)
        raw[23:67, 38:134] = 255

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, (128, 192, 3), image_rgb=np.full((128, 192, 3), 180, dtype=np.uint8))

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(mask[90, 88]), 0)
        self.assertNotIn("raw_mask_missing_geometry_components", text.get("qa_metrics", {}))

    def test_textured_status_panel_uses_geometry_when_raw_mask_is_fragmented(self):
        text = {
            "text": "FULL POWER!",
            "bbox": [38, 18, 430, 106],
            "text_pixel_bbox": [38, 18, 430, 106],
            "line_polygons": [[[38, 18], [430, 18], [430, 106], [38, 106]]],
            "balloon_bbox": [38, 18, 430, 106],
            "balloon_type": "textured",
            "block_profile": "standard",
            "background_rgb": [199, 172, 107],
            "tipo": "narracao",
        }
        image = np.full((130, 460, 3), (199, 172, 107), dtype=np.uint8)
        raw = np.zeros((130, 460), dtype=np.uint8)
        raw[28:82, 50:110] = 255
        raw[28:82, 140:200] = 255
        raw[28:82, 230:290] = 255
        raw[28:82, 320:380] = 255

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(mask[60, 390]), 0)
        self.assertNotIn("textured_status_geometry_mask_recovery", text.get("qa_metrics", {}))

    def test_card_panel_context_uses_geometry_when_raw_mask_is_missing(self):
        text = {
            "text": "CLASS: FLOATING SPIRIT",
            "bbox": [446, 104, 648, 170],
            "text_pixel_bbox": [446, 104, 648, 170],
            "line_polygons": [
                [[449, 107], [646, 107], [646, 128], [449, 128]],
                [[472, 142], [620, 142], [620, 169], [472, 169]],
            ],
            "balloon_bbox": [420, 84, 680, 190],
            "bubble_mask_source": "rejected_derived_bubble_mask",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "card_panel_text_context": True,
            "qa_flags": ["raw_text_evidence_missing"],
            "tipo": "narracao",
        }
        image = np.full((256, 800, 3), (58, 84, 112), dtype=np.uint8)

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=None):
            mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertGreater(int(mask[116, 500]), 0)
        self.assertEqual(int(mask[70, 500]), 0)

    def test_fragmented_geometry_union_does_not_fill_white_balloon(self):
        text = {
            "text": "FULL POWER!",
            "bbox": [38, 18, 430, 106],
            "text_pixel_bbox": [38, 18, 430, 106],
            "line_polygons": [[[38, 18], [430, 18], [430, 106], [38, 106]]],
            "balloon_bbox": [10, 8, 450, 122],
            "balloon_type": "white",
            "block_profile": "white_balloon",
            "background_rgb": [255, 255, 255],
            "tipo": "fala",
        }
        image = np.full((130, 460, 3), 255, dtype=np.uint8)
        raw = np.zeros((130, 460), dtype=np.uint8)
        raw[28:82, 50:110] = 255
        raw[28:82, 140:200] = 255
        raw[28:82, 230:290] = 255
        raw[28:82, 320:380] = 255

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(mask[60, 20]), 0)
        self.assertNotIn("textured_status_geometry_mask_recovery", text.get("qa_metrics", {}))

    def test_textured_status_panel_bridges_fragments_on_same_line(self):
        text = {
            "text": "FORMATION POWER STATUS",
            "bbox": [18, 12, 244, 48],
            "text_pixel_bbox": [20, 16, 240, 42],
            "line_polygons": [
                [[20, 18], [92, 18], [92, 40], [20, 40]],
                [[152, 18], [240, 18], [240, 40], [152, 40]],
            ],
            "balloon_bbox": [18, 12, 244, 48],
            "balloon_type": "textured",
            "block_profile": "standard",
            "background_rgb": [132, 84, 45],
            "tipo": "narracao",
        }
        image = np.full((72, 280, 3), (132, 84, 45), dtype=np.uint8)
        raw = np.zeros((72, 280), dtype=np.uint8)
        raw[18:40, 20:92] = 255
        raw[18:40, 152:240] = 255

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(mask[28, 122]), 0)
        self.assertNotIn("status_line_fragment_bridge", text.get("qa_metrics", {}))

    def test_status_line_bridge_does_not_apply_to_white_balloon(self):
        text = {
            "text": "FORMATION POWER STATUS",
            "bbox": [18, 12, 244, 48],
            "text_pixel_bbox": [20, 16, 240, 42],
            "line_polygons": [
                [[20, 18], [92, 18], [92, 40], [20, 40]],
                [[152, 18], [240, 18], [240, 40], [152, 40]],
            ],
            "balloon_bbox": [8, 8, 260, 60],
            "balloon_type": "white",
            "block_profile": "white_balloon",
            "background_rgb": [255, 255, 255],
            "tipo": "fala",
        }
        image = np.full((72, 280, 3), 255, dtype=np.uint8)
        raw = np.zeros((72, 280), dtype=np.uint8)
        raw[18:40, 20:92] = 255
        raw[18:40, 152:240] = 255

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(mask[28, 122]), 0)
        self.assertNotIn("status_line_fragment_bridge", text.get("qa_metrics", {}))

    def test_textured_clipped_rotated_text_forces_broad_search_without_frame(self):
        image = np.full((220, 220, 3), [150, 112, 68], dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (190, 190), (245, 245, 235), 2)
        cv2.putText(image, "MISS", (122, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (248, 248, 238), 2, cv2.LINE_AA)
        cv2.putText(image, "TEXT", (118, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (248, 248, 238), 2, cv2.LINE_AA)
        block = {
            "bbox": [20, 20, 190, 190],
            "balloon_bbox": [20, 20, 190, 190],
            "balloon_type": "textured",
            "confidence": 0.34,
            "rotation_deg": 74.31,
            "qa_flags": ["TEXT_CLIPPED"],
            "line_polygons": [
                [[54, 54], [68, 50], [104, 154], [90, 158]],
            ],
            "text_pixel_bbox": [54, 50, 104, 158],
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(mask[82, 130]), 0)
        self.assertGreater(int(mask[124, 126]), 0)
        self.assertEqual(int(mask[20, 80]), 0)
        self.assertEqual(int(mask[80, 20]), 0)

    def test_mask_outside_balloon_downgrades_when_clip_retains_text(self):
        text = {
            "bbox": [10, 10, 90, 90],
            "text_pixel_bbox": [10, 10, 90, 90],
            "balloon_bbox": [35, 10, 100, 90],
        }

        mask = build_inpaint_mask(text, (120, 120, 3), image_rgb=np.full((120, 120, 3), 255, dtype=np.uint8))

        self.assertIsNotNone(mask)
        self.assertNotIn("mask_outside_balloon", text.get("qa_flags", []))
        self.assertNotIn("mask_outside_balloon_critical", text.get("qa_flags", []))

    def test_line_polygons_downgrade_mask_outside_even_when_most_raw_mask_is_clipped(self):
        text = {
            "bbox": [10, 10, 90, 90],
            "text_pixel_bbox": [10, 10, 90, 90],
            "line_polygons": [[[10, 10], [90, 10], [90, 90], [10, 90]]],
            "balloon_bbox": [72, 10, 100, 90],
        }

        mask = build_inpaint_mask(text, (120, 120, 3), image_rgb=np.full((120, 120, 3), 255, dtype=np.uint8))

        self.assertIsNotNone(mask)
        self.assertNotIn("mask_outside_balloon", text.get("qa_flags", []))
        self.assertNotIn("mask_outside_balloon_critical", text.get("qa_flags", []))

    def test_textured_tight_anchor_does_not_clip_expanded_glyph_mask_to_fake_balloon(self):
        text = {
            "bbox": [24, 24, 76, 36],
            "text_pixel_bbox": [20, 22, 80, 38],
            "line_polygons": [[[20, 22], [80, 22], [80, 38], [20, 38]]],
            "balloon_bbox": [20, 22, 80, 38],
            "balloon_type": "textured",
            "layout_profile": "standard",
        }
        raw = np.zeros((80, 100), dtype=np.uint8)
        raw[23:37, 19:81] = 255

        with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
            mask = build_inpaint_mask(text, (80, 100, 3), image_rgb=np.full((80, 100, 3), 120, dtype=np.uint8))

        self.assertIsNotNone(mask)
        self.assertGreater(int(mask[23, 19]), 0)
        self.assertGreater(int(mask[36, 80]), 0)
        self.assertNotIn("mask_outside_balloon", text.get("qa_flags", []))
        self.assertNotIn("mask_outside_balloon_critical", text.get("qa_flags", []))

    def test_does_not_merge_by_balloon_bbox_iou_when_text_bboxes_are_separate(self):
        texts = [
            {"bbox": [40, 50, 80, 75], "balloon_bbox": [20, 20, 180, 120], "tipo": "fala"},
            {"bbox": [125, 80, 165, 105], "balloon_bbox": [25, 25, 175, 125], "tipo": "fala"},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(180, 220, 3))

        self.assertEqual(len(regions), 2)

    def test_text_bbox_proximity_can_merge_without_balloon_bbox_veto(self):
        texts = [
            {"bbox": [100, 100, 150, 130], "balloon_bbox": [90, 90, 151, 140], "tipo": "fala"},
            {"bbox": [152, 100, 200, 130], "balloon_bbox": [151, 90, 210, 140], "tipo": "fala"},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(240, 260, 3))

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["kind"], "cluster")

    def test_inpaint_only_route_allows_watermark_mask_without_skip_processing(self):
        block = {
            "bbox": [20, 20, 120, 48],
            "text_pixel_bbox": [24, 24, 116, 44],
            "tipo": "watermark",
            "content_class": "url_watermark",
            "route_action": "inpaint_only",
            "route_reason": "watermark_detected",
            "skip_processing": False,
        }

        mask = build_inpaint_mask(block, (90, 160, 3), image_rgb=np.full((90, 160, 3), 255, dtype=np.uint8))

        self.assertIsNotNone(mask)
        self.assertGreater(int(mask.sum()), 0)

    def test_koharu_text_mask_ignores_balloon_bbox_and_class_fields(self):
        image = np.full((80, 140, 3), 255, dtype=np.uint8)
        block = {
            "text": "SIM, NAO FUNCIONA",
            "bbox": [0, 0, 140, 80],
            "balloon_bbox": [0, 0, 140, 80],
            "content_class": "noise",
            "tipo": "sfx",
            "balloon_type": "white",
            "skip_processing": True,
            "preserve_original": True,
            "line_polygons": [
                [[42, 26], [98, 26], [98, 36], [42, 36]],
                [[42, 42], [104, 42], [104, 52], [42, 52]],
            ],
            "bubble_id": "b1",
            "bubble_mask": None,
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertEqual(int(np.count_nonzero(mask[:, :8])), 0)
        self.assertEqual(int(np.count_nonzero(mask[:, 132:])), 0)
        self.assertNotIn("mask_density_high", block.get("qa_flags", []))

    def test_build_inpaint_mask_derives_real_white_bubble_mask_without_segmenter(self):
        image = np.zeros((96, 180, 3), dtype=np.uint8)
        cv2.ellipse(image, (90, 46), (55, 32), 0, 0, 360, (255, 255, 255), -1)
        image[44:52, 24:140] = 8
        block = {
            "text": "Hey HoSu. R-resident number...",
            "bbox": [18, 32, 145, 58],
            "text_pixel_bbox": [18, 34, 142, 56],
            "line_polygons": [
                [[50, 34], [112, 34], [112, 43], [50, 43]],
                [[18, 44], [142, 44], [142, 56], [18, 56]],
            ],
            "balloon_bbox": [18, 12, 145, 80],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(np.count_nonzero(mask[44:52, 42:135])), 0)
        self.assertEqual(int(np.count_nonzero(mask[44:52, 24:34])), 0)
        self.assertIn("derived_white_bubble_mask", block.get("qa_metrics", {}))

    def test_derived_white_bubble_mask_rejects_plain_rectangular_crop(self):
        from inpainter.mask_builder import _derive_white_bubble_mask_from_image

        image = np.full((96, 180, 3), 24, dtype=np.uint8)
        image[18:76, 30:150] = 255
        block = {
            "bbox": [64, 40, 116, 54],
            "text_pixel_bbox": [64, 40, 116, 54],
            "balloon_bbox": [30, 18, 150, 76],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        mask = _derive_white_bubble_mask_from_image(block, image.shape, image)

        self.assertIsNone(mask)
        self.assertEqual(
            block.get("qa_metrics", {}).get("derived_white_bubble_mask", {}).get("reason"),
            "rejected_rectangular_crop",
        )

    def test_derived_white_bubble_mask_retries_expanded_crop_for_tight_ellipse_crop(self):
        from inpainter.mask_builder import _derive_white_bubble_mask_from_image

        image = np.full((160, 180, 3), 210, dtype=np.uint8)
        cv2.ellipse(image, (90, 80), (35, 22), 0, 0, 360, (255, 255, 255), -1)
        block = {
            "bbox": [78, 63, 102, 97],
            "text_pixel_bbox": [78, 63, 102, 97],
            "balloon_bbox": [78, 63, 102, 97],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        mask = _derive_white_bubble_mask_from_image(block, image.shape, image)

        self.assertIsNotNone(mask)
        assert mask is not None
        metrics = block.get("qa_metrics", {}).get("derived_white_bubble_mask", {})
        self.assertEqual(metrics.get("source"), "image_white_region")
        mask_bbox = metrics.get("mask_bbox") or [0, 0, 0, 0]
        bbox_area = max(1, (mask_bbox[2] - mask_bbox[0]) * (mask_bbox[3] - mask_bbox[1]))
        self.assertLess(metrics.get("mask_pixels", 0) / float(bbox_area), 0.92)

        ys, xs = np.where(mask > 0)
        density = int(np.count_nonzero(mask)) / float((int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1))
        self.assertLess(density, 0.92)
        self.assertLess(int(metrics.get("balloon_bbox", [0, 0, 0, 0])[0]), block["balloon_bbox"][0])
        self.assertLess(int(metrics.get("balloon_bbox", [0, 0, 0, 0])[1]), block["balloon_bbox"][1])

    def test_derived_balloon_component_must_enclose_text_anchor(self):
        from inpainter.mask_builder import choose_balloon_component

        text_bbox = [359, 203, 708, 353]
        wrong_component = [348, 761, 793, 1032]

        candidate = choose_balloon_component(
            text_bbox=text_bbox,
            candidates=[wrong_component],
            page_size=(800, 1200),
        )

        self.assertIsNone(candidate)

    def test_derived_balloon_component_accepts_component_around_text(self):
        from inpainter.mask_builder import choose_balloon_component

        text_bbox = [359, 203, 708, 353]
        real_component = [272, 140, 793, 413]

        candidate = choose_balloon_component(
            text_bbox=text_bbox,
            candidates=[real_component],
            page_size=(800, 1200),
        )

        self.assertEqual(candidate, real_component)

    def test_derived_white_bubble_mask_retries_when_tight_ellipse_is_misread_as_rectangular(self):
        from inpainter.mask_builder import _derive_white_bubble_mask_from_image

        image = np.full((180, 260, 3), 18, dtype=np.uint8)
        cv2.ellipse(image, (155, 95), (64, 42), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (155, 95), (64, 42), 0, 0, 360, (0, 0, 0), 2)
        block = {
            "bbox": [140, 90, 170, 105],
            "text_pixel_bbox": [140, 90, 170, 105],
            "balloon_bbox": [122, 70, 205, 130],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        mask = _derive_white_bubble_mask_from_image(block, image.shape, image)

        self.assertIsNotNone(mask)
        assert mask is not None
        metrics = block.get("qa_metrics", {}).get("derived_white_bubble_mask", {})
        self.assertEqual(metrics.get("source"), "image_white_region")
        self.assertLess(int(metrics.get("balloon_bbox", [0, 0, 0, 0])[0]), 122)
        ys, xs = np.where(mask > 0)
        density = int(np.count_nonzero(mask)) / float((int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1))
        self.assertLess(density, 0.92)

    def test_derived_white_bubble_mask_refines_connected_oval_noise(self):
        from inpainter.mask_builder import _derive_white_bubble_mask_from_image

        image = np.full((180, 260, 3), 24, dtype=np.uint8)
        cv2.ellipse(image, (140, 98), (58, 34), 0, 0, 360, (255, 255, 255), -1)
        image[48:78, 84:132] = 255
        block = {
            "bbox": [120, 88, 160, 108],
            "text_pixel_bbox": [120, 88, 160, 108],
            "balloon_bbox": [70, 40, 210, 150],
            "background_rgb": [255, 255, 255],
            "layout_profile": "white_balloon",
        }

        mask = _derive_white_bubble_mask_from_image(block, image.shape, image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(mask[58, 92]), 0)
        self.assertGreater(int(mask[98, 140]), 0)
        metrics = block["qa_metrics"]["derived_white_bubble_mask"]
        self.assertEqual(metrics["shape_refiner"]["shape_kind"], "oval")
        self.assertGreater(metrics["shape_refiner"]["removed_pixels"], 200)

    def test_derived_white_bubble_mask_uses_outline_when_white_page_connects_to_oval(self):
        from inpainter.mask_builder import _derive_white_bubble_mask_from_image

        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        image[88:180, :] = 235
        cv2.ellipse(image, (130, 70), (76, 44), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (130, 70), (76, 44), 0, 0, 360, (0, 0, 0), 2)
        image[62:70, 92:168] = 12
        block = {
            "bbox": [92, 62, 168, 70],
            "text_pixel_bbox": [92, 62, 168, 70],
            "line_polygons": [[[92, 62], [168, 62], [168, 70], [92, 70]]],
            "balloon_bbox": [0, 0, 240, 126],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        mask = _derive_white_bubble_mask_from_image(block, image.shape, image)

        self.assertIsNotNone(mask)
        assert mask is not None
        metrics = block.get("qa_metrics", {}).get("derived_white_bubble_mask", {})
        self.assertEqual(metrics.get("source"), "outline_seeded_contour")
        mask_bbox = metrics.get("mask_bbox") or [0, 0, 0, 0]
        self.assertGreater(mask_bbox[0], 30)
        self.assertLess(mask_bbox[2], 215)
        self.assertLess(mask_bbox[3], 125)
        self.assertEqual(int(mask[12, 12]), 0)
        self.assertGreater(int(mask[70, 130]), 0)

    def test_derived_white_bubble_mask_uses_seeded_outline_for_starburst_on_white_page(self):
        from inpainter.mask_builder import _derive_white_bubble_mask_from_image

        image = np.full((190, 280, 3), 255, dtype=np.uint8)
        starburst = np.asarray(
            [
                [84, 38],
                [104, 44],
                [124, 32],
                [145, 43],
                [169, 34],
                [181, 56],
                [205, 63],
                [188, 82],
                [197, 106],
                [170, 108],
                [151, 124],
                [129, 112],
                [102, 122],
                [92, 98],
                [68, 90],
                [82, 68],
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [starburst], True, (0, 0, 0), 2)
        image[70:82, 118:170] = 18
        block = {
            "bbox": [118, 70, 170, 82],
            "text_pixel_bbox": [118, 70, 170, 82],
            "line_polygons": [[[118, 70], [170, 70], [170, 82], [118, 82]]],
            "balloon_bbox": [40, 12, 236, 142],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        mask = _derive_white_bubble_mask_from_image(block, image.shape, image)

        self.assertIsNotNone(mask)
        assert mask is not None
        metrics = block.get("qa_metrics", {}).get("derived_white_bubble_mask", {})
        self.assertEqual(metrics.get("source"), "outline_seeded_contour")
        self.assertEqual(int(mask[20, 50]), 0)
        self.assertGreater(int(mask[76, 144]), 0)
        mask_bbox = metrics.get("mask_bbox") or [0, 0, 0, 0]
        self.assertGreater(mask_bbox[0], 55)
        self.assertLess(mask_bbox[2], 218)
        self.assertLess(mask_bbox[3], 135)

    def test_derived_white_bubble_mask_uses_seeded_outline_to_ignore_neighbor_text_on_white_page(self):
        from inpainter.mask_builder import _derive_white_bubble_mask_from_image

        image = np.full((220, 360, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (150, 92), (92, 60), 0, 0, 360, (0, 0, 0), 2)
        image[72:80, 92:208] = 16
        image[86:94, 88:212] = 16
        image[100:108, 118:184] = 16
        image[86:94, 302:338] = 16
        block = {
            "bbox": [88, 70, 212, 110],
            "text_pixel_bbox": [88, 70, 212, 110],
            "line_polygons": [
                [[92, 72], [208, 72], [208, 80], [92, 80]],
                [[88, 86], [212, 86], [212, 94], [88, 94]],
                [[118, 100], [184, 100], [184, 108], [118, 108]],
            ],
            "balloon_bbox": [0, 0, 350, 178],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        mask = _derive_white_bubble_mask_from_image(block, image.shape, image)

        self.assertIsNotNone(mask)
        assert mask is not None
        metrics = block.get("qa_metrics", {}).get("derived_white_bubble_mask", {})
        self.assertEqual(metrics.get("source"), "outline_seeded_contour")
        self.assertGreater(int(mask[92, 150]), 0)
        self.assertEqual(int(mask[90, 320]), 0)
        mask_bbox = metrics.get("mask_bbox") or [0, 0, 0, 0]
        self.assertGreater(mask_bbox[0], 35)
        self.assertLess(mask_bbox[2], 255)
        self.assertLess(mask_bbox[3], 160)

    def test_connected_white_crop_is_modeled_as_oval_not_text_side_rectangle(self):
        from inpainter.mask_builder import _derive_white_bubble_mask_from_image

        image = np.full((220, 360, 3), 245, dtype=np.uint8)
        image[:, :52] = [186, 198, 218]
        cv2.ellipse(image, (230, 112), (126, 70), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (230, 112), (126, 70), 0, 0, 360, (38, 38, 38), 1)
        image[92:100, 160:302] = 20
        image[112:120, 140:320] = 20
        image[132:140, 170:292] = 20
        block = {
            "bbox": [140, 92, 320, 140],
            "text_pixel_bbox": [140, 92, 320, 140],
            "line_polygons": [
                [[160, 92], [302, 92], [302, 100], [160, 100]],
                [[140, 112], [320, 112], [320, 120], [140, 120]],
                [[170, 132], [292, 132], [292, 140], [170, 140]],
            ],
            "balloon_bbox": [52, 36, 360, 188],
            "background_rgb": [255, 255, 255],
            "layout_profile": "white_balloon",
        }

        mask = _derive_white_bubble_mask_from_image(block, image.shape, image)

        self.assertIsNotNone(mask)
        assert mask is not None
        metrics = block.get("qa_metrics", {}).get("derived_white_bubble_mask", {})
        self.assertEqual(metrics.get("source"), "outline_seeded_contour")
        self.assertIn(
            metrics.get("shape_refiner", {}).get("shape_kind"),
            {"outline_seeded_contour", "ellipse_from_connected_white_crop"},
        )
        mask_bbox = metrics.get("mask_bbox") or [0, 0, 0, 0]
        self.assertLess(mask_bbox[0], 110)
        self.assertGreater(mask_bbox[2], 330)
        self.assertGreater(int(mask[116, 230]), 0)
        self.assertEqual(int(mask[45, 60]), 0)

    def test_dark_card_context_does_not_derive_fake_white_bubble_from_text(self):
        from inpainter.mask_builder import _derive_white_bubble_mask_from_image

        image = np.zeros((160, 260, 3), dtype=np.uint8)
        cv2.rectangle(image, (70, 42), (190, 98), (22, 20, 12), -1)
        image[64:76, 102:158] = [245, 245, 230]
        block = {
            "bbox": [102, 64, 158, 76],
            "text_pixel_bbox": [102, 64, 158, 76],
            "line_polygons": [[[102, 64], [158, 64], [158, 76], [102, 76]]],
            "balloon_bbox": [102, 64, 158, 76],
            "bubble_mask_bbox": [102, 64, 158, 76],
            "background_rgb": [42, 36, 18],
            "layout_profile": "standard",
        }

        mask = _derive_white_bubble_mask_from_image(block, image.shape, image)

        self.assertIsNone(mask)
        self.assertNotIn("derived_white_bubble_mask", block.get("qa_metrics", {}))

    def test_derived_white_bubble_mask_accepts_rectangular_balloon_with_border(self):
        from inpainter.mask_builder import _derive_white_bubble_mask_from_image

        image = np.full((96, 180, 3), 24, dtype=np.uint8)
        cv2.rectangle(image, (30, 18), (150, 76), (255, 255, 255), -1)
        cv2.rectangle(image, (30, 18), (150, 76), (0, 0, 0), 2)
        block = {
            "bbox": [64, 40, 116, 54],
            "text_pixel_bbox": [64, 40, 116, 54],
            "balloon_bbox": [28, 16, 152, 78],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        mask = _derive_white_bubble_mask_from_image(block, image.shape, image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertEqual(
            block.get("qa_metrics", {}).get("derived_white_bubble_mask", {}).get("source"),
            "derived_rectangular_balloon",
        )

    def test_component_bubble_cleaner_accepts_derived_white_bubble_mask(self):
        image = np.zeros((96, 180, 3), dtype=np.uint8)
        cv2.ellipse(image, (90, 46), (55, 32), 0, 0, 360, (255, 255, 255), -1)
        image[15:19, 70:110] = 8
        image[43:51, 76:104] = 8
        block = {
            "text": "center text",
            "bbox": [50, 20, 130, 60],
            "text_pixel_bbox": [50, 20, 130, 60],
            "line_polygons": [[[50, 20], [130, 20], [130, 60], [50, 60]]],
            "balloon_bbox": [18, 12, 145, 80],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(np.count_nonzero(mask[15:19, 70:110])), 0)
        self.assertGreater(int(np.count_nonzero(mask[43:51, 76:104])), 0)
        self.assertIn("component_bubble_cleaner", block.get("qa_metrics", {}))

    def test_build_inpaint_mask_uses_smaller_elliptic_text_dilation(self):
        image = np.full((64, 96, 3), 255, dtype=np.uint8)
        block = {
            "bbox": [0, 0, 96, 64],
            "line_polygons": [[[40, 28], [52, 28], [52, 40], [40, 40]]],
            "font_size": 18,
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(mask[34, 46]), 0)
        self.assertEqual(int(mask[25, 37]), 0)

    def test_koharu_text_mask_keeps_dark_glyph_pixels_inside_text_bbox(self):
        image = np.full((64, 96, 3), 240, dtype=np.uint8)
        image[30:34, 36:42] = 10
        image[30:34, 52:58] = 10
        bubble_mask = np.zeros((64, 96), dtype=np.uint8)
        bubble_mask[12:52, 16:80] = 3
        block = {
            "bbox": [0, 0, 96, 64],
            "text_pixel_bbox": [32, 28, 64, 36],
            "line_polygons": [[[32, 28], [64, 28], [64, 36], [32, 36]]],
            "bubble_mask": bubble_mask,
            "bubble_id": 3,
            "balloon_bbox": [0, 0, 96, 64],
            "content_class": "noise",
            "skip_processing": True,
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertTrue(np.all(mask[30:34, 36:42] == 255))
        self.assertTrue(np.all(mask[30:34, 52:58] == 255))
        self.assertEqual(int(mask[8, 8]), 0)

    def test_koharu_text_mask_clips_expansion_to_matching_bubble_id(self):
        image = np.full((64, 96, 3), 255, dtype=np.uint8)
        bubble_mask = np.zeros((64, 96), dtype=np.uint8)
        bubble_mask[:, :48] = 3
        bubble_mask[:, 48:] = 4
        block = {
            "bbox": [0, 0, 96, 64],
            "line_polygons": [[[40, 24], [54, 24], [54, 36], [40, 36]]],
            "bubble_id": 3,
            "bubble_mask": bubble_mask,
            "font_size": 28,
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(np.count_nonzero(mask[:, :48])), 0)
        self.assertEqual(int(np.count_nonzero(mask[:, 48:])), 0)

    def test_build_inpaint_mask_uses_local_real_bubble_mask_as_page_clip(self):
        image = np.full((84, 130, 3), 255, dtype=np.uint8)
        bubble_mask = np.zeros((52, 96), dtype=np.uint8)
        cv2.ellipse(bubble_mask, (48, 26), (38, 20), 0, 0, 360, 255, -1)
        block = {
            "bbox": [22, 21, 98, 63],
            "text_pixel_bbox": [22, 21, 98, 63],
            "line_polygons": [
                [[22, 21], [98, 21], [98, 63], [22, 63]],
            ],
            "bubble_id": "bubble_001",
            "bubble_mask": bubble_mask,
            "bubble_mask_bbox": [12, 16, 108, 68],
            "detected_font_size_px": 38,
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        page_bubble = np.zeros((84, 130), dtype=np.uint8)
        page_bubble[16:68, 12:108] = bubble_mask
        eroded = cv2.erode(
            page_bubble,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        outside_eroded = int(np.count_nonzero((mask > 0) & (eroded == 0)))
        self.assertEqual(outside_eroded, 0)

    def test_component_bubble_cleaner_rejects_outline_pixels_with_real_bubble_mask(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        bubble_mask = np.zeros((80, 100), dtype=np.uint8)
        bubble_mask[10:64, 12:88] = 3
        image[10:15, 36:52] = 8
        image[32:40, 44:56] = 8
        block = {
            "bbox": [0, 0, 100, 80],
            "text_pixel_bbox": [12, 10, 88, 64],
            "line_polygons": [[[12, 10], [88, 10], [88, 64], [12, 64]]],
            "bubble_mask": bubble_mask,
            "bubble_id": 3,
            "balloon_bbox": [12, 10, 88, 64],
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(np.count_nonzero(mask[10:15, 36:52])), 0)
        self.assertGreater(int(np.count_nonzero(mask[32:40, 44:56])), 0)
        self.assertIn("component_bubble_cleaner", block.get("qa_metrics", {}))

    def test_white_balloon_final_mask_drops_unanchored_outline_sliver(self):
        image = np.full((90, 160, 3), 255, dtype=np.uint8)
        bubble_mask = np.zeros((90, 160), dtype=np.uint8)
        cv2.ellipse(bubble_mask, (80, 44), (64, 34), 0, 0, 360, 255, -1)
        component_mask = np.zeros((90, 160), dtype=np.uint8)
        component_mask[44:56, 50:112] = 255
        component_mask[20:36, 136:144] = 255
        block = {
            "bbox": [0, 0, 160, 90],
            "text_pixel_bbox": [44, 20, 140, 60],
            "source_bbox": [44, 20, 140, 60],
            "bubble_mask": bubble_mask,
            "balloon_bbox": [16, 10, 148, 78],
            "layout_profile": "white_balloon",
        }

        with patch(
            "inpainter.mask_builder._build_component_bubble_cleaner_mask",
            return_value=(component_mask, "component_bubble_cleaner", None),
        ):
            mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(int(np.count_nonzero(mask[20:36, 136:144])), 0)
        self.assertGreater(int(np.count_nonzero(mask[44:56, 50:112])), 0)
        self.assertIn("white_balloon_unanchored_component_filter", block.get("qa_metrics", {}))

    def test_component_bubble_cleaner_preserves_raw_floor_on_dark_cards(self):
        image = np.full((72, 140, 3), 18, dtype=np.uint8)
        image[28:36, 44:98] = 245
        raw_floor = np.zeros((72, 140), dtype=np.uint8)
        raw_floor[28:36, 44:98] = 255
        smaller_component = np.zeros((72, 140), dtype=np.uint8)
        smaller_component[30:34, 60:82] = 255
        bubble_mask = np.ones((72, 140), dtype=np.uint8) * 3
        block = {
            "bbox": [40, 24, 102, 40],
            "text_pixel_bbox": [44, 28, 98, 36],
            "line_polygons": [[[44, 28], [98, 28], [98, 36], [44, 36]]],
            "bubble_mask": bubble_mask,
            "bubble_id": 3,
            "background_rgb": [18, 18, 18],
            "layout_profile": "dark_panel",
        }

        with patch(
            "inpainter.mask_builder.build_raw_text_mask_from_image",
            return_value=raw_floor,
        ), patch(
            "inpainter.mask_builder.build_notanother_text_mask",
            return_value=(smaller_component, {"accepted_components": 1}),
        ):
            mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreaterEqual(int(np.count_nonzero(mask)), int(np.count_nonzero(raw_floor)))
        self.assertTrue(np.all(mask[28:36, 44:98] == 255))
        self.assertIn("component_bubble_cleaner_raw_floor_preserved", block.get("qa_metrics", {}))

    def test_empty_real_bubble_mask_does_not_fallback_to_text_bbox(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        block = {
            "bbox": [0, 0, 100, 80],
            "text_pixel_bbox": [20, 20, 80, 54],
            "bubble_mask": np.zeros((80, 100), dtype=np.uint8),
            "bubble_id": 3,
            "balloon_bbox": [0, 0, 100, 80],
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNone(mask)
        self.assertEqual(block.get("mask_evidence", {}).get("kind"), "none")

    def test_sfx_outside_bubble_is_not_cleaned_by_bubble_pipeline(self):
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        image[12:20, 88:110] = 8
        bubble_mask = np.zeros((80, 120), dtype=np.uint8)
        bubble_mask[20:62, 18:78] = 3
        block = {
            "bbox": [84, 8, 114, 24],
            "text_pixel_bbox": [88, 12, 110, 20],
            "line_polygons": [[[86, 10], [112, 10], [112, 22], [86, 22]]],
            "bubble_mask": bubble_mask,
            "bubble_id": 3,
            "text": "끼익",
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        if mask is not None:
            self.assertEqual(int(np.count_nonzero(mask[12:20, 88:110])), 0)

    def test_art_text_without_real_bubble_mask_does_not_clean_distant_art(self):
        image = np.full((96, 140, 3), 245, dtype=np.uint8)
        image[20:76, 20:24] = 12
        image[44:52, 72:96] = 8
        block = {
            "bbox": [60, 34, 108, 62],
            "text_pixel_bbox": [72, 44, 96, 52],
            "line_polygons": [[[68, 40], [100, 40], [100, 56], [68, 56]]],
            "text": "ART",
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(np.count_nonzero(mask[44:52, 72:96])), 0)
        self.assertEqual(int(np.count_nonzero(mask[20:76, 20:24])), 0)

    def test_korean_sfx_does_not_expand_cleanup_from_nearby_dialogue_bubble(self):
        image = np.full((96, 140, 3), 255, dtype=np.uint8)
        bubble_mask = np.zeros((96, 140), dtype=np.uint8)
        bubble_mask[24:72, 20:86] = 7
        image[44:52, 44:62] = 8
        image[34:60, 100:118] = 8
        block = {
            "bbox": [32, 34, 74, 62],
            "text_pixel_bbox": [44, 44, 62, 52],
            "line_polygons": [[[40, 38], [68, 38], [68, 58], [40, 58]]],
            "bubble_mask": bubble_mask,
            "bubble_id": 7,
            "text": "HELLO",
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(np.count_nonzero(mask[44:52, 44:62])), 0)
        self.assertEqual(int(np.count_nonzero(mask[34:60, 100:118])), 0)

    def test_ambiguous_string_bubble_id_multi_label_mask_does_not_fallback_to_bbox(self):
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        image[28:36, 68:82] = 8
        bubble_mask = np.zeros((80, 100), dtype=np.uint8)
        bubble_mask[:, :50] = 3
        bubble_mask[:, 50:] = 4
        block = {
            "bbox": [0, 0, 100, 80],
            "text_pixel_bbox": [10, 20, 90, 48],
            "line_polygons": [[[10, 20], [90, 20], [90, 48], [10, 48]]],
            "bubble_mask": bubble_mask,
            "bubble_id": "bubble-status-bar",
            "balloon_bbox": [0, 0, 100, 80],
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNone(mask)
        self.assertEqual(block.get("mask_evidence", {}).get("kind"), "none")

    def test_contained_broad_bbox_can_merge_even_with_distinct_balloon_bboxes(self):
        texts = [
            {"bbox": [8, 49, 749, 828], "balloon_bbox": [303, 673, 691, 806], "tipo": "fala"},
            {"bbox": [296, 672, 693, 848], "balloon_bbox": [365, 820, 627, 843], "tipo": "fala"},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(900, 800, 3))

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["kind"], "cluster")

    def test_sfx_regions_do_not_merge_with_plain_dialogue_regions(self):
        texts = [
            {"bbox": [50, 50, 100, 80], "balloon_bbox": [30, 30, 160, 110], "tipo": "fala"},
            {"bbox": [105, 55, 145, 85], "balloon_bbox": [30, 30, 160, 110], "tipo": "sfx", "text": "쿵"},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(180, 220, 3))

        self.assertEqual(len(regions), 2)

    def test_sfx_build_inpaint_mask_uses_glyph_evidence_not_rectangle_fill(self):
        image = np.full((120, 220, 3), 236, dtype=np.uint8)
        cv2.rectangle(image, (42, 34), (54, 78), (12, 12, 12), -1)
        cv2.rectangle(image, (42, 68), (78, 80), (12, 12, 12), -1)
        cv2.rectangle(image, (132, 34), (144, 82), (12, 12, 12), -1)
        cv2.rectangle(image, (132, 34), (166, 46), (12, 12, 12), -1)
        cv2.rectangle(image, (152, 58), (166, 82), (12, 12, 12), -1)
        block = {
            "bbox": [20, 20, 200, 100],
            "text": "쿵",
            "content_class": "sfx",
            "script": "hangul",
            "route_action": "translate_sfx_inpaint_render",
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        bbox_area = (block["bbox"][2] - block["bbox"][0]) * (block["bbox"][3] - block["bbox"][1])
        self.assertLess(int(np.count_nonzero(mask)), int(bbox_area * 0.45))
        self.assertEqual(block.get("mask_evidence", {}).get("kind"), "sfx_glyph_mask")
        self.assertGreaterEqual(block.get("mask_evidence", {}).get("component_count", 0), 2)
        self.assertEqual(int(mask[25, 25]), 0)

    def test_recovered_dark_bubble_mask_preserves_full_text_bbox_when_raw_subcovers(self):
        image = np.zeros((120, 260, 3), dtype=np.uint8)
        cv2.rectangle(image, (104, 48), (166, 74), (245, 245, 245), -1)
        bubble_mask = np.zeros((120, 260), dtype=np.uint8)
        cv2.ellipse(bubble_mask, (145, 62), (110, 46), 0, 0, 360, 255, -1)
        block = {
            "bbox": [42, 38, 220, 86],
            "text_pixel_bbox": [42, 38, 220, 86],
            "line_polygons": [[[104, 48], [166, 48], [166, 74], [104, 74]]],
            "bubble_mask": bubble_mask,
            "bubble_mask_bbox": [20, 12, 250, 110],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["candidate_crop_direct_paddle_reocr", "partial_dark_bubble_lobe_reocr"],
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(np.count_nonzero(mask[42:82, 42:96])), 0)
        self.assertGreater(int(np.count_nonzero(mask[42:82, 172:220])), 0)
        self.assertIn("dark_bubble_recovered_text_bbox_floor", block.get("qa_flags", []))

    def test_dark_connected_bubble_contract_rejects_broad_text_bbox_union(self):
        source = np.zeros((240, 360), dtype=np.uint8)
        source[78:92, 198:280] = 255
        source[104:118, 210:292] = 255
        bubble_mask = np.zeros((240, 360), dtype=np.uint8)
        cv2.ellipse(bubble_mask, (116, 104), (96, 68), 0, 0, 360, 255, -1)
        cv2.ellipse(bubble_mask, (250, 104), (100, 68), 0, 0, 360, 255, -1)
        cv2.rectangle(bubble_mask, (116, 70), (250, 138), 255, -1)
        block = {
            "bbox": [18, 24, 350, 198],
            "text_pixel_bbox": [18, 24, 350, 198],
            "line_polygons": [
                [[198, 78], [280, 78], [280, 92], [198, 92]],
                [[210, 104], [292, 104], [292, 118], [210, 118]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": [
                "partial_dark_bubble_lobe_reocr",
                "detected_dark_bubble_without_text_reocr",
                "dark_bubble_full_crop_reocr_replaced",
            ],
        }

        with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "1"}):
            mask = _enforce_visual_text_only_inpaint_contract(
                block,
                source.copy(),
                source.copy(),
                bubble_mask,
                radius=6,
            )

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertLess(int(np.count_nonzero(mask)), int(np.count_nonzero(bubble_mask) * 0.28))
        self.assertEqual(int(mask[178, 250]), 0)
        self.assertIn("dark_connected_bubble_broad_bbox_union_skipped", block.get("qa_flags", []))
        self.assertNotIn("inpaint_mask_contract_text_bbox_union", block.get("qa_metrics", {}))

    def test_visual_text_contract_closes_hollow_source_and_keeps_minimal_expansion(self):
        source = np.zeros((160, 340), dtype=np.uint8)
        cv2.rectangle(source, (42, 42), (152, 64), 255, 2)
        cv2.rectangle(source, (48, 72), (168, 96), 255, 2)
        cv2.rectangle(source, (202, 56), (306, 82), 255, 2)
        cv2.rectangle(source, (214, 92), (314, 116), 255, 2)
        bubble_mask = np.zeros((160, 340), dtype=np.uint8)
        cv2.ellipse(bubble_mask, (104, 76), (88, 58), 0, 0, 360, 255, -1)
        cv2.ellipse(bubble_mask, (260, 86), (82, 50), 0, 0, 360, 255, -1)
        cv2.rectangle(bubble_mask, (104, 52), (260, 108), 255, -1)
        block = {
            "bbox": [20, 20, 328, 140],
            "text_pixel_bbox": [42, 42, 314, 116],
            "line_polygons": [
                [[42, 42], [152, 42], [152, 64], [42, 64]],
                [[48, 72], [168, 72], [168, 96], [48, 96]],
                [[202, 56], [306, 56], [306, 82], [202, 82]],
                [[214, 92], [314, 92], [314, 116], [214, 116]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["dark_bubble_connected_lobes_promoted"],
        }

        mask = _enforce_visual_text_only_inpaint_contract(
            block,
            source.copy(),
            source.copy(),
            bubble_mask,
            radius=6,
        )

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(mask[53, 96]), 0)
        self.assertGreater(int(mask[104, 260]), 0)
        self.assertGreater(int(np.count_nonzero(mask)), int(np.count_nonzero(source)))
        self.assertLess(int(np.count_nonzero(mask)), int(np.count_nonzero(bubble_mask) * 0.52))
        self.assertIn("visual_text_source_gap_close", block.get("qa_metrics", {}))
        contract = block.get("qa_metrics", {}).get("inpaint_mask_contract", {})
        self.assertGreater(int(contract.get("expanded_pixels") or 0), int(contract.get("source_pixels") or 0))

    def test_visual_text_gap_close_fills_line_local_gaps_without_full_bbox(self):
        source = np.zeros((90, 260), dtype=np.uint8)
        cv2.rectangle(source, (24, 20), (86, 38), 255, -1)
        cv2.rectangle(source, (116, 20), (182, 38), 255, -1)
        cv2.rectangle(source, (42, 54), (118, 72), 255, -1)
        cv2.rectangle(source, (148, 54), (218, 72), 255, -1)

        closed = _close_visual_text_source_mask_gaps(source)

        self.assertGreater(int(closed[29, 102]), 0)
        self.assertGreater(int(closed[63, 134]), 0)
        self.assertEqual(int(closed[8, 102]), 0)
        self.assertLess(int(np.count_nonzero(closed)), 16000)

    def test_dark_connected_bubble_contract_uses_full_text_bbox_when_source_bbox_subcovers(self):
        block = {
            "bbox": [244, 113, 313, 223],
            "source_bbox": [244, 113, 313, 223],
            "text_pixel_bbox": [131, 108, 337, 222],
            "line_polygons": [
                [[150, 108], [326, 108], [326, 141], [150, 141]],
                [[131, 147], [337, 147], [337, 179], [131, 179]],
                [[169, 179], [304, 183], [303, 222], [168, 218]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": [
                "dark_bubble_connected_lobes_promoted",
                "dark_bubble_connected_lobe_passthrough",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
            ],
            "qa_metrics": {
                "negative_evidence": [{"bbox": [132, 107, 340, 222]}],
                "image_dark_bubble_mask": {"anchor_bbox": [131, 108, 337, 222]},
            },
        }

        mask, bbox = _text_bbox_contract_mask(block, (360, 640))

        self.assertIsNotNone(mask)
        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertLessEqual(bbox[0], 131)
        self.assertGreaterEqual(bbox[2], 337)
        self.assertGreaterEqual(bbox[2] - bbox[0], 200)
        self.assertNotIn("dark_connected_compact_text_bbox_rejected_undercoverage", block.get("qa_flags", []))

    def test_dark_connected_bubble_contract_rejects_compact_source_bbox_undercoverage(self):
        source = np.zeros((360, 640), dtype=np.uint8)
        source[93:243, 228:329] = 255
        block = {
            "source_bbox": [244, 113, 313, 223],
            "bbox": [244, 113, 313, 223],
            "ocr_text_bbox": [244, 113, 313, 223],
            "line_polygons": [
                [[150, 108], [326, 108], [326, 141], [150, 141]],
                [[131, 147], [337, 147], [337, 179], [131, 179]],
                [[169, 179], [304, 183], [303, 222], [168, 218]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["dark_bubble_connected_lobes_promoted"],
            "qa_metrics": {
                "negative_evidence": [{"bbox": [132, 107, 340, 222]}],
                "image_dark_bubble_mask": {"anchor_bbox": [131, 108, 337, 222]},
            },
        }

        with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "1"}):
            mask = _enforce_visual_text_only_inpaint_contract(
                block,
                source.copy(),
                source.copy(),
                None,
                radius=6,
            )

        self.assertIsNotNone(mask)
        assert mask is not None
        bbox = [int(v) for v in np.r_[np.where(mask > 0)[1].min(), np.where(mask > 0)[0].min(), np.where(mask > 0)[1].max() + 1, np.where(mask > 0)[0].max() + 1]]
        self.assertLessEqual(bbox[0], 131)
        self.assertGreaterEqual(bbox[2], 337)
        self.assertIn("dark_connected_compact_text_bbox_rejected_undercoverage", block.get("qa_flags", []))

    def test_dark_connected_bubble_contract_replaces_tiny_source_with_reliable_text_bbox(self):
        source = np.zeros((360, 800), dtype=np.uint8)
        source[246:258, 512:565] = 255
        block = {
            "bbox": [442, 224, 711, 303],
            "source_bbox": [442, 224, 711, 303],
            "text_pixel_bbox": [442, 224, 711, 303],
            "line_polygons": [
                [[442, 224], [710, 224], [710, 252], [442, 252]],
                [[472, 260], [691, 260], [691, 303], [472, 303]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["dark_bubble_connected_lobes_promoted"],
            "qa_metrics": {
                "negative_evidence": [{"bbox": [441, 223, 710, 302]}],
                "image_dark_bubble_mask": {"anchor_bbox": [442, 224, 711, 303]},
            },
        }

        with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "1"}):
            mask = _enforce_visual_text_only_inpaint_contract(
                block,
                source.copy(),
                source.copy(),
                None,
                radius=6,
            )

        self.assertIsNotNone(mask)
        assert mask is not None
        ys, xs = np.where(mask > 0)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        self.assertLessEqual(bbox[0], 442)
        self.assertGreaterEqual(bbox[2], 711)
        self.assertIn("dark_connected_bubble_compact_bbox_replaced_aggregate_source", block.get("qa_flags", []))
        self.assertEqual(
            block.get("qa_metrics", {})
            .get("inpaint_mask_contract_text_bbox_replaced_aggregate_source", {})
            .get("reason"),
            "dark_connected_bubble_tiny_source_text_reference_bbox",
        )
        self.assertNotIn("dark_connected_bubble_broad_bbox_union_skipped", block.get("qa_flags", []))

    def test_mixed_sfx_and_white_bubble_detaches_sfx_before_inpaint_contract(self):
        block = {
            "bbox": [192, 176, 630, 1025],
            "source_bbox": [192, 176, 630, 1025],
            "text_pixel_bbox": [192, 176, 630, 1025],
            "text": "1/ WHAT'S THA",
            "original": "1/ WHAT'S THA",
            "translated": "1/ O QUE É ISSO",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bubble_mask_bbox": [0, 0, 726, 1127],
            "line_polygons": [
                [[192, 176], [388, 176], [388, 397], [192, 397]],
                [[447, 1000], [630, 1000], [630, 1025], [447, 1025]],
            ],
            "qa_flags": ["dark_bubble_ellipse_bbox_mask", "dark_bubble_recovered_text_bbox_floor"],
            "qa_metrics": {
                "derived_white_bubble_mask": {
                    "mask_bbox": [260, 344, 686, 1127],
                    "mask_pixels": 102595,
                }
            },
        }

        cleaned = _detach_mixed_sfx_from_white_bubble_block(block, 800, 1200)

        self.assertEqual(cleaned.get("bubble_mask_source"), "image_white_bubble_mask")
        self.assertEqual(cleaned.get("bubble_mask_bbox"), [260, 344, 686, 1127])
        self.assertEqual(cleaned.get("bbox"), [447, 1000, 631, 1026])
        self.assertEqual(cleaned.get("translated"), "O QUE É ISSO")
        self.assertIn("mixed_sfx_detached_from_white_bubble", cleaned.get("qa_flags") or [])
        self.assertNotIn("dark_bubble_recovered_text_bbox_floor", cleaned.get("qa_flags") or [])

    def test_build_inpaint_mask_forces_rect_panel_over_stale_dark_ellipse_mask(self):
        image = np.zeros((360, 720, 3), dtype=np.uint8)
        cv2.putText(image, "MAIN QUEST", (113, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
        stale_bubble = np.zeros((360, 720), dtype=np.uint8)
        cv2.ellipse(stale_bubble, (194, 227), (156, 105), 0, 0, 360, 255, -1)
        block = {
            "bbox": [113, 165, 259, 236],
            "text_pixel_bbox": [113, 165, 259, 236],
            "line_polygons": [[[113, 165], [259, 165], [259, 236], [113, 236]]],
            "bubble_mask": stale_bubble,
            "bubble_mask_bbox": [37, 122, 350, 332],
            "balloon_bbox": [37, 122, 350, 332],
            "bubble_mask_source": "image_dark_bubble_mask",
            "bubble_mask_shape": "ellipse",
            "bubble_mask_ellipse": {"center": [193.5, 227.0], "axes": [313.0, 210.0], "angle": 0.0},
            "card_panel_text_context": True,
            "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertEqual(block.get("bubble_mask_source"), "image_dark_panel_mask")
        self.assertNotIn("bubble_mask_shape", block)
        self.assertNotIn("bubble_mask_ellipse", block)
        self.assertNotIn("dark_bubble_ellipse_bbox_mask", block.get("qa_flags") or [])
        self.assertIn("dark_panel_rect_from_dark_bubble_bbox", block.get("qa_flags") or [])
        self.assertEqual(block.get("bubble_mask_bbox"), [70, 136, 302, 265])

    def test_sfx_build_inpaint_mask_uses_sfx_route_even_with_noise_content_class(self):
        image = np.full((120, 220, 3), 236, dtype=np.uint8)
        cv2.rectangle(image, (42, 34), (54, 78), (12, 12, 12), -1)
        cv2.rectangle(image, (42, 68), (78, 80), (12, 12, 12), -1)
        cv2.rectangle(image, (132, 34), (144, 82), (12, 12, 12), -1)
        cv2.rectangle(image, (132, 34), (166, 46), (12, 12, 12), -1)
        cv2.rectangle(image, (152, 58), (166, 82), (12, 12, 12), -1)
        block = {
            "bbox": [20, 20, 200, 100],
            "text": "쿵",
            "tipo": "sfx",
            "content_class": "noise",
            "route_action": "translate_sfx_inpaint_render",
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertEqual(block.get("mask_evidence", {}).get("kind"), "sfx_glyph_mask")

    def test_visual_sfx_review_candidate_uses_glyph_mask_without_auto_route(self):
        image = np.full((140, 220, 3), 236, dtype=np.uint8)
        for x in range(22, 200, 6):
            cv2.line(image, (x, 20), (max(20, x - 80), 120), (24, 24, 24), 1)
        cv2.rectangle(image, (45, 34), (62, 98), (104, 28, 32), -1)
        cv2.rectangle(image, (45, 84), (120, 102), (104, 28, 32), -1)
        block = {
            "bbox": [20, 20, 200, 120],
            "text": "",
            "content_class": "sfx",
            "tipo": "sfx",
            "detector": "sfx_visual",
            "route_action": "review_required",
            "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
            "sfx": {"visual_detector": "sfx_visual", "inpaint_allowed": False},
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        self.assertEqual(block.get("mask_evidence", {}).get("kind"), "sfx_glyph_mask")
        self.assertEqual(block["route_action"], "review_required")

    def test_explicit_3d_bubble_mask_is_normalized_to_single_channel(self):
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        image[14:20, 14:22] = 0
        bubble_mask = np.zeros((40, 40, 3), dtype=np.uint8)
        bubble_mask[8:30, 8:30, :] = 7
        block = {
            "bbox": [12, 12, 24, 22],
            "text_pixel_bbox": [14, 14, 22, 20],
            "line_polygons": [[[14, 14], [22, 14], [22, 20], [14, 20]]],
            "bubble_mask": bubble_mask,
            "bubble_id": 7,
        }

        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(mask.ndim, 2)
        self.assertGreater(int(np.count_nonzero(mask)), 0)


if __name__ == "__main__":
    unittest.main()
