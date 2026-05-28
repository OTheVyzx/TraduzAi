import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inpainter.mask_builder import (
    balloon_mask_from_block,
    bbox_overreach_ratio,
    build_inpaint_mask,
    build_mask_regions,
    build_region_pixel_mask,
)


class MaskBuilderTests(unittest.TestCase):
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

    def test_contained_broad_bbox_can_merge_even_with_distinct_balloon_bboxes(self):
        texts = [
            {"bbox": [8, 49, 749, 828], "balloon_bbox": [303, 673, 691, 806], "tipo": "fala"},
            {"bbox": [296, 672, 693, 848], "balloon_bbox": [365, 820, 627, 843], "tipo": "fala"},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(900, 800, 3))

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["kind"], "cluster")

    def test_legacy_special_classes_merge_as_plain_text(self):
        texts = [
            {"bbox": [50, 50, 100, 80], "balloon_bbox": [30, 30, 160, 110], "tipo": "fala"},
            {"bbox": [105, 55, 145, 85], "balloon_bbox": [30, 30, 160, 110], "tipo": "sfx"},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(180, 220, 3))

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["kind"], "cluster")


if __name__ == "__main__":
    unittest.main()
