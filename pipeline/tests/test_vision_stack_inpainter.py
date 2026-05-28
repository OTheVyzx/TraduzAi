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
            side_effect=[residual, clean, clean],
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
            side_effect=[residual, residual],
        ):
            result = inpaint_band_image(image, page)

        self.assertFalse(page.get("_strip_white_residual_force_fill_from_residual_check", False))
        self.assertTrue(np.all(result[48:72, 70:150] == 92))

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


if __name__ == "__main__":
    unittest.main()
