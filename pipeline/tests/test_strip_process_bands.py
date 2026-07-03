"""Testes de process_bands.py."""

import sys
import unittest
from pathlib import Path
import json
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BandToPageDictTests(unittest.TestCase):
    def test_candidate_crop_reocr_does_not_replace_existing_white_balloon_ocr_with_sfx_prefix(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        ocr_page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "WHAT'S THAT?",
                    "raw_ocr": "WHAT'S THAT?",
                    "bbox": [439, 985, 659, 1031],
                    "text_pixel_bbox": [448, 1000, 657, 1025],
                    "line_polygons": [[[448, 1000], [657, 1000], [657, 1025], [448, 1025]]],
                    "block_profile": "white_balloon",
                    "layout_profile": "white_balloon",
                    "background_rgb": [253, 253, 253],
                }
            ],
            "_vision_blocks": [],
        }
        recovered_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text": "1/ WHAT'S THA",
                    "raw_ocr": "1/ WHAT'S THA",
                    "bbox": [192, 176, 630, 1025],
                    "source_bbox": [192, 176, 630, 1025],
                    "text_pixel_bbox": [192, 176, 630, 1025],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "block_profile": "dark_bubble",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["candidate_crop_direct_paddle_reocr", "dark_bubble_oval_reocr"],
                }
            ],
            "_vision_blocks": [{"bbox": [192, 176, 630, 1025], "reocr_candidate_index": 0}],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(ocr_page, recovered_page)

        self.assertEqual(merged, 0)
        self.assertEqual(len(ocr_page["texts"]), 1)
        self.assertEqual(ocr_page["texts"][0]["id"], "ocr_002")
        self.assertEqual(ocr_page["texts"][0]["text_pixel_bbox"], [448, 1000, 657, 1025])

    def test_candidate_crop_reocr_still_adds_real_dark_bubble_when_no_existing_white_ocr(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        ocr_page = {"texts": [], "_vision_blocks": []}
        recovered_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text": "SYSTEM MESSAGE",
                    "raw_ocr": "SYSTEM MESSAGE",
                    "bbox": [100, 120, 300, 180],
                    "source_bbox": [100, 120, 300, 180],
                    "text_pixel_bbox": [100, 120, 300, 180],
                    "bubble_mask_bbox": [60, 60, 360, 240],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "block_profile": "dark_bubble",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["candidate_crop_direct_paddle_reocr", "dark_bubble_oval_reocr"],
                    "reocr_candidate_index": 0,
                }
            ],
            "_vision_blocks": [{"bbox": [60, 60, 360, 240], "reocr_candidate_index": 0}],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(ocr_page, recovered_page)

        self.assertEqual(merged, 1)
        self.assertEqual(ocr_page["texts"][0]["text"], "SYSTEM MESSAGE")

    def test_candidate_crop_reocr_composite_does_not_replace_independent_lower_lobe_ocr(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        ocr_page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text_id": "ocr_002",
                    "text": "There can't be two kings in an underworld!",
                    "raw_ocr": "There can't be two kings in an underworld!",
                    "bbox": [321, 169, 544, 306],
                    "source_bbox": [321, 169, 544, 306],
                    "text_pixel_bbox": [321, 169, 544, 306],
                    "line_polygons": [[[321, 169], [544, 169], [544, 306], [321, 306]]],
                }
            ],
            "_vision_blocks": [],
        }
        recovered_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "this underworld! There can't be two kings in an underworld!",
                    "raw_ocr": "this underworld! There can't be two kings in an underworld!",
                    "bbox": [202, 10, 546, 309],
                    "source_bbox": [202, 10, 546, 309],
                    "text_pixel_bbox": [202, 10, 546, 309],
                    "bubble_mask_bbox": [0, 0, 647, 151],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "block_profile": "dark_bubble",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["candidate_crop_direct_paddle_reocr", "dark_bubble_oval_reocr"],
                    "reocr_candidate_index": 0,
                }
            ],
            "_vision_blocks": [{"bbox": [0, 0, 647, 151], "reocr_candidate_index": 0}],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(ocr_page, recovered_page)

        self.assertEqual(merged, 0)
        self.assertEqual(len(ocr_page["texts"]), 1)
        kept = ocr_page["texts"][0]
        self.assertEqual(kept["id"], "ocr_002")
        self.assertEqual(kept["text_pixel_bbox"], [321, 169, 544, 306])
        self.assertEqual(kept["text"], "There can't be two kings in an underworld!")

    def test_candidate_crop_reocr_same_lobe_suffix_can_replace_partial_ocr(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        ocr_page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text_id": "ocr_002",
                    "text": "You thought friendship was significant, but you were",
                    "raw_ocr": "You thought friendship was significant, but you were",
                    "bbox": [467, 264, 575, 346],
                    "source_bbox": [467, 264, 575, 346],
                    "text_pixel_bbox": [429, 270, 746, 339],
                    "line_polygons": [[[429, 270], [746, 270], [746, 339], [429, 339]]],
                }
            ],
            "_vision_blocks": [],
        }
        recovered_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "You thought friendship was significant, but you were only used by your friends.",
                    "raw_ocr": "You thought friendship was significant, but you were only used by your friends.",
                    "bbox": [426, 268, 748, 379],
                    "source_bbox": [426, 268, 748, 379],
                    "text_pixel_bbox": [426, 268, 748, 379],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "block_profile": "dark_bubble",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["candidate_crop_direct_paddle_reocr", "dark_bubble_oval_reocr"],
                    "reocr_candidate_index": 0,
                }
            ],
            "_vision_blocks": [{"bbox": [426, 268, 748, 379], "reocr_candidate_index": 0}],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(ocr_page, recovered_page)

        self.assertEqual(merged, 1)
        self.assertEqual(len(ocr_page["texts"]), 1)
        kept = ocr_page["texts"][0]
        self.assertEqual(kept["id"], "direct_paddle_reocr_001")
        self.assertIn("only used by your friends", kept["text"])

    def test_candidate_crop_reocr_same_lobe_prefix_can_replace_partial_ocr(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        ocr_page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text_id": "ocr_002",
                    "text": "guides the host of King Yeomra in establishing an underworld.",
                    "raw_ocr": "guides the host of King Yeomra in establishing an underworld.",
                    "bbox": [157, 220, 400, 295],
                    "source_bbox": [157, 220, 400, 295],
                    "text_pixel_bbox": [158, 216, 396, 318],
                    "line_polygons": [
                        [[158, 216], [396, 216], [396, 318], [158, 318]],
                    ],
                }
            ],
            "_vision_blocks": [],
        }
        recovered_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "I'm a system that guides the host of King Yeomra in establishing an underworld.",
                    "raw_ocr": "I'm a system that guides the host of King Yeomra in establishing an underworld.",
                    "bbox": [157, 176, 396, 321],
                    "source_bbox": [157, 176, 396, 321],
                    "text_pixel_bbox": [157, 176, 396, 321],
                    "line_polygons": [
                        [[157, 176], [396, 176], [396, 202], [157, 202]],
                        [[157, 210], [396, 210], [396, 236], [157, 236]],
                        [[157, 244], [396, 244], [396, 270], [157, 270]],
                        [[157, 278], [396, 278], [396, 321], [157, 321]],
                    ],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "block_profile": "dark_bubble",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["candidate_crop_direct_paddle_reocr", "dark_bubble_oval_reocr"],
                    "reocr_candidate_index": 0,
                }
            ],
            "_vision_blocks": [{"bbox": [120, 136, 423, 346], "reocr_candidate_index": 0}],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(ocr_page, recovered_page)

        self.assertEqual(merged, 1)
        self.assertEqual(len(ocr_page["texts"]), 1)
        kept = ocr_page["texts"][0]
        self.assertEqual(kept["id"], "direct_paddle_reocr_001")
        self.assertEqual(len(kept["line_polygons"]), 4)
        self.assertIn("I'm a system", kept["text"])

    def test_dark_bubble_full_crop_ocr_rejects_gibberish_prefix_before_existing_text(self):
        from strip.process_bands import _dark_bubble_full_crop_ocr_is_better

        old_text = "guides the host of King Yeomra in establishing an underworld."
        new_text = "masystemtha guides the host of King Yeomra in establishing an underworld"

        self.assertFalse(_dark_bubble_full_crop_ocr_is_better(old_text, new_text))

    def test_trailing_clipped_dark_bubble_fragment_is_removed_but_keeps_cleanup_bbox(self):
        from strip.process_bands import _strip_false_trailing_dark_bubble_fragment

        text = {
            "text": "That's the power of this underworld! Th",
            "raw_ocr": "That's the power of this underworld! Th",
            "original": "That's the power of this underworld! Th",
            "bbox": [164, 451, 406, 676],
            "source_bbox": [164, 451, 406, 676],
            "text_pixel_bbox": [164, 451, 406, 676],
            "line_polygons": [
                [[179, 451], [397, 457], [397, 494], [177, 488]],
                [[164, 499], [406, 499], [406, 533], [164, 533]],
                [[349, 658], [383, 659], [382, 676], [348, 675]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "qa_flags": ["candidate_crop_direct_paddle_reocr", "dark_bubble_oval_reocr"],
        }

        self.assertTrue(_strip_false_trailing_dark_bubble_fragment(text))

        self.assertEqual(text["text"], "That's the power of this underworld!")
        self.assertEqual(len(text["line_polygons"]), 2)
        self.assertEqual(text["text_pixel_bbox"], [164, 451, 406, 533])
        self.assertEqual(text["clipped_overlap_fragment_cleanup_bbox"]["removed_tail"], "Th")
        cleanup = text["qa_metrics"]["clipped_overlap_fragment_cleanup_bbox"]["bbox"]
        self.assertLessEqual(cleanup[0], 321)
        self.assertGreaterEqual(cleanup[2], 544)
        self.assertIn("false_dark_bubble_trailing_clipped_fragment_removed", text["qa_flags"])

    def test_high_conf_dark_light_text_reocr_gate_allows_borderline_bright_ratio(self):
        from strip.process_bands import _candidate_crop_reocr_allows_high_conf_dark_light_text

        evidence = {
            "has_inner_light_text": True,
            "inner_light_component_count": 31,
            "inner_light_area": 10696,
            "bright_pixel_ratio": 0.1754,
            "dark_pixel_ratio": 0.7264,
        }

        self.assertTrue(
            _candidate_crop_reocr_allows_high_conf_dark_light_text(evidence, confidence=0.9302)
        )

    def test_dark_panel_rejected_overbroad_keeps_full_rect_contract(self):
        from strip.process_bands import _normalize_dark_bubble_contracts_for_stage
        import numpy as np

        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [113, 165, 259, 236],
                    "text_pixel_bbox": [113, 165, 259, 236],
                    "balloon_bbox": [37, 122, 350, 332],
                    "bubble_mask_bbox": [37, 122, 350, 332],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "bubble_mask_shape": "ellipse",
                    "bubble_mask_ellipse": {
                        "center": [193.5, 227.0],
                        "axes": [313.0, 210.0],
                        "angle": 0.0,
                    },
                    "qa_flags": [
                        "connected_layout_disabled_dark_panel_visual_mask",
                        "dark_bubble_ellipse_bbox_mask",
                    ],
                    "qa_metrics": {
                        "image_dark_panel_mask_rejected": [
                            {
                                "reason": "overbroad_against_balloon_bbox",
                                "mask_bbox": [37, 122, 688, 332],
                                "reference_bbox": [37, 122, 350, 332],
                            }
                        ]
                    },
                }
            ]
        }

        _normalize_dark_bubble_contracts_for_stage(page, np.zeros((360, 720, 3), dtype=np.uint8))

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_source"], "image_dark_panel_mask")
        self.assertEqual(text["bubble_mask_bbox"], [37, 122, 350, 332])
        self.assertEqual(text["balloon_bbox"], [37, 122, 350, 332])
        self.assertGreater(text["bubble_inner_bbox"][2] - text["bubble_inner_bbox"][0], 220)
        self.assertEqual(text["block_profile"], "dark_panel")
        self.assertEqual(text["layout_profile"], "dark_panel")
        self.assertIn("dark_panel_full_bbox_selected", text.get("qa_flags") or [])
        self.assertNotIn("bubble_mask_shape", text)
        self.assertNotIn("bubble_mask_ellipse", text)
        self.assertNotIn("dark_bubble_ellipse_bbox_mask", text.get("qa_flags") or [])
        self.assertIn("image_dark_panel_mask", text.get("qa_metrics") or {})

    def test_connected_dark_bubble_stage_does_not_become_dark_panel_rect(self):
        from strip.process_bands import _normalize_dark_bubble_contracts_for_stage
        import numpy as np

        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [237, 119, 677, 335],
                    "text_pixel_bbox": [237, 119, 677, 335],
                    "balloon_bbox": [83, 32, 744, 691],
                    "bubble_mask_bbox": [83, 32, 744, 691],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "layout_profile": "connected_balloon",
                    "block_profile": "dark_bubble",
                    "balloon_subregions": [[83, 32, 385, 691], [385, 32, 744, 691]],
                    "connected_lobe_bboxes": [[83, 32, 385, 691], [385, 32, 744, 691]],
                    "connected_balloon_orientation": "left-right",
                    "qa_flags": [
                        "connected_layout_disabled_dark_panel_visual_mask",
                        "dark_connected_lobes_repaired_from_visual_mask",
                        "dark_bubble_connected_lobe_passthrough",
                    ],
                    "qa_metrics": {
                        "image_dark_bubble_mask": {
                            "source": "image_dark_bubble_mask",
                            "mask_bbox": [83, 32, 744, 691],
                        }
                    },
                }
            ]
        }

        _normalize_dark_bubble_contracts_for_stage(page, np.zeros((720, 800, 3), dtype=np.uint8))

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_source"], "image_dark_bubble_mask")
        self.assertEqual(text["layout_profile"], "connected_balloon")
        self.assertEqual(text["balloon_bbox"], [83, 32, 744, 691])
        self.assertIn("image_dark_bubble_mask", text.get("qa_metrics") or {})

    def test_short_dark_text_does_not_select_full_panel_bbox(self):
        from strip.process_bands import _normalize_dark_bubble_contracts_for_stage
        import numpy as np

        page = {
            "texts": [
                {
                    "id": "negative_dark_002",
                    "text": "It's simple.",
                    "translated": "É simples.",
                    "bbox": [63, 553, 183, 590],
                    "source_bbox": [63, 553, 183, 590],
                    "text_pixel_bbox": [63, 553, 183, 590],
                    "balloon_bbox": [3, 519, 243, 624],
                    "bubble_mask_bbox": [3, 519, 243, 624],
                    "bubble_mask_source": "image_dark_panel_mask",
                    "block_profile": "dark_panel",
                    "layout_profile": "dark_panel",
                    "qa_flags": ["dark_panel_full_bbox_selected"],
                }
            ]
        }

        _normalize_dark_bubble_contracts_for_stage(page, np.zeros((700, 300, 3), dtype=np.uint8))

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_source"], "image_dark_panel_mask")
        self.assertLessEqual(text["bubble_mask_bbox"][2] - text["bubble_mask_bbox"][0], 160)
        self.assertNotIn("dark_panel_full_bbox_selected", text.get("qa_flags") or [])
        self.assertIn("short_dark_text_full_panel_bbox_rejected", text.get("qa_flags") or [])

    def test_dark_panel_stage_prefers_detected_visual_rect_over_overbroad_candidate(self):
        from strip.process_bands import _normalize_dark_bubble_contracts_for_stage
        import cv2
        import numpy as np

        image = np.zeros((332, 800, 3), dtype=np.uint8)
        image[:, :] = (3, 6, 8)
        cv2.rectangle(image, (38, 122), (330, 269), (150, 165, 170), 2)
        cv2.putText(image, "MAIN QUEST", (104, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)
        cv2.putText(image, "SHORTLY", (132, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [84, 88, 286, 297],
                    "text_pixel_bbox": [113, 165, 259, 236],
                    "balloon_bbox": [37, 122, 349, 332],
                    "bubble_mask_bbox": [37, 122, 349, 332],
                    "bubble_inner_bbox": [116, 161, 254, 224],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": [
                        "connected_layout_disabled_dark_panel_visual_mask",
                        "dark_bubble_ellipse_bbox_mask",
                    ],
                    "qa_metrics": {
                        "image_dark_panel_mask_rejected": [
                            {
                                "reason": "overbroad_against_balloon_bbox",
                                "mask_bbox": [37, 122, 687, 332],
                                "reference_bbox": [37, 122, 349, 332],
                            }
                        ]
                    },
                }
            ]
        }

        _normalize_dark_bubble_contracts_for_stage(page, image)

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_source"], "image_dark_panel_mask")
        self.assertLessEqual(abs(text["bubble_mask_bbox"][0] - 38), 2)
        self.assertLessEqual(abs(text["bubble_mask_bbox"][1] - 122), 2)
        self.assertLessEqual(abs(text["bubble_mask_bbox"][2] - 331), 2)
        self.assertLessEqual(abs(text["bubble_mask_bbox"][3] - 270), 2)
        self.assertEqual(text["balloon_bbox"], text["bubble_mask_bbox"])
        self.assertIn("dark_panel_visual_rect_candidate_selected", text.get("qa_flags") or [])
        self.assertTrue(text["qa_metrics"]["image_dark_panel_mask"]["visual_rect_candidate_selected"])

    def test_band_source_style_evidence_applies_light_glow_on_colored_card(self):
        from strip.process_bands import _apply_band_source_style_evidence
        import cv2
        import numpy as np

        image = np.full((130, 260, 3), (128, 170, 252), dtype=np.uint8)
        cv2.putText(image, "The host", (54, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (238, 250, 255), 4, cv2.LINE_AA)
        cv2.putText(image, "The host", (54, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (238, 250, 255), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "The host",
                    "bbox": [48, 24, 182, 74],
                    "text_pixel_bbox": [50, 28, 184, 70],
                    "background_rgb": [252, 170, 128],
                    "bubble_mask_source": "derived_white_crop_rejected",
                    "route_action": "translate_inpaint_render",
                }
            ]
        }

        applied = _apply_band_source_style_evidence(page, image, font_detector=object())

        self.assertEqual(applied, 1)
        text = page["texts"][0]
        self.assertEqual(text["style_origin"], "source_detected")
        self.assertEqual(text["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertTrue(text["estilo"]["cor"].startswith("#"))
        self.assertGreater(text["style_confidence"], 0.7)

    def test_band_source_style_evidence_uses_font_detector_for_visual_card(self):
        from strip.process_bands import _apply_band_source_style_evidence
        import cv2
        import numpy as np

        class FakeFontDetector:
            def detect_with_score(self, crop, allow_default=True):
                return "LeagueGothic-Regular-VariableFont_wdth.ttf", 0.88

        image = np.full((130, 260, 3), (128, 170, 252), dtype=np.uint8)
        cv2.putText(image, "Synching", (54, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (238, 250, 255), 3, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "Synching is complete.",
                    "bbox": [48, 24, 202, 74],
                    "text_pixel_bbox": [50, 28, 204, 70],
                    "background_rgb": [252, 170, 128],
                    "bubble_mask_source": "image_white_bubble_mask",
                    "route_action": "translate_inpaint_render",
                }
            ]
        }

        applied = _apply_band_source_style_evidence(page, image, font_detector=FakeFontDetector())

        self.assertEqual(applied, 1)
        text = page["texts"][0]
        self.assertEqual(text["style_origin"], "source_detected")
        self.assertEqual(text["style_evidence"]["font_name"], "LeagueGothic-Regular-VariableFont_wdth.ttf")
        self.assertEqual(text["estilo"]["fonte"], "LeagueGothic-Regular-VariableFont_wdth.ttf")

    def test_band_source_style_evidence_does_not_override_white_balloon(self):
        from strip.process_bands import _apply_band_source_style_evidence
        import cv2
        import numpy as np

        image = np.full((120, 240, 3), 255, dtype=np.uint8)
        cv2.putText(image, "HELLO", (54, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "HELLO",
                    "bbox": [44, 32, 156, 78],
                    "text_pixel_bbox": [46, 34, 158, 76],
                    "background_rgb": [255, 255, 255],
                    "layout_profile": "white_balloon",
                    "route_action": "translate_inpaint_render",
                }
            ]
        }

        applied = _apply_band_source_style_evidence(page, image)

        self.assertEqual(applied, 0)
        self.assertNotEqual(page["texts"][0].get("style_origin"), "source_detected")

    def test_drop_suppressed_records_for_inpaint_removes_scanlator_caption_pair(self):
        from strip.process_bands import _drop_suppressed_records_for_inpaint

        page = {
            "idioma_origem": "en",
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "TEXT: DARLING KARAOKE",
                    "bbox": [30, 318, 217, 339],
                    "route_action": "translate_inpaint_render",
                },
                {
                    "id": "ocr_002",
                    "text": "PLEASE!",
                    "bbox": [40, 20, 140, 64],
                    "route_action": "translate_inpaint_render",
                },
            ],
            "_vision_blocks": [
                {"id": "ocr_001", "text": "TEXT: DARLING KARAOKE", "bbox": [30, 318, 217, 339]},
                {"id": "ocr_002", "text": "PLEASE!", "bbox": [40, 20, 140, 64]},
            ],
        }

        _drop_suppressed_records_for_inpaint(page)

        self.assertEqual([text["id"] for text in page["texts"]], ["ocr_002"])
        self.assertEqual([block["id"] for block in page["_vision_blocks"]], ["ocr_002"])

    def test_fill_binary_holes_does_not_turn_foreground_touching_corner_into_full_rect(self):
        from strip.process_bands import _fill_binary_holes
        import numpy as np

        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[:18, :24] = 255
        mask[6:10, 6:10] = 0

        filled = _fill_binary_holes(mask)

        self.assertGreater(int(filled[7, 7]), 0)
        self.assertEqual(int(filled[25, 35]), 0)

    def test_attach_ocr_trace_metadata_expands_merged_source_text_ids(self):
        from strip.process_bands import _attach_ocr_trace_metadata

        page = {
            "numero": 3,
            "texts": [
                {
                    "id": "ocr_001",
                    "source_text_ids": ["ocr_001", "ocr_002"],
                    "_merged_source_bboxes": [[10, 10, 30, 20], [34, 28, 70, 42]],
                    "ocr_merged_source_count": 2,
                }
            ],
            "_vision_blocks": [{"bbox": [10, 10, 50, 50]}],
        }

        _attach_ocr_trace_metadata(page, band_id="page_003_band_019")

        text = page["texts"][0]
        block = page["_vision_blocks"][0]
        self.assertEqual(
            text["source_trace_ids"],
            ["ocr_001@page_003_band_019", "ocr_002@page_003_band_019"],
        )
        self.assertEqual(text["_source_trace_ids"], text["source_trace_ids"])
        self.assertEqual(text["merge_reason"], "clustered_line_fragments")
        self.assertEqual(block["_source_trace_ids"], text["source_trace_ids"])
        self.assertEqual(block["_merged_source_bboxes"], text["_merged_source_bboxes"])

    def test_attach_ocr_trace_metadata_makes_duplicate_text_ids_unique(self):
        from strip.process_bands import _attach_ocr_trace_metadata

        page = {
            "numero": 2,
            "texts": [
                {"id": "ocr_001", "text": "the king of being a pushover...", "bbox": [476, 289, 649, 362]},
                {"id": "ocr_001", "text": "You were loyal to others", "bbox": [132, 116, 419, 231]},
            ],
            "_vision_blocks": [
                {"bbox": [476, 289, 649, 362]},
                {"bbox": [132, 116, 419, 231]},
            ],
        }

        _attach_ocr_trace_metadata(page, band_id="page_002_band_023")

        self.assertEqual([text["text_id"] for text in page["texts"]], ["ocr_001", "ocr_001_002"])
        self.assertEqual(
            [text["trace_id"] for text in page["texts"]],
            ["ocr_001@page_002_band_023", "ocr_001_002@page_002_band_023"],
        )
        self.assertEqual([block["text_id"] for block in page["_vision_blocks"]], ["ocr_001", "ocr_001_002"])

    def test_attach_ocr_trace_metadata_copies_mask_context_to_vision_block(self):
        from strip.process_bands import _attach_ocr_trace_metadata

        page = {
            "numero": 6,
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [468, 102, 608, 173],
                    "text_pixel_bbox": [451, 107, 643, 168],
                    "line_polygons": [
                        [[471, 107], [624, 107], [624, 134], [471, 134]],
                        [[449, 137], [645, 137], [645, 168], [449, 168]],
                    ],
                    "background_rgb": [118, 98, 49],
                    "bubble_mask_source": "derived_white_crop_rejected",
                    "bubble_mask_error": "derived_mask_not_anchored_to_text",
                    "bubble_mask_bbox": [514, 139, 586, 169],
                    "bubble_id": "page_006_band_107_bubble_001",
                    "card_panel_text_context": True,
                }
            ],
            "_vision_blocks": [{"bbox": [468, 102, 608, 173], "confidence": 0.56}],
        }

        _attach_ocr_trace_metadata(page, band_id="page_006_band_107")

        block = page["_vision_blocks"][0]
        self.assertEqual(block["background_rgb"], [118, 98, 49])
        self.assertEqual(block["text_pixel_bbox"], [451, 107, 643, 168])
        self.assertEqual(block["line_polygons"][0][0], [471, 107])
        self.assertEqual(block["bubble_mask_source"], "derived_white_crop_rejected")
        self.assertEqual(block["bubble_mask_error"], "derived_mask_not_anchored_to_text")
        self.assertEqual(block["bubble_mask_bbox"], [514, 139, 586, 169])
        self.assertEqual(block["bubble_id"], "page_006_band_107_bubble_001")
        self.assertTrue(block["card_panel_text_context"])

    def test_dark_bubble_reocr_rejects_cross_lobe_text_bbox(self):
        from strip.process_bands import _filter_dark_bubble_reocr_to_balloon

        texts = [
            {
                "id": "direct_paddle_reocr_001",
                "text": "I'm a system that guides the host I an",
                "bbox": [20, 20, 470, 190],
                "text_pixel_bbox": [20, 20, 470, 190],
                "qa_flags": ["candidate_crop_direct_paddle_reocr"],
            }
        ]
        blocks = [{"bbox": [20, 20, 470, 190], "text_pixel_bbox": [20, 20, 470, 190]}]

        kept_texts, kept_blocks = _filter_dark_bubble_reocr_to_balloon(
            texts,
            blocks,
            bubble_bbox=[0, 0, 500, 220],
        )

        self.assertEqual(kept_texts, [])
        self.assertEqual(kept_blocks, [])
        self.assertIn("dark_bubble_reocr_cross_lobe_rejected", texts[0]["qa_flags"])

    def test_dark_bubble_reocr_keeps_contained_text_bbox(self):
        from strip.process_bands import _filter_dark_bubble_reocr_to_balloon

        texts = [
            {
                "id": "direct_paddle_reocr_001",
                "text": "1,000 points...",
                "bbox": [80, 64, 242, 126],
                "text_pixel_bbox": [80, 64, 242, 126],
                "qa_flags": ["candidate_crop_direct_paddle_reocr"],
            }
        ]
        blocks = [{"bbox": [80, 64, 242, 126], "text_pixel_bbox": [80, 64, 242, 126]}]

        kept_texts, kept_blocks = _filter_dark_bubble_reocr_to_balloon(
            texts,
            blocks,
            bubble_bbox=[0, 0, 360, 220],
        )

        self.assertEqual(kept_texts, texts)
        self.assertEqual(kept_blocks, blocks)

    def test_dark_bubble_reocr_rejects_edge_touching_cross_lobe_text_bbox(self):
        from strip.process_bands import _filter_dark_bubble_reocr_to_balloon

        texts = [
            {
                "id": "direct_paddle_reocr_001",
                "text": "I'm a system that guides the host of King Yeomra in establishing an underworld",
                "bbox": [156, 103, 527, 306],
                "text_pixel_bbox": [156, 103, 527, 306],
                "qa_flags": ["candidate_crop_direct_paddle_reocr"],
            }
        ]
        blocks = [{"bbox": [156, 103, 527, 306], "text_pixel_bbox": [156, 103, 527, 306]}]

        kept_texts, kept_blocks = _filter_dark_bubble_reocr_to_balloon(
            texts,
            blocks,
            bubble_bbox=[53, 5, 527, 404],
        )

        self.assertEqual(kept_texts, [])
        self.assertEqual(kept_blocks, [])
        self.assertIn("dark_bubble_reocr_cross_lobe_rejected", texts[0]["qa_flags"])

    def test_dark_connected_lobe_reocr_is_not_dropped_by_wide_existing_bbox(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "I'm a system that guides the host of King Yeomra in establishing an underworld.",
                    "bbox": [158, 105, 396, 244],
                    "text_pixel_bbox": [158, 105, 396, 244],
                    "balloon_bbox": [104, 76, 453, 254],
                }
            ],
            "_vision_blocks": [],
        }
        recovered = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text": "I am called System.",
                    "bbox": [338, 136, 444, 185],
                    "text_pixel_bbox": [338, 136, 444, 185],
                    "balloon_bbox": [328, 100, 450, 252],
                    "bubble_mask_bbox": [328, 100, 450, 252],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": ["partial_dark_bubble_lobe_reocr"],
                    "reocr_candidate_index": 1001,
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [328, 100, 450, 252],
                    "reocr_candidate_index": 1001,
                }
            ],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(page, recovered)

        self.assertEqual(merged, 1)
        self.assertEqual(len(page["texts"]), 2)
        self.assertEqual(page["texts"][1]["text"], "I am called System.")

    def test_dark_bubble_reocr_strips_false_trailing_ocr_fragment(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        page = {"texts": [], "_vision_blocks": []}
        recovered = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text": "No one cared whether you lived or died. WI for",
                    "original": "No one cared whether you lived or died. WI for",
                    "raw_ocr": "No one cared whether you lived or died. WI for",
                    "bbox": [91, 108, 456, 316],
                    "text_pixel_bbox": [91, 108, 456, 316],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": ["partial_dark_bubble_lobe_reocr"],
                    "reocr_candidate_index": 1000,
                }
            ],
            "_vision_blocks": [{"bbox": [0, 0, 584, 457], "reocr_candidate_index": 1000}],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(page, recovered)

        self.assertEqual(merged, 1)
        self.assertEqual(page["texts"][0]["text"], "No one cared whether you lived or died.")
        self.assertLess(page["texts"][0]["text_pixel_bbox"][2], 456)
        self.assertIn(
            "false_dark_bubble_trailing_ocr_fragment_removed",
            page["texts"][0]["qa_flags"],
        )

    def test_dark_bubble_full_crop_reocr_strips_false_trailing_fragment(self):
        from strip.process_bands import Band, _replace_dark_bubble_text_with_full_crop_ocr
        import numpy as np

        class Runtime:
            def run_ocr_stage(self, _crop, _page, **_kwargs):
                return {
                    "texts": [
                        {
                            "text": "No one cared whether you lived or died. WI for",
                            "raw_ocr": "No one cared whether you lived or died. WI for",
                            "bbox": [91, 108, 456, 316],
                            "text_pixel_bbox": [91, 108, 456, 316],
                            "confidence": 0.96,
                        }
                    ]
                }

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        band = Band(y_top=0, y_bottom=480, balloons=[], strip_slice=image, original_slice=image.copy())
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "No one cared whether you lived or died.",
                    "raw_ocr": "No one cared whether you lived or died.",
                    "bbox": [120, 140, 360, 250],
                    "text_pixel_bbox": [120, 140, 360, 250],
                    "bubble_mask_bbox": [0, 0, 584, 457],
                    "balloon_bbox": [0, 0, 584, 457],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": ["dark_bubble_oval_reocr"],
                }
            ]
        }

        replaced = _replace_dark_bubble_text_with_full_crop_ocr(
            page,
            band,
            runtime=Runtime(),
            idioma_origem="en",
        )

        self.assertEqual(replaced, 1)
        self.assertEqual(page["texts"][0]["text"], "No one cared whether you lived or died.")
        self.assertLess(page["texts"][0]["text_pixel_bbox"][2], 456)
        self.assertIn(
            "false_dark_bubble_trailing_ocr_fragment_removed",
            page["texts"][0]["qa_flags"],
        )

    def test_short_dark_lobe_fragment_is_dropped_when_covered_by_full_ocr(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "the king of being a pushover...",
                    "bbox": [476, 289, 659, 362],
                    "text_pixel_bbox": [476, 289, 649, 362],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": ["dark_bubble_oval_reocr"],
                }
            ],
            "_vision_blocks": [],
        }
        recovered = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text": "ing",
                    "bbox": [614, 287, 659, 331],
                    "text_pixel_bbox": [614, 287, 659, 331],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": ["partial_dark_bubble_lobe_reocr", "adjacent_dark_bubble_reocr"],
                    "reocr_candidate_index": 1001,
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [614, 287, 659, 331],
                    "reocr_candidate_index": 1001,
                }
            ],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(page, recovered)

        self.assertEqual(merged, 0)
        self.assertEqual(len(page["texts"]), 1)
        self.assertIn("short_dark_lobe_fragment_covered_by_ocr", recovered["texts"][0]["qa_flags"])

    def test_dark_lobe_semantic_duplicate_fragment_is_dropped(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text": "Searching for a world without an underworld",
                    "bbox": [115, 164, 345, 229],
                    "text_pixel_bbox": [115, 164, 345, 229],
                    "balloon_bbox": [64, 123, 435, 328],
                    "bubble_mask_bbox": [64, 123, 435, 328],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": ["dark_bubble_oval_reocr"],
                }
            ],
            "_vision_blocks": [],
        }
        recovered = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001_002",
                    "text": "for a world underworld.",
                    "bbox": [316, 159, 352, 232],
                    "text_pixel_bbox": [231, 164, 348, 229],
                    "balloon_bbox": [64, 123, 435, 328],
                    "bubble_mask_bbox": [226, 123, 442, 328],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": ["partial_dark_bubble_lobe_reocr", "adjacent_dark_bubble_reocr"],
                    "reocr_candidate_index": 1002,
                }
            ],
            "_vision_blocks": [{"bbox": [226, 123, 442, 328], "reocr_candidate_index": 1002}],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(page, recovered)

        self.assertEqual(merged, 0)
        self.assertEqual(len(page["texts"]), 1)
        self.assertIn(
            "dark_lobe_semantic_duplicate_fragment_suppressed",
            recovered["texts"][0]["qa_flags"],
        )

    def test_false_short_art_ocr_is_dropped_before_inpaint(self):
        from strip.process_bands import _drop_suppressed_records_for_inpaint

        page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "A",
                    "translated": "UM",
                    "qa_flags": ["mask_outside_balloon_critical"],
                    "block_profile": "white_balloon",
                },
                {"id": "ocr_003", "text": "WHAT IS THIS?", "translated": "O QUE E ISSO?"},
            ],
            "_vision_blocks": [
                {"text_id": "ocr_002", "bbox": [10, 10, 40, 40]},
                {"text_id": "ocr_003", "bbox": [50, 50, 140, 90]},
            ],
        }

        _drop_suppressed_records_for_inpaint(page)

        self.assertEqual([text["id"] for text in page["texts"]], ["ocr_003"])
        self.assertEqual([block["text_id"] for block in page["_vision_blocks"]], ["ocr_003"])

    def test_unverified_dark_art_ocr_is_dropped_before_inpaint(self):
        from strip.process_bands import _drop_suppressed_records_for_inpaint

        page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text": "1/ WHAT'S THA",
                    "translated": "1/ O QUE E",
                    "bbox": [192, 176, 388, 397],
                    "text_pixel_bbox": [192, 176, 388, 397],
                    "bubble_mask_bbox": [0, 0, 32, 32],
                    "qa_flags": [
                        "dark_bubble_oval_reocr",
                        "dark_bubble_lobe_mask_bbox_preferred",
                        "dark_bubble_visual_mask_rejected_tiny_text",
                    ],
                },
                {"id": "ocr_real", "text": "WHAT'S THAT?", "translated": "O QUE E ISSO?"},
            ],
            "_vision_blocks": [
                {"text_id": "direct_paddle_reocr_001", "bbox": [192, 176, 388, 397]},
                {"text_id": "ocr_real", "bbox": [447, 1000, 630, 1025]},
            ],
        }

        _drop_suppressed_records_for_inpaint(page)

        self.assertEqual([text["id"] for text in page["texts"]], ["ocr_real"])
        self.assertEqual(page["_vision_blocks"], [{"text_id": "ocr_real", "bbox": [447, 1000, 630, 1025]}])

    def test_stage_dark_bubble_uses_compact_existing_ellipse(self):
        from strip.process_bands import _normalize_dark_bubble_contracts_for_stage
        import numpy as np

        page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "bbox": [214, 112, 349, 196],
                    "text_pixel_bbox": [214, 112, 349, 196],
                    "balloon_bbox": [0, 0, 800, 785],
                    "bubble_mask_bbox": [0, 0, 800, 785],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "bubble_mask_ellipse": {"center": [282.0, 154.5], "axes": [278.0, 283.0], "angle": 0.0},
                    "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
                }
            ]
        }

        _normalize_dark_bubble_contracts_for_stage(page, np.zeros((785, 800, 3), dtype=np.uint8))

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_bbox"], [143, 13, 421, 296])
        self.assertEqual(text["balloon_bbox"], [143, 13, 421, 296])
        self.assertIn("dark_bubble_compact_ellipse_bbox_preferred", text["qa_flags"])

    def test_stage_dark_panel_uses_rejected_reference_bbox_as_full_type_area(self):
        from strip.process_bands import _normalize_dark_bubble_contracts_for_stage
        import numpy as np

        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [113, 165, 259, 236],
                    "text_pixel_bbox": [113, 165, 259, 236],
                    "balloon_bbox": [70, 136, 302, 265],
                    "bubble_mask_bbox": [70, 136, 302, 265],
                    "qa_flags": ["connected_layout_disabled_dark_panel_visual_mask"],
                    "qa_metrics": {
                        "image_dark_panel_mask_rejected": [
                            {
                                "reason": "overbroad_against_balloon_bbox",
                                "reference_bbox": [37, 122, 349, 332],
                                "mask_bbox": [37, 122, 687, 332],
                            }
                        ]
                    },
                }
            ]
        }

        _normalize_dark_bubble_contracts_for_stage(page, np.zeros((398, 800, 3), dtype=np.uint8))

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_bbox"], [37, 122, 349, 332])
        self.assertEqual(text["balloon_bbox"], [37, 122, 349, 332])
        self.assertGreater(text["bubble_inner_bbox"][2] - text["bubble_inner_bbox"][0], 220)
        self.assertIn("dark_panel_full_bbox_selected", text["qa_flags"])

    def test_stage_dark_panel_rect_from_dark_bubble_gets_inner_bbox_before_inpaint(self):
        from strip.process_bands import _normalize_dark_bubble_contracts_for_stage
        import numpy as np

        page = {
            "_band_y_top": 2732,
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [84, 2820, 286, 3029],
                    "text_pixel_bbox": [113, 2897, 259, 2968],
                    "balloon_bbox": [37, 2854, 349, 3064],
                    "bubble_mask_bbox": [37, 2854, 349, 3064],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": [
                        "dark_bubble_visual_glyph_mask_replaced_geometry",
                        "dark_panel_rect_from_dark_bubble_bbox",
                    ],
                }
            ]
        }

        _normalize_dark_bubble_contracts_for_stage(page, np.zeros((360, 800, 3), dtype=np.uint8))

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_source"], "image_dark_panel_mask")
        self.assertEqual(text["bubble_mask_bbox"], [37, 122, 349, 332])
        self.assertEqual(text["balloon_bbox"], [37, 122, 349, 332])
        self.assertEqual(text["_band_y_top"], 2732)
        self.assertIn("bubble_inner_bbox", text)
        self.assertIn("dark_panel_full_bbox_selected", text["qa_flags"])

    def test_stage_dark_panel_prefers_existing_balloon_bbox_over_text_center_crop(self):
        from strip.process_bands import _normalize_dark_bubble_contracts_for_stage
        import numpy as np

        page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "bbox": [405, 109, 683, 217],
                    "text_pixel_bbox": [405, 109, 683, 217],
                    "balloon_bbox": [344, 77, 744, 249],
                    "bubble_mask_bbox": [397, 99, 691, 227],
                    "layout_profile": "dark_panel",
                    "qa_flags": ["candidate_crop_direct_paddle_reocr"],
                }
            ]
        }

        _normalize_dark_bubble_contracts_for_stage(page, np.zeros((398, 800, 3), dtype=np.uint8))

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_bbox"], [344, 77, 744, 249])
        self.assertEqual(text["balloon_bbox"], [344, 77, 744, 249])
        self.assertGreaterEqual(text["bubble_inner_bbox"][2] - text["bubble_inner_bbox"][0], 280)
        self.assertIn("dark_panel_full_bbox_selected", text["qa_flags"])

    def test_dark_lobe_semantic_duplicate_is_dropped_before_inpaint(self):
        from strip.process_bands import _drop_suppressed_records_for_inpaint

        page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text": "Searching for a world without an underworld",
                    "bbox": [115, 164, 345, 229],
                    "text_pixel_bbox": [115, 164, 345, 229],
                    "balloon_bbox": [64, 123, 435, 328],
                    "bubble_mask_bbox": [64, 123, 435, 328],
                    "bubble_mask_source": "image_dark_panel_mask",
                    "qa_flags": ["candidate_crop_direct_paddle_reocr"],
                },
                {
                    "id": "direct_paddle_reocr_001_002",
                    "text": "for a world underworld.",
                    "bbox": [316, 159, 352, 232],
                    "text_pixel_bbox": [115, 164, 345, 229],
                    "balloon_bbox": [64, 123, 435, 328],
                    "bubble_mask_bbox": [107, 132, 353, 261],
                    "bubble_mask_source": "image_dark_panel_mask",
                    "qa_flags": [
                        "candidate_crop_direct_paddle_reocr",
                        "partial_dark_bubble_lobe_reocr",
                        "adjacent_dark_bubble_reocr",
                    ],
                },
            ],
            "_vision_blocks": [
                {"text_id": "direct_paddle_reocr_001", "bbox": [115, 164, 345, 229]},
                {"text_id": "direct_paddle_reocr_001_002", "bbox": [316, 159, 352, 232]},
            ],
        }

        _drop_suppressed_records_for_inpaint(page)

        self.assertEqual([text["id"] for text in page["texts"]], ["direct_paddle_reocr_001"])
        self.assertEqual([block["text_id"] for block in page["_vision_blocks"]], ["direct_paddle_reocr_001"])

    def test_candidate_crop_reocr_evidence_accepts_single_light_component_on_dark_bubble(self):
        from strip.process_bands import _candidate_crop_reocr_evidence_is_strong

        evidence = {
            "has_inner_light_text": False,
            "inner_light_component_count": 1,
            "inner_light_area": 264,
            "bright_pixel_ratio": 0.1386,
            "dark_pixel_ratio": 0.7869,
            "has_inner_dark_text": True,
            "inner_dark_component_count": 2,
            "inner_dark_area": 96,
            "significant_component_count": 0,
            "significant_area": 0,
        }

        self.assertTrue(_candidate_crop_reocr_evidence_is_strong(evidence))

    def test_sparse_dark_candidate_reocr_allows_high_confidence_without_light_area(self):
        from strip.process_bands import _candidate_crop_reocr_allows_sparse_dark_text

        evidence = {
            "has_inner_dark_text": True,
            "inner_dark_area": 64,
            "inner_dark_component_count": 3,
            "inner_light_area": 0,
            "inner_light_component_count": 0,
            "bright_pixel_ratio": 0.1657,
            "dark_pixel_ratio": 0.7485,
        }

        self.assertTrue(_candidate_crop_reocr_allows_sparse_dark_text(evidence, confidence=0.8081))
        self.assertFalse(_candidate_crop_reocr_allows_sparse_dark_text(evidence, confidence=0.60))

    def test_band_to_page_dict_remaps_balloon_coords_to_local(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import numpy as np

        slice_img = np.zeros((100, 300, 3), dtype=np.uint8)
        band = Band(
            y_top=500,
            y_bottom=600,
            balloons=[
                Balloon(strip_bbox=BBox(50, 510, 150, 590), confidence=0.9),
            ],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0)

        self.assertEqual(page_dict["width"], 300)
        self.assertEqual(page_dict["height"], 100)
        self.assertEqual(page_dict["numero"], 1)
        self.assertEqual(page_dict["_band_index"], 1)
        block = page_dict["_vision_blocks"][0]
        self.assertEqual(block["bbox"], [50, 10, 150, 90])

    def test_band_to_page_dict_can_use_source_page_number_for_ocr_profile(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import numpy as np

        slice_img = np.zeros((120, 300, 3), dtype=np.uint8)
        band = Band(
            y_top=900,
            y_bottom=1020,
            balloons=[Balloon(strip_bbox=BBox(40, 930, 180, 990), confidence=0.86)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=12, source_page_number=2)

        self.assertEqual(page_dict["numero"], 2)
        self.assertEqual(page_dict["_source_page_number"], 2)
        self.assertEqual(page_dict["_band_index"], 13)

    def test_band_to_page_dict_attaches_real_bubble_mask_for_white_balloon(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((120, 220, 3), 32, dtype=np.uint8)
        cv2.ellipse(slice_img, (110, 60), (72, 34), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(slice_img, (110, 60), (72, 34), 0, 0, 360, (0, 0, 0), 2)
        band = Band(
            y_top=300,
            y_bottom=420,
            balloons=[Balloon(strip_bbox=BBox(30, 315, 190, 395), confidence=0.91)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0, source_page_number=1)

        block = page_dict["_vision_blocks"][0]
        region = page_dict["_bubble_regions"][0]
        self.assertEqual(block["bubble_mask_bbox"], [30, 15, 190, 95])
        self.assertIn("bubble_mask", block)
        self.assertIn("bubble_mask", region)
        self.assertEqual(block["bubble_mask"].shape, (80, 160))
        self.assertEqual(region["bubble_mask"].shape, (80, 160))
        self.assertGreater(int(block["bubble_mask"][40, 80]), 0)
        self.assertEqual(int(block["bubble_mask"][0, 0]), 0)
        self.assertEqual(block["bubble_mask_source"], "image_white_bubble_mask")
        self.assertEqual(region["bubble_mask_source"], "image_white_bubble_mask")

    def test_attach_real_bubble_mask_does_not_promote_unsourced_existing_mask(self):
        from strip.process_bands import _attach_real_bubble_mask_to_block
        import numpy as np

        block = {
            "bbox": [20, 20, 80, 60],
            "bubble_mask": np.ones((40, 60), dtype=np.uint8) * 255,
            "bubble_mask_bbox": [20, 20, 80, 60],
        }

        _attach_real_bubble_mask_to_block(block, np.full((100, 120, 3), 255, dtype=np.uint8))

        self.assertNotEqual(block.get("bubble_mask_source"), "real")
        self.assertNotEqual(block.get("bubble_mask_source"), "real_bubble_mask")
        self.assertEqual(block.get("bubble_mask_source"), "bbox_fallback")
        self.assertEqual(block.get("bubble_mask_error"), "missing_real_bubble_mask_source")

    def test_band_to_page_dict_rejects_suspicious_rectangular_white_crop(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import numpy as np

        slice_img = np.full((120, 220, 3), 24, dtype=np.uint8)
        slice_img[20:82, 40:180] = 255
        band = Band(
            y_top=300,
            y_bottom=420,
            balloons=[Balloon(strip_bbox=BBox(40, 320, 180, 382), confidence=0.91)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0, source_page_number=1)

        block = page_dict["_vision_blocks"][0]
        region = page_dict["_bubble_regions"][0]
        self.assertNotIn("bubble_mask", block)
        self.assertNotIn("bubble_mask", region)
        self.assertEqual(block["bubble_mask_source"], "derived_white_crop_rejected")
        self.assertEqual(block["bubble_mask_error"], "rejected_rectangular_crop")

    def test_band_to_page_dict_expands_tight_oval_crop_to_recover_curve_shape(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((220, 260, 3), 24, dtype=np.uint8)
        cv2.ellipse(slice_img, (130, 110), (35, 35), 0, 0, 360, (255, 255, 255), -1)
        band = Band(
            y_top=0,
            y_bottom=220,
            balloons=[Balloon(strip_bbox=BBox(120, 90, 140, 130), confidence=0.91)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0, source_page_number=1)

        block = page_dict["_vision_blocks"][0]
        region = page_dict["_bubble_regions"][0]
        self.assertNotEqual(block["bubble_mask_bbox"], [120, 90, 140, 130])
        self.assertEqual(region["bubble_mask_bbox"], block["bubble_mask_bbox"])
        self.assertLessEqual(block["bubble_mask_bbox"][0], 104)
        self.assertLessEqual(block["bubble_mask_bbox"][1], 74)
        self.assertGreaterEqual(block["bubble_mask_bbox"][2], 156)
        self.assertGreaterEqual(block["bubble_mask_bbox"][3], 146)
        self.assertIn("bubble_mask", block)
        self.assertIn("bubble_mask", region)
        self.assertEqual(block["bubble_mask_source"], "image_white_bubble_mask")
        density = float(np.count_nonzero(block["bubble_mask"])) / float(block["bubble_mask"].size)
        self.assertLess(density, 0.92)

    def test_ensure_text_balloon_bboxes_keeps_expanded_bubble_mask_bbox(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((220, 260, 3), 24, dtype=np.uint8)
        cv2.ellipse(slice_img, (130, 110), (35, 35), 0, 0, 360, (255, 255, 255), -1)
        band = Band(
            y_top=0,
            y_bottom=220,
            balloons=[Balloon(strip_bbox=BBox(120, 90, 140, 130), confidence=0.91)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )
        page = {
            "numero": 1,
            "width": 260,
            "height": 220,
            "_band_id": "page_001_band_001",
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [124, 104, 136, 116],
                    "text_pixel_bbox": [124, 104, 136, 116],
                }
            ],
        }

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertNotEqual(text["bubble_mask_bbox"], [120, 90, 140, 130])
        self.assertLessEqual(text["bubble_mask_bbox"][0], 104)
        self.assertLessEqual(text["bubble_mask_bbox"][1], 74)
        self.assertGreaterEqual(text["bubble_mask_bbox"][2], 156)
        self.assertGreaterEqual(text["bubble_mask_bbox"][3], 146)
        self.assertEqual(text["bubble_mask_source"], "image_white_bubble_mask")
        self.assertNotIn("bubble_mask_error", text)

    def test_ensure_text_balloon_bboxes_clears_stale_rejected_mask_error(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((180, 260, 3), 24, dtype=np.uint8)
        cv2.ellipse(slice_img, (130, 96), (68, 38), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(slice_img, (130, 96), (68, 38), 0, 0, 360, (0, 0, 0), 2)
        band = Band(
            y_top=0,
            y_bottom=180,
            balloons=[Balloon(strip_bbox=BBox(62, 58, 198, 134), confidence=0.91)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )
        page = {
            "numero": 1,
            "width": 260,
            "height": 180,
            "_band_id": "page_001_band_001",
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [94, 84, 166, 106],
                    "text_pixel_bbox": [94, 84, 166, 106],
                    "bubble_mask_source": "derived_rectangular_balloon",
                    "bubble_mask_error": "rejected_rectangular_crop",
                }
            ],
        }

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertIn("bubble_mask", text)
        self.assertIn(text["bubble_mask_source"], {"image_white_bubble_mask", "image_contour_bubble_mask"})
        self.assertNotIn("bubble_mask_error", text)

    def test_band_to_page_dict_retries_when_tight_oval_is_misread_as_rectangular(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((180, 260, 3), 18, dtype=np.uint8)
        cv2.ellipse(slice_img, (155, 95), (64, 42), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(slice_img, (155, 95), (64, 42), 0, 0, 360, (0, 0, 0), 2)
        band = Band(
            y_top=400,
            y_bottom=580,
            balloons=[Balloon(strip_bbox=BBox(122, 470, 205, 530), confidence=0.91)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0, source_page_number=1)

        block = page_dict["_vision_blocks"][0]
        self.assertIn("bubble_mask", block)
        self.assertNotEqual(block["bubble_mask_bbox"], [122, 70, 205, 130])
        self.assertEqual(block["bubble_mask_source"], "image_white_bubble_mask")
        density = float(np.count_nonzero(block["bubble_mask"])) / float(block["bubble_mask"].size)
        self.assertLess(density, 0.92)

    def test_band_to_page_dict_retries_borderline_tight_oval_before_accepting_rectangular_source(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((180, 260, 3), 24, dtype=np.uint8)
        cv2.ellipse(slice_img, (130, 90), (70, 36), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(slice_img, (130, 90), (70, 36), 0, 0, 360, (0, 0, 0), 2)
        band = Band(
            y_top=0,
            y_bottom=180,
            balloons=[Balloon(strip_bbox=BBox(70, 64, 190, 116), confidence=0.91)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0, source_page_number=1)

        block = page_dict["_vision_blocks"][0]
        self.assertEqual(block["bubble_mask_source"], "image_white_bubble_mask")
        self.assertNotEqual(block["bubble_mask_bbox"], [70, 64, 190, 116])
        density = float(np.count_nonzero(block["bubble_mask"])) / float(block["bubble_mask"].size)
        self.assertLess(density, 0.92)

    def test_derived_bubble_mask_refiner_removes_connected_block_from_oval(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import cv2
        import numpy as np

        image = np.full((180, 260, 3), 24, dtype=np.uint8)
        cv2.ellipse(image, (140, 98), (58, 34), 0, 0, 360, (255, 255, 255), -1)
        image[48:78, 84:132] = 255

        mask, source, error, used_bbox = _derive_real_bubble_mask_from_crop(image, [70, 40, 210, 150])

        self.assertIsNotNone(mask)
        self.assertEqual(source, "derived_white_crop")
        self.assertIsNone(error)
        self.assertEqual(used_bbox, [70, 40, 210, 150])
        self.assertEqual(int(mask[18, 20]), 0)
        self.assertGreater(int(mask[58, 70]), 0)

    def test_derived_bubble_mask_refiner_preserves_pointed_extensions(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import cv2
        import numpy as np

        image = np.full((160, 280, 3), 24, dtype=np.uint8)
        cv2.ellipse(image, (132, 62), (72, 30), 0, 0, 360, (255, 255, 255), -1)
        for poly in (
            np.array([[55, 58], [18, 42], [61, 70]], dtype=np.int32),
            np.array([[206, 56], [252, 42], [212, 70]], dtype=np.int32),
            np.array([[120, 88], [136, 132], [146, 86]], dtype=np.int32),
        ):
            cv2.fillPoly(image, [poly], (255, 255, 255))

        mask, source, error, _used_bbox = _derive_real_bubble_mask_from_crop(image, [0, 20, 270, 145])

        self.assertIsNotNone(mask)
        self.assertEqual(source, "derived_white_crop")
        self.assertIsNone(error)
        self.assertGreater(int(mask[30, 30]), 0)
        self.assertGreater(int(mask[112, 136]), 0)

    def test_derived_bubble_mask_retries_when_accepted_crop_touches_bottom_edge(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import cv2
        import numpy as np

        image = np.full((180, 260, 3), 24, dtype=np.uint8)
        cv2.ellipse(image, (130, 96), (72, 42), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (130, 96), (72, 42), 0, 0, 360, (0, 0, 0), 2)

        mask, source, error, used_bbox = _derive_real_bubble_mask_from_crop(image, [50, 60, 210, 100])

        self.assertIsNotNone(mask)
        self.assertEqual(source, "derived_white_crop")
        self.assertIsNone(error)
        self.assertNotEqual(used_bbox, [50, 60, 210, 100])
        self.assertLessEqual(used_bbox[1], 44)
        self.assertGreaterEqual(used_bbox[3], 116)
        self.assertGreater(mask.shape[0], 40)

    def test_derive_real_bubble_mask_rejects_white_continuum_gutter_overreach(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import cv2
        import numpy as np

        image = np.full((180, 260, 3), 220, dtype=np.uint8)
        cv2.rectangle(image, (20, 15), (240, 165), (255, 255, 255), -1)
        cv2.ellipse(image, (130, 92), (42, 28), 0, 0, 360, (0, 0, 0), 3)
        cv2.ellipse(image, (130, 92), (42, 28), 0, 0, 360, (255, 255, 255), -1)
        for y in range(45, 145, 16):
            cv2.line(image, (20, y), (240, y), (90, 90, 90), 4)
        for x in range(30, 230, 16):
            cv2.line(image, (x, 15), (x, 165), (90, 90, 90), 4)

        mask, source, error, used_bbox = _derive_real_bubble_mask_from_crop(image, [80, 40, 180, 145])

        self.assertIsNotNone(mask)
        self.assertEqual(source, "derived_white_crop")
        self.assertIsNone(error)
        self.assertEqual(used_bbox, [80, 40, 180, 145])
        self.assertLess(int(np.count_nonzero(mask > 0)), int(mask.size * 0.75))
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[0, -1]), 0)
        self.assertEqual(int(mask[-1, 0]), 0)
        self.assertEqual(int(mask[-1, -1]), 0)

    def test_derive_real_bubble_mask_uses_outline_when_white_panel_contains_balloon(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import cv2
        import numpy as np

        image = np.full((420, 760, 3), 255, dtype=np.uint8)
        image[180:420, :] = 18
        cv2.rectangle(image, (42, 52), (718, 372), (255, 255, 255), -1)
        cv2.rectangle(image, (42, 52), (718, 372), (12, 12, 12), 2)
        cv2.ellipse(image, (380, 180), (145, 52), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (380, 180), (145, 52), 0, 0, 360, (0, 0, 0), 3)
        cv2.rectangle(image, (330, 160), (430, 184), (90, 90, 90), -1)

        mask, source, error, used_bbox = _derive_real_bubble_mask_from_crop(
            image,
            [42, 52, 718, 372],
            support_bbox=[330, 160, 430, 184],
        )

        self.assertIsNotNone(mask)
        self.assertEqual(source, "outline_seeded_contour")
        self.assertIsNone(error)
        self.assertLess(int(np.count_nonzero(mask > 0)), int(mask.size * 0.40))
        self.assertEqual(int(mask[5, 5]), 0)
        self.assertEqual(int(mask[-5, -5]), 0)
        self.assertGreater(int(mask[128, 338]), 0)
        self.assertEqual(used_bbox, [42, 52, 718, 372])

    def test_derive_real_bubble_mask_uses_seeded_outline_when_white_page_has_neighbor_text(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import cv2
        import numpy as np

        image = np.full((220, 360, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (150, 92), (92, 60), 0, 0, 360, (0, 0, 0), 2)
        image[72:80, 92:208] = 16
        image[86:94, 88:212] = 16
        image[100:108, 118:184] = 16
        image[86:94, 302:338] = 16

        mask, source, error, used_bbox = _derive_real_bubble_mask_from_crop(
            image,
            [0, 0, 350, 178],
            support_bbox=[88, 70, 212, 110],
        )

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(source, "outline_seeded_contour")
        self.assertIsNone(error)
        self.assertEqual(used_bbox, [0, 0, 350, 178])
        self.assertGreater(int(mask[92, 150]), 0)
        self.assertEqual(int(mask[90, 320]), 0)
        ys, xs = np.where(mask > 0)
        self.assertGreater(int(xs.min()), 35)
        self.assertLess(int(xs.max()) + 1, 255)
        self.assertLess(int(ys.max()) + 1, 160)

    def test_derive_real_bubble_mask_follows_closed_starburst_outline_on_white_page(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import cv2
        import numpy as np

        image = np.full((230, 320, 3), 255, dtype=np.uint8)
        starburst = np.asarray(
            [
                [46, 102],
                [82, 86],
                [80, 52],
                [122, 72],
                [164, 50],
                [176, 84],
                [244, 72],
                [206, 112],
                [246, 148],
                [176, 142],
                [150, 184],
                [124, 144],
                [66, 164],
                [88, 124],
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [starburst], True, (0, 0, 0), 3)
        image[98:112, 116:206] = 20
        image[121:135, 106:215] = 20

        mask, source, error, used_bbox = _derive_real_bubble_mask_from_crop(
            image,
            [0, 0, 320, 220],
            support_bbox=[106, 98, 215, 135],
        )

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(source, "outline_seeded_contour")
        self.assertIsNone(error)
        self.assertEqual(used_bbox, [0, 0, 320, 220])
        self.assertGreater(int(mask[80, 230]), 0)
        self.assertGreater(int(mask[170, 150]), 0)
        self.assertGreater(int(mask[112, 200]), 0)
        self.assertEqual(int(mask[112, 246]), 0)
        self.assertEqual(int(mask[185, 230]), 0)
        ys, xs = np.where(mask > 0)
        self.assertGreater(int(xs.max()) + 1, 235)
        self.assertGreater(int(ys.max()) + 1, 175)
        self.assertLess(int(np.count_nonzero(mask > 0)), int(mask.size * 0.34))

    def test_derive_real_bubble_mask_does_not_paint_support_bbox_corners_into_oval(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import cv2
        import numpy as np

        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (130, 90), (112, 62), 0, 0, 360, (0, 0, 0), 3)
        image[70:84, 54:206] = 30
        image[96:110, 48:212] = 30

        mask, source, error, _used_bbox = _derive_real_bubble_mask_from_crop(
            image,
            [0, 0, 260, 180],
            support_bbox=[20, 45, 240, 135],
        )

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(source, "outline_seeded_contour")
        self.assertIsNone(error)
        self.assertGreater(int(mask[90, 130]), 0)
        self.assertGreater(int(mask[90, 40]), 0)
        self.assertEqual(int(mask[45, 20]), 0)
        self.assertEqual(int(mask[134, 239]), 0)

    def test_derive_real_bubble_mask_rejects_overbroad_region_not_around_text(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import numpy as np

        image = np.full((520, 520, 3), 24, dtype=np.uint8)
        image[35:485, 35:485] = 255
        image[428:442, 420:468] = 20

        mask, source, error, used_bbox = _derive_real_bubble_mask_from_crop(
            image,
            [0, 0, 520, 520],
            support_bbox=[420, 428, 468, 442],
        )

        self.assertIsNone(mask)
        self.assertIsNone(source)
        self.assertEqual(error, "derived_mask_not_anchored_to_text")
        self.assertIsNone(used_bbox)

    def test_prune_mask_to_support_components_drops_unrelated_white_components(self):
        from strip.process_bands import _prune_mask_to_support_components
        import numpy as np

        mask = np.zeros((120, 180), dtype=np.uint8)
        mask[42:66, 70:118] = 255
        mask[8:40, 8:52] = 255
        mask[72:112, 132:174] = 255

        pruned, pruned_bbox, error = _prune_mask_to_support_components(
            mask,
            [0, 0, 180, 120],
            [76, 46, 112, 62],
        )

        self.assertIsNone(error)
        self.assertIsNotNone(pruned)
        self.assertEqual(pruned_bbox, [70, 42, 118, 66])
        self.assertEqual(pruned.shape, (24, 48))
        self.assertGreater(int(pruned[8, 8]), 0)
        self.assertLess(int(np.count_nonzero(pruned)), int(np.count_nonzero(mask)))

    def test_derive_real_bubble_mask_reconstructs_clipped_starburst_from_outline_points(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import cv2
        import numpy as np

        image = np.full((230, 320, 3), 255, dtype=np.uint8)
        starburst = np.asarray(
            [
                [46, 102],
                [82, 86],
                [80, 52],
                [122, 72],
                [164, 50],
                [176, 84],
                [244, 72],
                [206, 112],
                [246, 148],
                [176, 142],
                [150, 184],
                [124, 144],
                [66, 164],
                [88, 124],
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [starburst], True, (0, 0, 0), 3)
        cv2.rectangle(image, (132, 42), (168, 76), (255, 255, 255), -1)
        image[98:112, 116:206] = 20
        image[121:135, 106:215] = 20

        mask, source, error, _used_bbox = _derive_real_bubble_mask_from_crop(
            image,
            [0, 0, 320, 220],
            support_bbox=[106, 98, 215, 135],
        )

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(source, "outline_seeded_contour")
        self.assertIsNone(error)
        self.assertGreater(int(mask[80, 230]), 0)
        self.assertGreater(int(mask[160, 70]), 0)
        self.assertGreater(int(mask[170, 150]), 0)
        self.assertEqual(int(mask[112, 270]), 0)
        ys, xs = np.where(mask > 0)
        self.assertGreater(int(xs.max()) + 1, 235)
        self.assertGreater(int(ys.max()) + 1, 175)
        self.assertLess(int(np.count_nonzero(mask > 0)), int(mask.size * 0.42))

    def test_derive_real_bubble_mask_allows_band_edge_as_boundary_for_clipped_balloon(self):
        from strip.process_bands import _derive_real_bubble_mask_from_crop
        import cv2
        import numpy as np

        image = np.full((180, 320, 3), 255, dtype=np.uint8)
        starburst = np.asarray(
            [
                [54, 52],
                [90, 38],
                [112, 0],
                [142, 28],
                [182, 0],
                [198, 40],
                [276, 36],
                [232, 78],
                [278, 120],
                [206, 116],
                [176, 170],
                [144, 120],
                [72, 144],
                [96, 86],
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [starburst], True, (0, 0, 0), 3)
        image[68:82, 126:228] = 20
        image[94:108, 118:218] = 20

        mask, source, error, used_bbox = _derive_real_bubble_mask_from_crop(
            image,
            [40, 20, 300, 160],
            support_bbox=[118, 68, 228, 108],
        )

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(source, "outline_seeded_contour")
        self.assertIsNone(error)
        self.assertLess(used_bbox[0], 40)
        self.assertEqual(used_bbox[1], 0)
        self.assertEqual(used_bbox[2], 320)
        self.assertEqual(used_bbox[3], 180)
        ys, xs = np.where(mask > 0)
        self.assertLessEqual(int(ys.min()), 6)
        self.assertGreater(int(xs.max()) + 1, 250)
        self.assertGreater(int(ys.max()) + 1, 160)
        self.assertLess(int(np.count_nonzero(mask > 0)), int(mask.size * 0.55))

    def test_ensure_text_balloon_bboxes_rederives_fallback_mask_with_text_support(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

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
        band = Band(
            y_top=0,
            y_bottom=190,
            balloons=[Balloon(strip_bbox=BBox(40, 12, 236, 142), confidence=0.91)],
            strip_slice=image,
            original_slice=image.copy(),
        )
        page = {
            "numero": 1,
            "width": 280,
            "height": 190,
            "_band_id": "page_001_band_001",
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [118, 70, 170, 82],
                    "text_pixel_bbox": [118, 70, 170, 82],
                    "balloon_bbox": [40, 12, 236, 142],
                    "bubble_mask_bbox": [40, 12, 236, 142],
                    "bubble_mask_source": "bbox_fallback",
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [40, 12, 236, 142],
                    "bubble_mask_bbox": [40, 12, 236, 142],
                    "bubble_mask_source": "bbox_fallback",
                }
            ],
        }

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertIn("bubble_mask", text)
        self.assertEqual(text["bubble_mask_source"], "image_contour_bubble_mask")
        self.assertNotIn("bubble_mask_error", text)
        self.assertGreater(text["bubble_mask_bbox"][0], 55)
        self.assertLess(text["bubble_mask_bbox"][2], 218)
        self.assertLess(text["bubble_mask_bbox"][3], 135)
        self.assertEqual(int(text["bubble_mask"][0, 0]), 0)

    def test_ensure_text_balloon_bboxes_shrinks_overlarge_derived_region_to_visual_balloon(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        image = np.full((420, 760, 3), 255, dtype=np.uint8)
        image[180:420, :] = 18
        cv2.rectangle(image, (42, 52), (718, 372), (255, 255, 255), -1)
        cv2.rectangle(image, (42, 52), (718, 372), (12, 12, 12), 2)
        cv2.ellipse(image, (380, 180), (145, 52), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (380, 180), (145, 52), 0, 0, 360, (0, 0, 0), 3)
        band = Band(
            y_top=0,
            y_bottom=420,
            balloons=[Balloon(strip_bbox=BBox(42, 52, 718, 372), confidence=0.91)],
            strip_slice=image,
            original_slice=image.copy(),
        )
        page = {
            "numero": 1,
            "width": 760,
            "height": 420,
            "_band_id": "page_001_band_001",
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [330, 160, 430, 184],
                    "text_pixel_bbox": [330, 160, 430, 184],
                    "balloon_bbox": [235, 128, 525, 235],
                    "bubble_mask_bbox": [42, 52, 718, 372],
                    "bubble_mask_source": "derived_white_crop",
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [42, 52, 718, 372],
                    "bubble_mask_bbox": [42, 52, 718, 372],
                    "bubble_mask_source": "derived_white_crop",
                }
            ],
        }

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertLess(text["bubble_mask_bbox"][0], 260)
        self.assertGreater(text["bubble_mask_bbox"][2], 500)
        self.assertLess(text["bubble_mask_bbox"][2] - text["bubble_mask_bbox"][0], 360)
        self.assertLess(text["bubble_mask_bbox"][3] - text["bubble_mask_bbox"][1], 140)
        self.assertIn("bubble_mask", text)
        self.assertEqual(text["bubble_mask_source"], "image_contour_bubble_mask")

    def test_ensure_text_balloon_bboxes_rejects_unanchored_derived_contract(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import numpy as np

        image = np.full((900, 800, 3), 255, dtype=np.uint8)
        page = {
            "numero": 1,
            "width": 800,
            "height": 900,
            "_band_id": "page_001_band_001",
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [555, 783, 661, 795],
                    "text_pixel_bbox": [555, 783, 661, 795],
                    "balloon_bbox": [12, 210, 656, 865],
                    "bubble_mask_bbox": [12, 210, 656, 865],
                    "bubble_mask": np.ones((655, 644), dtype=np.uint8) * 255,
                    "bubble_mask_source": "derived_white_crop",
                }
            ],
            "_vision_blocks": [],
        }
        band = Band(y_top=0, y_bottom=900, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertNotIn("bubble_mask", text)
        self.assertEqual(text["bubble_mask_source"], "derived_white_crop_rejected")
        self.assertIn(text["bubble_mask_error"], {"derived_mask_not_anchored_to_text", "rejected_rectangular_crop"})

    def test_ensure_text_balloon_bboxes_rejects_large_outline_without_line_geometry(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import numpy as np

        image = np.full((900, 800, 3), 255, dtype=np.uint8)
        page = {
            "numero": 1,
            "width": 800,
            "height": 900,
            "_band_id": "page_001_band_001",
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [137, 239, 726, 705],
                    "text_pixel_bbox": [137, 239, 726, 705],
                    "layout_bbox": [137, 239, 726, 705],
                    "line_polygons": [],
                    "balloon_bbox": [0, 0, 800, 900],
                    "bubble_mask_bbox": [113, 212, 750, 729],
                    "bubble_mask": np.ones((517, 637), dtype=np.uint8) * 255,
                    "bubble_mask_source": "outline_seeded_contour",
                    "qa_flags": [
                        "raw_text_evidence_missing",
                        "fast_fill_no_glyph_evidence",
                    ],
                }
            ],
            "_vision_blocks": [],
        }
        band = Band(y_top=0, y_bottom=900, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertNotIn("bubble_mask", text)
        self.assertEqual(text["bubble_mask_source"], "derived_white_crop_rejected")
        self.assertEqual(text["bubble_mask_error"], "derived_mask_not_anchored_to_text")

    def test_ensure_text_balloon_bboxes_prunes_disconnected_derived_mask_components(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import cv2
        import numpy as np

        image = np.full((260, 360, 3), 255, dtype=np.uint8)
        mask = np.zeros((220, 320), dtype=np.uint8)
        cv2.ellipse(mask, (90, 82), (62, 38), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, (250, 165), (48, 28), 0, 0, 360, 255, -1)
        page = {
            "numero": 1,
            "width": 360,
            "height": 260,
            "_band_id": "page_001_band_001",
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [70, 68, 122, 94],
                    "text_pixel_bbox": [70, 68, 122, 94],
                    "line_polygons": [[[70, 68], [122, 68], [122, 94], [70, 94]]],
                    "balloon_bbox": [10, 10, 330, 230],
                    "bubble_mask_bbox": [10, 10, 330, 230],
                    "bubble_mask": mask,
                    "bubble_mask_source": "derived_white_crop",
                }
            ],
            "_vision_blocks": [],
        }
        band = Band(y_top=0, y_bottom=260, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertIn("bubble_mask", text)
        self.assertEqual(text["bubble_mask_source"], "derived_white_crop")
        self.assertLess(text["bubble_mask_bbox"][2], 180)
        self.assertLess(text["bubble_mask_bbox"][3], 140)
        self.assertLess(int(np.count_nonzero(text["bubble_mask"] > 0)), 9000)
        self.assertGreater(int(np.count_nonzero(text["bubble_mask"] > 0)), 3000)

    def test_ensure_text_balloon_bboxes_rederives_rectangular_image_white_mask_to_oval_shape(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import cv2
        import numpy as np

        image = np.full((342, 800, 3), 24, dtype=np.uint8)
        cv2.ellipse(image, (226, 170), (160, 95), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (226, 170), (160, 95), 0, 0, 360, (0, 0, 0), 2)
        page = {
            "numero": 1,
            "width": 800,
            "height": 342,
            "_band_id": "page_002_band_009",
            "_vision_blocks": [
                {
                    "bbox": [47, 48, 405, 294],
                    "bubble_mask_bbox": [47, 48, 405, 294],
                    "bubble_mask": np.ones((246, 358), dtype=np.uint8) * 255,
                    "bubble_mask_source": "image_white_bubble_mask",
                }
            ],
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [150, 130, 300, 190],
                    "text_pixel_bbox": [150, 130, 300, 190],
                }
            ],
        }
        band = Band(y_top=0, y_bottom=342, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertIn("bubble_mask", text)
        self.assertEqual(text["bubble_mask_source"], "image_white_bubble_mask")
        self.assertNotIn("bubble_mask_error", text)
        self.assertNotEqual(text["bubble_mask_bbox"], [47, 48, 405, 294])
        density = float(np.count_nonzero(text["bubble_mask"] > 0)) / float(text["bubble_mask"].size)
        self.assertLess(density, 0.92)
        self.assertEqual(int(text["bubble_mask"][0, 0]), 0)

    def test_ensure_text_balloon_bboxes_rederives_contaminated_connected_image_contour_mask(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import cv2
        import numpy as np

        image = np.full((954, 800, 3), 24, dtype=np.uint8)
        cv2.ellipse(image, (230, 245), (96, 72), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (230, 245), (96, 72), 0, 0, 360, (0, 0, 0), 3)
        cv2.ellipse(image, (455, 690), (150, 95), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (455, 690), (150, 95), 0, 0, 360, (0, 0, 0), 3)
        contaminated = np.zeros((503, 402), dtype=np.uint8)
        cv2.ellipse(contaminated, (96, 72), (96, 72), 0, 0, 360, 255, -1)
        bridge = np.asarray([[156, 62], [402, 280], [382, 344], [118, 503], [42, 120]], dtype=np.int32)
        cv2.fillPoly(contaminated, [bridge], 255)
        page = {
            "numero": 3,
            "width": 800,
            "height": 954,
            "_band_id": "page_003_band_035",
            "_vision_blocks": [
                {
                    "bbox": [133, 172, 535, 675],
                    "bubble_mask_bbox": [133, 172, 535, 675],
                    "bubble_mask": contaminated,
                    "bubble_mask_source": "image_contour_bubble_mask",
                }
            ],
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [165, 220, 315, 268],
                    "text_pixel_bbox": [165, 220, 315, 268],
                    "line_polygons": [[[165, 220], [315, 220], [315, 268], [165, 268]]],
                }
            ],
        }
        band = Band(y_top=0, y_bottom=954, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertIn("bubble_mask", text)
        self.assertIn(text["bubble_mask_source"], {"image_white_bubble_mask", "image_contour_bubble_mask"})
        self.assertNotIn("bubble_mask_error", text)
        self.assertLess(text["bubble_mask_bbox"][2], 340)
        self.assertLess(text["bubble_mask_bbox"][3], 340)
        self.assertLess(int(np.count_nonzero(text["bubble_mask"] > 0)), 35000)
        self.assertEqual(int(text["bubble_mask"][-1, -1]), 0)

    def test_ensure_text_balloon_bboxes_uses_tight_text_anchor_when_text_bbox_is_art_polluted(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import cv2
        import numpy as np

        image = np.full((954, 800, 3), 32, dtype=np.uint8)
        cv2.ellipse(image, (230, 245), (96, 72), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (230, 245), (96, 72), 0, 0, 360, (0, 0, 0), 3)
        for x in (166, 190, 218, 246, 276):
            cv2.rectangle(image, (x, 235), (x + 18, 256), (12, 12, 12), -1)
        cv2.line(image, (330, 370), (535, 675), (245, 245, 245), 38)
        cv2.line(image, (330, 370), (535, 675), (0, 0, 0), 4)
        cv2.ellipse(image, (455, 690), (150, 95), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (455, 690), (150, 95), 0, 0, 360, (0, 0, 0), 3)
        contaminated = np.zeros((503, 402), dtype=np.uint8)
        cv2.ellipse(contaminated, (96, 72), (96, 72), 0, 0, 360, 255, -1)
        bridge = np.asarray([[156, 62], [402, 280], [382, 344], [118, 503], [42, 120]], dtype=np.int32)
        cv2.fillPoly(contaminated, [bridge], 255)
        page = {
            "numero": 3,
            "width": 800,
            "height": 954,
            "_band_id": "page_003_band_035",
            "_vision_blocks": [
                {
                    "bbox": [133, 172, 535, 675],
                    "bubble_mask_bbox": [133, 172, 535, 675],
                    "bubble_mask": contaminated,
                    "bubble_mask_source": "image_contour_bubble_mask",
                }
            ],
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [29, 96, 642, 709],
                    "text_pixel_bbox": [148, 235, 537, 673],
                    "balloon_bbox": [29, 96, 642, 709],
                }
            ],
        }
        band = Band(y_top=0, y_bottom=954, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertIn("bubble_mask", text)
        self.assertNotIn("bubble_mask_error", text)
        self.assertLess(text["bubble_mask_bbox"][2], 340)
        self.assertLess(text["bubble_mask_bbox"][3], 340)
        self.assertLess(int(np.count_nonzero(text["bubble_mask"] > 0)), 35000)
        self.assertLess(text["_raw_text_evidence_bbox"][2], 340)
        self.assertLess(text["_raw_text_evidence_bbox"][3], 340)

    def test_ensure_text_balloon_bboxes_rederives_irregular_image_white_mask_touching_art(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import cv2
        import numpy as np

        image = np.full((851, 800, 3), 28, dtype=np.uint8)
        starburst = np.asarray(
            [
                [230, 520],
                [310, 500],
                [360, 440],
                [430, 505],
                [530, 500],
                [500, 590],
                [545, 690],
                [430, 665],
                [360, 750],
                [310, 665],
                [205, 680],
                [245, 590],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [starburst], (255, 255, 255))
        cv2.polylines(image, [starburst], True, (0, 0, 0), 3)
        cv2.polylines(
            image,
            [np.asarray([[110, 320], [220, 260], [285, 180], [250, 390], [350, 285]], dtype=np.int32)],
            False,
            (255, 255, 255),
            24,
        )
        cv2.polylines(
            image,
            [np.asarray([[110, 320], [220, 260], [285, 180], [250, 390], [350, 285]], dtype=np.int32)],
            False,
            (0, 0, 0),
            3,
        )
        contaminated = np.zeros((851, 800), dtype=np.uint8)
        cv2.fillPoly(contaminated, [starburst], 255)
        cv2.polylines(
            contaminated,
            [np.asarray([[110, 320], [220, 260], [285, 180], [250, 390], [350, 285]], dtype=np.int32)],
            False,
            255,
            42,
        )
        contaminated[680:760, 0:260] = 255
        page = {
            "numero": 3,
            "width": 800,
            "height": 851,
            "_band_id": "page_003_band_023",
            "_vision_blocks": [
                {
                    "bbox": [0, 96, 555, 755],
                    "bubble_mask_bbox": [0, 96, 555, 755],
                    "bubble_mask": contaminated[96:755, 0:555],
                    "bubble_mask_source": "image_white_bubble_mask",
                }
            ],
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [270, 555, 500, 650],
                    "text_pixel_bbox": [270, 555, 500, 650],
                    "line_polygons": [[[270, 555], [500, 555], [500, 650], [270, 650]]],
                }
            ],
        }
        band = Band(y_top=0, y_bottom=851, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertIn("bubble_mask", text)
        self.assertNotIn("bubble_mask_error", text)
        self.assertGreaterEqual(text["bubble_mask_bbox"][0], 160)
        self.assertGreaterEqual(text["bubble_mask_bbox"][1], 420)
        self.assertLess(text["bubble_mask_bbox"][2], 620)
        self.assertLess(text["bubble_mask_bbox"][3], 780)
        self.assertLess(int(np.count_nonzero(text["bubble_mask"] > 0)), 85000)

    def test_ensure_text_balloon_bboxes_rederives_connected_panel_band_from_lower_balloon(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import cv2
        import numpy as np

        image = np.full((1035, 800, 3), 30, dtype=np.uint8)
        cv2.rectangle(image, (8, 137), (749, 310), (255, 255, 255), -1)
        cv2.ellipse(image, (505, 800), (245, 145), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (505, 800), (245, 145), 0, 0, 360, (0, 0, 0), 3)
        cv2.line(image, (260, 745), (8, 745), (255, 255, 255), 70)
        contaminated = np.zeros((779, 741), dtype=np.uint8)
        contaminated[0:174, :] = 255
        cv2.ellipse(contaminated, (497, 663), (245, 145), 0, 0, 360, 255, -1)
        cv2.line(contaminated, (252, 608), (0, 608), 255, 70)
        page = {
            "numero": 2,
            "width": 800,
            "height": 1035,
            "_band_id": "page_002_band_014",
            "_vision_blocks": [
                {
                    "bbox": [8, 137, 749, 916],
                    "bubble_mask_bbox": [8, 137, 749, 916],
                    "bubble_mask": contaminated,
                    "bubble_mask_source": "image_white_bubble_mask",
                }
            ],
            "texts": [
                {
                    "id": "ocr_003",
                    "bbox": [300, 730, 700, 910],
                    "text_pixel_bbox": [300, 730, 700, 910],
                    "line_polygons": [[[300, 730], [700, 730], [700, 910], [300, 910]]],
                }
            ],
        }
        band = Band(y_top=0, y_bottom=1035, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertIn("bubble_mask", text)
        self.assertNotIn("bubble_mask_error", text)
        self.assertGreater(text["bubble_mask_bbox"][0], 230)
        self.assertGreater(text["bubble_mask_bbox"][1], 620)
        self.assertLess(text["bubble_mask_bbox"][2], 760)
        self.assertLessEqual(text["bubble_mask_bbox"][3], 980)
        self.assertLess(int(np.count_nonzero(text["bubble_mask"] > 0)), 120000)

    def test_ensure_text_balloon_bboxes_uses_text_rect_fallback_for_translator_note_without_balloon(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import cv2
        import numpy as np

        image = np.full((360, 800, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (560, 120), (790, 208), (248, 248, 248), -1)
        cv2.putText(
            image,
            "TL/N: AISH IS A FORM",
            (603, 152),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "OF IRRITATED SPEECH.",
            (603, 178),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        page = {
            "numero": 2,
            "width": 800,
            "height": 360,
            "_band_id": "page_002_band_014",
            "_vision_blocks": [
                {
                    "bbox": [560, 120, 790, 208],
                    "bubble_mask_bbox": [560, 120, 790, 208],
                    "bubble_mask_source": "image_contour_bubble_mask",
                }
            ],
            "texts": [
                {
                    "id": "ocr_004",
                    "text": "TL/N: AISH IS A FORM OF IRRITATED SPEECH.",
                    "bbox": [597, 130, 767, 190],
                    "text_pixel_bbox": [603, 140, 760, 181],
                    "balloon_bbox": [560, 120, 790, 208],
                    "bubble_mask_bbox": [560, 120, 790, 208],
                }
            ],
        }
        band = Band(y_top=0, y_bottom=360, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_source"], "text_rect_fallback")
        self.assertNotIn("bubble_mask_error", text)
        self.assertEqual(text["bubble_mask_bbox"], [599, 136, 764, 185])
        self.assertEqual(text["bubble_mask"].shape, (49, 165))
        self.assertEqual(text["text_rect_fallback_padding_px"], 4)
        self.assertTrue(np.all(text["bubble_mask"] == 255))

    def test_text_rect_fallback_uses_text_pixels_not_broad_note_bbox(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import numpy as np

        image = np.full((220, 800, 3), 255, dtype=np.uint8)
        image[178:180, 620:760] = 0
        page = {
            "numero": 3,
            "width": 800,
            "height": 220,
            "_band_id": "page_003_band_022",
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "T/N: AJUMMA IS A TERM USED FOR MARRIED WOMEN.",
                    "bbox": [610, 88, 765, 180],
                    "text_pixel_bbox": [623, 104, 745, 161],
                    "line_polygons": [
                        [[623, 104], [745, 104], [745, 120], [623, 120]],
                        [[623, 122], [745, 122], [745, 138], [623, 138]],
                        [[623, 140], [745, 140], [745, 161], [623, 161]],
                    ],
                }
            ],
        }
        band = Band(y_top=0, y_bottom=220, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_source"], "text_rect_fallback")
        self.assertEqual(text["bubble_mask_bbox"], [619, 100, 749, 165])
        self.assertLess(text["bubble_mask_bbox"][3], 178)

    def test_dark_translator_note_uses_text_rect_not_dark_bubble_mask(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import numpy as np

        image = np.full((340, 360, 3), 4, dtype=np.uint8)
        image[220:290, 32:184] = 10
        page = {
            "numero": 5,
            "width": 360,
            "height": 340,
            "_band_id": "page_005_band_080",
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "T/N: THERE IS A NOVEL CALLED DEMI-GODS IN REAL LIFE.",
                    "bbox": [39, 218, 154, 273],
                    "text_pixel_bbox": [36, 225, 178, 269],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "balloon_bbox": [0, 214, 316, 332],
                    "bubble_mask_bbox": [0, 223, 316, 332],
                    "block_profile": "standard",
                }
            ],
        }
        band = Band(y_top=0, y_bottom=340, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_source"], "text_rect_fallback")
        self.assertEqual(text["bubble_mask_bbox"], [32, 221, 182, 273])
        self.assertEqual(text["balloon_bbox"], [32, 221, 182, 273])
        self.assertEqual(text["text_rect_fallback_padding_px"], 4)

    def test_ensure_text_balloon_bboxes_skips_text_rect_fallback_for_rotated_note(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import numpy as np

        image = np.full((220, 320, 3), 255, dtype=np.uint8)
        page = {
            "numero": 3,
            "width": 320,
            "height": 220,
            "_band_id": "page_003_band_038",
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "T/N: DIAGONAL NOTE",
                    "bbox": [80, 70, 230, 150],
                    "text_pixel_bbox": [84, 76, 226, 144],
                    "line_polygons": [
                        [[84, 130], [210, 62], [220, 80], [94, 148]],
                    ],
                    "rotation_deg": -33.36,
                    "rotation_source": "line_polygons",
                    "balloon_bbox": [40, 40, 280, 180],
                }
            ],
        }
        band = Band(y_top=0, y_bottom=220, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertNotEqual(text.get("bubble_mask_source"), "text_rect_fallback")
        self.assertNotIn("text_rect_fallback_bbox", text)

    def test_ensure_text_balloon_bboxes_does_not_default_to_derived_source_for_mask(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band
        import numpy as np

        image = np.full((120, 220, 3), 255, dtype=np.uint8)
        image[30:90, 40:180] = 255
        page = {
            "numero": 1,
            "width": 220,
            "height": 120,
            "_band_id": "page_001_band_001",
            "texts": [
                {
                    "id": "ocr_001",
                    "bbox": [80, 54, 140, 72],
                    "text_pixel_bbox": [80, 54, 140, 72],
                    "line_polygons": [[[80, 54], [140, 54], [140, 72], [80, 72]]],
                    "balloon_bbox": [40, 30, 180, 90],
                    "bubble_mask_bbox": [40, 30, 180, 90],
                    "bubble_mask": np.ones((60, 140), dtype=np.uint8) * 255,
                }
            ],
            "_vision_blocks": [],
        }
        band = Band(y_top=0, y_bottom=120, balloons=[], strip_slice=image, original_slice=image.copy())

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertNotEqual(text.get("bubble_mask_source"), "derived_white_crop")
        self.assertEqual(text.get("bubble_mask_source"), "image_white_bubble_mask")

    def test_band_to_page_dict_accepts_true_rectangular_balloon_with_border(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((120, 220, 3), 24, dtype=np.uint8)
        cv2.rectangle(slice_img, (40, 20), (180, 82), (255, 255, 255), -1)
        cv2.rectangle(slice_img, (40, 20), (180, 82), (0, 0, 0), 2)
        band = Band(
            y_top=300,
            y_bottom=420,
            balloons=[Balloon(strip_bbox=BBox(38, 318, 182, 384), confidence=0.91)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0, source_page_number=1)

        block = page_dict["_vision_blocks"][0]
        region = page_dict["_bubble_regions"][0]
        self.assertIn("bubble_mask", block)
        self.assertIn("bubble_mask", region)
        self.assertEqual(block["bubble_mask_source"], "image_rect_bubble_mask")
        self.assertGreater(int(block["bubble_mask"][32, 72]), 0)

    def test_band_to_page_dict_marks_bbox_fallback_when_real_bubble_mask_missing(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import numpy as np

        slice_img = np.full((100, 180, 3), (40, 80, 120), dtype=np.uint8)
        band = Band(
            y_top=200,
            y_bottom=300,
            balloons=[Balloon(strip_bbox=BBox(30, 220, 150, 280), confidence=0.77)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0, source_page_number=1)

        block = page_dict["_vision_blocks"][0]
        region = page_dict["_bubble_regions"][0]
        self.assertEqual(block["bubble_mask_bbox"], [30, 20, 150, 80])
        self.assertNotIn("bubble_mask", block)
        self.assertNotIn("bubble_mask", region)
        self.assertEqual(block["bubble_mask_source"], "bbox_fallback")
        self.assertEqual(region["bubble_mask_source"], "bbox_fallback")
        self.assertEqual(block["bubble_mask_error"], "missing_real_bubble_mask")

    def test_ensure_text_balloon_bboxes_replaces_tight_text_bubble_mask_bbox(self):
        from strip.process_bands import _band_to_page_dict, _ensure_text_balloon_bboxes
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((120, 220, 3), 24, dtype=np.uint8)
        cv2.ellipse(slice_img, (110, 60), (72, 34), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(slice_img, (110, 60), (72, 34), 0, 0, 360, (0, 0, 0), 2)
        band = Band(
            y_top=300,
            y_bottom=420,
            balloons=[Balloon(strip_bbox=BBox(30, 315, 190, 395), confidence=0.91)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )
        page = _band_to_page_dict(band, page_idx=0, source_page_number=1)
        page["texts"] = [
            {
                "bbox": [86, 40, 145, 58],
                "bubble_mask_bbox": [82, 34, 150, 64],
            }
        ]

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertEqual(text["balloon_bbox"], [86, 40, 145, 58])
        self.assertEqual(text["bubble_mask_bbox"], [30, 15, 190, 95])
        self.assertIn("bubble_mask", text)
        self.assertEqual(text["bubble_mask"].shape, (80, 160))
        self.assertGreater(int(text["bubble_mask"][40, 80]), 0)
        self.assertEqual(text["bubble_mask_source"], "image_white_bubble_mask")

    def test_ensure_text_balloon_bboxes_uses_band_balloon_when_vision_block_is_tight(self):
        from strip.process_bands import _band_to_page_dict, _ensure_text_balloon_bboxes
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((120, 220, 3), 24, dtype=np.uint8)
        cv2.ellipse(slice_img, (110, 60), (72, 34), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(slice_img, (110, 60), (72, 34), 0, 0, 360, (0, 0, 0), 2)
        band = Band(
            y_top=300,
            y_bottom=420,
            balloons=[Balloon(strip_bbox=BBox(30, 315, 190, 395), confidence=0.91)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )
        page = _band_to_page_dict(band, page_idx=0, source_page_number=1)
        page["_vision_blocks"] = [
            {
                "bbox": [82, 34, 150, 64],
                "bubble_id": "tight_text_region",
                "bubble_mask_bbox": [82, 34, 150, 64],
            }
        ]
        page["texts"] = [{"bbox": [86, 40, 145, 58], "bubble_mask_bbox": [82, 34, 150, 64]}]

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertEqual(text["balloon_bbox"], [86, 40, 145, 58])
        self.assertEqual(text["bubble_mask_bbox"], [30, 15, 190, 95])
        self.assertNotEqual(text["bubble_id"], "tight_text_region")
        self.assertIn("bubble_mask", text)
        self.assertEqual(text["bubble_mask"].shape, (80, 160))
        self.assertEqual(text["bubble_mask_source"], "image_white_bubble_mask")

    def test_ensure_text_balloon_bboxes_uses_layout_balloon_bbox_for_real_mask(self):
        from strip.process_bands import _ensure_text_balloon_bboxes
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((120, 220, 3), 24, dtype=np.uint8)
        cv2.ellipse(slice_img, (110, 60), (72, 34), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(slice_img, (110, 60), (72, 34), 0, 0, 360, (0, 0, 0), 2)
        band = Band(
            y_top=300,
            y_bottom=420,
            balloons=[Balloon(strip_bbox=BBox(82, 334, 150, 364), confidence=0.70)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )
        page = {
            "numero": 1,
            "width": 220,
            "height": 120,
            "_band_id": "page_001_band_001",
            "_vision_blocks": [
                {
                    "bbox": [82, 34, 150, 64],
                    "bubble_id": "tight_text_region",
                    "bubble_mask_bbox": [82, 34, 150, 64],
                }
            ],
            "texts": [
                {
                    "bbox": [86, 40, 145, 58],
                    "balloon_bbox": [30, 15, 190, 95],
                    "bubble_mask_bbox": [82, 34, 150, 64],
                    "bubble_id": "tight_text_region",
                }
            ],
        }

        _ensure_text_balloon_bboxes(page, band)

        text = page["texts"][0]
        self.assertEqual(text["balloon_bbox"], [30, 15, 190, 95])
        self.assertIn("bubble_mask", text)
        mask_bbox = text["bubble_mask_bbox"]
        self.assertLessEqual(mask_bbox[0], 60)
        self.assertLessEqual(mask_bbox[1], 20)
        self.assertGreaterEqual(mask_bbox[2], 170)
        self.assertGreaterEqual(mask_bbox[3], 85)
        self.assertEqual(text["bubble_mask"].shape, (mask_bbox[3] - mask_bbox[1], mask_bbox[2] - mask_bbox[0]))
        self.assertGreater(int(text["bubble_mask"][text["bubble_mask"].shape[0] // 2, text["bubble_mask"].shape[1] // 2]), 0)


class CopyBackOutsideBalloonsTests(unittest.TestCase):
    def test_copy_back_preserves_pixels_outside_balloons(self):
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox
        import numpy as np

        original = np.full((100, 300, 3), 50, dtype=np.uint8)
        rendered = np.full((100, 300, 3), 200, dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9)],
            original_slice=original,
            rendered_slice=rendered,
        )

        result = _apply_copy_back_outside_balloons(band, balloon_margin=8)

        self.assertEqual(result[50, 100, 0], 200)
        self.assertEqual(result[5, 5, 0], 50)
        outside_y_top = result[:12, :, :]
        self.assertTrue(np.array_equal(outside_y_top, original[:12, :, :]))

    def test_copy_back_diff_below_2_outside_balloons(self):
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox
        import numpy as np

        rng = np.random.default_rng(42)
        original = rng.integers(0, 256, (100, 300, 3), dtype=np.uint8)
        rendered = rng.integers(0, 256, (100, 300, 3), dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9)],
            original_slice=original,
            rendered_slice=rendered,
        )

        result = _apply_copy_back_outside_balloons(band, balloon_margin=8)

        mask_inside = np.zeros(result.shape[:2], dtype=bool)
        mask_inside[12:88, 42:158] = True

        diff = np.abs(result.astype(np.int16) - original.astype(np.int16))
        # Fora da banda interna, pixels devem ser identicos ao original
        self.assertTrue(np.all(diff[~mask_inside] == 0),
            "Pixels fora do bbox+margin foram alterados pelo copy-back")

    def test_copy_back_pixel_perfect_outside_balloon(self):
        """Criterio Q2=a: pixels fora do balloon bbox+margin devem ser identicos ao original."""
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox
        import numpy as np

        rng = np.random.default_rng(7)
        original = rng.integers(0, 256, (300, 600, 3), dtype=np.uint8)
        rendered = rng.integers(0, 256, (300, 600, 3), dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=300,
            balloons=[Balloon(strip_bbox=BBox(100, 50, 300, 200), confidence=0.9)],
            original_slice=original.copy(),
            rendered_slice=rendered.copy(),
        )
        result = _apply_copy_back_outside_balloons(band, balloon_margin=8)

        # Fora do bbox+margin (y < 42, x < 92, etc.) deve ser identico ao original
        self.assertTrue(np.array_equal(result[:42, :, :], original[:42, :, :]),
            "Rows acima do balloon nao sao pixel-perfect iguais ao original")
        # Dentro do bbox deve ser identico ao rendered
        self.assertTrue(np.array_equal(result[60:190, 110:290, :], rendered[60:190, 110:290, :]),
            "Interior do balloon nao e identico ao rendered")


class SmartSkipShadowTests(unittest.TestCase):
    def test_apply_smart_skip_shadow_records_audit_without_skip_processing_mutation(self):
        from strip.process_bands import _apply_smart_skip_shadow

        page = {
            "numero": 1,
            "texts": [
                {
                    "id": "credit",
                    "text": "FOR FASTER UPDATE",
                    "confidence": 0.0,
                    "bbox": [10, 10, 180, 40],
                    "skip_processing": False,
                },
                {
                    "id": "dialogue",
                    "text": "IS THIS RECORDING?",
                    "confidence": 0.95,
                    "bbox": [20, 60, 220, 130],
                    "skip_processing": False,
                },
            ],
        }
        perf = {}

        _apply_smart_skip_shadow(page, perf)

        self.assertFalse(page["texts"][0]["skip_processing"])
        self.assertFalse(page["texts"][1]["skip_processing"])
        self.assertEqual(page["_smart_skip_shadow"]["candidate_count"], 1)
        self.assertEqual(perf["smart_skip_shadow_candidate_count"], 1)
        self.assertEqual(perf["smart_skip_shadow_not_safe_count"], 1)
        self.assertEqual(
            perf["smart_skip_shadow_category_counts"]["credit_or_watermark"],
            1,
        )

    def test_apply_smart_skip_real_records_audit_without_skip_processing_mutation(self):
        from strip.process_bands import _apply_smart_skip_real

        page = {
            "numero": 1,
            "texts": [
                {
                    "id": "credit",
                    "text": "All comics on this website are just previews...",
                    "confidence": 0.95,
                    "bbox": [10, 10, 220, 60],
                    "skip_processing": False,
                },
                {
                    "id": "timer",
                    "text": "00:00:05",
                    "confidence": 0.8,
                    "bbox": [20, 80, 120, 110],
                    "skip_processing": False,
                },
            ],
        }
        perf = {}

        applied = _apply_smart_skip_real(page, perf)

        self.assertFalse(applied)
        self.assertFalse(page["texts"][0]["skip_processing"])
        self.assertFalse(page["texts"][1]["skip_processing"])
        self.assertNotIn("skip_reason", page["texts"][0])
        self.assertNotIn("skip_reason", page["texts"][1])
        self.assertIn("smart_skip_decision", page["texts"][0])
        self.assertEqual(perf["smart_skip_real_candidate_count"], 2)
        self.assertEqual(perf["smart_skip_real_not_safe_count"], 0)
        self.assertFalse(perf["smart_skip_real_applied"])

    def test_apply_smart_skip_real_does_not_mutate_mixed_bands(self):
        from strip.process_bands import _apply_smart_skip_real

        page = {
            "numero": 1,
            "texts": [
                {
                    "id": "credit",
                    "text": "FOR FASTER UPDATE",
                    "confidence": 0.95,
                    "bbox": [10, 10, 220, 60],
                    "skip_processing": False,
                },
                {
                    "id": "dialogue",
                    "text": "IS THIS RECORDING?",
                    "confidence": 0.95,
                    "bbox": [20, 80, 220, 130],
                    "skip_processing": False,
                },
            ],
        }
        perf = {}

        applied = _apply_smart_skip_real(page, perf)

        self.assertFalse(applied)
        self.assertFalse(page["texts"][0]["skip_processing"])
        self.assertFalse(page["texts"][1]["skip_processing"])
        self.assertNotIn("skip_reason", page["texts"][0])
        self.assertEqual(perf["smart_skip_real_candidate_count"], 1)
        self.assertEqual(perf["smart_skip_real_not_safe_count"], 1)
        self.assertFalse(perf["smart_skip_real_applied"])


class ProcessBandTests(unittest.TestCase):
    def _make_band(self):
        from strip.types import Band, Balloon, BBox
        import numpy as np
        slice_img = np.full((100, 300, 3), 200, dtype=np.uint8)
        return Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )

    def test_process_band_does_not_recover_legacy_top_narration_visual_rect(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import process_band
        from strip.types import Band, Balloon, BBox
        import copy
        import cv2
        import numpy as np

        page_bgr = np.full((260, 800, 3), 255, dtype=np.uint8)
        cv2.rectangle(page_bgr, (259, 30), (734, 230), (0, 0, 0), 2)
        band_y_top = 80
        band_rgb = cv2.cvtColor(page_bgr[band_y_top:224, :, :], cv2.COLOR_BGR2RGB)
        band = Band(
            y_top=band_y_top,
            y_bottom=224,
            balloons=[Balloon(strip_bbox=BBox(160, 80, 800, 224), confidence=0.9)],
            strip_slice=band_rgb.copy(),
            original_slice=band_rgb.copy(),
        )

        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [314, 16, 682, 128],
                    "text": "LIVING IS NOT FUN BUT THAT DOESN'T MEAN I HAVE THE COURAGE TO DIE.",
                    "tipo": "narracao",
                    "block_profile": "top_narration",
                    "layout_profile": "top_narration",
                    "text_pixel_bbox": [317, 22, 680, 122],
                }
            ],
            "_vision_blocks": [{"bbox": [160, 0, 800, 144], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **_kw: pages
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _page: img.copy()
        captured = {}
        typesetter = MagicMock()

        def fake_render(img, page):
            captured["page"] = copy.deepcopy(page)
            return img.copy()

        typesetter.render_band_image.side_effect = fake_render

        with patch("ocr.contextual_reviewer.contextual_review_page", side_effect=lambda page, *_args: page):
            with patch("layout.balloon_layout.enrich_page_layout", side_effect=lambda page: page):
                process_band(
                    band,
                    runtime=runtime,
                    translator=translator,
                    inpainter=inpainter,
                    typesetter=typesetter,
                    page_idx=0,
                    layout_page_image_bgr=page_bgr,
                    layout_page_y_top=0,
                )

        text = captured["page"]["texts"][0]
        self.assertNotEqual(text.get("layout_reason"), "visual_rect_full_page")
        self.assertNotIn("_visual_rect_outer_bbox", text)
        self.assertNotIn("layout_safe_reason", text)

    def test_inpaint_stage_does_not_receive_legacy_decision_fields(self):
        from strip.process_bands import _run_inpaint_stage
        from strip.types import Band
        import numpy as np

        band = Band(
            y_top=0,
            y_bottom=80,
            strip_slice=np.full((80, 120, 3), 255, dtype=np.uint8),
        )
        translated_page = {
            "numero": 1,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [20, 20, 70, 44],
                    "translated": "OLA",
                    "skip_processing": True,
                    "preserve_original": True,
                    "tipo": "sfx",
                    "content_class": "sound_effect",
                    "balloon_type": "textured",
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [18, 18, 74, 48],
                    "skip_processing": True,
                    "preserve_original": True,
                    "tipo": "sfx",
                    "content_class": "sound_effect",
                    "balloon_type": "textured",
                }
            ],
        }
        captured = {}

        class CapturingInpainter:
            def inpaint_band_image(self, image, page):
                captured["page"] = page
                return image.copy()

        _run_inpaint_stage(
            band,
            inpainter=CapturingInpainter(),
            translated_page=translated_page,
        )

        for key in ("skip_processing", "preserve_original", "tipo", "content_class", "balloon_type"):
            self.assertNotIn(key, captured["page"]["texts"][0])
            self.assertNotIn(key, captured["page"]["_vision_blocks"][0])
        self.assertTrue(translated_page["texts"][0]["skip_processing"])
        self.assertEqual(translated_page["texts"][0]["tipo"], "sfx")

    def test_typeset_stage_does_not_receive_legacy_decision_fields(self):
        from strip.process_bands import _run_typeset_stage
        import numpy as np

        cleaned = np.full((80, 120, 3), 255, dtype=np.uint8)
        translated_page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [20, 20, 70, 44],
                    "translated": "OLA",
                    "skip_processing": True,
                    "preserve_original": True,
                    "tipo": "narracao",
                    "content_class": "narration",
                    "balloon_type": "white",
                }
            ]
        }
        captured = {}

        class CapturingTypesetter:
            def render_band_image(self, image, page):
                captured["page"] = page
                return image.copy()

        _run_typeset_stage(
            cleaned,
            typesetter=CapturingTypesetter(),
            translated_page=translated_page,
        )

        for key in ("skip_processing", "preserve_original", "tipo", "content_class", "balloon_type"):
            self.assertNotIn(key, captured["page"]["texts"][0])
        self.assertTrue(translated_page["texts"][0]["skip_processing"])
        self.assertEqual(translated_page["texts"][0]["tipo"], "narracao")

    def test_process_band_populates_rendered_slice(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        # Stages mockadas — só precisam retornar dict válido / ndarray
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [
                {"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"},
            ],
            "_ocr_stats": {"sparse_crop_fallback_max": 0, "crop_fallback_suppressed": 2},
        }
        translator = MagicMock()
        translator.translate_pages.return_value = ([{
            "texts": [{"id": "t1", "translated": "OLÁ", "tipo": "fala", "bbox": [50, 20, 150, 80]}]
        }], [])
        inpainter = MagicMock()

        def fake_inpaint(_image, page):
            page["_strip_fast_white_balloon_count"] = 1
            page["_strip_connected_white_geometry_fill_count"] = 1
            page["_strip_connected_white_geometry_fill_mask_pixels"] = 42
            page["_strip_fast_local_balloon_count"] = 2
            page["_strip_fast_dark_panel_fill_count"] = 3
            page["_strip_dark_panel_fill_count"] = 3
            page["_strip_remaining_inpaint_blocks"] = 3
            page["_strip_fast_white_rejection_reasons"] = {"no_white_fill_mask": 4}
            page["_strip_connected_white_rejection_reasons"] = {"mask_evidence:missing": 2}
            page["_strip_fast_local_rejection_reasons"] = {"no_flat_fill": 5}
            page["_strip_fast_dark_rejection_reasons"] = {"mask_evidence:missing": 6}
            return np.full((100, 300, 3), 255, dtype=np.uint8)

        inpainter.inpaint_band_image.side_effect = fake_inpaint
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        self.assertIs(result, band)
        self.assertIsNotNone(band.rendered_slice)
        # Stages foram chamadas
        runtime.run_ocr_stage.assert_called_once()
        translator.translate_pages.assert_called_once()
        inpainter.inpaint_band_image.assert_called_once()
        typesetter.render_band_image.assert_called_once()
        self.assertIn("durations_sec", band.perf)
        self.assertIn("ocr", band.perf["durations_sec"])
        self.assertIn("translate", band.perf["durations_sec"])
        self.assertIn("inpaint", band.perf["durations_sec"])
        self.assertIn("typeset", band.perf["durations_sec"])
        for stage in ("ocr", "inpaint", "typeset"):
            self.assertIn(f"{stage}_wait", band.perf["durations_sec"])
            self.assertIn(f"{stage}_compute", band.perf["durations_sec"])
        self.assertEqual(band.ocr_result.get("_perf", {}).get("ocr_text_count"), 1)
        self.assertEqual(band.perf["fast_white_balloon_count"], 1)
        self.assertEqual(band.perf["connected_white_geometry_fill_count"], 1)
        self.assertEqual(band.perf["connected_white_geometry_fill_mask_pixels"], 42)
        self.assertEqual(band.perf["fast_local_balloon_count"], 2)
        self.assertEqual(band.perf["fast_dark_panel_fill_count"], 3)
        self.assertEqual(band.perf["dark_panel_fill_count"], 3)
        self.assertEqual(band.perf["remaining_inpaint_blocks"], 3)
        self.assertEqual(band.perf["ocr_sparse_crop_fallback_max"], 0)
        self.assertEqual(band.perf["ocr_crop_fallback_suppressed"], 2)
        self.assertEqual(band.perf["fast_white_rejection_reasons"], {"no_white_fill_mask": 4})
        self.assertEqual(
            band.perf["connected_white_rejection_reasons"], {"mask_evidence:missing": 2}
        )
        self.assertEqual(band.perf["fast_local_rejection_reasons"], {"no_flat_fill": 5})
        self.assertEqual(band.perf["fast_dark_rejection_reasons"], {"mask_evidence:missing": 6})

    def test_process_band_recovers_empty_ocr_with_candidate_crop_reocr(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import process_band
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((120, 300, 3), 255, dtype=np.uint8)
        cv2.putText(
            slice_img,
            "ONE",
            (74, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        band = Band(
            y_top=100,
            y_bottom=220,
            balloons=[Balloon(strip_bbox=BBox(40, 120, 220, 200), confidence=0.9)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )

        runtime = MagicMock()
        runtime.run_ocr_stage.side_effect = [
            {"texts": [], "_vision_blocks": [], "_ocr_stats": {"quick_skipped_no_text": True}},
            {
                "texts": [
                    {
                        "id": "crop_001",
                        "bbox": [42, 28, 92, 52],
                        "text": "ONE",
                        "confidence": 0.91,
                    }
                ],
                "_vision_blocks": [{"bbox": [28, 14, 124, 64], "confidence": 0.9}],
            },
        ]
        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **_kw: [
            {
                **pages[0],
                "texts": [
                    {**text, "translated": "UM"}
                    for text in pages[0].get("texts", [])
                ],
            }
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _page: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _page: img.copy()

        with patch("ocr.contextual_reviewer.contextual_review_page", side_effect=lambda page, *_args: page):
            with patch("layout.balloon_layout.enrich_page_layout", side_effect=lambda page: page):
                process_band(
                    band,
                    runtime=runtime,
                    translator=translator,
                    inpainter=inpainter,
                    typesetter=typesetter,
                    page_idx=7,
                    source_page_number=3,
                )

        self.assertEqual(runtime.run_ocr_stage.call_count, 2)
        self.assertEqual(translator.translate_pages.call_count, 1)
        self.assertEqual(band.ocr_result["texts"][0]["text"], "ONE")
        self.assertEqual(band.ocr_result["texts"][0]["translated"], "UM")
        self.assertEqual(band.ocr_result["texts"][0]["bbox"], [62, 28, 112, 52])
        self.assertEqual(band.ocr_result["_perf"]["ocr_candidate_crop_recovered"], 1)
        self.assertNotEqual(
            band.ocr_result.get("_copyback_decision", {}).get("reason"),
            "no_texts",
        )

    def test_process_band_recovers_empty_dark_panel_without_light_text_evidence(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import process_band
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((260, 820, 3), [4, 8, 12], dtype=np.uint8)
        cv2.rectangle(slice_img, (545, 80), (795, 190), (4, 4, 5), -1)
        cv2.rectangle(slice_img, (545, 80), (795, 190), (176, 180, 196), 2)
        cv2.putText(
            slice_img,
            "The episode starts!",
            (584, 148),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.74,
            (244, 250, 255),
            2,
            cv2.LINE_AA,
        )
        band = Band(
            y_top=3000,
            y_bottom=3260,
            balloons=[Balloon(strip_bbox=BBox(578, 3101, 775, 3150), confidence=0.7607)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )

        runtime = MagicMock()
        runtime.run_ocr_stage.side_effect = [
            {"texts": [], "_vision_blocks": [], "_ocr_stats": {"quick_skipped_no_text": True}},
            {"texts": [], "_vision_blocks": []},
        ]
        direct_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "The episode starts!",
                    "bbox": [40, 95, 224, 124],
                    "source_bbox": [40, 95, 224, 124],
                    "text_pixel_bbox": [40, 95, 224, 124],
                    "line_polygons": [[[40, 95], [224, 95], [224, 124], [40, 124]]],
                    "confidence": 0.976,
                    "confidence_raw": 0.976,
                    "qa_flags": ["candidate_crop_direct_paddle_reocr"],
                }
            ],
            "_vision_blocks": [{"bbox": [40, 95, 224, 124], "confidence": 0.976}],
            "width": 270,
            "height": 129,
        }
        weak_dark_evidence = {
            "has_inner_dark_text": True,
            "has_inner_light_text": False,
            "inner_dark_component_count": 4,
            "inner_dark_area": 125,
            "inner_light_component_count": 0,
            "inner_light_area": 0,
            "significant_component_count": 0,
            "significant_area": 0,
            "bright_pixel_ratio": 0.1253,
            "dark_pixel_ratio": 0.7696,
        }
        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **_kw: [
            {
                **pages[0],
                "texts": [
                    {**text, "translated": "O EPISODIO COMECA!"}
                    for text in pages[0].get("texts", [])
                ],
            }
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _page: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _page: img.copy()

        with patch("strip.detect_balloons._inner_dark_text_evidence", return_value=weak_dark_evidence), \
            patch("strip.process_bands._run_direct_paddle_candidate_crop_reocr", return_value=direct_page) as direct_ocr, \
            patch("ocr.contextual_reviewer.contextual_review_page", side_effect=lambda page, *_args: page), \
            patch("layout.balloon_layout.enrich_page_layout", side_effect=lambda page: page):
            process_band(
                band,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
                page_idx=4,
                source_page_number=2,
            )

        self.assertEqual(runtime.run_ocr_stage.call_count, 2)
        self.assertEqual(direct_ocr.call_count, 1)
        text = band.ocr_result["texts"][0]
        self.assertEqual(text["text"], "The episode starts!")
        self.assertEqual(text["translated"], "O EPISODIO COMECA!")
        self.assertEqual(text["bubble_mask_source"], "image_dark_panel_mask")
        self.assertEqual(text["block_profile"], "dark_panel")
        self.assertEqual(text["card_panel_text_context"], True)
        self.assertEqual(band.ocr_result["_perf"]["ocr_candidate_crop_recovered"], 1)
        self.assertNotEqual(
            band.ocr_result.get("_copyback_decision", {}).get("reason"),
            "no_texts",
        )

    def test_process_band_keeps_system_ui_crop_reocr_with_panel_sized_bbox(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import process_band
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((260, 820, 3), [5, 8, 12], dtype=np.uint8)
        cv2.rectangle(slice_img, (38, 48), (244, 210), (3, 4, 6), -1)
        cv2.rectangle(slice_img, (38, 48), (244, 210), (158, 174, 192), 2)
        cv2.putText(
            slice_img,
            "Main Quest will",
            (58, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (244, 250, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            slice_img,
            "be shown shortly.",
            (58, 146),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (244, 250, 255),
            2,
            cv2.LINE_AA,
        )
        band = Band(
            y_top=2187,
            y_bottom=2447,
            balloons=[Balloon(strip_bbox=BBox(38, 2235, 244, 2397), confidence=0.86)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )

        runtime = MagicMock()
        runtime.run_ocr_stage.side_effect = [
            {"texts": [], "_vision_blocks": [], "_ocr_stats": {"quick_skipped_no_text": True}},
            {
                "texts": [
                    {
                        "id": "crop_001",
                        "text_id": "crop_001",
                        "text": "Main Quest will be shown shortly",
                        "bbox": [0, 0, 210, 168],
                        "source_bbox": [0, 0, 210, 168],
                        "text_pixel_bbox": [0, 0, 210, 168],
                        "confidence": 0.854,
                        "confidence_raw": 0.854,
                    }
                ],
                "_vision_blocks": [{"bbox": [0, 0, 210, 168], "confidence": 0.854}],
                "width": 210,
                "height": 168,
            },
        ]
        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **_kw: [
            {
                **pages[0],
                "texts": [
                    {**text, "translated": "A MISSAO PRINCIPAL SERA MOSTRADA EM BREVE."}
                    for text in pages[0].get("texts", [])
                ],
            }
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _page: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _page: img.copy()
        strong_light_text_evidence = {
            "has_inner_dark_text": False,
            "has_inner_light_text": True,
            "inner_dark_component_count": 0,
            "inner_dark_area": 0,
            "inner_light_component_count": 4,
            "inner_light_area": 420,
            "significant_component_count": 0,
            "significant_area": 0,
            "bright_pixel_ratio": 0.08,
            "dark_pixel_ratio": 0.74,
        }

        with patch("strip.detect_balloons._inner_dark_text_evidence", return_value=strong_light_text_evidence), \
            patch("strip.process_bands._detect_dark_rect_panel_frame_bbox", return_value=None), \
            patch("ocr.contextual_reviewer.contextual_review_page", side_effect=lambda page, *_args: page), \
            patch("layout.balloon_layout.enrich_page_layout", side_effect=lambda page: page):
            process_band(
                band,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
                page_idx=2,
                source_page_number=2,
            )

        self.assertEqual(runtime.run_ocr_stage.call_count, 2)
        self.assertEqual(translator.translate_pages.call_count, 1)
        self.assertEqual(band.ocr_result["texts"][0]["text"], "Main Quest will be shown shortly")
        self.assertEqual(
            band.ocr_result["texts"][0]["translated"],
            "A MISSAO PRINCIPAL SERA MOSTRADA EM BREVE.",
        )
        self.assertEqual(band.ocr_result["_perf"]["ocr_candidate_crop_recovered"], 1)
        self.assertNotEqual(
            band.ocr_result.get("_copyback_decision", {}).get("reason"),
            "no_texts",
        )

    def test_candidate_crop_reocr_rejects_panel_sized_scanlation_credit(self):
        from strip.process_bands import _candidate_crop_reocr_bbox_is_reasonable

        self.assertFalse(
            _candidate_crop_reocr_bbox_is_reasonable(
                {
                    "text": "SECRET SCANS PRESENTS",
                    "bbox": [0, 0, 210, 168],
                    "confidence": 0.95,
                },
                crop_width=210,
                crop_height=168,
            )
        )

    def test_candidate_crop_reocr_rejects_dark_bubble_scanlation_url(self):
        from strip.process_bands import _candidate_crop_reocr_text_is_usable

        self.assertFalse(
            _candidate_crop_reocr_text_is_usable(
                {
                    "text": "MZ http://mzfamily.co.kr",
                    "confidence": 0.95,
                    "block_profile": "dark_bubble",
                    "bubble_mask_source": "image_dark_bubble_mask",
                }
            )
        )

    def test_candidate_crop_reocr_rejects_truncated_discord_scanlation_url(self):
        from strip.process_bands import (
            _candidate_crop_reocr_result_has_scanlation_credit,
            _candidate_crop_reocr_text_is_usable,
        )

        text = {"text": "iscord.gg/xzeKn8V", "confidence": 0.95}

        self.assertFalse(_candidate_crop_reocr_text_is_usable(text))
        self.assertTrue(_candidate_crop_reocr_result_has_scanlation_credit({"texts": [text]}))

    def test_candidate_crop_reocr_result_rejects_scanlation_credit_panel(self):
        from strip.process_bands import _candidate_crop_reocr_result_has_scanlation_credit

        self.assertTrue(
            _candidate_crop_reocr_result_has_scanlation_credit(
                {
                    "texts": [
                        {"text": "SECRET SCANS", "confidence": 0.98},
                        {"text": "SCANS IS LOOKING FOR STAFF TO HELP US RELEASE", "confidence": 0.92},
                        {"text": "TROIENTRVE, FOR, TORS RAWERS PART JS OUT", "confidence": 0.90},
                    ]
                }
            )
        )

    def test_candidate_crop_reocr_result_keeps_dark_bubble_dialogue_panel(self):
        from strip.process_bands import _candidate_crop_reocr_result_has_scanlation_credit

        self.assertFalse(
            _candidate_crop_reocr_result_has_scanlation_credit(
                {"texts": [{"text": "You were loyal to others, but to them, you were being nosy.", "confidence": 0.96}]}
            )
        )

    def test_candidate_crop_merge_rejects_recovered_text_near_scanlation_credit(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        ocr_page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "WE ARE RECRUITING!",
                    "bbox": [187, 12184, 344, 12231],
                    "qa_flags": ["scanlation_credit_suppressed"],
                    "route_reason": "scanlation_credit_suppressed",
                },
                {
                    "id": "ocr_003",
                    "text": "WE ARE LOOKING FOR KR/JP TRANSLATORS PROOFREADERS.",
                    "bbox": [211, 12408, 588, 12603],
                    "qa_flags": [],
                },
            ],
            "_vision_blocks": [],
        }
        recovered_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text": "W SECRET OUR C",
                    "bbox": [164, 12254, 224, 12411],
                    "qa_flags": ["candidate_crop_direct_paddle_reocr"],
                },
                {
                    "id": "direct_paddle_reocr_001_002",
                    "text": "TROIENTRVE, FOR, TORS RAWERS PART JS OUT",
                    "bbox": [522, 12411, 632, 12421],
                    "qa_flags": ["candidate_crop_direct_paddle_reocr"],
                },
            ],
            "_vision_blocks": [],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(ocr_page, recovered_page)

        self.assertEqual(merged, 0)
        self.assertEqual(len(ocr_page["texts"]), 2)

    def test_candidate_crop_reocr_keeps_dark_bubble_dialogue_with_panel_sized_bbox(self):
        from strip.process_bands import _candidate_crop_reocr_bbox_is_reasonable

        self.assertTrue(
            _candidate_crop_reocr_bbox_is_reasonable(
                {
                    "text": "That means you're talented!",
                    "bbox": [0, 0, 260, 180],
                    "confidence": 0.91,
                    "block_profile": "dark_bubble",
                    "bubble_mask_source": "image_dark_bubble_mask",
                },
                crop_width=260,
                crop_height=180,
            )
        )

    def test_dark_bubble_reocr_replaces_overlapping_partial_ocr(self):
        from strip.process_bands import _merge_candidate_crop_recovery_into_ocr_page

        ocr_page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "text": "Your predecessors",
                    "bbox": [500, 210, 660, 236],
                    "text_pixel_bbox": [402, 203, 654, 241],
                    "bubble_mask_source": "image_white_bubble_mask",
                    "block_profile": "standard",
                }
            ],
            "_vision_blocks": [{"bbox": [402, 203, 654, 241]}],
        }
        recovered_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "You were chosen to become King Yeomra! Your predecessors chose you!",
                    "bbox": [388, 82, 710, 248],
                    "text_pixel_bbox": [388, 82, 710, 248],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "block_profile": "dark_bubble",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["candidate_crop_direct_paddle_reocr", "dark_bubble_oval_reocr"],
                    "reocr_candidate_index": 0,
                }
            ],
            "_vision_blocks": [{"bbox": [360, 40, 740, 280], "reocr_candidate_index": 0}],
        }

        merged = _merge_candidate_crop_recovery_into_ocr_page(ocr_page, recovered_page)

        self.assertEqual(merged, 1)
        self.assertEqual(len(ocr_page["texts"]), 1)
        self.assertEqual(ocr_page["texts"][0]["text_id"], "direct_paddle_reocr_001")
        self.assertEqual(ocr_page["texts"][0]["bubble_mask_source"], "image_dark_bubble_mask")

    def test_partial_dark_bubble_recovery_replaces_partial_text(self):
        from unittest.mock import patch
        from strip.process_bands import _recover_partial_dark_bubble_ocr_from_texts
        from strip.types import Band
        import cv2
        import numpy as np

        image = np.full((330, 820, 3), [3, 4, 6], dtype=np.uint8)
        cv2.ellipse(image, (555, 160), (210, 118), 0, 0, 360, (2, 2, 2), -1)
        cv2.ellipse(image, (555, 160), (210, 118), 0, 0, 360, (25, 118, 158), 4)
        ocr_page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "text": "Your predecessors",
                    "bbox": [500, 210, 660, 236],
                    "text_pixel_bbox": [402, 203, 654, 241],
                    "confidence": 0.56,
                    "confidence_raw": 0.56,
                    "bubble_mask_source": "image_white_bubble_mask",
                }
            ],
            "_vision_blocks": [{"bbox": [402, 203, 654, 241]}],
        }
        direct_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "You were chosen to become King Yeomra! Your predecessors chose you!",
                    "bbox": [36, 30, 380, 210],
                    "source_bbox": [36, 30, 380, 210],
                    "text_pixel_bbox": [36, 30, 380, 210],
                    "confidence": 0.93,
                    "confidence_raw": 0.93,
                    "qa_flags": ["candidate_crop_direct_paddle_reocr"],
                }
            ],
            "_vision_blocks": [{"bbox": [36, 30, 380, 210], "confidence": 0.93}],
            "width": 440,
            "height": 278,
        }
        evidence = {
            "has_inner_light_text": True,
            "inner_light_area": 1200,
            "bright_pixel_ratio": 0.08,
            "dark_pixel_ratio": 0.82,
        }
        band = Band(y_top=0, y_bottom=330, balloons=[], strip_slice=image, original_slice=image.copy())

        with patch("strip.detect_balloons._inner_dark_text_evidence", return_value=evidence), \
            patch("strip.process_bands._detect_dark_oval_bubble_bbox", return_value=None), \
            patch("strip.process_bands._run_direct_paddle_candidate_crop_reocr", return_value=direct_page):
            recovered = _recover_partial_dark_bubble_ocr_from_texts(
                ocr_page,
                band=band,
                page_dict={"width": 820, "height": 330},
                band_id="page_003_band_028",
            )

        self.assertEqual(recovered, 1)
        self.assertEqual(len(ocr_page["texts"]), 1)
        self.assertEqual(ocr_page["texts"][0]["text_id"], "direct_paddle_reocr_001")
        self.assertEqual(ocr_page["texts"][0]["bubble_mask_source"], "image_dark_bubble_mask")

    def test_partial_dark_bubble_recovery_adds_uncovered_connected_lobe(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import _recover_partial_dark_bubble_ocr_from_texts
        from strip.types import Band
        import cv2
        import numpy as np

        image = np.full((460, 820, 3), [3, 4, 6], dtype=np.uint8)
        cv2.ellipse(image, (220, 165), (190, 118), 0, 0, 360, (2, 2, 2), -1)
        cv2.ellipse(image, (220, 165), (190, 118), 0, 0, 360, (25, 118, 158), 4)
        cv2.ellipse(image, (590, 230), (245, 150), 0, 0, 360, (2, 2, 2), -1)
        cv2.ellipse(image, (590, 230), (245, 150), 0, 0, 360, (25, 118, 158), 4)
        cv2.putText(image, "Your heart felt", (110, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2, cv2.LINE_AA)
        cv2.putText(image, "empty all the time.", (88, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2, cv2.LINE_AA)
        ocr_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "You lived without planning for the future because you didn't have the will to live.",
                    "bbox": [455, 160, 760, 330],
                    "text_pixel_bbox": [455, 160, 760, 330],
                    "confidence": 0.96,
                    "confidence_raw": 0.96,
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["dark_bubble_oval_reocr"],
                }
            ],
            "_vision_blocks": [{"bbox": [455, 160, 760, 330]}],
        }
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "text": "Your heart felt empty and lonely all the time.",
                    "bbox": [55, 78, 330, 190],
                    "source_bbox": [55, 78, 330, 190],
                    "text_pixel_bbox": [55, 78, 330, 190],
                    "confidence": 0.91,
                    "confidence_raw": 0.91,
                }
            ],
            "_vision_blocks": [{"bbox": [55, 78, 330, 190], "confidence": 0.91}],
        }
        evidence = {
            "has_inner_light_text": True,
            "inner_light_area": 1300,
            "bright_pixel_ratio": 0.07,
            "dark_pixel_ratio": 0.80,
        }
        band = Band(y_top=0, y_bottom=460, balloons=[], strip_slice=image, original_slice=image.copy())

        with patch("strip.detect_balloons._inner_dark_text_evidence", return_value=evidence), \
            patch("strip.process_bands._detect_dark_oval_bubble_bbox", return_value=[40, 20, 810, 410]):
            recovered = _recover_partial_dark_bubble_ocr_from_texts(
                ocr_page,
                band=band,
                page_dict={"width": 820, "height": 460},
                band_id="page_003_band_023",
                runtime=runtime,
            )

        self.assertEqual(recovered, 1)
        self.assertEqual(len(ocr_page["texts"]), 2)
        self.assertTrue(any("Your heart" in text.get("text", "") for text in ocr_page["texts"]))
        recovered_text = [text for text in ocr_page["texts"] if "Your heart" in text.get("text", "")][0]
        self.assertEqual(recovered_text["bubble_mask_source"], "image_dark_bubble_mask")
        self.assertIn("partial_dark_bubble_lobe_reocr", recovered_text["qa_flags"])

    def test_partial_dark_bubble_recovery_localizes_page_space_bboxes(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import _recover_partial_dark_bubble_ocr_from_texts
        from strip.types import Band
        import cv2
        import numpy as np

        band_top = 5760
        image = np.full((460, 820, 3), [3, 4, 6], dtype=np.uint8)
        cv2.ellipse(image, (220, 165), (190, 118), 0, 0, 360, (2, 2, 2), -1)
        cv2.ellipse(image, (220, 165), (190, 118), 0, 0, 360, (25, 118, 158), 4)
        cv2.ellipse(image, (590, 230), (245, 150), 0, 0, 360, (2, 2, 2), -1)
        cv2.ellipse(image, (590, 230), (245, 150), 0, 0, 360, (25, 118, 158), 4)
        cv2.putText(image, "Your heart felt", (110, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2, cv2.LINE_AA)
        cv2.putText(image, "empty all the time.", (88, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2, cv2.LINE_AA)
        ocr_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "You lived without planning for the future because you didn't have the will to live.",
                    "bbox": [455, band_top + 160, 760, band_top + 330],
                    "text_pixel_bbox": [455, band_top + 160, 760, band_top + 330],
                    "confidence": 0.96,
                    "confidence_raw": 0.96,
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "layout_profile": "dark_bubble",
                    "bubble_mask_bbox": [40, band_top + 20, 810, band_top + 410],
                    "qa_flags": ["dark_bubble_oval_reocr"],
                }
            ],
            "_vision_blocks": [{"bbox": [455, band_top + 160, 760, band_top + 330]}],
        }
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "text": "Your heart felt empty and lonely all the time.",
                    "bbox": [55, 78, 330, 190],
                    "source_bbox": [55, 78, 330, 190],
                    "text_pixel_bbox": [55, 78, 330, 190],
                    "confidence": 0.91,
                    "confidence_raw": 0.91,
                }
            ],
            "_vision_blocks": [{"bbox": [55, 78, 330, 190], "confidence": 0.91}],
        }
        evidence = {
            "has_inner_light_text": True,
            "inner_light_area": 1300,
            "bright_pixel_ratio": 0.07,
            "dark_pixel_ratio": 0.80,
        }
        band = Band(y_top=band_top, y_bottom=band_top + 460, balloons=[], strip_slice=image, original_slice=image.copy())

        with patch("strip.detect_balloons._inner_dark_text_evidence", return_value=evidence):
            recovered = _recover_partial_dark_bubble_ocr_from_texts(
                ocr_page,
                band=band,
                page_dict={"width": 820, "height": 460},
                band_id="page_003_band_023",
                runtime=runtime,
            )

        self.assertEqual(recovered, 1)
        self.assertTrue(any("Your heart" in text.get("text", "") for text in ocr_page["texts"]))

    def test_partial_dark_bubble_recovery_recovers_detected_candidate_without_ocr_overlap(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import _recover_partial_dark_bubble_ocr_from_texts
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        image = np.full((460, 820, 3), [3, 4, 6], dtype=np.uint8)
        cv2.ellipse(image, (220, 165), (190, 118), 0, 0, 360, (2, 2, 2), -1)
        cv2.ellipse(image, (220, 165), (190, 118), 0, 0, 360, (25, 118, 158), 4)
        cv2.ellipse(image, (590, 230), (245, 150), 0, 0, 360, (2, 2, 2), -1)
        cv2.ellipse(image, (590, 230), (245, 150), 0, 0, 360, (25, 118, 158), 4)
        cv2.putText(image, "Your heart felt", (110, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2, cv2.LINE_AA)
        cv2.putText(image, "empty all the time.", (88, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2, cv2.LINE_AA)
        ocr_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "You lived without planning for the future because you didn't have the will to live.",
                    "bbox": [455, 160, 760, 330],
                    "text_pixel_bbox": [455, 160, 760, 330],
                    "confidence": 0.96,
                    "confidence_raw": 0.96,
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["dark_bubble_oval_reocr"],
                }
            ],
            "_vision_blocks": [{"bbox": [455, 160, 760, 330]}],
        }
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "text": "Your heart felt empty and lonely all the time.",
                    "bbox": [20, 20, 210, 130],
                    "source_bbox": [20, 20, 210, 130],
                    "text_pixel_bbox": [20, 20, 210, 130],
                    "confidence": 0.91,
                    "confidence_raw": 0.91,
                }
            ],
            "_vision_blocks": [{"bbox": [20, 20, 210, 130], "confidence": 0.91}],
        }
        band = Band(
            y_top=1000,
            y_bottom=1460,
            balloons=[
                Balloon(BBox(98, 1068, 295, 1203), 0.91),
                Balloon(BBox(428, 1150, 729, 1329), 0.94),
            ],
            strip_slice=image,
            original_slice=image.copy(),
        )

        with patch("strip.detect_balloons._inner_dark_text_evidence", return_value={
            "has_inner_light_text": True,
            "inner_light_area": 1300,
            "bright_pixel_ratio": 0.10,
            "dark_pixel_ratio": 0.74,
        }), patch("strip.process_bands._detect_dark_oval_bubble_bbox", return_value=None):
            recovered = _recover_partial_dark_bubble_ocr_from_texts(
                ocr_page,
                band=band,
                page_dict={"width": 820, "height": 460},
                band_id="page_003_band_023",
                runtime=runtime,
            )

        self.assertEqual(recovered, 1)
        recovered_text = [text for text in ocr_page["texts"] if "Your heart" in text.get("text", "")][0]
        self.assertEqual(recovered_text["bubble_mask_source"], "image_dark_bubble_mask")
        self.assertIn("detected_dark_bubble_without_text_reocr", recovered_text["qa_flags"])

    def test_partial_dark_bubble_recovery_tries_negative_crop_for_empty_detected_balloon(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import _recover_partial_dark_bubble_ocr_from_texts
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        image = np.full((220, 420, 3), [4, 5, 7], dtype=np.uint8)
        cv2.ellipse(image, (210, 112), (118, 70), 0, 0, 360, (1, 1, 1), -1)
        cv2.ellipse(image, (210, 112), (118, 70), 0, 0, 360, (25, 118, 158), 4)
        cv2.putText(image, "Wake up", (150, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 250, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "host.", (176, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 250, 255), 2, cv2.LINE_AA)
        band = Band(
            y_top=3000,
            y_bottom=3220,
            balloons=[Balloon(strip_bbox=BBox(92, 3042, 328, 3182), confidence=0.91)],
            strip_slice=image,
            original_slice=image.copy(),
        )
        ocr_page = {"texts": [], "_vision_blocks": []}
        runtime = MagicMock()
        runtime.run_ocr_stage.side_effect = [
            {"texts": [], "_vision_blocks": []},
            {
                "texts": [
                    {
                        "id": "ocr_negative_001",
                        "text": "Wake up host.",
                        "bbox": [62, 50, 175, 103],
                        "source_bbox": [62, 50, 175, 103],
                        "text_pixel_bbox": [62, 50, 175, 103],
                        "bubble_mask_bbox": [62, 50, 175, 103],
                        "confidence": 0.94,
                        "confidence_raw": 0.94,
                    }
                ],
                "_vision_blocks": [{"bbox": [62, 50, 175, 103], "confidence": 0.94}],
            },
        ]
        evidence = {
            "has_inner_light_text": True,
            "inner_light_area": 520,
            "bright_pixel_ratio": 0.09,
            "dark_pixel_ratio": 0.79,
        }

        with patch("strip.detect_balloons._inner_dark_text_evidence", return_value=evidence), \
            patch("strip.process_bands._detect_dark_oval_bubble_bbox", return_value=[92, 42, 328, 182]), \
            patch("strip.process_bands._run_direct_paddle_candidate_crop_reocr", return_value={"texts": [], "_vision_blocks": []}):
            recovered = _recover_partial_dark_bubble_ocr_from_texts(
                ocr_page,
                band=band,
                page_dict={"width": 420, "height": 220},
                band_id="page_003_band_027",
                runtime=runtime,
            )

        self.assertEqual(recovered, 1)
        self.assertEqual(runtime.run_ocr_stage.call_count, 2)
        self.assertGreater(float(np.mean(runtime.run_ocr_stage.call_args_list[1].args[0])), 180.0)
        text = ocr_page["texts"][0]
        self.assertEqual(text["text"], "Wake up host.")
        self.assertEqual(text["bubble_mask_source"], "image_dark_bubble_mask")
        self.assertEqual(text["block_profile"], "dark_bubble")
        self.assertEqual(text["layout_profile"], "dark_bubble")
        self.assertEqual(text["bubble_mask_bbox"], [92, 42, 328, 182])
        self.assertEqual(text["balloon_bbox"], [92, 42, 328, 182])
        self.assertIn("detected_dark_bubble_without_text_reocr", text["qa_flags"])

    def test_partial_dark_bubble_recovery_recovers_candidate_when_light_flag_misses_area(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import _recover_partial_dark_bubble_ocr_from_texts
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        image = np.full((418, 800, 3), [3, 4, 6], dtype=np.uint8)
        cv2.ellipse(image, (290, 190), (235, 175), 0, 0, 360, (2, 2, 2), -1)
        cv2.ellipse(image, (290, 190), (235, 175), 0, 0, 360, (25, 118, 158), 4)
        cv2.ellipse(image, (590, 285), (170, 92), 0, 0, 360, (2, 2, 2), -1)
        cv2.ellipse(image, (590, 285), (170, 92), 0, 0, 360, (25, 118, 158), 4)
        cv2.putText(image, "I am called 'System'.", (480, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2, cv2.LINE_AA)
        ocr_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "I'm a system that guides the host of King Yeomra in establishing an underworld. Ian",
                    "bbox": [158, 112, 526, 319],
                    "text_pixel_bbox": [158, 112, 526, 319],
                    "confidence": 0.936,
                    "confidence_raw": 0.936,
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "layout_profile": "dark_bubble",
                    "bubble_mask_bbox": [53, 14, 527, 418],
                    "qa_flags": ["dark_bubble_oval_reocr"],
                }
            ],
            "_vision_blocks": [{"bbox": [158, 112, 526, 319]}],
        }
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {"texts": [], "_vision_blocks": []}
        direct_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "I am called 'System'.",
                    "bbox": [18, 10, 240, 50],
                    "source_bbox": [18, 10, 240, 50],
                    "text_pixel_bbox": [18, 10, 240, 50],
                    "confidence": 0.95,
                    "confidence_raw": 0.95,
                    "qa_flags": ["candidate_crop_direct_paddle_reocr"],
                }
            ],
            "_vision_blocks": [{"bbox": [18, 10, 240, 50], "confidence": 0.95}],
            "width": 255,
            "height": 58,
        }
        band = Band(
            y_top=10282,
            y_bottom=10700,
            balloons=[
                Balloon(BBox(144, 10378, 399, 10540), 0.91),
                Balloon(BBox(462, 10546, 717, 10604), 0.73),
            ],
            strip_slice=image,
            original_slice=image.copy(),
        )
        evidence = {
            "has_inner_dark_text": True,
            "has_inner_light_text": False,
            "inner_dark_component_count": 6,
            "inner_dark_area": 108,
            "inner_light_component_count": 1,
            "inner_light_area": 374,
            "bright_pixel_ratio": 0.1274,
            "dark_pixel_ratio": 0.7879,
        }

        with patch("strip.detect_balloons._inner_dark_text_evidence", return_value=evidence), \
            patch("strip.process_bands._detect_dark_oval_bubble_bbox", return_value=[462, 264, 717, 322]), \
            patch("strip.process_bands._run_direct_paddle_candidate_crop_reocr", return_value=direct_page) as direct_ocr:
            recovered = _recover_partial_dark_bubble_ocr_from_texts(
                ocr_page,
                band=band,
                page_dict={"width": 800, "height": 418},
                band_id="page_002_band_011",
                runtime=runtime,
            )

        self.assertEqual(recovered, 1)
        self.assertEqual(direct_ocr.call_count, 1)
        self.assertNotIn("Ian", ocr_page["texts"][0]["text"])
        recovered_text = [text for text in ocr_page["texts"] if "System" in text.get("text", "") and text.get("text", "").startswith("I am")][0]
        self.assertEqual(recovered_text["bubble_mask_source"], "image_dark_bubble_mask")
        self.assertIn("detected_dark_bubble_without_text_reocr", recovered_text["qa_flags"])
        self.assertLessEqual(recovered_text["balloon_bbox"][0], 422)
        self.assertGreaterEqual(recovered_text["balloon_bbox"][2], 760)
        self.assertLessEqual(recovered_text["bubble_mask_bbox"][0], 422)
        self.assertGreaterEqual(recovered_text["bubble_mask_bbox"][2], 760)
        recovered_regions = [
            region for region in ocr_page.get("_bubble_regions", [])
            if region.get("source") == "candidate_crop_reocr_dark_bubble"
        ]
        self.assertTrue(recovered_regions)
        self.assertLessEqual(recovered_regions[0]["bubble_mask_bbox"][0], 422)
        self.assertGreaterEqual(recovered_regions[0]["bubble_mask_bbox"][2], 760)

    def test_dark_bubble_evidence_accepts_stylized_light_text_signal(self):
        from strip.process_bands import _dark_bubble_evidence_supports_lobe_reocr

        evidence = {
            "has_inner_light_text": False,
            "inner_light_area": 0,
            "significant_component_count": 2,
            "significant_area": 222,
            "bright_pixel_ratio": 0.1523,
            "dark_pixel_ratio": 0.7716,
        }

        self.assertTrue(
            _dark_bubble_evidence_supports_lobe_reocr(
                evidence,
                min_dark_ratio=0.45,
                max_bright_ratio=0.24,
                min_light_area=180,
            )
        )

    def test_process_band_recovers_empty_dark_oval_bubble_with_direct_reocr(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import process_band
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((260, 420, 3), [3, 4, 6], dtype=np.uint8)
        cv2.ellipse(slice_img, (210, 132), (130, 82), 0, 0, 360, (1, 1, 1), -1)
        cv2.ellipse(slice_img, (210, 132), (130, 82), 0, 0, 360, (25, 118, 158), 4)
        cv2.putText(
            slice_img,
            "That means",
            (128, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 250, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            slice_img,
            "you're talented!",
            (112, 154),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 250, 255),
            2,
            cv2.LINE_AA,
        )
        band = Band(
            y_top=5000,
            y_bottom=5260,
            balloons=[Balloon(strip_bbox=BBox(80, 5048, 340, 5216), confidence=0.88)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )
        runtime = MagicMock()
        runtime.run_ocr_stage.side_effect = [
            {"texts": [], "_vision_blocks": [], "_ocr_stats": {"quick_skipped_no_text": True}},
            {"texts": [], "_vision_blocks": []},
        ]
        direct_page = {
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "text": "That means you're talented!",
                    "bbox": [0, 0, 300, 236],
                    "source_bbox": [56, 70, 244, 140],
                    "text_pixel_bbox": [56, 70, 244, 140],
                    "line_polygons": [[[56, 70], [244, 70], [244, 140], [56, 140]]],
                    "confidence": 0.94,
                    "confidence_raw": 0.94,
                    "qa_flags": ["candidate_crop_direct_paddle_reocr"],
                }
            ],
            "_vision_blocks": [{"bbox": [56, 70, 244, 140], "confidence": 0.94}],
            "width": 300,
            "height": 236,
        }
        light_text_evidence = {
            "has_inner_dark_text": False,
            "has_inner_light_text": True,
            "inner_dark_component_count": 0,
            "inner_dark_area": 0,
            "inner_light_component_count": 6,
            "inner_light_area": 620,
            "significant_component_count": 0,
            "significant_area": 0,
            "bright_pixel_ratio": 0.071,
            "dark_pixel_ratio": 0.78,
        }
        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **_kw: [
            {
                **pages[0],
                "texts": [
                    {**text, "translated": "ISSO SIGNIFICA QUE VOCE E TALENTOSO!"}
                    for text in pages[0].get("texts", [])
                ],
            }
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _page: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _page: img.copy()

        with patch("strip.detect_balloons._inner_dark_text_evidence", return_value=light_text_evidence), \
            patch("strip.process_bands._run_direct_paddle_candidate_crop_reocr", return_value=direct_page) as direct_ocr, \
            patch("ocr.contextual_reviewer.contextual_review_page", side_effect=lambda page, *_args: page), \
            patch("layout.balloon_layout.enrich_page_layout", side_effect=lambda page: page):
            process_band(
                band,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
                page_idx=3,
                source_page_number=3,
            )

        self.assertEqual(direct_ocr.call_count, 1)
        text = band.ocr_result["texts"][0]
        self.assertEqual(text["text"], "That means you're talented!")
        self.assertEqual(text["bubble_mask_source"], "image_dark_bubble_mask")
        self.assertEqual(text["block_profile"], "dark_bubble")
        self.assertEqual(text["layout_profile"], "dark_bubble")
        self.assertEqual(band.ocr_result["_perf"]["ocr_candidate_crop_recovered"], 1)

    def test_process_band_does_not_probe_dark_art_without_rect_panel_frame(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import process_band
        from strip.types import Band, Balloon, BBox
        import cv2
        import numpy as np

        slice_img = np.full((220, 420, 3), [7, 9, 12], dtype=np.uint8)
        cv2.circle(slice_img, (250, 116), 72, (44, 44, 52), -1)
        cv2.line(slice_img, (160, 70), (335, 170), (190, 190, 205), 3)
        band = Band(
            y_top=1200,
            y_bottom=1420,
            balloons=[Balloon(strip_bbox=BBox(150, 1260, 350, 1360), confidence=0.82)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [],
            "_vision_blocks": [],
            "_ocr_stats": {"quick_skipped_no_text": True},
        }
        weak_dark_evidence = {
            "has_inner_dark_text": True,
            "has_inner_light_text": False,
            "inner_dark_component_count": 5,
            "inner_dark_area": 160,
            "inner_light_component_count": 0,
            "inner_light_area": 0,
            "bright_pixel_ratio": 0.12,
            "dark_pixel_ratio": 0.72,
        }

        with patch("strip.detect_balloons._inner_dark_text_evidence", return_value=weak_dark_evidence), \
            patch("strip.process_bands._run_direct_paddle_candidate_crop_reocr", return_value={"texts": []}) as direct_ocr:
            process_band(
                band,
                runtime=runtime,
                translator=MagicMock(),
                inpainter=MagicMock(),
                typesetter=MagicMock(),
                page_idx=4,
                source_page_number=2,
            )

        self.assertEqual(runtime.run_ocr_stage.call_count, 1)
        self.assertEqual(direct_ocr.call_count, 0)
        self.assertEqual(band.ocr_result.get("texts"), [])
        self.assertEqual(band.ocr_result.get("_perf", {}).get("ocr_candidate_crop_recovered"), 0)

    def test_process_band_notifies_ordered_context_after_translate_before_inpaint(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [
                {"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"},
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {
                "texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}],
                "_glossary_additions": {"FENRIS": "Fenris"},
            }
        ]
        events = []

        def on_language_ready(page):
            events.append(("callback", page["texts"][0]["translated"], dict(page.get("_glossary_additions") or {})))
            page["texts"][0]["translated"] = "MUTATED"

        def fake_inpaint(_image, page):
            events.append(("inpaint", page["texts"][0]["translated"]))
            return np.full((100, 300, 3), 255, dtype=np.uint8)

        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = fake_inpaint
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
            ordered_context_after_translate_callback=on_language_ready,
        )

        self.assertEqual(events[0], ("callback", "OLA", {"FENRIS": "Fenris"}))
        self.assertEqual(events[1], ("inpaint", "OLA"))

    def test_process_band_serializes_gpu_stages_when_lock_is_provided(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        class TrackingLock:
            def __init__(self):
                self.events = []

            def __enter__(self):
                self.events.append("lock_enter")

            def __exit__(self, _exc_type, _exc, _tb):
                self.events.append("lock_exit")

        band = self._make_band()
        runtime = MagicMock()

        def fake_ocr(_image, _page):
            lock.events.append("ocr")
            return {
                "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"}],
                "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
            }

        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}]}
        ]

        def fake_inpaint(_image, page):
            lock.events.append("inpaint")
            return np.full((100, 300, 3), 255, dtype=np.uint8)

        lock = TrackingLock()
        runtime.run_ocr_stage.side_effect = fake_ocr
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = fake_inpaint
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
            gpu_stage_lock=lock,
        )

        self.assertEqual(
            lock.events,
            ["lock_enter", "ocr", "lock_exit", "lock_enter", "inpaint", "lock_exit"],
        )

    def test_process_band_can_use_separate_ocr_and_inpaint_locks(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        class TrackingLock:
            def __init__(self, label):
                self.label = label
                self.events = []

            def __enter__(self):
                self.events.append(f"{self.label}_enter")

            def __exit__(self, _exc_type, _exc, _tb):
                self.events.append(f"{self.label}_exit")

        band = self._make_band()
        runtime = MagicMock()
        ocr_lock = TrackingLock("ocr_lock")
        inpaint_lock = TrackingLock("inpaint_lock")
        legacy_gpu_lock = TrackingLock("legacy_gpu_lock")

        def fake_ocr(_image, _page):
            ocr_lock.events.append("ocr")
            return {
                "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"}],
                "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
            }

        runtime.run_ocr_stage.side_effect = fake_ocr
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}]}
        ]

        def fake_inpaint(_image, _page):
            inpaint_lock.events.append("inpaint")
            return np.full((100, 300, 3), 255, dtype=np.uint8)

        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = fake_inpaint
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
            gpu_stage_lock=legacy_gpu_lock,
            ocr_stage_lock=ocr_lock,
            inpaint_stage_lock=inpaint_lock,
        )

        self.assertEqual(ocr_lock.events, ["ocr_lock_enter", "ocr", "ocr_lock_exit"])
        self.assertEqual(
            inpaint_lock.events,
            ["inpaint_lock_enter", "inpaint", "inpaint_lock_exit"],
        )
        self.assertEqual(legacy_gpu_lock.events, [])

    def test_process_band_serializes_typeset_when_lock_is_provided(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        class TrackingLock:
            def __init__(self):
                self.events = []

            def __enter__(self):
                self.events.append("typeset_lock_enter")

            def __exit__(self, _exc_type, _exc, _tb):
                self.events.append("typeset_lock_exit")

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}]}
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.return_value = np.full((100, 300, 3), 255, dtype=np.uint8)

        lock = TrackingLock()

        def fake_typeset(_image, _page):
            lock.events.append("typeset")
            return np.full((100, 300, 3), 100, dtype=np.uint8)

        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = fake_typeset

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
            typeset_stage_lock=lock,
        )

        self.assertEqual(lock.events, ["typeset_lock_enter", "typeset", "typeset_lock_exit"])

    def test_process_band_records_wait_and_compute_for_locked_stages(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np
        import time

        class SlowEnterLock:
            def __enter__(self):
                time.sleep(0.002)

            def __exit__(self, _exc_type, _exc, _tb):
                return False

        band = self._make_band()
        runtime = MagicMock()

        def fake_ocr(_image, _page):
            time.sleep(0.002)
            return {
                "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"}],
                "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
            }

        runtime.run_ocr_stage.side_effect = fake_ocr
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}]}
        ]

        inpainter = MagicMock()

        def fake_inpaint(_image, _page):
            time.sleep(0.002)
            return np.full((100, 300, 3), 255, dtype=np.uint8)

        inpainter.inpaint_band_image.side_effect = fake_inpaint
        typesetter = MagicMock()

        def fake_typeset(_image, _page):
            time.sleep(0.002)
            return np.full((100, 300, 3), 100, dtype=np.uint8)

        typesetter.render_band_image.side_effect = fake_typeset

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
            gpu_stage_lock=SlowEnterLock(),
            typeset_stage_lock=SlowEnterLock(),
        )

        durations = band.perf["durations_sec"]
        for stage in ("ocr", "inpaint", "typeset"):
            self.assertGreater(durations[f"{stage}_compute"], 0)
            self.assertGreater(durations[f"{stage}_wait"], 0)
            self.assertAlmostEqual(
                durations[stage],
                durations[f"{stage}_wait"] + durations[f"{stage}_compute"],
                delta=0.01,
            )

    def test_process_band_restores_ocr_metadata_when_translation_payload_is_reduced(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [40, 10, 160, 90],
                    "text_pixel_bbox": [62, 28, 140, 72],
                    "line_polygons": [[[62, 28], [140, 28], [140, 72], [62, 72]]],
                    "text": "HELLO",
                    "tipo": "fala",
                    "ocr_source": "paddleocr",
                    "ocr_confidence": 0.91,
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {
                "texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}],
                "_vision_blocks": [],
            }
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.return_value = np.full((100, 300, 3), 255, dtype=np.uint8)
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        inpaint_page = inpainter.inpaint_band_image.call_args[0][1]
        self.assertEqual(inpaint_page["texts"][0]["text_pixel_bbox"], [62, 28, 140, 72])
        self.assertEqual(inpaint_page["texts"][0]["ocr_source"], "paddleocr")
        self.assertEqual(inpaint_page["texts"][0]["bbox"], [50, 20, 150, 80])
        self.assertEqual(inpaint_page["_vision_blocks"][0]["bbox"], [40, 10, 160, 90])

        self.assertEqual(band.ocr_result["texts"][0]["translated"], "OLA")
        self.assertEqual(band.ocr_result["texts"][0]["line_polygons"][0][0], [62, 28])

    def test_process_band_forwards_translation_runtime_options(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {"texts": [
            {"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"},
        ]}
        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **_kw: pages
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _: img.copy()
        translation_context = {"memory": [{"source": "HELLO", "target": "OLA"}]}

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
            models_dir="D:/traduzai_data/models",
            ollama_host="http://127.0.0.1:11435",
            ollama_model="custom-translator",
            translation_context=translation_context,
        )

        kwargs = translator.translate_pages.call_args.kwargs
        self.assertEqual(kwargs["models_dir"], "D:/traduzai_data/models")
        self.assertEqual(kwargs["ollama_host"], "http://127.0.0.1:11435")
        self.assertEqual(kwargs["ollama_model"], "custom-translator")
        self.assertIs(kwargs["translation_context"], translation_context)

    def test_process_band_applies_smart_skip_shadow_only_when_flag_is_enabled(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "credit",
                    "bbox": [50, 20, 150, 40],
                    "text": "FOR FASTER UPDATE",
                    "confidence": 0.0,
                    "tipo": "fala",
                    "skip_processing": False,
                },
                {
                    "id": "dialogue",
                    "bbox": [50, 50, 160, 80],
                    "text": "IS THIS RECORDING?",
                    "confidence": 0.9,
                    "tipo": "fala",
                    "skip_processing": False,
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 170, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **_kw: pages
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _page: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _page: img.copy()

        with patch.dict("os.environ", {"TRADUZAI_SMART_SKIP_SHADOW": "1"}):
            process_band(
                band,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
                page_idx=0,
            )

        translated_input = translator.translate_pages.call_args.args[0][0]
        self.assertEqual(translated_input["_smart_skip_shadow"]["candidate_count"], 1)
        self.assertFalse(translated_input["texts"][0]["skip_processing"])
        self.assertFalse(translated_input["texts"][1]["skip_processing"])
        self.assertEqual(band.perf["smart_skip_shadow_candidate_count"], 1)
        self.assertEqual(band.perf["smart_skip_shadow_not_safe_count"], 1)

    def test_process_band_passes_source_page_number_to_ocr_runtime(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {"texts": []}

        process_band(
            band,
            runtime=runtime,
            translator=MagicMock(),
            inpainter=MagicMock(),
            typesetter=MagicMock(),
            page_idx=9,
            source_page_number=2,
        )

        page_dict = runtime.run_ocr_stage.call_args.args[1]
        self.assertEqual(page_dict["numero"], 2)
        self.assertEqual(page_dict["_source_page_number"], 2)
        self.assertEqual(page_dict["_band_index"], 10)

    def test_process_band_uses_precomputed_ocr_page_without_runtime_call(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.side_effect = AssertionError("runtime OCR should be skipped")
        precomputed_ocr_page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [40, 10, 160, 90],
                    "text": "HELLO",
                    "tipo": "fala",
                    "confidence": 0.93,
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
            "_ocr_stats": {
                "full_page_mapped": 1,
                "macro_ocr_real": True,
                "macro_window_count": 1,
                "macro_ocr_block_count": 1,
                "macro_ocr_empty_record_count": 0,
            },
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}]}
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.return_value = np.full((100, 300, 3), 255, dtype=np.uint8)
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=4,
            source_page_number=2,
            precomputed_ocr_page=precomputed_ocr_page,
        )

        self.assertIs(result, band)
        runtime.run_ocr_stage.assert_not_called()
        translator.translate_pages.assert_called_once()
        self.assertEqual(translator.translate_pages.call_args.args[0][0]["numero"], 2)
        self.assertEqual(translator.translate_pages.call_args.args[0][0]["width"], 300)
        self.assertEqual(translator.translate_pages.call_args.args[0][0]["height"], 100)
        self.assertEqual(band.ocr_result["texts"][0]["translated"], "OLA")
        self.assertTrue(band.perf["ocr_precomputed_page"])
        self.assertTrue(band.ocr_result["_perf"]["ocr_precomputed_page"])
        self.assertEqual(band.perf["ocr_full_page_mapped"], 1)
        self.assertTrue(band.perf["ocr_macro_ocr_real"])
        self.assertEqual(band.perf["ocr_macro_window_count"], 1)
        self.assertEqual(band.perf["ocr_macro_ocr_block_count"], 1)
        self.assertEqual(band.perf["ocr_macro_ocr_empty_record_count"], 0)

    def test_ocr_stage_result_is_snapshot_and_skips_runtime_for_precomputed_page(self):
        from unittest.mock import MagicMock
        from strip import process_bands

        band = self._make_band()
        page_dict = process_bands._band_to_page_dict(band, page_idx=3, source_page_number=2)
        precomputed_ocr_page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "text": "HELLO",
                    "tipo": "fala",
                }
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        runtime = MagicMock()
        runtime.run_ocr_stage.side_effect = AssertionError("runtime OCR should be skipped")

        output = process_bands._run_band_ocr_stage(
            band,
            runtime=runtime,
            page_dict=page_dict,
            precomputed_ocr_page=precomputed_ocr_page,
        )

        runtime.run_ocr_stage.assert_not_called()
        self.assertEqual(output.stage_id, "ocr")
        self.assertEqual(dict(output.perf_updates), {
            "ocr_precomputed_page": True,
            "ocr_runtime_skipped": True,
        })

        page_snapshot = output.to_page_dict()
        self.assertEqual(page_snapshot["numero"], 2)
        self.assertEqual(page_snapshot["_band_index"], 4)
        page_snapshot["texts"][0]["text"] = "MUTATED"
        precomputed_ocr_page["texts"][0]["text"] = "SOURCE MUTATED"

        self.assertEqual(output.to_page_dict()["texts"][0]["text"], "HELLO")

    def test_precomputed_ocr_page_deep_copies_negative_evidence(self):
        from strip import process_bands

        band = self._make_band()
        page_dict = process_bands._band_to_page_dict(band, page_idx=3, source_page_number=2)
        precomputed_ocr_page = {
            "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO"}],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
            "_negative_evidence": {
                "source": "negative_detect_ocr",
                "texts": [{"id": "neg", "text": "SHADOW"}],
                "blocks": [{"bbox": [44, 12, 164, 92]}],
                "eligible_for_promotion": False,
            },
        }

        prepared = process_bands._prepare_precomputed_ocr_page(precomputed_ocr_page, page_dict)
        precomputed_ocr_page["_negative_evidence"]["texts"][0]["text"] = "MUTATED"

        self.assertEqual(prepared["_negative_evidence"]["texts"][0]["text"], "SHADOW")
        self.assertFalse(prepared["_negative_evidence"]["eligible_for_promotion"])

    def test_merge_translated_page_metadata_preserves_negative_evidence(self):
        from strip import process_bands

        ocr_page = {
            "texts": [{"id": "t1", "text": "HELLO"}],
            "_vision_blocks": [{"bbox": [10, 10, 90, 40]}],
            "_negative_evidence": {
                "source": "negative_detect_ocr",
                "texts": [{"id": "neg", "text": "SHADOW"}],
                "blocks": [{"bbox": [20, 12, 100, 48]}],
                "eligible_for_promotion": False,
            },
        }
        translated_page = {"texts": [{"id": "t1", "translated": "OLA"}]}

        merged = process_bands._merge_translated_page_metadata(ocr_page, translated_page)

        self.assertEqual(merged["_negative_evidence"]["texts"][0]["text"], "SHADOW")
        self.assertEqual(merged["texts"][0]["translated"], "OLA")

    def test_negative_dark_bubble_candidate_promotes_when_normal_ocr_misses_white_text(self):
        from strip import process_bands
        import cv2
        import numpy as np

        image = np.zeros((180, 260, 3), dtype=np.uint8)
        cv2.ellipse(image, (130, 86), (82, 46), 0, 0, 360, (30, 135, 165), 3)
        cv2.putText(image, "SYSTEM", (82, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
        page = {"texts": [], "_vision_blocks": []}
        evidence = {
            "source": "negative_detect_ocr",
            "texts": [{"id": "neg", "text": "SYSTEM", "bbox": [78, 68, 182, 100], "confidence": 0.92}],
            "blocks": [{"bbox": [78, 68, 182, 100], "confidence": 0.92}],
            "eligible_for_promotion": False,
        }

        promoted = process_bands.fuse_negative_dark_bubble_candidates(page, evidence, image)

        self.assertEqual(promoted, 1)
        self.assertEqual(page["texts"][0]["ocr_source"], "negative_detect_ocr_promoted")
        self.assertEqual(page["texts"][0]["bubble_mask_source"], "image_dark_bubble_mask")
        self.assertIn("negative_pass_promoted", page["texts"][0]["qa_flags"])
        self.assertEqual(page["_vision_blocks"][0]["detector"], "negative_detect_ocr_promoted")

    def test_negative_candidate_is_attached_as_evidence_when_duplicate_of_normal(self):
        from strip import process_bands
        import numpy as np

        image = np.zeros((120, 220, 3), dtype=np.uint8)
        page = {
            "texts": [{"id": "ocr_001", "text": "SYSTEM", "bbox": [50, 40, 140, 70]}],
            "_vision_blocks": [{"bbox": [50, 40, 140, 70]}],
        }
        evidence = {
            "texts": [{"id": "neg", "text": "SYSTEM", "bbox": [52, 41, 142, 71], "confidence": 0.91}],
            "blocks": [{"bbox": [52, 41, 142, 71], "confidence": 0.91}],
        }

        promoted = process_bands.fuse_negative_dark_bubble_candidates(page, evidence, image)

        self.assertEqual(promoted, 0)
        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["qa_metrics"]["negative_evidence"][0]["text"], "SYSTEM")

    def test_negative_duplicate_promotes_stale_white_mask_to_dark_bubble_contract(self):
        from strip import process_bands
        import cv2
        import numpy as np

        image = np.zeros((180, 260, 3), dtype=np.uint8)
        cv2.ellipse(image, (130, 86), (82, 46), 0, 0, 360, (30, 135, 165), 3)
        cv2.putText(image, "SYSTEM", (82, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "SYSTEM",
                    "bbox": [78, 68, 182, 100],
                    "text_pixel_bbox": [78, 68, 182, 100],
                    "bubble_mask_source": "image_white_bubble_mask",
                    "bubble_mask_bbox": [95, 72, 170, 105],
                    "balloon_bbox": [95, 72, 170, 105],
                    "background_rgb": [12, 12, 12],
                }
            ],
            "_vision_blocks": [{"bbox": [78, 68, 182, 100]}],
        }
        evidence = {
            "texts": [{"id": "neg", "text": "SYSTEM", "bbox": [78, 68, 182, 100], "confidence": 0.92}],
            "blocks": [{"bbox": [78, 68, 182, 100], "confidence": 0.92}],
        }

        promoted = process_bands.fuse_negative_dark_bubble_candidates(page, evidence, image)

        self.assertEqual(promoted, 0)
        text = page["texts"][0]
        self.assertEqual(text["bubble_mask_source"], "image_dark_bubble_mask")
        self.assertIn("image_dark_bubble_mask", text["qa_metrics"])
        self.assertIn("dark_bubble_duplicate_contract_promoted", text["qa_flags"])
        self.assertGreater(text["bubble_mask_bbox"][2] - text["bubble_mask_bbox"][0], 120)

    def test_negative_duplicate_with_full_note_promotes_existing_text_and_bbox(self):
        from strip import process_bands
        import numpy as np

        image = np.zeros((140, 260, 3), dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "ocr_002",
                    "text": "BUDDHIST DEITY OF DEATH.",
                    "bbox": [128, 72, 205, 88],
                    "text_pixel_bbox": [128, 72, 205, 88],
                }
            ],
            "_vision_blocks": [{"bbox": [128, 72, 205, 88]}],
        }
        full_note = "T/N: KING YEOMRA OR YAMA IS A HINDU AND BUDDHIST DEITY OF DEATH."
        evidence = {
            "texts": [{"id": "neg", "text": full_note, "bbox": [116, 34, 232, 90], "confidence": 0.91}],
            "blocks": [{"bbox": [116, 34, 232, 90], "confidence": 0.91}],
        }

        promoted = process_bands.fuse_negative_dark_bubble_candidates(page, evidence, image)

        self.assertEqual(promoted, 0)
        self.assertEqual(len(page["texts"]), 1)
        text = page["texts"][0]
        self.assertEqual(text["text"], full_note)
        self.assertEqual(text["text_pixel_bbox"], [76, 20, 280, 108])
        self.assertEqual(text["line_polygons"][0], [[76, 20], [280, 20], [280, 108], [76, 108]])
        self.assertIn("negative_pass_attached_promoted", text["qa_flags"])
        self.assertEqual(text["qa_metrics"]["negative_evidence"][0]["text"], full_note)

    def test_dark_connected_lobe_full_crop_rejects_cross_lobe_replacement(self):
        from strip import process_bands

        text = {
            "text": "You were loyal to others, but to them, you were being nosy.",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bubble_mask_bbox": [63, 4962, 445, 5393],
            "qa_flags": [
                "dark_bubble_connected_lobes_promoted",
                "dark_bubble_lobe_mask_bbox_preferred",
            ],
        }
        candidate = {
            "text": "You were loyal to others, but to them, you were being nosy. You the king",
            "text_pixel_bbox": [132, 5078, 557, 5287],
            "line_polygons": [
                [[174, 5076], [384, 5078], [383, 5115], [173, 5113]],
                [[131, 5118], [426, 5122], [425, 5158], [130, 5153]],
                [[176, 5160], [378, 5163], [377, 5200], [175, 5197]],
                [[476, 5215], [650, 5215], [650, 5323], [476, 5323]],
            ],
        }

        rejected = process_bands._dark_connected_lobe_full_crop_candidate_crosses_lobe(
            text,
            candidate,
            bubble_bbox=text["bubble_mask_bbox"],
            old_text=text["text"],
        )

        self.assertTrue(rejected)

    def test_dark_bubble_full_crop_ocr_does_not_accept_prefix_plus_sibling_tail(self):
        from strip import process_bands

        old_text = "You were loyal to others, but to them, you were being nosy."
        new_text = old_text + " You the king"

        accepted = process_bands._dark_bubble_full_crop_ocr_is_better(old_text, new_text)

        self.assertFalse(accepted)

    def test_negative_duplicate_rejects_broad_prefix_contamination_for_dark_lobe(self):
        from strip import process_bands
        import numpy as np

        image = np.zeros((360, 620, 3), dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "You were loyal to others, but to them, you were being nosy.",
                    "bbox": [132, 115, 426, 239],
                    "text_pixel_bbox": [132, 115, 426, 239],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": ["dark_bubble_connected_lobes_promoted"],
                }
            ],
            "_vision_blocks": [{"bbox": [132, 115, 426, 239]}],
        }
        evidence = {
            "texts": [
                {
                    "id": "negative_dark_000",
                    "text": "You were loyal to others, but to them, you were being nosy. You the king",
                    "bbox": [0, 22, 558, 332],
                    "text_pixel_bbox": [132, 116, 557, 325],
                    "confidence": 0.91,
                }
            ],
            "blocks": [{"bbox": [0, 22, 558, 332], "confidence": 0.91}],
        }

        promoted = process_bands.fuse_negative_dark_bubble_candidates(page, evidence, image)

        self.assertEqual(promoted, 0)
        text = page["texts"][0]
        self.assertEqual(text["text"], "You were loyal to others, but to them, you were being nosy.")
        self.assertEqual(text["text_pixel_bbox"], [132, 115, 426, 239])
        self.assertIn("negative_broad_prefix_candidate_rejected", text["qa_flags"])
        self.assertEqual(text["qa_metrics"]["negative_evidence"][0]["text"].endswith("You the king"), True)

    def test_negative_dark_candidates_process_tight_lobe_before_broad_contamination(self):
        from strip import process_bands
        import numpy as np

        image = np.zeros((360, 620, 3), dtype=np.uint8)
        page = {"texts": [], "_vision_blocks": []}
        clean = "You were loyal to others, but to them, you were being nosy."
        broad = clean + " You the king"
        evidence = {
            "texts": [
                {
                    "id": "negative_dark_000",
                    "text": broad,
                    "bbox": [0, 22, 558, 332],
                    "text_pixel_bbox": [132, 116, 557, 325],
                    "source_bbox": [132, 115, 426, 239],
                    "confidence": 0.96,
                },
                {
                    "id": "negative_dark_001",
                    "text": clean,
                    "bbox": [132, 115, 426, 239],
                    "text_pixel_bbox": [132, 115, 426, 239],
                    "source_bbox": [132, 115, 426, 239],
                    "confidence": 0.96,
                },
            ],
            "blocks": [
                {"bbox": [0, 22, 558, 332], "confidence": 0.96},
                {"bbox": [132, 115, 426, 239], "confidence": 0.96},
            ],
        }

        promoted = process_bands.fuse_negative_dark_bubble_candidates(page, evidence, image)

        self.assertEqual(promoted, 1)
        self.assertEqual(len(page["texts"]), 1)
        self.assertEqual(page["texts"][0]["text"], clean)
        self.assertIn("negative_broad_prefix_candidate_rejected", page["texts"][0]["qa_flags"])

    def test_negative_candidate_rejects_low_confidence_partial_edge_noise(self):
        from strip import process_bands
        import numpy as np

        image = np.zeros((160, 320, 3), dtype=np.uint8)
        page = {"texts": [], "_vision_blocks": []}
        evidence = {
            "texts": [
                {
                    "id": "negative_dark_000",
                    "text": "Wituot woriu.",
                    "bbox": [148, -5, 298, 10],
                    "text_pixel_bbox": [148, -5, 298, 10],
                    "confidence": 0.65,
                    "line_polygons": [],
                }
            ],
            "blocks": [{"bbox": [148, -5, 298, 10], "confidence": 0.65}],
        }

        promoted = process_bands.fuse_negative_dark_bubble_candidates(page, evidence, image)

        self.assertEqual(promoted, 0)
        self.assertEqual(page["texts"], [])

    def test_negative_candidate_rejected_for_light_balloon(self):
        from strip import process_bands
        import numpy as np

        image = np.full((120, 220, 3), 245, dtype=np.uint8)
        page = {"texts": [], "_vision_blocks": []}
        evidence = {
            "texts": [{"id": "neg", "text": "SYSTEM", "bbox": [52, 41, 142, 71], "confidence": 0.91}],
            "blocks": [{"bbox": [52, 41, 142, 71], "confidence": 0.91}],
        }

        promoted = process_bands.fuse_negative_dark_bubble_candidates(page, evidence, image)

        self.assertEqual(promoted, 0)
        self.assertEqual(page["texts"], [])

    def test_negative_candidate_rejected_for_sfx_or_suppressed_route(self):
        from strip import process_bands
        import numpy as np

        image = np.zeros((120, 220, 3), dtype=np.uint8)
        page = {"texts": [], "_vision_blocks": []}
        evidence = {
            "texts": [
                {
                    "id": "neg",
                    "text": "KICK",
                    "bbox": [52, 41, 142, 71],
                    "confidence": 0.91,
                    "content_class": "sfx",
                }
            ],
            "blocks": [{"bbox": [52, 41, 142, 71], "confidence": 0.91}],
        }

        promoted = process_bands.fuse_negative_dark_bubble_candidates(page, evidence, image)

        self.assertEqual(promoted, 0)
        self.assertEqual(page["texts"], [])

    def test_negative_candidate_rejects_scanlation_credit_variants(self):
        from strip import process_bands
        import numpy as np

        image = np.zeros((150, 260, 3), dtype=np.uint8)
        for raw in (
            "SECRETSCANS",
            "SUPPORTUS ON ko-fi.com/Secretscans",
            "Discordggxzeknv",
            "JOIN US ATDISCORD",
            "LEAVE IT BLANK",
            "IF YOU WANT TO BE A PART OF OUR TEAM AND HELP US OUT",
        ):
            page = {"texts": [], "_vision_blocks": []}
            evidence = {
                "texts": [{"id": "neg", "text": raw, "bbox": [60, 40, 190, 78], "confidence": 0.91}],
                "blocks": [{"bbox": [60, 40, 190, 78], "confidence": 0.91}],
            }

            promoted = process_bands.fuse_negative_dark_bubble_candidates(page, evidence, image)

            self.assertEqual(promoted, 0, raw)
            self.assertEqual(page["texts"], [], raw)

    def test_ocr_stage_rejects_precomputed_page_with_out_of_band_geometry(self):
        from unittest.mock import MagicMock
        from strip import process_bands

        band = self._make_band()
        page_dict = process_bands._band_to_page_dict(band, page_idx=3, source_page_number=2)
        precomputed_ocr_page = {
            "texts": [
                {
                    "id": "bad",
                    "bbox": [50, 420, 150, 480],
                    "balloon_bbox": [40, 400, 160, 490],
                    "text": "WRONG SPACE",
                    "tipo": "fala",
                }
            ],
            "_vision_blocks": [{"bbox": [40, 400, 160, 490], "confidence": 0.9}],
            "_ocr_stats": {"macro_ocr_real": True},
        }
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [{"id": "fresh", "bbox": [50, 20, 150, 80], "text": "FRESH", "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }

        output = process_bands._run_band_ocr_stage(
            band,
            runtime=runtime,
            page_dict=page_dict,
            precomputed_ocr_page=precomputed_ocr_page,
        )

        runtime.run_ocr_stage.assert_called_once()
        self.assertEqual(output.to_page_dict()["texts"][0]["id"], "fresh")
        self.assertEqual(output.perf_updates["ocr_precomputed_page"], False)
        self.assertEqual(output.perf_updates["ocr_runtime_skipped"], False)
        self.assertTrue(output.perf_updates["ocr_precomputed_page_rejected"])
        self.assertIn("out_of_bounds", output.perf_updates["ocr_precomputed_page_reject_reason"])
        self.assertEqual(
            output.to_page_dict()["_ocr_stats"]["precomputed_ocr_reject_reason"],
            output.perf_updates["ocr_precomputed_page_reject_reason"],
        )

    def test_ocr_stage_rejects_precomputed_page_when_text_misses_balloon(self):
        from unittest.mock import MagicMock
        from strip import process_bands

        band = self._make_band()
        page_dict = process_bands._band_to_page_dict(band, page_idx=0, source_page_number=1)
        precomputed_ocr_page = {
            "texts": [
                {
                    "id": "bad",
                    "bbox": [5, 5, 30, 24],
                    "balloon_bbox": [210, 60, 280, 95],
                    "text": "WRONG BALLOON",
                    "tipo": "fala",
                }
            ],
            "_vision_blocks": [{"bbox": [210, 60, 280, 95], "confidence": 0.9}],
        }
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [{"id": "fresh", "bbox": [50, 20, 150, 80], "text": "FRESH", "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }

        output = process_bands._run_band_ocr_stage(
            band,
            runtime=runtime,
            page_dict=page_dict,
            precomputed_ocr_page=precomputed_ocr_page,
        )

        runtime.run_ocr_stage.assert_called_once()
        self.assertEqual(output.to_page_dict()["texts"][0]["id"], "fresh")
        self.assertIn("text_balloon_mismatch", output.perf_updates["ocr_precomputed_page_reject_reason"])

    def test_band_page_dict_keeps_sparse_ocr_mapping_default(self):
        from strip import process_bands

        band = self._make_band()
        page = process_bands._band_to_page_dict(band, page_idx=0, source_page_number=1)

        self.assertNotIn("_disable_sparse_ocr_mapping", page)

    def test_translate_stage_result_merges_ocr_metadata_as_snapshot(self):
        from unittest.mock import MagicMock
        from strip import process_bands

        ocr_page = {
            "numero": 2,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "text": "HELLO",
                    "line_polygons": [[[60, 30], [140, 30], [140, 70], [60, 70]]],
                }
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "t1", "translated": "OLA"}], "_vision_blocks": []}
        ]

        output = process_bands._run_translate_stage(
            ocr_page,
            translator=translator,
            context={"obra": "Demo"},
            glossario={"HELLO": "OLA"},
            idioma_origem="en",
            idioma_destino="pt-BR",
            obra="Demo",
            models_dir="D:/models",
            ollama_host="http://127.0.0.1:11434",
            ollama_model="model",
            translation_context={"memory": []},
        )

        self.assertEqual(output.stage_id, "translate")
        translated = output.to_page_dict()
        self.assertEqual(translated["texts"][0]["translated"], "OLA")
        self.assertEqual(translated["texts"][0]["bbox"], [50, 20, 150, 80])
        self.assertEqual(translated["texts"][0]["line_polygons"][0][0], [60, 30])
        self.assertEqual(translated["_vision_blocks"][0]["bbox"], [40, 10, 160, 90])

        translated["texts"][0]["translated"] = "MUTATED"
        ocr_page["texts"][0]["bbox"] = [0, 0, 1, 1]
        self.assertEqual(output.to_page_dict()["texts"][0]["translated"], "OLA")
        self.assertEqual(output.to_page_dict()["texts"][0]["bbox"], [50, 20, 150, 80])

    def test_finalize_ocr_page_before_translation_repairs_dark_lobe_duplicate_prefix(self):
        from unittest.mock import MagicMock
        from strip import process_bands

        ocr_page = {
            "numero": 5,
            "width": 800,
            "height": 981,
            "texts": [
                {
                    "id": "left",
                    "text": "The subspace retention is only five minutes.",
                    "bbox": [129, 111, 312, 234],
                    "text_pixel_bbox": [129, 111, 312, 234],
                    "balloon_bbox": [39, 32, 402, 437],
                    "bubble_mask_bbox": [39, 32, 402, 437],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "block_profile": "dark_bubble",
                },
                {
                    "id": "right",
                    "text": "space is only utes. If you exceed that time, you will return to your original world!",
                    "translated": "Space is only utes. se voce ultrapassar esse tempo!",
                    "bbox": [237, 36, 744, 557],
                    "text_pixel_bbox": [399, 204, 675, 335],
                    "balloon_bbox": [83, 32, 744, 691],
                    "bubble_mask_bbox": [83, 32, 744, 691],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "block_profile": "dark_bubble",
                },
            ],
            "_vision_blocks": [
                {
                    "id": "left",
                    "text": "The subspace retention is only five minutes.",
                    "bbox": [129, 111, 312, 234],
                    "text_pixel_bbox": [129, 111, 312, 234],
                    "bubble_mask_bbox": [39, 32, 402, 437],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "block_profile": "dark_bubble",
                },
                {
                    "id": "right",
                    "text": "space is only utes. If you exceed that time, you will return to your original world!",
                    "translated": "Space is only utes. se voce ultrapassar esse tempo!",
                    "bbox": [237, 36, 744, 557],
                    "text_pixel_bbox": [399, 204, 675, 335],
                    "bubble_mask_bbox": [83, 32, 744, 691],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "block_profile": "dark_bubble",
                },
            ],
        }

        changed = process_bands._finalize_ocr_page_before_translation(
            ocr_page,
            (981, 800, 3),
            page_number=5,
            source_language="en",
        )

        self.assertEqual(changed, 1)
        right = next(text for text in ocr_page["texts"] if text["id"] == "right")
        self.assertEqual(right["text"], "If you exceed that time, you will return to your original world!")
        self.assertNotIn("translated", right)
        self.assertIn("leading_dark_lobe_duplicate_fragment_removed", right["qa_flags"])

        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "right", "translated": "SE VOCE ULTRAPASSAR ESSE TEMPO!"}]}
        ]
        process_bands._run_translate_stage(ocr_page, translator=translator)

        sent_page = translator.translate_pages.call_args.args[0][0]
        sent_right = next(text for text in sent_page["texts"] if text["id"] == "right")
        self.assertEqual(sent_right["text"], "If you exceed that time, you will return to your original world!")
        self.assertNotIn("translated", sent_right)

    def test_review_layout_stage_result_adds_balloon_bbox_as_snapshot(self):
        from strip import process_bands

        band = self._make_band()
        ocr_page = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "text": "HELLO",
                    "tipo": "fala",
                }
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }

        output = process_bands._run_review_layout_stage(
            band,
            ocr_page=ocr_page,
            band_history=[],
            connected_reasoner_config={"enabled": True},
        )

        self.assertEqual(output.stage_id, "review_layout")
        reviewed = output.to_page_dict()
        self.assertEqual(reviewed["texts"][0]["balloon_bbox"], [50, 20, 150, 80])
        self.assertEqual(reviewed["_connected_balloon_reasoner"], {"enabled": True})

        reviewed["texts"][0]["balloon_bbox"] = [0, 0, 1, 1]
        ocr_page["texts"][0]["bbox"] = [0, 0, 1, 1]
        self.assertEqual(output.to_page_dict()["texts"][0]["balloon_bbox"], [50, 20, 150, 80])

    def test_inpaint_stage_result_snapshots_image_and_perf_updates(self):
        from unittest.mock import MagicMock
        import numpy as np
        from strip import process_bands

        band = self._make_band()
        translated_page = {
            "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "translated": "OLA"}],
        }
        inpainter = MagicMock()

        def fake_inpaint(_image, page):
            page["texts"][0]["mask_evidence"] = {
                "kind": "glyph_segmentation",
                "raw_mask_pixels": 12,
                "expanded_mask_pixels": 16,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
                "fast_fill_reject_reasons": [],
            }
            page["_vision_blocks"] = [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "mask_evidence": dict(page["texts"][0]["mask_evidence"]),
                }
            ]
            page["_strip_fast_white_balloon_count"] = 2
            page["_strip_fast_local_balloon_count"] = 1
            page["_strip_remaining_inpaint_blocks"] = 3
            page["_strip_used_real_inpaint"] = True
            page["_strip_fast_white_rejection_reasons"] = {"no_white_fill_mask": 4}
            return np.full((100, 300, 3), 210, dtype=np.uint8)

        inpainter.inpaint_band_image.side_effect = fake_inpaint

        output = process_bands._run_inpaint_stage(
            band,
            inpainter=inpainter,
            translated_page=translated_page,
        )

        self.assertEqual(output.stage_id, "inpaint")
        self.assertEqual(
            dict(output.perf_updates),
            {
                "fast_white_balloon_count": 2,
                "fast_local_balloon_count": 1,
                "remaining_inpaint_blocks": 3,
                "fast_white_rejection_reasons": {"no_white_fill_mask": 4},
                "used_real_inpaint": True,
            },
        )
        image_snapshot = output.to_image()
        image_snapshot[:, :, :] = 0
        self.assertEqual(int(output.to_image()[0, 0, 0]), 210)
        self.assertEqual(translated_page["texts"][0]["mask_evidence"]["kind"], "glyph_segmentation")
        self.assertEqual(translated_page["_vision_blocks"][0]["mask_evidence"]["raw_mask_pixels"], 12)

    def test_typeset_and_copy_back_stage_results_snapshot_images_without_mutating_band(self):
        from unittest.mock import MagicMock
        import numpy as np
        from strip import process_bands

        band = self._make_band()
        original = np.full((100, 300, 3), 50, dtype=np.uint8)
        cleaned = np.full((100, 300, 3), 180, dtype=np.uint8)
        rendered = np.full((100, 300, 3), 220, dtype=np.uint8)
        band.original_slice = original.copy()
        band.rendered_slice = None
        translated_page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [50, 20, 150, 80],
                    "translated": "OLA",
                }
            ],
        }
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = rendered.copy()

        typeset_output = process_bands._run_typeset_stage(
            cleaned,
            typesetter=typesetter,
            translated_page=translated_page,
        )
        copy_back_output = process_bands._run_copy_back_stage(
            band,
            rendered_slice=typeset_output.to_image(),
            translated_page=translated_page,
        )

        self.assertEqual(typeset_output.stage_id, "typeset")
        self.assertEqual(copy_back_output.stage_id, "copy_back")
        self.assertIsNone(band.rendered_slice)
        copy_back_image = copy_back_output.to_image()
        self.assertEqual(int(copy_back_image[50, 100, 0]), 220)
        self.assertEqual(int(copy_back_image[5, 5, 0]), 50)
        copy_back_image[:, :, :] = 0
        self.assertEqual(int(copy_back_output.to_image()[50, 100, 0]), 220)

    def test_commit_band_outputs_snapshots_final_band_state(self):
        import numpy as np
        from strip import process_bands

        band = self._make_band()
        cleaned = np.full((100, 300, 3), 180, dtype=np.uint8)
        rendered = np.full((100, 300, 3), 220, dtype=np.uint8)
        ocr_result = {
            "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "translated": "OLA"}],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90]}],
        }

        committed = process_bands._commit_band_outputs(
            band,
            cleaned_slice=cleaned,
            rendered_slice=rendered,
            ocr_result=ocr_result,
        )

        self.assertIs(committed, band)
        cleaned[:, :, :] = 0
        rendered[:, :, :] = 0
        ocr_result["texts"][0]["translated"] = "MUTATED"

        self.assertEqual(int(band.cleaned_slice[0, 0, 0]), 180)
        self.assertEqual(int(band.rendered_slice[0, 0, 0]), 220)
        self.assertEqual(band.ocr_result["texts"][0]["translated"], "OLA")

    def test_process_band_with_no_balloons_returns_original_slice(self):
        from strip.process_bands import process_band
        from strip.types import Band
        import numpy as np
        from unittest.mock import MagicMock

        slice_img = np.full((50, 300, 3), 80, dtype=np.uint8)
        band = Band(y_top=0, y_bottom=50, balloons=[], strip_slice=slice_img.copy(), original_slice=slice_img.copy())

        result = process_band(
            band,
            runtime=MagicMock(),
            translator=MagicMock(),
            inpainter=MagicMock(),
            typesetter=MagicMock(),
            page_idx=0,
        )

        self.assertIs(result, band)

    def test_process_band_with_no_accepted_texts_skips_expensive_stages(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [],
            "_vision_blocks": [{"bbox": [50, 20, 150, 80], "confidence": 0.9}],
            "_ocr_stats": {
                "scanlation_credit_skipped": True,
                "cover_editorial_skipped": True,
                "block_count": 1,
                "full_page_mapped": 0,
                "crop_fallback_attempts": 0,
                "crop_fallback_recovered": 0,
            },
        }
        translator = MagicMock()
        inpainter = MagicMock()
        typesetter = MagicMock()

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        self.assertIs(result, band)
        runtime.run_ocr_stage.assert_called_once()
        translator.translate_pages.assert_not_called()
        inpainter.inpaint_band_image.assert_not_called()
        typesetter.render_band_image.assert_not_called()
        self.assertTrue(np.array_equal(band.cleaned_slice, band.original_slice))
        self.assertTrue(np.array_equal(band.rendered_slice, band.original_slice))
        self.assertEqual(band.ocr_result["texts"], [])
        self.assertEqual(band.ocr_result["_vision_blocks"], [])
        self.assertTrue(band.perf["ocr_scanlation_credit_skipped"])
        self.assertTrue(band.perf["ocr_cover_editorial_skipped"])

    def test_process_band_excludes_discord_scanlation_promo_without_rendering(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "direct_paddle_reocr_001",
                    "text": "iscord.gg/xzeKn8V",
                    "bbox": [40, 55, 260, 78],
                    "confidence": 0.95,
                }
            ],
            "_vision_blocks": [{"bbox": [40, 55, 260, 78], "confidence": 0.95}],
            "_ocr_stats": {},
        }
        translator = MagicMock()
        inpainter = MagicMock()
        typesetter = MagicMock()
        empty_recovery = SimpleNamespace(to_page_dict=lambda: {"texts": []}, perf_updates={})

        with patch("strip.process_bands._recover_empty_ocr_with_candidate_crops", return_value=empty_recovery), patch(
            "strip.process_bands._recover_partial_dark_bubble_ocr_from_texts",
            return_value=0,
        ):
            result = process_band(
                band,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
                page_idx=0,
            )

        self.assertIs(result, band)
        translator.translate_pages.assert_not_called()
        inpainter.inpaint_band_image.assert_not_called()
        typesetter.render_band_image.assert_not_called()
        self.assertTrue(np.array_equal(band.cleaned_slice, band.original_slice))
        self.assertTrue(np.array_equal(band.rendered_slice, band.original_slice))
        self.assertEqual(band.ocr_result["export_policy"], "exclude_from_translated_output")
        self.assertEqual(band.ocr_result["exclusion_reason"], "scanlation_discord_promo")
        self.assertTrue(band.ocr_result["excluded_non_story"])
        self.assertTrue(band.ocr_result["texts"][0]["skip_processing"])
        self.assertEqual(band.perf["exclusion_reason"], "scanlation_discord_promo")

    def test_process_band_excludes_discord_promo_rejected_by_crop_reocr(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [],
            "_vision_blocks": [],
            "_ocr_stats": {},
        }
        translator = MagicMock()
        inpainter = MagicMock()
        typesetter = MagicMock()
        recovery_page = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [],
            "_vision_blocks": [],
            "_ocr_stats": {
                "candidate_crop_reocr_candidate_count": 1,
                "candidate_crop_reocr_attempts": 1,
                "candidate_crop_reocr_recovered": 0,
                "scanlation_discord_promo_detected": True,
                "scanlation_credit_rejected_texts": ["iscord.gg/xzeKn8V"],
            },
        }
        recovery = SimpleNamespace(to_page_dict=lambda: recovery_page, perf_updates={})

        with patch("strip.process_bands._recover_empty_ocr_with_candidate_crops", return_value=recovery), patch(
            "strip.process_bands._recover_partial_dark_bubble_ocr_from_texts",
            return_value=0,
        ):
            result = process_band(
                band,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
                page_idx=0,
            )

        self.assertIs(result, band)
        translator.translate_pages.assert_not_called()
        inpainter.inpaint_band_image.assert_not_called()
        typesetter.render_band_image.assert_not_called()
        self.assertTrue(np.array_equal(band.cleaned_slice, band.original_slice))
        self.assertTrue(np.array_equal(band.rendered_slice, band.original_slice))
        self.assertEqual(band.ocr_result["texts"], [])
        self.assertEqual(band.ocr_result["export_policy"], "exclude_from_translated_output")
        self.assertEqual(band.ocr_result["exclusion_reason"], "scanlation_discord_promo")
        self.assertTrue(band.ocr_result["excluded_non_story"])
        self.assertEqual(band.perf["exclusion_reason"], "scanlation_discord_promo")

    def test_process_band_runs_pipeline_when_ocr_texts_are_legacy_skip_processing(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        from debug_tools import DebugRecorder, bind_recorder
        import numpy as np

        band = self._make_band()
        cleaned = np.full_like(band.strip_slice, 80)
        rendered = np.full_like(band.strip_slice, 180)
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "text": "YOU...!!",
                    "original": "YOU...!!",
                    "translated": "YOU...!!",
                    "tipo": "narracao",
                    "skip_processing": True,
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {
                "texts": [
                    {
                        "id": "t1",
                        "bbox": [50, 20, 150, 80],
                        "balloon_bbox": [40, 10, 160, 90],
                        "original": "YOU...!!",
                        "translated": "VOCE...!!",
                        "tipo": "narracao",
                        "skip_processing": True,
                    }
                ],
            }
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.return_value = cleaned
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = rendered

        with tempfile.TemporaryDirectory() as tmp:
            recorder = DebugRecorder(Path(tmp), enabled=True, run_id="skip-processing-test")
            bind_recorder(recorder)
            try:
                result = process_band(
                    band,
                    runtime=runtime,
                    translator=translator,
                    inpainter=inpainter,
                    typesetter=typesetter,
                    page_idx=0,
                )
            finally:
                bind_recorder(None)
            decision_path = (
                Path(tmp)
                / "debug"
                / "e2e"
                / "08_inpaint"
                / "page_001_band_000"
                / "inpaint_decision.json"
            )
            self.assertFalse(decision_path.exists())

        self.assertIs(result, band)
        translator.translate_pages.assert_called_once()
        inpainter.inpaint_band_image.assert_called_once()
        typesetter.render_band_image.assert_called_once()
        self.assertTrue(np.array_equal(band.cleaned_slice, cleaned))
        self.assertFalse(np.array_equal(band.rendered_slice, band.original_slice))
        self.assertFalse(band.perf.get("skip_processing_copy", False))
        self.assertTrue(band.ocr_result["texts"][0]["skip_processing"])
        self.assertIn("balloon_bbox", band.ocr_result["texts"][0])

    def test_process_band_runs_pipeline_when_translation_marks_legacy_skip_processing(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        from debug_tools import DebugRecorder, bind_recorder
        import numpy as np

        band = self._make_band()
        cleaned = np.full_like(band.strip_slice, 80)
        rendered = np.full_like(band.strip_slice, 180)
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [40, 10, 160, 90],
                    "text": "YOU...!!",
                    "original": "YOU...!!",
                    "tipo": "narracao",
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {
                "texts": [
                    {
                        "id": "t1",
                        "original": "YOU...!!",
                        "translated": "VOCE...!!",
                        "tipo": "narracao",
                        "skip_processing": True,
                    }
                ],
            }
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.return_value = cleaned
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = rendered

        with tempfile.TemporaryDirectory() as tmp:
            recorder = DebugRecorder(Path(tmp), enabled=True, run_id="post-translate-skip-test")
            bind_recorder(recorder)
            try:
                result = process_band(
                    band,
                    runtime=runtime,
                    translator=translator,
                    inpainter=inpainter,
                    typesetter=typesetter,
                    page_idx=0,
                )
            finally:
                bind_recorder(None)
            decision_path = (
                Path(tmp)
                / "debug"
                / "e2e"
                / "08_inpaint"
                / "page_001_band_000"
                / "inpaint_decision.json"
            )
            self.assertFalse(decision_path.exists())

        self.assertIs(result, band)
        translator.translate_pages.assert_called_once()
        inpainter.inpaint_band_image.assert_called_once()
        typesetter.render_band_image.assert_called_once()
        self.assertTrue(np.array_equal(band.cleaned_slice, cleaned))
        self.assertFalse(np.array_equal(band.rendered_slice, band.original_slice))
        self.assertFalse(band.perf.get("skip_processing_copy", False))
        self.assertTrue(band.ocr_result["texts"][0]["skip_processing"])
        self.assertIn("balloon_bbox", band.ocr_result["texts"][0])

    def test_process_band_still_repaints_when_all_translations_are_unchanged(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [40, 10, 160, 90],
                    "text": "HYAAH!!",
                    "original": "HYAAH!!",
                    "tipo": "narracao",
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {
                "texts": [
                    {
                        "id": "t1",
                        "translated": "HYAAH!!",
                        "original": "HYAAH!!",
                        "tipo": "narracao",
                    }
                ],
            }
        ]
        cleaned = band.strip_slice.copy()
        cleaned[30:40, 70:90] = 250
        rendered = cleaned.copy()
        rendered[45:55, 80:120] = 10
        inpainter = MagicMock()
        inpainter.inpaint_band_image.return_value = cleaned
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = rendered

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        self.assertIs(result, band)
        translator.translate_pages.assert_called_once()
        inpainter.inpaint_band_image.assert_called_once()
        typesetter.render_band_image.assert_called_once()
        self.assertTrue(np.array_equal(band.cleaned_slice, cleaned))
        self.assertFalse(np.array_equal(band.rendered_slice, band.original_slice))
        self.assertFalse(band.perf.get("unchanged_translation_skip", False))
        self.assertEqual(band.ocr_result["texts"][0]["translated"], "HYAAH!!")
        self.assertIn("balloon_bbox", band.ocr_result["texts"][0])

class BandAdaptersTests(unittest.TestCase):
    def test_inpaint_band_image_returns_same_shape(self):
        from inpainter import inpaint_band_image
        import numpy as np
        band = np.full((100, 300, 3), 200, dtype=np.uint8)
        page = {"texts": [
            {"id": "t1", "bbox": [50, 20, 150, 80], "tipo": "fala", "original": "HELLO"},
        ]}
        cleaned = inpaint_band_image(band, page)
        self.assertEqual(cleaned.shape, band.shape)

    def test_render_band_image_returns_same_shape(self):
        from typesetter.renderer import render_band_image
        import numpy as np
        band = np.full((100, 300, 3), 255, dtype=np.uint8)
        page = {"texts": [
            {"id": "t1", "bbox": [50, 20, 150, 80], "tipo": "fala",
             "balloon_bbox": [50, 20, 150, 80],
             "translated": "OLÁ",
             "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "cor": "#000000"}},
        ]}
        rendered = render_band_image(band, page)
        self.assertEqual(rendered.shape, band.shape)

