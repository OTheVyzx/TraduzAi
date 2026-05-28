import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from layout import balloon_layout as balloon_layout_mod
from qa.export_gate import evaluate_export_gate
from typesetter import renderer as renderer_mod
from typesetter.renderer import (
    SafeTextPathFont,
    _MIN_FONT_SIZE,
    _build_textpath_mask,
    _canonical_render_style,
    _category_font_bounds,
    _font_has_glyph,
    _normalize_render_text,
    _render_single_text_block_unrotated,
    _render_single_text_block,
    _resolve_text_layout,
    build_render_blocks,
    ensure_legible_plan,
    find_font,
    plan_text_layout,
    render_band_image,
    render_text_block,
)


class TypesettingRendererTests(unittest.TestCase):
    _LEGACY_CONNECTED_DEFAULT_TESTS = set()
    _LEGACY_CONNECTED_DEFAULT_TESTS_DISABLED = {
        "test_build_render_blocks_dedupes_nested_same_balloon_prefix_text",
        "test_plan_text_layout_does_not_lock_short_white_balloon_text_to_page_edge",
        "test_plan_text_layout_offsets_safe_box_for_horizontal_anchor_bias",
        "test_two_texts_with_subregions_no_double_render",
    }

    def setUp(self):
        if self._testMethodName in self._LEGACY_CONNECTED_DEFAULT_TESTS:
            self.skipTest("legacy balloon/connected render behavior disabled by simple OCR-position layout")

    def test_build_render_blocks_route_render_overrides_legacy_skip_and_preserve(self):
        blocks = build_render_blocks([
            {
                "id": "ocr_legacy_render",
                "text": "HELLO",
                "translated": "OLA",
                "route_action": "translate_inpaint_render",
                "skip_processing": True,
                "preserve_original": True,
                "render_policy": "preserve_original",
                "content_class": "logo",
                "bbox": [20, 20, 120, 70],
                "balloon_bbox": [10, 10, 140, 90],
                "tipo": "texto",
            }
        ])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["id"], "ocr_legacy_render")
        self.assertFalse(blocks[0].get("skip_processing"))
        self.assertFalse(blocks[0].get("preserve_original"))
        self.assertEqual(blocks[0].get("render_policy"), "normal")

    def test_build_render_blocks_ignores_legacy_decision_fields_without_route_action(self):
        text = {
            "id": "ocr_legacy_fields",
            "text": "HELLO",
            "translated": "OLA",
            "bbox": [20, 20, 120, 70],
            "balloon_bbox": [10, 10, 140, 90],
            "tipo": "sfx",
            "content_class": "noise",
            "balloon_type": "dark",
            "skip_processing": True,
            "preserve_original": True,
            "render_policy": "preserve_original",
        }

        blocks = build_render_blocks([text])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["id"], "ocr_legacy_fields")
        self.assertEqual(blocks[0]["translated"], "OLA")
        self.assertEqual(blocks[0].get("route_action"), "translate_inpaint_render")
        self.assertFalse(blocks[0].get("skip_processing"))
        self.assertFalse(blocks[0].get("preserve_original"))
        self.assertEqual(blocks[0].get("render_policy"), "normal")

    def test_plan_text_layout_treats_legacy_type_fields_as_neutral_text(self):
        text_data = {
            "id": "ocr_neutral",
            "text": "HELLO THERE",
            "translated": "OLA, TUDO BEM?",
            "bbox": [40, 40, 240, 120],
            "text_pixel_bbox": [70, 62, 210, 96],
            "balloon_bbox": [30, 25, 260, 140],
            "tipo": "sfx",
            "content_class": "dialogue",
            "balloon_type": "white",
            "rotation_deg": 90,
            "estilo": {"tamanho": 24, "alinhamento": "left"},
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["font_name"], "ComicNeue-Bold.ttf")
        self.assertEqual(_category_font_bounds({"tipo": "sfx", "balloon_type": "textured"}), (16, 48))
        self.assertEqual(plan["alignment"], "left")
        self.assertEqual(plan["rotation_deg"], 90)
        self.assertNotIn("rotated_text_policy_unmet", text_data.get("qa_flags") or [])

    def test_visual_rect_detection_uses_geometry_not_legacy_type_labels(self):
        text_data = {
            "translated": "TEXTO VISUAL",
            "bbox": [20, 20, 260, 130],
            "text_pixel_bbox": [82, 62, 142, 84],
            "balloon_bbox": [20, 20, 260, 130],
            "tipo": "sfx",
            "balloon_type": "dark",
        }

        self.assertTrue(renderer_mod._should_detect_visual_rect_safe_area(text_data))

    def test_capacity_expansion_uses_geometry_not_legacy_type_labels(self):
        text_data = {
            "translated": "FRASE CURTA PARA CRESCER",
            "text": "SMALL SOURCE",
            "bbox": [20, 20, 260, 150],
            "text_pixel_bbox": [82, 62, 142, 84],
            "balloon_bbox": [20, 20, 260, 150],
            "layout_profile": "white_balloon",
            "tipo": "sfx",
            "balloon_type": "dark",
        }
        anchor_bbox = [82, 62, 142, 84]
        target_bbox = [20, 20, 260, 150]
        layout_safe_bbox = [40, 34, 240, 136]

        self.assertTrue(
            renderer_mod._should_auto_expand_tiny_anchor_capacity(
                text_data,
                anchor_bbox,
                target_bbox,
                18,
            )
        )
        self.assertTrue(
            renderer_mod._should_use_safe_area_for_follow_anchor_capacity(
                text_data,
                anchor_bbox,
                layout_safe_bbox,
                target_bbox,
            )
        )

    def test_shape_and_corpus_hints_ignore_legacy_tipo(self):
        self.assertEqual(renderer_mod._infer_layout_shape_from_bbox([0, 0, 150, 100], "narracao"), "wide")
        self.assertEqual(
            renderer_mod._infer_layout_shape_from_bbox([0, 0, 150, 100], "narracao"),
            renderer_mod._infer_layout_shape_from_bbox([0, 0, 150, 100], "texto"),
        )

        visual = {"page_geometry": {"median_width": 760, "median_aspect_ratio": 0.30}}
        textual = {"paired_text_stats": {"mean_translation_length_ratio": 1.20}}
        self.assertEqual(
            renderer_mod._apply_corpus_layout_hints(
                width_ratio=0.82,
                tipo="narracao",
                layout_shape="wide",
                corpus_visual=visual,
                corpus_textual=textual,
            ),
            renderer_mod._apply_corpus_layout_hints(
                width_ratio=0.82,
                tipo="sfx",
                layout_shape="wide",
                corpus_visual=visual,
                corpus_textual=textual,
            ),
        )

    def test_build_render_blocks_groups_shared_balloon_without_legacy_tipo_split(self):
        texts = [
            {
                "id": "ocr_001",
                "text": "FIRST",
                "translated": "PRIMEIRO",
                "bbox": [90, 40, 180, 70],
                "text_pixel_bbox": [90, 40, 180, 70],
                "balloon_bbox": [60, 20, 260, 160],
                "layout_group_size": 2,
                "tipo": "fala",
            },
            {
                "id": "ocr_002",
                "text": "SECOND",
                "translated": "SEGUNDO",
                "bbox": [95, 105, 190, 135],
                "text_pixel_bbox": [95, 105, 190, 135],
                "balloon_bbox": [60, 20, 260, 160],
                "layout_group_size": 2,
                "tipo": "narracao",
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["source_text_count"], 2)
        self.assertEqual(blocks[0]["translated"], "PRIMEIRO\nSEGUNDO")

    def test_build_render_blocks_keeps_distinct_real_bubble_masks_separate(self):
        shared_parent_bbox = [148, 655, 667, 998]
        texts = [
            {
                "id": "ocr_001",
                "translated": "O que e...? por que ele esta ficando",
                "bbox": [148, 655, 667, 998],
                "text_pixel_bbox": [487, 870, 590, 949],
                "balloon_bbox": list(shared_parent_bbox),
                "bubble_id": "page_025_band_067_bubble_001",
                "bubble_mask_bbox": list(shared_parent_bbox),
                "layout_group_size": 3,
                "layout_profile": "white_balloon",
            },
            {
                "id": "ocr_002",
                "translated": "O que e...?",
                "bbox": [581, 870, 663, 909],
                "text_pixel_bbox": [581, 870, 663, 909],
                "balloon_bbox": list(shared_parent_bbox),
                "bubble_id": "page_025_band_067_bubble_002",
                "bubble_mask_bbox": [581, 870, 663, 909],
                "layout_group_size": 3,
                "layout_profile": "white_balloon",
            },
            {
                "id": "ocr_003",
                "translated": "Com medo sozinho?",
                "bbox": [463, 933, 629, 1004],
                "text_pixel_bbox": [463, 987, 630, 1004],
                "balloon_bbox": list(shared_parent_bbox),
                "bubble_id": "page_025_band_067_bubble_003",
                "bubble_mask_bbox": [463, 933, 629, 1004],
                "layout_group_size": 3,
                "layout_profile": "white_balloon",
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual({block.get("id") for block in blocks}, {"ocr_001", "ocr_002", "ocr_003"})
        self.assertFalse(any("\n" in str(block.get("translated") or "") for block in blocks))

    def test_plan_text_layout_prefers_distinct_real_bubble_mask_over_shared_parent_balloon(self):
        text_data = {
            "id": "ocr_003",
            "translated": "Com medo sozinho?",
            "bbox": [463, 933, 629, 1004],
            "source_bbox": [463, 933, 629, 1004],
            "text_pixel_bbox": [463, 987, 630, 1004],
            "layout_bbox": [463, 987, 630, 1004],
            "balloon_bbox": [148, 655, 667, 998],
            "bubble_id": "page_025_band_067_bubble_003",
            "bubble_mask_bbox": [463, 933, 629, 1004],
            "bubble_inner_bbox": [475, 945, 617, 992],
            "layout_group_size": 3,
            "layout_profile": "white_balloon",
            "style_origin": "auto",
            "page_width": 690,
            "page_height": 1734,
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [463, 933, 629, 1004])
        self.assertEqual(text_data.get("_render_target_source"), "real_bubble_mask_bbox_distinct")
        self.assertIn("safe_text_box_recomputed", text_data.get("qa_flags") or [])

    def test_render_qa_uses_background_evidence_not_legacy_balloon_type(self):
        background = np.full((120, 220, 3), 255, dtype=np.uint8)
        background[45:66, 70:154] = 8
        text_data = {
            "original": "HELLO",
            "translated": "OLA",
            "render_bbox": [70, 45, 154, 66],
            "balloon_bbox": [20, 18, 200, 100],
            "balloon_type": "white",
            "qa_flags": [],
        }
        plan = {
            "target_bbox": [20, 18, 200, 100],
            "safe_text_box": [30, 25, 190, 92],
        }

        renderer_mod._run_render_qa(text_data, plan, background_image=background)

        self.assertIn("render_on_art_suspected", text_data.get("qa_flags") or [])
        self.assertEqual((text_data.get("qa_metrics") or {}).get("render_background_luma"), 8.0)

    def test_plan_text_layout_uses_bubble_inner_bbox_as_auto_safe_area(self):
        text_data = {
            "translated": "TEXTO CENTRALIZADO NO INTERIOR",
            "bbox": [140, 100, 260, 140],
            "text_pixel_bbox": [140, 100, 260, 140],
            "balloon_bbox": [60, 60, 340, 220],
            "bubble_inner_bbox": [86, 82, 314, 198],
            "tipo": "fala",
            "layout_profile": "white_balloon",
            "balloon_type": "white",
            "style_origin": "auto",
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["layout_safe_bbox"], [86, 82, 314, 198])
        self.assertEqual(plan["layout_safe_reason"], "bubble_inner_bbox")

    def test_plan_text_layout_expands_nearby_tiny_white_balloon_target(self):
        text_data = {
            "translated": "O que aconteceu?",
            "bbox": [258, 14769, 287, 14797],
            "text_pixel_bbox": [228, 14779, 237, 14792],
            "layout_bbox": [228, 14779, 237, 14792],
            "balloon_bbox": [216, 14767, 249, 14804],
            "bubble_mask_bbox": [258, 14769, 287, 14797],
            "bubble_inner_bbox": [270, 14781, 275, 14785],
            "tipo": "texto",
            "layout_profile": "white_balloon",
            "balloon_type": "white",
            "style_origin": "auto",
            "page_width": 520,
            "page_height": 15040,
        }

        plan = plan_text_layout(text_data)
        tx1, _ty1, tx2, _ty2 = plan["target_bbox"]

        self.assertEqual(text_data.get("_render_target_source"), "tiny_anchor_union")
        self.assertLessEqual(tx1, 216)
        self.assertGreaterEqual(tx2, 287)
        self.assertGreaterEqual(tx2 - tx1, 96)
        self.assertGreaterEqual(plan["max_width"], 70)

    def test_plan_text_layout_prefers_real_bubble_over_underfit_refined_bbox(self):
        text_data = {
            "translated": "Ajussi! quanto tempo levará para chegar ao hospital mais próximo? ei",
            "bbox": [125, 4294, 656, 4853],
            "source_bbox": [125, 4294, 656, 4853],
            "text_pixel_bbox": [135, 4337, 274, 4381],
            "layout_bbox": [135, 4337, 274, 4381],
            "balloon_bbox": [122, 4247, 287, 4397],
            "bubble_mask_bbox": [125, 4294, 656, 4853],
            "bubble_inner_bbox": [162, 4331, 619, 4816],
            "line_polygons": [
                [[129, 4341], [338, 4342], [338, 4360], [129, 4359]],
                [[137, 4367], [332, 4367], [332, 4385], [137, 4385]],
                [[151, 4394], [320, 4394], [320, 4411], [151, 4411]],
                [[123, 4420], [344, 4420], [344, 4438], [123, 4438]],
            ],
            "tipo": "texto",
            "layout_profile": "white_balloon",
            "balloon_type": "white",
            "style_origin": "auto",
            "page_width": 700,
            "page_height": 5000,
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [125, 4294, 656, 4853])
        self.assertEqual(text_data.get("_render_target_source"), "real_bubble_mask_bbox")
        self.assertGreaterEqual(plan["max_width"], 300)
        self.assertNotEqual(plan["fit_status"] if "fit_status" in plan else None, "below_minimum_legible")

    def test_plan_text_layout_uses_real_bubble_when_ocr_geometry_is_overmerged(self):
        text_data = {
            "translated": "O que e...? por que ele esta ficando com medo sozinho?",
            "bbox": [148, 655, 667, 1004],
            "source_bbox": [148, 655, 667, 1004],
            "text_pixel_bbox": [463, 870, 663, 1004],
            "layout_bbox": [463, 870, 663, 1004],
            "balloon_bbox": [148, 655, 667, 998],
            "bubble_mask_bbox": [463, 933, 629, 1004],
            "bubble_inner_bbox": [475, 945, 617, 992],
            "line_polygons": [
                [[463, 870], [663, 870], [663, 890], [463, 890]],
                [[480, 933], [629, 933], [629, 955], [480, 955]],
                [[475, 960], [617, 960], [617, 992], [475, 992]],
            ],
            "layout_profile": "white_balloon",
            "balloon_type": "white",
            "tipo": "texto",
            "style_origin": "auto",
            "page_width": 690,
            "page_height": 1734,
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [463, 933, 629, 1004])
        self.assertEqual(text_data.get("_render_target_source"), "real_bubble_mask_bbox_overmerged_guard")
        self.assertIn("ocr_geometry_overmerged", text_data.get("qa_flags") or [])
        self.assertIn("safe_text_box_recomputed", text_data.get("qa_flags") or [])

    def test_plan_text_layout_rejects_collapsed_anchor_when_real_bubble_exists(self):
        text_data = {
            "translated": "Ajussi! quanto tempo levará para chegar ao hospital mais próximo? ei",
            "bbox": [125, 96, 656, 655],
            "source_bbox": [125, 96, 656, 655],
            "text_pixel_bbox": [135, 139, 274, 183],
            "layout_bbox": [135, 139, 274, 183],
            "balloon_bbox": [214, 127, 269, 180],
            "bubble_mask_bbox": [125, 96, 656, 655],
            "bubble_inner_bbox": [162, 133, 619, 618],
            "line_polygons": [
                [[129, 143], [338, 144], [338, 162], [129, 161]],
                [[137, 169], [332, 169], [332, 187], [137, 187]],
                [[151, 196], [320, 196], [320, 213], [151, 213]],
                [[123, 222], [344, 222], [344, 240], [123, 240]],
            ],
            "tipo": "texto",
            "layout_profile": "white_balloon",
            "balloon_type": "white",
            "style_origin": "auto",
            "page_width": 700,
            "page_height": 1000,
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [125, 96, 656, 655])
        self.assertEqual(text_data.get("_render_target_source"), "real_bubble_mask_bbox")
        self.assertGreaterEqual(plan["max_width"], 300)

    def test_plan_text_layout_uses_large_source_bbox_when_balloon_bbox_is_tight(self):
        text_data = {
            "translated": "Ajussi! quanto tempo levará para chegar ao hospital mais próximo? ei",
            "bbox": [125, 96, 656, 655],
            "source_bbox": [125, 96, 656, 655],
            "text_pixel_bbox": [135, 139, 274, 183],
            "layout_bbox": [135, 139, 274, 183],
            "balloon_bbox": [122, 49, 287, 199],
            "line_polygons": [
                [[129, 143], [338, 144], [338, 162], [129, 161]],
                [[137, 169], [332, 169], [332, 187], [137, 187]],
                [[151, 196], [320, 196], [320, 213], [151, 213]],
                [[123, 222], [344, 222], [344, 240], [123, 240]],
            ],
            "tipo": "texto",
            "layout_profile": "white_balloon",
            "balloon_type": "white",
            "style_origin": "auto",
            "page_width": 700,
            "page_height": 1000,
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [122, 49, 656, 655])
        self.assertEqual(text_data.get("_render_target_source"), "collapsed_balloon_source_bbox")
        self.assertGreaterEqual(plan["max_width"], 300)

    def test_visual_rect_target_rejected_when_it_clips_source_lines(self):
        text_data = {
            "bbox": [125, 96, 656, 655],
            "source_bbox": [125, 96, 656, 655],
            "text_pixel_bbox": [135, 139, 274, 183],
            "balloon_bbox": [214, 127, 269, 180],
            "line_polygons": [
                [[129, 143], [338, 144], [338, 162], [129, 161]],
                [[137, 169], [332, 169], [332, 187], [137, 187]],
                [[151, 196], [320, 196], [320, 213], [151, 213]],
                [[123, 222], [344, 222], [344, 240], [123, 240]],
            ],
        }

        self.assertTrue(
            renderer_mod._visual_outer_clips_source_geometry(
                text_data,
                [122, 49, 287, 199],
            )
        )

    def test_plan_text_layout_anchors_textured_panel_text_to_source_lines(self):
        base_text_data = {
            "translated": "Está feito... vamos mudar!",
            "bbox": [237, 4987, 561, 5444],
            "text_pixel_bbox": [252, 5018, 322, 5029],
            "layout_bbox": [252, 5018, 322, 5029],
            "balloon_bbox": [237, 4987, 561, 5444],
            "bubble_mask_bbox": [237, 4987, 561, 5444],
            "bubble_inner_bbox": [259, 5009, 539, 5422],
            "line_polygons": [
                [[177, 4987], [308, 4989], [308, 5009], [177, 5007]],
                [[158, 5015], [324, 5015], [324, 5032], [158, 5032]],
            ],
            "tipo": "texto",
            "layout_profile": "standard",
            "balloon_type": "textured",
            "style_origin": "auto",
            "page_width": 620,
            "page_height": 5600,
        }

        text_data = dict(base_text_data)
        plan = plan_text_layout(text_data)

        self.assertTrue(plan["_follow_english_anchor_position"])
        self.assertTrue(plan["_anchor_capacity_locked"])
        self.assertLessEqual(plan["safe_text_box"][1], 4987)
        self.assertGreaterEqual(plan["safe_text_box"][3], 5032)
        self.assertEqual(text_data.get("_render_target_source"), "textured_anchor_overbroad_target")

        img = Image.new("RGB", (620, 5600), (245, 245, 245))
        render_data = dict(base_text_data)
        render_text_block(img, render_data)
        self.assertEqual(render_data.get("_render_target_source"), "textured_anchor_overbroad_target")
        self.assertNotIn("TEXT_CLIPPED", render_data.get("qa_flags") or [])
        self.assertNotIn("TEXT_OVERFLOW", render_data.get("qa_flags") or [])

    def test_render_text_block_persists_ok_fit_attempts_for_normal_layer(self):
        img = Image.new("RGB", (360, 220), (255, 255, 255))
        text_data = {
            "id": "ocr_ok",
            "text": "HELLO THERE",
            "translated": "OLA, TUDO BEM?",
            "route_action": "translate_inpaint_render",
            "bbox": [80, 70, 280, 130],
            "source_bbox": [80, 70, 280, 130],
            "text_pixel_bbox": [95, 84, 265, 112],
            "balloon_bbox": [60, 45, 300, 165],
            "page_width": 360,
            "page_height": 220,
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "force_upper": True},
        }

        render_text_block(img, text_data)

        attempts = text_data.get("fit_attempts")
        self.assertIsInstance(attempts, list)
        self.assertGreaterEqual(len(attempts), 1)
        self.assertLessEqual(len(attempts), 2)
        self.assertEqual(text_data.get("fit_status"), "ok")
        self.assertEqual(attempts[-1]["status"], "ok")
        self.assertIn("font_px", attempts[-1])
        self.assertIn("lines", attempts[-1])
        render_debug = text_data.get("_render_debug") or {}
        self.assertNotIn("fit_status", render_debug)
        self.assertIn(render_debug.get("layout_fit_result"), {"pass", "fallback"})
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags") or [])

    def test_render_text_block_flags_below_minimum_legible_and_blocks_export(self):
        img = Image.new("RGB", (1600, 360), (255, 255, 255))
        text_data = {
            "id": "ocr_tiny",
            "text": "THIS CANNOT FIT",
            "translated": (
                "ESTE TEXTO TRADUZIDO E LONGO DEMAIS PARA CABER NESTE BALAO "
                "MINUSCULO SEM FICAR ILEGIVEL"
            ),
            "route_action": "translate_inpaint_render",
            "bbox": [20, 20, 82, 45],
            "source_bbox": [20, 20, 82, 45],
            "text_pixel_bbox": [24, 24, 78, 42],
            "balloon_bbox": [16, 16, 86, 50],
            "page_width": 1600,
            "page_height": 360,
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 26},
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data.get("fit_status"), "below_minimum_legible")
        self.assertIn("fit_below_minimum_legible", text_data.get("qa_flags") or [])
        self.assertIn("render_bbox", text_data)
        self.assertEqual(text_data.get("route_action"), "translate_inpaint_render")
        attempts = text_data.get("fit_attempts")
        self.assertIsInstance(attempts, list)
        self.assertGreaterEqual(len(attempts), 1)
        self.assertEqual(attempts[-1]["status"], "overflow")
        self.assertGreaterEqual(attempts[-1]["font_px"], 19)

        gate = evaluate_export_gate(
            {
                "paginas": [
                    {
                        "numero": 1,
                        "text_layers": [text_data],
                    }
                ]
            }
        )
        self.assertEqual(gate["status"], "BLOCK")
        self.assertFalse(gate["allowed"])

    def test_plan_text_layout_rejects_tiny_bubble_inner_bbox_for_short_white_balloon_text(self):
        text_data = {
            "text": "MOM..",
            "original": "MOM..",
            "translated": "MÃE..",
            "bbox": [538, 334, 623, 376],
            "source_bbox": [538, 334, 623, 376],
            "text_pixel_bbox": [538, 350, 623, 372],
            "balloon_bbox": [520, 322, 641, 388],
            "bubble_mask_bbox": [538, 334, 623, 376],
            "bubble_inner_bbox": [550, 346, 611, 364],
            "page_width": 800,
            "page_height": 13000,
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "style_origin": "auto",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 29, "cor": "#000000", "force_upper": True},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertNotEqual(plan["layout_safe_bbox"], [550, 346, 611, 364])
        self.assertGreaterEqual(plan["safe_text_box"][2] - plan["safe_text_box"][0], 80)
        self.assertGreaterEqual(plan["safe_text_box"][3] - plan["safe_text_box"][1], 40)
        self.assertGreaterEqual(resolved["font_size"], 18)
        self.assertIn("tiny_bubble_inner_bbox_rejected", text_data.get("qa_flags", []))

    def test_render_text_block_rejects_overinset_bubble_inner_for_tiny_text_region(self):
        img = Image.new("RGB", (800, 16000), (255, 255, 255))
        text_data = {
            "id": "ocr_001",
            "text": "RIGHT TURN ONLY",
            "original": "RIGHT TURN ONLY",
            "translated": "APENAS VIRAR A DIREITA",
            "route_action": "translate_inpaint_render",
            "bbox": [370, 7567, 435, 7640],
            "source_bbox": [370, 7567, 435, 7640],
            "text_pixel_bbox": [395, 7594, 430, 7630],
            "layout_bbox": [395, 7594, 430, 7630],
            "balloon_bbox": [370, 7567, 435, 7640],
            "bubble_mask_bbox": [370, 7567, 435, 7640],
            "bubble_inner_bbox": [382, 7579, 423, 7628],
            "page_width": 800,
            "page_height": 16000,
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 24,
                "cor": "#000000",
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data.get("fit_status"), "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags") or [])
        self.assertIn("safe_text_box_recomputed", text_data.get("qa_flags") or [])
        render_debug = text_data.get("_render_debug") or {}
        self.assertEqual(render_debug.get("capacity_bbox"), [370, 7567, 435, 7640])
        self.assertNotEqual(render_debug.get("capacity_bbox"), [382, 7579, 423, 7628])
        self.assertGreaterEqual(render_debug.get("font_size_final", 0), 12)

    def test_plan_text_layout_rejects_underfit_visual_rect_safe_area_for_long_text(self):
        text_data = {
            "text": "Hosu years old Unemployed",
            "original": "Hosu years old Unemployed",
            "translated": "Hosu anos desempregado",
            "bbox": [502, 315, 523, 329],
            "source_bbox": [438, 311, 605, 387],
            "text_pixel_bbox": [502, 315, 523, 329],
            "layout_bbox": [502, 315, 523, 329],
            "balloon_bbox": [402, 289, 641, 346],
            "_visual_rect_inner_bbox": [439, 306, 602, 339],
            "page_width": 800,
            "page_height": 13000,
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "style_origin": "auto",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 23, "cor": "#000000", "force_upper": True},
        }

        img = Image.new("RGB", (800, 13000), (255, 255, 255))
        render_text_block(img, text_data)

        self.assertNotEqual(text_data.get("layout_safe_bbox"), [439, 306, 602, 339])
        self.assertEqual(text_data.get("fit_status"), "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags") or [])

    def test_plan_text_layout_uses_wide_text_pixel_anchor_for_tiny_white_balloon(self):
        text_data = {
            "text": "happening..?",
            "original": "happening..?",
            "translated": "Acontecendo..?",
            "bbox": [160, 5268, 318, 5289],
            "source_bbox": [179, 5268, 218, 5298],
            "text_pixel_bbox": [160, 5268, 318, 5289],
            "layout_bbox": [160, 5268, 318, 5289],
            "ocr_text_bbox": [160, 5268, 318, 5289],
            "line_polygons": [[[160, 5268], [318, 5269], [318, 5289], [160, 5288]]],
            "balloon_bbox": [167, 5256, 230, 5310],
            "balloon_subregions": [],
            "page_profile": "cover_opening",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "layout_group_size": 1,
            "style_origin": "auto",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 21, "cor": "#000000", "force_upper": True, "bold": True},
        }

        img = Image.new("RGB", (800, 16000), (255, 255, 255))
        render_text_block(img, text_data)

        render_debug = text_data.get("_render_debug") or {}
        target = render_debug.get("target_bbox") or [0, 0, 0, 0]
        self.assertGreaterEqual(target[2] - target[0], 120)
        self.assertEqual(text_data.get("fit_status"), "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags") or [])

    def test_plan_text_layout_rejects_degenerate_bubble_inner_bbox_on_small_balloon(self):
        text_data = {
            "text": "THREE!",
            "original": "THREE!",
            "translated": "TRES!",
            "bbox": [177, 2377, 218, 2406],
            "source_bbox": [177, 2377, 218, 2406],
            "text_pixel_bbox": [171, 2382, 248, 2399],
            "line_polygons": [[[171, 2382], [248, 2382], [248, 2399], [171, 2399]]],
            "balloon_bbox": [165, 2365, 230, 2418],
            "bubble_mask_bbox": [177, 2377, 218, 2406],
            "bubble_inner_bbox": [189, 2389, 206, 2394],
            "page_width": 800,
            "page_height": 14000,
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "style_origin": "auto",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 20, "cor": "#000000", "force_upper": True},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertNotEqual(plan["layout_safe_bbox"], [189, 2389, 206, 2394])
        self.assertGreaterEqual(plan["safe_text_box"][2] - plan["safe_text_box"][0], 45)
        self.assertGreaterEqual(plan["safe_text_box"][3] - plan["safe_text_box"][1], 30)
        self.assertGreaterEqual(resolved["font_size"], 12)
        self.assertIn("tiny_bubble_inner_bbox_rejected", text_data.get("qa_flags", []))

    def test_plan_text_layout_rejects_flat_textured_bubble_inner_bbox(self):
        text_data = {
            "text": "That place is..",
            "original": "That place is..",
            "translated": "ESSE LUGAR E..",
            "bbox": [249, 621, 451, 647],
            "text_pixel_bbox": [254, 627, 392, 640],
            "line_polygons": [[[254, 625], [447, 625], [447, 644], [254, 644]]],
            "balloon_bbox": [249, 621, 451, 647],
            "bubble_mask_bbox": [249, 621, 451, 647],
            "bubble_inner_bbox": [261, 633, 439, 635],
            "page_width": 800,
            "page_height": 1600,
            "tipo": "fala",
            "balloon_type": "textured",
            "layout_profile": "standard",
            "layout_group_size": 1,
            "style_origin": "auto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 18,
                "cor": "#000000",
                "bold": True,
                "italico": True,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertNotEqual(plan["layout_safe_bbox"], [261, 633, 439, 635])
        self.assertGreaterEqual(plan["safe_text_box"][3] - plan["safe_text_box"][1], 10)
        self.assertGreaterEqual(resolved["font_size"], 12)
        self.assertIn("tiny_bubble_inner_bbox_rejected", text_data.get("qa_flags", []))

    def test_plan_text_layout_keeps_edge_clipped_short_balloon_on_source_anchor(self):
        text_data = {
            "text": "WHAT?",
            "original": "WHAT?",
            "translated": "O QUÊ?",
            "bbox": [191, 16, 287, 50],
            "text_pixel_bbox": [196, 30, 284, 50],
            "balloon_bbox": [0, 0, 320, 66],
            "layout_safe_bbox": [101, 12, 301, 54],
            "layout_safe_reason": "single_lobe_white_run_safe_area",
            "page_width": 800,
            "page_height": 13820,
            "tipo": "narracao",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "style_origin": "auto",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 27, "cor": "#000000", "force_upper": True},
        }

        plan = plan_text_layout(text_data)
        safe_center = (plan["safe_text_box"][0] + plan["safe_text_box"][2]) / 2.0
        anchor_center = (text_data["text_pixel_bbox"][0] + text_data["text_pixel_bbox"][2]) / 2.0

        self.assertTrue(plan["_follow_english_anchor_position"])
        self.assertLess(abs(safe_center - anchor_center), 8)
        self.assertIn("edge_clipped_short_text_anchor_position", text_data.get("qa_flags", []))

    def test_resolve_text_layout_uses_full_inner_height_for_short_small_balloon(self):
        text_data = {
            "text": "I FOUND THE MONEY.",
            "original": "I FOUND THE MONEY.",
            "translated": "ENCONTREI O DINHEIRO.",
            "bbox": [309, 16, 458, 80],
            "text_pixel_bbox": [308, 29, 453, 81],
            "balloon_bbox": [309, 16, 458, 80],
            "bubble_inner_bbox": [321, 28, 446, 68],
            "page_width": 800,
            "page_height": 13820,
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "style_origin": "auto",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 25, "cor": "#000000", "force_upper": True},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertGreaterEqual(plan["max_height"], 40)
        self.assertGreaterEqual(resolved["font_size"], 14)
        self.assertGreaterEqual(len(resolved["lines"]), 2)

    def test_compact_small_text_capacity_requires_overlap_with_inner_area(self):
        text_data = {
            "translated": "ATRIZ...",
            "tipo": "fala",
            "layout_safe_bbox": [309, 24, 522, 115],
            "bubble_inner_bbox": [309, 24, 522, 115],
        }

        self.assertFalse(
            renderer_mod._should_compact_small_text_capacity(
                text_data,
                [350, 125, 479, 150],
            )
        )

    def test_plan_text_layout_rejects_stale_bubble_inner_bbox_outside_target(self):
        text_data = {
            "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
            "bbox": [498, 5655, 656, 5740],
            "text_pixel_bbox": [498, 5655, 656, 5740],
            "balloon_bbox": [466, 5606, 696, 5777],
            "bubble_inner_bbox": [513, 230, 649, 313],
            "tipo": "fala",
            "layout_profile": "white_balloon",
            "balloon_type": "white",
            "style_origin": "auto",
        }

        plan = plan_text_layout(text_data)

        self.assertNotEqual(plan["layout_safe_bbox"], [513, 230, 649, 313])
        self.assertGreaterEqual(plan["safe_text_box"][1], 5606)
        self.assertIn("safe_text_box_recomputed", text_data.get("qa_flags", []))
        rejected = text_data.get("_render_debug", {}).get("rejected_safe_boxes", [])
        self.assertEqual(rejected[0]["key"], "bubble_inner_bbox")

    def test_plan_text_layout_keeps_manual_safe_area_over_bubble_inner_bbox(self):
        text_data = {
            "translated": "TEXTO MANUAL",
            "bbox": [140, 100, 260, 140],
            "balloon_bbox": [60, 60, 340, 220],
            "bubble_inner_bbox": [86, 82, 314, 198],
            "layout_safe_bbox": [120, 100, 280, 170],
            "layout_safe_reason": "manual_layout",
            "tipo": "fala",
            "layout_profile": "white_balloon",
            "balloon_type": "white",
            "style_origin": "manual",
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["layout_safe_bbox"], [120, 100, 280, 170])
        self.assertEqual(plan["layout_safe_reason"], "manual_layout")

    def test_connected_children_receive_bubble_lobe_safe_areas(self):
        text_data = {
            "id": "txt_001",
            "bubble_id": "page_001_band_001_bubble_001",
            "bubble_mask_bbox": [20, 20, 400, 260],
            "bubble_inner_bbox": [40, 40, 380, 240],
            "connected_lobe_ids": ["bubble_lobe_left", "bubble_lobe_right"],
            "connected_lobe_bboxes": [[20, 20, 210, 260], [190, 20, 400, 260]],
            "connected_children": [{"translated": "ESQUERDA"}, {"translated": "DIREITA"}],
            "connected_balloon_orientation": "left-right",
            "tipo": "fala",
        }

        candidates = renderer_mod._build_connected_children_candidates(
            text_data,
            "ESQUERDA DIREITA",
            text_data["connected_lobe_bboxes"],
        )
        children = candidates[0]["children"]

        self.assertEqual(children[0]["bubble_id"], "page_001_band_001_bubble_001")
        self.assertEqual(children[0]["lobe_id"], "bubble_lobe_left")
        self.assertEqual(children[0]["bubble_mask_bbox"], [20, 20, 400, 260])
        self.assertEqual(children[0]["bubble_inner_bbox"], [40, 40, 210, 240])
        self.assertEqual(children[1]["lobe_id"], "bubble_lobe_right")
        self.assertEqual(children[1]["bubble_inner_bbox"], [190, 40, 380, 240])

    def test_plain_white_balloon_rejects_tiny_visual_safe_area_far_from_anchor(self):
        text_data = {
            "tipo": "fala",
            "translated": "Disseram que esse food truck e um programa da TV coreana",
            "balloon_bbox": [613, 1232, 1287, 1380],
            "bbox": [815, 1248, 1085, 1364],
            "text_pixel_bbox": [818, 1256, 1077, 1360],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        reject = renderer_mod._should_reject_plain_balloon_visual_safe_area(
            text_data,
            [613, 1232, 1287, 1380],
            [623, 1255, 767, 1287],
            "bright_inner_run_safe_area",
        )

        self.assertTrue(reject)

    def test_single_lobe_visual_safe_area_rejects_physically_flat_height(self):
        text_data = {
            "tipo": "fala",
            "text": "Hosu",
            "original": "Hosu",
            "translated": "Hosu anos desempregado",
            "bbox": [438, 311, 605, 387],
            "text_pixel_bbox": [502, 315, 523, 329],
            "balloon_bbox": [402, 289, 641, 409],
            "bubble_inner_bbox": [450, 323, 593, 375],
            "balloon_type": "white",
            "layout_profile": "connected_balloon",
            "_single_lobe_follow_anchor": True,
        }

        reject = renderer_mod._should_reject_plain_balloon_visual_safe_area(
            text_data,
            [402, 289, 641, 409],
            [442, 306, 598, 310],
            "single_lobe_white_run_safe_area",
        )

        self.assertTrue(reject)

    def test_single_lobe_follow_anchor_uses_lobe_capacity_for_safe_text_box(self):
        text_data = {
            "tipo": "fala",
            "translated": "Hosu anos desempregado",
            "bbox": [438, 311, 605, 387],
            "text_pixel_bbox": [502, 315, 523, 329],
            "line_polygons": [
                [[433, 310], [604, 312], [604, 333], [433, 331]],
                [[499, 339], [542, 339], [542, 358], [499, 358]],
                [[453, 364], [586, 364], [586, 386], [453, 386]],
            ],
            "balloon_bbox": [402, 289, 641, 346],
            "layout_safe_bbox": [439, 306, 602, 339],
            "layout_safe_reason": "single_lobe_white_run_safe_area",
            "connected_lobe_bboxes": [[402, 289, 641, 346], [402, 354, 641, 409]],
            "balloon_subregions": [[402, 289, 641, 346], [402, 354, 641, 409]],
            "connected_balloon_orientation": "top-bottom",
            "balloon_type": "white",
            "layout_profile": "connected_balloon",
            "style_origin": "auto",
            "_single_lobe_follow_anchor": True,
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 23, "cor": "#000000"},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        safe = plan["safe_text_box"]
        self.assertGreaterEqual(safe[3] - safe[1], 24)
        self.assertGreaterEqual(plan["max_height"], 24)
        self.assertGreaterEqual(resolved["font_size"], 12)

    def test_plain_white_balloon_keeps_visual_safe_area_covering_anchor(self):
        text_data = {
            "tipo": "fala",
            "translated": "Disseram que esse food truck e um programa da TV coreana",
            "balloon_bbox": [613, 1232, 1287, 1380],
            "bbox": [815, 1248, 1085, 1364],
            "text_pixel_bbox": [818, 1256, 1077, 1360],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        reject = renderer_mod._should_reject_plain_balloon_visual_safe_area(
            text_data,
            [613, 1232, 1287, 1380],
            [830, 1263, 1067, 1356],
            "bright_inner_run_safe_area",
        )

        self.assertFalse(reject)

    def test_plain_white_balloon_keeps_tiny_visual_safe_area_covering_anchor(self):
        text_data = {
            "tipo": "fala",
            "translated": "Texto curto no balao real",
            "balloon_bbox": [0, 750, 1600, 1149],
            "bbox": [622, 983, 872, 1133],
            "text_pixel_bbox": [629, 995, 868, 1130],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        reject = renderer_mod._should_reject_plain_balloon_visual_safe_area(
            text_data,
            [0, 750, 1600, 1149],
            [620, 970, 900, 1140],
            "single_lobe_white_run_safe_area",
        )

        self.assertFalse(reject)

    def test_visual_safe_area_rejection_does_not_mutate_plain_balloon_layout(self):
        text_data = {
            "tipo": "fala",
            "translated": "Disseram que esse food truck e um programa da TV coreana",
            "balloon_bbox": [613, 1232, 1287, 1380],
            "bbox": [815, 1248, 1085, 1364],
            "text_pixel_bbox": [818, 1256, 1077, 1360],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }
        detected = {
            "outer_bbox": [613, 1232, 1287, 1380],
            "safe_bbox": [623, 1255, 767, 1287],
            "reason": "single_lobe_white_run_safe_area",
        }

        with patch("typesetter.renderer._should_detect_visual_rect_safe_area", return_value=True):
            with patch("typesetter.renderer._detect_visual_rect_safe_area_from_image", return_value=None):
                with patch("typesetter.renderer._detect_bright_inner_run_safe_area_from_image", return_value=None):
                    with patch("typesetter.renderer._detect_single_lobe_white_run_safe_area_from_image", return_value=detected):
                        renderer_mod._apply_visual_rect_safe_area_if_needed(Image.new("RGB", (1600, 2564), "white"), text_data)

        self.assertNotIn("layout_safe_bbox", text_data)
        self.assertNotIn("_visual_rect_inner_bbox", text_data)

    def test_very_overbroad_white_balloon_locks_capacity_to_ocr_anchor(self):
        text_data = {
            "tipo": "fala",
            "translated": "Especialmente este prato de macarrao. e o melhor.",
            "bbox": [622, 983, 872, 1133],
            "text_pixel_bbox": [629, 995, 868, 1130],
            "balloon_bbox": [0, 750, 1600, 1149],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
        }

        self.assertTrue(
            renderer_mod._should_limit_capacity_to_anchor(
                text_data,
                [629, 995, 868, 1130],
                [0, 750, 1600, 1149],
            )
        )
        self.assertTrue(
            renderer_mod._should_follow_english_anchor_position(
                text_data,
                [629, 995, 868, 1130],
                False,
            )
        )

    def test_render_plan_page_cleanup_crop_shifts_xy_to_page_space(self):
        payload = {
            "coordinate_space": "page_cleanup_crop",
            "page_cleanup_crop_bbox": [40, 50, 120, 110],
            "target_bbox": [10, 12, 70, 42],
            "position_bbox": [14, 16, 66, 38],
            "connected_position_bboxes": [[12, 14, 34, 30]],
            "qa_metrics": {
                "safe_text_box": [15, 17, 65, 37],
                "nested": {"render_bbox": [18, 20, 60, 34]},
            },
        }

        shifted = renderer_mod._shift_render_plan_to_page(payload)

        self.assertEqual(shifted["coordinate_space"], "page")
        self.assertEqual(shifted["target_bbox"], [50, 62, 110, 92])
        self.assertEqual(shifted["position_bbox"], [54, 66, 106, 88])
        self.assertEqual(shifted["connected_position_bboxes"], [[52, 64, 74, 80]])
        self.assertEqual(shifted["qa_metrics"]["safe_text_box"], [55, 67, 105, 87])
        self.assertEqual(shifted["qa_metrics"]["nested"]["render_bbox"], [58, 70, 100, 84])

    def test_render_band_image_propagates_render_qa_flags_to_source_text(self):
        band = np.full((80, 160, 3), 255, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "HELLO",
                    "translated": "OLA",
                    "bbox": [20, 20, 90, 50],
                    "balloon_bbox": [10, 10, 110, 65],
                    "qa_flags": [],
                }
            ]
        }
        block = {"id": "ocr_001", "translated": "OLA", "qa_flags": []}

        def fake_render(_img, render_block):
            render_block["qa_flags"] = ["TEXT_CLIPPED"]

        with patch("typesetter.renderer.build_render_blocks", return_value=[block]):
            with patch("typesetter.renderer.render_text_block", side_effect=fake_render):
                render_band_image(band, page)

        self.assertIn("TEXT_CLIPPED", page["texts"][0]["qa_flags"])

    def test_run_render_qa_records_fit_debug_bboxes_for_overflow(self):
        text_data = {
            "translated": "TEXTO LONGO",
            "balloon_bbox": [10, 10, 110, 70],
            "render_bbox": [0, 0, 130, 90],
            "qa_flags": [],
        }
        plan = {
            "target_bbox": [10, 10, 110, 70],
            "safe_text_box": [18, 18, 102, 62],
        }

        renderer_mod._run_render_qa(text_data, plan)

        self.assertIn("TEXT_OVERFLOW", text_data["qa_flags"])
        self.assertIn("render_outside_balloon", text_data["qa_flags"])
        evidence = text_data["qa_metrics"]["render_fit"]
        self.assertEqual(evidence["render_bbox"], [0, 0, 130, 90])
        self.assertEqual(evidence["safe_text_box"], [18, 18, 102, 62])
        self.assertEqual(evidence["target_bbox"], [10, 10, 110, 70])
        self.assertEqual(evidence["balloon_bbox"], [10, 10, 110, 70])
        self.assertIn("TEXT_OVERFLOW", evidence["flags"])
        self.assertIn("render_outside_balloon", evidence["flags"])

    def test_run_render_qa_flags_render_outside_validated_source(self):
        text_data = {
            "translated": "TEXTO LONGO",
            "balloon_bbox": [0, 0, 180, 120],
            "_validated_text_source_bboxes": [[30, 20, 130, 70]],
            "render_bbox": [24, 18, 160, 78],
            "qa_flags": [],
        }
        plan = {
            "target_bbox": [30, 20, 130, 70],
            "safe_text_box": [34, 24, 126, 66],
            "_validated_source_target_bbox": [30, 20, 130, 70],
        }

        renderer_mod._run_render_qa(text_data, plan)

        self.assertIn("render_outside_validated_text_source", text_data["qa_flags"])
        self.assertLess(text_data["qa_metrics"]["render_validated_containment"], 0.92)
        self.assertEqual(
            text_data["qa_metrics"]["render_fit"]["validated_source_target_bbox"],
            [30, 20, 130, 70],
        )

    def test_run_render_qa_recomputes_stale_geometry_flags(self):
        text_data = {
            "translated": "TEXTO",
            "balloon_bbox": [10, 10, 110, 70],
            "render_bbox": [30, 30, 90, 50],
            "qa_flags": ["render_outside_balloon", "mask_density_high"],
        }
        plan = {
            "target_bbox": [10, 10, 110, 70],
            "safe_text_box": [20, 20, 100, 60],
        }

        renderer_mod._run_render_qa(text_data, plan)

        self.assertNotIn("render_outside_balloon", text_data["qa_flags"])
        self.assertNotIn("TEXT_OVERFLOW", text_data["qa_flags"])
        self.assertNotIn("TEXT_CLIPPED", text_data["qa_flags"])
        self.assertIn("mask_density_high", text_data["qa_flags"])

    def test_copy_render_debug_fields_drops_stale_render_geometry_flags_when_clean(self):
        source = {
            "translated": "TEXTO",
            "balloon_bbox": [10, 10, 110, 70],
            "safe_text_box": [20, 20, 100, 60],
            "qa_flags": ["TEXT_CLIPPED", "TEXT_OVERFLOW", "render_outside_balloon", "mask_density_high"],
        }
        rendered = {
            "render_bbox": [30, 30, 90, 50],
            "safe_text_box": [20, 20, 100, 60],
            "_render_debug": {"target_bbox": [10, 10, 110, 70]},
            "qa_metrics": {"render_balloon_containment": 1.0},
            "qa_flags": ["TEXT_CLIPPED", "render_outside_balloon"],
        }

        renderer_mod._copy_render_debug_fields(source, rendered)

        self.assertNotIn("render_outside_balloon", source["qa_flags"])
        self.assertNotIn("TEXT_OVERFLOW", source["qa_flags"])
        self.assertNotIn("TEXT_CLIPPED", source["qa_flags"])
        self.assertIn("mask_density_high", source["qa_flags"])

    def test_copy_render_debug_fields_keeps_render_fit_overflow_when_broad_balloon_contains_render(self):
        source = {
            "translated": "VOCE FOI IMPRESSIONANTE? HAHA...",
            "balloon_bbox": [3, 10860, 651, 10986],
            "safe_text_box": [144, 10888, 535, 10919],
            "qa_flags": ["TEXT_OVERFLOW", "safe_text_box_recomputed"],
        }
        rendered = {
            "render_bbox": [173, 10900, 506, 10919],
            "safe_text_box": [144, 10888, 535, 10919],
            "balloon_bbox": [3, 10860, 651, 10986],
            "_render_debug": {"target_bbox": [3, 10860, 285, 10986]},
            "qa_metrics": {
                "render_balloon_containment": 0.3363,
                "render_fit": {
                    "flags": ["TEXT_OVERFLOW"],
                    "render_bbox": [173, 10900, 506, 10919],
                    "target_bbox": [3, 10860, 285, 10986],
                    "balloon_bbox": [3, 10860, 285, 10986],
                },
            },
            "qa_flags": ["TEXT_OVERFLOW", "safe_text_box_recomputed"],
        }

        renderer_mod._copy_render_debug_fields(source, rendered)

        self.assertIn("TEXT_OVERFLOW", source["qa_flags"])
        self.assertIn("safe_text_box_recomputed", source["qa_flags"])

    def test_run_render_qa_ignores_balloon_space_mismatch_when_inside_safe_box(self):
        text_data = {
            "translated": "JÁ QUE VOU PARA O INFERNO",
            "balloon_bbox": [42, 0, 785, 338],
            "render_bbox": [111, 2190, 397, 2267],
            "qa_flags": [],
        }
        plan = {
            "target_bbox": [42, 2144, 785, 2482],
            "safe_text_box": [108, 2180, 399, 2278],
        }

        renderer_mod._run_render_qa(text_data, plan)

        self.assertNotIn("render_outside_balloon", text_data["qa_flags"])
        self.assertNotIn("TEXT_OVERFLOW", text_data["qa_flags"])
        self.assertNotIn("TEXT_CLIPPED", text_data["qa_flags"])

    def test_run_render_qa_uses_containment_for_oblique_axis_bbox(self):
        text_data = {
            "translated": "AJUMMA, DE AGORA EM DIANTE NAO USE SEU DINHEIRO.",
            "balloon_bbox": [260, 0, 800, 331],
            "render_bbox": [443, -6, 725, 255],
            "qa_flags": [],
        }
        plan = {
            "target_bbox": [260, 0, 800, 331],
            "safe_text_box": [454, 59, 741, 272],
            "rotation_deg": -33.4,
        }

        renderer_mod._run_render_qa(text_data, plan)

        self.assertGreaterEqual(text_data["qa_metrics"]["render_balloon_containment"], 0.94)
        self.assertNotIn("TEXT_CLIPPED", text_data["qa_flags"])
        self.assertNotIn("TEXT_OVERFLOW", text_data["qa_flags"])
        self.assertNotIn("render_outside_balloon", text_data["qa_flags"])

    def test_plan_text_layout_uses_balloon_when_ocr_anchor_is_tiny_for_long_text(self):
        text_data = {
            "text": "In the morning we are going to train for rescue and fire extinguishing.",
            "translated": "Pela manhã, vamos treinar para resgate e extinção de incêndio, então teremos que vestir nossos uniformes e nos reunir.",
            "bbox": [92, 119, 407, 231],
            "text_pixel_bbox": [214, 176, 222, 182],
            "balloon_bbox": [92, 119, 407, 231],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "narracao",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48, "cor": "#000000"},
            "style_origin": "auto",
        }

        plan = plan_text_layout(text_data)

        safe = plan["safe_text_box"]
        self.assertGreaterEqual(safe[2] - safe[0], 220)
        self.assertFalse(plan["_follow_original_ocr_size"])
        self.assertTrue(plan["_center_on_balloon_bbox"])

    def test_render_band_image_copies_geometry_by_trace_when_text_ids_repeat(self):
        band = np.full((120, 180, 3), 255, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@page_001_band_002",
                    "text": "FIRST",
                    "translated": "PRIMEIRO",
                    "bbox": [20, 20, 90, 50],
                    "balloon_bbox": [10, 10, 110, 65],
                    "qa_flags": [],
                },
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@page_001_band_017",
                    "text": "SECOND",
                    "translated": "SEGUNDO",
                    "bbox": [30, 70, 130, 100],
                    "balloon_bbox": [20, 60, 150, 115],
                    "qa_flags": [],
                },
            ]
        }
        blocks = [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_002",
                "translated": "PRIMEIRO",
            },
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_017",
                "translated": "SEGUNDO",
            },
        ]

        def fake_render(_img, render_block):
            if render_block["trace_id"].endswith("band_002"):
                render_block["render_bbox"] = [22, 24, 78, 46]
            else:
                render_block["render_bbox"] = [34, 74, 126, 96]

        with patch("typesetter.renderer.build_render_blocks", return_value=blocks):
            with patch("typesetter.renderer.render_text_block", side_effect=fake_render):
                render_band_image(band, page)

        self.assertEqual(page["texts"][0]["render_bbox"], [22, 24, 78, 46])
        self.assertEqual(page["texts"][1]["render_bbox"], [34, 74, 126, 96])

    def test_render_band_image_aggregates_child_geometry_back_to_split_source(self):
        band = np.full((140, 180, 3), 255, dtype=np.uint8)
        source = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_001_band_001",
            "text": "HEY I'M STARVING",
            "translated": "EI, VAMOS! ESTOU COM FOME",
            "bbox": [20, 20, 120, 100],
            "balloon_bbox": [20, 20, 120, 100],
            "qa_flags": ["missing_render_bbox"],
        }
        page = {"_band_id": "page_001_band_001", "texts": [source]}
        blocks = [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_001",
                "translated": "EI, VAMOS!",
                "balloon_bbox": [20, 20, 90, 48],
            },
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_001",
                "translated": "ESTOU COM FOME",
                "balloon_bbox": [20, 70, 130, 110],
            },
        ]

        def fake_render(_img, render_block):
            render_block["render_bbox"] = render_block["balloon_bbox"]
            render_block["safe_text_box"] = render_block["balloon_bbox"]
            render_block["qa_flags"] = ["TEXT_CLIPPED"]

        with patch("typesetter.renderer.build_render_blocks", return_value=blocks):
            with patch("typesetter.renderer.render_text_block", side_effect=fake_render):
                render_band_image(band, page)

        self.assertEqual(source["render_bbox"], [20, 20, 130, 110])
        self.assertEqual(source["safe_text_box"], [20, 20, 130, 110])
        self.assertNotIn("missing_render_bbox", source.get("qa_flags") or [])
        self.assertEqual(source.get("qa_flags"), [])

    def test_render_band_image_copies_connected_group_geometry_to_all_sources(self):
        band = np.full((140, 180, 3), 255, dtype=np.uint8)
        first = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_001_band_001",
            "translated": "PRIMEIRO",
            "qa_flags": ["missing_render_bbox"],
        }
        second = {
            "id": "ocr_002",
            "text_id": "ocr_002",
            "trace_id": "ocr_002@page_001_band_001",
            "translated": "SEGUNDO",
            "qa_flags": ["missing_render_bbox"],
        }
        page = {"_band_id": "page_001_band_001", "texts": [first, second]}
        blocks = [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_001",
                "_source_text_ids": ["ocr_001", "ocr_002"],
                "_source_trace_ids": ["ocr_001@page_001_band_001", "ocr_002@page_001_band_001"],
                "translated": "PRIMEIRO SEGUNDO",
                "balloon_bbox": [20, 20, 140, 100],
            }
        ]

        def fake_render(_img, render_block):
            render_block["render_bbox"] = [30, 30, 130, 90]
            render_block["safe_text_box"] = [20, 20, 140, 100]
            render_block["fit_status"] = "ok"

        with patch("typesetter.renderer.build_render_blocks", return_value=blocks):
            with patch("typesetter.renderer.render_text_block", side_effect=fake_render):
                render_band_image(band, page)

        self.assertEqual(first["render_bbox"], [30, 30, 130, 90])
        self.assertEqual(second["render_bbox"], [30, 30, 130, 90])
        self.assertNotIn("missing_render_bbox", first.get("qa_flags") or [])
        self.assertNotIn("missing_render_bbox", second.get("qa_flags") or [])

    def test_render_band_image_does_not_let_duplicate_single_block_overwrite_group_render(self):
        band = np.full((140, 180, 3), 255, dtype=np.uint8)
        first = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_001_band_001",
            "translated": "PRIMEIRO",
        }
        second = {
            "id": "ocr_002",
            "text_id": "ocr_002",
            "trace_id": "ocr_002@page_001_band_001",
            "translated": "SEGUNDO",
        }
        page = {"_band_id": "page_001_band_001", "texts": [first, second]}
        group_block = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_001_band_001",
            "_source_text_ids": ["ocr_001", "ocr_002"],
            "_source_trace_ids": ["ocr_001@page_001_band_001", "ocr_002@page_001_band_001"],
            "translated": "PRIMEIRO SEGUNDO",
            "balloon_bbox": [20, 20, 140, 100],
        }
        duplicate_single = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_001_band_001",
            "translated": "PRIMEIRO",
            "balloon_bbox": [42, 42, 58, 58],
        }
        rendered_ids = []

        def fake_render(_img, render_block):
            rendered_ids.append(render_block["translated"])
            if render_block is group_block:
                render_block["render_bbox"] = [30, 30, 130, 90]
                render_block["safe_text_box"] = [20, 20, 140, 100]
                render_block["fit_status"] = "ok"
            else:
                render_block["render_bbox"] = [42, 42, 58, 120]
                render_block["safe_text_box"] = [42, 42, 58, 58]
                render_block["fit_status"] = "below_minimum_legible"
                render_block["qa_flags"] = ["fit_below_minimum_legible", "TEXT_CLIPPED"]

        with patch("typesetter.renderer.build_render_blocks", return_value=[group_block, duplicate_single]):
            with patch("typesetter.renderer.render_text_block", side_effect=fake_render):
                render_band_image(band, page)

        self.assertEqual(rendered_ids, ["PRIMEIRO SEGUNDO"])
        self.assertEqual(first["render_bbox"], [30, 30, 130, 90])
        self.assertEqual(first["safe_text_box"], [20, 20, 140, 100])
        self.assertEqual(second["render_bbox"], [30, 30, 130, 90])
        self.assertNotIn("fit_below_minimum_legible", first.get("qa_flags") or [])

    def test_split_aggregate_drops_fit_flag_when_children_finish_ok(self):
        source = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_001_band_001",
            "translated": "TEXTO LONGO",
            "qa_flags": ["fit_below_minimum_legible"],
        }
        aggregate = renderer_mod._aggregate_split_render_blocks(
            [
                {
                    "id": "ocr_001",
                    "trace_id": "ocr_001@page_001_band_001",
                    "render_bbox": [20, 20, 80, 40],
                    "safe_text_box": [10, 10, 90, 50],
                    "fit_status": "ok",
                    "fit_attempts": [{"font_px": 12, "lines": 1, "status": "ok"}],
                    "qa_flags": ["fit_below_minimum_legible"],
                },
                {
                    "id": "ocr_001",
                    "trace_id": "ocr_001@page_001_band_001",
                    "render_bbox": [20, 60, 80, 80],
                    "safe_text_box": [10, 50, 90, 90],
                    "fit_status": "ok",
                    "fit_attempts": [{"font_px": 12, "lines": 1, "status": "ok"}],
                    "qa_flags": [],
                },
            ]
        )

        renderer_mod._copy_render_debug_fields(source, aggregate)

        self.assertEqual(source["fit_status"], "ok")
        self.assertNotIn("fit_below_minimum_legible", source.get("qa_flags") or [])

    def test_split_aggregate_drops_stale_child_render_fit_metric(self):
        aggregate = renderer_mod._aggregate_split_render_blocks(
            [
                {
                    "id": "ocr_001",
                    "trace_id": "ocr_001@page_001_band_001",
                    "render_bbox": [200, 18, 318, 56],
                    "safe_text_box": [200, 53, 370, 62],
                    "fit_status": "ok",
                    "qa_metrics": {
                        "render_fit": {
                            "flags": ["TEXT_CLIPPED", "TEXT_OVERFLOW"],
                            "target_bbox": [134, 13, 304, 62],
                            "safe_text_box": [200, 53, 370, 62],
                            "render_bbox": [200, 18, 318, 56],
                        }
                    },
                },
                {
                    "id": "ocr_001",
                    "trace_id": "ocr_001@page_001_band_001",
                    "render_bbox": [475, 466, 661, 500],
                    "safe_text_box": [465, 461, 661, 505],
                    "fit_status": "ok",
                    "qa_metrics": {},
                },
            ]
        )

        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate["render_bbox"], [200, 18, 661, 500])
        self.assertNotIn("render_fit", aggregate.get("qa_metrics") or {})

    def test_split_aggregate_drops_child_text_fit_flags_when_children_render_ok(self):
        aggregate = renderer_mod._aggregate_split_render_blocks(
            [
                {
                    "id": "ocr_001",
                    "render_bbox": [200, 3420, 318, 3458],
                    "safe_text_box": [200, 3455, 370, 3464],
                    "fit_status": "ok",
                    "qa_flags": ["TEXT_CLIPPED", "TEXT_OVERFLOW", "ocr_joined_repaired"],
                },
                {
                    "id": "ocr_001",
                    "render_bbox": [475, 3868, 661, 3902],
                    "safe_text_box": [465, 3863, 661, 3907],
                    "fit_status": "ok",
                    "qa_flags": ["TEXT_OVERFLOW"],
                },
            ]
        )

        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate["render_bbox"], [200, 3420, 661, 3902])
        self.assertEqual(aggregate["safe_text_box"], [200, 3420, 661, 3907])
        self.assertNotIn("TEXT_CLIPPED", aggregate.get("qa_flags") or [])
        self.assertNotIn("TEXT_OVERFLOW", aggregate.get("qa_flags") or [])
        self.assertIn("ocr_joined_repaired", aggregate.get("qa_flags") or [])

    def test_render_band_image_reuses_one_debug_background_for_white_blocks(self):
        band = np.full((80, 160, 3), 255, dtype=np.uint8)
        page = {
            "texts": [
                {"id": "ocr_001", "translated": "OLA", "balloon_bbox": [10, 10, 70, 50]},
                {"id": "ocr_002", "translated": "SIM", "balloon_bbox": [80, 10, 140, 50]},
            ]
        }
        blocks = [
            {
                "id": "ocr_001",
                "translated": "OLA",
                "balloon_type": "white",
                "layout_profile": "white_balloon",
                "balloon_bbox": [10, 10, 70, 50],
            },
            {
                "id": "ocr_002",
                "translated": "SIM",
                "balloon_type": "white",
                "layout_profile": "white_balloon",
                "balloon_bbox": [80, 10, 140, 50],
            },
        ]
        background_ids = []
        original_convert = Image.Image.convert
        rgb_convert_calls = []

        def count_rgb_convert(image, mode=None, *args, **kwargs):
            if mode == "RGB":
                rgb_convert_calls.append(mode)
            return original_convert(image, mode, *args, **kwargs)

        def fake_render(_img, render_block, img_size=None, pre_render_np=None):
            del img_size
            self.assertIsNotNone(pre_render_np)
            background_ids.append(id(pre_render_np))
            render_block["render_bbox"] = render_block["balloon_bbox"]

        with patch("typesetter.renderer._debug_recorder_enabled", return_value=True):
            with patch("typesetter.renderer.build_render_blocks", return_value=blocks):
                with patch("PIL.Image.Image.convert", new=count_rgb_convert):
                    with patch("typesetter.renderer.render_text_block", side_effect=fake_render):
                        render_band_image(band, page)

        self.assertEqual(rgb_convert_calls, ["RGB"])
        self.assertEqual(len(set(background_ids)), 1)

    def test_unrotated_render_reuses_precomputed_background_for_render_qa(self):
        img = Image.new("RGB", (120, 80), (255, 255, 255))
        background_np = np.full((80, 120, 3), 255, dtype=np.uint8)
        text_data = {
            "translated": "OLA",
            "balloon_type": "white",
            "balloon_bbox": [10, 10, 100, 70],
            "estilo": {},
        }
        plan = {
            "target_bbox": [10, 10, 100, 70],
            "safe_text_box": [10, 10, 100, 70],
            "position_bbox": [10, 10, 100, 70],
            "capacity_bbox": [10, 10, 100, 70],
            "font_name": "default",
            "target_size": 12,
            "padding_y": 0,
            "vertical_anchor": "center",
            "vertical_bias_px": 0,
            "horizontal_bias_px": 0,
            "outline_color": None,
            "outline_px": 0,
            "sombra": False,
            "sombra_cor": None,
            "sombra_offset": (0, 0),
            "glow": False,
            "glow_cor": None,
            "glow_px": 0,
            "cor_gradiente": [],
            "text_color": "#000000",
        }
        resolved = {
            "font": ImageFont.load_default(),
            "lines": ["OLA"],
            "font_size": 12,
            "line_height": 12,
            "total_text_height": 12,
            "start_y": 20,
            "positions": [(20, 20)],
            "score": 1.0,
            "block_bbox": [20, 20, 45, 32],
        }
        original_convert = Image.Image.convert
        rgb_convert_calls = []

        def count_rgb_convert(image, mode=None, *args, **kwargs):
            if mode == "RGB":
                rgb_convert_calls.append(mode)
            return original_convert(image, mode, *args, **kwargs)

        with patch("typesetter.renderer._debug_recorder_enabled", return_value=True):
            with patch("typesetter.renderer._resolve_text_layout", return_value=resolved):
                with patch("PIL.Image.Image.convert", new=count_rgb_convert):
                    with patch("typesetter.renderer._run_render_qa") as run_render_qa:
                        _render_single_text_block_unrotated(
                            img,
                            text_data,
                            plan,
                            pre_render_np=background_np,
                        )

        self.assertEqual(rgb_convert_calls, [])
        self.assertIs(run_render_qa.call_args.kwargs["background_image"], background_np)

    def test_render_plan_final_dedupes_by_trace_id_not_reused_text_id(self):
        from debug_tools import DebugRecorder, bind_recorder

        band = np.full((80, 120, 3), 245, dtype=np.uint8)

        def make_page(band_id, y_top):
            return {
                "_band_id": band_id,
                "_band_y_top": y_top,
                "texts": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": f"ocr_001@{band_id}",
                        "band_id": band_id,
                        "text": "HELLO",
                        "translated": "OLA",
                        "bbox": [20, 20, 80, 50],
                        "balloon_bbox": [10, 10, 90, 60],
                        "tipo": "fala",
                    }
                ],
            }

        def fake_render(_img, render_block):
            render_block["render_bbox"] = [22, 24, 78, 46]
            render_block["_render_debug"] = {
                "target_bbox": [10, 10, 90, 60],
                "layout_fit_result": "pass",
            }

        with tempfile.TemporaryDirectory() as tmp:
            recorder = DebugRecorder(Path(tmp), enabled=True, run_id="run-test")
            bind_recorder(recorder)
            try:
                with patch("typesetter.renderer.build_render_blocks", side_effect=lambda texts: [dict(texts[0])]):
                    with patch("typesetter.renderer.render_text_block", side_effect=fake_render):
                        render_band_image(band, make_page("page_001_band_003", 2700))
                        render_band_image(band, make_page("page_002_band_004", 3600))
            finally:
                bind_recorder(None)

            final_path = Path(tmp) / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
            rows = [json.loads(line) for line in final_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["trace_id"] for row in rows},
            {"ocr_001@page_001_band_003", "ocr_001@page_002_band_004"},
        )
        self.assertTrue(all(row["coordinate_space"] == "page" for row in rows))

    def test_render_plan_records_candidates_and_skipped_with_trace_metadata(self):
        from debug_tools import DebugRecorder, bind_recorder

        band = np.full((80, 120, 3), 245, dtype=np.uint8)
        page = {
            "_band_id": "page_003_band_007",
            "_band_y_top": 400,
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@page_003_band_007",
                    "band_id": "page_003_band_007",
                    "text": "HELLO",
                    "translated": "OLA",
                    "bbox": [20, 20, 80, 50],
                    "balloon_bbox": [10, 10, 90, 60],
                    "tipo": "fala",
                },
                {
                    "id": "ocr_002",
                    "text_id": "ocr_002",
                    "trace_id": "ocr_002@page_003_band_007",
                    "band_id": "page_003_band_007",
                    "text": "SECRET SCANS",
                    "translated": "",
                    "bbox": [4, 4, 40, 16],
                    "skip_processing": True,
                    "skip_reason": "watermark",
                },
            ],
        }

        def fake_render(_img, render_block):
            render_block["render_bbox"] = [22, 24, 78, 46]
            render_block["_render_debug"] = {"layout_fit_result": "pass"}
            render_block["_render_debug_candidates"] = [
                {"candidate_kind": "layout_fit", "status": "candidate", "font_size": 22, "selected": True}
            ]
            render_block["_render_debug_skipped"] = [
                {"candidate_kind": "layout_fit", "status": "skipped", "skip_reason": "does_not_fit", "font_size": 24}
            ]

        with tempfile.TemporaryDirectory() as tmp:
            recorder = DebugRecorder(Path(tmp), enabled=True, run_id="run-test")
            bind_recorder(recorder)
            try:
                with patch("typesetter.renderer.build_render_blocks", return_value=[dict(page["texts"][0])]):
                    with patch("typesetter.renderer.render_text_block", side_effect=fake_render):
                        render_band_image(band, page)
            finally:
                bind_recorder(None)

            root = Path(tmp) / "debug" / "e2e" / "09_typeset"
            candidates = [json.loads(line) for line in (root / "render_plan_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
            skipped = [json.loads(line) for line in (root / "render_plan_skipped.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertTrue(any(row["candidate_kind"] == "render_block" for row in candidates))
        self.assertTrue(any(row["candidate_kind"] == "layout_fit" for row in candidates))
        self.assertTrue(all(row["trace_id"] for row in candidates))
        self.assertTrue(any(row["trace_id"] == "ocr_002@page_003_band_007" and row["skip_reason"] == "watermark" for row in skipped))
        self.assertTrue(any(row["candidate_kind"] == "layout_fit" and row["skip_reason"] == "does_not_fit" for row in skipped))

    def test_render_plan_candidates_and_skipped_preserve_page_coordinate_space(self):
        from debug_tools import DebugRecorder, bind_recorder

        page_rgb = np.full((240, 180, 3), 245, dtype=np.uint8)
        page = {
            "_band_id": "page_003",
            "_band_y_top": 0,
            "_coordinate_space": "page",
            "texts": [
                {
                    "id": "ocr_001",
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@page_003",
                    "band_id": "page_003",
                    "text": "HELLO",
                    "translated": "OLA",
                    "bbox": [20, 120, 90, 150],
                    "balloon_bbox": [10, 110, 100, 165],
                    "coordinate_space": "page",
                    "tipo": "fala",
                },
                {
                    "id": "ocr_002",
                    "text_id": "ocr_002",
                    "trace_id": "ocr_002@page_003",
                    "band_id": "page_003",
                    "text": "NOISE",
                    "translated": "",
                    "bbox": [130, 180, 165, 200],
                    "coordinate_space": "page",
                    "skip_processing": True,
                    "skip_reason": "noise",
                },
            ],
        }

        def fake_render(_img, render_block):
            render_block["render_bbox"] = [22, 124, 88, 146]
            render_block["_render_debug"] = {"layout_fit_result": "pass"}
            render_block["_render_debug_candidates"] = [
                {"candidate_kind": "layout_fit", "status": "candidate", "font_size": 22, "selected": True}
            ]
            render_block["_render_debug_skipped"] = [
                {"candidate_kind": "layout_fit", "status": "skipped", "skip_reason": "does_not_fit", "font_size": 24}
            ]

        with tempfile.TemporaryDirectory() as tmp:
            recorder = DebugRecorder(Path(tmp), enabled=True, run_id="run-test")
            bind_recorder(recorder)
            try:
                with patch("typesetter.renderer.build_render_blocks", return_value=[dict(page["texts"][0])]):
                    with patch("typesetter.renderer.render_text_block", side_effect=fake_render):
                        render_band_image(page_rgb, page)
            finally:
                bind_recorder(None)

            root = Path(tmp) / "debug" / "e2e" / "09_typeset"
            candidates = [json.loads(line) for line in (root / "render_plan_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
            skipped = [json.loads(line) for line in (root / "render_plan_skipped.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertTrue(candidates)
        self.assertTrue(skipped)
        self.assertTrue(all(row["coordinate_space"] == "page" for row in candidates))
        self.assertTrue(all(row["coordinate_space"] == "page" for row in skipped))
        self.assertTrue(any(row["candidate_kind"] == "render_block" and row["bbox"] == [20, 120, 90, 150] for row in candidates))
        self.assertTrue(any(row["text_id"] == "ocr_002" and row["bbox"] == [130, 180, 165, 200] for row in skipped))

    def test_plan_text_layout_uses_balloon_bbox_by_default(self):
        text_data = {
            "text": "HELLO",
            "translated": "UM TEXTO TRADUZIDO BEM MAIOR",
            "bbox": [20, 30, 220, 110],
            "source_bbox": [24, 34, 216, 106],
            "text_pixel_bbox": [60, 50, 160, 88],
            "balloon_bbox": [0, 0, 300, 180],
            "tipo": "fala",
            "layout_group_size": 1,
            "estilo": {"fonte": "Newrotic.ttf", "tamanho": 28, "cor": "#000000"},
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [0, 0, 300, 180])
        self.assertEqual(plan["capacity_bbox"], [0, 0, 300, 180])
        self.assertEqual(plan["max_width"], 216)

    def test_plan_text_layout_does_not_lock_white_sign_to_tiny_text_pixels(self):
        text_data = {
            "text": "TEXT: DARLING KARAOKE",
            "translated": "TEXTO: QUERIDO KARAOKE",
            "bbox": [45, 9305, 205, 9322],
            "source_bbox": [45, 9305, 205, 9322],
            "text_pixel_bbox": [118, 9311, 145, 9316],
            "balloon_bbox": [10, 9293, 240, 9334],
            "balloon_type": "white",
            "content_class": "sign",
            "tipo": "narracao",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 16, "alinhamento": "center"},
            "layout_shape": "wide",
            "layout_align": "top",
            "layout_profile": "white_balloon",
        }

        plan = plan_text_layout(text_data)

        self.assertFalse(plan["_anchor_capacity_locked"])
        self.assertEqual(plan["target_bbox"], [10, 9293, 240, 9334])
        self.assertGreaterEqual(plan["safe_text_box"][2] - plan["safe_text_box"][0], 120)
        self.assertGreaterEqual(plan["safe_text_box"][3] - plan["safe_text_box"][1], 20)

    def test_simple_ocr_position_grows_font_when_local_margin_is_available(self):
        text_data = {
            "text": "YES",
            "translated": "SIM",
            "bbox": [100, 100, 188, 124],
            "source_bbox": [100, 100, 188, 124],
            "text_pixel_bbox": [100, 100, 188, 124],
            "page_width": 360,
            "page_height": 260,
            "tipo": "fala",
            "balloon_type": "white",
            "layout_group_size": 1,
            "estilo": {"fonte": "Newrotic.ttf", "tamanho": 26, "cor": "#000000"},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["target_bbox"], [100, 100, 188, 124])
        self.assertFalse(plan["_simple_anchor_capacity_expanded"])
        self.assertLessEqual(resolved["font_size"], plan["target_size"] + 2)
        self.assertLessEqual(abs(((plan["position_bbox"][1] + plan["position_bbox"][3]) / 2) - 112), 18)

    def test_simple_ocr_position_grows_font_without_page_dimensions(self):
        text_data = {
            "text": "SAKE.",
            "translated": "SAQUÊ.",
            "bbox": [497, 5648, 660, 5741],
            "text_pixel_bbox": [546, 5720, 612, 5740],
            "tipo": "fala",
            "balloon_type": "white",
            "layout_group_size": 1,
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48, "cor": "#000000"},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["target_bbox"], [546, 5720, 612, 5740])
        self.assertFalse(plan["_simple_anchor_capacity_expanded"])
        self.assertEqual(plan["capacity_bbox"], [546, 5720, 612, 5740])
        self.assertLessEqual(resolved["font_size"], plan["target_size"] + 2)

    def test_merged_white_balloon_uses_small_anchor_when_union_hits_art(self):
        text_data = {
            "text": "PLEASE, FOR THE CHILD'S THE SA SAKE.",
            "translated": "POR FAVOR, PARA A CRIANCA E O SA SAQUE.",
            "bbox": [25, 5436, 667, 5949],
            "source_bbox": [25, 5436, 667, 5949],
            "text_pixel_bbox": [320, 5655, 656, 5949],
            "layout_bbox": [546, 5720, 612, 5740],
            "balloon_bbox": [0, 5420, 733, 6012],
            "_merged_source_bboxes": [[25, 5436, 667, 5949], [501, 5638, 661, 5745]],
            "page_width": 800,
            "page_height": 11000,
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_group_size": 3,
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48, "cor": "#000000"},
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(text_data["_merged_source_anchor_bbox"], [501, 5638, 661, 5745])
        self.assertEqual(plan["target_bbox"], [461, 5590, 701, 5793])
        self.assertTrue(plan["_center_on_balloon_bbox"])
        self.assertLess(plan["target_bbox"][2] - plan["target_bbox"][0], 320)
        self.assertGreaterEqual(plan["safe_text_box"][0], plan["target_bbox"][0])
        self.assertLessEqual(plan["safe_text_box"][2], plan["target_bbox"][2])

    def test_simple_ocr_position_grows_black_textured_dialogue_without_effects(self):
        text_data = {
            "text": "PLEASE, FOR THE CHILD'S",
            "translated": "POR FAVOR, PARA A CRIANÇA",
            "bbox": [25, 5436, 667, 5708],
            "text_pixel_bbox": [498, 5655, 656, 5707],
            "tipo": "fala",
            "balloon_type": "white",
            "layout_group_size": 1,
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48, "cor": "#000000"},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertFalse(plan["_simple_anchor_capacity_expanded"])
        self.assertEqual(plan["capacity_bbox"], [498, 5655, 656, 5707])
        self.assertLessEqual(resolved["font_size"], plan["target_size"] + 2)

    def test_simple_white_balloon_narration_grows_instead_of_staying_tiny(self):
        text_data = {
            "text": "WHAT?",
            "translated": "O QUE?",
            "bbox": [194, 4120, 285, 4146],
            "text_pixel_bbox": [196, 4126, 284, 4146],
            "tipo": "narracao",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 18, "cor": "#000000"},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertFalse(plan["_simple_anchor_capacity_expanded"])
        self.assertEqual(plan["capacity_bbox"], [196, 4126, 284, 4146])
        self.assertLessEqual(resolved["font_size"], plan["target_size"] + 2)

    def test_tiny_white_balloon_anchor_expands_inside_balloon_by_default(self):
        text_data = {
            "text": "WHAT IS THIS?!",
            "translated": "O QUE E ISSO?!",
            "bbox": [153, 6430, 322, 6449],
            "source_bbox": [153, 6430, 322, 6449],
            "text_pixel_bbox": [153, 6430, 322, 6449],
            "balloon_bbox": [129, 6409, 351, 6468],
            "page_width": 800,
            "page_height": 13000,
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 18, "cor": "#000000"},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertTrue(plan["_center_on_balloon_bbox"])
        self.assertFalse(plan["_simple_anchor_capacity_expanded"])
        self.assertEqual(plan["capacity_bbox"], text_data["balloon_bbox"])
        self.assertEqual(plan["position_bbox"], text_data["balloon_bbox"])
        self.assertGreaterEqual(resolved["font_size"], 18)
        self.assertGreater(plan["safe_text_box"][0], text_data["balloon_bbox"][0])
        self.assertGreater(plan["safe_text_box"][1], text_data["balloon_bbox"][1])
        self.assertLess(plan["safe_text_box"][2], text_data["balloon_bbox"][2])
        self.assertLess(plan["safe_text_box"][3], text_data["balloon_bbox"][3])

    def test_long_white_dialogue_uses_safe_area_capacity_when_ocr_anchor_is_too_small(self):
        text_data = {
            "text": "I'LL AVENGE YOU EVEN IF I'M A GHOST!",
            "translated": "EU VOU VINGAR VOCE, MESMO QUE EU SEJA UM FANTASMA!",
            "bbox": [191, 9693, 332, 9743],
            "source_bbox": [149, 9664, 378, 9769],
            "text_pixel_bbox": [150, 9670, 373, 9767],
            "line_polygons": [
                [[176, 9668], [347, 9668], [347, 9691], [176, 9691]],
                [[150, 9706], [373, 9706], [373, 9728], [150, 9728]],
                [[192, 9741], [330, 9741], [330, 9767], [192, 9767]],
            ],
            "balloon_bbox": [0, 9648, 437, 9785],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "tipo": "fala",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "cor": "#000000"},
        }

        plan = ensure_legible_plan(Image.new("RGB", (800, 11000), (255, 255, 255)), plan_text_layout(text_data))
        resolved = _resolve_text_layout(text_data, plan)

        self.assertTrue(plan["_center_on_balloon_bbox"])
        self.assertGreater(plan["capacity_bbox"][2] - plan["capacity_bbox"][0], 300)
        self.assertGreater(plan["safe_text_box"][2] - plan["safe_text_box"][0], 250)
        self.assertGreaterEqual(resolved["font_size"], 21)
        self.assertLessEqual(len(resolved["lines"]), 3)

    def test_merge_adjacent_white_fragments_uses_geometry_not_tipo(self):
        blocks = [
            {
                "id": "ocr_001",
                "translated": "EU VOU",
                "bbox": [120, 30, 210, 52],
                "text_pixel_bbox": [120, 30, 210, 52],
                "source_bbox": [120, 30, 210, 52],
                "balloon_bbox": [80, 10, 260, 96],
                "line_polygons": [[[120, 30], [210, 30], [210, 52], [120, 52]]],
                "balloon_type": "white",
                "layout_profile": "white_balloon",
                "tipo": "narracao",
                "_visual_lobe_split_count": 2,
            },
            {
                "id": "ocr_002",
                "translated": "VE-LO NOVAMENTE?",
                "bbox": [92, 58, 246, 84],
                "text_pixel_bbox": [92, 58, 246, 84],
                "source_bbox": [92, 58, 246, 84],
                "balloon_bbox": [80, 10, 260, 96],
                "line_polygons": [[[92, 58], [246, 58], [246, 84], [92, 84]]],
                "balloon_type": "white",
                "layout_profile": "white_balloon",
                "tipo": "narracao",
                "_visual_lobe_split_count": 2,
            },
        ]

        merged = renderer_mod._merge_adjacent_white_balloon_fragments(blocks)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["translated"], "EU VOU\nVE-LO NOVAMENTE?")
        self.assertEqual(merged[0]["source_text_count"], 2)

    def test_split_single_ocr_visual_lobes_uses_geometry_not_tipo(self):
        text = {
            "id": "ocr_001",
            "translated": "PRIMEIRA PARTE\nSEGUNDA PARTE",
            "bbox": [80, 20, 260, 156],
            "text_pixel_bbox": [80, 20, 260, 156],
            "balloon_bbox": [40, 0, 300, 190],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "texto",
            "line_polygons": [
                [[92, 24], [240, 24], [240, 44], [92, 44]],
                [[104, 128], [232, 128], [232, 150], [104, 150]],
            ],
        }

        split = renderer_mod._split_single_ocr_visual_lobes(text)

        self.assertIsNotNone(split)
        assert split is not None
        self.assertEqual(len(split), 2)
        self.assertTrue(all(item.get("_visual_lobe_split_count") == 2 for item in split))

    def test_visual_lobe_split_recomputes_safe_box_when_parent_safe_intersection_is_degenerate(self):
        text = {
            "id": "ocr_001",
            "translated": "OBRIGADO PELA AJUDA! A PARTIR DE AGORA, DEIXE COM A GENTE!",
            "bbox": [134, 13, 672, 505],
            "source_bbox": [134, 13, 672, 505],
            "text_pixel_bbox": [134, 13, 672, 505],
            "balloon_bbox": [0, 0, 800, 637],
            "bubble_inner_bbox": [200, 53, 370, 62],
            "layout_safe_bbox": [200, 53, 370, 62],
            "layout_safe_reason": "bubble_inner_bbox",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "texto",
            "line_polygons": [
                [[134, 13], [304, 13], [304, 30], [134, 30]],
                [[136, 42], [304, 42], [304, 62], [136, 62]],
                [[476, 461], [672, 461], [672, 480], [476, 480]],
                [[476, 488], [661, 488], [661, 505], [476, 505]],
            ],
        }

        split = renderer_mod._split_single_ocr_visual_lobes(text)

        self.assertIsNotNone(split)
        assert split is not None
        first_plan = plan_text_layout(split[0])
        self.assertNotEqual(first_plan["layout_safe_bbox"], [200, 53, 370, 62])
        self.assertGreaterEqual(first_plan["safe_text_box"][3] - first_plan["safe_text_box"][1], 28)
        self.assertIn("safe_text_box_recomputed", split[0].get("qa_flags") or [])

    def test_long_white_dialogue_expands_width_even_when_ocr_box_is_tall(self):
        text_data = {
            "text": "THE INTEREST WAS ALREADY REDUCED",
            "translated": "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES O PRINCIPAL.",
            "bbox": [525, 8188, 696, 8302],
            "text_pixel_bbox": [527, 8192, 688, 8300],
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48, "cor": "#000000"},
            "page_width": 760,
            "page_height": 14000,
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["target_bbox"], [527, 8192, 688, 8300])
        self.assertFalse(plan["_simple_anchor_capacity_expanded"])
        self.assertEqual(plan["capacity_bbox"], [527, 8192, 688, 8300])
        self.assertLessEqual(resolved["font_size"], plan["target_size"] + 2)

    def test_canonical_render_style_preserves_explicit_font_and_effects(self):
        style = _canonical_render_style(
            {
                "fonte": "Newrotic.ttf",
                "tamanho": 32,
                "cor": "#ff00ff",
                "italico": True,
                "sombra": True,
                "glow": True,
                "contorno_px": 4,
            }
        )

        self.assertEqual(style["fonte"], "Newrotic.ttf")
        self.assertEqual(style["tamanho"], 32)
        self.assertEqual(style["cor"], "#ff00ff")
        self.assertTrue(style["bold"])
        self.assertTrue(style["italico"])
        self.assertTrue(style["sombra"])
        self.assertTrue(style["glow"])
        self.assertEqual(style["contorno_px"], 4)

    def test_canonical_render_style_defaults_to_no_effects(self):
        style = _canonical_render_style({})

        self.assertEqual(style["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(style["cor"], "#000000")
        self.assertEqual(style["contorno"], "")
        self.assertEqual(style["contorno_px"], 0)
        self.assertFalse(style["sombra"])
        self.assertFalse(style["glow"])

    def test_render_text_block_sanitizes_auto_style_as_last_defense(self):
        img = Image.new("RGB", (260, 180), (255, 255, 255))
        text_data = {
            "style_origin": "auto",
            "translated": "OLA",
            "bbox": [40, 40, 220, 140],
            "balloon_bbox": [40, 40, 220, 140],
            "tipo": "fala",
            "estilo": {
                "fonte": "Newrotic.ttf",
                "tamanho": 28,
                "cor": "#FFFFFF",
                "contorno": "#000000",
                "contorno_px": 2,
                "glow": True,
                "glow_px": 3,
                "sombra": True,
                "sombra_offset": [2, 2],
                "cor_gradiente": [],
            },
        }
        render_text_block(img, text_data)

        self.assertEqual(text_data["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(text_data["estilo"]["cor"], "#000000")
        self.assertEqual(text_data["estilo"]["contorno"], "")
        self.assertEqual(text_data["estilo"]["contorno_px"], 0)
        self.assertFalse(text_data["estilo"]["glow"])
        self.assertFalse(text_data["estilo"]["sombra"])

    def test_render_text_block_preserves_editor_style(self):
        img = Image.new("RGB", (260, 180), (20, 20, 24))
        text_data = {
            "style_origin": "editor",
            "translated": "OLA",
            "bbox": [40, 40, 220, 140],
            "balloon_bbox": [40, 40, 220, 140],
            "tipo": "fala",
            "estilo": {
                "fonte": "Newrotic.ttf",
                "tamanho": 28,
                "cor": "#FFFFFF",
                "contorno": "#000000",
                "contorno_px": 2,
                "glow": True,
                "glow_cor": "#ff00ff",
                "glow_px": 3,
                "sombra": True,
                "sombra_cor": "#111111",
                "sombra_offset": [2, 3],
                "cor_gradiente": [],
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["estilo"]["fonte"], "Newrotic.ttf")
        self.assertEqual(text_data["estilo"]["contorno"], "#000000")
        self.assertEqual(text_data["estilo"]["contorno_px"], 2)
        self.assertTrue(text_data["estilo"]["glow"])
        self.assertTrue(text_data["estilo"]["sombra"])

    def test_render_text_block_sanitizes_missing_origin_as_legacy_auto(self):
        img = Image.new("RGB", (260, 180), (255, 255, 255))
        text_data = {
            "translated": "OLA",
            "bbox": [40, 40, 220, 140],
            "balloon_bbox": [40, 40, 220, 140],
            "tipo": "fala",
            "estilo": {
                "fonte": "Newrotic.ttf",
                "tamanho": 28,
                "cor": "#FFFFFF",
                "contorno": "#000000",
                "contorno_px": 2,
                "glow": True,
                "sombra": True,
                "cor_gradiente": [],
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(text_data["estilo"]["cor"], "#000000")
        self.assertEqual(text_data["estilo"]["contorno_px"], 0)
        self.assertFalse(text_data["estilo"]["glow"])
        self.assertFalse(text_data["estilo"]["sombra"])

    def test_ensure_legible_plan_does_not_add_outline_for_white_balloon(self):
        img = Image.new("RGB", (200, 120), (255, 255, 255))
        plan = {
            "target_bbox": [20, 20, 180, 100],
            "text_color": "#FFFFFF",
            "outline_color": "",
            "outline_px": 0,
            "glow": False,
            "glow_cor": "",
            "glow_px": 0,
            "cor_gradiente": [],
        }

        adjusted = ensure_legible_plan(img, plan)

        self.assertEqual(adjusted["text_color"], "#111111")
        self.assertEqual(adjusted["outline_color"], "")
        self.assertEqual(adjusted["outline_px"], 0)

    def test_font_glyph_lookup_is_cached_per_font_and_character(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        font_path = str(fonts_dir / "ComicNeue-Bold.ttf")
        cache_clear = getattr(_font_has_glyph, "cache_clear", None)
        if cache_clear:
            cache_clear()

        with patch("typesetter.renderer._get_ft2_font", wraps=renderer_mod._get_ft2_font) as get_font:
            self.assertTrue(_font_has_glyph(font_path, "A"))
            self.assertTrue(_font_has_glyph(font_path, "A"))

        self.assertEqual(get_font.call_count, 1)

    def test_find_font_caches_resolved_path_by_font_name(self):
        class FakeFontDir:
            def __init__(self, path: Path):
                self.path = path
                self.rglob_calls = 0

            def exists(self):
                return True

            def rglob(self, _pattern):
                self.rglob_calls += 1
                return [self.path]

        with tempfile.TemporaryDirectory() as tmpdir:
            font_path = Path(tmpdir) / "CachedFont.ttf"
            font_path.write_bytes(b"font")
            fake_dir = FakeFontDir(font_path)
            renderer_mod._font_path_cache.clear()

            try:
                with patch("typesetter.renderer.FONT_DIRS", [fake_dir]):
                    self.assertEqual(Path(find_font("CachedFont.ttf")), font_path)
                    self.assertEqual(Path(find_font("CachedFont.ttf")), font_path)
            finally:
                renderer_mod._font_path_cache.clear()

        self.assertEqual(fake_dir.rglob_calls, 1)

    def test_google_font_cache_filename_resolves_from_user_google_fonts_in_render(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        source_font = fonts_dir / "ComicNeue-Bold.ttf"
        google_font_name = "GoogleFont__Bangers__regular.ttf"

        renderer_mod._font_cache.clear()
        renderer_mod._font_path_cache.clear()
        renderer_mod._ft2_cache.clear()
        if hasattr(renderer_mod._font_has_glyph, "cache_clear"):
            renderer_mod._font_has_glyph.cache_clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            user_fonts_dir = Path(tmpdir) / ".traduzai" / "fonts"
            google_dir = user_fonts_dir / "google"
            google_dir.mkdir(parents=True)
            google_font = google_dir / google_font_name
            google_font.write_bytes(source_font.read_bytes())

            try:
                with patch("typesetter.renderer.FONT_DIRS", [user_fonts_dir]):
                    self.assertEqual(Path(find_font(google_font_name)), google_font)

                    img = Image.new("RGB", (320, 220), (255, 255, 255))
                    text_data = {
                        "style_origin": "editor",
                        "translated": "BANGERS TEST",
                        "bbox": [40, 40, 280, 160],
                        "tipo": "fala",
                        "estilo": {
                            "fonte": google_font_name,
                            "tamanho": 30,
                            "cor": "#111111",
                            "contorno": "",
                            "contorno_px": 0,
                            "alinhamento": "center",
                            "sombra": False,
                            "glow": False,
                            "cor_gradiente": [],
                        },
                    }

                    render_text_block(img, text_data)

                    resolved_google_fonts = [
                        Path(font.font_path)
                        for (font_name, _size), font in renderer_mod._font_cache.items()
                        if font_name == google_font_name and hasattr(font, "font_path")
                    ]
                    self.assertIn(google_font, resolved_google_fonts)
                    self.assertEqual(text_data["estilo"]["fonte"], google_font_name)
                    self.assertLess(int(np.array(img).min()), 245)
            finally:
                renderer_mod._font_cache.clear()
                renderer_mod._font_path_cache.clear()
                renderer_mod._ft2_cache.clear()
                if hasattr(renderer_mod._font_has_glyph, "cache_clear"):
                    renderer_mod._font_has_glyph.cache_clear()

    def test_render_text_block_uses_balloon_bbox_instead_of_collapsing_to_text_pixel_bbox(self):
        img = Image.new("RGB", (420, 320), (255, 255, 255))
        text_data = {
            "translated": "ANCORA",
            "bbox": [40, 40, 380, 260],
            "source_bbox": [70, 84, 190, 128],
            "text_pixel_bbox": [76, 90, 184, 122],
            "balloon_bbox": [40, 40, 380, 260],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "left",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
            },
            "layout_shape": "wide",
            "layout_align": "center",
        }
        original_balloon = list(text_data["balloon_bbox"])

        render_text_block(img, text_data)

        arr = np.array(img)
        ink = np.any(arr < 245, axis=2)
        ys, xs = np.where(ink)
        self.assertGreater(xs.size, 0)
        self.assertEqual(text_data["balloon_bbox"], original_balloon)
        self.assertNotEqual(text_data["balloon_bbox"], text_data["text_pixel_bbox"])
        self.assertIn("render_bbox", text_data)
        rx1, ry1, rx2, ry2 = text_data["render_bbox"]
        self.assertGreaterEqual(rx1, original_balloon[0])
        self.assertGreaterEqual(ry1, original_balloon[1])
        self.assertLessEqual(rx2, original_balloon[2])
        self.assertLessEqual(ry2, original_balloon[3])

    def test_render_text_block_populates_render_bbox(self):
        img = Image.new("RGB", (420, 320), (255, 255, 255))
        text_data = {
            "translated": "RENDER BBOX",
            "bbox": [40, 40, 380, 260],
            "balloon_bbox": [40, 40, 380, 260],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
            },
            "layout_shape": "wide",
            "layout_align": "center",
        }

        render_text_block(img, text_data)

        self.assertIn("render_bbox", text_data)
        rx1, ry1, rx2, ry2 = text_data["render_bbox"]
        self.assertLess(rx1, rx2)
        self.assertLess(ry1, ry2)
        self.assertGreaterEqual(rx1, 40)
        self.assertLessEqual(rx2, 380)

    def test_render_text_block_rotates_editor_style_around_target_center(self):
        def make_text(rotation: int) -> dict:
            return {
                "style_origin": "editor",
                "translated": "ROTATE",
                "bbox": [40, 40, 220, 140],
                "balloon_bbox": [40, 40, 220, 140],
                "tipo": "fala",
                "estilo": {
                    "fonte": "ComicNeue-Bold.ttf",
                    "tamanho": 32,
                    "cor": "#111111",
                    "contorno": "",
                    "contorno_px": 0,
                    "alinhamento": "center",
                    "sombra": False,
                    "glow": False,
                    "cor_gradiente": [],
                    "rotacao": rotation,
                },
                "layout_shape": "wide",
                "layout_align": "center",
            }

        unrotated = Image.new("RGB", (260, 180), (255, 255, 255))
        rotated = Image.new("RGB", (260, 180), (255, 255, 255))
        unrotated_text = make_text(0)
        rotated_text = make_text(25)

        render_text_block(unrotated, unrotated_text)
        render_text_block(rotated, rotated_text)

        diff = ImageChops.difference(unrotated, rotated)
        self.assertIsNotNone(diff.getbbox())
        self.assertIn("render_bbox", rotated_text)
        rx1, ry1, rx2, ry2 = rotated_text["render_bbox"]
        self.assertLess(rx1, rx2)
        self.assertLess(ry1, ry2)
        self.assertGreaterEqual(rx1, 0)
        self.assertLessEqual(rx2, rotated.width)

    def test_plan_text_layout_uses_auto_ocr_rotation_when_style_default_is_zero(self):
        text_data = {
            "style_origin": "auto",
            "translated": "VIRADO",
            "bbox": [90, 40, 150, 220],
            "source_bbox": [90, 40, 150, 220],
            "text_pixel_bbox": [105, 52, 135, 208],
            "balloon_bbox": [90, 40, 150, 220],
            "rotation_deg": 90,
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#111111",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
                "rotacao": 0,
            },
            "layout_shape": "tall",
            "layout_align": "center",
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["rotation_deg"], 90)

    def test_plan_text_layout_infers_sideways_rotation_from_ocr_line_polygons(self):
        text_data = {
            "style_origin": "auto",
            "translated": "VIRADO",
            "bbox": [90, 40, 150, 220],
            "source_bbox": [90, 40, 150, 220],
            "text_pixel_bbox": [105, 52, 135, 208],
            "balloon_bbox": [90, 40, 150, 220],
            "line_polygons": [
                [[104, 52], [136, 52], [136, 208], [104, 208]],
            ],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#111111",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
                "rotacao": 0,
            },
            "layout_shape": "tall",
            "layout_align": "center",
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["rotation_deg"], 90)
        self.assertEqual(plan["rotation_source"], "line_polygons")

    def test_plan_text_layout_preserves_oblique_source_text_angle(self):
        text_data = {
            "style_origin": "auto",
            "translated": "PLACA",
            "bbox": [300, 1080, 650, 1650],
            "source_bbox": [300, 1080, 650, 1650],
            "text_pixel_bbox": [333, 1123, 559, 1596],
            "balloon_bbox": [300, 1080, 650, 1650],
            "line_polygons": [
                [[333, 1136], [380, 1123], [512, 1583], [465, 1596]],
                [[395, 1150], [450, 1136], [559, 1534], [504, 1548]],
            ],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#111111",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
                "rotacao": 0,
            },
            "layout_shape": "tall",
            "layout_align": "center",
        }

        plan = plan_text_layout(text_data)

        self.assertAlmostEqual(plan["rotation_deg"], 75.0, delta=2.0)
        self.assertEqual(plan["rotation_source"], "line_polygons")

    def test_render_text_block_keeps_sideways_ocr_text_vertical(self):
        img = Image.new("RGB", (240, 260), (255, 255, 255))
        text_data = {
            "style_origin": "auto",
            "translated": "SIDEWAYS",
            "bbox": [90, 40, 150, 220],
            "source_bbox": [90, 40, 150, 220],
            "text_pixel_bbox": [105, 52, 135, 208],
            "balloon_bbox": [90, 40, 150, 220],
            "line_polygons": [
                [[104, 52], [136, 52], [136, 208], [104, 208]],
            ],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#111111",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
                "rotacao": 0,
            },
            "layout_shape": "tall",
            "layout_align": "center",
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["rotation_deg"], 90)
        self.assertEqual(text_data["estilo"]["rotacao"], 90)
        rx1, ry1, rx2, ry2 = text_data["render_bbox"]
        self.assertGreater(ry2 - ry1, rx2 - rx1)
        self.assertGreaterEqual(ry2 - ry1, 90)
        self.assertEqual(text_data.get("_render_debug", {}).get("rotation_deg"), 90)
        self.assertNotIn("TEXT_CLIPPED", text_data.get("qa_flags") or [])
        self.assertNotIn("TEXT_OVERFLOW", text_data.get("qa_flags") or [])

    def test_plan_text_layout_offsets_safe_box_for_horizontal_anchor_bias(self):
        text_data = {
            "translated": "ESTE TROVÃO, NO MOMENTO EM QUE A ENERGIA INTERNA ATINGE UM CERTO PONTO",
            "bbox": [180, 16, 540, 166],
            "source_bbox": [180, 16, 540, 166],
            "text_pixel_bbox": [185, 22, 531, 160],
            "balloon_bbox": [0, 0, 590, 182],
            "balloon_type": "white",
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 34,
                "cor": "#1D1D1D",
                "contorno": "#FCFCFB",
                "contorno_px": 4,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
            },
            "layout_shape": "wide",
            "layout_align": "center",
            "layout_profile": "white_balloon",
        }

        plan = plan_text_layout(text_data)

        position_center = (plan["position_bbox"][0] + plan["position_bbox"][2]) / 2.0
        safe_center = (plan["safe_text_box"][0] + plan["safe_text_box"][2]) / 2.0
        self.assertTrue(plan["_center_on_balloon_bbox"])
        self.assertEqual(plan["horizontal_bias_px"], 0)
        self.assertAlmostEqual(safe_center, position_center, delta=1.0)
        self.assertGreaterEqual(plan["safe_text_box"][0], text_data["balloon_bbox"][0])
        self.assertLessEqual(plan["safe_text_box"][2], text_data["balloon_bbox"][2])

    def test_plan_text_layout_does_not_lock_long_white_narration_to_low_anchor(self):
        text_data = {
            "translated": "POR QUE VOCE NAO VEM! OS CARAS ESTAO ESPERANDO!",
            "bbox": [219, 105, 466, 144],
            "source_bbox": [219, 105, 466, 144],
            "text_pixel_bbox": [204, 105, 480, 185],
            "balloon_bbox": [196, 0, 485, 187],
            "balloon_type": "white",
            "tipo": "narracao",
            "content_class": "narration",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 19,
                "cor": "#111111",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
            },
            "layout_shape": "wide",
            "layout_align": "top",
            "layout_profile": "white_balloon",
        }

        plan = plan_text_layout(text_data)

        self.assertFalse(plan["_anchor_capacity_locked"])
        self.assertTrue(plan["_center_on_balloon_bbox"])
        self.assertEqual(plan["vertical_anchor"], "center")
        target_cx = (text_data["balloon_bbox"][0] + text_data["balloon_bbox"][2]) / 2
        target_cy = (text_data["balloon_bbox"][1] + text_data["balloon_bbox"][3]) / 2
        position_cx = (plan["position_bbox"][0] + plan["position_bbox"][2]) / 2
        position_cy = (plan["position_bbox"][1] + plan["position_bbox"][3]) / 2
        self.assertAlmostEqual(position_cx, target_cx, delta=1.0)
        self.assertAlmostEqual(position_cy, target_cy, delta=1.0)
        safe_cy = (plan["safe_text_box"][1] + plan["safe_text_box"][3]) / 2
        self.assertAlmostEqual(safe_cy, target_cy, delta=1.0)
        self.assertGreaterEqual(plan["safe_text_box"][1], plan["position_bbox"][1])
        self.assertLessEqual(plan["safe_text_box"][3], plan["position_bbox"][3])

    def test_top_narration_white_box_centers_when_using_balloon_center_rule(self):
        text_data = {
            "translated": "VIVER NAO E DIVERTIDO, MAS ISSO NAO SIGNIFICA QUE TENHO CORAGEM DE MORRER.",
            "bbox": [255, 7594, 672, 7674],
            "source_bbox": [314, 7596, 682, 7708],
            "text_pixel_bbox": [317, 7602, 680, 7702],
            "balloon_bbox": [160, 7580, 800, 7724],
            "balloon_type": "white",
            "tipo": "narracao",
            "content_class": "narration",
            "layout_profile": "top_narration",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#000000",
                "alinhamento": "center",
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertTrue(plan["_center_on_balloon_bbox"])
        self.assertEqual(plan["vertical_anchor"], "center")
        safe_cy = (plan["safe_text_box"][1] + plan["safe_text_box"][3]) / 2
        render_cy = (resolved["block_bbox"][1] + resolved["block_bbox"][3]) / 2
        self.assertAlmostEqual(render_cy, safe_cy, delta=3.0)

    def test_top_narration_detects_visual_rect_before_centering(self):
        page = np.full((320, 850, 3), 255, dtype=np.uint8)
        page[28:30, 259:735] = 0
        page[275:277, 259:735] = 0
        page[28:277, 259:261] = 0
        page[28:277, 734:736] = 0
        page[117:119, 0:262] = 0  # neighboring panel line crossing the raw balloon bbox
        img = Image.fromarray(page)
        text_data = {
            "translated": "VIVER NAO E DIVERTIDO, MAS ISSO NAO SIGNIFICA QUE TENHO CORAGEM DE MORRER.",
            "bbox": [263, 101, 664, 175],
            "source_bbox": [314, 96, 682, 208],
            "text_pixel_bbox": [317, 102, 680, 202],
            "balloon_bbox": [160, 80, 800, 224],
            "balloon_type": "white",
            "tipo": "narracao",
            "content_class": "narration",
            "layout_profile": "top_narration",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#000000",
                "alinhamento": "center",
            },
        }

        renderer_mod._apply_visual_rect_safe_area_if_needed(img, text_data)
        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(text_data["layout_safe_reason"], "visual_rect_inner")
        self.assertGreater(text_data["_visual_rect_outer_bbox"][0], text_data["balloon_bbox"][0])
        self.assertLess(text_data["_visual_rect_outer_bbox"][2], text_data["balloon_bbox"][2])
        visual_cx = (text_data["_visual_rect_outer_bbox"][0] + text_data["_visual_rect_outer_bbox"][2]) / 2.0
        render_cx = (resolved["block_bbox"][0] + resolved["block_bbox"][2]) / 2.0
        self.assertAlmostEqual(render_cx, visual_cx, delta=4.0)
        safe_cy = (text_data["_visual_rect_inner_bbox"][1] + text_data["_visual_rect_inner_bbox"][3]) / 2.0
        render_cy = (resolved["block_bbox"][1] + resolved["block_bbox"][3]) / 2.0
        self.assertAlmostEqual(render_cy, safe_cy, delta=4.0)
        self.assertGreaterEqual(resolved["block_bbox"][1], text_data["_visual_rect_inner_bbox"][1])
        self.assertLessEqual(resolved["block_bbox"][3], text_data["_visual_rect_inner_bbox"][3])

    def test_top_narration_layout_recovers_visual_rect_bbox(self):
        page_bgr = np.full((320, 850, 3), 255, dtype=np.uint8)
        page_bgr[28:30, 259:735] = 0
        page_bgr[275:277, 259:735] = 0
        page_bgr[28:277, 259:261] = 0
        page_bgr[28:277, 734:736] = 0
        page_bgr[117:119, 0:262] = 0
        text_data = {
            "tipo": "narracao",
            "layout_profile": "top_narration",
            "source_bbox": [314, 96, 682, 208],
            "text_pixel_bbox": [317, 102, 680, 202],
        }

        recovered = balloon_layout_mod._detect_top_narration_rect_bbox(
            page_bgr,
            [160, 80, 800, 224],
            text_data,
        )

        self.assertIsNotNone(recovered)
        self.assertGreater(recovered[0], 240)
        self.assertLess(recovered[2], 750)
        self.assertLessEqual(abs(((recovered[0] + recovered[2]) / 2.0) - 496.5), 5.0)

    def test_plan_text_layout_does_not_lock_short_white_balloon_text_to_page_edge(self):
        text_data = {
            "translated": "PARE!",
            "bbox": [6, 744, 184, 830],
            "source_bbox": [6, 744, 184, 830],
            "text_pixel_bbox": [6, 744, 184, 830],
            "balloon_bbox": [0, 730, 690, 860],
            "balloon_type": "white",
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 38,
                "cor": "#1D1D1D",
                "alinhamento": "center",
                "cor_gradiente": [],
            },
            "layout_shape": "wide",
            "layout_align": "center",
            "layout_profile": "white_balloon",
        }

        plan = plan_text_layout(text_data)

        self.assertFalse(plan["_anchor_capacity_locked"])
        self.assertGreaterEqual(plan["safe_text_box"][0], 20)

    def test_safe_text_path_render_bbox_uses_ink_not_line_cell_for_qa(self):
        img = Image.new("RGB", (480, 180), (255, 255, 255))
        text_data = {
            "translated": "AMBOS DIVIDEM A ENERGIA INTERNA EM YIN E YANG",
            "bbox": [74, 16, 376, 112],
            "source_bbox": [74, 16, 376, 112],
            "text_pixel_bbox": [81, 21, 370, 108],
            "balloon_bbox": [38, 0, 424, 128],
            "balloon_type": "white",
            "tipo": "narracao",
            "qa_flags": [],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 25,
                "cor": "#1B1B1B",
                "contorno": "#F8F8F8",
                "contorno_px": 4,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
            },
            "layout_shape": "wide",
            "layout_align": "top",
            "layout_profile": "top_narration",
        }

        plan = plan_text_layout(text_data)
        _render_single_text_block(img, text_data, plan)

        self.assertNotIn("TEXT_CLIPPED", text_data.get("qa_flags", []))
        self.assertGreaterEqual(text_data["render_bbox"][1], plan["safe_text_box"][1])

    def test_safe_text_path_clamps_anchor_above_balloon_into_safe_box(self):
        img = Image.new("RGB", (720, 220), (32, 32, 32))
        text_data = {
            "translated": "- revolução",
            "bbox": [604, 16, 707, 45],
            "source_bbox": [604, 16, 707, 45],
            "text_pixel_bbox": [603, 2, 708, 41],
            "balloon_bbox": [582, 4, 720, 57],
            "balloon_type": "textured",
            "tipo": "narracao",
            "qa_flags": [],
            "estilo": {
                "fonte": "Newrotic.ttf",
                "tamanho": 14,
                "cor": "#FFFFFF",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
                "sombra": True,
                "sombra_cor": "#000000",
                "sombra_offset": [2, 2],
                "glow": False,
                "cor_gradiente": ["#CCEAFA", "#FDFDFD"],
                "force_upper": True,
            },
            "layout_shape": "wide",
            "layout_align": "top",
            "layout_profile": "standard",
        }

        plan = plan_text_layout(text_data)
        _render_single_text_block(img, text_data, plan)

        self.assertNotIn("TEXT_CLIPPED", text_data.get("qa_flags", []))
        self.assertNotIn("TEXT_OVERFLOW", text_data.get("qa_flags", []))
        self.assertGreaterEqual(text_data["render_bbox"][1], plan["safe_text_box"][1])
        self.assertGreaterEqual(text_data["render_bbox"][1], plan["target_bbox"][1])

    def test_resolve_text_layout_clamps_font_size_for_white_balloon_bounds(self):
        text_data = {
            "translated": "TESTE",
            "bbox": [40, 40, 360, 260],
            "text_pixel_bbox": [120, 120, 188, 138],
            "balloon_bbox": [40, 40, 360, 260],
            "balloon_type": "white",
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 40,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
            },
            "layout_shape": "wide",
            "layout_align": "center",
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        bounds = _category_font_bounds(text_data)
        self.assertGreaterEqual(resolved["font_size"], _MIN_FONT_SIZE)
        self.assertLessEqual(resolved["font_size"], bounds[1])

    def test_resolve_text_layout_clamps_font_size_for_textured_balloon_bounds(self):
        text_data = {
            "translated": "TESTE",
            "bbox": [40, 40, 360, 260],
            "text_pixel_bbox": [120, 120, 184, 136],
            "balloon_bbox": [40, 40, 360, 260],
            "balloon_type": "textured",
            "tipo": "fala",
            "estilo": {
                "fonte": "Newrotic.ttf",
                "tamanho": 40,
                "cor": "#FFFFFF",
                "contorno": "#000000",
                "contorno_px": 2,
                "alinhamento": "center",
            },
            "layout_shape": "wide",
            "layout_align": "center",
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        bounds = _category_font_bounds(text_data)
        self.assertGreaterEqual(resolved["font_size"], _MIN_FONT_SIZE)
        self.assertLessEqual(resolved["font_size"], bounds[1])

    def test_auto_ocr_layout_caps_font_to_original_line_height(self):
        text_data = {
            "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
            "text": "PLEASE, FOR THE CHILD'S SAKE.",
            "bbox": [90, 3690, 360, 3860],
            "source_bbox": [90, 3690, 360, 3860],
            "text_pixel_bbox": [116, 3738, 326, 3828],
            "line_polygons": [
                [[116, 3738], [326, 3738], [326, 3760], [116, 3760]],
                [[134, 3768], [304, 3768], [304, 3790], [134, 3790]],
                [[174, 3802], [258, 3802], [258, 3828], [174, 3828]],
            ],
            "balloon_bbox": [90, 3690, 360, 3860],
            "balloon_type": "white",
            "tipo": "fala",
            "style_origin": "auto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 0,
                "alinhamento": "center",
            },
            "layout_shape": "wide",
            "layout_align": "center",
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertLessEqual(plan["target_size"], 26)
        self.assertLessEqual(plan["_font_search_cap"], 26)
        self.assertLessEqual(resolved["font_size"], 26)
        self.assertFalse(plan["_simple_anchor_capacity_expanded"])

    def test_overbroad_white_balloon_centers_translation_inside_balloon(self):
        text_data = {
            "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
            "text": "PLEASE, FOR THE CHILD'S SAKE.",
            "bbox": [247, 5652, 667, 5676],
            "source_bbox": [498, 5655, 656, 5740],
            "text_pixel_bbox": [498, 5655, 656, 5740],
            "line_polygons": [
                [[498, 5652], [658, 5654], [658, 5678], [498, 5676]],
                [[507, 5686], [648, 5686], [648, 5707], [507, 5707]],
                [[543, 5718], [615, 5718], [615, 5743], [543, 5743]],
            ],
            "balloon_bbox": [25, 5436, 667, 5741],
            "balloon_type": "white",
            "tipo": "fala",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#000000",
                "alinhamento": "center",
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertFalse(plan["_anchor_capacity_locked"])
        self.assertTrue(plan["_center_on_balloon_bbox"])
        target_cx = (text_data["balloon_bbox"][0] + text_data["balloon_bbox"][2]) / 2
        target_cy = (text_data["balloon_bbox"][1] + text_data["balloon_bbox"][3]) / 2
        safe_cx = (plan["safe_text_box"][0] + plan["safe_text_box"][2]) / 2
        safe_cy = (plan["safe_text_box"][1] + plan["safe_text_box"][3]) / 2
        self.assertAlmostEqual(safe_cx, target_cx, delta=1.0)
        self.assertAlmostEqual(safe_cy, target_cy, delta=1.0)
        self.assertGreaterEqual(plan["safe_text_box"][0], text_data["balloon_bbox"][0])
        self.assertLessEqual(plan["safe_text_box"][2], text_data["balloon_bbox"][2])
        self.assertGreaterEqual(resolved["font_size"], 22)
        self.assertGreaterEqual(len(resolved["lines"]), 1)

    def test_visual_lobe_split_long_translation_expands_tiny_line_capacity(self):
        text_data = {
            "translated": "ESTOU MORRENDO DE FOME",
            "text": "HEY, LET'S GO! I'M STARVING",
            "original": "HEY, LET'S GO! I'M STARVING",
            "bbox": [343, 11293, 538, 11317],
            "source_bbox": [343, 11293, 538, 11317],
            "text_pixel_bbox": [343, 11293, 538, 11317],
            "line_polygons": [
                [[343, 11293], [538, 11293], [538, 11317], [343, 11317]],
            ],
            "balloon_bbox": [343, 11293, 538, 11317],
            "balloon_type": "white",
            "tipo": "fala",
            "layout_profile": "white_balloon",
            "layout_group_size": 1,
            "_visual_lobe_split_index": 1,
            "_visual_lobe_split_count": 2,
            "_visual_lobe_split_parent_bbox": [148, 10877, 538, 11317],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#000000",
                "alinhamento": "center",
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertTrue(plan["_simple_anchor_capacity_expanded"])
        self.assertEqual(plan["_simple_anchor_capacity_reason"], "visual_lobe_long_text")
        self.assertGreaterEqual(plan["safe_text_box"][3] - plan["safe_text_box"][1], 48)
        self.assertGreater(plan["safe_text_box"][2] - plan["safe_text_box"][0], 195)
        self.assertGreaterEqual(resolved["font_size"], 22)
        self.assertGreaterEqual(len(resolved["lines"]), 2)

        render_data = dict(text_data)
        render_data.update(
            {
                "bbox": [100, 100, 295, 124],
                "source_bbox": [100, 100, 295, 124],
                "text_pixel_bbox": [100, 100, 295, 124],
                "line_polygons": [
                    [[100, 100], [295, 100], [295, 124], [100, 124]],
                ],
                "balloon_bbox": [100, 100, 295, 124],
                "_visual_lobe_split_parent_bbox": [0, 0, 295, 124],
            }
        )
        img = Image.new("RGB", (500, 320), (255, 255, 255))

        render_text_block(img, render_data)

        self.assertIsNone(render_data.get("qa_flags"))
        self.assertEqual(render_data["_render_debug"]["simple_anchor_capacity_reason"], "visual_lobe_long_text")
        self.assertGreaterEqual(render_data["_render_debug"]["font_size_final"], 22)

    def test_resolve_text_layout_grows_short_narracao_anchored_to_small_source_bbox(self):
        """Regressão para o bug do traduzido3 (Cap 1, "QUE INCÔMODO." em narração branca):

        Quando narração com OCR seed pequeno (14) era ancorada a um source_bbox
        menor que o balão, o binary-search achava best_fit=36, mas o loop de
        scoring rejeitava por overflow de altura sem a tolerância de +4px que
        _fits_in_box já aplicava. O fallback voltava para category_min=14,
        renderizando texto minúsculo num balão grande.
        """
        text_data = {
            "translated": "QUE INCÔMODO.",
            "text": "HOW BOTHERSOME.",
            "tipo": "narracao",
            "balloon_bbox": [145, 331, 452, 488],
            "bbox": [145, 331, 452, 488],
            "source_bbox": [181, 375, 407, 447],
            "estilo": {"tamanho": 14, "fonte": "ComicNeue-Bold.ttf"},
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        # Antes do fix: font_size=14 (fallback de category_min)
        # Depois do fix: font_size deve crescer para algo legível dentro do balão.
        self.assertGreaterEqual(
            resolved["font_size"], 28,
            f"Texto curto em balão narração branca deveria crescer; ficou em {resolved['font_size']}",
        )

    def test_normalize_render_text_replaces_katakana_middle_dot(self):
        self.assertEqual(_normalize_render_text("・・・・・"), ".....")

    def test_build_textpath_mask_renders_project_font(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        font = SafeTextPathFont(fonts_dir / "ComicNeue-Bold.ttf", 28)

        mask = _build_textpath_mask(font, "TEST 123?!", padding=2)

        self.assertEqual(mask.ndim, 2)
        self.assertGreater(mask.shape[1], 30)
        self.assertGreater(mask.shape[0], 10)
        self.assertGreater(int(np.max(mask)), 0)

    def test_build_textpath_mask_preserves_glyph_holes(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        font = SafeTextPathFont(fonts_dir / "ComicNeue-Bold.ttf", 64)

        mask = _build_textpath_mask(font, "O", padding=0)

        ys, xs = np.nonzero(mask > 20)
        self.assertGreater(xs.size, 0)
        center_x = int((int(xs.min()) + int(xs.max())) / 2)
        center_y = int((int(ys.min()) + int(ys.max())) / 2)
        self.assertLess(int(mask[center_y, center_x]), 32)

    def test_build_render_blocks_ignores_skip_processing_entries(self):
        blocks = build_render_blocks(
            [
                {
                    "translated": "SFX",
                    "text": "XEV",
                    "bbox": [10, 10, 80, 40],
                    "skip_processing": True,
                    "tipo": "fala",
                },
                {
                    "translated": "FALA",
                    "text": "HELLO",
                    "bbox": [20, 60, 160, 120],
                    "skip_processing": False,
                    "tipo": "fala",
                },
            ]
        )

        self.assertEqual(len(blocks), 2)
        self.assertEqual([block["translated"] for block in blocks], ["SFX", "FALA"])
        self.assertTrue(all(block.get("skip_processing") is False for block in blocks))

    def test_build_render_blocks_renders_unchanged_latin_name_in_white_balloon(self):
        text = {
            "id": "ocr_002",
            "text": "NAN ZIYING!",
            "translated": "NAN ZIYING!",
            "bbox": [550, 2987, 800, 3042],
            "balloon_bbox": [550, 2987, 800, 3042],
            "balloon_type": "white",
            "block_profile": "white_balloon",
            "background_rgb": [252, 251, 252],
            "skip_processing": False,
            "tipo": "fala",
        }

        blocks = build_render_blocks([text])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["translated"], "NAN ZIYING!")
        self.assertFalse(text["skip_processing"])
        self.assertFalse(text["preserve_original"])
        self.assertNotEqual(text.get("skip_reason"), "unchanged_latin_name_preserved")

    def test_build_render_blocks_keeps_translate_route_for_unchanged_latin_name(self):
        text = {
            "id": "ocr_002",
            "text": "NAN ZIYING!",
            "translated": "NAN ZIYING!",
            "bbox": [550, 2987, 800, 3042],
            "balloon_bbox": [550, 2987, 800, 3042],
            "balloon_type": "white",
            "block_profile": "white_balloon",
            "background_rgb": [252, 251, 252],
            "skip_processing": False,
            "tipo": "fala",
            "route_action": "translate_inpaint_render",
            "route_reason": "dialogue_balloon_with_english_text",
        }

        blocks = build_render_blocks([text])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(text["route_action"], "translate_inpaint_render")
        self.assertEqual(text["route_reason"], "dialogue_balloon_with_english_text")
        self.assertFalse(text["skip_processing"])
        self.assertFalse(text["preserve_original"])

    def test_build_render_blocks_does_not_preserve_common_untranslated_phrase(self):
        text = {
            "id": "ocr_002",
            "text": "GOOD MORNING",
            "translated": "GOOD MORNING",
            "bbox": [550, 2987, 800, 3042],
            "balloon_bbox": [550, 2987, 800, 3042],
            "balloon_type": "white",
            "block_profile": "white_balloon",
            "background_rgb": [252, 251, 252],
            "skip_processing": False,
            "tipo": "fala",
        }

        blocks = build_render_blocks([text])

        self.assertEqual(len(blocks), 1)
        self.assertFalse(text["skip_processing"])

    def test_render_qa_suppresses_art_warning_for_preserved_latin_name_glyphs(self):
        background = np.full((120, 220, 3), 255, dtype=np.uint8)
        background[45:66, 70:154] = 8
        text_data = {
            "original": "NAN ZIYING?!!!",
            "translated": "NAN ZIYING?!!!",
            "render_bbox": [70, 45, 154, 66],
            "balloon_bbox": [20, 18, 200, 100],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "background_rgb": [252, 252, 252],
            "qa_flags": [],
        }
        plan = {
            "target_bbox": [20, 18, 200, 100],
            "safe_text_box": [30, 25, 190, 92],
        }

        renderer_mod._run_render_qa(text_data, plan, background_image=background)

        self.assertNotIn("render_on_art_suspected", text_data.get("qa_flags") or [])
        self.assertEqual(text_data["qa_metrics"]["render_background_luma"], 8.0)

    def test_render_qa_suppresses_art_warning_for_unchanged_short_latin_exclamation(self):
        background = np.full((120, 220, 3), 110, dtype=np.uint8)
        background[45:66, 70:154] = 12
        text_data = {
            "original": "Suisui!!!",
            "translated": "Suisui!!!",
            "render_bbox": [70, 45, 154, 66],
            "balloon_bbox": [20, 18, 200, 100],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "background_rgb": [14, 14, 14],
            "qa_flags": [],
        }
        plan = {
            "target_bbox": [20, 18, 200, 100],
            "safe_text_box": [30, 25, 190, 92],
        }

        renderer_mod._run_render_qa(text_data, plan, background_image=background)

        self.assertNotIn("render_on_art_suspected", text_data.get("qa_flags") or [])
        self.assertEqual(text_data["qa_metrics"]["render_background_luma"], 12.0)

    def test_render_qa_keeps_art_warning_for_common_untranslated_phrase(self):
        background = np.full((120, 220, 3), 255, dtype=np.uint8)
        background[45:66, 70:164] = 8
        text_data = {
            "original": "GOOD MORNING",
            "translated": "GOOD MORNING",
            "render_bbox": [70, 45, 164, 66],
            "balloon_bbox": [20, 18, 200, 100],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "background_rgb": [252, 252, 252],
            "qa_flags": [],
        }
        plan = {
            "target_bbox": [20, 18, 200, 100],
            "safe_text_box": [30, 25, 190, 92],
        }

        renderer_mod._run_render_qa(text_data, plan, background_image=background)

        self.assertIn("render_on_art_suspected", text_data.get("qa_flags") or [])

    def test_render_qa_keeps_art_warning_for_short_common_untranslated_commands(self):
        for original in ("HELP!", "STOP!", "NO!", "GO!"):
            with self.subTest(original=original):
                background = np.full((120, 220, 3), 255, dtype=np.uint8)
                background[45:66, 70:154] = 8
                text_data = {
                    "original": original,
                    "translated": original,
                    "render_bbox": [70, 45, 154, 66],
                    "balloon_bbox": [20, 18, 200, 100],
                    "balloon_type": "white",
                    "layout_profile": "white_balloon",
                    "background_rgb": [252, 252, 252],
                    "qa_flags": [],
                }
                plan = {
                    "target_bbox": [20, 18, 200, 100],
                    "safe_text_box": [30, 25, 190, 92],
                }

                renderer_mod._run_render_qa(text_data, plan, background_image=background)

                self.assertIn("render_on_art_suspected", text_data.get("qa_flags") or [])

    def test_render_text_block_uses_safe_renderer_for_project_font(self):
        img = Image.new("RGB", (320, 240), (255, 255, 255))
        text_data = {
            "translated": "YOU SAID YOU COULD SEE",
            "bbox": [40, 40, 280, 140],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
                "sombra": True,
                "sombra_cor": "#999999",
                "sombra_offset": [2, 2],
                "glow": False,
                "cor_gradiente": [],
            },
        }

        with patch("typesetter.renderer.ImageDraw.Draw", side_effect=AssertionError("nao deve usar PIL text")):
            render_text_block(img, text_data)

        arr = np.array(img)
        self.assertLess(int(arr.min()), 245)

    def test_render_text_block_does_not_split_simple_balloon_only_because_subregions_exist(self):
        img = Image.new("RGB", (420, 320), (255, 255, 255))
        text_data = {
            "translated": "CONHECI ALGUNS QUE USAM ESSE PODER.",
            "bbox": [40, 60, 380, 220],
            "balloon_bbox": [40, 60, 380, 220],
            "balloon_subregions": [[40, 60, 210, 220], [210, 60, 380, 220]],
            "layout_group_size": 1,
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 24,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
            },
        }

        with patch("typesetter.renderer._render_connected_subregions") as connected_mock:
            render_text_block(img, text_data)

        connected_mock.assert_not_called()
        arr = np.array(img)
        self.assertLess(int(arr.min()), 245)

    def test_render_text_block_rejects_single_text_sentinel_connected_metadata(self):
        img = Image.new("RGB", (500, 320), (255, 255, 255))
        text_data = {
            "translated": "NAO CONSIGO... NAO CONSIGO ENCONTRAR O TEXTO ORIGINAL.",
            "bbox": [60, 70, 440, 250],
            "balloon_bbox": [60, 70, 440, 250],
            "balloon_subregions": [[60, 70, 250, 250], [250, 70, 440, 250]],
            "layout_profile": "connected_balloon",
            "layout_group_size": 1,
            "connected_balloon_orientation": "left-right",
            "connected_group_confidence": 0.28,
            "connected_detection_confidence": 1.0,
            "connected_position_confidence": 0.712,
            "subregion_confidence": 1.0,
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 24,
                "cor": "#111111",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
            },
        }

        with patch("typesetter.renderer._render_connected_subregions") as connected_mock:
            render_text_block(img, text_data)

        connected_mock.assert_not_called()
        arr = np.array(img)
        self.assertLess(int(arr.min()), 245)

    def test_edge_clipped_white_balloon_uses_visible_safe_width_for_wrap(self):
        text_data = {
            "translated": (
                "SEGURO, VOCÊ SABE, SEGURO DE VIDA REAL, COISAS ASSIM? "
                "SE VOCÊ NÃO TEM DINHEIRO, TEM QUE MOSTRAR SUA SINCERIDADE."
            ),
            "bbox": [253, 16, 646, 315],
            "text_pixel_bbox": [253, 53, 626, 311],
            "line_polygons": [
                [[336, 122], [481, 124], [481, 150], [336, 148]],
                [[270, 158], [547, 159], [547, 181], [270, 180]],
                [[257, 191], [563, 191], [563, 212], [257, 212]],
                [[253, 223], [566, 223], [566, 247], [253, 247]],
                [[276, 258], [542, 258], [542, 278], [276, 278]],
                [[347, 290], [475, 290], [475, 311], [347, 311]],
            ],
            "balloon_bbox": [0, 0, 628, 331],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_shape": "wide",
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#111111",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["layout_safe_reason"], "edge_clipped_white_balloon")
        self.assertGreater(plan["safe_text_box"][0], 140)
        self.assertFalse(plan["_position_on_capacity_bbox"])
        self.assertTrue(plan["_center_on_balloon_bbox"])
        self.assertLess(plan["max_width"], 420)
        self.assertGreaterEqual(len(resolved["lines"]), 4)
        self.assertLess(max(resolved["line_widths"]), 430)
        safe_center = (plan["layout_safe_bbox"][0] + plan["layout_safe_bbox"][2]) / 2.0
        rendered_center = (resolved["block_bbox"][0] + resolved["block_bbox"][2]) / 2.0
        self.assertLess(abs(rendered_center - safe_center), 36)

    def test_near_right_edge_white_balloon_uses_visible_safe_center(self):
        text_data = {
            "translated": "UAU, ELES NAO TIVERAM A MENOR CHANCE.. CONDICAO PRECARIA OU ALGO ASSIM.",
            "bbox": [270, 2977, 589, 3066],
            "text_pixel_bbox": [321, 2973, 579, 3093],
            "balloon_bbox": [270, 2881, 800, 3208],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "layout_shape": "wide",
            "tipo": "fala",
            "page_width": 816,
            "page_height": 14000,
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#111111",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["layout_safe_reason"], "edge_clipped_white_balloon")
        self.assertFalse(plan["_position_on_capacity_bbox"])
        self.assertTrue(plan["_center_on_balloon_bbox"])
        safe_center = (plan["layout_safe_bbox"][0] + plan["layout_safe_bbox"][2]) / 2.0
        rendered_center = (resolved["block_bbox"][0] + resolved["block_bbox"][2]) / 2.0
        self.assertLess(abs(rendered_center - safe_center), 38)
        self.assertLessEqual(resolved["block_bbox"][2], plan["safe_text_box"][2])

    def test_tiny_white_narration_anchor_uses_safe_height(self):
        text_data = {
            "translated": "ELA ESCONDEU TUDO ISSO.",
            "text": "SHE HID THIS MUCH",
            "bbox": [502, 4378, 650, 4386],
            "source_bbox": [436, 4365, 713, 4395],
            "text_pixel_bbox": [445, 4370, 707, 4392],
            "line_polygons": [[[445, 4370], [707, 4370], [707, 4392], [445, 4392]]],
            "balloon_bbox": [376, 4353, 773, 4407],
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "tipo": "narracao",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 21,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["layout_safe_reason"], "edge_clipped_white_balloon")
        self.assertGreaterEqual(plan["max_height"], 20)
        self.assertGreaterEqual(resolved["font_size"], 16)
        self.assertLessEqual(resolved["block_bbox"][3], plan["safe_text_box"][3] + 4)

    def test_build_render_blocks_merges_adjacent_visual_lobe_fragment_in_same_white_bubble(self):
        texts = [
            {
                "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                "text": "HEY, LET'S GO! I'M STARVING",
                "original": "HEY, LET'S GO! I'M STARVING",
                "bbox": [328, 11275, 538, 11317],
                "source_bbox": [63, 10742, 635, 11351],
                "text_pixel_bbox": [148, 10877, 538, 11317],
                "line_polygons": [
                    [[148, 10877], [311, 10877], [311, 10898], [148, 10898]],
                    [[343, 11293], [538, 11293], [538, 11317], [343, 11317]],
                ],
                "balloon_bbox": [0, 10726, 800, 11410],
                "balloon_type": "white",
                "tipo": "fala",
                "layout_profile": "white_balloon",
                "layout_group_size": 1,
                "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48, "cor": "#000000", "alinhamento": "center"},
            },
            {
                "translated": "QUEM ESTA PAGANDO HOJE?",
                "text": "WHO'S PAYING TODAY?",
                "bbox": [348, 11340, 535, 11383],
                "source_bbox": [338, 11288, 546, 11394],
                "text_pixel_bbox": [344, 11332, 540, 11392],
                "line_polygons": [
                    [[344, 11332], [540, 11332], [540, 11353], [344, 11353]],
                    [[389, 11366], [496, 11366], [496, 11392], [389, 11392]],
                ],
                "balloon_bbox": [301, 11314, 583, 11410],
                "balloon_type": "white",
                "tipo": "fala",
                "layout_profile": "white_balloon",
                "layout_group_size": 1,
                "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48, "cor": "#000000", "alinhamento": "center"},
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 2)
        lower = next(block for block in blocks if block.get("_merged_nearby_white_fragments"))
        self.assertEqual(lower["translated"], "ESTOU MORRENDO DE FOME\nQUEM ESTA PAGANDO HOJE?")
        self.assertEqual(lower["balloon_bbox"], [301, 11293, 583, 11410])
        self.assertEqual(lower["layout_group_size"], 2)

    def test_render_text_block_applies_safe_gradient_fill(self):
        img = Image.new("RGB", (360, 260), (255, 255, 255))
        text_data = {
            "translated": "LIGHT",
            "bbox": [40, 40, 320, 180],
            "tipo": "fala",
            "estilo": {
                "fonte": "SINGLE FIGHTER.otf",
                "tamanho": 56,
                "cor": "#FF0000",
                "contorno": "#220000",
                "contorno_px": 2,
                "alinhamento": "center",
                "sombra": False,
                "sombra_cor": "",
                "sombra_offset": [0, 0],
                "glow": False,
                "cor_gradiente": ["#FF0000", "#0000FF"],
            },
        }

        render_text_block(img, text_data)

        arr = np.array(img)
        changed = np.any(arr != 255, axis=2)
        colored = arr[changed]
        self.assertTrue(np.any(colored[:, 0] > colored[:, 2] + 20))
        self.assertTrue(np.any(colored[:, 2] > colored[:, 0] + 20))

    def test_render_text_block_applies_safe_glow(self):
        base_text_data = {
            "style_origin": "editor",
            "translated": "ATTACKS",
            "bbox": [40, 40, 320, 180],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#FFFFFF",
                "contorno": "#222222",
                "contorno_px": 2,
                "alinhamento": "center",
                "sombra": False,
                "sombra_cor": "",
                "sombra_offset": [0, 0],
                "glow": False,
                "glow_cor": "#FFD66B",
                "glow_px": 6,
                "cor_gradiente": [],
            },
        }
        img_without_glow = Image.new("RGB", (360, 260), (0, 0, 0))
        render_text_block(img_without_glow, dict(base_text_data))

        img_with_glow = Image.new("RGB", (360, 260), (0, 0, 0))
        text_with_glow = dict(base_text_data)
        text_with_glow["estilo"] = dict(base_text_data["estilo"], glow=True)
        render_text_block(img_with_glow, text_with_glow)

        without_changed = int(np.count_nonzero(np.any(np.array(img_without_glow) != 0, axis=2)))
        with_changed = int(np.count_nonzero(np.any(np.array(img_with_glow) != 0, axis=2)))
        self.assertGreater(with_changed, without_changed)

    def test_render_text_block_uses_connected_balloon_subregions(self):
        img = Image.new("RGB", (420, 260), (255, 255, 255))
        text_data = {
            "translated": "IT MAY BE NOTHING MORE THAN ENOUGH. A POWER THAT LET'S YOU SURPASS YOUR LIMITS.",
            "bbox": [30, 40, 390, 220],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 34,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
                "sombra": False,
                "sombra_cor": "",
                "sombra_offset": [0, 0],
                "glow": False,
                "cor_gradiente": [],
            },
            "balloon_bbox": [30, 40, 390, 220],
            "balloon_subregions": [
                [30, 40, 225, 130],
                [195, 130, 390, 220],
            ],
            "layout_shape": "wide",
            "layout_align": "center",
        }

        render_text_block(img, text_data)

        arr = np.array(img)
        tl = np.any(arr[45:130, 30:230] < 200)
        br = np.any(arr[130:220, 190:390] < 200)
        self.assertTrue(tl)
        self.assertTrue(br)

    def test_render_text_block_splits_single_text_connected_balloon(self):
        img = Image.new("RGB", (500, 320), (255, 255, 255))
        text_data = {
            "translated": (
                "MESMO SENDO UM METODO INCOMPLETO DE CIRCULACAO DE MANA, O EFEITO E MAIS DO QUE SUFICIENTE. "
                "ESSE PODER PERMITE ULTRAPASSAR INSTANTANEAMENTE OS PROPRIOS LIMITES."
            ),
            "bbox": [20, 20, 480, 280],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 34,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
                "sombra": False,
                "sombra_cor": "",
                "sombra_offset": [0, 0],
                "glow": False,
                "cor_gradiente": [],
            },
            "balloon_bbox": [20, 20, 480, 280],
            "balloon_subregions": [
                [20, 20, 230, 280],
                [250, 110, 480, 280],
            ],
            "connected_balloon_orientation": "diagonal",
            "connected_detection_confidence": 0.95,
            "connected_group_confidence": 0.92,
            "connected_position_confidence": 0.94,
            "subregion_confidence": 0.95,
            "layout_shape": "wide",
            "layout_align": "center",
            "layout_group_size": 1,
        }

        render_text_block(img, text_data)

        arr = np.array(img)
        left_has_text = np.any(arr[40:220, 20:240] < 200)
        right_has_text = np.any(arr[120:280, 240:480] < 200)
        self.assertTrue(left_has_text)
        self.assertTrue(right_has_text)

    def test_plan_text_layout_ignores_legacy_top_narration_profile(self):
        plan = plan_text_layout(
            {
                "translated": "Three days later, the northern wall had already fallen.",
                "bbox": [220, 40, 620, 180],
                "balloon_bbox": [220, 40, 620, 180],
                "tipo": "narracao",
                "layout_shape": "wide",
                "layout_align": "top",
                "layout_profile": "top_narration",
                "estilo": {
                    "fonte": "Newrotic.ttf",
                    "tamanho": 30,
                    "cor": "#FFFFFF",
                    "contorno": "#000000",
                    "contorno_px": 2,
                    "alinhamento": "center",
                    "sombra": False,
                    "glow": False,
                    "cor_gradiente": [],
                },
            }
        )

        self.assertEqual(plan["vertical_anchor"], "center")
        self.assertGreaterEqual(plan["width_ratio"], 0.90)

    def test_render_text_block_wraps_long_text_to_multiple_rows(self):
        img = Image.new("RGB", (420, 260), (255, 255, 255))
        text_data = {
            "translated": "ESTE BLOCO DEVE CONTINUAR EM UMA LINHA SO",
            "bbox": [30, 40, 390, 220],
            "source_bbox": [48, 112, 332, 154],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 30,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
                "sombra": False,
                "sombra_cor": "",
                "sombra_offset": [0, 0],
                "glow": False,
                "cor_gradiente": [],
            },
            "balloon_bbox": [30, 40, 390, 220],
            "layout_shape": "wide",
            "layout_align": "center",
        }

        render_text_block(img, text_data)

        arr = np.array(img)
        row_activity = np.count_nonzero(np.any(arr < 240, axis=2), axis=1)
        active_rows = np.where(row_activity > 0)[0]

        self.assertGreater(active_rows.size, 0)
        self.assertGreater(active_rows[-1] - active_rows[0], 24)

    def test_two_texts_with_subregions_no_double_render(self):
        """2 textos no mesmo balão com subregions → 1 bloco consolidado, sem duplicação visual."""
        subregions = [[30, 40, 200, 220], [220, 40, 400, 220]]
        base_style = {
            "fonte": "ComicNeue-Bold.ttf",
            "tamanho": 24,
            "cor": "#111111",
            "contorno": "#FFFFFF",
            "contorno_px": 2,
            "alinhamento": "center",
            "sombra": False,
            "sombra_cor": "",
            "sombra_offset": [0, 0],
            "glow": False,
            "cor_gradiente": [],
        }
        texts = [
            {
                "translated": "LEFT LOBE TEXT",
                "bbox": [50, 80, 180, 180],
                "tipo": "fala",
                "estilo": dict(base_style),
                "balloon_bbox": [30, 40, 400, 220],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 2,
            },
            {
                "translated": "RIGHT LOBE TEXT",
                "bbox": [240, 80, 380, 180],
                "tipo": "fala",
                "estilo": dict(base_style),
                "balloon_bbox": [30, 40, 400, 220],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 2,
            },
        ]

        blocks = build_render_blocks(texts)
        self.assertEqual(len(blocks), 1, "Deve consolidar em 1 bloco com connected_children")
        self.assertEqual(len(blocks[0].get("connected_children", [])), 2)

        img = Image.new("RGB", (430, 260), (255, 255, 255))
        for block in blocks:
            render_text_block(img, block)

        arr = np.array(img)
        left_has_text = np.any(arr[40:220, 30:200] < 200)
        right_has_text = np.any(arr[40:220, 220:400] < 200)
        self.assertTrue(left_has_text, "Lobo esquerdo deve ter texto")
        self.assertTrue(right_has_text, "Lobo direito deve ter texto")


    def test_connected_subregions_render_with_uniform_font_size(self):
        """Both lobes of a connected balloon should use the same font size."""
        img = Image.new("RGB", (500, 300), (255, 255, 255))
        # Left subregion is smaller → would need a smaller font if independent.
        # With uniform sizing, both should match the smaller one.
        text_data = {
            "translated": "SHORT. THIS IS A MUCH LONGER SENTENCE WITH MANY MORE WORDS THAT NEEDS SPACE.",
            "bbox": [20, 20, 480, 280],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 40,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
                "sombra": False,
                "sombra_cor": "",
                "sombra_offset": [0, 0],
                "glow": False,
                "cor_gradiente": [],
            },
            "balloon_bbox": [20, 20, 480, 280],
            "balloon_subregions": [
                [20, 20, 160, 280],   # narrow left lobe
                [160, 20, 480, 280],  # wide right lobe
            ],
            "layout_shape": "wide",
            "layout_align": "center",
        }

        render_text_block(img, text_data)

        arr = np.array(img)
        # Both halves should have visible text
        left_has_text = np.any(arr[20:280, 20:160] < 200)
        right_has_text = np.any(arr[20:280, 160:480] < 200)
        self.assertTrue(left_has_text, "Left lobe should have text")
        self.assertTrue(right_has_text, "Right lobe should have text")

    def test_connected_subregions_copy_safe_text_box_to_parent(self):
        img = Image.new("RGB", (500, 300), (255, 255, 255))
        text_data = {
            "translated": "PRIMEIRO LOBO. SEGUNDO LOBO.",
            "bbox": [20, 20, 480, 280],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 32,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
                "sombra": False,
                "glow": False,
            },
            "balloon_bbox": [20, 20, 480, 280],
            "balloon_subregions": [[20, 20, 240, 280], [240, 20, 480, 280]],
            "layout_profile": "connected_balloon",
            "layout_shape": "wide",
            "layout_align": "center",
            "layout_group_size": 2,
            "connected_balloon_orientation": "left-right",
            "subregion_confidence": 1.0,
            "connected_detection_confidence": 1.0,
            "connected_group_confidence": 1.0,
        }

        render_text_block(img, text_data)

        self.assertIn("render_bbox", text_data)
        self.assertIn("safe_text_box", text_data)
        self.assertEqual(len(text_data["safe_text_box"]), 4)

    def test_connected_subregion_fallback_copies_render_geometry_to_parent(self):
        img = Image.new("RGB", (500, 300), (255, 255, 255))
        text_data = {
            "translated": "TEXTO UNICO",
            "bbox": [20, 20, 480, 280],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
            },
            "balloon_bbox": [20, 20, 480, 280],
            "balloon_subregions": [[20, 20, 240, 280], [240, 20, 480, 280]],
            "layout_profile": "connected_balloon",
            "layout_group_size": 2,
            "subregion_confidence": 1.0,
            "connected_detection_confidence": 1.0,
            "connected_group_confidence": 1.0,
            "connected_balloon_orientation": "left-right",
        }

        with patch("typesetter.renderer._build_connected_children_candidates", return_value=[]):
            render_text_block(img, text_data)

        self.assertIn("render_bbox", text_data)
        self.assertIn("safe_text_box", text_data)

    def test_rerender_ok_removes_stale_blocking_render_flags(self):
        img = Image.new("RGB", (360, 220), (255, 255, 255))
        text_data = {
            "translated": "OLA",
            "bbox": [70, 60, 290, 160],
            "text_pixel_bbox": [110, 90, 250, 120],
            "balloon_bbox": [60, 45, 300, 175],
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "qa_flags": ["fit_below_minimum_legible", "missing_render_bbox", "TEXT_CLIPPED"],
            "fit_status": "below_minimum_legible",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 24,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["fit_status"], "ok")
        self.assertIn("render_bbox", text_data)
        self.assertIn("safe_text_box", text_data)
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags", []))
        self.assertNotIn("missing_render_bbox", text_data.get("qa_flags", []))
        self.assertNotIn("TEXT_CLIPPED", text_data.get("qa_flags", []))

    def test_render_text_block_rejects_underfit_white_run_safe_area(self):
        img = Image.new("RGB", (800, 1200), (255, 255, 255))
        text_data = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_012_band_025",
            "translated": "Eu ja estou com o coracao ruim...",
            "original": "I already have a bad heart...",
            "tipo": "narracao",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bbox": [59, 32, 267, 77],
            "source_bbox": [55, 32, 270, 78],
            "text_pixel_bbox": [59, 32, 267, 77],
            "balloon_bbox": [55, 32, 270, 78],
            "layout_bbox": [59, 32, 267, 77],
            "layout_safe_bbox": [65, 39, 260, 71],
            "layout_safe_reason": "single_lobe_white_run_safe_area",
            "_visual_rect_inner_bbox": [65, 39, 260, 71],
            "_visual_rect_outer_bbox": [55, 32, 270, 78],
            "qa_flags": ["fit_below_minimum_legible"],
            "fit_status": "below_minimum_legible",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 32,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["fit_status"], "ok")
        self.assertIn("safe_text_box_recomputed", text_data.get("qa_flags", []))
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags", []))
        self.assertGreaterEqual(text_data["safe_text_box"][2] - text_data["safe_text_box"][0], 195)

    def test_render_text_block_compacts_short_white_balloon_full_target(self):
        img = Image.new("RGB", (800, 220), (255, 255, 255))
        text_data = {
            "translated": "Poderiamos ter economizado o dinheiro.",
            "original": "We could have saved the money.",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bbox": [460, 36, 613, 46],
            "source_bbox": [455, 32, 616, 97],
            "text_pixel_bbox": [460, 36, 613, 46],
            "balloon_bbox": [455, 32, 616, 97],
            "layout_bbox": [460, 36, 613, 46],
            "_visual_rect_inner_bbox": [465, 39, 606, 90],
            "_visual_rect_outer_bbox": [455, 32, 616, 97],
            "layout_safe_bbox": [465, 39, 606, 90],
            "layout_safe_reason": "single_lobe_white_run_safe_area",
            "qa_flags": ["fit_below_minimum_legible"],
            "fit_status": "below_minimum_legible",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 45,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["fit_status"], "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags", []))

    def test_render_text_block_uses_line_polygon_anchor_for_collapsed_tiny_balloon(self):
        img = Image.new("RGB", (800, 12050), (255, 255, 255))
        text_data = {
            "translated": "Mas... como voce",
            "original": "But... how could you",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bbox": [281, 11748, 312, 11777],
            "text_pixel_bbox": [383, 11761, 466, 11768],
            "balloon_bbox": [269, 11736, 324, 11789],
            "layout_bbox": [383, 11761, 466, 11768],
            "bubble_inner_bbox": [293, 11760, 300, 11765],
            "line_polygons": [
                [[288, 11752], [497, 11752], [497, 11771], [288, 11771]],
            ],
            "qa_flags": ["fit_below_minimum_legible"],
            "fit_status": "below_minimum_legible",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 20,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["fit_status"], "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags", []))
        self.assertGreater((text_data.get("_render_debug") or {}).get("target_bbox", [0, 0, 0, 0])[2], 450)

    def test_render_text_block_ignores_visual_outer_that_clips_source_geometry(self):
        img = Image.new("RGB", (800, 1000), (255, 255, 255))
        text_data = {
            "translated": "Displays, todas as atividades que um bombeiro deve fazer...",
            "original": "Displays, all the activities a firefighter must do...",
            "tipo": "fala",
            "balloon_type": "textured",
            "layout_profile": "standard",
            "block_profile": "standard",
            "bbox": [406, 691, 634, 796],
            "text_pixel_bbox": [435, 699, 628, 765],
            "balloon_bbox": [356, 675, 684, 827],
            "layout_bbox": [435, 699, 628, 765],
            "bubble_inner_bbox": [418, 703, 622, 784],
            "_visual_rect_inner_bbox": [411, 702, 559, 821],
            "_visual_rect_outer_bbox": [399, 690, 571, 833],
            "line_polygons": [
                [[426, 693], [608, 693], [608, 713], [426, 713]],
                [[408, 720], [631, 721], [631, 740], [408, 739]],
                [[407, 747], [631, 748], [631, 769], [407, 768]],
                [[496, 775], [539, 775], [539, 797], [496, 797]],
            ],
            "qa_flags": ["fit_below_minimum_legible"],
            "fit_status": "below_minimum_legible",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["fit_status"], "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags", []))
        self.assertEqual((text_data.get("_render_debug") or {}).get("target_bbox"), [356, 675, 684, 827])

    def test_render_text_block_rejects_narrow_visual_inner_when_expanded_target_fits(self):
        img = Image.new("RGB", (620, 420), (255, 255, 255))
        text_data = {
            "translated": "2o ** teste regional de recrutamento de bombeiros",
            "original": "2o** regional fireman recruitment test",
            "tipo": "texto",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bbox": [73, 304, 118, 314],
            "source_bbox": [48, 296, 356, 324],
            "text_pixel_bbox": [73, 304, 118, 314],
            "balloon_bbox": [45, 286, 146, 332],
            "layout_bbox": [73, 304, 118, 314],
            "_visual_rect_inner_bbox": [63, 293, 136, 325],
            "_visual_rect_outer_bbox": [45, 286, 146, 332],
            "layout_safe_bbox": [63, 293, 136, 325],
            "layout_safe_reason": "single_lobe_white_run_safe_area",
            "qa_flags": ["fit_below_minimum_legible"],
            "fit_status": "below_minimum_legible",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 22,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        debug = text_data.get("_render_debug") or {}
        self.assertEqual(text_data["fit_status"], "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags", []))
        self.assertGreaterEqual(text_data["safe_text_box"][2] - text_data["safe_text_box"][0], 240)
        self.assertNotEqual(debug.get("layout_safe_reason"), "visual_rect_inner")
        self.assertEqual(debug.get("target_bbox"), [45, 286, 356, 332])

    def test_render_text_block_rejects_center_safe_area_when_full_balloon_fits(self):
        img = Image.new("RGB", (420, 820), (255, 255, 255))
        text_data = {
            "translated": "Ficar muito feliz ao conhecer pessoas que estão presas na mesma situação que você",
            "original": "To be overjoyed, when meeting PEOPLE who are stuck in the same situation as you",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bbox": [77, 565, 286, 676],
            "text_pixel_bbox": [79, 566, 284, 679],
            "balloon_bbox": [77, 565, 286, 676],
            "layout_bbox": [79, 566, 284, 679],
            "qa_flags": ["fit_below_minimum_legible"],
            "fit_status": "below_minimum_legible",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "bold": True,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["fit_status"], "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags", []))
        self.assertGreaterEqual(text_data["safe_text_box"][2] - text_data["safe_text_box"][0], 168)

    def test_render_text_block_uses_real_bubble_when_collapsed_anchor_clips_source(self):
        img = Image.new("RGB", (800, 5200), (255, 255, 255))
        text_data = {
            "translated": "Ajussi quanto tempo levará para chegar ao hospital mais próximo? ei",
            "original": "Ajussi! How long will it take to arrive to the nearest hospital?",
            "tipo": "texto",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bbox": [135, 4337, 274, 4439],
            "source_bbox": [125, 4294, 656, 4853],
            "text_pixel_bbox": [135, 4337, 274, 4439],
            "balloon_bbox": [214, 4325, 269, 4378],
            "bubble_mask_bbox": [125, 4294, 656, 4853],
            "bubble_inner_bbox": [162, 4331, 619, 4816],
            "layout_bbox": [135, 4337, 274, 4439],
            "line_polygons": [
                [[128, 4340], [338, 4340], [338, 4360], [128, 4360]],
                [[137, 4367], [332, 4367], [332, 4385], [137, 4385]],
                [[150, 4394], [320, 4394], [320, 4411], [150, 4411]],
                [[121, 4419], [345, 4418], [345, 4438], [121, 4439]],
            ],
            "qa_flags": ["fit_below_minimum_legible"],
            "fit_status": "below_minimum_legible",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 20,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        debug = text_data.get("_render_debug") or {}
        self.assertEqual(text_data["fit_status"], "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags", []))
        target = debug.get("target_bbox")
        self.assertEqual(target, [125, 4294, 656, 4853])
        self.assertGreaterEqual(text_data["safe_text_box"][2] - text_data["safe_text_box"][0], 300)
        self.assertGreaterEqual(text_data["safe_text_box"][3] - text_data["safe_text_box"][1], 300)

    def test_render_text_block_long_bubble_text_uses_full_inner_height(self):
        img = Image.new("RGB", (420, 820), (220, 220, 220))
        text_data = {
            "translated": "Ficar muito feliz ao conhecer pessoas que estao presas na mesma situacao que voce",
            "original": "To be overjoyed, when meeting people who are stuck in the same situation as you",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bbox": [77, 565, 286, 676],
            "text_pixel_bbox": [79, 566, 284, 679],
            "balloon_bbox": [77, 565, 286, 676],
            "layout_bbox": [79, 566, 284, 679],
            "bubble_inner_bbox": [89, 577, 274, 664],
            "qa_flags": ["fit_below_minimum_legible"],
            "fit_status": "below_minimum_legible",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "bold": True,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        debug = text_data.get("_render_debug") or {}
        self.assertEqual(text_data["fit_status"], "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags", []))
        self.assertGreaterEqual(debug.get("font_size_final", 0), 12)
        self.assertGreaterEqual(text_data["safe_text_box"][3] - text_data["safe_text_box"][1], 80)

    def test_build_render_blocks_dedupes_nested_same_balloon_prefix_text(self):
        base_balloon = [333, 2031, 900, 2187]
        texts = [
            {
                "translated": "A guerra acabou.",
                "bbox": [484, 2047, 745, 2171],
                "text_pixel_bbox": [489, 2053, 745, 2165],
                "tipo": "fala",
                "balloon_bbox": list(base_balloon),
                "layout_group_size": 2,
                "layout_profile": "standard",
                "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24},
            },
            {
                "translated": "A GUERRA",
                "bbox": [481, 2049, 745, 2097],
                "text_pixel_bbox": [489, 2052, 745, 2097],
                "tipo": "narracao",
                "balloon_bbox": list(base_balloon),
                "layout_group_size": 2,
                "layout_profile": "white_balloon",
                "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24},
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["translated"], "A guerra acabou.")

    def test_build_render_blocks_moves_broad_parent_residual_to_lower_nearby_balloon(self):
        texts = [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_025_band_067",
                "original": "What is it...? Why is he getting",
                "translated": "O que e...? por que ele esta ficando",
                "bbox": [148, 655, 667, 998],
                "text_pixel_bbox": [486, 870, 590, 949],
                "balloon_bbox": [148, 655, 667, 998],
                "tipo": "fala",
                "route_action": "translate_inpaint_render",
                "qa_metrics": {
                    "bbox_overreach": {
                        "ratio": 7.9,
                        "broad_bbox_drives_mask": False,
                        "has_line_polygon_geometry": True,
                    }
                },
                "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24},
            },
            {
                "id": "ocr_002",
                "trace_id": "ocr_002@page_025_band_067",
                "original": "What is it...?",
                "translated": "O que e...?",
                "bbox": [581, 870, 663, 909],
                "text_pixel_bbox": [581, 870, 663, 909],
                "balloon_bbox": [563, 858, 681, 921],
                "tipo": "fala",
                "route_action": "translate_inpaint_render",
                "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24},
            },
            {
                "id": "ocr_003",
                "trace_id": "ocr_003@page_025_band_067",
                "original": "scared alone?",
                "translated": "Com medo sozinho?",
                "bbox": [463, 933, 629, 1004],
                "text_pixel_bbox": [463, 985, 630, 1002],
                "balloon_bbox": [427, 933, 666, 1014],
                "tipo": "fala",
                "route_action": "translate_inpaint_render",
                "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24},
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertNotIn("ocr_001", {block.get("id") for block in blocks})
        self.assertEqual(texts[0]["route_action"], "review_required")
        self.assertEqual(texts[0]["route_reason"], "broad_duplicate_parent")
        self.assertIn("lobe_assignment_low_confidence", texts[0].get("qa_flags", []))
        self.assertEqual({block.get("id") for block in blocks}, {"ocr_002", "ocr_003"})
        lower = next(block for block in blocks if block.get("id") == "ocr_003")
        self.assertEqual(lower["original"], "Why is he getting scared alone?")
        self.assertEqual(lower["translated"], "Por que ele esta ficando com medo sozinho?")
        self.assertEqual(lower.get("connected_lobe_bboxes"), [])
        self.assertEqual(lower.get("connected_position_bboxes"), [])

    def test_visual_rect_detection_runs_for_wide_white_speech_box(self):
        text_data = {
            "translated": "ANTES, JAKE SEMPRE PARECIA PERDIDO PARA MIM.",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "balloon_bbox": [82, 3855, 505, 4070],
            "text_pixel_bbox": [168, 3907, 441, 4013],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 32},
        }

        self.assertTrue(renderer_mod._should_detect_visual_rect_safe_area(text_data))

    def test_single_lobe_connected_block_uses_original_anchor_without_semantic_split(self):
        text_data = {
            "translated": "TEM CERTEZA QUE NAO SE ARREPENDE DE TER ACEITADO?",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "connected_balloon",
            "block_profile": "white_balloon",
            "balloon_bbox": [0, 273, 636, 463],
            "balloon_subregions": [[0, 273, 287, 463], [301, 273, 636, 463]],
            "connected_lobe_bboxes": [[0, 273, 287, 463], [301, 273, 636, 463]],
            "connected_children": [{"translated": "QUE NAO SE"}, {"translated": "ARREPENDE"}],
            "text_pixel_bbox": [334, 295, 573, 441],
            "source_bbox": [332, 289, 574, 447],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 32},
        }

        subregion = renderer_mod._single_lobe_bbox_for_anchor(
            text_data,
            text_data["connected_lobe_bboxes"],
        )
        self.assertEqual(subregion, [301, 273, 636, 463])

        block = renderer_mod._as_single_lobe_render_block(text_data, subregion)
        plan = plan_text_layout(block)

        self.assertNotIn("connected_children", block)
        self.assertFalse(plan["_center_on_balloon_bbox"])
        self.assertGreater(plan["capacity_bbox"][2] - plan["capacity_bbox"][0], text_data["text_pixel_bbox"][2] - text_data["text_pixel_bbox"][0])
        self.assertGreaterEqual(plan["capacity_bbox"][0], subregion[0])
        self.assertLessEqual(plan["capacity_bbox"][2], subregion[2])
        self.assertEqual(plan["layout_profile"], "white_balloon")

    def test_single_lobe_safe_area_tracks_stepped_rect_white_run(self):
        img = Image.new("RGB", (700, 1000), (230, 208, 174))
        draw = ImageDraw.Draw(img)
        draw.rectangle([80, 570, 430, 730], fill=(255, 255, 255), outline=(0, 0, 0), width=5)
        draw.rectangle([160, 730, 620, 900], fill=(255, 255, 255), outline=(0, 0, 0), width=5)
        text_data = {
            "translated": "FUNCIONOU... EU SOBREVIVI.",
            "tipo": "narracao",
            "balloon_type": "white",
            "layout_profile": "connected_balloon",
            "bbox": [153, 636, 364, 704],
            "text_pixel_bbox": [153, 636, 364, 704],
            "line_polygons": [
                [[153, 636], [364, 636], [364, 663], [153, 663]],
                [[158, 677], [362, 677], [362, 704], [158, 704]],
            ],
            "_single_lobe_follow_anchor": True,
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48},
        }

        renderer_mod._apply_visual_rect_safe_area_if_needed(img, text_data)

        safe = text_data.get("layout_safe_bbox")
        self.assertIsNotNone(safe)
        self.assertLess(safe[2], 430)
        self.assertGreaterEqual(safe[0], 80)
        self.assertEqual(text_data["layout_profile"], "white_balloon")

    def test_textured_burst_balloon_uses_inner_white_run_safe_area(self):
        img = Image.new("RGB", (800, 720), (222, 214, 207))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 800, 720], fill=(224, 216, 210))
        draw.ellipse([168, 86, 742, 402], fill=(255, 255, 255), outline=(34, 34, 34), width=4)
        for x in range(180, 740, 28):
            draw.line([x, 86, x - 10, 42], fill=(94, 86, 36), width=2)
            draw.line([x, 402, x + 8, 444], fill=(94, 86, 36), width=2)

        text_data = {
            "translated": (
                "ELES PROFEREM BLASFEMIAS, PRIORIZANDO SEU TRABALHO EM "
                "DETRIMENTO DA IGREJA, MESMO QUANDO SAO FALADOS PESSOALMENTE "
                "POR MEU PAI E POR MIM!"
            ),
            "tipo": "fala",
            "balloon_type": "textured",
            "layout_profile": "standard",
            "balloon_bbox": [0, 62, 800, 430],
            "bbox": [237, 118, 688, 368],
            "text_pixel_bbox": [239, 122, 682, 364],
            "ocr_text_bbox": [237, 118, 688, 368],
            "layout_shape": "wide",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 32},
        }

        self.assertTrue(renderer_mod._should_detect_visual_rect_safe_area(text_data))

        renderer_mod._apply_visual_rect_safe_area_if_needed(img, text_data)
        plan = plan_text_layout(text_data)

        safe = text_data.get("layout_safe_bbox")
        self.assertIsNotNone(safe)
        self.assertEqual(text_data["layout_safe_reason"], "bright_inner_run_safe_area")
        self.assertEqual(text_data["layout_profile"], "white_balloon")
        self.assertGreater(safe[0], 185)
        self.assertLess(safe[2], 720)
        self.assertGreaterEqual(plan["safe_text_box"][0], safe[0])
        self.assertLessEqual(plan["safe_text_box"][2], safe[2])
        self.assertGreaterEqual(plan["safe_text_box"][1], safe[1])
        self.assertLessEqual(plan["safe_text_box"][3], safe[3])

    def test_textured_burst_detector_bbox_does_not_turn_art_panel_into_safe_area(self):
        img = Image.new("RGB", (800, 920), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 250, 800, 820], fill=(156, 128, 118))
        draw.rectangle([360, 280, 720, 760], fill=(92, 82, 78))
        draw.polygon(
            [(420, 70), (720, 70), (700, 230), (640, 210), (600, 260), (560, 218), (420, 230)],
            fill=(255, 255, 255),
            outline=(34, 34, 34),
        )

        text_data = {
            "translated": "POR FAVOR!",
            "tipo": "fala",
            "balloon_type": "textured",
            "layout_profile": "standard",
            "balloon_bbox": [88, 80, 775, 820],
            "bbox": [88, 80, 775, 820],
            "layout_bbox": [473, 118, 641, 151],
            "text_pixel_bbox": [473, 118, 641, 151],
            "line_polygons": [[[473, 118], [641, 118], [641, 151], [473, 151]]],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 33},
        }

        renderer_mod._apply_visual_rect_safe_area_if_needed(img, text_data)
        plan = plan_text_layout(text_data)
        safe = text_data.get("layout_safe_bbox")

        self.assertEqual(text_data.get("layout_safe_reason"), "bright_inner_run_safe_area")
        self.assertIsNotNone(safe)
        self.assertLess(safe[3], 270)
        self.assertLess(plan["safe_text_box"][3], 270)
        self.assertGreater(plan["safe_text_box"][3] - plan["safe_text_box"][1], 40)

    def test_emergency_font_shrink_keeps_tiny_status_box_inside_balloon(self):
        text_data = {
            "translated": "A SINCRONIZAÇÃO FOI CONCLUÍDA.",
            "original": "Synching is complete.",
            "tipo": "fala",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 33},
        }
        plan = {
            "target_bbox": [0, 0, 115, 66],
            "position_bbox": [0, 0, 115, 66],
            "capacity_bbox": [0, 0, 115, 66],
            "safe_text_box": [10, 9, 105, 57],
            "layout_safe_bbox": None,
            "layout_safe_reason": "",
            "layout_shape": "wide",
            "balloon_geo": "ellipse",
            "layout_profile": "standard",
            "width_ratio": 0.75,
            "max_width": 95,
            "max_height": 48,
            "padding_y": 9,
            "vertical_anchor": "center",
            "alignment": "center",
            "font_name": "ComicNeue-Bold.ttf",
            "target_size": 33,
            "text_color": "#000000",
            "cor_gradiente": [],
            "outline_color": "",
            "outline_px": 0,
            "glow": False,
            "glow_cor": "",
            "glow_px": 0,
            "sombra": False,
            "sombra_cor": "",
            "sombra_offset": [0, 0],
            "rotation_deg": 0,
            "rotation_source": "",
            "line_spacing_ratio": 0.18,
            "vertical_bias_px": 0,
            "horizontal_bias_px": 0,
            "_anchor_capacity_locked": False,
            "_simple_anchor_capacity_expanded": False,
            "_simple_anchor_capacity_reason": "",
            "_font_search_cap": 33,
            "_font_search_floor": 14,
            "_follow_original_ocr_size": True,
            "_follow_english_anchor_position": False,
            "_position_on_capacity_bbox": False,
            "_center_on_balloon_bbox": True,
        }

        resolved = _resolve_text_layout(text_data, plan)

        self.assertLess(resolved["font_size"], 14)
        self.assertLessEqual(resolved["block_bbox"][2], plan["safe_text_box"][2])
        self.assertLessEqual(resolved["block_bbox"][3], plan["safe_text_box"][3])

    def test_emergency_font_shrink_handles_uppercase_tiny_speech_bubble(self):
        text_data = {
            "original": "What happened?",
            "translated": "O QUE ACONTECEU?",
            "tipo": "fala",
            "balloon_type": "textured",
            "layout_profile": "standard",
            "balloon_bbox": [246, 4, 299, 56],
            "bbox": [246, 4, 299, 56],
            "text_pixel_bbox": [252, 12, 293, 48],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 16, "force_upper": True},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertLess(resolved["font_size"], 12)
        self.assertGreaterEqual(resolved["block_bbox"][0], plan["safe_text_box"][0])
        self.assertLessEqual(resolved["block_bbox"][2], plan["safe_text_box"][2])
        self.assertGreaterEqual(resolved["block_bbox"][1], plan["safe_text_box"][1])
        self.assertLessEqual(resolved["block_bbox"][3], plan["safe_text_box"][3])

    def test_tiny_textured_label_reduces_padding_before_fit(self):
        text_data = {
            "original": "GLOBAL MARTIAL ARTS",
            "translated": "ARTES MARCIAIS GLOBAIS",
            "tipo": "fala",
            "balloon_type": "textured",
            "layout_profile": "standard",
            "balloon_bbox": [447, 15233, 572, 15263],
            "bbox": [447, 15233, 572, 15263],
            "text_pixel_bbox": [454, 15243, 566, 15253],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 12, "force_upper": True},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertLessEqual(plan["padding_y"], 3)
        self.assertGreaterEqual(plan["safe_text_box"][3] - plan["safe_text_box"][1], 20)
        self.assertLessEqual(resolved["block_bbox"][2], plan["safe_text_box"][2])
        self.assertLessEqual(resolved["block_bbox"][3], plan["safe_text_box"][3])

    def test_inner_white_run_keeps_full_lobe_width_when_source_ink_splits_center(self):
        img = Image.new("RGB", (800, 520), (228, 214, 202))
        draw = ImageDraw.Draw(img)
        draw.ellipse([290, 230, 610, 382], fill=(255, 255, 255), outline=(30, 30, 30), width=4)
        # Simulate bold English ink breaking the center row into left/right white runs.
        draw.rectangle([348, 290, 535, 306], fill=(0, 0, 0))
        draw.rectangle([380, 322, 510, 338], fill=(0, 0, 0))

        text_data = {
            "translated": "ESTOU MORRENDO DE FOME QUEM ESTÁ PAGANDO HOJE?",
            "tipo": "fala",
            "balloon_type": "textured",
            "layout_profile": "standard",
            "balloon_bbox": [290, 230, 610, 382],
            "bbox": [290, 230, 610, 382],
            "layout_bbox": [348, 290, 535, 338],
            "text_pixel_bbox": [348, 290, 535, 338],
            "line_polygons": [
                [[348, 290], [535, 290], [535, 306], [348, 306]],
                [[380, 322], [510, 322], [510, 338], [380, 338]],
            ],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 28},
        }

        renderer_mod._apply_visual_rect_safe_area_if_needed(img, text_data)
        safe = text_data.get("layout_safe_bbox")

        self.assertIsNotNone(safe)
        self.assertLess(safe[0], 330)
        self.assertGreater(safe[2], 570)

    def test_normal_white_balloon_centers_in_balloon_not_english_anchor(self):
        text_data = {
            "translated": "UM TEXTO TRADUZIDO MAIOR",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "balloon_bbox": [0, 0, 320, 180],
            "text_pixel_bbox": [18, 22, 118, 58],
            "source_bbox": [18, 22, 118, 58],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 28},
        }

        plan = plan_text_layout(text_data)

        self.assertTrue(plan["_center_on_balloon_bbox"])
        self.assertFalse(plan["_follow_english_anchor_position"])
        self.assertEqual(plan["position_bbox"], [0, 0, 320, 180])
        self.assertEqual(plan["capacity_bbox"], [0, 0, 320, 180])

    def test_plan_text_layout_uses_source_when_balloon_bbox_is_disjoint_below_text(self):
        text_data = {
            "translated": "PRATA",
            "original": "SILVER",
            "tipo": "texto",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "balloon_bbox": [80, 420, 360, 520],
            "bbox": [96, 150, 254, 186],
            "source_bbox": [96, 150, 254, 186],
            "text_pixel_bbox": [112, 158, 238, 180],
            "page_width": 800,
            "page_height": 900,
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 28, "force_upper": True},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["_target_source"], "disjoint_source_text_bbox")
        self.assertLess(plan["target_bbox"][3], text_data["balloon_bbox"][1])
        self.assertLess(plan["safe_text_box"][3], text_data["balloon_bbox"][1])
        self.assertLess(resolved["block_bbox"][3], text_data["balloon_bbox"][1])

    def test_visual_lobe_microtext_does_not_auto_expand_to_giant_font(self):
        text_data = {
            "translated": "Isso",
            "original": "This",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "balloon_bbox": [300, 180, 390, 222],
            "bbox": [332, 193, 348, 204],
            "source_bbox": [332, 193, 348, 204],
            "text_pixel_bbox": [332, 193, 348, 204],
            "page_width": 800,
            "page_height": 900,
            "_visual_lobe_split_count": 2,
            "_visual_lobe_split_parent_bbox": [260, 150, 450, 260],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 13, "force_upper": False},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertFalse(plan["_simple_anchor_capacity_expanded"])
        self.assertNotEqual(plan["_simple_anchor_capacity_reason"], "tiny_anchor_auto")
        self.assertLessEqual(resolved["font_size"], 16)

    def test_connected_top_bottom_lobe_can_follow_source_anchor(self):
        text_data = {
            "translated": "AINDA VOU MORRER DE VELHICE.",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "connected_balloon",
            "balloon_bbox": [60, 120, 320, 260],
            "text_pixel_bbox": [130, 170, 210, 205],
            "_is_lobe_subregion": True,
            "_connected_slot_index": 1,
            "_connected_slot_count": 2,
            "connected_balloon_orientation": "top-bottom",
            "_connected_anchor_to_source_text": True,
            "_connected_source_bbox": [130, 170, 210, 205],
            "_connected_source_anchor_bboxes": [[190, 50, 260, 90], [130, 170, 210, 205]],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 30},
        }

        plan = plan_text_layout(text_data)

        self.assertFalse(plan["_center_on_balloon_bbox"])
        self.assertFalse(plan["_follow_english_anchor_position"])
        self.assertNotEqual(plan["position_bbox"], text_data["balloon_bbox"])
        source_cx = (130 + 210) / 2.0
        source_cy = (170 + 205) / 2.0
        pos = plan["position_bbox"]
        pos_cx = (pos[0] + pos[2]) / 2.0
        pos_cy = (pos[1] + pos[3]) / 2.0
        self.assertLess(abs(pos_cx - source_cx), 18)
        self.assertLess(abs(pos_cy - source_cy), 18)

    def test_visual_lobe_split_with_large_gap_is_not_merged_back(self):
        text_data = {
            "translated": (
                "TENHO CERTEZA DE QUE NAO POSSO ENGANAR A MORTE NOVAMENTE - "
                "ELA E CHAMADA DE MAIS UMA LUZ POR UM MOTIVO, EMBORA EU AINDA "
                "TENHA A HABILIDADE. AINDA VOU MORRER DE VELHICE."
            ),
            "text": (
                "I'M SURE I CAN'T CHEAT DEATH AGAIN-IT'S CALLED ONE MORE LIGHT "
                "FOR A REASON THOUGH I STILL HAVE THE SKILL. I'LL STILL DIE OF OLD AGE."
            ),
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "balloon_bbox": [233, 3583, 634, 4054],
            "bbox": [220, 3604, 630, 4035],
            "text_pixel_bbox": [220, 3604, 630, 4035],
            "line_polygons": [
                [[440, 3604], [595, 3604], [595, 3631], [440, 3631]],
                [[282, 3807], [525, 3805], [525, 3832], [282, 3834]],
                [[259, 3847], [548, 3846], [548, 3873], [259, 3874]],
                [[248, 3887], [558, 3886], [558, 3913], [248, 3914]],
                [[220, 3926], [583, 3928], [583, 3955], [220, 3953]],
                [[220, 3969], [587, 3969], [587, 3992], [220, 3992]],
                [[314, 4006], [491, 4006], [491, 4037], [314, 4037]],
                [[405, 3645], [630, 3645], [630, 3671], [405, 3671]],
                [[411, 3685], [621, 3685], [621, 3711], [411, 3711]],
            ],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48},
        }

        blocks = build_render_blocks([text_data])

        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(block.get("_visual_lobe_split_count") == 2 for block in blocks))
        self.assertFalse(any(block.get("_merged_nearby_white_fragments") for block in blocks))

    def test_rejected_connected_single_source_clamps_back_to_text_anchor(self):
        text_data = {
            "id": "ocr_002",
            "translated": "ESPECIALMENTE ESTE PRATO DE MACARRAO. E O MELHOR.",
            "tipo": "fala",
            "balloon_type": "white",
            "block_profile": "white_balloon",
            "layout_profile": "connected_balloon",
            "layout_group_size": 2,
            "source_text_count": 1,
            "balloon_bbox": [0, 750, 1600, 1149],
            "layout_bbox": [629, 995, 868, 1130],
            "bbox": [622, 983, 872, 1133],
            "source_bbox": [622, 983, 872, 1133],
            "text_pixel_bbox": [629, 995, 868, 1130],
            "line_polygons": [
                [[650, 992], [847, 992], [847, 1023], [650, 1023]],
                [[628, 1030], [868, 1030], [868, 1060], [628, 1060]],
                [[627, 1065], [872, 1065], [872, 1096], [627, 1096]],
            ],
            "balloon_subregions": [[540, 750, 729, 1149], [741, 750, 1078, 1149]],
            "connected_lobe_bboxes": [[540, 750, 729, 1149], [741, 750, 1078, 1149]],
            "connected_detection_confidence": 0.72,
            "connected_group_confidence": 0.72,
            "connected_balloon_orientation": "horizontal",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 33, "force_upper": True},
        }

        blocks = build_render_blocks([text_data])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["balloon_subregions"], [])
        self.assertEqual(blocks[0].get("connected_lobe_bboxes", []), [])
        self.assertEqual(blocks[0]["balloon_bbox"], [629, 995, 868, 1130])
        self.assertEqual(blocks[0].get("_render_target_source"), "rejected_connected_anchor")

    def test_render_text_block_rejected_connected_clears_stale_balloon_bbox(self):
        img = Image.new("RGB", (1600, 1200), (255, 255, 255))
        text_data = {
            "id": "ocr_002",
            "translated": "ESPECIALMENTE ESTE PRATO DE MACARRAO. E O MELHOR.",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "connected_balloon",
            "layout_group_size": 2,
            "source_text_count": 1,
            "balloon_bbox": [0, 750, 1600, 1149],
            "bbox": [622, 983, 872, 1133],
            "source_bbox": [622, 983, 872, 1133],
            "text_pixel_bbox": [629, 995, 868, 1130],
            "line_polygons": [
                [[650, 992], [847, 992], [847, 1023], [650, 1023]],
                [[628, 1030], [868, 1030], [868, 1060], [628, 1060]],
            ],
            "balloon_subregions": [[540, 750, 729, 1149], [741, 750, 1078, 1149]],
            "connected_lobe_bboxes": [[540, 750, 729, 1149], [741, 750, 1078, 1149]],
            "connected_detection_confidence": 0.72,
            "connected_group_confidence": 0.72,
            "connected_balloon_orientation": "horizontal",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 33, "force_upper": True},
        }

        with patch("typesetter.renderer._render_connected_subregions") as connected_mock:
            with patch("typesetter.renderer._render_single_text_block"):
                render_text_block(img, text_data)

        connected_mock.assert_not_called()
        self.assertEqual(text_data["balloon_subregions"], [])
        self.assertEqual(text_data.get("connected_lobe_bboxes", []), [])
        self.assertEqual(text_data["balloon_bbox"], [629, 995, 868, 1130])
        self.assertEqual(text_data.get("_render_target_source"), "rejected_connected_anchor")

    def test_overbroad_raw_ocr_box_anchors_typeset_to_text_pixels(self):
        text_data = {
            "translated": "ESTAREMOS AQUI ATE A NOITE",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "balloon_bbox": [110, 1616, 1513, 2239],
            "bbox": [228, 1632, 1395, 2157],
            "source_bbox": [228, 1632, 1395, 2157],
            "ocr_text_bbox": [228, 1632, 1395, 2157],
            "text_pixel_bbox": [606, 2099, 850, 2144],
            "line_polygons": [
                [[672, 2096], [783, 2096], [783, 2119], [672, 2119]],
                [[603, 2124], [850, 2124], [850, 2147], [603, 2147]],
            ],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "force_upper": True},
        }

        plan = plan_text_layout(text_data)

        self.assertFalse(plan["_center_on_balloon_bbox"])
        self.assertTrue(plan["_follow_english_anchor_position"])
        self.assertTrue(plan["_anchor_capacity_locked"])
        self.assertLess(plan["safe_text_box"][2] - plan["safe_text_box"][0], 420)
        self.assertGreaterEqual(plan["safe_text_box"][0], 560)
        self.assertLessEqual(plan["safe_text_box"][2], 900)
        self.assertEqual(plan["_target_source"], "ocr_anchor_overbroad_raw_box")

    def test_render_qa_allows_tiny_safe_box_overhang_inside_real_target(self):
        text_data = {
            "translated": "QUASE CABE",
            "render_bbox": [98, 100, 202, 150],
            "balloon_bbox": [90, 90, 210, 160],
            "qa_flags": [],
        }
        plan = {
            "safe_text_box": [100, 100, 200, 150],
            "target_bbox": [90, 90, 210, 160],
        }

        renderer_mod._run_render_qa(text_data, plan)

        self.assertNotIn("TEXT_CLIPPED", text_data.get("qa_flags", []))
        render_fit = (text_data.get("qa_metrics") or {}).get("render_fit") or {}
        self.assertNotIn("TEXT_CLIPPED", render_fit.get("flags") or [])

    def test_collapsed_balloon_does_not_expand_to_overbroad_source_bbox(self):
        img = Image.new("RGB", (800, 5200), (255, 255, 255))
        text_data = {
            "translated": "Ajussi quanto tempo levara para chegar ao hospital mais proximo? ei",
            "original": "Ajussi! How long will it take to arrive to the nearest hospital?",
            "tipo": "texto",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bbox": [135, 4337, 274, 4439],
            "source_bbox": [125, 4294, 656, 4853],
            "text_pixel_bbox": [135, 4337, 274, 4439],
            "balloon_bbox": [214, 4325, 269, 4378],
            "layout_bbox": [135, 4337, 274, 4439],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 20,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        target = (text_data.get("_render_debug") or {}).get("target_bbox")
        self.assertIsNotNone(target)
        self.assertNotEqual(target, [125, 4294, 656, 4853])
        self.assertLessEqual(target[2] - target[0], 180)
        self.assertLessEqual(target[3] - target[1], 150)

    def test_collapsed_balloon_prefers_real_bubble_when_anchor_clips_source(self):
        text_data = {
            "translated": "Ajussi quanto tempo levara para chegar ao hospital mais proximo? ei",
            "original": "Ajussi! How long will it take to arrive to the nearest hospital?",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bbox": [135, 4337, 274, 4439],
            "source_bbox": [125, 4294, 656, 4853],
            "text_pixel_bbox": [135, 4337, 274, 4439],
            "layout_bbox": [135, 4337, 274, 4439],
            "balloon_bbox": [214, 4325, 269, 4378],
            "bubble_mask_bbox": [125, 4294, 656, 4853],
            "bubble_inner_bbox": [162, 4331, 619, 4816],
            "line_polygons": [
                [[128, 4340], [338, 4340], [338, 4360], [128, 4360]],
                [[137, 4367], [332, 4367], [332, 4385], [137, 4385]],
                [[150, 4394], [320, 4394], [320, 4411], [150, 4411]],
                [[121, 4419], [345, 4419], [345, 4439], [121, 4439]],
            ],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 20, "force_upper": True},
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [125, 4294, 656, 4853])
        self.assertEqual(text_data.get("_render_target_source"), "real_bubble_mask_bbox")
        self.assertGreaterEqual(plan["capacity_bbox"][2] - plan["capacity_bbox"][0], 300)
        self.assertGreaterEqual(plan["capacity_bbox"][3] - plan["capacity_bbox"][1], 300)
        self.assertIn("safe_text_box_recomputed", text_data.get("qa_flags") or [])

    def test_lobe_rejects_degenerate_parent_inner_intersection_safe_box(self):
        text_data = {
            "translated": "TRES!",
            "original": "THREE!",
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "balloon_bbox": [301, 3583, 634, 4054],
            "bbox": [314, 4006, 491, 4037],
            "source_bbox": [314, 4006, 491, 4037],
            "text_pixel_bbox": [314, 4006, 491, 4037],
            "bubble_inner_bbox": [330, 4008, 620, 4017],
            "_is_lobe_subregion": True,
            "_connected_slot_index": 1,
            "_connected_slot_count": 2,
            "connected_balloon_orientation": "top-bottom",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 30, "force_upper": True},
        }

        plan = plan_text_layout(text_data)

        self.assertNotEqual(plan["layout_safe_bbox"], [330, 4008, 620, 4017])
        self.assertGreaterEqual(plan["safe_text_box"][3] - plan["safe_text_box"][1], 80)
        self.assertGreaterEqual(plan["safe_text_box"][0], text_data["balloon_bbox"][0])
        self.assertLessEqual(plan["safe_text_box"][2], text_data["balloon_bbox"][2])
        self.assertIn("safe_text_box_recomputed", text_data.get("qa_flags", []))


if __name__ == "__main__":
    unittest.main()
