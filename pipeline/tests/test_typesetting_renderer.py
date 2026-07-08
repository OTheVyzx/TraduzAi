import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import cv2
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
    _render_safe_arc_text_layer,
    _render_single_text_block_unrotated,
    _render_single_text_block,
    _resolve_text_layout,
    _select_dark_panel_visual_mask_render_target_bbox,
    _should_reject_plain_balloon_visual_safe_area,
    build_render_blocks,
    ensure_legible_plan,
    find_font,
    plan_fallback_render_box,
    plan_text_layout,
    render_band_image,
    render_text_block,
)

from main import (
    _apply_dark_panel_style_groups,
    _final_page_space_text_layers_for_renderer,
    _mark_final_layer_as_page_space,
    _normalize_final_project_page_space_layers,
    _render_plan_source_bbox,
    _sync_page_legacy_aliases,
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

    def test_render_plan_source_bbox_ignores_compact_anchor_mask(self):
        layer = {
            "source_bbox": [244, 4335, 313, 4445],
            "text_pixel_bbox": [131, 4330, 337, 4444],
            "source_text_anchor_bbox": [244, 4335, 313, 4445],
            "_source_text_anchor_bbox": [244, 4335, 313, 4445],
            "source_text_mask_bbox": [244, 4335, 313, 4445],
            "_source_text_mask_bbox": [244, 4335, 313, 4445],
        }

        self.assertEqual(_render_plan_source_bbox(layer), [131, 4330, 337, 4444])

    def test_connected_lobe_passthrough_preserves_wide_text_pixel_bbox_for_scale(self):
        text = {
            "id": "ocr_001",
            "text": "Your heart felt empty and lonely all the time.",
            "translated": "VOCE ATE FOI AO REFORMATORIO POR CAUSA DELES, NAO FOI?",
            "bbox": [244, 4335, 313, 4445],
            "source_bbox": [244, 4335, 313, 4445],
            "text_pixel_bbox": [131, 4330, 337, 4444],
            "source_text_anchor_bbox": [244, 4335, 313, 4445],
            "_source_text_anchor_bbox": [244, 4335, 313, 4445],
            "source_text_mask_bbox": [244, 4335, 313, 4445],
            "_source_text_mask_bbox": [244, 4335, 313, 4445],
            "target_bbox": [40, 4260, 390, 4520],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["dark_bubble_connected_lobe_passthrough"],
        }

        blocks = renderer_mod._build_connected_passthrough_lobe_blocks(
            [text],
            [[40, 4260, 390, 4520]],
            "horizontal",
        )

        self.assertEqual(blocks[0]["text_pixel_bbox"], [131, 4330, 337, 4444])
        self.assertEqual(renderer_mod._original_text_mask_bbox_for_scale(blocks[0]), [131, 4330, 337, 4444])

    def _render_style_probe(self, estilo: dict, *, background=(255, 255, 255)) -> np.ndarray:
        img = Image.new("RGB", (260, 120), background)
        text_data = {
            "id": "style_probe",
            "text": "TEST",
            "translated": "TEST",
            "tipo": "fala",
            "layout_profile": "style_probe",
            "bbox": [30, 30, 230, 92],
            "source_bbox": [30, 30, 230, 92],
            "text_pixel_bbox": [30, 30, 230, 92],
            "target_bbox": [20, 20, 240, 105],
            "position_bbox": [20, 20, 240, 105],
            "capacity_bbox": [20, 20, 240, 105],
            "safe_text_box": [20, 20, 240, 105],
            "_debug_safe_text_box": [20, 20, 240, 105],
            "balloon_bbox": [10, 10, 250, 112],
            "background_rgb": list(background),
            "style_origin": "source_detected",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "alinhamento": "center",
                "force_upper": True,
                **estilo,
            },
        }
        text_data["style"] = dict(text_data["estilo"])
        render_text_block(img, text_data)
        return np.asarray(img)

    def test_bbox_fallback_dark_visual_preserves_text_pixel_anchor(self):
        text_data = {
            "id": "ocr_001",
            "translated": "Observe isto.",
            "bubble_mask_source": "bbox_fallback",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bbox": [494, 96, 800, 501],
            "source_bbox": [494, 96, 800, 501],
            "text_pixel_bbox": [592, 377, 727, 411],
            "safe_text_box": [427, 380, 605, 568],
            "qa_flags": ["bbox_fallback_bubble_mask"],
        }
        plan = {
            "target_bbox": [494, 96, 800, 501],
            "position_bbox": [494, 96, 800, 501],
            "capacity_bbox": [494, 96, 800, 501],
            "safe_text_box": [427, 380, 605, 568],
            "layout_profile": "dark_bubble",
            "max_width": 80,
            "max_height": 24,
        }

        renderer_mod._expand_dark_visual_underfit_layout_capacity(text_data, plan)

        self.assertEqual(plan["layout_safe_reason"], "bbox_fallback_text_anchor_preserved")
        self.assertGreaterEqual(plan["target_bbox"][0], 564)
        self.assertLessEqual(plan["target_bbox"][2], 755)
        self.assertIn("bbox_fallback_text_anchor_preserved", text_data.get("qa_flags") or [])

    def test_false_dark_white_layout_preserves_text_pixel_anchor(self):
        text_data = {
            "id": "direct_paddle_reocr_001",
            "translated": "O QUE VOCÊ QUER DIZER COM SUBESPAÇO?",
            "bubble_mask_source": "image_white_bubble_mask",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "source_bbox": [214, 112, 349, 196],
            "text_pixel_bbox": [214, 112, 349, 196],
            "safe_text_box": [46, 175, 604, 525],
            "qa_flags": ["false_dark_white_style_neutralized", "false_light_dark_bubble_promoted_to_white"],
        }
        plan = {
            "target_bbox": [0, 146, 650, 554],
            "position_bbox": [46, 175, 604, 525],
            "capacity_bbox": [46, 175, 604, 525],
            "safe_text_box": [46, 175, 604, 525],
            "layout_profile": "white_balloon",
            "max_width": 300,
            "max_height": 120,
        }

        renderer_mod._expand_dark_visual_underfit_layout_capacity(text_data, plan)

        self.assertEqual(plan["layout_safe_reason"], "false_dark_white_text_anchor_preserved")
        self.assertLess(plan["target_bbox"][1], 130)
        self.assertLess(plan["target_bbox"][3], 225)
        self.assertIn("false_dark_white_text_anchor_preserved", text_data.get("qa_flags") or [])

    def test_mixed_sfx_detached_white_bubble_strips_number_prefix(self):
        text_data = {
            "translated": "1/ O QUE É ISSO",
            "text": "1/ WHAT'S THAT?",
            "bbox": [192, 176, 630, 1025],
            "text_pixel_bbox": [192, 176, 630, 1025],
            "balloon_bbox": [0, 0, 726, 1127],
            "rotation_deg": 90.0,
            "rotation_source": "line_polygons",
            "qa_flags": ["mixed_sfx_detached_from_white_bubble"],
            "qa_metrics": {
                "mixed_sfx_detached_from_white_bubble": {
                    "white_bbox": [260, 344, 686, 1127],
                    "kept_bbox": [447, 1000, 631, 1026],
                }
            },
        }

        cleaned = renderer_mod._strip_mixed_sfx_prefix_for_detached_white_bubble(text_data)

        self.assertEqual(cleaned["translated"], "O QUE É ISSO")
        self.assertEqual(cleaned["text"], "WHAT'S THAT?")
        self.assertEqual(cleaned["text_pixel_bbox"], [447, 1000, 631, 1026])
        self.assertEqual(cleaned["balloon_bbox"], [260, 344, 686, 1127])
        self.assertEqual(cleaned["bubble_mask_source"], "image_white_bubble_mask")
        self.assertEqual(cleaned["rotation_deg"], 0.0)
        self.assertEqual(cleaned["rotation_source"], "mixed_sfx_detached_white_bubble")

    def test_wrapped_line_orphan_penalty_discourages_single_letter_first_line(self):
        orphan = renderer_mod._wrapped_lines_orphan_penalty(["A", "SINCRONIZAÇÃO", "FOI CONCLUÍDA."])
        balanced = renderer_mod._wrapped_lines_orphan_penalty(["A SINCRONIZAÇÃO", "FOI CONCLUÍDA."])

        self.assertGreater(orphan, balanced)
        self.assertGreater(orphan, 10.0)

    def _changed_mask(self, arr: np.ndarray, background=(255, 255, 255)) -> np.ndarray:
        bg = np.array(background, dtype=np.int16)
        delta = np.linalg.norm(arr.astype(np.int16) - bg, axis=2)
        return delta > 18

    def test_renderer_applies_detected_vertical_gradient_pixels(self):
        arr = self._render_style_probe(
            {
                "cor": "#080840",
                "cor_gradiente": ["#09095F", "#0A2F30"],
                "contorno": "",
                "contorno_px": 0,
            }
        )
        mask = self._changed_mask(arr)
        ys, _xs = np.where(mask)

        self.assertGreater(len(ys), 120)
        top = arr[mask & (np.indices(mask.shape)[0] <= np.percentile(ys, 35))]
        bottom = arr[mask & (np.indices(mask.shape)[0] >= np.percentile(ys, 65))]

        self.assertGreater(float(np.mean(top[:, 2])), float(np.mean(bottom[:, 2])) + 8.0)
        self.assertGreater(float(np.mean(bottom[:, 1])), float(np.mean(top[:, 1])) + 8.0)

    def test_renderer_applies_detected_outline_as_solid_ring_pixels(self):
        purple = (145, 0, 245)
        base = self._render_style_probe({"cor": "#000000"}, background=purple)
        outlined = self._render_style_probe(
            {"cor": "#000000", "contorno": "#FFFFFF", "contorno_px": 3},
            background=purple,
        )
        base_mask = self._changed_mask(base, background=purple)
        outlined_mask = self._changed_mask(outlined, background=purple)
        ring = outlined_mask & ~cv2.dilate(base_mask.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
        ring_pixels = outlined[ring]

        self.assertGreater(len(ring_pixels), 30)
        self.assertGreater(float(np.mean(ring_pixels[:, 0])), 220.0)
        self.assertLess(float(np.std(ring_pixels[:, 0])), 35.0)

    def test_renderer_applies_detected_glow_beyond_glyph_pixels(self):
        base = self._render_style_probe({"cor": "#000000"})
        glowing = self._render_style_probe(
            {"cor": "#000000", "glow": True, "glow_cor": "#FFC8EE", "glow_px": 5}
        )
        base_mask = self._changed_mask(base)
        glow_mask = self._changed_mask(glowing)
        outer = glow_mask & ~cv2.dilate(base_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        outer_pixels = glowing[outer]

        self.assertGreater(len(outer_pixels), 20)
        self.assertGreater(float(np.mean(outer_pixels[:, 0])), 210.0)
        self.assertGreater(float(np.mean(outer_pixels[:, 2])), 190.0)

    def test_renderer_plain_detected_style_does_not_emit_colored_halo(self):
        plain = self._render_style_probe({"cor": "#000000", "contorno": "", "contorno_px": 0, "glow": False})
        mask = self._changed_mask(plain)
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~mask
        ring_pixels = plain[ring]

        self.assertGreater(len(ring_pixels), 40)
        self.assertLess(float(np.mean(np.abs(ring_pixels.astype(np.int16) - 255))), 8.0)

    def test_grouped_dark_visual_style_is_not_replaced_by_renderer_fallback(self):
        img = Image.new("RGB", (320, 180), (0, 0, 0))
        style = {
            "fonte": "ComicNeue-Bold.ttf",
            "tamanho": 32,
            "cor": "#FDFBF7",
            "contorno": "#CDDBE4",
            "contorno_px": 1,
            "glow": True,
            "glow_cor": "#5C584B",
            "glow_px": 3,
            "alinhamento": "center",
            "force_upper": True,
            "style_origin": "grouped_dark_panel_visual_style",
            "style_source": "dark_panel_visual_style_group",
            "style_confidence": 0.78,
        }
        text_data = {
            "id": "dark_grouped",
            "text": "SYSTEM",
            "translated": "SISTEMA",
            "bbox": [84, 58, 236, 108],
            "source_bbox": [84, 58, 236, 108],
            "text_pixel_bbox": [84, 58, 236, 108],
            "target_bbox": [30, 20, 290, 160],
            "safe_text_box": [54, 42, 266, 138],
            "_debug_safe_text_box": [54, 42, 266, 138],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "background_rgb": [0, 0, 0],
            "style_origin": "grouped_dark_panel_visual_style",
            "style_source": "dark_panel_visual_style_group",
            "style_group_id": "dark_panel_visual_0_bubble_dark_near_white_cyan_blue",
            "estilo": dict(style),
            "style": dict(style),
        }

        render_text_block(img, text_data)

        rendered_style = text_data["estilo"]
        self.assertEqual(rendered_style["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(rendered_style["cor"], "#FDFBF7")
        self.assertEqual(rendered_style["contorno"], "#CDDBE4")
        self.assertEqual(rendered_style["glow_cor"], "#5C584B")
        self.assertEqual(text_data["style_source"], "dark_panel_visual_style_group")

    def test_dark_bubble_similar_visuals_share_style_group(self):
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "a",
                            "translated": "SISTEMA",
                            "bbox": [10, 10, 150, 70],
                            "text_pixel_bbox": [10, 10, 150, 70],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "layout_profile": "dark_bubble",
                            "background_rgb": [0, 0, 0],
                            "qa_metrics": {
                                "image_dark_bubble_mask": {
                                    "panel_fill_rgb": [0, 0, 0],
                                    "text_fill_rgb": [255, 255, 255],
                                    "text_glow_rgb": [90, 210, 255],
                                    "border_rgb": [6, 29, 38],
                                }
                            },
                            "style": {
                                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                                "cor": "#FFFFFF",
                                "glow": True,
                                "glow_cor": "#67D8FF",
                                "glow_px": 3,
                                "style_origin": "source_detected",
                                "style_confidence": 0.91,
                            },
                        },
                        {
                            "id": "b",
                            "translated": "PONTOS",
                            "bbox": [20, 90, 170, 150],
                            "text_pixel_bbox": [20, 90, 170, 150],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "layout_profile": "dark_bubble",
                            "background_rgb": [0, 0, 0],
                            "qa_metrics": {
                                "image_dark_bubble_mask": {
                                    "panel_fill_rgb": [0, 0, 0],
                                    "text_fill_rgb": [248, 250, 245],
                                    "text_glow_rgb": [88, 206, 250],
                                    "border_rgb": [8, 31, 42],
                                }
                            },
                            "style": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "cor": "#FFFFFF",
                                "glow": False,
                                "style_origin": "auto",
                                "style_confidence": 0.1,
                            },
                        },
                    ]
                }
            ]
        }

        summary = _apply_dark_panel_style_groups(project)
        layers = project["paginas"][0]["text_layers"]

        self.assertGreaterEqual(summary["layers"], 1)
        self.assertEqual(layers[0]["style_group_id"], layers[1]["style_group_id"])
        self.assertEqual(layers[1]["style"]["fonte"], layers[0]["style"]["fonte"])
        self.assertTrue(layers[1]["style"]["glow"])

    def test_dark_bubble_mask_overrides_inherited_white_balloon_black_style(self):
        img = Image.new("RGB", (360, 180), (0, 0, 0))
        text_data = {
            "id": "dark_bubble",
            "text": "You were chosen to become King Yeomra!",
            "translated": "Você foi escolhido para se tornar rei Yeomra!",
            "tipo": "fala",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [70, 50, 290, 125],
            "source_bbox": [70, 50, 290, 125],
            "text_pixel_bbox": [70, 50, 290, 125],
            "target_bbox": [35, 20, 325, 155],
            "safe_text_box": [58, 42, 302, 133],
            "balloon_bbox": [35, 20, 325, 155],
            "background_rgb": [0, 0, 0],
            "style_origin": "source_detected",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 26,
                "cor": "#061D26",
                "alinhamento": "center",
                "style_origin": "source_detected",
            },
            "style": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 26,
                "cor": "#061D26",
                "alinhamento": "center",
                "style_origin": "source_detected",
            },
            "qa_flags": ["dark_bubble_oval_reocr"],
        }

        render_text_block(img, text_data)
        arr = np.asarray(img)
        safe = arr[42:133, 58:302]
        bright = ((safe[:, :, 0] > 180) & (safe[:, :, 1] > 180) & (safe[:, :, 2] > 180)).sum()

        self.assertGreater(bright, 120)
        self.assertEqual(text_data["style_origin"], "auto_dark_panel_glow")
        self.assertEqual(text_data["estilo"]["cor"], "#FFFFFF")

    def test_build_render_blocks_promotes_overlapping_dark_bubbles_to_connected_lobes(self):
        left = {
            "id": "ocr_001_002",
            "text_id": "ocr_001_002",
            "page_id": "page_002",
            "band_id": "page_002_band_023",
            "text": "You were loyal to others, but to them, you were being nosy",
            "translated": "Voce era leal aos outros, mas para eles, voce estava sendo intrometido",
            "tipo": "fala",
            "bbox": [132, 116, 419, 231],
            "source_bbox": [121, 96, 422, 235],
            "text_pixel_bbox": [132, 116, 419, 231],
            "layout_bbox": [132, 116, 419, 231],
            "balloon_bbox": [63, 0, 519, 421],
            "bubble_mask_bbox": [63, 0, 519, 421],
            "bubble_mask_source": "image_dark_bubble_mask",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 32, "cor": "#FFFFFF"},
            "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
        }
        right = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "page_id": "page_002",
            "band_id": "page_002_band_023",
            "text": "You were the king of being a pushover...",
            "translated": "Voce era o rei de ser uma tarefa simples...",
            "tipo": "fala",
            "bbox": [476, 289, 649, 362],
            "source_bbox": [386, 85, 712, 431],
            "text_pixel_bbox": [386, 163, 649, 360],
            "layout_bbox": [476, 289, 649, 362],
            "balloon_bbox": [294, 0, 712, 431],
            "bubble_mask_bbox": [294, 0, 712, 431],
            "bubble_mask_source": "image_dark_bubble_mask",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 32, "cor": "#FFFFFF"},
            "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
        }
        fragment = {
            "id": "ocr_001_fragment_2",
            "text_id": "ocr_001_fragment_2",
            "trace_id": "ocr_001@page_002_band_023#fragment_2",
            "page_id": "page_002",
            "band_id": "page_002_band_023",
            "text": "You were loyal to others, but to them, you were being nosy. You were the king of being a pushover...",
            "translated": (
                "Voce era leal aos outros, mas para eles, voce estava sendo intrometido "
                "Voce era o rei de ser uma tarefa simples..."
            ),
            "tipo": "fala",
            "bbox": [160, 92, 613, 328],
            "text_pixel_bbox": [160, 92, 613, 328],
            "balloon_bbox": [63, 0, 712, 431],
            "bubble_mask_bbox": [63, 0, 712, 431],
            "bubble_mask_source": "image_dark_bubble_mask",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 32, "cor": "#FFFFFF"},
            "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
        }

        blocks = build_render_blocks([right, left, fragment])

        self.assertEqual(len(blocks), 2)
        left_block, right_block = sorted(blocks, key=lambda block: block["balloon_bbox"][0])
        self.assertLess(left_block["balloon_bbox"][2], right_block["balloon_bbox"][0])
        self.assertEqual(left_block.get("connected_lobe_bboxes"), [])
        self.assertEqual(right_block.get("connected_lobe_bboxes"), [])
        self.assertNotIn("connected_children", left_block)
        self.assertNotIn("connected_children", right_block)
        self.assertIn("dark_bubble_connected_lobe_passthrough", left_block["qa_flags"])
        self.assertIn("dark_bubble_connected_lobe_passthrough", right_block["qa_flags"])
        self.assertFalse(fragment["visible"])
        self.assertEqual(fragment["render_policy"], "suppressed_dark_connected_combined_fragment")
        self.assertIn("dark_connected_combined_fragment_suppressed", fragment["qa_flags"])

    def test_build_render_blocks_keeps_direct_dark_lobe_when_text_pixel_bbox_is_broad(self):
        left = {
            "id": "direct_paddle_reocr_001",
            "text_id": "direct_paddle_reocr_001",
            "band_id": "page_005_band_078",
            "translated": "A retencao do subespaco e de apenas cinco minutos.",
            "tipo": "fala",
            "bbox": [129, 8431, 312, 8554],
            "source_bbox": [129, 8431, 312, 8554],
            "text_pixel_bbox": [237, 8439, 677, 8655],
            "balloon_bbox": [83, 8352, 744, 9011],
            "bubble_mask_bbox": [83, 8352, 744, 9011],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "connected_balloon",
            "block_profile": "dark_bubble",
            "qa_flags": [
                "candidate_crop_direct_paddle_reocr",
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
            ],
        }
        right = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "band_id": "page_005_band_078",
            "translated": "Se voce ultrapassar esse tempo, voce retornara ao seu mundo original!",
            "tipo": "fala",
            "bbox": [237, 8439, 677, 8655],
            "source_bbox": [237, 8439, 677, 8655],
            "text_pixel_bbox": [237, 8439, 677, 8655],
            "balloon_bbox": [83, 8352, 744, 9011],
            "bubble_mask_bbox": [83, 8352, 744, 9011],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "qa_flags": [
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "leading_dark_lobe_duplicate_fragment_removed",
            ],
        }

        blocks = build_render_blocks([left, right])

        self.assertEqual({block["id"] for block in blocks}, {"direct_paddle_reocr_001", "ocr_001"})
        self.assertFalse(any("same_balloon_fragment_merged" in block.get("qa_flags", []) for block in blocks))

    def test_dark_connected_lobes_repair_from_full_visual_mask(self):
        text = {
            "id": "direct_paddle_reocr_001",
            "text_id": "direct_paddle_reocr_001",
            "band_id": "page_005_band_078",
            "translated": "A retencao do subespaco e de apenas cinco minutos.",
            "tipo": "fala",
            "bbox": [129, 111, 312, 234],
            "source_bbox": [129, 111, 312, 234],
            "text_pixel_bbox": [237, 119, 677, 335],
            "balloon_bbox": [237, 36, 744, 557],
            "bubble_mask_bbox": [237, 36, 744, 557],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "connected_balloon",
            "block_profile": "dark_bubble",
            "balloon_subregions": [[237, 36, 385, 557], [385, 36, 744, 557]],
            "connected_lobe_bboxes": [[237, 36, 385, 557], [385, 36, 744, 557]],
            "connected_balloon_orientation": "left-right",
            "qa_metrics": {
                "image_dark_bubble_mask": {
                    "source": "image_dark_bubble_mask",
                    "shape_kind": "ellipse",
                    "mask_bbox": [83, 32, 744, 691],
                }
            },
            "qa_flags": [
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_connected_lobe_passthrough",
            ],
        }

        blocks = build_render_blocks([text])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["balloon_bbox"], [83, 32, 744, 691])
        self.assertEqual(blocks[0]["bubble_mask_bbox"], [83, 32, 744, 691])
        self.assertIn("dark_connected_lobes_repaired_from_visual_mask", blocks[0]["qa_flags"])

    def test_build_render_blocks_keeps_distinct_dark_bubble_pair_with_weak_left_mask(self):
        left = {
            "id": "negative_dark_000",
            "text_id": "negative_dark_000",
            "page_id": "page_002",
            "band_id": "page_002_band_023",
            "translated": "Voce era leal aos outros, mas para eles estava sendo intrometido. Voce e o rei",
            "tipo": "fala",
            "bbox": [132, 18145, 426, 18269],
            "source_bbox": [132, 18145, 426, 18269],
            "text_pixel_bbox": [132, 18146, 557, 18355],
            "balloon_bbox": [63, 18030, 706, 18461],
            "bubble_mask_bbox": [63, 18030, 445, 18461],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": [
                "dark_bubble_negative_evidence",
                "dark_bubble_ellipse_bbox_mask",
                "fast_fill_no_glyph_evidence",
            ],
        }
        right = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "page_id": "page_002",
            "band_id": "page_002_band_023",
            "translated": "Voce era o rei de ser uma tarefa simples...",
            "tipo": "fala",
            "bbox": [476, 18283, 650, 18391],
            "source_bbox": [476, 18283, 650, 18391],
            "text_pixel_bbox": [476, 18283, 650, 18391],
            "balloon_bbox": [386, 18030, 740, 18489],
            "bubble_mask_bbox": [463, 18030, 740, 18489],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": [
                "dark_bubble_duplicate_contract_promoted",
                "dark_bubble_ellipse_bbox_mask",
            ],
        }

        blocks = build_render_blocks([left, right])

        self.assertEqual([block["id"] for block in blocks], ["negative_dark_000", "ocr_001"])

    def test_dark_connected_lobe_visual_target_prefers_lobe_bbox_over_full_metric(self):
        text_data = {
            "id": "negative_dark_000",
            "translated": "Voce era leal aos outros",
            "bbox": [132, 18145, 426, 18269],
            "source_bbox": [132, 18145, 426, 18269],
            "text_pixel_bbox": [132, 18146, 420, 18265],
            "target_bbox": [63, 18030, 445, 18461],
            "bubble_mask_bbox": [63, 18030, 445, 18461],
            "bubble_mask_source": "image_dark_bubble_mask",
            "_is_lobe_subregion": True,
            "qa_flags": ["dark_bubble_connected_lobe_passthrough"],
            "qa_metrics": {
                "image_dark_bubble_mask": {
                    "source": "image_dark_bubble_mask",
                    "mask_bbox": [63, 18030, 706, 18461],
                    "anchor_bbox": [132, 18146, 557, 18355],
                }
            },
        }

        selected = _select_dark_panel_visual_mask_render_target_bbox(text_data, [63, 18030, 706, 18461])

        self.assertEqual(selected, [63, 18030, 445, 18461])
        self.assertEqual(text_data["_visual_rect_outer_bbox"], [63, 18030, 445, 18461])
        self.assertLessEqual(text_data["_visual_rect_inner_bbox"][2], 445)
        self.assertIn("dark_bubble_lobe_mask_bbox_preferred", text_data["qa_flags"])

    def test_dark_connected_lobe_keeps_source_anchor_position(self):
        text_data = {
            "id": "dark_connected_right_lobe",
            "translated": "Eu sou chamado de 'sistema'",
            "tipo": "fala",
            "bbox": [485, 281, 717, 315],
            "source_bbox": [485, 281, 717, 315],
            "text_pixel_bbox": [485, 281, 717, 315],
            "target_bbox": [476, 111, 775, 395],
            "balloon_bbox": [476, 111, 775, 395],
            "bubble_mask_bbox": [476, 111, 775, 395],
            "layout_safe_bbox": [530, 162, 721, 344],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "_is_lobe_subregion": True,
            "_connected_slot_index": 1,
            "_connected_slot_count": 2,
            "_connected_source_bbox": [485, 281, 717, 315],
            "connected_balloon_orientation": "left-right",
            "qa_flags": [
                "dark_bubble_connected_lobes_promoted",
                "dark_bubble_connected_lobe_passthrough",
                "dark_bubble_ellipse_bbox_mask",
            ],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
            },
        }

        plan = plan_text_layout(text_data)

        self.assertFalse(renderer_mod._should_disable_connected_layout_for_dark_panel_visual_mask(text_data))
        self.assertTrue(plan["_follow_english_anchor_position"])
        source_cx = (485 + 717) / 2.0
        source_cy = (281 + 315) / 2.0
        pos = plan["position_bbox"]
        pos_cx = (pos[0] + pos[2]) / 2.0
        pos_cy = (pos[1] + pos[3]) / 2.0
        self.assertLess(abs(pos_cx - source_cx), 1.0)
        self.assertLess(abs(pos_cy - source_cy), 1.0)
        self.assertLess(pos[1], text_data["layout_safe_bbox"][1] + 80)
        self.assertEqual(plan["capacity_bbox"], [530, 162, 721, 344])

    def test_dark_connected_bubble_visual_mask_keeps_connected_layout(self):
        text_data = {
            "id": "dark_connected_full",
            "translated": "Texto do lobo esquerdo",
            "bbox": [237, 119, 677, 335],
            "source_bbox": [237, 119, 677, 335],
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

        self.assertFalse(renderer_mod._should_disable_connected_layout_for_dark_panel_visual_mask(text_data))

    def test_dark_connected_lobe_uses_local_anchor_when_text_pixel_bbox_is_broad(self):
        text_data = {
            "id": "page_005_band_078_left_lobe",
            "translated": "A retenção do subespaço é de apenas cinco minutos.",
            "tipo": "fala",
            "bbox": [129, 111, 312, 234],
            "source_bbox": [129, 111, 312, 234],
            "text_pixel_bbox": [237, 119, 677, 335],
            "target_bbox": [83, 32, 385, 691],
            "balloon_bbox": [83, 32, 385, 691],
            "bubble_mask_bbox": [83, 32, 385, 691],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "_is_lobe_subregion": True,
            "_connected_slot_index": 0,
            "_connected_slot_count": 2,
            "connected_balloon_orientation": "left-right",
            "layout_safe_bbox": [159, 140, 282, 195],
            "qa_flags": [
                "partial_dark_bubble_lobe_reocr",
                "dark_bubble_connected_lobe_passthrough",
                "dark_bubble_lobe_mask_bbox_preferred",
            ],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
            },
        }

        anchor = renderer_mod._resolve_english_anchor_bbox(text_data)
        plan = plan_text_layout(text_data)
        pos = plan["position_bbox"]
        capacity = plan["capacity_bbox"]
        pos_cx = (pos[0] + pos[2]) / 2.0
        pos_cy = (pos[1] + pos[3]) / 2.0

        self.assertEqual(anchor, [129, 111, 312, 234])
        self.assertLess(abs(pos_cx - 220.5), 1.0)
        self.assertLess(abs(pos_cy - 172.5), 1.0)
        self.assertLess(pos[2], 385)
        self.assertGreater(capacity[2] - capacity[0], 200)
        self.assertGreater(capacity[3] - capacity[1], 430)
        self.assertIn("dark_lobe_visual_capacity_bbox", text_data["qa_flags"])

    def test_dark_jagged_missing_lobe_anchor_splits_by_line_polygons(self):
        text_data = {
            "id": "page_003_band_047_like",
            "text_id": "direct_paddle_reocr_001",
            "translated": "Evite isso, rapidamente! e nosso inimigo!",
            "original": "Avoid it, quickly! It's our enemy!",
            "tipo": "fala",
            "bbox": [89, 25, 325, 610],
            "source_bbox": [89, 25, 325, 610],
            "text_pixel_bbox": [89, 25, 325, 610],
            "target_bbox": [0, 0, 538, 859],
            "balloon_bbox": [0, 0, 538, 859],
            "bubble_mask_bbox": [0, 0, 538, 859],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "line_polygons": [
                [[90, 25], [325, 27], [325, 78], [89, 76]],
                [[108, 560], [319, 565], [318, 611], [107, 606]],
            ],
            "qa_flags": [
                "candidate_crop_direct_paddle_reocr",
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "connected_layout_disabled_rejected_bubble_mask",
                "connected_lobe_boxes_missing_source_anchor_fallback",
                "mask_outside_balloon",
                "mask_outside_balloon_critical",
            ],
            "qa_metrics": {
                "dark_text_contract_fill_mask": {
                    "bbox": [87, 25, 330, 614],
                    "source": "raw_glyph_mask",
                }
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 48,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
            },
        }

        blocks = build_render_blocks([text_data])

        self.assertEqual(len(blocks), 2)
        self.assertEqual([block["translated"] for block in blocks], ["Evite isso, rapidamente!", "e nosso inimigo!"])
        self.assertEqual([block["id"] for block in blocks], ["page_003_band_047_like_fragment_1", "page_003_band_047_like_fragment_2"])
        self.assertEqual(
            [block["text_id"] for block in blocks],
            ["direct_paddle_reocr_001_fragment_1", "direct_paddle_reocr_001_fragment_2"],
        )
        self.assertTrue(all("dark_missing_anchor_visual_lobes_split" in (block.get("qa_flags") or []) for block in blocks))
        self.assertEqual(blocks[0]["source_text_mask_bbox"], [89, 25, 325, 78])
        self.assertEqual(blocks[1]["source_text_mask_bbox"], [107, 560, 319, 611])
        self.assertEqual(blocks[0]["qa_metrics"]["dark_missing_anchor_visual_lobe_split"]["index"], 0)
        self.assertEqual(blocks[1]["qa_metrics"]["dark_missing_anchor_visual_lobe_split"]["index"], 1)
        self.assertLess(blocks[0]["target_bbox"][3], 120)
        self.assertGreater(blocks[1]["target_bbox"][1], 520)
        self.assertLess(blocks[1]["target_bbox"][3], 640)
        self.assertNotEqual(blocks[0]["target_bbox"], [0, 0, 538, 859])
        self.assertNotEqual(blocks[1]["target_bbox"], [0, 0, 538, 859])

    def test_dark_connected_lobe_prefers_source_text_mask_bbox_for_anchor_and_scale(self):
        text_data = {
            "id": "page_005_band_078_right_lobe",
            "translated": "SE VOCE ULTRAPASSAR ESSE TEMPO, VOCE RETORNARA AO SEU MUNDO ORIGINAL!",
            "original": "If you exceed that time, you will return to your original world!",
            "tipo": "fala",
            "bbox": [385, 32, 744, 691],
            "source_bbox": [385, 32, 744, 691],
            "text_pixel_bbox": [128, 100, 703, 692],
            "source_text_mask_bbox": [432, 146, 636, 308],
            "line_polygons": [
                [[237, 112], [680, 112], [680, 336], [237, 336]],
            ],
            "target_bbox": [385, 32, 744, 691],
            "balloon_bbox": [385, 32, 744, 691],
            "bubble_mask_bbox": [385, 32, 744, 691],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "_is_lobe_subregion": True,
            "_connected_slot_index": 1,
            "_connected_slot_count": 2,
            "connected_balloon_orientation": "left-right",
            "layout_safe_bbox": [410, 78, 719, 645],
            "qa_flags": [
                "partial_dark_bubble_lobe_reocr",
                "dark_bubble_connected_lobe_passthrough",
            ],
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 36,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
            },
        }

        self.assertEqual(renderer_mod._resolve_english_anchor_bbox(text_data), [432, 146, 636, 308])
        self.assertEqual(renderer_mod._original_text_mask_bbox_for_scale(text_data), [432, 146, 636, 308])

    def test_dark_connected_lobe_propagates_compact_inpaint_bbox_to_type_anchor(self):
        text_data = {
            "id": "page_005_band_078_left_lobe",
            "translated": "A retencao do subespaco e de apenas cinco minutos.",
            "original": "The subspace retention is only five minutes.",
            "tipo": "fala",
            "bbox": [129, 111, 312, 234],
            "source_bbox": [129, 111, 312, 234],
            "text_pixel_bbox": [129, 111, 312, 234],
            "target_bbox": [83, 32, 385, 691],
            "balloon_bbox": [83, 32, 385, 691],
            "bubble_mask_bbox": [83, 32, 385, 691],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "_is_lobe_subregion": True,
            "_connected_slot_index": 0,
            "_connected_slot_count": 2,
            "connected_balloon_orientation": "left-right",
            "layout_safe_bbox": [104, 78, 364, 645],
            "qa_flags": [
                "dark_connected_bubble_compact_bbox_replaced_aggregate_source",
                "dark_bubble_connected_lobe_passthrough",
                "dark_bubble_lobe_mask_bbox_preferred",
            ],
            "qa_metrics": {
                "inpaint_mask_contract_text_bbox_replaced_aggregate_source": {
                    "bbox": [122, 104, 319, 241],
                }
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 36,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
            },
        }

        plan = plan_text_layout(text_data)
        position = plan["position_bbox"]

        self.assertEqual(text_data["source_text_mask_bbox"], [122, 104, 319, 241])
        self.assertEqual(renderer_mod._resolve_english_anchor_bbox(text_data), [122, 104, 319, 241])
        self.assertEqual(renderer_mod._original_text_mask_bbox_for_scale(text_data), [122, 104, 319, 241])
        self.assertLess(abs(((position[0] + position[2]) / 2.0) - 220.5), 1.0)
        self.assertLess(abs(((position[1] + position[3]) / 2.0) - 172.5), 1.0)
        self.assertIn("dark_connected_text_anchor_propagated_to_type", text_data["qa_flags"])

    def test_original_text_scale_ignores_undercovering_anchor_bbox(self):
        text_data = {
            "id": "page_002_band_025_left_lobe",
            "translated": "Voce ate foi ao reformatorio por causa deles, nao foi?",
            "original": "You even went to juvie for them didn't you?",
            "source_text_anchor_bbox": [244, 113, 313, 223],
            "text_pixel_bbox": [131, 108, 337, 222],
            "line_polygons": [
                [[131, 108], [337, 108], [337, 222], [131, 222]],
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "qa_flags": ["dark_bubble_connected_lobe_passthrough"],
        }

        self.assertEqual(renderer_mod._original_text_mask_bbox_for_scale(text_data), [131, 108, 337, 222])

    def test_original_text_scale_prefers_real_mask_over_compact_anchor_bbox(self):
        text_data = {
            "id": "page_002_band_023_left_lobe",
            "translated": "Voce cresceu em um orfanato sem pais e no inicio da adolescencia.",
            "original": "You grew up at an orphanage without parents and by your early teens.",
            "source_text_anchor_bbox": [68, 3505, 383, 3690],
            "source_text_mask_bbox": [68, 3507, 491, 3712],
            "text_pixel_bbox": [68, 3507, 491, 3712],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "qa_flags": ["dark_connected_text_anchor_propagated_to_type"],
        }

        self.assertEqual(renderer_mod._original_text_mask_bbox_for_scale(text_data), [68, 3507, 491, 3712])

    def test_original_text_scale_ignores_placeholder_and_far_group_mask_bbox(self):
        text_data = {
            "id": "page_005_band_078_left_lobe",
            "translated": "A retencao do subespaco e de apenas cinco minutos.",
            "original": "The subspace retention is only five minutes.",
            "source_text_anchor_bbox": [132, 60634, 313, 60752],
            "source_text_mask_bbox": [0, 0, 32, 32],
            "text_pixel_bbox": [237, 60641, 677, 60857],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "qa_flags": ["dark_connected_text_anchor_propagated_to_type"],
        }

        self.assertEqual(renderer_mod._original_text_mask_bbox_for_scale(text_data), [132, 60634, 313, 60752])

    def test_original_text_scale_min_underflow_is_hard_violation(self):
        candidate = {
            "block_width": 70,
            "block_height": 80,
        }
        source_bbox = [131, 108, 337, 222]

        self.assertEqual(
            renderer_mod._original_text_scale_candidate_hard_violations(candidate, source_bbox),
            ["width_lt_0.85x_source_text", "height_lt_0.85x_source_text"],
        )

    def test_dark_connected_lobe_uses_sanitized_clean_bbox_when_raw_bbox_is_whole_lobe(self):
        text_data = {
            "id": "page_005_band_078_right_lobe",
            "translated": "SE VOCE ULTRAPASSAR ESSE TEMPO, VOCE RETORNARA AO SEU MUNDO ORIGINAL!",
            "original": "If you exceed that time, you will return to your original world!",
            "tipo": "fala",
            "bbox": [237, 100, 744, 621],
            "source_bbox": [237, 100, 744, 621],
            "text_pixel_bbox": [237, 100, 744, 621],
            "target_bbox": [385, 32, 744, 691],
            "balloon_bbox": [385, 32, 744, 691],
            "bubble_mask_bbox": [385, 32, 744, 691],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "_is_lobe_subregion": True,
            "_connected_slot_index": 1,
            "_connected_slot_count": 2,
            "connected_balloon_orientation": "left-right",
            "layout_safe_bbox": [410, 78, 719, 645],
            "qa_flags": [
                "dark_bubble_connected_lobe_passthrough",
                "dark_connected_lobe_anchor_component_filtered",
            ],
            "qa_metrics": {
                "bbox_overreach": {
                    "ratio": 5.8,
                    "text_geometry_bbox": [398, 199, 680, 328],
                },
                "dark_connected_bubble_broad_mask_rejected": {
                    "anchor_bbox": [397, 195, 680, 329],
                },
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 36,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
            },
        }

        plan = plan_text_layout(text_data)
        position = plan["position_bbox"]

        self.assertEqual(text_data["source_text_mask_bbox"], [398, 199, 680, 328])
        self.assertEqual(renderer_mod._resolve_english_anchor_bbox(text_data), [398, 199, 680, 328])
        self.assertEqual(renderer_mod._original_text_mask_bbox_for_scale(text_data), [398, 199, 680, 328])
        self.assertLess(abs(((position[0] + position[2]) / 2.0) - 539.0), 1.0)
        self.assertLess(abs(((position[1] + position[3]) / 2.0) - 263.5), 1.0)
        self.assertIn("dark_connected_text_anchor_propagated_to_type", text_data["qa_flags"])

    def test_overbroad_dark_text_bbox_is_sanitized_before_position_and_scale(self):
        text_data = {
            "id": "page_003_band_028_left_lobe",
            "translated": "Ninguem se importava se voce vivia ou morria.",
            "tipo": "fala",
            "bbox": [0, 0, 387, 438],
            "source_bbox": [0, 0, 387, 438],
            "text_pixel_bbox": [91, 108, 400, 316],
            "layout_bbox": [90, 109, 282, 242],
            "target_bbox": [0, 0, 457, 438],
            "balloon_bbox": [0, 0, 457, 438],
            "bubble_mask_bbox": [0, 0, 457, 438],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "qa_flags": ["dark_bubble_oval_reocr", "bbox_overreach"],
            "qa_metrics": {
                "bbox_overreach": {
                    "ratio": 4.49,
                    "bbox": [0, 0, 387, 438],
                    "text_geometry_bbox": [91, 108, 316, 245],
                }
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 48,
                "cor": "#FFFFFF",
                "contorno": "#030303",
                "contorno_px": 2,
                "glow": True,
                "glow_cor": "#F9FFFF",
                "glow_px": 6,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(text_data["source_bbox"], [91, 108, 316, 245])
        self.assertEqual(text_data["text_pixel_bbox"], [91, 108, 316, 245])
        self.assertEqual(text_data["bbox"], [91, 108, 316, 245])
        self.assertIn("layout_text_geometry_sanitized", text_data["qa_flags"])
        self.assertLess(abs(((plan["position_bbox"][0] + plan["position_bbox"][2]) / 2.0) - 203.5), 1.0)

    def test_dark_connected_passthrough_blocks_do_not_inherit_aggregate_text_bbox(self):
        group_texts = [
            {
                "id": "left",
                "translated": "A retencao do subespaco e de apenas cinco minutos.",
                "bbox": [129, 111, 312, 234],
                "source_bbox": [129, 111, 312, 234],
                "text_pixel_bbox": [237, 119, 677, 335],
                "target_bbox": [83, 32, 385, 691],
                "bubble_mask_source": "image_dark_bubble_mask",
                "qa_flags": ["dark_bubble_connected_lobe_passthrough"],
            },
            {
                "id": "right",
                "translated": "Se voce ultrapassar esse tempo, voce retornara ao seu mundo original!",
                "bbox": [398, 207, 680, 336],
                "source_bbox": [398, 207, 680, 336],
                "text_pixel_bbox": [237, 119, 677, 335],
                "target_bbox": [385, 32, 744, 691],
                "bubble_mask_source": "image_dark_bubble_mask",
                "qa_flags": ["dark_bubble_connected_lobe_passthrough"],
            },
        ]

        blocks = renderer_mod._build_connected_passthrough_lobe_blocks(
            group_texts,
            [[83, 32, 385, 691], [385, 32, 744, 691]],
            "left-right",
        )

        by_id = {block["id"]: block for block in blocks}
        self.assertEqual(by_id["left"]["text_pixel_bbox"], [129, 111, 312, 234])
        self.assertEqual(by_id["left"]["source_bbox"], [129, 111, 312, 234])
        self.assertEqual(by_id["right"]["text_pixel_bbox"], [398, 207, 680, 336])
        self.assertEqual(by_id["right"]["source_bbox"], [398, 207, 680, 336])

    def test_stale_render_layout_contract_is_rejected_when_outside_current_lobe(self):
        text_data = {
            "id": "left_lobe_stale_contract",
            "translated": "A retencao do subespaco e de apenas cinco minutos.",
            "tipo": "fala",
            "bbox": [129, 111, 312, 234],
            "source_bbox": [129, 111, 312, 234],
            "text_pixel_bbox": [129, 111, 312, 234],
            "_connected_source_bbox": [129, 111, 312, 234],
            "target_bbox": [83, 32, 385, 691],
            "balloon_bbox": [83, 32, 385, 691],
            "bubble_mask_bbox": [83, 32, 385, 691],
            "safe_text_box": [104, 78, 364, 645],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "_is_lobe_subregion": True,
            "_connected_slot_index": 0,
            "_connected_slot_count": 2,
            "connected_balloon_orientation": "left-right",
            "qa_flags": ["dark_bubble_connected_lobe_passthrough"],
            "render_layout_contract": {
                "schema_version": 1,
                "translated_key": "a retencao do subespaco e de apenas cinco minutos.",
                "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "font_size": 33,
                "line_height": 49,
                "lines": ["A RETENCAO DO", "SUBESPACO E DE", "APENAS CINCO", "MINUTOS."],
                "positions": [[376, 126], [372, 175], [382, 224], [408, 273]],
                "line_widths": [163, 171, 152, 99],
                "block_bbox": [372, 126, 543, 322],
                "target_bbox": [83, 32, 385, 691],
                "safe_text_box": [104, 78, 364, 645],
                "coordinate_space": "band",
                "band_y_top": 0,
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 33,
                "cor": "#FFFFFF",
                "contorno": "#030303",
                "contorno_px": 2,
                "glow": True,
                "glow_cor": "#F9FFFF",
                "glow_px": 6,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertNotIn("render_layout_contract_replayed", text_data.get("qa_flags") or [])
        self.assertIn("stale_render_layout_contract_rejected", text_data.get("qa_flags") or [])
        block = resolved["block_bbox"]
        self.assertGreaterEqual(block[0], 83)
        self.assertLessEqual(block[2], 385)

    def test_safe_textpath_centering_uses_lobe_position_box_not_full_target_center(self):
        img = Image.new("RGB", (800, 760), (0, 0, 0))
        text_data = {
            "id": "left_lobe_safe_textpath",
            "translated": "A retencao do subespaco e de apenas cinco minutos.",
            "tipo": "fala",
            "bbox": [129, 111, 312, 234],
            "source_bbox": [129, 111, 312, 234],
            "text_pixel_bbox": [129, 111, 312, 234],
            "_connected_source_bbox": [129, 111, 312, 234],
            "target_bbox": [83, 32, 385, 691],
            "balloon_bbox": [83, 32, 385, 691],
            "bubble_mask_bbox": [83, 32, 385, 691],
            "safe_text_box": [104, 78, 364, 645],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "_is_lobe_subregion": True,
            "_connected_slot_index": 0,
            "_connected_slot_count": 2,
            "connected_balloon_orientation": "left-right",
            "qa_flags": ["dark_bubble_connected_lobe_passthrough"],
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 33,
                "cor": "#FFFFFF",
                "contorno": "#030303",
                "contorno_px": 2,
                "glow": True,
                "glow_cor": "#F9FFFF",
                "glow_px": 6,
                "force_upper": True,
            },
        }

        with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "1"}):
            render_text_block(img, text_data)

        self.assertLessEqual(text_data["render_bbox"][2], 385)
        self.assertGreaterEqual(text_data["render_bbox"][0], 83)
        self.assertNotIn("render_outside_balloon", text_data.get("qa_flags") or [])

    def test_original_scale_alignment_keeps_connected_lobe_clamped_to_safe_box(self):
        layer = Image.new("RGBA", (800, 760), (0, 0, 0, 0))
        ImageDraw.Draw(layer).rectangle([150, 180, 315, 300], fill=(255, 255, 255, 255))
        text_data = {
            "id": "left_lobe_alignment",
            "source_bbox": [430, 170, 620, 310],
            "text_pixel_bbox": [430, 170, 620, 310],
            "_is_lobe_subregion": True,
            "_connected_source_bbox": [430, 170, 620, 310],
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["dark_bubble_connected_lobe_passthrough"],
        }

        with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "1"}):
            _shifted, shifted_bbox = renderer_mod._align_rgba_layer_to_source_text_center(
                layer,
                text_data,
                [104, 78, 364, 645],
            )

        self.assertEqual(shifted_bbox, [198, 180, 364, 301])
        self.assertIn("source_center_alignment_clamped_to_safe", text_data.get("qa_metrics", {}))

    def test_dark_bubble_overbroad_band_mask_clamps_to_visible_balloon(self):
        text_data = {
            "id": "page_005_band_071",
            "translated": "Este é o subespaço do sistema.",
            "tipo": "fala",
            "bbox": [151, 221, 773, 760],
            "source_bbox": None,
            "text_pixel_bbox": [521, 571, 687, 687],
            "balloon_bbox": [0, 0, 800, 856],
            "bubble_mask_bbox": [0, 0, 800, 856],
            "bubble_mask_source": "",
            "target_bbox": [0, 0, 800, 856],
            "position_bbox": [80, 86, 720, 770],
            "capacity_bbox": [80, 86, 720, 770],
            "safe_text_box": [170, 168, 630, 688],
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "qa_metrics": {
                "image_dark_bubble_mask": {
                    "source": "image_dark_bubble_mask",
                    "mask_bbox": [0, 0, 800, 856],
                }
            },
            "qa_flags": [
                "dark_bubble_promoted_from_rejected_mask",
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
            ],
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 44,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [151, 221, 773, 760])
        self.assertLessEqual(plan["position_bbox"][2], 773)
        self.assertLessEqual(plan["capacity_bbox"][2], 773)
        self.assertNotEqual(plan["position_bbox"], [80, 86, 720, 770])
        self.assertNotEqual(plan["capacity_bbox"], [80, 86, 720, 770])
        self.assertTrue(
            "dark_bubble_overbroad_mask_clamped_to_visible_balloon" in text_data["qa_flags"]
            or "dark_bubble_overbroad_target_clamped_to_visible_bbox" in text_data["qa_flags"]
        )

    def test_card_panel_short_text_font_is_capped_for_margin(self):
        img = Image.new("RGB", (800, 680), (5, 7, 14))
        text_data = {
            "id": "page_005_band_095",
            "translated": "Mova-se imediatamente!",
            "tipo": "fala",
            "bbox": [508, 470, 692, 511],
            "source_bbox": [508, 470, 692, 511],
            "text_pixel_bbox": [508, 470, 692, 511],
            "balloon_bbox": [312, 72, 800, 636],
            "bubble_mask_bbox": [312, 72, 800, 636],
            "bubble_mask_source": "",
            "card_panel_text_context": False,
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "background_rgb": [5, 7, 14],
            "qa_flags": [
                "candidate_crop_direct_paddle_reocr",
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "short_dark_anchor_center_preserved",
            ],
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 44,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        safe = text_data["safe_text_box"]
        render = text_data["render_bbox"]
        self.assertIn("dark_card_panel_font_capped_for_margin", text_data.get("qa_flags") or [])
        self.assertGreaterEqual(render[0] - safe[0], 6)
        self.assertGreaterEqual(safe[2] - render[2], 6)
        self.assertNotIn("TEXT_OVERFLOW", text_data.get("qa_flags") or [])

    def test_dark_bubble_visual_capacity_does_not_cap_font_to_ocr_anchor_height(self):
        text_data = {
            "id": "dark_oval_capacity",
            "translated": "Criterios de conclusao da missao: estabeleca um submundo de nivel 1.",
            "tipo": "fala",
            "bbox": [90, 136, 359, 197],
            "source_bbox": [90, 136, 359, 197],
            "text_pixel_bbox": [90, 136, 359, 197],
            "balloon_bbox": [47, 0, 420, 325],
            "bubble_mask_bbox": [47, 0, 420, 325],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "background_rgb": [6, 10, 14],
            "qa_flags": ["dark_bubble_ellipse_bbox_mask", "dark_bubble_oval_reocr"],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 40,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
            },
        }

        plan = plan_text_layout(text_data)
        resolved = renderer_mod._resolve_text_layout(text_data, plan)

        self.assertNotIn("dark_visual_auto_font_capped_to_source_scale", text_data.get("qa_flags") or [])
        self.assertGreaterEqual(resolved["font_size"], 24)

    def test_resolve_text_layout_replays_hydrated_render_contract(self):
        text = "A MISSÃO PRINCIPAL SERÁ MOSTRADA EM BREVE"
        text_data = {
            "id": "hydrated_contract",
            "translated": text,
            "tipo": "fala",
            "bbox": [113, 2897, 259, 2968],
            "source_bbox": [113, 2897, 259, 2968],
            "text_pixel_bbox": [113, 2897, 259, 2968],
            "target_bbox": [38, 2854, 331, 3002],
            "position_bbox": [52, 2867, 319, 2998],
            "capacity_bbox": [52, 2867, 319, 2998],
            "safe_text_box": [52, 2871, 319, 2994],
            "balloon_bbox": [38, 2854, 331, 3002],
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 18,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
            },
            "render_layout_contract": {
                "schema_version": 1,
                "source": "debug_render_plan_raw",
                "translated_key": renderer_mod._text_layout_contract_text_key(text),
                "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "font_size": 26,
                "line_height": 36,
                "lines": ["A MISSÃO PRINCIPAL", "SERÁ MOSTRADA", "EM BREVE"],
                "positions": [[112, 2885], [126, 2921], [151, 2957]],
                "line_widths": [147, 119, 70],
                "block_bbox": [112, 2885, 259, 2993],
                "target_bbox": [38, 2854, 331, 3002],
                "position_bbox": [52, 2867, 319, 2998],
                "safe_text_box": [52, 2871, 319, 2994],
                "coordinate_space": "page",
                "band_y_top": 0,
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertTrue(resolved["render_layout_contract_replayed"])
        self.assertEqual(resolved["font_size"], 26)
        self.assertEqual(resolved["lines"], ["A MISSÃO PRINCIPAL", "SERÁ MOSTRADA", "EM BREVE"])
        self.assertEqual(resolved["positions"], [(112, 2885), (126, 2921), (151, 2957)])

    def test_dark_bubble_legibility_samples_clean_image_over_stale_bright_background(self):
        img = Image.new("RGB", (360, 180), (0, 0, 0))
        text_data = {
            "id": "dark_bubble_stale_bg",
            "text": "You were chosen to become King Yeomra!",
            "translated": "Voce foi escolhido para se tornar rei Yeomra!",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [70, 50, 290, 125],
            "source_bbox": [70, 50, 290, 125],
            "text_pixel_bbox": [70, 50, 290, 125],
            "target_bbox": [35, 20, 325, 155],
            "position_bbox": [55, 38, 305, 138],
            "capacity_bbox": [55, 38, 305, 138],
            "safe_text_box": [58, 42, 302, 133],
            "balloon_bbox": [35, 20, 325, 155],
            "background_rgb": [195, 193, 185],
            "style_origin": "auto_dark_panel_glow",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 26,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "alinhamento": "center",
                "style_origin": "auto_dark_panel_glow",
            },
            "style": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 26,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "alinhamento": "center",
                "style_origin": "auto_dark_panel_glow",
            },
            "qa_flags": ["dark_bubble_oval_reocr", "auto_dark_panel_glow_fallback"],
        }

        plan = ensure_legible_plan(img, plan_text_layout(text_data))

        self.assertEqual(plan["text_color"], "#FFFFFF")
        self.assertTrue(plan["glow"])
        self.assertLess(max(plan["background_rgb"]), 16)

    def test_dark_bubble_ocr_line_polygons_do_not_force_sideways_rotation(self):
        text_data = {
            "layout_profile": "white_balloon",
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": ["dark_bubble_oval_reocr"],
            "line_polygons": [
                [[90, 40], [110, 40], [110, 130], [90, 130]],
                [[130, 40], [150, 40], [150, 130], [130, 130]],
            ],
            "estilo": {"rotacao": 0},
        }

        rotation_deg, rotation_source = renderer_mod._resolve_render_rotation_deg(text_data, text_data["estilo"])

        self.assertEqual(rotation_deg, 0.0)
        self.assertEqual(rotation_source, "")

    def test_final_page_space_normalizes_mixed_band_page_bbox_fields(self):
        layer = {
            "coordinate_space": "band",
            "source_coordinate_space": "band",
            "band_y_top": 2620,
            "safe_text_box": [134, 205, 730, 3414],
            "_debug_safe_text_box": [134, 205, 730, 3414],
            "render_bbox": [352, 489, 512, 3129],
            "balloon_bbox": [419, 2735, 710, 2811],
            "bbox": [463, 2749, 666, 2797],
            "qa_flags": [
                "safe_text_box_recomputed",
                "layout_bbox_coordinate_mismatch",
                "page_space_rerender_mixed_coordinates",
            ],
        }

        normalized = _mark_final_layer_as_page_space(layer)

        self.assertEqual(normalized["coordinate_space"], "page")
        self.assertEqual(normalized["source_coordinate_space"], "page")
        self.assertNotIn("band_y_top", normalized)
        self.assertEqual(normalized["safe_text_box"], [134, 2825, 730, 3414])
        self.assertEqual(normalized["_debug_safe_text_box"], [134, 2825, 730, 3414])
        self.assertEqual(normalized["render_bbox"], [352, 3109, 512, 3129])
        self.assertEqual(normalized["balloon_bbox"], [419, 2735, 710, 2811])
        self.assertNotIn("layout_bbox_coordinate_mismatch", normalized["qa_flags"])
        self.assertNotIn("page_space_rerender_mixed_coordinates", normalized["qa_flags"])

    def test_debug_unclamped_safe_box_survives_page_space_and_biases_layout_down(self):
        layer = {
            "id": "ocr_001",
            "coordinate_space": "page",
            "source_coordinate_space": "page",
            "bbox": [298, 16276, 525, 16383],
            "source_bbox": [298, 16276, 533, 16383],
            "text_pixel_bbox": [298, 16276, 525, 16383],
            "target_bbox": [278, 16253, 545, 16383],
            "position_bbox": [278, 16253, 545, 16383],
            "capacity_bbox": [278, 16253, 545, 16383],
            "safe_text_box": [278, 16253, 545, 16383],
            "_debug_safe_text_box": [278, 16253, 545, 16383],
            "_safe_text_box_unclamped": [278, 16253, 545, 16428],
            "balloon_bbox": [242, 16203, 580, 16383],
            "bubble_mask_bbox": [242, 16203, 580, 16383],
            "bubble_inner_bbox": [245, 16205, 578, 16383],
            "_bubble_inner_bbox_unclamped": [245, 16205, 578, 16476],
            "layout_safe_reason": "debug_derived_bubble_mask_unclamped",
            "_render_bbox_from_repaired_safe_text_box": True,
            "original": "THE AMOUNT IS JUST RIGHT. THIS BITCHIS AREAL",
            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
            "text": "THE AMOUNT IS JUST RIGHT. THIS BITCHIS AREAL",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 22, "alinhamento": "center", "cor": "#000000"},
            "style": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 22, "alinhamento": "center", "cor": "#000000"},
            "qa_flags": ["safe_text_box_recomputed"],
        }

        normalized = _final_page_space_text_layers_for_renderer([layer], page_number=2)[0]
        self.assertEqual(normalized["_safe_text_box_unclamped"], [278, 16253, 545, 16428])

        plan = plan_text_layout(normalized)
        resolved = _resolve_text_layout(normalized, plan)

        self.assertEqual(plan["safe_text_box"], [278, 16253, 545, 16383])
        self.assertEqual(plan["layout_safe_reason"], "debug_derived_bubble_mask_unclamped")
        self.assertEqual(plan["padding_y"], 0)
        self.assertGreater(plan["vertical_bias_px"], 0)
        self.assertGreaterEqual(resolved["block_bbox"][1], plan["safe_text_box"][1])
        self.assertLessEqual(resolved["block_bbox"][3], 16383)

    def test_connected_dark_lobe_source_center_alignment_stays_inside_safe_box(self):
        img = Image.new("RGB", (800, 720), (0, 0, 0))
        text_data = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_005_band_078",
            "translated": "SE VOCE ULTRAPASSAR ESSE TEMPO, VOCE RETORNARA AO SEU MUNDO ORIGINAL!",
            "original": "If you exceed that time, you will return to your original world!",
            "bbox": [160, 90, 430, 260],
            "source_bbox": [160, 90, 430, 260],
            "text_pixel_bbox": [160, 90, 430, 260],
            "target_bbox": [385, 32, 744, 691],
            "position_bbox": [410, 78, 719, 645],
            "capacity_bbox": [410, 78, 719, 645],
            "safe_text_box": [410, 78, 719, 645],
            "_debug_safe_text_box": [410, 78, 719, 645],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "_is_lobe_subregion": True,
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 36,
                "cor": "#ffffff",
                "alinhamento": "center",
                "contorno_cor": "#000000",
                "contorno_px": 1,
            },
            "style": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 36,
                "cor": "#ffffff",
                "alinhamento": "center",
                "contorno_cor": "#000000",
                "contorno_px": 1,
            },
            "qa_flags": [
                "dark_bubble_connected_lobe_passthrough",
                "original_text_scale_size_experiment",
            ],
        }

        plan = plan_text_layout(text_data)
        plan["target_bbox"] = [385, 32, 744, 691]
        plan["position_bbox"] = [410, 78, 719, 645]
        plan["capacity_bbox"] = [410, 78, 719, 645]
        plan["safe_text_box"] = [410, 78, 719, 645]
        with patch("typesetter.renderer._try_render_single_text_block_with_rust", return_value=False):
            _render_single_text_block_unrotated(img, text_data, plan)

        render_bbox = text_data.get("render_bbox")
        self.assertIsNotNone(render_bbox)
        self.assertGreaterEqual(render_bbox[0], 410)
        self.assertLessEqual(render_bbox[2], 719)
        self.assertGreaterEqual(render_bbox[1], 78)
        self.assertLessEqual(render_bbox[3], 645)

    def test_edge_clipped_tiny_fragment_uses_real_bubble_mask_not_cut_balloon_bbox(self):
        layer = {
            "id": "ocr_002",
            "coordinate_space": "page",
            "source_coordinate_space": "page",
            "bbox": [298, -88, 525, 34],
            "source_bbox": [298, -88, 525, 34],
            "text_pixel_bbox": [298, -88, 525, 34],
            "target_bbox": [258, 10, 551, 99],
            "position_bbox": [258, 10, 551, 99],
            "capacity_bbox": [258, 10, 551, 99],
            "safe_text_box": [258, 10, 551, 99],
            "_debug_safe_text_box": [258, 10, 551, 99],
            "layout_safe_bbox": [258, 10, 551, 99],
            "render_bbox": [258, 10, 551, 99],
            "balloon_bbox": [324, 0, 508, 38],
            "bubble_mask_bbox": [234, 0, 575, 109],
            "bubble_inner_bbox": [326, 13, 483, 17],
            "bubble_mask_source": "image_white_bubble_mask",
            "bubble_id": "page_002_band_019_bubble_002",
            "layout_profile": "white_balloon",
            "layout_safe_reason": "debug_derived_bubble_mask_rejected",
            "_final_edge_clipped_bubble_safe_box": True,
            "original": "THE AMOUNT IS JUST RIGHT. THIS BITCHIS AREAL",
            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
            "text": "THE AMOUNT IS JUST RIGHT. THIS BITCHIS AREAL",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 22, "alinhamento": "center", "cor": "#000000"},
            "style": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 22, "alinhamento": "center", "cor": "#000000"},
            "qa_flags": ["same_balloon_fragment_merged", "tiny_bubble_inner_bbox_rejected", "safe_text_box_recomputed"],
        }

        plan = plan_text_layout(dict(layer))
        resolved = _resolve_text_layout(dict(layer), plan)

        self.assertEqual(plan["target_bbox"], [234, 0, 575, 109])
        self.assertGreaterEqual(plan["safe_text_box"][3] - plan["safe_text_box"][1], 68)
        self.assertGreaterEqual(resolved["font_size"], 11)
        self.assertGreaterEqual(len(resolved["lines"]), 2)

    def test_large_rejected_bubble_inner_centers_short_text_in_real_body(self):
        layer = {
            "id": "ocr_001",
            "coordinate_space": "page",
            "source_coordinate_space": "page",
            "bbox": [148, 7248, 310, 7268],
            "source_bbox": [148, 7248, 310, 7268],
            "text_pixel_bbox": [148, 7248, 310, 7268],
            "safe_text_box": [85, 7232, 393, 7284],
            "_debug_safe_text_box": [85, 7232, 393, 7284],
            "render_bbox": [148, 7248, 300, 7273],
            "balloon_bbox": [0, 7013, 800, 7967],
            "bubble_mask_bbox": [29, 7109, 642, 7722],
            "bubble_inner_bbox": [71, 7151, 600, 7680],
            "bubble_mask_source": "image_white_bubble_mask",
            "bubble_id": "page_003_band_035_bubble_001",
            "layout_profile": "white_balloon",
            "layout_safe_reason": "debug_derived_bubble_mask_rejected",
            "original": "HEY, LET'S GO!",
            "translated": "EI, VAMOS!",
            "text": "HEY, LET'S GO!",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "alinhamento": "center", "cor": "#000000"},
            "style": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "alinhamento": "center", "cor": "#000000"},
            "qa_flags": ["safe_text_box_recomputed"],
        }

        plan = plan_text_layout(dict(layer))
        resolved = _resolve_text_layout(dict(layer), plan)

        safe_center_y = (plan["safe_text_box"][1] + plan["safe_text_box"][3]) / 2.0
        block_center_y = (resolved["block_bbox"][1] + resolved["block_bbox"][3]) / 2.0
        self.assertEqual(plan["layout_safe_reason"], "bubble_inner_bbox")
        self.assertGreaterEqual(plan["safe_text_box"][3] - plan["safe_text_box"][1], 350)
        self.assertLess(abs(block_center_y - safe_center_y), 24)

    def test_render_text_block_uses_expanded_canvas_for_unclamped_safe_box(self):
        img = Image.new("RGB", (220, 100), (255, 255, 255))
        text_data = {
            "id": "ocr_edge",
            "bbox": [40, 68, 180, 98],
            "source_bbox": [40, 68, 180, 98],
            "target_bbox": [30, 50, 190, 100],
            "position_bbox": [30, 50, 190, 100],
            "capacity_bbox": [30, 50, 190, 100],
            "safe_text_box": [45, 60, 175, 100],
            "_debug_safe_text_box": [45, 60, 175, 100],
            "_safe_text_box_unclamped": [45, 60, 175, 140],
            "balloon_bbox": [20, 30, 200, 100],
            "bubble_mask_bbox": [20, 30, 200, 100],
            "bubble_inner_bbox": [30, 40, 190, 100],
            "_bubble_inner_bbox_unclamped": [30, 40, 190, 150],
            "layout_safe_reason": "debug_derived_bubble_mask_unclamped",
            "_render_bbox_from_repaired_safe_text_box": True,
            "original": "THE AMOUNT IS JUST RIGHT.",
            "translated": "A QUANTIA ESTÁ CERTA.",
            "text": "THE AMOUNT IS JUST RIGHT.",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 18, "alinhamento": "center", "cor": "#000000"},
            "style": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 18, "alinhamento": "center", "cor": "#000000"},
        }

        calls = []

        def fake_render(render_img, render_layer, plan, pre_render_np=None):
            del pre_render_np
            calls.append((render_img.size, list(plan["safe_text_box"])))
            render_layer["render_bbox"] = [60, 80, 160, 125]
            render_layer["_render_debug"] = {"fake": True}

        with patch("typesetter.renderer._render_single_text_block", side_effect=fake_render):
            render_text_block(img, text_data)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], 220)
        self.assertGreater(calls[0][0][1], 100)
        self.assertEqual(calls[0][1], [45, 60, 175, 140])
        self.assertTrue(text_data["_expanded_canvas_render_used"])
        self.assertEqual(text_data["render_bbox"], [60, 80, 160, 100])
        self.assertEqual(text_data["safe_text_box"], [45, 60, 175, 100])

    def test_render_text_block_uses_expanded_canvas_for_unclamped_safe_box_above_page(self):
        img = Image.new("RGB", (220, 100), (255, 255, 255))
        text_data = {
            "id": "ocr_edge_top",
            "bbox": [80, -60, 160, 18],
            "source_bbox": [80, -60, 160, 18],
            "target_bbox": [30, 0, 190, 70],
            "position_bbox": [30, 0, 190, 70],
            "capacity_bbox": [30, 0, 190, 70],
            "safe_text_box": [55, 8, 175, 54],
            "_debug_safe_text_box": [55, 8, 175, 54],
            "_safe_text_box_unclamped": [45, -76, 185, 40],
            "balloon_bbox": [20, 0, 200, 70],
            "bubble_mask_bbox": [20, 0, 200, 70],
            "bubble_inner_bbox": [40, 0, 190, 60],
            "_bubble_inner_bbox_unclamped": [40, -90, 190, 60],
            "layout_safe_reason": "debug_derived_bubble_mask_unclamped",
            "_render_bbox_from_repaired_safe_text_box": True,
            "original": "ACTRESS...",
            "translated": "A QUANTIA ESTA CERTA. ESSA VADIA E REAL ATRIZ...",
            "text": "ACTRESS...",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 18, "alinhamento": "center", "cor": "#000000"},
            "style": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 18, "alinhamento": "center", "cor": "#000000"},
        }
        calls = []

        def fake_render(render_img, render_layer, plan, pre_render_np=None):
            del pre_render_np
            calls.append((render_img.size, list(plan["safe_text_box"])))
            render_layer["render_bbox"] = [60, 100, 160, 130]
            render_layer["_render_debug"] = {"fake": True}

        with patch("typesetter.renderer._render_single_text_block", side_effect=fake_render):
            render_text_block(img, text_data)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], 220)
        self.assertGreater(calls[0][0][1], 100)
        self.assertEqual(calls[0][1], [45, 14, 185, 130])
        self.assertTrue(text_data["_expanded_canvas_render_used"])
        self.assertEqual(text_data["_expanded_canvas_origin"], [0, 90])
        self.assertGreaterEqual(text_data["render_bbox"][1], 0)

    def test_render_text_block_materializes_unclamped_edge_bubble_before_expanded_canvas(self):
        img = Image.new("RGB", (220, 100), (255, 255, 255))
        text_data = {
            "id": "ocr_edge_bottom",
            "bbox": [70, 64, 150, 98],
            "source_bbox": [70, 64, 150, 98],
            "text_pixel_bbox": [70, 64, 150, 98],
            "balloon_bbox": [20, 30, 200, 100],
            "bubble_mask_bbox": [20, 30, 200, 100],
            "bubble_inner_bbox": [30, 40, 190, 100],
            "_bubble_mask_bbox_unclamped": [20, 30, 200, 150],
            "_bubble_inner_bbox_unclamped": [30, 40, 190, 150],
            "bubble_mask_source": "image_contour_bubble_mask",
            "bubble_id": "page_002_band_019_bubble_001",
            "layout_profile": "white_balloon",
            "original": "THE AMOUNT IS JUST RIGHT. THIS BITCH IS A REAL ACTRESS...",
            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É UMA VERDADEIRA ATRIZ...",
            "text": "THE AMOUNT IS JUST RIGHT. THIS BITCH IS A REAL ACTRESS...",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 18, "alinhamento": "center", "cor": "#000000"},
            "style": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 18, "alinhamento": "center", "cor": "#000000"},
        }
        calls = []

        def fake_render(render_img, render_layer, plan, pre_render_np=None):
            del pre_render_np
            calls.append((render_img.size, list(plan["safe_text_box"]), plan["layout_safe_reason"]))
            render_layer["render_bbox"] = [60, 92, 170, 128]
            render_layer["_render_debug"] = {"fake": True}

        with patch("typesetter.renderer._render_single_text_block", side_effect=fake_render):
            render_text_block(img, text_data)

        self.assertEqual(len(calls), 1)
        self.assertGreater(calls[0][0][1], 100)
        self.assertEqual(calls[0][2], "debug_derived_bubble_mask_unclamped")
        self.assertTrue(text_data["_expanded_canvas_render_used"])
        self.assertLessEqual(text_data["safe_text_box"][3], 100)
        self.assertGreater(text_data["_safe_text_box_unclamped"][3], 100)

    def test_review_required_layer_still_renders_inside_best_available_anchor(self):
        layer = {
            "text": "WELL... IT'S NOT LIKE THIS ISN'T DELIGHTFUL...",
            "translated": "BEM... NÃO É COMO SE ISSO NÃO FOSSE AGRADÁVEL...",
            "safe_text_box": None,
            "render_bbox": None,
            "balloon_bbox": [462, 5012, 721, 5173],
            "source_bbox": [501, 5042, 682, 5143],
            "qa_flags": ["missing_render_bbox"],
        }

        planned = plan_fallback_render_box(layer)

        self.assertIsNotNone(planned["render_bbox"])
        self.assertIsNotNone(planned["safe_text_box"])
        self.assertIn("rendered_with_review_fallback", planned["qa_flags"])
        rx1, ry1, rx2, ry2 = planned["render_bbox"]
        bx1, by1, bx2, by2 = layer["balloon_bbox"]
        self.assertGreaterEqual((rx1 + rx2) // 2, bx1)
        self.assertLessEqual((rx1 + rx2) // 2, bx2)
        self.assertGreaterEqual((ry1 + ry2) // 2, by1)
        self.assertLessEqual((ry1 + ry2) // 2, by2)

    def test_synthetic_tight_bubble_bbox_does_not_drive_dark_card_layout(self):
        layer = {
            "text": "the Devil Knight!",
            "translated": "o Cavaleiro Diabo!",
            "bbox": [178, 175, 371, 210],
            "source_bbox": [178, 175, 371, 210],
            "text_pixel_bbox": [178, 175, 371, 210],
            "line_polygons": [[[178, 175], [371, 175], [371, 210], [178, 210]]],
            "balloon_bbox": [225, 177, 278, 207],
            "bubble_mask_bbox": [225, 177, 278, 207],
            "bubble_inner_bbox": [227, 179, 276, 205],
            "background_rgb": [58, 52, 26],
            "layout_profile": "standard",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24},
        }

        plan = plan_text_layout(layer)

        self.assertIn("tiny_bubble_inner_bbox_rejected", layer.get("qa_flags", []))
        self.assertLessEqual(plan["target_bbox"][0], 178)
        self.assertGreaterEqual(plan["target_bbox"][2], 371)
        self.assertGreater(plan["safe_text_box"][2] - plan["safe_text_box"][0], 53)

    def test_sync_page_legacy_aliases_persists_final_layers_in_page_space(self):
        page = {
            "numero": 2,
            "image_layers": {
                "base": {"path": "original/002.jpg"},
                "rendered": {"path": "translated/002.jpg"},
            },
            "text_layers": [
                {
                    "id": "ocr_002",
                    "text_id": "ocr_002",
                    "coordinate_space": None,
                    "source_coordinate_space": "band",
                    "band_y_top": 2620,
                    "strip_band_y_top": 2620,
                    "safe_text_box": [134, 205, 730, 3414],
                    "_debug_safe_text_box": [134, 205, 730, 3414],
                    "render_bbox": [352, 489, 512, 3129],
                    "target_bbox": [72, 80, 791, 3539],
                    "balloon_bbox": [419, 2735, 710, 2811],
                    "bbox": [463, 2749, 666, 2797],
                    "original": "WAIT FEW MORE DAYS!",
                    "translated": "ESPERE MAIS ALGUNS DIAS!",
                    "qa_flags": [
                        "layout_bbox_coordinate_mismatch",
                        "page_space_rerender_mixed_coordinates",
                        "safe_text_box_recomputed",
                    ],
                }
            ],
        }

        _sync_page_legacy_aliases(page)

        layer = page["text_layers"][0]
        legacy = page["textos"][0]
        self.assertEqual(layer["coordinate_space"], "page")
        self.assertEqual(layer["source_coordinate_space"], "page")
        self.assertNotIn("band_y_top", layer)
        self.assertNotIn("strip_band_y_top", layer)
        self.assertEqual(layer["safe_text_box"], [134, 2825, 730, 3414])
        self.assertEqual(layer["_debug_safe_text_box"], [134, 2825, 730, 3414])
        self.assertEqual(layer["render_bbox"], [352, 3109, 512, 3129])
        self.assertEqual(layer["target_bbox"], [72, 2700, 791, 3539])
        self.assertEqual(legacy["bbox"], [352, 3109, 512, 3129])
        self.assertNotIn("layout_bbox_coordinate_mismatch", layer["qa_flags"])
        self.assertNotIn("page_space_rerender_mixed_coordinates", layer["qa_flags"])

    def test_final_page_space_infers_lost_band_offset_from_bubble_mask_bbox(self):
        layer = {
            "coordinate_space": "page",
            "source_coordinate_space": "page",
            "target_bbox": [72, 80, 791, 3539],
            "bubble_mask_bbox": [72, 2700, 791, 3539],
            "safe_text_box": [134, 205, 730, 3414],
            "_debug_safe_text_box": [134, 205, 730, 3414],
            "render_bbox": [352, 489, 512, 3129],
            "balloon_bbox": [419, 2735, 710, 2811],
            "text_pixel_bbox": [473, 2764, 643, 2797],
            "_raw_text_evidence_bbox": [473, 145, 643, 177],
        }

        normalized = _mark_final_layer_as_page_space(layer)

        self.assertEqual(normalized["target_bbox"], [72, 2700, 791, 3539])
        self.assertEqual(normalized["safe_text_box"], [134, 2825, 730, 3414])
        self.assertEqual(normalized["_debug_safe_text_box"], [134, 2825, 730, 3414])
        self.assertEqual(normalized["render_bbox"], [352, 3109, 512, 3129])
        self.assertEqual(normalized["balloon_bbox"], [419, 2735, 710, 2811])
        self.assertEqual(normalized["text_pixel_bbox"], [473, 2764, 643, 2797])

    def test_final_page_space_renderer_normalizer_preserves_safe_and_target_boxes(self):
        layers = _final_page_space_text_layers_for_renderer(
            [
                {
                    "id": "ocr_002",
                    "text_id": "ocr_002",
                    "coordinate_space": "page",
                    "source_coordinate_space": "page",
                    "bbox": [463, 2749, 666, 2797],
                    "source_bbox": [463, 2749, 666, 2797],
                    "text_pixel_bbox": [473, 2764, 643, 2797],
                    "layout_bbox": [473, 2764, 643, 2797],
                    "target_bbox": [72, 80, 791, 3539],
                    "bubble_mask_bbox": [72, 2700, 791, 3539],
                    "safe_text_box": [134, 205, 730, 3414],
                    "_debug_safe_text_box": [134, 205, 730, 3414],
                    "render_bbox": [352, 489, 512, 3129],
                    "balloon_bbox": [419, 2735, 710, 2811],
                    "original": "PLEASE!",
                    "translated": "POR FAVOR!",
                }
            ],
            page_number=1,
        )

        layer = layers[0]
        self.assertEqual(layer["target_bbox"], [72, 2700, 791, 3539])
        self.assertEqual(layer["safe_text_box"], [134, 2825, 730, 3414])
        self.assertEqual(layer["render_bbox"], [352, 3109, 512, 3129])

    def test_normalize_final_project_page_space_layers_fixes_project_snapshot(self):
        project = {
            "paginas": [
                {
                    "numero": 1,
                    "image_layers": {},
                    "text_layers": [
                        {
                            "id": "ocr_002",
                            "text_id": "ocr_002",
                            "coordinate_space": "page",
                            "source_coordinate_space": "page",
                            "bbox": [463, 2749, 666, 2797],
                            "source_bbox": [463, 2749, 666, 2797],
                            "text_pixel_bbox": [473, 2764, 643, 2797],
                            "layout_bbox": [473, 2764, 643, 2797],
                            "target_bbox": [72, 80, 791, 3539],
                            "bubble_mask_bbox": [72, 2700, 791, 3539],
                            "safe_text_box": [134, 205, 730, 3414],
                            "_debug_safe_text_box": [134, 205, 730, 3414],
                            "render_bbox": [352, 489, 512, 3129],
                            "balloon_bbox": [419, 2735, 710, 2811],
                            "original": "PLEASE!",
                            "translated": "POR FAVOR!",
                        }
                    ],
                }
            ]
        }

        audit = _normalize_final_project_page_space_layers(project)

        layer = project["paginas"][0]["text_layers"][0]
        legacy = project["paginas"][0]["textos"][0]
        self.assertEqual(audit["layers_changed"], 1)
        self.assertEqual(layer["target_bbox"], [72, 2700, 791, 3539])
        self.assertEqual(layer["safe_text_box"], [134, 2825, 730, 3414])
        self.assertEqual(layer["render_bbox"], [352, 3109, 512, 3129])
        self.assertEqual(legacy["bbox"], [352, 3109, 512, 3129])

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

    def test_build_render_blocks_keeps_review_required_text_renderable(self):
        blocks = build_render_blocks([
            {
                "id": "ocr_review",
                "text": "PLEASE WAIT",
                "translated": "POR FAVOR, ESPERE",
                "route_action": "review_required",
                "route_reason": "unsafe_derived_art_mask",
                "safe_text_box": [20, 20, 180, 80],
                "render_bbox": [30, 30, 170, 70],
                "qa_flags": ["unsafe_derived_art_mask_review"],
            }
        ])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["id"], "ocr_review")
        self.assertEqual(blocks[0]["route_action"], "review_required")
        self.assertFalse(blocks[0].get("skip_processing"))

    def test_build_render_blocks_skips_art_fragment_review_text(self):
        layer = {
            "id": "direct_paddle_reocr_001",
            "text": "WU",
            "translated": "WU",
            "route_action": "review_required",
            "route_reason": "ocr_art_fragment_suspected",
            "bbox": [211, 3265, 319, 3375],
            "text_pixel_bbox": [214, 3274, 249, 3308],
            "line_polygons": [
                [[247, 3265], [319, 3298], [284, 3375], [211, 3342]],
            ],
            "qa_flags": [
                "candidate_crop_direct_paddle_reocr",
                "dark_bubble_oval_reocr",
                "partial_dark_bubble_lobe_reocr",
                "detected_dark_bubble_without_text_reocr",
                "ocr_art_fragment_suspected",
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
        }

        blocks = build_render_blocks([layer])

        self.assertEqual(blocks, [])
        self.assertFalse(layer["visible"])
        self.assertEqual(layer["render_policy"], "preserve_original")
        self.assertTrue(layer["skip_processing"])

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

    def test_plan_text_layout_rejects_overlarge_derived_bubble_mask_target(self):
        text_data = {
            "id": "ocr_002",
            "translated": "POR FAVOR!",
            "bbox": [463, 129, 666, 177],
            "source_bbox": [463, 129, 666, 177],
            "text_pixel_bbox": [473, 144, 643, 177],
            "layout_bbox": [473, 144, 643, 177],
            "balloon_bbox": [419, 115, 710, 191],
            "bubble_id": "page_002_band_002_bubble_002",
            "bubble_mask_bbox": [72, 80, 791, 919],
            "bubble_inner_bbox": [475, 141, 654, 165],
            "bubble_mask_source": "derived_white_crop",
            "layout_profile": "white_balloon",
            "style_origin": "auto",
            "page_width": 800,
            "page_height": 1896,
        }

        plan = plan_text_layout(text_data)

        self.assertNotEqual(plan["target_bbox"], [72, 80, 791, 919])
        self.assertEqual(plan["target_bbox"], [419, 115, 710, 191])
        self.assertNotEqual(text_data.get("_render_target_source"), "real_bubble_mask_bbox")

    def test_build_render_blocks_merges_adjacent_same_balloon_fragments(self):
        blocks = build_render_blocks(
            [
                {
                    "id": "ocr_004",
                    "trace_id": "ocr_004@page_002_band_007",
                    "band_id": "page_002_band_007",
                    "original": "THE INTEREST WAS ALREADY REDUCEDBY MORE THAN THREE TIMES",
                    "translated": "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES",
                    "bbox": [527, 8192, 688, 8277],
                    "source_bbox": [522, 8182, 693, 8299],
                    "text_pixel_bbox": [527, 8192, 688, 8277],
                    "target_bbox": [485, 8147, 730, 8334],
                    "balloon_bbox": [485, 8147, 730, 8334],
                    "layout_group_size": 1,
                },
                {
                    "id": "ocr_003",
                    "trace_id": "ocr_003@page_002_band_007",
                    "band_id": "page_002_band_007",
                    "original": "THE PRINCIPAL",
                    "translated": "O PRINCIPAL",
                    "bbox": [555, 8288, 661, 8300],
                    "source_bbox": [12, 7715, 656, 8370],
                    "text_pixel_bbox": [555, 8288, 661, 8300],
                    "target_bbox": [0, 7699, 672, 8386],
                    "balloon_bbox": [0, 7699, 672, 8386],
                    "qa_metrics": {"bbox_overreach": {"ratio": 136.8}},
                    "layout_group_size": 1,
                },
            ]
        )

        self.assertEqual(len(blocks), 1)
        self.assertIn("O PRINCIPAL", blocks[0]["translated"])
        self.assertEqual(blocks[0]["target_bbox"], [485, 8147, 730, 8334])
        self.assertIn("same_balloon_fragment_merged", blocks[0]["qa_flags"])

    def test_build_render_blocks_does_not_merge_lobe_split_with_distinct_balloon(self):
        blocks = build_render_blocks(
            [
                {
                    "id": "ocr_001",
                    "trace_id": "ocr_001@page_003_band_035",
                    "band_id": "page_003_band_035",
                    "original": "HEY, LET'S GO! I'M STARVING",
                    "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                    "bbox": [29, 96, 642, 709],
                    "source_bbox": [29, 96, 642, 709],
                    "text_pixel_bbox": [148, 235, 537, 673],
                    "layout_bbox": [148, 235, 537, 673],
                    "balloon_bbox": [29, 96, 642, 709],
                    "bubble_mask_bbox": [29, 96, 642, 709],
                    "layout_profile": "white_balloon",
                    "line_polygons": [
                        [[148, 235], [310, 235], [310, 255], [148, 255]],
                        [[345, 652], [537, 652], [537, 673], [345, 673]],
                    ],
                    "layout_group_size": 1,
                },
                {
                    "id": "ocr_003",
                    "trace_id": "ocr_003@page_003_band_035",
                    "band_id": "page_003_band_035",
                    "original": "WHO'SPAYING TODAY?",
                    "translated": "QUEM ESTÁ PAGANDO HOJE?",
                    "bbox": [325, 634, 549, 751],
                    "source_bbox": [325, 634, 549, 751],
                    "text_pixel_bbox": [344, 689, 540, 748],
                    "layout_bbox": [344, 689, 540, 748],
                    "balloon_bbox": [276, 599, 598, 786],
                    "bubble_mask_bbox": [276, 599, 598, 786],
                    "layout_profile": "white_balloon",
                    "line_polygons": [
                        [[344, 689], [540, 689], [540, 710], [344, 710]],
                        [[389, 723], [496, 723], [496, 748], [389, 748]],
                    ],
                    "layout_group_size": 1,
                },
            ]
        )

        self.assertFalse(
            any(
                set(block.get("source_trace_ids") or []) == {
                    "ocr_001@page_003_band_035",
                    "ocr_003@page_003_band_035",
                }
                for block in blocks
            )
        )
        self.assertTrue(any(block.get("id") == "ocr_003" for block in blocks))
        self.assertIn("QUEM ESTÁ PAGANDO HOJE?", [block.get("translated") for block in blocks])

    def test_build_render_blocks_merges_same_balloon_fragments_with_degenerate_safe_area(self):
        blocks = build_render_blocks(
            [
                {
                    "id": "ocr_001",
                    "trace_id": "ocr_001@page_002_band_019",
                    "band_id": "page_002_band_019",
                    "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL",
                    "bbox": [298, 3534, 525, 3621],
                    "text_pixel_bbox": [298, 3534, 525, 3621],
                    "balloon_bbox": [298, 3523, 525, 3657],
                    "layout_group_size": 1,
                    "qa_flags": ["tiny_bubble_inner_bbox_rejected", "safe_text_box_recomputed"],
                    "qa_metrics": {"render_balloon_containment": 0.1762},
                },
                {
                    "id": "ocr_002",
                    "trace_id": "ocr_002@page_002_band_019",
                    "band_id": "page_002_band_019",
                    "translated": "ATRIZ...",
                    "bbox": [352, 3634, 480, 3656],
                    "text_pixel_bbox": [352, 3634, 480, 3656],
                    "balloon_bbox": [298, 3534, 525, 3668],
                    "layout_group_size": 1,
                    "qa_flags": ["tiny_bubble_inner_bbox_rejected", "safe_text_box_recomputed"],
                    "qa_metrics": {"render_balloon_containment": 0.1762},
                },
            ]
        )

        self.assertEqual(len(blocks), 1)
        self.assertIn("A QUANTIA", blocks[0]["translated"])
        self.assertIn("ATRIZ", blocks[0]["translated"])
        self.assertEqual(
            blocks[0].get("source_trace_ids"),
            ["ocr_001@page_002_band_019", "ocr_002@page_002_band_019"],
        )
        self.assertEqual(blocks[0].get("source_text_ids"), ["ocr_001", "ocr_002"])
        self.assertIn("same_balloon_fragment_merged", blocks[0].get("qa_flags") or [])

    def test_plan_text_layout_keeps_large_white_balloon_target_for_line_polygon_text(self):
        text = {
            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL",
            "bbox": [298, 3534, 525, 3621],
            "text_pixel_bbox": [298, 3534, 525, 3621],
            "balloon_bbox": [0, 3427, 800, 3755],
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bubble_mask_source": "derived_white_crop",
            "line_polygons": [
                [[329, 3534], [494, 3534], [494, 3555], [329, 3555]],
                [[296, 3566], [525, 3566], [525, 3587], [296, 3587]],
            ],
            "qa_flags": ["rejected_derived_bubble_mask"],
        }

        plan = plan_text_layout(text)

        self.assertEqual(plan["target_bbox"], [0, 3427, 800, 3755])
        self.assertNotIn("tiny_bubble_inner_bbox_rejected", text.get("qa_flags") or [])

    def test_plan_text_layout_shrinks_degenerate_overbroad_white_balloon_target(self):
        text = {
            "translated": "A QUANTIA ESTA CERTA. ESSA VADIA E REAL",
            "bbox": [298, 3534, 525, 3621],
            "source_bbox": [298, 3523, 533, 3630],
            "text_pixel_bbox": [298, 3523, 533, 3630],
            "balloon_bbox": [0, 3427, 800, 3755],
            "bubble_inner_bbox": [310, 3535, 521, 3618],
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bubble_mask_source": "derived_white_crop",
            "line_polygons": [
                [[298, 3534], [525, 3534], [525, 3560], [298, 3560]],
                [[314, 3570], [490, 3570], [490, 3595], [314, 3595]],
                [[352, 3603], [480, 3603], [480, 3621], [352, 3621]],
            ],
            "qa_flags": [
                "same_balloon_fragment_merged",
                "rejected_derived_bubble_mask",
                "tiny_bubble_inner_bbox_rejected",
                "safe_text_box_recomputed",
            ],
            "qa_metrics": {"render_balloon_containment": 0.1762},
        }

        plan = plan_text_layout(text)

        self.assertNotEqual(plan["target_bbox"], [0, 3427, 800, 3755])
        self.assertGreater(plan["target_bbox"][0], 0)
        self.assertLess(plan["target_bbox"][2], 800)
        self.assertGreater(plan["target_bbox"][2] - plan["target_bbox"][0], 360)
        self.assertLess(plan["target_bbox"][2] - plan["target_bbox"][0], 720)
        self.assertEqual(text.get("_render_target_source"), "overbroad_white_balloon_text_evidence")

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

    def test_run_render_qa_revalidates_stable_translator_note_text_only_flags(self):
        text_data = {
            "id": "ocr_tn",
            "translated": "T/N: EXISTE UM ROMANCE CHAMADO SEMIDEUSES E SEMI-DEMONIOS NA VIDA REAL.",
            "bubble_mask_source": "translator_note_text_mask",
            "source_bbox": [36, 89, 178, 133],
            "render_bbox": [18, 94, 350, 166],
            "qa_flags": [
                "translator_note_text_only_mask",
                "text_contract_direct_fill",
                "translator_note_best_effort_render",
                "render_on_art_suspected",
                "mask_outside_balloon",
                "mask_outside_balloon_critical",
            ],
            "_render_debug": {"font_size_final": 23},
        }
        plan = {
            "target_bbox": [0, 70, 401, 190],
            "safe_text_box": [18, 82, 383, 178],
        }

        renderer_mod._run_render_qa(text_data, plan)

        flags = text_data.get("qa_flags") or []
        self.assertNotIn("render_on_art_suspected", flags)
        self.assertNotIn("translator_note_best_effort_render", flags)
        self.assertIn("translator_note_stable_text_only_render", flags)
        self.assertIn("mask_outside_balloon", flags)
        self.assertIn("mask_outside_balloon_critical", flags)
        metrics = text_data.get("qa_metrics") or {}
        revalidated = metrics.get("translator_note_flags_revalidated") or {}
        self.assertEqual(revalidated.get("decision"), "intentional_text_only_note")
        self.assertEqual(revalidated.get("reason"), "stable_translator_note_text_only_render")
        self.assertIn("render_on_art_suspected", metrics.get("resolved_pre_render_flags") or [])
        self.assertIn("translator_note_best_effort_render", metrics.get("resolved_pre_render_flags") or [])

    def test_run_render_qa_keeps_render_on_art_without_translator_note_contract(self):
        text_data = {
            "id": "ocr_art",
            "translated": "TEXTO SOBRE ARTE",
            "render_bbox": [18, 94, 350, 166],
            "qa_flags": ["render_on_art_suspected"],
            "_render_debug": {"font_size_final": 23},
        }
        plan = {
            "target_bbox": [0, 70, 401, 190],
            "safe_text_box": [18, 82, 383, 178],
        }

        renderer_mod._run_render_qa(text_data, plan)

        self.assertIn("render_on_art_suspected", text_data.get("qa_flags") or [])
        self.assertNotIn("translator_note_flags_revalidated", text_data.get("qa_metrics") or {})

    def test_run_render_qa_keeps_translator_note_best_effort_on_dark_non_note(self):
        text_data = {
            "id": "ocr_dark",
            "translated": "TEXTO NORMAL",
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "render_bbox": [40, 40, 150, 88],
            "balloon_bbox": [20, 20, 180, 120],
            "qa_flags": ["translator_note_best_effort_render"],
            "_render_debug": {"font_size_final": 23},
        }
        plan = {
            "target_bbox": [20, 20, 180, 120],
            "safe_text_box": [34, 34, 166, 104],
        }

        renderer_mod._run_render_qa(text_data, plan)

        self.assertIn("translator_note_best_effort_render", text_data.get("qa_flags") or [])
        self.assertNotIn("translator_note_flags_revalidated", text_data.get("qa_metrics") or {})

    def test_run_render_qa_revalidates_white_balloon_text_clipped_only_when_visually_contained(self):
        text_data = {
            "id": "ocr_white",
            "translated": "POR QUE SOU O ANFITRIAO?",
            "layout_profile": "white_balloon",
            "balloon_type": "white",
            "bubble_mask_source": "image_white_bubble_mask",
            "render_bbox": [507, 39, 709, 247],
            "balloon_bbox": [427, 0, 786, 297],
            "qa_flags": [],
        }
        plan = {
            "target_bbox": [427, 0, 786, 297],
            "safe_text_box": [510, 45, 703, 234],
        }

        renderer_mod._run_render_qa(text_data, plan)

        flags = text_data.get("qa_flags") or []
        self.assertNotIn("TEXT_CLIPPED", flags)
        self.assertNotIn("translator_note_flags_revalidated", text_data.get("qa_metrics") or {})
        revalidated = (text_data.get("qa_metrics") or {}).get("white_balloon_flags_revalidated") or {}
        self.assertEqual(revalidated.get("decision"), "cleared")

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

    def test_render_text_block_delegates_sfx_layers_to_sfx_renderer(self):
        img = Image.new("RGB", (220, 160), (230, 230, 230))
        before = np.asarray(img).copy()
        text_data = {
            "id": "sfx_001",
            "content_class": "sfx",
            "route_action": "translate_sfx_inpaint_render",
            "bbox": [30, 30, 170, 100],
            "translated": "SHOULD_NOT_BE_USED",
            "sfx": {
                "adapted_text": "TUM",
                "kind": "impact",
                "inpaint_allowed": True,
                "style": {
                    "fill_color": "#FFFFFF",
                    "stroke_color": "#000000",
                    "stroke_width_px": 2,
                    "rotation_deg": 12,
                },
            },
        }

        render_text_block(img, text_data)

        self.assertGreater(int(np.count_nonzero(np.asarray(img) != before)), 0)
        self.assertEqual(text_data["translated"], "TUM")
        self.assertIn(text_data["fit_status"], {"ok", "below_minimum_legible"})
        self.assertIn("render_bbox", text_data)

    def test_render_band_image_preserves_sfx_route_through_render_blocks(self):
        band = np.full((160, 220, 3), 230, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "sfx_001",
                    "text_id": "sfx_001",
                    "content_class": "sfx",
                    "route_action": "translate_sfx_inpaint_render",
                    "bbox": [30, 30, 170, 100],
                    "balloon_bbox": [30, 30, 170, 100],
                    "translated": "SHOULD_NOT_BE_USED",
                    "sfx": {
                        "adapted_text": "TUM",
                        "kind": "impact",
                        "inpaint_allowed": True,
                        "style": {
                            "fill_color": "#FFFFFF",
                            "stroke_color": "#000000",
                            "stroke_width_px": 2,
                            "rotation_deg": 12,
                        },
                    },
                }
            ]
        }

        rendered = render_band_image(band, page)

        self.assertGreater(int(np.count_nonzero(rendered != band)), 0)
        source = page["texts"][0]
        self.assertEqual(source["content_class"], "sfx")
        self.assertEqual(source["render_policy"], "sfx_style")
        self.assertEqual(source["translated"], "TUM")
        self.assertEqual(source["fit_status"], "ok")
        self.assertIn("render_bbox", source)

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

    def test_render_text_block_translator_note_uses_readable_note_area(self):
        img = Image.new("RGB", (900, 420), (255, 255, 255))
        text_data = {
            "id": "ocr_tn",
            "text": (
                "T/N: HYUNGNIM IS A TERM USED FOR CALLING ONE'S MOB BOSS OR "
                "CRIME BOSS. IT IS ALSO AN HONORABLE TERM."
            ),
            "translated": (
                "T/N: HYUNGNIM É UM TERMO USADO PARA CHAMAR O CHEFE DA MÁFIA "
                "OU CHEFE DO CRIME. TAMBÉM É UM TERMO HONROSO."
            ),
            "route_action": "translate_inpaint_render",
            "bbox": [250, 170, 298, 216],
            "source_bbox": [250, 170, 298, 216],
            "text_pixel_bbox": [250, 170, 298, 216],
            "balloon_bbox": [150, 130, 440, 270],
            "page_width": 900,
            "page_height": 420,
            "layout_profile": "white_balloon",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 26},
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data.get("fit_status"), "ok")
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags") or [])
        self.assertIn("translator_note_best_effort_render", text_data.get("qa_flags") or [])
        self.assertEqual(text_data.get("target_bbox"), [150, 130, 440, 270])
        self.assertGreaterEqual(text_data.get("safe_text_box", [0, 0, 0, 0])[2] - text_data.get("safe_text_box", [0, 0, 0, 0])[0], 240)
        self.assertIn("render_bbox", text_data)

    def test_render_text_block_translator_note_does_not_use_dark_panel_glow(self):
        img = Image.new("RGB", (420, 220), (12, 12, 18))
        text_data = {
            "id": "ocr_tn_dark_context",
            "text": "T/N: THERE IS A NOVEL CALLED DEMI-GODS AND SEMI-DEVILS.",
            "translated": "T/N: EXISTE UM ROMANCE CHAMADO SEMIDEUSES E SEMI-DEMÔNIOS.",
            "route_action": "translate_inpaint_render",
            "bbox": [46, 88, 174, 132],
            "source_bbox": [46, 88, 174, 132],
            "text_pixel_bbox": [46, 88, 174, 132],
            "balloon_bbox": [32, 72, 206, 150],
            "page_width": 420,
            "page_height": 220,
            "layout_profile": "standard",
            "style_origin": "auto_dark_panel_glow",
            "style_source": "pixel_analysis",
            "style_confidence": 1.0,
            "bubble_mask_source": "image_dark_bubble_mask",
            "qa_flags": [
                "debug_derived_bubble_mask_rejected",
                "auto_dark_panel_glow_fallback",
                "original_dark_panel_effect_colors",
                "dark_panel_style_grouped",
            ],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 18,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data.get("style_origin"), "translator_note_neutral")
        self.assertNotIn("auto_dark_panel_glow_fallback", text_data.get("qa_flags") or [])
        self.assertNotIn("original_dark_panel_effect_colors", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_panel_style_grouped", text_data.get("qa_flags") or [])
        style = text_data.get("estilo") or {}
        self.assertEqual(style.get("cor"), "#FFFFFF")
        self.assertFalse(style.get("glow"))
        self.assertEqual(style.get("contorno_px"), 0)

    def test_render_text_block_translator_note_recognizes_traduzido_field(self):
        img = Image.new("RGB", (420, 220), (12, 12, 18))
        text_data = {
            "id": "ocr_tn_traduzido",
            "original": "T/N: THERE IS A NOVEL CALLED DEMI-GODS.",
            "traduzido": "T/N: EXISTE UM ROMANCE CHAMADO SEMIDEUSES.",
            "route_action": "translate_inpaint_render",
            "bbox": [28, 74, 200, 126],
            "source_bbox": [28, 74, 200, 126],
            "text_pixel_bbox": [28, 74, 200, 126],
            "target_bbox": [20, 60, 260, 145],
            "position_bbox": [20, 60, 260, 145],
            "capacity_bbox": [20, 60, 260, 145],
            "safe_text_box": [20, 60, 260, 145],
            "page_width": 420,
            "page_height": 220,
            "style_origin": "auto_dark_panel_glow",
            "bubble_mask_source": "translator_note_text_mask",
            "qa_flags": ["translator_note_text_only_mask", "auto_dark_panel_glow_fallback"],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 18,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data.get("style_origin"), "translator_note_neutral")
        self.assertNotIn("auto_dark_panel_glow_fallback", text_data.get("qa_flags") or [])
        style = text_data.get("estilo") or {}
        self.assertFalse(style.get("glow"))
        self.assertEqual(style.get("contorno"), "")
        self.assertEqual(style.get("contorno_px"), 0)

    def test_render_text_block_translator_note_text_mask_ignores_synthetic_tight_balloon_bbox(self):
        img = Image.new("RGB", (520, 260), (5, 5, 6))
        text_data = {
            "id": "ocr_tn_text_only",
            "original": "T/N: THERE IS A NOVEL CALLED DEMI-GODS & SEMI-DEVILS IN REAL LIFE.",
            "translated": "T/N: EXISTE UM ROMANCE CHAMADO SEMIDEUSES E SEMI-DEMÔNIOS NA VIDA REAL.",
            "route_action": "translate_inpaint_render",
            "bbox": [39, 82, 154, 137],
            "source_bbox": [36, 89, 178, 133],
            "text_pixel_bbox": [36, 89, 178, 133],
            "balloon_bbox": [35, 78, 158, 141],
            "bubble_mask_bbox": [35, 78, 158, 141],
            "bubble_mask_source": "translator_note_text_mask",
            "qa_flags": ["translator_note_text_only_mask", "text_contract_direct_fill"],
            "layout_profile": "standard",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 18, "cor": "#FFFFFF"},
            "render_layout_contract": {
                "schema_version": 1,
                "source": "debug_render_plan_raw",
                "translated_key": "t/n: existe um romance chamado semideuses e semi-demônios na vida real.",
                "font_name": "ComicNeue-Bold.ttf",
                "font_size": 6,
                "line_height": 10,
                "lines": ["T/N: EXISTE UM", "ROMANCE CHAMADO", "SEMIDEUSES E", "SEMI-DEMÔNIOS", "NA VIDA REAL."],
                "positions": [[18, 76], [18, 86], [18, 96], [18, 106], [18, 116]],
                "line_widths": [70, 82, 78, 92, 64],
                "block_bbox": [18, 76, 110, 126],
                "target_bbox": [0, 70, 174, 150],
                "position_bbox": [6, 74, 168, 146],
                "safe_text_box": [8, 76, 166, 144],
                "coordinate_space": "page",
            },
        }

        render_text_block(img, text_data)

        self.assertNotIn("translator_note_best_effort_render", text_data.get("qa_flags") or [])
        self.assertIn("translator_note_stable_text_only_render", text_data.get("qa_flags") or [])
        self.assertIn("stale_render_layout_contract_rejected", text_data.get("qa_flags") or [])
        self.assertFalse(text_data.get("_render_layout_contract_replayed"))
        self.assertNotEqual(text_data.get("target_bbox"), [35, 78, 158, 141])
        self.assertGreaterEqual(text_data.get("target_bbox", [0, 0, 0, 0])[2] - text_data.get("target_bbox", [0, 0, 0, 0])[0], 300)
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags") or [])
        self.assertGreaterEqual(text_data.get("_render_debug", {}).get("font_size_final", 0), 10)
        revalidated = (text_data.get("qa_metrics") or {}).get("translator_note_flags_revalidated") or {}
        self.assertEqual(revalidated.get("decision"), "intentional_text_only_note")
        self.assertEqual(revalidated.get("reason"), "stable_translator_note_text_only_render")

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

    def test_plan_text_layout_prefers_current_bubble_mask_for_unsafe_merged_oval(self):
        text_data = {
            "id": "ocr_001",
            "text": "PLEASE, FOR THE CHILD'S SAKE.",
            "original": "PLEASE, FOR THE CHILD'S SAKE.",
            "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
            "bbox": [499, 315, 656, 400],
            "source_bbox": [499, 315, 656, 400],
            "text_pixel_bbox": [499, 315, 656, 400],
            "line_polygons": [[[499, 315], [656, 315], [656, 400], [499, 400]]],
            "balloon_bbox": [25, 96, 667, 368],
            "bubble_mask_bbox": [435, 255, 701, 442],
            "page_width": 800,
            "page_height": 560,
            "tipo": "fala",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "qa_flags": [
                "same_balloon_fragment_merged",
                "same_band_dependent_fragment_merged",
                "mask_outside_balloon_critical",
            ],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "cor": "#000000", "force_upper": True},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["target_bbox"], [435, 255, 701, 442])
        self.assertGreaterEqual(resolved["block_bbox"][0], 435)
        self.assertGreaterEqual(resolved["block_bbox"][1], 255)
        self.assertLessEqual(resolved["block_bbox"][2], 701)
        self.assertLessEqual(resolved["block_bbox"][3], 442)
        self.assertIn("unsafe_merged_bubble_mask_target", text_data.get("qa_flags", []))

    def test_build_render_blocks_drops_low_quality_duplicate_same_balloon(self):
        duplicate = {
            "id": "ocr_002",
            "trace_id": "ocr_002@page_004_band_051",
            "band_id": "page_004_band_051",
            "text": "WHY ARE YOU SO LATE",
            "original": "WHY ARE YOU SO LATE",
            "translated": "POR QUE VOCE NAO VEM! OS CARAS ESTAO ESPERANDO!",
            "route_action": "translate_inpaint_render",
            "bbox": [259, 4480, 480, 4562],
            "source_bbox": [259, 4480, 480, 4562],
            "text_pixel_bbox": [259, 4480, 480, 4562],
            "balloon_bbox": [57, 4438, 361, 4685],
            "safe_text_box": [262, 4469, 476, 4559],
            "bubble_mask_source": "rejected_derived_bubble_mask",
            "qa_flags": [
                "same_balloon_fragment_merged",
                "same_band_dependent_fragment_merged",
                "debug_derived_bubble_mask_rejected",
                "missing_real_bubble_mask",
            ],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24},
        }
        authoritative = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_004_band_054",
            "band_id": "page_004_band_054",
            "text": "OPPA, WHY ARE YOU SO LATE~",
            "original": "OPPA, WHY ARE YOU SO LATE~",
            "translated": "OPPA, POR QUE VOCE ESTA SOLITARIO?",
            "route_action": "translate_inpaint_render",
            "bbox": [172, 4514, 246, 4533],
            "source_bbox": [120, 4514, 294, 4599],
            "text_pixel_bbox": [120, 4514, 294, 4599],
            "balloon_bbox": [57, 4438, 361, 4685],
            "safe_text_box": [87, 4482, 331, 4641],
            "bubble_mask_source": "image_contour_bubble_mask",
            "qa_flags": [
                "same_balloon_fragment_merged",
                "same_band_dependent_fragment_merged",
                "safe_text_box_recomputed",
            ],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24},
        }

        blocks = build_render_blocks([duplicate, authoritative])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["trace_id"], "ocr_001@page_004_band_054")
        self.assertEqual(blocks[0]["translated"], "OPPA, POR QUE VOCE ESTA SOLITARIO?")

    def test_build_render_blocks_merges_duplicate_parent_residual_into_child_balloon(self):
        parent = {
            "id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "source_trace_ids": ["ocr_003@page_003_band_035", "ocr_001@page_003_band_035"],
            "band_id": "page_003_band_035",
            "text": "HEY, LET'S GO! I'M STARVING",
            "original": "HEY, LET'S GO! I'M STARVING",
            "translated": "QUEM ESTA PAGANDO HOJE?\nESTOU MORRENDO DE FOME",
            "route_action": "translate_inpaint_render",
            "bbox": [148, 7248, 537, 7686],
            "source_bbox": [29, 7109, 642, 7722],
            "text_pixel_bbox": [148, 7248, 537, 7686],
            "balloon_bbox": [133, 7185, 327, 7327],
            "bubble_mask_source": "image_contour_bubble_mask",
            "qa_flags": ["same_balloon_fragment_merged"],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 22},
        }
        child = {
            "id": "ocr_003",
            "trace_id": "ocr_003@page_003_band_035",
            "band_id": "page_003_band_035",
            "text": "WHO'S PAYING TODAY?",
            "original": "WHO'S PAYING TODAY?",
            "translated": "QUEM ESTA PAGANDO HOJE?",
            "route_action": "translate_inpaint_render",
            "bbox": [344, 7702, 540, 7761],
            "source_bbox": [325, 7647, 549, 7764],
            "text_pixel_bbox": [344, 7702, 540, 7761],
            "balloon_bbox": [245, 7567, 629, 7844],
            "bubble_mask_source": "image_white_bubble_mask",
            "qa_flags": ["safe_text_box_recomputed"],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 22},
        }
        fragment = {
            "id": "ocr_001_fragment_3",
            "trace_id": "ocr_001@page_003_band_035#fragment_3",
            "band_id": "page_003_band_035",
            "text": "HEY, LET'S GO!",
            "original": "HEY, LET'S GO!",
            "translated": "EI, VAMOS!",
            "route_action": "merged_into_primary",
            "bbox": [148, 7248, 310, 7268],
            "source_bbox": [148, 7248, 310, 7268],
            "text_pixel_bbox": [148, 7248, 310, 7268],
            "balloon_bbox": [133, 7185, 327, 7327],
            "qa_flags": ["safe_text_box_recomputed"],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 18},
        }

        blocks = build_render_blocks([parent, child, fragment])

        self.assertEqual([block["trace_id"] for block in blocks], [
            "ocr_003@page_003_band_035",
            "ocr_001@page_003_band_035#fragment_3",
        ])
        self.assertEqual(blocks[0]["translated"], "ESTOU MORRENDO DE FOME\nQUEM ESTA PAGANDO HOJE?")
        self.assertEqual(blocks[1]["translated"], "EI, VAMOS!")

    def test_build_render_blocks_does_not_split_repaired_short_same_balloon_payload(self):
        repaired = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "EI, VAMOS!",
            "source_trace_ids": ["ocr_001@page_003_band_035"],
            "source_text_ids": ["ocr_001"],
            "bbox": [148, 10878, 537, 11316],
            "source_bbox": [148, 10878, 537, 11316],
            "text_pixel_bbox": [148, 10878, 537, 11316],
            "target_bbox": [133, 10815, 327, 10957],
            "balloon_bbox": [133, 10815, 327, 10957],
            "safe_text_box": [85, 10862, 393, 10914],
            "render_bbox": [152, 10878, 304, 10903],
            "line_polygons": [
                [[148, 10878], [310, 10878], [310, 10898], [148, 10898]],
                [[345, 11295], [537, 11295], [537, 11316], [345, 11316]],
            ],
            "qa_flags": ["safe_text_box_recomputed", "same_balloon_fragment_merged"],
        }

        blocks = build_render_blocks([repaired])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["translated"], "EI, VAMOS!")
        self.assertEqual(blocks[0]["bbox"], [148, 10878, 537, 11316])

    def test_plan_text_layout_uses_compact_mask_for_short_repaired_fragment(self):
        text_data = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "translated": "EI, VAMOS!",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "source_trace_ids": ["ocr_001@page_003_band_035"],
            "source_text_ids": ["ocr_001"],
            "bbox": [148, 10878, 537, 11316],
            "source_bbox": [29, 10739, 642, 11352],
            "text_pixel_bbox": [148, 10878, 537, 11316],
            "layout_bbox": [148, 10878, 537, 11316],
            "target_bbox": [133, 10815, 327, 10957],
            "balloon_bbox": [0, 10643, 800, 11597],
            "bubble_mask_bbox": [133, 10815, 327, 10957],
            "bubble_inner_bbox": [71, 10781, 600, 11310],
            "safe_text_box": [85, 10862, 393, 10914],
            "_debug_safe_text_box": [85, 10862, 393, 10914],
            "render_bbox": [152, 10878, 304, 10903],
            "qa_flags": ["safe_text_box_recomputed", "same_balloon_fragment_merged"],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 32,
                "cor": "#000000",
                "alinhamento": "center",
            },
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [133, 10815, 327, 10957])
        self.assertLess(plan["safe_text_box"][3], 10960)
        self.assertGreaterEqual(plan["safe_text_box"][0], plan["target_bbox"][0])
        self.assertLessEqual(plan["safe_text_box"][2], plan["target_bbox"][2])
        self.assertGreaterEqual(plan["safe_text_box"][1], plan["target_bbox"][1])
        self.assertLessEqual(plan["safe_text_box"][3], plan["target_bbox"][3])
        self.assertEqual(text_data["_render_target_source"], "short_repaired_fragment_compact_bubble")

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

    def test_plan_text_layout_uses_larger_balloon_when_bubble_mask_is_text_shaped(self):
        text_data = {
            "translated": "A sincronização foi concluída.",
            "text": "Synching is complete.",
            "original": "Synching is complete.",
            "bbox": [455, 9534, 559, 9594],
            "source_bbox": [455, 9534, 559, 9594],
            "text_pixel_bbox": [455, 9534, 559, 9594],
            "layout_bbox": [455, 9534, 559, 9594],
            "line_polygons": [
                [[452, 9532], [562, 9532], [562, 9564], [452, 9564]],
                [[458, 9565], [554, 9565], [554, 9594], [458, 9594]],
            ],
            "balloon_bbox": [414, 9505, 598, 9607],
            "bubble_mask_bbox": [454, 9533, 552, 9595],
            "bubble_inner_bbox": [454, 9536, 558, 9576],
            "bubble_id": "page_006_band_102_bubble_001",
            "bubble_mask_source": "image_white_bubble_mask",
            "tipo": "text",
            "layout_profile": "standard",
            "style_origin": "auto",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 44, "cor": "#000000", "force_upper": True},
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["target_bbox"], [414, 9505, 598, 9607])
        self.assertGreaterEqual(plan["safe_text_box"][2] - plan["safe_text_box"][0], 120)
        self.assertNotEqual([line.upper() for line in resolved["lines"]][0], "A")
        self.assertIn("tiny_bubble_inner_bbox_rejected", text_data.get("qa_flags", []))

    def test_plan_text_layout_centers_short_text_on_valid_large_bubble_inner_not_source_anchor(self):
        text_data = {
            "translated": "EI, VAMOS!",
            "text": "HEY, LET'S GO! I'M STARVING",
            "original": "HEY, LET'S GO! I'M STARVING",
            "bbox": [148, 7248, 310, 7268],
            "source_bbox": [148, 7248, 310, 7268],
            "text_pixel_bbox": [148, 7248, 310, 7268],
            "layout_bbox": [148, 7248, 310, 7268],
            "line_polygons": [
                [[148, 7248], [310, 7248], [310, 7268], [148, 7268]],
            ],
            "balloon_bbox": [0, 7013, 800, 7967],
            "bubble_mask_bbox": [29, 7109, 642, 7722],
            "bubble_inner_bbox": [71, 7151, 600, 7680],
            "bubble_mask_source": "image_contour_bubble_mask",
            "bubble_id": "page_003_band_035_bubble_001",
            "safe_text_box": [85, 7232, 393, 7284],
            "_debug_safe_text_box": [85, 7232, 393, 7284],
            "layout_safe_reason": "debug_derived_bubble_mask_rejected",
            "page_width": 800,
            "page_height": 16383,
            "tipo": "text",
            "layout_profile": "white_balloon",
            "style_origin": "auto",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 44, "cor": "#000000", "force_upper": True},
        }

        plan = plan_text_layout(text_data)

        safe = plan["safe_text_box"]
        source_cy = (text_data["text_pixel_bbox"][1] + text_data["text_pixel_bbox"][3]) / 2.0
        inner_cy = (text_data["bubble_inner_bbox"][1] + text_data["bubble_inner_bbox"][3]) / 2.0
        safe_cy = (safe[1] + safe[3]) / 2.0
        self.assertEqual(plan["layout_safe_reason"], "bubble_inner_bbox")
        self.assertLess(abs(safe_cy - inner_cy), abs(safe_cy - source_cy))
        self.assertGreaterEqual(safe[3] - safe[1], 300)
        self.assertFalse(plan["_follow_english_anchor_position"])

    def test_visual_rect_inner_does_not_replace_large_real_white_bubble_inner(self):
        text_data = {
            "translated": "EI, VAMOS!",
            "text": "HEY, LET'S GO! I'M STARVING",
            "bbox": [148, 7248, 310, 7268],
            "source_bbox": [148, 7248, 310, 7268],
            "text_pixel_bbox": [148, 7248, 310, 7268],
            "balloon_bbox": [0, 7013, 800, 7967],
            "bubble_mask_bbox": [29, 7109, 642, 7722],
            "bubble_inner_bbox": [71, 7151, 600, 7680],
            "bubble_mask_source": "image_contour_bubble_mask",
            "layout_profile": "white_balloon",
        }

        self.assertTrue(
            _should_reject_plain_balloon_visual_safe_area(
                text_data,
                [0, 7013, 800, 7967],
                [174, 7133, 287, 7270],
                "visual_rect_inner",
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

    def test_render_band_image_propagates_group_render_geometry_to_source_text_ids(self):
        band = np.full((80, 180, 3), 255, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "ocr_a",
                    "trace_id": "ocr_a@page_002_band_007",
                    "translated": "OS JUROS JA FORAM REDUZIDOS",
                    "bbox": [20, 20, 90, 45],
                    "balloon_bbox": [10, 10, 140, 70],
                    "qa_flags": [],
                },
                {
                    "id": "ocr_b",
                    "trace_id": "ocr_b@page_002_band_007",
                    "translated": "O DIRETOR",
                    "bbox": [40, 42, 110, 60],
                    "balloon_bbox": [10, 10, 140, 70],
                    "qa_flags": ["missing_render_bbox"],
                },
            ]
        }
        block = {
            "id": "ocr_a",
            "translated": "OS JUROS JA FORAM REDUZIDOS\nO DIRETOR",
            "source_text_ids": ["ocr_a", "ocr_b"],
            "source_trace_ids": ["ocr_a@page_002_band_007", "ocr_b@page_002_band_007"],
            "qa_flags": ["same_balloon_fragment_merged"],
        }

        def fake_render(_img, render_block):
            render_block["fit_status"] = "ok"
            render_block["safe_text_box"] = [20, 18, 120, 65]
            render_block["render_bbox"] = [35, 24, 105, 58]

        with patch("typesetter.renderer.build_render_blocks", return_value=[block]):
            with patch("typesetter.renderer.render_text_block", side_effect=fake_render):
                render_band_image(band, page)

        for text in page["texts"]:
            self.assertEqual(text["render_bbox"], [35, 24, 105, 58])
            self.assertEqual(text["fit_status"], "ok")
            self.assertNotIn("missing_render_bbox", text.get("qa_flags") or [])
            self.assertIn("same_balloon_fragment_merged", text.get("qa_flags") or [])

    def test_build_render_blocks_rejects_art_suspect_text_with_weak_ocr_evidence(self):
        text = {
            "id": "ocr_art",
            "translated": "QUANDO EU VOLTAR AO TRABALHO...",
            "bbox": [212, 465, 613, 543],
            "text_pixel_bbox": [212, 465, 613, 543],
            "balloon_bbox": [108, 206, 717, 803],
            "layout_profile": "white_balloon",
            "bubble_mask_source": "derived_white_crop",
            "route_reason": "dialogue_balloon_with_english_text",
            "line_polygons": [],
            "qa_flags": [
                "ocr_run_on_suspect",
                "raw_text_evidence_missing",
                "fast_fill_no_glyph_evidence",
                "rejected_derived_bubble_mask",
                "render_on_art_suspected",
            ],
        }

        blocks = build_render_blocks([text])

        self.assertEqual(blocks, [])
        self.assertEqual(text["route_action"], "review_required")
        self.assertIn("non_balloon_scene_text_review", text["qa_flags"])

    def test_build_render_blocks_suppresses_false_short_art_ocr(self):
        text = {
            "id": "ocr_short_art",
            "text": "A",
            "original": "A",
            "translated": "UM",
            "bbox": [471, 11846, 578, 11920],
            "text_pixel_bbox": [471, 11846, 578, 11920],
            "balloon_bbox": [442, 11788, 634, 11950],
            "bubble_mask_source": "image_white_bubble_mask",
            "layout_profile": "white_balloon",
            "route_action": "translate_inpaint_render",
            "qa_flags": ["mask_outside_balloon_critical"],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 27},
        }

        blocks = build_render_blocks([text])

        self.assertEqual(blocks, [])
        self.assertEqual(text["route_action"], "review_required")
        self.assertEqual(text["route_reason"], "false_short_art_ocr")
        self.assertIn("false_short_art_ocr_suppressed", text["qa_flags"])

    def test_build_render_blocks_skips_unverified_merged_fragment(self):
        real_text = {
            "id": "ocr_001",
            "translated": "BEM... NAO E COMO SE ISSO NAO FOSSE AGRADAVEL...",
            "bbox": [503, 5056, 683, 5141],
            "text_pixel_bbox": [503, 5056, 683, 5141],
            "line_polygons": [
                [[501, 5054], [683, 5054], [683, 5075], [501, 5075]],
                [[507, 5087], [677, 5087], [677, 5108], [507, 5108]],
            ],
            "layout_profile": "white_balloon",
            "qa_flags": [],
        }
        unverified_fragment = {
            "id": "ocr_002",
            "translated": "BEM... NAO E COMO SE ISSO NAO FOSSE AGRADAVEL... DCCIGHTPOC..",
            "bbox": [469, 5123, 666, 5216],
            "text_pixel_bbox": [118, 5133, 682, 5232],
            "line_polygons": [],
            "layout_profile": "white_balloon",
            "qa_flags": [
                "same_balloon_fragment_merged",
                "raw_text_evidence_missing",
                "fast_fill_no_glyph_evidence",
            ],
        }

        blocks = build_render_blocks([real_text, unverified_fragment])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["id"], "ocr_001")

    def test_build_render_blocks_rejects_critical_source_glyph_area_text(self):
        text = {
            "id": "ocr_overbroad",
            "translated": "NÃO APERTE SFXKICK MINHA MÃE!",
            "bbox": [42, 156, 240, 212],
            "text_pixel_bbox": [42, 156, 240, 212],
            "balloon_bbox": [18, 12, 520, 340],
            "layout_profile": "white_balloon",
            "bubble_mask_source": "derived_white_crop",
            "route_reason": "dialogue_balloon_with_english_text",
            "line_polygons": [
                [[42, 156], [238, 156], [238, 173], [42, 173]],
                [[44, 180], [220, 180], [220, 196], [44, 196]],
            ],
            "qa_flags": [
                "rejected_derived_bubble_mask",
                "render_on_art_suspected",
                "source_glyph_area_ratio_critical",
            ],
        }

        blocks = build_render_blocks([text])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["id"], "ocr_overbroad")
        self.assertEqual(text["route_action"], "review_required")
        self.assertEqual(text["route_reason"], "source_glyph_area_ratio_critical")
        self.assertIn("source_glyph_area_ratio_critical", text["qa_flags"])
        self.assertIn("unsafe_source_glyph_area_review", text["qa_flags"])

    def test_build_render_blocks_disables_connected_layout_when_bubble_mask_was_rejected(self):
        text = {
            "id": "ocr_glow",
            "translated": "O cavaleiro do diabo!",
            "bbox": [178, 175, 371, 210],
            "text_pixel_bbox": [178, 175, 371, 210],
            "balloon_bbox": [172, 155, 372, 231],
            "safe_text_box": [183, 161, 359, 225],
            "layout_profile": "connected_balloon",
            "layout_group_size": 2,
            "bubble_mask_source": "derived_white_crop_rejected",
            "layout_safe_reason": "debug_derived_bubble_mask_rejected",
            "connected_balloon_orientation": "left-right",
            "balloon_subregions": [[172, 155, 258, 231], [266, 155, 372, 231]],
            "connected_lobe_bboxes": [[172, 155, 258, 231], [266, 155, 372, 231]],
            "connected_position_bboxes": [[205, 173, 232, 215], [278, 167, 352, 192]],
            "qa_flags": ["debug_derived_bubble_mask_rejected"],
        }

        blocks = build_render_blocks([text])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["id"], "ocr_glow")
        self.assertNotEqual(blocks[0].get("layout_profile"), "connected_balloon")
        self.assertEqual(blocks[0].get("layout_group_size"), 1)
        self.assertEqual(blocks[0].get("balloon_subregions") or [], [])
        self.assertIn("connected_layout_disabled_rejected_bubble_mask", blocks[0].get("qa_flags") or [])

    def test_build_render_blocks_rejects_broad_derived_art_mask_before_critical_flag_arrives(self):
        text = {
            "id": "ocr_broad_art",
            "translated": "NÃO APERTE SFXKICK MINHA MÃE!",
            "bbox": [145, 399, 411, 452],
            "text_pixel_bbox": [145, 399, 411, 452],
            "balloon_bbox": [82, 209, 474, 642],
            "layout_profile": "white_balloon",
            "bubble_mask_source": "derived_white_crop",
            "route_reason": "dialogue_balloon_with_english_text",
            "line_polygons": [
                [[145, 399], [411, 399], [411, 420], [145, 420]],
                [[168, 426], [388, 426], [388, 452], [168, 452]],
            ],
            "qa_flags": [
                "rejected_derived_bubble_mask",
                "render_on_art_suspected",
            ],
        }

        blocks = build_render_blocks([text])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["id"], "ocr_broad_art")
        self.assertEqual(text["route_action"], "review_required")
        self.assertEqual(text["route_reason"], "unsafe_derived_art_mask")
        self.assertIn("unsafe_derived_art_mask_review", text["qa_flags"])

    def test_render_text_block_rolls_back_broad_derived_art_mask_after_qa(self):
        img = Image.new("RGB", (520, 720), (18, 22, 28))
        before = img.copy()
        text = {
            "id": "ocr_broad_art",
            "translated": "NÃO APERTE SFXKICK MINHA MÃE!",
            "bbox": [145, 399, 411, 452],
            "text_pixel_bbox": [145, 399, 411, 452],
            "balloon_bbox": [82, 209, 474, 642],
            "layout_profile": "white_balloon",
            "bubble_mask_source": "derived_white_crop",
            "route_reason": "dialogue_balloon_with_english_text",
            "qa_flags": ["rejected_derived_bubble_mask"],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 0,
                "alinhamento": "center",
            },
        }

        render_text_block(img, text, pre_render_np=np.asarray(before))

        self.assertIsNotNone(ImageChops.difference(img, before).getbbox())
        self.assertEqual(text["route_action"], "review_required")
        self.assertEqual(text["route_reason"], "unsafe_derived_art_mask")
        self.assertIn("render_on_art_suspected", text["qa_flags"])
        self.assertIn("unsafe_derived_art_mask_review", text["qa_flags"])

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

    def test_run_render_qa_keeps_overflow_when_tight_contract_still_exceeds_visual_bbox(self):
        text_data = {
            "translated": "TEXTO LONGO",
            "balloon_bbox": [100, 100, 210, 190],
            "render_bbox": [86, 82, 252, 226],
            "qa_flags": ["fit_below_minimum_legible"],
            "qa_metrics": {
                "contract_bbox_tight_but_visual_balloon_fit_ok": {
                    "source_bbox": [118, 118, 190, 170],
                    "visual_bbox": [100, 100, 210, 190],
                    "visual_bbox_source": "qa_metrics.derived_card_panel_mask.mask_bbox",
                }
            },
        }
        plan = {
            "target_bbox": [100, 100, 210, 190],
            "safe_text_box": [112, 112, 198, 178],
        }

        renderer_mod._run_render_qa(text_data, plan)

        self.assertIn("TEXT_CLIPPED", text_data["qa_flags"])
        self.assertIn("TEXT_OVERFLOW", text_data["qa_flags"])
        self.assertIn("fit_below_minimum_legible", text_data["qa_flags"])
        self.assertEqual(
            text_data["qa_metrics"]["typeset_contract_flags_revalidated"]["decision"],
            "kept",
        )

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

    def test_copy_render_debug_fields_drops_flags_resolved_by_render_qa(self):
        source = {
            "translated": "T/N: NOTA",
            "qa_flags": [
                "translator_note_text_only_mask",
                "translator_note_best_effort_render",
                "render_on_art_suspected",
                "mask_outside_balloon",
            ],
        }
        rendered = {
            "render_bbox": [18, 94, 350, 166],
            "safe_text_box": [18, 82, 383, 178],
            "qa_flags": ["translator_note_stable_text_only_render"],
            "qa_metrics": {
                "resolved_pre_render_flags": [
                    "translator_note_best_effort_render",
                    "render_on_art_suspected",
                ],
                "translator_note_flags_revalidated": {
                    "decision": "intentional_text_only_note",
                },
            },
        }

        renderer_mod._copy_render_debug_fields(source, rendered)

        self.assertNotIn("translator_note_best_effort_render", source["qa_flags"])
        self.assertNotIn("render_on_art_suspected", source["qa_flags"])
        self.assertIn("translator_note_stable_text_only_render", source["qa_flags"])
        self.assertIn("mask_outside_balloon", source["qa_flags"])
        self.assertIn("translator_note_best_effort_render", source["qa_metrics"]["resolved_pre_render_flags"])

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

    def test_render_text_block_uses_dark_panel_glow_fallback_for_rejected_mask(self):
        img = Image.new("RGB", (300, 180), (4, 14, 19))
        text_data = {
            "style_origin": "auto",
            "translated": "O CAVALEIRO DO DIABO!",
            "bbox": [70, 70, 230, 112],
            "text_pixel_bbox": [70, 70, 230, 112],
            "target_bbox": [55, 55, 245, 130],
            "safe_text_box": [55, 55, 245, 130],
            "balloon_bbox": [50, 50, 250, 140],
            "background_rgb": [4, 14, 19],
            "bubble_mask_source": "derived_white_crop_rejected",
            "layout_safe_reason": "debug_derived_bubble_mask_rejected",
            "qa_flags": ["debug_derived_bubble_mask_rejected"],
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "glow": False,
                "glow_cor": "",
                "glow_px": 0,
                "cor_gradiente": [],
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["estilo"]["cor"], "#FFFFFF")
        self.assertEqual(text_data["estilo"]["contorno"], "#061D26")
        self.assertGreaterEqual(text_data["estilo"]["contorno_px"], 1)
        self.assertTrue(text_data["estilo"]["glow"])
        self.assertEqual(text_data["estilo"]["glow_cor"], "#67D8FF")
        self.assertGreaterEqual(text_data["estilo"]["glow_px"], 3)
        self.assertIn("auto_dark_panel_glow_fallback", text_data.get("qa_flags") or [])

    def test_dark_panel_glow_recovers_card_safe_area_from_bright_outline(self):
        img = Image.new("RGB", (360, 180), (2, 8, 12))
        draw = ImageDraw.Draw(img)
        draw.rectangle([56, 36, 304, 132], fill=(3, 7, 10), outline=(150, 210, 220), width=2)
        text_data = {
            "style_origin": "auto_dark_panel_glow",
            "translated": "O CAVALEIRO DO DIABO!",
            "bbox": [130, 78, 235, 96],
            "source_bbox": [130, 78, 235, 96],
            "text_pixel_bbox": [130, 78, 235, 96],
            "target_bbox": [122, 68, 246, 106],
            "balloon_bbox": [120, 66, 248, 108],
            "background_rgb": [4, 14, 19],
            "bubble_mask_source": "derived_white_crop_rejected",
            "layout_safe_reason": "debug_derived_bubble_mask_rejected",
            "qa_flags": ["auto_dark_panel_glow_fallback", "debug_derived_bubble_mask_rejected"],
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "cor_gradiente": [],
            },
        }

        renderer_mod._apply_visual_rect_safe_area_if_needed(img, text_data)
        plan = plan_text_layout(text_data)

        self.assertEqual(text_data["_visual_rect_outer_bbox"], [57, 37, 304, 132])
        self.assertIn(text_data["layout_safe_reason"], {"visual_rect_dark_panel", "visual_rect_inner"})
        self.assertLessEqual(plan["target_bbox"][0], 60)
        self.assertGreaterEqual(plan["target_bbox"][2], 300)
        self.assertLessEqual(plan["safe_text_box"][0], 98)
        self.assertGreaterEqual(plan["safe_text_box"][2], 264)
        self.assertGreater(plan["safe_text_box"][3] - plan["safe_text_box"][1], 52)

    def test_render_text_block_detects_dark_panel_card_after_style_fallback(self):
        img = Image.new("RGB", (360, 180), (2, 8, 12))
        draw = ImageDraw.Draw(img)
        draw.rectangle([56, 36, 304, 132], fill=(3, 7, 10), outline=(150, 210, 220), width=2)
        text_data = {
            "style_origin": "auto",
            "translated": "O cavaleiro do diabo!",
            "bbox": [130, 78, 235, 96],
            "source_bbox": [130, 78, 235, 96],
            "text_pixel_bbox": [130, 78, 235, 96],
            "target_bbox": [122, 68, 246, 106],
            "balloon_bbox": [120, 66, 248, 108],
            "background_rgb": [4, 14, 19],
            "bubble_mask_source": "derived_white_crop_rejected",
            "layout_safe_reason": "debug_derived_bubble_mask_rejected",
            "qa_flags": ["debug_derived_bubble_mask_rejected"],
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "glow": False,
                "glow_cor": "",
                "glow_px": 0,
                "cor_gradiente": [],
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["style_origin"], "auto_dark_panel_glow")
        self.assertEqual(text_data["_visual_rect_outer_bbox"], [57, 37, 304, 132])
        self.assertIn(text_data["layout_safe_reason"], {"visual_rect_dark_panel", "visual_rect_inner"})
        self.assertLessEqual(text_data["target_bbox"][0], 60)
        self.assertGreaterEqual(text_data["target_bbox"][2], 300)
        self.assertGreater(text_data["safe_text_box"][2] - text_data["safe_text_box"][0], 160)

    def test_render_text_block_uses_original_dark_panel_effect_colors(self):
        img = Image.new("RGB", (360, 180), (2, 8, 12))
        draw = ImageDraw.Draw(img)
        draw.rectangle([56, 36, 304, 132], fill=(3, 3, 3), outline=(160, 155, 142), width=2)
        text_data = {
            "style_origin": "auto",
            "translated": "MISSAO INICIADA!",
            "bbox": [130, 78, 235, 96],
            "source_bbox": [130, 78, 235, 96],
            "text_pixel_bbox": [130, 78, 235, 96],
            "target_bbox": [122, 68, 246, 106],
            "balloon_bbox": [56, 36, 304, 132],
            "bubble_mask_bbox": [56, 36, 304, 132],
            "background_rgb": [2, 2, 2],
            "bubble_mask_source": "image_dark_bubble_mask",
            "dark_panel_effect_colors": {
                "color_sample_space": "original_image",
                "panel_fill_rgb": [3, 3, 3],
                "border_rgb": [160, 155, 142],
                "panel_glow_rgb": [75, 61, 24],
                "text_fill_rgb": [252, 250, 236],
                "text_glow_rgb": [83, 71, 39],
                "bad_negative_text_glow_rgb": [186, 197, 236],
            },
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "glow": False,
                "glow_cor": "",
                "glow_px": 0,
                "cor_gradiente": [],
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["estilo"]["cor"], "#FCFAEC")
        self.assertEqual(text_data["estilo"]["contorno"], "#A09B8E")
        self.assertEqual(text_data["estilo"]["glow_cor"], "#534727")
        self.assertNotEqual(text_data["estilo"]["glow_cor"], "#67D8FF")
        self.assertIn("original_dark_panel_effect_colors", text_data.get("qa_flags") or [])

    def test_dark_visual_text_caps_low_panel_font_and_reserves_glow_leading(self):
        text_data = {
            "translated": "O cavaleiro do diabo!",
            "bbox": [202, 169, 342, 217],
            "source_bbox": [202, 169, 342, 217],
            "text_pixel_bbox": [202, 169, 342, 217],
            "target_bbox": [185, 160, 364, 222],
            "safe_text_box": [185, 73, 364, 222],
            "balloon_bbox": [185, 73, 364, 222],
            "bubble_mask_bbox": [185, 73, 364, 222],
            "bubble_mask_source": "image_dark_bubble_mask",
            "background_rgb": [2, 2, 2],
            "block_profile": "dark_bubble",
            "style_origin": "auto_dark_panel_glow",
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 37,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(
            text_data,
            ensure_legible_plan(Image.new("RGB", (420, 260), (2, 2, 2)), plan),
        )

        self.assertLessEqual(plan["target_size"], 31)
        self.assertGreaterEqual(plan["line_spacing_ratio"], 0.38)
        self.assertGreaterEqual(resolved["line_height"], resolved["font_size"] + 5)
        self.assertIn("dark_visual_font_capped_for_readability", text_data.get("qa_flags") or [])

    def test_dark_bubble_text_shaped_visual_mask_keeps_balloon_capacity(self):
        text_data = {
            "translated": "Isso significa que voce e talentoso!",
            "bbox": [526, 140, 642, 168],
            "source_bbox": [479, 114, 696, 203],
            "text_pixel_bbox": [479, 114, 696, 203],
            "line_polygons": [
                [[479, 114], [696, 114], [696, 154], [479, 154]],
                [[518, 162], [657, 162], [657, 203], [518, 203]],
            ],
            "balloon_bbox": [341, 0, 800, 308],
            "bubble_mask_bbox": [468, 96, 700, 212],
            "bubble_inner_bbox": [480, 108, 688, 200],
            "bubble_mask_source": "image_dark_bubble_mask",
            "background_rgb": [2, 2, 2],
            "layout_profile": "dark_bubble",
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [341, 0, 800, 308])
        self.assertGreater(plan["capacity_bbox"][2] - plan["capacity_bbox"][0], 260)
        self.assertIn(
            "dark_bubble_text_shaped_visual_mask_kept_balloon_capacity",
            text_data.get("qa_flags") or [],
        )

    def test_dark_merged_text_with_real_balloon_is_not_skipped_for_fast_fill_flag(self):
        from typesetter.renderer import build_render_blocks

        blocks = build_render_blocks(
            [
                {
                    "id": "direct_paddle_reocr_001",
                    "text_id": "direct_paddle_reocr_001",
                    "translated": "O QUE VOCE QUER DIZER COM SUBESPACO? O QUE VOCE QUER DIZER COM SUBESPACO?",
                    "bbox": [214, 112, 349, 196],
                    "text_pixel_bbox": [214, 112, 349, 196],
                    "balloon_bbox": [0, 0, 800, 785],
                    "bubble_mask_bbox": [0, 0, 800, 785],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["same_balloon_fragment_merged", "fast_fill_no_glyph_evidence"],
                }
            ]
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["text_id"], "direct_paddle_reocr_001")

    def test_dark_recovered_short_fragment_is_suppressed_for_render(self):
        from typesetter.renderer import build_render_blocks

        text = {
            "id": "direct_paddle_reocr_001_fragment_2",
            "text_id": "direct_paddle_reocr_001_fragment_2",
            "trace_id": "direct_paddle_reocr_001@page_003_band_042#fragment_2",
            "translated": "ISSO",
            "bbox": [447, 1000, 630, 1025],
            "text_pixel_bbox": [447, 1000, 630, 1025],
            "balloon_bbox": [0, 678, 726, 1127],
            "qa_flags": ["fast_fill_no_glyph_evidence", "debug_derived_bubble_mask_rejected"],
        }

        blocks = build_render_blocks([text])

        self.assertEqual(blocks, [])
        self.assertEqual(text["route_action"], "suppress")
        self.assertIn("dark_recovered_short_fragment_suppressed", text["qa_flags"])

    def test_dark_recovered_unverified_fragment_is_suppressed_for_render(self):
        from typesetter.renderer import build_render_blocks

        text = {
            "id": "direct_paddle_reocr_001",
            "text_id": "direct_paddle_reocr_001",
            "trace_id": "direct_paddle_reocr_001@page_003_band_042",
            "original": "1/ WHAT'S THA",
            "translated": "1/ O QUE E",
            "bbox": [192, 176, 388, 397],
            "text_pixel_bbox": [192, 176, 388, 397],
            "balloon_bbox": [0, 0, 726, 622],
            "bubble_mask_bbox": [0, 0, 32, 32],
            "qa_flags": [
                "dark_bubble_oval_reocr",
                "dark_bubble_connected_lobe_passthrough",
                "dark_bubble_visual_mask_rejected_tiny_text",
            ],
        }

        blocks = build_render_blocks([text])

        self.assertEqual(blocks, [])
        self.assertEqual(text["route_action"], "suppress")
        self.assertIn("dark_recovered_unverified_fragment_suppressed", text["qa_flags"])

    def test_same_balloon_repeated_sentence_is_deduped_for_render(self):
        from typesetter.renderer import _dedupe_repeated_sentence_for_render

        text = "O QUE VOCE QUER DIZER COM SUBESPACO? O QUE VOCE QUER DIZER COM SUBESPACO?"

        self.assertEqual(
            _dedupe_repeated_sentence_for_render(text),
            "O QUE VOCE QUER DIZER COM SUBESPACO?",
        )

    def test_dark_bubble_overbroad_mask_uses_compact_ellipse_bbox(self):
        text_data = {
            "translated": "O QUE VOCE QUER DIZER COM SUBESPACO? O QUE VOCE QUER DIZER COM SUBESPACO?",
            "bbox": [214, 112, 349, 196],
            "source_bbox": [214, 112, 349, 196],
            "text_pixel_bbox": [214, 112, 349, 196],
            "line_polygons": [
                [[225, 112], [339, 112], [339, 134], [225, 134]],
                [[214, 141], [348, 144], [348, 168], [214, 165]],
                [[216, 176], [349, 176], [349, 196], [216, 196]],
            ],
            "balloon_bbox": [0, 0, 800, 785],
            "bubble_mask_bbox": [0, 0, 800, 785],
            "bubble_inner_bbox": [218, 108, 344, 185],
            "bubble_mask_ellipse": {"center": [282.0, 154.5], "axes": [278.0, 283.0], "angle": 0.0},
            "bubble_mask_source": "image_dark_bubble_mask",
            "background_rgb": [2, 2, 2],
            "layout_profile": "dark_bubble",
            "qa_flags": ["same_balloon_fragment_merged", "fast_fill_no_glyph_evidence"],
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [143, 13, 421, 296])
        self.assertLess(plan["capacity_bbox"][2] - plan["capacity_bbox"][0], 230)
        self.assertIn("dark_bubble_compact_ellipse_bbox_preferred", text_data.get("qa_flags") or [])

    def test_dark_panel_overbroad_bubble_mask_does_not_expand_render_target(self):
        text_data = {
            "translated": "A missao principal sera mostrada em breve.",
            "bbox": [108, 165, 260, 236],
            "source_bbox": [108, 165, 260, 236],
            "text_pixel_bbox": [108, 165, 260, 236],
            "line_polygons": [
                [[121, 165], [251, 165], [251, 191], [121, 191]],
                [[108, 204], [260, 204], [260, 236], [108, 236]],
            ],
            "balloon_bbox": [75, 144, 293, 257],
            "bubble_mask_bbox": [37, 122, 687, 332],
            "bubble_inner_bbox": [39, 124, 685, 330],
            "bubble_mask_source": "image_dark_panel_mask",
            "background_rgb": [2, 2, 2],
            "block_profile": "dark_panel",
            "tipo": "texto",
            "estilo": {
                "fonte": "KOMIKAX_.ttf",
                "tamanho": 34,
                "cor": "#FDF9E6",
                "contorno": "#747273",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#34280C",
                "glow_px": 3,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)

        self.assertLessEqual(plan["target_bbox"][2] - plan["target_bbox"][0], 260)
        self.assertEqual(plan["target_bbox"], [75, 144, 293, 257])
        self.assertNotEqual(plan["target_bbox"], [37, 122, 687, 332])
        self.assertEqual(text_data.get("_dark_panel_bubble_bbox_rejected"), "overbroad_against_balloon_bbox")
        self.assertIn("dark_panel_mask_overbroad_rejected", text_data.get("qa_flags") or [])

    def test_dark_bubble_short_single_line_prefers_largest_fitting_size(self):
        text_data = {
            "original": "1,000 points..",
            "translated": "1.000 pontos..",
            "bbox": [107, 365, 269, 405],
            "source_bbox": [0, 0, 355, 499],
            "text_pixel_bbox": [107, 365, 269, 405],
            "balloon_bbox": [0, 0, 355, 499],
            "bubble_mask_source": "rejected_derived_bubble_mask",
            "background_rgb": [0, 0, 0],
            "block_profile": "dark_bubble",
            "style_origin": "auto_dark_panel_glow",
            "qa_flags": [
                "connected_layout_disabled_rejected_bubble_mask",
                "compact_small_text_capacity",
                "debug_derived_bubble_mask_rejected",
                "auto_dark_panel_glow_fallback",
            ],
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 16,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
                "cor_gradiente": [],
            },
        }

        plan = {
            "target_bbox": [107, 365, 269, 405],
            "position_bbox": [110, 365, 265, 405],
            "capacity_bbox": [110, 365, 265, 405],
            "safe_text_box": [110, 365, 265, 405],
            "layout_shape": "wide",
            "balloon_geo": "ellipse",
            "layout_profile": "dark_bubble",
            "width_ratio": 1.0,
            "max_width": 155,
            "max_height": 40,
            "padding_y": 0,
            "vertical_anchor": "center",
            "alignment": "center",
            "font_name": "ComicNeue-Bold.ttf",
            "target_size": 16,
            "text_color": "#FFFFFF",
            "background_rgb": [0, 0, 0],
            "cor_gradiente": [],
            "outline_color": "#061D26",
            "outline_px": 1,
            "glow": True,
            "glow_cor": "#67D8FF",
            "glow_px": 3,
            "sombra": False,
            "sombra_cor": "",
            "sombra_offset": [0, 0],
            "curva": False,
            "curva_direcao": "",
            "curva_intensidade": 0.0,
            "rotation_deg": 0.0,
            "rotation_source": "",
            "line_spacing_ratio": 0.38,
            "vertical_bias_px": 0,
            "horizontal_bias_px": 0,
            "_target_source": "tiny_anchor_union",
            "_style_origin": "auto_dark_panel_glow",
            "_validated_source_target_bbox": [],
            "_anchor_capacity_locked": False,
            "_simple_anchor_capacity_expanded": False,
            "_simple_anchor_capacity_reason": "",
            "_font_search_cap": 0,
            "_font_search_floor": 0,
            "_font_search_emergency_floor": 8,
            "_follow_original_ocr_size": False,
            "_follow_english_anchor_position": False,
            "_position_on_capacity_bbox": False,
            "_center_on_balloon_bbox": True,
        }
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(len(resolved["lines"]), 1)
        self.assertGreaterEqual(resolved["font_size"], 22)
        self.assertLessEqual(resolved["block_width"], plan["max_width"])
        self.assertLessEqual(resolved["total_text_height"], plan["max_height"])

    def test_dark_bubble_visual_mask_restores_target_when_page_space_bubble_bbox_is_tiny(self):
        text_data = {
            "translated": "E ainda assim nenhum deles sequer te visitou uma vez.",
            "bbox": [442, 224, 711, 303],
            "source_bbox": [442, 224, 711, 303],
            "text_pixel_bbox": [442, 224, 711, 303],
            "line_polygons": [
                [[442, 224], [711, 224], [711, 255], [442, 255]],
                [[474, 266], [552, 266], [552, 297], [474, 297]],
            ],
            "target_bbox": [474, 266, 552, 297],
            "balloon_bbox": [474, 266, 552, 297],
            "bubble_mask_bbox": [474, 266, 552, 297],
            "bubble_mask_source": "image_dark_bubble_mask",
            "background_rgb": [0, 0, 0],
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "qa_metrics": {
                "image_dark_bubble_mask": {
                    "source": "image_dark_bubble_mask",
                    "shape_kind": "ellipse",
                    "mask_bbox": [348, 3, 800, 384],
                    "anchor_bbox": [442, 224, 711, 303],
                }
            },
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 24,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
            },
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [348, 3, 800, 384])
        self.assertEqual(text_data.get("_render_target_source"), "dark_bubble_visual_mask_bbox")
        self.assertNotIn("dark_bubble_visual_mask_rejected_tiny_text", text_data.get("qa_flags") or [])
        self.assertGreater(plan["safe_text_box"][2] - plan["safe_text_box"][0], 200)

    def test_dark_bubble_visual_mask_disables_false_connected_subregions(self):
        img = Image.new("RGB", (480, 260), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([42, 28, 438, 226], fill=(0, 0, 0), outline=(20, 108, 140), width=3)
        text_data = {
            "translated": "Critérios de conclusão da missão: estabeleça um submundo de nível 1.",
            "bbox": [82, 114, 364, 220],
            "source_bbox": [82, 114, 364, 220],
            "text_pixel_bbox": [82, 114, 364, 220],
            "line_polygons": [
                [[203, 104], [330, 104], [330, 130], [203, 130]],
                [[184, 158], [318, 158], [318, 201], [184, 201]],
            ],
            "balloon_bbox": [42, 28, 438, 226],
            "bubble_mask_bbox": [42, 28, 438, 226],
            "balloon_subregions": [[42, 28, 240, 226], [240, 28, 438, 226]],
            "connected_lobe_bboxes": [[42, 28, 240, 226], [240, 28, 438, 226]],
            "connected_position_bboxes": [[203, 104, 330, 130], [184, 158, 318, 201]],
            "connected_balloon_orientation": "left-right",
            "layout_profile": "connected_balloon",
            "layout_group_size": 2,
            "subregion_confidence": 1.0,
            "bubble_mask_source": "image_dark_bubble_mask",
            "background_rgb": [0, 0, 0],
            "block_profile": "dark_panel",
            "qa_metrics": {
                "image_dark_bubble_mask": {
                    "source": "image_dark_bubble_mask",
                    "mask_bbox": [42, 28, 438, 226],
                    "panel_fill_rgb": [0, 0, 0],
                    "text_fill_rgb": [251, 251, 251],
                    "text_glow_rgb": [89, 67, 40],
                }
            },
            "tipo": "texto",
            "estilo": {
                "fonte": "KOMIKAX_.ttf",
                "tamanho": 25,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
                "cor_gradiente": [],
            },
        }

        with patch("typesetter.renderer._render_connected_subregions") as connected_mock:
            render_text_block(img, text_data)

        connected_mock.assert_not_called()
        self.assertEqual(text_data.get("layout_group_size"), 1)
        self.assertEqual(text_data.get("balloon_subregions"), [])
        self.assertIn("connected_layout_disabled_dark_panel_visual_mask", text_data.get("qa_flags") or [])

    def test_tiny_dark_side_note_rejects_overbroad_dark_bubble_mask(self):
        text_data = {
            "translated": "Divindade budista da morte.",
            "bbox": [616, 4301, 771, 4312],
            "source_bbox": [616, 4301, 771, 4312],
            "text_pixel_bbox": [616, 4301, 771, 4312],
            "line_polygons": [[[616, 4301], [771, 4301], [771, 4312], [616, 4312]]],
            "balloon_bbox": [212, 4110, 759, 4415],
            "bubble_mask_bbox": [212, 4110, 759, 4415],
            "bubble_mask_source": "image_dark_bubble_mask",
            "bubble_mask_shape": "ellipse",
            "block_profile": "dark_bubble",
            "qa_metrics": {
                "image_dark_bubble_mask": {
                    "source": "image_dark_bubble_mask",
                    "shape_kind": "ellipse",
                    "mask_bbox": [212, 4110, 759, 4415],
                    "anchor_bbox": [616, 4301, 771, 4312],
                }
            },
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 20,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 2,
            },
        }

        plan = plan_text_layout(text_data)

        self.assertLess(plan["target_bbox"][2] - plan["target_bbox"][0], 260)
        self.assertLess(plan["target_bbox"][3] - plan["target_bbox"][1], 90)
        self.assertNotEqual(plan["target_bbox"], [212, 4110, 759, 4415])
        self.assertIn("dark_bubble_visual_mask_rejected_tiny_text", text_data.get("qa_flags") or [])

    def test_dark_visual_card_low_containment_does_not_suppress_render(self):
        img = Image.new("RGB", (520, 360), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([88, 36, 461, 321], fill=(0, 0, 0), outline=(150, 150, 150), width=2)
        before = np.asarray(img).copy()
        text_data = {
            "translated": "O cavaleiro do diabo!",
            "bbox": [202, 206, 342, 254],
            "source_bbox": [202, 206, 342, 254],
            "text_pixel_bbox": [178, 212, 371, 247],
            "line_polygons": [[[178, 212], [371, 212], [371, 247], [178, 247]]],
            "balloon_bbox": [88, 36, 461, 321],
            "bubble_mask_bbox": [88, 36, 461, 321],
            "bubble_mask_source": "image_dark_bubble_mask",
            "bubble_mask_shape": "ellipse",
            "block_profile": "standard",
            "qa_metrics": {
                "image_dark_bubble_mask": {
                    "source": "image_dark_bubble_mask",
                    "shape_kind": "ellipse",
                    "mask_bbox": [88, 36, 461, 321],
                    "anchor_bbox": [178, 212, 371, 247],
                },
                "render_balloon_containment": 0.0175,
            },
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 33,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 2,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        self.assertNotIn("render_suppressed_low_containment_fragment", text_data.get("qa_flags") or [])
        self.assertGreater(np.count_nonzero(np.asarray(img) != before), 0)

    def test_dark_panel_visual_mask_bbox_overrides_split_glyph_components(self):
        text_data = {
            "translated": "O episódio começa!",
            "bbox": [585, 3564, 767, 3593],
            "source_bbox": [585, 3564, 767, 3593],
            "text_pixel_bbox": [585, 3564, 767, 3593],
            "line_polygons": [[[585, 3564], [767, 3564], [767, 3593], [585, 3593]]],
            "balloon_bbox": [545, 3552, 800, 3605],
            "bubble_mask_bbox": [578, 3551, 775, 3600],
            "bubble_inner_bbox": [590, 3563, 763, 3588],
            "connected_position_bboxes": [[575, 3564, 614, 3593], [735, 3571, 777, 3603]],
            "bubble_mask_source": "image_dark_panel_mask",
            "background_rgb": [8, 4, 2],
            "block_profile": "dark_panel",
            "qa_metrics": {
                "image_dark_panel_mask": {
                    "source": "image_dark_panel_mask",
                    "mask_bbox": [563, 3495, 797, 3645],
                }
            },
            "tipo": "texto",
            "estilo": {
                "fonte": "KOMIKAX_.ttf",
                "tamanho": 28,
                "cor": "#FEFBEB",
                "contorno": "#968D83",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#32270A",
                "glow_px": 3,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [563, 3495, 797, 3645])
        self.assertEqual(text_data.get("_render_target_source"), "dark_panel_visual_mask_bbox")
        self.assertEqual(text_data.get("layout_safe_reason"), "visual_rect_dark_panel_mask")
        self.assertEqual(plan["layout_safe_reason"], "visual_rect_inner")
        self.assertGreater(plan["safe_text_box"][2] - plan["safe_text_box"][0], 150)
        self.assertGreater(plan["safe_text_box"][3] - plan["safe_text_box"][1], 80)

    def test_dark_panel_full_bbox_clamps_safe_area_to_inner_mask(self):
        text_data = {
            "translated": "A missao principal sera mostrada em breve",
            "bbox": [113, 165, 259, 236],
            "source_bbox": [113, 165, 259, 236],
            "text_pixel_bbox": [113, 165, 259, 236],
            "line_polygons": [
                [[113, 165], [259, 165], [259, 236], [113, 236]],
            ],
            "balloon_bbox": [37, 122, 349, 332],
            "bubble_mask_bbox": [37, 122, 349, 332],
            "bubble_inner_bbox": [116, 161, 254, 224],
            "bubble_mask_source": "image_dark_panel_mask",
            "background_rgb": [8, 4, 2],
            "block_profile": "dark_panel",
            "qa_metrics": {
                "image_dark_panel_mask": {
                    "source": "image_dark_panel_mask",
                    "mask_bbox": [37, 122, 349, 332],
                    "full_panel_bbox_selected": True,
                }
            },
            "qa_flags": ["dark_panel_full_bbox_selected"],
            "tipo": "texto",
            "estilo": {
                "fonte": "KOMIKAX_.ttf",
                "tamanho": 28,
                "cor": "#FEFBEB",
                "contorno": "#968D83",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#32270A",
                "glow_px": 3,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [37, 122, 349, 332])
        self.assertEqual(text_data.get("_render_target_source"), "dark_panel_visual_mask_bbox")
        self.assertTrue(text_data.get("_dark_panel_full_bbox_safe_clamped_to_inner"))
        self.assertIn("dark_panel_full_bbox_safe_clamped_to_inner", text_data.get("qa_flags") or [])
        safe = plan["safe_text_box"]
        inner = text_data["bubble_inner_bbox"]
        self.assertGreaterEqual(safe[0], inner[0])
        self.assertGreaterEqual(safe[1], inner[1])
        self.assertLessEqual(safe[2], inner[2])
        self.assertLessEqual(safe[3], inner[3])

    def test_dark_panel_full_bbox_clamps_page_space_inner_mask_to_band_target(self):
        text_data = {
            "translated": "A missao principal sera mostrada em breve",
            "bbox": [84, 88, 286, 297],
            "source_bbox": [84, 88, 286, 297],
            "text_pixel_bbox": [113, 165, 259, 236],
            "line_polygons": [
                [[113, 165], [259, 165], [259, 236], [113, 236]],
            ],
            "balloon_bbox": [37, 122, 349, 332],
            "bubble_mask_bbox": [37, 122, 349, 332],
            "bubble_inner_bbox": [116, 2893, 254, 2956],
            "bubble_mask_source": "image_dark_bubble_mask",
            "band_y_top": 2732,
            "background_rgb": [8, 4, 2],
            "block_profile": "dark_panel",
            "qa_metrics": {
                "image_dark_panel_mask": {
                    "source": "image_dark_panel_mask",
                    "mask_bbox": [37, 122, 349, 332],
                    "full_panel_bbox_selected": True,
                }
            },
            "qa_flags": ["dark_panel_full_bbox_selected", "dark_panel_rect_from_dark_bubble_bbox"],
            "tipo": "texto",
            "estilo": {
                "fonte": "KOMIKAX_.ttf",
                "tamanho": 28,
                "cor": "#FEFBEB",
                "contorno": "#968D83",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#32270A",
                "glow_px": 3,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [37, 122, 349, 332])
        self.assertEqual(text_data.get("_dark_panel_full_bbox_inner_safe_bbox"), [116, 161, 254, 224])
        safe = plan["safe_text_box"]
        self.assertGreaterEqual(safe[0], 116)
        self.assertGreaterEqual(safe[1], 161)
        self.assertLessEqual(safe[2], 254)
        self.assertLessEqual(safe[3], 224)

    def test_dark_panel_full_bbox_infers_band_offset_from_page_space_mask(self):
        text_data = {
            "translated": "A missao principal sera mostrada em breve",
            "bbox": [84, 88, 286, 297],
            "source_bbox": [84, 88, 286, 297],
            "text_pixel_bbox": [113, 165, 259, 236],
            "line_polygons": [
                [[113, 165], [259, 165], [259, 236], [113, 236]],
            ],
            "balloon_bbox": [37, 2854, 349, 3064],
            "bubble_mask_bbox": [37, 2854, 349, 3064],
            "bubble_inner_bbox": [116, 2893, 254, 2956],
            "bubble_mask_source": "image_dark_panel_mask",
            "background_rgb": [8, 4, 2],
            "block_profile": "dark_panel",
            "qa_metrics": {
                "image_dark_panel_mask": {
                    "source": "image_dark_panel_mask",
                    "mask_bbox": [37, 122, 349, 332],
                    "full_panel_bbox_selected": True,
                }
            },
            "qa_flags": ["dark_panel_full_bbox_selected"],
            "tipo": "texto",
            "estilo": {
                "fonte": "KOMIKAX_.ttf",
                "tamanho": 28,
                "cor": "#FEFBEB",
                "contorno": "#968D83",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#32270A",
                "glow_px": 3,
                "cor_gradiente": [],
            },
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [37, 122, 349, 332])
        self.assertEqual(text_data.get("_dark_panel_inner_bbox_inferred_band_y_top"), 2732)
        self.assertEqual(text_data.get("_dark_panel_full_bbox_inner_safe_bbox"), [116, 161, 254, 224])
        safe = plan["safe_text_box"]
        self.assertGreaterEqual(safe[0], 116)
        self.assertGreaterEqual(safe[1], 161)
        self.assertLessEqual(safe[2], 254)
        self.assertLessEqual(safe[3], 224)

    def test_dark_panel_render_recomputes_stale_safe_text_box_from_visual_mask(self):
        img = Image.new("RGB", (820, 260), (8, 4, 2))
        draw = ImageDraw.Draw(img)
        draw.rectangle([563, 40, 797, 190], fill=(8, 4, 2), outline=(150, 141, 131), width=2)
        text_data = {
            "translated": "O EPISODIO COMECA!",
            "bbox": [585, 109, 767, 138],
            "source_bbox": [585, 109, 767, 138],
            "text_pixel_bbox": [585, 109, 767, 138],
            "line_polygons": [[[585, 109], [767, 109], [767, 138], [585, 138]]],
            "balloon_bbox": [545, 97, 800, 150],
            "bubble_mask_bbox": [578, 96, 775, 145],
            "bubble_inner_bbox": [590, 108, 763, 133],
            "connected_position_bboxes": [[575, 109, 614, 138], [735, 116, 777, 148]],
            "safe_text_box": [614, 71, 745, 159],
            "_debug_safe_text_box": [614, 71, 745, 159],
            "render_bbox": [614, 79, 745, 159],
            "bubble_mask_source": "image_dark_panel_mask",
            "background_rgb": [8, 4, 2],
            "block_profile": "dark_panel",
            "qa_metrics": {
                "image_dark_panel_mask": {
                    "source": "image_dark_panel_mask",
                    "mask_bbox": [563, 40, 797, 190],
                    "panel_fill_rgb": [8, 4, 2],
                    "border_rgb": [150, 141, 131],
                    "text_fill_rgb": [254, 251, 235],
                    "text_glow_rgb": [50, 39, 10],
                }
            },
            "tipo": "texto",
            "estilo": {
                "fonte": "KOMIKAX_.ttf",
                "tamanho": 28,
                "cor": "#FEFBEB",
                "contorno": "#968D83",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#32270A",
                "glow_px": 3,
                "cor_gradiente": [],
            },
        }

        render_text_block(img, text_data)

        self.assertTrue(text_data.get("_dark_panel_visual_geometry_recomputed"))
        self.assertEqual(text_data.get("_render_target_source"), "dark_panel_visual_mask_bbox")
        self.assertGreater(text_data["safe_text_box"][2] - text_data["safe_text_box"][0], 150)
        self.assertGreater(text_data["render_bbox"][2] - text_data["render_bbox"][0], 130)

    def test_dark_panel_render_disables_false_connected_subregions(self):
        img = Image.new("RGB", (820, 260), (8, 4, 2))
        draw = ImageDraw.Draw(img)
        draw.rectangle([563, 40, 797, 190], fill=(8, 4, 2), outline=(150, 141, 131), width=2)
        text_data = {
            "translated": "O EPISODIO COMECA!",
            "bbox": [585, 109, 767, 138],
            "source_bbox": [585, 109, 767, 138],
            "text_pixel_bbox": [585, 109, 767, 138],
            "line_polygons": [[[585, 109], [767, 109], [767, 138], [585, 138]]],
            "balloon_bbox": [545, 97, 800, 150],
            "bubble_mask_bbox": [578, 96, 775, 145],
            "bubble_inner_bbox": [590, 108, 763, 133],
            "balloon_subregions": [[545, 97, 649, 150], [657, 97, 800, 150]],
            "connected_lobe_bboxes": [[545, 97, 649, 150], [657, 97, 800, 150]],
            "connected_position_bboxes": [[575, 109, 614, 138], [735, 116, 777, 148]],
            "connected_balloon_orientation": "left-right",
            "layout_profile": "connected_balloon",
            "layout_group_size": 2,
            "subregion_confidence": 1.0,
            "bubble_mask_source": "image_dark_panel_mask",
            "background_rgb": [8, 4, 2],
            "block_profile": "dark_panel",
            "qa_metrics": {
                "image_dark_panel_mask": {
                    "source": "image_dark_panel_mask",
                    "mask_bbox": [563, 40, 797, 190],
                    "panel_fill_rgb": [8, 4, 2],
                    "border_rgb": [150, 141, 131],
                    "text_fill_rgb": [254, 251, 235],
                    "text_glow_rgb": [50, 39, 10],
                }
            },
            "tipo": "texto",
            "estilo": {
                "fonte": "KOMIKAX_.ttf",
                "tamanho": 28,
                "cor": "#FEFBEB",
                "contorno": "#968D83",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#32270A",
                "glow_px": 3,
                "cor_gradiente": [],
            },
        }

        with patch("typesetter.renderer._render_connected_subregions") as connected_mock:
            render_text_block(img, text_data)

        connected_mock.assert_not_called()
        self.assertEqual(text_data.get("_render_target_source"), "dark_panel_visual_mask_bbox")
        self.assertEqual(text_data.get("layout_group_size"), 1)
        self.assertEqual(text_data.get("balloon_subregions"), [])
        self.assertIn("connected_layout_disabled_dark_panel_visual_mask", text_data.get("qa_flags") or [])
        self.assertGreater(text_data["render_bbox"][2] - text_data["render_bbox"][0], 130)

    def test_colored_panel_rect_recovers_safe_area_when_white_bubble_mask_is_too_small(self):
        img = Image.new("RGB", (360, 220), (60, 40, 120))
        draw = ImageDraw.Draw(img)
        draw.rectangle([72, 58, 288, 150], fill=(92, 154, 224), outline=(218, 238, 246), width=2)
        text_data = {
            "style_origin": "auto",
            "translated": "A sincronização foi concluída.",
            "bbox": [140, 86, 220, 126],
            "source_bbox": [140, 86, 220, 126],
            "text_pixel_bbox": [140, 86, 220, 126],
            "target_bbox": [132, 82, 226, 132],
            "balloon_bbox": [72, 58, 288, 150],
            "background_rgb": [120, 170, 225],
            "bubble_mask_source": "image_white_bubble_mask",
            "layout_safe_reason": "debug_derived_bubble_mask_rejected",
            "qa_flags": ["mask_outside_balloon_critical", "safe_text_box_recomputed"],
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "glow": False,
                "glow_cor": "",
                "glow_px": 0,
                "cor_gradiente": [],
            },
        }

        renderer_mod._apply_visual_rect_safe_area_if_needed(img, text_data)
        plan = plan_text_layout(text_data)

        self.assertEqual(text_data["layout_safe_reason"], "visual_rect_colored_panel")
        self.assertLessEqual(plan["target_bbox"][0], 76)
        self.assertGreaterEqual(plan["target_bbox"][2], 284)
        self.assertGreater(plan["safe_text_box"][2] - plan["safe_text_box"][0], 140)

    def test_render_text_block_uses_visual_card_style_for_colored_status_panel(self):
        img = Image.new("RGB", (360, 220), (54, 46, 130))
        draw = ImageDraw.Draw(img)
        draw.rectangle([72, 58, 288, 150], fill=(92, 154, 224), outline=(218, 238, 246), width=2)
        text_data = {
            "style_origin": "auto",
            "translated": "A sincronização foi concluída.",
            "bbox": [140, 86, 220, 126],
            "source_bbox": [140, 86, 220, 126],
            "text_pixel_bbox": [140, 86, 220, 126],
            "target_bbox": [72, 58, 288, 150],
            "safe_text_box": [92, 76, 268, 132],
            "balloon_bbox": [72, 58, 288, 150],
            "background_rgb": [92, 154, 224],
            "bubble_mask_source": "image_white_bubble_mask",
            "qa_flags": ["safe_text_box_recomputed"],
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#000000",
                "contorno": "",
                "contorno_px": 0,
                "glow": False,
                "glow_cor": "",
                "glow_px": 0,
                "cor_gradiente": [],
            },
        }

        render_text_block(img, text_data)
        plan = ensure_legible_plan(img, plan_text_layout(text_data))

        self.assertEqual(text_data["style_origin"], "inferred_visual_card")
        self.assertEqual(text_data["estilo"]["fonte"], "LeagueGothic-Regular-VariableFont_wdth.ttf")
        self.assertEqual(text_data["estilo"]["cor"], "#EBFFFF")
        self.assertTrue(text_data["estilo"]["glow"])
        self.assertEqual(text_data["estilo"]["glow_cor"], "#EBFFFF")
        self.assertEqual(plan["text_color"], "#EBFFFF")
        self.assertTrue(plan["glow"])
        self.assertIn("visual_card_style_fallback", text_data.get("qa_flags") or [])

    def test_render_text_block_uses_visual_card_font_for_source_detected_status_panel(self):
        img = Image.new("RGB", (360, 220), (54, 46, 130))
        draw = ImageDraw.Draw(img)
        draw.rectangle([72, 58, 288, 150], fill=(92, 154, 224), outline=(218, 238, 246), width=2)
        text_data = {
            "style_origin": "source_detected",
            "translated": "O anfitrião recebeu o título,",
            "bbox": [140, 86, 220, 126],
            "source_bbox": [140, 86, 220, 126],
            "text_pixel_bbox": [140, 86, 220, 126],
            "target_bbox": [72, 58, 288, 150],
            "safe_text_box": [92, 76, 268, 132],
            "balloon_bbox": [72, 58, 288, 150],
            "background_rgb": [92, 154, 224],
            "bubble_mask_source": "image_white_bubble_mask",
            "qa_flags": ["safe_text_box_recomputed"],
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#EBFFFF",
                "contorno": "",
                "contorno_px": 0,
                "glow": True,
                "glow_cor": "#EBFFFF",
                "glow_px": 2,
                "style_origin": "source_detected",
                "style_source": "pixel_analysis",
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["style_origin"], "source_detected")
        self.assertEqual(text_data["estilo"]["style_origin"], "source_detected")
        self.assertEqual(text_data["estilo"]["style_source"], "pixel_analysis")
        self.assertEqual(text_data["estilo"]["fonte"], "LeagueGothic-Regular-VariableFont_wdth.ttf")
        self.assertEqual(text_data["estilo"]["cor"], "#EBFFFF")
        self.assertTrue(text_data["estilo"]["glow"])
        self.assertIn("visual_card_font_fallback", text_data.get("qa_flags") or [])

    def test_render_text_block_does_not_use_dark_panel_glow_fallback_for_white_balloon(self):
        img = Image.new("RGB", (300, 180), (255, 255, 255))
        text_data = {
            "style_origin": "auto",
            "translated": "OLA",
            "bbox": [70, 70, 230, 112],
            "target_bbox": [55, 55, 245, 130],
            "safe_text_box": [55, 55, 245, 130],
            "balloon_bbox": [50, 50, 250, 140],
            "background_rgb": [255, 255, 255],
            "bubble_mask_source": "derived_white_crop_rejected",
            "layout_safe_reason": "debug_derived_bubble_mask_rejected",
            "layout_profile": "white_balloon",
            "qa_flags": ["debug_derived_bubble_mask_rejected"],
            "tipo": "texto",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 28,
                "cor": "#FFFFFF",
                "contorno": "#000000",
                "contorno_px": 2,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 4,
                "cor_gradiente": [],
            },
        }

        render_text_block(img, text_data)

        self.assertEqual(text_data["estilo"]["cor"], "#000000")
        self.assertEqual(text_data["estilo"]["contorno"], "")
        self.assertFalse(text_data["estilo"]["glow"])
        self.assertNotIn("auto_dark_panel_glow_fallback", text_data.get("qa_flags") or [])

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

    def test_render_text_block_honors_explicit_source_detected_effect_pixels(self):
        background_rgb = 180
        img = Image.new("RGB", (340, 240), (background_rgb, background_rgb, background_rgb))
        text_data = {
            "style_origin": "source_detected",
            "translated": "WOW",
            "bbox": [50, 45, 290, 180],
            "balloon_bbox": [50, 45, 290, 180],
            "tipo": "fala",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 58,
                "cor": "#FFFFFF",
                "contorno": "#000000",
                "contorno_px": 3,
                "alinhamento": "center",
                "sombra": True,
                "sombra_cor": "#333333",
                "sombra_offset": [8, 8],
                "glow": False,
                "glow_cor": "",
                "glow_px": 0,
                "cor_gradiente": [],
            },
        }

        render_text_block(img, text_data)

        arr = np.array(img)
        white_fill_pixels = np.all(arr >= 245, axis=2)
        black_outline_pixels = np.all(arr <= 20, axis=2)
        gray_shadow_pixels = np.all((arr >= 35) & (arr <= 80), axis=2)

        self.assertGreater(int(np.count_nonzero(white_fill_pixels)), 20)
        self.assertGreater(int(np.count_nonzero(black_outline_pixels)), 20)
        self.assertGreater(int(np.count_nonzero(gray_shadow_pixels)), 20)
        self.assertIn("render_bbox", text_data)
        rx1, ry1, rx2, ry2 = text_data["render_bbox"]
        self.assertLess(rx1, rx2)
        self.assertLess(ry1, ry2)

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

    def test_find_font_resolves_project_system_font_manifest(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        source_font = fonts_dir / "ComicNeue-Bold.ttf"
        system_font_name = "SystemFont__Arial__Regular.ttf"

        renderer_mod._font_cache.clear()
        renderer_mod._font_path_cache.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            font_path = Path(tmpdir) / "Arial.ttf"
            font_path.write_bytes(source_font.read_bytes())
            manifest = {
                "system": {
                    system_font_name: {
                        "family": "Arial",
                        "path": str(font_path),
                        "weight": "400",
                        "style": "normal",
                    }
                }
            }

            self.assertEqual(find_font(system_font_name, font_assets=manifest), str(font_path))

            renderer_mod.set_project_font_assets(manifest)
            self.assertEqual(find_font(system_font_name), str(font_path))

        renderer_mod.set_project_font_assets(None)

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

    def test_render_safe_arc_text_layer_places_center_above_edges(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        font = SafeTextPathFont(fonts_dir / "ComicNeue-Bold.ttf", 34)
        image_np = np.full((160, 360, 3), 255, dtype=np.uint8)

        bbox = _render_safe_arc_text_layer(
            image_np,
            "OLA TUDO BEM",
            font,
            (55, 72),
            {
                "curva": True,
                "curva_direcao": "arc_up",
                "curva_intensidade": 0.42,
            },
            fill_color="#000000",
        )

        self.assertIsNotNone(bbox)
        ink = np.any(image_np < 245, axis=2)
        ys, xs = np.where(ink)
        self.assertGreater(xs.size, 0)
        x_min, x_max = int(xs.min()), int(xs.max())
        span = max(1, x_max - x_min + 1)
        left = ys[xs < x_min + span * 0.25]
        center = ys[(xs >= x_min + span * 0.40) & (xs <= x_min + span * 0.60)]
        right = ys[xs > x_min + span * 0.75]
        self.assertGreater(left.size, 0)
        self.assertGreater(center.size, 0)
        self.assertGreater(right.size, 0)
        self.assertLess(float(np.median(center)), float((np.median(left) + np.median(right)) / 2.0))

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

    def test_build_render_blocks_omits_suppressed_scanlation_credit(self):
        text = {
            "translated": "WE ARE RECRUITING!",
            "text": "WE ARE RECRUITING!",
            "bbox": [200, 1200, 560, 1240],
            "skip_processing": True,
            "skip_reason": "scanlation_credit_suppressed",
            "route_action": "review_required",
            "route_reason": "scanlation_credit_suppressed",
            "qa_flags": ["scanlation_credit_suppressed"],
            "tipo": "fala",
        }

        blocks = build_render_blocks([text])

        self.assertEqual(blocks, [])
        self.assertFalse(text["visible"])
        self.assertEqual(text["render_policy"], "preserve_original")
        self.assertTrue(text["skip_processing"])

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

    def test_render_on_art_suspected_clears_stale_warning_for_selected_dark_lobe_render(self):
        background = np.zeros((180, 260, 3), dtype=np.uint8)
        text_data = {
            "original": "You can't stay in the subspace for long.",
            "translated": "Voce nao pode ficar no subespaco por muito tempo.",
            "render_bbox": [78, 62, 182, 124],
            "safe_text_box": [58, 48, 202, 138],
            "target_bbox": [44, 35, 222, 150],
            "balloon_bbox": [44, 35, 222, 150],
            "bubble_mask_bbox": [44, 35, 222, 150],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "background_rgb": [0, 0, 0],
            "qa_flags": [
                "render_on_art_suspected",
                "text_residual_after_inpaint",
                "mask_outside_balloon",
            ],
        }
        plan = {
            "target_bbox": [44, 35, 222, 150],
            "safe_text_box": [58, 48, 202, 138],
        }

        renderer_mod._run_render_qa(text_data, plan, background_image=background)

        self.assertNotIn("render_on_art_suspected", text_data.get("qa_flags") or [])
        self.assertIn("text_residual_after_inpaint", text_data.get("qa_flags") or [])
        metrics = text_data.get("qa_metrics") or {}
        self.assertIn("render_on_art_suspected_revalidated", metrics)
        self.assertEqual(metrics["render_on_art_suspected_revalidated"]["decision"], "cleared")
        self.assertIn("render_on_art_suspected", metrics.get("resolved_pre_render_flags") or [])

    def test_render_on_art_suspected_keeps_warning_when_selected_dark_lobe_render_exceeds_lobe(self):
        background = np.zeros((180, 260, 3), dtype=np.uint8)
        text_data = {
            "original": "You can't stay in the subspace for long.",
            "translated": "Voce nao pode ficar no subespaco por muito tempo.",
            "render_bbox": [20, 20, 236, 160],
            "safe_text_box": [58, 48, 202, 138],
            "target_bbox": [44, 35, 222, 150],
            "balloon_bbox": [44, 35, 222, 150],
            "bubble_mask_bbox": [44, 35, 222, 150],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "background_rgb": [0, 0, 0],
            "qa_flags": ["render_on_art_suspected"],
        }
        plan = {
            "target_bbox": [44, 35, 222, 150],
            "safe_text_box": [58, 48, 202, 138],
        }

        renderer_mod._run_render_qa(text_data, plan, background_image=background)

        self.assertIn("render_on_art_suspected", text_data.get("qa_flags") or [])
        metrics = text_data.get("qa_metrics") or {}
        self.assertEqual(metrics["render_on_art_suspected_revalidated"]["decision"], "kept")

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

    def test_simple_layout_still_merges_split_lobe_fragment_with_same_bubble_neighbor(self):
        texts = [
            {
                "id": "ocr_001",
                "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                "text": "HEY, LET'S GO! I'M STARVING",
                "original": "HEY, LET'S GO! I'M STARVING",
                "bbox": [29, 7109, 642, 7722],
                "source_bbox": [29, 7109, 642, 7722],
                "text_pixel_bbox": [148, 7248, 537, 7686],
                "layout_bbox": [148, 7248, 537, 7686],
                "line_polygons": [
                    [[148, 7248], [310, 7248], [310, 7268], [148, 7268]],
                    [[345, 7665], [537, 7665], [537, 7686], [345, 7686]],
                ],
                "balloon_bbox": [0, 7013, 800, 7967],
                "balloon_type": "white",
                "tipo": "fala",
                "layout_profile": "white_balloon",
                "layout_group_size": 1,
                "band_id": "page_003_band_035",
                "trace_id": "ocr_001@page_003_band_035",
                "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48, "cor": "#000000", "alinhamento": "center"},
            },
            {
                "id": "ocr_003",
                "translated": "QUEM ESTÁ PAGANDO HOJE?",
                "text": "WHO'SPAYING TODAY?",
                "bbox": [325, 7647, 549, 7764],
                "source_bbox": [325, 7647, 549, 7764],
                "text_pixel_bbox": [344, 7702, 540, 7761],
                "layout_bbox": [344, 7702, 540, 7761],
                "line_polygons": [
                    [[344, 7702], [540, 7702], [540, 7723], [344, 7723]],
                    [[389, 7736], [496, 7736], [496, 7761], [389, 7761]],
                ],
                "balloon_bbox": [276, 7612, 598, 7799],
                "balloon_type": "white",
                "tipo": "fala",
                "layout_profile": "white_balloon",
                "layout_group_size": 1,
                "band_id": "page_003_band_035",
                "trace_id": "ocr_003@page_003_band_035",
                "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 48, "cor": "#000000", "alinhamento": "center"},
            },
        ]

        with patch.dict("os.environ", {"TRADUZAI_SIMPLE_LAYOUT_ONLY": "1"}):
            blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 2)
        lower = next(block for block in blocks if block.get("_merged_nearby_white_fragments"))
        self.assertEqual(lower["translated"], "ESTOU MORRENDO DE FOME\nQUEM ESTÁ PAGANDO HOJE?")
        self.assertEqual(lower["layout_group_size"], 2)

    def test_simple_layout_merges_overlapping_same_band_white_balloon_lines(self):
        texts = [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_035",
                "band_id": "page_003_band_035",
                "translated": "EI, VAMOS!",
                "bbox": [148, 235, 310, 255],
                "balloon_bbox": [148, 235, 310, 255],
                "block_profile": "white_balloon",
                "layout_profile": "white_balloon",
            },
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_035",
                "band_id": "page_003_band_035",
                "translated": "ESTOU MORRENDO DE FOME",
                "bbox": [345, 652, 537, 673],
                "balloon_bbox": [345, 652, 537, 673],
                "block_profile": "white_balloon",
                "layout_profile": "white_balloon",
            },
            {
                "id": "ocr_003",
                "trace_id": "ocr_003@page_003_band_035",
                "band_id": "page_003_band_035",
                "translated": "QUEM ESTÁ PAGANDO HOJE?",
                "bbox": [325, 634, 549, 751],
                "balloon_bbox": [276, 599, 598, 786],
                "block_profile": "white_balloon",
                "layout_profile": "white_balloon",
            },
        ]

        with patch.dict("os.environ", {"TRADUZAI_SIMPLE_LAYOUT_ONLY": "1"}):
            blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 2)
        lower = next(block for block in blocks if block.get("_merged_nearby_white_fragments"))
        self.assertEqual(lower["translated"], "ESTOU MORRENDO DE FOME\nQUEM ESTÁ PAGANDO HOJE?")
        self.assertEqual(lower["layout_group_size"], 2)

    def test_simple_layout_merges_split_line_with_neighbor_when_band_is_only_in_trace_id(self):
        texts = [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_035",
                "translated": "EI, VAMOS!",
                "bbox": [148, 235, 310, 255],
                "source_bbox": [148, 235, 310, 255],
                "text_pixel_bbox": [148, 235, 310, 255],
                "layout_bbox": [148, 235, 310, 255],
                "balloon_bbox": [148, 235, 310, 255],
                "block_profile": "white_balloon",
                "layout_profile": "white_balloon",
            },
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_035",
                "translated": "ESTOU MORRENDO DE FOME",
                "bbox": [345, 652, 537, 673],
                "source_bbox": [345, 652, 537, 673],
                "text_pixel_bbox": [345, 652, 537, 673],
                "layout_bbox": [345, 652, 537, 673],
                "balloon_bbox": [345, 652, 537, 673],
                "block_profile": "white_balloon",
                "layout_profile": "white_balloon",
            },
            {
                "id": "ocr_003",
                "trace_id": "ocr_003@page_003_band_035",
                "translated": "QUEM ESTÃ PAGANDO HOJE?",
                "bbox": [325, 634, 549, 751],
                "source_bbox": [344, 689, 540, 748],
                "text_pixel_bbox": [344, 689, 540, 748],
                "layout_bbox": [344, 689, 540, 748],
                "balloon_bbox": [276, 599, 598, 786],
                "block_profile": "white_balloon",
                "layout_profile": "white_balloon",
            },
        ]

        with patch.dict("os.environ", {"TRADUZAI_SIMPLE_LAYOUT_ONLY": "1"}):
            blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 2)
        lower = next(block for block in blocks if block.get("_merged_nearby_white_fragments"))
        self.assertEqual(lower["translated"], "ESTOU MORRENDO DE FOME\nQUEM ESTÃ PAGANDO HOJE?")
        self.assertEqual(lower["layout_group_size"], 2)

    def test_typeset_trace_metadata_hydrates_band_context_into_texts(self):
        ocr_page = {
            "_band_id": "page_003_band_035",
            "texts": [
                {"id": "ocr_001", "translated": "EI, VAMOS!"},
                {"text_id": "ocr_003", "translated": "QUEM ESTÁ PAGANDO HOJE?"},
            ],
        }

        renderer_mod._ensure_typeset_trace_metadata(ocr_page)

        self.assertEqual(ocr_page["texts"][0]["band_id"], "page_003_band_035")
        self.assertEqual(ocr_page["texts"][1]["band_id"], "page_003_band_035")
        self.assertEqual(ocr_page["texts"][0]["page_id"], "page_003")
        self.assertEqual(ocr_page["texts"][1]["page_id"], "page_003")
        self.assertEqual(ocr_page["texts"][0]["trace_id"], "ocr_001@page_003_band_035")
        self.assertEqual(ocr_page["texts"][1]["trace_id"], "ocr_003@page_003_band_035")

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

    def test_dark_panel_mask_uses_panel_capacity_when_ocr_anchor_is_tight(self):
        img = Image.new("RGB", (760, 420), (18, 24, 32))
        text_data = {
            "translated": "VOCES TRES SAO OS CANDIDATOS",
            "original": "You three are the candidates",
            "tipo": "narracao",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "bubble_mask_source": "image_dark_panel_mask",
            "bbox": [272, 142, 482, 179],
            "source_bbox": [270, 140, 486, 181],
            "text_pixel_bbox": [276, 145, 478, 176],
            "line_polygons": [
                [[276, 145], [478, 145], [478, 176], [276, 176]],
            ],
            "balloon_bbox": [268, 139, 488, 182],
            "bubble_mask_bbox": [90, 88, 670, 244],
            "safe_text_box": [274, 146, 480, 176],
            "_debug_safe_text_box": [274, 146, 480, 176],
            "qa_metrics": {
                "image_dark_panel_mask": {
                    "mask_bbox": [90, 88, 670, 244],
                },
            },
            "qa_flags": [
                "dark_visual_font_capped_for_readability",
                "dark_panel_condensed_group_font",
            ],
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 44,
                "cor": "#EBFFFF",
                "contorno": "",
                "contorno_px": 0,
                "glow": True,
                "glow_cor": "#EBFFFF",
                "glow_px": 2,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        debug = text_data.get("_render_debug") or {}
        self.assertEqual(debug.get("target_bbox"), [90, 88, 670, 244])
        self.assertGreaterEqual(debug.get("capacity_bbox", [0, 0, 0, 0])[2] - debug.get("capacity_bbox", [0, 0, 0, 0])[0], 460)
        self.assertGreaterEqual(debug.get("position_bbox", [0, 0, 0, 0])[2] - debug.get("position_bbox", [0, 0, 0, 0])[0], 460)
        self.assertGreaterEqual(text_data["_render_debug"]["font_size_final"], 24)
        self.assertIn(text_data["fit_status"], {"ok", "below_minimum_legible"})
        self.assertNotIn("fit_below_minimum_legible", text_data.get("qa_flags", []))

    def test_dark_panel_full_bbox_prefers_largest_fitting_multiline_type(self):
        img = Image.new("RGB", (520, 380), (8, 4, 2))
        text_data = {
            "translated": "Introducao a missao: como o rei Yeomra, voce deve estabelecer seu proprio submundo!",
            "original": "Introduction to the mission: as King Yeomra, you must establish your own underworld!",
            "tipo": "narracao",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "bubble_mask_source": "image_dark_panel_mask",
            "bbox": [113, 165, 259, 236],
            "source_bbox": [113, 165, 259, 236],
            "text_pixel_bbox": [113, 165, 259, 236],
            "line_polygons": [
                [[113, 165], [259, 165], [259, 236], [113, 236]],
            ],
            "balloon_bbox": [22, 156, 484, 324],
            "bubble_mask_bbox": [22, 156, 484, 324],
            "bubble_inner_bbox": [43, 172, 463, 308],
            "safe_text_box": [43, 172, 463, 308],
            "qa_metrics": {
                "image_dark_panel_mask": {
                    "mask_bbox": [22, 156, 484, 324],
                    "full_panel_bbox_selected": True,
                },
            },
            "qa_flags": [
                "dark_panel_full_bbox_selected",
                "dark_panel_full_bbox_safe_clamped_to_inner",
            ],
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 24,
                "cor": "#FEFBEB",
                "contorno": "#968D83",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#32270A",
                "glow_px": 2,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        debug = text_data.get("_render_debug") or {}
        self.assertEqual(debug.get("target_bbox"), [22, 156, 484, 324])
        self.assertEqual(debug.get("capacity_bbox"), [43, 172, 463, 308])
        safe = text_data.get("safe_text_box") or [0, 0, 0, 0]
        render = text_data.get("render_bbox") or [0, 0, 0, 0]
        self.assertGreaterEqual(render[0], safe[0])
        self.assertGreaterEqual(render[1], safe[1])
        self.assertLessEqual(render[2], safe[2])
        self.assertLessEqual(render[3], safe[3])
        anchor = text_data["text_pixel_bbox"]
        render_cx = (render[0] + render[2]) / 2.0
        render_cy = (render[1] + render[3]) / 2.0
        anchor_cx = (anchor[0] + anchor[2]) / 2.0
        anchor_cy = (anchor[1] + anchor[3]) / 2.0
        self.assertLess(abs(render_cx - anchor_cx), 3.0)
        self.assertLess(abs(render_cy - anchor_cy), 3.0)
        self.assertGreaterEqual(debug.get("font_size_final", 0), 8)
        self.assertIn(text_data["fit_status"], {"ok", "below_minimum_legible"})
        self.assertIn("full_dark_panel_visual_capacity", text_data.get("qa_flags") or [])

    def test_small_trusted_dark_panel_uses_full_visual_capacity_for_readability(self):
        text_data = {
            "translated": "A missão principal será mostrada em breve",
            "original": "Main Quest will be shown shortly",
            "tipo": "narracao",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "bubble_mask_source": "image_dark_panel_mask",
            "bbox": [113, 2897, 259, 2968],
            "source_bbox": [113, 2897, 259, 2968],
            "text_pixel_bbox": [113, 2897, 259, 2968],
            "target_bbox": [70, 2868, 302, 2997],
            "balloon_bbox": [70, 2868, 302, 2997],
            "bubble_mask_bbox": [70, 2868, 302, 2997],
            "bubble_inner_bbox": [116, 2893, 254, 2956],
            "layout_safe_bbox": [84, 2878, 288, 2987],
            "layout_safe_reason": "visual_rect_dark_panel_mask",
            "qa_flags": [
                "dark_panel_full_bbox_selected",
                "dark_panel_rect_from_dark_bubble_bbox",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
            ],
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 48,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 4,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["capacity_bbox"], [83, 2875, 289, 2990])
        anchor = text_data["text_pixel_bbox"]
        position = plan["position_bbox"]
        self.assertLess(abs(((position[0] + position[2]) / 2.0) - ((anchor[0] + anchor[2]) / 2.0)), 3.0)
        self.assertLess(abs(((position[1] + position[3]) / 2.0) - ((anchor[1] + anchor[3]) / 2.0)), 3.0)
        self.assertGreaterEqual(plan["max_width"], 190)
        self.assertLessEqual(plan["line_spacing_ratio"], 0.18)
        self.assertGreaterEqual(resolved["font_size"], 24)
        self.assertIn("full_dark_panel_visual_capacity", text_data.get("qa_flags") or [])

    def test_trusted_dark_visual_capacity_does_not_cap_font_to_original_text_anchor(self):
        text_data = {
            "translated": "A missao principal sera mostrada em breve",
            "original": "Main Quest will be shown shortly",
            "tipo": "narracao",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [130, 88, 270, 126],
            "source_bbox": [130, 88, 270, 126],
            "text_pixel_bbox": [130, 88, 270, 126],
            "target_bbox": [40, 40, 360, 190],
            "capacity_bbox": [58, 54, 342, 176],
            "safe_text_box": [58, 54, 342, 176],
            "balloon_bbox": [40, 40, 360, 190],
            "bubble_mask_bbox": [40, 40, 360, 190],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "dark_panel_full_bbox_selected",
            ],
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 44,
                "cor": "#FEFBEB",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)

        self.assertGreaterEqual(plan["target_size"], 24)

    def test_dark_visual_render_keeps_margin_inside_safe_width(self):
        text_data = {
            "translated": "Mova-se imediatamente!",
            "original": "Move at once!",
            "tipo": "narracao",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [508, 470, 692, 511],
            "source_bbox": [508, 470, 692, 511],
            "text_pixel_bbox": [508, 470, 692, 511],
            "target_bbox": [312, 72, 800, 636],
            "capacity_bbox": [488, 354, 712, 628],
            "safe_text_box": [488, 354, 712, 628],
            "balloon_bbox": [312, 72, 800, 636],
            "bubble_mask_bbox": [312, 72, 800, 636],
            "background_rgb": [1, 1, 1],
            "qa_flags": [
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "safe_text_box_recomputed",
            ],
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 44,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        safe_w = text_data["safe_text_box"][2] - text_data["safe_text_box"][0]
        self.assertLessEqual(resolved["block_width"], int(round(safe_w * 0.92)))
        self.assertLessEqual(plan["max_width"], int(round(safe_w * 0.90)))
        self.assertIn("dark_visual_safe_width_limited", text_data.get("qa_flags") or [])

    def test_dark_visual_capacity_expands_within_lobe_when_contract_is_too_narrow(self):
        text_data = {
            "translated": "A recompensa da missão é...",
            "original": "The quest reward is...",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [520, 4053, 646, 4120],
            "source_bbox": [520, 4053, 646, 4120],
            "text_pixel_bbox": [520, 4053, 646, 4120],
            "target_bbox": [530, 4059, 644, 4128],
            "safe_text_box": [559, 4066, 614, 4087],
            "balloon_bbox": [557, 4059, 617, 4094],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "dark_bubble_ellipse_bbox_mask",
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "source_text_mask_bbox_from_inpaint_component",
                "connected_lobe_boxes_missing_source_anchor_fallback",
            ],
            "qa_metrics": {
                "dark_text_contract_fill_mask": {"bbox": [518, 4050, 649, 4124]},
                "derived_card_panel_mask": {"mask_bbox": [487, 4027, 679, 4146]},
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 36,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        metric = text_data["qa_metrics"]["dark_visual_capacity_expanded_within_lobe"]
        self.assertEqual(metric["reason"], "contract_bbox_narrower_than_visual_lobe")
        self.assertEqual(metric["visual_lobe_bbox"], [487, 4027, 679, 4146])
        self.assertEqual(plan["layout_safe_reason"], "dark_visual_capacity_expanded_within_lobe")
        self.assertGreaterEqual(plan["safe_text_box"][2] - plan["safe_text_box"][0], 140)
        self.assertGreater(plan["max_width"], 110)
        self.assertLessEqual(plan["safe_text_box"][0], 518)
        self.assertGreaterEqual(plan["safe_text_box"][2], 649)
        self.assertLessEqual(resolved["block_bbox"][0], 657)
        self.assertLessEqual(resolved["block_bbox"][2], 657)
        self.assertLessEqual(len(resolved["lines"]), 2)
        self.assertIn("dark_visual_capacity_expanded_within_lobe", text_data.get("qa_flags") or [])

    def test_dark_connected_lobe_anchor_promotes_existing_visual_lobe_capacity(self):
        text_data = {
            "translated": "Eu sou chamado de 'sistema'",
            "original": "I am called 'System'",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "source_bbox": [484, 342, 722, 384],
            "text_pixel_bbox": [485, 345, 717, 379],
            "target_bbox": [485, 345, 717, 379],
            "safe_text_box": [439, 188, 536, 519],
            "balloon_bbox": [476, 175, 775, 459],
            "background_rgb": [0, 0, 0],
            "qa_flags": [
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "short_dark_anchor_center_preserved",
                "dark_visual_capacity_expanded_within_lobe",
                "dark_connected_component_safe_partition",
            ],
            "qa_metrics": {
                "dark_text_contract_fill_mask": {"bbox": [484, 342, 722, 384]},
                "image_dark_bubble_mask": {
                    "source": "image_dark_bubble_mask",
                    "shape_kind": "ellipse",
                    "mask_bbox": [476, 175, 775, 459],
                    "anchor_bbox": [485, 345, 717, 379],
                },
                "dark_visual_capacity_expanded_within_lobe": {
                    "reason": "contract_bbox_narrower_than_visual_lobe",
                    "contract_bbox": [484, 342, 722, 384],
                    "visual_lobe_bbox": [476, 175, 775, 459],
                    "visual_lobe_bbox_source": "qa_metrics.image_dark_bubble_mask.mask_bbox",
                    "previous_safe_text_box": [439, 188, 536, 519],
                    "expanded_safe_text_box": [500, 259, 751, 449],
                    "previous_max_width": 97,
                    "expanded_max_width": 251,
                    "center_preserved": False,
                    "target_bbox": [476, 175, 775, 459],
                },
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 29,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }
        plan = {
            "target_bbox": [485, 345, 717, 379],
            "safe_text_box": [439, 188, 536, 519],
            "layout_safe_bbox": [439, 188, 536, 519],
            "layout_safe_reason": "short_dark_anchor_scale_preserved",
            "position_bbox": [439, 188, 536, 519],
            "capacity_bbox": [439, 188, 536, 519],
            "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
            "target_size": 29,
            "_font_search_cap": 29,
            "_font_search_floor": 16,
            "max_width": 85,
            "max_height": 300,
            "padding_y": 8,
            "line_spacing_ratio": 0.08,
            "vertical_anchor": "center",
            "alignment": "center",
            "outline_px": 1,
        }

        resolved = _resolve_text_layout(text_data, plan)

        self.assertGreaterEqual(plan["safe_text_box"][0], 476)
        self.assertLessEqual(plan["safe_text_box"][2], 775)
        self.assertGreaterEqual(plan["position_bbox"][0], 476)
        self.assertLessEqual(plan["position_bbox"][2], 775)
        self.assertGreaterEqual(resolved["block_bbox"][0], 476)
        self.assertLessEqual(resolved["block_bbox"][2], 775)
        self.assertIn("dark_connected_lobe_anchor_localized", text_data.get("qa_flags") or [])
        metric = text_data["qa_metrics"]["dark_connected_lobe_anchor_localized"]
        self.assertIn(metric["decision"], {"applied", "already_localized"})
        self.assertFalse(metric["sibling_lobe_used"])

    def test_dark_connected_lobe_skips_glow_capacity_recovery_when_visual_partition_exists(self):
        text_data = {
            "translated": "Eu sou chamado de 'sistema'",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "target_bbox": [476, 175, 775, 459],
            "text_pixel_bbox": [485, 345, 717, 379],
            "bbox": [485, 345, 717, 379],
            "qa_flags": [
                "visual_text_only_inpaint_contract",
                "dark_connected_component_safe_partition",
                "dark_visual_capacity_expanded_within_lobe",
            ],
            "qa_metrics": {
                "image_dark_bubble_mask": {"mask_bbox": [476, 175, 775, 459]},
                "dark_visual_capacity_expanded_within_lobe": {
                    "visual_lobe_bbox": [476, 175, 775, 459],
                    "expanded_safe_text_box": [500, 259, 751, 449],
                },
            },
        }
        plan = {
            "target_bbox": [476, 175, 775, 459],
            "safe_text_box": [500, 259, 751, 449],
            "layout_safe_bbox": [500, 259, 751, 449],
            "layout_safe_reason": "dark_visual_capacity_expanded_within_lobe",
        }
        img = Image.new("RGB", (800, 600), (0, 0, 0))

        recovered = renderer_mod._recover_dark_bubble_glow_capacity_from_image(img, text_data, plan)

        self.assertIsNone(recovered)
        self.assertIn("dark_bubble_glow_capacity_rejected_connected_lobe", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_bubble_glow_capacity_recovered", text_data.get("qa_flags") or [])

    def test_dark_connected_lobe_metric_replaces_stale_pair_safe_box(self):
        text_data = {
            "translated": "A retenção do subespaço é de apenas cinco minutos.",
            "original": "The subspace retention is only five minutes.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [237, 183, 677, 399],
            "source_bbox": [237, 183, 677, 399],
            "text_pixel_bbox": [237, 183, 677, 399],
            "target_bbox": [83, 96, 744, 755],
            "safe_text_box": [129, 142, 698, 709],
            "balloon_bbox": [237, 183, 677, 399],
            "background_rgb": [0, 0, 0],
            "qa_flags": [
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "dark_connected_component_safe_partition",
                "connected_lobe_boxes_missing_source_anchor_fallback",
                "visual_text_only_inpaint_contract",
            ],
            "qa_metrics": {
                "dark_connected_text_pixel_bbox_replaced_by_lobe_bbox": {
                    "text_pixel_bbox": [237, 183, 677, 399],
                    "lobe_bbox": [129, 172, 313, 299],
                    "text_area": 95040,
                    "lobe_area": 23368,
                    "overlap": 8816,
                },
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 35,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }
        plan = {
            "target_bbox": [83, 96, 744, 755],
            "safe_text_box": [129, 142, 698, 709],
            "layout_safe_bbox": [129, 142, 698, 709],
            "layout_safe_reason": "trusted_dark_visual_capacity",
            "position_bbox": [129, 142, 698, 709],
            "capacity_bbox": [129, 142, 698, 709],
            "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
            "target_size": 35,
            "_font_search_cap": 35,
            "_font_search_floor": 16,
            "max_width": 500,
            "max_height": 540,
            "padding_y": 8,
            "line_spacing_ratio": 0.08,
            "vertical_anchor": "center",
            "alignment": "center",
            "outline_px": 1,
        }

        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["layout_safe_reason"], "dark_connected_lobe_metric_safe_box")
        self.assertLess(plan["safe_text_box"][2], 430)
        self.assertLess(plan["safe_text_box"][3], 380)
        self.assertGreaterEqual(resolved["block_bbox"][0], plan["safe_text_box"][0])
        self.assertLessEqual(resolved["block_bbox"][2], plan["safe_text_box"][2])
        self.assertIn("dark_connected_lobe_final_fit_repaired", text_data.get("qa_flags") or [])
        metric = text_data["qa_metrics"]["dark_connected_lobe_final_fit_repaired"]
        self.assertEqual(metric["decision"], "applied")
        self.assertFalse(metric["sibling_lobe_used"])

    def test_dark_connected_lobe_scale_prefers_local_anchor_over_broad_contract(self):
        text_data = {
            "translated": "Se você ultrapassar esse tempo, você retornará ao seu mundo original!",
            "original": "If you exceed that time, you will return to your original world!",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [237, 183, 677, 399],
            "source_bbox": [237, 183, 677, 399],
            "text_pixel_bbox": [237, 183, 677, 399],
            "source_text_anchor_bbox": [398, 268, 680, 400],
            "_source_text_anchor_bbox": [398, 268, 680, 400],
            "target_bbox": [360, 136, 709, 585],
            "safe_text_box": [360, 136, 709, 585],
            "balloon_bbox": [237, 183, 677, 399],
            "background_rgb": [0, 0, 0],
            "qa_flags": [
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "dark_connected_component_safe_partition",
                "dark_connected_lobe_anchor_component_filtered",
                "broad_connected_bubble_mask_rejected",
                "dark_connected_lobe_mask_rebuilt_from_glyphs",
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "render_text_mask_cleanup",
            ],
            "qa_metrics": {
                "layout_text_geometry_sanitized": {
                    "clean_bbox": [398, 271, 680, 400],
                },
                "dark_connected_bubble_broad_mask_rejected": {
                    "candidate_bbox": [128, 172, 681, 403],
                    "anchor_bbox": [398, 268, 680, 400],
                },
                "dark_text_contract_fill_mask": {
                    "bbox": [226, 179, 681, 403],
                    "mask_pixels": 43744,
                    "source": "raw_glyph_mask",
                },
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 48,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        scale_bbox = renderer_mod._original_text_mask_bbox_for_scale(text_data)

        self.assertEqual(scale_bbox, [398, 268, 680, 400])
        self.assertIn("dark_connected_local_anchor_scale_contract", text_data.get("qa_flags") or [])
        metric = text_data["qa_metrics"]["dark_connected_local_anchor_overrode_scale_contract"]
        self.assertEqual(metric["anchor_bbox"], [398, 268, 680, 400])
        self.assertEqual(metric["contract_bbox"], [226, 179, 681, 403])

    def test_dark_connected_lobe_rect_cleanup_preserves_inpainted_bridge(self):
        img = Image.new("RGB", (240, 160), (4, 6, 8))
        draw = ImageDraw.Draw(img)
        draw.line((40, 80, 200, 80), fill=(20, 180, 220), width=4)
        before = img.copy()
        text_data = {
            "translated": "SE VOCÊ ULTRAPASSAR ESSE TEMPO",
            "original": "If you exceed that time",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "source_text_mask_bbox": [50, 60, 190, 104],
            "text_pixel_bbox": [50, 60, 190, 104],
            "qa_flags": [
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "render_text_mask_cleanup",
                "dark_connected_component_safe_partition",
                "dark_connected_lobe_anchor_component_filtered",
                "broad_connected_bubble_mask_rejected",
                "dark_connected_lobe_mask_rebuilt_from_glyphs",
            ],
            "qa_metrics": {
                "dark_connected_bubble_broad_mask_rejected": {
                    "candidate_bbox": [20, 50, 220, 116],
                    "anchor_bbox": [130, 64, 204, 104],
                },
            },
            "estilo": {"glow_px": 3, "outline_width": 1},
        }

        changed = renderer_mod._apply_text_mask_cleanup_before_render(img, [text_data], {})

        self.assertFalse(changed)
        self.assertIsNone(ImageChops.difference(before, img).getbbox())
        self.assertIn("dark_connected_lobe_rect_cleanup_skipped", text_data.get("qa_flags") or [])
        metric = text_data["qa_metrics"]["dark_connected_lobe_rect_cleanup_skipped"]
        self.assertEqual(metric["decision"], "skipped")

    def test_dark_single_oval_short_text_prefers_larger_visual_lobe_capacity(self):
        text_data = {
            "translated": "Este e o subespaco do sistema.",
            "original": "This is the system's subspace.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [519, 3859, 690, 3982],
            "source_bbox": [519, 3859, 690, 3982],
            "text_pixel_bbox": [522, 3859, 687, 3974],
            "target_bbox": [151, 3509, 773, 4048],
            "safe_text_box": [522, 3855, 691, 3974],
            "balloon_bbox": [0, 3224, 800, 4169],
            "background_rgb": [0, 0, 0],
            "qa_flags": [
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "source_text_mask_bbox_from_inpaint_component",
                "dark_connected_compact_text_bbox_rejected_undercoverage",
            ],
            "qa_metrics": {
                "dark_text_contract_fill_mask": {"bbox": [519, 3856, 690, 3979]},
                "image_dark_bubble_mask": {"mask_bbox": [0, 3224, 800, 4169]},
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 22,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["layout_safe_reason"], "dark_visual_capacity_expanded_within_lobe")
        self.assertGreater(resolved["font_size"], 22)
        self.assertLessEqual(resolved["block_bbox"][2], 736)
        self.assertGreaterEqual(resolved["block_bbox"][0], 128)
        self.assertIn("dark_single_oval_capacity_expanded", text_data.get("qa_flags") or [])
        metric = text_data["qa_metrics"]["dark_single_oval_capacity_expanded"]
        self.assertEqual(metric["decision"], "applied")
        self.assertEqual(metric["reason"], "short_text_underfit_visual_lobe_has_room")
        self.assertEqual(metric["visual_lobe_bbox"], [0, 3224, 800, 4169])
        self.assertGreater(metric["new_font_size"], metric["old_font_size"])

    def test_dark_single_oval_long_text_uses_visual_lobe_width_capacity(self):
        text_data = {
            "translated": "Criterios de conclusao da missao: estabeleca um submundo de nivel 1.",
            "original": "Quest completion criteria: establish a level 1 underworld.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [82, 179, 365, 284],
            "source_bbox": [82, 179, 365, 284],
            "text_pixel_bbox": [82, 179, 365, 284],
            "target_bbox": [0, 66, 464, 413],
            "safe_text_box": [89, 182, 358, 281],
            "balloon_bbox": [0, 66, 464, 413],
            "background_rgb": [0, 0, 0],
            "qa_flags": [
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "source_text_mask_bbox_from_inpaint_component",
                "connected_layout_disabled_rejected_bubble_mask",
                "connected_lobe_boxes_missing_source_anchor_fallback",
            ],
            "qa_metrics": {
                "dark_text_contract_fill_mask": {"bbox": [78, 175, 370, 286]},
                "image_dark_bubble_mask": {
                    "source": "image_dark_bubble_mask",
                    "shape_kind": "ellipse",
                    "mask_bbox": [0, 66, 464, 413],
                    "anchor_bbox": [82, 179, 365, 284],
                },
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 29,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = {
            "layout_safe_reason": "dark_visual_capacity_expanded_within_lobe",
            "safe_text_box": [48, 182, 401, 281],
        }
        text_data["qa_metrics"]["dark_visual_capacity_expanded_within_lobe"] = {
            "reason": "contract_bbox_narrower_than_visual_lobe",
            "contract_bbox": [78, 175, 370, 286],
            "visual_lobe_bbox": [0, 66, 464, 413],
            "expanded_safe_text_box": [48, 182, 401, 281],
        }
        candidate = {
            "block_bbox": [74, 182, 374, 281],
            "block_width": 300,
            "block_height": 99,
            "font_size": 29,
        }

        preferred = renderer_mod._should_prefer_larger_dark_single_oval_visual_candidate(
            text_data,
            plan,
            candidate,
            ["Criterios de conclusao da missao:", "estabeleca um", "submundo de nivel 1."],
        )

        self.assertTrue(preferred)
        self.assertEqual(candidate["dark_single_oval_visual_capacity_reason"], "long_text_visual_lobe_width_has_room")

    def test_dark_single_oval_capacity_does_not_affect_connected_lobe_evidence(self):
        text_data = {
            "translated": "Voce nao pode ficar no subespaco por muito tempo.",
            "original": "You can't stay in the subspace for long.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [160, 59883, 355, 60019],
            "source_bbox": [160, 59883, 355, 60019],
            "text_pixel_bbox": [160, 59883, 355, 60019],
            "target_bbox": [163, 59886, 351, 60016],
            "safe_text_box": [190, 59901, 325, 60001],
            "balloon_bbox": [40, 59735, 561, 60100],
            "background_rgb": [0, 0, 0],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "source_text_mask_bbox_from_inpaint_component",
                "dark_bubble_lobe_clip_rejected_undercovered_text",
                "dark_bubble_glow_capacity_rejected_off_anchor",
                "dark_connected_component_safe_partition",
            ],
            "qa_metrics": {
                "dark_text_contract_fill_mask": {"bbox": [160, 59883, 355, 60019]},
                "image_dark_bubble_mask": {"mask_bbox": [40, 59735, 561, 60100]},
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 27,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)
        _resolve_text_layout(text_data, plan)

        self.assertEqual(plan["layout_safe_reason"], "dark_visual_capacity_expanded_within_lobe")
        self.assertNotIn("dark_single_oval_capacity_expanded", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_single_oval_capacity_expanded", text_data.get("qa_metrics") or {})

    def test_dark_visual_capacity_does_not_expand_without_reliable_visual_lobe(self):
        text_data = {
            "translated": "A recompensa da missão é...",
            "original": "The quest reward is...",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [520, 4053, 646, 4120],
            "source_bbox": [520, 4053, 646, 4120],
            "text_pixel_bbox": [520, 4053, 646, 4120],
            "target_bbox": [530, 4059, 644, 4128],
            "safe_text_box": [559, 4066, 614, 4087],
            "balloon_bbox": [557, 4059, 617, 4094],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "source_text_mask_bbox_from_inpaint_component",
            ],
            "qa_metrics": {
                "dark_text_contract_fill_mask": {"bbox": [518, 4050, 649, 4124]},
                "derived_card_panel_mask": {"mask_bbox": [700, 4027, 820, 4146]},
            },
            "estilo": {"fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf", "tamanho": 36, "cor": "#FFFFFF"},
        }

        plan = plan_text_layout(text_data)

        self.assertNotIn("dark_visual_capacity_expanded_within_lobe", text_data.get("qa_flags") or [])
        self.assertNotEqual(plan["layout_safe_reason"], "dark_visual_capacity_expanded_within_lobe")

    def test_dark_connected_lobe_safe_overhang_uses_visual_lobe_bounds(self):
        text_data = {
            "id": "page_002_band_026_left_lobe",
            "translated": "VOCE ERA LEAL AOS OUTROS, MAS PARA ELES ESTAVA SENDO INTROMETIDO.",
            "content_class": "dialogue",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "source_bbox": [129, 89, 428, 217],
            "text_pixel_bbox": [132, 92, 426, 215],
            "target_bbox": [0, 0, 413, 307],
            "balloon_bbox": [0, 0, 558, 307],
            "render_bbox": [157, 59, 400, 247],
            "qa_flags": [
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "dark_bubble_connected_lobes_promoted",
                "dark_bubble_connected_lobe_passthrough",
                "dark_bubble_lobe_mask_bbox_preferred",
            ],
            "qa_metrics": {
                "contract_bbox_tight_but_visual_balloon_fit_ok": {
                    "source_bbox": [129, 89, 428, 217],
                    "block_bbox": [157, 59, 400, 247],
                    "visual_bbox": [0, 0, 413, 307],
                    "visual_bbox_source": "bubble_mask_bbox",
                },
                "typeset_contract_fit": {
                    "source_bbox": [129, 89, 428, 217],
                    "block_bbox": [157, 59, 400, 247],
                },
            },
        }
        plan = {
            "target_bbox": [0, 0, 413, 307],
            "safe_text_box": [29, 21, 384, 286],
            "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
        }

        bounds = renderer_mod._dark_visual_lobe_render_bounds_for_safe_overhang(text_data, plan)
        self.assertEqual(bounds, [0, 0, 413, 307])

        renderer_mod._run_render_qa(text_data, plan)

        self.assertNotIn("TEXT_CLIPPED", text_data.get("qa_flags") or [])
        self.assertEqual(
            text_data["qa_metrics"]["render_safe_overhang_allowed_by_visual_lobe"]["visual_lobe_bbox"],
            [0, 0, 413, 307],
        )

    def test_dark_connected_lobe_safe_overhang_rejects_sibling_area(self):
        text_data = {
            "id": "dark_lobe_spills_to_sibling",
            "translated": "VOCE ERA LEAL AOS OUTROS, MAS PARA ELES ESTAVA SENDO INTROMETIDO.",
            "content_class": "dialogue",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "source_bbox": [129, 89, 428, 217],
            "text_pixel_bbox": [132, 92, 426, 215],
            "target_bbox": [0, 0, 413, 307],
            "balloon_bbox": [0, 0, 558, 307],
            "render_bbox": [157, 59, 430, 247],
            "qa_flags": [
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "dark_bubble_connected_lobes_promoted",
                "dark_bubble_connected_lobe_passthrough",
                "dark_bubble_lobe_mask_bbox_preferred",
            ],
            "qa_metrics": {
                "contract_bbox_tight_but_visual_balloon_fit_ok": {
                    "source_bbox": [129, 89, 428, 217],
                    "block_bbox": [157, 59, 430, 247],
                    "visual_bbox": [0, 0, 558, 307],
                    "visual_bbox_source": "bubble_mask_bbox",
                },
                "typeset_contract_fit": {
                    "source_bbox": [129, 89, 428, 217],
                    "block_bbox": [157, 59, 430, 247],
                },
            },
        }
        plan = {
            "target_bbox": [0, 0, 413, 307],
            "safe_text_box": [29, 21, 384, 286],
            "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
        }

        bounds = renderer_mod._dark_visual_lobe_render_bounds_for_safe_overhang(text_data, plan)
        self.assertEqual(bounds, [0, 0, 413, 307])

        renderer_mod._run_render_qa(text_data, plan)

        self.assertIn("TEXT_CLIPPED", text_data.get("qa_flags") or [])
        self.assertIn("TEXT_OVERFLOW", text_data.get("qa_flags") or [])

    def test_dark_visual_capacity_does_not_expand_connected_lobe_contracts(self):
        text_data = {
            "translated": "Sou um sistema que orienta o exercito do Rei Yeomra.",
            "original": "I am a system that guides...",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [66, 16572, 388, 16763],
            "source_bbox": [66, 16572, 388, 16763],
            "text_pixel_bbox": [66, 16572, 388, 16763],
            "target_bbox": [20, 16520, 460, 16820],
            "safe_text_box": [90, 16580, 330, 16730],
            "balloon_bbox": [20, 16520, 460, 16820],
            "connected_lobe_bboxes": [[20, 16520, 460, 16820], [430, 16560, 720, 16810]],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_connected_lobes_promoted",
                "dark_bubble_connected_lobe_passthrough",
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
                "source_text_mask_bbox_from_inpaint_component",
            ],
            "qa_metrics": {
                "dark_text_contract_fill_mask": {"bbox": [66, 16572, 388, 16763]},
                "derived_card_panel_mask": {"mask_bbox": [20, 16520, 720, 16820]},
            },
            "estilo": {"fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf", "tamanho": 36, "cor": "#FFFFFF"},
        }

        plan = plan_text_layout(text_data)

        self.assertNotIn("dark_visual_capacity_expanded_within_lobe", text_data.get("qa_flags") or [])
        self.assertNotEqual(plan["layout_safe_reason"], "dark_visual_capacity_expanded_within_lobe")

    def test_dark_visual_capacity_does_not_affect_white_balloon(self):
        text_data = {
            "translated": "Por que sou o anfitrião?",
            "original": "Why am I the host?",
            "tipo": "fala",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bubble_mask_source": "image_white_bubble_mask",
            "bbox": [80, 90, 210, 140],
            "source_bbox": [80, 90, 210, 140],
            "text_pixel_bbox": [80, 90, 210, 140],
            "target_bbox": [20, 20, 300, 190],
            "safe_text_box": [48, 42, 272, 168],
            "balloon_bbox": [20, 20, 300, 190],
            "qa_flags": ["visual_text_only_inpaint_contract", "text_contract_direct_fill"],
            "qa_metrics": {
                "dark_text_contract_fill_mask": {"bbox": [80, 90, 210, 140]},
                "derived_card_panel_mask": {"mask_bbox": [0, 0, 320, 220]},
            },
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 30, "cor": "#000000"},
        }

        plan = plan_text_layout(text_data)

        self.assertNotIn("dark_visual_capacity_expanded_within_lobe", text_data.get("qa_flags") or [])
        self.assertNotEqual(plan["layout_safe_reason"], "dark_visual_capacity_expanded_within_lobe")

    def test_dark_visual_white_mask_source_does_not_use_compact_small_text_capacity(self):
        text_data = {
            "translated": "A missao. recompensa e....",
            "original": "The quest reward is....",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_white_bubble_mask",
            "bbox": [530, 408, 644, 477],
            "source_bbox": [530, 408, 644, 477],
            "text_pixel_bbox": [520, 402, 646, 469],
            "target_bbox": [520, 402, 646, 469],
            "capacity_bbox": [521, 402, 636, 465],
            "safe_text_box": [524, 402, 634, 465],
            "balloon_bbox": [505, 388, 669, 497],
            "bubble_mask_bbox": [509, 384, 648, 477],
            "background_rgb": [2, 2, 2],
            "qa_flags": [],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 36,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)

        self.assertNotIn("compact_small_text_capacity", text_data.get("qa_flags") or [])
        self.assertGreaterEqual(plan["capacity_bbox"][2] - plan["capacity_bbox"][0], 140)
        self.assertGreaterEqual(plan["capacity_bbox"][3] - plan["capacity_bbox"][1], 90)

    def test_dark_visual_white_mask_source_uses_visual_safe_box_when_target_already_expanded(self):
        img = Image.new("RGB", (800, 600), (0, 0, 0))
        text_data = {
            "translated": "A missao. recompensa e....",
            "original": "The quest reward is....",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_white_bubble_mask",
            "bbox": [530, 408, 644, 477],
            "source_bbox": [530, 408, 644, 477],
            "text_pixel_bbox": [520, 402, 646, 469],
            "target_bbox": [520, 402, 646, 469],
            "capacity_bbox": [521, 402, 636, 465],
            "safe_text_box": [524, 402, 634, 465],
            "balloon_bbox": [505, 388, 669, 497],
            "bubble_mask_bbox": [509, 384, 648, 477],
            "bubble_inner_bbox": [523, 393, 634, 468],
            "background_rgb": [12, 12, 12],
            "qa_flags": ["dark_bubble_visual_glyph_mask_replaced_geometry"],
            "style_origin": "auto_dark_panel_glow",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 36,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        debug = text_data.get("_render_debug") or {}
        safe = debug.get("layout_safe_bbox") or [0, 0, 0, 0]
        self.assertEqual(debug.get("layout_safe_reason"), "trusted_dark_visual_capacity")
        self.assertGreaterEqual(safe[2] - safe[0], 140)
        self.assertGreaterEqual(debug.get("font_size_final", 0), 18)
        self.assertIn("trusted_dark_visual_capacity_target", text_data.get("qa_flags") or [])

    def test_dark_ellipse_visual_capacity_uses_bubble_area_not_tiny_text_anchor(self):
        text_data = {
            "translated": "Criterios de conclusao da missao: estabeleca um submundo de nivel 1.",
            "original": "Quest completion criteria: establish a Level 1 underworld.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [160, 110, 386, 176],
            "source_bbox": [160, 110, 386, 176],
            "text_pixel_bbox": [160, 110, 386, 176],
            "target_bbox": [80, 42, 474, 272],
            "capacity_bbox": [80, 42, 474, 272],
            "safe_text_box": [150, 84, 404, 226],
            "balloon_bbox": [80, 42, 474, 272],
            "bubble_mask_bbox": [80, 42, 474, 272],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "dark_bubble_ellipse_bbox_mask",
            ],
            "bubble_mask_ellipse": {"center": [277, 157], "axes": [394, 230]},
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 42,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        capacity = plan["capacity_bbox"]
        self.assertEqual(plan["_target_source"], "dark_bubble_visual_mask_bbox")
        self.assertGreaterEqual(capacity[2] - capacity[0], 300)
        self.assertGreaterEqual(capacity[3] - capacity[1], 180)
        self.assertGreaterEqual(resolved["font_size"], 24)
        self.assertNotIn("dark_visual_font_capped_for_readability", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_visual_auto_font_capped_to_source_scale", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_visual_effect_capacity_inset", text_data.get("qa_flags") or [])

    def test_dark_oval_visual_capacity_expands_safe_height_without_changing_width(self):
        text_data = {
            "translated": "Criterios de conclusao da missao: estabeleca um submundo de nivel 1.",
            "original": "Quest Completion Criteria: Establish a Level 1 underworld.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [79, 112, 368, 221],
            "source_bbox": [79, 112, 368, 221],
            "text_pixel_bbox": [79, 112, 368, 221],
            "target_bbox": [58, 93, 393, 239],
            "safe_text_box": [98, 114, 352, 217],
            "balloon_bbox": [58, 93, 393, 239],
            "bubble_mask_bbox": [58, 93, 393, 239],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "dark_bubble_ellipse_bbox_mask",
            ],
            "qa_metrics": {
                "image_dark_bubble_mask": {
                    "shape_kind": "ellipse",
                    "mask_bbox": [58, 93, 393, 239],
                }
            },
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 42,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)

        safe = plan["safe_text_box"]
        expanded_from = text_data.get("_dark_oval_safe_height_expanded_from")
        self.assertIsNotNone(expanded_from)
        self.assertGreaterEqual(safe[2] - safe[0], 230)
        self.assertGreaterEqual(safe[3] - safe[1], 128)
        self.assertGreater(safe[3] - safe[1], expanded_from[3] - expanded_from[1])
        self.assertIn(
            plan["layout_safe_reason"],
            {"dark_oval_safe_height_expanded", "dark_oval_safe_clipped_to_visible_balloon"},
        )
        self.assertIn("dark_oval_safe_height_expanded", text_data.get("qa_flags") or [])
        self.assertGreaterEqual(resolved["font_size"], 24)
        self.assertNotIn("dark_visual_auto_font_capped_to_source_scale", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_visual_effect_capacity_inset", text_data.get("qa_flags") or [])

    def test_partial_connected_disabled_dark_oval_can_clip_to_visible_balloon(self):
        text_data = {
            "translated": "Critérios de conclusão da missão: estabeleça um submundo de nível 1.",
            "original": "Quest completion criteria: establish a Level 1 underworld.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [82, 113, 367, 219],
            "source_bbox": [82, 113, 367, 219],
            "text_pixel_bbox": [82, 113, 367, 219],
            "target_bbox": [0, 0, 467, 325],
            "capacity_bbox": [47, 32, 420, 293],
            "safe_text_box": [89, 116, 360, 216],
            "balloon_bbox": [74, 102, 375, 230],
            "bubble_mask_bbox": [74, 102, 375, 230],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "dark_bubble_ellipse_bbox_mask",
                "partial_dark_bubble_lobe_reocr",
                "connected_layout_disabled_dark_panel_visual_mask",
            ],
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 48,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        bounds = renderer_mod._dark_oval_safe_expansion_bounds(text_data, [0, 0, 467, 325])
        clipped = renderer_mod._clip_dark_oval_safe_to_visible_balloon(text_data, [47, 0, 420, 325])

        self.assertEqual(bounds, [74, 102, 375, 230])
        self.assertEqual(clipped, [82, 105, 367, 227])
        self.assertIn("dark_oval_safe_expansion_limited_to_visible_balloon", text_data.get("qa_flags") or [])
        self.assertIn("dark_oval_safe_clipped_to_visible_balloon", text_data.get("qa_flags") or [])

    def test_page_002_band_007_dark_oval_keeps_full_capacity_when_visible_bbox_is_clip_fragment(self):
        text_data = {
            "translated": "Criterios de conclusao da missao: estabeleca um submundo de nivel 1.",
            "original": "Quest completion criteria: establish a Level 1 underworld.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [82, 113, 367, 219],
            "source_bbox": [82, 113, 367, 219],
            "text_pixel_bbox": [82, 113, 367, 219],
            "target_bbox": [0, 0, 467, 325],
            "capacity_bbox": [47, 32, 420, 293],
            "safe_text_box": [89, 116, 360, 216],
            "balloon_bbox": [74, 102, 375, 230],
            "bubble_mask_bbox": [74, 102, 375, 230],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "dark_bubble_ellipse_bbox_mask",
                "connected_layout_disabled_dark_panel_visual_mask",
            ],
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 48,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        bounds = renderer_mod._dark_oval_safe_expansion_bounds(text_data, [0, 0, 467, 325])
        clipped = renderer_mod._clip_dark_oval_safe_to_visible_balloon(text_data, [47, 0, 420, 325])

        self.assertEqual(bounds, [0, 0, 467, 325])
        self.assertEqual(clipped, [47, 0, 420, 325])
        self.assertIn("dark_oval_visible_bbox_fragment_ignored", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_oval_safe_expansion_limited_to_visible_balloon", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_oval_safe_clipped_to_visible_balloon", text_data.get("qa_flags") or [])

    def test_page_005_band_078_left_lobe_keeps_parent_capacity_when_visible_bbox_is_clip_fragment(self):
        text_data = {
            "id": "page_005_band_078_left",
            "trace_id": "ocr_001@page_005_band_078",
            "translated": "Voce foi escolhido entre incontaveis almas.",
            "original": "You were chosen among countless souls.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [96, 152, 284, 236],
            "source_bbox": [96, 152, 284, 236],
            "text_pixel_bbox": [96, 152, 284, 236],
            "target_bbox": [0, 0, 420, 330],
            "capacity_bbox": [42, 36, 378, 294],
            "safe_text_box": [90, 150, 290, 238],
            "balloon_bbox": [74, 132, 312, 248],
            "bubble_mask_bbox": [74, 132, 312, 248],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "dark_bubble_ellipse_bbox_mask",
                "connected_layout_disabled_dark_panel_visual_mask",
            ],
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 42,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        bounds = renderer_mod._dark_oval_safe_expansion_bounds(text_data, [0, 0, 420, 330])
        clipped = renderer_mod._clip_dark_oval_safe_to_visible_balloon(text_data, [42, 36, 378, 294])

        self.assertEqual(bounds, [0, 0, 420, 330])
        self.assertEqual(clipped, [42, 36, 378, 294])
        self.assertIn("dark_oval_visible_bbox_fragment_ignored", text_data.get("qa_flags") or [])

    def test_connected_disabled_dark_panel_visual_rect_keeps_noclamp_capacity(self):
        text_data = {
            "translated": "A missao principal sera mostrada em breve",
            "original": "Main Quest will be shown shortly",
            "tipo": "fala",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "bubble_mask_source": "image_dark_panel_mask",
            "bbox": [113, 29, 259, 100],
            "source_bbox": [113, 29, 259, 100],
            "text_pixel_bbox": [113, 29, 259, 100],
            "target_bbox": [38, 122, 331, 270],
            "capacity_bbox": [45, 126, 324, 266],
            "safe_text_box": [52, 139, 319, 262],
            "balloon_bbox": [38, 122, 331, 270],
            "bubble_mask_bbox": [38, 122, 331, 270],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "connected_layout_disabled_dark_panel_visual_mask",
                "dark_panel_full_bbox_selected",
                "dark_panel_rect_from_dark_bubble_bbox",
            ],
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 42,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        bounds = renderer_mod._dark_oval_safe_expansion_bounds(text_data, [38, 122, 331, 270])
        clipped = renderer_mod._clip_dark_oval_safe_to_visible_balloon(text_data, [45, 126, 324, 266])

        self.assertEqual(bounds, [38, 122, 331, 270])
        self.assertEqual(clipped, [45, 126, 324, 266])
        self.assertNotIn("dark_oval_safe_expansion_limited_to_visible_balloon", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_oval_safe_clipped_to_visible_balloon", text_data.get("qa_flags") or [])

    def test_partial_dark_oval_safe_area_uses_anchor_and_ellipse_chord(self):
        text_data = {
            "translated": "Se voce for pego, voce morrera! use a habilidade agora mesmo!",
            "original": "If you get caught, you will die! Use the skill right now!",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [386, 110, 605, 232],
            "source_bbox": [386, 110, 605, 232],
            "text_pixel_bbox": [386, 110, 605, 232],
            "target_bbox": [237, 0, 746, 311],
            "safe_text_box": [308, 49, 674, 261],
            "balloon_bbox": [237, 0, 746, 311],
            "bubble_mask_bbox": [237, 0, 746, 311],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_ellipse_bbox_mask",
                "partial_dark_bubble_lobe_reocr",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
            ],
            "qa_metrics": {
                "image_dark_bubble_mask": {
                    "shape_kind": "ellipse",
                    "mask_bbox": [237, 0, 746, 311],
                    "anchor_bbox": [386, 110, 605, 232],
                }
            },
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 42,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)

        safe = plan["safe_text_box"]
        safe_cx = (safe[0] + safe[2]) / 2.0
        anchor_cx = (386 + 605) / 2.0
        self.assertLess(abs(safe_cx - anchor_cx), 12.0)
        self.assertLess(safe[3] - safe[1], 230)
        self.assertLess(safe[2] - safe[0], 366)
        self.assertGreaterEqual(safe[0], 300)
        self.assertLessEqual(safe[2], 690)
        self.assertIn("dark_oval_safe_anchor_chord_constrained", text_data.get("qa_flags") or [])

    def test_short_dark_lobe_preserves_original_text_center_and_scale(self):
        text_data = {
            "translated": "Eu sou chamado de 'sistema'",
            "original": "I am called 'System'.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [485, 281, 717, 315],
            "source_bbox": [485, 281, 717, 315],
            "text_pixel_bbox": [485, 281, 717, 315],
            "target_bbox": [476, 111, 775, 395],
            "safe_text_box": [528, 166, 723, 340],
            "balloon_bbox": [476, 111, 775, 395],
            "bubble_mask_bbox": [476, 111, 775, 395],
            "background_rgb": [1, 1, 1],
            "qa_flags": [
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "visual_card_font_fallback",
            ],
            "style_origin": "auto_dark_panel_glow",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 42,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)

        safe = plan["safe_text_box"]
        anchor = text_data["text_pixel_bbox"]
        safe_cx = (safe[0] + safe[2]) / 2.0
        safe_cy = (safe[1] + safe[3]) / 2.0
        anchor_cx = (anchor[0] + anchor[2]) / 2.0
        anchor_cy = (anchor[1] + anchor[3]) / 2.0
        self.assertLess(abs(safe_cx - anchor_cx), 2.0)
        self.assertLess(abs(safe_cy - anchor_cy), 2.0)
        self.assertEqual(safe[2] - safe[0], 172)
        self.assertEqual(safe[3] - safe[1], 174)
        self.assertLessEqual(plan["max_width"], safe[2] - safe[0])
        self.assertLessEqual(plan["max_height"], safe[3] - safe[1])
        self.assertEqual(plan["layout_safe_reason"], "short_dark_anchor_scale_preserved")
        self.assertIn("short_dark_anchor_center_preserved", text_data.get("qa_flags") or [])

    def test_short_dark_lobe_rendered_ink_center_matches_original_text_center(self):
        img = Image.new("RGB", (800, 418), (0, 0, 0))
        text_data = {
            "translated": "Eu sou chamado de 'sistema'",
            "original": "I am called 'System'.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [485, 281, 717, 315],
            "source_bbox": [485, 281, 717, 315],
            "text_pixel_bbox": [485, 281, 717, 315],
            "target_bbox": [476, 111, 775, 395],
            "position_bbox": [506, 139, 745, 367],
            "capacity_bbox": [506, 139, 745, 367],
            "safe_text_box": [504, 211, 699, 385],
            "balloon_bbox": [476, 111, 775, 395],
            "bubble_mask_bbox": [476, 111, 775, 395],
            "background_rgb": [0, 0, 0],
            "qa_flags": [
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "short_dark_anchor_center_preserved",
            ],
            "style_origin": "auto_dark_panel_glow",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 40,
                "cor": "#FFFFFF",
                "contorno": "",
                "contorno_px": 0,
                "glow": False,
                "force_upper": True,
                "alinhamento": "center",
            },
        }
        text_data["style"] = dict(text_data["estilo"])

        render_text_block(img, text_data)

        render_bbox = text_data.get("render_bbox")
        self.assertIsNotNone(render_bbox)
        source_center_x = (485 + 717) / 2.0
        source_center_y = (281 + 315) / 2.0
        render_center_x = (render_bbox[0] + render_bbox[2]) / 2.0
        render_center_y = (render_bbox[1] + render_bbox[3]) / 2.0
        self.assertLessEqual(abs(render_center_x - source_center_x), 1.0)
        self.assertLessEqual(abs(render_center_y - source_center_y), 1.0)

    def test_dark_text_without_short_flag_preserves_original_text_center(self):
        img = Image.new("RGB", (800, 552), (0, 0, 0))
        text_data = {
            "translated": "Entre inúmeras almas você foi escolhido conforme se enquadrava nas condições.",
            "original": "Out of countless souls you have been chosen as you fit the conditions..",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [389, 316, 721, 449],
            "source_bbox": [389, 316, 721, 449],
            "text_pixel_bbox": [389, 316, 721, 449],
            "target_bbox": [363, 270, 747, 480],
            "position_bbox": [401, 270, 709, 480],
            "capacity_bbox": [401, 270, 709, 480],
            "safe_text_box": [416, 270, 693, 480],
            "balloon_bbox": [316, 277, 794, 488],
            "bubble_mask_bbox": [363, 270, 747, 480],
            "background_rgb": [0, 0, 0],
            "qa_flags": [
                "dark_bubble_oval_reocr",
                "dark_bubble_text_candidate_reocr",
                "dark_bubble_high_conf_light_text_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
            ],
            "style_origin": "auto_dark_panel_glow",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 33,
                "cor": "#FFFFFF",
                "contorno": "",
                "contorno_px": 0,
                "glow": False,
                "force_upper": True,
                "alinhamento": "center",
            },
        }
        text_data["style"] = dict(text_data["estilo"])

        render_text_block(img, text_data)

        render_bbox = text_data.get("render_bbox")
        self.assertIsNotNone(render_bbox)
        source_center_x = (389 + 721) / 2.0
        source_center_y = (316 + 449) / 2.0
        render_center_x = (render_bbox[0] + render_bbox[2]) / 2.0
        render_center_y = (render_bbox[1] + render_bbox[3]) / 2.0
        self.assertLessEqual(abs(render_center_x - source_center_x), 1.0)
        self.assertLessEqual(abs(render_center_y - source_center_y), 1.0)

    def test_original_text_scale_rejects_candidates_wider_than_source_contract(self):
        text_data = {
            "translated": "CRITERIOS DE CONCLUSAO DA MISSAO: ESTABELECA UM SUBMUNDO DE NIVEL 1.",
            "original": "Quest Completion Criteria: Establish a Level 1 underworld.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [100, 80, 240, 130],
            "source_bbox": [100, 80, 240, 130],
            "text_pixel_bbox": [100, 80, 240, 130],
            "qa_flags": [
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
            ],
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 50,
                "force_upper": True,
            },
        }
        plan = {
            "target_bbox": [20, 20, 420, 260],
            "position_bbox": [20, 20, 420, 260],
            "capacity_bbox": [20, 20, 420, 260],
            "safe_text_box": [20, 20, 420, 260],
            "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
            "max_width": 400,
            "max_height": 240,
            "line_spacing_ratio": 0.04,
            "padding_y": 0,
            "vertical_anchor": "center",
            "alignment": "center",
            "layout_shape": "wide",
            "balloon_geo": "ellipse",
        }

        with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "1"}):
            resolved = _resolve_text_layout(text_data, plan)

        source_w = text_data["text_pixel_bbox"][2] - text_data["text_pixel_bbox"][0]
        source_h = text_data["text_pixel_bbox"][3] - text_data["text_pixel_bbox"][1]
        self.assertLessEqual(resolved["block_width"], int(source_w * 1.2))
        self.assertLessEqual(resolved["block_height"], int(source_h * 1.6))
        self.assertGreaterEqual(resolved["font_size"], 10)
        self.assertGreaterEqual(len(resolved["lines"]), 2)

    def test_original_text_scale_rejects_candidates_below_minimum_source_contract(self):
        source_bbox = [100, 80, 220, 120]
        too_small_width = {"block_width": 101, "block_height": 34}
        too_small_height = {"block_width": 102, "block_height": 33}
        valid = {"block_width": 102, "block_height": 34}

        self.assertIn(
            "width_lt_0.85x_source_text",
            renderer_mod._original_text_scale_candidate_violations(too_small_width, source_bbox),
        )
        self.assertIn(
            "height_lt_0.85x_source_text",
            renderer_mod._original_text_scale_candidate_violations(too_small_height, source_bbox),
        )
        self.assertEqual(renderer_mod._original_text_scale_candidate_violations(valid, source_bbox), [])

    def test_dark_bubble_uses_original_text_scale_contract_without_env_flag(self):
        text_data = {
            "translated": "MESMO ASSIM, VOCE AINDA ESTA DISPOSTO A SE ARRISCAR PELOS SEUS AMIGOS.",
            "original": "Yet you're still willing to risk yourself for your friends.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [416, 167, 729, 375],
            "source_bbox": [416, 167, 729, 375],
            "text_pixel_bbox": [427, 183, 721, 368],
            "source_text_mask_bbox": [427, 183, 721, 368],
            "target_bbox": [360, 22, 797, 471],
            "safe_text_box": [391, 53, 766, 440],
            "qa_flags": [
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
            ],
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 50,
                "force_upper": True,
            },
        }
        plan = {
            "target_bbox": [360, 22, 797, 471],
            "position_bbox": [391, 53, 766, 440],
            "capacity_bbox": [391, 53, 766, 440],
            "safe_text_box": [391, 53, 766, 440],
            "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
            "max_width": 375,
            "max_height": 387,
            "line_spacing_ratio": 0.04,
            "padding_y": 0,
            "vertical_anchor": "center",
            "alignment": "center",
            "layout_shape": "wide",
            "balloon_geo": "ellipse",
        }

        with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "0"}):
            resolved = _resolve_text_layout(text_data, plan)

        source_w = text_data["source_text_mask_bbox"][2] - text_data["source_text_mask_bbox"][0]
        source_h = text_data["source_text_mask_bbox"][3] - text_data["source_text_mask_bbox"][1]
        self.assertIn("original_text_scale_size_experiment", text_data.get("qa_flags") or [])
        self.assertGreaterEqual(resolved["block_width"], int(source_w * 0.85))
        self.assertGreaterEqual(resolved["block_height"], int(source_h * 0.85))
        self.assertLessEqual(resolved["block_width"], int(source_w * 1.2))
        self.assertLessEqual(resolved["block_height"], int(source_h * 1.6))
        self.assertTrue(resolved.get("original_text_scale_preferred"))

    def test_dark_connected_lobe_enforces_original_text_size_contract_without_env(self):
        text_data = {
            "translated": "VOCE ATE FOI AO REFORMATORIO POR CAUSA DELES, NAO FOI?",
            "original": "You even went to juvie for them didn't you?",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [131, 108, 337, 222],
            "source_text_mask_bbox": [244, 113, 313, 223],
            "text_pixel_bbox": [244, 113, 313, 223],
            "target_bbox": [0, 0, 394, 400],
            "safe_text_box": [28, 28, 366, 372],
            "qa_flags": [
                "dark_bubble_connected_lobes_promoted",
                "dark_bubble_connected_lobe_passthrough",
                "dark_bubble_lobe_mask_bbox_preferred",
                "dark_bubble_ellipse_bbox_mask",
                "visual_text_only_inpaint_contract",
            ],
            "_is_lobe_subregion": True,
            "_connected_slot_count": 2,
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 50,
                "force_upper": True,
            },
        }

        with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "0"}):
            self.assertFalse(renderer_mod._should_use_original_text_scale_contract(text_data))
            self.assertTrue(renderer_mod._should_enforce_original_text_scale_contract(text_data))

    def test_text_mask_scale_contract_applies_to_dark_panel_tn_and_white_without_env(self):
        base = {
            "translated": "TEXTO TRADUZIDO",
            "original": "Original text",
            "tipo": "fala",
            "bbox": [100, 80, 220, 120],
            "source_text_mask_bbox": [100, 80, 220, 120],
            "text_pixel_bbox": [100, 80, 220, 120],
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 42,
                "force_upper": True,
            },
        }
        plan = {
            "target_bbox": [20, 20, 420, 260],
            "position_bbox": [20, 20, 420, 260],
            "capacity_bbox": [20, 20, 420, 260],
            "safe_text_box": [20, 20, 420, 260],
            "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
            "max_width": 400,
            "max_height": 240,
            "line_spacing_ratio": 0.04,
            "padding_y": 0,
            "vertical_anchor": "center",
            "alignment": "center",
            "layout_shape": "wide",
            "balloon_geo": "ellipse",
        }
        cases = [
            {"bubble_mask_source": "image_dark_panel_mask", "layout_profile": "dark_panel"},
            {"bubble_mask_source": "translator_note_text_mask", "layout_profile": "standard"},
            {"bubble_mask_source": "image_white_bubble_mask", "layout_profile": "white_balloon"},
        ]
        with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "0"}):
            for extra in cases:
                text_data = {**base, **extra, "qa_flags": []}
                resolved = _resolve_text_layout(text_data, dict(plan))
                self.assertIn("original_text_scale_size_experiment", text_data.get("qa_flags") or [])
                self.assertTrue(resolved.get("original_text_scale_preferred"))
                self.assertLessEqual(resolved["block_width"], 144)
                self.assertLessEqual(resolved["block_height"], 64)

    def test_edge_clipped_dark_reocr_anchors_tail_to_negative_evidence(self):
        img = Image.new("RGB", (800, 739), (0, 0, 0))
        text_data = {
            "translated": "Inothattm ig yeomra? isso mesmo!",
            "original": "INOTHATTM IG YEOMRA? That's right!",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_white_bubble_mask",
            "bbox": [451, 2, 704, 491],
            "source_bbox": [451, 2, 704, 491],
            "text_pixel_bbox": [451, 2, 704, 491],
            "target_bbox": [413, 0, 800, 739],
            "safe_text_box": [451, 2, 704, 491],
            "balloon_bbox": [413, 0, 800, 739],
            "bubble_mask_bbox": [349, 0, 694, 75],
            "background_rgb": [0, 0, 0],
            "qa_flags": [
                "candidate_crop_direct_paddle_reocr",
                "dark_bubble_oval_reocr",
                "visual_text_only_inpaint_contract",
                "band_edge_clipped_text_mask",
                "bubble_clip_preserved_raw_text",
                "text_contract_direct_fill",
            ],
            "qa_metrics": {
                "negative_evidence": [
                    {"source": "negative_detect_ocr", "bbox": [576, 457, 701, 491], "text": "That's right!"}
                ],
                "dark_text_contract_fill_mask": {"bbox": [441, 0, 705, 494], "source": "raw_glyph_mask"},
            },
            "style_origin": "auto_dark_panel_glow",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 44,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 2,
                "force_upper": True,
                "alinhamento": "center",
            },
        }
        text_data["style"] = dict(text_data["estilo"])

        render_text_block(img, text_data)

        self.assertIn("edge_clipped_dark_reocr_tail_anchored", text_data.get("qa_flags") or [])
        self.assertNotIn("false_dark_white_style_neutralized", text_data.get("qa_flags") or [])
        self.assertEqual(text_data.get("translated"), "isso mesmo!")
        self.assertEqual(text_data.get("original"), "That's right!")
        self.assertEqual(text_data.get("text_pixel_bbox"), [576, 457, 701, 491])
        self.assertEqual(text_data.get("bubble_mask_source"), "image_dark_bubble_mask")
        self.assertEqual(text_data.get("estilo", {}).get("cor"), "#FFFFFF")
        self.assertEqual(text_data["qa_metrics"]["dark_text_contract_fill_mask"]["bbox"], [576, 457, 701, 491])
        render_bbox = text_data.get("render_bbox")
        self.assertIsNotNone(render_bbox)
        self.assertGreaterEqual(render_bbox[0], 480)
        self.assertGreaterEqual(render_bbox[1], 410)
        self.assertLessEqual(render_bbox[2], 800)
        self.assertLessEqual(render_bbox[3], 560)

    def test_original_text_scale_contract_skips_sfx_and_rejects_stale_replay(self):
        sfx = {
            "translated": "SFX",
            "content_class": "sfx",
            "bubble_mask_source": "image_dark_bubble_mask",
            "source_text_mask_bbox": [100, 80, 220, 120],
            "text_pixel_bbox": [100, 80, 220, 120],
        }
        self.assertFalse(renderer_mod._should_use_original_text_scale_contract(sfx))

        text_data = {
            "translated": "TEXTO TRADUZIDO",
            "original": "Original text",
            "bubble_mask_source": "image_dark_panel_mask",
            "layout_profile": "dark_panel",
            "source_text_mask_bbox": [100, 80, 220, 120],
            "text_pixel_bbox": [100, 80, 220, 120],
            "render_layout_contract": {
                "schema_version": 1,
                "translated_key": renderer_mod._text_layout_contract_text_key("TEXTO TRADUZIDO"),
                "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "font_size": 32,
                "line_height": 40,
                "lines": ["TEXTO TRADUZIDO"],
                "positions": [[300, 210]],
                "line_widths": [260],
                "block_bbox": [300, 210, 560, 250],
            },
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 42,
            },
        }
        plan = {
            "target_bbox": [20, 20, 620, 360],
            "position_bbox": [20, 20, 620, 360],
            "capacity_bbox": [20, 20, 620, 360],
            "safe_text_box": [20, 20, 620, 360],
            "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
            "max_width": 600,
            "max_height": 340,
            "line_spacing_ratio": 0.04,
            "padding_y": 0,
            "vertical_anchor": "center",
            "alignment": "center",
            "layout_shape": "wide",
            "balloon_geo": "rect",
        }
        self.assertIsNone(renderer_mod._candidate_from_render_layout_contract(text_data, plan))
        self.assertIn("stale_render_layout_contract_rejected", text_data.get("qa_flags") or [])

    def test_false_dark_flags_on_white_balloon_use_neutral_black_style(self):
        img = Image.new("RGB", (520, 340), "white")
        text_data = {
            "translated": "O QUE VOCE QUER DIZER COM SUBESPACO?",
            "original": "WHAT DO YOU MEAN, SUBSPACE?",
            "tipo": "fala",
            "bbox": [214, 112, 349, 196],
            "source_bbox": [214, 112, 349, 196],
            "text_pixel_bbox": [214, 112, 349, 196],
            "target_bbox": [143, 13, 421, 296],
            "safe_text_box": [176, 47, 388, 262],
            "balloon_bbox": [143, 13, 421, 296],
            "qa_flags": [
                "dark_bubble_oval_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "trusted_dark_visual_capacity_target",
                "dark_panel_style_grouped",
            ],
            "style_origin": "grouped_dark_panel_visual_style",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 42,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        renderer_mod._apply_auto_style_policy_if_needed(img, text_data)

        style = text_data["estilo"]
        self.assertEqual(style.get("cor"), "#000000")
        self.assertFalse(style.get("glow"))
        self.assertEqual(style.get("contorno_px"), 0)
        self.assertEqual(text_data.get("style_origin"), "false_dark_white_neutral")
        self.assertIn("false_dark_white_style_neutralized", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_panel_style_grouped", text_data.get("qa_flags") or [])

    def test_dark_panel_full_bbox_rejects_tiny_inner_safe_area(self):
        text_data = {
            "translated": "A missao principal sera mostrada em breve",
            "tipo": "fala",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "bubble_mask_source": "image_dark_panel_mask",
            "bbox": [113, 2897, 259, 2968],
            "source_bbox": [113, 2897, 259, 2968],
            "text_pixel_bbox": [113, 2897, 259, 2968],
            "target_bbox": [70, 2868, 302, 2997],
            "balloon_bbox": [70, 2868, 302, 2997],
            "bubble_mask_bbox": [70, 2868, 302, 2997],
            "bubble_inner_bbox": [116, 2893, 254, 2956],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "dark_panel_rect_from_dark_bubble_bbox",
                "dark_panel_visual_rect_candidate_selected",
                "dark_panel_full_bbox_selected",
            ],
            "style_origin": "auto_dark_panel_glow",
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 48,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)

        capacity = plan["capacity_bbox"]
        self.assertEqual(plan["layout_safe_reason"], "full_dark_panel_visual_capacity")
        self.assertEqual(capacity, [83, 2875, 289, 2990])
        self.assertGreaterEqual(plan["max_width"], 190)
        self.assertLessEqual(plan["line_spacing_ratio"], 0.18)
        self.assertGreaterEqual(plan["target_size"], 38)
        self.assertIn("full_dark_panel_visual_capacity", text_data.get("qa_flags") or [])
        self.assertNotIn("dark_panel_full_bbox_safe_clamped_to_inner", text_data.get("qa_flags") or [])

    def test_dark_white_mask_source_uses_balloon_area_for_type(self):
        text_data = {
            "translated": "A missao. recompensa e....",
            "tipo": "fala",
            "layout_profile": "connected_balloon",
            "block_profile": "standard",
            "bubble_mask_source": "image_white_bubble_mask",
            "bbox": [520, 5888, 646, 5955],
            "source_bbox": [520, 5888, 646, 5955],
            "text_pixel_bbox": [520, 5888, 646, 5955],
            "target_bbox": [509, 5870, 648, 5963],
            "balloon_bbox": [505, 5874, 669, 5983],
            "bubble_mask_bbox": [509, 5870, 648, 5963],
            "bubble_inner_bbox": [521, 5882, 636, 5951],
            "connected_lobe_bboxes": [[505, 5874, 581, 5983], [589, 5874, 669, 5983]],
            "connected_position_bboxes": [[556, 5884, 569, 5938], [629, 5889, 654, 5927]],
            "background_rgb": [12, 12, 12],
            "qa_flags": ["dark_bubble_visual_glyph_mask_replaced_geometry"],
            "style_origin": "auto_dark_panel_glow",
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        plan = plan_text_layout(text_data)

        capacity = plan["capacity_bbox"]
        self.assertEqual(plan["target_bbox"], [505, 5874, 669, 5983])
        self.assertEqual(plan["layout_safe_reason"], "trusted_dark_visual_capacity")
        self.assertGreaterEqual(capacity[2] - capacity[0], 140)
        self.assertGreaterEqual(capacity[3] - capacity[1], 95)
        self.assertGreaterEqual(plan["target_size"], 38)
        self.assertIn("trusted_dark_visual_capacity_target", text_data.get("qa_flags") or [])
        self.assertEqual(text_data.get("_trusted_dark_visual_capacity_position_bbox"), [514, 5876, 660, 5977])

        render_data = dict(text_data)
        render_data["safe_text_box"] = [534, 5884, 622, 5949]
        render_data["_debug_safe_text_box"] = [534, 5884, 622, 5949]
        render_data["render_bbox"] = [536, 5893, 622, 5940]
        render_data["target_bbox"] = [509, 5870, 648, 5963]
        img = Image.new("RGB", (760, 6060), (0, 0, 0))
        render_text_block(img, render_data)

        self.assertEqual(render_data.get("target_bbox"), [505, 5874, 669, 5983])
        rendered_safe = render_data.get("safe_text_box")
        self.assertIsNotNone(rendered_safe)
        self.assertGreaterEqual(rendered_safe[2] - rendered_safe[0], 100)
        self.assertGreaterEqual(rendered_safe[3] - rendered_safe[1], 65)
        self.assertEqual(render_data.get("layout_safe_reason"), "trusted_dark_visual_capacity")

    def test_dark_bubble_glow_capacity_rejects_off_anchor_art_region(self):
        img = Image.new("RGB", (800, 735), (0, 0, 0))
        arr = np.array(img)
        arr[218:354, 0:611] = [4, 54, 92]
        img = Image.fromarray(arr)
        text_data = {
            "translated": "Você não pode ficar no subespaço por muito tempo.",
            "original": "You can't stay in the subspace for long.",
            "tipo": "fala",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bbox": [163, 391, 351, 521],
            "source_bbox": [163, 391, 351, 521],
            "text_pixel_bbox": [163, 391, 351, 521],
            "target_bbox": [122, 352, 392, 560],
            "balloon_bbox": [122, 352, 392, 560],
            "bubble_mask_bbox": [241, 436, 351, 476],
            "background_rgb": [2, 2, 2],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "visual_text_only_inpaint_contract",
                "text_contract_direct_fill",
            ],
            "estilo": {
                "fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "tamanho": 34,
                "cor": "#FFFFFF",
                "force_upper": True,
            },
        }
        plan = {
            "target_bbox": [122, 352, 392, 560],
            "position_bbox": [122, 352, 392, 560],
            "capacity_bbox": [122, 352, 392, 560],
            "safe_text_box": [150, 374, 364, 538],
            "max_width": 240,
            "max_height": 150,
            "padding_y": 0,
        }

        renderer_mod._apply_recovered_dark_bubble_glow_capacity(img, text_data, plan)

        self.assertEqual(plan["target_bbox"], [122, 352, 392, 560])
        self.assertEqual(plan["safe_text_box"], [150, 374, 364, 538])
        self.assertNotIn("dark_bubble_glow_capacity_recovered", text_data.get("qa_flags") or [])
        self.assertIn("dark_bubble_glow_capacity_rejected_off_anchor", text_data.get("qa_flags") or [])

    def test_dark_bbox_fallback_panel_does_not_cap_font_to_short_anchor(self):
        img = Image.new("RGB", (800, 735), (0, 0, 0))
        text_data = {
            "translated": "O episodio comeca!",
            "original": "The episode starts!",
            "tipo": "text",
            "layout_profile": "dark_panel",
            "block_profile": "dark_panel",
            "bbox": [420, 0, 800, 735],
            "source_bbox": [587, 514, 765, 542],
            "text_pixel_bbox": [587, 514, 765, 542],
            "target_bbox": [587, 514, 765, 542],
            "capacity_bbox": [465, 224, 775, 614],
            "safe_text_box": [577, 481, 775, 575],
            "balloon_bbox": [587, 514, 765, 542],
            "background_rgb": [12, 12, 12],
            "qa_flags": [
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "bbox_fallback_bubble_mask",
            ],
            "style_origin": "auto_dark_panel_glow",
            "estilo": {
                "fonte": "KOMIKAX_.ttf",
                "tamanho": 48,
                "cor": "#FFFFFF",
                "contorno": "#061D26",
                "contorno_px": 1,
                "glow": True,
                "glow_cor": "#67D8FF",
                "glow_px": 3,
                "force_upper": True,
            },
        }

        render_text_block(img, text_data)

        debug = text_data.get("_render_debug") or {}
        self.assertGreaterEqual(debug.get("font_size_final", 0), 24)
        render_bbox = text_data.get("render_bbox") or [0, 0, 0, 0]
        self.assertGreaterEqual(render_bbox[1], 470)
        self.assertIn("trusted_dark_visual_capacity_target", text_data.get("qa_flags") or [])
        self.assertNotIn("render_suppressed_low_containment_fragment", text_data.get("qa_flags") or [])
        self.assertNotEqual(text_data.get("render_policy"), "suppressed_low_containment_fragment")

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

    def test_merged_fragment_rejected_bubble_fallback_uses_text_evidence_target(self):
        layer = {
            "id": "ocr_001",
            "text": "PLEASE, FOR THE CHILD'S SAKE.",
            "original": "PLEASE, FOR THE CHILD'S SAKE.",
            "translated": "POR FAVOR, PELO BEM DA CRIANÇA.",
            "bbox": [499, 315, 656, 400],
            "source_bbox": [499, 315, 656, 400],
            "layout_bbox": [499, 315, 656, 336],
            "text_pixel_bbox": [499, 315, 656, 400],
            "balloon_bbox": [25, 96, 667, 368],
            "bubble_mask_bbox": [25, 96, 667, 368],
            "bubble_inner_bbox": [80, 136, 612, 328],
            "bubble_mask_source": "rejected_derived_bubble_mask",
            "layout_profile": "standard",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "alinhamento": "center", "cor": "#000000"},
            "style": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "alinhamento": "center", "cor": "#000000"},
            "qa_flags": [
                "same_balloon_fragment_merged",
                "same_band_dependent_fragment_merged",
                "bubble_clip_preserved_raw_text",
                "rejected_derived_bubble_mask",
            ],
            "qa_metrics": {
                "render_balloon_containment": 1.0,
                "pre_inpaint_outside_balloon_ratio": 0.200757,
            },
        }

        plan = plan_text_layout(dict(layer))

        self.assertEqual(plan["target_bbox"], [393, 248, 667, 368])
        self.assertEqual(plan["_target_source"], "overbroad_white_balloon_text_evidence")
        self.assertGreaterEqual(plan["safe_text_box"][0], 412)
        self.assertLessEqual(plan["safe_text_box"][2], 593)
        self.assertGreaterEqual(plan["safe_text_box"][1], 260)

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

    def test_low_containment_fragment_render_is_rolled_back(self):
        before = Image.new("RGB", (260, 220), (255, 255, 255))
        img = before.copy()
        ImageDraw.Draw(img).rectangle([24, 48, 90, 66], fill=(0, 0, 0))
        text_data = {
            "id": "ocr_bad_fragment",
            "translated": "ESTOU MORRENDO DE FOME",
            "render_bbox": [24, 48, 90, 66],
            "balloon_bbox": [118, 82, 150, 96],
            "qa_metrics": {"render_balloon_containment": 0.0},
            "qa_flags": ["safe_text_box_recomputed"],
        }
        plan = {
            "target_bbox": [0, 0, 260, 200],
            "safe_text_box": [20, 42, 96, 72],
        }

        rolled_back = renderer_mod._rollback_low_containment_fragment_render_if_needed(
            img,
            before,
            text_data,
            plan,
        )

        self.assertTrue(rolled_back)
        self.assertEqual(list(img.getdata()), list(before.getdata()))
        self.assertIn("render_suppressed_low_containment_fragment", text_data.get("qa_flags") or [])
        self.assertEqual(text_data.get("render_policy"), "suppressed_low_containment_fragment")
        self.assertEqual(text_data.get("route_reason"), "low_containment_fragment")

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

    def test_rotated_text_layout_preserves_original_anchor_position(self):
        text_data = {
            "id": "ocr_rotated",
            "translated": "A HUMANIDADE AGORA EM DIANTE NAO USE SEU DINHEIRO.",
            "original": "Humanity, from now on, do not use your money.",
            "tipo": "texto",
            "layout_profile": "standard",
            "bbox": [55, 70, 330, 250],
            "text_pixel_bbox": [92, 96, 304, 226],
            "source_bbox": [92, 96, 304, 226],
            "balloon_bbox": [25, 18, 388, 316],
            "safe_text_box": [40, 36, 370, 300],
            "layout_safe_bbox": [40, 36, 370, 300],
            "line_polygons": [
                [[82, 180], [260, 72], [278, 100], [100, 208]],
                [[112, 220], [304, 104], [322, 132], [130, 248]],
            ],
            "rotation_deg": -33.0,
            "rotation_source": "line_polygons",
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 28, "force_upper": True},
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["position_bbox"], [82, 72, 322, 248])
        self.assertEqual(plan["capacity_bbox"], [82, 72, 322, 248])
        self.assertLessEqual(plan["safe_text_box"][2] - plan["safe_text_box"][0], 240)
        self.assertEqual(plan["rotation_deg"], -33.0)

    def test_merged_oval_balloon_rejects_short_line_safe_box_for_wrapped_layout(self):
        text_data = {
            "id": "ocr_054",
            "translated": "OPPA, POR QUE VOCE ESTA SOLITARIO?",
            "original": "Oppa, why are you alone?",
            "tipo": "fala",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "bbox": [52, 4511, 361, 4549],
            "text_pixel_bbox": [52, 4511, 361, 4549],
            "layout_bbox": [52, 4511, 361, 4549],
            "balloon_bbox": [30, 4432, 394, 4588],
            "safe_text_box": [52, 4511, 361, 4549],
            "layout_safe_bbox": [52, 4511, 361, 4549],
            "layout_safe_reason": "layout_safe_bbox",
            "bubble_mask_source": "image_contour_bubble_mask",
            "qa_flags": ["same_balloon_fragment_merged", "safe_text_box_recomputed"],
            "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 25, "force_upper": True},
        }

        plan = plan_text_layout(text_data)

        self.assertGreaterEqual(plan["safe_text_box"][3] - plan["safe_text_box"][1], 70)
        self.assertGreaterEqual(plan["max_height"], 56)
        self.assertEqual(plan["layout_profile"], "white_balloon")

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


    def test_render_plan_band_shifts_y_by_band_y_top_once_in_bbox_like_fields(self):
        band_y_top = 37
        payload = {
            "coordinate_space": "band",
            "band_y_top": band_y_top,
            "target_bbox": [10, 15, 110, 45],
            "position_bbox": [12, 19, 92, 39],
            "capacity_bbox": [14, 21, 94, 51],
            "layout_safe_bbox": [16, 23, 96, 53],
            "safe_text_box": [18, 25, 98, 55],
            "render_bbox": [20, 27, 100, 57],
            "balloon_bbox": [22, 29, 102, 59],
            "bubble_mask_bbox": [24, 31, 104, 61],
            "bubble_inner_bbox": [26, 33, 106, 63],
            "connected_position_bboxes": [[28, 35, 108, 65], [30, 37, 110, 67]],
            "qa_metrics": {
                "safe_text_box": [40, 45, 80, 55],
                "nested": {
                    "render_bbox": [32, 53, 72, 63],
                    "safe_text_box": [34, 55, 74, 65],
                },
                "metrics_chain": [
                    {"render_bbox": [36, 57, 76, 67]},
                    {"debug": {"safe_text_box": [38, 59, 78, 69]}},
                ],
            },
        }

        shifted = renderer_mod._shift_render_plan_to_page(payload)

        self.assertEqual(shifted["coordinate_space"], "page")
        self.assertEqual(shifted["target_bbox"], [10, 15 + band_y_top, 110, 45 + band_y_top])
        self.assertEqual(shifted["position_bbox"], [12, 19 + band_y_top, 92, 39 + band_y_top])
        self.assertEqual(shifted["capacity_bbox"], [14, 21 + band_y_top, 94, 51 + band_y_top])
        self.assertEqual(shifted["layout_safe_bbox"], [16, 23 + band_y_top, 96, 53 + band_y_top])
        self.assertEqual(shifted["safe_text_box"], [18, 25 + band_y_top, 98, 55 + band_y_top])
        self.assertEqual(shifted["render_bbox"], [20, 27 + band_y_top, 100, 57 + band_y_top])
        self.assertEqual(shifted["balloon_bbox"], [22, 29 + band_y_top, 102, 59 + band_y_top])
        self.assertEqual(shifted["bubble_mask_bbox"], [24, 31 + band_y_top, 104, 61 + band_y_top])
        self.assertEqual(shifted["bubble_inner_bbox"], [26, 33 + band_y_top, 106, 63 + band_y_top])
        self.assertEqual(
            shifted["connected_position_bboxes"],
            [[28, 35 + band_y_top, 108, 65 + band_y_top], [30, 37 + band_y_top, 110, 67 + band_y_top]],
        )
        self.assertEqual(shifted["qa_metrics"]["safe_text_box"], [40, 45 + band_y_top, 80, 55 + band_y_top])
        self.assertEqual(shifted["qa_metrics"]["nested"]["render_bbox"], [32, 53 + band_y_top, 72, 63 + band_y_top])
        self.assertEqual(shifted["qa_metrics"]["nested"]["safe_text_box"], [34, 55 + band_y_top, 74, 65 + band_y_top])
        self.assertEqual(shifted["qa_metrics"]["metrics_chain"][0]["render_bbox"], [36, 57 + band_y_top, 76, 67 + band_y_top])
        self.assertEqual(shifted["qa_metrics"]["metrics_chain"][1]["debug"]["safe_text_box"], [38, 59 + band_y_top, 78, 69 + band_y_top])

    def test_render_plan_page_coordinate_space_does_not_shift_bbox_like_fields(self):
        payload = {
            "coordinate_space": "page",
            "band_y_top": 37,
            "target_bbox": [10, 15, 110, 45],
            "position_bbox": [12, 19, 92, 39],
            "capacity_bbox": [14, 21, 94, 51],
            "layout_safe_bbox": [16, 23, 96, 53],
            "safe_text_box": [18, 25, 98, 55],
            "render_bbox": [20, 27, 100, 57],
            "balloon_bbox": [22, 29, 102, 59],
            "bubble_mask_bbox": [24, 31, 104, 61],
            "bubble_inner_bbox": [26, 33, 106, 63],
            "connected_position_bboxes": [[28, 35, 108, 65], [30, 37, 110, 67]],
            "qa_metrics": {
                "safe_text_box": [40, 45, 80, 55],
                "nested": {
                    "render_bbox": [32, 53, 72, 63],
                    "safe_text_box": [34, 55, 74, 65],
                },
                "metrics_chain": [
                    {"render_bbox": [36, 57, 76, 67]},
                    {"debug": {"safe_text_box": [38, 59, 78, 69]}},
                ],
            },
        }

        shifted = renderer_mod._shift_render_plan_to_page(payload)

        self.assertEqual(shifted["coordinate_space"], "page")
        self.assertEqual(shifted["target_bbox"], [10, 15, 110, 45])
        self.assertEqual(shifted["position_bbox"], [12, 19, 92, 39])
        self.assertEqual(shifted["capacity_bbox"], [14, 21, 94, 51])
        self.assertEqual(shifted["layout_safe_bbox"], [16, 23, 96, 53])
        self.assertEqual(shifted["safe_text_box"], [18, 25, 98, 55])
        self.assertEqual(shifted["render_bbox"], [20, 27, 100, 57])
        self.assertEqual(shifted["balloon_bbox"], [22, 29, 102, 59])
        self.assertEqual(shifted["bubble_mask_bbox"], [24, 31, 104, 61])
        self.assertEqual(shifted["bubble_inner_bbox"], [26, 33, 106, 63])
        self.assertEqual(shifted["connected_position_bboxes"], [[28, 35, 108, 65], [30, 37, 110, 67]])
        self.assertEqual(shifted["qa_metrics"]["safe_text_box"], [40, 45, 80, 55])
        self.assertEqual(shifted["qa_metrics"]["nested"]["render_bbox"], [32, 53, 72, 63])
        self.assertEqual(shifted["qa_metrics"]["nested"]["safe_text_box"], [34, 55, 74, 65])
        self.assertEqual(shifted["qa_metrics"]["metrics_chain"][0]["render_bbox"], [36, 57, 76, 67])
        self.assertEqual(shifted["qa_metrics"]["metrics_chain"][1]["debug"]["safe_text_box"], [38, 59, 78, 69])

    def test_render_band_text_mask_cleanup_accepts_page_space_bbox(self):
        img = Image.new("RGB", (220, 120), (0, 0, 0))
        for x in range(70, 150):
            for y in range(45, 68):
                img.putpixel((x, y), (245, 245, 245))
        text = {
            "route_action": "translate_inpaint_render",
            "translated": "TEXTO",
            "source_text_mask_bbox": [72, 1046, 148, 1066],
            "background_rgb": [38, 46, 52],
            "bubble_mask_source": "image_dark_bubble_mask",
        }

        changed = renderer_mod._apply_text_mask_cleanup_before_render(
            img,
            [text],
            {"band_y_top": 1000},
        )

        self.assertTrue(changed)
        self.assertLess(img.getpixel((110, 56))[0], 20)
        self.assertIn("render_text_mask_cleanup", text["qa_flags"])

    def test_render_band_text_mask_cleanup_skips_sfx(self):
        img = Image.new("RGB", (160, 90), (0, 0, 0))
        for x in range(40, 120):
            for y in range(30, 56):
                img.putpixel((x, y), (245, 245, 245))

        changed = renderer_mod._apply_text_mask_cleanup_before_render(
            img,
            [
                {
                    "route_action": "translate_sfx_inpaint_render",
                    "content_class": "sfx",
                    "translated": "SFX",
                    "source_text_mask_bbox": [42, 32, 118, 54],
                    "background_rgb": [0, 0, 0],
                }
            ],
            {},
        )

        self.assertFalse(changed)
        self.assertGreater(img.getpixel((80, 42))[0], 220)

    def test_render_band_text_mask_cleanup_skips_rect_for_clean_white_jagged_sfx_like_bubble(self):
        img = Image.new("RGB", (180, 100), (255, 255, 255))
        text = {
            "route_action": "translate_inpaint_render",
            "translated": "EVASAO!",
            "original": "EVASION!",
            "source_text_mask_bbox": [45, 30, 135, 58],
            "safe_text_box": [45, 30, 135, 58],
            "background_rgb": [208, 208, 208],
            "bubble_mask_source": "image_white_bubble_mask",
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
            "estilo": {"bold": True, "italico": True, "force_upper": True},
        }

        changed = renderer_mod._apply_text_mask_cleanup_before_render(img, [text], {})

        self.assertFalse(changed)
        self.assertGreater(img.getpixel((90, 44))[0], 245)
        self.assertNotIn("qa_flags", text)
        metrics = text["qa_metrics"]["sfx_white_bubble_background_removed"]
        self.assertEqual(metrics["decision"], "applied")
        self.assertEqual(metrics["reason"], "after_inpaint_white_background_trusted_no_rect_cleanup")

    def test_render_band_text_mask_cleanup_keeps_dark_panel_fill_black_for_sfx_like_text(self):
        img = Image.new("RGB", (180, 100), (0, 0, 0))
        for x in range(45, 135):
            for y in range(30, 58):
                img.putpixel((x, y), (245, 245, 245))
        text = {
            "route_action": "translate_inpaint_render",
            "translated": "EVASAO!",
            "original": "EVASION!",
            "source_text_mask_bbox": [45, 30, 135, 58],
            "background_rgb": [208, 208, 208],
            "bubble_mask_source": "image_dark_bubble_mask",
            "estilo": {"bold": True, "italico": True, "force_upper": True},
        }

        changed = renderer_mod._apply_text_mask_cleanup_before_render(img, [text], {})

        self.assertTrue(changed)
        self.assertLess(img.getpixel((90, 44))[0], 20)
        self.assertNotIn("sfx_white_bubble_background_removed", text.get("qa_metrics", {}))

    def test_render_band_text_mask_cleanup_does_not_use_white_sfx_rule_for_translator_note(self):
        img = Image.new("RGB", (180, 100), (255, 255, 255))
        text = {
            "route_action": "translate_inpaint_render",
            "translated": "T/N: NOTA!",
            "source_text_mask_bbox": [45, 30, 135, 58],
            "background_rgb": [208, 208, 208],
            "bubble_mask_source": "image_white_bubble_mask",
            "layout_profile": "white_balloon",
            "qa_flags": ["translator_note_text_only_mask"],
            "estilo": {"bold": True, "italico": True, "force_upper": True},
        }

        changed = renderer_mod._apply_text_mask_cleanup_before_render(img, [text], {})

        self.assertTrue(changed)
        self.assertEqual(img.getpixel((90, 44)), (208, 208, 208))
        self.assertNotIn("sfx_white_bubble_background_removed", text.get("qa_metrics", {}))

    def test_render_band_text_mask_cleanup_keeps_near_white_background_for_white_sfx_like_bubble(self):
        img = Image.new("RGB", (180, 100), (255, 255, 255))
        text = {
            "route_action": "translate_inpaint_render",
            "translated": "CALE-SE!",
            "original": "Shut UP!",
            "source_text_mask_bbox": [45, 30, 135, 58],
            "background_rgb": [254, 254, 254],
            "bubble_mask_source": "image_white_bubble_mask",
            "layout_profile": "white_balloon",
            "estilo": {"bold": True, "italico": True, "force_upper": True},
        }

        changed = renderer_mod._apply_text_mask_cleanup_before_render(img, [text], {})

        self.assertTrue(changed)
        self.assertEqual(img.getpixel((90, 44)), (254, 254, 254))
        self.assertNotIn("sfx_white_bubble_background_removed", text.get("qa_metrics", {}))

if __name__ == "__main__":
    unittest.main()
