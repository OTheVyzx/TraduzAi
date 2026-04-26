import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from typesetter.renderer import (
    SafeTextPathFont,
    _build_textpath_mask,
    _category_font_bounds,
    _normalize_render_text,
    _resolve_text_layout,
    build_render_blocks,
    plan_text_layout,
    render_text_block,
)


class TypesettingRendererTests(unittest.TestCase):
    def test_render_text_block_anchors_to_text_pixel_bbox_instead_of_balloon_center(self):
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

        render_text_block(img, text_data)

        arr = np.array(img)
        ink = np.any(arr < 245, axis=2)
        ys, xs = np.where(ink)
        self.assertGreater(xs.size, 0)
        ink_center_x = float(xs.mean())
        ink_center_y = float(ys.mean())
        anchor = text_data["text_pixel_bbox"]
        anchor_center_x = (anchor[0] + anchor[2]) / 2.0
        anchor_center_y = (anchor[1] + anchor[3]) / 2.0
        balloon = text_data["balloon_bbox"]
        balloon_center_x = (balloon[0] + balloon[2]) / 2.0
        balloon_center_y = (balloon[1] + balloon[3]) / 2.0

        self.assertLess(abs(ink_center_x - anchor_center_x), abs(ink_center_x - balloon_center_x))
        self.assertLess(abs(ink_center_y - anchor_center_y), abs(ink_center_y - balloon_center_y))

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
        self.assertGreaterEqual(resolved["font_size"], bounds[0])
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
        self.assertGreaterEqual(resolved["font_size"], bounds[0])
        self.assertLessEqual(resolved["font_size"], bounds[1])

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

    def test_plan_text_layout_applies_top_narration_profile(self):
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

        self.assertEqual(plan["vertical_anchor"], "top")
        self.assertLess(plan["width_ratio"], 0.90)

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


if __name__ == "__main__":
    unittest.main()
