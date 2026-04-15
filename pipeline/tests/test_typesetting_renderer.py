import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from typesetter.renderer import SafeTextPathFont, _build_textpath_mask, build_render_blocks, render_text_block


class TypesettingRendererTests(unittest.TestCase):
    def test_build_textpath_mask_renders_project_font(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        font = SafeTextPathFont(fonts_dir / "CCDaveGibbonsLower W00 Regular.ttf", 28)

        mask = _build_textpath_mask(font, "TEST 123?!", padding=2)

        self.assertEqual(mask.ndim, 2)
        self.assertGreater(mask.shape[1], 30)
        self.assertGreater(mask.shape[0], 10)
        self.assertGreater(int(np.max(mask)), 0)

    def test_build_textpath_mask_preserves_glyph_holes(self):
        fonts_dir = Path(__file__).resolve().parents[2] / "fonts"
        font = SafeTextPathFont(fonts_dir / "CCDaveGibbonsLower W00 Regular.ttf", 64)

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
                "fonte": "CCDaveGibbonsLower W00 Regular.ttf",
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
                "fonte": "CCDaveGibbonsLower W00 Regular.ttf",
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
                "fonte": "CCDaveGibbonsLower W00 Regular.ttf",
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

    def test_two_texts_with_subregions_no_double_render(self):
        """2 textos no mesmo balão com subregions → cada texto vai para seu lobo, sem duplicação."""
        subregions = [[30, 40, 200, 220], [220, 40, 400, 220]]
        base_style = {
            "fonte": "CCDaveGibbonsLower W00 Regular.ttf",
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
        self.assertEqual(len(blocks), 2, "Deve gerar 2 blocos, um por lobo")

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
                "fonte": "CCDaveGibbonsLower W00 Regular.ttf",
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
