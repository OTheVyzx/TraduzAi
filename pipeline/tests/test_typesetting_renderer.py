import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from typesetter.renderer import SafeTextPathFont, _build_textpath_mask, render_text_block


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


if __name__ == "__main__":
    unittest.main()
