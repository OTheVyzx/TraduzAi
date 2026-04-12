import unittest

from PIL import Image

from typesetter.renderer import (
    _resolve_text_layout,
    _split_text_for_connected_balloons,
    build_render_blocks,
    ensure_legible_plan,
    plan_text_layout,
)


class TypesettingLayoutTests(unittest.TestCase):
    def test_groups_texts_that_share_same_balloon(self):
        texts = [
            {
                "translated": "Ola! voce esta pronto",
                "bbox": [120, 200, 260, 250],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [100, 180, 280, 420],
                "layout_shape": "tall",
                "layout_align": "center",
                "layout_group_size": 2,
            },
            {
                "translated": "Para a batalha?",
                "bbox": [130, 260, 250, 300],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [100, 180, 280, 420],
                "layout_shape": "tall",
                "layout_align": "center",
                "layout_group_size": 2,
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 1)
        self.assertIn("Ola! voce esta pronto", blocks[0]["translated"])
        self.assertIn("Para a batalha?", blocks[0]["translated"])
        self.assertEqual(blocks[0]["estilo"]["contorno_px"], 2)
        self.assertEqual(blocks[0]["estilo"]["contorno"], "#000000")

    def test_tall_balloon_uses_narrower_wrap_width(self):
        style = {"tamanho": 24, "alinhamento": "center"}
        text_data = {
            "translated": "I will never forgive you for this",
            "bbox": [120, 200, 260, 250],
            "tipo": "fala",
            "estilo": style,
            "balloon_bbox": [100, 180, 280, 420],
            "layout_shape": "tall",
            "layout_align": "center",
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["target_bbox"], [100, 180, 280, 420])
        self.assertLess(plan["max_width"], 180)

    def test_wide_narration_anchors_near_top(self):
        style = {"tamanho": 26, "alinhamento": "center"}
        text_data = {
            "translated": "Three days later...",
            "bbox": [220, 80, 860, 180],
            "tipo": "narracao",
            "estilo": style,
            "balloon_bbox": [200, 60, 880, 210],
            "layout_shape": "wide",
            "layout_align": "top",
        }

        plan = plan_text_layout(text_data)

        self.assertEqual(plan["vertical_anchor"], "top")
        self.assertGreater(plan["max_width"], 500)

    def test_corpus_benchmark_reduces_size_and_strengthens_outline(self):
        style = {"tamanho": 24, "alinhamento": "center", "contorno_px": 1}
        text_data = {
            "translated": "This sentence tends to expand in Portuguese",
            "bbox": [120, 200, 260, 250],
            "tipo": "fala",
            "estilo": style,
            "balloon_bbox": [100, 180, 280, 420],
            "layout_shape": "tall",
            "layout_align": "center",
            "corpus_visual_benchmark": {
                "page_geometry": {"median_width": 800, "median_aspect_ratio": 0.32}
            },
            "corpus_textual_benchmark": {
                "paired_text_stats": {"mean_translation_length_ratio": 1.25}
            },
        }

        plan = plan_text_layout(text_data)

        self.assertLess(plan["target_size"], 24)
        self.assertGreaterEqual(plan["outline_px"], 2)
        self.assertLess(plan["max_width"], 140)

    def test_legibility_fallback_avoids_white_text_on_white_balloon(self):
        img = Image.new("RGB", (320, 240), (250, 250, 250))
        plan = {
            "target_bbox": [40, 40, 280, 180],
            "text_color": "#FFFFFF",
            "cor_gradiente": [],
            "outline_color": "",
            "outline_px": 0,
            "glow": True,
            "glow_cor": "#F7F7F7",
            "glow_px": 3,
        }

        adjusted = ensure_legible_plan(img, plan)

        self.assertEqual(adjusted["text_color"], "#111111")
        self.assertGreaterEqual(adjusted["outline_px"], 2)
        self.assertFalse(adjusted["glow"])

    def test_legibility_fallback_avoids_dark_text_on_dark_balloon(self):
        img = Image.new("RGB", (320, 240), (20, 20, 20))
        plan = {
            "target_bbox": [40, 40, 280, 180],
            "text_color": "#111111",
            "cor_gradiente": [],
            "outline_color": "",
            "outline_px": 0,
            "glow": False,
            "glow_cor": "",
            "glow_px": 0,
        }

        adjusted = ensure_legible_plan(img, plan)

        self.assertEqual(adjusted["text_color"], "#F5F5F5")
        self.assertEqual(adjusted["outline_color"], "#000000")
        self.assertGreaterEqual(adjusted["outline_px"], 2)

    def test_resolve_text_layout_balances_occupancy_and_centering(self):
        text_data = {
            "translated": "YOU SAID YOU COULD SEE THROUGH ALL MY ATTACKS, RIGHT?",
            "bbox": [206, 2172, 610, 2301],
            "tipo": "fala",
            "estilo": {
                "fonte": "CCDaveGibbonsLower W00 Regular.ttf",
                "tamanho": 48,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
            },
            "balloon_bbox": [170, 2115, 648, 2358],
            "layout_shape": "wide",
            "layout_align": "center",
        }

        layout = _resolve_text_layout(text_data, plan_text_layout(text_data))
        x1, y1, x2, y2 = text_data["balloon_bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bx1, by1, bx2, by2 = layout["block_bbox"]
        block_cx = (bx1 + bx2) / 2.0
        block_cy = (by1 + by2) / 2.0

        self.assertGreater(layout["width_ratio"], 0.45)
        self.assertLess(layout["width_ratio"], 0.88)
        self.assertGreater(layout["height_ratio"], 0.20)
        self.assertLess(layout["height_ratio"], 0.78)
        self.assertLess(abs(block_cx - cx), 8.0)
        self.assertLess(abs(block_cy - cy), 8.0)

    def test_split_text_for_connected_balloons_prefers_sentence_boundaries(self):
        chunks = _split_text_for_connected_balloons(
            "IT MAY BE NOTHING MORE THAN A HALF-FINISHED METHOD, BUT IT'S EFFECTS ARE MORE THAN ENOUGH. A POWER THAT LET'S YOU SURPASS YOUR OWN LIMITS IN AN INSTANT.",
            2,
        )

        self.assertEqual(len(chunks), 2)
        self.assertIn("MORE THAN ENOUGH.", chunks[0])
        self.assertTrue(chunks[1].startswith("A POWER"))


if __name__ == "__main__":
    unittest.main()
