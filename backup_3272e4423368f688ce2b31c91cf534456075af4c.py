import unittest

from PIL import Image

from typesetter.renderer import (
    _assign_texts_to_subregions,
    _build_connected_children_candidates,
    _build_textpath_mask,
    _measure_safe_text_block_bbox,
    _recenter_safe_text_positions,
    _resolve_connected_target_sizes,
    _resolve_text_layout,
    _score_connected_group_candidate,
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

    def test_resolve_text_layout_keeps_textured_balloon_lines_inside_real_width(self):
        text_data = {
            "translated": "EMBORA ALGUNS DUVIDAS DA VERACIDADE DA PUNICAO, ENFATIZAMOS O POSSIVEL.",
            "bbox": [70, 20, 330, 170],
            "tipo": "fala",
            "estilo": {
                "fonte": "Newrotic.ttf",
                "tamanho": 40,
                "cor": "#FFFFFF",
                "contorno": "#000000",
                "contorno_px": 2,
                "alinhamento": "center",
            },
            "balloon_bbox": [70, 20, 330, 170],
            "layout_shape": "wide",
            "layout_align": "center",
        }

        plan = plan_text_layout(text_data)
        layout = _resolve_text_layout(text_data, plan)
        real_widths = [
            _build_textpath_mask(layout["font"], line, padding=0).shape[1]
            for line in layout["lines"]
        ]

        self.assertTrue(real_widths)
        self.assertLessEqual(max(real_widths), plan["max_width"])

    def test_split_text_for_connected_balloons_prefers_sentence_boundaries(self):
        chunks = _split_text_for_connected_balloons(
            "IT MAY BE NOTHING MORE THAN A HALF-FINISHED METHOD, BUT IT'S EFFECTS ARE MORE THAN ENOUGH. A POWER THAT LET'S YOU SURPASS YOUR OWN LIMITS IN AN INSTANT.",
            2,
        )

        self.assertEqual(len(chunks), 2)
        self.assertIn("MORE THAN ENOUGH.", chunks[0])
        self.assertTrue(chunks[1].startswith("A POWER"))

    def test_build_render_blocks_assigns_each_text_to_matching_subregion(self):
        """2 textos + 2 subregions → cada texto vai para seu lobo, sem duplicação."""
        subregions = [[30, 40, 200, 220], [220, 40, 400, 220]]
        texts = [
            {
                "translated": "Texto do lobo esquerdo",
                "bbox": [50, 80, 180, 180],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [30, 40, 400, 220],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 2,
            },
            {
                "translated": "Texto do lobo direito",
                "bbox": [240, 80, 380, 180],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [30, 40, 400, 220],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 2,
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block["balloon_subregions"], [[30, 40, 200, 220], [220, 40, 400, 220]])
        self.assertEqual(len(block.get("connected_children", [])), 2)
        left_block = block["connected_children"][0]
        right_block = block["connected_children"][1]
        self.assertEqual(left_block["balloon_bbox"], [30, 40, 200, 220])
        self.assertEqual(right_block["balloon_bbox"], [220, 40, 400, 220])
        self.assertIn("esquerdo", left_block["translated"])
        self.assertIn("direito", right_block["translated"])
        self.assertTrue(left_block["_is_lobe_subregion"])
        self.assertTrue(right_block["_is_lobe_subregion"])

    def test_single_text_connected_balloon_still_splits(self):
        """Regressão: 1 texto + 2 subregions → passthrough com subregions intactas."""
        subregions = [[30, 40, 200, 130], [200, 130, 400, 220]]
        texts = [
            {
                "translated": "Metade um. Metade dois.",
                "bbox": [30, 40, 400, 220],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [30, 40, 400, 220],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 1,
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0].get("balloon_subregions", [])), 2)

    @unittest.skip("Compositor novo agrupa fragmentos OCR por lobo quando a atribuicao e clara.")
    def test_three_texts_two_subregions_merges_then_splits_legacy(self):
        """3 textos + 2 subregions → contagens não casam → merge normal com subregions."""
        subregions = [[30, 40, 200, 220], [220, 40, 400, 220]]
        texts = [
            {
                "translated": "Texto A",
                "bbox": [50, 60, 180, 120],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [30, 40, 400, 220],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 3,
            },
            {
                "translated": "Texto B",
                "bbox": [50, 130, 180, 190],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [30, 40, 400, 220],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 3,
            },
            {
                "translated": "Texto C",
                "bbox": [240, 80, 380, 180],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [30, 40, 400, 220],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 3,
            },
        ]

        blocks = build_render_blocks(texts)

        # 3 textos != 2 subregions → novo comportamento: merge em 1 bloco consolidado
        # com balloon_subregions intactas para o renderer dividir semanticamente.
        self.assertEqual(len(blocks), 1)
        merged = blocks[0]
        self.assertEqual(len(merged.get("balloon_subregions", [])), 2)
        # Texto combinado deve conter partes de todos os blocos
        combined_text = merged.get("translated", "")
        self.assertIn("Texto A", combined_text)
        self.assertIn("Texto B", combined_text)
        self.assertIn("Texto C", combined_text)


    def test_diagonal_assignment_matches_text_to_nearest_subregion(self):
        """Texto TL vai para sub TL, texto BR vai para sub BR (diagonal)."""
        subregions = [[0, 0, 400, 400], [400, 400, 800, 800]]  # TL, BR quadrants
        texts = [
            {
                "translated": "Texto top-left",
                "bbox": [100, 100, 300, 300],  # center at (200, 200) → near sub[0]
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [0, 0, 800, 800],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 2,
            },
            {
                "translated": "Texto bottom-right",
                "bbox": [500, 500, 700, 700],  # center at (600, 600) → near sub[1]
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [0, 0, 800, 800],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 2,
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 1)
        connected_children = blocks[0]["connected_children"]
        tl_block = sorted(connected_children, key=lambda b: b["balloon_bbox"][0] + b["balloon_bbox"][1])[0]
        br_block = sorted(connected_children, key=lambda b: b["balloon_bbox"][0] + b["balloon_bbox"][1])[1]
        self.assertIn("top-left", tl_block["translated"])
        self.assertIn("bottom-right", br_block["translated"])

    def test_assign_texts_to_subregions_greedy_distance(self):
        """Greedy matching atribui corretamente mesmo com textos fora de ordem."""
        subregions = [[0, 0, 200, 200], [400, 400, 600, 600]]
        texts = [
            {"translated": "Far", "bbox": [420, 420, 580, 580]},   # near sub[1]
            {"translated": "Close", "bbox": [20, 20, 180, 180]},   # near sub[0]
        ]
        assignments = _assign_texts_to_subregions(texts, subregions)
        self.assertEqual(len(assignments), 2)
        # "Close" → sub[0], "Far" → sub[1]
        assigned_map = {a[0]["translated"]: a[1] for a in assignments}
        self.assertEqual(assigned_map["Close"], [0, 0, 200, 200])
        self.assertEqual(assigned_map["Far"], [400, 400, 600, 600])

    def test_area_weighted_split_gives_more_text_to_larger_subregion(self):
        """Text split should allocate proportionally more words to larger subregions."""
        text = "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN"
        # Left lobe 3x larger than right
        chunks = _split_text_for_connected_balloons(text, 2, area_weights=[0.75, 0.25])
        self.assertEqual(len(chunks), 2)
        left_words = len(chunks[0].split())
        right_words = len(chunks[1].split())
        self.assertGreater(left_words, right_words)

    def test_area_weighted_split_with_sentences_respects_boundaries(self):
        """Sentence-level split with weights should still prefer sentence boundaries."""
        text = "THIS IS SENTENCE ONE. THIS IS SENTENCE TWO. AND SENTENCE THREE."
        chunks = _split_text_for_connected_balloons(text, 2, area_weights=[0.6, 0.4])
        self.assertEqual(len(chunks), 2)
        # First chunk should end at a sentence boundary
        self.assertTrue(chunks[0].rstrip().endswith("."))

    def test_split_without_weights_splits_evenly(self):
        """Without area weights, words should split roughly evenly."""
        text = "A B C D E F G H"
        chunks = _split_text_for_connected_balloons(text, 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].split()), 4)
        self.assertEqual(len(chunks[1].split()), 4)

    def test_lobe_subregion_gets_wider_layout(self):
        """Subregion lobes should get wider max_width than regular balloons of same size."""
        from typesetter.renderer import plan_text_layout
        base = {
            "translated": "SOME TEXT HERE",
            "bbox": [100, 100, 350, 320],
            "tipo": "fala",
            "estilo": {"tamanho": 24, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
            "balloon_bbox": [100, 100, 350, 320],
            "layout_shape": "square",
            "layout_align": "center",
        }
        normal_plan = plan_text_layout(base)
        lobe = dict(base)
        lobe["_is_lobe_subregion"] = True
        lobe_plan = plan_text_layout(lobe)
        self.assertGreater(lobe_plan["max_width"], normal_plan["max_width"])

    def test_connected_lobe_uses_border_driven_position_bbox(self):
        left = {
            "translated": "LEFT",
            "bbox": [113, 1513, 402, 1767],
            "balloon_bbox": [113, 1513, 402, 1767],
            "tipo": "fala",
            "estilo": {"tamanho": 48, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
            "layout_shape": "wide",
            "layout_align": "center",
            "_is_lobe_subregion": True,
            "_connected_slot_index": 0,
            "_connected_slot_count": 2,
            "connected_balloon_orientation": "left-right",
        }
        right = dict(left)
        right["bbox"] = [402, 1513, 705, 1767]
        right["balloon_bbox"] = [402, 1513, 705, 1767]
        right["_connected_slot_index"] = 1

        left_plan = plan_text_layout(left)
        right_plan = plan_text_layout(right)

        left_target = left_plan["target_bbox"]
        left_position = left_plan["position_bbox"]
        right_target = right_plan["target_bbox"]
        right_position = right_plan["position_bbox"]

        left_target_center = ((left_target[0] + left_target[2]) / 2.0, (left_target[1] + left_target[3]) / 2.0)
        left_position_center = ((left_position[0] + left_position[2]) / 2.0, (left_position[1] + left_position[3]) / 2.0)
        right_target_center = ((right_target[0] + right_target[2]) / 2.0, (right_target[1] + right_target[3]) / 2.0)
        right_position_center = ((right_position[0] + right_position[2]) / 2.0, (right_position[1] + right_position[3]) / 2.0)

        self.assertLess(left_position_center[0], left_target_center[0] - 8.0)
        self.assertLess(left_position_center[1], left_target_center[1] - 8.0)
        self.assertGreater(right_position_center[0], right_target_center[0] + 8.0)
        self.assertGreater(right_position_center[1], right_target_center[1] + 8.0)
        self.assertLess(left_plan["max_width"], left_target[2] - left_target[0])
        self.assertLessEqual(left_plan["max_width"], left_position[2] - left_position[0])
        self.assertLessEqual(right_plan["max_width"], right_position[2] - right_position[0])

    def test_recenter_safe_text_positions_uses_real_glyph_block_for_vertical_centering(self):
        text_data = {
            "translated": "PODE SER NADA MAIS DO QUE",
            "bbox": [0, 0, 240, 150],
            "balloon_bbox": [0, 0, 240, 150],
            "tipo": "fala",
            "estilo": {"tamanho": 28, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
            "layout_shape": "wide",
            "layout_align": "center",
            "_is_lobe_subregion": True,
        }

        plan = plan_text_layout(text_data)
        resolved = _resolve_text_layout(text_data, plan)
        bbox_before = _measure_safe_text_block_bbox(
            resolved["font"],
            resolved["lines"],
            resolved["positions"],
        )
        self.assertIsNotNone(bbox_before)

        recentered = _recenter_safe_text_positions(
            resolved["font"],
            resolved["lines"],
            resolved["positions"],
            target_bbox=plan["position_bbox"],
            padding_y=plan["padding_y"],
            vertical_anchor=plan["vertical_anchor"],
        )
        bbox_after = _measure_safe_text_block_bbox(
            resolved["font"],
            resolved["lines"],
            recentered,
        )
        self.assertIsNotNone(bbox_after)

        balloon_cy = (plan["position_bbox"][1] + plan["position_bbox"][3]) / 2.0
        before_cy = (bbox_before[1] + bbox_before[3]) / 2.0
        after_cy = (bbox_after[1] + bbox_after[3]) / 2.0

        self.assertLess(abs(after_cy - balloon_cy), abs(before_cy - balloon_cy))
        self.assertLess(abs(after_cy - balloon_cy), 6.0)

    def test_connected_target_sizes_difference_bounded(self):
        """Diferença de font size entre lobos nunca passa de 2px."""
        child_a = {
            "translated": "AB CD",
            "bbox": [0, 0, 200, 200],
            "balloon_bbox": [0, 0, 200, 200],
            "tipo": "fala",
            "estilo": {"tamanho": 24, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
            "layout_shape": "wide",
            "layout_align": "center",
            "_is_lobe_subregion": True,
        }
        child_b = dict(child_a)
        child_b["translated"] = "EF GH IJ"
        child_b["bbox"] = [200, 0, 400, 200]
        child_b["balloon_bbox"] = [200, 0, 400, 200]
        plans = [plan_text_layout(child_a), plan_text_layout(child_b)]
        sizes = _resolve_connected_target_sizes([child_a, child_b], plans)
        self.assertEqual(len(sizes), 2)
        self.assertLessEqual(abs(sizes[0] - sizes[1]), 2)

    def test_connected_target_sizes_large_gap_tolerant(self):
        """Gap > 4px → lobo maior pode ficar até 2px acima do menor."""
        child_a = {
            "translated": "AB",
            "bbox": [0, 0, 400, 400],
            "balloon_bbox": [0, 0, 400, 400],
            "tipo": "fala",
            "estilo": {"tamanho": 40, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
            "layout_shape": "wide",
            "layout_align": "center",
            "_is_lobe_subregion": True,
        }
        child_b = dict(child_a)
        child_b["translated"] = "CD EF GH IJ KL MN OP QR ST UV WX YZ"
        child_b["bbox"] = [400, 0, 600, 200]
        child_b["balloon_bbox"] = [400, 0, 600, 200]
        plans = [plan_text_layout(child_a), plan_text_layout(child_b)]
        sizes = _resolve_connected_target_sizes([child_a, child_b], plans)
        self.assertEqual(len(sizes), 2)
        self.assertLessEqual(abs(sizes[0] - sizes[1]), 2)

    def test_connected_one_to_one_stays_grouped_for_joint_composition(self):
        """Quando OCR ja separa um texto por lobo, o grupo deve continuar unido."""
        texts = [
            {
                "translated": "LEFT SIDE TEXT",
                "bbox": [40, 90, 180, 210],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
                "balloon_bbox": [0, 0, 420, 240],
                "balloon_subregions": [[0, 0, 200, 240], [220, 0, 420, 240]],
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 2,
            },
            {
                "translated": "RIGHT SIDE TEXT",
                "bbox": [250, 80, 390, 205],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
                "balloon_bbox": [0, 0, 420, 240],
                "balloon_subregions": [[0, 0, 200, 240], [220, 0, 420, 240]],
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 2,
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 1)
        self.assertIn("connected_children", blocks[0])
        self.assertEqual(len(blocks[0]["connected_children"]), 2)
        self.assertEqual(blocks[0]["balloon_subregions"], [[0, 0, 200, 240], [220, 0, 420, 240]])

    def test_connected_fragment_groups_preserve_diagonal_vertical_bias(self):
        texts = [
            {
                "translated": "LEFT UPPER",
                "bbox": [36, 24, 174, 62],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
                "balloon_bbox": [0, 0, 420, 240],
                "balloon_subregions": [[0, 0, 200, 240], [220, 0, 420, 240]],
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 4,
                "connected_balloon_orientation": "left-right",
            },
            {
                "translated": "LEFT MID",
                "bbox": [28, 68, 182, 108],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
                "balloon_bbox": [0, 0, 420, 240],
                "balloon_subregions": [[0, 0, 200, 240], [220, 0, 420, 240]],
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 4,
                "connected_balloon_orientation": "left-right",
            },
            {
                "translated": "RIGHT MID",
                "bbox": [246, 118, 388, 156],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
                "balloon_bbox": [0, 0, 420, 240],
                "balloon_subregions": [[0, 0, 200, 240], [220, 0, 420, 240]],
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 4,
                "connected_balloon_orientation": "left-right",
            },
            {
                "translated": "RIGHT LOWER",
                "bbox": [238, 160, 396, 206],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
                "balloon_bbox": [0, 0, 420, 240],
                "balloon_subregions": [[0, 0, 200, 240], [220, 0, 420, 240]],
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 4,
                "connected_balloon_orientation": "left-right",
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 1)
        children = blocks[0]["connected_children"]
        self.assertLess(children[0].get("_connected_vertical_bias_ratio", 0.0), -0.05)
        self.assertGreater(children[1].get("_connected_vertical_bias_ratio", 0.0), 0.05)

    def test_connected_children_candidates_keep_diagonal_stagger_from_child_bias(self):
        text_data = {
            "translated": (
                "PODE SER NADA MAIS DO QUE UM METODO DE CULTIVO INACABADO, "
                "MAS SEUS EFEITOS SAO MAIS QUE SUFICIENTE. "
                "ESSE PODER PERMITE SUPERAR SEUS PROPRIOS LIMITES EM UM INSTANTE."
            ),
            "bbox": [113, 1513, 705, 1767],
            "balloon_bbox": [113, 1513, 705, 1767],
            "balloon_subregions": [[113, 1513, 402, 1767], [402, 1513, 705, 1767]],
            "connected_balloon_orientation": "left-right",
            "connected_children": [
                {
                    "translated": "PODE SER NADA MAIS DO QUE UM METODO DE CULTIVO INACABADO, MAS SEUS EFEITOS SAO MAIS QUE SUFICIENTE.",
                    "bbox": [140, 1530, 360, 1644],
                    "tipo": "fala",
                    "estilo": {"tamanho": 48, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
                    "layout_shape": "wide",
                    "layout_align": "center",
                    "_connected_vertical_bias_ratio": -0.16,
                },
                {
                    "translated": "ESSE PODER PERMITE SUPERAR SEUS PROPRIOS LIMITES EM UM INSTANTE.",
                    "bbox": [430, 1628, 660, 1750],
                    "tipo": "fala",
                    "estilo": {"tamanho": 48, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
                    "layout_shape": "wide",
                    "layout_align": "center",
                    "_connected_vertical_bias_ratio": 0.16,
                },
            ],
            "tipo": "fala",
            "estilo": {"tamanho": 48, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
            "layout_shape": "wide",
            "layout_align": "center",
        }

        image = Image.new("RGB", (800, 2600), color="white")
        candidate = _build_connected_children_candidates(
            text_data,
            text_data["translated"],
            text_data["balloon_subregions"],
        )[0]
        children = candidate["children"]
        plans = [ensure_legible_plan(image, plan_text_layout(child)) for child in children]
        sizes = _resolve_connected_target_sizes(children, plans)

        resolved_items = []
        for child, plan, size in zip(children, plans, sizes):
            fixed_plan = dict(plan)
            fixed_plan["target_size"] = int(size)
            fixed_plan["_font_search_cap"] = int(size)
            fixed_plan["_font_search_floor"] = int(size)
            resolved_items.append(_resolve_text_layout(child, fixed_plan))

        left_box = resolved_items[0]["block_bbox"]
        right_box = resolved_items[1]["block_bbox"]
        left_center_y = (left_box[1] + left_box[3]) / 2.0
        right_center_y = (right_box[1] + right_box[3]) / 2.0
        avg_center_y = (left_center_y + right_center_y) / 2.0
        balloon_center_y = (text_data["balloon_bbox"][1] + text_data["balloon_bbox"][3]) / 2.0

        self.assertLess(left_center_y, right_center_y - 12.0)
        self.assertGreater(avg_center_y, balloon_center_y + 10.0)

    def test_semantic_connected_split_defaults_to_diagonal_stagger_for_left_right_balloon(self):
        text_data = {
            "translated": (
                "PODE SER NADA MAIS DO QUE UM METODO DE CULTIVO INACABADO, "
                "MAS SEUS EFEITOS SAO MAIS QUE SUFICIENTE. "
                "ESSE PODER PERMITE SUPERAR SEUS PROPRIOS LIMITES EM UM INSTANTE."
            ),
            "bbox": [113, 1513, 705, 1767],
            "balloon_bbox": [113, 1513, 705, 1767],
            "balloon_subregions": [[113, 1513, 402, 1767], [402, 1513, 705, 1767]],
            "connected_balloon_orientation": "left-right",
            "tipo": "fala",
            "estilo": {"tamanho": 48, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
            "layout_shape": "wide",
            "layout_align": "center",
        }

        image = Image.new("RGB", (800, 2600), color="white")
        candidate = _build_connected_children_candidates(
            text_data,
            text_data["translated"],
            text_data["balloon_subregions"],
        )[0]
        children = candidate["children"]
        self.assertLess(children[0].get("_connected_vertical_bias_ratio", 0.0), -0.05)
        self.assertGreater(children[1].get("_connected_vertical_bias_ratio", 0.0), 0.05)

        plans = [ensure_legible_plan(image, plan_text_layout(child)) for child in children]
        sizes = _resolve_connected_target_sizes(children, plans)
        resolved_items = []
        for child, plan, size in zip(children, plans, sizes):
            fixed_plan = dict(plan)
            fixed_plan["target_size"] = int(size)
            fixed_plan["_font_search_cap"] = int(size)
            fixed_plan["_font_search_floor"] = int(size)
            resolved_items.append(_resolve_text_layout(child, fixed_plan))

        left_box = resolved_items[0]["block_bbox"]
        right_box = resolved_items[1]["block_bbox"]
        left_center_y = (left_box[1] + left_box[3]) / 2.0
        right_center_y = (right_box[1] + right_box[3]) / 2.0
        avg_center_y = (left_center_y + right_center_y) / 2.0
        balloon_center_y = (text_data["balloon_bbox"][1] + text_data["balloon_bbox"][3]) / 2.0

        self.assertLess(left_center_y, right_center_y - 12.0)
        self.assertGreater(avg_center_y, balloon_center_y + 10.0)

    def test_connected_target_sizes_keep_small_variation_when_it_helps_density(self):
        """Lobos conectados nao devem forcar tamanho identico quando 2px melhora a composicao."""
        child_a = {
            "translated": "AB CD",
            "bbox": [0, 0, 200, 200],
            "balloon_bbox": [0, 0, 200, 200],
            "tipo": "fala",
            "estilo": {"tamanho": 24, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
            "layout_shape": "wide",
            "layout_align": "center",
            "_is_lobe_subregion": True,
        }
        child_b = dict(child_a)
        child_b["translated"] = "EF GH IJ"
        child_b["bbox"] = [200, 0, 400, 200]
        child_b["balloon_bbox"] = [200, 0, 400, 200]
        plans = [plan_text_layout(child_a), plan_text_layout(child_b)]

        sizes = _resolve_connected_target_sizes([child_a, child_b], plans)

        self.assertEqual(len(sizes), 2)
        self.assertLessEqual(abs(sizes[0] - sizes[1]), 2)
        self.assertGreater(sizes[0], sizes[1])

    def test_connected_candidate_scoring_prefers_sentence_boundary_over_word_balance(self):
        text_data = {
            "translated": (
                "PODE SER NADA MAIS DO QUE UM METODO DE CULTIVO INACABADO, "
                "MAS SEUS EFEITOS JA SAO MAIS DO QUE SUFICIENTES. "
                "UM PODER QUE PERMITE SUPERAR SEUS PROPRIOS LIMITES EM UM INSTANTE."
            ),
            "bbox": [0, 0, 720, 300],
            "balloon_bbox": [0, 0, 720, 300],
            "balloon_subregions": [[0, 0, 340, 300], [360, 0, 720, 300]],
            "connected_balloon_orientation": "left-right",
            "tipo": "fala",
            "estilo": {"tamanho": 28, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
            "layout_shape": "wide",
            "layout_align": "center",
        }
        image = Image.new("RGB", (720, 300), color="white")
        candidates = _build_connected_children_candidates(
            text_data,
            text_data["translated"],
            text_data["balloon_subregions"],
        )

        best_chunks = None
        best_score = float("-inf")
        for candidate in candidates:
            children = candidate["children"]
            plans = [ensure_legible_plan(image, plan_text_layout(child)) for child in children]
            sizes = _resolve_connected_target_sizes(children, plans)
            resolved = []
            final_plans = []
            for child, plan, size in zip(children, plans, sizes):
                fixed_plan = dict(plan)
                fixed_plan["target_size"] = int(size)
                fixed_plan["_font_search_cap"] = int(size)
                fixed_plan["_font_search_floor"] = int(size)
                resolved.append(_resolve_text_layout(child, fixed_plan))
                final_plans.append(fixed_plan)
            score = _score_connected_group_candidate(
                resolved,
                children,
                final_plans,
                semantic_bonus=float(candidate.get("semantic_bonus", 0.0)),
            )
            if score > best_score:
                best_score = score
                best_chunks = [child["translated"] for child in children]

        self.assertIsNotNone(best_chunks)
        self.assertTrue(best_chunks[0].rstrip().endswith("."))
        self.assertTrue(best_chunks[1].startswith("UM PODER"))

    def test_connected_layout_prefers_human_balanced_lobe_shapes_for_reference_sample(self):
        text_data = {
            "translated": (
                "PODE SER NADA MAIS DO QUE UM METODO DE CULTIVO INACABADO, "
                "MAS SEUS EFEITOS SAO MAIS QUE SUFICIENTE. "
                "ESSE PODER PERMITE SUPERAR SEUS PROPRIOS LIMITES EM UM INSTANTE."
            ),
            "bbox": [113, 1513, 705, 1767],
            "balloon_bbox": [113, 1513, 705, 1767],
            "balloon_subregions": [[113, 1513, 402, 1767], [402, 1513, 705, 1767]],
            "connected_balloon_orientation": "left-right",
            "tipo": "fala",
            "estilo": {"tamanho": 48, "alinhamento": "center", "fonte": "ComicNeue-Bold.ttf"},
            "layout_shape": "wide",
            "layout_align": "center",
        }
        image = Image.new("RGB", (800, 2600), color="white")
        candidates = _build_connected_children_candidates(
            text_data,
            text_data["translated"],
            text_data["balloon_subregions"],
        )

        best_children = None
        best_resolved = None
        best_score = float("-inf")
        for candidate in candidates:
            children = candidate["children"]
            plans = [ensure_legible_plan(image, plan_text_layout(child)) for child in children]
            sizes = _resolve_connected_target_sizes(children, plans)
            resolved = []
            final_plans = []
            for child, plan, size in zip(children, plans, sizes):
                fixed_plan = dict(plan)
                fixed_plan["target_size"] = int(size)
                fixed_plan["_font_search_cap"] = int(size)
                fixed_plan["_font_search_floor"] = int(size)
                resolved.append(_resolve_text_layout(child, fixed_plan))
                final_plans.append(fixed_plan)
            score = _score_connected_group_candidate(
                resolved,
                children,
                final_plans,
                semantic_bonus=float(candidate.get("semantic_bonus", 0.0)),
            )
            if score > best_score:
                best_score = score
                best_children = children
                best_resolved = resolved

        self.assertIsNotNone(best_children)
        self.assertIsNotNone(best_resolved)
        self.assertIn("SUFICIENTE.", best_children[0]["translated"])
        self.assertTrue(best_children[1]["translated"].startswith("ESSE PODER"))
        self.assertLessEqual(len(best_resolved[0]["lines"]), 5)
        self.assertLessEqual(len(best_resolved[1]["lines"]), 4)
        self.assertLessEqual(best_resolved[0]["font_size"], 25)
        self.assertLessEqual(best_resolved[1]["font_size"], 25)

    def test_many_fragments_two_subregions_group_into_connected_children(self):
        """Quando varios fragmentos OCR pertencem claramente a cada lobo, agrupa por lobo."""
        subregions = [[56, 1495, 402, 1643], [402, 1643, 753, 1791]]
        texts = [
            {
                "translated": "IT MAY BE NOTHING",
                "bbox": [168, 1514, 433, 1548],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [56, 1495, 753, 1791],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 8,
                "connected_balloon_orientation": "left-right",
            },
            {
                "translated": "MORE THAN A HALF-FINISHED",
                "bbox": [110, 1542, 492, 1579],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [56, 1495, 753, 1791],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 8,
                "connected_balloon_orientation": "left-right",
            },
            {
                "translated": "CULTIVATION METHOD, BUT",
                "bbox": [118, 1573, 484, 1612],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [56, 1495, 753, 1791],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 8,
                "connected_balloon_orientation": "left-right",
            },
            {
                "translated": "IT'S EFFECTS ARE MORE",
                "bbox": [140, 1604, 472, 1638],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [56, 1495, 753, 1791],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 8,
                "connected_balloon_orientation": "left-right",
            },
            {
                "translated": "THAN ENOUGH",
                "bbox": [202, 1634, 398, 1668],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [56, 1495, 753, 1791],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 8,
                "connected_balloon_orientation": "left-right",
            },
            {
                "translated": "A POWER THAT LET'S YOU",
                "bbox": [330, 1676, 683, 1713],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [56, 1495, 753, 1791],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 8,
                "connected_balloon_orientation": "left-right",
            },
            {
                "translated": "SURPASS YOUR OWN LIMITS",
                "bbox": [314, 1708, 695, 1741],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [56, 1495, 753, 1791],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 8,
                "connected_balloon_orientation": "left-right",
            },
            {
                "translated": "IN AN INSTANT",
                "bbox": [408, 1738, 602, 1772],
                "tipo": "fala",
                "estilo": {"tamanho": 24, "alinhamento": "center"},
                "balloon_bbox": [56, 1495, 753, 1791],
                "balloon_subregions": subregions,
                "layout_shape": "wide",
                "layout_align": "center",
                "layout_group_size": 8,
                "connected_balloon_orientation": "left-right",
            },
        ]

        blocks = build_render_blocks(texts)

        self.assertEqual(len(blocks), 1)
        connected = blocks[0]
        self.assertEqual(len(connected.get("connected_children", [])), 2)
        self.assertIn("IT MAY BE NOTHING", connected["connected_children"][0]["translated"])
        self.assertIn("THAN ENOUGH", connected["connected_children"][0]["translated"])
        self.assertIn("A POWER THAT LET'S YOU", connected["connected_children"][1]["translated"])
        self.assertIn("IN AN INSTANT", connected["connected_children"][1]["translated"])


if __name__ == "__main__":
    unittest.main()
