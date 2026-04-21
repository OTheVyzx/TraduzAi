import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from layout.balloon_layout import (
    _analyze_connected_subregions,
    _apply_geometric_fallback_subregions,
    _build_balloon_subregions_from_groups,
    _detect_connected_balloon_subregions_from_fill,
    _detect_lobes_via_distance_transform,
    _enforce_min_lobe_size,
    _geometric_fallback_subregions,
    _refine_connected_position_bboxes_with_ollama,
    _score_subregion_quality,
    enrich_page_layout,
)


def _fixture_image_path(name: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "testes" / name,
        root / "testes" / "debug_pipeline" / "originals" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(name)


class LayoutAnalysisTests(unittest.TestCase):
    def test_connected_reasoner_accepts_valid_ollama_position_boxes(self):
        image = np.full((240, 420, 3), 245, dtype=np.uint8)
        settings = {
            "provider": "ollama",
            "host": "http://localhost:11434",
            "model": "gemma4",
            "enabled": True,
        }
        text_groups = [[52, 44, 182, 136], [222, 118, 360, 214]]
        lobes = [[30, 24, 210, 182], [198, 86, 392, 232]]
        heuristic = [[40, 34, 178, 146], [230, 102, 372, 222]]

        with patch(
            "layout.balloon_layout._check_ollama",
            return_value={"running": True, "models": ["gemma4:e4b"], "has_translator": False},
        ):
            with patch(
                "layout.balloon_layout._call_ollama_json",
                return_value={
                    "position_bboxes": [[46, 36, 184, 144], [236, 112, 378, 228]],
                    "confidence": 0.91,
                    "notes": "left lobe top-left, right lobe lower-right",
                },
            ):
                refined = _refine_connected_position_bboxes_with_ollama(
                    image,
                    [30, 24, 392, 232],
                    [30, 24, 392, 232],
                    text_groups,
                    lobes,
                    heuristic,
                    "left-right",
                    settings,
                )

        self.assertIsNotNone(refined)
        self.assertEqual(refined["position_bboxes"], [[46, 36, 184, 144], [236, 112, 378, 228]])
        self.assertEqual(refined["source"], "ollama")
        self.assertEqual(refined["model"], "gemma4:e4b")
        self.assertGreaterEqual(float(refined["confidence"]), 0.9)

    def test_connected_reasoner_rejects_invalid_ollama_boxes_and_falls_back(self):
        image = np.full((240, 420, 3), 245, dtype=np.uint8)
        settings = {
            "provider": "ollama",
            "host": "http://localhost:11434",
            "model": "gemma4",
            "enabled": True,
        }
        text_groups = [[52, 44, 182, 136], [222, 118, 360, 214]]
        lobes = [[30, 24, 210, 182], [198, 86, 392, 232]]
        heuristic = [[40, 34, 178, 146], [230, 102, 372, 222]]

        with patch(
            "layout.balloon_layout._check_ollama",
            return_value={"running": True, "models": ["gemma4:e4b"], "has_translator": False},
        ):
            with patch(
                "layout.balloon_layout._call_ollama_json",
                return_value={
                    "position_bboxes": [[8, 8, 404, 210], [16, 16, 408, 220]],
                    "confidence": 0.97,
                    "notes": "invalid because boxes escape the lobes",
                },
            ):
                refined = _refine_connected_position_bboxes_with_ollama(
                    image,
                    [30, 24, 392, 232],
                    [30, 24, 392, 232],
                    text_groups,
                    lobes,
                    heuristic,
                    "left-right",
                    settings,
                )

        self.assertIsNone(refined)

    def test_connected_reasoner_accepts_anchor_label_selection(self):
        image = np.full((240, 420, 3), 245, dtype=np.uint8)
        settings = {
            "provider": "ollama",
            "host": "http://localhost:11434",
            "model": "qwen2.5",
            "enabled": True,
        }
        text_groups = [[52, 44, 182, 136], [222, 118, 360, 214]]
        lobes = [[30, 24, 210, 182], [198, 86, 392, 232]]
        heuristic = [[40, 34, 178, 146], [230, 102, 372, 222]]

        with patch(
            "layout.balloon_layout._check_ollama",
            return_value={"running": True, "models": ["qwen2.5:3b"], "has_translator": False},
        ):
            with patch(
                "layout.balloon_layout._call_ollama_json",
                return_value={
                    "selected_anchor_labels": ["outer-upper", "outer-lower"],
                    "confidence": 0.87,
                    "notes": "picked diagonal-biased anchors",
                },
            ):
                refined = _refine_connected_position_bboxes_with_ollama(
                    image,
                    [30, 24, 392, 232],
                    [30, 24, 392, 232],
                    text_groups,
                    lobes,
                    heuristic,
                    "left-right",
                    settings,
                )

        self.assertIsNotNone(refined)
        left_box, right_box = refined["position_bboxes"]
        left_heuristic, right_heuristic = heuristic
        self.assertLess((left_box[0] + left_box[2]) / 2.0, (left_heuristic[0] + left_heuristic[2]) / 2.0)
        self.assertGreater((right_box[1] + right_box[3]) / 2.0, (right_heuristic[1] + right_heuristic[3]) / 2.0)
        self.assertEqual(refined["source"], "ollama")

    def test_assigns_shared_balloon_bbox_for_clustered_dialogue(self):
        page = {
            "width": 800,
            "height": 1200,
            "texts": [
                {"text": "Who are you?", "bbox": [120, 200, 260, 250], "tipo": "fala", "confidence": 0.9},
                {"text": "Answer me!", "bbox": [130, 270, 250, 320], "tipo": "fala", "confidence": 0.88},
            ],
        }

        enriched = enrich_page_layout(page)
        texts = enriched["texts"]

        self.assertEqual(texts[0]["balloon_bbox"], texts[1]["balloon_bbox"])
        self.assertEqual(texts[0]["layout_shape"], "tall")
        self.assertEqual(texts[0]["layout_group_size"], 2)

    def test_prefers_detected_bubble_region_for_single_dialogue(self):
        page = {
            "width": 800,
            "height": 1200,
            "_bubble_regions": [
                {"bbox": [150, 150, 450, 390], "confidence": 0.97},
                {"bbox": [520, 180, 720, 340], "confidence": 0.91},
            ],
            "texts": [
                {"text": "Who are you?", "bbox": [220, 218, 382, 286], "tipo": "fala", "confidence": 0.93},
            ],
        }

        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]

        self.assertEqual(text["balloon_bbox"], [150, 150, 450, 390])
        self.assertEqual(text["ocr_text_bbox"], [220, 218, 382, 286])

    def test_narration_prefers_wide_layout(self):
        page = {
            "width": 1200,
            "height": 1800,
            "texts": [
                {"text": "Three days later...", "bbox": [220, 80, 860, 180], "tipo": "narracao", "confidence": 0.95},
            ],
        }

        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]

        self.assertEqual(text["layout_shape"], "wide")
        self.assertEqual(text["layout_align"], "top")

    def test_real_009_connected_balloon_creates_two_subregions(self):
        page = {
            "image": str(_fixture_image_path("009__001.jpg")),
            "width": 800,
            "height": 2600,
            "texts": [
                {
                    "text": "JTMAY BE NOTHING MORE THAN A HALF-FINISHED CULTINATIONMETHOD, BUT IT'S EFFECTS ARE MORE THAN ENOUGH. A POWER THAT LET'S YOU SURPASS YOUR OWN LIMITS IN AN INSTANT",
                    "bbox": [113, 1514, 705, 1767],
                    "tipo": "fala",
                    "confidence": 0.874,
                }
            ],
        }

        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]

        self.assertEqual(len(text.get("balloon_subregions", [])), 2)
        self.assertEqual(text.get("ocr_text_bbox"), [113, 1514, 705, 1767])
        self.assertEqual(text.get("connected_lobe_bboxes"), text.get("balloon_subregions"))
        self.assertEqual(text.get("connected_position_bboxes"), text.get("connected_focus_bboxes"))
        self.assertEqual(len(text.get("connected_text_groups", [])), 2)
        self.assertEqual(len(text.get("connected_lobe_bboxes", [])), 2)
        self.assertEqual(len(text.get("connected_position_bboxes", [])), 2)
        self.assertNotEqual(text["balloon_subregions"][0], text["balloon_subregions"][1])
        self.assertLess(_intersection_area(text["balloon_subregions"][0], text["balloon_subregions"][1]), 12)
        balloon_area = max(
            1,
            (text["balloon_bbox"][2] - text["balloon_bbox"][0]) * (text["balloon_bbox"][3] - text["balloon_bbox"][1]),
        )
        covered_area = sum(
            max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            for bbox in text.get("balloon_subregions", [])
        )
        self.assertGreater(covered_area / float(balloon_area), 0.55)
        self.assertEqual(len(text.get("connected_focus_bboxes", [])), 2)
        self.assertGreater(float(text.get("connected_detection_confidence", 0.0) or 0.0), 0.5)
        self.assertGreater(float(text.get("connected_group_confidence", 0.0) or 0.0), 0.5)
        self.assertGreater(float(text.get("connected_position_confidence", 0.0) or 0.0), 0.5)
        left_group, right_group = text["connected_text_groups"]
        left_focus, right_focus = text["connected_focus_bboxes"]
        left_sub, right_sub = text["balloon_subregions"]
        left_group_cx = (left_group[0] + left_group[2]) / 2.0
        left_group_cy = (left_group[1] + left_group[3]) / 2.0
        right_group_cx = (right_group[0] + right_group[2]) / 2.0
        right_group_cy = (right_group[1] + right_group[3]) / 2.0
        left_focus_cx = (left_focus[0] + left_focus[2]) / 2.0
        left_focus_cy = (left_focus[1] + left_focus[3]) / 2.0
        right_focus_cx = (right_focus[0] + right_focus[2]) / 2.0
        right_focus_cy = (right_focus[1] + right_focus[3]) / 2.0
        left_sub_cx = (left_sub[0] + left_sub[2]) / 2.0
        left_sub_cy = (left_sub[1] + left_sub[3]) / 2.0
        right_sub_cx = (right_sub[0] + right_sub[2]) / 2.0
        right_sub_cy = (right_sub[1] + right_sub[3]) / 2.0
        self.assertLess(left_group_cx, left_sub_cx)
        self.assertLess(left_group_cy, left_sub_cy)
        self.assertGreater(right_group_cx, right_sub_cx)
        self.assertGreater(right_group_cy, right_sub_cy)
        self.assertLess(left_focus_cx, left_sub_cx)
        self.assertLess(left_focus_cy, left_sub_cy)
        self.assertGreater(right_focus_cx, right_sub_cx)
        self.assertGreater(right_focus_cy, right_sub_cy)

    def test_real_009_single_balloon_keeps_single_region(self):
        page = {
            "image": str(_fixture_image_path("009__001.jpg")),
            "width": 800,
            "height": 2600,
            "texts": [
                {
                    "text": "YOU MEAN THE POWER HE FORCED OUT THROUGHA DEAL WITH 'THEM'",
                    "bbox": [241, 1095, 578, 1235],
                    "tipo": "fala",
                    "confidence": 0.94,
                }
            ],
        }

        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]

        self.assertEqual(text.get("balloon_subregions", []), [])
        self.assertLess(text["balloon_bbox"][2] - text["balloon_bbox"][0], 520)
        self.assertLess(text["balloon_bbox"][3] - text["balloon_bbox"][1], 260)

    def test_real_002_textured_balloon_does_not_merge_with_distant_white_gap_text(self):
        page = {
            "image": str(_fixture_image_path("002__002.jpg")),
            "width": 800,
            "height": 2560,
            "texts": [
                {
                    "text": "Etetob",
                    "bbox": [349, 1733, 759, 2020],
                    "tipo": "fala",
                    "confidence": 0.81,
                },
                {
                    "text": "THERE'S NO TURNING BACK MON",
                    "bbox": [190, 2273, 625, 2476],
                    "tipo": "fala",
                    "confidence": 0.92,
                },
            ],
        }

        enriched = enrich_page_layout(page)
        bottom = next(text for text in enriched["texts"] if "TURNING BACK" in text["text"])

        self.assertEqual(bottom["layout_group_size"], 1)
        self.assertGreater(bottom["balloon_bbox"][1], 2200)
        self.assertLess(bottom["balloon_bbox"][3], 2520)

    def test_partial_balloon_touching_page_edge_still_expands_layout_bbox(self):
        import cv2
        import numpy as np

        page = {
            "width": 260,
            "height": 140,
            "texts": [
                {
                    "text": "USELESS",
                    "bbox": [92, 12, 168, 42],
                    "tipo": "fala",
                    "confidence": 0.96,
                }
            ],
        }

        image = np.zeros((140, 260, 3), dtype=np.uint8)
        image[:] = 18
        cv2.ellipse(image, (130, 10), (94, 52), 0, 0, 360, (246, 246, 246), -1)
        page["_cached_image_bgr"] = image

        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]

        self.assertLessEqual(text["balloon_bbox"][0], 54)
        self.assertEqual(text["balloon_bbox"][1], 0)
        self.assertGreaterEqual(text["balloon_bbox"][2], 206)
        self.assertGreaterEqual(text["balloon_bbox"][3], 54)

    def test_connected_vertical_balloons_split_into_top_and_bottom_subregions(self):
        import cv2
        import numpy as np

        page = {
            "width": 380,
            "height": 320,
            "texts": [
                {
                    "text": "A. B.",
                    "bbox": [70, 45, 305, 245],
                    "tipo": "fala",
                    "confidence": 0.95,
                }
            ],
        }

        image = np.full((320, 380, 3), 230, dtype=np.uint8)
        cv2.ellipse(image, (190, 95), (120, 70), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(image, (200, 210), (125, 72), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(image, (195, 152), (26, 22), 0, 0, 360, (248, 248, 248), -1)
        cv2.rectangle(image, (120, 80), (250, 112), (20, 20, 20), -1)
        cv2.rectangle(image, (116, 194), (258, 226), (20, 20, 20), -1)
        page["_cached_image_bgr"] = image

        enriched = enrich_page_layout(page)
        subregions = enriched["texts"][0]["balloon_subregions"]

        self.assertEqual(len(subregions), 2)
        self.assertLess(subregions[0][1], subregions[1][1])
        self.assertEqual(subregions[0][0], 70)
        self.assertEqual(subregions[1][2], 305)


def _intersection_area(a: list[int], b: list[int]) -> int:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return int((ix2 - ix1) * (iy2 - iy1))


def _make_connected_balloon_image(w: int, h: int, lobe1_center, lobe1_r, lobe2_center, lobe2_r, neck_rect=None):
    """Cria imagem sintética com dois lobos brancos conectados por um pescoço."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, lobe1_center, lobe1_r, (255, 255, 255), -1)
    cv2.circle(img, lobe2_center, lobe2_r, (255, 255, 255), -1)
    if neck_rect:
        nx1, ny1, nx2, ny2 = neck_rect
        cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (255, 255, 255), -1)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 2)
    return img


class ConnectedBalloonDetectionTests(unittest.TestCase):
    def test_distance_transform_detects_wide_neck_balloon(self):
        """Dois lobos conectados por pescoço largo — erosão falha, distance transform detecta."""
        img = _make_connected_balloon_image(
            400, 200,
            lobe1_center=(100, 100), lobe1_r=70,
            lobe2_center=(300, 100), lobe2_r=70,
            neck_rect=(80, 70, 320, 130),
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, component = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        lobes = _detect_lobes_via_distance_transform(
            component, [0, 0, 400, 200], fill_area=int(np.count_nonzero(component)),
        )

        self.assertGreaterEqual(len(lobes), 2)
        centers = [(l["bbox"][0] + l["bbox"][2]) / 2 for l in lobes]
        self.assertTrue(any(c < 200 for c in centers))
        self.assertTrue(any(c > 200 for c in centers))

    def test_distance_transform_returns_empty_for_single_lobe(self):
        """Balão único sem pescoço — distance transform NÃO deve retornar lobos."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 80, (255, 255, 255), -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, component = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        lobes = _detect_lobes_via_distance_transform(
            component, [0, 0, 200, 200], fill_area=int(np.count_nonzero(component)),
        )

        self.assertEqual(len(lobes), 0)

    def test_fill_based_detection_uses_distance_transform_fallback(self):
        """_detect_connected_balloon_subregions_from_fill retorna 2 subregions via DT fallback."""
        img = _make_connected_balloon_image(
            500, 220,
            lobe1_center=(120, 110), lobe1_r=80,
            lobe2_center=(380, 110), lobe2_r=80,
            neck_rect=(100, 75, 400, 145),
        )

        subregions = _detect_connected_balloon_subregions_from_fill(
            img, [80, 40, 420, 180], [0, 0, 500, 220],
        )

        self.assertGreaterEqual(len(subregions), 2)

    def test_geometric_fallback_splits_wide_balloon_with_two_texts(self):
        """Fallback geométrico: balão largo sem imagem + 2 textos próximos → subregions inferidas."""
        # Textos com gap de 20px (< 25.5px threshold) → serão mesclados numa região
        page = {
            "width": 600,
            "height": 200,
            "image": None,
            "texts": [
                {"text": "Hello", "bbox": [20, 50, 240, 150], "tipo": "fala", "confidence": 0.9},
                {"text": "World", "bbox": [260, 50, 480, 150], "tipo": "fala", "confidence": 0.9},
            ],
        }

        enriched = enrich_page_layout(page)
        texts = enriched["texts"]

        has_subregions = any(len(t.get("balloon_subregions", [])) >= 2 for t in texts)
        self.assertTrue(has_subregions, "Fallback geométrico deveria ter inferido subregions")

    def test_geometric_fallback_skips_single_text_wide_balloon(self):
        """1 texto em balão largo → sem subregions pelo fallback."""
        page = {
            "width": 500,
            "height": 200,
            "image": None,
            "texts": [
                {"text": "Hello World", "bbox": [30, 50, 470, 150], "tipo": "fala", "confidence": 0.9},
            ],
        }

        enriched = enrich_page_layout(page)
        texts = enriched["texts"]

        for t in texts:
            self.assertFalse(t.get("balloon_subregions"), "Texto único não deve ter subregions")

    def test_geometric_fallback_handles_koharu_loose_bbox_aspect_below_two(self):
        """Regressão: bubble_bbox solto do koharu (aspect ~1.97) deve ativar Modo B.

        Na app real, o koharu devolve uma bbox maior do que o balão real; se o
        pipeline confia nela e a bbox tem aspect < 2.0, o Mode B antigo
        (aspect >= 2.0) desligava o split e o balão conectado saía como um
        bloco único sobrescrevendo os dois lobos.
        """
        # balloon [29,1455,789,1841]: bw=760, bh=386, aspect≈1.97
        # min(bw,bh)=386 >= 180, max(bw,bh)=760 >= 450
        balloon = [29, 1455, 789, 1841]
        # single-text scenario (group_size==1), so Mode B is the path that must fire
        group_text = {
            "balloon_bbox": balloon,
            "balloon_subregions": [],
            "layout_group_size": 1,
            "tipo": "fala",
            "bbox": [113, 1514, 705, 1765],
        }
        from layout.balloon_layout import _apply_geometric_fallback_subregions
        _apply_geometric_fallback_subregions([group_text])
        self.assertEqual(
            len(group_text.get("balloon_subregions", [])),
            2,
            "Mode B deveria emitir 2 subregions para balão com aspect ~1.97",
        )


class SubregionConfidenceTests(unittest.TestCase):
    def test_good_split_has_high_confidence(self):
        """Two equal halves covering the balloon → high confidence."""
        balloon = [0, 0, 400, 200]
        subregions = [[0, 0, 200, 200], [200, 0, 400, 200]]
        score = _score_subregion_quality(subregions, balloon)
        self.assertGreater(score, 0.7)

    def test_overlapping_subregions_have_lower_confidence(self):
        """Overlapping subregions should score lower."""
        balloon = [0, 0, 400, 200]
        good = [[0, 0, 200, 200], [200, 0, 400, 200]]
        bad = [[0, 0, 300, 200], [100, 0, 400, 200]]  # heavy overlap
        good_score = _score_subregion_quality(good, balloon)
        bad_score = _score_subregion_quality(bad, balloon)
        self.assertGreater(good_score, bad_score)

    def test_empty_subregions_return_zero(self):
        self.assertEqual(_score_subregion_quality([], [0, 0, 400, 200]), 0.0)
        self.assertEqual(_score_subregion_quality([[0, 0, 100, 100]], [0, 0, 400, 200]), 0.0)


class GeometricFallbackSubregionsTests(unittest.TestCase):
    """Testes para _geometric_fallback_subregions com splits horizontal, vertical e diagonal."""

    def test_horizontal_texts_produce_vertical_cut(self):
        """Textos lado-a-lado → corte vertical (L/R split)."""
        balloon = [0, 0, 800, 400]
        texts = [[50, 100, 250, 300], [550, 100, 750, 300]]  # left, right
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)
        # Left sub should be to the left of right sub
        self.assertLess(subs[0][2], subs[1][0] + 50)

    def test_vertical_texts_produce_horizontal_cut(self):
        """Textos empilhados → corte horizontal (T/B split)."""
        balloon = [0, 0, 400, 800]
        texts = [[100, 50, 300, 250], [100, 550, 300, 750]]  # top, bottom
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)
        # Top sub should be above bottom sub
        self.assertLess(subs[0][3], subs[1][1] + 50)

    def test_diagonal_backslash_texts_produce_quadrant_split(self):
        r"""Texto top-left + bottom-right → quadrantes diagonal \ ."""
        balloon = [0, 0, 800, 800]
        texts = [[50, 50, 250, 250], [550, 550, 750, 750]]  # TL, BR
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)
        # One sub covers top-left quadrant, other bottom-right
        tl = min(subs, key=lambda s: s[0] + s[1])
        br = max(subs, key=lambda s: s[0] + s[1])
        self.assertLess(tl[2], balloon[2])  # TL doesn't span full width
        self.assertGreater(br[0], balloon[0])  # BR doesn't start at left edge

    def test_diagonal_slash_texts_produce_quadrant_split(self):
        """Texto top-right + bottom-left → quadrantes diagonal / ."""
        balloon = [0, 0, 800, 800]
        texts = [[550, 50, 750, 250], [50, 550, 250, 750]]  # TR, BL
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)
        # One sub covers top-right, other bottom-left
        tr = min(subs, key=lambda s: s[1])  # lower y = top
        bl = max(subs, key=lambda s: s[1])  # higher y = bottom
        self.assertGreater(tr[0], balloon[0])  # TR starts past left
        self.assertLess(bl[2], balloon[2])  # BL doesn't reach right

    def test_close_centers_rejected(self):
        """Textos com centros muito próximos → retorna [] (balão único)."""
        balloon = [0, 0, 800, 400]
        texts = [[300, 150, 400, 200], [350, 160, 450, 210]]  # very close
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(subs, [])

    def test_single_text_wide_balloon_splits_vertically(self):
        """1 texto + balão largo → split L/R."""
        balloon = [0, 0, 800, 300]
        texts = [[200, 80, 600, 220]]
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)

    def test_single_text_tall_balloon_splits_horizontally(self):
        """1 texto + balão alto → split T/B."""
        balloon = [0, 0, 300, 800]
        texts = [[50, 200, 250, 600]]
        subs = _geometric_fallback_subregions(texts, balloon)
        self.assertEqual(len(subs), 2)

    def test_geometric_fallback_accepts_aspect_1_75_with_min_dims_160_420(self):
        """Balão 420×240 (aspect=1.75, min=160, max=420) + texto central → 2 subregions."""
        # bbox [0, 0, 420, 240]: w=420, h=240 → aspect=1.75, min(w,h)=240≥160, max(w,h)=420≥420
        text_entry = {
            "text": "Texto de teste",
            "bbox": [105, 60, 315, 180],  # centrado no balão
            "tipo": "fala",
            "confidence": 0.9,
            "balloon_bbox": [0, 0, 420, 240],
            "layout_group_size": 1,
        }
        texts = [text_entry]
        _apply_geometric_fallback_subregions(texts)
        subs = texts[0].get("balloon_subregions", [])
        self.assertEqual(len(subs), 2, f"Esperado 2 subregions, obteve {len(subs)}: {subs}")

    def test_text_pixel_gap_fallback_splits_tall_connected_balloon(self):
        """Quando há um grande gap vertical entre text_pixel_bbox, devemos dividir top/bottom."""
        texts = [
            {
                "text": "Parte superior",
                "bbox": [80, 60, 220, 120],
                "text_pixel_bbox": [92, 70, 208, 116],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_bbox": [0, 0, 300, 600],
                "layout_group_size": 2,
            },
            {
                "text": "Parte inferior",
                "bbox": [76, 360, 224, 424],
                "text_pixel_bbox": [88, 372, 212, 418],
                "tipo": "fala",
                "confidence": 0.9,
                "balloon_bbox": [0, 0, 300, 600],
                "layout_group_size": 2,
            },
        ]

        _apply_geometric_fallback_subregions(texts)

        subs = texts[0].get("balloon_subregions", [])
        self.assertEqual(len(subs), 2, f"Esperado 2 subregions, obteve {len(subs)}: {subs}")
        ordered = sorted(subs, key=lambda item: (item[1], item[0]))
        self.assertLess(ordered[0][3], ordered[1][1] + 40)

    def test_fill_area_1500_still_accepted(self):
        """Verifica no código-fonte que fill_area_min é 1500 (não 2500)."""
        import inspect
        import layout.balloon_layout as _mod
        src = inspect.getsource(_mod._detect_connected_balloon_subregions_from_fill)
        self.assertIn("fill_area < 1500", src, "Threshold deveria ser 1500, não 2500")
        self.assertNotIn("fill_area < 2500", src, "Threshold antigo 2500 ainda presente")

    def test_erosion_ratio_relaxed_to_20_percent(self):
        """Verifica no código-fonte que o corte de erosão foi relaxado para 20%."""
        import inspect
        import layout.balloon_layout as _mod

        src = inspect.getsource(_mod._detect_connected_balloon_subregions_from_fill)
        self.assertIn("fill_area * 0.20", src, "Threshold de erosão deveria ser 20%")
        self.assertNotIn("fill_area * 0.25", src, "Threshold antigo de 25% ainda presente")

    def test_gap_threshold_relaxed_to_1_point_5_percent(self):
        """Verifica no código-fonte que o gap mínimo foi relaxado para 1.5%."""
        import inspect
        import layout.balloon_layout as _mod

        src = inspect.getsource(_mod._detect_connected_balloon_subregions_from_fill)
        self.assertIn("0.015", src, "Threshold de gap deveria ser 1.5%")
        self.assertNotIn("0.02", src, "Threshold antigo de gap 2% ainda presente")

    def test_enriched_text_has_subregion_confidence(self):
        """enrich_page_layout should attach subregion_confidence to texts."""
        image = np.full((320, 380, 3), 230, dtype=np.uint8)
        cv2.ellipse(image, (190, 95), (120, 70), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(image, (200, 210), (125, 72), 0, 0, 360, (248, 248, 248), -1)
        cv2.ellipse(image, (195, 152), (26, 22), 0, 0, 360, (248, 248, 248), -1)
        page = {
            "width": 380,
            "height": 320,
            "_cached_image_bgr": image,
            "texts": [
                {
                    "text": "A. B.",
                    "bbox": [70, 45, 305, 245],
                    "tipo": "fala",
                    "confidence": 0.95,
                }
            ],
        }
        enriched = enrich_page_layout(page)
        text = enriched["texts"][0]
        self.assertIn("subregion_confidence", text)
        if text.get("balloon_subregions"):
            self.assertGreater(text["subregion_confidence"], 0.0)


class EnforceMinLobeSizeTests(unittest.TestCase):
    """Testes para _enforce_min_lobe_size e _build_balloon_subregions_from_groups."""

    def test_enforce_min_lobe_size_expands_narrow_right_lobe(self):
        """Lobo direito muito estreito → expandido para 30% mínimo."""
        balloon = [0, 0, 800, 400]
        subs = [[0, 0, 700, 400], [700, 0, 800, 400]]
        fixed = _enforce_min_lobe_size(subs, balloon)
        right_w = fixed[1][2] - fixed[1][0]
        self.assertGreaterEqual(right_w / 800.0, 0.29)

    def test_enforce_min_lobe_size_expands_narrow_left_lobe(self):
        """Lobo esquerdo muito estreito → expandido para 30% mínimo."""
        balloon = [0, 0, 800, 400]
        subs = [[0, 0, 100, 400], [100, 0, 800, 400]]
        fixed = _enforce_min_lobe_size(subs, balloon)
        left_w = fixed[0][2] - fixed[0][0]
        self.assertGreaterEqual(left_w / 800.0, 0.29)

    def test_enforce_min_lobe_size_keeps_balanced_split(self):
        """Split já balanceado → sem mudança."""
        balloon = [0, 0, 800, 400]
        subs = [[0, 0, 400, 400], [400, 0, 800, 400]]
        fixed = _enforce_min_lobe_size(subs, balloon)
        self.assertEqual(fixed[0], [0, 0, 400, 400])
        self.assertEqual(fixed[1], [400, 0, 800, 400])

    def test_build_subregions_horizontal_creates_gap(self):
        """Split horizontal deve ter gap entre os lobos."""
        left_bbox = [50, 100, 350, 300]
        right_bbox = [450, 100, 750, 300]
        balloon = [0, 0, 800, 400]
        subs = _build_balloon_subregions_from_groups([left_bbox, right_bbox], balloon)
        self.assertEqual(len(subs), 2)
        gap = subs[1][0] - subs[0][2]
        self.assertGreater(gap, 0, "Deve haver gap entre os lobos")

    def test_build_subregions_vertical_creates_gap(self):
        """Split vertical deve ter gap entre os lobos."""
        top_bbox = [100, 50, 300, 180]
        bottom_bbox = [100, 250, 300, 380]
        balloon = [0, 0, 400, 400]
        subs = _build_balloon_subregions_from_groups([top_bbox, bottom_bbox], balloon)
        self.assertEqual(len(subs), 2)
        gap = subs[1][1] - subs[0][3]
        self.assertGreater(gap, 0, "Deve haver gap vertical entre os lobos")

    def test_build_subregions_neither_lobe_too_narrow(self):
        """Nenhum lobo deve ter menos que 30% da dimensão principal."""
        left_bbox = [50, 100, 150, 300]
        right_bbox = [600, 100, 750, 300]
        balloon = [0, 0, 800, 400]
        subs = _build_balloon_subregions_from_groups([left_bbox, right_bbox], balloon)
        self.assertEqual(len(subs), 2)
        bw = 800
        for s in subs:
            sw = s[2] - s[0]
            self.assertGreaterEqual(sw / float(bw), 0.25,
                f"Lobo {s} muito estreito: {sw}px de {bw}px")

    def test_analyze_connected_subregions_detects_horizontal_reading_order(self):
        """Subregions horizontais devem ser ordenadas da esquerda para a direita."""
        plan = _analyze_connected_subregions(
            [[220, 0, 420, 240], [0, 0, 200, 240]],
            [0, 0, 420, 240],
        )

        self.assertEqual(plan["orientation"], "left-right")
        self.assertEqual(plan["ordered_subregions"][0], [0, 0, 200, 240])

    def test_analyze_connected_subregions_detects_vertical_reading_order(self):
        """Subregions verticais devem ser ordenadas de cima para baixo."""
        plan = _analyze_connected_subregions(
            [[0, 220, 240, 420], [0, 0, 240, 200]],
            [0, 0, 240, 420],
        )

        self.assertEqual(plan["orientation"], "top-bottom")
        self.assertEqual(plan["ordered_subregions"][0], [0, 0, 240, 200])


if __name__ == "__main__":
    unittest.main()
