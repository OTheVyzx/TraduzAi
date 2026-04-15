import unittest
from pathlib import Path

import cv2
import numpy as np

from layout.balloon_layout import (
    _detect_connected_balloon_subregions_from_fill,
    _detect_lobes_via_distance_transform,
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


if __name__ == "__main__":
    unittest.main()
