import unittest
from pathlib import Path

from layout.balloon_layout import enrich_page_layout


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
            "image": str(Path(__file__).resolve().parents[2] / "testes" / "009__001.jpg"),
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
        self.assertLess(text["balloon_subregions"][0][1], text["balloon_subregions"][1][1])

    def test_real_009_single_balloon_keeps_single_region(self):
        page = {
            "image": str(Path(__file__).resolve().parents[2] / "testes" / "009__001.jpg"),
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
            "image": str(Path(__file__).resolve().parents[2] / "testes" / "002__002.jpg"),
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


if __name__ == "__main__":
    unittest.main()
