import unittest

from ocr.recognizer_paddle import choose_primary_ocr_engine, normalize_paddle_results


class PaddlePrimaryTests(unittest.TestCase):
    def test_choose_primary_engine_prefers_paddle_when_available(self):
        engine = choose_primary_ocr_engine(paddle_ready=True)

        self.assertEqual(engine, "paddle")

    def test_choose_primary_engine_falls_back_to_easyocr_when_unavailable(self):
        engine = choose_primary_ocr_engine(paddle_ready=False)

        self.assertEqual(engine, "easyocr")

    def test_normalize_paddle_results_keeps_bbox_text_and_confidence(self):
        raw = [
            [
                [[[10, 20], [80, 20], [80, 44], [10, 44]], ("HELLO WORLD", 0.93)],
                [[[12, 60], [95, 60], [95, 82], [12, 82]], ("NEXT LINE", 0.81)],
            ]
        ]

        runs = normalize_paddle_results(raw)

        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["text"], "HELLO WORLD")
        self.assertEqual(runs[0]["source"], "primary-paddle")
        self.assertAlmostEqual(runs[0]["confidence"], 0.93, places=2)
        self.assertEqual(runs[1]["bbox_pts"][0], [12, 60])


if __name__ == "__main__":
    unittest.main()
