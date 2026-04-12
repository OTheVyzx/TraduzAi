import unittest

from ocr.reviewer import choose_best_candidate


class OcrReviewerTests(unittest.TestCase):
    def test_prefers_fallback_when_primary_is_low_confidence_and_noisy(self):
        primary = {
            "text": "1T'5 M3!!!",
            "confidence": 0.31,
            "source": "primary",
        }
        fallback_candidates = [
            {
                "text": "IT'S ME!!!",
                "confidence": 0.67,
                "source": "fallback-threshold",
            }
        ]

        chosen = choose_best_candidate(
            primary=primary,
            fallback_candidates=fallback_candidates,
            tipo="fala",
        )

        self.assertEqual(chosen["text"], "IT'S ME!!!")
        self.assertEqual(chosen["source"], "fallback-threshold")
        self.assertGreater(chosen["confidence"], primary["confidence"])

    def test_keeps_primary_when_fallback_is_less_legible(self):
        primary = {
            "text": "GET OUT OF HERE",
            "confidence": 0.76,
            "source": "primary",
        }
        fallback_candidates = [
            {
                "text": "6ET 0UT 0F HERE",
                "confidence": 0.64,
                "source": "fallback-upscale",
            }
        ]

        chosen = choose_best_candidate(
            primary=primary,
            fallback_candidates=fallback_candidates,
            tipo="fala",
        )

        self.assertEqual(chosen["text"], "GET OUT OF HERE")
        self.assertEqual(chosen["source"], "primary")


if __name__ == "__main__":
    unittest.main()
