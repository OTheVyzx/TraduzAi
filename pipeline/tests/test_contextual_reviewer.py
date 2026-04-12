import unittest

from ocr.contextual_reviewer import contextual_review_page


class ContextualReviewerTests(unittest.TestCase):
    def test_uses_page_lexicon_to_fix_low_confidence_name(self):
        page = {
            "texts": [
                {"text": "MARTHA", "confidence": 0.94, "tipo": "fala", "bbox": [0, 0, 10, 10]},
                {"text": "M4RTHA!", "confidence": 0.44, "tipo": "fala", "bbox": [20, 0, 40, 10]},
            ]
        }

        reviewed = contextual_review_page(page, previous_pages=[])

        self.assertEqual(reviewed["texts"][1]["text"], "MARTHA!")
        self.assertTrue(reviewed["texts"][1]["ocr_context_reviewed"])

    def test_uses_expected_terms_from_corpus_to_fix_low_confidence_word(self):
        page = {
            "texts": [
                {"text": "GHISLA1N", "confidence": 0.38, "tipo": "fala", "bbox": [0, 0, 40, 10]},
            ]
        }

        reviewed = contextual_review_page(
            page,
            previous_pages=[],
            expected_terms=["Ghislain", "Mercenary"],
        )

        self.assertEqual(reviewed["texts"][0]["text"], "GHISLAIN")
        self.assertTrue(reviewed["texts"][0]["ocr_context_reviewed"])


if __name__ == "__main__":
    unittest.main()
