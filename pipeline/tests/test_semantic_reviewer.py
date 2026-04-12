import unittest

from ocr.semantic_reviewer import semantic_refine_text


class SemanticReviewerTests(unittest.TestCase):
    def test_repairs_common_dialogue_confusions(self):
        reviewed = semantic_refine_text("D0NT Y0U M0VE!", tipo="fala", confidence=0.48)
        self.assertEqual(reviewed, "DON'T YOU MOVE!")

    def test_repairs_contraction_like_im(self):
        reviewed = semantic_refine_text("1M N0T G0ING", tipo="fala", confidence=0.43)
        self.assertEqual(reviewed, "I'M NOT GOING")

    def test_preserves_clean_high_confidence_text(self):
        reviewed = semantic_refine_text("GET OUT OF HERE!", tipo="fala", confidence=0.91)
        self.assertEqual(reviewed, "GET OUT OF HERE!")


if __name__ == "__main__":
    unittest.main()
