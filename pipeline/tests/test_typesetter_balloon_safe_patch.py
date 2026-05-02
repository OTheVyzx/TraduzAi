import unittest

from PIL import Image

from typesetter.safe_renderer_runtime_patch import _prepare_balloon_safe_text_container


class TypesetterBalloonSafePatchTests(unittest.TestCase):
    def test_long_translated_text_uses_balloon_bbox_instead_of_ocr_bbox(self):
        image = Image.new("RGB", (800, 800), (255, 255, 255))
        layer = {
            "id": "page-001-layer-0",
            "bbox": [164, 45, 653, 241],
            "text_pixel_bbox": [164, 45, 653, 241],
            "balloon_bbox": [0, 29, 800, 257],
            "translated": (
                "EM TEMPOS DE PAZ, É APENAS UM TÍTULO VAZIO. MAS QUANDO "
                "A GUERRA ECLODE, CONCEDE UMA ENORME AUTORIDADE"
            ),
            "estilo": {"tamanho": 31, "fonte": "ComicNeue-Bold.ttf"},
        }

        changed = _prepare_balloon_safe_text_container(image, layer)

        self.assertTrue(changed)
        self.assertEqual(layer["bbox"], [0, 29, 800, 257])
        self.assertEqual(layer["layout_bbox"], [0, 29, 800, 257])
        self.assertIsNone(layer["text_pixel_bbox"])
        self.assertEqual(layer["_traduzai_original_bbox"], [164, 45, 653, 241])
        self.assertEqual(layer["_traduzai_original_text_pixel_bbox"], [164, 45, 653, 241])

    def test_short_text_keeps_ocr_anchor(self):
        image = Image.new("RGB", (800, 800), (255, 255, 255))
        layer = {
            "id": "short-label",
            "bbox": [164, 45, 653, 241],
            "text_pixel_bbox": [164, 45, 653, 241],
            "balloon_bbox": [0, 29, 800, 257],
            "translated": "ANCORA",
            "estilo": {"tamanho": 28, "fonte": "ComicNeue-Bold.ttf"},
        }

        changed = _prepare_balloon_safe_text_container(image, layer)

        self.assertFalse(changed)
        self.assertEqual(layer["bbox"], [164, 45, 653, 241])
        self.assertEqual(layer["text_pixel_bbox"], [164, 45, 653, 241])
        self.assertNotIn("layout_bbox", layer)

    def test_page_001_second_balloon_regression(self):
        image = Image.new("RGB", (800, 800), (255, 255, 255))
        layer = {
            "id": "page-001-layer-1",
            "bbox": [160, 541, 636, 734],
            "text_pixel_bbox": [160, 541, 636, 734],
            "balloon_bbox": [0, 525, 800, 750],
            "translated": "VOCÊ É DIFÍCIL DE PREVER E AGORA DESEJA QUE EU O ARME TAMBÉM?",
            "estilo": {"tamanho": 40, "fonte": "ComicNeue-Bold.ttf"},
        }

        changed = _prepare_balloon_safe_text_container(image, layer)

        self.assertTrue(changed)
        self.assertEqual(layer["bbox"], [0, 525, 800, 750])
        self.assertEqual(layer["layout_bbox"], [0, 525, 800, 750])
        self.assertIsNone(layer["text_pixel_bbox"])


if __name__ == "__main__":
    unittest.main()
