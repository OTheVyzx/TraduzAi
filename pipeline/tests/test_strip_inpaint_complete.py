"""Garante que inpaint_band_image dilata bboxes antes de inpainting.

A dilation de 6px é o fix para striations — pixels anti-aliased que ficam
na borda do text bbox e sobrevivem ao inpainting sem a expansão.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock


class InpaintDilationTests(unittest.TestCase):
    """Verifica que os bboxes passados a clean_image são dilatados em 6px."""

    def _capture_inflated_texts(self):
        """Context manager que intercepta clean_image e captura os texts recebidos."""
        from PIL import Image

        captured = []

        def fake_clean_image(img, texts, corpus_visual_benchmark=None):
            captured.extend(texts)
            return img  # devolver imagem sem alteração

        return captured, fake_clean_image

    def test_bbox_is_dilated_by_6px_in_x(self):
        """Cada text bbox passado a clean_image deve ser 6px maior em cada borda."""
        from inpainter import inpaint_band_image

        band = np.full((100, 300, 3), 255, dtype=np.uint8)
        page = {"texts": [{
            "id": "t1",
            "bbox": [60, 20, 160, 60],  # original: 100x40
            "tipo": "fala",
        }]}

        captured, fake_clean = self._capture_inflated_texts()
        with patch("inpainter.classical.clean_image", side_effect=fake_clean):
            inpaint_band_image(band, page)

        self.assertEqual(len(captured), 1, "clean_image deveria receber exatamente 1 texto")
        ix1, iy1, ix2, iy2 = captured[0]["bbox"]
        # Deve ter sido dilatado em 6 px em cada borda
        self.assertLessEqual(ix1, 60 - 6 + 1,
            f"x1 esperado <= {60-6}, got {ix1}")
        self.assertLessEqual(iy1, 20 - 6 + 1,
            f"y1 esperado <= {20-6}, got {iy1}")
        self.assertGreaterEqual(ix2, 160 + 6 - 1,
            f"x2 esperado >= {160+6}, got {ix2}")
        self.assertGreaterEqual(iy2, 60 + 6 - 1,
            f"y2 esperado >= {60+6}, got {iy2}")

    def test_bbox_dilation_clipped_to_image_bounds(self):
        """Dilation não deve gerar coordenadas negativas ou maiores que o shape."""
        from inpainter import inpaint_band_image

        band = np.full((50, 100, 3), 255, dtype=np.uint8)
        page = {"texts": [{
            "id": "t2",
            "bbox": [0, 0, 100, 50],  # cobre a imagem inteira
            "tipo": "fala",
        }]}

        captured, fake_clean = self._capture_inflated_texts()
        with patch("inpainter.classical.clean_image", side_effect=fake_clean):
            inpaint_band_image(band, page)

        self.assertEqual(len(captured), 1)
        ix1, iy1, ix2, iy2 = captured[0]["bbox"]
        self.assertGreaterEqual(ix1, 0, "x1 não pode ser negativo")
        self.assertGreaterEqual(iy1, 0, "y1 não pode ser negativo")
        self.assertLessEqual(ix2, 100, "x2 não pode exceder width")
        self.assertLessEqual(iy2, 50, "y2 não pode exceder height")

    def test_texts_without_bbox_are_skipped(self):
        """Textos sem bbox não devem ser passados a clean_image."""
        from inpainter import inpaint_band_image

        band = np.full((100, 200, 3), 255, dtype=np.uint8)
        page = {"texts": [
            {"id": "t_good", "bbox": [10, 10, 100, 50], "tipo": "fala"},
            {"id": "t_bad", "tipo": "sfx"},  # sem bbox
        ]}

        captured, fake_clean = self._capture_inflated_texts()
        with patch("inpainter.classical.clean_image", side_effect=fake_clean):
            inpaint_band_image(band, page)

        # Só o texto com bbox deve chegar
        self.assertEqual(len(captured), 1, "Texto sem bbox não deve ser passado")
        self.assertEqual(captured[0]["id"], "t_good")

    def test_multiple_texts_all_dilated(self):
        """Todos os textos com bbox válido devem ser dilatados."""
        from inpainter import inpaint_band_image

        band = np.full((200, 400, 3), 255, dtype=np.uint8)
        page = {"texts": [
            {"id": "a", "bbox": [50, 30, 150, 70], "tipo": "fala"},
            {"id": "b", "bbox": [200, 100, 350, 140], "tipo": "fala"},
        ]}

        captured, fake_clean = self._capture_inflated_texts()
        with patch("inpainter.classical.clean_image", side_effect=fake_clean):
            inpaint_band_image(band, page)

        self.assertEqual(len(captured), 2, "Dois textos devem chegar ao clean_image")
        for txt in captured:
            orig = next(t for t in page["texts"] if t["id"] == txt["id"])
            ox1, oy1, ox2, oy2 = orig["bbox"]
            ix1, iy1, ix2, iy2 = txt["bbox"]
            self.assertLessEqual(ix1, ox1, f"id={txt['id']}: x1 não dilatado")
            self.assertLessEqual(iy1, oy1, f"id={txt['id']}: y1 não dilatado")
            self.assertGreaterEqual(ix2, ox2, f"id={txt['id']}: x2 não dilatado")
            self.assertGreaterEqual(iy2, oy2, f"id={txt['id']}: y2 não dilatado")


class InpaintPassthroughTests(unittest.TestCase):
    """Testa os casos de passthrough (sem processamento)."""

    def test_empty_texts_returns_copy(self):
        """Com texts vazio, inpaint_band_image deve devolver cópia da imagem."""
        from inpainter import inpaint_band_image

        band = np.full((100, 200, 3), 128, dtype=np.uint8)
        result = inpaint_band_image(band, {"texts": []})
        self.assertEqual(result.shape, band.shape)
        self.assertTrue(np.array_equal(result, band))
        self.assertIsNot(result, band)

    def test_empty_image_returns_copy(self):
        """Com imagem vazia, deve retornar cópia."""
        from inpainter import inpaint_band_image

        band = np.zeros((0, 200, 3), dtype=np.uint8)
        page = {"texts": [{"id": "t", "bbox": [0, 0, 100, 50], "tipo": "fala"}]}
        result = inpaint_band_image(band, page)
        self.assertEqual(result.size, 0)

    def test_output_shape_preserved(self):
        """Shape da saída deve ser idêntico ao da entrada."""
        from inpainter import inpaint_band_image

        for shape in [(50, 100, 3), (200, 400, 3), (1024, 800, 3)]:
            band = np.full(shape, 200, dtype=np.uint8)
            page = {"texts": [{"id": "t", "bbox": [10, 10, 50, 30], "tipo": "fala"}]}
            with patch("inpainter.classical.clean_image", side_effect=lambda img, t, **kw: img):
                result = inpaint_band_image(band, page)
            self.assertEqual(result.shape, shape, f"Shape {shape} foi alterado")
