"""Testes de process_bands.py."""

import unittest


class BandToPageDictTests(unittest.TestCase):
    def test_band_to_page_dict_remaps_balloon_coords_to_local(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import numpy as np

        slice_img = np.zeros((100, 300, 3), dtype=np.uint8)
        band = Band(
            y_top=500,
            y_bottom=600,
            balloons=[
                Balloon(strip_bbox=BBox(50, 510, 150, 590), confidence=0.9),
            ],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0)

        self.assertEqual(page_dict["width"], 300)
        self.assertEqual(page_dict["height"], 100)
        block = page_dict["_vision_blocks"][0]
        self.assertEqual(block["bbox"], [50, 10, 150, 90])


class CopyBackOutsideBalloonsTests(unittest.TestCase):
    def test_copy_back_preserves_pixels_outside_balloons(self):
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox
        import numpy as np

        original = np.full((100, 300, 3), 50, dtype=np.uint8)
        rendered = np.full((100, 300, 3), 200, dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9)],
            original_slice=original,
            rendered_slice=rendered,
        )

        result = _apply_copy_back_outside_balloons(band, balloon_margin=8)

        self.assertEqual(result[50, 100, 0], 200)
        self.assertEqual(result[5, 5, 0], 50)
        outside_y_top = result[:12, :, :]
        self.assertTrue(np.array_equal(outside_y_top, original[:12, :, :]))

    def test_copy_back_diff_below_2_outside_balloons(self):
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox
        import numpy as np

        rng = np.random.default_rng(42)
        original = rng.integers(0, 256, (100, 300, 3), dtype=np.uint8)
        rendered = rng.integers(0, 256, (100, 300, 3), dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9)],
            original_slice=original,
            rendered_slice=rendered,
        )

        result = _apply_copy_back_outside_balloons(band, balloon_margin=8)

        mask_inside = np.zeros(result.shape[:2], dtype=bool)
        mask_inside[12:88, 42:158] = True

        diff = np.abs(result.astype(np.int16) - original.astype(np.int16))
        # Fora da banda interna, pixels devem ser identicos ao original
        self.assertTrue(np.all(diff[~mask_inside] == 0),
            "Pixels fora do bbox+margin foram alterados pelo copy-back")

    def test_copy_back_pixel_perfect_outside_balloon(self):
        """Criterio Q2=a: pixels fora do balloon bbox+margin devem ser identicos ao original."""
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox
        import numpy as np

        rng = np.random.default_rng(7)
        original = rng.integers(0, 256, (300, 600, 3), dtype=np.uint8)
        rendered = rng.integers(0, 256, (300, 600, 3), dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=300,
            balloons=[Balloon(strip_bbox=BBox(100, 50, 300, 200), confidence=0.9)],
            original_slice=original.copy(),
            rendered_slice=rendered.copy(),
        )
        result = _apply_copy_back_outside_balloons(band, balloon_margin=8)

        # Fora do bbox+margin (y < 42, x < 92, etc.) deve ser identico ao original
        self.assertTrue(np.array_equal(result[:42, :, :], original[:42, :, :]),
            "Rows acima do balloon nao sao pixel-perfect iguais ao original")
        # Dentro do bbox deve ser identico ao rendered
        self.assertTrue(np.array_equal(result[60:190, 110:290, :], rendered[60:190, 110:290, :]),
            "Interior do balloon nao e identico ao rendered")


class ProcessBandTests(unittest.TestCase):
    def _make_band(self):
        from strip.types import Band, Balloon, BBox
        import numpy as np
        slice_img = np.full((100, 300, 3), 200, dtype=np.uint8)
        return Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )

    def test_process_band_populates_rendered_slice(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        # Stages mockadas — só precisam retornar dict válido / ndarray
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {"texts": [
            {"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"},
        ]}
        translator = MagicMock()
        translator.translate_pages.return_value = ([{
            "texts": [{"id": "t1", "translated": "OLÁ", "tipo": "fala", "bbox": [50, 20, 150, 80]}]
        }], [])
        inpainter = MagicMock()
        inpainter.inpaint_band_image.return_value = np.full((100, 300, 3), 255, dtype=np.uint8)
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        self.assertIs(result, band)
        self.assertIsNotNone(band.rendered_slice)
        # Stages foram chamadas
        runtime.run_ocr_stage.assert_called_once()
        translator.translate_pages.assert_called_once()
        inpainter.inpaint_band_image.assert_called_once()
        typesetter.render_band_image.assert_called_once()

    def test_process_band_with_no_balloons_returns_original_slice(self):
        from strip.process_bands import process_band
        from strip.types import Band
        import numpy as np
        from unittest.mock import MagicMock

        slice_img = np.full((50, 300, 3), 80, dtype=np.uint8)
        band = Band(y_top=0, y_bottom=50, balloons=[], strip_slice=slice_img.copy(), original_slice=slice_img.copy())

        result = process_band(
            band,
            runtime=MagicMock(),
            translator=MagicMock(),
            inpainter=MagicMock(),
            typesetter=MagicMock(),
            page_idx=0,
        )

        self.assertIs(result, band)

class BandAdaptersTests(unittest.TestCase):
    def test_inpaint_band_image_returns_same_shape(self):
        from inpainter import inpaint_band_image
        import numpy as np
        band = np.full((100, 300, 3), 200, dtype=np.uint8)
        page = {"texts": [
            {"id": "t1", "bbox": [50, 20, 150, 80], "tipo": "fala", "original": "HELLO"},
        ]}
        cleaned = inpaint_band_image(band, page)
        self.assertEqual(cleaned.shape, band.shape)

    def test_render_band_image_returns_same_shape(self):
        from typesetter.renderer import render_band_image
        import numpy as np
        band = np.full((100, 300, 3), 255, dtype=np.uint8)
        page = {"texts": [
            {"id": "t1", "bbox": [50, 20, 150, 80], "tipo": "fala",
             "balloon_bbox": [50, 20, 150, 80],
             "translated": "OLÁ",
             "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "cor": "#000000"}},
        ]}
        rendered = render_band_image(band, page)
        self.assertEqual(rendered.shape, band.shape)


