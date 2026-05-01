"""Regressões do adapter de inpaint do pipeline strip."""

import unittest
from unittest.mock import patch

import numpy as np


class StripInpaintAdapterTests(unittest.TestCase):
    def test_delegates_to_runtime_inpainting_round(self):
        from inpainter import inpaint_band_image

        band = np.full((100, 300, 3), 255, dtype=np.uint8)
        page = {
            "texts": [{"id": "t1", "bbox": [60, 20, 160, 60], "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [54, 14, 166, 66], "confidence": 0.92}],
        }
        expected = np.full_like(band, 127)

        with patch("vision_stack.runtime._get_inpainter", return_value="fake-inpainter") as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=expected,
        ) as apply_round:
            cleaned = inpaint_band_image(band, page)

        self.assertTrue(np.array_equal(cleaned, expected))
        get_inpainter.assert_called_once_with("quality")
        apply_round.assert_called_once()
        args = apply_round.call_args[0]
        self.assertTrue(np.array_equal(args[0], band))
        self.assertEqual(args[1], page)
        self.assertEqual(args[2], "fake-inpainter")

    def test_synthesizes_vision_blocks_from_texts_when_missing(self):
        from inpainter import inpaint_band_image

        band = np.full((80, 200, 3), 240, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [20, 12, 100, 42],
                    "text_pixel_bbox": [24, 16, 96, 38],
                    "tipo": "fala",
                    "text": "HELLO",
                }
            ]
        }

        def _capture_payload(image_np, payload, inpainter):
            self.assertEqual(inpainter, "fake-inpainter")
            self.assertEqual(len(payload["_vision_blocks"]), 1)
            self.assertEqual(payload["_vision_blocks"][0]["bbox"], [24, 16, 96, 38])
            return image_np.copy()

        with patch("vision_stack.runtime._get_inpainter", return_value="fake-inpainter"), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=_capture_payload,
        ) as apply_round:
            cleaned = inpaint_band_image(band, page)

        self.assertEqual(cleaned.shape, band.shape)
        apply_round.assert_called_once()

    def test_texts_without_geometry_are_skipped(self):
        from inpainter import inpaint_band_image

        band = np.full((100, 200, 3), 255, dtype=np.uint8)
        page = {
            "texts": [
                {"id": "t_bad", "tipo": "sfx", "text": "..."},
                {"id": "t_ok", "bbox": [10, 10, 80, 40]},
            ]
        }

        def _capture_payload(image_np, payload, inpainter):
            self.assertEqual(len(payload["_vision_blocks"]), 1)
            self.assertEqual(payload["_vision_blocks"][0]["bbox"], [10, 10, 80, 40])
            return image_np.copy()

        with patch("vision_stack.runtime._get_inpainter", return_value="fake-inpainter"), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=_capture_payload,
        ):
            cleaned = inpaint_band_image(band, page)

        self.assertEqual(cleaned.shape, band.shape)


class InpaintPassthroughTests(unittest.TestCase):
    def test_empty_texts_returns_copy(self):
        from inpainter import inpaint_band_image

        band = np.full((100, 200, 3), 128, dtype=np.uint8)
        result = inpaint_band_image(band, {"texts": []})
        self.assertEqual(result.shape, band.shape)
        self.assertTrue(np.array_equal(result, band))
        self.assertIsNot(result, band)

    def test_empty_image_returns_copy(self):
        from inpainter import inpaint_band_image

        band = np.zeros((0, 200, 3), dtype=np.uint8)
        page = {"texts": [{"id": "t", "bbox": [0, 0, 100, 50], "tipo": "fala"}]}
        result = inpaint_band_image(band, page)
        self.assertEqual(result.size, 0)

    def test_output_shape_preserved(self):
        from inpainter import inpaint_band_image

        for shape in [(50, 100, 3), (200, 400, 3), (1024, 800, 3)]:
            band = np.full(shape, 200, dtype=np.uint8)
            page = {"texts": [{"id": "t", "bbox": [10, 10, 50, 30], "tipo": "fala"}]}
            with patch("vision_stack.runtime._get_inpainter", return_value="fake-inpainter"), patch(
                "vision_stack.runtime._apply_inpainting_round",
                side_effect=lambda image_np, payload, inp: image_np.copy(),
            ):
                result = inpaint_band_image(band, page)
            self.assertEqual(result.shape, shape, f"Shape {shape} foi alterado")
