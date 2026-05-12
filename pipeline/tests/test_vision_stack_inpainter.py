import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from vision_stack.inpainter import Inpainter


class VisionStackInpainterTests(unittest.TestCase):
    def test_fast_white_fill_records_disabled_rejection_reason(self):
        from inpainter import _apply_fast_white_balloon_fill

        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        page = {
            "texts": [{"bbox": [10, 10, 60, 40], "balloon_bbox": [8, 8, 70, 50], "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [8, 8, 70, 50]}],
        }

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0"}, clear=False):
            _, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(remaining, [{"bbox": [8, 8, 70, 50]}])
        self.assertEqual(stats["white_balloon_count"], 0)
        self.assertEqual(page["_strip_fast_white_rejection_reasons"], {"disabled": 1})

    def test_fast_local_fill_records_no_vision_blocks_rejection_reason(self):
        from inpainter import _apply_fast_local_balloon_fill

        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        page = {
            "texts": [
                {"bbox": [10, 10, 60, 40], "balloon_bbox": [8, 8, 70, 50], "tipo": "fala"},
                {"bbox": [20, 50, 90, 70], "balloon_bbox": [18, 48, 95, 75], "tipo": "pensamento"},
            ],
            "_vision_blocks": [],
        }

        with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "1"}, clear=False):
            _, remaining, stats = _apply_fast_local_balloon_fill(image, page, [])

        self.assertEqual(remaining, [])
        self.assertEqual(stats["local_balloon_count"], 0)
        self.assertEqual(page["_strip_fast_local_rejection_reasons"], {"no_vision_blocks": 2})

    def test_fast_local_fill_is_opt_in_by_default(self):
        from inpainter import _apply_fast_local_balloon_fill

        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        page = {
            "texts": [{"bbox": [10, 10, 60, 40], "balloon_bbox": [8, 8, 70, 50], "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [8, 8, 70, 50]}],
        }

        with patch.dict("os.environ", {}, clear=True):
            _, remaining, stats = _apply_fast_local_balloon_fill(image, page, list(page["_vision_blocks"]))

        self.assertEqual(remaining, [{"bbox": [8, 8, 70, 50]}])
        self.assertEqual(stats["local_balloon_count"], 0)
        self.assertEqual(page["_strip_fast_local_rejection_reasons"], {"disabled": 1})

    def test_fast_white_fill_rejects_contextual_connected_white_balloon(self):
        from inpainter import _fast_white_rejection_reason

        text = {
            "tipo": "narracao",
            "balloon_type": "white",
            "layout_profile": "white_balloon",
            "context_after": "I KNOW.",
            "confidence": 0.91,
        }

        self.assertEqual(_fast_white_rejection_reason(text), "contextual_white_balloon")

    def test_fast_white_fill_allows_moderate_confidence_clean_top_narration(self):
        from inpainter import _fast_white_rejection_reason

        text = {
            "tipo": "narracao",
            "balloon_type": "white",
            "layout_profile": "top_narration",
            "block_profile": "top_narration",
            "confidence": 0.797,
            "text_pixel_bbox": [390, 3074, 789, 3175],
        }

        self.assertEqual(_fast_white_rejection_reason(text), "")

    def test_fallback_blocks_preserve_text_geometry_for_mask_refinement(self):
        from inpainter import _build_fallback_vision_blocks

        polygons = [[[12, 14], [48, 14], [48, 26], [12, 26]]]
        page = {
            "texts": [
                {
                    "bbox": [8, 8, 80, 40],
                    "text_pixel_bbox": [12, 14, 48, 26],
                    "line_polygons": polygons,
                    "balloon_type": "white",
                    "block_profile": "white_balloon",
                    "confidence": 0.42,
                }
            ]
        }

        blocks = _build_fallback_vision_blocks(page, 120, 80)

        self.assertEqual(blocks[0]["bbox"], [12, 14, 48, 26])
        self.assertEqual(blocks[0]["text_pixel_bbox"], [12, 14, 48, 26])
        self.assertEqual(blocks[0]["line_polygons"], polygons)
        self.assertEqual(blocks[0]["balloon_type"], "white")
        self.assertEqual(blocks[0]["block_profile"], "white_balloon")

    def test_large_dark_textured_text_region_does_not_use_rectangular_solid_fill(self):
        from inpainter import _try_solid_background_text_fill

        image = np.full((260, 260, 3), 8, dtype=np.uint8)
        for y in range(image.shape[0]):
            image[y, :, :] = 8 + (y % 9)
        image[74:86, 44:218, :] = 240
        image[114:126, 44:214, :] = 235
        image[154:166, 44:205, :] = 238

        result = _try_solid_background_text_fill(image, [35, 55, 230, 205], [30, 50, 235, 210])

        self.assertIsNone(result)

    def test_tiled_inpaint_handles_edge_tiles_smaller_than_tile_size(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter._run_inpaint = lambda tile_img, tile_mask: tile_img

        image = np.full((600, 1100, 3), 127, dtype=np.uint8)
        mask = np.zeros((600, 1100), dtype=np.uint8)
        mask[:, 980:1080] = 255

        result = inpainter._tiled_inpaint(image, mask, tile_size=512, overlap=64)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_simple_lama_run_normalizes_output_shape_to_input(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter._backend = "simple_lama"
        inpainter._model = lambda img, mask: Image.fromarray(np.full((104, 100, 3), 180, dtype=np.uint8))

        image = np.full((100, 100, 3), 127, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 20:80] = 255

        result = inpainter._run_inpaint(image, mask)

        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.all(result == 180))

    def test_load_model_prefers_lama_onnx_cuda_when_available(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter.device = type("FakeDevice", (), {"type": "cuda"})()
        inpainter.half = True
        inpainter._model = None

        fake_session = object()

        with patch("vision_stack.inpainter.Path.exists", return_value=True), patch(
            "inpainter.lama_onnx.is_lama_manga_available",
            return_value=True,
        ), patch(
            "vision_stack.inpainter.Inpainter._tensorrt_runtime_available",
            return_value=True,
        ), patch(
            "onnxruntime.preload_dlls",
            return_value=None,
        ), patch(
            "onnxruntime.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ), patch(
            "inpainter.lama_onnx.get_lama_session",
            return_value=fake_session,
        ):
            Inpainter._load_model(inpainter, "lama-manga")

        self.assertIs(inpainter._model, fake_session)
        self.assertEqual(inpainter._backend, "lama_onnx_cuda")

    def test_load_model_prefers_lama_onnx_cuda_by_default_even_when_tensorrt_is_available(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter.device = type("FakeDevice", (), {"type": "cuda"})()
        inpainter.half = True
        inpainter._model = None

        class FakeSession:
            def __init__(self, providers):
                self._providers = providers

            def get_providers(self):
                return self._providers

        fake_session = FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"])

        with patch("vision_stack.inpainter.Path.exists", return_value=True), patch(
            "inpainter.lama_onnx.is_lama_manga_available",
            return_value=True,
        ), patch(
            "onnxruntime.preload_dlls",
            return_value=None,
        ), patch(
            "onnxruntime.get_available_providers",
            return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ), patch(
            "inpainter.lama_onnx.get_lama_session",
            return_value=fake_session,
        ) as get_session:
            Inpainter._load_model(inpainter, "lama-manga")

        get_session.assert_called_once_with(
            unittest.mock.ANY,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertIs(inpainter._model, fake_session)
        self.assertEqual(inpainter._backend, "lama_onnx_cuda")

    def test_load_model_allows_tensorrt_when_opted_in(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter.device = type("FakeDevice", (), {"type": "cuda"})()
        inpainter.half = True
        inpainter._model = None

        class FakeSession:
            def __init__(self, providers):
                self._providers = providers

            def get_providers(self):
                return self._providers

        fake_session = FakeSession(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])

        with patch.dict("os.environ", {"MANGATL_ENABLE_TENSORRT": "1"}, clear=False), patch(
            "vision_stack.inpainter.Path.exists",
            return_value=True,
        ), patch(
            "inpainter.lama_onnx.is_lama_manga_available",
            return_value=True,
        ), patch(
            "vision_stack.inpainter.Inpainter._tensorrt_runtime_available",
            return_value=True,
        ), patch(
            "onnxruntime.preload_dlls",
            return_value=None,
        ), patch(
            "onnxruntime.get_available_providers",
            return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ), patch(
            "inpainter.lama_onnx.get_lama_session",
            return_value=fake_session,
        ) as get_session:
            Inpainter._load_model(inpainter, "lama-manga")

        get_session.assert_called_once_with(
            unittest.mock.ANY,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertIs(inpainter._model, fake_session)
        self.assertEqual(inpainter._backend, "lama_onnx_tensorrt")

    def test_run_inpaint_uses_onnx_backend(self):
        inpainter = Inpainter.__new__(Inpainter)
        inpainter._backend = "lama_onnx_cuda"
        inpainter._model = object()
        inpainter.half = False

        image = np.full((100, 100, 3), 127, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 20:80] = 255

        expected = np.full((100, 100, 3), 210, dtype=np.uint8)

        with patch("inpainter.lama_onnx.inpaint_region_with_lama", return_value=expected) as inpaint_onnx:
            result = inpainter._run_inpaint(image, mask)

        inpaint_onnx.assert_called_once_with(inpainter._model, image, mask)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.all(result == 210))


if __name__ == "__main__":
    unittest.main()
