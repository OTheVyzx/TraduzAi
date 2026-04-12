import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from vision_stack.inpainter import Inpainter


class VisionStackInpainterTests(unittest.TestCase):
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
