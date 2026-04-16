import unittest
import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vision_stack.detector import TextBlock, TextDetector


class VisionStackDetectorTests(unittest.TestCase):
    def test_load_comic_text_detector_native_uses_blk_det_checkpoint(self):
        class FakeModel:
            def __init__(self, cfg, ch=3, nc=2):
                self.cfg = cfg
                self.ch = ch
                self.nc = nc
                self.loaded = None
                self.to_device = None
                self.eval_called = False
                self.half_called = False

            def load_state_dict(self, weights, strict=False):
                self.loaded = (weights, strict)
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, device):
                self.to_device = device
                return self

            def eval(self):
                self.eval_called = True
                return self

            def half(self):
                self.half_called = True
                return self

        class FakeSegHead:
            def load_state_dict(self, weights, strict=True):
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, device):
                return self

            def eval(self):
                return self

            def half(self):
                return self

        detector = TextDetector.__new__(TextDetector)
        detector.device = torch.device("cpu")
        detector.half = False
        detector._model = None
        detector._model_type = "comic-text-detector"
        detector._model_path = Path(r"T:\mangatl\pipeline\models\comic-text-detector.pt")

        checkpoint = {
            "blk_det": {
                "cfg": {"nc": 2, "ch": 3},
                "weights": {"layer.weight": torch.ones(1)},
            }
        }
        existing_paths = {str(detector._model_path)}

        with patch("pathlib.Path.exists", autospec=True, side_effect=lambda path: str(path) in existing_paths), patch(
            "vision_stack.detector.torch.load",
            return_value=checkpoint,
        ), patch.object(
            TextDetector,
            "_import_yolov5_runtime",
            return_value=(FakeModel, object(), object(), object()),
        ), patch.object(
            TextDetector,
            "_make_comic_text_seg_head",
            return_value=FakeSegHead(),
        ):
            loaded = detector._load_comic_text_detector_native()

        self.assertTrue(loaded)
        self.assertEqual(detector._backend, "comic-text-detector")
        self.assertEqual(detector._ctd_input_size, 1024)
        self.assertIsNotNone(detector._model)

    def test_detect_comic_text_native_returns_scaled_blocks(self):
        detector = TextDetector.__new__(TextDetector)
        detector.device = torch.device("cpu")
        detector.half = False
        detector._backend = "comic-text-detector"
        detector._ctd_input_size = 1024
        detector._ctd_letterbox = lambda img, new_shape, auto, stride: (img, (1.0, 1.0), (0.0, 0.0))
        detector._ctd_nms = lambda pred, conf_thres, iou_thres: [pred[0]]

        class FakeModel:
            def __call__(self, tensor):
                out = torch.tensor(
                    [
                        [
                            [10.0, 20.0, 60.0, 80.0, 0.95, 0.0],
                            [11.0, 21.0, 59.0, 79.0, 0.90, 0.0],
                        ]
                    ],
                    dtype=torch.float32,
                )
                return out, None

        detector._model = FakeModel()

        image = np.full((100, 120, 3), 255, dtype=np.uint8)
        blocks = detector._detect_comic_text_native(image, conf_threshold=0.5)

        self.assertEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0], TextBlock)
        self.assertEqual(blocks[0].xyxy, (10.0, 20.0, 60.0, 80.0))
        self.assertAlmostEqual(blocks[0].confidence, 0.95, places=3)

    def test_load_comic_text_detector_native_prefers_safetensor_weights(self):
        class FakeModel:
            def __init__(self, cfg, ch=3, nc=2):
                self.loaded = None

            def load_state_dict(self, weights, strict=False):
                self.loaded = weights
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, device):
                return self

            def eval(self):
                return self

            def half(self):
                return self

        class FakeSegHead:
            def __init__(self):
                self.loaded = None

            def load_state_dict(self, weights, strict=True):
                self.loaded = weights
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, device):
                return self

            def eval(self):
                return self

            def half(self):
                return self

        detector = TextDetector.__new__(TextDetector)
        detector.device = torch.device("cpu")
        detector.half = False
        detector._model = None
        detector._model_type = "comic-text-detector"
        detector._model_path = Path(r"T:\mangatl\pipeline\models\comic-text-detector.pt")

        checkpoint = {"blk_det": {"cfg": {"nc": 2, "ch": 3}, "weights": {"from_pt": torch.ones(1)}}}
        yolo_weights = {"from_safetensor": torch.zeros(1)}
        seg_weights = {"seg_safetensor": torch.ones(1)}
        yolo_path = Path(r"T:\mangatl\pk\huggingface\mayocream\comic-text-detector\yolo-v5.safetensors")
        unet_path = Path(r"T:\mangatl\pk\huggingface\mayocream\comic-text-detector\unet.safetensors")
        existing_paths = {
            str(detector._model_path),
            str(yolo_path),
            str(unet_path),
        }

        with patch("pathlib.Path.exists", autospec=True, side_effect=lambda path: str(path) in existing_paths), patch(
            "vision_stack.detector.torch.load",
            return_value=checkpoint,
        ), patch.object(
            TextDetector,
            "_import_yolov5_runtime",
            return_value=(FakeModel, object(), object(), object()),
        ), patch.object(
            TextDetector,
            "_make_comic_text_seg_head",
            return_value=FakeSegHead(),
        ), patch.object(
            TextDetector,
            "_get_comic_text_safetensor_paths",
            return_value={
                "yolo": yolo_path,
                "unet": unet_path,
            },
        ), patch.object(
            TextDetector,
            "_load_safetensor_state_dict",
            side_effect=[yolo_weights, seg_weights],
        ):
            loaded = detector._load_comic_text_detector_native()

        self.assertTrue(loaded)
        self.assertEqual(detector._ctd_weight_source, "safetensors")
        self.assertEqual(detector._model.loaded, yolo_weights)
        self.assertEqual(detector._ctd_seg_head.loaded, seg_weights)

    def test_detect_comic_text_native_attaches_segmentation_mask(self):
        detector = TextDetector.__new__(TextDetector)
        detector.device = torch.device("cpu")
        detector.half = False
        detector._backend = "comic-text-detector"
        detector._ctd_attach_masks = True
        detector._ctd_input_size = 1024
        detector._ctd_letterbox = lambda img, new_shape, auto, stride: (img, (1.0, 1.0), (0.0, 0.0))
        detector._ctd_nms = lambda pred, conf_thres, iou_thres: [pred[0]]

        class FakeModel:
            def __call__(self, tensor):
                out = torch.tensor([[[10.0, 20.0, 60.0, 80.0, 0.95, 0.0]]], dtype=torch.float32)
                return out, None

        detector._model = FakeModel()
        detector._forward_comic_text_detector = lambda tensor: (
            torch.tensor([[[10.0, 20.0, 60.0, 80.0, 0.95, 0.0]]], dtype=torch.float32),
            {},
        )
        full_mask = np.zeros((100, 120), dtype=np.uint8)
        full_mask[25:75, 15:55] = 255
        detector._predict_comic_text_mask = lambda img_rgb, tensor, features=None: full_mask

        image = np.full((100, 120, 3), 255, dtype=np.uint8)
        blocks = detector._detect_comic_text_native(image, conf_threshold=0.5)

        self.assertEqual(len(blocks), 1)
        self.assertIsNotNone(blocks[0].mask)
        self.assertEqual(blocks[0].mask.shape, (60, 50))
        self.assertEqual(int(blocks[0].mask[10, 5]), 255)

    def test_detect_comic_text_native_keeps_masks_disabled_by_default(self):
        detector = TextDetector.__new__(TextDetector)
        detector.device = torch.device("cpu")
        detector.half = False
        detector._backend = "comic-text-detector"
        detector._ctd_input_size = 1024
        detector._ctd_letterbox = lambda img, new_shape, auto, stride: (img, (1.0, 1.0), (0.0, 0.0))
        detector._ctd_nms = lambda pred, conf_thres, iou_thres: [pred[0]]

        class FakeModel:
            def __call__(self, tensor):
                out = torch.tensor([[[10.0, 20.0, 60.0, 80.0, 0.95, 0.0]]], dtype=torch.float32)
                return out, None

        detector._model = FakeModel()

        image = np.full((100, 120, 3), 255, dtype=np.uint8)
        blocks = detector._detect_comic_text_native(image, conf_threshold=0.5)

        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0].mask)

    def test_load_comic_text_detector_native_falls_back_to_checkpoint_when_safetensors_missing(self):
        class FakeModel:
            def __init__(self, cfg, ch=3, nc=2):
                self.loaded = None

            def load_state_dict(self, weights, strict=False):
                self.loaded = weights
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, device):
                return self

            def eval(self):
                return self

            def half(self):
                return self

        class FakeSegHead:
            def __init__(self):
                self.loaded = None

            def load_state_dict(self, weights, strict=True):
                self.loaded = weights
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, device):
                return self

            def eval(self):
                return self

            def half(self):
                return self

        detector = TextDetector.__new__(TextDetector)
        detector.device = torch.device("cpu")
        detector.half = False
        detector._model = None
        detector._model_type = "comic-text-detector"
        detector._model_path = Path(r"T:\mangatl\pipeline\models\comic-text-detector.pt")

        checkpoint = {
            "blk_det": {
                "cfg": {"nc": 2, "ch": 3},
                "weights": {"from_checkpoint": torch.ones(1)},
            },
            "text_seg": {"seg_from_checkpoint": torch.ones(1)},
        }
        yolo_path = Path(r"T:\mangatl\pk\huggingface\mayocream\comic-text-detector\yolo-v5.safetensors")
        unet_path = Path(r"T:\mangatl\pk\huggingface\mayocream\comic-text-detector\unet.safetensors")
        existing_paths = {
            str(detector._model_path),
            str(yolo_path),
            str(unet_path),
        }

        with patch("pathlib.Path.exists", autospec=True, side_effect=lambda path: str(path) in existing_paths), patch(
            "vision_stack.detector.torch.load",
            return_value=checkpoint,
        ), patch.object(
            TextDetector,
            "_import_yolov5_runtime",
            return_value=(FakeModel, object(), object(), object()),
        ), patch.object(
            TextDetector,
            "_make_comic_text_seg_head",
            return_value=FakeSegHead(),
        ), patch.object(
            TextDetector,
            "_get_comic_text_safetensor_paths",
            return_value={"yolo": yolo_path, "unet": unet_path},
        ), patch.object(
            TextDetector,
            "_load_safetensor_state_dict",
            side_effect=ImportError("sem safetensors"),
        ):
            loaded = detector._load_comic_text_detector_native()

        self.assertTrue(loaded)
        self.assertEqual(detector._ctd_weight_source, "checkpoint")
        self.assertEqual(detector._model.loaded, checkpoint["blk_det"]["weights"])
        self.assertEqual(detector._ctd_seg_head.loaded, checkpoint["text_seg"])

    def test_load_model_falls_back_to_contour_detector_when_optional_backends_are_missing(self):
        detector = TextDetector.__new__(TextDetector)
        detector.device = torch.device("cpu")
        detector.half = False
        detector._model = None
        detector._model_type = "comic-text-detector"
        detector._model_path = Path(r"T:\mangatl\pipeline\models\comic-text-detector.pt")
        detector._load_comic_text_detector_native = MagicMock(return_value=False)

        fake_ultralytics = MagicMock()
        fake_ultralytics.YOLO.side_effect = RuntimeError("checkpoint incompatível")
        original_import = builtins.__import__

        with patch.dict("sys.modules", {"ultralytics": fake_ultralytics}, clear=False), patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError(name))
            if name == "paddleocr"
            else original_import(name, *args, **kwargs),
        ):
            TextDetector._load_model(detector)

        self.assertEqual(detector._backend, "contour-fallback")
        self.assertIsNone(detector._model)


if __name__ == "__main__":
    unittest.main()
