from types import SimpleNamespace

import numpy as np
import pytest

from vision_stack.comic_layout_detector import ComicLayoutRTDetrDetector


def test_comic_layout_detector_falls_back_when_model_is_missing(tmp_path):
    class Fallback:
        def __init__(self):
            self.calls = []
            self.name = "legacy_visual_stack"

        def detect(self, image_rgb, conf_threshold=None):
            self.calls.append(conf_threshold)
            return [SimpleNamespace(xyxy=(1, 2, 3, 4), confidence=0.8)]

    fallback = Fallback()
    detector = ComicLayoutRTDetrDetector(models_dir=tmp_path, fallback=fallback)

    blocks = detector.detect(np.zeros((20, 30, 3), dtype=np.uint8), conf_threshold=0.55)

    assert blocks[0].xyxy == (1, 2, 3, 4)
    assert fallback.calls == [0.55]


def test_comic_layout_detector_finds_local_hf_snapshot(tmp_path):
    snapshot = tmp_path / "huggingface" / "models--ogkalu--comic-text-and-bubble-detector" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    refs = tmp_path / "huggingface" / "models--ogkalu--comic-text-and-bubble-detector" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("abc", encoding="utf-8")

    detector = ComicLayoutRTDetrDetector(models_dir=tmp_path)

    assert detector._find_hf_model_dir() == snapshot


def test_comic_layout_detector_tries_hf_before_legacy_when_onnx_missing(tmp_path):
    class Detector(ComicLayoutRTDetrDetector):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hf_calls = 0

        def _get_hf_model(self):
            return object()

        def _detect_with_hf(self, model, image_rgb, threshold):
            self.hf_calls += 1
            return [SimpleNamespace(xyxy=(5, 6, 7, 8), confidence=0.9, region_type="text_bubble")]

    class Fallback:
        name = "legacy_visual_stack"

        def detect(self, image_rgb, conf_threshold=None):
            return [SimpleNamespace(xyxy=(1, 2, 3, 4), confidence=0.8)]

    detector = Detector(models_dir=tmp_path, fallback=Fallback())

    blocks = detector.detect(np.zeros((20, 30, 3), dtype=np.uint8))

    assert detector.hf_calls == 1
    assert blocks[0].xyxy == (5, 6, 7, 8)


def test_comic_layout_detector_parses_single_tensor_outputs(tmp_path):
    detector = ComicLayoutRTDetrDetector(models_dir=tmp_path, input_size=100)
    outputs = [np.array([[[10, 20, 40, 60, 0.9, 2], [1, 1, 2, 2, 0.2, 1]]], dtype=np.float32)]

    rows = detector._parse_outputs(outputs, threshold=0.5)

    assert rows == [([10.0, 20.0, 40.0, 60.0], 0.8999999761581421, 2)]


def test_comic_layout_detector_parses_boxes_scores_labels_outputs(tmp_path):
    detector = ComicLayoutRTDetrDetector(models_dir=tmp_path, input_size=100)
    outputs = [
        np.array([[[0.1, 0.2, 0.4, 0.6]]], dtype=np.float32),
        np.array([[0.95]], dtype=np.float32),
        np.array([[1]], dtype=np.int64),
    ]

    rows = detector._parse_outputs(outputs, threshold=0.5)

    assert rows[0][0] == pytest.approx([10.0, 20.0, 40.0, 60.0])
    assert rows[0][1] == pytest.approx(0.95)
    assert rows[0][2] == 1
