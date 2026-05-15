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
