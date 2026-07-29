from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Detector:
    def __init__(self):
        self.calls = []

    def detect(self, image_rgb):
        self.calls.append(image_rgb.copy())
        return [{"bbox": [3, 4, 18, 12], "confidence": 0.91, "provider": "fresh"}]


class _Runtime:
    def __init__(self):
        self.calls = []

    def run_ocr_stage(self, image_rgb, page):
        self.calls.append((image_rgb.copy(), page))
        return {
            "texts": [{"text": "SOURCE BODY", "bbox": [3, 4, 18, 12], "confidence": 0.88}],
        }


def _write_image(path: Path) -> bytes:
    image_rgb = np.full((20, 30, 3), [31, 97, 203], dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    assert ok
    payload = encoded.tobytes()
    path.write_bytes(payload)
    return payload


def test_observer_receives_persisted_file_without_project_boxes(tmp_path):
    from qa.final_pixel_observer import DetectorOcrFinalPixelObserver

    image_path = tmp_path / "final.png"
    _write_image(image_path)
    detector = _Detector()
    runtime = _Runtime()
    observer = DetectorOcrFinalPixelObserver(detector=detector, runtime=runtime)

    observation = observer.observe(image_path, source_language="en")

    assert observation.image_path == image_path
    assert observation.detected_blocks == (
        {"bbox": [3, 4, 18, 12], "confidence": 0.91, "provider": "fresh"},
    )
    assert runtime.calls[0][1]["_vision_blocks"] == list(observation.detected_blocks)
    assert "texts" not in runtime.calls[0][1]
    assert "project_boxes" not in runtime.calls[0][1]


def test_default_observer_runs_fresh_detector_then_ocr_on_persisted_pixels(tmp_path):
    from qa.final_pixel_observer import DetectorOcrFinalPixelObserver

    image_path = tmp_path / "final.png"
    _write_image(image_path)
    detector = _Detector()
    runtime = _Runtime()

    observation = DetectorOcrFinalPixelObserver(
        detector=detector,
        runtime=runtime,
    ).observe(image_path, source_language="en")

    assert len(detector.calls) == 1
    assert len(runtime.calls) == 1
    np.testing.assert_array_equal(detector.calls[0], runtime.calls[0][0])
    assert observation.ocr_records[0]["text"] == "SOURCE BODY"


def test_observer_hashes_exact_bytes_it_reads(tmp_path):
    from qa.final_pixel_observer import DetectorOcrFinalPixelObserver

    image_path = tmp_path / "final.png"
    payload = _write_image(image_path)

    observation = DetectorOcrFinalPixelObserver(
        detector=_Detector(),
        runtime=_Runtime(),
    ).observe(image_path, source_language="en")

    assert observation.persisted_sha256 == sha256(payload).hexdigest()


def test_detector_mutation_cannot_change_pixels_seen_by_fresh_ocr(tmp_path):
    from qa.final_pixel_observer import DetectorOcrFinalPixelObserver

    image_path = tmp_path / "final.png"
    _write_image(image_path)

    class MutatingDetector(_Detector):
        def detect(self, image_rgb):
            image_rgb[:, :, :] = 0
            return super().detect(image_rgb)

    runtime = _Runtime()
    observation = DetectorOcrFinalPixelObserver(
        detector=MutatingDetector(),
        runtime=runtime,
    ).observe(image_path, source_language="en")

    assert np.any(runtime.calls[0][0] != 0)
    np.testing.assert_array_equal(runtime.calls[0][0], observation.image_rgb)
