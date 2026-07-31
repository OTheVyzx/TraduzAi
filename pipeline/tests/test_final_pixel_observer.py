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

    def run_final_pixel_ocr_probe(self, image_rgb, **kwargs):
        self.calls.append((image_rgb.copy(), kwargs))
        return {
            "raw_ocr_records": [{"text": "SOURCE BODY", "bbox": [3, 4, 18, 12], "confidence": 0.88}],
            "ocr_attempts": [{"target_id": "detector:0", "status": "recognized"}],
            "expected_source_challenge_count": len(kwargs.get("source_challenges") or []),
            "completed_source_challenge_count": len(kwargs.get("source_challenges") or []),
            "coverage_complete": True,
            "coverage_failures": [],
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
    assert runtime.calls[0][1]["detected_blocks"] == list(observation.detected_blocks)
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


def test_detector_blocks_with_zero_usable_ocr_are_incomplete(tmp_path):
    from qa.final_pixel_observer import DetectorOcrFinalPixelObserver

    class EmptyProbeRuntime:
        def run_final_pixel_ocr_probe(self, image_rgb, **kwargs):
            return {
                "raw_ocr_records": [],
                "ocr_attempts": [{"target_id": "detector:0", "status": "no_usable_ocr"}],
                "expected_source_challenge_count": 0,
                "completed_source_challenge_count": 0,
                "coverage_complete": False,
                "coverage_failures": ["detector_blocks_without_usable_ocr"],
            }

    image_path = tmp_path / "final.png"
    _write_image(image_path)
    observation = DetectorOcrFinalPixelObserver(
        detector=_Detector(), runtime=EmptyProbeRuntime()
    ).observe(image_path, source_language="en")

    assert observation.detected_block_count == 1
    assert observation.ocr_record_count == 0
    assert observation.coverage_complete is False
    assert "detector_blocks_without_usable_ocr" in observation.coverage_failures


def test_zero_ocr_zero_detector_is_incomplete_when_material_components_exist(tmp_path):
    from qa.final_pixel_observer import DetectorOcrFinalPixelObserver

    class EmptyDetector:
        def detect(self, _image_rgb):
            return []

    class MaterialProbeRuntime:
        def run_final_pixel_ocr_probe(self, image_rgb, **kwargs):
            return {
                "raw_ocr_records": [],
                "ocr_attempts": [{"target_id": "component_1", "status": "no_usable_ocr"}],
                "expected_source_challenge_count": 1,
                "completed_source_challenge_count": 1,
                "coverage_complete": False,
                "coverage_failures": ["material_components_without_usable_ocr"],
            }

    image_path = tmp_path / "final.png"
    _write_image(image_path)
    observation = DetectorOcrFinalPixelObserver(
        detector=EmptyDetector(), runtime=MaterialProbeRuntime()
    ).observe(
        image_path,
        source_language="en",
        page_id="page_007",
        page_number=7,
        source_challenges=[{"component_id": "component_1", "bbox": [2, 2, 20, 14]}],
    )

    assert observation.coverage_complete is False
    assert observation.expected_source_challenge_count == 1


def test_empty_page_without_material_components_may_be_complete(tmp_path):
    from qa.final_pixel_observer import DetectorOcrFinalPixelObserver

    class EmptyDetector:
        def detect(self, _image_rgb):
            return []

    class EmptyPageRuntime:
        def run_final_pixel_ocr_probe(self, image_rgb, **kwargs):
            return {
                "raw_ocr_records": [],
                "ocr_attempts": [],
                "expected_source_challenge_count": 0,
                "completed_source_challenge_count": 0,
                "coverage_complete": True,
                "coverage_failures": [],
            }

    image_path = tmp_path / "final.png"
    _write_image(image_path)
    observation = DetectorOcrFinalPixelObserver(
        detector=EmptyDetector(), runtime=EmptyPageRuntime()
    ).observe(image_path, source_language="en", source_challenges=[])

    assert observation.coverage_complete is True
