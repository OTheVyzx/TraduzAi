from __future__ import annotations

import sys
import types
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qa import runtime_fingerprint
from typesetter.font_detector import DEFAULT_FONT, FontDetector
from vision_stack.detector import TextBlock, TextDetector
from vision_stack.ocr import OCREngine


@pytest.fixture
def engine_events(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    events: list[dict] = []

    def record_engine_event(**payload):
        events.append(payload)
        return payload

    monkeypatch.setattr(runtime_fingerprint, "record_engine_event", record_engine_event)
    return events


def _make_detector(*, backend: str, model_type: str = "comic-text-detector") -> TextDetector:
    detector = TextDetector.__new__(TextDetector)
    detector.device = torch.device("cpu")
    detector.half = False
    detector._backend = backend
    detector._model_type = model_type
    detector._model_path = Path("fake-detector.pt")
    detector._model = None
    return detector


def _make_ocr_engine(
    *,
    requested: str = "paddleocr",
    resolved: str = "paddleocr",
) -> OCREngine:
    engine = OCREngine.__new__(OCREngine)
    engine._requested_model = requested
    engine.model_name = resolved
    engine._backend = resolved
    engine.device = torch.device("cpu")
    engine.half = False
    engine.batch_size = 8
    engine.lang = "en"
    engine._model = None
    engine._processor = None
    engine._ocr_cache = OrderedDict()
    engine._last_batch_cache_stats = {"ocr_cache_hits": 0, "ocr_cache_misses": 0}
    return engine


def test_detector_construction_does_not_claim_execution(
    monkeypatch: pytest.MonkeyPatch,
    engine_events: list[dict],
) -> None:
    class FakeYolo:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def to(self, _device):
            return self

        def half(self):
            return self

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYolo
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    detector = TextDetector(
        model="fake-primary-detector",
        device="cpu",
        model_path="fake-primary-detector.pt",
    )

    assert detector._backend == "ultralytics"
    assert engine_events == []


def test_detector_records_primary_backend_only_after_successful_detect(
    engine_events: list[dict],
) -> None:
    detector = _make_detector(backend="ultralytics")
    model_calls: list[np.ndarray] = []

    class FakeModel:
        def __call__(self, image, **_kwargs):
            model_calls.append(image)
            return ["raw-detection"]

    detector._model = FakeModel()
    detector._get_inference_size = lambda height, width: (height, width)
    detector._parse_ultralytics = lambda *_args: [
        TextBlock(xyxy=(1.0, 2.0, 8.0, 10.0), confidence=0.9)
    ]

    blocks = detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))

    assert len(model_calls) == 1
    assert len(blocks) == 1
    assert len(engine_events) == 1
    event = engine_events[0]
    assert event["stage"] == "detector"
    assert event["requested_engine"] == "comic-text-detector"
    assert event["resolved_engine"] == "comic-text-detector"
    assert event["backend"] is detector
    assert event["model_path"] == Path("fake-detector.pt")
    assert event["execution_status"] == "succeeded"
    assert event["result_status"] == "accepted"
    assert event["fallback_used"] is False
    assert event["fallback_reason"] == ""
    assert event["execution_context"] == "chapter"


@pytest.mark.parametrize("backend", ["contour-fallback", "paddle-det"])
def test_detector_records_actual_fallback_backend_and_empty_result(
    backend: str,
    engine_events: list[dict],
) -> None:
    detector = _make_detector(backend=backend)
    if backend == "contour-fallback":
        detector._detect_contour_fallback = lambda _image: []
    else:
        detector._model = SimpleNamespace(ocr=lambda *_args, **_kwargs: [[]])
        detector._get_inference_size = lambda height, width: (height, width)
        detector._parse_paddle_detection = lambda *_args: []

    assert detector.detect(np.zeros((16, 16, 3), dtype=np.uint8)) == []

    assert len(engine_events) == 1
    event = engine_events[0]
    assert event["requested_engine"] == "comic-text-detector"
    assert event["resolved_engine"] == backend
    assert event["backend"] is detector
    assert event["result_status"] == "empty"
    assert event["fallback_used"] is True
    assert event["fallback_reason"] == "resolved_engine_differs_from_request"


def test_detector_failed_backend_call_does_not_record_success(
    engine_events: list[dict],
) -> None:
    detector = _make_detector(backend="ultralytics")

    class FailingModel:
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("inference failed")

    detector._model = FailingModel()
    detector._get_inference_size = lambda height, width: (height, width)

    with pytest.raises(RuntimeError, match="inference failed"):
        detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))

    assert engine_events == []


def test_ocr_construction_does_not_claim_execution(
    monkeypatch: pytest.MonkeyPatch,
    engine_events: list[dict],
) -> None:
    class FakeLoader:
        @classmethod
        def from_pretrained(cls, _model_id):
            return cls()

    class FakeModel(FakeLoader):
        def to(self, _device):
            return self

        def eval(self):
            return self

        def half(self):
            return self

    transformers = types.ModuleType("transformers")
    transformers.AutoFeatureExtractor = FakeLoader
    transformers.AutoTokenizer = FakeLoader
    transformers.VisionEncoderDecoderModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    engine = OCREngine(model="manga-ocr", device="cpu")

    assert engine._backend == "manga-ocr"
    assert engine_events == []


def test_ocr_cache_only_batch_does_not_record_execution(
    monkeypatch: pytest.MonkeyPatch,
    engine_events: list[dict],
) -> None:
    engine = _make_ocr_engine()
    crop = np.full((20, 20, 3), 255, dtype=np.uint8)
    cache_key = engine._crop_cache_key(crop)
    engine._ocr_cache[cache_key] = "cached text"
    monkeypatch.setenv("TRADUZAI_OCR_CACHE", "1")
    monkeypatch.setattr(
        engine,
        "_recognize_batch_impl",
        lambda _crops: (_ for _ in ()).throw(AssertionError("cache miss not expected")),
    )

    assert engine.recognize_batch([crop]) == ["cached text"]
    assert engine_events == []


def test_manga_ocr_records_generate_result_after_raw_model_call(
    engine_events: list[dict],
) -> None:
    engine = _make_ocr_engine(requested="manga-ocr", resolved="manga-ocr")
    generated = torch.tensor([[1, 2, 3]], dtype=torch.int64)

    class FakeProcessor:
        def __call__(self, **_kwargs):
            return SimpleNamespace(pixel_values=torch.zeros((1, 3, 224, 224)))

    class FakeModel:
        def generate(self, *_args, **_kwargs):
            return generated

    engine._processor = FakeProcessor()
    engine._model = FakeModel()
    engine._tokenizer = SimpleNamespace(
        batch_decode=lambda *_args, **_kwargs: ["recognized text"]
    )

    result = engine._manga_ocr_batch([np.zeros((12, 12, 3), dtype=np.uint8)])

    assert result == ["recognized text"]
    assert len(engine_events) == 1
    event = engine_events[0]
    assert event["stage"] == "ocr"
    assert event["requested_engine"] == "manga-ocr"
    assert event["resolved_engine"] == "manga-ocr"
    assert event["backend"] is engine
    assert event["execution_status"] == "succeeded"
    assert event["result_status"] == "accepted"
    assert event["fallback_used"] is False
    assert event["execution_context"] == "chapter"


def test_paddle_crop_and_retry_record_each_raw_call_with_fallback_truth(
    engine_events: list[dict],
) -> None:
    engine = _make_ocr_engine(requested="manga-ocr", resolved="paddleocr")
    engine._paddle_use_angle_cls = False
    raw_calls = 0

    class FakeModel:
        def ocr(self, *_args, **_kwargs):
            nonlocal raw_calls
            raw_calls += 1
            if raw_calls == 1:
                return [[]]
            box = [[0, 0], [10, 0], [10, 10], [0, 10]]
            return [[[box, ("FOUND TEXT", 0.99)]]]

    engine._model = FakeModel()

    text = engine._recognize_single_paddle_with_retry(
        np.zeros((20, 20, 3), dtype=np.uint8)
    )

    assert text == "FOUND TEXT"
    assert raw_calls == 2
    assert [event["result_status"] for event in engine_events] == ["empty", "accepted"]
    for event in engine_events:
        assert event["requested_engine"] == "manga-ocr"
        assert event["resolved_engine"] == "paddleocr"
        assert event["backend"] is engine
        assert event["fallback_used"] is True
        assert event["fallback_reason"] == "resolved_engine_differs_from_request"


def test_paddle_swallowed_raw_exception_does_not_record_success(
    engine_events: list[dict],
) -> None:
    engine = _make_ocr_engine()

    class FailingModel:
        def ocr(self, *_args, **_kwargs):
            raise RuntimeError("raw paddle failure")

    engine._model = FailingModel()

    assert engine._recognize_single_paddle(
        np.zeros((20, 20, 3), dtype=np.uint8)
    ) == ""
    assert engine_events == []


def test_paddle_full_page_records_empty_raw_result(
    engine_events: list[dict],
) -> None:
    engine = _make_ocr_engine()
    engine._model = SimpleNamespace(ocr=lambda *_args, **_kwargs: [[]])
    block = SimpleNamespace(x1=2, y1=2, x2=20, y2=20)

    result = engine._paddle_ocr_full_page_to_blocks(
        np.zeros((32, 32, 3), dtype=np.uint8),
        [block],
    )

    assert result is None
    assert len(engine_events) == 1
    assert engine_events[0]["result_status"] == "empty"


def test_paddle_deskew_records_empty_raw_result(
    engine_events: list[dict],
) -> None:
    engine = _make_ocr_engine()
    engine._model = SimpleNamespace(ocr=lambda *_args, **_kwargs: [[]])

    result = engine._recognize_skewed_block_lines(
        np.zeros((64, 64, 3), dtype=np.uint8),
        [10, 10, 40, 40],
        15.0,
    )

    assert result == []
    assert len(engine_events) == 1
    assert engine_events[0]["result_status"] == "empty"


def test_paddle_rotated_page_records_each_raw_call(
    engine_events: list[dict],
) -> None:
    engine = _make_ocr_engine()
    engine._model = SimpleNamespace(ocr=lambda *_args, **_kwargs: [[]])

    result = engine.recognize_rotated_full_page_lines(
        np.zeros((32, 48, 3), dtype=np.uint8),
        rotations=(90, 270),
    )

    assert result == []
    assert len(engine_events) == 2
    assert [event["result_status"] for event in engine_events] == ["empty", "empty"]


def test_font_reference_fingerprints_emit_nothing_but_page_region_does(
    monkeypatch: pytest.MonkeyPatch,
    engine_events: list[dict],
) -> None:
    model_path = Path("yuzumarker-font-detection.safetensors")
    detector = FontDetector(model_path, Path("fonts"))
    actual_model = object()
    detector._model = actual_model
    monkeypatch.setattr(detector, "_discover_candidate_fonts", lambda: ["Candidate.ttf"])
    monkeypatch.setattr(
        detector,
        "_render_font_sample",
        lambda _font_name: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        detector,
        "_extract_features",
        lambda _image: np.array([1.0, 0.0], dtype=np.float32),
    )

    detector._build_fingerprints()

    assert engine_events == []
    detector._loaded = True
    result = detector.detect_with_score(
        np.zeros((16, 16, 3), dtype=np.uint8),
        allow_default=False,
    )

    assert result == ("Candidate.ttf", 1.0)
    assert len(engine_events) == 1
    event = engine_events[0]
    assert event["stage"] == "font_detector"
    assert event["requested_engine"] == "yuzumarker-font-detection"
    assert event["resolved_engine"] == "yuzumarker-font-detection"
    assert event["backend"] is actual_model
    assert event["model_path"] == model_path
    assert event["execution_status"] == "succeeded"
    assert event["result_status"] == "accepted"
    assert event["fallback_used"] is False
    assert event["execution_context"] == "chapter"


def test_font_failed_page_region_extraction_does_not_record_success(
    monkeypatch: pytest.MonkeyPatch,
    engine_events: list[dict],
) -> None:
    detector = FontDetector(Path("fake.safetensors"), Path("fonts"))
    detector._loaded = True
    detector._model = object()
    monkeypatch.setattr(
        detector,
        "_extract_features",
        lambda _image: (_ for _ in ()).throw(RuntimeError("feature extraction failed")),
    )

    result = detector.detect_with_score(np.zeros((16, 16, 3), dtype=np.uint8))

    assert result == (DEFAULT_FONT, 0.0)
    assert engine_events == []
