from __future__ import annotations

import hashlib
import json

from debug_tools import DebugRecorder, bind_recorder
from qa.runtime_fingerprint import build_engine_fingerprint, record_engine_event


def test_runtime_fingerprint_distinguishes_requested_resolved_and_executed_engine(tmp_path):
    module_path = tmp_path / "backend.py"
    module_path.write_text("BACKEND = 'paddle'\n", encoding="utf-8")

    fingerprint = build_engine_fingerprint(
        stage="ocr",
        requested_engine="paddle-ocr-vl-1.6",
        resolved_engine="paddle-ocr-vl-1.5",
        executed_backend="paddleocr",
        fallback_used=True,
        fallback_reason="resolved_engine_differs_from_request",
        module_file=module_path,
        git_commit="abc123",
    )

    assert fingerprint["stage"] == "ocr"
    assert fingerprint["requested_engine"] == "paddle-ocr-vl-1.6"
    assert fingerprint["resolved_engine"] == "paddle-ocr-vl-1.5"
    assert fingerprint["executed_backend"] == "paddleocr"
    assert fingerprint["fallback_used"] is True
    assert fingerprint["fallback_reason"] == "resolved_engine_differs_from_request"


def test_runtime_fingerprint_records_git_module_and_model_hashes(tmp_path):
    module_path = tmp_path / "backend.py"
    module_path.write_bytes(b"module-v1")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model-v2")

    fingerprint = build_engine_fingerprint(
        stage="inpaint",
        requested_engine="aot-inpainting",
        resolved_engine="aot-inpainting",
        executed_backend="vision_stack.inpainter.Inpainter",
        module_file=module_path,
        model_path=model_path,
        model_revision="revision-7",
        git_commit="deadbeef",
        feature_flags={"safe_fast_fill": True, "mode": "shadow"},
    )

    assert fingerprint["git_commit"] == "deadbeef"
    assert fingerprint["module_file"] == str(module_path.resolve())
    assert fingerprint["module_sha256"] == hashlib.sha256(b"module-v1").hexdigest()
    assert fingerprint["model_file"] == str(model_path.resolve())
    assert fingerprint["model_sha256"] == hashlib.sha256(b"model-v2").hexdigest()
    assert fingerprint["model_revision"] == "revision-7"
    assert fingerprint["feature_flags"] == {"mode": "shadow", "safe_fast_fill": True}


def test_requested_engine_unavailable_is_not_reported_as_executed():
    fingerprint = build_engine_fingerprint(
        stage="bubble_segmentation",
        requested_engine="speech-bubble-segmentation",
        resolved_engine=None,
        executed_backend=None,
        fallback_reason="runtime_unavailable",
        fallback_used=False,
        execution_status="not_needed",
    )

    assert fingerprint["requested_engine"] == "speech-bubble-segmentation"
    assert fingerprint["resolved_engine"] is None
    assert fingerprint["executed_backend"] == "none"
    assert fingerprint["fallback_used"] is False
    assert fingerprint["resolution_status"] == "unavailable"
    assert fingerprint["execution_status"] == "not_needed"


def test_vision_runtime_records_the_backend_instance_that_actually_executed(tmp_path, monkeypatch):
    from vision_stack.runtime import _record_runtime_engine_fingerprint

    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")
    class FakeOCRBackend:
        _backend = "paddleocr"

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        _record_runtime_engine_fingerprint(
            stage="ocr",
            requested_engine="paddle-ocr-vl-1.6",
            resolved_engine="paddle-ocr-vl-1.5",
            backend=FakeOCRBackend(),
            execution_confirmed=True,
            result_status="accepted",
        )
        recorder.finalize()
    finally:
        bind_recorder(None)

    manifest = json.loads(
        (tmp_path / "debug" / "e2e" / "debug_manifest.json").read_text(encoding="utf-8")
    )
    fingerprint = manifest["runtime_fingerprints"][0]
    assert fingerprint["stage"] == "ocr"
    assert fingerprint["executed_backend"] == "paddleocr"
    assert fingerprint["requested_engine"] == "paddle-ocr-vl-1.6"
    assert fingerprint["resolved_engine"] == "paddle-ocr-vl-1.5"


def test_vision_runtime_records_unavailable_engine_without_claiming_execution(tmp_path, monkeypatch):
    from vision_stack.runtime import _record_runtime_engine_fingerprint

    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        _record_runtime_engine_fingerprint(
            stage="bubble_segmentation",
            requested_engine="speech-bubble-segmentation",
            resolved_engine=None,
            backend=None,
            fallback_reason="runtime_unavailable",
        )
        recorder.finalize()
    finally:
        bind_recorder(None)

    manifest = json.loads(
        (tmp_path / "debug" / "e2e" / "debug_manifest.json").read_text(encoding="utf-8")
    )
    fingerprint = manifest["runtime_fingerprints"][0]
    assert fingerprint["executed_backend"] == "none"
    assert fingerprint["fallback_used"] is False
    assert fingerprint["fallback_reason"] == "runtime_unavailable"


def test_vision_runtime_bubble_loader_records_unavailable_requested_engine(tmp_path, monkeypatch):
    from vision_stack import runtime

    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    monkeypatch.setattr(runtime, "_bubble_segmenter", None)
    monkeypatch.setattr(runtime, "_bubble_segmenter_model", "")
    try:
        result = runtime._get_bubble_segmenter_for_page(
            {"engine_preset": {"bubble_segmenter": "speech-bubble-segmentation"}}
        )
        recorder.finalize()
    finally:
        bind_recorder(None)

    manifest = json.loads(
        (tmp_path / "debug" / "e2e" / "debug_manifest.json").read_text(encoding="utf-8")
    )
    fingerprint = manifest["runtime_fingerprints"][0]
    assert result is None
    assert fingerprint["stage"] == "bubble_segmenter"
    assert fingerprint["requested_engine"] == "speech-bubble-segmentation"
    assert fingerprint["executed_backend"] == "none"
    assert fingerprint["fallback_used"] is False
    assert fingerprint["resolution_reason"] == "runtime_unavailable"
    assert fingerprint["feature_flags"]["runtime_fingerprint_v2"] is True


def test_vision_runtime_cached_loaders_record_actual_backends(tmp_path, monkeypatch):
    from vision_stack import runtime

    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")
    class FakeBackend:
        _backend = "fake-runtime"
        lang = "en"
        _requested_model = "paddle-ocr-vl-1.5"

    fake = FakeBackend()
    monkeypatch.setattr(runtime, "_detector", fake)
    monkeypatch.setattr(runtime, "_detector_model", "comic-text-detector")
    monkeypatch.setattr(runtime, "_ocr_engine", fake)
    monkeypatch.setattr(runtime, "_inpainter", fake)
    monkeypatch.setattr(runtime, "_inpainter_model", "aot-inpainting")
    monkeypatch.setattr(runtime, "_font_detector", fake)
    monkeypatch.setattr(runtime, "_profile_to_ocr_model", lambda _profile: "paddle-ocr-vl-1.5")

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        assert runtime._get_detector() is fake
        assert runtime._get_ocr_engine(lang="en") is fake
        assert runtime._get_inpainter() is fake
        assert runtime._get_font_detector() is fake
        recorder.finalize()
    finally:
        bind_recorder(None)

    manifest = json.loads(
        (tmp_path / "debug" / "e2e" / "debug_manifest.json").read_text(encoding="utf-8")
    )
    by_stage = {item["stage"]: item for item in manifest["runtime_fingerprints"]}
    assert set(by_stage) >= {"detector", "ocr", "inpainter", "font_detector"}
    assert all(by_stage[stage]["executed_backend"] == "none" for stage in by_stage)
    assert all(by_stage[stage]["execution_status"] == "not_started" for stage in by_stage)


def test_vision_runtime_ocr_fingerprint_records_actual_fallback_model(tmp_path, monkeypatch):
    from vision_stack import runtime

    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")

    class FakeFallbackOCR:
        _requested_model = "manga-ocr"
        model_name = "paddleocr"
        _backend = "paddleocr"
        lang = "en"

    monkeypatch.setattr(runtime, "_ocr_engine", FakeFallbackOCR())
    monkeypatch.setattr(runtime, "_profile_to_ocr_model", lambda _profile: "manga-ocr")
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        runtime._get_ocr_engine(lang="en")
        recorder.finalize()
    finally:
        bind_recorder(None)

    manifest = json.loads(
        (tmp_path / "debug" / "e2e" / "debug_manifest.json").read_text(encoding="utf-8")
    )
    fingerprint = manifest["runtime_fingerprints"][0]
    assert fingerprint["requested_engine"] == "manga-ocr"
    assert fingerprint["resolved_engine"] == "paddleocr"
    assert fingerprint["executed_backend"] == "none"
    assert fingerprint["execution_status"] == "not_started"
    assert fingerprint["fallback_used"] is False


def test_vision_runtime_finds_nested_aot_weights_for_fingerprint(tmp_path, monkeypatch):
    from vision_stack.runtime import _record_runtime_engine_fingerprint

    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")
    weights = tmp_path / "aot.ckpt"
    weights.write_bytes(b"aot-weights")

    class Paths:
        pass

    paths = Paths()
    paths.weights = weights

    class InnerAOT:
        pass

    inner = InnerAOT()
    inner.paths = paths

    class OuterInpainter:
        _backend = "aot_inpainting"

    backend = OuterInpainter()
    backend._model = inner

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        _record_runtime_engine_fingerprint(
            stage="inpainter",
            requested_engine="aot-inpainting",
            resolved_engine="aot-inpainting",
            backend=backend,
        )
        recorder.finalize()
    finally:
        bind_recorder(None)

    manifest = json.loads(
        (tmp_path / "debug" / "e2e" / "debug_manifest.json").read_text(encoding="utf-8")
    )
    fingerprint = manifest["runtime_fingerprints"][0]
    assert fingerprint["model_file"] == str(weights.resolve())
    assert fingerprint["model_sha256"] == hashlib.sha256(b"aot-weights").hexdigest()


def test_runtime_fingerprint_does_not_infer_fallback_from_names_or_no_execution():
    fingerprint = build_engine_fingerprint(
        stage="ocr",
        requested_engine="paddle-ocr-vl-1.6",
        resolved_engine="paddle-ocr-vl-1.5",
        executed_backend=None,
        execution_status="not_started",
    )

    assert fingerprint["fallback_used"] is False
    assert fingerprint["fallback_reason"] == ""
    assert fingerprint["execution_status"] == "not_started"
    assert fingerprint["result_status"] == "not_produced"


def test_record_engine_event_requires_active_recorder_before_hashing(monkeypatch):
    from qa import runtime_fingerprint

    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")
    bind_recorder(None)
    monkeypatch.setattr(
        runtime_fingerprint,
        "build_engine_fingerprint",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must short-circuit")),
    )

    assert record_engine_event(
        stage="detector",
        requested_engine="comic-text-detector",
        resolved_engine="comic-text-detector",
        backend=object(),
        execution_status="succeeded",
        result_status="accepted",
    ) == {}


def test_record_engine_event_records_confirmed_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")

    class Backend:
        _backend = "contour-fallback"

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        record_engine_event(
            stage="detector",
            requested_engine="comic-text-detector",
            resolved_engine="contour-fallback",
            backend=Backend(),
            execution_status="succeeded",
            result_status="accepted",
            fallback_used=True,
            fallback_reason="model_load_failed",
        )
        recorder.finalize()
    finally:
        bind_recorder(None)

    manifest = json.loads(
        (tmp_path / "debug" / "e2e" / "debug_manifest.json").read_text(encoding="utf-8")
    )
    fingerprint = manifest["runtime_fingerprints"][0]
    assert fingerprint["executed_backend"] == "contour-fallback"
    assert fingerprint["execution_status"] == "succeeded"
    assert fingerprint["result_status"] == "accepted"
    assert fingerprint["fallback_used"] is True


def test_call_inpainter_records_only_successful_chapter_execution(tmp_path, monkeypatch):
    from vision_stack import runtime

    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")

    class FakeInpainter:
        _backend = "aot_inpainting"
        _requested_model = "aot-inpainting"

        def inpaint(self, image, _mask, **_kwargs):
            return image.copy()

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        image = __import__("numpy").zeros((8, 8, 3), dtype="uint8")
        mask = __import__("numpy").ones((8, 8), dtype="uint8") * 255
        result = runtime._call_inpainter(FakeInpainter(), image, mask)
        recorder.finalize()
    finally:
        bind_recorder(None)

    assert result.shape == image.shape
    manifest = json.loads(
        (tmp_path / "debug" / "e2e" / "debug_manifest.json").read_text(encoding="utf-8")
    )
    execution = [
        item
        for item in manifest["runtime_fingerprints"]
        if item["stage"] == "inpainter"
    ][0]
    assert execution["executed_backend"] == "aot_inpainting"
    assert execution["execution_status"] == "succeeded"
    assert execution["execution_context"] == "chapter"


def test_call_inpainter_does_not_record_failed_execution(tmp_path, monkeypatch):
    from vision_stack import runtime

    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")

    class FailingInpainter:
        _backend = "aot_inpainting"
        _requested_model = "aot-inpainting"

        def inpaint(self, _image, _mask, **_kwargs):
            raise RuntimeError("boom")

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        image = __import__("numpy").zeros((8, 8, 3), dtype="uint8")
        mask = __import__("numpy").ones((8, 8), dtype="uint8") * 255
        with __import__("pytest").raises(RuntimeError, match="boom"):
            runtime._call_inpainter(FailingInpainter(), image, mask)
        recorder.finalize()
    finally:
        bind_recorder(None)

    manifest = json.loads(
        (tmp_path / "debug" / "e2e" / "debug_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["runtime_fingerprints"] == []
