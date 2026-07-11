import json
import sys
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools import DebugRecorder, bind_recorder, event, get_recorder


def test_debug_recorder_writes_versioned_manifest_and_config_snapshot(tmp_path):
    recorder = DebugRecorder(
        tmp_path,
        enabled=True,
        run_id="run-test",
        clock=lambda: "2026-05-17T18:00:00+00:00",
    )

    recorder.event("run", "start", {"obra": "Fixture"})
    recorder.write_json("00_run/config_snapshot.json", {"debug": True})
    recorder.finalize(config_snapshot={"debug": True})

    root = tmp_path / "debug" / "e2e"
    manifest = json.loads((root / "debug_manifest.json").read_text(encoding="utf-8"))
    config_snapshot = json.loads((root / "00_run" / "config_snapshot.json").read_text(encoding="utf-8"))
    events = (root / "events.jsonl").read_text(encoding="utf-8")

    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "run-test"
    assert config_snapshot["schema_version"] == 1
    assert config_snapshot["stage"] == "run"
    assert '"action": "start"' in events


def test_debug_recorder_disabled_does_not_create_e2e_tree(tmp_path):
    recorder = DebugRecorder(tmp_path, enabled=False, run_id="run-test")

    recorder.event("run", "start")
    recorder.write_json("00_run/config_snapshot.json", {"debug": False})
    recorder.finalize()

    assert not (tmp_path / "debug" / "e2e").exists()


def test_debug_recorder_records_own_write_failures_without_raising(tmp_path):
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")

    def fail_append(_path, _payload):
        raise RuntimeError("forced recorder failure")

    recorder._append_jsonl = fail_append

    recorder.event("run", "start")

    errors = (tmp_path / "debug" / "e2e" / "debug_errors.jsonl").read_text(encoding="utf-8")
    assert "forced recorder failure" in errors
    assert '"action": "start"' in errors


def test_debug_context_event_uses_bound_recorder(tmp_path):
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")

    bind_recorder(recorder)
    event("stdout", "emit", message_type="progress")

    assert get_recorder() is recorder
    events = (tmp_path / "debug" / "e2e" / "events.jsonl").read_text(encoding="utf-8")
    assert '"message_type": "progress"' in events


def test_debug_recorder_persists_latest_runtime_fingerprint_per_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")

    recorder.record_runtime_fingerprint(
        "ocr",
        {
            "requested_engine": "paddle-ocr-vl-1.6",
            "resolved_engine": None,
            "executed_backend": "none",
            "fallback_used": True,
            "fallback_reason": "runtime_unavailable",
        },
    )
    recorder.record_runtime_fingerprint(
        "ocr",
        {
            "requested_engine": "paddle-ocr-vl-1.6",
            "resolved_engine": "paddle-ocr-vl-1.5",
            "executed_backend": "paddleocr",
            "fallback_used": True,
            "fallback_reason": "resolved_engine_differs_from_request",
        },
    )
    recorder.record_runtime_fingerprint(
        "ocr",
        {
            "requested_engine": "paddle-ocr-vl-1.6",
            "resolved_engine": "paddle-ocr-vl-1.5",
            "executed_backend": "paddleocr",
            "fallback_used": True,
            "fallback_reason": "resolved_engine_differs_from_request",
        },
    )
    recorder.finalize()

    root = tmp_path / "debug" / "e2e"
    history = [
        json.loads(line)
        for line in (root / "00_run" / "runtime_fingerprints.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((root / "debug_manifest.json").read_text(encoding="utf-8"))

    assert len(history) == 2
    assert manifest["runtime_fingerprint_count"] == 1
    assert manifest["runtime_fingerprints"][0]["stage"] == "ocr"
    assert manifest["runtime_fingerprints"][0]["executed_backend"] == "paddleocr"


def test_debug_recorder_writes_deduplicated_lossless_canonical_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADUZAI_FLAG_VISUAL_BASELINE_LOSSLESS_V2", "1")
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    page = np.zeros((12, 16, 3), dtype=np.uint8)
    page[3:7, 4:10, :] = 127
    band = page[2:9, :, :].copy()

    first = recorder.write_canonical_image("page", page, page_id="page_001")
    duplicate = recorder.write_canonical_image("page", page.copy(), page_id="page_001")
    recorder.write_canonical_image(
        "final_band",
        band,
        page_id="page_001",
        band_id="page_001_band_000",
    )
    recorder.set_canonical_expected_coverage(
        page_ids=["page_001"],
        band_ids=["page_001_band_000"],
    )
    recorder.record_canonical_text_metrics(
        [
            {
                "page_id": "page_001",
                "band_id": "page_001_band_000",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_000",
                "font_size_final": 31,
                "render_balloon_containment": 0.98,
                "qa_metrics": {"source": "fixture"},
            }
        ]
    )
    recorder.finalize()

    root = tmp_path / "debug" / "e2e" / "00_run"
    manifest = json.loads((root / "canonical_manifest.json").read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256()
    expected_hash.update(b"uint8\0")
    expected_hash.update(b"12,16,3\0")
    expected_hash.update(page.tobytes(order="C"))

    assert first["buffer_sha256"] == expected_hash.hexdigest()
    assert first["buffer_color_space"] == "bgr"
    assert first["encoded_color_space"] == "rgb"
    assert duplicate["buffer_sha256"] == first["buffer_sha256"]
    assert manifest["entry_count"] == 2
    assert [entry["key"] for entry in manifest["entries"]] == [
        "final_band:page_001:page_001_band_000",
        "page:page_001:",
    ]
    assert manifest["expected_page_ids"] == ["page_001"]
    assert manifest["expected_final_band_ids"] == ["page_001_band_000"]
    assert manifest["text_metric_count"] == 1
    assert manifest["text_metrics"][0]["trace_id"] == "ocr_001@page_001_band_000"
    assert manifest["text_metrics"][0]["font_size_final"] == 31
    assert (root / "canonical_pages" / "page_001.png").exists()
    assert (root / "canonical_final_bands" / "page_001_band_000.png").exists()


def test_canonical_text_metrics_reject_duplicate_identity_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADUZAI_FLAG_VISUAL_BASELINE_LOSSLESS_V2", "1")
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")

    recorder.record_canonical_text_metrics(
        [
            {"trace_id": "ocr_001@page_001_band_000", "font_size_final": 31},
            {"trace_id": "ocr_001@page_001_band_000", "font_size_final": 18},
        ]
    )
    recorder.finalize()

    root = tmp_path / "debug" / "e2e"
    manifest = json.loads((root / "00_run" / "canonical_manifest.json").read_text(encoding="utf-8"))
    assert manifest["text_metric_count"] == 1
    assert manifest["text_metrics"][0]["font_size_final"] == 31
    errors = (root / "debug_errors.jsonl").read_text(encoding="utf-8")
    assert "duplicate_canonical_text_metric" in errors


def test_debug_recorder_canonical_write_failure_does_not_interrupt_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADUZAI_FLAG_VISUAL_BASELINE_LOSSLESS_V2", "1")
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")

    result = recorder.write_canonical_image("unsupported", np.zeros((2, 2, 3), dtype=np.uint8))

    assert result == {}
    errors = (tmp_path / "debug" / "e2e" / "debug_errors.jsonl").read_text(encoding="utf-8")
    assert '"action": "write_canonical_image"' in errors


def test_debug_recorder_default_false_flags_emit_no_v2_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", raising=False)
    monkeypatch.delenv("TRADUZAI_FLAG_VISUAL_BASELINE_LOSSLESS_V2", raising=False)
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")

    recorder.record_runtime_fingerprint("ocr", {"executed_backend": "paddleocr"})
    assert recorder.write_canonical_image(
        "page",
        np.zeros((2, 2, 3), dtype=np.uint8),
        page_id="page_001",
    ) == {}
    recorder.finalize()

    root = tmp_path / "debug" / "e2e"
    manifest = json.loads((root / "debug_manifest.json").read_text(encoding="utf-8"))
    assert "runtime_fingerprint_count" not in manifest
    assert "runtime_fingerprints" not in manifest
    assert "canonical_entry_count" not in manifest
    assert not (root / "00_run" / "runtime_fingerprints.jsonl").exists()
    assert not (root / "00_run" / "canonical_manifest.json").exists()


def test_runtime_fingerprint_deduplication_is_thread_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2", "1")
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    appended: list[dict] = []

    def slow_append(_path, payload):
        time.sleep(0.02)
        appended.append(dict(payload))

    monkeypatch.setattr(recorder, "_append_jsonl", slow_append)
    fingerprint = {
        "requested_engine": "paddle-ocr-vl-1.6",
        "resolved_engine": "paddleocr",
        "executed_backend": "paddleocr",
    }

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _index: recorder.record_runtime_fingerprint("ocr", fingerprint), range(16)))

    assert len(appended) == 1
    recorder.finalize()
    manifest = json.loads(
        (tmp_path / "debug" / "e2e" / "debug_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["runtime_fingerprint_summary"][0]["event_count"] == 16
