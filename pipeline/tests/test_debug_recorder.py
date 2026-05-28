import json
import sys
from pathlib import Path

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
