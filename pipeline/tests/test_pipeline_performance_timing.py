from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import main
from strip.run import _TimingScope


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_failed_run_finalizes_atomic_performance_timing(tmp_path: Path) -> None:
    recorder = main._PipelineTiming(
        work_dir=tmp_path,
        run_id="timing-failed-run",
    )

    error = RuntimeError("coverage exploded")
    with pytest.raises(RuntimeError, match="coverage exploded"):
        with recorder.measure("coverage_build"):
            raise error

    recorder.mark_failed(error)
    payload = recorder.finalize()

    final_path = tmp_path / "performance_timing.json"
    assert final_path.exists()
    assert not (tmp_path / "performance_timing.json.tmp").exists()
    assert payload == json.loads(final_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["run_id"] == "timing-failed-run"
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "coverage_build"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["exception_message"] == "coverage exploded"
    assert payload["total_sec"] >= 0
    assert payload["stages"]["coverage_build"]["status"] == "failed"


def test_running_recorder_persists_partial_snapshot(tmp_path: Path) -> None:
    recorder = main._PipelineTiming(work_dir=tmp_path, run_id="partial-run")

    with recorder.measure("extract"):
        partial_path = tmp_path / "performance_timing.partial.json"
        assert partial_path.exists()
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        assert partial["status"] == "running"
        assert partial["stages"]["extract"]["status"] == "running"

    recorder.finalize()
    assert not partial_path.exists()


def test_finalize_is_idempotent_and_does_not_rewrite(tmp_path: Path) -> None:
    recorder = main._PipelineTiming(work_dir=tmp_path, run_id="idempotent-run")
    first = recorder.finalize()
    final_path = tmp_path / "performance_timing.json"
    first_mtime = final_path.stat().st_mtime_ns

    second = recorder.finalize()

    assert second == first
    assert final_path.stat().st_mtime_ns == first_mtime


def test_nested_timers_do_not_double_count_instrumented_time(tmp_path: Path) -> None:
    recorder = main._PipelineTiming(work_dir=tmp_path, run_id="nested-run")

    with recorder.measure("parent"):
        with recorder.measure("child"):
            pass

    payload = recorder.finalize()

    assert payload["durations_sec"]["parent"] >= payload["durations_sec"]["child"]
    assert payload["instrumented_sec"] == pytest.approx(
        payload["durations_sec"]["parent"], abs=0.001
    )
    assert payload["unattributed_sec"] >= 0


def test_real_runner_failure_finalizes_timing_and_reraises(tmp_path: Path) -> None:
    config = {
        "source_path": str(tmp_path / "source.cbz"),
        "work_dir": str(tmp_path / "output"),
        "obra": "Fixture",
        "capitulo": 1,
        "mode": "real",
    }

    with patch.object(main, "_run_pipeline", side_effect=ValueError("broken input")):
        with pytest.raises(ValueError, match="broken input"):
            main._run_pipeline_runner_cli(config)

    payload = json.loads(
        (Path(config["work_dir"]) / "performance_timing.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "pipeline"
    assert payload["exception_type"] == "ValueError"


def test_profiler_write_failure_does_not_mask_pipeline_error(tmp_path: Path) -> None:
    config = {
        "source_path": str(tmp_path / "source.cbz"),
        "work_dir": str(tmp_path / "output"),
        "obra": "Fixture",
        "capitulo": 1,
        "mode": "real",
    }

    with patch.object(main, "_run_pipeline", side_effect=ValueError("original failure")):
        with patch.object(main._PipelineTiming, "finalize", side_effect=OSError("disk full")):
            with pytest.raises(ValueError, match="original failure"):
                main._run_pipeline_runner_cli(config)


def test_fake_clock_keeps_total_and_nested_spans_coherent(tmp_path: Path) -> None:
    clock = _FakeClock()
    recorder = main._PipelineTiming(
        work_dir=tmp_path,
        run_id="fake-clock-run",
        clock=clock,
    )

    with recorder.measure("parent"):
        clock.advance(2.0)
        with recorder.measure("child"):
            clock.advance(3.0)
        clock.advance(1.0)

    payload = recorder.finalize()

    assert payload["total_sec"] == 6.0
    assert payload["durations_sec"] == {"child": 3.0, "parent": 6.0}
    assert payload["instrumented_sec"] == 6.0
    assert payload["unattributed_sec"] == 0.0


def test_keyboard_interrupt_is_persisted_as_interrupted(tmp_path: Path) -> None:
    config = {
        "source_path": str(tmp_path / "source.cbz"),
        "work_dir": str(tmp_path / "output"),
        "obra": "Fixture",
        "capitulo": 1,
        "mode": "real",
    }

    with patch.object(main, "_run_pipeline", side_effect=KeyboardInterrupt()):
        with pytest.raises(KeyboardInterrupt):
            main._run_pipeline_runner_cli(config)

    payload = json.loads(
        (Path(config["work_dir"]) / "performance_timing.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "interrupted"
    assert payload["exception_type"] == "KeyboardInterrupt"


def test_strict_block_after_export_gate_is_technical_completion(tmp_path: Path) -> None:
    work_dir = tmp_path / "output"
    work_dir.mkdir()
    (work_dir / "project.json").write_text(
        json.dumps({"qa": {"export_gate": {"status": "BLOCK"}}}),
        encoding="utf-8",
    )
    config = {
        "source_path": str(tmp_path / "source.cbz"),
        "work_dir": str(work_dir),
        "obra": "Fixture",
        "capitulo": 1,
        "mode": "real",
    }

    with patch.object(main, "_run_pipeline", side_effect=SystemExit(2)):
        with pytest.raises(SystemExit) as exc_info:
            main._run_pipeline_runner_cli(config)

    assert exc_info.value.code == 2
    payload = json.loads((work_dir / "performance_timing.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["exception_type"] is None


def test_atomic_finalize_replaces_preexisting_invalid_json(tmp_path: Path) -> None:
    final_path = tmp_path / "performance_timing.json"
    final_path.write_text('{"truncated":', encoding="utf-8")
    recorder = main._PipelineTiming(work_dir=tmp_path, run_id="replace-run")

    payload = recorder.finalize()

    assert json.loads(final_path.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / "performance_timing.json.tmp").exists()


def test_successful_stage_is_terminal_completed(tmp_path: Path) -> None:
    recorder = main._PipelineTiming(work_dir=tmp_path, run_id="completed-stage-run")

    with recorder.measure("load_config"):
        pass

    payload = recorder.finalize()
    assert payload["status"] == "completed"
    assert payload["stages"]["load_config"]["status"] == "completed"


def test_recoverable_inner_stage_does_not_steal_terminal_failed_stage(
    tmp_path: Path,
) -> None:
    recorder = main._PipelineTiming(work_dir=tmp_path, run_id="terminal-stage-run")

    try:
        with recorder.measure("recoverable_owner_mask"):
            raise ValueError("localized rejection")
    except ValueError:
        pass

    terminal = RuntimeError("publication failed")
    with pytest.raises(RuntimeError):
        with recorder.measure("owner_page_composition"):
            raise terminal
    recorder.mark_failed(terminal)
    payload = recorder.finalize()

    assert payload["failed_stage"] == "owner_page_composition"


def test_strip_timing_scope_bridges_to_external_recorder(tmp_path: Path) -> None:
    recorder = main._PipelineTiming(work_dir=tmp_path, run_id="strip-bridge-run")
    telemetry = {"_performance_recorder": recorder}

    with _TimingScope(telemetry, "coverage_build"):
        pass

    payload = recorder.finalize()
    assert telemetry["durations_sec"]["coverage_build"] >= 0
    assert payload["stages"]["coverage_build"]["status"] == "completed"


def test_model_load_count_accumulates(tmp_path: Path) -> None:
    recorder = main._PipelineTiming(work_dir=tmp_path, run_id="model-load-run")

    recorder.record_model_load("detector", backend="comic-text-detector")
    recorder.record_model_load("detector", backend="comic-text-detector")

    payload = recorder.finalize()
    assert payload["models"]["detector"] == {
        "backend": "comic-text-detector",
        "load_count": 2,
    }
