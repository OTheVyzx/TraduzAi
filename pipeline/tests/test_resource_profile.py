import json
import sys
import threading
import time
from unittest.mock import patch

from pipeline.tools.measure_resource_profile import (
    measure_resource_profile,
    parse_nvidia_smi_memory_mb,
)


def test_resource_profile_records_elapsed_memory_cpu_and_exit_code(tmp_path):
    result = measure_resource_profile(
        [
            sys.executable,
            "-c",
            "import time; data='x'*2000000; time.sleep(0.15); print(len(data))",
        ],
        tmp_path / "profile",
        sample_interval=0.02,
    )

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["exit_code"] == 0
    assert result["gate"]["elapsed_seconds"] > 0
    assert result["gate"]["peak_rss_mb"] > 0
    assert result["gate"]["sample_count"] > 0
    assert (tmp_path / "profile" / "resources.json").exists()


def test_resource_profile_fails_when_command_exits_nonzero(tmp_path):
    result = measure_resource_profile(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        tmp_path / "profile",
        sample_interval=0.02,
    )

    assert result["gate"]["status"] == "FAIL"
    assert result["gate"]["exit_code"] == 3
    assert "command exited with code 3" in result["gate"]["reasons"]
    assert result["termination"]["classification"] == "unknown"
    assert result["termination"]["confidence"] == "low"


def test_resource_profile_blocks_and_kills_command_when_timeout_expires(tmp_path):
    result = measure_resource_profile(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path / "profile",
        sample_interval=0.02,
        timeout_seconds=0.1,
    )

    assert result["gate"]["status"] == "BLOCK"
    assert result["gate"]["timed_out"] is True
    assert result["termination"]["classification"] == "external_timeout"
    assert "resource profile timed out after 0.1s" in result["gate"]["reasons"]
    assert (tmp_path / "profile" / "resources.json").exists()


def test_resource_profile_does_not_deadlock_on_chatty_stdout(tmp_path):
    result = measure_resource_profile(
        [
            sys.executable,
            "-c",
            "for i in range(50000): print('x' * 100)",
        ],
        tmp_path / "profile",
        sample_interval=0.02,
        timeout_seconds=5,
    )

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["timed_out"] is False
    assert (tmp_path / "profile" / "command_stdout.log").exists()


def test_resource_profile_reports_nonzero_cpu_for_busy_command(tmp_path):
    result = measure_resource_profile(
        [
            sys.executable,
            "-c",
            "import time; end=time.time()+0.4\nwhile time.time()<end: pass",
        ],
        tmp_path / "profile",
        sample_interval=0.05,
        timeout_seconds=5,
    )

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["avg_cpu_percent"] > 0


def test_parse_nvidia_smi_memory_mb_returns_peak_value():
    assert parse_nvidia_smi_memory_mb("128\n512\n256\n") == 512
    assert parse_nvidia_smi_memory_mb("") is None


def test_resource_profile_persists_atomic_partial_journal_while_child_runs(tmp_path):
    out_dir = tmp_path / "profile"
    result_holder = {}

    thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result",
            measure_resource_profile(
                [sys.executable, "-c", "import time; time.sleep(0.4)"],
                out_dir,
                sample_interval=0.02,
                timeout_seconds=5,
            ),
        )
    )
    thread.start()
    journal_path = out_dir / "resources.partial.json"
    deadline = time.time() + 2
    while time.time() < deadline and not journal_path.exists():
        time.sleep(0.01)

    assert journal_path.exists()
    running = json.loads(journal_path.read_text(encoding="utf-8"))
    assert running["state"] == "running"
    assert running["sample_count"] >= 1

    thread.join(timeout=5)
    assert not thread.is_alive()
    finished = json.loads(journal_path.read_text(encoding="utf-8"))
    assert finished["state"] == "finished"
    assert result_holder["result"]["gate"]["status"] == "PASS"


def test_partial_journal_replace_failure_never_kills_measured_process(tmp_path):
    out_dir = tmp_path / "profile"
    with patch("pathlib.Path.replace", side_effect=PermissionError("locked")):
        result = measure_resource_profile(
            [sys.executable, "-c", "print('completed')"],
            out_dir,
            sample_interval=0.02,
            timeout_seconds=5,
        )

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["exit_code"] == 0
    assert "completed" in (out_dir / "command_stdout.log").read_text(encoding="utf-8")
