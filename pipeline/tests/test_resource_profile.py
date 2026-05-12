import sys

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


def test_resource_profile_blocks_and_kills_command_when_timeout_expires(tmp_path):
    result = measure_resource_profile(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path / "profile",
        sample_interval=0.02,
        timeout_seconds=0.1,
    )

    assert result["gate"]["status"] == "BLOCK"
    assert result["gate"]["timed_out"] is True
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
