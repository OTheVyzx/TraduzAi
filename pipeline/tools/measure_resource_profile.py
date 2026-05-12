"""Measure elapsed time and resource usage for a command."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import psutil


def measure_resource_profile(
    command: list[str],
    out_dir: str | Path | None = None,
    *,
    sample_interval: float = 0.5,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if not command:
        result = _result(
            command,
            status="BLOCK",
            reasons=["command is empty"],
            exit_code=None,
            elapsed_seconds=0.0,
            peak_rss_mb=0.0,
            avg_cpu_percent=0.0,
            peak_vram_mb=None,
            sample_count=0,
        )
        return _write_result(result, out_dir)

    output_path = Path(out_dir) if out_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    stdout_path = output_path / "command_stdout.log" if output_path is not None else None
    stderr_path = output_path / "command_stderr.log" if output_path is not None else None
    with ExitStack() as stack:
        stdout_target: Any
        stderr_target: Any
        if stdout_path is not None:
            stdout_target = stack.enter_context(stdout_path.open("w", encoding="utf-8"))
        else:
            stdout_target = subprocess.DEVNULL
        if stderr_path is not None:
            stderr_target = stack.enter_context(stderr_path.open("w", encoding="utf-8"))
        else:
            stderr_target = subprocess.DEVNULL

        process = psutil.Popen(
            command,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
        )
        peak_rss = 0
        cpu_samples: list[float] = []
        peak_vram_mb: int | None = None
        sample_count = 0
        timed_out = False
        cpu_times_by_pid = _capture_cpu_times(process)
        last_sample_time = time.perf_counter()

        while process.poll() is None:
            now = time.perf_counter()
            elapsed = now - start
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                timed_out = True
                _terminate_process_tree(process)
                break
            interval_seconds = max(0.001, now - last_sample_time)
            last_sample_time = now
            rss_bytes, cpu_percent = _sample_process_tree(
                process,
                cpu_times_by_pid,
                interval_seconds,
            )
            peak_rss = max(peak_rss, rss_bytes)
            cpu_samples.append(cpu_percent)
            sample_count += 1
            current_vram = _read_nvidia_smi_memory_mb()
            if current_vram is not None:
                peak_vram_mb = max(peak_vram_mb or 0, current_vram)
            time.sleep(max(0.01, sample_interval))

        try:
            process.wait(timeout=5)
        except psutil.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process)

        now = time.perf_counter()
        interval_seconds = max(0.001, now - last_sample_time)
        rss_bytes, cpu_percent = _sample_process_tree(
            process,
            cpu_times_by_pid,
            interval_seconds,
        )
        peak_rss = max(peak_rss, rss_bytes)
        if cpu_percent:
            cpu_samples.append(cpu_percent)
        sample_count += 1
        current_vram = _read_nvidia_smi_memory_mb()
        if current_vram is not None:
            peak_vram_mb = max(peak_vram_mb or 0, current_vram)

        elapsed = time.perf_counter() - start
        exit_code = int(process.returncode or 0) if process.returncode is not None else None
    reasons = ["resource profile captured"]
    status = "PASS"
    if timed_out:
        status = "BLOCK"
        reasons = [f"resource profile timed out after {timeout_seconds}s"]
    elif exit_code != 0:
        status = "FAIL"
        reasons = [f"command exited with code {exit_code}"]

    result = _result(
        command,
        status=status,
        reasons=reasons,
        exit_code=exit_code,
        elapsed_seconds=round(elapsed, 4),
        peak_rss_mb=round(peak_rss / (1024 * 1024), 3),
        avg_cpu_percent=round(sum(cpu_samples) / len(cpu_samples), 3)
        if cpu_samples
        else 0.0,
        peak_vram_mb=peak_vram_mb,
        sample_count=sample_count,
        timed_out=timed_out,
    )
    result["stdout_tail"] = _tail_file(stdout_path)
    result["stderr_tail"] = _tail_file(stderr_path)
    return _write_result(result, out_dir)


def parse_nvidia_smi_memory_mb(output: str) -> int | None:
    values: list[int] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    return max(values) if values else None


def _capture_cpu_times(process: psutil.Popen) -> dict[int, float]:
    samples: dict[int, float] = {}
    for proc in _process_tree(process):
        try:
            cpu_times = proc.cpu_times()
            samples[proc.pid] = float(cpu_times.user + cpu_times.system)
        except (psutil.Error, ProcessLookupError):
            continue
    return samples


def _sample_process_tree(
    process: psutil.Popen,
    cpu_times_by_pid: dict[int, float],
    interval_seconds: float,
) -> tuple[int, float]:
    rss_bytes = 0
    cpu_time_delta = 0.0
    for proc in _process_tree(process):
        try:
            with proc.oneshot():
                rss_bytes += int(proc.memory_info().rss)
                cpu_times = proc.cpu_times()
                cpu_time = float(cpu_times.user + cpu_times.system)
                previous = cpu_times_by_pid.get(proc.pid)
                cpu_times_by_pid[proc.pid] = cpu_time
                if previous is not None:
                    cpu_time_delta += max(0.0, cpu_time - previous)
        except (psutil.Error, ProcessLookupError):
            continue
    cpu_percent = (cpu_time_delta / interval_seconds) * 100.0
    return rss_bytes, cpu_percent


def _terminate_process_tree(process: psutil.Popen) -> None:
    procs = list(reversed(_process_tree(process)))
    for proc in procs:
        try:
            proc.terminate()
        except (psutil.Error, ProcessLookupError):
            continue
    _, alive = psutil.wait_procs(procs, timeout=3)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.Error, ProcessLookupError):
            continue


def _process_tree(process: psutil.Popen) -> list[psutil.Process]:
    processes: list[psutil.Process] = []
    try:
        processes.append(psutil.Process(process.pid))
        processes.extend(process.children(recursive=True))
    except (psutil.Error, ProcessLookupError):
        pass
    return processes


def _read_nvidia_smi_memory_mb() -> int | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return parse_nvidia_smi_memory_mb(completed.stdout)


def _result(
    command: list[str],
    *,
    status: str,
    reasons: list[str],
    exit_code: int | None,
    elapsed_seconds: float,
    peak_rss_mb: float,
    avg_cpu_percent: float,
    peak_vram_mb: int | None,
    sample_count: int,
    timed_out: bool = False,
) -> dict[str, Any]:
    return {
        "gate": {
            "name": "resource_profile",
            "status": status,
            "reasons": reasons,
            "command": command,
            "exit_code": exit_code,
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_mb": peak_rss_mb,
            "avg_cpu_percent": avg_cpu_percent,
            "peak_vram_mb": peak_vram_mb,
            "sample_count": sample_count,
            "timed_out": timed_out,
        }
    }


def _tail(text: str, *, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _tail_file(path: Path | None, *, max_chars: int = 2000) -> str:
    if path is None or not path.exists():
        return ""
    return _tail(path.read_text(encoding="utf-8", errors="replace"), max_chars=max_chars)


def _write_result(result: dict[str, Any], out_dir: str | Path | None) -> dict[str, Any]:
    if out_dir is not None:
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "resources.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    result = measure_resource_profile(
        command,
        args.out,
        sample_interval=args.sample_interval,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
