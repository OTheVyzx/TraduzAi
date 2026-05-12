from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from worker.config import WorkerSettings
from worker.fast_page import FastPageProcessClient


class FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        pass


class FakeStdout:
    def __init__(self, lines: list[dict]) -> None:
        self.lines = deque(json.dumps(line) + "\n" for line in lines)

    def readline(self) -> str:
        return self.lines.popleft()


class FakeProcess:
    def __init__(self, lines: list[dict], poll_result=None) -> None:
        self.stdin = FakeStdin()
        self.stdout = FakeStdout(lines)
        self.terminated = False
        self.poll_result = poll_result

    def poll(self):
        if self.terminated:
            return 0
        return self.poll_result

    def terminate(self) -> None:
        self.terminated = True


def test_fast_page_client_shutdowns_hot_process(tmp_path: Path) -> None:
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
    )
    process = FakeProcess([{"type": "ready"}, {"type": "bye"}])
    client = FastPageProcessClient(settings, popen_factory=lambda *args, **kwargs: process)

    client.warmup({"models_dir": str(tmp_path / "models")})
    client.close()

    written_requests = [json.loads(line) for line in process.stdin.writes]
    assert written_requests[-1] == {"type": "shutdown"}
    assert process.terminated is True


def test_fast_page_client_reuses_one_process_for_multiple_pages(tmp_path: Path) -> None:
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
    )
    process = FakeProcess(
        [
            {"type": "page_completed", "artifact_path": "translated/001.png"},
            {"type": "complete", "page_count": 1},
            {"type": "page_completed", "artifact_path": "translated/002.png"},
            {"type": "complete", "page_count": 1},
        ]
    )
    popen_calls = []

    def fake_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return process

    client = FastPageProcessClient(settings, popen_factory=fake_popen)

    first_events = client.process_page({"type": "process_page", "source_path": "001.png"})
    second_events = client.process_page({"type": "process_page", "source_path": "002.png"})

    assert len(popen_calls) == 1
    assert first_events[-1]["type"] == "complete"
    assert second_events[0]["artifact_path"] == "translated/002.png"
    written_requests = [json.loads(line) for line in process.stdin.writes]
    assert written_requests == [
        {"type": "process_page", "source_path": "001.png"},
        {"type": "process_page", "source_path": "002.png"},
    ]


def test_fast_page_client_reuses_open_pipe_even_when_poll_reports_zero(tmp_path: Path) -> None:
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
    )
    process = FakeProcess(
        [
            {"type": "complete", "page_count": 1},
            {"type": "complete", "page_count": 1},
        ],
        poll_result=0,
    )
    popen_calls = []

    def fake_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return process

    client = FastPageProcessClient(settings, popen_factory=fake_popen)

    client.process_page({"source_path": "001.png"})
    client.process_page({"source_path": "002.png"})

    assert len(popen_calls) == 1


def test_fast_page_client_writes_ascii_json_for_unicode_paths(tmp_path: Path) -> None:
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
    )
    process = FakeProcess([{"type": "complete", "page_count": 1}])
    client = FastPageProcessClient(settings, popen_factory=lambda *args, **kwargs: process)

    client.process_page({"source_path": "N:/TraduzAI/exemplos/\ud658\uc0dd\ucc9c\ub9c8/113\ud654.cbz"})

    written = process.stdin.writes[0]
    written.encode("ascii")
    assert "\\ud658\\uc0dd\\ucc9c\\ub9c8" in written
