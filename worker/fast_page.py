from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from typing import Any

from worker.config import WorkerSettings


PopenFactory = Callable[..., Any]


class FastPageTransportError(RuntimeError):
    pass


class FastPageProcessClient:
    def __init__(self, settings: WorkerSettings, popen_factory: PopenFactory = subprocess.Popen) -> None:
        self._settings = settings
        self._popen_factory = popen_factory
        self._process = None

    def process_page(self, request: dict) -> list[dict]:
        payload = {"type": "process_page", **request}
        return self._request(payload, terminal_types={"complete"})

    def warmup(self, request: dict) -> list[dict]:
        payload = {"type": "warmup", **request}
        return self._request(payload, terminal_types={"ready"})

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._write({"type": "shutdown"})
        except Exception:
            pass
        try:
            self._process.wait(timeout=3)
        except (AttributeError, subprocess.TimeoutExpired):
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
        finally:
            self._process = None

    def _request(self, payload: dict, *, terminal_types: set[str], retry: bool = True) -> list[dict]:
        self._ensure_started()
        try:
            self._write(payload)
            return self._read_until(terminal_types)
        except (BrokenPipeError, OSError, FastPageTransportError):
            self._discard_process()
            if retry:
                return self._request(payload, terminal_types=terminal_types, retry=False)
            raise

    def _ensure_started(self) -> None:
        if self._process is not None and self._has_open_pipes():
            return
        self._process = self._popen_factory(
            [self._settings.pipeline_python, str(self._settings.pipeline_main), "--serve-fast-page"],
            cwd=self._settings.project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("processo fast-page iniciou sem stdin/stdout")

    def _has_open_pipes(self) -> bool:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            return False
        stdin_closed = bool(getattr(self._process.stdin, "closed", False))
        stdout_closed = bool(getattr(self._process.stdout, "closed", False))
        return not stdin_closed and not stdout_closed

    def _discard_process(self) -> None:
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._process.terminate()
            except Exception:
                pass
        self._process = None

    def _write(self, payload: dict) -> None:
        assert self._process is not None
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._process.stdin.flush()

    def _read_until(self, terminal_types: set[str]) -> list[dict]:
        assert self._process is not None
        assert self._process.stdout is not None
        events: list[dict] = []
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise FastPageTransportError("processo fast-page encerrou sem resposta completa")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FastPageTransportError(f"resposta fast-page invalida: {line.strip()}") from exc
            if not isinstance(event, dict):
                raise FastPageTransportError("resposta fast-page nao e objeto JSON")
            events.append(event)
            if event.get("type") == "error":
                raise RuntimeError(str(event.get("message") or "fast-page retornou erro"))
            if event.get("type") in terminal_types:
                return events
