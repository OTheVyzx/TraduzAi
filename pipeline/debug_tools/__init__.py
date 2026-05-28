from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from .recorder import DebugLevel, DebugRecorder

_current: ContextVar[DebugRecorder | None] = ContextVar("traduzai_debug_recorder", default=None)


def get_recorder() -> DebugRecorder | None:
    return _current.get()


def bind_recorder(recorder: DebugRecorder | None) -> None:
    _current.set(recorder)


def event(stage: str, action: str, **payload: Any) -> None:
    recorder = get_recorder()
    if recorder and recorder.enabled:
        recorder.event(stage, action, payload)


__all__ = ["DebugLevel", "DebugRecorder", "bind_recorder", "event", "get_recorder"]
