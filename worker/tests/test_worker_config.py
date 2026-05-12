from __future__ import annotations

from pathlib import Path

from worker.config import WorkerSettings


def test_fast_page_server_is_enabled_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TRADUZAI_FAST_PAGE_SERVER", raising=False)
    monkeypatch.setenv("TRADUZAI_PROJECT_ROOT", str(tmp_path))

    settings = WorkerSettings.from_env()

    assert settings.fast_page_server_enabled is True


def test_fast_page_server_can_be_disabled_from_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRADUZAI_FAST_PAGE_SERVER", "0")
    monkeypatch.setenv("TRADUZAI_PROJECT_ROOT", str(tmp_path))

    settings = WorkerSettings.from_env()

    assert settings.fast_page_server_enabled is False
