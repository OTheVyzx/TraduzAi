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


def test_artifact_profile_defaults_to_fast(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TRADUZAI_ARTIFACT_PROFILE", raising=False)
    monkeypatch.setenv("TRADUZAI_PROJECT_ROOT", str(tmp_path))

    settings = WorkerSettings.from_env()

    assert settings.artifact_profile == "fast"


def test_artifact_profile_accepts_editor_alias(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRADUZAI_ARTIFACT_PROFILE", "editor")
    monkeypatch.setenv("TRADUZAI_PROJECT_ROOT", str(tmp_path))

    settings = WorkerSettings.from_env()

    assert settings.artifact_profile == "full"


def test_artifact_upload_workers_are_clamped_positive(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRADUZAI_ARTIFACT_UPLOAD_WORKERS", "0")
    monkeypatch.setenv("TRADUZAI_PROJECT_ROOT", str(tmp_path))

    settings = WorkerSettings.from_env()

    assert settings.artifact_upload_workers == 1
