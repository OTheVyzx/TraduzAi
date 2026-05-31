from __future__ import annotations

import io

from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Settings
from server.tests.project_test_utils import make_settings


def test_settings_loads_vast_automation_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VAST_API_KEY", "vast-key")
    monkeypatch.setenv("VAST_INSTANCE_ID", "38646242")
    monkeypatch.setenv("VAST_AUTOSTART", "1")
    monkeypatch.setenv("VAST_IDLE_STOP_MINUTES", "7")
    monkeypatch.setenv("VAST_TEMPLATE_HASH", "template-hash")
    monkeypatch.setenv("VAST_OFFER_ID", "12345")
    monkeypatch.setenv("VAST_OFFER_AUTO", "1")
    monkeypatch.setenv("VAST_DISK_GB", "80")
    monkeypatch.setenv("VAST_OFFER_MAX_DPH", "0.15")
    monkeypatch.setenv("VAST_OFFER_MIN_GPU_RAM_GB", "16")
    monkeypatch.setenv("VAST_OFFER_MIN_RELIABILITY", "0.99")
    monkeypatch.setenv("VAST_OFFER_GPU_NAMES", "Tesla P100,RTX 3090")
    monkeypatch.setenv("VAST_WORKER_API_URL", "https://api.example.test")
    monkeypatch.setenv("VAST_REPO_BRANCH", "Troca_de_motores")
    monkeypatch.setenv("VAST_WORKER_NAME", "worker-auto")
    monkeypatch.setenv("VAST_REQUIRE_GPU", "1")

    settings = Settings(database_url=f"sqlite:///{tmp_path / 'app.db'}", storage_dir=tmp_path / "storage")

    assert settings.vast_api_key == "vast-key"
    assert settings.vast_instance_id == "38646242"
    assert settings.vast_autostart is True
    assert settings.vast_idle_stop_minutes == 7
    assert settings.vast_template_hash == "template-hash"
    assert settings.vast_offer_id == "12345"
    assert settings.vast_offer_auto is True
    assert settings.vast_disk_gb == 80
    assert settings.vast_offer_max_dph == 0.15
    assert settings.vast_offer_min_gpu_ram_gb == 16
    assert settings.vast_offer_min_reliability == 0.99
    assert settings.vast_offer_gpu_names == ["Tesla P100", "RTX 3090"]
    assert settings.vast_worker_api_url == "https://api.example.test"
    assert settings.vast_repo_branch == "Troca_de_motores"
    assert settings.vast_worker_name == "worker-auto"
    assert settings.vast_require_gpu is True


def test_create_job_triggers_vast_orchestrator_when_autostart_is_enabled(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    settings.vast_autostart = True
    settings.vast_api_key = "vast-key"
    calls = []

    def fake_ensure_worker_available(received_settings):
        calls.append(received_settings)
        return {"action": "started", "instance_id": "38646242"}

    monkeypatch.setattr("server.jobs.api.ensure_worker_available", fake_ensure_worker_available, raising=False)
    client = TestClient(create_app(settings))
    assert client.post("/api/auth/login", json={"email": "admin@local", "password": "secret123"}).status_code == 200

    upload = client.post(
        "/api/jobs",
        data={"obra": "Obra", "capitulo": "1", "mode": "real"},
        files={"file": ("page.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32, "image/png")},
    )

    assert upload.status_code == 200
    assert upload.json()["job"]["status"] == "queued"
    assert calls == [settings]


def test_manual_job_does_not_trigger_vast_orchestrator(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    settings.vast_autostart = True
    settings.vast_api_key = "vast-key"
    calls = []
    monkeypatch.setattr("server.jobs.api.ensure_worker_available", lambda s: calls.append(s), raising=False)
    client = TestClient(create_app(settings))
    assert client.post("/api/auth/login", json={"email": "admin@local", "password": "secret123"}).status_code == 200

    upload = client.post(
        "/api/jobs",
        data={"mode": "real", "project_config": '{"mode":"manual"}'},
        files={"file": ("page.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32, "image/png")},
    )

    assert upload.status_code == 200
    assert upload.json()["job"]["status"] == "completed"
    assert calls == []
