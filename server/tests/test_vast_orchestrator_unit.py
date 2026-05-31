from __future__ import annotations

from server.config import Settings
from server.vast.orchestrator import ensure_worker_available, stop_idle_worker_if_needed


class FakeVastClient:
    def __init__(self, instance=None):
        self.instance = instance
        self.started = []
        self.stopped = []
        self.created = []

    def show_instance(self, instance_id):
        if self.instance is None:
            return None
        return self.instance

    def start_instance(self, instance_id):
        self.started.append(instance_id)
        return {"success": True}

    def stop_instance(self, instance_id):
        self.stopped.append(instance_id)
        return {"success": True}

    def create_instance(self, offer_id, *, template_hash_id=None, env=None, label=None, onstart=None):
        self.created.append(
            {
                "offer_id": offer_id,
                "template_hash_id": template_hash_id,
                "env": env,
                "label": label,
                "onstart": onstart,
            }
        )
        return {"success": True, "new_contract": 987}


def make_vast_settings(**overrides):
    values = {
        "env": "dev",
        "database_url": "sqlite:///:memory:",
        "storage_dir": "/tmp/traduzai-storage",
        "admin_password": "secret123",
        "worker_token": "worker-token",
        "vast_autostart": True,
        "vast_api_key": "vast-key",
        "vast_instance_id": "38646242",
        "vast_worker_api_url": "https://api.example.test",
    }
    values.update(overrides)
    return Settings(**values)


def test_ensure_worker_available_starts_stopped_instance():
    settings = make_vast_settings()
    client = FakeVastClient({"id": 38646242, "actual_status": "stopped"})

    result = ensure_worker_available(settings, client=client)

    assert result == {"ok": True, "action": "started", "instance_id": "38646242", "status": "stopped"}
    assert client.started == ["38646242"]
    assert client.created == []


def test_ensure_worker_available_does_not_restart_running_instance():
    settings = make_vast_settings()
    client = FakeVastClient({"id": 38646242, "actual_status": "running"})

    result = ensure_worker_available(settings, client=client)

    assert result == {"ok": True, "action": "already_running", "instance_id": "38646242", "status": "running"}
    assert client.started == []


def test_ensure_worker_available_creates_instance_when_no_existing_instance_is_configured():
    settings = make_vast_settings(vast_instance_id=None, vast_offer_id="12345", vast_template_hash="template-hash")
    client = FakeVastClient(None)

    result = ensure_worker_available(settings, client=client)

    assert result == {"ok": True, "action": "created", "instance_id": "987"}
    assert len(client.created) == 1
    created = client.created[0]
    assert created["offer_id"] == "12345"
    assert created["template_hash_id"] == "template-hash"
    assert created["label"] == "traduzai-worker"
    assert "TRADUZAI_API_URL=https://api.example.test" in created["env"]
    assert "TRADUZAI_WORKER_TOKEN=worker-token" in created["env"]
    assert "TRADUZAI_REPO_BRANCH=Troca_de_motores" in created["env"]
    assert "TRADUZAI_REQUIRE_GPU=1" in created["env"]
    assert "cat > /workspace/traduzai-worker.env" in created["onstart"]
    assert "bash \"$TRADUZAI_PROJECT_ROOT/scripts/vast/bootstrap.sh\"" in created["onstart"]
    assert "exec bash \"$TRADUZAI_PROJECT_ROOT/scripts/vast/start-worker.sh\"" in created["onstart"]


def test_ensure_worker_available_does_not_create_unconfigured_instance():
    settings = make_vast_settings(vast_instance_id=None, vast_offer_id="12345", vast_worker_api_url=None)
    client = FakeVastClient(None)

    result = ensure_worker_available(settings, client=client)

    assert result == {"ok": False, "action": "missing_worker_bootstrap_config", "missing": ["VAST_WORKER_API_URL"]}
    assert client.created == []


def test_ensure_worker_available_is_disabled_without_autostart():
    settings = make_vast_settings(vast_autostart=False)
    client = FakeVastClient({"id": 38646242, "actual_status": "stopped"})

    result = ensure_worker_available(settings, client=client)

    assert result == {"ok": False, "action": "disabled"}
    assert client.started == []


def test_stop_idle_worker_stops_running_configured_instance():
    settings = make_vast_settings(vast_idle_stop_minutes=10)
    client = FakeVastClient({"id": 38646242, "actual_status": "running"})

    result = stop_idle_worker_if_needed(settings, client=client, has_pending_work=False)

    assert result == {"ok": True, "action": "stopped", "instance_id": "38646242", "status": "running"}
    assert client.stopped == ["38646242"]


def test_stop_idle_worker_keeps_instance_when_queue_has_work():
    settings = make_vast_settings(vast_idle_stop_minutes=10)
    client = FakeVastClient({"id": 38646242, "actual_status": "running"})

    result = stop_idle_worker_if_needed(settings, client=client, has_pending_work=True)

    assert result == {"ok": False, "action": "busy"}
    assert client.stopped == []
