from __future__ import annotations

from server.config import Settings
from server.vast.orchestrator import ensure_worker_available, stop_idle_worker_if_needed


class FakeVastClient:
    def __init__(self, instance=None):
        self.instance = instance
        self.started = []
        self.stopped = []
        self.created = []
        self.offers = []
        self.offer_queries = []

    def show_instance(self, instance_id):
        if isinstance(self.instance, list):
            if not self.instance:
                return None
            if len(self.instance) == 1:
                return self.instance[0]
            return self.instance.pop(0)
        if self.instance is None:
            return None
        return self.instance

    def start_instance(self, instance_id):
        self.started.append(instance_id)
        return {"success": True}

    def stop_instance(self, instance_id):
        self.stopped.append(instance_id)
        return {"success": True}

    def search_offers(self, query):
        self.offer_queries.append(query)
        return self.offers

    def create_instance(self, offer_id, *, template_hash_id=None, env=None, disk=None, label=None, onstart=None):
        self.created.append(
            {
                "offer_id": offer_id,
                "template_hash_id": template_hash_id,
                "env": env,
                "disk": disk,
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

    result = ensure_worker_available(settings, client=client, scheduling_wait_seconds=0)

    assert result == {
        "ok": True,
        "action": "started",
        "instance_id": "38646242",
        "status": "stopped",
        "current_status": "stopped",
    }
    assert client.started == ["38646242"]
    assert client.created == []


def test_ensure_worker_available_does_not_restart_running_instance():
    settings = make_vast_settings()
    client = FakeVastClient({"id": 38646242, "actual_status": "running"})

    result = ensure_worker_available(settings, client=client, scheduling_wait_seconds=0)

    assert result == {"ok": True, "action": "already_running", "instance_id": "38646242", "status": "running"}
    assert client.started == []


def test_ensure_worker_available_creates_instance_when_no_existing_instance_is_configured():
    settings = make_vast_settings(vast_instance_id=None, vast_offer_id="12345", vast_template_hash="template-hash")
    client = FakeVastClient(None)

    result = ensure_worker_available(settings, client=client, scheduling_wait_seconds=0)

    assert result == {"ok": True, "action": "created", "instance_id": "987", "offer_id": "12345", "offer": None}
    assert len(client.created) == 1
    created = client.created[0]
    assert created["offer_id"] == "12345"
    assert created["template_hash_id"] == "template-hash"
    assert created["disk"] == 80
    assert created["label"] == "traduzai-worker"
    assert created["env"]["TRADUZAI_API_URL"] == "https://api.example.test"
    assert created["env"]["TRADUZAI_WORKER_TOKEN"] == "worker-token"
    assert created["env"]["TRADUZAI_REPO_BRANCH"] == "Troca_de_motores"
    assert created["env"]["TRADUZAI_REQUIRE_GPU"] == "1"
    assert "cat > /workspace/traduzai-worker.env" in created["onstart"]
    assert "TRADUZAI_API_URL=https://api.example.test" in created["onstart"]
    assert "bash \"$TRADUZAI_PROJECT_ROOT/scripts/vast/bootstrap.sh\"" in created["onstart"]
    assert "exec bash \"$TRADUZAI_PROJECT_ROOT/scripts/vast/start-worker.sh\"" in created["onstart"]


def test_ensure_worker_available_does_not_create_unconfigured_instance():
    settings = make_vast_settings(vast_instance_id=None, vast_offer_id="12345", vast_worker_api_url=None)
    client = FakeVastClient(None)

    result = ensure_worker_available(settings, client=client, scheduling_wait_seconds=0)

    assert result == {"ok": False, "action": "missing_worker_bootstrap_config", "missing": ["VAST_WORKER_API_URL"]}
    assert client.created == []


def test_ensure_worker_available_auto_selects_cheapest_matching_offer():
    settings = make_vast_settings(
        vast_instance_id=None,
        vast_offer_id=None,
        vast_offer_auto=True,
        vast_offer_max_dph=0.20,
        vast_offer_min_gpu_ram_gb=16,
        vast_offer_gpu_names=["Tesla P100", "RTX 3090"],
    )
    client = FakeVastClient(None)
    client.offers = [
        {"id": 111, "gpu_name": "RTX 3090", "gpu_ram": 24576, "dph_total": 0.19, "reliability": 0.99, "dlperf": 20},
        {"id": 222, "gpu_name": "Tesla P100", "gpu_ram": 16384, "dph_total": 0.10, "reliability": 0.98, "dlperf": 8},
    ]

    result = ensure_worker_available(settings, client=client, scheduling_wait_seconds=0)

    assert result["ok"] is True
    assert result["action"] == "created"
    assert result["offer_id"] == "222"
    assert client.created[0]["offer_id"] == "222"
    assert client.offer_queries == [
        {
            "limit": 50,
            "type": "ondemand",
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "gpu_arch": {"eq": "nvidia"},
            "num_gpus": {"eq": 1},
            "gpu_ram": {"gte": 16384},
            "disk_space": {"gte": 80},
            "reliability": {"gte": 0.98},
            "dlperf": {"gte": 5.0},
            "dph_total": {"lte": 0.20},
            "direct_port_count": {"gte": 1},
            "cuda_max_good": {"gte": 12.1},
            "order": [["dph_total", "asc"], ["reliability", "desc"], ["dlperf", "desc"]],
            "gpu_name": {"in": ["Tesla P100", "RTX 3090"]},
        }
    ]


def test_ensure_worker_available_replaces_existing_scheduling_instance():
    settings = make_vast_settings(
        vast_offer_id=None,
        vast_offer_auto=True,
        vast_offer_gpu_names=["RTX 3090"],
    )
    client = FakeVastClient(
        [
            {"id": 38646242, "actual_status": "scheduling"},
            {"id": 38646242, "actual_status": "scheduling"},
        ]
    )
    client.offers = [
        {
            "id": 777,
            "gpu_name": "RTX 3090",
            "gpu_ram": 24576,
            "dph_total": 0.15,
            "reliability": 0.99,
            "dlperf": 20,
        }
    ]

    result = ensure_worker_available(settings, client=client, scheduling_wait_seconds=0)

    assert result["ok"] is True
    assert result["action"] == "created_after_scheduling"
    assert result["previous_instance_id"] == "38646242"
    assert result["previous_status"] == "scheduling"
    assert result["offer_id"] == "777"
    assert client.created[0]["offer_id"] == "777"
    assert client.started == []


def test_ensure_worker_available_replaces_instance_that_becomes_scheduling_after_start():
    settings = make_vast_settings(
        vast_offer_id=None,
        vast_offer_auto=True,
        vast_offer_gpu_names=["RTX 4090"],
    )
    client = FakeVastClient(
        [
            {"id": 38646242, "actual_status": "stopped"},
            {"id": 38646242, "actual_status": "scheduling"},
            {"id": 38646242, "actual_status": "scheduling"},
        ]
    )
    client.offers = [
        {
            "id": 888,
            "gpu_name": "RTX 4090",
            "gpu_ram": 24576,
            "dph_total": 0.16,
            "reliability": 0.99,
            "dlperf": 30,
        }
    ]

    result = ensure_worker_available(settings, client=client, scheduling_wait_seconds=0)

    assert result["ok"] is True
    assert result["action"] == "created_after_scheduling"
    assert result["previous_instance_id"] == "38646242"
    assert result["previous_status"] == "scheduling"
    assert result["offer_id"] == "888"
    assert client.started == ["38646242"]
    assert client.created[0]["offer_id"] == "888"


def test_ensure_worker_available_reports_scheduling_without_replacement_offer():
    settings = make_vast_settings(vast_offer_id=None, vast_offer_auto=False)
    client = FakeVastClient(
        [
            {"id": 38646242, "actual_status": "scheduling"},
            {"id": 38646242, "actual_status": "scheduling"},
        ]
    )

    result = ensure_worker_available(settings, client=client, scheduling_wait_seconds=0)

    assert result == {
        "ok": False,
        "action": "scheduling_no_replacement_offer",
        "instance_id": "38646242",
        "status": "scheduling",
        "current_status": "scheduling",
    }
    assert client.created == []


def test_ensure_worker_available_is_disabled_without_autostart():
    settings = make_vast_settings(vast_autostart=False)
    client = FakeVastClient({"id": 38646242, "actual_status": "stopped"})

    result = ensure_worker_available(settings, client=client, scheduling_wait_seconds=0)

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
