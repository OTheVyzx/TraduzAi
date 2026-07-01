from __future__ import annotations

from server.config import Settings
from server.vast.serverless import ensure_serverless_endpoint, ensure_serverless_job_started


class FakeServerlessClient:
    def __init__(self):
        self.endpoints = []
        self.workergroups = []
        self.created_endpoints = []
        self.created_workergroups = []
        self.routes = []
        self.worker_posts = []
        self.route_response = {"url": "https://worker.example.test", "reqnum": 12, "signature": "sig"}

    def list_serverless_endpoints(self):
        return self.endpoints

    def create_serverless_endpoint(self, *, name, min_load, target_load, cold_mult):
        payload = {"id": "endpoint-new", "endpoint_name": name, "min_load": min_load, "target_load": target_load, "cold_mult": cold_mult}
        self.created_endpoints.append(payload)
        return payload

    def list_serverless_workergroups(self, endpoint_id=None):
        return self.workergroups

    def create_serverless_workergroup(
        self,
        *,
        endpoint_id,
        name,
        search_params,
        template_id=None,
        template_hash=None,
        max_load=100,
        min_load=0,
        test_workers=0,
    ):
        payload = {
            "id": "wg-new",
            "endpoint_id": endpoint_id,
            "name": name,
            "search_params": search_params,
            "template_id": template_id,
            "template_hash": template_hash,
            "max_load": max_load,
            "min_load": min_load,
            "test_workers": test_workers,
        }
        self.created_workergroups.append(payload)
        return payload

    def route_serverless_request(self, *, endpoint_id, cost):
        self.routes.append({"endpoint_id": endpoint_id, "cost": cost})
        return self.route_response

    def post_serverless_worker(self, worker_url, route_path, payload):
        self.worker_posts.append({"worker_url": worker_url, "route_path": route_path, "payload": payload})
        return {"ok": True, "code": 0}


def make_settings(**overrides):
    values = {
        "env": "dev",
        "database_url": "sqlite:///:memory:",
        "storage_dir": "/tmp/traduzai-storage",
        "admin_password": "secret123",
        "worker_token": "worker-token",
        "vast_autostart": True,
        "vast_api_key": "vast-key",
        "vast_provider": "serverless",
        "vast_worker_api_url": "https://api.example.test",
        "vast_serverless_template_hash": "template-hash",
        "vast_offer_gpu_names": ["RTX 3060", "RTX 4060"],
        "vast_offer_max_dph": 0.16,
    }
    values.update(overrides)
    return Settings(**values)


def test_ensure_serverless_endpoint_creates_endpoint_and_workergroup():
    settings = make_settings()
    client = FakeServerlessClient()

    result = ensure_serverless_endpoint(settings, client=client)

    assert result == {
        "ok": True,
        "action": "serverless_ready",
        "endpoint_id": "endpoint-new",
        "endpoint_action": "created_endpoint",
        "workergroup_id": "wg-new",
        "workergroup_action": "created_workergroup",
    }
    assert client.created_endpoints[0]["endpoint_name"] == "traduzai-serverless"
    assert client.created_workergroups[0]["template_hash"] == "template-hash"
    assert "dph_total<=0.16" in client.created_workergroups[0]["search_params"]
    assert "gpu_name in [RTX_3060,RTX_4060]" in client.created_workergroups[0]["search_params"]


def test_ensure_serverless_job_routes_specific_job_to_worker():
    settings = make_settings(vast_serverless_endpoint_id="endpoint-1")
    client = FakeServerlessClient()
    client.workergroups = [{"id": "wg-1", "name": "traduzai-worker"}]

    result = ensure_serverless_job_started(settings, "job-123", client=client)

    assert result["ok"] is True
    assert result["action"] == "serverless_routed"
    assert client.routes == [{"endpoint_id": "endpoint-1", "cost": 100.0}]
    assert client.worker_posts == [
        {
            "worker_url": "https://worker.example.test",
            "route_path": "/run",
            "payload": {
                "job_id": "job-123",
                "payload": {"job_id": "job-123"},
                "auth_data": {"reqnum": 12, "signature": "sig"},
            },
        }
    ]


def test_ensure_serverless_reports_missing_template():
    settings = make_settings(vast_serverless_template_hash=None, vast_serverless_template_id=None)
    client = FakeServerlessClient()

    result = ensure_serverless_endpoint(settings, client=client)

    assert result == {
        "ok": False,
        "action": "missing_serverless_config",
        "missing": ["VAST_SERVERLESS_TEMPLATE_ID or VAST_SERVERLESS_TEMPLATE_HASH"],
    }
    assert client.created_endpoints == []
