from __future__ import annotations

from typing import Any, Protocol

from server.config import Settings
from server.vast.client import VastClient


class VastServerlessClientProtocol(Protocol):
    def list_serverless_endpoints(self) -> list[dict[str, Any]]: ...

    def create_serverless_endpoint(self, *, name: str, min_load: int, target_load: int, cold_mult: float) -> dict[str, Any]: ...

    def list_serverless_workergroups(self, endpoint_id: str | None = None) -> list[dict[str, Any]]: ...

    def create_serverless_workergroup(
        self,
        *,
        endpoint_id: str,
        name: str,
        search_params: str,
        template_id: str | None = None,
        template_hash: str | None = None,
        max_load: int = 100,
        min_load: int = 0,
        test_workers: int = 0,
    ) -> dict[str, Any]: ...

    def route_serverless_request(self, *, endpoint_id: str, cost: float) -> dict[str, Any]: ...

    def post_serverless_worker(self, worker_url: str, route_path: str, payload: dict[str, Any]) -> dict[str, Any]: ...


def ensure_serverless_job_started(
    settings: Settings,
    job_id: str,
    *,
    client: VastServerlessClientProtocol | None = None,
) -> dict[str, Any]:
    if not settings.vast_autostart:
        return {"ok": False, "action": "disabled"}
    client = _client(settings, client)
    if isinstance(client, dict):
        return client
    ensured = ensure_serverless_endpoint(settings, client=client)
    if not ensured.get("ok"):
        return ensured
    endpoint_id = str(ensured["endpoint_id"])
    routed = client.route_serverless_request(endpoint_id=endpoint_id, cost=_job_route_cost(settings))
    worker_url = _route_worker_url(routed)
    if not worker_url:
        return {
            "ok": True,
            "action": "serverless_route_pending",
            "endpoint_id": endpoint_id,
            "route": _safe_route_summary(routed),
        }
    response = client.post_serverless_worker(
        worker_url,
        settings.vast_serverless_route_path,
        {
            "job_id": job_id,
            "payload": {"job_id": job_id},
            "auth_data": _route_auth_data(routed),
        },
    )
    return {
        "ok": bool(response.get("ok", True)),
        "action": "serverless_routed",
        "endpoint_id": endpoint_id,
        "worker_url": worker_url,
        "worker_response": response,
    }


def ensure_serverless_endpoint(
    settings: Settings,
    *,
    client: VastServerlessClientProtocol | None = None,
) -> dict[str, Any]:
    client = _client(settings, client)
    if isinstance(client, dict):
        return client
    missing = _missing_serverless_config(settings)
    if missing:
        return {"ok": False, "action": "missing_serverless_config", "missing": missing}

    endpoint = _resolve_endpoint(settings, client)
    endpoint_action = "existing_endpoint"
    if endpoint is None:
        endpoint = client.create_serverless_endpoint(
            name=settings.vast_serverless_endpoint_name,
            min_load=settings.vast_serverless_min_load,
            target_load=settings.vast_serverless_target_load,
            cold_mult=settings.vast_serverless_cold_mult,
        )
        endpoint_action = "created_endpoint"
    endpoint_id = str(_endpoint_id(endpoint))
    if not endpoint_id:
        return {"ok": False, "action": "serverless_endpoint_id_missing", "endpoint": endpoint}

    workergroup = _resolve_workergroup(settings, client, endpoint_id)
    workergroup_action = "existing_workergroup"
    if workergroup is None:
        workergroup = client.create_serverless_workergroup(
            endpoint_id=endpoint_id,
            name=settings.vast_serverless_workergroup_name,
            search_params=_serverless_search_params(settings),
            template_id=settings.vast_serverless_template_id,
            template_hash=settings.vast_serverless_template_hash,
            max_load=settings.vast_serverless_max_load,
            min_load=settings.vast_serverless_min_load,
            test_workers=settings.vast_serverless_test_workers,
        )
        workergroup_action = "created_workergroup"
    return {
        "ok": True,
        "action": "serverless_ready",
        "endpoint_id": endpoint_id,
        "endpoint_action": endpoint_action,
        "workergroup_id": _workergroup_id(workergroup),
        "workergroup_action": workergroup_action,
    }


def _client(settings: Settings, client: VastServerlessClientProtocol | None) -> VastServerlessClientProtocol | dict[str, Any]:
    if client is not None:
        return client
    if not settings.vast_api_key:
        return {"ok": False, "action": "missing_api_key"}
    return VastClient(settings.vast_api_key)


def _missing_serverless_config(settings: Settings) -> list[str]:
    missing = []
    if not settings.worker_token:
        missing.append("TRADUZAI_WORKER_TOKEN")
    if not settings.vast_worker_api_url:
        missing.append("VAST_WORKER_API_URL")
    if not settings.vast_serverless_template_id and not settings.vast_serverless_template_hash:
        missing.append("VAST_SERVERLESS_TEMPLATE_ID or VAST_SERVERLESS_TEMPLATE_HASH")
    return missing


def _resolve_endpoint(settings: Settings, client: VastServerlessClientProtocol) -> dict[str, Any] | None:
    if settings.vast_serverless_endpoint_id:
        return {"id": settings.vast_serverless_endpoint_id, "name": settings.vast_serverless_endpoint_name}
    for endpoint in client.list_serverless_endpoints():
        if _endpoint_name(endpoint) == settings.vast_serverless_endpoint_name:
            return endpoint
    return None


def _resolve_workergroup(
    settings: Settings,
    client: VastServerlessClientProtocol,
    endpoint_id: str,
) -> dict[str, Any] | None:
    for workergroup in client.list_serverless_workergroups(endpoint_id):
        if _workergroup_name(workergroup) == settings.vast_serverless_workergroup_name:
            return workergroup
    return None


def _endpoint_id(endpoint: dict[str, Any]) -> str:
    for key in ("id", "endpoint_id", "uid"):
        value = endpoint.get(key)
        if value is not None:
            return str(value)
    return ""


def _endpoint_name(endpoint: dict[str, Any]) -> str:
    for key in ("name", "endpoint_name"):
        value = endpoint.get(key)
        if isinstance(value, str):
            return value
    return ""


def _workergroup_id(workergroup: dict[str, Any]) -> str | None:
    for key in ("id", "workergroup_id", "uid"):
        value = workergroup.get(key)
        if value is not None:
            return str(value)
    return None


def _workergroup_name(workergroup: dict[str, Any]) -> str:
    for key in ("name", "workergroup_name"):
        value = workergroup.get(key)
        if isinstance(value, str):
            return value
    return ""


def _serverless_search_params(settings: Settings) -> str:
    if settings.vast_serverless_search_params:
        return settings.vast_serverless_search_params
    filters = [
        "verified=true",
        "rentable=true",
        "rented=false",
        "gpu_arch=nvidia",
        "num_gpus=1",
        f"gpu_ram>={settings.vast_offer_min_gpu_ram_gb * 1024}",
        f"reliability>={settings.vast_offer_min_reliability}",
        f"dlperf>={settings.vast_offer_min_dlperf}",
        f"dph_total<={settings.vast_offer_max_dph}",
        f"cuda_max_good>={settings.vast_offer_min_cuda}",
    ]
    if settings.vast_offer_gpu_names:
        gpu_values = ",".join(_serverless_gpu_name(name) for name in settings.vast_offer_gpu_names)
        filters.append(f"gpu_name in [{gpu_values}]")
    return " ".join(filters)


def _serverless_gpu_name(name: str) -> str:
    return "_".join(str(name).split())


def _job_route_cost(settings: Settings) -> float:
    return float(max(1, settings.vast_serverless_target_load))


def _route_worker_url(route: dict[str, Any]) -> str:
    for key in ("url", "worker_url", "endpoint_url"):
        value = route.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    worker = route.get("worker")
    if isinstance(worker, dict):
        return _route_worker_url(worker)
    return ""


def _route_auth_data(route: dict[str, Any]) -> dict[str, Any]:
    auth = route.get("auth_data")
    if isinstance(auth, dict):
        return auth
    return {key: route[key] for key in ("reqnum", "signature", "endpoint_id") if key in route}


def _safe_route_summary(route: dict[str, Any]) -> dict[str, Any]:
    return {key: route.get(key) for key in ("status", "message", "endpoint_id", "queue_position") if key in route}
