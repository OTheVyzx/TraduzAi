from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class VastApiError(RuntimeError):
    pass


class VastClient:
    def __init__(self, api_key: str, base_url: str = "https://console.vast.ai/api/v0", timeout: float = 20.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def show_instance(self, instance_id: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", f"/instances/{instance_id}/")
        except VastApiError as exc:
            if "404" in str(exc):
                return None
            raise

    def start_instance(self, instance_id: str) -> dict[str, Any]:
        return self._request("PUT", f"/instances/{instance_id}/", {"state": "running"})

    def stop_instance(self, instance_id: str) -> dict[str, Any]:
        return self._request("PUT", f"/instances/{instance_id}/", {"state": "stopped"})

    def search_offers(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        response = self._request("POST", "/bundles/", query)
        offers = response.get("offers", [])
        if isinstance(offers, list):
            return [offer for offer in offers if isinstance(offer, dict)]
        if isinstance(offers, dict):
            return [offers]
        return []

    def create_instance(
        self,
        offer_id: str,
        *,
        image: str | None = None,
        template_hash_id: str | None = None,
        runtype: str | None = None,
        env: dict[str, str] | None = None,
        disk: int | None = None,
        label: str | None = None,
        onstart: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"cancel_unavail": True, "target_state": "running"}
        if image:
            body["image"] = image
        if template_hash_id:
            body["template_hash_id"] = template_hash_id
        if runtype:
            body["runtype"] = runtype
            if runtype.startswith("jupyter"):
                body["use_jupyter_lab"] = True
                body["jupyter_dir"] = "/workspace"
        if env:
            body["env"] = env
        if disk:
            body["disk"] = disk
        if label:
            body["label"] = label
        if onstart:
            body["onstart"] = onstart
        return self._request("PUT", f"/asks/{offer_id}/", body)

    def list_serverless_endpoints(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/serverless/endpoints/")
        return _list_from_response(response, "endpoints")

    def create_serverless_endpoint(self, *, name: str, min_load: int, target_load: int, cold_mult: float) -> dict[str, Any]:
        return self._request(
            "PUT",
            "/serverless/endpoints/",
            {
                "endpoint_name": name,
                "min_load": min_load,
                "target_load": target_load,
                "cold_mult": cold_mult,
            },
        )

    def list_serverless_workergroups(self, endpoint_id: str | None = None) -> list[dict[str, Any]]:
        path = "/serverless/workergroups/"
        if endpoint_id:
            path = f"{path}?endpoint_id={endpoint_id}"
        response = self._request("GET", path)
        return _list_from_response(response, "workergroups")

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
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "endpoint_id": endpoint_id,
            "name": name,
            "search_params": search_params,
            "max_load": max_load,
            "min_load": min_load,
            "test_workers": test_workers,
        }
        if template_id:
            body["template_id"] = template_id
        if template_hash:
            body["template_hash"] = template_hash
        return self._request("PUT", "/serverless/workergroups/", body)

    def route_serverless_request(self, *, endpoint_id: str, cost: float) -> dict[str, Any]:
        return self._request_absolute("POST", "https://run.vast.ai/route/", {"endpoint_id": endpoint_id, "cost": cost})

    def post_serverless_worker(self, worker_url: str, route_path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = worker_url.rstrip("/") + "/" + route_path.strip("/")
        return self._request_absolute("POST", url, payload, include_api_auth=False)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request_absolute(method, f"{self.base_url}{path}", body)

    def _request_absolute(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        *,
        include_api_auth: bool = True,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if include_api_auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise VastApiError(f"Vast API HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise VastApiError(f"Vast API indisponivel: {exc.reason}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise VastApiError("Vast API retornou JSON invalido") from exc


def _list_from_response(response: dict[str, Any], key: str) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    value = response.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []
