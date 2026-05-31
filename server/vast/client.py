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
        template_hash_id: str | None = None,
        env: dict[str, str] | None = None,
        label: str | None = None,
        onstart: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"cancel_unavail": True, "target_state": "running"}
        if template_hash_id:
            body["template_hash_id"] = template_hash_id
        if env:
            body["env"] = env
        if label:
            body["label"] = label
        if onstart:
            body["onstart"] = onstart
        return self._request("PUT", f"/asks/{offer_id}/", body)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
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
