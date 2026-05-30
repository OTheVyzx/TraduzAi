from __future__ import annotations

from typing import Any, Protocol

from server.config import Settings
from server.db import session_scope
from server.models import Job
from server.queue import BUSY_STATUSES
from server.vast.client import VastClient


class VastClientProtocol(Protocol):
    def show_instance(self, instance_id: str) -> dict[str, Any] | None: ...

    def start_instance(self, instance_id: str) -> dict[str, Any]: ...

    def stop_instance(self, instance_id: str) -> dict[str, Any]: ...

    def create_instance(
        self,
        offer_id: str,
        *,
        template_hash_id: str | None = None,
        env: str | None = None,
        label: str | None = None,
        onstart: str | None = None,
    ) -> dict[str, Any]: ...


RUNNING_STATES = {"running", "connected", "loaded", "success"}
STOPPED_STATES = {"stopped", "inactive"}


def ensure_worker_available(settings: Settings, *, client: VastClientProtocol | None = None) -> dict[str, Any]:
    if not settings.vast_autostart:
        return {"ok": False, "action": "disabled"}
    if client is None:
        if not settings.vast_api_key:
            return {"ok": False, "action": "missing_api_key"}
        client = VastClient(settings.vast_api_key)

    instance_id = settings.vast_instance_id
    if instance_id:
        instance = client.show_instance(instance_id)
        if instance is None:
            if not settings.vast_offer_id:
                return {"ok": False, "action": "instance_not_found", "instance_id": instance_id}
            return _create_instance(settings, client)
        status = _instance_status(instance)
        if status in RUNNING_STATES:
            return {"ok": True, "action": "already_running", "instance_id": str(instance_id), "status": status}
        client.start_instance(str(instance_id))
        return {"ok": True, "action": "started", "instance_id": str(instance_id), "status": status}

    if settings.vast_offer_id:
        return _create_instance(settings, client)
    return {"ok": False, "action": "missing_instance_or_offer"}


def stop_idle_worker_if_needed(
    settings: Settings,
    *,
    client: VastClientProtocol | None = None,
    has_pending_work: bool | None = None,
) -> dict[str, Any]:
    if not settings.vast_autostart:
        return {"ok": False, "action": "disabled"}
    if settings.vast_idle_stop_minutes <= 0:
        return {"ok": False, "action": "idle_stop_disabled"}
    if has_pending_work is None:
        has_pending_work = _has_pending_work(settings)
    if has_pending_work:
        return {"ok": False, "action": "busy"}
    if not settings.vast_instance_id:
        return {"ok": False, "action": "missing_instance"}
    if client is None:
        if not settings.vast_api_key:
            return {"ok": False, "action": "missing_api_key"}
        client = VastClient(settings.vast_api_key)
    instance = client.show_instance(settings.vast_instance_id)
    if instance is None:
        return {"ok": False, "action": "instance_not_found", "instance_id": settings.vast_instance_id}
    status = _instance_status(instance)
    if status not in RUNNING_STATES:
        return {"ok": True, "action": "already_stopped", "instance_id": settings.vast_instance_id, "status": status}
    client.stop_instance(settings.vast_instance_id)
    return {"ok": True, "action": "stopped", "instance_id": settings.vast_instance_id, "status": status}


def _create_instance(settings: Settings, client: VastClientProtocol) -> dict[str, Any]:
    if not settings.vast_offer_id:
        return {"ok": False, "action": "missing_offer"}
    created = client.create_instance(
        settings.vast_offer_id,
        template_hash_id=settings.vast_template_hash,
        label=settings.vast_label,
    )
    new_id = created.get("new_contract") or created.get("id")
    return {"ok": bool(created.get("success", True)), "action": "created", "instance_id": str(new_id)}


def _instance_status(instance: dict[str, Any]) -> str:
    for key in ("actual_status", "cur_state", "state", "status", "intended_status"):
        value = instance.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if instance.get("is_started") is True:
        return "running"
    if instance.get("is_started") is False:
        return "stopped"
    return "unknown"


def _has_pending_work(settings: Settings) -> bool:
    with session_scope(settings) as db:
        return db.query(Job).filter(Job.status.in_(["queued", *BUSY_STATUSES])).count() > 0
