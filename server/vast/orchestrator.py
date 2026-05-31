from __future__ import annotations

from typing import Any, Protocol

from server.config import Settings
from server.db import session_scope
from server.models import Job
from server.queue import BUSY_STATUSES
from server.vast.client import VastClient


class VastClientProtocol(Protocol):
    def search_offers(self, query: dict[str, Any]) -> list[dict[str, Any]]: ...

    def show_instance(self, instance_id: str) -> dict[str, Any] | None: ...

    def start_instance(self, instance_id: str) -> dict[str, Any]: ...

    def stop_instance(self, instance_id: str) -> dict[str, Any]: ...

    def create_instance(
        self,
        offer_id: str,
        *,
        template_hash_id: str | None = None,
        env: dict[str, str] | None = None,
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
            if not settings.vast_offer_id and not settings.vast_offer_auto:
                return {"ok": False, "action": "instance_not_found", "instance_id": instance_id}
            return _create_instance(settings, client)
        status = _instance_status(instance)
        if status in RUNNING_STATES:
            return {"ok": True, "action": "already_running", "instance_id": str(instance_id), "status": status}
        client.start_instance(str(instance_id))
        return {"ok": True, "action": "started", "instance_id": str(instance_id), "status": status}

    if settings.vast_offer_id or settings.vast_offer_auto:
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
    offer = _resolve_offer(settings, client)
    if offer is None:
        return {"ok": False, "action": "missing_offer"}
    missing_bootstrap = _missing_worker_bootstrap_config(settings)
    if missing_bootstrap:
        return {"ok": False, "action": "missing_worker_bootstrap_config", "missing": missing_bootstrap}
    worker_env = _build_worker_env(settings)
    created = client.create_instance(
        offer["id"],
        template_hash_id=settings.vast_template_hash,
        env=worker_env,
        label=settings.vast_label,
        onstart=_build_onstart_script(settings, worker_env),
    )
    new_id = created.get("new_contract") or created.get("id")
    return {
        "ok": bool(created.get("success", True)),
        "action": "created",
        "instance_id": str(new_id),
        "offer_id": offer["id"],
        "offer": offer.get("summary"),
    }


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


def _resolve_offer(settings: Settings, client: VastClientProtocol) -> dict[str, Any] | None:
    if settings.vast_offer_id:
        return {"id": settings.vast_offer_id}
    if not settings.vast_offer_auto:
        return None
    offers = client.search_offers(_build_offer_query(settings))
    if not offers:
        return None
    offer = _sort_offers(offers)[0]
    offer_id = offer.get("id") or offer.get("ask_contract_id")
    if offer_id is None:
        return None
    return {"id": str(offer_id), "summary": _offer_summary(offer)}


def _build_offer_query(settings: Settings) -> dict[str, Any]:
    query: dict[str, Any] = {
        "limit": settings.vast_offer_limit,
        "type": "ondemand",
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "gpu_arch": {"eq": "nvidia"},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": settings.vast_offer_min_gpu_ram_gb * 1024},
        "reliability": {"gte": settings.vast_offer_min_reliability},
        "dlperf": {"gte": settings.vast_offer_min_dlperf},
        "dph_total": {"lte": settings.vast_offer_max_dph},
        "direct_port_count": {"gte": settings.vast_offer_min_direct_ports},
        "cuda_max_good": {"gte": settings.vast_offer_min_cuda},
        "order": [["dph_total", "asc"], ["reliability", "desc"], ["dlperf", "desc"]],
    }
    if settings.vast_offer_gpu_names:
        query["gpu_name"] = {"in": settings.vast_offer_gpu_names}
    return query


def _sort_offers(offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        offers,
        key=lambda offer: (
            _number(offer.get("dph_total"), 999.0),
            -_number(offer.get("reliability"), 0.0),
            -_number(offer.get("dlperf"), 0.0),
        ),
    )


def _offer_summary(offer: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "gpu_name",
        "gpu_ram",
        "dph_total",
        "reliability",
        "dlperf",
        "cuda_max_good",
        "direct_port_count",
        "geolocation",
        "machine_id",
    ]
    return {key: offer.get(key) for key in keys if key in offer}


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _missing_worker_bootstrap_config(settings: Settings) -> list[str]:
    missing = []
    if not settings.vast_worker_api_url:
        missing.append("VAST_WORKER_API_URL")
    if not settings.worker_token:
        missing.append("TRADUZAI_WORKER_TOKEN")
    return missing


def _build_worker_env(settings: Settings) -> dict[str, str]:
    worker_name = settings.vast_worker_name or settings.vast_label
    return {
        "TRADUZAI_API_URL": settings.vast_worker_api_url or "",
        "TRADUZAI_WORKER_TOKEN": settings.worker_token or "",
        "TRADUZAI_WORKER_NAME": worker_name,
        "TRADUZAI_REPO_URL": settings.vast_repo_url,
        "TRADUZAI_REPO_BRANCH": settings.vast_repo_branch,
        "TRADUZAI_FAST_PAGE_SERVER": "1",
        "TRADUZAI_WORKER_WARMUP_ON_START": "1",
        "TRADUZAI_WARMUP_PROFILE": "quality",
        "TRADUZAI_WARMUP_LANG": "en",
        "TRADUZAI_REQUIRE_GPU": "1" if settings.vast_require_gpu else "0",
    }


def _build_onstart_script(settings: Settings, worker_env: dict[str, str]) -> str:
    repo_url = _shell_quote(settings.vast_repo_url)
    repo_branch = _shell_quote(settings.vast_repo_branch)
    project_root = "/workspace/TraduzAI"
    env_file = _format_worker_env_file(worker_env)
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
exec > >(tee -a /tmp/traduzai-onstart.log) 2>&1
echo "[traduzai] onstart iniciado em $(date -Is)"

export TRADUZAI_REPO_URL={repo_url}
export TRADUZAI_REPO_BRANCH={repo_branch}
export TRADUZAI_PROJECT_ROOT={project_root}

mkdir -p /workspace
cat > /workspace/traduzai-worker.env <<'TRADUZAI_WORKER_ENV'
{env_file}TRADUZAI_WORKER_ENV

if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends git curl ca-certificates
  fi
fi

if [ ! -d "$TRADUZAI_PROJECT_ROOT/.git" ]; then
  mkdir -p "$(dirname "$TRADUZAI_PROJECT_ROOT")"
  git clone --depth 1 --branch "$TRADUZAI_REPO_BRANCH" "$TRADUZAI_REPO_URL" "$TRADUZAI_PROJECT_ROOT"
else
  git -C "$TRADUZAI_PROJECT_ROOT" fetch --depth 1 origin "$TRADUZAI_REPO_BRANCH"
  git -C "$TRADUZAI_PROJECT_ROOT" checkout "$TRADUZAI_REPO_BRANCH"
  git -C "$TRADUZAI_PROJECT_ROOT" reset --hard "origin/$TRADUZAI_REPO_BRANCH"
fi

bash "$TRADUZAI_PROJECT_ROOT/scripts/vast/bootstrap.sh"
exec bash "$TRADUZAI_PROJECT_ROOT/scripts/vast/start-worker.sh"
"""


def _format_worker_env_file(values: dict[str, str]) -> str:
    return "\n".join(f"{key}={_env_value(value)}" for key, value in values.items()) + "\n"


def _env_value(value: str) -> str:
    cleaned = str(value).replace("\r", "").replace("\n", "")
    if not cleaned or any(char.isspace() or char in {'"', "'", "\\", "$", "#"} for char in cleaned):
        return _shell_quote(cleaned)
    return cleaned


def _shell_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"
