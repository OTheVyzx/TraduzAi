from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class WorkerSettings:
    api_url: str
    worker_token: str
    project_root: Path
    pipeline_main: Path
    pipeline_python: str
    worker_name: str
    worker_work_dir: Path
    heartbeat_interval_seconds: int = 20
    fast_page_server_enabled: bool = True
    artifact_profile: str = "fast"
    artifact_upload_workers: int = 1

    @classmethod
    def from_env(cls) -> "WorkerSettings":
        project_root = Path(os.environ.get("TRADUZAI_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
        return cls(
            api_url=os.environ.get("TRADUZAI_API_URL", "http://127.0.0.1:8787").rstrip("/"),
            worker_token=os.environ.get("TRADUZAI_WORKER_TOKEN", "dev-token"),
            project_root=project_root,
            pipeline_main=Path(os.environ.get("TRADUZAI_PIPELINE_MAIN", project_root / "pipeline" / "main.py")),
            pipeline_python=os.environ.get("TRADUZAI_PIPELINE_PYTHON", "python"),
            worker_name=os.environ.get("TRADUZAI_WORKER_NAME", "admin-pc"),
            worker_work_dir=Path(os.environ.get("TRADUZAI_WORKER_WORK_DIR", project_root / "data" / "worker")),
            heartbeat_interval_seconds=int(os.environ.get("TRADUZAI_HEARTBEAT_INTERVAL_SECONDS", "20")),
            fast_page_server_enabled=_fast_page_server_enabled_from_env(),
            artifact_profile=_artifact_profile_from_env(),
            artifact_upload_workers=_positive_int_from_env("TRADUZAI_ARTIFACT_UPLOAD_WORKERS", 1),
        )

    def validate(self) -> list[str]:
        errors = []
        if not self.api_url.startswith(("http://", "https://")):
            errors.append("TRADUZAI_API_URL invalida")
        if not self.worker_token:
            errors.append("TRADUZAI_WORKER_TOKEN obrigatorio")
        if not self.project_root.exists():
            errors.append("TRADUZAI_PROJECT_ROOT nao existe")
        if self.artifact_profile not in {"fast", "full"}:
            errors.append("TRADUZAI_ARTIFACT_PROFILE deve ser fast ou full")
        if self.artifact_upload_workers < 1:
            errors.append("TRADUZAI_ARTIFACT_UPLOAD_WORKERS deve ser maior que zero")
        return errors


def _fast_page_server_enabled_from_env() -> bool:
    value = os.environ.get("TRADUZAI_FAST_PAGE_SERVER")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _artifact_profile_from_env() -> str:
    value = os.environ.get("TRADUZAI_ARTIFACT_PROFILE", "fast").strip().lower()
    aliases = {
        "prod": "fast",
        "production": "fast",
        "minimal": "fast",
        "debug": "full",
        "editor": "full",
    }
    return aliases.get(value, value)


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)
