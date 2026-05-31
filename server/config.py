from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("TRADUZAI_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def _split_origins(value: str) -> list[str]:
    return [origin.strip() for origin in value.split(",") if origin.strip()]


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


@dataclass(slots=True)
class Settings:
    env: str = field(default_factory=lambda: os.environ.get("TRADUZAI_ENV", "dev"))
    bind: str = field(default_factory=lambda: os.environ.get("TRADUZAI_BIND", "local"))
    port: int = field(default_factory=lambda: int(os.environ.get("TRADUZAI_PORT", "8787")))
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "TRADUZAI_DATABASE_URL",
            f"sqlite:///{PROJECT_ROOT / 'data' / 'saas' / 'app.db'}",
        )
    )
    storage_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("TRADUZAI_STORAGE_DIR", str(PROJECT_ROOT / "data" / "saas" / "storage"))
        )
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _split_origins(os.environ.get("TRADUZAI_CORS_ORIGINS", "http://127.0.0.1:5174"))
    )
    admin_email: str = field(default_factory=lambda: os.environ.get("TRADUZAI_ADMIN_EMAIL", "admin@local"))
    admin_password: str | None = field(default_factory=lambda: os.environ.get("TRADUZAI_ADMIN_PASSWORD"))
    google_client_id: str | None = field(default_factory=lambda: os.environ.get("TRADUZAI_GOOGLE_CLIENT_ID"))
    google_client_secret: str | None = field(default_factory=lambda: os.environ.get("TRADUZAI_GOOGLE_CLIENT_SECRET"))
    worker_token: str | None = field(default_factory=lambda: os.environ.get("TRADUZAI_WORKER_TOKEN"))
    lease_timeout_seconds: int = field(
        default_factory=lambda: int(os.environ.get("TRADUZAI_LEASE_TIMEOUT_SECONDS", "180"))
    )
    heartbeat_interval_seconds: int = field(
        default_factory=lambda: int(os.environ.get("TRADUZAI_HEARTBEAT_INTERVAL_SECONDS", "20"))
    )
    max_file_mb: int = field(default_factory=lambda: int(os.environ.get("TRADUZAI_MAX_FILE_MB", "80")))
    max_job_mb: int = field(default_factory=lambda: int(os.environ.get("TRADUZAI_MAX_JOB_MB", "1000")))
    max_files_per_job: int = field(default_factory=lambda: int(os.environ.get("TRADUZAI_MAX_FILES_PER_JOB", "300")))
    max_zip_expanded_mb: int = field(
        default_factory=lambda: int(os.environ.get("TRADUZAI_MAX_ZIP_EXPANDED_MB", "1500"))
    )
    vast_api_key: str | None = field(default_factory=lambda: os.environ.get("VAST_API_KEY"))
    vast_instance_id: str | None = field(default_factory=lambda: os.environ.get("VAST_INSTANCE_ID"))
    vast_offer_id: str | None = field(default_factory=lambda: os.environ.get("VAST_OFFER_ID"))
    vast_offer_auto: bool = field(default_factory=lambda: _env_flag("VAST_OFFER_AUTO", False))
    vast_offer_limit: int = field(default_factory=lambda: int(os.environ.get("VAST_OFFER_LIMIT", "50")))
    vast_disk_gb: int = field(default_factory=lambda: int(os.environ.get("VAST_DISK_GB", "80")))
    vast_offer_max_dph: float = field(default_factory=lambda: float(os.environ.get("VAST_OFFER_MAX_DPH", "0.20")))
    vast_offer_min_gpu_ram_gb: int = field(default_factory=lambda: int(os.environ.get("VAST_OFFER_MIN_GPU_RAM_GB", "16")))
    vast_offer_min_reliability: float = field(
        default_factory=lambda: float(os.environ.get("VAST_OFFER_MIN_RELIABILITY", "0.98"))
    )
    vast_offer_min_dlperf: float = field(default_factory=lambda: float(os.environ.get("VAST_OFFER_MIN_DLPERF", "5.0")))
    vast_offer_min_direct_ports: int = field(
        default_factory=lambda: int(os.environ.get("VAST_OFFER_MIN_DIRECT_PORTS", "1"))
    )
    vast_offer_min_cuda: float = field(default_factory=lambda: float(os.environ.get("VAST_OFFER_MIN_CUDA", "12.1")))
    vast_offer_gpu_names: list[str] = field(
        default_factory=lambda: _split_csv(os.environ.get("VAST_OFFER_GPU_NAMES", ""))
    )
    vast_image: str | None = field(
        default_factory=lambda: os.environ.get("VAST_IMAGE", "vastai/pytorch:cuda-12.1.1-auto") or None
    )
    vast_runtype: str = field(default_factory=lambda: os.environ.get("VAST_RUNTYPE", "jupyter_direct"))
    vast_template_hash: str | None = field(default_factory=lambda: os.environ.get("VAST_TEMPLATE_HASH"))
    vast_autostart: bool = field(default_factory=lambda: _env_flag("VAST_AUTOSTART", False))
    vast_idle_stop_minutes: int = field(default_factory=lambda: int(os.environ.get("VAST_IDLE_STOP_MINUTES", "0")))
    vast_label: str = field(default_factory=lambda: os.environ.get("VAST_LABEL", "traduzai-worker"))
    vast_worker_api_url: str | None = field(
        default_factory=lambda: os.environ.get("VAST_WORKER_API_URL") or os.environ.get("TRADUZAI_PUBLIC_API_URL")
    )
    vast_repo_url: str = field(
        default_factory=lambda: os.environ.get(
            "VAST_REPO_URL",
            os.environ.get("TRADUZAI_REPO_URL", "https://github.com/OTheVyzx/TraduzAi.git"),
        )
    )
    vast_repo_branch: str = field(
        default_factory=lambda: os.environ.get(
            "VAST_REPO_BRANCH",
            os.environ.get("TRADUZAI_REPO_BRANCH", "Troca_de_motores"),
        )
    )
    vast_worker_name: str | None = field(
        default_factory=lambda: os.environ.get("VAST_WORKER_NAME") or os.environ.get("TRADUZAI_WORKER_NAME")
    )
    vast_require_gpu: bool = field(default_factory=lambda: _env_flag("VAST_REQUIRE_GPU", True))

    @property
    def host(self) -> str:
        return "0.0.0.0" if self.bind == "lan" else "127.0.0.1"

    @property
    def site_origin(self) -> str:
        if self.cors_origins:
            return self.cors_origins[0]
        return "http://127.0.0.1:5174"


def load_settings() -> Settings:
    return Settings()
