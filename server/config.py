from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("TRADUZAI_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def _split_origins(value: str) -> list[str]:
    return [origin.strip() for origin in value.split(",") if origin.strip()]


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
