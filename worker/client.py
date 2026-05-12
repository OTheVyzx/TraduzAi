from __future__ import annotations

import mimetypes
import time
from collections.abc import Callable
from pathlib import Path

import requests

from worker.config import WorkerSettings


class WorkerClient:
    def __init__(self, settings: WorkerSettings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {settings.worker_token}"})

    def _send_with_retries(self, send: Callable[[], requests.Response], *, attempts: int = 3) -> requests.Response:
        last_error: requests.RequestException | None = None
        for attempt in range(attempts):
            try:
                response = send()
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise
                time.sleep(0.6 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def register(self, capabilities: dict | None = None) -> dict:
        response = self._send_with_retries(
            lambda: self.session.post(
                f"{self.settings.api_url}/api/workers/register",
                json={"name": self.settings.worker_name, "capabilities": capabilities or {"mode": ["mock", "real"]}},
                timeout=20,
            )
        )
        return response.json()

    def heartbeat(self, worker_id: str, status: str = "online", payload: dict | None = None) -> None:
        self._send_with_retries(
            lambda: self.session.post(
                f"{self.settings.api_url}/api/workers/heartbeat",
                json={"worker_id": worker_id, "status": status, "payload": payload or {}},
                timeout=20,
            )
        )

    def claim_job(self, worker_id: str, capabilities: dict | None = None) -> dict | None:
        response = self._send_with_retries(
            lambda: self.session.post(
                f"{self.settings.api_url}/api/workers/claim-job",
                json={"worker_id": worker_id, "capabilities": capabilities or {"mode": ["mock", "real"]}},
                timeout=20,
            )
        )
        data = response.json()
        return data.get("job")

    def download_input(self, worker_id: str, job: dict, dest_dir: Path) -> Path:
        url = job.get("input_download_url")
        if not url:
            raise RuntimeError("job sem input_download_url")
        filename = Path(job.get("input_filename") or "input.bin").name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        response = self._send_with_retries(
            lambda: self.session.get(f"{self.settings.api_url}{url}", params={"worker_id": worker_id}, timeout=60)
        )
        dest.write_bytes(response.content)
        return dest

    def post_event(self, worker_id: str, job_id: str, stage: str, kind: str, message: str, payload: dict | None = None) -> None:
        self._send_with_retries(
            lambda: self.session.post(
                f"{self.settings.api_url}/api/workers/jobs/{job_id}/event",
                json={"worker_id": worker_id, "stage": stage, "kind": kind, "message": message, "payload": payload or {}},
                timeout=20,
            )
        )

    def upload_artifact(self, worker_id: str, job_id: str, kind: str, path: Path) -> None:
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        def _upload() -> requests.Response:
            with path.open("rb") as handle:
                return self.session.post(
                    f"{self.settings.api_url}/api/workers/jobs/{job_id}/artifact",
                    data={"worker_id": worker_id, "kind": kind},
                    files={"file": (path.name, handle, mime_type)},
                    timeout=120,
                )

        self._send_with_retries(_upload)

    def complete(self, worker_id: str, job_id: str, page_count: int, processing_seconds: float) -> None:
        self._send_with_retries(
            lambda: self.session.post(
                f"{self.settings.api_url}/api/workers/jobs/{job_id}/complete",
                json={"worker_id": worker_id, "page_count": page_count, "processing_seconds": processing_seconds},
                timeout=20,
            )
        )

    def fail(self, worker_id: str, job_id: str, error_code: str, error_message: str) -> None:
        self._send_with_retries(
            lambda: self.session.post(
                f"{self.settings.api_url}/api/workers/jobs/{job_id}/fail",
                json={"worker_id": worker_id, "error_code": error_code, "error_message": error_message},
                timeout=20,
            )
        )
