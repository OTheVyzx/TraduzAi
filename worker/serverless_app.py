from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from worker.__main__ import close_fast_page_client, doctor, run_once, warmup_fast_page_client
from worker.config import WorkerSettings


class ServerlessHandler(BaseHTTPRequestHandler):
    server_version = "TraduzAIServerless/0.1"

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        request = self._read_json()
        job_id = _job_id_from_request(request)
        settings = WorkerSettings.from_env()
        code = run_once(settings, mock=False, job_id=job_id)
        self._json({"ok": code == 0, "code": code, "job_id": job_id})

    def log_message(self, format: str, *args) -> None:
        print(f"[serverless-http] {self.address_string()} {format % args}", flush=True)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve() -> None:
    settings = WorkerSettings.from_env()
    doctor_code = doctor(settings)
    if doctor_code != 0:
        raise SystemExit(doctor_code)
    if os.environ.get("TRADUZAI_WORKER_WARMUP_ON_START", "1").strip().lower() not in {"0", "false", "no", "off"}:
        warmup_fast_page_client(settings)
    host = os.environ.get("TRADUZAI_SERVERLESS_HOST", "127.0.0.1")
    port = int(os.environ.get("TRADUZAI_SERVERLESS_PORT", "18000"))
    server = ThreadingHTTPServer((host, port), ServerlessHandler)
    print(f"OK serverless worker http://{host}:{port}", flush=True)
    print("TRADUZAI_SERVERLESS_READY", flush=True)
    try:
        server.serve_forever()
    finally:
        close_fast_page_client()


def _job_id_from_request(request: dict) -> str | None:
    if isinstance(request.get("job_id"), str):
        return request["job_id"]
    payload = request.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("job_id"), str):
        return payload["job_id"]
    return None


if __name__ == "__main__":
    serve()
