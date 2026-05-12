from __future__ import annotations

import argparse
import sys
import time

from worker.client import WorkerClient
from worker.config import WorkerSettings
from worker.fast_page import FastPageProcessClient
from worker.heartbeat import HeartbeatLoop
from worker.runner import OutputArtifact, run_fast_page_job, run_mock_job, run_pipeline_job


_FAST_PAGE_CLIENT: FastPageProcessClient | None = None


def doctor(settings: WorkerSettings) -> int:
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"ERRO: {error}")
        return 1
    settings.worker_work_dir.mkdir(parents=True, exist_ok=True)
    print("OK worker")
    return 0


def run_once(settings: WorkerSettings, mock: bool) -> int:
    client = WorkerClient(settings)
    capabilities = build_worker_capabilities(settings, mock)
    registration = client.register(capabilities)
    worker_id = registration["worker_id"]
    client.heartbeat(worker_id)
    job = client.claim_job(worker_id, capabilities)
    if job is None:
        print("Nenhum job na fila")
        return 0
    heartbeat = HeartbeatLoop(client, worker_id, settings.heartbeat_interval_seconds)
    heartbeat.start()

    def safe_post_event(stage: str, kind: str, message: str, payload: dict | None = None) -> None:
        try:
            client.post_event(worker_id, job["id"], stage, kind, message, payload)
        except Exception as exc:
            print(f"AVISO: evento do worker nao enviado: {exc}")

    uploaded_artifacts: set[tuple[str, str]] = set()

    def artifact_key(artifact: OutputArtifact) -> tuple[str, str]:
        try:
            path_key = str(artifact.path.resolve())
        except OSError:
            path_key = str(artifact.path)
        return artifact.kind, path_key

    def upload_artifact_once(artifact: OutputArtifact) -> bool:
        key = artifact_key(artifact)
        if key in uploaded_artifacts:
            return False
        client.upload_artifact(worker_id, job["id"], artifact.kind, artifact.path)
        uploaded_artifacts.add(key)
        return True

    def safe_stream_page_artifact(artifact: OutputArtifact, event: dict) -> None:
        output_artifact = _normalize_artifact(artifact)
        try:
            upload_artifact_once(output_artifact)
            safe_post_event(
                event.get("step") or "page",
                "artifact",
                event.get("message") or "Pagina concluida",
                event,
            )
        except Exception as exc:
            print(f"AVISO: artifact de pagina nao enviado: {exc}")

    try:
        safe_post_event("worker", "status", "Job iniciado")
        if mock or job.get("mode") == "mock":
            result = run_mock_job(settings, job)
        else:
            input_path = client.download_input(worker_id, job, settings.worker_work_dir / "jobs" / job["id"] / "input")
            job["input_path"] = str(input_path)
            event_callback = lambda event: safe_post_event(
                event.get("step") or event.get("type") or "pipeline",
                "error" if event.get("type") == "error" else "status",
                event.get("message") or event.get("type") or "pipeline",
                event,
            )
            if settings.fast_page_server_enabled:
                result = run_fast_page_job(
                    settings,
                    job,
                    get_fast_page_client(settings),
                    event_callback,
                    page_artifact_callback=safe_stream_page_artifact,
                )
            else:
                result = run_pipeline_job(
                    settings,
                    job,
                    event_callback,
                    page_artifact_callback=safe_stream_page_artifact,
                )
        for artifact in result["artifacts"]:
            output_artifact = _normalize_artifact(artifact)
            upload_artifact_once(output_artifact)
        client.complete(worker_id, job["id"], result["page_count"], result["processing_seconds"])
        return 0
    except Exception as exc:
        runner_log = settings.worker_work_dir / "jobs" / job["id"] / "runner.log" if "job" in locals() else None
        if runner_log is not None and runner_log.exists():
            try:
                client.upload_artifact(worker_id, job["id"], "runner_log", runner_log)
            except Exception:
                pass
        try:
            client.fail(worker_id, job["id"], "worker_error", str(exc))
        except Exception as fail_exc:
            print(f"ERRO: falha ao registrar erro do job: {fail_exc}")
        return 1
    finally:
        heartbeat.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="run")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    settings = WorkerSettings.from_env()
    if args.command == "doctor":
        return doctor(settings)
    try:
        while True:
            code = run_once(settings, args.mock)
            if args.once:
                return code
            time.sleep(3)
    finally:
        close_fast_page_client()


def build_worker_capabilities(settings: WorkerSettings, mock: bool) -> dict:
    if mock:
        return {"mode": ["mock"]}
    runner = "fast-page" if settings.fast_page_server_enabled else "legacy"
    return {"mode": ["mock", "real"], "runner": [runner]}


def _normalize_artifact(artifact) -> OutputArtifact:
    if isinstance(artifact, OutputArtifact):
        return artifact
    if artifact.name == "runner.log":
        return OutputArtifact("runner_log", artifact)
    if artifact.name == "pipeline.log":
        return OutputArtifact("pipeline_log", artifact)
    return OutputArtifact("project_json", artifact)


def get_fast_page_client(settings: WorkerSettings) -> FastPageProcessClient:
    global _FAST_PAGE_CLIENT
    if _FAST_PAGE_CLIENT is None:
        _FAST_PAGE_CLIENT = FastPageProcessClient(settings)
    return _FAST_PAGE_CLIENT


def close_fast_page_client() -> None:
    global _FAST_PAGE_CLIENT
    if _FAST_PAGE_CLIENT is None:
        return
    try:
        _FAST_PAGE_CLIENT.close()
    finally:
        _FAST_PAGE_CLIENT = None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
