from pathlib import Path

import worker.__main__ as worker_main
from worker.config import WorkerSettings
from worker.runner import OutputArtifact


def test_run_once_can_disable_fast_page_runner_and_use_legacy_pipeline(tmp_path: Path, monkeypatch):
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
        fast_page_server_enabled=False,
    )
    uploaded = []
    events = []
    completed = []

    class FakeClient:
        def __init__(self, settings):
            self.settings = settings

        def register(self, capabilities):
            return {"worker_id": "worker-1"}

        def heartbeat(self, worker_id):
            pass

        def claim_job(self, worker_id, capabilities, job_id=None):
            return {
                "id": "job-1",
                "mode": "real",
                "input_download_url": "/input",
                "input_filename": "input.png",
            }

        def download_input(self, worker_id, job, dest_dir):
            dest_dir.mkdir(parents=True)
            input_path = dest_dir / "input.png"
            input_path.write_bytes(b"input")
            return input_path

        def post_event(self, worker_id, job_id, stage, kind, message, payload=None):
            events.append((stage, kind, message, payload or {}))

        def upload_artifact(self, worker_id, job_id, kind, path):
            uploaded.append((kind, Path(path).name))

        def complete(self, worker_id, job_id, page_count, processing_seconds):
            completed.append((job_id, page_count, processing_seconds))

        def fail(self, worker_id, job_id, error_code, error_message):
            raise AssertionError(error_message)

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    def fake_run_pipeline_job(settings, job, event_callback=None, page_artifact_callback=None):
        work_dir = settings.worker_work_dir / "jobs" / job["id"] / "work"
        translated_dir = work_dir / "translated"
        translated_dir.mkdir(parents=True)
        page_path = translated_dir / "001.png"
        page_path.write_bytes(b"translated")
        project_path = work_dir / "project.json"
        project_path.write_text("{}", encoding="utf-8")
        page_artifact = OutputArtifact("translated_image", page_path)
        if page_artifact_callback is not None:
            page_artifact_callback(
                page_artifact,
                {
                    "type": "page_completed",
                    "step": "page",
                    "message": "Pagina 1 concluida",
                    "current_page": 1,
                },
            )
        return {
            "page_count": 1,
            "processing_seconds": 0.25,
            "artifacts": [page_artifact, OutputArtifact("project_json", project_path)],
        }

    monkeypatch.setattr(worker_main, "WorkerClient", FakeClient)
    monkeypatch.setattr(worker_main, "HeartbeatLoop", FakeHeartbeat)
    monkeypatch.setattr(worker_main, "run_pipeline_job", fake_run_pipeline_job)

    assert worker_main.run_once(settings, mock=False) == 0

    assert uploaded == [("translated_image", "001.png"), ("project_json", "project.json")]
    assert any(event[0] == "page" and event[1] == "artifact" for event in events)
    assert completed == [("job-1", 1, 0.25)]


def test_worker_capabilities_advertise_fast_page_by_default(tmp_path: Path):
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
    )

    assert worker_main.build_worker_capabilities(settings, mock=False) == {
        "mode": ["mock", "real"],
        "runner": ["fast-page"],
    }


def test_worker_capabilities_can_advertise_legacy_runner(tmp_path: Path):
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
        fast_page_server_enabled=False,
    )

    assert worker_main.build_worker_capabilities(settings, mock=False) == {
        "mode": ["mock", "real"],
        "runner": ["legacy"],
    }


def test_run_once_uses_fast_page_runner_by_default(tmp_path: Path, monkeypatch):
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
    )
    completed = []

    class FakeClient:
        def __init__(self, settings):
            self.settings = settings

        def register(self, capabilities):
            return {"worker_id": "worker-1"}

        def heartbeat(self, worker_id):
            pass

        def claim_job(self, worker_id, capabilities, job_id=None):
            return {
                "id": "job-1",
                "mode": "real",
                "input_download_url": "/input",
                "input_filename": "input.png",
            }

        def download_input(self, worker_id, job, dest_dir):
            dest_dir.mkdir(parents=True)
            input_path = dest_dir / "input.png"
            input_path.write_bytes(b"input")
            return input_path

        def post_event(self, worker_id, job_id, stage, kind, message, payload=None):
            pass

        def upload_artifact(self, worker_id, job_id, kind, path):
            pass

        def complete(self, worker_id, job_id, page_count, processing_seconds):
            completed.append((job_id, page_count, processing_seconds))

        def fail(self, worker_id, job_id, error_code, error_message):
            raise AssertionError(error_message)

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    fast_calls = []

    def fake_run_fast_page_job(settings, job, fast_page_client, event_callback=None, page_artifact_callback=None):
        fast_calls.append((job["id"], fast_page_client))
        return {"page_count": 1, "processing_seconds": 0.1, "artifacts": []}

    monkeypatch.setattr(worker_main, "WorkerClient", FakeClient)
    monkeypatch.setattr(worker_main, "HeartbeatLoop", FakeHeartbeat)
    monkeypatch.setattr(worker_main, "get_fast_page_client", lambda settings: "hot-client")
    monkeypatch.setattr(worker_main, "run_fast_page_job", fake_run_fast_page_job)

    assert worker_main.run_once(settings, mock=False) == 0

    assert fast_calls == [("job-1", "hot-client")]
    assert completed == [("job-1", 1, 0.1)]


def test_main_closes_fast_page_client_on_exit(monkeypatch, tmp_path: Path):
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
    )
    closed = []

    class FakeFastClient:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(WorkerSettings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(worker_main, "run_once", lambda settings, mock, job_id=None: 0)
    worker_main._FAST_PAGE_CLIENT = FakeFastClient()

    assert worker_main.main(["--once"]) == 0

    assert closed == [True]
    assert worker_main._FAST_PAGE_CLIENT is None
