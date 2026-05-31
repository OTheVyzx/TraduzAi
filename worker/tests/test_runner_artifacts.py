from pathlib import Path

from worker.runner import collect_new_translated_page_artifacts, pipeline_mode_for_job
from worker.runner import collect_output_artifacts
from worker.runner import run_pipeline_job
from worker.runner import run_fast_page_job
from worker.config import WorkerSettings


def test_collect_output_artifacts_includes_translated_images(tmp_path: Path):
    runner_log = tmp_path / "runner.log"
    runner_log.write_text("runner\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    translated_dir = work_dir / "translated"
    originals_dir = work_dir / "originals"
    mask_dir = work_dir / "layers" / "mask"
    translated_dir.mkdir(parents=True)
    originals_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (work_dir / "pipeline.log").write_text("pipeline\n", encoding="utf-8")
    (work_dir / "project.json").write_text("{}", encoding="utf-8")
    (translated_dir / "002.jpg").write_bytes(b"jpg-2")
    (translated_dir / "001.png").write_bytes(b"png-1")
    (translated_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    (originals_dir / "001.png").write_bytes(b"original")
    (mask_dir / "001.png").write_bytes(b"mask")

    artifacts = collect_output_artifacts(runner_log, work_dir)

    assert [(artifact.kind, artifact.path.name) for artifact in artifacts] == [
        ("runner_log", "runner.log"),
        ("pipeline_log", "pipeline.log"),
        ("project_json", "project.json"),
        ("translated_image", "001.png"),
        ("translated_image", "002.jpg"),
        ("original_image", "001.png"),
        ("layer_mask", "001.png"),
    ]


def test_pipeline_mode_uses_project_config_manual_mode():
    assert pipeline_mode_for_job({"mode": "real", "project_config": {"mode": "manual"}}) == "manual"
    assert pipeline_mode_for_job({"mode": "real", "project_config": {"mode": "auto"}}) == "auto"


def test_collect_new_translated_page_artifacts_emits_each_page_once(tmp_path: Path):
    work_dir = tmp_path / "work"
    translated_dir = work_dir / "translated"
    translated_dir.mkdir(parents=True)
    seen: set[Path] = set()
    (translated_dir / "001.png").write_bytes(b"page-1")
    (translated_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    first = collect_new_translated_page_artifacts(work_dir, seen)

    assert [(artifact.kind, artifact.path.name) for artifact in first] == [("translated_image", "001.png")]
    assert {path.name for path in seen} == {"001.png"}

    assert collect_new_translated_page_artifacts(work_dir, seen) == []

    (translated_dir / "002.jpg").write_bytes(b"page-2")
    second = collect_new_translated_page_artifacts(work_dir, seen)

    assert [(artifact.kind, artifact.path.name) for artifact in second] == [("translated_image", "002.jpg")]


def test_run_pipeline_job_reports_translated_page_before_final_artifacts(tmp_path: Path, monkeypatch):
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
    )
    job = {"id": "job-1", "input_path": str(tmp_path / "input.png"), "mode": "real"}
    page_events = []

    class FakeStdout:
        def __iter__(self):
            work_dir = settings.worker_work_dir / "jobs" / job["id"] / "work"
            translated_dir = work_dir / "translated"
            translated_dir.mkdir(parents=True)
            (work_dir / "project.json").write_text('{"paginas":[{}]}', encoding="utf-8")
            (translated_dir / "001.png").write_bytes(b"translated")
            yield '{"type":"progress","step":"typeset","message":"pagina pronta"}\n'

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self):
            return 0

    def fake_popen(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setattr("worker.runner.subprocess.Popen", fake_popen)

    result = run_pipeline_job(
        settings,
        job,
        page_artifact_callback=lambda artifact, event: page_events.append((artifact, event)),
    )

    assert result["page_count"] == 1
    assert [(artifact.kind, artifact.path.name) for artifact, _event in page_events] == [("translated_image", "001.png")]
    assert page_events[0][1]["type"] == "page_completed"
    assert page_events[0][1]["current_page"] == 1


def test_run_fast_page_job_uses_hot_client_and_streams_artifact(tmp_path: Path):
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
    )
    job = {"id": "job-1", "input_path": str(tmp_path / "input.png"), "mode": "real"}
    event_payloads = []
    page_events = []

    class FakeFastPageClient:
        def __init__(self):
            self.requests = []

        def process_page(self, request, event_callback=None):
            self.requests.append(request)
            work_dir = Path(request["work_dir"])
            translated_dir = work_dir / "translated"
            translated_dir.mkdir(parents=True)
            (work_dir / "project.json").write_text('{"paginas":[{}]}', encoding="utf-8")
            (translated_dir / "001.png").write_bytes(b"translated")
            events = [
                {
                    "type": "page_completed",
                    "step": "page",
                    "current_page": 1,
                    "artifact_kind": "translated_image",
                    "artifact_path": "translated/001.png",
                },
                {"type": "complete", "page_count": 1},
            ]
            for event in events:
                event_callback(event)
            return events

    fast_client = FakeFastPageClient()

    result = run_fast_page_job(
        settings,
        job,
        fast_client,
        event_callback=event_payloads.append,
        page_artifact_callback=lambda artifact, event: page_events.append((artifact, event)),
    )

    assert result["page_count"] == 1
    assert fast_client.requests[0]["type"] == "process_page"
    assert fast_client.requests[0]["mode"] == "auto"
    assert [event["type"] for event in event_payloads] == ["page_completed", "complete"]
    assert [(artifact.kind, artifact.path.name) for artifact, _event in page_events] == [("translated_image", "001.png")]


def test_run_fast_page_job_merges_project_config_into_request(tmp_path: Path):
    settings = WorkerSettings(
        api_url="http://127.0.0.1:8787",
        worker_token="token",
        project_root=tmp_path,
        pipeline_main=tmp_path / "pipeline" / "main.py",
        pipeline_python="python",
        worker_name="worker",
        worker_work_dir=tmp_path / "worker",
    )
    job = {
        "id": "job-1",
        "input_path": str(tmp_path / "input.png"),
        "mode": "real",
        "project_config": {
            "mode": "auto",
            "_models_dir": str(tmp_path / "data-models"),
            "_vision_worker_path": str(tmp_path / "vision-worker.exe"),
            "contexto": {"sinopse": "ctx"},
            "idioma_origem": "ko",
            "idioma_destino": "pt",
            "runtime_profile": "balanced",
        },
    }

    class FakeFastPageClient:
        def __init__(self):
            self.requests = []

        def process_page(self, request, event_callback=None):
            self.requests.append(request)
            work_dir = Path(request["work_dir"])
            (work_dir / "translated").mkdir(parents=True)
            (work_dir / "translated" / "001.png").write_bytes(b"translated")
            events = [{"type": "complete", "page_count": 1}]
            if event_callback is not None:
                for event in events:
                    event_callback(event)
            return events

    fast_client = FakeFastPageClient()

    run_fast_page_job(settings, job, fast_client)

    request = fast_client.requests[0]
    assert request["models_dir"] == str(tmp_path / "data-models")
    assert request["vision_worker_path"] == str(tmp_path / "vision-worker.exe")
    assert request["contexto"] == {"sinopse": "ctx"}
    assert request["idioma_origem"] == "ko"
    assert request["idioma_destino"] == "pt"
    assert request["runtime_profile"] == "balanced"
