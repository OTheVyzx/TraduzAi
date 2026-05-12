from __future__ import annotations

import json
import subprocess
import time

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from worker.config import WorkerSettings


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class OutputArtifact:
    kind: str
    path: Path


PageArtifactCallback = Callable[[OutputArtifact, dict], None]


def run_mock_job(settings: WorkerSettings, job: dict) -> dict:
    job_dir = settings.worker_work_dir / "jobs" / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    runner_log = job_dir / "runner.log"
    project_json = job_dir / "project.json"
    started = time.monotonic()
    runner_log.write_text("mock job executado\n", encoding="utf-8")
    project_json.write_text(json.dumps({"job_id": job["id"], "mode": "mock"}, ensure_ascii=True), encoding="utf-8")
    return {
        "page_count": 1,
        "processing_seconds": time.monotonic() - started,
        "artifacts": [OutputArtifact("runner_log", runner_log), OutputArtifact("project_json", project_json)],
    }


def run_pipeline_job(
    settings: WorkerSettings,
    job: dict,
    event_callback: Callable[[dict], None] | None = None,
    page_artifact_callback: PageArtifactCallback | None = None,
) -> dict:
    job_dir = settings.worker_work_dir / "jobs" / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "runner_config.json"
    runner_log = job_dir / "runner.log"
    work_dir = job_dir / "work"
    config_path.write_text(
        json.dumps(build_pipeline_request(settings, job, work_dir, job_dir), ensure_ascii=True),
        encoding="utf-8",
    )
    started = time.monotonic()
    pipeline_error = None
    streamed_page_artifacts: set[Path] = set()
    with runner_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [settings.pipeline_python, str(settings.pipeline_main), str(config_path)],
            cwd=settings.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            event = _parse_pipeline_event(line)
            if event is not None:
                if event_callback is not None:
                    event_callback(event)
                if event.get("type") == "error":
                    pipeline_error = event.get("message") or "pipeline emitiu erro"
            _notify_new_translated_page_artifacts(work_dir, streamed_page_artifacts, page_artifact_callback)
        code = process.wait()
    _notify_new_translated_page_artifacts(work_dir, streamed_page_artifacts, page_artifact_callback)
    if code != 0:
        raise RuntimeError(f"pipeline saiu com codigo {code}")
    if pipeline_error:
        raise RuntimeError(str(pipeline_error))
    artifacts = collect_output_artifacts(runner_log, work_dir)
    page_count = 1
    project_path = work_dir / "project.json"
    if project_path.exists():
        try:
            project = json.loads(project_path.read_text(encoding="utf-8"))
            page_count = len(project.get("paginas") or []) or page_count
        except json.JSONDecodeError:
            page_count = 1
    return {"page_count": page_count, "processing_seconds": time.monotonic() - started, "artifacts": artifacts}


def run_fast_page_job(
    settings: WorkerSettings,
    job: dict,
    fast_page_client,
    event_callback: Callable[[dict], None] | None = None,
    page_artifact_callback: PageArtifactCallback | None = None,
) -> dict:
    job_dir = settings.worker_work_dir / "jobs" / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    runner_log = job_dir / "runner.log"
    work_dir = job_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner_log.write_text("fast-page server job\n", encoding="utf-8")
    started = time.monotonic()
    request = {"type": "process_page", **build_pipeline_request(settings, job, work_dir, job_dir)}
    events = fast_page_client.process_page(request)
    page_count = 1
    for event in events:
        if event_callback is not None:
            event_callback(event)
        if event.get("type") == "complete":
            page_count = int(event.get("page_count") or page_count)
        if event.get("type") == "page_completed" and page_artifact_callback is not None:
            artifact = _artifact_from_page_event(event, work_dir)
            if artifact is not None:
                page_artifact_callback(artifact, event)
    artifacts = collect_output_artifacts(runner_log, work_dir)
    return {"page_count": page_count, "processing_seconds": time.monotonic() - started, "artifacts": artifacts}


def build_pipeline_request(settings: WorkerSettings, job: dict, work_dir: Path, job_dir: Path) -> dict:
    project_config = project_config_for_job(job)
    models_dir = (
        job.get("models_dir")
        or project_config.get("models_dir")
        or project_config.get("_models_dir")
        or str(settings.project_root / "pipeline" / "models")
    )
    request = {
        "source_path": job.get("input_path"),
        "work_dir": str(work_dir),
        "models_dir": str(models_dir),
        "logs_dir": str(job_dir / "logs"),
        "job_id": job["id"],
        "obra": job.get("obra") or project_config.get("obra"),
        "capitulo": job.get("capitulo") or project_config.get("capitulo") or 1,
        "idioma_origem": job.get("src_lang") or project_config.get("idioma_origem") or "en",
        "idioma_destino": job.get("dst_lang") or project_config.get("idioma_destino") or "pt-BR",
        "mode": pipeline_mode_for_job(job),
    }
    optional_pairs = {
        "contexto": "contexto",
        "runtime_profile": "runtime_profile",
        "vision_worker_path": "vision_worker_path",
        "_vision_worker_path": "vision_worker_path",
        "export_mode": "export_mode",
        "skip_inpaint": "skip_inpaint",
        "skip_ocr": "skip_ocr",
    }
    for source_key, target_key in optional_pairs.items():
        if source_key in project_config and project_config[source_key] is not None:
            request[target_key] = project_config[source_key]
    if job.get("vision_worker_path"):
        request["vision_worker_path"] = job["vision_worker_path"]
    return request


def collect_new_translated_page_artifacts(work_dir: Path, seen_paths: set[Path]) -> list[OutputArtifact]:
    translated_dir = work_dir / "translated"
    if not translated_dir.exists():
        return []
    artifacts: list[OutputArtifact] = []
    for image_path in sorted(translated_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = image_path.resolve()
        if key in seen_paths:
            continue
        seen_paths.add(key)
        artifacts.append(OutputArtifact("translated_image", image_path))
    return artifacts


def _notify_new_translated_page_artifacts(
    work_dir: Path,
    seen_paths: set[Path],
    callback: PageArtifactCallback | None,
) -> None:
    if callback is None:
        return
    for artifact in collect_new_translated_page_artifacts(work_dir, seen_paths):
        callback(artifact, _build_page_completed_event(artifact, work_dir))


def _build_page_completed_event(artifact: OutputArtifact, work_dir: Path) -> dict:
    page_number = _page_number_from_filename(artifact.path)
    rel_path = _relative_artifact_path(artifact.path, work_dir)
    return {
        "type": "page_completed",
        "step": "page",
        "message": f"Pagina {page_number} concluida",
        "current_page": page_number,
        "artifact_kind": artifact.kind,
        "artifact_filename": artifact.path.name,
        "artifact_path": rel_path,
    }


def _page_number_from_filename(path: Path) -> int:
    digits = []
    for char in path.stem:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return 0
    return int("".join(digits))


def _relative_artifact_path(path: Path, work_dir: Path) -> str:
    try:
        return path.relative_to(work_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact_from_page_event(event: dict, work_dir: Path) -> OutputArtifact | None:
    raw_path = event.get("artifact_path")
    if not raw_path:
        return None
    artifact_path = Path(raw_path)
    if not artifact_path.is_absolute():
        artifact_path = work_dir / artifact_path
    if not artifact_path.exists():
        return None
    return OutputArtifact(str(event.get("artifact_kind") or "translated_image"), artifact_path)


def pipeline_mode_for_job(job: dict) -> str:
    project_config = project_config_for_job(job)
    requested_mode = project_config.get("mode") if isinstance(project_config, dict) else job.get("mode")
    return "manual" if requested_mode == "manual" else "auto"


def project_config_for_job(job: dict) -> dict:
    project_config = job.get("project_config")
    if isinstance(project_config, dict):
        return project_config
    if isinstance(project_config, str) and project_config.strip():
        try:
            parsed = json.loads(project_config)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def collect_output_artifacts(runner_log: Path, work_dir: Path) -> list[OutputArtifact]:
    artifacts = [OutputArtifact("runner_log", runner_log)]
    pipeline_log = work_dir / "pipeline.log"
    if pipeline_log.exists():
        artifacts.append(OutputArtifact("pipeline_log", pipeline_log))
    project_json = work_dir / "project.json"
    if project_json.exists():
        artifacts.append(OutputArtifact("project_json", project_json))
    translated_dir = work_dir / "translated"
    if translated_dir.exists():
        for image_path in sorted(translated_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                artifacts.append(OutputArtifact("translated_image", image_path))
    originals_dir = work_dir / "originals"
    if originals_dir.exists():
        for image_path in sorted(originals_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                artifacts.append(OutputArtifact("original_image", image_path))
    images_dir = work_dir / "images"
    if images_dir.exists():
        for image_path in sorted(images_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                artifacts.append(OutputArtifact("inpaint_image", image_path))
    for layer_name, kind in {"mask": "layer_mask", "brush": "layer_brush", "recovery": "layer_recovery"}.items():
        layer_dir = work_dir / "layers" / layer_name
        if layer_dir.exists():
            for image_path in sorted(layer_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    artifacts.append(OutputArtifact(kind, image_path))
    return artifacts


def _parse_pipeline_event(line: str) -> dict | None:
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) and "type" in payload else None
