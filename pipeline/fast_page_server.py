from __future__ import annotations

import json
import contextlib
import io
import sys
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TextIO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PipelineRunner = Callable[[str], None]
WarmupRunner = Callable[..., None]


class FastPageSession:
    def __init__(
        self,
        pipeline_runner: PipelineRunner,
        warmup_runner: WarmupRunner | None = None,
        session_id: str | None = None,
    ) -> None:
        self._pipeline_runner = pipeline_runner
        self._warmup_runner = warmup_runner or _default_warmup_runner
        self._warmed_keys: set[tuple[str, str, str]] = set()
        self.session_id = session_id or f"fast-page-{uuid.uuid4().hex}"

    def handle(self, request: dict) -> list[dict]:
        request_type = request.get("type")
        if request_type == "warmup":
            return self._handle_warmup(request)
        if request_type == "process_page":
            return self._handle_process_page(request)
        if request_type == "shutdown":
            return [{"type": "bye", "session_id": self.session_id}]
        raise ValueError(f"tipo de request desconhecido: {request_type!r}")

    def _handle_warmup(self, request: dict) -> list[dict]:
        models_dir = str(request.get("models_dir") or "")
        profile = str(request.get("profile") or "max")
        lang = str(request.get("idioma_origem") or request.get("lang") or "en")
        warm_key = (models_dir, profile, lang)
        reused = warm_key in self._warmed_keys
        if not reused:
            self._warmup_runner(models_dir=models_dir, profile=profile, lang=lang)
            self._warmed_keys.add(warm_key)
        event = {"type": "ready", "session_id": self.session_id, "warm": True}
        if reused:
            event["reused"] = True
        return [event]

    def _handle_process_page(self, request: dict) -> list[dict]:
        config_path = write_fast_page_config(request)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        work_dir = Path(config["work_dir"])
        source_page_number = source_page_number_for_request(Path(config["source_path"]))
        captured_stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured_stdout):
                self._pipeline_runner(str(config_path))
        except SystemExit as exc:
            raise RuntimeError(f"pipeline saiu durante process_page: {exc}") from exc

        events = [
            event
            for event in parse_pipeline_stdout_events(captured_stdout.getvalue())
            if event.get("type") != "complete"
        ]
        artifact_events = [
            build_page_completed_event(path, work_dir, self.session_id, source_page_number=source_page_number)
            for path in collect_translated_artifacts(work_dir)
        ]
        events.extend(artifact_events)
        page_count = read_project_page_count(work_dir) or max(1, len(artifact_events))
        events.append(
            {
                "type": "complete",
                "session_id": self.session_id,
                "output_path": str(work_dir),
                "work_dir": str(work_dir),
                "page_count": page_count,
            }
        )
        return events


def write_fast_page_config(request: dict) -> Path:
    source_path = _require_text(request, "source_path")
    work_dir = Path(_require_text(request, "work_dir"))
    models_dir = _require_text(request, "models_dir")
    work_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "source_path": source_path,
        "work_dir": str(work_dir),
        "models_dir": models_dir,
        "logs_dir": str(Path(request.get("logs_dir") or work_dir / "logs")),
        "job_id": request.get("job_id") or request.get("id") or f"{uuid.uuid4().hex}",
        "obra": request.get("obra") or "",
        "capitulo": request.get("capitulo") or 1,
        "idioma_origem": request.get("idioma_origem") or request.get("src_lang") or "en",
        "idioma_destino": request.get("idioma_destino") or request.get("dst_lang") or "pt-BR",
        "mode": request.get("mode") or "auto",
    }
    for optional_key in (
        "contexto",
        "debug",
        "export_mode",
        "runtime_profile",
        "skip_inpaint",
        "skip_ocr",
        "vision_worker_path",
    ):
        if optional_key in request:
            config[optional_key] = request[optional_key]

    config_path = work_dir / "runner_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def serve_jsonl(
    input_stream: Iterable[str],
    output_stream: TextIO,
    *,
    session: FastPageSession | None = None,
    pipeline_runner: PipelineRunner | None = None,
    warmup_runner: WarmupRunner | None = None,
) -> None:
    active_session = session or FastPageSession(
        pipeline_runner=pipeline_runner or _load_default_pipeline_runner(),
        warmup_runner=warmup_runner,
    )
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        should_shutdown = False
        try:
            request = json.loads(line)
            should_shutdown = request.get("type") == "shutdown"
            events = active_session.handle(request)
        except Exception as exc:
            events = [{"type": "error", "session_id": active_session.session_id, "message": str(exc)}]
        for event in events:
            if "session_id" not in event:
                event["session_id"] = active_session.session_id
            output_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            output_stream.flush()
        if should_shutdown:
            break


def serve_stdio(
    *,
    pipeline_runner: PipelineRunner | None = None,
    warmup_runner: WarmupRunner | None = None,
) -> int:
    serve_jsonl(sys.stdin, sys.stdout, pipeline_runner=pipeline_runner, warmup_runner=warmup_runner)
    return 0


def collect_translated_artifacts(work_dir: Path) -> list[Path]:
    translated_dir = work_dir / "translated"
    if not translated_dir.exists():
        return []
    return [
        path
        for path in sorted(translated_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def parse_pipeline_stdout_events(stdout_text: str) -> list[dict]:
    events = []
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and "type" in event:
            events.append(event)
    return events


def build_page_completed_event(
    path: Path,
    work_dir: Path,
    session_id: str,
    *,
    source_page_number: int = 0,
) -> dict:
    page_number = source_page_number or page_number_from_filename(path)
    return {
        "type": "page_completed",
        "session_id": session_id,
        "step": "page",
        "message": f"Pagina {page_number} concluida",
        "current_page": page_number,
        "source_page_number": source_page_number or page_number,
        "artifact_kind": "translated_image",
        "artifact_filename": path.name,
        "artifact_path": relative_artifact_path(path, work_dir),
    }


def read_project_page_count(work_dir: Path) -> int:
    project_path = work_dir / "project.json"
    if not project_path.exists():
        return 0
    try:
        project = json.loads(project_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return len(project.get("paginas") or [])


def page_number_from_filename(path: Path) -> int:
    digits = []
    for char in path.stem:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else 0


def source_page_number_for_request(source_path: Path) -> int:
    if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return 0
    return page_number_from_filename(source_path)


def relative_artifact_path(path: Path, work_dir: Path) -> str:
    try:
        return path.relative_to(work_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _require_text(request: dict, key: str) -> str:
    value = request.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"campo obrigatorio ausente: {key}")
    return str(value)


def _default_warmup_runner(*, models_dir: str, profile: str, lang: str) -> None:
    from vision_stack.runtime import warmup_visual_stack

    warmup_visual_stack(models_dir=models_dir, profile=profile, run_sample=False, lang=lang)


def _load_default_pipeline_runner() -> PipelineRunner:
    from main import _run_pipeline

    return _run_pipeline


if __name__ == "__main__":
    raise SystemExit(serve_stdio())
