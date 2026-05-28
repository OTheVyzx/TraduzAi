from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TextIO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PipelineRunner = Callable[[str], None]
WarmupRunner = Callable[..., None]
InpaintWarmupRunner = Callable[..., None]
PageActionOptions = dict[str, str | None]
PageActionRunner = Callable[[Path, int, dict | None, PageActionOptions | None], None]
RegionalPageActionRunner = Callable[[Path, int, dict | None], None]
PreloadRunner = Callable[[Path, int, PageActionOptions | None], dict]


class FastPageSession:
    def __init__(
        self,
        pipeline_runner: PipelineRunner,
        warmup_runner: WarmupRunner | None = None,
        inpaint_warmup_runner: InpaintWarmupRunner | None = None,
        detect_runner: PageActionRunner | None = None,
        detect_boxes_runner: PageActionRunner | None = None,
        ocr_runner: PageActionRunner | None = None,
        reinpaint_runner: RegionalPageActionRunner | None = None,
        preload_detect_runner: PreloadRunner | None = None,
        preload_ocr_runner: PreloadRunner | None = None,
        session_id: str | None = None,
    ) -> None:
        self._pipeline_runner = pipeline_runner
        self._warmup_runner = warmup_runner or _default_warmup_runner
        self._inpaint_warmup_runner = inpaint_warmup_runner or _default_inpaint_warmup_runner
        self._detect_runner = detect_runner or _load_default_detect_runner()
        self._detect_boxes_runner = detect_boxes_runner or _load_default_detect_boxes_runner()
        self._ocr_runner = ocr_runner or _load_default_ocr_runner()
        self._reinpaint_runner = reinpaint_runner or _load_default_reinpaint_runner()
        self._preload_detect_runner = preload_detect_runner or _load_default_preload_detect_runner()
        self._preload_ocr_runner = preload_ocr_runner or _load_default_preload_ocr_runner()
        self._warmed_keys: set[tuple[str, str, str]] = set()
        self._warmed_inpaint_keys: set[tuple[str, str]] = set()
        self.session_id = session_id or f"fast-page-{uuid.uuid4().hex}"

    def handle(self, request: dict) -> list[dict]:
        request_type = request.get("type")
        if request_type == "warmup":
            return self._handle_warmup(request)
        if request_type == "warmup_inpaint":
            return self._handle_warmup_inpaint(request)
        if request_type == "process_page":
            return self._handle_process_page(request)
        if request_type == "editor_detect_page":
            return self._handle_editor_page_action(request, self._detect_runner, "editor_detect_page")
        if request_type == "editor_detect_boxes_page":
            return self._handle_editor_page_action(request, self._detect_boxes_runner, "editor_detect_boxes_page")
        if request_type == "editor_ocr_page":
            return self._handle_editor_page_action(request, self._ocr_runner, "editor_ocr_page")
        if request_type == "editor_reinpaint":
            return self._handle_editor_reinpaint(request)
        if request_type == "editor_preload_detect_ocr":
            return self._handle_editor_preload(request, self._preload_detect_runner, "detect_ocr")
        if request_type == "editor_preload_ocr_layers":
            return self._handle_editor_preload(request, self._preload_ocr_runner, "ocr_layers")
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

    def _handle_warmup_inpaint(self, request: dict) -> list[dict]:
        models_dir = str(request.get("models_dir") or "")
        profile = str(request.get("profile") or "quality")
        warm_key = (models_dir, profile)
        reused = warm_key in self._warmed_inpaint_keys
        if not reused:
            captured_stdout = io.StringIO()
            with contextlib.redirect_stdout(captured_stdout):
                self._inpaint_warmup_runner(models_dir=models_dir, profile=profile)
            for line in captured_stdout.getvalue().splitlines():
                print(f"[fast-page warmup_inpaint] {line}", file=sys.stderr, flush=True)
            self._warmed_inpaint_keys.add(warm_key)
        event = {"type": "ready", "session_id": self.session_id, "warm": True, "target": "inpaint"}
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

    def _handle_editor_reinpaint(self, request: dict) -> list[dict]:
        project_path = Path(_require_text(request, "project_path"))
        page_index = _require_int(request, "page_index")
        region = request.get("region")
        if region is not None and not isinstance(region, dict):
            raise ValueError("campo region deve ser objeto quando informado")

        captured_stdout = io.StringIO()
        try:
            with (
                _temporary_env({"TRADUZAI_INPAINT_ROI_TIGHTEN": "1"}),
                contextlib.redirect_stdout(captured_stdout),
            ):
                self._reinpaint_runner(project_path, page_index, region)
        except SystemExit as exc:
            raise RuntimeError(f"pipeline saiu durante editor_reinpaint: {exc}") from exc

        events = parse_pipeline_stdout_events(captured_stdout.getvalue())
        if not any(event.get("type") in {"complete", "error"} for event in events):
            raise RuntimeError("editor_reinpaint terminou sem evento complete/error")
        return events

    def _handle_editor_page_action(
        self,
        request: dict,
        runner: PageActionRunner,
        action_name: str,
    ) -> list[dict]:
        project_path = Path(_require_text(request, "project_path"))
        page_index = _require_int(request, "page_index")
        region = request.get("region")
        if region is not None and not isinstance(region, dict):
            raise ValueError("campo region deve ser objeto quando informado")
        options = _page_action_options_from_request(request)

        captured_stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured_stdout):
                runner(project_path, page_index, region, options)
        except SystemExit as exc:
            raise RuntimeError(f"pipeline saiu durante {action_name}: {exc}") from exc

        events = parse_pipeline_stdout_events(captured_stdout.getvalue())
        if not any(event.get("type") in {"complete", "error"} for event in events):
            raise RuntimeError(f"{action_name} terminou sem evento complete/error")
        return events

    def _handle_editor_preload(
        self,
        request: dict,
        runner: PreloadRunner,
        target: str,
    ) -> list[dict]:
        project_path = Path(_require_text(request, "project_path"))
        page_index = _require_int(request, "page_index")
        options = _page_action_options_from_request(request)

        captured_stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured_stdout):
                result = runner(project_path, page_index, options) or {}
        except SystemExit as exc:
            raise RuntimeError(f"pipeline saiu durante editor_preload_{target}: {exc}") from exc

        state = str(result.get("cache") or "ready")
        return [
            {
                "type": "ready" if state == "ready" else "accepted",
                "target": target,
                "cache": state,
                "session_id": self.session_id,
            }
        ]


@contextlib.contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    for key, value in request.items():
        if key == "type" or value is None:
            continue
        config[key] = value

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
    inpaint_warmup_runner: InpaintWarmupRunner | None = None,
    detect_runner: PageActionRunner | None = None,
    ocr_runner: PageActionRunner | None = None,
    reinpaint_runner: RegionalPageActionRunner | None = None,
    preload_detect_runner: PreloadRunner | None = None,
    preload_ocr_runner: PreloadRunner | None = None,
) -> None:
    active_session = session or FastPageSession(
        pipeline_runner=pipeline_runner or _load_default_pipeline_runner(),
        warmup_runner=warmup_runner,
        inpaint_warmup_runner=inpaint_warmup_runner,
        detect_runner=detect_runner,
        ocr_runner=ocr_runner,
        reinpaint_runner=reinpaint_runner,
        preload_detect_runner=preload_detect_runner,
        preload_ocr_runner=preload_ocr_runner,
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
    inpaint_warmup_runner: InpaintWarmupRunner | None = None,
    detect_runner: PageActionRunner | None = None,
    ocr_runner: PageActionRunner | None = None,
    reinpaint_runner: RegionalPageActionRunner | None = None,
    preload_detect_runner: PreloadRunner | None = None,
    preload_ocr_runner: PreloadRunner | None = None,
) -> int:
    serve_jsonl(
        sys.stdin,
        sys.stdout,
        pipeline_runner=pipeline_runner,
        warmup_runner=warmup_runner,
        inpaint_warmup_runner=inpaint_warmup_runner,
        detect_runner=detect_runner,
        ocr_runner=ocr_runner,
        reinpaint_runner=reinpaint_runner,
        preload_detect_runner=preload_detect_runner,
        preload_ocr_runner=preload_ocr_runner,
    )
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


def _optional_text(request: dict, key: str) -> str | None:
    value = request.get(key)
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _page_action_options_from_request(request: dict) -> PageActionOptions:
    return {
        "idioma_origem": _optional_text(request, "idioma_origem") or _optional_text(request, "source_lang"),
        "idioma_destino": _optional_text(request, "idioma_destino") or _optional_text(request, "target_lang"),
        "engine_preset_id": _optional_text(request, "engine_preset_id") or _optional_text(request, "enginePresetId"),
    }


def _require_int(request: dict, key: str) -> int:
    value = request.get(key)
    if value is None:
        raise ValueError(f"campo obrigatorio ausente: {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"campo {key} deve ser inteiro") from exc


def _default_warmup_runner(*, models_dir: str, profile: str, lang: str) -> None:
    from vision_stack.runtime import warmup_visual_stack

    warmup_visual_stack(models_dir=models_dir, profile=profile, run_sample=False, lang=lang)


def _default_inpaint_warmup_runner(*, models_dir: str, profile: str) -> None:
    from vision_stack.runtime import _configure_model_roots, _get_inpainter

    _configure_model_roots(models_dir)
    _get_inpainter(profile)


def _load_default_pipeline_runner() -> PipelineRunner:
    from main import _run_pipeline

    return _run_pipeline


def _load_default_detect_runner() -> PageActionRunner:
    from main import _run_detect_page

    return _run_detect_page


def _load_default_detect_boxes_runner() -> PageActionRunner:
    from main import _run_detect_boxes_page

    return _run_detect_boxes_page


def _load_default_ocr_runner() -> PageActionRunner:
    from main import _run_ocr_page

    return _run_ocr_page


def _load_default_reinpaint_runner() -> RegionalPageActionRunner:
    from main import _run_reinpaint

    return _run_reinpaint


def _load_default_preload_detect_runner() -> PreloadRunner:
    from main import _preload_detect_ocr_page

    return _preload_detect_ocr_page


def _load_default_preload_ocr_runner() -> PreloadRunner:
    from main import _preload_ocr_layers_page

    return _preload_ocr_layers_page


if __name__ == "__main__":
    raise SystemExit(serve_stdio())
