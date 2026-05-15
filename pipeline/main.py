"""
TraduzAi Pipeline - Entry point
Receives a config JSON path as argument, runs the full pipeline,
and outputs JSON progress messages to stdout for the Tauri sidecar to consume.
"""

import os
import copy
import json
import math
import re
import shutil
import sys
import time
import faulthandler
import logging
import contextlib
from pathlib import Path

# Adiciona o diretório da pipeline ao path para resolver imports locais no Pyright/Linter
pipeline_root = Path(__file__).parent.absolute()
if str(pipeline_root) not in sys.path:
    sys.path.insert(0, str(pipeline_root))

# Lazy imports are moved inside functions to allow fast --hardware-info and --list-supported-languages calls

from utils.decision_log import configure_decision_trace, finalize_decision_trace
from typesetter.style_policy import normalize_auto_typesetting_style
from layout.simple_text_geometry import resolve_text_anchor_bbox, sanitize_simple_text_geometry

_EMIT_STDOUT_FAILED = False
_PIPELINE_FILE_HANDLER: logging.Handler | None = None
logger = logging.getLogger(__name__)


class _PipelineTiming:
    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._durations: dict[str, float] = {}
        self._events: list[dict] = []

    @contextlib.contextmanager
    def measure(self, stage: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - started)

    def add(self, stage: str, seconds: float) -> None:
        self._durations[stage] = self._durations.get(stage, 0.0) + float(seconds)
        self._events.append({"stage": stage, "seconds": round(float(seconds), 4)})

    def snapshot(self, *, total_seconds: float | None = None, extra: dict | None = None) -> dict:
        total = float(total_seconds) if total_seconds is not None else time.perf_counter() - self._started
        durations = {stage: round(seconds, 4) for stage, seconds in sorted(self._durations.items())}
        instrumented = round(sum(durations.values()), 4)
        payload = {
            "total_sec": round(total, 4),
            "instrumented_sec": instrumented,
            "unattributed_sec": round(max(0.0, total - instrumented), 4),
            "durations_sec": durations,
            "events": list(self._events),
        }
        if extra:
            payload.update(extra)
        return payload


class _AutoClosingFileHandler(logging.Handler):
    def __init__(self, log_path: Path):
        super().__init__(level=logging.INFO)
        self._log_path = Path(log_path)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except Exception:
            self.handleError(record)


def _configure_pipeline_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _attach_work_dir_log_handler(work_dir: str | Path) -> Path:
    global _PIPELINE_FILE_HANDLER

    log_path = Path(work_dir) / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()

    _detach_work_dir_log_handler()

    log_path.write_text("", encoding="utf-8")
    handler = _AutoClosingFileHandler(log_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root_logger.addHandler(handler)
    _PIPELINE_FILE_HANDLER = handler
    return log_path


def _detach_work_dir_log_handler() -> None:
    global _PIPELINE_FILE_HANDLER
    if _PIPELINE_FILE_HANDLER is None:
        return
    root_logger = logging.getLogger()
    root_logger.removeHandler(_PIPELINE_FILE_HANDLER)
    with contextlib.suppress(Exception):
        _PIPELINE_FILE_HANDLER.close()
    _PIPELINE_FILE_HANDLER = None


def _select_local_venv_python(current_executable: str, script_root: Path) -> Path | None:
    candidates = [
        script_root / "venv" / "Scripts" / "python.exe",
        script_root / "venv" / "bin" / "python3",
    ]
    current_path = Path(current_executable)

    try:
        current_resolved = current_path.resolve()
    except OSError:
        current_resolved = current_path

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            candidate_resolved = candidate.resolve()
        except OSError:
            candidate_resolved = candidate
        if candidate_resolved == current_resolved:
            return None
        return candidate_resolved

    return None


def _maybe_reexec_local_venv() -> None:
    if os.environ.get("TRADUZAI_SKIP_LOCAL_VENV_REEXEC") == "1":
        return

    preferred_python = _select_local_venv_python(sys.executable, pipeline_root)
    if not preferred_python:
        return

    os.environ["TRADUZAI_SKIP_LOCAL_VENV_REEXEC"] = "1"
    script_path = str(Path(__file__).resolve())
    os.execv(str(preferred_python), [str(preferred_python), script_path, *sys.argv[1:]])


def _report_emit_failure(exc: OSError) -> None:
    global _EMIT_STDOUT_FAILED
    if _EMIT_STDOUT_FAILED:
        return

    _EMIT_STDOUT_FAILED = True
    try:
        sys.stderr.write(f"Falha ao emitir evento JSON no stdout: {exc}\n")
        sys.stderr.flush()
    except OSError:
        pass


def emit(msg_type: str, **kwargs):
    """Emit a JSON message to stdout (consumed by Rust sidecar reader)."""
    payload = {"type": msg_type, **kwargs}
    try:
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    except OSError as exc:
        _report_emit_failure(exc)


def emit_progress(
    step: str,
    step_progress: float,
    overall: float,
    page: int = 0,
    total: int = 0,
    message: str = "",
    eta: float = 0,
):
    emit(
        "progress",
        step=step,
        step_progress=step_progress,
        overall_progress=overall,
        current_page=page,
        total_pages=total,
        message=message,
        eta_seconds=eta,
    )


def log_editor_action(phase: str, action: str, **fields):
    """Log estruturado da Fase 0: imprime em stderr com prefixo [EditorAction].

    O Rust agora captura stderr (Stdio::piped) e anexa em mensagens de erro,
    então qualquer crash de import ou exceção visível aqui chega na UI.
    """
    parts = [f"{k}={v}" for k, v in fields.items()]
    line = f"[EditorAction] {phase:<7} {action} {' '.join(parts)}"
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except OSError:
        pass


def wait_if_paused(config: dict):
    """Block cooperatively while the Tauri pause marker exists."""
    pause_file = config.get("pause_file")
    if not pause_file:
        return

    pause_path = Path(pause_file)
    while pause_path.exists():
        time.sleep(0.25)


def _parse_runner_cli_args(args: list[str]) -> dict:
    parsed = {
        "source_path": "",
        "obra": "",
        "idioma_origem": "en",
        "idioma_destino": "pt-BR",
        "mode": "real",
        "debug": False,
        "skip_inpaint": False,
        "skip_ocr": False,
        "strict": False,
        "export_mode": "with_warnings",
        "work_dir": str(Path("debug") / "runs" / "pipeline_cli"),
        "mock_critical": False,
    }
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--input" and index + 1 < len(args):
            parsed["source_path"] = args[index + 1]
            index += 2
            continue
        if arg == "--work" and index + 1 < len(args):
            parsed["obra"] = args[index + 1]
            index += 2
            continue
        if arg == "--source-lang" and index + 1 < len(args):
            parsed["idioma_origem"] = args[index + 1]
            index += 2
            continue
        if arg == "--target" and index + 1 < len(args):
            parsed["idioma_destino"] = args[index + 1]
            index += 2
            continue
        if arg == "--mode" and index + 1 < len(args):
            parsed["mode"] = args[index + 1]
            index += 2
            continue
        if arg == "--export-mode" and index + 1 < len(args):
            parsed["export_mode"] = args[index + 1]
            index += 2
            continue
        if arg == "--output" and index + 1 < len(args):
            parsed["work_dir"] = args[index + 1]
            index += 2
            continue
        if arg == "--debug":
            parsed["debug"] = True
            index += 1
            continue
        if arg == "--skip-inpaint":
            parsed["skip_inpaint"] = True
            index += 1
            continue
        if arg == "--skip-ocr":
            parsed["skip_ocr"] = True
            index += 1
            continue
        if arg == "--strict":
            parsed["strict"] = True
            index += 1
            continue
        if arg == "--mock-critical":
            parsed["mock_critical"] = True
            index += 1
            continue
        raise ValueError(f"Argumento CLI desconhecido ou incompleto: {arg}")

    if not parsed["source_path"]:
        raise ValueError("--input e obrigatorio para o runner CLI")
    if not parsed["obra"]:
        parsed["obra"] = Path(parsed["source_path"]).stem or "Obra sem titulo"
    return parsed


def _list_input_images(source_path: Path) -> list[Path]:
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    if source_path.is_file() and source_path.suffix.lower() in image_exts:
        return [source_path]
    if not source_path.exists():
        raise FileNotFoundError(f"Entrada nao encontrada: {source_path}")
    return sorted(path for path in source_path.iterdir() if path.suffix.lower() in image_exts and path.is_file())


def _write_mock_runner_reports(work_dir: Path, project: dict, issues: list[dict]) -> None:
    critical = sum(1 for issue in issues if issue.get("severity") == "critical")
    qa_report = {
        "summary": {
            "total": len(issues),
            "critical": critical,
            "high": sum(1 for issue in issues if issue.get("severity") == "high"),
        },
        "issues": issues,
    }
    (work_dir / "qa_report.json").write_text(json.dumps(qa_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (work_dir / "qa_report.md").write_text(
        "\n".join(
            [
                f"# QA - {project.get('obra', 'Projeto')}",
                "",
                f"- total: {len(issues)}",
                f"- critical: {critical}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rows = ["id,page,region_id,type,severity"]
    for issue in issues:
        rows.append(
            f"{issue['id']},{issue['page']},{issue['region_id']},{issue['type']},{issue['severity']}"
        )
    (work_dir / "issues.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (work_dir / "glossary_used.json").write_text("[]\n", encoding="utf-8")
    (work_dir / "ocr_corrections.json").write_text("[]\n", encoding="utf-8")
    (work_dir / "structured_log.jsonl").write_text(
        json.dumps({"event": "mock_run", "status": "complete"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_json_file(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _apply_runtime_profile_config(config: dict):
    from runtime_profiles import apply_runtime_profile_environment, resolve_runtime_profile

    decision = resolve_runtime_profile(config)
    applied_env = apply_runtime_profile_environment(decision)
    config["runtime_profile"] = decision.profile
    config["runtime_profile_decision"] = decision.to_dict()
    config["runtime_profile_env"] = applied_env
    return decision


def _run_mock_pipeline_runner(config: dict) -> int:
    source_path = Path(config["source_path"])
    work_dir = Path(config["work_dir"])
    originals_dir = work_dir / "originals"
    images_dir = work_dir / "images"
    translated_dir = work_dir / "translated"
    for directory in [
        originals_dir,
        images_dir,
        translated_dir,
        work_dir / "layers" / "mask",
        work_dir / "layers" / "brush",
        work_dir / "layers" / "recovery",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    image_files = _list_input_images(source_path)
    pages = []
    issues = []
    for index, image_path in enumerate(image_files, start=1):
        output_name = f"{index:03}{image_path.suffix.lower()}"
        for directory in [originals_dir, images_dir, translated_dir]:
            shutil.copy2(image_path, directory / output_name)
        from PIL import Image
        try:
            with Image.open(image_path) as src_img:
                empty_size = src_img.size
        except Exception:
            empty_size = (1, 1)
        empty_layer = Image.new("RGBA", empty_size, (0, 0, 0, 0))
        empty_layer.save(work_dir / "layers" / "mask" / f"{index:03}.png")
        empty_layer.save(work_dir / "layers" / "brush" / f"{index:03}.png")
        empty_layer.save(work_dir / "layers" / "recovery" / f"{index:03}.png")

        text_layers = []
        if index == 1 and config.get("mock_critical"):
            text_layers = [
                {
                    "id": "mock-critical-1",
                    "bbox": [0, 0, 10, 10],
                    "tipo": "fala",
                    "original": "YOUNG MASTER?!",
                    "traduzido": "YOUNG MASTER?!",
                    "qa_flags": ["visual_text_leak"],
                    "qa_actions": [],
                }
            ]
            issues.append(
                {
                    "id": "0:mock-critical-1:visual_text_leak",
                    "page": index,
                    "region_id": "mock-critical-1",
                    "type": "visual_text_leak",
                    "severity": "critical",
                }
            )

        pages.append(
            {
                "numero": index,
                "arquivo_original": f"originals/{output_name}",
                "arquivo_traduzido": f"translated/{output_name}",
                "image_layers": {
                    "base": {"key": "base", "path": f"originals/{output_name}", "visible": True, "locked": True},
                    "inpaint": {"key": "inpaint", "path": f"images/{output_name}", "visible": True, "locked": True},
                    "rendered": {"key": "rendered", "path": f"translated/{output_name}", "visible": True, "locked": True},
                    "mask": {"key": "mask", "path": f"layers/mask/{index:03}.png", "visible": True, "locked": False},
                    "brush": {"key": "brush", "path": f"layers/brush/{index:03}.png", "visible": False, "locked": False},
                    "recovery": {"key": "recovery", "path": f"layers/recovery/{index:03}.png", "visible": False, "locked": False},
                },
                "inpaint_blocks": [],
                "text_layers": text_layers,
                "textos": text_layers,
            }
        )

    project = {
        "obra": config.get("obra", ""),
        "capitulo": 1,
        "idioma_origem": config.get("idioma_origem", "en"),
        "idioma_destino": config.get("idioma_destino", "pt-BR"),
        "qualidade": "normal",
        "contexto": {},
        "paginas": pages,
        "estatisticas": {"total_paginas": len(pages), "total_textos": sum(len(p["text_layers"]) for p in pages)},
        "qa": {"summary": {"total": len(issues), "critical": sum(1 for issue in issues if issue["severity"] == "critical")}},
        "mode": "mock",
    }
    _save_project_json(work_dir / "project.json", project)
    _write_mock_runner_reports(work_dir, project, issues)
    emit("complete", output_path=str(work_dir))
    if config.get("strict") and any(issue["severity"] == "critical" for issue in issues):
        emit("error", message="Strict falhou: ha flags critical ativas")
        return 2
    return 0


def _run_pipeline_runner_cli(config: dict) -> int:
    if config.get("mode") == "mock":
        return _run_mock_pipeline_runner(config)

    work_dir = Path(config["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = {
        "source_path": config["source_path"],
        "work_dir": str(work_dir),
        "models_dir": str(pipeline_root / "models"),
        "obra": config.get("obra", ""),
        "capitulo": 1,
        "idioma_origem": config.get("idioma_origem", "en"),
        "idioma_destino": config.get("idioma_destino", "pt-BR"),
        "mode": "manual" if config.get("skip_ocr") else "auto",
        "debug": config.get("debug", False),
        "skip_inpaint": config.get("skip_inpaint", False),
        "skip_ocr": config.get("skip_ocr", False),
        "export_mode": config.get("export_mode", "with_warnings"),
    }
    config_path = work_dir / "runner_config.json"
    config_path.write_text(json.dumps(runtime_config, ensure_ascii=False, indent=2), encoding="utf-8")
    _run_pipeline(str(config_path))
    return 0


def main():
    _maybe_reexec_local_venv()
    faulthandler.enable()
    _configure_pipeline_logging()
    if len(sys.argv) < 2:
        emit("error", message="Nenhum arquivo de configuracao fornecido")
        sys.exit(1)

    if sys.argv[1] in {"-h", "--help", "help"}:
        print(_build_cli_help(), flush=True)
        return

    if "--input" in sys.argv[1:]:
        exit_code = _run_pipeline_runner_cli(_parse_runner_cli_args(sys.argv[1:]))
        if exit_code:
            sys.exit(exit_code)
        return

    if sys.argv[1] == "--serve-fast-page":
        from fast_page_server import serve_stdio

        serve_stdio(pipeline_runner=_run_pipeline, reinpaint_runner=_run_reinpaint)
        return

    if sys.argv[1] == "--warmup-visual":
        from vision_stack.runtime import warmup_visual_stack
        models_dir = ""
        profile = "normal"
        # ... rest of warmup logic ...
        args = sys.argv[2:]
        index = 0
        while index < len(args):
            arg = args[index]
            if arg == "--models-dir" and index + 1 < len(args):
                models_dir = args[index + 1]
                index += 2
                continue
            if arg == "--profile" and index + 1 < len(args):
                profile = args[index + 1]
                index += 2
                continue
            index += 1

        warmup_visual_stack(models_dir=models_dir, profile=profile)
        emit("complete", output_path="")
        return

    if sys.argv[1] == "--list-supported-languages":
        from translator.translate import list_supported_google_languages
        print(json.dumps(list_supported_google_languages(), ensure_ascii=False), flush=True)
        return

    if sys.argv[1] == "--retypeset" and len(sys.argv) >= 4:
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        _run_retypeset(project_json_path, page_idx)
        return

    if sys.argv[1] == "--render-preview-page" and len(sys.argv) >= 6:
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        override_page_path = Path(sys.argv[4])
        output_path = Path(sys.argv[5])
        _run_render_preview_page(project_json_path, page_idx, override_page_path, output_path)
        return

    if (sys.argv[1] == "--process-block") and len(sys.argv) >= 6:
        mode = sys.argv[2]  # "ocr" or "translate"
        project_json_path = Path(sys.argv[3])
        page_idx = int(sys.argv[4])
        block_id = sys.argv[5]
        _run_process_block(project_json_path, page_idx, block_id, mode)
        return

    # Despachador unificado dos 4 handlers per-página (Fase 0 — sem falhas silenciosas).
    # Cada exceção é logada com [EditorAction] error e re-emitida como JSON "error"
    # para que o Rust e a UI vejam a mensagem em vez de terminar 1 sem motivo claro.
    _ACTION_DISPATCH = {
        "--detect-page": ("detect", _run_detect_page),
        "--ocr-page": ("ocr", _run_ocr_page),
        "--translate-page": ("translate", _run_translate_page),
        "--reinpaint-page": ("inpaint", _run_reinpaint),
    }
    if sys.argv[1] in _ACTION_DISPATCH and len(sys.argv) >= 4:
        action_name, handler = _ACTION_DISPATCH[sys.argv[1]]
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        region = _page_action_region_from_args(sys.argv[4:])
        log_editor_action("start", action_name, page=page_idx)
        try:
            handler(project_json_path, page_idx, region)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            log_editor_action("error", action_name, page=page_idx, message=str(exc))
            try:
                sys.stderr.write(tb + "\n")
                sys.stderr.flush()
            except OSError:
                pass
            emit("error", message=f"{action_name} falhou: {exc}")
            sys.exit(1)
        log_editor_action("success", action_name, page=page_idx)
        return

    if sys.argv[1] == "--hardware-info":
        from utils.hardware import get_hardware_facts
        print(json.dumps(get_hardware_facts(), ensure_ascii=False), flush=True)
        return

    config_path = sys.argv[1]
    _log_env_info()
    try:
        _run_pipeline(config_path)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        try:
            finalize_decision_trace({"crashed": True, "error": str(exc)})
        except Exception:
            pass
        # Garantir que o erro vá para o stdout (Tauri) e stderr (Log)
        err_msg = f"FALHA CATASTROFICA NO PIPELINE: {exc}\n{tb}"
        emit("error", message=err_msg)
        sys.stderr.write(f"\n--- CRASH DUMP ---\n{err_msg}\n------------------\n")
        sys.stderr.flush()
        sys.exit(1)

def _log_env_info():
    try:
        import platform
        import cv2
        import numpy as np
        import matplotlib
        import PIL
        info = (
            f"OS: {platform.platform()} | Python: {sys.version}\n"
            f"CWD: {os.getcwd()}\n"
            f"CV2: {cv2.__version__} | NumPy: {np.__version__} | "
            f"Matplotlib: {matplotlib.__version__} | PIL: {PIL.__version__}\n"
            f"Backend Matplotlib: {matplotlib.get_backend()}"
        )
        sys.stderr.write(f"\n--- AMBIENTE DE EXECUCAO ---\n{info}\n----------------------------\n")
        sys.stderr.flush()
    except Exception:
        pass

def _resolve_strip_target_pages(config: dict, total_pages: int | None = None) -> int:
    """Determina o número alvo de páginas de output para o pipeline strip.

    Lógica:
    - Padrão: 60 (a maioria dos capítulos tem ~50-80 páginas)
    - Override via config["strip_target_pages"]
    - Clampado para 1..total_pages (se total_pages fornecido)
    """
    raw = config.get("strip_target_pages", 60)
    try:
        target = int(raw)
    except (TypeError, ValueError):
        target = 60
    if target < 1:
        target = 60
    if total_pages is not None and target > total_pages:
        target = total_pages
    return target


def _build_connected_reasoner_config(config: dict, *, ollama_status: dict | None = None) -> dict:
    provider = str(config.get("connected_balloon_reasoner", "ollama") or "ollama").strip().lower()
    enabled = bool(config.get("connected_balloon_reasoner_enabled", True))
    host = config.get(
        "connected_balloon_ollama_host",
        config.get("ollama_host", "http://localhost:11434"),
    )
    result = {
        "provider": provider,
        "enabled": enabled,
        "host": host,
        "model": config.get("connected_balloon_ollama_model", "qwen2.5"),
        "use_image": config.get("connected_balloon_ollama_use_image", True),
        "timeout_sec": int(config.get("connected_balloon_ollama_timeout_sec", 12) or 12),
    }
    if provider != "ollama" or not enabled:
        result["provider"] = "disabled"
        result["enabled"] = False
        return result

    status = ollama_status if isinstance(ollama_status, dict) else None
    if not status:
        try:
            from translator.translate import _check_ollama as _check_connected_ollama

            status = _check_connected_ollama(str(host))
        except Exception:
            status = {"running": False, "models": [], "has_translator": False}
    if not status.get("running") or not status.get("models"):
        result["provider"] = "disabled"
        result["enabled"] = False
        return result

    result["_ollama_status"] = status
    return result


def _run_pipeline(config_path: str):
    from corpus.runtime import extract_expected_terms, load_corpus_bundle, merge_corpus_into_context
    from extractor.extractor import cleanup, extract
    from inpainter.lama import run_inpainting
    from layout.balloon_layout import enrich_page_layout
    from ocr.contextual_reviewer import contextual_review_page
    from ocr.detector import run_ocr
    from translator.context import fetch_context, merge_context
    from translator.translate import translate_pages
    from typesetter.renderer import run_typesetting

    pipeline_timing = _PipelineTiming()
    start_time = time.time()
    with pipeline_timing.measure("load_config"):
        config = _load_json_file(config_path)
    with pipeline_timing.measure("apply_runtime_profile"):
        runtime_profile_decision = _apply_runtime_profile_config(config)

    wait_if_paused(config)
    work_dir = Path(config["work_dir"])
    models_dir = Path(config["models_dir"])
    _attach_work_dir_log_handler(work_dir)
    if config.get("debug"):
        os.environ.setdefault("STRIP_DEBUG", "1")
        os.environ["TRADUZAI_INPAINT_DEBUG_DIR"] = str(work_dir / "debug_inpaint")
    configure_decision_trace(
        work_dir,
        {
            "obra": config.get("obra", ""),
            "capitulo": config.get("capitulo", 1),
            "idioma_origem": config.get("idioma_origem", "en"),
            "idioma_destino": config.get("idioma_destino", "pt-BR"),
        },
    )

    raw_source = config["source_path"].strip()
    if raw_source.startswith("file:///"):
        raw_source = raw_source[8:]
    elif raw_source.startswith("file://"):
        raw_source = raw_source[7:]
    source_path = Path(raw_source)

    images_dir = work_dir / "images"
    originals_dir = work_dir / "originals"
    translated_dir = work_dir / "translated"
    images_dir.mkdir(parents=True, exist_ok=True)
    originals_dir.mkdir(parents=True, exist_ok=True)
    translated_dir.mkdir(parents=True, exist_ok=True)

    with pipeline_timing.measure("load_corpus"):
        corpus_bundle = load_corpus_bundle(
            config.get("obra", ""),
            models_root=models_dir / "corpus",
            fallback_root=Path(__file__).resolve().parent / "models" / "corpus",
        )
        corpus_expected_terms = extract_expected_terms(corpus_bundle)

    emit_progress("extract", 0, 0, message="Extraindo arquivos...")
    try:
        with pipeline_timing.measure("extract_source"):
            image_files, tmp_dir = extract(source_path, work_dir)
    except Exception as e:
        emit("error", message=str(e))
        sys.exit(1)

    with pipeline_timing.measure("copy_original_inputs"):
        for img_path in image_files:
            shutil.copy2(img_path, originals_dir / img_path.name)

    total_pages = len(image_files)
    mode = config.get("mode", "auto")
    ocr_results = []
    page_text_layers = []
    with pipeline_timing.measure("merge_corpus_context"):
        context = merge_corpus_into_context(config.get("contexto", {}), corpus_bundle)
    strip_chapter_telemetry = None

    if mode == "manual":
        emit_progress("extract", 100, 95, message="Modo Manual: Preparando projeto...")
        with pipeline_timing.measure("manual_prepare_project"):
            for img_path in image_files:
                # Em modo manual, o 'clean' (images) e o final (translated) sao apenas copias
                shutil.copy2(originals_dir / img_path.name, images_dir / img_path.name)
                shutil.copy2(originals_dir / img_path.name, translated_dir / img_path.name)

                # Garantir diretorios de camadas para o editor nao reclamar
                (work_dir / "layers" / "mask").mkdir(parents=True, exist_ok=True)
                (work_dir / "layers" / "brush").mkdir(parents=True, exist_ok=True)
                (work_dir / "layers" / "text-preview").mkdir(parents=True, exist_ok=True)

        inpainted_paths = [str(images_dir / f.name) for f in image_files]
        # Prepare empty lists for project building
        for _ in range(total_pages):
            ocr_results.append({"texts": [], "_vision_blocks": []})
            page_text_layers.append({"texts": []})
    else:
        # STRIP-BASED PIPELINE (Task 6.2)
        from types import SimpleNamespace
        import cv2
        from strip.run import run_chapter
        from vision_stack.runtime import warmup_visual_stack as _warmup_visual_stack
        from vision_stack.runtime import _get_detector, run_ocr_stage
        from translator import translate as translator_mod
        from inpainter import inpaint_band_image
        from typesetter import renderer as typesetter_mod
        from translator.context import fetch_context, merge_context

        # Start AniList context fetch in parallel
        _context_future = None
        if not context.get("sinopse") and config.get("obra"):
            with pipeline_timing.measure("context_fetch_submit"):
                from concurrent.futures import ThreadPoolExecutor as _CtxTPE
                _ctx_pool = _CtxTPE(max_workers=1)
                _context_future = _ctx_pool.submit(
                    fetch_context,
                    config["obra"],
                    work_dir / "context_cache",
                    config.get("glossario", {}),
                    config.get("context_sources") or [],
                )

        # Warmup models
        emit_progress("ocr", 0, 9, total=total_pages, message="Carregando modelos...")
        if runtime_profile_decision.visual_stack_warmup:
            with pipeline_timing.measure("visual_stack_warmup"):
                _warmup_visual_stack(
                    str(models_dir),
                    "max",
                    run_sample=False,
                    lang=config.get("idioma_origem", "en"),
                )
        else:
            logger.info(
                "Warmup visual opcional desativado pelo perfil runtime=%s",
                runtime_profile_decision.profile,
            )

        # Resolve AniList context
        if _context_future:
            with pipeline_timing.measure("context_fetch_wait_merge"):
                try:
                    context = merge_context(context, _context_future.result(timeout=10))
                except Exception:
                    pass
                _ctx_pool.shutdown(wait=False)

        # Bridges
        connected_reasoner_config = _build_connected_reasoner_config(config)

        class StripDetector:
            def detect(self, img, conf_threshold=None):
                from vision_stack.runtime import _profile_to_detection_threshold
                thresh = conf_threshold if conf_threshold is not None else _profile_to_detection_threshold("max")
                return _get_detector("max").detect(img, conf_threshold=thresh)
                
        class StripRuntime:
            def run_ocr_stage(self, img, page_dict):
                return run_ocr_stage(img, page_dict, profile="max", idioma_origem=config.get("idioma_origem", "en"))

            def run_koharu_cjk_page(self, img, image_path):
                from vision_stack.runtime import _run_koharu_cjk_http_detect_ocr

                return _run_koharu_cjk_http_detect_ocr(
                    image_rgb=img,
                    image_label=str(image_path),
                    models_dir=str(models_dir),
                    profile="max",
                    idioma_origem=config.get("idioma_origem", "en"),
                )

            def run_koharu_cjk_pages(self, jobs, models_dir="", idioma_origem="en", _models_dir=str(models_dir)):
                resolved_models_dir = str(models_dir or _models_dir)
                resolved_idioma = idioma_origem or config.get("idioma_origem", "en")
                worker_path = str(config.get("vision_worker_path") or config.get("_vision_worker_path") or "").strip()
                worker_batch_enabled = os.getenv("TRADUZAI_KOHARU_WORKER_BATCH", "1").strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                    "disabled",
                }
                if worker_path and worker_batch_enabled:
                    try:
                        from vision_stack.runtime import _run_koharu_worker_detect_ocr_batch

                        return _run_koharu_worker_detect_ocr_batch(
                            jobs,
                            vision_worker_path=worker_path,
                            models_dir=resolved_models_dir,
                            profile="max",
                            idioma_origem=resolved_idioma,
                        )
                    except Exception as exc:
                        logger.warning("Koharu worker batch falhou; fallback para Koharu HTTP batch: %s", exc)

                from vision_stack.runtime import _run_koharu_cjk_http_detect_ocr_batch

                return _run_koharu_cjk_http_detect_ocr_batch(
                    jobs,
                    models_dir=resolved_models_dir,
                    profile="max",
                    idioma_origem=resolved_idioma,
                )
        
        def progress_cb(stage, current, total, message=""):
            # Mapeamento de estágios para o progresso global (Tauri UI)
            p = current / max(1, total)
            if stage == "concat":
                emit_progress("ocr", p * 100, 10 + p * 5, message=f"Concatenando páginas ({current}/{total})...")
            elif stage == "detect":
                emit_progress("ocr", p * 100, 15 + p * 10, message="Detectando balões no strip...")
            elif stage == "process":
                # O processamento de bandas é a parte mais longa. 25% -> 95%
                emit_progress("ocr", p * 100, 25 + p * 70, message=f"Processando banda {current+1}/{total}...")
            elif stage == "ocr":
                # Chamado internamente por process_band (OCR da banda)
                pass 

        strip_chapter_telemetry: dict = {}
        with pipeline_timing.measure("strip_run_chapter"):
            output_pages = run_chapter(
                image_files=image_files,
                output_dir=translated_dir,
                detector=StripDetector(),
                runtime=StripRuntime(),
                translator=translator_mod,
                inpainter=SimpleNamespace(inpaint_band_image=inpaint_band_image),
                typesetter=typesetter_mod,
                target_count=_resolve_strip_target_pages(config, total_pages),
                progress_callback=progress_cb,
                context=context,
                glossario=config.get("glossario", {}),
                idioma_origem=config.get("idioma_origem", "en"),
                idioma_destino=config.get("idioma_destino", "pt-BR"),
                obra=config.get("obra", ""),
                connected_reasoner_config=connected_reasoner_config,
                models_dir=str(models_dir),
                ollama_host=config.get("ollama_host", "http://localhost:11434"),
                ollama_model=config.get("ollama_model", "traduzai-translator"),
                translation_context=config.get("translation_context") or None,
                chapter_telemetry=strip_chapter_telemetry,
            )
        strip_chapter_telemetry["internal_unattributed_sec"] = round(
            max(
                0.0,
                float(strip_chapter_telemetry.get("wall_total_sec", 0.0) or 0.0)
                - sum(float(v) for v in (strip_chapter_telemetry.get("durations_sec") or {}).values()),
            ),
            4,
        )


        
        # Sincroniza saídas para o wrap-up final (project.json)
        image_files = [p.path for p in output_pages]
        ocr_results = [p.ocr_result for p in output_pages]
        page_text_layers = [p.text_layers for p in output_pages]
        total_pages = len(output_pages)

        # Copia imagens processadas para images/ para que o editor as veja como base de inpaint
        with pipeline_timing.measure("sync_inpaint_images"):
            for p in output_pages:
                inpaint_target = images_dir / p.path.name
                if getattr(p, "inpainted_image", None) is not None:
                    cv2.imwrite(str(inpaint_target), p.inpainted_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                else:
                    shutil.copy2(p.path, inpaint_target)

        # Substituir originals/ pelos nomes reassemblados esperados pelo editor.
        # O overwrite direto evita falhar em ACLs/locks temporarios de arquivos ja
        # extraidos, e a limpeza residual fica como best-effort.
        with pipeline_timing.measure("sync_original_images"):
            expected_original_names = {p.path.name for p in output_pages}
            for p in output_pages:
                original_target = originals_dir / p.path.name
                if getattr(p, "original_image", None) is not None:
                    cv2.imwrite(str(original_target), p.original_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                else:
                    shutil.copy2(p.path, original_target)
            for stale in originals_dir.glob("*"):
                if not stale.is_file() or stale.name in expected_original_names:
                    continue
                try:
                    stale.unlink()
                except PermissionError:
                    logger.warning("Nao foi possivel remover original temporario residual: %s", stale)

        # Garantir diretorios de camadas + PNGs transparentes 1x1 para mask/brush
        # (UI Tauri reclama de 404 quando layers/mask/NNN.png nao existe).
        with pipeline_timing.measure("ensure_layer_placeholders"):
            layers_root = work_dir / "layers"
            (layers_root / "mask").mkdir(parents=True, exist_ok=True)
            (layers_root / "brush").mkdir(parents=True, exist_ok=True)
            (layers_root / "recovery").mkdir(parents=True, exist_ok=True)
            (layers_root / "text-preview").mkdir(parents=True, exist_ok=True)
            try:
                from PIL import Image as _PILImage
                _empty_png = _PILImage.new("RGBA", (1, 1), (0, 0, 0, 0))
                for i in range(1, len(output_pages) + 1):
                    mask_path = layers_root / "mask" / f"{i:03}.png"
                    brush_path = layers_root / "brush" / f"{i:03}.png"
                    recovery_path = layers_root / "recovery" / f"{i:03}.png"
                    if not mask_path.exists():
                        _empty_png.save(mask_path)
                    if not brush_path.exists():
                        _empty_png.save(brush_path)
                    if not recovery_path.exists():
                        _empty_png.save(recovery_path)
            except Exception:
                pass




    # Wrap up
    emit_progress("typeset", 100, 98, message="Finalizando projeto...")
    with pipeline_timing.measure("build_project_json"):
        project_data = build_project_json(config, context, ocr_results, page_text_layers, image_files, total_pages, time.time()-start_time)
    try:
        from runtime_profiles import build_chapter_route_shadow

        with pipeline_timing.measure("chapter_route_shadow"):
            project_data["chapter_route_decision"] = build_chapter_route_shadow(ocr_results)
    except Exception as exc:
        logger.warning("Falha ao montar chapter_route_decision: %s", exc)
    with pipeline_timing.measure("build_glossary_report"):
        glossary_used_report = build_glossary_used_report(config, context, page_text_layers)
        project_data["glossary_shadow"] = glossary_used_report["summary"]
    with pipeline_timing.measure("write_glossary_report"):
        (work_dir / "glossary_used.json").write_text(
            json.dumps(glossary_used_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    from structured_logger import StructuredLogger, build_log_summary
    with pipeline_timing.measure("structured_log_summary"):
        structured_logger = StructuredLogger(config.get("logs_dir") or (work_dir / "logs"), config.get("job_id", "run"))
        log_summary = build_log_summary(project_data)
        structured_logger.log(stage="summary", event="run_summary", payload=log_summary)
    project_data["log"] = {
        "structured_log_path": str(structured_logger.path),
        "summary": log_summary,
    }
    try:
        from qa.export_gate import evaluate_export_gate

        with pipeline_timing.measure("evaluate_export_gate"):
            export_gate = evaluate_export_gate(
                project_data,
                override=bool(config.get("allow_p0_export_override")),
            )
            project_data["qa"]["export_gate"] = export_gate
            project_data["needs_review"] = export_gate["status"] == "BLOCK"
    except Exception as exc:
        logger.warning("Falha ao avaliar export gate: %s", exc)
    project_data["performance"] = pipeline_timing.snapshot(
        total_seconds=time.time() - start_time,
        extra={
            "schema_version": 1,
            "mode": mode,
            "sidecar_path": "performance_timing.json",
            "strip": strip_chapter_telemetry or {},
        },
    )
    with pipeline_timing.measure("save_project_json"):
        _save_project_json(work_dir / "project.json", project_data)
    with pipeline_timing.measure("finalize_decision_trace"):
        finalize_decision_trace(
            {
                "total_paginas": total_pages,
                "total_textos": project_data.get("estatisticas", {}).get("total_textos", 0),
            }
        )

    with pipeline_timing.measure("cleanup_temp"):
        cleanup(tmp_dir)
    final_performance = pipeline_timing.snapshot(
        total_seconds=time.time() - start_time,
        extra={
            "schema_version": 1,
            "mode": mode,
            "sidecar_path": "performance_timing.json",
            "strip": strip_chapter_telemetry or {},
        },
    )
    try:
        (work_dir / "performance_timing.json").write_text(
            json.dumps(final_performance, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        project_data["performance"] = final_performance
    except Exception as exc:
        logger.warning("Falha ao escrever performance_timing.json: %s", exc)
    emit_progress("typeset", 100, 100, message="Concluido!")
    emit("complete", output_path=str(work_dir))


def _default_text_style() -> dict:
    return {
        "fonte": "ComicNeue-Bold.ttf",
        "tamanho": 28,
        "cor": "#000000",
        "cor_gradiente": [],
        "contorno": "",
        "contorno_px": 0,
        "glow": False,
        "glow_cor": "",
        "glow_px": 0,
        "sombra": False,
        "sombra_cor": "",
        "sombra_offset": [0, 0],
        "bold": True,
        "italico": False,
        "rotacao": 0,
        "alinhamento": "center",
        "force_upper": False,
    }


def _merge_style(style: dict | None) -> dict:
    merged = _default_text_style()
    if isinstance(style, dict):
        merged.update(style)
    return merged


def _coerce_background_rgb(value) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(max(0, min(255, int(round(float(channel))))) for channel in value[:3])  # type: ignore[return-value]
        except (TypeError, ValueError):
            pass
    return (255, 255, 255)


def _bbox4(value, fallback=None) -> list[int]:
    source = value if isinstance(value, (list, tuple)) and len(value) >= 4 else fallback
    if not isinstance(source, (list, tuple)) or len(source) < 4:
        source = [0, 0, 32, 32]
    return [int(source[0]), int(source[1]), int(source[2]), int(source[3])]


def _optional_bbox4(value) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    return [int(value[0]), int(value[1]), int(value[2]), int(value[3])]


def _preview_rel_path(page_number: int, layer_id: str) -> str:
    return f"layers/text-preview/{page_number:03}/{layer_id}.png"


def _bbox4_list(values) -> list[list[int]]:
    return [
        _bbox4(value)
        for value in (values or [])
        if isinstance(value, (list, tuple)) and len(value) >= 4
    ]


def _bbox_iou(left, right) -> float:
    a = _optional_bbox4(left)
    b = _optional_bbox4(right)
    if a is None or b is None:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / float(max(1, union))


def _bbox_intersects(left, right) -> bool:
    a = _optional_bbox4(left)
    b = _optional_bbox4(right)
    if a is None or b is None:
        return False
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _parse_region_bbox(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    if len(parts) != 4:
        return None
    try:
        bbox = [int(float(part)) for part in parts]
    except ValueError:
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0, x1), max(0, x2)))
    y1, y2 = sorted((max(0, y1), max(0, y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _mask_bounding_box(mask_path: str | Path | None) -> list[int] | None:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.exists():
        return None
    try:
        from PIL import Image
        with Image.open(path) as img:
            bbox = img.convert("L").getbbox()
            if not bbox:
                return None
            return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    except Exception:
        return None


def _page_action_region_from_args(args: list[str]) -> dict:
    region: dict = {"bbox": None, "mask_path": None}
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--region-bbox" and idx + 1 < len(args):
            region["bbox"] = _parse_region_bbox(args[idx + 1])
            idx += 2
            continue
        if arg == "--external-mask" and idx + 1 < len(args):
            region["mask_path"] = args[idx + 1]
            idx += 2
            continue
        idx += 1
    if region["bbox"] is None:
        region["bbox"] = _mask_bounding_box(region.get("mask_path"))
    return region


def _region_bbox(region: dict | None) -> list[int] | None:
    if not isinstance(region, dict):
        return None
    return _optional_bbox4(region.get("bbox"))


def _external_mask_vision_block(region: dict | None) -> dict | None:
    bbox = _region_bbox(region)
    if bbox is None or not isinstance(region, dict):
        return None
    mask_path = region.get("mask_path")
    if not mask_path:
        return None
    path = Path(str(mask_path))
    if not path.exists():
        return None
    try:
        from PIL import Image
        import numpy as np

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [int(v) for v in bbox]
        with Image.open(path) as img:
            mask = np.array(img.convert("L"), dtype=np.uint8)
        height, width = mask.shape[:2]
        bbox_width = max(0, bbox_x2 - bbox_x1)
        bbox_height = max(0, bbox_y2 - bbox_y1)
        if width == bbox_width and height == bbox_height:
            if int(np.count_nonzero(mask)) <= 0:
                return None
            return {"bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2], "confidence": 1.0, "mask": mask}

        x1, y1, x2, y2 = bbox_x1, bbox_y1, bbox_x2, bbox_y2
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        local_mask = mask[y1:y2, x1:x2]
        if int(np.count_nonzero(local_mask)) <= 0:
            return None
        return {"bbox": [x1, y1, x2, y2], "confidence": 1.0, "mask": local_mask}
    except Exception:
        return None


def _bbox_in_region(bbox, region: dict | None) -> bool:
    region_bbox = _region_bbox(region)
    if region_bbox is None:
        return True
    return _bbox_intersects(bbox, region_bbox)


def _layer_in_region(layer: dict, region: dict | None) -> bool:
    return _bbox_in_region(_resolved_layer_bbox(layer), region)


def _block_in_region(block: dict, region: dict | None) -> bool:
    return _bbox_in_region(block.get("bbox"), region)


def _bbox_center_distance(left, right) -> float:
    a = _optional_bbox4(left)
    b = _optional_bbox4(right)
    if a is None or b is None:
        return float("inf")
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def _layer_match_bbox(layer: dict) -> list[int]:
    return _bbox4(
        layer.get("text_pixel_bbox"),
        layer.get("source_bbox")
        or layer.get("layout_bbox")
        or layer.get("balloon_bbox")
        or layer.get("bbox"),
    )


def _normalize_match_text(text: str) -> str:
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


def _split_translated_sentences(text: str) -> list[str]:
    stripped = str(text or "").strip()
    if not stripped:
        return []
    parts = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?…])\s+", stripped)
        if segment.strip()
    ]
    if len(parts) <= 1 and " - " in stripped:
        parts = [segment.strip() for segment in stripped.split(" - ") if segment.strip()]
    return parts or [stripped]


def _carry_translations_for_detected_layers(existing_layers: list[dict], reviewed_texts: list[dict]) -> list[str]:
    if not existing_layers or not reviewed_texts:
        return ["" for _ in reviewed_texts]

    assignments = [""] * len(reviewed_texts)
    used_existing: set[int] = set()

    for old_idx, layer in enumerate(existing_layers):
        if old_idx in used_existing:
            continue
        old_text_norm = _normalize_match_text(
            layer.get("original", layer.get("text", ""))
        )
        old_translation = str(layer.get("translated", "") or "").strip()
        if not old_text_norm or not old_translation:
            continue

        candidate_new_indices: list[int] = []
        old_bbox = _layer_match_bbox(layer)
        for new_idx, text in enumerate(reviewed_texts):
            if assignments[new_idx]:
                continue
            new_text_norm = _normalize_match_text(text.get("text", ""))
            if not new_text_norm or new_text_norm == old_text_norm:
                continue
            if new_text_norm not in old_text_norm:
                continue
            if _bbox_center_distance(_layer_match_bbox(text), old_bbox) > max(
                220.0,
                (old_bbox[3] - old_bbox[1]) * 1.5,
            ):
                continue
            candidate_new_indices.append(new_idx)

        if len(candidate_new_indices) < 2:
            continue

        split_translation = _split_translated_sentences(old_translation)
        if len(split_translation) != len(candidate_new_indices):
            continue

        ordered_new_indices = sorted(
            candidate_new_indices,
            key=lambda idx: (
                _layer_match_bbox(reviewed_texts[idx])[1],
                _layer_match_bbox(reviewed_texts[idx])[0],
            ),
        )
        for part, new_idx in zip(split_translation, ordered_new_indices):
            assignments[new_idx] = part
        used_existing.add(old_idx)

    for new_idx, text in enumerate(reviewed_texts):
        if assignments[new_idx]:
            continue
        new_bbox = _layer_match_bbox(text)
        new_text_norm = _normalize_match_text(text.get("text", ""))

        merge_candidates: list[tuple[int, list[int]]] = []
        if new_text_norm:
            for old_idx, layer in enumerate(existing_layers):
                if old_idx in used_existing:
                    continue
                old_text_norm = _normalize_match_text(
                    layer.get("original", layer.get("text", ""))
                )
                if not old_text_norm or old_text_norm == new_text_norm:
                    continue
                old_bbox = _layer_match_bbox(layer)
                if old_text_norm in new_text_norm and _bbox_center_distance(new_bbox, old_bbox) <= max(
                    180.0,
                    (new_bbox[3] - new_bbox[1]) * 1.2,
                ):
                    merge_candidates.append((old_idx, old_bbox))

        if len(merge_candidates) >= 2:
            ordered_merge = sorted(
                merge_candidates,
                key=lambda item: (item[1][1], item[1][0]),
            )
            merged_translation = " ".join(
                str(existing_layers[old_idx].get("translated", "") or "").strip()
                for old_idx, _ in ordered_merge
                if str(existing_layers[old_idx].get("translated", "") or "").strip()
            ).strip()
            if merged_translation:
                assignments[new_idx] = merged_translation
                used_existing.update(old_idx for old_idx, _ in ordered_merge)
                continue

        best_idx = -1
        best_score = float("-inf")

        for old_idx, layer in enumerate(existing_layers):
            if old_idx in used_existing:
                continue

            old_bbox = _layer_match_bbox(layer)
            iou = _bbox_iou(new_bbox, old_bbox)
            center_dist = _bbox_center_distance(new_bbox, old_bbox)
            old_text_norm = _normalize_match_text(
                layer.get("original", layer.get("text", ""))
            )
            text_bonus = 2.5 if new_text_norm and new_text_norm == old_text_norm else 0.0
            score = (iou * 12.0) + text_bonus - (center_dist / 180.0)

            if iou < 0.05 and center_dist > max(96.0, (new_bbox[3] - new_bbox[1]) * 1.5):
                continue
            if score > best_score:
                best_score = score
                best_idx = old_idx

        if best_idx < 0:
            continue

        translated = str(existing_layers[best_idx].get("translated", "") or "")
        assignments[new_idx] = translated
        used_existing.add(best_idx)

    return assignments


def _normalize_relative_y_bbox(value, reference_bbox) -> list[int] | None:
    bbox = _optional_bbox4(value)
    reference = _optional_bbox4(reference_bbox)
    if bbox is None:
        return None
    if reference is None:
        return bbox

    bx1, by1, bx2, by2 = bbox
    _, ry1, _, ry2 = reference
    ref_h = max(1, ry2 - ry1)
    box_h = max(1, by2 - by1)

    overlaps_y = not (by2 <= ry1 or by1 >= ry2)
    if overlaps_y:
        return bbox

    if by1 >= ry1:
        return bbox

    if box_h > int(ref_h * 1.5):
        return bbox

    shifted = [bx1, by1 + ry1, bx2, by2 + ry1]
    shifted_by1 = shifted[1]
    shifted_by2 = shifted[3]
    if shifted_by2 <= ry1 or shifted_by1 >= ry2 + max(64, ref_h):
        return bbox
    return shifted


def _normalize_relative_y_bbox_list(values, reference_bbox) -> list[list[int]]:
    normalized: list[list[int]] = []
    for value in values or []:
        bbox = _normalize_relative_y_bbox(value, reference_bbox)
        if bbox is not None:
            normalized.append(bbox)
    return normalized


def _normalize_relative_y_polygons(polygons, reference_bbox) -> list:
    if not isinstance(polygons, list) or not polygons:
        return polygons

    ys: list[int] = []
    for polygon in polygons:
        if not isinstance(polygon, list):
            continue
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                ys.append(int(point[1]))
    if not ys:
        return polygons

    poly_bbox = [0, min(ys), 0, max(ys)]
    normalized_bbox = _normalize_relative_y_bbox(poly_bbox, reference_bbox)
    if normalized_bbox is None:
        return polygons
    delta_y = int(normalized_bbox[1]) - int(poly_bbox[1])
    if delta_y == 0:
        return polygons

    shifted = []
    for polygon in polygons:
        if not isinstance(polygon, list):
            shifted.append(polygon)
            continue
        shifted_polygon = []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                shifted_polygon.append([int(point[0]), int(point[1]) + delta_y])
            else:
                shifted_polygon.append(point)
        shifted.append(shifted_polygon)
    return shifted


def _resolved_layer_bbox(layer: dict, fallback=None) -> list[int]:
    return _bbox4(
        layer.get("bbox"),
        layer.get("layout_bbox") or layer.get("source_bbox") or layer.get("balloon_bbox") or fallback,
    )


def build_text_layer(
    *,
    page_number: int,
    layer_index: int,
    ocr_text: dict,
    translated: str,
    corpus_visual_benchmark: dict,
    corpus_textual_benchmark: dict,
) -> dict:
    from ocr.ocr_normalizer import normalize_ocr_record

    ocr_text = normalize_ocr_record(ocr_text)
    layer_id = ocr_text.get("id") or f"tl_{page_number:03}_{layer_index + 1:03}"
    source_bbox = _bbox4(ocr_text.get("source_bbox"), ocr_text.get("bbox"))
    text_pixel_bbox = _bbox4(
        _normalize_relative_y_bbox(ocr_text.get("text_pixel_bbox"), source_bbox),
        source_bbox,
    )
    layout_bbox = resolve_text_anchor_bbox(
        {
            "text_pixel_bbox": text_pixel_bbox,
            "source_bbox": source_bbox,
            "bbox": ocr_text.get("bbox"),
        }
    ) or _bbox4(
        ocr_text.get("layout_bbox"),
        ocr_text.get("balloon_bbox") or source_bbox,
    )
    force_black_text = (
        str(ocr_text.get("balloon_type") or "").strip().lower() == "white"
        or str(ocr_text.get("layout_profile") or ocr_text.get("block_profile") or "").strip().lower() == "white_balloon"
    )
    style = normalize_auto_typesetting_style(
        _merge_style(ocr_text.get("estilo")),
        _coerce_background_rgb(ocr_text.get("background_rgb")),
        force_black_text=force_black_text,
    )
    balloon_subregions = []
    connected_lobe_bboxes = []
    connected_text_groups = []
    connected_position_bboxes = []
    connected_focus_bboxes = []
    line_polygons = _normalize_relative_y_polygons(ocr_text.get("line_polygons"), source_bbox)

    return {
        "id": layer_id,
        "kind": "text",
        "source_bbox": source_bbox,
        "layout_bbox": layout_bbox,
        "render_bbox": None,
        "bbox": layout_bbox,
        "text_pixel_bbox": text_pixel_bbox,
        "tipo": ocr_text.get("tipo", "fala"),
        "original": ocr_text.get("text", ""),
        "raw_ocr": ocr_text.get("raw_ocr", ocr_text.get("text", "")),
        "normalized_ocr": ocr_text.get("normalized_ocr", ocr_text.get("text", "")),
        "normalization": ocr_text.get("normalization", {"changed": False, "corrections": [], "is_gibberish": False}),
        "translated": translated,
        "text": ocr_text.get("text", ""),
        "ocr_confidence": float(ocr_text.get("confidence", 0.0) or 0.0),
        "ocr_source": ocr_text.get("ocr_source"),
        "style": style,
        "estilo": style,
        "style_origin": "auto",
        "visible": True,
        "locked": False,
        "order": layer_index,
        "render_preview_path": _preview_rel_path(page_number, layer_id),
        "detector": ocr_text.get("detector"),
        "line_polygons": line_polygons,
        "source_direction": ocr_text.get("source_direction"),
        "rendered_direction": ocr_text.get("rendered_direction"),
        "source_language": ocr_text.get("source_language"),
        "rotation_deg": float(ocr_text.get("rotation_deg", 0) or 0),
        "detected_font_size_px": ocr_text.get("detected_font_size_px"),
        "skip_processing": bool(ocr_text.get("skip_processing", False)),
        "skip_reason": ocr_text.get("skip_reason"),
        "smart_skip_decision": ocr_text.get("smart_skip_decision"),
        "balloon_type": ocr_text.get("balloon_type"),
        "inpaint_mode": ocr_text.get("inpaint_mode"),
        "inpaint_strategy": ocr_text.get("inpaint_strategy"),
        "balloon_bbox": layout_bbox,
        "balloon_subregions": balloon_subregions,
        "layout_group_size": 1,
        "connected_children": None,
        "connected_text_groups": connected_text_groups,
        "connected_lobe_bboxes": connected_lobe_bboxes,
        "connected_position_bboxes": connected_position_bboxes,
        "connected_focus_bboxes": connected_focus_bboxes,
        "connected_balloon_orientation": "",
        "connected_detection_confidence": 0.0,
        "connected_group_confidence": 0.0,
        "connected_position_confidence": 0.0,
        "subregion_confidence": 0.0,
        "connected_position_reasoner": "",
        "connected_reasoner_model": "",
        "connected_reasoner_notes": "",
        "_connected_slot_index": None,
        "_connected_slot_count": None,
        "_connected_vertical_bias_ratio": None,
        "_is_lobe_subregion": False,
        "page_profile": ocr_text.get("page_profile"),
        "block_profile": ocr_text.get("block_profile"),
        "layout_profile": ocr_text.get("block_profile") if ocr_text.get("layout_profile") == "connected_balloon" else (ocr_text.get("layout_profile") or ocr_text.get("block_profile")),
        "entity_flags": list(ocr_text.get("entity_flags") or []),
        "entity_repairs": list(ocr_text.get("entity_repairs") or []),
        "glossary_hits": list(ocr_text.get("glossary_hits") or []),
        "qa_flags": list(ocr_text.get("qa_flags") or []),
        "corpus_visual_benchmark": corpus_visual_benchmark,
        "corpus_textual_benchmark": corpus_textual_benchmark,
    }


def _normalize_text_layer_for_renderer(raw_layer: dict, page_number: int, layer_index: int) -> dict:
    source_bbox = _bbox4(raw_layer.get("source_bbox"), raw_layer.get("bbox"))
    text_pixel_bbox = _bbox4(
        _normalize_relative_y_bbox(raw_layer.get("text_pixel_bbox"), source_bbox),
        source_bbox,
    )
    layout_bbox = _bbox4(raw_layer.get("layout_bbox"), text_pixel_bbox or source_bbox)
    style = _merge_style(raw_layer.get("style") or raw_layer.get("estilo"))
    layer_id = raw_layer.get("id") or f"tl_{page_number:03}_{layer_index + 1:03}"
    translated = raw_layer.get("translated", raw_layer.get("traduzido", ""))
    original = raw_layer.get("original", raw_layer.get("text", ""))
    balloon_subregions = []
    connected_lobe_bboxes = []
    connected_text_groups = []
    connected_position_bboxes = []
    connected_focus_bboxes = []
    line_polygons = _normalize_relative_y_polygons(raw_layer.get("line_polygons"), source_bbox)

    return {
        "id": layer_id,
        "kind": "text",
        "source_bbox": source_bbox,
        "layout_bbox": layout_bbox,
        "render_bbox": raw_layer.get("render_bbox"),
        "bbox": layout_bbox,
        "text_pixel_bbox": text_pixel_bbox,
        "tipo": raw_layer.get("tipo", "fala"),
        "original": original,
        "translated": translated,
        "text": original,
        "ocr_confidence": float(raw_layer.get("ocr_confidence", raw_layer.get("confianca_ocr", 0.0)) or 0.0),
        "ocr_source": raw_layer.get("ocr_source"),
        "style": style,
        "estilo": style,
        "style_origin": raw_layer.get("style_origin", "legacy"),
        "visible": bool(raw_layer.get("visible", True)),
        "locked": bool(raw_layer.get("locked", False)),
        "order": int(raw_layer.get("order", layer_index) or layer_index),
        "render_preview_path": raw_layer.get("render_preview_path") or _preview_rel_path(page_number, layer_id),
        "detector": raw_layer.get("detector"),
        "line_polygons": line_polygons,
        "source_direction": raw_layer.get("source_direction"),
        "rendered_direction": raw_layer.get("rendered_direction"),
        "source_language": raw_layer.get("source_language"),
        "rotation_deg": float(raw_layer.get("rotation_deg", 0) or 0),
        "detected_font_size_px": raw_layer.get("detected_font_size_px"),
        "skip_processing": bool(raw_layer.get("skip_processing", False)),
        "skip_reason": raw_layer.get("skip_reason"),
        "smart_skip_decision": raw_layer.get("smart_skip_decision"),
        "balloon_type": raw_layer.get("balloon_type"),
        "inpaint_mode": raw_layer.get("inpaint_mode"),
        "inpaint_strategy": raw_layer.get("inpaint_strategy"),
        "balloon_bbox": _bbox4(raw_layer.get("balloon_bbox"), layout_bbox),
        "balloon_subregions": balloon_subregions,
        "layout_group_size": 1,
        "connected_children": None,
        "connected_text_groups": connected_text_groups,
        "connected_lobe_bboxes": connected_lobe_bboxes,
        "connected_position_bboxes": connected_position_bboxes,
        "connected_focus_bboxes": connected_focus_bboxes,
        "connected_balloon_orientation": "",
        "connected_detection_confidence": 0.0,
        "connected_group_confidence": 0.0,
        "connected_position_confidence": 0.0,
        "subregion_confidence": 0.0,
        "connected_position_reasoner": "",
        "connected_reasoner_model": "",
        "connected_reasoner_notes": "",
        "_connected_slot_index": None,
        "_connected_slot_count": None,
        "_connected_vertical_bias_ratio": None,
        "_is_lobe_subregion": False,
        "page_profile": raw_layer.get("page_profile"),
        "block_profile": raw_layer.get("block_profile"),
        "layout_profile": raw_layer.get("block_profile") if raw_layer.get("layout_profile") == "connected_balloon" else (raw_layer.get("layout_profile") or raw_layer.get("block_profile")),
        "entity_flags": list(raw_layer.get("entity_flags") or []),
        "entity_repairs": list(raw_layer.get("entity_repairs") or []),
        "glossary_hits": list(raw_layer.get("glossary_hits") or []),
        "qa_flags": list(raw_layer.get("qa_flags") or []),
        "corpus_visual_benchmark": raw_layer.get("corpus_visual_benchmark", {}),
        "corpus_textual_benchmark": raw_layer.get("corpus_textual_benchmark", {}),
    }


def _page_text_layers_for_renderer(page: dict, page_index: int) -> list[dict]:
    page_number = int(page.get("numero", page_index + 1) or page_index + 1)
    raw_layers = page.get("text_layers")
    if isinstance(raw_layers, list) and raw_layers:
        return [
            _normalize_text_layer_for_renderer(layer, page_number, idx)
            for idx, layer in enumerate(raw_layers)
            if isinstance(layer, dict)
        ]

    return [
        _normalize_text_layer_for_renderer(layer, page_number, idx)
        for idx, layer in enumerate(page.get("textos", []))
        if isinstance(layer, dict)
    ]


def _sync_page_legacy_aliases(page: dict) -> None:
    image_layers = page.setdefault("image_layers", {})
    page["arquivo_original"] = ((image_layers.get("base") or {}).get("path")) or page.get("arquivo_original")
    page["arquivo_traduzido"] = ((image_layers.get("rendered") or {}).get("path")) or page.get("arquivo_traduzido")

    text_layers = [sanitize_simple_text_geometry(layer) for layer in (page.get("text_layers") or []) if isinstance(layer, dict)]
    page["text_layers"] = text_layers
    page["textos"] = [
        {
            "id": layer.get("id"),
            "bbox": _bbox4(
                layer.get("render_bbox"),
                _bbox4(
                    layer.get("layout_bbox"),
                    _bbox4(layer.get("source_bbox"), layer.get("bbox")),
                ),
            ),
            "tipo": layer.get("tipo", "fala"),
            "original": layer.get("original", ""),
            "translated": layer.get("translated", layer.get("traduzido", "")),
            "traduzido": layer.get("translated", ""),
            "confianca_ocr": float(layer.get("ocr_confidence", 0.0) or 0.0),
            "ocr_confidence": float(layer.get("ocr_confidence", 0.0) or 0.0),
            "ocr_source": layer.get("ocr_source"),
            "source_bbox": _bbox4(layer.get("source_bbox"), layer.get("bbox")),
            "text_pixel_bbox": _bbox4(
                layer.get("text_pixel_bbox"),
                _bbox4(layer.get("source_bbox"), layer.get("bbox")),
            ),
            "line_polygons": _normalize_relative_y_polygons(
                layer.get("line_polygons"),
                _bbox4(layer.get("source_bbox"), layer.get("bbox")),
            ),
            "estilo": _merge_style(layer.get("style") or layer.get("estilo")),
            "style_origin": layer.get("style_origin", "legacy"),
            "qa_flags": list(layer.get("qa_flags") or []),
            "balloon_bbox": _bbox4(layer.get("balloon_bbox"), layer.get("layout_bbox") or layer.get("bbox")),
            "balloon_type": layer.get("balloon_type"),
            "layout_profile": layer.get("layout_profile") or layer.get("block_profile"),
            "layout_group_size": 1,
            "skip_processing": bool(layer.get("skip_processing", False)),
            "skip_reason": layer.get("skip_reason"),
            "smart_skip_decision": layer.get("smart_skip_decision"),
            "balloon_subregions": [],
            "connected_lobe_bboxes": [],
            "connected_text_groups": [],
            "connected_position_bboxes": [],
            "connected_focus_bboxes": [],
            "connected_balloon_orientation": "",
            "connected_detection_confidence": 0.0,
            "connected_group_confidence": 0.0,
            "connected_position_confidence": 0.0,
            "subregion_confidence": 0.0,
        }
        for layer in text_layers
        if isinstance(layer, dict)
    ]


def _project_inpaint_block_from_vision_block(block: dict) -> dict | None:
    if not isinstance(block, dict):
        return None
    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        bbox = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None

    out = {"bbox": bbox, "confidence": block.get("confidence", 0.0)}
    if block.get("source_bbox") in (None, [], ""):
        out["source_bbox"] = list(bbox)
    for key in (
        "line_polygons",
        "text_pixel_bbox",
        "source_bbox",
        "balloon_bbox",
        "balloon_type",
        "block_profile",
        "background_type",
        "tipo",
        "font_size_px",
        "font_size",
    ):
        value = block.get(key)
        if value is not None and value != [] and value != "":
            out[key] = copy.deepcopy(value)
    if "text_pixel_bbox" not in out:
        out["text_pixel_bbox"] = list(bbox)
    return out


def _save_project_json(project_json_path: Path, project: dict) -> None:
    from project_writer import write_project_json_atomic

    log = project.get("log")
    if isinstance(log, dict) and isinstance(log.get("summary"), dict):
        try:
            from structured_logger import build_log_summary

            log["summary"] = build_log_summary(project)
        except Exception as exc:
            logger.warning("Falha ao atualizar log.summary antes de salvar project.json: %s", exc)

    write_project_json_atomic(project_json_path, project)


def _resolve_image_layer_path(page: dict, layer_key: str, fallback: str) -> str:
    image_layers = page.get("image_layers") or {}
    layer = image_layers.get(layer_key) or {}
    return layer.get("path") or fallback


def _ensure_image_layer(page: dict, layer_key: str, path: str, *, visible: bool, locked: bool) -> None:
    image_layers = page.setdefault("image_layers", {})
    layer = image_layers.setdefault(layer_key, {})
    layer["key"] = layer_key
    layer["path"] = path
    layer["visible"] = bool(layer.get("visible", visible))
    layer["locked"] = bool(layer.get("locked", locked))


def _apply_recovery_layer_for_page(project: dict, page: dict, rendered_path: Path) -> None:
    work_dir = Path(project.get("_work_dir", "."))
    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
    recovery_rel = _resolve_image_layer_path(page, "recovery", "")
    if not recovery_rel:
        return

    original_path = work_dir / original_rel
    if not original_path.exists():
        original_path = work_dir / "originals" / Path(original_rel).name
    if not original_path.exists():
        candidate = Path(original_rel)
        if candidate.exists():
            original_path = candidate

    recovery_path = work_dir / recovery_rel
    if not recovery_path.exists():
        candidate = Path(recovery_rel)
        if candidate.exists():
            recovery_path = candidate

    try:
        from recovery import apply_recovery_layer
        apply_recovery_layer(rendered_path, original_path, recovery_path)
    except Exception as exc:
        logger.warning("Falha ao aplicar recovery na pagina renderizada: %s", exc)


def render_page_image(project, page_idx, output_path):
    """Auxiliar para renderizar a versao final da pagina para visualizacao."""
    from typesetter.renderer import _typeset_single_page
    page = project["paginas"][page_idx]
    
    # Prepara o dicionario de textos no formato que o renderer espera
    trans_texts = _page_text_layers_for_renderer(page, page_idx)
    trans_page_dict = {"texts": _visible_render_texts(trans_texts)}
    
    # Determina a imagem de fundo: tenta 'images' (inpainted) primeiro, depois 'originals'
    work_dir = Path(project.get("_work_dir", "."))
    img_name = Path(page["arquivo_original"]).name
    bg_path = work_dir / "images" / img_name
    if not bg_path.exists():
        bg_path = work_dir / "originals" / img_name
        
    if not bg_path.exists():
        # Fallback para caminho direto se salvo no layout
        bg_path = Path(page.get("arquivo_original", ""))
        
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        _typeset_single_page((str(bg_path), trans_page_dict, str(out_dir)))
        rendered_path = Path(output_path)
        renderer_output = out_dir / Path(bg_path).name
        if renderer_output.exists() and renderer_output.resolve() != rendered_path.resolve():
            if rendered_path.exists():
                rendered_path.unlink()
            shutil.move(str(renderer_output), str(rendered_path))
        _apply_recovery_layer_for_page(project, page, rendered_path)
    except Exception as e:
        sys.stderr.write(f"Falha ao renderizar imagem da pagina: {e}\n")


def _merge_regional_inpaint_output(
    *,
    generated_path: Path,
    base_path: Path,
    fallback_path: Path,
    bbox: list[int] | None,
) -> None:
    if bbox is None or not generated_path.exists():
        return
    try:
        from PIL import Image
        target_path = base_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        base_source = target_path if target_path.exists() else fallback_path
        with Image.open(base_source) as base_src:
            base_img = base_src.convert("RGB")
        with Image.open(generated_path) as generated_src:
            generated_img = generated_src.convert("RGB")
        if generated_img.size != base_img.size:
            generated_img = generated_img.resize(base_img.size, Image.Resampling.LANCZOS)
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(base_img.width, x1))
        y1 = max(0, min(base_img.height, y1))
        x2 = max(0, min(base_img.width, x2))
        y2 = max(0, min(base_img.height, y2))
        if x2 <= x1 or y2 <= y1:
            return
        base_img.paste(generated_img.crop((x1, y1, x2, y2)), (x1, y1))
        base_img.save(target_path)
    except Exception as exc:
        logger.warning("Falha ao mesclar inpaint regional: %s", exc)


def _finalize_reinpaint_output_path(
    *,
    outputs: list,
    is_regional: bool,
    base_path: Path,
    fallback_path: Path,
    bbox: list[int] | None,
    default_path: Path,
) -> Path:
    if not outputs:
        return default_path

    generated_path = Path(outputs[0])
    if is_regional:
        _merge_regional_inpaint_output(
            generated_path=generated_path,
            base_path=base_path,
            fallback_path=fallback_path,
            bbox=bbox,
        )
        return base_path

    return generated_path


def _visible_render_texts(texts: list[dict]) -> list[dict]:
    return [text for text in texts if text.get("visible", True) is not False]


def _text_requires_final_cleanup(text: dict) -> bool:
    if not isinstance(text, dict) or text.get("skip_processing"):
        return False
    translated = str(text.get("translated") or text.get("traduzido") or "").strip()
    if not translated:
        return False
    original = str(text.get("original") or text.get("text") or "").strip()
    if original and translated == original:
        return False
    return True


def _texts_requiring_final_cleanup(texts: list[dict]) -> list[dict]:
    return [text for text in _visible_render_texts(texts) if _text_requires_final_cleanup(text)]


def _final_render_text_box_cleanup_enabled() -> bool:
    raw = os.getenv("TRADUZAI_ENABLE_FINAL_RENDER_TEXT_BOX_CLEANUP")
    if raw is None or not raw.strip():
        return False
    return bool(raw and raw.strip().lower() in {"1", "true", "yes", "on"})


def _prepare_inpaint_base_for_render(
    *,
    original_path: Path,
    inpainted_path: Path,
    texts: list[dict],
    temp_output_path: Path | None = None,
    update_inpaint: bool = False,
) -> Path:
    if not _final_render_text_box_cleanup_enabled():
        return inpainted_path

    cleanup_texts = _texts_requiring_final_cleanup(texts)
    if not cleanup_texts or not original_path.exists() or not inpainted_path.exists():
        return inpainted_path

    try:
        import cv2
        from vision_stack.runtime import (
            _apply_white_balloon_text_box_cleanup,
            _has_white_balloon_text_residual,
        )

        original_bgr = cv2.imread(str(original_path))
        inpaint_bgr = cv2.imread(str(inpainted_path))
        if original_bgr is None or inpaint_bgr is None:
            return inpainted_path
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        inpaint_rgb = cv2.cvtColor(inpaint_bgr, cv2.COLOR_BGR2RGB)
        if not _has_white_balloon_text_residual(original_rgb, inpaint_rgb, cleanup_texts):
            return inpainted_path

        cleaned_rgb = _apply_white_balloon_text_box_cleanup(original_rgb, inpaint_rgb, cleanup_texts)
        target_path = inpainted_path if update_inpaint else temp_output_path
        if target_path is None:
            return inpainted_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target_path), cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        return target_path
    except Exception as exc:
        logger.warning("Falha na limpeza final antes do render: %s", exc)
        return inpainted_path


def _run_render_preview_page(
    project_json_path: Path,
    page_idx: int,
    override_page_path: Path,
    output_path: Path,
):
    try:
        with open(project_json_path, "r", encoding="utf-8") as f:
            project = json.load(f)
        with open(override_page_path, "r", encoding="utf-8") as f:
            override_payload = json.load(f)

        work_dir = project_json_path.parent
        project["_work_dir"] = str(work_dir)
        pages = project.get("paginas", [])
        if page_idx < 0 or page_idx >= len(pages):
            emit("error", message="Indice de pagina invalido")
            return

        page = override_payload.get("page", override_payload)
        if not isinstance(page, dict):
            emit("error", message="Pagina temporaria do preview invalida")
            return
        pages[page_idx] = page

        original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
        inpaint_rel = _resolve_image_layer_path(page, "inpaint", f"images/{Path(original_rel).name}")
        img_name = Path(original_rel).name
        inpainted_path = work_dir / inpaint_rel
        if not inpainted_path.exists():
            inpainted_path = work_dir / "originals" / img_name
        if not inpainted_path.exists():
            candidate = Path(inpaint_rel)
            if candidate.exists():
                inpainted_path = candidate
        if not inpainted_path.exists():
            candidate = Path(original_rel)
            if candidate.exists():
                inpainted_path = candidate

        if not inpainted_path.exists():
            emit("error", message=f"Imagem base nao encontrada: {img_name}")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        trans_texts = _visible_render_texts(_page_text_layers_for_renderer(page, page_idx))
        original_path = work_dir / original_rel
        if not original_path.exists():
            candidate = Path(original_rel)
            if candidate.exists():
                original_path = candidate
        render_base_path = _prepare_inpaint_base_for_render(
            original_path=original_path,
            inpainted_path=inpainted_path,
            texts=trans_texts,
            temp_output_path=output_path.parent / f"{output_path.stem}-base{output_path.suffix}",
            update_inpaint=False,
        )
        trans_page_dict = {"texts": trans_texts}

        from typesetter.renderer import _typeset_single_page

        _typeset_single_page((str(render_base_path), trans_page_dict, str(output_path.parent)))
        renderer_output = output_path.parent / Path(render_base_path).name
        if renderer_output.exists() and renderer_output.resolve() != output_path.resolve():
            if output_path.exists():
                output_path.unlink()
            shutil.move(str(renderer_output), str(output_path))
        _apply_recovery_layer_for_page(project, page, output_path)

        emit("complete", output_path=str(output_path))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        emit("error", message=f"Falha no preview final: {e}\n{tb}")

def _run_retypeset(project_json_path: Path, page_idx: int):
    with open(project_json_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    work_dir = project_json_path.parent
    project["_work_dir"] = str(work_dir)
    pages = project.get("paginas", [])
    if page_idx < 0 or page_idx >= len(pages):
        emit("error", message="Indice de pagina invalido")
        return

    page = pages[page_idx]
    page_number = int(page.get("numero", page_idx + 1) or page_idx + 1)
    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
    inpaint_rel = _resolve_image_layer_path(page, "inpaint", f"images/{Path(original_rel).name}")
    rendered_rel = _resolve_image_layer_path(page, "rendered", f"translated/{Path(original_rel).name}")

    img_name = Path(original_rel).name
    inpainted_path = work_dir / inpaint_rel
    output_dir = work_dir / "translated"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not inpainted_path.exists():
        inpainted_path = work_dir / "originals" / img_name

    if not inpainted_path.exists():
        emit("error", message=f"Imagem base nao encontrada: {img_name}")
        return

    trans_texts = _page_text_layers_for_renderer(page, page_idx)
    trans_page_dict = {"texts": _visible_render_texts(trans_texts)}
    original_path = work_dir / original_rel
    if not original_path.exists():
        candidate = Path(original_rel)
        if candidate.exists():
            original_path = candidate
    inpainted_path = _prepare_inpaint_base_for_render(
        original_path=original_path,
        inpainted_path=inpainted_path,
        texts=trans_page_dict["texts"],
        update_inpaint=True,
    )

    try:
        from typesetter.renderer import _typeset_single_page
        _typeset_single_page((str(inpainted_path), trans_page_dict, str(output_dir)))
        rendered_path = output_dir / img_name
        _apply_recovery_layer_for_page(project, page, rendered_path)
        page["text_layers"] = trans_texts
        _ensure_image_layer(page, "base", original_rel, visible=True, locked=True)
        _ensure_image_layer(page, "mask", _resolve_image_layer_path(page, "mask", f"layers/mask/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "inpaint", inpaint_rel, visible=False, locked=True)
        _ensure_image_layer(page, "brush", _resolve_image_layer_path(page, "brush", f"layers/brush/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "recovery", _resolve_image_layer_path(page, "recovery", f"layers/recovery/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "rendered", rendered_rel, visible=True, locked=True)
        _sync_page_legacy_aliases(page)
        project["versao"] = "2.0"
        _save_project_json(project_json_path, project)
        emit("complete", output_path=str(rendered_path))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        emit("error", message=f"Falha no retypeset: {e}\n{tb}")


def _run_reinpaint(project_json_path: Path, page_idx: int, region: dict | None = None):
    started = time.perf_counter()
    with open(project_json_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    work_dir = project_json_path.parent
    project["_work_dir"] = str(work_dir)
    pages = project.get("paginas", [])
    if page_idx < 0 or page_idx >= len(pages):
        emit("error", message="Indice de pagina invalido")
        return

    page = pages[page_idx]
    page_number = int(page.get("numero", page_idx + 1) or page_idx + 1)
    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
    inpaint_rel = _resolve_image_layer_path(page, "inpaint", f"images/{Path(original_rel).name}")
    img_name = Path(original_rel).name
    original_path = work_dir / original_rel
    inpaint_path = work_dir / inpaint_rel
    if not original_path.exists():
        original_path = work_dir / "originals" / img_name

    if not original_path.exists():
        candidate = Path(original_rel)
        if candidate.exists():
            original_path = candidate

    if not original_path.exists():
        emit("error", message=f"Imagem original nao encontrada: {img_name}")
        return

    page_texts = _page_text_layers_for_renderer(page, page_idx)
    is_regional = _region_bbox(region) is not None
    regional_texts = [layer for layer in page_texts if _layer_in_region(layer, region)] if is_regional else page_texts
    source_path = inpaint_path if is_regional and inpaint_path.exists() else original_path
    models_dir = Path(
        os.getenv("TRADUZAI_MODELS_DIR")
        or project.get("_models_dir")
        or project.get("models_dir")
        or "D:/traduzai_data/models"
    )
    output_dir = (
        work_dir / "editor_cache" / "reinpaint_regions" / f"page-{page_number:04d}"
        if is_regional
        else inpaint_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    inpaint_blocks = page.get("inpaint_blocks") or [
        {
            "bbox": t.get("source_bbox", t.get("layout_bbox", [0, 0, 0, 0])),
            "confidence": float(t.get("ocr_confidence", t.get("confianca_ocr", 0.0)) or 0.0),
        }
        for t in page_texts
    ]
    external_mask_block = None
    if is_regional:
        inpaint_blocks = [
            block
            for block in inpaint_blocks
            if isinstance(block, dict) and _block_in_region(block, region)
        ]
        external_mask_block = _external_mask_vision_block(region)
        if external_mask_block is not None:
            inpaint_blocks = [external_mask_block]
        if not inpaint_blocks:
            inpaint_blocks = [
                {
                    "bbox": _resolved_layer_bbox(layer),
                    "confidence": float(layer.get("ocr_confidence", layer.get("confianca_ocr", 1.0)) or 1.0),
                }
                for layer in regional_texts
            ]

    try:
        from inpainter.lama import run_inpainting
        ocr_data = {
            "image": str(source_path),
            "width": 0,
            "height": 0,
            "texts": regional_texts,
            "_vision_blocks": inpaint_blocks,
        }
        if external_mask_block is not None:
            ocr_data["_skip_internal_post_cleanup"] = True
        inpaint_started = time.perf_counter()
        outputs = run_inpainting(
            image_files=[source_path],
            ocr_results=[ocr_data],
            output_dir=str(output_dir),
            models_dir=str(models_dir),
        )
        inpaint_seconds = time.perf_counter() - inpaint_started
        finalize_started = time.perf_counter()
        completed_output_path = _finalize_reinpaint_output_path(
            outputs=outputs,
            is_regional=is_regional,
            base_path=inpaint_path,
            fallback_path=original_path,
            bbox=_region_bbox(region),
            default_path=output_dir / source_path.name,
        )
        finalize_seconds = time.perf_counter() - finalize_started
        _ensure_image_layer(page, "base", original_rel, visible=True, locked=True)
        _ensure_image_layer(page, "mask", _resolve_image_layer_path(page, "mask", f"layers/mask/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "inpaint", inpaint_rel, visible=False, locked=True)
        _ensure_image_layer(page, "brush", _resolve_image_layer_path(page, "brush", f"layers/brush/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "recovery", _resolve_image_layer_path(page, "recovery", f"layers/recovery/{page_number:03}.png"), visible=False, locked=False)
        if isinstance(page.get("text_layers"), list):
            page["text_layers"] = page_texts
        _sync_page_legacy_aliases(page)
        project["versao"] = "2.0"
        _save_project_json(project_json_path, project)
        elapsed_seconds = time.perf_counter() - started
        stats = ocr_data.get("_inpaint_round_stats") if isinstance(ocr_data, dict) else None
        logger.info(
            "Editor reinpaint page=%s elapsed=%.3fs inpaint=%.3fs finalize=%.3fs stats=%s",
            page_idx,
            elapsed_seconds,
            inpaint_seconds,
            finalize_seconds,
            stats or {},
        )
        emit(
            "complete",
            output_path=str(completed_output_path),
            elapsed_seconds=round(elapsed_seconds, 3),
            inpaint_seconds=round(inpaint_seconds, 3),
            finalize_seconds=round(finalize_seconds, 3),
            inpaint_stats=stats or {},
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        emit("error", message=f"Falha no reinpaint: {e}\n{tb}")


def _append_glossary_entry(entries: list[dict], seen: set[tuple[str, str]], source: str, target: str, kind: str, status: str) -> None:
    source = " ".join(str(source or "").split()).strip()
    target = " ".join(str(target or "").split()).strip()
    if not source:
        return
    key = (kind, source.lower())
    if key in seen:
        return
    seen.add(key)
    entries.append(
        {
            "source": source,
            "target": target or source,
            "kind": kind,
            "status": status,
        }
    )


def build_glossary_used_report(config: dict, context: dict, page_text_layers: list[dict]) -> dict:
    """Build a passive glossary report without enforcing new terms yet."""
    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for source, target in (config.get("glossario") or {}).items():
        _append_glossary_entry(entries, seen, source, target, "manual_glossary", "locked")
    for source, target in ((context or {}).get("glossario") or {}).items():
        _append_glossary_entry(entries, seen, source, target, "context_glossary", "locked")
    for source, target in ((context or {}).get("memoria_lexical") or {}).items():
        _append_glossary_entry(entries, seen, source, target, "memory", "suggested")
    for name in (context or {}).get("personagens") or []:
        _append_glossary_entry(entries, seen, name, name, "character", "suggested")
    for term in (context or {}).get("termos") or []:
        _append_glossary_entry(entries, seen, term, term, "term", "suggested")
    for faction in (context or {}).get("faccoes") or []:
        _append_glossary_entry(entries, seen, faction, faction, "faction", "suggested")

    used_hits: list[dict] = []
    blocked_translations: list[dict] = []
    violations: list[dict] = []
    qa_flags: dict[str, int] = {}
    for page_index, page in enumerate(page_text_layers, start=1):
        for layer_index, text in enumerate(page.get("texts", []) or [], start=1):
            for flag in text.get("qa_flags") or []:
                qa_flags[str(flag)] = qa_flags.get(str(flag), 0) + 1
                if str(flag) in {"glossary_violation", "placeholder_lost", "forbidden_translation"}:
                    violations.append(
                        {
                            "page": page_index,
                            "layer": text.get("id") or f"t{layer_index}",
                            "type": str(flag),
                            "severity": "critical",
                        }
                    )
            for hit in text.get("glossary_hits") or []:
                source_term = hit.get("source") or hit.get("source_term") or ""
                canonical = hit.get("target") or hit.get("canonical") or source_term
                used_hits.append(
                    {
                        "page": page_index,
                        "layer": text.get("id") or f"t{layer_index}",
                        "block_id": text.get("id") or f"t{layer_index}",
                        "source_term": source_term,
                        "canonical": canonical,
                        "policy": hit.get("policy") or "locked",
                        "hit": hit,
                    }
                )
            if text.get("translation_blocked_text"):
                blocked_translations.append(
                    {
                        "page": page_index,
                        "layer": text.get("id") or f"t{layer_index}",
                        "original": text.get("original") or text.get("text") or "",
                        "blocked_translation": text.get("translation_blocked_text"),
                        "qa_flags": list(text.get("qa_flags") or []),
                    }
                )

    locked_terms_used = sum(
        1
        for hit in used_hits
        if str((hit.get("hit") or {}).get("phase") or "").lower() in {"target", "placeholder"}
    )
    empty_reason = ""
    if not entries:
        empty_reason = "work_identity_unresolved" if not ((context or {}).get("title") or config.get("obra")) else "no_terms_mined"

    return {
        "mode": "shadow",
        "work_identity": {
            "title": config.get("obra") or (context or {}).get("title") or "",
            "source_language": config.get("idioma_origem", "en"),
            "target_language": config.get("idioma_destino", "pt-BR"),
            "resolved": bool((context or {}).get("fontes_usadas") or entries),
        },
        "work": {
            "title": config.get("obra") or (context or {}).get("title") or "",
            "source_language": config.get("idioma_origem", "en"),
            "target_language": config.get("idioma_destino", "pt-BR"),
            "synopsis_available": bool((context or {}).get("sinopse")),
            "genres": list((context or {}).get("genero") or []),
            "sources": list((context or {}).get("fontes_usadas") or []),
        },
        "entries": entries,
        "used_hits": used_hits,
        "hits": used_hits,
        "violations": violations,
        "blocked_translations": blocked_translations,
        "qa_flags": qa_flags,
        "summary": {
            "entry_count": len(entries),
            "used_hit_count": len(used_hits),
            "blocked_translation_count": len(blocked_translations),
            "terms_loaded": len(entries),
            "terms_used": len(used_hits),
            "locked_terms_used": locked_terms_used,
            "violations": len(violations),
            "review_required": len(blocked_translations) + len(violations),
            "empty_reason": empty_reason,
        },
    }


def build_project_json(config, context, ocr_results, page_text_layers, image_files, total_pages, elapsed):
    """Build the project.json structure."""
    from layout.region_grouping import group_regions
    from qa.translation_qa import summarize_flags

    pages = []
    qa_regions = []
    for i, (img, ocr, text_page) in enumerate(zip(image_files, ocr_results, page_text_layers)):
        text_layers = text_page.get("texts", [])
        text_layers = group_regions(text_layers)
        text_layers = [sanitize_simple_text_geometry(layer) for layer in text_layers]
        qa_regions.extend(text_layers)
        page = {
            "numero": i + 1,
            "image_layers": {
                "base": {
                    "key": "base",
                    "path": f"originals/{img.name}",
                    "visible": True,
                    "locked": True,
                },
                "mask": {
                    "key": "mask",
                    "path": f"layers/mask/{i + 1:03}.png",
                    "visible": False,
                    "locked": False,
                },
                "inpaint": {
                    "key": "inpaint",
                    "path": f"images/{img.name}",
                    "visible": False,
                    "locked": True,
                },
                "brush": {
                    "key": "brush",
                    "path": f"layers/brush/{i + 1:03}.png",
                    "visible": False,
                    "locked": False,
                },
                "recovery": {
                    "key": "recovery",
                    "path": f"layers/recovery/{i + 1:03}.png",
                    "visible": False,
                    "locked": False,
                },
                "rendered": {
                    "key": "rendered",
                    "path": f"translated/{img.name}",
                    "visible": True,
                    "locked": True,
                },
            },
            "inpaint_blocks": [
                project_block
                for block in ocr.get("_vision_blocks", [])
                for project_block in [_project_inpaint_block_from_vision_block(block)]
                if project_block is not None
            ],
            "page_profile": ocr.get("page_profile"),
            "page_quality": ocr.get("page_quality"),
            "route_history": ocr.get("route_history") or [],
            "text_layers": text_layers,
        }
        _sync_page_legacy_aliases(page)
        pages.append(page)

    return {
        "versao": "2.0",
        "app": "traduzai",
        "obra": config.get("obra", ""),
        "capitulo": config.get("capitulo", 1),
        "idioma_origem": config.get("idioma_origem", "en"),
        "idioma_destino": config.get("idioma_destino", "pt-BR"),
        "_ollama_host": config.get("ollama_host"),
        "_ollama_model": config.get("ollama_model"),
        "_models_dir": config.get("models_dir"),
        "_vision_worker_path": config.get("vision_worker_path"),
        "_work_dir": config.get("work_dir"),
        "contexto": context,
        "work_context": config.get("work_context") or {},
        "preset": config.get("preset") or {},
        "runtime_profile": config.get("runtime_profile_decision") or {},
        "paginas": pages,
        "estatisticas": {
            "total_paginas": total_pages,
            "total_textos": sum(len(p["text_layers"]) for p in pages),
            "tempo_processamento_seg": round(elapsed, 1),
            "data_criacao": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "qa": {
            "summary": summarize_flags(qa_regions),
        },
    }


def _run_process_block(project_path: Path, page_idx: int, block_id: str, mode: str):
    """Refazer processo para um unico bloco."""
    from ocr.detector import run_ocr_on_block
    from translator.translate import translate_single_block
    
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    work_dir = project_path.parent
    _attach_work_dir_log_handler(work_dir)
    project["_work_dir"] = str(work_dir)
    page = project["paginas"][page_idx]

    found_block = None
    # Support both "text_layers" and "textos" keys
    layers = page.get("text_layers") or page.get("textos") or []
    for layer in layers:
        if layer["id"] == block_id:
            found_block = layer
            break

    if not found_block:
        emit("error", message=f"Bloco {block_id} nao encontrado na pagina {page_idx}")
        return

    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
    orig_img = work_dir / original_rel
    if not orig_img.exists():
        orig_img = work_dir / "originals" / Path(original_rel).name
    if not orig_img.exists():
        candidate = Path(original_rel)
        if candidate.exists():
            orig_img = candidate

    block_bbox = _resolved_layer_bbox(found_block)
    found_block["bbox"] = block_bbox

    if mode == "ocr":
        emit_progress("ocr", 10, 50, message="Redetectando texto...")
        text, conf = run_ocr_on_block(str(orig_img), block_bbox)
        found_block["original"] = text
        found_block["ocr_confidence"] = conf
        found_block["confianca_ocr"] = conf

    if mode == "translate":
        emit_progress("translate", 50, 80, message="Traduzindo bloco...")
        translate_single_block(found_block, project)

    if mode == "inpaint" or mode == "ocr":
        emit_progress("inpaint", 30, 70, message="Limpando balão...")
        from inpainter.lama import run_inpainting
        ocr_data = {
            "image": str(orig_img),
            "texts": [found_block],
            "_vision_blocks": [{"bbox": block_bbox, "confidence": 1.0}],
        }
        inpaint_out_dir = work_dir / "images"
        inpaint_out_dir.mkdir(parents=True, exist_ok=True)
        run_inpainting([orig_img], [ocr_data], str(inpaint_out_dir))

    page["text_layers"] = _page_text_layers_for_renderer(page, page_idx)
    _sync_page_legacy_aliases(page)
    _save_project_json(project_path, project)

    emit_progress("render", 80, 95, message="Rerenderizando visual...")
    out_img = work_dir / _resolve_image_layer_path(
        page,
        "rendered",
        page.get("arquivo_traduzido", f"translated/{Path(original_rel).name}"),
    )
    render_page_image(project, page_idx, str(out_img))

    emit("complete", output_path=str(out_img))

def _run_detect_page(project_path: Path, page_idx: int, region: dict | None = None):
    from ocr.detector import run_ocr
    from ocr.contextual_reviewer import contextual_review_page
    from layout.balloon_layout import enrich_page_layout
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    work_dir = project_path.parent
    _attach_work_dir_log_handler(work_dir)
    project["_work_dir"] = str(work_dir)
    page = project["paginas"][page_idx]
    page_number = int(page.get("numero", page_idx + 1) or page_idx + 1)
    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
    img_name = Path(original_rel).name
    orig_img = work_dir / original_rel
    if not orig_img.exists():
        orig_img = work_dir / "originals" / img_name
    if not orig_img.exists():
        candidate = Path(original_rel)
        if candidate.exists():
            orig_img = candidate
    
    emit_progress("ocr", 0, 10, message="Detectando balões e textos...")
    
    # Run full OCR (detect + read)
    ocr_data = run_ocr(
        str(orig_img),
        models_dir=project.get("_models_dir", "models"),
        vision_worker_path=project.get("_vision_worker_path", ""),
        idioma_origem=project.get("idioma_origem", "en")
    )
    
    reviewed = contextual_review_page(ocr_data, [], [])
    reviewed["_connected_balloon_reasoner"] = {
        "provider": project.get("connected_balloon_reasoner", "ollama"),
        "enabled": project.get("connected_balloon_reasoner_enabled", True),
        "host": project.get("connected_balloon_ollama_host", project.get("_ollama_host", "http://localhost:11434")),
        "model": project.get("connected_balloon_ollama_model", "qwen2.5"),
        "use_image": project.get("connected_balloon_ollama_use_image", True),
    }
    reviewed = enrich_page_layout(reviewed)

    existing_layers = _page_text_layers_for_renderer(page, page_idx)
    context = project.get("contexto", {}) or {}
    reviewed_texts = [text for text in (reviewed.get("texts") or []) if isinstance(text, dict)]
    is_regional = _region_bbox(region) is not None
    if is_regional:
        detected_texts = [text for text in reviewed_texts if _bbox_in_region(text.get("bbox"), region)]
        outside_layers = [layer for layer in existing_layers if not _layer_in_region(layer, region)]
        translation_source_layers = [layer for layer in existing_layers if _layer_in_region(layer, region)]
    else:
        detected_texts = reviewed_texts
        outside_layers = []
        translation_source_layers = existing_layers
    carried_translations = _carry_translations_for_detected_layers(translation_source_layers, detected_texts)
    detected_layers = [
        build_text_layer(
            page_number=page_number,
            layer_index=len(outside_layers) + idx,
            ocr_text=text,
            translated=carried_translations[idx] if idx < len(carried_translations) else "",
            corpus_visual_benchmark=context.get("corpus_visual_benchmark", {}),
            corpus_textual_benchmark=context.get("corpus_textual_benchmark", {}),
        )
        for idx, text in enumerate(detected_texts)
    ]
    page["text_layers"] = outside_layers + detected_layers

    detected_blocks = [
        {
            "bbox": _bbox4(block.get("bbox")),
            "confidence": float(block.get("confidence", 0.0) or 0.0),
        }
        for block in reviewed.get("_vision_blocks", [])
        if isinstance(block, dict)
    ]
    if is_regional:
        outside_blocks = [
            block for block in (page.get("inpaint_blocks") or [])
            if isinstance(block, dict) and not _block_in_region(block, region)
        ]
        detected_blocks = [block for block in detected_blocks if _block_in_region(block, region)]
        page["inpaint_blocks"] = outside_blocks + detected_blocks
    else:
        page["inpaint_blocks"] = detected_blocks
    
    _ensure_image_layer(page, "base", original_rel, visible=True, locked=True)
    _ensure_image_layer(page, "mask", _resolve_image_layer_path(page, "mask", f"layers/mask/{page_number:03}.png"), visible=False, locked=False)
    _ensure_image_layer(page, "inpaint", _resolve_image_layer_path(page, "inpaint", f"images/{img_name}"), visible=False, locked=True)
    _ensure_image_layer(page, "brush", _resolve_image_layer_path(page, "brush", f"layers/brush/{page_number:03}.png"), visible=False, locked=False)
    _ensure_image_layer(page, "recovery", _resolve_image_layer_path(page, "recovery", f"layers/recovery/{page_number:03}.png"), visible=False, locked=False)
    _ensure_image_layer(page, "rendered", _resolve_image_layer_path(page, "rendered", f"translated/{img_name}"), visible=True, locked=True)
    _sync_page_legacy_aliases(page)
    _save_project_json(project_path, project)
        
    emit_progress("render", 80, 95, message="Rerenderizando visual...")
    out_img = work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")
    render_page_image(project, page_idx, str(out_img))
    
    emit_progress("render", 100, 100, message="Detecção concluída!")
    emit("complete", output_path=str(out_img))

def _run_ocr_page(project_path: Path, page_idx: int, region: dict | None = None):
    from ocr.detector import run_ocr_on_block
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    work_dir = project_path.parent
    _attach_work_dir_log_handler(work_dir)
    project["_work_dir"] = str(work_dir)
    page = project["paginas"][page_idx]
    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
    img_name = Path(original_rel).name
    orig_img = work_dir / original_rel
    if not orig_img.exists():
        orig_img = work_dir / "originals" / img_name
    if not orig_img.exists():
        candidate = Path(original_rel)
        if candidate.exists():
            orig_img = candidate

    layers = page.get("text_layers") or _page_text_layers_for_renderer(page, page_idx)
    if not layers:
        page_number = int(page.get("numero", page_idx + 1) or page_idx + 1)
        context = project.get("contexto", {}) or {}
        layers = [
            build_text_layer(
                page_number=page_number,
                layer_index=idx,
                ocr_text={
                    "bbox": _bbox4(block.get("bbox")),
                    "balloon_bbox": _bbox4(block.get("bbox")),
                    "confidence": float(block.get("confidence", 0.0) or 0.0),
                    "tipo": block.get("tipo", "fala"),
                    "text": "",
                },
                translated="",
                corpus_visual_benchmark=context.get("corpus_visual_benchmark", {}),
                corpus_textual_benchmark=context.get("corpus_textual_benchmark", {}),
            )
            for idx, block in enumerate(page.get("inpaint_blocks") or [])
            if isinstance(block, dict)
        ]
    page["text_layers"] = layers
    target_layers = [layer for layer in layers if _layer_in_region(layer, region)]
    total = len(target_layers)
    
    for i, layer in enumerate(target_layers):
        progress = int((i / max(1, total)) * 100)
        emit_progress("ocr", progress, 10, message=f"OCR em bloco {i+1}/{total}...")
        block_bbox = _resolved_layer_bbox(layer)
        layer["bbox"] = block_bbox
        text, conf = run_ocr_on_block(str(orig_img), block_bbox)
        layer["original"] = text
        layer["ocr_confidence"] = conf
        layer["confianca_ocr"] = conf

    page["text_layers"] = _page_text_layers_for_renderer(page, page_idx)
    _sync_page_legacy_aliases(page)
    _save_project_json(project_path, project)

    emit_progress("render", 80, 95, message="Rerenderizando visual...")
    out_img = work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")
    render_page_image(project, page_idx, str(out_img))
    
    emit_progress("render", 100, 100, message="OCR concluído!")
    emit("complete", output_path=str(out_img))

def _run_translate_page(project_path: Path, page_idx: int, region: dict | None = None):
    from translator.translate import translate_pages
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    work_dir = project_path.parent
    _attach_work_dir_log_handler(work_dir)
    project["_work_dir"] = str(work_dir)
    # We only translate common fields for one page
    # translate_pages expects a list of pages
    page = project["paginas"][page_idx]
    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
    img_name = Path(original_rel).name
    
    emit_progress("translate", 0, 10, message="Traduzindo textos...")
    
    context = project.get("contexto", {}) or {}
    source_layers = page.get("text_layers") or page.get("textos") or []
    indexed_layers = list(enumerate(source_layers))
    target_indexed_layers = [
        (idx, layer)
        for idx, layer in indexed_layers
        if isinstance(layer, dict) and _layer_in_region(layer, region)
    ]
    page_to_translate = {
        "texts": [
            {
                **layer,
                "text": layer.get("text") or layer.get("original") or "",
            }
            for _, layer in target_indexed_layers
        ]
    }
    
    if target_indexed_layers:
        translated_pages = translate_pages(
            ocr_results=[page_to_translate],
            obra=project.get("obra", ""),
            context=context,
            glossario=context.get("glossario", {}) or {},
            idioma_origem=project.get("idioma_origem", "en"),
            idioma_destino=project.get("idioma_destino", "pt-BR"),
            ollama_host=project.get("_ollama_host") or "http://localhost:11434",
            ollama_model=project.get("_ollama_model") or "traduzai-translator",
            translation_context=project.get("translation_context") or None,
        )
        translated_targets = (
            translated_pages[0].get("texts", page_to_translate["texts"])
            if translated_pages else page_to_translate["texts"]
        )
        merged_layers = list(source_layers)
        for (source_idx, _), translated_layer in zip(target_indexed_layers, translated_targets):
            merged_layers[source_idx] = translated_layer
    else:
        merged_layers = list(source_layers)
    page["text_layers"] = merged_layers
    page["textos"] = merged_layers
    _sync_page_legacy_aliases(page)
    _save_project_json(project_path, project)
        
    emit_progress("render", 80, 95, message="Rerenderizando visual...")
    out_img = work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")
    render_page_image(project, page_idx, str(out_img))
    
    emit_progress("render", 100, 100, message="Tradução concluída!")
    emit("complete", output_path=str(out_img))

def _build_cli_help() -> str:
    return "\n".join(
        [
            "Uso: python pipeline/main.py <config.json> [comando]",
            "Uso novo: python pipeline/main.py --input <pasta> --work <obra> --target pt-BR --mode mock|real --output <dir>",
            "",
            "Comandos disponiveis:",
            "  --help                              Mostra esta ajuda",
            "  --hardware-info                     Exibe CPU, RAM e GPU detectadas",
            "  --list-supported-languages          Lista idiomas suportados pelo tradutor",
            "  --warmup-visual [args]              Inicializa a stack visual",
            "  --retypeset <project> <page>        Rerenderiza uma pagina",
            "  --render-preview-page <project> <page> <page-json> <output>  Renderiza preview sem salvar projeto",
            "  --detect-page <project> <page>      Reprocessa deteccao da pagina",
            "  --ocr-page <project> <page>         Reprocessa OCR da pagina",
            "  --translate-page <project> <page>   Reprocessa traducao da pagina",
            "  --reinpaint-page <project> <page>   Reprocessa limpeza/inpaint da pagina",
            "  page actions aceitam: --region-bbox x1,y1,x2,y2 --external-mask <path>",
            "  --process-block <mode> <project> <page> <block_id>  Reprocessa um bloco",
            "  --input <pasta|imagem>                Roda o pipeline runner/CLI debug",
            "  --mode mock|real                      Define runner offline mock ou pipeline real",
            "  --debug                              Gera artefatos de debug",
            "  --skip-inpaint                       Pula inpaint quando suportado pelo runner",
            "  --skip-ocr                           Pula OCR quando suportado pelo runner",
            "  --strict                             Retorna codigo != 0 quando QA bloqueia",
            "  --export-mode clean|with_warnings|debug  Modo de validacao do export",
        ]
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        emit("error", message=f"{e}\n--- traceback ---\n{tb}")
        sys.exit(1)
