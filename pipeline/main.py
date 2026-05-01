"""
TraduzAi Pipeline - Entry point
Receives a config JSON path as argument, runs the full pipeline,
and outputs JSON progress messages to stdout for the Tauri sidecar to consume.
"""

import os
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

_EMIT_STDOUT_FAILED = False
_PIPELINE_FILE_HANDLER: logging.Handler | None = None


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


def _run_mock_pipeline_runner(config: dict) -> int:
    source_path = Path(config["source_path"])
    work_dir = Path(config["work_dir"])
    originals_dir = work_dir / "originals"
    images_dir = work_dir / "images"
    translated_dir = work_dir / "translated"
    for directory in [originals_dir, images_dir, translated_dir, work_dir / "layers" / "mask"]:
        directory.mkdir(parents=True, exist_ok=True)

    image_files = _list_input_images(source_path)
    pages = []
    issues = []
    for index, image_path in enumerate(image_files, start=1):
        output_name = f"{index:03}{image_path.suffix.lower()}"
        for directory in [originals_dir, images_dir, translated_dir]:
            shutil.copy2(image_path, directory / output_name)
        (work_dir / "layers" / "mask" / f"{index:03}.png").write_bytes(b"")

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

    if sys.argv[1] == "--detect-page" and len(sys.argv) >= 4:
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        _run_detect_page(project_json_path, page_idx)
        return

    if sys.argv[1] == "--ocr-page" and len(sys.argv) >= 4:
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        _run_ocr_page(project_json_path, page_idx)
        return

    if sys.argv[1] == "--translate-page" and len(sys.argv) >= 4:
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        _run_translate_page(project_json_path, page_idx)
        return

    if sys.argv[1] == "--reinpaint-page" and len(sys.argv) >= 4:
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        _run_reinpaint(project_json_path, page_idx)
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

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    wait_if_paused(config)
    work_dir = Path(config["work_dir"])
    models_dir = Path(config["models_dir"])
    _attach_work_dir_log_handler(work_dir)
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

    corpus_bundle = load_corpus_bundle(
        config.get("obra", ""),
        models_root=models_dir / "corpus",
        fallback_root=Path(__file__).resolve().parent / "models" / "corpus",
    )
    corpus_expected_terms = extract_expected_terms(corpus_bundle)

    emit_progress("extract", 0, 0, message="Extraindo arquivos...")
    try:
        image_files, tmp_dir = extract(source_path, work_dir)
    except Exception as e:
        emit("error", message=str(e))
        sys.exit(1)

    for img_path in image_files:
        shutil.copy2(img_path, originals_dir / img_path.name)

    total_pages = len(image_files)
    mode = config.get("mode", "auto")
    ocr_results = []
    page_text_layers = []
    context = merge_corpus_into_context(config.get("contexto", {}), corpus_bundle)
    start_time = time.time()

    if mode == "manual":
        emit_progress("extract", 100, 95, message="Modo Manual: Preparando projeto...")
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
            from concurrent.futures import ThreadPoolExecutor as _CtxTPE
            _ctx_pool = _CtxTPE(max_workers=1)
            _context_future = _ctx_pool.submit(fetch_context, config["obra"])

        # Warmup models
        emit_progress("ocr", 0, 9, total=total_pages, message="Carregando modelos...")
        _warmup_visual_stack(str(models_dir), "max")

        # Resolve AniList context
        if _context_future:
            try:
                context = merge_context(context, _context_future.result(timeout=10))
            except Exception:
                pass
            _ctx_pool.shutdown(wait=False)

        # Bridges
        connected_reasoner_config = {
            "provider": config.get("connected_balloon_reasoner", "ollama"),
            "enabled": config.get("connected_balloon_reasoner_enabled", True),
            "host": config.get(
                "connected_balloon_ollama_host",
                config.get("ollama_host", "http://localhost:11434"),
            ),
            "model": config.get("connected_balloon_ollama_model", "qwen2.5"),
            "use_image": config.get("connected_balloon_ollama_use_image", True),
        }

        class StripDetector:
            def detect(self, img, conf_threshold=None):
                from vision_stack.runtime import _profile_to_detection_threshold
                thresh = conf_threshold if conf_threshold is not None else _profile_to_detection_threshold("max")
                return _get_detector("max").detect(img, conf_threshold=thresh)
                
        class StripRuntime:
            def run_ocr_stage(self, img, page_dict):
                return run_ocr_stage(img, page_dict, profile="max", idioma_origem=config.get("idioma_origem", "en"))
        
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
        )


        
        # Sincroniza saídas para o wrap-up final (project.json)
        image_files = [p.path for p in output_pages]
        ocr_results = [p.ocr_result for p in output_pages]
        page_text_layers = [p.text_layers for p in output_pages]
        total_pages = len(output_pages)

        # Copia imagens processadas para images/ para que o editor as veja como base de inpaint
        for p in output_pages:
            inpaint_target = images_dir / p.path.name
            if getattr(p, "inpainted_image", None) is not None:
                cv2.imwrite(str(inpaint_target), p.inpainted_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                shutil.copy2(p.path, inpaint_target)

        # Substituir originals/ (que tem os splits brutos do .cbz) pelas paginas
        # reassembladas originais. Sem isso, project.json aponta para originals/NNN.jpg
        # que nao existem (disco mantem nomes originais como originals/002__001.jpg).
        for stale in originals_dir.glob("*"):
            if stale.is_file():
                stale.unlink()
        for p in output_pages:
            original_target = originals_dir / p.path.name
            if getattr(p, "original_image", None) is not None:
                cv2.imwrite(str(original_target), p.original_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                shutil.copy2(p.path, original_target)

        # Garantir diretorios de camadas + PNGs transparentes 1x1 para mask/brush
        # (UI Tauri reclama de 404 quando layers/mask/NNN.png nao existe).
        layers_root = work_dir / "layers"
        (layers_root / "mask").mkdir(parents=True, exist_ok=True)
        (layers_root / "brush").mkdir(parents=True, exist_ok=True)
        (layers_root / "text-preview").mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image as _PILImage
            _empty_png = _PILImage.new("RGBA", (1, 1), (0, 0, 0, 0))
            for i in range(1, len(output_pages) + 1):
                mask_path = layers_root / "mask" / f"{i:03}.png"
                brush_path = layers_root / "brush" / f"{i:03}.png"
                if not mask_path.exists():
                    _empty_png.save(mask_path)
                if not brush_path.exists():
                    _empty_png.save(brush_path)
        except Exception:
            pass




    # Wrap up
    emit_progress("typeset", 100, 98, message="Finalizando projeto...")
    project_data = build_project_json(config, context, ocr_results, page_text_layers, image_files, total_pages, time.time()-start_time)
    from structured_logger import StructuredLogger, build_log_summary
    structured_logger = StructuredLogger(config.get("logs_dir") or (work_dir / "logs"), config.get("job_id", "run"))
    log_summary = build_log_summary(project_data)
    structured_logger.log(stage="summary", event="run_summary", payload=log_summary)
    project_data["log"] = {
        "structured_log_path": str(structured_logger.path),
        "summary": log_summary,
    }
    _save_project_json(work_dir / "project.json", project_data)
    finalize_decision_trace(
        {
            "total_paginas": total_pages,
            "total_textos": project_data.get("estatisticas", {}).get("total_textos", 0),
        }
    )
    
    cleanup(tmp_dir)
    emit_progress("typeset", 100, 100, message="Concluido!")
    emit("complete", output_path=str(work_dir))


def _default_text_style() -> dict:
    return {
        "fonte": "ComicNeue-Bold.ttf",
        "tamanho": 28,
        "cor": "#FFFFFF",
        "cor_gradiente": [],
        "contorno": "#000000",
        "contorno_px": 2,
        "glow": False,
        "glow_cor": "",
        "glow_px": 0,
        "sombra": False,
        "sombra_cor": "",
        "sombra_offset": [0, 0],
        "bold": False,
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
    layout_bbox = _bbox4(
        ocr_text.get("layout_bbox"),
        ocr_text.get("balloon_bbox") or source_bbox,
    )
    text_pixel_bbox = _bbox4(
        _normalize_relative_y_bbox(ocr_text.get("text_pixel_bbox"), source_bbox),
        source_bbox,
    )
    style = _merge_style(ocr_text.get("estilo"))
    balloon_subregions = _normalize_relative_y_bbox_list(ocr_text.get("balloon_subregions"), layout_bbox)
    connected_lobe_bboxes = _normalize_relative_y_bbox_list(ocr_text.get("connected_lobe_bboxes"), layout_bbox)
    connected_text_groups = _normalize_relative_y_bbox_list(ocr_text.get("connected_text_groups"), layout_bbox)
    connected_position_bboxes = _normalize_relative_y_bbox_list(ocr_text.get("connected_position_bboxes"), layout_bbox)
    connected_focus_bboxes = _normalize_relative_y_bbox_list(ocr_text.get("connected_focus_bboxes"), layout_bbox)
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
        "balloon_type": ocr_text.get("balloon_type"),
        "inpaint_mode": ocr_text.get("inpaint_mode"),
        "inpaint_strategy": ocr_text.get("inpaint_strategy"),
        "balloon_bbox": layout_bbox,
        "balloon_subregions": balloon_subregions,
        "layout_group_size": int(ocr_text.get("layout_group_size", 1) or 1),
        "connected_children": ocr_text.get("connected_children"),
        "connected_text_groups": connected_text_groups,
        "connected_lobe_bboxes": connected_lobe_bboxes,
        "connected_position_bboxes": connected_position_bboxes,
        "connected_focus_bboxes": connected_focus_bboxes,
        "connected_balloon_orientation": ocr_text.get("connected_balloon_orientation"),
        "connected_detection_confidence": float(ocr_text.get("connected_detection_confidence", 0.0) or 0.0),
        "connected_group_confidence": float(ocr_text.get("connected_group_confidence", 0.0) or 0.0),
        "connected_position_confidence": float(ocr_text.get("connected_position_confidence", 0.0) or 0.0),
        "subregion_confidence": float(ocr_text.get("subregion_confidence", 0.0) or 0.0),
        "connected_position_reasoner": ocr_text.get("connected_position_reasoner"),
        "connected_reasoner_model": ocr_text.get("connected_reasoner_model"),
        "connected_reasoner_notes": ocr_text.get("connected_reasoner_notes"),
        "_connected_slot_index": ocr_text.get("_connected_slot_index"),
        "_connected_slot_count": ocr_text.get("_connected_slot_count"),
        "_connected_vertical_bias_ratio": ocr_text.get("_connected_vertical_bias_ratio"),
        "_is_lobe_subregion": bool(ocr_text.get("_is_lobe_subregion", False)),
        "page_profile": ocr_text.get("page_profile"),
        "block_profile": ocr_text.get("block_profile"),
        "layout_profile": ocr_text.get("layout_profile") or ocr_text.get("block_profile"),
        "entity_flags": list(ocr_text.get("entity_flags") or []),
        "entity_repairs": list(ocr_text.get("entity_repairs") or []),
        "glossary_hits": list(ocr_text.get("glossary_hits") or []),
        "qa_flags": list(ocr_text.get("qa_flags") or []),
        "corpus_visual_benchmark": corpus_visual_benchmark,
        "corpus_textual_benchmark": corpus_textual_benchmark,
    }


def _normalize_text_layer_for_renderer(raw_layer: dict, page_number: int, layer_index: int) -> dict:
    source_bbox = _bbox4(raw_layer.get("source_bbox"), raw_layer.get("bbox"))
    layout_bbox = _bbox4(raw_layer.get("layout_bbox"), raw_layer.get("balloon_bbox") or source_bbox)
    text_pixel_bbox = _bbox4(
        _normalize_relative_y_bbox(raw_layer.get("text_pixel_bbox"), source_bbox),
        source_bbox,
    )
    style = _merge_style(raw_layer.get("style") or raw_layer.get("estilo"))
    layer_id = raw_layer.get("id") or f"tl_{page_number:03}_{layer_index + 1:03}"
    translated = raw_layer.get("translated", raw_layer.get("traduzido", ""))
    original = raw_layer.get("original", raw_layer.get("text", ""))
    balloon_subregions = _normalize_relative_y_bbox_list(raw_layer.get("balloon_subregions"), layout_bbox)
    connected_lobe_bboxes = _normalize_relative_y_bbox_list(raw_layer.get("connected_lobe_bboxes"), layout_bbox)
    connected_text_groups = _normalize_relative_y_bbox_list(raw_layer.get("connected_text_groups"), layout_bbox)
    connected_position_bboxes = _normalize_relative_y_bbox_list(raw_layer.get("connected_position_bboxes"), layout_bbox)
    connected_focus_bboxes = _normalize_relative_y_bbox_list(raw_layer.get("connected_focus_bboxes"), layout_bbox)
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
        "balloon_type": raw_layer.get("balloon_type"),
        "inpaint_mode": raw_layer.get("inpaint_mode"),
        "inpaint_strategy": raw_layer.get("inpaint_strategy"),
        "balloon_bbox": _bbox4(raw_layer.get("balloon_bbox"), layout_bbox),
        "balloon_subregions": balloon_subregions,
        "layout_group_size": int(raw_layer.get("layout_group_size", 1) or 1),
        "connected_children": raw_layer.get("connected_children"),
        "connected_text_groups": connected_text_groups,
        "connected_lobe_bboxes": connected_lobe_bboxes,
        "connected_position_bboxes": connected_position_bboxes,
        "connected_focus_bboxes": connected_focus_bboxes,
        "connected_balloon_orientation": raw_layer.get("connected_balloon_orientation"),
        "connected_detection_confidence": float(raw_layer.get("connected_detection_confidence", 0.0) or 0.0),
        "connected_group_confidence": float(raw_layer.get("connected_group_confidence", 0.0) or 0.0),
        "connected_position_confidence": float(raw_layer.get("connected_position_confidence", 0.0) or 0.0),
        "subregion_confidence": float(raw_layer.get("subregion_confidence", 0.0) or 0.0),
        "connected_position_reasoner": raw_layer.get("connected_position_reasoner"),
        "connected_reasoner_model": raw_layer.get("connected_reasoner_model"),
        "connected_reasoner_notes": raw_layer.get("connected_reasoner_notes"),
        "_connected_slot_index": raw_layer.get("_connected_slot_index"),
        "_connected_slot_count": raw_layer.get("_connected_slot_count"),
        "_connected_vertical_bias_ratio": raw_layer.get("_connected_vertical_bias_ratio"),
        "_is_lobe_subregion": bool(raw_layer.get("_is_lobe_subregion", False)),
        "page_profile": raw_layer.get("page_profile"),
        "block_profile": raw_layer.get("block_profile"),
        "layout_profile": raw_layer.get("layout_profile") or raw_layer.get("block_profile"),
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

    text_layers = page.get("text_layers") or []
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
            "qa_flags": list(layer.get("qa_flags") or []),
            "balloon_bbox": _bbox4(layer.get("balloon_bbox"), layer.get("layout_bbox") or layer.get("bbox")),
            "balloon_type": layer.get("balloon_type"),
            "layout_profile": layer.get("layout_profile") or layer.get("block_profile"),
            "layout_group_size": int(layer.get("layout_group_size", 1) or 1),
            "skip_processing": bool(layer.get("skip_processing", False)),
            "balloon_subregions": _bbox4_list(layer.get("balloon_subregions")),
        }
        for layer in text_layers
        if isinstance(layer, dict)
    ]


def _save_project_json(project_json_path: Path, project: dict) -> None:
    from project_writer import write_project_json_atomic

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
    except Exception as e:
        sys.stderr.write(f"Falha ao renderizar imagem da pagina: {e}\n")

def _visible_render_texts(texts: list[dict]) -> list[dict]:
    return [text for text in texts if text.get("visible", True) is not False]

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
        trans_page_dict = {"texts": trans_texts}

        from typesetter.renderer import _typeset_single_page

        _typeset_single_page((str(inpainted_path), trans_page_dict, str(output_path.parent)))
        renderer_output = output_path.parent / Path(inpainted_path).name
        if renderer_output.exists() and renderer_output.resolve() != output_path.resolve():
            if output_path.exists():
                output_path.unlink()
            shutil.move(str(renderer_output), str(output_path))

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

    try:
        from typesetter.renderer import _typeset_single_page
        _typeset_single_page((str(inpainted_path), trans_page_dict, str(output_dir)))
        page["text_layers"] = trans_texts
        _ensure_image_layer(page, "base", original_rel, visible=True, locked=True)
        _ensure_image_layer(page, "mask", _resolve_image_layer_path(page, "mask", f"layers/mask/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "inpaint", inpaint_rel, visible=False, locked=True)
        _ensure_image_layer(page, "brush", _resolve_image_layer_path(page, "brush", f"layers/brush/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "rendered", rendered_rel, visible=True, locked=True)
        _sync_page_legacy_aliases(page)
        project["versao"] = "2.0"
        _save_project_json(project_json_path, project)
        emit("complete", output_path=str(output_dir / img_name))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        emit("error", message=f"Falha no retypeset: {e}\n{tb}")


def _run_reinpaint(project_json_path: Path, page_idx: int):
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
    if not original_path.exists():
        original_path = work_dir / "originals" / img_name
    output_dir = (work_dir / inpaint_rel).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_path.exists():
        candidate = Path(original_rel)
        if candidate.exists():
            original_path = candidate

    if not original_path.exists():
        emit("error", message=f"Imagem original nao encontrada: {img_name}")
        return

    page_texts = _page_text_layers_for_renderer(page, page_idx)

    inpaint_blocks = page.get("inpaint_blocks") or [
        {
            "bbox": t.get("source_bbox", t.get("layout_bbox", [0, 0, 0, 0])),
            "confidence": float(t.get("ocr_confidence", t.get("confianca_ocr", 0.0)) or 0.0),
        }
        for t in page_texts
    ]

    try:
        from inpainter.lama import run_inpainting
        ocr_data = {
            "image": str(original_path),
            "width": 0,
            "height": 0,
            "texts": page_texts,
            "_vision_blocks": inpaint_blocks,
        }
        outputs = run_inpainting(
            image_files=[original_path],
            ocr_results=[ocr_data],
            output_dir=str(output_dir),
            models_dir=str(Path("D:/traduzai_data/models")),
        )
        _ensure_image_layer(page, "base", original_rel, visible=True, locked=True)
        _ensure_image_layer(page, "mask", _resolve_image_layer_path(page, "mask", f"layers/mask/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "inpaint", inpaint_rel, visible=False, locked=True)
        _ensure_image_layer(page, "brush", _resolve_image_layer_path(page, "brush", f"layers/brush/{page_number:03}.png"), visible=False, locked=False)
        if isinstance(page.get("text_layers"), list):
            page["text_layers"] = page_texts
        _sync_page_legacy_aliases(page)
        project["versao"] = "2.0"
        _save_project_json(project_json_path, project)
        emit("complete", output_path=str(outputs[0]) if outputs else str(output_dir / img_name))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        emit("error", message=f"Falha no reinpaint: {e}\n{tb}")


def build_project_json(config, context, ocr_results, page_text_layers, image_files, total_pages, elapsed):
    """Build the project.json structure."""
    from layout.region_grouping import group_regions
    from qa.translation_qa import summarize_flags

    pages = []
    qa_regions = []
    for i, (img, ocr, text_page) in enumerate(zip(image_files, ocr_results, page_text_layers)):
        text_layers = text_page.get("texts", [])
        text_layers = group_regions(text_layers)
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
                "rendered": {
                    "key": "rendered",
                    "path": f"translated/{img.name}",
                    "visible": True,
                    "locked": True,
                },
            },
            "inpaint_blocks": [
                {
                    "bbox": block.get("bbox", [0, 0, 0, 0]),
                    "confidence": block.get("confidence", 0.0),
                }
                for block in ocr.get("_vision_blocks", [])
            ],
            "page_profile": ocr.get("page_profile"),
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

def _run_detect_page(project_path: Path, page_idx: int):
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
    reviewed_texts = reviewed.get("texts") or []
    carried_translations = _carry_translations_for_detected_layers(existing_layers, reviewed_texts)
    page["text_layers"] = [
        build_text_layer(
            page_number=page_number,
            layer_index=idx,
            ocr_text=text,
            translated=carried_translations[idx] if idx < len(carried_translations) else "",
            corpus_visual_benchmark=context.get("corpus_visual_benchmark", {}),
            corpus_textual_benchmark=context.get("corpus_textual_benchmark", {}),
        )
        for idx, text in enumerate(reviewed_texts)
        if isinstance(text, dict)
    ]
    page["inpaint_blocks"] = [
        {
            "bbox": _bbox4(block.get("bbox")),
            "confidence": float(block.get("confidence", 0.0) or 0.0),
        }
        for block in reviewed.get("_vision_blocks", [])
        if isinstance(block, dict)
    ]
    
    _ensure_image_layer(page, "base", original_rel, visible=True, locked=True)
    _ensure_image_layer(page, "mask", _resolve_image_layer_path(page, "mask", f"layers/mask/{page_number:03}.png"), visible=False, locked=False)
    _ensure_image_layer(page, "inpaint", _resolve_image_layer_path(page, "inpaint", f"images/{img_name}"), visible=False, locked=True)
    _ensure_image_layer(page, "brush", _resolve_image_layer_path(page, "brush", f"layers/brush/{page_number:03}.png"), visible=False, locked=False)
    _ensure_image_layer(page, "rendered", _resolve_image_layer_path(page, "rendered", f"translated/{img_name}"), visible=True, locked=True)
    _sync_page_legacy_aliases(page)
    _save_project_json(project_path, project)
        
    emit_progress("render", 80, 95, message="Rerenderizando visual...")
    out_img = work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")
    render_page_image(project, page_idx, str(out_img))
    
    emit_progress("render", 100, 100, message="Detecção concluída!")
    emit("complete", output_path=str(out_img))

def _run_ocr_page(project_path: Path, page_idx: int):
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
    total = len(layers)
    
    for i, layer in enumerate(layers):
        emit_progress("ocr", int((i / total) * 100), 10, message=f"OCR em bloco {i+1}/{total}...")
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

def _run_translate_page(project_path: Path, page_idx: int):
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
    page_to_translate = {
        "texts": [
            {
                **layer,
                "text": layer.get("text") or layer.get("original") or "",
            }
            for layer in source_layers
        ]
    }
    
    translated_pages = translate_pages(
        ocr_results=[page_to_translate],
        obra=project.get("obra", ""),
        context=context,
        glossario=context.get("glossario", {}) or {},
        idioma_origem=project.get("idioma_origem", "en"),
        idioma_destino=project.get("idioma_destino", "pt-BR"),
        ollama_host=project.get("_ollama_host") or "http://localhost:11434",
        ollama_model=project.get("_ollama_model") or "traduzai-translator"
    )
    
    translated_layers = (
        translated_pages[0].get("texts", page_to_translate["texts"])
        if translated_pages else page_to_translate["texts"]
    )
    page["text_layers"] = translated_layers
    page["textos"] = translated_layers
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
