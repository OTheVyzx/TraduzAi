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
import subprocess
import sys
import time
import faulthandler
import logging
import contextlib
import importlib.util
from pathlib import Path

# Adiciona o diretório da pipeline ao path para resolver imports locais no Pyright/Linter
pipeline_root = Path(__file__).parent.absolute()
if str(pipeline_root) not in sys.path:
    sys.path.insert(0, str(pipeline_root))

# Lazy imports are moved inside functions to allow fast --hardware-info and --list-supported-languages calls

from utils.decision_log import configure_decision_trace, finalize_decision_trace
from typesetter.style_policy import SOURCE_STYLE_CONFIDENCE_THRESHOLD, normalize_auto_typesetting_style
from layout.simple_text_geometry import normalize_text_geometry, resolve_text_anchor_bbox, sanitize_simple_text_geometry
from ocr.postprocess import apply_language_guards, postprocess_ocr_fragments, split_sfx_inline
from ocr.text_router import ROUTE_ACTIONS
from sfx.candidate import enrich_sfx_candidate
from sfx.ocr_probe import probe_sfx_candidate_ocr
from sfx.promotion import promote_visual_sfx_candidate, suppress_normal_ocr_overlapping_sfx
from sfx.script_probe import probe_sfx_candidate_script

_EMIT_STDOUT_FAILED = False
_PIPELINE_FILE_HANDLER: logging.Handler | None = None
logger = logging.getLogger(__name__)
EDITOR_DETECT_OCR_CACHE_SCHEMA_VERSION = 7
STYLE_COPY_CANDIDATE_CONFIDENCE_THRESHOLD = SOURCE_STYLE_CONFIDENCE_THRESHOLD
STYLE_COPY_SFX_PROMOTION_THRESHOLD = 0.66
DARK_PANEL_RECT_MAX_HALF_WIDTH_FROM_TEXT_CENTER = 116
DARK_PANEL_RECT_MAX_HALF_HEIGHT_FROM_TEXT_CENTER = 64
SUPPRESSED_OCR_ROUTE_REASONS = {
    "english_ocr_gibberish_suppressed",
    "scanlator_text_caption_suppressed",
    "source_language_cjk_text_suppressed",
    "suppressed_duplicate_phrase_fragment",
    "visual_cjk_suppressed",
    "visual_sfx_overlap_suppressed",
}

REMOVED_AUTOMATIC_DECISION_FIELDS = {
    "tipo",
    "content_class",
    "balloon_type",
    "skip_processing",
    "preserve_original",
}


def _is_art_fragment_review_layer(layer: dict) -> bool:
    reason = str(layer.get("route_reason") or layer.get("skip_reason") or "").strip().lower()
    if reason in {"ocr_art_fragment_suspected", "sfx_art_fragment_suspected"}:
        return True
    return any(
        str(flag or "").strip().lower() in {"ocr_art_fragment_suspected", "sfx_art_fragment_suspected"}
        for flag in layer.get("qa_flags") or []
    )


def neutralize_removed_decision_fields(layer: dict) -> dict:
    """Keep removed legacy fields as neutral compatibility metadata only."""
    normalized = dict(layer or {})
    route_action = str(normalized.get("route_action") or "").strip().lower()
    render_policy = str(normalized.get("render_policy") or "").strip().lower()
    content_class = str(normalized.get("content_class") or "").strip().lower()
    if route_action == "translate_sfx_inpaint_render" or content_class == "sfx":
        normalized = enrich_sfx_candidate(normalized)
        normalized["skip_processing"] = False
        normalized["preserve_original"] = False
        return normalized
    if route_action == "merged_into_primary" or render_policy == "merged_into_primary":
        normalized["tipo"] = "text"
        normalized["content_class"] = "text"
        normalized["balloon_type"] = ""
        normalized["skip_processing"] = False
        normalized["preserve_original"] = False
        normalized["translate_policy"] = "translate"
        normalized["route_action"] = "merged_into_primary"
        normalized["render_policy"] = "merged_into_primary"
        normalized["visible"] = False
        return normalized
    if route_action == "review_required" and _is_art_fragment_review_layer(normalized):
        normalized["tipo"] = "text"
        normalized["content_class"] = "text"
        normalized["balloon_type"] = ""
        normalized["skip_processing"] = True
        normalized["preserve_original"] = True
        normalized["translate_policy"] = "skip_translation"
        normalized["route_action"] = "review_required"
        normalized["render_policy"] = "preserve_original"
        normalized["visible"] = False
        normalized["route_reason"] = normalized.get("route_reason") or "ocr_art_fragment_suspected"
        return normalized
    normalized["tipo"] = "text"
    normalized["content_class"] = "text"
    normalized["balloon_type"] = ""
    normalized["skip_processing"] = False
    normalized["preserve_original"] = False
    normalized["translate_policy"] = "translate"
    normalized["route_action"] = normalized.get("route_action") or "translate_inpaint_render"
    if str(normalized.get("route_action") or "").strip().lower() == "review_required":
        normalized["render_policy"] = "review_required"
    else:
        normalized["render_policy"] = "normal"
    return normalized


def _sfx_policy_or_default(layer: dict, key: str, default: str) -> str:
    if (
        str(layer.get("route_action") or "").strip().lower() == "translate_sfx_inpaint_render"
        or str(layer.get("content_class") or "").strip().lower() == "sfx"
    ):
        return layer.get(key) or default
    return default


def _promote_sfx_visual_candidates(
    page_result: dict,
    *,
    existing_texts: list[dict] | None = None,
    sfx_ocr_recognizer=None,
) -> list[dict]:
    candidates = [
        item for item in (page_result.get("_sfx_visual_candidates") or [])
        if isinstance(item, dict)
    ]
    if not candidates:
        return []
    existing_source = existing_texts if existing_texts is not None else page_result.get("texts")
    existing = [item for item in (existing_source or []) if isinstance(item, dict)]
    source_image_rgb = _page_result_source_image_rgb(page_result)
    promoted: list[dict] = []
    for candidate in candidates:
        bbox = _bbox4(candidate.get("bbox") or candidate.get("text_pixel_bbox"))
        if bbox is None:
            continue
        if any(_bbox_overlap_ratio(bbox, _bbox4(text.get("bbox") or text.get("text_pixel_bbox"))) >= 0.60 for text in existing):
            continue
        candidate = probe_sfx_candidate_ocr(candidate, source_image_rgb, recognizer=sfx_ocr_recognizer)
        recognized_text = str(candidate.get("recognized_text") or "")
        if recognized_text:
            prepared = probe_sfx_candidate_script(candidate, recognized_text)
        else:
            prepared = promote_visual_sfx_candidate(candidate, source_image_rgb)
        prepared["bbox"] = bbox
        prepared["text_pixel_bbox"] = bbox
        prepared["source_bbox"] = bbox
        prepared.setdefault("id", f"sfx_visual_{len(promoted) + 1:03d}")
        prepared.setdefault("text_id", prepared["id"])
        promoted.append(prepared)
    return promoted


def _page_result_source_image_rgb(page_result: dict):
    image_rgb = page_result.get("_cached_image_rgb")
    if hasattr(image_rgb, "shape") and getattr(image_rgb, "ndim", 0) == 3:
        return image_rgb
    image_bgr = page_result.get("_cached_image_bgr")
    if hasattr(image_bgr, "shape") and getattr(image_bgr, "ndim", 0) == 3:
        try:
            import cv2

            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
    return None


def _drop_suppressed_ocr_texts(
    texts: list[dict],
    source_language: str = "en",
    *,
    sfx_candidates: list[dict] | None = None,
) -> list[dict]:
    guarded = apply_language_guards(
        postprocess_ocr_fragments(texts, page_language=source_language),
        source_language=source_language,
    )
    guarded = suppress_normal_ocr_overlapping_sfx(
        guarded,
        sfx_candidates or [],
        source_language=source_language,
    )
    filtered: list[dict] = []
    for text in guarded:
        route_reason = str(text.get("route_reason") or "").strip().lower()
        flags = {
            str(flag).strip().lower()
            for flag in (text.get("qa_flags") or [])
            if str(flag).strip()
        }
        if str(text.get("route") or "").strip().lower() == "suppress":
            continue
        if route_reason in SUPPRESSED_OCR_ROUTE_REASONS:
            continue
        if flags & SUPPRESSED_OCR_ROUTE_REASONS:
            continue
        if bool(text.get("skip_processing")) and route_reason in SUPPRESSED_OCR_ROUTE_REASONS:
            continue
        filtered.append(text)
    return filtered


def _bbox_overlap_ratio(a: list[int] | None, b: list[int] | None) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / float(area)


def _dialogue_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", str(text or "")))


def _strip_inline_sfx_from_dialogue_text(text: str) -> tuple[str, str | None]:
    cleaned, sfx_word = split_sfx_inline(str(text or ""))
    if not sfx_word:
        return str(text or ""), None
    if _dialogue_word_count(cleaned) < 2:
        return str(text or ""), None
    return cleaned, sfx_word


def _remove_inline_sfx_noise_from_layer_texts(layer: dict) -> dict:
    normalized = dict(layer or {})
    source_text = str(
        normalized.get("text")
        or normalized.get("original")
        or normalized.get("raw_ocr")
        or normalized.get("normalized_ocr")
        or ""
    )
    cleaned_source, sfx_word = _strip_inline_sfx_from_dialogue_text(source_text)
    if not sfx_word:
        return normalized
    normalized["_inline_sfx_removed"] = sfx_word
    for key in ("text", "original", "raw_ocr", "normalized_ocr", "normalized_text_final", "source_text_sent_to_translator"):
        if str(normalized.get(key) or "").strip():
            cleaned, candidate_sfx = _strip_inline_sfx_from_dialogue_text(str(normalized.get(key) or ""))
            if candidate_sfx:
                normalized[key] = cleaned
    for key in ("translated", "traduzido"):
        if str(normalized.get(key) or "").strip():
            cleaned, candidate_sfx = _strip_inline_sfx_from_dialogue_text(str(normalized.get(key) or ""))
            if candidate_sfx:
                normalized[key] = cleaned
    return normalized


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
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return

    preferred_python = _select_local_venv_python(sys.executable, pipeline_root)
    if not preferred_python:
        return

    os.environ["TRADUZAI_SKIP_LOCAL_VENV_REEXEC"] = "1"
    script_path = str(Path(__file__).resolve())
    completed = subprocess.run([str(preferred_python), script_path, *sys.argv[1:]])
    sys.exit(completed.returncode)


def _report_emit_failure(exc: Exception) -> None:
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
        from debug_tools import event as debug_event

        debug_event("stdout", "emit", message_type=msg_type, payload=payload)
    except Exception:
        pass
    try:
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    except UnicodeEncodeError:
        try:
            print(json.dumps(payload, ensure_ascii=True), flush=True)
        except (OSError, UnicodeEncodeError) as exc:
            _report_emit_failure(exc)
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
        "engine_preset_id": "",
        "work_title_user_provided": False,
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
            parsed["work_title_user_provided"] = True
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
        if arg == "--engine-preset" and index + 1 < len(args):
            parsed["engine_preset_id"] = args[index + 1]
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
        parsed["work_title_user_provided"] = False
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
    high = sum(1 for issue in issues if issue.get("severity") == "high")
    export_gate = (project.get("qa") or {}).get("export_gate") or {}
    qa_report = {
        "summary": {
            "total": len(issues),
            "critical": critical,
            "high": high,
            "export_gate_status": export_gate.get("status", "PASS"),
            "critical_issue_count": export_gate.get("critical_issue_count", critical),
            "critical_flag_count": export_gate.get("critical_flag_count", critical),
            "review_issue_count": export_gate.get("review_issue_count", high),
        },
        "export_gate": export_gate,
        "needs_review": bool(project.get("needs_review")),
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


def _debug_e2e_enabled(config: dict) -> bool:
    env_value = os.getenv("TRADUZAI_DEBUG_E2E", "").strip().lower()
    return bool(config.get("debug")) or env_value in {"1", "true", "yes", "on"}


def _debug_level_from_env() -> str:
    return os.getenv("TRADUZAI_DEBUG_LEVEL", "standard").strip().lower() or "standard"


def _debug_env_snapshot() -> dict:
    prefixes = ("TRADUZAI_", "STRIP_", "PYTHON", "CUDA", "ORT_")
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes)
    }


def _bootstrap_debug_recorder(config: dict, config_path: str | Path):
    from debug_tools import DebugRecorder, bind_recorder
    from debug_tools.ids import make_run_id

    enabled = _debug_e2e_enabled(config)
    if not enabled:
        bind_recorder(None)
        return None

    work_dir = Path(config["work_dir"])
    run_id = str(config.get("run_id") or make_run_id(str(config.get("obra") or work_dir.name)))
    recorder = DebugRecorder(work_dir, enabled=True, run_id=run_id, level=_debug_level_from_env())
    bind_recorder(recorder)
    recorder.event("run", "debug_bootstrap", {"config_path": str(config_path)})
    recorder.write_json("00_run/config_snapshot.json", dict(config))
    recorder.write_json("00_run/env_snapshot.json", {"env_vars": _debug_env_snapshot()})
    recorder.write_json("00_run/pipeline_args.json", {"config_path": str(config_path), "argv": list(sys.argv)})
    try:
        loaded_config = _load_json_file(config_path)
        recorder.write_json("00_run/runner_config_snapshot.json", loaded_config)
    except Exception as exc:
        recorder.event("run", "runner_config_snapshot_failed", {"error": str(exc)})
    return recorder


def _finalize_debug_recorder(recorder, *, config_snapshot: dict | None = None, extra: dict | None = None) -> None:
    if not recorder:
        return
    try:
        recorder.finalize(config_snapshot=config_snapshot or {}, extra=extra or {})
    finally:
        try:
            from debug_tools import bind_recorder

            bind_recorder(None)
        except Exception:
            pass


def _strict_export_gate_active(config: dict) -> bool:
    return bool(config.get("strict")) or str(config.get("export_mode") or "").strip().lower() == "strict"


def _iter_project_text_layers(project_data: dict):
    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        layers = page.get("text_layers") or page.get("textos") or []
        for layer in layers:
            if isinstance(layer, dict):
                yield layer


def _ensure_project_mask_evidence(project_data: dict) -> int:
    """Ensure final project layers expose the single mask evidence contract."""
    try:
        from inpainter.mask_builder import consolidate_mask_evidence
    except Exception:
        return 0

    filled = 0
    for layer in _iter_project_text_layers(project_data):
        if isinstance(layer.get("mask_evidence"), dict):
            continue
        consolidate_mask_evidence(
            layer,
            kind="none",
            raw_mask_pixels=0,
            expanded_mask_pixels=0,
            evidence_score=0.0,
        )
        filled += 1
    filled += _clear_peer_covered_fast_fill_no_glyph_fragments(project_data)
    filled += _clear_non_bubble_panel_mask_flags(project_data)
    filled += _clear_stale_valid_image_bubble_mask_flags(project_data)
    filled += _clear_stale_panel_weak_residual_flags(project_data)
    return filled


def _mask_evidence_bbox4(value) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _is_placeholder_mask_bbox(value) -> bool:
    bbox = _mask_evidence_bbox4(value)
    if bbox is None:
        return False
    return [int(round(v)) for v in bbox] == [0, 0, 32, 32]


def _recover_source_text_mask_bbox(record: dict, text_pixel_bbox: list[int] | None) -> list[int] | None:
    bbox = _bbox4(record.get("source_text_mask_bbox") or record.get("_source_text_mask_bbox"))
    if bbox is not None and not _is_placeholder_mask_bbox(bbox):
        return bbox
    if text_pixel_bbox is None:
        return None
    flags = {str(flag).strip() for flag in record.get("qa_flags") or [] if str(flag).strip()}
    evidence = record.get("mask_evidence") if isinstance(record.get("mask_evidence"), dict) else {}
    try:
        evidence_pixels = int(evidence.get("raw_mask_pixels") or evidence.get("expanded_mask_pixels") or 0)
    except (TypeError, ValueError):
        evidence_pixels = 0
    route_action = str(record.get("route_action") or "").strip().lower()
    bubble_source = str(record.get("bubble_mask_source") or "").strip().lower()
    if (
        evidence_pixels > 0
        or route_action == "translate_inpaint_render"
        or bubble_source in {"image_dark_panel_mask", "image_dark_bubble_mask", "translator_note_text_mask"}
        or flags
        & {
            "visual_text_only_inpaint_contract",
            "text_contract_direct_fill",
            "translator_note_text_only_mask",
            "dark_bubble_visual_glyph_mask_replaced_geometry",
        }
    ):
        return [int(v) for v in text_pixel_bbox]
    return None


def _mask_evidence_bbox_overlap_ratio(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def _polygon_bbox4(polygon) -> list[float] | None:
    if not isinstance(polygon, (list, tuple)):
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except (TypeError, ValueError):
            continue
    if not xs or not ys:
        return None
    return _mask_evidence_bbox4([min(xs), min(ys), max(xs), max(ys)])


def _layer_has_valid_mask_evidence(layer: dict) -> bool:
    evidence = layer.get("mask_evidence") if isinstance(layer.get("mask_evidence"), dict) else {}
    try:
        raw_pixels = int(evidence.get("raw_mask_pixels") or 0)
        expanded_pixels = int(evidence.get("expanded_mask_pixels") or 0)
        score = float(evidence.get("evidence_score") or 0.0)
    except Exception:
        return False
    kind = str(evidence.get("kind") or "").strip().lower()
    return bool(kind not in {"", "none"} and raw_pixels > 0 and expanded_pixels >= raw_pixels and score >= 0.75)


def _layer_background_is_non_bubble_panel(layer: dict) -> bool:
    rgb = layer.get("background_rgb")
    if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
        return False
    try:
        channels = [float(value) for value in rgb[:3]]
    except (TypeError, ValueError):
        return False
    luma = sum(channels) / 3.0
    chroma = max(channels) - min(channels)
    return bool(luma < 225.0 or chroma > 32.0)


def _layer_is_non_balloon_text_annotation(layer: dict) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            layer.get("text"),
            layer.get("original"),
            layer.get("normalized_ocr"),
            layer.get("translated"),
            layer.get("context_before"),
            layer.get("context_after"),
        )
    ).strip().lower()
    flags = {str(flag).strip() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
    if "ocr_run_on_suspect" in flags:
        return True
    if text.startswith("t/n:") or " t/n:" in text:
        return True
    annotation_markers = (
        "patreon.com",
        "ko-fi.com",
        "discord",
        "readfirst",
        "secret scans",
        "secretscans",
    )
    return any(marker in text for marker in annotation_markers)


def _clear_non_bubble_panel_mask_flags(project_data: dict) -> int:
    cleared = 0
    removable = {
        "bbox_fallback_bubble_mask",
        "missing_real_bubble_mask",
        "rejected_derived_bubble_mask",
        "mask_outside_balloon",
        "mask_outside_balloon_critical",
    }
    for layer in _iter_project_text_layers(project_data):
        flags = [str(flag) for flag in layer.get("qa_flags") or [] if str(flag).strip()]
        is_annotation = _layer_is_non_balloon_text_annotation(layer)
        removable_for_layer = set(removable)
        if (
            is_annotation
            and _optional_bbox4(layer.get("render_bbox")) is not None
            and _optional_bbox4(layer.get("safe_text_box")) is not None
        ):
            removable_for_layer.add("fit_below_minimum_legible")
        if not flags or not (set(flags) & removable_for_layer):
            continue
        if not _layer_has_valid_mask_evidence(layer):
            continue
        if not (_layer_background_is_non_bubble_panel(layer) or is_annotation):
            continue
        source = str(layer.get("bubble_mask_source") or layer.get("bubbleMaskSource") or "").strip().lower()
        if source not in {"derived_card_panel_mask", "image_dark_panel_mask", "image_dark_bubble_mask", "derived_white_crop_rejected", "rejected_derived_bubble_mask", ""}:
            if not (
                is_annotation
                and source in {"image_white_bubble_mask", "image_contour_bubble_mask"}
            ):
                continue
        kept = [flag for flag in flags if flag not in removable_for_layer]
        if kept != flags:
            layer["qa_flags"] = kept
            cleared += 1
        reason = str(layer.get("route_reason") or "").strip()
        if reason in removable:
            layer.pop("route_reason", None)
            if layer.get("translated") or layer.get("texto_traduzido"):
                layer["route_action"] = "translate_inpaint_render"
                if str(layer.get("render_policy") or "").strip().lower() == "review_required":
                    layer.pop("render_policy", None)
                layer.pop("needs_review", None)
    return cleared


def _clear_stale_panel_weak_residual_flags(project_data: dict) -> int:
    cleared = 0
    removable = "weak_text_residual_after_inpaint"
    image_bubble_sources = {
        "image_white_bubble_mask",
        "image_dark_panel_mask",
        "image_dark_bubble_mask",
        "image_contour_bubble_mask",
        "derived_white_crop",
        "derived_white_connected_component",
    }
    for layer in _iter_project_text_layers(project_data):
        flags = [str(flag) for flag in layer.get("qa_flags") or [] if str(flag).strip()]
        if removable not in flags:
            continue
        if not _layer_has_valid_mask_evidence(layer):
            continue
        source = str(layer.get("bubble_mask_source") or layer.get("bubbleMaskSource") or "").strip().lower()
        if source in {"derived_card_panel_mask", "image_dark_panel_mask", "image_dark_bubble_mask"}:
            if not _layer_background_is_non_bubble_panel(layer):
                continue
        elif source not in image_bubble_sources:
            continue
        residual = layer.get("residual_text")
        if isinstance(residual, dict) and residual.get("has_residual"):
            continue
        kept = [flag for flag in flags if flag != removable]
        layer["qa_flags"] = kept
        cleared += 1
    return cleared


def _clear_stale_valid_image_bubble_mask_flags(project_data: dict) -> int:
    cleared = 0
    removable = {
        "bbox_fallback_bubble_mask",
        "debug_derived_bubble_mask_rejected",
    }
    image_bubble_sources = {
        "image_white_bubble_mask",
        "image_dark_panel_mask",
        "image_dark_bubble_mask",
        "image_rect_bubble_mask",
        "image_contour_bubble_mask",
    }
    for layer in _iter_project_text_layers(project_data):
        flags = [str(flag) for flag in layer.get("qa_flags") or [] if str(flag).strip()]
        if not flags or not (set(flags) & removable):
            continue
        source = str(layer.get("bubble_mask_source") or layer.get("bubbleMaskSource") or "").strip().lower()
        if source not in image_bubble_sources:
            continue
        if str(layer.get("bubble_mask_error") or "").strip():
            continue
        if not _layer_has_valid_mask_evidence(layer):
            continue
        residual = layer.get("residual_text")
        if isinstance(residual, dict) and residual.get("has_residual"):
            continue
        kept = [flag for flag in flags if flag not in removable]
        if kept != flags:
            layer["qa_flags"] = kept
            cleared += 1
        reason = str(layer.get("route_reason") or "").strip()
        if reason in removable:
            layer.pop("route_reason", None)
            if layer.get("translated") or layer.get("texto_traduzido"):
                layer["route_action"] = "translate_inpaint_render"
                if str(layer.get("render_policy") or "").strip().lower() == "review_required":
                    layer.pop("render_policy", None)
                layer.pop("needs_review", None)
    return cleared


def _fragment_bbox_is_covered_by_peer_mask(fragment: dict, peer: dict) -> bool:
    fragment_bbox = _mask_evidence_bbox4(fragment.get("text_pixel_bbox")) or _mask_evidence_bbox4(fragment.get("bbox"))
    if fragment_bbox is None:
        return False
    peer_bbox = _mask_evidence_bbox4(peer.get("text_pixel_bbox")) or _mask_evidence_bbox4(peer.get("bbox"))
    if peer_bbox is not None and _mask_evidence_bbox_overlap_ratio(fragment_bbox, peer_bbox) >= 0.70:
        return True
    for polygon in peer.get("line_polygons") or []:
        poly_bbox = _polygon_bbox4(polygon)
        if poly_bbox is not None and _mask_evidence_bbox_overlap_ratio(fragment_bbox, poly_bbox) >= 0.70:
            return True
    return False


def _clear_peer_covered_fast_fill_no_glyph_fragments(project_data: dict) -> int:
    cleared = 0
    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        layers = [layer for layer in page.get("text_layers") or page.get("textos") or [] if isinstance(layer, dict)]
        by_band: dict[str, list[dict]] = {}
        for layer in layers:
            band_id = str(layer.get("band_id") or "").strip()
            if band_id:
                by_band.setdefault(band_id, []).append(layer)
        for band_layers in by_band.values():
            peers = [layer for layer in band_layers if _layer_has_valid_mask_evidence(layer)]
            if not peers:
                continue
            for layer in band_layers:
                flags = [str(flag) for flag in layer.get("qa_flags") or [] if str(flag).strip()]
                if "fast_fill_no_glyph_evidence" not in flags:
                    continue
                evidence = layer.get("mask_evidence") if isinstance(layer.get("mask_evidence"), dict) else {}
                if str(evidence.get("kind") or "").strip().lower() not in {"", "none"}:
                    continue
                if not any(peer is not layer and _fragment_bbox_is_covered_by_peer_mask(layer, peer) for peer in peers):
                    continue
                layer["qa_flags"] = [flag for flag in flags if flag != "fast_fill_no_glyph_evidence"]
                layer["mask_evidence"] = {
                    "kind": "covered_by_peer_mask",
                    "raw_mask_pixels": 0,
                    "expanded_mask_pixels": 0,
                    "evidence_score": 1.0,
                    "fast_fill_allowed": False,
                    "fast_fill_reject_reasons": ["covered_by_peer_mask"],
                }
                cleared += 1
    return cleared


def _translated_text_for_merge(layer: dict) -> str:
    for key in ("translated", "traduzido", "text"):
        value = str(layer.get(key) or "").strip()
        if value:
            return value
    return ""


def _unique_preserve_order(values) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _layer_merge_order_bbox(layer: dict) -> list[int] | None:
    for key in ("text_pixel_bbox", "layout_bbox", "bbox", "source_bbox", "render_bbox"):
        bbox = _optional_bbox4(layer.get(key))
        if bbox is not None:
            return bbox
    return None


def _layer_can_absorb_fragment(primary: dict, fragment: dict) -> bool:
    if primary is fragment:
        return False
    if str(primary.get("band_id") or "").strip() != str(fragment.get("band_id") or "").strip():
        return False
    if _layers_are_distinct_dark_bubble_lobes(primary, fragment):
        return False
    if primary.get("visible") is False or str(primary.get("render_policy") or "") == "merged_into_primary":
        return False
    primary_text = _translated_text_for_merge(primary)
    fragment_text = _translated_text_for_merge(fragment)
    if not primary_text or not fragment_text:
        return False
    fragment_bbox = _layer_merge_order_bbox(fragment)
    if fragment_bbox is None:
        return False
    candidates = [
        _optional_bbox4(primary.get("target_bbox")),
        _optional_bbox4(primary.get("balloon_bbox")),
        _optional_bbox4(primary.get("bubble_mask_bbox")),
        _optional_bbox4(primary.get("safe_text_box")),
    ]
    candidates = [bbox for bbox in candidates if bbox is not None]
    if not candidates:
        return False
    fragment_area = max(1, _bbox_area4(fragment_bbox))
    for candidate in candidates:
        if _bbox_contains4_margin(candidate, fragment_bbox, margin=24):
            return True
        if _bbox_intersection_area4(candidate, fragment_bbox) / float(fragment_area) >= 0.55:
            return True
    return False


def _merge_primary_preference_score(layer: dict, reference_bbox: list[int] | None = None) -> tuple[int, int, int, int, int, int]:
    flags = {str(flag) for flag in layer.get("qa_flags") or []}
    visual_bbox = _visual_target_bbox_for_merge(layer)
    order_bbox = _layer_merge_order_bbox(layer) or visual_bbox
    top = int(order_bbox[1]) if order_bbox is not None else 10**9
    left = int(order_bbox[0]) if order_bbox is not None else 10**9
    source_area = _layer_source_area_for_merge(layer)
    visual_area = _bbox_area4(visual_bbox) if visual_bbox is not None else 0
    intersection = _bbox_intersection_area4(visual_bbox, reference_bbox) if visual_bbox is not None and reference_bbox is not None else 0
    return (
        0 if "bbox_fallback_bubble_mask" in flags else 1,
        0 if flags.intersection({"rejected_derived_bubble_mask", "debug_derived_bubble_mask_rejected"}) else 1,
        -top,
        -left,
        source_area,
        visual_area,
        intersection,
    )


def _is_low_containment_suppressed_fragment(layer: dict) -> bool:
    flags = {str(flag).strip() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
    return bool(
        "render_suppressed_low_containment_fragment" in flags
        or str(layer.get("render_policy") or "").strip().lower() == "suppressed_low_containment_fragment"
        or str(layer.get("_render_review_reason") or "").strip().lower() == "low_containment_fragment"
    )


def _clear_low_containment_suppression_markers(layer: dict) -> None:
    if isinstance(layer.get("qa_flags"), list):
        layer["qa_flags"] = [
            flag
            for flag in layer.get("qa_flags") or []
            if str(flag).strip() != "render_suppressed_low_containment_fragment"
        ]
    if str(layer.get("render_policy") or "").strip().lower() == "suppressed_low_containment_fragment":
        layer["render_policy"] = "normal"
    if str(layer.get("_render_review_reason") or "").strip().lower() == "low_containment_fragment":
        layer.pop("_render_review_reason", None)


def _merge_same_balloon_fragment_layers(project_data: dict) -> int:
    """Merge late OCR fragments that target the same rendered balloon."""

    merged = 0
    global_by_band: dict[str, list[dict]] = {}
    for layer in _iter_project_text_layers(project_data):
        if not isinstance(layer, dict):
            continue
        trace_id = str(layer.get("trace_id") or "").strip()
        band_id = str(layer.get("band_id") or "").strip() or (_trace_band_id(trace_id) or "")
        if band_id:
            global_by_band.setdefault(band_id, []).append(layer)

    for band_layers in global_by_band.values():
        for layer in band_layers:
            if layer.get("visible", True) is not False:
                _remove_other_bubble_fragment_suffix(layer, band_layers)
        candidates = [
            layer
            for layer in band_layers
            for layer_source_token_count in [
                len(
                    {
                        str(value or "").strip()
                        for field in ("source_trace_ids", "_source_trace_ids", "source_text_ids", "_source_text_ids")
                        for value in (layer.get(field) or [])
                        if str(value or "").strip()
                    }
                )
            ]
            if layer.get("visible", True) is not False
            and str(layer.get("render_policy") or "") != "merged_into_primary"
            and not _is_low_containment_suppressed_fragment(layer)
            and (
                "same_balloon_fragment_merged" in {str(flag) for flag in layer.get("qa_flags") or []}
                or layer_source_token_count > 1
            )
            and (
                len([value for value in layer.get("source_trace_ids") or layer.get("_source_trace_ids") or [] if str(value).strip()]) > 1
                or len([value for value in layer.get("source_text_ids") or layer.get("_source_text_ids") or [] if str(value).strip()]) > 1
            )
        ]
        for candidate in candidates:
            source_tokens = set()
            for field in ("source_trace_ids", "_source_trace_ids", "source_text_ids", "_source_text_ids"):
                for value in candidate.get(field) or []:
                    token = str(value or "").strip()
                    if token:
                        source_tokens.add(token)
            if len(source_tokens) < 2:
                continue
            matched_layers: list[dict] = []
            for layer in band_layers:
                identities = {
                    str(layer.get("trace_id") or "").strip(),
                    str(layer.get("text_id") or "").strip(),
                    str(layer.get("id") or "").strip(),
                }
                identities = {value for value in identities if value}
                if layer is candidate or identities.intersection(source_tokens):
                    matched_layers.append(layer)
            visible_matches = [
                layer
                for layer in matched_layers
                if layer.get("visible", True) is not False
                and str(layer.get("render_policy") or "") != "merged_into_primary"
                and not _is_low_containment_suppressed_fragment(layer)
            ]
            if len(visible_matches) < 2 and candidate in visible_matches and len(matched_layers) >= 2:
                if any(
                    _layers_are_distinct_dark_bubble_lobes(left, right)
                    for idx, left in enumerate(matched_layers)
                    for right in matched_layers[idx + 1 :]
                ):
                    continue
                ordered_texts: list[str] = []
                ordered_norms: list[str] = []
                text_layers = [
                    layer
                    for layer in matched_layers
                    if _translated_text_for_merge(layer)
                    and not _is_low_containment_suppressed_fragment(layer)
                    and (
                        layer is candidate
                        or (
                            str(layer.get("render_policy") or "").strip().lower() != "merged_into_primary"
                            and str(layer.get("route_action") or "").strip().lower() != "merged_into_primary"
                        )
                    )
                ]
                for ordered_layer in sorted(
                    text_layers,
                    key=lambda item: (
                        (_layer_merge_order_bbox(item) or [10**9, 10**9, 10**9, 10**9])[1],
                        (_layer_merge_order_bbox(item) or [10**9, 10**9, 10**9, 10**9])[0],
                    ),
                ):
                    part = str(_translated_text_for_merge(ordered_layer) or "").strip()
                    normalized_part = _normalized_merge_text(part)
                    if not normalized_part:
                        continue
                    contained_by_existing = any(normalized_part in existing for existing in ordered_norms)
                    if contained_by_existing:
                        continue
                    replaced = False
                    for index, existing in enumerate(list(ordered_norms)):
                        if existing in normalized_part:
                            ordered_texts[index] = part
                            ordered_norms[index] = normalized_part
                            replaced = True
                            break
                    if not replaced:
                        ordered_texts.append(part)
                        ordered_norms.append(normalized_part)
                if len(ordered_texts) >= 2:
                    candidate["translated"] = "\n".join(ordered_texts)
                    candidate["traduzido"] = candidate["translated"]
                    merged_trace_ids = _unique_preserve_order(
                        [
                            *(candidate.get("source_trace_ids") or candidate.get("_source_trace_ids") or []),
                            *[
                                str(layer.get("trace_id") or "").strip()
                                for layer in matched_layers
                                if str(layer.get("trace_id") or "").strip()
                            ],
                        ]
                    )
                    merged_text_ids = _unique_preserve_order(
                        [
                            *(candidate.get("source_text_ids") or candidate.get("_source_text_ids") or []),
                            *[
                                str(layer.get("text_id") or layer.get("id") or "").strip()
                                for layer in matched_layers
                                if str(layer.get("text_id") or layer.get("id") or "").strip()
                            ],
                        ]
                    )
                    if merged_trace_ids:
                        candidate["source_trace_ids"] = merged_trace_ids
                        candidate["_source_trace_ids"] = list(merged_trace_ids)
                    if merged_text_ids:
                        candidate["source_text_ids"] = merged_text_ids
                        candidate["_source_text_ids"] = list(merged_text_ids)
                    _merge_layer_qa_flags(candidate, ["same_balloon_fragment_merged"])
                    primary_trace = str(candidate.get("trace_id") or candidate.get("id") or "").strip()
                    for layer in matched_layers:
                        if layer is candidate:
                            continue
                        layer["visible"] = False
                        layer["render_policy"] = "merged_into_primary"
                        layer["merged_into_trace_id"] = primary_trace
                        layer["merged_into_text_id"] = candidate.get("text_id") or candidate.get("id")
                    merged += 1
                continue
            if len(visible_matches) < 2:
                continue
            if any(
                _layers_are_distinct_dark_bubble_lobes(left, right)
                for idx, left in enumerate(visible_matches)
                for right in visible_matches[idx + 1 :]
            ):
                continue
            if (
                not _layers_share_visual_merge_target(visible_matches)
                and not _source_linked_layers_have_text_overlap(visible_matches)
                and not _layers_look_like_cross_page_band_siblings(visible_matches)
            ):
                continue
            if _layers_have_conflicting_explicit_bubble_ids(visible_matches) and not _source_linked_layers_have_text_overlap(
                visible_matches
            ):
                primary_for_cleanup = candidate if candidate in visible_matches else max(
                    visible_matches,
                    key=lambda item: _merge_primary_preference_score(item, _layer_merge_order_bbox(candidate)),
                )
                _remove_other_bubble_fragment_suffix(primary_for_cleanup, visible_matches)
                merged += _suppress_rejected_other_bubble_fragments(primary_for_cleanup, visible_matches)
                continue

            candidate_reference_bbox = (
                _optional_bbox4(candidate.get("target_bbox"))
                or _optional_bbox4(candidate.get("balloon_bbox"))
                or _optional_bbox4(candidate.get("bubble_mask_bbox"))
                or _optional_bbox4(candidate.get("render_bbox"))
                or _layer_merge_order_bbox(candidate)
            )
            best_text_layer = max(
                visible_matches,
                key=lambda layer: (
                    len(_normalized_merge_text(layer.get("translated") or layer.get("traduzido") or "")),
                    1 if layer is candidate else 0,
                ),
            )
            source_texts_overlap = _source_linked_layers_have_text_overlap(visible_matches)
            if source_texts_overlap:
                primary = max(
                    visible_matches,
                    key=lambda layer: _merge_primary_preference_score(layer, candidate_reference_bbox),
                )
            else:
                primary = visible_matches[0]
            best_text_len = len(_normalized_merge_text(best_text_layer.get("translated") or best_text_layer.get("traduzido") or ""))
            primary_text_len = len(_normalized_merge_text(primary.get("translated") or primary.get("traduzido") or ""))
            if (
                best_text_layer is candidate
                and "same_balloon_fragment_merged" in {str(flag) for flag in candidate.get("qa_flags") or []}
                and source_texts_overlap
                and best_text_len > max(primary_text_len + 12, int(primary_text_len * 1.25))
            ):
                primary = best_text_layer
            ordered_texts: list[str] = []
            for ordered_layer in sorted(
                visible_matches,
                key=lambda item: (
                    (_layer_merge_order_bbox(item) or [10**9, 10**9, 10**9, 10**9])[1],
                    (_layer_merge_order_bbox(item) or [10**9, 10**9, 10**9, 10**9])[0],
                ),
            ):
                part = _translated_text_for_merge(ordered_layer)
                normalized_part = " ".join(part.split())
                if normalized_part and normalized_part not in ordered_texts:
                    ordered_texts.append(normalized_part)
            best_text = _translated_text_for_merge(best_text_layer)
            best_text_norm_for_parts = _normalized_merge_text(best_text)
            if len(ordered_texts) >= 2 and not all(
                _normalized_merge_text(part) in best_text_norm_for_parts for part in ordered_texts
            ):
                best_text = "\n".join(ordered_texts)
            if best_text:
                primary["translated"] = best_text
                primary["traduzido"] = best_text
                if not primary.get("text"):
                    primary["text"] = best_text
            merged_trace_ids = _unique_preserve_order(
                [
                    *(primary.get("source_trace_ids") or primary.get("_source_trace_ids") or []),
                    *(candidate.get("source_trace_ids") or candidate.get("_source_trace_ids") or []),
                ]
            )
            merged_text_ids = _unique_preserve_order(
                [
                    *(primary.get("source_text_ids") or primary.get("_source_text_ids") or []),
                    *(candidate.get("source_text_ids") or candidate.get("_source_text_ids") or []),
                ]
            )
            if merged_trace_ids:
                primary["source_trace_ids"] = merged_trace_ids
                primary["_source_trace_ids"] = list(merged_trace_ids)
            if merged_text_ids:
                primary["source_text_ids"] = merged_text_ids
                primary["_source_text_ids"] = list(merged_text_ids)
            _merge_layer_qa_flags(primary, ["same_balloon_fragment_merged"])
            primary_trace = str(primary.get("trace_id") or primary.get("id") or "").strip()
            for layer in visible_matches:
                if layer is primary:
                    continue
                layer["visible"] = False
                layer["render_policy"] = "merged_into_primary"
                layer["route_action"] = "merged_into_primary"
                layer["merged_into_trace_id"] = primary_trace
                layer["merged_into_text_id"] = primary.get("text_id") or primary.get("id")
                if isinstance(layer.get("qa_flags"), list):
                    layer["qa_flags"] = [
                        flag
                        for flag in layer.get("qa_flags") or []
                        if str(flag) not in {"missing_render_bbox", "fit_below_minimum_legible"}
                    ]
                merged += 1

    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        by_band: dict[str, list[dict]] = {}
        for layer in _project_page_text_layers(page):
            if not isinstance(layer, dict):
                continue
            trace_id = str(layer.get("trace_id") or "").strip()
            band_id = str(layer.get("band_id") or "").strip() or (_trace_band_id(trace_id) or "")
            if band_id:
                by_band.setdefault(band_id, []).append(layer)
        for band_layers in by_band.values():
            fragments = [
                layer
                for layer in band_layers
                if "_fragment_" in str(layer.get("id") or layer.get("text_id") or layer.get("trace_id") or "")
                and str(layer.get("render_policy") or "") != "merged_into_primary"
                and not _is_low_containment_suppressed_fragment(layer)
                and str(layer.get("route_action") or "").startswith("translate_")
                and _translated_text_for_merge(layer)
            ]
            if not fragments:
                continue
            primaries = [
                layer
                for layer in band_layers
                if layer not in fragments
                and str(layer.get("route_action") or "").startswith("translate_")
                and layer.get("visible") is not False
                and str(layer.get("render_policy") or "") != "merged_into_primary"
            ]
            for fragment in fragments:
                if _dark_connected_fragment_covers_multiple_lobes(fragment, primaries):
                    fragment["visible"] = False
                    fragment["render_policy"] = "suppressed_dark_connected_combined_fragment"
                    fragment["route_action"] = "suppressed_dark_connected_combined_fragment"
                    _merge_layer_qa_flags(fragment, ["dark_connected_combined_fragment_suppressed"])
                    continue
                matches = [primary for primary in primaries if _layer_can_absorb_fragment(primary, fragment)]
                if not matches:
                    continue
                fragment_bbox = _layer_merge_order_bbox(fragment)
                if fragment_bbox is None:
                    continue
                primary = max(matches, key=lambda item: _merge_primary_preference_score(item, fragment_bbox))
                parts: list[tuple[list[int], str]] = []
                for item in (fragment, primary):
                    bbox = _layer_merge_order_bbox(item)
                    text = _translated_text_for_merge(item)
                    if bbox is not None and text:
                        parts.append((bbox, text))
                if len(parts) < 2:
                    continue
                ordered_texts: list[str] = []
                for _bbox, text in sorted(parts, key=lambda entry: (entry[0][1], entry[0][0])):
                    normalized = " ".join(text.split())
                    if normalized and normalized not in ordered_texts:
                        ordered_texts.append(normalized)
                if len(ordered_texts) < 2:
                    continue
                merged_text = "\n".join(ordered_texts)
                primary["translated"] = merged_text
                primary["traduzido"] = merged_text
                primary["text"] = primary.get("text") or merged_text
                primary["_merged_same_balloon_fragments"] = list(primary.get("_merged_same_balloon_fragments") or []) + [
                    fragment.get("id") or fragment.get("text_id") or fragment.get("trace_id")
                ]
                _merge_layer_qa_flags(primary, ["same_balloon_fragment_merged"])
                fragment["visible"] = False
                fragment["render_policy"] = "merged_into_primary"
                fragment["route_action"] = "merged_into_primary"
                fragment["_merged_into_text_id"] = primary.get("id") or primary.get("text_id")
                if isinstance(fragment.get("qa_flags"), list):
                    fragment["qa_flags"] = [
                        flag for flag in fragment.get("qa_flags") or [] if str(flag) not in {"missing_render_bbox", "fit_below_minimum_legible"}
                    ]
                merged += 1
    merged += _dedupe_repeated_project_text_prefixes(list(_iter_project_text_layers(project_data)))
    return merged


def _fragment_base_identity(layer: dict) -> set[str]:
    identities: set[str] = set()
    for key in ("id", "text_id"):
        value = str(layer.get(key) or "").strip()
        if value:
            identities.add(value.rsplit("_fragment_", 1)[0].strip())
    for key in ("source_text_ids", "_source_text_ids"):
        for value in layer.get(key) or []:
            text_id = str(value or "").strip()
            if text_id:
                identities.add(text_id.rsplit("_fragment_", 1)[0].strip())
    trace_id = str(layer.get("trace_id") or "").strip()
    if trace_id:
        base_trace = trace_id.split("#", 1)[0].strip()
        identities.add(base_trace)
        if "@" in base_trace:
            identities.add(base_trace.split("@", 1)[0].strip())
    for key in ("source_trace_ids", "_source_trace_ids"):
        for value in layer.get(key) or []:
            source_trace = str(value or "").strip()
            if not source_trace:
                continue
            base_trace = source_trace.split("#", 1)[0].strip()
            identities.add(base_trace)
            if "@" in base_trace:
                identities.add(base_trace.split("@", 1)[0].strip())
    return {value for value in identities if value}


def _is_fragment_layer_id(layer: dict) -> bool:
    return any(
        "_fragment_" in str(layer.get(key) or "")
        for key in ("id", "text_id", "trace_id")
    )


def _suppress_same_identity_merged_fragments(project_data: dict) -> int:
    """Hide restored split fragments already covered by a merged primary layer."""

    suppressed = 0
    if not isinstance(project_data, dict):
        return suppressed
    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        by_band: dict[str, list[dict]] = {}
        for layer in _project_page_text_layers(page):
            if not isinstance(layer, dict):
                continue
            band_id = str(layer.get("band_id") or "").strip()
            if band_id:
                by_band.setdefault(band_id, []).append(layer)
        for band_layers in by_band.values():
            primaries = [
                layer
                for layer in band_layers
                if not _is_fragment_layer_id(layer)
                and layer.get("visible", True) is not False
                and str(layer.get("route_action") or "").strip() != "merged_into_primary"
                and str(layer.get("render_policy") or "").strip() != "merged_into_primary"
                and "same_balloon_fragment_merged" in {str(flag) for flag in layer.get("qa_flags") or []}
                and _translated_text_for_merge(layer)
            ]
            if not primaries:
                continue
            for fragment in band_layers:
                if not _is_fragment_layer_id(fragment):
                    continue
                if fragment.get("visible", True) is False:
                    continue
                if str(fragment.get("route_action") or "").strip() == "merged_into_primary":
                    continue
                fragment_text = _normalized_merge_text(_translated_text_for_merge(fragment))
                if not fragment_text:
                    continue
                fragment_ids = _fragment_base_identity(fragment)
                for primary in primaries:
                    primary_text = _normalized_merge_text(_translated_text_for_merge(primary))
                    if fragment_text not in primary_text:
                        continue
                    if fragment_ids and not fragment_ids.intersection(_fragment_base_identity(primary)):
                        continue
                    fragment["visible"] = False
                    fragment["render_policy"] = "merged_into_primary"
                    fragment["route_action"] = "merged_into_primary"
                    fragment["merged_into_trace_id"] = primary.get("trace_id") or primary.get("id")
                    fragment["merged_into_text_id"] = primary.get("text_id") or primary.get("id")
                    _merge_layer_qa_flags(fragment, ["same_balloon_fragment_merged"])
                    suppressed += 1
                    break
    return suppressed


def _restore_hidden_distinct_nonfragment_layers(project_data: dict) -> int:
    """Restore non-fragment layers incorrectly hidden behind a different visual target."""

    restored = 0
    if not isinstance(project_data, dict):
        return restored
    layers = [layer for layer in _iter_project_text_layers(project_data) if isinstance(layer, dict)]
    visible_layers = [
        layer
        for layer in layers
        if layer.get("visible", True) is not False
        and str(layer.get("route_action") or "").strip() != "merged_into_primary"
        and str(layer.get("render_policy") or "").strip() != "merged_into_primary"
    ]
    visible_by_identity = _layers_by_debug_identity(visible_layers)
    for layer in layers:
        if _is_fragment_layer_id(layer):
            continue
        if layer.get("visible", True) is not False:
            continue
        if str(layer.get("route_action") or "").strip() != "merged_into_primary":
            continue
        text_norm = _normalized_merge_text(_translated_text_for_merge(layer))
        if not text_norm:
            continue
        if any(text_norm in _normalized_merge_text(_translated_text_for_merge(peer)) for peer in visible_layers):
            continue
        merged_into = str(layer.get("merged_into_trace_id") or layer.get("merged_into_text_id") or "").strip()
        target_layers = visible_by_identity.get(merged_into, []) if merged_into else []
        if target_layers and _layers_share_visual_merge_target([layer, *target_layers]):
            continue
        layer["visible"] = True
        layer["route_action"] = "translate_inpaint_render"
        if str(layer.get("render_policy") or "").strip() == "merged_into_primary":
            layer["render_policy"] = "normal"
        layer.pop("merged_into_trace_id", None)
        layer.pop("merged_into_text_id", None)
        _merge_layer_qa_flags(layer, ["restored_distinct_hidden_nonfragment"])
        visible_layers.append(layer)
        for identity_key in _debug_identity_keys_from_payload(layer):
            visible_by_identity.setdefault(identity_key, []).append(layer)
        restored += 1
    return restored


def _page_id_from_page_number(page: dict) -> str | None:
    try:
        number = int(page.get("numero"))
    except Exception:
        return None
    if number <= 0:
        return None
    return f"page_{number:03d}"


def _rehome_cross_page_band_layers(project_data: dict) -> int:
    """Move text payload from layers restored onto the wrong page to the page owning their band."""

    if not isinstance(project_data, dict):
        return 0
    pages = [page for page in project_data.get("paginas") or [] if isinstance(page, dict)]
    pages_by_id = {
        page_id: page
        for page in pages
        for page_id in [_page_id_from_page_number(page)]
        if page_id
    }
    moved = 0
    for page in pages:
        current_page_id = _page_id_from_page_number(page)
        if not current_page_id:
            continue
        for layer in _project_page_text_layers(page):
            if not isinstance(layer, dict):
                continue
            band_id = str(layer.get("band_id") or "").strip()
            band_page_id = _page_id_from_band_id(band_id)
            if not band_page_id or band_page_id == current_page_id:
                continue
            destination = pages_by_id.get(band_page_id)
            if destination is None or destination is page:
                continue
            destination_layers = [
                candidate
                for candidate in _project_page_text_layers(destination)
                if str(candidate.get("band_id") or "").strip() == band_id
                and str(candidate.get("render_policy") or "") != "merged_into_primary"
            ]
            if not destination_layers:
                continue
            primary = max(
                destination_layers,
                key=lambda candidate: (
                    1 if candidate.get("visible", True) is not False else 0,
                    len(_normalized_merge_text(_translated_text_for_merge(candidate))),
                    _bbox_area4(_optional_bbox4(candidate.get("safe_text_box")) or _optional_bbox4(candidate.get("render_bbox")) or [0, 0, 0, 0]),
                ),
            )
            incoming_text = _translated_text_for_merge(layer)
            primary_text = _translated_text_for_merge(primary)
            if incoming_text and (
                not primary_text
                or _normalized_merge_text(primary_text) in _normalized_merge_text(incoming_text)
                or len(_normalized_merge_text(incoming_text)) > len(_normalized_merge_text(primary_text))
            ):
                primary["translated"] = incoming_text
                primary["traduzido"] = incoming_text
                if not primary.get("text"):
                    primary["text"] = incoming_text
            merged_trace_ids = _unique_preserve_order(
                [
                    *(primary.get("source_trace_ids") or primary.get("_source_trace_ids") or []),
                    *(layer.get("source_trace_ids") or layer.get("_source_trace_ids") or []),
                    str(primary.get("trace_id") or "").strip(),
                    str(layer.get("trace_id") or "").strip(),
                ]
            )
            merged_text_ids = _unique_preserve_order(
                [
                    *(primary.get("source_text_ids") or primary.get("_source_text_ids") or []),
                    *(layer.get("source_text_ids") or layer.get("_source_text_ids") or []),
                    str(primary.get("text_id") or primary.get("id") or "").strip(),
                    str(layer.get("text_id") or layer.get("id") or "").strip(),
                ]
            )
            if merged_trace_ids:
                primary["source_trace_ids"] = merged_trace_ids
                primary["_source_trace_ids"] = list(merged_trace_ids)
            if merged_text_ids:
                primary["source_text_ids"] = merged_text_ids
                primary["_source_text_ids"] = list(merged_text_ids)
            incoming_safe = _optional_bbox4(layer.get("safe_text_box")) or _optional_bbox4(layer.get("_debug_safe_text_box"))
            primary_safe = _optional_bbox4(primary.get("safe_text_box")) or _optional_bbox4(primary.get("_debug_safe_text_box"))
            incoming_target = _optional_bbox4(layer.get("target_bbox"))
            should_copy_geometry = False
            if incoming_safe is not None:
                if primary_safe is None:
                    should_copy_geometry = True
                else:
                    incoming_area = _bbox_area4(incoming_safe)
                    primary_area = _bbox_area4(primary_safe)
                    incoming_h = max(1, int(incoming_safe[3]) - int(incoming_safe[1]))
                    primary_h = max(1, int(primary_safe[3]) - int(primary_safe[1]))
                    should_copy_geometry = incoming_area >= int(primary_area * 1.15) or incoming_h >= primary_h + 32
            if should_copy_geometry:
                for key in (
                    "target_bbox",
                    "position_bbox",
                    "capacity_bbox",
                    "layout_safe_bbox",
                    "safe_text_box",
                    "_debug_safe_text_box",
                    "_safe_text_box_unclamped",
                    "render_bbox",
                    "_debug_render_bbox",
                    "balloon_bbox",
                    "bubble_mask_bbox",
                    "bubble_inner_bbox",
                    "_bubble_mask_bbox_unclamped",
                    "_bubble_inner_bbox_unclamped",
                ):
                    copied = _optional_bbox4(layer.get(key))
                    if copied is not None:
                        primary[key] = copied
                if incoming_target is not None and _optional_bbox4(primary.get("target_bbox")) is None:
                    primary["target_bbox"] = incoming_target
                if layer.get("layout_safe_reason"):
                    primary["layout_safe_reason"] = layer.get("layout_safe_reason")
                primary["_cross_page_band_rehomed_geometry"] = True
            _merge_layer_qa_flags(primary, ["cross_page_band_rehomed", "same_balloon_fragment_merged"])
            layer["visible"] = False
            layer["render_policy"] = "merged_into_primary"
            layer["route_action"] = "merged_into_primary"
            layer["merged_into_trace_id"] = primary.get("trace_id") or primary.get("id")
            layer["merged_into_text_id"] = primary.get("text_id") or primary.get("id")
            _merge_layer_qa_flags(layer, ["cross_page_band_rehomed"])
            moved += 1
    return moved


def _scrub_project_local_auxiliary_bboxes(project_data: dict) -> int:
    """Drop auxiliary render-plan bboxes that are still in band-local space."""

    if not isinstance(project_data, dict):
        return 0
    aux_keys = (
        "layout_bbox",
        "layout_safe_bbox",
        "position_bbox",
        "capacity_bbox",
        "_safe_text_box_unclamped",
    )
    scrubbed = 0
    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        for layer in _project_page_text_layers(page):
            if not isinstance(layer, dict):
                continue
            target = _optional_bbox4(layer.get("target_bbox"))
            if target is None:
                continue
            removed_any = False
            for key in aux_keys:
                bbox = _optional_bbox4(layer.get(key))
                if bbox is None:
                    continue
                if _bbox_intersection_area4(target, bbox) > 0:
                    continue
                layer.pop(key, None)
                scrubbed += 1
                removed_any = True
            if removed_any:
                if str(layer.get("layout_safe_reason") or "").strip().lower() == "debug_derived_bubble_mask_unclamped":
                    layer.pop("layout_safe_reason", None)
                _merge_layer_qa_flags(layer, ["page_space_aux_bbox_scrubbed"])
    return scrubbed


def _normalized_merge_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _suppress_broad_fallback_merge_layers(project_data: dict) -> int:
    """Hide broad fallback render layers when a smaller same-band peer owns the balloon."""

    suppressed = 0
    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        by_band: dict[str, list[dict]] = {}
        for layer in _project_page_text_layers(page):
            if not isinstance(layer, dict):
                continue
            band_id = str(layer.get("band_id") or "").strip()
            if band_id:
                by_band.setdefault(band_id, []).append(layer)
        for band_layers in by_band.values():
            visible_layers = [
                layer
                for layer in band_layers
                if layer.get("visible", True) is not False
                and str(layer.get("render_policy") or "") != "merged_into_primary"
                and _translated_text_for_merge(layer)
            ]
            for layer in list(visible_layers):
                flags = {str(flag) for flag in layer.get("qa_flags") or []}
                if not {"bbox_fallback_bubble_mask", "debug_derived_bubble_mask_rejected"}.issubset(flags):
                    continue
                layer_target = _visual_target_bbox_for_merge(layer)
                layer_bubble = _optional_bbox4(layer.get("bubble_mask_bbox")) or layer_target
                if layer_target is None or layer_bubble is None:
                    continue
                layer_target_area = max(1, _bbox_area4(layer_target))
                layer_bubble_area = max(1, _bbox_area4(layer_bubble))
                peers: list[dict] = []
                for peer in visible_layers:
                    if peer is layer:
                        continue
                    peer_target = _visual_target_bbox_for_merge(peer)
                    if peer_target is None:
                        continue
                    peer_area = max(1, _bbox_area4(peer_target))
                    if _bbox_intersection_area4(layer_target, peer_target) <= 0:
                        continue
                    broad_by_bubble = layer_bubble_area >= peer_area * 2.0
                    broad_by_target = layer_target_area >= peer_area * 2.0
                    text_overlap = (
                        _normalized_merge_text(peer.get("translated") or peer.get("traduzido") or "")
                        and _normalized_merge_text(peer.get("translated") or peer.get("traduzido") or "")
                        in _normalized_merge_text(layer.get("translated") or layer.get("traduzido") or "")
                    )
                    if text_overlap and (broad_by_bubble or broad_by_target or text_overlap):
                        peers.append(peer)
                if not peers:
                    continue
                peer = max(peers, key=lambda item: _merge_primary_preference_score(item, layer_target))
                layer_text = _normalized_merge_text(layer.get("translated") or layer.get("traduzido") or "")
                peer_text = _normalized_merge_text(peer.get("translated") or peer.get("traduzido") or "")
                if layer_text and peer_text and peer_text in layer_text:
                    peer["translated"] = layer_text
                    peer["traduzido"] = layer_text
                    _merge_layer_qa_flags(peer, ["broad_fallback_text_merged"])
                layer["visible"] = False
                layer["render_policy"] = "merged_into_primary"
                layer["route_action"] = "merged_into_primary"
                layer["merged_into_text_id"] = peer.get("text_id") or peer.get("id")
                _merge_layer_qa_flags(layer, ["broad_fallback_render_suppressed"])
                suppressed += 1
    return suppressed


def _is_project_bbox(value) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def _ensure_project_render_contract(project_data: dict) -> dict:
    """Audit final render metadata for translated layers before export gate."""

    audit = {
        "checked_layers": 0,
        "missing_render_bbox_count": 0,
        "missing_safe_text_box_count": 0,
        "filled_fit_metadata_count": 0,
        "dropped_stale_fit_flag_count": 0,
        "dropped_stale_render_background_flag_count": 0,
        "normalized_fit_status_count": 0,
    }
    for layer in _iter_project_text_layers(project_data):
        route_action = str(layer.get("route_action") or "").strip()
        if str(layer.get("render_policy") or "").strip() == "merged_into_primary":
            continue
        if not route_action.startswith("translate_"):
            continue
        audit["checked_layers"] += 1
        qa_flags = list(layer.get("qa_flags") or [])
        render_missing = not _is_project_bbox(layer.get("render_bbox"))
        safe_missing = not _is_project_bbox(layer.get("safe_text_box"))
        if render_missing:
            audit["missing_render_bbox_count"] += 1
        if safe_missing:
            audit["missing_safe_text_box_count"] += 1
        if render_missing or safe_missing or _translator_note_needs_compact_render(layer):
            if _repair_translator_note_best_effort_render(layer):
                audit["filled_fit_metadata_count"] += 1
                continue
        if render_missing or safe_missing:
            if "missing_render_bbox" not in qa_flags:
                qa_flags.append("missing_render_bbox")
            layer["qa_flags"] = qa_flags
            layer["fit_status"] = "below_minimum_legible"
            attempts = [item for item in list(layer.get("fit_attempts") or []) if isinstance(item, dict)]
            attempts.append(
                {
                    "font_px": int(((layer.get("estilo") or {}).get("tamanho") or 0) or 0),
                    "lines": 0,
                    "status": "missing_render_bbox",
                }
            )
            layer["fit_attempts"] = attempts[-4:]
            audit["filled_fit_metadata_count"] += 1
        else:
            if "missing_render_bbox" in qa_flags:
                layer["qa_flags"] = [
                    flag
                    for flag in qa_flags
                    if str(flag) != "missing_render_bbox"
                ]
                qa_flags = list(layer.get("qa_flags") or [])
            attempts = [item for item in list(layer.get("fit_attempts") or []) if isinstance(item, dict)]
            has_ok_attempt = any(str(item.get("status") or "").strip().lower() == "ok" for item in attempts)
            if (
                str(layer.get("fit_status") or "").strip().lower() == "below_minimum_legible"
                and has_ok_attempt
                and _bbox_contains4_margin(layer.get("safe_text_box"), layer.get("render_bbox"))
            ):
                layer["fit_status"] = "ok"
                audit["normalized_fit_status_count"] += 1
            if str(layer.get("fit_status") or "").strip().lower() == "ok" and "fit_below_minimum_legible" in qa_flags:
                layer["qa_flags"] = [
                    flag
                    for flag in qa_flags
                    if str(flag) != "fit_below_minimum_legible"
                ]
                qa_flags = list(layer.get("qa_flags") or [])
                audit["dropped_stale_fit_flag_count"] += 1
            if "render_on_art_suspected" in qa_flags and _render_background_art_flag_is_stale(layer):
                layer["qa_flags"] = [
                    flag
                    for flag in qa_flags
                    if str(flag) != "render_on_art_suspected"
                ]
                qa_flags = list(layer.get("qa_flags") or [])
                audit["dropped_stale_render_background_flag_count"] += 1
            if not isinstance(layer.get("fit_attempts"), list):
                layer["fit_attempts"] = [
                    {
                        "font_px": int(((layer.get("estilo") or {}).get("tamanho") or 0) or 0),
                        "lines": len(layer.get("linhas") or layer.get("lines") or []) or 1,
                        "status": "ok",
                    }
                ]
                audit["filled_fit_metadata_count"] += 1
            if not layer.get("fit_status"):
                layer["fit_status"] = "ok"
                audit["filled_fit_metadata_count"] += 1
    return audit


def _layer_is_translator_note(layer: dict) -> bool:
    text = str(layer.get("translated") or layer.get("text") or layer.get("original") or "").strip().lower()
    return text.startswith("t/n:") or text.startswith("tn:") or text.startswith("n/t:")


def _translator_note_needs_compact_render(layer: dict) -> bool:
    if not _layer_is_translator_note(layer):
        return False
    flags = {str(flag) for flag in layer.get("qa_flags") or []}
    fit_status = str(layer.get("fit_status") or "").strip().lower()
    style = _merge_style(layer.get("style") or layer.get("estilo"))
    try:
        size = int(style.get("tamanho") or 0)
    except (TypeError, ValueError):
        size = 0
    return bool(
        fit_status == "below_minimum_legible"
        or "fit_below_minimum_legible" in flags
        or size > 16
        or (
            "translator_note_best_effort_render" in flags
            and _translator_note_current_box_is_narrower_than_available(layer)
        )
    )


def _translator_note_current_box_is_narrower_than_available(layer: dict) -> bool:
    current = _optional_bbox4(layer.get("safe_text_box")) or _optional_bbox4(layer.get("render_bbox"))
    candidates = [
        _optional_bbox4(layer.get("target_bbox")),
        _optional_bbox4(layer.get("balloon_bbox")),
        current,
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if current is None or not candidates:
        return False
    current_area = max(0, current[2] - current[0]) * max(0, current[3] - current[1])
    best_area = max(max(0, box[2] - box[0]) * max(0, box[3] - box[1]) for box in candidates)
    return bool(best_area > 0 and current_area < best_area * 0.85)


def _repair_translator_note_best_effort_render(layer: dict) -> bool:
    if not _layer_is_translator_note(layer):
        return False
    candidates = [
        _optional_bbox4(layer.get("target_bbox")),
        _optional_bbox4(layer.get("balloon_bbox")),
        _optional_bbox4(layer.get("safe_text_box")),
        _optional_bbox4(layer.get("_debug_safe_text_box")),
        _optional_bbox4(layer.get("render_bbox")),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    safe = max(candidates, key=lambda box: max(0, box[2] - box[0]) * max(0, box[3] - box[1])) if candidates else None
    if safe is None:
        source = (
            _optional_bbox4(layer.get("source_bbox"))
            or _optional_bbox4(layer.get("text_pixel_bbox"))
            or _optional_bbox4(layer.get("bbox"))
        )
        if source is None:
            return False
        x1, y1, x2, y2 = source
        safe = [max(0, x1 - 80), max(0, y1 - 40), x2 + 160, y2 + 60]
    x1, y1, x2, y2 = [int(v) for v in safe]
    if x2 <= x1 or y2 <= y1:
        return False
    safe = [x1, y1, x2, y2]
    layer["safe_text_box"] = safe
    layer["_debug_safe_text_box"] = safe
    layer["target_bbox"] = safe
    layer["render_bbox"] = safe
    style = _merge_style(layer.get("style") or layer.get("estilo"))
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    compact_px = max(8, min(16, int(round(min(width / 18.0, height / 5.0)))))
    note_len = len(str(layer.get("translated") or layer.get("text") or layer.get("original") or ""))
    if note_len >= 60:
        compact_px = max(6, min(8, int(round(min(width / 32.0, height / 10.0)))))
    style["tamanho"] = min(int(style.get("tamanho") or compact_px), compact_px)
    style["force_upper"] = False
    style["alinhamento"] = "center"
    layer["style"] = style
    layer["estilo"] = style
    flags = [
        flag
        for flag in list(layer.get("qa_flags") or [])
        if str(flag) not in {"missing_render_bbox", "fit_below_minimum_legible"}
    ]
    if "translator_note_best_effort_render" not in flags:
        flags.append("translator_note_best_effort_render")
    layer["qa_flags"] = flags
    layer["fit_status"] = "ok"
    layer["fit_attempts"] = [{"font_px": int(style["tamanho"]), "lines": 1, "status": "ok"}]
    layer["render_policy"] = "normal"
    layer["_render_bbox_from_repaired_safe_text_box"] = True
    layer["layout_safe_reason"] = "translator_note_best_effort"
    if str(layer.get("route_action") or "").strip() == "review_required":
        layer["route_action"] = "translate_inpaint_render"
        layer.pop("route_reason", None)
        layer.pop("needs_review", None)
    return True


def _merge_layer_qa_flags(layer: dict, flags: list[str]) -> None:
    merged = list(layer.get("qa_flags") or [])
    for flag in flags:
        flag = str(flag).strip()
        if flag and flag not in merged:
            merged.append(flag)
    layer["qa_flags"] = merged


def _resolved_pre_render_flags(layer: dict) -> set[str]:
    metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    return {
        str(flag).strip()
        for flag in metrics.get("resolved_pre_render_flags") or []
        if str(flag).strip()
    }


def _drop_resolved_pre_render_flags(layer: dict, flags: list | set | tuple) -> list[str]:
    resolved = _resolved_pre_render_flags(layer)
    return [
        str(flag).strip()
        for flag in flags or []
        if str(flag).strip() and str(flag).strip() not in resolved
    ]


def _page_text_coordinate_audit_flags(page_texts: list[dict], *, height: int, width: int) -> list[str]:
    try:
        from debug_tools.bbox import audit_bbox_coordinate_space, coordinate_audit_flags, layout_block_records
    except Exception:
        return []
    page = {"height": int(height), "width": int(width), "texts": page_texts}
    try:
        audit = audit_bbox_coordinate_space(layout_block_records([page]))
        flags = coordinate_audit_flags(audit)
    except Exception:
        return []
    if "layout_bbox_coordinate_mismatch" not in flags:
        return flags

    findings = list(audit.get("findings") or [])
    summary = dict(audit.get("summary") or {})
    by_key = dict(summary.get("by_key") or {})
    all_explicit_page_space = all(
        str(text.get("coordinate_space") or "page") == "page"
        and str(text.get("source_coordinate_space") or "page") == "page"
        and not any(text.get(key) for key in ("band_y_top", "_band_y_top", "strip_band_y_top", "_strip_band_y_top"))
        for text in page_texts or []
        if isinstance(text, dict)
    )
    bbox_keys_consistent = all(int((value or {}).get("mismatch") or 0) == 0 for value in by_key.values())
    only_zero_offset_layout_findings = bool(findings) and all(
        item.get("blocker") == "layout_bbox_coordinate_mismatch"
        and item.get("issue") == "mixed_bbox_coordinate_space"
        and int(item.get("band_y_top") or 0) <= 0
        for item in findings
    )
    if (
        all_explicit_page_space
        and bbox_keys_consistent
        and only_zero_offset_layout_findings
        and int(summary.get("derived_bbox_coordinate_mismatch_count") or 0) == 0
        and int(summary.get("band_local_in_page_context_count") or 0) == 0
    ):
        return [flag for flag in flags if flag != "layout_bbox_coordinate_mismatch"]
    return flags


def _append_page_text_flags(page_texts: list[dict], flags: list[str]) -> None:
    if not flags:
        return
    for text in page_texts or []:
        if not isinstance(text, dict):
            continue
        _merge_layer_qa_flags(text, flags)


def _project_page_text_layers(page: dict) -> list[dict]:
    layers = page.get("text_layers")
    if isinstance(layers, dict):
        layers = layers.get("texts")
    if layers is None:
        layers = page.get("textos")
    return [layer for layer in list(layers or []) if isinstance(layer, dict)]


def _apply_final_project_coordinate_audit(project_data: dict) -> dict:
    try:
        from debug_tools.bbox import audit_bbox_coordinate_space, coordinate_audit_flags, layout_block_records
    except Exception:
        return {"applied": False, "flags_added": 0}

    pages = project_data.get("paginas") or []
    flags_added = 0
    checked_pages = 0
    flagged_pages = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        texts = _project_page_text_layers(page)
        if not texts:
            continue
        checked_pages += 1
        height = int(page.get("height") or page.get("altura") or 0)
        width = int(page.get("width") or page.get("largura") or 0)
        audit_page = {"height": height, "width": width, "texts": texts}
        try:
            flags = coordinate_audit_flags(audit_bbox_coordinate_space(layout_block_records([audit_page])))
        except Exception:
            flags = []
        if not flags:
            continue
        flagged_pages += 1
        before = sum(len(text.get("qa_flags") or []) for text in texts)
        _append_page_text_flags(texts, flags)
        after = sum(len(text.get("qa_flags") or []) for text in texts)
        flags_added += max(0, after - before)
    return {
        "applied": True,
        "checked_pages": checked_pages,
        "flagged_pages": flagged_pages,
        "flags_added": flags_added,
    }


def _load_debug_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    if not path.exists():
        return entries
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                entries.append(loaded)
    except Exception as exc:
        logger.warning("Falha ao ler jsonl de debug %s: %s", path, exc)
    return entries


def _debug_identity_values(payload: dict, fields: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for field in fields:
        raw_value = payload.get(field)
        raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in raw_values:
            key = str(value or "").strip()
            if key and key not in seen:
                values.append(key)
                seen.add(key)
    return values


def _debug_identity_key_groups_from_payload(payload: dict) -> list[list[str]]:
    groups: list[list[str]] = []
    seen_groups: set[tuple[str, ...]] = set()

    def add_group(values: list[str]) -> None:
        group: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = str(value or "").strip()
            if key and key not in seen:
                group.append(key)
                seen.add(key)
        group_key = tuple(group)
        if group and group_key not in seen_groups:
            groups.append(group)
            seen_groups.add(group_key)

    source_trace_ids = _debug_identity_values(payload, ("source_trace_ids", "_source_trace_ids"))
    source_text_ids = _debug_identity_values(payload, ("source_text_ids", "_source_text_ids"))
    text_ids = _debug_identity_values(payload, ("text_ids", "text_id", "id"))
    band_id = str(payload.get("band_id") or "").strip()
    add_group([*source_trace_ids, *_debug_identity_values(payload, ("trace_ids", "trace_id"))])
    add_group(_debug_identity_values(payload, ("text_instance_ids", "text_instance_id")))
    all_text_ids = [*source_text_ids, *text_ids]
    if band_id and all_text_ids:
        add_group([f"{text_id}@{band_id}" for text_id in all_text_ids])
        add_group([f"{band_id}_{text_id}" for text_id in all_text_ids])
    add_group(all_text_ids)
    return groups


def _debug_identity_keys_from_payload(payload: dict) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for group in _debug_identity_key_groups_from_payload(payload):
        for key in group:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _layer_debug_identity_keys(layer: dict) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    band_id = str(layer.get("band_id") or "").strip()

    def add_key(value) -> None:
        key = str(value or "").strip()
        if key and key not in seen:
            keys.append(key)
            seen.add(key)

    for field in ("trace_id", "text_instance_id"):
        add_key(layer.get(field))
    for field in ("source_trace_ids", "_source_trace_ids", "trace_ids"):
        for value in layer.get(field) or []:
            add_key(value)
    source_text_ids: list[str] = []
    for field in ("id", "text_id", "source_text_ids", "_source_text_ids", "text_ids"):
        values = layer.get(field)
        if field in {"id", "text_id"}:
            values = [values]
        for value in values or []:
            text_id = str(value or "").strip()
            if text_id and text_id not in source_text_ids:
                source_text_ids.append(text_id)
    if band_id:
        for text_id in source_text_ids:
            add_key(f"{text_id}@{band_id}")
            add_key(f"{band_id}_{text_id}")
    else:
        for text_id in source_text_ids:
            add_key(text_id)
    return keys


def _debug_root_from_project(project_data: dict) -> Path | None:
    work_dir = project_data.get("_work_dir") if isinstance(project_data, dict) else None
    if not work_dir:
        return None
    return Path(work_dir) / "debug" / "e2e"


def _qa_flag_claim(payload: dict, flags: set[str], source: str) -> dict | None:
    identity_groups = _debug_identity_key_groups_from_payload(payload)
    if not identity_groups or not flags:
        return None
    return {
        "identity_groups": identity_groups,
        "flags": set(flags),
        "source": source,
    }


MASK_SYNCED_QA_FLAGS = {
    "mask_outside_balloon",
    "mask_outside_balloon_critical",
    "bbox_fallback_bubble_mask",
    "glyph_mask_outside_bubble",
    "missing_real_bubble_mask",
    "source_glyph_area_ratio_critical",
}

RENDER_GEOMETRY_QA_FLAGS = {"TEXT_CLIPPED", "TEXT_OVERFLOW", "render_outside_balloon"}
RENDER_BACKGROUND_QA_FLAGS = {"render_on_art_suspected"}


def _render_fit_qa_flags(entry: dict) -> set[str]:
    if not isinstance(entry, dict):
        return set()
    qa_metrics = entry.get("qa_metrics") if isinstance(entry.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    return {
        str(flag).strip()
        for flag in render_fit.get("flags") or []
        if str(flag).strip() in RENDER_GEOMETRY_QA_FLAGS
    }


def _bbox_area4(value: list[int] | None) -> int:
    if value is None:
        return 0
    return max(0, int(value[2]) - int(value[0])) * max(0, int(value[3]) - int(value[1]))


def _bbox_intersection_area4(left: list[int] | None, right: list[int] | None) -> int:
    if left is None or right is None:
        return 0
    x1 = max(int(left[0]), int(right[0]))
    y1 = max(int(left[1]), int(right[1]))
    x2 = min(int(left[2]), int(right[2]))
    y2 = min(int(left[3]), int(right[3]))
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _compact_text_anchor_bbox_from_layer_metrics(layer: dict) -> list[int] | None:
    metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    candidates: list[list[int]] = []
    for metric_key, bbox_key in (
        ("inpaint_mask_contract_text_bbox_replaced_aggregate_source", "bbox"),
        ("layout_text_geometry_sanitized", "clean_bbox"),
        ("bbox_overreach", "text_geometry_bbox"),
        ("dark_connected_bubble_broad_mask_rejected", "anchor_bbox"),
    ):
        metric = metrics.get(metric_key) if isinstance(metrics.get(metric_key), dict) else {}
        bbox = _optional_bbox4(metric.get(bbox_key))
        if bbox is not None and _bbox_area4(bbox) >= 16:
            candidates.append(bbox)
    if not candidates:
        return None
    return min(candidates, key=_bbox_area4)


def _preserve_compact_text_anchor_against_debug_mask(
    layer: dict,
    candidate_mask_bbox: list[int] | None,
) -> list[int] | None:
    candidate = _optional_bbox4(candidate_mask_bbox)
    if candidate is None:
        return None
    flags = {str(flag).strip().lower() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
    source = str(layer.get("bubble_mask_source") or layer.get("balloon_mask_source") or "").strip().lower()
    dark_connected_context = bool(
        source == "image_dark_bubble_mask"
        and (
            "dark_bubble_connected_lobe_passthrough" in flags
            or "dark_connected_text_anchor_propagated_to_type" in flags
            or "dark_connected_lobe_anchor_component_filtered" in flags
            or layer.get("connected_lobe_bboxes")
            or layer.get("connected_position_bboxes")
            or str(layer.get("connected_balloon_orientation") or "").strip()
        )
    )
    if not dark_connected_context:
        return candidate

    metric_anchor = _compact_text_anchor_bbox_from_layer_metrics(layer)
    target_bbox = _optional_bbox4(
        layer.get("target_bbox")
        or layer.get("balloon_bbox")
        or layer.get("bubble_mask_bbox")
        or layer.get("safe_text_box")
    )
    existing_anchor = (
        _optional_bbox4(layer.get("source_text_anchor_bbox"))
        or _optional_bbox4(layer.get("_source_text_anchor_bbox"))
    )
    anchor = existing_anchor
    if metric_anchor is not None:
        metric_area = max(1, _bbox_area4(metric_anchor))
        metric_matches_target = target_bbox is None or (
            _bbox_intersection_area4(metric_anchor, target_bbox) / float(metric_area) >= 0.35
        )
        if metric_matches_target:
            anchor = metric_anchor
    if anchor is None:
        return candidate

    candidate_area = max(1, _bbox_area4(candidate))
    anchor_area = max(1, _bbox_area4(anchor))
    anchor_overlap = _bbox_intersection_area4(candidate, anchor) / float(anchor_area)
    if candidate_area <= int(anchor_area * 1.45) and anchor_overlap >= 0.60:
        return candidate

    text_pixel_bbox = _optional_bbox4(layer.get("text_pixel_bbox") or layer.get("bbox"))
    if text_pixel_bbox is not None:
        text_area = max(1, _bbox_area4(text_pixel_bbox))
        candidate_text_overlap = _bbox_intersection_area4(candidate, text_pixel_bbox) / float(text_area)
        if candidate_text_overlap >= 0.55:
            layer["source_text_anchor_bbox"] = list(anchor)
            layer["_source_text_anchor_bbox"] = list(anchor)
            layer["_anchor_center_only_layout"] = True
            _merge_layer_qa_flags(
                layer,
                ["source_text_mask_debug_bbox_recovered_from_text_pixels", "dark_connected_text_anchor_propagated_to_type"],
            )
            return text_pixel_bbox

    layer["source_text_anchor_bbox"] = list(anchor)
    layer["_source_text_anchor_bbox"] = list(anchor)
    layer["_anchor_center_only_layout"] = True
    _merge_layer_qa_flags(layer, ["source_text_mask_debug_bbox_rejected_broad", "dark_connected_text_anchor_propagated_to_type"])
    return None


def _bbox_under_covers_reference(candidate: list[int] | None, reference: list[int] | None) -> bool:
    candidate = _optional_bbox4(candidate)
    reference = _optional_bbox4(reference)
    if candidate is None or reference is None:
        return False
    cand_w = max(1, int(candidate[2]) - int(candidate[0]))
    cand_h = max(1, int(candidate[3]) - int(candidate[1]))
    ref_w = max(1, int(reference[2]) - int(reference[0]))
    ref_h = max(1, int(reference[3]) - int(reference[1]))
    ref_area = max(1, _bbox_area4(reference))
    overlap = _bbox_intersection_area4(candidate, reference)
    return bool(
        cand_w < int(ref_w * 0.72)
        or cand_h < int(ref_h * 0.72)
        or overlap / float(ref_area) < 0.68
    )


def _render_plan_source_bbox(layer: dict) -> list[int] | None:
    text_pixel = _optional_bbox4(layer.get("text_pixel_bbox"))
    source = _optional_bbox4(layer.get("source_bbox"))
    mask = _optional_bbox4(layer.get("source_text_mask_bbox") or layer.get("_source_text_mask_bbox"))
    anchor = _optional_bbox4(layer.get("source_text_anchor_bbox") or layer.get("_source_text_anchor_bbox"))
    if mask is not None and not _bbox_under_covers_reference(mask, text_pixel or source):
        return mask
    if text_pixel is not None and _bbox_area4(text_pixel) >= 16:
        return text_pixel
    if source is not None and not _bbox_under_covers_reference(source, mask):
        return source
    if anchor is not None and not _bbox_under_covers_reference(anchor, text_pixel or mask):
        return anchor
    return source or mask or anchor


def _bbox_overlap_ratio4(left: list[int] | None, right: list[int] | None) -> float:
    inter = _bbox_intersection_area4(left, right)
    if inter <= 0:
        return 0.0
    return inter / float(max(1, min(_bbox_area4(left), _bbox_area4(right))))


def _bbox_union_many(values: list[list[int]]) -> list[int] | None:
    bboxes = [_optional_bbox4(value) for value in values]
    bboxes = [bbox for bbox in bboxes if bbox is not None]
    if not bboxes:
        return None
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def _render_fit_evidence_is_stale(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    render_bbox = _optional_bbox4(entry.get("render_bbox"))
    if render_bbox is None:
        return False
    qa_metrics = entry.get("qa_metrics") if isinstance(entry.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    if not render_fit:
        return False
    current_target = (
        _optional_bbox4(entry.get("target_bbox"))
        or _optional_bbox4(entry.get("balloon_bbox"))
        or _optional_bbox4(entry.get("layout_bbox"))
        or _optional_bbox4(entry.get("capacity_bbox"))
    )
    fit_target = _optional_bbox4(render_fit.get("target_bbox")) or _optional_bbox4(render_fit.get("balloon_bbox"))
    if current_target is None or fit_target is None:
        return False
    current_area = _bbox_area4(current_target)
    fit_area = _bbox_area4(fit_target)
    if current_area <= 0 or fit_area <= 0:
        return False
    fit_inside_current = _bbox_contains4_margin(current_target, fit_target, margin=4)
    render_inside_current = _bbox_contains4_margin(current_target, render_bbox, margin=4)
    return bool(render_inside_current and fit_inside_current and fit_area < int(current_area * 0.40))


def _bbox_contains4_margin(outer: list[int] | None, inner: list[int] | None, margin: int = 2) -> bool:
    if outer is None or inner is None:
        return False
    return bool(
        inner[0] >= outer[0] - margin
        and inner[1] >= outer[1] - margin
        and inner[2] <= outer[2] + margin
        and inner[3] <= outer[3] + margin
    )


def _render_plan_geometry_is_clean(entry: dict) -> bool:
    render_bbox = _optional_bbox4(entry.get("render_bbox"))
    if render_bbox is None:
        return False
    safe_text_box = _optional_bbox4(entry.get("safe_text_box")) or _optional_bbox4(entry.get("_debug_safe_text_box"))
    if safe_text_box is not None and not _bbox_contains4_margin(safe_text_box, render_bbox):
        return False
    target_candidates = [
        _optional_bbox4(entry.get("target_bbox")),
        _optional_bbox4(entry.get("balloon_bbox")),
        _optional_bbox4(entry.get("layout_bbox")),
        _optional_bbox4(entry.get("capacity_bbox")),
    ]
    if any(_bbox_contains4_margin(candidate, render_bbox) for candidate in target_candidates if candidate is not None):
        return True
    qa_metrics = entry.get("qa_metrics") if isinstance(entry.get("qa_metrics"), dict) else {}
    try:
        containment = float(qa_metrics.get("render_balloon_containment"))
    except (TypeError, ValueError):
        containment = -1.0
    return containment >= 0.98


def _filter_render_plan_qa_flags(entry: dict, flags: set[str]) -> set[str]:
    filtered = set(flags)
    render_fit_flags = _render_fit_qa_flags(entry)
    if render_fit_flags and _render_fit_evidence_is_stale(entry):
        filtered.difference_update(render_fit_flags)
        render_fit_flags = set()
    if "bbox_overreach" in filtered:
        qa_metrics = entry.get("qa_metrics") if isinstance(entry, dict) else {}
        overreach = qa_metrics.get("bbox_overreach") if isinstance(qa_metrics, dict) else None
        if isinstance(overreach, dict) and overreach.get("broad_bbox_drives_mask") is False:
            filtered.discard("bbox_overreach")
    if filtered.intersection(RENDER_GEOMETRY_QA_FLAGS) and _render_plan_geometry_is_clean(entry):
        filtered.difference_update(RENDER_GEOMETRY_QA_FLAGS - render_fit_flags)
    if "render_on_art_suspected" in filtered and _render_background_art_flag_is_stale(entry):
        filtered.discard("render_on_art_suspected")
    if "fit_below_minimum_legible" in filtered and _render_plan_fit_is_clean(entry):
        filtered.discard("fit_below_minimum_legible")
    if "missing_render_bbox" in filtered and _render_plan_render_boxes_are_present(entry):
        filtered.discard("missing_render_bbox")
    if _scanlation_render_plan_flags_are_review_only(entry, filtered):
        filtered.clear()
    return filtered


def _scanlation_render_plan_flags_are_review_only(entry: dict, flags: set[str]) -> bool:
    if not flags:
        return False
    try:
        from qa.export_gate import (
            SCANLATION_HARMLESS_CONTEXT_FLAGS,
            SCANLATION_VISUAL_REVIEW_ONLY_FLAGS,
            _is_strong_scanlation_credit_layer,
        )
    except Exception:
        return False
    if not _is_strong_scanlation_credit_layer(entry):
        return False
    decision_flags = set(flags) - set(SCANLATION_HARMLESS_CONTEXT_FLAGS)
    return decision_flags.issubset(set(SCANLATION_VISUAL_REVIEW_ONLY_FLAGS))


def _render_background_art_flag_is_stale(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    qa_metrics = entry.get("qa_metrics") if isinstance(entry.get("qa_metrics"), dict) else {}
    try:
        background_luma = float(qa_metrics.get("render_background_luma"))
    except (TypeError, ValueError):
        background_luma = None
    if background_luma is not None:
        return background_luma >= 215.0
    return False


def _render_plan_fit_is_clean(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    if not _render_plan_render_boxes_are_present(entry):
        return False
    fit_status = str(entry.get("fit_status") or "").strip().lower()
    if fit_status == "ok":
        return True
    for attempt in entry.get("fit_attempts") or []:
        if isinstance(attempt, dict) and str(attempt.get("status") or "").strip().lower() == "ok":
            return True
    return False


def _render_plan_render_boxes_are_present(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    return _optional_bbox4(entry.get("render_bbox")) is not None and _optional_bbox4(entry.get("safe_text_box")) is not None


def _blocking_or_review_flags(flags: set[str]) -> set[str]:
    try:
        from qa.translation_qa import severity_for_flag
    except Exception:
        return set(flags)
    return {
        flag
        for flag in flags
        if severity_for_flag(flag) in {"critical", "high"}
    }


def _traceability_blocking_missing_flags(flags: set[str]) -> set[str]:
    try:
        from qa.translation_qa import severity_for_flag
    except Exception:
        return set(flags)
    return {flag for flag in flags if severity_for_flag(flag) == "critical"}


def _clean_inpaint_decision_for_mask_flags(decision: dict | None) -> bool:
    if not isinstance(decision, dict):
        return False
    flags = {str(flag).strip() for flag in decision.get("flags") or [] if str(flag).strip()}
    if _blocking_or_review_flags(flags):
        return False
    residual = decision.get("residual_text") if isinstance(decision.get("residual_text"), dict) else {}
    if residual.get("has_residual") is True:
        return False
    try:
        changed_outside_limit = int(decision.get("changed_pixels_outside_effective_limit") or 0)
    except Exception:
        changed_outside_limit = 0
    try:
        cleanup_outside_limit = int(decision.get("cleanup_changed_outside_limit_mask") or 0)
    except Exception:
        cleanup_outside_limit = 0
    return changed_outside_limit <= 0 and cleanup_outside_limit <= 8


def _load_inpaint_decision_for_band(debug_root: Path, band_id: str) -> dict | None:
    if not band_id:
        return None
    path = debug_root / "08_inpaint" / band_id / "inpaint_decision.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Falha ao ler inpaint_decision %s: %s", path, exc)
        return None


def _collect_render_plan_qa_flags(debug_root: Path) -> list[dict]:
    claims: list[dict] = []
    for entry in _load_debug_jsonl(debug_root / "09_typeset" / "render_plan_final.jsonl"):
        flags = {str(flag).strip() for flag in entry.get("qa_flags") or [] if str(flag).strip()}
        flags.update(_render_fit_qa_flags(entry))
        flags = {flag for flag in flags if flag not in MASK_SYNCED_QA_FLAGS}
        flags = _filter_render_plan_qa_flags(entry, flags)
        claim = _qa_flag_claim(entry, flags, "render_plan")
        if claim:
            claims.append(claim)
    return claims


def _debug_claim_missing_flag_is_review_only(flag: str, claim: dict) -> bool:
    if str(flag or "").strip() != "fast_fill_no_glyph_evidence":
        return False
    return str(claim.get("source") or "").strip() in {"render_plan", "inpaint_decision"}


def _collect_mask_decision_qa_flags(debug_root: Path) -> list[dict]:
    claims: list[dict] = []
    mask_root = debug_root / "06_mask_segmentation"
    if not mask_root.exists():
        return claims
    for path in mask_root.glob("*/mask_decision.json"):
        try:
            decision = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Falha ao ler mask_decision %s: %s", path, exc)
            continue
        flags = {str(flag).strip() for flag in decision.get("flags") or [] if str(flag).strip()}
        flags = {flag for flag in flags if flag in MASK_SYNCED_QA_FLAGS}
        if not flags:
            continue
        band_id = str(decision.get("band_id") or path.parent.name or "").strip()
        if _clean_inpaint_decision_for_mask_flags(_load_inpaint_decision_for_band(debug_root, band_id)):
            continue
        claim = _qa_flag_claim(decision, flags, "mask_decision")
        if claim:
            claims.append(claim)
    return claims


def _collect_inpaint_decision_qa_flags(debug_root: Path) -> list[dict]:
    claims: list[dict] = []
    inpaint_root = debug_root / "08_inpaint"
    if not inpaint_root.exists():
        return claims
    for path in inpaint_root.glob("*/inpaint_decision.json"):
        try:
            decision = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Falha ao ler inpaint_decision %s: %s", path, exc)
            continue
        flags = {str(flag).strip() for flag in decision.get("flags") or [] if str(flag).strip()}
        flags = _blocking_or_review_flags(flags)
        if not flags:
            continue
        claim = _qa_flag_claim(decision, flags, "inpaint_decision")
        if claim:
            claims.append(claim)
    return claims


def _page_id_from_band_id(band_id: str) -> str | None:
    match = re.search(r"(page_\d{3})_band_\d{3}", str(band_id or ""))
    return match.group(1) if match else None


def _metric_int(entry: dict, key: str) -> int:
    try:
        return int(entry.get(key) or 0)
    except Exception:
        return 0


def _metric_float(entry: dict, key: str) -> float:
    try:
        return float(entry.get(key) or 0.0)
    except Exception:
        return 0.0


def _candidate_bbox_geometry(entry: dict) -> tuple[int, int, int]:
    bbox = entry.get("bbox_page") or entry.get("bbox_strip")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return (0, 0, 0)
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    except Exception:
        return (0, 0, 0)
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return (width, height, width * height)


def _has_compact_unmatched_detect_geometry(entry: dict) -> bool:
    width, height, area = _candidate_bbox_geometry(entry)
    if width <= 0 or height <= 0:
        return False
    if area > 30000:
        return False
    if width > 300 or height > 220:
        return False
    return True


def _has_strong_unmatched_detect_text_evidence(entry: dict) -> bool:
    if entry.get("has_inner_dark_text") is not True:
        return False
    if not _has_compact_unmatched_detect_geometry(entry):
        return False
    if _metric_int(entry, "significant_component_count") < 4:
        return False
    if _metric_int(entry, "significant_area") < 600:
        return False
    if _metric_float(entry, "bright_pixel_ratio") < 0.25:
        return False
    if _metric_float(entry, "dark_pixel_ratio") > 0.12:
        return False
    return True


def _collect_unmatched_detect_candidates(debug_root: Path) -> list[dict]:
    candidates: list[dict] = []
    seen_candidate_ids: set[str] = set()
    for entry in _load_debug_jsonl(debug_root / "02_strip_detect" / "detect_candidates.jsonl"):
        if not isinstance(entry, dict) or entry.get("accepted") is not True:
            continue
        try:
            match_count = int(entry.get("match_count") or 0)
        except Exception:
            match_count = 0
        matched_ids = [item for item in list(entry.get("matched_text_ids") or []) if str(item).strip()]
        matched_traces = [item for item in list(entry.get("matched_trace_ids") or []) if str(item).strip()]
        matched_text_id = str(entry.get("matched_text_id") or "").strip()
        method = str(entry.get("match_method") or entry.get("match_reason") or "").strip()
        if match_count > 0 or matched_ids or matched_traces or matched_text_id:
            continue
        if method and method != "no_text_in_band":
            continue
        candidate_id = str(entry.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in seen_candidate_ids:
            continue
        if not _has_strong_unmatched_detect_text_evidence(entry):
            continue
        bbox_width, bbox_height, bbox_area = _candidate_bbox_geometry(entry)
        seen_candidate_ids.add(candidate_id)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "page_id": str(entry.get("page_id") or _page_id_from_band_id(str(entry.get("band_id") or "")) or "unresolved"),
                "band_id": str(entry.get("band_id") or "unresolved"),
                "bbox_page": entry.get("bbox_page"),
                "bbox_strip": entry.get("bbox_strip"),
                "match_reason": method or "no_text_in_band",
                "source": "detect_candidates",
                "flag": "detect_candidate_without_ocr_text",
                "evidence": {
                    "inner_dark_component_count": _metric_int(entry, "inner_dark_component_count"),
                    "inner_dark_area": _metric_int(entry, "inner_dark_area"),
                    "significant_component_count": _metric_int(entry, "significant_component_count"),
                    "significant_area": _metric_int(entry, "significant_area"),
                    "bright_pixel_ratio": _metric_float(entry, "bright_pixel_ratio"),
                    "dark_pixel_ratio": _metric_float(entry, "dark_pixel_ratio"),
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_area": bbox_area,
                },
            }
        )
    return candidates


def _claim_flag_count(claims: list[dict]) -> int:
    total = 0
    for claim in claims:
        primary_group = (claim.get("identity_groups") or [[]])[0]
        total += len(claim.get("flags") or []) * max(1, len(primary_group))
    return total


def _layers_by_debug_identity(project_layers: list[dict]) -> dict[str, list[dict]]:
    layers_by_identity: dict[str, list[dict]] = {}
    for layer in project_layers:
        for identity_key in _layer_debug_identity_keys(layer):
            bucket = layers_by_identity.setdefault(identity_key, [])
            if layer not in bucket:
                bucket.append(layer)
    return layers_by_identity


def _layout_block_bboxes(entry: dict) -> list[list[int]]:
    bboxes = entry.get("bboxes") if isinstance(entry.get("bboxes"), dict) else {}
    values: list[list[int]] = []
    for item in bboxes.values():
        if isinstance(item, dict):
            bbox = _optional_bbox4(item.get("value"))
        else:
            bbox = _optional_bbox4(item)
        if bbox is not None:
            values.append(bbox)
    return values


def _layout_blocks_by_debug_identity(debug_root: Path) -> dict[str, list[dict]]:
    blocks_by_identity: dict[str, list[dict]] = {}
    for entry in _load_debug_jsonl(debug_root / "05_layout_geometry" / "layout_blocks.jsonl"):
        if not isinstance(entry, dict):
            continue
        for identity_key in _debug_identity_keys_from_payload(entry):
            bucket = blocks_by_identity.setdefault(identity_key, [])
            bucket.append(entry)
    return blocks_by_identity


def _offset_bbox4(bbox: list[int] | None, dx: int, dy: int) -> list[int] | None:
    if bbox is None:
        return None
    return [int(bbox[0] + dx), int(bbox[1] + dy), int(bbox[2] + dx), int(bbox[3] + dy)]


def _debug_bbox_like_key(key: str) -> bool:
    key = str(key)
    return key.endswith("bbox") or key.endswith("_bbox") or key.endswith("_bboxes") or key in {
        "safe_text_box",
        "_debug_safe_text_box",
    }


def _offset_nested_debug_bboxes(value, dx: int, dy: int):
    if isinstance(value, dict):
        shifted = {}
        for key, item in value.items():
            if _debug_bbox_like_key(str(key)):
                if str(key).endswith("bboxes"):
                    shifted[key] = [
                        _offset_bbox4(_optional_bbox4(candidate), dx, dy) or candidate
                        for candidate in (item or [])
                    ]
                    continue
                bbox = _offset_bbox4(_optional_bbox4(item), dx, dy)
                if bbox is not None:
                    shifted[key] = bbox
                    continue
            shifted[key] = _offset_nested_debug_bboxes(item, dx, dy)
        return shifted
    if isinstance(value, list):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            bbox = _offset_bbox4(_optional_bbox4(value), dx, dy)
            if bbox is not None:
                return bbox
        return [_offset_nested_debug_bboxes(item, dx, dy) for item in value]
    return value


def _candidate_offset_from_layout(entry: dict, layout_blocks: list[dict]) -> tuple[int, int]:
    candidate_refs = [
        _optional_bbox4(entry.get("target_bbox")),
        _optional_bbox4(entry.get("bbox")),
        _optional_bbox4(entry.get("balloon_bbox")),
        _optional_bbox4(entry.get("render_bbox")),
    ]
    candidate_refs = [bbox for bbox in candidate_refs if bbox is not None]
    if not candidate_refs:
        return (0, 0)

    best: tuple[float, int, int] | None = None
    for candidate in candidate_refs:
        cw = max(1, candidate[2] - candidate[0])
        ch = max(1, candidate[3] - candidate[1])
        for layout_block in layout_blocks:
            for layout_bbox in _layout_block_bboxes(layout_block):
                lw = max(1, layout_bbox[2] - layout_bbox[0])
                lh = max(1, layout_bbox[3] - layout_bbox[1])
                width_delta = abs(cw - lw) / float(max(cw, lw))
                height_delta = abs(ch - lh) / float(max(ch, lh))
                if width_delta > 0.28 or height_delta > 0.28:
                    continue
                dx = int(layout_bbox[0] - candidate[0])
                dy = int(layout_bbox[1] - candidate[1])
                shifted = _offset_bbox4(candidate, dx, dy)
                if shifted is None:
                    continue
                score = _bbox_iou(shifted, layout_bbox) - ((abs(dx) + abs(dy)) / 1_000_000.0)
                if best is None or score > best[0]:
                    best = (score, dx, dy)
    if best is None:
        return (0, 0)
    return (best[1], best[2])


def _render_candidate_with_project_coordinates(entry: dict, layout_blocks: list[dict]) -> dict:
    candidate = dict(entry)
    dx, dy = _candidate_offset_from_layout(candidate, layout_blocks)
    if dx == 0 and dy == 0:
        return candidate
    for key in ("target_bbox", "safe_text_box", "_debug_safe_text_box", "render_bbox", "bbox", "source_bbox", "layout_bbox", "text_pixel_bbox"):
        shifted = _offset_bbox4(_optional_bbox4(candidate.get(key)), dx, dy)
        if shifted is not None:
            candidate[key] = shifted
    for key in ("qa_metrics", "_render_debug", "_render_debug_candidates", "_render_debug_skipped"):
        if isinstance(candidate.get(key), (dict, list)):
            candidate[key] = _offset_nested_debug_bboxes(candidate[key], dx, dy)
    candidate["_project_coordinate_offset"] = [dx, dy]
    return candidate


def _render_candidate_with_layer_coordinates(entry: dict, layer: dict) -> dict:
    candidate = dict(entry)
    candidate_qa = candidate.get("qa_metrics") if isinstance(candidate.get("qa_metrics"), dict) else {}
    candidate_overreach = candidate_qa.get("bbox_overreach") if isinstance(candidate_qa.get("bbox_overreach"), dict) else {}
    layer_qa = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    layer_overreach = layer_qa.get("bbox_overreach") if isinstance(layer_qa.get("bbox_overreach"), dict) else {}
    candidate_refs = [
        _optional_bbox4(candidate.get("target_bbox")),
        _optional_bbox4(candidate.get("bbox")),
        _optional_bbox4(candidate.get("balloon_bbox")),
        _optional_bbox4(candidate.get("safe_text_box")),
        _optional_bbox4(candidate.get("render_bbox")),
        _optional_bbox4(candidate_overreach.get("text_geometry_bbox")),
    ]
    candidate_refs = [bbox for bbox in candidate_refs if bbox is not None]
    layer_refs = [
        _optional_bbox4(layer.get("target_bbox")),
        _valid_layer_reference_bbox_for_render_match(layer, "bbox"),
        _valid_layer_text_pixel_bbox_for_render_match(layer),
        _valid_layer_reference_bbox_for_render_match(layer, "layout_bbox"),
        _optional_bbox4(layer.get("balloon_bbox")),
        _valid_layer_reference_bbox_for_render_match(layer, "source_bbox"),
        _optional_bbox4(layer_overreach.get("text_geometry_bbox")),
    ]
    layer_refs = [bbox for bbox in layer_refs if bbox is not None]
    best: tuple[float, int, int] | None = None
    for cand_ref in candidate_refs:
        cw = max(1, cand_ref[2] - cand_ref[0])
        ch = max(1, cand_ref[3] - cand_ref[1])
        for layer_ref in layer_refs:
            lw = max(1, layer_ref[2] - layer_ref[0])
            lh = max(1, layer_ref[3] - layer_ref[1])
            width_delta = abs(cw - lw) / float(max(cw, lw))
            height_delta = abs(ch - lh) / float(max(ch, lh))
            if width_delta > 0.35 or height_delta > 0.35:
                continue
            dx = int(layer_ref[0] - cand_ref[0])
            dy = int(layer_ref[1] - cand_ref[1])
            shifted = _offset_bbox4(cand_ref, dx, dy)
            if shifted is None:
                continue
            score = _bbox_iou(shifted, layer_ref) - ((abs(dx) + abs(dy)) / 1_000_000.0)
            if best is None or score > best[0]:
                best = (score, dx, dy)
    if best is None:
        return candidate
    _, dx, dy = best
    if dx == 0 and dy == 0:
        return candidate
    for key in ("target_bbox", "safe_text_box", "_debug_safe_text_box", "render_bbox", "bbox", "source_bbox", "layout_bbox", "text_pixel_bbox"):
        shifted = _offset_bbox4(_optional_bbox4(candidate.get(key)), dx, dy)
        if shifted is not None:
            candidate[key] = shifted
    for key in ("qa_metrics", "_render_debug", "_render_debug_candidates", "_render_debug_skipped"):
        if isinstance(candidate.get(key), (dict, list)):
            candidate[key] = _offset_nested_debug_bboxes(candidate[key], dx, dy)
    candidate["_render_metadata_layer_coordinate_offset"] = [dx, dy]
    return candidate


def _text_layout_contract_text_key(text: str) -> str:
    try:
        import re
        import unicodedata

        normalized = unicodedata.normalize("NFKC", text or "")
        normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return " ".join(normalized.strip().split()).casefold()
    except Exception:
        return " ".join(str(text or "").split()).casefold()


def _candidate_explicit_band_y_top(candidate: dict) -> int:
    if not isinstance(candidate, dict):
        return 0
    coordinate_space = str(candidate.get("coordinate_space") or "").strip().lower()
    if coordinate_space not in {"band", "local", "band_local"}:
        return 0
    for key in ("band_y_top", "_band_y_top", "strip_band_y_top", "_strip_band_y_top"):
        try:
            value = int(candidate.get(key) or 0)
        except Exception:
            value = 0
        if value:
            return value
    return 0


def _layer_looks_like_page_space_by_optional_bbox(layer: dict) -> bool:
    ys: list[int] = []
    for key in ("source_bbox", "text_pixel_bbox", "bbox", "layout_bbox"):
        bbox = _optional_bbox4(layer.get(key))
        if bbox is not None:
            ys.append(int(bbox[1]))
    return bool(ys and min(ys) >= 900)


def _shift_render_layout_contract(contract: dict, dx: int, dy: int, *, coordinate_space: str | None = None) -> dict:
    shifted = copy.deepcopy(contract)
    if dx or dy:
        positions = []
        for item in shifted.get("positions") or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                positions.append([int(item[0]) + int(dx), int(item[1]) + int(dy)])
            except Exception:
                continue
        if positions:
            shifted["positions"] = positions
        for key in ("block_bbox", "target_bbox", "position_bbox", "safe_text_box"):
            bbox = _offset_bbox4(_optional_bbox4(shifted.get(key)), dx, dy)
            if bbox is not None:
                shifted[key] = bbox
    if coordinate_space:
        shifted["coordinate_space"] = coordinate_space
        if coordinate_space == "page":
            shifted["band_y_top"] = 0
    return shifted


def _measure_contract_line_widths(font_name: str, font_size: int, lines: list[str]) -> list[int]:
    try:
        from typesetter.renderer import get_font, measure_text_width

        font = get_font(font_name, font_size)
        widths = [max(1, int(measure_text_width(font, line, font_size))) for line in lines]
        if len(widths) == len(lines):
            return widths
    except Exception:
        pass
    return []


def _render_layout_contract_from_candidate(candidate: dict) -> dict | None:
    if not isinstance(candidate, dict):
        return None
    if isinstance(candidate.get("render_layout_contract"), dict):
        return copy.deepcopy(candidate["render_layout_contract"])
    lines = [str(line) for line in candidate.get("wrapped_lines") or [] if str(line).strip()]
    if not lines:
        return None
    try:
        font_size = int(candidate.get("font_size_final") or candidate.get("font_size") or 0)
        line_height = int(candidate.get("line_height") or 0)
    except Exception:
        return None
    if font_size <= 0 or line_height <= 0:
        return None
    render_bbox = _optional_bbox4(candidate.get("render_bbox"))
    target_bbox = _optional_bbox4(candidate.get("target_bbox"))
    if render_bbox is None or target_bbox is None:
        return None
    font_name = str(candidate.get("font_name") or "")
    line_widths = [int(value) for value in candidate.get("line_widths") or [] if int(value or 0) > 0]
    if len(line_widths) != len(lines):
        line_widths = _measure_contract_line_widths(font_name, font_size, lines)
    if len(line_widths) != len(lines):
        max_width = max(1, int(render_bbox[2]) - int(render_bbox[0]))
        line_widths = [max_width for _ in lines]

    center_x = (int(render_bbox[0]) + int(render_bbox[2])) / 2.0
    center_y = (int(render_bbox[1]) + int(render_bbox[3])) / 2.0
    total_height = max(1, line_height * len(lines))
    start_y = int(round(center_y - (total_height / 2.0)))
    positions = [
        [int(round(center_x - (int(width) / 2.0))), int(start_y + index * line_height)]
        for index, width in enumerate(line_widths)
    ]
    block_bbox = _bbox_union_many(
        [
            [int(x), int(y), int(x) + int(width), int(y) + line_height]
            for (x, y), width in zip(positions, line_widths)
        ]
    )
    if block_bbox is None:
        return None
    translated = str(candidate.get("translated") or candidate.get("traduzido") or "")
    return {
        "schema_version": 1,
        "source": "debug_render_plan_raw",
        "translated_key": _text_layout_contract_text_key(translated),
        "font_name": font_name,
        "font_size": font_size,
        "line_height": line_height,
        "lines": lines,
        "positions": positions,
        "line_widths": line_widths,
        "block_bbox": block_bbox,
        "target_bbox": target_bbox,
        "position_bbox": _optional_bbox4(candidate.get("position_bbox")) or target_bbox,
        "safe_text_box": _optional_bbox4(candidate.get("safe_text_box"))
        or _optional_bbox4(candidate.get("_debug_safe_text_box"))
        or target_bbox,
        "coordinate_space": str(candidate.get("coordinate_space") or ""),
        "band_y_top": int(candidate.get("band_y_top") or candidate.get("_band_y_top") or 0),
    }


def _render_candidate_identity_token(candidate: dict) -> str:
    for key in ("trace_id", "text_instance_id", "text_id", "id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return ""


def _direct_render_identity_values(payload: dict) -> set[str]:
    values: set[str] = set()
    for key in ("trace_id", "text_id", "id"):
        value = str(payload.get(key) or "").strip()
        if value:
            values.add(value)
    band_id = str(payload.get("band_id") or "").strip()
    text_id = str(payload.get("text_id") or payload.get("id") or "").strip()
    if band_id and text_id:
        values.add(f"{text_id}@{band_id}")
        values.add(f"{band_id}_{text_id}")
    return values


def _candidate_matches_layer_direct_identity(layer: dict, candidate: dict) -> bool:
    return bool(_direct_render_identity_values(layer).intersection(_direct_render_identity_values(candidate)))


def _candidate_has_multiple_direct_sources(candidate: dict) -> bool:
    source_tokens: set[str] = set()
    for field in ("source_trace_ids", "_source_trace_ids", "source_text_ids", "_source_text_ids"):
        for value in candidate.get(field) or []:
            token = str(value or "").strip()
            if token:
                source_tokens.add(token)
    return len(source_tokens) > 1


def _candidate_has_other_same_band_direct_layer(
    layer: dict,
    candidate: dict,
    layers_by_identity: dict[str, list[dict]],
) -> bool:
    candidate_band_id = str(candidate.get("band_id") or "").strip()
    for identity_key in _direct_render_identity_values(candidate):
        for sibling in layers_by_identity.get(identity_key, []):
            if sibling is layer:
                continue
            if candidate_band_id and str(sibling.get("band_id") or "").strip() != candidate_band_id:
                continue
            if _candidate_matches_layer_direct_identity(sibling, candidate):
                return True
    return False


def _aggregate_render_candidates_for_layer(layer: dict, candidates: list[dict]) -> dict | None:
    source_tokens = set()
    for field in ("source_trace_ids", "_source_trace_ids", "trace_ids", "source_text_ids", "_source_text_ids", "text_ids"):
        for value in layer.get(field) or []:
            token = str(value or "").strip()
            if token:
                source_tokens.add(token)
    layer_identity_tokens = set(_layer_debug_identity_keys(layer))
    aggregate_same_layer_splits = len(source_tokens) < 2 and bool(layer_identity_tokens)

    def candidate_fragment_matches_layer(candidate: dict) -> bool:
        layer_values = [
            layer.get("translated"),
            layer.get("traduzido"),
            layer.get("normalized_text_final"),
            layer.get("normalized_ocr"),
            layer.get("text"),
        ]
        candidate_values = [
            candidate.get("translated"),
            candidate.get("traduzido"),
            candidate.get("render_text"),
            candidate.get("text"),
            candidate.get("normalized_text_final"),
            candidate.get("normalized_ocr"),
        ]
        layer_texts = [
            _normalize_match_text(str(value or ""))
            for value in layer_values
            if str(value or "").strip()
        ]
        candidate_texts = [
            _normalize_match_text(str(value or ""))
            for value in candidate_values
            if str(value or "").strip()
        ]
        for candidate_text in candidate_texts:
            if len(candidate_text) < 4:
                continue
            for layer_text in layer_texts:
                if candidate_text in layer_text:
                    return True
        return False

    if len(source_tokens) < 2 and not aggregate_same_layer_splits:
        return None

    selected: list[dict] = []
    seen_tokens: set[str] = set()
    seen_geometries: set[tuple[str, tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]] = set()
    layer_band_id = str(layer.get("band_id") or "").strip()
    layer_text_norm_for_candidate_filter = _normalized_merge_text(_translated_text_for_merge(layer))
    for candidate in candidates:
        if _is_low_containment_suppressed_fragment(candidate):
            continue
        candidate_band_id = str(candidate.get("band_id") or "").strip()
        if layer_band_id and candidate_band_id and candidate_band_id != layer_band_id:
            continue
        candidate_text_norm = _normalized_merge_text(_candidate_text_payload(candidate))
        if (
            not aggregate_same_layer_splits
            and layer_text_norm_for_candidate_filter
            and candidate_text_norm
            and candidate_text_norm not in layer_text_norm_for_candidate_filter
            and layer_text_norm_for_candidate_filter not in candidate_text_norm
        ):
            continue
        token = _render_candidate_identity_token(candidate)
        if not token:
            continue
        if aggregate_same_layer_splits:
            candidate_keys = set(_debug_identity_keys_from_payload(candidate))
            if not layer_identity_tokens.intersection(candidate_keys):
                continue
            if not candidate_fragment_matches_layer(candidate):
                continue
            geometry_key = (
                _normalize_match_text(str(candidate.get("translated") or candidate.get("render_text") or candidate.get("text") or "")),
                tuple(_optional_bbox4(candidate.get("target_bbox")) or []) or None,
                tuple(_optional_bbox4(candidate.get("render_bbox")) or []) or None,
            )
            if geometry_key in seen_geometries:
                continue
            selected.append(candidate)
            seen_geometries.add(geometry_key)
            continue
        if token in seen_tokens:
            continue
        if token not in source_tokens:
            text_id = str(candidate.get("text_id") or candidate.get("id") or "").strip()
            if text_id not in source_tokens:
                continue
        selected.append(candidate)
        seen_tokens.add(token)

    if aggregate_same_layer_splits and any(_render_bbox_overlaps_layer_source_text(layer, candidate.get("render_bbox")) for candidate in selected):
        selected = [
            candidate
            for candidate in selected
            if _render_bbox_overlaps_layer_source_text(layer, candidate.get("render_bbox"))
        ]

    if len(selected) < 2:
        return None

    explicit_same_layer_merge = bool(source_tokens) or (
        bool(layer_identity_tokens)
        and all(set(_debug_identity_keys_from_payload(candidate)).intersection(layer_identity_tokens) for candidate in selected)
    )
    if not explicit_same_layer_merge and not _render_candidates_share_same_visual_target(selected):
        return None

    target_union = _bbox_union_many(
        [bbox for bbox in (_optional_bbox4(candidate.get("target_bbox")) for candidate in selected) if bbox is not None]
    )
    layer_ref = (
        _valid_layer_reference_bbox_for_render_match(layer, "source_bbox")
        or _valid_layer_text_pixel_bbox_for_render_match(layer)
        or _valid_layer_reference_bbox_for_render_match(layer, "bbox")
        or _valid_layer_reference_bbox_for_render_match(layer, "layout_bbox")
    )
    if target_union is not None and layer_ref is not None:
        dx = int(layer_ref[0] - target_union[0])
        dy = int(layer_ref[1] - target_union[1])
        shifted_selected: list[dict] = []
        for candidate in selected:
            shifted = dict(candidate)
            for key in ("target_bbox", "safe_text_box", "_debug_safe_text_box", "render_bbox", "bbox", "source_bbox", "layout_bbox", "text_pixel_bbox"):
                bbox = _offset_bbox4(_optional_bbox4(shifted.get(key)), dx, dy)
                if bbox is not None:
                    shifted[key] = bbox
            for key in ("qa_metrics", "_render_debug", "_render_debug_candidates", "_render_debug_skipped"):
                if isinstance(shifted.get(key), (dict, list)):
                    shifted[key] = _offset_nested_debug_bboxes(shifted[key], dx, dy)
            shifted_selected.append(shifted)
        shifted_render_union = _bbox_union_many(
            [bbox for bbox in (_optional_bbox4(candidate.get("render_bbox")) for candidate in shifted_selected) if bbox is not None]
        )
        if _render_bbox_overlaps_layer_source_text(layer, shifted_render_union):
            selected = shifted_selected

    aggregate = dict(selected[0])
    render_bbox: list[int] | None = None
    safe_text_box: list[int] | None = None
    target_bbox: list[int] | None = None
    qa_flags: list[str] = []
    qa_metrics: dict = {}
    any_fit_ok = False
    any_fit_below = False
    for candidate in selected:
        render_bbox = _bbox_union_many([bbox for bbox in (render_bbox, _optional_bbox4(candidate.get("render_bbox"))) if bbox is not None])
        safe_text_box = _bbox_union_many([bbox for bbox in (safe_text_box, _optional_bbox4(candidate.get("safe_text_box")) or _optional_bbox4(candidate.get("_debug_safe_text_box"))) if bbox is not None])
        target_bbox = _bbox_union_many([bbox for bbox in (target_bbox, _optional_bbox4(candidate.get("target_bbox"))) if bbox is not None])
        for flag in candidate.get("qa_flags") or []:
            flag = str(flag).strip()
            if flag and flag not in qa_flags:
                qa_flags.append(flag)
        if isinstance(candidate.get("qa_metrics"), dict):
            qa_metrics.update(dict(candidate.get("qa_metrics") or {}))
        fit_status = str(candidate.get("fit_status") or "").strip().lower()
        any_fit_ok = any_fit_ok or fit_status == "ok"
        any_fit_below = any_fit_below or fit_status == "below_minimum_legible"
    if render_bbox is None or safe_text_box is None:
        return None
    aggregate["render_bbox"] = render_bbox
    aggregate["safe_text_box"] = safe_text_box
    aggregate["_debug_safe_text_box"] = safe_text_box
    if target_bbox is not None:
        aggregate["target_bbox"] = target_bbox
    aggregate["qa_flags"] = qa_flags
    if qa_metrics:
        aggregate["qa_metrics"] = qa_metrics
    if any_fit_ok:
        aggregate["fit_status"] = "ok"
    elif any_fit_below:
        aggregate["fit_status"] = "below_minimum_legible"
    if aggregate_same_layer_splits:
        layer_text = str(layer.get("translated") or layer.get("traduzido") or "").strip()
        if layer_text:
            aggregate["translated"] = layer_text
            aggregate["traduzido"] = layer_text
    aggregate["_render_metadata_aggregated_child_count"] = len(selected)
    return aggregate


def _render_candidates_share_same_visual_target(candidates: list[dict]) -> bool:
    if len(candidates) < 2:
        return True
    bubble_ids = {
        str(candidate.get("bubble_id") or candidate.get("balloon_id") or "").strip()
        for candidate in candidates
        if str(candidate.get("bubble_id") or candidate.get("balloon_id") or "").strip()
    }
    if len(bubble_ids) == 1:
        return True
    target_boxes = [
        _optional_bbox4(candidate.get("target_bbox"))
        or _optional_bbox4(candidate.get("safe_text_box"))
        or _optional_bbox4(candidate.get("_debug_safe_text_box"))
        or _optional_bbox4(candidate.get("render_bbox"))
        for candidate in candidates
    ]
    target_boxes = [bbox for bbox in target_boxes if bbox is not None]
    if len(target_boxes) < 2:
        return True
    for index, left in enumerate(target_boxes):
        for right in target_boxes[index + 1 :]:
            if _bbox_overlap_ratio4(left, right) < 0.35:
                return False
    return True


def _render_candidate_has_explicit_layer_source(layer: dict, candidate: dict) -> bool:
    layer_keys = set(_layer_debug_identity_keys(layer))
    if not layer_keys:
        return False
    candidate_keys = set(_debug_identity_keys_from_payload(candidate))
    return bool(layer_keys.intersection(candidate_keys))


def _candidate_merged_source_tokens(candidate: dict) -> set[str]:
    tokens: set[str] = set()
    for field in ("source_trace_ids", "_source_trace_ids", "source_text_ids", "_source_text_ids"):
        for value in candidate.get(field) or []:
            token = str(value or "").strip()
            if token:
                tokens.add(token)
    return tokens


def _candidate_source_layers_have_text_overlap(candidate: dict, layers_by_identity: dict[str, list[dict]]) -> bool:
    source_tokens = _candidate_merged_source_tokens(candidate)
    if len(source_tokens) < 2:
        return True
    band_id = str(candidate.get("band_id") or "").strip()
    matched_layers: list[dict] = []
    for token in source_tokens:
        for layer in layers_by_identity.get(token, []):
            if band_id and str(layer.get("band_id") or "").strip() != band_id:
                continue
            if layer not in matched_layers:
                matched_layers.append(layer)
    if len(matched_layers) < 2:
        return False
    return _source_linked_layers_have_text_overlap(matched_layers)


def _layer_source_area_for_merge(layer: dict) -> int:
    bbox = (
        _optional_bbox4(layer.get("text_pixel_bbox"))
        or _optional_bbox4(layer.get("source_bbox"))
        or _optional_bbox4(layer.get("layout_bbox"))
        or _optional_bbox4(layer.get("bbox"))
    )
    if bbox is None:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _visual_target_bbox_for_merge(layer: dict) -> list[int] | None:
    flags = {str(flag) for flag in layer.get("qa_flags") or []}
    bubble_bbox = _optional_bbox4(layer.get("bubble_mask_bbox"))
    target_bbox = _optional_bbox4(layer.get("target_bbox"))
    balloon_bbox = _optional_bbox4(layer.get("balloon_bbox"))
    if flags.intersection({"bbox_fallback_bubble_mask", "rejected_derived_bubble_mask", "debug_derived_bubble_mask_rejected"}):
        return (
            target_bbox
            or balloon_bbox
            or _optional_bbox4(layer.get("safe_text_box"))
            or _optional_bbox4(layer.get("_debug_safe_text_box"))
            or _optional_bbox4(layer.get("render_bbox"))
            or bubble_bbox
        )
    return (
        bubble_bbox
        or balloon_bbox
        or target_bbox
        or _optional_bbox4(layer.get("safe_text_box"))
        or _optional_bbox4(layer.get("_debug_safe_text_box"))
        or _optional_bbox4(layer.get("render_bbox"))
    )


def _dark_bubble_lobe_bbox_for_merge(layer: dict) -> list[int] | None:
    flags = {str(flag).strip().lower() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
    source = str(layer.get("bubble_mask_source") or layer.get("balloon_mask_source") or "").strip().lower()
    if source not in {"image_dark_bubble_mask", "image_dark_panel_mask"} and not any(
        flag.startswith("dark_bubble") for flag in flags
    ):
        return None
    return (
        _optional_bbox4(layer.get("balloon_bbox"))
        or _optional_bbox4(layer.get("target_bbox"))
        or _optional_bbox4(layer.get("safe_text_box"))
        or _optional_bbox4(layer.get("render_bbox"))
        or _optional_bbox4(layer.get("layout_bbox"))
        or _optional_bbox4(layer.get("bbox"))
    )


def _dark_bubble_anchor_bbox_for_merge(layer: dict) -> list[int] | None:
    text_bbox = _optional_bbox4(layer.get("text_pixel_bbox"))
    layout_bbox = _optional_bbox4(layer.get("layout_bbox"))
    bbox = _optional_bbox4(layer.get("bbox"))
    peer = layout_bbox or bbox
    if text_bbox is not None and peer is not None:
        text_area = max(1, _bbox_area4(text_bbox))
        peer_area = max(1, _bbox_area4(peer))
        inter = _bbox_intersection_area4(text_bbox, peer)
        if inter / float(min(text_area, peer_area)) >= 0.20 or _bbox_iou(text_bbox, peer) >= 0.12:
            return text_bbox
    elif text_bbox is not None:
        return text_bbox
    return (
        layout_bbox
        or bbox
        or _optional_bbox4(layer.get("render_bbox"))
        or _optional_bbox4(layer.get("safe_text_box"))
    )


def _layers_are_distinct_dark_bubble_lobes(left: dict, right: dict) -> bool:
    left_lobe = _dark_bubble_lobe_bbox_for_merge(left)
    right_lobe = _dark_bubble_lobe_bbox_for_merge(right)
    if left_lobe is None or right_lobe is None:
        return False
    left_anchor = _dark_bubble_anchor_bbox_for_merge(left)
    right_anchor = _dark_bubble_anchor_bbox_for_merge(right)
    if left_anchor is None or right_anchor is None:
        return False
    union = _bbox_union_many([left_lobe, right_lobe])
    if union is None:
        return False
    union_w = max(1, int(union[2]) - int(union[0]))
    union_h = max(1, int(union[3]) - int(union[1]))
    left_cx = (int(left_anchor[0]) + int(left_anchor[2])) / 2.0
    right_cx = (int(right_anchor[0]) + int(right_anchor[2])) / 2.0
    left_cy = (int(left_anchor[1]) + int(left_anchor[3])) / 2.0
    right_cy = (int(right_anchor[1]) + int(right_anchor[3])) / 2.0
    center_dx = abs(left_cx - right_cx)
    center_dy = abs(left_cy - right_cy)
    if center_dx < max(64, int(union_w * 0.22)) and center_dy < max(48, int(union_h * 0.18)):
        return False
    anchor_inter = _bbox_intersection_area4(left_anchor, right_anchor)
    anchor_min_area = max(1, min(_bbox_area4(left_anchor), _bbox_area4(right_anchor)))
    if (
        center_dx >= max(96, int(union_w * 0.45))
        and anchor_inter / float(anchor_min_area) < 0.20
    ):
        return True
    lobe_overlap = _bbox_overlap_ratio4(left_lobe, right_lobe)
    if lobe_overlap >= 0.82:
        return False
    return True


def _is_dark_connected_fragment_layer(layer: dict) -> bool:
    identity = " ".join(
        str(layer.get(key) or "")
        for key in ("id", "text_id", "trace_id")
    )
    if "_fragment_" not in identity and "#fragment_" not in identity:
        return False
    flags = {str(flag).strip().lower() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
    source = str(layer.get("bubble_mask_source") or layer.get("balloon_mask_source") or "").strip().lower()
    return bool(
        source in {"image_dark_bubble_mask", "image_dark_panel_mask"}
        or any(flag.startswith("dark_bubble") for flag in flags)
    )


def _dark_connected_fragment_covers_multiple_lobes(fragment: dict, siblings: list[dict]) -> bool:
    if not _is_dark_connected_fragment_layer(fragment):
        return False
    translated = _normalized_merge_text(_translated_text_for_merge(fragment))
    dark_siblings = [
        layer
        for layer in siblings
        if layer is not fragment
        and layer.get("visible", True) is not False
        and str(layer.get("render_policy") or "") != "merged_into_primary"
        and _dark_bubble_lobe_bbox_for_merge(layer) is not None
        and _translated_text_for_merge(layer)
    ]
    if len(dark_siblings) < 2:
        return False
    matched: list[dict] = []
    for sibling in dark_siblings:
        sibling_text = _normalized_merge_text(_translated_text_for_merge(sibling))
        if sibling_text and sibling_text in translated:
            matched.append(sibling)
    if len(matched) < 2:
        return False
    return any(
        _layers_are_distinct_dark_bubble_lobes(left, right)
        for idx, left in enumerate(matched)
        for right in matched[idx + 1 :]
    )


def _visual_targets_share_merge_region(left: dict, right: dict) -> bool:
    if _layers_are_distinct_dark_bubble_lobes(left, right):
        return False
    left_bbox = _visual_target_bbox_for_merge(left)
    right_bbox = _visual_target_bbox_for_merge(right)
    if left_bbox is None or right_bbox is None:
        return True
    left_area = max(1, (left_bbox[2] - left_bbox[0]) * (left_bbox[3] - left_bbox[1]))
    right_area = max(1, (right_bbox[2] - right_bbox[0]) * (right_bbox[3] - right_bbox[1]))
    inter_x1 = max(left_bbox[0], right_bbox[0])
    inter_y1 = max(left_bbox[1], right_bbox[1])
    inter_x2 = min(left_bbox[2], right_bbox[2])
    inter_y2 = min(left_bbox[3], right_bbox[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area <= 0:
        return False
    union_area = max(1, left_area + right_area - inter_area)
    iou = inter_area / float(union_area)
    containment = inter_area / float(max(1, min(left_area, right_area)))
    return iou >= 0.30 or containment >= 0.85


def _source_linked_layers_have_text_overlap(layers: list[dict]) -> bool:
    """Allow source-linked OCR fragments to merge when text evidence overlaps.

    Some repaired OCR fragments inherit separate synthetic bubble ids after a
    rejected fallback mask.  The explicit source ids are still trustworthy only
    when the actual text/glyph bboxes occupy the same region.
    """

    if len(layers) < 2:
        return True
    bboxes = [
        _optional_bbox4(layer.get("_merge_source_bbox_before_hydration")) or _layer_merge_order_bbox(layer)
        for layer in layers
    ]
    bboxes = [bbox for bbox in bboxes if bbox is not None]
    if len(bboxes) < 2:
        return False
    for index, left_bbox in enumerate(bboxes):
        for right_bbox in bboxes[index + 1 :]:
            inter = _bbox_intersection_area4(left_bbox, right_bbox)
            if inter <= 0:
                return False
            min_area = max(1, min(_bbox_area4(left_bbox), _bbox_area4(right_bbox)))
            if inter / float(min_area) < 0.55 and _bbox_iou(left_bbox, right_bbox) < 0.25:
                return False
    return True


def _clear_merge_source_bbox_before_hydration(layers: list[dict]) -> None:
    for layer in layers:
        if isinstance(layer, dict):
            layer.pop("_merge_source_bbox_before_hydration", None)


def _layer_has_rejected_merge_geometry(layer: dict) -> bool:
    flags = {str(flag) for flag in layer.get("qa_flags") or []}
    return bool(flags.intersection({"bbox_fallback_bubble_mask", "rejected_derived_bubble_mask", "debug_derived_bubble_mask_rejected"}))


def _candidate_matches_reliable_merge_target(candidate: dict, layers: list[dict]) -> bool:
    candidate_target = _optional_bbox4(candidate.get("target_bbox")) or _optional_bbox4(candidate.get("render_bbox"))
    if candidate_target is None:
        return False
    candidate_area = max(1, _bbox_area4(candidate_target))
    for layer in layers:
        if _layer_has_rejected_merge_geometry(layer):
            continue
        target = _visual_target_bbox_for_merge(layer)
        if target is None:
            continue
        target_area = max(1, _bbox_area4(target))
        inter = _bbox_intersection_area4(candidate_target, target)
        if inter <= 0:
            continue
        if inter / float(min(candidate_area, target_area)) >= 0.55 or _bbox_iou(candidate_target, target) >= 0.25:
            return True
    return False


def _layers_share_visual_merge_target(layers: list[dict]) -> bool:
    if len(layers) < 2:
        return True
    if _layers_have_conflicting_explicit_bubble_ids(layers):
        return False
    for index, left in enumerate(layers):
        for right in layers[index + 1 :]:
            if not _visual_targets_share_merge_region(left, right):
                return False
    return True


def _layers_look_like_cross_page_band_siblings(layers: list[dict]) -> bool:
    if len(layers) < 2:
        return False
    bboxes = [_visual_target_bbox_for_merge(layer) for layer in layers]
    bboxes = [bbox for bbox in bboxes if bbox is not None]
    if len(bboxes) < 2:
        return False
    has_page_bottom_piece = any(int(bbox[3]) >= 12000 for bbox in bboxes)
    has_page_top_piece = any(int(bbox[1]) <= 1000 for bbox in bboxes)
    if not has_page_bottom_piece or not has_page_top_piece:
        return False
    for index, left in enumerate(bboxes):
        for right in bboxes[index + 1 :]:
            left_w = max(1, int(left[2]) - int(left[0]))
            right_w = max(1, int(right[2]) - int(right[0]))
            x_overlap = max(0, min(int(left[2]), int(right[2])) - max(int(left[0]), int(right[0])))
            if x_overlap / float(min(left_w, right_w)) >= 0.35:
                return True
    return False


def _explicit_layer_bubble_id(layer: dict) -> str:
    for key in ("bubble_id", "balloon_id"):
        value = str(layer.get(key) or "").strip()
        if value:
            return value
    return ""


def _layers_have_conflicting_explicit_bubble_ids(layers: list[dict]) -> bool:
    ids = {_explicit_layer_bubble_id(layer) for layer in layers if _explicit_layer_bubble_id(layer)}
    return len(ids) > 1


def _remove_other_bubble_fragment_suffix(primary: dict, layers: list[dict]) -> bool:
    primary_bubble_id = _explicit_layer_bubble_id(primary)
    if not primary_bubble_id:
        return False
    original = str(primary.get("translated") or primary.get("traduzido") or "").strip()
    if not original:
        return False
    updated = original
    for layer in layers:
        if layer is primary:
            continue
        layer_bubble_id = _explicit_layer_bubble_id(layer)
        if not layer_bubble_id or layer_bubble_id == primary_bubble_id:
            continue
        fragment_text = _translated_text_for_merge(layer)
        if not fragment_text:
            continue
        pattern = re.compile(r"(?:\s+|\\n)+" + re.escape(fragment_text.strip()) + r"\s*$", re.IGNORECASE)
        updated = pattern.sub("", updated).strip()
    if updated and updated != original:
        primary["translated"] = updated
        primary["traduzido"] = updated
        return True
    return False


def _suppress_rejected_other_bubble_fragments(primary: dict, layers: list[dict]) -> int:
    primary_bubble_id = _explicit_layer_bubble_id(primary)
    if not primary_bubble_id:
        return 0
    primary_trace = str(primary.get("trace_id") or primary.get("id") or "").strip()
    suppressed = 0
    for layer in layers:
        if layer is primary:
            continue
        layer_bubble_id = _explicit_layer_bubble_id(layer)
        if not layer_bubble_id or layer_bubble_id == primary_bubble_id:
            continue
        flags = {str(flag) for flag in layer.get("qa_flags") or []}
        if not flags.intersection({"rejected_derived_bubble_mask", "debug_derived_bubble_mask_rejected"}):
            continue
        layer["visible"] = False
        layer["render_policy"] = "merged_into_primary"
        layer["route_action"] = "merged_into_primary"
        layer["merged_into_trace_id"] = primary_trace
        layer["merged_into_text_id"] = primary.get("text_id") or primary.get("id")
        _merge_layer_qa_flags(layer, ["cross_bubble_rejected_fragment_suppressed"])
        suppressed += 1
    return suppressed


def _primary_layer_for_merged_candidate(
    layer: dict,
    candidate: dict,
    layers_by_identity: dict[str, list[dict]],
) -> dict | None:
    source_tokens = _candidate_merged_source_tokens(candidate)
    if len(source_tokens) < 2:
        return None
    band_id = str(candidate.get("band_id") or "").strip()
    matched_layers: list[dict] = []
    for token in source_tokens:
        for candidate_layer in layers_by_identity.get(token, []):
            if band_id and str(candidate_layer.get("band_id") or "").strip() != band_id:
                continue
            if candidate_layer not in matched_layers:
                matched_layers.append(candidate_layer)
    if len(matched_layers) < 2:
        return None
    if any(
        _layers_are_distinct_dark_bubble_lobes(left, right)
        for idx, left in enumerate(matched_layers)
        for right in matched_layers[idx + 1 :]
    ):
        return None
    if _layers_have_conflicting_explicit_bubble_ids(matched_layers):
        if not _source_linked_layers_have_text_overlap(matched_layers):
            return None
    elif not _layers_share_visual_merge_target(matched_layers):
        return None
    if layer not in matched_layers:
        return None

    def primary_score(item: dict) -> tuple[int, int, int, int, int, int]:
        flags = {str(flag) for flag in item.get("qa_flags") or []}
        visual_bbox = _visual_target_bbox_for_merge(item)
        order_bbox = _layer_merge_order_bbox(item) or visual_bbox
        top = int(order_bbox[1]) if order_bbox is not None else 10**9
        left = int(order_bbox[0]) if order_bbox is not None else 10**9
        source_area = _layer_source_area_for_merge(item)
        visual_area = _bbox_area4(visual_bbox) if visual_bbox is not None else 0
        has_rejected_or_fallback = bool(
            flags.intersection({"bbox_fallback_bubble_mask", "rejected_derived_bubble_mask", "debug_derived_bubble_mask_rejected"})
        )
        has_bbox_fallback = "bbox_fallback_bubble_mask" in flags
        return (
            0 if has_bbox_fallback else 1,
            0 if has_rejected_or_fallback else 1,
            -top,
            -left,
            source_area,
            visual_area,
            1 if item is layer else 0,
        )

    return max(
        matched_layers,
        key=primary_score,
    )


def _hide_merged_candidate_sibling_layers(primary_layer: dict, candidate: dict, layers_by_identity: dict[str, list[dict]]) -> int:
    if _is_low_containment_suppressed_fragment(candidate):
        return 0
    source_tokens = _candidate_merged_source_tokens(candidate)
    if len(source_tokens) < 2:
        return 0
    resolved_primary = _primary_layer_for_merged_candidate(primary_layer, candidate, layers_by_identity)
    if resolved_primary is not primary_layer:
        return 0
    band_id = str(candidate.get("band_id") or "").strip()
    matched_layers: list[dict] = []
    for token in source_tokens:
        for layer in layers_by_identity.get(token, []):
            if band_id and str(layer.get("band_id") or "").strip() != band_id:
                continue
            if _is_low_containment_suppressed_fragment(layer):
                continue
            if layer not in matched_layers:
                matched_layers.append(layer)
    if _layers_have_conflicting_explicit_bubble_ids(matched_layers):
        if not _source_linked_layers_have_text_overlap(matched_layers):
            return 0
    elif not _layers_share_visual_merge_target(matched_layers):
        return 0
    hidden = 0
    primary_trace = str(primary_layer.get("trace_id") or primary_layer.get("id") or "").strip()
    candidate_text = _translated_text_for_merge(candidate)
    if (
        candidate_text
        and _source_linked_layers_have_text_overlap(matched_layers)
        and len(_normalized_merge_text(candidate_text)) >= len(
        _normalized_merge_text(primary_layer.get("translated") or primary_layer.get("traduzido") or "")
        )
    ):
        primary_layer["translated"] = candidate_text
        primary_layer["traduzido"] = candidate_text
        if not primary_layer.get("text"):
            primary_layer["text"] = candidate_text
    merged_trace_ids = _unique_preserve_order(
        [
            *(primary_layer.get("source_trace_ids") or primary_layer.get("_source_trace_ids") or []),
            *(candidate.get("source_trace_ids") or candidate.get("_source_trace_ids") or []),
        ]
    )
    merged_text_ids = _unique_preserve_order(
        [
            *(primary_layer.get("source_text_ids") or primary_layer.get("_source_text_ids") or []),
            *(candidate.get("source_text_ids") or candidate.get("_source_text_ids") or []),
        ]
    )
    if merged_trace_ids:
        primary_layer["source_trace_ids"] = merged_trace_ids
        primary_layer["_source_trace_ids"] = list(merged_trace_ids)
    if merged_text_ids:
        primary_layer["source_text_ids"] = merged_text_ids
        primary_layer["_source_text_ids"] = list(merged_text_ids)
    _merge_layer_qa_flags(primary_layer, ["same_balloon_fragment_merged"])
    hidden_layer_ids: set[int] = set()
    for token in source_tokens:
        for layer in layers_by_identity.get(token, []):
            if layer is primary_layer:
                continue
            if _is_low_containment_suppressed_fragment(layer):
                continue
            layer_identity = id(layer)
            if layer_identity in hidden_layer_ids:
                continue
            hidden_layer_ids.add(layer_identity)
            if band_id and str(layer.get("band_id") or "").strip() != band_id:
                continue
            layer["visible"] = False
            layer["render_policy"] = "merged_into_primary"
            layer["merged_into_trace_id"] = primary_trace
            layer["merged_into_text_id"] = primary_layer.get("text_id") or primary_layer.get("id")
            hidden += 1
    return hidden


def _project_page_image_size(project_data: dict, page: dict) -> tuple[int, int] | None:
    for width_key, height_key in (("width", "height"), ("largura", "altura")):
        try:
            width = int(page.get(width_key) or 0)
            height = int(page.get(height_key) or 0)
        except Exception:
            width = height = 0
        if width > 0 and height > 0:
            return (width, height)
    work_dir_value = project_data.get("_work_dir") if isinstance(project_data, dict) else None
    if not work_dir_value:
        return None
    work_dir = Path(work_dir_value)
    candidate_rels = [
        _resolve_image_layer_path(page, "rendered", page.get("arquivo_traduzido") or ""),
        _resolve_image_layer_path(page, "inpaint", ""),
        _resolve_image_layer_path(page, "base", page.get("arquivo_original") or ""),
        page.get("arquivo_original") or "",
    ]
    try:
        from PIL import Image

        for rel in candidate_rels:
            if not rel:
                continue
            path = work_dir / str(rel)
            if not path.exists():
                candidate = Path(str(rel))
                if candidate.exists():
                    path = candidate
            if not path.exists():
                continue
            with Image.open(path) as image:
                return image.size
    except Exception:
        return None
    return None


def _debug_mask_image_bbox(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            bbox = image.convert("L").getbbox()
    except Exception:
        return None
    if bbox is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _debug_mask_component_bbox(path: Path, layer: dict | None = None) -> list[int] | None:
    if not path.exists():
        return None
    try:
        import cv2
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            mask = np.array(image.convert("L")) > 0
    except Exception:
        return None
    if mask.size == 0 or not bool(mask.any()):
        return None
    try:
        components, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    except Exception:
        return None
    candidates: list[tuple[float, int, list[int]]] = []
    refs: list[list[int]] = []
    if isinstance(layer, dict):
        for key in (
            "source_text_anchor_bbox",
            "_source_text_anchor_bbox",
            "text_pixel_bbox",
            "bbox",
            "source_bbox",
            "layout_bbox",
        ):
            ref = _optional_bbox4(layer.get(key))
            if ref is not None and _bbox_area4(ref) >= 16:
                refs.append(ref)
    for label in range(1, int(components)):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 30 or w <= 1 or h <= 1:
            continue
        bbox = [x, y, x + w, y + h]
        overlap_score = 0.0
        for ref in refs:
            ref_local = list(ref)
            # Project/page-space refs keep the same x values but have page y.
            # Align only the vertical origin roughly when comparing local masks.
            if abs(int(ref_local[1]) - y) > 1000:
                ref_local = [ref_local[0], y, ref_local[2], y + max(1, int(ref[3]) - int(ref[1]))]
            overlap_score = max(overlap_score, _bbox_overlap_ratio4(bbox, ref_local))
        candidates.append((overlap_score, area, bbox))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen_overlap, _area, chosen = candidates[0]
    if refs and chosen_overlap <= 0.0:
        candidates.sort(key=lambda item: item[1], reverse=True)
        chosen = candidates[0][2]
    return chosen


def _debug_mask_decision_repair_reject_reason(decision: dict) -> str | None:
    if not isinstance(decision, dict):
        return None
    try:
        outside_ratio = float(decision.get("outside_balloon_ratio") or 0.0)
    except Exception:
        outside_ratio = 0.0
    try:
        outside_pixels = int(decision.get("outside_balloon_pixels") or 0)
    except Exception:
        outside_pixels = 0
    gates = decision.get("gates") if isinstance(decision.get("gates"), dict) else {}
    synthetic_tight = bool(decision.get("synthetic_tight_balloon_reference"))
    if bool(gates.get("mask_outside_balloon_critical")):
        return "mask_outside_balloon_critical"
    if outside_ratio >= 0.18 and outside_pixels >= 50:
        return "mask_outside_balloon"
    if synthetic_tight and outside_ratio >= 0.08 and outside_pixels >= 25:
        return "synthetic_tight_mask_outside_balloon"
    return None


def _debug_mask_path_segment(value: object) -> str:
    raw = str(value or "").strip() or "text"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return safe.strip("._") or "text"


def _candidate_direct_trace_id(candidate: dict) -> str:
    trace_id = str(candidate.get("trace_id") or "").strip()
    if trace_id:
        return trace_id
    source_trace_ids = [
        str(value or "").strip()
        for value in candidate.get("source_trace_ids") or candidate.get("_source_trace_ids") or []
        if str(value or "").strip()
    ]
    return source_trace_ids[0] if len(source_trace_ids) == 1 else ""


def _candidate_text_payload(candidate: dict) -> str:
    return str(candidate.get("translated") or candidate.get("traduzido") or "").strip()


def _candidate_layout_bbox(candidate: dict) -> list[int] | None:
    return (
        _optional_bbox4(candidate.get("bbox"))
        or _optional_bbox4(candidate.get("text_pixel_bbox"))
        or _optional_bbox4(candidate.get("layout_bbox"))
        or _optional_bbox4(candidate.get("render_bbox"))
        or _optional_bbox4(candidate.get("safe_text_box"))
    )


def _candidate_centers_overlap_or_touch(a: dict, b: dict) -> bool:
    abox = _candidate_layout_bbox(a)
    bbox = _candidate_layout_bbox(b)
    if abox is None or bbox is None:
        return False
    x_overlap = max(0, min(abox[2], bbox[2]) - max(abox[0], bbox[0]))
    min_width = max(1, min(abox[2] - abox[0], bbox[2] - bbox[0]))
    y_gap = max(0, max(abox[1], bbox[1]) - min(abox[3], bbox[3]))
    min_height = max(1, min(abox[3] - abox[1], bbox[3] - bbox[1]))
    if x_overlap / float(min_width) >= 0.12 and y_gap <= max(80, min_height * 3):
        return True
    return _bbox_overlap_ratio4(abox, bbox) >= 0.08 or _bbox_overlap_ratio4(bbox, abox) >= 0.08


def _layer_matches_trace_id(layer: dict, trace_id: str) -> bool:
    if not trace_id:
        return False
    identities = {
        str(layer.get("trace_id") or "").strip(),
        *[
            str(value or "").strip()
            for value in layer.get("source_trace_ids") or layer.get("_source_trace_ids") or []
            if str(value or "").strip()
        ],
    }
    return trace_id in identities


def _set_layer_text_payload(layer: dict, text: str) -> None:
    layer["translated"] = text
    layer["traduzido"] = text
    if not str(layer.get("text") or "").strip():
        layer["text"] = text


def _layer_identity_tokens(layer: dict) -> set[str]:
    tokens: set[str] = set()
    for key in ("id", "text_id", "trace_id"):
        value = str(layer.get(key) or "").strip()
        if value:
            tokens.add(value)
    return tokens


def _reset_layer_source_ids_to_self(layer: dict) -> None:
    trace_id = str(layer.get("trace_id") or "").strip()
    text_id = str(layer.get("text_id") or layer.get("id") or "").strip()
    if trace_id:
        layer["source_trace_ids"] = [trace_id]
        layer["_source_trace_ids"] = [trace_id]
    else:
        layer.pop("source_trace_ids", None)
        layer.pop("_source_trace_ids", None)
    if text_id:
        layer["source_text_ids"] = [text_id]
        layer["_source_text_ids"] = [text_id]
    else:
        layer.pop("source_text_ids", None)
        layer.pop("_source_text_ids", None)


def _layer_has_translated_payload(layer: dict) -> bool:
    return bool(str(layer.get("translated") or layer.get("traduzido") or "").strip())


def _split_dark_lobe_contaminated_translation(primary: dict, sibling: dict) -> tuple[str, str] | None:
    translated = re.sub(r"\s+", " ", _translated_text_for_merge(primary)).strip()
    if not translated:
        return None
    metrics = sibling.get("qa_metrics") if isinstance(sibling.get("qa_metrics"), dict) else {}
    cleanup = metrics.get("leading_dark_lobe_duplicate_fragment_removed") if isinstance(metrics, dict) else None
    if not isinstance(cleanup, dict):
        return None
    before = str(cleanup.get("from") or "").strip()
    after = str(cleanup.get("to") or sibling.get("text") or sibling.get("original") or "").strip()
    if not before or not after or before == after:
        return None
    duplicate_head = before.split(after, 1)[0].strip() if after in before else before
    duplicate_tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9']+", duplicate_head.lower())
        if len(token) > 1
    ]
    if len(set(duplicate_tokens)) < 2:
        return None
    lower = translated.lower()
    starts: list[int] = []
    for token in duplicate_tokens:
        match = re.search(rf"\b{re.escape(token)}\b", lower)
        if match:
            starts.append(match.start())
    if len(starts) < 2:
        return None
    leak_start = min(starts)
    window = lower[leak_start : leak_start + 120]
    matched_in_window = {
        token
        for token in duplicate_tokens
        if re.search(rf"\b{re.escape(token)}\b", window)
    }
    if len(matched_in_window) < 2:
        return None
    sentence_end_match = re.search(r"[.!?]", translated[leak_start:])
    if not sentence_end_match:
        return None
    leak_end = leak_start + sentence_end_match.end()
    primary_text = translated[:leak_start].strip(" \t\r\n,.;:!?")
    sibling_text = translated[leak_end:].strip(" \t\r\n,.;:!?")
    if not primary_text or not sibling_text:
        return None
    return primary_text, sibling_text


def _render_anchor_for_dark_lobe_safe(layer: dict, safe: list[int]) -> list[int]:
    anchor = (
        _dark_bubble_anchor_bbox_for_merge(layer)
        or _optional_bbox4(layer.get("text_pixel_bbox"))
        or _optional_bbox4(layer.get("bbox"))
        or _optional_bbox4(layer.get("layout_bbox"))
    )
    if anchor is None:
        return [int(v) for v in safe]
    x1 = max(int(safe[0]), int(anchor[0]))
    y1 = max(int(safe[1]), int(anchor[1]))
    x2 = min(int(safe[2]), int(anchor[2]))
    y2 = min(int(safe[3]), int(anchor[3]))
    if x2 <= x1 or y2 <= y1:
        return [int(v) for v in safe]
    return [x1, y1, x2, y2]


def _split_shared_dark_lobe_safe_areas(primary: dict, sibling: dict) -> bool:
    primary_anchor = _dark_bubble_anchor_bbox_for_merge(primary)
    sibling_anchor = _dark_bubble_anchor_bbox_for_merge(sibling)
    primary_existing_safe = (
        _optional_bbox4(primary.get("safe_text_box"))
        or _optional_bbox4(primary.get("_debug_safe_text_box"))
        or _optional_bbox4(primary.get("layout_safe_bbox"))
    )
    sibling_existing_safe = (
        _optional_bbox4(sibling.get("safe_text_box"))
        or _optional_bbox4(sibling.get("_debug_safe_text_box"))
        or _optional_bbox4(sibling.get("layout_safe_bbox"))
    )
    if primary_anchor is not None and sibling_anchor is not None and (primary_existing_safe is None) != (sibling_existing_safe is None):
        existing_safe = primary_existing_safe or sibling_existing_safe
        missing_layer = primary if primary_existing_safe is None else sibling
        missing_anchor = primary_anchor if primary_existing_safe is None else sibling_anchor
        bounds = (
            _optional_bbox4(missing_layer.get("target_bbox"))
            or _optional_bbox4(missing_layer.get("bubble_mask_bbox"))
            or _optional_bbox4(missing_layer.get("balloon_bbox"))
        )
        if existing_safe is not None and missing_anchor is not None and bounds is not None:
            ew = max(1, int(existing_safe[2]) - int(existing_safe[0]))
            cx = (int(missing_anchor[0]) + int(missing_anchor[2])) / 2.0
            x1 = int(round(cx - ew / 2.0))
            x2 = x1 + ew
            if x1 < int(bounds[0]):
                x2 += int(bounds[0]) - x1
                x1 = int(bounds[0])
            if x2 > int(bounds[2]):
                x1 -= x2 - int(bounds[2])
                x2 = int(bounds[2])
            x1 = max(int(bounds[0]), x1)
            x2 = min(int(bounds[2]), x2)
            safe = [x1, int(existing_safe[1]), x2, int(existing_safe[3])]
            if safe[2] > safe[0] and safe[3] > safe[1]:
                for key in ("safe_text_box", "_debug_safe_text_box", "position_bbox", "capacity_bbox", "layout_safe_bbox"):
                    missing_layer[key] = [int(v) for v in safe]
                missing_layer["render_bbox"] = _render_anchor_for_dark_lobe_safe(missing_layer, safe)
                _merge_layer_qa_flags(missing_layer, ["distinct_dark_lobe_safe_area_split"])
                return True
    shared_safe = (
        _optional_bbox4(primary.get("safe_text_box"))
        or _optional_bbox4(sibling.get("safe_text_box"))
        or _optional_bbox4(primary.get("layout_safe_bbox"))
        or _optional_bbox4(sibling.get("layout_safe_bbox"))
        or _optional_bbox4(primary.get("capacity_bbox"))
        or _optional_bbox4(sibling.get("capacity_bbox"))
        or _optional_bbox4(primary.get("position_bbox"))
        or _optional_bbox4(sibling.get("position_bbox"))
        or _optional_bbox4(primary.get("target_bbox"))
        or _optional_bbox4(sibling.get("target_bbox"))
        or _optional_bbox4(primary.get("bubble_mask_bbox"))
        or _optional_bbox4(sibling.get("bubble_mask_bbox"))
    )
    if primary_anchor is None or sibling_anchor is None or shared_safe is None:
        return False
    pcx = (int(primary_anchor[0]) + int(primary_anchor[2])) / 2.0
    scx = (int(sibling_anchor[0]) + int(sibling_anchor[2])) / 2.0
    pcy = (int(primary_anchor[1]) + int(primary_anchor[3])) / 2.0
    scy = (int(sibling_anchor[1]) + int(sibling_anchor[3])) / 2.0
    horizontal = abs(pcx - scx) >= abs(pcy - scy)
    sx1, sy1, sx2, sy2 = [int(v) for v in shared_safe]
    if sx2 <= sx1 or sy2 <= sy1:
        return False
    if horizontal:
        divider = int(round((pcx + scx) / 2.0))
        divider = max(sx1 + 48, min(sx2 - 48, divider))
        gap = 6
        first = [sx1, sy1, max(sx1 + 24, divider - gap), sy2]
        second = [min(sx2 - 24, divider + gap), sy1, sx2, sy2]
        primary_safe, sibling_safe = (first, second) if pcx <= scx else (second, first)
    else:
        divider = int(round((pcy + scy) / 2.0))
        divider = max(sy1 + 36, min(sy2 - 36, divider))
        gap = 6
        first = [sx1, sy1, sx2, max(sy1 + 24, divider - gap)]
        second = [sx1, min(sy2 - 24, divider + gap), sx2, sy2]
        primary_safe, sibling_safe = (first, second) if pcy <= scy else (second, first)
    for layer, safe in ((primary, primary_safe), (sibling, sibling_safe)):
        if safe[2] <= safe[0] or safe[3] <= safe[1]:
            continue
        layer["safe_text_box"] = [int(v) for v in safe]
        layer["_debug_safe_text_box"] = [int(v) for v in safe]
        layer["position_bbox"] = [int(v) for v in safe]
        layer["capacity_bbox"] = [int(v) for v in safe]
        layer["layout_safe_bbox"] = [int(v) for v in safe]
        layer["render_bbox"] = _render_anchor_for_dark_lobe_safe(layer, safe)
        for stale_key in ("_debug_render_bbox", "fit_status", "fit_attempts"):
            layer.pop(stale_key, None)
        _merge_layer_qa_flags(layer, ["distinct_dark_lobe_safe_area_split"])
    return True


def _finalize_distinct_dark_lobe_project_geometry(project_layers: list[dict]) -> int:
    repaired = 0
    layers_by_band: dict[str, list[dict]] = {}
    for layer in project_layers:
        if not isinstance(layer, dict):
            continue
        flags = {str(flag) for flag in layer.get("qa_flags") or []}
        if "distinct_dark_lobe_payload_merge_repaired" not in flags and "distinct_dark_lobe_safe_area_split" not in flags:
            continue
        band_id = str(layer.get("band_id") or "").strip() or (_trace_band_id(str(layer.get("trace_id") or "")) or "")
        if band_id:
            layers_by_band.setdefault(band_id, []).append(layer)
    for band_layers in layers_by_band.values():
        visible = [
            layer
            for layer in band_layers
            if layer.get("visible", True) is not False
            and str(layer.get("render_policy") or "").strip().lower() != "merged_into_primary"
        ]
        for idx, left in enumerate(visible):
            for right in visible[idx + 1 :]:
                if not _layers_are_distinct_dark_bubble_lobes(left, right):
                    continue
                left_missing = _optional_bbox4(left.get("safe_text_box")) is None
                right_missing = _optional_bbox4(right.get("safe_text_box")) is None
                if left_missing or right_missing:
                    if _split_shared_dark_lobe_safe_areas(left, right):
                        repaired += 1
    for layer in project_layers:
        if not isinstance(layer, dict):
            continue
        flags = {str(flag) for flag in layer.get("qa_flags") or []}
        if "distinct_dark_lobe_payload_merge_repaired" not in flags and "distinct_dark_lobe_safe_area_split" not in flags:
            continue
        safe = (
            _optional_bbox4(layer.get("safe_text_box"))
            or _optional_bbox4(layer.get("_debug_safe_text_box"))
            or _optional_bbox4(layer.get("layout_safe_bbox"))
            or _optional_bbox4(layer.get("position_bbox"))
            or _optional_bbox4(layer.get("capacity_bbox"))
        )
        changed = False
        if safe is not None:
            for key in ("safe_text_box", "_debug_safe_text_box", "position_bbox", "capacity_bbox"):
                if _optional_bbox4(layer.get(key)) != safe:
                    layer[key] = [int(v) for v in safe]
                    changed = True
            if _optional_bbox4(layer.get("render_bbox")) is None:
                layer["render_bbox"] = _render_anchor_for_dark_lobe_safe(layer, safe)
                changed = True
        source_tokens = {
            str(value or "").strip()
            for field in ("source_trace_ids", "_source_trace_ids", "source_text_ids", "_source_text_ids")
            for value in (layer.get(field) or [])
            if str(value or "").strip()
        }
        identities = _layer_identity_tokens(layer)
        source_is_self_only = bool(source_tokens) and source_tokens.issubset(identities)
        if isinstance(layer.get("qa_flags"), list) and source_is_self_only:
            cleaned_flags = [
                flag
                for flag in layer.get("qa_flags") or []
                if str(flag) not in {"same_balloon_fragment_merged", "missing_render_bbox", "fit_below_minimum_legible"}
            ]
            if cleaned_flags != layer.get("qa_flags"):
                layer["qa_flags"] = cleaned_flags
                changed = True
        if str(layer.get("fit_status") or "").strip().lower() == "below_minimum_legible" and _optional_bbox4(layer.get("render_bbox")) is not None:
            layer.pop("fit_status", None)
            changed = True
        if changed:
            repaired += 1
    return repaired


def _repair_distinct_dark_lobe_project_payload_merges(project_layers: list[dict]) -> int:
    repaired = 0
    layers_by_band: dict[str, list[dict]] = {}
    for layer in project_layers:
        if not isinstance(layer, dict):
            continue
        band_id = str(layer.get("band_id") or "").strip() or (_trace_band_id(str(layer.get("trace_id") or "")) or "")
        if band_id:
            layers_by_band.setdefault(band_id, []).append(layer)

    for band_layers in layers_by_band.values():
        visible_layers = [
            layer
            for layer in band_layers
            if layer.get("visible", True) is not False
            and str(layer.get("render_policy") or "").strip().lower() != "merged_into_primary"
        ]
        for primary in visible_layers:
            primary_sources = {
                str(value or "").strip()
                for field in ("source_trace_ids", "_source_trace_ids", "source_text_ids", "_source_text_ids")
                for value in (primary.get(field) or [])
                if str(value or "").strip()
            }
            if len(primary_sources) < 2:
                continue
            if "same_balloon_fragment_merged" not in {str(flag) for flag in primary.get("qa_flags") or []}:
                continue
            for sibling in visible_layers:
                if sibling is primary:
                    continue
                if not (_layer_identity_tokens(sibling) & primary_sources):
                    continue
                if not _layers_are_distinct_dark_bubble_lobes(primary, sibling):
                    continue
                split = _split_dark_lobe_contaminated_translation(primary, sibling)
                if split is None:
                    continue
                primary_text, sibling_text = split
                _set_layer_text_payload(primary, primary_text)
                if not _layer_has_translated_payload(sibling):
                    _set_layer_text_payload(sibling, sibling_text)
                _reset_layer_source_ids_to_self(primary)
                _reset_layer_source_ids_to_self(sibling)
                _split_shared_dark_lobe_safe_areas(primary, sibling)
                for repaired_layer in (primary, sibling):
                    if isinstance(repaired_layer.get("qa_flags"), list):
                        repaired_layer["qa_flags"] = [
                            flag
                            for flag in repaired_layer.get("qa_flags") or []
                            if str(flag) != "same_balloon_fragment_merged"
                        ]
                _merge_layer_qa_flags(primary, ["distinct_dark_lobe_payload_merge_repaired"])
                _merge_layer_qa_flags(sibling, ["distinct_dark_lobe_payload_merge_repaired"])
                sibling["visible"] = True
                if str(sibling.get("render_policy") or "").strip().lower() == "merged_into_primary":
                    sibling["render_policy"] = "normal"
                if str(sibling.get("route_action") or "").strip().lower() == "merged_into_primary":
                    sibling["route_action"] = "translate_inpaint_render"
                repaired += 1
                break
    repaired += _finalize_distinct_dark_lobe_project_geometry(project_layers)
    return repaired


def _candidate_text_matches_current_layer_payload(layer: dict, candidate: dict) -> bool:
    layer_norm = _normalized_merge_text(_translated_text_for_merge(layer))
    candidate_norm = _normalized_merge_text(_candidate_text_payload(candidate))
    if not layer_norm or not candidate_norm:
        return False
    return candidate_norm in layer_norm or layer_norm in candidate_norm


def _dedupe_repeated_project_text_prefixes(project_layers: list[dict]) -> int:
    repaired = 0
    for layer in project_layers:
        if not isinstance(layer, dict) or layer.get("visible", True) is False:
            continue
        text = _translated_text_for_merge(layer)
        if "\n" not in text:
            continue
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        cleaned: list[str] = []
        changed = False
        for line in lines:
            line_norm = _normalized_merge_text(line)
            if not line_norm:
                continue
            if cleaned:
                prev = cleaned[-1]
                prev_norm = _normalized_merge_text(prev)
                if prev_norm and line_norm.startswith(prev_norm) and line_norm != prev_norm:
                    suffix = line[len(prev):].strip(" ,.;:!?")
                    if suffix:
                        cleaned.append(suffix)
                        changed = True
                        continue
                if prev_norm == line_norm:
                    changed = True
                    continue
            cleaned.append(line)
        if changed and cleaned:
            _set_layer_text_payload(layer, "\n".join(cleaned))
            repaired += 1
    return repaired


def _repair_project_split_lobe_text_payloads(project_layers: list[dict], candidates: list[dict]) -> int:
    """Keep split visual-lobe text in the layer that owns that lobe.

    The renderer can split one OCR item into two visual lobes. If the lower
    lobe belongs to a neighboring balloon, the project layer for the original
    OCR must not keep the full translation while the neighbor also receives the
    lower text.
    """

    repaired = 0
    layers_by_band: dict[str, list[dict]] = {}
    candidates_by_band: dict[str, list[dict]] = {}
    for layer in project_layers:
        if not isinstance(layer, dict):
            continue
        band_id = str(layer.get("band_id") or "").strip() or (_trace_band_id(str(layer.get("trace_id") or "")) or "")
        if band_id:
            layers_by_band.setdefault(band_id, []).append(layer)
    for candidate in candidates:
        if not isinstance(candidate, dict) or _is_low_containment_suppressed_fragment(candidate):
            continue
        if not _candidate_text_payload(candidate):
            continue
        band_id = str(candidate.get("band_id") or "").strip() or (_trace_band_id(str(candidate.get("trace_id") or "")) or "")
        if band_id:
            candidates_by_band.setdefault(band_id, []).append(candidate)

    def _select_geometry_matched_candidates(band_layers: list[dict], band_candidates: list[dict]) -> list[dict]:
        if len(band_candidates) < 2:
            return band_candidates
        by_space: dict[str, list[dict]] = {}
        for candidate in band_candidates:
            space = str(candidate.get("coordinate_space") or "").strip().lower() or "unknown"
            by_space.setdefault(space, []).append(candidate)
        if len(by_space) < 2:
            return band_candidates
        layer_bboxes = [
            bbox
            for layer in band_layers
            for bbox in [_layer_merge_order_bbox(layer)]
            if bbox is not None
        ]
        if not layer_bboxes:
            return band_candidates
        scored: list[tuple[int, int, int, str, list[dict]]] = []
        for space, space_candidates in by_space.items():
            overlap_count = 0
            overlap_area = 0
            distance_score = 0
            for candidate in space_candidates:
                candidate_bbox = _candidate_layout_bbox(candidate)
                if candidate_bbox is None:
                    continue
                cx = (int(candidate_bbox[0]) + int(candidate_bbox[2])) // 2
                cy = (int(candidate_bbox[1]) + int(candidate_bbox[3])) // 2
                best_overlap = 0
                best_distance = 10**9
                for layer_bbox in layer_bboxes:
                    overlap = _bbox_intersection_area4(candidate_bbox, layer_bbox)
                    best_overlap = max(best_overlap, overlap)
                    lx = (int(layer_bbox[0]) + int(layer_bbox[2])) // 2
                    ly = (int(layer_bbox[1]) + int(layer_bbox[3])) // 2
                    best_distance = min(best_distance, abs(cx - lx) + abs(cy - ly))
                if best_overlap > 0:
                    overlap_count += 1
                    overlap_area += best_overlap
                distance_score -= best_distance
            scored.append((overlap_count, overlap_area, distance_score, space, space_candidates))
        best = max(scored, key=lambda item: (item[0], item[1], item[2], 1 if item[3] == "page" else 0))
        if best[0] <= 0 and best[1] <= 0:
            return band_candidates
        return best[4]

    for band_id, band_candidates in candidates_by_band.items():
        band_layers = layers_by_band.get(band_id) or []
        if not band_layers:
            continue
        band_candidates = _select_geometry_matched_candidates(band_layers, band_candidates)
        by_trace: dict[str, list[dict]] = {}
        for candidate in band_candidates:
            source_traces = _debug_identity_values(candidate, ("source_trace_ids", "_source_trace_ids"))
            source_text_ids = _debug_identity_values(candidate, ("source_text_ids", "_source_text_ids"))
            if len(source_traces) > 1 or len(source_text_ids) > 1:
                continue
            trace_id = _candidate_direct_trace_id(candidate)
            if trace_id:
                by_trace.setdefault(trace_id, []).append(candidate)
        for trace_id, trace_candidates in by_trace.items():
            unique_candidates: list[dict] = []
            seen: set[tuple[str, tuple[int, int, int, int] | None]] = set()
            for candidate in trace_candidates:
                key = (_normalized_merge_text(_candidate_text_payload(candidate)), tuple(_candidate_layout_bbox(candidate) or []))
                if key in seen:
                    continue
                seen.add(key)
                unique_candidates.append(candidate)
            if len(unique_candidates) < 2:
                continue
            unique_candidates.sort(key=lambda item: ((_candidate_layout_bbox(item) or [0, 0, 0, 0])[1], (_candidate_layout_bbox(item) or [0, 0, 0, 0])[0]))
            owner_layers = [
                layer
                for layer in band_layers
                if layer.get("visible", True) is not False
                and str(layer.get("render_policy") or "").strip().lower() != "merged_into_primary"
                and _layer_matches_trace_id(layer, trace_id)
            ]
            if not owner_layers:
                continue
            owner = owner_layers[0]
            owner_text_norm = _normalized_merge_text(_translated_text_for_merge(owner))
            first_text = _candidate_text_payload(unique_candidates[0])
            first_norm = _normalized_merge_text(first_text)
            later_candidate_norms = [
                _normalized_merge_text(_candidate_text_payload(candidate))
                for candidate in unique_candidates[1:]
                if _normalized_merge_text(_candidate_text_payload(candidate))
            ]
            owner_contains_later_split = any(
                later_norm in owner_text_norm for later_norm in later_candidate_norms
            )
            if first_text and first_norm and owner_text_norm != first_norm and (
                first_norm in owner_text_norm or owner_contains_later_split
            ):
                _set_layer_text_payload(owner, first_text)
                owner["source_trace_ids"] = [trace_id]
                owner["_source_trace_ids"] = [trace_id]
                repaired += 1

            for split_candidate in unique_candidates[1:]:
                split_text = _candidate_text_payload(split_candidate)
                split_norm = _normalized_merge_text(split_text)
                if not split_norm:
                    continue
                neighbor_candidates = [
                    candidate
                    for candidate in band_candidates
                    if _candidate_direct_trace_id(candidate)
                    and _candidate_direct_trace_id(candidate) != trace_id
                    and _candidate_centers_overlap_or_touch(split_candidate, candidate)
                ]
                if not neighbor_candidates:
                    continue
                neighbor_candidates.sort(
                    key=lambda item: (
                        (_candidate_layout_bbox(item) or [0, 0, 0, 0])[1],
                        (_candidate_layout_bbox(item) or [0, 0, 0, 0])[0],
                    )
                )
                neighbor = neighbor_candidates[0]
                neighbor_trace = _candidate_direct_trace_id(neighbor)
                neighbor_layers = [
                    layer
                    for layer in band_layers
                    if layer.get("visible", True) is not False
                    and str(layer.get("render_policy") or "").strip().lower() != "merged_into_primary"
                    and _layer_matches_trace_id(layer, neighbor_trace)
                ]
                if not neighbor_layers:
                    continue
                target_layer = neighbor_layers[0]
                target_text = _translated_text_for_merge(target_layer)
                target_norm = _normalized_merge_text(target_text)
                if split_norm in target_norm:
                    continue
                # The split candidate comes from an earlier OCR phrase that was
                # visually divided across lobes; preserve semantic order by
                # prepending that dependent phrase to the neighboring balloon.
                ordered_parts = [
                    part
                    for part in (split_text, _candidate_text_payload(neighbor))
                    if part
                ]
                combined = "\n".join(_unique_preserve_order(ordered_parts))
                if combined and combined != target_text:
                    _set_layer_text_payload(target_layer, combined)
                    merged_trace_ids = _unique_preserve_order(
                        [
                            trace_id,
                            neighbor_trace,
                            *(target_layer.get("source_trace_ids") or target_layer.get("_source_trace_ids") or []),
                        ]
                    )
                    target_layer["source_trace_ids"] = merged_trace_ids
                    target_layer["_source_trace_ids"] = list(merged_trace_ids)
                    _merge_layer_qa_flags(target_layer, ["same_balloon_fragment_merged"])
                    repaired += 1
    return repaired


def _debug_mask_bboxes_from_dir(mask_dir: Path, layer: dict | None = None) -> dict[str, list[int] | dict] | None:
    decision_path = mask_dir / "mask_decision.json"
    if not mask_dir.exists() or not decision_path.exists():
        return None
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    bubble_source = str(decision.get("bubble_mask_source") or "").strip().lower()
    if bubble_source == "rejected_derived_bubble_mask":
        return {
            "decision": decision,
            "rejection_reason": str(decision.get("bubble_mask_rejection_reason") or "rejected_derived_bubble_mask"),
        }
    if bubble_source in {"derived_white_crop", "image_white_region", "derived_rectangular_balloon", "outline_seeded_contour"}:
        return {
            "decision": decision,
            "rejection_reason": "untrusted_derived_bubble_mask",
        }
    if (
        bubble_source in {"image_white_bubble_mask", "image_rect_bubble_mask", "image_contour_bubble_mask"}
        and bool(decision.get("used_real_bubble_mask"))
    ):
        return {
            "decision": decision,
            "rejection_reason": "image_mask_misreported_as_real",
        }
    if bubble_source in {
        "image_white_bubble_mask",
        "image_rect_bubble_mask",
        "image_contour_bubble_mask",
        "image_dark_bubble_mask",
        "image_dark_panel_mask",
    }:
        if not bool(decision.get("used_image_bubble_mask")):
            return {
                "decision": decision,
                "rejection_reason": "missing_image_bubble_mask",
            }
    if bubble_source in {"debug_band_balloon_component", "bbox_fallback", "balloon_bbox_fallback"}:
        return {
            "decision": decision,
            "rejection_reason": "untrusted_fallback_bubble_mask",
        }
    if bubble_source not in {
        "real",
        "real_bubble_mask",
        "image_white_bubble_mask",
        "image_rect_bubble_mask",
        "image_contour_bubble_mask",
        "image_dark_bubble_mask",
        "image_dark_panel_mask",
    }:
        return None
    if bubble_source in {"real", "real_bubble_mask"} and not bool(decision.get("used_real_bubble_mask")):
        return {
            "decision": decision,
            "rejection_reason": "missing_real_bubble_mask",
        }
    if not bool(decision.get("used_balloon_clip")):
        return None
    text_ids = [str(value).strip() for value in (decision.get("text_ids") or []) if str(value).strip()]
    if len(set(text_ids)) > 1:
        return {
            "decision": decision,
            "rejection_reason": "multi_text_debug_mask_not_per_layer",
        }
    repair_reject_reason = _debug_mask_decision_repair_reject_reason(decision)
    if repair_reject_reason:
        return {
            "decision": decision,
            "rejection_reason": repair_reject_reason,
        }
    balloon = _debug_mask_image_bbox(mask_dir / "04_balloon_mask.png")
    inner = _debug_mask_image_bbox(mask_dir / "05_balloon_inner_mask.png") or balloon
    raw_text_mask = _debug_mask_image_bbox(mask_dir / "07_raw_text_mask.png")
    expanded_text_mask = _debug_mask_image_bbox(mask_dir / "08_expanded_text_mask.png")
    final_text_mask = _debug_mask_image_bbox(mask_dir / "09_final_inpaint_mask.png")
    component_text_mask = (
        _debug_mask_component_bbox(mask_dir / "09_final_inpaint_mask.png", layer)
        or _debug_mask_component_bbox(mask_dir / "08_expanded_text_mask.png", layer)
        or _debug_mask_component_bbox(mask_dir / "07_raw_text_mask.png", layer)
    )
    text_mask = final_text_mask or expanded_text_mask
    if balloon is None or inner is None or text_mask is None:
        return None
    if _bbox_intersection_area4(balloon, text_mask) <= 0:
        return None
    return {
        "decision": decision,
        "balloon_bbox": balloon,
        "inner_bbox": inner,
        "raw_text_mask_bbox": raw_text_mask,
        "expanded_text_mask_bbox": expanded_text_mask,
        "final_text_mask_bbox": final_text_mask,
        "component_text_mask_bbox": component_text_mask,
        "text_mask_bbox": text_mask,
    }


def _debug_text_mask_bboxes(debug_root: Path, band_id: str, layer: dict) -> dict[str, list[int] | dict] | None:
    base = debug_root / "06_mask_segmentation" / band_id / "per_text"
    if not base.exists():
        return None
    candidates: list[str] = []
    for key in ("id", "text_id", "trace_id"):
        raw = str(layer.get(key) or "").strip()
        if not raw:
            continue
        if key == "trace_id" and "@" in raw:
            raw = raw.split("@", 1)[0]
        if raw and raw not in candidates:
            candidates.append(raw)
    for key in ("source_text_ids", "_source_text_ids"):
        for raw_value in layer.get(key) or []:
            raw = str(raw_value or "").strip()
            if raw and raw not in candidates:
                candidates.append(raw)
    for candidate in candidates:
        bboxes = _debug_mask_bboxes_from_dir(base / _debug_mask_path_segment(candidate), layer)
        if bboxes:
            bboxes = dict(bboxes)
            bboxes["mask_debug_scope"] = "per_text"
            bboxes["mask_debug_text_id"] = candidate
            return bboxes
    return None


def _debug_band_mask_bboxes(debug_root: Path, band_id: str) -> dict[str, list[int] | dict] | None:
    bboxes = _debug_mask_bboxes_from_dir(debug_root / "06_mask_segmentation" / band_id)
    if bboxes:
        bboxes = dict(bboxes)
        bboxes["mask_debug_scope"] = "band"
    return bboxes


def _debug_mask_page_offset_for_layer(layer: dict, local_text_mask_bbox: list[int]) -> tuple[int, int] | None:
    local_center_x = (int(local_text_mask_bbox[0]) + int(local_text_mask_bbox[2])) / 2.0
    candidates: list[tuple[float, int, int]] = []
    for key in ("target_bbox", "source_bbox", "bbox", "layout_bbox", "text_pixel_bbox"):
        ref = _optional_bbox4(layer.get(key))
        if ref is None:
            continue
        ref_w = max(1, int(ref[2]) - int(ref[0]))
        local_w = max(1, int(local_text_mask_bbox[2]) - int(local_text_mask_bbox[0]))
        width_delta = abs(ref_w - local_w) / float(max(ref_w, local_w))
        ref_center_x = (int(ref[0]) + int(ref[2])) / 2.0
        center_delta = abs(ref_center_x - local_center_x)
        if width_delta > 0.55 or center_delta > 80:
            continue
        dx = int(round(ref_center_x - local_center_x))
        dy = int(ref[1]) - int(local_text_mask_bbox[1])
        score = width_delta + center_delta / 1000.0
        candidates.append((score, dx, dy))
    if not candidates:
        return None
    _, dx, dy = min(candidates, key=lambda item: item[0])
    if abs(dx) <= 12:
        dx = 0
    return dx, dy


def _map_debug_local_bbox_to_page(
    bbox: list[int] | None,
    *,
    dx: int,
    dy: int,
    page_size: tuple[int, int] | None,
    clamp: bool = True,
) -> list[int] | None:
    if bbox is None:
        return None
    mapped = [int(bbox[0]) + dx, int(bbox[1]) + dy, int(bbox[2]) + dx, int(bbox[3]) + dy]
    if clamp and page_size:
        page_width, page_height = page_size
        mapped = [
            max(0, min(int(page_width), mapped[0])),
            max(0, min(int(page_height), mapped[1])),
            max(0, min(int(page_width), mapped[2])),
            max(0, min(int(page_height), mapped[3])),
        ]
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        return None
    return mapped


def _clamp_page_bbox(bbox: list[int] | None, page_size: tuple[int, int] | None) -> list[int] | None:
    if bbox is None:
        return None
    if not page_size:
        return list(bbox)
    page_width, page_height = page_size
    clamped = [
        max(0, min(int(page_width), int(bbox[0]))),
        max(0, min(int(page_height), int(bbox[1]))),
        max(0, min(int(page_width), int(bbox[2]))),
        max(0, min(int(page_height), int(bbox[3]))),
    ]
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    return clamped


def _should_replace_project_bubble_bbox_with_debug_mask(layer: dict, candidate_bbox: list[int]) -> bool:
    current = _optional_bbox4(layer.get("bubble_mask_bbox")) or _optional_bbox4(layer.get("balloon_bbox"))
    if current is None:
        return True
    current_area = _bbox_area4(current)
    candidate_area = _bbox_area4(candidate_bbox)
    if current_area <= 0 or candidate_area <= 0:
        return True
    if current_area >= int(candidate_area * 1.55):
        return True
    if candidate_area >= int(current_area * 1.55):
        overlap = _bbox_intersection_area4(current, candidate_bbox)
        if overlap >= int(current_area * 0.80):
            return True
    current_w = int(current[2]) - int(current[0])
    candidate_w = int(candidate_bbox[2]) - int(candidate_bbox[0])
    if current[0] <= 2 and current_w >= int(candidate_w * 1.45):
        return True
    return _bbox_overlap_ratio4(current, candidate_bbox) < 0.45


def _restored_safe_text_box_is_usable(bbox: list[int] | None) -> bool:
    if bbox is None:
        return False
    width = int(bbox[2]) - int(bbox[0])
    height = int(bbox[3]) - int(bbox[1])
    return width >= 20 and height >= 12


def _partition_dark_connected_lobe_safe_boxes_from_components(project_data: dict) -> int:
    if not isinstance(project_data, dict):
        return 0
    updated = 0
    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        by_band: dict[str, list[dict]] = {}
        for layer in _project_page_text_layers(page):
            if not isinstance(layer, dict):
                continue
            band_id = str(layer.get("band_id") or "").strip()
            if not band_id:
                continue
            flags = {str(flag) for flag in layer.get("qa_flags") or []}
            source = str(layer.get("bubble_mask_source") or layer.get("balloon_mask_source") or "").strip().lower()
            if source != "image_dark_bubble_mask" or "source_text_mask_bbox_from_inpaint_component" not in flags:
                continue
            component = _optional_bbox4(layer.get("source_text_mask_bbox") or layer.get("_source_text_mask_bbox"))
            safe = _optional_bbox4(layer.get("safe_text_box"))
            if component is None or safe is None:
                continue
            by_band.setdefault(band_id, []).append(layer)
        for layers in by_band.values():
            if len(layers) < 2:
                continue
            layers.sort(
                key=lambda item: (
                    (_optional_bbox4(item.get("source_text_mask_bbox") or item.get("_source_text_mask_bbox")) or [0, 0, 0, 0])[0],
                    (_optional_bbox4(item.get("source_text_mask_bbox") or item.get("_source_text_mask_bbox")) or [0, 0, 0, 0])[1],
                )
            )
            for left, right in zip(layers, layers[1:]):
                left_component = _optional_bbox4(left.get("source_text_mask_bbox") or left.get("_source_text_mask_bbox"))
                right_component = _optional_bbox4(right.get("source_text_mask_bbox") or right.get("_source_text_mask_bbox"))
                left_safe = _optional_bbox4(left.get("safe_text_box"))
                right_safe = _optional_bbox4(right.get("safe_text_box"))
                if left_component is None or right_component is None or left_safe is None or right_safe is None:
                    continue
                if right_component[0] < left_component[0]:
                    continue
                split_x = int(round((int(left_component[2]) + int(right_component[0])) / 2.0))
                if split_x <= left_safe[0] + 20 or split_x >= right_safe[2] - 20:
                    continue
                changed = False
                if left_safe[2] > split_x:
                    left_safe[2] = split_x
                    changed = True
                if right_safe[0] < split_x:
                    right_safe[0] = split_x
                    changed = True
                if changed:
                    for layer, safe in ((left, left_safe), (right, right_safe)):
                        if not _restored_safe_text_box_is_usable(safe):
                            continue
                        for key in (
                            "target_bbox",
                            "safe_text_box",
                            "_debug_safe_text_box",
                            "position_bbox",
                            "capacity_bbox",
                            "layout_safe_bbox",
                        ):
                            layer[key] = list(safe)
                        # Stale render boxes from the earlier overlapped layout
                        # must not keep pulling text across the lobe boundary.
                        layer["render_bbox"] = list(safe)
                        layer["_debug_render_bbox"] = list(safe)
                        for stale_key in (
                            "render_layout_contract",
                            "_render_layout_contract_hydrated_from_debug",
                            "fit_status",
                            "layout_fit_result",
                        ):
                            layer.pop(stale_key, None)
                        layer["_render_bbox_from_repaired_safe_text_box"] = True
                        _merge_layer_qa_flags(layer, ["dark_connected_component_safe_partition"])
                    updated += 1
    return updated


def _repair_project_bubble_bboxes_from_debug_masks(project_data: dict) -> dict:
    audit = {"layers_checked": 0, "layers_repaired": 0, "missing_debug_root": False}
    debug_root = _debug_root_from_project(project_data)
    if debug_root is None:
        audit["missing_debug_root"] = True
        return audit
    if not isinstance(project_data, dict):
        return audit
    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        page_size = _project_page_image_size(project_data, page)
        for layer in _project_page_text_layers(page):
            if not isinstance(layer, dict):
                continue
            audit["layers_checked"] += 1
            band_id = str(layer.get("band_id") or "").strip()
            if not band_id:
                continue
            bboxes = _debug_text_mask_bboxes(debug_root, band_id, layer) or _debug_band_mask_bboxes(debug_root, band_id)
            if not bboxes:
                continue
            rejection_reason = bboxes.get("rejection_reason") if isinstance(bboxes, dict) else None
            if rejection_reason:
                layer["layout_safe_reason"] = "debug_derived_bubble_mask_rejected"
                layer["_debug_derived_bubble_bbox_rejected"] = str(rejection_reason)
                _merge_layer_qa_flags(layer, ["debug_derived_bubble_mask_rejected"])
                continue
            local_text_mask = (
                bboxes.get("component_text_mask_bbox")
                if str(bboxes.get("mask_debug_scope") or "") == "per_text"
                and isinstance(bboxes.get("component_text_mask_bbox"), list)
                else bboxes.get("text_mask_bbox")
            )
            if not isinstance(local_text_mask, list):
                continue
            offset = _debug_mask_page_offset_for_layer(layer, local_text_mask)
            if offset is None:
                continue
            dx, dy = offset
            balloon_unclamped = _map_debug_local_bbox_to_page(
                bboxes.get("balloon_bbox") if isinstance(bboxes.get("balloon_bbox"), list) else None,
                dx=dx,
                dy=dy,
                page_size=None,
                clamp=False,
            )
            inner_unclamped = _map_debug_local_bbox_to_page(
                bboxes.get("inner_bbox") if isinstance(bboxes.get("inner_bbox"), list) else None,
                dx=dx,
                dy=dy,
                page_size=None,
                clamp=False,
            )
            balloon_bbox = _clamp_page_bbox(balloon_unclamped, page_size)
            inner_bbox = _clamp_page_bbox(inner_unclamped, page_size)
            source_text_mask_local = None
            if str(bboxes.get("mask_debug_scope") or "") == "per_text":
                for key in (
                    "component_text_mask_bbox",
                    "raw_text_mask_bbox",
                    "final_text_mask_bbox",
                    "expanded_text_mask_bbox",
                    "text_mask_bbox",
                ):
                    candidate_mask = bboxes.get(key)
                    if isinstance(candidate_mask, list):
                        source_text_mask_local = candidate_mask
                        break
            source_text_mask_bbox = _clamp_page_bbox(
                _map_debug_local_bbox_to_page(
                    source_text_mask_local,
                    dx=dx,
                    dy=dy,
                    page_size=None,
                    clamp=False,
                ),
                page_size,
            )
            if balloon_bbox is None or inner_bbox is None:
                continue
            needs_unclamped_upgrade = (
                str(layer.get("layout_safe_reason") or "") == "debug_derived_bubble_mask_bbox"
                and _optional_bbox4(layer.get("_bubble_mask_bbox_unclamped")) is None
            )
            if not needs_unclamped_upgrade and not _should_replace_project_bubble_bbox_with_debug_mask(layer, balloon_bbox):
                checked_source_text_mask_bbox = _preserve_compact_text_anchor_against_debug_mask(layer, source_text_mask_bbox)
                if checked_source_text_mask_bbox is not None:
                    layer["source_text_mask_bbox"] = checked_source_text_mask_bbox
                    layer["_source_text_mask_bbox"] = checked_source_text_mask_bbox
                    source_name = (
                        "per_text_component"
                        if isinstance(bboxes.get("component_text_mask_bbox"), list)
                        and source_text_mask_local == bboxes.get("component_text_mask_bbox")
                        else str(bboxes.get("mask_debug_text_id") or "per_text")
                    )
                    layer["_source_text_mask_bbox_source"] = source_name
                    flags = ["source_text_mask_bbox_from_debug"]
                    if source_name == "per_text_component":
                        flags.append("source_text_mask_bbox_from_inpaint_component")
                    _merge_layer_qa_flags(layer, flags)
                continue
            safe_unclamped = _inset_bbox_for_restored_balloon_safe_text(inner_unclamped or inner_bbox)
            safe_bbox = _clamp_page_bbox(safe_unclamped, page_size)
            if safe_bbox is None:
                safe_bbox = _inset_bbox_for_restored_balloon_safe_text(inner_bbox)
            if not _restored_safe_text_box_is_usable(safe_bbox):
                layer["layout_safe_reason"] = "debug_derived_bubble_mask_rejected"
                layer["_debug_derived_bubble_bbox_rejected"] = "safe_text_box_degenerate"
                _merge_layer_qa_flags(layer, ["debug_derived_bubble_mask_rejected"])
                continue
            layer["balloon_bbox"] = balloon_bbox
            layer["bubble_mask_bbox"] = balloon_bbox
            layer["bubble_inner_bbox"] = inner_bbox
            layer["_bubble_mask_bbox_unclamped"] = balloon_unclamped
            layer["_bubble_inner_bbox_unclamped"] = inner_unclamped
            layer["_safe_text_box_unclamped"] = safe_unclamped
            decision = bboxes.get("decision") if isinstance(bboxes.get("decision"), dict) else {}
            layer["bubble_mask_source"] = str(decision.get("bubble_mask_source") or "real_bubble_mask")
            layer["layout_safe_bbox"] = safe_bbox
            layer["layout_safe_reason"] = "debug_derived_bubble_mask_unclamped"
            layer["safe_text_box"] = safe_bbox
            layer["_debug_safe_text_box"] = safe_bbox
            checked_source_text_mask_bbox = _preserve_compact_text_anchor_against_debug_mask(layer, source_text_mask_bbox)
            if checked_source_text_mask_bbox is not None:
                layer["source_text_mask_bbox"] = checked_source_text_mask_bbox
                layer["_source_text_mask_bbox"] = checked_source_text_mask_bbox
                source_name = (
                    "per_text_component"
                    if isinstance(bboxes.get("component_text_mask_bbox"), list)
                    and source_text_mask_local == bboxes.get("component_text_mask_bbox")
                    else str(bboxes.get("mask_debug_text_id") or "per_text")
                )
                layer["_source_text_mask_bbox_source"] = source_name
                flags = ["source_text_mask_bbox_from_debug"]
                if source_name == "per_text_component":
                    flags.append("source_text_mask_bbox_from_inpaint_component")
                _merge_layer_qa_flags(layer, flags)
            layer["render_bbox"] = safe_bbox
            layer["_debug_render_bbox"] = safe_bbox
            layer["_render_bbox_from_repaired_safe_text_box"] = True
            layer["_debug_derived_bubble_bbox_repaired"] = True
            _merge_layer_qa_flags(layer, ["safe_text_box_recomputed"])
            if isinstance(layer.get("qa_flags"), list):
                stale_flags = {"rejected_derived_bubble_mask", "tiny_bubble_inner_bbox_rejected"}
                layer["qa_flags"] = [flag for flag in layer["qa_flags"] if str(flag) not in stale_flags]
            audit["layers_repaired"] += 1
    audit["component_safe_partitions"] = _partition_dark_connected_lobe_safe_boxes_from_components(project_data)
    return audit


def _clamp_project_render_geometry_to_page(project_data: dict) -> dict:
    audit = {"pages_checked": 0, "layers_checked": 0, "bbox_clamped_count": 0}
    if not isinstance(project_data, dict):
        return audit
    bbox_keys = (
        "target_bbox",
        "safe_text_box",
        "_debug_safe_text_box",
        "render_bbox",
        "position_bbox",
        "capacity_bbox",
        "layout_safe_bbox",
        "balloon_bbox",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
    )
    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        size = _project_page_image_size(project_data, page)
        if not size:
            continue
        page_width, page_height = size
        if page_width <= 0 or page_height <= 0:
            continue
        audit["pages_checked"] += 1
        for layer in _project_page_text_layers(page):
            if not isinstance(layer, dict):
                continue
            audit["layers_checked"] += 1
            for key in bbox_keys:
                bbox = _optional_bbox4(layer.get(key))
                if bbox is None:
                    continue
                clamped = [
                    max(0, min(int(page_width), int(bbox[0]))),
                    max(0, min(int(page_height), int(bbox[1]))),
                    max(0, min(int(page_width), int(bbox[2]))),
                    max(0, min(int(page_height), int(bbox[3]))),
                ]
                if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
                    continue
                if clamped != bbox:
                    layer[key] = clamped
                    audit["bbox_clamped_count"] += 1
    return audit


def _repair_project_real_bubble_body_safe_areas(project_data: dict) -> dict:
    audit = {"layers_checked": 0, "safe_area_repaired_count": 0}
    if not isinstance(project_data, dict):
        return audit
    real_bubble_sources = {
        "real",
        "real_bubble_mask",
        "image_contour_bubble_mask",
        "image_white_bubble_mask",
        "debug_band_balloon_component",
    }
    for layer in _iter_project_text_layers(project_data):
        if not isinstance(layer, dict):
            continue
        audit["layers_checked"] += 1
        target_bbox = _optional_bbox4(layer.get("target_bbox"))
        bubble_bbox = _optional_bbox4(layer.get("bubble_mask_bbox"))
        balloon_bbox = _optional_bbox4(layer.get("balloon_bbox"))
        current_safe = _optional_bbox4(layer.get("safe_text_box")) or _optional_bbox4(layer.get("_debug_safe_text_box"))
        current_render = _optional_bbox4(layer.get("render_bbox")) or _optional_bbox4(layer.get("_debug_render_bbox"))
        bubble_source = str(layer.get("bubble_mask_source") or "").strip().lower()
        qa_flags = {str(flag) for flag in layer.get("qa_flags") or []}
        if (
            "same_balloon_fragment_merged" in qa_flags
            and str(layer.get("render_policy") or "").strip().lower() != "merged_into_primary"
            and bubble_bbox is not None
            and bubble_source in real_bubble_sources
        ):
            metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
            try:
                render_containment = float(metrics.get("render_balloon_containment"))
            except (TypeError, ValueError):
                render_containment = 0.0
            compact_text = _normalized_merge_text(_translated_text_for_merge(layer))
            if (
                current_render is not None
                and len(compact_text) <= 24
                and str(layer.get("fit_status") or "").strip().lower() in {"", "ok", "pass"}
                and render_containment >= 0.85
                and _bbox_contains4_margin(bubble_bbox, current_render, margin=6)
            ):
                continue
            current_area = _bbox_area4(current_safe) if current_safe is not None else 0
            bubble_area = _bbox_area4(bubble_bbox)
            current_outside_real_bubble = (
                current_safe is not None
                and not _bbox_contains4_margin(bubble_bbox, current_safe, margin=4)
                and (
                    _bbox_overlap_ratio4(current_safe, bubble_bbox) < 0.82
                    or current_area > int(bubble_area * 1.18)
                )
            )
            if bubble_area > 0 and (
                current_area <= 0
                or current_area < int(bubble_area * 0.72)
                or current_outside_real_bubble
            ):
                safe_bbox = _inset_bbox_for_restored_balloon_safe_text(bubble_bbox)
                if _restored_safe_text_box_is_usable(safe_bbox):
                    layer["target_bbox"] = list(bubble_bbox)
                    layer["balloon_bbox"] = list(bubble_bbox)
                    layer["bubble_mask_bbox"] = list(bubble_bbox)
                    layer["layout_safe_bbox"] = safe_bbox
                    layer["layout_safe_reason"] = "merged_real_bubble_mask_bbox"
                    layer["safe_text_box"] = safe_bbox
                    layer["_debug_safe_text_box"] = safe_bbox
                    layer["position_bbox"] = safe_bbox
                    layer["capacity_bbox"] = safe_bbox
                    layer["render_bbox"] = safe_bbox
                    layer["_debug_render_bbox"] = safe_bbox
                    layer["_render_bbox_from_repaired_safe_text_box"] = True
                    layer["_real_bubble_body_safe_area_repaired"] = True
                    layer["qa_flags"] = [
                        flag
                        for flag in list(layer.get("qa_flags") or [])
                        if str(flag)
                        not in {
                            "TEXT_OVERFLOW",
                            "fit_below_minimum_legible",
                            "mask_outside_balloon_critical",
                            "render_on_art_suspected",
                        }
                    ]
                    _merge_layer_qa_flags(layer, ["safe_text_box_recomputed"])
                    audit["safe_area_repaired_count"] += 1
                    continue
        if target_bbox is None or bubble_bbox is None or balloon_bbox is None:
            continue
        target_area = max(1, (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1]))
        bubble_area = max(1, (bubble_bbox[2] - bubble_bbox[0]) * (bubble_bbox[3] - bubble_bbox[1]))
        balloon_area = max(1, (balloon_bbox[2] - balloon_bbox[0]) * (balloon_bbox[3] - balloon_bbox[1]))
        if target_area <= 0 or bubble_area <= 0 or balloon_area <= 0:
            continue
        if _bbox_intersection_area4(target_bbox, bubble_bbox) < int(min(target_area, bubble_area) * 0.82):
            continue
        if _bbox_intersection_area4(bubble_bbox, balloon_bbox) < int(balloon_area * 0.88):
            continue
        inner_bbox = _optional_bbox4(layer.get("bubble_inner_bbox"))
        if (
            inner_bbox is not None
            and bubble_source in real_bubble_sources
            and _bbox_intersection_area4(inner_bbox, bubble_bbox) >= int(_bbox_area4(inner_bbox) * 0.85)
            and (
                current_safe is None
                or not _bbox_contains4_margin(inner_bbox, current_safe, margin=4)
                or _bbox_overlap_ratio4(inner_bbox, current_safe) < 0.85
            )
        ):
            safe_bbox = _inset_bbox_for_restored_balloon_safe_text(inner_bbox)
            if _restored_safe_text_box_is_usable(safe_bbox):
                layer["target_bbox"] = list(bubble_bbox)
                layer["balloon_bbox"] = list(bubble_bbox)
                layer["bubble_mask_bbox"] = list(bubble_bbox)
                layer["layout_safe_bbox"] = safe_bbox
                layer["layout_safe_reason"] = "real_bubble_inner_bbox"
                layer["safe_text_box"] = safe_bbox
                layer["_debug_safe_text_box"] = safe_bbox
                layer["position_bbox"] = safe_bbox
                layer["capacity_bbox"] = safe_bbox
                layer["render_bbox"] = safe_bbox
                layer["_debug_render_bbox"] = safe_bbox
                layer["_render_bbox_from_repaired_safe_text_box"] = True
                layer["_real_bubble_body_safe_area_repaired"] = True
                _merge_layer_qa_flags(layer, ["safe_text_box_recomputed"])
                audit["safe_area_repaired_count"] += 1
            continue
        if bubble_source not in real_bubble_sources:
            continue
        if bubble_area < int(balloon_area * 1.18):
            continue
        if current_safe is not None and _bbox_contains4_margin(balloon_bbox, current_safe, margin=4):
            continue
        bw = max(1, int(balloon_bbox[2]) - int(balloon_bbox[0]))
        bh = max(1, int(balloon_bbox[3]) - int(balloon_bbox[1]))
        inset_x = max(12, int(round(bw * 0.10)))
        inset_y = max(10, int(round(bh * 0.15)))
        safe_bbox = [
            int(balloon_bbox[0]) + inset_x,
            int(balloon_bbox[1]) + inset_y,
            int(balloon_bbox[2]) - inset_x,
            int(balloon_bbox[3]) - inset_y,
        ]
        if safe_bbox[2] <= safe_bbox[0] or safe_bbox[3] <= safe_bbox[1]:
            continue
        layer["target_bbox"] = list(balloon_bbox)
        layer["balloon_bbox"] = list(balloon_bbox)
        layer["layout_safe_bbox"] = safe_bbox
        layer["layout_safe_reason"] = "real_bubble_body_bbox"
        layer["safe_text_box"] = safe_bbox
        layer["_debug_safe_text_box"] = safe_bbox
        layer["position_bbox"] = safe_bbox
        layer["capacity_bbox"] = safe_bbox
        layer["render_bbox"] = safe_bbox
        layer["_debug_render_bbox"] = safe_bbox
        layer["_render_bbox_from_repaired_safe_text_box"] = True
        layer["_real_bubble_body_safe_area_repaired"] = True
        _merge_layer_qa_flags(layer, ["safe_text_box_recomputed"])
        audit["safe_area_repaired_count"] += 1
    return audit


def _repair_page_space_text_layers_for_typeset(
    page_texts: list[dict],
    *,
    page_number: int,
) -> tuple[list[dict], dict]:
    page_stub = {"numero": page_number, "text_layers": page_texts}
    project_stub = {"paginas": [page_stub]}
    audit = _repair_project_real_bubble_body_safe_areas(project_stub)
    audit["auxiliary_bbox_scrubbed_count"] = _scrub_project_local_auxiliary_bboxes(project_stub)
    return page_stub["text_layers"], audit


def _render_candidate_score_for_layer(layer: dict, candidate: dict) -> float:
    render_bbox = _optional_bbox4(candidate.get("render_bbox"))
    safe_text_box = _optional_bbox4(candidate.get("safe_text_box")) or _optional_bbox4(candidate.get("_debug_safe_text_box"))
    if render_bbox is None or safe_text_box is None:
        return -1.0
    layer_refs = [
        _optional_bbox4(layer.get("render_bbox")),
        _optional_bbox4(layer.get("safe_text_box")),
        _valid_layer_text_pixel_bbox_for_render_match(layer),
        _optional_bbox4(layer.get("layout_bbox")),
        _optional_bbox4(layer.get("balloon_bbox")),
        _optional_bbox4(layer.get("bbox")),
        _optional_bbox4(layer.get("source_bbox")),
    ]
    candidate_refs = [
        _optional_bbox4(candidate.get("target_bbox")),
        render_bbox,
        safe_text_box,
        _optional_bbox4(candidate.get("bbox")),
    ]
    best = 0.0
    for left in candidate_refs:
        if left is None:
            continue
        for right in layer_refs:
            if right is None:
                continue
            best = max(best, _bbox_iou(left, right))
            if _bbox_contains4_margin(right, left, margin=8):
                best = max(best, 0.92)
            if _bbox_contains4_margin(left, right, margin=8):
                best = max(best, 0.84)
    if str(candidate.get("fit_status") or "").strip().lower() == "ok":
        best += 0.05
    if _render_candidate_has_explicit_layer_source(layer, candidate):
        best += 0.25
    return best


def _valid_layer_text_pixel_bbox_for_render_match(layer: dict) -> list[int] | None:
    text_bbox = _optional_bbox4(layer.get("text_pixel_bbox"))
    if text_bbox is None:
        return None
    reference_boxes = [
        _optional_bbox4(layer.get("bbox")),
        _optional_bbox4(layer.get("layout_bbox")),
        _optional_bbox4(layer.get("balloon_bbox")),
    ]
    reference_boxes = [bbox for bbox in reference_boxes if bbox is not None]
    if not reference_boxes:
        return text_bbox
    reference = _bbox_union_many(reference_boxes)
    if reference is None:
        return text_bbox
    ref_width = max(1, reference[2] - reference[0])
    ref_height = max(1, reference[3] - reference[1])
    text_width = max(1, text_bbox[2] - text_bbox[0])
    text_height = max(1, text_bbox[3] - text_bbox[1])
    if text_width > ref_width * 4 or text_height > ref_height * 4:
        return None
    if _bbox_overlap_ratio4(text_bbox, reference) < 0.10:
        return None
    return text_bbox


def _valid_layer_reference_bbox_for_render_match(layer: dict, key: str) -> list[int] | None:
    bbox = _optional_bbox4(layer.get(key))
    if bbox is None:
        return None
    peer_keys = [item for item in ("bbox", "layout_bbox", "balloon_bbox") if item != key]
    peer_boxes = [_optional_bbox4(layer.get(item)) for item in peer_keys]
    peer_boxes = [peer for peer in peer_boxes if peer is not None]
    if not peer_boxes:
        return bbox
    peer_union = _bbox_union_many(peer_boxes)
    if peer_union is None:
        return bbox
    peer_width = max(1, peer_union[2] - peer_union[0])
    peer_height = max(1, peer_union[3] - peer_union[1])
    bbox_width = max(1, bbox[2] - bbox[0])
    bbox_height = max(1, bbox[3] - bbox[1])
    if bbox_width > peer_width * 4 or bbox_height > peer_height * 4:
        return None
    if _bbox_overlap_ratio4(bbox, peer_union) < 0.10:
        return None
    return bbox


def _copy_group_sibling_render_metadata(project_data: dict, candidates: list[dict]) -> int:
    project_layers = list(_iter_project_text_layers(project_data))
    layers_by_identity = _layers_by_debug_identity(project_layers)
    hydrated = 0
    for candidate in candidates:
        if _is_low_containment_suppressed_fragment(candidate):
            continue
        source_ids = _debug_identity_values(candidate, ("source_text_ids", "_source_text_ids"))
        source_traces = _debug_identity_values(candidate, ("source_trace_ids", "_source_trace_ids"))
        if len({*source_ids, *source_traces}) < 2:
            continue
        band_id = str(candidate.get("band_id") or "").strip()
        matched_layers: list[dict] = []
        for identity_key in _debug_identity_keys_from_payload(candidate):
            for layer in layers_by_identity.get(identity_key, []):
                if band_id and str(layer.get("band_id") or "").strip() != band_id:
                    continue
                if str(layer.get("render_policy") or "") == "merged_into_primary":
                    continue
                if _is_low_containment_suppressed_fragment(layer):
                    continue
                if layer.get("visible") is False:
                    has_missing_render = _optional_bbox4(layer.get("render_bbox")) is None or _optional_bbox4(layer.get("safe_text_box")) is None
                    is_translated_layer = str(layer.get("route_action") or "").strip().startswith("translate_")
                    if not (has_missing_render and is_translated_layer):
                        continue
                if layer not in matched_layers:
                    matched_layers.append(layer)
        if len(matched_layers) < 2:
            continue
        if (
            _optional_bbox4(candidate.get("target_bbox")) is not None
            and all(layer.get("visible", True) is not False for layer in matched_layers)
            and (
                _layers_share_visual_merge_target(matched_layers)
                or _candidate_matches_reliable_merge_target(candidate, matched_layers)
            )
        ):
            source_texts_overlap = _source_linked_layers_have_text_overlap(matched_layers)
            if source_texts_overlap:
                primary = max(
                    matched_layers,
                    key=lambda item: _merge_primary_preference_score(
                        item,
                        _optional_bbox4(candidate.get("target_bbox")) or _optional_bbox4(candidate.get("render_bbox")),
                    ),
                )
            else:
                primary = max(
                    matched_layers,
                    key=lambda item: _merge_primary_preference_score(
                        item,
                        _optional_bbox4(candidate.get("target_bbox")) or _optional_bbox4(candidate.get("render_bbox")),
                    ),
                )
            candidate_for_primary = (
                candidate
                if _render_bbox_overlaps_layer_source_text(primary, candidate.get("render_bbox"))
                else _render_candidate_with_layer_coordinates(candidate, primary)
            )
            candidate_text = _translated_text_for_merge(candidate_for_primary)
            primary_text_norm = _normalized_merge_text(primary.get("translated") or primary.get("traduzido") or "")
            candidate_text_norm = _normalized_merge_text(candidate_text)
            candidate_matches_primary_text = bool(
                candidate_text_norm
                and primary_text_norm
                and (
                    primary_text_norm in candidate_text_norm
                    or candidate_text_norm in primary_text_norm
                )
            )
            if not candidate_matches_primary_text:
                continue
            candidate_extends_primary_text = bool(
                candidate_text
                and primary_text_norm
                and primary_text_norm in candidate_text_norm
                and len(candidate_text_norm) >= len(primary_text_norm)
            )
            should_apply_candidate_text = bool(source_texts_overlap or candidate_extends_primary_text)
            if not should_apply_candidate_text:
                continue
            if candidate_text and should_apply_candidate_text:
                primary["translated"] = candidate_text
                primary["traduzido"] = candidate_text
            merged_trace_ids = _unique_preserve_order(
                [
                    *(primary.get("source_trace_ids") or primary.get("_source_trace_ids") or []),
                    *(candidate_for_primary.get("source_trace_ids") or candidate_for_primary.get("_source_trace_ids") or []),
                ]
            )
            merged_text_ids = _unique_preserve_order(
                [
                    *(primary.get("source_text_ids") or primary.get("_source_text_ids") or []),
                    *(candidate_for_primary.get("source_text_ids") or candidate_for_primary.get("_source_text_ids") or []),
                ]
            )
            if merged_trace_ids and should_apply_candidate_text:
                primary["source_trace_ids"] = merged_trace_ids
                primary["_source_trace_ids"] = list(merged_trace_ids)
            if merged_text_ids and should_apply_candidate_text:
                primary["source_text_ids"] = merged_text_ids
                primary["_source_text_ids"] = list(merged_text_ids)
            render_bbox = _optional_bbox4(candidate_for_primary.get("render_bbox"))
            safe_text_box = _optional_bbox4(candidate_for_primary.get("safe_text_box")) or _optional_bbox4(candidate_for_primary.get("_debug_safe_text_box"))
            if render_bbox is not None and safe_text_box is not None:
                primary["render_bbox"] = render_bbox
                primary["safe_text_box"] = safe_text_box
                primary["_debug_safe_text_box"] = safe_text_box
                target_bbox = _optional_bbox4(candidate_for_primary.get("target_bbox"))
                if target_bbox is not None:
                    primary["target_bbox"] = target_bbox
                if candidate_for_primary.get("fit_status"):
                    primary["fit_status"] = candidate_for_primary.get("fit_status")
                _merge_layer_qa_flags(primary, ["same_balloon_fragment_merged"])
            primary_trace = str(primary.get("trace_id") or primary.get("id") or "").strip()
            for layer in matched_layers:
                if layer is primary:
                    continue
                layer["visible"] = False
                layer["render_policy"] = "merged_into_primary"
                layer["merged_into_trace_id"] = primary_trace
                layer["merged_into_text_id"] = primary.get("text_id") or primary.get("id")
                if isinstance(layer.get("qa_flags"), list):
                    layer["qa_flags"] = [
                        flag
                        for flag in layer.get("qa_flags") or []
                        if str(flag) not in {"missing_render_bbox", "fit_below_minimum_legible"}
                    ]
            hydrated += 1
            continue
        rendered_siblings = [
            layer
            for layer in matched_layers
            if _optional_bbox4(layer.get("render_bbox")) is not None
            and _optional_bbox4(layer.get("safe_text_box")) is not None
            and not layer.get("_render_metadata_hydrated")
            and not _is_low_containment_suppressed_fragment(layer)
        ]
        source = rendered_siblings[0] if rendered_siblings else None
        if source is None:
            candidate_render_bbox = _optional_bbox4(candidate.get("render_bbox"))
            candidate_safe_text_box = _optional_bbox4(candidate.get("safe_text_box")) or _optional_bbox4(candidate.get("_debug_safe_text_box"))
            if candidate_render_bbox is None or candidate_safe_text_box is None:
                continue
            source = {
                "render_bbox": candidate_render_bbox,
                "safe_text_box": candidate_safe_text_box,
                "_debug_safe_text_box": candidate_safe_text_box,
                "target_bbox": _optional_bbox4(candidate.get("target_bbox")),
                "fit_status": candidate.get("fit_status") or "ok",
                "fit_attempts": candidate.get("fit_attempts") or [{"status": "ok"}],
                "qa_metrics": candidate.get("qa_metrics") if isinstance(candidate.get("qa_metrics"), dict) else {},
            }
        for layer in matched_layers:
            has_geometry = (
                _optional_bbox4(layer.get("render_bbox")) is not None
                and _optional_bbox4(layer.get("safe_text_box")) is not None
            )
            if (
                has_geometry
                and "missing_render_bbox" not in set(layer.get("qa_flags") or [])
            ):
                continue
            render_bbox = _optional_bbox4(source.get("render_bbox"))
            safe_text_box = _optional_bbox4(source.get("safe_text_box"))
            if render_bbox is None or safe_text_box is None:
                continue
            layer["render_bbox"] = render_bbox
            layer["safe_text_box"] = safe_text_box
            layer["_debug_safe_text_box"] = safe_text_box
            for key in ("target_bbox", "fit_status", "fit_attempts"):
                if source.get(key) is not None:
                    layer[key] = source.get(key)
            if isinstance(source.get("qa_metrics"), dict):
                layer["qa_metrics"] = {**dict(layer.get("qa_metrics") or {}), **dict(source.get("qa_metrics") or {})}
            layer["_render_metadata_group_sibling_geometry"] = True
            if "missing_render_bbox" in set(layer.get("qa_flags") or []):
                layer["qa_flags"] = [flag for flag in layer.get("qa_flags") or [] if flag != "missing_render_bbox"]
            hydrated += 1
    return hydrated


def _candidate_has_render_identity_in_project(candidate: dict, layers_by_identity: dict[str, list[dict]]) -> bool:
    direct_candidate_keys = {
        str(candidate.get(key) or "").strip()
        for key in ("trace_id", "text_id", "id")
        if str(candidate.get(key) or "").strip()
    }
    candidate_band_id = str(candidate.get("band_id") or "").strip()
    for layers in layers_by_identity.values():
        for layer in layers:
            if candidate_band_id and str(layer.get("band_id") or "").strip() != candidate_band_id:
                continue
            direct_layer_keys = {
                str(layer.get(key) or "").strip()
                for key in ("trace_id", "text_id", "id")
                if str(layer.get(key) or "").strip()
            }
            if direct_candidate_keys.intersection(direct_layer_keys):
                return True
    return False


def _render_candidate_fragment_key(candidate: dict) -> tuple:
    return (
        str(candidate.get("trace_id") or "").strip(),
        str(candidate.get("text_id") or candidate.get("id") or "").strip(),
        tuple(_optional_bbox4(candidate.get("render_bbox")) or []),
        tuple(_optional_bbox4(candidate.get("safe_text_box")) or _optional_bbox4(candidate.get("_debug_safe_text_box")) or []),
        _normalize_match_text(str(candidate.get("translated") or candidate.get("traduzido") or "")),
    )


def _render_candidate_fragment_source_key(candidate: dict) -> tuple:
    trace_id = str(candidate.get("trace_id") or "").strip()
    base_trace_id = trace_id.split("#", 1)[0].strip()
    text_id = str(candidate.get("text_id") or candidate.get("id") or "").strip()
    base_text_id = text_id.rsplit("_fragment_", 1)[0].strip() if "_fragment_" in text_id else text_id
    return (
        base_trace_id,
        base_text_id,
        str(candidate.get("band_id") or "").strip(),
        _normalize_match_text(str(candidate.get("translated") or candidate.get("traduzido") or "")),
    )


def _same_identity_fragment_rank(candidate: dict, band_candidates: list[dict]) -> int | None:
    candidate_tokens = _direct_render_identity_values(candidate)
    if not candidate_tokens:
        return None
    unique: dict[tuple, dict] = {}
    for peer in band_candidates:
        if not candidate_tokens.intersection(_direct_render_identity_values(peer)):
            continue
        render_bbox = _optional_bbox4(peer.get("render_bbox"))
        safe_text_box = _optional_bbox4(peer.get("safe_text_box")) or _optional_bbox4(peer.get("_debug_safe_text_box"))
        if render_bbox is None or safe_text_box is None:
            continue
        unique.setdefault(_render_candidate_fragment_key(peer), peer)
    if len(unique) < 2:
        return None
    translated_fragments = {
        _normalize_match_text(str(peer.get("translated") or peer.get("traduzido") or ""))
        for peer in unique.values()
        if str(peer.get("translated") or peer.get("traduzido") or "").strip()
    }
    if len(translated_fragments) < 2:
        return None
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            (_optional_bbox4(item.get("render_bbox")) or [0, 0, 0, 0])[1],
            (_optional_bbox4(item.get("render_bbox")) or [0, 0, 0, 0])[0],
            (_optional_bbox4(item.get("safe_text_box")) or _optional_bbox4(item.get("_debug_safe_text_box")) or [0, 0, 0, 0])[1],
            str(item.get("translated") or item.get("traduzido") or ""),
        ),
    )
    current_key = _render_candidate_fragment_key(candidate)
    for index, peer in enumerate(ordered):
        if _render_candidate_fragment_key(peer) == current_key:
            return index
    return None


def _is_restorable_same_identity_fragment(candidate: dict, band_candidates: list[dict]) -> bool:
    rank = _same_identity_fragment_rank(candidate, band_candidates)
    return rank is not None and rank > 0


def _infer_same_band_render_candidate_offset(
    band_layers: list[dict],
    band_candidates: list[dict],
    *,
    exclude_layer: dict | None = None,
) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    for layer in band_layers:
        if not isinstance(layer, dict):
            continue
        if exclude_layer is not None and layer is exclude_layer:
            continue
        for candidate in band_candidates:
            if not isinstance(candidate, dict) or not _candidate_matches_layer_direct_identity(layer, candidate):
                continue
            for key in ("target_bbox", "safe_text_box", "_debug_safe_text_box", "render_bbox", "bbox", "layout_bbox", "text_pixel_bbox"):
                layer_bbox = _optional_bbox4(layer.get(key))
                candidate_bbox = _optional_bbox4(candidate.get(key))
                if layer_bbox is None or candidate_bbox is None:
                    continue
                layer_w = max(1, layer_bbox[2] - layer_bbox[0])
                layer_h = max(1, layer_bbox[3] - layer_bbox[1])
                candidate_w = max(1, candidate_bbox[2] - candidate_bbox[0])
                candidate_h = max(1, candidate_bbox[3] - candidate_bbox[1])
                if abs(layer_w - candidate_w) / float(max(layer_w, candidate_w)) > 0.18:
                    continue
                if abs(layer_h - candidate_h) / float(max(layer_h, candidate_h)) > 0.18:
                    continue
                dx = int(layer_bbox[0] - candidate_bbox[0])
                dy = int(layer_bbox[1] - candidate_bbox[1])
                shifted = _offset_bbox4(candidate_bbox, dx, dy)
                if shifted is None:
                    continue
                score = _bbox_iou(shifted, layer_bbox) - ((abs(dx) + abs(dy)) / 1_000_000.0)
                if best is None or score > best[0]:
                    best = (score, dx, dy)
    if best is None:
        return (0, 0)
    return (best[1], best[2])


def _render_candidate_with_offset(entry: dict, dx: int, dy: int) -> dict:
    candidate = dict(entry)
    if dx == 0 and dy == 0:
        return candidate
    for key in ("target_bbox", "safe_text_box", "_debug_safe_text_box", "render_bbox", "bbox", "source_bbox", "layout_bbox", "text_pixel_bbox"):
        shifted = _offset_bbox4(_optional_bbox4(candidate.get(key)), dx, dy)
        if shifted is not None:
            candidate[key] = shifted
    for key in ("qa_metrics", "_render_debug", "_render_debug_candidates", "_render_debug_skipped"):
        if isinstance(candidate.get(key), (dict, list)):
            candidate[key] = _offset_nested_debug_bboxes(candidate[key], dx, dy)
    candidate["_same_band_restore_coordinate_offset"] = [int(dx), int(dy)]
    return candidate


def _render_candidate_with_page_space_offset(entry: dict, dx: int, dy: int) -> dict:
    candidate = _render_candidate_with_offset(entry, dx, dy)
    candidate.pop("_same_band_restore_coordinate_offset", None)
    if dx or dy:
        candidate["_page_space_coordinate_offset"] = [int(dx), int(dy)]
    return candidate


def _clear_stale_layer_bubble_geometry_for_render(
    layer: dict,
    render_bbox: list[int],
    safe_text_box: list[int],
) -> int:
    cleared = 0
    render_area = max(1, _bbox_area4(render_bbox))
    safe_area = max(1, _bbox_area4(safe_text_box))
    for key in (
        "balloon_bbox",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "_bubble_mask_bbox_unclamped",
        "_bubble_inner_bbox_unclamped",
    ):
        bbox = _optional_bbox4(layer.get(key))
        if bbox is None:
            continue
        render_overlap = _bbox_intersection_area4(bbox, render_bbox) / float(render_area)
        safe_overlap = _bbox_intersection_area4(bbox, safe_text_box) / float(safe_area)
        if (
            _bbox_contains4_margin(bbox, render_bbox, margin=12)
            or _bbox_contains4_margin(bbox, safe_text_box, margin=12)
            or render_overlap >= 0.35
            or safe_overlap >= 0.35
        ):
            continue
        layer.pop(key, None)
        cleared += 1
    if cleared:
        layer["_stale_bubble_geometry_cleared_after_page_space_hydration"] = True
    return cleared


def _inset_bbox_for_restored_balloon_safe_text(bbox: list[int]) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = max(12, int(width * 0.10))
    pad_y = max(12, int(height * 0.18))
    safe = [x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y]
    if safe[2] <= safe[0] or safe[3] <= safe[1]:
        return [x1, y1, x2, y2]
    return safe


def _debug_band_balloon_component_bbox(
    debug_root: Path,
    candidate: dict,
) -> list[int] | None:
    band_id = str(candidate.get("band_id") or "").strip()
    if not band_id:
        return None
    offset = candidate.get("_same_band_restore_coordinate_offset")
    if not isinstance(offset, (list, tuple)) or len(offset) != 2:
        return None
    try:
        dx = int(offset[0])
        dy = int(offset[1])
    except Exception:
        return None
    mask_path = debug_root / "06_mask_segmentation" / band_id / "04_balloon_mask.png"
    if not mask_path.exists():
        return None
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    binary = (mask > 127).astype("uint8")
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    raw_refs = []
    for key in ("target_bbox", "bbox", "layout_bbox", "text_pixel_bbox", "render_bbox", "safe_text_box"):
        shifted_ref = _optional_bbox4(candidate.get(key))
        if shifted_ref is None:
            continue
        raw_ref = _offset_bbox4(shifted_ref, -dx, -dy)
        if raw_ref is not None:
            raw_refs.append(raw_ref)
    if not raw_refs:
        return None

    best: tuple[float, list[int]] | None = None
    for idx in range(1, component_count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 512 or w < 16 or h < 16:
            continue
        component_bbox = [x, y, x + w, y + h]
        component_area = _bbox_area4(component_bbox)
        score = 0.0
        for ref in raw_refs:
            ref_area = _bbox_area4(ref)
            inter = _bbox_intersection_area4(component_bbox, ref)
            if inter <= 0:
                continue
            ref_cx = (ref[0] + ref[2]) / 2.0
            ref_cy = (ref[1] + ref[3]) / 2.0
            center_inside = component_bbox[0] <= ref_cx <= component_bbox[2] and component_bbox[1] <= ref_cy <= component_bbox[3]
            overlap_score = inter / float(max(1, min(component_area, ref_area)))
            score = max(score, overlap_score + (0.45 if center_inside else 0.0))
        if best is None or score > best[0]:
            best = (score, component_bbox)
    if best is None or best[0] < 0.18:
        return None
    return _offset_bbox4(best[1], dx, dy)


def _candidate_with_debug_band_balloon_component(candidate: dict, debug_root: Path) -> dict:
    component_bbox = _debug_band_balloon_component_bbox(debug_root, candidate)
    if component_bbox is None:
        return candidate
    updated = dict(candidate)
    safe_text_box = _inset_bbox_for_restored_balloon_safe_text(component_bbox)
    updated["target_bbox"] = component_bbox
    updated["balloon_bbox"] = component_bbox
    updated["bubble_mask_bbox"] = component_bbox
    updated["bubble_inner_bbox"] = safe_text_box
    updated["bubble_mask_source"] = "debug_band_balloon_component"
    updated["bubble_id"] = str(candidate.get("bubble_id") or candidate.get("band_id") or "debug_band_balloon_component")
    updated["safe_text_box"] = safe_text_box
    updated["_debug_safe_text_box"] = safe_text_box
    updated["render_bbox"] = safe_text_box
    updated["_restored_balloon_component_bbox"] = component_bbox
    return updated


def _layer_can_seed_missing_candidate(layer: dict, candidate: dict, band_candidate_count: int) -> bool:
    layer_text = _normalize_match_text(
        " ".join(
            str(layer.get(key) or "")
            for key in ("text", "original", "raw_ocr", "normalized_ocr", "normalized_text_final")
        )
    )
    candidate_text = _normalize_match_text(
        " ".join(str(candidate.get(key) or "") for key in ("text", "original", "raw_ocr", "normalized_ocr", "normalized_text_final"))
    )
    if candidate_text and len(candidate_text) >= 4 and candidate_text in layer_text:
        return True
    return band_candidate_count > 1 and _valid_layer_text_pixel_bbox_for_render_match(layer) is None


def _restore_missing_render_candidate_layers(project_data: dict, candidates: list[dict]) -> int:
    project_layers = list(_iter_project_text_layers(project_data))
    layers_by_identity = _layers_by_debug_identity(project_layers)
    existing_fragment_keys = {
        _render_candidate_fragment_source_key(layer)
        for layer in project_layers
        if str(layer.get("trace_id") or "").strip()
        and str(layer.get("translated") or layer.get("traduzido") or "").strip()
    }
    candidates_by_band: dict[str, list[dict]] = {}
    for candidate in candidates:
        if _is_low_containment_suppressed_fragment(candidate):
            continue
        band_id = str(candidate.get("band_id") or "").strip()
        if band_id:
            candidates_by_band.setdefault(band_id, []).append(candidate)

    restored = 0
    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        page_layers = page.get("text_layers")
        if not isinstance(page_layers, list):
            page_layers = _project_page_text_layers(page)
            page["text_layers"] = page_layers
        if not page_layers:
            continue
        layers_by_band: dict[str, list[dict]] = {}
        for layer in page_layers:
            band_id = str(layer.get("band_id") or "").strip()
            if band_id:
                layers_by_band.setdefault(band_id, []).append(layer)
        for band_id, band_candidates in candidates_by_band.items():
            band_layers = layers_by_band.get(band_id) or []
            if not band_layers:
                continue
            restored_fragment_keys: set[tuple] = set()
            band_dx, band_dy = _infer_same_band_render_candidate_offset(band_layers, band_candidates)
            for candidate in band_candidates:
                identity_exists = _candidate_has_render_identity_in_project(candidate, layers_by_identity)
                restore_same_identity_fragment = identity_exists and _is_restorable_same_identity_fragment(candidate, band_candidates)
                if identity_exists and not restore_same_identity_fragment:
                    continue
                fragment_rank = _same_identity_fragment_rank(candidate, band_candidates) if restore_same_identity_fragment else None
                seed = next(
                    (
                        layer
                        for layer in band_layers
                        if _layer_can_seed_missing_candidate(layer, candidate, len(band_candidates))
                    ),
                    None,
                )
                if seed is None:
                    continue
                if band_dx or band_dy:
                    candidate = _render_candidate_with_offset(candidate, band_dx, band_dy)
                restored_candidate_key = _render_candidate_fragment_source_key(candidate)
                if restored_candidate_key in existing_fragment_keys:
                    continue
                if restored_candidate_key in restored_fragment_keys:
                    continue
                restored_fragment_keys.add(restored_candidate_key)
                text_id = str(candidate.get("text_id") or candidate.get("id") or "").strip()
                trace_id = str(candidate.get("trace_id") or "").strip()
                if restore_same_identity_fragment:
                    fragment_suffix = f"fragment_{max(1, int(fragment_rank or 0) + 1)}"
                    restored_id = f"{text_id or trace_id}_{fragment_suffix}"
                    restored_trace_id = f"{trace_id}#{fragment_suffix}" if trace_id else restored_id
                else:
                    restored_id = text_id or trace_id
                    restored_trace_id = trace_id or None
                restored_layer = {
                    "id": restored_id,
                    "text_id": restored_id,
                    "trace_id": restored_trace_id,
                    "page_id": candidate.get("page_id") or seed.get("page_id"),
                    "band_id": band_id,
                    "route_action": seed.get("route_action") or "translate_inpaint_render",
                    "render_policy": seed.get("render_policy") or "normal",
                    "visible": True,
                    "text": candidate.get("text") or candidate.get("original") or "",
                    "original": candidate.get("original") or candidate.get("text") or "",
                    "raw_ocr": candidate.get("raw_ocr") or candidate.get("text") or candidate.get("original") or "",
                    "normalized_ocr": candidate.get("normalized_ocr") or candidate.get("text") or candidate.get("original") or "",
                    "normalized_text_final": candidate.get("normalized_text_final") or candidate.get("text") or candidate.get("original") or "",
                    "translated": candidate.get("translated") or candidate.get("traduzido") or "",
                    "traduzido": candidate.get("translated") or candidate.get("traduzido") or "",
                    "bbox": _optional_bbox4(candidate.get("text_pixel_bbox")) or _optional_bbox4(candidate.get("bbox")) or _optional_bbox4(seed.get("bbox")),
                    "source_bbox": _optional_bbox4(candidate.get("text_pixel_bbox")) or _optional_bbox4(candidate.get("bbox")) or _optional_bbox4(seed.get("source_bbox")),
                    "text_pixel_bbox": _optional_bbox4(candidate.get("text_pixel_bbox")) or _optional_bbox4(candidate.get("bbox")) or _optional_bbox4(seed.get("text_pixel_bbox")),
                    "target_bbox": _optional_bbox4(candidate.get("target_bbox")),
                    "safe_text_box": _optional_bbox4(candidate.get("safe_text_box")) or _optional_bbox4(candidate.get("_debug_safe_text_box")),
                    "_debug_safe_text_box": _optional_bbox4(candidate.get("safe_text_box")) or _optional_bbox4(candidate.get("_debug_safe_text_box")),
                    "render_bbox": _optional_bbox4(candidate.get("render_bbox")),
                    "fit_status": candidate.get("fit_status"),
                    "balloon_bbox": _optional_bbox4(candidate.get("target_bbox")) or _optional_bbox4(candidate.get("balloon_bbox")) or _optional_bbox4(seed.get("balloon_bbox")),
                    "qa_flags": [flag for flag in candidate.get("qa_flags") or [] if str(flag).strip()],
                    "_restored_from_render_plan_candidate": True,
                }
                if restore_same_identity_fragment:
                    restored_layer["_restored_fragment_source_text_id"] = text_id or None
                    restored_layer["_restored_fragment_source_trace_id"] = trace_id or None
                if candidate.get("_same_band_restore_coordinate_offset"):
                    restored_layer["_same_band_restore_coordinate_offset"] = candidate.get("_same_band_restore_coordinate_offset")
                page_layers.append(restored_layer)
                existing_fragment_keys.add(restored_candidate_key)
                for identity_key in _debug_identity_keys_from_payload(restored_layer):
                    layers_by_identity.setdefault(identity_key, []).append(restored_layer)
                band_layers.append(restored_layer)
                restored += 1
    return restored


def _render_candidate_text_matches_layer(layer: dict, candidate: dict) -> bool:
    layer_values = [
        layer.get("translated"),
        layer.get("traduzido"),
        layer.get("normalized_text_final"),
        layer.get("normalized_ocr"),
        layer.get("text"),
    ]
    candidate_values = [
        candidate.get("translated"),
        candidate.get("traduzido"),
        candidate.get("render_text"),
        candidate.get("text"),
        candidate.get("normalized_text_final"),
        candidate.get("normalized_ocr"),
    ]
    layer_texts = [
        _normalize_match_text(str(value or ""))
        for value in layer_values
        if str(value or "").strip()
    ]
    candidate_texts = [
        _normalize_match_text(str(value or ""))
        for value in candidate_values
        if str(value or "").strip()
    ]
    for layer_text in layer_texts:
        if len(layer_text) < 8:
            continue
        for candidate_text in candidate_texts:
            if layer_text in candidate_text:
                return True
    return False


def _render_bbox_overlaps_layer_source_text(layer: dict, render_bbox_value) -> bool:
    render_bbox = _optional_bbox4(render_bbox_value)
    if render_bbox is None:
        return False
    render_area = max(1, (render_bbox[2] - render_bbox[0]) * (render_bbox[3] - render_bbox[1]))
    for source_bbox in (
        _valid_layer_text_pixel_bbox_for_render_match(layer),
        _valid_layer_reference_bbox_for_render_match(layer, "source_bbox"),
        _valid_layer_reference_bbox_for_render_match(layer, "layout_bbox"),
        _valid_layer_reference_bbox_for_render_match(layer, "bbox"),
    ):
        if source_bbox is None:
            continue
        if _bbox_contains4_margin(source_bbox, render_bbox, margin=18):
            return True
        source_area = max(1, (source_bbox[2] - source_bbox[0]) * (source_bbox[3] - source_bbox[1]))
        iou = _bbox_iou(source_bbox, render_bbox)
        overlap = iou * float(source_area + render_area) / (1.0 + iou)
        if overlap >= float(render_area * 0.35):
            return True
    return False


def _layer_render_overlaps_source_text(layer: dict) -> bool:
    return _render_bbox_overlaps_layer_source_text(layer, layer.get("render_bbox"))


def _dark_connected_candidate_geometry_is_unsafe(layer: dict, candidate: dict) -> bool:
    if not isinstance(candidate, dict):
        return False
    flags = {
        str(flag).strip().lower()
        for flag in [*(layer.get("qa_flags") or []), *(candidate.get("qa_flags") or [])]
        if str(flag).strip()
    }
    source = str(
        candidate.get("bubble_mask_source")
        or candidate.get("balloon_mask_source")
        or layer.get("bubble_mask_source")
        or layer.get("balloon_mask_source")
        or ""
    ).strip().lower()
    dark_connected = bool(
        source == "image_dark_bubble_mask"
        and (
            "dark_bubble_connected_lobe_passthrough" in flags
            or "dark_bubble_connected_lobes_promoted" in flags
            or "partial_dark_bubble_lobe_reocr" in flags
            or candidate.get("_is_lobe_subregion")
            or layer.get("_is_lobe_subregion")
        )
    )
    if not dark_connected:
        return False
    render_bbox = _optional_bbox4(candidate.get("render_bbox"))
    if render_bbox is None:
        return False
    safe_bbox = (
        _optional_bbox4(candidate.get("safe_text_box"))
        or _optional_bbox4(candidate.get("_debug_safe_text_box"))
        or _optional_bbox4(layer.get("safe_text_box"))
        or _optional_bbox4(layer.get("_debug_safe_text_box"))
    )
    target_bbox = (
        _optional_bbox4(candidate.get("target_bbox"))
        or _optional_bbox4(candidate.get("balloon_bbox"))
        or _optional_bbox4(layer.get("target_bbox"))
        or _optional_bbox4(layer.get("balloon_bbox"))
    )
    render_area = max(1, _bbox_area4(render_bbox))
    if safe_bbox is not None:
        safe_overlap = _bbox_intersection_area4(render_bbox, safe_bbox) / float(render_area)
        if safe_overlap < 0.45:
            return True
    if target_bbox is not None:
        target_w = max(1, int(target_bbox[2]) - int(target_bbox[0]))
        target_h = max(1, int(target_bbox[3]) - int(target_bbox[1]))
        margin_x = max(10, int(target_w * 0.05))
        margin_y = max(10, int(target_h * 0.05))
        if not _bbox_contains4_margin(target_bbox, render_bbox, margin=max(margin_x, margin_y)):
            return True
        target_overlap = _bbox_intersection_area4(render_bbox, target_bbox) / float(render_area)
        if target_overlap < 0.55:
            return True
    return False


def _repair_low_containment_text_payloads(
    project_layers: list[dict],
    active_candidates: list[dict],
    suppressed_candidates: list[dict],
) -> int:
    repaired = 0
    for layer in project_layers:
        if not isinstance(layer, dict):
            continue
        current_text = _translated_text_for_merge(layer)
        if not current_text:
            continue
        layer_band_id = str(layer.get("band_id") or "").strip()
        suppressed_texts: list[str] = []
        for candidate in suppressed_candidates:
            if layer_band_id and str(candidate.get("band_id") or "").strip() != layer_band_id:
                continue
            if not _candidate_matches_layer_direct_identity(layer, candidate):
                continue
            suppressed_text = _translated_text_for_merge(candidate)
            if suppressed_text and suppressed_text not in suppressed_texts:
                suppressed_texts.append(suppressed_text)
        if not suppressed_texts or not any(text in current_text for text in suppressed_texts):
            continue
        clean_candidates = [
            candidate
            for candidate in active_candidates
            if (not layer_band_id or str(candidate.get("band_id") or "").strip() == layer_band_id)
            and _candidate_matches_layer_direct_identity(layer, candidate)
            and _translated_text_for_merge(candidate)
            and _translated_text_for_merge(candidate) != current_text
            and _translated_text_for_merge(candidate) in current_text
        ]
        if not clean_candidates:
            continue
        clean_candidate = min(clean_candidates, key=lambda candidate: len(_translated_text_for_merge(candidate)))
        clean_text = _translated_text_for_merge(clean_candidate)
        if not clean_text:
            continue
        layer["translated"] = clean_text
        layer["traduzido"] = clean_text
        geometry_candidate = clean_candidate
        if not _render_bbox_overlaps_layer_source_text(layer, geometry_candidate.get("render_bbox")):
            geometry_candidate = _render_candidate_with_layer_coordinates(clean_candidate, layer)
        render_bbox = _optional_bbox4(geometry_candidate.get("render_bbox"))
        safe_text_box = _optional_bbox4(geometry_candidate.get("safe_text_box")) or _optional_bbox4(
            geometry_candidate.get("_debug_safe_text_box")
        )
        if render_bbox is not None:
            layer["render_bbox"] = render_bbox
        if safe_text_box is not None:
            layer["safe_text_box"] = safe_text_box
            layer["_debug_safe_text_box"] = safe_text_box
        target_bbox = _optional_bbox4(geometry_candidate.get("target_bbox"))
        if target_bbox is not None:
            layer["target_bbox"] = target_bbox
        text_evidence_bbox = (
            _optional_bbox4(geometry_candidate.get("text_pixel_bbox"))
            or _optional_bbox4(geometry_candidate.get("layout_bbox"))
            or _optional_bbox4(geometry_candidate.get("bbox"))
        )
        if text_evidence_bbox is not None:
            layer["text_pixel_bbox"] = text_evidence_bbox
            layer["source_bbox"] = text_evidence_bbox
            layer["layout_bbox"] = text_evidence_bbox
            layer["bbox"] = text_evidence_bbox
        _clear_low_containment_suppression_markers(layer)
        repaired += 1
    return repaired


def _hydrate_project_render_metadata_from_debug_candidates(project_data: dict) -> dict:
    audit = {
        "applied": False,
        "candidate_count": 0,
        "suppressed_candidate_count": 0,
        "hydrated_layers": 0,
        "low_containment_text_payload_repairs": 0,
        "restored_missing_candidate_layers": 0,
        "missing_debug_root": False,
    }
    debug_root = _debug_root_from_project(project_data)
    if debug_root is None:
        audit["missing_debug_root"] = True
        return audit
    candidate_entries: list[dict] = []
    seen_candidates: set[tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]]] = set()
    raw_candidate_count = 0
    for source_name in ("render_plan_candidates.jsonl", "render_plan_raw.jsonl"):
        for entry in _load_debug_jsonl(debug_root / "09_typeset" / source_name):
            render_bbox = _optional_bbox4(entry.get("render_bbox"))
            safe_text_box = _optional_bbox4(entry.get("safe_text_box")) or _optional_bbox4(entry.get("_debug_safe_text_box"))
            if render_bbox is None and safe_text_box is not None:
                entry = {**entry, "render_bbox": list(safe_text_box), "_render_bbox_from_safe_text_box": True}
                render_bbox = safe_text_box
            if render_bbox is None or safe_text_box is None:
                continue
            key = (
                str(entry.get("trace_id") or entry.get("text_id") or ""),
                tuple(render_bbox),
                tuple(safe_text_box),
            )
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidate = dict(entry)
            candidate["_render_metadata_debug_source"] = source_name
            candidate_entries.append(candidate)
            if source_name == "render_plan_raw.jsonl":
                raw_candidate_count += 1
    audit["candidate_count"] = len(candidate_entries)
    audit["raw_candidate_count"] = raw_candidate_count
    if not candidate_entries:
        audit["applied"] = True
        return audit
    active_candidate_entries = [
        candidate
        for candidate in candidate_entries
        if not _is_low_containment_suppressed_fragment(candidate)
    ]
    suppressed_candidate_entries = [
        candidate
        for candidate in candidate_entries
        if _is_low_containment_suppressed_fragment(candidate)
    ]
    audit["suppressed_candidate_count"] = len(candidate_entries) - len(active_candidate_entries)
    if not active_candidate_entries:
        audit["applied"] = True
        return audit

    layout_blocks_by_identity = _layout_blocks_by_debug_identity(debug_root)
    candidates_by_identity: dict[str, list[dict]] = {}
    candidates_by_band: dict[str, list[dict]] = {}

    def index_candidate(candidate: dict) -> None:
        for identity_key in _debug_identity_keys_from_payload(candidate):
            candidates_by_identity.setdefault(identity_key, []).append(candidate)
        band_id = str(candidate.get("band_id") or "").strip()
        if band_id:
            candidates_by_band.setdefault(band_id, []).append(candidate)

    for entry in active_candidate_entries:
        layout_blocks: list[dict] = []
        for identity_key in _debug_identity_keys_from_payload(entry):
            layout_blocks.extend(layout_blocks_by_identity.get(identity_key, []))
        index_candidate(dict(entry))

    restored_missing_layers = _restore_missing_render_candidate_layers(project_data, active_candidate_entries)
    audit["restored_missing_candidate_layers"] = restored_missing_layers
    project_layers = list(_iter_project_text_layers(project_data))
    for layer in project_layers:
        if isinstance(layer, dict) and "_merge_source_bbox_before_hydration" not in layer:
            original_merge_bbox = _layer_merge_order_bbox(layer)
            if original_merge_bbox is not None:
                layer["_merge_source_bbox_before_hydration"] = list(original_merge_bbox)
    layers_by_identity = _layers_by_debug_identity(project_layers)
    hydrated = 0
    low_containment_text_payload_repairs = _repair_low_containment_text_payloads(
        project_layers,
        active_candidate_entries,
        suppressed_candidate_entries,
    )
    low_containment_clean_payloads: list[tuple[dict, str]] = []
    for layer in project_layers:
        layer_was_low_containment_suppressed = _is_low_containment_suppressed_fragment(layer)
        has_existing_render = (
            _optional_bbox4(layer.get("render_bbox")) is not None
            and _optional_bbox4(layer.get("safe_text_box")) is not None
        )
        target_for_existing_render = _optional_bbox4(layer.get("target_bbox"))
        render_for_existing_render = _optional_bbox4(layer.get("render_bbox"))
        safe_for_existing_render = _optional_bbox4(layer.get("safe_text_box")) or _optional_bbox4(layer.get("_debug_safe_text_box"))
        layer_flags_for_existing_render = {str(flag).strip().lower() for flag in layer.get("qa_flags") or []}
        layer_source_for_existing_render = str(layer.get("bubble_mask_source") or layer.get("balloon_mask_source") or "").strip().lower()
        layer_is_dark_visual_for_existing_render = bool(
            layer_source_for_existing_render in {"image_dark_bubble_mask", "image_dark_panel_mask", "derived_card_panel_mask"}
            or layer_flags_for_existing_render
            & {
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_oval_reocr",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "dark_panel_style_grouped",
            }
        )
        existing_render_is_safe_placeholder = False
        if render_for_existing_render is not None and safe_for_existing_render is not None and layer_is_dark_visual_for_existing_render:
            existing_render_is_safe_placeholder = (
                list(render_for_existing_render) == list(safe_for_existing_render)
                and _bbox_area4(render_for_existing_render) >= 2500
            )
        existing_render_is_far_from_target = False
        if target_for_existing_render is not None and render_for_existing_render is not None:
            tx = (int(target_for_existing_render[0]) + int(target_for_existing_render[2])) / 2.0
            ty = (int(target_for_existing_render[1]) + int(target_for_existing_render[3])) / 2.0
            rx = (int(render_for_existing_render[0]) + int(render_for_existing_render[2])) / 2.0
            ry = (int(render_for_existing_render[1]) + int(render_for_existing_render[3])) / 2.0
            target_h = max(1, int(target_for_existing_render[3]) - int(target_for_existing_render[1]))
            target_w = max(1, int(target_for_existing_render[2]) - int(target_for_existing_render[0]))
            existing_render_is_far_from_target = (
                abs(ty - ry) > max(400, target_h * 1.5)
                or abs(tx - rx) > max(300, target_w * 1.5)
            )
        if (
            has_existing_render
            and not existing_render_is_safe_placeholder
            and _layer_render_overlaps_source_text(layer)
            and not layer.get("_restored_from_render_plan_candidate")
            and not existing_render_is_far_from_target
        ):
            continue
        candidates: list[dict] = []
        seen_candidate_ids: set[int] = set()
        for identity_key in _layer_debug_identity_keys(layer):
            for candidate in candidates_by_identity.get(identity_key, []):
                if id(candidate) in seen_candidate_ids:
                    continue
                seen_candidate_ids.add(id(candidate))
                candidates.append(candidate)
        if not candidates:
            band_id = str(layer.get("band_id") or "").strip()
            for candidate in candidates_by_band.get(band_id, []):
                if id(candidate) in seen_candidate_ids:
                    continue
                if not _render_candidate_text_matches_layer(layer, candidate):
                    continue
                seen_candidate_ids.add(id(candidate))
                candidates.append(candidate)
        if not candidates:
            continue
        explicit_page_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("coordinate_space") or "").strip().lower() == "page"
            and (_optional_bbox4(candidate.get("render_bbox")) or [0, 0, 0, 0])[1] >= 900
        ]
        if explicit_page_candidates:
            candidates = explicit_page_candidates
        layer_candidates: list[dict] = []
        seen_layer_candidate_ids: set[tuple[int, tuple[int, int, int, int] | None]] = set()
        same_band_restore_offset = (0, 0)
        layer_band_id = str(layer.get("band_id") or "").strip()
        layer_is_page_space = (
            str(layer.get("coordinate_space") or "").strip().lower() == "page"
            or str(layer.get("source_coordinate_space") or "").strip().lower() == "page"
            or _layer_looks_like_page_space_by_optional_bbox(layer)
        )
        if layer_is_page_space:
            page_space_candidates = [
                candidate
                for candidate in candidates
                if str(candidate.get("coordinate_space") or "").strip().lower() == "page"
                or (
                    _candidate_explicit_band_y_top(candidate) == 0
                    and (_optional_bbox4(candidate.get("render_bbox")) or [0, 0, 0, 0])[1] >= 900
                )
            ]
            if page_space_candidates:
                candidates = page_space_candidates
        same_band_page_offset = (0, 0)
        if layer.get("_restored_from_render_plan_candidate") and layer_band_id:
            same_band_restore_offset = _infer_same_band_render_candidate_offset(
                [candidate_layer for candidate_layer in project_layers if str(candidate_layer.get("band_id") or "").strip() == layer_band_id],
                candidates_by_band.get(layer_band_id, []),
                exclude_layer=layer,
            )
        elif layer_is_page_space and layer_band_id:
            same_band_page_offset = _infer_same_band_render_candidate_offset(
                [
                    candidate_layer
                    for candidate_layer in project_layers
                    if str(candidate_layer.get("band_id") or "").strip() == layer_band_id
                ],
                candidates_by_band.get(layer_band_id, []),
                exclude_layer=layer,
            )
        for candidate in candidates:
            candidate_coordinate_space = str(candidate.get("coordinate_space") or "").strip().lower()
            candidate_is_explicit_page_space = candidate_coordinate_space == "page"
            if (
                same_band_restore_offset != (0, 0)
                and _candidate_matches_layer_direct_identity(layer, candidate)
            ):
                candidate_variants = [_render_candidate_with_offset(candidate, same_band_restore_offset[0], same_band_restore_offset[1])]
            elif layer_is_page_space and _candidate_explicit_band_y_top(candidate):
                candidate_variants = [
                    _render_candidate_with_page_space_offset(candidate, 0, _candidate_explicit_band_y_top(candidate))
                ]
            elif same_band_page_offset != (0, 0) and not candidate_is_explicit_page_space:
                candidate_variants = [_render_candidate_with_page_space_offset(candidate, same_band_page_offset[0], same_band_page_offset[1])]
            else:
                candidate_variants = [candidate]
            if existing_render_is_far_from_target and target_for_existing_render is not None:
                candidate_target_for_offset = _optional_bbox4(candidate.get("target_bbox"))
                if candidate_target_for_offset is not None:
                    dx = int(target_for_existing_render[0]) - int(candidate_target_for_offset[0])
                    dy = int(target_for_existing_render[1]) - int(candidate_target_for_offset[1])
                    if dx or dy:
                        candidate_variants.append(_render_candidate_with_offset(candidate, dx, dy))
            if (
                same_band_page_offset == (0, 0)
                and not _render_bbox_overlaps_layer_source_text(layer, candidate.get("render_bbox"))
                and not candidate_variants[0].get("_same_band_restore_coordinate_offset")
                and not candidate_variants[0].get("_page_space_coordinate_offset")
            ):
                candidate_variants.append(_render_candidate_with_layer_coordinates(candidate, layer))
            for candidate_variant in candidate_variants:
                key = (
                    id(candidate),
                    tuple(_optional_bbox4(candidate_variant.get("render_bbox")) or []),
                )
                if key in seen_layer_candidate_ids:
                    continue
                seen_layer_candidate_ids.add(key)
                layer_candidates.append(candidate_variant)
        if existing_render_is_far_from_target and target_for_existing_render is not None:
            page_space_variants: list[dict] = []
            for candidate_variant in layer_candidates:
                candidate_target = _optional_bbox4(candidate_variant.get("target_bbox"))
                if candidate_target is None:
                    continue
                dx = abs(int(candidate_target[0]) - int(target_for_existing_render[0]))
                dy = abs(int(candidate_target[1]) - int(target_for_existing_render[1]))
                if dx <= 12 and dy <= 12:
                    page_space_variants.append(candidate_variant)
            if page_space_variants:
                layer_candidates = page_space_variants
        candidates = layer_candidates
        safe_dark_connected_candidates = [
            candidate
            for candidate in candidates
            if not _dark_connected_candidate_geometry_is_unsafe(layer, candidate)
        ]
        if len(safe_dark_connected_candidates) != len(candidates):
            candidates = safe_dark_connected_candidates
            _merge_layer_qa_flags(layer, ["unsafe_dark_connected_render_candidate_rejected"])
            if not candidates:
                continue
        prefiltered_aggregate = None if layer_was_low_containment_suppressed else _aggregate_render_candidates_for_layer(layer, candidates)
        direct_identity_candidates = [
            candidate
            for candidate in candidates
            if _candidate_matches_layer_direct_identity(layer, candidate)
        ]
        primary_direct_identity_candidates = [
            candidate
            for candidate in direct_identity_candidates
            if not _is_restorable_same_identity_fragment(candidate, candidates)
        ]
        if primary_direct_identity_candidates:
            candidates = [
                candidate
                for candidate in candidates
                if not _candidate_matches_layer_direct_identity(layer, candidate)
                or candidate in primary_direct_identity_candidates
            ]
            direct_identity_candidates = primary_direct_identity_candidates
        non_direct_candidate_has_own_layer = any(
            _candidate_has_other_same_band_direct_layer(layer, candidate, layers_by_identity)
            for candidate in candidates
            if not _candidate_matches_layer_direct_identity(layer, candidate)
        )
        if direct_identity_candidates and non_direct_candidate_has_own_layer:
            candidates = direct_identity_candidates
        if len(direct_identity_candidates) > 1:
            text_compatible_direct = [
                candidate
                for candidate in direct_identity_candidates
                if _candidate_text_matches_current_layer_payload(layer, candidate)
            ]
            layer_text_norm_for_direct = _normalized_merge_text(_translated_text_for_merge(layer))
            exact_text_direct = [
                candidate
                for candidate in text_compatible_direct
                if layer_text_norm_for_direct
                and _normalized_merge_text(_candidate_text_payload(candidate)) == layer_text_norm_for_direct
            ]
            if exact_text_direct:
                text_compatible_direct = exact_text_direct
            if text_compatible_direct:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate not in direct_identity_candidates
                    or candidate in text_compatible_direct
                ]
                direct_identity_candidates = text_compatible_direct
                prefiltered_aggregate = None
        aggregate = prefiltered_aggregate or (
            None if layer_was_low_containment_suppressed else _aggregate_render_candidates_for_layer(layer, candidates)
        )
        if aggregate is not None:
            best = aggregate
        else:
            best = max(candidates, key=lambda candidate: _render_candidate_score_for_layer(layer, candidate))
        if _render_candidate_score_for_layer(layer, best) < 0.20:
            continue
        primary_layer = _primary_layer_for_merged_candidate(layer, best, layers_by_identity)
        if primary_layer is not None and primary_layer is not layer:
            continue
        if (
            _candidate_has_multiple_direct_sources(best)
            and not _candidate_matches_layer_direct_identity(layer, best)
            and _candidate_has_other_same_band_direct_layer(layer, best, layers_by_identity)
        ):
            continue
        if best.get("_same_band_restore_coordinate_offset"):
            best = _candidate_with_debug_band_balloon_component(best, debug_root)
        render_bbox = _optional_bbox4(best.get("render_bbox"))
        safe_text_box = _optional_bbox4(best.get("safe_text_box")) or _optional_bbox4(best.get("_debug_safe_text_box"))
        if render_bbox is None or safe_text_box is None:
            continue
        layer_text_norm = _normalized_merge_text(layer.get("translated") or layer.get("traduzido") or "")
        best_text = _translated_text_for_merge(best)
        best_text_norm = _normalized_merge_text(best_text)
        best_extends_layer_text = bool(
            best_text
            and layer_text_norm
            and layer_text_norm in best_text_norm
            and len(best_text_norm) >= len(layer_text_norm)
        )
        allow_candidate_text_payload = (
            not _candidate_has_multiple_direct_sources(best)
            or (
                _candidate_matches_layer_direct_identity(layer, best)
                and _candidate_source_layers_have_text_overlap(best, layers_by_identity)
            )
            or best_extends_layer_text
        )
        for text_key in ("translated", "traduzido"):
            if allow_candidate_text_payload and str(best.get(text_key) or "").strip():
                layer[text_key] = best.get(text_key)
        for list_key in ("source_trace_ids", "_source_trace_ids", "source_text_ids", "_source_text_ids"):
            values = _debug_identity_values(best, (list_key,))
            if allow_candidate_text_payload and values:
                layer[list_key] = values
        layer["render_bbox"] = render_bbox
        layer["safe_text_box"] = safe_text_box
        layer["_debug_safe_text_box"] = safe_text_box
        target_bbox = _optional_bbox4(best.get("target_bbox"))
        if target_bbox is not None:
            layer["target_bbox"] = target_bbox
        for bbox_key in ("balloon_bbox", "bubble_mask_bbox", "bubble_inner_bbox"):
            bbox_value = _optional_bbox4(best.get(bbox_key))
            if bbox_value is not None and (
                _bbox_contains4(bbox_value, render_bbox)
                and (safe_text_box is None or _bbox_contains4(bbox_value, safe_text_box))
            ):
                layer[bbox_key] = bbox_value
        if best.get("_page_space_coordinate_offset"):
            _clear_stale_layer_bubble_geometry_for_render(layer, render_bbox, safe_text_box)
        for value_key in ("bubble_id", "bubble_mask_source", "_restored_balloon_component_bbox"):
            if best.get(value_key) is not None:
                layer[value_key] = best.get(value_key)
        text_evidence_bbox = (
            _optional_bbox4(best.get("text_pixel_bbox"))
            or _optional_bbox4(best.get("layout_bbox"))
            or _optional_bbox4(best.get("bbox"))
            or _valid_layer_reference_bbox_for_render_match(layer, "bbox")
            or _valid_layer_reference_bbox_for_render_match(layer, "layout_bbox")
        )
        if text_evidence_bbox is not None:
            layer_has_corrupt_text_geometry = _valid_layer_text_pixel_bbox_for_render_match(layer) is None
            layer_is_direct_candidate = _candidate_matches_layer_direct_identity(layer, best)
            if layer_has_corrupt_text_geometry or layer_is_direct_candidate:
                layer["text_pixel_bbox"] = text_evidence_bbox
                layer["source_bbox"] = text_evidence_bbox
                layer["layout_bbox"] = text_evidence_bbox
                layer["bbox"] = text_evidence_bbox
        if best.get("fit_status"):
            layer["fit_status"] = best.get("fit_status")
        if isinstance(best.get("fit_attempts"), list):
            layer["fit_attempts"] = best.get("fit_attempts")
        try:
            font_size_final = int(best.get("font_size_final") or 0)
        except (TypeError, ValueError):
            font_size_final = 0
        if font_size_final > 0:
            style = layer.get("estilo") if isinstance(layer.get("estilo"), dict) else layer.get("style")
            style = dict(style or {})
            style["tamanho"] = font_size_final
            if str(best.get("font_name") or "").strip():
                style["fonte"] = str(best.get("font_name") or "").strip()
            layer["estilo"] = style
            layer["style"] = style
            layer["_render_metadata_font_size_hydrated"] = True
        render_contract = _render_layout_contract_from_candidate(best)
        if isinstance(render_contract, dict):
            if best.get("_page_space_coordinate_offset") or str(render_contract.get("coordinate_space") or "").strip().lower() == "page":
                render_contract["coordinate_space"] = "page"
                render_contract["band_y_top"] = 0
            layer["render_layout_contract"] = render_contract
            layer["_render_layout_contract_hydrated_from_debug"] = True
        if isinstance(best.get("qa_metrics"), dict):
            layer["qa_metrics"] = {**dict(layer.get("qa_metrics") or {}), **dict(best.get("qa_metrics") or {})}
        if best.get("_project_coordinate_offset"):
            layer["_render_metadata_project_coordinate_offset"] = best.get("_project_coordinate_offset")
        if best.get("_page_space_coordinate_offset"):
            layer["_render_metadata_page_space_coordinate_offset"] = best.get("_page_space_coordinate_offset")
        if best.get("_same_band_restore_coordinate_offset"):
            layer["_same_band_restore_coordinate_offset"] = best.get("_same_band_restore_coordinate_offset")
        else:
            layer.pop("_same_band_restore_coordinate_offset", None)
        _merge_layer_qa_flags(layer, [str(flag) for flag in best.get("qa_flags") or [] if str(flag).strip()])
        if not _is_low_containment_suppressed_fragment(best):
            _clear_low_containment_suppression_markers(layer)
            if layer_was_low_containment_suppressed and _candidate_matches_layer_direct_identity(layer, best):
                clean_text = _translated_text_for_merge(best)
                if clean_text:
                    low_containment_clean_payloads.append((layer, clean_text))
        if "missing_render_bbox" in set(layer.get("qa_flags") or []):
            layer["qa_flags"] = [flag for flag in layer.get("qa_flags") or [] if flag != "missing_render_bbox"]
        layer["_render_metadata_hydrated"] = True
        _hide_merged_candidate_sibling_layers(layer, best, layers_by_identity)
        hydrated += 1
    hydrated += _copy_group_sibling_render_metadata(project_data, active_candidate_entries)
    split_lobe_text_payload_repairs = _repair_project_split_lobe_text_payloads(project_layers, active_candidate_entries)
    distinct_dark_lobe_payload_merge_repairs = _repair_distinct_dark_lobe_project_payload_merges(project_layers)
    repeated_text_prefix_repairs = _dedupe_repeated_project_text_prefixes(project_layers)
    for layer, clean_text in low_containment_clean_payloads:
        layer["translated"] = clean_text
        layer["traduzido"] = clean_text
        _clear_low_containment_suppression_markers(layer)
    low_containment_text_payload_repairs += _repair_low_containment_text_payloads(
        project_layers,
        active_candidate_entries,
        suppressed_candidate_entries,
    )
    audit["applied"] = True
    audit["hydrated_layers"] = hydrated
    audit["low_containment_text_payload_repairs"] = low_containment_text_payload_repairs
    audit["split_lobe_text_payload_repairs"] = split_lobe_text_payload_repairs
    audit["distinct_dark_lobe_payload_merge_repairs"] = distinct_dark_lobe_payload_merge_repairs
    audit["repeated_text_prefix_repairs"] = repeated_text_prefix_repairs
    _clear_merge_source_bbox_before_hydration(project_layers)
    return audit


def _resolve_debug_claim_layers(
    claim: dict,
    layers_by_identity: dict[str, list[dict]],
) -> tuple[list[dict], list[str]]:
    identity_groups = claim.get("identity_groups") or []
    for group in identity_groups:
        matched_layers: list[dict] = []
        for identity_key in group:
            layers = layers_by_identity.get(identity_key, [])
            if len(layers) == 1:
                selected = layers[0]
            else:
                selected = _preferred_debug_claim_layer(layers)
            if selected is not None and selected not in matched_layers:
                matched_layers.append(selected)
        if matched_layers:
            return matched_layers, list(group)
    fallback_layers = _resolve_negative_dark_claim_by_band_flag(claim, layers_by_identity)
    if fallback_layers:
        checked = list(identity_groups[0]) if identity_groups else []
        return fallback_layers, checked
    return [], list(identity_groups[0]) if identity_groups else []


def _resolve_negative_dark_claim_by_band_flag(
    claim: dict,
    layers_by_identity: dict[str, list[dict]],
) -> list[dict]:
    identity_groups = claim.get("identity_groups") or []
    flat_identities = [str(value or "").strip() for group in identity_groups for value in group]
    if not any(value.startswith("negative_dark_") for value in flat_identities):
        return []
    band_id = ""
    for value in flat_identities:
        match = re.search(r"(page_\d{3}_band_\d{3})", value)
        if match:
            band_id = match.group(1)
            break
    if not band_id:
        return []
    flags = {str(flag).strip() for flag in claim.get("flags") or [] if str(flag).strip()}
    if not flags:
        return []
    seen: set[int] = set()
    matches: list[dict] = []
    for bucket in layers_by_identity.values():
        for layer in bucket:
            marker = id(layer)
            if marker in seen:
                continue
            seen.add(marker)
            if str(layer.get("band_id") or "").strip() != band_id:
                continue
            layer_flags = {str(flag).strip() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
            if flags.intersection(layer_flags):
                matches.append(layer)
    return matches


def _preferred_debug_claim_layer(layers: list[dict]) -> dict | None:
    if not layers:
        return None
    active_layers = [
        layer
        for layer in layers
        if str(layer.get("route_action") or "").strip() != "merged_into_primary"
    ]
    candidates = active_layers or list(layers)
    renderable = [
        layer
        for layer in candidates
        if _optional_bbox4(layer.get("render_bbox")) is not None
        or _optional_bbox4(layer.get("safe_text_box")) is not None
    ]
    if renderable:
        candidates = renderable
    translated = [layer for layer in candidates if str(layer.get("translated") or layer.get("traduzido") or "").strip()]
    if translated:
        candidates = translated
    return candidates[0] if len(candidates) == 1 else None


def _write_qa_flag_propagation_audit(debug_root: Path, audit: dict) -> None:
    try:
        target = debug_root / "11_qa_export_gate" / "qa_flag_propagation_audit.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Falha ao escrever qa_flag_propagation_audit.json: %s", exc)


def _augment_qa_summary_with_debug_contract(summary: dict, audit: dict | None) -> dict:
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(audit, dict):
        return summary
    audit_summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    missing_count = int(audit_summary.get("qa_flag_not_propagated_count") or 0)
    unmatched_detect_count = int(audit_summary.get("detect_accepted_unmatched_count") or 0)
    critical_debug_count = missing_count + unmatched_detect_count
    if critical_debug_count <= 0:
        return summary

    counts = summary.setdefault("counts", {})
    if not isinstance(counts, dict):
        counts = {}
        summary["counts"] = counts
    if missing_count > 0:
        counts["qa_flag_not_propagated"] = int(counts.get("qa_flag_not_propagated") or 0) + missing_count
    if unmatched_detect_count > 0:
        counts["detect_candidate_without_ocr_text"] = (
            int(counts.get("detect_candidate_without_ocr_text") or 0) + unmatched_detect_count
        )

    flags = summary.setdefault("flags", [])
    if not isinstance(flags, list):
        flags = []
        summary["flags"] = flags
    if missing_count > 0 and "qa_flag_not_propagated" not in flags:
        flags.append("qa_flag_not_propagated")
    if unmatched_detect_count > 0 and "detect_candidate_without_ocr_text" not in flags:
        flags.append("detect_candidate_without_ocr_text")

    summary["highest_severity"] = "critical"
    summary["issue_count"] = int(summary.get("issue_count") or 0) + critical_debug_count
    summary["critical_issue_count"] = int(summary.get("critical_issue_count") or 0) + critical_debug_count
    summary["critical_flag_count"] = int(summary.get("critical_flag_count") or summary.get("critical_count") or 0) + critical_debug_count
    summary["critical_count"] = summary["critical_flag_count"]
    summary["total"] = int(summary.get("total") or 0) + critical_debug_count
    return summary


def _apply_debug_flag_route_review(layer: dict, flags: set[str]) -> None:
    combined_flags = {
        str(flag).strip()
        for flag in [*(layer.get("qa_flags") or []), *list(flags or set())]
        if str(flag).strip()
    }
    try:
        rotation_deg = float(
            layer.get("rotation_deg")
            or (layer.get("estilo") or {}).get("rotacao")
            or (layer.get("style") or {}).get("rotacao")
            or 0.0
        )
    except Exception:
        rotation_deg = 0.0
    if abs(rotation_deg) >= 5.0 and "mask_outside_balloon_critical" in combined_flags:
        combined_flags.discard("mask_outside_balloon")
        combined_flags.discard("mask_outside_balloon_critical")
        cleaned_flags = [
            flag
            for flag in layer.get("qa_flags") or []
            if str(flag).strip() not in {"mask_outside_balloon", "mask_outside_balloon_critical"}
        ]
        if "rotated_text_mask_outside_balloon_allowed" not in cleaned_flags:
            cleaned_flags.append("rotated_text_mask_outside_balloon_allowed")
        layer["qa_flags"] = cleaned_flags
    reason = ""
    if "mask_outside_balloon_critical" in combined_flags:
        reason = "mask_outside_balloon_critical"
    elif "unsafe_derived_art_mask_review" in combined_flags:
        reason = "unsafe_derived_art_mask_review"
    elif "source_glyph_area_ratio_critical" in combined_flags and (
        "render_on_art_suspected" in combined_flags
        or "weak_text_residual_after_inpaint" in combined_flags
        or "unsafe_derived_art_mask_review" in combined_flags
    ):
        reason = "source_glyph_area_ratio_critical"
    if not reason:
        return
    layer["route_action"] = "review_required"
    layer["route_reason"] = reason
    layer["needs_review"] = True
    layer["render_policy"] = "review_required"
    layer["_review_required_preserved_render_geometry"] = bool(
        _optional_bbox4(layer.get("render_bbox")) is not None
        and (
            _optional_bbox4(layer.get("safe_text_box")) is not None
            or _optional_bbox4(layer.get("_debug_safe_text_box")) is not None
        )
    )


def _propagate_debug_qa_flags_to_project(project_data: dict) -> dict:
    debug_root = _debug_root_from_project(project_data)
    render_claims = _collect_render_plan_qa_flags(debug_root) if debug_root else []
    mask_claims = _collect_mask_decision_qa_flags(debug_root) if debug_root else []
    inpaint_claims = _collect_inpaint_decision_qa_flags(debug_root) if debug_root else []
    unmatched_detect_candidates = _collect_unmatched_detect_candidates(debug_root) if debug_root else []
    all_claims = [*render_claims, *mask_claims, *inpaint_claims]

    project_layers = list(_iter_project_text_layers(project_data))
    layers_by_identity = _layers_by_debug_identity(project_layers)
    if debug_root:
        for layer in project_layers:
            layer["qa_flags"] = [
                flag for flag in (layer.get("qa_flags") or []) if str(flag) not in MASK_SYNCED_QA_FLAGS
            ]

    missing: list[dict] = []
    for claim in all_claims:
        flags = set(claim.get("flags") or [])
        matched_layers, checked_identities = _resolve_debug_claim_layers(claim, layers_by_identity)
        if matched_layers:
            for layer in matched_layers:
                filtered_flags = _filter_debug_claim_flags_for_project_layer(layer, flags)
                _merge_layer_qa_flags(layer, sorted(filtered_flags))
                _apply_debug_flag_route_review(layer, filtered_flags)
        else:
            missing_flags = _traceability_blocking_missing_flags(flags)
            if not missing_flags:
                continue
            for identity_key in checked_identities or [""]:
                for flag in sorted(missing_flags):
                    missing.append(
                        {
                            "identity": identity_key,
                            "text_id": identity_key,
                            "flag": flag,
                            "source": claim.get("source"),
                            "in_render_plan": claim.get("source") == "render_plan",
                            "in_mask_decision": claim.get("source") == "mask_decision",
                            "in_inpaint_decision": claim.get("source") == "inpaint_decision",
                            "in_project": False,
                            "is_review_only": _debug_claim_missing_flag_is_review_only(flag, claim),
                        }
                    )

    project_layer_flag_count = sum(
        len({str(flag) for flag in (layer.get("qa_flags") or []) if str(flag).strip()})
        for layer in project_layers
    )
    audit = {
        "summary": {
            "render_plan_flags": _claim_flag_count(render_claims),
            "mask_decision_flags": _claim_flag_count(mask_claims),
            "inpaint_decision_flags": _claim_flag_count(inpaint_claims),
            "detect_accepted_unmatched_count": len(unmatched_detect_candidates),
            "project_layer_flags": project_layer_flag_count,
            "qa_flag_not_propagated_count": len(missing),
        },
        "missing_in_project": missing,
        "unmatched_detect_candidates": unmatched_detect_candidates,
    }
    if debug_root:
        _write_qa_flag_propagation_audit(debug_root, audit)
    project_data.setdefault("qa", {})["flag_propagation_audit"] = audit
    return audit


def _filter_debug_claim_flags_for_project_layer(layer: dict, flags: set[str]) -> set[str]:
    filtered = set(flags)
    filtered.difference_update(_resolved_pre_render_flags(layer))
    render_bbox = _optional_bbox4(layer.get("render_bbox"))
    safe_text_box = _optional_bbox4(layer.get("safe_text_box"))
    has_render_geometry = render_bbox is not None and safe_text_box is not None
    if (
        "render_suppressed_low_containment_fragment" in filtered
        and not _is_low_containment_suppressed_fragment(layer)
    ):
        filtered.discard("render_suppressed_low_containment_fragment")
    if "missing_render_bbox" in filtered and has_render_geometry:
        filtered.discard("missing_render_bbox")
    if "fit_below_minimum_legible" in filtered and has_render_geometry:
        fit_status = str(layer.get("fit_status") or "").strip().lower()
        attempts = [item for item in list(layer.get("fit_attempts") or []) if isinstance(item, dict)]
        has_ok_attempt = any(str(item.get("status") or "").strip().lower() == "ok" for item in attempts)
        if fit_status == "ok" or (has_ok_attempt and _bbox_contains4_margin(safe_text_box, render_bbox)):
            filtered.discard("fit_below_minimum_legible")
    if "render_on_art_suspected" in filtered and _render_background_art_flag_is_stale(layer):
        filtered.discard("render_on_art_suspected")
    return filtered


def _normalize_export_issue_for_debug(issue: dict, layers_by_id: dict[str, dict] | None = None) -> dict:
    normalized = dict(issue)
    layer_key = str(normalized.get("layer") or normalized.get("text_id") or "").strip()
    layer = (layers_by_id or {}).get(layer_key)
    if layer is not None:
        text_id = layer.get("text_id") or layer.get("id")
        band_id = layer.get("band_id")
        normalized.setdefault("text_id", text_id)
        normalized.setdefault("trace_id", layer.get("trace_id"))
        normalized.setdefault("text_instance_id", layer.get("text_instance_id") or (f"{band_id}_{text_id}" if band_id and text_id else None))
        normalized.setdefault("page_id", layer.get("page_id"))
        normalized.setdefault("band_id", band_id)
        normalized.setdefault("coordinate_space", layer.get("coordinate_space") or "page")
        normalized.setdefault("source_bbox", layer.get("source_bbox") or layer.get("bbox"))
        normalized.setdefault("balloon_bbox", layer.get("balloon_bbox"))
        normalized.setdefault("render_bbox", layer.get("render_bbox"))
    linked = list(normalized.get("linked_artifacts") or [])
    for rel_path in ("09_typeset/render_plan_final.jsonl", "05_layout_geometry/layout_blocks.jsonl"):
        if rel_path not in linked:
            linked.append(rel_path)
    normalized["linked_artifacts"] = linked
    return normalized


def _write_debug_jsonl_replace(recorder, rel_path: str, entries: list[dict]) -> None:
    if not recorder:
        return
    try:
        target = recorder._root / rel_path
        stage = recorder._stage_from_rel(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(recorder._header(entry, stage=stage), ensure_ascii=False)
            for entry in entries
        ]
        target.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        recorder.register_artifact(stage=stage, rel_path=rel_path, kind="jsonl")
    except Exception as exc:
        try:
            recorder.event("qa_export_gate", "write_jsonl_replace_failed", {"rel_path": rel_path, "error": str(exc)})
        except Exception:
            pass


def _trace_page_id(value: str | None) -> str | None:
    match = re.search(r"@(page_\d{3})_band_\d{3}", str(value or ""))
    return match.group(1) if match else None


def _trace_band_id(value: str | None) -> str | None:
    match = re.search(r"@(page_\d{3}_band_\d{3})", str(value or ""))
    return match.group(1) if match else None


def _project_page_id(page: dict, page_index: int) -> str:
    raw_page_id = str(page.get("page_id") or "").strip()
    if raw_page_id:
        return raw_page_id
    try:
        page_number = int(page.get("numero") or page_index + 1)
    except Exception:
        page_number = page_index + 1
    return f"page_{page_number:03d}"


def _bbox_contains4(outer: list[int] | None, inner: list[int] | None) -> bool:
    if outer is None or inner is None:
        return False
    return bool(outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3])


def _normalize_project_render_balloon_bboxes(project_data: dict) -> int:
    """Keep final project geometry aligned with the actual typeset target.

    Some render paths deliberately expand capacity from a tiny OCR balloon to a
    nearby visual target.  The rendered ink and safe box are then valid, but the
    original balloon bbox can remain too small in project.json and break strict
    traceability audits.  Prefer the renderer's own target when it contains the
    final ink.
    """
    if not isinstance(project_data, dict):
        return 0
    fixed = 0
    for layer in _iter_project_text_layers(project_data):
        render_bbox = _optional_bbox4(layer.get("render_bbox"))
        if render_bbox is None:
            continue
        balloon_bbox = _optional_bbox4(layer.get("balloon_bbox"))
        safe_text_box = _optional_bbox4(layer.get("safe_text_box")) or _optional_bbox4(layer.get("_debug_safe_text_box"))
        if _bbox_contains4(balloon_bbox, render_bbox) and (safe_text_box is None or _bbox_contains4(balloon_bbox, safe_text_box)):
            continue
        render_debug = layer.get("_render_debug") if isinstance(layer.get("_render_debug"), dict) else {}
        candidates = [
            _optional_bbox4(render_debug.get("target_bbox")),
            _optional_bbox4(render_debug.get("position_bbox")),
            _optional_bbox4(render_debug.get("capacity_bbox")),
            _optional_bbox4(layer.get("target_bbox")),
            _optional_bbox4(layer.get("layout_bbox")),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if not _bbox_contains4(candidate, render_bbox):
                continue
            if safe_text_box is not None and not _bbox_contains4(candidate, safe_text_box):
                continue
            if balloon_bbox is not None:
                cand_area = max(1, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
                balloon_area = max(1, (balloon_bbox[2] - balloon_bbox[0]) * (balloon_bbox[3] - balloon_bbox[1]))
                if cand_area > max(balloon_area * 16, balloon_area + 120_000):
                    continue
            layer["_original_balloon_bbox_before_render_sync"] = balloon_bbox
            layer["balloon_bbox"] = list(candidate)
            layer["_balloon_bbox_synced_from_render_debug"] = True
            fixed += 1
            break
    return fixed


def _project_render_plan_row(page: dict, layer: dict, page_index: int) -> dict | None:
    route_action = str(layer.get("route_action") or "").strip()
    render_policy = str(layer.get("render_policy") or "").strip()
    if route_action == "merged_into_primary" or render_policy == "merged_into_primary":
        return None
    if layer.get("visible", True) is False and layer.get("_force_render_hidden") is not True:
        return None
    render_bbox = _optional_bbox4(layer.get("render_bbox"))
    if render_bbox is None:
        return None

    trace_id = str(layer.get("trace_id") or "").strip()
    page_id = str(layer.get("page_id") or "").strip() or _trace_page_id(trace_id) or _project_page_id(page, page_index)
    band_id = str(layer.get("band_id") or "").strip() or _trace_band_id(trace_id) or None
    text_id = str(layer.get("text_id") or layer.get("id") or "").strip() or None
    safe_text_box = _optional_bbox4(layer.get("safe_text_box")) or _optional_bbox4(layer.get("_debug_safe_text_box"))
    translated = (
        layer.get("translated")
        or layer.get("traduzido")
        or layer.get("texto_traduzido")
        or ""
    )
    original = (
        layer.get("original")
        or layer.get("raw_ocr")
        or layer.get("normalized_ocr")
        or layer.get("text")
        or ""
    )
    source_bbox_for_plan = _render_plan_source_bbox(layer)
    row = {
        "stage": "typeset",
        "source": "project_json_final",
        "text_id": text_id,
        "trace_id": trace_id or None,
        "text_instance_id": layer.get("text_instance_id"),
        "page_id": page_id,
        "band_id": band_id,
        "coordinate_space": "page",
        "band_y_top": layer.get("band_y_top"),
        "original": original,
        "translated": translated,
        "target_bbox": _optional_bbox4(layer.get("target_bbox")) or _optional_bbox4(layer.get("layout_bbox")),
        "safe_text_box": safe_text_box,
        "_debug_safe_text_box": safe_text_box,
        "render_bbox": render_bbox,
        "balloon_bbox": _optional_bbox4(layer.get("balloon_bbox")),
        "source_bbox": source_bbox_for_plan,
        "text_pixel_bbox": _optional_bbox4(layer.get("text_pixel_bbox")),
        "bbox": _optional_bbox4(layer.get("bbox")),
        "content_class": layer.get("content_class"),
        "tipo": layer.get("tipo"),
        "qa_flags": _drop_resolved_pre_render_flags(layer, list(layer.get("qa_flags") or [])),
        "qa_metrics": dict(layer.get("qa_metrics") or {}),
        "warnings": list(layer.get("warnings") or []),
    }
    style = layer.get("estilo") if isinstance(layer.get("estilo"), dict) else layer.get("style")
    if isinstance(style, dict):
        row["estilo"] = copy.deepcopy(style)
        if style.get("fonte"):
            row["font_name"] = style.get("fonte")
        if style.get("tamanho"):
            row["font_size_final"] = style.get("tamanho")
    render_debug = layer.get("_render_debug") if isinstance(layer.get("_render_debug"), dict) else {}
    if render_debug.get("font_name"):
        row["font_name"] = render_debug.get("font_name")
    if render_debug.get("font_size_final"):
        row["font_size_final"] = render_debug.get("font_size_final")
    if render_debug.get("line_height"):
        row["line_height"] = render_debug.get("line_height")
    if render_debug.get("wrapped_lines"):
        row["wrapped_lines"] = list(render_debug.get("wrapped_lines") or [])
        row["line_count"] = len(row["wrapped_lines"])
    render_contract = layer.get("render_layout_contract") if isinstance(layer.get("render_layout_contract"), dict) else None
    if isinstance(render_contract, dict):
        row["render_layout_contract"] = copy.deepcopy(render_contract)
        row.setdefault("font_name", render_contract.get("font_name"))
        row.setdefault("font_size_final", render_contract.get("font_size"))
        row.setdefault("line_height", render_contract.get("line_height"))
        row.setdefault("wrapped_lines", list(render_contract.get("lines") or []))
        if row.get("wrapped_lines"):
            row.setdefault("line_count", len(row["wrapped_lines"]))
    qa_metrics = row.get("qa_metrics") if isinstance(row.get("qa_metrics"), dict) else {}
    for metric_name in (
        "render_balloon_containment",
        "render_outside_balloon",
        "fit_status",
    ):
        if metric_name in qa_metrics:
            row.setdefault(metric_name, copy.deepcopy(qa_metrics[metric_name]))
    for field in ("style_origin", "style_confidence", "style_source"):
        if field in layer:
            row[field] = copy.deepcopy(layer.get(field))
    if isinstance(layer.get("style_evidence"), dict):
        row["style_evidence"] = copy.deepcopy(layer["style_evidence"])
    return {key: value for key, value in row.items() if value is not None}


def _write_debug_render_plan_final_from_project(recorder, project_data: dict) -> dict:
    audit = {
        "summary": {
            "source": "project_json_final",
            "project_page_count": 0,
            "project_rendered_count": 0,
            "written_count": 0,
        },
        "missing_identity": [],
        "missing_mask_contract": [],
    }
    if not recorder or not isinstance(project_data, dict):
        return audit

    rows: list[dict] = []
    missing_identity: list[dict] = []
    missing_mask_contract: list[dict] = []
    debug_root = _debug_root_from_project(project_data)
    pages = [page for page in project_data.get("paginas") or [] if isinstance(page, dict)]
    audit["summary"]["project_page_count"] = len(pages)
    for page_index, page in enumerate(pages):
        layers = page.get("text_layers") or page.get("textos") or []
        for layer_index, layer in enumerate(layers):
            if not isinstance(layer, dict):
                continue
            row = _project_render_plan_row(page, layer, page_index)
            if row is None:
                continue
            band_id = str(row.get("band_id") or layer.get("band_id") or "").strip()
            if debug_root is not None and band_id:
                debug_bboxes = _debug_text_mask_bboxes(debug_root, band_id, layer) or _debug_band_mask_bboxes(debug_root, band_id)
                if not debug_bboxes:
                    flags = list(row.get("qa_flags") or [])
                    if "typeset_without_debug_mask_contract" not in flags:
                        flags.append("typeset_without_debug_mask_contract")
                    row["qa_flags"] = flags
                    row["fit_status"] = row.get("fit_status") or "review_required"
                    missing_mask_contract.append(
                        {
                            "page_id": row.get("page_id"),
                            "band_id": band_id,
                            "text_id": row.get("text_id"),
                            "layer_index": layer_index,
                        }
                    )
            audit["summary"]["project_rendered_count"] += 1
            if not row.get("trace_id") and not (row.get("text_id") and row.get("band_id")):
                missing_identity.append(
                    {
                        "page_id": row.get("page_id"),
                        "layer_index": layer_index,
                        "text_id": row.get("text_id"),
                    }
                )
            rows.append(row)

    audit["summary"]["written_count"] = len(rows)
    audit["missing_identity"] = missing_identity
    audit["missing_mask_contract"] = missing_mask_contract
    try:
        recorder.record_canonical_text_metrics(rows, replace_existing=True)
    except Exception:
        pass
    _write_debug_jsonl_replace(recorder, "09_typeset/render_plan_final.jsonl", rows)
    try:
        recorder.write_json("09_typeset/render_plan_final_sync.json", audit)
    except Exception:
        pass
    return audit


def _refresh_debug_final_band_crops_from_translated(recorder, work_dir: Path) -> dict:
    audit = {
        "source": "translated_after_main_final_page_space",
        "after_final_project_image_rerender": False,
        "after_late_render_contract_repair": False,
        "seen_count": 0,
        "refreshed_count": 0,
        "missing_count": 0,
        "error_count": 0,
    }
    if not recorder:
        return audit
    try:
        import cv2

        root = Path(work_dir) / "debug" / "e2e"
        crops_path = root / "10_copyback_reassemble" / "final_band_crops.jsonl"
        if not crops_path.exists():
            return audit
        for line in crops_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            audit["seen_count"] += 1
            try:
                row = json.loads(line)
            except Exception:
                audit["error_count"] += 1
                continue
            translated_name = str(row.get("translated_output_page") or "").strip()
            final_rel = str(row.get("final_crop_path") or "").strip()
            bbox = row.get("crop_bbox_in_translated_page")
            if not translated_name or not final_rel or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                audit["missing_count"] += 1
                continue
            translated_path = Path(translated_name)
            if not translated_path.is_absolute():
                if translated_path.parts and translated_path.parts[0].lower() == "translated":
                    translated_path = Path(work_dir) / translated_path
                else:
                    translated_path = Path(work_dir) / "translated" / translated_path
            image = cv2.imread(str(translated_path), cv2.IMREAD_COLOR)
            if image is None:
                audit["missing_count"] += 1
                continue
            try:
                x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
            except Exception:
                audit["error_count"] += 1
                continue
            height, width = image.shape[:2]
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                audit["missing_count"] += 1
                continue
            recorder.write_image(final_rel, image[y1:y2, x1:x2, :], quality=100)
            audit["refreshed_count"] += 1
        try:
            recorder.write_json("10_copyback_reassemble/final_band_crops_refresh.json", audit)
        except Exception:
            pass
    except Exception:
        audit["error_count"] += 1
    return audit


def _resolve_debug_e2e_path(work_dir: Path, value: object) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0].lower() == "debug":
        return Path(work_dir) / path
    return Path(work_dir) / "debug" / "e2e" / path


def _clean_final_band_source_for_crop_row(root: Path, work_dir: Path, row: dict) -> tuple[Path | None, str]:
    band_id = str(row.get("band_id") or "").strip()
    candidates: list[tuple[Path | None, str]] = []
    for key in ("post_copyback_path", "clean_band_path", "clean_source_path"):
        candidates.append((_resolve_debug_e2e_path(work_dir, row.get(key)), key))
    if band_id:
        post_dir = root / "10_copyback_reassemble" / band_id
        for name in ("post_copyback.jpg", "post_copyback.png", "post_copyback.jpeg"):
            candidates.append((post_dir / name, "post_copyback_convention"))
    candidates.append((_resolve_debug_e2e_path(work_dir, row.get("rendered_band_path")), "rendered_band_path"))
    candidates.append((_resolve_debug_e2e_path(work_dir, row.get("final_crop_path")), "existing_final_band"))
    for path, source in candidates:
        if path and path.exists() and path.is_file():
            return path, source
    return None, ""


def _final_band_paste_order_key(root: Path, work_dir: Path, item: dict) -> tuple[int, int, int]:
    bbox = item.get("crop_bbox_in_translated_page") or [0, 0, 0, 0]
    try:
        y_top = int(item.get("band_y_top", bbox[1]) or 0)
        y_bottom = int(item.get("band_y_bottom", bbox[3]) or 0)
    except Exception:
        y_top = 0
        y_bottom = 0
    _, source_kind = _clean_final_band_source_for_crop_row(root, work_dir, item)
    source_priority = 1 if str(source_kind).startswith("post_copyback") else 0
    return (source_priority, -y_top, -y_bottom)


def _read_non_story_exclusions(root: Path) -> tuple[list[str], dict[str, str], int]:
    exclusions_path = root / "10_copyback_reassemble" / "non_story_exclusions.json"
    if not exclusions_path.exists():
        return [], {}, 0
    try:
        exclusions = json.loads(exclusions_path.read_text(encoding="utf-8", errors="replace"))
        rows = list(exclusions.get("exclusions") or [])
        bands = [
            str(row.get("band_id") or "").strip()
            for row in rows
            if str(row.get("band_id") or "").strip()
        ]
        reasons = {
            str(row.get("band_id") or "").strip(): str(row.get("exclusion_reason") or "").strip()
            for row in rows
            if str(row.get("band_id") or "").strip()
        }
        return bands, reasons, 0
    except Exception:
        return [], {}, 1


def _write_translated_page_band_consistency_audit(recorder, work_dir: Path) -> dict:
    audit = {
        "schema_version": 1,
        "source": "clean_final_bands_visible_area_overlap_owner",
        "passed": False,
        "rows_total": 0,
        "row_count": 0,
        "rows_failed": 0,
        "max_allowed": 12,
        "changed_gt8_policy": "changed_gt8 <= max(256, visible_pixels*0.002)",
        "overlap_policy": "compare only pixels owned by this band after final paste order",
        "excluded_non_story_bands": [],
        "excluded_non_story_reasons": {},
        "rows": [],
        "error_count": 0,
        "missing_count": 0,
    }
    if not recorder:
        return audit
    try:
        import cv2
        import numpy as np

        root = Path(work_dir) / "debug" / "e2e"
        excluded_bands, excluded_reasons, exclusion_errors = _read_non_story_exclusions(root)
        audit["excluded_non_story_bands"] = excluded_bands
        audit["excluded_non_story_reasons"] = excluded_reasons
        audit["error_count"] += exclusion_errors

        crops_path = root / "10_copyback_reassemble" / "final_band_crops.jsonl"
        if not crops_path.exists():
            audit["missing_count"] += 1
            return audit

        rows: list[dict] = []
        for line in crops_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                audit["error_count"] += 1
                continue
            if isinstance(row, dict):
                rows.append(row)

        rows_by_page: dict[str, list[dict]] = {}
        for row in rows:
            translated_name = str(row.get("translated_output_page") or "").strip()
            if translated_name:
                rows_by_page.setdefault(translated_name, []).append(row)

        audit_rows: list[dict] = []
        for translated_name, page_rows in sorted(rows_by_page.items()):
            translated_path = _final_rerender_resolve_translated_path(Path(work_dir), translated_name)
            translated_image = cv2.imread(str(translated_path), cv2.IMREAD_COLOR)
            if translated_image is None:
                audit["missing_count"] += len(page_rows)
                for row in page_rows:
                    audit_rows.append(
                        {
                            "band_id": str(row.get("band_id") or "").strip(),
                            "translated_output_page": translated_name,
                            "status": "fail",
                            "passed": False,
                            "flags": ["missing_translated_page"],
                        }
                    )
                continue

            page_h, page_w = translated_image.shape[:2]
            ordered = sorted(page_rows, key=lambda item: _final_band_paste_order_key(root, Path(work_dir), item))
            owner = np.full((page_h, page_w), -1, dtype=np.int32)
            clipped_bboxes: dict[int, list[int]] = {}
            for index, row in enumerate(ordered):
                bbox = _optional_bbox4(row.get("crop_bbox_in_translated_page"))
                if bbox is None:
                    continue
                x1 = max(0, min(page_w, bbox[0]))
                y1 = max(0, min(page_h, bbox[1]))
                x2 = max(0, min(page_w, bbox[2]))
                y2 = max(0, min(page_h, bbox[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                clipped_bboxes[index] = [x1, y1, x2, y2]
                owner[y1:y2, x1:x2] = index

            for index, row in enumerate(ordered):
                band_id = str(row.get("band_id") or "").strip()
                bbox = _optional_bbox4(row.get("crop_bbox_in_translated_page"))
                final_rel = str(row.get("final_crop_path") or "").strip()
                result = {
                    "band_id": band_id,
                    "translated_output_page": translated_name,
                    "crop_bbox_in_translated_page": bbox,
                    "status": "pass",
                    "passed": True,
                    "flags": [],
                    "visible_pixels": 0,
                    "max_diff": 0,
                    "changed_gt8": 0,
                    "shape_mismatch": False,
                    "final_y0_compared": 0,
                    "overlap_policy": "overlap_owner_visible_area",
                    "overlap_owner_band_ids": [],
                    "ignored_overlap_pixels": 0,
                }
                if bbox is None or index not in clipped_bboxes:
                    result["status"] = "fail"
                    result["passed"] = False
                    result["flags"] = ["missing_translated_crop"]
                    audit_rows.append(result)
                    continue

                final_path = root / final_rel if final_rel else None
                final_image = cv2.imread(str(final_path), cv2.IMREAD_COLOR) if final_path else None
                if final_image is None:
                    result["status"] = "fail"
                    result["passed"] = False
                    result["flags"] = ["missing_final_band_crop"]
                    audit_rows.append(result)
                    continue

                x1, y1, x2, y2 = clipped_bboxes[index]
                target_h = y2 - y1
                target_w = x2 - x1
                try:
                    final_y0 = max(
                        0,
                        int(round(float(row.get("output_page_y_top") or 0)))
                        - int(round(float(row.get("band_y_top") or 0))),
                    )
                except Exception:
                    final_y0 = 0
                final_x0 = max(0, x1 - int(bbox[0]))
                result["final_y0_compared"] = final_y0

                final_slice = final_image[
                    final_y0 : final_y0 + target_h,
                    final_x0 : final_x0 + target_w,
                    :,
                ]
                translated_slice = translated_image[y1:y2, x1:x2, :]
                if final_slice.shape[0] != target_h or final_slice.shape[1] != target_w:
                    result["shape_mismatch"] = True
                    if final_slice.size == 0:
                        result["status"] = "fail"
                        result["passed"] = False
                        result["flags"] = ["missing_final_band_crop"]
                        audit_rows.append(result)
                        continue
                    final_slice = cv2.resize(final_image, (target_w, target_h), interpolation=cv2.INTER_AREA)

                visible = owner[y1:y2, x1:x2] == index
                visible_pixels = int(visible.sum())
                ignored = owner[y1:y2, x1:x2][~visible]
                ignored_ids = sorted(
                    {
                        str(ordered[int(value)].get("band_id") or "").strip()
                        for value in ignored.tolist()
                        if int(value) >= 0 and int(value) < len(ordered)
                    }
                )
                result["visible_pixels"] = visible_pixels
                result["ignored_overlap_pixels"] = int((~visible).sum())
                result["overlap_owner_band_ids"] = ignored_ids
                if len(ignored_ids) == 1:
                    result["overlap_owner_band_id"] = ignored_ids[0]

                if visible_pixels > 0:
                    diff = np.abs(final_slice.astype(np.int16) - translated_slice.astype(np.int16)).max(axis=2)
                    visible_values = diff[visible]
                    max_diff = int(visible_values.max()) if visible_values.size else 0
                    changed_gt8 = int((visible_values > 8).sum()) if visible_values.size else 0
                else:
                    max_diff = 0
                    changed_gt8 = 0
                result["max_diff"] = max_diff
                result["changed_gt8"] = changed_gt8
                failed = max_diff > 12 and changed_gt8 > max(256, int(visible_pixels * 0.002))
                if failed:
                    result["status"] = "fail"
                    result["passed"] = False
                    result["flags"] = ["translated_crop_mismatch_final_band_visible_area"]
                audit_rows.append(result)

        audit["rows"] = audit_rows
        audit["rows_total"] = len(audit_rows)
        audit["row_count"] = len(audit_rows)
        audit["rows_failed"] = sum(1 for row in audit_rows if not row.get("passed"))
        audit["passed"] = audit["rows_failed"] == 0 and audit["error_count"] == 0 and audit["missing_count"] == 0
        try:
            recorder.write_json("10_copyback_reassemble/translated_page_band_consistency_audit.json", audit)
        except Exception:
            audit["error_count"] += 1
            audit["passed"] = False
    except Exception as exc:
        audit["error_count"] += 1
        audit["error"] = str(exc)
    return audit


def _restore_clean_final_bands_after_rerender(recorder, work_dir: Path) -> dict:
    audit = {
        "source": "clean_final_bands_after_all_rerenders",
        "after_final_project_image_rerender": True,
        "after_late_render_contract_repair": True,
        "final_guard_ran_after_final_project_image_rerender": True,
        "final_output_source": "clean_final_bands_after_all_rerenders",
        "seen_count": 0,
        "clean_band_source_used": 0,
        "translated_crop_fallback_used": 0,
        "clean_band_final_mismatch_count": 0,
        "final_band_written_count": 0,
        "translated_pages_recomposed_count": 0,
        "missing_count": 0,
        "error_count": 0,
        "sources_by_band": {},
        "excluded_non_story_bands": [],
        "excluded_non_story_reasons": {},
    }
    if not recorder:
        return audit
    try:
        import cv2
        import numpy as np

        root = Path(work_dir) / "debug" / "e2e"
        exclusions_path = root / "10_copyback_reassemble" / "non_story_exclusions.json"
        if exclusions_path.exists():
            try:
                exclusions = json.loads(exclusions_path.read_text(encoding="utf-8", errors="replace"))
                rows = list(exclusions.get("exclusions") or [])
                audit["excluded_non_story_bands"] = [
                    str(row.get("band_id") or "").strip()
                    for row in rows
                    if str(row.get("band_id") or "").strip()
                ]
                audit["excluded_non_story_reasons"] = {
                    str(row.get("band_id") or "").strip(): str(row.get("exclusion_reason") or "").strip()
                    for row in rows
                    if str(row.get("band_id") or "").strip()
                }
            except Exception:
                audit["error_count"] += 1
        crops_path = root / "10_copyback_reassemble" / "final_band_crops.jsonl"
        if not crops_path.exists():
            audit["missing_count"] += 1
            return audit
        rows: list[dict] = []
        for line in crops_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            audit["seen_count"] += 1
            try:
                row = json.loads(line)
            except Exception:
                audit["error_count"] += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
        expected_band_ids = [
            str(row.get("band_id") or "").strip()
            for row in rows
            if str(row.get("band_id") or "").strip()
        ]
        expected_page_ids: list[str] = []
        for row in rows:
            try:
                page_number = int(row.get("output_page_number") or 0)
            except Exception:
                page_number = 0
            if page_number <= 0:
                stem = Path(str(row.get("translated_output_page") or "")).stem
                page_number = int(stem) if stem.isdigit() else 0
            if page_number > 0:
                expected_page_ids.append(f"page_{page_number:03d}")
        recorder.set_canonical_expected_coverage(
            page_ids=expected_page_ids,
            band_ids=expected_band_ids,
        )
        page_cache: dict[str, np.ndarray] = {}
        rows_by_page: dict[str, list[dict]] = {}
        for row in rows:
            band_id = str(row.get("band_id") or "").strip()
            translated_name = str(row.get("translated_output_page") or "").strip()
            final_path = _resolve_debug_e2e_path(Path(work_dir), row.get("final_crop_path"))
            bbox = row.get("crop_bbox_in_translated_page")
            if not final_path:
                audit["missing_count"] += 1
                continue
            source_path, source_kind = _clean_final_band_source_for_crop_row(root, Path(work_dir), row)
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR) if source_path else None
            if image is None and translated_name and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                translated_path = _final_rerender_resolve_translated_path(Path(work_dir), translated_name)
                translated_image = cv2.imread(str(translated_path), cv2.IMREAD_COLOR)
                if translated_image is not None:
                    try:
                        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
                        h, w = translated_image.shape[:2]
                        x1 = max(0, min(w, x1))
                        x2 = max(0, min(w, x2))
                        y1 = max(0, min(h, y1))
                        y2 = max(0, min(h, y2))
                        if x2 > x1 and y2 > y1:
                            image = translated_image[y1:y2, x1:x2, :].copy()
                            source_kind = "translated_after_final_project_rerender_fallback"
                            audit["translated_crop_fallback_used"] += 1
                    except Exception:
                        audit["error_count"] += 1
            if image is None:
                audit["missing_count"] += 1
                continue
            if source_kind != "translated_after_final_project_rerender_fallback":
                audit["clean_band_source_used"] += 1
            existing = cv2.imread(str(final_path), cv2.IMREAD_COLOR)
            if existing is None or existing.shape != image.shape or bool(np.any(existing != image)):
                audit["clean_band_final_mismatch_count"] += 1
            final_rel = str(row.get("final_crop_path") or "").strip()
            if final_rel:
                page_id = band_id.split("_band_", 1)[0] if "_band_" in band_id else "page_unknown"
                recorder.write_canonical_image(
                    "final_band",
                    image,
                    page_id=page_id,
                    band_id=band_id,
                    color_space="bgr",
                )
                recorder.write_image(final_rel, image, quality=100)
                audit["final_band_written_count"] += 1
            if band_id:
                audit["sources_by_band"][band_id] = source_kind
            if translated_name:
                rows_by_page.setdefault(translated_name, []).append(row)

        for translated_name, page_rows in rows_by_page.items():
            translated_path = _final_rerender_resolve_translated_path(Path(work_dir), translated_name)
            page_image = page_cache.get(translated_name)
            if page_image is None:
                page_image = cv2.imread(str(translated_path), cv2.IMREAD_COLOR)
                if page_image is None:
                    audit["missing_count"] += 1
                    continue
            page_changed = False
            for row in sorted(page_rows, key=lambda item: _final_band_paste_order_key(root, Path(work_dir), item)):
                final_path = _resolve_debug_e2e_path(Path(work_dir), row.get("final_crop_path"))
                bbox = row.get("crop_bbox_in_translated_page")
                final_image = cv2.imread(str(final_path), cv2.IMREAD_COLOR) if final_path else None
                if final_image is None or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                try:
                    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
                except Exception:
                    audit["error_count"] += 1
                    continue
                h, w = page_image.shape[:2]
                x1 = max(0, min(w, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h, y1))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                target_h = y2 - y1
                target_w = x2 - x1
                try:
                    final_y0 = max(
                        0,
                        int(round(float(row.get("output_page_y_top") or 0)))
                        - int(round(float(row.get("band_y_top") or 0))),
                    )
                except Exception:
                    final_y0 = 0
                final_x0 = 0
                final_slice = final_image[
                    final_y0 : final_y0 + target_h,
                    final_x0 : final_x0 + target_w,
                    :,
                ]
                if final_slice.shape[0] != target_h or final_slice.shape[1] != target_w:
                    final_slice = cv2.resize(final_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
                page_image[y1:y2, x1:x2, :] = final_slice
                page_changed = True
            if page_changed:
                try:
                    page_number = int((page_rows[0] or {}).get("output_page_number") or 0)
                except Exception:
                    page_number = 0
                if page_number <= 0:
                    stem = Path(translated_name).stem
                    page_number = int(stem) if stem.isdigit() else 0
                if page_number > 0:
                    recorder.write_canonical_image(
                        "page",
                        page_image,
                        page_id=f"page_{page_number:03d}",
                        color_space="bgr",
                    )
                translated_path.parent.mkdir(parents=True, exist_ok=True)
                if translated_path.suffix.lower() in {".jpg", ".jpeg"}:
                    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
                    sampling_flag = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
                    sampling_444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
                    if sampling_flag is not None and sampling_444 is not None:
                        jpeg_params.extend([sampling_flag, sampling_444])
                    cv2.imwrite(str(translated_path), page_image, jpeg_params)
                else:
                    cv2.imwrite(str(translated_path), page_image)
                audit["translated_pages_recomposed_count"] += 1
        try:
            recorder.write_json("10_copyback_reassemble/final_band_crops_refresh.json", audit)
        except Exception:
            pass
    except Exception:
        audit["error_count"] += 1
    return audit


def _main_final_page_space_typeset_enabled() -> bool:
    raw = os.getenv("TRADUZAI_MAIN_FINAL_PAGE_SPACE_TYPESET", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _final_rerender_trace_ids_for_crop(row: dict) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for key in ("trace_ids", "source_trace_ids", "trace_id"):
        raw = row.get(key)
        candidates = raw if isinstance(raw, list) else [raw]
        for candidate in candidates:
            value = str(candidate or "").strip()
            if value and value not in seen:
                values.append(value)
                seen.add(value)
    return values


def _final_rerender_layer_identity(layer: dict) -> set[str]:
    identities: set[str] = set()
    for key in ("trace_id", "text_id", "id"):
        value = str(layer.get(key) or "").strip()
        if value:
            identities.add(value)
    for key in ("source_trace_ids", "source_text_ids"):
        raw = layer.get(key)
        candidates = raw if isinstance(raw, list) else [raw]
        for candidate in candidates:
            value = str(candidate or "").strip()
            if value:
                identities.add(value)
    return identities


def _final_rerender_layers_for_crop(row: dict, layers: list[dict]) -> list[dict]:
    band_id = str(row.get("band_id") or "").strip()
    trace_ids = set(_final_rerender_trace_ids_for_crop(row))
    matched: list[dict] = []
    if trace_ids:
        for layer in layers:
            if trace_ids & _final_rerender_layer_identity(layer):
                matched.append(layer)
    if matched:
        return matched
    return [
        layer
        for layer in layers
        if band_id and str(layer.get("band_id") or "").strip() == band_id
    ]


def _final_rerender_resolve_translated_path(work_dir: Path, translated_name: str) -> Path:
    path = Path(str(translated_name or "").strip())
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "translated":
        return Path(work_dir) / path
    return Path(work_dir) / "translated" / path


def _bbox_center4(bbox: list[int] | None) -> tuple[float, float] | None:
    if bbox is None:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _bbox_center_distance4(left: list[int] | None, right: list[int] | None) -> float | None:
    left_center = _bbox_center4(left)
    right_center = _bbox_center4(right)
    if left_center is None or right_center is None:
        return None
    return math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])


def _bbox_outside_ratio4(inner: list[int] | None, outer: list[int] | None) -> float:
    inner_area = _bbox_area4(inner)
    if inner_area <= 0 or outer is None:
        return 1.0
    return max(0.0, 1.0 - (_bbox_intersection_area4(inner, outer) / float(inner_area)))


def _render_tiny_ratio(render_bbox: list[int] | None, source_bbox: list[int] | None, balloon_bbox: list[int] | None) -> float:
    render_area = _bbox_area4(render_bbox)
    if render_area <= 0:
        return 0.0
    source_area = _bbox_area4(source_bbox)
    balloon_area = _bbox_area4(balloon_bbox)
    ratios: list[float] = []
    if source_area > 0:
        ratios.append(math.sqrt(render_area / float(source_area)))
        source_h = max(1, int(source_bbox[3]) - int(source_bbox[1])) if source_bbox else 1
        render_h = max(0, int(render_bbox[3]) - int(render_bbox[1])) if render_bbox else 0
        ratios.append(render_h / float(source_h))
    if balloon_area > 0:
        ratios.append(math.sqrt(render_area / float(balloon_area)))
    return min(ratios) if ratios else 1.0


def _dark_text_underfill_metrics(
    layer: dict,
    render_bbox: list[int] | None,
    safe_bbox: list[int] | None,
    balloon_bbox: list[int] | None,
) -> dict[str, object]:
    available_bbox = safe_bbox or balloon_bbox
    render_area = _bbox_area4(render_bbox)
    available_area = _bbox_area4(available_bbox)
    if render_bbox is None or available_bbox is None or render_area <= 0 or available_area <= 0:
        return {"underfilled": False}

    text = ""
    for key in ("translated", "texto_traduzido", "translation", "text", "original_text", "source_text"):
        value = str(layer.get(key) or "").strip()
        if value:
            text = value
            break
    words = [token for token in re.split(r"\s+", text) if token]
    alnum_count = sum(1 for char in text if char.isalnum())
    if alnum_count <= 14 and len(words) <= 2:
        return {"underfilled": False}

    render_h = max(0, int(render_bbox[3]) - int(render_bbox[1]))
    available_h = max(1, int(available_bbox[3]) - int(available_bbox[1]))
    height_ratio = render_h / float(available_h)
    area_ratio = render_area / float(available_area)

    if alnum_count < 24 and len(words) <= 3:
        min_height_ratio = 0.42
        min_area_ratio = 0.24
    else:
        min_height_ratio = 0.52
        min_area_ratio = 0.36

    return {
        "underfilled": height_ratio < min_height_ratio and area_ratio < min_area_ratio,
        "height_ratio": height_ratio,
        "area_ratio": area_ratio,
        "min_height_ratio": min_height_ratio,
        "min_area_ratio": min_area_ratio,
        "text_alnum_count": alnum_count,
        "text_word_count": len(words),
    }


def _layer_is_dark_final_rerender_subject(layer: dict) -> bool:
    source = str(layer.get("bubble_mask_source") or layer.get("bubbleMaskSource") or "").strip().lower()
    profile = str(layer.get("layout_profile") or layer.get("block_profile") or "").strip().lower()
    flags = {str(flag).strip().lower() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
    return bool(
        source in {"image_dark_bubble_mask", "image_dark_panel_mask"}
        or profile in {"dark_bubble", "dark_panel"}
        or bool(flags & {"dark_panel_style_grouped", "dark_oval_safe_height_expanded"})
    )


def _layer_dark_connected_key(layer: dict) -> str:
    for key in ("dark_connected_group_id", "connected_lobe_group_id", "dark_lobe_group_id", "bubble_id"):
        value = str(layer.get(key) or "").strip()
        if value:
            return value
    flags = {str(flag).strip().lower() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
    if "dark_panel_style_grouped" in flags:
        return str(layer.get("band_id") or layer.get("trace_id") or "").strip()
    return ""


def _final_rerender_row_status(flags: list[str]) -> str:
    fail_flags = {
        "translated_crop_mismatch_final_band",
        "missing_translated_crop",
        "missing_final_band_crop",
        "render_bbox_missing",
        "render_bbox_outside_crop",
        "render_bbox_outside_safe_or_balloon",
        "tiny_text",
        "clipped_text_flag",
        "dark_text_tiny",
        "dark_text_center_drift",
        "dark_connected_lobe_issue",
        "dark_text_outside_balloon",
        "dark_original_residual",
        "dark_text_underfilled",
    }
    warn_flags = {"center_drift", "no_matching_project_layer"}
    if any(flag in fail_flags for flag in flags):
        return "fail"
    if any(flag in warn_flags for flag in flags):
        return "warn"
    return "pass"


def _qa_translated_final_crops_against_layers(recorder, project_data: dict, work_dir: Path) -> dict:
    audit = {
        "source": "translated_after_final_project_rerender",
        "row_count": 0,
        "pass_count": 0,
        "warn_count": 0,
        "fail_count": 0,
        "rows": [],
    }
    if not recorder or not isinstance(project_data, dict):
        return audit
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        audit["rows"].append(
            {
                "band_id": "",
                "translated_output_page": "",
                "trace_ids": [],
                "status": "fail",
                "flags": ["qa_runtime_import_failed"],
                "metrics": {"error": str(exc)},
            }
        )
        audit["fail_count"] = 1
        audit["row_count"] = 1
        return audit

    root = Path(work_dir) / "debug" / "e2e"
    crop_rows = _load_debug_jsonl(root / "10_copyback_reassemble" / "final_band_crops.jsonl")
    project_layers = [layer for layer in _iter_project_text_layers(project_data) if isinstance(layer, dict)]
    dark_group_sizes: dict[str, int] = {}
    for layer in project_layers:
        if not _layer_is_dark_final_rerender_subject(layer):
            continue
        group_key = _layer_dark_connected_key(layer)
        if group_key:
            dark_group_sizes[group_key] = dark_group_sizes.get(group_key, 0) + 1

    rows: list[dict] = []
    for crop_row in crop_rows:
        band_id = str(crop_row.get("band_id") or "").strip()
        translated_name = str(crop_row.get("translated_output_page") or "").strip()
        trace_ids = _final_rerender_trace_ids_for_crop(crop_row)
        flags: list[str] = []
        metrics: dict[str, object] = {}
        bbox = _optional_bbox4(crop_row.get("crop_bbox_in_translated_page"))
        final_rel = str(crop_row.get("final_crop_path") or "").strip()
        if bbox is None or not translated_name:
            flags.append("missing_translated_crop")
        else:
            translated_path = _final_rerender_resolve_translated_path(Path(work_dir), translated_name)
            translated_image = cv2.imread(str(translated_path), cv2.IMREAD_COLOR)
            if translated_image is None:
                flags.append("missing_translated_crop")
            else:
                h, w = translated_image.shape[:2]
                x1 = max(0, min(w, bbox[0]))
                y1 = max(0, min(h, bbox[1]))
                x2 = max(0, min(w, bbox[2]))
                y2 = max(0, min(h, bbox[3]))
                if x2 <= x1 or y2 <= y1:
                    flags.append("missing_translated_crop")
                else:
                    translated_crop = translated_image[y1:y2, x1:x2, :]
                    final_path = root / final_rel if final_rel else None
                    final_crop = cv2.imread(str(final_path), cv2.IMREAD_COLOR) if final_path else None
                    if final_crop is None:
                        flags.append("missing_final_band_crop")
                    elif final_crop.shape != translated_crop.shape:
                        flags.append("translated_crop_mismatch_final_band")
                        metrics["translated_crop_matches_final_band"] = False
                        metrics["final_band_shape"] = list(final_crop.shape[:2])
                        metrics["translated_crop_shape"] = list(translated_crop.shape[:2])
                    else:
                        diff = np.abs(final_crop.astype(np.int16) - translated_crop.astype(np.int16))
                        mean_diff = float(np.mean(diff)) if diff.size else 0.0
                        max_diff = int(np.max(diff)) if diff.size else 0
                        metrics["translated_crop_mean_abs_diff"] = mean_diff
                        metrics["translated_crop_max_abs_diff"] = max_diff
                        matched = mean_diff <= 8.0 and max_diff <= 32
                        metrics["translated_crop_matches_final_band"] = matched
                        if not matched:
                            flags.append("translated_crop_mismatch_final_band")

        matching_layers = _final_rerender_layers_for_crop(crop_row, project_layers)
        if not matching_layers:
            flags.append("no_matching_project_layer")

        layer_metrics: list[dict] = []
        for layer in matching_layers:
            render_bbox = _optional_bbox4(layer.get("render_bbox"))
            source_bbox = (
                _optional_bbox4(layer.get("text_pixel_bbox"))
                or _optional_bbox4(layer.get("source_bbox"))
                or _optional_bbox4(layer.get("bbox"))
            )
            safe_bbox = _optional_bbox4(layer.get("safe_text_box"))
            balloon_bbox = (
                _optional_bbox4(layer.get("balloon_bbox"))
                or _optional_bbox4(layer.get("bubble_mask_bbox"))
                or _optional_bbox4(layer.get("target_bbox"))
            )
            layer_flags = {str(flag).strip().upper() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
            dark_layer = _layer_is_dark_final_rerender_subject(layer)
            layer_metric: dict[str, object] = {
                "trace_id": layer.get("trace_id") or layer.get("id"),
                "dark_bubble": dark_layer,
            }
            if render_bbox is None:
                flags.append("render_bbox_missing")
                layer_metric["render_bbox_inside_crop"] = False
            else:
                crop_intersection = _bbox_intersection_area4(render_bbox, bbox)
                layer_metric["render_bbox_inside_crop"] = crop_intersection > 0
                if crop_intersection <= 0:
                    flags.append("render_bbox_outside_crop")

                containment_bbox = safe_bbox or balloon_bbox
                outside_safe = _bbox_outside_ratio4(render_bbox, containment_bbox)
                outside_balloon = _bbox_outside_ratio4(render_bbox, balloon_bbox)
                layer_metric["render_bbox_inside_safe_or_balloon"] = outside_safe <= 0.10
                layer_metric["text_outside_balloon_ratio"] = outside_balloon
                if outside_safe > 0.20:
                    flags.append("render_bbox_outside_safe_or_balloon")

                center_drift = _bbox_center_distance4(render_bbox, source_bbox)
                if center_drift is not None:
                    source_area = _bbox_area4(source_bbox)
                    drift_limit = max(24.0, math.sqrt(float(source_area)) * 0.75) if source_area > 0 else 24.0
                    layer_metric["center_drift_px"] = center_drift
                    if center_drift > drift_limit:
                        flags.append("center_drift")

                tiny_ratio = _render_tiny_ratio(render_bbox, source_bbox, balloon_bbox)
                layer_metric["tiny_text_ratio"] = tiny_ratio
                if tiny_ratio < 0.35:
                    flags.append("tiny_text")

                if layer_flags & {"TEXT_CLIPPED", "TEXT_OVERFLOW"}:
                    flags.append("clipped_text_flag")

                if dark_layer:
                    metrics.setdefault("dark_layer_count", 0)
                    metrics["dark_layer_count"] = int(metrics["dark_layer_count"]) + 1
                    metrics["dark_text_tiny_ratio"] = min(float(metrics.get("dark_text_tiny_ratio", 1.0)), tiny_ratio)
                    if center_drift is not None:
                        metrics["dark_text_center_drift"] = max(
                            float(metrics.get("dark_text_center_drift", 0.0)),
                            center_drift,
                        )
                    underfill = _dark_text_underfill_metrics(layer, render_bbox, safe_bbox, balloon_bbox)
                    if "height_ratio" in underfill:
                        layer_metric["dark_text_underfilled_height_ratio"] = underfill["height_ratio"]
                        layer_metric["dark_text_underfilled_area_ratio"] = underfill["area_ratio"]
                        metrics["dark_text_underfilled_height_ratio"] = min(
                            float(metrics.get("dark_text_underfilled_height_ratio", 1.0)),
                            float(underfill["height_ratio"]),
                        )
                        metrics["dark_text_underfilled_area_ratio"] = min(
                            float(metrics.get("dark_text_underfilled_area_ratio", 1.0)),
                            float(underfill["area_ratio"]),
                        )
                        metrics["dark_text_underfilled_min_height_ratio"] = float(underfill["min_height_ratio"])
                        metrics["dark_text_underfilled_min_area_ratio"] = float(underfill["min_area_ratio"])
                        metrics["dark_text_underfilled_text_alnum_count"] = int(underfill["text_alnum_count"])
                        metrics["dark_text_underfilled_text_word_count"] = int(underfill["text_word_count"])
                    metrics["dark_text_outside_balloon_ratio"] = max(
                        float(metrics.get("dark_text_outside_balloon_ratio", 0.0)),
                        outside_balloon,
                    )
                    residual = 0.0
                    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
                    for key in ("dark_original_residual_score", "original_residual_score"):
                        try:
                            residual = max(residual, float(qa_metrics.get(key) or layer.get(key) or 0.0))
                        except Exception:
                            pass
                    metrics["dark_original_residual_score"] = max(
                        float(metrics.get("dark_original_residual_score", 0.0)),
                        residual,
                    )
                    group_key = _layer_dark_connected_key(layer)
                    connected = bool(group_key and dark_group_sizes.get(group_key, 0) > 1)
                    metrics["dark_connected_lobe_overlap"] = max(
                        float(metrics.get("dark_connected_lobe_overlap", 0.0)),
                        max(
                            [
                                _bbox_overlap_ratio4(render_bbox, _optional_bbox4(peer.get("render_bbox")))
                                for peer in project_layers
                                if peer is not layer and _layer_dark_connected_key(peer) == group_key
                            ]
                            or [0.0]
                        ),
                    )
                    if tiny_ratio < 0.35:
                        flags.append("dark_text_tiny")
                    if center_drift is not None and center_drift > 22.0:
                        flags.append("dark_text_center_drift")
                    if outside_balloon > 0.20:
                        flags.append("dark_text_outside_balloon")
                    if underfill.get("underfilled") is True:
                        flags.append("dark_text_underfilled")
                    if residual > 0.20:
                        flags.append("dark_original_residual")
                    if connected and (
                        tiny_ratio < 0.35
                        or (center_drift is not None and center_drift > 22.0)
                        or outside_balloon > 0.20
                        or underfill.get("underfilled") is True
                    ):
                        flags.append("dark_connected_lobe_issue")
            layer_metrics.append(layer_metric)

        deduped_flags = list(dict.fromkeys(flags))
        metrics["layers"] = layer_metrics
        row = {
            "band_id": band_id,
            "translated_output_page": translated_name,
            "trace_ids": trace_ids,
            "status": _final_rerender_row_status(deduped_flags),
            "flags": deduped_flags,
            "metrics": metrics,
        }
        rows.append(row)

    audit["rows"] = rows
    audit["row_count"] = len(rows)
    audit["pass_count"] = sum(1 for row in rows if row.get("status") == "pass")
    audit["warn_count"] = sum(1 for row in rows if row.get("status") == "warn")
    audit["fail_count"] = sum(1 for row in rows if row.get("status") == "fail")
    try:
        recorder.write_json(
            "11_qa_export_gate/final_rerender_visual_qa.json",
            {
                "source": audit["source"],
                "summary": {
                    "row_count": audit["row_count"],
                    "pass_count": audit["pass_count"],
                    "warn_count": audit["warn_count"],
                    "fail_count": audit["fail_count"],
                },
                "rows": rows,
            },
        )
        _write_debug_jsonl_replace(recorder, "11_qa_export_gate/final_rerender_visual_qa.jsonl", rows)
    except Exception as exc:
        audit["fail_count"] += 1
        audit["rows"].append(
            {
                "band_id": "",
                "translated_output_page": "",
                "trace_ids": [],
                "status": "fail",
                "flags": ["qa_artifact_write_failed"],
                "metrics": {"error": str(exc)},
            }
        )
    return audit


def _run_post_rerender_final_visual_contract(
    recorder,
    project_data: dict,
    work_dir: Path,
    *,
    after_final_project_image_rerender: bool,
    after_late_render_contract_repair: bool,
) -> dict:
    should_refresh_crops = bool(after_final_project_image_rerender)
    if should_refresh_crops:
        refresh_audit = _restore_clean_final_bands_after_rerender(recorder, work_dir)
    else:
        refresh_audit = {
            "source": "clean_final_bands_after_all_rerenders",
            "after_final_project_image_rerender": False,
            "after_late_render_contract_repair": False,
            "final_guard_ran_after_final_project_image_rerender": False,
            "final_output_source": "existing_final_bands_no_final_rerender",
            "seen_count": 0,
            "refreshed_count": 0,
            "missing_count": 0,
            "error_count": 0,
            "skipped_no_final_rerender": True,
        }
    refresh_audit["source"] = str(refresh_audit.get("source") or "clean_final_bands_after_all_rerenders")
    refresh_audit["after_final_project_image_rerender"] = bool(after_final_project_image_rerender)
    refresh_audit["after_late_render_contract_repair"] = bool(after_late_render_contract_repair)
    try:
        if recorder:
            recorder.write_json("10_copyback_reassemble/final_band_crops_refresh.json", refresh_audit)
    except Exception:
        pass
    qa_audit = _qa_translated_final_crops_against_layers(recorder, project_data, work_dir)
    translated_consistency_audit = _write_translated_page_band_consistency_audit(recorder, work_dir)
    return {"refresh": refresh_audit, "qa": qa_audit, "translated_page_band_consistency": translated_consistency_audit}


def _write_debug_export_gate_artifacts(recorder, project_data: dict) -> dict:
    if not recorder:
        return {}
    render_plan_sync = _write_debug_render_plan_final_from_project(recorder, project_data)
    qa = project_data.get("qa") if isinstance(project_data, dict) else {}
    if not isinstance(qa, dict):
        qa = {}
    summary = qa.get("summary") if isinstance(qa.get("summary"), dict) else {}
    export_gate = qa.get("export_gate") if isinstance(qa.get("export_gate"), dict) else {}
    summary_critical_flags = int(summary.get("critical_flag_count", summary.get("critical_count", 0)) or 0)
    summary_critical_issues = int(
        summary.get("critical_issue_count", export_gate.get("critical_issue_count", 0)) or 0
    )
    gate_critical_flags = int(export_gate.get("critical_flag_count", export_gate.get("critical_issue_count", 0)) or 0)
    gate_critical_issues = int(export_gate.get("critical_issue_count", 0) or 0)
    summary_highest = str(summary.get("highest_severity") or "none")
    consistency = {
        "summary": {
            "critical_count": summary_critical_flags,
            "critical_flag_count": summary_critical_flags,
            "critical_issue_count": summary_critical_issues,
            "highest_severity": summary_highest,
        },
        "export_gate": {
            "status": export_gate.get("status"),
            "critical_issue_count": gate_critical_issues,
            "critical_flag_count": gate_critical_flags,
            "review_issue_count": int(export_gate.get("review_issue_count", 0) or 0),
        },
        "consistency": {
            "critical_count_matches": summary_critical_flags == gate_critical_flags,
            "critical_issue_count_matches": summary_critical_issues == gate_critical_issues,
            "highest_severity_matches_block": (summary_highest == "critical") == (gate_critical_flags > 0),
        },
        "render_plan_sync": render_plan_sync.get("summary", {}),
    }
    consistency["consistent"] = all(bool(value) for value in consistency["consistency"].values())
    layers_by_id = {
        str(layer.get("id") or layer.get("text_id") or "").strip(): layer
        for layer in _iter_project_text_layers(project_data)
        if str(layer.get("id") or layer.get("text_id") or "").strip()
    }
    issues = [
        _normalize_export_issue_for_debug(issue, layers_by_id)
        for issue in export_gate.get("issues") or []
        if isinstance(issue, dict)
    ]
    visual_blockers = [issue for issue in issues if issue.get("severity") == "critical"]
    recorder.write_json("11_qa_export_gate/export_gate.json", export_gate)
    _write_debug_jsonl_replace(recorder, "11_qa_export_gate/qa_issues.jsonl", issues)
    _write_debug_jsonl_replace(recorder, "11_qa_export_gate/visual_blockers.jsonl", visual_blockers)
    recorder.write_json("11_qa_export_gate/qa_export_gate_consistency.json", consistency)
    try:
        from debug_tools.report import generate_debug_report

        generate_debug_report(Path(recorder.work_dir) / "debug" / "e2e")
    except Exception as exc:
        recorder.event("report", "debug_report_failed", {"error": str(exc)})
    return consistency


def _build_strip_inpainter_for_config(config: dict, real_inpaint_band_image):
    from types import SimpleNamespace

    if not config.get("skip_inpaint"):
        return SimpleNamespace(inpaint_band_image=real_inpaint_band_image)

    def _noop_inpaint_band_image(band_rgb, ocr_page: dict):
        if isinstance(ocr_page, dict):
            ocr_page["_skip_inpaint_honored"] = True
            ocr_page["_strip_used_fast_white_fill"] = False
            ocr_page["_strip_used_fast_local_fill"] = False
            ocr_page["_strip_used_real_inpaint"] = False
            ocr_page["_strip_used_post_cleanup"] = False
        return band_rgb.copy() if hasattr(band_rgb, "copy") else band_rgb

    return SimpleNamespace(inpaint_band_image=_noop_inpaint_band_image)


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
    from qa.export_gate import evaluate_export_gate

    export_gate = evaluate_export_gate(project)
    if config.get("mock_critical") and issues and export_gate.get("status") != "BLOCK":
        export_gate = {
            "status": "BLOCK",
            "allowed": False,
            "override": False,
            "issue_count": len(issues),
            "critical_issue_count": sum(1 for issue in issues if issue.get("severity") == "critical"),
            "critical_flag_count": sum(1 for issue in issues if issue.get("severity") == "critical"),
            "review_issue_count": sum(1 for issue in issues if issue.get("severity") == "high"),
            "review_flag_count": sum(1 for issue in issues if issue.get("severity") == "high"),
            "needs_review": any(issue.get("severity") == "high" for issue in issues),
            "issues": [
                {
                    "type": "p0_render_blocker",
                    "severity": issue.get("severity", "critical"),
                    "page": issue.get("page"),
                    "layer": issue.get("region_id"),
                    "text_id": issue.get("region_id"),
                    "flags": [issue.get("type")] if issue.get("type") else [],
                }
                for issue in issues
            ],
        }
    project["qa"]["export_gate"] = export_gate
    project["needs_review"] = export_gate["status"] == "BLOCK"
    project["output_review_state"] = _output_review_state_for_export_gate(export_gate)
    performance = {
        "total_sec": 0.001,
        "instrumented_sec": 0.001,
        "unattributed_sec": 0.0,
        "durations_sec": {},
        "events": [],
        "schema_version": 1,
        "mode": "mock",
        "sidecar_path": "performance_timing.json",
    }
    project["performance"] = performance
    project.setdefault("qa", {})["timing"] = performance
    (work_dir / "performance_timing.json").write_text(
        json.dumps(performance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_project_json(work_dir / "project.json", project)
    _write_mock_runner_reports(work_dir, project, issues)
    if config.get("strict") and any(issue["severity"] == "critical" for issue in issues):
        emit("error", message="Strict falhou: ha flags critical ativas")
        return 2
    emit("complete", output_path=str(work_dir))
    return 0


def _output_review_state_for_export_gate(export_gate: dict) -> str:
    status = str((export_gate or {}).get("status") or "PASS").upper()
    if status == "BLOCK":
        return "blocked_preview"
    if status == "OVERRIDDEN":
        return "overridden"
    return "approved"


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
        "engine_preset_id": config.get("engine_preset_id", ""),
        "mode": "manual" if config.get("skip_ocr") else "auto",
        "debug": config.get("debug", False),
        "skip_inpaint": config.get("skip_inpaint", False),
        "skip_ocr": config.get("skip_ocr", False),
        "strict": config.get("strict", False),
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
        _region, language_options = _page_action_options_from_args(sys.argv[6:])
        _run_process_block(project_json_path, page_idx, block_id, mode, language_options)
        return

    # Despachador unificado dos 4 handlers per-página (Fase 0 — sem falhas silenciosas).
    # Cada exceção é logada com [EditorAction] error e re-emitida como JSON "error"
    # para que o Rust e a UI vejam a mensagem em vez de terminar 1 sem motivo claro.
    _ACTION_DISPATCH = {
        "--detect-page": ("detect", _run_detect_page),
        "--detect-boxes-page": ("detect_boxes", _run_detect_boxes_page),
        "--ocr-page": ("ocr", _run_ocr_page),
        "--translate-page": ("translate", _run_translate_page),
        "--reinpaint-page": ("inpaint", _run_reinpaint),
    }
    if sys.argv[1] in _ACTION_DISPATCH and len(sys.argv) >= 4:
        action_name, handler = _ACTION_DISPATCH[sys.argv[1]]
        project_json_path = Path(sys.argv[2])
        page_idx = int(sys.argv[3])
        region, language_options = _page_action_options_from_args(sys.argv[4:])
        log_editor_action("start", action_name, page=page_idx)
        try:
            handler(project_json_path, page_idx, region, language_options)
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
    from vision_stack.engine_presets import resolve_engine_preset

    pipeline_timing = _PipelineTiming()
    start_time = time.time()
    with pipeline_timing.measure("load_config"):
        config = _load_json_file(config_path)
    debug_recorder = _bootstrap_debug_recorder(config, config_path)
    engine_preset = resolve_engine_preset(config, idioma_origem=config.get("idioma_origem", ""))
    config["engine_preset_id"] = engine_preset.id
    config["engine_preset"] = engine_preset.to_dict()
    config["work_title_user_provided"] = bool(config.get("work_title_user_provided"))
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
        from strip.run import (
            run_chapter,
            _clamp_page_inpaint_to_mask,
            _apply_dark_visual_text_geometry_cleanup,
            _page_requires_page_space_typeset,
            _page_texts_from_text_layers,
        )
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
            def run_ocr_stage(
                self,
                img,
                page_dict,
                work_title: str = "",
                work_title_user_provided: bool = False,
            ):
                return run_ocr_stage(
                    img,
                    page_dict,
                    profile="max",
                    idioma_origem=config.get("idioma_origem", "en"),
                    engine_preset_id=config.get("engine_preset_id", ""),
                    work_title=work_title or config.get("obra", ""),
                    work_title_user_provided=bool(work_title_user_provided or config.get("work_title_user_provided")),
                )

            def run_koharu_cjk_page(
                self,
                img,
                image_path,
                work_title: str = "",
                work_title_user_provided: bool = False,
            ):
                from vision_stack.runtime import _run_koharu_cjk_http_detect_ocr

                return _run_koharu_cjk_http_detect_ocr(
                    image_rgb=img,
                    image_label=str(image_path),
                    models_dir=str(models_dir),
                    profile="max",
                    idioma_origem=config.get("idioma_origem", "en"),
                    engine_preset_id=config.get("engine_preset_id", ""),
                    work_title=work_title or config.get("obra", ""),
                    work_title_user_provided=bool(work_title_user_provided or config.get("work_title_user_provided")),
                )

            def run_koharu_cjk_pages(
                self,
                jobs,
                models_dir="",
                idioma_origem="en",
                _models_dir=str(models_dir),
                work_title: str = "",
                work_title_user_provided: bool = False,
            ):
                resolved_models_dir = str(models_dir or _models_dir)
                resolved_idioma = idioma_origem or config.get("idioma_origem", "en")
                resolved_work_title = work_title or config.get("obra", "")
                resolved_title_user_provided = bool(work_title_user_provided or config.get("work_title_user_provided"))
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
                            engine_preset_id=config.get("engine_preset_id", ""),
                            work_title=resolved_work_title,
                            work_title_user_provided=resolved_title_user_provided,
                        )
                    except Exception as exc:
                        logger.warning("Koharu worker batch falhou; fallback para Koharu HTTP batch: %s", exc)

                from vision_stack.runtime import _run_koharu_cjk_http_detect_ocr_batch

                return _run_koharu_cjk_http_detect_ocr_batch(
                    jobs,
                    models_dir=resolved_models_dir,
                    profile="max",
                    idioma_origem=resolved_idioma,
                    engine_preset_id=config.get("engine_preset_id", ""),
                    work_title=resolved_work_title,
                    work_title_user_provided=resolved_title_user_provided,
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
                inpainter=_build_strip_inpainter_for_config(config, inpaint_band_image),
                typesetter=typesetter_mod,
                target_count=_resolve_strip_target_pages(config, total_pages),
                progress_callback=progress_cb,
                context=context,
                glossario=config.get("glossario", {}),
                idioma_origem=config.get("idioma_origem", "en"),
                idioma_destino=config.get("idioma_destino", "pt-BR"),
                obra=config.get("obra", ""),
                work_title_user_provided=bool(config.get("work_title_user_provided")),
                connected_reasoner_config=connected_reasoner_config,
                models_dir=str(models_dir),
                ollama_host=config.get("ollama_host", "http://localhost:11434"),
                ollama_model=config.get("ollama_model", "traduzai-translator"),
                translation_context=config.get("translation_context") or None,
                chapter_telemetry=strip_chapter_telemetry,
                skip_page_cleanup_rerender=bool(config.get("skip_inpaint")),
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
            main_sync_page_clamp_count = 0
            for p in output_pages:
                inpaint_target = images_dir / p.path.name
                if getattr(p, "inpainted_image", None) is not None:
                    try:
                        page_texts = _page_texts_from_text_layers(p.text_layers)
                        fixed_clean, fixed_rendered, did_clamp = _clamp_page_inpaint_to_mask(
                            original_image=getattr(p, "original_image", None),
                            clean_image=p.inpainted_image,
                            rendered_image=getattr(p, "image", None),
                            page_texts=page_texts,
                            inpaint_blocks=getattr(p, "inpaint_blocks", None),
                        )
                        if did_clamp:
                            p.inpainted_image = fixed_clean
                            p.image = fixed_rendered
                            main_sync_page_clamp_count += 1
                        dark_clean, dark_changed = _apply_dark_visual_text_geometry_cleanup(
                            p.inpainted_image,
                            page_texts,
                        )
                        if dark_changed:
                            p.inpainted_image = dark_clean
                            main_sync_page_clamp_count += 1
                    except Exception:
                        pass
                    cv2.imwrite(str(inpaint_target), p.inpainted_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                else:
                    shutil.copy2(p.path, inpaint_target)
            strip_chapter_telemetry["main_sync_page_clamp_count"] = main_sync_page_clamp_count

        with pipeline_timing.measure("sync_final_page_space_typeset"):
            final_page_space_count = 0
            from PIL import Image as _PILImage

            if bool(config.get("skip_final_page_space_typeset")) or not _main_final_page_space_typeset_enabled():
                strip_chapter_telemetry["main_final_page_space_typeset_skipped"] = True
                if not bool(config.get("skip_final_page_space_typeset")):
                    strip_chapter_telemetry["main_final_page_space_typeset_skip_reason"] = "opt_in_disabled"
                for page_number, p in enumerate(output_pages, start=1):
                    page_texts = _final_page_space_text_layers_for_renderer(p.text_layers, page_number=page_number)
                    _replace_output_page_text_layers(p, page_texts)
                    page_text_layers[page_number - 1] = p.text_layers
            else:
                for page_number, p in enumerate(output_pages, start=1):
                    page_texts = _final_page_space_text_layers_for_renderer(p.text_layers, page_number=page_number)
                    page_texts, page_space_safe_area_audit = _repair_page_space_text_layers_for_typeset(
                        page_texts,
                        page_number=page_number,
                    )
                    if int(page_space_safe_area_audit.get("safe_area_repaired_count") or 0) > 0:
                        strip_chapter_telemetry["main_final_page_space_safe_area_repair_count"] = (
                            int(strip_chapter_telemetry.get("main_final_page_space_safe_area_repair_count") or 0)
                            + int(page_space_safe_area_audit.get("safe_area_repaired_count") or 0)
                        )
                    if not _page_requires_page_space_typeset(page_texts):
                        continue
                    clean_path = images_dir / p.path.name
                    if not clean_path.exists():
                        continue
                    clean_bgr = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
                    if clean_bgr is None:
                        continue
                    audit_flags = _page_text_coordinate_audit_flags(
                        page_texts,
                        height=int(clean_bgr.shape[0]),
                        width=int(clean_bgr.shape[1]),
                    )
                    if audit_flags:
                        _append_page_text_flags(page_texts, audit_flags)
                        _replace_output_page_text_layers(p, page_texts)
                        page_text_layers[page_number - 1] = p.text_layers
                        strip_chapter_telemetry["main_final_page_space_rerender_blocked_count"] = (
                            int(strip_chapter_telemetry.get("main_final_page_space_rerender_blocked_count") or 0) + 1
                        )
                        continue
                    page_stub = {"numero": page_number, "image_layers": {}, "text_layers": page_texts}
                    if _persist_real_bubble_mask_layer_for_page(
                        page_stub,
                        getattr(p, "ocr_result", None),
                        work_dir,
                        page_number=page_number,
                        image_size=(int(clean_bgr.shape[1]), int(clean_bgr.shape[0])),
                    ):
                        page_texts = page_stub["text_layers"]
                        page_texts, persisted_safe_area_audit = _repair_page_space_text_layers_for_typeset(
                            page_texts,
                            page_number=page_number,
                        )
                        if int(persisted_safe_area_audit.get("safe_area_repaired_count") or 0) > 0:
                            strip_chapter_telemetry["main_final_page_space_safe_area_repair_count"] = (
                                int(strip_chapter_telemetry.get("main_final_page_space_safe_area_repair_count") or 0)
                                + int(persisted_safe_area_audit.get("safe_area_repaired_count") or 0)
                            )
                    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
                    try:
                        rendered_rgb = typesetter_mod.render_band_image(
                            clean_rgb,
                            {"texts": page_texts, "_coordinate_space": "page"},
                        )
                    except Exception:
                        continue
                    _replace_output_page_text_layers(p, page_texts)
                    page_text_layers[page_number - 1] = p.text_layers
                    typesetter_mod.save_typeset_page_image(_PILImage.fromarray(rendered_rgb), p.path, quality=95)
                    p.image = rendered_rgb
                    final_page_space_count += 1
            strip_chapter_telemetry["main_final_page_space_typeset_count"] = final_page_space_count

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
    with pipeline_timing.measure("normalize_project_render_geometry"):
        synced_render_bboxes = _normalize_project_render_balloon_bboxes(project_data)
        if synced_render_bboxes:
            project_data.setdefault("qa", {}).setdefault("geometry_normalization", {})[
                "render_balloon_bbox_sync_count"
            ] = synced_render_bboxes
    with pipeline_timing.measure("hydrate_project_render_metadata"):
        render_metadata_hydration = _hydrate_project_render_metadata_from_debug_candidates(project_data)
        project_data.setdefault("qa", {})["render_metadata_hydration"] = render_metadata_hydration
    with pipeline_timing.measure("ensure_route_action_contract"):
        route_contract_audit = _ensure_project_route_action_contract(project_data)
        project_data.setdefault("qa", {})["route_action_contract"] = route_contract_audit
    with pipeline_timing.measure("ensure_mask_evidence_contract"):
        filled_mask_evidence = _ensure_project_mask_evidence(project_data)
        if filled_mask_evidence:
            project_data.setdefault("qa", {}).setdefault("mask_evidence_contract", {})[
                "filled_missing_count"
            ] = filled_mask_evidence
    with pipeline_timing.measure("ensure_render_contract"):
        render_contract_audit = _ensure_project_render_contract(project_data)
        project_data.setdefault("qa", {})["render_contract_audit"] = render_contract_audit
    try:
        from qa.translation_qa import summarize_flags

        with pipeline_timing.measure("propagate_debug_qa_flags"):
            qa_flag_audit = _propagate_debug_qa_flags_to_project(project_data)
            qa_summary = summarize_flags(
                [layer for page in project_data.get("paginas") or [] for layer in page.get("text_layers") or []]
            )
            project_data.setdefault("qa", {})["summary"] = _augment_qa_summary_with_debug_contract(
                qa_summary,
                qa_flag_audit,
            )
    except Exception as exc:
        logger.warning("Falha ao propagar flags QA do debug E2E: %s", exc)
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
        from qa.translation_qa import summarize_flags

        with pipeline_timing.measure("normalize_final_project_page_space_layers"):
            final_page_space_audit = _normalize_final_project_page_space_layers(project_data)
            qa_summary = summarize_flags(
                [
                    layer
                    for page in project_data.get("paginas") or []
                    for layer in _project_page_text_layers(page)
                ]
            )
            project_data.setdefault("qa", {})["summary"] = {
                **project_data.get("qa", {}).get("summary", {}),
                **qa_summary,
                "final_page_space_normalization": final_page_space_audit,
            }
    except Exception as exc:
        logger.warning("Falha ao normalizar camadas finais em page-space: %s", exc)
    try:
        with pipeline_timing.measure("hydrate_final_project_render_metadata"):
            final_render_metadata_hydration = _hydrate_project_render_metadata_from_debug_candidates(project_data)
            debug_mask_bbox_repair = _repair_project_bubble_bboxes_from_debug_masks(project_data)
            real_bubble_safe_area_repair = _repair_project_real_bubble_body_safe_areas(project_data)
            same_balloon_fragments_merged = _merge_same_balloon_fragment_layers(project_data)
            distinct_dark_lobe_payload_merge_repairs = _repair_distinct_dark_lobe_project_payload_merges(
                list(_iter_project_text_layers(project_data))
            )
            distinct_hidden_nonfragment_restored = _restore_hidden_distinct_nonfragment_layers(project_data)
            same_identity_fragments_suppressed = _suppress_same_identity_merged_fragments(project_data)
            cross_page_band_layers_rehomed = _rehome_cross_page_band_layers(project_data)
            broad_fallback_layers_suppressed = _suppress_broad_fallback_merge_layers(project_data)
            final_page_space_after_hydration = _normalize_final_project_page_space_layers(project_data)
            post_page_space_render_metadata_hydration = _hydrate_project_render_metadata_from_debug_candidates(project_data)
            post_page_space_real_bubble_safe_area_repair = _repair_project_real_bubble_body_safe_areas(project_data)
            post_page_space_same_balloon_fragments_merged = _merge_same_balloon_fragment_layers(project_data)
            post_page_space_distinct_dark_lobe_payload_merge_repairs = _repair_distinct_dark_lobe_project_payload_merges(
                list(_iter_project_text_layers(project_data))
            )
            post_page_space_distinct_hidden_nonfragment_restored = _restore_hidden_distinct_nonfragment_layers(project_data)
            post_page_space_same_identity_fragments_suppressed = _suppress_same_identity_merged_fragments(project_data)
            post_page_space_cross_page_band_layers_rehomed = _rehome_cross_page_band_layers(project_data)
            scrubbed_local_auxiliary_bboxes = _scrub_project_local_auxiliary_bboxes(project_data)
            final_distinct_dark_lobe_geometry_repairs = _finalize_distinct_dark_lobe_project_geometry(
                list(_iter_project_text_layers(project_data))
            )
            final_render_geometry_clamp = _clamp_project_render_geometry_to_page(project_data)
            final_render_contract_audit = _ensure_project_render_contract(project_data)
            dark_panel_glow_styles = _apply_dark_panel_glow_project_styles(project_data)
            dark_panel_visual_safe_area_repair = _repair_dark_panel_visual_mask_safe_areas(project_data)
            dark_panel_style_groups = _apply_dark_panel_style_groups(project_data)
            post_style_distinct_dark_lobe_geometry_repairs = _finalize_distinct_dark_lobe_project_geometry(
                list(_iter_project_text_layers(project_data))
            )
            post_style_component_safe_partitions = _partition_dark_connected_lobe_safe_boxes_from_components(project_data)
            post_style_render_geometry_clamp = _clamp_project_render_geometry_to_page(project_data)
            post_style_render_contract_audit = _ensure_project_render_contract(project_data)
            project_data.setdefault("qa", {})["final_render_metadata_hydration"] = final_render_metadata_hydration
            project_data.setdefault("qa", {})[
                "post_page_space_render_metadata_hydration"
            ] = post_page_space_render_metadata_hydration
            project_data.setdefault("qa", {})["debug_mask_bbox_repair"] = debug_mask_bbox_repair
            project_data.setdefault("qa", {})["real_bubble_safe_area_repair"] = real_bubble_safe_area_repair
            project_data.setdefault("qa", {})[
                "post_page_space_real_bubble_safe_area_repair"
            ] = post_page_space_real_bubble_safe_area_repair
            project_data.setdefault("qa", {})["same_balloon_fragment_merge_count"] = same_balloon_fragments_merged
            project_data.setdefault("qa", {})[
                "post_page_space_same_balloon_fragment_merge_count"
            ] = post_page_space_same_balloon_fragments_merged
            project_data.setdefault("qa", {})[
                "distinct_dark_lobe_payload_merge_repair_count"
            ] = distinct_dark_lobe_payload_merge_repairs
            project_data.setdefault("qa", {})[
                "post_page_space_distinct_dark_lobe_payload_merge_repair_count"
            ] = post_page_space_distinct_dark_lobe_payload_merge_repairs
            project_data.setdefault("qa", {})[
                "distinct_hidden_nonfragment_restored_count"
            ] = distinct_hidden_nonfragment_restored
            project_data.setdefault("qa", {})[
                "post_page_space_distinct_hidden_nonfragment_restored_count"
            ] = post_page_space_distinct_hidden_nonfragment_restored
            project_data.setdefault("qa", {})[
                "same_identity_fragment_suppressed_count"
            ] = same_identity_fragments_suppressed
            project_data.setdefault("qa", {})[
                "post_page_space_same_identity_fragment_suppressed_count"
            ] = post_page_space_same_identity_fragments_suppressed
            project_data.setdefault("qa", {})["cross_page_band_layers_rehomed_count"] = cross_page_band_layers_rehomed
            project_data.setdefault("qa", {})[
                "post_page_space_cross_page_band_layers_rehomed_count"
            ] = post_page_space_cross_page_band_layers_rehomed
            project_data.setdefault("qa", {})[
                "scrubbed_local_auxiliary_bbox_count"
            ] = scrubbed_local_auxiliary_bboxes
            project_data.setdefault("qa", {})[
                "final_distinct_dark_lobe_geometry_repair_count"
            ] = final_distinct_dark_lobe_geometry_repairs
            project_data.setdefault("qa", {})[
                "post_style_distinct_dark_lobe_geometry_repair_count"
            ] = post_style_distinct_dark_lobe_geometry_repairs
            project_data.setdefault("qa", {})[
                "post_style_component_safe_partition_count"
            ] = post_style_component_safe_partitions
            project_data.setdefault("qa", {})[
                "post_style_render_geometry_clamp"
            ] = post_style_render_geometry_clamp
            project_data.setdefault("qa", {})[
                "post_style_render_contract_audit"
            ] = post_style_render_contract_audit
            project_data.setdefault("qa", {})["broad_fallback_render_suppressed_count"] = broad_fallback_layers_suppressed
            project_data.setdefault("qa", {})["final_render_geometry_clamp"] = final_render_geometry_clamp
            project_data.setdefault("qa", {})["final_render_contract_audit"] = final_render_contract_audit
            project_data.setdefault("qa", {})["final_page_space_after_hydration"] = final_page_space_after_hydration
            project_data.setdefault("qa", {})["dark_panel_glow_style_count"] = dark_panel_glow_styles
            project_data.setdefault("qa", {})["dark_panel_style_groups"] = dark_panel_style_groups
            project_data.setdefault("qa", {})[
                "dark_panel_visual_safe_area_repair_count"
            ] = dark_panel_visual_safe_area_repair
            final_qa_flag_audit = _propagate_debug_qa_flags_to_project(project_data)
            final_post_qa_render_contract_audit = _ensure_project_render_contract(project_data)
            project_data.setdefault("qa", {})[
                "final_post_qa_render_contract_audit"
            ] = final_post_qa_render_contract_audit
            try:
                from qa.translation_qa import summarize_flags

                qa_summary = summarize_flags(
                    [
                        layer
                        for page in project_data.get("paginas") or []
                        for layer in _project_page_text_layers(page)
                    ]
                )
                project_data.setdefault("qa", {})["summary"] = _augment_qa_summary_with_debug_contract(
                    {
                        **project_data.get("qa", {}).get("summary", {}),
                        **qa_summary,
                    },
                    final_qa_flag_audit,
                )
            except Exception as exc:
                logger.warning("Falha ao resumir QA apÃ³s propagaÃ§Ã£o final de flags: %s", exc)
    except Exception as exc:
        logger.warning("Falha ao hidratar metadata de render final: %s", exc)
    try:
        from qa.translation_qa import summarize_flags

        with pipeline_timing.measure("final_project_coordinate_audit"):
            final_coordinate_audit = _apply_final_project_coordinate_audit(project_data)
            project_data.setdefault("qa", {}).setdefault("summary", {})[
                "final_coordinate_audit"
            ] = final_coordinate_audit
            if int(final_coordinate_audit.get("flags_added") or 0) > 0:
                qa_summary = summarize_flags(
                    [
                        layer
                        for page in project_data.get("paginas") or []
                        for layer in _project_page_text_layers(page)
                    ]
                )
                project_data.setdefault("qa", {})["summary"] = {
                    **project_data.get("qa", {}).get("summary", {}),
                    **qa_summary,
                    "final_coordinate_audit": final_coordinate_audit,
                }
    except Exception as exc:
        logger.warning("Falha ao aplicar auditoria final de coordenadas: %s", exc)
    try:
        with pipeline_timing.measure("rerender_final_project_images"):
            try:
                _sync_project_legacy_aliases(project_data)
            except Exception as exc:
                logger.warning("Falha ao sincronizar aliases antes do rerender final: %s", exc)
            if bool(config.get("skip_final_page_space_typeset")):
                final_rerender_audit = {
                    "pages_checked": 0,
                    "pages_rerendered": 0,
                    "errors": [],
                    "skipped": True,
                    "skip_reason": "skip_final_page_space_typeset",
                }
            else:
                final_rerender_audit = _rerender_final_project_images_from_metadata(project_data, work_dir)
            if bool(config.get("skip_final_page_space_typeset")):
                post_rerender_contract_audit = {
                    "post_rerender_contract_audit": {},
                    "pages_checked": 0,
                    "pages_rerendered": 0,
                    "errors": [],
                    "skipped": True,
                    "skip_reason": "skip_final_page_space_typeset",
                }
            else:
                post_rerender_contract_audit = _rerender_final_project_images_after_contract(project_data, work_dir)
            project_data.setdefault("qa", {}).setdefault("summary", {})[
                "final_project_image_rerender"
            ] = final_rerender_audit
            project_data.setdefault("qa", {}).setdefault("summary", {})[
                "post_rerender_contract_repair"
            ] = post_rerender_contract_audit
            project_data.setdefault("qa", {})[
                "post_rerender_contract_repair"
            ] = post_rerender_contract_audit
    except Exception as exc:
        logger.warning("Falha ao rerenderizar imagens finais a partir do project.json: %s", exc)
    try:
        from qa.translation_qa import summarize_flags

        with pipeline_timing.measure("final_non_bubble_panel_mask_flag_cleanup"):
            panel_mask_flags_cleared = _clear_non_bubble_panel_mask_flags(project_data)
            valid_image_bubble_mask_flags_cleared = _clear_stale_valid_image_bubble_mask_flags(project_data)
            stale_panel_weak_residual_flags_cleared = _clear_stale_panel_weak_residual_flags(project_data)
            if panel_mask_flags_cleared or valid_image_bubble_mask_flags_cleared or stale_panel_weak_residual_flags_cleared:
                qa_summary = summarize_flags(
                    [
                        layer
                        for page in project_data.get("paginas") or []
                        for layer in _project_page_text_layers(page)
                    ]
                )
                project_data.setdefault("qa", {})["summary"] = {
                    **project_data.get("qa", {}).get("summary", {}),
                    **qa_summary,
                    "non_bubble_panel_mask_flags_cleared": panel_mask_flags_cleared,
                    "valid_image_bubble_mask_flags_cleared": valid_image_bubble_mask_flags_cleared,
                    "stale_panel_weak_residual_flags_cleared": stale_panel_weak_residual_flags_cleared,
                }
    except Exception as exc:
        logger.warning("Falha ao limpar flags finais de painel sem balão: %s", exc)
    try:
        with pipeline_timing.measure("late_render_contract_repair"):
            try:
                _sync_project_legacy_aliases(project_data)
            except Exception as exc:
                logger.warning("Falha ao sincronizar aliases antes do reparo tardio de render: %s", exc)
            late_render_contract_repair = _rerender_final_project_images_after_contract(project_data, work_dir)
            project_data.setdefault("qa", {})[
                "late_render_contract_repair"
            ] = late_render_contract_repair
            project_data.setdefault("qa", {}).setdefault("summary", {})[
                "late_render_contract_repair"
            ] = late_render_contract_repair
    except Exception as exc:
        logger.warning("Falha ao aplicar reparo tardio de contrato de render: %s", exc)
    try:
        with pipeline_timing.measure("post_rerender_final_visual_contract"):
            post_rerender_visual_contract = _run_post_rerender_final_visual_contract(
                debug_recorder,
                project_data,
                work_dir,
                after_final_project_image_rerender=bool(
                    int((locals().get("final_rerender_audit") or {}).get("pages_rerendered") or 0)
                    or int((locals().get("post_rerender_contract_audit") or {}).get("pages_rerendered") or 0)
                    or int((locals().get("late_render_contract_repair") or {}).get("pages_rerendered") or 0)
                ),
                after_late_render_contract_repair=True,
            )
            project_data.setdefault("qa", {})[
                "post_rerender_final_visual_contract"
            ] = post_rerender_visual_contract
            project_data.setdefault("qa", {}).setdefault("summary", {})[
                "post_rerender_final_visual_contract"
            ] = {
                "refreshed_count": int(
                    (post_rerender_visual_contract.get("refresh") or {}).get("refreshed_count") or 0
                ),
                "qa_fail_count": int(
                    (post_rerender_visual_contract.get("qa") or {}).get("fail_count") or 0
                ),
                "qa_warn_count": int(
                    (post_rerender_visual_contract.get("qa") or {}).get("warn_count") or 0
                ),
            }
            strip_chapter_telemetry["debug_final_band_crops_refreshed_count"] = int(
                (post_rerender_visual_contract.get("refresh") or {}).get("refreshed_count") or 0
            )
    except Exception as exc:
        logger.warning("Falha ao executar QA visual pós-rerender final: %s", exc)
    try:
        from qa.export_gate import evaluate_export_gate

        with pipeline_timing.measure("evaluate_export_gate"):
            export_gate = evaluate_export_gate(
                project_data,
                override=bool(config.get("allow_p0_export_override")),
            )
            project_data["qa"]["export_gate"] = export_gate
            project_data["needs_review"] = export_gate["status"] == "BLOCK"
            project_data["output_review_state"] = _output_review_state_for_export_gate(export_gate)
            if debug_recorder:
                _write_debug_export_gate_artifacts(debug_recorder, project_data)
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
    project_data.setdefault("qa", {})["timing"] = project_data["performance"]
    with pipeline_timing.measure("save_project_json"):
        _save_project_json(work_dir / "project.json", project_data)
    with pipeline_timing.measure("finalize_decision_trace"):
        finalize_decision_trace(
            {
                "total_paginas": total_pages,
                "total_textos": project_data.get("estatisticas", {}).get("total_textos", 0),
                "qa_summary": (project_data.get("qa") or {}).get("summary") or {},
                "export_gate": (project_data.get("qa") or {}).get("export_gate") or {},
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
        if debug_recorder:
            debug_recorder.write_json("00_run/performance_timing_snapshot.json", final_performance)
    except Exception as exc:
        logger.warning("Falha ao escrever performance_timing.json: %s", exc)
    export_gate = ((project_data.get("qa") or {}).get("export_gate") or {})
    if _strict_export_gate_active(config) and export_gate.get("status") == "BLOCK":
        audit = {
            "strict_mode_active": True,
            "export_gate_status": export_gate.get("status"),
            "expected_exit_code": 2,
            "actual_exit_code": 2,
            "stdout_last_event_type": "error",
        }
        if debug_recorder:
            debug_recorder.write_json("11_qa_export_gate/strict_exit_audit.json", audit)
        emit("error", message="Strict falhou: export gate bloqueou a exportacao")
        if debug_recorder:
            _finalize_debug_recorder(
                debug_recorder,
                config_snapshot=config,
                extra={"exit_code": 2, "export_gate_status": export_gate.get("status")},
            )
        sys.exit(2)
    emit_progress("typeset", 100, 100, message="Concluido!")
    emit("complete", output_path=str(work_dir))
    _finalize_debug_recorder(
        debug_recorder,
        config_snapshot=config,
        extra={"exit_code": 0, "export_gate_status": export_gate.get("status")},
    )


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
        "curva": False,
        "curva_direcao": "",
        "curva_intensidade": 0.0,
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


def _style_evidence_to_dict(evidence) -> dict | None:
    if evidence is None:
        return None
    if isinstance(evidence, dict):
        return copy.deepcopy(evidence)
    if hasattr(evidence, "to_dict"):
        data = evidence.to_dict()
        return copy.deepcopy(data) if isinstance(data, dict) else None
    return None


def _load_source_image_rgb_for_style(image_path: str | Path, texts: list[dict]):
    if not any(isinstance(text, dict) and _style_evidence_to_dict(text.get("style_evidence")) is None for text in texts):
        return None
    try:
        from PIL import Image
        import numpy as np

        with Image.open(image_path) as image:
            return np.array(image.convert("RGB"))
    except Exception:
        logger.debug("Falha ao carregar imagem original para extracao de estilo", exc_info=True)
        return None


def _style_evidence_confidence(evidence: dict | None) -> float:
    if not evidence:
        return 0.0

    confidence_fields: list[str] = []
    if evidence.get("text_color"):
        confidence_fields.append("text_color_confidence")
    if evidence.get("stroke_color") or evidence.get("stroke_width_px"):
        confidence_fields.append("stroke_confidence")
    if evidence.get("font_name"):
        confidence_fields.append("font_confidence")
    if evidence.get("shadow") is True:
        confidence_fields.append("shadow_confidence")
    if evidence.get("glow") is True:
        confidence_fields.append("glow_confidence")
    if evidence.get("gradient") is True and evidence.get("gradient_colors"):
        confidence_fields.append("gradient_confidence")
    if evidence.get("curved") is True:
        confidence_fields.append("curve_confidence")

    confidences: list[float] = []
    for field in confidence_fields:
        try:
            confidences.append(float(evidence.get(field) or 0.0))
        except (TypeError, ValueError):
            continue
    return max(confidences, default=0.0)


def _get_source_style_font_detector():
    try:
        from vision_stack.runtime import _get_font_detector

        return _get_font_detector()
    except Exception:
        logger.debug("FontDetector indisponivel para copia de estilo", exc_info=True)
        return None


def _style_font_context_for_text(ocr_text: dict) -> str | None:
    background_rgb = _coerce_background_rgb(ocr_text.get("background_rgb"))
    candidate = dict(ocr_text or {})
    candidate["style_origin"] = "auto"
    if _should_apply_visual_card_project_style(candidate, background_rgb):
        return "visual_card"
    return None


def _extract_style_evidence_from_source_image(
    source_image_rgb,
    bbox: list[int] | None,
    *,
    font_detector=None,
    font_context: str | None = None,
) -> dict | None:
    if source_image_rgb is None or not bbox:
        return None
    if not hasattr(source_image_rgb, "shape") or len(source_image_rgb.shape) < 2:
        return None

    height, width = source_image_rgb.shape[:2]
    x1, y1, x2, y2 = _bbox4(bbox)
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    try:
        from typesetter.style_extractor import extract_text_style_evidence

        crop = source_image_rgb[y1:y2, x1:x2, :3]
        kwargs = {}
        if font_detector is not None:
            kwargs["font_detector"] = font_detector
        if font_context:
            kwargs["font_context"] = font_context
        return _style_evidence_to_dict(extract_text_style_evidence(crop, **kwargs))
    except Exception:
        logger.debug("Falha ao extrair evidencia de estilo do recorte de texto", exc_info=True)
        return None


def _non_empty_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _looks_like_renderable_latin_phrase(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    upper = text.upper()
    if _looks_like_scanlator_text(upper):
        return False
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text):
        return False
    letters = re.findall(r"[A-Za-z]", text)
    words = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", text)
    return len(letters) >= 10 and len(words) >= 2


def _looks_like_scanlator_text(value: object) -> bool:
    upper = str(value or "").upper()
    watermark_terms = ("THUNDERSCANS", "ASURACOMIC", "CHAPTER ON", "READ THE CHAPTER")
    return any(term in upper for term in watermark_terms)


def _hex_luma(value: object) -> float:
    text = str(value or "").strip().lstrip("#")
    if len(text) < 6:
        return 0.0
    try:
        r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return 0.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_confidence_from_fields(record: dict, fields: tuple[str, ...]) -> float | None:
    for field in fields:
        value = _float_or_none(record.get(field))
        if value is not None:
            return value
    return None


def _primary_text_style_candidate_confident(ocr_text: dict) -> bool:
    confidence = _candidate_confidence_from_fields(
        ocr_text,
        ("confidence", "ocr_confidence", "confianca_ocr"),
    )
    if confidence is None:
        return True
    return confidence >= STYLE_COPY_CANDIDATE_CONFIDENCE_THRESHOLD


def _sfx_style_candidate_confident(ocr_text: dict) -> bool:
    sfx = ocr_text.get("sfx") if isinstance(ocr_text.get("sfx"), dict) else {}
    sfx_ocr = ocr_text.get("sfx_ocr") if isinstance(ocr_text.get("sfx_ocr"), dict) else {}
    confidence_values = [
        _float_or_none(ocr_text.get("sfx_promotion_score")),
        _float_or_none(sfx.get("promotion_score")),
        _float_or_none(ocr_text.get("confidence")),
        _float_or_none(ocr_text.get("ocr_confidence")),
        _float_or_none(sfx.get("visual_confidence")),
        _float_or_none(sfx_ocr.get("confidence")),
        _float_or_none(sfx_ocr.get("ocr_confidence")),
    ]
    confidence_values = [value for value in confidence_values if value is not None]
    if not confidence_values:
        return True
    promotion_score = _float_or_none(ocr_text.get("sfx_promotion_score"))
    if promotion_score is None:
        promotion_score = _float_or_none(sfx.get("promotion_score"))
    if promotion_score is not None and promotion_score >= STYLE_COPY_SFX_PROMOTION_THRESHOLD:
        return True
    return max(confidence_values) >= STYLE_COPY_CANDIDATE_CONFIDENCE_THRESHOLD


def _style_evidence_allows_visual_text_without_ocr(ocr_text: dict, evidence: dict | None) -> bool:
    if not evidence:
        return False
    route_action = str(ocr_text.get("route_action") or "").strip().lower()
    render_policy = str(ocr_text.get("render_policy") or "").strip().lower()
    content_class = str(ocr_text.get("content_class") or "").strip().lower()
    if content_class != "sfx" or (route_action != "review_required" and render_policy != "review_required"):
        return False
    bbox = _bbox4(ocr_text.get("text_pixel_bbox") or ocr_text.get("bbox") or ocr_text.get("source_bbox"))
    if not bbox:
        return False
    width = max(0, bbox[2] - bbox[0])
    height = max(0, bbox[3] - bbox[1])
    try:
        stroke_confidence = float(evidence.get("stroke_confidence") or 0.0)
    except (TypeError, ValueError):
        stroke_confidence = 0.0
    return (
        width >= 250
        and height >= 70
        and stroke_confidence >= SOURCE_STYLE_CONFIDENCE_THRESHOLD
        and _hex_luma(evidence.get("text_color")) <= 45.0
        and _hex_luma(evidence.get("stroke_color")) >= 210.0
    )


def _review_text_fields_look_renderable(ocr_text: dict, translated: str | None = None) -> bool:
    return any(
        _looks_like_renderable_latin_phrase(str(value))
        for value in (
            ocr_text.get("text"),
            ocr_text.get("raw_ocr"),
            ocr_text.get("normalized_ocr"),
            ocr_text.get("normalized_text_final"),
            ocr_text.get("recognized_text"),
            ocr_text.get("original"),
            translated,
        )
        if value is not None
    )


def _style_evidence_allows_review_text_style_copy(
    ocr_text: dict,
    translated: str | None,
    evidence: dict | None,
) -> bool:
    if not evidence:
        return False
    route_action = str(ocr_text.get("route_action") or "").strip().lower()
    render_policy = str(ocr_text.get("render_policy") or "").strip().lower()
    content_class = str(ocr_text.get("content_class") or "").strip().lower()
    if content_class == "sfx":
        return False
    if route_action != "review_required" and render_policy != "review_required":
        return False
    if not _review_text_fields_look_renderable(ocr_text, translated):
        return False
    return _style_evidence_confidence(evidence) >= SOURCE_STYLE_CONFIDENCE_THRESHOLD


def _style_copy_allowed_for_text(ocr_text: dict, translated: str | None = None) -> bool:
    if any(
        _looks_like_scanlator_text(value)
        for value in (
            ocr_text.get("text"),
            ocr_text.get("raw_ocr"),
            ocr_text.get("normalized_ocr"),
            ocr_text.get("normalized_text_final"),
            ocr_text.get("recognized_text"),
            ocr_text.get("original"),
            translated,
        )
    ):
        return False
    content_class = str(ocr_text.get("content_class") or "").strip().lower()
    route_action = str(ocr_text.get("route_action") or "").strip().lower()
    render_policy = str(ocr_text.get("render_policy") or "").strip().lower()
    sfx = ocr_text.get("sfx") if isinstance(ocr_text.get("sfx"), dict) else {}

    if content_class != "sfx" and route_action != "translate_sfx_inpaint_render":
        return _primary_text_style_candidate_confident(ocr_text)
    if route_action == "review_required" or render_policy == "review_required":
        text_candidates = (
            ocr_text.get("text"),
            ocr_text.get("raw_ocr"),
            ocr_text.get("normalized_ocr"),
            ocr_text.get("normalized_text_final"),
            ocr_text.get("recognized_text"),
            ocr_text.get("original"),
        )
        if any(_looks_like_renderable_latin_phrase(str(value)) for value in text_candidates if value is not None):
            return _sfx_style_candidate_confident(ocr_text)
        return False
    if not _sfx_style_candidate_confident(ocr_text):
        return False

    text_fields = (
        ocr_text.get("text"),
        ocr_text.get("raw_ocr"),
        ocr_text.get("normalized_ocr"),
        ocr_text.get("normalized_text_final"),
        ocr_text.get("recognized_text"),
        ocr_text.get("original"),
        sfx.get("source_text"),
        sfx.get("adapted_text"),
        translated,
    )
    has_text_to_render = any(_non_empty_text(value) for value in text_fields)
    if has_text_to_render:
        return True

    if bool(sfx.get("visual_promotion")):
        return False
    if str(ocr_text.get("detector") or "").strip().lower() == "sfx_visual":
        return False
    return True


def _style_source_scan_allowed_for_text(ocr_text: dict, translated: str | None = None) -> bool:
    if _style_copy_allowed_for_text(ocr_text, translated):
        return True
    route_action = str(ocr_text.get("route_action") or "").strip().lower()
    render_policy = str(ocr_text.get("render_policy") or "").strip().lower()
    if route_action == "review_required" or render_policy == "review_required":
        content_class = str(ocr_text.get("content_class") or "").strip().lower()
        return content_class != "sfx" and _review_text_fields_look_renderable(ocr_text, translated)
    if str(ocr_text.get("detector") or "").strip().lower() == "sfx_visual":
        return False
    return False


def _neutralize_unallowed_source_style(layer: dict, *, force_black_text: bool = False) -> dict:
    try:
        style_confidence = float(layer.get("style_confidence") or 0.0)
    except (TypeError, ValueError):
        style_confidence = 0.0
    if str(layer.get("style_origin") or "").strip().lower() in {"auto_dark_panel_glow", "inferred_visual_card"}:
        return layer
    if (
        str(layer.get("style_origin") or "").strip().lower() == "source_detected"
        and style_confidence >= SOURCE_STYLE_CONFIDENCE_THRESHOLD
    ):
        return layer
    if _style_copy_allowed_for_text(layer, layer.get("translated")):
        return layer
    evidence = _style_evidence_to_dict(layer.get("style_evidence"))
    if _style_evidence_allows_visual_text_without_ocr(layer, evidence):
        return layer
    if _style_evidence_allows_review_text_style_copy(layer, layer.get("translated"), evidence):
        return layer
    style_input = _merge_style(None)
    style_input["tipo"] = layer.get("tipo", "fala")
    style_input["layout_profile"] = layer.get("layout_profile") or layer.get("block_profile")
    style = normalize_auto_typesetting_style(
        style_input,
        _coerce_background_rgb(layer.get("background_rgb")),
        force_black_text=force_black_text,
    )
    style["style_origin"] = "auto"
    style_confidence = _style_evidence_confidence(evidence)
    style["style_confidence"] = style_confidence
    style_source = str((evidence or {}).get("source") or "").strip()
    if style_source:
        style["style_source"] = style_source
    layer["style"] = style
    layer["estilo"] = style
    layer["style_origin"] = "auto"
    layer["style_confidence"] = style_confidence
    layer["style_source"] = style_source or None
    return layer


def _style_from_evidence(base_style: dict, evidence: dict | None) -> tuple[dict, str, float, str | None]:
    style = dict(base_style)
    confidence = _style_evidence_confidence(evidence)
    source = str((evidence or {}).get("source") or "").strip() or None
    origin = "source_detected" if confidence >= SOURCE_STYLE_CONFIDENCE_THRESHOLD else "auto"

    if evidence:
        if evidence.get("text_color"):
            style["cor"] = evidence["text_color"]
        if evidence.get("stroke_color"):
            style["contorno"] = evidence["stroke_color"]
        if evidence.get("stroke_width_px") is not None:
            style["contorno_px"] = evidence["stroke_width_px"]
        if evidence.get("font_name"):
            style["fonte"] = evidence["font_name"]
        if evidence.get("gradient") is True and evidence.get("gradient_colors"):
            colors = evidence.get("gradient_colors")
            if isinstance(colors, list) and len(colors) >= 2:
                style["cor_gradiente"] = [str(colors[0]), str(colors[1])]
                style["cor"] = str(colors[0])
        try:
            shadow_confidence = float(evidence.get("shadow_confidence") or 0.0)
        except (TypeError, ValueError):
            shadow_confidence = 0.0
        if evidence.get("shadow") is True and shadow_confidence >= SOURCE_STYLE_CONFIDENCE_THRESHOLD:
            style["sombra"] = True
            style["sombra_cor"] = evidence.get("shadow_color") or "#000000"
            style["sombra_offset"] = evidence.get("shadow_offset") if evidence.get("shadow_offset") is not None else [2, 2]
        try:
            glow_confidence = float(evidence.get("glow_confidence") or 0.0)
        except (TypeError, ValueError):
            glow_confidence = 0.0
        if evidence.get("glow") is True and glow_confidence >= SOURCE_STYLE_CONFIDENCE_THRESHOLD:
            style["glow"] = True
            style["glow_cor"] = evidence.get("glow_color") or evidence.get("text_color") or "#FFFFFF"
            style["glow_px"] = evidence.get("glow_px") if evidence.get("glow_px") is not None else 2
        try:
            curve_confidence = float(evidence.get("curve_confidence") or 0.0)
        except (TypeError, ValueError):
            curve_confidence = 0.0
        if evidence.get("curved") is True and curve_confidence >= SOURCE_STYLE_CONFIDENCE_THRESHOLD:
            style["curva"] = True
            style["curva_direcao"] = evidence.get("curve_direction") or "arc_up"
            try:
                style["curva_intensidade"] = float(evidence.get("curve_amount") or 0.0)
            except (TypeError, ValueError):
                style["curva_intensidade"] = 0.0
        style["style_origin"] = origin
        style["style_confidence"] = confidence
        if source:
            style["style_source"] = source

    return style, origin, confidence, source


def _coerce_background_rgb(value) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(max(0, min(255, int(round(float(channel))))) for channel in value[:3])  # type: ignore[return-value]
        except (TypeError, ValueError):
            pass
    return (255, 255, 255)


def _style_dark_panel_luminance(rgb: tuple[int, int, int]) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _style_rgb_chroma(rgb: tuple[int, int, int]) -> float:
    return float(max(rgb) - min(rgb))


def _qa_flags_set(layer: dict) -> set[str]:
    flags = layer.get("qa_flags") or []
    if not isinstance(flags, (list, tuple, set)):
        return set()
    return {str(flag).strip().lower() for flag in flags if str(flag).strip()}


def _is_dark_bubble_visual_layer(layer: dict) -> bool:
    mask_source = str(layer.get("bubble_mask_source") or "").strip().lower()
    if mask_source == "image_dark_bubble_mask":
        return True
    profile = str(layer.get("layout_profile") or layer.get("block_profile") or "").strip().lower()
    if profile == "dark_bubble":
        return True
    return "dark_bubble_oval_reocr" in _qa_flags_set(layer)


def _should_apply_dark_panel_glow_project_style(ocr_text: dict, background_rgb: tuple[int, int, int]) -> bool:
    profile = str(ocr_text.get("layout_profile") or ocr_text.get("block_profile") or "").strip().lower()
    dark_bubble_visual = _is_dark_bubble_visual_layer(ocr_text)
    if profile in {"white_balloon", "speech_balloon", "ui_form"} and not dark_bubble_visual:
        return False
    flags = _qa_flags_set(ocr_text)
    mask_source = str(ocr_text.get("bubble_mask_source") or "").strip().lower()
    reason = str(ocr_text.get("layout_safe_reason") or ocr_text.get("bubble_mask_rejection_reason") or "").strip().lower()
    rejected_mask = (
        mask_source in {"derived_white_crop_rejected", "rejected_derived_bubble_mask", "image_dark_panel_mask", "image_dark_bubble_mask"}
        or "debug_derived_bubble_mask_rejected" in flags
        or "rejected_derived_bubble_mask" in flags
        or "missing_real_bubble_mask" in flags
        or reason in {"debug_derived_bubble_mask_rejected", "derived_mask_not_anchored_to_text"}
        or dark_bubble_visual
    )
    if not rejected_mask:
        return False
    return _style_dark_panel_luminance(background_rgb) <= 90.0


def _style_rgb_to_hex(value) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        r, g, b = [max(0, min(255, int(round(float(channel))))) for channel in value[:3]]
    except (TypeError, ValueError):
        return None
    return f"#{r:02X}{g:02X}{b:02X}"


def _dark_panel_effect_color_evidence(ocr_text: dict) -> dict:
    direct = ocr_text.get("dark_panel_effect_colors")
    if isinstance(direct, dict):
        return dict(direct)
    metrics = ocr_text.get("qa_metrics") if isinstance(ocr_text.get("qa_metrics"), dict) else {}
    for key in ("image_dark_bubble_mask", "image_dark_panel_mask", "derived_card_panel_mask"):
        value = metrics.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _should_apply_visual_card_project_style(ocr_text: dict, background_rgb: tuple[int, int, int]) -> bool:
    profile = str(ocr_text.get("layout_profile") or ocr_text.get("block_profile") or "").strip().lower()
    if profile in {"white_balloon", "speech_balloon", "ui_form"}:
        return False
    if str(ocr_text.get("style_origin") or "").strip().lower() == "source_detected":
        return False

    flags = _qa_flags_set(ocr_text)
    mask_source = str(ocr_text.get("bubble_mask_source") or "").strip().lower()
    if mask_source not in {
        "image_white_bubble_mask",
        "image_dark_panel_mask",
        "image_dark_bubble_mask",
        "derived_card_panel_mask",
        "derived_white_crop_rejected",
        "rejected_derived_bubble_mask",
    }:
        return False

    luma = _style_dark_panel_luminance(background_rgb)
    chroma = _style_rgb_chroma(background_rgb)
    if 95.0 <= luma <= 235.0 and chroma >= 28.0:
        return True

    metrics = ocr_text.get("qa_metrics") if isinstance(ocr_text.get("qa_metrics"), dict) else {}
    try:
        render_luma = float(metrics.get("render_background_luma"))
        render_luma_std = float(metrics.get("render_background_luma_std", 999.0))
    except (TypeError, ValueError):
        return False
    return (
        "render_on_art_suspected" in flags
        and 95.0 <= render_luma <= 235.0
        and render_luma_std <= 28.0
    )


def _should_use_visual_card_font_project_style(style: dict, ocr_text: dict, background_rgb: tuple[int, int, int]) -> bool:
    candidate = dict(ocr_text)
    candidate["style_origin"] = "auto"
    if not _should_apply_visual_card_project_style(candidate, background_rgb):
        return False
    if str(ocr_text.get("style_origin") or "").strip().lower() != "source_detected":
        return False
    font_name = str(style.get("fonte") or "").strip()
    return font_name in {"ComicNeue-Bold.ttf", "ComicNeue-Regular.ttf"}


def _apply_dark_panel_glow_project_style(style: dict, ocr_text: dict, background_rgb: tuple[int, int, int]) -> dict:
    if not _should_apply_dark_panel_glow_project_style(ocr_text, background_rgb):
        return style
    styled = dict(style)
    colors = _dark_panel_effect_color_evidence(ocr_text)
    text_hex = _style_rgb_to_hex(colors.get("text_fill_rgb"))
    glow_hex = _style_rgb_to_hex(colors.get("text_glow_rgb") or colors.get("panel_glow_rgb"))
    border_hex = _style_rgb_to_hex(colors.get("border_rgb"))
    styled["cor"] = text_hex or "#FFFFFF"
    styled["contorno"] = border_hex or "#061D26"
    styled["contorno_px"] = max(1, int(styled.get("contorno_px", 0) or 0))
    styled["glow"] = True
    styled["glow_cor"] = glow_hex or "#67D8FF"
    styled["glow_px"] = max(3, int(styled.get("glow_px", 0) or 0))
    styled["style_origin"] = "auto_dark_panel_glow"
    if text_hex or glow_hex or border_hex:
        styled["style_source"] = "original_dark_panel_effect_colors"
    return styled


VISUAL_CARD_FONT = "LeagueGothic-Regular-VariableFont_wdth.ttf"


def _apply_visual_card_project_style(style: dict, ocr_text: dict, background_rgb: tuple[int, int, int]) -> dict:
    if not _should_apply_visual_card_project_style(ocr_text, background_rgb):
        return style
    styled = dict(style)
    styled["fonte"] = VISUAL_CARD_FONT
    styled["cor"] = "#EBFFFF"
    styled["contorno"] = ""
    styled["contorno_px"] = 0
    styled["glow"] = True
    styled["glow_cor"] = "#EBFFFF"
    styled["glow_px"] = max(2, int(styled.get("glow_px", 0) or 0))
    styled["style_origin"] = "inferred_visual_card"
    styled["style_confidence"] = max(0.75, float(styled.get("style_confidence", 0.0) or 0.0))
    styled["style_source"] = "visual_card_fallback"
    return styled


def _apply_visual_card_project_font(style: dict, ocr_text: dict, background_rgb: tuple[int, int, int]) -> dict:
    if not _should_use_visual_card_font_project_style(style, ocr_text, background_rgb):
        return style
    styled = dict(style)
    styled["fonte"] = VISUAL_CARD_FONT
    styled["force_upper"] = True
    return styled


def _apply_dark_panel_glow_project_styles(project_data: dict) -> int:
    applied = 0
    for layer in _iter_project_text_layers(project_data):
        background_rgb = _coerce_background_rgb(layer.get("background_rgb"))
        dark_panel = _should_apply_dark_panel_glow_project_style(layer, background_rgb)
        visual_card = _should_apply_visual_card_project_style(layer, background_rgb)
        style = _merge_style(layer.get("style") or layer.get("estilo"))
        visual_card_font = _should_use_visual_card_font_project_style(style, layer, background_rgb)
        if not dark_panel and not visual_card and not visual_card_font:
            continue
        styled = (
            _apply_dark_panel_glow_project_style(style, layer, background_rgb)
            if dark_panel
            else _apply_visual_card_project_style(style, layer, background_rgb)
        )
        if visual_card_font:
            styled = _apply_visual_card_project_font(styled, layer, background_rgb)
        layer["style"] = styled
        layer["estilo"] = styled
        layer["style_origin"] = styled.get("style_origin", layer.get("style_origin"))
        layer["style_confidence"] = styled.get("style_confidence", layer.get("style_confidence"))
        layer["style_source"] = styled.get("style_source", layer.get("style_source"))
        flags = list(layer.get("qa_flags") or [])
        if dark_panel or visual_card:
            flag = "auto_dark_panel_glow_fallback" if dark_panel else "visual_card_style_fallback"
            if flag not in flags:
                flags.append(flag)
        if styled.get("style_source") == "original_dark_panel_effect_colors" and "original_dark_panel_effect_colors" not in flags:
            flags.append("original_dark_panel_effect_colors")
        if visual_card_font and "visual_card_font_fallback" not in flags:
            flags.append("visual_card_font_fallback")
        layer["qa_flags"] = flags
        applied += 1
    return applied


_DARK_PANEL_STYLE_FIELDS = {
    "fonte",
    "cor",
    "contorno",
    "contorno_px",
    "glow",
    "glow_cor",
    "glow_px",
    "sombra",
    "sombra_cor",
    "sombra_offset",
    "bold",
    "italico",
    "force_upper",
}


def _style_hex_to_rgb(value) -> tuple[int, int, int] | None:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) == 7:
            try:
                return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
            except ValueError:
                return None
    return None


def _style_color_bucket(value, fallback: tuple[int, int, int] | None = None) -> str:
    rgb = _coerce_background_rgb(value) if isinstance(value, (list, tuple)) else _style_hex_to_rgb(value)
    if rgb is None:
        rgb = fallback
    if rgb is None:
        return "unknown"
    r, g, b = rgb
    luma = _style_dark_panel_luminance(rgb)
    chroma = _style_rgb_chroma(rgb)
    if luma >= 218 and chroma <= 38:
        return "near_white"
    if b >= max(r, g) + 18 or (g >= r + 18 and b >= r + 18):
        return "cyan_blue"
    if r >= g + 28 and r >= b + 28:
        return "warm"
    return f"luma_{int(luma // 48)}"


def _dark_panel_style_signature(layer: dict) -> tuple[str, str, str, str] | None:
    raw_background_rgb = layer.get("background_rgb")
    background_rgb = _coerce_background_rgb(raw_background_rgb)
    profile = str(layer.get("layout_profile") or layer.get("block_profile") or layer.get("content_class") or "").strip().lower()
    mask_source = str(layer.get("bubble_mask_source") or "").strip().lower()
    flags = _qa_flags_set(layer)
    if flags & {
        "false_light_bubble_dark_fill_blocked",
        "false_light_dark_bubble_promoted_to_white",
        "false_dark_white_style_neutralized",
    }:
        return None
    style = _merge_style(layer.get("style") or layer.get("estilo"))
    effect_colors = _dark_panel_effect_color_evidence(layer)

    card_like = mask_source in {"image_dark_panel_mask", "derived_card_panel_mask"} or "dark_panel" in profile or "visual_card" in profile or "card" in profile
    bubble_like = (
        mask_source == "image_dark_bubble_mask"
        or _is_dark_bubble_visual_layer(layer)
        or bool(flags & {"dark_bubble_ellipse_bbox_mask", "dark_bubble_oval_reocr", "dark_oval_safe_height_expanded"})
    )
    rejected_but_effect_sampled = (
        mask_source in {"derived_white_crop_rejected", "rejected_derived_bubble_mask", "image_white_bubble_mask"}
        and (
            bool(effect_colors)
            or str(style.get("style_source") or layer.get("style_source") or "").strip().lower() == "original_dark_panel_effect_colors"
        )
    )
    explicit_dark_panel = (
        card_like
        or bubble_like
        or rejected_but_effect_sampled
        or "dark_panel" in profile
        or "visual_card" in profile
        or "card" in profile
    )
    if not explicit_dark_panel:
        return None
    if (
        background_rgb is not None
        and _style_dark_panel_luminance(background_rgb) >= 205.0
        and mask_source not in {"image_dark_panel_mask", "image_dark_bubble_mask", "derived_card_panel_mask"}
        and not profile.startswith("dark_")
    ):
        return None
    if profile in {"white_balloon", "speech_balloon", "ui_form"} and not bubble_like:
        return None

    panel_rgb = effect_colors.get("panel_fill_rgb") or effect_colors.get("background_rgb") or background_rgb
    panel_rgb_tuple = _coerce_background_rgb(panel_rgb)
    if (
        mask_source == "image_dark_bubble_mask"
        and effect_colors.get("panel_fill_rgb") is None
        and effect_colors.get("background_rgb") is None
        and raw_background_rgb is None
    ):
        panel_rgb_tuple = (0, 0, 0)
    panel_luma = _style_dark_panel_luminance(panel_rgb_tuple)
    if panel_luma > 120.0 and not _should_apply_dark_panel_glow_project_style(layer, background_rgb):
        return None

    text_bucket = _style_color_bucket(effect_colors.get("text_fill_rgb") or style.get("cor"), (255, 255, 255))
    glow_bucket = _style_color_bucket(
        effect_colors.get("text_glow_rgb") or effect_colors.get("panel_glow_rgb") or style.get("glow_cor"),
        None,
    )
    panel_bucket = "dark" if panel_luma <= 80.0 else "mid_dark"
    strong_card = bool(
        card_like
        and (
            mask_source == "derived_card_panel_mask"
            or "dark_panel_full_bbox_selected" in flags
            or "dark_panel_rect_from_border_lines" in flags
            or ("dark_bubble_ellipse_bbox_mask" not in flags and "dark_bubble_oval_reocr" not in flags)
        )
    )
    shape_bucket = "card" if strong_card else "bubble" if bubble_like else "rejected_card"
    if panel_bucket == "dark" and text_bucket in {"near_white", "cyan_blue"}:
        # Dark system panels/bubbles in this title share a visual language even
        # when the per-crop glow sampler reads warm/cyan variants. Keep shape
        # separate, but do not split otherwise equivalent dark text by glow hue.
        glow_bucket = "dark_visual_glow"
    return shape_bucket, panel_bucket, text_bucket, glow_bucket


def _dark_panel_style_representative_score(layer: dict) -> tuple[float, float, float]:
    style = _merge_style(layer.get("style") or layer.get("estilo"))
    score = 0.0
    if str(style.get("style_origin") or layer.get("style_origin") or "").strip().lower() == "source_detected":
        score += 8.0
    if str(style.get("style_source") or layer.get("style_source") or "").strip().lower() == "original_dark_panel_effect_colors":
        score += 5.0
    try:
        score += min(1.0, max(0.0, float(style.get("style_confidence", layer.get("style_confidence", 0.0)) or 0.0))) * 4.0
    except (TypeError, ValueError):
        pass
    flags = _qa_flags_set(layer)
    if "mask_outside_balloon_critical" in flags or "real_inpaint_skipped_unsafe_mask" in flags:
        score -= 3.0
    if "debug_derived_bubble_mask_rejected" in flags:
        score -= 1.0
    bbox = _optional_bbox4(layer.get("text_pixel_bbox")) or _optional_bbox4(layer.get("source_bbox")) or _optional_bbox4(layer.get("bbox"))
    area = float(_bbox_area4(bbox)) if bbox else 0.0
    font_name = str(style.get("fonte") or "")
    condensed_bonus = 1.0 if font_name == VISUAL_CARD_FONT else 0.0
    return score + condensed_bonus, area, float(len(str(layer.get("text") or layer.get("translated") or "")))


def _dark_panel_group_prefers_condensed_status_font(layers: list[dict]) -> bool:
    if len(layers) < 2:
        return False
    luminous = 0
    visible_effect = 0
    dark_panel = 0
    for layer in layers:
        signature = _dark_panel_style_signature(layer)
        if signature is None:
            continue
        shape_bucket, panel_bucket, text_bucket, glow_bucket = signature
        if shape_bucket != "card":
            continue
        if panel_bucket == "dark":
            dark_panel += 1
        if text_bucket in {"near_white", "cyan_blue"}:
            luminous += 1
        if glow_bucket not in {"unknown"}:
            visible_effect += 1
    quorum = max(1, len(layers) // 2)
    return dark_panel >= quorum and luminous >= quorum and visible_effect >= quorum


def _dark_panel_layer_needs_readable_bubble_font(layer: dict) -> bool:
    signature = _dark_panel_style_signature(layer)
    if signature is None or signature[0] != "bubble":
        return False
    text = str(layer.get("translated") or layer.get("traduzido") or layer.get("text") or "").strip()
    words = [part for part in re.split(r"\s+", text) if part]
    compact_len = len(re.sub(r"\s+", "", text))
    return len(words) >= 4 or compact_len >= 28


def _cap_grouped_dark_panel_outline(style: dict, target: dict, representative: dict) -> tuple[dict, bool]:
    signature = _dark_panel_style_signature(target)
    if signature is None:
        return style, False
    try:
        outline_px = int(style.get("contorno_px", 0) or 0)
    except (TypeError, ValueError):
        outline_px = 0
    if outline_px <= 2:
        return style, False

    rep_style = _merge_style(representative.get("style") or representative.get("estilo"))
    try:
        rep_outline_px = int(rep_style.get("contorno_px", 0) or 0)
    except (TypeError, ValueError):
        rep_outline_px = 0
    if rep_outline_px <= 2:
        return style, False

    capped = dict(style)
    capped["contorno_px"] = 1
    if not capped.get("contorno"):
        capped["contorno"] = "#061D26"
    flags = [str(flag) for flag in target.get("qa_flags") or [] if str(flag)]
    if "dark_panel_group_outline_capped" not in flags:
        flags.append("dark_panel_group_outline_capped")
    target["qa_flags"] = flags
    metrics = target.get("qa_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    metrics["dark_panel_group_outline_capped"] = {
        "previous_px": outline_px,
        "representative_px": rep_outline_px,
        "capped_px": 1,
    }
    target["qa_metrics"] = metrics
    return capped, True


def _copy_dark_panel_group_style(representative: dict, target: dict, group_id: str, group_size: int, condensed_font: bool) -> bool:
    representative_style = _merge_style(representative.get("style") or representative.get("estilo"))
    target_style = _merge_style(target.get("style") or target.get("estilo"))
    updated = dict(target_style)
    changed = False
    for key in _DARK_PANEL_STYLE_FIELDS:
        if key not in representative_style:
            continue
        value = copy.deepcopy(representative_style[key])
        if (
            key == "fonte"
            and str(value).strip().upper() == "KOMIKAX_.TTF"
            and _dark_panel_layer_needs_readable_bubble_font(target)
        ):
            value = target_style.get("fonte") or "ComicNeue-Bold.ttf"
        if updated.get(key) != value:
            updated[key] = value
            changed = True

    if condensed_font and updated.get("fonte") != VISUAL_CARD_FONT:
        updated["fonte"] = VISUAL_CARD_FONT
        updated["force_upper"] = True
        updated["glow"] = True
        updated["glow_px"] = max(4, int(updated.get("glow_px", 0) or 0))
        if not updated.get("glow_cor"):
            updated["glow_cor"] = "#EBFFFF"
        changed = True

    updated, outline_capped = _cap_grouped_dark_panel_outline(updated, target, representative)
    changed = changed or outline_capped

    if not changed and target.get("style_group_id") == group_id:
        return False

    updated["style_origin"] = "grouped_dark_panel_visual_style"
    updated["style_source"] = "dark_panel_visual_style_group"
    try:
        confidence = float(updated.get("style_confidence", target.get("style_confidence", 0.0)) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    updated["style_confidence"] = max(0.78, confidence)
    target["style"] = updated
    target["estilo"] = updated
    target["style_origin"] = updated["style_origin"]
    target["style_source"] = updated["style_source"]
    target["style_confidence"] = updated["style_confidence"]
    target["style_group_id"] = group_id
    target["style_group_size"] = group_size
    target["style_group_representative"] = representative.get("id") or representative.get("uid")
    flags = [str(flag) for flag in target.get("qa_flags") or [] if str(flag)]
    for flag in ("dark_panel_style_grouped", "dark_panel_condensed_group_font" if condensed_font else ""):
        if flag and flag not in flags:
            flags.append(flag)
    target["qa_flags"] = flags
    return True


def _apply_dark_panel_style_groups(project_data: dict) -> dict:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for layer in _iter_project_text_layers(project_data):
        if not isinstance(layer, dict):
            continue
        signature = _dark_panel_style_signature(layer)
        if signature is None:
            continue
        groups.setdefault(signature, []).append(layer)

    grouped_layers = 0
    grouped_count = 0
    condensed_groups = 0
    for index, (signature, layers) in enumerate(sorted(groups.items(), key=lambda item: item[0])):
        if len(layers) < 2:
            continue
        representative = max(layers, key=_dark_panel_style_representative_score)
        condensed_font = _dark_panel_group_prefers_condensed_status_font(layers)
        group_id = f"dark_panel_visual_{index}_{signature[0]}_{signature[1]}_{signature[2]}_{signature[3]}"
        changed_in_group = 0
        for layer in layers:
            if _copy_dark_panel_group_style(representative, layer, group_id, len(layers), condensed_font):
                changed_in_group += 1
        if changed_in_group:
            grouped_count += 1
            grouped_layers += changed_in_group
            if condensed_font:
                condensed_groups += 1

    return {
        "groups": grouped_count,
        "layers": grouped_layers,
        "condensed_font_groups": condensed_groups,
    }


def _repair_dark_panel_visual_mask_safe_areas(project_data: dict) -> int:
    repaired = 0
    for layer in _iter_project_text_layers(project_data):
        if not isinstance(layer, dict):
            continue
        source = str(layer.get("bubble_mask_source") or layer.get("bubbleMaskSource") or "").strip().lower()
        if source not in {"image_dark_panel_mask", "image_dark_bubble_mask", "derived_card_panel_mask"}:
            continue
        flags = _qa_flags_set(layer)
        if (
            flags & {"dark_bubble_ellipse_bbox_mask", "dark_bubble_oval_reocr", "dark_oval_safe_height_expanded"}
            and "dark_panel_rect_from_border_lines" not in flags
            and "dark_panel_full_bbox_selected" not in flags
        ):
            continue
        metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
        panel_metrics = metrics.get("image_dark_panel_mask") if isinstance(metrics.get("image_dark_panel_mask"), dict) else {}
        panel_bbox = _optional_bbox4(panel_metrics.get("mask_bbox"))
        if panel_bbox is None:
            continue
        x1, y1, x2, y2 = panel_bbox
        panel_w = max(1, x2 - x1)
        panel_h = max(1, y2 - y1)
        if panel_w < 96 or panel_h < 42:
            continue
        anchor = (
            _optional_bbox4(layer.get("text_pixel_bbox"))
            or _optional_bbox4(layer.get("source_bbox"))
            or _optional_bbox4(layer.get("bbox"))
        )
        if anchor is None:
            continue
        ax1, ay1, ax2, ay2 = anchor
        anchor_cx = (ax1 + ax2) / 2.0
        anchor_cy = (ay1 + ay2) / 2.0
        anchor_inside = x1 <= anchor_cx <= x2 and y1 <= anchor_cy <= y2
        anchor_overlap = _bbox_intersection_area4(panel_bbox, anchor) / float(max(1, _bbox_area4(anchor)))
        if not anchor_inside and anchor_overlap < 0.55:
            continue
        pad_x = min(panel_w // 3, max(10, int(round(panel_w * 0.14))))
        pad_y = min(panel_h // 3, max(10, int(round(panel_h * 0.14))))
        safe = [x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y]
        if not _restored_safe_text_box_is_usable(safe):
            continue
        current_safe = _optional_bbox4(layer.get("safe_text_box")) or _optional_bbox4(layer.get("_debug_safe_text_box"))
        if current_safe == safe and _optional_bbox4(layer.get("target_bbox")) == panel_bbox:
            continue
        layer["target_bbox"] = panel_bbox
        layer["safe_text_box"] = safe
        layer["_debug_safe_text_box"] = safe
        layer["layout_safe_bbox"] = safe
        layer["layout_safe_reason"] = "visual_rect_dark_panel_mask"
        layer["_render_target_source"] = "dark_panel_visual_mask_bbox"
        flags = [str(flag) for flag in layer.get("qa_flags") or [] if str(flag)]
        flags = [
            flag
            for flag in flags
            if flag not in {"mask_outside_balloon", "mask_outside_balloon_critical", "fit_below_minimum_legible"}
        ]
        if "safe_text_box_recomputed" not in flags:
            flags.append("safe_text_box_recomputed")
        layer["qa_flags"] = flags
        repaired += 1
    return repaired


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


def _bubble_mask_u8_value(value) -> int | None:
    try:
        numeric = int(value)
    except Exception:
        return None
    if 1 <= numeric <= 255:
        return numeric
    return None


def _coerce_real_bubble_mask_to_canvas(mask_value, bbox, image_size: tuple[int, int] | None):
    if image_size is None:
        return None
    width, height = [int(v) for v in image_size]
    if width <= 0 or height <= 0:
        return None
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None

    x = y = None
    raw_mask = mask_value
    if isinstance(mask_value, dict):
        raw_mask = mask_value.get("pixels") or mask_value.get("data") or mask_value.get("mask")
        x = mask_value.get("x")
        y = mask_value.get("y")
        raw_width = mask_value.get("width")
        raw_height = mask_value.get("height")
    else:
        raw_width = raw_height = None

    try:
        mask = np.asarray(raw_mask, dtype=np.uint8)
    except Exception:
        return None
    if mask.size == 0:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim == 1 and raw_width and raw_height:
        try:
            mask = mask.reshape((int(raw_height), int(raw_width)))
        except Exception:
            return None
    if mask.ndim != 2 or not np.any(mask):
        return None
    if mask.shape[:2] == (height, width):
        return (mask > 0).astype(np.uint8)

    region_bbox = _optional_bbox4(bbox)
    if region_bbox is None and x is not None and y is not None:
        x1 = int(x)
        y1 = int(y)
        region_bbox = [x1, y1, x1 + int(mask.shape[1]), y1 + int(mask.shape[0])]
    if region_bbox is None:
        return None
    x1, y1, x2, y2 = region_bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    target_w = x2 - x1
    target_h = y2 - y1
    local = mask
    if local.shape[:2] != (target_h, target_w):
        try:
            local = np.array(
                Image.fromarray(local).resize((target_w, target_h), resample=Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        except Exception:
            return None
    canvas = np.zeros((height, width), dtype=np.uint8)
    canvas[y1:y2, x1:x2] = (local > 0).astype(np.uint8)
    return canvas if np.any(canvas) else None


def _infer_bubble_mask_image_size(ocr_page: dict | None, fallback_path=None) -> tuple[int, int] | None:
    if isinstance(ocr_page, dict):
        try:
            width = int(ocr_page.get("width") or 0)
            height = int(ocr_page.get("height") or 0)
        except Exception:
            width = height = 0
        if width > 0 and height > 0:
            return (width, height)
    if fallback_path:
        try:
            from PIL import Image

            with Image.open(fallback_path) as image:
                return image.size
        except Exception:
            pass
    if isinstance(ocr_page, dict):
        try:
            import numpy as np

            for region in ocr_page.get("_bubble_regions") or ocr_page.get("bubble_regions") or []:
                if not isinstance(region, dict):
                    continue
                for key in ("bubble_mask", "bubbleMask", "balloon_mask", "balloonMask", "segmentation_mask", "mask"):
                    if key not in region:
                        continue
                    mask = np.asarray(region.get(key))
                    if mask.ndim >= 2 and mask.shape[0] > 0 and mask.shape[1] > 0:
                        return (int(mask.shape[1]), int(mask.shape[0]))
        except Exception:
            pass
    return None


def _mask_value_under_bbox(mask, bbox) -> int | None:
    try:
        import numpy as np
    except Exception:
        return None
    box = _optional_bbox4(bbox)
    if box is None:
        return None
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    values, counts = np.unique(mask[y1:y2, x1:x2], return_counts=True)
    best_value = 0
    best_count = 0
    for value, count in zip(values.tolist(), counts.tolist()):
        value = int(value)
        if value <= 0:
            continue
        if int(count) > best_count:
            best_value = value
            best_count = int(count)
    return best_value if best_value > 0 else None


def _persist_real_bubble_mask_layer_for_page(
    page: dict,
    ocr_page: dict | None,
    work_dir: Path,
    *,
    page_number: int,
    image_size: tuple[int, int] | None = None,
    image_path=None,
) -> bool:
    """Persist the real per-bubble segmentation mask and attach renderer IDs."""
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return False

    resolved_size = image_size or _infer_bubble_mask_image_size(ocr_page, image_path)
    if resolved_size is None:
        return False
    width, height = [int(v) for v in resolved_size]
    if width <= 0 or height <= 0:
        return False

    regions = []
    if isinstance(ocr_page, dict):
        for region in ocr_page.get("_bubble_regions") or ocr_page.get("bubble_regions") or []:
            if isinstance(region, dict):
                regions.append(region)
    if isinstance(ocr_page, dict):
        for text in ocr_page.get("texts") or []:
            if isinstance(text, dict):
                regions.append(text)
    for layer in page.get("text_layers") or page.get("textos") or []:
        if isinstance(layer, dict):
            regions.append(layer)

    page_mask = np.zeros((height, width), dtype=np.uint8)
    value_by_bubble_id: dict[str, int] = {}
    next_value = 1
    non_real_bubble_mask_sources = {
        "bbox_fallback",
        "balloon_bbox_fallback",
        "derived_white_crop",
        "derived_rectangular_balloon",
        "image_white_region",
        "outline_seeded_contour",
        "image_white_bubble_mask",
        "image_rect_bubble_mask",
        "image_contour_bubble_mask",
        "derived_white_crop_rejected",
        "rejected_derived_bubble_mask",
    }
    for region in regions:
        if str(region.get("bubble_mask_source") or "").strip().lower() in non_real_bubble_mask_sources:
            continue
        bubble_id = str(region.get("bubble_id") or region.get("bubbleId") or region.get("id") or "").strip()
        for key in ("bubble_mask", "bubbleMask", "balloon_mask", "balloonMask", "segmentation_mask", "mask"):
            if key not in region:
                continue
            real_mask = _coerce_real_bubble_mask_to_canvas(
                region.get(key),
                region.get("bubble_mask_bbox") or region.get("bubbleMaskBbox") or region.get("balloon_bbox") or region.get("bbox"),
                (width, height),
            )
            if real_mask is None or not np.any(real_mask):
                continue
            configured_value = _bubble_mask_u8_value(region.get("bubble_mask_value") or region.get("bubbleMaskValue"))
            mask_value = configured_value or value_by_bubble_id.get(bubble_id)
            if mask_value is None:
                if next_value > 255:
                    break
                mask_value = next_value
                next_value += 1
            if bubble_id:
                value_by_bubble_id[bubble_id] = mask_value
            page_mask[real_mask > 0] = int(mask_value)
            region["bubble_mask_value"] = int(mask_value)
            break

    if not np.any(page_mask):
        return False

    rel_path = f"layers/bubble-mask/{int(page_number):03}.png"
    mask_path = Path(work_dir) / rel_path
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(page_mask, mode="L").save(mask_path)

    image_layers = page.setdefault("image_layers", {})
    image_layers["bubble_mask"] = {
        "key": "bubble_mask",
        "path": rel_path,
        "visible": False,
        "locked": True,
    }

    for collection_key in ("text_layers", "textos"):
        layers = page.get(collection_key)
        if not isinstance(layers, list):
            continue
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            bubble_id = str(layer.get("bubble_id") or layer.get("bubbleId") or "").strip()
            mask_value = value_by_bubble_id.get(bubble_id)
            if mask_value is None:
                mask_value = _mask_value_under_bbox(
                    page_mask,
                    layer.get("text_pixel_bbox") or layer.get("source_bbox") or layer.get("bbox") or layer.get("layout_bbox"),
                )
            if mask_value is None:
                continue
            layer["bubble_mask_path"] = str(mask_path)
            layer["bubble_mask_layer_path"] = rel_path
            layer["bubble_mask_value"] = int(mask_value)
    return True


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
    region, _language_options = _page_action_options_from_args(args)
    return region


def _page_action_options_from_args(args: list[str]) -> tuple[dict, dict]:
    region: dict = {"bbox": None, "mask_path": None}
    language_options: dict = {"idioma_origem": None, "idioma_destino": None, "engine_preset_id": None}
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
        if arg in {"--source-lang", "--idioma-origem"} and idx + 1 < len(args):
            language_options["idioma_origem"] = args[idx + 1]
            idx += 2
            continue
        if arg in {"--target-lang", "--idioma-destino"} and idx + 1 < len(args):
            language_options["idioma_destino"] = args[idx + 1]
            idx += 2
            continue
        if arg == "--engine-preset" and idx + 1 < len(args):
            language_options["engine_preset_id"] = args[idx + 1]
            idx += 2
            continue
        idx += 1
    if region["bbox"] is None:
        region["bbox"] = _mask_bounding_box(region.get("mask_path"))
    return region, language_options


def _apply_page_action_language_options(project: dict, language_options: dict | None) -> None:
    if not isinstance(language_options, dict):
        return
    idioma_origem = str(language_options.get("idioma_origem") or "").strip()
    idioma_destino = str(language_options.get("idioma_destino") or "").strip()
    engine_preset_id = str(language_options.get("engine_preset_id") or "").strip()
    if idioma_origem:
        project["idioma_origem"] = idioma_origem
    if idioma_destino:
        project["idioma_destino"] = idioma_destino
    if engine_preset_id:
        project["engine_preset_id"] = engine_preset_id


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


def _route_derived_skip_processing(layer: dict) -> bool:
    return False


_NOOP_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'.-]{1,23}$")
_COMMON_NOOP_WORDS = {
    "a",
    "am",
    "are",
    "be",
    "can",
    "do",
    "go",
    "he",
    "help",
    "hi",
    "i",
    "is",
    "it",
    "me",
    "no",
    "not",
    "ok",
    "she",
    "stop",
    "the",
    "they",
    "we",
    "what",
    "why",
    "yes",
    "you",
}


def _canonical_noop_token(value: object) -> str:
    text = str(value or "").replace("\u200b", "")
    text = re.sub(r"\s+", "", text)
    text = text.strip(" \t\r\n.!?…,:;\"'()[]{}")
    return text.lower()


def _layer_noop_without_glyph_evidence_should_preserve(layer: dict) -> bool:
    original = layer.get("original") or layer.get("text") or layer.get("raw_ocr") or ""
    translated = layer.get("translated") or layer.get("traduzido") or ""
    original_token = _canonical_noop_token(original)
    translated_token = _canonical_noop_token(translated)
    if not original_token or original_token != translated_token:
        return False
    if original_token in _COMMON_NOOP_WORDS:
        return False
    if not _NOOP_NAME_TOKEN_RE.match(original_token):
        return False
    mask_evidence = layer.get("mask_evidence") if isinstance(layer.get("mask_evidence"), dict) else {}
    try:
        raw_pixels = int(mask_evidence.get("raw_mask_pixels") or 0)
    except (TypeError, ValueError):
        raw_pixels = 0
    if raw_pixels > 0:
        return False
    return True


def _ensure_project_route_action_contract(project_data: dict) -> dict:
    """Backfill route_action while keeping removed decision fields neutral."""

    audit = {
        "checked_layers": 0,
        "filled_missing_count": 0,
        "overridden_special_count": 0,
        "preserved_noop_without_glyph_count": 0,
    }
    for page in project_data.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        for collection_name in ("text_layers", "textos", "texts"):
            for layer in page.get(collection_name) or []:
                if not isinstance(layer, dict):
                    continue
                audit["checked_layers"] += 1
                layer.update(neutralize_removed_decision_fields(layer))
                before = str(layer.get("route_action") or "").strip().lower()
                if before in ROUTE_ACTIONS:
                    layer["route_action"] = before
                    layer["skip_processing"] = False
                    layer["preserve_original"] = False
                    layer.pop("skip_reason", None)
                    continue
                layer["route_action"] = "translate_inpaint_render"
                layer.setdefault("route_reason", "translate_inpaint_render")
                layer["skip_processing"] = False
                layer["preserve_original"] = False
                layer.pop("skip_reason", None)
                audit["filled_missing_count"] += 1
    return audit


def build_text_layer(
    *,
    page_number: int,
    layer_index: int,
    ocr_text: dict,
    translated: str,
    corpus_visual_benchmark: dict,
    corpus_textual_benchmark: dict,
    source_image_rgb=None,
    font_detector=None,
) -> dict:
    from ocr.ocr_normalizer import normalize_ocr_record

    original_ocr_text = dict(ocr_text or {})
    ocr_text = normalize_ocr_record(ocr_text)
    if (
        str(original_ocr_text.get("route_action") or "").strip().lower() == "translate_sfx_inpaint_render"
        or str(original_ocr_text.get("content_class") or "").strip().lower() == "sfx"
    ):
        for key in ("content_class", "script", "translate_policy", "render_policy", "route_action", "route_reason", "sfx"):
            if original_ocr_text.get(key) is not None:
                ocr_text[key] = copy.deepcopy(original_ocr_text[key])
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
    ) or _bbox4(ocr_text.get("layout_bbox"), source_bbox)
    source_text_mask_bbox = _recover_source_text_mask_bbox(ocr_text, text_pixel_bbox)
    force_black_text = (
        str(ocr_text.get("layout_profile") or ocr_text.get("block_profile") or "").strip().lower() == "white_balloon"
        and not _is_dark_bubble_visual_layer(ocr_text)
    )
    style_copy_allowed = _style_copy_allowed_for_text(ocr_text, translated)
    style_evidence = _style_evidence_to_dict(ocr_text.get("style_evidence"))
    if style_evidence is None and _style_source_scan_allowed_for_text(ocr_text, translated):
        resolved_font_detector = font_detector if font_detector is not None else _get_source_style_font_detector()
        style_evidence = _extract_style_evidence_from_source_image(
            source_image_rgb,
            text_pixel_bbox or source_bbox,
            font_detector=resolved_font_detector,
            font_context=_style_font_context_for_text(ocr_text),
        )
    if not style_copy_allowed and _style_evidence_allows_visual_text_without_ocr(ocr_text, style_evidence):
        style_copy_allowed = True
    if not style_copy_allowed and _style_evidence_allows_review_text_style_copy(ocr_text, translated, style_evidence):
        style_copy_allowed = True
    style_evidence_for_layer = copy.deepcopy(style_evidence) if style_evidence is not None else None
    applied_style_evidence = style_evidence if style_copy_allowed else None
    style_input, style_origin, style_confidence, style_source = _style_from_evidence(
        _merge_style(ocr_text.get("estilo")),
        applied_style_evidence,
    )
    if not style_copy_allowed and style_evidence:
        style_origin = "auto"
        style_confidence = _style_evidence_confidence(style_evidence)
        style_source = str(style_evidence.get("source") or "").strip() or style_source
    style_input["tipo"] = ocr_text.get("tipo", "fala")
    style_input["layout_profile"] = ocr_text.get("layout_profile") or ocr_text.get("block_profile")
    background_rgb = _coerce_background_rgb(ocr_text.get("background_rgb"))
    style = normalize_auto_typesetting_style(
        style_input,
        background_rgb,
        force_black_text=force_black_text,
    )
    style_policy_text = dict(ocr_text)
    style_policy_text["style_origin"] = style_origin
    style = _apply_dark_panel_glow_project_style(style, style_policy_text, background_rgb)
    style = _apply_visual_card_project_style(style, style_policy_text, background_rgb)
    style = _apply_visual_card_project_font(style, style_policy_text, background_rgb)
    if not style_copy_allowed and style_evidence:
        style["style_origin"] = "auto"
        style["style_confidence"] = style_confidence
        if style_source:
            style["style_source"] = style_source
    style_origin = str(style.get("style_origin") or style_origin or "auto")
    style_confidence = float(style.get("style_confidence", style_confidence) or 0.0)
    style_source = style.get("style_source") or style_source
    balloon_subregions = _normalize_relative_y_bbox_list(
        ocr_text.get("balloon_subregions") or ocr_text.get("connected_lobe_bboxes") or [],
        source_bbox,
    )
    connected_lobe_bboxes = _normalize_relative_y_bbox_list(
        ocr_text.get("connected_lobe_bboxes") or balloon_subregions,
        source_bbox,
    )
    connected_text_groups = _normalize_relative_y_bbox_list(ocr_text.get("connected_text_groups") or [], source_bbox)
    connected_position_bboxes = _normalize_relative_y_bbox_list(
        ocr_text.get("connected_position_bboxes") or [],
        source_bbox,
    )
    connected_focus_bboxes = _normalize_relative_y_bbox_list(
        ocr_text.get("connected_focus_bboxes") or ocr_text.get("connected_position_bboxes") or [],
        source_bbox,
    )
    merged_source_bboxes = _normalize_relative_y_bbox_list(
        ocr_text.get("_merged_source_bboxes") or ocr_text.get("merged_source_bboxes") or [],
        source_bbox,
    )
    line_polygons = _normalize_relative_y_polygons(ocr_text.get("line_polygons"), source_bbox)
    connected_lobe_polygons = _normalize_relative_y_polygons(ocr_text.get("connected_lobe_polygons"), source_bbox)
    layout_group_size = int(ocr_text.get("layout_group_size") or 1)
    if balloon_subregions:
        layout_group_size = max(layout_group_size, len(balloon_subregions))
    ui_layout_evidence = copy.deepcopy(ocr_text.get("ui_layout_evidence")) if isinstance(ocr_text.get("ui_layout_evidence"), dict) else None
    block_profile_value = "ui_form" if ui_layout_evidence else ocr_text.get("block_profile")
    layout_profile_value = "ui_form" if ui_layout_evidence else (ocr_text.get("layout_profile") or ocr_text.get("block_profile"))

    layer = {
        "id": layer_id,
        "text_id": ocr_text.get("text_id") or layer_id,
        "trace_id": ocr_text.get("trace_id"),
        "source_trace_ids": list(ocr_text.get("source_trace_ids") or ocr_text.get("_source_trace_ids") or []),
        "source_text_ids": list(ocr_text.get("source_text_ids") or ocr_text.get("_source_text_ids") or []),
        "_merged_source_bboxes": merged_source_bboxes,
        "merged_source_bboxes": merged_source_bboxes,
        "merge_reason": ocr_text.get("merge_reason"),
        "ocr_merged_source_count": ocr_text.get("ocr_merged_source_count"),
        "text_instance_id": ocr_text.get("text_instance_id"),
        "page_id": ocr_text.get("page_id"),
        "band_id": ocr_text.get("band_id"),
        "coordinate_space": ocr_text.get("coordinate_space"),
        "source_coordinate_space": ocr_text.get("source_coordinate_space"),
        "strip_band_y_top": ocr_text.get("strip_band_y_top"),
        "_strip_band_y_top": ocr_text.get("_strip_band_y_top"),
        "band_y_top": ocr_text.get("band_y_top"),
        "_band_y_top": ocr_text.get("_band_y_top"),
        "_band_id": ocr_text.get("_band_id"),
        "kind": "text",
        "source_bbox": source_bbox,
        "layout_bbox": layout_bbox,
        "render_bbox": None,
        "bbox": layout_bbox,
        "text_pixel_bbox": text_pixel_bbox,
        "source_text_mask_bbox": source_text_mask_bbox,
        "_source_text_mask_bbox": source_text_mask_bbox,
        "_source_text_mask_bbox_source": "text_pixel_bbox_contract" if source_text_mask_bbox is not None else None,
        "tipo": ocr_text.get("tipo", "fala"),
        "original": ocr_text.get("text", ""),
        "raw_ocr": ocr_text.get("raw_ocr", ocr_text.get("text", "")),
        "normalized_ocr": ocr_text.get("normalized_ocr", ocr_text.get("text", "")),
        "normalized_text_final": ocr_text.get(
            "normalized_text_final",
            ocr_text.get("normalized_ocr", ocr_text.get("text", "")),
        ),
        "normalization": ocr_text.get("normalization", {"changed": False, "corrections": [], "is_gibberish": False}),
        "translated": translated,
        "text": ocr_text.get("text", ""),
        "ocr_confidence": float(ocr_text.get("confidence", 0.0) or 0.0),
        "ocr_source": ocr_text.get("ocr_source"),
        "background_rgb": list(ocr_text.get("background_rgb") or []) if isinstance(ocr_text.get("background_rgb"), (list, tuple)) else None,
        "ui_layout_evidence": ui_layout_evidence,
        "style": style,
        "estilo": style,
        "style_origin": style_origin,
        "style_confidence": style_confidence,
        "style_source": style_source,
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
        "rotation_source": ocr_text.get("rotation_source"),
        "detected_font_size_px": ocr_text.get("detected_font_size_px"),
        "skip_processing": _route_derived_skip_processing(ocr_text),
        "skip_reason": ocr_text.get("skip_reason"),
        "content_class": ocr_text.get("content_class"),
        "script": ocr_text.get("script"),
        "translate_policy": _sfx_policy_or_default(ocr_text, "translate_policy", "translate"),
        "render_policy": _sfx_policy_or_default(ocr_text, "render_policy", "normal"),
        "route_action": ocr_text.get("route_action"),
        "route_reason": ocr_text.get("route_reason"),
        "is_watermark": bool(ocr_text.get("is_watermark", False)),
        "is_non_english": bool(ocr_text.get("is_non_english", False)),
        "smart_skip_decision": ocr_text.get("smart_skip_decision"),
        "balloon_type": ocr_text.get("balloon_type"),
        "inpaint_mode": ocr_text.get("inpaint_mode"),
        "inpaint_strategy": ocr_text.get("inpaint_strategy"),
        "mask_evidence": copy.deepcopy(ocr_text.get("mask_evidence")) if isinstance(ocr_text.get("mask_evidence"), dict) else None,
        "balloon_bbox": _bbox4(ocr_text.get("balloon_bbox"), layout_bbox),
        "bubble_id": ocr_text.get("bubble_id") or ocr_text.get("bubbleId"),
        "bubble_mask_bbox": _bbox4(ocr_text.get("bubble_mask_bbox")),
        "bubble_inner_bbox": _bbox4(ocr_text.get("bubble_inner_bbox")),
        "bubble_mask_path": ocr_text.get("bubble_mask_path"),
        "bubble_mask_layer_path": ocr_text.get("bubble_mask_layer_path"),
        "bubble_mask_value": _bubble_mask_u8_value(ocr_text.get("bubble_mask_value")),
        "bubble_mask_source": ocr_text.get("bubble_mask_source"),
        "bubble_mask_shape": ocr_text.get("bubble_mask_shape"),
        "bubble_mask_ellipse": copy.deepcopy(ocr_text.get("bubble_mask_ellipse"))
        if isinstance(ocr_text.get("bubble_mask_ellipse"), dict)
        else None,
        "bubble_mask_error": ocr_text.get("bubble_mask_error"),
        "dark_panel_effect_colors": copy.deepcopy(ocr_text.get("dark_panel_effect_colors"))
        if isinstance(ocr_text.get("dark_panel_effect_colors"), dict)
        else None,
        "balloon_subregions": balloon_subregions,
        "layout_group_size": layout_group_size,
        "connected_children": ocr_text.get("connected_children"),
        "connected_text_groups": connected_text_groups,
        "connected_lobe_bboxes": connected_lobe_bboxes,
        "connected_lobe_polygons": connected_lobe_polygons,
        "connected_position_bboxes": connected_position_bboxes,
        "connected_focus_bboxes": connected_focus_bboxes,
        "connected_balloon_orientation": ocr_text.get("connected_balloon_orientation", ""),
        "connected_detection_confidence": float(ocr_text.get("connected_detection_confidence", 0.0) or 0.0),
        "connected_group_confidence": float(ocr_text.get("connected_group_confidence", 0.0) or 0.0),
        "connected_position_confidence": float(ocr_text.get("connected_position_confidence", 0.0) or 0.0),
        "subregion_confidence": float(ocr_text.get("subregion_confidence", 0.0) or 0.0),
        "connected_position_reasoner": ocr_text.get("connected_position_reasoner", ""),
        "connected_reasoner_model": ocr_text.get("connected_reasoner_model", ""),
        "connected_reasoner_notes": ocr_text.get("connected_reasoner_notes", ""),
        "_connected_slot_index": ocr_text.get("_connected_slot_index"),
        "_connected_slot_count": ocr_text.get("_connected_slot_count"),
        "_connected_vertical_bias_ratio": ocr_text.get("_connected_vertical_bias_ratio"),
        "_is_lobe_subregion": bool(ocr_text.get("_is_lobe_subregion", False)),
        "page_profile": ocr_text.get("page_profile"),
        "block_profile": block_profile_value,
        "layout_profile": layout_profile_value,
        "layout_safe_reason": ocr_text.get("layout_safe_reason"),
        "entity_flags": list(ocr_text.get("entity_flags") or []),
        "entity_repairs": list(ocr_text.get("entity_repairs") or []),
        "glossary_hits": list(ocr_text.get("glossary_hits") or []),
        "qa_flags": list(ocr_text.get("qa_flags") or []),
        "qa_metrics": copy.deepcopy(ocr_text.get("qa_metrics")) if isinstance(ocr_text.get("qa_metrics"), dict) else {},
        "sfx": copy.deepcopy(ocr_text.get("sfx")) if isinstance(ocr_text.get("sfx"), dict) else None,
        "corpus_visual_benchmark": corpus_visual_benchmark,
        "corpus_textual_benchmark": corpus_textual_benchmark,
    }
    if style_origin == "inferred_visual_card":
        flags = list(layer.get("qa_flags") or [])
        if "visual_card_style_fallback" not in flags:
            flags.append("visual_card_style_fallback")
        layer["qa_flags"] = flags
    if style.get("style_source") == "original_dark_panel_effect_colors":
        flags = list(layer.get("qa_flags") or [])
        if "auto_dark_panel_glow_fallback" not in flags:
            flags.append("auto_dark_panel_glow_fallback")
        if "original_dark_panel_effect_colors" not in flags:
            flags.append("original_dark_panel_effect_colors")
        layer["qa_flags"] = flags
    if style_evidence_for_layer is not None:
        layer["style_evidence"] = style_evidence_for_layer
    return _neutralize_unallowed_source_style(enrich_sfx_candidate(layer), force_black_text=force_black_text)


def _normalize_text_layer_for_renderer(raw_layer: dict, page_number: int, layer_index: int) -> dict:
    raw_layer = _remove_inline_sfx_noise_from_layer_texts(raw_layer)
    source_bbox = _bbox4(raw_layer.get("source_bbox"), raw_layer.get("bbox"))
    text_pixel_bbox = _bbox4(
        _normalize_relative_y_bbox(raw_layer.get("text_pixel_bbox"), source_bbox),
        source_bbox,
    )
    source_text_mask_bbox = _recover_source_text_mask_bbox(raw_layer, text_pixel_bbox)
    layout_bbox = _bbox4(raw_layer.get("layout_bbox"), text_pixel_bbox or source_bbox)
    style = _merge_style(raw_layer.get("style") or raw_layer.get("estilo"))
    layer_id = raw_layer.get("id") or f"tl_{page_number:03}_{layer_index + 1:03}"
    translated = raw_layer.get("translated", raw_layer.get("traduzido", ""))
    original = raw_layer.get("original", raw_layer.get("text", ""))
    raw_ocr = raw_layer.get("raw_ocr", original)
    normalized_ocr = raw_layer.get("normalized_ocr", raw_layer.get("normalized_text_final", raw_layer.get("text", original)))
    normalized_text_final = raw_layer.get("normalized_text_final", normalized_ocr)
    normalization = raw_layer.get("normalization")
    if isinstance(normalization, dict):
        normalization = copy.deepcopy(normalization)
    else:
        normalization = {"changed": False, "corrections": [], "is_gibberish": False}
    normalization_trace = raw_layer.get("normalization_trace")
    if isinstance(normalization_trace, dict):
        normalization_trace = copy.deepcopy(normalization_trace)
    elif isinstance(normalization, dict) and normalization.get("changed"):
        normalization_trace = copy.deepcopy(normalization)
    else:
        normalization_trace = None
    balloon_subregions = _normalize_relative_y_bbox_list(
        raw_layer.get("balloon_subregions") or raw_layer.get("connected_lobe_bboxes") or [],
        source_bbox,
    )
    connected_lobe_bboxes = _normalize_relative_y_bbox_list(
        raw_layer.get("connected_lobe_bboxes") or balloon_subregions,
        source_bbox,
    )
    connected_text_groups = _normalize_relative_y_bbox_list(raw_layer.get("connected_text_groups") or [], source_bbox)
    connected_position_bboxes = _normalize_relative_y_bbox_list(
        raw_layer.get("connected_position_bboxes") or [],
        source_bbox,
    )
    connected_focus_bboxes = _normalize_relative_y_bbox_list(
        raw_layer.get("connected_focus_bboxes") or raw_layer.get("connected_position_bboxes") or [],
        source_bbox,
    )
    merged_source_bboxes = _normalize_relative_y_bbox_list(
        raw_layer.get("_merged_source_bboxes") or raw_layer.get("merged_source_bboxes") or [],
        source_bbox,
    )
    line_polygons = _normalize_relative_y_polygons(raw_layer.get("line_polygons"), source_bbox)
    connected_lobe_polygons = _normalize_relative_y_polygons(raw_layer.get("connected_lobe_polygons"), source_bbox)
    layout_group_size = int(raw_layer.get("layout_group_size") or 1)
    if balloon_subregions:
        layout_group_size = max(layout_group_size, len(balloon_subregions))
    ui_layout_evidence = copy.deepcopy(raw_layer.get("ui_layout_evidence")) if isinstance(raw_layer.get("ui_layout_evidence"), dict) else None
    block_profile_value = "ui_form" if ui_layout_evidence else raw_layer.get("block_profile")
    layout_profile_value = "ui_form" if ui_layout_evidence else (raw_layer.get("layout_profile") or raw_layer.get("block_profile"))

    layer = {
        "id": layer_id,
        "text_id": raw_layer.get("text_id") or layer_id,
        "trace_id": raw_layer.get("trace_id"),
        "source_trace_ids": list(raw_layer.get("source_trace_ids") or raw_layer.get("_source_trace_ids") or []),
        "source_text_ids": list(raw_layer.get("source_text_ids") or raw_layer.get("_source_text_ids") or []),
        "_merged_source_bboxes": merged_source_bboxes,
        "merged_source_bboxes": merged_source_bboxes,
        "merge_reason": raw_layer.get("merge_reason"),
        "ocr_merged_source_count": raw_layer.get("ocr_merged_source_count"),
        "text_instance_id": raw_layer.get("text_instance_id"),
        "page_id": raw_layer.get("page_id"),
        "band_id": raw_layer.get("band_id"),
        "coordinate_space": raw_layer.get("coordinate_space"),
        "source_coordinate_space": raw_layer.get("source_coordinate_space"),
        "strip_band_y_top": raw_layer.get("strip_band_y_top"),
        "_strip_band_y_top": raw_layer.get("_strip_band_y_top"),
        "band_y_top": raw_layer.get("band_y_top"),
        "_band_y_top": raw_layer.get("_band_y_top"),
        "_band_id": raw_layer.get("_band_id"),
        "kind": "text",
        "source_bbox": source_bbox,
        "layout_bbox": layout_bbox,
        "target_bbox": raw_layer.get("target_bbox"),
        "safe_text_box": raw_layer.get("safe_text_box"),
        "_debug_safe_text_box": raw_layer.get("_debug_safe_text_box"),
        "_bubble_mask_bbox_unclamped": raw_layer.get("_bubble_mask_bbox_unclamped"),
        "_bubble_inner_bbox_unclamped": raw_layer.get("_bubble_inner_bbox_unclamped"),
        "_safe_text_box_unclamped": raw_layer.get("_safe_text_box_unclamped"),
        "render_bbox": raw_layer.get("render_bbox"),
        "_debug_render_bbox": raw_layer.get("_debug_render_bbox"),
        "bbox": layout_bbox,
        "text_pixel_bbox": text_pixel_bbox,
        "source_text_anchor_bbox": raw_layer.get("source_text_anchor_bbox"),
        "_source_text_anchor_bbox": raw_layer.get("_source_text_anchor_bbox"),
        "source_text_mask_bbox": source_text_mask_bbox,
        "_source_text_mask_bbox": source_text_mask_bbox,
        "_source_text_mask_bbox_source": raw_layer.get("_source_text_mask_bbox_source")
        or ("text_pixel_bbox_contract" if source_text_mask_bbox is not None else None),
        "tipo": raw_layer.get("tipo", "fala"),
        "original": original,
        "raw_ocr": raw_ocr,
        "normalized_ocr": normalized_ocr,
        "normalized_text_final": normalized_text_final,
        "normalization": normalization,
        "normalization_trace": normalization_trace,
        "_inline_sfx_removed": raw_layer.get("_inline_sfx_removed"),
        "translated": translated,
        "text": raw_layer.get("text", original),
        "ocr_confidence": float(raw_layer.get("ocr_confidence", raw_layer.get("confianca_ocr", 0.0)) or 0.0),
        "ocr_source": raw_layer.get("ocr_source"),
        "background_rgb": list(raw_layer.get("background_rgb") or []) if isinstance(raw_layer.get("background_rgb"), (list, tuple)) else None,
        "ui_layout_evidence": ui_layout_evidence,
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
        "skip_processing": _route_derived_skip_processing(raw_layer),
        "skip_reason": raw_layer.get("skip_reason"),
        "content_class": raw_layer.get("content_class"),
        "script": raw_layer.get("script"),
        "translate_policy": _sfx_policy_or_default(raw_layer, "translate_policy", "translate"),
        "render_policy": _sfx_policy_or_default(raw_layer, "render_policy", "normal"),
        "route_action": raw_layer.get("route_action"),
        "route_reason": raw_layer.get("route_reason"),
        "is_watermark": bool(raw_layer.get("is_watermark", False)),
        "is_non_english": bool(raw_layer.get("is_non_english", False)),
        "smart_skip_decision": raw_layer.get("smart_skip_decision"),
        "balloon_type": raw_layer.get("balloon_type"),
        "inpaint_mode": raw_layer.get("inpaint_mode"),
        "inpaint_strategy": raw_layer.get("inpaint_strategy"),
        "mask_evidence": copy.deepcopy(raw_layer.get("mask_evidence")) if isinstance(raw_layer.get("mask_evidence"), dict) else None,
        "balloon_bbox": _bbox4(raw_layer.get("balloon_bbox"), layout_bbox),
        "bubble_id": raw_layer.get("bubble_id") or raw_layer.get("bubbleId"),
        "bubble_mask_bbox": _bbox4(raw_layer.get("bubble_mask_bbox")),
        "bubble_inner_bbox": _bbox4(raw_layer.get("bubble_inner_bbox")),
        "bubble_mask_path": raw_layer.get("bubble_mask_path"),
        "bubble_mask_layer_path": raw_layer.get("bubble_mask_layer_path"),
        "bubble_mask_value": _bubble_mask_u8_value(raw_layer.get("bubble_mask_value")),
        "bubble_mask_source": raw_layer.get("bubble_mask_source"),
        "bubble_mask_shape": raw_layer.get("bubble_mask_shape"),
        "bubble_mask_ellipse": copy.deepcopy(raw_layer.get("bubble_mask_ellipse"))
        if isinstance(raw_layer.get("bubble_mask_ellipse"), dict)
        else None,
        "bubble_mask_error": raw_layer.get("bubble_mask_error"),
        "dark_panel_effect_colors": copy.deepcopy(raw_layer.get("dark_panel_effect_colors"))
        if isinstance(raw_layer.get("dark_panel_effect_colors"), dict)
        else None,
        "balloon_subregions": balloon_subregions,
        "layout_group_size": layout_group_size,
        "connected_children": raw_layer.get("connected_children"),
        "connected_text_groups": connected_text_groups,
        "connected_lobe_bboxes": connected_lobe_bboxes,
        "connected_lobe_polygons": connected_lobe_polygons,
        "connected_position_bboxes": connected_position_bboxes,
        "connected_focus_bboxes": connected_focus_bboxes,
        "connected_balloon_orientation": raw_layer.get("connected_balloon_orientation", ""),
        "connected_detection_confidence": float(raw_layer.get("connected_detection_confidence", 0.0) or 0.0),
        "connected_group_confidence": float(raw_layer.get("connected_group_confidence", 0.0) or 0.0),
        "connected_position_confidence": float(raw_layer.get("connected_position_confidence", 0.0) or 0.0),
        "subregion_confidence": float(raw_layer.get("subregion_confidence", 0.0) or 0.0),
        "connected_position_reasoner": raw_layer.get("connected_position_reasoner", ""),
        "connected_reasoner_model": raw_layer.get("connected_reasoner_model", ""),
        "connected_reasoner_notes": raw_layer.get("connected_reasoner_notes", ""),
        "_connected_slot_index": raw_layer.get("_connected_slot_index"),
        "_connected_slot_count": raw_layer.get("_connected_slot_count"),
        "_connected_vertical_bias_ratio": raw_layer.get("_connected_vertical_bias_ratio"),
        "_is_lobe_subregion": bool(raw_layer.get("_is_lobe_subregion", False)),
        "page_profile": raw_layer.get("page_profile"),
        "block_profile": block_profile_value,
        "layout_profile": layout_profile_value,
        "layout_safe_reason": raw_layer.get("layout_safe_reason"),
        "entity_flags": list(raw_layer.get("entity_flags") or []),
        "entity_repairs": list(raw_layer.get("entity_repairs") or []),
        "glossary_hits": list(raw_layer.get("glossary_hits") or []),
        "qa_flags": list(raw_layer.get("qa_flags") or []),
        "qa_metrics": copy.deepcopy(raw_layer.get("qa_metrics")) if isinstance(raw_layer.get("qa_metrics"), dict) else {},
        "sfx": copy.deepcopy(raw_layer.get("sfx")) if isinstance(raw_layer.get("sfx"), dict) else None,
        "corpus_visual_benchmark": raw_layer.get("corpus_visual_benchmark", {}),
        "corpus_textual_benchmark": raw_layer.get("corpus_textual_benchmark", {}),
        "_render_metadata_hydrated": bool(raw_layer.get("_render_metadata_hydrated", False)),
        "_restored_from_render_plan_candidate": bool(raw_layer.get("_restored_from_render_plan_candidate", False)),
        "_render_bbox_from_repaired_safe_text_box": bool(raw_layer.get("_render_bbox_from_repaired_safe_text_box", False)),
        "_final_render_anchor_from_repaired_safe_text_box": bool(
            raw_layer.get("_final_render_anchor_from_repaired_safe_text_box", False)
        ),
    }
    normalized = neutralize_removed_decision_fields(enrich_sfx_candidate(normalize_text_geometry(layer)))
    if isinstance(raw_layer.get("render_layout_contract"), dict):
        normalized["render_layout_contract"] = copy.deepcopy(raw_layer["render_layout_contract"])
    return normalized


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


def _raw_text_layers_from_collection(text_layers) -> list[dict]:
    if isinstance(text_layers, dict):
        text_layers = text_layers.get("texts")
    return [layer for layer in list(text_layers or []) if isinstance(layer, dict)]


def _mark_final_layer_as_page_space(layer: dict) -> dict:
    updated = dict(layer)
    preserve_final_band_render_contract = _has_valid_final_band_render_contract(updated)
    already_page_space = (
        str(updated.get("coordinate_space") or "").strip().lower() == "page"
        or str(updated.get("source_coordinate_space") or "").strip().lower() == "page"
    )
    band_y_top = 0
    band_y_keys = (
        ("band_y_top", "_band_y_top", "strip_band_y_top", "_strip_band_y_top")
        if already_page_space
        else ("strip_band_y_top", "_strip_band_y_top", "band_y_top", "_band_y_top")
    )
    for key in band_y_keys:
        try:
            band_y_top = int(updated.get(key) or 0)
        except Exception:
            band_y_top = 0
        if band_y_top:
            break

    def _infer_lost_band_y_top() -> int:
        target = _optional_bbox4(updated.get("target_bbox"))
        bubble_mask = _optional_bbox4(updated.get("bubble_mask_bbox"))
        if target is not None and bubble_mask is not None:
            same_horizontal_span = abs(int(target[0]) - int(bubble_mask[0])) <= 6 and abs(int(target[2]) - int(bubble_mask[2])) <= 6
            same_bottom = abs(int(target[3]) - int(bubble_mask[3])) <= 6
            if same_horizontal_span and same_bottom and int(target[1]) < int(bubble_mask[1]) <= int(target[3]):
                delta = int(bubble_mask[1]) - int(target[1])
                if 32 <= delta <= 50000:
                    return delta
            same_size_shift = (
                same_horizontal_span
                and abs((int(target[3]) - int(target[1])) - (int(bubble_mask[3]) - int(bubble_mask[1]))) <= 12
                and int(target[3]) < int(bubble_mask[1])
            )
            if same_size_shift:
                delta = int(bubble_mask[1]) - int(target[1])
                if 32 <= delta <= 50000:
                    return delta

        raw_text = _optional_bbox4(updated.get("_raw_text_evidence_bbox"))
        text_pixel = _optional_bbox4(updated.get("text_pixel_bbox"))
        if raw_text is not None and text_pixel is not None:
            same_horizontal_span = abs(int(raw_text[0]) - int(text_pixel[0])) <= 8 and abs(int(raw_text[2]) - int(text_pixel[2])) <= 8
            y1_delta = int(text_pixel[1]) - int(raw_text[1])
            y2_delta = int(text_pixel[3]) - int(raw_text[3])
            if same_horizontal_span and abs(y1_delta - y2_delta) <= 3 and 32 <= y2_delta <= 50000:
                return max(y1_delta, y2_delta)
        local_target = _optional_bbox4(updated.get("target_bbox")) or _optional_bbox4(updated.get("safe_text_box")) or _optional_bbox4(updated.get("render_bbox"))
        page_text_ref = (
            _optional_bbox4(updated.get("source_bbox"))
            or _optional_bbox4(updated.get("text_pixel_bbox"))
            or _optional_bbox4(updated.get("bbox"))
            or _optional_bbox4(updated.get("layout_bbox"))
        )
        if local_target is not None and page_text_ref is not None:
            same_horizontal_span = (
                abs(int(local_target[0]) - int(page_text_ref[0])) <= 24
                and abs(int(local_target[2]) - int(page_text_ref[2])) <= 32
            )
            page_ref_is_far_below = int(page_text_ref[1]) > int(local_target[3]) + 32
            if same_horizontal_span and page_ref_is_far_below:
                delta = int(page_text_ref[1]) - int(local_target[1])
                if 32 <= delta <= 50000:
                    return delta

        local_target = _optional_bbox4(updated.get("target_bbox")) or _optional_bbox4(updated.get("safe_text_box")) or _optional_bbox4(updated.get("render_bbox"))
        page_bubble = _optional_bbox4(updated.get("bubble_mask_bbox")) or _optional_bbox4(updated.get("bubble_inner_bbox"))
        if local_target is not None and page_bubble is not None and int(page_bubble[1]) > int(local_target[3]) + 32:
            overlap = _bbox_intersection_area4(
                [int(local_target[0]), int(page_bubble[1]), int(local_target[2]), int(page_bubble[3])],
                page_bubble,
            )
            local_width = max(1, int(local_target[2]) - int(local_target[0]))
            bubble_width = max(1, int(page_bubble[2]) - int(page_bubble[0]))
            horizontal_overlap = overlap / float(max(1, min(local_width, bubble_width) * max(1, int(page_bubble[3]) - int(page_bubble[1]))))
            if horizontal_overlap >= 0.35:
                delta = int(page_bubble[1]) - int(local_target[1])
                if 32 <= delta <= 50000:
                    return delta

        text_pixel = _optional_bbox4(updated.get("text_pixel_bbox")) or _optional_bbox4(updated.get("bbox"))
        if text_pixel is not None:
            polygon_bboxes: list[list[int]] = []
            for polygon in updated.get("line_polygons") or []:
                if not isinstance(polygon, (list, tuple)) or len(polygon) < 2:
                    continue
                xs: list[int] = []
                ys: list[int] = []
                for point in polygon:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        continue
                    try:
                        xs.append(int(round(float(point[0]))))
                        ys.append(int(round(float(point[1]))))
                    except Exception:
                        continue
                if xs and ys:
                    polygon_bboxes.append([min(xs), min(ys), max(xs), max(ys)])
            for poly_bbox in polygon_bboxes:
                same_horizontal_span = abs(int(poly_bbox[0]) - int(text_pixel[0])) <= 16 and abs(int(poly_bbox[2]) - int(text_pixel[2])) <= 16
                y1_delta = int(poly_bbox[1]) - int(text_pixel[1])
                y2_delta = int(poly_bbox[3]) - int(text_pixel[3])
                if same_horizontal_span and abs(y1_delta - y2_delta) <= 8 and 32 <= y2_delta <= 50000:
                    return max(y1_delta, y2_delta)
        return 0

    if band_y_top <= 0:
        band_y_top = _infer_lost_band_y_top()

    page_primary_bbox_keys = {"bbox", "source_bbox", "text_pixel_bbox", "_raw_text_evidence_bbox", "layout_bbox"}

    def _final_page_bbox(key: str, value):
        bbox = _optional_bbox4(value)
        if bbox is None:
            return bbox
        if key in {
            "source_text_anchor_bbox",
            "_source_text_anchor_bbox",
            "source_text_mask_bbox",
            "_source_text_mask_bbox",
        }:
            page_text_ref = (
                _optional_bbox4(updated.get("source_bbox"))
                or _optional_bbox4(updated.get("text_pixel_bbox"))
                or _optional_bbox4(updated.get("bbox"))
                or _optional_bbox4(updated.get("layout_bbox"))
            )
            if page_text_ref is not None and int(page_text_ref[1]) > int(bbox[3]) + 32:
                horizontal_overlap = max(0, min(int(page_text_ref[2]), int(bbox[2])) - max(int(page_text_ref[0]), int(bbox[0])))
                min_width = max(
                    1,
                    min(
                        max(1, int(page_text_ref[2]) - int(page_text_ref[0])),
                        max(1, int(bbox[2]) - int(bbox[0])),
                    ),
                )
                if horizontal_overlap / float(min_width) >= 0.35:
                    delta_y = int(page_text_ref[1]) - int(bbox[1])
                    if 32 <= delta_y <= 120000:
                        return [int(bbox[0]), int(bbox[1]) + delta_y, int(bbox[2]), int(bbox[3]) + delta_y]
        if band_y_top <= 0:
            return bbox
        if already_page_space and key in page_primary_bbox_keys:
            target = _optional_bbox4(updated.get("target_bbox"))
            bubble_mask = _optional_bbox4(updated.get("bubble_mask_bbox"))
            page_ref_y1 = None
            for ref in (target, bubble_mask):
                if ref is not None and int(ref[1]) > band_y_top:
                    page_ref_y1 = int(ref[1])
                    break
            if page_ref_y1 is None or int(bbox[3]) >= int(page_ref_y1) - 32:
                return bbox
        x1, y1, x2, y2 = bbox
        if y2 <= band_y_top:
            return [x1, y1 + band_y_top, x2, y2 + band_y_top]
        if y1 < band_y_top < y2 and y1 + band_y_top < y2:
            return [x1, y1 + band_y_top, x2, y2]
        return bbox

    for key in (
        "bbox",
        "source_bbox",
        "text_pixel_bbox",
        "source_text_anchor_bbox",
        "_source_text_anchor_bbox",
        "source_text_mask_bbox",
        "_source_text_mask_bbox",
        "_raw_text_evidence_bbox",
        "layout_bbox",
        "balloon_bbox",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "target_bbox",
        "position_bbox",
        "capacity_bbox",
        "layout_safe_bbox",
        "safe_text_box",
        "_debug_safe_text_box",
        "_bubble_mask_bbox_unclamped",
        "_bubble_inner_bbox_unclamped",
        "_safe_text_box_unclamped",
        "render_bbox",
        "_debug_render_bbox",
    ):
        fixed = _final_page_bbox(key, updated.get(key))
        if fixed is not None:
            updated[key] = fixed

    metric_text_anchor = _compact_text_anchor_bbox_from_layer_metrics(updated)
    target_for_metric_anchor = _optional_bbox4(
        updated.get("target_bbox")
        or updated.get("balloon_bbox")
        or updated.get("bubble_mask_bbox")
        or updated.get("safe_text_box")
    )
    metric_anchor_flags = {str(flag).strip().lower() for flag in updated.get("qa_flags") or [] if str(flag).strip()}
    metric_anchor_source = str(updated.get("bubble_mask_source") or updated.get("balloon_mask_source") or "").strip().lower()
    if (
        metric_text_anchor is not None
        and metric_anchor_source == "image_dark_bubble_mask"
        and (
            "dark_bubble_connected_lobe_passthrough" in metric_anchor_flags
            or "dark_connected_text_anchor_propagated_to_type" in metric_anchor_flags
            or "dark_connected_lobe_anchor_component_filtered" in metric_anchor_flags
            or updated.get("connected_lobe_bboxes")
            or updated.get("connected_position_bboxes")
            or str(updated.get("connected_balloon_orientation") or "").strip()
        )
    ):
        metric_area = max(1, _bbox_area4(metric_text_anchor))
        if target_for_metric_anchor is None or (
            _bbox_intersection_area4(metric_text_anchor, target_for_metric_anchor) / float(metric_area) >= 0.35
        ):
            updated["source_text_anchor_bbox"] = list(metric_text_anchor)
            updated["_source_text_anchor_bbox"] = list(metric_text_anchor)
            updated["_anchor_center_only_layout"] = True
            _merge_layer_qa_flags(updated, ["dark_connected_text_anchor_propagated_to_type"])

    for key in (
        "balloon_subregions",
        "connected_lobe_bboxes",
        "_merged_source_bboxes",
        "merged_source_bboxes",
    ):
        values = updated.get(key)
        if not isinstance(values, list):
            continue
        fixed_values = []
        changed = False
        for item in values:
            fixed = _final_page_bbox(key, item)
            if fixed is None:
                fixed_values.append(item)
                continue
            fixed_values.append(fixed)
            if isinstance(item, (list, tuple)):
                changed = changed or list(item) != fixed
            else:
                changed = True
        if changed:
            updated[key] = fixed_values

    def _layout_box_to_bubble_overlap(layout: list[int] | None, bubble: list[int] | None) -> float:
        if layout is None or bubble is None:
            return 1.0
        return _bbox_intersection_area4(layout, bubble) / float(max(1, _bbox_area4(layout)))

    def _horizontal_overlap_ratio(a: list[int] | None, b: list[int] | None) -> float:
        if a is None or b is None:
            return 0.0
        overlap = max(0, min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0])))
        return overlap / float(max(1, min(max(1, int(a[2]) - int(a[0])), max(1, int(b[2]) - int(b[0])))))

    def _plausible_page_bubble_for_reanchor(layout: list[int] | None, bubble: list[int] | None) -> bool:
        if layout is None or bubble is None:
            return False
        layout_w = max(1, int(layout[2]) - int(layout[0]))
        layout_h = max(1, int(layout[3]) - int(layout[1]))
        bubble_w = max(1, int(bubble[2]) - int(bubble[0]))
        bubble_h = max(1, int(bubble[3]) - int(bubble[1]))
        if bubble_w < max(72, int(layout_w * 0.40)):
            return False
        if bubble_h < max(48, int(layout_h * 0.40)):
            return False
        return _horizontal_overlap_ratio(layout, bubble) >= 0.35

    def _shift_bbox_y(value, delta_y: int) -> list[int] | None:
        bbox = _optional_bbox4(value)
        if bbox is None:
            return None
        return [int(bbox[0]), int(bbox[1]) + delta_y, int(bbox[2]), int(bbox[3]) + delta_y]

    if not _layer_is_translator_note(updated):
        page_bubble = _optional_bbox4(updated.get("bubble_mask_bbox"))
        layout_ref = (
            _optional_bbox4(updated.get("safe_text_box"))
            or _optional_bbox4(updated.get("_debug_safe_text_box"))
            or _optional_bbox4(updated.get("render_bbox"))
            or _optional_bbox4(updated.get("target_bbox"))
        )
        if (
            page_bubble is not None
            and layout_ref is not None
            and int(page_bubble[0]) > 2
            and int(page_bubble[1]) > 2
            and _plausible_page_bubble_for_reanchor(layout_ref, page_bubble)
            and _layout_box_to_bubble_overlap(layout_ref, page_bubble) < 0.25
        ):
            layout_center_y = (int(layout_ref[1]) + int(layout_ref[3])) // 2
            bubble_center_y = (int(page_bubble[1]) + int(page_bubble[3])) // 2
            delta_y = bubble_center_y - layout_center_y
            if 32 <= abs(delta_y) <= 50000:
                for key in (
                    "target_bbox",
                    "position_bbox",
                    "capacity_bbox",
                    "layout_safe_bbox",
                    "safe_text_box",
                    "_debug_safe_text_box",
                    "_safe_text_box_unclamped",
                    "render_bbox",
                ):
                    shifted = _shift_bbox_y(updated.get(key), delta_y)
                    if shifted is not None:
                        updated[key] = shifted
                updated["_final_page_space_reanchored_to_bubble_mask"] = True
                updated["_final_page_space_reanchor_delta_y"] = int(delta_y)

    line_polygons = updated.get("line_polygons")
    text_ref_bbox = _optional_bbox4(updated.get("text_pixel_bbox")) or _optional_bbox4(updated.get("bbox"))
    if isinstance(line_polygons, list) and len(line_polygons) > 1 and text_ref_bbox is not None:
        filtered_polygons = []
        for polygon in line_polygons:
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 2:
                continue
            xs: list[int] = []
            ys: list[int] = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    xs.append(int(round(float(point[0]))))
                    ys.append(int(round(float(point[1]))))
                except Exception:
                    continue
            if not xs or not ys:
                continue
            poly_bbox = [min(xs), min(ys), max(xs), max(ys)]
            if _bbox_intersection_area4(poly_bbox, text_ref_bbox) > 0 or _bbox_contains4_margin(text_ref_bbox, poly_bbox, margin=12):
                filtered_polygons.append(polygon)
        if filtered_polygons:
            updated["line_polygons"] = filtered_polygons

    safe_box = _optional_bbox4(updated.get("safe_text_box")) or _optional_bbox4(updated.get("_debug_safe_text_box"))
    safe_w = max(1, int(safe_box[2]) - int(safe_box[0])) if safe_box is not None else 1
    safe_h = max(1, int(safe_box[3]) - int(safe_box[1])) if safe_box is not None else 1

    def _plausible_edge_bubble_bbox(value) -> list[int] | None:
        bbox = _optional_bbox4(value)
        if bbox is None:
            return None
        width = max(1, int(bbox[2]) - int(bbox[0]))
        height = max(1, int(bbox[3]) - int(bbox[1]))
        if width < max(48, int(safe_w * 0.75)):
            return None
        if height < max(24, int(safe_h * 0.35)):
            return None
        return bbox

    edge_bubble = (
        _plausible_edge_bubble_bbox(updated.get("bubble_inner_bbox"))
        or _plausible_edge_bubble_bbox(updated.get("bubble_mask_bbox"))
        or _plausible_edge_bubble_bbox(updated.get("balloon_bbox"))
    )
    if edge_bubble is not None and safe_box is not None and not preserve_final_band_render_contract:
        bubble_area = max(1, _bbox_area4(edge_bubble))
        safe_area = max(1, _bbox_area4(safe_box))
        intersection = _bbox_intersection_area4(edge_bubble, safe_box)
        touches_page_edge = int(edge_bubble[1]) <= 2 or int(edge_bubble[0]) <= 2
        safe_outside_bubble = intersection / float(min(bubble_area, safe_area)) < 0.20
        if touches_page_edge and safe_outside_bubble:
            bw = max(1, int(edge_bubble[2]) - int(edge_bubble[0]))
            bh = max(1, int(edge_bubble[3]) - int(edge_bubble[1]))
            pad_x = min(24, max(4, bw // 12))
            pad_y = min(12, max(3, bh // 10))
            clamped = [
                int(edge_bubble[0]) + pad_x,
                int(edge_bubble[1]) + pad_y,
                int(edge_bubble[2]) - pad_x,
                int(edge_bubble[3]) - pad_y,
            ]
            if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
                clamped = list(edge_bubble)
            for key in (
                "target_bbox",
                "position_bbox",
                "capacity_bbox",
                "layout_safe_bbox",
                "safe_text_box",
                "_debug_safe_text_box",
                "render_bbox",
            ):
                updated[key] = list(clamped)
            updated["_final_edge_clipped_bubble_safe_box"] = True

    updated["coordinate_space"] = "page"
    updated["source_coordinate_space"] = "page"
    for key in ("band_y_top", "_band_y_top", "strip_band_y_top", "_strip_band_y_top"):
        updated.pop(key, None)
    if isinstance(updated.get("qa_flags"), list):
        stale_coordinate_flags = {
            "layout_bbox_coordinate_mismatch",
            "page_space_rerender_mixed_coordinates",
        }
        updated["qa_flags"] = [
            flag for flag in updated["qa_flags"] if str(flag) not in stale_coordinate_flags
        ]
    return updated


def _final_render_target_geometry_is_degenerate(layer: dict, qa_flags: set[str]) -> bool:
    target_bbox = _optional_bbox4(layer.get("target_bbox"))
    balloon_bbox = _optional_bbox4(layer.get("balloon_bbox"))
    if target_bbox is None or balloon_bbox is None:
        return False

    target_area = _bbox_area4(target_bbox)
    balloon_area = _bbox_area4(balloon_bbox)
    if target_area <= 0 or balloon_area <= 0:
        return False
    if target_area >= int(balloon_area * 0.45):
        return False

    target_overlap = _bbox_intersection_area4(target_bbox, balloon_bbox) / float(max(1, target_area))
    if target_overlap < 0.60:
        return False

    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    try:
        render_containment = float(qa_metrics.get("render_balloon_containment"))
    except (TypeError, ValueError):
        render_containment = 1.0

    has_degenerate_flag = bool(
        "tiny_bubble_inner_bbox_rejected" in qa_flags
        or "fit_below_minimum_legible" in qa_flags
        or "TEXT_CLIPPED" in qa_flags
        or "TEXT_OVERFLOW" in qa_flags
    )
    if not has_degenerate_flag and render_containment >= 0.30:
        return False

    has_text_geometry = bool(
        layer.get("line_polygons")
        or _optional_bbox4(layer.get("source_bbox"))
        or _optional_bbox4(layer.get("text_pixel_bbox"))
        or _optional_bbox4(layer.get("bbox"))
    )
    return has_text_geometry


def _has_valid_final_band_render_contract(layer: dict) -> bool:
    """Preserve band-resolved render geometry during final project rerender.

    The final page rerender runs after the strip/band path has already produced
    a visual result.  QA flags such as TEXT_CLIPPED can be useful review
    evidence, but they are too broad to discard a complete, internally
    consistent band render contract.
    """
    if not isinstance(layer, dict):
        return False
    if not str(layer.get("band_id") or "").strip():
        return False
    if not (
        str(layer.get("trace_id") or "").strip()
        or str(layer.get("id") or "").strip()
        or str(layer.get("text_id") or "").strip()
    ):
        return False
    qa_flags = {str(flag).strip() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
    if "missing_render_bbox" in qa_flags:
        return False
    render_bbox = _optional_bbox4(layer.get("render_bbox"))
    safe_bbox = _optional_bbox4(layer.get("safe_text_box") or layer.get("_debug_safe_text_box"))
    target_bbox = _optional_bbox4(layer.get("target_bbox") or layer.get("balloon_bbox"))
    if render_bbox is None or safe_bbox is None or target_bbox is None:
        return False
    render_area = _bbox_area4(render_bbox)
    safe_area = _bbox_area4(safe_bbox)
    target_area = _bbox_area4(target_bbox)
    if render_area <= 0 or safe_area <= 0 or target_area <= 0:
        return False
    if _bbox_intersection_area4(render_bbox, target_bbox) <= 0:
        return False
    visible_bubble_bbox = _optional_bbox4(layer.get("bubble_mask_bbox"))
    if visible_bubble_bbox is not None and _bbox_area4(visible_bubble_bbox) > 0:
        visible_render_overlap = _bbox_intersection_area4(render_bbox, visible_bubble_bbox) / float(max(1, render_area))
        visible_safe_overlap = _bbox_intersection_area4(safe_bbox, visible_bubble_bbox) / float(max(1, safe_area))
        if visible_render_overlap < 0.10 and visible_safe_overlap < 0.10:
            return False
    safe_overlap = _bbox_intersection_area4(render_bbox, safe_bbox) / float(max(1, render_area))
    target_overlap = _bbox_intersection_area4(render_bbox, target_bbox) / float(max(1, render_area))
    if safe_overlap < 0.35 or target_overlap < 0.55:
        return False
    source_text_mask = _optional_bbox4(layer.get("source_text_mask_bbox") or layer.get("_source_text_mask_bbox"))
    if source_text_mask is not None and _bbox_intersection_area4(source_text_mask, target_bbox) > 0:
        contract = layer.get("render_layout_contract") if isinstance(layer.get("render_layout_contract"), dict) else {}
        contract_block = _optional_bbox4(contract.get("block_bbox")) or render_bbox
        source_cx = (int(source_text_mask[0]) + int(source_text_mask[2])) / 2.0
        source_cy = (int(source_text_mask[1]) + int(source_text_mask[3])) / 2.0
        block_cx = (int(contract_block[0]) + int(contract_block[2])) / 2.0
        block_cy = (int(contract_block[1]) + int(contract_block[3])) / 2.0
        source_w = max(1, int(source_text_mask[2]) - int(source_text_mask[0]))
        source_h = max(1, int(source_text_mask[3]) - int(source_text_mask[1]))
        tolerance_x = max(18.0, min(36.0, source_w * 0.12))
        tolerance_y = max(14.0, min(32.0, source_h * 0.16))
        if abs(block_cx - source_cx) > tolerance_x or abs(block_cy - source_cy) > tolerance_y:
            return False
    # Do not preserve clearly absurd geometry; those still need the legacy
    # stale-geometry cleanup path.
    if render_area > int(max(safe_area, target_area) * 1.80):
        return False
    return True


def _drop_stale_final_render_geometry(layer: dict) -> dict:
    if _layer_is_translator_note(layer):
        return layer
    qa_flags = {str(flag) for flag in layer.get("qa_flags") or []}
    if (
        "dark_connected_component_safe_partition" in qa_flags
        and layer.get("_render_bbox_from_repaired_safe_text_box")
        and _optional_bbox4(layer.get("source_text_mask_bbox") or layer.get("_source_text_mask_bbox")) is not None
        and _optional_bbox4(layer.get("safe_text_box") or layer.get("_debug_safe_text_box")) is not None
        and _optional_bbox4(layer.get("render_bbox")) is not None
    ):
        for stale_key in (
            "render_layout_contract",
            "_render_layout_contract_hydrated_from_debug",
            "fit_status",
            "layout_fit_result",
        ):
            layer.pop(stale_key, None)
        layer["_final_band_render_contract_preserved"] = True
        return layer
    has_source_text_mask_anchor = _optional_bbox4(layer.get("source_text_mask_bbox") or layer.get("_source_text_mask_bbox")) is not None
    if (
        not has_source_text_mask_anchor
        and (
            layer.get("_render_bbox_from_repaired_safe_text_box")
            or layer.get("_real_bubble_body_safe_area_repaired")
            or str(layer.get("layout_safe_reason") or "") in {
                "real_bubble_body_bbox",
                "debug_derived_bubble_mask_bbox",
                "debug_derived_bubble_mask_unclamped",
            }
        )
    ):
        return layer
    merged_fragment_sources = _unique_preserve_order(
        [
            *(layer.get("source_trace_ids") or layer.get("_source_trace_ids") or []),
            *(layer.get("source_text_ids") or layer.get("_source_text_ids") or []),
        ]
    )
    if "same_balloon_fragment_merged" in qa_flags and (
        len(merged_fragment_sources) >= 2
        and
        _optional_bbox4(layer.get("render_bbox")) is not None
        and (
            _render_bbox_overlaps_layer_source_text(layer, layer.get("render_bbox"))
            or _optional_bbox4(layer.get("source_bbox")) is not None
        )
    ):
        return layer
    if _has_valid_final_band_render_contract(layer):
        layer["_final_band_render_contract_preserved"] = True
        return layer
    if (
        isinstance(layer.get("render_layout_contract"), dict)
        and _optional_bbox4(layer.get("source_text_mask_bbox") or layer.get("_source_text_mask_bbox")) is not None
    ):
        for stale_key in (
            "render_bbox",
            "_debug_render_bbox",
            "render_layout_contract",
            "_render_layout_contract_hydrated_from_debug",
            "fit_status",
            "layout_fit_result",
        ):
            layer.pop(stale_key, None)
        _merge_layer_qa_flags(layer, ["source_text_mask_render_contract_dropped"])
    stale_render_geometry = (
        str(layer.get("fit_status") or "").strip().lower()
        in {"below_minimum_legible", "failed", "overflow", "clipped"}
        or "fit_below_minimum_legible" in qa_flags
        or "TEXT_CLIPPED" in qa_flags
        or "TEXT_OVERFLOW" in qa_flags
        or "missing_render_bbox" in qa_flags
    )
    stale_target_geometry = _final_render_target_geometry_is_degenerate(layer, qa_flags)
    if stale_render_geometry or stale_target_geometry:
        for stale_key in (
            "target_bbox",
            "position_bbox",
            "capacity_bbox",
            "safe_text_box",
            "_debug_safe_text_box",
            "render_bbox",
            "_debug_render_bbox",
            "render_layout_contract",
            "_render_layout_contract_hydrated_from_debug",
            "fit_status",
            "layout_fit_result",
        ):
            layer.pop(stale_key, None)
        _merge_layer_qa_flags(layer, ["stale_final_render_contract_dropped"])
    return layer


def _bbox_height_for_page_space_filter(value) -> int:
    bbox = _bbox4(value)
    if bbox is None:
        return 0
    return max(0, int(bbox[3]) - int(bbox[1]))


def _bbox_min_y_for_page_space_filter(layer: dict) -> int | None:
    ys: list[int] = []
    for key in ("source_bbox", "text_pixel_bbox", "bbox", "layout_bbox"):
        bbox = _bbox4(layer.get(key))
        if bbox is not None:
            ys.append(int(bbox[1]))
    return min(ys) if ys else None


def _layer_has_cross_page_local_source_page_render_conflict(layer: dict) -> bool:
    band_id = str(layer.get("band_id") or "").strip()
    page_id = str(layer.get("page_id") or "").strip()
    band_page_id = _page_id_from_band_id(band_id)
    if not band_page_id or not page_id or band_page_id == page_id:
        return False

    source_boxes = [
        _optional_bbox4(layer.get(key))
        for key in ("source_bbox", "text_pixel_bbox", "bbox")
    ]
    source_boxes = [box for box in source_boxes if box is not None]
    render_boxes = [
        _optional_bbox4(layer.get(key))
        for key in ("target_bbox", "safe_text_box", "_debug_safe_text_box", "render_bbox", "_debug_render_bbox")
    ]
    render_boxes = [box for box in render_boxes if box is not None]
    if not source_boxes or not render_boxes:
        return False

    source_min_y = min(int(box[1]) for box in source_boxes)
    source_max_y = max(int(box[3]) for box in source_boxes)
    render_min_y = min(int(box[1]) for box in render_boxes)
    render_max_y = max(int(box[3]) for box in render_boxes)
    source_h = max(1, source_max_y - source_min_y)
    render_h = max(1, render_max_y - render_min_y)
    source_cy = (source_min_y + source_max_y) / 2.0
    render_cy = (render_min_y + render_max_y) / 2.0

    # A valid cross-page layer can exist after strip redivision, but its source
    # and render geometry must agree in page space.  If source/bbox stayed
    # band-local near the top while render/target is far down the output page,
    # the final page rerender will draw the local text at the wrong page spot.
    return bool(
        source_min_y < 900
        and render_min_y >= 900
        and abs(render_cy - source_cy) >= max(512.0, float(source_h + render_h) * 2.0)
    )


def _is_malformed_final_page_space_layer(layer: dict, *, page_has_band_space_peer: bool = False) -> bool:
    if not isinstance(layer, dict):
        return True
    if _layer_has_cross_page_local_source_page_render_conflict(layer):
        return True
    source_h = max(
        _bbox_height_for_page_space_filter(layer.get("source_bbox")),
        _bbox_height_for_page_space_filter(layer.get("text_pixel_bbox")),
        _bbox_height_for_page_space_filter(layer.get("bbox")),
        1,
    )
    render_h = max(
        _bbox_height_for_page_space_filter(layer.get("target_bbox")),
        _bbox_height_for_page_space_filter(layer.get("safe_text_box")),
        _bbox_height_for_page_space_filter(layer.get("_debug_safe_text_box")),
        _bbox_height_for_page_space_filter(layer.get("render_bbox")),
        _bbox_height_for_page_space_filter(layer.get("_debug_render_bbox")),
    )
    flags = {str(flag).strip().lower() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
    suspicious_flags = {
        "same_balloon_fragment_merged",
        "debug_derived_bubble_mask_rejected",
        "missing_render_bbox",
        "fit_below_minimum_legible",
        "connected_lobe_boxes_missing_source_anchor_fallback",
    }
    if render_h >= max(1200, source_h * 8) and flags & suspicious_flags:
        return True
    min_y = _bbox_min_y_for_page_space_filter(layer)
    if page_has_band_space_peer and min_y is not None and min_y < 900 and flags & suspicious_flags:
        return True
    return False


def _drop_malformed_final_page_space_layers(layers: list[dict]) -> list[dict]:
    band_has_page_space_peer: dict[str, bool] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        band_id = str(layer.get("band_id") or "").strip()
        if not band_id:
            continue
        min_y = _bbox_min_y_for_page_space_filter(layer)
        if min_y is not None and min_y >= 900:
            band_has_page_space_peer[band_id] = True
    kept: list[dict] = []
    for layer in layers:
        band_id = str(layer.get("band_id") or "").strip()
        if _is_malformed_final_page_space_layer(
            layer,
            page_has_band_space_peer=bool(band_id and band_has_page_space_peer.get(band_id)),
        ):
            continue
        kept.append(layer)
    return kept


def _apply_repaired_real_bubble_render_anchor(layer: dict) -> dict:
    if _has_valid_final_band_render_contract(layer):
        updated = dict(layer)
        updated["_final_band_render_contract_preserved"] = True
        return updated
    if not (
        layer.get("_render_bbox_from_repaired_safe_text_box")
        or layer.get("_real_bubble_body_safe_area_repaired")
        or str(layer.get("layout_safe_reason") or "") in {
            "real_bubble_body_bbox",
            "debug_derived_bubble_mask_bbox",
            "debug_derived_bubble_mask_unclamped",
        }
    ):
        return layer
    safe_bbox = _optional_bbox4(layer.get("safe_text_box")) or _optional_bbox4(layer.get("_debug_safe_text_box"))
    if safe_bbox is None:
        return layer
    updated = dict(layer)
    for key in (
        "target_bbox",
        "position_bbox",
        "capacity_bbox",
        "layout_bbox",
        "bbox",
        "text_pixel_bbox",
        "render_bbox",
        "safe_text_box",
        "_debug_safe_text_box",
    ):
        updated[key] = list(safe_bbox)
    updated["_final_render_anchor_from_repaired_safe_text_box"] = True
    return updated


def _final_page_space_text_layers_for_renderer(text_layers, *, page_number: int) -> list[dict]:
    normalized_layers: list[dict] = []
    for idx, layer in enumerate(_raw_text_layers_from_collection(text_layers)):
        normalized = _mark_final_layer_as_page_space(
            _normalize_text_layer_for_renderer(layer, int(page_number), idx)
        )
        normalized = _drop_stale_final_render_geometry(normalized)
        normalized_layers.append(_apply_repaired_real_bubble_render_anchor(normalized))
    return _drop_malformed_final_page_space_layers(normalized_layers)


def _replace_output_page_text_layers(output_page, text_layers: list[dict]) -> None:
    existing = getattr(output_page, "text_layers", None)
    if isinstance(existing, dict):
        existing["texts"] = text_layers
        output_page.text_layers = existing
    else:
        output_page.text_layers = {"texts": text_layers}


def _sync_page_legacy_aliases(page: dict) -> None:
    image_layers = page.setdefault("image_layers", {})
    page["arquivo_original"] = ((image_layers.get("base") or {}).get("path")) or page.get("arquivo_original")
    page["arquivo_traduzido"] = ((image_layers.get("rendered") or {}).get("path")) or page.get("arquivo_traduzido")

    text_layers = []
    for layer in page.get("text_layers") or []:
        if not isinstance(layer, dict):
            continue
        source_layer = enrich_sfx_candidate(layer)
        normalized_geometry = normalize_text_geometry(source_layer)
        for key in ("source_text_mask_bbox", "_source_text_mask_bbox", "_source_text_mask_bbox_source"):
            if source_layer.get(key) not in (None, [], ""):
                normalized_geometry[key] = copy.deepcopy(source_layer[key])
        if isinstance(source_layer.get("render_layout_contract"), dict):
            normalized_geometry["render_layout_contract"] = copy.deepcopy(source_layer["render_layout_contract"])
        if (
            str(source_layer.get("route_action") or "").strip().lower() == "translate_sfx_inpaint_render"
            or str(source_layer.get("content_class") or "").strip().lower() == "sfx"
            or str(source_layer.get("route_action") or "").strip().lower() == "review_required"
            and isinstance(source_layer.get("sfx"), dict)
        ):
            for key in ("content_class", "tipo", "script", "translate_policy", "render_policy", "route_action", "route_reason", "sfx"):
                if source_layer.get(key) is not None:
                    normalized_geometry[key] = copy.deepcopy(source_layer[key])
        normalized_layer = _mark_final_layer_as_page_space(
            neutralize_removed_decision_fields(normalized_geometry)
        )
        text_layers.append(_drop_stale_final_render_geometry(normalized_layer))
    text_layers = _drop_malformed_final_page_space_layers(text_layers)
    page["text_layers"] = text_layers
    page["textos"] = [
        {
            "id": layer.get("id"),
            "text_id": layer.get("text_id", layer.get("id")),
            "page_id": layer.get("page_id"),
            "band_id": layer.get("band_id"),
            "trace_id": layer.get("trace_id"),
            "source_trace_ids": list(layer.get("source_trace_ids") or layer.get("_source_trace_ids") or []),
            "source_text_ids": list(layer.get("source_text_ids") or layer.get("_source_text_ids") or []),
            "merge_reason": layer.get("merge_reason"),
            "ocr_merged_source_count": layer.get("ocr_merged_source_count"),
            "text_instance_id": layer.get("text_instance_id"),
            "bbox": _bbox4(
                layer.get("render_bbox"),
                _bbox4(
                    layer.get("layout_bbox"),
                    _bbox4(layer.get("source_bbox"), layer.get("bbox")),
                ),
            ),
            "tipo": layer.get("tipo", "fala"),
            "original": layer.get("original", ""),
            "raw_ocr": layer.get("raw_ocr", layer.get("original", "")),
            "normalized_ocr": layer.get("normalized_ocr", layer.get("normalized_text_final", layer.get("original", ""))),
            "normalized_text_final": layer.get("normalized_text_final", layer.get("normalized_ocr", layer.get("original", ""))),
            "normalization": layer.get("normalization", {"changed": False, "corrections": [], "is_gibberish": False}),
            "translated": layer.get("translated", layer.get("traduzido", "")),
            "traduzido": layer.get("translated", ""),
            "confianca_ocr": float(layer.get("ocr_confidence", 0.0) or 0.0),
            "ocr_confidence": float(layer.get("ocr_confidence", 0.0) or 0.0),
            "ocr_source": layer.get("ocr_source"),
            "background_rgb": list(layer.get("background_rgb") or []) if isinstance(layer.get("background_rgb"), (list, tuple)) else None,
            "ui_layout_evidence": copy.deepcopy(layer.get("ui_layout_evidence")) if isinstance(layer.get("ui_layout_evidence"), dict) else None,
            "source_bbox": _bbox4(layer.get("source_bbox"), layer.get("bbox")),
            "text_pixel_bbox": _bbox4(
                layer.get("text_pixel_bbox"),
                _bbox4(layer.get("source_bbox"), layer.get("bbox")),
            ),
            "source_text_anchor_bbox": _optional_bbox4(layer.get("source_text_anchor_bbox")),
            "_source_text_anchor_bbox": _optional_bbox4(layer.get("_source_text_anchor_bbox"))
            or _optional_bbox4(layer.get("source_text_anchor_bbox")),
            "source_text_mask_bbox": _bbox4(layer.get("source_text_mask_bbox")),
            "_source_text_mask_bbox": _bbox4(layer.get("_source_text_mask_bbox"), layer.get("source_text_mask_bbox")),
            "_source_text_mask_bbox_source": layer.get("_source_text_mask_bbox_source"),
            "line_polygons": _normalize_relative_y_polygons(
                layer.get("line_polygons"),
                _bbox4(layer.get("source_bbox"), layer.get("bbox")),
            ),
            "estilo": _merge_style(layer.get("style") or layer.get("estilo")),
            "style_origin": layer.get("style_origin", "legacy"),
            "qa_flags": list(layer.get("qa_flags") or []),
            "balloon_bbox": _bbox4(layer.get("balloon_bbox"), layer.get("layout_bbox") or layer.get("bbox")),
            "target_bbox": _bbox4(layer.get("target_bbox")),
            "position_bbox": _bbox4(layer.get("position_bbox")),
            "capacity_bbox": _bbox4(layer.get("capacity_bbox")),
            "safe_text_box": _bbox4(layer.get("safe_text_box")),
            "_debug_safe_text_box": _bbox4(layer.get("_debug_safe_text_box"), layer.get("safe_text_box")),
            "render_bbox": _bbox4(layer.get("render_bbox")),
            "_debug_render_bbox": _bbox4(layer.get("_debug_render_bbox"), layer.get("render_bbox")),
            "bubble_id": layer.get("bubble_id") or layer.get("bubbleId"),
            "bubble_mask_bbox": _bbox4(layer.get("bubble_mask_bbox")),
            "bubble_inner_bbox": _bbox4(layer.get("bubble_inner_bbox")),
            "bubble_mask_path": layer.get("bubble_mask_path"),
            "bubble_mask_layer_path": layer.get("bubble_mask_layer_path"),
            "bubble_mask_value": _bubble_mask_u8_value(layer.get("bubble_mask_value")),
            "bubble_mask_source": layer.get("bubble_mask_source"),
            "bubble_mask_error": layer.get("bubble_mask_error"),
            "balloon_type": layer.get("balloon_type"),
            "layout_profile": layer.get("layout_profile") or layer.get("block_profile"),
            "layout_safe_reason": layer.get("layout_safe_reason"),
            "layout_group_size": int(layer.get("layout_group_size") or 1),
            "skip_processing": _route_derived_skip_processing(layer),
            "skip_reason": layer.get("skip_reason"),
            "content_class": layer.get("content_class"),
            "script": layer.get("script"),
            "translate_policy": layer.get("translate_policy"),
            "render_policy": layer.get("render_policy"),
            "route_action": layer.get("route_action"),
            "route_reason": layer.get("route_reason"),
            "sfx": copy.deepcopy(layer.get("sfx")) if isinstance(layer.get("sfx"), dict) else None,
            "is_watermark": bool(layer.get("is_watermark", False)),
            "is_non_english": bool(layer.get("is_non_english", False)),
            "smart_skip_decision": layer.get("smart_skip_decision"),
            "balloon_subregions": list(layer.get("balloon_subregions") or []),
            "_merged_source_bboxes": list(layer.get("_merged_source_bboxes") or layer.get("merged_source_bboxes") or []),
            "merged_source_bboxes": list(layer.get("_merged_source_bboxes") or layer.get("merged_source_bboxes") or []),
            "connected_lobe_bboxes": list(layer.get("connected_lobe_bboxes") or []),
            "connected_lobe_polygons": list(layer.get("connected_lobe_polygons") or []),
            "connected_text_groups": list(layer.get("connected_text_groups") or []),
            "connected_position_bboxes": list(layer.get("connected_position_bboxes") or []),
            "connected_focus_bboxes": list(layer.get("connected_focus_bboxes") or []),
            "connected_balloon_orientation": layer.get("connected_balloon_orientation", ""),
            "connected_detection_confidence": float(layer.get("connected_detection_confidence", 0.0) or 0.0),
            "connected_group_confidence": float(layer.get("connected_group_confidence", 0.0) or 0.0),
            "connected_position_confidence": float(layer.get("connected_position_confidence", 0.0) or 0.0),
            "subregion_confidence": float(layer.get("subregion_confidence", 0.0) or 0.0),
        }
        for layer in text_layers
        if isinstance(layer, dict)
    ]


def _sync_project_legacy_aliases(project: dict) -> int:
    pages = project.get("paginas") if isinstance(project, dict) else None
    if not isinstance(pages, list):
        return 0
    synced = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        if isinstance(page.get("text_layers"), list):
            _sync_page_legacy_aliases(page)
            synced += 1
    return synced


def _normalize_dark_panel_rect_contract_layer(layer: dict) -> dict:
    if not isinstance(layer, dict):
        return layer
    source = str(layer.get("bubble_mask_source") or layer.get("bubbleMaskSource") or "").strip().lower()
    flags = [str(flag) for flag in layer.get("qa_flags") or [] if str(flag)]
    flag_set = {flag.strip().lower() for flag in flags}
    connected_dark_bubble_flags = {
        "dark_bubble_connected_lobe_passthrough",
        "dark_bubble_connected_lobes_promoted",
        "partial_dark_bubble_lobe_reocr",
    }
    metrics = layer.get("qa_metrics")
    has_dark_bubble_metric = bool(
        isinstance(metrics, dict)
        and isinstance(metrics.get("image_dark_bubble_mask"), dict)
        and (metrics.get("image_dark_bubble_mask") or {}).get("mask_bbox")
    )
    has_connected_lobes = bool(
        layer.get("connected_lobe_bboxes")
        or layer.get("connected_position_bboxes")
        or layer.get("connected_focus_bboxes")
        or len(layer.get("balloon_subregions") or []) >= 2
        or str(layer.get("connected_balloon_orientation") or "").strip()
    )
    if source == "image_dark_bubble_mask" and bool(flag_set & connected_dark_bubble_flags) and has_connected_lobes:
        return layer
    dark_oval_evidence = bool(
        flag_set
        & {
            "dark_bubble_ellipse_bbox_mask",
            "dark_bubble_oval_reocr",
            "dark_oval_safe_height_expanded",
        }
    )
    rect_line_evidence = bool(
        flag_set
        & {
            "dark_panel_rect_from_border_lines",
            "dark_panel_full_bbox_selected",
            "dark_panel_rect_from_uied",
        }
    )
    real_rect_evidence = rect_line_evidence or (bool(layer.get("card_panel_text_context")) and not dark_oval_evidence)
    if source == "image_dark_bubble_mask" and dark_oval_evidence and not real_rect_evidence:
        return layer
    should_rect = bool(
        source == "image_dark_bubble_mask"
        and (
            "dark_panel_rect_from_dark_bubble_bbox" in flag_set
            or ("connected_layout_disabled_dark_panel_visual_mask" in flag_set and real_rect_evidence)
            or bool(layer.get("card_panel_text_context"))
        )
    )
    if not should_rect:
        return layer
    normalized = dict(layer)
    normalized["bubble_mask_source"] = "image_dark_panel_mask"
    normalized["bubbleMaskSource"] = "image_dark_panel_mask"
    normalized["block_profile"] = "dark_panel"
    normalized["layout_profile"] = "dark_panel"
    normalized.pop("bubble_mask_shape", None)
    normalized.pop("bubble_mask_ellipse", None)
    normalized.pop("bubbleMaskEllipse", None)
    centered_bbox = _centered_capped_dark_panel_rect_layer_bbox(normalized)
    if centered_bbox is not None:
        normalized["bubble_mask_bbox"] = list(centered_bbox)
        normalized["bubbleMaskBbox"] = list(centered_bbox)
        normalized["balloon_bbox"] = list(centered_bbox)
    cleaned_flags = [flag for flag in flags if flag.strip().lower() != "dark_bubble_ellipse_bbox_mask"]
    if "dark_panel_rect_from_dark_bubble_bbox" not in {flag.strip().lower() for flag in cleaned_flags}:
        cleaned_flags.append("dark_panel_rect_from_dark_bubble_bbox")
    normalized["qa_flags"] = cleaned_flags
    metrics = normalized.get("qa_metrics")
    if isinstance(metrics, dict):
        metrics = dict(metrics)
        metrics.pop("image_dark_bubble_mask", None)
        panel_metric = dict(metrics.get("image_dark_panel_mask") or {})
        panel_metric.update(
            {
                "source": "image_dark_panel_mask",
                "detection_space": "final_project_dark_panel_rect_contract",
                "mask_bbox": list(normalized.get("bubble_mask_bbox") or normalized.get("balloon_bbox") or []),
                "anchor_bbox": list(normalized.get("text_pixel_bbox") or normalized.get("source_bbox") or normalized.get("bbox") or []),
                "centered_on_text": bool(centered_bbox is not None),
                "max_half_width_from_text_center": DARK_PANEL_RECT_MAX_HALF_WIDTH_FROM_TEXT_CENTER,
                "max_half_height_from_text_center": DARK_PANEL_RECT_MAX_HALF_HEIGHT_FROM_TEXT_CENTER,
            }
        )
        metrics["image_dark_panel_mask"] = panel_metric
        normalized["qa_metrics"] = metrics
    return normalized


def _centered_capped_dark_panel_rect_layer_bbox(layer: dict) -> list[int] | None:
    panel = layer.get("bubble_mask_bbox") or layer.get("balloon_bbox")
    anchor = layer.get("text_pixel_bbox") or layer.get("source_bbox") or layer.get("bbox")
    if not (
        isinstance(panel, (list, tuple))
        and len(panel) >= 4
        and isinstance(anchor, (list, tuple))
        and len(anchor) >= 4
    ):
        return None
    try:
        px1, py1, px2, py2 = [float(v) for v in panel[:4]]
        ax1, ay1, ax2, ay2 = [float(v) for v in anchor[:4]]
    except (TypeError, ValueError):
        return None
    if px2 <= px1 or py2 <= py1 or ax2 <= ax1 or ay2 <= ay1:
        return None
    cx = (ax1 + ax2) / 2.0
    cy = (ay1 + ay2) / 2.0
    text_half_w = max(1.0, (ax2 - ax1) / 2.0)
    text_half_h = max(1.0, (ay2 - ay1) / 2.0)
    current_half_w = max(abs(cx - px1), abs(px2 - cx))
    current_half_h = max(abs(cy - py1), abs(py2 - cy))
    max_half_w = max(float(DARK_PANEL_RECT_MAX_HALF_WIDTH_FROM_TEXT_CENTER), text_half_w + 8.0)
    max_half_h = max(float(DARK_PANEL_RECT_MAX_HALF_HEIGHT_FROM_TEXT_CENTER), text_half_h + 8.0)
    half_w = min(current_half_w, max_half_w)
    half_h = min(current_half_h, max_half_h)
    return [
        int(math.floor(cx - half_w)),
        int(math.floor(cy - half_h)),
        int(math.ceil(cx + half_w)),
        int(math.ceil(cy + half_h)),
    ]


def _normalize_final_project_page_space_layers(project_data: dict) -> dict:
    pages = project_data.get("paginas") if isinstance(project_data, dict) else None
    if not isinstance(pages, list):
        return {"pages_checked": 0, "layers_checked": 0, "layers_changed": 0}
    pages_checked = 0
    layers_checked = 0
    layers_changed = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        pages_checked += 1
        normalized_layers: list[dict] = []
        for layer in page.get("text_layers") or []:
            if not isinstance(layer, dict):
                continue
            layers_checked += 1
            before = json.dumps(layer, sort_keys=True, ensure_ascii=False, default=str)
            normalized = _mark_final_layer_as_page_space(
                neutralize_removed_decision_fields(normalize_text_geometry(layer))
            )
            normalized = _normalize_dark_panel_rect_contract_layer(normalized)
            after = json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
            if before != after:
                layers_changed += 1
            normalized_layers.append(normalized)
        if normalized_layers or isinstance(page.get("text_layers"), list):
            page["text_layers"] = normalized_layers
        _sync_page_legacy_aliases(page)
    return {
        "pages_checked": pages_checked,
        "layers_checked": layers_checked,
        "layers_changed": layers_changed,
    }


def _page_has_final_renderable_text(page: dict) -> bool:
    if not isinstance(page, dict):
        return False
    for layer in page.get("text_layers") or []:
        if not isinstance(layer, dict):
            continue
        route_action = str(layer.get("route_action") or "").strip()
        render_policy = str(layer.get("render_policy") or "").strip()
        if (
            (route_action == "merged_into_primary" or render_policy == "merged_into_primary")
            and not _has_renderable_translated_text(layer)
        ):
            continue
        if route_action == "merged_into_primary" or render_policy == "merged_into_primary":
            continue
        translated = str(layer.get("translated") or layer.get("traduzido") or "").strip()
        if translated:
            return True
    return False


def _project_has_late_render_geometry_repair(project_data: dict) -> bool:
    if not isinstance(project_data, dict):
        return False
    qa = project_data.get("qa") if isinstance(project_data.get("qa"), dict) else {}
    count_keys = (
        "post_style_component_safe_partition_count",
        "post_style_distinct_dark_lobe_geometry_repair_count",
        "final_distinct_dark_lobe_geometry_repair_count",
    )
    if any(int(qa.get(key) or 0) > 0 for key in count_keys):
        return True
    debug_repair = qa.get("debug_mask_bbox_repair") if isinstance(qa.get("debug_mask_bbox_repair"), dict) else {}
    if int(debug_repair.get("component_safe_partitions") or 0) > 0:
        return True
    for layer in _iter_project_text_layers(project_data):
        if not isinstance(layer, dict):
            continue
        flags = {str(flag) for flag in layer.get("qa_flags") or []}
        if (
            "dark_connected_component_safe_partition" in flags
            and layer.get("_render_bbox_from_repaired_safe_text_box")
            and _optional_bbox4(layer.get("render_bbox")) is not None
            and _optional_bbox4(layer.get("safe_text_box") or layer.get("_debug_safe_text_box")) is not None
        ):
            return True
    return False


def _layer_requires_strip_crop_rerender(layer: dict) -> bool:
    if not isinstance(layer, dict):
        return False
    flags = {str(flag) for flag in layer.get("qa_flags") or []}
    return "dark_connected_component_safe_partition" in flags


def _localize_layer_to_crop(layer: dict, crop_bbox: list[int], *, source_y_top: int | None = None) -> dict:
    local = copy.deepcopy(layer)
    dx = -int(crop_bbox[0])
    dy = -int(source_y_top if source_y_top is not None else crop_bbox[1])
    crop_dy = -int(crop_bbox[1])
    bbox_keys = (
        "bbox",
        "source_bbox",
        "text_pixel_bbox",
        "layout_bbox",
        "target_bbox",
        "safe_text_box",
        "_debug_safe_text_box",
        "position_bbox",
        "capacity_bbox",
        "layout_safe_bbox",
        "render_bbox",
        "_debug_render_bbox",
        "balloon_bbox",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "source_text_anchor_bbox",
        "_source_text_anchor_bbox",
        "source_text_mask_bbox",
        "_source_text_mask_bbox",
    )
    for key in bbox_keys:
        bbox = _offset_bbox4(_optional_bbox4(local.get(key)), dx, dy)
        if bbox is not None:
            local[key] = bbox
    for key in ("line_polygons", "connected_lobe_polygons"):
        if local.get(key):
            local[key] = _offset_nested_debug_bboxes(local[key], dx, dy)
    for key in ("balloon_subregions", "connected_lobe_bboxes", "connected_text_groups", "connected_position_bboxes", "connected_focus_bboxes"):
        if local.get(key):
            local[key] = _offset_nested_debug_bboxes(local[key], dx, crop_dy)
    if isinstance(local.get("render_layout_contract"), dict):
        local["render_layout_contract"] = _shift_render_layout_contract(
            local["render_layout_contract"],
            dx,
            dy,
            coordinate_space="band",
        )
    local["coordinate_space"] = "band"
    local["source_coordinate_space"] = "band"
    local["band_y_top"] = 0
    local["_band_y_top"] = 0
    local["strip_band_y_top"] = 0
    local["_strip_band_y_top"] = 0
    local.pop("_page_space_coordinate_offset", None)
    local.pop("qa_metrics", None)
    return local


def _positive_strip_band_base_for_rerender(row: dict, work_dir: Path, expected_size: tuple[int, int]):
    """Return the positive-color rendered band image for a strip crop when available."""

    try:
        import cv2
    except Exception:
        return None, None
    rendered_rel = str(row.get("rendered_band_path") or "").strip()
    candidates = []
    if rendered_rel:
        candidates.append(work_dir / "debug" / "e2e" / rendered_rel)
    expected_w, expected_h = expected_size
    for path in candidates:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        height, width = image_bgr.shape[:2]
        if width == expected_w and height == expected_h:
            return image_bgr, path
    return None, None


def _scrub_crop_rerender_text_regions(base_crop_rgb, local_layers: list[dict]) -> int:
    """Remove stale rendered text from a positive band base before final crop rerender."""

    if base_crop_rgb is None or not hasattr(base_crop_rgb, "shape"):
        return 0
    try:
        import cv2
        import numpy as _np
    except Exception:
        cv2 = None
        _np = None
    height, width = base_crop_rgb.shape[:2]
    scrubbed = 0
    for layer in local_layers:
        if not isinstance(layer, dict):
            continue
        flags = {str(flag).strip() for flag in layer.get("qa_flags") or [] if str(flag).strip()}
        if "dark_connected_component_safe_partition" not in flags:
            continue
        for key in ("safe_text_box", "render_bbox", "source_text_mask_bbox", "text_pixel_bbox", "bbox"):
            bbox = _optional_bbox4(layer.get(key))
            if bbox is None:
                continue
            pad = 20 if key in {"bbox", "text_pixel_bbox"} else (12 if key == "source_text_mask_bbox" else 6)
            x1 = max(0, min(width, int(bbox[0]) - pad))
            y1 = max(0, min(height, int(bbox[1]) - pad))
            x2 = max(0, min(width, int(bbox[2]) + pad))
            y2 = max(0, min(height, int(bbox[3]) + pad))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = base_crop_rgb[y1:y2, x1:x2, :]
            if cv2 is not None and _np is not None and roi.size:
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                ch_max = roi.max(axis=2)
                ch_min = roi.min(axis=2)
                near_white = (gray >= 90) & ((ch_max - ch_min) <= 86)
                pale_glow = (gray >= 70) & (roi[:, :, 0] >= 55) & (roi[:, :, 1] >= 55) & (roi[:, :, 2] >= 55)
                mask = (near_white | pale_glow).astype("uint8") * 255
                background_pixels = roi[(mask == 0) & (gray <= 78)]
                if background_pixels.size:
                    fill_rgb = _np.median(background_pixels, axis=0).astype("uint8")
                else:
                    fill_rgb = _np.array([0, 0, 0], dtype="uint8")
                roi[:, :, :] = fill_rgb
            else:
                base_crop_rgb[y1:y2, x1:x2, :] = 0
            scrubbed += 1
        if cv2 is None or _np is None:
            continue
        lobe_candidates: list[list[int]] = []
        for value in layer.get("connected_lobe_bboxes") or []:
            bbox = _optional_bbox4(value)
            if bbox is not None:
                lobe_candidates.append(bbox)
        for key in ("target_bbox", "safe_text_box", "source_text_mask_bbox", "text_pixel_bbox"):
            bbox = _optional_bbox4(layer.get(key))
            if bbox is not None:
                lobe_candidates.append(bbox)
        for bbox in lobe_candidates:
            x1 = max(0, min(width, int(bbox[0]) - 10))
            y1 = max(0, min(height, int(bbox[1]) - 10))
            x2 = max(0, min(width, int(bbox[2]) + 10))
            y2 = max(0, min(height, int(bbox[3]) + 10))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = base_crop_rgb[y1:y2, x1:x2, :]
            if not roi.size:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            ch_max = roi.max(axis=2)
            ch_min = roi.min(axis=2)
            neutral_bright = (gray >= 88) & ((ch_max - ch_min) <= 72)
            neutral_glow = (
                (gray >= 68)
                & (roi[:, :, 0] >= 48)
                & (roi[:, :, 1] >= 48)
                & (roi[:, :, 2] >= 48)
                & ((ch_max - ch_min) <= 58)
            )
            mask = (neutral_bright | neutral_glow).astype("uint8") * 255
            if int(mask.sum()) <= 0:
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=1)
            background_pixels = roi[(mask == 0) & (gray <= 70)]
            if background_pixels.size:
                fill_rgb = _np.median(background_pixels, axis=0).astype("uint8")
            else:
                fill_rgb = _np.array([0, 0, 0], dtype="uint8")
            roi[mask > 0] = fill_rgb
            scrubbed += 1
    return scrubbed


def _rerender_strip_reassembled_crops_from_metadata(project_data: dict, work_dir: Path) -> dict:
    audit = {
        "pages_checked": 0,
        "pages_rerendered": 0,
        "rows_checked": 0,
        "rows_rerendered": 0,
        "positive_band_base_used": 0,
        "stale_text_regions_scrubbed": 0,
        "errors": [],
        "strip_reassembled_output_rerender_allowed": True,
        "reason": "late_render_geometry_repair_requires_translated_crop_sync",
    }
    try:
        import cv2
        from typesetter import renderer as typesetter_mod
    except Exception as exc:
        audit["errors"].append({"error": str(exc), "stage": "import"})
        return audit

    crops_path = work_dir / "debug" / "e2e" / "10_copyback_reassemble" / "final_band_crops.jsonl"
    crop_rows = _load_debug_jsonl(crops_path)
    if not crop_rows:
        return audit
    project_layers = [layer for layer in _iter_project_text_layers(project_data) if isinstance(layer, dict)]
    translated_pages: dict[Path, np.ndarray] = {}
    touched_pages: set[Path] = set()
    seen_pages: set[Path] = set()
    for row in crop_rows:
        audit["rows_checked"] += 1
        bbox = _optional_bbox4(row.get("crop_bbox_in_translated_page"))
        translated_name = str(row.get("translated_output_page") or "").strip()
        if bbox is None or not translated_name:
            continue
        matching_layers = [
            layer for layer in _final_rerender_layers_for_crop(row, project_layers)
            if _layer_requires_strip_crop_rerender(layer)
        ]
        if not matching_layers:
            continue
        translated_path = _final_rerender_resolve_translated_path(work_dir, translated_name)
        seen_pages.add(translated_path)
        if translated_path not in translated_pages:
            current = cv2.imread(str(translated_path), cv2.IMREAD_COLOR)
            if current is None:
                audit["errors"].append({"translated_output_page": translated_name, "error": "translated_page_missing"})
                continue
            translated_pages[translated_path] = current
        page_bgr = translated_pages[translated_path]
        height, width = page_bgr.shape[:2]
        x1 = max(0, min(width, int(bbox[0])))
        y1 = max(0, min(height, int(bbox[1])))
        x2 = max(0, min(width, int(bbox[2])))
        y2 = max(0, min(height, int(bbox[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        crop_w = x2 - x1
        crop_h = y2 - y1
        positive_band_bgr, positive_band_path = _positive_strip_band_base_for_rerender(row, work_dir, (crop_w, crop_h))
        if positive_band_bgr is not None:
            base_crop_rgb = cv2.cvtColor(positive_band_bgr, cv2.COLOR_BGR2RGB)
            audit["positive_band_base_used"] += 1
        else:
            base_page_path = work_dir / "images" / Path(translated_name).name
            base_bgr = cv2.imread(str(base_page_path), cv2.IMREAD_COLOR)
            if base_bgr is None or base_bgr.shape[:2] != page_bgr.shape[:2]:
                base_bgr = page_bgr
            base_crop_rgb = cv2.cvtColor(base_bgr[y1:y2, x1:x2, :], cv2.COLOR_BGR2RGB)
        source_y_top = None
        try:
            source_y_top = int(row.get("band_y_top"))
        except Exception:
            source_y_top = None
        local_layers = [
            _localize_layer_to_crop(layer, [x1, y1, x2, y2], source_y_top=source_y_top)
            for layer in matching_layers
        ]
        audit["stale_text_regions_scrubbed"] += _scrub_crop_rerender_text_regions(base_crop_rgb, local_layers)
        try:
            rendered_rgb = typesetter_mod.render_band_image(
                base_crop_rgb,
                {"texts": local_layers, "_coordinate_space": "band"},
            )
        except Exception as exc:
            audit["errors"].append(
                {
                    "band_id": row.get("band_id"),
                    "translated_output_page": translated_name,
                    "positive_band_base": str(positive_band_path) if positive_band_path else None,
                    "error": str(exc),
                }
            )
            continue
        rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        page_bgr[y1:y2, x1:x2, :] = rendered_bgr
        final_rel = str(row.get("final_crop_path") or "").strip()
        if final_rel:
            final_path = work_dir / "debug" / "e2e" / final_rel
            final_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(final_path), rendered_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        audit["rows_rerendered"] += 1
        touched_pages.add(translated_path)
    for path in touched_pages:
        cv2.imwrite(str(path), translated_pages[path], [cv2.IMWRITE_JPEG_QUALITY, 95])
    audit["pages_checked"] = len(seen_pages)
    audit["pages_rerendered"] = len(touched_pages)
    return audit


def _rerender_final_project_images_from_metadata(project_data: dict, work_dir: Path) -> dict:
    """Make translated images match the final project-layer metadata.

    The strip path can normalize/hydrate render metadata after the first page
    image has already been written.  Rerendering here keeps the visual output in
    sync with the final project.json and export-gate evidence.
    """
    pages = project_data.get("paginas") if isinstance(project_data, dict) else None
    if not isinstance(pages, list):
        return {"pages_checked": 0, "pages_rerendered": 0, "errors": []}
    reassembled_crops_path = work_dir / "debug" / "e2e" / "10_copyback_reassemble" / "final_band_crops.jsonl"
    strip_debug_dir = work_dir / "_strip_debug"
    copyback_reassemble_dir = work_dir / "debug" / "e2e" / "10_copyback_reassemble"
    has_late_geometry_repair = _project_has_late_render_geometry_repair(project_data)
    if (
        strip_debug_dir.exists()
        or copyback_reassemble_dir.exists()
        or (reassembled_crops_path.exists() and reassembled_crops_path.stat().st_size > 0)
    ):
        if has_late_geometry_repair:
            return _rerender_strip_reassembled_crops_from_metadata(project_data, work_dir)
        return {
            "pages_checked": 0,
            "pages_rerendered": 0,
            "errors": [],
            "skipped_strip_reassembled_output": True,
            "reason": "strip_reassembled_output_owns_translated_pages",
        }

    previous_work_dir = project_data.get("_work_dir")
    had_work_dir = "_work_dir" in project_data
    project_data["_work_dir"] = str(work_dir)

    audit = {"pages_checked": 0, "pages_rerendered": 0, "errors": []}
    try:
        for page_idx, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            audit["pages_checked"] += 1
            if not _page_has_final_renderable_text(page):
                continue

            original_rel = (
                page.get("arquivo_original")
                or page.get("original_path")
                or page.get("translated_path")
                or page.get("rendered_path")
                or f"{page_idx + 1:03}.jpg"
            )
            img_name = Path(str(original_rel)).name
            rendered_rel = _resolve_image_layer_path(
                page,
                "rendered",
                page.get("arquivo_traduzido") or page.get("translated_path") or f"translated/{img_name}",
            )
            out_img = work_dir / rendered_rel
            try:
                render_page_image(project_data, page_idx, str(out_img))
                audit["pages_rerendered"] += 1
            except Exception as exc:
                audit["errors"].append(
                    {
                        "page_index": page_idx,
                        "page_number": page.get("numero", page_idx + 1),
                        "error": str(exc),
                    }
                )
    finally:
        if had_work_dir:
            project_data["_work_dir"] = previous_work_dir
        else:
            project_data.pop("_work_dir", None)

    return audit


def _rerender_final_project_images_after_contract(project_data: dict, work_dir: Path) -> dict:
    contract_audit = _ensure_project_render_contract(project_data)
    audit = {
        "post_rerender_contract_audit": contract_audit,
        "pages_checked": 0,
        "pages_rerendered": 0,
        "errors": [],
    }
    changed = any(
        int(contract_audit.get(key) or 0) > 0
        for key in (
            "filled_fit_metadata_count",
            "dropped_stale_fit_flag_count",
            "dropped_stale_render_background_flag_count",
            "normalized_fit_status_count",
        )
    )
    if not changed:
        return audit
    rerender_audit = _rerender_final_project_images_from_metadata(project_data, work_dir)
    audit.update(
        {
            "pages_checked": int(rerender_audit.get("pages_checked") or 0),
            "pages_rerendered": int(rerender_audit.get("pages_rerendered") or 0),
            "errors": list(rerender_audit.get("errors") or []),
        }
    )
    return audit


_RUNTIME_BUBBLE_MASK_ARRAY_KEYS = (
    "bubble_mask",
    "bubbleMask",
    "balloon_mask",
    "balloonMask",
    "segmentation_mask",
    "mask",
)


def _is_runtime_mask_array(value) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and value.__class__.__module__.startswith("numpy")
    )


def _drop_runtime_bubble_mask_arrays_from_page(page: dict) -> None:
    for collection_key in ("text_layers", "textos", "inpaint_blocks"):
        records = page.get(collection_key)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in _RUNTIME_BUBBLE_MASK_ARRAY_KEYS:
                if _is_runtime_mask_array(record.get(key)):
                    record.pop(key, None)


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
        "_merged_source_bboxes",
        "merged_source_bboxes",
        "bubble_id",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "bubble_mask_path",
        "bubble_mask_layer_path",
        "bubble_mask_value",
        "bubble_mask_source",
        "bubble_mask_shape",
        "bubble_mask_ellipse",
        "bubble_mask_error",
        "balloon_bbox",
        "balloon_type",
        "block_profile",
        "layout_profile",
        "layout_safe_reason",
        "ui_layout_evidence",
        "background_rgb",
        "background_type",
        "tipo",
        "font_size_px",
        "font_size",
        "rotation_deg",
        "rotation_source",
        "qa_flags",
        "allow_broad_bbox_text_search",
    ):
        value = block.get(key)
        if value is not None and value != [] and value != "":
            out[key] = copy.deepcopy(value)
    if "text_pixel_bbox" not in out:
        out["text_pixel_bbox"] = list(bbox)
    return out


def _bbox_overlap_ratio_for_project(a: list[int], b: list[int]) -> float:
    ix1 = max(int(a[0]), int(b[0]))
    iy1 = max(int(a[1]), int(b[1]))
    ix2 = min(int(a[2]), int(b[2]))
    iy2 = min(int(a[3]), int(b[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (int(a[2]) - int(a[0])) * (int(a[3]) - int(a[1])))
    area_b = max(1, (int(b[2]) - int(b[0])) * (int(b[3]) - int(b[1])))
    return inter / float(max(1, min(area_a, area_b)))


def _enrich_project_inpaint_block_from_text_layers(block: dict, text_layers: list[dict]) -> dict:
    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return block
    best_text = None
    best_score = 0.0
    for layer in text_layers:
        if not isinstance(layer, dict):
            continue
        layer_bbox = layer.get("bbox") or layer.get("balloon_bbox") or layer.get("text_pixel_bbox")
        if not isinstance(layer_bbox, (list, tuple)) or len(layer_bbox) != 4:
            continue
        score = _bbox_overlap_ratio_for_project([int(v) for v in bbox], [int(v) for v in layer_bbox])
        if score > best_score:
            best_score = score
            best_text = layer
    if best_text is None or best_score < 0.35:
        return block
    enriched = dict(block)
    for key in (
        "rotation_deg",
        "rotation_source",
        "qa_flags",
        "allow_broad_bbox_text_search",
        "balloon_type",
        "block_profile",
        "layout_profile",
        "layout_safe_reason",
        "ui_layout_evidence",
        "background_rgb",
        "line_polygons",
        "text_pixel_bbox",
        "bbox",
        "source_bbox",
        "balloon_bbox",
        "layout_bbox",
        "bubble_id",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "bubble_mask_path",
        "bubble_mask_layer_path",
        "bubble_mask_value",
        "_merged_source_bboxes",
        "merged_source_bboxes",
        "content_class",
        "skip_processing",
        "preserve_original",
        "translate_policy",
        "render_policy",
        "route_action",
        "route_reason",
        "is_watermark",
        "is_non_english",
    ):
        value = best_text.get(key)
        if value not in (None, [], ""):
            enriched[key] = copy.deepcopy(value)
    return enriched


def _refresh_project_qa_summary(project: dict) -> None:
    qa = project.get("qa")
    pages = project.get("paginas")
    if not isinstance(qa, dict) or not isinstance(qa.get("summary"), dict) or not isinstance(pages, list):
        return

    from qa.translation_qa import summarize_flags

    regions: list[dict] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        layers = page.get("text_layers")
        if isinstance(layers, list):
            regions.extend(layer for layer in layers if isinstance(layer, dict))
    previous_summary = qa.get("summary") if isinstance(qa.get("summary"), dict) else {}
    preserved_audits = {
        key: value
        for key, value in previous_summary.items()
        if str(key).startswith("final_") or str(key).endswith("_audit")
    }
    qa["summary"] = {**summarize_flags(regions), **preserved_audits}


def _save_project_json(project_json_path: Path, project: dict) -> None:
    from project_writer import write_project_json_atomic

    try:
        _sync_project_legacy_aliases(project)
    except Exception as exc:
        logger.warning("Falha ao sincronizar aliases legados antes de salvar project.json: %s", exc)

    try:
        _refresh_project_qa_summary(project)
    except Exception as exc:
        logger.warning("Falha ao atualizar qa.summary antes de salvar project.json: %s", exc)

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


def _load_editor_project_page(project_path: Path, page_idx: int) -> tuple[dict, Path, dict, Path, str, Path]:
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)
    work_dir = project_path.parent
    _attach_work_dir_log_handler(work_dir)
    project["_work_dir"] = str(work_dir)
    page = project["paginas"][page_idx]
    original_rel = _resolve_image_layer_path(
        page,
        "base",
        page.get("arquivo_original")
        or page.get("original_path")
        or page.get("translated_path")
        or page.get("rendered_path")
        or "",
    )
    img_name = Path(original_rel).name
    orig_img = work_dir / original_rel
    if not orig_img.exists():
        orig_img = work_dir / "originals" / img_name
    if not orig_img.exists():
        candidate = Path(original_rel)
        if candidate.exists():
            orig_img = candidate
    return project, work_dir, page, orig_img, img_name, Path(original_rel)


def _apply_recovery_layer_for_page(project: dict, page: dict, rendered_path: Path) -> None:
    work_dir = Path(project.get("_work_dir", "."))
    original_rel = _resolve_image_layer_path(
        page,
        "base",
        page.get("arquivo_original")
        or page.get("original_path")
        or page.get("translated_path")
        or page.get("rendered_path")
        or "",
    )
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
        try:
            from pipeline.recovery import apply_recovery_layer
        except ModuleNotFoundError:
            recovery_module_path = pipeline_root / "recovery.py"
            spec = importlib.util.spec_from_file_location("_traduzai_local_recovery", recovery_module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Nao foi possivel carregar recovery local: {recovery_module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            apply_recovery_layer = module.apply_recovery_layer
        apply_recovery_layer(rendered_path, original_path, recovery_path)
    except Exception as exc:
        logger.warning("Falha ao aplicar recovery na pagina renderizada: %s", exc)


def render_page_image(project, page_idx, output_path):
    """Auxiliar para renderizar a versao final da pagina para visualizacao."""
    from typesetter.renderer import _typeset_single_page
    page = project["paginas"][page_idx]
    
    # Prepara o dicionario de textos no formato que o renderer espera
    page_number = int(page.get("numero", page_idx + 1) or page_idx + 1)
    raw_text_layers = page.get("text_layers")
    if not isinstance(raw_text_layers, list) or not raw_text_layers:
        raw_text_layers = page.get("textos", [])
    trans_texts = _final_page_space_text_layers_for_renderer(raw_text_layers, page_number=page_number)
    trans_page_dict = {"texts": _visible_render_texts(trans_texts), "_coordinate_space": "page"}
    
    # Determina a imagem de fundo: tenta 'images' (inpainted) primeiro, depois 'originals'
    work_dir = Path(project.get("_work_dir", "."))
    img_name = Path(page["arquivo_original"]).name
    original_path = work_dir / "originals" / img_name
    bg_path = work_dir / "images" / img_name
    if not bg_path.exists():
        bg_path = original_path
        
    if not bg_path.exists():
        # Fallback para caminho direto se salvo no layout
        bg_path = Path(page.get("arquivo_original", ""))
        
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        render_base_path = _prepare_inpaint_base_for_render(
            original_path=original_path if original_path.exists() else Path(page.get("arquivo_original", "")),
            inpainted_path=bg_path,
            texts=trans_texts,
            temp_output_path=out_dir / f"{Path(output_path).stem}-base{Path(output_path).suffix}",
            update_inpaint=False,
        )
        _typeset_single_page((str(render_base_path), trans_page_dict, str(out_dir)))
        rendered_path = Path(output_path)
        renderer_output = out_dir / Path(render_base_path).name
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


def _has_renderable_translated_text(text: dict) -> bool:
    translated = str(text.get("translated") or text.get("traduzido") or "").strip()
    if not translated:
        return False
    original = str(text.get("original") or text.get("text") or "").strip()
    return not (original and translated == original)


def _visible_render_texts(texts: list[dict]) -> list[dict]:
    renderable: list[dict] = []
    for text in texts:
        if not isinstance(text, dict):
            continue
        route_action = str(text.get("route_action") or "").strip()
        render_policy = str(text.get("render_policy") or "").strip()
        merged_fragment = route_action == "merged_into_primary" or render_policy == "merged_into_primary"
        if merged_fragment:
            continue
        if text.get("visible", True) is not False:
            renderable.append(text)
            continue
        if text.get("_force_render_hidden") is True and _has_renderable_translated_text(text):
            renderable.append(text)
    return renderable


def _text_requires_final_cleanup(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    return _has_renderable_translated_text(text)


def _texts_requiring_final_cleanup(texts: list[dict]) -> list[dict]:
    return [text for text in _visible_render_texts(texts) if _text_requires_final_cleanup(text)]


def _final_render_text_box_cleanup_enabled() -> bool:
    raw = os.getenv("TRADUZAI_ENABLE_FINAL_RENDER_TEXT_BOX_CLEANUP")
    if raw is None or not raw.strip():
        return False
    return bool(raw and raw.strip().lower() in {"1", "true", "yes", "on"})


def _active_renderer_backend_label() -> str:
    try:
        from typesetter import rust_backend

        if rust_backend.rust_renderer_enabled():
            return "koharu_rust"
    except Exception:
        pass
    return "python"


def _prepare_inpaint_base_for_render(
    *,
    original_path: Path,
    inpainted_path: Path,
    texts: list[dict],
    temp_output_path: Path | None = None,
    update_inpaint: bool = False,
) -> Path:
    restored_path = _restore_preserved_art_fragment_regions_for_render(
        original_path=original_path,
        inpainted_path=inpainted_path,
        texts=texts,
        temp_output_path=temp_output_path,
        update_inpaint=update_inpaint,
    )
    if restored_path != inpainted_path:
        inpainted_path = restored_path

    cleaned_false_white_path = _clean_false_dark_white_regions_for_render(
        inpainted_path=inpainted_path,
        texts=texts,
        temp_output_path=temp_output_path,
        update_inpaint=update_inpaint,
    )
    if cleaned_false_white_path != inpainted_path:
        inpainted_path = cleaned_false_white_path

    cleaned_text_mask_path = _apply_final_text_mask_cleanup_for_render(
        inpainted_path=inpainted_path,
        texts=texts,
        temp_output_path=temp_output_path,
        update_inpaint=update_inpaint,
    )
    if cleaned_text_mask_path != inpainted_path:
        inpainted_path = cleaned_text_mask_path

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


def _preserved_art_fragment_restore_boxes(texts: list[dict]) -> list[list[int]]:
    boxes: list[list[int]] = []
    for text in texts or []:
        if not isinstance(text, dict) or not _is_art_fragment_review_layer(text):
            continue
        if text.get("visible", True) is not False and str(text.get("render_policy") or "").strip().lower() != "preserve_original":
            continue
        for key in ("bubble_mask_bbox", "balloon_bbox", "bubble_inner_bbox", "inpaint_bbox", "mask_bbox", "bbox"):
            bbox = text.get(key)
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            except Exception:
                continue
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                break
    return boxes


def _restore_preserved_art_fragment_regions_for_render(
    *,
    original_path: Path,
    inpainted_path: Path,
    texts: list[dict],
    temp_output_path: Path | None = None,
    update_inpaint: bool = False,
) -> Path:
    boxes = _preserved_art_fragment_restore_boxes(texts)
    if not boxes or not original_path.exists() or not inpainted_path.exists():
        return inpainted_path

    try:
        from PIL import Image

        with Image.open(original_path) as original_src:
            original_img = original_src.convert("RGB")
        with Image.open(inpainted_path) as inpaint_src:
            base_img = inpaint_src.convert("RGB")
        if original_img.size != base_img.size:
            return inpainted_path

        changed = False
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, min(base_img.width, x1))
            x2 = max(0, min(base_img.width, x2))
            y1 = max(0, min(base_img.height, y1))
            y2 = max(0, min(base_img.height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            base_img.paste(original_img.crop((x1, y1, x2, y2)), (x1, y1))
            changed = True
        if not changed:
            return inpainted_path

        target_path = inpainted_path if update_inpaint else temp_output_path
        if target_path is None:
            target_path = inpainted_path.with_name(f"{inpainted_path.stem}-preserve-art{inpainted_path.suffix}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        base_img.save(target_path)
        return target_path
    except Exception as exc:
        logger.warning("Falha ao restaurar fragmento de arte preservado antes do render: %s", exc)
        return inpainted_path


def _is_false_dark_white_layer(layer: dict) -> bool:
    flags = {
        str(flag).strip().lower()
        for flag in (layer.get("qa_flags") or [])
        if str(flag).strip()
    }
    if not flags & {
        "false_light_bubble_dark_fill_blocked",
        "false_light_dark_bubble_promoted_to_white",
        "false_dark_white_style_neutralized",
    }:
        return False
    profile = str(layer.get("layout_profile") or layer.get("block_profile") or "").strip().lower()
    return profile in {"white_balloon", "speech_balloon", "standard", ""}


def _false_dark_white_cleanup_boxes(texts: list[dict]) -> list[list[int]]:
    boxes: list[list[int]] = []
    for text in texts or []:
        if not isinstance(text, dict) or not _is_false_dark_white_layer(text):
            continue
        for key in ("safe_text_box", "render_bbox", "text_pixel_bbox", "bbox"):
            bbox = text.get(key)
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            except Exception:
                continue
            if x2 > x1 and y2 > y1:
                pad = 6 if key != "safe_text_box" else 0
                boxes.append([x1 - pad, y1 - pad, x2 + pad, y2 + pad])
                break
    return boxes


def _clean_false_dark_white_regions_for_render(
    *,
    inpainted_path: Path,
    texts: list[dict],
    temp_output_path: Path | None = None,
    update_inpaint: bool = False,
) -> Path:
    boxes = _false_dark_white_cleanup_boxes(texts)
    if not boxes or not inpainted_path.exists():
        return inpainted_path

    try:
        from PIL import Image, ImageDraw

        with Image.open(inpainted_path) as src:
            image = src.convert("RGB")
        changed = False
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, min(image.width, int(x1)))
            x2 = max(0, min(image.width, int(x2)))
            y1 = max(0, min(image.height, int(y1)))
            y2 = max(0, min(image.height, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            sample_pad = 18
            sx1 = max(0, x1 - sample_pad)
            sy1 = max(0, y1 - sample_pad)
            sx2 = min(image.width, x2 + sample_pad)
            sy2 = min(image.height, y2 + sample_pad)
            sample = image.crop((sx1, sy1, sx2, sy2))
            bright_pixels = [
                pixel
                for pixel in sample.getdata()
                if (pixel[0] + pixel[1] + pixel[2]) / 3.0 >= 210
                and (max(pixel) - min(pixel)) <= 36
            ]
            fill = (255, 255, 255)
            if bright_pixels:
                bright_pixels.sort()
                fill = bright_pixels[len(bright_pixels) // 2]
            draw = ImageDraw.Draw(image)
            draw.rectangle((x1, y1, x2, y2), fill=fill)
            changed = True
        if not changed:
            return inpainted_path
        target_path = inpainted_path if update_inpaint else temp_output_path
        if target_path is None:
            target_path = inpainted_path.with_name(f"{inpainted_path.stem}-false-white-clean{inpainted_path.suffix}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(target_path)
        return target_path
    except Exception as exc:
        logger.warning("Falha ao limpar falso painel escuro em balao branco antes do render: %s", exc)
        return inpainted_path


def _final_text_mask_cleanup_layer_allowed(text: dict) -> bool:
    if not isinstance(text, dict) or not _has_renderable_translated_text(text):
        return False
    route_action = str(text.get("route_action") or "").strip().lower()
    render_policy = str(text.get("render_policy") or "").strip().lower()
    content_class = str(text.get("content_class") or text.get("tipo") or "").strip().lower()
    if content_class == "sfx" or route_action == "translate_sfx_inpaint_render":
        return False
    if route_action in {"merged_into_primary", "review_required", "skip", "preserve_original"}:
        return False
    if render_policy in {"merged_into_primary", "preserve_original", "review_required"}:
        return False
    return route_action.startswith("translate") or route_action == ""


def _final_text_mask_cleanup_bbox(text: dict, width: int, height: int) -> list[int] | None:
    for key in ("source_text_mask_bbox", "_source_text_mask_bbox", "text_pixel_bbox"):
        bbox = _optional_bbox4(text.get(key))
        if bbox is None or _is_placeholder_mask_bbox(bbox):
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w > int(width * 0.72) or box_h > int(height * 0.20):
            continue
        if box_w * box_h > int(width * height * 0.075):
            continue
        return [x1, y1, x2, y2]
    return None


def _coerce_layer_rgb(value) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return tuple(int(max(0, min(255, round(float(v))))) for v in value[:3])
    except Exception:
        return None


def _final_text_mask_cleanup_fill_rgb(image, text: dict, bbox: list[int]) -> tuple[int, int, int]:
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if source in {"image_dark_bubble_mask", "image_dark_panel_mask", "derived_card_panel_mask"} or flags & {
        "visual_text_only_inpaint_contract",
        "text_contract_direct_fill",
        "dark_panel_style_grouped",
    }:
        return (0, 0, 0)
    hinted = _coerce_layer_rgb(text.get("background_rgb"))
    if hinted is not None:
        luma = sum(hinted) / 3.0
        if luma <= 80.0 or luma >= 180.0:
            return hinted

    try:
        import numpy as np

        x1, y1, x2, y2 = bbox
        pad = 18
        sx1 = max(0, x1 - pad)
        sy1 = max(0, y1 - pad)
        sx2 = min(image.width, x2 + pad)
        sy2 = min(image.height, y2 + pad)
        sample = np.asarray(image.crop((sx1, sy1, sx2, sy2)).convert("RGB"), dtype=np.uint8)
        if sample.size:
            flat = sample.reshape(-1, 3)
            luma = flat.astype(np.float32).mean(axis=1)
            dark = flat[luma <= 72]
            light = flat[luma >= 190]
            if dark.shape[0] >= max(24, flat.shape[0] // 12):
                rgb = np.median(dark, axis=0)
                return tuple(int(max(0, min(255, round(float(v))))) for v in rgb[:3])
            if light.shape[0] >= max(24, flat.shape[0] // 12):
                rgb = np.median(light, axis=0)
                return tuple(int(max(0, min(255, round(float(v))))) for v in rgb[:3])
    except Exception:
        pass

    return hinted if hinted is not None else (0, 0, 0)


def _apply_final_text_mask_cleanup_for_render(
    *,
    inpainted_path: Path,
    texts: list[dict],
    temp_output_path: Path | None = None,
    update_inpaint: bool = False,
) -> Path:
    if not texts or not inpainted_path.exists():
        return inpainted_path

    try:
        from PIL import Image, ImageDraw

        with Image.open(inpainted_path) as src:
            image = src.convert("RGB")

        changed = False
        draw = ImageDraw.Draw(image)
        for text in texts:
            if not _final_text_mask_cleanup_layer_allowed(text):
                continue
            bbox = _final_text_mask_cleanup_bbox(text, image.width, image.height)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            pad = 8
            style = text.get("estilo") or text.get("style")
            if isinstance(style, dict):
                for key in ("outline_width", "stroke_width", "glow_radius", "shadow_radius"):
                    try:
                        pad = max(pad, int(math.ceil(float(style.get(key) or 0))) + 4)
                    except Exception:
                        pass
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.width, x2 + pad)
            y2 = min(image.height, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            fill = _final_text_mask_cleanup_fill_rgb(image, text, [x1, y1, x2, y2])
            draw.rectangle((x1, y1, x2, y2), fill=fill)
            text.setdefault("qa_flags", [])
            if isinstance(text["qa_flags"], list) and "final_text_mask_cleanup" not in text["qa_flags"]:
                text["qa_flags"].append("final_text_mask_cleanup")
            changed = True

        if not changed:
            return inpainted_path
        target_path = inpainted_path if update_inpaint else temp_output_path
        if target_path is None:
            target_path = inpainted_path.with_name(f"{inpainted_path.stem}-text-mask-clean{inpainted_path.suffix}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(target_path)
        return target_path
    except Exception as exc:
        logger.warning("Falha ao limpar mascara textual final antes do render: %s", exc)
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

        _typeset_single_page((str(render_base_path), trans_page_dict, str(output_path.parent), project.get("font_assets")))
        renderer_output = output_path.parent / Path(render_base_path).name
        if renderer_output.exists() and renderer_output.resolve() != output_path.resolve():
            if output_path.exists():
                output_path.unlink()
            shutil.move(str(renderer_output), str(output_path))
        _apply_recovery_layer_for_page(project, page, output_path)

        emit("complete", output_path=str(output_path), renderer_backend=_active_renderer_backend_label())
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
    trans_page_dict = {"texts": _visible_render_texts(trans_texts), "_coordinate_space": "page"}
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
        _typeset_single_page((str(inpainted_path), trans_page_dict, str(output_dir), project.get("font_assets")))
        rendered_path = output_dir / img_name
        _apply_recovery_layer_for_page(project, page, rendered_path)
        page["text_layers"] = trans_texts
        _ensure_image_layer(page, "base", original_rel, visible=True, locked=True)
        _ensure_image_layer(page, "mask", _resolve_image_layer_path(page, "mask", f"layers/mask/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "inpaint", inpaint_rel, visible=True, locked=True)
        _ensure_image_layer(page, "brush", _resolve_image_layer_path(page, "brush", f"layers/brush/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "recovery", _resolve_image_layer_path(page, "recovery", f"layers/recovery/{page_number:03}.png"), visible=False, locked=False)
        _ensure_image_layer(page, "rendered", rendered_rel, visible=True, locked=True)
        _sync_page_legacy_aliases(page)
        project["versao"] = "2.0"
        _save_project_json(project_json_path, project)
        emit("complete", output_path=str(rendered_path), renderer_backend=_active_renderer_backend_label())
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        emit("error", message=f"Falha no retypeset: {e}\n{tb}")


def _run_reinpaint(project_json_path: Path, page_idx: int, region: dict | None = None, language_options: dict | None = None):
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
        _ensure_image_layer(page, "inpaint", inpaint_rel, visible=True, locked=True)
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
    work_dir = Path(config.get("work_dir")) if config.get("work_dir") else None
    for i, (img, ocr, text_page) in enumerate(zip(image_files, ocr_results, page_text_layers)):
        text_layers = text_page.get("texts", [])
        text_layers = _drop_suppressed_ocr_texts(
            text_layers,
            config.get("idioma_origem", "en"),
            sfx_candidates=ocr.get("_sfx_visual_candidates") if isinstance(ocr, dict) else [],
        )
        promoted_sfx = _promote_sfx_visual_candidates(ocr, existing_texts=text_layers)
        if promoted_sfx:
            text_layers = list(text_layers) + promoted_sfx
        text_layers = group_regions(text_layers)
        text_layers = [
            neutralize_removed_decision_fields(normalize_text_geometry(layer))
            for layer in text_layers
        ]
        qa_regions.extend(text_layers)
        engine_meta = ocr.get("_engine_preset") if isinstance(ocr.get("_engine_preset"), dict) else {}
        engine_steps = engine_meta.get("engine_steps")
        if not isinstance(engine_steps, list):
            engine_steps = []
        page_engine = {
            "engine_preset_id": ocr.get("engine_preset_id") or engine_meta.get("engine_preset_id") or config.get("engine_preset_id") or "",
            "content_family": engine_meta.get("content_family") or (config.get("engine_preset") or {}).get("content_family") or "",
            "mask_strategy": engine_meta.get("mask_strategy") or (config.get("engine_preset") or {}).get("mask_strategy") or "",
            "engine_steps": list(engine_steps),
        }
        inpaint_blocks = [
            project_block
            for block in ocr.get("_vision_blocks", [])
            for project_block in [_project_inpaint_block_from_vision_block(block)]
            if project_block is not None
        ]
        inpaint_blocks = [
            _enrich_project_inpaint_block_from_text_layers(block, text_layers)
            for block in inpaint_blocks
        ]

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
            "inpaint_blocks": inpaint_blocks,
            "page_profile": ocr.get("page_profile"),
            "page_quality": ocr.get("page_quality"),
            "route_history": ocr.get("route_history") or [],
            "vision_engine": page_engine,
            "text_layers": text_layers,
        }
        if work_dir is not None:
            _persist_real_bubble_mask_layer_for_page(
                page,
                ocr if isinstance(ocr, dict) else None,
                work_dir,
                page_number=i + 1,
                image_path=img,
            )
        _drop_runtime_bubble_mask_arrays_from_page(page)
        _sync_page_legacy_aliases(page)
        pages.append(page)

    return {
        "versao": "2.0",
        "app": "traduzai",
        "obra": config.get("obra", ""),
        "capitulo": config.get("capitulo", 1),
        "idioma_origem": config.get("idioma_origem", "en"),
        "idioma_destino": config.get("idioma_destino", "pt-BR"),
        "engine_preset_id": config.get("engine_preset_id") or "",
        "engine_preset": config.get("engine_preset") or {},
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


def _run_process_block(project_path: Path, page_idx: int, block_id: str, mode: str, language_options: dict | None = None):
    """Refazer processo para um unico bloco."""
    from ocr.detector import run_ocr_on_block
    from translator.translate import translate_single_block
    
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)
    _apply_page_action_language_options(project, language_options)

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
        text, conf = run_ocr_on_block(
            str(orig_img),
            block_bbox,
            idioma_origem=project.get("idioma_origem", "en"),
            engine_preset_id=project.get("engine_preset_id", ""),
        )
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

def _detector_block_to_vision_block(block) -> dict | None:
    if isinstance(block, dict):
        raw_bbox = block.get("bbox") or block.get("xyxy")
        confidence = block.get("confidence", 0.0)
    else:
        raw_bbox = getattr(block, "bbox", None) or getattr(block, "xyxy", None)
        confidence = getattr(block, "confidence", 0.0)
    bbox = _optional_bbox4(raw_bbox)
    if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    try:
        confidence_value = float(confidence or 0.0)
    except (TypeError, ValueError):
        confidence_value = 0.0
    return {"bbox": bbox, "confidence": confidence_value}


def _run_detect_boxes_page(
    project_path: Path,
    page_idx: int,
    region: dict | None = None,
    language_options: dict | None = None,
):
    from PIL import Image
    import numpy as np
    from vision_stack.runtime import _get_detector, _profile_to_detection_threshold

    project, work_dir, page, orig_img, img_name, original_rel_path = _load_editor_project_page(project_path, page_idx)
    _apply_page_action_language_options(project, language_options)

    page_number = int(page.get("numero", page_idx + 1) or page_idx + 1)
    original_rel = str(original_rel_path)
    is_regional = _region_bbox(region) is not None

    emit_progress("detect", 0, 10, message="Detectando caixas...")
    with Image.open(orig_img) as image:
        image_rgb = np.array(image.convert("RGB"))

    detector = _get_detector("max")
    raw_blocks = detector.detect(image_rgb, conf_threshold=_profile_to_detection_threshold("max"))
    detected_blocks = [
        project_block
        for block in raw_blocks
        for project_block in [_project_inpaint_block_from_vision_block(_detector_block_to_vision_block(block) or {})]
        if project_block is not None
    ]

    if is_regional:
        outside_blocks = [
            block
            for block in (page.get("inpaint_blocks") or [])
            if isinstance(block, dict) and not _block_in_region(block, region)
        ]
        detected_blocks = [block for block in detected_blocks if _block_in_region(block, region)]
        page["inpaint_blocks"] = outside_blocks + detected_blocks
    else:
        page["inpaint_blocks"] = detected_blocks

    page["text_layers"] = _page_text_layers_for_renderer(page, page_idx)
    _ensure_image_layer(page, "base", original_rel, visible=True, locked=True)
    _ensure_image_layer(page, "mask", _resolve_image_layer_path(page, "mask", f"layers/mask/{page_number:03}.png"), visible=False, locked=False)
    _ensure_image_layer(page, "inpaint", _resolve_image_layer_path(page, "inpaint", f"images/{img_name}"), visible=False, locked=True)
    _ensure_image_layer(page, "brush", _resolve_image_layer_path(page, "brush", f"layers/brush/{page_number:03}.png"), visible=False, locked=False)
    _ensure_image_layer(page, "recovery", _resolve_image_layer_path(page, "recovery", f"layers/recovery/{page_number:03}.png"), visible=False, locked=False)
    _ensure_image_layer(page, "rendered", _resolve_image_layer_path(page, "rendered", f"translated/{img_name}"), visible=True, locked=True)
    _sync_page_legacy_aliases(page)
    _save_project_json(project_path, project)

    out_img = work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")
    emit_progress("detect", 100, 100, message="Caixas detectadas!")
    emit("complete", output_path=str(out_img))

def _run_detect_page(project_path: Path, page_idx: int, region: dict | None = None, language_options: dict | None = None):
    from editor_vision_cache import (
        build_detect_ocr_cache_key,
        build_detect_ocr_payload,
        is_detect_ocr_payload,
        read_cache_entry,
        write_cache_entry,
    )

    project, work_dir, page, orig_img, img_name, original_rel_path = _load_editor_project_page(project_path, page_idx)
    _apply_page_action_language_options(project, language_options)

    page_number = int(page.get("numero", page_idx + 1) or page_idx + 1)
    original_rel = str(original_rel_path)
    is_regional = _region_bbox(region) is not None

    cache_key = None
    if not is_regional:
        cache_key = build_detect_ocr_cache_key(
            project_path=project_path,
            page_index=page_idx,
            image_path=orig_img,
            idioma_origem=project.get("idioma_origem", "en"),
            engine_preset_id=project.get("engine_preset_id", ""),
            schema_version=EDITOR_DETECT_OCR_CACHE_SCHEMA_VERSION,
        )
        cached = read_cache_entry(cache_key)
        if is_detect_ocr_payload(cached, page_index=page_idx):
            emit_progress("ocr", 0, 10, message="Aplicando deteccao em cache...")
            page["text_layers"] = cached["text_layers"]
            page["inpaint_blocks"] = cached["inpaint_blocks"]
            page["_ui_layout_components"] = list(cached.get("ui_layout_components") or [])
            _sync_page_legacy_aliases(page)
            _save_project_json(project_path, project)

            emit_progress("render", 80, 95, message="Rerenderizando visual...")
            out_img = work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")
            render_page_image(project, page_idx, str(out_img))

            emit_progress("render", 100, 100, message="Detecção concluída!")
            emit("complete", output_path=str(out_img))
            return

    from ocr.detector import run_ocr
    from ocr.contextual_reviewer import contextual_review_page
    from layout.balloon_layout import enrich_page_layout
    
    emit_progress("ocr", 0, 10, message="Detectando balões e textos...")
    
    # Run full OCR (detect + read)
    ocr_data = run_ocr(
        str(orig_img),
        models_dir=project.get("_models_dir", "models"),
        vision_worker_path=project.get("_vision_worker_path", ""),
        idioma_origem=project.get("idioma_origem", "en"),
        engine_preset_id=project.get("engine_preset_id", ""),
        work_title=project.get("obra", ""),
        work_title_user_provided=bool(project.get("work_title_user_provided")),
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
    reviewed_texts = _drop_suppressed_ocr_texts(
        reviewed_texts,
        project.get("idioma_origem", "en"),
        sfx_candidates=reviewed.get("_sfx_visual_candidates") if isinstance(reviewed, dict) else [],
    )
    reviewed_texts = reviewed_texts + _promote_sfx_visual_candidates(reviewed, existing_texts=reviewed_texts)
    if is_regional:
        detected_texts = [text for text in reviewed_texts if _bbox_in_region(text.get("bbox"), region)]
        outside_layers = [layer for layer in existing_layers if not _layer_in_region(layer, region)]
        translation_source_layers = [layer for layer in existing_layers if _layer_in_region(layer, region)]
    else:
        detected_texts = reviewed_texts
        outside_layers = []
        translation_source_layers = existing_layers
    carried_translations = _carry_translations_for_detected_layers(translation_source_layers, detected_texts)
    source_image_rgb = _load_source_image_rgb_for_style(orig_img, detected_texts)
    detected_layers = [
        build_text_layer(
            page_number=page_number,
            layer_index=len(outside_layers) + idx,
            ocr_text=text,
            translated=carried_translations[idx] if idx < len(carried_translations) else "",
            corpus_visual_benchmark=context.get("corpus_visual_benchmark", {}),
            corpus_textual_benchmark=context.get("corpus_textual_benchmark", {}),
            source_image_rgb=source_image_rgb,
        )
        for idx, text in enumerate(detected_texts)
    ]
    page["text_layers"] = outside_layers + detected_layers
    if not is_regional:
        page["_ui_layout_components"] = list(reviewed.get("_ui_layout_components") or [])
    _persist_real_bubble_mask_layer_for_page(
        page,
        reviewed,
        work_dir,
        page_number=page_number,
        image_path=orig_img,
    )

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

    if cache_key is not None:
        try:
            write_cache_entry(
                cache_key,
                build_detect_ocr_payload(
                    page_index=page_idx,
                    text_layers=page["text_layers"],
                    inpaint_blocks=page["inpaint_blocks"],
                    ui_layout_components=page.get("_ui_layout_components") or [],
                ),
            )
        except Exception as exc:
            logger.warning("Falha ao gravar cache de deteccao do editor: %s", exc)
    
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

def _run_ocr_page(project_path: Path, page_idx: int, region: dict | None = None, language_options: dict | None = None):
    from editor_vision_cache import (
        build_ocr_layers_cache_key,
        build_ocr_layers_payload,
        is_ocr_layers_payload,
        read_cache_entry,
        write_cache_entry,
    )

    project, work_dir, page, orig_img, img_name, _original_rel_path = _load_editor_project_page(project_path, page_idx)
    _apply_page_action_language_options(project, language_options)

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
    is_regional = _region_bbox(region) is not None
    cache_key = None
    if not is_regional:
        cache_key = build_ocr_layers_cache_key(
            project_path=project_path,
            page_index=page_idx,
            image_path=orig_img,
            layers=layers,
            idioma_origem=project.get("idioma_origem", "en"),
            engine_preset_id=project.get("engine_preset_id", ""),
            schema_version=1,
        )
        cached = read_cache_entry(cache_key)
        if is_ocr_layers_payload(cached, page_index=page_idx):
            emit_progress("ocr", 0, 10, message="Aplicando OCR em cache...")
            layer_updates = cached["layer_updates"]
            layers_by_id = {
                str(layer.get("id") or "").strip(): layer
                for layer in layers
                if isinstance(layer, dict) and str(layer.get("id") or "").strip()
            }
            applied_count = 0
            for update in layer_updates:
                if not isinstance(update, dict):
                    continue
                update_id = str(update.get("id") or "").strip()
                if not update_id:
                    continue
                layer = layers_by_id.get(update_id)
                if layer is None:
                    continue
                for field in ("original", "ocr_confidence", "confianca_ocr"):
                    if field in update:
                        layer[field] = update[field]
                applied_count += 1

            if applied_count > 0:
                page["text_layers"] = _page_text_layers_for_renderer(page, page_idx)
                _sync_page_legacy_aliases(page)
                _save_project_json(project_path, project)

                emit_progress("render", 80, 95, message="Rerenderizando visual...")
                out_img = work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")
                render_page_image(project, page_idx, str(out_img))

                emit_progress("render", 100, 100, message="OCR concluído!")
                emit("complete", output_path=str(out_img))
                return

    from ocr.detector import run_ocr_on_block

    target_layers = [layer for layer in layers if _layer_in_region(layer, region)]
    total = len(target_layers)
    
    for i, layer in enumerate(target_layers):
        progress = int((i / max(1, total)) * 100)
        emit_progress("ocr", progress, 10, message=f"OCR em bloco {i+1}/{total}...")
        block_bbox = _resolved_layer_bbox(layer)
        layer["bbox"] = block_bbox
        text, conf = run_ocr_on_block(
            str(orig_img),
            block_bbox,
            idioma_origem=project.get("idioma_origem", "en"),
            engine_preset_id=project.get("engine_preset_id", ""),
        )
        layer["original"] = text
        layer["ocr_confidence"] = conf
        layer["confianca_ocr"] = conf

    if cache_key is not None:
        try:
            write_cache_entry(
                cache_key,
                build_ocr_layers_payload(
                    page_index=page_idx,
                    layer_updates=[
                        {
                            "id": layer.get("id"),
                            "original": layer.get("original", ""),
                            "ocr_confidence": layer.get("ocr_confidence"),
                            "confianca_ocr": layer.get("confianca_ocr"),
                        }
                        for layer in target_layers
                        if isinstance(layer, dict) and layer.get("id")
                    ],
                ),
            )
        except Exception as exc:
            logger.warning("Falha ao gravar cache de OCR do editor: %s", exc)

    page["text_layers"] = _page_text_layers_for_renderer(page, page_idx)
    _sync_page_legacy_aliases(page)
    _save_project_json(project_path, project)

    emit_progress("render", 80, 95, message="Rerenderizando visual...")
    out_img = work_dir / _resolve_image_layer_path(page, "rendered", f"translated/{img_name}")
    render_page_image(project, page_idx, str(out_img))
    
    emit_progress("render", 100, 100, message="OCR concluído!")
    emit("complete", output_path=str(out_img))

def _preload_detect_ocr_page(
    project_path: Path,
    page_idx: int,
    language_options: dict | None = None,
) -> dict:
    from editor_vision_cache import (
        build_detect_ocr_cache_key,
        build_detect_ocr_payload,
        is_detect_ocr_payload,
        read_cache_entry,
        write_cache_entry,
    )
    from ocr.detector import run_ocr
    from ocr.contextual_reviewer import contextual_review_page
    from layout.balloon_layout import enrich_page_layout

    project, _work_dir, page, orig_img, _img_name, _original_rel_path = _load_editor_project_page(project_path, page_idx)
    _apply_page_action_language_options(project, language_options)
    cache_key = build_detect_ocr_cache_key(
        project_path=project_path,
        page_index=page_idx,
        image_path=orig_img,
        idioma_origem=project.get("idioma_origem", "en"),
        engine_preset_id=project.get("engine_preset_id", ""),
        schema_version=5,
    )
    if is_detect_ocr_payload(read_cache_entry(cache_key), page_index=page_idx):
        return {"cache": "ready", "kind": "detect_ocr", "reused": True}

    ocr_data = run_ocr(
        str(orig_img),
        models_dir=project.get("_models_dir", "models"),
        vision_worker_path=project.get("_vision_worker_path", ""),
        idioma_origem=project.get("idioma_origem", "en"),
        engine_preset_id=project.get("engine_preset_id", ""),
        work_title=project.get("obra", ""),
        work_title_user_provided=bool(project.get("work_title_user_provided")),
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

    page_number = int(page.get("numero", page_idx + 1) or page_idx + 1)
    existing_layers = _page_text_layers_for_renderer(page, page_idx)
    context = project.get("contexto", {}) or {}
    reviewed_texts = [text for text in (reviewed.get("texts") or []) if isinstance(text, dict)]
    reviewed_texts = _drop_suppressed_ocr_texts(
        reviewed_texts,
        project.get("idioma_origem", "en"),
        sfx_candidates=reviewed.get("_sfx_visual_candidates") if isinstance(reviewed, dict) else [],
    )
    reviewed_texts = reviewed_texts + _promote_sfx_visual_candidates(reviewed, existing_texts=reviewed_texts)
    carried_translations = _carry_translations_for_detected_layers(existing_layers, reviewed_texts)
    source_image_rgb = _load_source_image_rgb_for_style(orig_img, reviewed_texts)
    text_layers = [
        build_text_layer(
            page_number=page_number,
            layer_index=idx,
            ocr_text=text,
            translated=carried_translations[idx] if idx < len(carried_translations) else "",
            corpus_visual_benchmark=context.get("corpus_visual_benchmark", {}),
            corpus_textual_benchmark=context.get("corpus_textual_benchmark", {}),
            source_image_rgb=source_image_rgb,
        )
        for idx, text in enumerate(reviewed_texts)
    ]
    inpaint_blocks = [
        {
            "bbox": _bbox4(block.get("bbox")),
            "confidence": float(block.get("confidence", 0.0) or 0.0),
        }
        for block in reviewed.get("_vision_blocks", [])
        if isinstance(block, dict)
    ]
    try:
        write_cache_entry(
            cache_key,
            build_detect_ocr_payload(
                page_index=page_idx,
                text_layers=text_layers,
                inpaint_blocks=inpaint_blocks,
                ui_layout_components=reviewed.get("_ui_layout_components") or [],
            ),
        )
    except Exception as exc:
        logger.warning("Falha ao gravar preload de deteccao do editor: %s", exc)
        return {"cache": "miss", "kind": "detect_ocr"}
    return {"cache": "ready", "kind": "detect_ocr"}


def _preload_ocr_layers_page(
    project_path: Path,
    page_idx: int,
    language_options: dict | None = None,
) -> dict:
    from editor_vision_cache import (
        build_ocr_layers_cache_key,
        build_ocr_layers_payload,
        is_ocr_layers_payload,
        read_cache_entry,
        write_cache_entry,
    )
    from ocr.detector import run_ocr_on_block

    project, _work_dir, page, orig_img, _img_name, _original_rel_path = _load_editor_project_page(project_path, page_idx)
    _apply_page_action_language_options(project, language_options)

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

    cache_key = build_ocr_layers_cache_key(
        project_path=project_path,
        page_index=page_idx,
        image_path=orig_img,
        layers=layers,
        idioma_origem=project.get("idioma_origem", "en"),
        engine_preset_id=project.get("engine_preset_id", ""),
        schema_version=1,
    )
    if is_ocr_layers_payload(read_cache_entry(cache_key), page_index=page_idx):
        return {"cache": "ready", "kind": "ocr_layers", "reused": True}

    layer_updates: list[dict] = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_id = str(layer.get("id") or "").strip()
        if not layer_id:
            continue
        text, conf = run_ocr_on_block(
            str(orig_img),
            _resolved_layer_bbox(layer),
            idioma_origem=project.get("idioma_origem", "en"),
            engine_preset_id=project.get("engine_preset_id", ""),
        )
        layer_updates.append(
            {
                "id": layer_id,
                "original": text,
                "ocr_confidence": conf,
                "confianca_ocr": conf,
            }
        )
    try:
        write_cache_entry(
            cache_key,
            build_ocr_layers_payload(page_index=page_idx, layer_updates=layer_updates),
        )
    except Exception as exc:
        logger.warning("Falha ao gravar preload de OCR do editor: %s", exc)
        return {"cache": "miss", "kind": "ocr_layers"}
    return {"cache": "ready", "kind": "ocr_layers"}


def _run_translate_page(project_path: Path, page_idx: int, region: dict | None = None, language_options: dict | None = None):
    from translator.translate import translate_pages
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)
    _apply_page_action_language_options(project, language_options)

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
    merged_layers = list(source_layers)
    for idx, layer in indexed_layers:
        if isinstance(layer, dict) and _layer_in_region(layer, region):
            merged_layers[idx] = enrich_sfx_candidate(layer)
    target_indexed_layers = [
        (idx, layer)
        for idx, layer in enumerate(merged_layers)
        if (
            isinstance(layer, dict)
            and _layer_in_region(layer, region)
            and str(layer.get("content_class") or "").strip().lower() != "sfx"
            and str(layer.get("route_action") or "").strip().lower() != "translate_sfx_inpaint_render"
        )
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
        for (source_idx, _), translated_layer in zip(target_indexed_layers, translated_targets):
            merged_layers[source_idx] = translated_layer
    else:
        merged_layers = list(merged_layers)
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
            "  --detect-boxes-page <project> <page> Reprocessa apenas caixas detectadas da pagina",
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
