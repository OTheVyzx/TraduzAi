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
from pathlib import Path

# Adiciona o diretório da pipeline ao path para resolver imports locais no Pyright/Linter
pipeline_root = Path(__file__).parent.absolute()
if str(pipeline_root) not in sys.path:
    sys.path.insert(0, str(pipeline_root))

# Lazy imports are moved inside functions to allow fast --hardware-info and --list-supported-languages calls

from utils.decision_log import configure_decision_trace, finalize_decision_trace
from typesetter.style_policy import normalize_auto_typesetting_style
from layout.simple_text_geometry import normalize_text_geometry, resolve_text_anchor_bbox, sanitize_simple_text_geometry
from ocr.text_router import ROUTE_ACTIONS

_EMIT_STDOUT_FAILED = False
_PIPELINE_FILE_HANDLER: logging.Handler | None = None
logger = logging.getLogger(__name__)

REMOVED_AUTOMATIC_DECISION_FIELDS = {
    "tipo",
    "content_class",
    "balloon_type",
    "skip_processing",
    "preserve_original",
}


def neutralize_removed_decision_fields(layer: dict) -> dict:
    """Keep removed legacy fields as neutral compatibility metadata only."""
    normalized = dict(layer or {})
    normalized["tipo"] = "text"
    normalized["content_class"] = "text"
    normalized["balloon_type"] = ""
    normalized["skip_processing"] = False
    normalized["preserve_original"] = False
    normalized["translate_policy"] = "translate"
    normalized["render_policy"] = "normal"
    normalized["route_action"] = normalized.get("route_action") or "translate_inpaint_render"
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
    return filled


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


def _merge_layer_qa_flags(layer: dict, flags: list[str]) -> None:
    merged = list(layer.get("qa_flags") or [])
    for flag in flags:
        flag = str(flag).strip()
        if flag and flag not in merged:
            merged.append(flag)
    layer["qa_flags"] = merged


def _page_text_coordinate_audit_flags(page_texts: list[dict], *, height: int, width: int) -> list[str]:
    try:
        from debug_tools.bbox import audit_bbox_coordinate_space, coordinate_audit_flags, layout_block_records
    except Exception:
        return []
    page = {"height": int(height), "width": int(width), "texts": page_texts}
    try:
        return coordinate_audit_flags(audit_bbox_coordinate_space(layout_block_records([page])))
    except Exception:
        return []


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

    for field in ("trace_id", "text_instance_id", "id", "text_id"):
        add_key(layer.get(field))
    for field in ("source_trace_ids", "_source_trace_ids", "trace_ids"):
        for value in layer.get(field) or []:
            add_key(value)
    source_text_ids: list[str] = []
    for field in ("source_text_ids", "_source_text_ids", "text_ids"):
        for value in layer.get(field) or []:
            text_id = str(value or "").strip()
            if text_id and text_id not in source_text_ids:
                source_text_ids.append(text_id)
                add_key(text_id)
    if band_id:
        for text_id in source_text_ids:
            add_key(f"{text_id}@{band_id}")
            add_key(f"{band_id}_{text_id}")
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
    return filtered


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
    if _metric_int(entry, "significant_component_count") < 2:
        return False
    if _metric_int(entry, "significant_area") < 300:
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
        return [_offset_nested_debug_bboxes(item, dx, dy) for item in value]
    return value


def _candidate_offset_from_layout(entry: dict, layout_blocks: list[dict]) -> tuple[int, int]:
    candidate_refs = [
        _optional_bbox4(entry.get("target_bbox")),
        _optional_bbox4(entry.get("bbox")),
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
    for key in ("target_bbox", "safe_text_box", "_debug_safe_text_box", "render_bbox", "bbox"):
        shifted = _offset_bbox4(_optional_bbox4(candidate.get(key)), dx, dy)
        if shifted is not None:
            candidate[key] = shifted
    for key in ("qa_metrics", "_render_debug", "_render_debug_candidates", "_render_debug_skipped"):
        if isinstance(candidate.get(key), (dict, list)):
            candidate[key] = _offset_nested_debug_bboxes(candidate[key], dx, dy)
    candidate["_project_coordinate_offset"] = [dx, dy]
    return candidate


def _render_candidate_score_for_layer(layer: dict, candidate: dict) -> float:
    render_bbox = _optional_bbox4(candidate.get("render_bbox"))
    safe_text_box = _optional_bbox4(candidate.get("safe_text_box")) or _optional_bbox4(candidate.get("_debug_safe_text_box"))
    if render_bbox is None or safe_text_box is None:
        return -1.0
    layer_refs = [
        _optional_bbox4(layer.get("render_bbox")),
        _optional_bbox4(layer.get("safe_text_box")),
        _optional_bbox4(layer.get("text_pixel_bbox")),
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
    return best


def _hydrate_project_render_metadata_from_debug_candidates(project_data: dict) -> dict:
    audit = {
        "applied": False,
        "candidate_count": 0,
        "hydrated_layers": 0,
        "missing_debug_root": False,
    }
    debug_root = _debug_root_from_project(project_data)
    if debug_root is None:
        audit["missing_debug_root"] = True
        return audit
    candidate_entries = [
        entry
        for entry in _load_debug_jsonl(debug_root / "09_typeset" / "render_plan_candidates.jsonl")
        if _optional_bbox4(entry.get("render_bbox")) is not None
        and (
            _optional_bbox4(entry.get("safe_text_box")) is not None
            or _optional_bbox4(entry.get("_debug_safe_text_box")) is not None
        )
    ]
    audit["candidate_count"] = len(candidate_entries)
    if not candidate_entries:
        audit["applied"] = True
        return audit

    layout_blocks_by_identity = _layout_blocks_by_debug_identity(debug_root)
    candidates_by_identity: dict[str, list[dict]] = {}
    for entry in candidate_entries:
        layout_blocks: list[dict] = []
        for identity_key in _debug_identity_keys_from_payload(entry):
            layout_blocks.extend(layout_blocks_by_identity.get(identity_key, []))
        candidate = _render_candidate_with_project_coordinates(entry, layout_blocks)
        for identity_key in _debug_identity_keys_from_payload(candidate):
            candidates_by_identity.setdefault(identity_key, []).append(candidate)

    hydrated = 0
    for layer in _iter_project_text_layers(project_data):
        if _optional_bbox4(layer.get("render_bbox")) is not None and _optional_bbox4(layer.get("safe_text_box")) is not None:
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
            continue
        best = max(candidates, key=lambda candidate: _render_candidate_score_for_layer(layer, candidate))
        if _render_candidate_score_for_layer(layer, best) < 0.20:
            continue
        render_bbox = _optional_bbox4(best.get("render_bbox"))
        safe_text_box = _optional_bbox4(best.get("safe_text_box")) or _optional_bbox4(best.get("_debug_safe_text_box"))
        if render_bbox is None or safe_text_box is None:
            continue
        layer["render_bbox"] = render_bbox
        layer["safe_text_box"] = safe_text_box
        layer["_debug_safe_text_box"] = safe_text_box
        target_bbox = _optional_bbox4(best.get("target_bbox"))
        if target_bbox is not None:
            layer["target_bbox"] = target_bbox
        if best.get("fit_status"):
            layer["fit_status"] = best.get("fit_status")
        if isinstance(best.get("fit_attempts"), list):
            layer["fit_attempts"] = best.get("fit_attempts")
        if isinstance(best.get("qa_metrics"), dict):
            layer["qa_metrics"] = {**dict(layer.get("qa_metrics") or {}), **dict(best.get("qa_metrics") or {})}
        if best.get("_project_coordinate_offset"):
            layer["_render_metadata_project_coordinate_offset"] = best.get("_project_coordinate_offset")
        _merge_layer_qa_flags(layer, [str(flag) for flag in best.get("qa_flags") or [] if str(flag).strip()])
        hydrated += 1
    audit["applied"] = True
    audit["hydrated_layers"] = hydrated
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
            if len(layers) == 1 and layers[0] not in matched_layers:
                matched_layers.append(layers[0])
        if matched_layers:
            return matched_layers, list(group)
    return [], list(identity_groups[0]) if identity_groups else []


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
                _merge_layer_qa_flags(layer, sorted(_filter_debug_claim_flags_for_project_layer(layer, flags)))
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
    render_bbox = _optional_bbox4(layer.get("render_bbox"))
    safe_text_box = _optional_bbox4(layer.get("safe_text_box"))
    has_render_geometry = render_bbox is not None and safe_text_box is not None
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
        "source_bbox": _optional_bbox4(layer.get("source_bbox")),
        "text_pixel_bbox": _optional_bbox4(layer.get("text_pixel_bbox")),
        "bbox": _optional_bbox4(layer.get("bbox")),
        "content_class": layer.get("content_class"),
        "tipo": layer.get("tipo"),
        "qa_flags": list(layer.get("qa_flags") or []),
        "qa_metrics": dict(layer.get("qa_metrics") or {}),
        "warnings": list(layer.get("warnings") or []),
    }
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
    }
    if not recorder or not isinstance(project_data, dict):
        return audit

    rows: list[dict] = []
    missing_identity: list[dict] = []
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
    _write_debug_jsonl_replace(recorder, "09_typeset/render_plan_final.jsonl", rows)
    try:
        recorder.write_json("09_typeset/render_plan_final_sync.json", audit)
    except Exception:
        pass
    return audit


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
                    except Exception:
                        pass
                    cv2.imwrite(str(inpaint_target), p.inpainted_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                else:
                    shutil.copy2(p.path, inpaint_target)
            strip_chapter_telemetry["main_sync_page_clamp_count"] = main_sync_page_clamp_count

        with pipeline_timing.measure("sync_final_page_space_typeset"):
            final_page_space_count = 0
            from PIL import Image as _PILImage

            for page_number, p in enumerate(output_pages, start=1):
                page_texts = _final_page_space_text_layers_for_renderer(p.text_layers, page_number=page_number)
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
                    strip_chapter_telemetry["main_final_page_space_rerender_blocked_count"] = (
                        int(strip_chapter_telemetry.get("main_final_page_space_rerender_blocked_count") or 0) + 1
                    )
                    continue
                clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
                try:
                    rendered_rgb = typesetter_mod.render_band_image(
                        clean_rgb,
                        {"texts": page_texts, "_coordinate_space": "main_final_page_space_typeset"},
                    )
                except Exception:
                    continue
                _replace_output_page_text_layers(p, page_texts)
                _PILImage.fromarray(rendered_rgb).save(p.path, quality=92)
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
    ) or _bbox4(ocr_text.get("layout_bbox"), source_bbox)
    force_black_text = (
        str(ocr_text.get("layout_profile") or ocr_text.get("block_profile") or "").strip().lower() == "white_balloon"
    )
    style = normalize_auto_typesetting_style(
        _merge_style(ocr_text.get("estilo")),
        _coerce_background_rgb(ocr_text.get("background_rgb")),
        force_black_text=force_black_text,
    )
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

    return {
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
        "rotation_source": ocr_text.get("rotation_source"),
        "detected_font_size_px": ocr_text.get("detected_font_size_px"),
        "skip_processing": _route_derived_skip_processing(ocr_text),
        "skip_reason": ocr_text.get("skip_reason"),
        "content_class": ocr_text.get("content_class"),
        "translate_policy": "translate",
        "render_policy": "normal",
        "route_action": ocr_text.get("route_action"),
        "route_reason": ocr_text.get("route_reason"),
        "is_watermark": bool(ocr_text.get("is_watermark", False)),
        "is_non_english": bool(ocr_text.get("is_non_english", False)),
        "smart_skip_decision": ocr_text.get("smart_skip_decision"),
        "balloon_type": ocr_text.get("balloon_type"),
        "inpaint_mode": ocr_text.get("inpaint_mode"),
        "inpaint_strategy": ocr_text.get("inpaint_strategy"),
        "balloon_bbox": _bbox4(ocr_text.get("balloon_bbox"), layout_bbox),
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
    text_pixel_bbox = _bbox4(
        _normalize_relative_y_bbox(raw_layer.get("text_pixel_bbox"), source_bbox),
        source_bbox,
    )
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

    return {
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
        "render_bbox": raw_layer.get("render_bbox"),
        "bbox": layout_bbox,
        "text_pixel_bbox": text_pixel_bbox,
        "tipo": raw_layer.get("tipo", "fala"),
        "original": original,
        "raw_ocr": raw_ocr,
        "normalized_ocr": normalized_ocr,
        "normalized_text_final": normalized_text_final,
        "normalization": normalization,
        "normalization_trace": normalization_trace,
        "translated": translated,
        "text": raw_layer.get("text", original),
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
        "skip_processing": _route_derived_skip_processing(raw_layer),
        "skip_reason": raw_layer.get("skip_reason"),
        "content_class": raw_layer.get("content_class"),
        "translate_policy": "translate",
        "render_policy": "normal",
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
        "bubble_mask_bbox": _bbox4(raw_layer.get("bubble_mask_bbox")),
        "bubble_inner_bbox": _bbox4(raw_layer.get("bubble_inner_bbox")),
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


def _raw_text_layers_from_collection(text_layers) -> list[dict]:
    if isinstance(text_layers, dict):
        text_layers = text_layers.get("texts")
    return [layer for layer in list(text_layers or []) if isinstance(layer, dict)]


def _final_page_space_text_layers_for_renderer(text_layers, *, page_number: int) -> list[dict]:
    return [
        _normalize_text_layer_for_renderer(layer, int(page_number), idx)
        for idx, layer in enumerate(_raw_text_layers_from_collection(text_layers))
    ]


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

    text_layers = [
        neutralize_removed_decision_fields(normalize_text_geometry(layer))
        for layer in (page.get("text_layers") or [])
        if isinstance(layer, dict)
    ]
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
            "bubble_mask_bbox": _bbox4(layer.get("bubble_mask_bbox")),
            "bubble_inner_bbox": _bbox4(layer.get("bubble_inner_bbox")),
            "balloon_type": layer.get("balloon_type"),
            "layout_profile": layer.get("layout_profile") or layer.get("block_profile"),
            "layout_group_size": int(layer.get("layout_group_size") or 1),
            "skip_processing": _route_derived_skip_processing(layer),
            "skip_reason": layer.get("skip_reason"),
            "content_class": layer.get("content_class"),
            "translate_policy": layer.get("translate_policy"),
            "render_policy": layer.get("render_policy"),
            "route_action": layer.get("route_action"),
            "route_reason": layer.get("route_reason"),
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
        "balloon_bbox",
        "balloon_type",
        "block_profile",
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
        "line_polygons",
        "text_pixel_bbox",
        "bbox",
        "source_bbox",
        "balloon_bbox",
        "layout_bbox",
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
    qa["summary"] = summarize_flags(regions)


def _save_project_json(project_json_path: Path, project: dict) -> None:
    from project_writer import write_project_json_atomic

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
    original_rel = _resolve_image_layer_path(page, "base", page.get("arquivo_original", ""))
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
    if not isinstance(text, dict):
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
            schema_version=1,
        )
        cached = read_cache_entry(cache_key)
        if is_detect_ocr_payload(cached, page_index=page_idx):
            emit_progress("ocr", 0, 10, message="Aplicando deteccao em cache...")
            page["text_layers"] = cached["text_layers"]
            page["inpaint_blocks"] = cached["inpaint_blocks"]
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

    if cache_key is not None:
        try:
            write_cache_entry(
                cache_key,
                build_detect_ocr_payload(
                    page_index=page_idx,
                    text_layers=page["text_layers"],
                    inpaint_blocks=page["inpaint_blocks"],
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
        schema_version=1,
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
    carried_translations = _carry_translations_for_detected_layers(existing_layers, reviewed_texts)
    text_layers = [
        build_text_layer(
            page_number=page_number,
            layer_index=idx,
            ocr_text=text,
            translated=carried_translations[idx] if idx < len(carried_translations) else "",
            corpus_visual_benchmark=context.get("corpus_visual_benchmark", {}),
            corpus_textual_benchmark=context.get("corpus_textual_benchmark", {}),
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
