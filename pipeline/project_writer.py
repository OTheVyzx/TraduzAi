"""Transactional project.json writer."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


def _neutralize_removed_decision_fields(layer: dict[str, Any]) -> None:
    route_action = str(layer.get("route_action") or "").strip().lower()
    content_class = str(layer.get("content_class") or "").strip().lower()
    if route_action == "translate_sfx_inpaint_render" or content_class == "sfx" or isinstance(layer.get("sfx"), dict):
        layer["tipo"] = "sfx"
        layer["content_class"] = "sfx"
        layer["skip_processing"] = False
        render_policy = str(layer.get("render_policy") or "").strip().lower()
        translate_policy = str(layer.get("translate_policy") or "").strip().lower()
        sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
        preserve_sfx = bool(
            layer.get("preserve_original")
            or render_policy in {"preserve", "preserve_original"}
            or translate_policy == "skip_translation"
            or sfx.get("inpaint_allowed") is False
        )
        if preserve_sfx:
            layer["preserve_original"] = True
            layer["translate_policy"] = "skip_translation"
            layer["render_policy"] = "preserve_original"
            layer["route_action"] = "review_required"
            layer["route_reason"] = layer.get("route_reason") or "sfx_preserved"
            if isinstance(sfx, dict):
                sfx["inpaint_allowed"] = False
                layer["sfx"] = sfx
            return
        layer["preserve_original"] = False
        if str(layer.get("translate_policy") or "").strip().lower() in {"", "translate"}:
            layer["translate_policy"] = "adapt_sfx"
        if str(layer.get("render_policy") or "").strip().lower() in {"", "normal"}:
            layer["render_policy"] = "sfx_style"
        layer["route_action"] = layer.get("route_action") or "translate_sfx_inpaint_render"
        return
    layer["tipo"] = "text"
    layer["content_class"] = "text"
    layer["balloon_type"] = ""
    layer["skip_processing"] = False
    layer["preserve_original"] = False
    layer["translate_policy"] = "translate"
    layer["render_policy"] = "normal"
    layer["route_action"] = layer.get("route_action") or "translate_inpaint_render"
    layer.pop("skip_reason", None)


def neutralize_project_compatibility_metadata(project: dict[str, Any]) -> dict[str, Any]:
    for page in project.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        for key in ("text_layers", "textos", "texts"):
            for layer in page.get(key) or []:
                if isinstance(layer, dict):
                    _neutralize_removed_decision_fields(layer)
    return project


def validate_project_consistency(project: dict[str, Any]) -> None:
    pages = project.get("paginas")
    if not isinstance(pages, list):
        raise ValueError("project.json invalido: 'paginas' precisa ser lista")
    stats = project.get("estatisticas") or {}
    if "total_paginas" in stats and int(stats["total_paginas"]) != len(pages):
        raise ValueError("summary mismatch: estatisticas.total_paginas nao bate com paginas")
    qa = project.get("qa") or {}
    summary = qa.get("summary")
    if summary:
        flags = []
        for page in pages:
            for layer in page.get("text_layers", []) or []:
                if isinstance(layer, dict):
                    flags.extend(layer.get("qa_flags") or [])
        if int(summary.get("total", 0) or 0) != len(flags):
            raise ValueError("qa.summary nao bate com qa.flags")
    log_summary = (project.get("log") or {}).get("summary")
    if log_summary:
        from structured_logger import build_log_summary

        expected = build_log_summary(project)
        for key in ("actual_pages", "processed_pages", "translated_regions", "qa_flags", "critical_flags"):
            if log_summary.get(key) != expected.get(key):
                raise ValueError(f"log.summary nao bate com project.json: {key}")


def write_project_json_atomic(project_json_path: Path, project: dict[str, Any]) -> None:
    project_json_path = Path(project_json_path)
    neutralize_project_compatibility_metadata(project)
    validate_project_consistency(project)
    if project_json_path.exists():
        backup = project_json_path.with_name(f"project.backup.{int(time.time())}.json")
        backup.write_bytes(project_json_path.read_bytes())
    tmp_path = project_json_path.with_suffix(project_json_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False, indent=2)
    loaded = json.loads(tmp_path.read_text(encoding="utf-8"))
    validate_project_consistency(loaded)
    try:
        tmp_path.replace(project_json_path)
    except PermissionError:
        # Alguns diretórios Windows permitem escrita, mas bloqueiam o rename atômico.
        # Nesses casos, mantemos o conteúdo validado e fazemos fallback por cópia.
        shutil.copyfile(tmp_path, project_json_path)
        try:
            tmp_path.unlink(missing_ok=True)
        except PermissionError:
            pass
