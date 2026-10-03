"""Transactional project.json writer."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from ownership.project import (
    neutralize_review_candidate_compatibility,
    require_owner_project_consistency,
)


_FINAL_PIXEL_CONTRACTS = {
    "source_coverage_contract",
    "owner_graph_contract",
    "route_state_contract",
    "pixel_ownership_contract",
    "final_language_contract",
    "qa_integrity_contract",
}


def _validate_final_pixel_reports(project: dict[str, Any]) -> None:
    qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
    reports = qa.get("final_pixel_reports")
    if reports is None:
        return
    if not isinstance(reports, list):
        raise ValueError("qa.final_pixel_reports precisa ser lista")
    seen: set[str] = set()
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("qa.final_pixel_reports contem relatorio invalido")
        page_id = str(report.get("page_id") or "").strip()
        if not page_id or page_id in seen:
            raise ValueError("qa.final_pixel_reports exige page_id unico")
        seen.add(page_id)
        contracts = report.get("contracts")
        if not isinstance(contracts, dict) or not _FINAL_PIXEL_CONTRACTS.issubset(contracts):
            raise ValueError(f"final pixel contracts incompletos: {page_id}")
        if not isinstance(report.get("issues"), list):
            raise ValueError(f"final pixel issues invalidos: {page_id}")

    export_gate = qa.get("export_gate") if isinstance(qa.get("export_gate"), dict) else {}
    if str(export_gate.get("status") or "").upper() != "PASS":
        return
    for report in reports:
        artifact_path = Path(str(report.get("artifact_path") or ""))
        expected_hash = str(report.get("persisted_sha256") or "").strip().lower()
        try:
            actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            raise ValueError(f"final pixel artifact indisponivel: {report['page_id']}") from exc
        if actual_hash != expected_hash:
            raise ValueError(f"final pixel artifact mudou apos o gate: {report['page_id']}")


_RASTER_MASK_KEYS = frozenset(
    {
        "bubble_mask",
        "bubbleMask",
        "balloon_mask",
        "balloonMask",
        "segmentation_mask",
        "mask",
    }
)


def _strip_raster_mask_fields(record: Any) -> None:
    if isinstance(record, list):
        for item in record:
            _strip_raster_mask_fields(item)
        return
    if not isinstance(record, dict):
        return
    for key in tuple(record.keys()):
        if key in _RASTER_MASK_KEYS:
            record.pop(key, None)
        elif key == "metadata":
            _strip_raster_mask_fields(record[key])


def _neutralize_removed_decision_fields(layer: dict[str, Any]) -> None:
    route_action = str(layer.get("route_action") or "").strip().lower()
    content_class = str(layer.get("content_class") or "").strip().lower()
    if neutralize_review_candidate_compatibility(layer):
        # Review candidates cannot execute or render; preserve their strict
        # route/render contract while neutralizing legacy compatibility hints.
        return
    if (
        route_action == "translate_sfx_inpaint_render"
        or content_class == "sfx"
        or isinstance(layer.get("sfx"), dict)
    ):
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
        if str(layer.get("translate_policy") or "").strip().lower() in {
            "",
            "translate",
        }:
            layer["translate_policy"] = "adapt_sfx"
        if str(layer.get("render_policy") or "").strip().lower() in {"", "normal"}:
            layer["render_policy"] = "sfx_style"
        layer["route_action"] = (
            layer.get("route_action") or "translate_sfx_inpaint_render"
        )
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


def neutralize_project_compatibility_metadata(
    project: dict[str, Any],
) -> dict[str, Any]:
    for page in project.get("paginas") or []:
        if not isinstance(page, dict):
            continue
        for key in ("text_layers", "textos", "texts", "_vision_blocks", "_bubble_regions", "bubble_regions"):
            _strip_raster_mask_fields(page.get(key))
        if isinstance(page.get("metadata"), dict):
            _strip_raster_mask_fields(page["metadata"])

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
    require_owner_project_consistency(project)
    _validate_final_pixel_reports(project)
    stats = project.get("estatisticas") or {}
    if "total_paginas" in stats and int(stats["total_paginas"]) != len(pages):
        raise ValueError(
            "summary mismatch: estatisticas.total_paginas nao bate com paginas"
        )
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
        for key in (
            "actual_pages",
            "processed_pages",
            "translated_regions",
            "qa_flags",
            "critical_flags",
        ):
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
