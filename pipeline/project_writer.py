"""Transactional project.json writer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


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
    validate_project_consistency(project)
    if project_json_path.exists():
        backup = project_json_path.with_name(f"project.backup.{int(time.time())}.json")
        backup.write_bytes(project_json_path.read_bytes())
    tmp_path = project_json_path.with_suffix(project_json_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False, indent=2)
    loaded = json.loads(tmp_path.read_text(encoding="utf-8"))
    validate_project_consistency(loaded)
    tmp_path.replace(project_json_path)
