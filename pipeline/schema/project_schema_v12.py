from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = "12.0"

PROJECT_SCHEMA_V12: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TraduzAi Project Schema v12",
    "type": "object",
    "required": [
        "schema_version",
        "app",
        "run",
        "source",
        "work_context",
        "pages",
        "glossary_hits",
        "entity_flags",
        "qa",
        "export_report",
        "legacy",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "app": {"const": "traduzai"},
        "run": {"type": "object"},
        "source": {"type": "object"},
        "work_context": {"type": "object"},
        "pages": {"type": "array"},
        "glossary_hits": {"type": "array"},
        "entity_flags": {"type": "array"},
        "qa": {"type": "object"},
        "export_report": {"type": "object"},
        "legacy": {"type": "object"},
    },
}


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_empty_project_v12(
    *,
    input_path: str = "",
    page_count: int = 0,
    source_hash: str = "",
    mode: str = "mock",
) -> dict[str, Any]:
    now = iso_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "app": "traduzai",
        "run": {
            "run_id": str(uuid4()),
            "created_at": now,
            "started_at": now,
            "finished_at": None,
            "duration_ms": 0,
            "mode": mode,
            "pipeline_version": SCHEMA_VERSION,
        },
        "source": {
            "input_path": input_path,
            "page_count": page_count,
            "hash": source_hash,
        },
        "work_context": {
            "selected": False,
            "work_id": None,
            "title": None,
            "context_loaded": False,
            "glossary_loaded": False,
            "glossary_entries_count": 0,
            "risk_level": "unknown",
            "user_ignored_warning": False,
        },
        "pages": [],
        "glossary_hits": [],
        "entity_flags": [],
        "qa": {
            "summary": {
                "total_pages": page_count,
                "pages_with_flags": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            },
            "flags": [],
        },
        "export_report": {
            "status": "not_exported",
            "files": [],
        },
        "legacy": {
            "paginas": [],
        },
    }


def build_empty_region_v12(*, page: int, index: int) -> dict[str, Any]:
    return {
        "region_id": f"p{page:03}_r{index:03}",
        "page": page,
        "bbox": [0, 0, 0, 0],
        "polygon": [],
        "group_id": None,
        "reading_order": index - 1,
        "region_type": "unknown",
        "raw_ocr": "",
        "normalized_ocr": "",
        "ocr_confidence": 0.0,
        "normalization": {
            "changed": False,
            "corrections": [],
            "is_gibberish": False,
        },
        "entities": [],
        "term_protection": {
            "protected_text": "",
            "placeholders": [],
        },
        "translation": {
            "text": "",
            "engine": "",
            "confidence": 0.0,
            "used_glossary": [],
            "warnings": [],
        },
        "layout": {
            "font": "",
            "font_size": 0,
            "fit_score": 0.0,
            "overflow": False,
        },
        "mask": {
            "path": None,
            "type": None,
            "bbox": None,
            "valid": False,
        },
        "render_status": "pending",
        "qa_flags": [],
    }


def expected_qa_summary(project: dict[str, Any]) -> dict[str, int]:
    flags = project.get("qa", {}).get("flags", [])
    total_pages = int(project.get("source", {}).get("page_count") or len(project.get("pages", [])))
    summary = {
        "total_pages": total_pages,
        "pages_with_flags": 0,
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }
    pages_with_flags: set[int] = set()
    for flag in flags if isinstance(flags, list) else []:
        if not isinstance(flag, dict):
            continue
        page = flag.get("page")
        if isinstance(page, int):
            pages_with_flags.add(page)
        severity = flag.get("severity", "medium")
        if severity in {"critical", "high", "medium", "low"}:
            summary[severity] += 1
    summary["pages_with_flags"] = len(pages_with_flags)
    return summary


def validate_project_v12(project: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in PROJECT_SCHEMA_V12["required"]:
        if key not in project:
            errors.append(f"missing required key: {key}")

    if project.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if project.get("app") != "traduzai":
        errors.append("app must be traduzai")
    if not isinstance(project.get("pages"), list):
        errors.append("pages must be a list")

    qa = project.get("qa")
    if not isinstance(qa, dict):
        errors.append("qa must be an object")
    else:
        summary = qa.get("summary")
        flags = qa.get("flags")
        if not isinstance(summary, dict):
            errors.append("qa.summary must be an object")
        if not isinstance(flags, list):
            errors.append("qa.flags must be a list")
        if isinstance(summary, dict) and isinstance(flags, list):
            expected = expected_qa_summary(project)
            normalized_summary = {key: int(summary.get(key, 0) or 0) for key in expected}
            if normalized_summary != expected:
                errors.append(f"qa.summary does not match qa.flags: expected {expected}, got {normalized_summary}")

    return errors


def with_recomputed_qa_summary(project: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(project)
    updated.setdefault("qa", {}).setdefault("flags", [])
    updated["qa"]["summary"] = expected_qa_summary(updated)
    return updated

