from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ownership.project import (
    OWNER_GRAPH_SCHEMA_VERSION,
    OWNER_GRAPH_STATUS_LEGACY_UNVERIFIED,
    OWNER_GRAPH_STATUS_VERIFIED,
    OWNER_SUMMARY_FIELDS,
    owner_project_validation_errors,
)

SCHEMA_VERSION = "12.0"

_OWNER_ID_SCHEMA: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"^\S(?:.*\S)?$",
}
_OWNER_BBOX_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 4,
    "maxItems": 4,
}
_OWNER_POINT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
}
OWNER_GRAPH_DEFINITIONS: dict[str, Any] = {
    "sourceTextComponent": {
        "type": "object",
        "required": ["component_id", "page_id", "bbox_page"],
        "properties": {
            "component_id": _OWNER_ID_SCHEMA,
            "page_id": _OWNER_ID_SCHEMA,
            "bbox_page": _OWNER_BBOX_SCHEMA,
            "polygon_page": {"type": "array", "items": _OWNER_POINT_SCHEMA},
            "detector_sources": {"type": "array", "items": _OWNER_ID_SCHEMA},
            "confidence": {"type": ["number", "null"]},
            "script_evidence": {"type": "array", "items": _OWNER_ID_SCHEMA},
            "evidence_ids": {"type": "array", "items": _OWNER_ID_SCHEMA},
            "rotation_deg": {"type": ["number", "null"]},
            "rotation_source": {"type": ["string", "null"]},
        },
    },
    "textObservation": {
        "type": "object",
        "required": [
            "observation_id",
            "page_id",
            "component_ids",
            "text",
            "confidence",
            "provider",
            "bbox_page",
            "run_id",
            "origin_execution_id",
            "invocation_id",
            "attempt_id",
            "provider_family",
            "page_source_sha256",
            "root_input_pixel_sha256",
            "input_pixel_sha256",
            "payload_sha256",
        ],
        "properties": {
            "observation_id": _OWNER_ID_SCHEMA,
            "page_id": _OWNER_ID_SCHEMA,
            "component_ids": {"type": "array", "items": _OWNER_ID_SCHEMA},
            "text": {"type": "string"},
            "confidence": {"type": "number"},
            "provider": _OWNER_ID_SCHEMA,
            "bbox_page": _OWNER_BBOX_SCHEMA,
            "run_id": _OWNER_ID_SCHEMA,
            "origin_execution_id": _OWNER_ID_SCHEMA,
            "invocation_id": _OWNER_ID_SCHEMA,
            "attempt_id": _OWNER_ID_SCHEMA,
            "provider_family": _OWNER_ID_SCHEMA,
            "page_source_sha256": _OWNER_ID_SCHEMA,
            "root_input_pixel_sha256": _OWNER_ID_SCHEMA,
            "input_pixel_sha256": _OWNER_ID_SCHEMA,
            "payload_sha256": _OWNER_ID_SCHEMA,
        },
    },
    "textOwner": {
        "type": "object",
        "required": [
            "owner_id",
            "page_id",
            "component_ids",
            "observation_ids",
            "selected_observation_ids",
            "semantic_role",
            "source_payload",
            "translated_payload",
            "disposition",
            "state",
            "route_action",
            "execution_tile_id",
            "action_mask_ref",
        ],
        "properties": {
            "owner_id": _OWNER_ID_SCHEMA,
            "page_id": _OWNER_ID_SCHEMA,
            "component_ids": {"type": "array", "items": _OWNER_ID_SCHEMA},
            "observation_ids": {"type": "array", "items": _OWNER_ID_SCHEMA},
            "selected_observation_ids": {
                "type": "array",
                "items": _OWNER_ID_SCHEMA,
            },
            "semantic_role": _OWNER_ID_SCHEMA,
            "source_payload": {"type": "string"},
            "translated_payload": {"type": ["string", "null"]},
            "disposition": _OWNER_ID_SCHEMA,
            "state": _OWNER_ID_SCHEMA,
            "route_action": _OWNER_ID_SCHEMA,
            "execution_tile_id": {"type": ["string", "null"]},
            "action_mask_ref": {"type": ["string", "null"]},
        },
    },
    "ownerProjection": {
        "type": "object",
        "required": [
            "owner_id",
            "tile_id",
            "role",
            "bbox_page",
            "bbox_tile",
            "offset_xy",
        ],
        "properties": {
            "owner_id": _OWNER_ID_SCHEMA,
            "tile_id": _OWNER_ID_SCHEMA,
            "role": _OWNER_ID_SCHEMA,
            "bbox_page": _OWNER_BBOX_SCHEMA,
            "bbox_tile": _OWNER_BBOX_SCHEMA,
            "offset_xy": _OWNER_POINT_SCHEMA,
        },
    },
    "componentDisposition": {
        "type": "object",
        "required": ["component_id", "decision", "owner_id"],
        "properties": {
            "component_id": _OWNER_ID_SCHEMA,
            "decision": _OWNER_ID_SCHEMA,
            "owner_id": {"type": ["string", "null"]},
            "reason": {"type": ["string", "null"]},
        },
    },
    "ownerViolation": {
        "type": "object",
        "required": ["code", "severity", "message", "offenders"],
        "properties": {
            "code": _OWNER_ID_SCHEMA,
            "severity": _OWNER_ID_SCHEMA,
            "message": {"type": "string"},
            "offenders": {"type": "array", "items": _OWNER_ID_SCHEMA},
        },
    },
}
OWNER_GRAPH_DEFINITIONS["ownerGraph"] = {
    "type": "object",
    "required": [
        "schema_version",
        "page_id",
        "run_id",
        "origin_execution_id",
        "page_source_sha256",
        "verification_status",
        "components",
        "observations",
        "owners",
        "projections",
        "component_dispositions",
        "violations",
    ],
    "properties": {
        "schema_version": {"const": OWNER_GRAPH_SCHEMA_VERSION},
        "page_id": _OWNER_ID_SCHEMA,
        "run_id": _OWNER_ID_SCHEMA,
        "origin_execution_id": _OWNER_ID_SCHEMA,
        "page_source_sha256": _OWNER_ID_SCHEMA,
        "verification_status": {"const": "verified"},
        "components": {
            "type": "array",
            "items": {"$ref": "#/$defs/sourceTextComponent"},
        },
        "observations": {
            "type": "array",
            "items": {"$ref": "#/$defs/textObservation"},
        },
        "owners": {
            "type": "array",
            "items": {"$ref": "#/$defs/textOwner"},
        },
        "projections": {
            "type": "array",
            "items": {"$ref": "#/$defs/ownerProjection"},
        },
        "component_dispositions": {
            "type": "array",
            "items": {"$ref": "#/$defs/componentDisposition"},
        },
        "violations": {
            "type": "array",
            "items": {"$ref": "#/$defs/ownerViolation"},
        },
    },
}

_OWNER_SUMMARY_PROPERTIES: dict[str, Any] = {
    field: {"type": "integer", "minimum": 0} for field in OWNER_SUMMARY_FIELDS
}
OWNER_GRAPH_STATUS_CONDITIONS: list[dict[str, Any]] = [
    {
        "if": {
            "properties": {
                "owner_graph_status": {"const": OWNER_GRAPH_STATUS_LEGACY_UNVERIFIED}
            },
            "required": ["owner_graph_status"],
        },
        "then": {
            "properties": {
                "page_owner_graphs": {"maxItems": 0},
                "owner_invariant_summary": {"maxProperties": 0},
            },
        },
    },
    {
        "if": {
            "properties": {
                "owner_graph_status": {"const": OWNER_GRAPH_STATUS_VERIFIED}
            },
            "required": ["owner_graph_status"],
        },
        "then": {
            "properties": {
                "page_owner_graphs": {"minItems": 1},
                "owner_invariant_summary": {
                    "required": list(OWNER_SUMMARY_FIELDS),
                },
            },
        },
    },
]

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
        "owner_graph_schema_version",
        "owner_graph_status",
        "page_owner_graphs",
        "owner_invariant_summary",
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
        "owner_graph_schema_version": {"const": OWNER_GRAPH_SCHEMA_VERSION},
        "owner_graph_status": {
            "enum": ["verified", OWNER_GRAPH_STATUS_LEGACY_UNVERIFIED]
        },
        "page_owner_graphs": {
            "type": "array",
            "items": {"$ref": "#/$defs/ownerGraph"},
        },
        "owner_invariant_summary": {
            "type": "object",
            "properties": _OWNER_SUMMARY_PROPERTIES,
            "additionalProperties": False,
        },
    },
    "$defs": OWNER_GRAPH_DEFINITIONS,
    "allOf": OWNER_GRAPH_STATUS_CONDITIONS,
}


def iso_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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
        "owner_graph_schema_version": OWNER_GRAPH_SCHEMA_VERSION,
        "owner_graph_status": OWNER_GRAPH_STATUS_LEGACY_UNVERIFIED,
        "page_owner_graphs": [],
        "owner_invariant_summary": {},
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
    total_pages = int(
        project.get("source", {}).get("page_count") or len(project.get("pages", []))
    )
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

    errors.extend(owner_project_validation_errors(project, require_envelope=True))

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
            normalized_summary = {
                key: int(summary.get(key, 0) or 0) for key in expected
            }
            if normalized_summary != expected:
                errors.append(
                    f"qa.summary does not match qa.flags: expected {expected}, got {normalized_summary}"
                )

    return errors


def with_recomputed_qa_summary(project: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(project)
    updated.setdefault("qa", {}).setdefault("flags", [])
    updated["qa"]["summary"] = expected_qa_summary(updated)
    return updated
