"""Shadow audit for one-owner page-space text identities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schema.visual_scene import build_visual_text_instances


_MULTI_SOURCE_RISK_FLAGS = {
    "TEXT_OVERFLOW",
    "mask_outside_balloon_critical",
    "ocr_geometry_overmerged",
    "ocr_truncated_or_joined",
    "render_on_art_suspected",
    "render_outside_balloon",
    "same_balloon_fragment_merged",
}


def _multi_source_review_required(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_required: list[dict[str, Any]] = []
    for instance in instances:
        source_trace_ids = list(instance.get("source_trace_ids") or [])
        owner_flags = list(instance.get("owner_qa_flags") or [])
        risk_flags = [flag for flag in owner_flags if flag in _MULTI_SOURCE_RISK_FLAGS]
        if len(source_trace_ids) < 2 or not risk_flags:
            continue
        review_required.append(
            {
                "owner_band_id": instance["owner_band_id"],
                "owner_layer_id": instance["owner_layer_id"],
                "risk_flags": risk_flags,
                "source_trace_ids": source_trace_ids,
                "text_instance_id": instance["text_instance_id"],
            }
        )
    return review_required


def build_text_identity_audit(project_data: dict[str, Any]) -> dict[str, Any]:
    instances = build_visual_text_instances(project_data)
    accepted_without_terminal = sum(
        1
        for instance in instances
        if instance["lifecycle"] == "ocr_accepted"
    )
    missing_owner = sum(1 for instance in instances if not instance["owner_layer_id"] or not instance["owner_band_id"])
    owner_keys = {
        (instance["page_id"], instance["owner_band_id"], instance["owner_layer_id"])
        for instance in instances
        if instance["owner_layer_id"] and instance["owner_band_id"]
    }
    duplicate_owner_ids = len(instances) - len(owner_keys)
    review_required = _multi_source_review_required(instances)
    summary = {
        "accepted_without_terminal_lifecycle": accepted_without_terminal,
        "duplicate_owner_layer_ids": duplicate_owner_ids,
        "instances": len(instances),
        "missing_owner": missing_owner,
        "multi_source_review_required": len(review_required),
        "passed": accepted_without_terminal == 0 and missing_owner == 0 and duplicate_owner_ids == 0,
        "schema_version": 1,
    }
    return {"instances": instances, "review_required": review_required, "summary": summary}


def write_text_identity_audit(output_root: Path, project_data: dict[str, Any]) -> Path:
    audit = build_text_identity_audit(project_data)
    target = Path(output_root) / "text_identity_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
