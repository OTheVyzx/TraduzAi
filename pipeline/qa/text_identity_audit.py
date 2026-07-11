"""Shadow audit for one-owner page-space text identities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schema.visual_scene import build_visual_text_instances


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
    summary = {
        "accepted_without_terminal_lifecycle": accepted_without_terminal,
        "duplicate_owner_layer_ids": duplicate_owner_ids,
        "instances": len(instances),
        "missing_owner": missing_owner,
        "passed": accepted_without_terminal == 0 and missing_owner == 0 and duplicate_owner_ids == 0,
        "schema_version": 1,
    }
    return {"instances": instances, "summary": summary}


def write_text_identity_audit(output_root: Path, project_data: dict[str, Any]) -> Path:
    audit = build_text_identity_audit(project_data)
    target = Path(output_root) / "text_identity_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
