"""Produce the Vision-owned payload consumed by Integration's AnalysisRecord."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_FIELDS = (
    "source_sha256",
    "authenticated_neighbor_sha256s",
    "region",
    "coordinate_space",
    "transform_sha256",
    "source_language",
    "analysis_config_sha256",
    "provider_family",
    "provider_name",
    "provider_model",
    "provider_version",
    "capability_version",
)
_REFERENCE_FIELDS = (
    "artifact_refs",
    "transform_ref",
    "ocr_observations",
    "selected_observation_id",
    "selection_provenance",
    "logical_units",
    "physical_subblocks",
    "relations",
    "reading_order",
    "container_contour_ref",
    "writing_body_ref",
    "tail_ref",
    "dependency_hashes",
)
_MASK_NAMES = (
    "glyph",
    "outline",
    "shadow",
    "glow",
    "ignore_or_uncertain",
)


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _require_sha256(value: Any, field: str) -> str:
    digest = str(value or "")
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return digest


def _mask_channel(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    state = str(value.get("state") or "")
    if state == "confirmed_present":
        artifact_ref = str(value.get("artifact_ref") or "")
        if not artifact_ref:
            raise ValueError(f"{name}.artifact_ref is required")
        return {
            "state": state,
            "artifact_ref": artifact_ref.replace("\\", "/"),
            "sha256": _require_sha256(value.get("sha256"), f"{name}.sha256"),
        }
    if state == "confirmed_empty":
        return {
            "state": state,
            "evidence_sha256": _require_sha256(
                value.get("evidence_sha256"), f"{name}.evidence_sha256"
            ),
        }
    if state in {"unknown", "unavailable"}:
        reason = str(value.get("reason_code") or "").strip()
        if not reason:
            raise ValueError(f"{name}.reason_code is required")
        forbidden = {"artifact_ref", "sha256", "evidence_sha256"}.intersection(value)
        if forbidden:
            raise ValueError(f"{name} {state} state cannot alias a mask artifact")
        return {"state": state, "reason_code": reason}
    raise ValueError(f"{name}.state is invalid")


def build_analysis_payload(
    *,
    identity: Mapping[str, Any],
    references: Mapping[str, Any],
    mask_channels: Mapping[str, Mapping[str, Any]] | None = None,
    status: str = "complete",
) -> dict[str, Any]:
    """Build one contract payload without embedding heavy arrays or fake masks."""

    if status not in {"building", "complete", "cancelled", "failed"}:
        raise ValueError("analysis status is invalid")
    missing_identity = [field for field in _IDENTITY_FIELDS if field not in identity]
    missing_references = [field for field in _REFERENCE_FIELDS if field not in references]
    if missing_identity or missing_references:
        raise ValueError("analysis identity/reference fields are incomplete")
    for field in ("source_sha256", "transform_sha256", "analysis_config_sha256"):
        _require_sha256(identity[field], field)
    for digest in identity["authenticated_neighbor_sha256s"]:
        _require_sha256(digest, "authenticated_neighbor_sha256")
    region = identity["region"]
    if not isinstance(region, Mapping) or list(region.get("bbox") or ()) == []:
        raise ValueError("analysis region is required")

    defaults: dict[str, Mapping[str, Any]] = {
        "glyph": {"state": "unknown", "reason_code": "provider_not_run"},
        "outline": {"state": "unknown", "reason_code": "provider_not_run"},
        "shadow": {"state": "unknown", "reason_code": "provider_not_run"},
        "glow": {"state": "unavailable", "reason_code": "capability_unavailable"},
        "ignore_or_uncertain": {
            "state": "unknown",
            "reason_code": "provider_not_run",
        },
    }
    supplied = dict(mask_channels or {})
    unknown_names = set(supplied) - set(_MASK_NAMES)
    if unknown_names:
        raise ValueError("unsupported mask channels: " + ", ".join(sorted(unknown_names)))
    normalized_masks = {
        name: _mask_channel(name, supplied.get(name, defaults[name]))
        for name in _MASK_NAMES
    }
    return {
        "status": status,
        **{field: _copy(identity[field]) for field in _IDENTITY_FIELDS},
        **{field: _copy(references[field]) for field in _REFERENCE_FIELDS},
        "mask_channels": normalized_masks,
    }
