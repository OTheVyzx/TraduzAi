"""Immutable visual-style sidecar bound to a semantic page owner.

The owner graph remains the semantic authority.  This module captures only
source visual evidence and renderer-ready style, before destructive inpaint.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from typesetter.style_contract import (
    StyleEvidenceV2,
    style_evidence_v2_from_dict,
    style_evidence_v2_from_v1,
    style_evidence_v2_sha256,
)
from typesetter.style_policy import (
    OWNER_STYLE_FORBIDDEN_FIELDS,
    decide_style_copy_v2,
    normalize_auto_typesetting_style,
)
from typesetter.style_capture import validate_owner_style_capture


OWNER_VISUAL_PROFILE_SCHEMA_VERSION = 2
OWNER_VISUAL_PROFILE_STATUSES = frozenset(
    {"applied", "fallback", "not_applicable", "review_required"}
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _canonical_bbox(value: object) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("owner visual component requires a page-space bbox")
    bbox = tuple(int(item) for item in value)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("owner visual component bbox must have positive area")
    return bbox


def _source_crop_sha256(
    source_rgb: np.ndarray,
    bboxes: Iterable[tuple[int, int, int, int]],
) -> str:
    source = np.ascontiguousarray(source_rgb, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("owner visual source must be an RGB image")
    boxes = tuple(bboxes)
    if not boxes:
        raise ValueError("owner visual profile requires component geometry")
    x1 = max(0, min(item[0] for item in boxes))
    y1 = max(0, min(item[1] for item in boxes))
    x2 = min(int(source.shape[1]), max(item[2] for item in boxes))
    y2 = min(int(source.shape[0]), max(item[3] for item in boxes))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("owner visual source crop is outside the page")
    crop = np.ascontiguousarray(source[y1:y2, x1:x2, :3])
    header = _canonical_json(
        {"bbox_page": [x1, y1, x2, y2], "shape": list(crop.shape), "dtype": str(crop.dtype)}
    )
    return _sha256_bytes(header + b"\0" + crop.tobytes())


def _mask_sha256(mask: np.ndarray | None, shape: tuple[int, int]) -> str:
    if mask is None:
        canonical = np.zeros(shape, dtype=np.uint8)
    else:
        raw = np.asarray(mask)
        if raw.shape[:2] != shape:
            raise ValueError("owner glyph mask shape differs from source page")
        canonical = np.where(raw > 0, 255, 0).astype(np.uint8)
    return _sha256_bytes(_canonical_json({"shape": list(shape)}) + b"\0" + canonical.tobytes())


def _style_evidence(candidate: Mapping[str, Any], *, text_present: bool) -> StyleEvidenceV2:
    raw_capture = candidate.get("owner_style_capture")
    if isinstance(raw_capture, Mapping):
        capture = validate_owner_style_capture(raw_capture)
        if isinstance(capture.style_evidence_v2, StyleEvidenceV2):
            return capture.style_evidence_v2
        return style_evidence_v2_from_v1({"source": "none"})
    raw_v2 = candidate.get("style_evidence_v2")
    if isinstance(raw_v2, Mapping):
        return style_evidence_v2_from_dict(raw_v2)
    raw_v1 = candidate.get("style_evidence")
    if isinstance(raw_v1, Mapping):
        return style_evidence_v2_from_v1(dict(raw_v1))
    return style_evidence_v2_from_v1(
        {"source": "owner_source_pixels" if text_present else "none"}
    )


def _visual_group_contract(
    owner: object,
    candidate: Mapping[str, Any],
    selected_components: Iterable[object],
) -> tuple[str, str, str]:
    explicit_fields = (
        ("style_group_id", "explicit"),
        ("card_panel_id", "card"),
        ("balloon_id", "balloon"),
        ("burst_panel_id", "burst"),
        ("panel_id", "panel"),
    )
    for field_name, kind in explicit_fields:
        value = str(candidate.get(field_name) or "").strip()
        if value:
            role = str(
                candidate.get("card_panel_role")
                or candidate.get("style_group_role")
                or _field(owner, "semantic_role", "body")
                or "body"
            ).strip()
            return f"{kind}:{value}", kind, role
    evidence_ids = sorted(
        {
            str(evidence_id)
            for component in selected_components
            for evidence_id in (_field(component, "evidence_ids", ()) or ())
            if str(evidence_id)
        }
    )
    owner_id = str(_field(owner, "owner_id") or "")
    role = str(_field(owner, "semantic_role", "body") or "body").strip()
    if evidence_ids:
        return f"container:{evidence_ids[0]}", "container", role
    return f"owner:{owner_id}", "owner", role


def _materialize_style(
    candidate: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    background = candidate.get("background_rgb")
    if not isinstance(background, (list, tuple)) or len(background) < 3:
        background = (255, 255, 255)
    style = normalize_auto_typesetting_style({}, tuple(int(value) for value in background[:3]))
    applied = decision.get("applied_attributes")
    applied = applied if isinstance(applied, Mapping) else {}
    if "fill" in applied:
        style["cor"] = applied["fill"]
    if "font_name" in applied:
        style["fonte"] = applied["font_name"]
    if "font_weight" in applied:
        style["bold"] = str(applied["font_weight"]).strip().lower() in {
            "bold",
            "semibold",
            "black",
            "heavy",
        }
    if "font_width" in applied and "width_scale" not in applied:
        style["width_scale"] = {
            "condensed": 0.82,
            "regular": 1.0,
            "expanded": 1.18,
        }.get(str(applied["font_width"]).strip().lower(), 1.0)
    if "font_size_px" in applied:
        style["tamanho"] = int(round(float(applied["font_size_px"])))
    if "alignment" in applied:
        style["alinhamento"] = str(applied["alignment"])
    stroke = applied.get("stroke")
    if isinstance(stroke, Mapping):
        style["contorno"] = stroke.get("color") or ""
        style["contorno_px"] = int(stroke.get("width_px") or 0)
    shadow = applied.get("shadow")
    if isinstance(shadow, Mapping):
        style.update(
            sombra=True,
            sombra_cor=shadow.get("color") or "#000000",
            sombra_offset=list(shadow.get("offset") or [2, 2])[:2],
        )
    glow = applied.get("glow")
    if isinstance(glow, Mapping):
        style.update(
            glow=True,
            glow_cor=glow.get("color") or style.get("cor") or "#FFFFFF",
            glow_px=int(glow.get("width_px") or 2),
        )
    gradient = applied.get("gradient")
    if isinstance(gradient, (list, tuple)) and len(gradient) >= 2:
        style["cor_gradiente"] = [str(gradient[0]), str(gradient[1])]
        style["cor"] = str(gradient[0])
    curve = applied.get("curve")
    if isinstance(curve, Mapping):
        style.update(
            curva=True,
            curva_direcao=curve.get("direction") or "arc_up",
            curva_intensidade=float(curve.get("amount") or 0.0),
        )
    if "rotation_deg" in applied:
        style["rotacao"] = float(applied["rotation_deg"])
    if "container" in applied:
        style["container"] = copy.deepcopy(applied["container"])
    if "multistroke" in applied:
        style["multistroke"] = copy.deepcopy(applied["multistroke"])
    for geometry_name in (
        "tracking_xh",
        "slant_tangent",
        "width_scale",
        "scale_y",
    ):
        if geometry_name in applied:
            style[geometry_name] = float(applied[geometry_name])
    style["canonical_applied_attributes"] = copy.deepcopy(dict(applied))
    return {
        str(key): copy.deepcopy(value)
        for key, value in style.items()
        if key not in OWNER_STYLE_FORBIDDEN_FIELDS
    }


def owner_visual_profile_sha256(profile: Mapping[str, Any]) -> str:
    """Hash a profile without trusting or recursively hashing its hash field."""

    canonical = {
        str(key): copy.deepcopy(value)
        for key, value in profile.items()
        if key != "visual_profile_sha256"
    }
    return _sha256_bytes(_canonical_json(canonical))


def validate_owner_visual_profile(
    profile: Mapping[str, Any],
    *,
    expected_owner_id: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a detached normalized V2 profile or fail closed."""

    normalized = copy.deepcopy(dict(profile))
    if normalized.get("schema_version") != OWNER_VISUAL_PROFILE_SCHEMA_VERSION:
        raise ValueError("owner renderer requires normalized visual_profile_v2")
    owner_id = str(normalized.get("owner_id") or "")
    if not owner_id or (expected_owner_id is not None and owner_id != expected_owner_id):
        raise ValueError("owner visual profile identity mismatch")
    status = str(normalized.get("status") or "")
    if status not in OWNER_VISUAL_PROFILE_STATUSES:
        raise ValueError("owner visual profile has invalid style status")
    evidence_payload = normalized.get("style_evidence_v2")
    decision = normalized.get("style_application_decision_v2")
    applied_style = normalized.get("applied_style")
    if not isinstance(evidence_payload, Mapping) or not isinstance(decision, Mapping):
        raise ValueError("owner visual profile is missing V2 evidence or decision")
    if not isinstance(applied_style, Mapping):
        raise ValueError("owner visual profile is missing materialized style")
    evidence = style_evidence_v2_from_dict(evidence_payload)
    evidence_hash = style_evidence_v2_sha256(evidence)
    if str(decision.get("evidence_sha256") or "") != evidence_hash:
        raise ValueError("owner visual profile evidence hash mismatch")
    if any(key in OWNER_STYLE_FORBIDDEN_FIELDS for key in applied_style):
        raise ValueError("owner visual profile contains semantic or geometry fields")
    actual_hash = owner_visual_profile_sha256(normalized)
    serialized_hash = str(normalized.get("visual_profile_sha256") or "")
    if serialized_hash != actual_hash or (
        expected_sha256 is not None and expected_sha256 != actual_hash
    ):
        raise ValueError("owner visual profile hash mismatch")
    return normalized


def build_owner_visual_profile(
    owner: object,
    source_rgb: np.ndarray,
    *,
    components: Iterable[object],
    observations: Iterable[object],
    glyph_mask: np.ndarray | None,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture one owner's source visual contract before destructive mutation."""

    owner_id = str(_field(owner, "owner_id") or "")
    if not owner_id:
        raise ValueError("owner visual profile requires owner_id")
    component_ids = {str(value) for value in (_field(owner, "component_ids", []) or [])}
    observation_ids = {
        str(value) for value in (_field(owner, "selected_observation_ids", []) or [])
    }
    selected_components = sorted(
        (
            item for item in components
            if str(_field(item, "component_id") or "") in component_ids
        ),
        key=lambda item: str(_field(item, "component_id") or ""),
    )
    selected_observations = sorted(
        (
            item for item in observations
            if str(_field(item, "observation_id") or "") in observation_ids
        ),
        key=lambda item: str(_field(item, "observation_id") or ""),
    )
    geometry = [
        {
            "component_id": str(_field(item, "component_id")),
            "bbox_page": list(_canonical_bbox(_field(item, "bbox_page"))),
            "polygon_page": [list(point) for point in (_field(item, "polygon_page", ()) or ())],
        }
        for item in selected_components
    ]
    boxes = [tuple(item["bbox_page"]) for item in geometry]
    source = np.ascontiguousarray(source_rgb, dtype=np.uint8)
    glyph_hash = _mask_sha256(glyph_mask, tuple(source.shape[:2]))
    observed_text = any(str(_field(item, "text", "") or "").strip() for item in selected_observations)
    candidate_payload = copy.deepcopy(dict(candidate or {}))
    evidence = _style_evidence(candidate_payload, text_present=observed_text)
    decision = decide_style_copy_v2(candidate_payload, evidence).to_dict()
    group_id, group_kind, group_role = _visual_group_contract(
        owner,
        candidate_payload,
        selected_components,
    )
    profile: dict[str, Any] = {
        "schema_version": OWNER_VISUAL_PROFILE_SCHEMA_VERSION,
        "owner_id": owner_id,
        "page_id": str(_field(owner, "page_id") or ""),
        "style_group_id": group_id,
        "style_group_kind": group_kind,
        "style_group_role": group_role,
        "source_capture_phase": "pre_inpaint",
        "source_sha256": _source_crop_sha256(source, boxes),
        "component_geometry_sha256": _sha256_bytes(_canonical_json(geometry)),
        "glyph_mask_sha256": glyph_hash,
        "style_evidence_v2": evidence.to_dict(),
        "style_application_decision_v2": decision,
        "status": decision["status"],
        "applied_style": _materialize_style(candidate_payload, decision),
    }
    profile["visual_profile_sha256"] = owner_visual_profile_sha256(profile)
    return validate_owner_visual_profile(profile, expected_owner_id=owner_id)


def build_owner_visual_profiles(
    graph: object,
    source_rgb: np.ndarray,
    *,
    glyph_masks_by_owner: Mapping[str, np.ndarray] | None = None,
    candidates_by_owner: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build stable profiles for every owner eligible for rendering."""

    components = list(_field(graph, "components", []) or [])
    observations = list(_field(graph, "observations", []) or [])
    result: dict[str, dict[str, Any]] = {}
    for owner in sorted(
        list(_field(graph, "owners", []) or []),
        key=lambda item: str(_field(item, "owner_id") or ""),
    ):
        if str(_field(owner, "disposition") or "") != "owned":
            continue
        if str(_field(owner, "state") or "") == "review_required":
            continue
        if str(_field(owner, "route_action") or "") not in {
            "translate_inpaint_render", "translate_sfx_inpaint_render"
        }:
            continue
        owner_id = str(_field(owner, "owner_id") or "")
        result[owner_id] = build_owner_visual_profile(
            owner,
            source_rgb,
            components=components,
            observations=observations,
            glyph_mask=(glyph_masks_by_owner or {}).get(owner_id),
            candidate=(candidates_by_owner or {}).get(owner_id),
        )
    return result


def attach_owner_visual_profile(
    record: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach a verified visual profile without mutating its semantic record."""

    output = copy.deepcopy(dict(record))
    envelope = copy.deepcopy(dict(profile))
    nested_profile = envelope.get("visual_profile_v2")
    raw_profile = nested_profile if isinstance(nested_profile, Mapping) else envelope
    owner_id = str(output.get("owner_id") or raw_profile.get("owner_id") or "")
    normalized = validate_owner_visual_profile(raw_profile, expected_owner_id=owner_id)
    existing = output.get("visual_profile_v2")
    if isinstance(existing, Mapping):
        existing_hash = str(output.get("visual_profile_sha256") or "")
        validate_owner_visual_profile(
            existing,
            expected_owner_id=owner_id,
            expected_sha256=existing_hash,
        )
        if existing_hash != normalized["visual_profile_sha256"]:
            immutable_fields = (
                "source_sha256",
                "component_geometry_sha256",
                "glyph_mask_sha256",
                "style_evidence_v2",
            )
            authorized_group_resolution = bool(
                normalized.get("style_group_resolution_v2")
                and all(existing.get(key) == normalized.get(key) for key in immutable_fields)
            )
            if not authorized_group_resolution:
                raise ValueError(f"owner {owner_id} has divergent visual profiles")
    output["visual_profile_v2"] = normalized
    output["visual_profile_sha256"] = normalized["visual_profile_sha256"]
    output["style_copy_status"] = normalized["status"]
    if isinstance(nested_profile, Mapping):
        if envelope.get("visual_profile_sha256") != normalized["visual_profile_sha256"]:
            raise ValueError("owner visual profile envelope hash mismatch")
        for field_name in ("style_group_resolution_v3", "style_resolved_intent_v1"):
            value = envelope.get(field_name)
            if not isinstance(value, Mapping):
                raise ValueError(f"owner visual profile envelope is missing {field_name}")
            output[field_name] = copy.deepcopy(dict(value))
    return output
