"""Serializable page-global ownership model and fail-closed invariants."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from hashlib import sha256
import json
import math
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from .coverage import CANONICAL_COVERAGE_STATES

try:
    from typesetter.owner_render_quality import OwnerRenderQuality
    from typesetter.style_contract import STYLE_V2_ATTRIBUTE_NAME_SET
except ImportError:  # pragma: no cover - supports package imports
    from ..typesetter.owner_render_quality import OwnerRenderQuality
    from ..typesetter.style_contract import STYLE_V2_ATTRIBUTE_NAME_SET


BBox = tuple[int, int, int, int]
Point = tuple[int, int]

OWNER_GRAPH_SCHEMA_VERSION = 2
OWNER_GRAPH_LEGACY_SCHEMA_VERSION = 1

FINAL_COMPONENT_DECISIONS = frozenset({"owned", "preserve", "suppress", "review"})
OWNER_DISPOSITIONS = frozenset({"owned", "review"})
LEGACY_OWNER_STATES = frozenset(
    {
        "ocr_ready",
        "execution_planned",
        "translated",
        "mask_ready",
        "inpainted",
        "laid_out",
        "verified",
        "review_required",
    }
)
OWNER_STATES = CANONICAL_COVERAGE_STATES | LEGACY_OWNER_STATES
OWNER_ROUTE_ACTIONS = frozenset(
    {
        "translate_inpaint_render",
        "translate_sfx_inpaint_render",
        "translate_render_only",
        "inpaint_only",
        "review_required",
    }
)
TRANSLATION_ROUTE_ACTIONS = frozenset(
    {
        "translate_inpaint_render",
        "translate_sfx_inpaint_render",
        "translate_render_only",
    }
)
INPAINT_ROUTE_ACTIONS = frozenset(
    {
        "translate_inpaint_render",
        "translate_sfx_inpaint_render",
        "inpaint_only",
    }
)
POST_TRANSLATION_STATES = frozenset(
    {
        "translated",
        "mask_ready",
        "inpainted",
        "laid_out",
        "verified",
        "target_ready",
        "execution_attempt",
        "repair_pending",
        "cleaned",
        "rendered",
        "final_verified",
    }
)
MASK_REQUIRED_STATES = frozenset(
    {
        "mask_ready",
        "inpainted",
        "laid_out",
        "verified",
        "execution_attempt",
        "repair_pending",
        "cleaned",
        "rendered",
        "final_verified",
    }
)
EXECUTOR_REQUIRED_STATES = POST_TRANSLATION_STATES | frozenset({"execution_planned"})
EXECUTION_ROUTE_ACTIONS = frozenset(
    {
        "translate_inpaint_render",
        "translate_sfx_inpaint_render",
        "translate_render_only",
        "inpaint_only",
    }
)


def _immutable_array_sha256(value: Any) -> str:
    dtype = getattr(getattr(value, "dtype", None), "str", None)
    shape = getattr(value, "shape", None)
    tobytes = getattr(value, "tobytes", None)
    if not isinstance(dtype, str) or shape is None or not callable(tobytes):
        raise TypeError("owner mutation array evidence must be hashable")
    digest = sha256()
    digest.update(b"traduzai.ndarray.v1\0")
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(dimension) for dimension in shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(tobytes(order="C"))
    return digest.hexdigest()


def _deep_frozen_array_copy(value: Any) -> Any:
    """Copy ndarray evidence onto an immutable bytes-backed buffer."""

    if not isinstance(value, np.ndarray):
        copy_value = getattr(value, "copy", None)
        return copy_value() if callable(copy_value) else value
    array = np.ascontiguousarray(value)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    frozen.setflags(write=False)
    return frozen


ACTIVE_OWNER_STATES = OWNER_STATES - frozenset(
    {"review_required", "explicit_non_dialogue_preserve"}
)
ROUTE_ALLOWED_STATES = {
    "translate_inpaint_render": ACTIVE_OWNER_STATES,
    "translate_sfx_inpaint_render": ACTIVE_OWNER_STATES,
    "translate_render_only": ACTIVE_OWNER_STATES
    - frozenset({"mask_ready", "inpainted"}),
    "inpaint_only": frozenset(
        {
            "discovered",
            "challenged",
            "observed",
            "owned",
            "target_ready",
            "execution_attempt",
            "repair_pending",
            "cleaned",
            "rendered",
            "final_verified",
            "ocr_ready",
            "execution_planned",
            "mask_ready",
            "inpainted",
            "verified",
        }
    ),
    "review_required": frozenset({"review_required"}),
}


def _is_canonical_page_bbox(value: Any) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and min(value) >= 0
        and value[0] < value[2]
        and value[1] < value[3]
    )


def _duplicate_values(values: Iterable[Any]) -> tuple[str, ...]:
    ordered = list(values)
    duplicates: list[str] = []
    for index, value in enumerate(ordered):
        if any(value == previous for previous in ordered[:index]):
            rendered = str(value)
            if rendered not in duplicates:
                duplicates.append(rendered)
    return tuple(duplicates)


def _action_mask_ref_matches_owner(owner_id: str, action_mask_ref: str) -> bool:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner_id).strip("._")
    if not safe:
        return False
    identity_hash = sha256(owner_id.encode("utf-8")).hexdigest()[:12]
    expected_segment = f"{safe}--{identity_hash}"
    parts = PurePosixPath(action_mask_ref.replace("\\", "/")).parts
    return bool(
        len(parts) == 4
        and parts[0] == "owner_masks"
        and parts[1] == expected_segment
        and parts[2] not in {"", ".", ".."}
        and parts[3] == "action_mask.png"
    )


@dataclass(frozen=True)
class SourceTextComponent:
    """Visual evidence that text exists, independent of any OCR payload."""

    component_id: str
    page_id: str
    bbox_page: BBox
    polygon_page: tuple[Point, ...]
    detector_sources: tuple[str, ...]
    confidence: float | None = None
    script_evidence: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    rotation_deg: float | None = None
    rotation_source: str | None = None


@dataclass(frozen=True)
class TextObservation:
    """One immutable OCR reading candidate in page coordinates."""

    observation_id: str
    page_id: str
    component_ids: tuple[str, ...]
    text: str
    confidence: float
    provider: str
    bbox_page: BBox
    polygons_page: tuple[tuple[Point, ...], ...] = ()
    tile_provenance: tuple[str, ...] = ()
    coverage_score: float | None = None
    language_score: float | None = None
    rejection_reason: str | None = None
    legacy_rejection_reason: str | None = None
    legacy_selected: bool = False
    provider_variant: str = ""
    provider_record_id: str | None = None
    projection_ids: tuple[str, ...] = ()
    raw_text: str | None = None
    source_bbox_page: BBox | None = None
    text_pixel_bbox_page: BBox | None = None
    layout_bbox_page: BBox | None = None
    line_texts: tuple[str, ...] = ()
    rotation_deg: float | None = None
    rotation_source: str | None = None
    _: KW_ONLY
    run_id: str = ""
    origin_execution_id: str = ""
    invocation_id: str = ""
    attempt_id: str = ""
    provider_family: str = ""
    page_source_sha256: str = ""
    root_input_pixel_sha256: str = ""
    input_pixel_sha256: str = ""
    payload_sha256: str = ""

    @property
    def identity_complete(self) -> bool:
        return all(
            str(value or "").strip()
            for value in (
                self.run_id,
                self.origin_execution_id,
                self.page_id,
                self.invocation_id,
                self.attempt_id,
                self.provider_family,
                self.page_source_sha256,
                self.root_input_pixel_sha256,
                self.input_pixel_sha256,
                self.payload_sha256,
            )
        )


@dataclass(frozen=True)
class ComponentDisposition:
    """Final, auditable decision for one source component."""

    component_id: str
    decision: str
    owner_id: str | None = None
    reason: str | None = None


@dataclass
class TextOwner:
    """Single semantic payload authorized to move through the visual pipeline."""

    owner_id: str
    page_id: str
    component_ids: list[str]
    observation_ids: list[str]
    selected_observation_ids: list[str]
    semantic_role: str
    source_payload: str
    translated_payload: str | None
    disposition: str
    state: str
    route_action: str
    execution_tile_id: str | None
    action_mask_ref: str | None = None


@dataclass(frozen=True)
class OwnerProjection:
    """Owner geometry projected into a tile without changing its identity."""

    owner_id: str
    tile_id: str
    role: str
    bbox_page: BBox
    bbox_tile: BBox
    offset_xy: Point


@dataclass(frozen=True)
class OwnerMutation:
    """One owner-scoped cleanup result with authoritative pixel evidence."""

    owner_id: str
    page_id: str
    coordinate_space: str
    action_mask_ref: str
    result_rgb: Any
    action_mask: Any
    protected_art_mask: Any
    changed_mask: Any
    engine: str
    mask_pixels: int
    changed_pixels: int
    changed_outside_owner_pixels: int
    protected_art_changed_pixels: int
    before_sha256: str
    after_sha256: str
    action_mask_sha256: str
    changed_mask_sha256: str
    engine_crop_bbox_page: BBox
    owner_bbox_page: BBox
    component_geometry_sha256: str
    owner_render_geometry_sha256: str = ""
    protected_art_mask_sha256: str | None = None
    residual_score: float | None = None
    residual_verified: bool = False
    residual_threshold: float | None = None
    residual_method: str | None = None
    residual_evidence_sha256: str | None = None
    residual_flags: tuple[str, ...] = ()
    changed_mask_ref: str | None = None
    execution_tile_id: str | None = None
    projection_role: str = "executor"
    color_space: str = "RGB"
    component_geometry_verified: bool = False
    text_execution_authority_sha256: str = ""
    text_execution_authority: Any = None

    def __post_init__(self) -> None:
        for field_name in (
            "result_rgb",
            "action_mask",
            "protected_art_mask",
            "changed_mask",
        ):
            value = getattr(self, field_name)
            copy_value = getattr(value, "copy", None)
            if not callable(copy_value):
                continue
            frozen_value = copy_value()
            setflags = getattr(frozen_value, "setflags", None)
            if callable(setflags):
                setflags(write=False)
            object.__setattr__(self, field_name, frozen_value)
        if self.protected_art_mask_sha256 is None:
            object.__setattr__(
                self,
                "protected_art_mask_sha256",
                _immutable_array_sha256(self.protected_art_mask),
            )
        object.__setattr__(
            self,
            "residual_flags",
            tuple(sorted(set(str(flag) for flag in self.residual_flags))),
        )

    @property
    def changed_outside_action_mask_pixels(self) -> int:
        return self.changed_outside_owner_pixels

    @property
    def original_sha256(self) -> str:
        return self.before_sha256

    @property
    def result_sha256(self) -> str:
        return self.after_sha256


def owner_residual_evidence_sha256(
    *,
    owner_id: str,
    page_id: str,
    before_sha256: str,
    after_sha256: str,
    action_mask_sha256: str,
    protected_art_mask_sha256: str,
    component_geometry_sha256: str,
    residual_score: float,
    residual_threshold: float,
    residual_method: str,
    residual_flags: tuple[str, ...] | list[str],
) -> str:
    """Hash the complete, immutable residual verification contract."""

    payload = {
        "schema": "traduzai.owner-residual-evidence.v1",
        "owner_id": owner_id,
        "page_id": page_id,
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "action_mask_sha256": action_mask_sha256,
        "protected_art_mask_sha256": protected_art_mask_sha256,
        "component_geometry_sha256": component_geometry_sha256,
        "residual_score": residual_score,
        "residual_threshold": residual_threshold,
        "residual_method": residual_method,
        "residual_flags": sorted(set(str(flag) for flag in residual_flags)),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OwnerStyleRasterContract:
    """Immutable evidence binding one owner style decision to raster pixels."""

    schema_version: int
    page_id: str
    owner_id: str
    visual_profile_sha256: str
    profile_component_geometry_sha256: str
    execution_component_geometry_sha256: str
    source_artifact_sha256: str
    source_glyph_mask_sha256: str
    status: str
    backend: str
    backend_version: str
    capabilities: tuple[str, ...]
    requested_attributes: Mapping[str, Any]
    applied_attributes: Mapping[str, Any]
    abstained_attributes: Mapping[str, Any]
    glyph_core_envelope: Mapping[str, Any]
    effect_envelope: Mapping[str, Any]
    render_metrics: Mapping[str, Any]
    segments: tuple[Mapping[str, Any], ...]
    rendered_before_sha256: str
    rendered_patch_sha256: str
    rendered_after_sha256: str
    contract_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            tuple(sorted(set(str(value) for value in self.capabilities))),
        )
        for field_name in (
            "requested_attributes",
            "applied_attributes",
            "abstained_attributes",
            "glyph_core_envelope",
            "effect_envelope",
            "render_metrics",
        ):
            object.__setattr__(
                self,
                field_name,
                _freeze_owner_style_json(getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "segments",
            tuple(_freeze_owner_style_json(segment) for segment in self.segments),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "page_id": self.page_id,
            "owner_id": self.owner_id,
            "visual_profile_sha256": self.visual_profile_sha256,
            "profile_component_geometry_sha256": (
                self.profile_component_geometry_sha256
            ),
            "execution_component_geometry_sha256": (
                self.execution_component_geometry_sha256
            ),
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_glyph_mask_sha256": self.source_glyph_mask_sha256,
            "status": self.status,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "capabilities": list(self.capabilities),
            "requested_attributes": _thaw_owner_style_json(
                self.requested_attributes
            ),
            "applied_attributes": _thaw_owner_style_json(self.applied_attributes),
            "abstained_attributes": _thaw_owner_style_json(
                self.abstained_attributes
            ),
            "glyph_core_envelope": _thaw_owner_style_json(
                self.glyph_core_envelope
            ),
            "effect_envelope": _thaw_owner_style_json(self.effect_envelope),
            "render_metrics": _thaw_owner_style_json(self.render_metrics),
            "segments": [
                _thaw_owner_style_json(segment) for segment in self.segments
            ],
            "rendered_before_sha256": self.rendered_before_sha256,
            "rendered_patch_sha256": self.rendered_patch_sha256,
            "rendered_after_sha256": self.rendered_after_sha256,
            "contract_sha256": self.contract_sha256,
        }


def _freeze_owner_style_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_owner_style_json(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_owner_style_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("owner style raster contract contains non-finite value")
        return value
    raise TypeError(
        f"owner style raster contract contains non-JSON value: {type(value).__name__}"
    )


def _thaw_owner_style_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_owner_style_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_thaw_owner_style_json(item) for item in value]
    return value


@dataclass(frozen=True)
class OwnerStyleRasterContractV2(OwnerStyleRasterContract):
    """Materialization-aware raster contract used during the V2 migration."""

    render_status: str
    materialization_status: str
    style_intent_sha256: str
    materialization_plan_sha256: str
    materialization_observation_sha256: str
    materialization_plan: Mapping[str, Any]
    materialization_observation: Mapping[str, Any]
    materialization_comparison: Mapping[str, Any]
    backend_selection_reason: str

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in (
            "materialization_plan",
            "materialization_observation",
            "materialization_comparison",
        ):
            object.__setattr__(
                self,
                field_name,
                _freeze_owner_style_json(getattr(self, field_name)),
            )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "render_status": self.render_status,
                "materialization_status": self.materialization_status,
                "style_intent_sha256": self.style_intent_sha256,
                "materialization_plan_sha256": self.materialization_plan_sha256,
                "materialization_observation_sha256": (
                    self.materialization_observation_sha256
                ),
                "materialization_plan": _thaw_owner_style_json(
                    self.materialization_plan
                ),
                "materialization_observation": _thaw_owner_style_json(
                    self.materialization_observation
                ),
                "materialization_comparison": _thaw_owner_style_json(
                    self.materialization_comparison
                ),
                "backend_selection_reason": self.backend_selection_reason,
            }
        )
        return payload


OWNER_STYLE_RASTER_CONTRACT_SCHEMA_VERSION = 1
OWNER_STYLE_RASTER_CONTRACT_STATUSES = frozenset(
    {"applied", "fallback", "review_required"}
)
OWNER_STYLE_RASTER_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "page_id",
        "owner_id",
        "visual_profile_sha256",
        "profile_component_geometry_sha256",
        "execution_component_geometry_sha256",
        "source_artifact_sha256",
        "source_glyph_mask_sha256",
        "status",
        "backend",
        "backend_version",
        "capabilities",
        "requested_attributes",
        "applied_attributes",
        "abstained_attributes",
        "glyph_core_envelope",
        "effect_envelope",
        "render_metrics",
        "segments",
        "rendered_before_sha256",
        "rendered_patch_sha256",
        "rendered_after_sha256",
        "contract_sha256",
    }
)
OWNER_STYLE_RASTER_CONTRACT_V2_FIELDS = OWNER_STYLE_RASTER_CONTRACT_FIELDS | frozenset(
    {
        "render_status",
        "materialization_status",
        "style_intent_sha256",
        "materialization_plan_sha256",
        "materialization_observation_sha256",
        "materialization_plan",
        "materialization_observation",
        "materialization_comparison",
        "backend_selection_reason",
    }
)
OWNER_STYLE_RASTER_HASH_FIELDS = (
    "visual_profile_sha256",
    "profile_component_geometry_sha256",
    "execution_component_geometry_sha256",
    "source_artifact_sha256",
    "source_glyph_mask_sha256",
    "rendered_before_sha256",
    "rendered_patch_sha256",
    "rendered_after_sha256",
)
_OWNER_STYLE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OWNER_STYLE_RASTER_SEGMENT_FIELDS = frozenset(
    {
        "segment_id",
        "order",
        "owner_id",
        "visual_profile_sha256",
        "bbox_page",
        "status",
        "applied_attributes",
        "abstained_attributes",
        "glyph_core_envelope",
        "effect_envelope",
        "rendered_before_sha256",
        "rendered_patch_sha256",
        "rendered_after_sha256",
        "segment_sha256",
    }
)


def owner_style_raster_contract_sha256(
    value: OwnerStyleRasterContract | OwnerStyleRasterContractV2 | Mapping[str, Any],
) -> str:
    """Return the canonical self-hash for an owner style raster contract."""

    payload = (
        value.to_dict()
        if isinstance(value, OwnerStyleRasterContract)
        else _thaw_owner_style_json(value)
    )
    if not isinstance(payload, dict):
        raise TypeError("owner style raster contract must be a mapping")
    payload.pop("contract_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def owner_style_raster_segment_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical self-hash for one child raster segment."""

    payload = _thaw_owner_style_json(value)
    if not isinstance(payload, dict):
        raise TypeError("owner style raster segment must be a mapping")
    payload.pop("segment_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_owner_style_raster_segment(
    value: Mapping[str, Any],
    *,
    expected_owner_id: str,
    expected_visual_profile_sha256: str,
) -> dict[str, Any]:
    """Validate one child segment before it can enter a parent contract."""

    payload = _thaw_owner_style_json(value)
    if not isinstance(payload, dict):
        raise ValueError("child raster contract must be a mapping")
    fields = set(payload)
    if fields != OWNER_STYLE_RASTER_SEGMENT_FIELDS:
        missing = sorted(OWNER_STYLE_RASTER_SEGMENT_FIELDS - fields)
        unknown = sorted(fields - OWNER_STYLE_RASTER_SEGMENT_FIELDS)
        details = ", ".join(missing or unknown)
        raise ValueError(f"child raster contract fields are invalid: {details}")
    segment_id = str(payload.get("segment_id") or "").strip()
    if not segment_id:
        raise ValueError("child raster contract segment_id is missing")
    order = payload.get("order")
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
        raise ValueError("child raster contract order is invalid")
    if payload.get("owner_id") != expected_owner_id:
        raise ValueError("child raster contract owner mismatch")
    if payload.get("visual_profile_sha256") != expected_visual_profile_sha256:
        raise ValueError("child raster contract visual profile mismatch")
    bbox = payload.get("bbox_page")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in bbox)
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
    ):
        raise ValueError("child raster contract bbox_page is invalid")
    status = str(payload.get("status") or "")
    if status not in OWNER_STYLE_RASTER_CONTRACT_STATUSES:
        raise ValueError("child raster contract status is unsupported")
    for field_name in ("applied_attributes", "abstained_attributes"):
        attributes = payload.get(field_name)
        if not isinstance(attributes, dict):
            raise ValueError(f"child raster contract {field_name} must be a mapping")
        unsupported = sorted(set(attributes) - STYLE_V2_ATTRIBUTE_NAME_SET)
        if unsupported:
            raise ValueError("child raster contract has unsupported style attribute")
    if set(payload["applied_attributes"]) & set(payload["abstained_attributes"]):
        raise ValueError("child raster contract attribute resolution overlaps")
    if status != "applied" and payload["applied_attributes"]:
        raise ValueError("non-applied child raster contract claims applied attributes")
    for field_name in ("glyph_core_envelope", "effect_envelope"):
        if not isinstance(payload.get(field_name), dict):
            raise ValueError(f"child raster contract {field_name} must be a mapping")
    for field_name in (
        "visual_profile_sha256",
        "rendered_before_sha256",
        "rendered_patch_sha256",
        "rendered_after_sha256",
        "segment_sha256",
    ):
        if not isinstance(payload.get(field_name), str) or not _OWNER_STYLE_SHA256_RE.fullmatch(
            payload[field_name]
        ):
            raise ValueError(f"child raster contract malformed sha256: {field_name}")
    if payload["segment_sha256"] != owner_style_raster_segment_sha256(payload):
        raise ValueError("child raster contract hash mismatch")
    return payload


def _validate_owner_style_raster_contract_v1(
    value: OwnerStyleRasterContract | Mapping[str, Any],
    *,
    expected_owner_id: str | None = None,
    expected_page_id: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize one owner style raster contract."""

    payload = (
        value.to_dict()
        if isinstance(value, OwnerStyleRasterContract)
        else _thaw_owner_style_json(value)
    )
    if not isinstance(payload, dict):
        raise TypeError("owner style raster contract must be a mapping")
    payload_fields = set(payload)
    missing_fields = sorted(OWNER_STYLE_RASTER_CONTRACT_FIELDS - payload_fields)
    if missing_fields:
        raise ValueError(
            "owner style raster contract missing required fields: "
            + ", ".join(missing_fields)
        )
    unknown_fields = sorted(payload_fields - OWNER_STYLE_RASTER_CONTRACT_FIELDS)
    if unknown_fields:
        raise ValueError(
            "owner style raster contract contains unknown fields: "
            + ", ".join(unknown_fields)
        )
    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != OWNER_STYLE_RASTER_CONTRACT_SCHEMA_VERSION
    ):
        raise ValueError("owner style raster contract schema_version is unsupported")
    owner_id = str(payload.get("owner_id") or "").strip()
    page_id = str(payload.get("page_id") or "").strip()
    if not owner_id or not page_id:
        raise ValueError("owner style raster contract identity is missing")
    if expected_owner_id is not None and owner_id != expected_owner_id:
        raise ValueError("owner style raster contract owner identity mismatch")
    if expected_page_id is not None and page_id != expected_page_id:
        raise ValueError("owner style raster contract page identity mismatch")
    for field_name in OWNER_STYLE_RASTER_HASH_FIELDS:
        field_value = payload.get(field_name)
        if not isinstance(field_value, str) or not _OWNER_STYLE_SHA256_RE.fullmatch(
            field_value
        ):
            raise ValueError(
                f"owner style raster contract malformed sha256: {field_name}"
            )
    status = str(payload.get("status") or "").strip()
    if status not in OWNER_STYLE_RASTER_CONTRACT_STATUSES:
        raise ValueError("owner style raster contract status is unsupported")
    for field_name in ("backend", "backend_version"):
        if not isinstance(payload.get(field_name), str) or not str(
            payload[field_name]
        ).strip():
            raise ValueError(
                f"owner style raster contract {field_name} is missing"
            )
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list) or capabilities != sorted(
        set(str(value) for value in capabilities)
    ):
        raise ValueError("owner style raster contract capabilities are not canonical")
    for field_name in (
        "requested_attributes",
        "applied_attributes",
        "abstained_attributes",
    ):
        attributes = payload.get(field_name)
        if not isinstance(attributes, dict):
            raise ValueError(
                f"owner style raster contract {field_name} must be a mapping"
            )
        unsupported = sorted(set(attributes) - STYLE_V2_ATTRIBUTE_NAME_SET)
        if unsupported:
            raise ValueError(
                "owner style raster contract unsupported style attribute: "
                + ", ".join(unsupported)
            )
    requested_names = set(payload["requested_attributes"])
    applied_names = set(payload["applied_attributes"])
    abstained_names = set(payload["abstained_attributes"])
    if applied_names & abstained_names:
        raise ValueError(
            "owner style raster contract attribute cannot be applied and abstained"
        )
    if requested_names != applied_names | abstained_names:
        raise ValueError(
            "owner style raster contract requested attributes are not fully resolved"
        )
    if status == "applied" and not applied_names:
        raise ValueError("applied owner style raster contract has no applied attributes")
    if status != "applied" and applied_names:
        raise ValueError(
            "non-applied owner style raster contract contains applied attributes"
        )
    for field_name in (
        "glyph_core_envelope",
        "effect_envelope",
        "render_metrics",
    ):
        if not isinstance(payload.get(field_name), dict):
            raise ValueError(
                f"owner style raster contract {field_name} must be a mapping"
            )
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("owner style raster contract segments must be a list")
    normalized_segments: list[dict[str, Any]] = []
    segment_ids: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("owner style raster contract segment must be a mapping")
        normalized_segment = validate_owner_style_raster_segment(
            segment,
            expected_owner_id=owner_id,
            expected_visual_profile_sha256=payload["visual_profile_sha256"],
        )
        normalized_segments.append(normalized_segment)
        segment_ids.append(normalized_segment["segment_id"])
    duplicate_segment_ids = _duplicate_values(segment_ids)
    if duplicate_segment_ids:
        raise ValueError(
            "owner style raster contract duplicate segment_id: "
            + ", ".join(duplicate_segment_ids)
        )
    canonical_segments = sorted(
        normalized_segments,
        key=lambda item: (item["order"], item["segment_id"]),
    )
    if normalized_segments != canonical_segments:
        raise ValueError("owner style raster contract segments are not canonical")
    for index, left in enumerate(normalized_segments):
        lx1, ly1, lx2, ly2 = left["bbox_page"]
        for right in normalized_segments[index + 1 :]:
            rx1, ry1, rx2, ry2 = right["bbox_page"]
            if min(lx2, rx2) > max(lx1, rx1) and min(ly2, ry2) > max(ly1, ry1):
                raise ValueError("owner style raster contract segments overlap")
    actual_hash = owner_style_raster_contract_sha256(payload)
    if str(payload.get("contract_sha256") or "") != actual_hash:
        raise ValueError("owner style raster contract hash mismatch")
    return payload


def _v2_legacy_projections(
    materialization_plan: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    attribute_plans = materialization_plan.get("attribute_plans")
    if not isinstance(attribute_plans, Mapping):
        raise ValueError("materialization plan attribute projections are missing")
    requested = {
        str(name): _thaw_owner_style_json(row.get("intent_value"))
        for name, row in sorted(attribute_plans.items())
        if isinstance(row, Mapping)
    }
    status = str(comparison.get("status") or "")
    applied: dict[str, Any] = {}
    abstained: dict[str, Any] = {}
    mismatch_reasons = {
        str(row.get("attribute")): str(row.get("reason") or "materialization_mismatch")
        for row in comparison.get("mismatches") or []
        if isinstance(row, Mapping) and row.get("attribute")
    }
    for name, raw in sorted(attribute_plans.items()):
        if not isinstance(raw, Mapping):
            raise ValueError("materialization plan attribute projection is invalid")
        kind = str(raw.get("resolution_kind") or "")
        if status == "match" and kind in {"exact", "policy_adjusted", "derived"}:
            applied[str(name)] = _thaw_owner_style_json(raw.get("target_value"))
        elif kind == "abstained":
            abstained[str(name)] = str(raw.get("reason") or "approved_abstention")
        elif kind == "superseded":
            winner = str(raw.get("superseded_by") or "")
            abstained[str(name)] = f"superseded_by:{winner}"
        else:
            abstained[str(name)] = mismatch_reasons.get(
                str(name), "materialization_not_match"
            )
    return requested, applied, abstained


def _validate_owner_style_raster_contract_v2(
    value: OwnerStyleRasterContractV2 | Mapping[str, Any],
    *,
    expected_owner_id: str | None = None,
    expected_page_id: str | None = None,
) -> dict[str, Any]:
    from typesetter.style_materialization import (
        compare_materialization_payloads,
        validate_materialization_observation,
        validate_materialization_plan,
    )

    payload = value.to_dict() if isinstance(value, OwnerStyleRasterContractV2) else _thaw_owner_style_json(value)
    if not isinstance(payload, dict):
        raise TypeError("owner style raster contract v2 must be a mapping")
    if set(payload) != OWNER_STYLE_RASTER_CONTRACT_V2_FIELDS:
        raise ValueError("owner style raster contract v2 materialization fields are invalid")
    if payload.get("schema_version") != 2:
        raise ValueError("owner style raster contract v2 schema_version is unsupported")
    owner_id = str(payload.get("owner_id") or "").strip()
    page_id = str(payload.get("page_id") or "").strip()
    if not owner_id or not page_id:
        raise ValueError("owner style raster contract v2 identity is missing")
    if expected_owner_id is not None and owner_id != expected_owner_id:
        raise ValueError("owner style raster contract owner identity mismatch")
    if expected_page_id is not None and page_id != expected_page_id:
        raise ValueError("owner style raster contract page identity mismatch")
    for field_name in (
        *OWNER_STYLE_RASTER_HASH_FIELDS,
        "style_intent_sha256",
        "materialization_plan_sha256",
        "materialization_observation_sha256",
    ):
        value_sha = payload.get(field_name)
        if not isinstance(value_sha, str) or not _OWNER_STYLE_SHA256_RE.fullmatch(value_sha):
            raise ValueError(f"owner style raster contract malformed sha256: {field_name}")
    if payload.get("render_status") not in {"completed", "failed"}:
        raise ValueError("owner style raster contract v2 render_status is invalid")
    if payload.get("materialization_status") not in {"match", "mismatch", "review_required"}:
        raise ValueError("owner style raster contract v2 materialization_status is invalid")
    if not isinstance(payload.get("backend_selection_reason"), str) or not payload["backend_selection_reason"].strip():
        raise ValueError("owner style raster contract v2 backend selection reason is missing")
    plan = validate_materialization_plan(payload.get("materialization_plan") or {})
    observation = validate_materialization_observation(
        payload.get("materialization_observation") or {}
    )
    if plan["owner_id"] != owner_id or plan["page_id"] != page_id:
        raise ValueError("materialization plan owner/page binding mismatch")
    if plan["visual_profile_sha256"] != payload["visual_profile_sha256"]:
        raise ValueError("materialization plan visual profile binding mismatch")
    if plan["intent_sha256"] != payload["style_intent_sha256"]:
        raise ValueError("materialization intent binding mismatch")
    if plan["plan_sha256"] != payload["materialization_plan_sha256"]:
        raise ValueError("materialization plan hash binding mismatch")
    if observation["owner_id"] != owner_id or observation["page_id"] != page_id:
        raise ValueError("materialization observation owner/page binding mismatch")
    if observation["visual_profile_sha256"] != payload["visual_profile_sha256"]:
        raise ValueError("materialization observation visual profile binding mismatch")
    if observation["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("materialization observation plan binding mismatch")
    if observation["observation_sha256"] != payload["materialization_observation_sha256"]:
        raise ValueError("materialization observation hash binding mismatch")
    recomputed = compare_materialization_payloads(plan, observation).to_dict()
    serialized_comparison = payload.get("materialization_comparison")
    if not isinstance(serialized_comparison, dict) or serialized_comparison != recomputed:
        raise ValueError("materialization comparison does not match recomputed evidence")
    if payload["materialization_status"] != recomputed["status"]:
        raise ValueError("materialization status does not match comparison")
    expected_status = (
        "applied"
        if payload["render_status"] == "completed" and recomputed["status"] == "match"
        else "review_required"
    )
    if payload.get("status") != expected_status:
        raise ValueError("materialization legacy status projection mismatch")
    requested, applied, abstained = _v2_legacy_projections(plan, recomputed)
    if payload.get("requested_attributes") != requested:
        raise ValueError("materialization requested projection mismatch")
    if payload.get("applied_attributes") != applied:
        raise ValueError("materialization applied projection mismatch")
    if payload.get("abstained_attributes") != abstained:
        raise ValueError("materialization abstained projection mismatch")
    for field_name in (
        "glyph_core_envelope",
        "effect_envelope",
        "render_metrics",
    ):
        if not isinstance(payload.get(field_name), dict):
            raise ValueError(f"owner style raster contract v2 {field_name} must be a mapping")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list) or capabilities != sorted(set(str(item) for item in capabilities)):
        raise ValueError("owner style raster contract v2 capabilities are not canonical")
    if not isinstance(payload.get("segments"), list):
        raise ValueError("owner style raster contract v2 segments must be a list")
    if payload.get("contract_sha256") != owner_style_raster_contract_sha256(payload):
        raise ValueError("owner style raster contract v2 hash mismatch")
    return payload


def validate_owner_style_raster_contract(
    value: OwnerStyleRasterContract | OwnerStyleRasterContractV2 | Mapping[str, Any],
    *,
    expected_owner_id: str | None = None,
    expected_page_id: str | None = None,
) -> dict[str, Any]:
    payload = value.to_dict() if isinstance(value, OwnerStyleRasterContract) else _thaw_owner_style_json(value)
    if not isinstance(payload, dict):
        raise TypeError("owner style raster contract must be a mapping")
    if payload.get("schema_version") == 2:
        return _validate_owner_style_raster_contract_v2(
            value,
            expected_owner_id=expected_owner_id,
            expected_page_id=expected_page_id,
        )
    return _validate_owner_style_raster_contract_v1(
        value,
        expected_owner_id=expected_owner_id,
        expected_page_id=expected_page_id,
    )


@dataclass(frozen=True)
class OwnerGlyphPatch:
    """One owner-scoped rendered glyph result chained to a cleanup mutation."""

    owner_id: str
    page_id: str
    coordinate_space: str
    result_rgb: Any
    glyph_mask: Any
    glyph_bbox_page: BBox | None
    render_completed: bool
    fit_status: str
    before_sha256: str
    after_sha256: str
    glyph_mask_sha256: str
    changed_outside_glyph_mask_pixels: int
    render_safe_polygon_page: tuple[Point, ...]
    render_safe_polygon_sha256: str
    component_geometry_sha256: str
    render_quality_contract: OwnerRenderQuality
    style_raster_contract: OwnerStyleRasterContract | OwnerStyleRasterContractV2
    owner_render_geometry_sha256: str = ""
    owner_render_geometry: Any = None
    execution_tile_id: str | None = None
    projection_role: str = "executor"
    color_space: str = "RGB"
    glyph_core_mask: Any = None
    paint_mask: Any = None
    glyph_span_core_masks: tuple[Any, ...] = ()
    glyph_span_runs: tuple[Any, ...] = ()
    glyph_core_mask_sha256: str = ""
    paint_mask_sha256: str = ""
    text_execution_authority_sha256: str = ""
    text_execution_authority: Any = None
    delivery_contract: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.render_quality_contract, OwnerRenderQuality):
            raise TypeError("owner glyph patch requires an OwnerRenderQuality contract")
        if not isinstance(
            self.style_raster_contract,
            (OwnerStyleRasterContract, OwnerStyleRasterContractV2),
        ):
            raise TypeError("owner glyph patch requires a style raster contract")
        validate_owner_style_raster_contract(
            self.style_raster_contract,
            expected_owner_id=self.owner_id,
            expected_page_id=self.page_id,
        )
        if (
            self.style_raster_contract.execution_component_geometry_sha256
            != self.component_geometry_sha256
        ):
            raise ValueError(
                "owner glyph patch style raster contract geometry mismatch"
            )
        if self.owner_render_geometry is not None:
            try:
                from .render_geometry import OwnerRenderGeometry

                geometry = (
                    self.owner_render_geometry
                    if isinstance(self.owner_render_geometry, OwnerRenderGeometry)
                    else OwnerRenderGeometry.from_dict(self.owner_render_geometry)
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError("owner glyph patch render geometry is invalid") from exc
            if geometry.owner_id != self.owner_id or geometry.page_id != self.page_id:
                raise ValueError("owner glyph patch render geometry identity mismatch")
            if geometry.geometry_sha256 != self.owner_render_geometry_sha256:
                raise ValueError("owner glyph patch render geometry hash mismatch")
            object.__setattr__(self, "owner_render_geometry", geometry)
        if self.style_raster_contract.rendered_after_sha256 != self.after_sha256:
            raise ValueError(
                "owner glyph patch style raster contract after hash mismatch"
            )
        if self.render_completed and self.render_quality_contract.status != "ok":
            raise ValueError(
                "completed owner glyph patch requires render quality status=ok"
            )
        for field_name in ("result_rgb", "glyph_mask", "glyph_core_mask", "paint_mask"):
            value = getattr(self, field_name)
            if value is None:
                continue
            copy_value = getattr(value, "copy", None)
            if not callable(copy_value):
                continue
            frozen_value = copy_value()
            setflags = getattr(frozen_value, "setflags", None)
            if callable(setflags):
                setflags(write=False)
            object.__setattr__(self, field_name, frozen_value)
        frozen_span_masks: list[Any] = []
        for value in self.glyph_span_core_masks:
            copy_value = getattr(value, "copy", None)
            frozen_value = copy_value() if callable(copy_value) else value
            setflags = getattr(frozen_value, "setflags", None)
            if callable(setflags):
                setflags(write=False)
            frozen_span_masks.append(frozen_value)
        object.__setattr__(self, "glyph_span_core_masks", tuple(frozen_span_masks))
        object.__setattr__(self, "glyph_span_runs", tuple(self.glyph_span_runs))


@dataclass(frozen=True)
class OwnerCompositionConflict:
    """One deterministic reason why a page composition cannot be committed."""

    code: str
    phase: str
    owner_ids: tuple[str, ...]
    pixel_count: int = 0
    message: str = ""


@dataclass(frozen=True)
class PageCompositionResult:
    """Immutable result produced by the pure page-space owner compositor."""

    final_rgb: Any
    cleanup_owner_map: Any
    glyph_owner_map: Any
    conflicts: tuple[OwnerCompositionConflict, ...]
    write_counts: dict[str, int]
    sha256: str
    page_id: str | None = None
    coordinate_space: str = "logical_page"
    committed: bool = True
    page_surface_geometry_sha256: str | None = None
    page_surface_geometry: Any = None
    owner_text_execution_authorities: dict[str, Any] = field(default_factory=dict)
    owner_text_delivery_contracts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "final_rgb",
            "cleanup_owner_map",
            "glyph_owner_map",
        ):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, _deep_frozen_array_copy(value))
        object.__setattr__(self, "conflicts", tuple(self.conflicts))
        object.__setattr__(
            self,
            "write_counts",
            MappingProxyType(
                {
                    str(key): int(value)
                    for key, value in dict(self.write_counts).items()
                }
            ),
        )
        for field_name in (
            "owner_text_execution_authorities",
            "owner_text_delivery_contracts",
        ):
            values = {}
            for owner_id, value in dict(getattr(self, field_name)).items():
                to_dict = getattr(value, "to_dict", None)
                values[str(owner_id)] = to_dict() if callable(to_dict) else dict(value)
            object.__setattr__(self, field_name, MappingProxyType(values))
        if self.coordinate_space == "framed_page" and self.page_surface_geometry is None:
            raise ValueError("framed page composition requires page surface geometry")
        if self.page_surface_geometry is None and self.page_surface_geometry_sha256:
            raise ValueError("page surface geometry hash requires geometry")
        if self.page_surface_geometry is not None:
            try:
                from strip.page_surface_geometry import PageSurfaceGeometry

                geometry = (
                    self.page_surface_geometry
                    if isinstance(self.page_surface_geometry, PageSurfaceGeometry)
                    else PageSurfaceGeometry.from_dict(self.page_surface_geometry)
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError("page surface geometry is invalid") from exc
            if geometry.geometry_sha256 != self.page_surface_geometry_sha256:
                raise ValueError("page surface geometry hash mismatch")
            expected_shape = (
                (geometry.frame_height, geometry.frame_width)
                if self.coordinate_space == "framed_page"
                else (geometry.logical_height, geometry.logical_width)
            )
            if any(
                tuple(np.asarray(value).shape[:2]) != expected_shape
                for value in (self.final_rgb, self.cleanup_owner_map, self.glyph_owner_map)
            ):
                raise ValueError("page surface geometry shape mismatch")
            object.__setattr__(self, "page_surface_geometry", geometry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_rgb": np.asarray(self.final_rgb).tolist(),
            "cleanup_owner_map": np.asarray(self.cleanup_owner_map).tolist(),
            "glyph_owner_map": np.asarray(self.glyph_owner_map).tolist(),
            "conflicts": [vars(item) for item in self.conflicts],
            "write_counts": dict(self.write_counts),
            "sha256": self.sha256,
            "page_id": self.page_id,
            "coordinate_space": self.coordinate_space,
            "committed": self.committed,
            "page_surface_geometry_sha256": self.page_surface_geometry_sha256,
            "page_surface_geometry": (
                self.page_surface_geometry.to_dict()
                if self.page_surface_geometry is not None
                else None
            ),
            "owner_text_execution_authorities": dict(
                self.owner_text_execution_authorities
            ),
            "owner_text_delivery_contracts": dict(
                self.owner_text_delivery_contracts
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, enforce: bool = False) -> "PageCompositionResult":
        coordinate_space = str(data.get("coordinate_space") or "")
        geometry_payload = data.get("page_surface_geometry")
        if enforce and coordinate_space == "framed_page" and not isinstance(geometry_payload, Mapping):
            raise ValueError("framed page composition requires page surface geometry")
        if enforce and coordinate_space not in {"logical_page", "framed_page"}:
            raise ValueError("page composition uses legacy coordinate space")
        conflicts = tuple(
            OwnerCompositionConflict(**item)
            for item in data.get("conflicts") or ()
        )
        return cls(
            final_rgb=np.asarray(data["final_rgb"], dtype=np.uint8),
            cleanup_owner_map=np.asarray(data["cleanup_owner_map"], dtype=str),
            glyph_owner_map=np.asarray(data["glyph_owner_map"], dtype=str),
            conflicts=conflicts,
            write_counts=dict(data.get("write_counts") or {}),
            sha256=str(data.get("sha256") or ""), page_id=data.get("page_id"),
            coordinate_space=coordinate_space, committed=bool(data.get("committed", True)),
            page_surface_geometry_sha256=data.get("page_surface_geometry_sha256"),
            page_surface_geometry=geometry_payload,
            owner_text_execution_authorities=dict(
                data.get("owner_text_execution_authorities") or {}
            ),
            owner_text_delivery_contracts=dict(
                data.get("owner_text_delivery_contracts") or {}
            ),
        )


@dataclass(frozen=True)
class OwnerExecutionCommit:
    """Atomic outcome of one cleanup-and-render chain for a resolved owner."""

    owner_id: str
    page_id: str
    coordinate_space: str
    result_rgb: Any
    mutation: OwnerMutation
    glyph_patch: OwnerGlyphPatch | None
    committed: bool
    cleanup_committed: bool
    render_committed: bool
    review_required: bool
    state: str
    reason: str
    before_sha256: str
    after_sha256: str
    rollback_mask_sha256: str
    rollback_pixels: int
    execution_tile_id: str | None = None

    def __post_init__(self) -> None:
        copy_value = getattr(self.result_rgb, "copy", None)
        if not callable(copy_value):
            return
        frozen_value = copy_value()
        setflags = getattr(frozen_value, "setflags", None)
        if callable(setflags):
            setflags(write=False)
        object.__setattr__(self, "result_rgb", frozen_value)


@dataclass(frozen=True)
class OwnerViolation:
    """Stable invariant failure with explicit offending identities."""

    code: str
    severity: str
    message: str
    offenders: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "offenders": list(self.offenders),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OwnerViolation":
        return cls(
            code=str(data.get("code") or ""),
            severity=str(data.get("severity") or "critical"),
            message=str(data.get("message") or ""),
            offenders=tuple(str(value) for value in data.get("offenders") or ()),
        )


class OwnerGraphValidationError(ValueError):
    """Raised when an owner graph cannot safely enter an execution stage."""

    def __init__(self, violations: Iterable[OwnerViolation]):
        self.violations = tuple(violations)
        codes = ", ".join(violation.code for violation in self.violations)
        super().__init__(f"owner graph validation failed: {codes}")


@dataclass
class OwnerGraph:
    """Authoritative page-level graph shared by all downstream stages."""

    schema_version: int
    page_id: str
    components: list[SourceTextComponent]
    observations: list[TextObservation]
    owners: list[TextOwner]
    projections: list[OwnerProjection]
    component_dispositions: list[ComponentDisposition] = field(default_factory=list)
    violations: list[OwnerViolation] = field(default_factory=list)
    _: KW_ONLY
    run_id: str = ""
    origin_execution_id: str = ""
    page_source_sha256: str = ""
    verification_status: Literal["verified", "legacy_unverified"] = "verified"

    def validate(self, *, mode: str = "legacy") -> tuple[OwnerViolation, ...]:
        violations = list(self.violations)
        if mode not in {"legacy", "shadow", "enforce"}:
            raise ValueError(f"unsupported owner graph validation mode: {mode}")
        if self.schema_version not in {
            OWNER_GRAPH_LEGACY_SCHEMA_VERSION,
            OWNER_GRAPH_SCHEMA_VERSION,
        }:
            violations.append(
                _violation(
                    "owner_graph_schema_unknown",
                    "Owner graph schema version is unsupported.",
                    str(self.schema_version),
                )
            )
        if mode == "enforce":
            if self.schema_version != OWNER_GRAPH_SCHEMA_VERSION:
                violations.append(
                    _violation(
                        "owner_graph_schema_legacy",
                        "Legacy owner graphs cannot enter enforce mode.",
                        str(self.schema_version),
                    )
                )
            if self.verification_status != "verified":
                violations.append(
                    _violation(
                        "owner_graph_unverified",
                        "Only explicitly verified owner graphs can enter enforce mode.",
                        self.verification_status,
                    )
                )
            for field_name, value in (
                ("run_id", self.run_id),
                ("origin_execution_id", self.origin_execution_id),
                ("page_id", self.page_id),
                ("page_source_sha256", self.page_source_sha256),
            ):
                if not str(value or "").strip():
                    violations.append(
                        _violation(
                            f"owner_graph_{field_name}_missing",
                            f"Verified owner graph is missing {field_name}.",
                        )
                    )
        component_ids = {component.component_id for component in self.components}
        observation_ids = {observation.observation_id for observation in self.observations}
        observations_by_id = {
            observation.observation_id: observation for observation in self.observations
        }
        owners_by_id = {owner.owner_id: owner for owner in self.owners}

        for label, identities in (
            ("component", [component.component_id for component in self.components]),
            (
                "observation",
                [observation.observation_id for observation in self.observations],
            ),
            ("owner", [owner.owner_id for owner in self.owners]),
        ):
            counts = {
                identity: identities.count(identity) for identity in set(identities)
            }
            for identity, count in sorted(counts.items()):
                if count > 1:
                    violations.append(
                        _violation(
                            f"{label}_identity_duplicated",
                            f"Page owner graph contains duplicate {label} identity.",
                            identity,
                        )
                    )

        for component in self.components:
            if component.page_id != self.page_id:
                violations.append(
                    _violation(
                        "component_page_mismatch",
                        "Source component belongs to another page.",
                        component.component_id,
                        component.page_id,
                    )
                )
            if not _is_canonical_page_bbox(component.bbox_page):
                violations.append(
                    _violation(
                        "component_bbox_invalid",
                        "Source component bbox_page is not canonical page geometry.",
                        component.component_id,
                    )
                )
            duplicate_evidence_ids = _duplicate_values(component.evidence_ids)
            if duplicate_evidence_ids:
                violations.append(
                    _violation(
                        "component_evidence_ids_duplicated",
                        "Source component contains duplicate evidence identities.",
                        component.component_id,
                        *duplicate_evidence_ids,
                    )
                )
        for observation in self.observations:
            if mode == "enforce":
                if not observation.identity_complete:
                    violations.append(
                        _violation(
                            "observation_identity_incomplete",
                            "Text observation has incomplete OCR request identity.",
                            observation.observation_id,
                        )
                    )
                for field_name, observed, expected in (
                    ("run_id", observation.run_id, self.run_id),
                    (
                        "origin_execution_id",
                        observation.origin_execution_id,
                        self.origin_execution_id,
                    ),
                    ("page_id", observation.page_id, self.page_id),
                    (
                        "page_source_sha256",
                        observation.page_source_sha256,
                        self.page_source_sha256,
                    ),
                ):
                    if observed != expected:
                        violations.append(
                            _violation(
                                f"observation_{field_name}_mismatch",
                                f"Text observation {field_name} differs from its graph.",
                                observation.observation_id,
                                str(observed),
                                str(expected),
                            )
                        )
                expected_payload_sha256 = sha256(
                    str(observation.text or "").encode("utf-8")
                ).hexdigest()
                if observation.payload_sha256 != expected_payload_sha256:
                    violations.append(
                        _violation(
                            "observation_payload_hash_mismatch",
                            "Text observation payload hash does not match its text.",
                            observation.observation_id,
                        )
                    )
            if observation.page_id != self.page_id:
                violations.append(
                    _violation(
                        "observation_page_mismatch",
                        "Text observation belongs to another page.",
                        observation.observation_id,
                        observation.page_id,
                    )
                )
            # Rejected provider rows remain immutable audit evidence and are
            # never eligible for owner selection. Their malformed coordinates
            # explain the rejection; they must not poison an otherwise valid
            # executable graph.
            if observation.rejection_reason is None:
                for field_name, bbox, code in (
                    ("bbox_page", observation.bbox_page, "observation_bbox_invalid"),
                    (
                        "source_bbox_page",
                        observation.source_bbox_page,
                        "observation_source_bbox_invalid",
                    ),
                    (
                        "text_pixel_bbox_page",
                        observation.text_pixel_bbox_page,
                        "observation_text_pixel_bbox_invalid",
                    ),
                    (
                        "layout_bbox_page",
                        observation.layout_bbox_page,
                        "observation_layout_bbox_invalid",
                    ),
                ):
                    if bbox is not None and not _is_canonical_page_bbox(bbox):
                        violations.append(
                            _violation(
                                code,
                                f"Text observation {field_name} is not canonical page geometry.",
                                observation.observation_id,
                            )
                        )
            unknown_components = sorted(set(observation.component_ids) - component_ids)
            if unknown_components:
                violations.append(
                    _violation(
                        "observation_component_unknown",
                        "Text observation references unknown source components.",
                        observation.observation_id,
                        *unknown_components,
                    )
                )
            for field_name, identities in (
                ("component_ids", observation.component_ids),
                ("projection_ids", observation.projection_ids),
            ):
                duplicates = _duplicate_values(identities)
                if duplicates:
                    violations.append(
                        _violation(
                            f"observation_{field_name}_duplicated",
                            f"Text observation contains duplicate {field_name} identities.",
                            observation.observation_id,
                            *duplicates,
                        )
                    )
        for owner in self.owners:
            if owner.page_id != self.page_id:
                violations.append(
                    _violation(
                        "owner_page_mismatch",
                        "Text owner belongs to another page.",
                        owner.owner_id,
                        owner.page_id,
                    )
                )
        for projection in self.projections:
            if projection.owner_id not in owners_by_id:
                violations.append(
                    _violation(
                        "projection_owner_unknown",
                        "Owner projection references an unknown owner.",
                        projection.owner_id,
                        projection.tile_id,
                    )
                )
            if projection.role not in {"executor", "context_only"}:
                violations.append(
                    _violation(
                        "projection_role_invalid",
                        "Owner projection role is unsupported.",
                        projection.owner_id,
                        projection.tile_id,
                    )
                )
            page_bbox = projection.bbox_page
            tile_bbox = projection.bbox_tile
            offset = projection.offset_xy
            structures_are_valid = (
                isinstance(page_bbox, (list, tuple))
                and isinstance(tile_bbox, (list, tuple))
                and isinstance(offset, (list, tuple))
                and len(page_bbox) == 4
                and len(tile_bbox) == 4
                and len(offset) == 2
            )
            values_are_valid = structures_are_valid and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (*page_bbox, *tile_bbox, *offset)
            )
            geometry_is_valid = (
                values_are_valid
                and min(page_bbox) >= 0
                and page_bbox[0] < page_bbox[2]
                and page_bbox[1] < page_bbox[3]
                and tile_bbox[0] < tile_bbox[2]
                and tile_bbox[1] < tile_bbox[3]
            )
            if not geometry_is_valid:
                violations.append(
                    _violation(
                        "projection_bbox_invalid",
                        "Owner projection has invalid page/tile geometry.",
                        projection.owner_id,
                        projection.tile_id,
                    )
                )
                continue
            offset_x, offset_y = offset
            expected_page_bbox = (
                tile_bbox[0] + offset_x,
                tile_bbox[1] + offset_y,
                tile_bbox[2] + offset_x,
                tile_bbox[3] + offset_y,
            )
            if tuple(page_bbox) != expected_page_bbox:
                violations.append(
                    _violation(
                        "projection_transform_mismatch",
                        "Owner projection page geometry does not match tile offset.",
                        projection.owner_id,
                        projection.tile_id,
                    )
                )
            projection_owner = owners_by_id.get(projection.owner_id)
            if projection_owner is not None:
                owner_component_boxes = [
                    component.bbox_page
                    for component in self.components
                    if component.component_id in set(projection_owner.component_ids)
                ]
                boxes_are_canonical = (
                    len(owner_component_boxes) == len(set(projection_owner.component_ids))
                    and bool(owner_component_boxes)
                    and all(
                        isinstance(box, (list, tuple))
                        and len(box) == 4
                        and all(
                            isinstance(value, int) and not isinstance(value, bool)
                            for value in box
                        )
                        and min(box) >= 0
                        and box[0] < box[2]
                        and box[1] < box[3]
                        for box in owner_component_boxes
                    )
                )
                if boxes_are_canonical:
                    owner_bbox = (
                        min(box[0] for box in owner_component_boxes),
                        min(box[1] for box in owner_component_boxes),
                        max(box[2] for box in owner_component_boxes),
                        max(box[3] for box in owner_component_boxes),
                    )
                    if tuple(page_bbox) != owner_bbox:
                        violations.append(
                            _violation(
                                "projection_owner_geometry_mismatch",
                                "Owner projection does not match its component union.",
                                projection.owner_id,
                                projection.tile_id,
                            )
                        )

        projection_identity_counts: dict[tuple[str, str], int] = {}
        for projection in self.projections:
            identity = (projection.owner_id, projection.tile_id)
            projection_identity_counts[identity] = (
                projection_identity_counts.get(identity, 0) + 1
            )
        for (owner_id, tile_id), count in sorted(projection_identity_counts.items()):
            if count > 1:
                violations.append(
                    _violation(
                        "owner_projection_duplicated",
                        "Owner has duplicate projections for one tile.",
                        owner_id,
                        tile_id,
                    )
                )

        dispositions_by_component: dict[str, list[ComponentDisposition]] = {}
        for disposition in self.component_dispositions:
            dispositions_by_component.setdefault(disposition.component_id, []).append(disposition)
            if disposition.component_id not in component_ids:
                violations.append(
                    _violation(
                        "disposition_component_unknown",
                        "Disposition references an unknown source component.",
                        disposition.component_id,
                    )
                )
            if disposition.decision not in FINAL_COMPONENT_DECISIONS:
                violations.append(
                    _violation(
                        "component_disposition_invalid",
                        "Source component has an unsupported final disposition.",
                        disposition.component_id,
                    )
                )
            elif disposition.decision in {"owned", "review"}:
                disposition_owner = owners_by_id.get(disposition.owner_id or "")
                if disposition_owner is not None:
                    if disposition_owner.disposition != disposition.decision:
                        violations.append(
                            _violation(
                                "disposition_owner_decision_mismatch",
                                "Component disposition does not match its owner's disposition.",
                                disposition.component_id,
                                disposition_owner.owner_id,
                            )
                        )
                    if disposition.component_id not in disposition_owner.component_ids:
                        violations.append(
                            _violation(
                                "disposition_owner_component_mismatch",
                                "Component disposition references an owner that does not contain the component.",
                                disposition.component_id,
                                disposition_owner.owner_id,
                            )
                        )
            elif disposition.owner_id is not None:
                violations.append(
                    _violation(
                        "disposition_owner_forbidden",
                        "Preserved or suppressed components cannot retain an owner identity.",
                        disposition.component_id,
                        disposition.owner_id,
                    )
                )

        for component_id in sorted(component_ids):
            decisions = dispositions_by_component.get(component_id, [])
            if not decisions:
                violations.append(
                    _violation(
                        "source_text_unowned",
                        "Source text component has no final disposition.",
                        component_id,
                    )
                )
            elif len(decisions) > 1:
                violations.append(
                    _violation(
                        "component_disposition_conflict",
                        "Source text component has multiple final dispositions.",
                        component_id,
                    )
                )

        active_owners_by_component: dict[str, list[str]] = {}
        for owner in self.owners:
            for field_name, identities in (
                ("component_ids", owner.component_ids),
                ("observation_ids", owner.observation_ids),
                ("selected_observation_ids", owner.selected_observation_ids),
            ):
                duplicates = _duplicate_values(identities)
                if duplicates:
                    violations.append(
                        _violation(
                            f"owner_{field_name}_duplicated",
                            f"Text owner contains duplicate {field_name} identities.",
                            owner.owner_id,
                            *duplicates,
                        )
                    )

            disposition_is_valid = owner.disposition in OWNER_DISPOSITIONS
            state_is_valid = owner.state in OWNER_STATES
            route_is_valid = owner.route_action in OWNER_ROUTE_ACTIONS
            if not disposition_is_valid:
                violations.append(
                    _violation(
                        "owner_disposition_invalid",
                        "Text owner disposition is not canonical.",
                        owner.owner_id,
                        owner.disposition,
                    )
                )
            if not state_is_valid:
                violations.append(
                    _violation(
                        "owner_state_invalid",
                        "Text owner lifecycle state is not canonical.",
                        owner.owner_id,
                        owner.state,
                    )
                )
            if mode == "enforce" and owner.state in LEGACY_OWNER_STATES:
                violations.append(
                    _violation(
                        "owner_state_legacy_in_enforce",
                        "Enforce mode requires the canonical owner lifecycle.",
                        owner.owner_id,
                        owner.state,
                    )
                )
            if not route_is_valid:
                violations.append(
                    _violation(
                        "owner_route_action_invalid",
                        "Text owner route action is not canonical.",
                        owner.owner_id,
                        owner.route_action,
                    )
                )
            if (
                state_is_valid
                and route_is_valid
                and owner.state not in ROUTE_ALLOWED_STATES[owner.route_action]
            ):
                violations.append(
                    _violation(
                        "owner_state_route_mismatch",
                        "Text owner state and route action are not a valid lifecycle pair.",
                        owner.owner_id,
                        owner.state,
                        owner.route_action,
                    )
                )
            if owner.disposition == "review" and (
                owner.state != "review_required"
                or owner.route_action != "review_required"
            ):
                violations.append(
                    _violation(
                        "owner_disposition_lifecycle_mismatch",
                        "Review owner must remain in the review-required lifecycle.",
                        owner.owner_id,
                    )
                )
            if mode == "enforce" and (
                owner.disposition == "review" or owner.state == "review_required"
            ):
                violations.append(
                    _violation(
                        "review_terminal_forbidden_in_enforce",
                        "Enforce mode cannot terminate an owner in review.",
                        owner.owner_id,
                    )
                )

            owner_component_ids = set(owner.component_ids)
            if owner.disposition == "owned" and not owner_component_ids:
                violations.append(
                    _violation(
                        "owner_components_missing",
                        "Owned owner must contain at least one source component.",
                        owner.owner_id,
                    )
                )
            for component_id in owner.component_ids:
                if component_id not in component_ids:
                    violations.append(
                        _violation(
                            "owner_component_unknown",
                            "Owner references an unknown source component.",
                            owner.owner_id,
                            component_id,
                        )
                    )
                if owner.disposition == "owned":
                    active_owners_by_component.setdefault(component_id, []).append(owner.owner_id)

            if disposition_is_valid:
                for component_id in sorted(owner_component_ids):
                    component_dispositions = dispositions_by_component.get(
                        component_id, ()
                    )
                    matching_disposition = any(
                        disposition.decision == owner.disposition
                        and disposition.owner_id == owner.owner_id
                        for disposition in component_dispositions
                    )
                    if component_dispositions and not matching_disposition:
                        violations.append(
                            _violation(
                                "owner_component_disposition_mismatch",
                                "Owner component has no matching final disposition.",
                                owner.owner_id,
                                component_id,
                            )
                        )

            unknown_observations = sorted(set(owner.observation_ids) - observation_ids)
            if unknown_observations:
                violations.append(
                    _violation(
                        "owner_observation_unknown",
                        "Owner references observations that are not present in the graph.",
                        owner.owner_id,
                        *unknown_observations,
                    )
                )

            unknown_selected = sorted(
                set(owner.selected_observation_ids) - set(owner.observation_ids)
            )
            if unknown_selected:
                violations.append(
                    _violation(
                        "owner_selected_observation_unknown",
                        "Selected observation is not part of the owner evidence.",
                        owner.owner_id,
                        *unknown_selected,
                    )
                )

            rejected_selected = sorted(
                observation_id
                for observation_id in owner.selected_observation_ids
                if observation_id in observations_by_id
                and observations_by_id[observation_id].rejection_reason is not None
            )
            if rejected_selected:
                violations.append(
                    _violation(
                        "owner_selected_observation_rejected",
                        "Rejected OCR evidence cannot become an executable owner payload.",
                        owner.owner_id,
                        *rejected_selected,
                    )
                )

            evidence_coverage = {
                component_id
                for observation_id in owner.observation_ids
                if observation_id in observations_by_id
                for component_id in observations_by_id[observation_id].component_ids
            }
            selected_coverage = {
                component_id
                for observation_id in owner.selected_observation_ids
                if observation_id in observations_by_id
                for component_id in observations_by_id[observation_id].component_ids
            }
            if owner.disposition == "owned":
                missing_evidence = sorted(owner_component_ids - evidence_coverage)
                if missing_evidence:
                    violations.append(
                        _violation(
                            "owner_observation_coverage_incomplete",
                            "Owned owner observations do not cover every source component.",
                            owner.owner_id,
                            *missing_evidence,
                        )
                    )
                missing_selected = sorted(owner_component_ids - selected_coverage)
                if missing_selected:
                    violations.append(
                        _violation(
                            "owner_selected_coverage_incomplete",
                            "Owned owner selected observations do not cover every source component.",
                            owner.owner_id,
                            *missing_selected,
                        )
                    )
            selected_foreign = sorted(selected_coverage - owner_component_ids)
            if selected_foreign:
                violations.append(
                    _violation(
                        "owner_selected_observation_contaminated",
                        "Selected owner observations contain foreign source components.",
                        owner.owner_id,
                        *selected_foreign,
                    )
                )

            owner_projections = [
                projection for projection in self.projections if projection.owner_id == owner.owner_id
            ]
            executors = [projection for projection in owner_projections if projection.role == "executor"]

            if owner.disposition == "owned":
                if not isinstance(owner.source_payload, str) or not owner.source_payload.strip():
                    code = (
                        "semantic_body_split"
                        if owner.semantic_role.endswith("body")
                        and not isinstance(owner.source_payload, str)
                        else "owner_source_payload_missing"
                    )
                    violations.append(
                        _violation(
                            code,
                            "Translatable owner must expose one non-empty source payload.",
                            owner.owner_id,
                        )
                    )
                if (
                    owner.route_action in TRANSLATION_ROUTE_ACTIONS
                    and owner.state in POST_TRANSLATION_STATES
                    and (
                        not isinstance(owner.translated_payload, str)
                        or not owner.translated_payload.strip()
                    )
                ):
                    violations.append(
                        _violation(
                            "owner_translation_payload_missing",
                            "Translated owner state requires one non-empty translated payload.",
                            owner.owner_id,
                        )
                    )
                mask_is_required = (
                    owner.route_action in INPAINT_ROUTE_ACTIONS
                    and owner.state in MASK_REQUIRED_STATES
                )
                if mask_is_required and (
                    not isinstance(owner.action_mask_ref, str)
                    or not owner.action_mask_ref.strip()
                ):
                    violations.append(
                        _violation(
                            "owner_action_mask_missing",
                            "Mask-ready owner requires one authoritative action mask reference.",
                            owner.owner_id,
                        )
                    )
                elif (
                    mask_is_required
                    and not _action_mask_ref_matches_owner(
                        owner.owner_id,
                        owner.action_mask_ref,
                    )
                ):
                    violations.append(
                        _violation(
                            "owner_action_mask_owner_mismatch",
                            "Action mask reference does not belong to its owner identity.",
                            owner.owner_id,
                        )
                    )
                if owner.state in EXECUTOR_REQUIRED_STATES and not executors:
                    violations.append(
                        _violation(
                            "owner_executor_missing",
                            "Active owner has no executor projection.",
                            owner.owner_id,
                        )
                    )
                elif len(executors) > 1:
                    violations.append(
                        _violation(
                            "owner_executor_duplicated",
                            "Active owner has more than one executor projection.",
                            owner.owner_id,
                            *(projection.tile_id for projection in executors),
                        )
                    )
                elif executors and owner.execution_tile_id != executors[0].tile_id:
                    violations.append(
                        _violation(
                            "owner_executor_identity_mismatch",
                            "Owner execution tile does not match its executor projection.",
                            owner.owner_id,
                            executors[0].tile_id,
                        )
                    )

            if owner.disposition != "owned" and (
                owner.state in EXECUTOR_REQUIRED_STATES
                or owner.route_action in EXECUTION_ROUTE_ACTIONS
                or bool(owner.action_mask_ref)
                or bool(executors)
            ):
                violations.append(
                    _violation(
                        "non_owned_owner_in_execution_plan",
                        "Non-owned owner cannot retain executable state, route, mask, or projection.",
                        owner.owner_id,
                    )
                )

            if (
                owner.disposition == "review" or owner.state == "review_required"
            ) and ("render" in owner.route_action or bool(executors)):
                violations.append(
                    _violation(
                        "review_owner_in_render_plan",
                        "Review-required owner cannot enter the render plan.",
                        owner.owner_id,
                    )
                )

        for component_id, owner_ids in sorted(active_owners_by_component.items()):
            unique_owner_ids = sorted(set(owner_ids))
            if len(unique_owner_ids) > 1:
                violations.append(
                    _violation(
                        "component_multiple_active_owners",
                        "Source component belongs to multiple active owners.",
                        component_id,
                        *unique_owner_ids,
                    )
                )

        for disposition in self.component_dispositions:
            if disposition.decision in {"owned", "review"}:
                if not disposition.owner_id or disposition.owner_id not in owners_by_id:
                    violations.append(
                        _violation(
                            "disposition_owner_missing",
                            "Owned or reviewed source component must reference a graph owner.",
                            disposition.component_id,
                        )
                    )

        return tuple(_dedupe_and_sort_violations(violations))

    def require_valid(self, *, mode: str = "legacy") -> None:
        violations = self.validate(mode=mode)
        if any(violation.severity == "critical" for violation in violations):
            raise OwnerGraphValidationError(violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "page_id": self.page_id,
            "run_id": self.run_id,
            "origin_execution_id": self.origin_execution_id,
            "page_source_sha256": self.page_source_sha256,
            "verification_status": (
                "legacy_unverified"
                if self.schema_version == OWNER_GRAPH_LEGACY_SCHEMA_VERSION
                else self.verification_status
            ),
            "components": [
                {
                    "component_id": component.component_id,
                    "page_id": component.page_id,
                    "bbox_page": list(component.bbox_page),
                    "polygon_page": [list(point) for point in component.polygon_page],
                    "detector_sources": list(component.detector_sources),
                    "confidence": (
                        float(component.confidence)
                        if component.confidence is not None
                        else None
                    ),
                    "script_evidence": list(component.script_evidence),
                    "evidence_ids": list(component.evidence_ids),
                    "rotation_deg": (
                        float(component.rotation_deg)
                        if component.rotation_deg is not None
                        else None
                    ),
                    "rotation_source": component.rotation_source,
                }
                for component in sorted(self.components, key=lambda item: item.component_id)
            ],
            "observations": [
                {
                    "observation_id": observation.observation_id,
                    "page_id": observation.page_id,
                    "component_ids": list(observation.component_ids),
                    "text": observation.text,
                    "confidence": float(observation.confidence),
                    "provider": observation.provider,
                    "bbox_page": list(observation.bbox_page),
                    "polygons_page": [
                        [list(point) for point in polygon]
                        for polygon in observation.polygons_page
                    ],
                    "tile_provenance": list(observation.tile_provenance),
                    "coverage_score": observation.coverage_score,
                    "language_score": observation.language_score,
                    "rejection_reason": observation.rejection_reason,
                    "legacy_rejection_reason": observation.legacy_rejection_reason,
                    "legacy_selected": bool(observation.legacy_selected),
                    "provider_variant": observation.provider_variant,
                    "provider_record_id": observation.provider_record_id,
                    "projection_ids": list(observation.projection_ids),
                    "raw_text": observation.raw_text,
                    "source_bbox_page": (
                        list(observation.source_bbox_page)
                        if observation.source_bbox_page is not None
                        else None
                    ),
                    "text_pixel_bbox_page": (
                        list(observation.text_pixel_bbox_page)
                        if observation.text_pixel_bbox_page is not None
                        else None
                    ),
                    "layout_bbox_page": (
                        list(observation.layout_bbox_page)
                        if observation.layout_bbox_page is not None
                        else None
                    ),
                    "line_texts": list(observation.line_texts),
                    "rotation_deg": (
                        float(observation.rotation_deg)
                        if observation.rotation_deg is not None
                        else None
                    ),
                    "rotation_source": observation.rotation_source,
                    "run_id": observation.run_id,
                    "origin_execution_id": observation.origin_execution_id,
                    "invocation_id": observation.invocation_id,
                    "attempt_id": observation.attempt_id,
                    "provider_family": observation.provider_family,
                    "page_source_sha256": observation.page_source_sha256,
                    "root_input_pixel_sha256": observation.root_input_pixel_sha256,
                    "input_pixel_sha256": observation.input_pixel_sha256,
                    "payload_sha256": observation.payload_sha256,
                }
                for observation in sorted(
                    self.observations, key=lambda item: item.observation_id
                )
            ],
            "owners": [
                {
                    "owner_id": owner.owner_id,
                    "page_id": owner.page_id,
                    "component_ids": sorted(owner.component_ids),
                    "observation_ids": sorted(owner.observation_ids),
                    "selected_observation_ids": sorted(owner.selected_observation_ids),
                    "semantic_role": owner.semantic_role,
                    "source_payload": owner.source_payload,
                    "translated_payload": owner.translated_payload,
                    "disposition": owner.disposition,
                    "state": owner.state,
                    "route_action": owner.route_action,
                    "execution_tile_id": owner.execution_tile_id,
                    "action_mask_ref": owner.action_mask_ref,
                }
                for owner in sorted(self.owners, key=lambda item: item.owner_id)
            ],
            "projections": [
                {
                    "owner_id": projection.owner_id,
                    "tile_id": projection.tile_id,
                    "role": projection.role,
                    "bbox_page": list(projection.bbox_page),
                    "bbox_tile": list(projection.bbox_tile),
                    "offset_xy": list(projection.offset_xy),
                }
                for projection in sorted(
                    self.projections, key=lambda item: (item.owner_id, item.tile_id, item.role)
                )
            ],
            "component_dispositions": [
                {
                    "component_id": disposition.component_id,
                    "decision": disposition.decision,
                    "owner_id": disposition.owner_id,
                    "reason": disposition.reason,
                }
                for disposition in sorted(
                    self.component_dispositions,
                    key=lambda item: (item.component_id, item.decision, item.owner_id or ""),
                )
            ],
            "violations": [
                violation.to_dict()
                for violation in _dedupe_and_sort_violations(self.violations)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, enforce: bool = False) -> "OwnerGraph":
        schema_version = int(data.get("schema_version") or OWNER_GRAPH_LEGACY_SCHEMA_VERSION)
        if schema_version not in {
            OWNER_GRAPH_LEGACY_SCHEMA_VERSION,
            OWNER_GRAPH_SCHEMA_VERSION,
        }:
            raise OwnerGraphValidationError(
                (
                    _violation(
                        "owner_graph_schema_unknown",
                        "Owner graph schema version is unsupported.",
                        str(schema_version),
                    ),
                )
            )
        graph = cls(
            schema_version=schema_version,
            page_id=str(data.get("page_id") or ""),
            components=[
                SourceTextComponent(
                    component_id=str(item.get("component_id") or ""),
                    page_id=str(item.get("page_id") or ""),
                    bbox_page=_bbox(item.get("bbox_page")),
                    polygon_page=_polygon(item.get("polygon_page")),
                    detector_sources=tuple(
                        str(value) for value in item.get("detector_sources") or ()
                    ),
                    confidence=(
                        float(item["confidence"])
                        if item.get("confidence") is not None
                        else None
                    ),
                    script_evidence=tuple(
                        str(value) for value in item.get("script_evidence") or ()
                    ),
                    evidence_ids=tuple(
                        str(value) for value in item.get("evidence_ids") or ()
                    ),
                    rotation_deg=(
                        float(item["rotation_deg"])
                        if item.get("rotation_deg") is not None
                        else None
                    ),
                    rotation_source=(
                        str(item["rotation_source"])
                        if item.get("rotation_source") is not None
                        else None
                    ),
                )
                for item in data.get("components") or ()
            ],
            observations=[
                TextObservation(
                    observation_id=str(item.get("observation_id") or ""),
                    page_id=str(item.get("page_id") or ""),
                    component_ids=tuple(
                        str(value) for value in item.get("component_ids") or ()
                    ),
                    text=str(item.get("text") or ""),
                    confidence=float(item.get("confidence") or 0.0),
                    provider=str(item.get("provider") or ""),
                    bbox_page=_bbox(item.get("bbox_page")),
                    polygons_page=tuple(
                        tuple((int(point[0]), int(point[1])) for point in polygon)
                        for polygon in item.get("polygons_page") or ()
                    ),
                    tile_provenance=tuple(
                        str(value) for value in item.get("tile_provenance") or ()
                    ),
                    coverage_score=_optional_float(item.get("coverage_score")),
                    language_score=_optional_float(item.get("language_score")),
                    rejection_reason=(
                        str(item["rejection_reason"])
                        if item.get("rejection_reason") is not None
                        else None
                    ),
                    legacy_rejection_reason=(
                        str(item["legacy_rejection_reason"])
                        if item.get("legacy_rejection_reason") is not None
                        else None
                    ),
                    legacy_selected=bool(item.get("legacy_selected", False)),
                    provider_variant=str(item.get("provider_variant") or ""),
                    provider_record_id=(
                        str(item["provider_record_id"])
                        if item.get("provider_record_id") is not None
                        else None
                    ),
                    projection_ids=tuple(
                        str(value) for value in item.get("projection_ids") or ()
                    ),
                    raw_text=(
                        (
                            str(item["raw_text"])
                            if item.get("raw_text") is not None
                            else None
                        )
                        if "raw_text" in item
                        else str(item.get("text") or "")
                    ),
                    source_bbox_page=_optional_bbox(item.get("source_bbox_page")),
                    text_pixel_bbox_page=_optional_bbox(item.get("text_pixel_bbox_page")),
                    layout_bbox_page=_optional_bbox(item.get("layout_bbox_page")),
                    line_texts=tuple(str(value) for value in item.get("line_texts") or ()),
                    rotation_deg=_optional_float(item.get("rotation_deg")),
                    rotation_source=(
                        str(item["rotation_source"])
                        if item.get("rotation_source") is not None
                        else None
                    ),
                    run_id=str(item.get("run_id") or ""),
                    origin_execution_id=str(item.get("origin_execution_id") or ""),
                    invocation_id=str(item.get("invocation_id") or ""),
                    attempt_id=str(
                        item.get("attempt_id") or item.get("ocr_attempt_id") or ""
                    ),
                    provider_family=str(item.get("provider_family") or ""),
                    page_source_sha256=str(item.get("page_source_sha256") or ""),
                    root_input_pixel_sha256=str(
                        item.get("root_input_pixel_sha256") or ""
                    ),
                    input_pixel_sha256=str(item.get("input_pixel_sha256") or ""),
                    payload_sha256=str(item.get("payload_sha256") or ""),
                )
                for item in data.get("observations") or ()
            ],
            owners=[
                TextOwner(
                    owner_id=str(item.get("owner_id") or ""),
                    page_id=str(item.get("page_id") or ""),
                    component_ids=[
                        str(value) for value in item.get("component_ids") or ()
                    ],
                    observation_ids=[
                        str(value) for value in item.get("observation_ids") or ()
                    ],
                    selected_observation_ids=[
                        str(value)
                        for value in item.get("selected_observation_ids") or ()
                    ],
                    semantic_role=str(item.get("semantic_role") or ""),
                    source_payload=item.get("source_payload"),
                    translated_payload=(
                        str(item["translated_payload"])
                        if item.get("translated_payload") is not None
                        else None
                    ),
                    disposition=str(item.get("disposition") or ""),
                    state=str(item.get("state") or ""),
                    route_action=str(item.get("route_action") or ""),
                    execution_tile_id=(
                        str(item["execution_tile_id"])
                        if item.get("execution_tile_id") is not None
                        else None
                    ),
                    action_mask_ref=(
                        str(item["action_mask_ref"])
                        if item.get("action_mask_ref") is not None
                        else None
                    ),
                )
                for item in data.get("owners") or ()
            ],
            projections=[
                OwnerProjection(
                    owner_id=str(item.get("owner_id") or ""),
                    tile_id=str(item.get("tile_id") or ""),
                    role=str(item.get("role") or ""),
                    bbox_page=_bbox(item.get("bbox_page")),
                    bbox_tile=_bbox(item.get("bbox_tile")),
                    offset_xy=_point(item.get("offset_xy")),
                )
                for item in data.get("projections") or ()
            ],
            component_dispositions=[
                ComponentDisposition(
                    component_id=str(item.get("component_id") or ""),
                    decision=str(item.get("decision") or ""),
                    owner_id=(
                        str(item["owner_id"])
                        if item.get("owner_id") is not None
                        else None
                    ),
                    reason=(
                        str(item["reason"]) if item.get("reason") is not None else None
                    ),
                )
                for item in data.get("component_dispositions") or ()
            ],
            violations=[
                OwnerViolation.from_dict(item) for item in data.get("violations") or ()
            ],
            run_id=str(data.get("run_id") or ""),
            origin_execution_id=str(data.get("origin_execution_id") or ""),
            page_source_sha256=str(data.get("page_source_sha256") or ""),
            verification_status=(
                str(data.get("verification_status") or "")
                if schema_version == OWNER_GRAPH_SCHEMA_VERSION
                else "legacy_unverified"
            ),
        )
        for item in data.get("observations") or ():
            legacy_attempt = item.get("ocr_attempt_id")
            canonical_attempt = item.get("attempt_id")
            if legacy_attempt and canonical_attempt and str(legacy_attempt) != str(canonical_attempt):
                raise OwnerGraphValidationError(
                    (
                        _violation(
                            "observation_attempt_identity_conflict",
                            "Legacy and canonical OCR attempt identities disagree.",
                            str(item.get("observation_id") or ""),
                        ),
                    )
                )
        if enforce:
            graph.require_valid(mode="enforce")
        return graph


def _violation(code: str, message: str, *offenders: str) -> OwnerViolation:
    return OwnerViolation(
        code=code,
        severity="critical",
        message=message,
        offenders=tuple(str(value) for value in offenders if value),
    )


def _dedupe_and_sort_violations(
    violations: Iterable[OwnerViolation],
) -> list[OwnerViolation]:
    unique = {
        (item.code, item.severity, item.message, item.offenders): item
        for item in violations
    }
    return sorted(
        unique.values(),
        key=lambda item: (item.code, item.offenders, item.message),
    )


def _bbox(value: Any) -> BBox:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"bbox must be a list or tuple, got {value!r}")
    values = list(value)
    if len(values) != 4:
        raise ValueError(f"bbox must have four values, got {values!r}")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        raise ValueError(f"bbox coordinates must be canonical integers, got {values!r}")
    return tuple(values)  # type: ignore[return-value]


def _point(value: Any) -> Point:
    values = list(value or ())
    if len(values) != 2:
        raise ValueError(f"point must have two values, got {values!r}")
    return int(values[0]), int(values[1])


def _polygon(value: Any) -> tuple[Point, ...]:
    return tuple(_point(point) for point in value or ())


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_bbox(value: Any) -> BBox | None:
    return _bbox(value) if value is not None else None
