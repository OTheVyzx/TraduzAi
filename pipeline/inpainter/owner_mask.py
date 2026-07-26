"""Fail-closed operational masks and mutations for resolved text owners."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

try:
    from ownership.model import OwnerMutation, TextOwner
except ImportError:  # pragma: no cover - supports package imports
    from ..ownership.model import OwnerMutation, TextOwner

try:
    from ocr.text_router import INPAINT_ROUTE_ACTIONS
except ImportError:  # pragma: no cover - supports package imports
    from ..ocr.text_router import INPAINT_ROUTE_ACTIONS


class UnsafeOwnerMaskError(ValueError):
    """Raised when an owner has no safe pixel evidence for automatic cleanup."""


OWNER_MASK_COORDINATE_SPACE = "page"
OWNER_MASK_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class OwnerMaskEvidence:
    """Explicit positive and negative pixel evidence for one source component."""

    evidence_id: str
    component_id: str
    glyph_mask: np.ndarray | None = None
    line_mask: np.ndarray | None = None
    protected_art_mask: np.ndarray | None = None
    observation_id: str | None = None


@dataclass(frozen=True)
class OwnerMaskPlan:
    """Authoritative, addressable mask pair for one resolved owner."""

    owner_id: str
    page_id: str
    action_mask_ref: str
    action_mask: np.ndarray
    protected_art_mask: np.ndarray
    evidence_ids: tuple[str, ...]
    source_sha256: str = ""
    execution_tile_id: str | None = None
    protected_art_mask_ref: str | None = None
    protected_evidence_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    component_action_bboxes_page: tuple[
        tuple[str, tuple[int, int, int, int]], ...
    ] = ()
    component_bboxes_page: tuple[tuple[str, tuple[int, int, int, int]], ...] = ()
    owner_bbox_page: tuple[int, int, int, int] = (0, 0, 0, 0)
    component_geometry_sha256: str = ""
    component_geometry_verified: bool = False
    coordinate_space: str = OWNER_MASK_COORDINATE_SPACE


def _canonical_identity(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise UnsafeOwnerMaskError(f"{label} must be a canonical non-empty string")
    return value


def _canonical_evidence_ids(values: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise UnsafeOwnerMaskError(f"{label} must be a sequence")
    normalized = tuple(
        sorted({_canonical_identity(value, label=f"{label} entry") for value in values})
    )
    if tuple(values) != normalized:
        raise UnsafeOwnerMaskError(f"{label} must be sorted and unique")
    return normalized


def _mask_bbox_page(mask: np.ndarray) -> tuple[int, int, int, int]:
    positive_y, positive_x = np.nonzero(mask)
    if positive_x.size <= 0:
        raise UnsafeOwnerMaskError("component action mask is empty")
    return (
        int(positive_x.min()),
        int(positive_y.min()),
        int(positive_x.max()) + 1,
        int(positive_y.max()) + 1,
    )


def _canonical_bbox_page(
    value: Any,
    *,
    shape: tuple[int, int],
    label: str,
) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise UnsafeOwnerMaskError(f"{label} must contain four canonical integers")
    x1, y1, x2, y2 = (int(item) for item in value)
    height, width = shape
    if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2 or x2 > width or y2 > height:
        raise UnsafeOwnerMaskError(f"{label} is outside canonical page geometry")
    return x1, y1, x2, y2


def _canonical_component_bbox_entries(
    value: Any,
    *,
    shape: tuple[int, int],
    label: str,
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    if isinstance(value, Mapping):
        raw_entries = list(value.items())
    elif isinstance(value, (list, tuple)):
        raw_entries = list(value)
    else:
        raise UnsafeOwnerMaskError(f"{label} must be a component geometry mapping")
    entries: list[tuple[str, tuple[int, int, int, int]]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, (list, tuple)) or len(raw_entry) != 2:
            raise UnsafeOwnerMaskError(f"{label} entry is malformed")
        component_id = _canonical_identity(
            raw_entry[0],
            label=f"{label} component_id",
        )
        bbox = _canonical_bbox_page(
            raw_entry[1],
            shape=shape,
            label=f"{label} bbox",
        )
        entries.append((component_id, bbox))
    canonical = tuple(sorted(entries, key=lambda item: item[0]))
    if len({component_id for component_id, _bbox in canonical}) != len(canonical):
        raise UnsafeOwnerMaskError(f"{label} contains duplicate component geometry")
    return canonical


def _component_geometry_sha256(
    entries: tuple[tuple[str, tuple[int, int, int, int]], ...],
) -> str:
    payload = json.dumps(
        [[component_id, list(bbox)] for component_id, bbox in entries],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _owner_bbox_from_components(
    entries: tuple[tuple[str, tuple[int, int, int, int]], ...],
) -> tuple[int, int, int, int]:
    if not entries:
        raise UnsafeOwnerMaskError("owner component geometry is empty")
    boxes = [bbox for _component_id, bbox in entries]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _binary_mask(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(value, np.ndarray) or value.size == 0:
        return None
    array = np.asarray(value)
    if array.shape[:2] != shape:
        raise UnsafeOwnerMaskError(
            f"owner mask evidence shape mismatch: expected {shape}, got {array.shape[:2]}"
        )
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise UnsafeOwnerMaskError("owner mask evidence dtype must be numeric")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise UnsafeOwnerMaskError("owner mask evidence contains non-finite pixels")
    if array.ndim == 3:
        array = np.any(array > 0, axis=2)
    elif array.ndim != 2:
        raise UnsafeOwnerMaskError("owner mask evidence must be a 2D or RGB array")
    return np.where(array > 0, 255, 0).astype(np.uint8)


def _polygon_mask(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    polygons: list[Any]
    first = value[0]
    if (
        isinstance(first, (list, tuple))
        and len(first) >= 2
        and all(isinstance(coordinate, (int, float)) for coordinate in first[:2])
    ):
        polygons = [value]
    else:
        polygons = list(value)
    mask = np.zeros(shape, dtype=np.uint8)
    valid = False
    for polygon in polygons:
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
            raise UnsafeOwnerMaskError("owner mask polygon evidence is malformed")
        try:
            raw_points = [(float(x), float(y)) for x, y, *_ in polygon]
        except (TypeError, ValueError):
            raise UnsafeOwnerMaskError("owner mask polygon evidence is malformed")
        if any(
            not math.isfinite(x)
            or not math.isfinite(y)
            or x < 0
            or y < 0
            or x >= shape[1]
            or y >= shape[0]
            for x, y in raw_points
        ):
            raise UnsafeOwnerMaskError(
                "owner mask polygon falls outside the page bounds"
            )
        points = np.asarray(
            [[int(round(x)), int(round(y))] for x, y in raw_points],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
        valid = True
    if not valid or not np.any(mask):
        raise UnsafeOwnerMaskError("owner mask polygon evidence is malformed")
    return mask


def _evidence_value(evidence: OwnerMaskEvidence | Mapping[str, Any], key: str) -> Any:
    if isinstance(evidence, OwnerMaskEvidence):
        return getattr(evidence, key, None)
    return evidence.get(key)


def _evidence_component_ids(
    evidence: OwnerMaskEvidence | Mapping[str, Any],
) -> set[str]:
    values = _evidence_value(evidence, "component_ids")
    if isinstance(values, (list, tuple, set)):
        return {str(value) for value in values if isinstance(value, str) and value}
    value = _evidence_value(evidence, "component_id")
    return {value} if isinstance(value, str) and value else set()


def _evidence_mask(
    evidence: OwnerMaskEvidence | Mapping[str, Any],
    shape: tuple[int, int],
    *keys: str,
) -> np.ndarray | None:
    result = np.zeros(shape, dtype=np.uint8)
    found = False
    for key in keys:
        value = _evidence_value(evidence, key)
        mask = _binary_mask(value, shape)
        if mask is None and "polygon" in key:
            mask = _polygon_mask(value, shape)
        if mask is not None and np.any(mask):
            result = np.maximum(result, mask)
            found = True
    return result if found else None


def _mark_owner_mask_review(owner: TextOwner) -> None:
    owner.state = "review_required"
    owner.route_action = "review_required"
    owner.action_mask_ref = None


def _owner_artifact_segment(owner_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner_id).strip("._")
    if not safe:
        raise UnsafeOwnerMaskError("owner_id cannot produce a safe mask reference")
    identity_hash = sha256(owner_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe}--{identity_hash}"


def _safe_owner_ref(
    owner_id: str,
    page_id: str,
    source_sha256: str,
    execution_tile_id: str | None,
    action_mask: np.ndarray,
    protected_art_mask: np.ndarray,
    evidence_ids: Sequence[str] = (),
    protected_evidence_ids: Sequence[str] = (),
    observation_ids: Sequence[str] = (),
    component_action_bboxes_page: Sequence[
        tuple[str, tuple[int, int, int, int]]
    ] = (),
    component_bboxes_page: Sequence[tuple[str, tuple[int, int, int, int]]] = (),
    owner_bbox_page: tuple[int, int, int, int] = (0, 0, 0, 0),
    component_geometry_sha256: str = "",
    component_geometry_verified: bool = False,
) -> tuple[str, str]:
    owner_segment = _owner_artifact_segment(owner_id)
    normalized_evidence_ids = tuple(sorted(set(evidence_ids)))
    normalized_protected_evidence_ids = tuple(sorted(set(protected_evidence_ids)))
    normalized_observation_ids = tuple(sorted(set(observation_ids)))
    pair_payload = json.dumps(
        {
            "action_mask_sha256": _array_sha256(action_mask),
            "component_action_bboxes_page": [
                [component_id, list(bbox)]
                for component_id, bbox in component_action_bboxes_page
            ],
            "component_bboxes_page": [
                [component_id, list(bbox)]
                for component_id, bbox in component_bboxes_page
            ],
            "component_geometry_sha256": component_geometry_sha256,
            "component_geometry_verified": component_geometry_verified,
            "coordinate_space": OWNER_MASK_COORDINATE_SPACE,
            "evidence_ids": normalized_evidence_ids,
            "execution_tile_id": execution_tile_id,
            "observation_ids": normalized_observation_ids,
            "page_id": page_id,
            "owner_bbox_page": list(owner_bbox_page),
            "protected_art_mask_sha256": _array_sha256(protected_art_mask),
            "protected_evidence_ids": normalized_protected_evidence_ids,
            "source_sha256": source_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    pair_hash = sha256(pair_payload.encode("utf-8")).hexdigest()[:16]
    directory = f"owner_masks/{owner_segment}/{pair_hash}"
    return (
        f"{directory}/action_mask.png",
        f"{directory}/protected_art_mask.png",
    )


def _positive_mask_is_overbroad(
    mask: np.ndarray,
    *,
    allow_dense_single_glyph: bool = False,
) -> bool:
    positive = mask > 0
    pixels = int(np.count_nonzero(positive))
    if pixels <= 0:
        return False
    height, width = positive.shape
    if pixels / float(height * width) > 0.35:
        return True
    touches_opposite_edges = (
        np.any(positive[0, :]) and np.any(positive[-1, :])
    ) or (
        np.any(positive[:, 0]) and np.any(positive[:, -1])
    )
    if touches_opposite_edges:
        return True
    positive_y, positive_x = np.nonzero(positive)
    tight_width = int(positive_x.max() - positive_x.min() + 1)
    tight_height = int(positive_y.max() - positive_y.min() + 1)
    tight_bbox_pixels = tight_width * tight_height
    dense_single_glyph_limit = max(12, int(round(min(height, width) * 0.02)))
    is_dense_single_glyph = (
        allow_dense_single_glyph
        and tight_bbox_pixels <= 4096
        and min(tight_width, tight_height) <= dense_single_glyph_limit
        and max(tight_width, tight_height) / float(min(tight_width, tight_height))
        >= 2.5
    )
    if is_dense_single_glyph:
        return False
    if (
        tight_bbox_pixels >= 256
        and min(tight_width, tight_height) >= 4
        and pixels / float(tight_bbox_pixels) >= 0.70
    ):
        return True
    tight = positive[
        int(positive_y.min()) : int(positive_y.max()) + 1,
        int(positive_x.min()) : int(positive_x.max()) + 1,
    ].astype(np.uint8)
    occupancy = pixels / float(tight_bbox_pixels)
    if tight_bbox_pixels >= 256 and occupancy >= 0.45:
        closed = cv2.morphologyEx(
            tight,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        closed_occupancy = int(np.count_nonzero(closed)) / float(tight_bbox_pixels)
        if closed_occupancy >= 0.85:
            return True
    return False


def _validated_component_geometry(
    *,
    action: np.ndarray,
    component_action_bboxes_page: Any,
    component_bboxes_page: Any,
    owner_bbox_page: Any,
    component_geometry_sha256: Any,
    component_geometry_verified: Any,
) -> tuple[
    tuple[tuple[str, tuple[int, int, int, int]], ...],
    tuple[tuple[str, tuple[int, int, int, int]], ...],
    tuple[int, int, int, int],
    str,
]:
    """Validate graph-bound component geometry for an operational mask."""

    if component_geometry_verified is not True:
        raise UnsafeOwnerMaskError(
            "owner component geometry must be verified before operational use"
        )
    shape = (int(action.shape[0]), int(action.shape[1]))
    action_entries = _canonical_component_bbox_entries(
        component_action_bboxes_page,
        shape=shape,
        label="owner component action geometry",
    )
    component_entries = _canonical_component_bbox_entries(
        component_bboxes_page,
        shape=shape,
        label="owner component geometry",
    )
    try:
        raw_action_entries = tuple(
            (entry[0], tuple(entry[1])) for entry in component_action_bboxes_page
        )
    except (TypeError, IndexError) as exc:
        raise UnsafeOwnerMaskError(
            "owner component action geometry is malformed"
        ) from exc
    if raw_action_entries != action_entries:
        raise UnsafeOwnerMaskError(
            "owner component action geometry must be canonical and sorted"
        )
    try:
        raw_component_entries = tuple(
            (entry[0], tuple(entry[1])) for entry in component_bboxes_page
        )
    except (TypeError, IndexError) as exc:
        raise UnsafeOwnerMaskError("owner component geometry is malformed") from exc
    if raw_component_entries != component_entries:
        raise UnsafeOwnerMaskError(
            "owner component geometry must be canonical and sorted"
        )
    action_ids = tuple(component_id for component_id, _bbox in action_entries)
    component_ids = tuple(component_id for component_id, _bbox in component_entries)
    if not action_ids or action_ids != component_ids:
        raise UnsafeOwnerMaskError(
            "owner component action geometry does not match owner components"
        )
    components_by_id = dict(component_entries)
    for component_id, action_bbox in action_entries:
        component_bbox = components_by_id[component_id]
        if not (
            component_bbox[0] <= action_bbox[0] < action_bbox[2] <= component_bbox[2]
            and component_bbox[1]
            <= action_bbox[1]
            < action_bbox[3]
            <= component_bbox[3]
        ):
            raise UnsafeOwnerMaskError(
                "owner component action bbox escapes authoritative component geometry"
            )
        action_slice = action[
            action_bbox[1] : action_bbox[3],
            action_bbox[0] : action_bbox[2],
        ]
        if not np.any(action_slice):
            raise UnsafeOwnerMaskError(
                "owner component action bbox contains no action-mask pixels"
            )

    canonical_owner_bbox = _canonical_bbox_page(
        owner_bbox_page,
        shape=shape,
        label="owner bbox_page",
    )
    if canonical_owner_bbox != _owner_bbox_from_components(component_entries):
        raise UnsafeOwnerMaskError(
            "owner bbox_page does not match authoritative component geometry"
        )
    action_bbox = _mask_bbox_page(action)
    if not (
        canonical_owner_bbox[0] <= action_bbox[0] < action_bbox[2] <= canonical_owner_bbox[2]
        and canonical_owner_bbox[1]
        <= action_bbox[1]
        < action_bbox[3]
        <= canonical_owner_bbox[3]
    ):
        raise UnsafeOwnerMaskError(
            "owner action mask escapes authoritative owner geometry"
        )
    geometry_sha256 = _canonical_identity(
        component_geometry_sha256,
        label="owner component geometry hash",
    )
    if (
        not re.fullmatch(r"[0-9a-f]{64}", geometry_sha256)
        or geometry_sha256 != _component_geometry_sha256(component_entries)
    ):
        raise UnsafeOwnerMaskError(
            "owner component geometry hash does not match authoritative geometry"
        )
    return action_entries, component_entries, canonical_owner_bbox, geometry_sha256


def _validated_plan_masks(plan: OwnerMaskPlan) -> tuple[np.ndarray, np.ndarray]:
    owner_id = _canonical_identity(plan.owner_id, label="owner mask plan owner_id")
    page_id = _canonical_identity(plan.page_id, label="owner mask plan page_id")
    source_sha256 = _canonical_identity(
        plan.source_sha256,
        label="owner mask plan source_sha256",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise UnsafeOwnerMaskError("owner mask plan source hash is not canonical")
    execution_tile_id = _canonical_identity(
        plan.execution_tile_id,
        label="owner mask plan executor tile_id",
    )
    if plan.coordinate_space != OWNER_MASK_COORDINATE_SPACE:
        raise UnsafeOwnerMaskError(
            "owner mask plan coordinate space must be canonical page-space"
        )
    action = np.asarray(plan.action_mask)
    protected = np.asarray(plan.protected_art_mask)
    if action.shape != protected.shape:
        raise UnsafeOwnerMaskError("owner action/protected mask shape mismatch")
    for label, mask in (("action", action), ("protected-art", protected)):
        if (
            mask.ndim != 2
            or mask.dtype != np.uint8
            or not np.all((mask == 0) | (mask == 255))
        ):
            raise UnsafeOwnerMaskError(
                f"owner {label} mask must be a canonical binary uint8 array"
            )
    if not np.any(action):
        raise UnsafeOwnerMaskError("owner action mask is empty")
    if np.any((action > 0) & (protected > 0)):
        raise UnsafeOwnerMaskError("owner action mask overlaps protected art")
    if _positive_mask_is_overbroad(action, allow_dense_single_glyph=True):
        raise UnsafeOwnerMaskError("owner action mask is overbroad for the page")
    evidence_ids = _canonical_evidence_ids(
        plan.evidence_ids,
        label="owner mask plan evidence_ids",
    )
    if not evidence_ids:
        raise UnsafeOwnerMaskError("owner mask plan has no positive evidence ids")
    protected_evidence_ids = _canonical_evidence_ids(
        plan.protected_evidence_ids,
        label="owner mask plan protected_evidence_ids",
    )
    observation_ids = _canonical_evidence_ids(
        plan.observation_ids,
        label="owner mask plan observation_ids",
    )
    if not observation_ids:
        raise UnsafeOwnerMaskError("owner mask plan has no selected observation provenance")
    (
        component_action_bboxes_page,
        component_bboxes_page,
        owner_bbox_page,
        component_geometry_sha256,
    ) = _validated_component_geometry(
        action=action,
        component_action_bboxes_page=plan.component_action_bboxes_page,
        component_bboxes_page=plan.component_bboxes_page,
        owner_bbox_page=plan.owner_bbox_page,
        component_geometry_sha256=plan.component_geometry_sha256,
        component_geometry_verified=plan.component_geometry_verified,
    )
    expected_action_ref, expected_protected_ref = _safe_owner_ref(
        owner_id,
        page_id,
        source_sha256,
        execution_tile_id,
        action,
        protected,
        evidence_ids,
        protected_evidence_ids,
        observation_ids,
        component_action_bboxes_page,
        component_bboxes_page,
        owner_bbox_page,
        component_geometry_sha256,
        True,
    )
    if (
        plan.action_mask_ref != expected_action_ref
        or plan.protected_art_mask_ref != expected_protected_ref
    ):
        raise UnsafeOwnerMaskError(
            "owner mask reference does not match its identity and content"
        )
    return action, protected


def build_owner_mask_plan(
    original_rgb: np.ndarray,
    owner: TextOwner,
    evidence: Sequence[OwnerMaskEvidence | Mapping[str, Any]],
    *,
    owner_component_bboxes_page: Mapping[str, Sequence[int]] | None = None,
) -> OwnerMaskPlan:
    """Build a mask only from explicit owned glyph/line evidence, never a bbox."""

    owner.action_mask_ref = None
    try:
        if (
            not isinstance(original_rgb, np.ndarray)
            or original_rgb.ndim != 3
            or original_rgb.shape[0] <= 0
            or original_rgb.shape[1] <= 0
            or original_rgb.shape[2] != 3
            or original_rgb.dtype != np.uint8
        ):
            raise UnsafeOwnerMaskError(
                "original_rgb must be a non-empty RGB uint8 array"
            )
        shape = (int(original_rgb.shape[0]), int(original_rgb.shape[1]))
        owner_id = _canonical_identity(owner.owner_id, label="owner_id")
        page_id = _canonical_identity(owner.page_id, label="page_id")
        execution_tile_id = _canonical_identity(
            owner.execution_tile_id,
            label="owner executor tile_id",
        )
        if (
            owner.disposition != "owned"
            or owner.state != "translated"
            or owner.route_action not in INPAINT_ROUTE_ACTIONS
        ):
            raise UnsafeOwnerMaskError(
                "owner must be translated, owned, and routed to inpaint"
            )
        owned_components = {
            _canonical_identity(component_id, label="owner.component_id")
            for component_id in owner.component_ids
        }
        if not owned_components:
            raise UnsafeOwnerMaskError(
                "owner has no source components for safe mask evidence"
            )

        component_actions = {
            component_id: np.zeros(shape, dtype=np.uint8)
            for component_id in owned_components
        }
        protected = np.zeros(shape, dtype=np.uint8)
        used_evidence_ids: list[str] = []
        protected_evidence_ids: list[str] = []
        used_observation_ids: list[str] = []
        selected_observations = set(owner.selected_observation_ids)
        for record in evidence:
            if not isinstance(record, (OwnerMaskEvidence, Mapping)):
                raise UnsafeOwnerMaskError("owner mask evidence record is invalid")
            evidence_id_raw = _evidence_value(record, "evidence_id")
            protected_mask = _evidence_mask(
                record,
                shape,
                "protected_art_mask",
                "protected_art_polygon",
                "protected_art_polygons",
            )
            if protected_mask is not None:
                protected = np.maximum(protected, protected_mask)
                protected_evidence_ids.append(
                    _canonical_identity(
                        evidence_id_raw,
                        label="protected mask evidence_id",
                    )
                )

            glyph_positive = _evidence_mask(
                record,
                shape,
                "glyph_mask",
                "raw_glyph_mask",
            )
            line_positive = _evidence_mask(
                record,
                shape,
                "line_mask",
                "line_geometry_mask",
            )
            positive: np.ndarray | None = None
            for positive_part in (glyph_positive, line_positive):
                if positive_part is not None:
                    positive = (
                        positive_part.copy()
                        if positive is None
                        else np.maximum(positive, positive_part)
                    )
            polygon_support = _evidence_mask(
                record,
                shape,
                "glyph_polygon",
                "glyph_polygons",
                "line_polygon",
                "line_polygons",
            )
            if polygon_support is not None:
                if positive is None or not np.any(positive):
                    raise UnsafeOwnerMaskError(
                        "owner polygon evidence is support-only and requires a stroke raster"
                    )
                if np.any((positive > 0) & (polygon_support == 0)):
                    raise UnsafeOwnerMaskError(
                        "owner stroke raster escapes its polygon evidence support"
                    )
            if positive is None or not np.any(positive):
                continue
            component_ids = _evidence_component_ids(record)
            if not component_ids:
                raise UnsafeOwnerMaskError(
                    "positive mask evidence has no canonical component owner"
                )
            owned_intersection = component_ids.intersection(owned_components)
            foreign_components = component_ids - owned_components
            if owned_intersection and foreign_components:
                raise UnsafeOwnerMaskError(
                    "mixed owner mask evidence contains owned and foreign components"
                )
            evidence_id = _canonical_identity(
                evidence_id_raw,
                label="mask evidence_id",
            )
            if not owned_intersection:
                protected = np.maximum(protected, positive)
                protected_evidence_ids.append(evidence_id)
                continue
            if len(component_ids) != 1:
                raise UnsafeOwnerMaskError(
                    "owner mask evidence must be partitioned by one component"
                )
            observation_id = _evidence_value(record, "observation_id")
            if observation_id is None and len(selected_observations) == 1:
                observation_id = next(iter(selected_observations))
            observation_id = _canonical_identity(
                observation_id,
                label="mask evidence observation_id",
            )
            if (
                observation_id not in selected_observations
                or observation_id not in set(owner.observation_ids)
            ):
                raise UnsafeOwnerMaskError(
                    "owner mask evidence is not bound to a selected observation"
                )
            if (
                glyph_positive is not None
                and _positive_mask_is_overbroad(
                    glyph_positive,
                    allow_dense_single_glyph=True,
                )
            ) or (
                line_positive is not None
                and _positive_mask_is_overbroad(line_positive)
            ):
                raise UnsafeOwnerMaskError(
                    "owner positive mask is overbroad for the page"
                )
            component_id = next(iter(component_ids))
            component_actions[component_id] = np.maximum(
                component_actions[component_id],
                positive,
            )
            used_evidence_ids.append(evidence_id)
            used_observation_ids.append(observation_id)

        action = np.zeros(shape, dtype=np.uint8)
        component_action_entries: list[
            tuple[str, tuple[int, int, int, int]]
        ] = []
        missing_components: list[str] = []
        for component_id, component_mask in sorted(component_actions.items()):
            if _positive_mask_is_overbroad(
                component_mask,
                allow_dense_single_glyph=True,
            ):
                raise UnsafeOwnerMaskError(
                    "owner component mask union is overbroad for the page"
                )
            component_pixels = int(np.count_nonzero(component_mask))
            allowed_component = component_mask.copy()
            allowed_component[protected > 0] = 0
            allowed_pixels = int(np.count_nonzero(allowed_component))
            if allowed_pixels <= 0:
                missing_components.append(component_id)
                continue
            if allowed_pixels != component_pixels:
                raise UnsafeOwnerMaskError(
                    "protected art would partially authorize action-mask coverage "
                    f"for owner component: {component_id}"
                )
            component_action_entries.append(
                (component_id, _mask_bbox_page(allowed_component))
            )
            action = np.maximum(action, allowed_component)
        if missing_components:
            raise UnsafeOwnerMaskError(
                "safe mask evidence coverage is missing for owner components: "
                + ", ".join(missing_components)
            )
        if not np.any(action):
            raise UnsafeOwnerMaskError(
                "safe owner action mask evidence unavailable; bbox fallback is forbidden"
            )
        if _positive_mask_is_overbroad(action, allow_dense_single_glyph=True):
            raise UnsafeOwnerMaskError(
                "owner action mask union is overbroad for the page"
            )

        action = np.ascontiguousarray(action, dtype=np.uint8)
        protected = np.ascontiguousarray(protected, dtype=np.uint8)
        source_sha256 = _array_sha256(original_rgb)
        evidence_ids = tuple(sorted(set(used_evidence_ids)))
        protected_evidence_ids = tuple(sorted(set(protected_evidence_ids)))
        observation_ids = tuple(sorted(set(used_observation_ids)))
        component_action_bboxes_page = tuple(component_action_entries)
        if owner_component_bboxes_page is None:
            component_bboxes_page = component_action_bboxes_page
            component_geometry_verified = False
        else:
            component_bboxes_page = _canonical_component_bbox_entries(
                owner_component_bboxes_page,
                shape=shape,
                label="authoritative owner component geometry",
            )
            expected_component_ids = tuple(sorted(owned_components))
            if tuple(
                component_id for component_id, _bbox in component_bboxes_page
            ) != expected_component_ids:
                raise UnsafeOwnerMaskError(
                    "authoritative component geometry must exactly cover owner components"
                )
            component_geometry_verified = True
        owner_bbox_page = _owner_bbox_from_components(component_bboxes_page)
        component_geometry_sha256 = _component_geometry_sha256(
            component_bboxes_page
        )
        if component_geometry_verified:
            (
                component_action_bboxes_page,
                component_bboxes_page,
                owner_bbox_page,
                component_geometry_sha256,
            ) = _validated_component_geometry(
                action=action,
                component_action_bboxes_page=component_action_bboxes_page,
                component_bboxes_page=component_bboxes_page,
                owner_bbox_page=owner_bbox_page,
                component_geometry_sha256=component_geometry_sha256,
                component_geometry_verified=True,
            )
        action_mask_ref, protected_art_mask_ref = _safe_owner_ref(
            owner_id,
            page_id,
            source_sha256,
            execution_tile_id,
            action,
            protected,
            evidence_ids,
            protected_evidence_ids,
            observation_ids,
            component_action_bboxes_page,
            component_bboxes_page,
            owner_bbox_page,
            component_geometry_sha256,
            component_geometry_verified,
        )
        action.setflags(write=False)
        protected.setflags(write=False)
        return OwnerMaskPlan(
            owner_id=owner_id,
            page_id=page_id,
            action_mask_ref=action_mask_ref,
            action_mask=action,
            protected_art_mask=protected,
            evidence_ids=evidence_ids,
            source_sha256=source_sha256,
            execution_tile_id=execution_tile_id,
            protected_art_mask_ref=protected_art_mask_ref,
            protected_evidence_ids=protected_evidence_ids,
            observation_ids=observation_ids,
            component_action_bboxes_page=component_action_bboxes_page,
            component_bboxes_page=component_bboxes_page,
            owner_bbox_page=owner_bbox_page,
            component_geometry_sha256=component_geometry_sha256,
            component_geometry_verified=component_geometry_verified,
        )
    except Exception as exc:
        _mark_owner_mask_review(owner)
        if isinstance(exc, UnsafeOwnerMaskError):
            raise
        raise UnsafeOwnerMaskError("owner mask planning failed closed") from exc


def _safe_artifact_path(root: Path, relative_ref: str) -> Path:
    root = Path(root).resolve()
    candidate = (root / Path(relative_ref)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise UnsafeOwnerMaskError(
            "owner mask reference escapes artifact root"
        ) from exc
    return candidate


def _write_mask_file(path: Path, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8)):
        raise OSError(f"failed to persist owner mask: {path}")


def _mask_manifest(plan: OwnerMaskPlan) -> dict[str, Any]:
    protected_ref = plan.protected_art_mask_ref or str(
        Path(plan.action_mask_ref).with_name("protected_art_mask.png")
    ).replace("\\", "/")
    return {
        "schema_version": OWNER_MASK_SCHEMA_VERSION,
        "coordinate_space": plan.coordinate_space,
        "owner_id": plan.owner_id,
        "page_id": plan.page_id,
        "execution_tile_id": plan.execution_tile_id,
        "action_mask_ref": plan.action_mask_ref,
        "protected_art_mask_ref": protected_ref,
        "source_sha256": plan.source_sha256,
        "action_mask_sha256": _array_sha256(plan.action_mask),
        "action_mask_shape": list(plan.action_mask.shape),
        "action_mask_dtype": str(plan.action_mask.dtype),
        "protected_art_mask_sha256": _array_sha256(plan.protected_art_mask),
        "protected_art_mask_shape": list(plan.protected_art_mask.shape),
        "protected_art_mask_dtype": str(plan.protected_art_mask.dtype),
        "evidence_ids": list(plan.evidence_ids),
        "protected_evidence_ids": list(plan.protected_evidence_ids),
        "observation_ids": list(plan.observation_ids),
        "component_action_bboxes_page": [
            [component_id, list(bbox)]
            for component_id, bbox in plan.component_action_bboxes_page
        ],
        "component_bboxes_page": [
            [component_id, list(bbox)]
            for component_id, bbox in plan.component_bboxes_page
        ],
        "owner_bbox_page": list(plan.owner_bbox_page),
        "component_geometry_sha256": plan.component_geometry_sha256,
        "component_geometry_verified": plan.component_geometry_verified,
    }


def _validate_persisted_mask_pair(plan: OwnerMaskPlan, directory: Path) -> None:
    action_path = directory / "action_mask.png"
    protected_path = directory / "protected_art_mask.png"
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeOwnerMaskError("owner mask manifest is missing or invalid") from exc
    if not isinstance(manifest, dict):
        raise UnsafeOwnerMaskError("owner mask manifest is missing or invalid")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != OWNER_MASK_SCHEMA_VERSION
    ):
        raise UnsafeOwnerMaskError("owner mask manifest version is unsupported")
    action = cv2.imread(str(action_path), cv2.IMREAD_GRAYSCALE)
    protected = cv2.imread(str(protected_path), cv2.IMREAD_GRAYSCALE)
    if action is None or protected is None:
        raise UnsafeOwnerMaskError("owner action/protected mask pair is incomplete")
    if not np.all((action == 0) | (action == 255)) or not np.all(
        (protected == 0) | (protected == 255)
    ):
        raise UnsafeOwnerMaskError(
            "persisted owner mask PNGs must contain canonical binary pixels"
        )
    if (
        action.shape != plan.action_mask.shape
        or protected.shape != plan.protected_art_mask.shape
    ):
        raise UnsafeOwnerMaskError("persisted owner mask pair shape mismatch")
    expected = _mask_manifest(plan)
    for key in (
        "schema_version",
        "coordinate_space",
        "owner_id",
        "page_id",
        "execution_tile_id",
        "action_mask_ref",
        "protected_art_mask_ref",
        "source_sha256",
        "action_mask_sha256",
        "action_mask_shape",
        "action_mask_dtype",
        "protected_art_mask_sha256",
        "protected_art_mask_shape",
        "protected_art_mask_dtype",
        "evidence_ids",
        "protected_evidence_ids",
        "observation_ids",
        "component_action_bboxes_page",
        "component_bboxes_page",
        "owner_bbox_page",
        "component_geometry_sha256",
        "component_geometry_verified",
    ):
        if manifest.get(key) != expected.get(key):
            raise UnsafeOwnerMaskError(f"persisted owner mask manifest mismatch: {key}")
    if (
        _array_sha256(np.ascontiguousarray(action, dtype=np.uint8))
        != expected["action_mask_sha256"]
    ):
        raise UnsafeOwnerMaskError("persisted owner action mask hash mismatch")
    if (
        _array_sha256(np.ascontiguousarray(protected, dtype=np.uint8))
        != expected["protected_art_mask_sha256"]
    ):
        raise UnsafeOwnerMaskError("persisted protected-art mask hash mismatch")


def persist_owner_mask_plan(
    plan: OwnerMaskPlan,
    artifact_root: Path,
    *,
    owner: TextOwner | None = None,
) -> Path:
    """Atomically publish a lossless action/protected mask pair and manifest."""

    temporary_directory: Path | None = None
    try:
        _validated_plan_masks(plan)
        action_path = _safe_artifact_path(Path(artifact_root), plan.action_mask_ref)
        protected_ref = plan.protected_art_mask_ref or str(
            Path(plan.action_mask_ref).with_name("protected_art_mask.png")
        ).replace("\\", "/")
        protected_path = _safe_artifact_path(Path(artifact_root), protected_ref)
        if action_path.parent != protected_path.parent:
            raise UnsafeOwnerMaskError(
                "owner mask pair must share one artifact directory"
            )
        if owner is not None:
            if (
                owner.owner_id != plan.owner_id
                or owner.page_id != plan.page_id
                or owner.execution_tile_id != plan.execution_tile_id
                or owner.disposition != "owned"
                or owner.state != "translated"
                or owner.route_action not in INPAINT_ROUTE_ACTIONS
            ):
                raise UnsafeOwnerMaskError(
                    "owner mask plan identity or state does not match owner"
                )
            owner_component_ids = tuple(
                sorted(
                    _canonical_identity(
                        component_id,
                        label="owner.component_id",
                    )
                    for component_id in owner.component_ids
                )
            )
            plan_component_ids = tuple(
                component_id for component_id, _bbox in plan.component_bboxes_page
            )
            if (
                len(set(owner_component_ids)) != len(owner_component_ids)
                or owner_component_ids != plan_component_ids
            ):
                raise UnsafeOwnerMaskError(
                    "owner component geometry does not match current owner components"
                )
            owner_observation_ids = tuple(
                sorted(
                    _canonical_identity(
                        observation_id,
                        label="owner.observation_id",
                    )
                    for observation_id in owner.observation_ids
                )
            )
            selected_observation_ids = tuple(
                sorted(
                    _canonical_identity(
                        observation_id,
                        label="owner.selected_observation_id",
                    )
                    for observation_id in owner.selected_observation_ids
                )
            )
            if (
                len(set(owner_observation_ids)) != len(owner_observation_ids)
                or len(set(selected_observation_ids))
                != len(selected_observation_ids)
                or not set(selected_observation_ids).issubset(
                    owner_observation_ids
                )
                or selected_observation_ids != plan.observation_ids
            ):
                raise UnsafeOwnerMaskError(
                    "owner mask observation provenance does not match current owner"
                )

        target_directory = action_path.parent
        target_directory.parent.mkdir(parents=True, exist_ok=True)
        if not target_directory.exists():
            temporary_directory = Path(
                tempfile.mkdtemp(
                    prefix=f".{target_directory.name}.tmp-",
                    dir=str(target_directory.parent),
                )
            )
            _write_mask_file(
                temporary_directory / "action_mask.png",
                plan.action_mask,
            )
            _write_mask_file(
                temporary_directory / "protected_art_mask.png",
                plan.protected_art_mask,
            )
            (temporary_directory / "manifest.json").write_text(
                json.dumps(_mask_manifest(plan), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _validate_persisted_mask_pair(plan, temporary_directory)
            try:
                temporary_directory.replace(target_directory)
                temporary_directory = None
            except OSError:
                if not target_directory.exists():
                    raise
        _validate_persisted_mask_pair(plan, target_directory)
        if owner is not None:
            owner.action_mask_ref = plan.action_mask_ref
            owner.state = "mask_ready"
        return action_path
    except Exception:
        if owner is not None:
            _mark_owner_mask_review(owner)
        raise
    finally:
        if temporary_directory is not None and temporary_directory.exists():
            shutil.rmtree(temporary_directory, ignore_errors=True)


def load_owner_action_mask(
    artifact_root: Path,
    action_mask_ref: str,
    *,
    expected_owner_id: str,
    expected_page_id: str,
    expected_source_sha256: str,
    expected_execution_tile_id: str,
    expected_component_geometry_sha256: str,
) -> np.ndarray:
    """Load an addressable lossless owner action mask."""

    path = _safe_artifact_path(Path(artifact_root), action_mask_ref)
    manifest_path = path.with_name("manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise UnsafeOwnerMaskError("owner action mask manifest is missing") from exc
    if not isinstance(manifest, dict):
        raise UnsafeOwnerMaskError("owner action mask manifest is invalid")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != OWNER_MASK_SCHEMA_VERSION
    ):
        raise UnsafeOwnerMaskError("owner action mask manifest version is unsupported")
    if manifest.get("coordinate_space") != OWNER_MASK_COORDINATE_SPACE:
        raise UnsafeOwnerMaskError(
            "owner action mask coordinate space must be canonical page-space"
        )
    if manifest.get("action_mask_ref") != action_mask_ref:
        raise UnsafeOwnerMaskError("owner action mask reference mismatch")
    manifest_owner_id = _canonical_identity(
        manifest.get("owner_id"),
        label="owner mask manifest owner_id",
    )
    manifest_page_id = _canonical_identity(
        manifest.get("page_id"),
        label="owner mask manifest page_id",
    )
    ref_parts = Path(action_mask_ref.replace("\\", "/")).parts
    if (
        len(ref_parts) != 4
        or ref_parts[0] != "owner_masks"
        or ref_parts[1] != _owner_artifact_segment(manifest_owner_id)
        or ref_parts[3] != "action_mask.png"
    ):
        raise UnsafeOwnerMaskError("owner action mask reference has wrong owner scope")
    expected_owner_id = _canonical_identity(
        expected_owner_id,
        label="expected owner_id",
    )
    expected_page_id = _canonical_identity(
        expected_page_id,
        label="expected page_id",
    )
    expected_source_sha256 = _canonical_identity(
        expected_source_sha256,
        label="expected source_sha256",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
        raise UnsafeOwnerMaskError("expected source hash is not canonical")
    expected_execution_tile_id = _canonical_identity(
        expected_execution_tile_id,
        label="expected executor tile_id",
    )
    expected_component_geometry_sha256 = _canonical_identity(
        expected_component_geometry_sha256,
        label="expected component geometry hash",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_component_geometry_sha256):
        raise UnsafeOwnerMaskError(
            "expected component geometry hash is not canonical"
        )
    if manifest_owner_id != expected_owner_id:
        raise UnsafeOwnerMaskError("owner action mask belongs to another owner")
    if manifest_page_id != expected_page_id:
        raise UnsafeOwnerMaskError("owner action mask belongs to another page")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise UnsafeOwnerMaskError(f"owner action mask is missing: {action_mask_ref}")
    if not np.all((mask == 0) | (mask == 255)):
        raise UnsafeOwnerMaskError(
            "owner action mask PNG must contain canonical binary pixels"
        )
    normalized = np.ascontiguousarray(mask, dtype=np.uint8)
    if list(normalized.shape) != manifest.get("action_mask_shape") or str(
        normalized.dtype
    ) != manifest.get("action_mask_dtype"):
        raise UnsafeOwnerMaskError("owner action mask shape or dtype mismatch")
    if _array_sha256(normalized) != manifest.get("action_mask_sha256"):
        raise UnsafeOwnerMaskError("owner action mask hash mismatch")
    protected_ref = manifest.get("protected_art_mask_ref")
    if not isinstance(protected_ref, str) or not protected_ref:
        raise UnsafeOwnerMaskError("protected-art mask reference is missing")
    protected_path = _safe_artifact_path(Path(artifact_root), protected_ref)
    if protected_path.parent != path.parent:
        raise UnsafeOwnerMaskError("owner mask pair does not share one directory")
    protected = cv2.imread(str(protected_path), cv2.IMREAD_GRAYSCALE)
    if protected is None:
        raise UnsafeOwnerMaskError("protected-art mask is missing")
    if not np.all((protected == 0) | (protected == 255)):
        raise UnsafeOwnerMaskError(
            "protected-art mask PNG must contain canonical binary pixels"
        )
    protected_normalized = np.ascontiguousarray(protected, dtype=np.uint8)
    if list(protected_normalized.shape) != manifest.get(
        "protected_art_mask_shape"
    ) or str(protected_normalized.dtype) != manifest.get("protected_art_mask_dtype"):
        raise UnsafeOwnerMaskError("protected-art mask shape or dtype mismatch")
    if _array_sha256(protected_normalized) != manifest.get("protected_art_mask_sha256"):
        raise UnsafeOwnerMaskError("protected-art mask hash mismatch")
    if not np.any(normalized):
        raise UnsafeOwnerMaskError("owner action mask is empty")
    if np.any((normalized > 0) & (protected_normalized > 0)):
        raise UnsafeOwnerMaskError("owner action mask overlaps protected art")
    if _positive_mask_is_overbroad(normalized, allow_dense_single_glyph=True):
        raise UnsafeOwnerMaskError("owner action mask is overbroad for the page")
    manifest_evidence_ids = _canonical_evidence_ids(
        manifest.get("evidence_ids"),
        label="owner mask manifest evidence_ids",
    )
    if not manifest_evidence_ids:
        raise UnsafeOwnerMaskError("owner mask manifest has no positive evidence ids")
    manifest_protected_evidence_ids = _canonical_evidence_ids(
        manifest.get("protected_evidence_ids"),
        label="owner mask manifest protected_evidence_ids",
    )
    manifest_observation_ids = _canonical_evidence_ids(
        manifest.get("observation_ids"),
        label="owner mask manifest observation_ids",
    )
    if not manifest_observation_ids:
        raise UnsafeOwnerMaskError(
            "owner mask manifest has no selected observation provenance"
        )
    (
        manifest_component_action_bboxes_page,
        manifest_component_bboxes_page,
        manifest_owner_bbox_page,
        manifest_component_geometry_sha256,
    ) = _validated_component_geometry(
        action=normalized,
        component_action_bboxes_page=manifest.get(
            "component_action_bboxes_page"
        ),
        component_bboxes_page=manifest.get("component_bboxes_page"),
        owner_bbox_page=manifest.get("owner_bbox_page"),
        component_geometry_sha256=manifest.get("component_geometry_sha256"),
        component_geometry_verified=manifest.get("component_geometry_verified"),
    )
    manifest_source_sha256 = _canonical_identity(
        manifest.get("source_sha256"),
        label="owner mask manifest source_sha256",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_source_sha256):
        raise UnsafeOwnerMaskError("owner mask manifest source hash is not canonical")
    if manifest_source_sha256 != expected_source_sha256:
        raise UnsafeOwnerMaskError("owner action mask belongs to another source page")
    manifest_execution_tile_id = _canonical_identity(
        manifest.get("execution_tile_id"),
        label="owner mask manifest executor tile_id",
    )
    if manifest_execution_tile_id != expected_execution_tile_id:
        raise UnsafeOwnerMaskError("owner action mask belongs to another executor tile")
    if manifest_component_geometry_sha256 != expected_component_geometry_sha256:
        raise UnsafeOwnerMaskError(
            "owner action mask belongs to another component geometry revision"
        )
    expected_action_ref, expected_protected_ref = _safe_owner_ref(
        manifest_owner_id,
        manifest_page_id,
        manifest_source_sha256,
        manifest_execution_tile_id,
        normalized,
        protected_normalized,
        manifest_evidence_ids,
        manifest_protected_evidence_ids,
        manifest_observation_ids,
        manifest_component_action_bboxes_page,
        manifest_component_bboxes_page,
        manifest_owner_bbox_page,
        manifest_component_geometry_sha256,
        True,
    )
    if (
        action_mask_ref != expected_action_ref
        or protected_ref != expected_protected_ref
    ):
        raise UnsafeOwnerMaskError(
            "owner mask artifact path does not match manifest content"
        )
    normalized.setflags(write=False)
    return normalized


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(b"traduzai.ndarray.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(dimension) for dimension in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _engine_crop_bbox_page(action_mask: np.ndarray) -> tuple[int, int, int, int]:
    action_y, action_x = np.nonzero(action_mask)
    if action_x.size <= 0:
        raise UnsafeOwnerMaskError("owner action mask is empty")
    tight_x1 = int(action_x.min())
    tight_y1 = int(action_y.min())
    tight_x2 = int(action_x.max()) + 1
    tight_y2 = int(action_y.max()) + 1
    tight_width = tight_x2 - tight_x1
    tight_height = tight_y2 - tight_y1
    context_padding = max(16, min(96, int(math.ceil(max(tight_width, tight_height) * 0.5))))
    height, width = action_mask.shape
    return (
        max(0, tight_x1 - context_padding),
        max(0, tight_y1 - context_padding),
        min(width, tight_x2 + context_padding),
        min(height, tight_y2 + context_padding),
    )


def execute_owner_inpaint(
    original_rgb: np.ndarray,
    plan: OwnerMaskPlan,
    inpainter: Any,
) -> OwnerMutation:
    """Run one engine and clamp its result to the owner's safe action mask."""

    action_raw, protected_raw = _validated_plan_masks(plan)
    original = np.ascontiguousarray(original_rgb).copy()
    if (
        original.ndim != 3
        or original.shape[2] != 3
        or original.dtype != np.uint8
        or original.shape[:2] != plan.action_mask.shape
        or original.shape[:2] != plan.protected_art_mask.shape
    ):
        raise UnsafeOwnerMaskError("owner plan shape does not match original image")
    if not plan.source_sha256 or _array_sha256(original) != plan.source_sha256:
        raise UnsafeOwnerMaskError(
            "owner mask plan source hash does not match the input image"
        )
    action = action_raw > 0
    protected = protected_raw > 0
    allowed = action & ~protected
    if not np.any(allowed):
        raise UnsafeOwnerMaskError("owner action mask is empty after art protection")
    inpaint = getattr(inpainter, "inpaint", None)
    if not callable(inpaint):
        raise UnsafeOwnerMaskError(
            "owner inpaint engine has no callable inpaint method"
        )
    crop_x1, crop_y1, crop_x2, crop_y2 = _engine_crop_bbox_page(allowed)
    original_crop = original[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    allowed_crop = allowed[crop_y1:crop_y2, crop_x1:crop_x2]
    candidate = inpaint(
        original_crop,
        np.where(allowed_crop, 255, 0).astype(np.uint8),
        batch_size=1,
        force_no_tiling=False,
    )
    candidate = np.asarray(candidate)
    if candidate.shape != original_crop.shape:
        raise UnsafeOwnerMaskError(
            "owner inpaint engine returned shape "
            f"{candidate.shape}, expected crop shape {original_crop.shape}"
        )
    if candidate.dtype != np.uint8:
        raise UnsafeOwnerMaskError(
            "owner inpaint engine must return a canonical RGB uint8 array"
        )

    result = original.copy()
    result_crop = result[crop_y1:crop_y2, crop_x1:crop_x2]
    result_crop[allowed_crop] = candidate[allowed_crop]
    changed = np.any(result != original, axis=2)
    if not np.any(changed):
        raise UnsafeOwnerMaskError(
            "owner inpaint engine was a no-op and changed no pixels"
        )
    changed_mask = np.where(changed, 255, 0).astype(np.uint8)
    outside = changed & ~allowed
    protected_changed = changed & protected
    engine = str(
        getattr(inpainter, "engine_name", None)
        or getattr(inpainter, "name", None)
        or inpainter.__class__.__name__
    )
    action_mask = np.where(allowed, 255, 0).astype(np.uint8)
    protected_mask = np.where(protected, 255, 0).astype(np.uint8)
    for array in (result, action_mask, protected_mask, changed_mask):
        array.setflags(write=False)
    return OwnerMutation(
        owner_id=plan.owner_id,
        page_id=plan.page_id,
        coordinate_space=plan.coordinate_space,
        action_mask_ref=plan.action_mask_ref,
        result_rgb=result,
        action_mask=action_mask,
        protected_art_mask=protected_mask,
        changed_mask=changed_mask,
        engine=engine,
        mask_pixels=int(np.count_nonzero(allowed)),
        changed_pixels=int(np.count_nonzero(changed)),
        changed_outside_owner_pixels=int(np.count_nonzero(outside)),
        protected_art_changed_pixels=int(np.count_nonzero(protected_changed)),
        before_sha256=_array_sha256(original),
        after_sha256=_array_sha256(result),
        action_mask_sha256=_array_sha256(action_mask),
        changed_mask_sha256=_array_sha256(changed_mask),
        engine_crop_bbox_page=(crop_x1, crop_y1, crop_x2, crop_y2),
        owner_bbox_page=plan.owner_bbox_page,
        component_geometry_sha256=plan.component_geometry_sha256,
        residual_score=None,
        execution_tile_id=plan.execution_tile_id,
    )
