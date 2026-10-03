"""Immutable page-space geometry authorized for one text owner render.

Cleanup geometry and layout geometry are deliberately separate.  The former
describes source pixels that may be replaced; the latter must come from
independent visual-container evidence (except for explicitly typed freeform
text such as SFX).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .model import BBox, OwnerGraph, Point


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIALOGUE_ROLE_TOKENS = ("dialogue", "speech", "thought", "balloon")
_FREEFORM_ROLE_TOKENS = ("sfx", "freeform", "sound_effect", "onomatop")
_CARD_ROLE_TOKENS = ("card", "ui", "system", "status", "interface")


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _check_sha256(value: str, label: str) -> str:
    canonical = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(canonical):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return canonical


def _ndarray_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(b"traduzai.ndarray.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(dimension) for dimension in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _bbox(value: Sequence[Any], *, width: int, height: int, label: str) -> BBox:
    if isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    result = tuple(int(item) for item in value)
    x1, y1, x2, y2 = result
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise ValueError(f"{label} is outside logical page bounds")
    return result  # type: ignore[return-value]


def _polygon(
    value: Sequence[Sequence[Any]], *, width: int, height: int, label: str
) -> tuple[Point, ...]:
    if isinstance(value, (str, bytes)) or len(value) < 3:
        raise ValueError(f"{label} requires at least three points")
    points = tuple((int(point[0]), int(point[1])) for point in value)
    if len(set(points)) < 3:
        raise ValueError(f"{label} is degenerate")
    if any(x < 0 or y < 0 or x > width or y > height for x, y in points):
        raise ValueError(f"{label} is outside logical page bounds")
    area2 = sum(
        (points[index][0] * points[(index + 1) % len(points)][1])
        - (points[(index + 1) % len(points)][0] * points[index][1])
        for index in range(len(points))
    )
    if area2 == 0:
        raise ValueError(f"{label} has zero area")
    # Rotation and winding must not affect the authenticated geometry.
    forward = points if area2 > 0 else tuple(reversed(points))
    start = min(range(len(forward)), key=lambda index: forward[index])
    return forward[start:] + forward[:start]


def _rect_polygon(bbox: BBox) -> tuple[Point, ...]:
    x1, y1, x2, y2 = bbox
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _union_bbox(boxes: Sequence[BBox]) -> BBox:
    if not boxes:
        raise ValueError("geometry union requires at least one bbox")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _consensus_bbox(boxes: Sequence[BBox]) -> BBox:
    """Return the majority OCR envelope without one coarse crop dominating it."""

    if len(boxes) < 3:
        return _union_bbox(boxes)
    ordered = [sorted(box[index] for box in boxes) for index in range(4)]
    median = tuple(values[len(values) // 2] for values in ordered)
    median_area = max(1, (median[2] - median[0]) * (median[3] - median[1]))
    inliers: list[BBox] = []
    for box in boxes:
        ix1, iy1 = max(box[0], median[0]), max(box[1], median[1])
        ix2, iy2 = min(box[2], median[2]), min(box[3], median[3])
        intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        union_area = box_area + median_area - intersection_area
        if intersection_area / max(1, union_area) >= 0.6:
            inliers.append(box)
    if len(inliers) * 2 <= len(boxes):
        return _union_bbox(boxes)
    return _union_bbox(inliers)


def _polygon_bbox(points: Sequence[Point]) -> BBox:
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _intersection(left: BBox, right: BBox) -> BBox:
    candidate = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return candidate if candidate[2] > candidate[0] and candidate[3] > candidate[1] else left


@dataclass(frozen=True)
class OwnerComponentGeometry:
    component_id: str
    bbox_page: BBox
    polygon_page: tuple[Point, ...]
    geometry_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "bbox_page": list(self.bbox_page),
            "polygon_page": [list(point) for point in self.polygon_page],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "geometry_sha256": self.geometry_sha256}


@dataclass(frozen=True)
class OwnerObservationGeometry:
    observation_id: str
    source_sha256: str
    bbox_page: BBox
    polygon_page: tuple[Point, ...]
    confidence: float
    observation_geometry_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source_sha256": self.source_sha256,
            "bbox_page": list(self.bbox_page),
            "polygon_page": [list(point) for point in self.polygon_page],
            "confidence": self.confidence,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "observation_geometry_sha256": self.observation_geometry_sha256}


@dataclass(frozen=True)
class OwnerProjectionGeometry:
    projection_id: str
    tile_id: str
    role: str
    tile_bbox_page: BBox
    component_bbox_page: BBox
    tile_to_page_offset_xy: Point
    projection_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "tile_id": self.tile_id,
            "role": self.role,
            "tile_bbox_page": list(self.tile_bbox_page),
            "component_bbox_page": list(self.component_bbox_page),
            "tile_to_page_offset_xy": list(self.tile_to_page_offset_xy),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "projection_sha256": self.projection_sha256}


@dataclass(frozen=True)
class ConnectedLayoutSubregion:
    subregion_id: str
    order: int
    component_ids: tuple[str, ...]
    bbox_page: BBox
    polygon_page: tuple[Point, ...]
    capacity_px2: int
    evidence_source: str
    evidence_ids: tuple[str, ...]
    subregion_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "subregion_id": self.subregion_id,
            "order": self.order,
            "component_ids": list(self.component_ids),
            "bbox_page": list(self.bbox_page),
            "polygon_page": [list(point) for point in self.polygon_page],
            "capacity_px2": self.capacity_px2,
            "evidence_source": self.evidence_source,
            "evidence_ids": list(self.evidence_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "subregion_sha256": self.subregion_sha256}


@dataclass(frozen=True)
class OwnerRenderGeometry:
    schema_version: int
    owner_id: str
    page_id: str
    logical_space: Literal["logical_page"]
    page_width: int
    page_height: int
    component_ids: tuple[str, ...]
    components: tuple[OwnerComponentGeometry, ...]
    component_geometry_sha256: str
    selected_observations: tuple[OwnerObservationGeometry, ...]
    projections: tuple[OwnerProjectionGeometry, ...]
    semantic_body_bbox_page: BBox
    source_replacement_bbox_page: BBox
    layout_container_bbox_page: BBox | None
    layout_container_polygon_page: tuple[Point, ...] | None
    layout_container_source: str
    safe_polygons_page: tuple[tuple[Point, ...], ...]
    connected_subregions: tuple[ConnectedLayoutSubregion, ...]
    container_evidence_ids: tuple[str, ...]
    container_evidence_confidence: float
    protected_art_mask_sha256: str
    status: Literal["ready", "review_required"]
    reason: str
    geometry_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner_id": self.owner_id,
            "page_id": self.page_id,
            "logical_space": self.logical_space,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "component_ids": list(self.component_ids),
            "components": [item.to_dict() for item in self.components],
            "component_geometry_sha256": self.component_geometry_sha256,
            "selected_observations": [item.to_dict() for item in self.selected_observations],
            "projections": [item.to_dict() for item in self.projections],
            "semantic_body_bbox_page": list(self.semantic_body_bbox_page),
            "source_replacement_bbox_page": list(self.source_replacement_bbox_page),
            "layout_container_bbox_page": list(self.layout_container_bbox_page) if self.layout_container_bbox_page else None,
            "layout_container_polygon_page": [list(point) for point in self.layout_container_polygon_page] if self.layout_container_polygon_page else None,
            "layout_container_source": self.layout_container_source,
            "safe_polygons_page": [[list(point) for point in polygon] for polygon in self.safe_polygons_page],
            "connected_subregions": [item.to_dict() for item in self.connected_subregions],
            "container_evidence_ids": list(self.container_evidence_ids),
            "container_evidence_confidence": self.container_evidence_confidence,
            "protected_art_mask_sha256": self.protected_art_mask_sha256,
            "status": self.status,
            "reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "geometry_sha256": self.geometry_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OwnerRenderGeometry":
        width = int(value.get("page_width") or 0)
        height = int(value.get("page_height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError("owner render geometry requires positive page dimensions")

        components = []
        for raw in value.get("components") or ():
            item = OwnerComponentGeometry(
                component_id=str(raw["component_id"]),
                bbox_page=_bbox(raw["bbox_page"], width=width, height=height, label="component bbox"),
                polygon_page=_polygon(raw["polygon_page"], width=width, height=height, label="component polygon"),
                geometry_sha256=_check_sha256(raw["geometry_sha256"], "component geometry hash"),
            )
            if _json_sha256(item.payload()) != item.geometry_sha256:
                raise ValueError("component geometry hash mismatch")
            components.append(item)

        observations = []
        for raw in value.get("selected_observations") or ():
            item = OwnerObservationGeometry(
                observation_id=str(raw["observation_id"]),
                source_sha256=_check_sha256(raw["source_sha256"], "observation source hash"),
                bbox_page=_bbox(raw["bbox_page"], width=width, height=height, label="observation bbox"),
                polygon_page=_polygon(raw["polygon_page"], width=width, height=height, label="observation polygon"),
                confidence=float(raw["confidence"]),
                observation_geometry_sha256=_check_sha256(raw["observation_geometry_sha256"], "observation geometry hash"),
            )
            if _json_sha256(item.payload()) != item.observation_geometry_sha256:
                raise ValueError("observation geometry hash mismatch")
            observations.append(item)

        projections = []
        for raw in value.get("projections") or ():
            item = OwnerProjectionGeometry(
                projection_id=str(raw["projection_id"]), tile_id=str(raw["tile_id"]), role=str(raw["role"]),
                tile_bbox_page=_bbox(raw["tile_bbox_page"], width=width, height=height, label="projection tile bbox"),
                component_bbox_page=_bbox(raw["component_bbox_page"], width=width, height=height, label="projection component bbox"),
                tile_to_page_offset_xy=tuple(int(item) for item in raw["tile_to_page_offset_xy"]),  # type: ignore[arg-type]
                projection_sha256=_check_sha256(raw["projection_sha256"], "projection hash"),
            )
            if _json_sha256(item.payload()) != item.projection_sha256:
                raise ValueError("projection hash mismatch")
            projections.append(item)

        subregions = []
        for raw in value.get("connected_subregions") or ():
            item = ConnectedLayoutSubregion(
                subregion_id=str(raw["subregion_id"]), order=int(raw["order"]),
                component_ids=tuple(str(item) for item in raw["component_ids"]),
                bbox_page=_bbox(raw["bbox_page"], width=width, height=height, label="subregion bbox"),
                polygon_page=_polygon(raw["polygon_page"], width=width, height=height, label="subregion polygon"),
                capacity_px2=int(raw["capacity_px2"]), evidence_source=str(raw["evidence_source"]),
                evidence_ids=tuple(str(item) for item in raw["evidence_ids"]),
                subregion_sha256=_check_sha256(raw["subregion_sha256"], "subregion hash"),
            )
            if _json_sha256(item.payload()) != item.subregion_sha256:
                raise ValueError("subregion hash mismatch")
            subregions.append(item)

        layout_bbox = value.get("layout_container_bbox_page")
        layout_polygon = value.get("layout_container_polygon_page")
        geometry = cls(
            schema_version=int(value["schema_version"]), owner_id=str(value["owner_id"]), page_id=str(value["page_id"]),
            logical_space=str(value["logical_space"]),  # type: ignore[arg-type]
            page_width=width, page_height=height,
            component_ids=tuple(str(item) for item in value["component_ids"]), components=tuple(components),
            component_geometry_sha256=_check_sha256(value["component_geometry_sha256"], "component geometry hash"),
            selected_observations=tuple(observations), projections=tuple(projections),
            semantic_body_bbox_page=_bbox(value["semantic_body_bbox_page"], width=width, height=height, label="semantic body bbox"),
            source_replacement_bbox_page=_bbox(value["source_replacement_bbox_page"], width=width, height=height, label="source replacement bbox"),
            layout_container_bbox_page=_bbox(layout_bbox, width=width, height=height, label="layout container bbox") if layout_bbox else None,
            layout_container_polygon_page=_polygon(layout_polygon, width=width, height=height, label="layout container polygon") if layout_polygon else None,
            layout_container_source=str(value["layout_container_source"]),
            safe_polygons_page=tuple(_polygon(item, width=width, height=height, label="safe polygon") for item in value.get("safe_polygons_page") or ()),
            connected_subregions=tuple(subregions),
            container_evidence_ids=tuple(str(item) for item in value.get("container_evidence_ids") or ()),
            container_evidence_confidence=float(value.get("container_evidence_confidence") or 0.0),
            protected_art_mask_sha256=_check_sha256(value["protected_art_mask_sha256"], "protected art mask hash"),
            status=str(value["status"]), reason=str(value.get("reason") or ""),  # type: ignore[arg-type]
            geometry_sha256=_check_sha256(value["geometry_sha256"], "geometry hash"),
        )
        geometry._validate()
        return geometry

    def _validate(self) -> None:
        if self.schema_version != 1 or self.logical_space != "logical_page":
            raise ValueError("owner render geometry requires schema v1 logical_page")
        if self.component_ids != tuple(item.component_id for item in self.components):
            raise ValueError("owner render geometry component identities mismatch")
        expected_component_hash = _json_sha256([item.to_dict() for item in self.components])
        if expected_component_hash != self.component_geometry_sha256:
            raise ValueError("component geometry hash mismatch")
        if self.status not in {"ready", "review_required"}:
            raise ValueError("owner render geometry status is invalid")
        if self.status == "ready" and self.layout_container_bbox_page is None:
            raise ValueError("ready owner render geometry requires a layout container")
        if _json_sha256(self.payload()) != self.geometry_sha256:
            raise ValueError("owner render geometry hash mismatch")


def _hashed_component(component: Any, *, width: int, height: int) -> OwnerComponentGeometry:
    item = OwnerComponentGeometry(
        component_id=str(component.component_id),
        bbox_page=_bbox(component.bbox_page, width=width, height=height, label="component bbox"),
        polygon_page=_polygon(component.polygon_page, width=width, height=height, label="component polygon"),
        geometry_sha256="",
    )
    return replace(item, geometry_sha256=_json_sha256(item.payload()))


def _hashed_observation(observation: Any, *, width: int, height: int) -> OwnerObservationGeometry:
    polygons = tuple(
        _polygon(item, width=width, height=height, label="observation polygon")
        for item in observation.polygons_page
    )
    raw_bbox = _bbox(
        observation.bbox_page,
        width=width,
        height=height,
        label="observation bbox",
    )
    bbox = _bbox(
        observation.text_pixel_bbox_page or raw_bbox,
        width=width,
        height=height,
        label="observation text pixel bbox",
    )
    polygon = _rect_polygon(_union_bbox(tuple(_polygon_bbox(item) for item in polygons))) if polygons else _rect_polygon(bbox)
    source_sha = _json_sha256({
        "observation_id": str(observation.observation_id), "text": str(observation.text),
        "provider": str(observation.provider), "component_ids": sorted(str(item) for item in observation.component_ids),
        "bbox_page": list(raw_bbox), "text_pixel_bbox_page": list(bbox),
        "polygons_page": [[[x, y] for x, y in item] for item in polygons],
    })
    item = OwnerObservationGeometry(str(observation.observation_id), source_sha, bbox, polygon, round(float(observation.confidence), 6), "")
    return replace(item, observation_geometry_sha256=_json_sha256(item.payload()))


def _container_candidates(value: Any, *, width: int, height: int) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)) and not value:
        return ()
    raw_items = value if isinstance(value, (list, tuple)) and value and isinstance(value[0], Mapping) else (value,)
    candidates = []
    for index, raw in enumerate(raw_items):
        if isinstance(raw, Mapping):
            polygon_raw = raw.get("polygon_page")
            bbox_raw = raw.get("bbox_page")
            if polygon_raw is None and bbox_raw is not None:
                polygon_raw = _rect_polygon(tuple(int(item) for item in bbox_raw))
            polygon = _polygon(polygon_raw, width=width, height=height, label="container polygon")
            bbox = _bbox(bbox_raw or _polygon_bbox(polygon), width=width, height=height, label="container bbox")
            candidates.append({
                "evidence_id": str(raw.get("evidence_id") or f"container_{index}"),
                "source": str(raw.get("source") or "verified_visual_container"),
                "bbox": bbox, "polygon": polygon,
                "confidence": round(max(0.0, min(1.0, float(raw.get("confidence", 0.0)))), 6),
            })
        else:
            polygon = _polygon(raw, width=width, height=height, label="container polygon")
            candidates.append({"evidence_id": f"container_{index}", "source": "balloon_inner_polygon", "bbox": _polygon_bbox(polygon), "polygon": polygon, "confidence": 1.0})
    return tuple(candidates)


def _partition_container_from_foreign_components(
    graph: OwnerGraph,
    *,
    owner_component_ids: set[str],
    source_bbox: BBox,
    container_bbox: BBox,
) -> tuple[BBox, tuple[str, ...], bool]:
    """Clip a verified container at foreign boundaries outside source ink.

    Detector components include padding around their observed text.  A foreign
    component may touch that padding without competing for any source glyph;
    only overlap with the selected source-replacement bbox is ambiguous.
    """

    suppressed = {
        disposition.component_id
        for disposition in graph.component_dispositions
        if disposition.decision == "suppress"
        and disposition.reason == "redundant_container_without_ocr_evidence"
    }
    observed = {
        component_id
        for observation in graph.observations
        if observation.polygons_page
        for component_id in observation.component_ids
    }
    preserved = {
        disposition.component_id
        for disposition in graph.component_dispositions
        if disposition.decision == "preserve"
    }
    foreign = [
        component
        for component in sorted(graph.components, key=lambda item: item.component_id)
        if component.component_id not in owner_component_ids
        and component.component_id not in suppressed
        and component.component_id in observed
        and not (
            component.bbox_page[2] <= container_bbox[0]
            or component.bbox_page[0] >= container_bbox[2]
            or component.bbox_page[3] <= container_bbox[1]
            or component.bbox_page[1] >= container_bbox[3]
        )
    ]
    left, top, right, bottom = container_bbox
    sx1, sy1, sx2, sy2 = source_bbox
    exclusions: list[str] = []
    conflict = False
    for component in foreign:
        foreign_bbox = component.bbox_page
        if component.component_id in preserved:
            accepted_observation_boxes = tuple(
                observation.text_pixel_bbox_page or observation.bbox_page
                for observation in graph.observations
                if component.component_id in observation.component_ids
                and observation.rejection_reason is None
                and observation.polygons_page
            )
            if accepted_observation_boxes:
                foreign_bbox = _consensus_bbox(accepted_observation_boxes)
        fx1, fy1, fx2, fy2 = foreign_bbox
        horizontal_overlap = min(fx2, sx2) > max(fx1, sx1)
        vertical_overlap = min(fy2, sy2) > max(fy1, sy1)
        if horizontal_overlap and vertical_overlap:
            conflict = True
            exclusions.append(f"foreign_component:{component.component_id}")
            continue
        clipped = False
        if horizontal_overlap and fy2 <= sy1:
            top = max(top, fy2)
            clipped = True
        elif horizontal_overlap and fy1 >= sy2:
            bottom = min(bottom, fy1)
            clipped = True
        elif vertical_overlap and fx2 <= sx1:
            left = max(left, fx2)
            clipped = True
        elif vertical_overlap and fx1 >= sx2:
            right = min(right, fx1)
            clipped = True
        if clipped:
            exclusions.append(f"foreign_component:{component.component_id}")
    result = (left, top, right, bottom)
    if not _contains(result, source_bbox):
        conflict = True
    return result, tuple(sorted(set(exclusions))), conflict


def _maximal_protected_safe_bbox(
    protected_art_mask: np.ndarray,
    *,
    source_bbox: BBox,
    container_bbox: BBox,
) -> BBox | None:
    """Grow a source-backed safe rectangle without crossing protected pixels."""

    protected = np.asarray(protected_art_mask) > 0
    left, top, right, bottom = source_bbox
    cx1, cy1, cx2, cy2 = container_bbox
    if np.any(protected[top:bottom, left:right]):
        # Cleanup remains forbidden on protected art. Typography is a separate
        # overlay and may reuse only the authenticated prior text footprint.
        return source_bbox
    while True:
        candidates: list[tuple[int, int, BBox]] = []
        if left > cx1 and not np.any(protected[top:bottom, left - 1 : left]):
            candidate = (left - 1, top, right, bottom)
            candidates.append(((right - left + 1) * (bottom - top), 0, candidate))
        if right < cx2 and not np.any(protected[top:bottom, right : right + 1]):
            candidate = (left, top, right + 1, bottom)
            candidates.append(((right - left + 1) * (bottom - top), 1, candidate))
        if top > cy1 and not np.any(protected[top - 1 : top, left:right]):
            candidate = (left, top - 1, right, bottom)
            candidates.append(((right - left) * (bottom - top + 1), 2, candidate))
        if bottom < cy2 and not np.any(protected[bottom : bottom + 1, left:right]):
            candidate = (left, top, right, bottom + 1)
            candidates.append(((right - left) * (bottom - top + 1), 3, candidate))
        if not candidates:
            break
        _area, _order, (left, top, right, bottom) = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
    return (left, top, right, bottom)


def build_owner_render_geometry(
    graph: OwnerGraph,
    owner_id: str,
    *,
    page_width: int,
    page_height: int,
    container_evidence: Any = None,
    protected_art_mask: np.ndarray | None = None,
    protected_art_mask_sha256: str | None = None,
) -> OwnerRenderGeometry:
    """Build the sole geometry authority used after owner reconciliation."""

    width, height = int(page_width), int(page_height)
    if width <= 0 or height <= 0:
        raise ValueError("owner render geometry requires positive page dimensions")
    matching = [owner for owner in graph.owners if owner.owner_id == owner_id]
    if len(matching) != 1:
        raise ValueError(f"owner render geometry requires exactly one owner {owner_id!r}")
    owner = matching[0]
    component_set = set(owner.component_ids)
    components = tuple(
        _hashed_component(component, width=width, height=height)
        for component in sorted(graph.components, key=lambda item: item.component_id)
        if component.component_id in component_set
    )
    if tuple(item.component_id for item in components) != tuple(sorted(component_set)):
        raise ValueError("owner render geometry is missing component geometry")
    semantic_bbox = _union_bbox(tuple(item.bbox_page for item in components))

    selected_ids = set(owner.selected_observation_ids)
    observations = tuple(
        _hashed_observation(observation, width=width, height=height)
        for observation in sorted(graph.observations, key=lambda item: item.observation_id)
        if observation.observation_id in selected_ids
    )
    if tuple(item.observation_id for item in observations) != tuple(sorted(selected_ids)):
        raise ValueError("owner render geometry is missing selected observation geometry")
    source_bbox = _union_bbox(tuple(item.bbox_page for item in observations))
    source_area = max(1, (source_bbox[2] - source_bbox[0]) * (source_bbox[3] - source_bbox[1]))
    # A discovery component can enclose a whole balloon (or a nearby owner)
    # while its OCR lines occupy only a small, supported portion. Retain that
    # component in provenance, but do not make its envelope the text body or a
    # paintable connected subregion. The selected source glyphs stay included.
    semantic_components = tuple(
        item for item in components
        if (item.bbox_page[2] - item.bbox_page[0]) * (item.bbox_page[3] - item.bbox_page[1])
        <= 3 * source_area
    )
    if semantic_components and len(semantic_components) < len(components):
        semantic_bbox = _union_bbox((source_bbox, *(item.bbox_page for item in semantic_components)))
    else:
        semantic_components = components
    protected_hash = (
        _check_sha256(protected_art_mask_sha256, "protected art mask hash")
        if protected_art_mask_sha256
        else _json_sha256(
            {
                "kind": "unbound_protected_art_mask",
                "owner_id": owner_id,
                "page_id": graph.page_id,
            }
        )
    )
    if protected_art_mask is not None:
        protected = np.asarray(protected_art_mask)
        if (
            protected.shape != (height, width)
            or protected.dtype != np.uint8
            or protected.ndim != 2
            or not np.all((protected == 0) | (protected == 255))
        ):
            raise ValueError("protected art mask must be canonical logical-page uint8")
        if _ndarray_sha256(protected) != protected_hash:
            raise ValueError("protected art mask content hash mismatch")

    projections = []
    for projection in sorted(graph.projections, key=lambda item: (item.tile_id, item.role, item.offset_xy)):
        if projection.owner_id != owner_id:
            continue
        tile_bbox = _bbox(projection.bbox_page, width=width, height=height, label="projection tile bbox")
        item = OwnerProjectionGeometry(
            projection_id=f"{owner_id}:{projection.tile_id}:{projection.role}",
            tile_id=str(projection.tile_id), role=str(projection.role), tile_bbox_page=tile_bbox,
            component_bbox_page=_intersection(semantic_bbox, tile_bbox),
            tile_to_page_offset_xy=(int(projection.offset_xy[0]), int(projection.offset_xy[1])), projection_sha256="",
        )
        projections.append(replace(item, projection_sha256=_json_sha256(item.payload())))

    candidates = tuple(candidate for candidate in _container_candidates(container_evidence, width=width, height=height) if _contains(candidate["bbox"], semantic_bbox))
    chosen = min(candidates, key=lambda item: ((item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]), -item["confidence"], item["evidence_id"])) if candidates else None
    role = str(owner.semantic_role or "").strip().casefold()
    freeform = any(token in role for token in _FREEFORM_ROLE_TOKENS)
    dialogue = any(token in role for token in _DIALOGUE_ROLE_TOKENS)
    card = any(token in role for token in _CARD_ROLE_TOKENS)
    selected_providers = {
        str(observation.provider or "").strip().casefold()
        for observation in graph.observations
        if observation.observation_id in selected_ids
    }
    visual_card_slot = any(
        provider.startswith("visual_card") for provider in selected_providers
    ) and any(
        provider.startswith("candidate_crop_direct_paddle")
        for provider in selected_providers
    )
    if chosen is not None:
        layout_bbox, layout_polygon, layout_source = chosen["bbox"], chosen["polygon"], chosen["source"]
        status, reason = "ready", "independent_container_verified"
        evidence_ids = tuple(sorted(candidate["evidence_id"] for candidate in candidates))
        evidence_confidence = max(candidate["confidence"] for candidate in candidates)
        exclusive_bbox, exclusion_ids, exclusion_conflict = (
            _partition_container_from_foreign_components(
                graph,
                owner_component_ids=component_set,
                source_bbox=source_bbox,
                container_bbox=layout_bbox,
            )
        )
        if exclusion_conflict:
            layout_bbox = layout_polygon = None
            layout_source = "none"
            status = "review_required"
            reason = "foreign_owner_geometry_overlaps_semantic_body"
        elif exclusive_bbox != layout_bbox:
            layout_bbox = exclusive_bbox
            layout_polygon = _rect_polygon(layout_bbox)
            layout_source = f"{layout_source}:owner_exclusive"
            evidence_ids = tuple(sorted({*evidence_ids, *exclusion_ids}))
        if status == "ready" and protected_art_mask is not None:
            safe_bbox = _maximal_protected_safe_bbox(
                protected_art_mask,
                source_bbox=source_bbox,
                container_bbox=layout_bbox,
            )
            if safe_bbox is None:
                layout_bbox = layout_polygon = None
                layout_source = "none"
                status = "review_required"
                reason = "source_replacement_overlaps_protected_art"
            elif safe_bbox != layout_bbox:
                layout_bbox = safe_bbox
                layout_polygon = _rect_polygon(layout_bbox)
                layout_source = f"{layout_source}:protected_mask_safe"
                evidence_ids = tuple(
                    sorted({*evidence_ids, f"protected_art_mask:{protected_hash}"})
                )
        # The full-page visual recovery is a weak envelope, not proof that
        # every pixel inside its rectangular bbox is a paintable balloon.
        # Keep the source untouched when it bridges separate text supports or
        # expands a small source slot across an implausibly large page area.
        if status == "ready" and str(layout_source).startswith("full_page_visual_container"):
            ordered = sorted((item.bbox_page for item in components), key=lambda box: (box[1], box[3]))
            vertical_clusters: list[tuple[int, int]] = []
            for box in ordered:
                if vertical_clusters and box[1] <= vertical_clusters[-1][1]:
                    vertical_clusters[-1] = (vertical_clusters[-1][0], max(vertical_clusters[-1][1], box[3]))
                else:
                    vertical_clusters.append((box[1], box[3]))
            bridges_gap = any(
                bottom[0] - top[1] >= max(20, .7 * max(top[1] - top[0], bottom[1] - bottom[0]))
                for top, bottom in zip(vertical_clusters, vertical_clusters[1:])
            )
            source_area = max(1, (source_bbox[2] - source_bbox[0]) * (source_bbox[3] - source_bbox[1]))
            container_area = (layout_bbox[2] - layout_bbox[0]) * (layout_bbox[3] - layout_bbox[1])
            unsafe_reason = (
                "recovered_container_bridges_disjoint_components" if bridges_gap
                else "recovered_container_exceeds_source_slot" if container_area > 6 * source_area
                else None
            )
            if unsafe_reason:
                layout_bbox = layout_polygon = None
                layout_source = "none"
                status, reason = "review_required", unsafe_reason
            elif any(str(item).startswith("full_page_visual_container:") for item in evidence_ids):
                # The protected-art mask only excludes known foreign ink.
                # It cannot certify the balloon interior or its curved edge.
                # A full-page recovery remains an envelope even when its
                # rectangular dimensions pass the source-slot heuristic.
                layout_bbox = layout_polygon = None
                layout_source = "none"
                status, reason = "review_required", "unverified_full_page_balloon_interior"
    elif visual_card_slot:
        layout_bbox = semantic_bbox
        layout_polygon = _rect_polygon(semantic_bbox)
        layout_source = "visual_card_text_slot"
        status, reason = "ready", "typed_visual_card_text_slot"
        evidence_ids = tuple(
            sorted(observation.observation_id for observation in observations)
        )
        evidence_confidence = max(
            (observation.confidence for observation in observations),
            default=0.0,
        )
    elif freeform:
        layout_bbox, layout_polygon, layout_source = semantic_bbox, _rect_polygon(semantic_bbox), "freeform_component_union"
        status, reason, evidence_ids, evidence_confidence = "ready", "typed_freeform_component_union", (), 1.0
    else:
        layout_bbox = layout_polygon = None
        layout_source = "none"
        status = "review_required"
        reason = "missing_independent_dialogue_container" if dialogue else "missing_verified_card_container" if card else "missing_independent_layout_container"
        evidence_ids, evidence_confidence = (), 0.0
        if dialogue:
            from . import experimental_ocr_region
            if experimental_ocr_region.enabled():
                foreign_boxes = tuple(
                    component.bbox_page for component in graph.components
                    if component.component_id not in component_set
                )
                proposal, refusal = experimental_ocr_region.propose(
                    source_bbox, width=width, height=height,
                    protected_art_mask=protected_art_mask,
                    foreign_boxes=foreign_boxes,
                )
                if proposal is not None:
                    layout_bbox, layout_polygon = proposal
                    layout_polygon = _polygon(
                        layout_polygon, width=width, height=height,
                        label="experimental virtual OCR layout polygon",
                    )
                    layout_source = "experimental_virtual_ocr_region"
                    status, reason = "ready", "experimental_ocr_layout_requires_review"
                    evidence_ids = tuple(sorted(selected_ids))
                    evidence_confidence = min(
                        (observation.confidence for observation in observations),
                        default=0.0,
                    )
                else:
                    reason = f"experimental_ocr_region_refused:{refusal}"

    subregions = []
    for order, component in enumerate(semantic_components):
        evidence_ids_for_component = tuple(sorted(
            observation.observation_id
            for observation in graph.observations
            if observation.observation_id in selected_ids and component.component_id in observation.component_ids
        ))
        item = ConnectedLayoutSubregion(
            subregion_id=f"{owner_id}:{component.component_id}", order=order,
            component_ids=(component.component_id,), bbox_page=component.bbox_page,
            polygon_page=component.polygon_page,
            capacity_px2=(component.bbox_page[2] - component.bbox_page[0]) * (component.bbox_page[3] - component.bbox_page[1]),
            evidence_source="selected_observation_component_geometry",
            evidence_ids=evidence_ids_for_component, subregion_sha256="",
        )
        subregions.append(replace(item, subregion_sha256=_json_sha256(item.payload())))

    component_hash = _json_sha256([item.to_dict() for item in components])
    draft = OwnerRenderGeometry(
        schema_version=1, owner_id=str(owner_id), page_id=str(graph.page_id), logical_space="logical_page",
        page_width=width, page_height=height, component_ids=tuple(item.component_id for item in components),
        components=components, component_geometry_sha256=component_hash,
        selected_observations=observations, projections=tuple(projections), semantic_body_bbox_page=semantic_bbox,
        source_replacement_bbox_page=source_bbox, layout_container_bbox_page=layout_bbox,
        layout_container_polygon_page=layout_polygon, layout_container_source=layout_source,
        safe_polygons_page=(layout_polygon,) if layout_polygon else (), connected_subregions=tuple(subregions),
        container_evidence_ids=evidence_ids, container_evidence_confidence=evidence_confidence,
        protected_art_mask_sha256=protected_hash, status=status, reason=reason, geometry_sha256="",
    )
    result = replace(draft, geometry_sha256=_json_sha256(draft.payload()))
    result._validate()
    return result


def owner_source_replacement_bbox(graph: OwnerGraph, owner_id: str) -> BBox:
    """Return the selected source-text footprint without cleanup dilation."""

    owners = [owner for owner in graph.owners if owner.owner_id == owner_id]
    if len(owners) != 1:
        raise ValueError(f"source replacement geometry requires one owner {owner_id!r}")
    selected_ids = set(owners[0].selected_observation_ids)
    boxes = tuple(
        tuple(
            int(value)
            for value in (
                observation.text_pixel_bbox_page or observation.bbox_page
            )
        )
        for observation in graph.observations
        if observation.observation_id in selected_ids
    )
    if len(boxes) != len(selected_ids) or not boxes:
        raise ValueError("source replacement geometry is missing selected observations")
    return _union_bbox(boxes)


def release_source_replacement_from_protection(
    protected_mask: np.ndarray,
    source_replacement_bbox: BBox,
    *,
    source_replacement_mask: np.ndarray | None = None,
    foreign_component_masks: Sequence[tuple[str, np.ndarray]] = (),
) -> np.ndarray:
    """Release authenticated source ink and then restore foreign-owner pixels."""

    protected = np.asarray(protected_mask)
    if protected.dtype != np.uint8 or protected.ndim != 2:
        raise ValueError("protected art mask must be a canonical uint8 page mask")
    height, width = protected.shape
    x1, y1, x2, y2 = _bbox(
        source_replacement_bbox,
        width=width,
        height=height,
        label="source replacement bbox",
    )
    result = np.where(protected > 0, 255, 0).astype(np.uint8)
    if source_replacement_mask is None:
        authorized = np.zeros_like(result)
        authorized[y1:y2, x1:x2] = 255
    else:
        replacement = np.asarray(source_replacement_mask)
        if (
            replacement.shape != result.shape
            or replacement.dtype != np.uint8
            or replacement.ndim != 2
        ):
            raise ValueError("source replacement mask does not match logical page")
        authorized = np.where(replacement > 0, 255, 0).astype(np.uint8)
        clipped = np.zeros_like(authorized)
        clipped[y1:y2, x1:x2] = authorized[y1:y2, x1:x2]
        authorized = clipped
        if not np.any(authorized):
            raise ValueError("source replacement mask is empty")
    result[authorized > 0] = 0
    for evidence_id, raw_mask in foreign_component_masks:
        if not str(evidence_id or "").strip():
            raise ValueError("foreign component evidence id is empty")
        mask = np.asarray(raw_mask)
        if mask.shape != result.shape or mask.dtype != np.uint8 or mask.ndim != 2:
            raise ValueError("foreign component mask does not match logical page")
        foreign = np.where(mask > 0, 255, 0).astype(np.uint8)
        # A coarse detector/legacy union may geometrically overlap the source
        # slot even when its accepted OCR ink does not.  The caller verifies
        # true foreign semantic overlap through OwnerRenderGeometry before any
        # mutation; do not re-protect the very source pixels being replaced.
        foreign[y1:y2, x1:x2] = 0
        result[foreign > 0] = 255
    return np.ascontiguousarray(result, dtype=np.uint8)


def positive_evidence_excluding_protected(
    positive_mask: np.ndarray,
    protected_mask: np.ndarray,
) -> np.ndarray:
    """Remove protected pixels from positive text evidence without dilation."""

    positive = np.asarray(positive_mask)
    protected = np.asarray(protected_mask)
    if (
        positive.dtype != np.uint8
        or protected.dtype != np.uint8
        or positive.ndim != 2
        or protected.ndim != 2
        or positive.shape != protected.shape
    ):
        raise ValueError("positive/protected evidence must share a uint8 logical page")
    result = np.where(positive > 0, 255, 0).astype(np.uint8)
    result[protected > 0] = 0
    return np.ascontiguousarray(result, dtype=np.uint8)


__all__ = [
    "ConnectedLayoutSubregion", "OwnerComponentGeometry", "OwnerObservationGeometry",
    "OwnerProjectionGeometry", "OwnerRenderGeometry", "build_owner_render_geometry",
    "owner_source_replacement_bbox",
    "positive_evidence_excluding_protected",
    "release_source_replacement_from_protection",
]
