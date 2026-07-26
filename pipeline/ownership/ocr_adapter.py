"""Lossless OCR-record capture in immutable page coordinates.

Legacy OCR code may still select, merge, or suppress records while the owner
control plane is introduced.  This adapter captures every provider output
before those mutations and never infers a semantic owner from a band.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from typing import Literal, Mapping, Sequence

from .coordinates import bbox_tile_to_page, polygon_tile_to_page, stable_observation_id
from .model import BBox, Point, TextObservation


Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class TileProjection:
    """Identity and page-space offset of one OCR tile."""

    page_id: str
    tile_id: str
    offset_xy: Point = (0, 0)
    provider_variant: str = ""
    attempt_id: str = "primary"
    coordinate_space: Literal["tile", "page"] = "tile"
    projection_id: str = ""
    page_size: tuple[int, int] | None = None
    tile_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.coordinate_space not in {"tile", "page"}:
            raise ValueError("coordinate_space must be 'tile' or 'page'")
        try:
            valid_offset = len(self.offset_xy) == 2
            if valid_offset:
                tuple(int(value) for value in self.offset_xy)
        except (TypeError, ValueError, OverflowError):
            valid_offset = False
        if not valid_offset:
            raise ValueError("offset_xy must contain exactly two numeric coordinates")
        for field_name in ("page_size", "tile_size"):
            value = getattr(self, field_name)
            if value is None:
                continue
            try:
                valid = len(value) == 2 and all(int(item) > 0 for item in value)
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(f"{field_name} must contain positive width and height")

    @property
    def resolved_projection_id(self) -> str:
        if self.projection_id:
            return str(self.projection_id)
        offset_x, offset_y = (int(value) for value in self.offset_xy)
        return f"{self.tile_id}:{self.coordinate_space}:{offset_x},{offset_y}"


def _bbox(value: object) -> BBox:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return (0, 0, 0, 0)
    try:
        x1, y1, x2, y2 = (int(round(float(item))) for item in value[:4])
    except (TypeError, ValueError, OverflowError):
        return (0, 0, 0, 0)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _bbox_has_reversed_order(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in value[:4])
    except (TypeError, ValueError, OverflowError):
        return False
    return x2 < x1 or y2 < y1


def _bbox_exceeds_size(bbox: BBox, size: tuple[int, int]) -> bool:
    width, height = (int(value) for value in size)
    x1, y1, x2, y2 = bbox
    return x1 < 0 or y1 < 0 or x2 > width or y2 > height


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _polygon(value: object) -> Polygon:
    if not isinstance(value, (list, tuple)):
        return ()
    points: list[Point] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return ()
        try:
            points.append((int(round(float(point[0]))), int(round(float(point[1])))))
        except (TypeError, ValueError, OverflowError):
            return ()
    return tuple(points) if len(points) >= 3 else ()


def _record_polygons(record: dict) -> tuple[tuple[Polygon, ...], bool]:
    page_values = record.get("polygons_page")
    if isinstance(page_values, (list, tuple)):
        return tuple(polygon for item in page_values if (polygon := _polygon(item))), True

    values = record.get("line_polygons") or record.get("polygons")
    if values is None:
        single = record.get("line_polygon") or record.get("polygon")
        values = [single] if single else []
    return tuple(polygon for item in values if (polygon := _polygon(item))), False


def _first_present(record: dict, keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in record and record.get(key) is not None:
            return record.get(key)
    return None


def _bbox_to_page(
    value: object,
    projection: TileProjection,
    *,
    already_page_space: bool = False,
) -> BBox | None:
    if value is None:
        return None
    parsed = _bbox(value)
    if already_page_space or projection.coordinate_space == "page":
        return parsed
    return bbox_tile_to_page(parsed, projection.offset_xy)


def _rejection_reason(record: dict) -> str | None:
    explicit = record.get("rejection_reason")
    if explicit:
        return str(explicit)
    rejected = bool(record.get("rejected")) or record.get("accepted") is False
    route_action = str(record.get("route_action") or "").strip().lower()
    if route_action in {"reject", "rejected", "suppress", "suppressed"}:
        rejected = True
    if rejected:
        return str(record.get("route_reason") or "legacy_rejected_candidate")
    return None


def observation_to_dict(observation: TextObservation) -> dict:
    """Serialize one observation without introducing mutable owner fields."""

    return {
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
        "legacy_selected": bool(observation.legacy_selected),
        "provider_variant": observation.provider_variant,
        "attempt_id": observation.attempt_id,
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
        "line_texts": list(observation.line_texts),
        "rotation_deg": observation.rotation_deg,
        "rotation_source": observation.rotation_source,
    }


def record_to_observation(record: dict, projection: TileProjection) -> TextObservation:
    """Convert one provider record without dropping empty/rejected candidates."""

    provider = str(
        record.get("provider")
        or record.get("ocr_provider")
        or record.get("detector")
        or "unknown_ocr"
    )
    bbox_value = _first_present(record, ("bbox", "source_bbox", "text_pixel_bbox"))
    bbox_input: BBox | None = None
    if record.get("bbox_page") is not None:
        bbox_page = _bbox(record.get("bbox_page"))
    else:
        bbox_input = _bbox(bbox_value or (0, 0, 0, 0))
        bbox_page = (
            bbox_input
            if projection.coordinate_space == "page"
            else bbox_tile_to_page(bbox_input, projection.offset_xy)
        )

    polygons, polygons_are_page_space = _record_polygons(record)
    has_polygon_input = any(
        bool(record.get(key))
        for key in (
            "polygons_page",
            "line_polygons",
            "polygons",
            "line_polygon",
            "polygon",
        )
    )
    polygons_page = (
        polygons
        if polygons_are_page_space
        else (
            polygons
            if projection.coordinate_space == "page"
            else tuple(polygon_tile_to_page(polygon, projection.offset_xy) for polygon in polygons)
        )
    )
    component_ids = tuple(
        sorted({str(value) for value in record.get("component_ids") or () if value})
    )
    occurrence_index = max(0, int(record.get("_observation_occurrence_index") or 0))
    provider_variant = str(
        record.get("provider_variant")
        or record.get("variant")
        or record.get("ocr_variant")
        or record.get("attempt_kind")
        or projection.provider_variant
        or ""
    )
    attempt_id = str(record.get("attempt_id") or projection.attempt_id or "primary")
    provider_record_id_value = record.get("provider_record_id") or record.get("record_id")
    provider_record_id = (
        str(provider_record_id_value) if provider_record_id_value not in (None, "") else None
    )
    identity_provider = json.dumps(
        [provider, provider_variant, attempt_id, provider_record_id or ""],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    text = str(
        _first_present(record, ("text", "raw_ocr", "original")) or ""
    )
    raw_text_value = _first_present(record, ("raw_text", "raw_ocr", "original", "text"))
    source_bbox_value = _first_present(record, ("source_bbox_page", "source_bbox"))
    source_bbox_page = _bbox_to_page(
        source_bbox_value,
        projection,
        already_page_space=record.get("source_bbox_page") is not None,
    )
    text_pixel_bbox_value = _first_present(
        record,
        ("text_pixel_bbox_page", "text_pixel_bbox"),
    )
    text_pixel_bbox_page = _bbox_to_page(
        text_pixel_bbox_value,
        projection,
        already_page_space=record.get("text_pixel_bbox_page") is not None,
    )
    confidence_value = _first_present(record, ("confidence", "confidence_raw", "score"))
    confidence = _optional_float(confidence_value)
    rotation_value = _first_present(record, ("rotation_deg", "rotation"))
    rotation_deg = _optional_float(rotation_value)
    rejection_reason = _rejection_reason(record)
    if rejection_reason is None and record.get("bbox_page") is None and bbox_value is None:
        rejection_reason = "missing_bbox"
    raw_bbox_value = record.get("bbox_page") if record.get("bbox_page") is not None else bbox_value
    if rejection_reason is None and _bbox_has_reversed_order(raw_bbox_value):
        rejection_reason = "invalid_bbox_order"
    if rejection_reason is None and (
        bbox_page[2] <= bbox_page[0] or bbox_page[3] <= bbox_page[1]
    ):
        rejection_reason = "invalid_bbox"
    if (
        rejection_reason is None
        and source_bbox_value is not None
        and source_bbox_page is not None
        and (
            source_bbox_page[2] <= source_bbox_page[0]
            or source_bbox_page[3] <= source_bbox_page[1]
        )
    ):
        rejection_reason = "invalid_source_bbox"
    if (
        rejection_reason is None
        and text_pixel_bbox_value is not None
        and text_pixel_bbox_page is not None
        and (
            text_pixel_bbox_page[2] <= text_pixel_bbox_page[0]
            or text_pixel_bbox_page[3] <= text_pixel_bbox_page[1]
        )
    ):
        rejection_reason = "invalid_text_pixel_bbox"
    if (
        rejection_reason is None
        and projection.coordinate_space == "tile"
        and projection.tile_size is not None
        and bbox_input is not None
        and _bbox_exceeds_size(bbox_input, projection.tile_size)
    ):
        rejection_reason = "bbox_outside_tile"
    if (
        rejection_reason is None
        and projection.page_size is not None
        and _bbox_exceeds_size(bbox_page, projection.page_size)
    ):
        rejection_reason = "bbox_outside_page"
    if rejection_reason is None and has_polygon_input and not polygons_page:
        rejection_reason = "invalid_polygons"
    if rejection_reason is None and confidence_value is not None and confidence is None:
        rejection_reason = "invalid_confidence"
    if rejection_reason is None and not text.strip():
        rejection_reason = "empty_text"
    tile_provenance = tuple(
        dict.fromkeys(
            [
                *(str(value) for value in record.get("tile_provenance") or () if value),
                *([projection.tile_id] if projection.tile_id else []),
            ]
        )
    )
    return TextObservation(
        observation_id=stable_observation_id(
            projection.page_id,
            identity_provider,
            bbox_page,
            component_ids,
            0 if provider_record_id is not None else occurrence_index,
        ),
        page_id=projection.page_id,
        component_ids=component_ids,
        text=text,
        confidence=confidence or 0.0,
        provider=provider,
        bbox_page=bbox_page,
        polygons_page=polygons_page,
        tile_provenance=tile_provenance,
        coverage_score=_optional_float(record.get("coverage_score")),
        language_score=_optional_float(record.get("language_score")),
        rejection_reason=rejection_reason,
        legacy_selected=bool(record.get("legacy_selected", False)),
        provider_variant=provider_variant,
        attempt_id=attempt_id,
        provider_record_id=provider_record_id,
        projection_ids=(projection.resolved_projection_id,),
        raw_text=str(raw_text_value) if raw_text_value is not None else None,
        source_bbox_page=source_bbox_page,
        text_pixel_bbox_page=text_pixel_bbox_page,
        line_texts=tuple(str(value) for value in record.get("line_texts") or ()),
        rotation_deg=rotation_deg,
        rotation_source=(
            str(record.get("rotation_source"))
            if record.get("rotation_source") is not None
            else None
        ),
    )


def collect_page_observations(
    records_by_provider: Mapping[str, Sequence[dict]],
    projection: TileProjection,
) -> list[TextObservation]:
    """Capture every record from every provider, preserving provider order."""

    observations: list[TextObservation] = []
    for provider, records in records_by_provider.items():
        for occurrence_index, source_record in enumerate(records):
            record = dict(source_record)
            record["provider"] = str(record.get("provider") or provider)
            record["provider_variant"] = str(
                record.get("provider_variant")
                or record.get("variant")
                or record.get("ocr_variant")
                or record.get("attempt_kind")
                or provider
            )
            record["_observation_occurrence_index"] = occurrence_index
            observations.append(record_to_observation(record, projection))
    return observations


def attach_observation_manifest(
    page: dict,
    observations: Sequence[TextObservation],
) -> dict:
    """Return a shadow-mode page with an append-only observation manifest."""

    result = dict(page)
    merged: list[dict] = []
    index_by_id: dict[str, int] = {}
    items: list[object] = [
        *list(page.get("owner_observations") or ()),
        *list(observations),
    ]
    for item in items:
        if isinstance(item, TextObservation):
            payload = observation_to_dict(item)
        elif isinstance(item, Mapping):
            payload = copy.deepcopy(dict(item))
        else:
            raise TypeError("owner_observations entries must be mappings or TextObservation values")
        observation_id = str(payload.get("observation_id") or "")
        if not observation_id or observation_id not in index_by_id:
            if observation_id:
                index_by_id[observation_id] = len(merged)
            merged.append(payload)
            continue
        position = index_by_id[observation_id]
        merged[position] = _union_manifest_observation(merged[position], payload)
    result["owner_observations"] = merged
    return result


def _union_manifest_observation(existing: dict, incoming: dict) -> dict:
    for field in (
        "page_id",
        "provider",
        "provider_variant",
        "attempt_id",
        "provider_record_id",
        "bbox_page",
    ):
        existing_value = existing.get(field)
        incoming_value = incoming.get(field)
        if existing_value in (None, "", [], ()) or incoming_value in (None, "", [], ()):
            continue
        if existing_value != incoming_value:
            observation_id = str(existing.get("observation_id") or incoming.get("observation_id") or "")
            raise ValueError(
                f"observation_id collision for {observation_id!r}: conflicting {field}"
            )

    merged = copy.deepcopy(existing)
    for field in ("component_ids", "polygons_page", "tile_provenance", "projection_ids"):
        values: list = []
        seen: set[str] = set()
        for value in [*list(existing.get(field) or ()), *list(incoming.get(field) or ())]:
            token = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
            if token in seen:
                continue
            seen.add(token)
            values.append(copy.deepcopy(value))
        merged[field] = values

    for field in (
        "page_id",
        "text",
        "raw_text",
        "confidence",
        "provider",
        "provider_variant",
        "attempt_id",
        "provider_record_id",
        "bbox_page",
        "source_bbox_page",
        "text_pixel_bbox_page",
        "coverage_score",
        "language_score",
        "rejection_reason",
        "line_texts",
        "rotation_deg",
        "rotation_source",
    ):
        if merged.get(field) in (None, "", [], ()) and incoming.get(field) not in (
            None,
            "",
            [],
            (),
        ):
            merged[field] = copy.deepcopy(incoming.get(field))
    merged["legacy_selected"] = bool(existing.get("legacy_selected")) or bool(
        incoming.get("legacy_selected")
    )
    return merged
