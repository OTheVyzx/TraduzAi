"""Pure coordinate transforms and stable spatial identities.

Identifiers in this module deliberately exclude OCR text, translations, work
names, chapter names, and tile boundaries.  They describe visual evidence in
the immutable page coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable, Sequence

from .model import BBox, Point


Offset = tuple[int, int]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class ComponentSeed:
    """Minimal immutable evidence used to assign a source component ID."""

    bbox_page: BBox
    detector_source: str


def _normalise_bbox(bbox: Sequence[int]) -> BBox:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    x1, y1, x2, y2 = (int(value) for value in bbox)
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox coordinates must be ordered")
    return x1, y1, x2, y2


def _scope_token(scope_id: str) -> str:
    value = str(scope_id).strip().lower()
    match = re.fullmatch(r"page[_-]?(\d+)", value)
    if match:
        return f"p{int(match.group(1)):03d}"
    token = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return token or "unknown"


def _digest(payload: object) -> str:
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:10]


def stable_spatial_id(
    prefix: str,
    scope_id: str,
    bbox: Sequence[int],
    occurrence_index: int,
    *,
    evidence_source: str = "",
) -> str:
    """Return a deterministic ID for geometry inside one stable scope."""

    normalised_bbox = _normalise_bbox(bbox)
    ordinal = int(occurrence_index)
    if ordinal < 0:
        raise ValueError("occurrence_index must be non-negative")
    safe_prefix = re.sub(r"[^a-z0-9]+", "_", str(prefix).lower()).strip("_")
    if not safe_prefix:
        raise ValueError("prefix must contain at least one alphanumeric character")
    digest = _digest(
        {
            "bbox": normalised_bbox,
            "evidence_source": str(evidence_source),
            "occurrence_index": ordinal,
            "scope_id": str(scope_id),
        }
    )
    return f"{safe_prefix}_{_scope_token(scope_id)}_{ordinal + 1:03d}_{digest}"


def assign_component_ids(
    page_id: str,
    seeds: Iterable[ComponentSeed],
) -> tuple[str, ...]:
    """Assign component IDs in canonical spatial order, preserving input shape."""

    materialised = list(seeds)
    canonical = sorted(
        enumerate(materialised),
        key=lambda pair: (
            _normalise_bbox(pair[1].bbox_page)[1],
            _normalise_bbox(pair[1].bbox_page)[0],
            _normalise_bbox(pair[1].bbox_page)[3],
            _normalise_bbox(pair[1].bbox_page)[2],
            str(pair[1].detector_source),
        ),
    )
    assigned: list[str | None] = [None] * len(materialised)
    for ordinal, (original_index, seed) in enumerate(canonical):
        assigned[original_index] = stable_spatial_id(
            "region",
            page_id,
            seed.bbox_page,
            ordinal,
            evidence_source=seed.detector_source,
        )
    return tuple(value for value in assigned if value is not None)


def stable_observation_id(
    page_id: str,
    provider: str,
    bbox_page: Sequence[int],
    component_ids: Sequence[str],
    occurrence_index: int,
) -> str:
    """Identify an OCR observation without coupling identity to mutable text."""

    bbox = _normalise_bbox(bbox_page)
    ordinal = int(occurrence_index)
    if ordinal < 0:
        raise ValueError("occurrence_index must be non-negative")
    digest = _digest(
        {
            "bbox_page": bbox,
            "component_ids": sorted(str(value) for value in component_ids),
            "occurrence_index": ordinal,
            "page_id": str(page_id),
            "provider": str(provider),
        }
    )
    return f"observation_{_scope_token(page_id)}_{ordinal + 1:03d}_{digest}"


def bbox_page_to_tile(bbox_page: Sequence[int], tile_offset_xy: Offset) -> BBox:
    """Project a page-space bbox into a tile whose origin is page-space offset."""

    x1, y1, x2, y2 = _normalise_bbox(bbox_page)
    offset_x, offset_y = (int(value) for value in tile_offset_xy)
    return x1 - offset_x, y1 - offset_y, x2 - offset_x, y2 - offset_y


def bbox_tile_to_page(bbox_tile: Sequence[int], tile_offset_xy: Offset) -> BBox:
    """Project a tile-space bbox back into immutable page coordinates."""

    x1, y1, x2, y2 = _normalise_bbox(bbox_tile)
    offset_x, offset_y = (int(value) for value in tile_offset_xy)
    return x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y


def polygon_page_to_tile(polygon_page: Iterable[Sequence[int]], tile_offset_xy: Offset) -> Polygon:
    offset_x, offset_y = (int(value) for value in tile_offset_xy)
    return tuple((int(point[0]) - offset_x, int(point[1]) - offset_y) for point in polygon_page)


def polygon_tile_to_page(polygon_tile: Iterable[Sequence[int]], tile_offset_xy: Offset) -> Polygon:
    offset_x, offset_y = (int(value) for value in tile_offset_xy)
    return tuple((int(point[0]) + offset_x, int(point[1]) + offset_y) for point in polygon_tile)
