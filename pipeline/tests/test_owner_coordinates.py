"""Contract tests for stable page-space owner geometry."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.coordinates import (
    ComponentSeed,
    assign_component_ids,
    bbox_page_to_tile,
    bbox_tile_to_page,
    polygon_page_to_tile,
    polygon_tile_to_page,
    stable_observation_id,
)


def test_component_ids_are_stable_under_detector_order() -> None:
    seeds = [
        ComponentSeed(bbox_page=(90, 140, 210, 190), detector_source="primary"),
        ComponentSeed(bbox_page=(20, 30, 160, 80), detector_source="recall"),
    ]

    forward = dict(zip((seed.bbox_page for seed in seeds), assign_component_ids("page_001", seeds)))
    reversed_seeds = list(reversed(seeds))
    backward = dict(
        zip(
            (seed.bbox_page for seed in reversed_seeds),
            assign_component_ids("page_001", reversed_seeds),
        )
    )

    assert forward == backward
    assert all(component_id.startswith("region_p001_") for component_id in forward.values())


def test_observation_identity_does_not_depend_on_mutable_text() -> None:
    signature = inspect.signature(stable_observation_id)
    assert "text" not in signature.parameters

    kwargs = {
        "page_id": "page_001",
        "provider": "paddle",
        "bbox_page": (20, 30, 160, 80),
        "component_ids": ("region_p001_001_abcd1234",),
        "occurrence_index": 0,
    }
    assert stable_observation_id(**kwargs) == stable_observation_id(**kwargs)


def test_bbox_projection_round_trips_between_page_and_tile() -> None:
    bbox_page = (104, 208, 266, 292)
    tile_offset = (40, 180)

    bbox_tile = bbox_page_to_tile(bbox_page, tile_offset)

    assert bbox_tile == (64, 28, 226, 112)
    assert bbox_tile_to_page(bbox_tile, tile_offset) == bbox_page


def test_polygon_projection_round_trips_between_page_and_tile() -> None:
    polygon_page = ((104, 208), (266, 208), (250, 292), (110, 288))
    tile_offset = (40, 180)

    polygon_tile = polygon_page_to_tile(polygon_page, tile_offset)

    assert polygon_tile == ((64, 28), (226, 28), (210, 112), (70, 108))
    assert polygon_tile_to_page(polygon_tile, tile_offset) == polygon_page
