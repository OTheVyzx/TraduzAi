from __future__ import annotations

import numpy as np
import pytest

from strip.page_surface_geometry import PageSurfaceGeometry


def centered_geometry() -> PageSurfaceGeometry:
    return PageSurfaceGeometry.build(
        logical_width=690,
        logical_height=1600,
        frame_width=800,
        frame_height=1600,
        content_origin_xy=(55, 0),
    )


def test_logical_bbox_round_trips_through_centered_frame():
    geometry = centered_geometry()
    logical = (109, 985, 318, 1104)

    assert geometry.logical_bbox_to_frame(logical) == (164, 985, 373, 1104)
    assert geometry.frame_bbox_to_logical((164, 985, 373, 1104)) == logical


def test_polygon_and_mask_use_same_transform_without_resampling():
    geometry = centered_geometry()
    assert geometry.logical_polygon_to_frame(((0, 0), (690, 1600))) == (
        (55, 0),
        (745, 1600),
    )
    mask = np.zeros((1600, 690), dtype=np.uint8)
    mask[985, 109] = 255
    framed = geometry.logical_array_to_frame(mask, fill_value=0)
    assert framed[985, 164] == 255
    assert np.count_nonzero(framed) == 1


@pytest.mark.parametrize(
    "shape",
    [(1600, 690), (1600, 690, 1), (1600, 690, 3), (1600, 690, 4)],
)
def test_array_transform_preserves_trailing_dimensions_dtype_and_round_trip(shape):
    source = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    geometry = centered_geometry()

    framed = geometry.logical_array_to_frame(source, fill_value=0)

    assert framed.shape == (1600, 800, *shape[2:])
    assert framed.dtype == source.dtype
    assert np.array_equal(geometry.frame_array_to_logical(framed), source)


def test_string_owner_map_round_trips_without_resize():
    owner_map = np.full((1600, 690), "", dtype=object)
    owner_map[985, 109] = "owner_a"

    framed = centered_geometry().logical_array_to_frame(owner_map, fill_value="")

    assert framed[985, 164] == "owner_a"
    assert centered_geometry().frame_array_to_logical(framed)[985, 109] == "owner_a"


def test_identity_geometry_and_odd_offsets_are_exact():
    identity = PageSurfaceGeometry.build(
        logical_width=690,
        logical_height=1600,
        frame_width=690,
        frame_height=1600,
        content_origin_xy=(0, 0),
    )
    odd = PageSurfaceGeometry.build(
        logical_width=3,
        logical_height=2,
        frame_width=8,
        frame_height=5,
        content_origin_xy=(3, 1),
    )

    assert identity.logical_bbox_to_frame((0, 0, 690, 1600)) == (0, 0, 690, 1600)
    assert odd.content_bbox_frame == (3, 1, 6, 3)


def test_half_open_bbox_and_continuous_polygon_boundaries_are_distinct():
    geometry = centered_geometry()

    assert geometry.logical_bbox_to_frame((0, 0, 690, 1600)) == (55, 0, 745, 1600)
    assert geometry.logical_polygon_to_frame(((690, 1600),)) == ((745, 1600),)
    with pytest.raises(ValueError, match="bbox"):
        geometry.logical_bbox_to_frame((0, 0, 691, 1600))


def test_geometry_serialization_rejects_tampered_hash_or_derived_bbox():
    payload = centered_geometry().to_dict()
    assert PageSurfaceGeometry.from_dict(payload).to_dict() == payload

    tampered_hash = dict(payload, geometry_sha256="0" * 64)
    with pytest.raises(ValueError, match="hash"):
        PageSurfaceGeometry.from_dict(tampered_hash)
    tampered_bbox = dict(payload, content_bbox_frame=[0, 0, 690, 1600])
    with pytest.raises(ValueError, match="content_bbox_frame"):
        PageSurfaceGeometry.from_dict(tampered_bbox)


def test_page_surface_geometry_rejects_content_outside_frame():
    with pytest.raises(ValueError, match="outside frame"):
        PageSurfaceGeometry.build(
            logical_width=900,
            logical_height=100,
            frame_width=800,
            frame_height=100,
            content_origin_xy=(0, 0),
        )
