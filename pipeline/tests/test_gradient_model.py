from __future__ import annotations

import pytest
import numpy as np

from typesetter.gradient_model import (
    canonicalize_linear_gradient,
    gradient_parameter_map,
    render_linear_gradient_rgb,
)


def test_legacy_two_color_gradient_becomes_vertical_canonical_field() -> None:
    assert canonicalize_linear_gradient(["#ff0000", "#0000ff"]) == {
        "kind": "linear",
        "colors": ["#FF0000", "#0000FF"],
        "stops": [0.0, 1.0],
        "start": [0.5, 0.0],
        "end": [0.5, 1.0],
        "coordinate_space": "glyph_bbox_normalized",
    }


def test_structured_gradient_is_rounded_and_preserves_direction() -> None:
    value = canonicalize_linear_gradient(
        {
            "kind": "linear",
            "colors": ["#123abc", "#abcdef"],
            "stops": [0, 1],
            "start": [0.123456789, 0.9],
            "end": [0.876543219, 0.1],
            "coordinate_space": "glyph_bbox_normalized",
        }
    )

    assert value is not None
    assert value["colors"] == ["#123ABC", "#ABCDEF"]
    assert value["start"] == [0.123457, 0.9]
    assert value["end"] == [0.876543, 0.1]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        ["#000000"],
        ["bad", "#FFFFFF"],
        {
            "colors": ["#000000", "#FFFFFF"],
            "start": [0, 0],
            "end": [0, 0],
        },
    ],
)
def test_invalid_or_degenerate_gradient_abstains(value: object) -> None:
    assert canonicalize_linear_gradient(value) is None


def _gradient(
    colors: tuple[str, str],
    start: tuple[float, float],
    end: tuple[float, float],
) -> dict[str, object]:
    return {
        "kind": "linear",
        "colors": list(colors),
        "stops": [0.0, 1.0],
        "start": list(start),
        "end": list(end),
        "coordinate_space": "glyph_bbox_normalized",
    }


def test_parameter_map_projects_the_complete_glyph_bbox_onto_the_axis() -> None:
    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[1:6, 2:8] = 255

    values = gradient_parameter_map(
        mask,
        _gradient(("#FF0000", "#0000FF"), (0.0, 0.5), (1.0, 0.5)),
    )

    assert values[3, 2] == pytest.approx(0.0)
    assert values[3, 7] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("gradient", "first_pixel", "last_pixel", "expected_first", "expected_last"),
    [
        (
            _gradient(("#FF0000", "#0000FF"), (0.5, 0.0), (0.5, 1.0)),
            (1, 2),
            (5, 2),
            (255, 0, 0),
            (0, 0, 255),
        ),
        (
            _gradient(("#FF0000", "#FFFF00"), (0.0, 0.5), (1.0, 0.5)),
            (1, 2),
            (1, 7),
            (255, 0, 0),
            (255, 255, 0),
        ),
        (
            _gradient(("#0000FF", "#00FF00"), (0.0, 0.0), (1.0, 1.0)),
            (1, 2),
            (5, 7),
            (0, 0, 255),
            (0, 255, 0),
        ),
    ],
)
def test_render_linear_gradient_reaches_declared_supported_endpoints(
    gradient: dict[str, object],
    first_pixel: tuple[int, int],
    last_pixel: tuple[int, int],
    expected_first: tuple[int, int, int],
    expected_last: tuple[int, int, int],
) -> None:
    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[1:6, 2:8] = 255

    pixels = render_linear_gradient_rgb(mask, gradient)

    assert np.max(np.abs(pixels[first_pixel].astype(int) - expected_first)) <= 2
    assert np.max(np.abs(pixels[last_pixel].astype(int) - expected_last)) <= 2
    assert np.count_nonzero(pixels[mask == 0]) == 0


def test_reversed_axis_reverses_the_spatial_progression() -> None:
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 1:6] = 255
    gradient = _gradient(
        ("#FF0000", "#0000FF"),
        (1.0, 0.5),
        (0.0, 0.5),
    )

    pixels = render_linear_gradient_rgb(mask, gradient)

    assert tuple(pixels[2, 1]) == (0, 0, 255)
    assert tuple(pixels[2, 5]) == (255, 0, 0)
