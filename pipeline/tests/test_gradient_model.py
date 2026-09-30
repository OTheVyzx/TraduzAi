from __future__ import annotations

import pytest
import numpy as np

from typesetter.gradient_model import (
    canonicalize_linear_gradient,
    detect_linear_gradient,
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
        ["#111111", "#111111"],
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


def _hex_rgb(value: str) -> np.ndarray:
    return np.asarray([int(value[index : index + 2], 16) for index in (1, 3, 5)])


def _uneven_multiline_mask() -> np.ndarray:
    mask = np.zeros((150, 280), dtype=np.uint8)
    rows = (
        (18, 22, 258),
        (42, 48, 232),
        (66, 14, 266),
        (90, 38, 246),
        (114, 18, 262),
    )
    for y, left, right in rows:
        for x in range(left, right, 14):
            mask[y : y + 11, x : min(x + 8, right)] = 255
    return mask


def _paint_synthetic_source(
    mask: np.ndarray,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    colors: tuple[str, str],
) -> np.ndarray:
    image = np.full((*mask.shape, 3), 246, dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    grid_y, grid_x = np.indices(mask.shape, dtype=np.float32)
    xn = (grid_x - x0) / max(1.0, float(x1 - x0))
    yn = (grid_y - y0) / max(1.0, float(y1 - y0))
    axis = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    parameter = (
        (xn - start[0]) * axis[0] + (yn - start[1]) * axis[1]
    ) / float(np.dot(axis, axis))
    parameter = np.clip(parameter, 0.0, 1.0)[..., None]
    first = _hex_rgb(colors[0]).astype(np.float32)
    second = _hex_rgb(colors[1]).astype(np.float32)
    field = np.rint(first * (1.0 - parameter) + second * parameter).astype(np.uint8)
    image[mask > 0] = field[mask > 0]
    return image


def _axis(value: dict[str, object]) -> np.ndarray:
    return np.asarray(value["end"], dtype=np.float64) - np.asarray(
        value["start"], dtype=np.float64
    )


@pytest.mark.parametrize(
    ("start", "end", "colors"),
    [
        ((0.5, 0.0), (0.5, 1.0), ("#6633CC", "#08080A")),
        ((0.0, 0.5), (1.0, 0.5), ("#E02020", "#F0E020")),
        ((0.0, 0.0), (1.0, 1.0), ("#2040E0", "#20D050")),
        ((0.0, 0.0), (1.0, 1.0), ("#F4E0B0", "#101040")),
        ((1.0, 0.0), (0.0, 1.0), ("#E020A0", "#20D8E8")),
    ],
)
def test_detect_linear_gradient_recovers_direction_and_supported_colors(
    start: tuple[float, float],
    end: tuple[float, float],
    colors: tuple[str, str],
) -> None:
    mask = _uneven_multiline_mask()
    image = _paint_synthetic_source(mask, start=start, end=end, colors=colors)

    detection = detect_linear_gradient(image, mask)

    assert detection.value is not None, detection.reason
    assert detection.confidence >= 0.60
    expected_axis = np.asarray(end) - np.asarray(start)
    detected_axis = _axis(detection.value)
    cosine = float(
        np.dot(expected_axis, detected_axis)
        / (np.linalg.norm(expected_axis) * np.linalg.norm(detected_axis))
    )
    assert abs(cosine) >= 0.93
    detected_colors = detection.value["colors"]
    expected_colors = colors if cosine >= 0.0 else tuple(reversed(colors))
    for detected, expected in zip(detected_colors, expected_colors):
        assert np.linalg.norm(_hex_rgb(detected) - _hex_rgb(expected)) <= 24.0


def test_detect_linear_gradient_abstains_for_solid_fill() -> None:
    mask = _uneven_multiline_mask()
    image = np.full((*mask.shape, 3), 246, dtype=np.uint8)
    image[mask > 0] = (18, 18, 18)

    detection = detect_linear_gradient(image, mask)

    assert detection.value is None
    assert detection.reason == "solid_fill_no_gradient"


def test_detect_linear_gradient_abstains_for_sparse_support() -> None:
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[5:8, 6:11] = 255
    image = _paint_synthetic_source(
        mask,
        start=(0.0, 0.0),
        end=(1.0, 1.0),
        colors=("#FF0000", "#0000FF"),
    )

    detection = detect_linear_gradient(image, mask)

    assert detection.value is None
    assert detection.reason == "insufficient_gradient_spatial_support"
