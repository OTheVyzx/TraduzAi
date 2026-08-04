"""Canonical linear-gradient values shared by Style V2 consumers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np


LINEAR_GRADIENT_SPACE = "glyph_bbox_normalized"
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _canonical_color(value: object) -> str | None:
    color = str(value or "").strip()
    if not _HEX_COLOR.fullmatch(color):
        return None
    return color.upper()


def _canonical_point(value: object) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        return None
    point: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, Real):
            return None
        number = float(component)
        if not math.isfinite(number):
            return None
        point.append(round(max(0.0, min(1.0, number)), 6))
    return point


def canonicalize_linear_gradient(value: object) -> dict[str, object] | None:
    """Return a deterministic two-stop normalized linear-gradient value."""

    if isinstance(value, Mapping):
        if str(value.get("kind") or "linear") != "linear":
            return None
        if str(value.get("coordinate_space") or LINEAR_GRADIENT_SPACE) != LINEAR_GRADIENT_SPACE:
            return None
        raw_colors = value.get("colors")
        start = _canonical_point(value.get("start"))
        end = _canonical_point(value.get("end"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw_colors = value
        start = [0.5, 0.0]
        end = [0.5, 1.0]
    else:
        return None

    if (
        not isinstance(raw_colors, Sequence)
        or isinstance(raw_colors, (str, bytes, bytearray))
        or len(raw_colors) < 2
        or start is None
        or end is None
    ):
        return None
    colors = [_canonical_color(raw_colors[0]), _canonical_color(raw_colors[1])]
    if any(color is None for color in colors):
        return None
    axis_length = math.hypot(end[0] - start[0], end[1] - start[1])
    if axis_length < 1e-6:
        return None
    return {
        "kind": "linear",
        "colors": [str(colors[0]), str(colors[1])],
        "stops": [0.0, 1.0],
        "start": list(start),
        "end": list(end),
        "coordinate_space": LINEAR_GRADIENT_SPACE,
    }


def gradient_parameter_map(mask: np.ndarray, gradient: object) -> np.ndarray:
    """Evaluate one block-level normalized field over the mask's glyph bbox."""

    canonical = canonicalize_linear_gradient(gradient)
    if canonical is None:
        raise ValueError("invalid linear gradient")
    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or not np.any(binary):
        raise ValueError("gradient mask must contain glyph pixels")
    ys, xs = np.where(binary)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    grid_y, grid_x = np.indices(binary.shape, dtype=np.float32)
    x_normalized = (grid_x - x_min) / max(1.0, float(x_max - x_min))
    y_normalized = (grid_y - y_min) / max(1.0, float(y_max - y_min))
    start = np.asarray(canonical["start"], dtype=np.float32)
    end = np.asarray(canonical["end"], dtype=np.float32)
    axis = end - start
    denominator = float(np.dot(axis, axis))
    parameter = (
        (x_normalized - start[0]) * axis[0]
        + (y_normalized - start[1]) * axis[1]
    ) / denominator
    return np.clip(parameter, 0.0, 1.0)


def render_linear_gradient_rgb(mask: np.ndarray, gradient: object) -> np.ndarray:
    """Return RGB pixels for the canonical field, zero outside the mask."""

    canonical = canonicalize_linear_gradient(gradient)
    if canonical is None:
        raise ValueError("invalid linear gradient")
    binary = np.asarray(mask) > 0
    parameter = gradient_parameter_map(binary, canonical)[..., None]
    colors = canonical["colors"]
    start_color = np.asarray(
        [int(str(colors[0])[index : index + 2], 16) for index in (1, 3, 5)],
        dtype=np.float32,
    )
    end_color = np.asarray(
        [int(str(colors[1])[index : index + 2], 16) for index in (1, 3, 5)],
        dtype=np.float32,
    )
    pixels = np.rint(start_color * (1.0 - parameter) + end_color * parameter)
    output = np.zeros((*binary.shape, 3), dtype=np.uint8)
    output[binary] = np.clip(pixels[binary], 0, 255).astype(np.uint8)
    return output
