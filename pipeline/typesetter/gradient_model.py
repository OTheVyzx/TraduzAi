"""Canonical linear-gradient values shared by Style V2 consumers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np


LINEAR_GRADIENT_SPACE = "glyph_bbox_normalized"
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


@dataclass(frozen=True)
class GradientDetection:
    """Result of fitting a linear color field to supported glyph pixels."""

    value: dict[str, object] | None
    confidence: float
    reason: str
    metrics: dict[str, object]


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
        point.append(round(number, 6))
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


def _rgb_hex(color: np.ndarray) -> str:
    values = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    return "#" + "".join(f"{int(value):02X}" for value in values)


def detect_linear_gradient(
    image_rgb: np.ndarray,
    sampling_mask: np.ndarray,
) -> GradientDetection:
    """Fit a supported rank-one linear color field to glyph-interior samples."""

    rgb = np.asarray(image_rgb)
    binary = np.asarray(sampling_mask) > 0
    if rgb.ndim != 3 or rgb.shape[2] < 3 or binary.shape != rgb.shape[:2]:
        raise ValueError("gradient image and sampling mask must share dimensions")
    ys, xs = np.where(binary)
    if len(xs) < 24:
        return GradientDetection(
            None,
            0.0,
            "insufficient_gradient_spatial_support",
            {"sample_pixels": int(len(xs)), "supported_cells": 0},
        )

    x_span = max(1.0, float(xs.max() - xs.min()))
    y_span = max(1.0, float(ys.max() - ys.min()))
    xn = (xs.astype(np.float64) - float(xs.min())) / x_span
    yn = (ys.astype(np.float64) - float(ys.min())) / y_span
    samples = rgb[ys, xs, :3].astype(np.float64)
    cell_x = np.minimum(5, np.floor(xn * 6.0).astype(np.int32))
    cell_y = np.minimum(5, np.floor(yn * 6.0).astype(np.int32))
    cell_ids = cell_y * 6 + cell_x
    cell_positions: list[np.ndarray] = []
    cell_colors: list[np.ndarray] = []
    for cell_id in np.unique(cell_ids):
        selected = cell_ids == cell_id
        if int(np.count_nonzero(selected)) < 4:
            continue
        cell_positions.append(
            np.asarray(
                [np.median(xn[selected]), np.median(yn[selected])],
                dtype=np.float64,
            )
        )
        cell_colors.append(np.median(samples[selected], axis=0))
    metrics: dict[str, object] = {
        "sample_pixels": int(len(xs)),
        "supported_cells": len(cell_positions),
    }
    if len(cell_positions) < 6:
        return GradientDetection(
            None,
            0.0,
            "insufficient_gradient_spatial_support",
            metrics,
        )

    positions = np.asarray(cell_positions, dtype=np.float64)
    colors = np.asarray(cell_colors, dtype=np.float64)
    span_x = float(np.ptp(positions[:, 0]))
    span_y = float(np.ptp(positions[:, 1]))
    metrics.update({"span_x": round(span_x, 6), "span_y": round(span_y, 6)})
    if max(span_x, span_y) < 0.35 or min(span_x, span_y) < 0.20:
        return GradientDetection(
            None,
            0.0,
            "insufficient_gradient_spatial_support",
            metrics,
        )

    design = np.column_stack((np.ones(len(positions)), positions))
    coefficients = np.linalg.lstsq(design, colors, rcond=None)[0]
    spatial_coefficients = coefficients[1:, :]
    spatial_vectors, singular_values, _color_vectors = np.linalg.svd(
        spatial_coefficients,
        full_matrices=False,
    )
    axis = spatial_vectors[:, 0]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis = -axis
    singular_energy = np.square(singular_values)
    rank_one_energy = float(
        singular_energy[0] / max(1e-9, float(np.sum(singular_energy)))
    )

    raw_positions = np.column_stack((xn, yn))
    projections = raw_positions @ axis
    tail_fraction = min(0.125, max(0.025, 8.0 / float(len(projections))))
    low_projection = float(np.quantile(projections, tail_fraction))
    high_projection = float(np.quantile(projections, 1.0 - tail_fraction))
    low_samples = samples[projections <= low_projection]
    high_samples = samples[projections >= high_projection]
    first_color = np.median(low_samples, axis=0)
    second_color = np.median(high_samples, axis=0)
    endpoint_delta = float(np.linalg.norm(second_color - first_color))

    predicted = design @ coefficients
    residuals = np.linalg.norm(colors - predicted, axis=1)
    median_residual = float(np.median(residuals))
    centroid = np.mean(raw_positions, axis=0)
    centroid_projection = float(np.dot(centroid, axis))
    start = centroid + axis * (low_projection - centroid_projection)
    end = centroid + axis * (high_projection - centroid_projection)
    projection_capacity = max(1e-6, float(np.sum(np.abs(axis))))
    projection_coverage = min(
        1.0,
        max(0.0, (high_projection - low_projection) / projection_capacity),
    )
    residual_score = max(
        0.0,
        min(1.0, 1.0 - median_residual / max(1.0, endpoint_delta)),
    )
    confidence = min(
        0.99,
        0.35
        + 0.25 * min(1.0, endpoint_delta / 100.0)
        + 0.20 * rank_one_energy
        + 0.10 * projection_coverage
        + 0.10 * residual_score,
    )
    metrics.update(
        {
            "axis": [round(float(axis[0]), 6), round(float(axis[1]), 6)],
            "endpoint_delta_rgb": round(endpoint_delta, 6),
            "median_residual_rgb": round(median_residual, 6),
            "projection_coverage": round(projection_coverage, 6),
            "rank_one_explained_energy": round(rank_one_energy, 6),
            "tail_pixels": [int(len(low_samples)), int(len(high_samples))],
            "tail_fraction": round(tail_fraction, 6),
        }
    )
    if endpoint_delta < 24.0:
        return GradientDetection(None, 0.0, "solid_fill_no_gradient", metrics)
    if rank_one_energy < 0.72 or median_residual >= endpoint_delta:
        return GradientDetection(
            None,
            0.0,
            "nonlinear_or_contaminated_gradient",
            metrics,
        )
    if confidence < 0.60:
        return GradientDetection(
            None,
            confidence,
            "insufficient_gradient_spatial_support",
            metrics,
        )
    value = canonicalize_linear_gradient(
        {
            "kind": "linear",
            "colors": [_rgb_hex(first_color), _rgb_hex(second_color)],
            "stops": [0.0, 1.0],
            "start": start.tolist(),
            "end": end.tolist(),
            "coordinate_space": LINEAR_GRADIENT_SPACE,
        }
    )
    if value is None:
        return GradientDetection(None, 0.0, "invalid_gradient_axis", metrics)
    return GradientDetection(value, round(confidence, 6), "", metrics)
