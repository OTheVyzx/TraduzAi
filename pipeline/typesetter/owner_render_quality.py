"""Pure, immutable proportional-quality evaluation for owner renders."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np


OWNER_RENDER_QUALITY_SCHEMA_VERSION = 1
_VALID_STATUSES = frozenset(
    {
        "ok",
        "invalid",
        "below_minimum",
        "under_source_scale",
        "over_source_scale",
        "underfilled",
        "outside_safe_region",
    }
)


@dataclass(frozen=True)
class OwnerRenderQuality:
    schema_version: int
    status: str
    font_size_final: int
    minimum_legible_font_px: int
    source_ink_height_px: float | None
    render_ink_height_px: int
    source_x_height_px: float | None
    render_x_height_px: float
    source_scale_ratio: float | None
    x_height_ratio: float | None
    rendered_line_core_heights_px: tuple[int, ...]
    safe_height_occupancy: float
    safe_area_occupancy: float
    wrapped_line_count: int
    containment_status: str
    outside_safe_pixels: int
    page_width: int
    page_height: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != OWNER_RENDER_QUALITY_SCHEMA_VERSION:
            raise ValueError("unsupported owner render quality schema")
        if self.status not in _VALID_STATUSES:
            raise ValueError("invalid owner render quality status")
        object.__setattr__(
            self,
            "rendered_line_core_heights_px",
            tuple(int(value) for value in self.rendered_line_core_heights_px),
        )
        object.__setattr__(
            self,
            "reasons",
            tuple(sorted(set(str(value) for value in self.reasons))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "font_size_final": self.font_size_final,
            "minimum_legible_font_px": self.minimum_legible_font_px,
            "source_ink_height_px": self.source_ink_height_px,
            "render_ink_height_px": self.render_ink_height_px,
            "source_x_height_px": self.source_x_height_px,
            "render_x_height_px": self.render_x_height_px,
            "source_scale_ratio": self.source_scale_ratio,
            "x_height_ratio": self.x_height_ratio,
            "rendered_line_core_heights_px": list(
                self.rendered_line_core_heights_px
            ),
            "safe_height_occupancy": self.safe_height_occupancy,
            "safe_area_occupancy": self.safe_area_occupancy,
            "wrapped_line_count": self.wrapped_line_count,
            "containment_status": self.containment_status,
            "outside_safe_pixels": self.outside_safe_pixels,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OwnerRenderQuality":
        if not isinstance(payload, Mapping):
            raise ValueError("owner render quality payload must be a mapping")
        required = {item.name for item in fields(cls)}
        if set(payload) != required:
            raise ValueError("owner render quality payload fields are incomplete")
        values = dict(payload)
        values["rendered_line_core_heights_px"] = tuple(
            values["rendered_line_core_heights_px"]
        )
        values["reasons"] = tuple(values["reasons"])
        return cls(**values)


def _bbox(
    value: Sequence[int],
    *,
    page_width: int,
    page_height: int,
) -> tuple[int, int, int, int] | None:
    if isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    if any(type(item) is not int for item in value):
        return None
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= page_width and 0 <= y1 < y2 <= page_height):
        return None
    return x1, y1, x2, y2


def _canonical_mask(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    array = np.asarray(value)
    if array.shape != shape or array.ndim != 2:
        return None
    if not np.all((array == 0) | (array == 1) | (array == 255)):
        return None
    return np.ascontiguousarray(array > 0)


def _positive_finite(values: Sequence[Any]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric) and numeric > 0.0:
            result.append(numeric)
    return tuple(result)


def _line_core_heights(mask: np.ndarray) -> tuple[int, ...]:
    active_rows = np.any(mask, axis=1)
    heights: list[int] = []
    start: int | None = None
    for index, active in enumerate([*active_rows.tolist(), False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            heights.append(index - start)
            start = None
    return tuple(heights)


def _underfill_threshold(layout_profile: str, translated_text: str) -> float:
    profile = str(layout_profile or "").strip().casefold()
    length = len("".join(str(translated_text or "").split()))
    base = 0.14 if profile in {"card", "item_card", "system_card"} else 0.12
    if length <= 24:
        return max(base, 0.12)
    if length >= 100:
        return min(base, 0.08)
    return base


def evaluate_owner_render_quality(
    *,
    render_bbox: list[int],
    safe_bbox: list[int],
    safe_mask: np.ndarray,
    glyph_core_mask: np.ndarray,
    glyph_pixels: int,
    font_size_final: int,
    minimum_legible_font_px: int,
    source_ink_heights_px: tuple[int, ...],
    source_x_heights_px: tuple[float, ...],
    source_evidence_confidence: float,
    page_width: int,
    page_height: int,
    translated_text: str,
    layout_profile: str,
    trusted_container: bool,
) -> OwnerRenderQuality:
    """Evaluate render evidence without mutating text, layout, style, or pixels."""

    reasons: list[str] = []
    valid_page = (
        type(page_width) is int
        and type(page_height) is int
        and page_width > 0
        and page_height > 0
    )
    shape = (page_height, page_width) if valid_page else (0, 0)
    safe = _canonical_mask(safe_mask, shape) if valid_page else None
    glyph = _canonical_mask(glyph_core_mask, shape) if valid_page else None
    render_box = (
        _bbox(render_bbox, page_width=page_width, page_height=page_height)
        if valid_page
        else None
    )
    safe_box = (
        _bbox(safe_bbox, page_width=page_width, page_height=page_height)
        if valid_page
        else None
    )
    if not valid_page or safe is None or glyph is None or render_box is None or safe_box is None:
        reasons.append("invalid_geometry_metrics")

    line_heights = _line_core_heights(glyph) if glyph is not None else ()
    actual_glyph_pixels = int(np.count_nonzero(glyph)) if glyph is not None else 0
    if actual_glyph_pixels <= 0 or not line_heights:
        reasons.append("missing_render_ink")
    if type(glyph_pixels) is not int or glyph_pixels != actual_glyph_pixels:
        reasons.append("glyph_pixel_count_mismatch")
    if (
        type(font_size_final) is not int
        or type(minimum_legible_font_px) is not int
        or font_size_final <= 0
        or minimum_legible_font_px <= 0
    ):
        reasons.append("invalid_font_metrics")

    render_ink_height = int(round(float(median(line_heights)))) if line_heights else 0
    render_x_height = float(render_ink_height) * 0.70
    source_heights = _positive_finite(source_ink_heights_px)
    source_x_heights = _positive_finite(source_x_heights_px)
    source_ink_height = float(median(source_heights)) if source_heights else None
    source_x_height = float(median(source_x_heights)) if source_x_heights else None
    confidence_valid = (
        isinstance(source_evidence_confidence, (int, float))
        and not isinstance(source_evidence_confidence, bool)
        and math.isfinite(float(source_evidence_confidence))
        and 0.0 <= float(source_evidence_confidence) <= 1.0
    )
    if not confidence_valid:
        reasons.append("invalid_source_evidence_confidence")
    source_trusted = confidence_valid and float(source_evidence_confidence) >= 0.70
    if source_trusted and source_ink_height is None:
        reasons.append("missing_source_scale_metrics")

    source_scale_ratio = (
        float(render_ink_height) / source_ink_height
        if render_ink_height > 0 and source_ink_height
        else None
    )
    x_height_ratio = (
        render_x_height / source_x_height
        if render_x_height > 0 and source_x_height
        else None
    )
    safe_height = (safe_box[3] - safe_box[1]) if safe_box else 0
    safe_pixels = int(np.count_nonzero(safe)) if safe is not None else 0
    safe_height_occupancy = (
        float(render_box[3] - render_box[1]) / float(safe_height)
        if render_box and safe_height > 0
        else 0.0
    )
    safe_area_occupancy = (
        float(actual_glyph_pixels) / float(safe_pixels) if safe_pixels > 0 else 0.0
    )
    outside_safe_pixels = (
        int(np.count_nonzero(glyph & ~safe))
        if glyph is not None and safe is not None
        else actual_glyph_pixels
    )
    containment_status = "ok" if outside_safe_pixels == 0 else "outside_safe_region"

    invalid = bool(reasons)
    status = "invalid" if invalid else "ok"
    if not invalid and outside_safe_pixels > 0:
        reasons.append("outside_safe_region")
        status = "outside_safe_region"
    elif not invalid and font_size_final < minimum_legible_font_px:
        reasons.append("below_minimum")
        status = "below_minimum"
    elif not invalid and source_scale_ratio is not None and source_scale_ratio < 0.40:
        # Even weak source-size evidence must not silently accept a catastrophic
        # shrink.  The owner is sent to review; the value is not treated as a
        # calibrated source-scale measurement.
        reasons.append("catastrophic_source_scale_mismatch")
        status = "under_source_scale"
    elif not invalid and source_trusted and source_scale_ratio is not None:
        if source_scale_ratio < 0.75:
            reasons.append("under_source_scale")
            status = "under_source_scale"
        elif source_scale_ratio > 1.35:
            reasons.append("over_source_scale")
            status = "over_source_scale"
    elif (
        not invalid
        and trusted_container is True
        and safe_height_occupancy
        < _underfill_threshold(layout_profile, translated_text)
    ):
        reasons.append("underfilled")
        status = "underfilled"

    return OwnerRenderQuality(
        schema_version=OWNER_RENDER_QUALITY_SCHEMA_VERSION,
        status=status,
        font_size_final=int(font_size_final) if type(font_size_final) is int else 0,
        minimum_legible_font_px=(
            int(minimum_legible_font_px)
            if type(minimum_legible_font_px) is int
            else 0
        ),
        source_ink_height_px=source_ink_height,
        render_ink_height_px=render_ink_height,
        source_x_height_px=source_x_height,
        render_x_height_px=render_x_height,
        source_scale_ratio=source_scale_ratio,
        x_height_ratio=x_height_ratio,
        rendered_line_core_heights_px=line_heights,
        safe_height_occupancy=safe_height_occupancy,
        safe_area_occupancy=safe_area_occupancy,
        wrapped_line_count=len(line_heights),
        containment_status=containment_status,
        outside_safe_pixels=outside_safe_pixels,
        page_width=int(page_width) if type(page_width) is int else 0,
        page_height=int(page_height) if type(page_height) is int else 0,
        reasons=tuple(reasons),
    )


__all__ = [
    "OWNER_RENDER_QUALITY_SCHEMA_VERSION",
    "OwnerRenderQuality",
    "evaluate_owner_render_quality",
]
