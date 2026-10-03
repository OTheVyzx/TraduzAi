"""Source-calibrated lettering contract. No rendering, files, or fallback.

Both a preview builder and the product renderer must pass their *actual* glyph
mask and a verified interior mask to this gate. A missing mask is REVIEW.
This module intentionally does not infer balloon interiors from RGB thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Any

import cv2
import numpy as np


@dataclass(frozen=True)
class OpticalCalibration:
    font_name: str
    nominal_size_px: int
    line_advance_px: int
    optical_body_px: float
    optical_tolerance_px: float
    min_contour_px: float
    max_center_drift_px: float
    source_evidence_sha256: str


def resolve_calibration(raw: Mapping[str, Any], *, page_width_px: int) -> OpticalCalibration:
    """Scale a source-backed work profile to the rendered page resolution."""
    try:
        name = str(raw['font_name'])
        ref_width = int(raw['reference_page_width_px'])
        evidence = str(raw['source_evidence_sha256'])
        size = float(raw['nominal_size_px'])
        advance = float(raw['line_advance_px'])
        body = float(raw['optical_body_px'])
        tolerance = float(raw['optical_tolerance_px'])
        margin = float(raw['min_contour_px'])
        drift = float(raw['max_center_drift_px'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('optical_calibration_incomplete') from exc
    values = (size, advance, body, tolerance, margin, drift)
    if (not name or ref_width <= 0 or page_width_px <= 0 or len(evidence) != 64
            or any(not np.isfinite(v) or v <= 0 for v in values)):
        raise ValueError('optical_calibration_invalid')
    scale = page_width_px / ref_width
    return OpticalCalibration(
        name, max(1, round(size * scale)), max(1, round(advance * scale)),
        body * scale, tolerance * scale, margin * scale, drift * scale, evidence,
    )


@dataclass(frozen=True)
class ComfortDecision:
    status: Literal["OK", "REVIEW_REQUIRED"]
    reasons: tuple[str, ...]
    source_center_xy: tuple[float, float] | None
    ink_center_xy: tuple[float, float] | None
    ink_center_drift_px: float | None
    min_contour_px: float | None
    min_contour_letter_heights: float | None
    apparent_body_px: tuple[int, ...]
    calibration_body_px: int | None
    ink_occupancy: float | None
    mask_kind: str | None


def source_center(observation_boxes: Iterable[Iterable[float]]) -> tuple[float, float] | None:
    boxes = [tuple(float(v) for v in box) for box in observation_boxes]
    if not boxes or any(len(box) != 4 or box[2] <= box[0] or box[3] <= box[1] for box in boxes):
        return None
    return ((min(box[0] for box in boxes) + max(box[2] for box in boxes)) / 2,
            (min(box[1] for box in boxes) + max(box[3] for box in boxes)) / 2)


def _ink_center(mask: np.ndarray) -> tuple[float, float] | None:
    yy, xx = np.nonzero(mask > 127)
    if not len(xx):
        return None
    # Bounding-box center describes the complete lettering block, not its
    # darkest or longest line. This matches the original OCR union contract.
    return ((float(xx.min()) + float(xx.max())) / 2,
            (float(yy.min()) + float(yy.max())) / 2)


def evaluate(
    *,
    font_name: str,
    nominal_size_px: int,
    line_advance_px: int,
    observation_boxes: Iterable[Iterable[float]],
    glyph_mask: np.ndarray | None,
    line_body_heights_px: Iterable[int],
    calibration_body_px: int | None,
    interior_mask: np.ndarray | None,
    mask_kind: str | None,
    visual_role: str = "ordinary_dialogue",
    visual_exception_evidence: str | None = None,
    fallback_used: bool = False,
    calibration: OpticalCalibration | None = None,
) -> ComfortDecision:
    reasons: list[str] = []
    source_xy = source_center(observation_boxes)
    if source_xy is None:
        reasons.append("source_anchor_missing")
    if fallback_used:
        reasons.append("historical_fallback_used")

    ordinary = visual_role == "ordinary_dialogue"
    if calibration is None:
        reasons.append("optical_calibration_missing")
    if not ordinary and not visual_exception_evidence:
        reasons.append("visual_exception_evidence_missing")
    if ordinary:
        if calibration is not None and font_name != calibration.font_name:
            reasons.append("font_calibration_mismatch")
        if calibration is not None and nominal_size_px != calibration.nominal_size_px:
            reasons.append("nominal_size_changed")
        if calibration is not None and line_advance_px != calibration.line_advance_px:
            reasons.append("line_advance_changed")

    bodies = tuple(int(v) for v in line_body_heights_px)
    if not bodies:
        reasons.append("line_body_metrics_missing")
    # Accents and descenders change a particular line's visible bbox. The
    # cross-font comparison uses the same uppercase cap sample, e.g. HATO.
    if calibration_body_px is None:
        reasons.append("optical_calibration_missing")
    elif ordinary and calibration is not None and abs(calibration_body_px - calibration.optical_body_px) > calibration.optical_tolerance_px:
        reasons.append("apparent_body_outside_target")

    ink_xy = None
    drift = None
    margin = None
    relative_margin = None
    occupancy = None
    if glyph_mask is None or glyph_mask.ndim != 2:
        reasons.append("glyph_mask_missing")
    else:
        ink_xy = _ink_center(glyph_mask)
        if ink_xy is None:
            reasons.append("glyph_mask_empty")
        elif source_xy is not None:
            drift = float(np.hypot(ink_xy[0] - source_xy[0], ink_xy[1] - source_xy[1]))
            if calibration is not None and drift > calibration.max_center_drift_px:
                reasons.append("ink_center_drift")

    if interior_mask is None:
        reasons.append("interior_missing")
    elif mask_kind != "verified_closed_interior":
        reasons.append("interior_not_verified")
    if interior_mask is not None and glyph_mask is not None and glyph_mask.ndim == 2 and glyph_mask.shape == interior_mask.shape:
        ink = glyph_mask > 127
        interior = interior_mask > 0
        if np.any(ink):
            outside = int(np.count_nonzero(ink & ~interior))
            if outside:
                reasons.append("ink_outside_interior")
            distance = cv2.distanceTransform(interior.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            margin = float(np.min(distance[ink]))
            if calibration is not None and margin < calibration.min_contour_px:
                reasons.append("contour_margin_low")
            if calibration_body_px:
                relative_margin = margin / float(calibration_body_px)
            # Fraction of verified interior covered by ink; area alone is a
            # diagnostic, not an approval criterion for long dialogue.
            occupancy = float(np.count_nonzero(ink & interior)) / max(1, int(np.count_nonzero(interior)))
    elif interior_mask is not None:
        reasons.append("mask_shape_mismatch")

    return ComfortDecision(
        "REVIEW_REQUIRED" if reasons else "OK", tuple(dict.fromkeys(reasons)),
        source_xy, ink_xy, drift, margin, relative_margin, bodies,
        calibration_body_px,
        occupancy, mask_kind,
    )
