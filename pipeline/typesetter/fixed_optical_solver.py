"""Generic fixed optical size line search for an owner with verified interior.

No chapter coordinates, text, masks, or font overrides are embedded here.
Failure is a review result; the function never edits source pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .comfort_contract import OpticalCalibration, source_center
from .stable_baseline import rasterize


@dataclass(frozen=True)
class FixedOpticalLayout:
    status: str
    reason: str
    lines: tuple[str, ...] = ()
    positions: tuple[tuple[int, int], ...] = ()
    min_contour_px: float | None = None
    center_drift_px: float | None = None


def solve_fixed_optical(
    *, text: str, font_path: str, font_name: str,
    observation_boxes: list[list[float]], interior_mask: np.ndarray | None,
    mask_kind: str | None, calibration: OpticalCalibration,
    max_lines: int = 8,
) -> FixedOpticalLayout:
    if interior_mask is None or mask_kind != 'verified_closed_interior' or interior_mask.ndim != 2:
        return FixedOpticalLayout('REVIEW_REQUIRED', 'interior_not_verified')
    center = source_center(observation_boxes)
    if center is None:
        return FixedOpticalLayout('REVIEW_REQUIRED', 'source_anchor_missing')
    words = text.split()
    if not words:
        return FixedOpticalLayout('REVIEW_REQUIRED', 'text_missing')
    if font_name != calibration.font_name:
        return FixedOpticalLayout('REVIEW_REQUIRED', 'font_not_calibrated')
    size = calibration.nominal_size_px
    interior = interior_mask > 0
    yy, xx = np.nonzero(interior)
    if not len(xx):
        return FixedOpticalLayout('REVIEW_REQUIRED', 'interior_empty')
    width_max = int(xx.max() - xx.min() + 1)
    distance = cv2.distanceTransform(interior.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    @lru_cache(maxsize=None)
    def glyph(line: str) -> np.ndarray:
        return rasterize(font_path, size, line)

    @lru_cache(maxsize=None)
    def wraps(index: int, width: int) -> tuple[tuple[str, ...], ...]:
        if index == len(words):
            return ((),)
        found = []
        for end in range(index + 1, len(words) + 1):
            line = ' '.join(words[index:end])
            if glyph(line).shape[1] > width:
                break
            for rest in wraps(end, width):
                if len(rest) < max_lines:
                    found.append((line,) + rest)
        found.sort(key=lambda lines: (len(lines), sum((width - glyph(line).shape[1]) ** 2 for line in lines)))
        return tuple(found[:16])

    best: FixedOpticalLayout | None = None
    for width in range(max(1, width_max - 24), max(60, int(width_max * .55)) - 1, -8):
        for lines in wraps(0, width):
            if not lines or len(lines) > max_lines or ' '.join(lines) != ' '.join(words):
                continue
            masks = [glyph(line) for line in lines]
            positions = tuple((round(center[0] - mask.shape[1] / 2),
                               round(center[1] + (i - (len(lines)-1)/2) * calibration.line_advance_px - mask.shape[0]/2))
                              for i, mask in enumerate(masks))
            ink = np.zeros(interior.shape, np.uint8)
            separate = True
            for mask, (x, y) in zip(masks, positions):
                if x < 0 or y < 0 or x + mask.shape[1] > ink.shape[1] or y + mask.shape[0] > ink.shape[0]:
                    separate = False
                    break
                dest = ink[y:y+mask.shape[0], x:x+mask.shape[1]]
                if np.any((dest > 127) & (mask > 127)):
                    separate = False
                    break
                np.maximum(dest, mask, out=dest)
            if not separate:
                continue
            iy, ix = np.nonzero(ink > 0)
            if not len(ix) or np.any(~interior[iy, ix]):
                continue
            clearance = float(distance[iy, ix].min())
            if clearance < calibration.min_contour_px:
                continue
            ink_center = ((int(ix.min()) + int(ix.max())) / 2, (int(iy.min()) + int(iy.max())) / 2)
            drift = float(np.hypot(ink_center[0] - center[0], ink_center[1] - center[1]))
            if drift > calibration.max_center_drift_px:
                continue
            candidate = FixedOpticalLayout('OK', '', tuple(lines), positions, clearance, drift)
            if best is None or (candidate.min_contour_px, -len(candidate.lines)) > (best.min_contour_px, -len(best.lines)):
                best = candidate
        if best is not None:
            return best
    return FixedOpticalLayout('REVIEW_REQUIRED', 'fixed_size_text_overflow_or_anchor_drift')
