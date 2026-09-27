"""Bounded residual-glyph cleanup on an authenticated white source body."""

from __future__ import annotations

import cv2
import numpy as np


def clear_residual_white_glyphs(clean: np.ndarray, source_original: np.ndarray, prepared: dict,
                                observation_bbox: list[int], *, pad_px: int = 6) -> tuple[np.ndarray, dict]:
    """Remove only dark residual ink in one observation, away from its contour.

    The caller authenticates the observation as source lettering. This routine
    does not infer text from a dark region or expand cleanup into neighboring art.
    """
    if (clean.ndim != 3 or clean.shape[2] != 3 or clean.dtype != np.uint8 or
            source_original.shape != clean.shape or source_original.dtype != np.uint8):
        raise ValueError("invalid clean or original canvas")
    ox1, oy1, ox2, oy2 = map(int, observation_bbox)
    x1, y1, x2, y2 = ox1, oy1, ox2, oy2
    rx1, ry1, rx2, ry2 = prepared["source_roi_bbox"]
    x1, y1, x2, y2 = max(rx1, x1-pad_px), max(ry1, y1-pad_px), min(rx2, x2+pad_px), min(ry2, y2+pad_px)
    if not (0 <= x1 < x2 <= clean.shape[1] and 0 <= y1 < y2 <= clean.shape[0]):
        raise ValueError("invalid residual observation")
    region = clean[y1:y2, x1:x2]
    source_region = source_original[y1:y2, x1:x2]
    distance = prepared["distance"][y1-ry1:y2-ry1, x1-rx1:x2-rx1]
    # Never touch the contour or a colored drawing. Lettering here is dark,
    # nearly neutral, and fully inside the original white speech body.
    chroma = region.max(2).astype(np.int16) - region.min(2).astype(np.int16)
    source_chroma = source_region.max(2).astype(np.int16) - source_region.min(2).astype(np.int16)
    source_ink = (source_region.max(2) < 245) & (source_chroma <= 10)
    _, labels = cv2.connectedComponents(source_ink.astype(np.uint8), 8)
    seed = labels[max(0, oy1-y1):min(y2-y1, oy2-y1),
                  max(0, ox1-x1):min(x2-x1, ox2-x1)]
    seeded = np.unique(seed[seed > 0])
    linked_to_observation = np.isin(labels, seeded) & (labels > 0)
    # Compression can leave faint, disconnected fringe pixels beside a glyph.
    # JPEG fringe may lie several pixels from the dark core of a glyph. Keep
    # the expansion within the authenticated observation, white body and
    # contour-distance gates below.
    fringe_of_selected = cv2.dilate(linked_to_observation.astype(np.uint8),
                                    np.ones((9, 9), np.uint8)) > 0
    eligible = ((region.max(2) < 245) & (chroma <= 10) & source_ink &
                fringe_of_selected &
                (distance >= max(3, prepared["minimum_px"]-1)))
    if not np.any(eligible):
        raise ValueError("no source-bound residual ink in observation")
    candidate = clean.copy()
    candidate_region = candidate[y1:y2, x1:x2]
    candidate_region[eligible] = 255
    remaining = (candidate_region.max(2) < 245) & (chroma <= 10) & (distance >= max(3, prepared["minimum_px"]-1))
    return candidate, dict(schema="source_white_glyph_cleanup_v1",
                           observation_bbox=list(map(int, observation_bbox)),
                           applied_bbox=[x1, y1, x2, y2],
                           changed_pixels=int(np.count_nonzero(eligible)),
                           seeded_source_components=int(len(seeded)),
                           extension_pad_px=pad_px,
                           residual_dark_pixels=int(np.count_nonzero(remaining)),
                           contour_distance_min_px=float(distance[eligible].min()),
                           color_or_contour_pixels_preserved=int(np.count_nonzero((region.max(2)<245) & ~eligible)))
