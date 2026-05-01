"""
Connected Balloon Splitter v2.

Detects whether a white balloon contour actually contains two (or more)
connected lobes — joined by a narrow neck — and splits the text assignment
so each lobe receives the OCR box(es) that belong to it.

Strategy (layered, each layer adds confidence):
  1. Ratio test: bbox is unusually wide or tall for a single text block.
  2. OCR multi-box: two or more OCR bboxes inside the same balloon contour.
  3. Horizontal/vertical gap analysis: empty horizontal/vertical stripe separates
     distinct text clusters.
  4. Watershed / skeleton neck detection (when OpenCV is available).
  5. Centre-of-mass split: assign each OCR box to the nearest lobe centre.
"""
from __future__ import annotations

import math
from typing import Optional

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# --- Data types ---

Bbox = list[int]  # [x1, y1, x2, y2]


def _bbox_area(b: Bbox) -> int:
    return max(0, (b[2] - b[0]) * (b[3] - b[1]))


def _bbox_center(b: Bbox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --- Core API ---

def detect_connected_balloon(
    *,
    balloon_bbox: Bbox,
    ocr_bboxes: list[Bbox],
    balloon_mask: Optional["np.ndarray"] = None,  # type: ignore[name-defined]
    min_confidence: float = 0.5,
) -> dict:
    """
    Determine whether balloon_bbox contains connected lobes.

    Args:
        balloon_bbox: outer balloon [x1,y1,x2,y2].
        ocr_bboxes: OCR text boxes detected inside this balloon.
        balloon_mask: optional binary mask (uint8, same coords as page).
        min_confidence: confidence threshold to call it connected.

    Returns:
        {
            "connected_balloon": bool,
            "orientation": "horizontal"|"vertical"|"diagonal"|"single",
            "lobes": [
                {
                    "bbox": [...],
                    "safe_bbox": [...],          # = bbox with standard padding
                    "assigned_text_ids": [int]   # indices into ocr_bboxes
                }
            ],
            "confidence": float,
            "method": str,
        }
    """
    result_single = {
        "connected_balloon": False,
        "orientation": "single",
        "lobes": [],
        "confidence": 0.0,
        "method": "none",
    }

    if len(ocr_bboxes) < 2:
        return result_single

    # --- Signal 1: two well-separated OCR centres ---
    centres = [_bbox_center(b) for b in ocr_bboxes]
    bw = balloon_bbox[2] - balloon_bbox[0]
    bh = balloon_bbox[3] - balloon_bbox[1]

    # Pairwise distance between the two widest-apart OCR centres
    max_sep = 0.0
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            max_sep = max(max_sep, _dist(centres[i], centres[j]))

    # If the furthest pair of OCR centres spans >35% of the balloon dimension
    min_dim = min(bw, bh)
    max_dim = max(bw, bh)
    sep_ratio = max_sep / max(1, max_dim)
    confidence = 0.0

    if sep_ratio > 0.35:
        confidence += 0.4

    # --- Signal 2: aspect ratio unusually large ---
    aspect = max_dim / max(1, min_dim)
    if aspect > 1.9:
        confidence += 0.2

    # --- Signal 3: horizontal/vertical gap between OCR clusters ---
    gap_found, orientation = _find_gap(balloon_bbox, ocr_bboxes)
    if gap_found:
        confidence += 0.3

    # --- Signal 4: mask neck detection (when available) ---
    if _CV2_AVAILABLE and balloon_mask is not None:
        neck_found, neck_orient = _detect_neck(balloon_bbox, balloon_mask)
        if neck_found:
            confidence += 0.4
            if neck_orient:
                orientation = neck_orient

    if confidence < min_confidence:
        return {**result_single, "confidence": confidence}

    # --- Build lobes ---
    if not orientation or orientation == "single":
        orientation = _infer_orientation(balloon_bbox, ocr_bboxes)

    lobes = _build_lobes(balloon_bbox, ocr_bboxes, orientation)

    return {
        "connected_balloon": True,
        "orientation": orientation,
        "lobes": lobes,
        "confidence": min(1.0, confidence),
        "method": "gap+centre" if gap_found else "centre",
    }


def _find_gap(balloon_bbox: Bbox, ocr_bboxes: list[Bbox]) -> tuple[bool, str]:
    """Look for an empty horizontal or vertical stripe between OCR boxes."""
    if len(ocr_bboxes) < 2:
        return False, ""

    bx1, by1, bx2, by2 = balloon_bbox

    # Horizontal gap: sort by x-centre, check if there's a gap wider than
    # the average OCR width between consecutive boxes.
    sorted_x = sorted(ocr_bboxes, key=lambda b: (b[0] + b[2]) / 2)
    avg_w = sum(b[2] - b[0] for b in ocr_bboxes) / len(ocr_bboxes)
    for i in range(len(sorted_x) - 1):
        gap = sorted_x[i + 1][0] - sorted_x[i][2]
        if gap > avg_w * 0.6:
            return True, "horizontal"

    # Vertical gap: sort by y-centre
    sorted_y = sorted(ocr_bboxes, key=lambda b: (b[1] + b[3]) / 2)
    avg_h = sum(b[3] - b[1] for b in ocr_bboxes) / len(ocr_bboxes)
    for i in range(len(sorted_y) - 1):
        gap = sorted_y[i + 1][1] - sorted_y[i][3]
        if gap > avg_h * 0.6:
            return True, "vertical"

    return False, ""


def _infer_orientation(balloon_bbox: Bbox, ocr_bboxes: list[Bbox]) -> str:
    bw = balloon_bbox[2] - balloon_bbox[0]
    bh = balloon_bbox[3] - balloon_bbox[1]
    if bw > bh * 1.4:
        return "horizontal"
    if bh > bw * 1.4:
        return "vertical"
    # Fall back to actual centre spread
    centres = [_bbox_center(b) for b in ocr_bboxes]
    dx = max(c[0] for c in centres) - min(c[0] for c in centres)
    dy = max(c[1] for c in centres) - min(c[1] for c in centres)
    return "horizontal" if dx >= dy else "vertical"


def _detect_neck(
    balloon_bbox: Bbox, mask: "np.ndarray"  # type: ignore[name-defined]
) -> tuple[bool, str]:
    """
    Use horizontal/vertical profile minima to find a neck in the mask.
    Returns (found, orientation).
    """
    x1, y1, x2, y2 = balloon_bbox
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False, ""

    h, w = roi.shape[:2]

    # Horizontal profile: for each column, count white pixels
    col_sum = roi.sum(axis=0).astype(float) / max(1, h * 255)
    # Vertical profile
    row_sum = roi.sum(axis=1).astype(float) / max(1, w * 255)

    def has_neck(profile: "np.ndarray", threshold: float = 0.25) -> bool:  # type: ignore[name-defined]
        if len(profile) < 5:
            return False
        mn = float(profile.min())
        mx = float(profile.max())
        if mx < 0.1:
            return False
        # A neck is a local minimum that is below threshold * max
        # and is not at the edges
        mid = profile[2:-2]
        return bool(float(mid.min()) < threshold * mx)

    h_neck = has_neck(col_sum)
    v_neck = has_neck(row_sum)

    if h_neck and not v_neck:
        return True, "horizontal"
    if v_neck and not h_neck:
        return True, "vertical"
    if h_neck and v_neck:
        return True, "diagonal"
    return False, ""


def _build_lobes(
    balloon_bbox: Bbox,
    ocr_bboxes: list[Bbox],
    orientation: str,
) -> list[dict]:
    """
    Split the balloon into two lobes along the orientation axis
    and assign each OCR box to the nearest lobe centre.
    """
    bx1, by1, bx2, by2 = balloon_bbox
    bw = bx2 - bx1
    bh = by2 - by1

    # Split line at midpoint along orientation axis
    if orientation == "horizontal":
        mid = bx1 + bw // 2
        lobe_a: Bbox = [bx1, by1, mid, by2]
        lobe_b: Bbox = [mid, by1, bx2, by2]
    else:  # vertical or diagonal → split vertically
        mid = by1 + bh // 2
        lobe_a = [bx1, by1, bx2, mid]
        lobe_b = [bx1, mid, bx2, by2]

    centre_a = _bbox_center(lobe_a)
    centre_b = _bbox_center(lobe_b)

    assigned_a: list[int] = []
    assigned_b: list[int] = []

    for idx, ocr_box in enumerate(ocr_bboxes):
        c = _bbox_center(ocr_box)
        if _dist(c, centre_a) <= _dist(c, centre_b):
            assigned_a.append(idx)
        else:
            assigned_b.append(idx)

    def _safe_bbox(bbox: Bbox, pad: int = 10) -> Bbox:
        return [bbox[0] + pad, bbox[1] + pad, bbox[2] - pad, bbox[3] - pad]

    lobes = [
        {
            "bbox": lobe_a,
            "safe_bbox": _safe_bbox(lobe_a),
            "assigned_text_ids": assigned_a,
        },
        {
            "bbox": lobe_b,
            "safe_bbox": _safe_bbox(lobe_b),
            "assigned_text_ids": assigned_b,
        },
    ]

    # Drop empty lobes (all text went to one side)
    lobes = [l for l in lobes if l["assigned_text_ids"]]
    return lobes
