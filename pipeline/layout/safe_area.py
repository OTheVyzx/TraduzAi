"""
Computes the safe text rendering area inside a balloon, accounting for
padding, balloon shape, connected lobes, and page edges.
"""
from __future__ import annotations

from typing import Optional


# Padding tiers by balloon area (px²)
_PADDING_SMALL = {"left": 8, "right": 8, "top": 8, "bottom": 8}    # < 8 000 px²
_PADDING_MEDIUM = {"left": 14, "right": 14, "top": 10, "bottom": 10}  # 8 000–40 000 px²
_PADDING_LARGE = {"left": 20, "right": 20, "top": 14, "bottom": 14}  # > 40 000 px²
_MIN_EDGE_MARGIN = 8  # never let safe_bbox start within 8 px of the page edge


def _choose_padding(bbox: list[int]) -> dict:
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    if area < 8_000:
        return dict(_PADDING_SMALL)
    if area < 40_000:
        return dict(_PADDING_MEDIUM)
    return dict(_PADDING_LARGE)


def _clamp_to_page(bbox: list[int], page_w: int, page_h: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(_MIN_EDGE_MARGIN, x1)
    y1 = max(_MIN_EDGE_MARGIN, y1)
    x2 = min(page_w - _MIN_EDGE_MARGIN, x2)
    y2 = min(page_h - _MIN_EDGE_MARGIN, y2)
    return [x1, y1, x2, y2]


def build_safe_area(
    *,
    balloon_bbox: list[int],
    page_width: int,
    page_height: int,
    balloon_polygon: Optional[list[list[int]]] = None,
    connected_lobe_bboxes: Optional[list[list[int]]] = None,
    balloon_type: str = "white",
) -> dict:
    """
    Compute the safe text area for a single balloon or lobe.

    Args:
        balloon_bbox: outer bounding box [x1,y1,x2,y2].
        page_width / page_height: full page dimensions (pixels).
        balloon_polygon: optional polygon points [[x,y], ...].
        connected_lobe_bboxes: if the balloon has detected lobes, pass each
            lobe bbox here; a safe area is returned per lobe.
        balloon_type: "white" | "textured" | "unknown".

    Returns dict:
        {
            "safe_bbox": [x1,y1,x2,y2],
            "padding": {"left":int, "right":int, "top":int, "bottom":int},
            "reason": str,
            "lobes": [                         # only when connected_lobe_bboxes supplied
                {"lobe_index":int, "safe_bbox":[...], "padding":{...}},
                ...
            ]
        }
    """
    # Use polygon tight bbox when available (shrinks wide bboxes)
    if balloon_polygon and len(balloon_polygon) >= 4:
        xs = [p[0] for p in balloon_polygon]
        ys = [p[1] for p in balloon_polygon]
        poly_bbox = [min(xs), min(ys), max(xs), max(ys)]
        # Only tighten — never expand beyond balloon_bbox
        bx1 = max(balloon_bbox[0], poly_bbox[0])
        by1 = max(balloon_bbox[1], poly_bbox[1])
        bx2 = min(balloon_bbox[2], poly_bbox[2])
        by2 = min(balloon_bbox[3], poly_bbox[3])
        effective_bbox = [bx1, by1, bx2, by2]
        reason = "balloon_polygon"
    else:
        effective_bbox = list(balloon_bbox)
        reason = "fallback_bbox"

    padding = _choose_padding(effective_bbox)

    def apply_padding(bbox: list[int], pad: dict) -> list[int]:
        x1, y1, x2, y2 = bbox
        return [
            x1 + pad["left"],
            y1 + pad["top"],
            x2 - pad["right"],
            y2 - pad["bottom"],
        ]

    # --- Connected lobes: return per-lobe safe areas ---
    if connected_lobe_bboxes:
        lobes = []
        for i, lobe_bbox in enumerate(connected_lobe_bboxes):
            lobe_pad = _choose_padding(lobe_bbox)
            lobe_safe = apply_padding(lobe_bbox, lobe_pad)
            lobe_safe = _clamp_to_page(lobe_safe, page_width, page_height)
            lobes.append({
                "lobe_index": i,
                "safe_bbox": lobe_safe,
                "padding": lobe_pad,
            })

        # Overall safe area = union of lobe safe areas
        all_x1 = min(l["safe_bbox"][0] for l in lobes)
        all_y1 = min(l["safe_bbox"][1] for l in lobes)
        all_x2 = max(l["safe_bbox"][2] for l in lobes)
        all_y2 = max(l["safe_bbox"][3] for l in lobes)

        return {
            "safe_bbox": [all_x1, all_y1, all_x2, all_y2],
            "padding": padding,
            "reason": "connected_lobe",
            "lobes": lobes,
        }

    # --- Single balloon ---
    safe = apply_padding(effective_bbox, padding)
    safe = _clamp_to_page(safe, page_width, page_height)

    return {
        "safe_bbox": safe,
        "padding": padding,
        "reason": reason,
        "lobes": [],
    }
