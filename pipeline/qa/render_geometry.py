from __future__ import annotations

from typing import Any

import numpy as np


def _coerce_bbox(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_containment_ratio(inner_bbox: Any, outer_bbox: Any) -> float | None:
    inner = _coerce_bbox(inner_bbox)
    outer = _coerce_bbox(outer_bbox)
    if not inner or not outer:
        return None

    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    overlap_w = max(0, min(ix2, ox2) - max(ix1, ox1))
    overlap_h = max(0, min(iy2, oy2) - max(iy1, oy1))
    return round((overlap_w * overlap_h) / float(inner_area), 4)


def check_render_inside_balloon(
    *,
    render_bbox: Any,
    balloon_bbox: Any,
    threshold: float = 0.85,
) -> dict[str, Any]:
    containment = bbox_containment_ratio(render_bbox, balloon_bbox)
    flags: list[str] = []
    if containment is not None and containment < float(threshold):
        flags.append("render_outside_balloon")
    return {"containment": containment, "flags": flags}


def check_render_background(
    image: Any,
    *,
    render_bbox: Any,
    balloon_bbox: Any,
    balloon_type: str,
    luma_threshold: float = 215.0,
) -> dict[str, Any]:
    render = _coerce_bbox(render_bbox)
    balloon = _coerce_bbox(balloon_bbox)
    if not render or not balloon:
        return {"background_luma": None, "flags": []}

    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    else:
        rgb = arr[:, :, :3]
    height, width = rgb.shape[:2]
    x1 = max(0, min(width, max(render[0], balloon[0])))
    y1 = max(0, min(height, max(render[1], balloon[1])))
    x2 = max(0, min(width, min(render[2], balloon[2])))
    y2 = max(0, min(height, min(render[3], balloon[3])))
    if x2 <= x1 or y2 <= y1:
        return {"background_luma": None, "flags": []}

    crop = rgb[y1:y2, x1:x2].astype(np.float32)
    luma_plane = (0.2126 * crop[:, :, 0]) + (0.7152 * crop[:, :, 1]) + (0.0722 * crop[:, :, 2])
    luma = round(float(np.median(luma_plane)), 2)
    flags = ["render_on_art_suspected"] if luma < float(luma_threshold) else []
    return {"background_luma": luma, "flags": flags}
