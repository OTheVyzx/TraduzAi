"""Independent, pre-inpaint visual container recovery for owner layout."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


def _bbox(value: Any, *, width: int, height: int) -> BBox:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("container recovery requires a canonical bbox")
    result = tuple(int(item) for item in value)
    if (
        result[0] < 0
        or result[1] < 0
        or result[2] <= result[0]
        or result[3] <= result[1]
        or result[2] > width
        or result[3] > height
    ):
        raise ValueError("container recovery bbox is outside the logical page")
    return result


def _contains(outer: BBox, inner: BBox) -> bool:
    return bool(
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _proportional_fallback(seed: BBox, *, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = seed
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = max(12, int(box_width * 0.22))
    pad_y = max(12, int(box_height * 0.30))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def recover_full_page_visual_container(
    image_rgb: np.ndarray,
    *,
    owner_id: str,
    semantic_body_bbox_page: BBox,
    source_replacement_bbox_page: BBox,
) -> dict[str, Any] | None:
    """Recover an enclosing visual region without promoting cleanup geometry.

    Capacity comes from the semantic component envelope.  A pre-inpaint edge
    crossing that envelope authenticates it as visual layout evidence; the
    cleanup footprint is validated but never used to size the container.
    """

    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("container recovery requires an RGB logical-page image")
    height, width = image_rgb.shape[:2]
    semantic = _bbox(semantic_body_bbox_page, width=width, height=height)
    cleanup = _bbox(source_replacement_bbox_page, width=width, height=height)
    # Capacity is derived from semantic component geometry, never from the
    # cleanup footprint.  Pixels only authenticate that the envelope reaches a
    # real visual boundary instead of being a blind geometric expansion.
    recovered = _proportional_fallback(semantic, width=width, height=height)
    if recovered == semantic or recovered == (0, 0, width, height):
        return None
    x1, y1, x2, y2 = recovered
    image_bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    border_width = max(3, int(min(edges.shape[:2]) * 0.06))
    border = np.zeros_like(edges, dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    edge_density = float(np.count_nonzero(edges[border])) / float(
        max(1, np.count_nonzero(border))
    )
    if edge_density < 0.008:
        return None
    payload = {
        "owner_id": str(owner_id),
        "source": "full_page_visual_container",
        "bbox_page": list(recovered),
        "edge_density": round(edge_density, 6),
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "evidence_id": f"{owner_id}:full_page_visual_container:{digest}",
        "source": "full_page_visual_container",
        "bbox_page": recovered,
        "confidence": round(min(0.95, 0.70 + edge_density * 4.0), 6),
    }
