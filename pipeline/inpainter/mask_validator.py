"""Mask validation for reopenable project layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def validate_mask(mask_path: str | Path, bbox: list[int] | tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    path = Path(mask_path)
    if not path.exists():
        return _invalid("mask_missing", "regenerate_mask")
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return _invalid("mask_unreadable", "regenerate_mask")
    if img.width <= 1 or img.height <= 1:
        return _invalid("mask_too_small", "regenerate_mask")
    alpha = np.array(img.getchannel("A"))
    if int(alpha.max()) == 0:
        return _invalid("mask_transparent", "regenerate_mask")
    opaque = alpha > 0
    if not np.any(opaque):
        return _invalid("mask_empty", "regenerate_mask")
    ys, xs = np.where(opaque)
    mask_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if bbox is not None and not _bbox_compatible(mask_bbox, [int(v) for v in bbox]):
        return _invalid("mask_bbox_mismatch", "regenerate_mask", mask_bbox=mask_bbox)
    return {"valid": True, "reason": "", "mask_bbox": mask_bbox, "flag": None}


def validate_export_contains_masks(project: dict[str, Any], root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    flags = []
    for page in project.get("paginas", []) or []:
        for layer in page.get("text_layers", []) or []:
            mask_path = layer.get("mask_path")
            if not mask_path:
                flags.append({"type": "mask_missing", "severity": "high", "region_id": layer.get("id"), "action": "regenerate_mask"})
                continue
            result = validate_mask(root / mask_path, layer.get("bbox"))
            if not result["valid"]:
                flags.append({"type": result["reason"], "severity": "high", "region_id": layer.get("id"), "action": "regenerate_mask"})
    return flags


def _invalid(reason: str, action: str, **extra) -> dict[str, Any]:
    return {"valid": False, "reason": reason, "flag": {"type": reason, "severity": "high", "action": action}, **extra}


def _bbox_compatible(mask_bbox: list[int], bbox: list[int]) -> bool:
    mx1, my1, mx2, my2 = mask_bbox
    bx1, by1, bx2, by2 = bbox
    mask_area = max(1, (mx2 - mx1) * (my2 - my1))
    bbox_area = max(1, (bx2 - bx1) * (by2 - by1))
    inside = mx1 >= bx1 - 8 and my1 >= by1 - 8 and mx2 <= bx2 + 8 and my2 <= by2 + 8
    return inside and mask_area <= bbox_area * 1.6

