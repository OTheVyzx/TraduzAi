from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class SmartMaskResult:
    mask: np.ndarray
    entries: list[dict[str, Any]]
    stats: dict[str, Any]


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _coerce_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _mask_confidence(entry: dict[str, Any], mask: np.ndarray, bbox: list[int]) -> float:
    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    mask_area = int(np.count_nonzero(mask))
    area_ratio = max(0.0, min(1.0, mask_area / float(bbox_area)))
    base = 0.82
    if entry.get("line_polygons"):
        base = 0.91
    elif entry.get("text_pixel_bbox"):
        base = 0.87
    if area_ratio < 0.02:
        base -= 0.12
    if area_ratio > 0.75:
        base -= 0.10
    return round(max(0.0, min(0.99, base)), 3)


class SmartTextMaskEngine:
    name = "smart_text_mask"

    def build_mask(self, image_rgb: Any, text_entries: list[dict[str, Any]], *, quality: str = "normal") -> SmartMaskResult:
        if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
            empty = np.zeros((0, 0), dtype=np.uint8)
            return SmartMaskResult(mask=empty, entries=list(text_entries or []), stats={"built": 0, "failed": 0})

        height, width = image_rgb.shape[:2]
        union_mask = np.zeros((height, width), dtype=np.uint8)
        entries: list[dict[str, Any]] = []
        built = 0
        failed = 0
        total_pixels = 0
        dilate_px = 2 if str(quality).strip().lower() == "ultra" else 1

        try:
            from inpainter.mask_builder import build_inpaint_mask
        except ImportError:
            from ..inpainter.mask_builder import build_inpaint_mask

        for entry in text_entries or []:
            if not isinstance(entry, dict):
                continue
            next_entry = dict(entry)
            bbox = _coerce_bbox(
                entry.get("text_pixel_bbox") or entry.get("source_bbox") or entry.get("bbox"),
                width,
                height,
            )
            if bbox is None:
                failed += 1
                entries.append(next_entry)
                continue

            try:
                raw_mask = build_inpaint_mask(entry, image_rgb.shape, image_rgb=image_rgb)
            except Exception:
                raw_mask = None
            if raw_mask is None or not np.any(raw_mask):
                failed += 1
                entries.append(next_entry)
                continue

            smart_mask = raw_mask.astype(np.uint8)
            if dilate_px > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
                smart_mask = cv2.dilate(smart_mask, kernel, iterations=1)
            mask_bbox = _bbox_from_mask(smart_mask)
            if mask_bbox is None:
                failed += 1
                entries.append(next_entry)
                continue

            mask_pixels = int(np.count_nonzero(smart_mask))
            bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            if mask_pixels < 4 or mask_pixels > int(bbox_area * 1.65):
                failed += 1
                entries.append(next_entry)
                continue

            x1, y1, x2, y2 = mask_bbox
            local_mask = smart_mask[y1:y2, x1:x2].copy()
            next_entry.update(
                {
                    "mask": local_mask,
                    "mask_bbox": mask_bbox,
                    "mask_source": self.name,
                    "mask_confidence": _mask_confidence(entry, smart_mask, bbox),
                    "mask_pixels": mask_pixels,
                }
            )
            entries.append(next_entry)
            union_mask = np.maximum(union_mask, smart_mask)
            built += 1
            total_pixels += mask_pixels

        return SmartMaskResult(
            mask=union_mask,
            entries=entries,
            stats={
                "engine": self.name,
                "quality": "ultra" if str(quality).strip().lower() == "ultra" else "normal",
                "built": built,
                "failed": failed,
                "pixels": total_pixels,
            },
        )
