from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TextEvidence:
    bbox: list[int]
    text: str = ""
    confidence: float = 0.0
    source: str = "unknown"
    line_polygons: list[list[list[float]]] = field(default_factory=list)
    word_boxes: list[list[int]] = field(default_factory=list)
    char_boxes: list[list[int]] = field(default_factory=list)
    preserve_original: bool = False


def normalize_bbox(value: Any, width: int, height: int) -> list[int] | None:
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


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_boxes(value: Any, width: int, height: int) -> list[list[int]]:
    if not isinstance(value, (list, tuple)):
        return []
    boxes: list[list[int]] = []
    for item in value:
        bbox = normalize_bbox(item, width, height)
        if bbox is not None:
            boxes.append(bbox)
    return boxes


def _normalize_line_polygons(value: Any) -> list[list[list[float]]]:
    if not isinstance(value, (list, tuple)):
        return []
    polygons: list[list[list[float]]] = []
    for polygon in value:
        if not isinstance(polygon, (list, tuple)):
            continue
        points: list[list[float]] = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                points.append([float(point[0]), float(point[1])])
            except Exception:
                continue
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def _item_to_evidence(item: dict[str, Any], *, width: int, height: int, source: str) -> TextEvidence | None:
    bbox = None
    for key in ("text_pixel_bbox", "bbox", "balloon_bbox"):
        bbox = normalize_bbox(item.get(key), width, height)
        if bbox is not None:
            break
    if bbox is None:
        return None

    return TextEvidence(
        bbox=bbox,
        text=str(item.get("text") or item.get("recognized_text") or ""),
        confidence=_coerce_float(item.get("confidence"), 0.0),
        source=source,
        line_polygons=_normalize_line_polygons(item.get("line_polygons")),
        word_boxes=_normalize_boxes(item.get("word_boxes") or item.get("wordBoxes"), width, height),
        char_boxes=_normalize_boxes(item.get("char_boxes") or item.get("charBoxes"), width, height),
        preserve_original=False,
    )


def normalize_text_evidence(page_result: dict[str, Any], width: int, height: int) -> list[TextEvidence]:
    evidence: list[TextEvidence] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for source, key in (("ocr", "texts"), ("detector", "_vision_blocks"), ("oar-ocr", "_oar_ocr_regions")):
        for item in page_result.get(key) or []:
            if not isinstance(item, dict):
                continue
            candidate = _item_to_evidence(item, width=width, height=height, source=source)
            if candidate is None:
                continue
            dedupe_key = (*candidate.bbox, candidate.text)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            evidence.append(candidate)
    return evidence


def evidence_support_mask(
    shape: tuple[int, int],
    evidence: list[TextEvidence],
    *,
    pad: int = 6,
) -> np.ndarray:
    height, width = shape[:2]
    support = np.zeros((height, width), dtype=np.uint8)
    for item in evidence:
        boxes = item.char_boxes or item.word_boxes or [item.bbox]
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)
            if x2 > x1 and y2 > y1:
                support[y1:y2, x1:x2] = 255
        for polygon in item.line_polygons:
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
            if points.shape[0] >= 3:
                cv2.fillPoly(support, [points], 255)
    return support


def measure_mask_coverage(mask: np.ndarray, image_rgb: np.ndarray, evidence: list[TextEvidence]) -> dict[str, float | int]:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.astype(np.uint8)
    support = evidence_support_mask(mask.shape[:2], evidence, pad=2)
    if not np.any(support):
        return {
            "mask_pixels": int(np.count_nonzero(mask)),
            "dark_pixels": 0,
            "dark_inside_mask": 0,
            "dark_outside_mask": 0,
            "coverage_ratio": 1.0,
        }
    dark = (gray < 128) & (support > 0)
    dark_pixels = int(np.count_nonzero(dark))
    dark_inside = int(np.count_nonzero(dark & (mask > 0)))
    dark_outside = int(np.count_nonzero(dark & (mask == 0)))
    return {
        "mask_pixels": int(np.count_nonzero((mask > 0) & (support > 0))),
        "dark_pixels": dark_pixels,
        "dark_inside_mask": dark_inside,
        "dark_outside_mask": dark_outside,
        "coverage_ratio": float(dark_inside / dark_pixels) if dark_pixels else 1.0,
    }
