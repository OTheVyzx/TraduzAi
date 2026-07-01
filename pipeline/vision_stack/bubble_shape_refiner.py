"""Geometry refinement for derived speech-bubble masks.

The refiner is intentionally conservative: it is meant to remove obvious
connected white noise from a bubble body, not to replace stylized balloons with
perfect mathematical shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ShapeRefineResult:
    mask: np.ndarray
    shape_kind: str
    removed_pixels: int
    added_pixels: int
    accepted: bool


def _binary(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(_binary(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _ellipse_mask_for(component: np.ndarray) -> np.ndarray | None:
    bbox = _bbox_from_mask(component)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    local = component[y1:y2, x1:x2]
    h, w = local.shape[:2]
    if h < 8 or w < 8:
        return None
    k = max(5, int(round(min(h, w) * 0.14)) | 1)
    opened = cv2.morphologyEx(
        local,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=1,
    )
    if int(np.count_nonzero(opened)) < int(np.count_nonzero(local) * 0.45):
        opened = local
    contour = _largest_contour(opened)
    if contour is None or len(contour) < 5:
        return None
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (axis_a, axis_b), angle = ellipse
    model = np.zeros_like(component)
    center = (int(round(cx + x1)), int(round(cy + y1)))
    axes = (
        max(1, int(round(axis_a * 0.515))),
        max(1, int(round(axis_b * 0.515))),
    )
    cv2.ellipse(model, center, axes, float(angle), 0, 360, 255, -1)
    return model


def _robust_rect_mask(component: np.ndarray, bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    local = component[y1:y2, x1:x2] > 0
    if local.size == 0:
        return np.zeros_like(component)
    row_cov = local.mean(axis=1)
    col_cov = local.mean(axis=0)
    row_threshold = 0.55
    col_threshold = 0.75
    rows = np.where(row_cov >= row_threshold)[0]
    cols = np.where(col_cov >= col_threshold)[0]
    if len(rows) < 4 or len(cols) < 4:
        rows = np.where(row_cov > 0)[0]
        cols = np.where(col_cov > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros_like(component)
    rx1 = x1 + int(cols.min())
    rx2 = x1 + int(cols.max()) + 1
    ry1 = y1 + int(rows.min())
    ry2 = y1 + int(rows.max()) + 1
    model = np.zeros_like(component)
    cv2.rectangle(model, (rx1, ry1), (rx2 - 1, ry2 - 1), 255, -1)
    return model


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    union = int(np.count_nonzero(aa | bb))
    if union <= 0:
        return 0.0
    return float(np.count_nonzero(aa & bb)) / float(union)


def _off_model_extension_stats(component: np.ndarray, model: np.ndarray) -> tuple[int, int, int]:
    outside = cv2.subtract(_binary(component), _binary(model))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats((outside > 0).astype(np.uint8), 8)
    if count <= 1:
        return 0, 0, 0
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)]
    max_width = max(int(stats[i, cv2.CC_STAT_WIDTH]) for i in range(1, count))
    return int(sum(areas)), int(max(areas)), int(max_width)


def _touches_crop_edge(component: np.ndarray, bbox: list[int]) -> bool:
    height, width = component.shape[:2]
    x1, y1, x2, y2 = bbox
    touches = y1 <= 0 or x1 <= 0 or y2 >= height or x2 >= width
    if not touches:
        return False
    local = component[y1:y2, x1:x2] > 0
    if local.size == 0:
        return False
    edge_hits = 0
    if y1 <= 0 and float(np.count_nonzero(local[0, :])) / float(max(1, local.shape[1])) >= 0.08:
        edge_hits += 1
    if y2 >= height and float(np.count_nonzero(local[-1, :])) / float(max(1, local.shape[1])) >= 0.08:
        edge_hits += 1
    if x1 <= 0 and float(np.count_nonzero(local[:, 0])) / float(max(1, local.shape[0])) >= 0.08:
        edge_hits += 1
    if x2 >= width and float(np.count_nonzero(local[:, -1])) / float(max(1, local.shape[0])) >= 0.08:
        edge_hits += 1
    return edge_hits > 0


def _rectangle_likeness(component: np.ndarray, bbox: list[int]) -> float:
    x1, y1, x2, y2 = bbox
    local = component[y1:y2, x1:x2] > 0
    if local.size == 0:
        return 0.0
    area = int(np.count_nonzero(local))
    extent = area / float(max(1, local.size))
    row_cov = local.mean(axis=1)
    col_cov = local.mean(axis=0)
    dense_rows = float(np.count_nonzero(row_cov >= 0.75)) / float(max(1, len(row_cov)))
    dense_cols = float(np.count_nonzero(col_cov >= 0.75)) / float(max(1, len(col_cov)))
    return (extent * 0.6) + (dense_rows * 0.2) + (dense_cols * 0.2)


def _classify(component: np.ndarray) -> tuple[str, np.ndarray]:
    bbox = _bbox_from_mask(component)
    if bbox is None:
        return "empty", np.zeros_like(component)

    rect_model = _robust_rect_mask(component, bbox)
    rect_score = _rectangle_likeness(component, bbox)
    rect_iou = _iou(component, rect_model)
    if (rect_score >= 0.86 and rect_iou >= 0.82) or (rect_score >= 0.78 and rect_iou >= 0.84):
        return "rectangle", rect_model

    ellipse_model = _ellipse_mask_for(component)
    if ellipse_model is None:
        return "irregular", component.copy()

    ellipse_iou = _iou(component, ellipse_model)
    if _touches_crop_edge(component, bbox):
        if ellipse_iou >= 0.84:
            return "oval", ellipse_model
        return "irregular", component.copy()

    outside_total, outside_largest, outside_max_width = _off_model_extension_stats(component, ellipse_model)
    area = max(1, int(np.count_nonzero(component)))
    outside_ratio = outside_total / float(area)
    largest_ratio = outside_largest / float(area)

    # Thin off-ellipse parts are usually tails/points. Thick blocks are noise.
    if outside_ratio >= 0.13 and (largest_ratio < 0.11 or outside_max_width < max(12, int((bbox[2] - bbox[0]) * 0.18))):
        return "irregular", component.copy()
    if ellipse_iou >= 0.78:
        return "oval", ellipse_model
    return "irregular", component.copy()


def _safe_accept(original: np.ndarray, refined: np.ndarray, shape_kind: str) -> bool:
    original_count = max(1, int(np.count_nonzero(original)))
    removed = int(np.count_nonzero((original > 0) & (refined == 0)))
    added = int(np.count_nonzero((refined > 0) & (original == 0)))
    if shape_kind == "irregular":
        return removed <= max(96, int(original_count * 0.03))
    if shape_kind == "rectangle":
        return removed <= max(32, int(original_count * 0.20)) and added <= max(32, int(original_count * 0.08))
    return removed <= max(64, int(original_count * 0.20)) and added <= max(64, int(original_count * 0.15))


def refine_bubble_shape_mask(mask: np.ndarray, prefer_shape: str | None = None) -> ShapeRefineResult:
    original = _binary(mask)
    if original.size == 0 or not np.any(original):
        return ShapeRefineResult(original, "empty", 0, 0, False)

    if prefer_shape is None:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats((original > 0).astype(np.uint8), 8)
        component_ids = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= max(16, int(np.count_nonzero(original) * 0.01))
        ]
        if len(component_ids) > 1:
            refined_mask = np.zeros_like(original)
            kinds: set[str] = set()
            for label in component_ids:
                component = np.where(labels == label, 255, 0).astype(np.uint8)
                component_kind, component_refined = _classify(component)
                component_refined = _binary(component_refined)
                if not _safe_accept(component, component_refined, component_kind):
                    component_refined = component
                    component_kind = "irregular"
                refined_mask = cv2.bitwise_or(refined_mask, component_refined)
                kinds.add(component_kind)
            removed = int(np.count_nonzero((original > 0) & (refined_mask == 0)))
            added = int(np.count_nonzero((refined_mask > 0) & (original == 0)))
            shape_kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
            return ShapeRefineResult(refined_mask, shape_kind, removed, added, True)

    shape_kind, refined = _classify(original)
    if prefer_shape in {"oval", "rectangle", "irregular"} and prefer_shape != shape_kind:
        if prefer_shape == "rectangle":
            bbox = _bbox_from_mask(original)
            refined = _robust_rect_mask(original, bbox) if bbox else original.copy()
            shape_kind = "rectangle"
        elif prefer_shape == "oval":
            refined = _ellipse_mask_for(original) or original.copy()
            shape_kind = "oval"
        else:
            refined = original.copy()
            shape_kind = "irregular"

    refined = _binary(refined)
    accepted = _safe_accept(original, refined, shape_kind)
    if not accepted:
        refined = original.copy()
        shape_kind = "irregular"
    removed = int(np.count_nonzero((original > 0) & (refined == 0)))
    added = int(np.count_nonzero((refined > 0) & (original == 0)))
    return ShapeRefineResult(refined, shape_kind, removed, added, True)
