"""Recover bounded white dialogue containers from source pixels and OCR geometry."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _bbox(value: Any) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        result = ()
    if len(result) != 4 or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("observation bbox must be non-empty")
    return result


def _stable_id(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "container_" + hashlib.sha256(encoded).hexdigest()[:20]


def _split_lobes(component: np.ndarray, offset: tuple[int, int]) -> list[list[int]]:
    area = int(np.count_nonzero(component))
    height, width = component.shape
    minimum = max(200, int(area * 0.08))
    kernel_size = 13 if min(height, width) >= 160 else 9
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for iterations in range(1, 5):
        eroded = cv2.erode(component, kernel, iterations=iterations)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, 8)
        valid = sorted(
            (index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= minimum),
            key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        if len(valid) < 2:
            continue
        first, second = valid[:2]
        if int(stats[second, cv2.CC_STAT_AREA]) < int(stats[first, cv2.CC_STAT_AREA]) * 0.20:
            continue
        ys, xs = np.where(component > 0)
        distances = [
            (xs - float(centroids[index][0])) ** 2
            + (ys - float(centroids[index][1])) ** 2
            for index in (first, second)
        ]
        assignment = distances[0] <= distances[1]
        boxes: list[list[int]] = []
        for chosen in (assignment, ~assignment):
            chosen_x = xs[chosen]
            chosen_y = ys[chosen]
            if len(chosen_x) < minimum:
                boxes = []
                break
            boxes.append([
                int(chosen_x.min()) + offset[0],
                int(chosen_y.min()) + offset[1],
                int(chosen_x.max()) + 1 + offset[0],
                int(chosen_y.max()) + 1 + offset[1],
            ])
        if len(boxes) == 2:
            return sorted(boxes, key=lambda box: (box[1], box[0]))
    return []


def corroborate_lobes(
    lobe_bboxes: Sequence[Sequence[int]],
    observations: Sequence[Mapping[str, Any]],
    container_bbox: Sequence[int],
) -> list[list[int]]:
    """Require distinct text clusters before publishing a geometric lobe split."""

    boxes = [list(_bbox(value)) for value in lobe_bboxes]
    if len(boxes) != 2:
        return []
    assigned: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
    for member in observations:
        member_bbox = _bbox(member.get("bbox_page"))
        center = (
            (member_bbox[0] + member_bbox[2]) / 2.0,
            (member_bbox[1] + member_bbox[3]) / 2.0,
        )
        candidates = [
            index for index, box in enumerate(boxes)
            if box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]
        ]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda index: (
            (((boxes[index][0] + boxes[index][2]) / 2.0) - center[0]) ** 2
            + (((boxes[index][1] + boxes[index][3]) / 2.0) - center[1]) ** 2,
            index,
        ))
        assigned[chosen].append(center)
    if not assigned[0] or not assigned[1]:
        return []
    lobe_centers = [
        ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) for box in boxes
    ]
    vertical = abs(lobe_centers[0][1] - lobe_centers[1][1]) >= abs(
        lobe_centers[0][0] - lobe_centers[1][0]
    )
    if not vertical:
        return boxes
    cluster_centers = [
        (
            sum(point[0] for point in assigned[index]) / len(assigned[index]),
            sum(point[1] for point in assigned[index]) / len(assigned[index]),
        )
        for index in (0, 1)
    ]
    container = _bbox(container_bbox)
    cross_shift = abs(cluster_centers[0][0] - cluster_centers[1][0])
    cross_extent = container[2] - container[0]
    if cross_shift < max(12.0, cross_extent * 0.12):
        return []
    return boxes


def infer_text_cluster_lobes(
    container_bbox: Sequence[int],
    observations: Sequence[Mapping[str, Any]],
) -> list[list[int]]:
    """Infer two physical text clusters inside one authenticated container."""

    container = _bbox(container_bbox)
    boxes = sorted(
        (_bbox(row.get("bbox_page")) for row in observations),
        key=lambda box: (box[1], box[0], box[3], box[2]),
    )
    if len(boxes) < 4:
        return []
    heights = sorted(box[3] - box[1] for box in boxes)
    median_height = float(heights[len(heights) // 2])
    candidates = []
    for split in range(2, len(boxes) - 1):
        first = boxes[:split]
        second = boxes[split:]
        gap = second[0][1] - first[-1][3]
        if gap < max(16.0, median_height * 0.65):
            continue
        first_center_x = sum((box[0] + box[2]) / 2.0 for box in first) / len(first)
        second_center_x = sum((box[0] + box[2]) / 2.0 for box in second) / len(second)
        horizontal_shift = abs(first_center_x - second_center_x)
        if horizontal_shift < max(20.0, (container[2] - container[0]) * 0.12):
            continue
        candidates.append((gap + horizontal_shift, first, second))
    if not candidates:
        return []
    _, first, second = max(candidates, key=lambda item: item[0])
    padding = max(8, int(round(median_height)))
    results = []
    for group in (first, second):
        results.append([
            max(container[0], min(box[0] for box in group) - padding),
            max(container[1], min(box[1] for box in group) - padding),
            min(container[2], max(box[2] for box in group) + padding),
            min(container[3], max(box[3] for box in group) + padding),
        ])
    return results


def discover_white_containers(
    image_rgb: np.ndarray,
    observations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Bind OCR observations only to enclosed white source components."""

    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        raise ValueError("RGB source pixels are required")
    height, width = image_rgb.shape[:2]
    white = (np.min(image_rgb[:, :, :3], axis=2) >= 245).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    valid_labels: set[int] = set()
    page_area = max(1, height * width)
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if not touches_border and 200 <= area <= int(page_area * 0.45):
            valid_labels.add(label)

    groups: dict[int, list[dict[str, Any]]] = {}
    for row in observations:
        bbox = _bbox(row.get("bbox_page"))
        x1, y1, x2, y2 = (
            max(0, bbox[0]), max(0, bbox[1]), min(width, bbox[2]), min(height, bbox[3])
        )
        if x2 <= x1 or y2 <= y1:
            continue
        values, frequencies = np.unique(labels[y1:y2, x1:x2], return_counts=True)
        ranked = sorted(
            (
                (int(frequency), int(label))
                for label, frequency in zip(values, frequencies)
                if int(label) in valid_labels
            ),
            reverse=True,
        )
        if not ranked:
            continue
        groups.setdefault(ranked[0][1], []).append({
            "observation_id": str(row.get("observation_id") or ""),
            "bbox_page": list(bbox),
            "selection_state": str(row.get("selection_state") or "eligible"),
        })

    containers: list[dict[str, Any]] = []
    for label, members in groups.items():
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        local = (labels[y:y + h, x:x + w] == label).astype(np.uint8)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.003), True)
        polygon_page = [
            [int(point[0][0]) + x, int(point[0][1]) + y] for point in polygon
        ]
        bbox_page = [x, y, x + w, y + h]
        supporting_members = [
            row for row in members if row["selection_state"] == "eligible"
        ]
        geometric_lobes = _split_lobes(local, (x, y))
        lobe_bboxes = corroborate_lobes(
            geometric_lobes, supporting_members, bbox_page
        )
        lobe_evidence = "eroded_source_component" if lobe_bboxes else None
        if not lobe_bboxes:
            lobe_bboxes = infer_text_cluster_lobes(bbox_page, supporting_members)
            if lobe_bboxes:
                lobe_evidence = "spatial_text_clusters_within_authenticated_component"
        observation_ids = [
            row["observation_id"]
            for row in sorted(members, key=lambda row: (
                row["bbox_page"][1], row["bbox_page"][0]
            ))
        ]
        containers.append({
            "container_id": _stable_id({
                "bbox_page": bbox_page,
                "observation_ids": observation_ids,
                "polygon_page": polygon_page,
            }),
            "bbox_page": bbox_page,
            "polygon_page": polygon_page,
            "kind": "connected_balloon" if len(lobe_bboxes) >= 2 else "balloon",
            "lobe_bboxes": lobe_bboxes,
            "lobe_evidence": lobe_evidence,
            "observation_ids": observation_ids,
            "evidence": "enclosed_white_source_component",
            "uncertainty_reasons": (
                ["geometric_lobes_not_corroborated_by_text_clusters"]
                if geometric_lobes and not lobe_bboxes else []
            ),
        })
    return tuple(sorted(containers, key=lambda row: (
        row["bbox_page"][1], row["bbox_page"][0]
    )))
