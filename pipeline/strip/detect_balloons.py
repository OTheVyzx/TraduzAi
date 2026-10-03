"""Detecção de balões sobre o strip via sliding window + NMS."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from strip.types import Balloon, BBox, VerticalStrip


def _block_value(block, *names, default=None):
    for name in names:
        if isinstance(block, dict) and block.get(name) is not None:
            return block.get(name)
        value = getattr(block, name, None)
        if value is not None:
            return value
    return default


def _normalise_script_evidence(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        return sorted(str(key) for key, score in value.items() if score)
    if isinstance(value, (list, tuple, set, frozenset)):
        return sorted({str(item) for item in value if item})
    return []


def _normalise_line_polygons_strip(block, *, y_offset: int) -> list[list[list[int]]]:
    raw_polygons = _block_value(block, "line_polygons", "linePolygons", default=()) or ()
    polygons: list[list[list[int]]] = []
    for raw_polygon in raw_polygons:
        if isinstance(raw_polygon, np.ndarray):
            raw_polygon = raw_polygon.tolist()
        if not isinstance(raw_polygon, (list, tuple)):
            continue
        points: list[list[int]] = []
        for raw_point in raw_polygon:
            if isinstance(raw_point, np.ndarray):
                raw_point = raw_point.tolist()
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
                continue
            try:
                points.append(
                    [
                        int(round(float(raw_point[0]))),
                        int(round(float(raw_point[1]))) + int(y_offset),
                    ]
                )
            except (TypeError, ValueError):
                continue
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def _detector_block_metadata(block, *, y_offset: int, default_source: str) -> dict:
    raw_source = _block_value(block, "detector_source", "detector", "source")
    detector_source = (
        raw_source.strip()
        if isinstance(raw_source, str) and raw_source.strip()
        else default_source
    )
    metadata: dict = {
        "detector_source": detector_source,
        "detector_sources": [detector_source],
        "candidate_kind": str(
            _block_value(block, "candidate_kind", "kind", default="text_region")
            or "text_region"
        ),
    }
    line_polygons = _normalise_line_polygons_strip(block, y_offset=y_offset)
    if line_polygons:
        metadata["line_polygons_strip"] = line_polygons
    script_evidence = _normalise_script_evidence(
        _block_value(block, "script_evidence", "scripts", "script")
    )
    if script_evidence:
        metadata["script_evidence"] = script_evidence
    raw_rotation = _block_value(block, "rotation_deg", "rotation")
    if raw_rotation is not None:
        try:
            metadata["rotation_deg"] = float(raw_rotation)
            metadata["rotation_source"] = str(
                _block_value(block, "rotation_source", default="detector") or "detector"
            )
        except (TypeError, ValueError):
            pass
    return metadata


def _merge_provenance_metadata(primary: dict, secondary: dict) -> dict:
    merged = dict(primary or {})
    collection_keys = {
        "detector_source",
        "detector_sources",
        "line_polygons_strip",
        "script_evidence",
        "rotation_candidates",
        "candidate_kinds",
    }
    # NMS orders candidates by confidence, so scalar decisions belong to the
    # primary candidate.  Secondary evidence may fill a missing value but must
    # never rewrite the geometry/classification metadata of the winner.
    for key, value in dict(secondary or {}).items():
        if key not in collection_keys and key not in merged:
            merged[key] = value
    sources = {
        str(value)
        for metadata in (primary or {}, secondary or {})
        for value in [
            *(metadata.get("detector_sources") or []),
            metadata.get("detector_source"),
        ]
        if value
    }
    if sources:
        merged["detector_sources"] = sorted(sources)
        merged["detector_source"] = str(
            (primary or {}).get("detector_source") or sorted(sources)[0]
        )
    polygons: list = []
    seen_polygons: set[tuple] = set()
    for metadata in (primary or {}, secondary or {}):
        for polygon in metadata.get("line_polygons_strip") or []:
            key = tuple(tuple(int(value) for value in point[:2]) for point in polygon)
            if len(key) >= 3 and key not in seen_polygons:
                seen_polygons.add(key)
                polygons.append([list(point) for point in key])
    if polygons:
        merged["line_polygons_strip"] = polygons
    scripts = {
        str(value)
        for metadata in (primary or {}, secondary or {})
        for value in metadata.get("script_evidence") or []
        if value
    }
    if scripts:
        merged["script_evidence"] = sorted(scripts)
    candidate_kinds = {
        str(value)
        for metadata in (primary or {}, secondary or {})
        for value in [
            *(metadata.get("candidate_kinds") or []),
            metadata.get("candidate_kind"),
        ]
        if value
    }
    if candidate_kinds:
        merged["candidate_kinds"] = sorted(candidate_kinds)
    rotations: dict[tuple[float, str], dict] = {}
    for metadata in (primary or {}, secondary or {}):
        for candidate in metadata.get("rotation_candidates") or []:
            if not isinstance(candidate, dict) or candidate.get("rotation_deg") is None:
                continue
            try:
                angle = float(candidate["rotation_deg"])
            except (TypeError, ValueError):
                continue
            source = str(candidate.get("rotation_source") or "detector")
            rotations[(angle, source)] = {
                "rotation_deg": angle,
                "rotation_source": source,
            }
        if metadata.get("rotation_deg") is not None:
            try:
                angle = float(metadata["rotation_deg"])
            except (TypeError, ValueError):
                continue
            source = str(metadata.get("rotation_source") or "detector")
            rotations[(angle, source)] = {
                "rotation_deg": angle,
                "rotation_source": source,
            }
    if rotations:
        merged["rotation_candidates"] = [
            rotations[key] for key in sorted(rotations, key=lambda item: (item[1], item[0]))
        ]
    return merged


def _iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union entre dois bboxes."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _intersection_over_smaller_area(a: BBox, b: BBox) -> float:
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0, min(a.y2, b.y2) - max(a.y1, b.y1)
    )
    smaller = min(_bbox_area(a), _bbox_area(b))
    return intersection / float(smaller) if smaller > 0 else 0.0


def _nms_balloons(
    balloons: list[Balloon],
    iou_threshold: float = 0.5,
    *,
    intersection_smaller_threshold: float | None = None,
    require_same_class: bool = False,
) -> list[Balloon]:
    """Remove balões redundantes; mantém o de maior confidence em cada cluster."""
    if not balloons:
        return []
    sorted_balloons = sorted(
        balloons,
        key=lambda balloon: (
            -float(balloon.confidence),
            int(balloon.strip_bbox.y1),
            int(balloon.strip_bbox.x1),
            int(balloon.strip_bbox.y2),
            int(balloon.strip_bbox.x2),
        ),
    )
    kept: list[Balloon] = []
    for cand in sorted_balloons:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(kept)
                if (
                    not require_same_class
                    or str((cand.metadata or {}).get("candidate_kind") or "text_region")
                    == str((existing.metadata or {}).get("candidate_kind") or "text_region")
                )
                and (
                    (
                        _iou(cand.strip_bbox, existing.strip_bbox) >= iou_threshold
                        if require_same_class
                        else _iou(cand.strip_bbox, existing.strip_bbox) > iou_threshold
                    )
                    or (
                        intersection_smaller_threshold is not None
                        and _intersection_over_smaller_area(
                            cand.strip_bbox, existing.strip_bbox
                        )
                        >= intersection_smaller_threshold
                    )
                )
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(cand)
            continue
        existing = kept[duplicate_index]
        existing.metadata = _merge_provenance_metadata(
            dict(getattr(existing, "metadata", {}) or {}),
            dict(getattr(cand, "metadata", {}) or {}),
        )
    return kept


def _bbox_contains_center(container: BBox, inner: BBox, margin: int = 12) -> bool:
    cx = (inner.x1 + inner.x2) / 2.0
    cy = (inner.y1 + inner.y2) / 2.0
    return (
        container.x1 - margin <= cx <= container.x2 + margin
        and container.y1 - margin <= cy <= container.y2 + margin
    )


def _bbox_area(bbox: BBox) -> int:
    return max(0, bbox.x2 - bbox.x1) * max(0, bbox.y2 - bbox.y1)


def _bbox_union(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return BBox(
        min(box.x1 for box in boxes),
        min(box.y1 for box in boxes),
        max(box.x2 for box in boxes),
        max(box.y2 for box in boxes),
    )


def _expand_bbox(bbox: BBox, image_shape: tuple[int, int] | tuple[int, int, int]) -> BBox:
    height, width = image_shape[:2]
    box_w = max(1, bbox.x2 - bbox.x1)
    box_h = max(1, bbox.y2 - bbox.y1)
    pad_x = max(10, int(box_w * 0.25))
    pad_y = max(8, int(box_h * 0.45))
    return BBox(
        max(0, bbox.x1 - pad_x),
        max(0, bbox.y1 - pad_y),
        min(width, bbox.x2 + pad_x),
        min(height, bbox.y2 + pad_y),
    )


def _extract_inner_dark_text_boxes(image: np.ndarray, bbox: BBox) -> list[BBox]:
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(bbox.x1)))
    x2 = max(0, min(width, int(bbox.x2)))
    y1 = max(0, min(height, int(bbox.y1)))
    y2 = max(0, min(height, int(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        return []

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    pad_x = max(5, int(w * 0.08))
    pad_y = max(5, int(h * 0.10))
    if w <= pad_x * 2 or h <= pad_y * 2:
        return []

    inner = gray[pad_y : h - pad_y, pad_x : w - pad_x]
    dark = (inner <= 105).astype(np.uint8) * 255
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    boxes: list[BBox] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp_x = int(stats[label, cv2.CC_STAT_LEFT])
        comp_y = int(stats[label, cv2.CC_STAT_TOP])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        inner_h, inner_w = inner.shape[:2]
        if comp_x <= 1 or comp_y <= 1 or (comp_x + comp_w) >= inner_w - 1 or (comp_y + comp_h) >= inner_h - 1:
            continue
        if area < 8 or area > max(700, int(w * h * 0.18)):
            continue
        if comp_w < 2 or comp_h < 3:
            continue
        if comp_w > 90 or comp_h > 48:
            continue
        if comp_w > int(w * 0.38) or comp_h > int(h * 0.35):
            continue
        boxes.append(
            BBox(
                x1 + pad_x + comp_x,
                y1 + pad_y + comp_y,
                x1 + pad_x + comp_x + comp_w,
                y1 + pad_y + comp_y + comp_h,
            )
        )

    return boxes


def _has_inner_dark_text(image: np.ndarray, bbox: BBox) -> bool:
    boxes = _extract_inner_dark_text_boxes(image, bbox)
    if len(boxes) < 2:
        return False
    return sum(_bbox_area(box) for box in boxes) >= 18


def _inner_dark_text_evidence(image: np.ndarray, bbox: BBox) -> dict:
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(bbox.x1)))
    x2 = max(0, min(width, int(bbox.x2)))
    y1 = max(0, min(height, int(bbox.y1)))
    y2 = max(0, min(height, int(bbox.y2)))
    boxes = _extract_inner_dark_text_boxes(image, bbox)
    inner_dark_area = sum(_bbox_area(box) for box in boxes)
    significant_count, significant_area = _significant_text_component_count(boxes)
    bright_ratio = 0.0
    dark_ratio = 0.0
    if x2 > x1 and y2 > y1:
        crop = image[y1:y2, x1:x2]
        if crop.size:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            if gray.size:
                bright_ratio = float(np.mean(gray >= 238))
                dark_ratio = float(np.mean(gray <= 105))
                h, w = gray.shape[:2]
                pad_x = max(5, int(w * 0.08))
                pad_y = max(5, int(h * 0.10))
                light_boxes: list[BBox] = []
                if w > pad_x * 2 and h > pad_y * 2:
                    inner = gray[pad_y : h - pad_y, pad_x : w - pad_x]
                    light = (inner >= 150).astype(np.uint8) * 255
                    light = cv2.morphologyEx(
                        light,
                        cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                        iterations=1,
                    )
                    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=8)
                    inner_h, inner_w = inner.shape[:2]
                    for label in range(1, num_labels):
                        area = int(stats[label, cv2.CC_STAT_AREA])
                        comp_x = int(stats[label, cv2.CC_STAT_LEFT])
                        comp_y = int(stats[label, cv2.CC_STAT_TOP])
                        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
                        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
                        if comp_x <= 1 or comp_y <= 1 or (comp_x + comp_w) >= inner_w - 1 or (comp_y + comp_h) >= inner_h - 1:
                            continue
                        if area < 8 or area > max(900, int(w * h * 0.20)):
                            continue
                        if comp_w < 2 or comp_h < 3:
                            continue
                        if comp_w > 120 or comp_h > 56:
                            continue
                        if comp_w > int(w * 0.42) or comp_h > int(h * 0.38):
                            continue
                        light_boxes.append(
                            BBox(
                                x1 + pad_x + comp_x,
                                y1 + pad_y + comp_y,
                                x1 + pad_x + comp_x + comp_w,
                                y1 + pad_y + comp_y + comp_h,
                            )
                        )
                light_count, light_area = _significant_text_component_count(light_boxes)
            else:
                light_count = 0
                light_area = 0
        else:
            light_count = 0
            light_area = 0
    else:
        light_count = 0
        light_area = 0
    return {
        "has_inner_dark_text": len(boxes) >= 2 and inner_dark_area >= 18,
        "has_inner_light_text": int(light_count) >= 2 and int(light_area) >= 180 and dark_ratio >= 0.35,
        "inner_dark_component_count": int(len(boxes)),
        "inner_dark_area": int(inner_dark_area),
        "inner_light_component_count": int(light_count),
        "inner_light_area": int(light_area),
        "significant_component_count": int(significant_count),
        "significant_area": int(significant_area),
        "bright_pixel_ratio": round(bright_ratio, 4),
        "dark_pixel_ratio": round(dark_ratio, 4),
    }


def _significant_text_components(boxes: list[BBox]) -> list[BBox]:
    return [
        box
        for box in boxes
        if (box.x2 - box.x1) >= 6
        and (box.y2 - box.y1) >= 12
        and _bbox_area(box) >= 80
    ]


def _significant_text_component_count(boxes: list[BBox]) -> tuple[int, int]:
    significant = _significant_text_components(boxes)
    return len(significant), sum(_bbox_area(box) for box in significant)


def _cluster_text_components_for_band_scan(boxes: list[BBox]) -> list[list[BBox]]:
    significant = sorted(_significant_text_components(boxes), key=lambda box: (box.y1, box.x1))
    if not significant:
        return []
    heights = [max(1, box.y2 - box.y1) for box in significant]
    median_height = float(np.median(np.asarray(heights, dtype=np.float32)))
    max_gap = max(48, int(round(median_height * 3.2)))

    clusters: list[list[BBox]] = []
    current: list[BBox] = []
    current_bottom = -1
    for box in significant:
        if current and box.y1 - current_bottom > max_gap:
            clusters.append(current)
            current = []
        current.append(box)
        current_bottom = max(current_bottom, box.y2)
    if current:
        clusters.append(current)
    return clusters


def _white_balloon_band_scan_enabled() -> bool:
    raw = os.getenv("TRADUZAI_STRIP_WHITE_BALLOON_BAND_SCAN", "1")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _dark_balloon_band_scan_enabled() -> bool:
    raw = os.getenv("TRADUZAI_STRIP_DARK_BALLOON_BAND_SCAN", "1")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _extract_inner_light_text_boxes(image: np.ndarray, bbox: BBox) -> list[BBox]:
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(bbox.x1)))
    x2 = max(0, min(width, int(bbox.x2)))
    y1 = max(0, min(height, int(bbox.y1)))
    y2 = max(0, min(height, int(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        return []

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    pad_x = max(5, int(w * 0.08))
    pad_y = max(5, int(h * 0.10))
    if w <= pad_x * 2 or h <= pad_y * 2:
        return []

    inner = gray[pad_y : h - pad_y, pad_x : w - pad_x]
    light = (inner >= 150).astype(np.uint8) * 255
    light = cv2.morphologyEx(
        light,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=8)
    boxes: list[BBox] = []
    inner_h, inner_w = inner.shape[:2]
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp_x = int(stats[label, cv2.CC_STAT_LEFT])
        comp_y = int(stats[label, cv2.CC_STAT_TOP])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if comp_x <= 1 or comp_y <= 1 or (comp_x + comp_w) >= inner_w - 1 or (comp_y + comp_h) >= inner_h - 1:
            continue
        if area < 8 or area > max(900, int(w * h * 0.20)):
            continue
        if comp_w < 2 or comp_h < 3:
            continue
        if comp_w > 120 or comp_h > 56:
            continue
        if comp_w > int(w * 0.42) or comp_h > int(h * 0.38):
            continue
        boxes.append(
            BBox(
                x1 + pad_x + comp_x,
                y1 + pad_y + comp_y,
                x1 + pad_x + comp_x + comp_w,
                y1 + pad_y + comp_y + comp_h,
            )
        )
    return boxes


def _extract_light_text_boxes_for_band_scan(image: np.ndarray) -> list[BBox]:
    if image.size == 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    light = (gray >= 172).astype(np.uint8) * 255
    light = cv2.morphologyEx(
        light,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=8)
    height, width = gray.shape[:2]
    boxes: list[BBox] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 10 or area > 1400:
            continue
        if w < 2 or h < 4:
            continue
        if w > 140 or h > 64:
            continue
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            continue
        boxes.append(BBox(x, y, x + w, y + h))
    return boxes


def _scan_dark_balloon_band_candidates(
    image: np.ndarray,
    existing: list[Balloon],
    *,
    y_offset: int = 0,
) -> list[Balloon]:
    """Add lightweight bands for dark bubbles whose light text was missed."""
    if image.size == 0:
        return []

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return []

    local_existing = [
        Balloon(
            strip_bbox=BBox(
                b.strip_bbox.x1,
                b.strip_bbox.y1 - y_offset,
                b.strip_bbox.x2,
                b.strip_bbox.y2 - y_offset,
            ),
            confidence=b.confidence,
            lobe_count=b.lobe_count,
            metadata=dict(getattr(b, "metadata", {}) or {}),
        )
        for b in existing
        if b.strip_bbox.y2 > y_offset and b.strip_bbox.y1 < y_offset + height
    ]

    added: list[Balloon] = []
    for cluster in _cluster_text_components_for_band_scan(_extract_light_text_boxes_for_band_scan(image)):
        text_union = _bbox_union(cluster)
        if text_union is None:
            continue
        if _bbox_area(text_union) < 220:
            continue
        candidate = _expand_bbox(text_union, image.shape)
        evidence = _dark_light_text_evidence(image, candidate)
        if not evidence["useful"]:
            continue
        if evidence["dark_ratio"] < 0.35:
            continue
        if evidence["light_count"] < 2 and evidence["light_area"] < 120:
            continue
        if any(
            _iou(candidate, existing_box.strip_bbox) >= 0.35
            or _bbox_contains_center(existing_box.strip_bbox, candidate, margin=12)
            or _bbox_intersection_ratio(candidate, existing_box.strip_bbox) >= 0.45
            for existing_box in local_existing
        ):
            continue

        metadata = _dark_negative_candidate_metadata(image, candidate)
        metadata["dark_band_scan_candidate"] = True
        metadata["detector_source"] = "dark_balloon_band_scan"
        metadata["detector_sources"] = ["dark_balloon_band_scan"]
        added_balloon = Balloon(
            strip_bbox=BBox(
                candidate.x1,
                candidate.y1 + y_offset,
                candidate.x2,
                candidate.y2 + y_offset,
            ),
            confidence=0.57,
            metadata=metadata,
        )
        added.append(added_balloon)
        local_existing.append(Balloon(strip_bbox=candidate, confidence=0.57, metadata=metadata))

    return added


def _scan_white_balloon_band_candidates(
    image: np.ndarray,
    existing: list[Balloon],
    *,
    y_offset: int = 0,
) -> list[Balloon]:
    """Add lightweight bands for white balloons missed by the detector."""
    if image.size == 0:
        return []

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bright = (gray >= 238).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    local_existing = [
        Balloon(
            strip_bbox=BBox(
                b.strip_bbox.x1,
                b.strip_bbox.y1 - y_offset,
                b.strip_bbox.x2,
                b.strip_bbox.y2 - y_offset,
            ),
            confidence=b.confidence,
        )
        for b in existing
        if b.strip_bbox.y2 > y_offset and b.strip_bbox.y1 < y_offset + height
    ]

    added: list[Balloon] = []
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    image_area = max(1, width * height)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_side_edge = x <= 1 or (x + w) >= width - 1
        touches_vertical_edge = y <= 1 or (y + h) >= height - 1
        large_white_panel = area > int(image_area * 0.18) or (
            touches_side_edge and w >= int(width * 0.82) and h >= 80
        )
        if area < 1800:
            continue
        if not large_white_panel and area > int(image_area * 0.18):
            continue
        if w < 48 or h < 28:
            continue
        aspect = w / float(max(1, h))
        if not large_white_panel and (aspect < 0.45 or aspect > 5.8):
            continue
        if touches_side_edge and not large_white_panel:
            continue

        candidate = BBox(x, y, x + w, y + h)
        if any(
            _iou(candidate, b.strip_bbox) >= 0.55
            or (
                _bbox_contains_center(b.strip_bbox, candidate, margin=18)
                and _bbox_area(b.strip_bbox) >= int(_bbox_area(candidate) * 0.45)
            )
            for b in local_existing
        ):
            continue
        dark_boxes = _extract_inner_dark_text_boxes(image, candidate)
        uncovered_dark_boxes = [
            box
            for box in dark_boxes
            if not any(
                _bbox_contains_center(existing_box.strip_bbox, box, margin=8)
                or _iou(existing_box.strip_bbox, box) >= 0.08
                for existing_box in local_existing
            )
        ]
        significant_count, significant_area = _significant_text_component_count(uncovered_dark_boxes)
        if touches_vertical_edge and not large_white_panel and significant_count < 3:
            continue
        if significant_count < 2 or significant_area < 200:
            continue

        clusters = _cluster_text_components_for_band_scan(uncovered_dark_boxes) if large_white_panel else [
            _significant_text_components(uncovered_dark_boxes)
        ]
        for cluster in clusters:
            text_union = _bbox_union(cluster)
            if text_union is None:
                continue
            band_bbox = _expand_bbox(text_union, image.shape)
            if any(
                _iou(band_bbox, existing_box.strip_bbox) >= 0.45
                or _bbox_contains_center(existing_box.strip_bbox, band_bbox, margin=8)
                for existing_box in local_existing
            ):
                continue

            added.append(
                Balloon(
                    strip_bbox=BBox(
                        band_bbox.x1,
                        band_bbox.y1 + y_offset,
                        band_bbox.x2,
                        band_bbox.y2 + y_offset,
                    ),
                    confidence=0.56,
                    metadata={
                        "detector_source": "white_balloon_band_scan",
                        "detector_sources": ["white_balloon_band_scan"],
                        "candidate_kind": "white_balloon_text_region",
                    },
                )
            )
            local_existing.append(Balloon(strip_bbox=band_bbox, confidence=0.56))

    return added


def _ui_layout_band_scan_enabled() -> bool:
    return _env_flag("TRADUZAI_STRIP_UI_LAYOUT_BAND_SCAN", True)


def _negative_detect_merge_enabled() -> bool:
    return _env_flag("TRADUZAI_STRIP_NEGATIVE_DETECT_MERGE", True)


def _bbox_intersection_ratio(a: BBox, b: BBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / float(max(1, min(_bbox_area(a), _bbox_area(b))))


def _dark_light_text_evidence(image: np.ndarray, bbox: BBox) -> dict:
    evidence = _inner_dark_text_evidence(image, bbox)
    dark_ratio = float(evidence.get("dark_pixel_ratio", 0.0) or 0.0)
    bright_ratio = float(evidence.get("bright_pixel_ratio", 0.0) or 0.0)
    light_count = int(evidence.get("inner_light_component_count", 0) or 0)
    light_area = int(evidence.get("inner_light_area", 0) or 0)
    return {
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "light_count": light_count,
        "light_area": light_area,
        "useful": bool(dark_ratio >= 0.25 and ((light_count >= 1 and light_area >= 60) or bright_ratio >= 0.025)),
    }


def _dark_negative_candidate_metadata(image: np.ndarray, bbox: BBox) -> dict:
    evidence = _dark_light_text_evidence(image, bbox)
    return {
        "background_polarity": "dark",
        "balloon_type": "dark",
        "dark_light_text_evidence": evidence,
        "negative_detect_candidate": True,
    }


def _negative_detect_candidate_is_useful(
    image: np.ndarray,
    bbox: BBox,
    existing: list[Balloon],
) -> bool:
    if image.size == 0:
        return False
    height, width = image.shape[:2]
    if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
        return False
    if bbox.x1 < 0 or bbox.y1 < 0 or bbox.x2 > width or bbox.y2 > height:
        return False

    box_w = bbox.x2 - bbox.x1
    box_h = bbox.y2 - bbox.y1
    box_area = _bbox_area(bbox)
    if box_w < 48 or box_h < 22 or box_area < 900:
        return False
    aspect = box_w / float(max(1, box_h))
    if aspect > 8.0 or aspect < 0.20:
        return False

    overlaps_existing = any(
        _iou(bbox, item.strip_bbox) >= 0.18
        or _bbox_intersection_ratio(bbox, item.strip_bbox) >= 0.45
        or _bbox_contains_center(item.strip_bbox, bbox, margin=14)
        for item in existing
    )

    evidence = _dark_light_text_evidence(image, bbox)
    dark_ratio = evidence["dark_ratio"]
    bright_ratio = evidence["bright_ratio"]
    light_count = evidence["light_count"]
    light_area = evidence["light_area"]
    has_light_text = (light_count >= 2 and light_area >= 180) or bright_ratio >= 0.035
    if overlaps_existing:
        return dark_ratio >= 0.25 and ((light_count >= 1 and light_area >= 60) or bright_ratio >= 0.025)
    return dark_ratio >= 0.35 and has_light_text


def _copy_balloon_with_bbox(balloon: Balloon, bbox: BBox) -> Balloon:
    return Balloon(
        strip_bbox=bbox,
        confidence=float(balloon.confidence),
        lobe_count=int(balloon.lobe_count),
        metadata=dict(getattr(balloon, "metadata", {}) or {}),
    )


def _expand_sparse_dark_negative_candidate(image: np.ndarray, bbox: BBox, *, confidence: float) -> BBox:
    if confidence < 0.70:
        return bbox
    box_w = max(1, bbox.x2 - bbox.x1)
    box_h = max(1, bbox.y2 - bbox.y1)
    if box_w > 260 or box_h > 60:
        return bbox
    evidence = _dark_light_text_evidence(image, bbox)
    if evidence["dark_ratio"] < 0.45 or evidence["bright_ratio"] < 0.08:
        return bbox
    height, width = image.shape[:2]
    pad_left = max(90, int(round(box_w * 0.70)))
    pad_right = max(80, int(round(box_w * 0.55)))
    pad_top = max(260, int(round(box_h * 6.0)))
    pad_bottom = max(90, int(round(box_h * 1.6)))
    expanded = BBox(
        max(0, bbox.x1 - pad_left),
        max(0, bbox.y1 - pad_top),
        min(width, bbox.x2 + pad_right),
        min(height, bbox.y2 + pad_bottom),
    )
    expanded_evidence = _dark_light_text_evidence(image, expanded)
    if expanded_evidence["dark_ratio"] < 0.35:
        return bbox
    if expanded_evidence["light_area"] < max(90, int(evidence["light_area"] * 0.65)):
        return bbox
    return expanded


def _dark_band_text_is_covered(
    image: np.ndarray,
    band_bbox: BBox,
    detector_bbox: BBox,
) -> bool:
    """Check that a precise detector region contains all text behind a broad scan band."""
    text_boxes = [
        box
        for box in _extract_light_text_boxes_for_band_scan(image)
        if _bbox_contains_center(band_bbox, box, margin=0)
    ]
    return bool(text_boxes) and all(
        detector_bbox.x1 <= box.x1
        and detector_bbox.y1 <= box.y1
        and detector_bbox.x2 >= box.x2
        and detector_bbox.y2 >= box.y2
        for box in text_boxes
    )


def _merge_negative_detect_candidates(
    image: np.ndarray,
    current: list[Balloon],
    candidates: list[Balloon],
) -> list[Balloon]:
    if not candidates:
        return current

    merged = list(current)
    for candidate in candidates:
        candidate_metadata = dict(getattr(candidate, "metadata", {}) or {})
        if candidate_metadata.get("negative_detect_candidate"):
            expanded_bbox = _expand_sparse_dark_negative_candidate(
                image,
                candidate.strip_bbox,
                confidence=float(candidate.confidence),
            )
            if expanded_bbox != candidate.strip_bbox:
                candidate_metadata["dark_sparse_text_candidate_expanded"] = {
                    "from": [
                        int(candidate.strip_bbox.x1),
                        int(candidate.strip_bbox.y1),
                        int(candidate.strip_bbox.x2),
                        int(candidate.strip_bbox.y2),
                    ],
                    "to": [int(expanded_bbox.x1), int(expanded_bbox.y1), int(expanded_bbox.x2), int(expanded_bbox.y2)],
                }
                candidate = Balloon(
                    strip_bbox=expanded_bbox,
                    confidence=float(candidate.confidence),
                    lobe_count=int(candidate.lobe_count),
                    metadata=candidate_metadata,
                )
        if not _negative_detect_candidate_is_useful(image, candidate.strip_bbox, merged):
            continue

        best_index = -1
        best_score = 0.0
        for index, existing in enumerate(merged):
            score = max(
                _iou(candidate.strip_bbox, existing.strip_bbox),
                _bbox_intersection_ratio(candidate.strip_bbox, existing.strip_bbox),
            )
            if score > best_score:
                best_score = score
                best_index = index

        if best_index >= 0 and best_score >= 0.25:
            existing = merged[best_index]
            existing_metadata = dict(getattr(existing, "metadata", {}) or {})
            prefer_detector_bbox = bool(
                existing_metadata.get("dark_band_scan_candidate")
                and candidate_metadata.get("negative_detect_candidate")
                and float(candidate.confidence) > float(existing.confidence)
                and _dark_band_text_is_covered(
                    image,
                    existing.strip_bbox,
                    candidate.strip_bbox,
                )
            )
            union = (
                candidate.strip_bbox
                if prefer_detector_bbox
                else _bbox_union([existing.strip_bbox, candidate.strip_bbox])
            )
            if union is None:
                continue
            primary_metadata, secondary_metadata = (
                (candidate_metadata, existing_metadata)
                if float(candidate.confidence) >= float(existing.confidence)
                else (existing_metadata, candidate_metadata)
            )
            metadata = _merge_provenance_metadata(
                primary_metadata,
                secondary_metadata,
            )
            for key in ("dark_band_scan_candidate", "negative_detect_candidate"):
                if existing_metadata.get(key) or candidate_metadata.get(key):
                    metadata[key] = True
            merged[best_index] = Balloon(
                strip_bbox=union,
                confidence=max(float(existing.confidence), float(candidate.confidence)),
                lobe_count=max(int(existing.lobe_count), int(candidate.lobe_count)),
                metadata=metadata,
            )
        else:
            merged.append(candidate)
    return merged


def _scan_ui_layout_band_candidates(
    image: np.ndarray,
    existing: list[Balloon],
    *,
    y_offset: int = 0,
) -> list[Balloon]:
    """Add UI/form panels as band candidates before OCR crops are created."""
    if image.size == 0:
        return []
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return []

    try:
        from vision_stack.ui_layout import detect_uied_like_components
    except Exception:
        return []

    local_existing = [
        Balloon(
            strip_bbox=BBox(
                b.strip_bbox.x1,
                b.strip_bbox.y1 - y_offset,
                b.strip_bbox.x2,
                b.strip_bbox.y2 - y_offset,
            ),
            confidence=b.confidence,
        )
        for b in existing
        if b.strip_bbox.y2 > y_offset and b.strip_bbox.y1 < y_offset + height
    ]

    added: list[Balloon] = []
    image_area = max(1, width * height)
    for component in detect_uied_like_components(image):
        x1, y1, x2, y2 = [int(v) for v in component.bbox]
        box = BBox(max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        if box.width < 48 or box.height < 12:
            continue
        area = _bbox_area(box)
        if area < 480 or area > int(image_area * 0.28):
            continue
        if any(
            _iou(box, existing_box.strip_bbox) >= 0.45
            or _bbox_contains_center(existing_box.strip_bbox, box, margin=14)
            for existing_box in local_existing
        ):
            continue
        confidence = max(0.52, min(0.86, float(component.confidence)))
        added.append(
            Balloon(
                strip_bbox=BBox(
                    box.x1,
                    box.y1 + y_offset,
                    box.x2,
                    box.y2 + y_offset,
                ),
                confidence=confidence,
                metadata={
                    "detector_source": "ui_layout_band_scan",
                    "detector_sources": ["ui_layout_band_scan"],
                    "candidate_kind": "ui_layout_text_region",
                },
            )
        )
        local_existing.append(Balloon(strip_bbox=box, confidence=confidence))

    return added


def _split_into_chunks(
    strip_height: int,
    chunk_height: int = 4096,
    overlap: int = 512,
) -> list[tuple[int, int]]:
    """Retorna lista de (y_start, y_end) para sliding window."""
    if strip_height <= chunk_height:
        return [(0, strip_height)]
    step = chunk_height - overlap
    chunks: list[tuple[int, int]] = []
    cursor = 0
    while cursor < strip_height:
        end = min(cursor + chunk_height, strip_height)
        chunks.append((cursor, end))
        if end >= strip_height:
            break
        cursor += step
    return chunks


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _source_page_chunks(strip: VerticalStrip) -> list[tuple[int, int]]:
    breaks: list[int] = []
    for value in strip.source_page_breaks or []:
        try:
            item = int(value)
        except Exception:
            continue
        if item >= 0:
            breaks.append(item)
    if not breaks:
        return [(0, int(strip.height))]
    if breaks[0] != 0:
        breaks.insert(0, 0)
    if breaks[-1] != int(strip.height):
        breaks.append(int(strip.height))
    chunks: list[tuple[int, int]] = []
    for y0, y1 in zip(breaks, breaks[1:]):
        y0 = max(0, min(int(strip.height), int(y0)))
        y1 = max(0, min(int(strip.height), int(y1)))
        if y1 > y0:
            chunks.append((y0, y1))
    return chunks or [(0, int(strip.height))]


def _detect_chunks_for_strip(
    strip: VerticalStrip,
    *,
    chunk_height: int,
    overlap: int,
) -> tuple[list[tuple[int, int]], str]:
    if _env_flag("TRADUZAI_STRIP_DETECT_FULL_PAGE", False):
        return _source_page_chunks(strip), "source_page"
    return _split_into_chunks(strip.height, chunk_height, overlap), "sliding_window"


def _is_oversized(
    bbox: BBox,
    strip_width: int,
    strip_height: int,
    max_height_fraction: float = 0.25,
    max_width_fraction: float = 0.95,
) -> bool:
    """Retorna True se o bbox parece ser um false-positive do detector (muito grande)."""
    cap_h = int(strip_height * max_height_fraction)
    cap_w = int(strip_width * max_width_fraction)
    return bbox.height > cap_h or bbox.width > cap_w


def _source_page_height_for_bbox(strip: VerticalStrip, bbox: BBox) -> int:
    center_y = int(round((int(bbox.y1) + int(bbox.y2)) / 2.0))
    for y0, y1 in _source_page_chunks(strip):
        if y0 <= center_y < y1:
            return max(1, y1 - y0)
    return max(1, int(strip.height))


def _is_white_balloon_text_candidate(image: np.ndarray, bbox: BBox) -> bool:
    if image.size == 0:
        return False
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(bbox.x1)))
    x2 = max(0, min(width, int(bbox.x2)))
    y1 = max(0, min(height, int(bbox.y1)))
    y2 = max(0, min(height, int(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        return False
    evidence = _inner_dark_text_evidence(image, BBox(x1, y1, x2, y2))
    bright_ratio = float(evidence.get("bright_pixel_ratio", 0.0) or 0.0)
    dark_ratio = float(evidence.get("dark_pixel_ratio", 0.0) or 0.0)
    significant_count = int(evidence.get("significant_component_count", 0) or 0)
    significant_area = int(evidence.get("significant_area", 0) or 0)
    return bool(
        bright_ratio >= 0.50
        and dark_ratio >= 0.035
        and significant_count >= 3
        and significant_area >= 180
    )


def _page_location_for_bbox(strip: VerticalStrip, bbox: BBox) -> tuple[int, int, int]:
    """Return (page index, page y origin, page x origin) by maximum overlap."""

    chunks = _source_page_chunks(strip)
    best_index = 0
    best_overlap = -1
    for index, (page_y0, page_y1) in enumerate(chunks):
        overlap = max(0, min(int(bbox.y2), page_y1) - max(int(bbox.y1), page_y0))
        if overlap > best_overlap:
            best_index = index
            best_overlap = overlap
    page_y0 = chunks[best_index][0]
    offsets = list(getattr(strip, "page_x_offsets", None) or [])
    page_x0 = int(offsets[best_index]) if best_index < len(offsets) else 0
    return best_index, int(page_y0), page_x0


def _attach_page_region_identities(strip: VerticalStrip, balloons: list[Balloon]) -> list[Balloon]:
    """Attach stable page-space region identity after all geometry mutations."""

    from ownership.coordinates import ComponentSeed, assign_component_ids

    source_widths = [
        int(value) for value in list(getattr(strip, "source_page_widths", None) or [])
    ]
    by_page: dict[str, list[tuple[Balloon, ComponentSeed, list[int], dict]]] = {}
    for balloon in balloons:
        page_index, page_y0, page_x0 = _page_location_for_bbox(strip, balloon.strip_bbox)
        page_number_offset = int(getattr(strip, "page_number_offset", 0) or 0)
        page_id = f"page_{page_number_offset + page_index + 1:03d}"
        page_chunks = _source_page_chunks(strip)
        page_height = max(1, int(page_chunks[page_index][1] - page_y0))
        page_width = (
            source_widths[page_index]
            if page_index < len(source_widths) and source_widths[page_index] > 0
            else max(1, int(strip.width) - 2 * page_x0)
        )
        bbox_page = [
            max(0, min(page_width, int(balloon.strip_bbox.x1) - page_x0)),
            max(0, min(page_height, int(balloon.strip_bbox.y1) - page_y0)),
            max(0, min(page_width, int(balloon.strip_bbox.x2) - page_x0)),
            max(0, min(page_height, int(balloon.strip_bbox.y2) - page_y0)),
        ]
        metadata = dict(getattr(balloon, "metadata", {}) or {})
        detector_source = str(
            metadata.get("detector_source")
            or metadata.get("candidate_source")
            or ("negative_region_detector" if metadata.get("negative_detect_candidate") else "primary_region_detector")
        )
        detector_sources = sorted(
            {
                detector_source,
                *(str(value) for value in metadata.get("detector_sources") or [] if value),
            }
        )
        metadata["detector_source"] = detector_source
        metadata["detector_sources"] = detector_sources

        page_polygons: list[list[list[int]]] = []
        for polygon in metadata.get("line_polygons_strip") or []:
            page_polygon = [
                [
                    max(0, min(page_width, int(round(float(point[0]))) - page_x0)),
                    max(0, min(page_height, int(round(float(point[1]))) - page_y0)),
                ]
                for point in polygon
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            if len(page_polygon) >= 3:
                page_polygons.append(page_polygon)
        if page_polygons:
            metadata["line_polygons_page"] = page_polygons
            if len(page_polygons) == 1:
                metadata["polygon_page"] = page_polygons[0]
            else:
                hull_points = np.asarray(
                    [point for polygon in page_polygons for point in polygon],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                hull = cv2.convexHull(hull_points, clockwise=False, returnPoints=True)
                metadata["polygon_page"] = [
                    [int(point[0][0]), int(point[0][1])] for point in hull
                ]
        else:
            x1, y1, x2, y2 = bbox_page
            metadata["polygon_page"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        seed = ComponentSeed(tuple(bbox_page), "+".join(detector_sources))
        by_page.setdefault(page_id, []).append((balloon, seed, bbox_page, metadata))

    for page_id, entries in by_page.items():
        component_ids = assign_component_ids(page_id, [entry[1] for entry in entries])
        for (balloon, _seed, bbox_page, metadata), component_id in zip(entries, component_ids):
            metadata.update(
                {
                    "bbox_page": bbox_page,
                    "coordinate_space": "page",
                    "page_id": page_id,
                    "region_id": component_id,
                    "evidence_id": component_id,
                }
            )
            balloon.metadata = metadata
    return balloons


def detect_strip_balloons(
    strip,
    detector,
    chunk_height: int = 4096,
    overlap: int = 512,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    max_height_fraction: float = 0.25,
    max_width_fraction: float = 0.95,
) -> list[Balloon]:
    """Detecta balões no strip via sliding window + NMS.

    Filtros pós-NMS:
    - Descarta bboxes com altura > `max_height_fraction` * strip.height
    - Descarta bboxes com largura > `max_width_fraction` * strip.width
    """
    chunks, chunk_mode = _detect_chunks_for_strip(strip, chunk_height=chunk_height, overlap=overlap)
    all_balloons: list[Balloon] = []

    for y0, y1 in chunks:
        chunk_img = strip.image[y0:y1, :, :]
        blocks = detector.detect(chunk_img, conf_threshold=confidence_threshold)
        chunk_balloons: list[Balloon] = []
        for b in blocks:
            bbox = BBox(
                x1=int(b.x1),
                y1=int(b.y1) + y0,
                x2=int(b.x2),
                y2=int(b.y2) + y0,
            )
            chunk_balloons.append(
                Balloon(
                    strip_bbox=bbox,
                    confidence=float(b.confidence),
                    metadata=_detector_block_metadata(
                        b,
                        y_offset=y0,
                        default_source="primary_region_detector",
                    ),
                )
            )
        all_balloons.extend(chunk_balloons)
        if _white_balloon_band_scan_enabled():
            added = _scan_white_balloon_band_candidates(
                chunk_img,
                all_balloons,
                y_offset=y0,
            )
            all_balloons.extend(added)
        if _dark_balloon_band_scan_enabled():
            added = _scan_dark_balloon_band_candidates(
                chunk_img,
                all_balloons,
                y_offset=y0,
            )
            all_balloons.extend(added)
        if _ui_layout_band_scan_enabled():
            added = _scan_ui_layout_band_candidates(
                chunk_img,
                all_balloons,
                y_offset=y0,
            )
            all_balloons.extend(added)
        if _negative_detect_merge_enabled() and chunk_img.size:
            negative_blocks = detector.detect(cv2.bitwise_not(chunk_img), conf_threshold=confidence_threshold)
            negative_balloons = [
                Balloon(
                    strip_bbox=BBox(
                        x1=int(b.x1),
                        y1=int(b.y1) + y0,
                        x2=int(b.x2),
                        y2=int(b.y2) + y0,
                    ),
                    confidence=float(b.confidence),
                    metadata={
                        **_detector_block_metadata(
                            b,
                            y_offset=y0,
                            default_source="negative_region_detector",
                        ),
                        "negative_detect_candidate": True,
                    },
                )
                for b in negative_blocks
            ]
            if negative_balloons:
                local_existing = [
                    Balloon(
                        strip_bbox=BBox(
                            item.strip_bbox.x1,
                            item.strip_bbox.y1 - y0,
                            item.strip_bbox.x2,
                            item.strip_bbox.y2 - y0,
                        ),
                        confidence=item.confidence,
                        lobe_count=item.lobe_count,
                        metadata=dict(getattr(item, "metadata", {}) or {}),
                    )
                    for item in all_balloons
                    if item.strip_bbox.y2 > y0 and item.strip_bbox.y1 < y1
                ]
                local_negative = [
                    Balloon(
                        strip_bbox=BBox(
                            item.strip_bbox.x1,
                            item.strip_bbox.y1 - y0,
                            item.strip_bbox.x2,
                            item.strip_bbox.y2 - y0,
                        ),
                        confidence=item.confidence,
                        lobe_count=item.lobe_count,
                        metadata=dict(getattr(item, "metadata", {}) or {}),
                    )
                    for item in negative_balloons
                ]
                for item in local_negative:
                    item.metadata.update(_dark_negative_candidate_metadata(chunk_img, item.strip_bbox))
                local_merged = _merge_negative_detect_candidates(
                    chunk_img,
                    local_existing,
                    local_negative,
                )
                all_balloons = [
                    item
                    for item in all_balloons
                    if not (item.strip_bbox.y2 > y0 and item.strip_bbox.y1 < y1)
                ]
                all_balloons.extend(
                    Balloon(
                        strip_bbox=BBox(
                            item.strip_bbox.x1,
                            item.strip_bbox.y1 + y0,
                            item.strip_bbox.x2,
                            item.strip_bbox.y2 + y0,
                        ),
                        confidence=item.confidence,
                        lobe_count=item.lobe_count,
                        metadata=dict(getattr(item, "metadata", {}) or {}),
                    )
                    for item in local_merged
                )

    page_native = str(getattr(strip, "raster_mode", "")) == "page_map_v1"
    after_nms = _nms_balloons(
        all_balloons,
        iou_threshold=iou_threshold,
        intersection_smaller_threshold=0.8 if page_native else None,
        require_same_class=page_native,
    )

    # Local test only: Mayo proposes balloon geometry, never a cleanup mask.
    # The normal OCR/owner gates still decide whether any text may be changed.
    from strip import experimental_mayo
    if experimental_mayo.enabled():
        import json
        import logging
        from pathlib import Path

        try:
            proposals, mayo_stats = experimental_mayo.detect(strip.image)
            mayo_stats["accepted_new_candidates"] = 0
            mayo_stats["rejected_no_text_evidence"] = 0
            mayo_stats["overlapped_existing"] = 0
            for proposal in proposals:
                x1, y1, x2, y2 = map(int, proposal["bbox"])
                if not (0 <= x1 < x2 <= strip.width and 0 <= y1 < y2 <= strip.height):
                    continue
                bbox = BBox(x1, y1, x2, y2)
                if any(_iou(bbox, existing.strip_bbox) >= 0.5 or
                       _intersection_over_smaller_area(bbox, existing.strip_bbox) >= 0.8
                       for existing in after_nms):
                    mayo_stats["overlapped_existing"] += 1
                    continue
                ink = _inner_dark_text_evidence(strip.image, bbox)
                if not (ink["has_inner_dark_text"] or ink["has_inner_light_text"]):
                    mayo_stats["rejected_no_text_evidence"] += 1
                    continue
                after_nms.append(Balloon(
                    strip_bbox=bbox,
                    confidence=float(proposal["confidence"]),
                    metadata={
                        "detector_source": "experimental_mayo_safetensors",
                        "candidate_kind": "text_region",
                        "mayo_candidate_review_required": True,
                        "mayo_mask_role": "ownership_candidate_only",
                        "mayo_mask_sha256": proposal["mask_sha256"],
                        "mayo_repair": proposal["repair"],
                        "mayo_model": proposal["model"],
                        "mayo_ink_evidence": ink,
                    },
                ))
                mayo_stats["accepted_new_candidates"] += 1
            log_path = os.getenv("TRADUZAI_EXPERIMENTAL_MAYO_LOG")
            if log_path:
                with Path(log_path).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(mayo_stats, ensure_ascii=False) + "\n")
        except Exception as exc:
            # Experiment failure cannot make the official detector invent boxes.
            logging.getLogger(__name__).warning("Mayo experimental recusado: %s", exc)

    # Filtro pós-NMS: descartar false-positives gigantes
    filtered = []
    for balloon in after_nms:
        expanded_bbox = _expand_sparse_dark_negative_candidate(
            strip.image,
            balloon.strip_bbox,
            confidence=float(balloon.confidence),
        )
        if expanded_bbox != balloon.strip_bbox:
            metadata = dict(getattr(balloon, "metadata", {}) or {})
            metadata["dark_sparse_text_candidate_expanded"] = {
                "from": [
                    int(balloon.strip_bbox.x1),
                    int(balloon.strip_bbox.y1),
                    int(balloon.strip_bbox.x2),
                    int(balloon.strip_bbox.y2),
                ],
                "to": [int(expanded_bbox.x1), int(expanded_bbox.y1), int(expanded_bbox.x2), int(expanded_bbox.y2)],
            }
            balloon = Balloon(
                strip_bbox=expanded_bbox,
                confidence=float(balloon.confidence),
                lobe_count=int(balloon.lobe_count),
                metadata=metadata,
            )
        filter_height = (
            _source_page_height_for_bbox(strip, balloon.strip_bbox)
            if chunk_mode == "source_page"
            else strip.height
        )
        dark_negative_candidate = bool((getattr(balloon, "metadata", {}) or {}).get("negative_detect_candidate"))
        white_balloon_scan_candidate = (
            str((getattr(balloon, "metadata", {}) or {}).get("candidate_kind") or "").strip()
            == "white_balloon_text_region"
        )
        white_balloon_text_candidate = white_balloon_scan_candidate or _is_white_balloon_text_candidate(
            strip.image,
            balloon.strip_bbox,
        )
        oversized = _is_oversized(
            balloon.strip_bbox,
            strip.width,
            filter_height,
            max_height_fraction,
            max_width_fraction,
        )
        if oversized and dark_negative_candidate:
            oversized = _is_oversized(
                balloon.strip_bbox,
                strip.width,
                filter_height,
                max(max_height_fraction, 0.55),
                max_width_fraction,
            )
        if oversized and white_balloon_text_candidate:
            oversized = _is_oversized(
                balloon.strip_bbox,
                strip.width,
                filter_height,
                max(max_height_fraction, 0.90),
                max_width_fraction,
            )
        if not oversized:
            filtered.append(balloon)
    ordered = sorted(
        filtered,
        key=lambda balloon: (
            int(balloon.strip_bbox.y1),
            int(balloon.strip_bbox.x1),
            int(balloon.strip_bbox.y2),
            int(balloon.strip_bbox.x2),
        ),
    )
    return _attach_page_region_identities(strip, ordered)
