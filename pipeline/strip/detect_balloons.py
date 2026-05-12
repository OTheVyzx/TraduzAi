"""Detecção de balões sobre o strip via sliding window + NMS."""

from __future__ import annotations

import os

import cv2
import numpy as np

from strip.types import Balloon, BBox


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


def _nms_balloons(balloons: list[Balloon], iou_threshold: float = 0.5) -> list[Balloon]:
    """Remove balões redundantes; mantém o de maior confidence em cada cluster."""
    if not balloons:
        return []
    sorted_balloons = sorted(balloons, key=lambda b: b.confidence, reverse=True)
    kept: list[Balloon] = []
    for cand in sorted_balloons:
        is_dup = any(_iou(cand.strip_bbox, k.strip_bbox) > iou_threshold for k in kept)
        if not is_dup:
            kept.append(cand)
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
                )
            )
            local_existing.append(Balloon(strip_bbox=band_bbox, confidence=0.56))

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
    chunks = _split_into_chunks(strip.height, chunk_height, overlap)
    all_balloons: list[Balloon] = []

    for y0, y1 in chunks:
        chunk_img = strip.image[y0:y1, :, :]
        blocks = detector.detect(chunk_img, conf_threshold=confidence_threshold)
        for b in blocks:
            bbox = BBox(
                x1=int(b.x1),
                y1=int(b.y1) + y0,
                x2=int(b.x2),
                y2=int(b.y2) + y0,
            )
            all_balloons.append(
                Balloon(strip_bbox=bbox, confidence=float(b.confidence))
            )
        if _white_balloon_band_scan_enabled():
            all_balloons.extend(
                _scan_white_balloon_band_candidates(
                    chunk_img,
                    all_balloons,
                    y_offset=y0,
                )
            )

    after_nms = _nms_balloons(all_balloons, iou_threshold=iou_threshold)

    # Filtro pós-NMS: descartar false-positives gigantes
    filtered = [
        b for b in after_nms
        if not _is_oversized(b.strip_bbox, strip.width, strip.height,
                             max_height_fraction, max_width_fraction)
    ]
    return filtered
