"""Detecção de balões sobre o strip via sliding window + NMS."""

from __future__ import annotations

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


def detect_strip_balloons(
    strip,
    detector,
    chunk_height: int = 4096,
    overlap: int = 512,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
) -> list[Balloon]:
    """Detecta balões no strip via sliding window + NMS."""
    import numpy as np
    from strip.types import VerticalStrip
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

    return _nms_balloons(all_balloons, iou_threshold=iou_threshold)
