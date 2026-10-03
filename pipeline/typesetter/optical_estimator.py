"""Estimate a work-level ordinary dialogue scale from source typography.

This consumes existing source-scale evidence, never target text or balloon size.
Expressive text is excluded and a weak sample set stays unresolved.
"""
from __future__ import annotations

import hashlib
import json
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from .stable_baseline import line_advance, rasterize


def estimate_optical_profile(
    samples: Sequence[Mapping[str, Any]], *, font_path: str,
    font_name: str, page_width_px: int,
) -> tuple[dict[str, Any] | None, str]:
    """Return a reusable profile only for independent, consistent source lines."""
    if page_width_px <= 0 or not font_name or not font_path:
        return None, 'invalid_font_or_page'
    rows: list[tuple[str, str, float, float]] = []
    for sample in samples:
        if str(sample.get('visual_role') or '') != 'ordinary_dialogue':
            continue
        try:
            confidence = float(sample.get('source_scale_evidence_confidence'))
            source_width = int(sample.get('page_width_px'))
            owner = str(sample.get('owner_id') or '')
            ids = [str(value) for value in sample.get('source_scale_evidence_ids') or []]
            heights = [float(value) for value in sample.get('source_x_heights_px') or []]
        except (TypeError, ValueError):
            continue
        if confidence < 0.75 or source_width <= 0 or not owner or not ids:
            continue
        for index, height in enumerate(heights):
            if not np.isfinite(height) or height <= 0:
                continue
            evidence_id = ids[min(index, len(ids) - 1)]
            rows.append((owner, evidence_id, height * page_width_px / source_width, confidence))
    unique = {(owner, evidence): (owner, evidence, height, confidence)
              for owner, evidence, height, confidence in rows}
    rows = list(unique.values())
    if len(rows) < 3 or len({row[0] for row in rows}) < 2:
        return None, 'source_scale_evidence_insufficient'
    heights = [row[2] for row in rows]
    target = float(median(heights))
    spread = float(median(abs(value - target) for value in heights))
    if spread > target * 0.25:
        return None, 'ordinary_scale_inconsistent'
    def body_at(size: int) -> int:
        mask = rasterize(font_path, size, 'HATO')
        yy = np.nonzero(mask > 127)[0]
        return int(yy.max() - yy.min() + 1) if len(yy) else 0

    low, high = 8, max(9, min(192, round(page_width_px * 0.2)))
    while low < high:
        mid = (low + high) // 2
        if body_at(mid) < target:
            low = mid + 1
        else:
            high = mid
    best_size = None
    best_delta = float('inf')
    for size in range(max(8, low - 2), min(192, low + 2) + 1):
        body = body_at(size)
        delta = abs(body - target)
        if delta < best_delta:
            best_size, best_delta = size, delta
    if best_size is None or best_delta > max(2.0, target * 0.15):
        return None, 'font_scale_match_uncertain'
    advance = line_advance(font_path, best_size)
    canonical = json.dumps(
        [{'owner_id': owner, 'evidence_id': evidence, 'body_px': round(height, 3),
          'confidence': round(confidence, 3)} for owner, evidence, height, confidence in sorted(rows)],
        sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')
    return {
        'font_name': font_name,
        'reference_page_width_px': page_width_px,
        'nominal_size_px': best_size,
        'line_advance_px': advance,
        'optical_body_px': target,
        'optical_tolerance_px': max(1.0, target * 0.12),
        'min_contour_px': max(2.0, target * 0.60),
        'max_center_drift_px': max(1.0, target * 0.10),
        'source_evidence_sha256': hashlib.sha256(canonical).hexdigest(),
    }, 'estimated_from_independent_source_lines'
