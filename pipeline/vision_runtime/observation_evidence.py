"""Conservative evidence states for OCR/detection proposals."""

from __future__ import annotations

import re
from statistics import median
from typing import Any, Mapping, Sequence


def _bbox(value: Any) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        result = ()
    if len(result) != 4 or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("observation bbox must be non-empty")
    return result


def assess_observation_evidence(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Mark implausibly large sparse glyphs uncertain without deleting them."""

    copied: list[dict[str, Any]] = []
    baseline_heights: list[int] = []
    for row in observations:
        item = dict(row)
        bbox = _bbox(row.get("bbox_page"))
        item["bbox_page"] = list(bbox)
        item["confidence"] = float(row.get("confidence") or 0.0)
        item["_alnum_count"] = len(re.sub(r"[^0-9A-Za-z]+", "", str(row.get("text") or "")))
        item["_height"] = bbox[3] - bbox[1]
        if item["_alnum_count"] >= 2 and item["confidence"] >= 0.7:
            baseline_heights.append(item["_height"])
        copied.append(item)
    if not copied:
        return ()
    reference_height = float(median(baseline_heights or [row["_height"] for row in copied]))
    large_threshold = max(64.0, reference_height * 2.5)

    assessed = []
    for item in copied:
        reasons = [str(value) for value in item.get("uncertainty_reasons", ())]
        if item["_height"] >= large_threshold:
            if item["_alnum_count"] <= 1:
                reasons.append("sparse_large_glyph_candidate")
            elif item["confidence"] < 0.7:
                reasons.append("low_confidence_large_shape_candidate")
        reasons = list(dict.fromkeys(reasons))
        item["selection_state"] = "uncertain" if reasons else "eligible"
        item["uncertainty_reasons"] = reasons
        item.pop("_alnum_count", None)
        item.pop("_height", None)
        assessed.append(item)
    return tuple(assessed)
