"""Bounded OCR refinement for tall sources and overlap-safe reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any, Mapping, Sequence


def _bbox(value: Any) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        result = ()
    if len(result) != 4 or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("observation bbox must be non-empty")
    return result


def _signature(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).upper()
    return re.sub(r"[^0-9A-Z]+", "", normalized)


def _similarity(first: str, second: str) -> float:
    if not first and not second:
        return 1.0
    return SequenceMatcher(a=first, b=second, autojunk=False).ratio()


def _intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / float(first_area + second_area - intersection)


@dataclass(frozen=True)
class RefinementPlan:
    regions: tuple[tuple[int, int, int, int], ...]
    coverage_complete: bool
    selection_basis: str


@dataclass(frozen=True)
class ReconciledObservations:
    observations: tuple[dict[str, Any], ...]
    duplicate_count: int
    status: str
    uncertainty_reasons: tuple[str, ...]


def plan_vertical_refinement(
    *,
    page_width: int,
    page_height: int,
    tile_height: int = 1800,
    overlap: int = 180,
    max_tiles: int = 8,
) -> RefinementPlan:
    """Cover a tall page using only source dimensions, never page identity/content."""

    width = int(page_width)
    height = int(page_height)
    window = int(tile_height)
    shared = int(overlap)
    budget = int(max_tiles)
    if width <= 0 or height <= 0:
        raise ValueError("page dimensions must be positive")
    if window <= 0 or shared < 0 or shared >= window:
        raise ValueError("tile height and overlap are invalid")
    if budget <= 0:
        raise ValueError("refinement budget must be positive")
    if height <= window:
        return RefinementPlan(((0, 0, width, height),), True, "source_dimensions_only")

    stride = window - shared
    starts = list(range(0, max(1, height - window + 1), stride))
    final_start = height - window
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    starts = sorted(set(starts))
    if len(starts) > budget:
        raise ValueError(
            f"refinement budget {budget} cannot cover {height}px with {window}px tiles"
        )
    regions = tuple((0, start, width, min(height, start + window)) for start in starts)
    complete = regions[0][1] == 0 and regions[-1][3] == height and all(
        left[3] >= right[1] for left, right in zip(regions, regions[1:])
    )
    return RefinementPlan(regions, complete, "source_dimensions_only")


def reconcile_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    duplicate_iou: float = 0.5,
) -> ReconciledObservations:
    """Deduplicate only matching text at overlapping geometry; preserve uncertainty."""

    normalized: list[dict[str, Any]] = []
    for ordinal, row in enumerate(observations):
        identity = str(row.get("observation_id") or "").strip()
        if not identity:
            raise ValueError("observation_id is required")
        bbox = _bbox(row.get("bbox_page"))
        copied = dict(row)
        copied["bbox_page"] = list(bbox)
        copied["confidence"] = float(row.get("confidence") or 0.0)
        prior_sources = [
            str(value) for value in row.get("reconciled_from", ()) if str(value)
        ]
        copied["reconciled_from"] = prior_sources or [identity]
        prior_alternatives = row.get("alternatives")
        copied["alternatives"] = (
            [dict(value) for value in prior_alternatives]
            if isinstance(prior_alternatives, (list, tuple)) and prior_alternatives
            else [{
                "observation_id": identity,
                "attempt_id": str(row.get("attempt_id") or ""),
                "text": str(row.get("text") or ""),
                "confidence": copied["confidence"],
                "bbox_page": list(bbox),
            }]
        )
        copied["_ordinal"] = ordinal
        copied["_signature"] = _signature(row.get("text"))
        normalized.append(copied)

    groups: list[dict[str, Any]] = []
    duplicate_count = 0
    for candidate in normalized:
        match = None
        for existing in groups:
            overlap = _intersection_over_union(
                tuple(candidate["bbox_page"]), tuple(existing["bbox_page"])
            )
            text_similarity = _similarity(
                candidate["_signature"], existing["_signature"]
            )
            if overlap >= 0.8 or (
                overlap >= float(duplicate_iou) and text_similarity >= 0.8
            ):
                match = existing
                break
        if match is None:
            groups.append(candidate)
            continue
        duplicate_count += 1
        provenance = list(dict.fromkeys(
            list(match["reconciled_from"]) + list(candidate["reconciled_from"])
        ))
        alternatives = list(match["alternatives"]) + list(candidate["alternatives"])
        if (
            candidate["confidence"], -candidate["_ordinal"]
        ) > (
            match["confidence"], -match["_ordinal"]
        ):
            candidate["reconciled_from"] = provenance
            candidate["alternatives"] = alternatives
            groups[groups.index(match)] = candidate
        else:
            match["reconciled_from"] = provenance
            match["alternatives"] = alternatives

    published = []
    for row in sorted(groups, key=lambda item: (
        item["bbox_page"][1], item["bbox_page"][0], item["_ordinal"]
    )):
        row = dict(row)
        row.pop("_ordinal", None)
        row.pop("_signature", None)
        published.append(row)
    if not published:
        return ReconciledObservations(
            (), duplicate_count, "review_required",
            ("no_observations_not_negative_evidence",),
        )
    return ReconciledObservations(tuple(published), duplicate_count, "complete", ())
