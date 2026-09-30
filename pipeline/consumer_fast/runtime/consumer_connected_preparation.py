"""Conservative source-driven planning for independent connected text bodies.

This module only proposes subblocks. It does not turn a polygon inferred from
lettering into a confirmed balloon or grant translation/render authority.
"""
from __future__ import annotations

import re
import statistics
from typing import Any


def _box(values: Any) -> list[int]:
    box = [int(value) for value in values]
    if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("invalid source geometry")
    return box


def _union(boxes: list[list[int]]) -> list[int]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _crop(box: list[int], width: int, height: int) -> list[int]:
    pad_x = max(8, round((box[2] - box[0]) * 0.06))
    pad_y = max(8, round((box[3] - box[1]) * 0.08))
    return [max(0, box[0] - pad_x), max(0, box[1] - pad_y),
            min(width, box[2] + pad_x), min(height, box[3] + pad_y)]


def _uncovered_line_gap(observations: list[dict]) -> list[dict]:
    ordered = sorted(observations, key=lambda row: (_box(row["bbox_page"])[1],
                                                    _box(row["bbox_page"])[0]))
    heights = [box[3] - box[1] for row in ordered if (box := _box(row["bbox_page"]))]
    threshold = max(8, round(statistics.median(heights) * 0.40))
    gaps = []
    for first, second in zip(ordered, ordered[1:]):
        upper, lower = _box(first["bbox_page"]), _box(second["bbox_page"])
        gap = lower[1] - upper[3]
        if gap > threshold:
            gaps.append(dict(after=first["observation_id"],
                             before=second["observation_id"], gap_px=gap,
                             threshold_px=threshold))
    return gaps


def tokenization_conflicts(variants: list[dict]) -> list[dict]:
    """Find overlapping OCR lines with identical letters but different spaces."""
    conflicts = {}
    for index, left in enumerate(variants):
        a = _box(left["bbox_page"])
        letters = re.sub(r"[^A-Z0-9]", "", str(left.get("text") or "").upper())
        if len(letters) < 5:
            continue
        for right in variants[index + 1:]:
            b = _box(right["bbox_page"])
            overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if overlap < 0.50 * min(a[3] - a[1], b[3] - b[1]):
                continue
            other_letters = re.sub(r"[^A-Z0-9]", "", str(right.get("text") or "").upper())
            if letters != other_letters:
                continue
            texts = sorted({str(left["text"]).strip(), str(right["text"]).strip()})
            if len(texts) == 2 and len(texts[0].split()) != len(texts[1].split()):
                conflicts[(letters, *texts)] = dict(normalized_letters=letters, texts=texts)
    return list(conflicts.values())


def plan_connected_subblocks(
    owner_id: str,
    connected_subregions: list[dict],
    selected_observations: list[dict],
    page_width: int,
    page_height: int,
) -> dict:
    """Propose local source units only when every boundary has evidence.

    Distinct components and completed utterances are required. This abstains
    for a normal multiline sentence even if its detector emitted many lines.
    """
    if len(connected_subregions) < 2:
        return dict(status="not_independent", reason="fewer_than_two_subregions", units=[])
    observed = {str(row["observation_id"]): row for row in selected_observations}
    if len(observed) != len(selected_observations):
        raise ValueError("duplicate selected observation identity")
    units = []
    consumed = set()
    components = set()
    for region in sorted(connected_subregions, key=lambda row: (int(row.get("order", 0)),
                                                                str(row.get("subregion_id")))):
        component_ids = {str(value) for value in region.get("component_ids") or []}
        evidence_ids = [str(value) for value in region.get("evidence_ids") or []]
        if not component_ids or component_ids & components or not evidence_ids:
            return dict(status="review_required", reason="component_partition_unproven", units=[])
        if len(evidence_ids) != len(set(evidence_ids)) or not set(evidence_ids) <= set(observed):
            return dict(status="review_required", reason="observation_partition_unproven", units=[])
        if set(evidence_ids) & consumed:
            return dict(status="review_required", reason="observation_partition_unproven", units=[])
        rows = [observed[value] for value in evidence_ids]
        if any(not set(row.get("component_ids") or []) <= component_ids for row in rows):
            return dict(status="review_required", reason="observation_component_mismatch", units=[])
        rows.sort(key=lambda row: (_box(row["bbox_page"])[1], _box(row["bbox_page"])[0]))
        box = _box(region["bbox_page"])
        if not (0 <= box[0] < box[2] <= page_width and
                0 <= box[1] < box[3] <= page_height):
            raise ValueError("connected subregion outside source page")
        source = " ".join(str(row.get("text") or "").strip() for row in rows).strip()
        if not source:
            return dict(status="review_required", reason="source_text_missing", units=[])
        gaps = _uncovered_line_gap(rows)
        confusions = re.findall(r"(?i)(?:\b[A-Z]*[015!][A-Z]+\b|\b[A-Z]+[015!][A-Z]*\b)", source)
        units.append(dict(subregion_id=str(region["subregion_id"]),
                          component_ids=sorted(component_ids),
                          source_text=source,
                          selected_observation_ids=[row["observation_id"] for row in rows],
                          source_anchor_bbox=_union([_box(row["bbox_page"]) for row in rows]),
                          safe_bbox_page=box,
                          safe_polygon_page=region["polygon_page"],
                          source_observations=rows,
                          uncovered_line_gaps=gaps,
                          possible_ocr_confusions=confusions,
                          ocr_recovery_required=bool(gaps or confusions),
                          ocr_crop_bbox_page=_crop(box, page_width, page_height)))
        consumed.update(evidence_ids)
        components.update(component_ids)
    if consumed != set(observed):
        return dict(status="review_required", reason="selected_observations_outside_subregions", units=[])
    if any(not re.search(r"[.!?][\s\u201d\u2019]*$", unit["source_text"]) for unit in units):
        return dict(status="review_required", reason="independent_utterance_boundary_unproven", units=units)
    return dict(status="candidate_independent_subblocks", reason="distinct_components_and_terminal_utterances",
                owner_id=owner_id, units=units)
