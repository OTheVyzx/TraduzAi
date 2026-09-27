"""Source-driven proposals for recovering split or merged dialogue bodies.

This stage does not authorize OCR corrections, translation, cleanup or render.
Its output is a proposal for those downstream gates, with no page/owner IDs in
the decision logic.
"""
from __future__ import annotations

import re

from consumer_connected_preparation import plan_connected_subblocks


def _observations(row: dict) -> list[dict]:
    return list((row.get("owner_render_geometry") or {}).get("selected_observations") or [])


def _box(rows: list[dict]) -> list[int]:
    boxes = [item["bbox_page"] for row in rows for item in _observations(row)]
    if not boxes:
        raise ValueError("selected source observations missing")
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def discover_recovery_proposals(records: list[dict], width: int, height: int,
                                *, source_observations: dict[str, dict] | None = None) -> list[dict]:
    """Find evidence-backed local split/merge candidates from source records.

    A continuation is only proposed; same white body and corrected OCR must
    still be established before it is allowed to alter any pixels.
    """
    if width <= 0 or height <= 0:
        raise ValueError("invalid source dimensions")
    if len({row["owner_id"] for row in records}) != len(records):
        raise ValueError("duplicate source owner")
    if source_observations is not None:
        hydrated = []
        for row in records:
            geometry = dict(row.get("owner_render_geometry") or {})
            selected = []
            for item in geometry.get("selected_observations") or []:
                identity = item["observation_id"]
                if identity not in source_observations:
                    raise ValueError("selected observation absent from source graph")
                selected.append(dict(item, **source_observations[identity]))
            geometry["selected_observations"] = selected
            hydrated.append(dict(row, owner_render_geometry=geometry))
        records = hydrated
    proposals = []
    consumed = set()
    for row in records:
        geometry = row.get("owner_render_geometry") or {}
        regions = geometry.get("connected_subregions") or []
        if len(regions) < 2:
            continue
        plan = plan_connected_subblocks(row["owner_id"], regions,
            _observations(row), width, height)
        if plan["status"] == "candidate_independent_subblocks":
            proposals.append(dict(kind="split_connected_bodies", owner_ids=[row["owner_id"]],
                                  source_anchor_bbox=_box([row]), plan=plan,
                                  requires_local_ocr=any(unit["ocr_recovery_required"]
                                                         for unit in plan["units"])))
            consumed.add(row["owner_id"])
    remaining = sorted((row for row in records if row["owner_id"] not in consumed and
                        _observations(row)), key=lambda row: (_box([row])[1], _box([row])[0]))
    for first, second in zip(remaining, remaining[1:]):
        a, b = _box([first]), _box([second])
        source = str(first.get("source_payload") or "").strip()
        if (not source or re.search(r"[.!?][\s\u201d\u2019]*$", source) or
                first.get("semantic_role") != second.get("semantic_role")):
            continue
        line_heights = [box[3]-box[1] for row in (first, second)
                        for box in (item["bbox_page"] for item in _observations(row))]
        if not line_heights:
            continue
        median = sorted(line_heights)[len(line_heights)//2]
        gap = b[1]-a[3]
        center_delta = abs((a[0]+a[2])-(b[0]+b[2]))/2
        if (gap < -.5*median or gap > 1.5*median or
                center_delta > .35*max(a[2]-a[0], b[2]-b[0])):
            continue
        proposals.append(dict(kind="possible_fragmented_body",
                              owner_ids=[first["owner_id"], second["owner_id"]],
                              source_anchor_bbox=_box([first, second]),
                              source_observation_ids=[item["observation_id"]
                                  for row in (first, second) for item in _observations(row)],
                              reason="unterminated_first_fragment_and_spatial_continuation",
                              requires_same_source_white_body=True,
                              requires_local_ocr=True))
    return proposals
