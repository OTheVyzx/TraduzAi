"""Geometry-owned logical units and physical subblocks for visual analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


def _bbox(value: Any, field: str) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        result = ()
    if len(result) != 4 or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{field} must be a non-empty bbox")
    return result


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _contains_center(
    container: tuple[int, int, int, int], observation: tuple[int, int, int, int]
) -> bool:
    x, y = _center(observation)
    return container[0] <= x <= container[2] and container[1] <= y <= container[3]


def _area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _union(boxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    )


def _stable_id(prefix: str, payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:20]}"


@dataclass(frozen=True)
class StructuralAnalysis:
    logical_units: tuple[dict[str, Any], ...]
    physical_subblocks: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]
    reading_order: tuple[str, ...]


def build_structural_analysis(
    *,
    observations: Sequence[Mapping[str, Any]],
    containers: Sequence[Mapping[str, Any]],
) -> StructuralAnalysis:
    """Group OCR by supported container geometry, never by IDs or proximity alone."""

    normalized_observations = [
        {
            "observation_id": str(row.get("observation_id") or ""),
            "text": str(row.get("text") or ""),
            "bbox_page": _bbox(row.get("bbox_page"), "observation bbox"),
        }
        for row in observations
    ]
    normalized_containers = []
    for row in containers:
        normalized_containers.append({
            "container_id": str(row.get("container_id") or ""),
            "bbox_page": _bbox(row.get("bbox_page"), "container bbox"),
            "kind": str(row.get("kind") or "unknown_container"),
            "lobe_bboxes": tuple(
                _bbox(value, "lobe bbox") for value in row.get("lobe_bboxes", ())
            ),
        })

    grouped: dict[str | None, list[dict[str, Any]]] = {
        row["container_id"]: [] for row in normalized_containers
    }
    container_by_id = {row["container_id"]: row for row in normalized_containers}
    uncontained: list[dict[str, Any]] = []
    for observation in normalized_observations:
        candidates = [
            container for container in normalized_containers
            if _contains_center(container["bbox_page"], observation["bbox_page"])
        ]
        if not candidates:
            uncontained.append(observation)
            continue
        chosen = sorted(candidates, key=lambda item: (
            _area(item["bbox_page"]), item["bbox_page"][1], item["bbox_page"][0],
        ))[0]
        grouped[chosen["container_id"]].append(observation)

    pending: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for container_id, members in grouped.items():
        if members:
            pending.append((container_by_id[container_id], members))
    for observation in uncontained:
        pending.append(({
            "container_id": None,
            "bbox_page": observation["bbox_page"],
            "kind": "unknown_text",
            "lobe_bboxes": (),
        }, [observation]))
    pending.sort(key=lambda item: (
        _union([row["bbox_page"] for row in item[1]])[1],
        _union([row["bbox_page"] for row in item[1]])[0],
    ))

    logical_units: list[dict[str, Any]] = []
    subblocks: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for container, members in pending:
        ordered = sorted(members, key=lambda row: (
            row["bbox_page"][1], row["bbox_page"][0],
            row["bbox_page"][3], row["bbox_page"][2],
        ))
        observation_ids = [row["observation_id"] for row in ordered]
        unit_bbox = _union([row["bbox_page"] for row in ordered])
        unit_id = _stable_id("logical_unit", {
            "container_ref": container["container_id"],
            "bbox_page": unit_bbox,
            "observation_ids": observation_ids,
        })
        logical_units.append({
            "logical_unit_id": unit_id,
            "container_ref": container["container_id"],
            "classification": (
                "unknown_text" if container["container_id"] is None
                else container["kind"]
            ),
            "bbox_page": list(unit_bbox),
            "observation_ids": observation_ids,
            "text": " ".join(row["text"].strip() for row in ordered if row["text"].strip()),
            "uncertainty_reasons": (
                ["container_not_observed"] if container["container_id"] is None else []
            ),
        })

        lobe_bboxes = tuple(container["lobe_bboxes"])
        if not lobe_bboxes:
            lobe_bboxes = (unit_bbox,)
        ordered_lobes = sorted(lobe_bboxes, key=lambda box: (box[1], box[0], box[3], box[2]))
        lobe_assignment: dict[str, int] = {}
        for row in ordered:
            candidates = [
                index for index, box in enumerate(ordered_lobes)
                if _contains_center(box, row["bbox_page"])
            ]
            if candidates:
                observation_center = _center(row["bbox_page"])
                lobe_assignment[row["observation_id"]] = min(
                    candidates,
                    key=lambda index: (
                        (_center(ordered_lobes[index])[0] - observation_center[0]) ** 2
                        + (_center(ordered_lobes[index])[1] - observation_center[1]) ** 2,
                        index,
                    ),
                )
        for order, lobe_bbox in enumerate(ordered_lobes):
            lobe_members = [
                row for row in ordered
                if lobe_assignment.get(row["observation_id"]) == order
            ]
            block_id = _stable_id("physical_subblock", {
                "logical_unit_id": unit_id,
                "bbox_page": lobe_bbox,
                "order": order,
            })
            subblocks.append({
                "physical_subblock_id": block_id,
                "logical_unit_id": unit_id,
                "bbox_page": list(lobe_bbox),
                "observation_ids": [row["observation_id"] for row in lobe_members],
                "order": order,
                "uncertainty_reasons": (
                    [] if lobe_members else ["lobe_has_no_selected_observation"]
                ),
            })
            relations.append({
                "kind": "physical_subblock_of",
                "source_id": block_id,
                "target_id": unit_id,
            })

    return StructuralAnalysis(
        logical_units=tuple(logical_units),
        physical_subblocks=tuple(subblocks),
        relations=tuple(relations),
        reading_order=tuple(row["logical_unit_id"] for row in logical_units),
    )
