"""Group detected text regions before OCR/translation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .reading_order import order_regions

BBox = list[int]


def _bbox(region: dict[str, Any], key: str = "bbox") -> BBox:
    raw = region.get(key) or region.get("source_bbox") or region.get("bbox") or [0, 0, 0, 0]
    return [int(v) for v in raw[:4]]


def _area(bbox: BBox) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _intersection(a: BBox, b: BBox) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return _area([x1, y1, x2, y2])


def _iou(a: BBox, b: BBox) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0


def _vertical_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return overlap / max(1, min(a[3] - a[1], b[3] - b[1]))


def _horizontal_gap(a: BBox, b: BBox) -> int:
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return 0


def _same_balloon(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ab = a.get("balloon_bbox")
    bb = b.get("balloon_bbox")
    if not ab or not bb:
        return False
    return _iou(_bbox(a, "balloon_bbox"), _bbox(b, "balloon_bbox")) >= 0.72


def _compatible_text_pair(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("tipo") == "sfx" or b.get("tipo") == "sfx":
        return False
    if a.get("tipo") == "narracao" or b.get("tipo") == "narracao":
        return False
    ab = _bbox(a)
    bb = _bbox(b)
    if _same_balloon(a, b):
        return True
    return _horizontal_gap(ab, bb) <= 32 and _vertical_overlap_ratio(ab, bb) >= 0.55


def group_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign group_id/grouping_status and reading_order to detected regions."""

    ordered = order_regions(regions)
    groups: list[list[int]] = []
    for index, region in enumerate(ordered):
        placed = False
        for group in groups:
            if any(_compatible_text_pair(region, ordered[member]) for member in group):
                group.append(index)
                placed = True
                break
        if not placed:
            groups.append([index])

    for group_index, group in enumerate(groups):
        status = "grouped" if len(group) > 1 else "separated"
        group_id = f"g{group_index + 1:03}"
        for member in group:
            ordered[member]["group_id"] = group_id
            ordered[member]["grouping_status"] = status
            ordered[member]["layout_group_size"] = len(group)
    return ordered


def write_debug_overlay(
    image_size: tuple[int, int],
    grouped_regions: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write a lightweight visual overlay for grouping debug."""

    from PIL import Image, ImageDraw

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)
    colors = ["red", "blue", "green", "purple", "orange"]
    for region in grouped_regions:
        bbox = _bbox(region)
        order = region.get("reading_order", "?")
        group_id = str(region.get("group_id", "g?"))
        color = colors[hash(group_id) % len(colors)]
        draw.rectangle(bbox, outline=color, width=2)
        draw.text((bbox[0] + 2, bbox[1] + 2), f"{group_id}:{order}", fill=color)
    img.save(output)
    return output

