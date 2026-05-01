"""Reading-order helpers for detected manga text regions."""

from __future__ import annotations

from typing import Any

BBox = list[int]


def _bbox(region: dict[str, Any]) -> BBox:
    raw = region.get("bbox") or region.get("source_bbox") or region.get("balloon_bbox") or [0, 0, 0, 0]
    return [int(v) for v in raw[:4]]


def _center_y(region: dict[str, Any]) -> float:
    x1, y1, x2, y2 = _bbox(region)
    del x1, x2
    return (y1 + y2) / 2.0


def _row_key(region: dict[str, Any], row_tolerance: int = 48) -> int:
    return int(_center_y(region) // max(1, row_tolerance))


def order_regions(regions: list[dict[str, Any]], *, direction: str = "manga") -> list[dict[str, Any]]:
    """Return regions with stable reading_order.

    Default is manga-style rows: top-to-bottom, right-to-left within each row.
    Narration and SFX keep the same geometric rule, but region type is preserved
    so downstream stages can style them differently.
    """

    def key(region: dict[str, Any]) -> tuple[int, int, int]:
        x1, y1, x2, _ = _bbox(region)
        row = _row_key(region)
        horizontal = -x2 if direction == "manga" else x1
        return (row, horizontal, y1)

    ordered = [dict(region) for region in sorted(regions, key=key)]
    for index, region in enumerate(ordered):
        region["reading_order"] = index
    return ordered

