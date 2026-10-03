"""Opt-in, page-bound regional OCR planning with no cut through an input box."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class GroupedCropRefused(ValueError):
    """A bounded regional plan cannot preserve the original boxes safely."""


@dataclass(frozen=True)
class Region:
    region_id: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Crop:
    box: tuple[int, int, int, int]
    region_ids: tuple[str, ...]


def _near(a: Region, b: Region, gap: int) -> bool:
    x1, y1, x2, y2 = a.box
    u1, v1, u2, v2 = b.box
    return x1 <= u2 + gap and u1 <= x2 + gap and y1 <= v2 + gap and v1 <= y2 + gap


def plan_grouped_crops(
    page_rgb: np.ndarray,
    regions: list[Region],
    *,
    context_px: int = 12,
    group_gap_px: int = 24,
    max_height_px: int = 1600,
    max_pixels: int = 2_000_000,
) -> tuple[Crop, ...]:
    """Group nearby source boxes, add context once, and split only on safe rows.

    Coordinates always refer to the original page. A refusal lets the caller
    retain its existing OCR route; no shortened crop is silently substituted.
    """
    if page_rgb.ndim != 3 or page_rgb.shape[2] < 3:
        raise GroupedCropRefused("expected RGB page")
    height, width = page_rgb.shape[:2]
    if min(context_px, group_gap_px) < 0 or min(max_height_px, max_pixels) <= 0:
        raise GroupedCropRefused("invalid crop budget")
    ids = [r.region_id for r in regions]
    if len(ids) != len(set(ids)):
        raise GroupedCropRefused("duplicate region identity")
    for r in regions:
        x1, y1, x2, y2 = r.box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise GroupedCropRefused("region outside source page")
    if not regions:
        return ()

    parent = list(range(len(regions)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, region in enumerate(regions):
        for other in range(index):
            left, right = root(index), root(other)
            if left != right and _near(region, regions[other], group_gap_px):
                parent[left] = right
    groups: dict[int, list[Region]] = {}
    for index, region in enumerate(regions):
        groups.setdefault(root(index), []).append(region)

    crops: list[Crop] = []
    for members in groups.values():
        x1 = max(0, min(r.box[0] for r in members) - context_px)
        y1 = max(0, min(r.box[1] for r in members) - context_px)
        x2 = min(width, max(r.box[2] for r in members) + context_px)
        y2 = min(height, max(r.box[3] for r in members) + context_px)
        span = x2 - x1
        limit = min(max_height_px, max_pixels // span)
        if limit <= 0 or any(r.box[3] - r.box[1] > limit for r in members):
            raise GroupedCropRefused("region exceeds bounded OCR crop")
        start = y1
        while start < y2:
            maximum = min(y2, start + limit)
            if maximum == y2:
                end = y2
            else:
                rows = (
                    row for row in range(start + 1, maximum + 1)
                    if all(not (r.box[1] < row < r.box[3]) for r in members)
                )
                candidates = list(rows)
                if not candidates:
                    raise GroupedCropRefused("no source-safe bounded cut")

                def ink_score(row: int) -> tuple[float, int]:
                    pixels = page_rgb[max(0, row - 1):min(height, row + 1), x1:x2, :3]
                    dark = np.mean(pixels, axis=2) < 190
                    return float(np.mean(dark)), maximum - row

                end = min(candidates, key=ink_score)
            included = tuple(
                r.region_id for r in members
                if r.box[1] >= start and r.box[3] <= end
            )
            if included:
                crops.append(Crop((x1, start, x2, end), included))
            if end == y2:
                break
            start = end
    covered = [region_id for crop in crops for region_id in crop.region_ids]
    if len(covered) != len(regions) or set(covered) != set(ids):
        raise GroupedCropRefused("source region lost or duplicated")
    return tuple(sorted(crops, key=lambda crop: (crop.box[1], crop.box[0])))
