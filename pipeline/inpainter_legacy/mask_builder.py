"""
Builds expanded mask regions from OCR boxes.
This is the first step toward page-level inpainting that respects balloons
instead of removing each OCR box in isolation.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np


def build_mask_regions(texts: list[dict], image_shape: tuple[int, int, int]) -> list[dict]:
    image_height, image_width = image_shape[:2]
    seeds = []

    for text in texts:
        if text.get("skip_processing"):
            continue
        bbox = text.get("bbox")
        if not bbox:
            continue
        seeds.append(
            {
                "bbox": expand_bbox(
                    bbox=bbox,
                    image_width=image_width,
                    image_height=image_height,
                    tipo=text.get("tipo", "fala"),
                    confidence=float(text.get("confidence", 0.0)),
                ),
                "tipo": text.get("tipo", "fala"),
                "text": text,
            }
        )

    clusters: list[dict] = []
    for seed in seeds:
        merged = False
        for cluster in clusters:
            if should_merge(cluster["bbox"], seed["bbox"]):
                cluster["bbox"] = union_bbox(cluster["bbox"], seed["bbox"])
                cluster["texts"].append(seed["text"])
                cluster["tipos"].append(seed["tipo"])
                merged = True
                break
        if not merged:
            clusters.append(
                {
                    "bbox": seed["bbox"],
                    "texts": [seed["text"]],
                    "tipos": [seed["tipo"]],
                }
            )

    regions = []
    for cluster in clusters:
        regions.append(
            {
                "bbox": cluster["bbox"],
                "texts": cluster["texts"],
                "tipo": Counter(cluster["tipos"]).most_common(1)[0][0],
                "kind": "cluster" if len(cluster["texts"]) > 1 else "single",
            }
        )
    return regions


def _is_vertical_text(bbox: list[int]) -> bool:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return h > w * 2.5


def expand_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    tipo: str = "fala",
    confidence: float = 1.0,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    if _is_vertical_text(bbox):
        # Texto vertical: máscara estreita horizontalmente, justa verticalmente
        margin_x = max(3, int(width * 0.10))
        margin_y = max(3, int(height * 0.08))
    elif tipo == "narracao":
        margin_x = max(4, int(width * 0.16))
        margin_y = max(4, int(height * 0.36))
    elif tipo == "sfx":
        margin_x = max(4, int(width * 0.18))
        margin_y = max(4, int(height * 0.20))
    else:
        margin_x = max(4, int(width * 0.12))
        margin_y = max(4, int(height * 0.30))

    if confidence < 0.65:
        margin_x += 3
        margin_y += 3

    return [
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(image_width, x2 + margin_x),
        min(image_height, y2 + margin_y),
    ]


def should_merge(a: list[int], b: list[int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    horizontal_gap = max(0, max(ax1, bx1) - min(ax2, bx2), max(bx1, ax1) - min(bx2, ax2))
    vertical_gap = max(0, max(ay1, by1) - min(ay2, by2), max(by1, ay1) - min(by2, ay2))
    width = min(ax2 - ax1, bx2 - bx1)
    height = min(ay2 - ay1, by2 - by1)
    overlaps = not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    if overlaps:
        return True

    return horizontal_gap <= max(12, int(width * 0.25)) and vertical_gap <= max(18, int(height * 0.45))


def union_bbox(a: list[int], b: list[int]) -> list[int]:
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def build_region_pixel_mask(image_shape: tuple[int, int], region: dict) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for text in region.get("texts", []):
        bbox = text.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = expand_bbox(
            bbox=bbox,
            image_width=width,
            image_height=height,
            tipo=text.get("tipo", region.get("tipo", "fala")),
            confidence=float(text.get("confidence", 0.0)),
        )
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    x1, y1, x2, y2 = region["bbox"]
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped
