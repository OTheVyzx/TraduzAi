"""
Builds expanded mask regions from OCR boxes.
This is the first step toward page-level inpainting that respects balloons
instead of removing each OCR box in isolation.
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np


def _image_hw(image_shape: tuple[int, ...]) -> tuple[int, int]:
    return int(image_shape[0]), int(image_shape[1])


def _normalize_bbox(value, width: int, height: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _normalize_polygon(value, width: int, height: int) -> list[list[int]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    points: list[list[int]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
        except Exception:
            return None
        points.append([max(0, min(width - 1, x)), max(0, min(height - 1, y))])
    return points if len(points) >= 3 else None


def _normalize_polygons(value, width: int, height: int) -> list[list[list[int]]]:
    if not isinstance(value, (list, tuple)) or not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2 and not (
        first and isinstance(first[0], (list, tuple))
    ):
        polygon = _normalize_polygon(value, width, height)
        return [polygon] if polygon else []

    polygons: list[list[list[int]]] = []
    for item in value:
        polygon = _normalize_polygon(item, width, height)
        if polygon:
            polygons.append(polygon)
    return polygons


def _bbox_to_polygon(bbox: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]]


def _font_size_from_block(block: dict) -> int | None:
    for key in ("font_size_px", "font_size", "tamanho_fonte"):
        value = block.get(key)
        try:
            if value is not None:
                return int(round(float(value)))
        except Exception:
            pass
    estilo = block.get("estilo")
    if isinstance(estilo, dict):
        for key in ("tamanho", "font_size", "font_size_px"):
            try:
                value = estilo.get(key)
                if value is not None:
                    return int(round(float(value)))
            except Exception:
                pass
    return None


def glyph_padding(font_size_px: int | float | None) -> int:
    try:
        font_size = int(round(float(font_size_px)))
    except Exception:
        font_size = 16
    return max(3, int(font_size * 0.06))


def polygon_to_mask(points, image_shape: tuple[int, ...]) -> np.ndarray:
    height, width = _image_hw(image_shape)
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = _normalize_polygon(points, width, height)
    if not polygon:
        return mask
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return mask


def balloon_mask_from_block(block: dict, image_shape: tuple[int, ...]) -> np.ndarray | None:
    height, width = _image_hw(image_shape)
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = _normalize_polygons(block.get("balloon_polygon"), width, height)
    polygons.extend(_normalize_polygons(block.get("connected_lobe_polygons"), width, height))
    if not polygons:
        for key in ("balloon_subregions", "connected_lobe_bboxes"):
            for bbox_value in block.get(key) or []:
                bbox = _normalize_bbox(bbox_value, width, height)
                if bbox:
                    cv2.fillPoly(mask, [np.asarray(_bbox_to_polygon(bbox), dtype=np.int32)], 255)
        if not np.any(mask):
            bbox = _normalize_bbox(block.get("balloon_bbox"), width, height)
            if bbox:
                cv2.fillPoly(mask, [np.asarray(_bbox_to_polygon(bbox), dtype=np.int32)], 255)
    else:
        for polygon in polygons:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return mask if np.any(mask) else None


def mask_from_text_geometry(block: dict, image_shape: tuple[int, ...]) -> np.ndarray | None:
    height, width = _image_hw(image_shape)
    mask = np.zeros((height, width), dtype=np.uint8)
    pad = glyph_padding(_font_size_from_block(block))
    polygons = _normalize_polygons(block.get("line_polygons"), width, height)
    if polygons:
        for polygon in polygons:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        if np.any(mask):
            kernel_size = max(3, pad * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            return cv2.dilate(mask, kernel, iterations=1)

    bbox = (
        _normalize_bbox(block.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(block.get("bbox"), width, height)
    )
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    mask[y1:y2, x1:x2] = 255
    return mask


def build_inpaint_mask(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None = None,
) -> np.ndarray | None:
    text_mask = mask_from_text_geometry(block, image_shape)
    if text_mask is None or not np.any(text_mask):
        return None

    balloon_mask = balloon_mask_from_block(block, image_shape)
    if balloon_mask is None or not np.any(balloon_mask):
        return text_mask

    interior = cv2.erode(
        balloon_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    clipped = cv2.bitwise_and(text_mask, interior if np.any(interior) else balloon_mask)
    if not np.any(clipped):
        clipped = cv2.bitwise_and(text_mask, balloon_mask)
    if not np.any(clipped):
        return text_mask

    if image_rgb is not None and image_rgb.size:
        outline_band = cv2.subtract(balloon_mask, interior)
        if np.any(outline_band):
            clipped = cv2.bitwise_and(clipped, cv2.bitwise_not(outline_band))
    return clipped


def build_mask_regions(texts: list[dict], image_shape: tuple[int, int, int]) -> list[dict]:
    image_height, image_width = image_shape[:2]
    seeds = []

    for text in texts:
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
                    estilo=text.get("estilo", {}),
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

    # Second pass: merge clusters that now overlap after growth.
    # The greedy single-pass above is order-dependent — a block arriving
    # before its eventual neighbours can start its own cluster that never
    # gets re-checked.  This convergence loop fixes that.
    changed = True
    while changed:
        changed = False
        new_clusters: list[dict] = []
        skip: set[int] = set()
        for i in range(len(clusters)):
            if i in skip:
                continue
            current = clusters[i]
            for j in range(i + 1, len(clusters)):
                if j in skip:
                    continue
                if should_merge(current["bbox"], clusters[j]["bbox"]):
                    current["bbox"] = union_bbox(current["bbox"], clusters[j]["bbox"])
                    current["texts"].extend(clusters[j]["texts"])
                    current["tipos"].extend(clusters[j]["tipos"])
                    skip.add(j)
                    changed = True
            new_clusters.append(current)
        clusters = new_clusters

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
    estilo: dict | None = None,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    if _is_vertical_text(bbox):
        margin_x = max(2, int(width * 0.08))
        margin_y = max(2, int(height * 0.06))
    elif tipo == "narracao":
        margin_x = max(3, int(width * 0.12))
        margin_y = max(3, int(height * 0.18))
    elif tipo == "sfx":
        margin_x = max(4, int(width * 0.15))
        margin_y = max(4, int(height * 0.15))
    else:
        # Texto normal (fala): margens reduzidas para evitar comer spikes e bordas
        margin_x = max(4, int(width * 0.10))
        margin_y = max(4, int(height * 0.12))

    if confidence < 0.65:
        margin_x += 2
        margin_y += 2

    if estilo:
        contorno_px = int(estilo.get("contorno_px", 0))
        glow_px = int(estilo.get("glow_px", 0))
        sombra = estilo.get("sombra_offset", [0, 0])
        sombra_w = max(0, abs(int(sombra[0])))
        sombra_h = max(0, abs(int(sombra[1])))
        
        extra_x = contorno_px + glow_px + sombra_w
        extra_y = contorno_px + glow_px + sombra_h
        
        margin_x += min(18, extra_x)
        margin_y += min(18, extra_y)

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

    # Se os bboxes se sobrepõem, merge (estão no mesmo balão)
    if overlaps:
        return True

    # Gap moderado: próximos o bastante para ser o mesmo balão,
    # mas não tão largo que junte balões separados
    return horizontal_gap <= max(8, int(width * 0.15)) and vertical_gap <= max(12, int(height * 0.25))


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
            estilo=text.get("estilo", {}),
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
