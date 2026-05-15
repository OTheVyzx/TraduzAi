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


def bbox_to_octagon_polygon(bbox: list[int], cut_ratio: float = 0.18) -> list[list[int]]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    max_cut = max(0, (min(width, height) - 1) // 2)
    cut = min(max_cut, max(0, int(round(min(width, height) * cut_ratio))))
    if cut <= 0:
        return _bbox_to_polygon(bbox)
    right = x2 - 1
    bottom = y2 - 1
    return [
        [x1 + cut, y1],
        [right - cut, y1],
        [right, y1 + cut],
        [right, bottom - cut],
        [right - cut, bottom],
        [x1 + cut, bottom],
        [x1, bottom - cut],
        [x1, y1 + cut],
    ]


def bbox_to_octagon_mask(
    width: int,
    height: int,
    bbox: list[int],
    padding: int = 0,
    cut_ratio: float = 0.18,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    normalized = _normalize_bbox(
        [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
        width,
        height,
    )
    if normalized is None:
        return mask
    cv2.fillPoly(
        mask,
        [np.asarray(bbox_to_octagon_polygon(normalized, cut_ratio=cut_ratio), dtype=np.int32)],
        255,
    )
    return mask


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


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _candidate_text_search_mask(block: dict, width: int, height: int) -> tuple[np.ndarray, list[int]] | None:
    candidate = np.zeros((height, width), dtype=np.uint8)
    polygons = _normalize_polygons(block.get("line_polygons"), width, height)
    if polygons:
        for polygon in polygons:
            cv2.fillPoly(candidate, [np.asarray(polygon, dtype=np.int32)], 255)
        bbox = _bbox_from_mask(candidate)
        if bbox:
            return candidate, bbox

    bbox = (
        _normalize_bbox(block.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(block.get("bbox"), width, height)
    )
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    candidate[y1:y2, x1:x2] = 255
    return candidate, bbox


def _has_explicit_text_geometry(block: dict, width: int, height: int) -> bool:
    if _normalize_polygons(block.get("line_polygons"), width, height):
        return True
    return _normalize_bbox(block.get("text_pixel_bbox"), width, height) is not None


def _add_broad_bbox_search_candidate(
    block: dict,
    width: int,
    height: int,
    candidate_infos: list[tuple[np.ndarray, list[int]]],
) -> None:
    bbox = _normalize_bbox(block.get("bbox"), width, height)
    if not bbox:
        return
    if candidate_infos:
        existing = np.zeros((height, width), dtype=np.uint8)
        for candidate_mask, _ in candidate_infos:
            existing = np.maximum(existing, candidate_mask)
        existing_bbox = _bbox_from_mask(existing)
        if existing_bbox:
            bx1, by1, bx2, by2 = bbox
            ex1, ey1, ex2, ey2 = existing_bbox
            bbox_h = max(1, by2 - by1)
            existing_h = max(1, ey2 - ey1)
            bbox_w = max(1, bx2 - bx1)
            existing_w = max(1, ex2 - ex1)
            if bbox_h < existing_h * 1.65 and bbox_w < existing_w * 1.65:
                return
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(_bbox_to_polygon(bbox), dtype=np.int32)], 255)
    candidate_infos.append((mask, bbox))


def _odd_kernel_size(value: int, maximum: int = 31) -> int:
    value = max(3, min(maximum, int(value)))
    return value if value % 2 else value - 1


def _raw_text_search_expand_px(block: dict) -> int:
    style = block.get("estilo") if isinstance(block.get("estilo"), dict) else {}
    profile = str(block.get("layout_profile") or block.get("block_profile") or "").strip().lower()
    balloon_type = str(block.get("balloon_type") or "").strip().lower()
    expand = 0
    if style.get("italico") or profile in {"top_narration", "sfx"} or balloon_type in {"textured", "colored", "dark"}:
        expand = 7
    font_size = _font_size_from_block(block)
    if font_size is not None and font_size >= 40:
        expand = max(expand, 6)
    return expand


def build_raw_text_mask_from_image(
    block: dict,
    image_rgb: np.ndarray,
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    height, width = _image_hw(image_shape)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.shape[0] < height or image_rgb.shape[1] < width:
        return None

    candidate_infos: list[tuple[np.ndarray, list[int]]] = []
    polygons = _normalize_polygons(block.get("line_polygons"), width, height)
    has_explicit_geometry = bool(polygons) or _normalize_bbox(block.get("text_pixel_bbox"), width, height) is not None
    allow_broad_bbox_search = bool(block.get("allow_broad_bbox_text_search"))
    if polygons:
        for polygon in polygons:
            candidate = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(candidate, [np.asarray(polygon, dtype=np.int32)], 255)
            bbox = _bbox_from_mask(candidate)
            if bbox:
                candidate_infos.append((candidate, bbox))
        if allow_broad_bbox_search:
            _add_broad_bbox_search_candidate(block, width, height, candidate_infos)
    else:
        candidate_info = _candidate_text_search_mask(block, width, height)
        if candidate_info is not None:
            candidate_infos.append(candidate_info)
        if allow_broad_bbox_search or not has_explicit_geometry:
            _add_broad_bbox_search_candidate(block, width, height, candidate_infos)

    if not candidate_infos:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    search_expand_px = _raw_text_search_expand_px(block)
    search_kernel = None
    if search_expand_px > 0:
        search_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (search_expand_px * 2 + 1, search_expand_px * 2 + 1),
        )

    for candidate_mask, bbox in candidate_infos:
        if search_kernel is not None:
            candidate_mask = cv2.dilate(candidate_mask, search_kernel, iterations=1)
            expanded_bbox = _bbox_from_mask(candidate_mask)
            if expanded_bbox:
                bbox = expanded_bbox
        candidate_area = int(np.count_nonzero(candidate_mask))
        if candidate_area < 8:
            continue

        x1, y1, x2, y2 = bbox
        pad = 2
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = image_rgb[y1:y2, x1:x2]
        roi_candidate = candidate_mask[y1:y2, x1:x2] > 0
        if roi.size == 0 or int(np.count_nonzero(roi_candidate)) < 8:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi.astype(np.uint8)
        kernel_size = _odd_kernel_size(max(7, min(gray.shape[:2]) // 2), maximum=31)
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0).astype(np.float32)
        gray_f = gray.astype(np.float32)
        candidate_gray = gray_f[roi_candidate]
        if candidate_gray.size == 0:
            continue
        low_luma = float(np.percentile(candidate_gray, 8))
        median_luma = float(np.median(candidate_gray))
        high_luma = float(np.percentile(candidate_gray, 92))
        dark_span = median_luma - low_luma
        light_span = high_luma - median_luma
        dark_cutoff = min(median_luma - max(16.0, dark_span * 0.35), low_luma + 22.0)
        light_cutoff = max(median_luma + max(16.0, light_span * 0.35), high_luma - 22.0)

        side_candidates: list[tuple[float, np.ndarray]] = []
        if dark_span >= 24.0:
            dark_like = gray_f <= dark_cutoff
            dark_area = int(np.count_nonzero(dark_like & roi_candidate))
            dark_ratio = dark_area / max(1, candidate_area)
            if 0.001 <= dark_ratio <= 0.45:
                side_candidates.append((dark_ratio, dark_like))
        if light_span >= 24.0:
            light_like = gray_f >= light_cutoff
            light_area = int(np.count_nonzero(light_like & roi_candidate))
            light_ratio = light_area / max(1, candidate_area)
            if 0.001 <= light_ratio <= 0.45:
                side_candidates.append((light_ratio, light_like))

        text_like = np.zeros_like(gray_f, dtype=bool)
        if side_candidates:
            _, text_like = min(side_candidates, key=lambda item: item[0])

        if not np.any(text_like & roi_candidate):
            dark_contrast = background - gray_f
            light_contrast = gray_f - background
            candidate_values = dark_contrast[roi_candidate]
            light_values = light_contrast[roi_candidate]
            if candidate_values.size == 0 or light_values.size == 0:
                continue
            threshold = max(18.0, min(50.0, float(np.percentile(candidate_values, 95)) * 0.5))
            light_threshold = max(18.0, min(50.0, float(np.percentile(light_values, 95)) * 0.5))
            text_like = (dark_contrast >= threshold) | (light_contrast >= light_threshold)
        raw_roi = (text_like & roi_candidate).astype(np.uint8) * 255
        if not np.any(raw_roi):
            continue

        cleaned = np.zeros_like(raw_roi)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((raw_roi > 0).astype(np.uint8), connectivity=8)
        min_area = max(2, int(candidate_area * 0.0005))
        max_component_area = max(8, int(candidate_area * 0.45))
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if min_area <= area <= max_component_area:
                cleaned[labels == label] = 255
        if not np.any(cleaned):
            continue

        raw_area = int(np.count_nonzero(cleaned))
        if raw_area < max(8, int(candidate_area * 0.001)):
            continue
        if raw_area > int(candidate_area * 0.45):
            continue
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], cleaned)

    return mask if np.any(mask) else None


def expand_text_mask(mask: np.ndarray, expand_px: int = 5) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        return mask
    if expand_px <= 0 or not np.any(mask):
        return mask.copy()
    kernel_size = int(expand_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def clip_text_mask_to_balloon_interior(
    text_mask: np.ndarray,
    balloon_mask: np.ndarray | None,
    erode_px: int = 2,
) -> np.ndarray:
    if not isinstance(balloon_mask, np.ndarray) or not np.any(balloon_mask):
        return text_mask
    if erode_px > 0:
        kernel_size = int(erode_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        interior = cv2.erode(balloon_mask.astype(np.uint8), kernel, iterations=1)
    else:
        interior = balloon_mask.astype(np.uint8)
    if not np.any(interior):
        interior = balloon_mask.astype(np.uint8)
    clipped = cv2.bitwise_and(text_mask.astype(np.uint8), interior)
    return clipped if np.any(clipped) else text_mask


def _text_geometry_bbox(block: dict, image_shape: tuple[int, ...]) -> list[int] | None:
    height, width = _image_hw(image_shape)
    polygons = _normalize_polygons(block.get("line_polygons"), width, height)
    if polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        bbox = _bbox_from_mask(mask)
        if bbox:
            return bbox
    return _normalize_bbox(block.get("text_pixel_bbox"), width, height)


def _expand_bbox_px(bbox: list[int], width: int, height: int, pad_x: int, pad_y: int) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    return _normalize_bbox([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def _text_geometry_has_white_context(
    block: dict,
    image_rgb: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> bool:
    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return False
    height, width = _image_hw(image_shape)
    if image_rgb.shape[0] < height or image_rgb.shape[1] < width:
        return False
    geometry_bbox = _text_geometry_bbox(block, image_shape)
    if not geometry_bbox:
        return False
    expanded = _expand_bbox_px(geometry_bbox, width, height, pad_x=8, pad_y=8)
    if not expanded:
        return False
    x1, y1, x2, y2 = expanded
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.astype(np.uint8)
    if crop.ndim == 3:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        bright = (gray >= 220) & (value >= 220) & (saturation <= 70)
    else:
        bright = gray >= 220
    bright_ratio = float(np.mean(bright)) if bright.size else 0.0
    if bright_ratio < 0.48:
        return False
    bright_pixels = gray[bright]
    if bright_pixels.size < 24:
        return False
    return float(np.percentile(bright_pixels, 70)) >= 228.0


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
            fallback_bbox = _normalize_bbox(block.get("bbox"), width, height)
            if bbox and fallback_bbox:
                bx1, by1, bx2, by2 = bbox
                fx1, fy1, fx2, fy2 = fallback_bbox
                bbox_area = max(1, (bx2 - bx1) * (by2 - by1))
                fallback_area = max(1, (fx2 - fx1) * (fy2 - fy1))
                fallback_contains_bbox = fx1 <= bx1 and fy1 <= by1 and fx2 >= bx2 and fy2 >= by2
                if fallback_contains_bbox and bbox_area < int(fallback_area * 0.65):
                    bbox = fallback_bbox
            elif fallback_bbox:
                bbox = fallback_bbox
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
    mask = bbox_to_octagon_mask(width, height, bbox, padding=pad)
    return mask if np.any(mask) else None


def build_inpaint_mask(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None = None,
) -> np.ndarray | None:
    text_mask = None
    height, width = _image_hw(image_shape)
    has_explicit_geometry = _has_explicit_text_geometry(block, width, height)
    geometry_mask = mask_from_text_geometry(block, image_shape) if has_explicit_geometry else None
    white_text_context = _text_geometry_has_white_context(block, image_rgb, image_shape)
    if image_rgb is not None and image_rgb.size:
        raw_text_mask = build_raw_text_mask_from_image(block, image_rgb, image_shape)
        if raw_text_mask is not None and np.any(raw_text_mask):
            text_mask = expand_text_mask(raw_text_mask, expand_px=5)
            if geometry_mask is not None and np.any(geometry_mask):
                raw_bbox = _bbox_from_mask(text_mask)
                geometry_bbox = _bbox_from_mask(geometry_mask)
                raw_is_sparse = False
                if raw_bbox and geometry_bbox:
                    raw_w = max(1, raw_bbox[2] - raw_bbox[0])
                    raw_h = max(1, raw_bbox[3] - raw_bbox[1])
                    geometry_w = max(1, geometry_bbox[2] - geometry_bbox[0])
                    geometry_h = max(1, geometry_bbox[3] - geometry_bbox[1])
                    raw_is_sparse = raw_w < int(geometry_w * 0.55) or raw_h < int(geometry_h * 0.55)
                if white_text_context or raw_is_sparse:
                    text_mask = np.maximum(text_mask.astype(np.uint8), geometry_mask.astype(np.uint8))
    if text_mask is None or not np.any(text_mask):
        text_mask = geometry_mask if geometry_mask is not None and np.any(geometry_mask) else mask_from_text_geometry(block, image_shape)
    if text_mask is None or not np.any(text_mask):
        return None

    balloon_mask = balloon_mask_from_block(block, image_shape)
    if balloon_mask is None or not np.any(balloon_mask):
        return text_mask

    if image_rgb is not None and image_rgb.size:
        text_mask = clip_text_mask_to_balloon_interior(text_mask, balloon_mask, erode_px=2)

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
        mask = np.maximum(mask, bbox_to_octagon_mask(width, height, [x1, y1, x2, y2]))

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    x1, y1, x2, y2 = region["bbox"]
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped
