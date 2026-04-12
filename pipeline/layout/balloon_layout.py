"""
Infer layout regions for text balloons/narration areas from OCR clusters.
"""

from __future__ import annotations

import cv2
import numpy as np

from inpainter.mask_builder import build_mask_regions


def enrich_page_layout(page_result: dict) -> dict:
    texts = page_result.get("texts", [])
    width = int(page_result.get("width", 0) or 0)
    height = int(page_result.get("height", 0) or 0)
    if not texts or width <= 0 or height <= 0:
        return page_result

    regions = build_mask_regions(texts=texts, image_shape=(height, width, 3))
    page_image = _load_page_image(page_result)
    enriched_texts = []
    subregion_cache: dict[tuple, list[list[int]]] = {}

    for text in texts:
        region = _find_region_for_text(text, regions)
        use_shared_layout = bool(region) and _region_supports_shared_layout(region, text.get("tipo", "fala"))
        inferred_bbox = region["bbox"] if use_shared_layout and region else text.get("bbox", [0, 0, 0, 0])
        balloon_bbox = (
            refine_balloon_bbox_from_image(page_image, inferred_bbox, text.get("tipo", "fala"))
            if page_image is not None
            else inferred_bbox
        )
        layout_shape = classify_layout_shape(
            balloon_bbox,
            text.get("tipo", "fala"),
            region,
        )
        layout_align = classify_layout_align(text.get("tipo", "fala"), layout_shape)
        updated = dict(text)
        updated["balloon_bbox"] = balloon_bbox
        updated["layout_shape"] = layout_shape
        updated["layout_align"] = layout_align
        updated["layout_group_size"] = len(region["texts"]) if use_shared_layout and region else 1
        subregion_key = (
            tuple(int(v) for v in inferred_bbox),
            tuple(int(v) for v in balloon_bbox),
            str(text.get("tipo", "fala")),
        )
        if subregion_key not in subregion_cache:
            subregion_cache[subregion_key] = _detect_connected_balloon_subregions(
                page_image,
                inferred_bbox,
                balloon_bbox,
                text.get("tipo", "fala"),
            )
        updated["balloon_subregions"] = subregion_cache[subregion_key]
        enriched_texts.append(updated)

    updated_page = dict(page_result)
    updated_page["texts"] = enriched_texts
    updated_page.pop("_cached_image_bgr", None)
    return updated_page


def classify_layout_shape(bbox: list[int], tipo: str, region: dict | None = None) -> str:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect = width / height

    if region and len(region.get("texts", [])) > 1:
        stacked = _is_vertically_stacked(region["texts"])
        if stacked and aspect < 1.6:
            return "tall"

    if tipo == "narracao":
        return "wide" if aspect >= 1.6 else "square"
    if aspect >= 1.45:
        return "wide"
    if aspect <= 0.9:
        return "tall"
    return "square"


def classify_layout_align(tipo: str, layout_shape: str) -> str:
    if tipo == "narracao":
        return "top"
    if tipo == "sfx" and layout_shape == "tall":
        return "center"
    return "center"


def _find_region_for_text(text: dict, regions: list[dict]) -> dict | None:
    bbox = text.get("bbox")
    if not bbox:
        return None
    for region in regions:
        if _bbox_in_region(bbox, region["bbox"]):
            for candidate in region.get("texts", []):
                if candidate.get("bbox") == bbox:
                    return region
    return None


def _bbox_in_region(bbox: list[int], region_bbox: list[int]) -> bool:
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = region_bbox
    return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2


def _is_vertically_stacked(texts: list[dict]) -> bool:
    if len(texts) < 2:
        return False
    ordered = sorted(texts, key=lambda text: text.get("bbox", [0, 0, 0, 0])[1])
    total_vertical_gap = 0
    overlap_count = 0

    for previous, current in zip(ordered, ordered[1:]):
        px1, py1, px2, py2 = previous.get("bbox", [0, 0, 0, 0])
        cx1, cy1, cx2, cy2 = current.get("bbox", [0, 0, 0, 0])
        horizontal_overlap = min(px2, cx2) - max(px1, cx1)
        if horizontal_overlap > 0:
            overlap_count += 1
        total_vertical_gap += max(0, cy1 - py2)

    return overlap_count >= 1 and total_vertical_gap <= 40 * (len(ordered) - 1)


def _region_supports_shared_layout(region: dict, tipo: str) -> bool:
    texts = list(region.get("texts", []))
    if len(texts) <= 1:
        return True
    if tipo == "narracao":
        return _is_compact_text_cluster(texts, max_vertical_gap=40, max_horizontal_gap=50)
    return _is_compact_text_cluster(texts, max_vertical_gap=35, max_horizontal_gap=30)


def _is_compact_text_cluster(
    texts: list[dict],
    max_vertical_gap: int,
    max_horizontal_gap: int,
) -> bool:
    ordered = sorted(texts, key=lambda text: (text.get("bbox", [0, 0, 0, 0])[1], text.get("bbox", [0, 0, 0, 0])[0]))
    if _is_vertically_stacked(ordered):
        return True

    pair_count = 0
    for previous, current in zip(ordered, ordered[1:]):
        px1, py1, px2, py2 = previous.get("bbox", [0, 0, 0, 0])
        cx1, cy1, cx2, cy2 = current.get("bbox", [0, 0, 0, 0])
        prev_w = max(1, px2 - px1)
        prev_h = max(1, py2 - py1)
        curr_w = max(1, cx2 - cx1)
        curr_h = max(1, cy2 - cy1)
        overlap_x = max(0, min(px2, cx2) - max(px1, cx1))
        overlap_y = max(0, min(py2, cy2) - max(py1, cy1))
        overlap_x_ratio = overlap_x / float(max(1, min(prev_w, curr_w)))
        overlap_y_ratio = overlap_y / float(max(1, min(prev_h, curr_h)))
        vertical_gap = max(0, cy1 - py2)
        horizontal_gap = max(0, cx1 - px2, px1 - cx2)

        vertical_limit = max(max_vertical_gap, int(min(prev_h, curr_h) * 0.42))
        horizontal_limit = max(max_horizontal_gap, int(min(prev_w, curr_w) * 0.22))

        vertical_pair = overlap_x_ratio >= 0.55 and vertical_gap <= vertical_limit
        horizontal_pair = overlap_y_ratio >= 0.55 and horizontal_gap <= horizontal_limit
        if not vertical_pair and not horizontal_pair:
            return False
        pair_count += 1

    return pair_count == max(1, len(ordered) - 1)


def refine_balloon_bbox_from_image(
    image_bgr: np.ndarray | None,
    cluster_bbox: list[int],
    tipo: str,
) -> list[int]:
    if image_bgr is None or tipo not in {"fala", "narracao"}:
        return cluster_bbox

    x1, y1, x2, y2 = cluster_bbox
    height, width = image_bgr.shape[:2]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = max(18, int(box_w * 1.2))
    pad_y = max(18, int(box_h * 1.3))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(width, x2 + pad_x)
    ry2 = min(height, y2 + pad_y)

    roi = image_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return cluster_bbox

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 205, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    seed_x = min(rx2 - rx1 - 1, max(0, ((x1 + x2) // 2) - rx1))
    seed_y = min(ry2 - ry1 - 1, max(0, ((y1 + y2) // 2) - ry1))
    if thresh[seed_y, seed_x] == 0:
        return cluster_bbox

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    cluster_area = box_w * box_h

    for label in range(1, num_labels):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_edge = left <= 0 or top <= 0 or (left + comp_w) >= roi.shape[1] or (top + comp_h) >= roi.shape[0]

        if labels[seed_y, seed_x] != label:
            continue
        if touches_edge:
            return cluster_bbox
        if area < cluster_area * 1.4:
            return cluster_bbox
        if area > cluster_area * 30:
            return cluster_bbox
        max_width_factor = 4.8 if box_w <= 90 else 1.85
        max_height_factor = 5.8 if box_h <= 40 else 2.25
        if comp_w > int(box_w * max_width_factor) or comp_h > int(box_h * max_height_factor):
            return cluster_bbox

        refined = _expand_with_margin(
            [rx1 + left, ry1 + top, rx1 + left + comp_w, ry1 + top + comp_h],
            width,
            height,
            margin=3,
        )
        if not _contains_bbox(refined, cluster_bbox):
            return cluster_bbox
        return refined

    return cluster_bbox


def _contains_bbox(outer: list[int], inner: list[int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _load_page_image(page_result: dict):
    cached = page_result.get("_cached_image_bgr")
    if cached is not None:
        return cached
    image_path = page_result.get("image")
    if not image_path:
        return None
    try:
        image = cv2.imread(image_path)
        return image if image is not None else None
    except Exception:
        return None


def _expand_with_margin(bbox: list[int], image_width: int, image_height: int, margin: int = 2) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(image_width, x2 + margin),
        min(image_height, y2 + margin),
    ]


def _detect_connected_balloon_subregions(
    image_bgr: np.ndarray | None,
    text_bbox: list[int],
    balloon_bbox: list[int],
    tipo: str,
) -> list[list[int]]:
    if image_bgr is None or tipo not in {"fala", "pensamento"}:
        return []

    fill_subregions = _detect_connected_balloon_subregions_from_fill(
        image_bgr,
        text_bbox,
        balloon_bbox,
    )
    if len(fill_subregions) >= 2:
        return fill_subregions

    components = _extract_text_cluster_components(image_bgr, text_bbox)
    if len(components) < 2:
        return []

    merged_groups = _merge_text_cluster_components(components)
    if len(merged_groups) < 2:
        return []

    merged_groups = sorted(merged_groups, key=lambda item: item["area"], reverse=True)
    top_two = merged_groups[:2]
    remaining_area = sum(item["area"] for item in merged_groups)
    dominant_area = sum(item["area"] for item in top_two)
    if dominant_area < max(1400, int(remaining_area * 0.72)):
        return []

    first_bbox = top_two[0]["bbox"]
    second_bbox = top_two[1]["bbox"]
    box_w = max(1, balloon_bbox[2] - balloon_bbox[0])
    box_h = max(1, balloon_bbox[3] - balloon_bbox[1])
    first_cx = (first_bbox[0] + first_bbox[2]) / 2.0
    first_cy = (first_bbox[1] + first_bbox[3]) / 2.0
    second_cx = (second_bbox[0] + second_bbox[2]) / 2.0
    second_cy = (second_bbox[1] + second_bbox[3]) / 2.0
    if abs(first_cx - second_cx) < box_w * 0.12 and abs(first_cy - second_cy) < box_h * 0.12:
        return []

    ordered_groups = sorted(top_two, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    a_bbox = ordered_groups[0]["bbox"]
    b_bbox = ordered_groups[1]["bbox"]
    vertical_gap = max(0, int(b_bbox[1]) - int(a_bbox[3]))
    horizontal_gap = max(0, int(b_bbox[0]) - int(a_bbox[2]), int(a_bbox[0]) - int(b_bbox[2]))

    is_vertical_stack = abs(first_cx - second_cx) < box_w * 0.28
    is_horizontal_stack = abs(first_cy - second_cy) < box_h * 0.28
    if is_vertical_stack and vertical_gap < max(22, int(box_h * 0.12)):
        return []
    if is_horizontal_stack and horizontal_gap < max(22, int(box_w * 0.12)):
        return []
    if not is_vertical_stack and not is_horizontal_stack:
        # Diagonal: exigir separação visível em pelo menos um eixo
        if max(vertical_gap, horizontal_gap) < max(28, int(min(box_w, box_h) * 0.14)):
            return []

    subregions = [
        _expand_text_group_to_subregion(group["bbox"], balloon_bbox)
        for group in top_two
    ]
    subregions = sorted(subregions, key=lambda bbox: (bbox[1], bbox[0]))
    if subregions[0][3] > subregions[1][1]:
        seam_y = int((subregions[0][3] + subregions[1][1]) / 2)
        subregions[0][3] = max(subregions[0][1] + 28, seam_y)
        subregions[1][1] = min(subregions[1][3] - 28, seam_y)
    if _bbox_iou(subregions[0], subregions[1]) > 0.32:
        return []
    return subregions


def _detect_connected_balloon_subregions_from_fill(
    image_bgr: np.ndarray,
    seed_bbox: list[int],
    balloon_bbox: list[int],
) -> list[list[int]]:
    """Detecta balões conectados (2 lobos) a partir do preenchimento branco do balão.

    Ideia: erodir a região branca do balão. Se houver um "pescoço" fino conectando dois
    balões, a erosão separa em dois componentes grandes.
    """
    height, width = image_bgr.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in balloon_bbox]
    bx1 = max(0, min(width, bx1))
    bx2 = max(0, min(width, bx2))
    by1 = max(0, min(height, by1))
    by2 = max(0, min(height, by2))
    if bx2 <= bx1 or by2 <= by1:
        return []

    roi = image_bgr[by1:by2, bx1:bx2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    p85 = float(np.percentile(blur, 85))
    thresh_val = int(max(180, min(235, p85 - 8.0)))
    _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )

    sx1, sy1, sx2, sy2 = [int(v) for v in seed_bbox]
    seed_x = int(((sx1 + sx2) / 2.0) - bx1)
    seed_y = int(((sy1 + sy2) / 2.0) - by1)
    seed_x = max(0, min(roi.shape[1] - 1, seed_x))
    seed_y = max(0, min(roi.shape[0] - 1, seed_y))
    if thresh[seed_y, seed_x] == 0:
        return []

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats((thresh > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return []

    seed_label = int(labels[seed_y, seed_x])
    if seed_label <= 0:
        return []

    component = (labels == seed_label).astype(np.uint8) * 255
    fill_area = int(np.count_nonzero(component))
    if fill_area < 2500:
        return []

    min_dim = min(component.shape[:2])
    if min_dim >= 140:
        k = 11
    elif min_dim >= 90:
        k = 9
    else:
        k = 7
    eroded = cv2.erode(
        component,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        iterations=1,
    )

    num2, labels2, stats2, _ = cv2.connectedComponentsWithStats((eroded > 0).astype(np.uint8), connectivity=8)
    if num2 <= 2:
        return []

    min_area = max(800, int(fill_area * 0.12))
    lobes: list[dict] = []
    for label in range(1, num2):
        area = int(stats2[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        left = int(stats2[label, cv2.CC_STAT_LEFT])
        top = int(stats2[label, cv2.CC_STAT_TOP])
        comp_w = int(stats2[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats2[label, cv2.CC_STAT_HEIGHT])
        lobes.append(
            {
                "bbox": [bx1 + left, by1 + top, bx1 + left + comp_w, by1 + top + comp_h],
                "area": area,
            }
        )

    if len(lobes) < 2:
        return []

    lobes.sort(key=lambda item: item["area"], reverse=True)
    top_two = lobes[:2]

    # Separação mínima entre lobos (evita falso positivo em balão único com cintura estreita)
    box_w = max(1, bx2 - bx1)
    box_h = max(1, by2 - by1)
    ordered = sorted(top_two, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    a = ordered[0]["bbox"]
    b = ordered[1]["bbox"]
    gap_y = max(0, int(b[1]) - int(a[3]))
    gap_x = max(0, int(b[0]) - int(a[2]), int(a[0]) - int(b[2]))
    if max(gap_y, gap_x) < max(10, int(min(box_w, box_h) * 0.04)):
        return []

    subregions = [
        _expand_balloon_lobe_to_subregion(lobe["bbox"], balloon_bbox)
        for lobe in top_two
    ]
    subregions = sorted(subregions, key=lambda bbox: (bbox[1], bbox[0]))
    if _bbox_iou(subregions[0], subregions[1]) > 0.35:
        return []
    return subregions


def _expand_balloon_lobe_to_subregion(lobe_bbox: list[int], balloon_bbox: list[int]) -> list[int]:
    gx1, gy1, gx2, gy2 = lobe_bbox
    bx1, by1, bx2, by2 = balloon_bbox
    group_w = max(1, gx2 - gx1)
    group_h = max(1, gy2 - gy1)
    expand_x = max(10, int(group_w * 0.12))
    expand_y = max(10, int(group_h * 0.12))
    return [
        max(bx1, gx1 - expand_x),
        max(by1, gy1 - expand_y),
        min(bx2, gx2 + expand_x),
        min(by2, gy2 + expand_y),
    ]


def _extract_text_cluster_components(image_bgr: np.ndarray, text_bbox: list[int]) -> list[dict]:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in text_bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return []

    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    threshold = min(170, max(120, int(np.percentile(gray, 28))))
    dark = (gray <= threshold).astype(np.uint8) * 255
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    close_w = max(13, min(25, ((box_w // 26) * 2) + 1))
    close_h = max(7, min(13, ((box_h // 26) * 2) + 1))
    merged = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_w, close_h)),
        iterations=1,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype(np.uint8), 8)
    min_area = max(140, int((box_w * box_h) * 0.006))
    components: list[dict] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        components.append(
            {
                "bbox": [x1 + left, y1 + top, x1 + left + comp_w, y1 + top + comp_h],
                "area": area,
            }
        )
    return components


def _merge_text_cluster_components(components: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for component in components:
        merged = False
        for group in groups:
            if _should_merge_text_cluster_boxes(group["bbox"], component["bbox"]):
                group["bbox"] = _union_bbox(group["bbox"], component["bbox"])
                group["area"] += component["area"]
                merged = True
                break
        if not merged:
            groups.append(dict(component))
    return groups


def _should_merge_text_cluster_boxes(a: list[int], b: list[int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
    overlap_x_ratio = overlap_x / float(max(1, min(aw, bw)))
    overlap_y_ratio = overlap_y / float(max(1, min(ah, bh)))
    vertical_gap = max(0, max(ay1, by1) - min(ay2, by2))
    horizontal_gap = max(0, max(ax1, bx1) - min(ax2, bx2))

    if overlap_x_ratio >= 0.58 and vertical_gap <= max(12, int(min(ah, bh) * 0.24)):
        return True
    if overlap_y_ratio >= 0.58 and horizontal_gap <= max(20, int(min(aw, bw) * 0.18)):
        return True
    return False


def _expand_text_group_to_subregion(group_bbox: list[int], balloon_bbox: list[int]) -> list[int]:
    gx1, gy1, gx2, gy2 = group_bbox
    bx1, by1, bx2, by2 = balloon_bbox
    group_w = max(1, gx2 - gx1)
    group_h = max(1, gy2 - gy1)
    expand_x = max(14, int(group_w * 0.18))
    expand_y = max(12, int(group_h * 0.36))
    return [
        max(bx1, gx1 - expand_x),
        max(by1, gy1 - expand_y),
        min(bx2, gx2 + expand_x),
        min(by2, gy2 + expand_y),
    ]


def _union_bbox(a: list[int], b: list[int]) -> list[int]:
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (a[2] - a[0]) * (a[3] - a[1])))
    area_b = float(max(1, (b[2] - b[0]) * (b[3] - b[1])))
    return intersection / max(1.0, area_a + area_b - intersection)
