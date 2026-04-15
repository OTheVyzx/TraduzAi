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
        shared_group_size = len(region["texts"]) if region else 1
        use_region_bbox = bool(region) and use_shared_layout and shared_group_size > 1
        inferred_bbox = region["bbox"] if use_region_bbox else text.get("bbox", [0, 0, 0, 0])
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
        updated["layout_group_size"] = shared_group_size if use_shared_layout and region else 1
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
        subs = subregion_cache[subregion_key]
        updated["balloon_subregions"] = subs
        updated["subregion_confidence"] = _score_subregion_quality(subs, balloon_bbox) if subs else 0.0
        enriched_texts.append(updated)

    _apply_geometric_fallback_subregions(enriched_texts)

    updated_page = dict(page_result)
    updated_page["texts"] = enriched_texts
    updated_page.pop("_cached_image_bgr", None)
    return updated_page


def _apply_geometric_fallback_subregions(texts: list[dict]) -> None:
    """Fallback: para balões largos sem subregions, infere split geométrico.

    Funciona em dois modos:
      A) Multi-texto (group_size > 1): agrupa textos pelo balloon_bbox e usa
         a distribuição dos centros X para refinar o ponto de corte.
      B) Texto único (group_size == 1): se o balão é largo o suficiente
         (aspect >= 2.0), divide no centro — o renderer faz o split semântico.
    """
    groups: dict[tuple, list[dict]] = {}
    for text in texts:
        balloon = text.get("balloon_bbox")
        if not balloon or text.get("balloon_subregions"):
            continue
        tipo = text.get("tipo", "fala")
        if tipo not in {"fala", "pensamento"}:
            continue
        key = (tipo, tuple(int(v) for v in balloon))
        groups.setdefault(key, []).append(text)

    for (tipo, bbox_tuple), group in groups.items():
        balloon = list(bbox_tuple)
        bw = max(1, balloon[2] - balloon[0])
        bh = max(1, balloon[3] - balloon[1])
        aspect = bw / float(bh)

        if len(group) >= 2 and int(group[0].get("layout_group_size", 1)) > 1:
            # Modo A: multi-texto — precisa de aspect >= 1.8
            if aspect < 1.8:
                continue
            text_bboxes = [t.get("bbox", [0, 0, 0, 0]) for t in group]
            subregions = _geometric_fallback_subregions(text_bboxes, balloon)
        elif len(group) == 1 and aspect >= 2.0 and min(bw, bh) >= 200 and max(bw, bh) >= 500:
            # Modo B: texto único em balão muito largo E grande — provável balão conectado
            # Requer aspect >= 2.0, dimensão menor >= 200px, maior >= 500px
            # para evitar falsos positivos em balões simples que são apenas largos
            subregions = _geometric_fallback_subregions(
                [group[0].get("bbox", [0, 0, 0, 0])], balloon
            )
        else:
            continue

        if len(subregions) >= 2:
            for text in group:
                text["balloon_subregions"] = subregions


def _geometric_fallback_subregions(
    text_bboxes: list[list[int]],
    balloon_bbox: list[int],
) -> list[list[int]]:
    """Divide balão em 2 subregions baseado na geometria e posições dos textos.

    Estratégia com 2+ textos:
      1. Calcula dx e dy entre os dois centros de texto mais distantes
      2. Se predominantemente horizontal (dx > dy*1.2) → corte vertical
      3. Se predominantemente vertical (dy > dx*1.2) → corte horizontal
      4. Se diagonal (nenhum predomina) → corte diagonal via quadrantes

    Estratégia com 1 texto:
      - Aspect ratio do balão decide: largo → vertical, alto → horizontal

    Reject com 2+ textos:
      - Se os centros estão muito próximos (< 25% da dimensão principal),
        retorna [] — é um balão único, não conectado.
    """
    bx1, by1, bx2, by2 = balloon_bbox
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    if len(text_bboxes) >= 2:
        # Encontrar os 2 centros mais distantes
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in text_bboxes]
        best_pair = (0, 1)
        best_sep = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                sep = ((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2) ** 0.5
                if sep > best_sep:
                    best_sep = sep
                    best_pair = (i, j)

        c0, c1 = centers[best_pair[0]], centers[best_pair[1]]
        dx = abs(c0[0] - c1[0])
        dy = abs(c0[1] - c1[1])

        # Reject: centros muito próximos → balão único
        if dx < bw * 0.25 and dy < bh * 0.25:
            return []

        if dx > dy * 1.2:
            # Predominantemente horizontal → corte vertical
            seam_x = int((c0[0] + c1[0]) / 2)
            seam_x = max(bx1 + 24, min(bx2 - 24, seam_x))
            return [[bx1, by1, seam_x, by2], [seam_x, by1, bx2, by2]]
        elif dy > dx * 1.2:
            # Predominantemente vertical → corte horizontal
            seam_y = int((c0[1] + c1[1]) / 2)
            seam_y = max(by1 + 24, min(by2 - 24, seam_y))
            return [[bx1, by1, bx2, seam_y], [bx1, seam_y, bx2, by2]]
        else:
            # Diagonal: dividir em quadrantes opostos (top-left / bottom-right
            # ou top-right / bottom-left) baseado nos centros de texto.
            seam_x = int((c0[0] + c1[0]) / 2)
            seam_y = int((c0[1] + c1[1]) / 2)
            seam_x = max(bx1 + 24, min(bx2 - 24, seam_x))
            seam_y = max(by1 + 24, min(by2 - 24, seam_y))
            # Qual par de quadrantes? O que contém os centros dos textos.
            if (c0[0] < c1[0]) == (c0[1] < c1[1]):
                # top-left + bottom-right (diagonal \)
                return [
                    [bx1, by1, seam_x, seam_y],
                    [seam_x, seam_y, bx2, by2],
                ]
            else:
                # top-right + bottom-left (diagonal /)
                return [
                    [seam_x, by1, bx2, seam_y],
                    [bx1, seam_y, seam_x, by2],
                ]

    # 1 texto — usar aspect ratio do balão
    aspect = bw / float(bh)
    if aspect >= 1.4:
        seam_x = bx1 + bw // 2
        seam_x = max(bx1 + 24, min(bx2 - 24, seam_x))
        return [[bx1, by1, seam_x, by2], [seam_x, by1, bx2, by2]]
    else:
        seam_y = by1 + bh // 2
        seam_y = max(by1 + 24, min(by2 - 24, seam_y))
        return [[bx1, by1, bx2, seam_y], [bx1, seam_y, bx2, by2]]


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
    cluster_area = box_w * box_h
    expand_scales = [1.2, 1.75, 2.35]

    for scale_index, scale in enumerate(expand_scales):
        pad_x = max(18, int(box_w * scale))
        pad_y = max(18, int(box_h * (scale + 0.15)))
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(width, x2 + pad_x)
        ry2 = min(height, y2 + pad_y)

        roi = image_bgr[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh_val = int(max(205, min(232, np.percentile(blur, 76))))
        _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        search_seed = np.zeros_like(thresh, dtype=np.uint8)
        seed_pad_x = max(6, int(box_w * 0.10))
        seed_pad_y = max(6, int(box_h * 0.18))
        sx1 = max(0, x1 - rx1 - seed_pad_x)
        sy1 = max(0, y1 - ry1 - seed_pad_y)
        sx2 = min(thresh.shape[1], x2 - rx1 + seed_pad_x)
        sy2 = min(thresh.shape[0], y2 - ry1 + seed_pad_y)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        search_seed[sy1:sy2, sx1:sx2] = 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        candidate_bbox = None
        candidate_score = float("-inf")
        should_retry_with_larger_roi = False

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(cluster_area * 1.12):
                continue

            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
            comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            component = labels == label
            overlap = int(np.count_nonzero(search_seed[component]))
            if overlap <= 0:
                continue

            global_bbox = [rx1 + left, ry1 + top, rx1 + left + comp_w, ry1 + top + comp_h]
            touches_left = left <= 0
            touches_top = top <= 0
            touches_right = (left + comp_w) >= roi.shape[1]
            touches_bottom = (top + comp_h) >= roi.shape[0]
            touches_image_left = global_bbox[0] <= 0
            touches_image_top = global_bbox[1] <= 0
            touches_image_right = global_bbox[2] >= width
            touches_image_bottom = global_bbox[3] >= height

            if (
                (touches_left and not touches_image_left)
                or (touches_top and not touches_image_top)
                or (touches_right and not touches_image_right)
                or (touches_bottom and not touches_image_bottom)
            ):
                should_retry_with_larger_roi = True
                continue

            if area > cluster_area * 42:
                continue
            overlap_ratio = _bbox_overlap_ratio(global_bbox, cluster_bbox)
            if overlap_ratio < 0.60:
                continue

            max_width_factor = 5.0 if box_w <= 90 else 2.2
            max_height_factor = 6.0 if box_h <= 40 else 2.9
            if comp_w > int(box_w * max_width_factor) and not (touches_image_left or touches_image_right):
                continue
            if comp_h > int(box_h * max_height_factor) and not (touches_image_top or touches_image_bottom):
                continue

            score = float(overlap * 8) + float(area * 0.02)
            if touches_image_top or touches_image_bottom or touches_image_left or touches_image_right:
                score += 40.0
            if score > candidate_score:
                candidate_bbox = global_bbox
                candidate_score = score

        if candidate_bbox is not None:
            refined = _expand_with_margin(candidate_bbox, width, height, margin=3)
            if _contains_bbox(refined, cluster_bbox) or _bbox_overlap_ratio(refined, cluster_bbox) >= 0.78:
                return refined

        if not should_retry_with_larger_roi or scale_index == len(expand_scales) - 1:
            break

    return cluster_bbox


def _contains_bbox(outer: list[int], inner: list[int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _bbox_overlap_ratio(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area = float(max(1, (b[2] - b[0]) * (b[3] - b[1])))
    return inter / area


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

    fill_result = _detect_connected_balloon_subregions_from_fill(
        image_bgr,
        text_bbox,
        balloon_bbox,
    )
    if len(fill_result) >= 2:
        return fill_result

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

    is_vertical_stack = abs(first_cx - second_cx) < box_w * 0.35
    is_horizontal_stack = abs(first_cy - second_cy) < box_h * 0.35
    if is_vertical_stack and vertical_gap < max(18, int(box_h * 0.08)):
        return []
    if is_horizontal_stack and horizontal_gap < max(18, int(box_w * 0.08)):
        return []
    if not is_vertical_stack and not is_horizontal_stack:
        if max(vertical_gap, horizontal_gap) < max(22, int(min(box_w, box_h) * 0.10)):
            return []

    subregions = _build_balloon_subregions_from_groups(
        [group["bbox"] for group in top_two],
        balloon_bbox,
    )
    if len(subregions) < 2 or _bbox_iou(subregions[0], subregions[1]) > 0.32:
        return []
    return subregions


def _score_subregion_quality(subregions: list[list[int]], balloon_bbox: list[int]) -> float:
    """Score how good a subregion split is (0.0 = bad, 1.0 = excellent).

    Factors:
      - Coverage: subregions should cover most of the balloon area
      - Balance: subregion areas shouldn't be wildly different (max 4:1 ratio)
      - Overlap: subregions shouldn't overlap significantly
    """
    if len(subregions) < 2:
        return 0.0

    bx1, by1, bx2, by2 = balloon_bbox
    balloon_area = max(1, (bx2 - bx1) * (by2 - by1))
    sub_areas = [max(1, (s[2] - s[0]) * (s[3] - s[1])) for s in subregions]
    total_sub_area = sum(sub_areas)

    # Coverage: how much of the balloon is covered by subregions
    coverage = min(1.0, total_sub_area / float(balloon_area))
    coverage_score = min(1.0, coverage / 0.85)  # 85%+ coverage → full score

    # Balance: ratio between smallest and largest subregion
    min_area = min(sub_areas)
    max_area = max(sub_areas)
    ratio = min_area / float(max_area)
    balance_score = min(1.0, ratio / 0.25)  # 25%+ ratio → full score

    # Overlap: penalize overlapping subregions
    overlap = _bbox_iou(subregions[0], subregions[1]) if len(subregions) >= 2 else 0.0
    overlap_penalty = max(0.0, 1.0 - overlap * 5.0)

    return round(coverage_score * 0.4 + balance_score * 0.3 + overlap_penalty * 0.3, 3)


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
    base_k = 7
    if min_dim >= 180:
        base_k = 13
    elif min_dim >= 120:
        base_k = 11
    elif min_dim >= 80:
        base_k = 9

    # Tenta erosão progressiva para separar lobos conectados por "pescoços"
    eroded_components = []
    for iters in range(1, 4):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_k, base_k))
        eroded = cv2.erode(component, kernel, iterations=iters)
        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats((eroded > 0).astype(np.uint8), connectivity=8)
        
        # Filtra componentes significantes após erosão
        current_lobes = []
        min_lobe_area = max(600, int(fill_area * 0.08))
        for label in range(1, num2):
            area = int(stats2[label, cv2.CC_STAT_AREA])
            if area < min_lobe_area:
                continue
            left = int(stats2[label, cv2.CC_STAT_LEFT])
            top = int(stats2[label, cv2.CC_STAT_TOP])
            comp_w = int(stats2[label, cv2.CC_STAT_WIDTH])
            comp_h = int(stats2[label, cv2.CC_STAT_HEIGHT])
            current_lobes.append({
                "bbox": [bx1 + left, by1 + top, bx1 + left + comp_w, by1 + top + comp_h],
                "area": area,
            })
        
        if len(current_lobes) >= 2:
            eroded_components = current_lobes
            break
        if float(np.count_nonzero(eroded)) < fill_area * 0.25:
            break

    if len(eroded_components) < 2:
        # Fallback: distance transform para pescoços largos que erosão não quebra
        dt_lobes = _detect_lobes_via_distance_transform(
            component, [bx1, by1, bx2, by2], fill_area,
        )
        if len(dt_lobes) < 2:
            return []
        dt_lobes.sort(key=lambda item: item["area"], reverse=True)
        top_two_dt = dt_lobes[:2]
        subregions = _build_balloon_subregions_from_groups(
            [lobe["bbox"] for lobe in top_two_dt],
            balloon_bbox,
        )
        if len(subregions) < 2 or _bbox_iou(subregions[0], subregions[1]) > 0.35:
            return []
        return subregions

    lobes = eroded_components

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

    # Gap check mais flexível para erosão profunda
    if max(gap_y, gap_x) < max(6, int(min(box_w, box_h) * 0.02)):
        return []

    subregions = _build_balloon_subregions_from_groups(
        [lobe["bbox"] for lobe in top_two],
        balloon_bbox,
    )
    if len(subregions) < 2 or _bbox_iou(subregions[0], subregions[1]) > 0.35:
        return []
    return subregions


def _detect_lobes_via_distance_transform(
    component: np.ndarray,
    balloon_bbox: list[int],
    fill_area: int,
) -> list[dict]:
    """Detecta lobos de balão conectado via distance transform.

    Fallback para quando a erosão progressiva não consegue separar os lobos
    (ex.: pescoço largo entre dois lobos).
    """
    # Preenche buracos internos (ex.: pixels de texto escuro) antes do DT
    # Sem isso, retângulos de texto criam peaks falsos nas laterais
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_component = np.zeros_like(component)
    if contours:
        cv2.drawContours(filled_component, contours, -1, 255, -1)
    else:
        filled_component = component.copy()

    dist = cv2.distanceTransform(filled_component, cv2.DIST_L2, 5)
    max_val = float(dist.max())
    if max_val < 3.0:
        return []

    k_size = max(5, (min(filled_component.shape[:2]) // 8) | 1)
    blurred = cv2.GaussianBlur(dist, (k_size, k_size), 0)

    peak_thresh = 0.45 * max_val
    _, peak_mask = cv2.threshold(blurred, peak_thresh, 255, cv2.THRESH_BINARY)
    peak_mask = peak_mask.astype(np.uint8)

    num_peaks, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(
        peak_mask, connectivity=8,
    )
    if num_peaks < 3:
        return []

    bx1, by1, bx2, by2 = balloon_bbox
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    min_peak_val = 0.3 * max_val
    min_lobe_area = max(600, int(fill_area * 0.08))

    peaks = []
    for label in range(1, num_peaks):
        area = int(peak_stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        cy, cx = float(peak_centroids[label][1]), float(peak_centroids[label][0])
        val = float(blurred[int(cy), int(cx)])
        if val < min_peak_val:
            continue
        peaks.append({"cx": cx, "cy": cy, "val": val, "area": area})

    if len(peaks) < 2:
        return []

    peaks.sort(key=lambda p: p["val"], reverse=True)
    top_peaks = peaks[:2]

    sep = ((top_peaks[0]["cx"] - top_peaks[1]["cx"]) ** 2 + (top_peaks[0]["cy"] - top_peaks[1]["cy"]) ** 2) ** 0.5
    if sep < min(bw, bh) * 0.25:
        return []

    p0_cx, p0_cy = top_peaks[0]["cx"], top_peaks[0]["cy"]
    p1_cx, p1_cy = top_peaks[1]["cx"], top_peaks[1]["cy"]
    mid_x = (p0_cx + p1_cx) / 2.0
    mid_y = (p0_cy + p1_cy) / 2.0
    dx = abs(p0_cx - p1_cx)
    dy = abs(p0_cy - p1_cy)

    h, w = component.shape[:2]
    lobe_masks = [np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)]

    # Partição por proximidade (Voronoi entre os 2 peaks).
    # Cada pixel do componente vai para o lobo cujo peak está mais perto.
    # Funciona para splits horizontais, verticais e diagonais sem
    # assumir um eixo dominante.
    ys, xs = np.where(component > 0)
    if len(xs) > 0:
        d0 = (xs - p0_cx) ** 2 + (ys - p0_cy) ** 2
        d1 = (xs - p1_cx) ** 2 + (ys - p1_cy) ** 2
        mask0 = d0 <= d1
        lobe_masks[0][ys[mask0], xs[mask0]] = 255
        lobe_masks[1][ys[~mask0], xs[~mask0]] = 255

    lobes = []
    for mask in lobe_masks:
        area = int(np.count_nonzero(mask))
        if area < min_lobe_area:
            continue
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        lx1, ly1 = int(xs.min()), int(ys.min())
        lx2, ly2 = int(xs.max()) + 1, int(ys.max()) + 1
        lobes.append({
            "bbox": [bx1 + lx1, by1 + ly1, bx1 + lx2, by1 + ly2],
            "area": area,
        })

    return lobes


def _build_balloon_subregions_from_groups(group_bboxes: list[list[int]], balloon_bbox: list[int]) -> list[list[int]]:
    if len(group_bboxes) < 2:
        return []

    ordered = [list(bbox) for bbox in group_bboxes[:2]]
    bx1, by1, bx2, by2 = balloon_bbox
    centers = [
        ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0, bbox)
        for bbox in ordered
    ]
    dx = abs(centers[0][0] - centers[1][0])
    dy = abs(centers[0][1] - centers[1][1])

    if dy >= dx * 1.1:
        top_bbox, bottom_bbox = sorted(ordered, key=lambda bbox: (bbox[1], bbox[0]))
        if top_bbox[3] < bottom_bbox[1]:
            seam_y = int((top_bbox[3] + bottom_bbox[1]) / 2)
        else:
            seam_y = int((centers[0][1] + centers[1][1]) / 2.0)
        seam_y = max(by1 + 32, min(by2 - 32, seam_y))
        return [
            [bx1, by1, bx2, seam_y],
            [bx1, seam_y, bx2, by2],
        ]

    if dx >= dy * 1.1:
        left_bbox, right_bbox = sorted(ordered, key=lambda bbox: (bbox[0], bbox[1]))
        if left_bbox[2] < right_bbox[0]:
            seam_x = int((left_bbox[2] + right_bbox[0]) / 2)
        else:
            seam_x = int((centers[0][0] + centers[1][0]) / 2.0)
        seam_x = max(bx1 + 32, min(bx2 - 32, seam_x))
        return [
            [bx1, by1, seam_x, by2],
            [seam_x, by1, bx2, by2],
        ]

    # Diagonal connected balloon: use the full balloon more aggressively instead
    # of just expanding the OCR group box. This leaves each lobe with a larger,
    # more editorial text area and avoids the left lobe becoming too narrow.
    top_bbox, bottom_bbox = sorted(ordered, key=lambda bbox: (bbox[1], bbox[0]))
    top_cx = (top_bbox[0] + top_bbox[2]) / 2.0
    top_cy = (top_bbox[1] + top_bbox[3]) / 2.0
    bottom_cx = (bottom_bbox[0] + bottom_bbox[2]) / 2.0
    bottom_cy = (bottom_bbox[1] + bottom_bbox[3]) / 2.0
    seam_x = int((top_bbox[2] + bottom_bbox[0]) / 2) if top_bbox[2] < bottom_bbox[0] else int((top_cx + bottom_cx) / 2.0)
    seam_y = int((top_bbox[3] + bottom_bbox[1]) / 2) if top_bbox[3] < bottom_bbox[1] else int((top_cy + bottom_cy) / 2.0)
    seam_x = max(bx1 + 32, min(bx2 - 32, seam_x))
    seam_y = max(by1 + 28, min(by2 - 28, seam_y))
    overlap_x = max(18, int((bx2 - bx1) * 0.08))
    overlap_y = max(14, int((by2 - by1) * 0.07))

    if top_cx <= bottom_cx:
        first = [bx1, by1, min(bx2, seam_x + overlap_x), min(by2, seam_y + overlap_y)]
        second = [max(bx1, seam_x - overlap_x), max(by1, seam_y - overlap_y), bx2, by2]
    else:
        first = [max(bx1, seam_x - overlap_x), by1, bx2, min(by2, seam_y + overlap_y)]
        second = [bx1, max(by1, seam_y - overlap_y), min(bx2, seam_x + overlap_x), by2]

    return sorted([first, second], key=lambda bbox: (bbox[1], bbox[0]))


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
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    threshold = min(170, max(120, int(np.percentile(gray, 28))))
    dark = (gray <= threshold).astype(np.uint8) * 255
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    close_w = max(13, min(25, ((box_w // 26) * 2) + 1))
    close_h = max(7, min(13, ((box_h // 26) * 2) + 1))
    merged = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_w, close_h)),
        iterations=1,
    )

    min_area = max(140, int((box_w * box_h) * 0.006))
    components = _component_boxes_from_binary_mask(merged, x1, y1, min_area)
    if _needs_text_component_blackhat_retry(components, box_w, box_h):
        fallback = _extract_text_cluster_components_blackhat(gray, x1, y1, box_w, box_h)
        if len(fallback) >= 2:
            return fallback
    return components


def _merge_text_cluster_components(components: list[dict]) -> list[dict]:
    groups = [dict(component) for component in components]
    changed = True
    while changed:
        changed = False
        new_groups: list[dict] = []
        pending = groups[:]
        while pending:
            current = pending.pop(0)
            index = 0
            while index < len(pending):
                other = pending[index]
                if _should_merge_text_cluster_boxes(current["bbox"], other["bbox"]):
                    current["bbox"] = _union_bbox(current["bbox"], other["bbox"])
                    current["area"] += other["area"]
                    pending.pop(index)
                    changed = True
                    index = 0
                    continue
                index += 1
            new_groups.append(current)
        groups = new_groups
    return groups


def _component_boxes_from_binary_mask(mask: np.ndarray, offset_x: int, offset_y: int, min_area: int) -> list[dict]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
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
                "bbox": [offset_x + left, offset_y + top, offset_x + left + comp_w, offset_y + top + comp_h],
                "area": area,
            }
        )
    return components


def _needs_text_component_blackhat_retry(components: list[dict], box_w: int, box_h: int) -> bool:
    if len(components) < 2:
        return True
    box_area = max(1, box_w * box_h)
    dominant = max(component["area"] for component in components)
    return dominant > int(box_area * 0.42)


def _extract_text_cluster_components_blackhat(
    gray: np.ndarray,
    offset_x: int,
    offset_y: int,
    box_w: int,
    box_h: int,
) -> list[dict]:
    kernel_w = max(21, min(45, ((box_w // 18) * 2) + 1))
    kernel_h = max(17, min(31, ((box_h // 18) * 2) + 1))
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)),
    )
    threshold = max(18, int(np.percentile(blackhat, 88)))
    candidate = (blackhat >= threshold).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 9)),
        iterations=1,
    )
    min_area = max(90, int((box_w * box_h) * 0.0014))
    return _component_boxes_from_binary_mask(candidate, offset_x, offset_y, min_area)


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
