"""
Infer layout regions for text balloons/narration areas from OCR clusters.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.request

import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

_TEXTURED_FONT_KEYWORDS = (
    "newrotic",
    "single fighter",
    "singlefighter",
    "badaboom",
)

try:
    from inpainter.mask_builder import build_mask_regions # type: ignore
    from translator.translate import OLLAMA_HOST, _check_ollama, _pick_ollama_model # type: ignore
except ImportError:
    # Fallback para o analisador não reclamar de falta de pasta pai no IDE
    from ..inpainter.mask_builder import build_mask_regions  # type: ignore
    from ..translator.translate import OLLAMA_HOST, _check_ollama, _pick_ollama_model  # type: ignore

try:
    from utils.decision_log import infer_page_number, record_decision
except ImportError:
    from ..utils.decision_log import infer_page_number, record_decision


def _resolve_page_number(page_result: dict) -> int | None:
    raw = page_result.get("numero")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    image_ref = (
        page_result.get("image")
        or page_result.get("arquivo_original")
        or page_result.get("arquivo_traduzido")
        or ""
    )
    return infer_page_number(image_ref)


def _bbox_area(bbox: list[int]) -> int:
    return max(1, int(bbox[2]) - int(bbox[0])) * max(1, int(bbox[3]) - int(bbox[1]))


def _clamp_top_narration_bbox(
    balloon_bbox: list[int],
    inferred_bbox: list[int],
    page_width: int,
    page_height: int,
    tipo: str,
    profile: str = "standard",
) -> tuple[list[int], float | None]:
    if tipo != "narracao" or not inferred_bbox or len(inferred_bbox) != 4:
        return balloon_bbox, None

    ox1, oy1, ox2, oy2 = [int(v) for v in inferred_bbox]
    bx1, by1, bx2, by2 = [int(v) for v in balloon_bbox]
    source_w = max(1, ox2 - ox1)
    expanded_w = max(1, bx2 - bx1)
    expansion_ratio = expanded_w / float(source_w)

    top_limit_ratio = 0.28 if profile == "top_narration" else 0.22
    expansion_limit = 1.75 if profile == "top_narration" else 2.2
    if oy1 > int(page_height * top_limit_ratio) or expansion_ratio <= expansion_limit:
        return balloon_bbox, None

    max_pad_x = max(32, int(source_w * (0.42 if profile == "top_narration" else 0.55)))
    max_pad_top = max(18, int(max(1, oy2 - oy1) * (0.9 if profile == "top_narration" else 1.0)))
    clamped = [
        max(0, ox1 - max_pad_x),
        max(0, min(by1, oy1 - max_pad_top)),
        min(page_width, ox2 + max_pad_x),
        min(page_height, by2),
    ]
    return clamped, round(expansion_ratio, 3)


def _resolve_text_font_name(text: dict) -> str:
    style = text.get("estilo") or text.get("style") or {}
    return str(style.get("fonte") or style.get("font") or "")


def _looks_like_textured_lettering(text: dict) -> bool:
    balloon_type = str(text.get("balloon_type") or "").strip().lower()
    if balloon_type == "textured":
        return True
    font_name = _resolve_text_font_name(text).strip().lower()
    return any(keyword in font_name for keyword in _TEXTURED_FONT_KEYWORDS)


def _balloon_region_looks_white(image_bgr: np.ndarray | None, balloon_bbox: list[int]) -> bool:
    if image_bgr is None or not balloon_bbox or len(balloon_bbox) != 4:
        return False

    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in balloon_bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return False

    bw = x2 - x1
    bh = y2 - y1
    inset_x = min(max(6, int(bw * 0.05)), max(0, (bw // 2) - 1))
    inset_y = min(max(6, int(bh * 0.05)), max(0, (bh // 2) - 1))
    sx1 = min(x2 - 1, x1 + inset_x)
    sx2 = max(sx1 + 1, x2 - inset_x)
    sy1 = min(y2 - 1, y1 + inset_y)
    sy2 = max(sy1 + 1, y2 - inset_y)
    roi = image_bgr[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 1] <= 44) & (hsv[:, :, 2] >= 182)
    white_ratio = float(np.count_nonzero(white_mask)) / float(max(1, white_mask.size))
    return white_ratio >= 0.58


def _can_try_connected_balloon_detection(
    text: dict,
    *,
    balloon_bbox: list[int],
    layout_shape: str,
    page_image: np.ndarray | None,
) -> bool:
    tipo = str(text.get("tipo") or "fala").strip().lower()
    if _looks_like_textured_lettering(text):
        return False

    block_profile = str(text.get("block_profile") or text.get("layout_profile") or "").strip().lower()
    if layout_shape in {"box", "rectangle"}:
        return False

    if tipo in {"fala", "pensamento"}:
        return True
    if tipo != "narracao":
        return False

    text_body = str(text.get("translated") or text.get("text") or "")
    return (
        block_profile in {"white_balloon", "connected_balloon"}
        and _has_multiple_sentences(text_body)
        and _balloon_region_looks_white(page_image, balloon_bbox)
    )


def _has_multiple_sentences(text: str) -> bool:
    return len(re.findall(r"[.!?…]+", str(text or ""))) >= 2


def enrich_page_layout(page_result: dict) -> dict:
    texts = page_result.get("texts", [])
    width = int(page_result.get("width", 0) or 0)
    height = int(page_result.get("height", 0) or 0)
    if not texts or width <= 0 or height <= 0:
        return page_result

    regions = build_mask_regions(texts=texts, image_shape=(height, width, 3))
    page_image = _load_page_image(page_result)
    bubble_regions = _normalize_bubble_regions(page_result)
    reasoner_settings = _resolve_connected_reasoner_settings(page_result)
    enriched_texts = []
    subregion_cache: dict[tuple, list[dict]] = {}
    page_number = _resolve_page_number(page_result)

    for index, text in enumerate(texts, start=1):
        layer_ref = str(text.get("id") or f"ocr_{index:03d}")
        region = _find_region_for_text(text, regions)
        use_shared_layout = bool(region) and _region_supports_shared_layout(region, text.get("tipo", "fala"))
        shared_group_size = len(region["texts"]) if region else 1
        use_region_bbox = bool(region) and use_shared_layout and shared_group_size > 1
        inferred_bbox = region["bbox"] if use_region_bbox else text.get("bbox", [0, 0, 0, 0])
        bubble_bbox = _select_bubble_region_for_bbox(inferred_bbox, bubble_regions)
        refined_bbox = None
        if page_image is not None:
            refined_bbox = refine_balloon_bbox_from_image(
                page_image, inferred_bbox, text.get("tipo", "fala")
            )
            if refined_bbox is not None:
                refined_area = _bbox_area(refined_bbox)
                inferred_area = _bbox_area(inferred_bbox)
                page_area = max(1, width * height)
                if refined_area >= int(page_area * 0.88) and refined_area >= int(inferred_area * 3.0):
                    refined_bbox = list(inferred_bbox)
        if refined_bbox is not None and list(refined_bbox) != list(inferred_bbox):
            balloon_bbox = refined_bbox
            layout_reason = "refined_from_image"
        elif bubble_bbox is not None:
            balloon_bbox = bubble_bbox
            layout_reason = "bubble_region"
        elif refined_bbox is not None:
            balloon_bbox = refined_bbox
            layout_reason = "refined_same_as_cluster"
        else:
            balloon_bbox = inferred_bbox
            layout_reason = "cluster_bbox"
        block_profile = str(text.get("block_profile") or "")
        layout_profile = str(text.get("layout_profile") or block_profile or "standard")
        balloon_bbox, expansion_ratio = _clamp_top_narration_bbox(
            balloon_bbox,
            inferred_bbox,
            width,
            height,
            text.get("tipo", "fala"),
            profile=layout_profile,
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
        updated["layout_profile"] = layout_profile
        updated["layout_group_size"] = shared_group_size if use_shared_layout and region else 1
        updated["ocr_text_bbox"] = [int(v) for v in inferred_bbox] if inferred_bbox else []
        updated["connected_text_groups"] = []
        updated["connected_lobe_bboxes"] = []
        updated["connected_lobe_polygons"] = []
        updated["connected_position_bboxes"] = []
        updated["connected_detection_confidence"] = 0.0
        updated["connected_group_confidence"] = 0.0
        updated["connected_position_confidence"] = 0.0
        if expansion_ratio is not None:
            record_decision(
                stage="layout",
                action="clamp_balloon_bbox",
                reason="top_caption_overexpand",
                page=page_number,
                layer=layer_ref,
                text=text.get("text", ""),
                bbox=balloon_bbox,
                details={"expansion_ratio": expansion_ratio, "layout_profile": layout_profile},
            )
        record_decision(
            stage="layout",
            action="assign_balloon_bbox",
            reason=layout_reason,
            page=page_number,
            layer=layer_ref,
            text=text.get("text", ""),
            bbox=balloon_bbox,
            details={
                "source_bbox": [int(v) for v in inferred_bbox] if inferred_bbox else [],
                "layout_shape": layout_shape,
                "layout_profile": layout_profile,
            },
        )
        subregion_key = (
            tuple(int(v) for v in inferred_bbox),
            tuple(int(v) for v in balloon_bbox),
            str(text.get("tipo", "fala")),
        )
        can_try_connected = _can_try_connected_balloon_detection(
            text,
            balloon_bbox=balloon_bbox,
            layout_shape=layout_shape,
            page_image=page_image,
        )
        if subregion_key not in subregion_cache:
            # TRAVA ANTI-BURACO: Caixas de narração retangulares não têm lobos.
            # Mas narração dentro de balão branco (block_profile == "white_balloon")
            # pode ser um balão duplo conectado — deixa a detecção topológica decidir.
            block_profile = str(text.get("block_profile") or "")
            tipo_text = text.get("tipo", "fala")
            is_rect_narration = layout_shape in ("box", "rectangle") or (
                tipo_text == "narracao" and block_profile != "white_balloon"
            )

            if is_rect_narration or not can_try_connected:
                subregion_cache[subregion_key] = []
            else:
                subregion_cache[subregion_key] = _detect_connected_balloon_subregions_rich(
                    page_image,
                    inferred_bbox,
                    balloon_bbox,
                    text.get("tipo", "fala"),
                )
        rich_subs = subregion_cache[subregion_key]
        subs = [item["bbox"] for item in rich_subs]
        # Polígonos de cada lobo (None quando não disponível)
        lobe_polygons: list[list[list[int]] | None] = [
            item.get("polygon") for item in rich_subs
        ]
        connected_plan = _analyze_connected_subregions(subs, balloon_bbox, profile=layout_profile) if subs else {}
        ordered_subregions = connected_plan.get("ordered_subregions", subs)
        # Reordenar polígonos para combinar com ordered_subregions
        if len(ordered_subregions) == len(subs) and subs:
            ordered_polygons = _reorder_polygons_to_match_subregions(
                subs, ordered_subregions, lobe_polygons,
            )
        else:
            ordered_polygons = lobe_polygons
        orientation = connected_plan.get("orientation", "")
        connected_visuals = _derive_connected_visual_boxes(
            page_image,
            updated.get("ocr_text_bbox", []),
            balloon_bbox,
            ordered_subregions,
            orientation,
            reasoner_settings=reasoner_settings,
        ) if subs else _empty_connected_visuals()
        updated["balloon_subregions"] = connected_visuals["connected_lobe_bboxes"]
        updated["connected_lobe_bboxes"] = connected_visuals["connected_lobe_bboxes"]
        updated["connected_lobe_polygons"] = [
            poly if poly is not None else []
            for poly in ordered_polygons
        ] if ordered_polygons else []
        
        # Se detectamos lobos reais, garantimos que o grupo seja tratado como tal para o renderer
        if len(updated["balloon_subregions"]) >= 2:
            count = len(updated["balloon_subregions"])
            updated["layout_profile"] = "connected_balloon"
            logger.info(f"DECISAO LAYOUT: Balao {text.get('id', 'N/A')} split em {count} lobos geometricos.")
            record_decision(
                stage="layout",
                action="split_balloon_subregions",
                reason="connected_lobes_detected",
                page=page_number,
                layer=layer_ref,
                text=text.get("text", ""),
                bbox=balloon_bbox,
                details={"count": count, "orientation": orientation, "layout_profile": updated["layout_profile"]},
            )
            updated["layout_group_size"] = max(updated["layout_group_size"], count)

        updated["connected_balloon_orientation"] = orientation
        updated["connected_text_groups"] = connected_visuals["connected_text_groups"]
        updated["connected_position_bboxes"] = connected_visuals["connected_position_bboxes"]
        updated["connected_focus_bboxes"] = connected_visuals["connected_position_bboxes"]
        updated["connected_detection_confidence"] = connected_visuals["connected_detection_confidence"]
        updated["connected_group_confidence"] = connected_visuals["connected_group_confidence"]
        updated["connected_position_confidence"] = connected_visuals["connected_position_confidence"]
        updated["connected_position_reasoner"] = connected_visuals["connected_position_reasoner"]
        updated["connected_reasoner_model"] = connected_visuals["connected_reasoner_model"]
        updated["connected_reasoner_notes"] = connected_visuals["connected_reasoner_notes"]
        updated["subregion_confidence"] = connected_visuals["connected_detection_confidence"]
        enriched_texts.append(updated)

    _apply_geometric_fallback_subregions(enriched_texts, page_result, page_image)

    updated_page = dict(page_result)
    updated_page["texts"] = enriched_texts
    updated_page.pop("_cached_image_bgr", None)
    return updated_page


def _group_looks_like_line_stack(group: list[dict], balloon_bbox: list[int]) -> bool:
    if len(group) < 2:
        return False

    ordered = sorted(
        group,
        key=lambda item: (
            int(item.get("bbox", [0, 0, 0, 0])[1]),
            int(item.get("bbox", [0, 0, 0, 0])[0]),
        ),
    )
    balloon_h = max(1, int(balloon_bbox[3]) - int(balloon_bbox[1]))
    gap_limit = max(42, int(balloon_h * 0.14))
    overlap_hits = 0

    for first, second in zip(ordered, ordered[1:]):
        a = [int(v) for v in first.get("bbox", [0, 0, 0, 0])]
        b = [int(v) for v in second.get("bbox", [0, 0, 0, 0])]
        if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
            return False
        overlap_x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        min_w = max(1, min(a[2] - a[0], b[2] - b[0]))
        vertical_gap = max(0, b[1] - a[3])
        center_dx = abs(((a[0] + a[2]) / 2.0) - ((b[0] + b[2]) / 2.0))
        if (
            overlap_x >= int(min_w * 0.35)
            and vertical_gap <= gap_limit
            and center_dx <= max(56, int(min_w * 0.35))
        ):
            overlap_hits += 1

    return overlap_hits >= max(1, len(ordered) - 1)


def _apply_geometric_fallback_subregions(
    texts: list[dict],
    page_result: dict | None = None,
    page_image: np.ndarray | None = None,
) -> None:
    """Funciona em dois modos:
      A) Multi-texto (group_size > 1): agrupa textos pelo balloon_bbox e usa
         a distribuição dos centros X para refinar o ponto de corte.
      B) Texto único (group_size == 1): se o balão é largo o suficiente
         (aspect >= 2.0), divide no centro — o renderer faz o split semântico.
    """
    page_result = page_result or {}
    groups: dict[tuple, list[dict]] = {}
    for text in texts:
        balloon = text.get("balloon_bbox")
        if not balloon or text.get("balloon_subregions"):
            continue
        tipo = text.get("tipo", "fala")
        if tipo not in {"fala", "pensamento", "narracao"}:
            continue
        key = (tipo, tuple(int(v) for v in balloon))
        groups.setdefault(key, []).append(text)

    for (tipo, bbox_tuple), group in groups.items():
        balloon = list(bbox_tuple)
        bw = max(1, balloon[2] - balloon[0])
        bh = max(1, balloon[3] - balloon[1])
        aspect = bw / float(bh)
        sample = group[0]
        sample_shape = str(sample.get("layout_shape") or classify_layout_shape(balloon, tipo))
        if not _can_try_connected_balloon_detection(
            sample,
            balloon_bbox=balloon,
            layout_shape=sample_shape,
            page_image=page_image,
        ):
            continue
        if _group_looks_like_line_stack(group, balloon):
            continue
        text_gap_subregions = _split_balloon_by_text_gap(
            balloon,
            [text.get("text_pixel_bbox") for text in group],
        )
        subregions: list[list[int]] = []

        if len(text_gap_subregions) >= 2:
            subregions = text_gap_subregions
        elif len(group) >= 2 and int(group[0].get("layout_group_size", 1)) > 1:
            # Modo A: multi-texto. Para balões pouco largos, só aceitamos o
            # fallback quando a separação dos textos for fortemente diagonal —
            # caso real de dois lobos conectados em "S"/diagonal.
            text_bboxes = [t.get("bbox", [0, 0, 0, 0]) for t in group]
            candidate_subregions = _geometric_fallback_subregions(text_bboxes, balloon)
            if aspect < 1.7:
                centers = [((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0) for bbox in text_bboxes]
                if len(centers) < 2:
                    continue
                first_cx, first_cy = centers[0]
                second_cx, second_cy = centers[-1]
                dx = abs(first_cx - second_cx)
                dy = abs(first_cy - second_cy)
                if dx < bw * 0.28 or dy < bh * 0.28:
                    continue
            if len(candidate_subregions) < 2:
                continue
            subregions = candidate_subregions
        elif len(group) == 1 and min(bw, bh) >= 120 and max(bw, bh) >= 300:
            # Modo B: texto único em balão grande.
            # Split geométrico direto só é seguro para balões realmente grandes.
            # Balões médios passam pela heurística de máscara/fill ratio para não
            # dividir balões simples só porque são largos.
            text_body = str(group[0].get("translated") or group[0].get("text") or "")
            aspect_gate_connected = page_image is None and (
                aspect >= 1.75
                and min(bw, bh) >= 160
                and max(bw, bh) >= 420
            )
            semantic_gate_connected = (
                page_image is not None
                and _balloon_region_looks_white(page_image, balloon)
                and aspect >= 1.65
                and min(bw, bh) >= 150
                and max(bw, bh) >= 300
                and _has_multiple_sentences(text_body)
            )
            is_likely_connected = aspect_gate_connected or semantic_gate_connected

            if not is_likely_connected and page_image is not None:
                # Heurística de preenchimento: elipse ~0.78. Lóbulos conectados ~0.45-0.65.
                mask_path = _resolve_mask_layer_path(page_result)
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        roi = mask[balloon[1]:balloon[3], balloon[0]:balloon[2]]
                        fill_ratio = np.count_nonzero(roi) / float(max(1, bw * bh))
                        if fill_ratio < 0.62:
                            is_likely_connected = True
            
            if is_likely_connected:
                subregions = _geometric_fallback_subregions(
                    [group[0].get("bbox", [0, 0, 0, 0])], balloon
                )
            else:
                continue

        if len(subregions) >= 2:
            connected_plan = _analyze_connected_subregions(subregions, balloon)
            for text in group:
                ordered_subregions = connected_plan.get("ordered_subregions", subregions)
                orientation = connected_plan.get("orientation", "")
                fallback_visuals = _derive_connected_visual_boxes(
                    None,
                    text.get("ocr_text_bbox") or text.get("bbox", [0, 0, 0, 0]),
                    balloon,
                    ordered_subregions,
                    orientation,
                )
                text["balloon_subregions"] = fallback_visuals["connected_lobe_bboxes"]
                text["connected_lobe_bboxes"] = fallback_visuals["connected_lobe_bboxes"]
                # Fallback geométrico não tem polígonos de contorno — preservar existentes ou vazio
                if not text.get("connected_lobe_polygons"):
                    text["connected_lobe_polygons"] = [[] for _ in fallback_visuals["connected_lobe_bboxes"]]
                text["connected_balloon_orientation"] = orientation
                text["connected_text_groups"] = fallback_visuals["connected_text_groups"]
                text["connected_position_bboxes"] = fallback_visuals["connected_position_bboxes"]
                text["connected_focus_bboxes"] = fallback_visuals["connected_position_bboxes"]
                text["connected_detection_confidence"] = fallback_visuals["connected_detection_confidence"]
                text["connected_group_confidence"] = fallback_visuals["connected_group_confidence"]
                text["connected_position_confidence"] = fallback_visuals["connected_position_confidence"]
                text["connected_position_reasoner"] = fallback_visuals["connected_position_reasoner"]
                text["connected_reasoner_model"] = fallback_visuals["connected_reasoner_model"]
                text["connected_reasoner_notes"] = fallback_visuals["connected_reasoner_notes"]
                text["subregion_confidence"] = fallback_visuals["connected_detection_confidence"]
                text["layout_profile"] = "connected_balloon"
                text["layout_group_size"] = max(int(text.get("layout_group_size", 1) or 1), len(text["balloon_subregions"]))


def _resolve_mask_layer_path(page_result: dict) -> Path:
    work_dir = Path(page_result.get("_work_dir", "."))
    image_layers = page_result.get("image_layers") or {}
    mask_rel = ((image_layers.get("mask") or {}).get("path")) or None
    if mask_rel:
        path = Path(mask_rel)
        return path if path.is_absolute() else work_dir / path

    page_number = page_result.get("numero")
    try:
        if page_number is not None:
            return work_dir / "layers" / "mask" / f"{int(page_number):03}.png"
    except (TypeError, ValueError):
        pass

    return work_dir / "layers" / "mask" / Path(page_result.get("arquivo_original", "")).name


def _split_balloon_by_text_gap(balloon_bbox: list[int], text_pixel_bboxes: list) -> list[list[int]]:
    normalized = []
    for bbox in text_pixel_bboxes:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        normalized.append([x1, y1, x2, y2])

    if len(normalized) < 2:
        return []

    bx1, by1, bx2, by2 = [int(v) for v in balloon_bbox]
    balloon_h = max(1, by2 - by1)
    ordered = sorted(normalized, key=lambda item: (item[1], item[0]))
    best_gap = 0
    best_pair: tuple[list[int], list[int]] | None = None
    for first, second in zip(ordered, ordered[1:]):
        gap = int(second[1]) - int(first[3])
        if gap > best_gap:
            best_gap = gap
            best_pair = (first, second)

    if best_pair is None or best_gap <= int(balloon_h * 0.20):
        return []

    top_bbox, bottom_bbox = best_pair
    split_y = int(round((top_bbox[3] + bottom_bbox[1]) / 2.0))
    split_y = max(by1 + 1, min(by2 - 1, split_y))
    top = [bx1, by1, bx2, split_y]
    bottom = [bx1, split_y, bx2, by2]
    if top[3] <= top[1] or bottom[3] <= bottom[1]:
        return []
    return [top, bottom]


def _empty_connected_visuals() -> dict:
    return {
        "connected_text_groups": [],
        "connected_lobe_bboxes": [],
        "connected_position_bboxes": [],
        "connected_detection_confidence": 0.0,
        "connected_group_confidence": 0.0,
        "connected_position_confidence": 0.0,
        "connected_position_reasoner": "heuristic",
        "connected_reasoner_model": "",
        "connected_reasoner_notes": "",
    }


def _coerce_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "off", "no", "none", "disabled"}


def _resolve_connected_reasoner_settings(page_result: dict) -> dict:
    raw = page_result.get("_connected_balloon_reasoner") or {}
    provider = str(
        raw.get("provider")
        or raw.get("mode")
        or os.environ.get("TRADUZAI_CONNECTED_BALLOON_REASONER", "ollama")
    ).strip().lower()
    enabled = provider == "ollama" and _coerce_bool(raw.get("enabled"), True)
    if provider in {"0", "false", "off", "none", "disabled"}:
        enabled = False
    return {
        "provider": "ollama" if enabled else "disabled",
        "enabled": enabled,
        "host": str(
            raw.get("host")
            or os.environ.get("TRADUZAI_CONNECTED_BALLOON_OLLAMA_HOST")
            or os.environ.get("OLLAMA_HOST")
            or OLLAMA_HOST
        ).strip(),
        "model": str(
            raw.get("model")
            or os.environ.get("TRADUZAI_CONNECTED_BALLOON_OLLAMA_MODEL")
            or "qwen2.5"
        ).strip(),
        "use_image": _coerce_bool(
            raw.get("use_image"),
            _coerce_bool(os.environ.get("TRADUZAI_CONNECTED_BALLOON_OLLAMA_USE_IMAGE"), True),
        ),
        "timeout_sec": int(raw.get("timeout_sec") or os.environ.get("TRADUZAI_CONNECTED_BALLOON_OLLAMA_TIMEOUT_SEC") or 120),
        "temperature": float(raw.get("temperature") or os.environ.get("TRADUZAI_CONNECTED_BALLOON_OLLAMA_TEMPERATURE") or 0.1),
    }


def _strip_json_fences(content: str) -> str:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    if cleaned:
        return cleaned
    return ""


def _call_ollama_json(
    model: str,
    system: str,
    user_msg: str,
    host: str,
    *,
    images: list[str] | None = None,
    temperature: float = 0.1,
    timeout: int = 90,
):
    user_payload = {"role": "user", "content": user_msg}
    if images:
        user_payload["images"] = images
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                user_payload,
            ],
            "stream": False,
            "options": {"temperature": float(temperature)},
            "format": "json",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=max(15, int(timeout))) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    content = _strip_json_fences(data.get("message", {}).get("content", ""))
    if not content:
        return {}
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise


def _pick_connected_reasoner_model(models: list[str], preferred: str) -> str:
    preferred = (preferred or "").strip()
    if preferred:
        picked = _pick_ollama_model(models, preferred)
        if picked:
            return picked
    for token in ("gemma4", "qwen2.5", "llava", "llama3.2-vision", "moondream", "minicpm-v"):
        for model in models:
            if token in model.lower():
                return model
    for model in models:
        lowered = model.lower()
        if "translator" in lowered or "cloud" in lowered:
            continue
        return model
    return _pick_ollama_model(models, preferred)


def _pick_connected_reasoner_models(models: list[str], preferred: str) -> list[str]:
    ordered: list[str] = []
    primary = _pick_connected_reasoner_model(models, preferred)
    if primary:
        ordered.append(primary)
    for token in ("qwen2.5", "gemma4", "llava", "llama3.2-vision", "moondream", "minicpm-v"):
        for model in models:
            if model in ordered:
                continue
            if token in model.lower():
                ordered.append(model)
    for model in models:
        if model not in ordered:
            ordered.append(model)
    return ordered


def _model_supports_inline_images(model_name: str) -> bool:
    lowered = (model_name or "").lower()
    return any(token in lowered for token in ("gemma4", "llava", "llama3.2-vision", "moondream", "minicpm-v"))


def _analyze_connected_subregions(
    subregions: list[list[int]],
    balloon_bbox: list[int],
    profile: str = "standard",
) -> dict:
    normalized = []
    for bbox in subregions or []:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        normalized.append([x1, y1, x2, y2])

    if len(normalized) < 2:
        return {
            "orientation": "",
            "ordered_subregions": normalized,
            "balloon_bbox": [int(v) for v in balloon_bbox] if balloon_bbox else [],
        }

    centers = [
        ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        for bbox in normalized
    ]
    dx = abs(centers[0][0] - centers[1][0])
    dy = abs(centers[0][1] - centers[1][1])

    if dx >= dy * 1.1:
        orientation = "left-right"
        ordered = sorted(normalized, key=lambda b: (((b[0] + b[2]) / 2.0), ((b[1] + b[3]) / 2.0)))
    elif dy >= dx * 1.1:
        orientation = "top-bottom"
        ordered = sorted(normalized, key=lambda b: (((b[1] + b[3]) / 2.0), ((b[0] + b[2]) / 2.0)))
    else:
        orientation = "diagonal"
        ordered = sorted(
            normalized,
            key=lambda b: (
                ((b[1] + b[3]) / 2.0) + ((b[0] + b[2]) / 2.0),
                ((b[1] + b[3]) / 2.0),
                ((b[0] + b[2]) / 2.0),
            ),
        )

    # TRAVA DE PROXIMIDADE:
    # splits geométricos válidos frequentemente encostam na costura (gap 0) ou
    # deixam um vão mínimo. O que realmente indica "mesmo lobo" aqui é
    # sobreposição relevante no eixo principal, não um gap pequeno.
    if len(ordered) == 2:
        should_merge = False
        overlap_min = 18 if profile in {"white_balloon", "connected_balloon"} else 12
        overlap_ratio = 0.12 if profile in {"white_balloon", "connected_balloon"} else 0.08
        diagonal_merge_factor = 0.24 if profile in {"white_balloon", "connected_balloon"} else 0.32
        if orientation == "top-bottom":
            overlap = max(0, ordered[0][3] - ordered[1][1])
            should_merge = overlap > max(overlap_min, int((balloon_bbox[3] - balloon_bbox[1]) * overlap_ratio))
        elif orientation == "left-right":
            overlap = max(0, ordered[0][2] - ordered[1][0])
            should_merge = overlap > max(overlap_min, int((balloon_bbox[2] - balloon_bbox[0]) * overlap_ratio))
        else:
            # Diagonal continua ambíguo; aqui a distância entre centros ainda é
            # um bom freio para não separar um balão único em dois quadrantes.
            c0 = ((ordered[0][0] + ordered[0][2]) / 2.0, (ordered[0][1] + ordered[0][3]) / 2.0)
            c1 = ((ordered[1][0] + ordered[1][2]) / 2.0, (ordered[1][1] + ordered[1][3]) / 2.0)
            dist = ((c0[0] - c1[0])**2 + (c0[1] - c1[1])**2)**0.5
            bw = max(1, balloon_bbox[2] - balloon_bbox[0])
            bh = max(1, balloon_bbox[3] - balloon_bbox[1])
            diag = (bw**2 + bh**2)**0.5
            should_merge = dist < diag * diagonal_merge_factor

        if should_merge:
            return {
                "orientation": "",
                "ordered_subregions": [balloon_bbox],
                "balloon_bbox": [int(v) for v in balloon_bbox] if balloon_bbox else [],
            }

    return {
        "orientation": orientation,
        "ordered_subregions": ordered,
        "balloon_bbox": [int(v) for v in balloon_bbox] if balloon_bbox else [],
    }


def _assign_group_boxes_to_subregions(
    group_bboxes: list[list[int]],
    ordered_subregions: list[list[int]],
) -> list[list[list[int]]]:
    if not group_bboxes or not ordered_subregions:
        return []
    groups = [[] for _ in ordered_subregions]
    centers = [
        ((sub[0] + sub[2]) / 2.0, (sub[1] + sub[3]) / 2.0)
        for sub in ordered_subregions
    ]
    for bbox in group_bboxes:
        gx1, gy1, gx2, gy2 = [int(v) for v in bbox]
        gcx = (gx1 + gx2) / 2.0
        gcy = (gy1 + gy2) / 2.0
        best_idx = 0
        best_score = float("-inf")
        for idx, sub in enumerate(ordered_subregions):
            sx1, sy1, sx2, sy2 = sub
            ix1 = max(gx1, sx1)
            iy1 = max(gy1, sy1)
            ix2 = min(gx2, sx2)
            iy2 = min(gy2, sy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            scx, scy = centers[idx]
            dist = ((gcx - scx) ** 2 + (gcy - scy) ** 2) ** 0.5
            score = inter - dist * 0.5
            if score > best_score:
                best_score = score
                best_idx = idx
        groups[best_idx].append([gx1, gy1, gx2, gy2])
    return groups


def _expand_group_to_focus_bbox(group_bbox: list[int], subregion: list[int]) -> list[int]:
    gx1, gy1, gx2, gy2 = [int(v) for v in group_bbox]
    sx1, sy1, sx2, sy2 = [int(v) for v in subregion]
    gw = max(1, gx2 - gx1)
    gh = max(1, gy2 - gy1)
    expand_x = max(6, int(gw * 0.08))
    expand_y = max(6, int(gh * 0.12))
    return [
        max(sx1, gx1 - expand_x),
        max(sy1, gy1 - expand_y),
        min(sx2, gx2 + expand_x),
        min(sy2, gy2 + expand_y),
    ]


def _expand_group_to_text_group_bbox(group_bbox: list[int], subregion: list[int]) -> list[int]:
    gx1, gy1, gx2, gy2 = [int(v) for v in group_bbox]
    sx1, sy1, sx2, sy2 = [int(v) for v in subregion]
    gw = max(1, gx2 - gx1)
    gh = max(1, gy2 - gy1)
    expand_x = max(4, int(gw * 0.04))
    expand_y = max(4, int(gh * 0.06))
    return [
        max(sx1, gx1 - expand_x),
        max(sy1, gy1 - expand_y),
        min(sx2, gx2 + expand_x),
        min(sy2, gy2 + expand_y),
    ]


def _fallback_connected_text_groups(
    ordered_subregions: list[list[int]],
    orientation: str,
) -> list[list[int]]:
    group_boxes: list[list[int]] = []
    for index, sub in enumerate(ordered_subregions):
        x1, y1, x2, y2 = [int(v) for v in sub]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if orientation == "left-right" and len(ordered_subregions) == 2:
            if index == 0:
                group = [
                    x1 + max(10, int(w * 0.08)),
                    y1 + max(10, int(h * 0.08)),
                    x2 - max(26, int(w * 0.22)),
                    y2 - max(30, int(h * 0.22)),
                ]
            else:
                group = [
                    x1 + max(24, int(w * 0.16)),
                    y1 + max(26, int(h * 0.18)),
                    x2 - max(10, int(w * 0.08)),
                    y2 - max(12, int(h * 0.10)),
                ]
            group_boxes.append(_shape_focus_bbox_for_lobe(group, sub, orientation, index, len(ordered_subregions)))
            continue
        group_boxes.append(
            [
                x1 + max(8, int(w * 0.08)),
                y1 + max(8, int(h * 0.08)),
                x2 - max(8, int(w * 0.08)),
                y2 - max(8, int(h * 0.08)),
            ]
        )
    return group_boxes


def _fallback_connected_focus_bboxes(
    ordered_subregions: list[list[int]],
    orientation: str,
) -> list[list[int]]:
    focus_boxes: list[list[int]] = []
    for index, sub in enumerate(ordered_subregions):
        x1, y1, x2, y2 = [int(v) for v in sub]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if orientation == "left-right" and len(ordered_subregions) == 2:
            if index == 0:
                focus = [x1 + max(10, int(w * 0.05)), y1 + max(10, int(h * 0.05)), x2 - max(24, int(w * 0.18)), y2 - max(20, int(h * 0.16))]
            else:
                focus = [x1 + max(24, int(w * 0.14)), y1 + max(20, int(h * 0.14)), x2 - max(10, int(w * 0.05)), y2 - max(10, int(h * 0.05))]
        else:
            focus = [x1 + max(8, int(w * 0.06)), y1 + max(8, int(h * 0.06)), x2 - max(8, int(w * 0.06)), y2 - max(8, int(h * 0.06))]
        focus_boxes.append(focus)
    return focus_boxes


def _shape_focus_bbox_for_lobe(
    focus_bbox: list[int],
    subregion: list[int],
    orientation: str,
    index: int,
    count: int,
) -> list[int]:
    fx1, fy1, fx2, fy2 = [int(v) for v in focus_bbox]
    sx1, sy1, sx2, sy2 = [int(v) for v in subregion]
    w = max(1, sx2 - sx1)
    h = max(1, sy2 - sy1)
    if orientation == "left-right" and count == 2:
        horizontal_trim = max(12, int(w * 0.12))
        vertical_trim = max(12, int(h * 0.14))
        if index == 0:
            fx2 = min(fx2, sx2 - horizontal_trim)
            fy2 = min(fy2, sy2 - vertical_trim)
        else:
            fx1 = max(fx1, sx1 + horizontal_trim)
            fy1 = max(fy1, sy1 + vertical_trim)
    if fx2 <= fx1:
        fx1, fx2 = sx1, sx2
    if fy2 <= fy1:
        fy1, fy2 = sy1, sy2
    return [fx1, fy1, fx2, fy2]


def _shape_text_group_bbox_for_lobe(
    group_bbox: list[int],
    subregion: list[int],
    orientation: str,
    index: int,
    count: int,
) -> list[int]:
    gx1, gy1, gx2, gy2 = [int(v) for v in group_bbox]
    sx1, sy1, sx2, sy2 = [int(v) for v in subregion]
    w = max(1, sx2 - sx1)
    h = max(1, sy2 - sy1)
    if orientation == "left-right" and count == 2:
        horizontal_trim = max(8, int(w * 0.08))
        vertical_trim = max(14, int(h * 0.18))
        if index == 0:
            gx2 = min(gx2, sx2 - horizontal_trim)
            gy2 = min(gy2, sy2 - vertical_trim)
        else:
            gx1 = max(gx1, sx1 + horizontal_trim)
            gy1 = max(gy1, sy1 + vertical_trim)
    if gx2 <= gx1:
        gx1, gx2 = sx1, sx2
    if gy2 <= gy1:
        gy1, gy2 = sy1, sy2
    return [gx1, gy1, gx2, gy2]


def _derive_connected_text_groups(
    image_bgr: np.ndarray | None,
    ocr_text_bbox: list[int],
    ordered_subregions: list[list[int]],
    orientation: str,
) -> tuple[list[list[int]], float]:
    if image_bgr is None or len(ordered_subregions) < 2:
        return _fallback_connected_text_groups(ordered_subregions, orientation), 0.28

    components = _extract_text_cluster_components(image_bgr, ocr_text_bbox)
    merged = _merge_text_cluster_components(components)
    if len(merged) < 2:
        return _fallback_connected_text_groups(ordered_subregions, orientation), 0.28

    merged = sorted(merged, key=lambda item: item["area"], reverse=True)
    grouped = _assign_group_boxes_to_subregions([item["bbox"] for item in merged], ordered_subregions)
    if len(grouped) != len(ordered_subregions) or any(not group for group in grouped):
        return _fallback_connected_text_groups(ordered_subregions, orientation), 0.28

    group_boxes: list[list[int]] = []
    confidence_terms: list[float] = []
    for index, (group, subregion) in enumerate(zip(grouped, ordered_subregions)):
        union = group[0]
        for bbox in group[1:]:
            union = _union_bbox(union, bbox)
        group_box = _shape_text_group_bbox_for_lobe(
            _expand_group_to_text_group_bbox(union, subregion),
            subregion,
            orientation,
            index,
            len(ordered_subregions),
        )
        group_boxes.append(group_box)
        sx1, sy1, sx2, sy2 = [int(v) for v in subregion]
        gcx = (group_box[0] + group_box[2]) / 2.0
        gcy = (group_box[1] + group_box[3]) / 2.0
        scx = (sx1 + sx2) / 2.0
        scy = (sy1 + sy2) / 2.0
        drift_x = abs(gcx - scx) / float(max(1, sx2 - sx1))
        drift_y = abs(gcy - scy) / float(max(1, sy2 - sy1))
        confidence_terms.append(max(0.0, 1.0 - (drift_x * 0.8 + drift_y * 0.8)))
    confidence = round(max(0.55, min(0.98, sum(confidence_terms) / max(1, len(confidence_terms)))), 3)
    return group_boxes, confidence


def _derive_connected_position_bboxes(
    connected_text_groups: list[list[int]],
    ordered_subregions: list[list[int]],
    orientation: str,
) -> list[list[int]]:
    if len(connected_text_groups) != len(ordered_subregions):
        return _fallback_connected_focus_bboxes(ordered_subregions, orientation)

    position_boxes: list[list[int]] = []
    for index, (group_box, subregion) in enumerate(zip(connected_text_groups, ordered_subregions)):
        focus = _expand_group_to_focus_bbox(group_box, subregion)
        position_boxes.append(
            _shape_focus_bbox_for_lobe(
                focus,
                subregion,
                orientation,
                index,
                len(ordered_subregions),
            )
        )
    return position_boxes


def _bbox_area(bbox: list[int]) -> int:
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _bbox_inside(inner: list[int], outer: list[int]) -> bool:
    return (
        int(inner[0]) >= int(outer[0])
        and int(inner[1]) >= int(outer[1])
        and int(inner[2]) <= int(outer[2])
        and int(inner[3]) <= int(outer[3])
        and int(inner[2]) > int(inner[0])
        and int(inner[3]) > int(inner[1])
    )


def _bbox_to_local(bbox: list[int], crop_bbox: list[int]) -> list[int]:
    return [
        int(bbox[0]) - int(crop_bbox[0]),
        int(bbox[1]) - int(crop_bbox[1]),
        int(bbox[2]) - int(crop_bbox[0]),
        int(bbox[3]) - int(crop_bbox[1]),
    ]


def _bbox_to_absolute(bbox: list[int], crop_bbox: list[int]) -> list[int]:
    return [
        int(bbox[0]) + int(crop_bbox[0]),
        int(bbox[1]) + int(crop_bbox[1]),
        int(bbox[2]) + int(crop_bbox[0]),
        int(bbox[3]) + int(crop_bbox[1]),
    ]


def _bbox_center_point(bbox: list[int]) -> list[int]:
    return [
        int(round((int(bbox[0]) + int(bbox[2])) / 2.0)),
        int(round((int(bbox[1]) + int(bbox[3])) / 2.0)),
    ]


def _clamp_point_inside_bbox(point: list[int], bbox: list[int], *, pad_x: int = 0, pad_y: int = 0) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return [
        max(x1 + int(pad_x), min(x2 - int(pad_x), int(point[0]))),
        max(y1 + int(pad_y), min(y2 - int(pad_y), int(point[1]))),
    ]


def _blend_anchor_points(a: list[int], b: list[int], weight_a: float, weight_b: float) -> list[int]:
    total = max(0.001, float(weight_a) + float(weight_b))
    return [
        int(round((a[0] * float(weight_a) + b[0] * float(weight_b)) / total)),
        int(round((a[1] * float(weight_a) + b[1] * float(weight_b)) / total)),
    ]


def _encode_png_base64(image_bgr: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _draw_reasoner_boxes(
    image_bgr: np.ndarray,
    crop_bbox: list[int],
    *,
    ocr_text_bbox: list[int],
    text_groups: list[list[int]],
    lobe_bboxes: list[list[int]],
    position_bboxes: list[list[int]],
) -> np.ndarray:
    overlay = image_bgr.copy()
    items = []
    if ocr_text_bbox:
        items.append((ocr_text_bbox, (255, 0, 0), "OCR"))
    for index, bbox in enumerate(text_groups):
        items.append((bbox, (0, 200, 0), f"G{index}"))
    for index, bbox in enumerate(lobe_bboxes):
        items.append((bbox, (0, 0, 255), f"L{index}"))
    for index, bbox in enumerate(position_bboxes):
        items.append((bbox, (0, 165, 255), f"P{index}"))

    for bbox, color, label in items:
        local = _bbox_to_local(bbox, crop_bbox)
        x1, y1, x2, y2 = [int(v) for v in local]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            label,
            (max(0, x1 + 2), max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _build_connected_reasoner_crop_payload(
    image_bgr: np.ndarray,
    ocr_text_bbox: list[int],
    balloon_bbox: list[int],
    text_groups: list[list[int]],
    lobe_bboxes: list[list[int]],
    heuristic_position_bboxes: list[list[int]],
) -> tuple[list[int], list[str], dict]:
    height, width = image_bgr.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in balloon_bbox]
    pad_x = max(14, int((bx2 - bx1) * 0.06))
    pad_y = max(14, int((by2 - by1) * 0.08))
    crop_bbox = [
        max(0, bx1 - pad_x),
        max(0, by1 - pad_y),
        min(width, bx2 + pad_x),
        min(height, by2 + pad_y),
    ]
    crop = image_bgr[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]].copy()
    overlay = _draw_reasoner_boxes(
        crop,
        crop_bbox,
        ocr_text_bbox=ocr_text_bbox,
        text_groups=text_groups,
        lobe_bboxes=lobe_bboxes,
        position_bboxes=heuristic_position_bboxes,
    )
    local_payload = {
        "crop_bbox": crop_bbox,
        "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
        "ocr_text_bbox": _bbox_to_local(ocr_text_bbox, crop_bbox) if ocr_text_bbox else [],
        "text_groups": [_bbox_to_local(bbox, crop_bbox) for bbox in text_groups],
        "lobe_bboxes": [_bbox_to_local(bbox, crop_bbox) for bbox in lobe_bboxes],
        "heuristic_position_bboxes": [_bbox_to_local(bbox, crop_bbox) for bbox in heuristic_position_bboxes],
    }
    overlay_encoded = _encode_png_base64(overlay)
    if overlay_encoded:
        encoded_images = [overlay_encoded]
    else:
        encoded_images = []
    return crop_bbox, encoded_images, local_payload


def _build_reasoner_anchor_candidates(
    local_text_groups: list[list[int]],
    local_lobes: list[list[int]],
    local_heuristic_boxes: list[list[int]],
    orientation: str,
) -> list[list[dict]]:
    candidates_by_lobe: list[list[dict]] = []
    for index, (group_bbox, lobe_bbox, heuristic_bbox) in enumerate(zip(local_text_groups, local_lobes, local_heuristic_boxes)):
        lobe_w = max(1, int(lobe_bbox[2]) - int(lobe_bbox[0]))
        lobe_h = max(1, int(lobe_bbox[3]) - int(lobe_bbox[1]))
        pad_x = max(8, int(lobe_w * 0.08))
        pad_y = max(8, int(lobe_h * 0.08))
        group_center = _bbox_center_point(group_bbox)
        lobe_center = _bbox_center_point(lobe_bbox)
        heuristic_center = _bbox_center_point(heuristic_bbox)
        outer_dx = max(10, int(lobe_w * 0.08))
        outer_dy = max(10, int(lobe_h * 0.08))

        candidates: list[tuple[str, list[int]]] = [
            ("heuristic-center", heuristic_center),
            ("group-center", group_center),
            ("balanced", _blend_anchor_points(group_center, lobe_center, 0.72, 0.28)),
        ]
        if orientation == "left-right" and len(local_lobes) == 2:
            if index == 0:
                candidates.extend(
                    [
                        (
                            "outer-upper",
                            [
                                min(group_center[0], heuristic_center[0]) - outer_dx,
                                min(group_center[1], heuristic_center[1]) - outer_dy,
                            ],
                        ),
                        (
                            "outer-left",
                            [
                                min(group_center[0], heuristic_center[0]) - outer_dx,
                                int(round((group_center[1] + heuristic_center[1]) / 2.0)),
                            ],
                        ),
                    ]
                )
            else:
                candidates.extend(
                    [
                        (
                            "outer-lower",
                            [
                                max(group_center[0], heuristic_center[0]) + outer_dx,
                                max(group_center[1], heuristic_center[1]) + outer_dy,
                            ],
                        ),
                        (
                            "outer-right",
                            [
                                max(group_center[0], heuristic_center[0]) + outer_dx,
                                int(round((group_center[1] + heuristic_center[1]) / 2.0)),
                            ],
                        ),
                    ]
                )
        normalized = []
        seen_labels = set()
        for label, point in candidates:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            description = ""
            if label == "heuristic-center":
                description = "baseline atual; use so se estiver claramente melhor que as outras opcoes"
            elif label == "group-center":
                description = "segue o centro do grupo verde"
            elif label == "balanced":
                description = "compromisso entre centro do grupo verde e centro do lobo"
            elif label == "outer-upper":
                description = "puxa para fora e para cima; costuma parecer mais humano no lobo esquerdo"
            elif label == "outer-left":
                description = "puxa para fora na horizontal sem subir tanto"
            elif label == "outer-lower":
                description = "puxa para fora e para baixo; costuma parecer mais humano no lobo direito"
            elif label == "outer-right":
                description = "puxa para fora na horizontal sem descer tanto"
            normalized.append(
                {
                    "label": label,
                    "point": _clamp_point_inside_bbox(point, lobe_bbox, pad_x=pad_x, pad_y=pad_y),
                    "description": description,
                    "scale_x": (
                        1.0 if label == "heuristic-center"
                        else 0.98 if label in {"group-center", "balanced"}
                        else 0.95 if label in {"outer-left", "outer-right"}
                        else 0.93
                    ),
                    "scale_y": (
                        1.0 if label == "heuristic-center"
                        else 0.98 if label in {"group-center", "balanced"}
                        else 0.95 if label in {"outer-left", "outer-right"}
                        else 0.91
                    ),
                }
            )
        candidates_by_lobe.append(normalized)
    return candidates_by_lobe


def _extract_reasoner_position_bboxes(payload) -> list[list[int]]:
    if isinstance(payload, dict):
        candidates = (
            payload.get("position_bboxes"),
            payload.get("boxes"),
            payload.get("positions"),
        )
        for candidate in candidates:
            if isinstance(candidate, list):
                payload = candidate
                break
    if not isinstance(payload, list):
        return []
    normalized: list[list[int]] = []
    for bbox in payload:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return []
        try:
            normalized.append([int(round(float(v))) for v in bbox])
        except Exception:
            return []
    return normalized


def _validate_reasoner_position_bboxes(
    local_boxes: list[list[int]],
    local_lobes: list[list[int]],
    local_heuristic_boxes: list[list[int]],
    orientation: str,
    local_text_groups: list[list[int]] | None = None,
) -> bool:
    if len(local_boxes) != len(local_lobes) or len(local_boxes) != len(local_heuristic_boxes):
        return False
    for index, (bbox, lobe, heuristic) in enumerate(zip(local_boxes, local_lobes, local_heuristic_boxes)):
        if not _bbox_inside(bbox, lobe):
            return False
        heuristic_area = max(1, _bbox_area(heuristic))
        bbox_area = max(1, _bbox_area(bbox))
        heuristic_w = max(1, int(heuristic[2]) - int(heuristic[0]))
        heuristic_h = max(1, int(heuristic[3]) - int(heuristic[1]))
        bbox_w = max(1, int(bbox[2]) - int(bbox[0]))
        bbox_h = max(1, int(bbox[3]) - int(bbox[1]))
        if bbox_area < max(240, int(heuristic_area * 0.45)):
            return False
        if bbox_area > int(heuristic_area * 1.2):
            return False
        if not (heuristic_w * 0.82 <= bbox_w <= heuristic_w * 1.18):
            return False
        if not (heuristic_h * 0.82 <= bbox_h <= heuristic_h * 1.18):
            return False
        if orientation == "left-right" and len(local_boxes) == 2:
            box_cx = (bbox[0] + bbox[2]) / 2.0
            lobe_cx = (lobe[0] + lobe[2]) / 2.0
            if index == 0 and box_cx > lobe_cx + max(8.0, (lobe[2] - lobe[0]) * 0.12):
                return False
            if index == 1 and box_cx < lobe_cx - max(8.0, (lobe[2] - lobe[0]) * 0.12):
                return False
    if (
        orientation == "left-right"
        and len(local_boxes) == 2
        and isinstance(local_text_groups, list)
        and len(local_text_groups) == 2
    ):
        left_group_cy = (local_text_groups[0][1] + local_text_groups[0][3]) / 2.0
        right_group_cy = (local_text_groups[1][1] + local_text_groups[1][3]) / 2.0
        left_box_cy = (local_boxes[0][1] + local_boxes[0][3]) / 2.0
        right_box_cy = (local_boxes[1][1] + local_boxes[1][3]) / 2.0
        group_delta_y = right_group_cy - left_group_cy
        box_delta_y = right_box_cy - left_box_cy
        if abs(group_delta_y) >= 10.0:
            min_preserved_delta = max(8.0, abs(group_delta_y) * 0.25)
            if group_delta_y > 0 and box_delta_y < min_preserved_delta:
                return False
            if group_delta_y < 0 and box_delta_y > -min_preserved_delta:
                return False
    return True


def _extract_reasoner_anchor_points(payload) -> list[list[int]]:
    if not isinstance(payload, dict):
        return []
    anchors = payload.get("anchor_points")
    if not isinstance(anchors, list):
        return []
    normalized: list[list[int]] = []
    for point in anchors:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return []
        try:
            normalized.append([int(round(float(point[0]))), int(round(float(point[1])))])
        except Exception:
            return []
    return normalized


def _extract_reasoner_selected_anchor_labels(payload) -> list[str]:
    if not isinstance(payload, dict):
        return []
    labels = payload.get("selected_anchor_labels")
    if not isinstance(labels, list):
        return []
    normalized: list[str] = []
    for label in labels:
        if label is None:
            return []
        normalized.append(str(label).strip())
    return normalized


def _extract_reasoner_selected_anchor_indexes(payload) -> list[int]:
    if not isinstance(payload, dict):
        return []
    indexes = payload.get("selected_anchor_indexes")
    if not isinstance(indexes, list):
        return []
    normalized: list[int] = []
    for index in indexes:
        try:
            normalized.append(int(index))
        except Exception:
            return []
    return normalized


def _position_bbox_from_anchor(
    anchor: list[int],
    heuristic_bbox: list[int],
    lobe_bbox: list[int],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> list[int]:
    heuristic_w = max(1, int(heuristic_bbox[2]) - int(heuristic_bbox[0]))
    heuristic_h = max(1, int(heuristic_bbox[3]) - int(heuristic_bbox[1]))
    hw = max(1, int(round(heuristic_w * float(scale_x))))
    hh = max(1, int(round(heuristic_h * float(scale_y))))
    lx1, ly1, lx2, ly2 = [int(v) for v in lobe_bbox]
    ax, ay = [int(v) for v in anchor]
    x1 = int(round(ax - (hw / 2.0)))
    y1 = int(round(ay - (hh / 2.0)))
    x2 = x1 + hw
    y2 = y1 + hh
    if x1 < lx1:
        x2 += lx1 - x1
        x1 = lx1
    if x2 > lx2:
        x1 -= x2 - lx2
        x2 = lx2
    if y1 < ly1:
        y2 += ly1 - y1
        y1 = ly1
    if y2 > ly2:
        y1 -= y2 - ly2
        y2 = ly2
    x1 = max(lx1, x1)
    y1 = max(ly1, y1)
    x2 = min(lx2, x2)
    y2 = min(ly2, y2)
    return [int(x1), int(y1), int(x2), int(y2)]


def _refine_connected_position_bboxes_with_ollama(
    image_bgr: np.ndarray | None,
    ocr_text_bbox: list[int],
    balloon_bbox: list[int],
    connected_text_groups: list[list[int]],
    connected_lobe_bboxes: list[list[int]],
    heuristic_position_bboxes: list[list[int]],
    orientation: str,
    reasoner_settings: dict | None,
) -> dict | None:
    if (
        image_bgr is None
        or not reasoner_settings
        or not reasoner_settings.get("enabled")
        or reasoner_settings.get("provider") != "ollama"
        or len(connected_text_groups) != 2
        or len(connected_lobe_bboxes) != 2
        or len(heuristic_position_bboxes) != 2
    ):
        return None

    host = str(reasoner_settings.get("host", OLLAMA_HOST) or OLLAMA_HOST)
    status = reasoner_settings.get("_ollama_status")
    if not isinstance(status, dict):
        status = _check_ollama(host)
        reasoner_settings["_ollama_status"] = status
    if not status.get("running") or not status.get("models"):
        return None

    candidate_models = _pick_connected_reasoner_models(
        list(status.get("models") or []),
        str(reasoner_settings.get("model", "") or ""),
    )
    if not candidate_models:
        return None

    crop_bbox, encoded_images, local_payload = _build_connected_reasoner_crop_payload(
        image_bgr,
        ocr_text_bbox,
        balloon_bbox,
        connected_text_groups,
        connected_lobe_bboxes,
        heuristic_position_bboxes,
    )
    system_prompt = (
        "Voce e um compositor especialista em baloes conectados de manga. "
        "Recebera um crop do balao e um overlay com caixas: azul=OCR bruto, "
        "verde=grupos reais de texto, vermelho=lobos reais do balao, "
        "laranja=caixas de posicao heuristicas. Sua tarefa e ajustar somente as "
        "caixas laranja para ficar com alinhamento humano. Priorize mover o "
        "centro do texto, nao redimensionar a caixa. Regras obrigatorias: "
        "mantenha exatamente 2 caixas; cada caixa deve ficar 100% dentro do seu "
        "lobo vermelho correspondente; nao atravesse a costura entre os lobos; "
        "preserve a leitura diagonal natural; preserve o stagger vertical "
        "revelado pelos grupos verdes; o lobo esquerdo tende para "
        "esquerda/topo, o direito tende para direita/parte baixa, mas siga a "
        "forma real do balao e os grupos verdes. O tamanho deve ficar quase igual "
        "ao laranja atual. Responda apenas JSON. Se houver anchor_candidates, "
        "prefira selecionar labels ou indexes dessas ancoras em vez de inventar "
        "caixas livres. Evite heuristic-center se houver uma opcao mais humana "
        "e ainda segura. Use anchor_points ou position_bboxes apenas se realmente "
        "necessario."
    )
    user_payload = {
        "orientation": orientation,
        "crop_size": local_payload["crop_size"],
        "ocr_text_bbox": local_payload["ocr_text_bbox"],
        "text_groups": local_payload["text_groups"],
        "text_group_centers": [
            [
                round((bbox[0] + bbox[2]) / 2.0, 2),
                round((bbox[1] + bbox[3]) / 2.0, 2),
            ]
            for bbox in local_payload["text_groups"]
        ],
        "stagger_hint": {
            "right_minus_left_dy": round(
                (
                    ((local_payload["text_groups"][1][1] + local_payload["text_groups"][1][3]) / 2.0)
                    - ((local_payload["text_groups"][0][1] + local_payload["text_groups"][0][3]) / 2.0)
                ),
                2,
            ) if len(local_payload["text_groups"]) == 2 else 0.0,
            "rule": "if positive, the right box center must stay lower than the left box center",
        },
        "lobe_bboxes": local_payload["lobe_bboxes"],
        "heuristic_position_bboxes": local_payload["heuristic_position_bboxes"],
        "decision_rule": (
            "escolha uma opcao por lobo. prefira outer-upper no lobo esquerdo e "
            "outer-lower no lobo direito quando isso ainda parecer centralizado, "
            "seguro e coerente com os grupos verdes. use heuristic-center apenas "
            "se ele estiver claramente melhor que as alternativas."
        ),
        "response_schema": {
            "selected_anchor_labels": ["heuristic-center", "heuristic-center"],
            "selected_anchor_indexes": [0, 0],
            "anchor_points": [[0, 0], [0, 0]],
            "position_bboxes": [[0, 0, 0, 0], [0, 0, 0, 0]],
            "confidence": 0.0,
            "notes": "curta explicacao",
        },
    }
    local_lobes = local_payload["lobe_bboxes"]
    local_heuristic_boxes = local_payload["heuristic_position_bboxes"]
    anchor_candidates = _build_reasoner_anchor_candidates(
        local_payload["text_groups"],
        local_lobes,
        local_heuristic_boxes,
        orientation,
    )
    for model in candidate_models:
        images = encoded_images if reasoner_settings.get("use_image", True) and _model_supports_inline_images(model) else None
        prefer_anchor_labels = not bool(images)
        prompt_anchor_candidates = anchor_candidates
        if prefer_anchor_labels:
            prompt_anchor_candidates = [
                [candidate for candidate in candidates if candidate.get("label") != "heuristic-center"] or candidates
                for candidates in anchor_candidates
            ]
        user_payload["anchor_candidates"] = prompt_anchor_candidates
        user_payload["response_schema"]["selected_anchor_labels"] = [
            candidates[0]["label"] if candidates else "balanced"
            for candidates in prompt_anchor_candidates
        ]
        user_payload["response_schema"]["selected_anchor_indexes"] = [0 for _ in prompt_anchor_candidates]
        user_payload["preferred_response"] = (
            "selected_anchor_labels"
            if prefer_anchor_labels
            else "anchor_points"
        )
        try:
            response = _call_ollama_json(
                model,
                system_prompt,
                json.dumps(user_payload, ensure_ascii=False),
                host,
                images=images,
                temperature=float(reasoner_settings.get("temperature", 0.1) or 0.1),
                timeout=int(reasoner_settings.get("timeout_sec", 90) or 90),
            )
        except Exception:
            continue

        notes_value = str(response.get("notes", "") if isinstance(response, dict) else "")
        local_boxes = []
        selected_labels = _extract_reasoner_selected_anchor_labels(response)
        if len(selected_labels) == len(prompt_anchor_candidates):
            resolved_points = []
            resolved_specs = []
            for label, candidates in zip(selected_labels, prompt_anchor_candidates):
                spec = next((candidate for candidate in candidates if candidate["label"] == label), None)
                if spec is None:
                    resolved_points = []
                    break
                resolved_points.append(spec["point"])
                resolved_specs.append(spec)
            if len(resolved_points) == len(local_lobes):
                local_boxes = [
                    _position_bbox_from_anchor(
                        anchor,
                        heuristic_bbox,
                        lobe_bbox,
                        scale_x=float(spec.get("scale_x", 1.0) or 1.0),
                        scale_y=float(spec.get("scale_y", 1.0) or 1.0),
                    )
                    for anchor, heuristic_bbox, lobe_bbox, spec in zip(
                        resolved_points,
                        local_heuristic_boxes,
                        local_lobes,
                        resolved_specs,
                    )
                ]
                if not notes_value:
                    notes_value = "selected_anchor_labels=" + ",".join(selected_labels)

        if not local_boxes:
            selected_indexes = _extract_reasoner_selected_anchor_indexes(response)
            if len(selected_indexes) == len(prompt_anchor_candidates):
                resolved_points = []
                resolved_specs = []
                for chosen_index, candidates in zip(selected_indexes, prompt_anchor_candidates):
                    if 0 <= chosen_index < len(candidates):
                        resolved_specs.append(candidates[chosen_index])
                        resolved_points.append(candidates[chosen_index]["point"])
                    else:
                        resolved_points = []
                        break
                if len(resolved_points) == len(local_lobes):
                    local_boxes = [
                        _position_bbox_from_anchor(
                            anchor,
                            heuristic_bbox,
                            lobe_bbox,
                            scale_x=float(spec.get("scale_x", 1.0) or 1.0),
                            scale_y=float(spec.get("scale_y", 1.0) or 1.0),
                        )
                        for anchor, heuristic_bbox, lobe_bbox, spec in zip(
                            resolved_points,
                            local_heuristic_boxes,
                            local_lobes,
                            resolved_specs,
                        )
                    ]
                    if not notes_value:
                        notes_value = "selected_anchor_indexes=" + ",".join(str(i) for i in selected_indexes)

        anchor_points = _extract_reasoner_anchor_points(response)
        if not local_boxes and len(anchor_points) == len(local_lobes):
            local_boxes = [
                _position_bbox_from_anchor(anchor, heuristic_bbox, lobe_bbox)
                for anchor, heuristic_bbox, lobe_bbox in zip(
                    anchor_points,
                    local_heuristic_boxes,
                    local_lobes,
                )
            ]
        if not local_boxes:
            local_boxes = _extract_reasoner_position_bboxes(response)
        if not _validate_reasoner_position_bboxes(
            local_boxes,
            local_lobes,
            local_heuristic_boxes,
            orientation,
            local_text_groups=local_payload["text_groups"],
        ):
            absolute_boxes = _extract_reasoner_position_bboxes(response)
            translated_local_boxes = [_bbox_to_local(bbox, crop_bbox) for bbox in absolute_boxes]
            if _validate_reasoner_position_bboxes(
                translated_local_boxes,
                local_lobes,
                local_heuristic_boxes,
                orientation,
                local_text_groups=local_payload["text_groups"],
            ):
                local_boxes = translated_local_boxes
            else:
                continue

        absolute_boxes = [_bbox_to_absolute(bbox, crop_bbox) for bbox in local_boxes]
        reasoner_settings["_resolved_model"] = model
        return {
            "position_bboxes": absolute_boxes,
            "confidence": float(response.get("confidence", 0.88) if isinstance(response, dict) else 0.88),
            "source": "ollama",
            "model": model,
            "notes": notes_value,
        }
    return None


def _derive_connected_visual_boxes(
    image_bgr: np.ndarray | None,
    ocr_text_bbox: list[int],
    balloon_bbox: list[int],
    ordered_subregions: list[list[int]],
    orientation: str,
    reasoner_settings: dict | None = None,
) -> dict:
    if len(ordered_subregions) < 2:
        return _empty_connected_visuals()

    normalized_lobes = [[int(v) for v in bbox] for bbox in ordered_subregions]
    detection_confidence = round(_score_subregion_quality(normalized_lobes, balloon_bbox), 3)
    text_groups, group_confidence = _derive_connected_text_groups(
        image_bgr,
        ocr_text_bbox,
        normalized_lobes,
        orientation,
    )
    position_bboxes = _derive_connected_position_bboxes(text_groups, normalized_lobes, orientation)
    position_confidence = round(min(1.0, detection_confidence * 0.6 + group_confidence * 0.4), 3)
    position_reasoner = "heuristic"
    reasoner_model = ""
    reasoner_notes = ""
    # Otimização: se a heurística já é muito confiável, não precisamos do Ollama (economiza ~30s por bloco)
    if position_confidence >= 0.88:
        return {
            "connected_text_groups": text_groups,
            "connected_lobe_bboxes": normalized_lobes,
            "connected_position_bboxes": position_bboxes,
            "connected_detection_confidence": detection_confidence,
            "connected_group_confidence": group_confidence,
            "connected_position_confidence": position_confidence,
            "connected_position_reasoner": "heuristic",
            "connected_reasoner_model": "",
            "connected_reasoner_notes": "Heurística de alta confiança (>88%)",
        }

    reasoned = _refine_connected_position_bboxes_with_ollama(
        image_bgr,
        ocr_text_bbox,
        balloon_bbox,
        text_groups,
        normalized_lobes,
        position_bboxes,
        orientation,
        reasoner_settings,
    )
    if reasoned:
        position_bboxes = [[int(v) for v in bbox] for bbox in reasoned.get("position_bboxes", position_bboxes)]
        llm_confidence = float(reasoned.get("confidence", position_confidence) or position_confidence)
        position_confidence = round(
            min(1.0, detection_confidence * 0.3 + group_confidence * 0.25 + llm_confidence * 0.45),
            3,
        )
        position_reasoner = "ollama"
        reasoner_model = str(reasoned.get("model", "") or "")
        reasoner_notes = str(reasoned.get("notes", "") or "")
    return {
        "connected_text_groups": text_groups,
        "connected_lobe_bboxes": normalized_lobes,
        "connected_position_bboxes": position_bboxes,
        "connected_detection_confidence": detection_confidence,
        "connected_group_confidence": group_confidence,
        "connected_position_confidence": position_confidence,
        "connected_position_reasoner": position_reasoner,
        "connected_reasoner_model": reasoner_model,
        "connected_reasoner_notes": reasoner_notes,
    }


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


def _normalize_bubble_regions(page_result: dict) -> list[dict]:
    normalized: list[dict] = []
    for item in page_result.get("_bubble_regions") or []:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        normalized.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(item.get("confidence", 0.0) or 0.0),
            }
        )
    return normalized


def _select_bubble_region_for_bbox(seed_bbox: list[int], bubble_regions: list[dict]) -> list[int] | None:
    if not isinstance(seed_bbox, (list, tuple)) or len(seed_bbox) != 4 or not bubble_regions:
        return None

    sx1, sy1, sx2, sy2 = [int(v) for v in seed_bbox]
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    seed_area = float(max(1, (sx2 - sx1) * (sy2 - sy1)))
    seed_center = _bbox_center_point([sx1, sy1, sx2, sy2])
    best_bbox: list[int] | None = None
    best_score = float("-inf")

    for item in bubble_regions:
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        if bx2 <= bx1 or by2 <= by1:
            continue

        ix1 = max(sx1, bx1)
        iy1 = max(sy1, by1)
        ix2 = min(sx2, bx2)
        iy2 = min(sy2, by2)
        intersection = float(max(0, ix2 - ix1) * max(0, iy2 - iy1))
        overlap_ratio = intersection / seed_area
        center_inside = bx1 <= seed_center[0] <= bx2 and by1 <= seed_center[1] <= by2
        contains_seed = (
            bx1 <= sx1 + 8
            and by1 <= sy1 + 12
            and bx2 >= sx2 - 8
            and by2 >= sy2 - 12
        )
        if intersection <= 0.0 and not center_inside and not contains_seed:
            continue

        region_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
        area_ratio = region_area / seed_area
        if area_ratio > 28.0:
            continue

        region_center = _bbox_center_point([bx1, by1, bx2, by2])
        norm_dx = abs(float(region_center[0] - seed_center[0])) / float(max(1, bx2 - bx1))
        norm_dy = abs(float(region_center[1] - seed_center[1])) / float(max(1, by2 - by1))

        score = overlap_ratio * 8.0
        if contains_seed:
            score += 3.4
        if center_inside:
            score += 1.1
        if area_ratio >= 1.05:
            score += min(2.8, (area_ratio - 1.0) * 0.34)
        else:
            score -= (1.0 - area_ratio) * 4.0
        score -= norm_dx * 2.2
        score -= norm_dy * 2.8
        score += float(item.get("confidence", 0.0) or 0.0) * 0.35

        if score > best_score:
            best_score = score
            best_bbox = [bx1, by1, bx2, by2]

    if best_score < 1.0:
        return None
    return best_bbox


def _expand_with_margin(bbox: list[int], image_width: int, image_height: int, margin: int = 2) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(image_width, x2 + margin),
        min(image_height, y2 + margin),
    ]


def _reorder_polygons_to_match_subregions(
    original_subs: list[list[int]],
    ordered_subs: list[list[int]],
    polygons: list,
) -> list:
    """Reordena polígonos para combinar com a nova ordem de ordered_subs.

    _analyze_connected_subregions pode reordenar subs (ex: leitura japonesa right-to-left).
    Esta função garante que polygons[i] corresponda a ordered_subs[i].

    Estratégia: match por maior IoU entre cada ordered_sub e os original_subs.
    """
    if len(original_subs) != len(polygons) or len(original_subs) < 2:
        return polygons
    result = list(polygons)
    for i, osub in enumerate(ordered_subs):
        if i >= len(result):
            break
        best_j = i
        best_iou = -1.0
        for j, orig in enumerate(original_subs):
            iou = _bbox_iou(osub, orig)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j != i and best_j < len(polygons):
            result[i] = polygons[best_j]
    return result


def _detect_connected_balloon_subregions_rich(
    image_bgr: np.ndarray | None,
    text_bbox: list[int],
    balloon_bbox: list[int],
    tipo: str,
) -> list[dict]:
    """Versão rica de _detect_connected_balloon_subregions.

    Retorna lista de dicts com:
      - bbox: list[int] (x1,y1,x2,y2)
      - polygon: list[list[int]] (N,2) em coords globais, ou None
      - area: int

    Tenta primeiro detecção topológica (_detect_connected_lobes_from_outline).
    Cai no método baseado em fill como fallback.
    Retorna [] se não detectar balão conectado.
    """
    if image_bgr is None or tipo not in {"fala", "pensamento", "narracao"}:
        return []

    # 1. Tentativa topológica (baseada no contorno real do balão)
    outline_lobes = _detect_connected_lobes_from_outline(image_bgr, balloon_bbox, text_bbox)
    if len(outline_lobes) >= 2:
        # Validar distância entre lobos (mesmo critério que fill_result)
        fa = outline_lobes[0]["bbox"]
        fb = outline_lobes[1]["bbox"]
        dist = (((fa[0]+fa[2])-(fb[0]+fb[2]))**2 + ((fa[1]+fa[3])-(fb[1]+fb[3]))**2)**0.5
        if dist >= min(fa[2]-fa[0], fb[2]-fb[0]) * 0.4:
            # Expandir bboxes para subregions completas (como _build_balloon_subregions_from_groups faz)
            expanded_bboxes = _build_balloon_subregions_from_groups(
                [lobe["bbox"] for lobe in outline_lobes[:2]],
                balloon_bbox,
            )
            if len(expanded_bboxes) >= 2 and _bbox_iou(expanded_bboxes[0], expanded_bboxes[1]) <= 0.28:
                return [
                    {
                        "bbox": expanded_bboxes[i],
                        "polygon": outline_lobes[i]["polygon"].tolist() if isinstance(outline_lobes[i].get("polygon"), np.ndarray) else outline_lobes[i].get("polygon"),
                        "area": outline_lobes[i].get("area", 0),
                    }
                    for i in range(2)
                ]

    # 2. Fallback: fill-based detection
    fill_bboxes = _detect_connected_balloon_subregions_plain(image_bgr, text_bbox, balloon_bbox)
    return [{"bbox": bbox, "polygon": None, "area": 0} for bbox in fill_bboxes]


def _detect_connected_balloon_subregions(
    image_bgr: np.ndarray | None,
    text_bbox: list[int],
    balloon_bbox: list[int],
    tipo: str,
) -> list[list[int]]:
    """Detecção de subregions de balão conectado (retorna apenas bboxes).

    Mantida para retrocompatibilidade. Internamente usa _detect_connected_balloon_subregions_rich.
    """
    rich = _detect_connected_balloon_subregions_rich(image_bgr, text_bbox, balloon_bbox, tipo)
    return [item["bbox"] for item in rich]


def _detect_connected_balloon_subregions_plain(
    image_bgr: np.ndarray | None,
    text_bbox: list[int],
    balloon_bbox: list[int],
) -> list[list[int]]:
    """Implementação original de detecção fill-based (sem topologia).

    Separada para ser usada como fallback em _detect_connected_balloon_subregions_rich.
    """
    if image_bgr is None:
        return []

    fill_result = _detect_connected_balloon_subregions_from_fill(
        image_bgr,
        text_bbox,
        balloon_bbox,
    )
    if len(fill_result) >= 2:
        # Se o floodfill separou, faz um check de distância extra para evitar falsos positivos
        fa, fb = fill_result[0], fill_result[1]
        dist = (((fa[0]+fa[2])-(fb[0]+fb[2]))**2 + ((fa[1]+fa[3])-(fb[1]+fb[3]))**2)**0.5
        if dist < min(fa[2]-fa[0], fb[2]-fb[0]) * 0.4:
            return []
        return fill_result

    components = _extract_text_cluster_components(image_bgr, text_bbox)
    if len(components) < 2:
        return []

    merged_groups = _merge_text_cluster_components(components)
    if len(merged_groups) < 2:
        return []

    merged_groups = sorted(merged_groups, key=lambda item: item["area"], reverse=True)
    top_two = merged_groups[:2]
    total_text_area = sum(item["area"] for item in merged_groups)
    dominant_area = sum(item["area"] for item in top_two)
    
    # Se os dois maiores grupos não dominam o balão, não divide (provavelmente é um balão complexo único)
    if dominant_area < max(1800, int(total_text_area * 0.75)):
        return []

    first_bbox = top_two[0]["bbox"]
    second_bbox = top_two[1]["bbox"]
    box_w = max(1, balloon_bbox[2] - balloon_bbox[0])
    box_h = max(1, balloon_bbox[3] - balloon_bbox[1])
    first_cx = (first_bbox[0] + first_bbox[2]) / 2.0
    first_cy = (first_bbox[1] + first_bbox[3]) / 2.0
    second_cx = (second_bbox[0] + second_bbox[2]) / 2.0
    second_cy = (second_bbox[1] + second_bbox[3]) / 2.0
    
    # Check de proximidade dos centros (se estiverem muito perto, é o mesmo balão)
    if abs(first_cx - second_cx) < box_w * 0.15 and abs(first_cy - second_cy) < box_h * 0.15:
        return []

    ordered_groups = sorted(top_two, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    a_bbox = ordered_groups[0]["bbox"]
    b_bbox = ordered_groups[1]["bbox"]
    vertical_gap = max(0, int(b_bbox[1]) - int(a_bbox[3]))
    horizontal_gap = max(0, int(b_bbox[0]) - int(a_bbox[2]), int(a_bbox[0]) - int(b_bbox[2]))

    is_vertical_stack = abs(first_cx - second_cx) < box_w * 0.30
    is_horizontal_stack = abs(first_cy - second_cy) < box_h * 0.30
    
    # Limites aumentados para evitar separação por simples quebra de linha (Imagem 4)
    if is_vertical_stack and vertical_gap < max(28, int(box_h * 0.12)):
        return []
    if is_horizontal_stack and horizontal_gap < max(28, int(box_w * 0.12)):
        return []
    if not is_vertical_stack and not is_horizontal_stack:
        if max(vertical_gap, horizontal_gap) < max(35, int(min(box_w, box_h) * 0.15)):
            return []

    subregions = _build_balloon_subregions_from_groups(
        [group["bbox"] for group in top_two],
        balloon_bbox,
    )
    if len(subregions) < 2 or _bbox_iou(subregions[0], subregions[1]) > 0.28:
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
    if fill_area < 1500:
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
        min_lobe_area = max(600, int(fill_area * 0.05))
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
        if float(np.count_nonzero(eroded)) < fill_area * 0.20:
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
    if max(gap_y, gap_x) < max(6, int(min(box_w, box_h) * 0.015)):
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
    min_lobe_area = max(600, int(fill_area * 0.035))

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
    if sep < min(bw, bh) * 0.18:
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


def _extract_balloon_outline_polygon(
    image_bgr: np.ndarray,
    balloon_bbox: list[int],
) -> np.ndarray | None:
    """Retorna o contorno externo da região BRANCA do balão como array de pontos (N, 2)
    em coordenadas GLOBAIS. Retorna None se não conseguir extrair.

    Estratégia: detectar a região clara (interior branco do balão), fazer morph close
    para fechar gaps da borda preta, e extrair o contorno da maior região contínua.
    Isso captura a FORMA DO BALÃO, não apenas o anel de tinta preta.

    Passos:
      1. Extrair ROI com padding leve (~8px)
      2. Grayscale + threshold adaptativo para detectar região clara (interior branco)
      3. Morph close para fechar gaps da borda preta (kernel ~7px)
      4. FindContours RETR_EXTERNAL, pegar o de maior área
      5. approxPolyDP com epsilon = 0.003 * perimeter (polígono suave)
      6. Converter para coords globais (adicionar rx1, ry1)
    """
    if image_bgr is None:
        return None
    height, width = image_bgr.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in balloon_bbox]
    bx1 = max(0, bx1)
    by1 = max(0, by1)
    bx2 = min(width, bx2)
    by2 = min(height, by2)
    if bx2 <= bx1 or by2 <= by1:
        return None

    pad = 8
    rx1 = max(0, bx1 - pad)
    ry1 = max(0, by1 - pad)
    rx2 = min(width, bx2 + pad)
    ry2 = min(height, by2 + pad)
    roi = image_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Detectar região clara (interior branco do balão).
    # Usar percentil 85 como referência (igual ao fill-based detection).
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    p85 = float(np.percentile(blur, 85))
    thresh_val = int(max(180, min(235, p85 - 8.0)))
    _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

    # Morph close para fechar gaps da borda preta (~7px para borda típica de mangá)
    close_k = max(7, min(roi.shape[0], roi.shape[1]) // 20 | 1)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        iterations=2,
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    # Rejeitar contornos muito pequenos (< 5% da ROI)
    min_area = roi.shape[0] * roi.shape[1] * 0.05
    if area < min_area:
        return None

    perimeter = cv2.arcLength(largest, True)
    if perimeter < 10:
        return None

    epsilon = 0.003 * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) < 4:
        return None

    # Converter para coords globais (N, 2)
    points = approx.reshape(-1, 2).astype(np.int32)
    points[:, 0] += rx1
    points[:, 1] += ry1
    return points


def _detect_connected_lobes_from_outline(
    image_bgr: np.ndarray,
    balloon_bbox: list[int],
    seed_bbox: list[int],
) -> list[dict]:
    """Detecta lobos de balão conectado com polígonos de contorno real.

    Retorna lista de dicts com:
      - bbox: list[int] (x1,y1,x2,y2) do lobo em coords globais
      - polygon: np.ndarray (N,2) em coords globais (polígono do lobo)
      - area: int (pixels)

    Algoritmo:
      1. Usar mesma lógica fill-based de _detect_connected_balloon_subregions_from_fill
         para obter a máscara binária do interior branco do balão (confiável,
         evita artefatos de texto)
      2. Erosão progressiva da máscara para separar lobos
      3. Para cada nível com 2+ componentes:
         - Voronoi expansion de volta à máscara original
         - Validação: gap mínimo, razão de área
      4. Extrair contorno externo de cada lobo → polígono suave
    """
    if image_bgr is None:
        return []
    height, width = image_bgr.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in balloon_bbox]
    bx1 = max(0, min(width, bx1))
    bx2 = max(0, min(width, bx2))
    by1 = max(0, min(height, by1))
    by2 = max(0, min(height, by2))
    if bx2 <= bx1 or by2 <= by1:
        return []

    # Obter a máscara do interior branco do balão (lógica idêntica ao fill-based)
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

    # Seed: centro do seed_bbox
    sx1, sy1, sx2, sy2 = [int(v) for v in seed_bbox]
    seed_x = int(((sx1 + sx2) / 2.0) - bx1)
    seed_y = int(((sy1 + sy2) / 2.0) - by1)
    seed_x = max(0, min(roi.shape[1] - 1, seed_x))
    seed_y = max(0, min(roi.shape[0] - 1, seed_y))
    if thresh[seed_y, seed_x] == 0:
        return []

    # Encontrar o componente conectado que contém o seed
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        (thresh > 0).astype(np.uint8), connectivity=8,
    )
    seed_label = int(labels[seed_y, seed_x])
    if seed_label <= 0:
        return []
    component = (labels == seed_label).astype(np.uint8) * 255
    fill_area = int(np.count_nonzero(component))
    if fill_area < 1500:
        return []

    # Preencher buracos internos (texto escuro cria buracos na máscara branca)
    contours_fill, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_fill:
        full_mask = np.zeros_like(component)
        cv2.drawContours(full_mask, contours_fill, -1, 255, -1)
    else:
        full_mask = component.copy()
    fill_area_filled = int(np.count_nonzero(full_mask))
    local_h, local_w = full_mask.shape[:2]

    min_dim = min(local_h, local_w)
    base_k = 7
    if min_dim >= 180:
        base_k = 13
    elif min_dim >= 120:
        base_k = 11
    elif min_dim >= 80:
        base_k = 9

    min_lobe_area = max(600, int(fill_area_filled * 0.08))
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    # Erosão progressiva + Voronoi expansion
    for iters in range(1, 4):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_k, base_k))
        eroded = cv2.erode(full_mask, kernel, iterations=iters)
        num_e, labels_e, stats_e, centroids_e = cv2.connectedComponentsWithStats(
            (eroded > 0).astype(np.uint8), connectivity=8,
        )

        # Filtrar e ordenar componentes por área (maior primeiro)
        valid_labels = sorted(
            [lbl for lbl in range(1, num_e) if int(stats_e[lbl, cv2.CC_STAT_AREA]) >= min_lobe_area],
            key=lambda lbl: int(stats_e[lbl, cv2.CC_STAT_AREA]),
            reverse=True,
        )

        if len(valid_labels) < 2:
            if float(np.count_nonzero(eroded)) < fill_area_filled * 0.20:
                break
            continue

        # Usar os 2 maiores componentes
        top2_labels = valid_labels[:2]
        area0 = int(stats_e[top2_labels[0], cv2.CC_STAT_AREA])
        area1 = int(stats_e[top2_labels[1], cv2.CC_STAT_AREA])

        # Rejeitar se o menor lobo for muito pequeno em relação ao maior
        if area1 < area0 * 0.20:
            if float(np.count_nonzero(eroded)) < fill_area_filled * 0.20:
                break
            continue

        # Verificar gap entre os bboxes dos dois componentes eroded
        a_left = int(stats_e[top2_labels[0], cv2.CC_STAT_LEFT])
        a_top = int(stats_e[top2_labels[0], cv2.CC_STAT_TOP])
        a_w_e = int(stats_e[top2_labels[0], cv2.CC_STAT_WIDTH])
        a_h_e = int(stats_e[top2_labels[0], cv2.CC_STAT_HEIGHT])
        b_left = int(stats_e[top2_labels[1], cv2.CC_STAT_LEFT])
        b_top = int(stats_e[top2_labels[1], cv2.CC_STAT_TOP])
        b_w_e = int(stats_e[top2_labels[1], cv2.CC_STAT_WIDTH])
        b_h_e = int(stats_e[top2_labels[1], cv2.CC_STAT_HEIGHT])
        gap_y = max(0, max(a_top, b_top) - min(a_top + a_h_e, b_top + b_h_e))
        gap_x = max(0, max(a_left, b_left) - min(a_left + a_w_e, b_left + b_w_e))
        # Mesmo critério conservador do fill-based: gap mínimo significativo
        if max(gap_y, gap_x) < max(6, int(min(bw, bh) * 0.015)):
            if float(np.count_nonzero(eroded)) < fill_area_filled * 0.20:
                break
            continue

        # Voronoi expansion: cada pixel da máscara original vai ao centroide mais próximo
        centers = [
            (float(centroids_e[top2_labels[0]][0]), float(centroids_e[top2_labels[0]][1])),
            (float(centroids_e[top2_labels[1]][0]), float(centroids_e[top2_labels[1]][1])),
        ]
        ys, xs = np.where(full_mask > 0)
        if len(xs) == 0:
            break

        d0 = (xs - centers[0][0]) ** 2 + (ys - centers[0][1]) ** 2
        d1 = (xs - centers[1][0]) ** 2 + (ys - centers[1][1]) ** 2
        mask0 = np.zeros((local_h, local_w), dtype=np.uint8)
        mask1 = np.zeros((local_h, local_w), dtype=np.uint8)
        belong0 = d0 <= d1
        mask0[ys[belong0], xs[belong0]] = 255
        mask1[ys[~belong0], xs[~belong0]] = 255

        # Extrair polígono e bbox de cada lobo
        lobes = []
        for lobe_mask in (mask0, mask1):
            area = int(np.count_nonzero(lobe_mask))
            if area < min_lobe_area:
                continue
            lobe_ys, lobe_xs = np.where(lobe_mask > 0)
            if len(lobe_xs) == 0:
                continue
            lx1 = int(lobe_xs.min())
            ly1 = int(lobe_ys.min())
            lx2 = int(lobe_xs.max()) + 1
            ly2 = int(lobe_ys.max()) + 1
            # Extrair contorno do lobo para polígono suave
            lobe_contours, _ = cv2.findContours(
                lobe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if lobe_contours:
                largest_c = max(lobe_contours, key=cv2.contourArea)
                peri = cv2.arcLength(largest_c, True)
                epsilon = 0.004 * peri if peri > 0 else 2.0
                approx = cv2.approxPolyDP(largest_c, epsilon, True)
                poly_local = approx.reshape(-1, 2).astype(np.int32)
                poly_global = poly_local.copy()
                poly_global[:, 0] += bx1
                poly_global[:, 1] += by1
            else:
                poly_global = np.array([
                    [bx1 + lx1, by1 + ly1],
                    [bx1 + lx2, by1 + ly1],
                    [bx1 + lx2, by1 + ly2],
                    [bx1 + lx1, by1 + ly2],
                ], dtype=np.int32)

            lobes.append({
                "bbox": [bx1 + lx1, by1 + ly1, bx1 + lx2, by1 + ly2],
                "polygon": poly_global,
                "area": area,
            })

        if len(lobes) < 2:
            continue

        # Validar: overlap entre lobes bboxes < 10%
        a_bbox = lobes[0]["bbox"]
        b_bbox = lobes[1]["bbox"]
        if _bbox_iou(a_bbox, b_bbox) > 0.10:
            continue

        return lobes

    return []


def _build_balloon_subregions_from_groups(group_bboxes: list[list[int]], balloon_bbox: list[int]) -> list[list[int]]:
    if len(group_bboxes) < 2:
        return []

    ordered = [list(bbox) for bbox in group_bboxes[:2]]
    bx1, by1, bx2, by2 = balloon_bbox
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    centers = [
        ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        for bbox in ordered
    ]
    dx = abs(centers[0][0] - centers[1][0])
    dy = abs(centers[0][1] - centers[1][1])

    if dy >= dx * 1.1:
        # Vertical split (T/B)
        top_bbox, bottom_bbox = sorted(ordered, key=lambda b: (b[1], b[0]))
        if top_bbox[3] < bottom_bbox[1]:
            seam_y = int((top_bbox[3] + bottom_bbox[1]) / 2.0)
        else:
            seam_y = int((centers[0][1] + centers[1][1]) / 2.0)
        seam_y = max(by1 + 32, min(by2 - 32, seam_y))
        gap_y = max(4, int(bh * 0.03))
        subs = [
            [bx1, by1, bx2, seam_y - gap_y],
            [bx1, seam_y + gap_y, bx2, by2],
        ]
        return _enforce_min_lobe_size(subs, balloon_bbox)

    if dx >= dy * 1.1:
        # Horizontal split (L/R)
        left_bbox, right_bbox = sorted(ordered, key=lambda b: (b[0], b[1]))
        if left_bbox[2] < right_bbox[0]:
            seam_x = int((left_bbox[2] + right_bbox[0]) / 2.0)
        else:
            seam_x = int((centers[0][0] + centers[1][0]) / 2.0)
        seam_x = max(bx1 + 32, min(bx2 - 32, seam_x))
        gap_x = max(4, int(bw * 0.0125))
        subs = [
            [bx1, by1, seam_x - gap_x, by2],
            [seam_x + gap_x, by1, bx2, by2],
        ]
        return _enforce_min_lobe_size(subs, balloon_bbox)

    # Diagonal split
    diagonals = sorted(ordered, key=lambda b: (b[1], b[0]))
    first, second = diagonals[0], diagonals[1]
    fc = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    sc = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)

    if first[2] < second[0]:
        seam_x = int((first[2] + second[0]) / 2.0)
    else:
        seam_x = int((fc[0] + sc[0]) / 2.0)
    if first[3] < second[1]:
        seam_y = int((first[3] + second[1]) / 2.0)
    else:
        seam_y = int((fc[1] + sc[1]) / 2.0)

    seam_x = max(bx1 + 32, min(bx2 - 32, seam_x))
    seam_y = max(by1 + 28, min(by2 - 28, seam_y))
    gap_x = max(6, int(bw * 0.02))
    overlap_y = max(6, int(bh * 0.03))

    if fc[0] <= sc[0]:
        subs = [
            [bx1, by1, min(bx2, seam_x - gap_x), min(by2, seam_y + overlap_y)],
            [max(bx1, seam_x + gap_x), max(by1, seam_y - overlap_y), bx2, by2],
        ]
    else:
        subs = [
            [max(bx1, seam_x + gap_x), by1, bx2, min(by2, seam_y + overlap_y)],
            [bx1, max(by1, seam_y - overlap_y), min(bx2, seam_x - gap_x), by2],
        ]
    return _enforce_min_lobe_size(subs, balloon_bbox)


def _enforce_min_lobe_size(
    subs: list[list[int]], balloon_bbox: list[int],
) -> list[list[int]]:
    """Garante que cada lobo tem pelo menos 30% da dimensão principal do balão.

    Quando um lobo fica estreito demais (ex: seam muito perto de uma borda),
    recentraliza o seam para dar espaço suficiente.
    """
    bx1, by1, bx2, by2 = balloon_bbox
    MIN_RATIO = 0.30

    result = []
    for s in subs:
        result.append([max(s[0], bx1), max(s[1], by1), min(s[2], bx2), min(s[3], by2)])

    if len(result) != 2:
        return result

    a, b = result
    aw = a[2] - a[0]
    bw_sub = b[2] - b[0]
    ah = a[3] - a[1]
    bh_sub = b[3] - b[1]

    # Checar largura (para splits horizontais e diagonais)
    if aw > 0 and bw_sub > 0:
        total_w = aw + bw_sub
        if aw / float(total_w) < MIN_RATIO:
            needed = int(total_w * MIN_RATIO) - aw
            a[2] = min(bx2, a[2] + needed)
            b[0] = a[2]
        elif bw_sub / float(total_w) < MIN_RATIO:
            needed = int(total_w * MIN_RATIO) - bw_sub
            b[0] = max(bx1, b[0] - needed)
            a[2] = b[0]

    # Checar altura (para splits verticais e diagonais)
    if ah > 0 and bh_sub > 0:
        total_h = ah + bh_sub
        if ah / float(total_h) < MIN_RATIO:
            needed = int(total_h * MIN_RATIO) - ah
            a[3] = min(by2, a[3] + needed)
            b[1] = a[3]
        elif bh_sub / float(total_h) < MIN_RATIO:
            needed = int(total_h * MIN_RATIO) - bh_sub
            b[1] = max(by1, b[1] - needed)
            a[3] = b[1]

    return result


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

    if overlap_x_ratio >= 0.58 and vertical_gap <= max(24, int(min(ah, bh) * 0.45)):
        return True
    if overlap_y_ratio >= 0.58 and horizontal_gap <= max(30, int(min(aw, bw) * 0.28)):
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
