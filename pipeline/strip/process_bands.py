"""Reusa as stages existentes para processar uma Band como se fosse uma página."""

from __future__ import annotations

import numpy as np

from strip.types import Band


def _band_to_page_dict(band: Band, page_idx: int) -> dict:
    """Converte uma Band para o formato dict que vision_stack.runtime aceita."""
    if band.strip_slice is None:
        raise ValueError("Band sem strip_slice; chame attach_band_slices primeiro")

    blocks = []
    for balloon in band.balloons:
        bbox_local = [
            balloon.strip_bbox.x1,
            balloon.strip_bbox.y1 - band.y_top,
            balloon.strip_bbox.x2,
            balloon.strip_bbox.y2 - band.y_top,
        ]
        blocks.append({"bbox": bbox_local, "confidence": balloon.confidence})

    return {
        "numero": page_idx + 1,
        "width": band.strip_slice.shape[1],
        "height": band.strip_slice.shape[0],
        "_vision_blocks": blocks,
        "_band_y_top": band.y_top,
    }


def _apply_copy_back_outside_balloons(
    band: Band,
    balloon_margin: int = 8,
) -> np.ndarray:
    """Copy-back defensivo: preserva pixels fora dos balões da banda."""
    if band.original_slice is None or band.rendered_slice is None:
        raise ValueError("Band precisa de original_slice e rendered_slice")

    h, w = band.original_slice.shape[:2]
    mask_inside = np.zeros((h, w), dtype=bool)

    for balloon in band.balloons:
        x1 = max(0, balloon.strip_bbox.x1 - balloon_margin)
        y1 = max(0, balloon.strip_bbox.y1 - band.y_top - balloon_margin)
        x2 = min(w, balloon.strip_bbox.x2 + balloon_margin)
        y2 = min(h, balloon.strip_bbox.y2 - band.y_top + balloon_margin)
        if x2 > x1 and y2 > y1:
            mask_inside[y1:y2, x1:x2] = True

    result = np.where(
        mask_inside[:, :, None],
        band.rendered_slice,
        band.original_slice,
    )
    return result.astype(np.uint8)


def _merge_translated_page_metadata(ocr_page: dict, translated_page: dict) -> dict:
    if not isinstance(translated_page, dict):
        return {"texts": []}

    merged_page = dict(translated_page)
    ocr_texts = list((ocr_page or {}).get("texts") or [])
    translated_texts = list((translated_page or {}).get("texts") or [])

    ocr_by_id = {
        text.get("id"): text
        for text in ocr_texts
        if isinstance(text, dict) and text.get("id")
    }

    merged_texts = []
    for index, translated_text in enumerate(translated_texts):
        if not isinstance(translated_text, dict):
            continue
        source_text = None
        text_id = translated_text.get("id")
        if text_id in ocr_by_id:
            source_text = ocr_by_id[text_id]
        elif index < len(ocr_texts) and isinstance(ocr_texts[index], dict):
            source_text = ocr_texts[index]
        merged_texts.append({**(source_text or {}), **translated_text})

    merged_page["texts"] = merged_texts

    if not merged_page.get("_vision_blocks"):
        merged_page["_vision_blocks"] = list((ocr_page or {}).get("_vision_blocks") or [])

    for key in ("numero", "width", "height", "page_profile"):
        if key not in merged_page and key in ocr_page:
            merged_page[key] = ocr_page[key]

    return merged_page


def process_band(
    band: Band,
    runtime,
    translator,
    inpainter,
    typesetter,
    page_idx: int,
    context: dict | None = None,
    glossario: dict | None = None,
    idioma_origem: str = "en",
    idioma_destino: str = "pt-BR",
    obra: str = "",
    connected_reasoner_config: dict | None = None,
    band_history: list[dict] | None = None,
) -> Band:


    """Processa uma banda pelas stages OCR -> translate -> inpaint -> typeset."""
    if not band.balloons:
        band.rendered_slice = band.original_slice.copy()
        band.ocr_result = {"texts": [], "_vision_blocks": []}
        return band

    page_dict = _band_to_page_dict(band, page_idx)
    ocr_page = runtime.run_ocr_stage(band.strip_slice, page_dict)

    # Qualidade: Revisão contextual e enriquecimento de layout (SFX vs Fala, Balões Conectados)
    from ocr.contextual_reviewer import contextual_review_page
    from layout.balloon_layout import enrich_page_layout
    import cv2

    ocr_page = contextual_review_page(ocr_page, band_history or [], [])  # Histórico rolante de bandas
    if connected_reasoner_config:
        ocr_page["_connected_balloon_reasoner"] = connected_reasoner_config

    # enrich_page_layout precisa da imagem em BGR para análise geométrica/cor
    ocr_page["_cached_image_bgr"] = cv2.cvtColor(band.strip_slice, cv2.COLOR_RGB2BGR)
    ocr_page = enrich_page_layout(ocr_page)

    # Fallback defensivo: garantir balloon_bbox em CADA text após enrich
    vision_blocks = ocr_page.get("_vision_blocks", [])
    for txt in ocr_page.get("texts", []):
        if txt.get("balloon_bbox"):
            continue
        # Achar vision_block que melhor contém o text bbox (maior IoU com texto)
        tx1, ty1, tx2, ty2 = txt.get("bbox", [0, 0, 0, 0])
        best = None
        best_iou = 0.0
        for vb in vision_blocks:
            vx1, vy1, vx2, vy2 = vb["bbox"]
            ix = max(0, min(tx2, vx2) - max(tx1, vx1))
            iy = max(0, min(ty2, vy2) - max(ty1, vy1))
            inter = ix * iy
            ta = max(1, (tx2 - tx1) * (ty2 - ty1))
            ratio = inter / ta
            if ratio > best_iou:
                best_iou = ratio
                best = vb
        if best:
            txt["balloon_bbox"] = list(best["bbox"])
        else:
            # Último recurso: balloon_bbox = text_bbox + 8 px de margem
            w = ocr_page.get("width", band.strip_slice.shape[1])
            h = ocr_page.get("height", band.strip_slice.shape[0])
            txt["balloon_bbox"] = [
                max(0, tx1 - 8), max(0, ty1 - 8),
                min(w, tx2 + 8), min(h, ty2 + 8),
            ]

    translated_pages = translator.translate_pages(
        [ocr_page],
        obra=obra,
        context=context or {},
        glossario=glossario or {},
        idioma_origem=idioma_origem,
        idioma_destino=idioma_destino,
    )


    translated_page = translated_pages[0] if translated_pages else {"texts": []}
    translated_page = _merge_translated_page_metadata(ocr_page, translated_page)

    cleaned = inpainter.inpaint_band_image(band.strip_slice, translated_page)
    rendered = typesetter.render_band_image(cleaned, translated_page)

    band.cleaned_slice = cleaned
    band.rendered_slice = rendered
    band.rendered_slice = _apply_copy_back_outside_balloons(band)
    band.ocr_result = translated_page
    return band
