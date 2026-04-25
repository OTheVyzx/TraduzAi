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
) -> Band:
    """Processa uma banda pelas stages OCR -> translate -> inpaint -> typeset."""
    if not band.balloons:
        band.rendered_slice = band.original_slice.copy()
        return band

    page_dict = _band_to_page_dict(band, page_idx)
    ocr_page = runtime.run_ocr_stage(band.strip_slice, page_dict)

    translated_pages, _ = translator.translate_pages(
        [ocr_page],
        context=context or {},
        glossario=glossario or {},
        idioma_origem=idioma_origem,
        idioma_destino=idioma_destino,
    )
    translated_page = translated_pages[0] if translated_pages else {"texts": []}

    cleaned = inpainter.inpaint_band_image(band.strip_slice, translated_page)
    rendered = typesetter.render_band_image(cleaned, translated_page)

    band.rendered_slice = rendered
    band.rendered_slice = _apply_copy_back_outside_balloons(band)
    return band
