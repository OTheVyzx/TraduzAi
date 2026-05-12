"""Reusa as stages existentes para processar uma Band como se fosse uma página."""

from __future__ import annotations

import copy
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from strip.types import Band


@dataclass(frozen=True)
class BandStageOutput:
    stage_id: str
    _page: dict[str, Any] = field(repr=False)
    perf_updates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_page", copy.deepcopy(self._page))
        object.__setattr__(self, "perf_updates", MappingProxyType(dict(self.perf_updates)))

    def to_page_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._page)


@dataclass(frozen=True)
class BandImageStageOutput:
    stage_id: str
    _image: np.ndarray = field(repr=False)
    perf_updates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_image", np.array(self._image, copy=True))
        object.__setattr__(self, "perf_updates", MappingProxyType(copy.deepcopy(dict(self.perf_updates))))

    def to_image(self) -> np.ndarray:
        return np.array(self._image, copy=True)


def _band_to_page_dict(band: Band, page_idx: int, source_page_number: int | None = None) -> dict:
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

    page_number = int(source_page_number or page_idx + 1)
    band_index = int(page_idx + 1)

    return {
        "numero": page_number,
        "width": band.strip_slice.shape[1],
        "height": band.strip_slice.shape[0],
        "_vision_blocks": blocks,
        "_band_y_top": band.y_top,
        "_band_index": band_index,
        "_source_page_number": page_number,
    }


def _apply_copy_back_outside_balloons(
    band: Band,
    balloon_margin: int = 8,
    ocr_page: dict | None = None,
    rendered_slice: np.ndarray | None = None,
) -> np.ndarray:
    """Copy-back defensivo: preserva pixels fora dos balões da banda.

    A máscara é a UNIÃO de:
      1. strip_bbox de cada balão (bbox do detector, em coords absolutas)
      2. balloon_bbox de cada texto no ocr_page (pode ser expandida por
         enrich_page_layout para cobrir a área branca real do balão)

    Sem a segunda fonte, texto renderizado na área expandida do balão seria
    sobrescrito pelo original, causando clipping visual nas bordas.
    """
    rendered = rendered_slice if rendered_slice is not None else band.rendered_slice
    if band.original_slice is None or rendered is None:
        raise ValueError("Band precisa de original_slice e rendered_slice")

    h, w = band.original_slice.shape[:2]
    mask_inside = np.zeros((h, w), dtype=bool)

    def _mark(x1: int, y1: int, x2: int, y2: int) -> None:
        bx1 = max(0, x1)
        by1 = max(0, y1)
        bx2 = min(w, x2)
        by2 = min(h, y2)
        if bx2 > bx1 and by2 > by1:
            mask_inside[by1:by2, bx1:bx2] = True

    # 1. Detector bbox (coords absolutas → band-local)
    for balloon in band.balloons:
        _mark(
            balloon.strip_bbox.x1 - balloon_margin,
            balloon.strip_bbox.y1 - band.y_top - balloon_margin,
            balloon.strip_bbox.x2 + balloon_margin,
            balloon.strip_bbox.y2 - band.y_top + balloon_margin,
        )

    # 2. balloon_bbox das camadas de texto (já em coords band-local)
    if ocr_page:
        for txt in ocr_page.get("texts", []):
            bbox = txt.get("balloon_bbox") or txt.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            _mark(
                bx1 - balloon_margin,
                by1 - balloon_margin,
                bx2 + balloon_margin,
                by2 + balloon_margin,
            )

    result = np.where(
        mask_inside[:, :, None],
        rendered,
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


def _prepare_precomputed_ocr_page(precomputed_ocr_page: dict, page_dict: dict) -> dict:
    """Copia e completa uma página OCR já resolvida para o contrato band-local."""
    ocr_page = dict(precomputed_ocr_page or {})
    ocr_page["texts"] = [
        dict(text)
        for text in list(ocr_page.get("texts") or [])
        if isinstance(text, dict)
    ]
    ocr_page["_vision_blocks"] = [
        dict(block)
        for block in list(ocr_page.get("_vision_blocks") or [])
        if isinstance(block, dict)
    ]
    if isinstance(ocr_page.get("_ocr_stats"), dict):
        ocr_page["_ocr_stats"] = dict(ocr_page["_ocr_stats"])

    for key in (
        "numero",
        "width",
        "height",
        "_band_y_top",
        "_band_index",
        "_source_page_number",
    ):
        if key not in ocr_page and key in page_dict:
            ocr_page[key] = page_dict[key]
    return ocr_page


def _run_band_ocr_stage(
    band: Band,
    *,
    runtime,
    page_dict: dict,
    precomputed_ocr_page: dict | None = None,
) -> BandStageOutput:
    if isinstance(precomputed_ocr_page, dict):
        return BandStageOutput(
            "ocr",
            _prepare_precomputed_ocr_page(precomputed_ocr_page, page_dict),
            {
                "ocr_precomputed_page": True,
                "ocr_runtime_skipped": True,
            },
        )
    return BandStageOutput(
        "ocr",
        runtime.run_ocr_stage(band.strip_slice, page_dict),
    )


def _run_translate_stage(
    ocr_page: dict,
    *,
    translator,
    context: dict | None = None,
    glossario: dict | None = None,
    idioma_origem: str = "en",
    idioma_destino: str = "pt-BR",
    obra: str = "",
    models_dir: str = "",
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "traduzai-translator",
    translation_context: dict | None = None,
) -> BandStageOutput:
    translated_pages = translator.translate_pages(
        [ocr_page],
        obra=obra,
        context=context or {},
        glossario=glossario or {},
        idioma_origem=idioma_origem,
        idioma_destino=idioma_destino,
        models_dir=models_dir,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        translation_context=translation_context,
    )
    translated_page = translated_pages[0] if translated_pages else {"texts": []}
    return BandStageOutput(
        "translate",
        _merge_translated_page_metadata(ocr_page, translated_page),
    )


def _ensure_text_balloon_bboxes(page: dict, band: Band) -> None:
    vision_blocks = page.get("_vision_blocks", [])
    for txt in page.get("texts", []):
        if txt.get("balloon_bbox"):
            continue
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
            w = page.get("width", band.strip_slice.shape[1])
            h = page.get("height", band.strip_slice.shape[0])
            txt["balloon_bbox"] = [
                max(0, tx1 - 8), max(0, ty1 - 8),
                min(w, tx2 + 8), min(h, ty2 + 8),
            ]


def _run_review_layout_stage(
    band: Band,
    *,
    ocr_page: dict,
    band_history: list[dict] | None = None,
    connected_reasoner_config: dict | None = None,
) -> BandStageOutput:
    from ocr.contextual_reviewer import contextual_review_page
    from layout.balloon_layout import enrich_page_layout
    import cv2

    reviewed_page = contextual_review_page(copy.deepcopy(ocr_page), band_history or [], [])
    if connected_reasoner_config:
        reviewed_page["_connected_balloon_reasoner"] = connected_reasoner_config

    reviewed_page["_cached_image_bgr"] = cv2.cvtColor(band.strip_slice, cv2.COLOR_RGB2BGR)
    reviewed_page = enrich_page_layout(reviewed_page)
    _ensure_text_balloon_bboxes(reviewed_page, band)
    return BandStageOutput("review_layout", reviewed_page)


def _collect_inpaint_perf_updates(translated_page: dict) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if "_strip_fast_white_balloon_count" in translated_page:
        updates["fast_white_balloon_count"] = int(translated_page.get("_strip_fast_white_balloon_count") or 0)
    if "_strip_fast_local_balloon_count" in translated_page:
        updates["fast_local_balloon_count"] = int(translated_page.get("_strip_fast_local_balloon_count") or 0)
    if "_strip_remaining_inpaint_blocks" in translated_page:
        updates["remaining_inpaint_blocks"] = int(translated_page.get("_strip_remaining_inpaint_blocks") or 0)
    for reason_key, perf_key in (
        ("_strip_fast_white_rejection_reasons", "fast_white_rejection_reasons"),
        ("_strip_fast_local_rejection_reasons", "fast_local_rejection_reasons"),
    ):
        reasons = translated_page.get(reason_key)
        if isinstance(reasons, dict):
            updates[perf_key] = dict(reasons)
    for flag in (
        "used_fast_white_fill",
        "used_fast_local_fill",
        "used_real_inpaint",
        "used_post_cleanup",
    ):
        key = f"_strip_{flag}"
        if key in translated_page:
            updates[flag] = bool(translated_page.get(key))
    for key in (
        "_t_roi_select_ms",
        "_t_lama_ms",
        "_t_lama_total_ms",
        "_t_cleanup_total_ms",
        "_t_cleanup_seam_ms",
        "_t_cleanup_band_artifact_ms",
        "_t_cleanup_white_line_ms",
        "_t_cleanup_white_box_ms",
        "_t_cleanup_micro_ms",
        "used_roi_crop",
        "roi_area_ratio",
        "cleanup_reason",
        "cleanup_skipped_seam",
        "cleanup_skipped_band_artifact",
        "cleanup_skipped_white_line",
        "cleanup_skipped_white_box",
    ):
        if key in translated_page:
            updates[key] = translated_page.get(key)
    return updates


def _run_inpaint_stage(
    band: Band,
    *,
    inpainter,
    translated_page: dict,
) -> BandImageStageOutput:
    page_for_inpaint = copy.deepcopy(translated_page)
    cleaned = inpainter.inpaint_band_image(band.strip_slice, page_for_inpaint)
    translated_page.update(
        {
            key: value
            for key, value in page_for_inpaint.items()
            if str(key).startswith("_strip_")
            or key
            in {
                "used_roi_crop",
                "roi_area_ratio",
                "cleanup_reason",
                "cleanup_skipped_seam",
                "cleanup_skipped_band_artifact",
                "cleanup_skipped_white_line",
                "cleanup_skipped_white_box",
            }
        }
    )
    return BandImageStageOutput(
        "inpaint",
        cleaned,
        _collect_inpaint_perf_updates(translated_page),
    )


def _run_typeset_stage(
    cleaned_slice: np.ndarray,
    *,
    typesetter,
    translated_page: dict,
) -> BandImageStageOutput:
    return BandImageStageOutput(
        "typeset",
        typesetter.render_band_image(cleaned_slice, translated_page),
    )


def _run_copy_back_stage(
    band: Band,
    *,
    rendered_slice: np.ndarray,
    translated_page: dict,
) -> BandImageStageOutput:
    return BandImageStageOutput(
        "copy_back",
        _apply_copy_back_outside_balloons(
            band,
            ocr_page=translated_page,
            rendered_slice=rendered_slice,
        ),
    )


def _commit_band_outputs(
    band: Band,
    *,
    cleaned_slice: np.ndarray,
    rendered_slice: np.ndarray,
    ocr_result: dict,
) -> Band:
    band.cleaned_slice = np.array(cleaned_slice, copy=True)
    band.rendered_slice = np.array(rendered_slice, copy=True)
    band.ocr_result = copy.deepcopy(ocr_result)
    return band


def _all_translations_unchanged(page: dict) -> bool:
    texts = list((page or {}).get("texts") or [])
    if not texts:
        return False
    for text in texts:
        if not isinstance(text, dict):
            return False
        source = str(text.get("original") or text.get("text") or "").strip()
        translated = str(text.get("translated") or text.get("traduzido") or "").strip()
        if not source or not translated or source != translated:
            return False
    return True


def _all_texts_skip_processing(page: dict) -> bool:
    texts = list((page or {}).get("texts") or [])
    return bool(texts) and all(isinstance(text, dict) and bool(text.get("skip_processing")) for text in texts)


def _smart_skip_shadow_enabled() -> bool:
    value = os.environ.get("TRADUZAI_SMART_SKIP_SHADOW", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _smart_skip_real_enabled() -> bool:
    value = os.environ.get("TRADUZAI_SMART_SKIP", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_smart_skip_shadow(page: dict, perf: dict) -> None:
    from strip.smart_skip import annotate_page_with_smart_skip_shadow

    annotate_page_with_smart_skip_shadow(page)
    shadow = page.get("_smart_skip_shadow") or {}
    perf["smart_skip_shadow_candidate_count"] = int(shadow.get("candidate_count") or 0)
    perf["smart_skip_shadow_not_safe_count"] = int(shadow.get("not_safe_count") or 0)
    perf["smart_skip_shadow_category_counts"] = dict(shadow.get("category_counts") or {})


def _apply_smart_skip_real(page: dict, perf: dict) -> bool:
    from strip.smart_skip import annotate_page_with_smart_skip_shadow

    annotate_page_with_smart_skip_shadow(page)
    shadow = page.get("_smart_skip_shadow") or {}
    candidates = list(shadow.get("candidates") or [])
    candidate_count = int(shadow.get("candidate_count") or 0)
    not_safe_count = int(shadow.get("not_safe_count") or 0)
    category_counts = dict(shadow.get("category_counts") or {})

    perf["smart_skip_real_candidate_count"] = candidate_count
    perf["smart_skip_real_not_safe_count"] = not_safe_count
    perf["smart_skip_real_category_counts"] = category_counts
    perf["smart_skip_real_applied"] = False

    texts = [text for text in list(page.get("texts") or []) if isinstance(text, dict)]
    if not texts or candidate_count != len(texts) or not_safe_count:
        return False

    candidates_by_index = {
        int(candidate.get("text_index")): candidate
        for candidate in candidates
        if candidate.get("text_index") is not None
    }
    if len(candidates_by_index) != len(texts):
        return False

    for index, text in enumerate(texts):
        decision = dict(candidates_by_index[index])
        text["skip_processing"] = True
        text["skip_reason"] = "smart_skip"
        text["smart_skip_decision"] = decision
        if not text.get("translated"):
            text["translated"] = text.get("original") or text.get("text") or ""
    perf["smart_skip_real_applied"] = True
    return True


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
    source_page_number: int | None = None,
    models_dir: str = "",
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "traduzai-translator",
    translation_context: dict | None = None,
    precomputed_ocr_page: dict | None = None,
    ordered_context_after_translate_callback=None,
    gpu_stage_lock=None,
    typeset_stage_lock=None,
) -> Band:


    """Processa uma banda pelas stages OCR -> translate -> inpaint -> typeset."""
    total_start = time.perf_counter()
    durations: dict[str, float] = {}
    perf = {
        "band_index": int(page_idx),
        "y_top": int(band.y_top),
        "y_bottom": int(band.y_bottom),
        "height": int(band.height),
        "balloon_count": int(len(band.balloons)),
        "ocr_text_count": 0,
        "text_count": 0,
        "durations_sec": durations,
    }

    def _mark(stage: str, started_at: float) -> None:
        elapsed = time.perf_counter() - started_at
        durations[stage] = round(elapsed, 4)
        perf[f"_t_{stage}_ms"] = round(elapsed * 1000.0, 3)

    def _finish(ocr_result: dict | None = None) -> None:
        if isinstance(ocr_result, dict):
            perf["text_count"] = int(len(ocr_result.get("texts") or []))
            perf["total_sec"] = round(time.perf_counter() - total_start, 4)
            ocr_result["_perf"] = dict(perf)
        else:
            perf["total_sec"] = round(time.perf_counter() - total_start, 4)
        band.perf = dict(perf)
    if not band.balloons:
        original = band.original_slice if band.original_slice is not None else band.strip_slice
        _commit_band_outputs(
            band,
            cleaned_slice=original,
            rendered_slice=original,
            ocr_result={"texts": [], "_vision_blocks": []},
        )
        _finish(band.ocr_result)
        return band

    page_dict = _band_to_page_dict(band, page_idx, source_page_number=source_page_number)
    stage_start = time.perf_counter()
    with gpu_stage_lock if gpu_stage_lock is not None else nullcontext():
        ocr_stage = _run_band_ocr_stage(
            band,
            runtime=runtime,
            page_dict=page_dict,
            precomputed_ocr_page=precomputed_ocr_page,
        )
    ocr_page = ocr_stage.to_page_dict()
    perf.update(dict(ocr_stage.perf_updates))
    _mark("ocr", stage_start)
    perf["ocr_text_count"] = int(len(ocr_page.get("texts") or []))
    ocr_stats = ocr_page.get("_ocr_stats") if isinstance(ocr_page, dict) else None
    if isinstance(ocr_stats, dict):
        for key in (
            "full_page_mapped",
            "crop_fallback_max",
            "crop_fallback_attempts",
            "crop_fallback_recovered",
        ):
            if key in ocr_stats:
                try:
                    perf[f"ocr_{key}"] = int(ocr_stats.get(key) or 0)
                except Exception:
                    continue
        if "quick_skipped_no_text" in ocr_stats:
            perf["ocr_quick_skipped_no_text"] = bool(ocr_stats.get("quick_skipped_no_text"))
        if "scanlation_credit_skipped" in ocr_stats:
            perf["ocr_scanlation_credit_skipped"] = bool(ocr_stats.get("scanlation_credit_skipped"))
        if "cover_editorial_skipped" in ocr_stats:
            perf["ocr_cover_editorial_skipped"] = bool(ocr_stats.get("cover_editorial_skipped"))
        if "macro_ocr_real" in ocr_stats:
            perf["ocr_macro_ocr_real"] = bool(ocr_stats.get("macro_ocr_real"))
        if "macro_ocr_page_window_owner" in ocr_stats:
            perf["ocr_macro_ocr_page_window_owner"] = bool(
                ocr_stats.get("macro_ocr_page_window_owner")
            )
        for key in (
            "macro_window_count",
            "macro_window_reports",
            "macro_ocr_page_number",
            "macro_ocr_block_count",
            "macro_ocr_empty_record_count",
        ):
            if key in ocr_stats:
                try:
                    perf[f"ocr_{key}"] = int(ocr_stats.get(key) or 0)
                except Exception:
                    continue
        for key in (
            "ocr_cache_hits",
            "ocr_cache_misses",
            "ocr_dedup_removed",
            "quick_text_check_stage",
            "ocr_run_on_suspect_count",
            "ocr_run_on_resolved_count",
        ):
            if key in ocr_stats:
                perf[key] = ocr_stats.get(key)
    elif isinstance(ocr_page, dict) and "quick_skipped_no_text" in ocr_page:
        perf["ocr_quick_skipped_no_text"] = bool(ocr_page.get("quick_skipped_no_text"))
    if isinstance(ocr_page, dict) and "scanlation_credit_skipped" in ocr_page:
        perf["ocr_scanlation_credit_skipped"] = bool(ocr_page.get("scanlation_credit_skipped"))
    if isinstance(ocr_page, dict) and "cover_editorial_skipped" in ocr_page:
        perf["ocr_cover_editorial_skipped"] = bool(ocr_page.get("cover_editorial_skipped"))
    if not list(ocr_page.get("texts") or []):
        original = band.original_slice if band.original_slice is not None else band.strip_slice
        _commit_band_outputs(
            band,
            cleaned_slice=original,
            rendered_slice=original,
            ocr_result={**ocr_page, "texts": [], "_vision_blocks": []},
        )
        _finish(band.ocr_result)
        return band
    if _smart_skip_real_enabled():
        _apply_smart_skip_real(ocr_page, perf)
    if _all_texts_skip_processing(ocr_page):
        original = band.original_slice if band.original_slice is not None else band.strip_slice
        _commit_band_outputs(
            band,
            cleaned_slice=original,
            rendered_slice=original,
            ocr_result=ocr_page,
        )
        perf["skip_processing_copy"] = True
        _finish(band.ocr_result)
        return band

    # Qualidade: Revisão contextual e enriquecimento de layout (SFX vs Fala, Balões Conectados)
    stage_start = time.perf_counter()
    review_layout_stage = _run_review_layout_stage(
        band,
        ocr_page=ocr_page,
        band_history=band_history,
        connected_reasoner_config=connected_reasoner_config,
    )
    ocr_page = review_layout_stage.to_page_dict()
    _mark("review_layout", stage_start)

    if _smart_skip_shadow_enabled():
        _apply_smart_skip_shadow(ocr_page, perf)

    stage_start = time.perf_counter()
    translate_stage = _run_translate_stage(
        ocr_page,
        translator=translator,
        context=context,
        glossario=glossario,
        idioma_origem=idioma_origem,
        idioma_destino=idioma_destino,
        obra=obra,
        models_dir=models_dir,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        translation_context=translation_context,
    )
    _mark("translate", stage_start)

    translated_page = translate_stage.to_page_dict()
    if callable(ordered_context_after_translate_callback):
        ordered_context_after_translate_callback(copy.deepcopy(translated_page))

    if _all_texts_skip_processing(translated_page):
        original = band.original_slice if band.original_slice is not None else band.strip_slice
        _commit_band_outputs(
            band,
            cleaned_slice=original,
            rendered_slice=original,
            ocr_result=translated_page,
        )
        perf["skip_processing_copy"] = True
        _finish(band.ocr_result)
        return band

    if _all_translations_unchanged(translated_page):
        original = band.original_slice if band.original_slice is not None else band.strip_slice
        _commit_band_outputs(
            band,
            cleaned_slice=original,
            rendered_slice=original,
            ocr_result=translated_page,
        )
        perf["unchanged_translation_skip"] = True
        _finish(band.ocr_result)
        return band

    stage_start = time.perf_counter()
    with gpu_stage_lock if gpu_stage_lock is not None else nullcontext():
        inpaint_stage = _run_inpaint_stage(
            band,
            inpainter=inpainter,
            translated_page=translated_page,
        )
    _mark("inpaint", stage_start)
    cleaned = inpaint_stage.to_image()
    perf.update(dict(inpaint_stage.perf_updates))
    stage_start = time.perf_counter()
    with typeset_stage_lock if typeset_stage_lock is not None else nullcontext():
        typeset_stage = _run_typeset_stage(
            cleaned,
            typesetter=typesetter,
            translated_page=translated_page,
        )
    _mark("typeset", stage_start)
    stage_start = time.perf_counter()
    copy_back_stage = _run_copy_back_stage(
        band,
        rendered_slice=typeset_stage.to_image(),
        translated_page=translated_page,
    )
    _mark("copy_back", stage_start)
    _commit_band_outputs(
        band,
        cleaned_slice=cleaned,
        rendered_slice=copy_back_stage.to_image(),
        ocr_result=translated_page,
    )
    _finish(band.ocr_result)
    return band
