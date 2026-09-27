"""Cold-path adapters for the installed OCR and translation providers.

The adapters expose only hash-bound evidence. Context may influence inference,
but is intentionally absent from their returned/public payloads.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Callable

import numpy as np

from ownership.hash_contract import canonical_json_sha256, canonical_page_sha256
from ownership.ocr_contract import OCRRequest


class ProviderUnavailable(RuntimeError):
    """A real provider could not produce an acceptable, publishable unit."""


_OCR_ENGINE = None
_OCR_ENGINE_LOCK = threading.Lock()


def _default_ocr_invocation(**kwargs):
    from vision_stack.ocr import OCREngine

    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        with _OCR_ENGINE_LOCK:
            if _OCR_ENGINE is None:
                _OCR_ENGINE = OCREngine(
                    model="paddleocr", device="cuda", half=True, batch_size=8, lang="en"
                )
    engine = _OCR_ENGINE
    return engine.recognize_region_with_evidence(**kwargs)


def recover_source_with_ocr(
    *,
    page_rgb: np.ndarray,
    bbox_page: tuple[int, int, int, int],
    page_id: str,
    page_source_sha256: str,
    run_id: str,
    origin_execution_id: str,
    invocation: Callable[..., Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a fresh anchored OCR request and return immutable public evidence."""

    del context  # inference-only input; never copied to evidence/project payloads
    root_hash = canonical_page_sha256(page_rgb)
    request = OCRRequest(
        run_id=str(run_id),
        origin_execution_id=str(origin_execution_id),
        page_id=str(page_id),
        page_source_sha256=str(page_source_sha256),
        root_input_pixel_sha256=root_hash,
        invocation_id=f"{run_id}:{page_id}:{canonical_json_sha256(list(bbox_page))[:16]}",
        provider_family="vision-paddleocr",
    )
    invoke = invocation or _default_ocr_invocation
    try:
        result = invoke(
            page_rgb=page_rgb,
            bbox_page=tuple(int(value) for value in bbox_page),
            request=request,
            variants=("anchored_crop", "gray", "inverted", "scale_2x"),
            stop_on_first_text=False,
        )
    except Exception as exc:
        raise ProviderUnavailable(f"local OCR provider unavailable: {exc.__class__.__name__}") from exc

    attempts = tuple(getattr(result, "attempts", ()) or ())
    fresh = [attempt for attempt in attempts if getattr(attempt, "qualifies_as_fresh_physical_inference", False)]
    if not fresh:
        raise ProviderUnavailable("fresh physical OCR evidence is required")
    observations = [row for row in tuple(getattr(result, "observations", ()) or ())
                    if str(getattr(row, "text", "")).strip()]
    if not observations:
        raise ProviderUnavailable("fresh physical OCR returned no text")
    observations.sort(key=lambda row: (
        int(getattr(row, "bbox_page", (0, 0, 0, 0))[1]),
        int(getattr(row, "bbox_page", (0, 0, 0, 0))[0]),
        str(getattr(row, "observation_id", "")),
    ))
    selected = [{
        "observation_id": str(row.observation_id),
        "attempt_id": str(row.attempt_id),
        "text": str(row.text).strip(),
        "confidence": float(row.confidence),
        "bbox_page": [int(value) for value in row.bbox_page],
        "payload_sha256": str(row.payload_sha256),
    } for row in observations]
    source = " ".join(row["text"] for row in selected).strip()
    review = {
        "schema": "traduzai.cold-ocr-selection.v1",
        "source": source,
        "selected_observation_ids": [row["observation_id"] for row in selected],
        "attempt_chain_sha256": str(getattr(result, "attempt_chain_sha256", "")),
        "root_input_pixel_sha256": root_hash,
        "page_source_sha256": str(page_source_sha256),
    }
    return {
        **review,
        "selected": selected,
        "provider_called": True,
        "cache_hit": False,
        "source_review_sha256": canonical_json_sha256(review),
    }


def translate_complete_unit(
    *,
    source: str,
    owner_id: str,
    page_id: str,
    context: dict[str, Any] | None = None,
    glossary: dict[str, str] | None = None,
    work_title: str = "TraduzAi",
    invoke: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Translate one whole logical unit with the real provider and no cache."""

    from translator.translate import TranslationAttemptControl, translate_one_owner_attempt

    normalized_source = " ".join(str(source or "").split())
    if not normalized_source:
        raise ProviderUnavailable("empty source unit")
    call = invoke or translate_one_owner_attempt
    page = {
        "page_id": str(page_id),
        "texts": [{
            "id": str(owner_id),
            "owner_id": str(owner_id),
            "text": normalized_source,
            "original": normalized_source,
            "semantic_role": "dialogue_body",
            "tipo": "dialogue_body",
            "route_action": "translate_inpaint_render",
        }],
    }
    try:
        result = call(
            page,
            str(work_title),
            dict(context or {}),
            dict(glossary or {}),
            idioma_destino="pt-BR",
            idioma_origem="en",
            qualidade="normal",
            ollama_host="http://localhost:11434",
            ollama_model="traduzai-translator",
            models_dir="",
            translation_context=None,
            control=TranslationAttemptControl("google", "cold_complete_unit", True),
        )
    except Exception as exc:
        raise ProviderUnavailable(f"complete-unit translation unavailable: {exc.__class__.__name__}") from exc
    if not bool(getattr(result, "provider_called", False)) or bool(getattr(result, "cache_hit", True)):
        raise ProviderUnavailable("fresh physical translation evidence is required")
    items = tuple(getattr(result, "translated_items", ()) or ())
    if len(items) != 1:
        raise ProviderUnavailable("translation provider returned invalid unit cardinality")
    translated = items[0].read()
    records = [row for row in translated.get("texts") or []
               if str(row.get("owner_id") or row.get("id") or "") == str(owner_id)]
    if len(records) != 1:
        raise ProviderUnavailable("translation response lost owner identity")
    target = " ".join(str(records[0].get("translated") or "").split())
    if not target:
        raise ProviderUnavailable("empty translated unit")
    comparable_source = re.sub(r"\s+", "", normalized_source).casefold()
    comparable_target = re.sub(r"\s+", "", target).casefold()
    if comparable_target == comparable_source:
        raise ProviderUnavailable("unchanged_source_dialogue")
    return {
        "target": target,
        "provenance": "fresh_complete_unit_translation",
        "backend": str(result.backend),
        "variant": str(result.variant),
        "provider_model": result.provider_model,
        "provider_called": True,
        "cache_hit": False,
        "provider_metadata_sha256": str(result.provider_metadata_sha256),
    }
