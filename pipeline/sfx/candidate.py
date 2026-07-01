from __future__ import annotations

from copy import deepcopy
from typing import Any

from .hangul_adapter import adapt_hangul_sfx


SFX_ROUTE_ACTION = "translate_sfx_inpaint_render"
SFX_TRANSLATION_MODE = "onomatopoeia_adaptation"
LOW_CONFIDENCE_THRESHOLD = 0.7


def enrich_sfx_candidate(layer: dict) -> dict:
    if not isinstance(layer, dict):
        return layer

    route_action = str(layer.get("route_action") or "").strip().lower()
    content_class = str(layer.get("content_class") or "").strip().lower()
    if route_action != SFX_ROUTE_ACTION and content_class != "sfx":
        return layer

    enriched = deepcopy(layer)
    source_text = _source_text(enriched)
    existing_sfx = enriched.get("sfx") if isinstance(enriched.get("sfx"), dict) else {}
    if not source_text and existing_sfx.get("visual_promotion"):
        existing_sfx_flags = [str(flag) for flag in existing_sfx.get("qa_flags") or [] if flag]
        qa_flags = list(dict.fromkeys([*(enriched.get("qa_flags") or []), *existing_sfx_flags, "sfx_text_unknown"]))
        enriched["content_class"] = "sfx"
        enriched["tipo"] = "sfx"
        enriched["script"] = str(enriched.get("script") or "visual_unknown")
        enriched["route_action"] = SFX_ROUTE_ACTION
        enriched["translate_policy"] = "review"
        enriched["render_policy"] = "sfx_style"
        enriched["route_reason"] = str(enriched.get("route_reason") or "visual_sfx_promoted_without_ocr")
        enriched["skip_processing"] = False
        enriched["preserve_original"] = False
        enriched["qa_flags"] = qa_flags
        enriched["sfx"] = {
            **existing_sfx,
            "source_text": "",
            "adapted_text": str(existing_sfx.get("adapted_text") or ""),
            "translation_mode": str(existing_sfx.get("translation_mode") or "visual_sfx_manual_text_required"),
            "inpaint_allowed": bool(existing_sfx.get("inpaint_allowed") and existing_sfx.get("adapted_text")),
            "qa_flags": list(dict.fromkeys([*existing_sfx_flags, "sfx_text_unknown"])),
        }
        return enriched
    if not source_text and existing_sfx.get("visual_detector"):
        existing_sfx_flags = [str(flag) for flag in existing_sfx.get("qa_flags") or [] if flag]
        qa_flags = list(dict.fromkeys([*(enriched.get("qa_flags") or []), *existing_sfx_flags, "sfx_script_unknown"]))
        enriched["content_class"] = "sfx"
        enriched["tipo"] = "sfx"
        enriched["script"] = "unknown"
        enriched["route_action"] = "review_required"
        enriched["translate_policy"] = "review"
        enriched["render_policy"] = "review_required"
        enriched["qa_flags"] = qa_flags
        enriched["sfx"] = {
            **existing_sfx,
            "source_text": "",
            "adapted_text": "",
            "inpaint_allowed": False,
            "qa_flags": list(dict.fromkeys([*existing_sfx_flags, "sfx_script_unknown"])),
        }
        return enriched
    adaptation = adapt_hangul_sfx(source_text)
    existing_sfx_flags = [str(flag) for flag in existing_sfx.get("qa_flags") or [] if flag]
    qa_flags = list(dict.fromkeys([*adaptation.qa_flags, *existing_sfx_flags]))
    review_required = bool(adaptation.review_required)
    if 0.0 < adaptation.confidence < LOW_CONFIDENCE_THRESHOLD:
        review_required = True
        qa_flags = list(dict.fromkeys([*qa_flags, "low_confidence"]))

    existing_inpaint_allowed = existing_sfx.get("inpaint_allowed", enriched.get("inpaint_allowed", False))
    enriched["content_class"] = "sfx"
    enriched["tipo"] = "sfx"
    enriched["script"] = "hangul"
    if str(enriched.get("translate_policy") or "").strip().lower() in {"", "translate"}:
        enriched["translate_policy"] = "adapt_sfx"
    if str(enriched.get("render_policy") or "").strip().lower() in {"", "normal"}:
        enriched["render_policy"] = "sfx_style"
    enriched["route_action"] = "review_required" if review_required else SFX_ROUTE_ACTION
    if review_required:
        enriched["render_policy"] = "review_required"

    enriched["sfx"] = {
        **existing_sfx,
        "source_text": adaptation.source_text,
        "adapted_text": adaptation.adapted_text,
        "confidence": adaptation.confidence,
        "kind": adaptation.kind,
        "translation_mode": SFX_TRANSLATION_MODE,
        "qa_flags": qa_flags,
        "inpaint_allowed": bool(existing_inpaint_allowed),
    }

    existing_translation = _existing_user_translation(enriched, adaptation.source_text)
    if existing_translation is None and not review_required:
        enriched["translated"] = adaptation.adapted_text
        enriched["traduzido"] = adaptation.adapted_text

    if qa_flags:
        enriched["qa_flags"] = list(dict.fromkeys([*(enriched.get("qa_flags") or []), *qa_flags]))
    return enriched


def _source_text(layer: dict[str, Any]) -> str:
    for key in ("raw_ocr", "normalized_ocr", "normalized_text_final", "original", "text"):
        value = str(layer.get(key) or "").strip()
        if value:
            return value
    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    return str(sfx.get("source_text") or "")


def _existing_user_translation(layer: dict[str, Any], source_text: str) -> str | None:
    source_token = _token(source_text)
    for key in ("translated", "traduzido"):
        value = str(layer.get(key) or "").strip()
        if value and _token(value) != source_token:
            return value
    return None


def _token(value: str) -> str:
    return "".join(str(value or "").split()).casefold()
