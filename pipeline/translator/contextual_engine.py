"""Structured contextual translation engine."""

from __future__ import annotations

import json
from typing import Any, Protocol

from translator.term_protection import protect_terms, restore_terms


class SegmentTranslator(Protocol):
    def translate(self, payload: dict[str, Any]) -> str:
        ...


class MockTranslator:
    def __init__(self, response: dict[str, Any] | None = None, raw: str | None = None):
        self.response = response
        self.raw = raw

    def translate(self, payload: dict[str, Any]) -> str:
        if self.raw is not None:
            return self.raw
        return json.dumps(self.response or {"segments": []}, ensure_ascii=False)


def build_contextual_payload(
    *,
    work: dict[str, Any],
    chapter: str,
    page: int,
    source_language: str,
    target_language: str,
    glossary: list[dict[str, Any]],
    previous_segments: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    scene_summary: str = "",
    style: str = "Português brasileiro natural de manhwa de fantasia medieval.",
) -> dict[str, Any]:
    protected_segments = []
    for segment in segments:
        protected = protect_terms(segment["source"], glossary)
        protected_segments.append(
            {
                **segment,
                "protected_source": protected["protected_source"],
                "protected_terms": protected["terms"],
            }
        )
    return {
        "work": work,
        "chapter": chapter,
        "page": page,
        "scene_summary": scene_summary,
        "source_language": source_language,
        "target_language": target_language,
        "style": style,
        "glossary": glossary,
        "previous_segments": previous_segments,
        "segments": protected_segments,
    }


def parse_translation_response(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Resposta de traducao nao e JSON valido: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("segments"), list):
        raise ValueError("Resposta de traducao sem lista 'segments'.")
    return parsed


def translate_contextual(payload: dict[str, Any], translator: SegmentTranslator) -> dict[str, Any]:
    try:
        parsed = parse_translation_response(translator.translate(payload))
    except Exception as exc:
        return _fallback(payload, f"translation_api_failed:{exc}")

    by_id = {segment.get("id"): segment for segment in parsed["segments"] if isinstance(segment, dict)}
    output_segments = []
    for segment in payload["segments"]:
        response = by_id.get(segment["id"])
        if not response or not response.get("translation"):
            output_segments.append(_fallback_segment(segment, "missing_translation"))
            continue
        restored = restore_terms(str(response["translation"]), segment.get("protected_terms", []))
        warnings = list(response.get("warnings") or [])
        warnings.extend(flag["reason"] for flag in restored["flags"])
        output_segments.append(
            {
                "id": segment["id"],
                "translation": restored["text"],
                "confidence": float(response.get("confidence", 0.0) or 0.0),
                "used_glossary": list(response.get("used_glossary") or []),
                "warnings": warnings,
                "qa_flags": ["blocked"] if restored["blocked"] else [],
            }
        )
    return {"segments": output_segments}


def _fallback(payload: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"segments": [_fallback_segment(segment, reason) for segment in payload.get("segments", [])]}


def _fallback_segment(segment: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": segment["id"],
        "translation": segment.get("source", ""),
        "confidence": 0.0,
        "used_glossary": [],
        "warnings": [reason],
        "qa_flags": ["needs_translation_review"],
    }

