"""Visual text leak QA checks."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

import numpy as np

def detect_visual_text_leak(
    *,
    page: int,
    original_image: np.ndarray | None = None,
    final_image: np.ndarray | None = None,
    final_ocr_text: str = "",
    expected_layers: list[dict[str, Any]] | None = None,
    allowed_terms: list[str] | None = None,
    ocr_reader: Callable[[np.ndarray], str] | None = None,
) -> list[dict[str, Any]]:
    allowed_terms = allowed_terms or []
    expected_layers = expected_layers or []
    text = final_ocr_text
    if not text and ocr_reader is not None and final_image is not None:
        text = ocr_reader(final_image)

    flags: list[dict[str, Any]] = []
    if text and _has_english_leak(text, allowed_terms, expected_layers):
        flags.append({
            "type": "visual_text_leak",
            "severity": "critical",
            "found_text": text,
            "page": page,
            "action": "block_export_or_warn",
        })

    if original_image is not None and final_image is not None and expected_layers:
        similarity = image_similarity(original_image, final_image)
        if similarity >= 0.99 and any(_layer_has_detectable_text(layer) for layer in expected_layers):
            flags.append({
                "type": "page_not_processed",
                "severity": "critical",
                "page": page,
                "similarity": round(similarity, 4),
            })
    return flags


def image_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    diff = np.mean(np.abs(a.astype("float32") - b.astype("float32"))) / 255.0
    return max(0.0, 1.0 - float(diff))


def _source_tokens(value: Any) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return tuple(
        token
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        if len(token) >= 2
    )


def _payload_visible(source: str, observed: str) -> bool:
    source_tokens = _source_tokens(source)
    observed_tokens = _source_tokens(observed)
    if not source_tokens or not observed_tokens:
        return False
    observed_set = set(observed_tokens)
    if len(source_tokens) == 1:
        return len(source_tokens[0]) >= 4 and source_tokens[0] in observed_set
    if set(zip(source_tokens, source_tokens[1:])) & set(
        zip(observed_tokens, observed_tokens[1:])
    ):
        return True
    matched = sum(token in observed_set for token in source_tokens)
    return matched >= 2 and matched / len(source_tokens) >= 0.75


def _has_english_leak(text: str, allowed_terms: list[str], expected_layers: list[dict[str, Any]]) -> bool:
    filtered = text
    for term in allowed_terms:
        filtered = re.sub(re.escape(term), "", filtered, flags=re.I)
    for layer in expected_layers:
        if _is_preserved_sfx_layer(layer):
            for value in (layer.get("original"), layer.get("raw_ocr"), layer.get("text")):
                token = str(value or "").strip()
                if token:
                    filtered = re.sub(re.escape(token), "", filtered, flags=re.I)
    source_payloads = [
        str(value).strip()
        for layer in expected_layers
        if not _is_preserved_sfx_layer(layer)
        for value in (layer.get("original"), layer.get("raw_ocr"), layer.get("text"))
        if str(value or "").strip()
    ]
    return any(_payload_visible(source, filtered) for source in source_payloads)


def _layer_has_detectable_text(layer: dict[str, Any]) -> bool:
    return bool(layer.get("original") or layer.get("raw_ocr") or layer.get("text"))


def _is_preserved_sfx_layer(layer: dict[str, Any]) -> bool:
    kind = str(layer.get("tipo") or layer.get("content_class") or "").strip().lower()
    return kind == "sfx" and bool(layer.get("preserve_original") or layer.get("skip_processing"))

