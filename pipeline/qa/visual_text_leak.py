"""Visual text leak QA checks."""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

ENGLISH_PATTERNS = [
    re.compile(r"\bYOUNG MASTER\b", re.I),
    re.compile(r"\bOLD MAN\b", re.I),
    re.compile(r"\bRAID SQUAD\b", re.I),
    re.compile(r"\bWHAT\b|\bWHY\b|\bTHE\b", re.I),
]


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


def _has_english_leak(text: str, allowed_terms: list[str], expected_layers: list[dict[str, Any]]) -> bool:
    filtered = text
    for term in allowed_terms:
        filtered = re.sub(re.escape(term), "", filtered, flags=re.I)
    if all(layer.get("tipo") == "sfx" and layer.get("preserve_original") for layer in expected_layers):
        return False
    return any(pattern.search(filtered) for pattern in ENGLISH_PATTERNS)


def _layer_has_detectable_text(layer: dict[str, Any]) -> bool:
    if layer.get("ignored_reason") or layer.get("skip_processing"):
        return False
    if layer.get("tipo") == "sfx" and layer.get("preserve_original"):
        return False
    return bool(layer.get("original") or layer.get("raw_ocr") or layer.get("text"))

