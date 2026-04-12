"""
Context-aware reviewer for OCR results.
Uses a short page/history lexicon to repair low-confidence words that already
appear in cleaner form elsewhere in the chapter.
"""

from __future__ import annotations

import re


def contextual_review_page(
    page_result: dict,
    previous_pages: list[dict],
    expected_terms: list[str] | None = None,
) -> dict:
    texts = page_result.get("texts", [])
    lexicon = _build_context_lexicon(texts, previous_pages, expected_terms or [])

    reviewed_texts = []
    for text_data in texts:
        original = text_data.get("text", "")
        confidence = float(text_data.get("confidence", 0.0))
        reviewed = _apply_context_to_text(original, confidence, lexicon)
        updated = dict(text_data)
        updated["text"] = reviewed
        updated["ocr_context_reviewed"] = reviewed != original
        reviewed_texts.append(updated)

    updated_page = dict(page_result)
    updated_page["texts"] = reviewed_texts
    return updated_page


def _build_context_lexicon(
    current_texts: list[dict],
    previous_pages: list[dict],
    expected_terms: list[str],
) -> dict[str, str]:
    lexicon: dict[str, str] = {}
    candidates = []
    for page in [*previous_pages, {"texts": current_texts}]:
        candidates.extend(page.get("texts", []))

    for text_data in candidates:
        confidence = float(text_data.get("confidence", 0.0))
        if confidence < 0.82:
            continue
        for token in _tokenize(text_data.get("text", "")):
            if len(token) < 3:
                continue
            lexicon[_ocr_signature(token)] = token

    for term in expected_terms:
        for token in _tokenize(term):
            if len(token) < 3:
                continue
            lexicon[_ocr_signature(token)] = token
    return lexicon


def _apply_context_to_text(text: str, confidence: float, lexicon: dict[str, str]) -> str:
    if confidence >= 0.8:
        return text

    parts = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9']+", text)
    changed = False
    reviewed_parts = []
    for part in parts:
        if re.fullmatch(r"[A-Za-z0-9']+", part):
            replacement = _replace_token_from_lexicon(part, lexicon)
            changed = changed or (replacement != part)
            reviewed_parts.append(replacement)
        else:
            reviewed_parts.append(part)

    return "".join(reviewed_parts) if changed else text


def _replace_token_from_lexicon(token: str, lexicon: dict[str, str]) -> str:
    suffix = ""
    if token and token[-1] in "!?.,;:":
        token, suffix = token[:-1], token[-1]

    signature = _ocr_signature(token)
    replacement = lexicon.get(signature)
    if not replacement:
        return f"{token}{suffix}"

    if token.isupper():
        replacement = replacement.upper()
    elif token[:1].isupper():
        replacement = replacement.capitalize()
    else:
        replacement = replacement.lower()
    return f"{replacement}{suffix}"


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9']+", text.upper()) if token]


def _ocr_signature(token: str) -> str:
    normalized = token.upper()
    translation = str.maketrans({
        "0": "O",
        "1": "I",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
        "|": "I",
    })
    return normalized.translate(translation).replace("'", "")
