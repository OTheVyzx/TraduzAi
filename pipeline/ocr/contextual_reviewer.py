"""
Context-aware reviewer for OCR results.
Uses a short page/history lexicon to repair low-confidence words that already
appear in cleaner form elsewhere in the chapter.
"""

from __future__ import annotations

import re

from ocr.text_normalizer import normalize_text


def contextual_review_page(
    page_result: dict,
    previous_pages: list[dict],
    expected_terms: list[str] | None = None,
) -> dict:
    texts = _dedupe_geometry_blocks(page_result.get("texts", []), page_result)
    lexicon = _build_context_lexicon(texts, previous_pages, expected_terms or [])

    reviewed_texts = []
    for text_data in texts:
        original = text_data.get("text", "")
        confidence = float(text_data.get("confidence", 0.0))
        reviewed = _apply_context_to_text(original, confidence, lexicon)
        normalization = normalize_text(
            reviewed,
            text_id=text_data.get("id") or text_data.get("text_id"),
            confidence=confidence,
        )
        reviewed = normalization["normalized"]
        updated = dict(text_data)
        updated["text"] = reviewed
        updated["ocr_context_reviewed"] = reviewed != original
        updated["raw_ocr"] = text_data.get("raw_ocr", original)
        updated["normalized_ocr"] = reviewed
        updated["normalized_text_final"] = reviewed
        updated["normalization"] = normalization
        updated["normalization_trace"] = normalization
        if normalization.get("joined_word_suspect"):
            updated["qa_flags"] = _merge_qa_flags(updated.get("qa_flags"), ["ocr_run_on_suspect"])
        if _normalization_needs_joined_word_review(normalization):
            updated["qa_flags"] = _merge_qa_flags(updated.get("qa_flags"), ["ocr_joined_word_review"])
            updated["needs_review"] = True
        _record_normalization_debug(updated, normalization)
        reviewed_texts.append(updated)

    updated_page = dict(page_result)
    updated_page["texts"] = reviewed_texts
    return updated_page


def _dedupe_geometry_blocks(texts: list[dict], page_result: dict) -> list[dict]:
    if len(texts) < 2:
        return texts

    groups: dict[tuple[tuple[int, int, int, int] | None, str], list[dict]] = {}
    for text_data in texts:
        signature = _text_signature(text_data.get("text", ""))
        if not signature:
            continue
        balloon_bbox = _bbox_tuple(text_data.get("balloon_bbox"))
        source_bbox = _bbox_tuple(text_data.get("source_bbox") or text_data.get("bbox"))
        group_key = (balloon_bbox, signature) if balloon_bbox else (source_bbox, signature)
        groups.setdefault(group_key, []).append(text_data)

    dropped_ids: set[int] = set()
    for duplicates in groups.values():
        if len(duplicates) < 2:
            continue
        ranked = sorted(
            duplicates,
            key=lambda item: (
                _geometry_quality_score(item),
                float(item.get("confidence") or 0.0),
                -_bbox_area(_bbox_tuple(item.get("source_bbox") or item.get("bbox"))),
            ),
            reverse=True,
        )
        kept = ranked[0]
        kept_score = _geometry_quality_score(kept)
        for dropped in ranked[1:]:
            dropped_score = _geometry_quality_score(dropped)
            dropped_ids.add(id(dropped))
            _record_dedupe_decision(page_result, kept, dropped, kept_score, dropped_score)

    if not dropped_ids:
        return texts
    return [text_data for text_data in texts if id(text_data) not in dropped_ids]


def _geometry_quality_score(text_data: dict) -> float:
    confidence = float(text_data.get("confidence") or 0.0)
    score = confidence * 100.0
    source_bbox = _bbox_tuple(text_data.get("source_bbox") or text_data.get("bbox"))
    text_pixel_bbox = _bbox_tuple(text_data.get("text_pixel_bbox") or text_data.get("bbox"))
    balloon_bbox = _bbox_tuple(text_data.get("balloon_bbox"))

    source_area = _bbox_area(source_bbox)
    text_area = _bbox_area(text_pixel_bbox)
    if source_area and text_area and source_area / float(text_area) > 8.0:
        score -= 45.0
    if not text_data.get("line_polygons"):
        score -= 25.0
    if confidence < 0.75:
        score -= 20.0
    if _is_white_balloon_with_non_white_background(text_data):
        score -= 15.0
    if source_bbox is not None and balloon_bbox is not None and source_bbox == balloon_bbox:
        score -= 50.0
    return round(score, 3)


def _is_white_balloon_with_non_white_background(text_data: dict) -> bool:
    if str(text_data.get("balloon_type") or "").strip().lower() != "white":
        return False
    rgb = text_data.get("background_rgb")
    if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
        return False
    try:
        channels = [int(value) for value in rgb[:3]]
    except (TypeError, ValueError):
        return False
    return any(channel < 245 for channel in channels)


def _record_dedupe_decision(
    page_result: dict,
    kept: dict,
    dropped: dict,
    kept_score: float,
    dropped_score: float,
) -> None:
    try:
        from debug_tools import get_recorder

        recorder = get_recorder()
        if recorder and recorder.enabled:
            recorder.write_jsonl(
                "03_ocr/ocr_dedupe_decisions.jsonl",
                {
                    "action": "dedupe_blocks",
                    "page": page_result.get("page") or page_result.get("page_number"),
                    "reason": "geometry_quality_score",
                    "kept_score": kept_score,
                    "dropped_score": dropped_score,
                    "kept": _dedupe_trace_entry(kept),
                    "dropped": _dedupe_trace_entry(dropped),
                },
            )
    except Exception:
        return


def _dedupe_trace_entry(text_data: dict) -> dict:
    return {
        "text_id": text_data.get("id") or text_data.get("text_id"),
        "text": text_data.get("text"),
        "bbox": _bbox_tuple(text_data.get("bbox")),
        "source_bbox": _bbox_tuple(text_data.get("source_bbox")),
        "text_pixel_bbox": _bbox_tuple(text_data.get("text_pixel_bbox")),
        "balloon_bbox": _bbox_tuple(text_data.get("balloon_bbox")),
        "confidence": text_data.get("confidence"),
        "line_polygons_count": len(text_data.get("line_polygons") or []),
    }


def _bbox_tuple(value) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _text_signature(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(text or "").upper())


def _merge_qa_flags(existing, additions: list[str]) -> list[str]:
    merged: list[str] = []
    for flag in [*(existing or []), *additions]:
        if flag and flag not in merged:
            merged.append(str(flag))
    return merged


def _normalization_needs_joined_word_review(normalization: dict) -> bool:
    if not normalization.get("joined_word_suspect"):
        return False
    confidence_after = normalization.get("confidence_after_estimate")
    try:
        return float(confidence_after) < 0.7
    except (TypeError, ValueError):
        return True


def _record_normalization_debug(text_data: dict, normalization: dict) -> None:
    if not normalization.get("changed"):
        return
    try:
        from debug_tools import get_recorder

        recorder = get_recorder()
        if recorder and recorder.enabled:
            recorder.write_jsonl("04_text_normalization_router/normalization_trace.jsonl", normalization)
            if normalization.get("joined_word_suspect"):
                recorder.write_jsonl("04_text_normalization_router/joined_word_splits.jsonl", normalization)
    except Exception:
        return


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
