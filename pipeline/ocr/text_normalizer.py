from __future__ import annotations

import re
from typing import Any


OCR_TOKEN_CORRECTIONS: dict[str, str] = {
    "VHEN": "WHEN",
}

JOINED_WORD_SPLITS_V2: dict[str, str] = {
    "AISHIT": "AISH! IT",
    "AISHIT'S": "AISH! IT'S",
    "AISHIT'SNOT": "AISH! IT'S NOT",
    "CANYOUFINDAGOOD": "CAN YOU FIND A GOOD",
    "THATGIVESINTERESTUP": "THAT GIVES INTEREST UP",
    "TILLTHREEMONTHS": "TILL THREE MONTHS",
    "TOSHOWYOUR": "TO SHOW YOUR",
    "TOBELIEVE": "TO BELIEVE",
    "WE'REFOOL'S": "WE'RE FOOLS",
    "AJUMMAYOU": "AJUMMA, YOU",
    "THERE'SNO": "THERE'S NO",
    "GETMONEYFROM": "GET MONEY FROM",
    "ONLYTHINKS": "ONLY THINKS",
    "SOWHY": "SO WHY",
    "TOMORROW'SPROBLEMS": "TOMORROW'S PROBLEMS",
    "EVENTHINK": "EVEN THINK",
    "PAYUSBACK": "PAY US BACK",
    "CANDIE": "CAN DIE",
    "IDON'T": "I DON'T",
    "IGETBACK": "I GET BACK",
    "LET'SJUST": "LET'S JUST",
    "REAL-LIFEINSURANCE": "REAL-LIFE INSURANCE",
    "TOWORK": "TO WORK",
}


def _normalize_token_key(token: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9']+$", "", str(token or "")).upper()


def joined_word_suspect(token: str) -> bool:
    return _normalize_token_key(token) in JOINED_WORD_SPLITS_V2


def split_joined_word(token: str) -> str | None:
    return JOINED_WORD_SPLITS_V2.get(_normalize_token_key(token))


def normalize_text(
    text: str,
    *,
    text_id: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    raw = str(text or "")
    normalized_tokens: list[str] = []
    token_diff: list[dict[str, str]] = []
    rules_applied: list[str] = []
    has_joined_word = False

    for token in raw.split():
        before, core, after = _split_outer_punctuation(token)
        key = core.upper()
        replacement = OCR_TOKEN_CORRECTIONS.get(key)
        rule = "ocr_token_correction" if replacement is not None else None
        if replacement is None:
            replacement = JOINED_WORD_SPLITS_V2.get(key) or split_joined_word(core)
            if replacement is not None:
                rule = "joined_word_suspect"
                has_joined_word = True
        if replacement is None:
            replacement = core

        normalized_tokens.append(f"{before}{replacement}{after}")
        if replacement != core:
            token_diff.append({"from": core, "to": replacement, "rule": rule or "normalization"})
            if rule == "joined_word_suspect" and "split_joined_words" not in rules_applied:
                rules_applied.append("split_joined_words")
            elif rule and rule != "joined_word_suspect" and rule not in rules_applied:
                rules_applied.append(rule)
        elif key in JOINED_WORD_SPLITS_V2 or joined_word_suspect(core):
            has_joined_word = True

    normalized = _normalize_spacing(" ".join(normalized_tokens))
    normalized, phrase_rules = _apply_phrase_repairs(normalized)
    for phrase_rule in phrase_rules:
        if phrase_rule not in rules_applied:
            rules_applied.append(phrase_rule)
    changed = normalized != raw
    needs_review = has_joined_word or len(token_diff) >= 2 or (confidence is not None and confidence < 0.65)
    result = {
        "text_id": text_id,
        "raw": raw,
        "normalized": normalized,
        "changed": changed,
        "rules_applied": rules_applied,
        "token_diff": token_diff,
        "joined_word_suspect": has_joined_word,
        "confidence_before": confidence,
        "confidence_after_estimate": _estimate_confidence(confidence, token_diff),
        "needs_review": needs_review,
        "review_reason": _review_reason(needs_review, confidence, token_diff, has_joined_word),
    }
    return result


def _apply_phrase_repairs(text: str) -> tuple[str, list[str]]:
    repaired = str(text or "")
    rules: list[str] = []

    updated = _repair_missing_punctuation_spacing(repaired)
    if updated != repaired:
        repaired = updated
        rules.append("repair_missing_punctuation_spacing")

    patterns = [
        (
            r"^\s*ANCE,\s*(?:TE\s+)?YOU\s+YOU\s+KNOW,",
            "INSURANCE, YOU KNOW,",
            "repair_leading_insurance_fragment",
        ),
        (
            r"\bTE\s+YOU\s+YOU\s+KNOW\b",
            "YOU KNOW",
            "repair_repeated_you_know",
        ),
    ]
    for pattern, replacement, rule in patterns:
        updated = re.sub(pattern, replacement, repaired, flags=re.IGNORECASE)
        if updated != repaired:
            repaired = updated
            rules.append(rule)

    return _normalize_spacing(repaired), rules


def _repair_missing_punctuation_spacing(text: str) -> str:
    repaired = str(text or "")
    repaired = re.sub(r"([!?]+)([\"']?)(?=[A-Za-z])", r"\1\2 ", repaired)
    repaired = re.sub(r",(?=[A-Za-z])", ", ", repaired)
    return repaired


def _split_outer_punctuation(token: str) -> tuple[str, str, str]:
    match = re.match(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9']*)$", token)
    if not match:
        return "", token, ""
    before, core, after = match.groups()
    return before, core, after


def _normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _estimate_confidence(confidence: float | None, token_diff: list[dict[str, str]]) -> float | None:
    if confidence is None:
        return None
    bump = 0.075 * len(token_diff)
    return round(min(0.99, confidence + bump), 3)


def _review_reason(
    needs_review: bool,
    confidence: float | None,
    token_diff: list[dict[str, str]],
    has_joined_word: bool,
) -> str | None:
    if not needs_review:
        return None
    if confidence is not None and confidence < 0.65 and token_diff:
        return "low_initial_confidence_and_corrections"
    if has_joined_word:
        return "joined_word_suspect"
    if token_diff:
        return "normalization_changed_text"
    return "low_initial_confidence"
