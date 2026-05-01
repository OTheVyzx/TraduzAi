"""OCR normalization before translation."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


MANDATORY_CORRECTIONS = {
    "RAID SOUAD": "RAID SQUAD",
    "DRCS": "ORCS",
    "RDC": "ORCS",
    "CARBAGE": "GARBAGE",
    "TRAE": "TRAP",
    "FENRISNOW": "FENRIS NOW",
}

COMMON_WORDS = {"THE", "AND", "YOU", "FOR", "ARE", "IS", "A", "I", "TO"}


@dataclass
class OcrCorrection:
    from_text: str
    to: str
    reason: str
    confidence: float

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["from"] = data.pop("from_text")
        return data


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).upper()


def _is_gibberish(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    alpha = sum(ch.isalpha() for ch in compact)
    if alpha / max(1, len(compact)) < 0.45:
        return True
    if re.search(r"([A-Z])\1{5,}", compact.upper()):
        return True
    return False


def _apply_mandatory(text: str) -> tuple[str, list[OcrCorrection]]:
    key = _norm_key(text)
    if key in MANDATORY_CORRECTIONS:
        target = MANDATORY_CORRECTIONS[key]
        return target, [OcrCorrection(text, target, "mandatory_ocr_correction", 1.0)]
    return text, []


def _apply_glossary_fuzzy(text: str, glossary: dict[str, str] | None) -> tuple[str, list[OcrCorrection]]:
    if not glossary:
        return text, []
    key = _norm_key(text)
    if key in COMMON_WORDS or len(key) < 4:
        return text, []
    best: tuple[str, float] | None = None
    for source in glossary:
        source_key = _norm_key(source)
        if source_key in COMMON_WORDS or len(source_key) < 4:
            continue
        score = _similarity(key, source_key)
        if score >= 0.84 and (best is None or score > best[1]):
            best = (source, score)
    if best is None:
        return text, []
    return best[0], [OcrCorrection(text, best[0], "glossary_fuzzy_match", best[1])]


def _apply_stutter(text: str, glossary: dict[str, str] | None) -> tuple[str, list[OcrCorrection]]:
    match = re.match(r"^([A-Za-z])-([A-Za-z][A-Za-z ]+)([?!。！？.]*)$", text.strip())
    if not match or not glossary:
        return text, []
    _, term, punctuation = match.groups()
    target = glossary.get(term.upper()) or glossary.get(term.title()) or glossary.get(term)
    if not target:
        return text, []
    normalized = f"{target[0]}-{target}{punctuation}"
    return normalized, [OcrCorrection(text, normalized, "stutter_glossary_translation", 0.95)]


def normalize_ocr_text(text: str, glossary: dict[str, str] | None = None) -> dict[str, Any]:
    raw = text or ""
    normalized = raw
    corrections: list[OcrCorrection] = []

    normalized, new = _apply_stutter(normalized, glossary)
    corrections.extend(new)
    if not corrections:
        normalized, new = _apply_mandatory(normalized)
        corrections.extend(new)
    if not corrections:
        normalized, new = _apply_glossary_fuzzy(normalized, glossary)
        corrections.extend(new)

    is_gibberish = _is_gibberish(normalized)
    return {
        "raw_ocr": raw,
        "normalized_ocr": normalized,
        "normalization": {
            "changed": normalized != raw,
            "corrections": [item.to_json() for item in corrections],
            "is_gibberish": is_gibberish,
        },
    }


def normalize_ocr_record(record: dict[str, Any], glossary: dict[str, str] | None = None) -> dict[str, Any]:
    raw = str(record.get("raw_ocr") or record.get("text") or record.get("original") or "")
    normalized = normalize_ocr_text(raw, glossary)
    updated = dict(record)
    updated.update(normalized)
    if not normalized["normalization"]["is_gibberish"]:
        updated["text"] = normalized["normalized_ocr"]
    else:
        flags = list(updated.get("qa_flags") or [])
        if "ocr_gibberish" not in flags:
            flags.append("ocr_gibberish")
        updated["qa_flags"] = flags
        updated["skip_processing"] = True
    return updated


def _similarity(left: str, right: str) -> float:
    max_len = max(len(left), len(right))
    if max_len == 0:
        return 1.0
    return 1.0 - (_levenshtein(left, right) / max_len)


def _levenshtein(left: str, right: str) -> int:
    prev = list(range(len(right) + 1))
    for i, lc in enumerate(left, 1):
        cur = [i]
        for j, rc in enumerate(right, 1):
            cur.append(prev[j - 1] if lc == rc else 1 + min(prev[j], cur[j - 1], prev[j - 1]))
        prev = cur
    return prev[-1]

