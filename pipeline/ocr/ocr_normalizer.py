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

INLINE_MANDATORY_CORRECTIONS: list[tuple[str, str, int]] = [
    (r"\bDWAS\s+UNABLE\s+TO\s+HIRHSTAND\s+TRHE\s+SRIIGSMIANDRANAWAY\s+FROM\s+HOME\b", "WAS UNABLE TO WITHSTAND THE STIGMA AND RAN AWAY FROM HOME", re.IGNORECASE),
    (r"\bSURVIVED\s+COUNTLESS\s+BAUES\.\s+BUTUP\s+MY\s+SADLAND\s+MADE\s+A\s+NAME\s+FOR\s+MYSELF\b", "SURVIVED COUNTLESS BATTLES. BUILT UP MY SKILLS AND MADE A NAME FOR MYSELF", re.IGNORECASE),
    (r"\bALMDST\s+ALL\s+OF\s+US\s+ENDED\s+IP\s+DYNG\b", "ALMOST ALL OF US ENDED UP DYING", re.IGNORECASE),
    (r"\bGHISLAIN\s+PERDIUM,\s+THEMERCENARYKING,\s+AND\s+ONE\s+OF\s+THE\s+CONTINENT'S\s+SEVENSTRONGESTMEN\.", "GHISLAIN PERDIUM, THE MERCENARY KING, AND ONE OF THE CONTINENT'S SEVEN STRONGEST MEN.", re.IGNORECASE),
    (r"\bWELL,\s+THERE'SNOPOINTIN\s+EXPLAINING\s+ITFURTHER\b", "WELL, THERE'S NO POINT IN EXPLAINING IT FURTHER", re.IGNORECASE),
    (r"\bIT'S\s+JUSTA\s+SHAME\s+THAT\s+I\s+WAS\s+UNABLE\s+TOFULFILL\b", "IT'S JUST A SHAME THAT I WAS UNABLE TO FULFILL", re.IGNORECASE),
    (r"\bBACK\s+TTRAVELED\s+TO\s+THE\s+PAST\?", "HAVE I TRAVELED BACK TO THE PAST?", re.IGNORECASE),
    (r"\bRAID\s+SOUAD\b", "RAID SQUAD", re.IGNORECASE),
    (r"\bSOUAD\b", "SQUAD", re.IGNORECASE),
    (r"\bDRCS\b", "ORCS", re.IGNORECASE),
    (r"\bRDC\b", "ORCS", re.IGNORECASE),
    (r"\bDRC\b", "ORC", re.IGNORECASE),
    (r"\bCARBAGE\b", "GARBAGE", re.IGNORECASE),
    (r"\bTRA[Ee]\b", "TRAP", re.IGNORECASE),
    (r"\bOEFENSE\b", "DEFENSE", re.IGNORECASE),
    (r"\bOOWN\b", "DOWN", re.IGNORECASE),
    (r"\bKINGDOME\b", "KINGDOM", re.IGNORECASE),
    (r"\bAGOE\b", "AGO", re.IGNORECASE),
    (r"\bHOUSEHOID\b", "HOUSEHOLD", re.IGNORECASE),
    (r"\bHOUSEHOLDAUOIDEDME\b", "HOUSEHOLD AVOIDED ME", re.IGNORECASE),
    (r"\bREJDICE\b", "REJOICE", re.IGNORECASE),
    (r"(?<![A-Za-z0-9])%NIGHT\b", "KNIGHT", re.IGNORECASE),
    (r"\bTEDQNG\b", "TELLING", re.IGNORECASE),
    (r"\bTS\s+TO\s+EARLY\b", "ITS TOO EARLY", re.IGNORECASE),
    (r"\bTHS\b", "THIS", re.IGNORECASE),
    (r"\bPEOPLE\b", "PEOPLE", re.IGNORECASE),
    (r"\bPEPLE\b", "PEOPLE", re.IGNORECASE),
    (r"\bSTLL\b", "STILL", re.IGNORECASE),
    (r"\bNTO\b", "INTO", re.IGNORECASE),
    (r"\bQUE\$TIONS\b", "QUESTIONS", re.IGNORECASE),
    (r"\bTIO\b", "TO", re.IGNORECASE),
    (r"\bANDAS\b", "AND AS", re.IGNORECASE),
    (r"\bMORETIME\b", "MORE TIME", re.IGNORECASE),
    (r"\bIDIDN'T\b", "I DIDN'T", re.IGNORECASE),
    (r"\bRICARDOE\b", "RICARDO", re.IGNORECASE),
    (r"\bHIRHSTAND\b", "WITHSTAND", re.IGNORECASE),
    (r"\bTRHE\b", "THE", re.IGNORECASE),
    (r"\bIMMATIURITYBORNE\b", "IMMATURITY BORNE", re.IGNORECASE),
    (r"\bFEEUING\b", "FEELING", re.IGNORECASE),
    (r"\bBAUES\b", "BATTLES", re.IGNORECASE),
    (r"\bDYNG\b", "DYING", re.IGNORECASE),
    (r"\bMASTER\.P\b", "MASTER.", re.IGNORECASE),
    (r"\bRALD\b", "RAID", re.IGNORECASE),
    (r"\bRGHT\b", "RIGHT", re.IGNORECASE),
    (r"\bSTHRT\b", "START", re.IGNORECASE),
    (r"\brt\b", "it", re.IGNORECASE),
    (r"\bMSTAES\b", "MISTAKES", re.IGNORECASE),
    (r"\blfe\b", "LIFE", re.IGNORECASE),
    (r"\bln\b", "IN", re.IGNORECASE),
    (r"\bSHOUID\b", "SHOULD", re.IGNORECASE),
    (r"\bSINCETHOSE\b", "SINCE THOSE", re.IGNORECASE),
    (r"\bHAVEA\b", "HAVE A", re.IGNORECASE),
    (r"\bWlll\b", "WILL", re.IGNORECASE),
    (r"(?<![A-Za-z0-9])THIN%(?![A-Za-z0-9])", "THINK", re.IGNORECASE),
    (r"\bldun\b", "Idun", re.IGNORECASE),
]

COMMON_WORDS = {"THE", "AND", "YOU", "FOR", "ARE", "IS", "A", "I", "TO", "WE", "NO", "YES", "OK"}
SHORT_QUOTED_DIALOGUE_WORDS = {"I", "WE", "YOU", "NO", "YES", "OK"}
KNOWN_LATIN_OCR_ARTIFACTS = {"HFOR"}
CJK_LETTER_PATTERN = re.compile(
    r"[\u1100-\u11FF\u3000-\u303F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]"
)


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
    if _is_short_quoted_dialogue(text):
        return False
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if CJK_LETTER_PATTERN.search(compact):
        return False
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


def _apply_inline_mandatory(text: str) -> tuple[str, list[OcrCorrection]]:
    result = text
    corrections: list[OcrCorrection] = []
    for pattern, replacement, flags in INLINE_MANDATORY_CORRECTIONS:
        updated = re.sub(pattern, replacement, result, flags=flags)
        if updated != result:
            corrections.append(
                OcrCorrection(result, updated, "mandatory_ocr_inline_correction", 1.0)
            )
            result = updated
    return result, corrections


def _is_scanlation_credit(text: str) -> bool:
    normalized = _norm_key(text)
    compact = re.sub(r"[^A-Z0-9]+", "", normalized)
    if not normalized:
        return False
    if any(token in compact for token in ("ASURA", "ASURASCANS", "ASURASOANS", "ILEAFSKY")):
        return True
    if "FASTEST RELEASES" in normalized:
        return True
    if ". COM" in normalized or normalized.endswith(".COM") or "DISCORD" in normalized:
        return True
    return False


def _is_short_quoted_dialogue(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped or not re.search(r"[?!]", stripped):
        return False
    words = re.findall(r"[A-Za-z]+", stripped)
    if len(words) != 1:
        return False
    return words[0].upper() in SHORT_QUOTED_DIALOGUE_WORDS


def _is_known_latin_ocr_artifact(text: str) -> bool:
    normalized = re.sub(r"[^A-Z]", "", str(text or "").upper())
    return normalized in KNOWN_LATIN_OCR_ARTIFACTS


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
        normalized, new = _apply_inline_mandatory(normalized)
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
    if _is_known_latin_ocr_artifact(raw):
        flags = list(updated.get("qa_flags") or [])
        if "suspected_ocr_error" not in flags:
            flags.append("suspected_ocr_error")
        updated["qa_flags"] = flags
        updated["text"] = raw
        updated["skip_processing"] = True
        updated["skip_reason"] = "ocr_artifact"
        return updated
    if _is_scanlation_credit(raw) or _is_scanlation_credit(normalized["normalized_ocr"]):
        flags = list(updated.get("qa_flags") or [])
        if "scanlation_credit" not in flags:
            flags.append("scanlation_credit")
        updated["qa_flags"] = flags
        updated["text"] = raw
        updated["skip_processing"] = True
        updated["skip_reason"] = "scanlation_credit"
        return updated
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

