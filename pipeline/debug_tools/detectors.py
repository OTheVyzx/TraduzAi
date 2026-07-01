from __future__ import annotations

import re
from typing import Any


JOINED_WORD_SPLITS: dict[str, str] = {
    "ITSAEXAMPLE": "ITS A EXAMPLE",
    "IGETBACK": "I GET BACK",
    "TOWORK": "TO WORK",
    "CANYOUFIND": "CAN YOU FIND",
    "AISHIT'SNOT": "AI SHIT'S NOT",
    "SOPLEASE": "SO PLEASE",
    "ALITTLELONGER": "A LITTLE LONGER",
    "AREWAITI": "ARE WAITI",
    "HOSPITALBILLS": "HOSPITAL BILLS",
    "REAL-LIFEINSURANCE": "REAL-LIFE INSURANCE",
    "LOANFOR": "LOAN FOR",
    "WHENI": "WHEN I",
}

SUSPECT_OCR_TOKENS: set[str] = {
    *JOINED_WORD_SPLITS.keys(),
    "VHEN",
    "OWT",
    "AU",
}

SFX_MARKER_PATTERN = re.compile(r"\bSFX(?:\s*:?\s*[A-Za-z0-9'_-]+)?", re.IGNORECASE)
MOJIBAKE_PATTERN = re.compile(
    # Double-encoded UTF-8 markers such as "VOCÃƒÅ ", "JÃƒï¿½",
    # or "CÃ‚NCER". Do not flag valid Portuguese letters like Ã/Â
    # by themselves ("NÃO", "CÂNCER").
    r"(?:Ãƒ[^\s]{0,3})"
    r"|(?:Ã‚[^\s]{0,3})"
    r"|(?:Ã„[^\s]{0,3})"
    r"|(?:ÃÅ[^\s]{0,3})"
    r"|(?:â[€\u0080-\u009f][^\s]{0,3})"
    r"|(?:ï¿½)"
    r"|(?:\ufffd)"
    r"|(?:[\ud800-\udfff])",
    re.UNICODE,
)

MOJIBAKE_PATTERN = re.compile(
    r"(?:\u00c3[\u0080-\u00bf\u0100-\u017f\u0192\u201a-\u2026\u2030\u20ac])"
    r"|(?:\u00c2[\u0080-\u00bf])"
    r"|(?:\u00e2[\u0080-\u009f\u20ac][^\s]{0,2})"
    r"|(?:[\u0102\u0103\u0118\u0119])"
    r"|(?:\ufffd)"
    r"|(?:[\ud800-\udfff])",
    re.UNICODE,
)

CP1250_PORTUGUESE_MOJIBAKE = str.maketrans(
    {
        "\u0102": "\u00c3",  # NĂO -> NÃO, MĂE -> MÃE
        "\u0103": "\u00e3",
        "\u0118": "\u00ca",  # VOCĘ -> VOCÊ, TRĘS -> TRÊS
        "\u0119": "\u00ea",
    }
)


def normalize_token_key(token: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9']+$", "", str(token or "")).upper()


def joined_word_suspect(token: str) -> bool:
    return normalize_token_key(token) in SUSPECT_OCR_TOKENS


def split_joined_word(token: str) -> str | None:
    return JOINED_WORD_SPLITS.get(normalize_token_key(token))


def has_sfx_marker(text: str) -> bool:
    return bool(SFX_MARKER_PATTERN.search(str(text or "")))


def mojibake_samples(text: str) -> list[str]:
    samples: list[str] = []
    for match in MOJIBAKE_PATTERN.finditer(str(text or "")):
        sample = match.group(0)
        if sample not in samples:
            samples.append(sample)
    return samples


def has_mojibake(text: str) -> bool:
    return bool(mojibake_samples(text))


def _decode_cp1252_utf8_once(text: str) -> str | None:
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def fix_mojibake(text: str, *, max_passes: int = 3) -> str:
    current = str(text or "")
    for _ in range(max(1, max_passes)):
        translated = current.translate(CP1250_PORTUGUESE_MOJIBAKE)
        if translated != current:
            current = translated
            if not has_mojibake(current):
                break
            continue
        decoded = _decode_cp1252_utf8_once(current)
        if decoded is None or decoded == current:
            break
        current = decoded
        if not has_mojibake(current):
            break
    return current


def audit_mojibake(
    text: str,
    *,
    text_id: str | None = None,
    stage: str = "translation_output",
) -> dict[str, Any]:
    translated = str(text or "")
    samples = mojibake_samples(translated)
    suggested_fix = fix_mojibake(translated) if samples else translated
    has_issue = bool(samples)
    return {
        "text_id": text_id,
        "stage": stage,
        "translated": translated,
        "mojibake_match_count": len(samples),
        "mojibake_samples": samples,
        "suggested_fix": suggested_fix,
        "fix_method": "decode_cp1252_encode_utf8_safe" if has_issue else "none",
        "flags": ["mojibake_in_translation"] if has_issue else [],
        "mojibake_in_translation": has_issue,
    }
