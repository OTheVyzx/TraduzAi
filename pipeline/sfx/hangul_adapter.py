from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


HANGUL_RE = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
ASCII_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
LEXICON_PATH = Path(__file__).with_name("lexicon_ko_pt.json")


@dataclass(frozen=True)
class SfxAdaptation:
    source_text: str
    adapted_text: str
    confidence: float
    kind: str
    review_required: bool
    qa_flags: list[str]


def adapt_hangul_sfx(text: str) -> SfxAdaptation:
    source_text = _normalize_sfx_text(text)

    if not source_text:
        return SfxAdaptation(
            source_text="",
            adapted_text="",
            confidence=0.0,
            kind="empty",
            review_required=True,
            qa_flags=["empty_sfx"],
        )

    if not HANGUL_RE.search(source_text):
        return SfxAdaptation(
            source_text=source_text,
            adapted_text=source_text,
            confidence=0.0,
            kind="non_hangul",
            review_required=True,
            qa_flags=["non_hangul_sfx"],
        )

    if ASCII_ALNUM_RE.search(source_text):
        return SfxAdaptation(
            source_text=source_text,
            adapted_text=source_text,
            confidence=0.0,
            kind="mixed_script",
            review_required=True,
            qa_flags=["mixed_script_sfx"],
        )

    entry = _load_lexicon().get(source_text)
    if entry is None:
        return SfxAdaptation(
            source_text=source_text,
            adapted_text=source_text,
            confidence=0.0,
            kind="unknown",
            review_required=True,
            qa_flags=["unknown_sfx"],
        )

    return SfxAdaptation(
        source_text=source_text,
        adapted_text=str(entry["pt"]),
        confidence=float(entry["confidence"]),
        kind=str(entry["kind"]),
        review_required=False,
        qa_flags=[],
    )


def _normalize_sfx_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", unicodedata.normalize("NFKC", str(text or "")))
    compact = re.sub(r"\s+", "", normalized)
    if not HANGUL_RE.search(compact):
        return compact
    if ASCII_ALNUM_RE.search(compact):
        return compact
    return "".join(HANGUL_RE.findall(compact))


@lru_cache(maxsize=1)
def _load_lexicon() -> dict[str, dict[str, Any]]:
    with LEXICON_PATH.open("r", encoding="utf-8") as file:
        return {_normalize_sfx_text(key): value for key, value in json.load(file).items()}
