"""Conservative text classifier for Smart Skip decisions.

The classifier is intentionally pure and cheap. Real skipping happens elsewhere;
this module only decides whether a recognized text is a safe candidate.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


CATEGORY_CREDIT_OR_WATERMARK = "credit_or_watermark"
CATEGORY_TIMER_OR_UI = "timer_or_ui"
CATEGORY_DECORATIVE_LOGO = "decorative_logo"
CATEGORY_SFX_KEEP_ORIGINAL = "sfx_keep_original"
CATEGORY_LOW_VALUE_NOISE = "low_value_noise"
CATEGORY_NOT_SAFE_TO_SKIP = "not_safe_to_skip"


_CREDIT_PATTERNS = (
    "all comics on this website",
    "just previews",
    "original version",
    "buy the comic",
    "faster update",
    "read on",
    "readon",
    "scanlation",
)

_TIMER_RE = re.compile(r"^[0oO]{1,2}\s*[:.]\s*[0oO]{1,2}\s*[:.]\s*[0-9oOsS]{1,2}$")
_CLOCK_RE = re.compile(r"^\d{1,2}\s*[:.]\s*\d{2}\s*[:.]\s*\d{2}$")


@dataclass(frozen=True)
class SmartSkipDecision:
    category: str
    safe_to_skip: bool
    confidence: float
    reason: str
    source_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def annotate_page_with_smart_skip_shadow(page: dict[str, Any]) -> dict[str, Any]:
    """Attach Smart Skip audit data to a page without changing behavior."""

    candidates: list[dict[str, Any]] = []
    not_safe_count = 0
    category_counts: dict[str, int] = {}
    page_number = _optional_int(page.get("numero"))

    for index, text_item in enumerate(list(page.get("texts") or [])):
        if not isinstance(text_item, dict):
            continue
        decision = classify_text_for_skip(
            str(
                text_item.get("original")
                or text_item.get("text")
                or text_item.get("source")
                or ""
            ),
            page_number=page_number,
            confidence=text_item.get("ocr_confidence", text_item.get("confidence")),
            bbox=text_item.get("balloon_bbox") or text_item.get("bbox"),
        )
        category_counts[decision.category] = category_counts.get(decision.category, 0) + 1
        if decision.safe_to_skip:
            candidate = decision.to_dict()
            candidate.update(
                {
                    "text_index": index,
                    "text_id": text_item.get("id"),
                    "bbox": text_item.get("balloon_bbox") or text_item.get("bbox"),
                    "page_number": page_number,
                }
            )
            candidates.append(candidate)
        else:
            not_safe_count += 1

    page["_smart_skip_shadow"] = {
        "candidate_count": len(candidates),
        "not_safe_count": not_safe_count,
        "category_counts": category_counts,
        "candidates": candidates,
    }
    return page


def classify_text_for_skip(
    text: str,
    *,
    page_number: int | None,
    confidence: float | None,
    bbox: tuple[int, int, int, int] | list[int] | None,
) -> SmartSkipDecision:
    normalized = _normalize_text(text)
    ocr_confidence = _coerce_confidence(confidence)
    on_opening_page = page_number in (None, 0, 1)

    if not normalized:
        return _decision(
            text,
            CATEGORY_LOW_VALUE_NOISE,
            False,
            ocr_confidence,
            "empty text is not enough to alter pipeline behavior",
        )

    if _looks_like_timer_or_ui(normalized):
        return _decision(
            text,
            CATEGORY_TIMER_OR_UI,
            True,
            ocr_confidence,
            "timer or UI-like text",
        )

    if _looks_like_credit_or_watermark(normalized) and _is_safe_boundary_credit(
        normalized,
        on_opening_page=on_opening_page,
        confidence=ocr_confidence,
    ):
        return _decision(
            text,
            CATEGORY_CREDIT_OR_WATERMARK,
            True,
            ocr_confidence,
            "opening-page credit or reader/update notice",
        )

    if _looks_like_decorative_logo(normalized, confidence=ocr_confidence, bbox=bbox):
        return _decision(
            text,
            CATEGORY_DECORATIVE_LOGO,
            False,
            ocr_confidence,
            "decorative-looking text needs visual confirmation before real skip",
        )

    return _decision(
        text,
        CATEGORY_NOT_SAFE_TO_SKIP,
        False,
        ocr_confidence,
        "text may be dialogue or narration",
    )


def _decision(
    text: str,
    category: str,
    safe_to_skip: bool,
    confidence: float,
    reason: str,
) -> SmartSkipDecision:
    return SmartSkipDecision(
        category=category,
        safe_to_skip=safe_to_skip,
        confidence=round(confidence, 4),
        reason=reason,
        source_text=str(text or ""),
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _coerce_confidence(confidence: float | None) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _looks_like_timer_or_ui(text: str) -> bool:
    compact = text.replace(" ", "")
    return bool(_CLOCK_RE.match(compact) or _TIMER_RE.match(compact))


def _looks_like_credit_or_watermark(text: str) -> bool:
    return any(pattern in text for pattern in _CREDIT_PATTERNS)


def _is_safe_boundary_credit(
    text: str,
    *,
    on_opening_page: bool,
    confidence: float,
) -> bool:
    if not on_opening_page:
        return False
    if "read on" in text or "readon" in text:
        return confidence <= 0.3
    return True


def _looks_like_decorative_logo(
    text: str,
    *,
    confidence: float,
    bbox: tuple[int, int, int, int] | list[int] | None,
) -> bool:
    if confidence > 0.25:
        return False
    words = re.findall(r"[a-z0-9]+", text)
    if len(words) < 4:
        return False
    if any(pattern in text for pattern in _CREDIT_PATTERNS):
        return False
    if bbox is None or len(bbox) != 4:
        return True
    try:
        x1, y1, x2, y2 = [int(value) for value in bbox]
    except (TypeError, ValueError):
        return True
    return (x2 - x1) > 120 and (y2 - y1) < 180
