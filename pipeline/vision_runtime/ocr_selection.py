"""Consensus selection for hash-bound physical OCR alternatives."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Literal


SelectionStatus = Literal["selected", "selected_with_uncertainty", "review_required"]


def _signature(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).upper()
    return re.sub(r"[^0-9A-Z]+", "", normalized)


@dataclass(frozen=True)
class OCRAlternative:
    attempt_id: str
    variant_id: str
    observation_ids: tuple[str, ...]
    text: str
    mean_confidence: float
    comparison_signature: str


@dataclass(frozen=True)
class OCRSelection:
    status: SelectionStatus
    selected_observation_ids: tuple[str, ...]
    selected_source: str | None
    provenance: str | None
    alternatives: tuple[OCRAlternative, ...]
    uncertainty_reasons: tuple[str, ...]


def _alternative(attempt: Any, observations: tuple[Any, ...]) -> OCRAlternative:
    ordered = tuple(sorted(observations, key=lambda item: (
        int(item.bbox_page[1]), int(item.bbox_page[0]),
        int(item.bbox_page[3]), int(item.bbox_page[2]), str(item.observation_id),
    )))
    text = " ".join(str(item.text).strip() for item in ordered if str(item.text).strip())
    confidence = (
        sum(float(item.confidence) for item in ordered) / len(ordered)
        if ordered else 0.0
    )
    return OCRAlternative(
        attempt_id=str(attempt.attempt_id),
        variant_id=str(attempt.variant_id),
        observation_ids=tuple(str(item.observation_id) for item in ordered),
        text=text,
        mean_confidence=confidence,
        comparison_signature=_signature(text),
    )


def select_ocr_invocation(invocation: Any) -> OCRSelection:
    """Choose corroborated text once while retaining every physical alternative."""

    by_attempt: dict[str, list[Any]] = {}
    for observation in tuple(invocation.observations):
        by_attempt.setdefault(str(observation.attempt_id), []).append(observation)
    alternatives = tuple(
        _alternative(attempt, tuple(by_attempt.get(str(attempt.attempt_id), ())))
        for attempt in tuple(invocation.attempts)
    )
    nonempty = tuple(item for item in alternatives if item.comparison_signature)
    if not nonempty:
        return OCRSelection(
            "review_required", (), None, None, alternatives,
            ("illegible_or_unrecognized",),
        )
    if len(nonempty) == 1:
        chosen = nonempty[0]
        return OCRSelection(
            "selected_with_uncertainty", chosen.observation_ids, chosen.text,
            "single_variant_v1", alternatives, ("single_variant_support",),
        )
    counts = Counter(item.comparison_signature for item in nonempty)
    best_count = max(counts.values())
    winners = tuple(key for key, count in counts.items() if count == best_count)
    if best_count < 2 or len(winners) != 1:
        return OCRSelection(
            "review_required", (), None, None, alternatives,
            ("conflicting_readings",),
        )
    corroborated = tuple(item for item in nonempty if item.comparison_signature == winners[0])
    chosen = sorted(corroborated, key=lambda item: (
        -item.mean_confidence, -len(item.text), item.variant_id, item.attempt_id,
    ))[0]
    return OCRSelection(
        "selected", chosen.observation_ids, chosen.text,
        "variant_consensus_v1", alternatives, (),
    )
