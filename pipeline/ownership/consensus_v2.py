"""Consensus authority that counts physical OCR invocations, not aliases."""

from __future__ import annotations

from collections.abc import Sequence
import unicodedata

from .model import TextObservation


def independent_origins(group: Sequence[TextObservation]) -> frozenset[str]:
    return frozenset(
        str(observation.invocation_id)
        for observation in group
        if str(observation.invocation_id or "").strip()
    )


def _payload_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())


def select_consensus_observation(
    observations: Sequence[TextObservation],
) -> TextObservation:
    eligible = tuple(
        observation
        for observation in observations
        if observation.rejection_reason is None and str(observation.text or "").strip()
    )
    if not eligible:
        raise ValueError("consensus requires at least one eligible text observation")

    groups: dict[str, list[TextObservation]] = {}
    for observation in eligible:
        groups.setdefault(_payload_key(observation.text), []).append(observation)

    def group_rank(item: tuple[str, list[TextObservation]]) -> tuple[int, float, int, str]:
        key, group = item
        return (
            len(independent_origins(group)),
            max(float(observation.confidence) for observation in group),
            len(key),
            key,
        )

    _, winner_group = max(groups.items(), key=group_rank)
    return max(
        winner_group,
        key=lambda observation: (
            float(observation.confidence),
            len(str(observation.text or "")),
            str(observation.observation_id),
        ),
    )
