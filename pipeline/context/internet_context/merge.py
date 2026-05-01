from __future__ import annotations

from .models import ContextCandidate, SourceResult
from .normalizer import candidate_key


def merge_candidates(source_results: list[SourceResult], reviewed_glossary: dict[str, str]) -> list[ContextCandidate]:
    merged: dict[str, ContextCandidate] = {}
    reviewed = {candidate_key(source): target for source, target in reviewed_glossary.items()}

    for result in source_results:
        if result.status != "found":
            continue
        for candidate in result.candidates:
            key = candidate_key(candidate.source)
            if not key:
                continue
            if key in reviewed:
                merged[key] = ContextCandidate(
                    kind=candidate.kind,
                    source=candidate.source,
                    target=reviewed[key],
                    confidence=1.0,
                    sources=sorted(set([*candidate.sources, result.source])),
                    status="reviewed",
                    protect=True,
                    aliases=candidate.aliases,
                    forbidden=candidate.forbidden,
                    notes="Mantido do glossario revisado pelo usuario.",
                )
                continue

            existing = merged.get(key)
            sources = sorted(set([*candidate.sources, result.source]))
            if existing is None:
                merged[key] = ContextCandidate(
                    **{**candidate.to_dict(), "sources": sources, "status": "candidate"}
                )
                continue
            existing.sources = sorted(set([*existing.sources, *sources]))
            existing.confidence = max(existing.confidence, candidate.confidence)
            if candidate.confidence > existing.confidence:
                existing.target = candidate.target

    return sorted(merged.values(), key=lambda item: (-item.confidence, item.source.lower()))


def best_title(source_results: list[SourceResult], fallback: str) -> str:
    found = [item for item in source_results if item.status == "found" and item.title]
    if not found:
        return fallback
    return max(found, key=lambda item: item.confidence).title


def best_synopsis(source_results: list[SourceResult]) -> str:
    found = [item for item in source_results if item.status == "found" and item.synopsis]
    if not found:
        return ""
    return max(found, key=lambda item: (item.confidence, len(item.synopsis))).synopsis


def merged_genres(source_results: list[SourceResult]) -> list[str]:
    values: list[str] = []
    seen = set()
    for result in source_results:
        for genre in result.genres:
            key = candidate_key(genre)
            if key and key not in seen:
                seen.add(key)
                values.append(genre)
    return values
