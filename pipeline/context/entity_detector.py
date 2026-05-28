"""Entity detection backed by glossary entries."""

from __future__ import annotations

import re
from typing import Any


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _entry_terms(entry: dict[str, Any]) -> list[str]:
    terms = [str(entry.get("source", ""))]
    terms.extend(str(alias) for alias in entry.get("aliases", []) or [])
    return [term for term in terms if term.strip()]


def _term_pattern(term: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in str(term or "").split() if part]
    body = r"\s+".join(parts) if parts else re.escape(str(term or ""))
    return re.compile(rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])", re.IGNORECASE)


def detect_entities(text: str, glossary_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for entry in glossary_entries:
        for term in _entry_terms(entry):
            term_norm = _norm(term)
            for match in _term_pattern(term).finditer(text or ""):
                matches.append(
                    {
                        "entry_id": entry.get("id", term_norm),
                        "source": term,
                        "target": entry.get("target", term),
                        "type": entry.get("type", entry.get("entry_type", "generic_term")),
                        "protect": bool(entry.get("protect", False)),
                        "start": match.start(),
                        "end": match.end(),
                        "mode": "preserve" if entry.get("protect") else "translate_fixed",
                        "forbidden": list(entry.get("forbidden", []) or []),
                    }
                )

    filtered: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for match in sorted(matches, key=lambda item: (item["start"], -(item["end"] - item["start"]))):
        span = (int(match["start"]), int(match["end"]))
        if any(span[0] < end and start < span[1] for start, end in occupied):
            continue
        filtered.append(match)
        occupied.append(span)
    return filtered

