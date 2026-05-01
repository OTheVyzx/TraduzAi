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


def detect_entities(text: str, glossary_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    lowered = _norm(text)
    for entry in glossary_entries:
        for term in _entry_terms(entry):
            term_norm = _norm(term)
            start = lowered.find(term_norm)
            if start < 0:
                continue
            matches.append(
                {
                    "entry_id": entry.get("id", term_norm),
                    "source": term,
                    "target": entry.get("target", term),
                    "type": entry.get("type", entry.get("entry_type", "generic_term")),
                    "protect": bool(entry.get("protect", False)),
                    "start": start,
                    "end": start + len(term_norm),
                    "mode": "preserve" if entry.get("protect") else "translate_fixed",
                    "forbidden": list(entry.get("forbidden", []) or []),
                }
            )
            break
    return sorted(matches, key=lambda item: (item["start"], -(item["end"] - item["start"])))

