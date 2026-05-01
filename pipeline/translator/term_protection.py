"""Placeholder-based term protection for translation."""

from __future__ import annotations

import re
from typing import Any

from context.entity_detector import detect_entities

PLACEHOLDER_TEMPLATE = "⟦TA_TERM_{index:03}⟧"


def protect_terms(text: str, glossary_entries: list[dict[str, Any]]) -> dict[str, Any]:
    entities = detect_entities(text, glossary_entries)
    protected = text
    replacements: list[dict[str, Any]] = []
    for index, entity in enumerate(sorted(entities, key=lambda item: item["start"], reverse=True), 1):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=index)
        source = entity["source"]
        pattern = re.compile(re.escape(source), re.IGNORECASE)
        protected, count = pattern.subn(placeholder, protected, count=1)
        if count:
            replacements.append(
                {
                    "placeholder": placeholder,
                    "source": source,
                    "target": entity["target"],
                    "mode": entity["mode"],
                    "protect": entity["protect"],
                    "forbidden": entity["forbidden"],
                }
            )
    replacements.sort(key=lambda item: item["placeholder"])
    return {"protected_source": protected, "terms": replacements}


def restore_terms(translated: str, terms: list[dict[str, Any]]) -> dict[str, Any]:
    restored = translated
    flags: list[dict[str, Any]] = []
    for term in terms:
        placeholder = term["placeholder"]
        target = term["target"]
        if placeholder not in restored:
            flags.append({"severity": "blocked", "reason": "placeholder_missing", "placeholder": placeholder})
            continue
        restored = restored.replace(placeholder, target)

    for term in terms:
        for forbidden in term.get("forbidden", []) or []:
            if forbidden and forbidden.casefold() in restored.casefold():
                flags.append({"severity": "critical", "reason": "forbidden_translation", "source": term["source"]})

    if "⟦TA_TERM_" in restored:
        flags.append({"severity": "blocked", "reason": "placeholder_leftover"})

    return {"text": restored, "flags": flags, "blocked": any(flag["severity"] == "blocked" for flag in flags)}


def validate_placeholders(translated: str, terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    for term in terms:
        if term["placeholder"] not in translated:
            flags.append({"severity": "blocked", "reason": "placeholder_missing", "placeholder": term["placeholder"]})
    if re.search(r"TA[_ -]?TERM[_ -]?\d+", translated) and "⟦TA_TERM_" not in translated:
        flags.append({"severity": "blocked", "reason": "placeholder_corrupted"})
    return flags

