"""Placeholder-based term protection for translation."""

from __future__ import annotations

import re
from typing import Any

from context.entity_detector import detect_entities

PLACEHOLDER_TEMPLATE = "__TZN_NAME_{index}__"
PLACEHOLDER_RE = re.compile(r"__TZN_NAME_\d+__")
CORRUPTED_PLACEHOLDER_RE = re.compile(r"TZN[_ -]?NAME[_ -]?\d+", re.IGNORECASE)
NAME_LOCK_DENYLIST = {"ONE", "HOSPITAL", "READ", "THE", "I"}


def _denylist_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", " ", str(value or "")).strip().upper()


def _is_denylisted(entity: dict[str, Any]) -> bool:
    return _denylist_key(str(entity.get("source", ""))) in NAME_LOCK_DENYLIST


def _term_pattern(term: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in str(term or "").split() if part]
    body = r"\s+".join(parts) if parts else re.escape(str(term or ""))
    return re.compile(rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])", re.IGNORECASE)


def _placeholder_flag(issue: str, placeholder: str | None = None, **details: Any) -> dict[str, Any]:
    flag: dict[str, Any] = {"severity": "critical", "reason": "unrestored_placeholder", "issue": issue}
    if placeholder:
        flag["placeholder"] = placeholder
    flag.update(details)
    return flag


def _placeholder_count_flags(translated: str, terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_counts: dict[str, int] = {}
    for term in terms:
        placeholder = str(term.get("placeholder") or "")
        if placeholder:
            expected_counts[placeholder] = expected_counts.get(placeholder, 0) + 1

    observed_counts: dict[str, int] = {}
    for placeholder in PLACEHOLDER_RE.findall(translated or ""):
        observed_counts[placeholder] = observed_counts.get(placeholder, 0) + 1

    flags: list[dict[str, Any]] = []
    expected_total = sum(expected_counts.values())
    observed_total = sum(observed_counts.values())
    if expected_total != observed_total:
        flags.append(
            _placeholder_flag(
                "placeholder_count_mismatch",
                expected_count=expected_total,
                observed_count=observed_total,
            )
        )

    for placeholder, expected_count in expected_counts.items():
        observed_count = observed_counts.get(placeholder, 0)
        if observed_count != expected_count:
            flags.append(
                _placeholder_flag(
                    "placeholder_count_mismatch",
                    placeholder,
                    expected_count=expected_count,
                    observed_count=observed_count,
                )
            )

    return flags


def protect_terms(text: str, glossary_entries: list[dict[str, Any]]) -> dict[str, Any]:
    entities = [entity for entity in detect_entities(text, glossary_entries) if not _is_denylisted(entity)]
    protected = text
    replacements: list[dict[str, Any]] = []
    numbered_entities = list(enumerate(sorted(entities, key=lambda item: item["start"])))
    for index, entity in reversed(numbered_entities):
        placeholder = PLACEHOLDER_TEMPLATE.format(index=index)
        source = entity["source"]
        start = int(entity["start"])
        end = int(entity["end"])
        protected = f"{protected[:start]}{placeholder}{protected[end:]}"
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
    flags: list[dict[str, Any]] = validate_placeholders(translated, terms)
    for term in terms:
        placeholder = term["placeholder"]
        target = term["target"]
        if placeholder not in restored:
            continue
        restored = restored.replace(placeholder, target)

    for term in terms:
        for forbidden in term.get("forbidden", []) or []:
            if forbidden and forbidden.casefold() in restored.casefold():
                flags.append({"severity": "critical", "reason": "forbidden_translation", "source": term["source"]})

    if PLACEHOLDER_RE.search(restored):
        flags.append(_placeholder_flag("placeholder_leftover"))
    if CORRUPTED_PLACEHOLDER_RE.search(restored) and not PLACEHOLDER_RE.search(restored):
        flags.append(_placeholder_flag("placeholder_corrupted"))

    return {"text": restored, "flags": flags, "blocked": any(flag["severity"] in {"blocked", "critical"} for flag in flags)}


def validate_placeholders(translated: str, terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = _placeholder_count_flags(translated, terms)
    for term in terms:
        if term["placeholder"] not in translated:
            flags.append(_placeholder_flag("placeholder_missing", term["placeholder"]))
    if CORRUPTED_PLACEHOLDER_RE.search(translated) and not PLACEHOLDER_RE.search(translated):
        flags.append(_placeholder_flag("placeholder_corrupted"))
    return flags
