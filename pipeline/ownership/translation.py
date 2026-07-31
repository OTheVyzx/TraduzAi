"""Authoritative one-request-per-owner translation boundary."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Iterable

from .model import OwnerGraph, OwnerViolation, TextOwner


_TRANSLATABLE_STATES = frozenset({"ocr_ready", "execution_planned"})


def _translation_owners(graph: OwnerGraph) -> list[TextOwner]:
    return sorted(
        (
            owner
            for owner in graph.owners
            if owner.disposition == "owned"
            and owner.state in _TRANSLATABLE_STATES
            and owner.route_action == "translate_inpaint_render"
        ),
        key=lambda owner: owner.owner_id,
    )


def owners_to_translation_page(
    graph: OwnerGraph,
    *,
    target_locale: str = "pt-BR",
) -> dict[str, Any]:
    """Serialize each validated semantic owner as exactly one translation record."""

    graph.require_valid()
    owners = _translation_owners(graph)
    records: list[dict[str, Any]] = []
    for owner in owners:
        source_payload = str(owner.source_payload or "")
        records.append(
            {
                "id": owner.owner_id,
                "owner_id": owner.owner_id,
                "page_id": owner.page_id,
                "text": source_payload,
                "original": source_payload,
                "semantic_role": owner.semantic_role,
                "tipo": owner.semantic_role,
                "route_action": owner.route_action,
                "component_ids": list(owner.component_ids),
                "observation_ids": list(owner.observation_ids),
                "selected_observation_ids": list(owner.selected_observation_ids),
                "target_locale": target_locale,
            }
        )
    owner_ids = [owner.owner_id for owner in owners]
    return {
        "page_id": graph.page_id,
        "texts": records,
        "_owner_translation_contract": {
            "schema_version": 1,
            "page_id": graph.page_id,
            "expected_owner_ids": owner_ids,
            "join_key": "owner_id",
            "target_locale": target_locale,
        },
    }


def _violation(
    code: str,
    message: str,
    offenders: Iterable[str],
) -> OwnerViolation:
    return OwnerViolation(
        code=code,
        severity="critical",
        message=message,
        offenders=tuple(sorted({str(value) for value in offenders if str(value)})),
    )


def _append_violations(graph: OwnerGraph, additions: Iterable[OwnerViolation]) -> None:
    unique = {
        (item.code, item.severity, item.message, tuple(item.offenders)): item
        for item in [*graph.violations, *additions]
    }
    graph.violations = sorted(
        unique.values(),
        key=lambda item: (item.code, item.offenders, item.message, item.severity),
    )


def _block_translation_owners(graph: OwnerGraph, owner_ids: set[str]) -> None:
    for owner in graph.owners:
        if owner.owner_id not in owner_ids:
            continue
        owner.translated_payload = None
        owner.state = "review_required"
        owner.route_action = "review_required"


def merge_owner_translations(
    graph: OwnerGraph,
    translated_page: dict[str, Any],
) -> OwnerGraph:
    """Join a translation response strictly by owner identity, atomically.

    The input graph is never mutated. Any response-set mismatch blocks every
    owner in the pending page batch so a partial translation cannot progress.
    """

    graph.require_valid()
    merged = copy.deepcopy(graph)
    owners = _translation_owners(merged)
    expected_ids = {owner.owner_id for owner in owners}

    raw_records = (
        translated_page.get("texts")
        if isinstance(translated_page, dict)
        else None
    )
    records = list(raw_records) if isinstance(raw_records, list) else []
    valid_records = [record for record in records if isinstance(record, dict)]
    response_ids = [str(record.get("owner_id") or "").strip() for record in valid_records]
    counts = Counter(response_ids)

    violations: list[OwnerViolation] = []
    invalid_record_count = len(records) - len(valid_records) + counts.get("", 0)
    if invalid_record_count:
        violations.append(
            _violation(
                "owner_translation_id_missing",
                "Every translation response record must contain an owner_id.",
                (f"invalid_record_count:{invalid_record_count}",),
            )
        )

    missing_ids = sorted(owner_id for owner_id in expected_ids if counts.get(owner_id, 0) == 0)
    duplicate_ids = sorted(owner_id for owner_id in expected_ids if counts.get(owner_id, 0) > 1)
    unknown_ids = sorted(
        owner_id for owner_id in counts if owner_id and owner_id not in expected_ids
    )
    if missing_ids:
        violations.append(
            _violation(
                "owner_translation_missing",
                "A translatable owner is missing from the translation response.",
                missing_ids,
            )
        )
    if duplicate_ids:
        violations.append(
            _violation(
                "owner_translation_duplicate",
                "A translation response contains the same owner more than once.",
                duplicate_ids,
            )
        )
    if unknown_ids:
        violations.append(
            _violation(
                "owner_translation_unknown",
                "A translation response references an owner outside the request.",
                unknown_ids,
            )
        )

    record_by_owner = {
        owner_id: record
        for owner_id, record in zip(response_ids, valid_records)
        if owner_id in expected_ids and counts.get(owner_id, 0) == 1
    }
    invalid_payload_ids = sorted(
        owner_id
        for owner_id, record in record_by_owner.items()
        if record.get("translated") is not None
        and not isinstance(record.get("translated"), str)
    )
    empty_payload_ids = sorted(
        owner_id
        for owner_id, record in record_by_owner.items()
        if record.get("translated") is None
        or (
            isinstance(record.get("translated"), str)
            and not record["translated"].strip()
        )
    )
    if invalid_payload_ids:
        violations.append(
            _violation(
                "owner_translation_payload_invalid",
                "A translated owner payload must be a non-empty string.",
                invalid_payload_ids,
            )
        )
    if empty_payload_ids:
        violations.append(
            _violation(
                "owner_translation_payload_missing",
                "A translation response contains an empty translated payload.",
                empty_payload_ids,
            )
        )

    locale_blocked_ids = sorted(
        owner_id
        for owner_id, record in record_by_owner.items()
        if (
            isinstance(record.get("locale_validation"), dict)
            and record["locale_validation"].get("status") == "blocked"
        )
        or "translation_locale_mismatch" in set(record.get("qa_flags") or [])
    )
    if locale_blocked_ids:
        violations.append(
            _violation(
                "owner_translation_locale_mismatch",
                "A deterministic target-locale mismatch blocks owner translation.",
                locale_blocked_ids,
            )
        )

    if violations:
        _append_violations(merged, violations)
        _block_translation_owners(merged, expected_ids)
        return merged

    for owner in owners:
        record = record_by_owner[owner.owner_id]
        owner.translated_payload = record["translated"].strip()
        owner.state = "translated"
    return merged


__all__ = ["merge_owner_translations", "owners_to_translation_page"]
