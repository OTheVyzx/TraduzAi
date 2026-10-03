"""Authoritative one-request-per-owner translation boundary."""

from __future__ import annotations

import copy
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import json
import logging
import re
import time
from typing import Any, Iterable, Literal, Mapping, Sequence

from .hash_contract import canonical_json_bytes, canonical_json_sha256, sha256_bytes, sha256_text
from .model import (
    TRANSLATION_ROUTE_ACTIONS,
    ComponentDisposition,
    OwnerGraph,
    OwnerViolation,
    TextOwner,
)
try:
    from translator.language_policy import (
        PageLanguageEvidence,
        TargetLanguageVerdict,
        validate_target_language,
    )
except ImportError:  # pragma: no cover - package import
    from ..translator.language_policy import (
        PageLanguageEvidence,
        TargetLanguageVerdict,
        validate_target_language,
    )


_TRANSLATABLE_STATES = frozenset({"owned", "ocr_ready", "execution_planned"})
logger = logging.getLogger(__name__)
_NOOP_DISPLAY_MAX_TOKEN_LENGTH = 12


def _performance_measure(recorder: Any | None, stage: str):
    if recorder is None or not callable(getattr(recorder, "measure", None)):
        return nullcontext()
    return recorder.measure(stage)


def _persist_translation_attempt(
    recorder: Any | None,
    attempt: "TranslationAttempt",
    *,
    latency_sec: float,
) -> None:
    writer = getattr(recorder, "record_translation_attempt", None)
    if not callable(writer):
        return
    try:
        writer(attempt, latency_sec=latency_sec)
    except Exception as exc:
        logger.warning("failed to persist translation attempt: %s", exc)
_NOOP_DISPLAY_MIN_INK_HEIGHT_RATIO = 0.68
_NOOP_DISPLAY_MAX_COMPONENT_ASPECT = 1.8
_NOOP_EMPHATIC_DISPLAY_MAX_COMPONENT_ASPECT = 4.0


def _already_target_requires_source_replacement(
    source_text: str,
    verdict: TargetLanguageVerdict,
) -> bool:
    """Identify short mixed-overlay OCR without repainting coherent PT-BR pages."""

    source = str(source_text or "").strip()
    if not source or re.search(
        r"\b[a-z0-9][a-z0-9-]*\.(?:com|net|org|io|co)\b",
        source,
        flags=re.IGNORECASE,
    ):
        return False
    tokens = re.findall(r"[^\W_]+", source, flags=re.UNICODE)
    return bool(
        2 <= len(tokens) <= 6
        and verdict.ptbr_token_ratio_ppm <= 500_000
        and any(len(token) >= 11 for token in tokens)
    )
_NOOP_DISPLAY_TRAILING_MARKS = frozenset("!?….-~")


def _translation_owners(graph: OwnerGraph) -> list[TextOwner]:
    return sorted(
        (
            owner
            for owner in graph.owners
            if owner.disposition == "owned"
            and owner.state in _TRANSLATABLE_STATES
            and owner.route_action in TRANSLATION_ROUTE_ACTIONS
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


def _is_short_noop_display_sfx(
    graph: OwnerGraph,
    owner: TextOwner,
    translated_payload: str,
) -> bool:
    """Return whether a no-op result is large display SFX that must stay original.

    The geometry guard prevents unchanged dialogue or ordinary labels from being
    silently preserved when a translator fails to localize them.
    """

    source = " ".join(str(owner.source_payload or "").split())
    translated = " ".join(str(translated_payload or "").split())
    if not source or source.casefold() != translated.casefold():
        return False
    lexeme = source.rstrip("".join(sorted(_NOOP_DISPLAY_TRAILING_MARKS)))
    trailing_marks = source[len(lexeme) :]
    emphatic_display = bool(trailing_marks) and all(
        mark in _NOOP_DISPLAY_TRAILING_MARKS for mark in trailing_marks
    )
    if not lexeme.isalpha() or not (
        2 <= len(lexeme) <= _NOOP_DISPLAY_MAX_TOKEN_LENGTH
    ):
        return False

    selected_ids = set(owner.selected_observation_ids)
    selected_observations = [
        observation
        for observation in graph.observations
        if observation.observation_id in selected_ids
    ]
    if emphatic_display and any(
        observation.layout_bbox_page is not None
        for observation in selected_observations
    ):
        return False

    component_ids = set(owner.component_ids)
    component_boxes = [
        component.bbox_page
        for component in graph.components
        if component.component_id in component_ids
    ]
    if not component_boxes:
        return False
    component_width = max(box[2] for box in component_boxes) - min(
        box[0] for box in component_boxes
    )
    component_height = max(box[3] for box in component_boxes) - min(
        box[1] for box in component_boxes
    )
    if component_width <= 0 or component_height <= 0:
        return False
    max_component_aspect = (
        _NOOP_EMPHATIC_DISPLAY_MAX_COMPONENT_ASPECT
        if emphatic_display
        else _NOOP_DISPLAY_MAX_COMPONENT_ASPECT
    )
    if component_width / component_height > max_component_aspect:
        return False

    heights = sorted(
        observation.bbox_page[3] - observation.bbox_page[1]
        for observation in selected_observations
    )
    if not heights:
        return False
    middle = len(heights) // 2
    median_height = (
        heights[middle]
        if len(heights) % 2
        else (heights[middle - 1] + heights[middle]) / 2.0
    )
    return median_height / component_height >= _NOOP_DISPLAY_MIN_INK_HEIGHT_RATIO


def _preserve_short_noop_display_sfx(
    graph: OwnerGraph,
    owner_ids: set[str],
) -> None:
    if not owner_ids:
        return
    component_ids = {
        component_id
        for owner in graph.owners
        if owner.owner_id in owner_ids
        for component_id in owner.component_ids
    }
    graph.owners = [owner for owner in graph.owners if owner.owner_id not in owner_ids]
    graph.projections = [
        projection
        for projection in graph.projections
        if projection.owner_id not in owner_ids
    ]
    graph.component_dispositions = [
        ComponentDisposition(
            component_id=disposition.component_id,
            decision="preserve",
            owner_id=None,
            reason="policy:short_noop_display_sfx",
        )
        if disposition.component_id in component_ids
        else disposition
        for disposition in graph.component_dispositions
    ]


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

    preserved_owner_ids: set[str] = set()
    for owner in owners:
        record = record_by_owner[owner.owner_id]
        translated_payload = record["translated"].strip()
        if _is_short_noop_display_sfx(merged, owner, translated_payload):
            preserved_owner_ids.add(owner.owner_id)
            continue
        owner.translated_payload = translated_payload
        owner.state = "translated"
    _preserve_short_noop_display_sfx(merged, preserved_owner_ids)
    merged.require_valid()
    return merged


class TranslationIdentityError(ValueError):
    """Raised when owner, attempt, binding, or page identities diverge."""


class TranslationValidationExhausted(RuntimeError):
    def __init__(self, attempts=()):
        self.attempts = tuple(attempts)
        super().__init__("translation validation exhausted")


class TranslationInfrastructureError(RuntimeError):
    def __init__(self, attempts=()):
        self.attempts = tuple(attempts)
        super().__init__("all translation providers were unavailable")


@dataclass(frozen=True)
class OwnerTranslationRequest:
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    owner_id: str
    component_ids: tuple[str, ...]
    semantic_role: str
    route_action: str
    source_text: str
    source_payload_sha256: str
    request_sha256: str

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (
            self.run_id,
            self.origin_execution_id,
            self.page_id,
            self.page_source_sha256,
            self.owner_id,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "origin_execution_id": self.origin_execution_id,
            "page_id": self.page_id,
            "page_source_sha256": self.page_source_sha256,
            "owner_id": self.owner_id,
            "component_ids": list(self.component_ids),
            "semantic_role": self.semantic_role,
            "route_action": self.route_action,
            "source_text": self.source_text,
            "source_payload_sha256": self.source_payload_sha256,
        }

    @classmethod
    def from_graph(cls, graph: OwnerGraph, owner_id: str) -> "OwnerTranslationRequest":
        graph.require_valid()
        matches = [owner for owner in graph.owners if owner.owner_id == owner_id]
        if len(matches) != 1:
            raise TranslationIdentityError(f"owner request is not unique: {owner_id}")
        owner = matches[0]
        if owner not in _translation_owners(graph):
            raise TranslationIdentityError(f"owner is not translatable: {owner_id}")
        source_text = str(owner.source_payload or "")
        values = {
            "run_id": graph.run_id,
            "origin_execution_id": graph.origin_execution_id,
            "page_id": graph.page_id,
            "page_source_sha256": graph.page_source_sha256,
            "owner_id": owner.owner_id,
            "component_ids": tuple(owner.component_ids),
            "semantic_role": owner.semantic_role,
            "route_action": owner.route_action,
            "source_text": source_text,
            "source_payload_sha256": sha256_text(source_text),
        }
        payload = {
            **values,
            "component_ids": list(values["component_ids"]),
        }
        return cls(**values, request_sha256=canonical_json_sha256(payload))


@dataclass(frozen=True)
class TranslationAttempt:
    attempt_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    owner_id: str
    backend: str
    variant: str
    provider_model: str | None
    provider_metadata_json_bytes: bytes
    provider_metadata_sha256: str
    source_payload_sha256: str
    request_sha256: str
    response_sha256: str | None
    provider_called: bool
    cache_hit: bool
    status: Literal["accepted", "accepted_with_warnings", "rejected", "operational_error"]
    error_code: str | None
    language_verdict: TargetLanguageVerdict | None
    attempt_sha256: str

    @property
    def usable(self) -> bool:
        if self.status == "accepted":
            return bool(self.language_verdict and self.language_verdict.accepted)
        if self.status != "accepted_with_warnings" or self.language_verdict is None:
            return False
        from translator.delivery_policy import POLICY_ID
        metadata = json.loads(self.provider_metadata_json_bytes)
        return bool(self.response_sha256 and self.response_sha256 != sha256_text("")
                    and (self.provider_called or self.cache_hit)
                    and not self.language_verdict.accepted
                    and self.language_verdict.reason not in {"empty_target","placeholder_mismatch","entity_mismatch"}
                    and metadata.get("quality_usage_policy_id") == POLICY_ID
                    and metadata.get("quality_warning_reason") == self.language_verdict.reason
                    and metadata.get("quality_approved") is False
                    and metadata.get("validation_performed") is True)

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (
            self.run_id,
            self.origin_execution_id,
            self.page_id,
            self.page_source_sha256,
            self.owner_id,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "origin_execution_id": self.origin_execution_id,
            "page_id": self.page_id,
            "page_source_sha256": self.page_source_sha256,
            "owner_id": self.owner_id,
            "backend": self.backend,
            "variant": self.variant,
            "provider_model": self.provider_model,
            "provider_metadata": json.loads(self.provider_metadata_json_bytes.decode("utf-8")),
            "provider_metadata_sha256": self.provider_metadata_sha256,
            "source_payload_sha256": self.source_payload_sha256,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "provider_called": self.provider_called,
            "cache_hit": self.cache_hit,
            "status": self.status,
            "error_code": self.error_code,
            "language_verdict": (
                self.language_verdict.to_dict()
                if self.language_verdict is not None
                else None
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return self.canonical_payload() | {"attempt_sha256": self.attempt_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TranslationAttempt":
        metadata = payload.get("provider_metadata")
        if not isinstance(metadata, Mapping):
            raise TranslationIdentityError("translation attempt metadata is invalid")
        metadata_bytes = canonical_json_bytes(dict(metadata))
        metadata_sha = sha256_bytes(metadata_bytes)
        if metadata_sha != str(payload.get("provider_metadata_sha256") or ""):
            raise TranslationIdentityError("translation attempt metadata hash mismatch")
        verdict_payload = payload.get("language_verdict")
        verdict = (
            TargetLanguageVerdict.from_dict(dict(verdict_payload))
            if isinstance(verdict_payload, Mapping)
            else None
        )
        values = {
            "attempt_id": str(payload.get("attempt_id") or ""),
            "run_id": str(payload.get("run_id") or ""),
            "origin_execution_id": str(payload.get("origin_execution_id") or ""),
            "page_id": str(payload.get("page_id") or ""),
            "page_source_sha256": str(payload.get("page_source_sha256") or ""),
            "owner_id": str(payload.get("owner_id") or ""),
            "backend": str(payload.get("backend") or ""),
            "variant": str(payload.get("variant") or ""),
            "provider_model": (
                str(payload["provider_model"])
                if payload.get("provider_model") is not None
                else None
            ),
            "provider_metadata_json_bytes": metadata_bytes,
            "provider_metadata_sha256": metadata_sha,
            "source_payload_sha256": str(payload.get("source_payload_sha256") or ""),
            "request_sha256": str(payload.get("request_sha256") or ""),
            "response_sha256": (
                str(payload["response_sha256"])
                if payload.get("response_sha256") is not None
                else None
            ),
            "provider_called": bool(payload.get("provider_called")),
            "cache_hit": bool(payload.get("cache_hit")),
            "status": str(payload.get("status") or ""),
            "error_code": (
                str(payload["error_code"])
                if payload.get("error_code") is not None
                else None
            ),
            "language_verdict": verdict,
        }
        attempt = cls(
            **values,
            attempt_sha256=str(payload.get("attempt_sha256") or ""),
        )
        if canonical_json_sha256(attempt.canonical_payload()) != attempt.attempt_sha256:
            raise TranslationIdentityError("translation attempt hash mismatch")
        if attempt.status in {"accepted","accepted_with_warnings"} and not attempt.usable:
            raise TranslationIdentityError("translation attempt usage authorization is invalid")
        return attempt

    @classmethod
    def build(
        cls,
        *,
        request: OwnerTranslationRequest,
        backend: str,
        variant: str,
        provider_model: str | None,
        provider_metadata: Mapping[str, object],
        target_text: str | None,
        provider_called: bool,
        cache_hit: bool,
        status: Literal["accepted", "accepted_with_warnings", "rejected", "operational_error"],
        language_verdict: TargetLanguageVerdict | None,
        error_code: str | None = None,
        attempt_index: int = 1,
    ) -> "TranslationAttempt":
        recorded_metadata = dict(provider_metadata)
        if target_text is not None:
            recorded_metadata["target_produced"] = target_text
        metadata_bytes = canonical_json_bytes(recorded_metadata)
        metadata_sha = sha256_bytes(metadata_bytes)
        response_sha = sha256_text(target_text) if target_text is not None else None
        if status == "accepted" and (
            response_sha is None
            or language_verdict is None
            or not language_verdict.accepted
            or not (provider_called or cache_hit)
        ):
            raise ValueError("accepted translation attempt lacks valid provider response")
        if status == "accepted_with_warnings":
            from translator.delivery_policy import quality_warning_metadata
            required = quality_warning_metadata(target_text, language_verdict)
            if not (provider_called or cache_hit) or any(provider_metadata.get(k) != v for k,v in required.items()):
                raise ValueError("warning translation attempt lacks an authorized response")
        seed = {
            "request_sha256": request.request_sha256,
            "backend": str(backend),
            "variant": str(variant),
            "attempt_index": int(attempt_index),
            "provider_metadata_sha256": metadata_sha,
            "response_sha256": response_sha,
            "status": status,
        }
        attempt_id = f"translation-attempt:{canonical_json_sha256(seed)[:24]}"
        values = {
            "attempt_id": attempt_id,
            "run_id": request.run_id,
            "origin_execution_id": request.origin_execution_id,
            "page_id": request.page_id,
            "page_source_sha256": request.page_source_sha256,
            "owner_id": request.owner_id,
            "backend": str(backend),
            "variant": str(variant),
            "provider_model": provider_model,
            "provider_metadata_json_bytes": metadata_bytes,
            "provider_metadata_sha256": metadata_sha,
            "source_payload_sha256": request.source_payload_sha256,
            "request_sha256": request.request_sha256,
            "response_sha256": response_sha,
            "provider_called": bool(provider_called),
            "cache_hit": bool(cache_hit),
            "status": status,
            "error_code": error_code,
            "language_verdict": language_verdict,
        }
        provisional = cls(**values, attempt_sha256="")
        return cls(
            **values,
            attempt_sha256=canonical_json_sha256(provisional.canonical_payload()),
        )


@dataclass(frozen=True)
class TranslationBinding:
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    owner_id: str
    component_ids: tuple[str, ...]
    source_payload_sha256: str
    target_payload_sha256: str
    source_text: str
    target_text: str
    target_locale: Literal["pt-BR"]
    language_verdict: TargetLanguageVerdict
    attempt_ids: tuple[str, ...]
    translation_binding_sha256: str
    quality_usage_policy_id: str | None = None
    quality_warning_reason: str | None = None

    @property
    def usable(self) -> bool:
        if self.language_verdict.accepted:
            return self.quality_usage_policy_id is None and self.quality_warning_reason is None
        from translator.delivery_policy import POLICY_ID, usable_with_warning
        return bool(self.quality_usage_policy_id == POLICY_ID
                    and self.quality_warning_reason == self.language_verdict.reason
                    and usable_with_warning(self.target_text,self.language_verdict))

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (
            self.run_id,
            self.origin_execution_id,
            self.page_id,
            self.page_source_sha256,
            self.owner_id,
        )

    @property
    def preserves_original_pixels(self) -> bool:
        """Return whether complete language evidence authorizes a visual no-op."""

        verdict = self.language_verdict
        return bool(
            verdict.accepted
            and self.target_locale == "pt-BR"
            and verdict.policy_id
            in {
                "already_target_language",
                "source_neutral_nonlexical",
                "source_neutral_proper_name",
                "source_neutral_structured_identifiers",
            }
            and verdict.normalized_source_sha256 == verdict.normalized_target_sha256
        )

    def canonical_payload(self) -> dict[str, object]:
        payload = {
            "run_id": self.run_id,
            "origin_execution_id": self.origin_execution_id,
            "page_id": self.page_id,
            "page_source_sha256": self.page_source_sha256,
            "owner_id": self.owner_id,
            "component_ids": list(self.component_ids),
            "source_payload_sha256": self.source_payload_sha256,
            "target_payload_sha256": self.target_payload_sha256,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "target_locale": self.target_locale,
            "language_verdict": self.language_verdict.to_dict(),
            "attempt_ids": list(self.attempt_ids),
        }
        if self.quality_usage_policy_id is not None:
            payload["quality_usage_policy_id"] = self.quality_usage_policy_id
            payload["quality_warning_reason"] = self.quality_warning_reason
        return payload

    def to_dict(self) -> dict[str, object]:
        return self.canonical_payload() | {
            "translation_binding_sha256": self.translation_binding_sha256
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TranslationBinding":
        verdict_payload = payload.get("language_verdict")
        if not isinstance(verdict_payload, Mapping):
            raise TranslationIdentityError("translation binding verdict is missing")
        binding = cls(
            run_id=str(payload.get("run_id") or ""),
            origin_execution_id=str(payload.get("origin_execution_id") or ""),
            page_id=str(payload.get("page_id") or ""),
            page_source_sha256=str(payload.get("page_source_sha256") or ""),
            owner_id=str(payload.get("owner_id") or ""),
            component_ids=tuple(str(value) for value in payload.get("component_ids") or ()),
            source_payload_sha256=str(payload.get("source_payload_sha256") or ""),
            target_payload_sha256=str(payload.get("target_payload_sha256") or ""),
            source_text=str(payload.get("source_text") or ""),
            target_text=str(payload.get("target_text") or ""),
            target_locale=str(payload.get("target_locale") or ""),
            language_verdict=TargetLanguageVerdict.from_dict(dict(verdict_payload)),
            attempt_ids=tuple(str(value) for value in payload.get("attempt_ids") or ()),
            translation_binding_sha256=str(
                payload.get("translation_binding_sha256") or ""
            ),
            quality_usage_policy_id=payload.get("quality_usage_policy_id"),
            quality_warning_reason=payload.get("quality_warning_reason"),
        )
        if binding.target_locale != "pt-BR" or not binding.usable:
            raise TranslationIdentityError("translation binding target locale is invalid")
        if sha256_text(binding.source_text) != binding.source_payload_sha256:
            raise TranslationIdentityError("translation binding source hash mismatch")
        if sha256_text(binding.target_text) != binding.target_payload_sha256:
            raise TranslationIdentityError("translation binding target hash mismatch")
        if canonical_json_sha256(binding.canonical_payload()) != binding.translation_binding_sha256:
            raise TranslationIdentityError("translation binding hash mismatch")
        return binding


@dataclass(frozen=True)
class OwnerPageTranslationResult:
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    attempts: tuple[TranslationAttempt, ...]
    bindings: tuple[TranslationBinding, ...]
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(cls, attempts, bindings) -> "OwnerPageTranslationResult":
        attempts = tuple(attempts)
        bindings = tuple(bindings)
        identities = {
            item.identity[:4] for item in (*attempts, *bindings)
        }
        if len(identities) != 1:
            raise TranslationIdentityError("translation page result crossed execution identity")
        run_id, origin_execution_id, page_id, page_source_sha256 = next(iter(identities))
        attempts_by_id = {item.attempt_id: item for item in attempts}
        if len(attempts_by_id) != len(attempts):
            raise TranslationIdentityError("translation attempt identity duplicated")
        binding_owner_ids = [item.owner_id for item in bindings]
        if len(binding_owner_ids) != len(set(binding_owner_ids)):
            raise TranslationIdentityError("translation owner binding duplicated")
        for binding in bindings:
            owner_attempts = tuple(
                attempt for attempt in attempts if attempt.owner_id == binding.owner_id
            )
            if binding.attempt_ids != tuple(item.attempt_id for item in owner_attempts):
                raise TranslationIdentityError("translation binding attempt chain mismatch")
            if any(attempt.identity != binding.identity for attempt in owner_attempts):
                raise TranslationIdentityError("translation binding crossed owner identity")
            if not owner_attempts or not owner_attempts[-1].usable:
                raise TranslationIdentityError("translation binding has no accepted terminal attempt")
            terminal = owner_attempts[-1]
            if terminal.response_sha256 != binding.target_payload_sha256 or terminal.language_verdict != binding.language_verdict or not binding.usable:
                raise TranslationIdentityError("translation binding terminal response mismatch")
            if terminal.status == "accepted_with_warnings":
                metadata = json.loads(terminal.provider_metadata_json_bytes)
                if binding.quality_usage_policy_id != metadata.get("quality_usage_policy_id") or binding.quality_warning_reason != metadata.get("quality_warning_reason"):
                    raise TranslationIdentityError("translation binding warning policy mismatch")
        payload = {
            "run_id": run_id,
            "origin_execution_id": origin_execution_id,
            "page_id": page_id,
            "page_source_sha256": page_source_sha256,
            "attempts": [item.to_dict() for item in attempts],
            "bindings": [item.to_dict() for item in bindings],
        }
        encoded = canonical_json_bytes(payload)
        return cls(
            run_id=run_id,
            origin_execution_id=origin_execution_id,
            page_id=page_id,
            page_source_sha256=page_source_sha256,
            attempts=attempts,
            bindings=bindings,
            canonical_json_bytes=encoded,
            sha256=sha256_bytes(encoded),
        )

    @classmethod
    def from_canonical_json_bytes(cls, encoded: bytes) -> "OwnerPageTranslationResult":
        if not isinstance(encoded, bytes):
            raise TypeError("translation result canonical payload must be bytes")
        payload = json.loads(encoded.decode("utf-8"))
        result = cls.build(
            tuple(TranslationAttempt.from_dict(item) for item in payload.get("attempts") or ()),
            tuple(TranslationBinding.from_dict(item) for item in payload.get("bindings") or ()),
        )
        if result.canonical_json_bytes != encoded:
            raise TranslationIdentityError("translation result reopen changed canonical bytes")
        return result


def bind_translation(
    request: OwnerTranslationRequest,
    target_text: str,
    attempts: Sequence[TranslationAttempt],
) -> TranslationBinding:
    attempts = tuple(attempts)
    if not attempts or any(item.identity != request.identity for item in attempts):
        raise TranslationIdentityError("translation attempts do not belong to owner request")
    terminal = attempts[-1]
    if (
        not terminal.usable
        or terminal.language_verdict is None
        or terminal.response_sha256 != sha256_text(target_text)
    ):
        raise ValueError("translation binding requires an accepted terminal attempt")
    values = {
        "run_id": request.run_id,
        "origin_execution_id": request.origin_execution_id,
        "page_id": request.page_id,
        "page_source_sha256": request.page_source_sha256,
        "owner_id": request.owner_id,
        "component_ids": request.component_ids,
        "source_payload_sha256": request.source_payload_sha256,
        "target_payload_sha256": sha256_text(target_text),
        "source_text": request.source_text,
        "target_text": str(target_text),
        "target_locale": "pt-BR",
        "language_verdict": terminal.language_verdict,
        "attempt_ids": tuple(item.attempt_id for item in attempts),
    }
    if terminal.status == "accepted_with_warnings":
        from translator.delivery_policy import quality_warning_metadata
        warning = quality_warning_metadata(target_text,terminal.language_verdict)
        metadata = json.loads(terminal.provider_metadata_json_bytes)
        if any(metadata.get(k) != v for k,v in warning.items()):
            raise ValueError("translation warning delivery authorization mismatch")
        values.update(quality_usage_policy_id=warning["quality_usage_policy_id"],
                      quality_warning_reason=warning["quality_warning_reason"])
    elif not terminal.language_verdict.accepted:
        raise ValueError("translation binding requires accepted language or explicit warning policy")
    provisional = TranslationBinding(**values, translation_binding_sha256="")
    return TranslationBinding(
        **values,
        translation_binding_sha256=canonical_json_sha256(provisional.canonical_payload()),
    )


def translate_owner(
    request: OwnerTranslationRequest,
    *,
    backends: Sequence = (),
    max_attempts_per_backend: int = 2,
    attempt_fn=None,
    attempt_controls: Sequence = (),
    attempt_kwargs: Mapping[str, object] | None = None,
    page_language_evidence: PageLanguageEvidence | None = None,
    repaint_already_target_pixels: bool = False,
    performance_recorder: Any | None = None,
    explicit_entities: tuple[str, ...] = (),
    allow_quality_warnings: bool = False,
) -> tuple[TranslationBinding, tuple[TranslationAttempt, ...]]:
    attempts: list[TranslationAttempt] = []
    from translator.language_policy import _entity_pattern
    explicit_entities = tuple(entity for entity in explicit_entities if entity and _entity_pattern(entity).search(request.source_text))
    with _performance_measure(performance_recorder, "translation_validation"):
        source_verdict = validate_target_language(
            source=request.source_text,
            target=request.source_text,
            role=request.semantic_role,
            page_language_evidence=page_language_evidence,
            explicit_entities=explicit_entities,
        )
    if (
        repaint_already_target_pixels
        and source_verdict.accepted
        and source_verdict.policy_id == "already_target_language"
        and _already_target_requires_source_replacement(
            request.source_text,
            source_verdict,
        )
    ):
        verdict_payload = source_verdict.to_dict()
        verdict_payload.pop("verdict_sha256", None)
        verdict_payload.update(
            policy_id="already_target_language_repaint",
            reason="already_target_language_requires_source_replacement",
        )
        repaint_verdict = TargetLanguageVerdict.from_dict(verdict_payload)
        attempt = TranslationAttempt.build(
            request=request,
            backend="language_policy",
            variant=repaint_verdict.policy_id,
            provider_model=None,
            provider_metadata={"decision": "replace_source_pixels_with_verified_target"},
            target_text=request.source_text,
            provider_called=False,
            cache_hit=True,
            status="accepted",
            language_verdict=repaint_verdict,
            attempt_index=1,
        )
        _persist_translation_attempt(performance_recorder, attempt, latency_sec=0.0)
        return bind_translation(
            request,
            request.source_text,
            (attempt,),
        ), (attempt,)
    if source_verdict.accepted and source_verdict.policy_id in {
        "already_target_language",
        "source_neutral_nonlexical",
        "source_neutral_proper_name",
        "source_neutral_structured_identifiers",
    }:
        attempt = TranslationAttempt.build(
            request=request,
            backend="language_policy",
            variant=source_verdict.policy_id,
            provider_model=None,
            provider_metadata={"decision": "preserve_verified_source_pixels"},
            target_text=request.source_text,
            provider_called=False,
            cache_hit=True,
            status="accepted",
            language_verdict=source_verdict,
            attempt_index=1,
        )
        _persist_translation_attempt(performance_recorder, attempt, latency_sec=0.0)
        return bind_translation(
            request,
            request.source_text,
            (attempt,),
        ), (attempt,)
    if callable(attempt_fn):
        controls = tuple(attempt_controls)
        if not controls:
            try:
                from translator.translate import TranslationAttemptControl
            except ImportError:  # pragma: no cover - package import
                from ..translator.translate import TranslationAttemptControl
            controls = (
                TranslationAttemptControl("google", "primary", False),
                TranslationAttemptControl("google", "contextual", True),
                TranslationAttemptControl("ollama", "local", False),
                TranslationAttemptControl("ollama", "contextual", True),
            )
        provider_kwargs = dict(attempt_kwargs or {})
        for attempt_index, control in enumerate(controls, 1):
            attempt_started = time.perf_counter()
            provider_result = None
            target_text: str | None = None
            try:
                provider_result = attempt_fn(
                    {
                        "page_id": request.page_id,
                        "texts": [
                            {
                                "id": request.owner_id,
                                "owner_id": request.owner_id,
                                "text": request.source_text,
                                "original": request.source_text,
                                "semantic_role": request.semantic_role,
                                "tipo": request.semantic_role,
                                "route_action": request.route_action,
                                "component_ids": list(request.component_ids),
                            }
                        ],
                    },
                    **provider_kwargs,
                    control=control,
                )
                frozen_items = tuple(provider_result.translated_items)
                if len(frozen_items) != 1 or frozen_items[0].owner_id != request.owner_id:
                    raise TranslationIdentityError("provider response crossed owner identity")
                translated_page = frozen_items[0].read()
                records = [
                    item
                    for item in translated_page.get("texts") or []
                    if isinstance(item, dict)
                    and str(item.get("owner_id") or item.get("id") or "")
                    == request.owner_id
                ]
                if len(records) != 1:
                    raise TranslationIdentityError("provider response owner cardinality mismatch")
                target_text = str(records[0].get("translated") or "")
                with _performance_measure(performance_recorder, "translation_validation"):
                    verdict = validate_target_language(
                        source=request.source_text,
                        target=target_text,
                        role=request.semantic_role,
                        page_language_evidence=page_language_evidence,
                        explicit_entities=explicit_entities,
                    )
                status = "accepted" if verdict.accepted else "rejected"
                metadata = json.loads(
                    provider_result.provider_metadata_json_bytes.decode("utf-8")
                )
                from translator.delivery_policy import usable_with_warning, quality_warning_metadata
                if allow_quality_warnings and usable_with_warning(target_text,verdict):
                    status = "accepted_with_warnings"
                    metadata.update(quality_warning_metadata(target_text,verdict))
                attempt = TranslationAttempt.build(
                    request=request,
                    backend=provider_result.backend,
                    variant=provider_result.variant,
                    provider_model=provider_result.provider_model,
                    provider_metadata=metadata,
                    target_text=target_text,
                    provider_called=provider_result.provider_called,
                    cache_hit=provider_result.cache_hit,
                    status=status,
                    language_verdict=verdict,
                    attempt_index=attempt_index,
                )
            except Exception as exc:
                metadata_bytes = getattr(exc, "provider_metadata_json_bytes", None)
                try:
                    metadata = (
                        json.loads(metadata_bytes.decode("utf-8"))
                        if isinstance(metadata_bytes, bytes)
                        else {"error_code": exc.__class__.__name__}
                    )
                except (UnicodeDecodeError, json.JSONDecodeError):
                    metadata = {"error_code": exc.__class__.__name__}
                attempt = TranslationAttempt.build(
                    request=request,
                    backend=str(getattr(control, "backend", "unknown")),
                    variant=str(getattr(control, "variant", "unknown")),
                    provider_model=getattr(control, "provider_model", None),
                    provider_metadata=metadata,
                    target_text=None,
                    provider_called=bool(metadata.get("provider_called", False)),
                    cache_hit=bool(metadata.get("cache_hit", False)),
                    status="operational_error",
                    language_verdict=None,
                    error_code=exc.__class__.__name__,
                    attempt_index=attempt_index,
                )
                attempts.append(attempt)
                _persist_translation_attempt(
                    performance_recorder,
                    attempt,
                    latency_sec=time.perf_counter() - attempt_started,
                )
                continue
            attempts.append(attempt)
            _persist_translation_attempt(
                performance_recorder,
                attempt,
                latency_sec=time.perf_counter() - attempt_started,
            )
            if attempt.status in {"accepted","accepted_with_warnings"} and target_text is not None:
                return bind_translation(request, target_text, attempts), tuple(attempts)
        if any(item.status == "rejected" for item in attempts):
            raise TranslationValidationExhausted(attempts)
        raise TranslationInfrastructureError(attempts)

    for backend in backends:
        backend_name = str(
            getattr(backend, "backend_name", None)
            or getattr(backend, "__name__", None)
            or backend.__class__.__name__
        )
        for backend_attempt in range(max(1, int(max_attempts_per_backend))):
            attempt_started = time.perf_counter()
            variant = "primary" if backend_attempt == 0 else "contextual"
            attempt_index = len(attempts) + 1
            try:
                target_text = str(backend(request, variant) or "")
                with _performance_measure(performance_recorder, "translation_validation"):
                    verdict = validate_target_language(
                        source=request.source_text,
                        target=target_text,
                        role=request.semantic_role,
                        page_language_evidence=page_language_evidence,
                        explicit_entities=explicit_entities,
                    )
                status = "accepted" if verdict.accepted else "rejected"
                metadata = {"adapter":backend_name,"attempt_index":attempt_index}
                from translator.delivery_policy import usable_with_warning, quality_warning_metadata
                if allow_quality_warnings and usable_with_warning(target_text,verdict):
                    status = "accepted_with_warnings"
                    metadata.update(quality_warning_metadata(target_text,verdict))
                attempt = TranslationAttempt.build(
                    request=request,
                    backend=backend_name,
                    variant=variant,
                    provider_model=getattr(backend, "provider_model", None),
                    provider_metadata=metadata,
                    target_text=target_text,
                    provider_called=True,
                    cache_hit=False,
                    status=status,
                    language_verdict=verdict,
                    attempt_index=attempt_index,
                )
            except Exception as exc:
                attempt = TranslationAttempt.build(
                    request=request,
                    backend=backend_name,
                    variant=variant,
                    provider_model=getattr(backend, "provider_model", None),
                    provider_metadata={"adapter": backend_name, "attempt_index": attempt_index,
                                       "error_detail": str(exc), "adapter_called": True},
                    target_text=None,
                    provider_called=bool(getattr(exc, "provider_called", False)),
                    cache_hit=False,
                    status="operational_error",
                    language_verdict=None,
                    error_code=exc.__class__.__name__,
                    attempt_index=attempt_index,
                )
                attempts.append(attempt)
                _persist_translation_attempt(
                    performance_recorder,
                    attempt,
                    latency_sec=time.perf_counter() - attempt_started,
                )
                continue
            attempts.append(attempt)
            _persist_translation_attempt(
                performance_recorder,
                attempt,
                latency_sec=time.perf_counter() - attempt_started,
            )
            if attempt.status in {"accepted","accepted_with_warnings"}:
                return bind_translation(request, target_text, attempts), tuple(attempts)
    if any(item.status == "rejected" for item in attempts):
        raise TranslationValidationExhausted(attempts)
    raise TranslationInfrastructureError(attempts)


def translate_owner_page(
    requests: Sequence[OwnerTranslationRequest],
    *,
    backends: Sequence = (),
    attempt_fn=None,
    attempt_controls: Sequence = (),
    attempt_kwargs: Mapping[str, object] | None = None,
    page_language_evidence_by_owner: Mapping[str, PageLanguageEvidence] | None = None,
    repaint_already_target_pixels: bool = False,
    performance_recorder: Any | None = None,
    continue_on_owner_failure: bool = True,
    explicit_entities: tuple[str, ...] = (),
    allow_quality_warnings: bool = False,
) -> OwnerPageTranslationResult:
    attempts: list[TranslationAttempt] = []
    bindings: list[TranslationBinding] = []
    owner_errors: list[RuntimeError] = []
    for request in requests:
        try:
            binding, owner_attempts = translate_owner(
                request,
                backends=backends,
                attempt_fn=attempt_fn,
                attempt_controls=attempt_controls,
                attempt_kwargs=attempt_kwargs,
                page_language_evidence=(page_language_evidence_by_owner or {}).get(
                    request.owner_id
                ),
                repaint_already_target_pixels=repaint_already_target_pixels,
                performance_recorder=performance_recorder,
                explicit_entities=explicit_entities,
                allow_quality_warnings=allow_quality_warnings,
            )
        except (TranslationValidationExhausted, TranslationInfrastructureError) as exc:
            attempts.extend(exc.attempts)
            owner_errors.append(exc)
            if continue_on_owner_failure:
                continue
            error_type = type(exc)
            raise error_type(tuple(attempts)) from exc
        attempts.extend(owner_attempts)
        bindings.append(binding)
    if owner_errors and not bindings and all(
        isinstance(item, TranslationInfrastructureError) for item in owner_errors
    ):
        raise TranslationInfrastructureError(attempts) from owner_errors[-1]
    return OwnerPageTranslationResult.build(attempts, bindings)


def apply_owner_translation_result(
    graph: OwnerGraph,
    result: OwnerPageTranslationResult,
    *,
    review_unbound_attempts: bool = False,
) -> OwnerGraph:
    """Atomically attach accepted bindings to the same graph owners."""

    graph.require_valid()
    graph_identity = (
        graph.run_id,
        graph.origin_execution_id,
        graph.page_id,
        graph.page_source_sha256,
    )
    result_identity = (
        result.run_id,
        result.origin_execution_id,
        result.page_id,
        result.page_source_sha256,
    )
    if graph_identity != result_identity:
        raise TranslationIdentityError("translation result belongs to another graph")
    merged = copy.deepcopy(graph)
    owners = _translation_owners(merged)
    expected_ids = {owner.owner_id for owner in owners}
    bindings_by_owner = {binding.owner_id: binding for binding in result.bindings}
    if not set(bindings_by_owner).issubset(expected_ids):
        raise TranslationIdentityError("translation bindings include an unexpected owner")
    failed_owner_ids = expected_ids - set(bindings_by_owner)
    attempts_by_owner = {
        owner_id: tuple(
            attempt for attempt in result.attempts if attempt.owner_id == owner_id
        )
        for owner_id in failed_owner_ids
    }
    if review_unbound_attempts:
        if any(
            not owner_attempts
            or any(attempt.status in {"accepted","accepted_with_warnings"} for attempt in owner_attempts)
            for owner_attempts in attempts_by_owner.values()
        ):
            raise TranslationIdentityError(
                "review-only translation owner lacks a rejected attempt chain"
            )
    for owner in owners:
        binding = bindings_by_owner.get(owner.owner_id)
        if binding is None:
            continue
        if tuple(owner.component_ids) != binding.component_ids:
            raise TranslationIdentityError("translation binding component identity mismatch")
        if sha256_text(str(owner.source_payload or "")) != binding.source_payload_sha256:
            raise TranslationIdentityError("translation binding source payload mismatch")
        if binding.target_locale != "pt-BR" or not binding.usable:
            raise TranslationIdentityError("translation binding is not accepted PT-BR")
        owner.translated_payload = binding.target_text
        owner.state = "target_ready"
    if failed_owner_ids and review_unbound_attempts:
        failed_components_by_owner = {
            owner.owner_id: set(owner.component_ids)
            for owner in owners
            if owner.owner_id in failed_owner_ids
        }
        failed_component_ids = {
            component_id
            for component_ids in failed_components_by_owner.values()
            for component_id in component_ids
        }
        components_by_id = {
            component.component_id: component for component in merged.components
        }
        failed_owner_by_component = {
            component_id: owner_id
            for owner_id, component_ids in failed_components_by_owner.items()
            for component_id in component_ids
        }
        merged.component_dispositions = [
            (
                ComponentDisposition(
                    component_id=disposition.component_id,
                    decision="uncertain",
                    owner_id=None,
                    reason="owner_translation_rejected",
                    policy_id="coverage_ambiguous_candidate",
                    policy_bbox_page=components_by_id[
                        disposition.component_id
                    ].bbox_page,
                    policy_evidence_ids=tuple(
                        attempt.attempt_id
                        for attempt in attempts_by_owner[
                            failed_owner_by_component[disposition.component_id]
                        ]
                    ),
                    policy_reason=(
                        "translation validation exhausted; source pixels preserved "
                        "for review"
                    ),
                )
                if disposition.component_id in failed_component_ids
                else disposition
            )
            for disposition in merged.component_dispositions
        ]
        merged.projections = [
            projection
            for projection in merged.projections
            if projection.owner_id not in failed_owner_ids
        ]
        merged.owners = [
            owner for owner in merged.owners if owner.owner_id not in failed_owner_ids
        ]
    merged.require_valid(mode="enforce")
    return merged


__all__ = [
    "OwnerPageTranslationResult",
    "OwnerTranslationRequest",
    "TranslationAttempt",
    "TranslationBinding",
    "TranslationIdentityError",
    "TranslationInfrastructureError",
    "TranslationValidationExhausted",
    "apply_owner_translation_result",
    "bind_translation",
    "merge_owner_translations",
    "owners_to_translation_page",
    "translate_owner",
    "translate_owner_page",
]
