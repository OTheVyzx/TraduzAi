"""Authoritative one-request-per-owner translation boundary."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
import json
from typing import Any, Iterable, Literal, Mapping, Sequence

from .hash_contract import canonical_json_bytes, canonical_json_sha256, sha256_bytes, sha256_text
from .model import TRANSLATION_ROUTE_ACTIONS, OwnerGraph, OwnerViolation, TextOwner
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
    status: Literal["accepted", "rejected", "operational_error"]
    error_code: str | None
    language_verdict: TargetLanguageVerdict | None
    attempt_sha256: str

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
        status: Literal["accepted", "rejected", "operational_error"],
        language_verdict: TargetLanguageVerdict | None,
        error_code: str | None = None,
        attempt_index: int = 1,
    ) -> "TranslationAttempt":
        metadata_bytes = canonical_json_bytes(dict(provider_metadata))
        metadata_sha = sha256_bytes(metadata_bytes)
        response_sha = sha256_text(target_text) if target_text is not None else None
        if status == "accepted" and (
            response_sha is None
            or language_verdict is None
            or not language_verdict.accepted
            or not (provider_called or cache_hit)
        ):
            raise ValueError("accepted translation attempt lacks valid provider response")
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
            }
            and verdict.normalized_source_sha256 == verdict.normalized_target_sha256
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
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
        )
        if binding.target_locale != "pt-BR" or not binding.language_verdict.accepted:
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
            if not owner_attempts or owner_attempts[-1].status != "accepted":
                raise TranslationIdentityError("translation binding has no accepted terminal attempt")
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
        terminal.status != "accepted"
        or terminal.language_verdict is None
        or not terminal.language_verdict.accepted
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
) -> tuple[TranslationBinding, tuple[TranslationAttempt, ...]]:
    attempts: list[TranslationAttempt] = []
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
                verdict = validate_target_language(
                    source=request.source_text,
                    target=target_text,
                    role=request.semantic_role,
                    page_language_evidence=page_language_evidence,
                )
                status = "accepted" if verdict.accepted else "rejected"
                metadata = json.loads(
                    provider_result.provider_metadata_json_bytes.decode("utf-8")
                )
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
                continue
            attempts.append(attempt)
            if attempt.status == "accepted" and target_text is not None:
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
            variant = "primary" if backend_attempt == 0 else "contextual"
            attempt_index = len(attempts) + 1
            try:
                target_text = str(backend(request, variant) or "")
                verdict = validate_target_language(
                    source=request.source_text,
                    target=target_text,
                    role=request.semantic_role,
                    page_language_evidence=page_language_evidence,
                )
                status = "accepted" if verdict.accepted else "rejected"
                attempt = TranslationAttempt.build(
                    request=request,
                    backend=backend_name,
                    variant=variant,
                    provider_model=getattr(backend, "provider_model", None),
                    provider_metadata={"adapter": backend_name, "attempt_index": attempt_index},
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
                    provider_metadata={"adapter": backend_name, "attempt_index": attempt_index},
                    target_text=None,
                    provider_called=False,
                    cache_hit=False,
                    status="operational_error",
                    language_verdict=None,
                    error_code=exc.__class__.__name__,
                    attempt_index=attempt_index,
                )
                attempts.append(attempt)
                continue
            attempts.append(attempt)
            if attempt.status == "accepted":
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
) -> OwnerPageTranslationResult:
    attempts: list[TranslationAttempt] = []
    bindings: list[TranslationBinding] = []
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
            )
        except (TranslationValidationExhausted, TranslationInfrastructureError) as exc:
            error_type = type(exc)
            raise error_type((*attempts, *exc.attempts)) from exc
        attempts.extend(owner_attempts)
        bindings.append(binding)
    return OwnerPageTranslationResult.build(attempts, bindings)


def apply_owner_translation_result(
    graph: OwnerGraph,
    result: OwnerPageTranslationResult,
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
    if set(bindings_by_owner) != expected_ids:
        raise TranslationIdentityError("translation bindings do not cover every owner exactly once")
    for owner in owners:
        binding = bindings_by_owner[owner.owner_id]
        if tuple(owner.component_ids) != binding.component_ids:
            raise TranslationIdentityError("translation binding component identity mismatch")
        if sha256_text(str(owner.source_payload or "")) != binding.source_payload_sha256:
            raise TranslationIdentityError("translation binding source payload mismatch")
        if binding.target_locale != "pt-BR" or not binding.language_verdict.accepted:
            raise TranslationIdentityError("translation binding is not accepted PT-BR")
        owner.translated_payload = binding.target_text
        owner.state = "target_ready"
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
