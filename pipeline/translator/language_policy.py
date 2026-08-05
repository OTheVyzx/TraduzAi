"""Deterministic target-language validation for owner-bound translations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable, Literal
import unicodedata

try:
    from ownership.hash_contract import canonical_json_bytes, sha256_bytes, sha256_text
except ImportError:  # pragma: no cover - package import
    from ..ownership.hash_contract import canonical_json_bytes, sha256_bytes, sha256_text

from .locale_policy import validate_target_locale


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-z]+)?", re.UNICODE)
_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}")
_ENGLISH_WORDS = frozenset(
    """
    a an and are arena as at away be begin behind but can could did do does down
    event every everything for from go has have he her here him his honestly how
    i if in into is it its just kills let lets me mode my no not of on one only
    or our player players right she should something survival that the their them
    there they this to up wait we what when where which who why will with would
    you your
    """.split()
) | frozenset(
    """
    achievement arts been characteristics different dodge estimated first guild
    heavenly lose match martial probably rank selective slaughter star strike
    sword thoughts though unlocked vip while
    """.split()
)
_PTBR_WORDS = frozenset(
    """
    a agora ainda antes assim ao arena as até bem com como da das de dele dela depois
    do dos e ela ele em entrar entrarem entre era esse esta está eu foi há isso
    jogador jogadores já logo mais mas me meu minha não nos nós o os ou para pela
    pelo por porque possa que quem se sem ser seu sua tem todos um uma vamos você
    vocês abate abates
    """.split()
) | frozenset(
    """
    artes certo caracteristicas guilda idiota limpa limpo marciais mestre mestra
    nacional seletiva seletivo tao voce voces
    """.split()
)
_PTBR_MORPHOLOGY_SUFFIXES = (
    "ado",
    "ada",
    "ados",
    "adas",
    "ido",
    "ida",
    "idos",
    "idas",
    "ando",
    "endo",
    "indo",
    "mente",
    "dade",
    "dades",
)
_PTBR_JOIN_MARKERS = tuple(
    sorted(
        (
            token
            for token in _PTBR_WORDS
            if len(token) >= 5 and token not in _ENGLISH_WORDS
        ),
        key=lambda token: (-len(token), token),
    )
)


def _is_ptbr_token(token: str) -> bool:
    normalized = str(token or "").casefold()
    if normalized in _PTBR_WORDS:
        return True
    if len(normalized) >= 6 and normalized.endswith(_PTBR_MORPHOLOGY_SUFFIXES):
        return True
    compact = re.sub(r"[^a-z]", "", normalized)
    return len(compact) >= 8 and any(marker in compact for marker in _PTBR_JOIN_MARKERS)


def _is_ptbr_specific_token(token: str) -> bool:
    normalized = str(token or "").casefold()
    return _is_ptbr_token(normalized) and normalized not in _ENGLISH_WORDS


_ENGLISH_JOIN_MARKERS = tuple(
    sorted(
        (
            token
            for token in _ENGLISH_WORDS
            if len(token) >= 4 and token not in _PTBR_WORDS
        ),
        key=lambda token: (-len(token), token),
    )
)


def _english_only_markers(token: str) -> tuple[str, ...]:
    normalized = str(token or "").casefold()
    if normalized in _ENGLISH_WORDS and normalized not in _PTBR_WORDS:
        return (normalized,)
    compact = re.sub(r"[^a-z]", "", normalized)
    if len(compact) < 8:
        return ()
    return tuple(marker for marker in _ENGLISH_JOIN_MARKERS if marker in compact)


def _looks_like_neutral_proper_name(value: str, tokens: tuple[str, ...]) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", str(value or ""))
    if not (1 <= len(words) <= 4) or len(words) != len(tokens):
        return False
    if any(_english_only_markers(token) or _is_ptbr_token(token) for token in tokens):
        return False
    return all(word[:1].isupper() for word in words)


@dataclass(frozen=True)
class PageLanguageEvidence:
    coverage_complete: bool
    source_only_tokens: tuple[str, ...]

    @classmethod
    def build(
        cls,
        *,
        coverage_complete: bool,
        source_only_tokens: tuple[str, ...],
    ) -> "PageLanguageEvidence":
        return cls(
            bool(coverage_complete),
            tuple(sorted({str(token).casefold() for token in source_only_tokens if str(token)})),
        )


def build_page_language_evidence(
    *,
    texts: tuple[str, ...] | list[str],
    coverage_complete: bool,
) -> PageLanguageEvidence:
    """Summarize fresh regional OCR without treating shared PT/EN words as source-only."""

    tokens = {
        token
        for text in texts
        for token in _tokens(str(text or ""))
    }
    english_only = tuple(
        sorted(
            {
                marker
                for token in tokens
                for marker in _english_only_markers(token)
            }
        )
    )
    return PageLanguageEvidence.build(
        coverage_complete=coverage_complete,
        source_only_tokens=english_only,
    )


@dataclass(frozen=True)
class TargetLanguageVerdict:
    accepted: bool
    retryable: bool
    reason: str
    policy_id: str
    target_locale: Literal["pt-BR"]
    normalized_source_sha256: str
    normalized_target_sha256: str
    source_only_tokens: tuple[str, ...]
    english_token_ratio_ppm: int
    ptbr_token_ratio_ppm: int
    entities_equivalent: bool
    numbers_equivalent: bool
    placeholders_equivalent: bool
    canonical_json_bytes: bytes
    verdict_sha256: str

    def to_dict(self) -> dict[str, object]:
        payload = self.canonical_payload()
        payload["verdict_sha256"] = self.verdict_sha256
        return payload

    def canonical_payload(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "retryable": self.retryable,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "target_locale": self.target_locale,
            "normalized_source_sha256": self.normalized_source_sha256,
            "normalized_target_sha256": self.normalized_target_sha256,
            "source_only_tokens": list(self.source_only_tokens),
            "english_token_ratio_ppm": self.english_token_ratio_ppm,
            "ptbr_token_ratio_ppm": self.ptbr_token_ratio_ppm,
            "entities_equivalent": self.entities_equivalent,
            "numbers_equivalent": self.numbers_equivalent,
            "placeholders_equivalent": self.placeholders_equivalent,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TargetLanguageVerdict":
        verdict = _build_verdict(
            accepted=bool(payload.get("accepted")),
            retryable=bool(payload.get("retryable")),
            reason=str(payload.get("reason") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            normalized_source_sha256=str(payload.get("normalized_source_sha256") or ""),
            normalized_target_sha256=str(payload.get("normalized_target_sha256") or ""),
            source_only_tokens=tuple(
                str(value) for value in payload.get("source_only_tokens") or ()
            ),
            english_token_ratio_ppm=int(payload.get("english_token_ratio_ppm") or 0),
            ptbr_token_ratio_ppm=int(payload.get("ptbr_token_ratio_ppm") or 0),
            entities_equivalent=bool(payload.get("entities_equivalent")),
            numbers_equivalent=bool(payload.get("numbers_equivalent")),
            placeholders_equivalent=bool(payload.get("placeholders_equivalent")),
        )
        supplied_sha = payload.get("verdict_sha256")
        if supplied_sha is not None and str(supplied_sha) != verdict.verdict_sha256:
            raise ValueError("target-language verdict hash mismatch")
        return verdict


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(value))


def _ratio_ppm(count: int, total: int) -> int:
    return (int(count) * 1_000_000) // max(1, int(total))


def _entities_equivalent(
    source: str,
    target: str,
    explicit_entities: tuple[str, ...],
) -> bool:
    source_folded = _normalize(source).casefold()
    target_folded = _normalize(target).casefold()
    return all(
        _normalize(entity).casefold() in source_folded
        and _normalize(entity).casefold() in target_folded
        for entity in explicit_entities
        if _normalize(entity)
    )


def _numbers_equivalent(source: str, target: str) -> bool:
    validation = validate_target_locale(
        source_text=source,
        target_text=target,
        target_locale="pt-BR",
    )
    return validation.numbers_equivalent_for_binding


def _build_verdict(
    *,
    accepted: bool,
    retryable: bool,
    reason: str,
    policy_id: str,
    normalized_source_sha256: str,
    normalized_target_sha256: str,
    source_only_tokens: tuple[str, ...],
    english_token_ratio_ppm: int,
    ptbr_token_ratio_ppm: int,
    entities_equivalent: bool,
    numbers_equivalent: bool,
    placeholders_equivalent: bool,
) -> TargetLanguageVerdict:
    if accepted and not (
        entities_equivalent and numbers_equivalent and placeholders_equivalent
    ):
        raise ValueError("accepted target-language verdict failed mandatory equivalence")
    fields = {
        "accepted": bool(accepted),
        "retryable": bool(retryable),
        "reason": str(reason),
        "policy_id": str(policy_id),
        "target_locale": "pt-BR",
        "normalized_source_sha256": str(normalized_source_sha256),
        "normalized_target_sha256": str(normalized_target_sha256),
        "source_only_tokens": list(tuple(source_only_tokens)),
        "english_token_ratio_ppm": int(english_token_ratio_ppm),
        "ptbr_token_ratio_ppm": int(ptbr_token_ratio_ppm),
        "entities_equivalent": bool(entities_equivalent),
        "numbers_equivalent": bool(numbers_equivalent),
        "placeholders_equivalent": bool(placeholders_equivalent),
    }
    encoded = canonical_json_bytes(fields)
    return TargetLanguageVerdict(
        **{**fields, "source_only_tokens": tuple(source_only_tokens)},
        canonical_json_bytes=encoded,
        verdict_sha256=sha256_bytes(encoded),
    )


def validate_target_language(
    *,
    source: str,
    target: str,
    role: str,
    explicit_entities: tuple[str, ...] = (),
    page_language_evidence: PageLanguageEvidence | None = None,
) -> TargetLanguageVerdict:
    normalized_source = _normalize(source)
    normalized_target = _normalize(target)
    source_tokens = _tokens(normalized_source)
    target_tokens = _tokens(normalized_target)
    explicit_entity_tokens = {
        token
        for entity in explicit_entities
        for token in _tokens(entity)
    }
    considered_tokens = tuple(
        token for token in target_tokens if token not in explicit_entity_tokens
    )
    english_tokens = tuple(
        marker
        for token in considered_tokens
        for marker in _english_only_markers(token)
    )
    ptbr_tokens = tuple(
        token for token in considered_tokens if _is_ptbr_token(token)
    )
    english_only_tokens = tuple(
        marker
        for token in considered_tokens
        for marker in _english_only_markers(token)
    )
    ptbr_only_tokens = tuple(
        token
        for token in considered_tokens
        if _is_ptbr_specific_token(token) and token not in _ENGLISH_WORDS
    )
    source_english_markers = {
        marker
        for token in source_tokens
        for marker in _english_only_markers(token)
    }
    target_english_markers = {
        marker
        for token in considered_tokens
        for marker in _english_only_markers(token)
    }
    source_only_tokens = tuple(
        sorted(source_english_markers & target_english_markers)
    )
    english_ratio = _ratio_ppm(len(english_tokens), len(considered_tokens))
    ptbr_ratio = _ratio_ppm(len(ptbr_tokens), len(considered_tokens))
    source_only_ratio = _ratio_ppm(
        sum(
            1
            for token in considered_tokens
            if set(_english_only_markers(token)) & set(source_only_tokens)
        ),
        len(considered_tokens),
    )
    entities_ok = _entities_equivalent(source, target, explicit_entities)
    numbers_ok = _numbers_equivalent(source, target)
    placeholders_ok = Counter(_PLACEHOLDER_RE.findall(source)) == Counter(
        _PLACEHOLDER_RE.findall(target)
    )

    accepted = False
    retryable = True
    reason = "valid_pt_br"
    policy_id = "validated_pt_br"
    if not normalized_target:
        reason = "empty_target"
    elif not placeholders_ok:
        reason = "placeholder_mismatch"
    elif not numbers_ok:
        reason = "numeric_mismatch"
    elif not entities_ok:
        reason = "entity_mismatch"
    elif normalized_source.casefold() == normalized_target.casefold():
        evidence = page_language_evidence
        already_target = bool(
            evidence is not None
            and evidence.coverage_complete
            and not evidence.source_only_tokens
            and len(ptbr_only_tokens) > 0
            and len(english_only_tokens) == 0
        )
        if already_target:
            accepted = True
            retryable = False
            reason = "already_target_language"
            policy_id = "already_target_language"
        elif bool(
            evidence is not None
            and evidence.coverage_complete
            and not evidence.source_only_tokens
            and not target_tokens
        ):
            accepted = True
            retryable = False
            reason = "source_neutral_nonlexical"
            policy_id = "source_neutral_nonlexical"
        elif bool(
            evidence is not None
            and evidence.coverage_complete
            and not evidence.source_only_tokens
            and _looks_like_neutral_proper_name(normalized_target, target_tokens)
        ):
            accepted = True
            retryable = False
            reason = "source_neutral_proper_name"
            policy_id = "source_neutral_proper_name"
        else:
            normalized_role = str(role or "dialogue").casefold()
            reason = (
                "unchanged_source_dialogue"
                if "dialogue" in normalized_role
                else "unchanged_source_text"
            )
    elif (
        len(source_only_tokens) >= 2
        and (english_ratio >= 400_000 or source_only_ratio >= 350_000)
    ) or (
        len(english_tokens) >= 3
        and english_ratio > ptbr_ratio
    ):
        reason = "mostly_english_target"
    else:
        accepted = True
        retryable = False

    return _build_verdict(
        accepted=accepted,
        retryable=retryable,
        reason=reason,
        policy_id=policy_id,
        normalized_source_sha256=sha256_text(normalized_source),
        normalized_target_sha256=sha256_text(normalized_target),
        source_only_tokens=source_only_tokens,
        english_token_ratio_ppm=english_ratio,
        ptbr_token_ratio_ppm=ptbr_ratio,
        entities_equivalent=entities_ok,
        numbers_equivalent=numbers_ok,
        placeholders_equivalent=placeholders_ok,
    )


_RECOVERY_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_RECOVERY_SHORT_FUZZY = frozenset({"how", "why", "who", "what", "where"})
_RECOVERY_CONTENT_ANCHORS = frozenset(word for word in _ENGLISH_WORDS if len(word) >= 5)


@dataclass(frozen=True)
class NoisySourceRecovery:
    anchors: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class _RecoveryAnchorCandidate:
    start: int
    end: int
    value: str
    score: int
    content: bool


def _recovery_levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def _recovery_best_window(chunk: str, target: str) -> tuple[int, int, int] | None:
    if not chunk or not target:
        return None
    best: tuple[int, int, int] | None = None
    for width in range(max(1, len(target) - 2), min(len(chunk), len(target) + 2) + 1):
        for start in range(0, len(chunk) - width + 1):
            candidate = (_recovery_levenshtein(chunk[start : start + width], target), start, width)
            if best is None or candidate < best:
                best = candidate
    return best


def _recovery_compact(value: str) -> str:
    return re.sub(r"[^a-z]", "", str(value or "").casefold())


def _recovery_context_names(context_names: Iterable[str]) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for raw_name in context_names:
        display = " ".join(str(raw_name or "").split())
        compact = _recovery_compact(display)
        if len(compact) >= 5 and display:
            result.append((display.upper(), compact))
    return tuple(result)


def _recovery_is_low_coherence(text: str) -> bool:
    chunks = [match.group(0) for match in _RECOVERY_WORD_RE.finditer(str(text or ""))]
    return len(chunks) >= 3 and sum(len(_recovery_compact(chunk)) >= 10 for chunk in chunks) >= 2


def _recovery_word_candidates(text: str) -> list[_RecoveryAnchorCandidate]:
    candidates: list[_RecoveryAnchorCandidate] = []
    for match in _RECOVERY_WORD_RE.finditer(text):
        chunk = _recovery_compact(match.group(0))
        if len(chunk) < 3:
            continue
        for word in _ENGLISH_WORDS:
            if len(word) < 3:
                continue
            best = _recovery_best_window(chunk, word)
            if best is None:
                continue
            distance, local_start, width = best
            exact = distance == 0
            allowed = 1 if word in _RECOVERY_SHORT_FUZZY and len(chunk) <= 7 else (2 if len(word) >= 5 else 0)
            if distance > allowed:
                continue
            fragment = chunk[local_start : local_start + width]
            if not exact and fragment[:1] != word[:1]:
                continue
            start = match.start() + local_start
            candidates.append(_RecoveryAnchorCandidate(start, start + width, word.upper(), (100 if exact else 70) + len(word) * 4 - distance * 8, word in _RECOVERY_CONTENT_ANCHORS))
    return candidates


def _recovery_name_candidates(text: str, context_names: Iterable[str]) -> list[_RecoveryAnchorCandidate]:
    candidates: list[_RecoveryAnchorCandidate] = []
    for match in _RECOVERY_WORD_RE.finditer(text):
        chunk = _recovery_compact(match.group(0))
        if len(chunk) < 7:
            continue
        for display, compact_name in _recovery_context_names(context_names):
            best = _recovery_best_window(chunk, compact_name)
            if best is None:
                continue
            distance, local_start, width = best
            if distance > max(2, min(4, len(compact_name) // 2)):
                continue
            start = match.start() + local_start
            candidates.append(_RecoveryAnchorCandidate(start, start + width, display, 180 + len(compact_name) * 4 - distance * 8, True))
    return candidates


def _select_recovery_candidates(candidates: Iterable[_RecoveryAnchorCandidate]) -> tuple[_RecoveryAnchorCandidate, ...]:
    selected: list[_RecoveryAnchorCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.start, -(item.end - item.start), item.value)):
        if any(candidate.start < item.end and item.start < candidate.end for item in selected):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: (item.start, item.end, item.value)))


def recover_noisy_source_anchors(text: str, *, context_names: Iterable[str] = ()) -> NoisySourceRecovery | None:
    """Recover ordered anchors only from clearly incoherent long OCR chunks."""
    source = str(text or "").strip()
    if not _recovery_is_low_coherence(source):
        return None
    selected = _select_recovery_candidates([*_recovery_word_candidates(source), *_recovery_name_candidates(source, context_names)])
    anchors: list[str] = []
    for candidate in selected:
        if candidate.value not in anchors:
            anchors.append(candidate.value)
    if len(anchors) < 3 or sum(1 for item in selected if item.content) < 2:
        return None
    return NoisySourceRecovery(tuple(anchors), "bilingual_overlay_low_lexical_coherence")


__all__ = [
    "NoisySourceRecovery",
    "PageLanguageEvidence",
    "TargetLanguageVerdict",
    "build_page_language_evidence",
    "recover_noisy_source_anchors",
    "validate_target_language",
]
