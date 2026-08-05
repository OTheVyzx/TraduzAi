"""Immutable owner lifecycle with identity-preserving transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .coverage import (
    CANONICAL_COVERAGE_STATES,
    TERMINAL_COVERAGE_STATES,
    CoverageInvariantError,
)
from .hash_contract import canonical_json_sha256


_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "discovered": frozenset({"challenged", "observed", "explicit_non_dialogue_preserve"}),
    "challenged": frozenset({"observed", "explicit_non_dialogue_preserve"}),
    "observed": frozenset({"owned", "explicit_non_dialogue_preserve"}),
    "owned": frozenset({"target_ready"}),
    "target_ready": frozenset({"execution_attempt"}),
    "execution_attempt": frozenset({"repair_pending", "cleaned"}),
    "repair_pending": frozenset({"execution_attempt"}),
    "cleaned": frozenset({"rendered"}),
    "rendered": frozenset({"repair_pending", "final_verified"}),
    "final_verified": frozenset(),
    "explicit_non_dialogue_preserve": frozenset(),
}


@dataclass(frozen=True)
class OwnerLifecycleIdentity:
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    owner_id: str

    def __post_init__(self) -> None:
        if not all(str(value or "").strip() for value in asdict(self).values()):
            raise CoverageInvariantError("owner lifecycle identity is incomplete")
        if len(self.page_source_sha256) != 64:
            raise CoverageInvariantError("owner lifecycle page hash is invalid")


@dataclass(frozen=True)
class LifecycleEvidence:
    identity: OwnerLifecycleIdentity
    evidence_id: str
    evidence_kind: str
    payload_sha256: str
    evidence_sha256: str

    @classmethod
    def build(
        cls,
        *,
        identity: OwnerLifecycleIdentity,
        evidence_id: str,
        evidence_kind: str,
        payload_sha256: str,
    ) -> "LifecycleEvidence":
        if not evidence_id or not evidence_kind or len(payload_sha256) != 64:
            raise CoverageInvariantError("lifecycle evidence is incomplete")
        payload = {
            "identity": asdict(identity),
            "evidence_id": evidence_id,
            "evidence_kind": evidence_kind,
            "payload_sha256": payload_sha256,
        }
        return cls(
            identity=identity,
            evidence_id=evidence_id,
            evidence_kind=evidence_kind,
            payload_sha256=payload_sha256,
            evidence_sha256=canonical_json_sha256(payload),
        )

    def require_valid(self) -> None:
        expected = canonical_json_sha256(
            {
                "identity": asdict(self.identity),
                "evidence_id": self.evidence_id,
                "evidence_kind": self.evidence_kind,
                "payload_sha256": self.payload_sha256,
            }
        )
        if expected != self.evidence_sha256:
            raise CoverageInvariantError("lifecycle evidence hash mismatch")


@dataclass(frozen=True)
class OwnerLifecycle:
    identity: OwnerLifecycleIdentity
    state: str
    evidence_sha256s: tuple[str, ...]
    history: tuple[str, ...]

    @classmethod
    def start(
        cls,
        identity: OwnerLifecycleIdentity,
        *,
        evidence: LifecycleEvidence,
    ) -> "OwnerLifecycle":
        evidence.require_valid()
        if evidence.identity != identity:
            raise CoverageInvariantError("lifecycle evidence belongs to another owner")
        return cls(
            identity=identity,
            state="discovered",
            evidence_sha256s=(evidence.evidence_sha256,),
            history=("discovered",),
        )

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_COVERAGE_STATES

    def advance(
        self,
        state: str,
        *,
        evidence: LifecycleEvidence,
    ) -> "OwnerLifecycle":
        if state not in CANONICAL_COVERAGE_STATES:
            raise CoverageInvariantError(f"owner lifecycle state is invalid: {state}")
        if state not in _ALLOWED_TRANSITIONS.get(self.state, frozenset()):
            raise CoverageInvariantError(
                f"owner lifecycle transition is invalid: {self.state}->{state}"
            )
        evidence.require_valid()
        if evidence.identity != self.identity:
            raise CoverageInvariantError("lifecycle transition changed owner identity")
        return replace(
            self,
            state=state,
            evidence_sha256s=(*self.evidence_sha256s, evidence.evidence_sha256),
            history=(*self.history, state),
        )


def advance_owner_with_translation(lifecycle: OwnerLifecycle, binding) -> OwnerLifecycle:
    """Advance one owner only from its own accepted, hash-bound PT-BR binding."""

    binding_identity = OwnerLifecycleIdentity(
        run_id=str(getattr(binding, "run_id", "")),
        origin_execution_id=str(getattr(binding, "origin_execution_id", "")),
        page_id=str(getattr(binding, "page_id", "")),
        page_source_sha256=str(getattr(binding, "page_source_sha256", "")),
        owner_id=str(getattr(binding, "owner_id", "")),
    )
    if binding_identity != lifecycle.identity:
        raise CoverageInvariantError("translation binding belongs to another owner")
    verdict = getattr(binding, "language_verdict", None)
    if (
        getattr(binding, "target_locale", None) != "pt-BR"
        or verdict is None
        or not bool(getattr(verdict, "accepted", False))
    ):
        raise CoverageInvariantError("target_ready requires accepted PT-BR binding")
    binding_sha256 = str(getattr(binding, "translation_binding_sha256", ""))
    canonical_payload = getattr(binding, "canonical_payload", None)
    if (
        len(binding_sha256) != 64
        or not callable(canonical_payload)
        or canonical_json_sha256(canonical_payload()) != binding_sha256
    ):
        raise CoverageInvariantError("translation binding hash mismatch")
    evidence = LifecycleEvidence.build(
        identity=lifecycle.identity,
        evidence_id=f"translation-binding:{binding_sha256[:24]}",
        evidence_kind="translation_binding",
        payload_sha256=binding_sha256,
    )
    return lifecycle.advance("target_ready", evidence=evidence)


def reopen_owner_from_final_issue(lifecycle: OwnerLifecycle, issue) -> OwnerLifecycle:
    """Move one rendered owner back to repair_pending from its canonical QA issue."""

    if lifecycle.state != "rendered":
        raise CoverageInvariantError("only a rendered owner can reopen from final QA")
    issue_identity = (
        str(getattr(issue, "run_id", "")),
        str(getattr(issue, "execution_id", "")),
        str(getattr(issue, "page_id", "")),
        str(getattr(issue, "page_source_sha256", "")),
        str(getattr(issue, "owner_id", "")),
    )
    expected = (
        lifecycle.identity.run_id,
        lifecycle.identity.origin_execution_id,
        lifecycle.identity.page_id,
        lifecycle.identity.page_source_sha256,
        lifecycle.identity.owner_id,
    )
    if issue_identity != expected or not bool(getattr(issue, "repair_required", False)):
        raise CoverageInvariantError("final QA issue belongs to another owner or is not repairable")
    issue_sha = str(getattr(issue, "issue_sha256", ""))
    evidence = LifecycleEvidence.build(
        identity=lifecycle.identity,
        evidence_id=f"final-qa-issue:{getattr(issue, 'issue_id', '')}",
        evidence_kind="final_qa_repair",
        payload_sha256=issue_sha,
    )
    return lifecycle.advance("repair_pending", evidence=evidence)


def mark_owner_final_verified(lifecycle: OwnerLifecycle, verdict) -> OwnerLifecycle:
    """Close one rendered owner only with its final replacement verdict."""

    if lifecycle.state != "rendered":
        raise CoverageInvariantError("only a rendered owner can become final_verified")
    verdict_identity = (
        str(getattr(verdict, "run_id", "")),
        str(getattr(verdict, "execution_id", "")),
        str(getattr(verdict, "page_id", "")),
        str(getattr(verdict, "page_source_sha256", "")),
        str(getattr(verdict, "owner_id", "")),
    )
    expected = (
        lifecycle.identity.run_id,
        lifecycle.identity.origin_execution_id,
        lifecycle.identity.page_id,
        lifecycle.identity.page_source_sha256,
        lifecycle.identity.owner_id,
    )
    if (
        verdict_identity != expected
        or getattr(verdict, "status", None) != "final_verified"
        or not bool(getattr(verdict, "source_support_removed", False))
        or not bool(getattr(verdict, "target_materialized", False))
    ):
        raise CoverageInvariantError("final verdict does not close this owner")
    evidence = LifecycleEvidence.build(
        identity=lifecycle.identity,
        evidence_id=f"final-verdict:{getattr(verdict, 'verdict_id', '')}",
        evidence_kind="final_replacement_verdict",
        payload_sha256=str(getattr(verdict, "verdict_sha256", "")),
    )
    return lifecycle.advance("final_verified", evidence=evidence)
