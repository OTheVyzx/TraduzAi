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
    "rendered": frozenset({"final_verified"}),
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
