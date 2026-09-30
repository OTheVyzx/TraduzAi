from __future__ import annotations

from dataclasses import replace

import pytest

from ownership.lifecycle import (
    CoverageInvariantError,
    LifecycleEvidence,
    OwnerLifecycle,
    OwnerLifecycleIdentity,
)


def _identity() -> OwnerLifecycleIdentity:
    return OwnerLifecycleIdentity(
        run_id="run-lifecycle",
        origin_execution_id="execution-lifecycle",
        page_id="page-001",
        page_source_sha256="a" * 64,
        owner_id="owner-a",
    )


def _evidence(identity=None, evidence_id="evidence-a") -> LifecycleEvidence:
    return LifecycleEvidence.build(
        identity=identity or _identity(),
        evidence_id=evidence_id,
        evidence_kind="lifecycle_test",
        payload_sha256="b" * 64,
    )


def test_owner_identity_survives_every_lifecycle_transition() -> None:
    lifecycle = OwnerLifecycle.start(_identity(), evidence=_evidence())
    for state in (
        "observed",
        "owned",
        "target_ready",
        "execution_attempt",
        "cleaned",
        "rendered",
        "final_verified",
    ):
        lifecycle = lifecycle.advance(
            state,
            evidence=_evidence(evidence_id=f"evidence-{state}"),
        )

    assert lifecycle.identity == _identity()
    assert lifecycle.state == "final_verified"
    assert lifecycle.is_terminal


def test_owner_lifecycle_rejects_skipped_or_backward_transition() -> None:
    lifecycle = OwnerLifecycle.start(_identity(), evidence=_evidence())
    with pytest.raises(CoverageInvariantError):
        lifecycle.advance("final_verified", evidence=_evidence(evidence_id="jump"))


def test_owner_lifecycle_rejects_evidence_from_other_identity() -> None:
    lifecycle = OwnerLifecycle.start(_identity(), evidence=_evidence())
    other = replace(_identity(), owner_id="owner-other")
    with pytest.raises(CoverageInvariantError):
        lifecycle.advance("observed", evidence=_evidence(other, "foreign"))


def test_repair_pending_is_nonterminal_and_returns_only_to_execution_attempt() -> None:
    lifecycle = OwnerLifecycle.start(_identity(), evidence=_evidence())
    for state in ("observed", "owned", "target_ready", "execution_attempt", "repair_pending"):
        lifecycle = lifecycle.advance(state, evidence=_evidence(evidence_id=state))
    assert not lifecycle.is_terminal
    with pytest.raises(CoverageInvariantError):
        lifecycle.advance("cleaned", evidence=_evidence(evidence_id="invalid-clean"))
    retried = lifecycle.advance("execution_attempt", evidence=_evidence(evidence_id="retry"))
    assert retried.state == "execution_attempt"
