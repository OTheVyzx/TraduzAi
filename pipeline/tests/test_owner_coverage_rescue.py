from __future__ import annotations

from dataclasses import replace

import pytest

from ownership.coverage import (
    CoverageComponentInventoryEntry,
    CoverageEntry,
    CoverageInvariantError,
    CoverageObservationDisposition,
    build_component_inventory_entry,
    build_page_coverage_ledger,
    build_recovery_decision,
    build_recovery_request,
    validate_inventory_successor,
    validate_recovery_chain,
)


RUN_ID = "run-coverage"
EXECUTION_ID = "execution-coverage"
PAGE_ID = "page_010"
PAGE_SHA = "a" * 64
POLYGON_A = ((10, 20), (100, 20), (100, 60), (10, 60))
POLYGON_B = ((120, 80), (220, 80), (220, 130), (120, 130))


def _entry(
    component_id: str = "component-a",
    *,
    state: str = "final_verified",
    semantic_role: str = "dialogue_body",
    owner_id: str | None = "owner-a",
    ocr_attempt_ids: tuple[str, ...] = ("attempt-a",),
    observation_ids: tuple[str, ...] = ("observation-a",),
    polygon=POLYGON_A,
    preserve_policy: str | None = None,
) -> CoverageEntry:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return CoverageEntry(
        component_id=component_id,
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=PAGE_SHA,
        bbox_page=(min(xs), min(ys), max(xs), max(ys)),
        polygon_page=polygon,
        materiality="material",
        ocr_attempt_ids=ocr_attempt_ids,
        observation_ids=observation_ids,
        owner_id=owner_id,
        semantic_role=semantic_role,
        state=state,
        preserve_policy=preserve_policy,
    )


def _inventory(component_id: str, ordinal: int = 0) -> CoverageComponentInventoryEntry:
    polygon = POLYGON_A if component_id == "component-a" else POLYGON_B
    return build_component_inventory_entry(
        component_id=component_id,
        origin="discovery",
        introduced_by_decision_id=None,
        anchor_polygon_page=polygon,
        ordinal=ordinal,
    )


def _ledger(*, inventory=None, entries=None, expected_observations=None, dispositions=()):
    return build_page_coverage_ledger(
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=PAGE_SHA,
        inventory_version=1,
        parent_ledger_sha256=None,
        component_inventory=tuple(inventory or (_inventory("component-a"),)),
        expected_observation_ids=tuple(expected_observations or ("observation-a",)),
        entries=tuple(entries or (_entry(),)),
        observation_dispositions=tuple(dispositions),
    )


def test_every_material_component_has_exactly_one_terminal_lifecycle() -> None:
    preserved = _entry(
        "component-b",
        state="explicit_non_dialogue_preserve",
        semantic_role="external_credit",
        owner_id=None,
        ocr_attempt_ids=("attempt-b",),
        observation_ids=("observation-b",),
        polygon=POLYGON_B,
        preserve_policy="policy:external_credit",
    )
    ledger = _ledger(
        inventory=(_inventory("component-a", 0), _inventory("component-b", 1)),
        entries=(_entry(), preserved),
        expected_observations=("observation-a", "observation-b"),
    )

    ledger.require_complete()


@pytest.mark.parametrize("state", ["suppress", "review_required", "rendered"])
def test_dialogue_component_cannot_finish_nonterminal(state: str) -> None:
    with pytest.raises(CoverageInvariantError):
        _ledger(entries=(_entry(state=state),)).require_complete()


def test_component_without_ocr_attempt_cannot_be_classified_non_text() -> None:
    entry = replace(
        _entry(
            state="explicit_non_dialogue_preserve",
            semantic_role="external_credit",
            owner_id=None,
            preserve_policy="policy:external_credit",
        ),
        materiality="non_text",
        ocr_attempt_ids=(),
    )
    with pytest.raises(CoverageInvariantError):
        _ledger(entries=(entry,)).require_complete()


def test_ledger_rejects_expected_component_missing_from_entries() -> None:
    ledger = _ledger(
        inventory=(_inventory("component-a", 0), _inventory("component-b", 1)),
        entries=(_entry(),),
    )
    with pytest.raises(CoverageInvariantError, match="component-b"):
        ledger.require_complete()


def test_expected_observation_requires_exactly_one_entry_or_disposition() -> None:
    disposition = CoverageObservationDisposition(
        observation_id="observation-unassociated",
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=PAGE_SHA,
        disposition="explicit_non_dialogue_preserve",
        reason="policy:decorative_non_dialogue",
        evidence_ids=("evidence-a",),
    )
    ledger = _ledger(
        expected_observations=("observation-a", "observation-unassociated"),
        dispositions=(disposition,),
    )
    ledger.require_complete()

    with pytest.raises(CoverageInvariantError, match="observation-unassociated"):
        duplicated = build_page_coverage_ledger(
            run_id=ledger.run_id,
            origin_execution_id=ledger.origin_execution_id,
            page_id=ledger.page_id,
            page_source_sha256=ledger.page_source_sha256,
            inventory_version=ledger.inventory_version,
            parent_ledger_sha256=ledger.parent_ledger_sha256,
            component_inventory=ledger.component_inventory,
            expected_observation_ids=ledger.expected_observation_ids,
            entries=(replace(ledger.entries[0], observation_ids=("observation-a", "observation-unassociated")),),
            observation_dispositions=ledger.observation_dispositions,
        )
        duplicated.require_complete()


def _unassociated_request(*, parent_decision=None, attempt_kind="full_page_rescue"):
    return build_recovery_request(
        request_id=f"request-{attempt_kind}",
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=PAGE_SHA,
        component_id=None,
        observation_ids=("observation-b",),
        anchor_polygon_page=POLYGON_B,
        attempt_kind=attempt_kind,
        transform_spec_sha256="b" * 64,
        geometry_sha256="c" * 64,
        input_pixel_sha256="d" * 64,
        reason="unassociated_full_page_observation",
        evidence_ids=("observation-b",),
        parent_decision=parent_decision,
        next_strategy=None,
    )


def test_component_inventory_allows_only_recovery_bound_append() -> None:
    first = _ledger()
    request = _unassociated_request()
    decision = build_recovery_decision(
        decision_id="decision-materialized-b",
        request=request,
        materialized_component_id="component-b",
        attempt_id="attempt-materialized-b",
        status="succeeded",
        reason="component_materialized",
        evidence_ids=("component-b",),
        next_strategy=None,
    )
    inventory_entry = build_component_inventory_entry(
        component_id="component-b",
        origin="recovery_materialization",
        introduced_by_decision_id=decision.decision_id,
        anchor_polygon_page=POLYGON_B,
        ordinal=1,
        decision=decision,
    )

    second = first.extend_component_inventory(
        inventory_entry,
        request=request,
        decision=decision,
    )

    assert second.inventory_version == first.inventory_version + 1
    assert second.parent_ledger_sha256 == first.sha256
    assert second.expected_component_ids == ("component-a", "component-b")
    validate_inventory_successor(first, second)


@pytest.mark.parametrize("tamper", ["remove", "replace", "reorder", "unbound_append"])
def test_component_inventory_rejects_non_monotonic_or_unbound_change(tamper: str) -> None:
    first = _ledger()
    request = _unassociated_request()
    decision = build_recovery_decision(
        decision_id="decision-materialized-b",
        request=request,
        materialized_component_id="component-b",
        attempt_id="attempt-materialized-b",
        status="succeeded",
        reason="component_materialized",
        evidence_ids=("component-b",),
        next_strategy=None,
    )
    entry = build_component_inventory_entry(
        component_id="component-b",
        origin="recovery_materialization",
        introduced_by_decision_id=decision.decision_id,
        anchor_polygon_page=POLYGON_B,
        ordinal=1,
        decision=decision,
    )
    second = first.extend_component_inventory(entry, request=request, decision=decision)
    with pytest.raises(CoverageInvariantError):
        if tamper == "remove":
            candidate = replace(second, component_inventory=second.component_inventory[1:])
        elif tamper == "replace":
            candidate = replace(second, component_inventory=(second.component_inventory[0], _inventory("component-b", 1)))
        elif tamper == "reorder":
            candidate = replace(second, component_inventory=tuple(reversed(second.component_inventory)))
        else:
            candidate = replace(second, recovery_decisions=(), recovery_requests=())
        validate_inventory_successor(first, candidate)


def test_recovery_anchor_is_exactly_component_or_unassociated_observation_region() -> None:
    with pytest.raises(CoverageInvariantError):
        build_recovery_request(
            request_id="invalid-anchor",
            run_id=RUN_ID,
            origin_execution_id=EXECUTION_ID,
            page_id=PAGE_ID,
            page_source_sha256=PAGE_SHA,
            component_id="component-a",
            observation_ids=("observation-b",),
            anchor_polygon_page=POLYGON_B,
            attempt_kind="full_page_rescue",
            transform_spec_sha256="b" * 64,
            geometry_sha256="c" * 64,
            input_pixel_sha256="d" * 64,
            reason="mixed_anchor",
            evidence_ids=(),
            parent_decision=None,
            next_strategy=None,
        )


def test_failed_recovery_requires_exactly_one_hash_bound_strategy_child() -> None:
    first = _unassociated_request()
    failed = build_recovery_decision(
        decision_id="decision-failed",
        request=first,
        materialized_component_id=None,
        attempt_id="attempt-failed",
        status="failed",
        reason="insufficient_coverage",
        evidence_ids=("attempt-failed",),
        next_strategy="expanded_context_rescue",
    )
    child = build_recovery_request(
        request_id="request-expanded",
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=PAGE_SHA,
        component_id=None,
        observation_ids=first.observation_ids,
        anchor_polygon_page=first.anchor_polygon_page,
        attempt_kind="expanded_context_rescue",
        transform_spec_sha256="e" * 64,
        geometry_sha256=first.geometry_sha256,
        input_pixel_sha256=first.input_pixel_sha256,
        reason="retry_failed_recovery",
        evidence_ids=first.evidence_ids,
        parent_decision=failed,
        next_strategy=None,
    )

    validate_recovery_chain((first, child), (failed,))
    with pytest.raises(CoverageInvariantError):
        validate_recovery_chain((first, child, replace(child, request_id="duplicate-child")), (failed,))
