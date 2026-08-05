from __future__ import annotations

from dataclasses import replace
import json

import pytest

from ownership.coverage import (
    CoverageComponentInventoryEntry,
    CoverageEntry,
    CoverageInvariantError,
    CoverageIdentityError,
    CoverageObservationDisposition,
    CoverageRecoveryExhausted,
    PageCoverageResult,
    build_component_inventory_entry,
    build_page_coverage_ledger,
    build_recovery_decision,
    build_recovery_request,
    complete_page_coverage,
    complete_container_coverage,
    recover_unassociated_observations,
    validate_inventory_successor,
    validate_recovery_chain,
)
from ownership.hash_contract import canonical_json_bytes, canonical_page_sha256
from ownership.model import SourceTextComponent
from ownership.ocr_contract import OCRRequest
from vision_stack.ocr import OCREngine


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


def _coverage_page() -> object:
    import numpy as np

    return np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3)


def _coverage_component() -> SourceTextComponent:
    return SourceTextComponent(
        component_id="component-coverage",
        page_id=PAGE_ID,
        bbox_page=(20, 20, 100, 55),
        polygon_page=((20, 20), (100, 20), (100, 55), (20, 55)),
        detector_sources=("glyph_scan",),
    )


def _coverage_runner():
    class FakePaddleModel:
        def ocr(self, image, det=True, rec=True, cls=False):
            del det, rec, cls
            if image.shape[:2] == (80, 120):
                return [[]]
            return [[([[2, 3], [60, 3], [60, 20], [2, 20]], ("VISIBLE ENGLISH", 0.96))]]

    engine = OCREngine.__new__(OCREngine)
    engine._backend = "paddleocr"
    engine._model = FakePaddleModel()

    def run(page, *, request, bbox_page, variants):
        if bbox_page is None:
            return engine.recognize_page_with_evidence(page, [], request=request, force_full_page=True)
        return engine.recognize_region_with_evidence(
            page,
            bbox_page=bbox_page,
            request=request,
            variants=variants,
        )

    return run


def _completed_coverage() -> PageCoverageResult:
    page = _coverage_page()
    coverage = complete_page_coverage(
        page,
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=canonical_page_sha256(page),
        components=(_coverage_component(),),
        band_evidence=[],
        ocr_runner=_coverage_runner(),
    )
    return complete_container_coverage(page, coverage)


def _coverage_recovery_request(
    coverage: PageCoverageResult,
    request_id: str = "request-coverage-recovery",
):
    return build_recovery_request(
        request_id=request_id,
        run_id=coverage.run_id,
        origin_execution_id=coverage.origin_execution_id,
        page_id=coverage.page_id,
        page_source_sha256=coverage.page_source_sha256,
        component_id=None,
        observation_ids=(coverage.observations[0].observation_id,),
        anchor_polygon_page=_coverage_component().polygon_page,
        attempt_kind="full_page_rescue",
        transform_spec_sha256="b" * 64,
        geometry_sha256="c" * 64,
        input_pixel_sha256="d" * 64,
        reason="coverage_recovery",
        evidence_ids=(coverage.observations[0].observation_id,),
        parent_decision=None,
        next_strategy=None,
    )


def _unassociated_coverage(*texts: str) -> tuple[object, PageCoverageResult]:
    page = _coverage_page()

    class FakePaddleModel:
        def ocr(self, image, det=True, rec=True, cls=False):
            del image, det, rec, cls
            lines = []
            for index, text in enumerate(texts):
                x1, y1 = 10 + index * 55, 12 + index * 22
                lines.append(([[x1, y1], [x1 + 42, y1], [x1 + 42, y1 + 16], [x1, y1 + 16]], (text, 0.96)))
            return [lines]

    engine = OCREngine.__new__(OCREngine)
    engine._backend = "paddleocr"
    engine._model = FakePaddleModel()

    def runner(page_rgb, *, request, bbox_page, variants):
        assert bbox_page is None
        del variants
        return engine.recognize_page_with_evidence(
            page_rgb, [], request=request, force_full_page=True
        )

    coverage = complete_page_coverage(
        page,
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=canonical_page_sha256(page),
        components=(),
        band_evidence=(),
        ocr_runner=runner,
    )
    return page, coverage


def test_component_without_band_gets_anchored_ocr_before_owner_resolution() -> None:
    coverage = _completed_coverage()

    assert coverage.ledger.entries[0].ocr_attempt_ids
    assert coverage.observations
    assert coverage.observations[0].component_ids == ("component-coverage",)
    assert coverage.ocr_requests
    assert coverage.ocr_invocations
    physical_attempt_ids = {
        attempt.attempt_id
        for invocation in coverage.ocr_invocations
        for attempt in invocation.attempts
    }
    assert set(coverage.ledger.entries[0].ocr_attempt_ids) <= physical_attempt_ids


def test_unique_full_page_observation_without_component_materializes_component() -> None:
    page, initial = _unassociated_coverage("READ ME")

    result = recover_unassociated_observations(page, initial)

    assert len(result.components) == 1
    assert result.observations[0].component_ids == (result.components[0].component_id,)
    assert result.recovery_decisions[0].reason == "materialized_from_full_page_observation"
    assert result.ledger.expected_component_ids == (result.components[0].component_id,)
    assert (
        result.ledger.component_inventory[0].introduced_by_decision_id
        == result.recovery_decisions[0].decision_id
    )
    assert result.ledger.parent_ledger_sha256 == initial.ledger.sha256
    assert result.ocr_requests == initial.ocr_requests
    assert result.ocr_invocations == initial.ocr_invocations
    assert result.entries[0].protection_conflict
    assert result.entries[0].protection_evidence_ids == (
        result.observations[0].observation_id,
    )


def test_two_unassociated_regions_get_distinct_pending_recovery_fingerprints() -> None:
    page, initial = _unassociated_coverage("FIRST", "SECOND")

    pending = recover_unassociated_observations(page, initial)

    assert len(pending.pending_requests) == 2
    assert len({item.anchor_sha256 for item in pending.pending_requests}) == 2
    assert len({item.attempt_fingerprint for item in pending.pending_requests}) == 2
    assert not any(item.rejection_reason == "suppressed" for item in pending.observations)


def test_ambiguous_full_page_observation_requests_anchored_recovery() -> None:
    page, raw = _unassociated_coverage("WAIT")
    components = (
        SourceTextComponent(
            component_id="component-left",
            page_id=PAGE_ID,
            bbox_page=(8, 10, 35, 30),
            polygon_page=((8, 10), (35, 10), (35, 30), (8, 30)),
            detector_sources=("glyph_scan",),
        ),
        SourceTextComponent(
            component_id="component-right",
            page_id=PAGE_ID,
            bbox_page=(30, 10, 55, 30),
            polygon_page=((30, 10), (55, 10), (55, 30), (30, 30)),
            detector_sources=("glyph_scan",),
        ),
    )
    initial = PageCoverageResult.initialize(
        run_id=raw.run_id,
        origin_execution_id=raw.origin_execution_id,
        page_id=raw.page_id,
        page_source_sha256=raw.page_source_sha256,
        components=components,
    )
    initial = PageCoverageResult.build_from(
        initial,
        observations=raw.observations,
        ocr_requests=raw.ocr_requests,
        ocr_invocations=raw.ocr_invocations,
    )

    result = recover_unassociated_observations(page, initial)

    assert len(result.pending_requests) == 1
    request = result.pending_requests[0]
    assert request.component_id is None
    assert request.observation_ids == (raw.observations[0].observation_id,)
    assert request.anchor_polygon_page == raw.observations[0].polygons_page[0]
    assert result.observations[0].rejection_reason != "suppressed"


def test_ocr_confirmed_component_recovers_container_before_graph_build() -> None:
    coverage = _completed_coverage()

    recovered = complete_container_coverage(_coverage_page(), coverage)

    assert recovered.entries[0].container_id
    assert recovered.entries[0].state == "observed"


def test_coverage_recovery_snapshot_never_drops_request_or_invocation_evidence() -> None:
    first = _completed_coverage()
    page = _coverage_page()
    request = OCRRequest(
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=canonical_page_sha256(page),
        root_input_pixel_sha256=canonical_page_sha256(page),
        invocation_id=f"{PAGE_ID}:coverage:fresh-recovery",
        provider_family="paddleocr",
    )
    invocation = _coverage_runner()(
        page,
        request=request,
        bbox_page=_coverage_component().bbox_page,
        variants=("native",),
    )
    recovered = PageCoverageResult.build_from(
        first,
        observations=first.observations,
        ocr_requests=(*first.ocr_requests, request),
        ocr_invocations=(*first.ocr_invocations, invocation),
    )

    assert first.ocr_requests == recovered.ocr_requests[: len(first.ocr_requests)]
    assert first.ocr_invocations == recovered.ocr_invocations[: len(first.ocr_invocations)]
    assert recovered.sha256 != first.sha256


def test_coverage_ledger_history_is_reopenable_and_hash_chained_after_restart() -> None:
    first = _completed_coverage()
    reopened = PageCoverageResult.from_canonical_json_bytes(first.canonical_json_bytes)

    assert tuple(item.sha256 for item in reopened.ledger_history) == tuple(
        item.sha256 for item in first.ledger_history
    )
    assert reopened.ledger_history[0].parent_ledger_sha256 is None
    reopened.require_ready_for_ownership()


def test_coverage_build_from_rejects_dropped_reordered_or_stale_history() -> None:
    first = _completed_coverage()
    with pytest.raises(CoverageInvariantError):
        PageCoverageResult.build_from(first, ocr_requests=tuple(reversed(first.ocr_requests)))

    stale = replace(first.observations[0], origin_execution_id="execution-other")
    with pytest.raises((CoverageIdentityError, CoverageInvariantError)):
        PageCoverageResult.build_from(first, observations=(*first.observations, stale))


def test_coverage_recovery_history_derives_pending_ids_and_preserves_request() -> None:
    first = _completed_coverage()
    request = _coverage_recovery_request(first)
    pending = PageCoverageResult.build_from(
        first,
        recovery_requests=(request,),
    )
    assert pending.pending_request_ids == (request.request_id,)

    decision = build_recovery_decision(
        decision_id="decision-coverage-success",
        request=request,
        materialized_component_id="component-b",
        attempt_id="attempt-coverage-success",
        status="succeeded",
        reason="recovered",
        evidence_ids=("evidence-success",),
        next_strategy=None,
    )
    resolved = PageCoverageResult.build_from(
        pending,
        recovery_decisions=(decision,),
    )
    assert resolved.recovery_requests == (request,)
    assert resolved.pending_request_ids == ()


def test_coverage_rejects_duplicate_attempt_fingerprint_with_distinct_request_ids() -> None:
    first = _completed_coverage()
    request = _coverage_recovery_request(first)
    duplicate = build_recovery_request(
        request_id="request-duplicate-fingerprint",
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        component_id=request.component_id,
        observation_ids=request.observation_ids,
        anchor_polygon_page=request.anchor_polygon_page,
        attempt_kind=request.attempt_kind,
        transform_spec_sha256=request.transform_spec_sha256,
        geometry_sha256=request.geometry_sha256,
        input_pixel_sha256=request.input_pixel_sha256,
        reason=request.reason,
        evidence_ids=request.evidence_ids,
        parent_decision=None,
        next_strategy=request.next_strategy,
    )
    with pytest.raises(CoverageInvariantError, match="attempt_fingerprint|hash"):
        PageCoverageResult.build_from(
            first,
            recovery_requests=(request, duplicate),
        )


def test_coverage_ledger_reopen_rejects_missing_predecessor() -> None:
    first = _completed_coverage()
    request = _coverage_recovery_request(first)
    second = PageCoverageResult.build_from(first, recovery_requests=(request,))
    decision = build_recovery_decision(
        decision_id="decision-history-success",
        request=request,
        materialized_component_id="component-materialized",
        attempt_id="attempt-history-success",
        status="succeeded",
        reason="history_success",
        evidence_ids=("history-success",),
        next_strategy=None,
    )
    third = PageCoverageResult.build_from(second, recovery_decisions=(decision,))
    payload = json.loads(third.canonical_json_bytes.decode("utf-8"))
    del payload["ledger_history"][1]

    with pytest.raises(CoverageInvariantError, match="parent_ledger_sha256"):
        PageCoverageResult.from_canonical_json_bytes(canonical_json_bytes(payload))


def test_coverage_readiness_rejects_inconsistent_pending_view() -> None:
    first = _completed_coverage()
    request = _coverage_recovery_request(first)

    with pytest.raises(CoverageInvariantError, match="pending"):
        PageCoverageResult.build_from(
            first,
            recovery_requests=(request,),
            pending_request_ids=(),
        )


def test_failed_recovery_is_ready_only_after_hash_linked_successor_succeeds() -> None:
    first = _completed_coverage()
    root = _coverage_recovery_request(first, "request-root")
    failed = build_recovery_decision(
        decision_id="decision-root-failed",
        request=root,
        materialized_component_id=None,
        attempt_id="attempt-root-failed",
        status="failed",
        reason="insufficient_coverage",
        evidence_ids=("attempt-root-failed",),
        next_strategy="anchored_gray",
    )
    child = build_recovery_request(
        request_id="request-child",
        run_id=root.run_id,
        origin_execution_id=root.origin_execution_id,
        page_id=root.page_id,
        page_source_sha256=root.page_source_sha256,
        component_id=root.component_id,
        observation_ids=root.observation_ids,
        anchor_polygon_page=root.anchor_polygon_page,
        attempt_kind="anchored_gray",
        transform_spec_sha256="e" * 64,
        geometry_sha256=root.geometry_sha256,
        input_pixel_sha256=root.input_pixel_sha256,
        reason="retry_failed_recovery",
        evidence_ids=root.evidence_ids,
        parent_decision=failed,
        next_strategy=None,
    )
    succeeded = build_recovery_decision(
        decision_id="decision-child-success",
        request=child,
        materialized_component_id="component-materialized-child",
        attempt_id="attempt-child-success",
        status="succeeded",
        reason="child_succeeded",
        evidence_ids=("attempt-child-success",),
        next_strategy=None,
    )
    result = PageCoverageResult.build_from(
        first,
        recovery_requests=(root, child),
        recovery_decisions=(failed, succeeded),
    )

    assert child.parent_decision_id == failed.decision_id
    assert child.parent_decision_sha256 == failed.decision_sha256
    assert result.pending_request_ids == ()
    result.require_ready_for_ownership()


def test_exhausted_recovery_aborts_even_with_empty_pending_view() -> None:
    first = _completed_coverage()
    request = _coverage_recovery_request(first)
    exhausted = build_recovery_decision(
        decision_id="decision-exhausted",
        request=request,
        materialized_component_id=None,
        attempt_id="attempt-exhausted",
        status="exhausted",
        reason="strategies_exhausted",
        evidence_ids=("attempt-exhausted",),
        next_strategy=None,
    )
    result = PageCoverageResult.build_from(
        first,
        recovery_requests=(request,),
        recovery_decisions=(exhausted,),
    )

    assert result.pending_request_ids == ()
    with pytest.raises(CoverageRecoveryExhausted):
        result.require_ready_for_ownership()


@pytest.mark.parametrize(
    "component_id,observation_ids,polygon",
    [
        (None, (), None),
        ("component-a", ("observation-a",), POLYGON_A),
        (None, ("observation-a",), None),
        (None, (), POLYGON_A),
    ],
)
def test_coverage_recovery_request_requires_exactly_one_anchor(
    component_id, observation_ids, polygon
) -> None:
    with pytest.raises(CoverageInvariantError):
        build_recovery_request(
            request_id="request-invalid-anchor",
            run_id=RUN_ID,
            origin_execution_id=EXECUTION_ID,
            page_id=PAGE_ID,
            page_source_sha256=PAGE_SHA,
            component_id=component_id,
            observation_ids=observation_ids,
            anchor_polygon_page=polygon,
            attempt_kind="anchored_native",
            transform_spec_sha256="b" * 64,
            geometry_sha256="c" * 64,
            input_pixel_sha256="d" * 64,
            reason="invalid_anchor",
            evidence_ids=(),
            parent_decision=None,
            next_strategy=None,
        )
