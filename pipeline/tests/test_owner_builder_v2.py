from __future__ import annotations

from dataclasses import replace

import pytest

from ownership.coverage import (
    CoverageEntry,
    CoverageInvariantError,
    PageCoverageResult,
    build_component_inventory_entry,
    build_page_coverage_ledger,
)
from ownership.hash_contract import sha256_text
from ownership.model import OwnerGraph, SourceTextComponent, TextObservation
from ownership.ocr_contract import (
    OCRAttempt,
    OCRDiagnostics,
    OCRInvocationResult,
    OCRObservationRecord,
    OCRRequest,
    OCRTransformOperation,
    OCRTransformSpec,
)
from ownership.owner_builder import build_owner_page_graph_from_coverage


RUN_ID = "run-owner-builder-v2"
EXECUTION_ID = "execution-owner-builder-v2"
PAGE_ID = "page_001"
PAGE_SHA = "a" * 64
ROOT_SHA = "b" * 64


def _component(component_id: str, bbox) -> SourceTextComponent:
    x1, y1, x2, y2 = bbox
    return SourceTextComponent(
        component_id=component_id,
        page_id=PAGE_ID,
        bbox_page=bbox,
        polygon_page=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
        detector_sources=("glyph_scan",),
    )


def _coverage(
    parts,
    *,
    containers,
    preserve_policy: str | None = None,
    semantic_role: str = "dialogue_body",
) -> PageCoverageResult:
    components = tuple(
        _component(component_id, bbox) for component_id, _text, bbox in parts
    )
    request = OCRRequest(
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=PAGE_SHA,
        root_input_pixel_sha256=ROOT_SHA,
        invocation_id="invocation-owner-builder-v2",
        provider_family="paddleocr",
    )
    transform = OCRTransformSpec.build((OCRTransformOperation(kind="identity"),))
    attempt = OCRAttempt(
        attempt_id="attempt-owner-builder-v2",
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=PAGE_SHA,
        root_input_pixel_sha256=ROOT_SHA,
        invocation_id=request.invocation_id,
        provider_family=request.provider_family,
        variant_id="full_page",
        input_pixel_sha256=ROOT_SHA,
        parent_input_pixel_sha256=ROOT_SHA,
        input_bbox_page=None,
        input_kind="full_page",
        transform_spec=transform,
        input_width=400,
        input_height=300,
        input_mode="RGB",
        provider_called=True,
        cache_hit=False,
    )
    records = tuple(
        OCRObservationRecord(
            observation_id=f"observation-{index}",
            attempt_id=attempt.attempt_id,
            run_id=RUN_ID,
            origin_execution_id=EXECUTION_ID,
            page_id=PAGE_ID,
            page_source_sha256=PAGE_SHA,
            root_input_pixel_sha256=ROOT_SHA,
            input_pixel_sha256=ROOT_SHA,
            invocation_id=request.invocation_id,
            provider_family=request.provider_family,
            variant_id=attempt.variant_id,
            payload_sha256=sha256_text(text),
            text=text,
            confidence=0.96,
            bbox_page=bbox,
            polygon_page=component.polygon_page,
            source="paddle_full_page",
        )
        for index, ((component_id, text, bbox), component) in enumerate(
            zip(parts, components), 1
        )
    )
    invocation = OCRInvocationResult.build(
        request=request,
        observations=records,
        full_page_lines=records,
        attempts=(attempt,),
        diagnostics=OCRDiagnostics("paddleocr"),
    )
    observations = tuple(
        TextObservation(
            observation_id=record.observation_id,
            page_id=PAGE_ID,
            component_ids=(component.component_id,),
            text=record.text,
            confidence=record.confidence,
            provider=record.source,
            bbox_page=record.bbox_page,
            polygons_page=(record.polygon_page,),
            run_id=RUN_ID,
            origin_execution_id=EXECUTION_ID,
            invocation_id=record.invocation_id,
            attempt_id=record.attempt_id,
            provider_family=record.provider_family,
            page_source_sha256=PAGE_SHA,
            root_input_pixel_sha256=ROOT_SHA,
            input_pixel_sha256=ROOT_SHA,
            payload_sha256=record.payload_sha256,
        )
        for component, record in zip(components, records)
    )
    inventory = tuple(
        build_component_inventory_entry(
            component_id=component.component_id,
            origin="discovery",
            introduced_by_decision_id=None,
            anchor_polygon_page=component.polygon_page,
            ordinal=index,
        )
        for index, component in enumerate(components)
    )
    entries = tuple(
        CoverageEntry(
            component_id=component.component_id,
            run_id=RUN_ID,
            origin_execution_id=EXECUTION_ID,
            page_id=PAGE_ID,
            page_source_sha256=PAGE_SHA,
            bbox_page=component.bbox_page,
            polygon_page=component.polygon_page,
            materiality="material",
            container_id=containers[index],
            ocr_attempt_ids=(attempt.attempt_id,),
            observation_ids=(observations[index].observation_id,),
            semantic_role=semantic_role,
            state=(
                "explicit_non_dialogue_preserve"
                if preserve_policy is not None
                else "observed"
            ),
            preserve_policy=preserve_policy,
        )
        for index, component in enumerate(components)
    )
    ledger = build_page_coverage_ledger(
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=PAGE_SHA,
        inventory_version=1,
        parent_ledger_sha256=None,
        component_inventory=inventory,
        expected_observation_ids=tuple(item.observation_id for item in observations),
        entries=entries,
    )
    return PageCoverageResult._build(
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=PAGE_SHA,
        ledger_history=(ledger,),
        components=components,
        observations=observations,
        ocr_requests=(request,),
        ocr_invocations=(invocation,),
        recovery_requests=(),
        recovery_decisions=(),
    )


def _replace_entries(
    coverage: PageCoverageResult,
    entries: tuple[CoverageEntry, ...],
    *,
    observations=None,
) -> PageCoverageResult:
    effective_observations = (
        coverage.observations if observations is None else tuple(observations)
    )
    ledger = build_page_coverage_ledger(
        run_id=coverage.run_id,
        origin_execution_id=coverage.origin_execution_id,
        page_id=coverage.page_id,
        page_source_sha256=coverage.page_source_sha256,
        inventory_version=1,
        parent_ledger_sha256=None,
        component_inventory=coverage.ledger.component_inventory,
        expected_observation_ids=tuple(
            item.observation_id for item in effective_observations
        ),
        entries=entries,
    )
    return PageCoverageResult._build(
        run_id=coverage.run_id,
        origin_execution_id=coverage.origin_execution_id,
        page_id=coverage.page_id,
        page_source_sha256=coverage.page_source_sha256,
        ledger_history=(ledger,),
        components=coverage.components,
        observations=effective_observations,
        ocr_requests=coverage.ocr_requests,
        ocr_invocations=coverage.ocr_invocations,
        recovery_requests=(),
        recovery_decisions=(),
    )


def _replace_single_component_with_lines(
    coverage: PageCoverageResult,
    lines: tuple[tuple[str, tuple[int, int, int, int]], ...],
) -> PageCoverageResult:
    invocation = coverage.ocr_invocations[0]
    record_template = invocation.observations[0]
    observation_template = coverage.observations[0]
    records = tuple(
        replace(
            record_template,
            observation_id=f"line-record-{index}",
            text=text,
            payload_sha256=sha256_text(text),
            bbox_page=bbox,
            polygon_page=(
                (bbox[0], bbox[1]),
                (bbox[2], bbox[1]),
                (bbox[2], bbox[3]),
                (bbox[0], bbox[3]),
            ),
        )
        for index, (text, bbox) in enumerate(lines, 1)
    )
    observations = tuple(
        replace(
            observation_template,
            observation_id=record.observation_id,
            text=record.text,
            payload_sha256=record.payload_sha256,
            bbox_page=record.bbox_page,
            polygons_page=(record.polygon_page,),
        )
        for record in records
    )
    rebuilt_invocation = OCRInvocationResult.build(
        request=invocation.request,
        observations=records,
        full_page_lines=records,
        attempts=invocation.attempts,
        diagnostics=invocation.diagnostics,
    )
    entry = replace(
        coverage.entries[0],
        observation_ids=tuple(item.observation_id for item in observations),
    )
    ledger = build_page_coverage_ledger(
        run_id=coverage.run_id,
        origin_execution_id=coverage.origin_execution_id,
        page_id=coverage.page_id,
        page_source_sha256=coverage.page_source_sha256,
        inventory_version=1,
        parent_ledger_sha256=None,
        component_inventory=coverage.ledger.component_inventory,
        expected_observation_ids=tuple(item.observation_id for item in observations),
        entries=(entry,),
    )
    return PageCoverageResult._build(
        run_id=coverage.run_id,
        origin_execution_id=coverage.origin_execution_id,
        page_id=coverage.page_id,
        page_source_sha256=coverage.page_source_sha256,
        ledger_history=(ledger,),
        components=coverage.components,
        observations=observations,
        ocr_requests=coverage.ocr_requests,
        ocr_invocations=(rebuilt_invocation,),
        recovery_requests=(),
        recovery_decisions=(),
    )


def test_zero_ocr_attempts_can_never_produce_no_ocr_non_text_suppression() -> None:
    component = _component("component-a", (20, 20, 120, 50))
    result = PageCoverageResult.initialize(
        run_id=RUN_ID,
        origin_execution_id=EXECUTION_ID,
        page_id=PAGE_ID,
        page_source_sha256=PAGE_SHA,
        components=(component,),
    )

    with pytest.raises(CoverageInvariantError, match="OCR attempt"):
        build_owner_page_graph_from_coverage(result)


def test_fragments_in_one_balloon_produce_one_owner_with_one_complete_body() -> None:
    coverage = _coverage(
        (
            ("component-a", "HELLO", (20, 20, 120, 45)),
            ("component-b", "WORLD", (20, 50, 120, 75)),
        ),
        containers=("container-one", "container-one"),
    )

    graph = build_owner_page_graph_from_coverage(coverage)

    assert len(graph.owners) == 1
    assert graph.owners[0].component_ids == ["component-a", "component-b"]
    assert graph.owners[0].source_payload == "HELLO WORLD"


def test_multiple_lines_from_one_invocation_remain_one_complete_owner_body() -> None:
    coverage = _coverage(
        (("component-a", "placeholder", (20, 20, 240, 90)),),
        containers=("container-one",),
    )
    coverage = _replace_single_component_with_lines(
        coverage,
        (
            ("UAU, VOCE ME", (35, 30, 210, 50)),
            ("ASSUSTOU.", (55, 55, 185, 78)),
        ),
    )

    graph = build_owner_page_graph_from_coverage(coverage)

    assert len(graph.owners) == 1
    assert graph.owners[0].selected_observation_ids == [
        "line-record-1",
        "line-record-2",
    ]
    assert graph.owners[0].source_payload == "UAU, VOCE ME ASSUSTOU."


def test_adjacent_containers_remain_separate_owners() -> None:
    coverage = _coverage(
        (
            ("component-a", "FIRST", (20, 20, 120, 45)),
            ("component-b", "SECOND", (140, 20, 240, 45)),
        ),
        containers=("container-left", "container-right"),
    )

    graph = build_owner_page_graph_from_coverage(coverage)

    assert len(graph.owners) == 2
    assert {owner.source_payload for owner in graph.owners} == {"FIRST", "SECOND"}


def test_short_english_word_inside_balloon_defaults_to_dialogue_not_sfx() -> None:
    graph = build_owner_page_graph_from_coverage(
        _coverage(
            (("component-a", "WAIT", (20, 20, 120, 50)),),
            containers=("container-dialogue",),
        )
    )

    assert graph.owners[0].semantic_role == "dialogue_body"
    assert graph.component_dispositions[0].decision == "owned"


def test_credit_preserve_policy_roundtrip_keeps_audit_fields() -> None:
    coverage = _coverage(
        (("component-credit", "SITE.COM", (20, 250, 180, 275)),),
        containers=("container-credit",),
        preserve_policy="policy:explicit_credit_outside_translatable_container",
        semantic_role="external_credit",
    )
    graph = build_owner_page_graph_from_coverage(coverage)
    disposition = graph.component_dispositions[0]

    assert disposition.decision == "preserve"
    assert disposition.policy_id == "explicit_credit_outside_translatable_container"
    assert disposition.policy_bbox_page
    assert disposition.policy_evidence_ids
    assert OwnerGraph.from_dict(graph.to_dict(), enforce=True).component_dispositions[0] == disposition


def test_negative_ocr_preserve_requires_explicit_visual_non_text_evidence() -> None:
    coverage = _coverage(
        (("component-art", "NOISE", (20, 20, 120, 50)),),
        containers=(None,),
        preserve_policy="policy:explicit_visual_non_text",
        semantic_role="visual_non_text",
    )
    entry = replace(
        coverage.entries[0],
        observation_ids=(),
        protection_evidence_ids=("visual-evidence-1",),
    )

    graph = build_owner_page_graph_from_coverage(
        _replace_entries(coverage, (entry,), observations=())
    )

    assert graph.component_dispositions[0].decision == "preserve"
    assert graph.component_dispositions[0].policy_id == "explicit_visual_non_text"


def test_ocr_empty_near_uniform_false_glyph_policy_roundtrips_as_preserve() -> None:
    coverage = _coverage(
        (("component-border", "", (20, 20, 55, 55)),),
        containers=(None,),
        preserve_policy="policy:ocr_empty_near_uniform_false_glyph",
        semantic_role="visual_non_text",
    )
    entry = replace(
        coverage.entries[0],
        materiality="non_text",
        observation_ids=(),
    )

    graph = build_owner_page_graph_from_coverage(
        _replace_entries(coverage, (entry,), observations=())
    )

    disposition = graph.component_dispositions[0]
    assert graph.owners == []
    assert disposition.decision == "preserve"
    assert disposition.policy_id == "ocr_empty_near_uniform_false_glyph"
    assert disposition.policy_evidence_ids == entry.ocr_attempt_ids


def test_ocr_empty_tiny_isolated_false_glyph_policy_roundtrips_as_preserve() -> None:
    coverage = _coverage(
        (("component-tiny-art", "", (20, 20, 39, 39)),),
        containers=(None,),
        preserve_policy="policy:ocr_empty_tiny_isolated_false_glyph",
        semantic_role="visual_non_text",
    )
    entry = replace(
        coverage.entries[0],
        materiality="non_text",
        observation_ids=(),
    )

    graph = build_owner_page_graph_from_coverage(
        _replace_entries(coverage, (entry,), observations=())
    )

    disposition = graph.component_dispositions[0]
    assert graph.owners == []
    assert disposition.decision == "preserve"
    assert disposition.policy_id == "ocr_empty_tiny_isolated_false_glyph"
    assert disposition.policy_evidence_ids == entry.ocr_attempt_ids


def test_unknown_preserve_policy_is_rejected() -> None:
    coverage = _coverage(
        (("component-a", "TEXT", (20, 20, 120, 50)),),
        containers=(None,),
        preserve_policy="policy:skip_for_convenience",
        semantic_role="external_credit",
    )

    with pytest.raises(CoverageInvariantError, match="unsupported preserve policy"):
        build_owner_page_graph_from_coverage(coverage)


def test_credit_policy_cannot_share_a_translatable_container() -> None:
    coverage = _coverage(
        (
            ("component-dialogue", "HELLO", (20, 20, 120, 45)),
            ("component-credit", "SITE.COM", (20, 50, 120, 75)),
        ),
        containers=("container-one", "container-one"),
    )
    entries = (
        coverage.entries[0],
        replace(
            coverage.entries[1],
            semantic_role="external_credit",
            state="explicit_non_dialogue_preserve",
            preserve_policy="policy:explicit_credit_outside_translatable_container",
        ),
    )

    with pytest.raises(CoverageInvariantError, match="translatable container"):
        build_owner_page_graph_from_coverage(_replace_entries(coverage, entries))


def test_graph_never_loses_a_material_component() -> None:
    coverage = _coverage(
        (
            ("component-a", "FIRST", (20, 20, 120, 45)),
            ("component-b", "SECOND", (20, 50, 120, 75)),
        ),
        containers=("container-one", "container-one"),
    )
    graph = build_owner_page_graph_from_coverage(coverage)

    assert {item.component_id for item in graph.component_dispositions} == set(
        coverage.ledger.expected_component_ids
    )
