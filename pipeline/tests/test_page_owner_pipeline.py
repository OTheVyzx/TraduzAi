from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ownership.coverage import (
    CoverageEntry,
    PageCoverageResult,
    build_component_inventory_entry,
    build_page_coverage_ledger,
)
from ownership.hash_contract import canonical_page_sha256, sha256_text
from ownership.model import SourceTextComponent, TextObservation
from ownership.ocr_contract import (
    OCRAttempt,
    OCRDiagnostics,
    OCRInvocationResult,
    OCRObservationRecord,
    OCRRequest,
    OCRTransformOperation,
    OCRTransformSpec,
)
from strip.page_pipeline import (
    OriginalPageSnapshot,
    PagePipelineRequest,
    PagePipelineServices,
    PagePipelineIdentityError,
    PagePipelineStateError,
    run_page_owner_pipeline,
)
from strip.types import Band
from strip.process_bands import _run_translate_stage


def _page() -> np.ndarray:
    return np.full((90, 140, 3), 245, dtype=np.uint8)


def _ready_coverage(request: PagePipelineRequest) -> PageCoverageResult:
    component = SourceTextComponent(
        component_id="component-dialogue",
        page_id=request.page_id,
        bbox_page=(20, 20, 120, 60),
        polygon_page=((20, 20), (120, 20), (120, 60), (20, 60)),
        detector_sources=("fixture",),
    )
    ocr_request = OCRRequest(
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        root_input_pixel_sha256=request.page_source_sha256,
        invocation_id="invocation-page-first",
        provider_family="fixture",
    )
    transform = OCRTransformSpec.build((OCRTransformOperation(kind="identity"),))
    attempt = OCRAttempt(
        attempt_id="attempt-page-first",
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        root_input_pixel_sha256=request.page_source_sha256,
        invocation_id=ocr_request.invocation_id,
        provider_family=ocr_request.provider_family,
        variant_id="full_page",
        input_pixel_sha256=request.page_source_sha256,
        parent_input_pixel_sha256=request.page_source_sha256,
        input_bbox_page=None,
        input_kind="full_page",
        transform_spec=transform,
        input_width=140,
        input_height=90,
        input_mode="RGB",
        provider_called=True,
        cache_hit=False,
    )
    raw = OCRObservationRecord(
        observation_id="observation-page-first",
        attempt_id=attempt.attempt_id,
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        root_input_pixel_sha256=request.page_source_sha256,
        input_pixel_sha256=request.page_source_sha256,
        invocation_id=ocr_request.invocation_id,
        provider_family=ocr_request.provider_family,
        variant_id=attempt.variant_id,
        payload_sha256=sha256_text("LOOK AT THIS GUY"),
        text="LOOK AT THIS GUY",
        confidence=0.98,
        bbox_page=component.bbox_page,
        polygon_page=component.polygon_page,
        source="fixture",
    )
    invocation = OCRInvocationResult.build(
        request=ocr_request,
        observations=(raw,),
        full_page_lines=(raw,),
        attempts=(attempt,),
        diagnostics=OCRDiagnostics("fixture"),
    )
    observation = TextObservation(
        observation_id=raw.observation_id,
        page_id=request.page_id,
        component_ids=(component.component_id,),
        text=raw.text,
        confidence=raw.confidence,
        provider=raw.source,
        bbox_page=raw.bbox_page,
        polygons_page=(raw.polygon_page,),
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        invocation_id=raw.invocation_id,
        attempt_id=raw.attempt_id,
        provider_family=raw.provider_family,
        page_source_sha256=request.page_source_sha256,
        root_input_pixel_sha256=request.page_source_sha256,
        input_pixel_sha256=request.page_source_sha256,
        payload_sha256=raw.payload_sha256,
    )
    inventory = (
        build_component_inventory_entry(
            component_id=component.component_id,
            origin="discovery",
            introduced_by_decision_id=None,
            anchor_polygon_page=component.polygon_page,
            ordinal=0,
        ),
    )
    entry = CoverageEntry(
        component_id=component.component_id,
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        bbox_page=component.bbox_page,
        polygon_page=component.polygon_page,
        materiality="material",
        container_id="container-dialogue",
        ocr_attempt_ids=(attempt.attempt_id,),
        observation_ids=(observation.observation_id,),
        semantic_role="dialogue_body",
        state="observed",
    )
    ledger = build_page_coverage_ledger(
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        inventory_version=1,
        parent_ledger_sha256=None,
        component_inventory=inventory,
        expected_observation_ids=(observation.observation_id,),
        entries=(entry,),
    )
    return PageCoverageResult._build(
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        ledger_history=(ledger,),
        components=(component,),
        observations=(observation,),
        ocr_requests=(ocr_request,),
        ocr_invocations=(invocation,),
        recovery_requests=(),
        recovery_decisions=(),
    )


def _request(*, bands=()) -> PagePipelineRequest:
    original = OriginalPageSnapshot.from_pixels(_page(), source_file_sha256="f" * 64)
    return PagePipelineRequest.from_legacy_bands(
        original,
        bands,
        run_id="run-page-first",
        execution_id="execution-page-first",
        page_id="page_001",
    )


def _services() -> PagePipelineServices:
    def backend(_request, _variant):
        return "OLHE PARA ESSE CARA"

    backend.backend_name = "fixture"
    return PagePipelineServices(coverage_fn=_ready_coverage, translation_backends=(backend,))


def test_enforce_mode_processes_page_without_creating_band_placeholder():
    result = run_page_owner_pipeline(_request(), _services())

    assert result.page_id == "page_001"
    assert result.coverage.entries
    assert result.owner_graph.read().owners
    assert result.translations[0].target_locale == "pt-BR"
    assert result.status == "candidate_ready"
    assert result.final_page is None


def test_stale_mutable_band_payloads_are_not_imported_or_mutated_in_enforce():
    band = Band(y_top=10, y_bottom=80, tile_id="band-legacy")
    band.ocr_result = {"texts": [{"translated": "STALE"}]}
    band.cleaned_slice = np.zeros((70, 140, 3), dtype=np.uint8)
    band.rendered_slice = np.ones((70, 140, 3), dtype=np.uint8)
    before = copy.deepcopy(band)

    request = _request(bands=(band,))
    result = run_page_owner_pipeline(request, _services())

    assert result.translations[0].target_text == "OLHE PARA ESSE CARA"
    assert band.ocr_result == before.ocr_result
    assert np.array_equal(band.cleaned_slice, before.cleaned_slice)
    assert np.array_equal(band.rendered_slice, before.rendered_slice)
    assert not hasattr(request.band_projections[0], "ocr_result")


def test_adapter_cannot_mutate_canonical_original_snapshot_between_attempts():
    request = _request()
    first = request.original_page.mutable_attempt_copy()
    first[:] = 0
    second = request.original_page.mutable_attempt_copy()

    assert np.array_equal(second, _page())
    assert request.original_page.page_source_sha256 == canonical_page_sha256(_page())
    with pytest.raises(ValueError):
        request.original_page.read_only_rgb()[0, 0] = 0


def test_band_partitioning_is_not_owner_authority():
    one_band = (Band(y_top=0, y_bottom=90, tile_id="whole"),)
    two_bands = (
        Band(y_top=0, y_bottom=45, tile_id="top"),
        Band(y_top=45, y_bottom=90, tile_id="bottom"),
    )

    first = run_page_owner_pipeline(_request(bands=one_band), _services())
    second = run_page_owner_pipeline(_request(bands=two_bands), _services())

    assert first.owner_graph.sha256 == second.owner_graph.sha256
    assert [binding.target_text for binding in first.translations] == [
        binding.target_text for binding in second.translations
    ]


def test_page_execution_result_carries_direct_page_commits():
    def execute(request, _graph, _translation_result):
        return (
            SimpleNamespace(
                commit_id="commit-page-001",
                run_id=request.run_id,
                execution_id=request.execution_id,
                page_id=request.page_id,
                page_source_sha256=request.page_source_sha256,
            ),
        )

    services = replace(_services(), execution_fn=execute)
    result = run_page_owner_pipeline(_request(), services)

    assert result.page_commits[0].page_id == result.page_id
    assert not hasattr(result, "commits_by_band")


def test_page_result_rejects_graph_binding_or_commit_from_another_identity():
    result = run_page_owner_pipeline(_request(), _services())
    parts = dict(
        request=result.request,
        coverage=result.coverage,
        owner_graph=result.owner_graph,
        translation_attempts=result.translation_attempts,
        translations=result.translations,
    )

    with pytest.raises(PagePipelineIdentityError, match="owner graph"):
        type(result).build(**(parts | {"owner_graph": replace(result.owner_graph, run_id="run-other")}))
    with pytest.raises(PagePipelineIdentityError, match="translation"):
        type(result).build(
            **(parts | {"translations": (replace(result.translations[0], page_id="page_002"),)})
        )
    with pytest.raises(PagePipelineIdentityError, match="execution_id"):
        type(result).build(
            **parts,
            page_commits=(
                SimpleNamespace(
                    commit_id="commit-other",
                    run_id=result.request.run_id,
                    execution_id="execution-other",
                    page_id=result.page_id,
                    page_source_sha256=result.request.page_source_sha256,
                ),
            ),
        )


def test_final_verified_status_requires_final_page_and_terminal_proof():
    result = run_page_owner_pipeline(_request(), _services())
    parts = dict(
        request=result.request,
        coverage=result.coverage,
        owner_graph=result.owner_graph,
        translation_attempts=result.translation_attempts,
        translations=result.translations,
    )

    with pytest.raises(PagePipelineStateError):
        type(result).build(**parts, status="final_verified", final_page=object(), terminal_proof=None)


def test_candidate_evidence_reserves_final_schema_sections_without_raster_bytes():
    result = run_page_owner_pipeline(_request(), _services())
    payload = result.to_canonical_dict()

    assert payload["repair_requests"] == []
    assert payload["owner_target_materializations"] == []
    assert payload["final_qa_ocr_requests"] == []
    assert payload["final_qa_ocr_invocations"] == []
    assert payload["language_residual_issues"] == []
    assert payload["qa_probes"] == []
    assert payload["final_replacement_verdicts"] == []
    assert payload["final_page"] is None
    assert payload["terminal_proof"] is None
    serialized = str(payload)
    assert "lossless_png_bytes" not in serialized
    assert "ndarray" not in serialized


def test_real_owner_translate_stage_uses_attempt_boundary_not_legacy_batch():
    from ownership.hash_contract import canonical_json_bytes, sha256_bytes
    from translator.translate import (
        FrozenTranslationOCRResult,
        TranslationAttemptControl,
        TranslationProviderAttemptResult,
    )

    candidate = run_page_owner_pipeline(_request(), _services())
    graph = candidate.owner_graph.read()
    for owner in graph.owners:
        owner.state = "owned"
        owner.translated_payload = None

    def attempt(ocr_result, **kwargs):
        owner = ocr_result["texts"][0]
        translated = {
            "page_id": ocr_result["page_id"],
            "texts": [{**owner, "translated": "OLHE PARA ESSE CARA"}],
        }
        metadata = {
            "backend": kwargs["control"].backend,
            "provider_called": True,
            "cache_hit": False,
        }
        encoded = canonical_json_bytes(metadata)
        return TranslationProviderAttemptResult(
            translated_items=(FrozenTranslationOCRResult.build(owner["owner_id"], translated),),
            backend=kwargs["control"].backend,
            variant=kwargs["control"].variant,
            provider_model=None,
            provider_called=True,
            cache_hit=False,
            provider_metadata_json_bytes=encoded,
            provider_metadata_sha256=sha256_bytes(encoded),
        )

    translator = SimpleNamespace(
        TranslationAttemptControl=TranslationAttemptControl,
        translate_one_owner_attempt=attempt,
        translate_pages=lambda *_args, **_kwargs: pytest.fail("legacy batch was called"),
    )

    stage = _run_translate_stage({}, translator=translator, owner_graph=graph)
    payload = stage.to_page_dict()

    assert payload["texts"][0]["translated"] == "OLHE PARA ESSE CARA"
    assert payload["texts"][0]["translation_binding_sha256"]
    assert payload["_owner_graph_snapshot"]["owners"][0]["state"] == "target_ready"
