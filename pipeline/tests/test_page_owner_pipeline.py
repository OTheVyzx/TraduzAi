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
    _material_post_cleanup_residual_mask,
    _terminal_source_support_mask,
    run_page_owner_pipeline,
)
from strip.types import Band
from strip.process_bands import _run_translate_stage


def _page() -> np.ndarray:
    return np.full((90, 140, 3), 245, dtype=np.uint8)


def test_terminal_replacement_verdict_uses_measured_post_cleanup_residual_mask() -> None:
    original = _page()
    cleanup = original.copy()
    action = np.zeros(original.shape[:2], dtype=np.uint8)
    action[20:60, 20:120] = 255
    cleanup[35:45, 55:85] = 20

    residual = _material_post_cleanup_residual_mask(original, cleanup, action)

    assert np.count_nonzero(residual[35:45, 55:85]) > 0


def test_terminal_replacement_verdict_ignores_submaterial_detector_noise() -> None:
    original = _page()
    cleanup = original.copy()
    action = np.zeros(original.shape[:2], dtype=np.uint8)
    action[20:60, 20:120] = 255
    cleanup[35, 55] = 20

    residual = _material_post_cleanup_residual_mask(original, cleanup, action)

    assert np.count_nonzero(residual) == 0


def test_terminal_replacement_verdict_projects_framed_cleanup_to_logical_page() -> None:
    from strip.page_surface_geometry import PageSurfaceGeometry

    original = _page()
    logical_cleanup = original.copy()
    logical_cleanup[35:45, 55:85] = 20
    action = np.zeros(original.shape[:2], dtype=np.uint8)
    action[20:60, 20:120] = 255
    geometry = PageSurfaceGeometry.build(
        logical_width=140,
        logical_height=90,
        frame_width=200,
        frame_height=90,
        content_origin_xy=(30, 0),
    )
    framed_cleanup = geometry.logical_array_to_frame(logical_cleanup, fill_value=255)

    residual = _material_post_cleanup_residual_mask(
        original,
        framed_cleanup,
        action,
        page_surface_geometry=geometry,
    )

    assert residual.shape == action.shape
    assert np.count_nonzero(residual[35:45, 55:85]) > 0


def test_terminal_verdict_prefers_authenticated_glyph_support_over_inpaint_area() -> None:
    action = np.zeros((90, 140), dtype=np.uint8)
    action[20:60, 20:120] = 255
    source_support = np.zeros_like(action)
    source_support[35:45, 55:85] = 255
    mutation = SimpleNamespace(
        action_mask=action,
        source_support_mask=source_support,
    )

    measured = _terminal_source_support_mask(mutation)

    assert np.array_equal(measured, source_support)
    assert np.count_nonzero(measured) < np.count_nonzero(action)


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


def test_page_pipeline_passes_fresh_complete_language_evidence(monkeypatch):
    import strip.page_pipeline as page_pipeline

    captured = {}
    original = page_pipeline.translate_owner_page

    def recording_translate_owner_page(*args, **kwargs):
        captured.update(kwargs["page_language_evidence_by_owner"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        page_pipeline,
        "translate_owner_page",
        recording_translate_owner_page,
    )

    result = run_page_owner_pipeline(_request(), _services())

    owner_id = result.translations[0].owner_id
    assert captured[owner_id].coverage_complete
    assert "this" in captured[owner_id].source_only_tokens


def test_page_pipeline_requires_repaint_for_already_target_ocr_pixels(monkeypatch):
    import strip.page_pipeline as page_pipeline

    captured = {}
    original = page_pipeline.translate_owner_page

    def recording_translate_owner_page(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        page_pipeline,
        "translate_owner_page",
        recording_translate_owner_page,
    )

    run_page_owner_pipeline(_request(), _services())

    assert captured["repaint_already_target_pixels"] is True


def test_page_pipeline_preserves_rejected_translation_as_review_and_continues():
    def unchanged(owner_request, _variant):
        return owner_request.source_text

    unchanged.backend_name = "fixture"
    result = run_page_owner_pipeline(
        _request(),
        replace(_services(), translation_backends=(unchanged,)),
    )

    assert result.translations == ()
    assert result.translation_attempts
    assert all(
        attempt.status == "rejected"
        for attempt in result.translation_attempts
    )
    assert result.owner_graph.read().owners == []
    assert result.page_commits == ()
    review = result.text_layers_view.read()["texts"][0]
    assert review["state"] == "review_required"
    assert review["route_action"] == "review_required"
    assert review["visible"] is False
    assert review["translated"] == ""
    assert review["translation_attempt_ids"]
    assert "unchanged_source_dialogue" in review[
        "owner_execution_rejection_reason"
    ]


def test_execution_record_adapter_converts_nested_tuples_but_not_invalid_sets() -> None:
    from ownership.execution import FrozenJSONSnapshot
    from strip.page_pipeline import _json_compatible_execution_value

    source = {
        "texts": [
            {
                "owner_style_capture": {
                    "style_evidence_v2": {
                        "attribute_provenance": {
                            "fill": {"masks": ("owner_glyph_core",)},
                        }
                    }
                }
            }
        ]
    }

    converted = _json_compatible_execution_value(source)

    assert converted["texts"][0]["owner_style_capture"]["style_evidence_v2"][
        "attribute_provenance"
    ]["fill"]["masks"] == ["owner_glyph_core"]
    assert source["texts"][0]["owner_style_capture"]["style_evidence_v2"][
        "attribute_provenance"
    ]["fill"]["masks"] == ("owner_glyph_core",)
    assert FrozenJSONSnapshot.build(converted).read() == converted
    with pytest.raises(TypeError, match="unsupported canonical JSON value"):
        FrozenJSONSnapshot.build(
            _json_compatible_execution_value({"texts": [{"invalid": {"set"}}]})
        )


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


def test_page_pipeline_persists_execution_repair_journal():
    from ownership.repair import (
        OwnerRepairCase,
        RepairExecutionFeedback,
        run_repair_ladder,
    )

    def execute(request, graph, translation_result):
        support = np.zeros((request.original_page.height, request.original_page.width), dtype=np.uint8)
        support[28:34, 36:84] = 255
        interior = np.zeros_like(support)
        interior[20:48, 22:112] = 255
        empty = np.zeros_like(support)
        case = OwnerRepairCase.build(
            original_rgb=request.original_page.mutable_attempt_copy(),
            translation=translation_result.bindings[0],
            execution_id=request.execution_id,
            source_support_mask=support,
            container_interior_mask=interior,
            container_border_mask=empty,
            protected_art_mask=empty,
            positive_residual_mask=empty,
            attempt_executor=lambda original_rgb, **_kwargs: RepairExecutionFeedback.committed(
                original_rgb
            ),
        )
        ladder = run_repair_ladder(
            case,
            max_strategy="R0",
            scheduler=lambda _seconds: None,
        )
        return SimpleNamespace(
            graph=graph,
            commits=(),
            records=(
                {
                    "owner_id": translation_result.bindings[0].owner_id,
                    "translated": translation_result.bindings[0].target_text,
                    "state": "repair_pending",
                },
            ),
            target_materializations=(),
            repair_requests=ladder.repair_requests,
            repair_history=ladder.attempts,
            repair_budget_policy_sha256=ladder.repair_budget_policy_sha256,
        )

    result = run_page_owner_pipeline(
        _request(),
        replace(_services(), execution_fn=execute),
    )

    assert result.repair_requests
    assert result.repair_history
    assert result.repair_budget_policy_sha256 == (
        result.repair_history[0].repair_budget_policy_sha256
    )


def test_page_result_selects_post_execution_owner_graph():
    from strip.page_pipeline import _owner_graph_after_execution

    request = _request()
    coverage = _ready_coverage(request)
    translated_graph = _services().graph_fn(coverage)
    executed_graph = copy.deepcopy(translated_graph)
    executed_graph.owners[0].state = "rendered"

    selected = _owner_graph_after_execution(
        translated_graph,
        SimpleNamespace(graph=executed_graph),
    )

    assert selected.owners[0].state == "rendered"
    assert translated_graph.owners[0].state != "rendered"


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


def test_runner_uses_full_deterministic_owner_translation_retry_budget() -> None:
    from strip.run import _build_owner_translation_attempt_controls
    from translator.translate import TranslationAttemptControl

    controls = _build_owner_translation_attempt_controls(
        TranslationAttemptControl,
        ollama_model="qwen2.5:7b",
    )

    assert [control.backend for control in controls] == [
        "google",
        "google",
        "ollama",
        "ollama",
        "ocr_recovery",
    ]
    assert [control.variant for control in controls] == [
        "owner_primary",
        "owner_contextual",
        "owner_configured",
        "owner_contextual",
        "owner_ocr_recovery",
    ]
    assert [control.disable_cache for control in controls] == [False, True, False, True, True]
    assert [control.provider_model for control in controls] == [
        None,
        None,
        "qwen2.5:7b",
        "qwen2.5:7b",
        "qwen2.5:7b",
    ]
