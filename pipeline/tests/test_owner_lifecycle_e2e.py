from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ownership.hash_contract import canonical_page_sha256, sha256_text
from ownership.model import FinalQAProbe, LanguageResidualIssue
from ownership.ocr_contract import (
    OCRAttempt,
    OCRDiagnostics,
    OCRInvocationResult,
    OCRRequest,
    OCRTransformOperation,
    OCRTransformSpec,
)
from strip.page_pipeline import PageExecutionResult, PagePipelineIdentityError, run_page_owner_pipeline
from test_page_owner_pipeline import _request, _services


def test_systemic_recipe_corpus_covers_visual_and_ownership_dimensions():
    fixture = Path(__file__).parent / "fixtures" / "english_owner_recovery"
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    recipes = json.loads((fixture / "recipes.json").read_text(encoding="utf-8"))["recipes"]
    selected = [recipes[item["recipe_id"]] for item in manifest["cases"]]
    glyphs = [glyph for recipe in selected for glyph in recipe["glyph_layers"]]

    assert len(manifest["cases"]) == 12
    assert {recipe["canvas"]["background"] for recipe in selected} >= {"light", "dark", "gradient"}
    assert {glyph.get("fill") for glyph in glyphs if glyph.get("fill")} >= {"solid", "gradient"}
    assert {glyph.get("outline") for glyph in glyphs if "outline" in glyph} == {False, True}
    assert any(glyph.get("anti_alias") is True for glyph in glyphs)
    assert {glyph.get("line_count") for glyph in glyphs if glyph.get("line_count")} >= {1, 2, 3}
    assert {recipe.get("band_mode") for recipe in selected if recipe.get("band_mode")} >= {"absent", "cross_band"}
    assert {recipe["language"].get("input") for recipe in selected if recipe["language"].get("input")} >= {
        "source", "target", "mixed"
    }


def test_no_band_dialogue_recipe_completes_basic_page_first_lifecycle():
    request = _request(bands=())

    result = run_page_owner_pipeline(request, _services())

    assert result.coverage.entries
    assert result.owner_graph.read().owners[0].source_payload
    assert result.translations[0].target_locale == "pt-BR"
    assert result.status == "candidate_ready"
    assert result.final_page is None


def _detached_execution_rejection_services(*, mutate_candidate=None):
    services = _services()

    def execute(request, graph, translation):
        owner = graph.owners[0]
        binding = next(
            item for item in translation.bindings if item.owner_id == owner.owner_id
        )
        component_by_id = {
            item.component_id: item for item in graph.components
        }
        dispositions = []
        for disposition in graph.component_dispositions:
            if disposition.component_id not in owner.component_ids:
                dispositions.append(disposition)
                continue
            component = component_by_id[disposition.component_id]
            evidence_ids = tuple(
                sorted(
                    observation.observation_id
                    for observation in graph.observations
                    if disposition.component_id in observation.component_ids
                )
            )
            dispositions.append(
                replace(
                    disposition,
                    decision="uncertain",
                    owner_id=None,
                    reason="owner_execution_rejected",
                    policy_id="coverage_ambiguous_candidate",
                    policy_bbox_page=component.bbox_page,
                    policy_evidence_ids=evidence_ids,
                    policy_reason=(
                        "source pixels preserved after owner execution rejection"
                    ),
                )
            )
        graph.owners = []
        graph.projections = []
        graph.component_dispositions = dispositions
        candidate = {
            "id": owner.owner_id,
            "owner_id": None,
            "candidate_owner_id": owner.owner_id,
            "page_id": request.page_id,
            "component_ids": list(owner.component_ids),
            "observation_ids": list(owner.observation_ids),
            "selected_observation_ids": list(owner.selected_observation_ids),
            "semantic_role": owner.semantic_role,
            "route_action": "review_required",
            "action_mask_ref": None,
            "layout_region_ids": [],
            "disposition": "review",
            "state": "review_required",
            "execution_tile_id": None,
            "owner_execution_rejection_reason": "unsafe source replacement",
            "owner_graph_run_id": graph.run_id,
            "owner_graph_origin_execution_id": graph.origin_execution_id,
            "owner_graph_page_source_sha256": graph.page_source_sha256,
            "execution_rejected": True,
            "derived_qa_status": "review_required",
            "write_authority": "revoked",
            "source_pixels_preserved": True,
            "committed": False,
            "blocking": True,
            "qa_action": "BLOCK",
            "visible": False,
            "render_policy": "review_required",
            "text": binding.source_text,
            "original": binding.source_text,
            "source_payload": binding.source_text,
            "translated": binding.target_text,
            "translated_payload": binding.target_text,
            "translation_binding_sha256": binding.translation_binding_sha256,
            "source_payload_sha256": binding.source_payload_sha256,
            "target_payload_sha256": binding.target_payload_sha256,
            "qa_flags": ["owner_render_geometry_review"],
        }
        if mutate_candidate is not None:
            mutate_candidate(candidate)
        return SimpleNamespace(commits=(), records=(candidate,), graph=graph)

    return replace(services, execution_fn=execute)


def test_detached_execution_rejection_candidate_preserves_translation_binding():
    result = run_page_owner_pipeline(
        _request(bands=()),
        _detached_execution_rejection_services(),
    )

    assert result.status == "candidate_ready"
    assert result.owner_graph.read().owners == []
    layer = result.text_layers_view.read()["texts"][0]
    assert layer["owner_id"] is None
    assert layer["candidate_owner_id"] == result.translations[0].owner_id
    assert layer["source_payload"] == result.translations[0].source_text
    assert layer["translated"] == result.translations[0].target_text
    assert layer["translation_binding_sha256"] == result.translations[0].translation_binding_sha256


def test_detached_execution_rejection_candidate_cannot_retain_write_authority():
    with pytest.raises(PagePipelineIdentityError, match="detached review candidate is invalid"):
        run_page_owner_pipeline(
            _request(bands=()),
            _detached_execution_rejection_services(
                mutate_candidate=lambda candidate: candidate.update(visible=True)
            ),
        )


def test_detached_execution_rejection_candidate_must_match_translation_hashes():
    with pytest.raises(PagePipelineIdentityError, match="mismatched target_payload_sha256"):
        run_page_owner_pipeline(
            _request(bands=()),
            _detached_execution_rejection_services(
                mutate_candidate=lambda candidate: candidate.update(
                    target_payload_sha256="0" * 64
                )
            ),
        )


def test_detached_execution_rejection_candidate_must_match_source_hash():
    with pytest.raises(
        PagePipelineIdentityError,
        match="mismatched source_payload_sha256",
    ):
        run_page_owner_pipeline(
            _request(bands=()),
            _detached_execution_rejection_services(
                mutate_candidate=lambda candidate: candidate.update(
                    source_payload_sha256="0" * 64
                )
            ),
        )


def _qa_journal(candidate):
    pixels = candidate.request.original_page.mutable_attempt_copy()
    root_sha = canonical_page_sha256(pixels)
    request = OCRRequest(
        run_id=candidate.request.run_id,
        origin_execution_id=candidate.request.execution_id,
        page_id=candidate.page_id,
        page_source_sha256=candidate.request.page_source_sha256,
        root_input_pixel_sha256=root_sha,
        invocation_id="final-qa-invocation-1",
        provider_family="paddle-final",
    )
    transform = OCRTransformSpec.build((OCRTransformOperation(kind="identity"),))
    attempt = OCRAttempt(
        attempt_id="final-qa-attempt-1",
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        root_input_pixel_sha256=root_sha,
        invocation_id=request.invocation_id,
        provider_family=request.provider_family,
        variant_id="full_page",
        input_pixel_sha256=root_sha,
        parent_input_pixel_sha256=root_sha,
        input_bbox_page=None,
        input_kind="full_page",
        transform_spec=transform,
        input_width=pixels.shape[1],
        input_height=pixels.shape[0],
        input_mode="RGB",
        provider_called=True,
        cache_hit=False,
    )
    invocation = OCRInvocationResult.build(
        request=request,
        attempts=(attempt,),
        diagnostics=OCRDiagnostics("paddle-final"),
    )
    issue = LanguageResidualIssue.build(
        issue_id="issue-final-1",
        run_id=request.run_id,
        execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        page_output_pixel_sha256=root_sha,
        owner_id=candidate.translations[0].owner_id,
        component_id=candidate.translations[0].component_ids[0],
        container_id="container-final",
        invocation_ids=(request.invocation_id,),
        kind="source_language_visible",
        observed_text="SOURCE",
        observed_text_sha256=sha256_text("SOURCE"),
        source_binding_sha256=candidate.translations[0].translation_binding_sha256,
        region_sha256="e" * 64,
        source_only_tokens=("source",),
        repair_required=True,
    )
    probe = FinalQAProbe.build(
        probe_id="probe-final-1",
        run_id=request.run_id,
        execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        input_origin="redecoded_persisted_candidate",
        candidate_file_sha256="d" * 64,
        candidate_pixel_sha256=root_sha,
        root_input_pixel_sha256=root_sha,
        ocr_invocation_id=request.invocation_id,
        fresh_ocr_attempt_ids=(attempt.attempt_id,),
        fresh_ocr_attempt_chain_sha256=invocation.attempt_chain_sha256,
        ocr_payload_sha256=sha256_text("SOURCE"),
        observer_available=True,
        coverage_complete=True,
        issue_ids=(issue.issue_id,),
    )
    return request, invocation, issue, probe


def test_final_qa_journal_closes_request_invocation_probe_and_issue():
    candidate = run_page_owner_pipeline(_request(bands=()), _services())
    request, invocation, issue, probe = _qa_journal(candidate)

    result = PageExecutionResult.build_from(
        candidate,
        final_qa_ocr_requests=(request,),
        final_qa_ocr_invocations=(invocation,),
        language_residual_issues=(issue,),
        qa_probes=(probe,),
    )

    assert result.final_qa_ocr_requests == (request,)
    assert result.final_qa_ocr_invocations == (invocation,)
    assert result.qa_probes[0].fresh_ocr_attempt_ids == ("final-qa-attempt-1",)
    assert result.to_canonical_dict()["language_residual_issues"][0]["issue_id"] == issue.issue_id


def test_final_result_rejects_orphan_final_qa_issue_invocation():
    candidate = run_page_owner_pipeline(_request(bands=()), _services())
    request, invocation, issue, probe = _qa_journal(candidate)
    payload = issue.to_dict()
    payload.pop("issue_sha256")
    payload["invocation_ids"] = ("missing-invocation",)
    orphan = LanguageResidualIssue.build(**payload)

    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build_from(
            candidate,
            final_qa_ocr_requests=(request,),
            final_qa_ocr_invocations=(invocation,),
            language_residual_issues=(orphan,),
            qa_probes=(probe,),
        )
