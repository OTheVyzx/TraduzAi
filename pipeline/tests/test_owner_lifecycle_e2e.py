from __future__ import annotations

import json
from pathlib import Path

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
