from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ownership.execution import (
    AtomicReplacementError,
    OwnerReplacementTransaction,
    PageCandidateTransaction,
    FrozenJSONSnapshot,
    PageCompositionSnapshot,
    canonical_glyph_patch_sha256,
    execute_owner_replacement,
)
from ownership.hash_contract import canonical_page_sha256
from ownership.model import OwnerGlyphPatch, OwnerTargetMaterialization
from ownership.translation import TranslationAttempt, bind_translation
from test_owner_atomic_execution import _glyph_patch, _mutation, _page
from test_owner_translation import _owner_request
from test_page_owner_pipeline import _request, _services
from strip.page_pipeline import PageExecutionResult, PagePipelineIdentityError, run_page_owner_pipeline
from translator.language_policy import validate_target_language


def _binding():
    request = _owner_request("a")
    target = "O JOGADOR TEM 10 ABATES"
    verdict = validate_target_language(
        source=request.source_text,
        target=target,
        role=request.semantic_role,
    )
    attempt = TranslationAttempt.build(
        request=request,
        backend="fixture",
        variant="primary",
        provider_model=None,
        provider_metadata={},
        target_text=target,
        provider_called=True,
        cache_hit=False,
        status="accepted",
        language_verdict=verdict,
    )
    return bind_translation(request, target, (attempt,))


def _transaction():
    original = _page()
    mutation = _mutation(original)
    return original, mutation, OwnerReplacementTransaction.build(
        original_rgb=original,
        mutation=mutation,
        translation=_binding(),
        execution_id="execution-owner-atomic",
    )


def test_cleanup_and_pt_br_render_commit_as_one_hash_chain():
    original, mutation, tx = _transaction()
    glyph_patch = _glyph_patch(
        mutation, OwnerGlyphPatch, render_completed=True, fit_status="ok"
    )

    committed = tx.commit(mutation, glyph_patch)

    assert committed.commit.before_sha256 == tx.original_pixel_sha256
    assert committed.commit.translation_binding_sha256 == tx.translation.translation_binding_sha256
    assert committed.commit.source_payload_sha256 == tx.translation.source_payload_sha256
    assert committed.commit.target_payload_sha256 == tx.translation.target_payload_sha256
    assert committed.commit.target_glyph_patch_sha256 == canonical_glyph_patch_sha256(glyph_patch)
    assert committed.final_pixel_sha256 == canonical_page_sha256(committed.final_page)


@pytest.mark.parametrize(
    "field",
    [
        "translation_binding_sha256",
        "source_payload_sha256",
        "target_payload_sha256",
        "target_glyph_patch_sha256",
    ],
)
def test_atomic_commit_rejects_stale_binding_or_target_hash(field):
    _original, mutation, tx = _transaction()
    glyph_patch = _glyph_patch(mutation, OwnerGlyphPatch, render_completed=True, fit_status="ok")
    with pytest.raises(AtomicReplacementError):
        tx.commit(mutation, glyph_patch, overrides={field: "0" * 64})


def test_cleanup_without_target_or_target_without_cleanup_cannot_commit():
    _original, mutation, tx = _transaction()
    glyph_patch = _glyph_patch(mutation, OwnerGlyphPatch, render_completed=True, fit_status="ok")
    with pytest.raises(AtomicReplacementError):
        tx.commit(mutation, None)
    with pytest.raises(AtomicReplacementError):
        tx.commit(None, glyph_patch)


def test_mask_failure_requests_repair_instead_of_finishing_with_source():
    _original, mutation, tx = _transaction()

    result = execute_owner_replacement(
        tx,
        mutation,
        None,
        failed_stage="mask",
        failure_reason="mask_failure",
        evidence_ids=("mask-empty",),
        next_strategy="r1_support_expansion",
    )

    assert result.status == "repair_pending"
    assert result.repair_request.reason == "mask_failure"
    assert result.outcome is None
    assert result.final_page is None


def test_canonical_target_ready_owner_can_build_operational_mask():
    from inpainter.owner_mask import OwnerMaskEvidence, build_owner_mask_plan
    from test_owner_mask import _single_component_owner

    image = np.full((24, 32, 3), 220, dtype=np.uint8)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[6:14, 8:18] = 255
    owner = _single_component_owner(owner_id="owner-target-ready")
    owner.state = "target_ready"

    plan = build_owner_mask_plan(
        image,
        owner,
        (
            OwnerMaskEvidence(
                evidence_id="target-ready-glyph",
                component_id="cmp_body_top",
                glyph_mask=glyph,
            ),
        ),
        owner_component_bboxes_page={"cmp_body_top": (7, 5, 20, 16)},
    )

    assert plan.coverage_complete
    assert np.count_nonzero(plan.action_mask) > 0


def test_replacing_owner_commit_recomposes_from_original_pixels():
    original, mutation, tx = _transaction()
    glyph_patch = _glyph_patch(mutation, OwnerGlyphPatch, render_completed=True, fit_status="ok")
    outcome = tx.commit(mutation, glyph_patch)
    composition = PageCandidateTransaction.from_original(
        original,
        commits=(outcome.commit,),
    ).compose()

    assert composition.base_pixel_sha256 == canonical_page_sha256(original)
    assert composition.commit_ids == (outcome.commit.commit_id,)
    assert np.array_equal(composition.final_page, outcome.final_page)


def _candidate_result_with_one_bound_owner():
    candidate = run_page_owner_pipeline(_request(), _services())
    binding = candidate.translations[0]
    commit = SimpleNamespace(
        commit_id="commit-bound-owner",
        run_id=candidate.request.run_id,
        execution_id=candidate.request.execution_id,
        page_id=candidate.page_id,
        page_source_sha256=candidate.request.page_source_sha256,
        owner_id=binding.owner_id,
        before_sha256=candidate.request.page_source_sha256,
        after_sha256="a" * 64,
        translation_binding_sha256=binding.translation_binding_sha256,
        source_payload_sha256=binding.source_payload_sha256,
        target_payload_sha256=binding.target_payload_sha256,
        target_glyph_patch_sha256="b" * 64,
    )
    materialization = OwnerTargetMaterialization.build(
        materialization_id="materialization-bound-owner",
        run_id=candidate.request.run_id,
        execution_id=candidate.request.execution_id,
        page_id=candidate.page_id,
        page_source_sha256=candidate.request.page_source_sha256,
        owner_id=binding.owner_id,
        translation_binding_sha256=binding.translation_binding_sha256,
        source_payload_sha256=binding.source_payload_sha256,
        target_payload_sha256=binding.target_payload_sha256,
        target_glyph_patch_sha256=commit.target_glyph_patch_sha256,
        glyph_mask_sha256="c" * 64,
        base_pixel_sha256=candidate.request.page_source_sha256,
        result_pixel_sha256=commit.after_sha256,
    )
    text_layers = FrozenJSONSnapshot.build(
        {
            "texts": [
                {
                    "owner_id": binding.owner_id,
                    "translation_binding_sha256": binding.translation_binding_sha256,
                    "target_payload_sha256": binding.target_payload_sha256,
                    "translated": binding.target_text,
                }
            ]
        }
    )
    composition = PageCompositionSnapshot.build(
        run_id=candidate.request.run_id,
        execution_id=candidate.request.execution_id,
        page_id=candidate.page_id,
        page_source_sha256=candidate.request.page_source_sha256,
        base_pixel_sha256=candidate.request.page_source_sha256,
        final_pixel_sha256=commit.after_sha256,
        commits=(commit,),
        materializations=(materialization,),
    )
    return PageExecutionResult.build_from(
        candidate,
        page_commits=(commit,),
        owner_target_materializations=(materialization,),
        text_layers_view=text_layers,
        page_composition=composition,
    )


def test_candidate_requires_exact_binding_commit_materialization_composition_chain():
    result = _candidate_result_with_one_bound_owner()
    binding = result.translations[0]
    commit = result.page_commits[0]
    materialization = result.owner_target_materializations[0]

    assert binding.translation_binding_sha256 == commit.translation_binding_sha256
    assert commit.target_payload_sha256 == materialization.target_payload_sha256
    assert result.page_composition.target_materialization_sha256s == (
        materialization.materialization_sha256,
    )
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build_from(
            result,
            owner_target_materializations=(
                replace(materialization, target_payload_sha256="d" * 64),
            ),
        )
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build_from(
            result,
            text_layers_view=FrozenJSONSnapshot.build(
                {
                    "texts": [
                        {
                            "owner_id": binding.owner_id,
                            "translation_binding_sha256": binding.translation_binding_sha256,
                            "target_payload_sha256": binding.source_payload_sha256,
                            "translated": binding.source_text,
                        }
                    ]
                }
            ),
        )
