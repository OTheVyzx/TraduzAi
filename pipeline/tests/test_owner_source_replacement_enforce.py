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


def test_rendered_owner_record_binding_fields_preserve_translation_identity():
    from strip.process_bands import _owner_translation_binding_fields

    binding = _binding()

    assert _owner_translation_binding_fields(binding) == {
        "translation_binding_sha256": binding.translation_binding_sha256,
        "source_payload_sha256": binding.source_payload_sha256,
        "target_payload_sha256": binding.target_payload_sha256,
    }


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


def test_verified_already_ptbr_owner_preserves_original_pixels_without_visual_stages():
    from ownership.translation import (
        OwnerTranslationRequest,
        apply_owner_translation_result,
        translate_owner_page,
    )
    from strip.process_bands import execute_owner_page_graph
    from test_final_pixel_qa import _graph
    from translator.language_policy import build_page_language_evidence

    source = "COMO ESPERADO DE KIM SIHYEOK!"
    graph = _graph(source=source, translated=None, state="execution_planned")
    graph.owners[0].route_action = "translate_inpaint_render"
    request = OwnerTranslationRequest.from_graph(graph, graph.owners[0].owner_id)

    def unchanged(owner_request, _variant):
        return owner_request.source_text

    unchanged.backend_name = "fixture"
    translation_result = translate_owner_page(
        (request,),
        backends=(unchanged,),
        page_language_evidence_by_owner={
            request.owner_id: build_page_language_evidence(
                texts=(source,),
                coverage_complete=True,
            )
        },
    )
    translated_graph = apply_owner_translation_result(graph, translation_result)

    class MustNotRun:
        def __getattr__(self, name):
            raise AssertionError(f"visual stage must not run: {name}")

    page = np.full((24, 40, 3), 230, dtype=np.uint8)
    execution = execute_owner_page_graph(
        page,
        translated_graph,
        translator=object(),
        inpainter=MustNotRun(),
        typesetter=MustNotRun(),
        translation_result_override=translation_result,
    )

    assert execution.commits == ()
    assert execution.target_materializations == ()
    assert execution.graph.owners[0].state == "target_ready"
    assert execution.records[0]["render_policy"] == "preserve_original"
    assert execution.records[0]["preserve_original"] is True
    assert execution.records[0]["no_repaint_policy_id"] == "already_target_language"


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


def test_candidate_excludes_verified_no_repaint_binding_from_commit_cardinality():
    from ownership.hash_contract import canonical_json_sha256, sha256_text
    from ownership.translation import TranslationBinding
    from translator.language_policy import (
        build_page_language_evidence,
        validate_target_language,
    )

    result = _candidate_result_with_one_bound_owner()
    rendered_binding = result.translations[0]
    source = "COMO ESPERADO DE KIM SIHYEOK!"
    verdict = validate_target_language(
        source=source,
        target=source,
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=(source,), coverage_complete=True
        ),
    )
    provisional = TranslationBinding(
        run_id=rendered_binding.run_id,
        origin_execution_id=rendered_binding.origin_execution_id,
        page_id=rendered_binding.page_id,
        page_source_sha256=rendered_binding.page_source_sha256,
        owner_id="owner-preserved-ptbr",
        component_ids=("component-preserved-ptbr",),
        source_payload_sha256=sha256_text(source),
        target_payload_sha256=sha256_text(source),
        source_text=source,
        target_text=source,
        target_locale="pt-BR",
        language_verdict=verdict,
        attempt_ids=(),
        translation_binding_sha256="",
    )
    preserved_binding = replace(
        provisional,
        translation_binding_sha256=canonical_json_sha256(
            provisional.canonical_payload()
        ),
    )
    layers = result.text_layers_view.read()["texts"] + [
        {
            "owner_id": preserved_binding.owner_id,
            "translation_binding_sha256": preserved_binding.translation_binding_sha256,
            "target_payload_sha256": preserved_binding.target_payload_sha256,
            "translated": preserved_binding.target_text,
            "render_policy": "preserve_original",
            "preserve_original": True,
        }
    ]

    mixed = PageExecutionResult.build_from(
        result,
        translations=(rendered_binding, preserved_binding),
        text_layers_view=FrozenJSONSnapshot.build({"texts": layers}),
    )

    assert len(mixed.translations) == 2
    assert len(mixed.page_commits) == 1
    assert mixed.translations[1].preserves_original_pixels

    repair_layers = [
        {
            "owner_id": rendered_binding.owner_id,
            "translated": rendered_binding.target_text,
            "state": "repair_pending",
        },
        layers[1],
    ]
    repair_pending = PageExecutionResult.build_from(
        result,
        translations=(rendered_binding, preserved_binding),
        page_commits=(),
        owner_target_materializations=(),
        text_layers_view=FrozenJSONSnapshot.build({"texts": repair_layers}),
        page_composition=None,
    )

    assert repair_pending.page_commits == ()
    assert repair_pending.text_layers_view.read()["texts"][0]["state"] == "repair_pending"


def test_candidate_allows_review_binding_beside_one_materialized_owner():
    from ownership.hash_contract import canonical_json_sha256

    result = _candidate_result_with_one_bound_owner()
    rendered_binding = result.translations[0]
    provisional_review = replace(
        rendered_binding,
        owner_id="owner-review-required",
        component_ids=("component-review-required",),
        translation_binding_sha256="",
    )
    reviewed_binding = replace(
        provisional_review,
        translation_binding_sha256=canonical_json_sha256(
            provisional_review.canonical_payload()
        ),
    )
    layers = result.text_layers_view.read()["texts"] + [
        {
            "owner_id": reviewed_binding.owner_id,
            "translation_binding_sha256": reviewed_binding.translation_binding_sha256,
            "target_payload_sha256": reviewed_binding.target_payload_sha256,
            "translated": reviewed_binding.target_text,
            "state": "review_required",
            "route_action": "review_required",
            "visible": False,
        }
    ]

    mixed = PageExecutionResult.build_from(
        result,
        translations=(rendered_binding, reviewed_binding),
        text_layers_view=FrozenJSONSnapshot.build({"texts": layers}),
    )

    assert len(mixed.translations) == 2
    assert len(mixed.page_commits) == 1
    assert len(mixed.owner_target_materializations) == 1
    assert mixed.text_layers_view.read()["texts"][1]["state"] == "review_required"
