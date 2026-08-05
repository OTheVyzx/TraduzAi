from __future__ import annotations

import numpy as np
import pytest
import cv2

from ownership.hash_contract import canonical_page_sha256
from ownership.execution import bind_repair_ladder_to_page_result
from ownership.model import OwnerRepairRequest
from ownership.repair import (
    DuplicateRepairAttemptError,
    OwnerRepairCase,
    RepairBudgetPolicy,
    RepairController,
    RepairExecutionFeedback,
    RepairInfrastructureExhausted,
    RepairPolicyIdentityError,
    RepairStrategy,
    build_repair_attempt,
    run_repair_ladder,
)
from test_owner_source_replacement_enforce import _binding
from test_page_owner_pipeline import _request as _page_request, _services
from strip.page_pipeline import PageExecutionResult, run_page_owner_pipeline
from qa.inpaint_residual import detect_residual_text


def _masks():
    support = np.zeros((48, 80), dtype=np.uint8)
    support[18:21, 20:48] = 255
    support[27:30, 24:54] = 255
    interior = np.zeros_like(support)
    interior[5:43, 7:73] = 255
    border = np.zeros_like(support)
    border[5:7, 7:73] = 255
    border[41:43, 7:73] = 255
    border[5:43, 7:9] = 255
    border[5:43, 71:73] = 255
    protected = np.zeros_like(support)
    protected[14:25, 58:67] = 255
    residual = np.zeros_like(support)
    residual[20:23, 48:52] = 255
    return support, interior, border, protected, residual


def _case(executor=None):
    support, interior, border, protected, residual = _masks()
    binding = _binding()
    original = np.full((48, 80, 3), 245, dtype=np.uint8)
    original[support > 0] = 20
    return OwnerRepairCase.build(
        original_rgb=original,
        translation=binding,
        execution_id="execution-repair",
        source_support_mask=support,
        container_interior_mask=interior,
        container_border_mask=border,
        protected_art_mask=protected,
        positive_residual_mask=residual,
        attempt_executor=executor,
    )


def _request(case, *, request_id="request-initial", issue_id=None):
    return OwnerRepairRequest.build(
        request_id=request_id,
        issue_id=issue_id,
        run_id=case.translation.run_id,
        execution_id=case.execution_id,
        page_id=case.translation.page_id,
        page_source_sha256=case.translation.page_source_sha256,
        owner_id=case.translation.owner_id,
        original_page_sha256=canonical_page_sha256(case.original_rgb),
        failed_stage="pre_execution",
        reason="initial_atomic_replacement",
        evidence_ids=("source-support",),
        next_strategy="R0",
    )


def test_r0_uses_precise_source_support_and_preserves_container_border():
    case = _case()
    attempt = build_repair_attempt(
        case,
        request=_request(case),
        policy=RepairBudgetPolicy.default(),
        strategy=RepairStrategy.R0_PRECISE_GLYPH,
    )
    support, _interior, border, protected, _residual = _masks()

    assert np.all(attempt.cleanup_mask[support > 0] > 0)
    assert not np.any((attempt.cleanup_mask > 0) & (border > 0))
    assert not np.any((attempt.cleanup_mask > 0) & (protected > 0))


def test_r1_expands_only_from_positive_source_support():
    case = _case()
    policy = RepairBudgetPolicy.default()
    r0 = build_repair_attempt(
        case, request=_request(case), policy=policy, strategy="R0"
    )
    residual_request = OwnerRepairRequest.build(
        request_id="request-residual",
        issue_id=None,
        run_id=case.translation.run_id,
        execution_id=case.execution_id,
        page_id=case.translation.page_id,
        page_source_sha256=case.translation.page_source_sha256,
        owner_id=case.translation.owner_id,
        original_page_sha256=canonical_page_sha256(case.original_rgb),
        failed_stage="residual",
        reason="positive_source_residual",
        evidence_ids=("residual",),
        next_strategy="R1",
    )
    r1 = build_repair_attempt(
        case,
        request=residual_request,
        policy=policy,
        strategy="R1",
        previous=r0.record,
    )
    _support, interior, border, protected, residual = _masks()

    assert np.count_nonzero(r1.cleanup_mask) > np.count_nonzero(r0.cleanup_mask)
    assert np.all(r1.cleanup_mask[residual > 0] > 0)
    assert not np.any((r1.cleanup_mask > 0) & (interior == 0))
    assert not np.any((r1.cleanup_mask > 0) & ((border > 0) | (protected > 0)))


def test_residual_probe_exposes_only_positive_pixels_for_r1():
    before = np.full((64, 120, 3), 245, dtype=np.uint8)
    cv2.putText(before, "EN", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (15, 15, 15), 2)
    region = np.zeros(before.shape[:2], dtype=np.uint8)
    region[12:52, 12:100] = 255

    result = detect_residual_text(
        before,
        before.copy(),
        region,
        include_unchanged_dark=True,
        return_mask=True,
    )

    positive = result["positive_residual_mask"]
    assert result["has_residual"] is True
    assert np.count_nonzero(positive) == result["dark_residual_pixels"]
    assert not np.any((positive > 0) & (region == 0))


def test_repair_attempt_consumes_request_and_policy_identity():
    case = _case()
    policy = RepairBudgetPolicy.default()
    request = _request(case, request_id="request-qa", issue_id="issue-qa")
    attempt = build_repair_attempt(case, request=request, policy=policy, strategy="R0")

    assert attempt.request_id == request.request_id
    assert attempt.issue_id == request.issue_id
    assert attempt.consumed_request_ids == (request.request_id,)
    assert attempt.repair_budget_policy_sha256 == policy.policy_sha256


def test_duplicate_attempt_fingerprint_is_rejected():
    case = _case()
    policy = RepairBudgetPolicy.default()
    attempt = build_repair_attempt(
        case, request=_request(case), policy=policy, strategy="R0"
    )
    controller = RepairController(policy=policy)
    controller.record(attempt.record)
    with pytest.raises(DuplicateRepairAttemptError):
        controller.record(attempt.record)


def test_r0_residual_advances_to_r1_from_original_pixels():
    calls = []

    def executor(*, strategy, original_rgb, cleanup_mask, **_kwargs):
        calls.append((strategy, canonical_page_sha256(original_rgb)))
        if strategy == "R0":
            return RepairExecutionFeedback.visual_residual("residual-r0")
        final = np.array(original_rgb, copy=True)
        final[cleanup_mask > 0] = 245
        return RepairExecutionFeedback.committed(final)

    case = _case(executor)
    result = run_repair_ladder(case, max_strategy="R1", scheduler=lambda _seconds: None)

    assert [attempt.strategy for attempt in result.attempts] == ["R0", "R1"]
    assert len({attempt.input_sha256 for attempt in result.attempts}) == 1
    assert len({value for _strategy, value in calls}) == 1
    assert result.status == "committed"


def test_r1_failure_keeps_final_page_empty_and_points_to_r2():
    case = _case(
        lambda **_kwargs: RepairExecutionFeedback.visual_residual("still-visible")
    )
    result = run_repair_ladder(case, max_strategy="R1", scheduler=lambda _seconds: None)

    assert result.status == "repair_pending"
    assert result.next_strategy == "R2"
    assert result.final_page is None


def test_transient_failure_exhausts_exact_budget_without_final_page():
    case = _case(lambda **_kwargs: RepairExecutionFeedback.transient("gpu-unavailable"))
    policy = RepairBudgetPolicy.default()
    with pytest.raises(RepairInfrastructureExhausted) as exc:
        run_repair_ladder(case, policy=policy, scheduler=lambda _seconds: None)

    assert exc.value.attempt_count == policy.max_transient_retries


def test_page_result_rejects_repair_policy_hash_swap():
    candidate = run_page_owner_pipeline(_page_request(), _services())
    original = candidate.request.original_page.read_only_rgb()
    support = np.zeros(original.shape[:2], dtype=np.uint8)
    support[30:34, 35:82] = 255
    interior = np.full(original.shape[:2], 255, dtype=np.uint8)
    empty = np.zeros(original.shape[:2], dtype=np.uint8)
    case = OwnerRepairCase.build(
        original_rgb=original,
        translation=candidate.translations[0],
        execution_id=candidate.request.execution_id,
        source_support_mask=support,
        container_interior_mask=interior,
        container_border_mask=empty,
        protected_art_mask=empty,
        positive_residual_mask=empty,
        attempt_executor=lambda original_rgb, **_kwargs: RepairExecutionFeedback.committed(
            original_rgb
        ),
    )
    ladder = run_repair_ladder(case, max_strategy="R0", scheduler=lambda _seconds: None)
    result = bind_repair_ladder_to_page_result(candidate, ladder)

    assert result.repair_history[0].repair_budget_policy_sha256 == (
        result.repair_budget_policy_sha256
    )
    with pytest.raises(RepairPolicyIdentityError):
        PageExecutionResult.build_from(result, repair_budget_policy_sha256="f" * 64)
