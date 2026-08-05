"""Deterministic, identity-bound owner repair ladder."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Callable

import numpy as np

from .hash_contract import (
    canonical_json_sha256,
    canonical_page_sha256,
    sha256_bytes,
)
from .model import OwnerRepairRequest, RepairAttempt


class RepairStrategy(str, Enum):
    R0_PRECISE_GLYPH = "R0"
    R1_EXPANDED_SUPPORT = "R1"
    R2_TEXT_REGION_REBUILD = "R2"
    R3_CONTAINER_INTERIOR_REBUILD = "R3"


class DuplicateRepairAttemptError(ValueError):
    """Raised before an identical repair attempt can run twice."""


class RepairInfrastructureExhausted(RuntimeError):
    """Raised when transient infrastructure failures consume their budget."""

    def __init__(self, attempt_count: int):
        super().__init__(f"repair infrastructure exhausted after {attempt_count} attempts")
        self.attempt_count = int(attempt_count)


class RepairPolicyIdentityError(ValueError):
    """Raised when history is linked to another repair budget policy."""


@dataclass(frozen=True)
class OwnerRepairCase:
    original_rgb: np.ndarray
    translation: Any
    execution_id: str
    source_support_mask: np.ndarray
    container_interior_mask: np.ndarray
    container_border_mask: np.ndarray
    protected_art_mask: np.ndarray
    positive_residual_mask: np.ndarray
    attempt_executor: Callable[..., Any] | None = None

    @classmethod
    def build(cls, **values: Any) -> "OwnerRepairCase":
        original = np.asarray(values["original_rgb"])
        if original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3:
            raise ValueError("repair original must be HxWx3 uint8 RGB")
        shape = original.shape[:2]

        def mask(name: str, *, required: bool = True) -> np.ndarray:
            value = values.get(name)
            if value is None and not required:
                result = np.zeros(shape, dtype=np.uint8)
            else:
                result = np.asarray(value)
                if result.ndim == 3:
                    result = result[:, :, 0]
                if result.shape != shape:
                    raise ValueError(f"repair {name} geometry mismatch")
                result = np.where(result > 0, 255, 0).astype(np.uint8)
            result = np.ascontiguousarray(result)
            result.setflags(write=False)
            return result

        frozen_original = np.ascontiguousarray(original).copy()
        frozen_original.setflags(write=False)
        translation = values["translation"]
        if not str(values.get("execution_id") or ""):
            raise ValueError("repair execution identity is missing")
        if not getattr(translation, "translation_binding_sha256", None):
            raise ValueError("repair case requires a translation binding")
        return cls(
            original_rgb=frozen_original,
            translation=translation,
            execution_id=str(values["execution_id"]),
            source_support_mask=mask("source_support_mask"),
            container_interior_mask=mask("container_interior_mask"),
            container_border_mask=mask("container_border_mask", required=False),
            protected_art_mask=mask("protected_art_mask", required=False),
            positive_residual_mask=mask("positive_residual_mask", required=False),
            attempt_executor=values.get("attempt_executor"),
        )


@dataclass(frozen=True)
class RepairExecutionFeedback:
    status: str
    reason: str
    final_page: np.ndarray | None
    evidence_ids: tuple[str, ...]

    @classmethod
    def visual_residual(cls, evidence_id: str) -> "RepairExecutionFeedback":
        return cls("visual_residual", "positive_source_residual", None, (str(evidence_id),))

    @classmethod
    def committed(cls, final_page: Any) -> "RepairExecutionFeedback":
        pixels = np.ascontiguousarray(np.asarray(final_page, dtype=np.uint8)).copy()
        pixels.setflags(write=False)
        return cls("committed", "", pixels, ())

    @classmethod
    def transient(cls, reason: str) -> "RepairExecutionFeedback":
        return cls("transient", str(reason), None, ())


@dataclass(frozen=True)
class RepairBudgetPolicy:
    max_attempts_by_variant: tuple[tuple[str, int], ...] = (
        ("R0:precise", 1),
        ("R1:expanded", 2),
        ("R2:text_region", 2),
        ("R3:contextual", 2),
        ("R3:deterministic_interior_fill", 1),
        ("R3:deterministic_support_local_fill", 1),
    )
    max_geometry_expansions: int = 3
    max_transient_retries: int = 2
    transient_backoff_ms: tuple[int, ...] = (250, 500)
    policy_sha256: str = ""

    def __post_init__(self) -> None:
        payload = {
            "max_attempts_by_variant": [list(item) for item in self.max_attempts_by_variant],
            "max_geometry_expansions": int(self.max_geometry_expansions),
            "max_transient_retries": int(self.max_transient_retries),
            "transient_backoff_ms": list(self.transient_backoff_ms),
        }
        expected = canonical_json_sha256(payload)
        if self.policy_sha256 and self.policy_sha256 != expected:
            raise RepairPolicyIdentityError("repair policy hash mismatch")
        if self.max_geometry_expansions < 0 or self.max_transient_retries < 1:
            raise ValueError("repair policy budget is invalid")
        object.__setattr__(self, "policy_sha256", expected)

    @classmethod
    def default(cls) -> "RepairBudgetPolicy":
        return cls()


@dataclass(frozen=True)
class RepairAttemptRuntime:
    record: RepairAttempt
    cleanup_mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.ascontiguousarray(np.asarray(self.cleanup_mask, dtype=np.uint8)).copy()
        mask.setflags(write=False)
        object.__setattr__(self, "cleanup_mask", mask)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.record, name)


@dataclass(frozen=True)
class RepairLadderResult:
    status: str
    repair_requests: tuple[OwnerRepairRequest, ...]
    attempts: tuple[RepairAttempt, ...]
    repair_budget_policy_sha256: str
    final_page: np.ndarray | None
    next_strategy: str | None

    def __post_init__(self) -> None:
        if self.final_page is not None:
            pixels = np.ascontiguousarray(np.asarray(self.final_page, dtype=np.uint8)).copy()
            pixels.setflags(write=False)
            object.__setattr__(self, "final_page", pixels)


def _strategy(value: RepairStrategy | str) -> RepairStrategy:
    return value if isinstance(value, RepairStrategy) else RepairStrategy(str(value).upper())


def _variant_for(strategy: RepairStrategy) -> str:
    return {
        RepairStrategy.R0_PRECISE_GLYPH: "precise",
        RepairStrategy.R1_EXPANDED_SUPPORT: "expanded",
        RepairStrategy.R2_TEXT_REGION_REBUILD: "text_region",
        RepairStrategy.R3_CONTAINER_INTERIOR_REBUILD: "contextual",
    }[strategy]


def build_repair_attempt(
    case: OwnerRepairCase,
    *,
    request: OwnerRepairRequest,
    policy: RepairBudgetPolicy,
    strategy: RepairStrategy | str,
    variant: str | None = None,
    previous: RepairAttempt | None = None,
) -> RepairAttemptRuntime:
    """Build one immutable repair record and its runtime-only cleanup mask."""

    from inpainter.owner_mask import build_repair_cleanup_mask

    selected = _strategy(strategy)
    if selected not in {
        RepairStrategy.R0_PRECISE_GLYPH,
        RepairStrategy.R1_EXPANDED_SUPPORT,
    }:
        raise ValueError("Task 11 only materializes R0 and R1")
    binding = case.translation
    OwnerRepairRequest.from_dict(request.to_dict())
    expected_identity = (
        binding.run_id,
        case.execution_id,
        binding.page_id,
        binding.page_source_sha256,
        binding.owner_id,
    )
    actual_identity = (
        request.run_id,
        request.execution_id,
        request.page_id,
        request.page_source_sha256,
        request.owner_id,
    )
    if actual_identity != expected_identity:
        raise RepairPolicyIdentityError("repair request belongs to another owner execution")
    cleanup_mask = build_repair_cleanup_mask(
        case.source_support_mask,
        container_interior_mask=case.container_interior_mask,
        container_border_mask=case.container_border_mask,
        protected_art_mask=case.protected_art_mask,
        positive_residual_mask=(
            case.positive_residual_mask
            if selected is RepairStrategy.R1_EXPANDED_SUPPORT else None
        ),
        strategy=selected.value,
    )
    chosen_variant = str(variant or _variant_for(selected))
    input_sha = canonical_page_sha256(case.original_rgb)
    cleanup_sha = sha256_bytes(np.ascontiguousarray(cleanup_mask).tobytes())
    protected_sha = sha256_bytes(np.ascontiguousarray(case.protected_art_mask).tobytes())
    ys, xs = np.nonzero(cleanup_mask)
    geometry = {
        "shape": list(cleanup_mask.shape),
        "bbox": [
            int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        ],
        "positive_pixels": int(xs.size),
        "previous_attempt_sha256": getattr(previous, "attempt_sha256", None),
    }
    geometry_sha = canonical_json_sha256(geometry)
    fingerprint = canonical_json_sha256(
        {
            "request_sha256": request.request_sha256,
            "strategy": selected.value,
            "variant": chosen_variant,
            "geometry_sha256": geometry_sha,
            "input_sha256": input_sha,
        }
    )
    record = RepairAttempt.build(
        run_id=binding.run_id,
        execution_id=case.execution_id,
        page_id=binding.page_id,
        page_source_sha256=binding.page_source_sha256,
        attempt_id=f"repair-{fingerprint[:32]}",
        request_id=request.request_id,
        issue_id=request.issue_id,
        consumed_request_ids=(request.request_id,),
        translation_binding_sha256=binding.translation_binding_sha256,
        repair_budget_policy_sha256=policy.policy_sha256,
        strategy=selected.value,
        variant=chosen_variant,
        owner_id=binding.owner_id,
        input_sha256=input_sha,
        cleanup_mask_sha256=cleanup_sha,
        protected_mask_sha256=protected_sha,
        geometry_sha256=geometry_sha,
        attempt_fingerprint=fingerprint,
        outcome="pending",
        evidence_ids=request.evidence_ids,
    )
    return RepairAttemptRuntime(record=record, cleanup_mask=cleanup_mask)


def _with_outcome(
    runtime: RepairAttemptRuntime,
    *,
    outcome: str,
    evidence_ids: tuple[str, ...] = (),
) -> RepairAttempt:
    payload = runtime.record.to_dict()
    payload.pop("attempt_sha256")
    payload["outcome"] = str(outcome)
    payload["evidence_ids"] = tuple(
        sorted({*runtime.record.evidence_ids, *(str(item) for item in evidence_ids)})
    )
    return RepairAttempt.build(**payload)


class RepairController:
    def __init__(self, *, policy: RepairBudgetPolicy):
        self.policy = policy
        self._fingerprints: set[str] = set()

    def record(self, attempt: RepairAttempt) -> None:
        if attempt.repair_budget_policy_sha256 != self.policy.policy_sha256:
            raise RepairPolicyIdentityError("repair attempt policy hash mismatch")
        if attempt.attempt_fingerprint in self._fingerprints:
            raise DuplicateRepairAttemptError(attempt.attempt_fingerprint)
        self._fingerprints.add(attempt.attempt_fingerprint)


def _request_for(
    case: OwnerRepairCase,
    *,
    failed_stage: str,
    reason: str,
    evidence_ids: tuple[str, ...],
    next_strategy: str,
    issue_id: str | None = None,
) -> OwnerRepairRequest:
    binding = case.translation
    request_id = canonical_json_sha256(
        {
            "run_id": binding.run_id,
            "execution_id": case.execution_id,
            "page_id": binding.page_id,
            "page_source_sha256": binding.page_source_sha256,
            "owner_id": binding.owner_id,
            "failed_stage": failed_stage,
            "reason": reason,
            "evidence_ids": list(evidence_ids),
            "next_strategy": next_strategy,
            "issue_id": issue_id,
        }
    )
    return OwnerRepairRequest.build(
        request_id=request_id,
        issue_id=issue_id,
        run_id=binding.run_id,
        execution_id=case.execution_id,
        page_id=binding.page_id,
        page_source_sha256=binding.page_source_sha256,
        owner_id=binding.owner_id,
        original_page_sha256=canonical_page_sha256(case.original_rgb),
        failed_stage=failed_stage,
        reason=reason,
        evidence_ids=evidence_ids,
        next_strategy=next_strategy,
    )


def run_repair_ladder(
    case: OwnerRepairCase,
    *,
    policy: RepairBudgetPolicy | None = None,
    max_strategy: RepairStrategy | str = RepairStrategy.R1_EXPANDED_SUPPORT,
    scheduler: Callable[[float], None] = time.sleep,
) -> RepairLadderResult:
    """Run R0/R1 from the immutable original; never expose rollback as final."""

    selected_policy = policy or RepairBudgetPolicy.default()
    maximum = _strategy(max_strategy)
    if maximum not in {
        RepairStrategy.R0_PRECISE_GLYPH,
        RepairStrategy.R1_EXPANDED_SUPPORT,
    }:
        raise ValueError("Task 11 ladder supports max_strategy R0 or R1")
    controller = RepairController(policy=selected_policy)
    initial = _request_for(
        case,
        failed_stage="pre_execution",
        reason="initial_atomic_replacement",
        evidence_ids=("source-support",),
        next_strategy="R0",
    )
    requests = [initial]
    attempts: list[RepairAttempt] = []
    request = initial
    previous: RepairAttempt | None = None
    strategies = [RepairStrategy.R0_PRECISE_GLYPH]
    if maximum is RepairStrategy.R1_EXPANDED_SUPPORT:
        strategies.append(RepairStrategy.R1_EXPANDED_SUPPORT)
    transient_count = 0
    for selected in strategies:
        while True:
            variant = _variant_for(selected)
            if transient_count:
                variant = f"{variant}_transient_{transient_count + 1}"
            runtime = build_repair_attempt(
                case,
                request=request,
                policy=selected_policy,
                strategy=selected,
                variant=variant,
                previous=previous,
            )
            controller.record(runtime.record)
            if case.attempt_executor is None:
                feedback = RepairExecutionFeedback.visual_residual(
                    f"{selected.value.lower()}-unverified"
                )
            else:
                raw_feedback = case.attempt_executor(
                    strategy=selected.value,
                    variant=variant,
                    original_rgb=np.array(case.original_rgb, copy=True),
                    cleanup_mask=np.array(runtime.cleanup_mask, copy=True),
                    request=request,
                    translation=case.translation,
                )
                if not isinstance(raw_feedback, RepairExecutionFeedback):
                    raise TypeError("repair executor returned an invalid feedback object")
                feedback = raw_feedback
            if feedback.status == "transient":
                transient_count += 1
                attempts.append(
                    _with_outcome(runtime, outcome="transient_failure", evidence_ids=feedback.evidence_ids)
                )
                if transient_count >= selected_policy.max_transient_retries:
                    raise RepairInfrastructureExhausted(transient_count)
                retry_request = _request_for(
                    case,
                    failed_stage="infrastructure",
                    reason=feedback.reason,
                    evidence_ids=(f"transient-{transient_count}",),
                    next_strategy=selected.value,
                )
                requests.append(retry_request)
                request = retry_request
                delay_index = min(transient_count - 1, len(selected_policy.transient_backoff_ms) - 1)
                scheduler(selected_policy.transient_backoff_ms[delay_index] / 1000.0)
                continue
            transient_count = 0
            if feedback.status == "committed":
                attempts.append(_with_outcome(runtime, outcome="committed"))
                return RepairLadderResult(
                    status="committed",
                    repair_requests=tuple(requests),
                    attempts=tuple(attempts),
                    repair_budget_policy_sha256=selected_policy.policy_sha256,
                    final_page=feedback.final_page,
                    next_strategy=None,
                )
            if feedback.status != "visual_residual":
                raise ValueError("repair feedback status is unsupported")
            previous = _with_outcome(
                runtime, outcome="visual_residual", evidence_ids=feedback.evidence_ids
            )
            attempts.append(previous)
            next_strategy = "R1" if selected is RepairStrategy.R0_PRECISE_GLYPH else "R2"
            request = _request_for(
                case,
                failed_stage="residual",
                reason=feedback.reason,
                evidence_ids=feedback.evidence_ids,
                next_strategy=next_strategy,
            )
            requests.append(request)
            break
    return RepairLadderResult(
        status="repair_pending",
        repair_requests=tuple(requests),
        attempts=tuple(attempts),
        repair_budget_policy_sha256=selected_policy.policy_sha256,
        final_page=None,
        next_strategy=("R1" if maximum is RepairStrategy.R0_PRECISE_GLYPH else "R2"),
    )


__all__ = [
    "DuplicateRepairAttemptError",
    "OwnerRepairCase",
    "RepairAttemptRuntime",
    "RepairBudgetPolicy",
    "RepairController",
    "RepairExecutionFeedback",
    "RepairInfrastructureExhausted",
    "RepairLadderResult",
    "RepairPolicyIdentityError",
    "RepairStrategy",
    "build_repair_attempt",
    "run_repair_ladder",
]
