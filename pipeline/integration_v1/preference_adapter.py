"""Integration adapter from immutable renderer receipts to Studio A/B comparisons."""
from __future__ import annotations

from typing import Any, Mapping

from integration_v1.contracts import adapt_contract_payload
from typesetter.preference_contract import PreferenceCandidate, PreferenceComparison


def candidate_from_recipe_receipt(
    receipt: Mapping[str, Any],
    *,
    owner_id: str,
    target_text: str,
    style_sha256: str,
    layout_plan_sha256: str,
    preview_ref: Mapping[str, Any],
    context_ref: Mapping[str, Any],
    metrics: Mapping[str, Any],
    hard_safety_passed: bool,
) -> PreferenceCandidate:
    """Validate a Recipe receipt and preserve its renderer-owned identities."""

    validated = adapt_contract_payload("Recipe", receipt)
    dependency_layout = validated["dependency_hashes"].get("layout_plan")
    if dependency_layout is not None and dependency_layout != layout_plan_sha256:
        raise ValueError("layout plan hash diverges from the recipe dependency")
    from typesetter.raster_safety import validate_raster_safety_evidence

    safety = validate_raster_safety_evidence(metrics.get("raster_safety"))
    dependency_safety = validated["dependency_hashes"].get("raster_safety")
    if dependency_safety is not None and dependency_safety != safety["evidence_sha256"]:
        raise ValueError("raster safety hash diverges from the recipe dependency")
    return PreferenceCandidate.build(
        owner_id=owner_id,
        target_text=target_text,
        source_sha256=validated["source_sha256"],
        style_sha256=style_sha256,
        layout_plan_sha256=layout_plan_sha256,
        recipe_sha256=validated["recipe_sha256"],
        output_sha256=validated["output_sha256"],
        preview_ref=preview_ref,
        context_ref=context_ref,
        metrics={**dict(metrics), "raster_safety": safety},
        hard_safety_passed=hard_safety_passed,
    )


def comparison_from_recipe_receipts(
    first_receipt: Mapping[str, Any],
    second_receipt: Mapping[str, Any],
    *,
    first_layout_plan_sha256: str,
    second_layout_plan_sha256: str,
    owner_id: str,
    target_text: str,
    style_sha256: str,
    first_preview_ref: Mapping[str, Any],
    second_preview_ref: Mapping[str, Any],
    context_ref: Mapping[str, Any],
    first_metrics: Mapping[str, Any],
    second_metrics: Mapping[str, Any],
    first_hard_safety_passed: bool,
    second_hard_safety_passed: bool,
    randomization_nonce: str,
) -> PreferenceComparison:
    """Build a tamper-evident comparison without replacing recipe/output hashes."""

    shared = {
        "owner_id": owner_id,
        "target_text": target_text,
        "style_sha256": style_sha256,
        "context_ref": context_ref,
    }
    first = candidate_from_recipe_receipt(
        first_receipt,
        layout_plan_sha256=first_layout_plan_sha256,
        preview_ref=first_preview_ref,
        metrics=first_metrics,
        hard_safety_passed=first_hard_safety_passed,
        **shared,
    )
    second = candidate_from_recipe_receipt(
        second_receipt,
        layout_plan_sha256=second_layout_plan_sha256,
        preview_ref=second_preview_ref,
        metrics=second_metrics,
        hard_safety_passed=second_hard_safety_passed,
        **shared,
    )
    return PreferenceComparison.build(first, second, randomization_nonce=randomization_nonce)
