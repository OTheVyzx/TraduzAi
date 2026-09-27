from __future__ import annotations

import pytest
import numpy as np

from integration_v1.preference_adapter import comparison_from_recipe_receipts


def _receipt(recipe: str, output: str, layout: str) -> dict:
    return {
        "recipe_sha256": recipe,
        "output_sha256": output,
        "relative_path": f"recipes/{recipe[:8]}.json",
        "runtime_id": "consumer-fast-v1",
        "runtime_sha256": "d" * 64,
        "source_sha256": "a" * 64,
        "dependency_hashes": {"layout_plan": layout, "font": "e" * 64},
    }


def _comparison(*, first_safe: bool = True):
    from typesetter.raster_safety import assess_raster_safety

    first_layout = "b" * 64
    second_layout = "c" * 64
    authorized = np.ones((20, 30), dtype=np.uint8)
    protected = np.zeros((20, 30), dtype=np.uint8)
    if not first_safe:
        protected[8:12, 10:20] = 1
    alpha = np.zeros((10, 20), dtype=np.uint8)
    alpha[3:7, 4:16] = 255
    first_safety = assess_raster_safety(
        alpha=alpha, bbox=[5, 5, 25, 15],
        authorized_body_mask=authorized, protected_art_mask=protected)
    second_safety = assess_raster_safety(
        alpha=alpha, bbox=[5, 5, 25, 15],
        authorized_body_mask=authorized,
        protected_art_mask=np.zeros_like(protected))
    return comparison_from_recipe_receipts(
        _receipt("1" * 64, "2" * 64, first_layout),
        _receipt("3" * 64, "4" * 64, second_layout),
        first_layout_plan_sha256=first_layout,
        second_layout_plan_sha256=second_layout,
        owner_id="owner-001",
        target_text="Ação!",
        style_sha256="f" * 64,
        first_preview_ref={"relative_path": "previews/a.png", "sha256": "2" * 64},
        second_preview_ref={"relative_path": "previews/b.png", "sha256": "4" * 64},
        context_ref={"relative_path": "context/page.png", "sha256": "a" * 64},
        first_metrics={"font_size_px": 30, "raster_safety": first_safety},
        second_metrics={"font_size_px": 28, "raster_safety": second_safety},
        first_hard_safety_passed=first_safe,
        second_hard_safety_passed=True,
        randomization_nonce="review-owner-001-r4",
    )


def test_adapter_preserves_renderer_recipe_and_output_identities():
    comparison = _comparison()
    candidates = comparison.candidates
    assert {item.recipe_sha256 for item in candidates} == {"1" * 64, "3" * 64}
    assert {item.output_sha256 for item in candidates} == {"2" * 64, "4" * 64}
    assert set(comparison.positions.values()) == {item.candidate_id for item in candidates}


def test_adapter_rejects_recipe_layout_dependency_mismatch():
    receipt = _receipt("1" * 64, "2" * 64, "b" * 64)
    with pytest.raises(ValueError, match="layout plan hash"):
        comparison_from_recipe_receipts(
            receipt,
            _receipt("3" * 64, "4" * 64, "c" * 64),
            first_layout_plan_sha256="9" * 64,
            second_layout_plan_sha256="c" * 64,
            owner_id="owner-001",
            target_text="Ação!",
            style_sha256="f" * 64,
            first_preview_ref={"relative_path": "previews/a.png", "sha256": "2" * 64},
            second_preview_ref={"relative_path": "previews/b.png", "sha256": "4" * 64},
            context_ref={"relative_path": "context/page.png", "sha256": "a" * 64},
            first_metrics={},
            second_metrics={},
            first_hard_safety_passed=True,
            second_hard_safety_passed=True,
            randomization_nonce="nonce",
        )


def test_adapter_refuses_unsafe_candidate_before_studio_display():
    with pytest.raises(ValueError, match="hard safety"):
        _comparison(first_safe=False)


def test_adapter_rejects_recipe_bound_to_different_raster_safety():
    from typesetter.raster_safety import assess_raster_safety

    authorized = np.ones((10, 10), dtype=np.uint8)
    evidence = assess_raster_safety(
        alpha=np.ones((4, 4), dtype=np.uint8), bbox=[2, 2, 6, 6],
        authorized_body_mask=authorized,
        protected_art_mask=np.zeros_like(authorized))
    receipt = _receipt("1" * 64, "2" * 64, "b" * 64)
    receipt["dependency_hashes"]["raster_safety"] = "9" * 64
    from integration_v1.preference_adapter import candidate_from_recipe_receipt
    with pytest.raises(ValueError, match="raster safety hash"):
        candidate_from_recipe_receipt(
            receipt, owner_id="owner-001", target_text="Ação!",
            style_sha256="f" * 64, layout_plan_sha256="b" * 64,
            preview_ref={"relative_path": "previews/a.png", "sha256": "2" * 64},
            context_ref={"relative_path": "context/page.png", "sha256": "a" * 64},
            metrics={"raster_safety": evidence}, hard_safety_passed=True)
