from __future__ import annotations

import pytest

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
    first_layout = "b" * 64
    second_layout = "c" * 64
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
        first_metrics={"font_size_px": 30},
        second_metrics={"font_size_px": 28},
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
