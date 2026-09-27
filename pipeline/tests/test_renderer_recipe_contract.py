from __future__ import annotations

from hashlib import sha256

import pytest


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_F = "f" * 64


def _recipe(*, target_text: str = "AÇÃO total", lines=None, separators=None):
    from typesetter.recipe_contract import ExactLinePlan, RendererRecipe

    plan = ExactLinePlan.build(
        target_text=target_text,
        lines=lines or [target_text],
        separators=separators or [],
    )
    return RendererRecipe.build(
        owner_id="owner-001",
        source_sha256=_A,
        output_sha256=_B,
        target_text=target_text,
        font={"family": "Comic Neue", "relative_path": "fonts/ComicNeue-Bold.ttf", "sha256": _C},
        rasterizer={"runtime_id": "python-textpath-v1", "runtime_sha256": _D, "config_sha256": _E},
        line_plan=plan,
        bbox=[100, 200, 400, 360],
        font_size_px=33,
        line_advance_px=36,
        effects={"fill": "#000000", "stroke_width_px": 0},
        anchors={"source_text_bbox": [130, 220, 360, 330], "mode": "source_center"},
        geometry={"usable_body_sha256": _E, "coordinate_space": "logical_page"},
        policy_versions={"layout": "renderer-v1", "leading": "compact_safe_leading_v1"},
        dependency_hashes={"layout_plan": _F, "font_map": _C},
    )


@pytest.mark.parametrize(
    ("target_text", "lines", "separators"),
    [
        ("AÇÃO total", ["AÇÃO", "total"], [" "]),
        ("AC\u0327A\u0303O\ntotal", ["AC\u0327A\u0303O", "total"], ["\n"]),
    ],
)
def test_exact_line_plan_roundtrips_unicode_and_break_separators(
    target_text: str, lines: list[str], separators: list[str]
) -> None:
    from typesetter.recipe_contract import ExactLinePlan

    plan = ExactLinePlan.build(
        target_text=target_text,
        lines=lines,
        separators=separators,
    )
    rebuilt = ExactLinePlan.from_dict(plan.to_dict())

    assert rebuilt.reconstruct() == target_text
    assert rebuilt.reconstruct().encode("utf-8") == target_text.encode("utf-8")


def test_exact_line_plan_rejects_changed_content_order_or_punctuation() -> None:
    from typesetter.recipe_contract import ExactLinePlan

    with pytest.raises(ValueError, match="reconstruct"):
        ExactLinePlan.build(
            target_text="Olá, mundo!",
            lines=["mundo!", "Olá"],
            separators=[" "],
        )


def test_recipe_is_canonical_hash_bound_and_roundtrippable() -> None:
    from typesetter.recipe_contract import RendererRecipe

    recipe = _recipe(target_text="AÇÃO total", lines=["AÇÃO", "total"], separators=[" "])
    rebuilt = RendererRecipe.from_dict(recipe.to_dict())

    assert rebuilt == recipe
    assert recipe.target_utf8_sha256 == sha256("AÇÃO total".encode("utf-8")).hexdigest()
    assert recipe.recipe_sha256 == rebuilt.recipe_sha256
    assert recipe.line_plan.reconstruct() == recipe.target_text


def test_recipe_rejects_tampered_plan_and_output() -> None:
    from typesetter.recipe_contract import RendererRecipe

    recipe = _recipe()
    tampered = recipe.to_dict()
    tampered["font_size_px"] = 8
    with pytest.raises(ValueError, match="recipe hash mismatch"):
        RendererRecipe.from_dict(tampered)

    with pytest.raises(ValueError, match="output SHA-256"):
        recipe.verify_output(b"not-the-rendered-pixels")


def test_recipe_fails_closed_for_runtime_font_or_dependency_drift() -> None:
    recipe = _recipe()

    recipe.assert_replay_compatible(
        runtime_id="python-textpath-v1",
        runtime_sha256=_D,
        font_sha256=_C,
        dependency_hashes={"layout_plan": _F, "font_map": _C},
    )
    with pytest.raises(ValueError, match="runtime"):
        recipe.assert_replay_compatible(
            runtime_id="python-textpath-v2",
            runtime_sha256=_D,
            font_sha256=_C,
            dependency_hashes={"layout_plan": _F, "font_map": _C},
        )
    with pytest.raises(ValueError, match="font"):
        recipe.assert_replay_compatible(
            runtime_id="python-textpath-v1",
            runtime_sha256=_D,
            font_sha256=_B,
            dependency_hashes={"layout_plan": _F, "font_map": _C},
        )
    with pytest.raises(ValueError, match="dependencies"):
        recipe.assert_replay_compatible(
            runtime_id="python-textpath-v1",
            runtime_sha256=_D,
            font_sha256=_C,
            dependency_hashes={"layout_plan": _A, "font_map": _C},
        )


def test_recipe_exports_the_existing_integration_receipt_shape() -> None:
    from integration_v1.contracts import adapt_contract_payload

    recipe = _recipe()
    receipt = recipe.to_receipt("recipes/page-001/owner-001.json")

    assert adapt_contract_payload("Recipe", receipt) == receipt
    assert receipt["recipe_sha256"] == recipe.recipe_sha256
    assert receipt["runtime_sha256"] == recipe.rasterizer["runtime_sha256"]
