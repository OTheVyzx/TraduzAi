from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.style_contract import STYLE_V2_ATTRIBUTE_NAME_SET


def _materialization_module():
    spec = importlib.util.find_spec("typesetter.style_materialization")
    assert spec is not None, "typesetter.style_materialization must exist"
    return importlib.import_module("typesetter.style_materialization")


def test_materialization_module_and_domain_registry_are_complete():
    module = _materialization_module()

    assert set(module.ATTRIBUTE_DOMAIN) == STYLE_V2_ATTRIBUTE_NAME_SET
    assert set(module.attributes_for_domain("layout")) == {
        "font_size_px",
        "alignment",
        "container",
        "tracking_xh",
        "curve",
    }
    assert set(module.attributes_for_domain("font")) == {
        "font_name",
        "font_weight",
        "font_width",
    }
    assert set(module.attributes_for_domain("raster")) == (
        STYLE_V2_ATTRIBUTE_NAME_SET
        - set(module.attributes_for_domain("layout"))
        - set(module.attributes_for_domain("font"))
    )


def test_canonicalization_normalizes_equivalent_values():
    module = _materialization_module()

    assert module.canonicalize_style_attribute("fill", "fff") == "#FFFFFF"
    assert module.canonicalize_style_attribute("rotation_deg", -0.0) == 0.0
    assert module.canonicalize_style_attribute(
        "stroke", {"width_xh": 0.1, "color": "#fff"}
    ) == {"color": "#FFFFFF", "width_xh": 0.1}


def test_numeric_canonicalization_removes_sub_precision_pixel_arithmetic_noise():
    module = _materialization_module()
    observed = (0.15 * 21.0 + 2.0e-14) / 21.0

    comparison = module.compare_style_attribute("tracking_xh", 0.15, observed)

    assert comparison.matches is True
    assert comparison.expected == comparison.observed == 0.15


def test_material_difference_is_not_normalized_away():
    module = _materialization_module()

    result = module.compare_style_attribute("fill", "#FFFFFF", "#F0A000")

    assert result.matches is False
    assert result.delta_e_2000 is not None and result.delta_e_2000 > 12


def test_unknown_or_incomplete_domain_registry_is_rejected():
    module = _materialization_module()

    with pytest.raises(ValueError, match="domain registry"):
        module.validate_attribute_domain_registry({"fill": "raster"})
    with pytest.raises(ValueError, match="domain registry"):
        module.validate_attribute_domain_registry(
            dict(module.ATTRIBUTE_DOMAIN) | {"unknown": "raster"}
        )


def _intent(module, approved=None, abstained=None):
    return module.build_resolved_style_intent(
        owner_id="owner_a",
        page_id="page_001",
        visual_profile_sha256="a" * 64,
        decision_sha256="b" * 64,
        group_resolution_sha256="c" * 64,
        approved=approved or {"fill": "#fff"},
        approved_abstentions=abstained or {},
        attribute_provenance={"fill": {"evidence_id": "fill-source"}},
    )


def _plan(module, intent, targets, kinds=None, reasons=None, x_height=20):
    return module.build_materialization_plan(
        intent=intent,
        render_layout_contract_sha256="d" * 64,
        targets=targets,
        resolution_kinds=kinds or {name: "exact" for name in targets},
        resolution_reasons=reasons or {},
        rendered_x_height_px=x_height,
    )


def _observation(module, plan, rows, render_completed=True):
    return module.build_materialization_observation(
        plan=plan,
        domain_observations=rows,
        render_completed=render_completed,
    )


def test_materialization_plan_resolves_every_approved_attribute_once():
    module = _materialization_module()
    intent = _intent(
        module,
        approved={"fill": "#fff", "font_name": "ComicNeue-Bold.ttf"},
        abstained={"glow": "low_confidence"},
    )
    plan = _plan(
        module,
        intent,
        targets={
            "fill": "#FFFFFF",
            "font_name": {
                "filename": "ComicNeue-Bold.ttf",
                "file_sha256": "e" * 64,
            },
        },
        kinds={"fill": "exact", "font_name": "exact", "glow": "abstained"},
    )

    assert set(plan.attribute_plans) == {"fill", "font_name", "glow"}
    assert plan.attribute_plans["fill"].domain == "raster"
    assert plan.attribute_plans["glow"].resolution_kind == "abstained"
    assert len(plan.plan_sha256) == 64


def test_fit_adjustment_is_preserved_as_intent_but_compared_to_resolved_plan():
    module = _materialization_module()
    intent = _intent(module, approved={"font_size_px": 48})
    plan = _plan(
        module,
        intent,
        targets={"font_size_px": 36},
        kinds={"font_size_px": "policy_adjusted"},
        reasons={"font_size_px": "fit_to_verified_container"},
    )
    observation = _observation(
        module,
        plan,
        {
            "layout": {
                "font_size_px": {
                    "value": 36,
                    "evidence_kind": "final_glyph_positions",
                    "evidence_sha256": "f" * 64,
                }
            }
        },
    )

    assert plan.attribute_plans["font_size_px"].intent_value == 48
    assert plan.attribute_plans["font_size_px"].target_value == 36
    assert module.compare_materialization(plan, observation).status == "match"


def test_gradient_supersedes_unobservable_solid_fill_explicitly():
    module = _materialization_module()
    intent = _intent(
        module,
        approved={"fill": "#FFFFFF", "gradient": ["#FFFFFF", "#000000"]},
    )
    plan = _plan(
        module,
        intent,
        targets={"gradient": ["#FFFFFF", "#000000"]},
        kinds={"gradient": "exact", "fill": "superseded"},
        reasons={"fill": "gradient_is_observable_fill"},
    )
    observation = _observation(
        module,
        plan,
        {
            "raster": {
                "gradient": {
                    "value": ["#FFFFFF", "#000000"],
                    "evidence_kind": "layer_pixels_and_mask",
                    "evidence_sha256": "1" * 64,
                }
            }
        },
    )

    assert plan.attribute_plans["fill"].superseded_by == "gradient"
    assert module.compare_materialization(plan, observation).status == "match"


def test_xheight_units_resolve_before_raster_and_compare_in_execution_units():
    module = _materialization_module()
    intent = _intent(
        module,
        approved={
            "stroke": {"color": "#000", "width_xh": 0.10},
            "glow": {"color": "#fff", "radius_xh": 0.15},
        },
    )
    plan = _plan(
        module,
        intent,
        targets={
            "stroke": {"color": "#000", "width_xh": 0.10},
            "glow": {"color": "#fff", "radius_xh": 0.15},
        },
        x_height=20,
    )

    assert plan.attribute_plans["stroke"].target_value["width_px"] == 2
    assert plan.attribute_plans["glow"].target_value["radius_px"] == 3
    assert len(plan.unit_resolution_sha256) == 64


def test_observation_cannot_claim_requested_value_without_domain_evidence():
    module = _materialization_module()
    plan = _plan(module, _intent(module), targets={"fill": "#FFFFFF"})

    with pytest.raises(ValueError, match="evidence"):
        _observation(
            module,
            plan,
            {"raster": {"fill": {"value": "#FFFFFF"}}},
        )


def test_comparison_reports_exact_attribute_domain_and_reason():
    module = _materialization_module()
    plan = _plan(module, _intent(module), targets={"fill": "#FFFFFF"})
    observation = _observation(
        module,
        plan,
        {
            "raster": {
                "fill": {
                    "value": "#000000",
                    "evidence_kind": "layer_pixels_and_mask",
                    "evidence_sha256": "2" * 64,
                }
            }
        },
    )

    comparison = module.compare_materialization(plan, observation)

    assert comparison.status == "mismatch"
    assert comparison.mismatches[0]["attribute"] == "fill"
    assert comparison.mismatches[0]["domain"] == "raster"
    assert comparison.mismatches[0]["reason"] == "canonical_value_mismatch"


def test_plan_and_observation_are_hash_bound_and_immutable():
    module = _materialization_module()
    plan = _plan(module, _intent(module), targets={"fill": "#FFFFFF"})
    observation = _observation(
        module,
        plan,
        {
            "raster": {
                "fill": {
                    "value": "#FFFFFF",
                    "evidence_kind": "layer_pixels_and_mask",
                    "evidence_sha256": "3" * 64,
                }
            }
        },
    )

    with pytest.raises(TypeError):
        observation.attributes["fill"] = {}
    assert (
        module.validate_materialization_observation(observation)["observation_sha256"]
        == observation.observation_sha256
    )
    tampered = observation.to_dict()
    tampered["attributes"]["fill"]["canonical_value"] = "#000000"
    with pytest.raises(ValueError, match="hash"):
        module.validate_materialization_observation(tampered)


def test_plan_hash_is_invariant_to_mapping_permutation():
    module = _materialization_module()
    first_intent = _intent(
        module,
        approved={"fill": "#fff", "rotation_deg": 0},
    )
    second_intent = _intent(
        module,
        approved={"rotation_deg": 0, "fill": "#fff"},
    )
    first = _plan(
        module,
        first_intent,
        targets={"fill": "#FFFFFF", "rotation_deg": 0},
    )
    second = _plan(
        module,
        second_intent,
        targets={"rotation_deg": 0, "fill": "#FFFFFF"},
    )

    assert first_intent.intent_sha256 == second_intent.intent_sha256
    assert first.plan_sha256 == second.plan_sha256
