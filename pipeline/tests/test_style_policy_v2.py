from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.style_contract import StyleAttributeEvidenceV2, style_evidence_v2_from_v1
from typesetter.gradient_model import canonicalize_linear_gradient
from typesetter.style_policy import (
    decide_style_copy_v2,
    style_candidate_copy_allowed,
    style_evidence_v2_shadow_policy,
)


def test_v2_policy_is_shadow_only_even_for_high_confidence_style_evidence():
    evidence = style_evidence_v2_from_v1(
        {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.99,
            "font_name": "ComicNeue-Bold.ttf",
            "font_confidence": 0.95,
            "stroke_color": "#000000",
            "stroke_width_px": 2,
            "stroke_confidence": 0.94,
            "shadow": False,
            "shadow_confidence": 0.0,
            "glow": False,
            "glow_confidence": 0.0,
            "gradient": False,
            "gradient_confidence": 0.0,
        }
    )

    decision = style_evidence_v2_shadow_policy(evidence)

    assert decision == {
        "apply_to_renderer": False,
        "reason": "shadow_mode_no_runtime_behavior_change",
        "schema_version": 2,
    }


def _candidate(**overrides):
    candidate = {
        "content_class": "speech",
        "confidence": 0.95,
        "route_action": "translate_inpaint_render",
        "render_policy": "normal",
    }
    candidate.update(overrides)
    return candidate


def _evidence(**overrides):
    payload = {
        "source": "pixel_analysis",
        "text_color": "#F4F4F4",
        "text_color_confidence": 0.91,
        "font_name": "ComicNeue-Bold.ttf",
        "font_confidence": 0.92,
        "stroke_color": "#161616",
        "stroke_width_px": 3,
        "stroke_confidence": 0.90,
        "glow": False,
        "glow_confidence": 0.0,
    }
    payload.update(overrides)
    return style_evidence_v2_from_v1(payload)


def test_missing_candidate_confidence_never_applies_style():
    candidate = _candidate(confidence=None)

    decision = decide_style_copy_v2(candidate, _evidence())

    assert style_candidate_copy_allowed(candidate) is False
    assert decision.status == "fallback"
    assert decision.applied_attributes == {}


def test_high_glow_confidence_cannot_authorize_low_confidence_fill():
    decision = decide_style_copy_v2(
        _candidate(),
        _evidence(
            text_color_confidence=0.31,
            glow=True,
            glow_color="#4A7FFF",
            glow_px=5,
            glow_confidence=0.96,
            gradient=True,
            gradient_colors=["#FFFFFF", "#78D7FF"],
            gradient_confidence=0.95,
        ),
    )

    assert decision.status == "applied"
    assert decision.applied_attributes["glow"]["width_px"] == 5
    assert decision.applied_attributes["gradient"] == canonicalize_linear_gradient(
        ["#FFFFFF", "#78D7FF"]
    )
    assert "fill" not in decision.applied_attributes
    assert decision.abstained_attributes["fill"] == "attribute_confidence_below_threshold"


def test_raw_sfx_candidate_never_triggers_v2_style_scan():
    candidate = _candidate(
        content_class="sfx",
        detector="sfx_visual",
        confidence=0.99,
        route_action="review_required",
        render_policy="review_required",
        sfx_promotion_score=None,
    )

    assert style_candidate_copy_allowed(candidate) is False
    assert decide_style_copy_v2(candidate, _evidence()).status == "review_required"


def test_review_required_candidate_abstains():
    decision = decide_style_copy_v2(
        _candidate(route_action="review_required", render_policy="review_required"),
        _evidence(),
    )

    assert decision.status == "review_required"
    assert decision.applied_attributes == {}


def test_each_attribute_requires_its_own_confidence():
    decision = decide_style_copy_v2(
        _candidate(),
        _evidence(
            font_confidence=0.99,
            stroke_confidence=0.42,
            gradient=True,
            gradient_colors=["#FFFFFF", "#78D7FF"],
            gradient_confidence=0.95,
        ),
    )

    assert decision.applied_attributes["font_name"] == "ComicNeue-Bold.ttf"
    assert "stroke" not in decision.applied_attributes
    assert decision.abstained_attributes["stroke"] == "attribute_confidence_below_threshold"


def test_v2_source_style_copy_applies_matched_font_but_not_font_geometry_attributes():
    decision = decide_style_copy_v2(
        _candidate(),
        _evidence(
            font_name="KOMIKAX_.ttf",
            font_confidence=0.99,
            text_color_confidence=0.99,
            stroke_confidence=0.99,
            gradient=True,
            gradient_colors=["#FFFFFF", "#78D7FF"],
            gradient_confidence=0.99,
        ),
    )

    assert decision.status == "applied"
    assert decision.applied_attributes["fill"] == "#F4F4F4"
    assert decision.applied_attributes["stroke"]["color"] == "#161616"
    assert decision.applied_attributes["gradient"] == canonicalize_linear_gradient(
        ["#FFFFFF", "#78D7FF"]
    )
    assert decision.applied_attributes["font_name"] == "KOMIKAX_.ttf"
    assert "font_weight" not in decision.applied_attributes
    assert "font_width" not in decision.applied_attributes
    assert "slant_tangent" not in decision.applied_attributes
    assert "width_scale" not in decision.applied_attributes
    assert "scale_y" not in decision.applied_attributes


def test_low_confidence_evidence_is_preserved_but_not_applied():
    evidence = _evidence(text_color_confidence=0.22, font_confidence=0.18, stroke_confidence=0.12)

    decision = decide_style_copy_v2(_candidate(), evidence)

    assert evidence.attributes["fill"].value == "#F4F4F4"
    assert evidence.attributes["fill"].confidence == 0.22
    assert decision.status == "fallback"
    assert decision.applied_attributes == {}
    assert set(decision.abstained_attributes) >= {"fill", "font_name", "stroke"}


def test_v2_source_style_copy_applies_independent_stroke_without_gradient():
    decision = decide_style_copy_v2(
        _candidate(),
        _evidence(
            text_color_confidence=0.99,
            font_confidence=0.99,
            stroke_confidence=0.99,
            glow=True,
            glow_color="#4A7FFF",
            glow_px=5,
            glow_confidence=0.99,
            gradient=False,
            gradient_confidence=0.0,
        ),
    )

    assert decision.status == "applied"
    assert decision.applied_attributes["stroke"]["width_px"] == 3
    assert decision.applied_attributes["glow"]["width_px"] == 5
    assert decision.applied_attributes["font_name"] == "ComicNeue-Bold.ttf"
    assert "gradient" not in decision.applied_attributes


def test_v2_source_style_copy_applies_measured_gradient_with_lower_gradient_threshold():
    decision = decide_style_copy_v2(
        _candidate(),
        _evidence(
            text_color="#030204",
            text_color_confidence=0.94,
            font_confidence=0.99,
            stroke_color="#EDE5ED",
            stroke_width_px=1,
            stroke_confidence=0.84,
            gradient=True,
            gradient_colors=["#160A36", "#000000"],
            gradient_confidence=0.6154,
        ),
    )

    assert decision.status == "applied"
    assert decision.applied_attributes["gradient"] == canonicalize_linear_gradient(
        ["#160A36", "#000000"]
    )
    assert decision.applied_attributes["fill"] == "#030204"
    assert decision.applied_attributes["stroke"]["color"] == "#EDE5ED"
    assert decision.applied_attributes["font_name"] == "ComicNeue-Bold.ttf"


def test_v2_policy_preserves_a_confident_structured_gradient() -> None:
    evidence = _evidence(text_color_confidence=0.94)
    gradient = {
        "kind": "linear",
        "colors": ["#6633CC", "#08080A"],
        "stops": [0.0, 1.0],
        "start": [0.12, 0.08],
        "end": [0.88, 0.92],
        "coordinate_space": "glyph_bbox_normalized",
    }
    evidence.attributes["gradient"] = StyleAttributeEvidenceV2(
        value=gradient,
        confidence=0.93,
        top_k=(gradient,),
        margin=0.93,
    )

    decision = decide_style_copy_v2(_candidate(), evidence)

    assert decision.applied_attributes["gradient"] == gradient


def test_v2_policy_abstains_from_a_degenerate_structured_gradient() -> None:
    evidence = _evidence(text_color_confidence=0.94)
    evidence.attributes["gradient"] = StyleAttributeEvidenceV2(
        value={
            "kind": "linear",
            "colors": ["#6633CC", "#08080A"],
            "start": [0.5, 0.5],
            "end": [0.5, 0.5],
        },
        confidence=0.93,
        top_k=(),
        margin=0.93,
    )

    decision = decide_style_copy_v2(_candidate(), evidence)

    assert "gradient" not in decision.applied_attributes
    assert decision.abstained_attributes["gradient"] == "invalid_gradient_value"


def test_v2_source_style_copy_can_apply_only_matched_font_and_solid_color():
    decision = decide_style_copy_v2(
        _candidate(),
        _evidence(
            text_color_confidence=0.99,
            font_confidence=0.99,
            stroke_color="",
            stroke_width_px=0,
            stroke_confidence=0.0,
            glow=False,
            glow_color="",
            glow_px=0,
            glow_confidence=0.0,
            gradient=False,
            gradient_confidence=0.0,
        ),
    )

    assert decision.status == "applied"
    assert decision.applied_attributes == {
        "fill": "#F4F4F4",
        "font_name": "ComicNeue-Bold.ttf",
    }


def test_high_confidence_style_copies_font_color_outline_and_shadow_without_gradient():
    decision = decide_style_copy_v2(
        _candidate(),
        _evidence(
            font_name="LeagueGothic-Regular-VariableFont_wdth.ttf",
            font_confidence=0.96,
            text_color="#7B2CBF",
            text_color_confidence=0.97,
            stroke_color="#FFFFFF",
            stroke_width_px=2,
            stroke_confidence=0.95,
            shadow=True,
            shadow_color="#24113A",
            shadow_offset=[2, 3],
            shadow_confidence=0.93,
            gradient=False,
            gradient_confidence=0.0,
        ),
    )

    assert decision.status == "applied"
    assert decision.applied_attributes["font_name"] == "LeagueGothic-Regular-VariableFont_wdth.ttf"
    assert decision.applied_attributes["fill"] == "#7B2CBF"
    assert decision.applied_attributes["stroke"] == {"color": "#FFFFFF", "width_px": 2.0}
    assert decision.applied_attributes["shadow"] == {"color": "#24113A", "offset": [2.0, 3.0]}
    assert "gradient" not in decision.applied_attributes
