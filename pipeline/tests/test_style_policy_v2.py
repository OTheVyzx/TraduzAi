from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.style_contract import style_evidence_v2_from_v1
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


def test_low_confidence_evidence_is_preserved_but_not_applied():
    evidence = _evidence(text_color_confidence=0.22, font_confidence=0.18, stroke_confidence=0.12)

    decision = decide_style_copy_v2(_candidate(), evidence)

    assert evidence.attributes["fill"].value == "#F4F4F4"
    assert evidence.attributes["fill"].confidence == 0.22
    assert decision.status == "fallback"
    assert decision.applied_attributes == {}
    assert set(decision.abstained_attributes) >= {"fill", "font_name", "stroke"}


def test_v2_source_style_copy_requires_authenticated_gradient():
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

    assert decision.status == "fallback"
    assert decision.applied_attributes == {}
    assert set(decision.abstained_attributes.values()) == {
        "authenticated_gradient_required"
    }
