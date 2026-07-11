from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.style_contract import StyleEvidenceV2, style_evidence_v2_from_v1


def test_v2_marks_every_attribute_unknown_when_v1_has_no_text_evidence():
    v1 = {
        "source": "none",
        "text_color": "",
        "text_color_confidence": 0.0,
        "font_name": "ComicNeue-Bold.ttf",
        "font_confidence": 1.0,
        "stroke_color": "",
        "stroke_width_px": 0,
        "stroke_confidence": 0.0,
        "shadow": False,
        "shadow_confidence": 0.0,
        "glow": False,
        "glow_confidence": 0.0,
        "gradient": False,
        "gradient_confidence": 0.0,
    }

    evidence = style_evidence_v2_from_v1(v1)

    assert isinstance(evidence, StyleEvidenceV2)
    assert evidence.text_present is False
    assert all(attribute.value == "unknown" for attribute in evidence.attributes.values())
    assert all(attribute.abstention_reason == "no_text_evidence" for attribute in evidence.attributes.values())


def test_v2_keeps_attribute_confidence_top_k_margin_and_abstention_reason():
    evidence = style_evidence_v2_from_v1(
        {
            "source": "light_fill_dark_outline",
            "text_color": "#F4F4F4",
            "text_color_confidence": 0.82,
            "font_name": "ComicNeue-Bold.ttf",
            "font_confidence": 0.73,
            "stroke_color": "#161616",
            "stroke_width_px": 3,
            "stroke_confidence": 0.67,
            "shadow": False,
            "shadow_confidence": 0.0,
            "glow": True,
            "glow_color": "#2C7FFF",
            "glow_px": 4,
            "glow_confidence": 0.61,
            "gradient": False,
            "gradient_confidence": 0.0,
        }
    )

    serialized = evidence.to_dict()
    assert serialized["schema_version"] == 2
    assert serialized["attributes"]["fill"] == {
        "abstention_reason": "",
        "confidence": 0.82,
        "margin": 0.82,
        "top_k": ["#F4F4F4"],
        "value": "#F4F4F4",
    }
    assert serialized["attributes"]["font_name"]["top_k"] == ["ComicNeue-Bold.ttf"]
    assert serialized["attributes"]["stroke"]["value"] == {"color": "#161616", "width_px": 3}
    assert serialized["attributes"]["shadow"]["value"] == "unknown"
    assert serialized["attributes"]["shadow"]["abstention_reason"] == "insufficient_effect_confidence"
    assert serialized["attributes"]["glow"]["value"] == {
        "color": "#2C7FFF",
        "width_px": 4,
    }
