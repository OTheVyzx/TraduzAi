from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.style_contract import style_evidence_v2_from_v1
from typesetter.style_policy import style_evidence_v2_shadow_policy


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
