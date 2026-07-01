import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfx.script_probe import probe_sfx_candidate_script


def _visual_candidate():
    return {
        "id": "sfx_visual_001",
        "bbox": [10, 20, 90, 120],
        "content_class": "sfx",
        "tipo": "sfx",
        "detector": "sfx_visual",
        "route_action": "review_required",
        "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
        "sfx": {
            "visual_detector": "sfx_visual",
            "visual_confidence": 0.72,
            "inpaint_allowed": False,
            "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
        },
    }


def test_probe_keeps_empty_visual_candidate_review_only():
    result = probe_sfx_candidate_script(_visual_candidate())

    assert result["route_action"] == "review_required"
    assert result["script"] == "unknown"
    assert result["sfx"]["inpaint_allowed"] is False
    assert "sfx_script_unknown" in result["qa_flags"]


def test_probe_promotes_hangul_visual_candidate_to_sfx_route():
    result = probe_sfx_candidate_script(_visual_candidate(), "\ucff5")

    assert result["route_action"] == "translate_sfx_inpaint_render"
    assert result["script"] == "hangul"
    assert result["translate_policy"] == "adapt_sfx"
    assert result["sfx"]["source_text"] == "\ucff5"
    assert "sfx_script_unknown" not in result["qa_flags"]


def test_probe_keeps_kana_visual_candidate_review_only():
    result = probe_sfx_candidate_script(_visual_candidate(), "\u30ba\u30c9")

    assert result["route_action"] == "review_required"
    assert result["script"] == "cjk_unknown"
    assert "sfx_script_unknown" in result["qa_flags"]
