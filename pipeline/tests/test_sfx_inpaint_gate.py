import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfx.inpaint_gate import evaluate_sfx_inpaint_gate


def test_safe_sparse_sfx_glyph_mask_allows_component_roi_inpaint():
    result = evaluate_sfx_inpaint_gate(
        {
            "background_type": "sfx_text",
            "mask_evidence": {
                "kind": "sfx_glyph_mask",
                "bbox_fill_ratio": 0.18,
                "expanded_mask_pixels": 320,
            },
            "sfx": {"qa_flags": []},
        }
    )

    assert result["allow_inpaint"] is True
    assert result["strategy"] == "lama_component_roi"
    assert result["reason"] == "safe_sfx_glyph_mask"


def test_high_density_sfx_mask_requires_review():
    result = evaluate_sfx_inpaint_gate(
        {
            "mask_evidence": {
                "kind": "sfx_glyph_mask",
                "bbox_fill_ratio": 0.62,
                "expanded_mask_pixels": 900,
            }
        }
    )

    assert result["allow_inpaint"] is False
    assert result["strategy"] == "review_required"
    assert "sfx_mask_density_high" in result["qa_flags"]


def test_moderate_density_componentized_sfx_mask_allows_inpaint():
    result = evaluate_sfx_inpaint_gate(
        {
            "mask_evidence": {
                "kind": "sfx_glyph_mask",
                "bbox_fill_ratio": 0.41,
                "expanded_mask_pixels": 900,
                "component_count": 3,
            },
            "sfx": {"qa_flags": []},
        }
    )

    assert result["allow_inpaint"] is True
    assert result["reason"] == "safe_sfx_glyph_mask"


def test_character_art_overlap_requires_review():
    result = evaluate_sfx_inpaint_gate(
        {
            "mask_evidence": {
                "kind": "sfx_glyph_mask",
                "bbox_fill_ratio": 0.12,
                "expanded_mask_pixels": 180,
            },
            "sfx": {"qa_flags": ["sfx_overlaps_character_art"]},
        }
    )

    assert result["allow_inpaint"] is False
    assert result["reason"] == "sfx_overlaps_character_art"


def test_missing_sfx_mask_evidence_requires_review():
    result = evaluate_sfx_inpaint_gate({"mask_evidence": {"kind": "ocr_pixels"}})

    assert result["allow_inpaint"] is False
    assert result["strategy"] == "review_required"
    assert "sfx_mask_evidence_missing" in result["qa_flags"]


def test_manual_override_allows_inpaint_and_marks_flag():
    result = evaluate_sfx_inpaint_gate(
        {
            "manual_sfx_inpaint_override": True,
            "mask_evidence": {"kind": "ocr_pixels"},
            "sfx": {"qa_flags": ["complex_background"]},
        }
    )

    assert result["allow_inpaint"] is True
    assert result["strategy"] == "lama_component_roi"
    assert "manual_override" in result["qa_flags"]
