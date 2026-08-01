from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.style_contract import (
    StyleEvidenceV2,
    style_evidence_v2_from_dict,
    style_evidence_v2_from_v1,
)
from ownership import model as ownership_model
from style_v2_fixtures import (
    valid_owner_style_raster_contract,
    valid_owner_style_raster_segment,
)


def test_owner_style_raster_contract_api_is_defined_at_ownership_boundary():
    assert hasattr(ownership_model, "OwnerStyleRasterContract")
    assert callable(
        getattr(ownership_model, "owner_style_raster_contract_sha256", None)
    )
    assert callable(
        getattr(ownership_model, "validate_owner_style_raster_contract", None)
    )


def test_owner_style_raster_contract_is_canonical_hash_bound_and_immutable():
    contract = valid_owner_style_raster_contract()

    normalized = ownership_model.validate_owner_style_raster_contract(contract)

    assert normalized["contract_sha256"] == (
        ownership_model.owner_style_raster_contract_sha256(normalized)
    )
    assert normalized["owner_id"] == "owner_p001_fixture"
    with pytest.raises(TypeError):
        contract.applied_attributes["fill"] = "#FFFFFF"


def _rehash_raster_contract(payload: dict) -> dict:
    normalized = copy.deepcopy(payload)
    normalized["contract_sha256"] = (
        ownership_model.owner_style_raster_contract_sha256(normalized)
    )
    return normalized


def test_owner_style_raster_contract_requires_complete_schema():
    payload = valid_owner_style_raster_contract().to_dict()
    payload.pop("schema_version")
    payload = _rehash_raster_contract(payload)

    with pytest.raises(ValueError, match="missing required fields"):
        ownership_model.validate_owner_style_raster_contract(payload)


def test_owner_style_raster_contract_binds_expected_owner_and_page():
    contract = valid_owner_style_raster_contract()

    with pytest.raises(ValueError, match="owner identity mismatch"):
        ownership_model.validate_owner_style_raster_contract(
            contract,
            expected_owner_id="owner_other",
        )
    with pytest.raises(ValueError, match="page identity mismatch"):
        ownership_model.validate_owner_style_raster_contract(
            contract,
            expected_page_id="page_999",
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "visual_profile_sha256",
        "profile_component_geometry_sha256",
        "execution_component_geometry_sha256",
        "source_artifact_sha256",
        "source_glyph_mask_sha256",
        "rendered_before_sha256",
        "rendered_patch_sha256",
        "rendered_after_sha256",
    ],
)
def test_owner_style_raster_contract_rejects_malformed_hashes(field_name):
    payload = valid_owner_style_raster_contract().to_dict()
    payload[field_name] = "not-a-sha256"
    payload = _rehash_raster_contract(payload)

    with pytest.raises(ValueError, match="malformed sha256"):
        ownership_model.validate_owner_style_raster_contract(payload)


def test_owner_style_raster_contract_rejects_noncanonical_attribute():
    payload = valid_owner_style_raster_contract().to_dict()
    payload["requested_attributes"]["not_supported"] = "value"
    payload = _rehash_raster_contract(payload)

    with pytest.raises(ValueError, match="unsupported style attribute"):
        ownership_model.validate_owner_style_raster_contract(payload)


def test_owner_style_raster_contract_rejects_duplicate_segment_ids():
    payload = valid_owner_style_raster_contract().to_dict()
    profile_sha256 = payload["visual_profile_sha256"]
    first = valid_owner_style_raster_segment(
        visual_profile_sha256=profile_sha256,
    )
    duplicate = valid_owner_style_raster_segment(
        visual_profile_sha256=profile_sha256,
    )
    payload["segments"] = [
        first,
        duplicate,
    ]
    payload = _rehash_raster_contract(payload)

    with pytest.raises(ValueError, match="duplicate segment_id"):
        ownership_model.validate_owner_style_raster_contract(payload)


def test_owner_style_raster_contract_rejects_noncanonical_or_overlapping_segments():
    payload = valid_owner_style_raster_contract().to_dict()
    profile_sha256 = payload["visual_profile_sha256"]
    first = valid_owner_style_raster_segment(
        visual_profile_sha256=profile_sha256,
        segment_id="region_0",
        order=0,
        bbox_page=(0, 0, 3, 3),
    )
    second = valid_owner_style_raster_segment(
        visual_profile_sha256=profile_sha256,
        segment_id="region_1",
        order=1,
        bbox_page=(2, 2, 5, 4),
    )
    payload["segments"] = [first, second]
    payload = _rehash_raster_contract(payload)

    with pytest.raises(ValueError, match="segments overlap"):
        ownership_model.validate_owner_style_raster_contract(payload)

    second = valid_owner_style_raster_segment(
        visual_profile_sha256=profile_sha256,
        segment_id="region_1",
        order=1,
        bbox_page=(4, 0, 6, 2),
    )
    payload["segments"] = [second, first]
    payload = _rehash_raster_contract(payload)

    with pytest.raises(ValueError, match="not canonical"):
        ownership_model.validate_owner_style_raster_contract(payload)


def test_owner_style_raster_contract_rejects_tampered_child_hash():
    payload = valid_owner_style_raster_contract().to_dict()
    segment = valid_owner_style_raster_segment(
        visual_profile_sha256=payload["visual_profile_sha256"],
    )
    segment["rendered_patch_sha256"] = "f" * 64
    payload["segments"] = [segment]
    payload = _rehash_raster_contract(payload)

    with pytest.raises(ValueError, match="child raster contract hash mismatch"):
        ownership_model.validate_owner_style_raster_contract(payload)


def test_owner_style_raster_contract_rejects_self_hash_tampering():
    payload = valid_owner_style_raster_contract().to_dict()
    payload["render_metrics"]["core_pixel_count"] = 99

    with pytest.raises(ValueError, match="hash mismatch"):
        ownership_model.validate_owner_style_raster_contract(payload)


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


def test_v2_uses_calibrated_font_match_without_changing_legacy_attributes():
    evidence = style_evidence_v2_from_v1(
        {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.8,
            "font_name": "ComicNeue-Bold.ttf",
            "font_confidence": 1.0,
        },
        font_match={
            "abstention_reason": "low_top_k_margin",
            "confidence": 0.19,
            "margin": 0.02,
            "status": "family",
            "top_k": [
                {"font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf", "similarity": 0.81},
                {"font_name": "ComicNeue-Bold.ttf", "similarity": 0.79},
            ],
            "value": "LeagueGothic-Regular-VariableFont_wdth.ttf",
        },
    )

    font = evidence.to_dict()["attributes"]["font_name"]
    assert font == {
        "abstention_reason": "low_top_k_margin",
        "confidence": 0.19,
        "margin": 0.02,
        "top_k": ["LeagueGothic-Regular-VariableFont_wdth.ttf", "ComicNeue-Bold.ttf"],
        "value": "LeagueGothic-Regular-VariableFont_wdth.ttf",
    }


def test_shadow_v2_contract_migrates_without_losing_source_or_text_present():
    original = style_evidence_v2_from_v1(
        {
            "source": "masked_source_crop",
            "text_color": "#FAFAFA",
            "text_color_confidence": 0.87,
        }
    ).to_dict()
    original["source_sha256"] = "a" * 64
    original["attribute_provenance"] = {
        "fill": {"mask_sha256": "b" * 64, "method": "glyph_core"}
    }

    migrated = style_evidence_v2_from_dict(original)

    assert migrated.source == "masked_source_crop"
    assert migrated.text_present is True
    assert migrated.source_sha256 == "a" * 64
    assert migrated.attribute_provenance["fill"]["method"] == "glyph_core"
