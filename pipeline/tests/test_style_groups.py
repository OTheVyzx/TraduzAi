"""Conservative contextual visual-style group contracts."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.owner_style import attach_owner_visual_profile, build_owner_visual_profile  # noqa: E402
from typesetter.style_contract import (  # noqa: E402
    STYLE_V2_ATTRIBUTE_NAMES,
    StyleAttributeEvidenceV2,
    StyleEvidenceV2,
)
from typesetter.style_groups import resolve_contextual_style_groups  # noqa: E402


def _attribute(value, confidence: float):
    if value == "unknown":
        return StyleAttributeEvidenceV2(value, 0.0, (), 0.0, "not_observed")
    return StyleAttributeEvidenceV2(value, confidence, (value,), confidence, "")


def _profile(
    owner_id: str,
    *,
    group_id: str,
    role: str,
    font: str,
    font_confidence: float,
    weight: str,
    effect: str = "none",
) -> dict:
    attributes = {name: _attribute("unknown", 0.0) for name in STYLE_V2_ATTRIBUTE_NAMES}
    attributes["font_name"] = _attribute(font, font_confidence)
    attributes["font_weight"] = _attribute(weight, 0.94)
    if effect == "stroke":
        attributes["stroke"] = _attribute(
            {"color": "#202060", "width_px": 3, "width_xh": 0.12}, 0.91
        )
    elif effect == "glow":
        attributes["glow"] = _attribute(
            {"color": "#FFFFFF", "width_px": 4, "width_xh": 0.18}, 0.91
        )
    evidence = StyleEvidenceV2("owner_mask_v2", True, attributes)
    owner = {
        "owner_id": owner_id,
        "component_ids": [f"component_{owner_id}"],
        "selected_observation_ids": [f"observation_{owner_id}"],
    }
    component = {
        "component_id": f"component_{owner_id}",
        "bbox_page": [4, 4, 28, 20],
        "polygon_page": [[4, 4], [28, 4], [28, 20], [4, 20]],
    }
    observation = {
        "observation_id": f"observation_{owner_id}",
        "text": "SOURCE",
    }
    source = np.full((24, 32, 3), 245, dtype=np.uint8)
    glyph = np.zeros(source.shape[:2], dtype=np.uint8)
    glyph[8:16, 8:24] = 255
    return build_owner_visual_profile(
        owner,
        source,
        components=[component],
        observations=[observation],
        glyph_mask=glyph,
        candidate={
            "confidence": 0.97,
            "style_evidence_v2": evidence.to_dict(),
            "card_panel_id": group_id,
            "card_panel_role": role,
        },
    )


def test_card_rows_share_font_and_effect_class() -> None:
    profiles = {
        "title": _profile("title", group_id="card_1", role="title", font="League.ttf", font_confidence=0.96, weight="bold", effect="stroke"),
        "body": _profile("body", group_id="card_1", role="body", font="unknown", font_confidence=0.0, weight="regular"),
        "value": _profile("value", group_id="card_1", role="value", font="unknown", font_confidence=0.0, weight="bold"),
    }

    resolved = resolve_contextual_style_groups(profiles)

    assert {
        item["style_resolved_intent_v1"]["approved_attributes"]["font_name"]
        for item in resolved.values()
    } == {"League.ttf"}
    assert all(
        "stroke" in item["style_resolved_intent_v1"]["approved_attributes"]
        for item in resolved.values()
    )


def test_same_balloon_cannot_mix_unrelated_styles() -> None:
    profiles = {
        "a": _profile("a", group_id="balloon_1", role="body", font="FontA.ttf", font_confidence=0.96, weight="regular"),
        "b": _profile("b", group_id="balloon_1", role="body", font="FontB.ttf", font_confidence=0.82, weight="regular"),
    }

    resolved = resolve_contextual_style_groups(profiles)

    assert {
        item["style_resolved_intent_v1"]["approved_attributes"]["font_name"]
        for item in resolved.values()
    } == {"FontA.ttf", "FontB.ttf"}
    assert all(item["style_group_resolution_v3"]["conflicts"] for item in resolved.values())


def test_group_resolution_preserves_owner_count_and_payloads() -> None:
    profiles = {
        "a": _profile("a", group_id="card_2", role="title", font="League.ttf", font_confidence=0.95, weight="bold"),
        "b": _profile("b", group_id="card_2", role="body", font="unknown", font_confidence=0.0, weight="regular"),
    }
    records = {
        owner_id: attach_owner_visual_profile(
            {
                "owner_id": owner_id,
                "source_payload": f"SOURCE {owner_id}",
                "translated_payload": f"TRADUCAO {owner_id}",
                "route_action": "translate_inpaint_render",
            },
            profile,
        )
        for owner_id, profile in profiles.items()
    }
    semantic_before = {
        owner_id: {
            key: value
            for key, value in record.items()
            if not key.startswith(("visual_", "style_"))
        }
        for owner_id, record in records.items()
    }

    resolved = resolve_contextual_style_groups(profiles)
    updated = {
        owner_id: attach_owner_visual_profile(records[owner_id], envelope)
        for owner_id, envelope in resolved.items()
    }

    assert set(updated) == set(records)
    assert {
        owner_id: {
            key: value
            for key, value in record.items()
            if not key.startswith(("visual_", "style_"))
        }
        for owner_id, record in updated.items()
    } == semantic_before


def test_low_confidence_member_inherits_only_group_safe_fields() -> None:
    donor = _profile("donor", group_id="card_3", role="title", font="League.ttf", font_confidence=0.97, weight="bold", effect="glow")
    member = _profile("member", group_id="card_3", role="body", font="unknown", font_confidence=0.0, weight="regular")
    member_before = copy.deepcopy(member)

    resolved = resolve_contextual_style_groups({"donor": donor, "member": member})["member"]

    assert resolved["style_resolved_intent_v1"]["approved_attributes"]["font_name"] == "League.ttf"
    assert resolved["style_group_resolution_v3"]["inherited_attributes"] == ["font_name", "glow"]
    assert resolved["visual_profile_v2"] == member_before
    assert resolved["visual_profile_v2"]["style_evidence_v2"]["attributes"]["font_weight"] == member_before["style_evidence_v2"]["attributes"]["font_weight"]
    assert resolved["visual_profile_v2"]["style_evidence_v2"]["attributes"]["fill"] == member_before["style_evidence_v2"]["attributes"]["fill"]


def test_title_body_and_value_keep_role_hierarchy() -> None:
    profiles = {
        "title": _profile("title", group_id="card_4", role="title", font="League.ttf", font_confidence=0.95, weight="bold"),
        "body": _profile("body", group_id="card_4", role="body", font="unknown", font_confidence=0.0, weight="regular"),
        "value": _profile("value", group_id="card_4", role="value", font="unknown", font_confidence=0.0, weight="bold"),
    }
    before = {
        owner_id: (
            profile["style_group_role"],
            profile["style_evidence_v2"]["attributes"]["font_weight"]["value"],
        )
        for owner_id, profile in profiles.items()
    }

    resolved = resolve_contextual_style_groups(profiles)

    after = {
        owner_id: (
            envelope["visual_profile_v2"]["style_group_role"],
            envelope["visual_profile_v2"]["style_evidence_v2"]["attributes"]["font_weight"]["value"],
        )
        for owner_id, envelope in resolved.items()
    }
    assert after == before


def test_group_resolution_preserves_compatible_owner_local_effects():
    profile = _profile(
        "card",
        group_id="card_effects",
        role="body",
        font="League.ttf",
        font_confidence=0.99,
        weight="bold",
        effect="stroke",
    )
    profile["style_evidence_v2"]["attributes"]["glow"] = _attribute(
        {"color": "#FFD34D", "width_px": 3}, 0.95
    ).to_dict()
    profile["style_evidence_v2"]["attributes"]["fill"] = _attribute(
        "#FFFFFF", 0.98
    ).to_dict()
    evidence = StyleEvidenceV2(
        "owner_mask_v2",
        True,
        {
            name: StyleAttributeEvidenceV2(
                value=row["value"],
                confidence=row["confidence"],
                top_k=tuple(row["top_k"]),
                margin=row["margin"],
                abstention_reason=row["abstention_reason"],
            )
            for name, row in profile["style_evidence_v2"]["attributes"].items()
        },
    )
    candidate = {
        "confidence": 0.99,
        "style_evidence_v2": evidence.to_dict(),
        "card_panel_id": "card_effects",
        "card_panel_role": "body",
    }
    rebuilt = _profile(
        "card",
        group_id="card_effects",
        role="body",
        font="League.ttf",
        font_confidence=0.99,
        weight="bold",
        effect="stroke",
    )
    # Rebuild through the real owner-profile builder so decision/evidence hashes
    # remain authoritative after adding the compatible glow observation.
    original_builder = build_owner_visual_profile
    source = np.full((24, 32, 3), 245, dtype=np.uint8)
    glyph = np.zeros(source.shape[:2], dtype=np.uint8)
    glyph[8:16, 8:24] = 255
    rebuilt = original_builder(
        {
            "owner_id": "card",
            "component_ids": ["component_card"],
            "selected_observation_ids": ["observation_card"],
        },
        source,
        components=[{
            "component_id": "component_card",
            "bbox_page": [4, 4, 28, 20],
            "polygon_page": [[4, 4], [28, 4], [28, 20], [4, 20]],
        }],
        observations=[{"observation_id": "observation_card", "text": "SOURCE"}],
        glyph_mask=glyph,
        candidate=candidate,
    )
    decision_before = copy.deepcopy(rebuilt["style_application_decision_v2"])

    resolved = resolve_contextual_style_groups({"card": rebuilt})["card"]

    assert set(resolved["style_resolved_intent_v1"]["approved_attributes"]) >= {
        "fill",
        "stroke",
        "glow",
    }
    assert resolved["visual_profile_v2"]["style_application_decision_v2"] == decision_before
    assert "style_materialization_plan_v1" not in resolved


def test_group_inheritance_never_replaces_high_confidence_owner_decision():
    profiles = {
        "a": _profile("a", group_id="conflict", role="body", font="ComicNeue-Bold.ttf", font_confidence=0.99, weight="bold"),
        "b": _profile("b", group_id="conflict", role="body", font="KOMIKAX_.ttf", font_confidence=0.98, weight="bold"),
    }

    resolved = resolve_contextual_style_groups(profiles)

    assert resolved["a"]["style_resolved_intent_v1"]["approved_attributes"]["font_name"] == "ComicNeue-Bold.ttf"
    assert resolved["b"]["style_resolved_intent_v1"]["approved_attributes"]["font_name"] == "KOMIKAX_.ttf"
    assert resolved["a"]["visual_profile_sha256"] == profiles["a"]["visual_profile_sha256"]
    assert resolved["b"]["visual_profile_sha256"] == profiles["b"]["visual_profile_sha256"]


def test_group_resolution_records_explicit_conflict_instead_of_silent_drop():
    profiles = {
        "a": _profile("a", group_id="conflict_2", role="body", font="FontA.ttf", font_confidence=0.99, weight="bold", effect="stroke"),
        "b": _profile("b", group_id="conflict_2", role="body", font="FontB.ttf", font_confidence=0.98, weight="bold", effect="glow"),
    }

    resolved = resolve_contextual_style_groups(profiles)

    for owner_id, envelope in resolved.items():
        assert envelope["style_group_resolution_v3"]["conflicts"]
        original_applied = profiles[owner_id]["style_application_decision_v2"]["applied_attributes"]
        resolved_applied = envelope["style_resolved_intent_v1"]["approved_attributes"]
        assert set(original_applied).issubset(resolved_applied)
