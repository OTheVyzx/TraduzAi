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

    assert {item["applied_style"]["fonte"] for item in resolved.values()} == {"League.ttf"}
    assert {item["style_group_resolution_v2"]["effect_class"] for item in resolved.values()} == {"stroke"}


def test_same_balloon_cannot_mix_unrelated_styles() -> None:
    profiles = {
        "a": _profile("a", group_id="balloon_1", role="body", font="FontA.ttf", font_confidence=0.96, weight="regular"),
        "b": _profile("b", group_id="balloon_1", role="body", font="FontB.ttf", font_confidence=0.82, weight="regular"),
    }

    resolved = resolve_contextual_style_groups(profiles)

    assert len({item["applied_style"]["fonte"] for item in resolved.values()}) == 1
    assert all(item["style_group_resolution_v2"]["conflict_resolved"] for item in resolved.values())


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
        owner_id: {key: value for key, value in record.items() if not key.startswith("visual_") and key != "style_copy_status"}
        for owner_id, record in records.items()
    }

    resolved = resolve_contextual_style_groups(profiles)
    updated = {
        owner_id: attach_owner_visual_profile(records[owner_id], profile)
        for owner_id, profile in resolved.items()
    }

    assert set(updated) == set(records)
    assert {
        owner_id: {key: value for key, value in record.items() if not key.startswith("visual_") and key != "style_copy_status"}
        for owner_id, record in updated.items()
    } == semantic_before


def test_low_confidence_member_inherits_only_group_safe_fields() -> None:
    donor = _profile("donor", group_id="card_3", role="title", font="League.ttf", font_confidence=0.97, weight="bold", effect="glow")
    member = _profile("member", group_id="card_3", role="body", font="unknown", font_confidence=0.0, weight="regular")
    member_before = copy.deepcopy(member)

    resolved = resolve_contextual_style_groups({"donor": donor, "member": member})["member"]

    assert resolved["applied_style"]["fonte"] == "League.ttf"
    assert resolved["style_group_resolution_v2"]["inherited_fields"] == ["effect_class", "font_name"]
    assert resolved["style_evidence_v2"]["attributes"]["font_weight"] == member_before["style_evidence_v2"]["attributes"]["font_weight"]
    assert resolved["style_evidence_v2"]["attributes"]["fill"] == member_before["style_evidence_v2"]["attributes"]["fill"]


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
            profile["style_group_role"],
            profile["style_evidence_v2"]["attributes"]["font_weight"]["value"],
        )
        for owner_id, profile in resolved.items()
    }
    assert after == before
