"""Conservative contextual consistency for immutable owner visual profiles."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from typesetter.owner_style import (
    owner_visual_profile_sha256,
    validate_owner_visual_profile,
)


GROUP_CONFIDENCE_THRESHOLD = 0.70
GROUP_SAFE_ATTRIBUTES = frozenset({"font_name", "stroke", "shadow", "glow", "gradient"})
_EFFECT_ATTRIBUTES = ("stroke", "shadow", "glow", "gradient")
_EFFECT_STYLE_KEYS = {
    "stroke": ("contorno", "contorno_px"),
    "shadow": ("sombra", "sombra_cor", "sombra_offset"),
    "glow": ("glow", "glow_cor", "glow_px"),
    "gradient": ("cor_gradiente",),
}


def _attribute(profile: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    evidence = profile.get("style_evidence_v2")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    attributes = evidence.get("attributes")
    attributes = attributes if isinstance(attributes, Mapping) else {}
    raw = attributes.get(name)
    return raw if isinstance(raw, Mapping) else {}


def _confidence(attribute: Mapping[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(attribute.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _observed_candidates(
    members: list[tuple[str, dict[str, Any]]],
    attribute_name: str,
) -> list[tuple[float, str, str, Any]]:
    candidates: list[tuple[float, str, str, Any]] = []
    for owner_id, profile in members:
        attribute = _attribute(profile, attribute_name)
        value = attribute.get("value", "unknown")
        confidence = _confidence(attribute)
        if value in (None, "", "unknown") or confidence < GROUP_CONFIDENCE_THRESHOLD:
            continue
        candidates.append(
            (
                confidence,
                repr(value),
                owner_id,
                copy.deepcopy(value),
            )
        )
    return sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))


def _selected_effect(
    members: list[tuple[str, dict[str, Any]]],
) -> tuple[str, Any, str | None, bool]:
    candidates: list[tuple[float, str, str, Any]] = []
    for effect_name in _EFFECT_ATTRIBUTES:
        for confidence, _stable_value, owner_id, value in _observed_candidates(
            members, effect_name
        ):
            candidates.append((confidence, effect_name, owner_id, value))
    if not candidates:
        return "none", None, None, False
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], repr(item[3])))
    selected = candidates[0]
    distinct = {(item[1], repr(item[3])) for item in candidates}
    return selected[1], copy.deepcopy(selected[3]), selected[2], len(distinct) > 1


def _copy_effect_style(
    target: dict[str, Any],
    donor: Mapping[str, Any],
    effect_class: str,
) -> None:
    for effect_name, keys in _EFFECT_STYLE_KEYS.items():
        if effect_name != effect_class:
            for key in keys:
                target.pop(key, None)
    for key in _EFFECT_STYLE_KEYS.get(effect_class, ()):
        if key in donor:
            target[key] = copy.deepcopy(donor[key])
    target["effect_class"] = effect_class


def resolve_contextual_style_groups(
    profiles_by_owner: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Share only font family/effect class; never merge semantic owners."""

    normalized = {
        str(owner_id): validate_owner_visual_profile(
            profile,
            expected_owner_id=str(owner_id),
        )
        for owner_id, profile in profiles_by_owner.items()
    }
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for owner_id, profile in normalized.items():
        group_id = str(profile.get("style_group_id") or f"owner:{owner_id}")
        groups.setdefault(group_id, []).append((owner_id, profile))

    resolved: dict[str, dict[str, Any]] = {}
    for group_id, raw_members in sorted(groups.items()):
        members = sorted(raw_members, key=lambda item: item[0])
        font_candidates = _observed_candidates(members, "font_name")
        selected_font = copy.deepcopy(font_candidates[0][3]) if font_candidates else None
        font_donor = font_candidates[0][2] if font_candidates else None
        font_conflict = len({item[1] for item in font_candidates}) > 1
        effect_class, effect_value, effect_donor, effect_conflict = _selected_effect(members)
        donor_by_id = {owner_id: profile for owner_id, profile in members}

        for owner_id, profile in members:
            output = copy.deepcopy(profile)
            decision = output.get("style_application_decision_v2")
            decision = copy.deepcopy(dict(decision)) if isinstance(decision, Mapping) else {}
            applied = decision.get("applied_attributes")
            applied = copy.deepcopy(dict(applied)) if isinstance(applied, Mapping) else {}
            style = output.get("applied_style")
            style = copy.deepcopy(dict(style)) if isinstance(style, Mapping) else {}
            inherited: list[str] = []

            current_font = _attribute(output, "font_name").get("value", "unknown")
            current_font_confidence = _confidence(_attribute(output, "font_name"))
            if selected_font is not None:
                applied["font_name"] = copy.deepcopy(selected_font)
                style["fonte"] = copy.deepcopy(selected_font)
                if owner_id != font_donor and (
                    current_font != selected_font
                    or current_font_confidence < GROUP_CONFIDENCE_THRESHOLD
                ):
                    inherited.append("font_name")

            if effect_class != "none" and effect_value is not None:
                applied[effect_class] = copy.deepcopy(effect_value)
                donor_style = donor_by_id.get(effect_donor or "", {}).get("applied_style")
                _copy_effect_style(
                    style,
                    donor_style if isinstance(donor_style, Mapping) else {},
                    effect_class,
                )
                current_effect = _attribute(output, effect_class)
                if owner_id != effect_donor and (
                    current_effect.get("value", "unknown") != effect_value
                    or _confidence(current_effect) < GROUP_CONFIDENCE_THRESHOLD
                ):
                    inherited.append("effect_class")
            else:
                style["effect_class"] = "none"

            decision["applied_attributes"] = applied
            if applied:
                decision["status"] = "applied"
                output["status"] = "applied"
            output["style_application_decision_v2"] = decision
            output["applied_style"] = style
            output["style_group_resolution_v2"] = {
                "schema_version": 2,
                "group_id": group_id,
                "group_kind": str(output.get("style_group_kind") or "unknown"),
                "group_role": str(output.get("style_group_role") or "body"),
                "member_count": len(members),
                "font_donor_owner_id": font_donor,
                "effect_donor_owner_id": effect_donor,
                "effect_class": effect_class,
                "inherited_fields": sorted(inherited),
                "conflict_resolved": bool(font_conflict or effect_conflict),
                "safe_attributes": sorted(GROUP_SAFE_ATTRIBUTES),
            }
            output["visual_profile_sha256"] = owner_visual_profile_sha256(output)
            resolved[owner_id] = validate_owner_visual_profile(
                output,
                expected_owner_id=owner_id,
            )
    if set(resolved) != set(normalized):
        raise ValueError("style group resolution changed owner cardinality")
    return resolved
