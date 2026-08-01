"""Conservative contextual consistency for immutable owner visual profiles."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from typesetter.owner_style import (
    validate_owner_visual_profile,
)
from typesetter.style_materialization import build_resolved_style_intent


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


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _attribute_candidates_by_name(
    members: list[tuple[str, dict[str, Any]]],
) -> dict[str, list[tuple[float, str, str, Any]]]:
    return {
        name: _observed_candidates(members, name)
        for name in sorted(GROUP_SAFE_ATTRIBUTES)
    }


def _group_conflicts(
    candidates_by_name: Mapping[str, list[tuple[float, str, str, Any]]],
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for name, candidates in candidates_by_name.items():
        distinct = {stable for _confidence_value, stable, _owner_id, _value in candidates}
        if len(distinct) <= 1:
            continue
        conflicts.append(
            {
                "attribute": name,
                "candidates": [
                    {
                        "owner_id": owner_id,
                        "confidence": confidence,
                        "value": copy.deepcopy(value),
                    }
                    for confidence, _stable, owner_id, value in candidates
                ],
                "resolution": "preserve_owner_local_or_abstain",
            }
        )
    return conflicts


def resolve_contextual_style_groups(
    profiles_by_owner: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Derive group intent without mutating an approved owner profile."""

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
        candidates_by_name = _attribute_candidates_by_name(members)
        group_conflicts = _group_conflicts(candidates_by_name)
        conflicting_names = {row["attribute"] for row in group_conflicts}

        for owner_id, profile in members:
            decision = profile.get("style_application_decision_v2")
            decision = copy.deepcopy(dict(decision)) if isinstance(decision, Mapping) else {}
            applied = decision.get("applied_attributes")
            applied = copy.deepcopy(dict(applied)) if isinstance(applied, Mapping) else {}
            abstained = decision.get("abstained_attributes")
            abstained = copy.deepcopy(dict(abstained)) if isinstance(abstained, Mapping) else {}
            inherited: list[str] = []
            donors: dict[str, str] = {}
            attribute_provenance: dict[str, Any] = {
                name: {
                    "source": "owner_local_decision",
                    "owner_id": owner_id,
                    "evidence_id": f"{owner_id}:{name}",
                }
                for name in applied
            }

            for name in sorted(GROUP_SAFE_ATTRIBUTES):
                if name in applied:
                    continue
                candidates = candidates_by_name[name]
                if not candidates:
                    continue
                if name in conflicting_names:
                    abstained[name] = "group_conflict_requires_owner_local_evidence"
                    continue
                _confidence_value, _stable, donor_owner_id, donor_value = candidates[0]
                applied[name] = copy.deepcopy(donor_value)
                abstained.pop(name, None)
                inherited.append(name)
                donors[name] = donor_owner_id
                attribute_provenance[name] = {
                    "source": "group_inheritance",
                    "donor_owner_id": donor_owner_id,
                    "evidence_id": f"{donor_owner_id}:{name}",
                }

            resolution_contract: dict[str, Any] = {
                "schema_version": 3,
                "group_id": group_id,
                "owner_id": owner_id,
                "group_kind": str(profile.get("style_group_kind") or "unknown"),
                "group_role": str(profile.get("style_group_role") or "body"),
                "member_count": len(members),
                "member_profile_sha256": {
                    member_owner_id: member_profile["visual_profile_sha256"]
                    for member_owner_id, member_profile in members
                },
                "donors": donors,
                "inherited_attributes": sorted(inherited),
                "conflicts": copy.deepcopy(group_conflicts),
                "safe_attributes": sorted(GROUP_SAFE_ATTRIBUTES),
            }
            resolution_sha256 = _canonical_sha256(resolution_contract)
            resolution_contract["group_resolution_sha256"] = resolution_sha256
            decision_sha256 = _canonical_sha256(decision)
            intent = build_resolved_style_intent(
                owner_id=owner_id,
                page_id=str(profile.get("page_id") or f"style:{owner_id}"),
                visual_profile_sha256=profile["visual_profile_sha256"],
                decision_sha256=decision_sha256,
                group_resolution_sha256=resolution_sha256,
                approved=applied,
                approved_abstentions=abstained,
                attribute_provenance=attribute_provenance,
            )
            resolved[owner_id] = {
                "visual_profile_v2": copy.deepcopy(profile),
                "visual_profile_sha256": profile["visual_profile_sha256"],
                "style_group_resolution_v3": resolution_contract,
                "style_resolved_intent_v1": intent.to_dict(),
            }
    if set(resolved) != set(normalized):
        raise ValueError("style group resolution changed owner cardinality")
    return resolved
