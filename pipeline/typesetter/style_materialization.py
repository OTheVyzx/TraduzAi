"""Canonical contracts shared by style materialization producers.

This module owns value normalization and comparison.  Domain-specific
observers may only claim attributes registered to their own domain.
"""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping, Sequence

from typesetter.style_contract import STYLE_V2_ATTRIBUTE_NAME_SET


MaterializationDomain = Literal["layout", "font", "raster"]

ATTRIBUTE_DOMAIN: Final[dict[str, MaterializationDomain]] = {
    "font_size_px": "layout",
    "alignment": "layout",
    "container": "layout",
    "tracking_xh": "layout",
    "curve": "layout",
    "font_name": "font",
    "font_weight": "font",
    "font_width": "font",
    "slant_tangent": "raster",
    "width_scale": "raster",
    "scale_y": "raster",
    "fill": "raster",
    "stroke": "raster",
    "multistroke": "raster",
    "shadow": "raster",
    "glow": "raster",
    "gradient": "raster",
    "rotation_deg": "raster",
}


@dataclass(frozen=True)
class AttributeComparison:
    name: str
    domain: MaterializationDomain
    expected: Any
    observed: Any
    matches: bool
    tolerance: Mapping[str, Any]
    delta_e_2000: float | None = None
    reason: str = ""


_COLOR_FIELDS = frozenset({"color", "cor", "fill"})
_NUMERIC_ATTRIBUTES = frozenset(
    {"font_size_px", "tracking_xh", "slant_tangent", "width_scale", "scale_y", "rotation_deg"}
)
_ALIGNMENT_ALIASES = {
    "start": "left",
    "left": "left",
    "center": "center",
    "centre": "center",
    "middle": "center",
    "right": "right",
    "end": "right",
}
_WEIGHT_ALIASES = {
    "thin": 100,
    "extralight": 200,
    "extra-light": 200,
    "light": 300,
    "regular": 400,
    "normal": 400,
    "medium": 500,
    "semibold": 600,
    "semi-bold": 600,
    "bold": 700,
    "extrabold": 800,
    "extra-bold": 800,
    "black": 900,
}
_WIDTH_ALIASES = {
    "ultra-condensed": 1,
    "extra-condensed": 2,
    "condensed": 3,
    "semi-condensed": 4,
    "normal": 5,
    "medium": 5,
    "semi-expanded": 6,
    "expanded": 7,
    "extra-expanded": 8,
    "ultra-expanded": 9,
}


def validate_attribute_domain_registry(
    registry: Mapping[str, MaterializationDomain],
) -> None:
    names = set(registry)
    domains = set(registry.values())
    if names != STYLE_V2_ATTRIBUTE_NAME_SET or not domains <= {"layout", "font", "raster"}:
        missing = sorted(STYLE_V2_ATTRIBUTE_NAME_SET - names)
        extra = sorted(names - STYLE_V2_ATTRIBUTE_NAME_SET)
        raise ValueError(
            f"invalid materialization domain registry: missing={missing}, extra={extra}, domains={sorted(domains)}"
        )


def attributes_for_domain(domain: MaterializationDomain) -> tuple[str, ...]:
    if domain not in {"layout", "font", "raster"}:
        raise ValueError(f"unsupported materialization domain: {domain}")
    validate_attribute_domain_registry(ATTRIBUTE_DOMAIN)
    return tuple(name for name, owner in ATTRIBUTE_DOMAIN.items() if owner == domain)


def _finite_number(value: Any, *, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    canonical = round(number, 9)
    if canonical == 0:
        return 0.0
    return int(canonical) if isinstance(value, int) else canonical


def canonicalize_srgb_color(value: Any) -> str:
    text = str(value or "").strip().lstrip("#")
    if len(text) in {3, 4}:
        text = "".join(character * 2 for character in text)
    if len(text) == 8:
        if text[6:8].upper() != "FF":
            raise ValueError("style colors must be opaque sRGB")
        text = text[:6]
    if len(text) != 6:
        raise ValueError(f"invalid sRGB color: {value!r}")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError(f"invalid sRGB color: {value!r}") from exc
    return f"#{text.upper()}"


def _canonicalize_mapping(value: Any, *, parent: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent} must be a mapping")
    result: dict[str, Any] = {}
    for raw_key in sorted(value, key=lambda item: str(item)):
        key = str(raw_key)
        item = value[raw_key]
        if key.lower() in _COLOR_FIELDS:
            result[key] = canonicalize_srgb_color(item)
        elif key.endswith("_px") or key.endswith("_xh") or key in {
            "angle_deg",
            "offset_x",
            "offset_y",
            "position",
            "x",
            "y",
        }:
            result[key] = _finite_number(item, field=f"{parent}.{key}")
        elif isinstance(item, Mapping):
            result[key] = _canonicalize_mapping(item, parent=f"{parent}.{key}")
        elif isinstance(item, (list, tuple)):
            result[key] = [
                _canonicalize_mapping(nested, parent=f"{parent}.{key}")
                if isinstance(nested, Mapping)
                else canonicalize_srgb_color(nested)
                if key in {"colors", "stops"} and isinstance(nested, str)
                else _finite_number(nested, field=f"{parent}.{key}")
                if isinstance(nested, (int, float)) and not isinstance(nested, bool)
                else nested
                for nested in item
            ]
        else:
            result[key] = item.strip() if isinstance(item, str) else item
    return result


def _canonicalize_effect(value: Any, *, name: str) -> Any:
    if name == "multistroke":
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("multistroke must contain at least one stroke layer")
        return [_canonicalize_mapping(layer, parent="multistroke") for layer in value]
    return _canonicalize_mapping(value, parent=name)


def canonicalize_style_attribute(
    name: str,
    value: Any,
    *,
    font_catalog: Any | None = None,
) -> Any:
    validate_attribute_domain_registry(ATTRIBUTE_DOMAIN)
    if name not in ATTRIBUTE_DOMAIN:
        raise ValueError(f"unknown style attribute: {name}")
    if name == "font_name":
        if font_catalog is None:
            raise ValueError("font_name requires resolved font identity")
        from typesetter.font_identity import canonicalize_font_intent

        return canonicalize_font_intent(value, font_catalog).to_dict()
    if name in {"fill"}:
        return canonicalize_srgb_color(value)
    if name in _NUMERIC_ATTRIBUTES:
        return _finite_number(value, field=name)
    if name == "alignment":
        canonical = _ALIGNMENT_ALIASES.get(str(value or "").strip().lower())
        if canonical is None:
            raise ValueError(f"unsupported alignment: {value!r}")
        return canonical
    if name == "font_weight":
        if isinstance(value, str) and value.strip().lower() in _WEIGHT_ALIASES:
            return _WEIGHT_ALIASES[value.strip().lower()]
        weight = _finite_number(value, field=name)
        if not 1 <= float(weight) <= 1000:
            raise ValueError("font_weight must be between 1 and 1000")
        return int(round(float(weight)))
    if name == "font_width":
        if isinstance(value, str) and value.strip().lower() in _WIDTH_ALIASES:
            return _WIDTH_ALIASES[value.strip().lower()]
        width = _finite_number(value, field=name)
        if not 1 <= float(width) <= 9:
            raise ValueError("font_width must be an OpenType width class from 1 to 9")
        return int(round(float(width)))
    if name in {"stroke", "multistroke", "shadow", "glow"}:
        return _canonicalize_effect(value, name=name)
    if name == "gradient":
        if isinstance(value, Mapping):
            return _canonicalize_mapping(value, parent=name)
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError("gradient must contain at least two colors")
        return [canonicalize_srgb_color(item) for item in value]
    if name in {"curve", "container"}:
        return _canonicalize_mapping(value, parent=name)
    raise ValueError(f"canonicalization is not implemented for style attribute: {name}")


def _hex_rgb(value: Any) -> tuple[int, int, int] | None:
    try:
        text = canonicalize_srgb_color(value).lstrip("#")
    except ValueError:
        return None
    return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))


def _rgb_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    values = []
    for channel in rgb:
        value = channel / 255.0
        values.append(((value + 0.055) / 1.055) ** 2.4 if value > 0.04045 else value / 12.92)
    r, g, b = values
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    def pivot(component: float) -> float:
        return component ** (1 / 3) if component > 0.008856 else 7.787 * component + 16 / 116

    fx, fy, fz = pivot(x), pivot(y), pivot(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e_2000(first: Any, second: Any) -> float | None:
    """Return CIEDE2000 distance for two opaque sRGB colors."""

    rgb1, rgb2 = _hex_rgb(first), _hex_rgb(second)
    if rgb1 is None or rgb2 is None:
        return None
    l1, a1, b1 = _rgb_lab(rgb1)
    l2, a2, b2 = _rgb_lab(rgb2)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    mean_c = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(mean_c**7 / (mean_c**7 + 25**7)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = math.hypot(ap1, b1), math.hypot(ap2, b2)

    def hue(a: float, b: float) -> float:
        return (math.degrees(math.atan2(b, a)) + 360) % 360 if a or b else 0.0

    hp1, hp2 = hue(ap1, b1), hue(ap2, b2)
    dl, dc = l2 - l1, cp2 - cp1
    dh_raw = hp2 - hp1
    dh = dh_raw - 360 if dh_raw > 180 else dh_raw + 360 if dh_raw < -180 else dh_raw
    dh_term = 2 * math.sqrt(cp1 * cp2) * math.sin(math.radians(dh / 2))
    mean_l, mean_cp = (l1 + l2) / 2, (cp1 + cp2) / 2
    if cp1 * cp2 == 0:
        mean_h = hp1 + hp2
    else:
        correction = 360 if abs(hp1 - hp2) > 180 and hp1 + hp2 < 360 else -360 if abs(hp1 - hp2) > 180 else 0
        mean_h = (hp1 + hp2 + correction) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(mean_h - 30))
        + 0.24 * math.cos(math.radians(2 * mean_h))
        + 0.32 * math.cos(math.radians(3 * mean_h + 6))
        - 0.20 * math.cos(math.radians(4 * mean_h - 63))
    )
    sl = 1 + 0.015 * (mean_l - 50) ** 2 / math.sqrt(20 + (mean_l - 50) ** 2)
    sc, sh = 1 + 0.045 * mean_cp, 1 + 0.015 * mean_cp * t
    rt = -2 * math.sqrt(mean_cp**7 / (mean_cp**7 + 25**7)) * math.sin(
        math.radians(60 * math.exp(-((mean_h - 275) / 25) ** 2))
    )
    return round(
        math.sqrt(
            (dl / sl) ** 2
            + (dc / sc) ** 2
            + (dh_term / sh) ** 2
            + rt * (dc / sc) * (dh_term / sh)
        ),
        4,
    )


def _compare_effect_raster(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], float | None]:
    """Compare discrete effect measurements using the R5 raster budget."""

    if set(expected) != set(observed):
        return False, {"kind": "effect_raster"}, None
    numeric_error_max = 2.0
    color_distances: list[float] = []
    matches = True
    for key in expected:
        left = expected[key]
        right = observed[key]
        if key.lower() in _COLOR_FIELDS:
            distance = delta_e_2000(left, right)
            if distance is None:
                matches = False
            else:
                color_distances.append(distance)
                matches = matches and distance <= 12.0
            continue
        if key.endswith("_px") or key in {"offset_x", "offset_y"}:
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (left, right)
            ):
                matches = False
                continue
            allowed = max(2.0, abs(float(left)) * 0.25)
            numeric_error_max = max(numeric_error_max, allowed)
            matches = matches and abs(float(left) - float(right)) <= allowed
            continue
        if key == "offset" and isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right) or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in [*left, *right]
            ):
                matches = False
                continue
            for expected_item, observed_item in zip(left, right, strict=True):
                allowed = max(2.0, abs(float(expected_item)) * 0.25)
                numeric_error_max = max(numeric_error_max, allowed)
                matches = matches and abs(float(expected_item) - float(observed_item)) <= allowed
            continue
        matches = matches and left == right
    return (
        matches,
        {
            "kind": "effect_raster",
            "color_delta_e_max": 12.0,
            "numeric_error_max_px": numeric_error_max,
        },
        max(color_distances) if color_distances else None,
    )


def compare_style_attribute(name: str, expected: Any, observed: Any) -> AttributeComparison:
    domain = ATTRIBUTE_DOMAIN.get(name)
    if domain is None:
        raise ValueError(f"unknown style attribute: {name}")
    try:
        canonical_expected = canonicalize_style_attribute(name, expected)
        canonical_observed = canonicalize_style_attribute(name, observed)
    except ValueError as exc:
        return AttributeComparison(
            name=name,
            domain=domain,
            expected=expected,
            observed=observed,
            matches=False,
            tolerance={},
            reason=f"canonicalization_error:{exc}",
        )
    color_distance = delta_e_2000(canonical_expected, canonical_observed) if name == "fill" else None
    tolerance: dict[str, Any] = {"kind": "canonical_exact"}
    if (
        name in {"glow", "shadow"}
        and isinstance(canonical_expected, Mapping)
        and isinstance(canonical_observed, Mapping)
    ):
        matches, tolerance, color_distance = _compare_effect_raster(
            canonical_expected,
            canonical_observed,
        )
    else:
        matches = canonical_expected == canonical_observed
    return AttributeComparison(
        name=name,
        domain=domain,
        expected=canonical_expected,
        observed=canonical_observed,
        matches=matches,
        tolerance=tolerance,
        delta_e_2000=color_distance,
        reason="" if matches else "canonical_value_mismatch",
    )


validate_attribute_domain_registry(ATTRIBUTE_DOMAIN)


AttributeStatus = Literal["materialized", "abstained", "mismatch", "unavailable"]
ResolutionKind = Literal[
    "exact",
    "policy_adjusted",
    "derived",
    "superseded",
    "abstained",
    "review_required",
]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(value[key]) for key in sorted(value, key=lambda item: str(item))}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, float) and value == 0:
        return 0.0
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _contract_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _thaw(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: str, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return text


@dataclass(frozen=True)
class OwnerStyleResolvedIntent:
    schema_version: int
    owner_id: str
    page_id: str
    visual_profile_sha256: str
    decision_sha256: str
    group_resolution_sha256: str
    approved_attributes: Mapping[str, Any]
    approved_abstentions: Mapping[str, str]
    attribute_provenance: Mapping[str, Any]
    intent_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner_id": self.owner_id,
            "page_id": self.page_id,
            "visual_profile_sha256": self.visual_profile_sha256,
            "decision_sha256": self.decision_sha256,
            "group_resolution_sha256": self.group_resolution_sha256,
            "approved_attributes": _thaw(self.approved_attributes),
            "approved_abstentions": _thaw(self.approved_abstentions),
            "attribute_provenance": _thaw(self.attribute_provenance),
            "intent_sha256": self.intent_sha256,
        }


@dataclass(frozen=True)
class MaterializationAttributePlan:
    name: str
    domain: MaterializationDomain
    intent_value: Any
    target_value: Any
    resolution_kind: ResolutionKind
    reason: str
    superseded_by: str
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "intent_value": _thaw(self.intent_value),
            "target_value": _thaw(self.target_value),
            "resolution_kind": self.resolution_kind,
            "reason": self.reason,
            "superseded_by": self.superseded_by,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class OwnerStyleMaterializationPlan:
    schema_version: int
    owner_id: str
    page_id: str
    visual_profile_sha256: str
    intent_sha256: str
    render_layout_contract_sha256: str
    rendered_x_height_px: float
    unit_resolution_sha256: str
    attribute_plans: Mapping[str, MaterializationAttributePlan]
    plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner_id": self.owner_id,
            "page_id": self.page_id,
            "visual_profile_sha256": self.visual_profile_sha256,
            "intent_sha256": self.intent_sha256,
            "render_layout_contract_sha256": self.render_layout_contract_sha256,
            "rendered_x_height_px": self.rendered_x_height_px,
            "unit_resolution_sha256": self.unit_resolution_sha256,
            "attribute_plans": {
                name: plan.to_dict() for name, plan in self.attribute_plans.items()
            },
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True)
class OwnerStyleMaterializationObservation:
    schema_version: int
    owner_id: str
    page_id: str
    visual_profile_sha256: str
    plan_sha256: str
    attributes: Mapping[str, Mapping[str, Any]]
    render_completed: bool
    observation_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner_id": self.owner_id,
            "page_id": self.page_id,
            "visual_profile_sha256": self.visual_profile_sha256,
            "plan_sha256": self.plan_sha256,
            "attributes": _thaw(self.attributes),
            "render_completed": self.render_completed,
            "observation_sha256": self.observation_sha256,
        }


@dataclass(frozen=True)
class MaterializationComparison:
    status: Literal["match", "mismatch", "review_required"]
    mismatches: tuple[Mapping[str, Any], ...]
    compared_attributes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mismatches": [_thaw(item) for item in self.mismatches],
            "compared_attributes": list(self.compared_attributes),
        }


def build_resolved_style_intent(
    *,
    owner_id: str,
    page_id: str,
    visual_profile_sha256: str,
    decision_sha256: str,
    group_resolution_sha256: str,
    approved: Mapping[str, Any],
    approved_abstentions: Mapping[str, str],
    attribute_provenance: Mapping[str, Any] | None = None,
) -> OwnerStyleResolvedIntent:
    approved_names = set(approved)
    abstained_names = set(approved_abstentions)
    unknown = (approved_names | abstained_names) - STYLE_V2_ATTRIBUTE_NAME_SET
    if unknown:
        raise ValueError(f"resolved style intent contains unknown attributes: {sorted(unknown)}")
    overlap = approved_names & abstained_names
    if overlap:
        raise ValueError(f"resolved style intent duplicates approved and abstained attributes: {sorted(overlap)}")
    if not owner_id or not page_id:
        raise ValueError("resolved style intent requires owner_id and page_id")
    contract = {
        "schema_version": 1,
        "owner_id": owner_id,
        "page_id": page_id,
        "visual_profile_sha256": _require_sha256(
            visual_profile_sha256, field="visual_profile_sha256"
        ),
        "decision_sha256": _require_sha256(decision_sha256, field="decision_sha256"),
        "group_resolution_sha256": _require_sha256(
            group_resolution_sha256, field="group_resolution_sha256"
        ),
        "approved_attributes": _thaw(_deep_freeze(approved)),
        "approved_abstentions": {
            str(name): str(reason) for name, reason in sorted(approved_abstentions.items())
        },
        "attribute_provenance": _thaw(_deep_freeze(attribute_provenance or {})),
    }
    intent_sha256 = _contract_sha256(contract)
    return OwnerStyleResolvedIntent(
        schema_version=1,
        owner_id=owner_id,
        page_id=page_id,
        visual_profile_sha256=contract["visual_profile_sha256"],
        decision_sha256=contract["decision_sha256"],
        group_resolution_sha256=contract["group_resolution_sha256"],
        approved_attributes=_deep_freeze(approved),
        approved_abstentions=_deep_freeze(contract["approved_abstentions"]),
        attribute_provenance=_deep_freeze(attribute_provenance or {}),
        intent_sha256=intent_sha256,
    )


def _resolve_execution_units(value: Any, *, rendered_x_height_px: float) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("_xh"):
                px_key = f"{key[:-3]}_px"
                pixels = float(item) * rendered_x_height_px
                result[px_key] = int(round(pixels)) if round(pixels) == pixels else pixels
            else:
                result[str(key)] = _resolve_execution_units(
                    item, rendered_x_height_px=rendered_x_height_px
                )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _resolve_execution_units(item, rendered_x_height_px=rendered_x_height_px)
            for item in value
        ]
    return value


def _canonical_plan_target(name: str, value: Any, *, rendered_x_height_px: float) -> Any:
    if name == "font_name":
        if not isinstance(value, Mapping):
            raise ValueError("font_name target must be a resolved file-backed identity")
        file_sha256 = _require_sha256(str(value.get("file_sha256") or ""), field="font file_sha256")
        return _deep_freeze(dict(value) | {"file_sha256": file_sha256})
    canonical = canonicalize_style_attribute(name, value)
    return _deep_freeze(
        _resolve_execution_units(canonical, rendered_x_height_px=rendered_x_height_px)
    )


def _superseding_attribute(name: str, intent: OwnerStyleResolvedIntent) -> str:
    approved = set(intent.approved_attributes)
    if name == "fill" and "gradient" in approved:
        return "gradient"
    if name == "stroke" and "multistroke" in approved:
        return "multistroke"
    return ""


def build_materialization_plan(
    *,
    intent: OwnerStyleResolvedIntent,
    render_layout_contract_sha256: str,
    targets: Mapping[str, Any],
    resolution_kinds: Mapping[str, ResolutionKind],
    resolution_reasons: Mapping[str, str] | None = None,
    rendered_x_height_px: float,
) -> OwnerStyleMaterializationPlan:
    if not math.isfinite(float(rendered_x_height_px)) or float(rendered_x_height_px) <= 0:
        raise ValueError("rendered_x_height_px must be finite and positive")
    names = set(intent.approved_attributes) | set(intent.approved_abstentions)
    if set(resolution_kinds) != names:
        raise ValueError("materialization plan must resolve every intent attribute exactly once")
    unknown_targets = set(targets) - names
    if unknown_targets:
        raise ValueError(f"materialization plan contains unknown targets: {sorted(unknown_targets)}")
    reasons = dict(resolution_reasons or {})
    plans: dict[str, MaterializationAttributePlan] = {}
    unit_resolution: dict[str, Any] = {}
    for name in sorted(names):
        kind = resolution_kinds[name]
        if kind not in {
            "exact",
            "policy_adjusted",
            "derived",
            "superseded",
            "abstained",
            "review_required",
        }:
            raise ValueError(f"invalid resolution kind for {name}: {kind}")
        intent_value = (
            intent.approved_attributes[name]
            if name in intent.approved_attributes
            else intent.approved_abstentions[name]
        )
        superseded_by = _superseding_attribute(name, intent) if kind == "superseded" else ""
        if kind == "superseded" and not superseded_by:
            raise ValueError(f"attribute {name} is superseded without an explicit winner")
        if kind in {"abstained", "superseded", "review_required"}:
            target_value = None
        else:
            if name not in targets:
                raise ValueError(f"materialization target missing for {name}")
            target_value = _canonical_plan_target(
                name, targets[name], rendered_x_height_px=float(rendered_x_height_px)
            )
            if target_value != _deep_freeze(targets[name]):
                unit_resolution[name] = _thaw(target_value)
        provenance = intent.attribute_provenance.get(name, {})
        evidence_ids: Sequence[str]
        if isinstance(provenance, Mapping):
            raw_ids = provenance.get("evidence_ids")
            if isinstance(raw_ids, (list, tuple)):
                evidence_ids = [str(item) for item in raw_ids if str(item)]
            else:
                evidence_id = str(provenance.get("evidence_id") or "")
                evidence_ids = [evidence_id] if evidence_id else []
        else:
            evidence_ids = []
        reason = str(reasons.get(name) or "")
        if kind == "abstained" and not reason:
            reason = str(intent.approved_abstentions.get(name) or "")
        plans[name] = MaterializationAttributePlan(
            name=name,
            domain=ATTRIBUTE_DOMAIN[name],
            intent_value=_deep_freeze(intent_value),
            target_value=target_value,
            resolution_kind=kind,
            reason=reason,
            superseded_by=superseded_by,
            evidence_ids=tuple(sorted(set(evidence_ids))),
        )
    unit_contract = {
        "rendered_x_height_px": float(rendered_x_height_px),
        "resolved": unit_resolution,
    }
    unit_resolution_sha256 = _contract_sha256(unit_contract)
    contract = {
        "schema_version": 1,
        "owner_id": intent.owner_id,
        "page_id": intent.page_id,
        "visual_profile_sha256": intent.visual_profile_sha256,
        "intent_sha256": intent.intent_sha256,
        "render_layout_contract_sha256": _require_sha256(
            render_layout_contract_sha256, field="render_layout_contract_sha256"
        ),
        "rendered_x_height_px": float(rendered_x_height_px),
        "unit_resolution_sha256": unit_resolution_sha256,
        "attribute_plans": {name: plan.to_dict() for name, plan in plans.items()},
    }
    return OwnerStyleMaterializationPlan(
        schema_version=1,
        owner_id=intent.owner_id,
        page_id=intent.page_id,
        visual_profile_sha256=intent.visual_profile_sha256,
        intent_sha256=intent.intent_sha256,
        render_layout_contract_sha256=contract["render_layout_contract_sha256"],
        rendered_x_height_px=float(rendered_x_height_px),
        unit_resolution_sha256=unit_resolution_sha256,
        attribute_plans=MappingProxyType(plans),
        plan_sha256=_contract_sha256(contract),
    )


def _canonical_observed_value(name: str, value: Any) -> Any:
    if name == "font_name":
        if not isinstance(value, Mapping):
            raise ValueError("font observation must contain a resolved identity")
        return _deep_freeze(value)
    return _deep_freeze(canonicalize_style_attribute(name, value))


def build_materialization_observation(
    *,
    plan: OwnerStyleMaterializationPlan,
    domain_observations: Mapping[str, Mapping[str, Mapping[str, Any]]],
    render_completed: bool,
) -> OwnerStyleMaterializationObservation:
    attributes: dict[str, Mapping[str, Any]] = {}
    for raw_domain, rows in domain_observations.items():
        domain = str(raw_domain)
        if domain not in {"layout", "font", "raster"}:
            raise ValueError(f"unsupported observation domain: {domain}")
        if not isinstance(rows, Mapping):
            raise ValueError("domain observations must be mappings")
        for name, raw_row in rows.items():
            if name not in plan.attribute_plans:
                raise ValueError(f"observation contains attribute absent from plan: {name}")
            if plan.attribute_plans[name].domain != domain:
                raise ValueError(f"observation domain mismatch for {name}")
            if name in attributes:
                raise ValueError(f"attribute observed more than once: {name}")
            row = dict(raw_row)
            status = str(row.get("status") or "materialized")
            if status == "unavailable":
                reason = str(row.get("reason") or "")
                if not reason:
                    raise ValueError(f"unavailable observation requires reason: {name}")
                attributes[name] = _deep_freeze(
                    {"status": status, "domain": domain, "reason": reason}
                )
                continue
            evidence_kind = str(row.get("evidence_kind") or "")
            evidence_sha256 = str(row.get("evidence_sha256") or "")
            if not evidence_kind or not evidence_sha256:
                raise ValueError(f"materialized observation requires evidence for {name}")
            _require_sha256(evidence_sha256, field=f"{name}.evidence_sha256")
            value = row.get("canonical_value", row.get("value"))
            canonical_value = _canonical_observed_value(name, value)
            attributes[name] = _deep_freeze(
                {
                    "status": "materialized",
                    "domain": domain,
                    "canonical_value": _thaw(canonical_value),
                    "evidence_kind": evidence_kind,
                    "evidence_sha256": evidence_sha256.lower(),
                    "metrics": row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {},
                }
            )
    contract = {
        "schema_version": 1,
        "owner_id": plan.owner_id,
        "page_id": plan.page_id,
        "visual_profile_sha256": plan.visual_profile_sha256,
        "plan_sha256": plan.plan_sha256,
        "attributes": _thaw(_deep_freeze(attributes)),
        "render_completed": bool(render_completed),
    }
    return OwnerStyleMaterializationObservation(
        schema_version=1,
        owner_id=plan.owner_id,
        page_id=plan.page_id,
        visual_profile_sha256=plan.visual_profile_sha256,
        plan_sha256=plan.plan_sha256,
        attributes=_deep_freeze(attributes),
        render_completed=bool(render_completed),
        observation_sha256=_contract_sha256(contract),
    )


def validate_materialization_observation(
    observation: OwnerStyleMaterializationObservation | Mapping[str, Any],
) -> dict[str, Any]:
    payload = observation.to_dict() if isinstance(observation, OwnerStyleMaterializationObservation) else _thaw(observation)
    required = {
        "schema_version",
        "owner_id",
        "page_id",
        "visual_profile_sha256",
        "plan_sha256",
        "attributes",
        "render_completed",
        "observation_sha256",
    }
    if set(payload) != required:
        raise ValueError("materialization observation has incomplete schema")
    expected_hash = str(payload["observation_sha256"])
    contract = {key: value for key, value in payload.items() if key != "observation_sha256"}
    if _contract_sha256(contract) != expected_hash:
        raise ValueError("materialization observation hash mismatch")
    return payload


def validate_materialization_plan(
    plan: OwnerStyleMaterializationPlan | Mapping[str, Any],
) -> dict[str, Any]:
    payload = plan.to_dict() if isinstance(plan, OwnerStyleMaterializationPlan) else _thaw(plan)
    required = {
        "schema_version",
        "owner_id",
        "page_id",
        "visual_profile_sha256",
        "intent_sha256",
        "render_layout_contract_sha256",
        "rendered_x_height_px",
        "unit_resolution_sha256",
        "attribute_plans",
        "plan_sha256",
    }
    if set(payload) != required:
        raise ValueError("materialization plan has incomplete schema")
    if payload.get("schema_version") != 1:
        raise ValueError("materialization plan schema version is unsupported")
    for field in (
        "visual_profile_sha256",
        "intent_sha256",
        "render_layout_contract_sha256",
        "unit_resolution_sha256",
        "plan_sha256",
    ):
        _require_sha256(str(payload.get(field) or ""), field=field)
    if not isinstance(payload.get("attribute_plans"), dict):
        raise ValueError("materialization plan attribute_plans must be a mapping")
    contract = {key: value for key, value in payload.items() if key != "plan_sha256"}
    if _contract_sha256(contract) != payload["plan_sha256"]:
        raise ValueError("materialization plan hash mismatch")
    return payload


def materialization_plan_from_dict(
    value: OwnerStyleMaterializationPlan | Mapping[str, Any],
) -> OwnerStyleMaterializationPlan:
    payload = validate_materialization_plan(value)
    attribute_plans: dict[str, MaterializationAttributePlan] = {}
    expected_fields = {
        "name",
        "domain",
        "intent_value",
        "target_value",
        "resolution_kind",
        "reason",
        "superseded_by",
        "evidence_ids",
    }
    for name, raw in sorted(payload["attribute_plans"].items()):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ValueError(f"materialization attribute plan schema mismatch: {name}")
        if raw.get("name") != name or raw.get("domain") != ATTRIBUTE_DOMAIN.get(name):
            raise ValueError(f"materialization attribute plan identity/domain mismatch: {name}")
        attribute_plans[name] = MaterializationAttributePlan(
            name=name,
            domain=raw["domain"],
            intent_value=_deep_freeze(raw.get("intent_value")),
            target_value=_deep_freeze(raw.get("target_value")),
            resolution_kind=raw["resolution_kind"],
            reason=str(raw.get("reason") or ""),
            superseded_by=str(raw.get("superseded_by") or ""),
            evidence_ids=tuple(str(item) for item in raw.get("evidence_ids") or ()),
        )
    return OwnerStyleMaterializationPlan(
        schema_version=payload["schema_version"],
        owner_id=str(payload["owner_id"]),
        page_id=str(payload["page_id"]),
        visual_profile_sha256=str(payload["visual_profile_sha256"]),
        intent_sha256=str(payload["intent_sha256"]),
        render_layout_contract_sha256=str(payload["render_layout_contract_sha256"]),
        rendered_x_height_px=float(payload["rendered_x_height_px"]),
        unit_resolution_sha256=str(payload["unit_resolution_sha256"]),
        attribute_plans=MappingProxyType(attribute_plans),
        plan_sha256=str(payload["plan_sha256"]),
    )


def materialization_observation_from_dict(
    value: OwnerStyleMaterializationObservation | Mapping[str, Any],
) -> OwnerStyleMaterializationObservation:
    payload = validate_materialization_observation(value)
    return OwnerStyleMaterializationObservation(
        schema_version=int(payload["schema_version"]),
        owner_id=str(payload["owner_id"]),
        page_id=str(payload["page_id"]),
        visual_profile_sha256=str(payload["visual_profile_sha256"]),
        plan_sha256=str(payload["plan_sha256"]),
        attributes=_deep_freeze(payload["attributes"]),
        render_completed=bool(payload["render_completed"]),
        observation_sha256=str(payload["observation_sha256"]),
    )


def compare_materialization_payloads(
    plan: OwnerStyleMaterializationPlan | Mapping[str, Any],
    observation: OwnerStyleMaterializationObservation | Mapping[str, Any],
) -> MaterializationComparison:
    return compare_materialization(
        materialization_plan_from_dict(plan),
        materialization_observation_from_dict(observation),
    )


def _compare_target(name: str, expected: Any, observed: Any) -> AttributeComparison:
    if name == "font_name":
        expected_payload = _thaw(expected)
        observed_payload = _thaw(observed)
        expected_sha = str(expected_payload.get("file_sha256") or "") if isinstance(expected_payload, Mapping) else ""
        observed_sha = str(observed_payload.get("file_sha256") or "") if isinstance(observed_payload, Mapping) else ""
        matches = bool(expected_sha) and expected_sha == observed_sha
        return AttributeComparison(
            name=name,
            domain="font",
            expected=expected_payload,
            observed=observed_payload,
            matches=matches,
            tolerance={"kind": "resolved_file_sha256"},
            reason="" if matches else "resolved_font_identity_mismatch",
        )
    return compare_style_attribute(name, _thaw(expected), _thaw(observed))


def compare_materialization(
    plan: OwnerStyleMaterializationPlan,
    observation: OwnerStyleMaterializationObservation,
) -> MaterializationComparison:
    validate_materialization_observation(observation)
    mismatches: list[Mapping[str, Any]] = []
    compared: list[str] = []
    review_required = not observation.render_completed
    if observation.plan_sha256 != plan.plan_sha256:
        mismatches.append(
            _deep_freeze(
                {
                    "attribute": "*",
                    "domain": "raster",
                    "reason": "materialization_plan_hash_mismatch",
                }
            )
        )
    for name, attribute_plan in plan.attribute_plans.items():
        row = observation.attributes.get(name)
        if attribute_plan.resolution_kind in {"abstained", "superseded"}:
            if row is not None:
                mismatches.append(
                    _deep_freeze(
                        {
                            "attribute": name,
                            "domain": attribute_plan.domain,
                            "reason": "unexpected_observation_for_non_materialized_attribute",
                        }
                    )
                )
            continue
        if attribute_plan.resolution_kind == "review_required":
            review_required = True
            continue
        compared.append(name)
        if row is None:
            mismatches.append(
                _deep_freeze(
                    {
                        "attribute": name,
                        "domain": attribute_plan.domain,
                        "reason": "missing_domain_observation",
                    }
                )
            )
            continue
        if row.get("status") == "unavailable":
            mismatches.append(
                _deep_freeze(
                    {
                        "attribute": name,
                        "domain": attribute_plan.domain,
                        "reason": str(row.get("reason") or "unavailable"),
                    }
                )
            )
            continue
        comparison = _compare_target(
            name, attribute_plan.target_value, row.get("canonical_value")
        )
        if not comparison.matches:
            mismatches.append(
                _deep_freeze(
                    {
                        "attribute": name,
                        "domain": attribute_plan.domain,
                        "reason": comparison.reason or "canonical_value_mismatch",
                        "expected": comparison.expected,
                        "observed": comparison.observed,
                    }
                )
            )
    status: Literal["match", "mismatch", "review_required"]
    status = "mismatch" if mismatches else "review_required" if review_required else "match"
    return MaterializationComparison(
        status=status,
        mismatches=tuple(mismatches),
        compared_attributes=tuple(compared),
    )
