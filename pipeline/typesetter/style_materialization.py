"""Canonical contracts shared by style materialization producers.

This module owns value normalization and comparison.  Domain-specific
observers may only claim attributes registered to their own domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Literal, Mapping

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
    if number == 0:
        return 0.0
    return int(number) if isinstance(value, int) else number


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


def canonicalize_style_attribute(name: str, value: Any) -> Any:
    validate_attribute_domain_registry(ATTRIBUTE_DOMAIN)
    if name not in ATTRIBUTE_DOMAIN:
        raise ValueError(f"unknown style attribute: {name}")
    if name == "font_name":
        raise ValueError("font_name requires resolved font identity")
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
    matches = canonical_expected == canonical_observed
    return AttributeComparison(
        name=name,
        domain=domain,
        expected=canonical_expected,
        observed=canonical_observed,
        matches=matches,
        tolerance={"kind": "canonical_exact"},
        delta_e_2000=color_distance,
        reason="" if matches else "canonical_value_mismatch",
    )


validate_attribute_domain_registry(ATTRIBUTE_DOMAIN)
