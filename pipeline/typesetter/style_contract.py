"""Versioned shadow contract for source-style evidence.

The v2 contract intentionally coexists with ``TextStyleEvidence``. It is a
debug/measurement surface until a later rollout phase explicitly enables it in
the renderer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping


STYLE_V2_ATTRIBUTE_NAMES = (
    "font_name",
    "font_weight",
    "font_width",
    "tracking_xh",
    "slant_tangent",
    "width_scale",
    "scale_y",
    "font_size_px",
    "alignment",
    "fill",
    "stroke",
    "multistroke",
    "shadow",
    "glow",
    "gradient",
    "curve",
    "rotation_deg",
    "container",
)
STYLE_V2_ATTRIBUTE_NAME_SET = frozenset(STYLE_V2_ATTRIBUTE_NAMES)


@dataclass(frozen=True)
class StyleAttributeEvidenceV2:
    value: Any
    confidence: float
    top_k: tuple[Any, ...]
    margin: float
    abstention_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["top_k"] = list(self.top_k)
        return payload


@dataclass(frozen=True)
class StyleEvidenceV2:
    source: str
    text_present: bool
    attributes: dict[str, StyleAttributeEvidenceV2]
    schema_version: int = 2
    source_sha256: str = ""
    attribute_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "attributes": {name: attribute.to_dict() for name, attribute in self.attributes.items()},
            "schema_version": self.schema_version,
            "source": self.source,
            "text_present": self.text_present,
        }
        if self.source_sha256:
            payload["source_sha256"] = self.source_sha256
        if self.attribute_provenance:
            payload["attribute_provenance"] = self.attribute_provenance
        return payload

    def measurement_attributes(self) -> dict[str, dict[str, Any]]:
        return {name: attribute.to_dict() for name, attribute in self.attributes.items()}


StyleCopyStatus = Literal["applied", "fallback", "not_applicable", "review_required"]


@dataclass(frozen=True)
class StyleApplicationDecisionV2:
    status: StyleCopyStatus
    applied_attributes: Mapping[str, Any]
    abstained_attributes: Mapping[str, str]
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "applied_attributes": dict(self.applied_attributes),
            "abstained_attributes": dict(self.abstained_attributes),
            "evidence_sha256": self.evidence_sha256,
        }


def style_evidence_v2_sha256(evidence: StyleEvidenceV2) -> str:
    canonical = json.dumps(
        evidence.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def style_evidence_v2_from_dict(payload: Mapping[str, Any]) -> StyleEvidenceV2:
    """Migrate serialized v2 evidence while preserving its source identity."""

    raw_attributes = payload.get("attributes")
    raw_attributes = raw_attributes if isinstance(raw_attributes, Mapping) else {}
    attributes: dict[str, StyleAttributeEvidenceV2] = {}
    for name in STYLE_V2_ATTRIBUTE_NAMES:
        raw = raw_attributes.get(name)
        if not isinstance(raw, Mapping):
            attributes[name] = _unknown("missing_attribute_evidence")
            continue
        top_k = raw.get("top_k")
        if not isinstance(top_k, (list, tuple)):
            top_k = ()
        attributes[name] = StyleAttributeEvidenceV2(
            value=raw.get("value", "unknown"),
            confidence=_bounded_confidence(raw.get("confidence")),
            top_k=tuple(top_k),
            margin=_bounded_confidence(raw.get("margin")),
            abstention_reason=str(raw.get("abstention_reason") or ""),
        )
    raw_provenance = payload.get("attribute_provenance")
    provenance = {
        str(name): dict(value)
        for name, value in raw_provenance.items()
        if isinstance(value, Mapping)
    } if isinstance(raw_provenance, Mapping) else {}
    return StyleEvidenceV2(
        source=str(payload.get("source") or "none"),
        text_present=bool(payload.get("text_present")),
        attributes=attributes,
        schema_version=2,
        source_sha256=str(payload.get("source_sha256") or ""),
        attribute_provenance=provenance,
    )


def _bounded_confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, round(float(value), 4)))
    except (TypeError, ValueError):
        return 0.0


def _unknown(reason: str) -> StyleAttributeEvidenceV2:
    return StyleAttributeEvidenceV2(
        value="unknown",
        confidence=0.0,
        top_k=(),
        margin=0.0,
        abstention_reason=reason,
    )


def _observed(value: Any, confidence: object) -> StyleAttributeEvidenceV2:
    score = _bounded_confidence(confidence)
    return StyleAttributeEvidenceV2(
        value=value,
        confidence=score,
        top_k=(value,),
        margin=score,
    )


def _effect(
    *,
    detected: object,
    confidence: object,
    value: Any,
) -> StyleAttributeEvidenceV2:
    score = _bounded_confidence(confidence)
    if bool(detected) and score > 0.0:
        return _observed(value, score)
    return _unknown("insufficient_effect_confidence")


def style_evidence_v2_from_v1(
    v1_evidence: dict[str, Any],
    *,
    font_match: dict[str, Any] | None = None,
) -> StyleEvidenceV2:
    """Adapt legacy evidence without claiming default/fallback values were observed."""
    source = str(v1_evidence.get("source") or "none")
    text_present = source not in {"", "none"}
    if not text_present:
        return StyleEvidenceV2(
            source=source,
            text_present=False,
            attributes={name: _unknown("no_text_evidence") for name in STYLE_V2_ATTRIBUTE_NAMES},
        )

    text_color = str(v1_evidence.get("text_color") or "")
    fill = _observed(text_color, v1_evidence.get("text_color_confidence")) if text_color else _unknown("no_fill_evidence")
    font_name = str(v1_evidence.get("font_name") or "")
    font = _observed(font_name, v1_evidence.get("font_confidence")) if font_name else _unknown("no_font_evidence")
    if isinstance(font_match, dict):
        matched_value = str(font_match.get("value") or "")
        if matched_value and matched_value != "unknown":
            ranked = [item for item in font_match.get("top_k", []) if isinstance(item, dict)]
            font = StyleAttributeEvidenceV2(
                value=matched_value,
                confidence=_bounded_confidence(font_match.get("confidence")),
                top_k=tuple(str(item.get("font_name")) for item in ranked if item.get("font_name")),
                margin=_bounded_confidence(font_match.get("margin")),
                abstention_reason=str(font_match.get("abstention_reason") or ""),
            )
        elif font_match.get("abstention_reason"):
            font = _unknown(str(font_match["abstention_reason"]))
    stroke_color = str(v1_evidence.get("stroke_color") or "")
    stroke_width = int(v1_evidence.get("stroke_width_px") or 0)
    stroke = (
        _observed({"color": stroke_color, "width_px": stroke_width}, v1_evidence.get("stroke_confidence"))
        if stroke_color and stroke_width > 0
        else _unknown("insufficient_stroke_confidence")
    )
    glow = _effect(
        detected=v1_evidence.get("glow"),
        confidence=v1_evidence.get("glow_confidence"),
        value={
            "color": str(v1_evidence.get("glow_color") or ""),
            "width_px": int(v1_evidence.get("glow_px") or 0),
        },
    )
    shadow = _effect(
        detected=v1_evidence.get("shadow"),
        confidence=v1_evidence.get("shadow_confidence"),
        value={
            "color": str(v1_evidence.get("shadow_color") or ""),
            "offset": list(v1_evidence.get("shadow_offset") or [2, 2])[:2],
        },
    )
    gradient = _effect(
        detected=v1_evidence.get("gradient"),
        confidence=v1_evidence.get("gradient_confidence"),
        value=list(v1_evidence.get("gradient_colors") or [])[:2],
    )
    curve = _effect(
        detected=v1_evidence.get("curved"),
        confidence=v1_evidence.get("curve_confidence"),
        value={
            "direction": str(v1_evidence.get("curve_direction") or "arc_up"),
            "amount": float(v1_evidence.get("curve_amount") or 0.0),
        },
    )

    attributes = {
        "font_name": font,
        "font_weight": _unknown("legacy_v1_does_not_measure_font_weight"),
        "font_width": _unknown("legacy_v1_does_not_measure_font_width"),
        "tracking_xh": _unknown("legacy_v1_does_not_measure_tracking"),
        "slant_tangent": _unknown("legacy_v1_does_not_measure_slant"),
        "width_scale": _unknown("legacy_v1_does_not_measure_width_scale"),
        "scale_y": _unknown("legacy_v1_does_not_measure_vertical_scale"),
        "font_size_px": _unknown("legacy_v1_does_not_measure_font_size"),
        "alignment": _unknown("legacy_v1_does_not_measure_alignment"),
        "fill": fill,
        "stroke": stroke,
        "multistroke": _unknown("legacy_v1_does_not_measure_multistroke"),
        "shadow": shadow,
        "glow": glow,
        "gradient": gradient,
        "curve": curve,
        "rotation_deg": _unknown("legacy_v1_does_not_measure_rotation"),
        "container": _unknown("legacy_v1_does_not_measure_container"),
    }
    return StyleEvidenceV2(source=source, text_present=True, attributes=attributes)
