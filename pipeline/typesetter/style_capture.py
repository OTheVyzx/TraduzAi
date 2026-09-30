"""Immutable owner-scoped source-style capture and eligibility contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from sfx.promotion import VISUAL_PROMOTION_THRESHOLD
from typesetter.style_contract import (
    StyleAttributeEvidenceV2,
    StyleEvidenceV2,
    style_evidence_v2_from_dict,
)
from typesetter.style_extractor import extract_text_style_evidence_v2
from typesetter.font_matcher import FontShapeMatcher, load_font_catalog
from typesetter.style_policy import SOURCE_STYLE_CONFIDENCE_THRESHOLD


OWNER_STYLE_CAPTURE_SCHEMA_VERSION = 1
_SHA256_EMPTY_MASK = hashlib.sha256(b"").hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_confidence(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _source_artifact_sha256(source_rgb: np.ndarray) -> str:
    source = np.ascontiguousarray(source_rgb, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("owner style capture requires an RGB source page")
    header = _canonical_json({"shape": list(source.shape), "dtype": str(source.dtype)})
    return _sha256(header + b"\0" + source.tobytes(order="C"))


def _binary_mask(value: np.ndarray, shape: tuple[int, int], *, label: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape:
        raise ValueError(f"{label} must match the source page")
    return np.ascontiguousarray(np.where(mask > 0, 255, 0), dtype=np.uint8)


def _mask_sha256(mask: np.ndarray) -> str:
    canonical = np.ascontiguousarray(mask, dtype=np.uint8)
    header = _canonical_json({"shape": list(canonical.shape), "dtype": str(canonical.dtype)})
    return _sha256(header + b"\0" + canonical.tobytes(order="C"))


@lru_cache(maxsize=1)
def _runtime_font_matcher() -> FontShapeMatcher:
    repo_root = Path(__file__).resolve().parents[2]
    fonts_dir = repo_root / "fonts"
    return FontShapeMatcher(
        load_font_catalog(fonts_dir, fonts_dir / "font-map.json")
    )


@dataclass(frozen=True)
class StyleCaptureMasks:
    """Canonical page masks used by the pre-inpaint style extractor."""

    glyph_core_mask: np.ndarray
    stroke_ring_mask: np.ndarray
    context_mask: np.ndarray
    effect_region_mask: np.ndarray
    foreign_owner_mask: np.ndarray
    protected_art_mask: np.ndarray
    support_mask: np.ndarray


def build_style_capture_masks(
    graph: object,
    owner_id: str,
    shape: tuple[int, int],
    *,
    glyph_mask: np.ndarray,
    foreign_owner_mask: np.ndarray | None = None,
    protected_art_mask: np.ndarray | None = None,
) -> StyleCaptureMasks:
    """Build non-overlapping owner masks without sampling foreign visual content."""

    owners = [item for item in (_field(graph, "owners", []) or []) if str(_field(item, "owner_id") or "") == owner_id]
    if len(owners) != 1:
        raise ValueError(f"style masks require exactly one owner {owner_id!r}")
    owner = owners[0]
    glyph = _binary_mask(glyph_mask, shape, label="glyph mask")
    if int(np.count_nonzero(glyph)) < 8:
        raise ValueError("glyph mask has insufficient source text evidence")
    component_ids = {str(item) for item in (_field(owner, "component_ids", []) or [])}
    support = np.zeros(shape, dtype=np.uint8)
    for component in (_field(graph, "components", []) or []):
        if str(_field(component, "component_id") or "") not in component_ids:
            continue
        polygon = _field(component, "polygon_page", ()) or ()
        if len(polygon) >= 3:
            points = np.asarray([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
            cv2.fillPoly(support, [points], 255)
        else:
            x1, y1, x2, y2 = (int(item) for item in _field(component, "bbox_page"))
            support[max(0, y1):min(shape[0], y2), max(0, x1):min(shape[1], x2)] = 255
    if int(np.count_nonzero(support)) < 12:
        raise ValueError("owner support mask has insufficient area")
    glyph = cv2.bitwise_and(glyph, support)
    if int(np.count_nonzero(glyph)) < 8:
        raise ValueError("glyph mask has insufficient owner-supported evidence")
    foreign = (
        np.zeros(shape, dtype=np.uint8)
        if foreign_owner_mask is None
        else _binary_mask(foreign_owner_mask, shape, label="foreign owner mask")
    )
    protected = (
        np.zeros(shape, dtype=np.uint8)
        if protected_art_mask is None
        else _binary_mask(protected_art_mask, shape, label="protected art mask")
    )
    forbidden = cv2.bitwise_or(foreign, protected)
    glyph[forbidden > 0] = 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(glyph, 8)
    heights = [
        int(stats[label, cv2.CC_STAT_HEIGHT])
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= 4
    ]
    if not heights:
        raise ValueError("glyph mask has insufficient measurable components")
    x_height = max(1.0, float(np.median(heights)))
    stroke_radius = max(1, int(round(x_height * 0.16)))
    effect_radius = max(stroke_radius + 1, int(round(x_height * 0.48)))
    stroke_dilated = cv2.dilate(
        glyph,
        np.ones((stroke_radius * 2 + 1, stroke_radius * 2 + 1), dtype=np.uint8),
    )
    effect_dilated = cv2.dilate(
        glyph,
        np.ones((effect_radius * 2 + 1, effect_radius * 2 + 1), dtype=np.uint8),
    )
    allowed = (support > 0) & (forbidden == 0)
    stroke_ring = np.where(
        allowed & (stroke_dilated > 0) & (glyph == 0), 255, 0
    ).astype(np.uint8)
    effect_region = np.where(
        allowed & (effect_dilated > 0) & (stroke_dilated == 0), 255, 0
    ).astype(np.uint8)
    context = np.where(
        allowed & (effect_dilated == 0), 255, 0
    ).astype(np.uint8)
    if int(np.count_nonzero(context)) < 12:
        raise ValueError("context mask has insufficient clean background evidence")
    return StyleCaptureMasks(
        glyph_core_mask=np.ascontiguousarray(glyph),
        stroke_ring_mask=np.ascontiguousarray(stroke_ring),
        context_mask=np.ascontiguousarray(context),
        effect_region_mask=np.ascontiguousarray(effect_region),
        foreign_owner_mask=np.ascontiguousarray(foreign),
        protected_art_mask=np.ascontiguousarray(protected),
        support_mask=np.ascontiguousarray(support),
    )


@dataclass(frozen=True)
class OwnerStyleCapture:
    """Source-side style eligibility that never enters semantic translation."""

    schema_version: int
    page_id: str
    owner_id: str
    component_ids: tuple[str, ...]
    route_action: str
    source_artifact_sha256: str
    source_text_sha256: str
    glyph_mask_sha256: str
    context_mask_sha256: str
    effect_region_mask_sha256: str
    candidate_kind: str
    candidate_confidence: float | None
    candidate_confidence_provenance: tuple[str, ...]
    promotion_status: str
    promotion_provenance: tuple[str, ...]
    eligible: bool
    style_evidence_v2: object | None
    font_match_evidence: object | None
    capture_sha256: str

    def to_dict(self) -> dict[str, Any]:
        style_evidence = (
            self.style_evidence_v2.to_dict()
            if isinstance(self.style_evidence_v2, StyleEvidenceV2)
            else self.style_evidence_v2
        )
        return {
            "schema_version": self.schema_version,
            "page_id": self.page_id,
            "owner_id": self.owner_id,
            "component_ids": list(self.component_ids),
            "route_action": self.route_action,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_text_sha256": self.source_text_sha256,
            "glyph_mask_sha256": self.glyph_mask_sha256,
            "context_mask_sha256": self.context_mask_sha256,
            "effect_region_mask_sha256": self.effect_region_mask_sha256,
            "candidate_kind": self.candidate_kind,
            "candidate_confidence": self.candidate_confidence,
            "candidate_confidence_provenance": list(
                self.candidate_confidence_provenance
            ),
            "promotion_status": self.promotion_status,
            "promotion_provenance": list(self.promotion_provenance),
            "eligible": self.eligible,
            "style_evidence_v2": style_evidence,
            "font_match_evidence": self.font_match_evidence,
            "capture_sha256": self.capture_sha256,
        }


def _capture_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("capture_sha256", None)
    return _sha256(_canonical_json(normalized))


def validate_owner_style_capture(
    value: OwnerStyleCapture | Mapping[str, Any],
    *,
    expected_owner_id: str | None = None,
) -> OwnerStyleCapture:
    payload = value.to_dict() if isinstance(value, OwnerStyleCapture) else dict(value)
    if int(payload.get("schema_version") or 0) != OWNER_STYLE_CAPTURE_SCHEMA_VERSION:
        raise ValueError("unsupported owner style capture schema")
    owner_id = str(payload.get("owner_id") or "")
    page_id = str(payload.get("page_id") or "")
    if not owner_id or not page_id or (expected_owner_id and owner_id != expected_owner_id):
        raise ValueError("owner style capture identity mismatch")
    serialized_hash = str(payload.get("capture_sha256") or "")
    if serialized_hash != _capture_sha256(payload):
        raise ValueError("owner style capture hash mismatch")
    confidence = _finite_confidence(payload.get("candidate_confidence"))
    if payload.get("candidate_confidence") is not None and confidence is None:
        raise ValueError("owner style capture confidence is invalid")
    capture = OwnerStyleCapture(
        schema_version=OWNER_STYLE_CAPTURE_SCHEMA_VERSION,
        page_id=page_id,
        owner_id=owner_id,
        component_ids=tuple(sorted(str(item) for item in payload.get("component_ids") or ())),
        route_action=str(payload.get("route_action") or ""),
        source_artifact_sha256=str(payload.get("source_artifact_sha256") or ""),
        source_text_sha256=str(payload.get("source_text_sha256") or ""),
        glyph_mask_sha256=str(payload.get("glyph_mask_sha256") or ""),
        context_mask_sha256=str(payload.get("context_mask_sha256") or ""),
        effect_region_mask_sha256=str(payload.get("effect_region_mask_sha256") or ""),
        candidate_kind=str(payload.get("candidate_kind") or ""),
        candidate_confidence=confidence,
        candidate_confidence_provenance=tuple(
            sorted(str(item) for item in payload.get("candidate_confidence_provenance") or ())
        ),
        promotion_status=str(payload.get("promotion_status") or ""),
        promotion_provenance=tuple(
            sorted(str(item) for item in payload.get("promotion_provenance") or ())
        ),
        eligible=bool(payload.get("eligible")),
        style_evidence_v2=(
            style_evidence_v2_from_dict(payload["style_evidence_v2"])
            if isinstance(payload.get("style_evidence_v2"), Mapping)
            else None
        ),
        font_match_evidence=payload.get("font_match_evidence"),
        capture_sha256=serialized_hash,
    )
    canonical_hash = _capture_sha256(capture.to_dict())
    return replace(capture, capture_sha256=canonical_hash)


def build_owner_style_capture(
    graph: object,
    owner_id: str,
    source_rgb: np.ndarray,
    *,
    promotion_status: str | None = None,
    promotion_confidence: float | None = None,
    promotion_provenance: Iterable[str] = (),
    glyph_mask: np.ndarray | None = None,
    foreign_owner_mask: np.ndarray | None = None,
    protected_art_mask: np.ndarray | None = None,
) -> OwnerStyleCapture:
    """Derive eligibility only from selected source observations and promotion facts."""

    owners = [item for item in (_field(graph, "owners", []) or []) if str(_field(item, "owner_id") or "") == owner_id]
    if len(owners) != 1:
        raise ValueError(f"owner style capture requires exactly one owner {owner_id!r}")
    owner = owners[0]
    selected_ids = tuple(
        sorted(str(item) for item in (_field(owner, "selected_observation_ids", []) or []))
    )
    observations_by_id = {
        str(_field(item, "observation_id") or ""): item
        for item in (_field(graph, "observations", []) or [])
    }
    selected = [observations_by_id[item] for item in selected_ids if item in observations_by_id]
    confidences = [_finite_confidence(_field(item, "confidence")) for item in selected]
    primary_confidence = (
        max(item for item in confidences if item is not None)
        if selected and all(item is not None for item in confidences)
        else None
    )
    route_action = str(_field(owner, "route_action") or "")
    is_sfx = route_action == "translate_sfx_inpaint_render"
    provenance = tuple(sorted({str(item) for item in promotion_provenance if str(item)}))
    normalized_promotion_status = str(
        promotion_status or ("not_promoted" if is_sfx else "not_applicable")
    )
    if is_sfx:
        candidate_kind = "promoted_sfx" if normalized_promotion_status == "promoted" else "raw_sfx"
        candidate_confidence = _finite_confidence(promotion_confidence)
        eligible = bool(
            normalized_promotion_status == "promoted"
            and provenance
            and candidate_confidence is not None
            and candidate_confidence >= VISUAL_PROMOTION_THRESHOLD
        )
    else:
        candidate_kind = "primary_ocr"
        candidate_confidence = primary_confidence
        eligible = bool(
            route_action == "translate_inpaint_render"
            and candidate_confidence is not None
            and candidate_confidence >= SOURCE_STYLE_CONFIDENCE_THRESHOLD
        )
    masks = None
    style_evidence = None
    font_match_evidence = None
    if glyph_mask is not None:
        masks = build_style_capture_masks(
            graph,
            owner_id,
            tuple(np.asarray(source_rgb).shape[:2]),
            glyph_mask=glyph_mask,
            foreign_owner_mask=foreign_owner_mask,
            protected_art_mask=protected_art_mask,
        )
        style_evidence = extract_text_style_evidence_v2(
            source_rgb,
            masks.glyph_core_mask,
            masks.context_mask,
            stroke_ring_mask=masks.stroke_ring_mask,
            effect_region_mask=masks.effect_region_mask,
            owner_id=owner_id,
            semantic_role=str(_field(owner, "semantic_role") or "text"),
            source_phase="pre_inpaint",
        )
        geometry_profile = {
            name: style_evidence.attributes[name].value
            for name in (
                "rotation_deg",
                "slant_tangent",
                "width_scale",
                "scale_y",
            )
            if name in style_evidence.attributes
            and style_evidence.attributes[name].value != "unknown"
            and style_evidence.attributes[name].confidence >= 0.7
        }
        font_result = _runtime_font_matcher().match(
            masks.glyph_core_mask,
            source_text=str(_field(owner, "source_payload") or ""),
            profile=geometry_profile,
            semantic_role=str(_field(owner, "semantic_role") or "text"),
        )
        font_match_evidence = font_result.to_dict()
        font_match_evidence["source_text_sha256"] = _sha256(
            str(_field(owner, "source_payload") or "").encode("utf-8")
        )
        font_match_evidence["glyph_mask_sha256"] = _mask_sha256(
            masks.glyph_core_mask
        )
        attributes = dict(style_evidence.attributes)
        if font_result.selected_font is not None:
            attributes["font_name"] = StyleAttributeEvidenceV2(
                value=font_result.selected_font,
                confidence=font_result.confidence,
                top_k=tuple(
                    str(item["font_name"])
                    for item in font_result.top_k
                    if item.get("font_name")
                ),
                margin=font_result.margin,
            )
        else:
            attributes["font_name"] = StyleAttributeEvidenceV2(
                value="unknown",
                confidence=0.0,
                top_k=tuple(
                    str(item["font_name"])
                    for item in font_result.top_k
                    if item.get("font_name")
                ),
                margin=font_result.margin,
                abstention_reason=font_result.abstention_reason,
            )
        provenance = dict(style_evidence.attribute_provenance)
        provenance["font_name"] = {
            "catalog_version": font_result.catalog_version,
            "cache_key": font_result.cache_key,
            "source": "frozen_source_text_and_owner_glyph_mask",
        }
        style_evidence = replace(
            style_evidence,
            attributes=attributes,
            attribute_provenance=provenance,
        )
        style_evidence = style_evidence_v2_from_dict(style_evidence.to_dict())
    payload: dict[str, Any] = {
        "schema_version": OWNER_STYLE_CAPTURE_SCHEMA_VERSION,
        "page_id": str(_field(owner, "page_id") or _field(graph, "page_id") or ""),
        "owner_id": owner_id,
        "component_ids": sorted(str(item) for item in (_field(owner, "component_ids", []) or [])),
        "route_action": route_action,
        "source_artifact_sha256": _source_artifact_sha256(source_rgb),
        "source_text_sha256": _sha256(str(_field(owner, "source_payload") or "").encode("utf-8")),
        "glyph_mask_sha256": (
            _mask_sha256(masks.glyph_core_mask) if masks is not None else _SHA256_EMPTY_MASK
        ),
        "context_mask_sha256": (
            _mask_sha256(masks.context_mask) if masks is not None else _SHA256_EMPTY_MASK
        ),
        "effect_region_mask_sha256": (
            _mask_sha256(masks.effect_region_mask) if masks is not None else _SHA256_EMPTY_MASK
        ),
        "candidate_kind": candidate_kind,
        "candidate_confidence": candidate_confidence,
        "candidate_confidence_provenance": list(selected_ids),
        "promotion_status": normalized_promotion_status,
        "promotion_provenance": list(provenance),
        "eligible": eligible,
        "style_evidence_v2": style_evidence.to_dict() if style_evidence is not None else None,
        "font_match_evidence": font_match_evidence,
    }
    payload["capture_sha256"] = _capture_sha256(payload)
    return validate_owner_style_capture(payload, expected_owner_id=owner_id)


def build_owner_style_captures(
    graph: object,
    source_rgb: np.ndarray,
    *,
    promotions_by_owner: Mapping[str, Mapping[str, Any]] | None = None,
    glyph_masks_by_owner: Mapping[str, np.ndarray] | None = None,
    protected_art_masks_by_owner: Mapping[str, np.ndarray] | None = None,
) -> dict[str, OwnerStyleCapture]:
    """Capture all renderable owners in stable owner-id order."""

    captures: dict[str, OwnerStyleCapture] = {}
    promotions = promotions_by_owner or {}
    for owner in sorted(
        (_field(graph, "owners", []) or []),
        key=lambda item: str(_field(item, "owner_id") or ""),
    ):
        if str(_field(owner, "disposition") or "") != "owned":
            continue
        route_action = str(_field(owner, "route_action") or "")
        if route_action not in {"translate_inpaint_render", "translate_sfx_inpaint_render"}:
            continue
        owner_id = str(_field(owner, "owner_id") or "")
        promotion = promotions.get(owner_id) or {}
        foreign_mask = np.zeros(np.asarray(source_rgb).shape[:2], dtype=np.uint8)
        for foreign_owner_id, foreign_glyph in (glyph_masks_by_owner or {}).items():
            if foreign_owner_id != owner_id:
                foreign_mask = np.maximum(
                    foreign_mask,
                    _binary_mask(
                        foreign_glyph,
                        tuple(foreign_mask.shape),
                        label="foreign owner glyph mask",
                    ),
                )
        kwargs = {
            "promotion_status": promotion.get("promotion_status"),
            "promotion_confidence": promotion.get("promotion_confidence"),
            "promotion_provenance": promotion.get("promotion_provenance") or (),
        }
        try:
            captures[owner_id] = build_owner_style_capture(
                graph,
                owner_id,
                source_rgb,
                **kwargs,
                glyph_mask=(glyph_masks_by_owner or {}).get(owner_id),
                foreign_owner_mask=foreign_mask,
                protected_art_mask=(protected_art_masks_by_owner or {}).get(owner_id),
            )
        except ValueError as exc:
            if "insufficient" not in str(exc):
                raise
            captures[owner_id] = build_owner_style_capture(
                graph,
                owner_id,
                source_rgb,
                **kwargs,
            )
    return captures


__all__ = [
    "OWNER_STYLE_CAPTURE_SCHEMA_VERSION",
    "OwnerStyleCapture",
    "StyleCaptureMasks",
    "build_owner_style_capture",
    "build_owner_style_captures",
    "build_style_capture_masks",
    "validate_owner_style_capture",
]
