"""Serializable page-global ownership model and fail-closed invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


BBox = tuple[int, int, int, int]
Point = tuple[int, int]

FINAL_COMPONENT_DECISIONS = frozenset({"owned", "preserve", "suppress", "review"})
POST_TRANSLATION_STATES = frozenset(
    {"translated", "mask_ready", "inpainted", "laid_out", "rendered", "verified"}
)
EXECUTOR_REQUIRED_STATES = POST_TRANSLATION_STATES | frozenset({"execution_planned"})


@dataclass(frozen=True)
class SourceTextComponent:
    """Visual evidence that text exists, independent of any OCR payload."""

    component_id: str
    page_id: str
    bbox_page: BBox
    polygon_page: tuple[Point, ...]
    detector_sources: tuple[str, ...]


@dataclass(frozen=True)
class TextObservation:
    """One immutable OCR reading candidate in page coordinates."""

    observation_id: str
    page_id: str
    component_ids: tuple[str, ...]
    text: str
    confidence: float
    provider: str
    bbox_page: BBox
    tile_provenance: tuple[str, ...] = ()
    coverage_score: float | None = None
    language_score: float | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class ComponentDisposition:
    """Final, auditable decision for one source component."""

    component_id: str
    decision: str
    owner_id: str | None = None
    reason: str | None = None


@dataclass
class TextOwner:
    """Single semantic payload authorized to move through the visual pipeline."""

    owner_id: str
    page_id: str
    component_ids: list[str]
    observation_ids: list[str]
    selected_observation_ids: list[str]
    semantic_role: str
    source_payload: str
    translated_payload: str | None
    disposition: str
    state: str
    route_action: str
    execution_tile_id: str | None


@dataclass(frozen=True)
class OwnerProjection:
    """Owner geometry projected into a tile without changing its identity."""

    owner_id: str
    tile_id: str
    role: str
    bbox_page: BBox
    bbox_tile: BBox
    offset_xy: Point


@dataclass(frozen=True)
class OwnerViolation:
    """Stable invariant failure with explicit offending identities."""

    code: str
    severity: str
    message: str
    offenders: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "offenders": list(self.offenders),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OwnerViolation":
        return cls(
            code=str(data.get("code") or ""),
            severity=str(data.get("severity") or "critical"),
            message=str(data.get("message") or ""),
            offenders=tuple(str(value) for value in data.get("offenders") or ()),
        )


class OwnerGraphValidationError(ValueError):
    """Raised when an owner graph cannot safely enter an execution stage."""

    def __init__(self, violations: Iterable[OwnerViolation]):
        self.violations = tuple(violations)
        codes = ", ".join(violation.code for violation in self.violations)
        super().__init__(f"owner graph validation failed: {codes}")


@dataclass
class OwnerGraph:
    """Authoritative page-level graph shared by all downstream stages."""

    schema_version: int
    page_id: str
    components: list[SourceTextComponent]
    observations: list[TextObservation]
    owners: list[TextOwner]
    projections: list[OwnerProjection]
    component_dispositions: list[ComponentDisposition] = field(default_factory=list)
    violations: list[OwnerViolation] = field(default_factory=list)

    def validate(self) -> list[OwnerViolation]:
        violations = list(self.violations)
        component_ids = {component.component_id for component in self.components}
        observation_ids = {observation.observation_id for observation in self.observations}
        owners_by_id = {owner.owner_id: owner for owner in self.owners}

        dispositions_by_component: dict[str, list[ComponentDisposition]] = {}
        for disposition in self.component_dispositions:
            dispositions_by_component.setdefault(disposition.component_id, []).append(disposition)
            if disposition.component_id not in component_ids:
                violations.append(
                    _violation(
                        "disposition_component_unknown",
                        "Disposition references an unknown source component.",
                        disposition.component_id,
                    )
                )
            if disposition.decision not in FINAL_COMPONENT_DECISIONS:
                violations.append(
                    _violation(
                        "component_disposition_invalid",
                        "Source component has an unsupported final disposition.",
                        disposition.component_id,
                    )
                )

        for component_id in sorted(component_ids):
            decisions = dispositions_by_component.get(component_id, [])
            if not decisions:
                violations.append(
                    _violation(
                        "source_text_unowned",
                        "Source text component has no final disposition.",
                        component_id,
                    )
                )
            elif len(decisions) > 1:
                violations.append(
                    _violation(
                        "component_disposition_conflict",
                        "Source text component has multiple final dispositions.",
                        component_id,
                    )
                )

        active_owners_by_component: dict[str, list[str]] = {}
        for owner in self.owners:
            for component_id in owner.component_ids:
                if component_id not in component_ids:
                    violations.append(
                        _violation(
                            "owner_component_unknown",
                            "Owner references an unknown source component.",
                            owner.owner_id,
                            component_id,
                        )
                    )
                if owner.disposition == "owned":
                    active_owners_by_component.setdefault(component_id, []).append(owner.owner_id)

            unknown_observations = sorted(set(owner.observation_ids) - observation_ids)
            if unknown_observations:
                violations.append(
                    _violation(
                        "owner_observation_unknown",
                        "Owner references observations that are not present in the graph.",
                        owner.owner_id,
                        *unknown_observations,
                    )
                )

            unknown_selected = sorted(
                set(owner.selected_observation_ids) - set(owner.observation_ids)
            )
            if unknown_selected:
                violations.append(
                    _violation(
                        "owner_selected_observation_unknown",
                        "Selected observation is not part of the owner evidence.",
                        owner.owner_id,
                        *unknown_selected,
                    )
                )

            owner_projections = [
                projection for projection in self.projections if projection.owner_id == owner.owner_id
            ]
            executors = [projection for projection in owner_projections if projection.role == "executor"]

            if owner.disposition == "owned":
                if not isinstance(owner.source_payload, str) or not owner.source_payload.strip():
                    code = (
                        "semantic_body_split"
                        if owner.semantic_role.endswith("body")
                        and not isinstance(owner.source_payload, str)
                        else "owner_source_payload_missing"
                    )
                    violations.append(
                        _violation(
                            code,
                            "Translatable owner must expose one non-empty source payload.",
                            owner.owner_id,
                        )
                    )
                if (
                    owner.state in POST_TRANSLATION_STATES
                    and (
                        not isinstance(owner.translated_payload, str)
                        or not owner.translated_payload.strip()
                    )
                ):
                    violations.append(
                        _violation(
                            "owner_translation_payload_missing",
                            "Translated owner state requires one non-empty translated payload.",
                            owner.owner_id,
                        )
                    )
                if owner.state in EXECUTOR_REQUIRED_STATES and not executors:
                    violations.append(
                        _violation(
                            "owner_executor_missing",
                            "Active owner has no executor projection.",
                            owner.owner_id,
                        )
                    )
                elif len(executors) > 1:
                    violations.append(
                        _violation(
                            "owner_executor_duplicated",
                            "Active owner has more than one executor projection.",
                            owner.owner_id,
                            *(projection.tile_id for projection in executors),
                        )
                    )
                elif executors and owner.execution_tile_id != executors[0].tile_id:
                    violations.append(
                        _violation(
                            "owner_executor_identity_mismatch",
                            "Owner execution tile does not match its executor projection.",
                            owner.owner_id,
                            executors[0].tile_id,
                        )
                    )

            if (
                owner.disposition == "review" or owner.state == "review_required"
            ) and ("render" in owner.route_action or bool(executors)):
                violations.append(
                    _violation(
                        "review_owner_in_render_plan",
                        "Review-required owner cannot enter the render plan.",
                        owner.owner_id,
                    )
                )

        for component_id, owner_ids in sorted(active_owners_by_component.items()):
            unique_owner_ids = sorted(set(owner_ids))
            if len(unique_owner_ids) > 1:
                violations.append(
                    _violation(
                        "component_multiple_active_owners",
                        "Source component belongs to multiple active owners.",
                        component_id,
                        *unique_owner_ids,
                    )
                )

        for disposition in self.component_dispositions:
            if disposition.decision in {"owned", "review"}:
                if not disposition.owner_id or disposition.owner_id not in owners_by_id:
                    violations.append(
                        _violation(
                            "disposition_owner_missing",
                            "Owned or reviewed source component must reference a graph owner.",
                            disposition.component_id,
                        )
                    )

        return _dedupe_and_sort_violations(violations)

    def require_valid(self) -> None:
        violations = self.validate()
        if any(violation.severity == "critical" for violation in violations):
            raise OwnerGraphValidationError(violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "page_id": self.page_id,
            "components": [
                {
                    "component_id": component.component_id,
                    "page_id": component.page_id,
                    "bbox_page": list(component.bbox_page),
                    "polygon_page": [list(point) for point in component.polygon_page],
                    "detector_sources": list(component.detector_sources),
                }
                for component in sorted(self.components, key=lambda item: item.component_id)
            ],
            "observations": [
                {
                    "observation_id": observation.observation_id,
                    "page_id": observation.page_id,
                    "component_ids": list(observation.component_ids),
                    "text": observation.text,
                    "confidence": float(observation.confidence),
                    "provider": observation.provider,
                    "bbox_page": list(observation.bbox_page),
                    "tile_provenance": list(observation.tile_provenance),
                    "coverage_score": observation.coverage_score,
                    "language_score": observation.language_score,
                    "rejection_reason": observation.rejection_reason,
                }
                for observation in sorted(
                    self.observations, key=lambda item: item.observation_id
                )
            ],
            "owners": [
                {
                    "owner_id": owner.owner_id,
                    "page_id": owner.page_id,
                    "component_ids": sorted(owner.component_ids),
                    "observation_ids": sorted(owner.observation_ids),
                    "selected_observation_ids": sorted(owner.selected_observation_ids),
                    "semantic_role": owner.semantic_role,
                    "source_payload": owner.source_payload,
                    "translated_payload": owner.translated_payload,
                    "disposition": owner.disposition,
                    "state": owner.state,
                    "route_action": owner.route_action,
                    "execution_tile_id": owner.execution_tile_id,
                }
                for owner in sorted(self.owners, key=lambda item: item.owner_id)
            ],
            "projections": [
                {
                    "owner_id": projection.owner_id,
                    "tile_id": projection.tile_id,
                    "role": projection.role,
                    "bbox_page": list(projection.bbox_page),
                    "bbox_tile": list(projection.bbox_tile),
                    "offset_xy": list(projection.offset_xy),
                }
                for projection in sorted(
                    self.projections, key=lambda item: (item.owner_id, item.tile_id, item.role)
                )
            ],
            "component_dispositions": [
                {
                    "component_id": disposition.component_id,
                    "decision": disposition.decision,
                    "owner_id": disposition.owner_id,
                    "reason": disposition.reason,
                }
                for disposition in sorted(
                    self.component_dispositions,
                    key=lambda item: (item.component_id, item.decision, item.owner_id or ""),
                )
            ],
            "violations": [
                violation.to_dict()
                for violation in _dedupe_and_sort_violations(self.violations)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OwnerGraph":
        return cls(
            schema_version=int(data.get("schema_version") or 1),
            page_id=str(data.get("page_id") or ""),
            components=[
                SourceTextComponent(
                    component_id=str(item.get("component_id") or ""),
                    page_id=str(item.get("page_id") or ""),
                    bbox_page=_bbox(item.get("bbox_page")),
                    polygon_page=_polygon(item.get("polygon_page")),
                    detector_sources=tuple(
                        str(value) for value in item.get("detector_sources") or ()
                    ),
                )
                for item in data.get("components") or ()
            ],
            observations=[
                TextObservation(
                    observation_id=str(item.get("observation_id") or ""),
                    page_id=str(item.get("page_id") or ""),
                    component_ids=tuple(
                        str(value) for value in item.get("component_ids") or ()
                    ),
                    text=str(item.get("text") or ""),
                    confidence=float(item.get("confidence") or 0.0),
                    provider=str(item.get("provider") or ""),
                    bbox_page=_bbox(item.get("bbox_page")),
                    tile_provenance=tuple(
                        str(value) for value in item.get("tile_provenance") or ()
                    ),
                    coverage_score=_optional_float(item.get("coverage_score")),
                    language_score=_optional_float(item.get("language_score")),
                    rejection_reason=(
                        str(item["rejection_reason"])
                        if item.get("rejection_reason") is not None
                        else None
                    ),
                )
                for item in data.get("observations") or ()
            ],
            owners=[
                TextOwner(
                    owner_id=str(item.get("owner_id") or ""),
                    page_id=str(item.get("page_id") or ""),
                    component_ids=[
                        str(value) for value in item.get("component_ids") or ()
                    ],
                    observation_ids=[
                        str(value) for value in item.get("observation_ids") or ()
                    ],
                    selected_observation_ids=[
                        str(value)
                        for value in item.get("selected_observation_ids") or ()
                    ],
                    semantic_role=str(item.get("semantic_role") or ""),
                    source_payload=item.get("source_payload"),
                    translated_payload=(
                        str(item["translated_payload"])
                        if item.get("translated_payload") is not None
                        else None
                    ),
                    disposition=str(item.get("disposition") or ""),
                    state=str(item.get("state") or ""),
                    route_action=str(item.get("route_action") or ""),
                    execution_tile_id=(
                        str(item["execution_tile_id"])
                        if item.get("execution_tile_id") is not None
                        else None
                    ),
                )
                for item in data.get("owners") or ()
            ],
            projections=[
                OwnerProjection(
                    owner_id=str(item.get("owner_id") or ""),
                    tile_id=str(item.get("tile_id") or ""),
                    role=str(item.get("role") or ""),
                    bbox_page=_bbox(item.get("bbox_page")),
                    bbox_tile=_bbox(item.get("bbox_tile")),
                    offset_xy=_point(item.get("offset_xy")),
                )
                for item in data.get("projections") or ()
            ],
            component_dispositions=[
                ComponentDisposition(
                    component_id=str(item.get("component_id") or ""),
                    decision=str(item.get("decision") or ""),
                    owner_id=(
                        str(item["owner_id"])
                        if item.get("owner_id") is not None
                        else None
                    ),
                    reason=(
                        str(item["reason"]) if item.get("reason") is not None else None
                    ),
                )
                for item in data.get("component_dispositions") or ()
            ],
            violations=[
                OwnerViolation.from_dict(item) for item in data.get("violations") or ()
            ],
        )


def _violation(code: str, message: str, *offenders: str) -> OwnerViolation:
    return OwnerViolation(
        code=code,
        severity="critical",
        message=message,
        offenders=tuple(str(value) for value in offenders if value),
    )


def _dedupe_and_sort_violations(
    violations: Iterable[OwnerViolation],
) -> list[OwnerViolation]:
    unique = {
        (item.code, item.severity, item.message, item.offenders): item
        for item in violations
    }
    return sorted(
        unique.values(),
        key=lambda item: (item.code, item.offenders, item.message),
    )


def _bbox(value: Any) -> BBox:
    values = list(value or ())
    if len(values) != 4:
        raise ValueError(f"bbox must have four values, got {values!r}")
    return tuple(int(item) for item in values)  # type: ignore[return-value]


def _point(value: Any) -> Point:
    values = list(value or ())
    if len(values) != 2:
        raise ValueError(f"point must have two values, got {values!r}")
    return int(values[0]), int(values[1])


def _polygon(value: Any) -> tuple[Point, ...]:
    return tuple(_point(point) for point in value or ())


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
