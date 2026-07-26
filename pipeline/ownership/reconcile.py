"""Deterministic reconciliation of page-global text evidence into owners."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Iterable, Sequence

from .model import (
    BBox,
    ComponentDisposition,
    OwnerGraph,
    OwnerViolation,
    SourceTextComponent,
    TextObservation,
    TextOwner,
)


@dataclass(frozen=True)
class SemanticRegion:
    """A semantic grouping decided from page geometry and layout evidence."""

    region_id: str
    component_ids: tuple[str, ...]
    semantic_role: str
    disposition: str = "owned"
    reason: str | None = None


@dataclass(frozen=True)
class _ResolvedRegion:
    region_id: str
    components: tuple[SourceTextComponent, ...]
    semantic_role: str
    disposition: str
    reason: str | None


def _scope_token(value: str) -> str:
    match = re.fullmatch(r"page[_-]?(\d+)", str(value).strip().lower())
    if match:
        return f"p{int(match.group(1)):03d}"
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "unknown"


def _stable_owner_id(page_id: str, semantic_role: str, component_ids: Sequence[str]) -> str:
    payload = json.dumps(
        {
            "component_ids": sorted(str(value) for value in component_ids),
            "page_id": str(page_id),
            "semantic_role": str(semantic_role),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"owner_{_scope_token(page_id)}_{digest}"


def _component_order(component: SourceTextComponent) -> tuple[int, int, int, int, str]:
    x1, y1, x2, y2 = component.bbox_page
    return y1, x1, y2, x2, component.component_id


def _observation_order(observation: TextObservation) -> tuple[int, int, int, int, str]:
    x1, y1, x2, y2 = observation.bbox_page
    return y1, x1, y2, x2, observation.observation_id


def _bbox_union(boxes: Iterable[BBox]) -> BBox:
    materialised = list(boxes)
    if not materialised:
        return 0, 0, 0, 0
    return (
        min(box[0] for box in materialised),
        min(box[1] for box in materialised),
        max(box[2] for box in materialised),
        max(box[3] for box in materialised),
    )


def _bbox_area(box: BBox) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _bbox_iou(left: BBox, right: BBox) -> float:
    intersection = (
        max(0, min(left[2], right[2]) - max(left[0], right[0]))
        * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    )
    if intersection <= 0:
        return 0.0
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / float(max(1, union))


def _canonical_observation(
    observation: TextObservation,
    component_by_id: dict[str, SourceTextComponent],
) -> TextObservation:
    component_ids = sorted(
        {str(value) for value in observation.component_ids},
        key=lambda value: (
            _component_order(component_by_id[value])
            if value in component_by_id
            else (10**9, 10**9, 10**9, 10**9, value)
        ),
    )
    return replace(
        observation,
        component_ids=tuple(component_ids),
        tile_provenance=tuple(sorted({str(value) for value in observation.tile_provenance})),
    )


def _associated_component_ids(
    observation: TextObservation,
    components: Sequence[SourceTextComponent],
) -> frozenset[str]:
    known_ids = {component.component_id for component in components}
    explicit = frozenset(value for value in observation.component_ids if value in known_ids)
    return explicit


def _observation_support_rank(
    observation: TextObservation,
    covered_ids: frozenset[str],
    region_bbox: BBox,
) -> tuple[float, float, float, float, float]:
    explicit_coverage = float(observation.coverage_score or 0.0)
    geometry = _bbox_iou(observation.bbox_page, region_bbox)
    language = float(observation.language_score or 0.0)
    confidence = float(observation.confidence or 0.0)
    return (
        -float(len(covered_ids)),
        -geometry,
        -language,
        -confidence,
        -explicit_coverage,
    )


def _select_observations(
    region: _ResolvedRegion,
    observations: Sequence[TextObservation],
) -> tuple[list[TextObservation], list[TextObservation], dict[str, str]]:
    expected_ids = frozenset(component.component_id for component in region.components)
    region_bbox = _bbox_union(component.bbox_page for component in region.components)
    evidence: list[tuple[TextObservation, frozenset[str]]] = []
    selectable: list[tuple[TextObservation, frozenset[str]]] = []
    rejection_reasons: dict[str, str] = {}

    for observation in observations:
        associated = _associated_component_ids(observation, region.components)
        if not associated:
            continue
        all_explicit_ids = frozenset(observation.component_ids)
        if all_explicit_ids and not all_explicit_ids.issubset(expected_ids):
            continue
        evidence.append((observation, associated))
        if observation.rejection_reason is None and str(observation.text).strip():
            selectable.append((observation, associated))

    full_candidates = [item for item in selectable if item[1] == expected_ids]
    if full_candidates:
        ranked_full = sorted(
            full_candidates,
            key=lambda item: (
                _observation_support_rank(item[0], item[1], region_bbox),
                str(item[0].provider),
                str(item[0].observation_id),
            ),
        )
        best_support = _observation_support_rank(ranked_full[0][0], ranked_full[0][1], region_bbox)
        equally_supported = [
            item
            for item in ranked_full
            if _observation_support_rank(item[0], item[1], region_bbox) == best_support
        ]
        normalised_payloads = {
            " ".join(str(item[0].text).split()) for item in equally_supported
        }
        if len(normalised_payloads) > 1:
            for observation, _associated in equally_supported:
                rejection_reasons[observation.observation_id] = "ambiguous_reading"
            for observation, associated in selectable:
                if observation.observation_id not in rejection_reasons:
                    rejection_reasons[observation.observation_id] = (
                        "dominated_subcoverage"
                        if associated != expected_ids
                        else "dominated_candidate"
                    )
            return [], [item[0] for item in evidence], rejection_reasons

        selected_observation, _covered = equally_supported[0]
        for observation, associated in selectable:
            if observation.observation_id == selected_observation.observation_id:
                continue
            if associated != expected_ids:
                reason = "dominated_subcoverage"
            elif observation in [item[0] for item in equally_supported]:
                reason = "duplicate_equivalent"
            else:
                reason = "dominated_candidate"
            rejection_reasons[observation.observation_id] = reason
        return [selected_observation], [item[0] for item in evidence], rejection_reasons

    selected: list[TextObservation] = []
    covered: frozenset[str] = frozenset()
    remaining = list(selectable)
    while remaining and covered != expected_ids:
        ranked = sorted(
            remaining,
            key=lambda item: (
                -len(item[1] - covered),
                *_observation_support_rank(item[0], item[1], region_bbox)[1:],
                str(item[0].provider),
                str(item[0].observation_id),
            ),
        )
        observation, associated = ranked[0]
        if not (associated - covered):
            break
        if associated & covered:
            for candidate, _candidate_ids in selectable:
                rejection_reasons[candidate.observation_id] = "ambiguous_component_overlap"
            return [], [item[0] for item in evidence], rejection_reasons
        selected.append(observation)
        covered = frozenset(set(covered) | set(associated))
        remaining = [item for item in remaining if item[0].observation_id != observation.observation_id]

    if covered != expected_ids:
        for observation, _associated in selectable:
            rejection_reasons[observation.observation_id] = "incomplete_component_coverage"
        return [], [item[0] for item in evidence], rejection_reasons
    selected_ids = {item.observation_id for item in selected}
    for observation, _associated in selectable:
        if observation.observation_id not in selected_ids:
            rejection_reasons[observation.observation_id] = "dominated_subcoverage"
    return (
        sorted(selected, key=_observation_order),
        [item[0] for item in evidence],
        rejection_reasons,
    )


def _normalise_regions(
    components: Sequence[SourceTextComponent],
    semantic_regions: Sequence[SemanticRegion],
) -> tuple[list[_ResolvedRegion], list[OwnerViolation]]:
    component_by_id = {component.component_id: component for component in components}
    membership: dict[str, list[SemanticRegion]] = {}
    violations: list[OwnerViolation] = []

    for region in semantic_regions:
        for component_id in sorted(set(region.component_ids)):
            if component_id not in component_by_id:
                violations.append(
                    OwnerViolation(
                        code="semantic_region_component_unknown",
                        severity="critical",
                        message="Semantic region references an unknown source component.",
                        offenders=(str(region.region_id), str(component_id)),
                    )
                )
                continue
            membership.setdefault(component_id, []).append(region)

    normalised: list[_ResolvedRegion] = []
    for region in semantic_regions:
        unique_components = [
            component_by_id[component_id]
            for component_id in set(region.component_ids)
            if component_id in component_by_id and len(membership.get(component_id, ())) == 1
        ]
        if not unique_components:
            continue
        normalised.append(
            _ResolvedRegion(
                region_id=str(region.region_id),
                components=tuple(sorted(unique_components, key=_component_order)),
                semantic_role=str(region.semantic_role or "body"),
                disposition=str(region.disposition or "owned").strip().lower(),
                reason=region.reason,
            )
        )

    for component in sorted(components, key=_component_order):
        memberships = membership.get(component.component_id, [])
        if len(memberships) > 1:
            violations.append(
                OwnerViolation(
                    code="semantic_region_membership_conflict",
                    severity="critical",
                    message="Source component belongs to multiple semantic regions.",
                    offenders=(component.component_id,),
                )
            )
            normalised.append(
                _ResolvedRegion(
                    region_id=f"review_{component.component_id}",
                    components=(component,),
                    semantic_role="unresolved",
                    disposition="review",
                    reason="semantic_region_membership_conflict",
                )
            )
        elif not memberships:
            normalised.append(
                _ResolvedRegion(
                    region_id=f"review_{component.component_id}",
                    components=(component,),
                    semantic_role="unresolved",
                    disposition="review",
                    reason="semantic_region_missing",
                )
            )

    return sorted(
        normalised,
        key=lambda region: (_component_order(region.components[0]), region.semantic_role, region.region_id),
    ), violations


def _atomic_payload(selected: Sequence[TextObservation]) -> str:
    return " ".join(
        part
        for part in (" ".join(str(item.text).split()) for item in selected)
        if part
    )


def build_page_owner_graph(
    *,
    page_id: str,
    components: Sequence[SourceTextComponent],
    observations: Sequence[TextObservation],
    semantic_regions: Sequence[SemanticRegion],
) -> OwnerGraph:
    """Resolve immutable page evidence without allowing crops to define identity."""

    ordered_components = sorted(components, key=_component_order)
    component_by_id = {component.component_id: component for component in ordered_components}
    canonical_observations = sorted(
        (_canonical_observation(item, component_by_id) for item in observations),
        key=lambda item: item.observation_id,
    )
    regions, violations = _normalise_regions(ordered_components, semantic_regions)
    region_key_by_component = {
        component.component_id: region.region_id
        for region in regions
        for component in region.components
    }
    blocked_component_ids: set[str] = set()
    audit_reasons: dict[str, str] = {}

    for observation in canonical_observations:
        known_ids = set(observation.component_ids) & set(component_by_id)
        if not known_ids:
            audit_reasons[observation.observation_id] = "unassociated_observation"
            continue
        region_keys = {
            region_key_by_component[component_id]
            for component_id in known_ids
            if component_id in region_key_by_component
        }
        if len(region_keys) > 1:
            audit_reasons[observation.observation_id] = "cross_semantic_region"
            blocked_component_ids.update(known_ids)

    canonical_observations = [
        replace(
            observation,
            rejection_reason=(
                observation.rejection_reason
                or audit_reasons.get(observation.observation_id)
            ),
        )
        for observation in canonical_observations
    ]

    for component in ordered_components:
        if component.page_id != page_id:
            violations.append(
                OwnerViolation(
                    code="component_page_mismatch",
                    severity="critical",
                    message="Source component does not belong to the requested page.",
                    offenders=(component.component_id,),
                )
            )
    for observation in canonical_observations:
        if observation.page_id != page_id:
            violations.append(
                OwnerViolation(
                    code="observation_page_mismatch",
                    severity="critical",
                    message="OCR observation does not belong to the requested page.",
                    offenders=(observation.observation_id,),
                )
            )
        unknown = sorted(set(observation.component_ids) - set(component_by_id))
        if unknown:
            violations.append(
                OwnerViolation(
                    code="observation_component_unknown",
                    severity="critical",
                    message="OCR observation references unknown source components.",
                    offenders=(observation.observation_id, *unknown),
                )
            )

    owners: list[TextOwner] = []
    dispositions: list[ComponentDisposition] = []
    supported_dispositions = {"owned", "preserve", "suppress", "review"}

    for region in regions:
        component_ids = [component.component_id for component in region.components]
        requested_disposition = (
            region.disposition if region.disposition in supported_dispositions else "review"
        )
        if region.disposition not in supported_dispositions:
            violations.append(
                OwnerViolation(
                    code="semantic_region_disposition_invalid",
                    severity="critical",
                    message="Semantic region has an unsupported disposition.",
                    offenders=(region.region_id,),
                )
            )

        if requested_disposition in {"preserve", "suppress"}:
            dispositions.extend(
                ComponentDisposition(
                    component_id=component_id,
                    decision=requested_disposition,
                    owner_id=None,
                    reason=region.reason or f"semantic_region_{requested_disposition}",
                )
                for component_id in component_ids
            )
            continue

        selected, evidence, region_rejections = _select_observations(
            region,
            canonical_observations,
        )
        audit_reasons.update(region_rejections)
        if blocked_component_ids.intersection(component_ids):
            selected = []
            for observation in evidence:
                if (
                    observation.rejection_reason is None
                    and observation.observation_id not in audit_reasons
                ):
                    audit_reasons[
                        observation.observation_id
                    ] = "owner_blocked_by_cross_semantic_observation"
        payload = _atomic_payload(selected)
        final_disposition = requested_disposition
        if requested_disposition == "owned" and (not selected or not payload):
            final_disposition = "review"
        owner_id = _stable_owner_id(page_id, region.semantic_role, component_ids)
        owner = TextOwner(
            owner_id=owner_id,
            page_id=page_id,
            component_ids=component_ids,
            observation_ids=sorted({item.observation_id for item in evidence}),
            selected_observation_ids=[item.observation_id for item in selected],
            semantic_role=region.semantic_role,
            source_payload=payload,
            translated_payload=None,
            disposition=final_disposition,
            state="ocr_ready" if final_disposition == "owned" else "review_required",
            route_action=(
                "translate_inpaint_render" if final_disposition == "owned" else "review_required"
            ),
            execution_tile_id=None,
        )
        owners.append(owner)
        dispositions.extend(
            ComponentDisposition(
                component_id=component_id,
                decision=final_disposition,
                owner_id=owner_id,
                reason=(
                    region.reason
                    or (
                        "semantic_owner_resolved"
                        if final_disposition == "owned"
                        else "semantic_owner_unresolved"
                    )
                ),
            )
            for component_id in component_ids
        )

    canonical_observations = [
        replace(
            observation,
            rejection_reason=(
                observation.rejection_reason
                or audit_reasons.get(observation.observation_id)
            ),
        )
        for observation in canonical_observations
    ]

    return OwnerGraph(
        schema_version=1,
        page_id=page_id,
        components=ordered_components,
        observations=canonical_observations,
        owners=sorted(owners, key=lambda item: item.owner_id),
        projections=[],
        component_dispositions=sorted(dispositions, key=lambda item: item.component_id),
        violations=sorted(
            violations,
            key=lambda item: (item.code, item.offenders, item.message),
        ),
    )
