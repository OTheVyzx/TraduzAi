"""Deterministic reconciliation of page-global text evidence into owners."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Iterable, Sequence

from .evidence import normalize_evidence_tokens, safely_dominates
from .model import (
    BBox,
    ComponentDisposition,
    OwnerGraph,
    OwnerViolation,
    SourceTextComponent,
    TextObservation,
    TextOwner,
)


_SELECTED_INK_MIN_SAFETY_MARGIN_PX = 3
_SELECTED_INK_MAX_SAFETY_MARGIN_PX = 18


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


def _observation_polygon_bbox(polygon: Sequence[Sequence[int]]) -> BBox | None:
    if len(polygon) < 3:
        return None
    try:
        xs = [int(point[0]) for point in polygon]
        ys = [int(point[1]) for point in polygon]
    except (IndexError, TypeError, ValueError):
        return None
    if min(xs) < 0 or min(ys) < 0:
        return None
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _selected_ink_safety_margin(bbox: BBox) -> int:
    """Reserve the same proportional halo that owner-mask dilation may consume."""

    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    proportional = int(math.ceil(min(width, height) * 0.30))
    return max(
        _SELECTED_INK_MIN_SAFETY_MARGIN_PX,
        min(_SELECTED_INK_MAX_SAFETY_MARGIN_PX, proportional),
    )


def _expand_owned_components_to_selected_ink(
    components: Sequence[SourceTextComponent],
    observations: Sequence[TextObservation],
    owners: Sequence[TextOwner],
) -> list[SourceTextComponent]:
    """Include every selected OCR line in one authoritative owner component."""

    component_by_id = {component.component_id: component for component in components}
    observation_by_id = {
        observation.observation_id: observation for observation in observations
    }
    assigned_boxes: dict[str, list[BBox]] = {
        component.component_id: [component.bbox_page] for component in components
    }
    for owner in owners:
        if owner.disposition != "owned":
            continue
        owner_component_ids = set(owner.component_ids)
        for observation_id in owner.selected_observation_ids:
            observation = observation_by_id.get(observation_id)
            if observation is None:
                continue
            candidate_ids = sorted(
                owner_component_ids.intersection(observation.component_ids)
            )
            if not candidate_ids:
                continue
            polygon_boxes = [
                box
                for polygon in observation.polygons_page
                if (box := _observation_polygon_bbox(polygon)) is not None
            ]
            if not polygon_boxes:
                polygon_boxes = [observation.bbox_page]
            safety_margin = _selected_ink_safety_margin(_bbox_union(polygon_boxes))
            for polygon_bbox in polygon_boxes:
                polygon_bbox = (
                    max(0, polygon_bbox[0] - safety_margin),
                    max(0, polygon_bbox[1] - safety_margin),
                    polygon_bbox[2] + safety_margin,
                    polygon_bbox[3] + safety_margin,
                )
                polygon_center = (
                    (polygon_bbox[0] + polygon_bbox[2]) / 2.0,
                    (polygon_bbox[1] + polygon_bbox[3]) / 2.0,
                )

                def assignment_rank(component_id: str) -> tuple[int, float, str]:
                    component_bbox = component_by_id[component_id].bbox_page
                    intersection = (
                        max(
                            0,
                            min(polygon_bbox[2], component_bbox[2])
                            - max(polygon_bbox[0], component_bbox[0]),
                        )
                        * max(
                            0,
                            min(polygon_bbox[3], component_bbox[3])
                            - max(polygon_bbox[1], component_bbox[1]),
                        )
                    )
                    component_center = (
                        (component_bbox[0] + component_bbox[2]) / 2.0,
                        (component_bbox[1] + component_bbox[3]) / 2.0,
                    )
                    distance = (
                        (polygon_center[0] - component_center[0]) ** 2
                        + (polygon_center[1] - component_center[1]) ** 2
                    )
                    return -intersection, distance, component_id

                assigned_boxes[min(candidate_ids, key=assignment_rank)].append(
                    polygon_bbox
                )

    expanded: list[SourceTextComponent] = []
    for component in components:
        boxes = assigned_boxes[component.component_id]
        bbox = _bbox_union(boxes)
        if bbox == component.bbox_page:
            expanded.append(component)
            continue
        x1, y1, x2, y2 = bbox
        expanded.append(
            replace(
                component,
                bbox_page=bbox,
                polygon_page=(
                    (x1, y1),
                    (x2 - 1, y1),
                    (x2 - 1, y2 - 1),
                    (x1, y2 - 1),
                ),
            )
        )
    return expanded


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

    normalized_tokens = {
        observation.observation_id: normalize_evidence_tokens(observation.text)
        for observation, _associated in selectable
    }
    corroboration_counts = {
        observation.observation_id: sum(
            normalized_tokens[candidate.observation_id]
            == normalized_tokens[observation.observation_id]
            for candidate, _candidate_ids in selectable
        )
        for observation, _associated in selectable
    }
    dominant_by_id: dict[str, TextObservation] = {}
    for truncated, truncated_ids in selectable:
        dominators: list[tuple[TextObservation, frozenset[str]]] = []
        for complete, complete_ids in selectable:
            if complete.observation_id == truncated.observation_id:
                continue
            same_region = bool(
                truncated_ids
                and complete_ids
                and truncated_ids.issubset(complete_ids)
                and _bbox_iou(complete.bbox_page, truncated.bbox_page) > 0.0
            )
            if safely_dominates(
                complete=complete,
                truncated=truncated,
                same_region=same_region,
                corroboration_count=corroboration_counts[complete.observation_id],
            ):
                dominators.append((complete, complete_ids))
        if dominators:
            dominant_by_id[truncated.observation_id] = sorted(
                dominators,
                key=lambda item: (
                    _observation_support_rank(item[0], item[1], region_bbox),
                    str(item[0].provider),
                    str(item[0].observation_id),
                ),
            )[0][0]

    maximal = [
        item
        for item in selectable
        if item[0].observation_id not in dominant_by_id
    ]
    full_candidates = [item for item in maximal if item[1] == expected_ids]
    if full_candidates:
        maximal_payloads = {
            normalized_tokens[observation.observation_id]
            for observation, _associated in full_candidates
        }
        if len(maximal_payloads) > 1:
            for observation, _associated in full_candidates:
                rejection_reasons[observation.observation_id] = "ambiguous_reading"
            for observation, associated in selectable:
                if observation.observation_id in rejection_reasons:
                    continue
                rejection_reasons[observation.observation_id] = (
                    "dominated_subcoverage"
                    if associated != expected_ids
                    else "dominated_truncation"
                )
            return [], [item[0] for item in evidence], rejection_reasons

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
        selected_observation, _covered = equally_supported[0]
        selected_group = [
            selected_observation,
            *sorted(
                (
                    observation
                    for observation, _associated in full_candidates
                    if observation.observation_id != selected_observation.observation_id
                    and normalized_tokens[observation.observation_id]
                    == normalized_tokens[selected_observation.observation_id]
                ),
                key=_observation_order,
            ),
        ]
        selected_ids = {item.observation_id for item in selected_group}
        for observation, associated in selectable:
            if observation.observation_id in selected_ids:
                continue
            if associated != expected_ids:
                reason = "dominated_subcoverage"
            elif observation.observation_id in dominant_by_id:
                reason = "dominated_truncation"
            elif (
                normalized_tokens[observation.observation_id]
                == normalized_tokens[selected_observation.observation_id]
            ):
                reason = "duplicate_equivalent"
            else:
                reason = "dominated_candidate"
            rejection_reasons[observation.observation_id] = reason
        return selected_group, [item[0] for item in evidence], rejection_reasons

    selected: list[TextObservation] = []
    covered: frozenset[str] = frozenset()
    remaining = list(maximal)
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
    parts: list[str] = []
    seen_evidence: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for item in selected:
        part = " ".join(str(item.text).split())
        if not part:
            continue
        evidence_key = (
            normalize_evidence_tokens(part),
            tuple(sorted(str(value) for value in item.component_ids)),
        )
        if evidence_key in seen_evidence:
            continue
        seen_evidence.add(evidence_key)
        parts.append(part)
    return " ".join(parts)


def _multi_ocr_confirms_non_text(
    region: _ResolvedRegion,
    evidence: Sequence[TextObservation],
) -> bool:
    """Return true only for weak geometry repeatedly confirmed to contain no text."""

    if len(evidence) < 3 or any(str(item.text).strip() for item in evidence):
        return False
    if any(component.script_evidence for component in region.components):
        return False
    return True


_EXTERNAL_IDENTIFIER_RE = re.compile(
    r"(?ix)(?:"
    r"https?://|www\.|discord(?:\.gg|app\.com/invite)/|"
    r"\b[a-z0-9][a-z0-9.-]+\.(?:com|net|org|io|gg|co|me|tv)\b"
    r")"
)


def _is_nontranslatable_external_identifier(payload: str) -> bool:
    """Recognize URLs/invite handles whose source pixels are intentional."""

    normalized = " ".join(str(payload or "").split())
    credit_roles = re.findall(r"(?i)(?:^|\s)(?:TL|PR|RD|TS|CL)\s*:", normalized)
    return bool(
        normalized
        and (
            _EXTERNAL_IDENTIFIER_RE.search(normalized)
            or len(credit_roles) >= 3
        )
    )


def _scanlation_apparatus_cutoff(
    observations: Sequence[TextObservation],
) -> int | None:
    """Find a page-space boundary for explicit scanlation/promotional matter."""

    strong_y: list[int] = []
    weak_by_family: dict[str, list[int]] = {
        "contact": [],
        "commission": [],
        "discord": [],
        "patreon": [],
        "recruiting": [],
        "support": [],
    }
    for observation in observations:
        text = "".join(char for char in str(observation.text).casefold() if char.isalnum())
        if not text or len(text) > 180 or float(observation.confidence) < 0.60:
            continue
        y1 = int(observation.bbox_page[1])
        if any(marker in text for marker in ("wescanlate", "nonstopscans", "scanspresents")):
            strong_y.append(y1)
        if "joinourdiscord" in text:
            weak_by_family["discord"].append(y1)
        if "discordcominvite" in text:
            weak_by_family["discord"].append(y1)
        if "youcancontactus" in text or "contactusatoursite" in text:
            weak_by_family["contact"].append(y1)
        if "commissionyourfavorite" in text or "commissionyourseries" in text:
            weak_by_family["commission"].append(y1)
        if (
            "haveapatreon" in text
            or "nowhaveapatreon" in text
            or "specialthankstoourpatreon" in text
            or "patreoncom" in text
        ):
            weak_by_family["patreon"].append(y1)
        if "scansisrecruiting" in text or "translatorsopen" in text:
            weak_by_family["recruiting"].append(y1)
        if "supportusat" in text or "paypalcom" in text or "koficom" in text:
            weak_by_family["support"].append(y1)
    candidates = list(strong_y)
    present_weak_families = [values for values in weak_by_family.values() if values]
    if len(present_weak_families) >= 2:
        candidates.extend(min(values) for values in present_weak_families)
    return max(0, min(candidates) - 100) if candidates else None


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
    region_role_by_component = {
        component.component_id: region.semantic_role
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
            semantic_roles = {
                region_role_by_component[component_id]
                for component_id in known_ids
                if component_id in region_role_by_component
            }
            if len(semantic_roles) > 1:
                audit_reasons[observation.observation_id] = "cross_semantic_region"
                blocked_component_ids.update(known_ids)
            else:
                audit_reasons[observation.observation_id] = (
                    "cross_region_same_role_observation"
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
    scanlation_cutoff = _scanlation_apparatus_cutoff(canonical_observations)

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

        if (
            requested_disposition == "owned"
            and scanlation_cutoff is not None
            and min(component.bbox_page[1] for component in region.components)
            >= scanlation_cutoff
        ):
            dispositions.extend(
                ComponentDisposition(
                    component_id=component_id,
                    decision="preserve",
                    owner_id=None,
                    reason="policy:scanlation_apparatus",
                )
                for component_id in component_ids
            )
            continue

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
        if (
            requested_disposition == "owned"
            and not evidence
            and not any(component.script_evidence for component in region.components)
            and not any(
                _bbox_iou(
                    observation.bbox_page,
                    _bbox_union(component.bbox_page for component in region.components),
                )
                > 0.0
                for observation in canonical_observations
            )
        ):
            dispositions.extend(
                ComponentDisposition(
                    component_id=component_id,
                    decision="suppress",
                    owner_id=None,
                    reason="no_ocr_evidence_non_text",
                )
                for component_id in component_ids
            )
            continue
        if requested_disposition == "owned" and _multi_ocr_confirms_non_text(
            region,
            evidence,
        ):
            for observation in evidence:
                audit_reasons.setdefault(
                    observation.observation_id,
                    "multi_ocr_confirmed_non_text",
                )
            dispositions.extend(
                ComponentDisposition(
                    component_id=component_id,
                    decision="suppress",
                    owner_id=None,
                    reason="multi_ocr_confirmed_non_text",
                )
                for component_id in component_ids
            )
            continue
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
        if (
            requested_disposition == "owned"
            and payload
            and _is_nontranslatable_external_identifier(payload)
        ):
            dispositions.extend(
                ComponentDisposition(
                    component_id=component_id,
                    decision="preserve",
                    owner_id=None,
                    reason="policy:nontranslatable_external_identifier",
                )
                for component_id in component_ids
            )
            continue
        final_disposition = requested_disposition
        if requested_disposition == "owned" and (
            not selected
            or not payload
            or region.reason == "semantic_container_missing"
        ):
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

    component_disposition_by_id = {
        disposition.component_id: disposition for disposition in dispositions
    }
    owned_component_ids = {
        component_id
        for owner in owners
        if owner.disposition == "owned"
        for component_id in owner.component_ids
    }
    redundant_review_owner_ids: set[str] = set()
    for owner in owners:
        if (
            owner.disposition != "review"
            or str(owner.source_payload).strip()
            or owner.observation_ids
            or not owner.component_ids
            or any(
                component_disposition_by_id.get(component_id) is None
                or component_disposition_by_id[component_id].reason
                != "semantic_container_missing"
                for component_id in owner.component_ids
            )
        ):
            continue
        outer_bbox = _bbox_union(
            component_by_id[component_id].bbox_page
            for component_id in owner.component_ids
        )
        if any(
            outer_bbox[0] <= component_by_id[component_id].bbox_page[0]
            and outer_bbox[1] <= component_by_id[component_id].bbox_page[1]
            and outer_bbox[2] >= component_by_id[component_id].bbox_page[2]
            and outer_bbox[3] >= component_by_id[component_id].bbox_page[3]
            for component_id in owned_component_ids
        ):
            redundant_review_owner_ids.add(owner.owner_id)

    if redundant_review_owner_ids:
        redundant_component_ids = {
            component_id
            for owner in owners
            if owner.owner_id in redundant_review_owner_ids
            for component_id in owner.component_ids
        }
        owners = [
            owner
            for owner in owners
            if owner.owner_id not in redundant_review_owner_ids
        ]
        dispositions = [
            (
                replace(
                    disposition,
                    decision="suppress",
                    owner_id=None,
                    reason="redundant_container_without_ocr_evidence",
                )
                if disposition.component_id in redundant_component_ids
                else disposition
            )
            for disposition in dispositions
        ]

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
    ordered_components = _expand_owned_components_to_selected_ink(
        ordered_components,
        canonical_observations,
        owners,
    )

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
