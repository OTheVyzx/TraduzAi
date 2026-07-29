"""Fail-closed QA over fresh OCR observations of persisted final pixels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
import unicodedata

import numpy as np

from ownership.model import OwnerGraph, PageCompositionResult
from qa.final_pixel_observer import FinalPixelObservation, FinalPixelObserver


_CONTRACT_NAMES = (
    "source_coverage_contract",
    "owner_graph_contract",
    "route_state_contract",
    "pixel_ownership_contract",
    "final_language_contract",
    "qa_integrity_contract",
)
_RENDER_ROUTES = frozenset(
    {"translate_inpaint_render", "translate_sfx_inpaint_render", "translate_render_only"}
)


@dataclass(frozen=True)
class FinalPixelQaIssue:
    issue_id: str
    page_id: str
    owner_id: str | None
    component_ids: tuple[str, ...]
    severity: str
    reason: str
    offenders: tuple[str, ...]
    contract: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "page_id": self.page_id,
            "owner_id": self.owner_id,
            "component_ids": list(self.component_ids),
            "severity": self.severity,
            "reason": self.reason,
            "offenders": list(self.offenders),
            "contract": self.contract,
        }


@dataclass(frozen=True)
class FinalPixelQaReport:
    page_id: str
    persisted_sha256: str
    issues: tuple[FinalPixelQaIssue, ...]
    contracts: dict[str, str]
    passed: bool
    observed_text_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "contracts", MappingProxyType(dict(self.contracts)))


def _tokens(value: Any) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return tuple(
        token
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        if len(token) >= 2
    )


def source_payload_visible(source_payload: str, observed_text: str) -> bool:
    source = _tokens(source_payload)
    observed = _tokens(observed_text)
    if not source or not observed:
        return False
    observed_set = set(observed)
    if len(source) == 1:
        return len(source[0]) >= 4 and source[0] in observed_set
    source_bigrams = set(zip(source, source[1:]))
    observed_bigrams = set(zip(observed, observed[1:]))
    if source_bigrams & observed_bigrams:
        return True
    matched = sum(token in observed_set for token in source)
    return matched >= 2 and matched / len(source) >= 0.75


def _bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(item))) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _overlap_ratio(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    return overlap / float(left_area)


def _record_text(record: dict[str, Any]) -> str:
    for key in ("text", "raw_ocr", "original"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def _record_bbox(record: dict[str, Any]) -> tuple[int, int, int, int] | None:
    for key in ("bbox", "text_pixel_bbox", "source_bbox"):
        value = _bbox(record.get(key))
        if value is not None:
            return value
    return None


def _explicit_preserve_policy(reason: str | None) -> bool:
    normalized = str(reason or "").strip().casefold()
    return normalized.startswith("policy:") and len(normalized) > len("policy:")


def evaluate_final_pixel_observation(
    *,
    graph: OwnerGraph,
    composition: PageCompositionResult,
    observation: FinalPixelObservation,
) -> FinalPixelQaReport:
    if not isinstance(graph, OwnerGraph):
        raise TypeError("final pixel QA requires an OwnerGraph")
    if not isinstance(composition, PageCompositionResult):
        raise TypeError("final pixel QA requires a PageCompositionResult")
    if not isinstance(observation, FinalPixelObservation):
        raise TypeError("final pixel QA requires a FinalPixelObservation")

    issues: list[FinalPixelQaIssue] = []

    def add(
        reason: str,
        contract: str,
        *,
        owner_id: str | None = None,
        component_ids: tuple[str, ...] = (),
        offenders: tuple[str, ...] = (),
        severity: str = "critical",
    ) -> None:
        identity = f"{graph.page_id}:{contract}:{reason}:{owner_id or '-'}:{len(issues) + 1}"
        issues.append(
            FinalPixelQaIssue(
                issue_id=identity,
                page_id=graph.page_id,
                owner_id=owner_id,
                component_ids=tuple(component_ids),
                severity=severity,
                reason=reason,
                offenders=tuple(str(value) for value in offenders),
                contract=contract,
            )
        )

    if composition.page_id != graph.page_id:
        add(
            "composition_page_identity_mismatch",
            "qa_integrity_contract",
            offenders=(str(composition.page_id), graph.page_id),
        )
    for violation in graph.validate():
        add(
            violation.code,
            "owner_graph_contract",
            offenders=tuple(violation.offenders),
        )

    dispositions = {item.component_id: item for item in graph.component_dispositions}
    for component in graph.components:
        disposition = dispositions.get(component.component_id)
        if disposition is None:
            add(
                "source_component_without_final_disposition",
                "source_coverage_contract",
                component_ids=(component.component_id,),
                offenders=(component.component_id,),
            )
        elif disposition.decision == "preserve" and not _explicit_preserve_policy(
            disposition.reason
        ):
            add(
                "preserved_source_without_explicit_policy",
                "source_coverage_contract",
                owner_id=disposition.owner_id,
                component_ids=(component.component_id,),
                offenders=(component.component_id,),
            )

    glyph_map = np.asarray(composition.glyph_owner_map)
    cleanup_map = np.asarray(composition.cleanup_owner_map)
    final_rgb = np.asarray(composition.final_rgb)
    expected_shape = final_rgb.shape[:2]
    if (
        final_rgb.ndim != 3
        or final_rgb.shape[2] != 3
        or glyph_map.shape != expected_shape
        or cleanup_map.shape != expected_shape
        or observation.image_rgb.shape[:2] != expected_shape
    ):
        add("pixel_evidence_shape_mismatch", "qa_integrity_contract")
    else:
        owned_pixels = int(np.count_nonzero((glyph_map != "") | (cleanup_map != "")))
        owned_mask = (glyph_map != "") | (cleanup_map != "")
        final_changed = composition.write_counts.get("final_changed_pixels")
        if not isinstance(final_changed, int) or final_changed != owned_pixels:
            add(
                "pixel_change_outside_owned_masks",
                "pixel_ownership_contract",
                offenders=(str(final_changed), str(owned_pixels)),
            )
        persisted_delta = np.max(
            np.abs(
                observation.image_rgb.astype(np.int16)
                - final_rgb.astype(np.int16)
            ),
            axis=2,
        )
        material_outside = (persisted_delta > 24) & ~owned_mask
        if np.any(material_outside):
            add(
                "pixel_change_outside_owned_masks",
                "pixel_ownership_contract",
                offenders=(
                    f"persisted_material_pixels:{int(np.count_nonzero(material_outside))}",
                ),
            )
        known_owner_ids = {owner.owner_id for owner in graph.owners}
        mapped_owner_ids = {
            str(value)
            for owner_map in (glyph_map, cleanup_map)
            for value in np.unique(owner_map)
            if str(value)
        }
        unknown = tuple(sorted(mapped_owner_ids - known_owner_ids))
        if unknown:
            add(
                "pixel_map_references_unknown_owner",
                "pixel_ownership_contract",
                offenders=unknown,
            )

    for conflict in composition.conflicts:
        reason = (
            "protected_art_damage"
            if "protected" in str(conflict.code).casefold()
            else "owner_composition_conflict"
        )
        add(
            reason,
            "pixel_ownership_contract",
            owner_id=conflict.owner_ids[0] if len(conflict.owner_ids) == 1 else None,
            offenders=tuple(conflict.owner_ids) or (conflict.code,),
        )
    if composition.committed is not True and not composition.conflicts:
        add("owner_composition_not_committed", "pixel_ownership_contract")

    owner_by_component = {
        component_id: owner
        for owner in graph.owners
        for component_id in owner.component_ids
    }
    for owner in graph.owners:
        if owner.state == "review_required" or owner.route_action == "review_required":
            add(
                "owner_route_not_final",
                "route_state_contract",
                owner_id=owner.owner_id,
                component_ids=tuple(owner.component_ids),
                offenders=(owner.state, owner.route_action),
            )
        elif owner.route_action in _RENDER_ROUTES and owner.state not in {
            "rendered",
            "verified",
        }:
            add(
                "owner_route_not_final",
                "route_state_contract",
                owner_id=owner.owner_id,
                component_ids=tuple(owner.component_ids),
                offenders=(owner.state,),
            )
        if owner.route_action in _RENDER_ROUTES and not np.any(glyph_map == owner.owner_id):
            add(
                "missing_owner_glyphs",
                "pixel_ownership_contract",
                owner_id=owner.owner_id,
                component_ids=tuple(owner.component_ids),
                offenders=(owner.owner_id,),
            )

    for index, record in enumerate(observation.ocr_records):
        text = _record_text(record)
        if not text:
            continue
        bbox = _record_bbox(record)
        matching_components = [
            component
            for component in graph.components
            if bbox is not None and _overlap_ratio(bbox, component.bbox_page) >= 0.2
        ]
        if not matching_components:
            add(
                "independently_detected_text_without_owner",
                "source_coverage_contract",
                offenders=(f"ocr_record_{index}", text),
            )
            continue
        candidate_owners = {
            owner_by_component[component.component_id].owner_id
            for component in matching_components
            if component.component_id in owner_by_component
        }
        for owner_id in sorted(candidate_owners):
            owner = next(item for item in graph.owners if item.owner_id == owner_id)
            if source_payload_visible(owner.source_payload, text):
                add(
                    "source_payload_visible",
                    "final_language_contract",
                    owner_id=owner.owner_id,
                    component_ids=tuple(owner.component_ids),
                    offenders=(text, owner.source_payload),
                )
        if not candidate_owners:
            for component in matching_components:
                disposition = dispositions.get(component.component_id)
                source_observations = [
                    item.text
                    for item in graph.observations
                    if component.component_id in item.component_ids
                ]
                if disposition is None or disposition.decision != "preserve":
                    add(
                        "independently_detected_text_without_owner",
                        "source_coverage_contract",
                        component_ids=(component.component_id,),
                        offenders=(text,),
                    )
                elif not _explicit_preserve_policy(disposition.reason) and any(
                    source_payload_visible(source, text) for source in source_observations
                ):
                    # The policy issue is already emitted above; do not hide the
                    # independently observed source evidence behind metadata.
                    pass

    contracts = {
        name: (
            "BLOCK"
            if any(issue.contract == name and issue.severity == "critical" for issue in issues)
            else "PASS"
        )
        for name in _CONTRACT_NAMES
    }
    return FinalPixelQaReport(
        page_id=graph.page_id,
        persisted_sha256=observation.persisted_sha256,
        issues=tuple(issues),
        contracts=contracts,
        passed=not any(issue.severity == "critical" for issue in issues),
        observed_text_count=len(observation.ocr_records),
    )


def evaluate_final_pixels(
    *,
    image_path: Path,
    graph: OwnerGraph,
    composition: PageCompositionResult,
    observer: FinalPixelObserver,
    source_language: str = "en",
) -> FinalPixelQaReport:
    observation = observer.observe(Path(image_path), source_language=source_language)
    if Path(observation.image_path) != Path(image_path):
        raise ValueError("final pixel observer returned evidence for another file")
    return evaluate_final_pixel_observation(
        graph=graph,
        composition=composition,
        observation=observation,
    )
