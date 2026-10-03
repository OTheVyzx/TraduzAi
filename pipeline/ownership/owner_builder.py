"""Identity-validating owner evidence grouping boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import re

from .consensus_v2 import (
    independent_origins,
    select_consensus_observation,
    select_ordered_consensus_body_with_decisions,
)
from .coverage import CoverageEntry, CoverageInvariantError, PageCoverageResult
from .evidence import merge_observation_strict
from .hash_contract import canonical_json_sha256, sha256_text
from .model import (
    OWNER_GRAPH_SCHEMA_VERSION,
    ComponentDisposition,
    OwnerGraph,
    TextObservation,
    TextOwner,
)


class OwnerEvidenceIdentityError(ValueError):
    """Raised before consensus when evidence does not belong to the page request."""


_AUDITED_PRESERVE_POLICY_IDS = frozenset(
    {
        "explicit_visual_non_text",
        "explicit_sfx_outside_translatable_container",
        "explicit_credit_outside_translatable_container",
        "explicit_url_outside_translatable_container",
        "explicit_mark_outside_translatable_container",
        "ocr_empty_near_uniform_false_glyph",
        "ocr_empty_tiny_isolated_false_glyph",
        "ocr_empty_uncorroborated_glyph_scan",
        "ocr_empty_small_corroborated_sfx",
        "scanlation_credit_art",
        "scanlation_apparatus",
    }
)


@dataclass(frozen=True)
class OwnerPageEvidenceContext:
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str


@dataclass(frozen=True)
class OwnerObservationGroup:
    group_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    component_ids: tuple[str, ...]
    container_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    independent_invocation_ids: tuple[str, ...]
    source_payload: str
    source_payload_sha256: str
    consensus_sha256: str


def validate_and_group_owner_observations(
    page_context: OwnerPageEvidenceContext,
    observations: Sequence[TextObservation],
) -> tuple[OwnerObservationGroup, ...]:
    if not isinstance(page_context, OwnerPageEvidenceContext):
        raise TypeError("page_context must be an immutable OwnerPageEvidenceContext")
    required_context = (
        page_context.run_id,
        page_context.origin_execution_id,
        page_context.page_id,
        page_context.page_source_sha256,
    )
    if not all(str(value or "").strip() for value in required_context):
        raise OwnerEvidenceIdentityError("owner page evidence context is incomplete")

    by_id: dict[str, TextObservation] = {}
    for observation in observations:
        if not observation.identity_complete:
            raise OwnerEvidenceIdentityError(
                f"observation identity incomplete: {observation.observation_id}"
            )
        expected = (
            page_context.run_id,
            page_context.origin_execution_id,
            page_context.page_id,
            page_context.page_source_sha256,
        )
        actual = (
            observation.run_id,
            observation.origin_execution_id,
            observation.page_id,
            observation.page_source_sha256,
        )
        if actual != expected:
            raise OwnerEvidenceIdentityError(
                f"observation belongs to another request: {observation.observation_id}"
            )
        if observation.payload_sha256 != sha256_text(observation.text):
            raise OwnerEvidenceIdentityError(
                f"observation payload hash mismatch: {observation.observation_id}"
            )
        current = by_id.get(observation.observation_id)
        by_id[observation.observation_id] = (
            merge_observation_strict(current, observation)
            if current is not None
            else observation
        )

    grouped: dict[tuple[object, ...], list[TextObservation]] = {}
    for observation in by_id.values():
        support_key: tuple[object, ...] = (
            ("components", *tuple(sorted(observation.component_ids)))
            if observation.component_ids
            else ("bbox", *observation.bbox_page)
        )
        grouped.setdefault(support_key, []).append(observation)

    results: list[OwnerObservationGroup] = []
    for support_key, group_items in sorted(grouped.items(), key=lambda item: repr(item[0])):
        canonical_group = tuple(sorted(group_items, key=lambda item: item.observation_id))
        selected = select_consensus_observation(canonical_group)
        observation_ids = tuple(item.observation_id for item in canonical_group)
        invocation_ids = tuple(sorted(independent_origins(canonical_group)))
        component_ids = tuple(
            sorted({value for item in canonical_group for value in item.component_ids})
        )
        group_seed = {
            "run_id": page_context.run_id,
            "origin_execution_id": page_context.origin_execution_id,
            "page_id": page_context.page_id,
            "page_source_sha256": page_context.page_source_sha256,
            "support": list(support_key),
            "observation_ids": list(observation_ids),
        }
        consensus_seed = {
            "group": group_seed,
            "independent_invocation_ids": list(invocation_ids),
            "source_payload_sha256": sha256_text(selected.text),
        }
        results.append(
            OwnerObservationGroup(
                group_id=f"owner-observation-group:{canonical_json_sha256(group_seed)[:24]}",
                run_id=page_context.run_id,
                origin_execution_id=page_context.origin_execution_id,
                page_id=page_context.page_id,
                page_source_sha256=page_context.page_source_sha256,
                component_ids=component_ids,
                container_ids=(),
                observation_ids=observation_ids,
                independent_invocation_ids=invocation_ids,
                source_payload=selected.text,
                source_payload_sha256=sha256_text(selected.text),
                consensus_sha256=canonical_json_sha256(consensus_seed),
            )
        )
    return tuple(results)


def _entry_page_order(entry: CoverageEntry) -> tuple[int, int, int, int, str]:
    x1, y1, x2, y2 = entry.bbox_page
    return (y1, x1, y2, x2, entry.component_id)


def _weak_container_neighbours(left: CoverageEntry, right: CoverageEntry) -> bool:
    """Return whether two components are locally supported as one weak container."""
    ax1, ay1, ax2, ay2 = left.bbox_page
    bx1, by1, bx2, by2 = right.bbox_page
    left_width, left_height = ax2 - ax1, ay2 - ay1
    right_width, right_height = bx2 - bx1, by2 - by1
    x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    y_overlap = max(0, min(ay2, by2) - max(ay1, by1))
    x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
    y_gap = max(0, max(ay1, by1) - min(ay2, by2))
    same_line = (
        y_overlap >= 0.40 * min(left_height, right_height)
        and x_gap <= 2.0 * max(left_height, right_height)
    )
    stacked_lines = (
        x_overlap >= 0.15 * min(left_width, right_width)
        and y_gap <= max(8.0, 1.35 * min(left_height, right_height))
    )
    return bool(same_line or stacked_lines)


def _semantic_role(entries: Sequence[CoverageEntry]) -> str:
    roles = tuple(
        dict.fromkeys(
            str(entry.semantic_role or "").strip()
            for entry in entries
            if str(entry.semantic_role or "").strip()
        )
    )
    if any("dialogue" in role for role in roles):
        return "dialogue_body"
    return roles[0] if len(roles) == 1 else "dialogue_body"


def _is_scanlation_credit_page(observations: Sequence[TextObservation]) -> bool:
    """Recognize page-level scan credits without matching ordinary dialogue."""

    compact_texts = tuple(
        "".join(character for character in str(item.text).casefold() if character.isalnum())
        for item in observations
        if float(item.confidence) >= 0.60
    )
    strong_marker = any(
        any(marker in text for marker in ("wescanlate", "scanspresents", "nonstopscans"))
        for text in compact_texts
    )
    weak_families = {
        family
        for family, markers in (
            ("discord", ("joinourdiscord", "discordcominvite")),
            ("patreon", ("patreon",)),
            ("support", ("supportourwork", "supportusat", "donatetous")),
            ("contact", ("contactus",)),
            ("recruiting", ("scansisrecruiting", "translatorsopen")),
        )
        if any(marker in text for text in compact_texts for marker in markers)
    }
    return bool(
        strong_marker
        or len(weak_families) >= 3
    )


def _owner_id(
    coverage: PageCoverageResult,
    entries: Sequence[CoverageEntry],
    *,
    container_id: str | None = None,
) -> str:
    seed = {
        "run_id": coverage.run_id,
        "origin_execution_id": coverage.origin_execution_id,
        "page_id": coverage.page_id,
        "page_source_sha256": coverage.page_source_sha256,
        "container_id": container_id or entries[0].container_id,
        "component_ids": [entry.component_id for entry in entries],
    }
    return f"owner:{canonical_json_sha256(seed)[:24]}"


def _group_material_entries(
    entries: Sequence[CoverageEntry],
) -> tuple[tuple[str, tuple[CoverageEntry, ...]], ...]:
    """Join components proven equivalent by container or OCR observation."""

    ordered = tuple(sorted(entries, key=_entry_page_order))
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    by_container: dict[str, list[int]] = {}
    by_observation: dict[str, int] = {}
    for index, entry in enumerate(ordered):
        container_id = str(entry.container_id or "")
        if container_id:
            previous_indices = by_container.setdefault(container_id, [])
            if container_id.startswith("full_page_visual_container_group:"):
                roots: dict[int, list[int]] = {}
                for previous in previous_indices:
                    roots.setdefault(find(previous), []).append(previous)
                matching_roots = [
                    root
                    for root, members in roots.items()
                    if all(
                        _weak_container_neighbours(entry, ordered[member])
                        for member in members
                    )
                ]
                if len(matching_roots) == 1:
                    root_members = roots[matching_roots[0]]
                    union(root_members[0], index)
            elif container_id.startswith("full_page_visual_container:"):
                # A page envelope alone is weak. A containing OCR block and
                # its contained lines must still be reconciled together so
                # competing readings fail closed instead of becoming owners.
                for previous in previous_indices:
                    other = ordered[previous]
                    ax1, ay1, ax2, ay2 = entry.bbox_page
                    bx1, by1, bx2, by2 = other.bbox_page
                    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
                    smaller = min((ax2-ax1)*(ay2-ay1), (bx2-bx1)*(by2-by1))
                    if smaller > 0 and intersection >= .80 * smaller:
                        union(previous, index)
            elif previous_indices:
                union(previous_indices[0], index)
            previous_indices.append(index)
        for observation_id in entry.observation_ids:
            previous = by_observation.setdefault(observation_id, index)
            other = ordered[previous]
            page_level_pair = any(
                str(item.container_id or "").startswith("full_page_visual_container:")
                for item in (entry, other)
            )
            ax1, ay1, ax2, ay2 = entry.bbox_page
            bx1, by1, bx2, by2 = other.bbox_page
            component_boxes_overlap = (
                min(ax2, bx2) > max(ax1, bx1)
                and min(ay2, by2) > max(ay1, by1)
            )
            if not page_level_pair or component_boxes_overlap:
                union(previous, index)

    grouped: dict[int, list[CoverageEntry]] = {}
    for index, entry in enumerate(ordered):
        grouped.setdefault(find(index), []).append(entry)

    result: list[tuple[str, tuple[CoverageEntry, ...]]] = []
    for members in grouped.values():
        canonical_members = tuple(sorted(members, key=_entry_page_order))
        container_ids = tuple(
            sorted({str(entry.container_id) for entry in canonical_members})
        )
        if len(container_ids) == 1:
            group_container_id = container_ids[0]
        else:
            seed = {
                "container_ids": list(container_ids),
                "component_ids": [entry.component_id for entry in canonical_members],
            }
            group_container_id = (
                f"container-evidence-group:{canonical_json_sha256(seed)[:24]}"
            )
        result.append((group_container_id, canonical_members))
    return tuple(
        sorted(result, key=lambda item: _entry_page_order(item[1][0]))
    )




def _normalized_overlap_text(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def quarantine_verified_overlap_aliases(graph: OwnerGraph) -> OwnerGraph:
    """Keep duplicate nested detections in the graph while rendering only the
    smallest independently OCR-verified component; quarantine the broad scan
    for review. This intentionally handles only exact text aliases with strict
    containment and evidence-box checks.
    """
    from dataclasses import replace

    components = {item.component_id: item for item in graph.components}
    observations = {item.observation_id: item for item in graph.observations}
    owner_by_id = {item.owner_id: item for item in graph.owners}
    disposition_by_id = {item.component_id: item for item in graph.component_dispositions}
    replacements: dict[str, ComponentDisposition] = {}

    for owner in graph.owners:
        if owner.disposition != "owned" or len(owner.component_ids) < 2:
            continue
        expected = _normalized_overlap_text(owner.source_payload)
        if not expected:
            continue
        local = {}
        for component_id in owner.component_ids:
            evidence = tuple(
                observations[observation_id]
                for observation_id in owner.observation_ids
                if observation_id in observations
                and component_id in observations[observation_id].component_ids
                and observations[observation_id].rejection_reason is None
            )
            if not evidence:
                continue
            try:
                selected, payload, _ = select_ordered_consensus_body_with_decisions((evidence,))
            except Exception:
                continue
            if _normalized_overlap_text(payload) == expected:
                local[component_id] = (evidence, selected, payload)
        if len(local) < 2:
            continue
        canonical_id = min(
            local,
            key=lambda component_id: (
                (components[component_id].bbox_page[2] - components[component_id].bbox_page[0])
                * (components[component_id].bbox_page[3] - components[component_id].bbox_page[1]),
                component_id,
            ),
        )
        canonical = components[canonical_id]
        cx1, cy1, cx2, cy2 = canonical.bbox_page
        canonical_area = (cx2 - cx1) * (cy2 - cy1)
        if canonical_area <= 0:
            continue
        source_candidates = []
        for source_id, (source_evidence, _, source_payload) in local.items():
            if source_id == canonical_id:
                continue
            source = components[source_id]
            sx1, sy1, sx2, sy2 = source.bbox_page
            source_area = (sx2 - sx1) * (sy2 - sy1)
            if not (sx1 <= cx1 and sy1 <= cy1 and sx2 >= cx2 and sy2 >= cy2):
                continue
            area_ratio = canonical_area / source_area if source_area > 0 else 1.0
            if source_area <= canonical_area or area_ratio > 0.35:
                continue
            # Above the ordinary .25 ratio, permit aliasing only with tight
            # OCR geometry (one pixel). The broad component remains uncertain.
            bbox_allowance = 6 if area_ratio <= 0.25 else 1
            if any(
                observation.bbox_page[0] < cx1 - bbox_allowance
                or observation.bbox_page[1] < cy1 - bbox_allowance
                or observation.bbox_page[2] > cx2 + bbox_allowance
                or observation.bbox_page[3] > cy2 + bbox_allowance
                for observation in source_evidence
            ):
                continue
            if _normalized_overlap_text(source_payload) != expected:
                continue
            source_candidates.append(source_id)
        if len(source_candidates) != 1:
            continue
        canonical_evidence = local[canonical_id][0]
        candidate_source = components[source_candidates[0]]
        source_area = (candidate_source.bbox_page[2] - candidate_source.bbox_page[0]) * (candidate_source.bbox_page[3] - candidate_source.bbox_page[1])
        area_ratio = canonical_area / source_area
        bbox_allowance = 6 if area_ratio <= 0.25 else 1
        if any(
            observation.bbox_page[0] < cx1 - bbox_allowance
            or observation.bbox_page[1] < cy1 - bbox_allowance
            or observation.bbox_page[2] > cx2 + bbox_allowance
            or observation.bbox_page[3] > cy2 + bbox_allowance
            for observation in canonical_evidence
        ):
            continue
        source_id = source_candidates[0]
        source = components[source_id]
        source_evidence, _, source_payload = local[source_id]
        canonical_evidence, selected, canonical_payload = local[canonical_id]
        evidence_ids = sorted(
            {item.observation_id for item in (*source_evidence, *canonical_evidence)}
        )
        proof = {
            "version": 1,
            "page_id": graph.page_id,
            "page_source_sha256": graph.page_source_sha256,
            "owner_id": owner.owner_id,
            "source_component_id": source_id,
            "source_bbox_page": list(source.bbox_page),
            "source_observation_ids": sorted(item.observation_id for item in source_evidence),
            "canonical_component_id": canonical_id,
            "canonical_bbox_page": list(canonical.bbox_page),
            "canonical_observation_ids": sorted(item.observation_id for item in canonical_evidence),
            "observation_bbox_allowance_px": bbox_allowance,
            "area_ratio_limit": 0.25 if area_ratio <= 0.25 else 0.35,
            "normalized_payload_sha256": sha256_text(expected),
            "area_ratio": area_ratio,
            "source_text": source_payload,
            "canonical_text": canonical_payload,
        }
        proof["proof_sha256"] = canonical_json_sha256(proof)
        replacements[source_id] = ComponentDisposition(
            component_id=source_id,
            decision="uncertain",
            reason="coverage:duplicate_overlap_preserved_for_review",
            policy_id="coverage_ambiguous_candidate",
            policy_bbox_page=source.bbox_page,
            policy_evidence_ids=tuple(evidence_ids),
            policy_reason=json.dumps(proof, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            related_component_ids=(canonical_id,),
        )
        owner.component_ids = [canonical_id]
        owner.observation_ids = sorted(item.observation_id for item in canonical_evidence)
        owner.selected_observation_ids = [item.observation_id for item in selected]
        # Keep the stable owner identity; this routine runs before any execution.

    if replacements:
        graph.component_dispositions = [
            replacements.get(item.component_id, item)
            for item in graph.component_dispositions
        ]
    return graph


def build_owner_page_graph_from_coverage(
    coverage: PageCoverageResult,
) -> OwnerGraph:
    """Build the enforce graph from the complete, immutable page ledger only."""

    if not isinstance(coverage, PageCoverageResult):
        raise TypeError("coverage must be an immutable PageCoverageResult")
    coverage.require_ready_for_ownership()

    observations_by_id = {
        observation.observation_id: observation
        for observation in coverage.observations
    }
    scanlation_credit_page = _is_scanlation_credit_page(coverage.observations)
    material_entries: list[CoverageEntry] = []
    preserved_entries: list[CoverageEntry] = []
    uncertain_entries: list[CoverageEntry] = []
    for entry in coverage.entries:
        if entry.materiality == "uncertain":
            uncertain_entries.append(entry)
            continue
        if entry.state == "explicit_non_dialogue_preserve" or scanlation_credit_page:
            preserved_entries.append(entry)
            continue
        if entry.materiality == "uncertain" and entry.state == "review_required":
            uncertain_entries.append(entry)
            continue
        if entry.materiality != "material":
            raise CoverageInvariantError(
                f"non-text component lacks explicit preserve disposition: {entry.component_id}"
            )
        if not entry.container_id:
            raise CoverageInvariantError(
                f"translatable component lacks container: {entry.component_id}"
            )
        material_entries.append(entry)

    translatable_container_ids = frozenset(
        str(entry.container_id) for entry in material_entries
    )
    groups = _group_material_entries(material_entries)

    owners: list[TextOwner] = []
    dispositions: list[ComponentDisposition] = []
    for container_id, entries in groups:
        observation_groups: list[tuple[TextObservation, ...]] = []
        all_observation_ids: list[str] = []
        for entry in entries:
            candidates = tuple(
                observations_by_id[observation_id]
                for observation_id in entry.observation_ids
                if observation_id in observations_by_id
                and entry.component_id
                in observations_by_id[observation_id].component_ids
            )
            if not candidates:
                raise CoverageInvariantError(
                    f"translatable component lacks associated OCR evidence: {entry.component_id}"
                )
            observation_groups.append(candidates)
            all_observation_ids.extend(item.observation_id for item in candidates)

        selected, source_payload, selection_decisions = (
            select_ordered_consensus_body_with_decisions(observation_groups)
        )
        ambiguous_observations = sorted(
            observation_id
            for observation_id, decision in selection_decisions.items()
            if decision.get("decision") in {
                "ambiguous_block_line_overlap",
                "ambiguous_same_physical_line",
            }
        )
        if ambiguous_observations:
            raise CoverageInvariantError(
                "competing_ocr_readings_require_review: "
                + ", ".join(ambiguous_observations)
            )
        owner_id = _owner_id(coverage, entries, container_id=container_id)
        component_ids = [entry.component_id for entry in entries]
        owner = TextOwner(
            owner_id=owner_id,
            page_id=coverage.page_id,
            component_ids=component_ids,
            observation_ids=list(dict.fromkeys(all_observation_ids)),
            selected_observation_ids=[item.observation_id for item in selected],
            semantic_role=_semantic_role(entries),
            source_payload=source_payload,
            translated_payload=None,
            disposition="owned",
            state="observed",
            route_action="translate_inpaint_render",
            execution_tile_id=None,
        )
        owners.append(owner)
        dispositions.extend(
            ComponentDisposition(
                component_id=entry.component_id,
                decision="owned",
                owner_id=owner_id,
                reason=f"complete_page_coverage:{container_id}",
            )
            for entry in entries
        )

    for entry in sorted(preserved_entries, key=_entry_page_order):
        policy = str(
            entry.preserve_policy
            or ("policy:scanlation_apparatus" if scanlation_credit_page else "")
        )
        if not policy.startswith("policy:"):
            raise CoverageInvariantError(
                f"preserved component lacks explicit policy: {entry.component_id}"
            )
        policy_id = policy.removeprefix("policy:")
        if policy_id not in _AUDITED_PRESERVE_POLICY_IDS:
            raise CoverageInvariantError(
                f"unsupported preserve policy: {entry.component_id}/{policy_id}"
            )
        if entry.container_id and entry.container_id in translatable_container_ids:
            raise CoverageInvariantError(
                "preserved component shares a translatable container: "
                f"{entry.component_id}/{entry.container_id}"
            )
        if "sfx" in policy_id and "sfx" not in str(entry.semantic_role or "").casefold():
            raise CoverageInvariantError(
                f"SFX preserve policy lacks semantic evidence: {entry.component_id}"
            )
        evidence_ids = tuple(
            dict.fromkeys(
                (
                    *entry.protection_evidence_ids,
                    *entry.observation_ids,
                    *entry.ocr_attempt_ids,
                )
            )
        )
        if not evidence_ids:
            raise CoverageInvariantError(
                f"preserved component lacks audit evidence: {entry.component_id}"
            )
        dispositions.append(
            ComponentDisposition(
                component_id=entry.component_id,
                decision="preserve",
                reason=policy,
                policy_id=policy_id,
                policy_bbox_page=entry.bbox_page,
                policy_evidence_ids=evidence_ids,
                policy_reason=(
                    f"explicit non-dialogue preservation authorized by {policy}"
                ),
            )
        )

    for entry in sorted(uncertain_entries, key=_entry_page_order):
        component = next(
            item for item in coverage.components if item.component_id == entry.component_id
        )
        evidence_ids = tuple(
            dict.fromkeys(
                (*component.evidence_ids, *entry.ocr_attempt_ids,
                 *entry.protection_evidence_ids)
            )
        )
        if not evidence_ids or entry.state != "review_required":
            raise CoverageInvariantError(
                f"uncertain component lacks review evidence: {entry.component_id}"
            )
        dispositions.append(
            ComponentDisposition(
                component_id=entry.component_id,
                decision="uncertain",
                reason="coverage:empty_uncorroborated_primary_candidate",
                policy_id="coverage_ambiguous_candidate",
                policy_bbox_page=entry.bbox_page,
                policy_evidence_ids=evidence_ids,
                policy_reason=(
                    "Coverage OCR empty; primary visual hypothesis lacks independent "
                    "glyph or script corroboration; human review required"
                ),
            )
        )

    graph = OwnerGraph(
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id=coverage.page_id,
        components=list(coverage.components),
        observations=list(coverage.observations),
        owners=owners,
        projections=[],
        component_dispositions=dispositions,
        run_id=coverage.run_id,
        origin_execution_id=coverage.origin_execution_id,
        page_source_sha256=coverage.page_source_sha256,
        verification_status="verified",
    )
    graph = quarantine_verified_overlap_aliases(graph)
    graph.require_valid(mode="enforce")
    return graph
