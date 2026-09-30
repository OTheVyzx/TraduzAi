"""Identity-validating owner evidence grouping boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
            elif previous_indices:
                union(previous_indices[0], index)
            previous_indices.append(index)
        for observation_id in entry.observation_ids:
            previous = by_observation.setdefault(observation_id, index)
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

    for entry in sorted(uncertain_entries, key=_entry_page_order):
        dispositions.append(
            ComponentDisposition(
                component_id=entry.component_id,
                decision="uncertain",
                reason="coverage_uncertain_requires_review",
                policy_id="coverage_ambiguous_candidate",
                policy_bbox_page=entry.bbox_page,
                policy_evidence_ids=entry.ocr_attempt_ids,
                policy_reason=(
                    "OCR-empty primary candidate lacks independent semantic "
                    "corroboration and requires human review"
                ),
            )
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
    graph.require_valid(mode="enforce")
    return graph
