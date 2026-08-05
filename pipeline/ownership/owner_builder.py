"""Identity-validating owner evidence grouping boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .consensus_v2 import (
    independent_origins,
    select_consensus_observation,
    select_ordered_consensus_body,
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


def _owner_id(coverage: PageCoverageResult, entries: Sequence[CoverageEntry]) -> str:
    seed = {
        "run_id": coverage.run_id,
        "origin_execution_id": coverage.origin_execution_id,
        "page_id": coverage.page_id,
        "page_source_sha256": coverage.page_source_sha256,
        "container_id": entries[0].container_id,
        "component_ids": [entry.component_id for entry in entries],
    }
    return f"owner:{canonical_json_sha256(seed)[:24]}"


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
    groups: dict[str, list[CoverageEntry]] = {}
    preserved_entries: list[CoverageEntry] = []
    for entry in coverage.entries:
        if entry.state == "explicit_non_dialogue_preserve":
            preserved_entries.append(entry)
            continue
        if entry.materiality != "material":
            raise CoverageInvariantError(
                f"non-text component lacks explicit preserve disposition: {entry.component_id}"
            )
        if not entry.container_id:
            raise CoverageInvariantError(
                f"translatable component lacks container: {entry.component_id}"
            )
        groups.setdefault(entry.container_id, []).append(entry)

    translatable_container_ids = frozenset(groups)

    owners: list[TextOwner] = []
    dispositions: list[ComponentDisposition] = []
    for container_id, group_entries in sorted(groups.items()):
        entries = tuple(sorted(group_entries, key=_entry_page_order))
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

        selected, source_payload = select_ordered_consensus_body(observation_groups)
        owner_id = _owner_id(coverage, entries)
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
        policy = str(entry.preserve_policy or "")
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
