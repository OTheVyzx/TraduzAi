"""Identity-validating owner evidence grouping boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from .consensus_v2 import independent_origins, select_consensus_observation
from .evidence import merge_observation_strict
from .hash_contract import canonical_json_sha256, sha256_text
from .model import TextObservation


class OwnerEvidenceIdentityError(ValueError):
    """Raised before consensus when evidence does not belong to the page request."""


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
