"""Project persistence boundary for verified page-global owner graphs.

The editor project and schema-v12 project use different page shapes.  This
module owns the additive ownership envelope shared by both formats and keeps
legacy projects explicitly outside the verified execution path.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Iterable, Sequence

from .model import (
    OWNER_GRAPH_LEGACY_SCHEMA_VERSION,
    OWNER_GRAPH_SCHEMA_VERSION,
    OwnerGraph,
    OwnerGraphValidationError,
)


OWNER_GRAPH_STATUS_VERIFIED = "verified"
OWNER_GRAPH_STATUS_LEGACY_UNVERIFIED = "legacy_unverified"
OWNER_GRAPH_STATUSES = frozenset(
    {OWNER_GRAPH_STATUS_VERIFIED, OWNER_GRAPH_STATUS_LEGACY_UNVERIFIED}
)
OWNER_PROJECT_FIELDS = (
    "owner_graph_schema_version",
    "owner_graph_status",
    "page_owner_graphs",
    "owner_invariant_summary",
)
OWNER_SUMMARY_FIELDS = (
    "page_count",
    "component_count",
    "observation_count",
    "owner_count",
    "projection_count",
    "violation_count",
    "critical_violation_count",
)
_GRAPH_LIST_FIELDS = (
    "components",
    "observations",
    "owners",
    "projections",
    "component_dispositions",
    "violations",
)


class OwnerProjectValidationError(ValueError):
    """Raised when persisted ownership metadata cannot be trusted."""


def _nonempty_identity(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OwnerProjectValidationError(
            f"owner graph {label} must be a canonical non-empty string"
        )
    return value


def _identity_set(
    records: Sequence[dict[str, Any]], key: str, *, label: str
) -> set[str]:
    identities = [
        _nonempty_identity(record.get(key), label=label) for record in records
    ]
    duplicates = sorted({value for value in identities if identities.count(value) > 1})
    if duplicates:
        raise OwnerProjectValidationError(
            f"owner graph has duplicate {label}: {', '.join(duplicates)}"
        )
    return set(identities)


def _record_list(graph: dict[str, Any], field: str) -> list[dict[str, Any]]:
    value = graph.get(field)
    if not isinstance(value, list):
        raise OwnerProjectValidationError(f"owner graph {field} must be a list")
    if any(not isinstance(item, dict) for item in value):
        raise OwnerProjectValidationError(
            f"owner graph {field} entries must be objects"
        )
    return value


def _require_page_identity(
    records: Sequence[dict[str, Any]],
    *,
    page_id: str,
    label: str,
) -> None:
    identities = [
        _nonempty_identity(record.get("page_id"), label=f"{label}.page_id")
        for record in records
    ]
    mismatched = sorted(identity for identity in identities if identity != page_id)
    if mismatched:
        raise OwnerProjectValidationError(
            f"owner graph {label} page_id mismatch for {page_id}: {', '.join(mismatched)}"
        )


def _require_known_references(
    records: Sequence[dict[str, Any]],
    *,
    field: str,
    known: set[str],
    label: str,
) -> None:
    for record in records:
        references = record.get(field)
        if not isinstance(references, list):
            raise OwnerProjectValidationError(
                f"owner graph {label}.{field} must be a list"
            )
        unknown = sorted(
            {
                _nonempty_identity(value, label=f"{label}.{field}")
                for value in references
            }
            - known
        )
        if unknown:
            raise OwnerProjectValidationError(
                f"owner graph {label} references unknown {field}: {', '.join(unknown)}"
            )


def validate_serialized_owner_graph(
    graph_payload: Any,
    *,
    enforce: bool = False,
) -> OwnerGraph:
    """Validate a serialized graph more strictly than the tolerant reader."""

    if not isinstance(graph_payload, dict):
        raise OwnerProjectValidationError("owner graph must be an object")
    schema_version = graph_payload.get("schema_version")
    if schema_version == OWNER_GRAPH_LEGACY_SCHEMA_VERSION:
        try:
            return OwnerGraph.from_dict(deepcopy(graph_payload), enforce=enforce)
        except OwnerGraphValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise OwnerProjectValidationError(f"owner graph is invalid: {exc}") from exc
    if schema_version != OWNER_GRAPH_SCHEMA_VERSION:
        raise OwnerProjectValidationError(
            f"owner graph schema_version must be {OWNER_GRAPH_SCHEMA_VERSION}"
        )

    page_id = _nonempty_identity(graph_payload.get("page_id"), label="page_id")
    records = {
        field: _record_list(graph_payload, field) for field in _GRAPH_LIST_FIELDS
    }
    components = records["components"]
    observations = records["observations"]
    owners = records["owners"]
    projections = records["projections"]
    dispositions = records["component_dispositions"]

    component_ids = _identity_set(components, "component_id", label="component_id")
    observation_ids = _identity_set(
        observations, "observation_id", label="observation_id"
    )
    owner_ids = _identity_set(owners, "owner_id", label="owner_id")
    observations_by_id = {
        observation["observation_id"]: observation for observation in observations
    }
    owners_by_id = {owner["owner_id"]: owner for owner in owners}
    _require_page_identity(components, page_id=page_id, label="component")
    _require_page_identity(observations, page_id=page_id, label="observation")
    _require_page_identity(owners, page_id=page_id, label="owner")

    _require_known_references(
        observations,
        field="component_ids",
        known=component_ids,
        label="observation",
    )
    _require_known_references(
        owners,
        field="component_ids",
        known=component_ids,
        label="owner",
    )
    _require_known_references(
        owners,
        field="observation_ids",
        known=observation_ids,
        label="owner",
    )
    _require_known_references(
        owners,
        field="selected_observation_ids",
        known=observation_ids,
        label="owner",
    )

    for owner in owners:
        owner_id = owner["owner_id"]
        owner_components = set(owner.get("component_ids") or [])
        evidence = [
            observations_by_id[observation_id]
            for observation_id in owner.get("observation_ids") or []
        ]
        selected = [
            observations_by_id[observation_id]
            for observation_id in owner.get("selected_observation_ids") or []
        ]
        evidence_coverage = {
            component_id
            for observation in evidence
            for component_id in observation.get("component_ids") or []
        }
        selected_coverage = {
            component_id
            for observation in selected
            for component_id in observation.get("component_ids") or []
        }
        if owner.get("disposition") == "owned" and not owner_components:
            raise OwnerProjectValidationError(
                f"owner graph owner_id {owner_id} owns no source components"
            )
        if owner.get("disposition") == "owned" and not selected:
            raise OwnerProjectValidationError(
                f"owner graph owner_id {owner_id} has no selected observation evidence"
            )
        if owner.get("disposition") == "owned" and not owner_components.issubset(
            evidence_coverage
        ):
            raise OwnerProjectValidationError(
                f"owner graph owner_id {owner_id} observations do not cover its components"
            )
        if owner.get("disposition") == "owned" and not owner_components.issubset(
            selected_coverage
        ):
            raise OwnerProjectValidationError(
                f"owner graph owner_id {owner_id} selected observations do not cover its components"
            )
        contaminated = sorted(selected_coverage - owner_components)
        if contaminated:
            raise OwnerProjectValidationError(
                f"owner graph owner_id {owner_id} selected observations contain foreign "
                f"components: {', '.join(contaminated)}"
            )

    projection_identities: set[tuple[str, str, str]] = set()
    for projection in projections:
        projection_owner = _nonempty_identity(
            projection.get("owner_id"), label="projection.owner_id"
        )
        if projection_owner not in owner_ids:
            raise OwnerProjectValidationError(
                f"owner graph projection references unknown owner_id: {projection_owner}"
            )
        projection_identity = (
            projection_owner,
            _nonempty_identity(projection.get("tile_id"), label="projection.tile_id"),
            _nonempty_identity(projection.get("role"), label="projection.role"),
        )
        if projection_identity in projection_identities:
            raise OwnerProjectValidationError(
                f"owner graph has duplicate projection: {'/'.join(projection_identity)}"
            )
        projection_identities.add(projection_identity)
    for disposition in dispositions:
        component_id = _nonempty_identity(
            disposition.get("component_id"), label="disposition.component_id"
        )
        if component_id not in component_ids:
            raise OwnerProjectValidationError(
                f"owner graph disposition references unknown component_id: {component_id}"
            )
        disposition_owner = disposition.get("owner_id")
        if (
            disposition_owner is not None
            and _nonempty_identity(disposition_owner, label="disposition.owner_id")
            not in owner_ids
        ):
            raise OwnerProjectValidationError(
                "owner graph disposition references unknown owner_id: "
                f"{disposition_owner}"
            )
        if disposition.get("decision") in {"owned", "review"}:
            canonical_owner_id = _nonempty_identity(
                disposition_owner, label="disposition.owner_id"
            )
            if component_id not in set(
                owners_by_id[canonical_owner_id].get("component_ids") or []
            ):
                raise OwnerProjectValidationError(
                    "owner graph disposition owner does not contain component_id: "
                    f"{canonical_owner_id}/{component_id}"
                )

    try:
        graph = OwnerGraph.from_dict(deepcopy(graph_payload), enforce=enforce)
        graph.require_valid(mode="enforce" if enforce else "shadow")
    except OwnerGraphValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise OwnerProjectValidationError(f"owner graph is invalid: {exc}") from exc
    return graph


def build_owner_invariant_summary(
    graphs: Iterable[OwnerGraph | dict[str, Any]],
) -> dict[str, int]:
    """Build the non-authoritative cached summary from validated graphs."""

    validated = [
        graph
        if isinstance(graph, OwnerGraph)
        else validate_serialized_owner_graph(graph, enforce=True)
        for graph in graphs
    ]
    violations = [violation for graph in validated for violation in graph.violations]
    return {
        "page_count": len(validated),
        "component_count": sum(len(graph.components) for graph in validated),
        "observation_count": sum(len(graph.observations) for graph in validated),
        "owner_count": sum(len(graph.owners) for graph in validated),
        "projection_count": sum(len(graph.projections) for graph in validated),
        "violation_count": len(violations),
        "critical_violation_count": sum(
            violation.severity == "critical" for violation in violations
        ),
    }


def _graph_payload_candidates(records: Iterable[Any]) -> Iterable[Any]:
    for record in records:
        if isinstance(record, OwnerGraph):
            yield record.to_dict()
            continue
        if not isinstance(record, dict):
            continue
        snapshot = record.get("_owner_graph_snapshot")
        if snapshot is not None:
            yield snapshot
        for key in ("_page_owner_graphs", "page_owner_graphs"):
            snapshots = record.get(key)
            if isinstance(snapshots, list):
                yield from snapshots


def serialize_page_owner_graphs(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Collect, validate, canonicalize and deduplicate page graph snapshots."""

    by_page: dict[str, dict[str, Any]] = {}
    owner_pages: dict[str, str] = {}
    component_pages: dict[str, str] = {}
    observation_pages: dict[str, str] = {}
    for payload in _graph_payload_candidates(records):
        graph = validate_serialized_owner_graph(payload, enforce=True)
        canonical = graph.to_dict()
        existing = by_page.get(graph.page_id)
        if existing is not None:
            if existing != canonical:
                raise OwnerProjectValidationError(
                    f"conflicting owner graphs for page_id {graph.page_id}"
                )
            continue
        by_page[graph.page_id] = canonical
        for identity, page_id, label in (
            *((owner.owner_id, graph.page_id, "owner_id") for owner in graph.owners),
            *(
                (component.component_id, graph.page_id, "component_id")
                for component in graph.components
            ),
            *(
                (observation.observation_id, graph.page_id, "observation_id")
                for observation in graph.observations
            ),
        ):
            registry = {
                "owner_id": owner_pages,
                "component_id": component_pages,
                "observation_id": observation_pages,
            }[label]
            previous_page = registry.get(identity)
            if previous_page is not None and previous_page != page_id:
                raise OwnerProjectValidationError(
                    f"duplicate {label} across owner graphs: {identity}"
                )
            registry[identity] = page_id
    return [by_page[page_id] for page_id in sorted(by_page)]


def build_owner_project_envelope(
    *record_groups: Iterable[Any],
) -> dict[str, Any]:
    """Build the additive project envelope without granting trust when absent."""

    records = [record for group in record_groups for record in group]
    graphs = serialize_page_owner_graphs(records)
    if not graphs:
        return {
            "owner_graph_schema_version": OWNER_GRAPH_SCHEMA_VERSION,
            "owner_graph_status": OWNER_GRAPH_STATUS_LEGACY_UNVERIFIED,
            "page_owner_graphs": [],
            "owner_invariant_summary": {},
        }
    return {
        "owner_graph_schema_version": OWNER_GRAPH_SCHEMA_VERSION,
        "owner_graph_status": OWNER_GRAPH_STATUS_VERIFIED,
        "page_owner_graphs": graphs,
        "owner_invariant_summary": build_owner_invariant_summary(graphs),
    }


OWNER_LAYER_CONTRACT_FIELDS = (
    "id", "owner_id", "candidate_owner_id", "page_id", "component_ids",
    "observation_ids", "selected_observation_ids", "semantic_role",
    "action_mask_ref", "layout_region_ids", "route_action", "render_policy",
    "disposition", "state", "execution_tile_id", "owner_graph_run_id",
    "owner_graph_origin_execution_id", "owner_graph_page_source_sha256",
    "execution_rejected", "derived_qa_status", "write_authority",
    "source_pixels_preserved", "committed", "blocking", "qa_action", "visible",
    "text", "original", "source_payload", "translated", "translated_payload",
    "translation_binding_sha256", "source_payload_sha256", "target_payload_sha256",
    "owner_execution_rejection_reason", "translation_attempt_ids",
    "translation_attempt_sha256s",
    "language_verdict", "translation_quality_usage_policy_id",
    "translation_quality_warning_reason", "translation_delivery_notice",
)


def is_review_candidate(layer: dict[str, Any]) -> bool:
    """Recognize candidate metadata without repairing an invalid identity."""
    return layer.get("candidate_owner_id") is not None


def preserve_owner_layer_contract(
    source: dict[str, Any], target: dict[str, Any],
) -> dict[str, Any]:
    """Copy the canonical ownership contract through compatibility projections.

    Values are copied, never inferred or repaired. Invalid authority and hashes
    remain invalid so the persistence validator can reject them.
    """
    if source.get("owner_id") is None and not is_review_candidate(source):
        return target
    for field in OWNER_LAYER_CONTRACT_FIELDS:
        if field in source:
            target[field] = deepcopy(source[field])
        else:
            target.pop(field, None)
    return target


def neutralize_review_candidate_compatibility(layer: dict[str, Any]) -> bool:
    """Neutralize legacy execution hints without changing candidate evidence."""
    if not is_review_candidate(layer):
        return False
    layer["skip_processing"] = True
    layer["preserve_original"] = True
    layer["translate_policy"] = "skip_translation"
    return True


def detach_owner_review_candidate(
    record: dict[str, Any], graph: OwnerGraph, owner_id: str,
) -> dict[str, Any]:
    """Build the shared non-authoritative record for a retired owner."""
    record = deepcopy(record)
    record.update({
        "id": owner_id, "owner_id": None, "candidate_owner_id": owner_id,
        "owner_graph_run_id": graph.run_id,
        "owner_graph_origin_execution_id": graph.origin_execution_id,
        "owner_graph_page_source_sha256": graph.page_source_sha256,
        "disposition": "review", "state": "review_required",
        "route_action": "review_required", "render_policy": "review_required",
        "execution_tile_id": None, "action_mask_ref": None, "layout_region_ids": [],
        "execution_rejected": True, "derived_qa_status": "review_required",
        "write_authority": "revoked", "source_pixels_preserved": True,
        "committed": False, "blocking": True, "qa_action": "BLOCK", "visible": False,
    })
    neutralize_review_candidate_compatibility(record)
    return record


def normalize_owner_text_layer_for_project(layer: dict[str, Any]) -> dict[str, Any]:
    """Materialize additive owner fields without deriving an identity."""

    normalized = deepcopy(layer)
    if normalized.get("owner_id") is None:
        return normalized
    for field in ("component_ids", "observation_ids", "layout_region_ids"):
        value = normalized.get(field)
        normalized[field] = list(value) if isinstance(value, (list, tuple)) else []
    normalized.setdefault("semantic_role", None)
    normalized.setdefault("route_action", None)
    normalized.setdefault("action_mask_ref", None)
    return normalized


def _project_pages(project: dict[str, Any]) -> list[Any] | None:
    containers = [
        (name, project.get(name))
        for name in ("paginas", "pages")
        if isinstance(project.get(name), list)
    ]
    populated = [(name, pages) for name, pages in containers if pages]
    if len(populated) > 1:
        names = ", ".join(name for name, _ in populated)
        raise OwnerProjectValidationError(
            f"verified project has multiple populated page containers: {names}"
        )
    if populated:
        return populated[0][1]
    return containers[0][1] if containers else None


def _expected_page_count(project: dict[str, Any]) -> int | None:
    pages = _project_pages(project)
    if pages is not None:
        return len(pages)
    source = project.get("source")
    if isinstance(source, dict) and isinstance(source.get("page_count"), int):
        return int(source["page_count"])
    return None


def _page_layer_collections(
    page: Any,
) -> list[tuple[str, list[dict[str, Any]]]]:
    if not isinstance(page, dict):
        return []
    collections: list[tuple[str, list[dict[str, Any]]]] = []
    for key in ("text_layers", "textos", "texts", "regions"):
        value = page.get(key)
        if isinstance(value, list):
            collections.append(
                (key, [item for item in value if isinstance(item, dict)])
            )
    return collections


def _container_page_id(page: Any, page_index: int) -> str:
    if not isinstance(page, dict):
        raise OwnerProjectValidationError(
            f"owner graph project page {page_index} must be an object"
        )
    for field in ("numero", "page"):
        value = page.get(field)
        if value is None:
            continue
        if type(value) is not int or value < 1:
            raise OwnerProjectValidationError(
                f"owner graph page container {field} must be a positive integer"
            )
        return f"page_{value:03d}"
    explicit = page.get("page_id")
    if explicit is not None:
        return _nonempty_identity(explicit, label="page container page_id")
    return f"page_{page_index:03d}"


def _layer_owner_contract(layer: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(sorted(layer.get("component_ids") or [])),
        tuple(sorted(layer.get("observation_ids") or [])),
        layer.get("semantic_role"),
        layer.get("route_action"),
        layer.get("action_mask_ref"),
        tuple(layer.get("layout_region_ids") or []),
    )


def _retired_review_candidate_contract(
    layer: dict[str, Any],
    *,
    alias_name: str,
    page_id: str,
    graph: dict[str, Any] | None,
    owner_ids: set[str],
) -> tuple[str | None, tuple[Any, ...] | None, list[str]]:
    """Validate a detached review candidate against its current source graph.

    A candidate can retain source and proposed target text for editor review after
    a rejected owner is removed from an enforce graph. It never represents an
    owner and must carry no write, mask, or render authority.
    """

    errors: list[str] = []
    candidate_value = layer.get("candidate_owner_id")
    try:
        candidate_id = _nonempty_identity(
            candidate_value, label=f"{alias_name}.candidate_owner_id"
        )
    except OwnerProjectValidationError as exc:
        errors.append(str(exc))
        return None, None, errors

    if "owner_id" not in layer or layer.get("owner_id") is not None:
        errors.append(
            f"{alias_name} review candidate {candidate_id} must have owner_id=None"
        )
    try:
        layer_id = _nonempty_identity(layer.get("id"), label=f"{alias_name}.id")
        if layer_id != candidate_id:
            errors.append(
                f"{alias_name} review candidate id does not match candidate_owner_id"
            )
    except OwnerProjectValidationError as exc:
        errors.append(str(exc))
    if candidate_id in owner_ids:
        errors.append(
            f"review candidate identity collides with canonical owner_id: {candidate_id}"
        )

    try:
        layer_page_id = _nonempty_identity(
            layer.get("page_id"), label=f"{alias_name}.page_id"
        )
        if layer_page_id != page_id:
            errors.append(
                f"{alias_name} review candidate {candidate_id} page mismatch: "
                f"container={page_id}, layer={layer_page_id}"
            )
    except OwnerProjectValidationError as exc:
        errors.append(str(exc))

    if graph is None:
        errors.append(
            f"{alias_name} review candidate {candidate_id} has no current page graph"
        )
    else:
        graph_bindings = {
            "owner_graph_run_id": graph.get("run_id"),
            "owner_graph_origin_execution_id": graph.get("origin_execution_id"),
            "owner_graph_page_source_sha256": graph.get("page_source_sha256"),
        }
        for field, expected in graph_bindings.items():
            try:
                actual = _nonempty_identity(
                    layer.get(field), label=f"{alias_name}.{field}"
                )
            except OwnerProjectValidationError as exc:
                errors.append(str(exc))
                continue
            if actual != expected:
                errors.append(
                    f"{alias_name} review candidate {candidate_id} has stale {field}"
                )

    required_values = {
        "disposition": "review",
        "state": "review_required",
        "route_action": "review_required",
        "derived_qa_status": "review_required",
        "execution_rejected": True,
        "write_authority": "revoked",
        "source_pixels_preserved": True,
        "committed": False,
        "blocking": True,
        "qa_action": "BLOCK",
        "visible": False,
        "render_policy": "review_required",
        "action_mask_ref": None,
        "execution_tile_id": None,
    }
    for field, expected in required_values.items():
        if layer.get(field) != expected or type(layer.get(field)) is not type(expected):
            errors.append(
                f"{alias_name} review candidate {candidate_id} has unsafe {field}"
            )
    if layer.get("layout_region_ids") != [] or type(layer.get("layout_region_ids")) is not list:
        errors.append(
            f"{alias_name} review candidate {candidate_id} has unsafe layout_region_ids"
        )

    def identity_list(field: str, *, required: bool) -> list[str]:
        value = layer.get(field)
        if not isinstance(value, list) or (required and not value):
            errors.append(
                f"{alias_name} review candidate {candidate_id} has invalid {field}"
            )
            return []
        values: list[str] = []
        for item in value:
            try:
                values.append(_nonempty_identity(item, label=f"{alias_name}.{field}"))
            except OwnerProjectValidationError as exc:
                errors.append(str(exc))
        if len(values) != len(set(values)):
            errors.append(
                f"{alias_name} review candidate {candidate_id} has duplicate {field}"
            )
        return values

    component_ids = identity_list("component_ids", required=True)
    observation_ids = identity_list("observation_ids", required=True)
    selected_observation_ids = identity_list(
        "selected_observation_ids", required=True
    )
    if not set(selected_observation_ids).issubset(observation_ids):
        errors.append(
            f"{alias_name} review candidate {candidate_id} selects unknown observations"
        )

    source_payload = layer.get("source_payload")
    if not isinstance(source_payload, str) or not source_payload.strip():
        errors.append(
            f"{alias_name} review candidate {candidate_id} is missing source text"
        )
        source_payload = ""
    for field in ("text", "original"):
        if layer.get(field) != source_payload:
            errors.append(
                f"{alias_name} review candidate {candidate_id} has mismatched {field}"
            )
    translated_payload = layer.get("translated_payload")
    if translated_payload is not None and not isinstance(translated_payload, str):
        errors.append(
            f"{alias_name} review candidate {candidate_id} has invalid translated_payload"
        )
    translated = layer.get("translated")
    expected_translated = translated_payload or ""
    if translated != expected_translated:
        errors.append(
            f"{alias_name} review candidate {candidate_id} has mismatched translated text"
        )
    from .hash_contract import sha256_text
    for field, payload in (("source_payload_sha256", source_payload),
                           ("target_payload_sha256", expected_translated)):
        if field not in layer:
            continue
        value = layer[field]
        if value != sha256_text(payload):
            errors.append(
                f"{alias_name} review candidate {candidate_id} has mismatched {field}"
            )
    if "translation_binding_sha256" in layer and (
        not isinstance(layer["translation_binding_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", layer["translation_binding_sha256"]) is None
    ):
        errors.append(f"{alias_name} review candidate {candidate_id} has invalid translation_binding_sha256")

    if graph is not None:
        components = {
            str(item.get("component_id")): item
            for item in graph.get("components") or []
            if isinstance(item, dict)
        }
        observations = {
            str(item.get("observation_id")): item
            for item in graph.get("observations") or []
            if isinstance(item, dict)
        }
        dispositions = {
            str(item.get("component_id")): item
            for item in graph.get("component_dispositions") or []
            if isinstance(item, dict)
        }
        missing_components = sorted(set(component_ids) - set(components))
        missing_observations = sorted(set(observation_ids) - set(observations))
        if missing_components:
            errors.append(
                f"{alias_name} review candidate {candidate_id} references unknown components: "
                f"{', '.join(missing_components)}"
            )
        if missing_observations:
            errors.append(
                f"{alias_name} review candidate {candidate_id} references unknown observations: "
                f"{', '.join(missing_observations)}"
            )
        expected_evidence: set[str] = set()
        retired_reasons: set[str] = set()
        policy_reasons: set[str] = set()
        for component_id in component_ids:
            disposition = dispositions.get(component_id)
            if not isinstance(disposition, dict) or any(
                disposition.get(field) != expected
                for field, expected in (
                    ("decision", "uncertain"),
                    ("owner_id", None),
                    ("policy_id", "coverage_ambiguous_candidate"),
                )
            ):
                errors.append(
                    f"{alias_name} review candidate {candidate_id} is not bound to "
                    f"a retired uncertain component: {component_id}"
                )
                continue
            reason = disposition.get("reason")
            policy_reason = disposition.get("policy_reason")
            if not isinstance(reason, str) or not reason:
                errors.append(
                    f"{alias_name} review candidate {candidate_id} has no retirement reason"
                )
            else:
                retired_reasons.add(reason)
            if not isinstance(policy_reason, str) or not policy_reason:
                errors.append(
                    f"{alias_name} review candidate {candidate_id} has no policy reason"
                )
            else:
                policy_reasons.add(policy_reason)
            evidence_ids = disposition.get("policy_evidence_ids")
            if not isinstance(evidence_ids, list) or not evidence_ids:
                errors.append(
                    f"{alias_name} review candidate {candidate_id} has invalid "
                    f"policy evidence for {component_id}"
                )
                continue
            for evidence_id in evidence_ids:
                try:
                    expected_evidence.add(
                        _nonempty_identity(
                            evidence_id,
                            label=f"{alias_name}.policy_evidence_ids",
                        )
                    )
                except OwnerProjectValidationError as exc:
                    errors.append(str(exc))

        if retired_reasons == {"owner_execution_rejected"}:
            expected_policy_reason = "source pixels preserved after owner execution rejection"
            if policy_reasons != {expected_policy_reason}:
                errors.append(
                    f"{alias_name} review candidate {candidate_id} has invalid "
                    "owner execution rejection policy"
                )
            if expected_evidence != set(observation_ids):
                errors.append(
                    f"{alias_name} review candidate {candidate_id} does not match "
                    "current rejected-owner OCR evidence"
                )
        elif retired_reasons == {"owner_translation_rejected"}:
            attempt_ids = identity_list("translation_attempt_ids", required=True)
            attempt_hashes = layer.get("translation_attempt_sha256s")
            if (
                not isinstance(attempt_hashes, list)
                or len(attempt_hashes) != len(attempt_ids)
                or any(
                    not isinstance(value, str)
                    or re.fullmatch(r"[0-9a-f]{64}", value) is None
                    for value in (attempt_hashes or [])
                )
            ):
                errors.append(
                    f"{alias_name} review candidate {candidate_id} has invalid "
                    "translation attempt hashes"
                )
            if expected_evidence != set(attempt_ids):
                errors.append(
                    f"{alias_name} review candidate {candidate_id} does not match "
                    "current rejected translation attempts"
                )
            if len(attempt_ids) != len(set(attempt_ids)):
                errors.append(
                    f"{alias_name} review candidate {candidate_id} has duplicate "
                    "translation attempts"
                )
            valid_hashes = [
                value for value in attempt_hashes or [] if isinstance(value, str)
            ]
            if len(valid_hashes) != len(set(valid_hashes)):
                errors.append(
                    f"{alias_name} review candidate {candidate_id} has duplicate "
                    "translation attempt hashes"
                )
            if not policy_reasons or any(
                "translation validation exhausted" not in value
                or "source pixels preserved" not in value
                for value in policy_reasons
            ):
                errors.append(
                    f"{alias_name} review candidate {candidate_id} has invalid "
                    "translation rejection policy"
                )
        else:
            errors.append(
                f"{alias_name} review candidate {candidate_id} has unsupported "
                f"retirement reasons: {sorted(retired_reasons)}"
            )
        for observation_id in observation_ids:
            observation = observations.get(observation_id)
            if not isinstance(observation, dict):
                continue
            observation_components = set(observation.get("component_ids") or [])
            if not observation_components.intersection(component_ids):
                errors.append(
                    f"{alias_name} review candidate {candidate_id} includes "
                    f"unrelated observation: {observation_id}"
                )

    contract = (
        candidate_id,
        layer.get("page_id"),
        tuple(component_ids),
        tuple(observation_ids),
        tuple(selected_observation_ids),
        layer.get("semantic_role"),
        layer.get("route_action"),
        layer.get("action_mask_ref"),
        tuple(layer.get("layout_region_ids") or []),
        layer.get("disposition"),
        layer.get("state"),
        layer.get("execution_rejected"),
        layer.get("write_authority"),
        layer.get("source_pixels_preserved"),
        layer.get("committed"),
        layer.get("blocking"),
        layer.get("qa_action"),
        layer.get("visible"),
        layer.get("render_policy"),
        source_payload,
        translated_payload,
        translated,
        layer.get("owner_graph_run_id"),
        layer.get("owner_graph_origin_execution_id"),
        layer.get("owner_graph_page_source_sha256"),
        tuple(layer.get("translation_attempt_ids") or []),
        tuple(layer.get("translation_attempt_sha256s") or []),
        layer.get("source_payload_sha256"),
        layer.get("target_payload_sha256"),
        layer.get("translation_binding_sha256"),
    )
    return candidate_id, contract, errors


def owner_project_validation_errors(
    project: dict[str, Any],
    *,
    require_envelope: bool = False,
) -> list[str]:
    """Return fail-closed ownership envelope errors without mutating a project."""

    present = [field in project for field in OWNER_PROJECT_FIELDS]
    if not any(present):
        return ["missing owner graph project envelope"] if require_envelope else []
    if not all(present):
        missing = [
            field
            for field, is_present in zip(OWNER_PROJECT_FIELDS, present)
            if not is_present
        ]
        return [f"incomplete owner graph project envelope: {', '.join(missing)}"]

    errors: list[str] = []
    if project.get("owner_graph_schema_version") != OWNER_GRAPH_SCHEMA_VERSION:
        errors.append(
            f"owner_graph_schema_version must be {OWNER_GRAPH_SCHEMA_VERSION}"
        )
    status = project.get("owner_graph_status")
    if status not in OWNER_GRAPH_STATUSES:
        errors.append(f"owner_graph_status is invalid: {status!r}")
    raw_graphs = project.get("page_owner_graphs")
    if not isinstance(raw_graphs, list):
        errors.append("page_owner_graphs must be a list")
        raw_graphs = []
    summary = project.get("owner_invariant_summary")
    if not isinstance(summary, dict):
        errors.append("owner_invariant_summary must be an object")
        summary = {}

    if status == OWNER_GRAPH_STATUS_LEGACY_UNVERIFIED:
        if raw_graphs:
            errors.append(
                "legacy_unverified project cannot carry trusted page owner graphs"
            )
        if summary:
            errors.append(
                "legacy_unverified project cannot publish owner invariant summary claims"
            )
        return errors
    if status != OWNER_GRAPH_STATUS_VERIFIED:
        return errors
    if not raw_graphs:
        errors.append("verified owner graph status requires page_owner_graphs")
        return errors

    try:
        canonical = serialize_page_owner_graphs([{"page_owner_graphs": raw_graphs}])
        if len(canonical) != len(raw_graphs):
            errors.append("verified project contains duplicate page owner graphs")
        expected_count = _expected_page_count(project)
        if expected_count is not None and len(canonical) != expected_count:
            errors.append(
                "verified owner graph page count mismatch: "
                f"expected {expected_count}, got {len(canonical)}"
            )
        expected_summary = build_owner_invariant_summary(canonical)
        summary_fields = set(summary)
        expected_summary_fields = set(OWNER_SUMMARY_FIELDS)
        if summary_fields != expected_summary_fields:
            errors.append(
                "owner_invariant_summary fields must match the verified contract: "
                f"expected {sorted(expected_summary_fields)}, got {sorted(summary_fields)}"
            )
        summary_types_valid = True
        for field in OWNER_SUMMARY_FIELDS:
            value = summary.get(field)
            if type(value) is not int or value < 0:
                summary_types_valid = False
                errors.append(
                    f"owner_invariant_summary.{field} must be a non-negative integer"
                )
        if summary_types_valid and summary != expected_summary:
            errors.append(
                "owner_invariant_summary does not match page_owner_graphs: "
                f"expected {expected_summary}, got {summary}"
            )

        owners: dict[str, tuple[str, dict[str, Any]]] = {}
        for graph in canonical:
            graph_page_id = str(graph["page_id"])
            for owner in graph["owners"]:
                owner_id = str(owner["owner_id"])
                if owner_id in owners:
                    errors.append(f"duplicate persisted owner_id: {owner_id}")
                owners[owner_id] = (graph_page_id, owner)

        pages = _project_pages(project)
        page_records = pages or []
        container_page_ids = [
            _container_page_id(page, page_index)
            for page_index, page in enumerate(page_records, start=1)
        ]
        if len(set(container_page_ids)) != len(container_page_ids):
            errors.append(
                "verified project contains duplicate page container identities"
            )
        graph_page_ids = {str(graph["page_id"]) for graph in canonical}
        if pages is not None and graph_page_ids != set(container_page_ids):
            errors.append(
                "verified owner graph page identities do not match project containers: "
                f"graphs={sorted(graph_page_ids)}, containers={sorted(container_page_ids)}"
            )

        materialized_owner_pages: dict[str, str] = {}
        materialized_candidate_pages: dict[str, str] = {}
        graphs_by_page = {str(graph["page_id"]): graph for graph in canonical}
        for page_index, page in enumerate(page_records, start=1):
            container_page_id = container_page_ids[page_index - 1]
            alias_contracts: list[tuple[str, dict[str, tuple[Any, ...]]]] = []
            for alias_name, layers in _page_layer_collections(page):
                contracts: dict[str, tuple[Any, ...]] = {}
                for layer in layers:
                    owner_value = layer.get("owner_id")
                    if owner_value is None:
                        candidate_id, candidate_contract, candidate_errors = (
                            _retired_review_candidate_contract(
                                layer,
                                alias_name=alias_name,
                                page_id=container_page_id,
                                graph=graphs_by_page.get(container_page_id),
                                owner_ids=set(owners),
                            )
                        )
                        errors.extend(candidate_errors)
                        if candidate_id is None or candidate_contract is None:
                            continue
                        contract_key = f"candidate:{candidate_id}"
                        if contract_key in contracts:
                            errors.append(
                                f"{alias_name} text layer duplicates candidate_owner_id: "
                                f"{candidate_id}"
                            )
                            continue
                        contracts[contract_key] = candidate_contract
                        continue
                    owner_id = _nonempty_identity(
                        owner_value, label=f"{alias_name}.owner_id"
                    )
                    if owner_id in contracts:
                        errors.append(
                            f"{alias_name} text layer duplicates owner_id: {owner_id}"
                        )
                        continue
                    if owner_id not in owners:
                        errors.append(
                            f"text layer references unknown owner_id: {owner_id!r}"
                        )
                        continue
                    graph_page_id, owner = owners[owner_id]
                    layer_page_id = _nonempty_identity(
                        layer.get("page_id"), label=f"{alias_name}.page_id"
                    )
                    if (
                        graph_page_id != container_page_id
                        or layer_page_id != container_page_id
                    ):
                        errors.append(
                            f"text layer owner_id {owner_id} page mismatch: "
                            f"graph={graph_page_id}, container={container_page_id}, "
                            f"layer={layer_page_id}"
                        )
                    for field in ("component_ids", "observation_ids"):
                        layer_ids = layer.get(field)
                        if not isinstance(layer_ids, list):
                            errors.append(
                                f"text layer owner_id {owner_id} is missing {field}"
                            )
                            continue
                        canonical_layer_ids = sorted(
                            _nonempty_identity(value, label=f"{alias_name}.{field}")
                            for value in layer_ids
                        )
                        if canonical_layer_ids != sorted(owner.get(field) or []):
                            errors.append(
                                f"text layer owner_id {owner_id} has mismatched {field}"
                            )
                    for field in ("semantic_role", "route_action"):
                        layer_value = _nonempty_identity(
                            layer.get(field), label=f"{alias_name}.{field}"
                        )
                        if layer_value != owner.get(field):
                            errors.append(
                                f"text layer owner_id {owner_id} has mismatched {field}"
                            )
                    layout_region_ids = layer.get("layout_region_ids")
                    if not isinstance(layout_region_ids, list):
                        errors.append(
                            f"text layer owner_id {owner_id} is missing layout_region_ids"
                        )
                    else:
                        for value in layout_region_ids:
                            _nonempty_identity(
                                value, label=f"{alias_name}.layout_region_ids"
                            )
                    action_mask_ref = layer.get("action_mask_ref")
                    if action_mask_ref is not None:
                        _nonempty_identity(
                            action_mask_ref, label=f"{alias_name}.action_mask_ref"
                        )
                    contracts[owner_id] = _layer_owner_contract(layer)
                alias_contracts.append((alias_name, contracts))

            if alias_contracts:
                baseline_name, baseline_contract = alias_contracts[0]
                for alias_name, contract in alias_contracts[1:]:
                    if contract != baseline_contract:
                        errors.append(
                            f"owner layer aliases diverge: {baseline_name} != {alias_name}"
                        )
                for owner_id in {
                    owner_id for _, contract in alias_contracts for owner_id in contract
                }:
                    previous_page = materialized_owner_pages.get(owner_id)
                    if previous_page is not None and previous_page != container_page_id:
                        errors.append(
                            f"owner_id {owner_id} is materialized on multiple pages"
                        )
                    materialized_owner_pages[owner_id] = container_page_id

            for _, contract in alias_contracts:
                for contract_key in contract:
                    if not contract_key.startswith("candidate:"):
                        continue
                    candidate_id = contract_key.removeprefix("candidate:")
                    previous_page = materialized_candidate_pages.get(candidate_id)
                    if previous_page is not None and previous_page != container_page_id:
                        errors.append(
                            f"candidate_owner_id {candidate_id} is materialized on multiple pages"
                        )
                    materialized_candidate_pages[candidate_id] = container_page_id

        missing_owner_layers = sorted(set(owners) - set(materialized_owner_pages))
        if pages is not None and missing_owner_layers:
            errors.append(
                "verified owners are missing project text layers: "
                f"{', '.join(missing_owner_layers)}"
            )
    except (OwnerProjectValidationError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def require_owner_project_consistency(
    project: dict[str, Any],
    *,
    require_envelope: bool = False,
) -> None:
    errors = owner_project_validation_errors(project, require_envelope=require_envelope)
    if errors:
        raise OwnerProjectValidationError("; ".join(errors))


def verified_owner_graphs_from_project(project: dict[str, Any]) -> list[OwnerGraph]:
    """Load graphs only when the persisted project explicitly proves verification."""

    require_owner_project_consistency(project, require_envelope=True)
    if project.get("owner_graph_status") != OWNER_GRAPH_STATUS_VERIFIED:
        raise OwnerProjectValidationError(
            "legacy_unverified project must reprocess the page before owner execution"
        )
    return [
        validate_serialized_owner_graph(payload, enforce=True)
        for payload in project.get("page_owner_graphs") or []
    ]
