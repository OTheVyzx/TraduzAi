"""Project persistence boundary for verified page-global owner graphs.

The editor project and schema-v12 project use different page shapes.  This
module owns the additive ownership envelope shared by both formats and keeps
legacy projects explicitly outside the verified execution path.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

from .model import OwnerGraph, OwnerGraphValidationError


OWNER_GRAPH_SCHEMA_VERSION = 1
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


def validate_serialized_owner_graph(graph_payload: Any) -> OwnerGraph:
    """Validate a serialized graph more strictly than the tolerant reader."""

    if not isinstance(graph_payload, dict):
        raise OwnerProjectValidationError("owner graph must be an object")
    if graph_payload.get("schema_version") != OWNER_GRAPH_SCHEMA_VERSION:
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
        graph = OwnerGraph.from_dict(deepcopy(graph_payload))
        graph.require_valid()
    except (OwnerGraphValidationError, TypeError, ValueError) as exc:
        raise OwnerProjectValidationError(f"owner graph is invalid: {exc}") from exc
    return graph


def build_owner_invariant_summary(
    graphs: Iterable[OwnerGraph | dict[str, Any]],
) -> dict[str, int]:
    """Build the non-authoritative cached summary from validated graphs."""

    validated = [
        graph
        if isinstance(graph, OwnerGraph)
        else validate_serialized_owner_graph(graph)
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
        graph = validate_serialized_owner_graph(payload)
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
        for page_index, page in enumerate(page_records, start=1):
            container_page_id = container_page_ids[page_index - 1]
            alias_contracts: list[tuple[str, dict[str, tuple[Any, ...]]]] = []
            for alias_name, layers in _page_layer_collections(page):
                contracts: dict[str, tuple[Any, ...]] = {}
                for layer in layers:
                    owner_value = layer.get("owner_id")
                    if owner_value is None:
                        errors.append(
                            f"verified {alias_name} text layer is missing owner_id"
                        )
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
        validate_serialized_owner_graph(payload)
        for payload in project.get("page_owner_graphs") or []
    ]
