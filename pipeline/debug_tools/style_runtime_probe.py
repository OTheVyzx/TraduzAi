"""Read-only owner style runtime contract probe."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping


_RENDERABLE_ROUTES = frozenset(
    {
        "translate_inpaint_render",
        "translate_sfx_inpaint_render",
    }
)


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        value = value.get("texts") or value.get("owners") or []
    if not isinstance(value, (list, tuple)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _project_layers(project: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    pages = project.get("paginas") or project.get("pages") or []
    layers: list[Mapping[str, Any]] = []
    for page in _mapping_rows(pages):
        layers.extend(_mapping_rows(page.get("text_layers") or page.get("texts")))
    return layers


def _rendered_owners(project: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rendered: dict[str, Mapping[str, Any]] = {}
    graphs = project.get("page_owner_graphs") or []
    for graph in _mapping_rows(graphs):
        for owner in _mapping_rows(graph.get("owners")):
            owner_id = str(owner.get("owner_id") or "").strip()
            state = str(owner.get("state") or "").strip().lower()
            route = str(owner.get("route_action") or "").strip().lower()
            if owner_id and (state == "rendered" or route in _RENDERABLE_ROUTES):
                rendered.setdefault(owner_id, owner)
    return rendered


def _owner_categories(owner: Mapping[str, Any]) -> set[str]:
    raw = owner.get("style_categories") or owner.get("categories") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return set()
    return {str(value).strip() for value in raw if str(value).strip()}


def probe_style_runtime(
    project: Mapping[str, Any],
    required_categories: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Summarize owner style runtime contracts without mutating the project."""

    rendered = _rendered_owners(project)
    layers_by_owner: dict[str, Mapping[str, Any]] = {}
    duplicate_layer_ids: set[str] = set()
    for layer in _project_layers(project):
        owner_id = str(layer.get("owner_id") or "").strip()
        if not owner_id:
            continue
        if owner_id in layers_by_owner:
            duplicate_layer_ids.add(owner_id)
        else:
            layers_by_owner[owner_id] = layer

    contracts: set[str] = set()
    status_counts: Counter[str] = Counter()
    abstention_reasons: Counter[str] = Counter()
    profile_count = 0
    raster_contract_count = 0
    applied_attribute_count = 0
    abstained_attribute_count = 0

    if not rendered:
        contracts.add("no_rendered_owners")
    if duplicate_layer_ids:
        contracts.add("duplicate_owner_layer")

    for owner_id in sorted(rendered):
        layer = layers_by_owner.get(owner_id)
        if layer is None:
            contracts.add("missing_rendered_owner_layer")
            continue
        profile = layer.get("visual_profile_v2")
        if not isinstance(profile, Mapping):
            contracts.add("missing_visual_profile")
            continue
        if str(profile.get("owner_id") or "").strip() != owner_id:
            contracts.add("owner_identity_mismatch")
            continue

        profile_count += 1
        profile_status = str(profile.get("status") or "not_scanned").strip().lower()
        status_counts[profile_status] += 1
        decision = profile.get("style_application_decision_v2")
        decision = decision if isinstance(decision, Mapping) else {}
        applied = decision.get("applied_attributes")
        applied = applied if isinstance(applied, Mapping) else {}
        abstained = decision.get("abstained_attributes")
        abstained = abstained if isinstance(abstained, Mapping) else {}
        applied_attribute_count += len(applied)
        abstained_attribute_count += len(abstained)
        abstention_reasons.update(str(reason) for reason in abstained.values())

        raster = layer.get("style_v2_raster_contract")
        if not isinstance(raster, Mapping):
            contracts.add("missing_raster_contract")
            continue
        raster_owner_id = str(raster.get("owner_id") or owner_id).strip()
        if raster_owner_id != owner_id:
            contracts.add("owner_identity_mismatch")
            continue
        raster_contract_count += 1

    requirements = {
        str(category).strip(): max(0, int(minimum))
        for category, minimum in dict(required_categories or {}).items()
        if str(category).strip()
    }
    category_counts: Counter[str] = Counter()
    for owner in rendered.values():
        category_counts.update(_owner_categories(owner))
    all_categories = sorted(set(category_counts) | set(requirements))
    category_metrics: dict[str, dict[str, Any]] = {}
    for category in all_categories:
        owner_count = int(category_counts[category])
        required_minimum = int(requirements.get(category, 0))
        complete = owner_count >= required_minimum
        if not complete:
            contracts.add("category_coverage_incomplete")
        category_metrics[category] = {
            "owner_count": owner_count,
            "required_minimum": required_minimum,
            "complete": complete,
        }

    rendered_owner_count = len(rendered)
    profile_coverage = (
        profile_count / rendered_owner_count if rendered_owner_count else 0.0
    )
    raster_coverage = (
        raster_contract_count / rendered_owner_count if rendered_owner_count else 0.0
    )
    return {
        "schema_version": 1,
        "status": "BLOCK" if contracts else "PASS",
        "contracts": sorted(contracts),
        "rendered_owner_count": rendered_owner_count,
        "profile_count": profile_count,
        "profile_coverage": profile_coverage,
        "raster_contract_count": raster_contract_count,
        "raster_contract_coverage": raster_coverage,
        "applied_attribute_count": applied_attribute_count,
        "abstained_attribute_count": abstained_attribute_count,
        "profile_status_counts": dict(sorted(status_counts.items())),
        "abstention_reasons": dict(sorted(abstention_reasons.items())),
        "category_metrics": category_metrics,
    }


def _aggregate_reports(
    rows: list[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    contracts: set[str] = set()
    status_counts: Counter[str] = Counter()
    abstention_reasons: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    rendered_owner_count = 0
    profile_count = 0
    raster_contract_count = 0
    applied_attribute_count = 0
    abstained_attribute_count = 0
    projects: list[dict[str, Any]] = []
    for path, report in rows:
        contracts.update(str(code) for code in report.get("contracts") or [])
        status_counts.update(dict(report.get("profile_status_counts") or {}))
        abstention_reasons.update(dict(report.get("abstention_reasons") or {}))
        for category, metrics in dict(report.get("category_metrics") or {}).items():
            if isinstance(metrics, Mapping):
                category_counts[str(category)] += int(metrics.get("owner_count") or 0)
        rendered_owner_count += int(report.get("rendered_owner_count") or 0)
        profile_count += int(report.get("profile_count") or 0)
        raster_contract_count += int(report.get("raster_contract_count") or 0)
        applied_attribute_count += int(report.get("applied_attribute_count") or 0)
        abstained_attribute_count += int(report.get("abstained_attribute_count") or 0)
        projects.append(
            {
                "project_path": str(path.resolve()),
                "status": str(report.get("status") or "BLOCK"),
                "rendered_owner_count": int(
                    report.get("rendered_owner_count") or 0
                ),
                "contracts": list(report.get("contracts") or []),
            }
        )
    return {
        "schema_version": 1,
        "status": "BLOCK" if contracts else "PASS",
        "contracts": sorted(contracts),
        "project_count": len(rows),
        "rendered_owner_count": rendered_owner_count,
        "profile_count": profile_count,
        "profile_coverage": (
            profile_count / rendered_owner_count if rendered_owner_count else 0.0
        ),
        "raster_contract_count": raster_contract_count,
        "raster_contract_coverage": (
            raster_contract_count / rendered_owner_count
            if rendered_owner_count
            else 0.0
        ),
        "applied_attribute_count": applied_attribute_count,
        "abstained_attribute_count": abstained_attribute_count,
        "profile_status_counts": dict(sorted(status_counts.items())),
        "abstention_reasons": dict(sorted(abstention_reasons.items())),
        "category_metrics": {
            category: {"owner_count": count}
            for category, count in sorted(category_counts.items())
        },
        "projects": projects,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit owner style runtime contracts in project.json files."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "projects",
        nargs="+",
        type=Path,
        help="One or more project.json files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows: list[tuple[Path, Mapping[str, Any]]] = []
    for path in args.projects:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"project JSON root must be an object: {path}")
        rows.append((path, probe_style_runtime(payload)))
    report = _aggregate_reports(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
