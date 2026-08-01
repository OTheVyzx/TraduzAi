"""Owner-scoped source-to-final style fidelity audit and rollout gate."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ownership.model import validate_owner_style_raster_contract
from ownership.project import validate_serialized_owner_graph
from qa.style_fidelity_policy import (
    CATASTROPHIC_STYLE_ATTRIBUTES,
    HIGH_CONFIDENCE_THRESHOLD,
    REQUIRED_RENDER_METRICS,
    is_rendered_owner,
    required_style_categories,
)
from typesetter.owner_style import validate_owner_visual_profile
from typesetter.style_materialization import delta_e_2000


STYLE_MODES = frozenset({"shadow", "render", "enforce"})


def resolve_original_path(run_dir: Path, page: dict[str, Any], page_number: int) -> Path | None:
    root = Path(run_dir).resolve()
    for key in ("original_path", "source_path", "image_path", "arquivo", "path"):
        raw = page.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        candidate = Path(raw)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if candidate.is_file():
            return candidate.resolve()
    for folder in ("originals", "images"):
        for stem in (f"{page_number:03d}", f"page_{page_number:03d}", str(page_number)):
            for suffix in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = root / folder / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate.resolve()
    return None


def _same_value(expected: Any, observed: Any) -> bool:
    if isinstance(expected, str) and isinstance(observed, str):
        return expected.strip().upper() == observed.strip().upper()
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return abs(float(expected) - float(observed)) <= max(0.05, abs(float(expected)) * 0.08)
    return expected == observed


def _attribute_confidence(profile: dict[str, Any], name: str) -> float:
    evidence = profile.get("style_evidence_v2") if isinstance(profile.get("style_evidence_v2"), dict) else {}
    attributes = evidence.get("attributes") if isinstance(evidence.get("attributes"), dict) else {}
    item = attributes.get(name) if isinstance(attributes.get(name), dict) else {}
    try:
        return float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _page_id(page: dict[str, Any], page_number: int) -> str:
    return str(page.get("page_id") or page.get("id") or f"page_{page_number:03d}").strip()


def _layer_index(pages: list[Any]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, tuple[int, dict[str, Any]]]]:
    layers: dict[tuple[str, str], list[dict[str, Any]]] = {}
    page_records: dict[str, tuple[int, dict[str, Any]]] = {}
    for page_number, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        page_id = _page_id(page, page_number)
        page_records[page_id] = (page_number, page)
        for layer in page.get("text_layers") or page.get("texts") or []:
            if not isinstance(layer, dict):
                continue
            owner_id = str(layer.get("owner_id") or "").strip()
            if owner_id:
                layers.setdefault((page_id, owner_id), []).append(layer)
    return layers, page_records


def _bbox_inside(inner: Any, outer: tuple[int, int, int, int] | None) -> bool:
    if not isinstance(inner, list) or len(inner) != 4 or outer is None:
        return False
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in inner):
        return False
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    return x1 >= ox1 and y1 >= oy1 and x2 <= ox2 and y2 <= oy2 and x2 > x1 and y2 > y1


def _metric_contract_errors(
    contract: dict[str, Any],
    owner_bounds: tuple[int, int, int, int] | None,
) -> list[str]:
    metrics = contract.get("render_metrics")
    if not isinstance(metrics, dict):
        return ["render_metrics"]
    errors: list[str] = []
    for name in REQUIRED_RENDER_METRICS:
        value = metrics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(name)
    core = contract.get("glyph_core_envelope")
    effect = contract.get("effect_envelope")
    for metric_name, envelope in (
        ("core_pixel_count", core),
        ("effect_pixel_count", effect),
    ):
        if not isinstance(envelope, dict):
            errors.append(f"{metric_name}_envelope")
            continue
        pixel_count = envelope.get("pixel_count")
        if metrics.get(metric_name) != pixel_count:
            errors.append(f"{metric_name}_consistency")
        bbox = envelope.get("bbox_page")
        if isinstance(pixel_count, int) and pixel_count > 0:
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"{metric_name}_bbox")
        elif bbox not in ([], None):
            errors.append(f"{metric_name}_empty_bbox")
    if isinstance(core, dict) and int(core.get("pixel_count") or 0) > 0:
        if not _bbox_inside(core.get("bbox_page"), owner_bounds):
            errors.append("core_containment")
    return sorted(set(errors))


def _owner_bounds(graph: Any, owner: Any) -> tuple[int, int, int, int] | None:
    component_ids = set(owner.component_ids)
    boxes = [component.bbox_page for component in graph.components if component.component_id in component_ids]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _binding_errors(profile: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    expected = {
        "visual_profile_sha256": profile.get("visual_profile_sha256"),
        "profile_component_geometry_sha256": profile.get("component_geometry_sha256"),
        "source_artifact_sha256": profile.get("source_sha256"),
        "source_glyph_mask_sha256": profile.get("glyph_mask_sha256"),
    }
    return [name for name, value in expected.items() if contract.get(name) != value]


def _attribute_results(profile: dict[str, Any], contract: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    decision = profile.get("style_application_decision_v2")
    decision = decision if isinstance(decision, dict) else {}
    expected_applied = decision.get("applied_attributes")
    expected_applied = expected_applied if isinstance(expected_applied, dict) else {}
    expected_abstained = decision.get("abstained_attributes")
    expected_abstained = expected_abstained if isinstance(expected_abstained, dict) else {}
    observed_applied = contract.get("applied_attributes")
    observed_applied = observed_applied if isinstance(observed_applied, dict) else {}
    observed_abstained = contract.get("abstained_attributes")
    observed_abstained = observed_abstained if isinstance(observed_abstained, dict) else {}
    attributes: dict[str, Any] = {}
    catastrophic: list[str] = []
    for name in sorted(set(expected_applied) | set(expected_abstained)):
        confidence = _attribute_confidence(profile, name)
        if name in expected_abstained:
            observed_reason = observed_abstained.get(name)
            matches = name not in observed_applied and observed_reason == expected_abstained[name]
            status = "abstained" if matches else "mismatch"
            expected_value = expected_abstained[name]
            observed_value = observed_reason
        else:
            observed_value = observed_applied.get(name)
            matches = name not in observed_abstained and name in observed_applied and _same_value(
                expected_applied[name], observed_value
            )
            status = "applied" if matches else "mismatch"
            expected_value = expected_applied[name]
        high_confidence_failure = status == "mismatch" and (
            confidence >= HIGH_CONFIDENCE_THRESHOLD or name in CATASTROPHIC_STYLE_ATTRIBUTES
        )
        if status == "mismatch" and name in CATASTROPHIC_STYLE_ATTRIBUTES:
            catastrophic.append(name)
        attributes[name] = {
            "status": status,
            "expected": copy.deepcopy(expected_value),
            "observed": copy.deepcopy(observed_value),
            "confidence": confidence,
            "high_confidence_failure": high_confidence_failure,
        }
        color_distance = delta_e_2000(expected_value, observed_value)
        if color_distance is not None:
            attributes[name]["delta_e_2000"] = color_distance
    return attributes, catastrophic


def audit_style_fidelity(project: dict[str, Any], run_dir: Path, *, mode: str = "shadow") -> dict[str, Any]:
    rollout = str(mode or "shadow").strip().lower()
    if rollout not in STYLE_MODES:
        raise ValueError(f"unsupported style fidelity mode: {mode}")
    pages = project.get("paginas") or project.get("pages") or []
    pages = pages if isinstance(pages, list) else []
    layer_index, page_records = _layer_index(pages)
    owners: list[dict[str, Any]] = []
    blocking: set[str] = set()
    global_findings: list[dict[str, Any]] = []
    eligible: list[tuple[str, Any, tuple[int, int, int, int] | None]] = []
    graphs = project.get("page_owner_graphs")
    graphs = graphs if isinstance(graphs, list) else []
    for graph_payload in graphs:
        try:
            graph = validate_serialized_owner_graph(graph_payload)
        except (TypeError, ValueError) as exc:
            global_findings.append({"code": "invalid_owner_graph", "detail": str(exc)})
            continue
        eligible.extend(
            (graph.page_id, owner, _owner_bounds(graph, owner))
            for owner in graph.owners
            if is_rendered_owner(owner)
        )

    denominator = len(eligible)
    profile_count = contract_count = metrics_count = 0
    category_totals: dict[str, int] = {"all_renderable": denominator}
    category_covered: dict[str, int] = {"all_renderable": 0}
    attribute_metrics: dict[str, dict[str, int]] = {}
    fallback_reason_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    for page_id, owner, owner_bounds in eligible:
        owner_id = owner.owner_id
        semantic_role = str(owner.semantic_role or "unknown")
        category_totals[semantic_role] = category_totals.get(semantic_role, 0) + 1
        category_covered.setdefault(semantic_role, 0)
        page_number, page = page_records.get(page_id, (0, {}))
        source_path = resolve_original_path(Path(run_dir), page, page_number) if page else None
        findings: list[dict[str, Any]] = []
        candidates = layer_index.get((page_id, owner_id), [])
        if len(candidates) != 1:
            findings.append({"code": "missing_owner_layer" if not candidates else "duplicate_owner_layer"})
            layer: dict[str, Any] = {}
        else:
            layer = candidates[0]
        raw_profile = layer.get("visual_profile_v2")
        profile: dict[str, Any] | None = None
        if not isinstance(raw_profile, dict):
            findings.append({"code": "missing_visual_profile"})
        else:
            try:
                profile = validate_owner_visual_profile(
                    raw_profile,
                    expected_owner_id=owner_id,
                    expected_sha256=layer.get("visual_profile_sha256"),
                )
                profile_count += 1
            except (TypeError, ValueError) as exc:
                findings.append({"code": "invalid_visual_profile", "detail": str(exc)})
        raw_contract = layer.get("style_v2_raster_contract")
        contract: dict[str, Any] | None = None
        if not isinstance(raw_contract, dict):
            findings.append({"code": "missing_raster_contract"})
        else:
            try:
                contract = validate_owner_style_raster_contract(
                    raw_contract,
                    expected_owner_id=owner_id,
                    expected_page_id=page_id,
                )
                contract_count += 1
            except (TypeError, ValueError) as exc:
                findings.append({"code": "invalid_raster_contract", "detail": str(exc)})
        attributes: dict[str, Any] = {}
        catastrophic: list[str] = []
        if profile is not None and contract is not None:
            backend = str(contract.get("backend") or "unknown")
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            for binding in _binding_errors(profile, contract):
                findings.append({"code": "raster_binding_mismatch", "field": binding})
            attributes, catastrophic = _attribute_results(profile, contract)
            for name, result in attributes.items():
                status_counts = attribute_metrics.setdefault(name, {})
                status = str(result["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
                if result["status"] == "mismatch":
                    findings.append({"code": "style_attribute_mismatch", "attribute": name})
                elif result["status"] == "abstained":
                    reason = str(result["expected"])
                    fallback_reason_counts[reason] = fallback_reason_counts.get(reason, 0) + 1
            metric_errors = _metric_contract_errors(contract, owner_bounds)
            if not metric_errors:
                metrics_count += 1
            else:
                findings.extend(
                    {"code": "invalid_render_metric", "metric": name}
                    for name in metric_errors
                )
        owner_blocking = bool(findings)
        if owner_blocking:
            blocking.add(owner_id)
        else:
            category_covered["all_renderable"] += 1
            category_covered[semantic_role] += 1
        owners.append(
            {
                "page": page_number,
                "page_id": page_id,
                "owner_id": owner_id,
                "semantic_role": semantic_role,
                "source_path": str(source_path) if source_path else None,
                "source_sha256": profile.get("source_sha256") if profile else None,
                "glyph_mask_sha256": profile.get("glyph_mask_sha256") if profile else None,
                "visual_profile_sha256": profile.get("visual_profile_sha256") if profile else None,
                "profile_status": profile.get("status") if profile else "missing",
                "attributes": attributes,
                "fields": copy.deepcopy(attributes),
                "catastrophic_mismatches": catastrophic,
                "findings": findings,
            }
        )

    if denominator == 0:
        global_findings.append({"code": "empty_style_fidelity_denominator"})
    required_categories = required_style_categories(project)
    categories: dict[str, Any] = {}
    for name in sorted(set(category_totals) | set(required_categories)):
        total = category_totals.get(name, 0)
        covered = category_covered.get(name, 0)
        categories[name] = {
            "eligible": total,
            "covered": covered,
            "coverage": covered / total if total else 0.0,
            "required": name in required_categories,
        }
        if name in required_categories and total == 0:
            global_findings.append({"code": "required_category_empty", "category": name})
    coverage = {
        "profile": profile_count / denominator if denominator else 0.0,
        "contract": contract_count / denominator if denominator else 0.0,
        "metrics": metrics_count / denominator if denominator else 0.0,
    }
    if denominator and any(value < 1.0 for value in coverage.values()):
        global_findings.append({"code": "incomplete_style_fidelity_coverage"})
    would_block = bool(blocking or global_findings)
    status = "BLOCK" if rollout == "enforce" and would_block else "PASS"
    return {
        "schema_version": 2,
        "mode": rollout,
        "owners": owners,
        "coverage": coverage,
        "categories": categories,
        "attribute_metrics": attribute_metrics,
        "fallback_reason_counts": fallback_reason_counts,
        "backend_counts": backend_counts,
        "findings": global_findings,
        "summary": {
            "owner_count": len(owners),
            "eligible_owner_count": denominator,
            "blocking_owner_count": len(blocking),
            "catastrophic_mismatch_count": sum(
                len(owner["catastrophic_mismatches"]) for owner in owners
            ),
        },
        "gate": {
            "status": status,
            "would_block": would_block,
            "blocking_owner_ids": sorted(blocking),
        },
    }


def merge_style_and_functional_gates(functional_gate: dict[str, Any], style_gate: dict[str, Any]) -> dict[str, Any]:
    functional = copy.deepcopy(functional_gate or {"status": "PASS"})
    style = copy.deepcopy(style_gate or {"status": "PASS"})
    blocked = str(functional.get("status") or "PASS").upper() == "BLOCK" or str(style.get("status") or "PASS").upper() == "BLOCK"
    return {"status": "BLOCK" if blocked else "PASS", "functional_gate": functional, "style_gate": style}
