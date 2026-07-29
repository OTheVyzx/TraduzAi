"""Derived, write-only debug artifacts for page-global text ownership."""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Any, Iterable, Mapping


def composite_owner_trace(page_id: str, owner_id: str | None, event: str) -> str:
    return ":".join(
        (
            str(page_id or "run"),
            str(owner_id or "unowned"),
            str(event or "snapshot"),
        )
    )


def _snapshot(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    fields = getattr(value, "__dict__", None)
    return dict(fields) if isinstance(fields, dict) else {}


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _safe_owner_segment(owner_id: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner_id).strip("._")
    return segment or "unowned"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _base_row(
    *,
    page_id: str,
    owner_id: str | None,
    event: str,
    coordinate_space: str = "page",
    hashes: Mapping[str, Any] | None = None,
    offenders: Iterable[Any] = (),
) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "owner_id": owner_id,
        "trace_id": composite_owner_trace(page_id, owner_id, event),
        "coordinate_space": coordinate_space,
        "hashes": {
            str(key): str(value)
            for key, value in dict(hashes or {}).items()
            if value not in {None, ""}
        },
        "offenders": [str(value) for value in offenders],
    }


def _derived_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    authoritative_fields = {
        "schema_version",
        "run_id",
        "stage",
        "page_id",
        "owner_id",
        "trace_id",
        "coordinate_space",
        "hashes",
        "offenders",
    }
    return {str(key): item for key, item in value.items() if key not in authoritative_fields}


def validate_qa_rows(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in rows:
        trace_id = str(row.get("trace_id") or "").strip()
        if trace_id.count(":") < 2:
            failures.append("qa_row_trace_missing")
        offenders = row.get("offenders")
        if not isinstance(offenders, list) or not offenders:
            failures.append("qa_row_offenders_missing")
    return list(dict.fromkeys(failures))


def validate_gate_integrity(
    *,
    summary: Mapping[str, Any],
    gate: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    rows = list(rows)
    failures: list[str] = []
    critical_count = sum(str(row.get("severity") or "") == "critical" for row in rows)
    blocking_count = sum(
        str(row.get("severity") or "") == "critical" or bool(row.get("blocks_export"))
        for row in rows
    )
    for source_name, source in (("summary", summary), ("gate", gate)):
        if int(source.get("critical_issue_count", 0) or 0) != critical_count:
            failures.append("critical_issue_count_mismatch")
        if int(source.get("blocking_issue_count", 0) or 0) != blocking_count:
            failures.append("blocking_issue_count_mismatch")
    failures.extend(validate_qa_rows(rows))
    return list(dict.fromkeys(failures))


class OwnerArtifactPublisher:
    """Serialize immutable owner snapshots; never read them back into control flow."""

    def __init__(self, recorder: Any) -> None:
        self.recorder = recorder

    def publish(
        self,
        *,
        graphs: Mapping[str, Any] | None = None,
        executions: Iterable[Any] | None = None,
        compositions: Mapping[str, Any] | None = None,
        final_pixel_reports: Iterable[Mapping[str, Any]] | None = None,
        export_gate: Mapping[str, Any] | None = None,
    ) -> dict[str, int]:
        graph_snapshots = {
            str(page_id): _snapshot(graph)
            for page_id, graph in dict(graphs or {}).items()
        }
        components: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        render_plan: list[dict[str, Any]] = []
        all_violations: list[dict[str, Any]] = []

        for page_id, graph in sorted(graph_snapshots.items()):
            dispositions = {
                str(item.get("component_id")): item
                for item in graph.get("component_dispositions") or []
                if isinstance(item, dict)
            }
            owners = [item for item in graph.get("owners") or [] if isinstance(item, dict)]
            owner_by_component = {
                str(component_id): str(owner.get("owner_id"))
                for owner in owners
                for component_id in owner.get("component_ids") or []
            }
            owner_by_observation = {
                str(observation_id): str(owner.get("owner_id"))
                for owner in owners
                for observation_id in owner.get("observation_ids") or []
            }
            for component in graph.get("components") or []:
                if not isinstance(component, dict):
                    continue
                component_id = str(component.get("component_id") or "")
                owner_id = owner_by_component.get(component_id)
                disposition = dispositions.get(component_id) or {}
                components.append(
                    {
                        **_derived_payload(component),
                        **_base_row(
                            page_id=page_id,
                            owner_id=owner_id,
                            event=f"component:{component_id}",
                            hashes={"component_sha256": _canonical_sha256(component)},
                            offenders=([component_id] if component_id else []),
                        ),
                        "disposition": disposition.get("decision"),
                        "disposition_reason": disposition.get("reason"),
                    }
                )
            for observation in graph.get("observations") or []:
                if not isinstance(observation, dict):
                    continue
                observation_id = str(observation.get("observation_id") or "")
                owner_id = owner_by_observation.get(observation_id)
                observations.append(
                    {
                        **_derived_payload(observation),
                        **_base_row(
                            page_id=page_id,
                            owner_id=owner_id,
                            event=f"observation:{observation_id}",
                            hashes={"observation_sha256": _canonical_sha256(observation)},
                            offenders=observation.get("component_ids") or [],
                        ),
                    }
                )
            for owner in owners:
                owner_id = str(owner.get("owner_id") or "")
                if (
                    str(owner.get("route_action") or "") == "review_required"
                    or str(owner.get("state") or "") == "review_required"
                ):
                    continue
                render_plan.append(
                    {
                        **_base_row(
                            page_id=page_id,
                            owner_id=owner_id,
                            event="render",
                            hashes={"owner_snapshot_sha256": _canonical_sha256(owner)},
                            offenders=owner.get("component_ids") or [],
                        ),
                        "route_action": owner.get("route_action"),
                        "state": owner.get("state"),
                        "component_ids": list(owner.get("component_ids") or []),
                        "source_payload": owner.get("source_payload"),
                        "translated_payload": owner.get("translated_payload"),
                    }
                )
            all_violations.extend(
                dict(item) for item in graph.get("violations") or [] if isinstance(item, dict)
            )

        if graphs is not None:
            self.recorder.write_jsonl_replace(
                "02_strip_detect/page_owner_components.jsonl", components
            )
            self.recorder.write_jsonl_replace(
                "03_ocr/page_owner_observations.jsonl", observations
            )
            self.recorder.write_json(
                "04_text_normalization_router/page_owner_graph.json",
                {
                    "pages": [graph_snapshots[key] for key in sorted(graph_snapshots)],
                    "rows": [
                        {
                            **_base_row(
                                page_id=page_id,
                                owner_id=None,
                                event="graph",
                                hashes={
                                    "graph_sha256": _canonical_sha256(graph_snapshots[page_id])
                                },
                                offenders=[
                                    offender
                                    for violation in graph_snapshots[page_id].get("violations") or []
                                    for offender in violation.get("offenders") or []
                                ],
                            ),
                            "graph": graph_snapshots[page_id],
                        }
                        for page_id in sorted(graph_snapshots)
                    ],
                },
            )
            self.recorder.write_jsonl_replace(
                "09_typeset/owner_render_plan.jsonl", render_plan
            )

        execution_count = 0
        for execution in executions or ():
            page_id = str(_field(execution, "page_id") or "")
            owner_id = str(_field(execution, "owner_id") or "")
            action_mask = _field(execution, "action_mask")
            if action_mask is None:
                mutation = _field(execution, "mutation")
                action_mask = _field(mutation, "action_mask")
            if page_id and owner_id and action_mask is not None:
                self.recorder.write_image(
                    f"06_mask_segmentation/owner_masks/{_safe_owner_segment(owner_id)}/action_mask.png",
                    action_mask,
                    color_space="GRAY",
                )
                execution_count += 1

        composition_rows: list[dict[str, Any]] = []
        for page_id, value in sorted(dict(compositions or {}).items()):
            payload = _snapshot(value)
            owner_ids = list(payload.get("owner_ids") or [])
            conflicts = list(payload.get("conflicts") or [])
            composition_rows.append(
                {
                    **_base_row(
                        page_id=str(page_id),
                        owner_id=(str(owner_ids[0]) if len(owner_ids) == 1 else None),
                        event="composition",
                        coordinate_space=str(payload.get("coordinate_space") or "page"),
                        hashes={
                            "composition_sha256": payload.get("sha256")
                            or _canonical_sha256(payload)
                        },
                        offenders=owner_ids or conflicts,
                    ),
                    "write_counts": dict(payload.get("write_counts") or {}),
                    "owner_ids": owner_ids,
                    "conflicts": conflicts,
                }
            )
        if compositions is not None:
            self.recorder.write_jsonl_replace(
                "10_copyback_reassemble/owner_composition.jsonl", composition_rows
            )

        final_ocr_rows: list[dict[str, Any]] = []
        pixel_rows: list[dict[str, Any]] = []
        final_reports = [dict(report) for report in final_pixel_reports or ()]
        for report in final_reports:
            page_id = str(report.get("page_id") or "")
            persisted_sha = report.get("persisted_sha256")
            records = report.get("ocr_records") or []
            for index, record in enumerate(records, start=1):
                if not isinstance(record, dict):
                    continue
                owner_id = str(record.get("owner_id") or "").strip() or None
                final_ocr_rows.append(
                    {
                        **_derived_payload(record),
                        **_base_row(
                            page_id=page_id,
                            owner_id=owner_id,
                            event=f"final_ocr:{index}",
                            hashes={
                                "persisted_sha256": persisted_sha
                                or _canonical_sha256(record)
                            },
                            offenders=([record.get("text")] if record.get("text") else []),
                        ),
                    }
                )
            issues = [item for item in report.get("issues") or [] if isinstance(item, dict)]
            pixel_rows.append(
                {
                    **_base_row(
                        page_id=page_id,
                        owner_id=None,
                        event="pixel_checks",
                        hashes={
                            "persisted_sha256": persisted_sha
                            or _canonical_sha256(report)
                        },
                        offenders=[
                            offender
                            for issue in issues
                            for offender in issue.get("offenders") or []
                        ],
                    ),
                    "contracts": dict(report.get("contracts") or {}),
                    "issue_count": len(issues),
                }
            )
        if final_pixel_reports is not None:
            self.recorder.write_jsonl_replace(
                "11_qa_export_gate/final_pixel_ocr.jsonl", final_ocr_rows
            )
            self.recorder.write_jsonl_replace(
                "11_qa_export_gate/owner_pixel_checks.jsonl", pixel_rows
            )

        gate = dict(export_gate or {})
        gate_issues = [item for item in gate.get("issues") or [] if isinstance(item, dict)]
        if graphs is not None or export_gate is not None:
            self.recorder.write_json(
                "11_qa_export_gate/owner_invariant_report.json",
                {
                    "graph_violation_count": len(all_violations),
                    "graph_violations": all_violations,
                    "gate_status": gate.get("status"),
                    "gate_issue_count": len(gate_issues),
                    "offenders": [
                        offender
                        for issue in [*all_violations, *gate_issues]
                        for offender in issue.get("offenders") or []
                    ],
                },
            )
        return {
            "graph_count": len(graph_snapshots),
            "component_count": len(components),
            "observation_count": len(observations),
            "render_plan_count": len(render_plan),
            "execution_mask_count": execution_count,
            "composition_count": len(composition_rows),
            "final_ocr_count": len(final_ocr_rows),
            "pixel_check_count": len(pixel_rows),
        }
