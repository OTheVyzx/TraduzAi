"""Hash-bound handoff from a consumed Vision revision to renderer artifacts.

This adapter performs no discovery and imports no OCR provider.  It only
validates already-materialized contracts and records that the renderer made
zero provider calls while deriving its plan, recipe, and raster candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np

from integration_v1.contracts import AnalysisRecord, adapt_contract_payload
from ownership.hash_contract import canonical_json_sha256
from typesetter.preference_contract import PreferenceCandidate
from typesetter.raster_safety import assess_raster_safety, raster_mask_sha256, validate_raster_safety_evidence
from typesetter.recipe_contract import RendererRecipe


LINEAGE_SCHEMA = "traduzai.renderer-analysis-lineage.v1"


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))


def _sha256(value: Any, field: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return candidate


def _json_artifact(value: bytes, expected_sha256: str, field: str) -> Any:
    if not isinstance(value, bytes) or sha256(value).hexdigest() != _sha256(expected_sha256, field):
        raise ValueError(f"{field} bytes do not match AnalysisRecord")
    try:
        return json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must be UTF-8 JSON") from exc


@dataclass(frozen=True)
class RendererAnalysisLineageReceipt:
    schema: str
    owner_id: str
    analysis_record_sha256: str
    layout_plan_sha256: str
    recipe_sha256: str
    raster_candidate_id: str
    output_sha256: str
    raster_safety_evidence_sha256: str
    raster_safety_status: str
    owner_route_state: str
    owner_binding_sha256: str
    provider_trace_sha256: str
    state: str
    provider_calls_ocr: int
    provider_calls_analysis_discovery: int
    publication_allowed: bool
    receipt_sha256: str

    @property
    def provider_calls(self) -> dict[str, int]:
        return {
            "ocr": self.provider_calls_ocr,
            "analysis_discovery": self.provider_calls_analysis_discovery,
        }

    @classmethod
    def build(
        cls,
        *,
        analysis_record: Mapping[str, Any],
        owner_binding: Mapping[str, Any],
        provider_trace: Mapping[str, Any],
        logical_units_artifact_bytes: bytes,
        physical_subblocks_artifact_bytes: bytes,
        owner_route_evidence_bytes: bytes,
        renderer_event_log_bytes: bytes,
        layout_plan: Mapping[str, Any],
        recipe: Mapping[str, Any],
        raster_candidate: Mapping[str, Any],
        raster_rgba: np.ndarray,
        authorized_body_mask: np.ndarray,
        protected_art_mask: np.ndarray,
    ) -> "RendererAnalysisLineageReceipt":
        analysis_payload = _copy(analysis_record)
        claimed_analysis_sha256 = _sha256(
            analysis_payload.pop("analysis_record_sha256", None),
            "analysis_record_sha256",
        )
        analysis = AnalysisRecord.build(analysis_payload)
        if analysis.analysis_record_sha256 != claimed_analysis_sha256:
            raise ValueError("analysis record SHA-256 mismatch")
        if analysis.payload["status"] != "complete":
            raise ValueError("renderer requires a complete analysis revision")

        binding = _copy(owner_binding)
        binding_sha256 = _sha256(binding.pop("binding_sha256", None), "owner_binding.binding_sha256")
        if canonical_json_sha256(binding) != binding_sha256:
            raise ValueError("owner binding SHA-256 mismatch")
        required_binding = {
            "owner_id", "analysis_record_sha256", "logical_unit_id",
            "physical_subblock_ids", "route_state", "route_evidence_sha256", "producer",
        }
        if set(binding) != required_binding:
            raise ValueError("owner binding schema is incomplete")
        if binding["analysis_record_sha256"] != claimed_analysis_sha256:
            raise ValueError("owner binding analysis dependency mismatch")
        if binding["route_state"] not in {"ready_for_layout", "review_required"}:
            raise ValueError("owner binding route state is unsupported")
        if not binding["logical_unit_id"] or not binding["physical_subblock_ids"]:
            raise ValueError("owner binding requires logical and physical destinations")
        producer = binding["producer"]
        if not isinstance(producer, Mapping) or set(producer) != {"runtime_id", "runtime_sha256"}:
            raise ValueError("owner binding producer is incomplete")
        _sha256(producer["runtime_sha256"], "owner_binding.producer.runtime_sha256")

        logical_ref = analysis.payload["logical_units"]
        physical_ref = analysis.payload["physical_subblocks"]
        logical_artifact = _json_artifact(
            logical_units_artifact_bytes, logical_ref["sha256"], "logical_units_artifact",
        )
        physical_artifact = _json_artifact(
            physical_subblocks_artifact_bytes, physical_ref["sha256"], "physical_subblocks_artifact",
        )
        logical_rows = logical_artifact.get("logical_units") if isinstance(logical_artifact, Mapping) else None
        physical_rows = physical_artifact.get("physical_subblocks") if isinstance(physical_artifact, Mapping) else None
        if not isinstance(logical_rows, list) or not isinstance(physical_rows, list):
            raise ValueError("analysis structure artifacts are incomplete")
        if not any(row.get("logical_unit_id") == binding["logical_unit_id"] for row in logical_rows if isinstance(row, Mapping)):
            raise ValueError("owner binding logical unit is absent from AnalysisRecord artifact")
        expected_physical_ids = set(binding["physical_subblock_ids"])
        observed_physical_ids = {
            row.get("physical_subblock_id")
            for row in physical_rows
            if isinstance(row, Mapping) and row.get("logical_unit_id") == binding["logical_unit_id"]
        }
        if expected_physical_ids != observed_physical_ids:
            raise ValueError("owner binding physical destinations differ from AnalysisRecord artifact")

        route_sha256 = _sha256(binding["route_evidence_sha256"], "owner_binding.route_evidence_sha256")
        route = _json_artifact(owner_route_evidence_bytes, route_sha256, "owner_route_evidence")
        route_units = route.get("units") if isinstance(route, Mapping) else None
        route_rows = [row for row in (route_units or []) if isinstance(row, Mapping) and row.get("owner_id") == binding["owner_id"]]
        if len(route_rows) != 1:
            raise ValueError("owner route evidence must contain exactly one owner")
        observed_route_state = "ready_for_layout" if route_rows[0].get("status") == "translation_ready" else "review_required"
        if observed_route_state != binding["route_state"]:
            raise ValueError("owner route state differs from route evidence")

        trace = _copy(provider_trace)
        trace_sha256 = _sha256(trace.pop("trace_sha256", None), "provider_trace.trace_sha256")
        if canonical_json_sha256(trace) != trace_sha256:
            raise ValueError("renderer provider trace SHA-256 mismatch")
        if set(trace) != {"schema", "analysis_record_sha256", "scope", "provider_calls", "event_log_sha256", "producer"}:
            raise ValueError("renderer provider trace schema is incomplete")
        if trace["schema"] != "traduzai.renderer-provider-trace.v1" or trace["scope"] != "renderer":
            raise ValueError("renderer provider trace identity is unsupported")
        if trace["analysis_record_sha256"] != claimed_analysis_sha256:
            raise ValueError("renderer provider trace analysis dependency mismatch")
        trace_producer = trace["producer"]
        if not isinstance(trace_producer, Mapping) or set(trace_producer) != {"runtime_id", "runtime_sha256"}:
            raise ValueError("renderer provider trace producer is incomplete")
        _sha256(trace_producer["runtime_sha256"], "provider_trace.producer.runtime_sha256")
        event_log_sha256 = _sha256(trace["event_log_sha256"], "provider_trace.event_log_sha256")
        event_log = _json_artifact(renderer_event_log_bytes, event_log_sha256, "renderer_event_log")
        if not isinstance(event_log, Mapping) or event_log.get("schema") != "traduzai.renderer-provider-events.v1":
            raise ValueError("renderer provider event log schema is unsupported")
        if event_log.get("analysis_record_sha256") != claimed_analysis_sha256:
            raise ValueError("renderer provider event log analysis dependency mismatch")
        events = event_log.get("events")
        if not isinstance(events, list) or [event.get("kind") for event in events if isinstance(event, Mapping)] != ["stage_started", "stage_completed"]:
            raise ValueError("renderer provider event log must delimit one complete stage")
        if any(event.get("stage") != "renderer" for event in events if isinstance(event, Mapping)):
            raise ValueError("renderer provider event log contains another stage")
        provider_calls = {
            "ocr": sum(1 for event in events if event.get("kind") == "provider_call" and event.get("provider") == "ocr"),
            "analysis_discovery": sum(1 for event in events if event.get("kind") == "provider_call" and event.get("provider") == "analysis_discovery"),
        }
        if trace["provider_calls"] != provider_calls:
            raise ValueError("renderer provider trace counters differ from event log")
        if provider_calls != {"ocr": 0, "analysis_discovery": 0}:
            raise ValueError("renderer provider trace must prove zero OCR and discovery calls")

        plan = adapt_contract_payload("LayoutPlan", layout_plan)
        plan_sha256 = _sha256(plan["plan_sha256"], "layout_plan.plan_sha256")
        plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
        if canonical_json_sha256(plan_body) != plan_sha256:
            raise ValueError("layout plan SHA-256 mismatch")
        plan_dependencies = plan.get("dependency_hashes")
        if not isinstance(plan_dependencies, Mapping) or plan_dependencies.get("analysis_record") != claimed_analysis_sha256:
            raise ValueError("layout plan analysis dependency mismatch")

        validated_recipe = RendererRecipe.from_dict(recipe)
        rgba = np.asarray(raster_rgba)
        if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError("renderer lineage requires an uint8 RGBA raster")
        expected_shape = (
            validated_recipe.bbox[3] - validated_recipe.bbox[1],
            validated_recipe.bbox[2] - validated_recipe.bbox[0],
            4,
        )
        if rgba.shape != expected_shape:
            raise ValueError("renderer RGBA shape differs from recipe bbox")
        validated_recipe.verify_output(np.ascontiguousarray(rgba).tobytes())
        candidate = PreferenceCandidate.from_dict(raster_candidate)
        owners = {plan["owner_id"], validated_recipe.owner_id, candidate.owner_id}
        if len(owners) != 1:
            raise ValueError("renderer lineage owner mismatch")
        owner_id = owners.pop()
        if binding["owner_id"] != owner_id:
            raise ValueError("owner binding identity mismatch")
        source_sha256 = analysis.payload["source_sha256"]
        if validated_recipe.source_sha256 != source_sha256 or candidate.source_sha256 != source_sha256:
            raise ValueError("renderer lineage source mismatch")
        if validated_recipe.dependency_hashes.get("analysis_record") != claimed_analysis_sha256:
            raise ValueError("recipe analysis dependency mismatch")
        if validated_recipe.dependency_hashes.get("layout_plan") != plan_sha256:
            raise ValueError("recipe layout dependency mismatch")
        if candidate.metrics.get("analysis_record_sha256") != claimed_analysis_sha256:
            raise ValueError("candidate analysis dependency mismatch")
        if candidate.layout_plan_sha256 != plan_sha256:
            raise ValueError("candidate layout dependency mismatch")
        if candidate.recipe_sha256 != validated_recipe.recipe_sha256:
            raise ValueError("candidate recipe dependency mismatch")
        if candidate.output_sha256 != validated_recipe.output_sha256:
            raise ValueError("candidate output dependency mismatch")
        if candidate.target_text != validated_recipe.target_text:
            raise ValueError("candidate target differs from renderer recipe")
        if tuple(plan["lines"]) != validated_recipe.line_plan.lines:
            raise ValueError("layout plan lines differ from renderer recipe")
        metrics = plan.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("layout plan metrics are missing")
        if metrics.get("font_size_px") != validated_recipe.font_size_px:
            raise ValueError("layout plan font size differs from renderer recipe")
        if metrics.get("line_advance_px") != validated_recipe.line_advance_px:
            raise ValueError("layout plan line advance differs from renderer recipe")
        usable_body_sha256 = canonical_json_sha256(plan["usable_body"])
        if validated_recipe.geometry.get("usable_body_sha256") != usable_body_sha256:
            raise ValueError("layout usable body differs from renderer recipe")
        expected_style_sha256 = canonical_json_sha256({
            "font": validated_recipe.font,
            "effects": validated_recipe.effects,
        })
        if candidate.style_sha256 != expected_style_sha256:
            raise ValueError("candidate style differs from renderer recipe")
        safety = validate_raster_safety_evidence(candidate.metrics.get("raster_safety"))
        if safety["ink_pixel_count"] == 0:
            raise ValueError("candidate hard safety cannot use an empty raster")
        if list(safety["bbox"]) != list(validated_recipe.bbox):
            raise ValueError("candidate raster safety bbox differs from renderer recipe")
        derived_alpha_sha256 = raster_mask_sha256(rgba[:, :, 3])
        if derived_alpha_sha256 != safety["alpha_sha256"]:
            raise ValueError("candidate raster safety alpha differs from RGBA output")
        if validated_recipe.geometry.get("alpha_sha256") != derived_alpha_sha256:
            raise ValueError("candidate raster safety alpha differs from renderer recipe")
        derived_safety = assess_raster_safety(
            alpha=rgba[:, :, 3],
            bbox=validated_recipe.bbox,
            authorized_body_mask=authorized_body_mask,
            protected_art_mask=protected_art_mask,
        )
        if safety != derived_safety:
            raise ValueError("candidate raster safety evidence differs from authoritative masks")
        state = (
            "ready_for_review"
            if binding["route_state"] == "ready_for_layout" and candidate.hard_safety_passed
            else "review_required"
        )
        body = {
            "schema": LINEAGE_SCHEMA,
            "owner_id": owner_id,
            "analysis_record_sha256": claimed_analysis_sha256,
            "layout_plan_sha256": plan_sha256,
            "recipe_sha256": validated_recipe.recipe_sha256,
            "raster_candidate_id": candidate.candidate_id,
            "output_sha256": candidate.output_sha256,
            "raster_safety_evidence_sha256": safety["evidence_sha256"],
            "raster_safety_status": safety["status"],
            "owner_route_state": binding["route_state"],
            "owner_binding_sha256": binding_sha256,
            "provider_trace_sha256": trace_sha256,
            "state": state,
            "provider_calls": provider_calls,
            "publication_allowed": False,
        }
        return cls(
            schema=body["schema"],
            owner_id=body["owner_id"],
            analysis_record_sha256=body["analysis_record_sha256"],
            layout_plan_sha256=body["layout_plan_sha256"],
            recipe_sha256=body["recipe_sha256"],
            raster_candidate_id=body["raster_candidate_id"],
            output_sha256=body["output_sha256"],
            raster_safety_evidence_sha256=body["raster_safety_evidence_sha256"],
            raster_safety_status=body["raster_safety_status"],
            owner_route_state=body["owner_route_state"],
            owner_binding_sha256=body["owner_binding_sha256"],
            provider_trace_sha256=body["provider_trace_sha256"],
            state=body["state"],
            provider_calls_ocr=provider_calls["ocr"],
            provider_calls_analysis_discovery=provider_calls["analysis_discovery"],
            publication_allowed=False,
            receipt_sha256=canonical_json_sha256(body),
        )

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "owner_id": self.owner_id,
            "analysis_record_sha256": self.analysis_record_sha256,
            "layout_plan_sha256": self.layout_plan_sha256,
            "recipe_sha256": self.recipe_sha256,
            "raster_candidate_id": self.raster_candidate_id,
            "output_sha256": self.output_sha256,
            "raster_safety_evidence_sha256": self.raster_safety_evidence_sha256,
            "raster_safety_status": self.raster_safety_status,
            "owner_route_state": self.owner_route_state,
            "owner_binding_sha256": self.owner_binding_sha256,
            "provider_trace_sha256": self.provider_trace_sha256,
            "state": self.state,
            "provider_calls": self.provider_calls,
            "publication_allowed": self.publication_allowed,
            "receipt_sha256": self.receipt_sha256,
        }
        return _copy(body)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RendererAnalysisLineageReceipt":
        payload = _copy(value)
        required = {
            "schema", "owner_id", "analysis_record_sha256", "layout_plan_sha256",
            "recipe_sha256", "raster_candidate_id", "output_sha256",
            "raster_safety_evidence_sha256", "raster_safety_status",
            "owner_route_state", "owner_binding_sha256", "provider_trace_sha256",
            "state", "provider_calls", "publication_allowed", "receipt_sha256",
        }
        if set(payload) != required or payload["schema"] != LINEAGE_SCHEMA:
            raise ValueError("renderer lineage receipt schema is incomplete")
        receipt_sha256 = _sha256(payload.pop("receipt_sha256"), "receipt_sha256")
        if canonical_json_sha256(payload) != receipt_sha256:
            raise ValueError("renderer lineage receipt SHA-256 mismatch")
        for field in (
            "analysis_record_sha256", "layout_plan_sha256", "recipe_sha256",
            "output_sha256", "raster_safety_evidence_sha256",
            "owner_binding_sha256", "provider_trace_sha256",
        ):
            _sha256(payload[field], field)
        if payload["provider_calls"] != {"ocr": 0, "analysis_discovery": 0}:
            raise ValueError("renderer lineage receipt provider calls are invalid")
        if payload["publication_allowed"] is not False:
            raise ValueError("renderer lineage receipt cannot authorize publication")
        if payload["owner_route_state"] not in {"ready_for_layout", "review_required"}:
            raise ValueError("renderer lineage receipt owner route state is invalid")
        if payload["raster_safety_status"] not in {"pass", "review_required"}:
            raise ValueError("renderer lineage receipt raster safety status is invalid")
        if payload["state"] not in {"ready_for_review", "review_required"}:
            raise ValueError("renderer lineage receipt state is invalid")
        expected_state = (
            "ready_for_review"
            if payload["owner_route_state"] == "ready_for_layout"
            and payload["raster_safety_status"] == "pass"
            else "review_required"
        )
        if payload["state"] != expected_state:
            raise ValueError("renderer lineage receipt state is inconsistent")
        calls = payload["provider_calls"]
        return cls(
            schema=payload["schema"],
            owner_id=payload["owner_id"],
            analysis_record_sha256=payload["analysis_record_sha256"],
            layout_plan_sha256=payload["layout_plan_sha256"],
            recipe_sha256=payload["recipe_sha256"],
            raster_candidate_id=payload["raster_candidate_id"],
            output_sha256=payload["output_sha256"],
            raster_safety_evidence_sha256=payload["raster_safety_evidence_sha256"],
            raster_safety_status=payload["raster_safety_status"],
            owner_route_state=payload["owner_route_state"],
            owner_binding_sha256=payload["owner_binding_sha256"],
            provider_trace_sha256=payload["provider_trace_sha256"],
            state=payload["state"],
            provider_calls_ocr=calls["ocr"],
            provider_calls_analysis_discovery=calls["analysis_discovery"],
            publication_allowed=False,
            receipt_sha256=receipt_sha256,
        )
