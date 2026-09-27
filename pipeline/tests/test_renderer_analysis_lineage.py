from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
import pytest

from integration_v1.contracts import AnalysisRecord, contract_snapshot
from ownership.hash_contract import canonical_json_sha256
from typesetter.preference_contract import PreferenceCandidate
from typesetter.raster_safety import assess_raster_safety
from typesetter.recipe_contract import ExactLinePlan, RendererRecipe


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class Chain:
    analysis: dict
    owner_binding: dict
    provider_trace: dict
    logical_bytes: bytes
    physical_bytes: bytes
    route_bytes: bytes
    event_bytes: bytes
    plan: dict
    recipe: dict
    candidate: dict
    rgba: np.ndarray
    authorized: np.ndarray
    protected: np.ndarray


def _chain(*, safe: bool = True, owner_ready: bool = True) -> Chain:
    logical_bytes = _json_bytes({"logical_units": [{"logical_unit_id": "logical-unit-001"}]})
    physical_bytes = _json_bytes({"physical_subblocks": [{
        "physical_subblock_id": "block-001", "logical_unit_id": "logical-unit-001",
    }]})
    base = contract_snapshot()["examples"]["AnalysisRecord"]
    base["logical_units"]["sha256"] = sha256(logical_bytes).hexdigest()
    base["physical_subblocks"]["sha256"] = sha256(physical_bytes).hexdigest()
    analysis = AnalysisRecord.build(base)
    analysis_payload = analysis.to_dict()

    plan_body = {
        "owner_id": "owner-001",
        "lines": ["TEXTO", "TRADUZIDO"],
        "metrics": {"font_size_px": 30, "line_advance_px": 33},
        "usable_body": {"bbox": [100, 200, 500, 600], "coordinate_space": "logical_page"},
        "dependency_hashes": {"analysis_record": analysis.analysis_record_sha256},
    }
    plan = plan_body | {"plan_sha256": canonical_json_sha256(plan_body)}
    rgba = np.zeros((400, 400, 4), dtype=np.uint8)
    rgba[:, :, :3] = 17
    rgba[:, :, 3] = 255
    authorized = np.ones((600, 600), dtype=np.uint8)
    if not safe:
        authorized[200:600, 499] = 0
    protected = np.zeros((600, 600), dtype=np.uint8)
    raster_safety = assess_raster_safety(
        alpha=rgba[:, :, 3], bbox=[100, 200, 500, 600],
        authorized_body_mask=authorized,
        protected_art_mask=protected,
    )
    font = {"family": "Comic Neue", "relative_path": "fonts/ComicNeue-Bold.ttf", "sha256": "c" * 64}
    effects = {"fill": "#000000"}
    target = "TEXTO TRADUZIDO"
    recipe = RendererRecipe.build(
        owner_id="owner-001",
        source_sha256=analysis.payload["source_sha256"],
        output_sha256=sha256(np.ascontiguousarray(rgba).tobytes()).hexdigest(),
        target_text=target,
        font=font,
        rasterizer={"runtime_id": "python-textpath-v1", "runtime_sha256": "d" * 64, "config_sha256": "e" * 64},
        line_plan=ExactLinePlan.build(target_text=target, lines=["TEXTO", "TRADUZIDO"], separators=[" "]),
        bbox=[100, 200, 500, 600],
        font_size_px=30,
        line_advance_px=33,
        effects=effects,
        anchors={"mode": "source_center"},
        geometry={
            "coordinate_space": "logical_page",
            "usable_body_sha256": canonical_json_sha256(plan["usable_body"]),
            "alpha_sha256": raster_safety["alpha_sha256"],
        },
        policy_versions={"layout": "renderer-v1"},
        dependency_hashes={"analysis_record": analysis.analysis_record_sha256, "layout_plan": plan["plan_sha256"]},
    )
    candidate = PreferenceCandidate.build(
        owner_id="owner-001",
        target_text=target,
        source_sha256=analysis.payload["source_sha256"],
        style_sha256=canonical_json_sha256({"font": font, "effects": effects}),
        layout_plan_sha256=plan["plan_sha256"],
        recipe_sha256=recipe.recipe_sha256,
        output_sha256=recipe.output_sha256,
        preview_ref={"relative_path": "previews/owner-001.png", "sha256": "1" * 64},
        context_ref={"relative_path": "context/page-001.png", "sha256": "2" * 64},
        metrics={"analysis_record_sha256": analysis.analysis_record_sha256, "raster_safety": raster_safety},
        hard_safety_passed=safe,
    )

    route_bytes = _json_bytes({"units": [{
        "owner_id": "owner-001", "status": "translation_ready" if owner_ready else "review_required",
    }]})
    binding_body = {
        "owner_id": "owner-001",
        "analysis_record_sha256": analysis.analysis_record_sha256,
        "logical_unit_id": "logical-unit-001",
        "physical_subblock_ids": ["block-001"],
        "route_state": "ready_for_layout" if owner_ready else "review_required",
        "route_evidence_sha256": sha256(route_bytes).hexdigest(),
        "producer": {"runtime_id": "integration-consumer-v1", "runtime_sha256": "9" * 64},
    }
    owner_binding = binding_body | {"binding_sha256": canonical_json_sha256(binding_body)}
    event_bytes = _json_bytes({
        "schema": "traduzai.renderer-provider-events.v1",
        "analysis_record_sha256": analysis.analysis_record_sha256,
        "events": [
            {"kind": "stage_started", "stage": "renderer"},
            {"kind": "stage_completed", "stage": "renderer"},
        ],
    })
    trace_body = {
        "schema": "traduzai.renderer-provider-trace.v1",
        "analysis_record_sha256": analysis.analysis_record_sha256,
        "scope": "renderer",
        "provider_calls": {"ocr": 0, "analysis_discovery": 0},
        "event_log_sha256": sha256(event_bytes).hexdigest(),
        "producer": {"runtime_id": "renderer-stage-v1", "runtime_sha256": "8" * 64},
    }
    provider_trace = trace_body | {"trace_sha256": canonical_json_sha256(trace_body)}
    return Chain(
        analysis_payload, owner_binding, provider_trace, logical_bytes, physical_bytes,
        route_bytes, event_bytes, plan, recipe.to_dict(), candidate.to_dict(), rgba,
        authorized, protected,
    )


def _build(chain: Chain, **overrides):
    from typesetter.analysis_lineage import RendererAnalysisLineageReceipt

    values = {
        "analysis_record": chain.analysis,
        "owner_binding": chain.owner_binding,
        "provider_trace": chain.provider_trace,
        "logical_units_artifact_bytes": chain.logical_bytes,
        "physical_subblocks_artifact_bytes": chain.physical_bytes,
        "owner_route_evidence_bytes": chain.route_bytes,
        "renderer_event_log_bytes": chain.event_bytes,
        "layout_plan": chain.plan,
        "recipe": chain.recipe,
        "raster_candidate": chain.candidate,
        "raster_rgba": chain.rgba,
        "authorized_body_mask": chain.authorized,
        "protected_art_mask": chain.protected,
    }
    values.update(overrides)
    return RendererAnalysisLineageReceipt.build(**values)


def test_lineage_binds_consumed_analysis_to_plan_recipe_and_candidate_without_ocr() -> None:
    chain = _chain()
    receipt = _build(chain)
    assert receipt.analysis_record_sha256 == chain.analysis["analysis_record_sha256"]
    assert receipt.layout_plan_sha256 == chain.plan["plan_sha256"]
    assert receipt.recipe_sha256 == chain.recipe["recipe_sha256"]
    assert receipt.raster_candidate_id == chain.candidate["candidate_id"]
    assert receipt.provider_calls == {"ocr": 0, "analysis_discovery": 0}
    assert receipt.state == "ready_for_review"
    assert receipt.publication_allowed is False


def test_lineage_rejects_tampered_analysis_revision() -> None:
    chain = _chain()
    chain.analysis["provider_version"] = "tampered"
    with pytest.raises(ValueError, match="analysis record SHA-256 mismatch"):
        _build(chain)


def test_lineage_rejects_dependency_substitution() -> None:
    chain = _chain()
    payload = dict(chain.candidate)
    payload.pop("candidate_id")
    payload["metrics"]["analysis_record_sha256"] = sha256(b"other revision").hexdigest()
    with pytest.raises(ValueError, match="candidate analysis dependency"):
        _build(chain, raster_candidate=PreferenceCandidate.build(**payload).to_dict())


def test_lineage_keeps_unsafe_candidate_review_required() -> None:
    receipt = _build(_chain(safe=False))
    assert receipt.state == "review_required"
    assert receipt.publication_allowed is False
    assert receipt.raster_safety_status == "review_required"


def test_lineage_rejects_semantically_different_candidate_text() -> None:
    chain = _chain()
    payload = dict(chain.candidate)
    payload.pop("candidate_id")
    payload["target_text"] = "OUTRO TEXTO"
    payload.pop("target_sha256")
    with pytest.raises(ValueError, match="candidate target"):
        _build(chain, raster_candidate=PreferenceCandidate.build(**payload).to_dict())


def test_lineage_rejects_alpha_receipt_from_another_raster() -> None:
    chain = _chain()
    other = chain.rgba.copy()
    other[:20, :20, 3] = 0
    recipe_payload = dict(chain.recipe)
    recipe_payload["output_sha256"] = sha256(np.ascontiguousarray(other).tobytes()).hexdigest()
    recipe_payload.pop("recipe_sha256")
    recipe = RendererRecipe.build(**recipe_payload)
    candidate_payload = dict(chain.candidate)
    candidate_payload.pop("candidate_id")
    candidate_payload["recipe_sha256"] = recipe.recipe_sha256
    candidate_payload["output_sha256"] = recipe.output_sha256
    candidate = PreferenceCandidate.build(**candidate_payload).to_dict()
    with pytest.raises(ValueError, match="alpha differs from RGBA"):
        _build(chain, recipe=recipe.to_dict(), raster_candidate=candidate, raster_rgba=other)


def test_lineage_rejects_owner_ids_absent_from_analysis_artifacts() -> None:
    chain = _chain()
    body = {key: value for key, value in chain.owner_binding.items() if key != "binding_sha256"}
    body["logical_unit_id"] = "does-not-exist"
    body["physical_subblock_ids"] = ["also-does-not-exist"]
    binding = body | {"binding_sha256": canonical_json_sha256(body)}
    with pytest.raises(ValueError, match="logical unit is absent"):
        _build(chain, owner_binding=binding)


def test_lineage_recalculates_safety_from_authoritative_masks() -> None:
    chain = _chain()
    forged = dict(chain.candidate)
    forged.pop("candidate_id")
    forged_safety = dict(forged["metrics"]["raster_safety"])
    forged_safety["authorized_body_sha256"] = "0" * 64
    forged_safety["protected_art_sha256"] = "1" * 64
    forged_body = {key: value for key, value in forged_safety.items() if key != "evidence_sha256"}
    forged_safety["evidence_sha256"] = canonical_json_sha256(forged_body)
    forged["metrics"]["raster_safety"] = forged_safety
    candidate = PreferenceCandidate.build(**forged).to_dict()
    with pytest.raises(ValueError, match="authoritative masks"):
        _build(chain, raster_candidate=candidate)


def test_lineage_requires_hash_bound_zero_provider_event_log() -> None:
    chain = _chain()
    event_bytes = _json_bytes({
        "schema": "traduzai.renderer-provider-events.v1",
        "analysis_record_sha256": chain.analysis["analysis_record_sha256"],
        "events": [
            {"kind": "stage_started", "stage": "renderer"},
            {"kind": "provider_call", "stage": "renderer", "provider": "ocr"},
            {"kind": "stage_completed", "stage": "renderer"},
        ],
    })
    body = {key: value for key, value in chain.provider_trace.items() if key != "trace_sha256"}
    body["event_log_sha256"] = sha256(event_bytes).hexdigest()
    body["provider_calls"] = {"ocr": 1, "analysis_discovery": 0}
    trace = body | {"trace_sha256": canonical_json_sha256(body)}
    with pytest.raises(ValueError, match="complete stage|zero OCR and discovery"):
        _build(chain, provider_trace=trace, renderer_event_log_bytes=event_bytes)


def test_lineage_receipt_provider_calls_are_not_mutable() -> None:
    receipt = _build(_chain())
    exported = receipt.to_dict()
    exported["provider_calls"]["ocr"] = 99
    assert receipt.provider_calls == {"ocr": 0, "analysis_discovery": 0}
    assert receipt.to_dict()["provider_calls"]["ocr"] == 0


def test_lineage_owner_review_state_cannot_be_promoted_by_safe_raster() -> None:
    receipt = _build(_chain(owner_ready=False))
    assert receipt.raster_safety_status == "pass"
    assert receipt.state == "review_required"
    assert receipt.publication_allowed is False


def test_lineage_receipt_roundtrips_and_rejects_tampering_or_invalid_enums() -> None:
    from typesetter.analysis_lineage import RendererAnalysisLineageReceipt

    receipt = _build(_chain())
    assert RendererAnalysisLineageReceipt.from_dict(receipt.to_dict()) == receipt
    tampered = receipt.to_dict()
    tampered["provider_calls"]["ocr"] = 1
    with pytest.raises(ValueError, match="receipt SHA-256 mismatch"):
        RendererAnalysisLineageReceipt.from_dict(tampered)
    invalid = receipt.to_dict()
    invalid["owner_route_state"] = "invented"
    invalid["state"] = "review_required"
    body = {key: value for key, value in invalid.items() if key != "receipt_sha256"}
    invalid["receipt_sha256"] = canonical_json_sha256(body)
    with pytest.raises(ValueError, match="route state"):
        RendererAnalysisLineageReceipt.from_dict(invalid)
