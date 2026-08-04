from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from qa import style_fidelity as style_fidelity_mod

from ownership.model import owner_style_raster_contract_sha256
from qa.style_fidelity import (
    audit_style_fidelity,
    delta_e_2000,
    merge_style_and_functional_gates,
    resolve_original_path,
)
from style_v2_fixtures import valid_owner_style_raster_contract
from test_final_pixel_qa import _graph
from typesetter.owner_style import build_owner_visual_profile


def test_style_fidelity_marks_rotated_gradient_as_mismatch():
    expected = {
        "kind": "linear",
        "colors": ["#6633CC", "#08080A"],
        "stops": [0.0, 1.0],
        "start": [0.0, 0.0],
        "end": [1.0, 1.0],
        "coordinate_space": "glyph_bbox_normalized",
    }
    observed = {**expected, "start": [0.5, 0.0], "end": [0.5, 1.0]}
    profile = {
        "style_evidence_v2": {
            "attributes": {"gradient": {"confidence": 0.95}}
        }
    }
    contract = {
        "schema_version": 2,
        "materialization_plan": {
            "attribute_plans": {
                "gradient": {
                    "resolution_kind": "exact",
                    "target_value": expected,
                }
            }
        },
        "materialization_observation": {
            "attributes": {"gradient": {"canonical_value": observed}}
        },
        "materialization_comparison": {"mismatches": []},
    }

    attributes, catastrophic = style_fidelity_mod._attribute_results(profile, contract)

    assert attributes["gradient"]["status"] == "mismatch"
    assert attributes["gradient"]["expected"] == expected
    assert attributes["gradient"]["observed"] == observed
    assert "gradient" in catastrophic


def test_style_fidelity_matches_identical_resolved_font_identity():
    identity = {
        "family": "CCTotallyAwesomeW00-Bold",
        "file_sha256": "9" * 64,
        "filename": "CCTotallyAwesome W00 Bold.ttf",
        "postscript_name": "CCTotallyAwesomeW00-Bold",
        "subfamily": "Regular",
        "variation_axes": [],
        "weight_class": 700,
        "width_class": 5,
    }
    profile = {
        "style_evidence_v2": {
            "attributes": {"font_name": {"confidence": 0.91}}
        }
    }
    contract = {
        "schema_version": 2,
        "materialization_plan": {
            "attribute_plans": {
                "font_name": {
                    "resolution_kind": "exact",
                    "target_value": identity,
                }
            }
        },
        "materialization_observation": {
            "attributes": {"font_name": {"canonical_value": dict(identity)}}
        },
        "materialization_comparison": {"mismatches": []},
    }

    attributes, catastrophic = style_fidelity_mod._attribute_results(profile, contract)

    assert attributes["font_name"]["status"] == "applied"
    assert attributes["font_name"]["expected"] == identity
    assert attributes["font_name"]["observed"] == identity
    assert catastrophic == []


def _project(*, confidence: float = 0.95, with_contract: bool = True) -> dict:
    graph = _graph(state="rendered")
    page_rgb = np.full((24, 40, 3), 245, dtype=np.uint8)
    glyph = np.zeros(page_rgb.shape[:2], dtype=np.uint8)
    glyph[7:12, 9:24] = 255
    page_rgb[glyph > 0] = 20
    profile = build_owner_visual_profile(
        graph.owners[0],
        page_rgb,
        components=graph.components,
        observations=graph.observations,
        glyph_mask=glyph,
        candidate={
            "confidence": confidence,
            "route_action": "translate_inpaint_render",
            "style_evidence": {
                "source": "pixel_analysis",
                "text_color": "#141414",
                "text_color_confidence": 0.94,
                "font_name": "",
                "font_confidence": 0.0,
                "stroke_color": "",
                "stroke_width_px": 0,
                "stroke_confidence": 0.0,
                "glow": False,
                "glow_confidence": 0.15,
            },
        },
    )
    layer = {
        "owner_id": "owner_a",
        "page_id": "page_001",
        "visual_profile_v2": profile,
        "visual_profile_sha256": profile["visual_profile_sha256"],
    }
    if with_contract:
        rendered = page_rgb.copy()
        rendered[glyph > 0] = 30
        contract = valid_owner_style_raster_contract(
            owner_id="owner_a",
            page_id="page_001",
            before=page_rgb,
            result=rendered,
            glyph_mask=glyph,
            component_geometry_sha256=profile["component_geometry_sha256"],
            visual_profile_sha256=profile["visual_profile_sha256"],
            profile_component_geometry_sha256=profile["component_geometry_sha256"],
            source_artifact_sha256=profile["source_sha256"],
            source_glyph_mask_sha256=profile["glyph_mask_sha256"],
            style_decision=profile["style_application_decision_v2"],
        ).to_dict()
        decision = profile["style_application_decision_v2"]
        contract["applied_attributes"] = copy.deepcopy(decision["applied_attributes"])
        contract["abstained_attributes"] = copy.deepcopy(decision["abstained_attributes"])
        contract["render_metrics"]["core_pixels_outside_safe"] = 0
        contract["render_metrics"]["effect_pixels_outside_safe"] = 0
        contract["status"] = "applied" if contract["applied_attributes"] else "fallback"
        contract["contract_sha256"] = owner_style_raster_contract_sha256(contract)
        layer["style_v2_raster_contract"] = contract
    return {
        "owner_graph_status": "verified",
        "page_owner_graphs": [graph.to_dict()],
        "paginas": [
            {
                "page_id": "page_001",
                "text_layers": [layer],
            }
        ],
    }


def test_png_original_is_resolved_from_project_record(tmp_path):
    image = tmp_path / "source" / "page-one.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    page = {"original_path": "source/page-one.png"}
    assert resolve_original_path(tmp_path, page, 1) == image.resolve()


def test_enforce_blocks_renderable_owner_without_raster_contract(tmp_path):
    report = audit_style_fidelity(_project(with_contract=False), tmp_path, mode="enforce")

    assert report["gate"]["status"] == "BLOCK"
    assert report["coverage"]["contract"] == 0.0


@pytest.mark.parametrize(
    "tamper",
    [
        "owner",
        "profile_hash",
        "evidence_hash",
        "execution_geometry",
        "patch_hash",
        "final_hash",
        "contract_hash",
    ],
)
def test_enforce_blocks_identity_or_hash_mismatch(tmp_path, tamper):
    project = _project()
    layer = project["paginas"][0]["text_layers"][0]
    profile = layer["visual_profile_v2"]
    contract = layer["style_v2_raster_contract"]
    if tamper == "owner":
        contract["owner_id"] = "owner_forged"
    elif tamper == "profile_hash":
        contract["visual_profile_sha256"] = "f" * 64
    elif tamper == "evidence_hash":
        profile["style_application_decision_v2"]["evidence_sha256"] = "f" * 64
    elif tamper == "execution_geometry":
        contract["execution_component_geometry_sha256"] = "f" * 64
    elif tamper == "patch_hash":
        contract["rendered_patch_sha256"] = "f" * 64
    elif tamper == "final_hash":
        contract["rendered_after_sha256"] = "f" * 64
    else:
        contract["contract_sha256"] = "f" * 64

    assert audit_style_fidelity(project, tmp_path, mode="enforce")["gate"]["status"] == "BLOCK"


def test_audit_compares_canonical_decision_to_canonical_raster(tmp_path):
    project = _project()
    contract = project["paginas"][0]["text_layers"][0]["style_v2_raster_contract"]
    contract["applied_attributes"]["fill"] = "#FFFFFF"
    contract["contract_sha256"] = owner_style_raster_contract_sha256(contract)

    report = audit_style_fidelity(project, tmp_path, mode="enforce")

    assert report["owners"][0]["attributes"]["fill"]["status"] == "mismatch"
    assert report["gate"]["status"] == "BLOCK"


def test_low_confidence_abstention_is_explicit_but_not_individual_p0(tmp_path):
    project = _project(confidence=0.2)

    report = audit_style_fidelity(project, tmp_path, mode="enforce")

    assert report["owners"][0]["attributes"]["glow"]["status"] == "abstained"
    assert not report["owners"][0]["catastrophic_mismatches"]


def test_complete_owner_coverage_passes_enforce(tmp_path):
    report = audit_style_fidelity(_project(), tmp_path, mode="enforce")

    assert report["coverage"] == {"profile": 1.0, "contract": 1.0, "metrics": 1.0}
    assert report["gate"]["status"] == "PASS"
    assert report["run_id"] == tmp_path.resolve().name
    assert len(report["owners"][0]["raster_contract_sha256"]) == 64


def test_python_backend_is_audited_by_the_same_canonical_contract(tmp_path):
    project = _project()
    contract = project["paginas"][0]["text_layers"][0]["style_v2_raster_contract"]
    contract["backend"] = "python_ft2font"
    contract["backend_version"] = "matplotlib-test"
    contract["contract_sha256"] = owner_style_raster_contract_sha256(contract)

    report = audit_style_fidelity(project, tmp_path, mode="enforce")

    assert report["backend_counts"] == {"python_ft2font": 1}
    assert report["gate"]["status"] == "PASS"


def test_invalid_segment_contract_blocks_owner(tmp_path):
    project = _project()
    contract = project["paginas"][0]["text_layers"][0]["style_v2_raster_contract"]
    contract["segments"] = [{"segment_id": "forged"}]
    contract["contract_sha256"] = owner_style_raster_contract_sha256(contract)

    assert audit_style_fidelity(project, tmp_path, mode="enforce")["gate"]["status"] == "BLOCK"


def test_missing_required_category_blocks_even_with_complete_owner(tmp_path):
    project = _project()
    project["style_fidelity_required_categories"] = ["all_renderable", "sfx"]

    report = audit_style_fidelity(project, tmp_path, mode="enforce")

    assert report["categories"]["sfx"]["eligible"] == 0
    assert report["gate"]["status"] == "BLOCK"


def test_core_envelope_may_use_verified_safe_geometry_beyond_source_component(tmp_path):
    project = _project()
    contract = project["paginas"][0]["text_layers"][0]["style_v2_raster_contract"]
    contract["glyph_core_envelope"]["bbox_page"] = [0, 0, 4, 4]
    contract["contract_sha256"] = owner_style_raster_contract_sha256(contract)

    report = audit_style_fidelity(project, tmp_path, mode="enforce")

    assert report["coverage"]["metrics"] == 1.0
    assert report["gate"]["status"] == "PASS"


def test_core_pixels_outside_verified_safe_geometry_blocks(tmp_path):
    project = _project()
    contract = project["paginas"][0]["text_layers"][0]["style_v2_raster_contract"]
    contract["render_metrics"]["core_pixels_outside_safe"] = 1
    contract["contract_sha256"] = owner_style_raster_contract_sha256(contract)

    report = audit_style_fidelity(project, tmp_path, mode="enforce")

    assert report["coverage"]["metrics"] == 0.0
    assert report["gate"]["status"] == "BLOCK"
    assert report["owners"][0]["findings"] == [
        {"code": "unsafe_render_metric", "metric": "core_pixels_outside_safe"}
    ]


def test_missing_required_renderer_metric_blocks(tmp_path):
    project = _project()
    contract = project["paginas"][0]["text_layers"][0]["style_v2_raster_contract"]
    del contract["render_metrics"]["core_pixel_count"]
    contract["contract_sha256"] = owner_style_raster_contract_sha256(contract)

    report = audit_style_fidelity(project, tmp_path, mode="enforce")

    assert report["coverage"]["metrics"] == 0.0
    assert report["gate"]["status"] == "BLOCK"


def test_empty_or_zero_denominator_report_never_passes_enforce(tmp_path):
    report = audit_style_fidelity(
        {"owner_graph_status": "verified", "page_owner_graphs": [], "paginas": []},
        tmp_path,
        mode="enforce",
    )
    assert report["gate"]["status"] == "BLOCK"


def test_shadow_preserves_findings_without_blocking_export(tmp_path):
    report = audit_style_fidelity(_project(with_contract=False), tmp_path, mode="shadow")

    assert report["gate"]["status"] == "PASS"
    assert report["gate"]["would_block"] is True


def test_style_gate_cannot_override_blocked_functional_gate():
    merged = merge_style_and_functional_gates(
        {"status": "BLOCK", "issues": [{"code": "english"}]},
        {"status": "PASS"},
    )
    assert merged["status"] == "BLOCK"
    assert merged["functional_gate"]["issues"][0]["code"] == "english"


def test_color_fidelity_reports_delta_e_2000():
    assert delta_e_2000("#FFFFFF", "#FFFFFF") == 0.0
    assert delta_e_2000("#FFFFFF", "#000000") > 90.0


def test_owner_qa_emits_safe_containment_with_explicit_denominator(tmp_path):
    report = audit_style_fidelity(_project(), tmp_path, mode="enforce")

    assert report["metrics"]["safe_containment"] == {
        "evaluated": 1, "contained": 1, "rate": 1.0,
    }
    assert report["metrics"]["effect_containment"] == {
        "evaluated": 1, "contained": 1, "rate": 1.0,
    }
    sealed = dict(report)
    report_sha256 = sealed.pop("report_sha256")
    assert report_sha256 == hashlib.sha256(
        json.dumps(sealed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_missing_outside_safe_measurement_blocks_instead_of_assuming_zero(tmp_path):
    project = _project()
    contract = project["paginas"][0]["text_layers"][0]["style_v2_raster_contract"]
    del contract["render_metrics"]["core_pixels_outside_safe"]
    contract["contract_sha256"] = owner_style_raster_contract_sha256(contract)

    report = audit_style_fidelity(project, tmp_path, mode="enforce")

    assert report["gate"]["status"] == "BLOCK"
    assert any(row["code"] == "required_metric_missing" for row in report["findings"])


def test_catastrophic_count_requires_high_confidence_and_true_mismatch(tmp_path):
    project = _project()
    contract = project["paginas"][0]["text_layers"][0]["style_v2_raster_contract"]
    contract["applied_attributes"]["fill"] = "#FFFFFF"
    contract["contract_sha256"] = owner_style_raster_contract_sha256(contract)

    report = audit_style_fidelity(project, tmp_path, mode="enforce")

    assert report["metrics"]["catastrophic_mismatches"]["count"] == 1
    assert report["metrics"]["catastrophic_mismatches"]["evaluated"] >= 1


def test_style_eligible_owner_remains_in_denominator_after_execution_rollback(tmp_path):
    project = _project(with_contract=False)
    rolled_back = _graph(state="review_required").to_dict()
    rolled_back["owners"][0]["execution_tile_id"] = None
    rolled_back["projections"] = []
    project["page_owner_graphs"] = [rolled_back]
    project["paginas"][0]["text_layers"][0]["owner_style_capture"] = {
        "owner_id": "owner_a", "page_id": "page_001", "eligible": True,
    }

    report = audit_style_fidelity(project, tmp_path, mode="enforce")

    assert report["summary"]["eligible_owner_count"] == 1
    assert report["summary"]["rendered_owner_count"] == 0
    assert report["gate"]["status"] == "BLOCK"
    assert report["owners"][0]["findings"][0]["code"] == "owner_materialization_not_committed"
