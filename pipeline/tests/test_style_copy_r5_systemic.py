from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ownership.delivery import (
    GlyphRunObservation,
    build_owner_text_delivery_contract,
    seal_owner_text_execution_authority,
)
from strip.page_surface_geometry import PageSurfaceGeometry
from test_owner_atomic_execution import (
    _atomic_api,
    _glyph_patch,
    _mutation,
    _v2_contract_with_fill,
)
from typesetter.glyph_rasterizer import rasterize_v2_glyph_layers
from typesetter.style_materialization import (
    build_materialization_observation,
    build_materialization_plan,
    build_resolved_style_intent,
    compare_materialization,
)


FIXTURE = Path(__file__).parent / "fixtures" / "style_copy_r5" / "no_go_cases.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_r5_case(case_id: str) -> dict:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["origin"]["generation"] == "R4"
    return payload["cases"][case_id]


def _case_masks(case: dict) -> tuple[np.ndarray, np.ndarray]:
    height, width = case["shape"]
    core = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in case["glyph_boxes"]:
        cv2.rectangle(core, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
    safe = np.zeros_like(core)
    x1, y1, x2, y2 = case["safe_bbox"]
    safe[y1:y2, x1:x2] = 255
    return core, safe


def _delivery(case: dict, core: np.ndarray):
    authority = seal_owner_text_execution_authority(
        owner_id=f"systemic_{case['category']}",
        page_id="page_systemic",
        source_payload=case["source"],
        translated_payload=case["translated"],
        normalized_chunks=[case["translated"]],
    )
    run = GlyphRunObservation.build(
        text=case["translated"], font_identity="systemic-real-render", span_index=0
    )
    return build_owner_text_delivery_contract(
        execution_authority=authority,
        layout_payload=case["translated"],
        rendered_lines=[case["translated"]],
        rendered_glyph_runs=[run],
        glyph_core_mask=core,
        glyph_span_core_masks=[core],
        rendered_patch_sha256="a" * 64,
    )


def execute_synthetic_owner_case(case: dict):
    core, safe = _case_masks(case)
    raster = rasterize_v2_glyph_layers(
        core,
        safe,
        case["style"],
        rendered_x_height_px=24,
        enforce=True,
    )
    owner_id = f"systemic_{case['category']}"
    approved = dict(case["style"])
    intent = build_resolved_style_intent(
        owner_id=owner_id,
        page_id="page_systemic",
        visual_profile_sha256="1" * 64,
        decision_sha256="2" * 64,
        group_resolution_sha256="3" * 64,
        approved=approved,
        approved_abstentions={},
        attribute_provenance={name: {"evidence_id": f"case:{name}"} for name in approved},
    )
    targets = {name: raster.observed_attributes[name] for name in approved}
    plan = build_materialization_plan(
        intent=intent,
        render_layout_contract_sha256="4" * 64,
        targets=targets,
        resolution_kinds={name: "exact" for name in targets},
        rendered_x_height_px=24,
    )
    observations = {
        "raster": {
            name: {
                "value": raster.observed_attributes[name],
                **raster.attribute_evidence[name],
            }
            for name in targets
        }
    }
    observation = build_materialization_observation(
        plan=plan, domain_observations=observations, render_completed=True
    )
    comparison = compare_materialization(plan, observation)
    protected_changed = 0
    if case.get("protected_art_bbox"):
        protected = np.zeros_like(core)
        x1, y1, x2, y2 = case["protected_art_bbox"]
        protected[y1:y2, x1:x2] = 255
        protected_changed = int(np.count_nonzero((raster.rgba[:, :, 3] > 0) & (protected > 0)))
    return SimpleNamespace(
        render_completed=raster.status in {"applied", "fallback"},
        glyph_core_mask=raster.glyph_core_mask,
        style_raster_contract=SimpleNamespace(
            materialization_comparison=comparison.to_dict(),
            materialization_observation=observation.to_dict(),
        ),
        delivery=_delivery(case, raster.glyph_core_mask),
        core_pixels_outside_safe=raster.metrics["core_pixels_outside_safe"],
        protected_art_changed_pixels=protected_changed,
    )


def execute_case(case_id: str):
    case = load_r5_case(case_id)
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = np.full((12, 16, 3), 220, dtype=np.uint8)
    mutation = _mutation(original)
    rendered = np.array(mutation.result_rgb, copy=True)
    glyph_mask = np.zeros(rendered.shape[:2], dtype=np.uint8)
    glyph_mask[5:7, 5:7] = 255
    rendered[glyph_mask > 0] = (7, 9, 11)
    contract, intent = _v2_contract_with_fill(
        mutation,
        rendered=rendered,
        glyph_mask=glyph_mask,
        target_fill=case["target_fill"],
        observed_fill=case["observed_fill"],
    )
    glyph = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
        style_raster_contract=contract,
    )
    return apply_atomic(
        original,
        mutation,
        glyph,
        expected_style_intent=intent.to_dict(),
        expected_materialization_plan_sha256=contract.materialization_plan_sha256,
    )


def execute_cross_frame_case(case: dict):
    geometry = PageSurfaceGeometry.build(
        logical_width=case["logical_size"][0],
        logical_height=case["logical_size"][1],
        frame_width=case["frame_size"][0],
        frame_height=case["frame_size"][1],
        content_origin_xy=tuple(case["content_origin_xy"]),
    )
    core = np.zeros((24, 160), dtype=np.uint8)
    core[6:18, 8:152] = 255
    return SimpleNamespace(
        delivery=_delivery(case, core),
        frame_bbox=geometry.logical_bbox_to_frame(tuple(case["logical_bbox"])),
        final_text=case["translated"],
    )


def test_r5_card_profile_materializes_nonempty_complete_raster():
    patch = execute_synthetic_owner_case(load_r5_case("card_fill_stroke_glow"))
    assert patch.render_completed is True
    assert np.count_nonzero(patch.glyph_core_mask) > 0
    assert patch.style_raster_contract.materialization_comparison["status"] == "match"


def test_r5_fill_equivalence_commits_but_true_difference_rolls_back():
    assert execute_case("canonical_fill_alias").committed is True
    mismatch = execute_case("material_fill_difference")
    assert mismatch.committed is False
    assert ":raster:fill:" in mismatch.reason


def test_r5_cross_frame_case_preserves_complete_body_and_correct_crop():
    result = execute_cross_frame_case(load_r5_case("cross_tile_framed_output"))
    assert result.delivery.status == "delivered"
    assert result.frame_bbox == (164, 985, 373, 1104)
    assert result.final_text == "É POR ISSO QUE VOCÊ DEVERIA TER IDO ANTES"


@pytest.mark.parametrize("case_id", ["white_balloon", "burst"])
def test_r5_speech_containers_keep_all_glyphs_inside_verified_safe_geometry(case_id):
    result = execute_synthetic_owner_case(load_r5_case(case_id))
    assert result.delivery.status == "delivered"
    assert result.core_pixels_outside_safe == 0


def test_r5_dark_panel_observes_real_font_and_fill_without_request_echo():
    patch = execute_synthetic_owner_case(load_r5_case("dark_panel"))
    assert patch.style_raster_contract.materialization_comparison["status"] == "match"
    fill = patch.style_raster_contract.materialization_observation["attributes"]["fill"]
    assert fill["evidence_kind"] == "layer_pixels_and_mask"
    assert fill["evidence_sha256"]


def test_r5_text_over_art_preserves_protected_art_and_complete_body():
    result = execute_synthetic_owner_case(load_r5_case("text_over_art"))
    assert result.delivery.status == "delivered"
    assert result.protected_art_changed_pixels == 0


def test_r5_fixture_is_traceable_to_all_nine_r4_no_go_categories():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report_path = REPO_ROOT / fixture["origin"]["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_categories = {
        "colored_card",
        "table_ranking",
        "translucent_balloon",
        "primary_ocr_omission",
        "white_balloon",
        "burst",
        "cross_tile_owner",
        "dark_panel",
        "text_over_art",
    }

    assert report["verdicts"]["overall"] == fixture["origin"]["verdict"] == "NO-GO"
    assert set(report["matrix"]["categories"]) == expected_categories
    assert all(
        row["functional"] == "NO-GO" and row["style"] == "NO-GO"
        for row in report["matrix"]["categories"].values()
    )
