"""TDD contracts for atomic cleanup-and-render execution by owner."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership import model as owner_model  # noqa: E402
from ownership.model import (  # noqa: E402
    OwnerGraph,
    OwnerMutation,
    OwnerProjection,
    SourceTextComponent,
    TextObservation,
    TextOwner,
    owner_residual_evidence_sha256,
)
from ownership.render_geometry import build_owner_render_geometry  # noqa: E402
from strip import process_bands  # noqa: E402
from style_v2_fixtures import (  # noqa: E402
    valid_owner_style_raster_contract,
    valid_owner_style_raster_contract_v2,
)
from typesetter.style_materialization import (  # noqa: E402
    build_materialization_observation,
    build_materialization_plan,
    build_resolved_style_intent,
    compare_materialization,
)
from typesetter.owner_render_quality import OwnerRenderQuality  # noqa: E402


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(b"traduzai.ndarray.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(dimension) for dimension in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _action_mask_ref(owner_id: str, tile_id: str) -> str:
    identity_hash = sha256(owner_id.encode("utf-8")).hexdigest()[:12]
    return f"owner_masks/{owner_id}--{identity_hash}/{tile_id}/action_mask.png"


def _safe_polygon_for_bbox(
    bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], ...]:
    x1, y1, x2, y2 = bbox
    return ((x1, y1), (x2 - 1, y1), (x2 - 1, y2 - 1), (x1, y2 - 1))


def _polygon_sha256(points: tuple[tuple[int, int], ...]) -> str:
    payload = json.dumps(points, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _render_quality(glyph_mask: np.ndarray) -> OwnerRenderQuality:
    ys, _xs = np.nonzero(glyph_mask)
    ink_height = int(ys.max() - ys.min() + 1) if len(ys) else 0
    return OwnerRenderQuality(
        schema_version=1,
        status="ok",
        font_size_final=max(12, ink_height),
        minimum_legible_font_px=12,
        source_ink_height_px=None,
        render_ink_height_px=ink_height,
        source_x_height_px=None,
        render_x_height_px=float(ink_height) * 0.7,
        source_scale_ratio=None,
        x_height_ratio=None,
        rendered_line_core_heights_px=(ink_height,) if ink_height else (),
        safe_height_occupancy=0.25 if ink_height else 0.0,
        safe_area_occupancy=0.10 if ink_height else 0.0,
        wrapped_line_count=1 if ink_height else 0,
        containment_status="ok",
        outside_safe_pixels=0,
        page_width=int(glyph_mask.shape[1]),
        page_height=int(glyph_mask.shape[0]),
        reasons=(),
    )


def _render_geometry(
    *,
    owner_id: str,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    protected_art_mask_sha256: str,
):
    height, width = shape
    component_id = f"component_{owner_id}"
    observation_id = f"observation_{owner_id}"
    x1, y1, x2, y2 = bbox
    polygon = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    graph = OwnerGraph(
        2,
        "page_001",
        [SourceTextComponent(component_id, "page_001", bbox, polygon, ("fixture",))],
        [TextObservation(observation_id, "page_001", (component_id,), "SOURCE", 1.0, "fixture", bbox, polygons_page=(polygon,))],
        [TextOwner(owner_id, "page_001", [component_id], [observation_id], [observation_id], "freeform_sfx", "SOURCE", "ALVO", "owned", "translated", "translate_inpaint_render", "tile_executor")],
        [OwnerProjection(owner_id, "tile_executor", "executor", (0, 0, width, height), (0, 0, width, height), (0, 0))],
    )
    return build_owner_render_geometry(
        graph,
        owner_id,
        page_width=width,
        page_height=height,
        protected_art_mask_sha256=protected_art_mask_sha256,
    )


def _atomic_api() -> tuple[Any, type[Any], type[Any]]:
    """Resolve the wished-for Task 11 API inside each test for six useful REDs."""

    return (
        getattr(process_bands, "apply_atomic_owner_execution"),
        getattr(owner_model, "OwnerGlyphPatch"),
        getattr(owner_model, "OwnerExecutionCommit"),
    )


def _mutation(
    original_rgb: np.ndarray,
    *,
    owner_id: str = "owner_a",
    action_box: tuple[int, int, int, int] = (4, 4, 8, 8),
    owner_bbox_page: tuple[int, int, int, int] = (2, 2, 14, 10),
    execution_tile_id: str | None = "tile_executor",
) -> OwnerMutation:
    action_mask = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = action_box
    action_mask[y1:y2, x1:x2] = 255
    result_rgb = original_rgb.copy()
    result_rgb[action_mask > 0] = (32, 48, 64)
    changed = np.any(result_rgb != original_rgb, axis=2)
    changed_mask = np.where(changed, 255, 0).astype(np.uint8)
    protected_art_mask = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
    before_sha256 = _array_sha256(original_rgb)
    after_sha256 = _array_sha256(result_rgb)
    action_mask_sha256 = _array_sha256(action_mask)
    protected_art_mask_sha256 = _array_sha256(protected_art_mask)
    component_geometry_sha256 = sha256(b"component-geometry").hexdigest()
    render_geometry = _render_geometry(
        owner_id=owner_id,
        bbox=owner_bbox_page,
        shape=original_rgb.shape[:2],
        protected_art_mask_sha256=protected_art_mask_sha256,
    )
    residual_evidence_sha256 = owner_residual_evidence_sha256(
        owner_id=owner_id,
        page_id="page_001",
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        action_mask_sha256=action_mask_sha256,
        protected_art_mask_sha256=protected_art_mask_sha256,
        component_geometry_sha256=component_geometry_sha256,
        residual_score=0.0,
        residual_threshold=0.01,
        residual_method="fixture_residual_v1",
        residual_flags=(),
    )
    return OwnerMutation(
        owner_id=owner_id,
        page_id="page_001",
        coordinate_space="logical_page",
        action_mask_ref=_action_mask_ref(owner_id, execution_tile_id or "context_only"),
        result_rgb=result_rgb,
        action_mask=action_mask,
        protected_art_mask=protected_art_mask,
        changed_mask=changed_mask,
        engine="test_engine",
        mask_pixels=int(np.count_nonzero(action_mask)),
        changed_pixels=int(np.count_nonzero(changed_mask)),
        changed_outside_owner_pixels=0,
        protected_art_changed_pixels=0,
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        action_mask_sha256=action_mask_sha256,
        changed_mask_sha256=_array_sha256(changed_mask),
        engine_crop_bbox_page=(0, 0, original_rgb.shape[1], original_rgb.shape[0]),
        owner_bbox_page=owner_bbox_page,
        component_geometry_sha256=component_geometry_sha256,
        owner_render_geometry_sha256=render_geometry.geometry_sha256,
        protected_art_mask_sha256=protected_art_mask_sha256,
        residual_score=0.0,
        residual_verified=True,
        residual_threshold=0.01,
        residual_method="fixture_residual_v1",
        residual_evidence_sha256=residual_evidence_sha256,
        residual_flags=(),
        execution_tile_id=execution_tile_id,
        component_geometry_verified=True,
    )


def test_owner_commit_rejects_unverified_residual_score() -> None:
    apply_atomic, glyph_patch_type, _commit_type = _atomic_api()
    original = np.full((12, 16, 3), 220, dtype=np.uint8)
    mutation = replace(
        _mutation(original),
        residual_score=None,
        residual_verified=False,
    )
    glyph = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
    )

    commit = apply_atomic(original, mutation, glyph)

    assert commit.committed is False
    assert commit.review_required is True
    assert "residual" in commit.reason


def test_owner_commit_rejects_residual_above_profile_threshold() -> None:
    apply_atomic, glyph_patch_type, _commit_type = _atomic_api()
    original = np.full((12, 16, 3), 220, dtype=np.uint8)
    mutation = replace(
        _mutation(original),
        residual_score=0.25,
        residual_verified=True,
        residual_threshold=0.01,
        residual_method="fixture_residual_v1",
        residual_evidence_sha256="a" * 64,
        residual_flags=("dark_residual_pixels",),
    )
    glyph = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
    )

    commit = apply_atomic(original, mutation, glyph)

    assert commit.committed is False
    assert commit.review_required is True
    assert "residual" in commit.reason


def test_residual_threshold_and_evidence_are_part_of_mutation_hash_chain() -> None:
    original = np.full((12, 16, 3), 220, dtype=np.uint8)
    mutation = _mutation(original)

    assert mutation.residual_verified is True
    assert mutation.residual_threshold >= 0.0
    assert mutation.residual_method
    assert len(mutation.residual_evidence_sha256) == 64

    changed_threshold_hash = owner_residual_evidence_sha256(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        before_sha256=mutation.before_sha256,
        after_sha256=mutation.after_sha256,
        action_mask_sha256=mutation.action_mask_sha256,
        protected_art_mask_sha256=mutation.protected_art_mask_sha256,
        component_geometry_sha256=mutation.component_geometry_sha256,
        residual_score=mutation.residual_score,
        residual_threshold=mutation.residual_threshold + 0.01,
        residual_method=mutation.residual_method,
        residual_flags=mutation.residual_flags,
    )
    changed_method_hash = owner_residual_evidence_sha256(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        before_sha256=mutation.before_sha256,
        after_sha256=mutation.after_sha256,
        action_mask_sha256=mutation.action_mask_sha256,
        protected_art_mask_sha256=mutation.protected_art_mask_sha256,
        component_geometry_sha256=mutation.component_geometry_sha256,
        residual_score=mutation.residual_score,
        residual_threshold=mutation.residual_threshold,
        residual_method="tampered_method",
        residual_flags=mutation.residual_flags,
    )
    assert changed_threshold_hash != mutation.residual_evidence_sha256
    assert changed_method_hash != mutation.residual_evidence_sha256


def _glyph_patch(
    mutation: OwnerMutation,
    glyph_patch_type: type[Any],
    *,
    render_completed: bool,
    fit_status: str,
    projection_role: str = "executor",
    before_sha256: str | None = None,
    render_safe_polygon_page: tuple[tuple[int, int], ...] | None = None,
    style_raster_contract: Any | None = None,
) -> Any:
    rendered_rgb = np.array(mutation.result_rgb, copy=True)
    glyph_mask = np.zeros(rendered_rgb.shape[:2], dtype=np.uint8)
    if render_completed:
        glyph_mask[5:7, 5:7] = 255
        rendered_rgb[glyph_mask > 0] = (7, 9, 11)
    safe_polygon = render_safe_polygon_page or _safe_polygon_for_bbox(
        mutation.owner_bbox_page
    )
    render_geometry = _render_geometry(
        owner_id=mutation.owner_id,
        bbox=mutation.owner_bbox_page,
        shape=rendered_rgb.shape[:2],
        protected_art_mask_sha256=str(mutation.protected_art_mask_sha256),
    )
    return glyph_patch_type(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        coordinate_space="logical_page",
        result_rgb=rendered_rgb,
        glyph_mask=glyph_mask,
        glyph_bbox_page=(5, 5, 7, 7) if render_completed else None,
        render_completed=render_completed,
        fit_status=fit_status,
        before_sha256=before_sha256 or mutation.after_sha256,
        after_sha256=_array_sha256(rendered_rgb),
        glyph_mask_sha256=_array_sha256(glyph_mask),
        changed_outside_glyph_mask_pixels=0,
        render_safe_polygon_page=safe_polygon,
        render_safe_polygon_sha256=_polygon_sha256(safe_polygon),
        component_geometry_sha256=mutation.component_geometry_sha256,
        render_quality_contract=_render_quality(glyph_mask),
        style_raster_contract=(
            style_raster_contract
            or valid_owner_style_raster_contract(
                owner_id=mutation.owner_id,
                page_id=mutation.page_id,
                before=np.asarray(mutation.result_rgb),
                result=rendered_rgb,
                glyph_mask=glyph_mask,
                component_geometry_sha256=mutation.component_geometry_sha256,
            )
        ),
        owner_render_geometry_sha256=render_geometry.geometry_sha256,
        owner_render_geometry=render_geometry,
        execution_tile_id=mutation.execution_tile_id,
        projection_role=projection_role,
    )


def test_atomic_commit_accepts_canonically_equivalent_v2_observation() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = np.full((12, 16, 3), 220, dtype=np.uint8)
    mutation = _mutation(original)
    rendered = np.array(mutation.result_rgb, copy=True)
    glyph_mask = np.zeros(rendered.shape[:2], dtype=np.uint8)
    glyph_mask[5:7, 5:7] = 255
    rendered[glyph_mask > 0] = (7, 9, 11)
    contract = valid_owner_style_raster_contract_v2(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        before=np.asarray(mutation.result_rgb),
        result=rendered,
        glyph_mask=glyph_mask,
        component_geometry_sha256=mutation.component_geometry_sha256,
    )
    glyph = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
        style_raster_contract=contract,
    )
    intent = build_resolved_style_intent(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        visual_profile_sha256=contract.visual_profile_sha256,
        decision_sha256="a" * 64,
        group_resolution_sha256="b" * 64,
        approved={"fill": "#fff"},
        approved_abstentions={},
        attribute_provenance={"fill": {"evidence_id": "fixture-fill"}},
    )

    commit = apply_atomic(
        original,
        mutation,
        glyph,
        expected_style_intent=intent.to_dict(),
        expected_materialization_plan_sha256=contract.materialization_plan_sha256,
    )

    assert commit.committed is True


def _v2_contract_with_fill(
    mutation: OwnerMutation,
    *,
    rendered: np.ndarray,
    glyph_mask: np.ndarray,
    target_fill: Any,
    observed_fill: Any,
    attribute_name: str = "fill",
    intent_value: Any = "#fff",
    domain: str = "raster",
) -> tuple[Any, Any]:
    base = valid_owner_style_raster_contract_v2(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        before=np.asarray(mutation.result_rgb),
        result=rendered,
        glyph_mask=glyph_mask,
        component_geometry_sha256=mutation.component_geometry_sha256,
    ).to_dict()
    intent = build_resolved_style_intent(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        visual_profile_sha256=base["visual_profile_sha256"],
        decision_sha256="a" * 64,
        group_resolution_sha256="b" * 64,
        approved={attribute_name: intent_value},
        approved_abstentions={},
        attribute_provenance={
            attribute_name: {"evidence_id": f"fixture-{attribute_name}"}
        },
    )
    plan = build_materialization_plan(
        intent=intent,
        render_layout_contract_sha256="c" * 64,
        targets={attribute_name: target_fill},
        resolution_kinds={
            attribute_name: (
                "policy_adjusted"
                if attribute_name == "font_size_px" and target_fill != intent_value
                else "exact"
            )
        },
        rendered_x_height_px=20,
    )
    observation = build_materialization_observation(
        plan=plan,
        domain_observations={
            domain: {
                attribute_name: {
                    "value": observed_fill,
                    "evidence_kind": "layer_pixels_and_mask",
                    "evidence_sha256": "d" * 64,
                }
            }
        },
        render_completed=True,
    )
    comparison = compare_materialization(plan, observation)
    mismatch_reason = (
        str(comparison.mismatches[0]["reason"])
        if comparison.mismatches
        else ""
    )
    base.update(
        {
            "schema_version": 2,
            "status": "applied" if comparison.status == "match" else "review_required",
            "render_status": "completed",
            "materialization_status": comparison.status,
            "style_intent_sha256": intent.intent_sha256,
            "materialization_plan_sha256": plan.plan_sha256,
            "materialization_observation_sha256": observation.observation_sha256,
            "materialization_plan": plan.to_dict(),
            "materialization_observation": observation.to_dict(),
            "materialization_comparison": comparison.to_dict(),
            "backend_selection_reason": "fixture_observable_backend",
            "requested_attributes": {attribute_name: intent_value},
            "applied_attributes": (
                {
                    attribute_name: plan.to_dict()["attribute_plans"]
                    [attribute_name]["target_value"]
                }
                if comparison.status == "match"
                else {}
            ),
            "abstained_attributes": (
                {}
                if comparison.status == "match"
                else {attribute_name: mismatch_reason}
            ),
        }
    )
    base["contract_sha256"] = owner_model.owner_style_raster_contract_sha256(base)
    return owner_model.OwnerStyleRasterContractV2(**base), intent


def test_atomic_commit_does_not_compare_functional_layout_to_style_decision() -> None:
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
        target_fill=36,
        observed_fill=36,
        attribute_name="font_size_px",
        intent_value=48,
        domain="layout",
    )
    glyph = _glyph_patch(
        mutation, glyph_patch_type, render_completed=True, fit_status="ok",
        style_raster_contract=contract,
    )

    commit = apply_atomic(
        original,
        mutation,
        glyph,
        expected_style_decision={
            "applied_attributes": {"font_size_px": 48},
            "abstained_attributes": {},
        },
        expected_style_intent=intent.to_dict(),
        expected_materialization_plan_sha256=contract.materialization_plan_sha256,
    )

    assert commit.committed is True


def test_atomic_commit_rolls_back_true_material_divergence_with_precise_reason() -> None:
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
        target_fill="#FFFFFF",
        observed_fill="#FF0000",
    )
    glyph = _glyph_patch(
        mutation, glyph_patch_type, render_completed=True, fit_status="ok",
        style_raster_contract=contract,
    )

    commit = apply_atomic(
        original,
        mutation,
        glyph,
        expected_style_intent=intent.to_dict(),
        expected_materialization_plan_sha256=contract.materialization_plan_sha256,
    )

    assert commit.committed is False
    assert commit.reason == (
        "render_contract_invalid:materialization_mismatch:"
        "raster:fill:canonical_value_mismatch"
    )
    assert np.array_equal(commit.result_rgb, original)


def test_atomic_commit_rejects_self_consistent_patch_diverging_from_sealed_plan() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = np.full((12, 16, 3), 220, dtype=np.uint8)
    mutation = _mutation(original)
    rendered = np.array(mutation.result_rgb, copy=True)
    glyph_mask = np.zeros(rendered.shape[:2], dtype=np.uint8)
    glyph_mask[5:7, 5:7] = 255
    rendered[glyph_mask > 0] = (7, 9, 11)
    sealed_contract, intent = _v2_contract_with_fill(
        mutation, rendered=rendered, glyph_mask=glyph_mask,
        target_fill="#FFFFFF", observed_fill="#FFFFFF",
    )
    forged_contract, _ = _v2_contract_with_fill(
        mutation, rendered=rendered, glyph_mask=glyph_mask,
        target_fill="#FF0000", observed_fill="#FF0000",
    )
    glyph = _glyph_patch(
        mutation, glyph_patch_type, render_completed=True, fit_status="ok",
        style_raster_contract=forged_contract,
    )

    commit = apply_atomic(
        original,
        mutation,
        glyph,
        expected_style_intent=intent.to_dict(),
        expected_materialization_plan_sha256=(
            sealed_contract.materialization_plan_sha256
        ),
    )

    assert commit.committed is False
    assert commit.reason == "render_contract_invalid:sealed_materialization_plan_mismatch"


def _glyph_patch_for_mask(
    mutation: OwnerMutation,
    glyph_patch_type: type[Any],
    glyph_mask: np.ndarray,
    *,
    render_safe_polygon_page: tuple[tuple[int, int], ...] | None = None,
) -> Any:
    rendered_rgb = np.array(mutation.result_rgb, copy=True)
    rendered_rgb[glyph_mask > 0] = (7, 9, 11)
    positive_y, positive_x = np.nonzero(glyph_mask)
    glyph_bbox = (
        int(positive_x.min()),
        int(positive_y.min()),
        int(positive_x.max()) + 1,
        int(positive_y.max()) + 1,
    )
    safe_polygon = render_safe_polygon_page or _safe_polygon_for_bbox(
        mutation.owner_bbox_page
    )
    render_geometry = _render_geometry(
        owner_id=mutation.owner_id,
        bbox=mutation.owner_bbox_page,
        shape=rendered_rgb.shape[:2],
        protected_art_mask_sha256=str(mutation.protected_art_mask_sha256),
    )
    return glyph_patch_type(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        coordinate_space="logical_page",
        result_rgb=rendered_rgb,
        glyph_mask=glyph_mask,
        glyph_bbox_page=glyph_bbox,
        render_completed=True,
        fit_status="ok",
        before_sha256=mutation.after_sha256,
        after_sha256=_array_sha256(rendered_rgb),
        glyph_mask_sha256=_array_sha256(glyph_mask),
        changed_outside_glyph_mask_pixels=0,
        render_safe_polygon_page=safe_polygon,
        render_safe_polygon_sha256=_polygon_sha256(safe_polygon),
        component_geometry_sha256=mutation.component_geometry_sha256,
        render_quality_contract=_render_quality(glyph_mask),
        style_raster_contract=valid_owner_style_raster_contract(
            owner_id=mutation.owner_id,
            page_id=mutation.page_id,
            before=np.asarray(mutation.result_rgb),
            result=rendered_rgb,
            glyph_mask=glyph_mask,
            component_geometry_sha256=mutation.component_geometry_sha256,
        ),
        owner_render_geometry_sha256=render_geometry.geometry_sha256,
        owner_render_geometry=render_geometry,
        execution_tile_id=mutation.execution_tile_id,
        projection_role="executor",
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "owner_id",
        "visual_profile_sha256",
        "execution_geometry",
        "patch_sha256",
        "after_sha256",
        "contract_sha256",
    ],
)
def test_atomic_commit_rolls_back_tampered_style_contract(tamper: str) -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = np.full((18, 24, 3), 230, dtype=np.uint8)
    mutation = _mutation(original)
    glyph_patch = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
    )
    payload = glyph_patch.style_raster_contract.to_dict()
    if tamper == "owner_id":
        payload["owner_id"] = "owner_other"
    elif tamper == "visual_profile_sha256":
        payload["visual_profile_sha256"] = "b" * 64
    elif tamper == "execution_geometry":
        payload["execution_component_geometry_sha256"] = "c" * 64
    elif tamper == "patch_sha256":
        payload["rendered_patch_sha256"] = "d" * 64
    elif tamper == "after_sha256":
        payload["rendered_after_sha256"] = "e" * 64
    if tamper == "contract_sha256":
        payload["contract_sha256"] = "f" * 64
    else:
        payload["contract_sha256"] = owner_model.owner_style_raster_contract_sha256(
            payload
        )
    forged = owner_model.OwnerStyleRasterContract(**payload)
    object.__setattr__(glyph_patch, "style_raster_contract", forged)

    commit = apply_atomic(
        original,
        mutation,
        glyph_patch,
        expected_visual_profile_sha256=(
            glyph_patch.style_raster_contract.visual_profile_sha256
            if tamper != "visual_profile_sha256"
            else "a" * 64
        ),
        expected_profile_component_geometry_sha256=(
            valid_owner_style_raster_contract().profile_component_geometry_sha256
        ),
    )

    assert commit.committed is False
    assert commit.review_required is True
    assert np.array_equal(commit.result_rgb, original)
    assert commit.reason.startswith("render_contract_invalid:")


def test_missing_second_line_mask_revokes_entire_owner() -> None:
    from inpainter.owner_mask import (
        OwnerMaskEvidence,
        UnsafeOwnerMaskError,
        build_owner_mask_plan,
    )
    from ownership.model import TextOwner

    image = np.full((40, 56, 3), 220, dtype=np.uint8)
    owner = TextOwner(
        owner_id="owner_atomic_multiline",
        page_id="page_001",
        component_ids=["component_body"],
        observation_ids=["observation_body"],
        selected_observation_ids=["observation_body"],
        semantic_role="dialogue_body",
        source_payload="FIRST LINE SECOND LINE",
        translated_payload="PRIMEIRA LINHA SEGUNDA LINHA",
        disposition="owned",
        state="translated",
        route_action="translate_inpaint_render",
        execution_tile_id="tile_executor",
    )
    first_line = np.zeros(image.shape[:2], dtype=np.uint8)
    first_line[8:11, 10:30] = 255

    with np.testing.assert_raises_regex(
        UnsafeOwnerMaskError,
        "line|coverage|identity",
    ):
        build_owner_mask_plan(
            image,
            owner,
            [
                OwnerMaskEvidence(
                    evidence_id="first_line",
                    component_id="component_body",
                    glyph_mask=first_line,
                    observation_id="observation_body",
                    line_index=0,
                )
            ],
            expected_line_ids=(("observation_body", 0), ("observation_body", 1)),
            owner_component_bboxes_page={"component_body": (6, 5, 36, 30)},
        )

    assert owner.state == "review_required"
    assert owner.route_action == "review_required"
    assert owner.execution_tile_id == "tile_executor"
    assert owner.action_mask_ref is None


def _page() -> np.ndarray:
    page = np.full((12, 16, 3), 210, dtype=np.uint8)
    # A translated glyph belonging to a previously committed neighboring owner.
    page[3:6, 11:13] = (19, 23, 29)
    return page


def test_failed_render_rolls_back_only_its_owner_action_mask() -> None:
    apply_atomic, glyph_patch_type, commit_type = _atomic_api()
    original = _page()
    mutation = _mutation(original)
    glyph_patch = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=False,
        fit_status="failed",
    )

    commit = apply_atomic(original, mutation, glyph_patch)

    assert isinstance(commit, commit_type)
    assert commit.committed is False
    assert commit.review_required is True
    assert commit.state == "review_required"
    np.testing.assert_array_equal(
        commit.result_rgb[mutation.action_mask > 0],
        original[mutation.action_mask > 0],
    )
    np.testing.assert_array_equal(
        commit.result_rgb[mutation.action_mask == 0],
        mutation.result_rgb[mutation.action_mask == 0],
    )


def test_rollback_never_uses_balloon_bbox_when_action_mask_exists() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    mutation = _mutation(
        original,
        action_box=(4, 4, 6, 6),
        owner_bbox_page=(1, 1, 15, 11),
    )
    glyph_patch = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=False,
        fit_status="overflow",
    )

    commit = apply_atomic(original, mutation, glyph_patch)

    assert commit.rollback_mask_sha256 == mutation.action_mask_sha256
    assert commit.rollback_pixels == mutation.mask_pixels
    bbox_only_pixel = (3, 12)
    assert mutation.action_mask[bbox_only_pixel] == 0
    np.testing.assert_array_equal(
        commit.result_rgb[bbox_only_pixel],
        original[bbox_only_pixel],
    )


def test_neighbor_owner_pixels_survive_rollback() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    neighbor_pixels_before = original[3:6, 11:13].copy()
    mutation = _mutation(
        original,
        action_box=(4, 4, 7, 8),
        owner_bbox_page=(2, 2, 14, 10),
    )
    glyph_patch = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=False,
        fit_status="below_minimum_legible",
    )

    commit = apply_atomic(original, mutation, glyph_patch)

    np.testing.assert_array_equal(commit.result_rgb[3:6, 11:13], neighbor_pixels_before)
    np.testing.assert_array_equal(original[3:6, 11:13], neighbor_pixels_before)


def test_context_only_projection_produces_no_owner_mutation() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    mutation = _mutation(original, execution_tile_id="tile_context")
    glyph_patch = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
        projection_role="context_only",
    )

    commit = apply_atomic(original, mutation, glyph_patch)

    assert commit.committed is False
    assert commit.review_required is True
    assert "context" in commit.reason
    np.testing.assert_array_equal(commit.result_rgb, original)


def test_copyback_cannot_restore_source_over_another_owner_glyph() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    neighbor_pixels_before = original[3:6, 11:13].copy()
    mutation = _mutation(
        original,
        action_box=(4, 4, 8, 8),
        owner_bbox_page=(2, 2, 14, 10),
    )
    glyph_patch = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
    )

    commit = apply_atomic(original, mutation, glyph_patch)

    assert commit.committed is True
    assert commit.review_required is False
    np.testing.assert_array_equal(commit.result_rgb[3:6, 11:13], neighbor_pixels_before)
    assert np.all(commit.result_rgb[5:7, 5:7] == (7, 9, 11))


def test_inpaint_without_successful_render_is_not_committed() -> None:
    apply_atomic, _, _ = _atomic_api()
    original = _page()
    mutation = _mutation(original)

    commit = apply_atomic(original, mutation, None)

    assert commit.committed is False
    assert commit.cleanup_committed is False
    assert commit.render_committed is False
    assert commit.review_required is True
    np.testing.assert_array_equal(commit.result_rgb, original)


def test_owner_cleanup_and_glyph_hash_chain_must_match() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    mutation = _mutation(original)
    glyph_patch = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
        before_sha256="0" * 64,
    )

    commit = apply_atomic(original, mutation, glyph_patch)

    assert commit.committed is False
    assert commit.review_required is True
    assert "hash" in commit.reason
    np.testing.assert_array_equal(commit.result_rgb, original)


def test_glyph_fit_status_must_be_canonical_ok() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    mutation = _mutation(original)

    for fit_status in ("OK", " ok ", "Ok"):
        glyph_patch = _glyph_patch(
            mutation,
            glyph_patch_type,
            render_completed=True,
            fit_status=fit_status,
        )
        commit = apply_atomic(original, mutation, glyph_patch)
        assert commit.committed is False
        assert commit.review_required is True
        assert "fit status" in commit.reason
        np.testing.assert_array_equal(commit.result_rgb, original)


def test_action_mask_cannot_authorize_protected_art_even_when_pixel_is_unchanged() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    mutation = _mutation(original)
    result_rgb = np.array(mutation.result_rgb, copy=True)
    result_rgb[4, 4] = original[4, 4]
    changed = np.any(result_rgb != original, axis=2)
    changed_mask = np.where(changed, 255, 0).astype(np.uint8)
    protected = np.zeros(original.shape[:2], dtype=np.uint8)
    protected[4, 4] = 255
    mutation = replace(
        mutation,
        result_rgb=result_rgb,
        protected_art_mask=protected,
        changed_mask=changed_mask,
        changed_pixels=int(np.count_nonzero(changed)),
        after_sha256=_array_sha256(result_rgb),
        changed_mask_sha256=_array_sha256(changed_mask),
    )
    glyph_patch = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
    )

    commit = apply_atomic(original, mutation, glyph_patch)

    assert commit.committed is False
    assert commit.review_required is True
    assert "protected" in commit.reason
    np.testing.assert_array_equal(commit.result_rgb, original)


def test_action_mask_reference_must_be_bound_to_owner_identity() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    mutation = _mutation(original)
    mutation = replace(
        mutation,
        action_mask_ref=_action_mask_ref("owner_b", "tile_executor"),
    )
    glyph_patch = _glyph_patch(
        mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
    )

    commit = apply_atomic(original, mutation, glyph_patch)

    assert commit.committed is False
    assert commit.review_required is True
    assert "action_mask_ref" in commit.reason
    np.testing.assert_array_equal(commit.result_rgb, original)


def test_glyph_patch_cannot_escape_render_safe_polygon() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    mutation = _mutation(original, owner_bbox_page=(2, 2, 14, 10))
    glyph_mask = np.zeros(original.shape[:2], dtype=np.uint8)
    glyph_mask[10:12, 14:16] = 255
    glyph_patch = _glyph_patch_for_mask(mutation, glyph_patch_type, glyph_mask)

    commit = apply_atomic(original, mutation, glyph_patch)

    assert commit.committed is False
    assert commit.review_required is True
    assert "safe polygon" in commit.reason
    np.testing.assert_array_equal(commit.result_rgb, original)


def test_translated_glyph_may_expand_beyond_source_bbox_inside_safe_polygon() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    mutation = _mutation(
        original,
        action_box=(4, 4, 8, 8),
        owner_bbox_page=(4, 4, 8, 8),
    )
    safe_polygon = _safe_polygon_for_bbox((2, 2, 14, 10))
    glyph_mask = np.zeros(original.shape[:2], dtype=np.uint8)
    glyph_mask[5:8, 8:13] = 255
    glyph_patch = _glyph_patch_for_mask(
        mutation,
        glyph_patch_type,
        glyph_mask,
        render_safe_polygon_page=safe_polygon,
    )

    commit = apply_atomic(original, mutation, glyph_patch)

    assert commit.committed is True
    assert commit.review_required is False
    assert glyph_patch.glyph_bbox_page[2] > mutation.owner_bbox_page[2]


def test_atomic_owner_execution_rejects_page_wide_masks() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    broad_mutation = _mutation(
        original,
        action_box=(0, 0, original.shape[1], original.shape[0]),
        owner_bbox_page=(0, 0, original.shape[1], original.shape[0]),
    )
    small_glyph = _glyph_patch(
        broad_mutation,
        glyph_patch_type,
        render_completed=True,
        fit_status="ok",
    )
    broad_glyph_mask = np.full(original.shape[:2], 255, dtype=np.uint8)
    small_mutation = _mutation(
        original,
        owner_bbox_page=(0, 0, original.shape[1], original.shape[0]),
    )
    broad_glyph = _glyph_patch_for_mask(
        small_mutation,
        glyph_patch_type,
        broad_glyph_mask,
    )

    cleanup_commit = apply_atomic(original, broad_mutation, small_glyph)
    render_commit = apply_atomic(original, small_mutation, broad_glyph)

    assert cleanup_commit.committed is False
    assert render_commit.committed is False
    assert "page-wide" in cleanup_commit.reason
    assert "overbroad" in render_commit.reason
    np.testing.assert_array_equal(cleanup_commit.result_rgb, original)
    np.testing.assert_array_equal(render_commit.result_rgb, original)


def test_atomic_owner_execution_rejects_dense_solid_masks_below_page_threshold() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = np.full((100, 100, 3), 210, dtype=np.uint8)
    dense_cleanup = _mutation(
        original,
        action_box=(20, 20, 50, 50),
        owner_bbox_page=(10, 10, 60, 60),
    )
    dense_cleanup = replace(dense_cleanup, component_geometry_verified=False)
    cleanup_glyph_mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cleanup_glyph_mask[24:26, 24:28] = 255
    cleanup_glyph = _glyph_patch_for_mask(
        dense_cleanup,
        glyph_patch_type,
        cleanup_glyph_mask,
    )
    small_cleanup = _mutation(
        original,
        action_box=(4, 4, 8, 8),
        owner_bbox_page=(2, 2, 90, 90),
    )
    dense_glyph_mask = np.zeros(original.shape[:2], dtype=np.uint8)
    dense_glyph_mask[20:60, 20:60] = 255
    dense_glyph = _glyph_patch_for_mask(
        small_cleanup,
        glyph_patch_type,
        dense_glyph_mask,
    )

    cleanup_commit = apply_atomic(original, dense_cleanup, cleanup_glyph)
    render_commit = apply_atomic(original, small_cleanup, dense_glyph)

    assert cleanup_commit.committed is False
    assert render_commit.committed is False
    assert "component geometry" in cleanup_commit.reason
    assert "overbroad" in render_commit.reason


def test_atomic_owner_execution_accepts_dense_mask_from_verified_component_geometry() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = np.full((100, 100, 3), 210, dtype=np.uint8)
    verified_cleanup = replace(
        _mutation(
            original,
            action_box=(20, 20, 50, 50),
            owner_bbox_page=(10, 10, 60, 60),
        ),
        component_geometry_verified=True,
    )
    glyph_mask = np.zeros(original.shape[:2], dtype=np.uint8)
    glyph_mask[24:26, 24:28] = 255
    glyph_patch = _glyph_patch_for_mask(
        verified_cleanup,
        glyph_patch_type,
        glyph_mask,
    )

    commit = apply_atomic(original, verified_cleanup, glyph_patch)

    assert commit.committed is True
    assert commit.review_required is False


def test_protected_art_mask_hash_is_immutable_across_execution_chain() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    mutation = _mutation(original)
    protected = np.zeros(original.shape[:2], dtype=np.uint8)
    protected[8, 8] = 255
    mutation = replace(
        mutation,
        protected_art_mask=protected,
        protected_art_mask_sha256=_array_sha256(protected),
    )
    stripped_protection = np.zeros(original.shape[:2], dtype=np.uint8)
    tampered = replace(mutation, protected_art_mask=stripped_protection)
    glyph_mask = np.zeros(original.shape[:2], dtype=np.uint8)
    glyph_mask[8, 8] = 255
    glyph_patch = _glyph_patch_for_mask(tampered, glyph_patch_type, glyph_mask)

    commit = apply_atomic(original, tampered, glyph_patch)

    assert commit.committed is False
    assert commit.review_required is True
    assert "protected" in commit.reason
    np.testing.assert_array_equal(commit.result_rgb, original)


def test_cleanup_execution_metadata_must_be_canonical() -> None:
    apply_atomic, glyph_patch_type, _ = _atomic_api()
    original = _page()
    valid = _mutation(original)
    invalid_mutations = (
        replace(valid, engine=""),
        replace(valid, engine_crop_bbox_page=(False, 0, 16, 12)),
        replace(valid, engine_crop_bbox_page=(0, 0, 999, 999)),
        replace(valid, mask_pixels=float(valid.mask_pixels)),
    )

    for mutation in invalid_mutations:
        glyph_patch = _glyph_patch(
            mutation,
            glyph_patch_type,
            render_completed=True,
            fit_status="ok",
        )
        commit = apply_atomic(original, mutation, glyph_patch)
        assert commit.committed is False
        assert commit.review_required is True
        np.testing.assert_array_equal(commit.result_rgb, original)
