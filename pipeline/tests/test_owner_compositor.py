"""TDD contracts for the pure page-space owner compositor."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compositor.owner_compositor import (  # noqa: E402
    OwnerCompositionError,
    compose_page,
)
from ownership.delivery import (  # noqa: E402
    GlyphRunObservation,
    build_owner_text_delivery_contract,
    seal_owner_text_execution_authority,
)
from ownership.model import (  # noqa: E402
    OwnerGlyphPatch,
    OwnerGraph,
    OwnerMutation,
    OwnerProjection,
    SourceTextComponent,
    TextObservation,
    TextOwner,
    owner_residual_evidence_sha256,
)
from ownership.render_geometry import build_owner_render_geometry  # noqa: E402
from typesetter.owner_render_quality import OwnerRenderQuality  # noqa: E402
from style_v2_fixtures import valid_owner_style_raster_contract  # noqa: E402


PAGE_ID = "page_001"
PAGE_SHAPE = (18, 24, 3)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(b"traduzai.ndarray.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _polygon_sha256(points: tuple[tuple[int, int], ...]) -> str:
    return sha256(json.dumps(points, separators=(",", ":")).encode("utf-8")).hexdigest()


def _original() -> np.ndarray:
    page = np.full(PAGE_SHAPE, 220, dtype=np.uint8)
    page[:, :, 1] = 218
    return page


def _box_mask(
    box: tuple[int, int, int, int],
    *,
    shape: tuple[int, int] = PAGE_SHAPE[:2],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 255
    return mask


def _component_hash(owner_id: str) -> str:
    return sha256(f"component:{owner_id}".encode("utf-8")).hexdigest()


def _render_geometry(owner_id: str, protected_sha256: str):
    bbox = (2, 2, 12, 12)
    polygon = ((2, 2), (12, 2), (12, 12), (2, 12))
    graph = OwnerGraph(
        2,
        PAGE_ID,
        [SourceTextComponent(f"component_{owner_id}", PAGE_ID, bbox, polygon, ("fixture",))],
        [TextObservation(f"observation_{owner_id}", PAGE_ID, (f"component_{owner_id}",), "SOURCE", 1.0, "fixture", bbox, polygons_page=(polygon,))],
        [TextOwner(owner_id, PAGE_ID, [f"component_{owner_id}"], [f"observation_{owner_id}"], [f"observation_{owner_id}"], "freeform_sfx", "SOURCE", "ALVO", "owned", "translated", "translate_inpaint_render", f"tile_{owner_id}")],
        [OwnerProjection(owner_id, f"tile_{owner_id}", "executor", (0, 0, PAGE_SHAPE[1], PAGE_SHAPE[0]), (0, 0, PAGE_SHAPE[1], PAGE_SHAPE[0]), (0, 0))],
    )
    return build_owner_render_geometry(
        graph, owner_id, page_width=PAGE_SHAPE[1], page_height=PAGE_SHAPE[0],
        protected_art_mask_sha256=protected_sha256,
    )


def _action_mask_ref(owner_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner_id).strip("._")
    identity_hash = sha256(owner_id.encode("utf-8")).hexdigest()[:12]
    return (
        f"owner_masks/{safe}--{identity_hash}/tile_{safe}/action_mask.png"
    )


def _mutation(
    original: np.ndarray,
    *,
    owner_id: str = "owner_a",
    box: tuple[int, int, int, int] = (2, 2, 7, 7),
    color: tuple[int, int, int] = (40, 52, 64),
    protected_art_mask: np.ndarray | None = None,
    projection_role: str = "executor",
    color_space: str = "RGB",
) -> OwnerMutation:
    action_mask = _box_mask(box, shape=original.shape[:2])
    result = original.copy()
    result[action_mask > 0] = color
    changed = np.any(result != original, axis=2)
    changed_mask = np.where(changed, 255, 0).astype(np.uint8)
    protected = (
        np.zeros(original.shape[:2], dtype=np.uint8)
        if protected_art_mask is None
        else protected_art_mask.copy()
    )
    before_sha256 = _array_sha256(original)
    after_sha256 = _array_sha256(result)
    action_sha256 = _array_sha256(action_mask)
    protected_sha256 = _array_sha256(protected)
    render_geometry = _render_geometry(owner_id, protected_sha256)
    component_sha256 = _component_hash(owner_id)
    residual_evidence_sha256 = owner_residual_evidence_sha256(
        owner_id=owner_id,
        page_id=PAGE_ID,
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        action_mask_sha256=action_sha256,
        protected_art_mask_sha256=protected_sha256,
        component_geometry_sha256=component_sha256,
        residual_score=0.0,
        residual_threshold=0.01,
        residual_method="fixture_residual_v1",
        residual_flags=(),
    )
    authority = seal_owner_text_execution_authority(
        owner_id=owner_id,
        page_id=PAGE_ID,
        source_payload="SOURCE",
        translated_payload="ALVO",
        normalized_chunks=["ALVO"],
    )
    return OwnerMutation(
        owner_id=owner_id,
        page_id=PAGE_ID,
        coordinate_space="logical_page",
        action_mask_ref=_action_mask_ref(owner_id),
        result_rgb=result,
        action_mask=action_mask,
        protected_art_mask=protected,
        changed_mask=changed_mask,
        engine="test_engine",
        mask_pixels=int(np.count_nonzero(action_mask)),
        changed_pixels=int(np.count_nonzero(changed_mask)),
        changed_outside_owner_pixels=0,
        protected_art_changed_pixels=int(
            np.count_nonzero(changed & (protected > 0))
        ),
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        action_mask_sha256=action_sha256,
        changed_mask_sha256=_array_sha256(changed_mask),
        engine_crop_bbox_page=(0, 0, original.shape[1], original.shape[0]),
        owner_bbox_page=(0, 0, original.shape[1], original.shape[0]),
        component_geometry_sha256=component_sha256,
        owner_render_geometry_sha256=render_geometry.geometry_sha256,
        protected_art_mask_sha256=protected_sha256,
        residual_score=0.0,
        residual_verified=True,
        residual_threshold=0.01,
        residual_method="fixture_residual_v1",
        residual_evidence_sha256=residual_evidence_sha256,
        component_geometry_verified=True,
        execution_tile_id=f"tile_{owner_id}",
        projection_role=projection_role,
        color_space=color_space,
        text_execution_authority_sha256=authority.authority_sha256,
        text_execution_authority=authority,
    )


def _glyph_patch(
    original: np.ndarray,
    *,
    owner_id: str = "owner_a",
    mutation: OwnerMutation | None = None,
    box: tuple[int, int, int, int] = (3, 3, 6, 6),
    color: tuple[int, int, int] = (8, 12, 18),
    projection_role: str = "executor",
    color_space: str = "RGB",
) -> OwnerGlyphPatch:
    baseline = original if mutation is None else np.asarray(mutation.result_rgb)
    glyph_mask = _box_mask(box, shape=original.shape[:2])
    result = baseline.copy()
    result[glyph_mask > 0] = color
    safe_polygon = (
        (0, 0),
        (original.shape[1] - 1, 0),
        (original.shape[1] - 1, original.shape[0] - 1),
        (0, original.shape[0] - 1),
    )
    x1, y1, x2, y2 = box
    protected_sha256 = (
        str(mutation.protected_art_mask_sha256)
        if mutation is not None
        else _array_sha256(np.zeros(original.shape[:2], dtype=np.uint8))
    )
    render_geometry = _render_geometry(owner_id, protected_sha256)
    authority = (
        mutation.text_execution_authority
        if mutation is not None
        else seal_owner_text_execution_authority(
            owner_id=owner_id,
            page_id=PAGE_ID,
            source_payload="SOURCE",
            translated_payload="ALVO",
            normalized_chunks=["ALVO"],
        )
    )
    style_contract = valid_owner_style_raster_contract(
        owner_id=owner_id,
        page_id=PAGE_ID,
        before=baseline,
        result=result,
        glyph_mask=glyph_mask,
        component_geometry_sha256=(
            mutation.component_geometry_sha256
            if mutation is not None
            else _component_hash(owner_id)
        ),
    )
    run = GlyphRunObservation.build(
        text=authority.translated_payload,
        font_identity="fixture-font",
        span_index=0,
    )
    delivery = build_owner_text_delivery_contract(
        execution_authority=authority,
        layout_payload=authority.translated_payload,
        rendered_lines=[authority.translated_payload],
        rendered_glyph_runs=[run],
        glyph_core_mask=glyph_mask,
        glyph_span_core_masks=[glyph_mask],
        rendered_patch_sha256=style_contract.rendered_patch_sha256,
    )
    return OwnerGlyphPatch(
        owner_id=owner_id,
        page_id=PAGE_ID,
        coordinate_space="logical_page",
        result_rgb=result,
        glyph_mask=glyph_mask,
        glyph_bbox_page=(x1, y1, x2, y2),
        render_completed=True,
        fit_status="ok",
        before_sha256=_array_sha256(baseline),
        after_sha256=_array_sha256(result),
        glyph_mask_sha256=_array_sha256(glyph_mask),
        changed_outside_glyph_mask_pixels=0,
        render_safe_polygon_page=safe_polygon,
        render_safe_polygon_sha256=_polygon_sha256(safe_polygon),
        component_geometry_sha256=(
            mutation.component_geometry_sha256
            if mutation is not None
            else _component_hash(owner_id)
        ),
        render_quality_contract=OwnerRenderQuality(
            schema_version=1,
            status="ok",
            font_size_final=14,
            minimum_legible_font_px=12,
            source_ink_height_px=None,
            render_ink_height_px=max(1, y2 - y1),
            source_x_height_px=None,
            render_x_height_px=float(max(1, y2 - y1)) * 0.70,
            source_scale_ratio=None,
            x_height_ratio=None,
            rendered_line_core_heights_px=(max(1, y2 - y1),),
            safe_height_occupancy=float(y2 - y1) / float(original.shape[0]),
            safe_area_occupancy=float(np.count_nonzero(glyph_mask)) / float(glyph_mask.size),
            wrapped_line_count=1,
            containment_status="ok",
            outside_safe_pixels=0,
            page_width=original.shape[1],
            page_height=original.shape[0],
            reasons=(),
        ),
        style_raster_contract=style_contract,
        owner_render_geometry_sha256=render_geometry.geometry_sha256,
        owner_render_geometry=render_geometry,
        execution_tile_id=f"tile_{owner_id}",
        projection_role=projection_role,
        color_space=color_space,
        glyph_core_mask=glyph_mask,
        paint_mask=glyph_mask,
        glyph_span_core_masks=(glyph_mask,),
        glyph_span_runs=(run,),
        glyph_core_mask_sha256=_array_sha256(glyph_mask),
        paint_mask_sha256=_array_sha256(glyph_mask),
        text_execution_authority_sha256=authority.authority_sha256,
        text_execution_authority=authority,
        delivery_contract=delivery,
    )


def _empty_protected(original: np.ndarray) -> np.ndarray:
    return np.zeros(original.shape[:2], dtype=np.uint8)


def test_owner_glyph_patch_requires_style_raster_contract() -> None:
    original = np.full(PAGE_SHAPE, 230, dtype=np.uint8)
    valid_patch = _glyph_patch(original)

    with pytest.raises(TypeError, match="style raster contract"):
        replace(valid_patch, style_raster_contract=None)


def test_owner_glyph_patch_requires_render_quality_contract() -> None:
    patch = _glyph_patch(_original())

    assert hasattr(patch, "render_quality_contract")
    assert patch.render_quality_contract.status == "ok"


def test_owner_compositor_rejects_legacy_page_coordinate_alias() -> None:
    original = _original()
    mutation = _mutation(original)
    object.__setattr__(mutation, "coordinate_space", "page")

    with pytest.raises(OwnerCompositionError, match="legacy coordinate space"):
        compose_page(original, [mutation], [], _empty_protected(original))


def test_published_framed_composition_deserialize_requires_geometry_and_shapes() -> None:
    from strip.page_surface_geometry import PageSurfaceGeometry
    from ownership.model import PageCompositionResult

    geometry = PageSurfaceGeometry.build(
        logical_width=24, logical_height=18, frame_width=30, frame_height=18,
        content_origin_xy=(3, 0),
    )
    internal = compose_page(_original(), [], [], _empty_protected(_original()))
    assert internal.coordinate_space == "logical_page"
    payload = PageCompositionResult(
        final_rgb=geometry.logical_array_to_frame(internal.final_rgb, fill_value=0),
        cleanup_owner_map=geometry.logical_array_to_frame(internal.cleanup_owner_map, fill_value=""),
        glyph_owner_map=geometry.logical_array_to_frame(internal.glyph_owner_map, fill_value=""),
        conflicts=(), write_counts={}, sha256="a" * 64, page_id=PAGE_ID,
        coordinate_space="framed_page",
        page_surface_geometry_sha256=geometry.geometry_sha256,
        page_surface_geometry=geometry,
    ).to_dict()

    assert PageCompositionResult.from_dict(payload, enforce=True).coordinate_space == "framed_page"
    with pytest.raises(ValueError, match="page surface geometry"):
        PageCompositionResult.from_dict({**payload, "page_surface_geometry": None}, enforce=True)
    with pytest.raises(ValueError, match="geometry hash"):
        PageCompositionResult.from_dict({**payload, "page_surface_geometry_sha256": "f" * 64}, enforce=True)


def test_atomic_owner_commit_rejects_missing_render_quality_contract() -> None:
    from strip.process_bands import apply_atomic_owner_execution

    original = _original()
    mutation = _mutation(original)
    patch = _glyph_patch(original, mutation=mutation)
    object.__setattr__(patch, "render_quality_contract", None)

    commit = apply_atomic_owner_execution(original, mutation, patch)

    assert commit.committed is False
    assert commit.review_required is True
    assert "render quality contract is missing" in commit.reason


def test_changed_pixels_must_be_subset_of_owner_action_mask() -> None:
    original = _original()
    mutation = _mutation(original)
    forged_result = np.asarray(mutation.result_rgb).copy()
    forged_result[12, 18] = (1, 2, 3)
    forged_changed = np.any(forged_result != original, axis=2)
    forged_mask = np.where(forged_changed, 255, 0).astype(np.uint8)
    forged = replace(
        mutation,
        result_rgb=forged_result,
        changed_mask=forged_mask,
        changed_pixels=int(np.count_nonzero(forged_mask)),
        changed_outside_owner_pixels=1,
        after_sha256=_array_sha256(forged_result),
        changed_mask_sha256=_array_sha256(forged_mask),
    )

    with pytest.raises(OwnerCompositionError, match="outside.*action mask"):
        compose_page(original, [forged], [], _empty_protected(original))


def test_action_mask_cannot_touch_protected_art() -> None:
    original = _original()
    protected = _box_mask((4, 4, 9, 9))
    mutation = _mutation(original, box=(2, 2, 7, 7))

    with pytest.raises(OwnerCompositionError, match="protected art"):
        compose_page(original, [mutation], [], protected)


@pytest.mark.parametrize("artifact_kind", ["mutation", "glyph"])
def test_context_only_tile_cannot_write_pixels(artifact_kind: str) -> None:
    original = _original()
    mutation = _mutation(original, projection_role="context_only")
    glyph = _glyph_patch(
        original,
        mutation=None,
        projection_role="context_only",
    )
    mutations = [mutation] if artifact_kind == "mutation" else []
    glyphs = [glyph] if artifact_kind == "glyph" else []

    with pytest.raises(OwnerCompositionError, match="context_only|executor"):
        compose_page(original, mutations, glyphs, _empty_protected(original))


def test_conflicting_owner_mutations_fail_closed() -> None:
    original = _original()
    left = _mutation(original, owner_id="owner_a", box=(2, 2, 8, 8))
    right = _mutation(
        original,
        owner_id="owner_b",
        box=(6, 6, 12, 12),
        color=(80, 90, 100),
    )

    result = compose_page(original, [left, right], [], _empty_protected(original))

    assert result.committed is False
    assert any(conflict.code == "owner_pixel_conflict" for conflict in result.conflicts)
    np.testing.assert_array_equal(result.final_rgb, original)
    assert not np.any(result.cleanup_owner_map != "")
    assert not np.any(result.glyph_owner_map != "")


def test_identical_duplicate_patch_for_same_owner_is_applied_once() -> None:
    original = _original()
    mutation = _mutation(original)
    patch = _glyph_patch(original, mutation=mutation)

    result = compose_page(
        original,
        [mutation, mutation],
        [patch, patch],
        _empty_protected(original),
    )

    assert result.committed is True
    assert not result.conflicts
    assert result.write_counts["cleanup_pixels"] == mutation.changed_pixels
    assert result.write_counts["glyph_pixels"] == int(
        np.count_nonzero(patch.glyph_mask)
    )


def test_divergent_duplicate_patch_for_same_owner_blocks() -> None:
    original = _original()
    mutation = _mutation(original)
    first = _glyph_patch(original, mutation=mutation)
    second = _glyph_patch(
        original,
        mutation=mutation,
        color=(120, 10, 30),
    )

    result = compose_page(
        original,
        [mutation],
        [first, second],
        _empty_protected(original),
    )

    assert result.committed is False
    assert any(
        conflict.code == "divergent_duplicate_glyph_patch"
        for conflict in result.conflicts
    )
    np.testing.assert_array_equal(result.final_rgb, original)


def test_divergent_duplicate_mutation_blocks_before_glyph_chain_resolution() -> None:
    original = _original()
    first = _mutation(original, color=(40, 50, 60))
    second = _mutation(original, color=(90, 100, 110))
    glyph = _glyph_patch(original, mutation=first)

    for mutations in ([first, second], [second, first]):
        result = compose_page(
            original,
            mutations,
            [glyph],
            _empty_protected(original),
        )
        assert result.committed is False
        assert any(
            conflict.code == "divergent_duplicate_mutation"
            for conflict in result.conflicts
        )
        np.testing.assert_array_equal(result.final_rgb, original)


def test_all_inpaints_precede_all_glyph_patches() -> None:
    original = _original()
    mutation = _mutation(original, box=(2, 2, 9, 9), color=(40, 50, 60))
    patch = _glyph_patch(
        original,
        mutation=mutation,
        box=(4, 4, 7, 7),
        color=(5, 7, 9),
    )

    result = compose_page(original, [mutation], [patch], _empty_protected(original))

    assert result.committed is True
    np.testing.assert_array_equal(result.final_rgb[5, 5], np.array([5, 7, 9]))
    np.testing.assert_array_equal(result.final_rgb[2, 2], np.array([40, 50, 60]))
    assert result.cleanup_owner_map[2, 2] == "owner_a"
    assert result.glyph_owner_map[5, 5] == "owner_a"


def test_glyph_only_and_mutation_only_owners_are_supported() -> None:
    original = _original()
    cleanup_only = _mutation(
        original,
        owner_id="owner_cleanup",
        box=(1, 1, 6, 6),
    )
    glyph_only = _glyph_patch(
        original,
        owner_id="owner_glyph",
        mutation=None,
        box=(14, 10, 19, 15),
    )

    result = compose_page(
        original,
        [cleanup_only],
        [glyph_only],
        _empty_protected(original),
    )

    assert result.committed is True
    assert result.cleanup_owner_map[2, 2] == "owner_cleanup"
    assert result.glyph_owner_map[11, 15] == "owner_glyph"


def test_glyph_mask_cannot_touch_global_protected_art() -> None:
    original = _original()
    glyph = _glyph_patch(original, mutation=None, box=(14, 10, 19, 15))
    protected = _box_mask((16, 12, 20, 16))

    with pytest.raises(OwnerCompositionError, match="protected art"):
        compose_page(original, [], [glyph], protected)


def test_diff_from_original_is_subset_of_owned_masks() -> None:
    original = _original()
    first = _mutation(original, owner_id="owner_a", box=(1, 1, 5, 5))
    second = _mutation(
        original,
        owner_id="owner_b",
        box=(12, 10, 18, 15),
        color=(70, 80, 90),
    )
    glyph = _glyph_patch(
        original,
        owner_id="owner_b",
        mutation=second,
        box=(13, 11, 16, 14),
    )

    result = compose_page(
        original,
        [first, second],
        [glyph],
        _empty_protected(original),
    )
    changed = np.any(result.final_rgb != original, axis=2)
    owned = (
        (np.asarray(first.action_mask) > 0)
        | (np.asarray(second.action_mask) > 0)
        | (np.asarray(glyph.glyph_mask) > 0)
    )

    assert result.committed is True
    assert not np.any(changed & ~owned)


def test_changed_pixels_may_be_strict_subset_of_effect_paint_mask() -> None:
    original = _original()
    mutation = _mutation(original)
    patch = _glyph_patch(original, mutation=mutation, box=(5, 5, 8, 8))
    paint = _box_mask((4, 4, 9, 9), shape=original.shape[:2])
    style_contract = valid_owner_style_raster_contract(
        owner_id=patch.owner_id,
        page_id=patch.page_id,
        before=np.asarray(mutation.result_rgb),
        result=np.asarray(patch.result_rgb),
        glyph_mask=paint,
        component_geometry_sha256=patch.component_geometry_sha256,
    )
    delivery = build_owner_text_delivery_contract(
        execution_authority=patch.text_execution_authority,
        layout_payload="ALVO",
        rendered_lines=["ALVO"],
        rendered_glyph_runs=patch.glyph_span_runs,
        glyph_core_mask=patch.glyph_core_mask,
        glyph_span_core_masks=patch.glyph_span_core_masks,
        rendered_patch_sha256=style_contract.rendered_patch_sha256,
    )
    patch = replace(
        patch,
        glyph_mask=paint,
        paint_mask=paint,
        glyph_bbox_page=(4, 4, 9, 9),
        glyph_mask_sha256=_array_sha256(paint),
        paint_mask_sha256=_array_sha256(paint),
        style_raster_contract=style_contract,
        delivery_contract=delivery,
    )

    result = compose_page(original, [mutation], [patch], _empty_protected(original))

    assert result.committed is True
    actual_changed = np.any(
        np.asarray(patch.result_rgb) != np.asarray(mutation.result_rgb), axis=2
    )
    assert np.count_nonzero(actual_changed) < np.count_nonzero(paint)


def test_color_space_mismatch_is_rejected() -> None:
    original = _original()
    mutation = _mutation(original, color_space="BGR")

    with pytest.raises(OwnerCompositionError, match="RGB|color space"):
        compose_page(original, [mutation], [], _empty_protected(original))


def test_action_mask_ref_accepts_canonical_sanitized_owner_identity() -> None:
    original = _original()
    mutation = _mutation(original, owner_id="owner/a")

    result = compose_page(
        original,
        [mutation],
        [],
        _empty_protected(original),
    )

    assert result.committed is True


@pytest.mark.parametrize(
    "forged_ref",
    [
        "../owner_a/action_mask.png",
        "untrusted/owner_a/payload.bin",
        "owner_a",
    ],
)
def test_action_mask_ref_rejects_unbound_or_traversal_paths(
    forged_ref: str,
) -> None:
    original = _original()
    mutation = replace(_mutation(original), action_mask_ref=forged_ref)

    with pytest.raises(OwnerCompositionError, match="bound to owner_id"):
        compose_page(original, [mutation], [], _empty_protected(original))


def test_composition_hash_is_deterministic() -> None:
    original = _original()
    first = _mutation(original, owner_id="owner_a", box=(1, 1, 5, 5))
    second = _mutation(
        original,
        owner_id="owner_b",
        box=(12, 10, 18, 15),
        color=(70, 80, 90),
    )

    forward = compose_page(
        original,
        [first, second],
        [],
        _empty_protected(original),
    )
    reverse = compose_page(
        original,
        [second, first],
        [],
        _empty_protected(original),
    )

    assert forward.sha256 == reverse.sha256 == _array_sha256(forward.final_rgb)
    np.testing.assert_array_equal(forward.final_rgb, reverse.final_rgb)
