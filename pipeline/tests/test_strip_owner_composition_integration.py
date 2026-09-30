from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def _action_ref(owner_id: str) -> str:
    safe = owner_id.replace("/", "_")
    identity = sha256(owner_id.encode("utf-8")).hexdigest()[:12]
    return f"owner_masks/{safe}--{identity}/mask/action_mask.png"


def _render_geometry(owner_id: str, page_id: str, bbox, shape, protected_sha256):
    from ownership.hash_contract import sha256_text
    from ownership.model import OWNER_GRAPH_SCHEMA_VERSION, OwnerGraph, OwnerProjection, SourceTextComponent, TextObservation, TextOwner
    from ownership.render_geometry import build_owner_render_geometry

    height, width = shape
    component_id = f"component-{owner_id}"
    observation_id = f"observation-{owner_id}"
    x1, y1, x2, y2 = bbox
    polygon = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    graph = OwnerGraph(
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id=page_id,
        components=[SourceTextComponent(component_id, page_id, bbox, polygon, ("fixture",))],
        observations=[
            TextObservation(
                observation_id,
                page_id,
                (component_id,),
                "SOURCE",
                1.0,
                "fixture",
                bbox,
                polygons_page=(polygon,),
                run_id="run-strip-composition",
                origin_execution_id="execution-strip-composition",
                invocation_id=f"invocation-{observation_id}",
                attempt_id=f"attempt-{observation_id}",
                provider_family="fixture",
                page_source_sha256="a" * 64,
                root_input_pixel_sha256="b" * 64,
                input_pixel_sha256="c" * 64,
                payload_sha256=sha256_text("SOURCE"),
            )
        ],
        owners=[TextOwner(owner_id, page_id, [component_id], [observation_id], [observation_id], "freeform_sfx", "SOURCE", "ALVO", "owned", "translated", "translate_inpaint_render", f"tile-{owner_id}")],
        projections=[OwnerProjection(owner_id, f"tile-{owner_id}", "executor", (0, 0, width, height), (0, 0, width, height), (0, 0))],
        run_id="run-strip-composition",
        origin_execution_id="execution-strip-composition",
        page_source_sha256="a" * 64,
    )
    return build_owner_render_geometry(graph, owner_id, page_width=width, page_height=height, protected_art_mask_sha256=protected_sha256)


def _mutation(
    original: np.ndarray,
    *,
    owner_id: str,
    page_id: str = "page_001",
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
):
    from ownership.delivery import seal_owner_text_execution_authority
    from ownership.model import OwnerMutation, owner_residual_evidence_sha256

    x1, y1, x2, y2 = bbox
    action = np.zeros(original.shape[:2], dtype=np.uint8)
    action[y1 + 2 : y2 - 2 : 3, x1 + 2 : x2 - 2] = 255
    result = original.copy()
    result[action > 0] = color
    changed = np.where(np.any(result != original, axis=2), 255, 0).astype(np.uint8)
    protected = np.zeros(original.shape[:2], dtype=np.uint8)
    before_sha256 = _array_sha256(original)
    after_sha256 = _array_sha256(result)
    action_sha256 = _array_sha256(action)
    protected_sha256 = _array_sha256(protected)
    component_sha256 = sha256(owner_id.encode("utf-8")).hexdigest()
    render_geometry = _render_geometry(owner_id, page_id, bbox, original.shape[:2], protected_sha256)
    residual_evidence_sha256 = owner_residual_evidence_sha256(
        owner_id=owner_id,
        page_id=page_id,
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
        owner_id=owner_id, page_id=page_id, source_payload="SOURCE",
        translated_payload="ALVO", normalized_chunks=["ALVO"],
    )
    return OwnerMutation(
        owner_id=owner_id,
        page_id=page_id,
        coordinate_space="logical_page",
        action_mask_ref=_action_ref(owner_id),
        result_rgb=result,
        action_mask=action,
        protected_art_mask=protected,
        changed_mask=changed,
        engine="test-engine",
        mask_pixels=int(np.count_nonzero(action)),
        changed_pixels=int(np.count_nonzero(changed)),
        changed_outside_owner_pixels=0,
        protected_art_changed_pixels=0,
        before_sha256=before_sha256,
        after_sha256=after_sha256,
        action_mask_sha256=action_sha256,
        changed_mask_sha256=_array_sha256(changed),
        engine_crop_bbox_page=bbox,
        owner_bbox_page=bbox,
        component_geometry_sha256=component_sha256,
        owner_render_geometry_sha256=render_geometry.geometry_sha256,
        protected_art_mask_sha256=protected_sha256,
        residual_score=0.0,
        residual_verified=True,
        residual_threshold=0.01,
        residual_method="fixture_residual_v1",
        residual_evidence_sha256=residual_evidence_sha256,
        component_geometry_verified=True,
        execution_tile_id=f"tile-{owner_id}",
        text_execution_authority_sha256=authority.authority_sha256,
        text_execution_authority=authority,
    )


def _chapter_fixture():
    from strip.process_bands import apply_atomic_owner_execution
    from strip.types import Band, VerticalStrip

    original = np.full((80, 100, 3), 240, dtype=np.uint8)
    strip = VerticalStrip(
        image=original.copy(),
        width=100,
        height=80,
        source_page_breaks=[0, 80],
        page_x_offsets=[0],
        source_page_widths=[100],
    )
    left = Band(y_top=0, y_bottom=55, original_slice=original[:55].copy())
    right = Band(y_top=25, y_bottom=80, original_slice=original[25:].copy())
    left_mutation = _mutation(
        original,
        owner_id="owner-left",
        bbox=(12, 18, 34, 38),
        color=(20, 80, 160),
    )
    right_mutation = _mutation(
        original,
        owner_id="owner-right",
        bbox=(62, 44, 88, 68),
        color=(180, 40, 30),
    )
    left.owner_execution_commits = [
        apply_atomic_owner_execution(
            original,
            left_mutation,
            _glyph_patch(left_mutation, bbox=(16, 21, 28, 27)),
        )
    ]
    right.owner_execution_commits = [
        apply_atomic_owner_execution(
            original,
            right_mutation,
            _glyph_patch(right_mutation, bbox=(68, 50, 80, 56)),
        )
    ]
    return original, strip, [left, right]


def _glyph_patch(mutation, *, bbox=(16, 21, 28, 27)):
    import json
    from ownership.model import OwnerGlyphPatch
    from ownership.delivery import GlyphRunObservation, build_owner_text_delivery_contract
    from style_v2_fixtures import valid_owner_style_raster_contract
    from typesetter.owner_render_quality import OwnerRenderQuality

    before = np.asarray(mutation.result_rgb)
    x1, y1, x2, y2 = bbox
    result = before.copy()
    result[y1:y2, x1:x2] = (3, 3, 3)
    mask = np.zeros(before.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    polygon = (
        (max(0, x1 - 4), max(0, y1 - 4)),
        (min(before.shape[1] - 1, x2 + 4), max(0, y1 - 4)),
        (min(before.shape[1] - 1, x2 + 4), min(before.shape[0] - 1, y2 + 4)),
        (max(0, x1 - 4), min(before.shape[0] - 1, y2 + 4)),
    )
    render_geometry = _render_geometry(
        mutation.owner_id,
        mutation.page_id,
        mutation.owner_bbox_page,
        before.shape[:2],
        str(mutation.protected_art_mask_sha256),
    )
    style_contract = valid_owner_style_raster_contract(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        before=before,
        result=result,
        glyph_mask=mask,
        component_geometry_sha256=mutation.component_geometry_sha256,
    )
    authority = mutation.text_execution_authority
    run = GlyphRunObservation.build(text=authority.translated_payload, font_identity="fixture-font", span_index=0)
    delivery = build_owner_text_delivery_contract(
        execution_authority=authority, layout_payload=authority.translated_payload,
        rendered_lines=[authority.translated_payload], rendered_glyph_runs=[run],
        glyph_core_mask=mask, glyph_span_core_masks=[mask],
        rendered_patch_sha256=style_contract.rendered_patch_sha256,
    )
    return OwnerGlyphPatch(
        owner_id=mutation.owner_id,
        page_id=mutation.page_id,
        coordinate_space="logical_page",
        result_rgb=result,
        glyph_mask=mask,
        glyph_bbox_page=bbox,
        render_completed=True,
        fit_status="ok",
        before_sha256=mutation.after_sha256,
        after_sha256=_array_sha256(result),
        glyph_mask_sha256=_array_sha256(mask),
        changed_outside_glyph_mask_pixels=0,
        render_safe_polygon_page=polygon,
        render_safe_polygon_sha256=sha256(
            json.dumps(polygon, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        component_geometry_sha256=mutation.component_geometry_sha256,
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
            safe_height_occupancy=0.25,
            safe_area_occupancy=0.10,
            wrapped_line_count=1,
            containment_status="ok",
            outside_safe_pixels=0,
            page_width=result.shape[1],
            page_height=result.shape[0],
            reasons=(),
        ),
        style_raster_contract=style_contract,
        owner_render_geometry_sha256=render_geometry.geometry_sha256,
        owner_render_geometry=render_geometry,
        execution_tile_id=mutation.execution_tile_id,
        glyph_core_mask=mask,
        paint_mask=mask,
        glyph_span_core_masks=(mask,),
        glyph_span_runs=(run,),
        glyph_core_mask_sha256=_array_sha256(mask),
        paint_mask_sha256=_array_sha256(mask),
        text_execution_authority_sha256=authority.authority_sha256,
        text_execution_authority=authority,
        delivery_contract=delivery,
    )


def test_owner_stage_outputs_preserve_executor_mutation_and_glyph_patch():
    from strip.process_bands import _run_inpaint_stage, _run_typeset_stage
    from strip.types import Band

    original = np.full((80, 100, 3), 240, dtype=np.uint8)
    mutation = _mutation(
        original,
        owner_id="owner-left",
        bbox=(12, 18, 34, 38),
        color=(20, 80, 160),
    )
    glyph = _glyph_patch(mutation)
    band = Band(y_top=0, y_bottom=80, strip_slice=original, original_slice=original)

    class Inpainter:
        @staticmethod
        def inpaint_band_image(_image, _page):
            return mutation

    class Typesetter:
        @staticmethod
        def render_band_image(_image, _page):
            return glyph

    translated_page = {"texts": []}
    inpaint_stage = _run_inpaint_stage(
        band,
        inpainter=Inpainter(),
        translated_page=translated_page,
    )
    typeset_stage = _run_typeset_stage(
        inpaint_stage.to_image(),
        typesetter=Typesetter(),
        translated_page=translated_page,
        owner_mutation=mutation,
    )

    assert inpaint_stage.owner_mutation is mutation
    assert typeset_stage.owner_glyph_patch is glyph
    np.testing.assert_array_equal(inpaint_stage.to_image(), mutation.result_rgb)
    np.testing.assert_array_equal(typeset_stage.to_image(), glyph.result_rgb)


def test_owner_composition_rejects_raw_artifacts_that_bypass_atomic_commit():
    from compositor.owner_compositor import OwnerCompositionError
    from strip.run import _compose_owner_output_pages
    from strip.types import Band, VerticalStrip

    original = np.full((40, 50, 3), 240, dtype=np.uint8)
    strip = VerticalStrip(
        image=original.copy(),
        width=50,
        height=40,
        source_page_breaks=[0, 40],
        page_x_offsets=[0],
        source_page_widths=[50],
    )
    band = Band(y_top=0, y_bottom=40, original_slice=original.copy())
    band.owner_mutations = [
        _mutation(
            original,
            owner_id="owner-raw",
            bbox=(8, 8, 20, 20),
            color=(10, 20, 30),
        )
    ]

    try:
        _compose_owner_output_pages(
            original_strip_image=original,
            strip=strip,
            bands=[band],
            balloons=[],
            target_count=1,
        )
    except OwnerCompositionError as exc:
        assert "atomic" in str(exc)
    else:
        raise AssertionError("raw owner mutation bypassed atomic execution")


def test_owner_typeset_patch_without_cleanup_mutation_fails_closed():
    from strip.process_bands import _run_typeset_stage

    original = np.full((80, 100, 3), 240, dtype=np.uint8)
    mutation = _mutation(
        original,
        owner_id="owner-left",
        bbox=(12, 18, 34, 38),
        color=(20, 80, 160),
    )
    glyph = _glyph_patch(mutation)

    class Typesetter:
        @staticmethod
        def render_band_image(_image, _page):
            return glyph

    try:
        _run_typeset_stage(
            original,
            typesetter=Typesetter(),
            translated_page={"texts": []},
        )
    except ValueError as exc:
        assert "cleanup mutation" in str(exc)
    else:
        raise AssertionError("owner glyph patch was accepted without cleanup mutation")


def test_owner_final_page_binding_preserves_composed_bytes_without_late_clamp():
    from strip.run import _bind_owner_final_page_images
    from strip.page_surface_geometry import PageSurfaceGeometry
    from strip.types import OutputPage

    final = np.full((30, 40, 3), [17, 91, 203], dtype=np.uint8)
    clean = np.full((30, 40, 3), [210, 220, 230], dtype=np.uint8)
    original = np.full((30, 40, 3), 240, dtype=np.uint8)
    geometry = PageSurfaceGeometry.build(logical_width=40, logical_height=30, frame_width=40, frame_height=30, content_origin_xy=(0, 0))
    page = OutputPage(y_top=0, y_bottom=30, image=final.copy(), page_surface_geometry=geometry)
    clean_page = OutputPage(y_top=0, y_bottom=30, image=clean.copy(), page_surface_geometry=geometry)
    original_page = OutputPage(y_top=0, y_bottom=30, image=original.copy(), page_surface_geometry=geometry)

    _bind_owner_final_page_images(page, original_page, clean_page)

    np.testing.assert_array_equal(page.image, final)
    np.testing.assert_array_equal(page.inpainted_image, clean)
    np.testing.assert_array_equal(page.original_image, original)


def test_owner_finalization_passes_bound_frame_geometry_to_terminal_gate(tmp_path):
    from strip.run import _finalize_bound_owner_output_page
    from strip.page_surface_geometry import PageSurfaceGeometry
    from strip.types import OutputPage

    geometry = PageSurfaceGeometry.build(
        logical_width=40,
        logical_height=30,
        frame_width=70,
        frame_height=30,
        content_origin_xy=(15, 0),
    )
    page = OutputPage(
        y_top=0,
        y_bottom=30,
        image=np.full((30, 70, 3), 240, dtype=np.uint8),
        original_image=np.full((30, 70, 3), 235, dtype=np.uint8),
        inpainted_image=np.full((30, 70, 3), 245, dtype=np.uint8),
        page_surface_geometry=geometry,
    )
    captured = {}

    def finalizer(result, **kwargs):
        captured.update(kwargs)
        return result, "evidence-ref"

    result, evidence_ref = _finalize_bound_owner_output_page(
        page,
        "candidate-result",
        private_execution_root=tmp_path,
        detector=object(),
        runtime=object(),
        source_language="en",
        page_number=4,
        finalizer=finalizer,
    )

    assert result == "candidate-result"
    assert evidence_ref == "evidence-ref"
    assert captured["page_surface_geometry"] is geometry
    np.testing.assert_array_equal(captured["original_pixels"], page.original_image)


def test_style_replay_recovers_logical_pixels_from_canonical_framed_original():
    import strip.run as run
    from ownership.hash_contract import canonical_page_sha256
    from types import SimpleNamespace

    logical = np.full((30, 40, 3), [17, 91, 203], dtype=np.uint8)
    framed = np.full((30, 70, 3), 255, dtype=np.uint8)
    framed[:, 15:55] = logical
    source_entry = SimpleNamespace(
        width=40,
        height=30,
        page_source_sha256=canonical_page_sha256(logical),
    )

    geometry, recovered = run._recover_style_replay_surface(
        framed,
        source_entry,
    )

    assert geometry.content_origin_xy == (15, 0)
    assert geometry.logical_width == 40
    assert geometry.frame_width == 70
    np.testing.assert_array_equal(recovered, logical)


def test_style_replay_strips_only_derived_visual_state_without_mutating_parent():
    import copy

    import strip.run as run

    parent = {
        "owner_id": "owner:style-replay",
        "translated": "TEXTO TRADUZIDO",
        "target_bbox": [11, 13, 47, 61],
        "translation_binding_sha256": "a" * 64,
        "render_layout_contract": {
            "schema_version": 1,
            "owner_id": "owner:style-replay",
            "fit_status": "fit",
            "font_size": 27,
            "lines": ["TEXTO TRADUZIDO"],
        },
        "owner_render_quality": {"font_size_final": 27, "status": "ok"},
        "owner_style_capture": {
            "owner_id": "owner:style-replay",
            "style_evidence_v2": {"source": "owner_source_pixels"},
        },
        "visual_profile_v2": {"owner_id": "owner:style-replay", "status": "applied"},
        "visual_profile_sha256": "b" * 64,
        "style_copy_status": "applied",
        "style_group_resolution_v3": {"group_id": "old-group"},
        "style_resolved_intent_v1": {"intent_sha256": "c" * 64},
        "style_v2_raster_contract": {"rendered_patch_sha256": "d" * 64},
    }
    parent_before = copy.deepcopy(parent)

    replay_record = run._strip_style_replay_derived_visual_state(parent)

    assert parent == parent_before
    assert replay_record["owner_id"] == parent["owner_id"]
    assert replay_record["translated"] == parent["translated"]
    assert replay_record["target_bbox"] == parent["target_bbox"]
    assert replay_record["translation_binding_sha256"] == parent["translation_binding_sha256"]
    assert replay_record["render_layout_contract"] == parent["render_layout_contract"]
    assert replay_record["style_replay_max_font_size_px"] == 27
    assert replay_record["owner_style_capture"] == parent["owner_style_capture"]
    for field_name in (
        "visual_profile_v2",
        "visual_profile_sha256",
        "style_copy_status",
        "style_group_resolution_v3",
        "style_resolved_intent_v1",
        "style_v2_raster_contract",
    ):
        assert field_name not in replay_record


def test_style_replay_renders_each_owner_with_explicit_projected_graph_and_composes_masks():
    from types import SimpleNamespace

    import strip.run as run

    cleanup = np.full((8, 10, 3), 245, dtype=np.uint8)
    records = [
        {"owner_id": "owner-a", "translated": "ALVO A"},
        {"owner_id": "owner-b", "translated": "ALVO B"},
    ]
    calls = []

    class Typesetter:
        def render_band_image(self, pixels, payload, *, owner_graph=None):
            owner_id = payload["texts"][0]["owner_id"]
            calls.append((owner_id, owner_graph, pixels.copy(), payload))
            paint_mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
            x = 2 if owner_id == "owner-a" else 7
            paint_mask[3, x] = 255
            rendered = pixels.copy()
            rendered[3, x] = (20, 40, 60) if owner_id == "owner-a" else (90, 110, 130)
            return SimpleNamespace(
                owner_id=owner_id,
                paint_mask=paint_mask,
                glyph_mask=paint_mask,
                result_rgb=rendered,
            )

    rendered, patches = run._render_style_replay_owner_patches(
        Typesetter(),
        cleanup,
        SimpleNamespace(page_id="page_001"),
        records,
        owner_ids=("owner-a", "owner-b"),
        graph_projector=lambda _graph, owner_id: f"single:{owner_id}",
    )

    assert [item[0] for item in calls] == ["owner-a", "owner-b"]
    assert [item[1] for item in calls] == ["single:owner-a", "single:owner-b"]
    assert all(np.array_equal(item[2], cleanup) for item in calls)
    assert all(len(item[3]["texts"]) == 1 for item in calls)
    assert set(patches) == {"owner-a", "owner-b"}
    np.testing.assert_array_equal(rendered[3, 2], [20, 40, 60])
    np.testing.assert_array_equal(rendered[3, 7], [90, 110, 130])


def test_style_replay_materialization_binds_new_execution_and_new_glyph_patch():
    from types import SimpleNamespace

    import strip.run as run
    from ownership.execution import canonical_glyph_patch_sha256

    pixels = np.full((6, 8, 3), 250, dtype=np.uint8)
    rendered = pixels.copy()
    rendered[2:4, 3:5] = 20
    paint = np.zeros(pixels.shape[:2], dtype=np.uint8)
    paint[2:4, 3:5] = 255
    patch = SimpleNamespace(
        owner_id="owner-a",
        page_id="page_001",
        coordinate_space="logical_page",
        before_sha256="1" * 64,
        after_sha256="2" * 64,
        glyph_mask_sha256="3" * 64,
        paint_mask_sha256="4" * 64,
        component_geometry_sha256="5" * 64,
        text_execution_authority_sha256="6" * 64,
        delivery_contract=SimpleNamespace(contract_sha256="7" * 64),
        style_raster_contract=SimpleNamespace(contract_sha256="8" * 64),
        paint_mask=paint,
        glyph_mask=paint,
        result_rgb=rendered,
    )
    binding = SimpleNamespace(
        run_id="content-run",
        page_id="page_001",
        page_source_sha256="9" * 64,
        owner_id="owner-a",
        translation_binding_sha256="a" * 64,
        source_payload_sha256="b" * 64,
        target_payload_sha256="c" * 64,
    )

    materialization = run._build_style_replay_target_materialization(
        binding,
        patch,
        execution_id="render-execution",
    )

    assert materialization.execution_id == "render-execution"
    assert materialization.run_id == "content-run"
    assert materialization.target_glyph_patch_sha256 == canonical_glyph_patch_sha256(patch)
    assert materialization.glyph_mask_sha256 == patch.glyph_mask_sha256
    assert materialization.base_pixel_sha256 == patch.before_sha256
    assert materialization.result_pixel_sha256 == patch.after_sha256


def test_style_replay_record_persists_fresh_raster_evidence_from_new_patch():
    from types import SimpleNamespace

    import strip.run as run

    record = {
        "owner_id": "owner-a",
        "translated": "ALVO",
        "render_layout_contract": {"owner_id": "owner-a"},
    }
    patch = SimpleNamespace(
        owner_id="owner-a",
        render_completed=True,
        fit_status="ok",
        glyph_core_mask_sha256="1" * 64,
        paint_mask_sha256="2" * 64,
        style_raster_contract=SimpleNamespace(to_dict=lambda: {"contract_sha256": "3" * 64}),
        render_quality_contract=SimpleNamespace(to_dict=lambda: {"quality_sha256": "4" * 64}),
        delivery_contract=SimpleNamespace(to_dict=lambda: {"contract_sha256": "5" * 64}),
    )

    refreshed = run._attach_style_replay_patch_evidence(record, patch)

    assert "style_v2_raster_contract" not in record
    assert refreshed["translated"] == "ALVO"
    assert refreshed["render_completed"] is True
    assert refreshed["fit_status"] == "ok"
    assert refreshed["glyph_core_mask_sha256"] == "1" * 64
    assert refreshed["paint_mask_sha256"] == "2" * 64
    assert refreshed["style_v2_raster_contract"]["contract_sha256"] == "3" * 64
    assert refreshed["owner_render_quality"]["quality_sha256"] == "4" * 64
    assert refreshed["owner_text_delivery_contract"]["contract_sha256"] == "5" * 64


def test_style_replay_aligns_owner_id_list_order_only_when_membership_is_identical():
    from types import SimpleNamespace

    import pytest
    import strip.run as run

    owner = SimpleNamespace(
        owner_id="owner-a",
        component_ids=["component-a"],
        observation_ids=["observation-a", "observation-b"],
        selected_observation_ids=["observation-a", "observation-b"],
    )
    record = {
        "owner_id": "owner-a",
        "component_ids": ["component-a"],
        "observation_ids": ["observation-b", "observation-a"],
        "selected_observation_ids": ["observation-b", "observation-a"],
        "translated": "ALVO",
    }

    aligned = run._align_style_replay_record_owner_lists(record, owner)

    assert record["observation_ids"] == ["observation-b", "observation-a"]
    assert aligned["observation_ids"] == owner.observation_ids
    assert aligned["selected_observation_ids"] == owner.selected_observation_ids
    assert aligned["translated"] == "ALVO"

    divergent = dict(record, observation_ids=["observation-a", "observation-extra"])
    with pytest.raises(ValueError, match="membership differs"):
        run._align_style_replay_record_owner_lists(divergent, owner)


def test_verified_output_adapter_exposes_authenticated_text_layer_snapshot():
    from types import SimpleNamespace

    from strip.page_pipeline import adapt_page_execution_result_to_output_page

    pixels = np.full((5, 7, 3), 230, dtype=np.uint8)
    layer = {
        "owner_id": "owner-a",
        "translation_binding_sha256": "a" * 64,
        "original": "SOURCE",
        "translated": "ALVO",
    }
    result = SimpleNamespace(
        final_page=None,
        request=SimpleNamespace(
            original_page=SimpleNamespace(mutable_attempt_copy=lambda: pixels.copy())
        ),
        owner_graph=SimpleNamespace(read=lambda: SimpleNamespace(to_dict=lambda: {"page_id": "page_001"})),
        page_id="page_001",
        translations=(
            SimpleNamespace(
                owner_id="owner-a",
                source_text="SOURCE",
                target_text="ALVO",
                target_locale="pt-BR",
            ),
        ),
        text_layers_view=SimpleNamespace(read=lambda: {"texts": [layer]}),
        page_composition=None,
    )

    output = adapt_page_execution_result_to_output_page(result)

    assert output.text_layers == {"texts": [layer]}
    assert output.text_layers["texts"][0]["translation_binding_sha256"] == "a" * 64


def test_style_replay_builds_runtime_composition_with_frame_geometry_and_owner_maps():
    from types import SimpleNamespace

    import strip.run as run
    from strip.page_surface_geometry import PageSurfaceGeometry

    geometry = PageSurfaceGeometry.build(
        logical_width=6,
        logical_height=4,
        frame_width=10,
        frame_height=4,
        content_origin_xy=(2, 0),
    )
    framed = np.full((4, 10, 3), 240, dtype=np.uint8)
    cleanup_mask = np.zeros((4, 6), dtype=np.uint8)
    cleanup_mask[1, 1] = 255
    paint_mask = np.zeros((4, 6), dtype=np.uint8)
    paint_mask[2, 4] = 255
    authority = SimpleNamespace(
        authority_sha256="a" * 64,
        to_dict=lambda: {"authority_sha256": "a" * 64},
    )
    delivery = SimpleNamespace(
        contract_sha256="b" * 64,
        to_dict=lambda: {"contract_sha256": "b" * 64},
    )
    patch = SimpleNamespace(
        owner_id="owner-a",
        paint_mask=paint_mask,
        glyph_mask=paint_mask,
        text_execution_authority=authority,
        delivery_contract=delivery,
    )

    composition = run._build_style_replay_runtime_composition(
        page_id="page_001",
        framed_original_rgb=framed.copy(),
        framed_rendered_rgb=framed,
        surface_geometry=geometry,
        cleanup_masks_by_owner={"owner-a": cleanup_mask},
        glyph_patches_by_owner={"owner-a": patch},
    )

    assert composition.committed is True
    assert composition.coordinate_space == "framed_page"
    assert composition.page_surface_geometry_sha256 == geometry.geometry_sha256
    assert composition.cleanup_owner_map[1, 3] == "owner-a"
    assert composition.glyph_owner_map[2, 6] == "owner-a"
    assert composition.owner_text_execution_authorities["owner-a"]["authority_sha256"] == "a" * 64
    assert composition.owner_text_delivery_contracts["owner-a"]["contract_sha256"] == "b" * 64

    empty = run._build_style_replay_runtime_composition(
        page_id="page_002",
        framed_original_rgb=framed.copy(),
        framed_rendered_rgb=framed.copy(),
        surface_geometry=geometry,
        cleanup_masks_by_owner={},
        glyph_patches_by_owner={},
    )
    assert empty.write_counts["owner_count"] == 0
    assert not np.any(empty.cleanup_owner_map != "")
    assert not np.any(empty.glyph_owner_map != "")


def test_style_replay_partitions_complete_cleanup_delta_without_gaps_or_overlap():
    import pytest
    import strip.run as run

    delta = np.zeros((6, 12), dtype=np.uint8)
    delta[2, 2] = 255
    delta[2, 9] = 255
    delta[4, 6] = 255  # feather pixel outside both component boxes

    masks = run._partition_style_replay_cleanup_delta(
        delta,
        {
            "owner-a": (1, 1, 4, 4),
            "owner-b": (8, 1, 11, 4),
        },
    )

    assert masks["owner-a"][2, 2] == 255
    assert masks["owner-b"][2, 9] == 255
    assert sum(mask[4, 6] > 0 for mask in masks.values()) == 1
    union = np.zeros(delta.shape, dtype=np.uint8)
    write_count = np.zeros(delta.shape, dtype=np.uint8)
    for mask in masks.values():
        union[mask > 0] = 255
        write_count[mask > 0] += 1
    np.testing.assert_array_equal(union, delta)
    assert np.max(write_count) == 1

    with pytest.raises(ValueError, match="lacks owners"):
        run._partition_style_replay_cleanup_delta(delta, {})


def test_run_chapter_uses_owner_compositor_as_only_final_pixel_authority(monkeypatch):
    import strip.run as run

    original, strip, bands = _chapter_fixture()
    monkeypatch.setattr(
        run,
        "_paste_band_attr_into_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy paste called")),
    )

    result = run._compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )

    assert result.compositions["page_001"].committed is True
    assert result.compositions["page_001"].page_id == "page_001"
    assert np.all(result.output_pages[0].image[20, 20] == (20, 80, 160))
    assert np.all(result.output_pages[0].image[61, 70] == (180, 40, 30))


def test_owner_composition_accepts_page_space_commits_without_band_transport():
    import strip.run as run

    original, strip, bands = _chapter_fixture()
    commits = tuple(
        commit
        for band in bands
        for commit in band.owner_execution_commits
    )
    for band in bands:
        band.owner_execution_commits = []

    result = run._compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
        owner_execution_commits=commits,
    )

    assert result.compositions["page_001"].committed is True
    assert np.all(result.output_pages[0].image[20, 20] == (20, 80, 160))
    assert np.all(result.output_pages[0].image[61, 70] == (180, 40, 30))


def test_owner_reassembly_preserves_source_pages_and_frames_pixel_maps():
    from compositor.owner_compositor import _array_sha256
    from strip.run import _compose_owner_output_pages
    from strip.types import Balloon, BBox, VerticalStrip

    original = np.full((80, 100, 3), 255, dtype=np.uint8)
    original[:40, 20:80] = 180
    original[40:, :] = 120
    strip = VerticalStrip(
        image=original.copy(),
        width=100,
        height=80,
        source_page_breaks=[0, 40, 80],
        page_x_offsets=[20, 0],
        source_page_widths=[60, 100],
    )
    crossing_false_positive = Balloon(
        strip_bbox=BBox(10, 30, 90, 50),
        confidence=0.9,
    )

    result = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=[],
        balloons=[crossing_false_positive],
        target_count=2,
    )

    assert [(page.y_top, page.y_bottom) for page in result.output_pages] == [
        (0, 40),
        (40, 80),
    ]
    assert [(page.y_top, page.y_bottom) for page in result.original_pages] == [
        (0, 40),
        (40, 80),
    ]
    for original_page, final_page in zip(result.original_pages, result.output_pages):
        assert original_page.image.shape == final_page.image.shape
        np.testing.assert_array_equal(
            original_page.image,
            original[original_page.y_top : original_page.y_bottom],
        )
    page_one = result.compositions["page_001"]
    assert page_one.page_id == "page_001"
    assert page_one.final_rgb.shape == (40, 100, 3)
    assert page_one.cleanup_owner_map.shape == (40, 100)
    assert page_one.glyph_owner_map.shape == (40, 100)
    assert page_one.sha256 == _array_sha256(page_one.final_rgb)


def test_run_chapter_output_is_identical_after_band_permutation():
    from strip.run import _compose_owner_output_pages

    original, strip, bands = _chapter_fixture()
    forward = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )
    reverse = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=list(reversed(bands)),
        balloons=[],
        target_count=1,
    )

    np.testing.assert_array_equal(forward.output_pages[0].image, reverse.output_pages[0].image)
    assert forward.compositions["page_001"].sha256 == reverse.compositions["page_001"].sha256


def test_changed_context_from_overlapping_tile_never_restores_source():
    from strip.run import _compose_owner_output_pages

    original, strip, bands = _chapter_fixture()
    bands[1].rendered_slice = original[25:].copy()
    bands[1].rendered_slice[0:20, 12:34] = original[25:45, 12:34]

    result = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )

    assert np.all(result.output_pages[0].image[29, 20] == (20, 80, 160))


def test_production_composition_succeeds_without_debug_directory(tmp_path, monkeypatch):
    from strip.run import _compose_owner_output_pages

    original, strip, bands = _chapter_fixture()
    monkeypatch.setenv("TRADUZAI_DEBUG_DIR", str(tmp_path / "does-not-exist"))

    result = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )

    assert result.output_pages[0].image.shape == original.shape
    assert not (tmp_path / "does-not-exist").exists()


def test_cross_band_gradient_owner_inpaints_once_in_page_space():
    from strip.process_bands import apply_atomic_owner_execution
    from strip.run import _compose_owner_output_pages
    from strip.types import Band, VerticalStrip

    yy, xx = np.indices((80, 100))
    original = np.stack(
        (
            120 + (xx // 3),
            90 + (yy // 2),
            70 + ((xx + yy) // 5),
        ),
        axis=2,
    ).astype(np.uint8)
    mutation = _mutation(
        original,
        owner_id="owner-cross-band",
        bbox=(18, 24, 72, 58),
        color=(112, 103, 91),
    )
    commit = apply_atomic_owner_execution(
        original,
        mutation,
        _glyph_patch(mutation, bbox=(30, 36, 58, 44)),
    )
    assert commit.committed is True
    bands = [
        Band(y_top=0, y_bottom=48, original_slice=original[:48].copy()),
        Band(y_top=32, y_bottom=80, original_slice=original[32:].copy()),
    ]
    for band in bands:
        band.owner_execution_commits = [commit]
    strip = VerticalStrip(
        image=original.copy(),
        width=100,
        height=80,
        source_page_breaks=[0, 80],
        page_x_offsets=[0],
        source_page_widths=[100],
    )

    result = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )

    composition = result.compositions["page_001"]
    assert composition.write_counts["cleanup_write:owner-cross-band"] == 1
    assert composition.write_counts["glyph_write:owner-cross-band"] == 1
    np.testing.assert_array_equal(result.output_pages[0].image, commit.result_rgb)


def test_projection_never_estimates_independent_background_color():
    from strip.process_bands import apply_atomic_owner_execution
    from strip.run import _compose_owner_output_pages
    from strip.types import Band, VerticalStrip

    yy, xx = np.indices((60, 84))
    original = np.stack((80 + xx, 100 + yy, 60 + ((xx + yy) // 2)), axis=2).astype(
        np.uint8
    )
    mutation = _mutation(
        original,
        owner_id="owner-projection-pure",
        bbox=(20, 18, 64, 46),
        color=(121, 117, 106),
    )
    commit = apply_atomic_owner_execution(
        original,
        mutation,
        _glyph_patch(mutation, bbox=(29, 27, 53, 35)),
    )
    bands = [
        Band(y_top=0, y_bottom=38, original_slice=original[:38].copy()),
        Band(y_top=22, y_bottom=60, original_slice=original[22:].copy()),
    ]
    bands[0].owner_execution_commits = [commit]
    bands[1].rendered_slice = np.full_like(original[22:], (1, 240, 3))
    strip = VerticalStrip(
        image=original.copy(),
        width=84,
        height=60,
        source_page_breaks=[0, 60],
        page_x_offsets=[0],
        source_page_widths=[84],
    )

    result = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )

    np.testing.assert_array_equal(result.output_pages[0].image, commit.result_rgb)


def test_owner_composition_does_not_promote_peer_local_protection_to_global_mask():
    """One owner's local protection cannot veto a peer's valid action mask."""

    from dataclasses import replace

    from ownership.model import owner_residual_evidence_sha256
    from strip.run import _compose_owner_output_pages

    original, strip, bands = _chapter_fixture()
    left_commit = bands[0].owner_execution_commits[0]
    right_commit = bands[1].owner_execution_commits[0]
    right_mutation = right_commit.mutation
    peer_local_protected = np.asarray(right_mutation.protected_art_mask).copy()
    peer_local_protected[np.asarray(left_commit.mutation.action_mask) > 0] = 255
    protected_sha256 = _array_sha256(peer_local_protected)
    right_mutation = replace(
        right_mutation,
        protected_art_mask=peer_local_protected,
        protected_art_mask_sha256=protected_sha256,
        residual_evidence_sha256=owner_residual_evidence_sha256(
            owner_id=right_mutation.owner_id,
            page_id=right_mutation.page_id,
            before_sha256=right_mutation.before_sha256,
            after_sha256=right_mutation.after_sha256,
            action_mask_sha256=right_mutation.action_mask_sha256,
            protected_art_mask_sha256=protected_sha256,
            component_geometry_sha256=right_mutation.component_geometry_sha256,
            residual_score=right_mutation.residual_score,
            residual_threshold=right_mutation.residual_threshold,
            residual_method=right_mutation.residual_method,
            residual_flags=right_mutation.residual_flags,
        ),
    )
    render_geometry = _render_geometry(
        right_mutation.owner_id,
        right_mutation.page_id,
        right_mutation.owner_bbox_page,
        original.shape[:2],
        protected_sha256,
    )
    right_mutation = replace(
        right_mutation,
        owner_render_geometry_sha256=render_geometry.geometry_sha256,
    )
    right_glyph = replace(
        right_commit.glyph_patch,
        owner_render_geometry_sha256=render_geometry.geometry_sha256,
        owner_render_geometry=render_geometry,
    )
    bands[1].owner_execution_commits = [
        replace(right_commit, mutation=right_mutation, glyph_patch=right_glyph)
    ]

    result = _compose_owner_output_pages(
        original_strip_image=original,
        strip=strip,
        bands=bands,
        balloons=[],
        target_count=1,
    )

    composition = result.compositions["page_001"]
    assert composition.committed is True
    assert composition.write_counts["cleanup_write:owner-left"] == 1
    assert composition.write_counts["glyph_write:owner-left"] == 1
    assert composition.write_counts["cleanup_write:owner-right"] == 1
    assert composition.write_counts["glyph_write:owner-right"] == 1


def test_narrow_page_owner_composition_publishes_logical_to_framed_geometry():
    from strip.process_bands import apply_atomic_owner_execution
    from strip.run import _compose_owner_output_pages
    from strip.types import Band, VerticalStrip

    logical = np.full((80, 690, 3), 240, dtype=np.uint8)
    framed = np.zeros((80, 800, 3), dtype=np.uint8)
    framed[:, 55:745] = logical
    strip = VerticalStrip(
        image=framed.copy(), width=800, height=80,
        source_page_breaks=[0, 80], page_x_offsets=[55], source_page_widths=[690],
    )
    mutation = _mutation(logical, owner_id="owner-narrow", bbox=(109, 20, 318, 60), color=(20, 80, 160))
    commit = apply_atomic_owner_execution(
        logical, mutation, _glyph_patch(mutation, bbox=(120, 30, 128, 34))
    )
    assert commit.committed is True
    band = Band(y_top=0, y_bottom=80, original_slice=framed.copy())
    band.owner_execution_commits = [commit]

    result = _compose_owner_output_pages(
        original_strip_image=framed, strip=strip, bands=[band], balloons=[], target_count=1,
    )

    geometry = result.output_pages[0].page_surface_geometry
    assert geometry.logical_width == 690
    assert geometry.frame_width == 800
    assert geometry.content_origin_xy == (55, 0)
    composition = result.compositions["page_001"]
    assert composition.coordinate_space == "framed_page"
    assert composition.final_rgb.shape[:2] == (80, 800)
    assert composition.page_surface_geometry_sha256 == geometry.geometry_sha256
    assert composition.page_surface_geometry.geometry_sha256 == geometry.geometry_sha256
