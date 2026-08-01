from __future__ import annotations

import ast
import copy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))


def test_new_automatic_run_defaults_to_owner_enforce():
    import main

    assert main._automatic_owner_graph_mode({}) == "enforce"
    assert main._automatic_owner_graph_mode({"owner_graph_mode": "shadow"}) == "shadow"


def test_missing_container_is_deferred_to_evidence_reconciliation():
    from ownership.model import SourceTextComponent
    from strip.run import _semantic_regions_for_components

    component = SourceTextComponent(
        component_id="detector_only",
        page_id="page_001",
        bbox_page=(10, 10, 40, 30),
        polygon_page=((10, 10), (40, 10), (40, 30), (10, 30)),
        detector_sources=("fixture",),
        evidence_ids=(),
    )

    region = _semantic_regions_for_components([component])[0]

    assert region.disposition == "owned"
    assert region.reason == "semantic_container_missing"


def test_numeric_footer_and_header_reach_translator_as_one_payload():
    from ownership.model import SourceTextComponent, TextObservation
    from ownership.reconcile import SemanticRegion, build_page_owner_graph
    from ownership.translation import owners_to_translation_page

    component = SourceTextComponent(
        component_id="numeric_body",
        page_id="page_001",
        bbox_page=(100, 100, 300, 145),
        polygon_page=((100, 100), (300, 100), (300, 145), (100, 145)),
        detector_sources=("fixture",),
    )
    observations = [
        TextObservation(
            observation_id="truncated",
            page_id="page_001",
            component_ids=("numeric_body",),
            text="TOTAL PURCHASE AMOUNT",
            confidence=0.99,
            provider="fixture",
            bbox_page=(100, 100, 300, 145),
        ),
        TextObservation(
            observation_id="complete_a",
            page_id="page_001",
            component_ids=("numeric_body",),
            text="TOTAL PURCHASE AMOUNT 200MILLION",
            confidence=0.91,
            provider="fixture",
            bbox_page=(95, 95, 305, 205),
        ),
        TextObservation(
            observation_id="complete_b",
            page_id="page_001",
            component_ids=("numeric_body",),
            text="TOTAL PURCHASE AMOUNT 200 MILLION",
            confidence=0.88,
            provider="fixture",
            bbox_page=(94, 94, 306, 206),
        ),
    ]
    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=observations,
        semantic_regions=[SemanticRegion("body", ("numeric_body",), "body")],
    )

    page = owners_to_translation_page(graph)

    assert len(page["texts"]) == 1
    assert page["texts"][0]["text"].replace(" ", "").endswith("200MILLION")


def test_verified_graph_never_calls_cross_band_reconcile_or_late_layer_merge(monkeypatch):
    import main
    from ownership.legacy_adapter import LegacyUnverifiedAdapter

    project = {"owner_graph_status": "verified", "paginas": []}
    called = []
    monkeypatch.setattr(main, "_merge_same_balloon_fragment_layers", lambda *_args: called.append("merge"))
    monkeypatch.setattr(main, "_rehome_cross_page_band_layers", lambda *_args: called.append("rehome"))

    audit = main._apply_owner_mode_project_repairs(project)

    assert audit["owner_mode"] == "verified"
    assert audit["legacy_helpers_called"] == []
    assert called == []
    with pytest.raises(ValueError, match="legacy_unverified"):
        LegacyUnverifiedAdapter(project)


def test_verified_graph_never_calls_debug_crop_rerender(tmp_path, monkeypatch):
    import main

    project = {"owner_graph_status": "verified", "paginas": []}
    monkeypatch.setattr(
        main,
        "_rerender_strip_reassembled_crops_from_metadata",
        lambda *_args: (_ for _ in ()).throw(AssertionError("legacy crop rerender called")),
    )

    audit = main._rerender_final_project_images_from_metadata(project, tmp_path)

    assert audit["skipped_verified_owner_project"] is True
    assert audit["pages_rerendered"] == 0


def test_legacy_unverified_project_uses_explicit_adapter_only():
    from ownership.legacy_adapter import LegacyUnverifiedAdapter, owner_mode_for_project

    project = {"owner_graph_status": "legacy_unverified", "paginas": []}
    adapter = LegacyUnverifiedAdapter(project)

    assert owner_mode_for_project(project) == "legacy"
    assert adapter.project is project
    assert adapter.can_use_band_semantics is True


def test_legacy_adapter_cannot_mark_graph_verified():
    from ownership.legacy_adapter import LegacyUnverifiedAdapter

    project = {"owner_graph_status": "legacy_unverified", "paginas": []}
    adapter = LegacyUnverifiedAdapter(project)

    with pytest.raises(ValueError, match="cannot mark|verified"):
        adapter.mark_verified()
    assert project["owner_graph_status"] == "legacy_unverified"


def test_owner_mode_does_not_call_mask_or_renderer_semantic_merge_helpers():
    targets = {
        PIPELINE / "inpainter" / "owner_mask.py": {"merge_same_balloon_fragments_before_translation"},
        PIPELINE / "typesetter" / "renderer.py": {"_merge_same_balloon_fragment_group"},
    }
    for path, forbidden in targets.items():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        owner_functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and "owner" in node.name
        ]
        called = {
            node.func.id
            for function in owner_functions
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not (called & forbidden), path


def test_enforce_executes_one_atomic_page_space_chain_per_owner(monkeypatch):
    from inpainter.owner_mask import execute_owner_inpaint
    from ownership.model import (
        ComponentDisposition,
        OwnerGlyphPatch,
        OwnerGraph,
        OwnerProjection,
        SourceTextComponent,
        TextObservation,
        TextOwner,
    )
    from strip.process_bands import execute_owner_page_graph

    def array_hash(value):
        array = np.ascontiguousarray(value)
        digest = sha256()
        digest.update(b"traduzai.ndarray.v1\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
        return digest.hexdigest()

    bbox = (8, 7, 42, 27)
    polygon = ((8, 7), (42, 7), (42, 27), (8, 27))
    graph = OwnerGraph(
        schema_version=1,
        page_id="page_001",
        components=[
            SourceTextComponent(
                component_id="component_a",
                page_id="page_001",
                bbox_page=bbox,
                polygon_page=polygon,
                detector_sources=("fixture",),
                evidence_ids=("region_a",),
            )
        ],
        observations=[
            TextObservation(
                observation_id="observation_a",
                page_id="page_001",
                component_ids=("component_a",),
                text="SOURCE",
                confidence=0.99,
                provider="fixture",
                bbox_page=bbox,
                polygons_page=(polygon,),
                layout_bbox_page=(4, 3, 48, 32),
            )
        ],
        owners=[
            TextOwner(
                owner_id="owner_a",
                page_id="page_001",
                component_ids=["component_a"],
                observation_ids=["observation_a"],
                selected_observation_ids=["observation_a"],
                semantic_role="dialogue_body",
                source_payload="SOURCE",
                translated_payload=None,
                disposition="owned",
                state="execution_planned",
                route_action="translate_inpaint_render",
                execution_tile_id="tile_executor",
            )
        ],
        projections=[
            OwnerProjection(
                owner_id="owner_a",
                tile_id="tile_executor",
                role="executor",
                bbox_page=bbox,
                bbox_tile=bbox,
                offset_xy=(0, 0),
            )
        ],
        component_dispositions=[
            ComponentDisposition(
                component_id="component_a",
                decision="owned",
                owner_id="owner_a",
                reason="fixture",
            )
        ],
    )
    graph.require_valid()
    page = np.full((36, 52, 3), 244, dtype=np.uint8)
    cv2.putText(page, "SRC", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (8, 8, 8), 1)
    calls = []
    state = {}
    from typesetter import style_capture as style_capture_module

    original_extract = style_capture_module.extract_text_style_evidence_v2

    def tracked_extract(*args, **kwargs):
        calls.append(("style_extract", kwargs.get("source_phase")))
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(
        style_capture_module,
        "extract_text_style_evidence_v2",
        tracked_extract,
    )

    class Translator:
        @staticmethod
        def translate_pages(pages, **_kwargs):
            calls.append(("translate", len(pages[0]["texts"])))
            state["translation_payload"] = copy.deepcopy(pages[0])
            return [{"texts": [{"owner_id": "owner_a", "translated": "DESTINO"}]}]

    class Engine:
        engine_name = "fixture"

        @staticmethod
        def inpaint(image, mask, **_kwargs):
            result = image.copy()
            result[mask > 0] = 244
            return result

    class Inpainter:
        @staticmethod
        def inpaint_band_image(image, _record, *, owner_mask_plan):
            calls.append(("inpaint", owner_mask_plan.owner_id))
            mutation = execute_owner_inpaint(image, owner_mask_plan, Engine())
            state["mutation"] = mutation
            return mutation

    class Typesetter:
        @staticmethod
        def render_band_image(image, _record, *, owner_graph):
            from style_v2_fixtures import valid_owner_style_raster_contract
            from typesetter.owner_render_quality import OwnerRenderQuality

            owner = owner_graph.owners[0]
            profile_record = _record["texts"][0]
            visual_profile = profile_record["visual_profile_v2"]
            render_geometry = profile_record["owner_render_geometry"]
            calls.append(("typeset", owner.owner_id))
            result = image.copy()
            result[12:15, 16:25] = 3
            glyph = np.zeros(image.shape[:2], dtype=np.uint8)
            glyph[12:15, 16:25] = 255
            polygon_hash = sha256(
                json.dumps(polygon, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            return OwnerGlyphPatch(
                owner_id=owner.owner_id,
                page_id=owner.page_id,
                coordinate_space="logical_page",
                result_rgb=result,
                glyph_mask=glyph,
                glyph_bbox_page=(16, 12, 25, 15),
                render_completed=True,
                fit_status="ok",
                before_sha256=array_hash(image),
                after_sha256=array_hash(result),
                glyph_mask_sha256=array_hash(glyph),
                changed_outside_glyph_mask_pixels=0,
                render_safe_polygon_page=polygon,
                render_safe_polygon_sha256=polygon_hash,
                component_geometry_sha256=state["mutation"].component_geometry_sha256,
                render_quality_contract=OwnerRenderQuality(
                    schema_version=1,
                    status="ok",
                    font_size_final=14,
                    minimum_legible_font_px=12,
                    source_ink_height_px=None,
                    render_ink_height_px=3,
                    source_x_height_px=None,
                    render_x_height_px=2.1,
                    source_scale_ratio=None,
                    x_height_ratio=None,
                    rendered_line_core_heights_px=(3,),
                    safe_height_occupancy=0.15,
                    safe_area_occupancy=0.05,
                    wrapped_line_count=1,
                    containment_status="ok",
                    outside_safe_pixels=0,
                    page_width=image.shape[1],
                    page_height=image.shape[0],
                    reasons=(),
                ),
                style_raster_contract=valid_owner_style_raster_contract(
                    owner_id=owner.owner_id,
                    page_id=owner.page_id,
                    before=image,
                    result=result,
                    glyph_mask=glyph,
                    component_geometry_sha256=(
                        state["mutation"].component_geometry_sha256
                    ),
                    visual_profile_sha256=visual_profile[
                        "visual_profile_sha256"
                    ],
                    profile_component_geometry_sha256=visual_profile[
                        "component_geometry_sha256"
                    ],
                    source_artifact_sha256=visual_profile["source_sha256"],
                    source_glyph_mask_sha256=visual_profile[
                        "glyph_mask_sha256"
                    ],
                    style_decision=visual_profile[
                        "style_application_decision_v2"
                    ],
                ),
                owner_render_geometry_sha256=render_geometry["geometry_sha256"],
                owner_render_geometry=render_geometry,
                execution_tile_id=owner.execution_tile_id,
            )

    execution = execute_owner_page_graph(
        page,
        graph,
        translator=Translator(),
        inpainter=Inpainter(),
        typesetter=Typesetter(),
    )

    assert calls == [
        ("style_extract", "pre_inpaint"),
        ("translate", 1),
        ("inpaint", "owner_a"),
        ("typeset", "owner_a"),
    ]
    assert len(execution.commits) == 1
    assert execution.commits[0].committed is True
    assert execution.records[0]["owner_id"] == "owner_a"
    assert execution.records[0]["translated"] == "DESTINO"
    assert "owner_style_capture" not in state["translation_payload"]
    assert execution.records[0]["owner_style_capture"]["owner_id"] == "owner_a"
    assert execution.records[0]["owner_style_capture"]["eligible"] is True
    assert execution.records[0]["band_id"] == "tile_executor"
    assert execution.records[0]["render_bbox"] == [16, 12, 25, 15]
    assert execution.records[0]["fit_status"] == "ok"
    assert execution.records[0]["style_v2_raster_contract"] == (
        execution.commits[0].glyph_patch.style_raster_contract.to_dict()
    )
    assert execution.records[0]["safe_text_box"] == [4, 3, 48, 32]
    assert execution.records[0]["target_bbox"] == [4, 3, 48, 32]
    assert execution.records[0]["owner_mask_coverage"] == {
        "selected_observation_ids": ["observation_a"],
        "expected_line_ids": [["observation_a", 0]],
        "covered_line_ids": [["observation_a", 0]],
        "expected_line_polygon_count": 1,
        "covered_line_polygon_count": 1,
        "uncovered_source_ink_pixels": 0,
        "coverage_complete": True,
    }
    assert "missing_render_bbox" not in execution.records[0].get("qa_flags", [])
    assert "fast_fill_no_glyph_evidence" not in execution.records[0].get("qa_flags", [])
    assert execution.records[0]["mask_evidence"]["kind"] == "owner_glyph_mask"
    assert execution.records[0]["mask_evidence"]["raw_mask_pixels"] > 0

    from main import _drop_stale_final_render_geometry

    normalized = _drop_stale_final_render_geometry(dict(execution.records[0]))
    assert normalized["render_bbox"] == [16, 12, 25, 15]
    assert normalized["safe_text_box"] == [4, 3, 48, 32]
    assert normalized["_final_band_render_contract_preserved"] is True


def test_owner_source_glyph_raster_is_clipped_to_authoritative_component_geometry():
    from ownership.model import SourceTextComponent
    from strip.process_bands import _owner_component_glyph_raster

    page = np.full((30, 60, 3), 245, dtype=np.uint8)
    page[10:15, 8:18] = 5
    page[10:15, 38:50] = 5
    component = SourceTextComponent(
        component_id="component_a",
        page_id="page_001",
        bbox_page=(6, 8, 22, 18),
        polygon_page=((6, 8), (22, 8), (22, 18), (6, 18)),
        detector_sources=("fixture",),
    )

    raster = _owner_component_glyph_raster(
        page,
        component=component,
        support_polygons=(((4, 6), (54, 6), (54, 20), (4, 20)),),
    )

    assert np.any(raster[10:15, 8:18])
    assert not np.any(raster[:, :6])
    assert not np.any(raster[:, 22:])


def test_owner_source_glyph_raster_includes_bounded_cleanup_halo():
    from ownership.model import SourceTextComponent
    from strip.process_bands import _owner_component_glyph_raster

    page = np.full((30, 40, 3), 245, dtype=np.uint8)
    page[10:18, 14:22] = 242
    page[11:17, 15:21] = 225
    page[12:16, 16:20] = 5
    component = SourceTextComponent(
        component_id="component_a",
        page_id="page_001",
        bbox_page=(14, 10, 22, 18),
        polygon_page=((14, 10), (22, 10), (22, 18), (14, 18)),
        detector_sources=("fixture",),
    )

    raster = _owner_component_glyph_raster(
        page,
        component=component,
        support_polygons=(((14, 10), (22, 10), (22, 18), (14, 18)),),
    )

    assert np.any(raster[11, 15:21])
    assert np.any(raster[16, 15:21])
    assert not raster[10, 14]
    assert not np.any(raster[:, :14])
    assert not np.any(raster[:, 22:])


def test_atomic_rejection_revokes_all_owner_write_authority():
    from test_final_pixel_qa import _graph
    from strip.process_bands import _transition_owner_to_review

    graph = _graph()
    owner = graph.owners[0]
    owner.action_mask_ref = "owner_masks/owner_a/action.png"
    reviewed = _transition_owner_to_review(graph, owner.owner_id)
    reviewed_owner = reviewed.owners[0]

    assert reviewed_owner.disposition == "review"
    assert reviewed_owner.state == "review_required"
    assert reviewed_owner.route_action == "review_required"
    assert reviewed_owner.execution_tile_id is None
    assert reviewed_owner.action_mask_ref is None
    assert reviewed.projections == []
    assert reviewed.component_dispositions[0].decision == "review"
    assert reviewed.validate() == []


def test_owner_layout_safe_polygon_uses_raster_coordinates_at_page_edges():
    from test_final_pixel_qa import _graph
    from strip.process_bands import _owner_layout_regions

    graph = _graph()
    component = graph.components[0]
    graph.components[0] = replace(
        component,
        bbox_page=(6, 6, 30, 20),
        polygon_page=((6, 6), (30, 6), (30, 20), (6, 20)),
    )
    graph.projections[0] = replace(
        graph.projections[0],
        bbox_page=(6, 6, 30, 20),
        bbox_tile=(6, 6, 30, 20),
    )

    regions = _owner_layout_regions(graph, page_width=30, page_height=20)

    assert max(point[0] for point in regions[0]["safe_polygon_page"]) == 29
    assert max(point[1] for point in regions[0]["safe_polygon_page"]) == 19


def test_owner_layout_uses_selected_observation_container_not_source_glyph_bbox():
    from test_final_pixel_qa import _graph
    from strip.process_bands import _owner_layout_regions

    graph = _graph()
    graph.observations[0] = replace(
        graph.observations[0],
        layout_bbox_page=(2, 3, 36, 24),
    )

    regions = _owner_layout_regions(graph, page_width=40, page_height=30)

    assert regions[0]["bbox_page"] == [2, 3, 36, 24]
    assert regions[0]["safe_polygon_page"] == [
        [2, 3],
        [35, 3],
        [35, 23],
        [2, 23],
    ]


def test_owner_layout_region_derives_source_ink_height_from_selected_polygon():
    from test_final_pixel_qa import _graph
    from strip.process_bands import _owner_layout_regions

    graph = _graph()
    graph.observations[0] = replace(
        graph.observations[0],
        bbox_page=(6, 10, 34, 18),
        polygons_page=(((6, 10), (34, 10), (34, 18), (6, 18)),),
        confidence=0.92,
        coverage_score=0.95,
    )

    region = _owner_layout_regions(graph, page_width=40, page_height=30)[0]

    assert region["source_ink_heights_px"] == [8]
    assert region["source_ink_height_median_px"] == 8.0
    assert region["source_x_heights_px"] == [5.6]
    assert region["source_scale_evidence_confidence"] == pytest.approx(0.874)
    assert region["source_scale_evidence_ids"] == ["observation_a:0"]


def test_owner_layout_ignores_overbroad_container_height_as_source_scale():
    from test_final_pixel_qa import _graph
    from strip.process_bands import _owner_layout_regions

    graph = _graph()
    graph.observations[0] = replace(
        graph.observations[0],
        bbox_page=(7, 11, 31, 19),
        polygons_page=(((7, 11), (31, 11), (31, 19), (7, 19)),),
        layout_bbox_page=(1, 1, 39, 29),
    )

    region = _owner_layout_regions(graph, page_width=40, page_height=30)[0]

    assert region["bbox_page"] == [1, 1, 39, 29]
    assert region["source_ink_height_median_px"] == 8.0
    assert region["source_ink_height_median_px"] != 28.0


def test_connected_owner_uses_complete_selected_ocr_bbox_as_explicit_container():
    from test_final_pixel_qa import _graph
    from strip.process_bands import _owner_layout_regions

    graph = _graph()
    first = graph.components[0]
    second = replace(
        first,
        component_id="component_b",
        bbox_page=(8, 14, 28, 19),
        polygon_page=((8, 14), (28, 14), (28, 19), (8, 19)),
    )
    graph.components.append(second)
    graph.owners[0] = replace(
        graph.owners[0],
        component_ids=(first.component_id, second.component_id),
    )
    graph.observations[0] = replace(
        graph.observations[0],
        component_ids=(first.component_id, second.component_id),
        bbox_page=(4, 3, 32, 22),
        layout_bbox_page=None,
    )

    regions = _owner_layout_regions(graph, page_width=40, page_height=30)

    component_union = (
        min(first.bbox_page[0], second.bbox_page[0]),
        min(first.bbox_page[1], second.bbox_page[1]),
        max(first.bbox_page[2], second.bbox_page[2]),
        max(first.bbox_page[3], second.bbox_page[3]),
    )
    expected = [
        [component_union[0], component_union[1]],
        [component_union[2] - 1, component_union[1]],
        [component_union[2] - 1, component_union[3] - 1],
        [component_union[0], component_union[3] - 1],
    ]
    assert len(regions) == 2
    assert all(region["owner_safe_polygon_page"] == expected for region in regions)
    assert [region["bbox_page"] for region in regions] == [
        list(first.bbox_page),
        list(second.bbox_page),
    ]


def test_review_owner_is_materialized_as_non_rendering_project_record():
    from test_final_pixel_qa import _graph
    from strip.process_bands import (
        _owner_non_rendering_record,
        _transition_owner_to_review,
    )

    graph = _graph()
    _transition_owner_to_review(graph, graph.owners[0].owner_id)
    record = _owner_non_rendering_record(graph, graph.owners[0])

    assert record["owner_id"] == graph.owners[0].owner_id
    assert record["disposition"] == "review"
    assert record["state"] == "review_required"
    assert record["route_action"] == "review_required"
    assert record["visible"] is False
    assert record["action_mask_ref"] is None
    assert record["layout_region_ids"] == []


def test_unsafe_owner_mask_fails_closed_as_review_without_crashing_page(monkeypatch):
    from inpainter.owner_mask import UnsafeOwnerMaskError
    from test_final_pixel_qa import _graph
    from strip.process_bands import execute_owner_page_graph

    graph = _graph(state="execution_planned")
    owner = graph.owners[0]
    owner.route_action = "translate_inpaint_render"
    owner.translated_payload = None
    graph.observations[0] = replace(
        graph.observations[0],
        layout_bbox_page=(0, 0, 40, 24),
    )
    page = np.full((24, 40, 3), 230, dtype=np.uint8)
    page[7:12, 9:24] = 12

    class Translator:
        @staticmethod
        def translate_pages(_pages, **_kwargs):
            return [{"texts": [{"owner_id": owner.owner_id, "translated": "DESTINO"}]}]

    monkeypatch.setattr(
        "inpainter.owner_mask.build_owner_mask_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            UnsafeOwnerMaskError("fixture overbroad mask")
        ),
    )

    execution = execute_owner_page_graph(
        page,
        graph,
        translator=Translator(),
        inpainter=object(),
        typesetter=object(),
    )

    assert execution.commits == ()
    assert execution.graph.owners[0].disposition == "review"
    assert execution.graph.owners[0].route_action == "review_required"
    assert execution.records[0]["visible"] is False
    assert execution.records[0]["route_action"] == "review_required"
    assert "style_v2_raster_contract" not in execution.records[0]


def test_dialogue_without_independent_container_stops_before_inpaint():
    from test_final_pixel_qa import _graph
    from strip.process_bands import execute_owner_page_graph

    graph = _graph(state="execution_planned")
    owner = graph.owners[0]
    owner.route_action = "translate_inpaint_render"
    owner.translated_payload = None
    graph.observations[0] = replace(graph.observations[0], layout_bbox_page=None)
    page = np.full((24, 40, 3), 230, dtype=np.uint8)

    class Translator:
        @staticmethod
        def translate_pages(_pages, **_kwargs):
            return [{"texts": [{"owner_id": owner.owner_id, "translated": "DESTINO"}]}]

    class MustNotRun:
        @staticmethod
        def inpaint_band_image(*_args, **_kwargs):
            raise AssertionError("inpaint must not run without independent container")

        @staticmethod
        def render_band_image(*_args, **_kwargs):
            raise AssertionError("typeset must not run without independent container")

    execution = execute_owner_page_graph(
        page,
        graph,
        translator=Translator(),
        inpainter=MustNotRun(),
        typesetter=MustNotRun(),
    )

    assert execution.commits == ()
    assert execution.records[0]["route_action"] == "review_required"
    assert execution.records[0]["owner_render_geometry"]["status"] == "review_required"
    assert execution.records[0]["owner_execution_rejection_reason"] == "missing_independent_dialogue_container"


def test_only_source_replacement_is_released_before_positive_evidence_sanitization(
    monkeypatch,
):
    from inpainter.owner_mask import UnsafeOwnerMaskError
    from test_final_pixel_qa import _graph
    from strip.process_bands import execute_owner_page_graph

    graph = _graph(state="execution_planned")
    owner = graph.owners[0]
    owner.route_action = "translate_inpaint_render"
    owner.translated_payload = None
    graph.observations[0] = replace(
        graph.observations[0],
        layout_bbox_page=(0, 0, 40, 24),
    )
    page = np.full((24, 40, 3), 230, dtype=np.uint8)
    glyph = np.zeros(page.shape[:2], dtype=np.uint8)
    glyph[7:12, 9:24] = 255
    raw_protected = np.full_like(glyph, 255)
    sanitizer_inputs = []

    class Translator:
        @staticmethod
        def translate_pages(_pages, **_kwargs):
            return [{"texts": [{"owner_id": owner.owner_id, "translated": "DESTINO"}]}]

    monkeypatch.setattr(
        "strip.process_bands._owner_component_glyph_raster",
        lambda *_args, **_kwargs: glyph.copy(),
    )
    monkeypatch.setattr(
        "strip.process_bands._owner_protected_evidence",
        lambda *_args, **_kwargs: (raw_protected.copy(), ("fixture",), 0.0),
    )

    def capture_sanitizer(positive_mask, protected_mask):
        sanitizer_inputs.append(protected_mask.copy())
        result = positive_mask.copy()
        result[protected_mask > 0] = 0
        return result

    monkeypatch.setattr(
        "strip.process_bands.positive_evidence_excluding_protected",
        capture_sanitizer,
    )
    monkeypatch.setattr(
        "inpainter.owner_mask.build_owner_mask_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            UnsafeOwnerMaskError("stop after evidence capture")
        ),
    )

    execute_owner_page_graph(
        page,
        graph,
        translator=Translator(),
        inpainter=object(),
        typesetter=object(),
    )

    assert sanitizer_inputs
    assert int(sanitizer_inputs[0][8, 12]) == 0
    assert int(sanitizer_inputs[0][18, 32]) == 255


def test_semantic_modules_do_not_branch_on_work_chapter_page_number_or_band_id():
    roots = [
        PIPELINE / "ownership",
        PIPELINE / "compositor",
    ]
    files = [path for root in roots for path in root.glob("*.py")]
    files.extend(
        [
            PIPELINE / "qa" / "final_pixel_observer.py",
            PIPELINE / "qa" / "final_pixel_qa.py",
        ]
    )
    forbidden = {"obra", "work_title", "chapter", "chapter_number", "page_number", "band_id"}
    violations = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.IfExp, ast.While)):
                continue
            names = {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
            for name in sorted(names & forbidden):
                violations.append(f"{path.name}:{node.lineno}:{name}")
    assert violations == []
