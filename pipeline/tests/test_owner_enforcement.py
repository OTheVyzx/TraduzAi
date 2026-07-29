from __future__ import annotations

import ast
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


def test_enforce_executes_one_atomic_page_space_chain_per_owner():
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

    class Translator:
        @staticmethod
        def translate_pages(pages, **_kwargs):
            calls.append(("translate", len(pages[0]["texts"])))
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
            owner = owner_graph.owners[0]
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
                coordinate_space="page",
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
                execution_tile_id=owner.execution_tile_id,
            )

    execution = execute_owner_page_graph(
        page,
        graph,
        translator=Translator(),
        inpainter=Inpainter(),
        typesetter=Typesetter(),
    )

    assert calls == [("translate", 1), ("inpaint", "owner_a"), ("typeset", "owner_a")]
    assert len(execution.commits) == 1
    assert execution.commits[0].committed is True
    assert execution.records[0]["owner_id"] == "owner_a"
    assert execution.records[0]["translated"] == "DESTINO"


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
