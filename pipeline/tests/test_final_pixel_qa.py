from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _graph(*, source="SOURCE BODY", translated="CORPO TRADUZIDO", state="rendered"):
    from ownership.model import (
        ComponentDisposition,
        OwnerGraph,
        OwnerProjection,
        SourceTextComponent,
        TextObservation,
        TextOwner,
    )

    component = SourceTextComponent(
        component_id="component_a",
        page_id="page_001",
        bbox_page=(5, 5, 30, 17),
        polygon_page=((5, 5), (30, 5), (30, 17), (5, 17)),
        detector_sources=("fresh_detector",),
    )
    observation = TextObservation(
        observation_id="observation_a",
        page_id="page_001",
        component_ids=("component_a",),
        text=source,
        confidence=0.9,
        provider="source_ocr",
        bbox_page=(5, 5, 30, 17),
    )
    owner = TextOwner(
        owner_id="owner_a",
        page_id="page_001",
        component_ids=["component_a"],
        observation_ids=["observation_a"],
        selected_observation_ids=["observation_a"],
        semantic_role="dialogue_body",
        source_payload=source,
        translated_payload=translated,
        disposition="owned",
        state=state,
        route_action=("review_required" if state == "review_required" else "translate_render_only"),
        execution_tile_id="tile_a",
    )
    projection = OwnerProjection(
        owner_id="owner_a",
        tile_id="tile_a",
        role="executor",
        bbox_page=(5, 5, 30, 17),
        bbox_tile=(5, 5, 30, 17),
        offset_xy=(0, 0),
    )
    disposition = ComponentDisposition(
        component_id="component_a",
        decision="owned",
        owner_id="owner_a",
        reason="resolved_by_owner_graph",
    )
    return OwnerGraph(
        schema_version=1,
        page_id="page_001",
        components=[component],
        observations=[observation],
        owners=[owner],
        projections=[projection],
        component_dispositions=[disposition],
    )


def _composition(
    *, glyph=True, changed_pixels=None, conflicts=(),
    source="SOURCE BODY", translated="CORPO TRADUZIDO",
):
    from ownership.delivery import (
        GlyphRunObservation,
        build_owner_text_delivery_contract,
        seal_owner_text_execution_authority,
    )
    from ownership.model import PageCompositionResult
    from strip.page_surface_geometry import PageSurfaceGeometry

    final = np.full((24, 40, 3), 230, dtype=np.uint8)
    cleanup_map = np.full((24, 40), "", dtype="<U16")
    glyph_map = np.full((24, 40), "", dtype="<U16")
    if glyph:
        glyph_map[7:12, 9:24] = "owner_a"
        final[7:12, 9:24] = 20
    owned = int(np.count_nonzero((cleanup_map != "") | (glyph_map != "")))
    geometry = PageSurfaceGeometry.build(
        logical_width=40,
        logical_height=24,
        frame_width=40,
        frame_height=24,
        content_origin_xy=(0, 0),
    )
    authority = seal_owner_text_execution_authority(
        owner_id="owner_a", page_id="page_001", source_payload=source,
        translated_payload=translated, normalized_chunks=[translated],
    )
    core = np.zeros((24, 40), dtype=np.uint8)
    core[7:12, 9:24] = 255
    run = GlyphRunObservation.build(
        text=translated, font_identity="fixture-font", span_index=0
    )
    delivery = build_owner_text_delivery_contract(
        execution_authority=authority, layout_payload=translated,
        rendered_lines=[translated], rendered_glyph_runs=[run],
        glyph_core_mask=core, glyph_span_core_masks=[core],
        rendered_patch_sha256="c" * 64,
    )
    return PageCompositionResult(
        final_rgb=final,
        cleanup_owner_map=cleanup_map,
        glyph_owner_map=glyph_map,
        conflicts=tuple(conflicts),
        write_counts={
            "cleanup_pixels": 0,
            "glyph_pixels": int(np.count_nonzero(glyph_map != "")),
            "final_changed_pixels": owned if changed_pixels is None else changed_pixels,
            "owner_count": 1,
        },
        sha256="a" * 64,
        page_id="page_001",
        coordinate_space="framed_page",
        committed=not conflicts,
        page_surface_geometry_sha256=geometry.geometry_sha256,
        page_surface_geometry=geometry,
        owner_text_execution_authorities={"owner_a": authority},
        owner_text_delivery_contracts={"owner_a": delivery},
    )


def _observation(*records):
    from qa.final_pixel_observer import FinalPixelObservation
    from strip.page_surface_geometry import PageSurfaceGeometry

    geometry = PageSurfaceGeometry.build(
        logical_width=40,
        logical_height=24,
        frame_width=40,
        frame_height=24,
        content_origin_xy=(0, 0),
    )

    return FinalPixelObservation(
        image_path=Path("final.png"),
        persisted_sha256="b" * 64,
        image_rgb=np.full((24, 40, 3), 230, dtype=np.uint8),
        detected_blocks=tuple(
            {"bbox": record.get("bbox", [5, 5, 30, 17]), "confidence": 0.9}
            for record in records
        ),
        ocr_records=tuple(records),
        source_language="en",
        page_surface_geometry_sha256=geometry.geometry_sha256,
        geometry_projection_count=1,
    )


def _evaluate(graph, composition, observation):
    from qa.final_pixel_qa import evaluate_final_pixel_observation

    return evaluate_final_pixel_observation(
        graph=graph,
        composition=composition,
        observation=observation,
    )


def _reasons(report):
    return {issue.reason for issue in report.issues}


def test_final_pixel_qa_projects_logical_owner_geometry_to_frame_once():
    from ownership.model import PageCompositionResult
    from qa.final_pixel_observer import FinalPixelObservation
    from strip.page_surface_geometry import PageSurfaceGeometry

    geometry = PageSurfaceGeometry.build(
        logical_width=40,
        logical_height=24,
        frame_width=50,
        frame_height=24,
        content_origin_xy=(5, 0),
    )
    final = np.full((24, 50, 3), 230, dtype=np.uint8)
    cleanup = np.full((24, 50), "", dtype="<U16")
    glyph = np.full((24, 50), "", dtype="<U16")
    glyph[7:12, 14:29] = "owner_a"
    composition = PageCompositionResult(
        final_rgb=final,
        cleanup_owner_map=cleanup,
        glyph_owner_map=glyph,
        conflicts=(),
        write_counts={"cleanup_pixels": 0, "glyph_pixels": 75, "final_changed_pixels": 75, "owner_count": 1},
        sha256="a" * 64,
        page_id="page_001",
        coordinate_space="framed_page",
        page_surface_geometry_sha256=geometry.geometry_sha256,
        page_surface_geometry=geometry,
    )
    observation = FinalPixelObservation(
        image_path=Path("final.png"),
        persisted_sha256="b" * 64,
        image_rgb=final,
        detected_blocks=(),
        ocr_records=(),
        source_language="en",
        page_surface_geometry_sha256=geometry.geometry_sha256,
        geometry_projection_count=1,
    )

    report = _evaluate(_graph(), composition, observation)

    assert report.geometry_projection_count == 1
    assert report.owner_support_bbox_frame == (14, 7, 29, 12)
    assert "missing_owner_glyphs" not in _reasons(report)


def test_source_payload_visible_in_final_pixels_blocks_with_empty_metadata_flags():
    report = _evaluate(
        _graph(),
        _composition(),
        _observation({"text": "SOURCE BODY", "bbox": [6, 6, 28, 16], "qa_flags": []}),
    )
    assert "source_payload_visible" in _reasons(report)


def test_independently_detected_text_without_owner_blocks():
    report = _evaluate(
        _graph(),
        _composition(),
        _observation({"text": "MYSTERY", "bbox": [32, 18, 39, 23]}),
    )
    assert "independently_detected_text_without_owner" in _reasons(report)


def test_translated_glyphs_moved_within_layout_are_linked_by_owner_pixel_map():
    from dataclasses import replace

    graph = _graph(source="ANOTHER WAY", translated="OUTRA MANEIRA")
    graph.components[0] = replace(
        graph.components[0],
        bbox_page=(5, 1, 30, 5),
        polygon_page=((5, 1), (30, 1), (30, 5), (5, 5)),
    )
    composition = _composition(
        glyph=True, source="ANOTHER WAY", translated="OUTRA MANEIRA"
    )

    report = _evaluate(
        graph,
        composition,
        _observation({"text": "OUTRA MANEIRA", "bbox": [9, 7, 24, 12]}),
    )

    assert "independently_detected_text_without_owner" not in _reasons(report)


def test_missing_owner_glyphs_block_even_when_render_state_is_complete():
    report = _evaluate(
        _graph(state="rendered"),
        _composition(glyph=False),
        _observation({"text": "CORPO TRADUZIDO", "bbox": [6, 6, 28, 16]}),
    )
    assert "missing_owner_glyphs" in _reasons(report)


def test_owner_glyphs_survive_overlap_composition():
    report = _evaluate(
        _graph(state="rendered"),
        _composition(glyph=True),
        _observation({"text": "CORPO TRADUZIDO", "bbox": [6, 6, 28, 16]}),
    )
    assert "missing_owner_glyphs" not in _reasons(report)


def test_pixel_change_outside_owned_masks_blocks():
    report = _evaluate(
        _graph(),
        _composition(changed_pixels=999),
        _observation({"text": "CORPO TRADUZIDO", "bbox": [6, 6, 28, 16]}),
    )
    assert "pixel_change_outside_owned_masks" in _reasons(report)


def test_later_owned_write_may_restore_pixels_without_becoming_outside_write():
    report = _evaluate(
        _graph(),
        _composition(changed_pixels=70),
        _observation({"text": "CORPO TRADUZIDO", "bbox": [6, 6, 28, 16]}),
    )

    assert "pixel_change_outside_owned_masks" not in _reasons(report)


def test_material_persisted_pixel_change_outside_owner_maps_blocks():
    from dataclasses import replace

    observation = _observation(
        {"text": "CORPO TRADUZIDO", "bbox": [6, 6, 28, 16]}
    )
    persisted = np.asarray(observation.image_rgb).copy()
    persisted[20:23, 34:38] = [0, 0, 0]
    report = _evaluate(
        _graph(),
        _composition(),
        replace(observation, image_rgb=persisted),
    )

    assert "pixel_change_outside_owned_masks" in _reasons(report)


def test_protected_art_damage_blocks():
    from ownership.model import OwnerCompositionConflict

    conflict = OwnerCompositionConflict(
        code="protected_art_changed",
        phase="cleanup",
        owner_ids=("owner_a",),
        pixel_count=4,
    )
    report = _evaluate(_graph(), _composition(conflicts=(conflict,)), _observation())
    assert "protected_art_damage" in _reasons(report)


def test_preserved_name_sfx_or_credit_requires_explicit_policy():
    from ownership.model import ComponentDisposition

    graph = _graph(source="CHOI JIN-SOO")
    graph.owners = []
    graph.projections = []
    graph.component_dispositions = [
        ComponentDisposition(component_id="component_a", decision="preserve", reason=None)
    ]
    report = _evaluate(
        graph,
        _composition(glyph=False),
        _observation({"text": "CHOI JIN-SOO", "bbox": [6, 6, 28, 16]}),
    )
    assert "preserved_source_without_explicit_policy" in _reasons(report)


def test_source_language_detection_uses_owner_ngrams_not_fixed_phrase_list():
    report = _evaluate(
        _graph(source="ZEPHYR QUANTUM", translated="ZEFIRO QUANTICO"),
        _composition(source="ZEPHYR QUANTUM", translated="ZEFIRO QUANTICO"),
        _observation({"text": "Zephyr Quantum", "bbox": [6, 6, 28, 16]}),
    )
    assert "source_payload_visible" in _reasons(report)


def test_review_required_owner_blocks_route_state_contract():
    report = _evaluate(_graph(state="review_required"), _composition(glyph=False), _observation())
    assert "owner_route_not_final" in _reasons(report)


def test_rendered_owner_with_tampered_delivery_blocks_route_and_language_contracts():
    from dataclasses import replace

    composition = _composition()
    delivery = dict(composition.owner_text_delivery_contracts["owner_a"])
    delivery["rendered_payload_sha256"] = "f" * 64
    composition = replace(
        composition,
        owner_text_delivery_contracts={"owner_a": delivery},
    )

    report = _evaluate(_graph(), composition, _observation())

    assert report.contracts["route_state_contract"] == "BLOCK"
    assert report.contracts["final_language_contract"] == "BLOCK"


def test_review_owner_with_visible_source_pixels_blocks_route_and_language():
    report = _evaluate(
        _graph(state="review_required"),
        _composition(glyph=False),
        _observation({"text": "SOURCE BODY", "bbox": [6, 6, 28, 16]}),
    )

    assert report.contracts["route_state_contract"] == "BLOCK"
    assert report.contracts["final_language_contract"] == "BLOCK"


def test_unselected_high_confidence_source_suffix_blocks_incomplete_payload():
    from dataclasses import replace
    from ownership.model import TextObservation

    graph = _graph(source="THE REWARD", translated="A RECOMPENSA")
    extra = TextObservation(
        observation_id="observation_suffix",
        page_id="page_001",
        component_ids=("component_a",),
        text="THE REWARD IS 200 MILLION GOLD",
        confidence=0.96,
        provider="anchored_crop_2x",
        bbox_page=(5, 5, 30, 17),
    )
    graph.observations.append(extra)
    graph.owners[0] = replace(
        graph.owners[0],
        observation_ids=["observation_a", "observation_suffix"],
        selected_observation_ids=["observation_a"],
    )

    report = _evaluate(
        graph,
        _composition(source="THE REWARD", translated="A RECOMPENSA"),
        _observation(),
    )

    assert "source_payload_incomplete" in _reasons(report)


def test_source_observation_polygon_outside_cleanup_map_blocks():
    from dataclasses import replace

    graph = _graph()
    graph.owners[0] = replace(
        graph.owners[0],
        route_action="translate_inpaint_render",
    )
    composition = _composition()
    cleanup = np.asarray(composition.cleanup_owner_map).copy()
    cleanup[5:10, 5:18] = "owner_a"
    composition = replace(composition, cleanup_owner_map=cleanup)

    report = _evaluate(graph, composition, _observation())

    assert "source_evidence_outside_cleanup" in _reasons(report)


def test_anchored_final_ocr_blocks_200_million_residual():
    report = _evaluate(
        _graph(source="THE REWARD IS 200 MILLION GOLD"),
        _composition(
            source="THE REWARD IS 200 MILLION GOLD",
            translated="CORPO TRADUZIDO",
        ),
        _observation({
            "text": "200MILLION",
            "bbox": [6, 6, 28, 16],
            "final_probe_target_id": "component_a",
        }),
    )

    assert "source_payload_visible" in _reasons(report)


def test_letter_digit_boundary_normalization_detects_200million():
    from qa.final_pixel_qa import source_payload_visible

    assert source_payload_visible("PAY 200 MILLION GOLD", "200million")
