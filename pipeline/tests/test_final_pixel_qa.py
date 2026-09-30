from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from ownership.hash_contract import sha256_text


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _graph(*, source="SOURCE BODY", translated="CORPO TRADUZIDO", state="rendered"):
    from ownership.model import (
        OWNER_GRAPH_SCHEMA_VERSION,
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
        run_id="run-final-pixel-qa",
        origin_execution_id="execution-final-pixel-qa",
        invocation_id="invocation-final-pixel-qa-primary",
        attempt_id="attempt-final-pixel-qa-primary",
        provider_family="source_ocr",
        page_source_sha256="a" * 64,
        root_input_pixel_sha256="b" * 64,
        input_pixel_sha256="c" * 64,
        payload_sha256=sha256_text(source),
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
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id="page_001",
        run_id="run-final-pixel-qa",
        origin_execution_id="execution-final-pixel-qa",
        page_source_sha256="a" * 64,
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


def test_uncertain_source_component_remains_a_terminal_qa_blocker():
    from ownership.model import ComponentDisposition, SourceTextComponent

    graph = _graph()
    graph.components.append(
        SourceTextComponent(
            component_id="ambiguous_candidate",
            page_id="page_001",
            bbox_page=(31, 5, 39, 17),
            polygon_page=((31, 5), (39, 5), (39, 17), (31, 17)),
            detector_sources=("primary_region_detector",),
            confidence=0.27,
        )
    )
    graph.component_dispositions.append(
        ComponentDisposition(
            component_id="ambiguous_candidate",
            decision="uncertain",
            reason="coverage:empty_uncorroborated_primary_candidate",
            policy_id="coverage_ambiguous_candidate",
            policy_bbox_page=(31, 5, 39, 17),
            policy_evidence_ids=("ocr_attempt",),
            policy_reason="Human review required after empty Coverage OCR",
        )
    )

    report = _evaluate(graph, _composition(), _observation())

    assert "uncertain_source_component_requires_review" in _reasons(report)


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


def test_final_ocr_matches_legacy_frame_space_component_without_waiving_text():
    from dataclasses import replace

    from ownership.model import ComponentDisposition, PageCompositionResult
    from qa.final_pixel_observer import FinalPixelObservation
    from strip.page_surface_geometry import PageSurfaceGeometry

    geometry = PageSurfaceGeometry.build(
        logical_width=40,
        logical_height=24,
        frame_width=80,
        frame_height=24,
        content_origin_xy=(40, 0),
    )
    graph = _graph()
    graph.components[0] = replace(
        graph.components[0],
        bbox_page=(45, 5, 70, 17),
        polygon_page=((45, 5), (70, 5), (70, 17), (45, 17)),
    )
    graph.observations = []
    graph.owners = []
    graph.projections = []
    graph.component_dispositions = [
        ComponentDisposition(
            component_id="component_a",
            decision="preserve",
            reason="policy:explicit_sfx_outside_translatable_container",
        )
    ]
    final = np.full((24, 80, 3), 230, dtype=np.uint8)
    empty_map = np.full((24, 80), "", dtype="<U16")
    composition = PageCompositionResult(
        final_rgb=final,
        cleanup_owner_map=empty_map,
        glyph_owner_map=empty_map.copy(),
        conflicts=(),
        write_counts={"cleanup_pixels": 0, "glyph_pixels": 0, "final_changed_pixels": 0, "owner_count": 0},
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
        ocr_records=({"text": "HO", "bbox": [5, 5, 30, 17]},),
        source_language="en",
        page_surface_geometry_sha256=geometry.geometry_sha256,
        geometry_projection_count=1,
    )

    report = _evaluate(graph, composition, observation)

    assert "independently_detected_text_without_owner" not in _reasons(report)


def test_source_payload_visible_in_final_pixels_blocks_with_empty_metadata_flags():
    report = _evaluate(
        _graph(),
        _composition(),
        _observation({"text": "SOURCE BODY", "bbox": [6, 6, 28, 16], "qa_flags": []}),
    )
    assert "source_payload_visible" in _reasons(report)


def test_source_token_shared_with_distinct_translation_is_not_residual():
    report = _evaluate(
        _graph(
            source="IS THAT ALSO A HYBRID OF GURITA",
            translated="ISSO TAMBEM E UM HIBRIDO DE GURITA",
        ),
        _composition(
            source="IS THAT ALSO A HYBRID OF GURITA",
            translated="ISSO TAMBEM E UM HIBRIDO DE GURITA",
        ),
        _observation(
            {
                "text": "Isso tambem e um hibrido de padre e gurita?",
                "bbox": [6, 6, 28, 16],
            }
        ),
    )

    assert "source_payload_visible" not in _reasons(report)


def test_accented_target_token_shared_with_source_is_not_residual():
    from dataclasses import replace

    graph = _graph(
        source="UM DOS TRES DE SIHYUK HEAVENLY SLAUGHTER STAR",
        translated="UM DOS TRÊS DE SIHYUK ESTRELA DO MASSACRE CELESTIAL",
    )
    graph.observations[0] = replace(
        graph.observations[0],
        text="UM DOS TRES DE SIHYUK",
    )
    report = _evaluate(
        graph,
        _composition(
            source="UM DOS TRES DE SIHYUK HEAVENLY SLAUGHTER STAR",
            translated="UM DOS TRÊS DE SIHYUK ESTRELA DO MASSACRE CELESTIAL",
        ),
        _observation(
            {
                "text": "UM DOS TRES DE SIHYUK ESTRELA DO MASSACRE CELESTIAL",
                "bbox": [6, 6, 28, 16],
            }
        ),
    )

    assert "source_payload_visible" not in _reasons(report)


def test_identical_verified_target_repaint_is_not_its_own_source_residual():
    report = _evaluate(
        _graph(source="DING", translated="DING"),
        _composition(source="DING", translated="DING"),
        _observation({"text": "DING", "bbox": [6, 6, 28, 16]}),
    )

    assert "source_payload_visible" not in _reasons(report)


def test_independently_detected_text_without_owner_blocks():
    report = _evaluate(
        _graph(),
        _composition(),
        _observation({"text": "MYSTERY", "bbox": [32, 18, 39, 23]}),
    )
    assert "independently_detected_text_without_owner" in _reasons(report)


@pytest.mark.parametrize(
    "observed",
    [
        "NOVATO",
        "O QUE...?",
        "VICE-MESTRE DA GUILDA",
        "000000",
    ],
)
def test_independently_detected_verified_pt_br_or_neutral_text_does_not_block(observed):
    graph = _graph()
    graph.owners = []
    graph.projections = []
    report = _evaluate(
        graph,
        _composition(glyph=False),
        _observation({"text": observed, "bbox": [32, 18, 39, 23]}),
    )

    assert "independently_detected_text_without_owner" not in _reasons(report)


def test_two_unknown_words_are_not_waived_as_a_proper_name():
    graph = _graph()
    graph.owners = []
    graph.projections = []
    report = _evaluate(
        graph,
        _composition(glyph=False),
        _observation({"text": "UNOWNED ENGLISH", "bbox": [32, 18, 39, 23]}),
    )

    assert "independently_detected_text_without_owner" in _reasons(report)


def test_full_page_aggregate_is_global_language_evidence_not_local_unowned_text():
    report = _evaluate(
        _graph(),
        _composition(),
        _observation({
            "text": "CORPO TRADUZIDO",
            "bbox": [0, 0, 999, 999],
            "final_probe_target_id": "full-page:full_page",
        }),
    )

    assert "observation_bbox_outside_logical_page" not in _reasons(report)
    assert "independently_detected_text_without_owner" not in _reasons(report)


def test_full_page_aggregate_still_blocks_known_source_payload():
    report = _evaluate(
        _graph(),
        _composition(),
        _observation({
            "text": "SOURCE BODY",
            "bbox": [0, 0, 999, 999],
            "final_probe_target_id": "full-page:full_page",
        }),
    )

    assert "source_payload_visible" in _reasons(report)


def test_punctuation_only_final_ocr_is_not_independent_text():
    report = _evaluate(
        _graph(),
        _composition(),
        _observation({"text": "......", "bbox": [32, 18, 39, 23]}),
    )

    assert "independently_detected_text_without_owner" not in _reasons(report)


def test_short_noise_anchored_to_suppressed_non_text_is_not_independent_text():
    from ownership.model import ComponentDisposition

    graph = _graph()
    graph.owners = []
    graph.projections = []
    graph.component_dispositions = [
        ComponentDisposition(
            component_id="component_a",
            decision="suppress",
            reason="redundant_container_without_ocr_evidence",
        )
    ]
    report = _evaluate(
        graph,
        _composition(glyph=False),
        _observation(
            {
                "text": "oe",
                "bbox": [6, 6, 28, 16],
                "final_probe_target_id": "component_a",
            }
        ),
    )

    assert "independently_detected_text_without_owner" not in _reasons(report)


def test_unassociated_ocr_inside_explicit_scanlation_zone_is_preserved():
    from dataclasses import replace
    from ownership.model import ComponentDisposition

    graph = _graph()
    graph.components[0] = replace(
        graph.components[0],
        bbox_page=(0, 14, 12, 23),
        polygon_page=((0, 14), (12, 14), (12, 23), (0, 23)),
    )
    graph.owners = []
    graph.projections = []
    graph.component_dispositions = [
        ComponentDisposition(
            component_id="component_a",
            decision="preserve",
            reason="policy:scanlation_apparatus",
        )
    ]
    report = _evaluate(
        graph,
        _composition(glyph=False),
        _observation(
            {
                "text": "SUPPORT OUR TRANSLATORS AT OUR SITE",
                "bbox": [22, 16, 39, 22],
            }
        ),
    )

    assert "independently_detected_text_without_owner" not in _reasons(report)


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


def test_verified_target_language_owner_preserves_original_without_cleanup_or_glyphs():
    from dataclasses import replace

    source = "UAU, VOCE ME ASSUSTOU."
    graph = _graph(source=source, translated=source, state="target_ready")
    graph.owners[0] = replace(
        graph.owners[0],
        route_action="translate_inpaint_render",
    )
    report = _evaluate(
        graph,
        _composition(glyph=False, source=source, translated=source),
        _observation({"text": source, "bbox": [6, 6, 28, 16]}),
    )

    assert "owner_route_not_final" not in _reasons(report)
    assert "missing_owner_glyphs" not in _reasons(report)
    assert "source_evidence_outside_cleanup" not in _reasons(report)
    assert "source_payload_visible" not in _reasons(report)
    assert report.passed is True


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
        run_id="run-final-pixel-qa",
        origin_execution_id="execution-final-pixel-qa",
        invocation_id="invocation-final-pixel-qa-suffix",
        attempt_id="attempt-final-pixel-qa-suffix",
        provider_family="anchored_crop",
        page_source_sha256="a" * 64,
        root_input_pixel_sha256="b" * 64,
        input_pixel_sha256="d" * 64,
        payload_sha256=sha256_text("THE REWARD IS 200 MILLION GOLD"),
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


def test_selected_line_support_may_have_small_geometric_edge_outside_cleanup():
    from dataclasses import replace

    graph = _graph()
    graph.owners[0] = replace(
        graph.owners[0],
        route_action="translate_inpaint_render",
    )
    graph.observations[0] = replace(
        graph.observations[0],
        polygons_page=(((5, 5), (30, 5), (30, 17), (5, 17)),),
    )
    composition = _composition()
    cleanup = np.asarray(composition.cleanup_owner_map).copy()
    cleanup[5:17, 5:31] = "owner_a"
    composition = replace(composition, cleanup_owner_map=cleanup)

    report = _evaluate(graph, composition, _observation())

    assert "source_evidence_outside_cleanup" not in _reasons(report)


def test_selected_line_support_accepts_material_glyph_density_cleanup():
    from dataclasses import replace

    graph = _graph()
    graph.owners[0] = replace(
        graph.owners[0],
        route_action="translate_inpaint_render",
    )
    graph.observations[0] = replace(
        graph.observations[0],
        polygons_page=(((5, 5), (30, 5), (30, 17), (5, 17)),),
    )
    composition = _composition()
    cleanup = np.asarray(composition.cleanup_owner_map).copy()
    cleanup[5:18, 5:31:2] = "owner_a"
    composition = replace(composition, cleanup_owner_map=cleanup)

    report = _evaluate(graph, composition, _observation())

    assert "source_evidence_outside_cleanup" not in _reasons(report)


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
