from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util

import numpy as np
import pytest

from ownership.model import (
    ComponentDisposition,
    OwnerGraph,
    OwnerProjection,
    SourceTextComponent,
    TextObservation,
    TextOwner,
)


def _polygon(bbox: tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _graph(*, role: str = "dialogue", reverse: bool = False) -> OwnerGraph:
    components = [
        SourceTextComponent("component_a", "page_1", (20, 30, 70, 55), _polygon((20, 30, 70, 55)), ("detector",), evidence_ids=("det_a",)),
        SourceTextComponent("component_b", "page_1", (22, 58, 82, 86), _polygon((22, 58, 82, 86)), ("detector",), evidence_ids=("det_b",)),
    ]
    observations = [
        TextObservation(
            "observation_a", "page_1", ("component_a", "component_b"),
            "SOURCE BODY", 0.94, "paddle", (24, 32, 78, 83),
            polygons_page=(_polygon((24, 32, 68, 53)), _polygon((26, 60, 78, 83))),
            tile_provenance=("tile_002", "tile_001"),
            projection_ids=("projection_b", "projection_a"),
        )
    ]
    projections = [
        OwnerProjection("owner_a", "tile_002", "support", (0, 40, 100, 100), (0, 0, 100, 60), (0, 40)),
        OwnerProjection("owner_a", "tile_001", "executor", (0, 0, 100, 60), (0, 0, 100, 60), (0, 0)),
    ]
    if reverse:
        components.reverse()
        observations.reverse()
        projections.reverse()
    return OwnerGraph(
        schema_version=2,
        page_id="page_1",
        components=components,
        observations=observations,
        owners=[TextOwner("owner_a", "page_1", ["component_b", "component_a"], ["observation_a"], ["observation_a"], role, "SOURCE BODY", "CORPO COMPLETO", "owned", "translated", "translate", "tile_001")],
        projections=projections,
    )


def _container():
    return {
        "evidence_id": "balloon_7",
        "source": "balloon_inner_polygon",
        "bbox_page": (10, 15, 95, 100),
        "polygon_page": _polygon((10, 15, 95, 100)),
        "confidence": 0.91,
    }


def _mask_sha256(mask: np.ndarray) -> str:
    array = np.ascontiguousarray(mask)
    digest = sha256()
    digest.update(b"traduzai.ndarray.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def test_owner_render_geometry_contract_is_importable_before_behavioral_checks():
    assert importlib.util.find_spec("ownership.render_geometry") is not None


def test_owner_render_geometry_contains_complete_page_space_owner_evidence():
    from ownership.render_geometry import build_owner_render_geometry

    geometry = build_owner_render_geometry(
        _graph(), "owner_a", page_width=100, page_height=120,
        container_evidence=_container(), protected_art_mask_sha256="a" * 64,
    )

    assert geometry.logical_space == "logical_page"
    assert geometry.component_ids == ("component_a", "component_b")
    assert all(component.geometry_sha256 for component in geometry.components)
    assert all(observation.polygon_page for observation in geometry.selected_observations)
    assert all(projection.projection_sha256 for projection in geometry.projections)
    assert geometry.semantic_body_bbox_page == (20, 30, 82, 86)
    assert geometry.source_replacement_bbox_page == (24, 32, 78, 83)
    assert geometry.layout_container_source == "balloon_inner_polygon"
    assert geometry.layout_container_bbox_page == (10, 15, 95, 100)
    assert geometry.container_evidence_ids == ("balloon_7",)
    assert geometry.container_evidence_confidence == 0.91
    assert len(geometry.geometry_sha256) == 64


def test_verified_container_is_partitioned_away_from_adjacent_foreign_owner():
    from ownership.render_geometry import build_owner_render_geometry

    graph = _graph()
    graph.components.append(
        SourceTextComponent(
            "component_foreign",
            "page_1",
            (25, 15, 75, 28),
            _polygon((25, 15, 75, 28)),
            ("detector",),
            evidence_ids=("det_foreign",),
        )
    )
    graph.observations.append(
        TextObservation(
            "observation_foreign",
            "page_1",
            ("component_foreign",),
            "FOREIGN",
            0.9,
            "paddle",
            (25, 15, 75, 28),
            polygons_page=(_polygon((25, 15, 75, 28)),),
        )
    )
    graph.owners.append(
        TextOwner(
            "owner_foreign",
            "page_1",
            ["component_foreign"],
            ["observation_foreign"],
            ["observation_foreign"],
            "dialogue",
            "FOREIGN",
            "ESTRANGEIRO",
            "review",
            "review_required",
            "review_required",
            None,
        )
    )

    geometry = build_owner_render_geometry(
        graph,
        "owner_a",
        page_width=100,
        page_height=120,
        container_evidence=_container(),
        protected_art_mask_sha256="a" * 64,
    )

    assert geometry.status == "ready"
    assert geometry.layout_container_bbox_page == (10, 28, 95, 100)
    assert geometry.layout_container_polygon_page == _polygon((10, 28, 95, 100))
    assert geometry.layout_container_source == "balloon_inner_polygon:owner_exclusive"
    assert "foreign_component:component_foreign" in geometry.container_evidence_ids


def test_preserved_foreign_component_uses_accepted_ocr_geometry_not_rejected_coarse_union():
    from ownership.render_geometry import build_owner_render_geometry

    graph = _graph()
    graph.components.append(
        SourceTextComponent(
            "component_ding",
            "page_1",
            (25, 15, 75, 35),
            _polygon((25, 15, 75, 35)),
            ("detector",),
        )
    )
    graph.observations.extend(
        [
            TextObservation(
                "observation_ding",
                "page_1",
                ("component_ding",),
                "DING",
                0.9,
                "paddle_full_page",
                (25, 15, 75, 27),
                polygons_page=(_polygon((25, 15, 75, 27)),),
            ),
            TextObservation(
                "observation_ding_elixir",
                "page_1",
                ("component_ding",),
                "DING ELIXIR",
                0.94,
                "candidate_crop_direct_paddle",
                (25, 15, 75, 35),
                polygons_page=(_polygon((25, 15, 75, 35)),),
                rejection_reason="cross_region_same_role_observation",
            ),
        ]
    )
    graph.component_dispositions.append(
        ComponentDisposition(
            component_id="component_ding",
            decision="preserve",
            reason="policy:short_noop_display_sfx",
        )
    )

    geometry = build_owner_render_geometry(
        graph,
        "owner_a",
        page_width=100,
        page_height=120,
        container_evidence=_container(),
        protected_art_mask_sha256="a" * 64,
    )

    assert geometry.status == "ready"
    assert geometry.layout_container_bbox_page == (10, 27, 95, 100)
    assert "foreign_component:component_ding" in geometry.container_evidence_ids


def test_foreign_component_may_overlap_component_padding_without_source_ink_overlap():
    from ownership.render_geometry import build_owner_render_geometry

    graph = _graph()
    graph.components.append(
        SourceTextComponent(
            "component_ding",
            "page_1",
            (60, 15, 90, 31),
            _polygon((60, 15, 90, 31)),
            ("detector",),
        )
    )
    graph.observations.append(
        TextObservation(
            "observation_ding",
            "page_1",
            ("component_ding",),
            "DING",
            0.9,
            "paddle_full_page",
            (60, 15, 90, 31),
            polygons_page=(_polygon((60, 15, 90, 31)),),
        )
    )

    geometry = build_owner_render_geometry(
        graph,
        "owner_a",
        page_width=100,
        page_height=120,
        container_evidence=_container(),
        protected_art_mask_sha256="a" * 64,
    )

    assert geometry.status == "ready"
    assert geometry.layout_container_bbox_page == (10, 31, 95, 100)
    assert geometry.source_replacement_bbox_page == (24, 32, 78, 83)


def test_foreign_component_overlapping_source_replacement_still_requires_review():
    from ownership.render_geometry import build_owner_render_geometry

    graph = _graph()
    graph.components.append(
        SourceTextComponent(
            "component_foreign",
            "page_1",
            (60, 15, 90, 40),
            _polygon((60, 15, 90, 40)),
            ("detector",),
        )
    )
    graph.observations.append(
        TextObservation(
            "observation_foreign",
            "page_1",
            ("component_foreign",),
            "FOREIGN",
            0.9,
            "paddle_full_page",
            (60, 15, 90, 40),
            polygons_page=(_polygon((60, 15, 90, 40)),),
        )
    )

    geometry = build_owner_render_geometry(
        graph,
        "owner_a",
        page_width=100,
        page_height=120,
        container_evidence=_container(),
        protected_art_mask_sha256="a" * 64,
    )

    assert geometry.status == "review_required"
    assert geometry.reason == "foreign_owner_geometry_overlaps_semantic_body"


def test_verified_container_expands_only_through_protected_mask_safe_pixels():
    from ownership.render_geometry import build_owner_render_geometry

    protected = np.zeros((120, 100), dtype=np.uint8)
    protected[15:100, 80:95] = 255
    protected[88:100, 10:95] = 255

    geometry = build_owner_render_geometry(
        _graph(),
        "owner_a",
        page_width=100,
        page_height=120,
        container_evidence=_container(),
        protected_art_mask=protected,
        protected_art_mask_sha256=_mask_sha256(protected),
    )

    assert geometry.status == "ready"
    assert geometry.layout_container_bbox_page == (10, 15, 80, 88)
    assert geometry.layout_container_polygon_page == _polygon((10, 15, 80, 88))
    assert geometry.layout_container_source.endswith(":protected_mask_safe")


def test_protected_art_inside_source_slot_keeps_authenticated_typography_slot():
    from ownership.render_geometry import build_owner_render_geometry

    protected = np.zeros((120, 100), dtype=np.uint8)
    protected[48:58, 50:60] = 255

    geometry = build_owner_render_geometry(
        _graph(),
        "owner_a",
        page_width=100,
        page_height=120,
        container_evidence=_container(),
        protected_art_mask=protected,
        protected_art_mask_sha256=_mask_sha256(protected),
    )

    assert geometry.status == "ready"
    assert geometry.layout_container_bbox_page == geometry.source_replacement_bbox_page
    assert geometry.layout_container_source.endswith(":protected_mask_safe")


def test_legacy_layout_bbox_never_expands_source_replacement_beyond_text_pixels():
    from ownership.render_geometry import (
        build_owner_render_geometry,
        owner_source_replacement_bbox,
    )

    graph = _graph()
    graph.observations[0] = replace(
        graph.observations[0],
        bbox_page=(10, 15, 95, 100),
        text_pixel_bbox_page=(24, 32, 78, 83),
        legacy_selected=True,
        provider="legacy_selected",
    )
    protected = np.zeros((120, 100), dtype=np.uint8)
    protected[90:100, 20:80] = 255

    geometry = build_owner_render_geometry(
        graph,
        "owner_a",
        page_width=100,
        page_height=120,
        container_evidence=_container(),
        protected_art_mask=protected,
        protected_art_mask_sha256=_mask_sha256(protected),
    )

    assert owner_source_replacement_bbox(graph, "owner_a") == (24, 32, 78, 83)
    assert geometry.source_replacement_bbox_page == (24, 32, 78, 83)
    assert geometry.status == "ready"


def test_coarse_foreign_mask_never_reprotects_authorized_source_replacement():
    from ownership.render_geometry import release_source_replacement_from_protection

    protected = np.zeros((20, 30), dtype=np.uint8)
    foreign = np.zeros_like(protected)
    foreign[5:15, 8:22] = 255

    released = release_source_replacement_from_protection(
        protected,
        (10, 7, 20, 13),
        foreign_component_masks=(("coarse_preserved_component", foreign),),
    )

    assert not np.any(released[7:13, 10:20])
    assert np.any(released[5:7, 8:22])


def test_source_glyph_mask_releases_text_without_unprotecting_crossing_art():
    from ownership.render_geometry import release_source_replacement_from_protection

    protected = np.zeros((24, 36), dtype=np.uint8)
    protected[8:16, 4:32] = 255
    source_glyph = np.zeros_like(protected)
    source_glyph[10:14, 12:24] = 255

    released = release_source_replacement_from_protection(
        protected,
        (8, 6, 28, 18),
        source_replacement_mask=source_glyph,
    )

    assert not np.any(released[10:14, 12:24])
    assert np.all(released[8:10, 4:32] == 255)
    assert np.all(released[14:16, 4:32] == 255)


def test_source_slot_ignores_coarse_foreign_union_when_glyph_mask_is_precise():
    from ownership.render_geometry import release_source_replacement_from_protection

    protected = np.zeros((24, 36), dtype=np.uint8)
    source_glyph = np.zeros_like(protected)
    source_glyph[10:14, 12:24] = 255
    coarse_foreign = np.zeros_like(protected)
    coarse_foreign[6:18, 8:28] = 255

    released = release_source_replacement_from_protection(
        protected,
        (8, 6, 28, 18),
        source_replacement_mask=source_glyph,
        foreign_component_masks=(("coarse_foreign", coarse_foreign),),
    )

    assert not np.any(released[6:18, 8:28])


def test_source_replacement_never_becomes_dialogue_layout_container_without_evidence():
    from ownership.render_geometry import build_owner_render_geometry

    geometry = build_owner_render_geometry(_graph(), "owner_a", page_width=100, page_height=120)

    assert geometry.source_replacement_bbox_page == (24, 32, 78, 83)
    assert geometry.layout_container_bbox_page is None
    assert geometry.status == "review_required"
    assert geometry.reason == "missing_independent_dialogue_container"


def test_independently_detected_visual_card_text_slot_is_a_typed_local_container():
    from ownership.render_geometry import build_owner_render_geometry

    graph = _graph()
    graph.observations[0] = replace(
        graph.observations[0], provider="visual_card_full_page_raw"
    )
    candidate = replace(
        graph.observations[0],
        observation_id="observation_card_candidate",
        provider="candidate_crop_direct_paddle_native",
    )
    graph.observations.append(candidate)
    graph.owners[0].observation_ids.append(candidate.observation_id)
    graph.owners[0].selected_observation_ids.append(candidate.observation_id)

    geometry = build_owner_render_geometry(
        graph,
        "owner_a",
        page_width=100,
        page_height=120,
    )

    assert geometry.status == "ready"
    assert geometry.reason == "typed_visual_card_text_slot"
    assert geometry.layout_container_source == "visual_card_text_slot"
    assert geometry.layout_container_bbox_page == geometry.semantic_body_bbox_page


def test_freeform_sfx_may_use_typed_component_union_not_cleanup_footprint():
    from ownership.render_geometry import build_owner_render_geometry

    geometry = build_owner_render_geometry(_graph(role="freeform_sfx"), "owner_a", page_width=100, page_height=120)

    assert geometry.layout_container_source == "freeform_component_union"
    assert geometry.layout_container_bbox_page == geometry.semantic_body_bbox_page
    assert geometry.layout_container_bbox_page != geometry.source_replacement_bbox_page
    assert geometry.status == "ready"


def test_geometry_hash_covers_polygons_not_only_ids_and_union_bbox():
    from ownership.render_geometry import build_owner_render_geometry

    first_graph = _graph(role="freeform_sfx")
    second_graph = _graph(role="freeform_sfx")
    second_graph.components[0] = replace(
        second_graph.components[0],
        polygon_page=((20, 30), (70, 30), (65, 55), (20, 55)),
    )

    first = build_owner_render_geometry(first_graph, "owner_a", page_width=100, page_height=120)
    second = build_owner_render_geometry(second_graph, "owner_a", page_width=100, page_height=120)

    assert first.component_ids == second.component_ids
    assert first.semantic_body_bbox_page == second.semantic_body_bbox_page
    assert first.component_geometry_sha256 != second.component_geometry_sha256
    assert first.geometry_sha256 != second.geometry_sha256


def test_geometry_is_invariant_to_graph_and_executor_order():
    from ownership.render_geometry import build_owner_render_geometry

    first = build_owner_render_geometry(_graph(), "owner_a", page_width=100, page_height=120, container_evidence=_container(), protected_art_mask_sha256="a" * 64)
    second = build_owner_render_geometry(_graph(reverse=True), "owner_a", page_width=100, page_height=120, container_evidence=_container(), protected_art_mask_sha256="a" * 64)

    assert first.to_dict() == second.to_dict()


def test_round_trip_rejects_tampered_geometry_or_nested_hash():
    from ownership.render_geometry import OwnerRenderGeometry, build_owner_render_geometry

    geometry = build_owner_render_geometry(_graph(), "owner_a", page_width=100, page_height=120, container_evidence=_container(), protected_art_mask_sha256="a" * 64)
    payload = geometry.to_dict()
    assert OwnerRenderGeometry.from_dict(payload) == geometry

    with pytest.raises(ValueError, match="geometry hash"):
        OwnerRenderGeometry.from_dict({**payload, "geometry_sha256": "0" * 64})
    payload["components"][0]["geometry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="component geometry hash"):
        OwnerRenderGeometry.from_dict(payload)


def test_connected_subregions_and_projection_offsets_are_typed_and_bound():
    from ownership.render_geometry import build_owner_render_geometry

    geometry = build_owner_render_geometry(_graph(role="freeform_sfx"), "owner_a", page_width=100, page_height=120)

    assert tuple(region.component_ids for region in geometry.connected_subregions) == (("component_a",), ("component_b",))
    assert tuple(region.order for region in geometry.connected_subregions) == (0, 1)
    assert geometry.projections[0].tile_to_page_offset_xy == (0, 0)
    assert geometry.projections[0].role == "executor"
    assert geometry.projections[1].tile_to_page_offset_xy == (0, 40)
