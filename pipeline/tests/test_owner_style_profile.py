"""Contracts for the immutable page-owner visual style sidecar."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.model import (  # noqa: E402
    ComponentDisposition,
    OwnerGraph,
    OwnerProjection,
    SourceTextComponent,
    TextObservation,
    TextOwner,
)
from ownership.translation import owners_to_translation_page  # noqa: E402
from typesetter import renderer as renderer_mod  # noqa: E402
from typesetter.owner_style import (  # noqa: E402
    _materialize_style,
    attach_owner_visual_profile,
    build_owner_visual_profile,
    build_owner_visual_profiles,
    owner_visual_profile_sha256,
)
from typesetter.style_capture import build_owner_style_capture  # noqa: E402


def _fixture() -> tuple[np.ndarray, OwnerGraph]:
    bbox = (6, 5, 26, 19)
    component = SourceTextComponent(
        component_id="component_a",
        page_id="page_001",
        bbox_page=bbox,
        polygon_page=((6, 5), (26, 5), (26, 19), (6, 19)),
        detector_sources=("independent_text_recall",),
        evidence_ids=("container_a",),
    )
    observation = TextObservation(
        observation_id="observation_a",
        page_id="page_001",
        component_ids=(component.component_id,),
        text="SOURCE BODY",
        confidence=0.96,
        provider="paddle_full_page",
        bbox_page=bbox,
        polygons_page=(component.polygon_page,),
        tile_provenance=("tile_bottom", "tile_top"),
        coverage_score=1.0,
    )
    owner = TextOwner(
        owner_id="owner_a",
        page_id="page_001",
        component_ids=[component.component_id],
        observation_ids=[observation.observation_id],
        selected_observation_ids=[observation.observation_id],
        semantic_role="dialogue_body",
        source_payload="SOURCE BODY",
        translated_payload="CORPO TRADUZIDO",
        disposition="owned",
        state="translated",
        route_action="translate_inpaint_render",
        execution_tile_id="tile_top",
    )
    projections = [
        OwnerProjection("owner_a", "tile_top", "executor", bbox, bbox, (0, 0)),
        OwnerProjection("owner_a", "tile_bottom", "context_only", bbox, bbox, (0, 0)),
    ]
    graph = OwnerGraph(
        schema_version=1,
        page_id="page_001",
        components=[component],
        observations=[observation],
        owners=[owner],
        projections=projections,
        component_dispositions=[
            ComponentDisposition(component.component_id, "owned", "owner_a", "resolved")
        ],
    )
    page = np.full((24, 32, 3), 245, dtype=np.uint8)
    page[8:16, 9:23] = 20
    return page, graph


def _glyph_mask(page: np.ndarray) -> np.ndarray:
    mask = np.zeros(page.shape[:2], dtype=np.uint8)
    mask[8:16, 9:23] = 255
    return mask


def _semantic_signature(graph: OwnerGraph) -> dict:
    return graph.to_dict()


def test_every_renderable_owner_receives_explicit_style_status() -> None:
    page, graph = _fixture()
    profiles = build_owner_visual_profiles(
        graph,
        page,
        glyph_masks_by_owner={"owner_a": _glyph_mask(page)},
        candidates_by_owner={"owner_a": {"confidence": 0.96}},
    )

    assert set(profiles) == {"owner_a"}
    assert profiles["owner_a"]["status"] in {
        "applied", "fallback", "not_applicable", "review_required"
    }
    assert profiles["owner_a"]["source_capture_phase"] == "pre_inpaint"
    assert profiles["owner_a"]["page_id"] == "page_001"


def test_owner_profile_is_captured_before_inpaint_mutates_pixels() -> None:
    page, graph = _fixture()
    profile = build_owner_visual_profile(
        graph.owners[0],
        page,
        components=graph.components,
        observations=graph.observations,
        glyph_mask=_glyph_mask(page),
        candidate={"confidence": 0.96},
    )
    captured_hash = profile["source_sha256"]
    page[5:19, 6:26] = 128

    assert profile["source_sha256"] == captured_hash
    assert build_owner_visual_profile(
        graph.owners[0],
        page,
        components=graph.components,
        observations=graph.observations,
        glyph_mask=_glyph_mask(page),
        candidate={"confidence": 0.96},
    )["source_sha256"] != captured_hash


def test_owner_profile_round_trips_to_renderer() -> None:
    page, graph = _fixture()
    profile = build_owner_visual_profile(
        graph.owners[0], page, components=graph.components,
        observations=graph.observations, glyph_mask=_glyph_mask(page),
        candidate={"confidence": 0.96},
    )
    record = attach_owner_visual_profile({"owner_id": "owner_a"}, profile)

    assert renderer_mod._owner_visual_profile(record) == profile["applied_style"]


def test_owner_profile_consumes_masked_capture_instead_of_legacy_empty_evidence() -> None:
    page, graph = _fixture()
    glyph = np.zeros(page.shape[:2], dtype=np.uint8)
    glyph[10:14, 12:20] = 255
    capture = build_owner_style_capture(
        graph,
        "owner_a",
        page,
        glyph_mask=glyph,
    )

    profile = build_owner_visual_profile(
        graph.owners[0],
        page,
        components=graph.components,
        observations=graph.observations,
        glyph_mask=glyph,
        candidate={
            "confidence": capture.candidate_confidence,
            "route_action": capture.route_action,
            "owner_style_capture": capture.to_dict(),
        },
    )

    assert profile["style_evidence_v2"] == capture.style_evidence_v2.to_dict()
    assert profile["style_evidence_v2"]["source"] == "owner_mask_v2"


def test_materialization_maps_every_applied_canonical_attribute() -> None:
    applied = {
        "font_name": "KOMIKAX_.ttf",
        "font_weight": "bold",
        "font_width": "condensed",
        "fill": "#F4F4F4",
        "stroke": {"color": "#111111", "width_px": 2},
        "shadow": {"color": "#222222", "offset": [2, 3]},
        "glow": {"color": "#66AAFF", "width_px": 4},
        "gradient": ["#FFFFFF", "#88AAFF"],
        "rotation_deg": -8.0,
        "tracking_xh": 0.12,
        "slant_tangent": 0.18,
        "width_scale": 0.82,
        "scale_y": 1.1,
    }

    style = _materialize_style(
        {"background_rgb": [20, 20, 20]},
        {"applied_attributes": applied},
    )

    assert set(style["canonical_applied_attributes"]) == set(applied)
    assert style["canonical_applied_attributes"] == applied


def test_style_profile_cannot_change_owner_semantic_signature() -> None:
    page, graph = _fixture()
    before = _semantic_signature(graph)
    profile = build_owner_visual_profile(
        graph.owners[0], page, components=graph.components,
        observations=graph.observations, glyph_mask=_glyph_mask(page),
        candidate={"confidence": 0.96, "source_payload": "FORGED"},
    )
    attach_owner_visual_profile({"owner_id": "owner_a"}, profile)

    assert _semantic_signature(graph) == before
    assert "source_payload" not in profile["applied_style"]


def test_divergent_profiles_for_same_owner_are_rejected() -> None:
    page, graph = _fixture()
    profile = build_owner_visual_profile(
        graph.owners[0], page, components=graph.components,
        observations=graph.observations, glyph_mask=_glyph_mask(page),
        candidate={"confidence": 0.96},
    )
    first = attach_owner_visual_profile(
        {"owner_id": "owner_a", "coordinate_space": "logical_page"}, profile
    )
    divergent = copy.deepcopy(profile)
    divergent["source_sha256"] = "0" * 64
    divergent["visual_profile_sha256"] = owner_visual_profile_sha256(divergent)
    second = attach_owner_visual_profile(
        {"owner_id": "owner_a", "coordinate_space": "logical_page"}, divergent
    )

    with pytest.raises(ValueError, match="divergent duplicates"):
        renderer_mod._build_owner_render_blocks([first, second], graph)


def test_tile_permutation_does_not_change_owner_profile() -> None:
    page, graph = _fixture()
    first = build_owner_visual_profiles(
        graph, page, glyph_masks_by_owner={"owner_a": _glyph_mask(page)}
    )["owner_a"]
    permuted = copy.deepcopy(graph)
    permuted.projections.reverse()
    permuted.observations[0] = replace(
        permuted.observations[0],
        tile_provenance=tuple(reversed(permuted.observations[0].tile_provenance)),
    )
    second = build_owner_visual_profiles(
        permuted, page, glyph_masks_by_owner={"owner_a": _glyph_mask(page)}
    )["owner_a"]

    assert first == second


def test_translation_page_never_serializes_visual_profile_fields() -> None:
    _page, graph = _fixture()
    graph.owners[0].translated_payload = None
    graph.owners[0].state = "execution_planned"
    payload = owners_to_translation_page(graph)

    forbidden = {"visual_profile_v2", "visual_profile_sha256", "style_copy_status"}
    assert forbidden.isdisjoint(payload)
    assert all(forbidden.isdisjoint(record) for record in payload["texts"])
