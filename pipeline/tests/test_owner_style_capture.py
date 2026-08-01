"""Owner-scoped style eligibility captured before translation or inpaint."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ownership.model import (
    ComponentDisposition,
    OwnerGraph,
    SourceTextComponent,
    TextObservation,
    TextOwner,
)
from typesetter.style_capture import (
    build_owner_style_capture,
    build_style_capture_masks,
    validate_owner_style_capture,
)


SOURCE = np.full((32, 48, 3), 240, dtype=np.uint8)


def _graph(*confidences: float, route_action: str = "translate_inpaint_render") -> OwnerGraph:
    component = SourceTextComponent(
        component_id="component_a",
        page_id="page_001",
        bbox_page=(5, 6, 35, 24),
        polygon_page=((5, 6), (35, 6), (35, 24), (5, 24)),
        detector_sources=("independent_text_recall",),
    )
    observations = [
        TextObservation(
            observation_id=f"obs_{index}",
            page_id="page_001",
            component_ids=(component.component_id,),
            text="SOURCE",
            confidence=confidence,
            provider="paddle_full_page",
            bbox_page=component.bbox_page,
            polygons_page=(component.polygon_page,),
        )
        for index, confidence in enumerate(confidences, start=1)
    ]
    owner = TextOwner(
        owner_id="owner_a",
        page_id="page_001",
        component_ids=[component.component_id],
        observation_ids=[item.observation_id for item in observations],
        selected_observation_ids=[item.observation_id for item in observations],
        semantic_role="sfx" if route_action == "translate_sfx_inpaint_render" else "dialogue_body",
        source_payload="SOURCE",
        translated_payload=None,
        disposition="owned",
        state="ocr_ready",
        route_action=route_action,
        execution_tile_id=None,
    )
    return OwnerGraph(
        schema_version=1,
        page_id="page_001",
        components=[component],
        observations=observations,
        owners=[owner],
        projections=[],
        component_dispositions=[
            ComponentDisposition(component.component_id, "owned", owner.owner_id, "resolved")
        ],
    )


def test_primary_owner_capture_derives_confidence_from_selected_observations() -> None:
    capture = build_owner_style_capture(_graph(0.91, 0.84), "owner_a", SOURCE)

    assert capture.candidate_kind == "primary_ocr"
    assert capture.candidate_confidence == pytest.approx(0.91)
    assert capture.candidate_confidence_provenance == ("obs_1", "obs_2")
    assert capture.eligible is True
    assert validate_owner_style_capture(capture.to_dict()) == capture


def test_capture_is_invariant_to_observation_order() -> None:
    graph = _graph(0.84, 0.91)
    first = build_owner_style_capture(graph, "owner_a", SOURCE)
    graph.observations.reverse()
    graph.owners[0].selected_observation_ids.reverse()

    assert build_owner_style_capture(graph, "owner_a", SOURCE) == first


def test_capture_cannot_change_semantic_owner_graph() -> None:
    graph = _graph(0.91)
    semantic_before = graph.to_dict()

    build_owner_style_capture(graph, "owner_a", SOURCE)

    assert graph.to_dict() == semantic_before


@pytest.mark.parametrize("confidence", [None, float("nan"), float("inf")])
def test_missing_or_non_finite_primary_confidence_is_ineligible(confidence) -> None:
    graph = _graph(0.91)
    graph.observations[0] = replace(graph.observations[0], confidence=confidence)

    capture = build_owner_style_capture(graph, "owner_a", SOURCE)

    assert capture.candidate_confidence is None
    assert capture.eligible is False


def test_promoted_sfx_requires_explicit_promotion_provenance() -> None:
    graph = _graph(0.88, route_action="translate_sfx_inpaint_render")

    raw = build_owner_style_capture(graph, "owner_a", SOURCE)
    promoted = build_owner_style_capture(
        graph,
        "owner_a",
        SOURCE,
        promotion_status="promoted",
        promotion_confidence=0.88,
        promotion_provenance=("sfx_visual:record_7",),
    )

    assert raw.eligible is False
    assert promoted.eligible is True
    assert promoted.candidate_kind == "promoted_sfx"
    assert promoted.promotion_provenance == ("sfx_visual:record_7",)


def test_capture_hash_rejects_tampering() -> None:
    payload = build_owner_style_capture(_graph(0.91), "owner_a", SOURCE).to_dict()
    payload["candidate_confidence"] = 0.99

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_owner_style_capture(payload)


def test_capture_extracts_fill_from_authoritative_owner_masks() -> None:
    graph = _graph(0.93)
    source = np.full_like(SOURCE, (245, 245, 245))
    glyph = np.zeros(source.shape[:2], dtype=np.uint8)
    glyph[11:18, 12:28] = 255
    source[glyph > 0] = (18, 70, 190)

    capture = build_owner_style_capture(
        graph,
        "owner_a",
        source,
        glyph_mask=glyph,
    )

    assert capture.style_evidence_v2.attributes["fill"].value == "#1246BE"
    assert capture.style_evidence_v2.attribute_provenance["fill"]["masks"] == (
        "owner_glyph_core",
    )
    assert capture.glyph_mask_sha256 != capture.context_mask_sha256


def test_context_mask_excludes_foreign_owner_and_protected_art() -> None:
    graph = _graph(0.93)
    glyph = np.zeros(SOURCE.shape[:2], dtype=np.uint8)
    glyph[11:18, 12:28] = 255
    foreign = np.zeros_like(glyph)
    foreign[8:14, 28:34] = 255
    protected = np.zeros_like(glyph)
    protected[18:23, 8:15] = 255

    masks = build_style_capture_masks(
        graph,
        "owner_a",
        SOURCE.shape[:2],
        glyph_mask=glyph,
        foreign_owner_mask=foreign,
        protected_art_mask=protected,
    )

    assert not np.any((masks.context_mask > 0) & (foreign > 0))
    assert not np.any((masks.context_mask > 0) & (protected > 0))
    assert not np.any((masks.context_mask > 0) & (masks.glyph_core_mask > 0))
    assert not np.any((masks.effect_region_mask > 0) & (foreign > 0))


@pytest.mark.parametrize("pixel_count", [0, 4])
def test_empty_or_too_small_glyph_mask_fails_closed(pixel_count: int) -> None:
    glyph = np.zeros(SOURCE.shape[:2], dtype=np.uint8)
    glyph.flat[:pixel_count] = 255

    with pytest.raises(ValueError, match="insufficient"):
        build_owner_style_capture(_graph(0.93), "owner_a", SOURCE, glyph_mask=glyph)
