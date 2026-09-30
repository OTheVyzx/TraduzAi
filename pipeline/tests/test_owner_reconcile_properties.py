"""Deterministic permutation properties for semantic owner reconciliation."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.hash_contract import sha256_text  # noqa: E402
from ownership.model import SourceTextComponent, TextObservation  # noqa: E402
from ownership.reconcile import SemanticRegion, build_page_owner_graph  # noqa: E402


def _component(component_id: str, y1: int) -> SourceTextComponent:
    bbox = (100, y1, 300, y1 + 40)
    return SourceTextComponent(
        component_id=component_id,
        page_id="page_001",
        bbox_page=bbox,
        polygon_page=((100, y1), (300, y1), (300, y1 + 40), (100, y1 + 40)),
        detector_sources=("synthetic",),
    )


def _observation(
    observation_id: str,
    component_ids: tuple[str, ...],
    text: str,
    bbox: tuple[int, int, int, int],
    confidence: float,
    tiles: tuple[str, ...],
) -> TextObservation:
    return TextObservation(
        observation_id=observation_id,
        page_id="page_001",
        component_ids=component_ids,
        text=text,
        confidence=confidence,
        provider="synthetic",
        bbox_page=bbox,
        tile_provenance=tiles,
        coverage_score=None,
        language_score=0.9,
        run_id="run-owner-reconcile-properties",
        origin_execution_id="execution-owner-reconcile-properties",
        invocation_id=f"invocation-{observation_id}",
        attempt_id=f"attempt-{observation_id}",
        provider_family="synthetic",
        page_source_sha256="a" * 64,
        root_input_pixel_sha256="b" * 64,
        input_pixel_sha256=sha256_text(f"pixels:{observation_id}"),
        payload_sha256=sha256_text(text),
    )


def _semantic_snapshot(graph) -> tuple:
    return tuple(
        (
            owner.owner_id,
            tuple(owner.component_ids),
            tuple(owner.observation_ids),
            tuple(owner.selected_observation_ids),
            owner.semantic_role,
            owner.source_payload,
            owner.disposition,
            owner.state,
            owner.route_action,
        )
        for owner in sorted(graph.owners, key=lambda item: item.owner_id)
    )


def test_graph_is_invariant_to_observation_component_and_tile_permutation() -> None:
    components = [_component("top", 100), _component("bottom", 160)]
    observations = [
        _observation(
            "full",
            ("bottom", "top"),
            "FULL BODY",
            (100, 100, 300, 200),
            0.82,
            ("tile_4", "tile_1", "tile_3", "tile_2"),
        ),
        _observation("top_only", ("top",), "FULL", (100, 100, 300, 140), 0.98, ("tile_1",)),
    ]
    regions = [SemanticRegion("body", ("bottom", "top"), "body")]
    expected = None

    for component_order in permutations(components):
        for observation_order in permutations(observations):
            permuted_observations = [
                replace(
                    observation,
                    component_ids=tuple(reversed(observation.component_ids)),
                    tile_provenance=tuple(reversed(observation.tile_provenance)),
                )
                for observation in observation_order
            ]
            graph = build_page_owner_graph(
                page_id="page_001",
                components=component_order,
                observations=permuted_observations,
                semantic_regions=regions,
            )
            snapshot = _semantic_snapshot(graph)
            expected = snapshot if expected is None else expected
            assert snapshot == expected


def test_graph_is_invariant_to_one_two_or_four_tile_partition() -> None:
    components = [_component("top", 100), _component("bottom", 160)]
    regions = [SemanticRegion("body", ("top", "bottom"), "body")]
    partitions = [
        ("tile_full",),
        ("tile_top", "tile_bottom"),
        ("tile_q1", "tile_q2", "tile_q3", "tile_q4"),
    ]
    snapshots = []

    for tiles in partitions:
        observation = _observation(
            "full",
            ("top", "bottom"),
            "FULL BODY",
            (100, 100, 300, 200),
            0.9,
            tiles,
        )
        graph = build_page_owner_graph(
            page_id="page_001",
            components=components,
            observations=[observation],
            semantic_regions=regions,
        )
        snapshots.append(_semantic_snapshot(graph))

    assert snapshots[0] == snapshots[1] == snapshots[2]


def test_complete_reading_selection_is_permutation_invariant() -> None:
    components = [_component("numeric_body", 100)]
    regions = [SemanticRegion("body", ("numeric_body",), "body")]
    observations = [
        _observation(
            "truncated",
            ("numeric_body",),
            "TOTAL PURCHASE AMOUNT",
            (100, 100, 300, 140),
            0.99,
            ("tile_top",),
        ),
        _observation(
            "complete_a",
            ("numeric_body",),
            "TOTAL PURCHASE AMOUNT 200MILLION",
            (95, 95, 305, 205),
            0.9,
            ("tile_top", "tile_bottom"),
        ),
        _observation(
            "complete_b",
            ("numeric_body",),
            "TOTAL PURCHASE AMOUNT 200 MILLION",
            (94, 94, 306, 206),
            0.88,
            ("tile_bottom", "tile_top"),
        ),
    ]
    expected = None

    for order in permutations(observations):
        graph = build_page_owner_graph(
            page_id="page_001",
            components=components,
            observations=order,
            semantic_regions=regions,
        )
        snapshot = _semantic_snapshot(graph)
        expected = snapshot if expected is None else expected
        assert snapshot == expected
        assert graph.owners[0].source_payload.replace(" ", "").endswith("200MILLION")


def test_every_input_order_produces_exactly_one_final_component_disposition() -> None:
    components = [_component("a", 100), _component("b", 180), _component("c", 260)]
    observation = _observation("a_ocr", ("a",), "A", (100, 100, 300, 140), 0.9, ("tile",))
    regions = [
        SemanticRegion("owned", ("a",), "body"),
        SemanticRegion("preserved", ("b",), "sfx", disposition="preserve"),
    ]

    for order in permutations(components):
        graph = build_page_owner_graph(
            page_id="page_001",
            components=order,
            observations=[observation],
            semantic_regions=regions,
        )
        decisions = [item.component_id for item in graph.component_dispositions]
        assert sorted(decisions) == ["a", "b", "c"]
        assert len(decisions) == len(set(decisions))
