"""Systemic contracts for page-global semantic owner reconciliation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.model import SourceTextComponent, TextObservation  # noqa: E402
from ownership.reconcile import SemanticRegion, build_page_owner_graph  # noqa: E402


def _component(component_id: str, bbox: tuple[int, int, int, int]) -> SourceTextComponent:
    x1, y1, x2, y2 = bbox
    return SourceTextComponent(
        component_id=component_id,
        page_id="page_001",
        bbox_page=bbox,
        polygon_page=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
        detector_sources=("region_detector",),
    )


def _observation(
    observation_id: str,
    component_ids: tuple[str, ...],
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    confidence: float = 0.9,
    tile_provenance: tuple[str, ...] = ("tile_a",),
    coverage_score: float | None = None,
    language_score: float | None = None,
) -> TextObservation:
    return TextObservation(
        observation_id=observation_id,
        page_id="page_001",
        component_ids=component_ids,
        text=text,
        confidence=confidence,
        provider="test_ocr",
        bbox_page=bbox,
        tile_provenance=tile_provenance,
        coverage_score=coverage_score,
        language_score=language_score,
    )


def _owner_signature(graph):
    return [
        {
            "component_ids": tuple(owner.component_ids),
            "selected": tuple(owner.selected_observation_ids),
            "role": owner.semantic_role,
            "payload": owner.source_payload,
            "disposition": owner.disposition,
        }
        for owner in sorted(graph.owners, key=lambda item: item.owner_id)
    ]


def test_one_body_seen_in_multiple_tiles_has_one_owner() -> None:
    components = [
        _component("line_top", (100, 100, 300, 140)),
        _component("line_bottom", (110, 150, 290, 190)),
    ]
    observations = [
        _observation(
            "body_full",
            ("line_top", "line_bottom"),
            "ONE COMPLETE BODY",
            (100, 100, 300, 190),
            tile_provenance=("tile_1", "tile_2", "tile_3", "tile_4"),
        )
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=observations,
        semantic_regions=[SemanticRegion("body", ("line_top", "line_bottom"), "body")],
    )

    assert len(graph.owners) == 1
    assert graph.owners[0].component_ids == ["line_top", "line_bottom"]
    assert graph.owners[0].selected_observation_ids == ["body_full"]
    assert graph.owners[0].source_payload == "ONE COMPLETE BODY"


def test_partial_observation_never_replaces_full_coverage() -> None:
    components = [
        _component("line_top", (100, 100, 300, 140)),
        _component("line_bottom", (110, 150, 290, 190)),
    ]
    observations = [
        _observation(
            "partial_high_confidence",
            ("line_bottom",),
            "BODY",
            (110, 150, 290, 190),
            confidence=0.99,
            coverage_score=1.0,
            language_score=1.0,
        ),
        _observation(
            "complete_lower_confidence",
            ("line_top", "line_bottom"),
            "ONE COMPLETE BODY",
            (100, 100, 300, 190),
            confidence=0.72,
        ),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=observations,
        semantic_regions=[SemanticRegion("body", ("line_top", "line_bottom"), "body")],
    )

    assert graph.owners[0].selected_observation_ids == ["complete_lower_confidence"]
    assert graph.owners[0].source_payload == "ONE COMPLETE BODY"
    partial = next(item for item in graph.observations if item.observation_id == "partial_high_confidence")
    assert partial.rejection_reason == "dominated_subcoverage"


def test_equally_supported_divergent_full_readings_require_review() -> None:
    component = _component("body", (100, 100, 300, 160))
    observations = [
        _observation("reading_a", ("body",), "FIRST READING", component.bbox_page),
        _observation("reading_b", ("body",), "SECOND READING", component.bbox_page),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=observations,
        semantic_regions=[SemanticRegion("body", ("body",), "body")],
    )

    assert graph.owners[0].disposition == "review"
    assert graph.owners[0].state == "review_required"
    assert graph.owners[0].selected_observation_ids == []


def test_ocr_ready_graph_does_not_require_executor_before_execution_planning() -> None:
    component = _component("body", (100, 100, 300, 160))
    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=[_observation("reading", ("body",), "SOURCE", component.bbox_page)],
        semantic_regions=[SemanticRegion("body", ("body",), "body")],
    )

    assert graph.owners[0].state == "ocr_ready"
    assert graph.validate() == []


def test_overlapping_partial_component_sets_require_review() -> None:
    components = [
        _component("a", (100, 100, 300, 140)),
        _component("b", (100, 150, 300, 190)),
        _component("c", (100, 200, 300, 240)),
    ]
    observations = [
        _observation("ab", ("a", "b"), "FIRST SHARED BODY", (100, 100, 300, 190)),
        _observation("bc", ("b", "c"), "SHARED BODY END", (100, 150, 300, 240)),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=observations,
        semantic_regions=[SemanticRegion("body", ("a", "b", "c"), "body")],
    )

    assert graph.owners[0].disposition == "review"
    assert graph.owners[0].selected_observation_ids == []
    assert {
        item.rejection_reason for item in graph.observations
    } == {"ambiguous_component_overlap"}


def test_cross_semantic_observation_blocks_both_affected_regions() -> None:
    components = [
        _component("title", (100, 100, 300, 140)),
        _component("body", (100, 200, 300, 240)),
    ]
    observations = [
        _observation("title_clean", ("title",), "TITLE", (100, 100, 300, 140)),
        _observation("body_clean", ("body",), "BODY", (100, 200, 300, 240)),
        _observation(
            "cross_role",
            ("title", "body"),
            "TITLE BODY",
            (100, 100, 300, 240),
        ),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=observations,
        semantic_regions=[
            SemanticRegion("title", ("title",), "title"),
            SemanticRegion("body", ("body",), "body"),
        ],
    )

    assert {owner.disposition for owner in graph.owners} == {"review"}
    reasons = {item.observation_id: item.rejection_reason for item in graph.observations}
    assert reasons == {
        "body_clean": "owner_blocked_by_cross_semantic_observation",
        "cross_role": "cross_semantic_region",
        "title_clean": "owner_blocked_by_cross_semantic_observation",
    }


def test_unassociated_geometry_cannot_be_selected_into_multiple_regions() -> None:
    components = [
        _component("left", (100, 100, 220, 160)),
        _component("right", (200, 100, 320, 160)),
    ]
    observation = _observation(
        "geometry_only",
        (),
        "AMBIGUOUS",
        (100, 100, 320, 160),
    )

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=[observation],
        semantic_regions=[
            SemanticRegion("left", ("left",), "body"),
            SemanticRegion("right", ("right",), "body"),
        ],
    )

    assert {owner.disposition for owner in graph.owners} == {"review"}
    assert graph.observations[0].rejection_reason == "unassociated_observation"


def test_identical_text_in_distant_geometry_has_two_owners() -> None:
    components = [
        _component("top", (20, 20, 120, 60)),
        _component("bottom", (500, 900, 600, 940)),
    ]
    observations = [
        _observation("top_ocr", ("top",), "SAME", (20, 20, 120, 60)),
        _observation("bottom_ocr", ("bottom",), "SAME", (500, 900, 600, 940)),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=observations,
        semantic_regions=[
            SemanticRegion("top_region", ("top",), "body"),
            SemanticRegion("bottom_region", ("bottom",), "body"),
        ],
    )

    assert len(graph.owners) == 2
    assert {tuple(owner.component_ids) for owner in graph.owners} == {("top",), ("bottom",)}
    assert len({owner.owner_id for owner in graph.owners}) == 2


def test_card_body_lines_stay_atomic_while_roles_stay_distinct() -> None:
    components = [
        _component("title", (100, 100, 300, 140)),
        _component("body_top", (80, 200, 320, 240)),
        _component("body_bottom", (80, 250, 320, 290)),
        _component("footer", (120, 360, 280, 390)),
    ]
    observations = [
        _observation("title_ocr", ("title",), "TITLE", (100, 100, 300, 140)),
        _observation(
            "body_ocr",
            ("body_top", "body_bottom"),
            "COMPLETE EFFECT DESCRIPTION",
            (80, 200, 320, 290),
        ),
        _observation("footer_ocr", ("footer",), "FOOTER", (120, 360, 280, 390)),
    ]
    regions = [
        SemanticRegion("card_title", ("title",), "title"),
        SemanticRegion("card_body", ("body_top", "body_bottom"), "body"),
        SemanticRegion("card_footer", ("footer",), "footer"),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=observations,
        semantic_regions=regions,
    )

    assert {owner.semantic_role for owner in graph.owners} == {"title", "body", "footer"}
    body = next(owner for owner in graph.owners if owner.semantic_role == "body")
    assert body.component_ids == ["body_top", "body_bottom"]
    assert body.source_payload == "COMPLETE EFFECT DESCRIPTION"
    assert isinstance(body.source_payload, str)


def test_connected_layout_lobes_do_not_force_multiple_translations() -> None:
    components = [
        _component("lobe_top", (100, 100, 300, 140)),
        _component("lobe_bottom", (110, 220, 290, 260)),
    ]
    observation = _observation(
        "connected_body",
        ("lobe_top", "lobe_bottom"),
        "ONE SEMANTIC BODY",
        (100, 100, 300, 260),
    )

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=[observation],
        semantic_regions=[SemanticRegion("connected", ("lobe_top", "lobe_bottom"), "body")],
    )

    assert len(graph.owners) == 1
    assert graph.owners[0].source_payload == "ONE SEMANTIC BODY"


def test_partial_fragments_form_one_atomic_owner_payload() -> None:
    components = [
        _component("line_top", (100, 100, 300, 140)),
        _component("line_bottom", (110, 150, 290, 190)),
    ]
    observations = [
        _observation("fragment_bottom", ("line_bottom",), "SECOND LINE", (110, 150, 290, 190)),
        _observation("fragment_top", ("line_top",), "FIRST LINE", (100, 100, 300, 140)),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=observations,
        semantic_regions=[SemanticRegion("body", ("line_top", "line_bottom"), "body")],
    )

    assert len(graph.owners) == 1
    assert graph.owners[0].selected_observation_ids == ["fragment_top", "fragment_bottom"]
    assert graph.owners[0].source_payload == "FIRST LINE SECOND LINE"


def test_every_evidence_is_owned_preserved_suppressed_or_review() -> None:
    components = [
        _component("owned", (10, 10, 100, 50)),
        _component("preserved", (10, 80, 100, 120)),
        _component("suppressed", (10, 150, 100, 190)),
        _component("unresolved", (10, 220, 100, 260)),
    ]
    observations = [_observation("owned_ocr", ("owned",), "SOURCE", (10, 10, 100, 50))]
    regions = [
        SemanticRegion("owned_region", ("owned",), "body"),
        SemanticRegion("preserve_region", ("preserved",), "sfx", disposition="preserve"),
        SemanticRegion("suppress_region", ("suppressed",), "credit", disposition="suppress"),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=observations,
        semantic_regions=regions,
    )

    decisions = {item.component_id: item.decision for item in graph.component_dispositions}
    assert decisions == {
        "owned": "owned",
        "preserved": "preserve",
        "suppressed": "suppress",
        "unresolved": "review",
    }
    unresolved_owner = next(owner for owner in graph.owners if owner.component_ids == ["unresolved"])
    assert unresolved_owner.state == "review_required"
    assert unresolved_owner.route_action == "review_required"


def test_manifest_cases_resolve_without_production_specific_exceptions() -> None:
    manifest_path = (
        Path(__file__).parent / "fixtures" / "systemic_visual_contracts" / "owner_cases.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for case in manifest["cases"]:
        components = [
            SourceTextComponent(
                component_id=item["component_id"],
                page_id="page_001",
                bbox_page=tuple(item["bbox"]),
                polygon_page=tuple(tuple(point) for point in item["polygon"]),
                detector_sources=tuple(item["detector_sources"]),
            )
            for item in case["source_components"]
        ]
        observations = [
            _observation(
                item["observation_id"],
                tuple(item["component_ids"]),
                item["text"],
                tuple(item["bbox"]),
                confidence=item["confidence"],
                tile_provenance=tuple(item["tile_provenance"]),
            )
            for item in case["observations"]
        ]
        component_ids = [item.component_id for item in components]
        if case["category"] == "card_semantic_roles":
            regions = [
                SemanticRegion("title", ("component_title",), "title"),
                SemanticRegion(
                    "body",
                    ("component_body_top", "component_body_bottom"),
                    "body",
                ),
                SemanticRegion("footer", ("component_footer",), "footer"),
            ]
        elif case["category"] == "repeated_text_distinct_geometry":
            regions = [
                SemanticRegion(f"region_{index}", (component_id,), "body")
                for index, component_id in enumerate(component_ids)
            ]
        else:
            regions = [SemanticRegion("body", tuple(component_ids), "body")]

        graph = build_page_owner_graph(
            page_id="page_001",
            components=components,
            observations=observations,
            semantic_regions=regions,
        )

        assert len(graph.owners) == case["expected"]["owner_count"], case["case_id"]
        assert len(graph.component_dispositions) == len(components), case["case_id"]
        assert all(item.decision == "owned" for item in graph.component_dispositions), case["case_id"]
        if "selected_observation_ids" in case["expected"]:
            assert graph.owners[0].selected_observation_ids == case["expected"][
                "selected_observation_ids"
            ]
