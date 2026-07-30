"""Systemic contracts for page-global semantic owner reconciliation."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
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


def test_selected_multiline_observation_expands_undercropped_component_geometry() -> None:
    components = [
        _component("line_top", (100, 130, 300, 170)),
        _component("line_bottom", (110, 180, 290, 220)),
    ]
    observation = replace(
        _observation(
            "body_full",
            ("line_top", "line_bottom"),
            "ONE COMPLETE BODY",
            (90, 90, 310, 220),
        ),
        polygons_page=(
            ((90, 90), (310, 90), (310, 125), (90, 125)),
            ((100, 135), (300, 135), (300, 165), (100, 165)),
            ((110, 185), (290, 185), (290, 215), (110, 215)),
        ),
    )

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=[observation],
        semantic_regions=[
            SemanticRegion("body", ("line_top", "line_bottom"), "body")
        ],
    )

    expanded = {component.component_id: component for component in graph.components}
    assert expanded["line_top"].bbox_page == (72, 72, 329, 184)
    assert max(point[0] for point in expanded["line_top"].polygon_page) == 328
    assert max(point[1] for point in expanded["line_top"].polygon_page) == 183
    assert expanded["line_bottom"].bbox_page == (92, 167, 309, 234)


def test_selected_ink_geometry_reserves_proportional_glyph_halo() -> None:
    component = _component("glowing_line", (100, 100, 300, 130))
    observation = replace(
        _observation(
            "glowing_ocr",
            (component.component_id,),
            "GLOWING SOURCE",
            component.bbox_page,
        ),
        polygons_page=(
            ((100, 100), (299, 100), (299, 129), (100, 129)),
        ),
    )

    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=[observation],
        semantic_regions=[SemanticRegion("body", (component.component_id,), "body")],
    )

    expanded = graph.components[0]
    assert expanded.bbox_page == (91, 91, 309, 139)


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


def test_complete_consensus_reading_beats_tighter_truncated_reading() -> None:
    component = _component("numeric_body", (100, 100, 300, 145))
    observations = [
        _observation(
            "truncated_tight",
            ("numeric_body",),
            "TOTAL PURCHASE AMOUNT",
            (100, 100, 300, 145),
            confidence=0.99,
        ),
        _observation(
            "complete_consensus_a",
            ("numeric_body",),
            "TOTAL PURCHASE AMOUNT 200MILLION",
            (95, 95, 305, 205),
            confidence=0.91,
        ),
        _observation(
            "complete_consensus_b",
            ("numeric_body",),
            "TOTAL PURCHASE AMOUNT 200 MILLION",
            (94, 94, 306, 206),
            confidence=0.88,
        ),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=observations,
        semantic_regions=[SemanticRegion("body", ("numeric_body",), "body")],
    )

    owner = graph.owners[0]
    assert owner.source_payload in {
        "TOTAL PURCHASE AMOUNT 200MILLION",
        "TOTAL PURCHASE AMOUNT 200 MILLION",
    }
    assert set(owner.selected_observation_ids) == {
        "complete_consensus_a",
        "complete_consensus_b",
    }
    assert graph.components[0].bbox_page[3] >= 206
    truncated = next(
        item for item in graph.observations if item.observation_id == "truncated_tight"
    )
    assert truncated.rejection_reason == "dominated_truncation"


def test_incompatible_non_dominated_readings_fail_to_review() -> None:
    component = _component("single_region", (100, 100, 300, 180))
    observations = [
        _observation(
            "reading_tight",
            ("single_region",),
            "TOTAL PURCHASE AMOUNT 200 MILLION",
            component.bbox_page,
            confidence=0.99,
        ),
        _observation(
            "reading_wide",
            ("single_region",),
            "PLAYER NAME KIM SIMUN",
            (90, 90, 310, 200),
            confidence=0.71,
        ),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=observations,
        semantic_regions=[SemanticRegion("body", ("single_region",), "body")],
    )

    assert graph.owners[0].state == "review_required"
    assert graph.owners[0].selected_observation_ids == []
    assert {
        item.rejection_reason for item in graph.observations
    } == {"ambiguous_reading"}


def test_legacy_rejection_reason_remains_evidence_not_destructive_filter() -> None:
    component = _component("body", (100, 100, 300, 180))
    observation = replace(
        _observation(
            "legacy_rejected_but_material",
            ("body",),
            "COMPLETE MATERIAL BODY",
            component.bbox_page,
        ),
        legacy_rejection_reason="cover_visual_art_ocr",
    )

    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=[observation],
        semantic_regions=[SemanticRegion("body", ("body",), "body")],
    )

    assert graph.owners[0].source_payload == "COMPLETE MATERIAL BODY"
    stored = graph.observations[0]
    assert stored.legacy_rejection_reason == "cover_visual_art_ocr"
    assert stored.rejection_reason is None


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


def test_cross_region_observation_with_same_role_does_not_veto_local_readings() -> None:
    components = [
        _component("line_top", (100, 100, 300, 140)),
        _component("line_bottom", (100, 160, 300, 200)),
    ]
    observations = [
        _observation("top_clean", ("line_top",), "FIRST LINE", (100, 100, 300, 140)),
        _observation(
            "bottom_clean",
            ("line_bottom",),
            "SECOND LINE",
            (100, 160, 300, 200),
        ),
        _observation(
            "macro_over_both",
            ("line_top", "line_bottom"),
            "FIRST LINE SECOND LINE",
            (100, 100, 300, 200),
        ),
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=components,
        observations=observations,
        semantic_regions=[
            SemanticRegion("top", ("line_top",), "dialogue_body"),
            SemanticRegion("bottom", ("line_bottom",), "dialogue_body"),
        ],
    )

    assert {owner.disposition for owner in graph.owners} == {"owned"}
    assert {owner.source_payload for owner in graph.owners} == {
        "FIRST LINE",
        "SECOND LINE",
    }
    reasons = {item.observation_id: item.rejection_reason for item in graph.observations}
    assert reasons["macro_over_both"] == "cross_region_same_role_observation"
    assert reasons["top_clean"] is None
    assert reasons["bottom_clean"] is None


def test_repeated_empty_ocr_overrides_detector_geometry_confidence() -> None:
    component = SourceTextComponent(
        component_id="art_false_positive",
        page_id="page_001",
        bbox_page=(100, 100, 360, 280),
        polygon_page=((100, 100), (360, 100), (360, 280), (100, 280)),
        detector_sources=("primary_region_detector",),
        confidence=0.95,
        script_evidence=(),
    )
    observations = [
        _observation(
            f"empty_attempt_{index}",
            (component.component_id,),
            "",
            component.bbox_page,
        )
        for index in range(3)
    ]

    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=observations,
        semantic_regions=[
            SemanticRegion("candidate", (component.component_id,), "dialogue_body")
        ],
    )

    assert graph.owners == []
    assert len(graph.component_dispositions) == 1
    assert graph.component_dispositions[0].decision == "suppress"
    assert graph.component_dispositions[0].reason == "multi_ocr_confirmed_non_text"


def test_empty_container_duplicating_resolved_text_component_is_suppressed() -> None:
    outer = _component("outer_container", (20, 20, 300, 220))
    inner = _component("inner_text", (80, 60, 240, 120))
    graph = build_page_owner_graph(
        page_id="page_001",
        components=[outer, inner],
        observations=[
            _observation(
                "inner_ocr",
                (inner.component_id,),
                "REAL TEXT",
                inner.bbox_page,
            )
        ],
        semantic_regions=[
            SemanticRegion(
                "outer_region",
                (outer.component_id,),
                "body",
                reason="semantic_container_missing",
            ),
            SemanticRegion("inner_region", (inner.component_id,), "body"),
        ],
    )

    assert [owner.source_payload for owner in graph.owners] == ["REAL TEXT"]
    decisions = {item.component_id: item for item in graph.component_dispositions}
    assert decisions[outer.component_id].decision == "suppress"
    assert decisions[outer.component_id].reason == "redundant_container_without_ocr_evidence"


def test_external_url_identifier_is_explicitly_preserved_not_translated() -> None:
    component = _component("scanlation_url", (20, 20, 180, 42))
    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=[
            _observation(
                "url_ocr",
                (component.component_id,),
                "www.example-scan.com",
                component.bbox_page,
                confidence=0.99,
            )
        ],
        semantic_regions=[
            SemanticRegion(
                "url_region",
                (component.component_id,),
                "dialogue_body",
            )
        ],
    )

    assert graph.owners == []
    assert graph.component_dispositions[0].decision == "preserve"
    assert graph.component_dispositions[0].reason == "policy:nontranslatable_external_identifier"


def test_detector_region_without_any_ocr_evidence_is_suppressed() -> None:
    component = _component("art_false_positive", (30, 30, 90, 70))
    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=[],
        semantic_regions=[
            SemanticRegion(
                "false_positive_region",
                (component.component_id,),
                "dialogue_body",
                reason="semantic_container_missing",
            )
        ],
    )

    assert graph.owners == []
    assert graph.component_dispositions[0].decision == "suppress"
    assert graph.component_dispositions[0].reason == "no_ocr_evidence_non_text"


def test_text_with_missing_semantic_container_remains_review_required() -> None:
    component = _component("uncontained_text", (30, 30, 180, 70))
    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=[
            _observation(
                "uncontained_ocr",
                (component.component_id,),
                "REAL TEXT",
                component.bbox_page,
            )
        ],
        semantic_regions=[
            SemanticRegion(
                "uncontained_region",
                (component.component_id,),
                "dialogue_body",
                reason="semantic_container_missing",
            )
        ],
    )

    assert graph.owners[0].disposition == "review"
    assert graph.component_dispositions[0].reason == "semantic_container_missing"


def test_scanlation_apparatus_below_page_marker_is_explicitly_preserved() -> None:
    dialogue = _component("dialogue", (100, 20, 300, 70))
    marker = _component("scan_marker", (80, 400, 340, 440))
    promo = _component("promo_title", (90, 520, 250, 560))
    graph = build_page_owner_graph(
        page_id="page_001",
        components=[dialogue, marker, promo],
        observations=[
            _observation("dialogue_ocr", (dialogue.component_id,), "ARE YOU READY?", dialogue.bbox_page),
            _observation("marker_ocr", (marker.component_id,), "SERIES WE SCANLATE", marker.bbox_page),
            _observation("promo_ocr", (promo.component_id,), "SUPER CUBE", promo.bbox_page),
        ],
        semantic_regions=[
            SemanticRegion("dialogue_region", (dialogue.component_id,), "dialogue_body"),
            SemanticRegion("marker_region", (marker.component_id,), "dialogue_body"),
            SemanticRegion("promo_region", (promo.component_id,), "dialogue_body"),
        ],
    )

    assert [owner.source_payload for owner in graph.owners] == ["ARE YOU READY?"]
    decisions = {item.component_id: item for item in graph.component_dispositions}
    assert decisions[marker.component_id].decision == "preserve"
    assert decisions[promo.component_id].decision == "preserve"
    assert decisions[promo.component_id].reason == "policy:scanlation_apparatus"


def test_scanlation_credit_page_uses_independent_contact_and_support_markers() -> None:
    contact = _component("contact", (20, 500, 240, 530))
    thanks = _component("thanks", (20, 620, 300, 650))
    credit = _component("credit_name", (20, 700, 260, 730))
    graph = build_page_owner_graph(
        page_id="page_001",
        components=[contact, thanks, credit],
        observations=[
            _observation(
                "contact_ocr",
                (contact.component_id,),
                "YOU CAN CONTACT US AT OUR SITE",
                contact.bbox_page,
            ),
            _observation(
                "thanks_ocr",
                (thanks.component_id,),
                "SPECIAL THANKS TO OUR PATREON",
                thanks.bbox_page,
            ),
            _observation(
                "credit_ocr",
                (credit.component_id,),
                "EDITOR NAME",
                credit.bbox_page,
            ),
        ],
        semantic_regions=[
            SemanticRegion("contact_region", (contact.component_id,), "body"),
            SemanticRegion("thanks_region", (thanks.component_id,), "body"),
            SemanticRegion("credit_region", (credit.component_id,), "body"),
        ],
    )

    assert graph.owners == []
    assert {
        item.reason for item in graph.component_dispositions
    } == {"policy:scanlation_apparatus"}


def test_scanlation_commission_marker_extends_apparatus_cutoff_upward() -> None:
    commission = _component("commission", (20, 300, 300, 330))
    promo_body = _component("promo_body", (20, 360, 300, 390))
    contact = _component("contact", (20, 600, 300, 630))
    graph = build_page_owner_graph(
        page_id="page_001",
        components=[commission, promo_body, contact],
        observations=[
            _observation(
                "commission_ocr",
                (commission.component_id,),
                "COMMISSION YOUR FAVORITE SERIES",
                commission.bbox_page,
            ),
            _observation(
                "promo_ocr",
                (promo_body.component_id,),
                "NEW SERIES AVAILABLE",
                promo_body.bbox_page,
            ),
            _observation(
                "contact_ocr",
                (contact.component_id,),
                "YOU CAN CONTACT US AT OUR SITE",
                contact.bbox_page,
            ),
        ],
        semantic_regions=[
            SemanticRegion("commission_region", (commission.component_id,), "body"),
            SemanticRegion("promo_region", (promo_body.component_id,), "body"),
            SemanticRegion("contact_region", (contact.component_id,), "body"),
        ],
    )

    assert graph.owners == []
    assert {
        item.reason for item in graph.component_dispositions
    } == {"policy:scanlation_apparatus"}


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
