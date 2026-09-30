from dataclasses import replace
from hashlib import sha256

import pytest

from layout.balloon_layout import _region_supports_shared_layout, enrich_page_layout
from ownership.hash_contract import sha256_text
from ownership.model import (
    OWNER_GRAPH_SCHEMA_VERSION,
    ComponentDisposition,
    OwnerGraph,
    OwnerProjection,
    SourceTextComponent,
    TextObservation,
    TextOwner,
)


def _verified_owner_graph() -> OwnerGraph:
    component = SourceTextComponent(
        component_id="component_body",
        page_id="page_001",
        bbox_page=(8, 8, 48, 24),
        polygon_page=((8, 8), (48, 8), (48, 24), (8, 24)),
        detector_sources=("independent_text_recall",),
    )
    observation = TextObservation(
        observation_id="observation_body",
        page_id="page_001",
        component_ids=(component.component_id,),
        text="SOURCE BODY",
        confidence=0.96,
        provider="paddle_full_page",
        bbox_page=component.bbox_page,
        tile_provenance=("tile_executor",),
        run_id="run-balloon-layout",
        origin_execution_id="execution-balloon-layout",
        invocation_id="invocation-balloon-layout-body",
        attempt_id="attempt-balloon-layout-body",
        provider_family="paddle",
        page_source_sha256="a" * 64,
        root_input_pixel_sha256="b" * 64,
        input_pixel_sha256="c" * 64,
        payload_sha256=sha256_text("SOURCE BODY"),
    )
    owner = TextOwner(
        owner_id="owner_body",
        page_id="page_001",
        component_ids=[component.component_id],
        observation_ids=[observation.observation_id],
        selected_observation_ids=[observation.observation_id],
        semantic_role="dialogue_body",
        source_payload="SOURCE BODY",
        translated_payload="CORPO COMPLETO",
        disposition="owned",
        state="inpainted",
        route_action="translate_inpaint_render",
        execution_tile_id="tile_executor",
        action_mask_ref=(
            "owner_masks/owner_body--"
            f"{sha256(b'owner_body').hexdigest()[:12]}/"
            "tile_executor/action_mask.png"
        ),
    )
    graph = OwnerGraph(
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id="page_001",
        run_id="run-balloon-layout",
        origin_execution_id="execution-balloon-layout",
        page_source_sha256="a" * 64,
        components=[component],
        observations=[observation],
        owners=[owner],
        projections=[
            OwnerProjection(
                owner_id=owner.owner_id,
                tile_id="tile_executor",
                role="executor",
                bbox_page=component.bbox_page,
                bbox_tile=component.bbox_page,
                offset_xy=(0, 0),
            )
        ],
        component_dispositions=[
            ComponentDisposition(
                component_id=component.component_id,
                decision="owned",
                owner_id=owner.owner_id,
            )
        ],
    )
    graph.require_valid()
    return graph


def test_shared_layout_rejects_distinct_bubble_mask_regions():
    region = {
        "texts": [
            {
                "bbox": [140, 120, 300, 150],
                "bubble_mask_bbox": [80, 80, 340, 190],
            },
            {
                "bbox": [360, 310, 520, 360],
                "bubble_mask_bbox": [300, 260, 580, 430],
            },
        ]
    }

    assert _region_supports_shared_layout(region, "text") is False


def test_shared_layout_allows_same_bubble_stack():
    region = {
        "texts": [
            {
                "bbox": [180, 120, 300, 150],
                "bubble_mask_bbox": [100, 80, 380, 240],
            },
            {
                "bbox": [184, 156, 298, 190],
                "bubble_mask_bbox": [100, 80, 380, 240],
            },
        ]
    }

    assert _region_supports_shared_layout(region, "text") is True


def test_verified_owner_layout_uses_explicit_regions_without_mask_grouping(monkeypatch):
    graph = _verified_owner_graph()
    owner = graph.owners[0]
    page = {
        "page_id": "page_001",
        "width": 320,
        "height": 240,
        "texts": [
            {
                "owner_id": "owner_body",
                "translated": "CORPO COMPLETO",
                "route_action": "translate_inpaint_render",
                "action_mask_ref": owner.action_mask_ref,
                "bbox": [8, 8, 48, 24],
            }
        ],
    }
    regions = [
        {
            "layout_region_id": "region_body",
            "owner_id": "owner_body",
            "order": 0,
            "bbox_page": [80, 50, 280, 190],
            "safe_polygon_page": [[80, 50], [280, 50], [280, 190], [80, 190]],
            "source_font_bounds_px": [18, 30],
            "container_font_bounds_px": [14, 26],
        }
    ]

    monkeypatch.setattr(
        "layout.balloon_layout.build_mask_regions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy mask grouping ran in verified owner mode")
        ),
    )

    enriched = enrich_page_layout(page, owner_graph=graph, layout_regions=regions)

    assert len(enriched["texts"]) == 1
    assert enriched["texts"][0]["owner_id"] == "owner_body"
    assert enriched["texts"][0]["translated"] == "CORPO COMPLETO"
    assert enriched["texts"][0]["layout_region_ids"] == ["region_body"]
    assert enriched["texts"][0]["safe_text_box"] == [80, 50, 280, 190]


def test_verified_owner_layout_preserves_sealed_style_v2_envelope():
    graph = _verified_owner_graph()
    page = {
        "page_id": "page_001",
        "width": 320,
        "height": 240,
        "texts": [
            {
                "owner_id": "owner_body",
                "visual_profile_v2": {"profile": "sealed"},
                "visual_profile_sha256": "a" * 64,
                "style_copy_status": "applied",
                "style_group_resolution_v3": {"resolution": "sealed"},
                "style_resolved_intent_v1": {"intent": "sealed"},
            }
        ],
    }
    region = {
        "layout_region_id": "region_body",
        "owner_id": "owner_body",
        "order": 0,
        "bbox_page": [80, 50, 280, 190],
        "safe_polygon_page": [[80, 50], [280, 50], [280, 190], [80, 190]],
    }

    record = enrich_page_layout(
        page,
        owner_graph=graph,
        layout_regions=[region],
    )["texts"][0]

    assert record["style_group_resolution_v3"] == {"resolution": "sealed"}
    assert record["style_resolved_intent_v1"] == {"intent": "sealed"}


def test_verified_owner_layout_rejects_unvalidated_graph_object():
    class ForgedGraph:
        page_id = "page_001"
        owners = []
        components = []

    with pytest.raises(TypeError, match="OwnerGraph"):
        enrich_page_layout(
            {"page_id": "page_001", "width": 320, "height": 240, "texts": []},
            owner_graph=ForgedGraph(),
            layout_regions=[],
        )


@pytest.mark.parametrize(
    "safe_polygon",
    [
        None,
        [[80, 50], [120, 50], [160, 50]],
        [[70, 40], [290, 40], [290, 200], [70, 200]],
        [[80.5, 50], [280, 50], [280, 190], [80, 190]],
        [[True, 50], [280, 50], [280, 190], [80, 190]],
    ],
)
def test_verified_owner_layout_rejects_noncanonical_safe_polygon(safe_polygon):
    graph = _verified_owner_graph()
    region = {
        "layout_region_id": "region_body",
        "owner_id": "owner_body",
        "order": 0,
        "bbox_page": [80, 50, 280, 190],
        "safe_polygon_page": safe_polygon,
    }

    with pytest.raises(ValueError, match="safe_polygon"):
        enrich_page_layout(
            {"page_id": "page_001", "width": 320, "height": 240, "texts": []},
            owner_graph=graph,
            layout_regions=[region],
        )


def test_verified_render_owner_never_falls_back_to_component_geometry():
    with pytest.raises(ValueError, match="no explicit layout region"):
        enrich_page_layout(
            {"page_id": "page_001", "width": 320, "height": 240, "texts": []},
            owner_graph=_verified_owner_graph(),
            layout_regions=[],
        )


def test_connected_owner_requires_explicit_owner_safe_polygon():
    graph = _verified_owner_graph()
    regions = [
        {
            "layout_region_id": "region_top",
            "owner_id": "owner_body",
            "order": 0,
            "bbox_page": [80, 40, 280, 100],
            "safe_polygon_page": [[80, 40], [280, 40], [280, 100], [80, 100]],
        },
        {
            "layout_region_id": "region_bottom",
            "owner_id": "owner_body",
            "order": 1,
            "bbox_page": [80, 120, 280, 190],
            "safe_polygon_page": [[80, 120], [280, 120], [280, 190], [80, 190]],
        },
    ]

    with pytest.raises(ValueError, match="owner_safe_polygon_page"):
        enrich_page_layout(
            {"page_id": "page_001", "width": 320, "height": 240, "texts": []},
            owner_graph=graph,
            layout_regions=regions,
        )


def test_review_required_owner_is_not_materialized_as_renderable_text():
    graph = _verified_owner_graph()
    graph.owners[0] = replace(
        graph.owners[0],
        translated_payload=None,
        disposition="review",
        state="review_required",
        route_action="review_required",
        execution_tile_id=None,
        action_mask_ref=None,
    )
    graph.projections = []
    graph.component_dispositions = [
        ComponentDisposition(
            component_id="component_body",
            decision="review",
            owner_id="owner_body",
        )
    ]
    graph.require_valid()
    region = {
        "layout_region_id": "region_body",
        "owner_id": "owner_body",
        "order": 0,
        "bbox_page": [80, 50, 280, 190],
        "safe_polygon_page": [[80, 50], [280, 50], [280, 190], [80, 190]],
    }

    enriched = enrich_page_layout(
        {"page_id": "page_001", "width": 320, "height": 240, "texts": []},
        owner_graph=graph,
        layout_regions=[region],
    )

    assert enriched["texts"] == []
    assert enriched["_owner_layout_contract"]["blocked_owner_ids"] == ["owner_body"]


def test_verified_owner_layout_drops_legacy_render_controls():
    graph = _verified_owner_graph()
    page = {
        "page_id": "page_001",
        "width": 320,
        "height": 240,
        "texts": [
            {
                "owner_id": "owner_body",
                "translated": "CORPO COMPLETO",
                "visible": False,
                "skip_processing": True,
                "render_policy": "suppress",
                "route_reason": "legacy_heuristic",
            }
        ],
    }
    region = {
        "layout_region_id": "region_body",
        "owner_id": "owner_body",
        "order": 0,
        "bbox_page": [80, 50, 280, 190],
        "safe_polygon_page": [[80, 50], [280, 50], [280, 190], [80, 190]],
    }

    record = enrich_page_layout(
        page,
        owner_graph=graph,
        layout_regions=[region],
    )["texts"][0]

    for forbidden in ("visible", "skip_processing", "render_policy", "route_reason"):
        assert forbidden not in record


def test_connected_owner_preserves_source_scale_evidence_per_region():
    graph = _verified_owner_graph()
    owner_polygon = [[60, 30], [300, 30], [300, 210], [60, 210]]
    regions = [
        {
            "layout_region_id": "region_top",
            "owner_id": "owner_body",
            "order": 0,
            "bbox_page": [80, 50, 280, 105],
            "safe_polygon_page": [[80, 50], [280, 50], [280, 105], [80, 105]],
            "owner_safe_polygon_page": owner_polygon,
            "source_ink_heights_px": [18, 20],
            "source_x_heights_px": [12.6, 14.0],
            "source_ink_height_median_px": 19.0,
            "source_x_height_median_px": 13.3,
            "source_scale_evidence_confidence": 0.91,
            "source_scale_evidence_ids": ["obs_top:0", "obs_top:1"],
        },
        {
            "layout_region_id": "region_bottom",
            "owner_id": "owner_body",
            "order": 1,
            "bbox_page": [80, 125, 280, 190],
            "safe_polygon_page": [[80, 125], [280, 125], [280, 190], [80, 190]],
            "owner_safe_polygon_page": owner_polygon,
            "source_ink_heights_px": [34],
            "source_x_heights_px": [23.8],
            "source_ink_height_median_px": 34.0,
            "source_x_height_median_px": 23.8,
            "source_scale_evidence_confidence": 0.88,
            "source_scale_evidence_ids": ["obs_bottom:0"],
        },
    ]

    enriched = enrich_page_layout(
        {"page_id": "page_001", "width": 320, "height": 240, "texts": []},
        owner_graph=graph,
        layout_regions=regions,
    )

    preserved = enriched["texts"][0]["layout_regions"]
    assert preserved[0]["source_ink_heights_px"] == [18, 20]
    assert preserved[1]["source_ink_heights_px"] == [34]
    assert preserved[0]["source_scale_evidence_ids"] == ["obs_top:0", "obs_top:1"]
    assert preserved[1]["source_scale_evidence_ids"] == ["obs_bottom:0"]
