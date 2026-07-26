from dataclasses import replace
from hashlib import sha256

import pytest

from layout.balloon_layout import _region_supports_shared_layout, enrich_page_layout
from ownership.model import (
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
        schema_version=1,
        page_id="page_001",
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
