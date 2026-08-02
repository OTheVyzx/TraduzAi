"""TDD contracts for owner-authoritative layout and typesetting.

The page-global owner is the only semantic unit in these tests.  Layout
regions and connected lobes may change visual flow, but they may not create,
merge, suppress, or reroute semantic payloads.
"""

from __future__ import annotations

import copy
from hashlib import sha256
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layout import balloon_layout as balloon_layout_mod  # noqa: E402
from ownership.delivery import seal_owner_text_execution_authority  # noqa: E402
from ownership.model import (  # noqa: E402
    ComponentDisposition,
    OwnerGraph,
    OwnerProjection,
    SourceTextComponent,
    TextObservation,
    TextOwner,
)
from ownership.render_geometry import build_owner_render_geometry  # noqa: E402
from typesetter import renderer as renderer_mod  # noqa: E402
from typesetter.owner_style import (  # noqa: E402
    attach_owner_visual_profile,
    build_owner_visual_profile,
    owner_visual_profile_sha256,
)


PAGE_ID = "page_001"
PAGE_WIDTH = 360
PAGE_HEIGHT = 280


def _action_mask_ref(owner_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner_id).strip("._")
    identity_hash = sha256(owner_id.encode("utf-8")).hexdigest()[:12]
    return f"owner_masks/{safe}--{identity_hash}/tile_executor/action_mask.png"


def _polygon_for_bbox(bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    x1, y1, x2, y2 = bbox
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _owner_graph(
    payloads: list[tuple[str, str, str]],
    *,
    shared_bbox: tuple[int, int, int, int] | None = None,
) -> OwnerGraph:
    components: list[SourceTextComponent] = []
    observations: list[TextObservation] = []
    owners: list[TextOwner] = []
    projections: list[OwnerProjection] = []
    dispositions: list[ComponentDisposition] = []

    for index, (owner_id, source_payload, translated_payload) in enumerate(payloads):
        bbox = shared_bbox or (24, 24 + index * 100, 176, 88 + index * 100)
        component_id = f"component_{index:03d}"
        observation_id = f"observation_{index:03d}"
        components.append(
            SourceTextComponent(
                component_id=component_id,
                page_id=PAGE_ID,
                bbox_page=bbox,
                polygon_page=_polygon_for_bbox(bbox),
                detector_sources=("independent_text_recall",),
                evidence_ids=(f"container_{index:03d}",),
            )
        )
        observations.append(
            TextObservation(
                observation_id=observation_id,
                page_id=PAGE_ID,
                component_ids=(component_id,),
                text=source_payload,
                confidence=0.96,
                provider="paddle_full_page",
                bbox_page=bbox,
                polygons_page=(_polygon_for_bbox(bbox),),
                tile_provenance=("tile_executor", "tile_context"),
                coverage_score=1.0,
            )
        )
        owners.append(
            TextOwner(
                owner_id=owner_id,
                page_id=PAGE_ID,
                component_ids=[component_id],
                observation_ids=[observation_id],
                selected_observation_ids=[observation_id],
                semantic_role="dialogue_body",
                source_payload=source_payload,
                translated_payload=translated_payload,
                disposition="owned",
                state="inpainted",
                route_action="translate_inpaint_render",
                execution_tile_id="tile_executor",
                action_mask_ref=_action_mask_ref(owner_id),
            )
        )
        projections.append(
            OwnerProjection(
                owner_id=owner_id,
                tile_id="tile_executor",
                role="executor",
                bbox_page=bbox,
                bbox_tile=bbox,
                offset_xy=(0, 0),
            )
        )
        dispositions.append(
            ComponentDisposition(
                component_id=component_id,
                decision="owned",
                owner_id=owner_id,
                reason="semantic_owner_resolved",
            )
        )

    graph = OwnerGraph(
        schema_version=1,
        page_id=PAGE_ID,
        components=components,
        observations=observations,
        owners=owners,
        projections=projections,
        component_dispositions=dispositions,
    )
    graph.require_valid()
    return graph


def _layout_region(
    owner_id: str,
    region_id: str,
    bbox: tuple[int, int, int, int],
    *,
    order: int = 0,
    safe_polygon: tuple[tuple[int, int], ...] | None = None,
    owner_safe_polygon: tuple[tuple[int, int], ...] | None = None,
) -> dict:
    polygon = safe_polygon or _polygon_for_bbox(bbox)
    region = {
        "layout_region_id": region_id,
        "owner_id": owner_id,
        "order": order,
        "bbox_page": list(bbox),
        "safe_polygon_page": [list(point) for point in polygon],
        "source_font_bounds_px": [18, 32],
        "container_font_bounds_px": [14, 28],
    }
    if owner_safe_polygon is not None:
        region["owner_safe_polygon_page"] = [
            list(point) for point in owner_safe_polygon
        ]
    return region


def _owner_page(
    graph: OwnerGraph,
    layout_regions: list[dict],
    *,
    style_evidence_by_owner: dict[str, dict | None] | None = None,
) -> dict:
    component_by_id = {
        component.component_id: component for component in graph.components
    }
    region_ids_by_owner: dict[str, list[str]] = {}
    for region in layout_regions:
        region_ids_by_owner.setdefault(str(region["owner_id"]), []).append(
            str(region["layout_region_id"])
        )

    texts = []
    for owner in graph.owners:
        component = component_by_id[owner.component_ids[0]]
        owner_regions = [
            region for region in layout_regions if region["owner_id"] == owner.owner_id
        ]
        owner_polygons = [
            region["owner_safe_polygon_page"]
            for region in owner_regions
            if region.get("owner_safe_polygon_page")
        ]
        if owner_polygons:
            container_polygon = owner_polygons[0]
        else:
            boxes = [tuple(region["bbox_page"]) for region in owner_regions]
            container_polygon = _polygon_for_bbox(
                (
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                )
            )
        render_geometry = build_owner_render_geometry(
            graph,
            owner.owner_id,
            page_width=PAGE_WIDTH,
            page_height=PAGE_HEIGHT,
            container_evidence={
                "evidence_id": f"{owner.owner_id}:fixture_container",
                "source": "balloon_inner_polygon",
                "polygon_page": container_polygon,
                "confidence": 1.0,
            },
            protected_art_mask_sha256=renderer_mod._owner_array_sha256(
                np.zeros((PAGE_HEIGHT, PAGE_WIDTH), dtype=np.uint8)
            ),
        )
        execution_authority = seal_owner_text_execution_authority(
            owner_id=owner.owner_id,
            page_id=owner.page_id,
            source_payload=owner.source_payload,
            translated_payload=str(owner.translated_payload or ""),
            normalized_chunks=[str(owner.translated_payload or "")],
        )
        for region in owner_regions:
            region["owner_render_geometry_sha256"] = render_geometry.geometry_sha256
        record = {
            "id": owner.owner_id,
            "owner_id": owner.owner_id,
            "page_id": owner.page_id,
            "coordinate_space": "logical_page",
            "text": owner.source_payload,
            "original": owner.source_payload,
            "translated": owner.translated_payload,
            "semantic_role": owner.semantic_role,
            "tipo": "fala",
            "route_action": owner.route_action,
            "action_mask_ref": owner.action_mask_ref,
            "component_ids": list(owner.component_ids),
            "observation_ids": list(owner.observation_ids),
            "selected_observation_ids": list(owner.selected_observation_ids),
            "layout_region_ids": region_ids_by_owner.get(owner.owner_id, []),
            "bbox": list(component.bbox_page),
            "source_bbox": list(component.bbox_page),
            "text_pixel_bbox": list(component.bbox_page),
            "page_width": PAGE_WIDTH,
            "page_height": PAGE_HEIGHT,
            "layout_profile": "white_balloon",
            "owner_render_geometry": render_geometry.to_dict(),
            "owner_render_geometry_sha256": render_geometry.geometry_sha256,
            "owner_text_execution_authority": execution_authority.to_dict(),
            "text_execution_authority_sha256": execution_authority.authority_sha256,
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 26,
                "cor": "#111111",
                "contorno": "",
                "contorno_px": 0,
                "alinhamento": "center",
            },
        }
        evidence = (style_evidence_by_owner or {}).get(owner.owner_id)
        if evidence is not None:
            record["style_evidence"] = copy.deepcopy(evidence)
        profile = build_owner_visual_profile(
            owner,
            np.full((PAGE_HEIGHT, PAGE_WIDTH, 3), 255, dtype=np.uint8),
            components=graph.components,
            observations=graph.observations,
            glyph_mask=np.zeros((PAGE_HEIGHT, PAGE_WIDTH), dtype=np.uint8),
            candidate=record,
        )
        texts.append(attach_owner_visual_profile(record, profile))

    return {
        "page_id": graph.page_id,
        "width": PAGE_WIDTH,
        "height": PAGE_HEIGHT,
        "texts": texts,
    }


def _owner_mode_blocks(
    graph: OwnerGraph,
    layout_regions: list[dict],
    *,
    style_evidence_by_owner: dict[str, dict | None] | None = None,
) -> tuple[dict, list[dict]]:
    page = _owner_page(
        graph,
        layout_regions,
        style_evidence_by_owner=style_evidence_by_owner,
    )
    enriched = balloon_layout_mod.enrich_page_layout(
        page,
        owner_graph=graph,
        layout_regions=layout_regions,
    )
    blocks = renderer_mod.build_render_blocks(
        enriched["texts"],
        owner_graph=graph,
    )
    return enriched, blocks


def _render_connected_owner_block(
    payload: str,
    layout_regions: list[dict],
) -> dict:
    """Render one connected owner without legacy OCR anchor geometry."""

    graph = _owner_graph([("owner_connected", "SOURCE BODY", payload)])
    page = _owner_page(graph, layout_regions)
    enriched = balloon_layout_mod.enrich_page_layout(
        page,
        owner_graph=graph,
        layout_regions=layout_regions,
    )
    block = renderer_mod.build_render_blocks(
        enriched["texts"],
        owner_graph=graph,
    )[0]
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (235, 235, 235))
    renderer_mod.render_text_block(canvas, block)
    renderer_mod._finalize_render_completion_contract(block)
    return block


def _normalize_payload(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _bbox_center(bbox: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _semantic_signature(record: dict) -> dict:
    return {
        "owner_id": record.get("owner_id"),
        "translated": record.get("translated"),
        "route_action": record.get("route_action"),
        "action_mask_ref": record.get("action_mask_ref"),
        "component_ids": record.get("component_ids"),
        "observation_ids": record.get("observation_ids"),
        "layout_region_ids": record.get("layout_region_ids"),
    }


def _owner_semantic_signature(graph: OwnerGraph) -> list[tuple]:
    return [
        (
            owner.owner_id,
            owner.source_payload,
            owner.translated_payload,
            owner.route_action,
            owner.action_mask_ref,
            tuple(owner.component_ids),
        )
        for owner in graph.owners
    ]


def test_renderer_receives_one_complete_payload_per_owner() -> None:
    graph = _owner_graph(
        [
            ("owner_alpha", "SOURCE BODY ALPHA", "CORPO COMPLETO ALFA"),
            ("owner_beta", "SOURCE BODY BETA", "CORPO COMPLETO BETA"),
        ]
    )
    regions = [
        _layout_region("owner_alpha", "region_alpha", (18, 18, 182, 94)),
        _layout_region("owner_beta", "region_beta", (18, 116, 182, 198)),
    ]

    _enriched, blocks = _owner_mode_blocks(graph, regions)

    assert {
        block["owner_id"]: block["translated"] for block in blocks
    } == {
        "owner_alpha": "CORPO COMPLETO ALFA",
        "owner_beta": "CORPO COMPLETO BETA",
    }
    assert len(blocks) == len(graph.owners)


def test_connected_owner_may_use_many_layout_regions_but_one_translation() -> None:
    payload = "PRIMEIRA FRASE COMPLETA. SEGUNDA FRASE CONTINUA A MESMA FALA."
    graph = _owner_graph([("owner_connected", "FULL SOURCE BODY", payload)])
    owner_safe_polygon = ((20, 16), (184, 16), (184, 194), (20, 194))
    regions = [
        _layout_region(
            "owner_connected",
            "region_connected_top",
            (24, 20, 168, 92),
            order=0,
            owner_safe_polygon=owner_safe_polygon,
        ),
        _layout_region(
            "owner_connected",
            "region_connected_bottom",
            (28, 106, 176, 190),
            order=1,
            owner_safe_polygon=owner_safe_polygon,
        ),
    ]

    enriched, blocks = _owner_mode_blocks(graph, regions)

    assert len(enriched["texts"]) == 1
    assert len(blocks) == 1
    assert blocks[0]["owner_id"] == "owner_connected"
    assert blocks[0]["translated"] == payload
    assert blocks[0]["layout_region_ids"] == [
        "region_connected_top",
        "region_connected_bottom",
    ]
    assert [region["layout_region_id"] for region in blocks[0]["layout_regions"]] == [
        "region_connected_top",
        "region_connected_bottom",
    ]


def test_connected_owner_uses_one_common_font_that_fits_every_region() -> None:
    payload = (
        "ESTE TEXTO LONGO CABE APENAS MENOR. "
        "OUTRA FRASE LONGA CABE APENAS MENOR."
    )
    owner_safe_polygon = ((10, 10), (190, 10), (190, 210), (10, 210))
    regions = [
        _layout_region(
            "owner_connected",
            "region_connected_top",
            (20, 20, 180, 100),
            order=0,
            owner_safe_polygon=owner_safe_polygon,
        ),
        _layout_region(
            "owner_connected",
            "region_connected_bottom",
            (20, 120, 180, 200),
            order=1,
            owner_safe_polygon=owner_safe_polygon,
        ),
    ]

    block = _render_connected_owner_block(payload, regions)

    chunks = block["visual_chunks"]
    common_sizes = {int(chunk["font_size"]) for chunk in chunks}
    child_bboxes = block["_render_debug"]["child_render_bboxes"]
    child_safe_boxes = block["_render_debug"]["child_safe_text_boxes"]
    assert len(common_sizes) == 1
    assert block["fit_status"] == "ok"
    assert block["render_completed"] is True
    assert len(child_bboxes) == len(child_safe_boxes) == len(regions)
    assert all(
        safe[0] <= rendered[0] < rendered[2] <= safe[2]
        and safe[1] <= rendered[1] < rendered[3] <= safe[3]
        for rendered, safe in zip(child_bboxes, child_safe_boxes, strict=True)
    )


def test_connected_owner_fails_closed_when_no_common_font_fits_bounds() -> None:
    payload = (
        "ESTE TEXTO LONGO CABE APENAS MENOR. "
        "OUTRA FRASE LONGA CABE APENAS MENOR."
    )
    owner_safe_polygon = ((10, 10), (190, 10), (190, 170), (10, 170))
    regions = [
        _layout_region(
            "owner_connected",
            "region_connected_top",
            (20, 20, 180, 72),
            order=0,
            owner_safe_polygon=owner_safe_polygon,
        ),
        _layout_region(
            "owner_connected",
            "region_connected_bottom",
            (20, 100, 180, 152),
            order=1,
            owner_safe_polygon=owner_safe_polygon,
        ),
    ]

    block = _render_connected_owner_block(payload, regions)

    assert block["fit_status"] != "ok"
    assert block["render_completed"] is False
    assert block["fit_status"] == "below_proportional_legibility"
    assert "fit_below_proportional_legibility" in set(block.get("qa_flags") or [])


def test_owner_glyph_patch_preserves_below_proportional_fit_status() -> None:
    payload = (
        "ESTE TEXTO LONGO CABE APENAS MENOR. "
        "OUTRA FRASE LONGA CABE APENAS MENOR."
    )
    graph = _owner_graph([("owner_connected", "SOURCE BODY", payload)])
    owner_safe_polygon = ((10, 10), (190, 10), (190, 170), (10, 170))
    regions = [
        _layout_region(
            "owner_connected",
            "region_connected_top",
            (20, 20, 180, 72),
            order=0,
            owner_safe_polygon=owner_safe_polygon,
        ),
        _layout_region(
            "owner_connected",
            "region_connected_bottom",
            (20, 100, 180, 152),
            order=1,
            owner_safe_polygon=owner_safe_polygon,
        ),
    ]
    enriched = balloon_layout_mod.enrich_page_layout(
        _owner_page(graph, regions),
        owner_graph=graph,
        layout_regions=regions,
    )

    glyph_patch = renderer_mod.render_band_image(
        np.full((PAGE_HEIGHT, PAGE_WIDTH, 3), 235, dtype=np.uint8),
        enriched,
        owner_graph=graph,
    )

    assert glyph_patch.glyph_bbox_page is None
    assert glyph_patch.render_completed is False
    assert glyph_patch.fit_status == "below_proportional_legibility"


def test_single_owner_render_ignores_legacy_ocr_anchor_and_centers_in_safe_polygon() -> None:
    graph = _owner_graph(
        [("owner_single", "TINY SOURCE", "CENTRO SEGURO")],
        shared_bbox=(8, 12, 58, 38),
    )
    safe_bbox = (150, 60, 330, 220)
    regions = [_layout_region("owner_single", "region_single", safe_bbox)]
    _enriched, blocks = _owner_mode_blocks(graph, regions)
    block = blocks[0]
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (235, 235, 235))

    renderer_mod.render_text_block(canvas, block)
    renderer_mod._finalize_render_completion_contract(block)

    render_bbox = block["render_bbox"]
    font_size = int(block["font_size_final"])
    render_center = _bbox_center(render_bbox)
    safe_center = _bbox_center(safe_bbox)
    assert block["fit_status"] == "ok"
    assert block["render_completed"] is True
    assert 18 <= font_size <= 28
    assert safe_bbox[0] <= render_bbox[0] < render_bbox[2] <= safe_bbox[2]
    assert safe_bbox[1] <= render_bbox[1] < render_bbox[3] <= safe_bbox[3]
    assert render_center == pytest.approx(safe_center, abs=4.0)


def test_owner_short_body_grows_beyond_default_24px_to_preserve_source_scale() -> None:
    safe_bbox = (40, 40, 320, 230)
    block = {
        "owner_id": "owner_short_scale",
        "translated": "NAO!",
        "translated_payload": "NAO!",
        "render_safe_polygon_page": [list(point) for point in _polygon_for_bbox(safe_bbox)],
        "layout_regions": [],
        "bbox": list(safe_bbox),
        "safe_text_box": list(safe_bbox),
        "layout_safe_bbox": list(safe_bbox),
        "layout_bbox": list(safe_bbox),
        "balloon_bbox": list(safe_bbox),
        "page_width": PAGE_WIDTH,
        "page_height": PAGE_HEIGHT,
        "layout_profile": "white_balloon",
        "source_ink_heights_px": [46],
        "source_x_heights_px": [32.2],
        "source_scale_evidence_confidence": 0.96,
        "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "cor": "#111111"},
        "_owner_render_mode": True,
    }
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (235, 235, 235))

    renderer_mod.render_text_block(canvas, block)

    assert block["render_completed"] is True
    assert block["fit_status"] == "ok"
    assert int(block["font_size_final"]) > 24
    assert block["owner_render_quality"]["status"] == "ok"


def test_owner_renderer_does_not_grow_above_source_scale_ceiling() -> None:
    safe_bbox = (30, 30, 330, 250)
    block = {
        "owner_id": "owner_scale_ceiling",
        "translated": "SIM",
        "translated_payload": "SIM",
        "render_safe_polygon_page": [list(point) for point in _polygon_for_bbox(safe_bbox)],
        "layout_regions": [],
        "bbox": list(safe_bbox),
        "safe_text_box": list(safe_bbox),
        "layout_safe_bbox": list(safe_bbox),
        "layout_bbox": list(safe_bbox),
        "balloon_bbox": list(safe_bbox),
        "page_width": PAGE_WIDTH,
        "page_height": PAGE_HEIGHT,
        "layout_profile": "white_balloon",
        "source_ink_heights_px": [18],
        "source_x_heights_px": [12.6],
        "source_scale_evidence_confidence": 0.99,
        "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "cor": "#111111"},
        "_owner_render_mode": True,
    }
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (235, 235, 235))

    renderer_mod.render_text_block(canvas, block)

    quality = block["owner_render_quality"]
    assert block["render_completed"] is True
    assert quality["status"] == "ok"
    assert float(quality["source_scale_ratio"]) <= 1.35


def test_connected_owner_requires_common_proportional_font() -> None:
    payload = "PRIMEIRA FRASE COMPLETA. SEGUNDA FRASE COMPLETA."
    owner_safe_polygon = ((10, 10), (350, 10), (350, 270), (10, 270))
    regions = [
        _layout_region("owner_connected", "top", (30, 25, 330, 120), order=0, owner_safe_polygon=owner_safe_polygon),
        _layout_region("owner_connected", "bottom", (30, 150, 330, 245), order=1, owner_safe_polygon=owner_safe_polygon),
    ]

    block = _render_connected_owner_block(payload, regions)

    assert block["render_completed"] is True
    assert len({chunk["font_size"] for chunk in block["visual_chunks"]}) == 1
    assert all(chunk["render_quality"]["status"] == "ok" for chunk in block["visual_chunks"])


def test_impossible_proportional_owner_fit_rolls_back_to_review() -> None:
    safe_bbox = (20, 20, 150, 42)
    payload = "ESTE CORPO INTEIRO NAO PODE CABER NESTA REGIAO MINUSCULA"
    block = {
        "owner_id": "owner_impossible",
        "translated": payload,
        "translated_payload": payload,
        "render_safe_polygon_page": [list(point) for point in _polygon_for_bbox(safe_bbox)],
        "layout_regions": [],
        "bbox": list(safe_bbox),
        "safe_text_box": list(safe_bbox),
        "layout_safe_bbox": list(safe_bbox),
        "layout_bbox": list(safe_bbox),
        "balloon_bbox": list(safe_bbox),
        "page_width": PAGE_WIDTH,
        "page_height": PAGE_HEIGHT,
        "layout_profile": "white_balloon",
        "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "cor": "#111111"},
        "_owner_render_mode": True,
    }
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (235, 235, 235))
    before = np.asarray(canvas).copy()

    renderer_mod.render_text_block(canvas, block)

    assert block["fit_status"] == "below_proportional_legibility"
    assert block["render_completed"] is False
    assert block["route_action"] == "review_required"
    assert np.array_equal(np.asarray(canvas), before)


def test_owner_body_is_never_split_or_truncated_to_fit() -> None:
    payload = "PRIMEIRA FRASE INTEIRA. SEGUNDA FRASE INTEIRA E SEM TRUNCAMENTO."
    owner_safe_polygon = ((10, 10), (350, 10), (350, 270), (10, 270))
    regions = [
        _layout_region("owner_connected", "top", (20, 20, 340, 125), order=0, owner_safe_polygon=owner_safe_polygon),
        _layout_region("owner_connected", "bottom", (20, 145, 340, 250), order=1, owner_safe_polygon=owner_safe_polygon),
    ]

    block = _render_connected_owner_block(payload, regions)

    visual_payload = " ".join(chunk["text"] for chunk in block["visual_chunks"])
    assert block["translated"] == payload
    assert block["translated_payload"] == payload
    assert _normalize_payload(visual_payload) == _normalize_payload(payload)


def test_connected_owner_render_ignores_legacy_ocr_anchor_per_region() -> None:
    payload = "PRIMEIRA FRASE NO ALTO. SEGUNDA FRASE NA PARTE INFERIOR."
    graph = _owner_graph(
        [("owner_connected", "TINY SOURCE", payload)],
        shared_bbox=(8, 12, 58, 38),
    )
    owner_safe_polygon = ((140, 20), (345, 20), (345, 250), (140, 250))
    regions = [
        _layout_region(
            "owner_connected",
            "region_connected_top",
            (160, 40, 330, 110),
            order=0,
            owner_safe_polygon=owner_safe_polygon,
        ),
        _layout_region(
            "owner_connected",
            "region_connected_bottom",
            (160, 150, 330, 230),
            order=1,
            owner_safe_polygon=owner_safe_polygon,
        ),
    ]
    _enriched, blocks = _owner_mode_blocks(graph, regions)
    block = blocks[0]
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (235, 235, 235))

    renderer_mod.render_text_block(canvas, block)
    renderer_mod._finalize_render_completion_contract(block)

    child_bboxes = block["_render_debug"]["child_render_bboxes"]
    child_safe_boxes = block["_render_debug"]["child_safe_text_boxes"]
    common_sizes = {int(chunk["font_size"]) for chunk in block["visual_chunks"]}
    assert block["fit_status"] == "ok"
    assert block["render_completed"] is True
    assert len(common_sizes) == 1
    assert 18 <= next(iter(common_sizes)) <= 28
    assert len(child_bboxes) == len(child_safe_boxes) == 2
    for rendered, safe in zip(child_bboxes, child_safe_boxes, strict=True):
        assert safe[0] <= rendered[0] < rendered[2] <= safe[2]
        assert safe[1] <= rendered[1] < rendered[3] <= safe[3]
        assert _bbox_center(rendered) == pytest.approx(_bbox_center(safe), abs=4.0)


def test_connected_owner_region_polygon_must_be_raster_subset_of_concave_owner() -> None:
    graph = _owner_graph(
        [("owner_connected", "SOURCE BODY", "CORPO TRADUZIDO")]
    )
    owner_safe_polygon = (
        (10, 10),
        (190, 10),
        (190, 190),
        (140, 190),
        (140, 60),
        (60, 60),
        (60, 190),
        (10, 190),
    )
    regions = [
        _layout_region(
            "owner_connected",
            "region_crosses_concavity",
            (30, 140, 170, 160),
            order=0,
            safe_polygon=((30, 140), (170, 140), (170, 160), (30, 160)),
            owner_safe_polygon=owner_safe_polygon,
        ),
        _layout_region(
            "owner_connected",
            "region_top",
            (80, 20, 120, 40),
            order=1,
            owner_safe_polygon=owner_safe_polygon,
        ),
    ]
    page = _owner_page(graph, regions)

    with pytest.raises(ValueError, match="owner_safe_polygon_page"):
        balloon_layout_mod.enrich_page_layout(
            page,
            owner_graph=graph,
            layout_regions=regions,
        )


def test_connected_owner_glyph_patch_rejects_ink_outside_exact_region_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _owner_graph(
        [("owner_connected", "SOURCE BODY", "PRIMEIRA PARTE. SEGUNDA PARTE.")]
    )
    owner_safe_polygon = ((20, 16), (184, 16), (184, 194), (20, 194))
    regions = [
        _layout_region(
            "owner_connected",
            "region_connected_top",
            (24, 20, 176, 76),
            order=0,
            owner_safe_polygon=owner_safe_polygon,
        ),
        _layout_region(
            "owner_connected",
            "region_connected_bottom",
            (24, 126, 176, 190),
            order=1,
            owner_safe_polygon=owner_safe_polygon,
        ),
    ]
    page = _owner_page(graph, regions)
    enriched = balloon_layout_mod.enrich_page_layout(
        page,
        owner_graph=graph,
        layout_regions=regions,
    )

    def render_only_in_region_gap(img, block, *_args, **_kwargs):
        ImageDraw.Draw(img).rectangle((70, 92, 110, 108), fill=(8, 12, 18))
        block["render_bbox"] = [70, 92, 111, 109]
        block["font_size_final"] = 20
        block["minimum_legible_font_px"] = 12
        block["fit_status"] = "ok"
        block["render_completed"] = True

    monkeypatch.setattr(renderer_mod, "render_text_block", render_only_in_region_gap)
    glyph_patch = renderer_mod.render_band_image(
        np.full((PAGE_HEIGHT, PAGE_WIDTH, 3), 235, dtype=np.uint8),
        enriched,
        owner_graph=graph,
    )

    region_union = np.zeros((PAGE_HEIGHT, PAGE_WIDTH), dtype=np.uint8)
    for region in regions:
        balloon_layout_mod.cv2.fillPoly(
            region_union,
            [np.asarray(region["safe_polygon_page"], dtype=np.int32)],
            255,
        )
    assert int(
        np.count_nonzero((glyph_patch.glyph_mask > 0) & (region_union == 0))
    ) > 0
    assert glyph_patch.render_completed is False
    assert glyph_patch.fit_status != "ok"


def test_normalized_chunks_reconstruct_exact_owner_payload() -> None:
    payload = "Primeira linha,  ainda inteira.\nSegunda linha — também completa!"

    first = renderer_mod._split_text_for_connected_balloons(
        payload,
        3,
        [0.25, 0.35, 0.40],
    )
    second = renderer_mod._split_text_for_connected_balloons(
        payload,
        3,
        [0.25, 0.35, 0.40],
    )

    assert first == second
    assert len(first) == 3
    assert all(chunk.strip() for chunk in first)
    assert _normalize_payload(" ".join(first)) == _normalize_payload(payload)


def test_renderer_never_merges_suppresses_or_reroutes_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _owner_graph(
        [
            ("owner_left", "REPEATED SOURCE", "TEXTO REPETIDO"),
            ("owner_right", "REPEATED SOURCE", "TEXTO REPETIDO"),
        ],
        shared_bbox=(30, 30, 210, 130),
    )
    regions = [
        _layout_region("owner_left", "region_left", (26, 26, 214, 134)),
        _layout_region("owner_right", "region_right", (26, 26, 214, 134)),
    ]
    before = _owner_semantic_signature(graph)

    def forbidden_semantic_inference(*_args, **_kwargs):
        raise AssertionError("legacy semantic inference ran in verified owner mode")

    monkeypatch.setattr(
        balloon_layout_mod,
        "build_mask_regions",
        forbidden_semantic_inference,
    )
    for helper_name in (
        "_merge_adjacent_white_balloon_fragments",
        "_merge_adjacent_same_balloon_fragments",
        "_drop_low_quality_duplicate_balloon_blocks",
        "_suppress_unsafe_automatic_render",
        "_split_single_ocr_visual_lobes",
    ):
        monkeypatch.setattr(
            renderer_mod,
            helper_name,
            forbidden_semantic_inference,
        )

    _enriched, blocks = _owner_mode_blocks(graph, regions)

    assert [block["owner_id"] for block in blocks] == ["owner_left", "owner_right"]
    assert all(block["route_action"] == "translate_inpaint_render" for block in blocks)
    assert _owner_semantic_signature(graph) == before


def test_renderer_rejects_owner_duplicates_with_divergent_visual_profiles() -> None:
    graph = _owner_graph(
        [("owner_style", "SOURCE BODY", "CORPO TRADUZIDO")]
    )
    regions = [
        _layout_region("owner_style", "region_style", (30, 30, 250, 180))
    ]
    enriched = balloon_layout_mod.enrich_page_layout(
        _owner_page(graph, regions),
        owner_graph=graph,
        layout_regions=regions,
    )
    first = copy.deepcopy(enriched["texts"][0])
    second = copy.deepcopy(first)
    second["visual_profile_v2"]["source_sha256"] = "0" * 64
    second["visual_profile_v2"]["visual_profile_sha256"] = (
        owner_visual_profile_sha256(second["visual_profile_v2"])
    )
    second["visual_profile_sha256"] = second["visual_profile_v2"][
        "visual_profile_sha256"
    ]

    with pytest.raises(ValueError, match="divergent duplicates"):
        renderer_mod.build_render_blocks(
            [first, second],
            owner_graph=graph,
        )


def _owner_layout_text(
    *,
    translated: str,
    ocr_bbox: tuple[int, int, int, int],
    safe_polygon: tuple[tuple[int, int], ...],
) -> dict:
    return {
        "owner_id": "owner_body",
        "page_id": PAGE_ID,
        "coordinate_space": "page",
        "translated": translated,
        "text": "SOURCE LINE",
        "bbox": list(ocr_bbox),
        "source_bbox": list(ocr_bbox),
        "text_pixel_bbox": list(ocr_bbox),
        "page_width": PAGE_WIDTH,
        "page_height": PAGE_HEIGHT,
        "layout_profile": "white_balloon",
        "render_safe_polygon_page": [list(point) for point in safe_polygon],
        "source_font_bounds_px": [20, 30],
        "container_font_bounds_px": [16, 24],
        "estilo": {
            "fonte": "ComicNeue-Bold.ttf",
            "tamanho": 46,
            "cor": "#111111",
            "contorno": "",
            "contorno_px": 0,
            "alinhamento": "center",
        },
    }


def test_owner_layout_source_evidence_is_permutation_stable() -> None:
    from dataclasses import replace
    from strip.process_bands import _owner_layout_regions

    graph = _owner_graph([("owner_stable", "SOURCE", "DESTINO")])
    original = graph.observations[0]
    first = replace(
        original,
        observation_id="obs_a",
        bbox_page=(24, 28, 176, 44),
        polygons_page=(((24, 28), (176, 28), (176, 44), (24, 44)),),
        confidence=0.90,
        coverage_score=0.80,
    )
    second = replace(
        original,
        observation_id="obs_b",
        bbox_page=(24, 50, 176, 74),
        polygons_page=(((24, 50), (176, 50), (176, 74), (24, 74)),),
        confidence=0.80,
        coverage_score=0.90,
    )
    graph.observations = [first, second]
    graph.owners[0] = replace(
        graph.owners[0],
        observation_ids=("obs_a", "obs_b"),
        selected_observation_ids=("obs_b", "obs_a"),
    )
    forward = _owner_layout_regions(
        graph,
        page_width=PAGE_WIDTH,
        page_height=PAGE_HEIGHT,
    )[0]
    graph.observations.reverse()
    graph.owners[0] = replace(
        graph.owners[0],
        observation_ids=("obs_b", "obs_a"),
        selected_observation_ids=("obs_a", "obs_b"),
    )
    reverse = _owner_layout_regions(
        graph,
        page_width=PAGE_WIDTH,
        page_height=PAGE_HEIGHT,
    )[0]

    keys = (
        "source_ink_heights_px",
        "source_x_heights_px",
        "source_ink_height_median_px",
        "source_x_height_median_px",
        "source_scale_evidence_confidence",
        "source_scale_evidence_ids",
    )
    assert {key: forward[key] for key in keys} == {
        key: reverse[key] for key in keys
    }


def test_edge_backed_dialogue_container_uses_central_safe_chord() -> None:
    from strip.process_bands import _owner_layout_regions

    graph = _owner_graph([("owner_edge", "SOURCE", "DESTINO")])
    geometry = build_owner_render_geometry(
        graph,
        "owner_edge",
        page_width=PAGE_WIDTH,
        page_height=PAGE_HEIGHT,
        container_evidence={
            "evidence_id": "owner_edge:full_page_visual_container:test",
            "source": "full_page_visual_container",
            "bbox_page": (10, 10, 200, 120),
            "confidence": 0.8,
        },
    )

    region = _owner_layout_regions(
        graph,
        page_width=PAGE_WIDTH,
        page_height=PAGE_HEIGHT,
        owner_render_geometry=geometry,
    )[0]

    assert region["bbox_page"] == [10, 10, 200, 120]
    assert region["safe_bbox_page"] == [39, 27, 171, 103]
    assert region["safe_polygon_page"] == [
        [39, 27], [170, 27], [170, 102], [39, 102]
    ]
    assert region["owner_safe_polygon_page"] == [
        [39, 27], [170, 27], [170, 102], [39, 102]
    ]


def test_same_body_lines_share_font_size_and_safe_polygon() -> None:
    safe_polygon = ((120, 42), (320, 42), (320, 218), (120, 218))
    line_records = [
        _owner_layout_text(
            translated="PRIMEIRA LINHA DO MESMO CORPO",
            ocr_bbox=(18, 56, 96, 82),
            safe_polygon=safe_polygon,
        ),
        _owner_layout_text(
            translated="SEGUNDA LINHA DO MESMO CORPO",
            ocr_bbox=(28, 160, 112, 190),
            safe_polygon=safe_polygon,
        ),
    ]

    plans = [renderer_mod.plan_text_layout(record) for record in line_records]

    assert {tuple(tuple(point) for point in plan["render_safe_polygon_page"]) for plan in plans} == {
        safe_polygon
    }
    assert len({tuple(plan["safe_text_box"]) for plan in plans}) == 1
    assert len({int(plan["target_size"]) for plan in plans}) == 1


def test_font_size_stays_within_source_and_container_bounds() -> None:
    safe_polygon = ((40, 30), (330, 30), (330, 230), (40, 230))
    text = _owner_layout_text(
        translated="SIM",
        ocr_bbox=(8, 8, 42, 24),
        safe_polygon=safe_polygon,
    )
    text["source_font_bounds_px"] = [22, 30]
    text["container_font_bounds_px"] = [14, 24]
    text["estilo"]["tamanho"] = 72

    plan = renderer_mod.plan_text_layout(text)
    resolved = renderer_mod._resolve_text_layout(text, plan)

    assert plan["font_size_bounds_px"] == [22, 24]
    assert plan["_font_search_floor"] == 22
    assert plan["_font_search_cap"] == 24
    assert 22 <= int(plan["target_size"]) <= 24
    assert 22 <= int(resolved["font_size"]) <= 24


def test_text_is_centered_by_safe_polygon_not_ocr_bbox() -> None:
    safe_polygon = ((150, 60), (330, 60), (330, 220), (150, 220))
    text = _owner_layout_text(
        translated="CENTRALIZADO NO CONTAINER SEGURO",
        ocr_bbox=(8, 12, 58, 38),
        safe_polygon=safe_polygon,
    )

    plan = renderer_mod.plan_text_layout(text)

    safe_center = _bbox_center(plan["safe_text_box"])
    polygon_center = (240.0, 140.0)
    ocr_center = _bbox_center(text["text_pixel_bbox"])
    assert safe_center == pytest.approx(polygon_center, abs=1.0)
    assert safe_center != pytest.approx(ocr_center, abs=1.0)
    assert plan["_follow_english_anchor_position"] is False


def test_style_evidence_permutation_does_not_change_owner_body_route_or_mask() -> None:
    graph = _owner_graph(
        [("owner_style", "COMPLETE SOURCE BODY", "CORPO TRADUZIDO COMPLETO")]
    )
    regions = [
        _layout_region("owner_style", "region_style", (30, 30, 250, 180))
    ]
    graph_before = _owner_semantic_signature(graph)
    poisoned_semantics = {
        "owner_id": "owner_forged",
        "page_id": "page_forged",
        "component_ids": ["component_forged"],
        "observation_ids": ["observation_forged"],
        "source_payload": "FORGED SOURCE",
        "translated_payload": "FORGED TRANSLATION",
        "disposition": "review",
        "state": "review_required",
        "route_action": "review_required",
        "execution_tile_id": "tile_forged",
        "action_mask_ref": "owner_masks/forged/action_mask.png",
        "layout_region_ids": ["region_forged"],
    }
    evidence_variants = [None]
    for confidence in (0.69, 0.70, 0.96, "invalid-confidence"):
        evidence_variants.append(
            {
                "style_origin": "source_detected",
                "style_confidence": confidence,
                "fonte": "Newrotic.ttf",
                "cor": "#552211",
                **poisoned_semantics,
            }
        )

    outputs = []
    for evidence in evidence_variants:
        _enriched, blocks = _owner_mode_blocks(
            graph,
            regions,
            style_evidence_by_owner={"owner_style": evidence},
        )
        assert len(blocks) == 1
        outputs.append(blocks[0])

    expected_semantics = _semantic_signature(outputs[0])
    assert all(
        _semantic_signature(block) == expected_semantics for block in outputs
    )
    assert all(isinstance(block.get("visual_profile"), dict) for block in outputs)
    assert all(
        set(block["visual_profile"]).isdisjoint(poisoned_semantics)
        for block in outputs
    )
    assert _owner_semantic_signature(graph) == graph_before
