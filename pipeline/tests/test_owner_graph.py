import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.model import (  # noqa: E402
    ComponentDisposition,
    OwnerGraph,
    OwnerGraphValidationError,
    OwnerProjection,
    SourceTextComponent,
    TextObservation,
    TextOwner,
)


def _component(component_id: str = "component_a") -> SourceTextComponent:
    return SourceTextComponent(
        component_id=component_id,
        page_id="page_a",
        bbox_page=(100, 120, 300, 180),
        polygon_page=((100, 120), (300, 120), (300, 180), (100, 180)),
        detector_sources=("region_detector",),
    )


def _observation(
    observation_id: str = "observation_a",
    component_ids: tuple[str, ...] = ("component_a",),
) -> TextObservation:
    return TextObservation(
        observation_id=observation_id,
        page_id="page_a",
        component_ids=component_ids,
        text="SOURCE BODY",
        confidence=0.9,
        provider="full_page_ocr",
        bbox_page=(100, 120, 300, 180),
        tile_provenance=("tile_a",),
    )


def _owner(
    owner_id: str = "owner_a",
    component_ids: list[str] | None = None,
) -> TextOwner:
    return TextOwner(
        owner_id=owner_id,
        page_id="page_a",
        component_ids=list(component_ids or ["component_a"]),
        observation_ids=["observation_a"],
        selected_observation_ids=["observation_a"],
        semantic_role="dialogue_body",
        source_payload="SOURCE BODY",
        translated_payload="CORPO TRADUZIDO",
        disposition="owned",
        state="translated",
        route_action="translate_inpaint_render",
        execution_tile_id="tile_a",
    )


def _projection(owner_id: str = "owner_a", role: str = "executor") -> OwnerProjection:
    return OwnerProjection(
        owner_id=owner_id,
        tile_id="tile_a",
        role=role,
        bbox_page=(100, 120, 300, 180),
        bbox_tile=(100, 20, 300, 80),
        offset_xy=(0, 100),
    )


def _disposition(
    component_id: str = "component_a",
    owner_id: str | None = "owner_a",
    decision: str = "owned",
) -> ComponentDisposition:
    return ComponentDisposition(
        component_id=component_id,
        decision=decision,
        owner_id=owner_id,
        reason="resolved_by_owner_graph",
    )


def _valid_graph() -> OwnerGraph:
    return OwnerGraph(
        schema_version=1,
        page_id="page_a",
        components=[_component()],
        observations=[_observation()],
        owners=[_owner()],
        projections=[_projection()],
        component_dispositions=[_disposition()],
    )


def _codes(graph: OwnerGraph) -> set[str]:
    return {violation.code for violation in graph.validate()}


def test_every_component_has_exactly_one_final_disposition():
    graph = _valid_graph()
    graph.component_dispositions = []
    assert "source_text_unowned" in _codes(graph)

    graph.component_dispositions = [_disposition(), _disposition(decision="review")]
    assert "component_disposition_conflict" in _codes(graph)


def test_component_cannot_belong_to_two_active_owners():
    graph = _valid_graph()
    graph.owners.append(_owner(owner_id="owner_b"))
    graph.projections.append(_projection(owner_id="owner_b"))

    assert "component_multiple_active_owners" in _codes(graph)


def test_translatable_owner_has_one_source_and_translation_payload():
    graph = _valid_graph()
    graph.owners[0] = replace(
        graph.owners[0],
        source_payload="",
        translated_payload=None,
    )

    assert {
        "owner_source_payload_missing",
        "owner_translation_payload_missing",
    } <= _codes(graph)


def test_body_owner_cannot_expose_multiple_semantic_payloads():
    graph = _valid_graph()
    graph.owners[0] = replace(
        graph.owners[0],
        source_payload=["SOURCE LINE A", "SOURCE LINE B"],  # type: ignore[arg-type]
    )

    assert "semantic_body_split" in _codes(graph)


def test_owner_has_exactly_one_executor_projection():
    graph = _valid_graph()
    graph.projections = []
    assert "owner_executor_missing" in _codes(graph)

    graph.projections = [_projection(), replace(_projection(), tile_id="tile_b")]
    assert "owner_executor_duplicated" in _codes(graph)


def test_review_required_owner_cannot_enter_render_plan():
    graph = _valid_graph()
    graph.owners[0] = replace(
        graph.owners[0],
        disposition="review",
        state="review_required",
        route_action="translate_inpaint_render",
    )
    graph.component_dispositions = [_disposition(owner_id="owner_a", decision="review")]

    assert "review_owner_in_render_plan" in _codes(graph)


def test_graph_serialization_is_deterministic():
    graph = _valid_graph()
    component_b = replace(
        _component("component_b"),
        bbox_page=(400, 500, 600, 560),
        polygon_page=((400, 500), (600, 500), (600, 560), (400, 560)),
    )
    graph.components.insert(0, component_b)
    graph.component_dispositions.insert(
        0,
        _disposition("component_b", owner_id=None, decision="preserve"),
    )

    first = graph.to_dict()
    second = graph.to_dict()

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [item["component_id"] for item in first["components"]] == [
        "component_a",
        "component_b",
    ]
    assert OwnerGraph.from_dict(first).to_dict() == first


def test_require_valid_raises_with_stable_offenders():
    graph = _valid_graph()
    graph.component_dispositions = []

    with pytest.raises(OwnerGraphValidationError) as exc_info:
        graph.require_valid()

    assert exc_info.value.violations[0].offenders == ("component_a",)
