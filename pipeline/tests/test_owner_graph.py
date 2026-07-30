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


def test_graph_rejects_duplicate_authoritative_ids():
    graph = _valid_graph()
    graph.components.append(replace(graph.components[0], bbox_page=(1, 2, 3, 4)))
    graph.observations.append(replace(graph.observations[0], text="CONFLICT"))
    graph.owners.append(replace(graph.owners[0], source_payload="CONFLICT"))

    assert {
        "component_identity_duplicated",
        "observation_identity_duplicated",
        "owner_identity_duplicated",
    } <= _codes(graph)


def test_graph_rejects_cross_page_and_orphan_identity_records():
    graph = _valid_graph()
    graph.components[0] = replace(graph.components[0], page_id="page_b")
    graph.observations[0] = replace(graph.observations[0], page_id="page_b")
    graph.owners[0] = replace(graph.owners[0], page_id="page_b")
    graph.projections.append(_projection(owner_id="owner_ghost", role="context_only"))

    assert {
        "component_page_mismatch",
        "observation_page_mismatch",
        "owner_page_mismatch",
        "projection_owner_unknown",
    } <= _codes(graph)


def test_graph_rejects_invalid_or_inconsistent_projection_geometry():
    graph = _valid_graph()
    graph.projections = [
        replace(_projection(), bbox_tile=(300, 80, 100, 20)),
        replace(_projection(), tile_id="tile_b", bbox_tile=(-1, 20, 199, 80)),
        replace(
            _projection(),
            tile_id="tile_c",
            role="context_only",
            bbox_page=(101, 120, 301, 180),
        ),
    ]

    assert {
        "projection_bbox_invalid",
        "projection_transform_mismatch",
    } <= _codes(graph)


def test_graph_accepts_negative_tile_offset_when_transform_is_canonical():
    graph = _valid_graph()
    graph.projections = [
        replace(
            _projection(),
            bbox_tile=(112, 100, 312, 160),
            offset_xy=(-12, 20),
        )
    ]

    assert not {
        "projection_bbox_invalid",
        "projection_transform_mismatch",
    }.intersection(_codes(graph))


def test_graph_rejects_unsupported_and_duplicate_owner_projections():
    graph = _valid_graph()
    graph.projections.append(replace(_projection(), role="side_effect"))

    assert {
        "projection_role_invalid",
        "owner_projection_duplicated",
    } <= _codes(graph)


def test_projection_bbox_must_equal_its_owner_component_union():
    graph = _valid_graph()
    graph.projections = [
        replace(
            _projection(),
            bbox_page=(10, 10, 30, 30),
            bbox_tile=(10, 10, 30, 30),
            offset_xy=(0, 0),
        )
    ]

    assert "projection_owner_geometry_mismatch" in _codes(graph)


def test_non_owned_owner_cannot_retain_execution_authority():
    graph = _valid_graph()
    graph.owners[0] = replace(
        graph.owners[0],
        disposition="preserve",
        state="mask_ready",
        route_action="translate_inpaint_render",
        action_mask_ref="owner_masks/stale/action_mask.png",
    )

    assert "non_owned_owner_in_execution_plan" in _codes(graph)


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


def test_observation_cannot_reference_unknown_source_components():
    graph = _valid_graph()
    graph.observations[0] = replace(
        graph.observations[0],
        component_ids=("component_a", "component_ghost"),
    )

    assert "observation_component_unknown" in _codes(graph)


def test_owned_owner_observation_and_selection_must_cover_every_component():
    graph = _valid_graph()
    component_b = replace(
        _component("component_b"),
        bbox_page=(320, 120, 400, 180),
        polygon_page=((320, 120), (400, 120), (400, 180), (320, 180)),
    )
    graph.components.append(component_b)
    graph.owners[0] = replace(
        graph.owners[0],
        component_ids=["component_a", "component_b"],
    )
    graph.projections[0] = replace(
        graph.projections[0],
        bbox_page=(100, 120, 400, 180),
        bbox_tile=(100, 20, 400, 80),
    )
    graph.component_dispositions.append(_disposition("component_b"))

    assert {
        "owner_observation_coverage_incomplete",
        "owner_selected_coverage_incomplete",
    } <= _codes(graph)


def test_owned_owner_cannot_claim_vacuously_complete_empty_coverage():
    graph = _valid_graph()
    graph.owners[0] = replace(
        graph.owners[0],
        component_ids=[],
        observation_ids=[],
        selected_observation_ids=[],
    )
    graph.component_dispositions[0] = _disposition(
        owner_id=None,
        decision="preserve",
    )
    graph.projections = []

    assert "owner_components_missing" in _codes(graph)


def test_selected_observation_cannot_contain_foreign_component():
    graph = _valid_graph()
    graph.components.append(_component("component_foreign"))
    graph.component_dispositions.append(
        _disposition("component_foreign", owner_id=None, decision="preserve")
    )
    graph.observations[0] = replace(
        graph.observations[0],
        component_ids=("component_a", "component_foreign"),
    )

    assert "owner_selected_observation_contaminated" in _codes(graph)


def test_component_disposition_must_match_owner_decision_and_membership():
    graph = _valid_graph()
    graph.owners.append(
        replace(
            _owner("owner_b", component_ids=["component_b"]),
            observation_ids=[],
            selected_observation_ids=[],
            disposition="review",
            state="review_required",
            route_action="review_required",
            execution_tile_id=None,
        )
    )
    graph.component_dispositions[0] = _disposition(
        "component_a",
        owner_id="owner_b",
        decision="owned",
    )

    assert {
        "disposition_owner_decision_mismatch",
        "disposition_owner_component_mismatch",
        "owner_component_disposition_mismatch",
    } <= _codes(graph)


def test_preserve_or_suppress_disposition_cannot_retain_an_owner():
    graph = _valid_graph()
    graph.component_dispositions[0] = _disposition(
        owner_id="owner_a",
        decision="preserve",
    )

    assert {
        "disposition_owner_forbidden",
        "owner_component_disposition_mismatch",
    } <= _codes(graph)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_code"),
    (
        ("disposition", "translated", "owner_disposition_invalid"),
        ("state", " TRANSLATED ", "owner_state_invalid"),
        ("route_action", "Translate_Inpaint_Render", "owner_route_action_invalid"),
    ),
)
def test_owner_lifecycle_values_are_exact_canonical_enums(
    field_name,
    invalid_value,
    expected_code,
):
    graph = _valid_graph()
    graph.owners[0] = replace(graph.owners[0], **{field_name: invalid_value})

    assert expected_code in _codes(graph)


@pytest.mark.parametrize(
    ("state", "route_action"),
    (
        ("mask_ready", "translate_render_only"),
        ("inpainted", "translate_render_only"),
        ("translated", "review_required"),
        ("review_required", "translate_inpaint_render"),
    ),
)
def test_owner_state_and_route_must_form_a_canonical_lifecycle_pair(
    state,
    route_action,
):
    graph = _valid_graph()
    graph.owners[0] = replace(
        graph.owners[0],
        state=state,
        route_action=route_action,
    )

    assert "owner_state_route_mismatch" in _codes(graph)


def test_review_disposition_must_remain_in_review_lifecycle():
    graph = _valid_graph()
    graph.owners[0] = replace(graph.owners[0], disposition="review")
    graph.component_dispositions[0] = _disposition(decision="review")

    assert "owner_disposition_lifecycle_mismatch" in _codes(graph)


def test_graph_rejects_duplicates_inside_identity_reference_lists():
    graph = _valid_graph()
    graph.components[0] = replace(
        graph.components[0],
        evidence_ids=("evidence_a", "evidence_a"),
    )
    graph.observations[0] = replace(
        graph.observations[0],
        component_ids=("component_a", "component_a"),
        projection_ids=("projection_a", "projection_a"),
    )
    graph.owners[0] = replace(
        graph.owners[0],
        component_ids=["component_a", "component_a"],
        observation_ids=["observation_a", "observation_a"],
        selected_observation_ids=["observation_a", "observation_a"],
    )

    assert {
        "component_evidence_ids_duplicated",
        "observation_component_ids_duplicated",
        "observation_projection_ids_duplicated",
        "owner_component_ids_duplicated",
        "owner_observation_ids_duplicated",
        "owner_selected_observation_ids_duplicated",
    } <= _codes(graph)


def test_component_and_observation_bboxes_must_be_canonical_page_geometry():
    graph = _valid_graph()
    graph.components[0] = replace(
        graph.components[0],
        bbox_page=(100, 120, 100, 180),
    )
    graph.observations[0] = replace(
        graph.observations[0],
        bbox_page=(True, 120, 300, 180),  # type: ignore[arg-type]
        source_bbox_page=(-1, 120, 300, 180),
        text_pixel_bbox_page=(100, 180, 300, 120),
    )

    assert {
        "component_bbox_invalid",
        "observation_bbox_invalid",
        "observation_source_bbox_invalid",
        "observation_text_pixel_bbox_invalid",
    } <= _codes(graph)


def test_explicitly_rejected_malformed_observation_remains_auditable_not_executable():
    graph = _valid_graph()
    graph.observations.append(
        replace(
            _observation("observation_rejected", component_ids=()),
            bbox_page=(0, 0, 0, 0),
            source_bbox_page=(0, 0, 0, 0),
            text_pixel_bbox_page=(0, 0, 0, 0),
            rejection_reason="invalid_bbox",
        )
    )

    assert not {
        "observation_bbox_invalid",
        "observation_source_bbox_invalid",
        "observation_text_pixel_bbox_invalid",
    } & _codes(graph)
    assert graph.observations[-1].rejection_reason == "invalid_bbox"


def test_serialized_bbox_coordinates_cannot_be_coerced_from_non_integers():
    payload = _valid_graph().to_dict()
    payload["components"][0]["bbox_page"][0] = 100.5

    with pytest.raises(ValueError, match="bbox.*integer|integer.*bbox"):
        OwnerGraph.from_dict(payload)


def test_graph_serialization_is_deterministic():
    graph = _valid_graph()
    graph.components[0] = replace(
        graph.components[0],
        confidence=0.87,
        script_evidence=("latin_likely",),
        evidence_ids=("detector_17", "glyph_04"),
        rotation_deg=-12.5,
        rotation_source="detector_orientation",
    )
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
    assert first["components"][0]["script_evidence"] == ["latin_likely"]
    assert first["components"][0]["evidence_ids"] == ["detector_17", "glyph_04"]
    assert first["components"][0]["rotation_deg"] == -12.5
    assert OwnerGraph.from_dict(first).to_dict() == first


def test_require_valid_raises_with_stable_offenders():
    graph = _valid_graph()
    graph.component_dispositions = []

    with pytest.raises(OwnerGraphValidationError) as exc_info:
        graph.require_valid()

    assert exc_info.value.violations[0].offenders == ("component_a",)
