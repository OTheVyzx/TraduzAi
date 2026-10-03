from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import pytest

from ownership.consensus_v2 import independent_origins, select_consensus_observation, select_ordered_consensus_body, select_ordered_consensus_body_with_decisions
from ownership.evidence import OwnerObservationCollisionError, merge_observation_strict
from ownership.hash_contract import sha256_text
from ownership.model import (
    OWNER_GRAPH_LEGACY_SCHEMA_VERSION,
    OWNER_GRAPH_SCHEMA_VERSION,
    OwnerGraph,
    OwnerGraphValidationError,
    TextObservation,
)
from ownership.owner_builder import (
    OwnerEvidenceIdentityError,
    OwnerPageEvidenceContext,
    validate_and_group_owner_observations,
)
from ownership.project import validate_serialized_owner_graph


def _sha(value: str) -> str:
    return sha256_text(value)


def _observation(
    text: str,
    *,
    observation_id: str,
    invocation_id: str,
    provider: str,
    run_id: str = "run-current",
    origin_execution_id: str = "execution-current",
    page_id: str = "page_010",
    page_source_sha256: str | None = None,
    component_ids: tuple[str, ...] = ("component-1",),
) -> TextObservation:
    page_hash = page_source_sha256 or _sha("page-10")
    return TextObservation(
        observation_id=observation_id,
        page_id=page_id,
        component_ids=component_ids,
        text=text,
        confidence=0.94,
        provider=provider,
        bbox_page=(10, 20, 180, 70),
        run_id=run_id,
        origin_execution_id=origin_execution_id,
        invocation_id=invocation_id,
        attempt_id=f"{invocation_id}-attempt",
        provider_family="paddleocr",
        page_source_sha256=page_hash,
        root_input_pixel_sha256=page_hash,
        input_pixel_sha256=_sha(f"pixels:{invocation_id}"),
        payload_sha256=_sha(text),
    )


def _context() -> OwnerPageEvidenceContext:
    return OwnerPageEvidenceContext(
        run_id="run-current",
        origin_execution_id="execution-current",
        page_id="page_010",
        page_source_sha256=_sha("page-10"),
    )


def test_wrappers_from_one_ocr_invocation_count_as_one_consensus_vote() -> None:
    observations = (
        _observation(
            "STALE PAGE",
            observation_id="stale-1",
            invocation_id="infer-stale",
            provider="paddle_full_page",
        ),
        _observation(
            "STALE PAGE",
            observation_id="stale-2",
            invocation_id="infer-stale",
            provider="paddle_full_page_raw_line",
        ),
        _observation(
            "STALE PAGE",
            observation_id="stale-3",
            invocation_id="infer-stale",
            provider="visual_card_full_page_raw",
        ),
        _observation(
            "CURRENT ENGLISH BODY",
            observation_id="current",
            invocation_id="infer-current",
            provider="negative_detect_ocr",
        ),
    )

    assert independent_origins(observations[:3]) == frozenset({"infer-stale"})
    selected = select_consensus_observation(observations)
    assert selected.text == "CURRENT ENGLISH BODY"


def test_overlapping_compatible_readings_from_one_invocation_are_not_repeated() -> None:
    full = _observation("WAIT, WHAT DO YOU MEAN", observation_id="full", invocation_id="scan", provider="paddle")
    fragment = dataclasses.replace(_observation("WAIT, WHAT DO", observation_id="part", invocation_id="scan", provider="paddle"), bbox_page=(10, 20, 110, 70))
    selected, payload = select_ordered_consensus_body(((full, fragment),))
    assert payload == "WAIT, WHAT DO YOU MEAN"
    assert [item.observation_id for item in selected] == ["full"]


def test_spatially_separate_repetition_and_multiline_remain() -> None:
    first = _observation("WAIT", observation_id="first", invocation_id="scan", provider="paddle")
    second = dataclasses.replace(_observation("WAIT", observation_id="second", invocation_id="scan", provider="paddle"), bbox_page=(10, 100, 180, 150))
    third = dataclasses.replace(_observation("WHAT DO", observation_id="third", invocation_id="scan", provider="paddle"), bbox_page=(10, 155, 180, 200))
    selected, payload = select_ordered_consensus_body(((first, second, third),))
    assert [item.observation_id for item in selected] == ["first", "second", "third"]
    assert payload == "WAIT WAIT WHAT DO"


def test_rotated_card_lines_keep_physical_order_and_adjacent_fragments() -> None:
    # Measured page-space polygons from CH57/023, first crop OCR invocation.
    rows = (
        ("Reference for production.", ((17, 1473), (251, 1356), (262, 1377), (27, 1494))),
        ("it", ((249, 1497), (260, 1486), (270, 1496), (259, 1507))),
        ("This is a mock-up cell phone.", ((18, 1518), (289, 1381), (300, 1402), (28, 1540))),
        ("you inside", ((141, 1549), (246, 1493), (255, 1511), (150, 1567))),
        ("that has an icon and emoticons", ((20, 1562), (322, 1409), (332, 1429), (30, 1582))),
        ("Han Yoo-Hyun", ((161, 1632), (303, 1563), (313, 1585), (171, 1653))),
    )
    observations = []
    for index, (text, polygon) in enumerate(rows):
        xs, ys = zip(*polygon)
        observations.append(dataclasses.replace(
            _observation(text, observation_id=f"card-{index}", invocation_id="crop-card", provider="paddle_detected_crop"),
            bbox_page=(min(xs), min(ys), max(xs), max(ys)),
            polygons_page=(polygon,),
        ))
    selected, payload, decisions = select_ordered_consensus_body_with_decisions((tuple(observations),))
    assert len(selected) == 6
    assert payload == (
        "Reference for production. This is a mock-up cell phone. "
        "that has an icon and emoticons you inside it Han Yoo-Hyun"
    )
    assert not any(value["decision"].startswith("ambiguous") for value in decisions.values())

def test_cross_invocation_spatial_repeat_is_not_a_duplicate_when_boxes_are_separate() -> None:
    first = _observation("WAIT", observation_id="first", invocation_id="full", provider="paddle_full_page")
    second = dataclasses.replace(
        _observation("WAIT", observation_id="second", invocation_id="native", provider="paddle_anchored_native"),
        bbox_page=(10, 110, 180, 160),
    )
    selected, payload = select_ordered_consensus_body(((first,), (second,)))
    assert [item.observation_id for item in selected] == ["first", "second"]
    assert payload == "WAIT WAIT"


def test_cross_invocation_alternative_records_winner_and_reason() -> None:
    full = _observation("WAIT, WHAT DO", observation_id="full", invocation_id="full-pass", provider="paddle_full_page")
    native = dataclasses.replace(
        _observation("WA!T, WHAT DO", observation_id="native", invocation_id="native-pass", provider="paddle_anchored_native", component_ids=("different-component",)),
        bbox_page=(13, 19, 178, 69), confidence=0.99,
    )
    selected, payload, decisions = select_ordered_consensus_body_with_decisions(((full,), (native,)))
    assert payload == "WAIT, WHAT DO"
    assert [item.observation_id for item in selected] == ["full"]
    assert decisions["native"] == {"decision": "alternate_same_physical_line", "selected_observation_id": "full"}


def test_incompatible_readings_at_same_location_are_marked_ambiguous() -> None:
    first = _observation("YES", observation_id="first", invocation_id="full", provider="paddle_full_page")
    second = _observation("NO", observation_id="second", invocation_id="native", provider="paddle_anchored_native")
    _, _, decisions = select_ordered_consensus_body_with_decisions(((first,), (second,)))
    assert {item["decision"] for item in decisions.values()} == {
        "selected", "ambiguous_same_physical_line"
    }


def test_multiline_block_alternative_does_not_duplicate_its_lines() -> None:
    block = dataclasses.replace(
        _observation("WAIT WHAT DO YOU", observation_id="block", invocation_id="block-pass", provider="block"),
        bbox_page=(10, 10, 180, 90),
    )
    first = dataclasses.replace(
        _observation("WAIT WHAT", observation_id="line-1", invocation_id="line-pass", provider="line"),
        bbox_page=(20, 20, 165, 40),
    )
    second = dataclasses.replace(
        _observation("DO YOU", observation_id="line-2", invocation_id="line-pass", provider="line"),
        bbox_page=(20, 55, 165, 75),
    )
    selected, payload, decisions = select_ordered_consensus_body_with_decisions(((block,), (first, second)))
    assert [item.observation_id for item in selected] == ["line-1", "line-2"]
    assert payload == "WAIT WHAT DO YOU"
    assert decisions["block"]["decision"] == "alternate_block_covered_by_lines"


def test_conflicting_multiline_block_is_marked_ambiguous() -> None:
    block = dataclasses.replace(
        _observation("SOMETHING ELSE", observation_id="block", invocation_id="block-pass", provider="block"),
        bbox_page=(10, 10, 180, 90),
    )
    lines = [
        dataclasses.replace(_observation(text, observation_id=f"line-{index}", invocation_id="line-pass", provider="line"), bbox_page=box)
        for index, (text, box) in enumerate((("WAIT WHAT", (20, 20, 165, 40)), ("DO YOU", (20, 55, 165, 75))), 1)
    ]
    selected, _, decisions = select_ordered_consensus_body_with_decisions(((block,), tuple(lines)))
    assert [item.observation_id for item in selected] == ["line-1", "line-2"]
    assert decisions["block"]["decision"] == "ambiguous_block_line_overlap"


def test_previous_page_hash_is_rejected_before_consensus() -> None:
    stale = _observation(
        "STALE PAGE",
        observation_id="stale",
        invocation_id="infer-stale",
        provider="paddle_full_page",
        page_id="page_009",
        page_source_sha256=_sha("page-9"),
    )
    with pytest.raises(OwnerEvidenceIdentityError):
        validate_and_group_owner_observations(_context(), (stale,))


def test_previous_run_id_is_rejected_before_consensus() -> None:
    stale = _observation(
        "STALE RUN",
        observation_id="stale",
        invocation_id="infer-stale",
        provider="paddle_full_page",
        run_id="run-previous",
    )
    with pytest.raises(OwnerEvidenceIdentityError):
        validate_and_group_owner_observations(_context(), (stale,))


def test_provider_alias_permutation_never_changes_selected_payload() -> None:
    observations = (
        _observation("STALE PAGE", observation_id="s1", invocation_id="same", provider="alias-a"),
        _observation("STALE PAGE", observation_id="s2", invocation_id="same", provider="alias-b"),
        _observation("CURRENT ENGLISH BODY", observation_id="c1", invocation_id="fresh", provider="negative"),
    )
    assert {
        select_consensus_observation(permutation).text
        for permutation in itertools.permutations(observations)
    } == {"CURRENT ENGLISH BODY"}


def test_new_observation_identity_fields_are_keyword_only_and_enforceable() -> None:
    observation = _observation(
        "CURRENT ENGLISH BODY",
        observation_id="current",
        invocation_id="infer-current",
        provider="negative",
    )
    assert observation.identity_complete
    with pytest.raises(TypeError):
        TextObservation(
            "obs",
            "page_010",
            (),
            "TEXT",
            0.9,
            "fixture",
            (0, 0, 20, 10),
            (),
            (),
            None,
            None,
            None,
            None,
            False,
            "",
            "attempt",
            None,
            (),
            None,
            None,
            None,
            None,
            (),
            None,
            None,
            "run-positional-forbidden",
        )


def test_new_owner_graph_roundtrip_preserves_identity() -> None:
    observation = _observation(
        "CURRENT ENGLISH BODY",
        observation_id="current",
        invocation_id="infer-current",
        provider="negative",
    )
    graph = OwnerGraph(
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id="page_010",
        components=[],
        observations=[dataclasses.replace(observation, component_ids=())],
        owners=[],
        projections=[],
        run_id="run-current",
        origin_execution_id="execution-current",
        page_source_sha256=_sha("page-10"),
        verification_status="verified",
    )
    loaded = OwnerGraph.from_dict(graph.to_dict(), enforce=True)
    assert loaded.schema_version == OWNER_GRAPH_SCHEMA_VERSION == 2
    assert loaded.run_id == graph.run_id
    assert loaded.origin_execution_id == graph.origin_execution_id
    assert loaded.page_source_sha256 == graph.page_source_sha256
    assert loaded.observations[0].invocation_id == observation.invocation_id
    assert loaded.observations[0].input_pixel_sha256 == observation.input_pixel_sha256


def test_legacy_graph_is_readable_only_as_legacy_unverified() -> None:
    payload = {
        "schema_version": OWNER_GRAPH_LEGACY_SCHEMA_VERSION,
        "page_id": "page_010",
        "components": [],
        "observations": [],
        "owners": [],
        "projections": [],
        "component_dispositions": [],
        "violations": [],
    }
    legacy = OwnerGraph.from_dict(payload, enforce=False)
    assert legacy.verification_status == "legacy_unverified"
    with pytest.raises(OwnerGraphValidationError):
        OwnerGraph.from_dict(payload, enforce=True)


def test_verified_graph_rejects_observation_from_other_execution() -> None:
    observation = _observation(
        "CURRENT ENGLISH BODY",
        observation_id="current",
        invocation_id="infer-current",
        provider="negative",
    )
    graph = OwnerGraph(
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id="page_010",
        components=[],
        observations=[dataclasses.replace(observation, origin_execution_id="execution-other")],
        owners=[],
        projections=[],
        run_id="run-current",
        origin_execution_id="execution-current",
        page_source_sha256=_sha("page-10"),
        verification_status="verified",
    )
    with pytest.raises(OwnerGraphValidationError):
        graph.require_valid(mode="enforce")


def _verified_graph(observation: TextObservation | None = None) -> OwnerGraph:
    item = observation or _observation(
        "CURRENT ENGLISH BODY",
        observation_id="current",
        invocation_id="infer-current",
        provider="negative",
    )
    return OwnerGraph(
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id="page_010",
        components=[],
        observations=[dataclasses.replace(item, component_ids=())],
        owners=[],
        projections=[],
        run_id="run-current",
        origin_execution_id="execution-current",
        page_source_sha256=_sha("page-10"),
        verification_status="verified",
    )


def test_current_schema_without_explicit_verification_status_is_not_enforceable() -> None:
    payload = _verified_graph().to_dict()
    payload.pop("verification_status")

    with pytest.raises(OwnerGraphValidationError):
        OwnerGraph.from_dict(payload, enforce=True)


def test_serialized_graph_boundary_propagates_enforce_mode() -> None:
    legacy = {
        "schema_version": OWNER_GRAPH_LEGACY_SCHEMA_VERSION,
        "page_id": "page_010",
        "components": [],
        "observations": [],
        "owners": [],
        "projections": [],
        "component_dispositions": [],
        "violations": [],
    }

    assert (
        validate_serialized_owner_graph(legacy, enforce=False).verification_status
        == "legacy_unverified"
    )
    with pytest.raises(OwnerGraphValidationError):
        validate_serialized_owner_graph(legacy, enforce=True)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("run_id", "run-other"),
        ("origin_execution_id", "execution-other"),
        ("page_source_sha256", _sha("page-other")),
    ],
)
def test_verified_graph_rejects_observation_from_other_request_identity(
    field_name: str,
    replacement: str,
) -> None:
    graph = _verified_graph()
    graph.observations[0] = dataclasses.replace(
        graph.observations[0],
        **{field_name: replacement},
    )

    with pytest.raises(OwnerGraphValidationError):
        graph.require_valid(mode="enforce")


def test_legacy_positional_observation_remains_loadable_but_not_enforceable() -> None:
    legacy = TextObservation(
        "obs",
        "page_010",
        (),
        "TEXT",
        0.9,
        "fixture",
        (0, 0, 20, 10),
    )
    assert not legacy.identity_complete

    with pytest.raises(OwnerGraphValidationError):
        _verified_graph(legacy).require_valid(mode="enforce")


def test_owner_observation_group_is_deterministic_and_hash_bound() -> None:
    observations = (
        _observation(
            "CURRENT ENGLISH BODY",
            observation_id="current-b",
            invocation_id="infer-b",
            provider="negative-b",
        ),
        _observation(
            "CURRENT ENGLISH BODY",
            observation_id="current-a",
            invocation_id="infer-a",
            provider="negative-a",
        ),
    )

    first = validate_and_group_owner_observations(_context(), observations)[0]
    second = validate_and_group_owner_observations(
        _context(), tuple(reversed(observations))
    )[0]

    assert first == second
    assert first.observation_ids == ("current-a", "current-b")
    assert first.independent_invocation_ids == ("infer-a", "infer-b")
    assert first.source_payload_sha256 == _sha(first.source_payload)
    assert len(first.consensus_sha256) == 64


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("text", "CHANGED BODY"),
        ("payload_sha256", _sha("CHANGED BODY")),
        ("run_id", "run-other"),
        ("root_input_pixel_sha256", _sha("other-root")),
        ("attempt_id", "other-attempt"),
        ("input_pixel_sha256", _sha("other-input")),
    ],
)
def test_strict_observation_merge_rejects_same_id_collision(
    field_name: str,
    replacement: str,
) -> None:
    observation = _observation(
        "CURRENT ENGLISH BODY",
        observation_id="same-id",
        invocation_id="infer-current",
        provider="negative",
    )
    conflicting = dataclasses.replace(observation, **{field_name: replacement})

    with pytest.raises(OwnerObservationCollisionError):
        merge_observation_strict(observation, conflicting)


def test_strict_observation_merge_unions_only_provenance() -> None:
    observation = _observation(
        "CURRENT ENGLISH BODY",
        observation_id="same-id",
        invocation_id="infer-current",
        provider="negative",
    )
    merged = merge_observation_strict(
        dataclasses.replace(observation, tile_provenance=("tile-a",)),
        dataclasses.replace(observation, tile_provenance=("tile-b",)),
    )

    assert isinstance(merged, TextObservation)
    assert merged.tile_provenance == ("tile-a", "tile-b")
