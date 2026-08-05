from __future__ import annotations

import dataclasses
import itertools

import pytest

from ownership.consensus_v2 import independent_origins, select_consensus_observation
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
