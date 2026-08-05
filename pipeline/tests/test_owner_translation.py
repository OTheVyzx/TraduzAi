"""TDD contracts for the authoritative one-request-per-owner translation boundary."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.hash_contract import sha256_text
from ownership.model import (
    OWNER_GRAPH_SCHEMA_VERSION,
    ComponentDisposition,
    OwnerGraph,
    OwnerGraphValidationError,
    OwnerProjection,
    OwnerViolation,
    SourceTextComponent,
    TextObservation,
    TextOwner,
    TRANSLATION_ROUTE_ACTIONS,
)


def _translation_api() -> tuple[Callable[[OwnerGraph], dict], Callable[[OwnerGraph, dict], OwnerGraph]]:
    try:
        from ownership.translation import merge_owner_translations, owners_to_translation_page
    except (ImportError, ModuleNotFoundError):
        pytest.fail(
            "ownership.translation must implement the authoritative owner translation boundary",
            pytrace=False,
        )
    return owners_to_translation_page, merge_owner_translations


def _component(owner_suffix: str, index: int) -> SourceTextComponent:
    y1 = 10 + index * 50
    y2 = y1 + 30
    return SourceTextComponent(
        component_id=f"component_{owner_suffix}",
        page_id="page_001",
        bbox_page=(20, y1, 180, y2),
        polygon_page=((20, y1), (180, y1), (180, y2), (20, y2)),
        detector_sources=("independent_text_recall",),
        evidence_ids=(f"container_{owner_suffix}",),
    )


def _observation(
    owner_suffix: str,
    index: int,
    text: str,
    *,
    observation_suffix: str | None = None,
    tile_provenance: tuple[str, ...] | None = None,
) -> TextObservation:
    component = _component(owner_suffix, index)
    suffix = observation_suffix or owner_suffix
    return TextObservation(
        observation_id=f"observation_{suffix}",
        page_id="page_001",
        component_ids=(component.component_id,),
        text=text,
        confidence=0.94,
        provider="paddle_full_page",
        bbox_page=component.bbox_page,
        tile_provenance=tile_provenance or (f"tile_{owner_suffix}",),
        coverage_score=1.0,
        run_id="run-owner-translation",
        origin_execution_id="execution-owner-translation",
        invocation_id=f"invocation-owner-translation-{suffix}",
        attempt_id=f"attempt-owner-translation-{suffix}",
        provider_family="paddle",
        page_source_sha256="a" * 64,
        root_input_pixel_sha256="b" * 64,
        input_pixel_sha256=sha256_text(f"pixels:{suffix}"),
        payload_sha256=sha256_text(text),
    )


def _graph(
    payloads: list[tuple[str, str]],
    *,
    observations_by_owner: dict[str, list[TextObservation]] | None = None,
) -> OwnerGraph:
    components: list[SourceTextComponent] = []
    observations: list[TextObservation] = []
    owners: list[TextOwner] = []
    projections: list[OwnerProjection] = []
    dispositions: list[ComponentDisposition] = []
    supplied = observations_by_owner or {}

    for index, (owner_suffix, payload) in enumerate(payloads):
        component = _component(owner_suffix, index)
        owner_observations = supplied.get(owner_suffix) or [
            _observation(owner_suffix, index, payload)
        ]
        components.append(component)
        observations.extend(owner_observations)
        owner_id = f"owner_{owner_suffix}"
        tile_id = f"tile_{owner_suffix}"
        owners.append(
            TextOwner(
                owner_id=owner_id,
                page_id="page_001",
                component_ids=[component.component_id],
                observation_ids=[item.observation_id for item in owner_observations],
                selected_observation_ids=[item.observation_id for item in owner_observations],
                semantic_role="dialogue_body",
                source_payload=payload,
                translated_payload=None,
                disposition="owned",
                state="execution_planned",
                route_action="translate_inpaint_render",
                execution_tile_id=tile_id,
            )
        )
        projections.append(
            OwnerProjection(
                owner_id=owner_id,
                tile_id=tile_id,
                role="executor",
                bbox_page=component.bbox_page,
                bbox_tile=component.bbox_page,
                offset_xy=(0, 0),
            )
        )
        dispositions.append(
            ComponentDisposition(
                component_id=component.component_id,
                decision="owned",
                owner_id=owner_id,
                reason="semantic_owner_resolved",
            )
        )

    return OwnerGraph(
        schema_version=OWNER_GRAPH_SCHEMA_VERSION,
        page_id="page_001",
        run_id="run-owner-translation",
        origin_execution_id="execution-owner-translation",
        page_source_sha256="a" * 64,
        components=components,
        observations=observations,
        owners=owners,
        projections=projections,
        component_dispositions=dispositions,
    )


def _critical_codes(graph: OwnerGraph) -> set[str]:
    return {
        violation.code
        for violation in graph.violations
        if isinstance(violation, OwnerViolation) and violation.severity == "critical"
    }


def test_translator_receives_one_complete_payload_per_owner() -> None:
    owners_to_translation_page, _merge_owner_translations = _translation_api()
    graph = _graph(
        [
            ("a", "FIRST COMPLETE SOURCE BODY"),
            ("b", "SECOND COMPLETE SOURCE BODY"),
        ]
    )

    translation_page = owners_to_translation_page(graph)

    assert translation_page["page_id"] == "page_001"
    assert [
        (item["owner_id"], item["text"], item["semantic_role"])
        for item in translation_page["texts"]
    ] == [
        ("owner_a", "FIRST COMPLETE SOURCE BODY", "dialogue_body"),
        ("owner_b", "SECOND COMPLETE SOURCE BODY", "dialogue_body"),
    ]


def test_owner_payload_is_unchanged_between_reconcile_and_translation() -> None:
    from ownership.reconcile import SemanticRegion, build_page_owner_graph

    owners_to_translation_page, _merge_owner_translations = _translation_api()
    component = _component("body", 0)
    observation = _observation("body", 0, "HEADER 200MILLION")
    graph = build_page_owner_graph(
        page_id="page_001",
        components=[component],
        observations=[observation],
        semantic_regions=[
            SemanticRegion("body", (component.component_id,), "dialogue_body")
        ],
    )
    source_payload = graph.owners[0].source_payload

    translation_page = owners_to_translation_page(graph)

    assert translation_page["texts"][0]["text"] == source_payload
    assert translation_page["texts"][0]["original"] == source_payload


def test_body_lines_never_reach_translator_as_separate_requests() -> None:
    owners_to_translation_page, _merge_owner_translations = _translation_api()
    line_observations = [
        _observation("body", 0, "FIRST LINE", observation_suffix="body_line_1"),
        _observation("body", 0, "SECOND LINE", observation_suffix="body_line_2"),
        _observation("body", 0, "THIRD LINE", observation_suffix="body_line_3"),
    ]
    graph = _graph(
        [("body", "FIRST LINE\nSECOND LINE\nTHIRD LINE")],
        observations_by_owner={"body": line_observations},
    )

    translation_page = owners_to_translation_page(graph)

    assert len(translation_page["texts"]) == 1
    assert translation_page["texts"][0]["owner_id"] == "owner_body"
    assert translation_page["texts"][0]["text"] == "FIRST LINE\nSECOND LINE\nTHIRD LINE"
    assert all(
        item["text"] not in {"FIRST LINE", "SECOND LINE", "THIRD LINE"}
        for item in translation_page["texts"]
    )


def test_translation_response_is_joined_by_owner_id_not_list_position() -> None:
    _owners_to_translation_page, merge_owner_translations = _translation_api()
    graph = _graph([("a", "SOURCE A"), ("b", "SOURCE B")])
    response_in_reverse_order = {
        "texts": [
            {"owner_id": "owner_b", "translated": "TRADUCAO B"},
            {"owner_id": "owner_a", "translated": "TRADUCAO A"},
        ]
    }

    merged = merge_owner_translations(graph, response_in_reverse_order)
    by_id = {owner.owner_id: owner for owner in merged.owners}

    assert by_id["owner_a"].translated_payload == "TRADUCAO A"
    assert by_id["owner_b"].translated_payload == "TRADUCAO B"
    assert by_id["owner_a"].state == "translated"
    assert by_id["owner_b"].state == "translated"
    assert not merged.violations


def test_locale_blocker_prevents_owner_translation_ready() -> None:
    _owners_to_translation_page, merge_owner_translations = _translation_api()
    graph = _graph([("a", "The reward is 2 billion coins.")])
    target = "A recompensa é de 2 milhões de moedas."

    merged = merge_owner_translations(
        graph,
        {
            "texts": [
                {
                    "owner_id": "owner_a",
                    "translated": target,
                    "target_locale": "pt-BR",
                    "locale_validation": {
                        "status": "blocked",
                        "issues": [{"code": "numeric_magnitude_mismatch"}],
                    },
                    "qa_flags": ["translation_locale_mismatch"],
                }
            ]
        },
    )

    owner = merged.owners[0]
    assert owner.state == "review_required"
    assert owner.translated_payload is None
    assert "owner_translation_locale_mismatch" in _critical_codes(merged)


def test_owner_translation_request_declares_target_locale() -> None:
    owners_to_translation_page, _merge_owner_translations = _translation_api()
    page = owners_to_translation_page(_graph([("a", "SOURCE")]), target_locale="pt-BR")

    assert page["_owner_translation_contract"]["target_locale"] == "pt-BR"
    assert page["texts"][0]["target_locale"] == "pt-BR"


def test_missing_owner_translation_becomes_review_required() -> None:
    _owners_to_translation_page, merge_owner_translations = _translation_api()
    graph = _graph([("a", "SOURCE A"), ("b", "SOURCE B")])
    response_missing_owner_b = {
        "texts": [{"owner_id": "owner_a", "translated": "TRADUCAO A"}]
    }

    merged = merge_owner_translations(graph, response_missing_owner_b)
    by_id = {owner.owner_id: owner for owner in merged.owners}

    assert by_id["owner_b"].state == "review_required"
    assert by_id["owner_b"].translated_payload is None
    assert "owner_translation_missing" in _critical_codes(merged)
    assert any(
        violation.code == "owner_translation_missing"
        and "owner_b" in violation.offenders
        for violation in merged.violations
    )
    with pytest.raises(OwnerGraphValidationError):
        merged.require_valid()


@pytest.mark.parametrize(
    ("response", "expected_code", "offender"),
    [
        (
            {
                "texts": [
                    {"owner_id": "owner_a", "translated": "PRIMEIRA"},
                    {"owner_id": "owner_a", "translated": "SEGUNDA"},
                ]
            },
            "owner_translation_duplicate",
            "owner_a",
        ),
        (
            {
                "texts": [
                    {"owner_id": "owner_a", "translated": "VALIDA"},
                    {"owner_id": "owner_unknown", "translated": "INTRUSA"},
                ]
            },
            "owner_translation_unknown",
            "owner_unknown",
        ),
    ],
)
def test_duplicate_or_unknown_owner_translation_blocks(
    response: dict,
    expected_code: str,
    offender: str,
) -> None:
    _owners_to_translation_page, merge_owner_translations = _translation_api()
    graph = _graph([("a", "SOURCE A")])

    merged = merge_owner_translations(graph, response)

    assert expected_code in _critical_codes(merged)
    assert any(
        violation.code == expected_code and offender in violation.offenders
        for violation in merged.violations
    )
    assert all(owner.state == "review_required" for owner in merged.owners)
    assert all(owner.translated_payload is None for owner in merged.owners)
    with pytest.raises(OwnerGraphValidationError):
        merged.require_valid()


@pytest.mark.parametrize("invalid_payload", [17, ["not", "text"], {"bad": "shape"}])
def test_non_string_owner_translation_payload_blocks(invalid_payload: object) -> None:
    _owners_to_translation_page, merge_owner_translations = _translation_api()
    graph = _graph([("a", "SOURCE A")])

    merged = merge_owner_translations(
        graph,
        {"texts": [{"owner_id": "owner_a", "translated": invalid_payload}]},
    )

    assert "owner_translation_payload_invalid" in _critical_codes(merged)
    assert merged.owners[0].state == "review_required"
    assert merged.owners[0].translated_payload is None


def test_same_owner_observed_in_many_tiles_is_translated_once() -> None:
    owners_to_translation_page, _merge_owner_translations = _translation_api()
    observations = [
        _observation(
            "a",
            0,
            "COMPLETE SOURCE BODY",
            observation_suffix=f"a_{tile_id}",
            tile_provenance=(tile_id,),
        )
        for tile_id in ("tile_top", "tile_middle", "tile_bottom")
    ]
    graph = _graph(
        [("a", "COMPLETE SOURCE BODY")],
        observations_by_owner={"a": observations},
    )

    translation_page = owners_to_translation_page(graph)

    assert [item["owner_id"] for item in translation_page["texts"]] == ["owner_a"]
    assert [item["text"] for item in translation_page["texts"]] == [
        "COMPLETE SOURCE BODY"
    ]


def test_translation_page_never_serializes_visual_profile_fields() -> None:
    owners_to_translation_page, _merge_owner_translations = _translation_api()
    graph = _graph([("a", "COMPLETE SOURCE BODY")])

    translation_page = owners_to_translation_page(graph)

    forbidden = {
        "visual_profile_v2",
        "visual_profile_sha256",
        "style_copy_status",
        "owner_style_capture",
        "style_evidence_v2",
        "candidate_confidence",
        "glyph_mask_sha256",
        "font_match_evidence",
    }
    assert forbidden.isdisjoint(translation_page)
    assert all(forbidden.isdisjoint(record) for record in translation_page["texts"])


def test_run_translate_stage_uses_owner_boundary_and_returns_graph_snapshot() -> None:
    from strip.process_bands import _run_translate_stage

    graph = _graph([("a", "SOURCE A"), ("b", "SOURCE B")])
    translator = MagicMock()
    translator.translate_pages.return_value = [
        {
            "texts": [
                {"owner_id": "owner_b", "translated": "TRADUCAO B"},
                {"owner_id": "owner_a", "translated": "TRADUCAO A"},
            ]
        }
    ]

    output = _run_translate_stage(
        {},
        translator=translator,
        owner_graph=graph,
    ).to_page_dict()

    sent_page = translator.translate_pages.call_args.args[0][0]
    assert [(item["owner_id"], item["text"]) for item in sent_page["texts"]] == [
        ("owner_a", "SOURCE A"),
        ("owner_b", "SOURCE B"),
    ]
    merged_graph = OwnerGraph.from_dict(output["_owner_graph_snapshot"])
    assert {
        owner.owner_id: owner.translated_payload for owner in merged_graph.owners
    } == {"owner_a": "TRADUCAO A", "owner_b": "TRADUCAO B"}
    assert [item["owner_id"] for item in output["texts"]] == ["owner_a", "owner_b"]
    assert [item["translated"] for item in output["texts"]] == [
        "TRADUCAO A",
        "TRADUCAO B",
    ]
    assert all(owner.translated_payload is None for owner in graph.owners)


def test_run_translate_stage_rejects_response_metadata_identity_overrides() -> None:
    from strip.process_bands import _run_translate_stage

    graph = _graph([("a", "AUTHORITATIVE SOURCE")])
    translator = MagicMock()
    translator.translate_pages.return_value = [
        {
            "texts": [
                {
                    "id": "forged_id",
                    "owner_id": "owner_a",
                    "page_id": "page_forged",
                    "text": "FORGED SOURCE",
                    "semantic_role": "forged_role",
                    "component_ids": ["component_forged"],
                    "observation_ids": ["observation_forged"],
                    "selected_observation_ids": ["observation_forged"],
                    "route_action": "preserve",
                    "translated": "TRADUCAO VALIDA",
                    "qa_flags": ["translation_quality_warning"],
                }
            ]
        }
    ]

    output = _run_translate_stage(
        {},
        translator=translator,
        owner_graph=graph,
    ).to_page_dict()

    record = output["texts"][0]
    assert record["id"] == "owner_a"
    assert record["owner_id"] == "owner_a"
    assert record["page_id"] == "page_001"
    assert record["text"] == "AUTHORITATIVE SOURCE"
    assert record["semantic_role"] == "dialogue_body"
    assert record["component_ids"] == ["component_a"]
    assert record["observation_ids"] == ["observation_a"]
    assert record["selected_observation_ids"] == ["observation_a"]
    assert record["route_action"] == "translate_inpaint_render"
    assert record["translated"] == "TRADUCAO VALIDA"
    assert record["qa_flags"] == ["translation_quality_warning"]


def test_run_translate_stage_materializes_already_translated_graph_without_backend() -> None:
    from strip.process_bands import _run_translate_stage

    graph = _graph([("a", "SOURCE A")])
    graph.owners[0].state = "translated"
    graph.owners[0].translated_payload = "TRADUCAO A"
    translator = MagicMock()

    output = _run_translate_stage(
        {},
        translator=translator,
        owner_graph=graph,
    ).to_page_dict()

    translator.translate_pages.assert_not_called()
    assert [(item["owner_id"], item["translated"]) for item in output["texts"]] == [
        ("owner_a", "TRADUCAO A")
    ]
    assert OwnerGraph.from_dict(output["_owner_graph_snapshot"]).to_dict() == graph.to_dict()


@pytest.mark.parametrize(
    "translated_pages",
    [
        None,
        [],
        [
            {"texts": [{"owner_id": "owner_a", "translated": "FIRST"}]},
            {"texts": [{"owner_id": "owner_a", "translated": "SECOND"}]},
        ],
    ],
)
def test_run_translate_stage_blocks_invalid_response_page_cardinality(
    translated_pages: object,
) -> None:
    from strip.process_bands import _run_translate_stage

    graph = _graph([("a", "SOURCE A")])
    translator = MagicMock()
    translator.translate_pages.return_value = translated_pages

    output = _run_translate_stage(
        {},
        translator=translator,
        owner_graph=graph,
    ).to_page_dict()
    merged = OwnerGraph.from_dict(output["_owner_graph_snapshot"])

    assert "owner_translation_page_count_mismatch" in _critical_codes(merged)
    assert merged.owners[0].state == "review_required"
    assert merged.owners[0].translated_payload is None


def _owner_request(owner_suffix: str = "a", *, route_action: str = "translate_inpaint_render"):
    from ownership.translation import OwnerTranslationRequest

    graph = _graph([(owner_suffix, "THE PLAYER HAS 10 KILLS")])
    graph.owners[0].route_action = route_action
    return OwnerTranslationRequest.from_graph(graph, graph.owners[0].owner_id)


@pytest.mark.parametrize(
    "source",
    [
        "COMO ESPERADO DE UMA RAPOSA, PARECE QUE YUJEONG JA FEZ UM MOVIMENTO",
        "ANTES",
    ],
)
def test_fresh_complete_owner_evidence_accepts_already_ptbr_provider_noop(
    source: str,
) -> None:
    from ownership.translation import OwnerTranslationRequest, translate_owner_page
    from translator.language_policy import PageLanguageEvidence

    graph = _graph([("ptbr", source)])
    request = OwnerTranslationRequest.from_graph(graph, "owner_ptbr")

    def unchanged(owner_request, _variant):
        return owner_request.source_text

    unchanged.backend_name = "fixture"
    result = translate_owner_page(
        (request,),
        backends=(unchanged,),
        page_language_evidence_by_owner={
            request.owner_id: PageLanguageEvidence.build(
                coverage_complete=True,
                source_only_tokens=(),
            )
        },
    )

    assert result.bindings[0].target_text == source
    assert result.bindings[0].language_verdict.policy_id == "already_target_language"
    assert result.bindings[0].preserves_original_pixels


def test_fresh_complete_owner_evidence_still_rejects_english_provider_noop() -> None:
    from ownership.translation import TranslationValidationExhausted, translate_owner_page
    from translator.language_policy import PageLanguageEvidence

    request = _owner_request()

    def unchanged(owner_request, _variant):
        return owner_request.source_text

    unchanged.backend_name = "fixture"
    with pytest.raises(TranslationValidationExhausted):
        translate_owner_page(
            (request,),
            backends=(unchanged,),
            page_language_evidence_by_owner={
                request.owner_id: PageLanguageEvidence.build(
                    coverage_complete=True,
                    source_only_tokens=(),
                )
            },
        )


def test_translation_binding_preserves_owner_and_hash_chain() -> None:
    from ownership.translation import TranslationAttempt, bind_translation
    from translator.language_policy import validate_target_language

    request = _owner_request()
    verdict = validate_target_language(
        source=request.source_text,
        target="O JOGADOR TEM 10 ABATES",
        role=request.semantic_role,
    )
    attempt = TranslationAttempt.build(
        request=request,
        backend="google",
        variant="primary",
        provider_model=None,
        provider_metadata={"request_id": "provider-1"},
        target_text="O JOGADOR TEM 10 ABATES",
        provider_called=True,
        cache_hit=False,
        status="accepted",
        language_verdict=verdict,
    )
    binding = bind_translation(request, "O JOGADOR TEM 10 ABATES", (attempt,))

    assert binding.owner_id == request.owner_id
    assert binding.source_payload_sha256 == request.source_payload_sha256
    assert binding.target_payload_sha256 == sha256_text(binding.target_text)
    assert binding.target_locale == "pt-BR"
    assert binding.attempt_ids == (attempt.attempt_id,)
    assert binding.translation_binding_sha256
    assert not binding.preserves_original_pixels


@pytest.mark.parametrize("route_action", sorted(TRANSLATION_ROUTE_ACTIONS))
def test_every_declared_translation_route_receives_exactly_one_binding(
    route_action: str,
) -> None:
    from ownership.translation import translate_owner_page

    request = _owner_request(route_action=route_action)
    calls: list[str] = []

    def backend(owner_request, variant):
        calls.append(owner_request.owner_id)
        return "O JOGADOR TEM 10 ABATES"

    backend.backend_name = "fixture"
    result = translate_owner_page((request,), backends=(backend,))

    assert calls == [request.owner_id]
    assert [binding.owner_id for binding in result.bindings] == [request.owner_id]


def test_invalid_english_targets_exhaust_without_binding() -> None:
    from ownership.translation import TranslationValidationExhausted, translate_owner

    request = _owner_request()

    def unchanged(owner_request, variant):
        return owner_request.source_text

    unchanged.backend_name = "fixture"
    with pytest.raises(TranslationValidationExhausted) as exc:
        translate_owner(request, backends=(unchanged,), max_attempts_per_backend=2)

    assert len(exc.value.attempts) == 2
    assert all(attempt.status == "rejected" for attempt in exc.value.attempts)


def test_bulk_adapter_never_cross_assigns_owner_payloads() -> None:
    from ownership.translation import translate_owner_page

    requests = (_owner_request("a"), _owner_request("b"))
    calls: list[tuple[str, str]] = []

    def backend(owner_request, variant):
        calls.append((owner_request.owner_id, variant))
        return (
            "O JOGADOR TEM 10 ABATES"
            if owner_request.owner_id == "owner_a"
            else "O PARTICIPANTE TEM 10 ABATES"
        )

    backend.backend_name = "fixture"
    result = translate_owner_page(requests, backends=(backend,))

    assert [owner_id for owner_id, _variant in calls] == ["owner_a", "owner_b"]
    assert [binding.owner_id for binding in result.bindings] == ["owner_a", "owner_b"]
    assert {attempt.owner_id for attempt in result.attempts} == {"owner_a", "owner_b"}


def test_translation_result_rejects_binding_from_other_execution() -> None:
    from ownership.translation import (
        OwnerPageTranslationResult,
        TranslationIdentityError,
        translate_owner_page,
    )

    request = _owner_request()

    def backend(_owner_request, _variant):
        return "O JOGADOR TEM 10 ABATES"

    backend.backend_name = "fixture"
    result = translate_owner_page((request,), backends=(backend,))
    stale = replace(result.bindings[0], origin_execution_id="execution-other")

    with pytest.raises(TranslationIdentityError):
        OwnerPageTranslationResult.build(result.attempts, (stale,))


def test_owner_reaches_target_ready_only_with_its_valid_ptbr_binding() -> None:
    from ownership.lifecycle import (
        LifecycleEvidence,
        OwnerLifecycle,
        OwnerLifecycleIdentity,
        advance_owner_with_translation,
    )
    from ownership.translation import translate_owner_page

    request = _owner_request()
    identity = OwnerLifecycleIdentity(
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        owner_id=request.owner_id,
    )
    discovery = LifecycleEvidence.build(
        identity=identity,
        evidence_id="discovery",
        evidence_kind="coverage",
        payload_sha256=request.source_payload_sha256,
    )
    lifecycle = OwnerLifecycle.start(identity, evidence=discovery)
    for state in ("observed", "owned"):
        lifecycle = lifecycle.advance(
            state,
            evidence=LifecycleEvidence.build(
                identity=identity,
                evidence_id=f"to-{state}",
                evidence_kind="coverage",
                payload_sha256=request.source_payload_sha256,
            ),
        )

    def backend(_owner_request, _variant):
        return "O JOGADOR TEM 10 ABATES"

    backend.backend_name = "fixture"
    binding = translate_owner_page((request,), backends=(backend,)).bindings[0]
    target_ready = advance_owner_with_translation(lifecycle, binding)

    assert target_ready.state == "target_ready"
    assert target_ready.evidence_sha256s[-1]


def test_invalid_google_target_advances_to_real_ollama_attempt(monkeypatch) -> None:
    from ownership.translation import translate_owner_page
    from translator import translate as translate_module

    request = _owner_request()

    def response(target: str) -> list[dict]:
        return [
            {
                "texts": [
                    {
                        "id": request.owner_id,
                        "owner_id": request.owner_id,
                        "translated": target,
                    }
                ]
            }
        ]

    google = MagicMock(return_value=response(request.source_text))
    ollama = MagicMock(return_value=response("O JOGADOR TEM 10 ABATES"))
    monkeypatch.setattr(translate_module, "_translate_with_google", google)
    monkeypatch.setattr(translate_module, "_translate_with_ollama", ollama)
    monkeypatch.setattr(
        translate_module,
        "_google",
        type("Google", (), {"_cache": {}, "_persistent_cache": None})(),
    )
    controls = (
        translate_module.TranslationAttemptControl("google", "primary", True),
        translate_module.TranslationAttemptControl("ollama", "local", True),
    )

    result = translate_owner_page(
        (request,),
        attempt_fn=translate_module.translate_one_owner_attempt,
        attempt_controls=controls,
        attempt_kwargs={
            "obra": "obra",
            "context": {},
            "glossario": {},
            "idioma_destino": "pt-BR",
            "idioma_origem": "en",
            "qualidade": "normal",
            "ollama_host": "http://localhost:11434",
            "ollama_model": "traduzai-translator",
            "models_dir": "",
            "translation_context": None,
        },
    )

    assert [attempt.backend for attempt in result.attempts] == ["google", "ollama"]
    assert [attempt.status for attempt in result.attempts] == ["rejected", "accepted"]
    assert result.bindings[0].attempt_ids == tuple(
        attempt.attempt_id for attempt in result.attempts
    )
    google.assert_called_once()
    ollama.assert_called_once()


def test_unavailable_real_providers_abort_with_persistable_attempts(monkeypatch) -> None:
    from ownership.translation import TranslationInfrastructureError, translate_owner_page
    from translator import translate as translate_module

    request = _owner_request()

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(translate_module, "_translate_with_google", unavailable)
    monkeypatch.setattr(translate_module, "_translate_with_ollama", unavailable)
    monkeypatch.setattr(
        translate_module,
        "_google",
        type("Google", (), {"_cache": {}, "_persistent_cache": None})(),
    )
    controls = (
        translate_module.TranslationAttemptControl("google", "primary", True),
        translate_module.TranslationAttemptControl("ollama", "local", True),
    )

    with pytest.raises(TranslationInfrastructureError) as exc:
        translate_owner_page(
            (request,),
            attempt_fn=translate_module.translate_one_owner_attempt,
            attempt_controls=controls,
            attempt_kwargs={
                "obra": "obra",
                "context": {},
                "glossario": {},
                "idioma_destino": "pt-BR",
                "idioma_origem": "en",
                "qualidade": "normal",
                "ollama_host": "http://localhost:11434",
                "ollama_model": "traduzai-translator",
                "models_dir": "",
                "translation_context": None,
            },
        )

    assert [attempt.status for attempt in exc.value.attempts] == [
        "operational_error",
        "operational_error",
    ]
    assert all(not attempt.cache_hit for attempt in exc.value.attempts)


def test_translation_result_reopens_with_identical_attempts_and_bindings() -> None:
    from ownership.translation import OwnerPageTranslationResult, translate_owner_page

    request = _owner_request()

    def backend(_owner_request, _variant):
        return "O JOGADOR TEM 10 ABATES"

    backend.backend_name = "fixture"
    result = translate_owner_page((request,), backends=(backend,))
    reopened = OwnerPageTranslationResult.from_canonical_json_bytes(
        result.canonical_json_bytes
    )

    assert reopened == result
    assert reopened.attempts[0].provider_metadata_sha256 == sha256_text(
        reopened.attempts[0].provider_metadata_json_bytes.decode("utf-8")
    )


def test_validated_binding_advances_same_graph_owner_to_target_ready() -> None:
    from ownership.translation import apply_owner_translation_result, translate_owner_page

    graph = _graph([("a", "THE PLAYER HAS 10 KILLS")])
    graph.owners[0].state = "owned"
    request = _owner_request()

    def backend(_owner_request, _variant):
        return "O JOGADOR TEM 10 ABATES"

    backend.backend_name = "fixture"
    result = translate_owner_page((request,), backends=(backend,))
    merged = apply_owner_translation_result(graph, result)

    assert merged.owners[0].owner_id == graph.owners[0].owner_id
    assert merged.owners[0].translated_payload == "O JOGADOR TEM 10 ABATES"
    assert merged.owners[0].state == "target_ready"
