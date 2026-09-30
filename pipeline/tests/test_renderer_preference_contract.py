from __future__ import annotations

import pytest


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_F = "f" * 64


def _raster_safety(*, protected_art_overlap_px: int = 0):
    from ownership.hash_contract import canonical_json_sha256

    body = {
        "schema": "traduzai.raster-safety.v1",
        "status": "pass" if protected_art_overlap_px == 0 else "review_required",
        "bbox": [10, 20, 30, 40],
        "ink_pixel_count": 120,
        "alpha_sha256": _A,
        "authorized_body_sha256": _B,
        "protected_art_sha256": _C,
        "outside_authorized_body_px": 0,
        "protected_art_overlap_px": protected_art_overlap_px,
    }
    return body | {"evidence_sha256": canonical_json_sha256(body)}


def _candidate(*, recipe_sha256: str, output_sha256: str, target_text: str = "Ação!"):
    from typesetter.preference_contract import PreferenceCandidate

    return PreferenceCandidate.build(
        owner_id="owner-001",
        target_text=target_text,
        source_sha256=_A,
        style_sha256=_B,
        layout_plan_sha256=_C,
        recipe_sha256=recipe_sha256,
        output_sha256=output_sha256,
        preview_ref={"relative_path": f"previews/{output_sha256[:8]}.png", "sha256": output_sha256},
        context_ref={"relative_path": "context/page-001.png", "sha256": _D},
        metrics={
            "minimum_ink_gap_px": 8.5,
            "font_size_px": 31,
            "raster_safety": _raster_safety(),
        },
        hard_safety_passed=True,
    )


def test_candidate_identity_is_canonical_and_does_not_include_display_position() -> None:
    from typesetter.preference_contract import PreferenceCandidate

    first = _candidate(recipe_sha256=_E, output_sha256=_F)
    rebuilt = PreferenceCandidate.from_dict(first.to_dict())

    assert rebuilt == first
    assert first.candidate_id.startswith("renderer-candidate:")
    assert first.preference_profile == "uncalibrated"
    assert first.target_sha256 != _A
    assert "display_position" not in first.to_dict()


@pytest.mark.parametrize("target_text", ["ação, bênção, órgão", "ac\u0327a\u0303o, be\u0302nc\u0327a\u0303o, o\u0301rga\u0303o"])
def test_candidate_contract_roundtrips_exact_unicode_form(target_text: str) -> None:
    from typesetter.preference_contract import PreferenceCandidate

    candidate = _candidate(recipe_sha256=_E, output_sha256=_F, target_text=target_text)
    rebuilt = PreferenceCandidate.from_dict(candidate.to_dict())

    assert rebuilt.target_text == target_text
    assert rebuilt.target_text.encode("utf-8") == target_text.encode("utf-8")


def test_comparison_requires_two_distinct_safe_candidates_for_same_rendering_task() -> None:
    from typesetter.preference_contract import PreferenceCandidate, PreferenceComparison

    first = _candidate(recipe_sha256=_E, output_sha256=_F)
    second = _candidate(recipe_sha256=_F, output_sha256=_E)

    comparison = PreferenceComparison.build(
        first,
        second,
        randomization_nonce="test-only-seed",
    )
    assert set(comparison.positions) == {"A", "B"}
    assert set(comparison.positions.values()) == {first.candidate_id, second.candidate_id}

    unsafe_payload = second.to_dict() | {"hard_safety_passed": False}
    unsafe_payload.pop("candidate_id")
    unsafe_payload["metrics"]["raster_safety"] = _raster_safety(protected_art_overlap_px=1)
    unsafe = PreferenceCandidate.build(**unsafe_payload)
    with pytest.raises(ValueError, match="hard safety"):
        PreferenceComparison.build(first, unsafe, randomization_nonce="seed")

    different_text = _candidate(recipe_sha256=_F, output_sha256=_E, target_text="Outra ação")
    with pytest.raises(ValueError, match="same target"):
        PreferenceComparison.build(first, different_text, randomization_nonce="seed")

    with pytest.raises(ValueError, match="distinct"):
        PreferenceComparison.build(first, first, randomization_nonce="seed")


def test_candidate_cannot_claim_hard_safety_without_raster_evidence() -> None:
    from typesetter.preference_contract import PreferenceCandidate

    payload = _candidate(recipe_sha256=_E, output_sha256=_F).to_dict()
    payload.pop("candidate_id")
    payload["metrics"] = {"font_size_px": 31}

    with pytest.raises(ValueError, match="raster safety evidence"):
        PreferenceCandidate.build(**payload)


def test_randomized_positions_are_reproducible_and_tamper_evident() -> None:
    from typesetter.preference_contract import PreferenceComparison

    first = _candidate(recipe_sha256=_E, output_sha256=_F)
    second = _candidate(recipe_sha256=_F, output_sha256=_E)
    one = PreferenceComparison.build(first, second, randomization_nonce="nonce-17")
    two = PreferenceComparison.build(second, first, randomization_nonce="nonce-17")

    assert one == two
    assert len(one.randomization_sha256) == 64
    assert PreferenceComparison.from_dict(one.to_dict()) == one

    tampered = one.to_dict()
    tampered["positions"] = {"A": one.positions["B"], "B": one.positions["A"]}
    with pytest.raises(ValueError, match="hash mismatch"):
        PreferenceComparison.from_dict(tampered)

    mixed = one.to_dict()
    mixed["candidates"][1] = _candidate(
        recipe_sha256=_F, output_sha256=_E, target_text="Texto diferente"
    ).to_dict()
    mixed_body = {key: value for key, value in mixed.items() if key != "comparison_sha256"}
    from ownership.hash_contract import canonical_json_sha256
    mixed["comparison_sha256"] = canonical_json_sha256(mixed_body)
    mixed["positions"]["B"] = mixed["candidates"][1]["candidate_id"]
    mixed_body = {key: value for key, value in mixed.items() if key != "comparison_sha256"}
    mixed["comparison_sha256"] = canonical_json_sha256(mixed_body)
    with pytest.raises(ValueError, match="same target"):
        PreferenceComparison.from_dict(mixed)


@pytest.mark.parametrize(
    ("choice", "expected_position"),
    [("A", "A"), ("B", "B"), ("equivalent", None), ("neither", None), ("unsure", None)],
)
def test_preference_response_resolves_position_without_claiming_human_taste(
    choice: str, expected_position: str | None
) -> None:
    from typesetter.preference_contract import PreferenceComparison, resolve_preference_choice

    first = _candidate(recipe_sha256=_E, output_sha256=_F)
    second = _candidate(recipe_sha256=_F, output_sha256=_E)
    comparison = PreferenceComparison.build(first, second, randomization_nonce="nonce-23")

    result = resolve_preference_choice(comparison, choice)

    assert result["schema"] == "traduzai.renderer-preference-resolution.v1"
    assert result["choice"] == choice
    assert result["preference_profile"] == "uncalibrated"
    assert result["selected_candidate_id"] == (
        comparison.positions[expected_position] if expected_position else None
    )
    assert result["comparison_sha256"] == comparison.comparison_sha256
    displayed = {
        position: next(item for item in comparison.candidates if item.candidate_id == candidate_id)
        for position, candidate_id in comparison.positions.items()
    }
    assert result["candidate_recipe_sha256s"] == [
        displayed["A"].recipe_sha256,
        displayed["B"].recipe_sha256,
    ]
    assert result["candidate_output_sha256s"] == [
        displayed["A"].output_sha256,
        displayed["B"].output_sha256,
    ]


def test_preference_response_rejects_unknown_choice() -> None:
    from typesetter.preference_contract import PreferenceComparison, resolve_preference_choice

    first = _candidate(recipe_sha256=_E, output_sha256=_F)
    second = _candidate(recipe_sha256=_F, output_sha256=_E)
    comparison = PreferenceComparison.build(first, second, randomization_nonce="nonce-29")

    with pytest.raises(ValueError, match="choice"):
        resolve_preference_choice(comparison, "left")


def test_ranker_adapter_has_deterministic_uncalibrated_fallback() -> None:
    from typesetter.preference_contract import rank_preference_candidates

    first = _candidate(recipe_sha256=_E, output_sha256=_F)
    second = _candidate(recipe_sha256=_F, output_sha256=_E)

    ranked = rank_preference_candidates((second, first))

    assert ranked["candidate_ids"] == sorted((first.candidate_id, second.candidate_id))
    assert ranked["policy"] == "deterministic_candidate_id_v1"
    assert ranked["preference_profile"] == "uncalibrated"
    assert ranked["ranker_version"] is None


def test_capability_checked_ranker_may_only_reorder_safe_candidates() -> None:
    from typesetter.preference_contract import rank_preference_candidates

    first = _candidate(recipe_sha256=_E, output_sha256=_F)
    second = _candidate(recipe_sha256=_F, output_sha256=_E)

    class ReverseRanker:
        version = "fixture-ranker-v1"
        capabilities = {"renderer_safe_candidate_ranking_v1"}

        def rank_candidate_ids(self, candidates):
            return [item.candidate_id for item in reversed(candidates)]

    ranked = rank_preference_candidates((first, second), ranker=ReverseRanker())

    assert ranked["candidate_ids"] == [second.candidate_id, first.candidate_id]
    assert ranked["policy"] == "capability_ranker_v1"
    assert ranked["ranker_version"] == "fixture-ranker-v1"
    assert ranked["preference_profile"] == "uncalibrated"
