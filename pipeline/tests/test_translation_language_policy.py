from __future__ import annotations

import pytest

from translator.language_policy import PageLanguageEvidence, validate_target_language


def test_unchanged_english_dialogue_is_not_valid_pt_br() -> None:
    verdict = validate_target_language(
        source="LET'S GO RIGHT AWAY!",
        target="LET'S GO RIGHT AWAY!",
        role="dialogue",
        explicit_entities=(),
    )

    assert not verdict.accepted
    assert verdict.reason == "unchanged_source_dialogue"


def test_mostly_english_target_requests_next_backend() -> None:
    verdict = validate_target_language(
        source="THE ARENA WILL BEGIN",
        target="THE ARENA WILL BEGIN assim que todos entrarem",
        role="dialogue",
    )

    assert not verdict.accepted
    assert verdict.retryable
    assert verdict.reason == "mostly_english_target"


def test_unknown_english_source_terms_cannot_hide_inside_ptbr_padding() -> None:
    verdict = validate_target_language(
        source="ACHIEVEMENT UNLOCKED",
        target="ACHIEVEMENT UNLOCKED para o jogador",
        role="card",
    )

    assert not verdict.accepted
    assert verdict.retryable
    assert set(verdict.source_only_tokens) == {"achievement", "unlocked"}


def test_pt_br_with_explicit_proper_name_and_equivalent_number_is_valid() -> None:
    verdict = validate_target_language(
        source="KIM SIMUN HAS 10 KILLS",
        target="KIM SIMUN TEM 10 ABATES",
        role="card",
        explicit_entities=("KIM SIMUN",),
    )

    assert verdict.accepted
    assert verdict.target_locale == "pt-BR"


@pytest.mark.parametrize(
    ("source", "target", "reason"),
    [
        ("YOU HAVE 10 KILLS", "VOCÊ TEM 100 ABATES", "numeric_mismatch"),
        ("PLAYER {name} HAS {count}", "O JOGADOR {name} TEM", "placeholder_mismatch"),
        ("THE ARENA WILL BEGIN", "", "empty_target"),
    ],
)
def test_semantically_incomplete_targets_are_rejected_before_binding(
    source: str,
    target: str,
    reason: str,
) -> None:
    verdict = validate_target_language(source=source, target=target, role="dialogue")

    assert not verdict.accepted
    assert verdict.reason == reason


def test_already_ptbr_container_needs_fresh_complete_page_evidence() -> None:
    without_evidence = validate_target_language(
        source="VAMOS ENTRAR AGORA!",
        target="VAMOS ENTRAR AGORA!",
        role="dialogue",
    )
    with_evidence = validate_target_language(
        source="VAMOS ENTRAR AGORA!",
        target="VAMOS ENTRAR AGORA!",
        role="dialogue",
        page_language_evidence=PageLanguageEvidence.build(
            coverage_complete=True,
            source_only_tokens=(),
        ),
    )

    assert not without_evidence.accepted
    assert with_evidence.accepted
    assert with_evidence.policy_id == "already_target_language"


def test_verdict_is_canonical_and_hash_bound() -> None:
    verdict = validate_target_language(
        source="THE PLAYER HAS 2 KILLS",
        target="O JOGADOR TEM 2 ABATES",
        role="card",
    )

    assert verdict.canonical_json_bytes
    assert len(verdict.verdict_sha256) == 64
    assert verdict.from_dict(verdict.to_dict()) == verdict
