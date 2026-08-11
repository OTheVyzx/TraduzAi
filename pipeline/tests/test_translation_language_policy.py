from __future__ import annotations

import pytest

from translator.language_policy import (
    PageLanguageEvidence,
    build_page_language_evidence,
    recover_noisy_source_anchors,
    validate_target_language,
)


def test_recovers_ordered_english_and_context_name_anchors_from_bilingual_overlay() -> None:
    recovery = recover_noisy_source_anchors(
        "ISSOANSRRAWO!HOOMO DIDINBEDODGEKMSSIMEOK'S SWPORBDSTRIKETFAOOEH...",
        context_names=("Kim Sihyeok",),
    )
    assert recovery is not None
    assert recovery.anchors == ("HOW", "DID", "DODGE", "KIM SIHYEOK", "SWORD", "STRIKE")
    assert recovery.reason == "bilingual_overlay_low_lexical_coherence"


def test_noisy_anchor_recovery_does_not_rewrite_coherent_language_text() -> None:
    assert recover_noisy_source_anchors("HOW DID HE DODGE KIM SIHYEOK'S SWORD STRIKE?", context_names=("Kim Sihyeok",)) is None
    assert recover_noisy_source_anchors("COMO ELE DESVIOU DO GOLPE DE ESPADA?", context_names=("Kim Sihyeok",)) is None


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


def test_shared_unknown_names_do_not_turn_valid_ptbr_into_english_residual() -> None:
    verdict = validate_target_language(
        source="YUJEONG SIHYEOK EXPECTED FOX",
        target="YUJEONG SIHYEOK, A RAPOSA AVANCOU",
        role="dialogue_body",
        explicit_entities=(),
    )

    assert verdict.accepted
    assert verdict.reason == "valid_pt_br"
    assert verdict.english_token_ratio_ppm == 0
    assert verdict.source_only_tokens == ()


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


def test_already_ptbr_accepts_shared_homographs_only_with_ptbr_specific_evidence() -> None:
    source = "UAU, VOCE ME ASSUSTOU."
    evidence = build_page_language_evidence(
        texts=(source,),
        coverage_complete=True,
    )

    verdict = validate_target_language(
        source=source,
        target=source,
        role="dialogue",
        page_language_evidence=evidence,
    )
    ambiguous_only = validate_target_language(
        source="ME",
        target="ME",
        role="dialogue",
        page_language_evidence=build_page_language_evidence(
            texts=("ME",),
            coverage_complete=True,
        ),
    )

    assert evidence.source_only_tokens == ()
    assert verdict.accepted
    assert verdict.policy_id == "already_target_language"
    assert not ambiguous_only.accepted


def test_already_ptbr_recognizes_morphology_when_ocr_contains_unknown_tokens() -> None:
    source = "ESTIMADO WVIP?"
    verdict = validate_target_language(
        source=source,
        target=source,
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=(source,),
            coverage_complete=True,
        ),
    )
    english = validate_target_language(
        source="ESTIMATED VIP?",
        target="ESTIMATED VIP?",
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=("ESTIMATED VIP?",),
            coverage_complete=True,
        ),
    )

    assert verdict.accepted
    assert verdict.policy_id == "already_target_language"
    assert not english.accepted


def test_joined_english_markers_block_mixed_unchanged_owner_text() -> None:
    source = "ISSOANSRRAWO DIDINBEDODGEKMSSIMEOK SWPORBDSTRIKETFAOOEH"
    evidence = build_page_language_evidence(
        texts=(source,),
        coverage_complete=True,
    )
    verdict = validate_target_language(
        source=source,
        target=source,
        role="dialogue_body",
        page_language_evidence=evidence,
    )

    assert {"dodge", "strike"} <= set(evidence.source_only_tokens)
    assert not verdict.accepted


def test_language_neutral_proper_name_uses_explicit_non_dialogue_policy() -> None:
    source = "KIM SHYEOK!"
    verdict = validate_target_language(
        source=source,
        target=source,
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=(source,),
            coverage_complete=True,
        ),
    )
    english = validate_target_language(
        source="MATCH.",
        target="MATCH.",
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=("MATCH.",),
            coverage_complete=True,
        ),
    )

    assert verdict.accepted
    assert verdict.policy_id == "source_neutral_proper_name"
    assert not english.accepted


def test_joined_ptbr_marker_and_nonlexical_text_do_not_exhaust_translation() -> None:
    joined = "VICE-MESTREDA"
    joined_verdict = validate_target_language(
        source=joined,
        target=joined,
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=(joined,),
            coverage_complete=True,
        ),
    )
    numeric = validate_target_language(
        source="000000",
        target="000000",
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=("000000",),
            coverage_complete=True,
        ),
    )

    assert joined_verdict.accepted
    assert joined_verdict.policy_id == "already_target_language"
    assert numeric.accepted
    assert numeric.policy_id == "source_neutral_nonlexical"


def test_unit_glyph_fragment_is_nonlexical_but_english_pronoun_still_retries() -> None:
    fragment = "t!"
    fragment_verdict = validate_target_language(
        source=fragment,
        target=fragment,
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=(fragment,),
            coverage_complete=True,
        ),
    )
    pronoun = "I!"
    pronoun_verdict = validate_target_language(
        source=pronoun,
        target=pronoun,
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=(pronoun,),
            coverage_complete=True,
        ),
    )

    assert fragment_verdict.accepted
    assert fragment_verdict.policy_id == "source_neutral_nonlexical"
    assert not pronoun_verdict.accepted


def test_structured_identifier_list_is_neutral_but_labelled_prose_still_retries() -> None:
    identifiers = "TL:ROK2343 PR:KILLSWITCH2315 RD:MOV TS:MOV CL:MOV"
    identifier_verdict = validate_target_language(
        source=identifiers,
        target=identifiers,
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=(identifiers,),
            coverage_complete=True,
        ),
    )
    prose = "NOTE: THIS SHOULD BE TRANSLATED"
    prose_verdict = validate_target_language(
        source=prose,
        target=prose,
        role="dialogue_body",
        page_language_evidence=build_page_language_evidence(
            texts=(prose,),
            coverage_complete=True,
        ),
    )

    assert identifier_verdict.accepted
    assert identifier_verdict.policy_id == "source_neutral_structured_identifiers"
    assert not prose_verdict.accepted


def test_page_language_evidence_tracks_material_english_but_not_ptbr() -> None:
    english = build_page_language_evidence(
        texts=("THE PLAYER HAS 10 KILLS",),
        coverage_complete=True,
    )
    ptbr = build_page_language_evidence(
        texts=("COMO ESPERADO, YUJEONG JA FEZ UM MOVIMENTO", "ANTES"),
        coverage_complete=True,
    )

    assert english.coverage_complete
    assert set(english.source_only_tokens) >= {"the", "player", "has", "kills"}
    assert ptbr == PageLanguageEvidence.build(
        coverage_complete=True,
        source_only_tokens=(),
    )


def test_verdict_is_canonical_and_hash_bound() -> None:
    verdict = validate_target_language(
        source="THE PLAYER HAS 2 KILLS",
        target="O JOGADOR TEM 2 ABATES",
        role="card",
    )

    assert verdict.canonical_json_bytes
    assert len(verdict.verdict_sha256) == 64
    assert verdict.from_dict(verdict.to_dict()) == verdict
