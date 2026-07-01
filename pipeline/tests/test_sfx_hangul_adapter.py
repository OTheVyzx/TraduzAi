from sfx import SfxAdaptation, adapt_hangul_sfx


def test_adapts_common_hangul_sfx_to_ptbr():
    assert adapt_hangul_sfx("쿵") == SfxAdaptation(
        source_text="쿵",
        adapted_text="TUM",
        confidence=0.9,
        kind="impact",
        review_required=False,
        qa_flags=[],
    )
    assert adapt_hangul_sfx("쾅").adapted_text == "BOOM"
    assert adapt_hangul_sfx("탕").kind == "shot"
    assert adapt_hangul_sfx("철컥").adapted_text == "CLAC"
    assert adapt_hangul_sfx("우우웅").adapted_text == "VUUUM"
    assert adapt_hangul_sfx("파앗").confidence == 0.76


def test_normalizes_punctuation_noise_around_hangul_sfx():
    result = adapt_hangul_sfx("쿵!!")

    assert result.source_text == "쿵"
    assert result.adapted_text == "TUM"
    assert result.review_required is False
    assert result.qa_flags == []


def test_normalizes_decomposed_hangul_jamo_for_lookup():
    result = adapt_hangul_sfx("쿵")

    assert result.source_text == "쿵"
    assert result.adapted_text == "TUM"
    assert result.review_required is False
    assert result.qa_flags == []


def test_unknown_hangul_sfx_requires_review():
    result = adapt_hangul_sfx("흐그르")

    assert result.source_text == "흐그르"
    assert result.adapted_text == "흐그르"
    assert result.confidence == 0.0
    assert result.kind == "unknown"
    assert result.review_required is True
    assert "unknown_sfx" in result.qa_flags


def test_normalizes_surrounding_and_internal_whitespace():
    result = adapt_hangul_sfx("  철 컥 \n")

    assert result.source_text == "철컥"
    assert result.adapted_text == "CLAC"
    assert result.review_required is False
    assert result.qa_flags == []


def test_unknown_hangul_with_punctuation_requires_review_after_noise_cleanup():
    result = adapt_hangul_sfx("\ud750\uadf8\ub974!!")

    assert result.source_text == "\ud750\uadf8\ub974"
    assert result.adapted_text == "\ud750\uadf8\ub974"
    assert result.kind == "unknown"
    assert result.review_required is True
    assert "unknown_sfx" in result.qa_flags


def test_mixed_hangul_latin_noise_requires_review_instead_of_confident_mapping():
    result = adapt_hangul_sfx("\ucff5abc")

    assert result.source_text == "\ucff5abc"
    assert result.adapted_text == "\ucff5abc"
    assert result.kind == "mixed_script"
    assert result.review_required is True
    assert "mixed_script_sfx" in result.qa_flags


def test_empty_sfx_requires_review_without_adapted_text():
    result = adapt_hangul_sfx(" \t\n")

    assert result.source_text == ""
    assert result.adapted_text == ""
    assert result.confidence == 0.0
    assert result.kind == "empty"
    assert result.review_required is True
    assert "empty_sfx" in result.qa_flags


def test_non_hangul_sfx_requires_review():
    result = adapt_hangul_sfx("BOOM")

    assert result.source_text == "BOOM"
    assert result.adapted_text == "BOOM"
    assert result.confidence == 0.0
    assert result.kind == "non_hangul"
    assert result.review_required is True
    assert "non_hangul_sfx" in result.qa_flags
