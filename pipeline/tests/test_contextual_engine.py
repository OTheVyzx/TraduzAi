from translator.contextual_engine import (
    MockTranslator,
    build_contextual_payload,
    parse_translation_response,
    translate_contextual,
)


GLOSSARY = [
    {"id": "orcs", "source": "ORCS", "target": "ORCS", "type": "race", "protect": True},
]


def _payload():
    return build_contextual_payload(
        work={"id": "work", "title": "Work"},
        chapter="1",
        page=33,
        source_language="en",
        target_language="pt-BR",
        glossary=GLOSSARY,
        previous_segments=[],
        segments=[{"id": "p033_r001", "source": "ORCS!", "bbox": [100, 200, 300, 80]}],
    )


def test_valid_json_response_restores_placeholders_and_used_glossary():
    payload = _payload()
    response = {"segments": [{"id": "p033_r001", "translation": "⟦TA_TERM_001⟧!", "confidence": 0.94, "used_glossary": ["ORCS"], "warnings": []}]}

    result = translate_contextual(payload, MockTranslator(response=response))

    assert result["segments"][0]["translation"] == "ORCS!"
    assert result["segments"][0]["used_glossary"] == ["ORCS"]
    assert result["segments"][0]["warnings"] == []


def test_invalid_json_does_not_continue_as_success():
    payload = _payload()

    result = translate_contextual(payload, MockTranslator(raw="not json"))

    assert result["segments"][0]["translation"] == "ORCS!"
    assert result["segments"][0]["qa_flags"] == ["needs_translation_review"]
    assert result["segments"][0]["warnings"][0].startswith("translation_api_failed")


def test_missing_segment_uses_safe_fallback():
    result = translate_contextual(_payload(), MockTranslator(response={"segments": []}))

    assert result["segments"][0]["warnings"] == ["missing_translation"]
    assert result["segments"][0]["qa_flags"] == ["needs_translation_review"]


def test_parse_rejects_json_without_segments():
    try:
        parse_translation_response("{}")
    except ValueError as exc:
        assert "segments" in str(exc)
    else:
        raise AssertionError("deveria rejeitar payload invalido")


def test_placeholder_warning_is_preserved():
    payload = _payload()
    response = {"segments": [{"id": "p033_r001", "translation": "ORCS!", "confidence": 0.3, "used_glossary": ["ORCS"], "warnings": ["low_confidence"]}]}

    result = translate_contextual(payload, MockTranslator(response=response))

    assert "placeholder_missing" in result["segments"][0]["warnings"]
    assert "low_confidence" in result["segments"][0]["warnings"]
