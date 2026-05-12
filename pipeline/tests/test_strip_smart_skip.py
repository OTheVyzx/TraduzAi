import pytest

from pipeline.strip.smart_skip import (
    CATEGORY_CREDIT_OR_WATERMARK,
    CATEGORY_NOT_SAFE_TO_SKIP,
    CATEGORY_TIMER_OR_UI,
    annotate_page_with_smart_skip_shadow,
    classify_text_for_skip,
)


@pytest.mark.parametrize(
    "text",
    [
        "All comics on this website are just previews...",
        "For the original version, please buy the comic...",
        "READ On",
        "FOR FASTER UPDATE",
    ],
)
def test_classifies_first_page_site_credit_as_safe_skip_candidate(text):
    decision = classify_text_for_skip(
        text,
        page_number=1,
        confidence=0.0,
        bbox=(10, 10, 200, 80),
    )

    assert decision.safe_to_skip is True
    assert decision.category == CATEGORY_CREDIT_OR_WATERMARK
    assert decision.reason


@pytest.mark.parametrize("text", ["00:00:05", "oo:oo:os", "Oo:OO:OS"])
def test_classifies_timer_or_ui_noise_as_safe_skip_candidate(text):
    decision = classify_text_for_skip(
        text,
        page_number=1,
        confidence=0.0,
        bbox=(10, 10, 120, 40),
    )

    assert decision.safe_to_skip is True
    assert decision.category == CATEGORY_TIMER_OR_UI


@pytest.mark.parametrize(
    "text",
    [
        "AH, AH, MIC TEST.",
        "IS THIS RECORDING?",
        "I can't go back now.",
    ],
)
def test_keeps_dialogue_and_narration_out_of_auto_skip(text):
    decision = classify_text_for_skip(
        text,
        page_number=1,
        confidence=0.9,
        bbox=(10, 10, 300, 120),
    )

    assert decision.safe_to_skip is False
    assert decision.category == CATEGORY_NOT_SAFE_TO_SKIP


def test_does_not_auto_skip_ambiguous_read_on_outside_opening_page():
    decision = classify_text_for_skip(
        "READ On",
        page_number=8,
        confidence=0.8,
        bbox=(10, 10, 200, 80),
    )

    assert decision.safe_to_skip is False
    assert decision.category == CATEGORY_NOT_SAFE_TO_SKIP


def test_shadow_annotation_records_candidates_without_mutating_skip_processing():
    page = {
        "numero": 1,
        "texts": [
            {
                "id": "credit",
                "text": "All comics on this website are just previews...",
                "confidence": 0.0,
                "bbox": [10, 10, 200, 80],
                "skip_processing": False,
            },
            {
                "id": "dialogue",
                "text": "IS THIS RECORDING?",
                "confidence": 0.9,
                "bbox": [10, 120, 240, 200],
                "skip_processing": False,
            },
        ],
    }

    result = annotate_page_with_smart_skip_shadow(page)

    assert result is page
    assert page["texts"][0]["skip_processing"] is False
    assert page["texts"][1]["skip_processing"] is False
    assert page["_smart_skip_shadow"]["candidate_count"] == 1
    assert page["_smart_skip_shadow"]["not_safe_count"] == 1
    assert page["_smart_skip_shadow"]["candidates"][0]["text_id"] == "credit"
