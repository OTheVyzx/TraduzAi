import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.text_normalizer import normalize_text
from ocr.contextual_reviewer import contextual_review_page


def test_joined_words_are_split_with_debug_metadata():
    result = normalize_text("IGETBACK TOWORK", text_id="ocr_017", confidence=0.56)

    assert result["raw"] == "IGETBACK TOWORK"
    assert result["normalized"] == "I GET BACK TO WORK"
    assert result["changed"] is True
    assert result["joined_word_suspect"] is True
    assert result["rules_applied"] == ["split_joined_words"]
    assert result["token_diff"] == [
        {"from": "IGETBACK", "to": "I GET BACK", "rule": "joined_word_suspect"},
        {"from": "TOWORK", "to": "TO WORK", "rule": "joined_word_suspect"},
    ]
    assert result["confidence_before"] == 0.56
    assert result["needs_review"] is True


def test_contextual_review_applies_normalization_trace_before_translation():
    page = {"texts": [{"id": "ocr_017", "text": "IGETBACK TOWORK", "confidence": 0.56}]}

    reviewed = contextual_review_page(page, [], [])

    text = reviewed["texts"][0]
    assert text["text"] == "I GET BACK TO WORK"
    assert text["normalization_trace"]["raw"] == "IGETBACK TOWORK"
    assert text["normalization_trace"]["joined_word_suspect"] is True
    assert "ocr_run_on_suspect" in text["qa_flags"]
