import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.contextual_reviewer import contextual_review_page
from ocr.text_normalizer import normalize_text


MINIMUM_JOINED_WORD_CASES = {
    "AISHIT'SNOT": "AISH! IT'S NOT",
    "CANYOUFINDAGOOD": "CAN YOU FIND A GOOD",
    "THATGIVESINTERESTUP": "THAT GIVES INTEREST UP",
    "TILLTHREEMONTHS": "TILL THREE MONTHS",
    "TOSHOWYOUR": "TO SHOW YOUR",
    "TOBELIEVE": "TO BELIEVE",
    "WE'REFOOL'S": "WE'RE FOOLS",
    "AJUMMAYOU": "AJUMMA, YOU",
    "THERE'SNO": "THERE'S NO",
    "GETMONEYFROM": "GET MONEY FROM",
    "ONLYTHINKS": "ONLY THINKS",
    "SOWHY": "SO WHY",
    "TOMORROW'SPROBLEMS": "TOMORROW'S PROBLEMS",
    "EVENTHINK": "EVEN THINK",
    "PAYUSBACK": "PAY US BACK",
    "CANDIE": "CAN DIE",
    "IDON'T": "I DON'T",
    "LET'SJUST": "LET'S JUST",
    "REAL-LIFEINSURANCE": "REAL-LIFE INSURANCE",
}


def test_joined_word_normalization_v2_covers_minimum_tokens():
    for raw, expected in MINIMUM_JOINED_WORD_CASES.items():
        result = normalize_text(raw, text_id=f"case-{raw}", confidence=0.82)

        assert result["normalized"] == expected
        assert result["changed"] is True
        assert result["joined_word_suspect"] is True
        assert result["confidence_after_estimate"] >= 0.7


def test_aish_interjection_is_not_split_as_ai_profanity():
    result = normalize_text("AISHIT'SNOT LIKE WE'REFOOL'S...", text_id="ocr_aish", confidence=0.82)

    assert result["normalized"] == "AISH! IT'S NOT LIKE WE'RE FOOLS..."
    assert "AI SHIT" not in result["normalized"]
    assert result["changed"] is True


def test_repairs_leading_insurance_fragment_before_translation():
    result = normalize_text(
        "ANCE, TE YOU YOU KNOW, REAL-LIFEINSURANCE, STUFF LIKE THAT? IF YOU DON'T HAVE MONEY, YOU HAVE TOSHOWYOUR SINCERITY.",
        text_id="ocr_insurance",
        confidence=0.82,
    )

    assert result["normalized"] == (
        "INSURANCE, YOU KNOW, REAL-LIFE INSURANCE, STUFF LIKE THAT? "
        "IF YOU DON'T HAVE MONEY, YOU HAVE TO SHOW YOUR SINCERITY."
    )
    assert not result["normalized"].startswith("ANCE,")
    assert "repair_leading_insurance_fragment" in result["rules_applied"]


def test_repairs_missing_spacing_after_dialogue_punctuation():
    result = normalize_text(
        "What!Then,why did we come to the cafe,what are you hiding?",
        text_id="ocr_cafe",
        confidence=0.82,
    )

    assert result["normalized"] == "What! Then, why did we come to the cafe, what are you hiding?"
    assert result["changed"] is True
    assert result["joined_word_suspect"] is False
    assert result["needs_review"] is False
    assert "repair_missing_punctuation_spacing" in result["rules_applied"]


def test_contextual_review_adds_canonical_normalized_text_and_preserves_raw_ocr():
    page = {
        "texts": [
            {
                "id": "ocr_013",
                "text": "WE'REFOOL'S",
                "confidence": 0.82,
                "raw_ocr": "WE'REFOOL'S",
            }
        ]
    }

    reviewed = contextual_review_page(page, [], [])

    text = reviewed["texts"][0]
    assert text["text"] == "WE'RE FOOLS"
    assert text["raw_ocr"] == "WE'REFOOL'S"
    assert text["normalized_text_final"] == "WE'RE FOOLS"
    assert text["normalization"]["changed"] is True
    assert text["normalization"]["confidence_after_estimate"] >= 0.7


def test_uncertain_joined_word_normalization_marks_review():
    page = {"texts": [{"id": "ocr_low", "text": "CANYOUFINDAGOOD", "confidence": 0.4}]}

    reviewed = contextual_review_page(page, [], [])

    text = reviewed["texts"][0]
    assert text["normalized_text_final"] == "CAN YOU FIND A GOOD"
    assert text["needs_review"] is True
    assert "ocr_joined_word_review" in text["qa_flags"]
