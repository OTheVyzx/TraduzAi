import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools.detectors import audit_mojibake, fix_mojibake, has_mojibake


def test_mojibake_detector_and_fixer_return_audit_friendly_metadata():
    text = "VOC\u00c3\u0192\u00c5\u00a0 SABE"

    audit = audit_mojibake(text, text_id="ocr_001", stage="translation_output")

    assert has_mojibake(text) is True
    assert fix_mojibake(text) == "VOC\u00ca SABE"
    assert audit["text_id"] == "ocr_001"
    assert audit["stage"] == "translation_output"
    assert audit["translated"] == text
    assert audit["mojibake_match_count"] >= 1
    assert audit["mojibake_samples"]
    assert audit["suggested_fix"] == "VOC\u00ca SABE"
    assert audit["fix_method"] == "decode_cp1252_encode_utf8_safe"
    assert audit["flags"] == ["mojibake_in_translation"]
    assert audit["mojibake_in_translation"] is True


def test_clean_text_has_empty_mojibake_audit():
    audit = audit_mojibake("VOCE SABE", text_id="ocr_002")

    assert has_mojibake("VOCE SABE") is False
    assert audit["mojibake_match_count"] == 0
    assert audit["suggested_fix"] == "VOCE SABE"
    assert audit["flags"] == []


def test_valid_portuguese_diacritics_are_not_mojibake():
    text = "ENTÃO, VOCÊ NÃO TEM CÂNCER?"
    audit = audit_mojibake(text, text_id="ocr_003")

    assert has_mojibake(text) is False
    assert audit["mojibake_match_count"] == 0
    assert audit["suggested_fix"] == text
    assert audit["flags"] == []
