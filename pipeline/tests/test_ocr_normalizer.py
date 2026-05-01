from ocr.ocr_normalizer import normalize_ocr_record, normalize_ocr_text


def test_required_corrections_are_applied():
    cases = {
        "RAID SOUAD": "RAID SQUAD",
        "DRCS": "ORCS",
        "RDC": "ORCS",
        "CARBAGE": "GARBAGE",
        "TRAe": "TRAP",
        "FENRISNOW": "FENRIS NOW",
    }

    for raw, expected in cases.items():
        result = normalize_ocr_text(raw)
        assert result["normalized_ocr"] == expected
        assert result["normalization"]["changed"] is True


def test_low_confidence_common_word_is_not_fuzzy_corrected():
    result = normalize_ocr_text("THE", {"TREE": "arvore"})

    assert result["normalized_ocr"] == "THE"
    assert result["normalization"]["changed"] is False


def test_gibberish_is_flagged_and_skipped_in_record():
    record = normalize_ocr_record({"text": "///// 12345"})

    assert record["raw_ocr"] == "///// 12345"
    assert record["normalization"]["is_gibberish"] is True
    assert record["skip_processing"] is True
    assert "ocr_gibberish" in record["qa_flags"]


def test_stutter_uses_glossary_translation():
    result = normalize_ocr_text("Y-YOUNG MASTER?", {"YOUNG MASTER": "Jovem mestre"})

    assert result["normalized_ocr"] == "J-Jovem mestre?"
    assert result["normalization"]["corrections"][0]["reason"] == "stutter_glossary_translation"


def test_record_persists_raw_normalized_and_reason():
    record = normalize_ocr_record({"text": "RAID SOUAD"})

    assert record["raw_ocr"] == "RAID SOUAD"
    assert record["normalized_ocr"] == "RAID SQUAD"
    assert record["text"] == "RAID SQUAD"
    assert record["normalization"]["corrections"][0]["from"] == "RAID SOUAD"
