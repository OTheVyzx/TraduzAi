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


def test_required_corrections_are_applied_inside_sentences():
    cases = {
        "Y-YOU'RE THAT CARBAGE GHISLAIN...?": "Y-YOU'RE THAT GARBAGE GHISLAIN...?",
        "A TRAe? ARE YOU TELLING ME": "A TRAP? ARE YOU TELLING ME",
        "DO NOT PANIC GET INTO OEFENSE FORMATION!": "DO NOT PANIC GET INTO DEFENSE FORMATION!",
        "LUTHANIA KINGDOME NEAR THE PERDIUM ESTATE": "LUTHANIA KINGDOM NEAR THE PERDIUM ESTATE",
        "BUT TS TO EARLY TO REJDICE": "BUT ITS TOO EARLY TO REJOICE",
        "IS THE DAY MOST PEOPlE IN THE RAID SQUAD WILL DIE": "IS THE DAY MOST PEOPLE IN THE RAID SQUAD WILL DIE",
        "GET NTO FORMATION!": "GET INTO FORMATION!",
        "I'LL LET YOU LIVE IF YOU ANSWER My Que$TIONS!": "I'LL LET YOU LIVE IF YOU ANSWER My QUESTIONS!",
        "BUT MY Household HAD Already BEEN CRUSHED Tio DUST WHEN RETURNED": "BUT MY Household HAD Already BEEN CRUSHED TO DUST WHEN RETURNED",
        "ANDAS MORETIME PASSED, MORE PEOPLE FROM THE HOUSEHOLDAUOIDEDME": "AND AS MORE TIME PASSED, MORE PEOPLE FROM THE HOUSEHOLD AVOIDED ME",
        "IDIDN'T EXPECT YOU TO HAVE BEEN THE OLDEST SON": "I DIDN'T EXPECT YOU TO HAVE BEEN THE OLDEST SON",
        "PERDIUMS' SOLDIER RICARDOE": "PERDIUMS' SOLDIER RICARDO",
        "DWAS UNABLE TO HIRHSTAND TRHE SRIIGSMIANDRANAWAY FROM HOME": "WAS UNABLE TO WITHSTAND THE STIGMA AND RAN AWAY FROM HOME",
        "SURVIVED COUNTLESS BAUES. BUTUP MY SADLAND MADE A NAME FOR MYSELF": "SURVIVED COUNTLESS BATTLES. BUILT UP MY SKILLS AND MADE A NAME FOR MYSELF",
        "ALMDST ALL Of Us ENDED IP DYNG BECAUSE HAD BEEN STUBBORN AND WANTED TO COMMAND THE SQUAD": "ALMOST ALL OF US ENDED UP DYING BECAUSE HAD BEEN STUBBORN AND WANTED TO COMMAND THE SQUAD",
        "YOUNG MASTER.P": "YOUNG MASTER.",
        "GHISLAIN PERDIUM, THEMERCENARYKING, AND ONE OF THE CONTINENT'S SEVENSTRONGESTMEN.": "GHISLAIN PERDIUM, THE MERCENARY KING, AND ONE OF THE CONTINENT'S SEVEN STRONGEST MEN.",
        "WELL, THERE'SNOPOINTIN EXPLAINING ITFURTHER SINCE YOU'RE GONNA BE DEAD SOON.": "WELL, THERE'S NO POINT IN EXPLAINING IT FURTHER SINCE YOU'RE GONNA BE DEAD SOON.",
        "IT'S JUSTA SHAME THAT I WAS UNABLE TOFULFILL MY REVENGE...": "IT'S JUST A SHAME THAT I WAS UNABLE TO FULFILL MY REVENGE...",
        "YOURE TELLING ME THAT THE best %NIGHT OF THE NORTH!": "YOURE TELLING ME THAT THE best KNIGHT OF THE NORTH!",
        "knew rt ALL FROM THE STHRT.": "knew it ALL FROM THE START.",
        "BACK TTRAVELED TO THE PAST?": "HAVE I TRAVELED BACK TO THE PAST?",
        "TRHE IMMATIURITYBORNE FROM FEEUING INFERIOR LED TO TROUBLE": "THE IMMATURITY BORNE FROM FEELING INFERIOR LED TO TROUBLE",
        "RETURNED TO THE PERDIUM County In ORDER TO MAKE UP FOR MY past MSTAES": "RETURNED TO THE PERDIUM County In ORDER TO MAKE UP FOR MY past MISTAKES",
        "IN MY PAST lfe HAD FOLLOWED THE DRC Rald SQUAD ln AN ATTEMPT TO MAKE A NAME FOR Myself": "IN MY PAST LIFE HAD FOLLOWED THE ORC RAID SQUAD IN AN ATTEMPT TO MAKE A NAME FOR Myself",
        "IT SHOUID BE DOABLE ENOUGH SINCETHOSE ORCS DIDN'T REALLY HAVEA STRATEGY": "IT SHOULD BE DOABLE ENOUGH SINCE THOSE ORCS DIDN'T REALLY HAVE A STRATEGY",
        "THE Problem IS IF THESE GUYS Wlll TRUST ME SINCE THEY ALL THIN% I'M SOME Useless BASTARD.": "THE Problem IS IF THESE GUYS WILL TRUST ME SINCE THEY ALL THINK I'M SOME Useless BASTARD.",
        "One of the continents top seven Noble Knight ldun": "One of the continents top seven Noble Knight Idun",
    }

    for raw, expected in cases.items():
        result = normalize_ocr_text(raw)
        assert result["normalized_ocr"] == expected
        assert result["normalization"]["changed"] is True


def test_scanlation_credit_is_skipped_in_record():
    for raw in ["ASURASOANS. COM", "FOR THE FASTEST RELEASES", "ILEAFSKY PR"]:
        record = normalize_ocr_record({"text": raw})

        assert record["text"] == raw
        assert record["skip_processing"] is True
        assert "scanlation_credit" in record["qa_flags"]
        assert record["skip_reason"] == "scanlation_credit"


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


def test_known_low_confidence_latin_artifact_is_skipped_in_record():
    record = normalize_ocr_record({"text": "Hfor"})

    assert record["text"] == "Hfor"
    assert record["skip_processing"] is True
    assert record["skip_reason"] == "ocr_artifact"
    assert "suspected_ocr_error" in record["qa_flags"]


def test_short_quoted_dialogue_is_not_marked_gibberish():
    record = normalize_ocr_record({"text": '"WE"?!'})

    assert record["text"] == '"WE"?!'
    assert record["normalization"]["is_gibberish"] is False
    assert record.get("skip_processing") is not True


def test_short_korean_dialogue_is_not_marked_gibberish():
    record = normalize_ocr_record({"text": "뭐?!"})

    assert record["text"] == "뭐?!"
    assert record["normalization"]["is_gibberish"] is False
    assert record.get("skip_processing") is not True


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
