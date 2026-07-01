import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr import ocr_normalizer
from ocr.ocr_normalizer import merge_same_balloon_fragments_before_translation, normalize_ocr_record, normalize_ocr_text


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
        "SYNCHRONIZED BUMIANS WILL ATTACK ENEMIES": "SYNCHRONIZED HUMANS WILL ATTACK ENEMIES",
    }

    for raw, expected in cases.items():
        result = normalize_ocr_text(raw)
        assert result["normalized_ocr"] == expected
        assert result["normalization"]["changed"] is True


def test_punctuation_joined_dialogue_is_repaired_before_review_route():
    record = normalize_ocr_record(
        {
            "text": "What!Then,why did we come to the cafe,what are you hiding?",
            "confidence": 0.82,
            "bbox": [32, 120, 312, 198],
            "balloon_bbox": [20, 88, 344, 226],
            "tipo": "fala",
            "content_class": "dialogue",
        }
    )

    assert record["text"] == "What! Then, why did we come to the cafe, what are you hiding?"
    assert record["normalized_ocr"] == "What! Then, why did we come to the cafe, what are you hiding?"
    assert record["route_action"] == "translate_inpaint_render"
    assert record["route_reason"] == "dialogue_balloon_with_english_text"
    assert record.get("needs_review") is not True
    assert "ocr_truncated_or_joined" not in record.get("qa_flags", [])
    assert record["normalization"]["corrections"][0]["reason"] == "repair_missing_punctuation_spacing"


def test_synchronized_humans_dialogue_is_not_false_joined_review():
    record = normalize_ocr_record(
        {
            "text": "SYNCHRONIZED BUMIANS WILL ATTACK ENEMIES",
            "confidence": 0.94,
            "bbox": [191, 6599, 514, 6710],
            "balloon_bbox": [190, 6592, 519, 6717],
            "line_polygons": [
                [[212, 6599], [495, 6599], [495, 6629], [212, 6629]],
                [[228, 6641], [481, 6641], [481, 6667], [228, 6667]],
                [[191, 6680], [514, 6680], [514, 6710], [191, 6710]],
            ],
        }
    )

    assert record["text"] == "SYNCHRONIZED HUMANS WILL ATTACK ENEMIES"
    assert record["normalized_text_final"] == "SYNCHRONIZED HUMANS WILL ATTACK ENEMIES"
    assert record["route_action"] == "translate_inpaint_render"
    assert record["route_reason"] == "dialogue_balloon_with_english_text"
    assert record.get("needs_review") is not True
    assert "ocr_truncated_or_joined" not in record.get("qa_flags", [])


def test_same_band_dependent_fragments_merge_before_translation_without_exact_bubble_match():
    records = [
        {
            "id": "ocr_004",
            "text": "THE INTEREST WAS ALREADY REDUCEDBY MORE THAN THREE TIMES",
            "bbox": [527, 7113, 688, 7221],
            "text_pixel_bbox": [527, 7113, 688, 7221],
            "balloon_bbox": [485, 7068, 730, 7255],
            "bubble_mask_bbox": [461, 7044, 754, 7279],
            "band_id": "page_002_band_007",
            "trace_id": "ocr_004@page_002_band_007",
        },
        {
            "id": "ocr_003",
            "text": "THE PRINCIPAL",
            "bbox": [555, 7209, 661, 7221],
            "text_pixel_bbox": [555, 7209, 661, 7221],
            "balloon_bbox": [12, 6636, 656, 7291],
            "bubble_mask_bbox": [12, 6636, 656, 7291],
            "band_id": "page_002_band_007",
            "trace_id": "ocr_003@page_002_band_007",
        },
    ]

    merged = merge_same_balloon_fragments_before_translation(records)

    assert len(merged) == 1
    assert merged[0]["normalized_text_final"] == (
        "THE INTEREST WAS ALREADY REDUCED BY MORE THAN THREE TIMES THE PRINCIPAL"
    )
    assert merged[0]["source_text_ids"] == ["ocr_004", "ocr_003"]
    assert merged[0]["source_trace_ids"] == ["ocr_004@page_002_band_007", "ocr_003@page_002_band_007"]
    assert "same_balloon_fragment_merged" in merged[0]["qa_flags"]
    assert "ocr_joined_repaired" in merged[0]["qa_flags"]


def test_same_band_dependent_fragments_merge_common_tail_cases_before_translation():
    cases = [
        (
            [
                ("ocr_001", "PLEASE, FOR", [502, 4528, 642, 4580]),
                ("ocr_002", "THE CHILD'S SAKE.", [500, 4578, 674, 4661]),
            ],
            "PLEASE, FOR THE CHILD'S SAKE.",
        ),
        (
            [
                ("ocr_003", "After All, IT'S CANCER, WHYBOTHER USING APRIVATE LOAN FOR A Patient?YOUR LIFE IS SO", [298, 12404, 703, 12548]),
                ("ocr_005", "FRUSTRATINGTOO", [445, 12535, 620, 12596]),
            ],
            "After All, IT'S CANCER, WHY BOTHER USING A PRIVATE LOAN FOR A Patient? YOUR LIFE IS SO FRUSTRATING TOO",
        ),
        (
            [
                ("ocr_001", "OPPA,", [118, 4493, 258, 4528]),
                ("ocr_002", "WHY ARE YOU SOLATE~", [111, 4526, 303, 4602]),
            ],
            "OPPA, WHY ARE YOU SOLATE~",
        ),
    ]

    for parts, expected in cases:
        records = [
            {
                "id": text_id,
                "text": text,
                "bbox": bbox,
                "text_pixel_bbox": bbox,
                "band_id": "page_999_band_001",
                "trace_id": f"{text_id}@page_999_band_001",
            }
            for text_id, text, bbox in parts
        ]

        merged = merge_same_balloon_fragments_before_translation(records)

        assert len(merged) == 1
        assert merged[0]["normalized_text_final"] == expected
        assert merged[0]["source_text_ids"] == [part[0] for part in parts]


def test_same_balloon_merge_drops_duplicate_sentence_fragment_before_translation():
    records = [
        {
            "id": "ocr_good",
            "text": "The subspace retention is only five minutes",
            "bbox": [131, 112, 312, 231],
            "text_pixel_bbox": [131, 112, 312, 231],
            "bubble_mask_bbox": [56, 28, 713, 593],
            "band_id": "page_005_band_078",
            "trace_id": "ocr_good@page_005_band_078",
            "confidence": 0.91,
        },
        {
            "id": "ocr_low_fragment",
            "text": "space is only utes. If you exceed that time, you will return to your original world!",
            "bbox": [399, 270, 675, 401],
            "text_pixel_bbox": [399, 270, 675, 401],
            "bubble_mask_bbox": [56, 28, 713, 593],
            "band_id": "page_005_band_078",
            "trace_id": "ocr_low_fragment@page_005_band_078",
            "confidence": 0.0,
        },
    ]

    merged = merge_same_balloon_fragments_before_translation(records)

    assert len(merged) == 1
    assert merged[0]["normalized_text_final"] == (
        "The subspace retention is only five minutes If you exceed that time, you will return to your original world!"
    )
    assert "space is only utes" not in merged[0]["normalized_text_final"]
    assert "ocr_joined_repaired" in merged[0]["qa_flags"]


def test_dark_connected_lobes_do_not_merge_before_translation():
    common = {
        "bubble_mask_bbox": [56, 28, 713, 593],
        "bubble_mask_source": "image_dark_bubble_mask",
        "layout_profile": "dark_bubble",
        "band_id": "page_005_band_078",
        "qa_flags": ["dark_bubble_oval_reocr"],
    }
    records = [
        {
            **common,
            "id": "ocr_001",
            "text": "The subspace retention is only five minutes",
            "bbox": [131, 112, 312, 231],
            "layout_bbox": [131, 112, 312, 231],
            "text_pixel_bbox": [131, 112, 312, 231],
            "bubble_id": "page_005_band_078_partial_dark_lobe_1000",
            "trace_id": "ocr_001@page_005_band_078",
        },
        {
            **common,
            "id": "ocr_001_002",
            "text": "space is only utes. If you exceed that time, you will return to your original world!",
            "bbox": [399, 204, 675, 335],
            "layout_bbox": [399, 204, 675, 335],
            "text_pixel_bbox": [399, 204, 675, 335],
            "bubble_id": "page_005_band_078_partial_dark_lobe_1001",
            "trace_id": "ocr_001_002@page_005_band_078",
        },
    ]

    merged = merge_same_balloon_fragments_before_translation(records)

    assert len(merged) == 2
    assert [item["id"] for item in merged] == ["ocr_001", "ocr_001_002"]
    assert "same_balloon_fragment_merged" not in (merged[0].get("qa_flags") or [])


def test_joined_ocr_is_repaired_before_review_flag_survives():
    assert hasattr(ocr_normalizer, "repair_ocr_truncated_or_joined")
    repair_ocr_truncated_or_joined = ocr_normalizer.repair_ocr_truncated_or_joined

    repaired = repair_ocr_truncated_or_joined(
        {
            "text": "What!Then,why did we come to the cafe,what are you hiding?",
            "bbox": [24, 18, 300, 120],
            "qa_flags": ["ocr_truncated_or_joined"],
            "route_action": "review_required",
            "route_reason": "ocr_truncated_or_joined",
            "line_polygons": [
                [[24, 18], [290, 18], [290, 50], [24, 50]],
                [[24, 54], [280, 54], [280, 88], [24, 88]],
            ],
        }
    )

    assert repaired["text"] == "What! Then, why did we come to the cafe, what are you hiding?"
    assert repaired.get("ocr_repair_status") == "repaired"
    assert "ocr_truncated_or_joined" not in repaired.get("qa_flags", [])
    assert repaired.get("route_action") not in {"review_required", "preserve_original"}
    assert repaired.get("needs_review") is not True


def test_scanlation_credit_is_suppressed_before_translation_and_inpaint():
    for raw in [
        "ASURASOANS. COM",
        "FOR THE FASTEST RELEASES",
        "ILEAFSKY PR",
        "The God of Death SUPPORTUS ON ko-fi.com/Secretscans patreon.com/Secretscans JOIN US AT DISCORD Discordggxzeknv",
        "WE ARE RECRUITING!",
        "WE ARE LOOKING FOR,",
        "READ FIRSTAT:",
        "TL Kiki Pr Yuu CI Shadow Erian Ts Qc Shadow Rp Shadow",
    ]:
        record = normalize_ocr_record({"text": raw})

        assert record["text"] == raw
        assert record["route_action"] == "review_required"
        assert record["skip_processing"] is True
        assert record["skip_reason"] == "scanlation_credit_suppressed"
        assert record["route_reason"] == "scanlation_credit_suppressed"
        assert record["content_class"] == "text"


def test_no_dot_scanlation_credit_domain_artifact_is_suppressed():
    for raw in ("sb3ag9 NEWIOKLGOCOM", "TRAIOIE a3aoag9 CKVGCOM"):
        record = normalize_ocr_record({"text": raw})

        assert record["text"] == raw
        assert record["route_action"] == "review_required"
        assert record["route_reason"] == "scanlation_credit_suppressed"
        assert record["skip_processing"] is True
        assert record["content_class"] == "text"


def test_email_like_scanlation_credit_is_suppressed():
    record = normalize_ocr_record({"text": "Kgm sini_@naver.com llshinheell choiplinonavercom kwangn"})

    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "scanlation_credit_suppressed"
    assert record["skip_processing"] is True
    assert record["content_class"] == "text"


def test_scanlation_credit_overrides_translate_route_action():
    record = normalize_ocr_record(
        {
            "text": "Kgm sini_@naver.com llshinheell choiplinonavercom kwangn",
            "route_action": "translate_inpaint_render",
            "route_reason": "dialogue_balloon_with_english_text",
        }
    )

    assert record["content_class"] == "text"
    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "scanlation_credit_suppressed"
    assert record["skip_processing"] is True


def test_hyphenated_credit_name_list_is_suppressed():
    record = normalize_ocr_record({"text": "-NISMOI29O -ARCTICWOLF"})

    assert record["content_class"] == "text"
    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "scanlation_credit_suppressed"
    assert record["skip_processing"] is True


def test_hyphenated_credit_name_list_with_trailing_names_does_not_override_review_route():
    record = normalize_ocr_record(
        {
            "text": "-KANJI2E2 -NEONNIGHTMARE -DRAGON EMPRYEAN SHADOWLESS",
            "content_class": "narration",
            "route_action": "review_required",
            "route_reason": "ocr_truncated_or_joined",
            "needs_review": True,
            "qa_flags": [
                "ocr_run_on_suspect",
                "ocr_truncated_or_joined",
                "mask_outside_balloon_critical",
            ],
        }
    )

    assert record["content_class"] == "text"
    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_truncated_or_joined"
    assert record.get("needs_review") is True


def test_low_confidence_common_word_is_not_fuzzy_corrected():
    result = normalize_ocr_text("THE", {"TREE": "arvore"})

    assert result["normalized_ocr"] == "THE"
    assert result["normalization"]["changed"] is False


def test_gibberish_is_flagged_and_skipped_in_record():
    record = normalize_ocr_record({"text": "///// 12345"})

    assert record["raw_ocr"] == "///// 12345"
    assert record["normalization"]["is_gibberish"] is True
    assert record["route_action"] == "translate_inpaint_render"
    assert record["skip_processing"] is False
    assert "ocr_gibberish" in record["qa_flags"]


def test_known_low_confidence_latin_artifact_is_skipped_in_record():
    record = normalize_ocr_record({"text": "Hfor"})

    assert record["text"] == "Hfor"
    assert record["route_action"] == "translate_inpaint_render"
    assert record["skip_processing"] is False
    assert "suspected_ocr_error" in record["qa_flags"]


def test_short_consonant_art_fragment_routes_to_review_without_skip():
    record = normalize_ocr_record(
        {
            "text": "MRAL",
            "bbox": [246, 1396, 538, 1602],
            "text_pixel_bbox": [246, 1396, 538, 1602],
            "background_rgb": [154, 169, 93],
            "line_polygons": [
                [[283, 1396], [538, 1459], [502, 1602], [246, 1539]],
            ],
        }
    )

    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_art_fragment_suspected"
    assert record["skip_processing"] is False
    assert record["content_class"] == "text"
    assert "ocr_art_fragment_suspected" in record["qa_flags"]


def test_short_art_fragment_without_line_polygons_routes_to_review_without_skip():
    record = normalize_ocr_record(
        {
            "text": "YEWTOI",
            "bbox": [290, 9166, 429, 9226],
            "text_pixel_bbox": [290, 9166, 429, 9226],
            "background_rgb": [95, 105, 124],
            "line_polygons": [],
        }
    )

    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_art_fragment_suspected"
    assert record["skip_processing"] is False
    assert "ocr_art_fragment_suspected" in record["qa_flags"]


def test_short_dark_visual_reocr_sfx_routes_to_review_without_inpaint():
    record = normalize_ocr_record(
        {
            "text": "WU",
            "bbox": [211, 3265, 319, 3375],
            "text_pixel_bbox": [214, 3274, 249, 3308],
            "line_polygons": [
                [[247, 3265], [319, 3298], [284, 3375], [211, 3342]],
            ],
            "qa_flags": [
                "candidate_crop_direct_paddle_reocr",
                "dark_bubble_oval_reocr",
                "partial_dark_bubble_lobe_reocr",
                "detected_dark_bubble_without_text_reocr",
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
            ],
            "bubble_mask_source": "image_dark_bubble_mask",
            "layout_profile": "dark_bubble",
            "block_profile": "dark_bubble",
        }
    )

    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_art_fragment_suspected"
    assert record["skip_processing"] is False
    assert "ocr_art_fragment_suspected" in record["qa_flags"]


def test_mixed_digit_repeated_letter_art_fragment_routes_to_review():
    record = normalize_ocr_record(
        {
            "text": "21848 OOOO!",
            "bbox": [635, 1248, 689, 1300],
            "text_pixel_bbox": [639, 1251, 687, 1300],
            "background_rgb": [205, 175, 191],
            "line_polygons": [[[635, 1248], [689, 1248], [689, 1300], [635, 1300]]],
            "confidence": 0.769,
        }
    )

    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_art_fragment_suspected"
    assert record["needs_review"] is True
    assert "ocr_gibberish" in record["qa_flags"]
    assert "ocr_art_fragment_suspected" in record["qa_flags"]


def test_gibberish_art_fragment_without_line_polygons_routes_to_review_without_skip():
    record = normalize_ocr_record(
        {
            "text": "000 000.",
            "bbox": [9, 6360, 152, 6458],
            "text_pixel_bbox": [27, 6360, 152, 6414],
            "background_rgb": [88, 103, 115],
            "line_polygons": [],
        }
    )

    assert record["normalization"]["is_gibberish"] is True
    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_art_fragment_suspected"
    assert record["skip_processing"] is False
    assert "ocr_gibberish" in record["qa_flags"]
    assert "ocr_art_fragment_suspected" in record["qa_flags"]


def test_existing_art_fragment_flag_overrides_translate_route_without_skip():
    record = normalize_ocr_record(
        {
            "text": "C-CUT!!!",
            "bbox": [614, 4702, 988, 4823],
            "qa_flags": ["ocr_art_fragment_suspected"],
            "route_action": "translate_inpaint_render",
            "route_reason": "dialogue_balloon_with_english_text",
        }
    )

    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_art_fragment_suspected"
    assert record["skip_processing"] is False
    assert record["content_class"] == "text"


def test_raw_text_missing_fast_fill_evidence_overrides_translate_route_without_skip():
    record = normalize_ocr_record(
        {
            "text": "? doy",
            "bbox": [191, 1502, 600, 1655],
            "qa_flags": ["raw_text_evidence_missing", "fast_fill_no_glyph_evidence"],
            "route_action": "translate_inpaint_render",
            "route_reason": "dialogue_balloon_with_english_text",
        }
    )

    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_visual_evidence_missing"
    assert record["skip_processing"] is False
    assert record["content_class"] == "text"


def test_art_suspected_scanlation_credit_overrides_translate_route_without_skip():
    record = normalize_ocr_record(
        {
            "text": "HIVETOON. COM",
            "bbox": [1054, 6731, 1280, 6764],
            "qa_flags": ["render_on_art_suspected"],
            "route_action": "translate_inpaint_render",
            "route_reason": "dialogue_balloon_with_english_text",
        }
    )

    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_visual_art_suspected"
    assert record["skip_processing"] is False
    assert record["content_class"] == "text"


def test_numeric_art_fragment_routes_to_review_without_skip():
    record = normalize_ocr_record(
        {
            "text": "712",
            "bbox": [55, 5117, 180, 5209],
            "text_pixel_bbox": [55, 5130, 82, 5144],
            "background_rgb": [158, 96, 56],
            "line_polygons": [[[55, 5130], [93, 5130], [93, 5144], [55, 5144]]],
        }
    )

    assert record["normalization"]["is_gibberish"] is True
    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_art_fragment_suspected"
    assert record["skip_processing"] is False
    assert "ocr_gibberish" in record["qa_flags"]
    assert "ocr_art_fragment_suspected" in record["qa_flags"]


def test_single_letter_large_art_fragment_routes_to_review_without_skip():
    record = normalize_ocr_record(
        {
            "text": "M",
            "bbox": [54, 8805, 320, 9065],
            "text_pixel_bbox": [54, 8805, 320, 9065],
            "background_rgb": [254, 255, 249],
            "line_polygons": [
                [[54, 8785], [320, 8785], [320, 9065], [54, 9065]],
            ],
        }
    )

    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "ocr_art_fragment_suspected"
    assert record["skip_processing"] is False
    assert "ocr_art_fragment_suspected" in record["qa_flags"]


def test_short_quoted_dialogue_is_not_marked_gibberish():
    record = normalize_ocr_record({"text": '"WE"?!'})

    assert record["text"] == '"WE"?!'
    assert record["normalization"]["is_gibberish"] is False
    assert record["route_action"] == "translate_inpaint_render"
    assert record.get("skip_processing") is not True


def test_normal_dialogue_gets_translate_route_action():
    record = normalize_ocr_record({"text": "What are you doing here?"})

    assert record["text"] == "What are you doing here?"
    assert record["route_action"] == "translate_inpaint_render"
    assert record["route_reason"] == "dialogue_balloon_with_english_text"
    assert record["skip_processing"] is False


def test_short_korean_dialogue_is_not_marked_gibberish():
    record = normalize_ocr_record({"text": "뭐?!"})

    assert record["text"] == "뭐?!"
    assert record["normalization"]["is_gibberish"] is False
    assert record.get("skip_processing") is not True


def test_standard_scene_text_routes_to_review_before_inpaint():
    record = normalize_ocr_record(
        {
            "text": "CORNER",
            "bbox": [440, 120, 540, 145],
            "layout_profile": "standard",
            "block_profile": "standard",
            "background_rgb": [230, 220, 190],
            "route_action": "translate_inpaint_render",
        }
    )

    assert record["route_action"] == "review_required"
    assert record["route_reason"] == "non_balloon_scene_text"
    assert record["skip_processing"] is False
    assert "non_balloon_scene_text_review" in record["qa_flags"]


def test_real_bubble_single_token_text_is_not_scene_filtered():
    record = normalize_ocr_record(
        {
            "text": "WAIT",
            "bbox": [67, 123, 172, 137],
            "balloon_bbox": [39, 107, 199, 152],
            "bubble_id": "bubble_001",
            "bubble_mask_bbox": [39, 107, 199, 152],
            "bubble_inner_bbox": [50, 114, 188, 146],
            "layout_profile": "white_balloon",
            "block_profile": "white_balloon",
        }
    )

    assert record["route_action"] == "translate_inpaint_render"
    assert record.get("route_reason") != "non_balloon_scene_text"


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
