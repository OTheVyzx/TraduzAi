from pipeline.ocr.macro_ocr import (
    MacroOcrLine,
    BandWindow,
    collect_page_ocr_blocks,
    collect_band_windows_from_project,
    classify_ocr_text_difference,
    compare_aligned_macro_ocr_texts,
    estimate_macro_ocr_fallback_cost,
    estimate_macro_ocr_shadow,
    map_macro_lines_to_bands,
)


def test_collect_band_windows_infers_page_local_ranges_from_strip_offsets():
    project = {
        "paginas": [
            {"numero": 1, "page_profile": {"y_in_strip_top": 0, "y_in_strip_bottom": 1000}},
            {"numero": 2, "page_profile": {"y_in_strip_top": 1000, "y_in_strip_bottom": 2000}},
        ]
    }
    project["paginas"][0]["page_profile"]["strip_perf_summary"] = {
        "entries": [
            {"band_index": 0, "y_top": 100, "y_bottom": 200, "durations_sec": {"ocr": 1.0}},
            {"band_index": 1, "y_top": 1200, "y_bottom": 1350, "durations_sec": {"ocr": 2.0}},
        ],
        "durations_sec": {"ocr": 3.0},
    }

    windows = collect_band_windows_from_project(project)

    assert windows == [
        BandWindow(band_index=0, page_number=1, y_top=100, y_bottom=200),
        BandWindow(band_index=1, page_number=2, y_top=200, y_bottom=350),
    ]


def test_map_macro_lines_to_bands_maps_by_page_and_vertical_center():
    bands = [
        BandWindow(band_index=0, page_number=1, y_top=0, y_bottom=100),
        BandWindow(band_index=1, page_number=1, y_top=120, y_bottom=240),
        BandWindow(band_index=2, page_number=2, y_top=0, y_bottom=100),
    ]
    lines = [
        MacroOcrLine(text="HELLO", bbox=(10, 20, 80, 60), confidence=0.9, page_number=1),
        MacroOcrLine(text="WORLD", bbox=(10, 140, 80, 180), confidence=0.9, page_number=1),
        MacroOcrLine(text="PAGE TWO", bbox=(10, 20, 80, 60), confidence=0.9, page_number=2),
    ]

    mappings = map_macro_lines_to_bands(lines, bands)

    assert [mapping.band_index for mapping in mappings] == [0, 1, 2]
    assert all(mapping.status == "mapped" for mapping in mappings)


def test_map_macro_lines_to_bands_marks_boundary_lines_as_fallback():
    bands = [
        BandWindow(band_index=0, page_number=1, y_top=0, y_bottom=100),
        BandWindow(band_index=1, page_number=1, y_top=100, y_bottom=200),
    ]
    lines = [
        MacroOcrLine(text="CROSS", bbox=(10, 80, 80, 120), confidence=0.9, page_number=1),
        MacroOcrLine(text="MISS", bbox=(10, 220, 80, 260), confidence=0.9, page_number=1),
    ]

    mappings = map_macro_lines_to_bands(lines, bands)

    assert mappings[0].status == "fallback"
    assert mappings[0].band_index == 0
    assert mappings[1].status == "missing"
    assert mappings[1].band_index is None


def test_estimate_macro_ocr_shadow_reports_expected_savings_and_rates():
    project = {
        "paginas": [
            {
                "numero": 1,
                "page_profile": {"y_in_strip_top": 0, "y_in_strip_bottom": 1000},
                "text_layers": [{"text": "HELLO", "bbox": [10, 20, 80, 60], "confidence": 0.9}],
            },
            {
                "numero": 2,
                "page_profile": {"y_in_strip_top": 1000, "y_in_strip_bottom": 2000},
                "text_layers": [{"text": "WORLD", "bbox": [10, 20, 80, 60], "confidence": 0.9}],
            },
        ]
    }
    project["paginas"][0]["page_profile"]["strip_perf_summary"] = {
        "entries": [
            {"band_index": 0, "y_top": 0, "y_bottom": 100, "durations_sec": {"ocr": 10.0}},
            {"band_index": 1, "y_top": 1000, "y_bottom": 1100, "durations_sec": {"ocr": 10.0}},
            {"band_index": 2, "y_top": 1200, "y_bottom": 1300, "durations_sec": {"ocr": 10.0}},
            {"band_index": 3, "y_top": 1400, "y_bottom": 1500, "durations_sec": {"ocr": 10.0}},
        ],
        "durations_sec": {"ocr": 40.0},
    }

    report = estimate_macro_ocr_shadow(project)

    assert report.current_ocr_band_calls == 4
    assert report.macro_window_count == 2
    assert report.estimated_savings_seconds == 20.0
    assert report.missing_text_rate == 0.0
    assert report.fallback_rate == 0.0


def test_compare_aligned_macro_ocr_texts_counts_missing_and_exact_matches():
    baseline = [
        {"text": "Hello, world!"},
        {"original": "FOR FASTER UPDATE"},
        {"text": "IS THIS RECORDING?"},
    ]
    macro = [
        {"text": "hello world"},
        {"text": ""},
        "IS THIS RECORDING",
    ]

    report = compare_aligned_macro_ocr_texts(baseline, macro)

    assert report["total"] == 3
    assert report["missing_count"] == 1
    assert report["exact_match_count"] == 2
    assert report["missing_text_rate"] == 0.3333
    assert report["exact_match_rate"] == 0.6667


def test_classify_ocr_text_difference_separates_line_marker_artifacts_from_material_changes():
    assert classify_ocr_text_difference("Hello, world!", "hello world") == "exact"
    assert (
        classify_ocr_text_difference(
            "THIS IS MAJOR DONG YOUNGSOO, CREW MEMBER OF THE ATALANTE.",
            "THIS IS MAJOR DONG YOUNGSOO, CREW MEMBER 67 OF THE ATALANTE.",
        )
        == "line_marker_artifact"
    )
    assert classify_ocr_text_difference("HELLO", "WORLD") == "material"


def test_compare_aligned_macro_ocr_texts_reports_material_difference_rate():
    baseline = [
        {"text": "THIS IS MAJOR DONG YOUNGSOO, CREW MEMBER OF THE ATALANTE."},
        {"text": "HELLO"},
    ]
    macro = [
        {"text": "THIS IS MAJOR DONG YOUNGSOO, CREW MEMBER 67 OF THE ATALANTE."},
        {"text": "WORLD"},
    ]

    report = compare_aligned_macro_ocr_texts(baseline, macro)

    assert report["different_count"] == 2
    assert report["line_marker_artifact_count"] == 1
    assert report["material_different_count"] == 1
    assert report["material_different_text_rate"] == 0.5


def test_estimate_macro_ocr_fallback_cost_counts_material_differences_as_block_fallbacks():
    cost = estimate_macro_ocr_fallback_cost(
        block_count=114,
        macro_window_count=81,
        material_different_count=25,
    )

    assert cost == {
        "fallback_call_count": 25,
        "effective_ocr_call_count": 106,
        "fallback_adjusted_window_reduction_rate": 0.0702,
    }


def test_collect_page_ocr_blocks_prefers_inpaint_blocks_and_falls_back_to_text_layers():
    page_with_blocks = {
        "inpaint_blocks": [{"bbox": [1, 2, 30, 40]}, {"bbox": [50, 60, 80, 90]}],
        "text_layers": [{"bbox": [9, 9, 19, 19]}],
    }
    page_with_text_layers = {
        "inpaint_blocks": [],
        "text_layers": [{"bbox": [9, 9, 19, 19], "confidence": 0.7}],
    }

    assert collect_page_ocr_blocks(page_with_blocks) == [
        {"bbox": [1, 2, 30, 40], "confidence": 1.0},
        {"bbox": [50, 60, 80, 90], "confidence": 1.0},
    ]
    assert collect_page_ocr_blocks(page_with_text_layers) == [
        {"bbox": [9, 9, 19, 19], "confidence": 0.7},
    ]
