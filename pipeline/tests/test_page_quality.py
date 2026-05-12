from qa.page_quality import evaluate_page_quality


def test_page_quality_routes_cjk_partial_ocr_to_bbox_reocr_before_page_detect():
    page = {
        "texts": [
            {
                "text": "\ud558",
                "bbox": [20, 20, 320, 50],
                "confidence": 0.4,
                "line_polygons": [[[20, 20], [320, 20], [320, 32], [20, 32]], [[20, 36], [300, 36], [300, 50], [20, 50]]],
            }
        ],
        "_vision_blocks": [{"bbox": [18, 18, 330, 58]}],
    }

    quality = evaluate_page_quality(page, source_lang="ko")

    assert quality["should_try_bbox_expanded_reocr"] is True
    assert quality["should_try_page_detect"] is False
    assert {issue["type"] for issue in quality["issues"]} == {"partial_multiline_ocr"}

    after_reocr = evaluate_page_quality(page, source_lang="ko", expanded_reocr_attempted=True)
    assert after_reocr["should_try_page_detect"] is True


def test_page_quality_does_not_trigger_page_detect_for_translation_or_typesetting_flags():
    page = {
        "texts": [
            {
                "text": "\ud558\ud558\ud558",
                "bbox": [20, 20, 120, 60],
                "confidence": 0.9,
                "qa_flags": ["translation_fallback_phrase", "text_overflow"],
            }
        ],
        "_vision_blocks": [{"bbox": [18, 18, 130, 70]}],
    }

    quality = evaluate_page_quality(page, source_lang="ko", expanded_reocr_attempted=True)

    assert quality["should_try_bbox_expanded_reocr"] is False
    assert quality["should_try_page_detect"] is False
    assert {issue["type"] for issue in quality["non_rerun_issues"]} == {"non_rerun_quality_flag"}


def test_page_quality_flags_known_balloon_without_text_and_low_prior_coverage():
    page = {
        "texts": [],
        "_vision_blocks": [
            {
                "bbox": [30, 40, 180, 120],
                "balloon_polygon": [[30, 40], [180, 40], [180, 120], [30, 120]],
            }
        ],
    }

    quality = evaluate_page_quality(
        page,
        source_lang="ko",
        chapter_prior={"expected_text_count": 4},
    )

    assert quality["should_try_bbox_expanded_reocr"] is True
    issue_types = {issue["type"] for issue in quality["issues"]}
    assert "known_speech_balloon_without_ocr" in issue_types
    assert "low_ocr_coverage_vs_chapter_prior" in issue_types
