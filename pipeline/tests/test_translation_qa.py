import pytest

from qa.translation_qa import can_export, render_decision, severity_for_flag, summarize_flags


@pytest.mark.parametrize(
    ("flag", "severity"),
    [
        ("glossary_violation", "critical"),
        ("forbidden_translation", "critical"),
        ("untranslated_english", "high"),
        ("suspected_ocr_error", "medium"),
        ("duplicate_log_event", "low"),
    ],
)
def test_flag_severity(flag, severity):
    assert severity_for_flag(flag) == severity


def test_summary_dedupes_flags_and_counts_consistently():
    summary = summarize_flags([
        {"qa_flags": ["forbidden_translation", "forbidden_translation"]},
        {"qa_flags": ["low_ocr_confidence"]},
    ])

    assert summary["flags"] == ["forbidden_translation", "low_ocr_confidence"]
    assert summary["counts"]["forbidden_translation"] == 2
    assert summary["critical_count"] == 2
    assert summary["highest_severity"] == "critical"


def test_render_blocks_critical_region():
    decision = render_decision({"qa_flags": ["placeholder_lost"]})

    assert decision["status"] == "blocked"
    assert decision["action"] == "debug_overlay_only"


def test_render_warning_still_renders_with_flag():
    decision = render_decision({"qa_flags": ["text_overflow"]})

    assert decision["status"] == "warning"
    assert decision["action"] == "render_with_flag"


def test_export_policy_modes():
    summary = {"critical_count": 1}

    assert can_export(summary, "strict")["allowed"] is False
    assert can_export(summary, "review")["allowed"] is True
    assert can_export(summary, "unsafe/manual")["allowed"] is True
