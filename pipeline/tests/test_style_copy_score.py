from debug_tools.style_copy_score import _matches_real_kind


def test_real_style_score_uses_applied_gradient_when_report_has_applied_fields():
    detected_but_not_applied = {
        "text_color": "#102040",
        "gradient": True,
        "gradient_colors": ["#102040", "#284060"],
        "glow": False,
        "applied_text_color": "#000000",
        "applied_gradient": False,
        "applied_gradient_colors": [],
        "applied_glow": False,
        "bbox": [0, 0, 160, 60],
    }
    applied = {
        **detected_but_not_applied,
        "applied_gradient": True,
        "applied_gradient_colors": ["#102040", "#284060"],
    }

    assert _matches_real_kind(detected_but_not_applied, "dark_text_gradient") is False
    assert _matches_real_kind(applied, "dark_text_gradient") is True


def test_real_style_score_falls_back_to_detected_fields_for_old_reports():
    old_report_record = {
        "text_color": "#102040",
        "gradient": True,
        "gradient_colors": ["#102040", "#284060"],
        "glow": False,
        "bbox": [0, 0, 160, 60],
    }

    assert _matches_real_kind(old_report_record, "dark_text_gradient") is True


def test_real_style_score_accepts_solid_dark_text_without_false_gradient():
    record = {
        "applied_text_color": "#020202",
        "applied_gradient": False,
        "applied_gradient_colors": [],
        "applied_glow": False,
        "applied_stroke_color": "",
        "bbox": [0, 0, 260, 90],
    }

    assert _matches_real_kind(record, "solid_dark_text") is True
