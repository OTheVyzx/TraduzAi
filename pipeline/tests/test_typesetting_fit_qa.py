from typesetter.fit_qa import assess_fit, fit_text


def test_text_fits_inside_bbox():
    result = assess_fit({"bbox": [0, 0, 200, 120], "text_bbox": [20, 20, 160, 80], "font_size": 24, "lines": 2})

    assert result["ok"] is True


def test_reduces_font_when_that_solves_fit():
    result = fit_text("Texto", {"bbox": [0, 0, 200, 120], "text_bbox": [10, 10, 170, 90], "font_size": 20, "lines": 2})

    assert result["method"] in {"fits", "reduced_font"}
    assert result["layout"]["font_size"] >= 16


def test_rewrites_short_when_needed():
    result = fit_text(
        "Texto longo demais",
        {"bbox": [0, 0, 200, 120], "text_bbox": [0, 0, 199, 119], "font_size": 16, "lines": 5},
        shortener=lambda _: "Curto",
    )

    assert result["method"] == "shortened"
    assert result["text"] == "Curto"


def test_overflow_gets_flag_when_still_bad():
    result = fit_text("Texto", {"bbox": [0, 0, 100, 60], "text_bbox": [-5, -5, 110, 70], "font_size": 12, "lines": 6})

    assert result["qa_flags"] == ["text_overflow"]


def test_does_not_leave_bbox_when_shortened():
    result = fit_text(
        "Texto longo",
        {"bbox": [0, 0, 200, 120], "text_bbox": [0, 0, 199, 119], "font_size": 16, "lines": 5},
        shortener=lambda _: "Curto",
    )

    text_bbox = result["layout"]["text_bbox"]
    assert text_bbox[0] >= 0 and text_bbox[2] <= 200
