import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.text_router import route_text


def test_shadow_repeated_cover_text_routes_as_noise_skip_processing():
    result = route_text(
        "Shadow Erian Shadow",
        text_id="ocr_cover_shadow",
        tipo="fala",
        bbox=[90, 60, 410, 110],
        page_number=1,
        page_height=1000,
    )

    assert result["route"] == "noise"
    assert result["content_class"] == "noise"
    assert result["skip_processing"] is True
    assert result["translate_policy"] == "skip_translation"
    assert result["render_policy"] == "skip"
    assert "cover_repeated_words_noise" in result["rules_applied"]


def test_short_ornamental_cover_text_routes_as_noise_skip_processing():
    result = route_text(
        "NTEEM",
        text_id="ocr_cover_ornament",
        tipo="fala",
        bbox=[120, 80, 230, 118],
        page_number=1,
        page_height=1000,
    )

    assert result["route"] == "noise"
    assert result["content_class"] == "noise"
    assert result["skip_processing"] is True
    assert "cover_short_ornamental_noise" in result["rules_applied"]


def test_cover_scanlator_roles_route_as_scanlator_credit_not_dialogue():
    result = route_text(
        "TL Kiki PR Mars TS Luna CL Sol",
        text_id="ocr_cover_credit",
        tipo="fala",
        bbox=[40, 100, 620, 145],
        page_number=1,
        page_height=1000,
    )

    assert result["route"] == "scanlator_credit"
    assert result["content_class"] == "scanlator_credit"
    assert result["route_action"] == "inpaint_only"
    assert result["skip_processing"] is False
    assert result["translate_policy"] == "skip_translation"
    assert result["render_policy"] == "preserve"


def test_cover_title_defaults_to_cover_credit_review_not_dialogue():
    result = route_text(
        "DARLING KARAOKE",
        text_id="ocr_cover_title",
        tipo="fala",
        bbox=[100, 70, 520, 130],
        page_number=1,
        page_height=1000,
    )

    assert result["route"] == "cover_credit"
    assert result["content_class"] == "cover_credit"
    assert result["route_action"] == "review_required"
    assert result["skip_processing"] is False
    assert result["needs_review"] is True
    assert result["render_policy"] == "preserve"


def test_cover_region_keeps_plausible_english_dialogue_for_translation():
    result = route_text(
        "I can't believe you came here.",
        text_id="ocr_cover_dialogue",
        tipo="fala",
        bbox=[110, 70, 520, 135],
        page_number=1,
        page_height=1000,
    )

    assert result["route"] == "speech"
    assert result["content_class"] == "dialogue"
    assert result["skip_processing"] is False
    assert result["translate_policy"] == "translate"
