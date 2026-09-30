import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.text_router import route_text


def test_shadow_repeated_cover_text_is_not_discarded_by_cover_heuristic():
    result = route_text(
        "Shadow Erian Shadow",
        text_id="ocr_cover_shadow",
        tipo="fala",
        bbox=[90, 60, 410, 110],
        page_number=1,
        page_height=1000,
    )

    assert result["route"] == "text"
    assert result["content_class"] == "text"
    assert result["skip_processing"] is False
    assert result["translate_policy"] == "translate"
    assert result["render_policy"] == "normal"
    assert result["input"] == "Shadow Erian Shadow"
    assert "cover_repeated_words_noise" not in result["rules_applied"]


def test_short_cover_text_is_not_discarded_without_evidence():
    result = route_text(
        "NTEEM",
        text_id="ocr_cover_ornament",
        tipo="fala",
        bbox=[120, 80, 230, 118],
        page_number=1,
        page_height=1000,
    )

    assert result["route"] == "text"
    assert result["content_class"] == "text"
    assert result["skip_processing"] is False
    assert result["input"] == "NTEEM"
    assert "cover_short_ornamental_noise" not in result["rules_applied"]


def test_cover_scanlator_roles_do_not_trigger_erase_only():
    result = route_text(
        "TL Kiki PR Mars TS Luna CL Sol",
        text_id="ocr_cover_credit",
        tipo="fala",
        bbox=[40, 100, 620, 145],
        page_number=1,
        page_height=1000,
    )

    assert result["route"] == "text"
    assert result["content_class"] == "text"
    assert result["route_action"] == "translate_inpaint_render"
    assert result["skip_processing"] is False
    assert result["translate_policy"] == "translate"
    assert result["render_policy"] == "normal"
    assert result["input"] == "TL Kiki PR Mars TS Luna CL Sol"


def test_cover_title_without_user_title_uses_neutral_route():
    result = route_text(
        "DARLING KARAOKE",
        text_id="ocr_cover_title",
        tipo="fala",
        bbox=[100, 70, 520, 130],
        page_number=1,
        page_height=1000,
    )

    assert result["route"] == "text"
    assert result["content_class"] == "text"
    assert result["route_action"] == "translate_inpaint_render"
    assert result["skip_processing"] is False
    assert result["needs_review"] is False
    assert result["render_policy"] == "normal"
    assert result["input"] == "DARLING KARAOKE"


def test_cover_region_keeps_plausible_english_dialogue_for_translation():
    result = route_text(
        "I can't believe you came here.",
        text_id="ocr_cover_dialogue",
        tipo="fala",
        bbox=[110, 70, 520, 135],
        page_number=1,
        page_height=1000,
    )

    assert result["route"] == "text"
    assert result["content_class"] == "text"
    assert result["skip_processing"] is False
    assert result["translate_policy"] == "translate"
