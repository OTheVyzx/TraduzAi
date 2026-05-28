import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.text_router import route_text


def test_speech_with_inline_sfx_marker_is_split_into_speech_and_sfx_parts():
    result = route_text("DON'T HIT SFX: KICK MY MOM!", text_id="ocr_031")

    assert result["route"] == "speech_with_sfx_split"
    assert result["rules_applied"] == ["split_sfx_marker_inside_dialogue"]
    assert result["parts"] == [
        {"class": "speech", "text": "DON'T HIT MY MOM!", "text_id_synthetic": "ocr_031_a"},
        {"class": "sfx", "text": "KICK", "text_id_synthetic": "ocr_031_b"},
    ]
    assert result["needs_review"] is False


def test_translator_note_skips_translation_and_preserves_rendering():
    result = route_text("T/N: HYUNGNIM SIGNIFICA IRMAO MAIS VELHO", text_id="ocr_022")

    assert result["route"] == "editorial_note"
    assert result["translate_policy"] == "skip_translation"
    assert result["render_policy"] == "preserve"
    assert result["needs_review"] is True


def test_url_or_handle_routes_as_url_watermark():
    for text in ["https://lagoonscans.com", "Read at lagoonscans.com", "@lagoonscans"]:
        result = route_text(text, text_id="ocr_url")

        assert result["route"] == "url_watermark"
        assert result["translate_policy"] == "skip_translation"
        assert "url_or_handle_watermark" in result["rules_applied"]
