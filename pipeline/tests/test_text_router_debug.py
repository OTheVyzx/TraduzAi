import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.text_router import (
    apply_route_action,
    route_action_requires_inpaint,
    route_action_requires_render,
    route_action_requires_translation,
    route_text,
)


def test_hangul_sfx_routes_to_sfx_engine():
    result = route_text("쿵", tipo="sfx")

    assert result["route"] == "manhwa_sfx"
    assert result["reason"] == "hangul_sfx_candidate"
    assert result["content_class"] == "sfx"
    assert result["tipo"] == "sfx"
    assert result["script"] == "hangul"
    assert result["route_action"] == "translate_sfx_inpaint_render"
    assert result["route_reason"] == "hangul_sfx_candidate"
    assert result["translate_policy"] == "adapt_sfx"
    assert result["render_policy"] == "sfx_style"


def test_sfx_route_action_requires_translation_render_and_inpaint():
    route_action = "translate_sfx_inpaint_render"

    assert route_action_requires_translation(route_action) is True
    assert route_action_requires_render(route_action) is True
    assert route_action_requires_inpaint(route_action) is True


def test_speech_with_inline_sfx_marker_is_split_into_speech_and_sfx_parts():
    result = route_text("DON'T HIT SFX: KICK MY MOM!", text_id="ocr_031")

    assert result["route"] == "speech_with_sfx_split"
    assert result["rules_applied"] == ["split_sfx_marker_inside_dialogue"]
    assert result["parts"] == [
        {"class": "speech", "text": "DON'T HIT MY MOM!", "text_id_synthetic": "ocr_031_a"},
        {"class": "sfx", "text": "KICK", "text_id_synthetic": "ocr_031_b"},
    ]
    assert result["needs_review"] is False


def test_speech_with_compact_inline_sfx_marker_is_split_into_speech_and_sfx_parts():
    result = route_text("DON'T HIT SFXKICK MY MOM!", text_id="ocr_031")

    assert result["route"] == "speech_with_sfx_split"
    assert result["parts"] == [
        {"class": "speech", "text": "DON'T HIT MY MOM!", "text_id_synthetic": "ocr_031_a"},
        {"class": "sfx", "text": "KICK", "text_id_synthetic": "ocr_031_b"},
    ]


def test_compact_sfx_without_dialogue_does_not_create_empty_speech_split():
    result = route_text("SFXKICK", text_id="ocr_sfx")

    assert result["route"] == "text"
    assert "parts" not in result


def test_translator_note_skips_translation_and_preserves_rendering():
    result = route_text("T/N: HYUNGNIM SIGNIFICA IRMAO MAIS VELHO", text_id="ocr_022")

    assert result["route"] == "editorial_note"
    assert result["translate_policy"] == "skip_translation"
    assert result["render_policy"] == "preserve"
    assert result["route_action"] == "review_required"
    assert result["needs_review"] is True


def test_url_or_handle_routes_as_url_watermark():
    for text in ["https://lagoonscans.com", "Read at lagoonscans.com", "@lagoonscans"]:
        result = route_text(text, text_id="ocr_url")

        assert result["route"] == "url_watermark"
        assert result["translate_policy"] == "skip_translation"
        assert result["route_action"] == "review_required"
        assert "url_or_handle_watermark" in result["rules_applied"]


def test_preserve_policy_overrides_stale_translate_route_action():
    result = apply_route_action(
        {
            "translate_policy": "skip_translation",
            "render_policy": "preserve",
            "route_action": "translate_inpaint_render",
            "reason": "url_or_handle_watermark",
        }
    )

    assert result["route_action"] == "review_required"


def test_short_hangul_dialogue_does_not_route_as_sfx_without_sfx_tipo():
    result = route_text("안녕하세요")

    assert result["route"] == "text"
    assert result["content_class"] == "text"
    assert result["route_action"] == "translate_inpaint_render"
