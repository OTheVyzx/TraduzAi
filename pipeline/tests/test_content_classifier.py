from ocr.postprocess import ContentClass, classify_content, split_sfx_inline
from ocr.text_router import ROUTE_ACTIONS, route_text


def test_classify_inline_sfx_dialogue_and_split():
    text = "DON'T HIT SFX: KICK MY MOM!"

    assert classify_content(text, "fala") == ContentClass.DIALOGUE
    assert split_sfx_inline(text) == ("DON'T HIT MY MOM!", "KICK")


def test_classify_tn_note():
    assert classify_content("T/N: thanks for reading", "fala") == ContentClass.TN_NOTE


def test_classify_url_watermark():
    assert classify_content("Read at discord.gg/example", "fala") == ContentClass.URL_WATERMARK


def test_classify_scanlator_credit():
    assert classify_content("Secret Scans translation team", "fala") == ContentClass.SCANLATOR_CREDIT


def test_classify_sign_keeps_separate_from_dialogue():
    assert classify_content("TEXT: DARLING KARAOKE", "narracao") == ContentClass.SIGN


def test_route_action_contract_values_and_skip_invariant():
    assert ROUTE_ACTIONS == {
        "translate_inpaint_render",
        "translate_render_only",
        "inpaint_only",
        "preserve",
        "review_required",
        "skip",
    }

    cases = [
        route_text("I can't believe you came here.", tipo="fala"),
        route_text("Read at lagoonscans.com", tipo="fala"),
        route_text("", tipo="fala"),
    ]

    for routed in cases:
        assert routed["route_action"] in ROUTE_ACTIONS
        assert routed["route_reason"]
        assert routed["skip_processing"] is (routed["route_action"] == "skip")


def test_watermark_routes_to_inpaint_only_not_skip():
    routed = route_text("Read at lagoonscans.com", tipo="fala")

    assert routed["route_action"] == "inpaint_only"
    assert routed["route_reason"] == "watermark_detected"
    assert routed["skip_processing"] is False


def test_normal_dialogue_uses_schema_neutral_text_type():
    routed = route_text("I can't believe you came here.", tipo="fala")

    assert routed["route_action"] == "translate_inpaint_render"
    assert routed["content_class"] == "dialogue"
    assert routed["tipo"] == "texto"
