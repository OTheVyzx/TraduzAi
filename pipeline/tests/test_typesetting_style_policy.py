import numpy as np

from typesetter.gradient_model import canonicalize_linear_gradient
from typesetter.style_policy import (
    CANONICAL_AUTO_FONT,
    normalize_auto_typesetting_style,
    sample_text_background_rgb,
    source_style_copy_allowed,
)


def test_auto_style_removes_effects_font_and_bad_white_on_light_background():
    style = normalize_auto_typesetting_style(
        {
            "fonte": "Newrotic.ttf",
            "cor": "#FFFFFF",
            "contorno": "#000000",
            "contorno_px": 3,
            "glow": True,
            "glow_cor": "#ffffff",
            "glow_px": 8,
            "sombra": True,
            "sombra_cor": "#111111",
            "sombra_offset": [3, 3],
            "tamanho": 34,
            "alinhamento": "left",
        },
        background_rgb=(245, 245, 245),
    )

    assert style["fonte"] == CANONICAL_AUTO_FONT
    assert style["cor"] == "#000000"
    assert style["contorno"] == ""
    assert style["contorno_px"] == 0
    assert style["glow"] is False
    assert style["glow_cor"] == ""
    assert style["glow_px"] == 0
    assert style["sombra"] is False
    assert style["sombra_cor"] == ""
    assert style["sombra_offset"] == [0, 0]
    assert style["tamanho"] == 34
    assert style["alinhamento"] == "left"


def test_auto_style_keeps_conservative_default_without_detected_style():
    style = normalize_auto_typesetting_style({}, (255, 255, 255))

    assert CANONICAL_AUTO_FONT == "CCTotallyAwesome W00 Bold.ttf"
    assert style["fonte"] == CANONICAL_AUTO_FONT
    assert style["cor"] == "#000000"
    assert style["contorno"] == ""
    assert style["contorno_px"] == 0
    assert style["sombra"] is False
    assert style["glow"] is False
    assert style["bold"] is True


def test_auto_style_uppercases_dialogue_from_semantic_role():
    style = normalize_auto_typesetting_style(
        {"force_upper": False},
        (255, 255, 255),
        semantic_role="dialogue_body",
    )

    assert style["force_upper"] is True


def test_auto_style_uppercases_standard_generic_manga_text():
    style = normalize_auto_typesetting_style(
        {"tipo": "text", "force_upper": False},
        (255, 255, 255),
        content_class="text",
        layout_profile="standard",
    )

    assert style["force_upper"] is True


def test_auto_style_preserves_case_for_ui_form_even_with_uppercase_input():
    style = normalize_auto_typesetting_style(
        {"force_upper": True},
        (255, 255, 255),
        semantic_role="dialogue_body",
        layout_profile="ui_form",
    )

    assert style["force_upper"] is False


def test_auto_style_preserves_confident_detected_source_paint_only():
    style = normalize_auto_typesetting_style(
        {
            "fonte": "KOMIKAX_.ttf",
            "cor": "#FFFFFF",
            "cor_gradiente": ["#0D172E", "#07080E"],
            "contorno": "#000000",
            "contorno_px": 3,
            "glow": True,
            "glow_cor": "#FFD36A",
            "glow_px": 4,
            "sombra": True,
            "sombra_cor": "#333333",
            "sombra_offset": [2, 3],
            "curva": True,
            "curva_direcao": "arc_up",
            "curva_intensidade": 0.35,
            "rotacao": -8,
            "style_origin": "source_detected",
            "style_confidence": 0.82,
        },
        (240, 240, 240),
    )

    assert style["fonte"] == CANONICAL_AUTO_FONT
    assert style["cor"] == "#FFFFFF"
    assert style["cor_gradiente"] == canonicalize_linear_gradient(
        ["#0D172E", "#07080E"]
    )
    assert style["contorno"] == "#000000"
    assert style["contorno_px"] == 3
    assert style["glow"] is True
    assert style["glow_cor"] == "#FFD36A"
    assert style["glow_px"] == 4
    assert style["sombra"] is True
    assert style["sombra_cor"] == "#333333"
    assert style["sombra_offset"] == [2, 3]
    assert style["curva"] is False
    assert style["curva_direcao"] == ""
    assert style["curva_intensidade"] == 0.0
    assert style["rotacao"] == 0


def test_auto_style_uses_normal_font_when_detected_source_has_no_gradient():
    style = normalize_auto_typesetting_style(
        {
            "fonte": "KOMIKAX_.ttf",
            "cor": "#FFFFFF",
            "contorno": "#000000",
            "contorno_px": 3,
            "style_origin": "source_detected",
            "style_confidence": 0.96,
        },
        (240, 240, 240),
    )

    assert style["fonte"] == CANONICAL_AUTO_FONT
    assert style["cor"] == "#000000"
    assert style["contorno"] == ""
    assert style["contorno_px"] == 0
    assert style["glow"] is False
    assert style["sombra"] is False


def test_auto_style_reverts_low_confidence_detected_style_to_conservative_default():
    style = normalize_auto_typesetting_style(
        {
            "fonte": "KOMIKAX_.ttf",
            "cor": "#FFFFFF",
            "contorno": "#000000",
            "contorno_px": 3,
            "style_origin": "source_detected",
            "style_confidence": 0.3,
        },
        (240, 240, 240),
    )

    assert style["fonte"] == CANONICAL_AUTO_FONT
    assert style["cor"] == "#000000"
    assert style["contorno"] == ""
    assert style["contorno_px"] == 0
    assert style["bold"] is True


def test_force_black_text_uses_normal_font_without_gradient_for_white_balloon():
    style = normalize_auto_typesetting_style(
        {
            "tipo": "dialogue",
            "layout_profile": "white_balloon",
            "fonte": "KOMIKAX_.ttf",
            "cor": "#FFFFFF",
            "style_origin": "source_detected",
            "style_confidence": 0.92,
        },
        (245, 245, 245),
        force_black_text=True,
    )

    assert style["fonte"] == CANONICAL_AUTO_FONT
    assert style["cor"] == "#000000"


def test_force_black_text_discards_detected_effects_without_gradient():
    style = normalize_auto_typesetting_style(
        {
            "tipo": "dialogue",
            "layout_profile": "white_balloon",
            "fonte": "KOMIKAX_.ttf",
            "cor": "#E7FFFF",
            "glow": True,
            "glow_cor": "#E7FFFF",
            "glow_px": 2,
            "style_origin": "source_detected",
            "style_confidence": 0.92,
        },
        (245, 245, 245),
        force_black_text=True,
    )

    assert style["fonte"] == CANONICAL_AUTO_FONT
    assert style["cor"] == "#000000"
    assert style["glow"] is False
    assert style["glow_cor"] == ""


def test_force_black_text_uses_normal_style_for_sfx_without_gradient():
    style = normalize_auto_typesetting_style(
        {
            "tipo": "sfx",
            "layout_profile": "white_balloon",
            "fonte": "KOMIKAX_.ttf",
            "cor": "#FFFFFF",
            "style_origin": "source_detected",
            "style_confidence": 0.92,
        },
        (245, 245, 245),
        force_black_text=True,
    )

    assert style["fonte"] == CANONICAL_AUTO_FONT
    assert style["cor"] == "#000000"


def test_auto_style_uses_white_only_when_dark_background_needs_it():
    style = normalize_auto_typesetting_style({"cor": "#000000"}, background_rgb=(18, 18, 24))

    assert style["fonte"] == CANONICAL_AUTO_FONT
    assert style["cor"] == "#FFFFFF"
    assert style["contorno_px"] == 0
    assert style["glow"] is False
    assert style["sombra"] is False
    assert style["bold"] is True


def test_background_sensor_prefers_inner_balloon_region():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[20:100, 20:100] = [245, 245, 245]
    image[20:100, 20:23] = [0, 0, 0]
    image[55:60, 35:85] = [0, 0, 0]

    assert sample_text_background_rgb(image, [20, 20, 100, 100]) == (245, 245, 245)


def test_background_sensor_handles_dark_panel():
    image = np.full((100, 100, 3), [20, 24, 30], dtype=np.uint8)

    rgb = sample_text_background_rgb(image, [10, 10, 90, 90])

    assert rgb[0] < 40
    assert rgb[1] < 40
    assert rgb[2] < 50


def test_source_style_copy_threshold_is_finite_and_explicit():
    assert source_style_copy_allowed("source_detected", 0.69) is False
    assert source_style_copy_allowed("source_detected", 0.70) is True
    assert source_style_copy_allowed("source_detected", 0.96) is True
    assert source_style_copy_allowed("source_detected", "invalid") is False
    assert source_style_copy_allowed("source_detected", float("inf")) is False
    assert source_style_copy_allowed("auto", 0.99) is False


def test_normalized_visual_style_cannot_carry_owner_semantics():
    reserved = {
        "owner_id": "owner_forged",
        "page_id": "page_forged",
        "source_payload": "FORGED SOURCE",
        "translated_payload": "FORGED TRANSLATION",
        "route_action": "review_required",
        "action_mask_ref": "owner_masks/forged/action_mask.png",
        "component_ids": ["component_forged"],
        "observation_ids": ["observation_forged"],
        "layout_region_ids": ["region_forged"],
    }

    normalized = normalize_auto_typesetting_style(
        {
            "style_origin": "source_detected",
            "style_confidence": 0.96,
            "fonte": "KOMIKAX_.ttf",
            "cor": "#FFFFFF",
            "cor_gradiente": ["#FFFFFF", "#78D7FF"],
            **reserved,
        },
        (18, 18, 24),
    )

    assert set(normalized).isdisjoint(reserved)
    assert normalized["fonte"] == CANONICAL_AUTO_FONT
    assert normalized["cor"] == "#FFFFFF"
    assert normalized["cor_gradiente"] == canonicalize_linear_gradient(
        ["#FFFFFF", "#78D7FF"]
    )


def test_normalized_visual_style_is_whitelisted_and_deep_copied():
    source = {
        "fonte": "KOMIKAX_.ttf",
        "cor_gradiente": ["#111111", "#222222"],
        "sombra_offset": [2, 3],
        "style_origin": "source_detected",
        "style_confidence": 0.95,
        "visible": False,
        "skip_processing": True,
        "render_policy": "suppress",
        "route_reason": "legacy_heuristic",
        "band_id": "band_forged",
        "mask_ref": "forged-mask.png",
        "translation": "FORGED",
    }

    normalized = normalize_auto_typesetting_style(source, (18, 18, 24))

    assert set(normalized).isdisjoint(
        {
            "visible",
            "skip_processing",
            "render_policy",
            "route_reason",
            "band_id",
            "mask_ref",
            "translation",
        }
    )
    normalized["cor_gradiente"]["colors"].append("#333333")
    normalized["sombra_offset"][0] = 99
    assert source["cor_gradiente"] == ["#111111", "#222222"]
    assert source["sombra_offset"] == [2, 3]


def test_normalized_directional_gradient_is_authenticated_and_deep_copied():
    gradient = {
        "kind": "linear",
        "colors": ["#6633CC", "#08080A"],
        "stops": [0.0, 1.0],
        "start": [0.1, 0.2],
        "end": [0.9, 0.8],
        "coordinate_space": "glyph_bbox_normalized",
    }
    source = {
        "style_origin": "source_detected",
        "style_confidence": 0.95,
        "cor_gradiente": gradient,
    }

    normalized = normalize_auto_typesetting_style(source, (245, 245, 245))

    assert source_style_copy_allowed(source) is True
    assert normalized["cor_gradiente"] == gradient
    normalized["cor_gradiente"]["colors"][0] = "#FFFFFF"
    assert source["cor_gradiente"]["colors"][0] == "#6633CC"
