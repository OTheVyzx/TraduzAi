import numpy as np

from typesetter.style_policy import (
    CANONICAL_AUTO_FONT,
    normalize_auto_typesetting_style,
    sample_text_background_rgb,
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

    assert style["fonte"] == "ComicNeue-Bold.ttf"
    assert style["cor"] == "#000000"
    assert style["contorno"] == ""
    assert style["contorno_px"] == 0
    assert style["sombra"] is False
    assert style["glow"] is False


def test_auto_style_preserves_confident_detected_source_style():
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

    assert style["fonte"] == "KOMIKAX_.ttf"
    assert style["cor"] == "#FFFFFF"
    assert style["cor_gradiente"] == ["#0D172E", "#07080E"]
    assert style["contorno"] == "#000000"
    assert style["contorno_px"] == 3
    assert style["glow"] is True
    assert style["glow_cor"] == "#FFD36A"
    assert style["glow_px"] == 4
    assert style["sombra"] is True
    assert style["sombra_cor"] == "#333333"
    assert style["sombra_offset"] == [2, 3]
    assert style["curva"] is True
    assert style["curva_direcao"] == "arc_up"
    assert style["curva_intensidade"] == 0.35
    assert style["rotacao"] == -8


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

    assert style["fonte"] == "ComicNeue-Bold.ttf"
    assert style["cor"] == "#000000"
    assert style["contorno"] == ""
    assert style["contorno_px"] == 0


def test_force_black_text_overrides_confident_source_style_for_white_balloon():
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

    assert style["fonte"] == "KOMIKAX_.ttf"
    assert style["cor"] == "#000000"


def test_force_black_text_preserves_confident_source_color_when_effects_are_detected():
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

    assert style["fonte"] == "KOMIKAX_.ttf"
    assert style["cor"] == "#E7FFFF"
    assert style["glow"] is True
    assert style["glow_cor"] == "#E7FFFF"


def test_force_black_text_allows_confident_source_style_for_sfx():
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

    assert style["fonte"] == "KOMIKAX_.ttf"
    assert style["cor"] == "#FFFFFF"


def test_auto_style_uses_white_only_when_dark_background_needs_it():
    style = normalize_auto_typesetting_style({"cor": "#000000"}, background_rgb=(18, 18, 24))

    assert style["fonte"] == "ComicNeue-Bold.ttf"
    assert style["cor"] == "#FFFFFF"
    assert style["contorno_px"] == 0
    assert style["glow"] is False
    assert style["sombra"] is False


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
