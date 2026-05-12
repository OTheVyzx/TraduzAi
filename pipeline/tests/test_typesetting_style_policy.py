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
