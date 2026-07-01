import cv2
import numpy as np

from typesetter.style_extractor import (
    DEFAULT_BALLOON_FONT,
    TextStyleEvidence,
    extract_sfx_style_evidence,
    extract_text_style_evidence,
)


class FakeFontDetector:
    def detect(self, crop, allow_default=True):
        return "KOMIKAX_.ttf"


class FakeScoreFontDetector:
    def __init__(self, font_name="CCDaveGibbonsLower W00 Regular.ttf", confidence=0.92):
        self.font_name = font_name
        self.confidence = confidence

    def detect_with_score(self, crop, allow_default=True):
        return self.font_name, self.confidence


class RaisingFontDetector:
    def detect(self, crop, allow_default=True):
        raise RuntimeError("font model unavailable")


def test_extracts_black_fill_from_dark_text_on_white_crop():
    crop = np.full((80, 160, 3), 255, dtype=np.uint8)
    crop[25:55, 40:120] = 0

    evidence = extract_text_style_evidence(crop)

    assert isinstance(evidence, TextStyleEvidence)
    assert evidence.text_color == "#000000"
    assert evidence.text_color_confidence >= 0.75
    assert evidence.source == "pixel_analysis"


def test_returns_low_confidence_for_empty_or_flat_crop():
    crop = np.full((80, 160, 3), 255, dtype=np.uint8)

    evidence = extract_text_style_evidence(crop)

    assert evidence.text_color_confidence < 0.5
    assert evidence.stroke_width_px == 0


def test_detects_light_text_with_dark_outline():
    crop = np.full((100, 180, 3), [240, 220, 210], dtype=np.uint8)
    crop[30:70, 45:135] = 0
    crop[36:64, 55:125] = 255

    evidence = extract_text_style_evidence(crop)

    assert evidence.text_color == "#FFFFFF"
    assert evidence.stroke_color == "#000000"
    assert evidence.stroke_width_px >= 2
    assert evidence.stroke_confidence >= 0.5


def test_detects_dark_text_with_light_outline_on_colored_background():
    crop = np.full((110, 210, 3), [255, 220, 210], dtype=np.uint8)
    crop[30:80, 45:165] = 255
    crop[40:70, 60:150] = 0

    evidence = extract_text_style_evidence(crop)

    assert evidence.text_color == "#000000"
    assert evidence.stroke_color == "#FFFFFF"
    assert evidence.stroke_width_px >= 2
    assert evidence.stroke_confidence >= 0.5
    assert evidence.glow is False


def test_solid_light_outline_is_not_classified_as_glow():
    crop = np.full((120, 220, 3), [145, 0, 245], dtype=np.uint8)
    crop[36:84, 42:178] = 255
    crop[45:75, 58:162] = 0

    evidence = extract_text_style_evidence(crop)

    assert evidence.text_color == "#000000"
    assert evidence.stroke_color == "#FFFFFF"
    assert evidence.stroke_width_px >= 2
    assert evidence.stroke_confidence >= 0.5
    assert evidence.glow is False


def test_detects_dark_outline_around_light_text_on_art_background():
    crop = np.full((120, 260, 3), [236, 230, 170], dtype=np.uint8)
    crop[28:88, 34:226] = 12
    crop[36:78, 48:212] = 252

    evidence = extract_text_style_evidence(crop)

    assert evidence.text_color == "#FCFCFC"
    assert evidence.stroke_color == "#0C0C0C"
    assert evidence.stroke_width_px >= 2
    assert evidence.glow is False


def test_detects_offset_gray_shadow_behind_black_text():
    crop = np.full((100, 180, 3), 255, dtype=np.uint8)
    crop[36:66, 60:130] = 135
    crop[30:60, 52:122] = 0

    evidence = extract_text_style_evidence(crop)

    assert evidence.shadow is True
    assert evidence.shadow_confidence >= 0.6
    assert evidence.to_dict()["shadow_color"] == "#878787"
    assert evidence.to_dict()["shadow_offset"] == [8, 6]
    assert evidence.glow is False


def test_detects_bright_blurred_glow_on_dark_background():
    crop = np.full((120, 200, 3), 18, dtype=np.uint8)
    halo = np.zeros((120, 200), dtype=np.uint8)
    halo[46:76, 72:132] = 255
    halo = cv2.GaussianBlur(halo, (0, 0), sigmaX=5.0)
    crop = np.maximum(crop, np.dstack([halo, halo, halo]).astype(np.uint8))
    crop[48:74, 76:128] = [255, 238, 120]

    evidence = extract_text_style_evidence(crop)

    assert evidence.glow is True
    assert evidence.glow_confidence >= 0.6
    assert evidence.to_dict()["glow_color"] == "#FFEE78"
    assert evidence.to_dict()["glow_px"] >= 2
    assert evidence.shadow is False


def test_detects_light_glow_on_saturated_card_background():
    crop = np.full((130, 260, 3), [85, 195, 242], dtype=np.uint8)
    halo = np.zeros((130, 260), dtype=np.uint8)
    halo[30:100, 58:202] = 255
    halo = cv2.GaussianBlur(halo, (0, 0), sigmaX=3.0)
    halo_rgb = np.dstack([halo, halo, halo]).astype(np.uint8)
    crop = np.maximum(crop, halo_rgb)
    crop[42:88, 74:186] = 255

    evidence = extract_text_style_evidence(crop)

    assert evidence.text_color == "#FFFFFF"
    assert evidence.stroke_color == ""
    assert evidence.glow is True
    assert evidence.glow_confidence >= 0.88


def test_detects_vertical_gradient_fill_on_text_mask():
    crop = np.full((120, 220, 3), 18, dtype=np.uint8)
    for y in range(32, 88):
        t = (y - 32) / 55.0
        color = np.array([70 + 120 * t, 220 - 120 * t, 255 - 40 * t], dtype=np.uint8)
        crop[y, 60:160] = color

    evidence = extract_text_style_evidence(crop)

    assert evidence.gradient is True
    assert evidence.gradient_confidence >= 0.62
    assert evidence.gradient_colors is not None
    assert len(evidence.gradient_colors) == 2
    assert evidence.gradient_colors[0] != evidence.gradient_colors[1]


def test_detects_subtle_dark_blue_gradient_fill():
    crop = np.full((130, 260, 3), 255, dtype=np.uint8)
    for y in range(35, 95):
        t = (y - 35) / 59.0
        color = np.array([8, 14 + 18 * t, 32 + 52 * t], dtype=np.uint8)
        crop[y, 58:202] = color

    evidence = extract_text_style_evidence(crop)

    assert evidence.gradient is True
    assert evidence.gradient_confidence >= 0.62
    assert evidence.gradient_colors is not None


def test_solid_text_does_not_enable_gradient():
    crop = np.full((100, 180, 3), 255, dtype=np.uint8)
    crop[30:70, 45:135] = [20, 20, 20]

    evidence = extract_text_style_evidence(crop)

    assert evidence.gradient is False
    assert evidence.gradient_colors is None
    assert evidence.gradient_confidence == 0.0


def test_curve_detection_is_disabled_for_now():
    crop = np.full((160, 360, 3), [138, 0, 247], dtype=np.uint8)
    for x in range(50, 310, 8):
        t = (x - 180) / 130.0
        y = int(96 - 28 * (1.0 - t * t))
        crop[y : y + 18, x : x + 6] = 0

    evidence = extract_text_style_evidence(crop)

    assert evidence.curved is False
    assert evidence.curve_direction == ""
    assert evidence.curve_confidence == 0.0
    assert evidence.curve_amount == 0.0


def test_straight_text_does_not_enable_curvature():
    crop = np.full((120, 320, 3), 255, dtype=np.uint8)
    crop[52:72, 40:280] = 0

    evidence = extract_text_style_evidence(crop)

    assert evidence.curved is False
    assert evidence.curve_confidence == 0.0


def test_solid_light_text_on_colored_background_does_not_use_background_gradient():
    crop = np.full((100, 220, 3), [80, 190, 235], dtype=np.uint8)
    crop[30:70, 50:170] = 255

    evidence = extract_text_style_evidence(crop)

    assert evidence.text_color == "#FFFFFF"
    assert evidence.gradient is False
    assert evidence.gradient_colors is None


def test_detects_solid_white_text_on_saturated_background_as_white_fill():
    crop = np.full((120, 220, 3), [138, 0, 247], dtype=np.uint8)
    crop[34:86, 35:185] = 255

    evidence = extract_text_style_evidence(crop)

    assert evidence.text_color == "#FFFFFF"
    assert evidence.text_color_confidence >= 0.55
    assert evidence.gradient is False


def test_flat_crop_does_not_enable_shadow_or_glow():
    crop = np.full((80, 160, 3), 180, dtype=np.uint8)

    evidence = extract_text_style_evidence(crop)

    assert evidence.shadow is False
    assert evidence.shadow_confidence == 0.0
    assert evidence.glow is False
    assert evidence.glow_confidence == 0.0


def test_text_style_evidence_serializes_to_dict():
    evidence = TextStyleEvidence(
        source="pixel_analysis",
        text_color="#000000",
        text_color_confidence=0.9,
        stroke_color="",
        stroke_width_px=0,
        stroke_confidence=0.0,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
        gradient=False,
        gradient_colors=None,
        gradient_confidence=0.0,
        curved=False,
        curve_direction="",
        curve_amount=0.0,
        curve_confidence=0.0,
    )

    data = evidence.to_dict()

    assert data["source"] == "pixel_analysis"
    assert data["text_color"] == "#000000"
    assert data["stroke_width_px"] == 0


def test_uses_default_font_for_non_heavy_balloon_text():
    crop = np.full((80, 160, 3), 255, dtype=np.uint8)
    crop[25:55, 40:120] = 0

    evidence = extract_text_style_evidence(crop, font_detector=FakeFontDetector())

    assert evidence.font_name == DEFAULT_BALLOON_FONT
    assert evidence.font_confidence == 1.0


def test_visual_card_text_can_use_detector_font_even_when_not_heavy():
    crop = np.full((110, 210, 3), [92, 154, 224], dtype=np.uint8)
    crop[30:84, 54:156] = [235, 255, 255]

    evidence = extract_text_style_evidence(
        crop,
        font_detector=FakeScoreFontDetector("LeagueGothic-Regular-VariableFont_wdth.ttf", 0.88),
        font_context="visual_card",
    )

    assert evidence.font_name == "LeagueGothic-Regular-VariableFont_wdth.ttf"
    assert evidence.font_confidence == 0.88


def test_visual_card_context_maps_non_panel_detector_font_to_league_gothic():
    crop = np.full((110, 210, 3), [92, 154, 224], dtype=np.uint8)
    crop[30:84, 54:156] = [235, 255, 255]

    evidence = extract_text_style_evidence(
        crop,
        font_detector=FakeScoreFontDetector("CCDaveGibbonsLower W00 Regular.ttf", 0.93),
        font_context="visual_card",
    )

    assert evidence.font_name == "LeagueGothic-Regular-VariableFont_wdth.ttf"
    assert evidence.font_confidence == 0.93


def test_maps_heavy_text_to_impact_font_family():
    crop = np.full((80, 160, 3), 255, dtype=np.uint8)
    crop[10:70, 20:140] = 0

    evidence = extract_text_style_evidence(
        crop,
        font_detector=FakeScoreFontDetector(),
    )

    assert evidence.font_name == "KOMIKAX_.ttf"
    assert evidence.font_confidence >= 0.9


def test_keeps_luckiest_guy_for_heavy_text_when_detector_prefers_it():
    crop = np.full((80, 160, 3), 255, dtype=np.uint8)
    crop[10:70, 20:140] = 0

    evidence = extract_text_style_evidence(
        crop,
        font_detector=FakeScoreFontDetector("LuckiestGuy-Regular.ttf", 0.94),
    )

    assert evidence.font_name == "LuckiestGuy-Regular.ttf"
    assert evidence.font_confidence >= 0.9


def test_keeps_style_evidence_when_optional_font_detector_raises():
    crop = np.full((80, 160, 3), 255, dtype=np.uint8)
    crop[25:55, 40:120] = 0

    evidence = extract_text_style_evidence(crop, font_detector=RaisingFontDetector())

    assert evidence.text_color == "#000000"
    assert evidence.text_color_confidence >= 0.75
    assert evidence.font_name == DEFAULT_BALLOON_FONT
    assert evidence.font_confidence == 1.0


def test_extract_sfx_style_evidence_bridge_reuses_sfx_extractor():
    crop = np.full((80, 160, 3), 240, dtype=np.uint8)
    mask = np.zeros((80, 160), dtype=np.uint8)
    mask[20:60, 40:120] = 255
    crop[mask > 0] = [224, 36, 48]

    evidence = extract_sfx_style_evidence(crop, mask)

    assert evidence.fill_color == "#E02430"
    assert evidence.scale_x > 0
    assert evidence.confidence >= 0.4
