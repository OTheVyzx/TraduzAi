import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_stack.sfx_detector import (
    _contains_hangul,
    _looks_like_large_panel_or_scene_artifact,
    _looks_like_character_artifact,
    _looks_like_dark_narration_caption_artifact,
    _looks_like_latin_balloon_text,
    _looks_like_latin_ocr_text_artifact,
    _looks_like_long_page_footer_credit_artifact,
    _looks_like_long_page_scanlator_or_ui_artifact,
    _looks_like_low_detail_logo_artifact,
    _looks_like_low_detail_blue_scene_artifact,
    _looks_like_low_conf_horizontal_texture_artifact,
    _looks_like_long_page_visual_rescue_artifact,
    _looks_like_low_conf_warm_artifact,
    _looks_like_overbroad_low_conf_long_page_artifact,
    _looks_like_pale_vertical_artifact,
    _looks_like_plain_dialogue_or_caption_crop,
    _looks_like_scanlator_logo_artifact,
    _looks_like_short_hangul_caption_artifact,
    _looks_like_single_glyph_white_bubble_artifact,
    _looks_like_spurious_recognized_caption_artifact,
    _looks_like_top_credit_artifact,
    _looks_like_top_chapter_ornament_artifact,
    _looks_like_unconfirmed_visual_sfx_artifact,
    _looks_like_unconfirmed_light_color_sfx,
    _looks_like_white_dialogue_or_caption_artifact,
    _merge_nearby_short_page_visual_candidates,
    _should_drop_non_cjk_sfx_artifact,
    _suppressed_by_existing_text,
    _suppress_color_fragments_near_white_candidates,
    detect_sfx_candidates,
    filter_sfx_candidates_after_ocr,
    merge_sfx_candidates,
    text_blocks_to_sfx_candidates,
)
from tools.run_sfx_detection_probe import _make_final_crops_sheet, _validate_expectations
from tools.validate_sfx_expectations import main as validate_sfx_expectations_main


def _speed_line_red_sfx_page() -> np.ndarray:
    image = np.full((240, 320, 3), 236, dtype=np.uint8)
    for x in range(0, 320, 7):
        cv2.line(image, (x, 20), (max(0, x - 120), 210), (38, 42, 52), 1)
    cv2.rectangle(image, (62, 48), (82, 172), (116, 32, 38), -1)
    cv2.rectangle(image, (62, 142), (164, 166), (116, 32, 38), -1)
    cv2.rectangle(image, (202, 46), (226, 178), (116, 32, 38), -1)
    cv2.rectangle(image, (202, 46), (282, 70), (116, 32, 38), -1)
    cv2.rectangle(image, (246, 104), (282, 178), (116, 32, 38), -1)
    return image


def _white_blue_sfx_page() -> np.ndarray:
    image = np.full((260, 340, 3), [220, 231, 238], dtype=np.uint8)
    cv2.line(image, (30, 200), (310, 60), (32, 46, 95), 2)
    cv2.rectangle(image, (78, 48), (100, 178), (52, 70, 98), -1)
    cv2.rectangle(image, (78, 148), (170, 174), (52, 70, 98), -1)
    cv2.rectangle(image, (208, 50), (232, 184), (52, 70, 98), -1)
    cv2.rectangle(image, (208, 50), (288, 74), (52, 70, 98), -1)
    cv2.rectangle(image, (246, 108), (288, 184), (52, 70, 98), -1)
    cv2.rectangle(image, (84, 54), (94, 170), (250, 253, 255), -1)
    cv2.rectangle(image, (84, 154), (164, 168), (250, 253, 255), -1)
    cv2.rectangle(image, (214, 56), (226, 178), (250, 253, 255), -1)
    cv2.rectangle(image, (214, 56), (282, 68), (250, 253, 255), -1)
    return image


def _dialogue_balloon_page() -> tuple[np.ndarray, list[dict]]:
    image = np.full((220, 320, 3), [224, 230, 236], dtype=np.uint8)
    cv2.ellipse(image, (160, 90), (92, 52), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(image, (160, 90), (92, 52), 0, 0, 360, (20, 20, 20), 2)
    cv2.rectangle(image, (112, 76), (226, 88), (15, 15, 15), -1)
    cv2.rectangle(image, (130, 100), (206, 112), (15, 15, 15), -1)
    blocks = [{"bbox": [68, 38, 252, 142], "detector": "balloon"}]
    return image, blocks


def test_contains_hangul_detects_syllables_and_jamo():
    assert _contains_hangul("마")
    assert _contains_hangul("ᄆ")
    assert not _contains_hangul("MA")


def test_detects_colored_sfx_over_speed_lines():
    candidates = detect_sfx_candidates(_speed_line_red_sfx_page())

    assert candidates
    assert candidates[0]["content_class"] == "sfx"
    assert candidates[0]["detector"] == "sfx_visual"
    assert candidates[0]["confidence"] >= 0.45
    assert "sfx_visual_candidate" in candidates[0]["qa_flags"]


def test_visual_detector_uses_restricted_color_rescue_on_long_webtoon_pages_by_default():
    image = np.full((1200, 320, 3), 236, dtype=np.uint8)
    cv2.putText(image, "S", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (116, 32, 38), 10, cv2.LINE_AA)
    cv2.putText(image, "FX", (115, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (116, 32, 38), 10, cv2.LINE_AA)

    candidates = detect_sfx_candidates(image)

    assert candidates
    assert {(candidate.get("sfx") or {}).get("visual_source") for candidate in candidates} <= {
        "red_chroma",
        "color_chroma",
    }


def test_long_page_visual_filter_keeps_unconfirmed_color_neighbor_of_cjk_visual_candidate():
    image = np.full((1200, 320, 3), 236, dtype=np.uint8)
    cv2.rectangle(image, (86, 728), (134, 780), (82, 118, 95), -1)
    cv2.rectangle(image, (156, 773), (214, 840), (124, 28, 38), -1)
    recognized = {
        "bbox": [80, 720, 145, 790],
        "text_pixel_bbox": [80, 720, 145, 790],
        "detector": "sfx_visual",
        "confidence": 0.92,
        "qa_flags": ["sfx_visual_candidate"],
        "sfx": {"visual_source": "color_chroma"},
        "sfx_ocr": {"status": "recognized", "attempts": [{"text": "カン", "confidence": 0.91}]},
        "recognized_text": "カン",
    }
    neighbor = {
        "bbox": [150, 765, 225, 850],
        "text_pixel_bbox": [150, 765, 225, 850],
        "detector": "sfx_visual",
        "confidence": 0.88,
        "qa_flags": ["sfx_visual_candidate"],
        "sfx": {"visual_source": "red_chroma"},
        "sfx_ocr": {"status": "no_confident_cjk", "attempts": []},
    }
    far_artifact = {
        "bbox": [30, 980, 120, 1070],
        "text_pixel_bbox": [30, 980, 120, 1070],
        "detector": "sfx_visual",
        "confidence": 0.89,
        "qa_flags": ["sfx_visual_candidate"],
        "sfx": {"visual_source": "red_chroma"},
        "sfx_ocr": {"status": "no_confident_cjk", "attempts": []},
    }

    kept = filter_sfx_candidates_after_ocr([recognized, neighbor, far_artifact], image)

    assert [item["bbox"] for item in kept] == [[80, 720, 225, 850]]
    assert "sfx_long_page_visual_cluster_merged" in kept[0]["qa_flags"]
    assert "sfx_long_page_visual_neighbor_rescue" in neighbor["qa_flags"]
    assert "sfx_artifact_long_page_visual_rescue_rejected" in far_artifact["qa_flags"]


def test_long_page_visual_filter_does_not_use_artifact_cjk_anchor_for_rescue():
    image = np.full((1200, 320, 3), [128, 151, 128], dtype=np.uint8)
    recognized_artifact = {
        "bbox": [120, 720, 190, 840],
        "text_pixel_bbox": [120, 720, 190, 840],
        "detector": "sfx_visual",
        "confidence": 0.90,
        "qa_flags": ["sfx_visual_candidate"],
        "sfx": {"visual_source": "color_chroma"},
        "sfx_ocr": {"status": "recognized", "attempts": [{"text": "川", "confidence": 0.91}]},
        "recognized_text": "川",
    }
    neighbor = {
        "bbox": [196, 780, 250, 850],
        "text_pixel_bbox": [196, 780, 250, 850],
        "detector": "sfx_visual",
        "confidence": 0.86,
        "qa_flags": ["sfx_visual_candidate"],
        "sfx": {"visual_source": "color_chroma"},
        "sfx_ocr": {"status": "no_confident_cjk", "attempts": []},
    }

    kept = filter_sfx_candidates_after_ocr([recognized_artifact, neighbor], image)

    assert kept == []
    assert "sfx_artifact_long_page_visual_rescue_rejected" in recognized_artifact["qa_flags"]
    assert "sfx_artifact_long_page_visual_rescue_rejected" in neighbor["qa_flags"]


def test_long_page_visual_rescue_rejects_bright_balloon_border_artifact():
    image = np.full((240, 360, 3), 248, dtype=np.uint8)
    cv2.ellipse(image, (180, 80), (150, 80), 0, 0, 360, (135, 70, 48), 4)
    candidate = {"sfx": {"visual_source": "red_chroma"}}

    assert _looks_like_long_page_visual_rescue_artifact(candidate, image, [20, 20, 340, 150])


def test_long_page_visual_rescue_rejects_skin_hand_artifact():
    image = np.full((160, 220, 3), [176, 132, 104], dtype=np.uint8)
    cv2.line(image, (20, 40), (190, 120), (55, 36, 30), 5)
    candidate = {"sfx": {"visual_source": "red_chroma"}}

    assert _looks_like_long_page_visual_rescue_artifact(candidate, image, [0, 0, 220, 160])


def test_long_page_visual_rescue_rejects_muted_cloth_artifact():
    image = np.full((180, 160, 3), [128, 151, 128], dtype=np.uint8)
    cv2.line(image, (30, 0), (80, 180), (76, 82, 67), 3)
    candidate = {"sfx": {"visual_source": "color_chroma"}}

    assert _looks_like_long_page_visual_rescue_artifact(candidate, image, [0, 0, 160, 180])


def test_unconfirmed_light_color_sfx_rescue_keeps_bright_outlined_glyph():
    image = np.full((260, 180, 3), 238, dtype=np.uint8)
    for offset in range(5):
        x = 15 + offset * 18
        cv2.line(image, (x, 30), (x + 45, 230), (170, 115, 72), 10, cv2.LINE_AA)
        cv2.line(image, (x + 4, 36), (x + 38, 224), (255, 255, 248), 3, cv2.LINE_AA)
    candidate = {"sfx": {"visual_source": "color_chroma"}}

    assert _looks_like_unconfirmed_light_color_sfx(candidate, image, [0, 0, 180, 260])


def test_unconfirmed_light_color_sfx_rescue_rejects_bright_balloon_border():
    image = np.full((180, 260, 3), 248, dtype=np.uint8)
    cv2.ellipse(image, (130, 40), (120, 68), 0, 0, 360, (160, 92, 60), 3)
    candidate = {"sfx": {"visual_source": "color_chroma"}}

    assert not _looks_like_unconfirmed_light_color_sfx(candidate, image, [0, 0, 260, 180])


def test_detects_white_blue_sfx_with_outline():
    candidates = detect_sfx_candidates(_white_blue_sfx_page())

    assert candidates
    assert candidates[0]["content_class"] == "sfx"
    assert candidates[0]["route_action"] == "review_required"
    assert candidates[0]["sfx"]["inpaint_allowed"] is False


def test_does_not_mark_dialogue_balloon_text_as_sfx():
    image, blocks = _dialogue_balloon_page()

    candidates = detect_sfx_candidates(image, existing_blocks=blocks)

    assert candidates == []


def test_existing_text_suppresses_candidate_that_contains_ocr_text_box():
    assert _suppressed_by_existing_text([20, 20, 220, 140], [[72, 58, 180, 96]])


def test_latin_balloon_text_gate_rejects_flat_horizontal_text():
    crop = np.full((48, 150, 3), 255, dtype=np.uint8)
    cv2.putText(crop, "HYAAH!!", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2, cv2.LINE_AA)

    assert _looks_like_latin_balloon_text(crop, "local_contrast", 150, 48)


def test_plain_dialogue_caption_gate_rejects_wide_white_text_box():
    assert _looks_like_plain_dialogue_or_caption_crop(
        bw=260,
        bh=56,
        dark_ratio=0.09,
        bright_ratio=0.74,
        color_ratio=0.0,
        edge_ratio=0.045,
    )


def test_character_artifact_gate_rejects_skin_toned_color_crop():
    crop = np.full((80, 90, 3), [145, 95, 65], dtype=np.uint8)
    cv2.line(crop, (10, 10), (80, 70), (30, 25, 22), 3)

    assert _looks_like_character_artifact(crop, "color_chroma")


def test_large_panel_art_gate_rejects_broad_low_density_scene_crop():
    crop = np.full((180, 140, 3), [215, 225, 230], dtype=np.uint8)
    cv2.rectangle(crop, (0, 0), (139, 179), (110, 130, 145), 2)
    cv2.line(crop, (8, 150), (132, 118), (90, 105, 118), 2)

    assert _looks_like_large_panel_or_scene_artifact(
        crop,
        "local_contrast",
        area_ratio=0.12,
        fill_ratio=0.06,
    )


def test_large_panel_art_gate_keeps_dense_colored_sfx_crop():
    crop = np.full((180, 180, 3), [238, 238, 238], dtype=np.uint8)
    cv2.rectangle(crop, (35, 20), (70, 155), (125, 20, 35), -1)
    cv2.rectangle(crop, (35, 120), (145, 155), (125, 20, 35), -1)
    cv2.rectangle(crop, (110, 20), (145, 155), (125, 20, 35), -1)

    assert not _looks_like_large_panel_or_scene_artifact(
        crop,
        "color_chroma",
        area_ratio=0.12,
        fill_ratio=0.32,
    )


def test_long_page_gate_rejects_scanlator_footer_block():
    image = np.full((1200, 320, 3), 255, dtype=np.uint8)
    cv2.putText(image, "READ AT SCANLATOR", (28, 1130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    assert _looks_like_long_page_scanlator_or_ui_artifact(
        image,
        [20, 1080, 300, 1160],
        source="comic_text_detector_fallback",
        confidence=0.35,
    )


def test_long_page_gate_rejects_top_promo_or_ui_block():
    image = np.full((1200, 320, 3), 255, dtype=np.uint8)
    cv2.putText(image, "ENGLISH RELEASE", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

    assert _looks_like_long_page_scanlator_or_ui_artifact(
        image,
        [20, 20, 260, 100],
        source="comic_text_detector_fallback",
        confidence=0.30,
    )


def test_long_page_gate_keeps_mid_page_compact_sfx_block():
    image = np.full((1200, 320, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (92, 530), (122, 650), (118, 28, 42), -1)
    cv2.rectangle(image, (92, 620), (214, 650), (118, 28, 42), -1)

    assert not _looks_like_long_page_scanlator_or_ui_artifact(
        image,
        [80, 520, 220, 660],
        source="anime_text_yolo_low_conf",
        confidence=0.025,
    )


def test_post_ocr_gate_rejects_scanlator_logo_artifact():
    image = np.full((1200, 320, 3), [120, 160, 130], dtype=np.uint8)
    bbox = [70, 520, 170, 620]
    cv2.circle(image, (120, 570), 42, (70, 185, 105), -1)
    cv2.circle(image, (120, 570), 30, (190, 130, 70), -1)
    cv2.putText(image, "DS", (98, 583), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (35, 75, 40), 2, cv2.LINE_AA)

    assert _looks_like_scanlator_logo_artifact(
        {"confidence": 0.10, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_top_credit_artifact():
    image = np.full((1200, 320, 3), [20, 20, 22], dtype=np.uint8)
    bbox = [42, 22, 148, 145]
    cv2.rectangle(image, (42, 22), (148, 145), (245, 245, 238), -1)
    cv2.circle(image, (95, 72), 30, (80, 150, 90), -1)
    cv2.putText(image, "TEAM", (52, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA)

    assert _looks_like_top_credit_artifact(
        {"confidence": 0.12, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )


def test_post_ocr_gate_keeps_dark_compact_sfx_candidate():
    image = np.full((1200, 320, 3), 245, dtype=np.uint8)
    bbox = [80, 520, 220, 660]
    cv2.rectangle(image, (92, 530), (122, 650), (25, 25, 28), -1)
    cv2.rectangle(image, (92, 620), (214, 650), (25, 25, 28), -1)

    assert not _looks_like_scanlator_logo_artifact(
        {"confidence": 0.10, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )
    assert not _looks_like_overbroad_low_conf_long_page_artifact(
        {"confidence": 0.025, "sfx": {"visual_source": "anime_text_yolo_low_conf"}},
        image,
        bbox,
    )
    assert not _looks_like_top_credit_artifact(
        {"confidence": 0.12, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_unconfirmed_warm_local_contrast_artifact():
    image = np.full((220, 260, 3), [160, 105, 72], dtype=np.uint8)
    cv2.line(image, (20, 22), (230, 170), (55, 42, 35), 3, cv2.LINE_AA)
    cv2.line(image, (60, 180), (210, 38), (230, 190, 145), 2, cv2.LINE_AA)

    assert _looks_like_unconfirmed_visual_sfx_artifact(
        {"sfx": {"visual_source": "local_contrast"}, "sfx_ocr": {"status": "no_confident_cjk"}},
        image,
        [0, 0, 260, 220],
    )


def test_post_ocr_gate_rejects_large_unconfirmed_colored_panel_artifact():
    image = np.full((260, 320, 3), [84, 62, 150], dtype=np.uint8)
    cv2.circle(image, (170, 125), 72, (45, 30, 95), -1)
    cv2.line(image, (8, 230), (280, 52), (180, 80, 230), 3, cv2.LINE_AA)

    assert _looks_like_unconfirmed_visual_sfx_artifact(
        {"sfx": {"visual_source": "local_contrast"}, "sfx_ocr": {"status": "no_confident_cjk"}},
        image,
        [0, 0, 320, 260],
    )


def test_post_ocr_gate_keeps_large_unconfirmed_black_on_white_sfx():
    image = np.full((260, 220, 3), 255, dtype=np.uint8)
    cv2.line(image, (50, 15), (82, 212), (15, 15, 15), 11, cv2.LINE_AA)
    cv2.line(image, (105, 30), (150, 220), (15, 15, 15), 11, cv2.LINE_AA)

    assert not _looks_like_unconfirmed_visual_sfx_artifact(
        {"sfx": {"visual_source": "local_contrast"}, "sfx_ocr": {"status": "no_confident_cjk"}},
        image,
        [18, 0, 178, 238],
    )


def test_post_ocr_gate_rejects_short_page_blue_character_artifact():
    image = np.full((180, 220, 3), [214, 222, 238], dtype=np.uint8)
    bbox = [40, 30, 150, 135]
    cv2.ellipse(image, (95, 82), (56, 48), 20, 0, 360, (170, 160, 232), -1)
    cv2.line(image, (48, 120), (142, 30), (62, 72, 128), 3, cv2.LINE_AA)

    assert _looks_like_unconfirmed_visual_sfx_artifact(
        {"sfx": {"visual_source": "local_contrast"}, "sfx_ocr": {"status": "no_confident_cjk"}},
        image,
        bbox,
    )


def test_post_ocr_gate_keeps_short_page_dense_red_sfx():
    image = np.full((180, 220, 3), 248, dtype=np.uint8)
    bbox = [52, 32, 150, 118]
    cv2.line(image, (64, 42), (142, 100), (124, 24, 32), 12, cv2.LINE_AA)
    cv2.line(image, (136, 36), (70, 112), (124, 24, 32), 10, cv2.LINE_AA)

    assert not _looks_like_unconfirmed_visual_sfx_artifact(
        {"sfx": {"visual_source": "color_chroma"}, "sfx_ocr": {"status": "no_confident_cjk"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_short_page_warm_red_chroma_artifact():
    image = np.full((180, 220, 3), [180, 128, 82], dtype=np.uint8)
    bbox = [40, 32, 154, 140]
    cv2.rectangle(image, (40, 32), (154, 140), (160, 90, 62), -1)
    cv2.line(image, (48, 132), (146, 42), (70, 45, 38), 3, cv2.LINE_AA)

    assert _looks_like_unconfirmed_visual_sfx_artifact(
        {"sfx": {"visual_source": "red_chroma"}, "sfx_ocr": {"status": "no_confident_cjk"}},
        image,
        bbox,
    )


def test_detect_sfx_candidates_emits_tight_red_chroma_candidate():
    image = np.full((180, 240, 3), [28, 34, 46], dtype=np.uint8)
    cv2.putText(image, "ZZ", (86, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (128, 28, 34), 5, cv2.LINE_AA)

    candidates = detect_sfx_candidates(image, min_confidence=0.40)

    assert any((candidate.get("sfx") or {}).get("visual_source") == "red_chroma" for candidate in candidates)


def test_post_ocr_gate_keeps_short_page_cyan_glow_sfx():
    image = np.full((160, 220, 3), [20, 28, 36], dtype=np.uint8)
    bbox = [70, 48, 150, 116]
    cv2.putText(image, "S", (82, 104), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (115, 245, 250), 5, cv2.LINE_AA)
    cv2.line(image, (78, 54), (144, 112), (92, 220, 245), 3, cv2.LINE_AA)
    cv2.line(image, (144, 54), (82, 112), (92, 220, 245), 3, cv2.LINE_AA)

    assert not _looks_like_unconfirmed_visual_sfx_artifact(
        {"sfx": {"visual_source": "local_contrast"}, "sfx_ocr": {"status": "no_confident_cjk"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_latin_caption_attempts():
    candidate = {
        "sfx_ocr": {
            "status": "no_confident_cjk",
            "attempts": [
                {"text": "BOTH", "confidence": 0.99},
                {"text": "DIVIDE", "confidence": 0.98},
                {"text": "INNER", "confidence": 0.97},
            ],
        }
    }

    assert _looks_like_latin_ocr_text_artifact(candidate)


def test_post_ocr_gate_does_not_reject_noisy_short_latin_sfx_attempts():
    candidate = {
        "sfx_ocr": {
            "status": "no_confident_cjk",
            "attempts": [
                {"text": "Ko", "confidence": 0.54},
                {"text": "ot", "confidence": 0.56},
                {"text": "N", "confidence": 0.57},
            ],
        }
    }

    assert not _looks_like_latin_ocr_text_artifact(candidate)


def test_post_ocr_gate_rejects_long_page_footer_credit_even_when_recognized():
    image = np.full((1200, 320, 3), 245, dtype=np.uint8)
    bbox = [70, 1080, 270, 1145]
    cv2.putText(image, "ROUGH STUDIO", (76, 1120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 2, cv2.LINE_AA)

    assert _looks_like_long_page_footer_credit_artifact(
        {"sfx": {"visual_source": "comic_text_detector_fallback"}, "sfx_ocr": {"status": "recognized", "text": "작화"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_short_hangul_caption_artifact():
    image = np.full((1200, 320, 3), 255, dtype=np.uint8)
    bbox = [42, 520, 250, 650]
    cv2.putText(image, "THIS IS A", (48, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(image, "CAPTION TEXT", (48, 615), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (20, 20, 20), 2, cv2.LINE_AA)

    assert _looks_like_short_hangul_caption_artifact(
        {"sfx": {"visual_source": "comic_text_detector_fallback"}, "sfx_ocr": {"status": "recognized", "text": "야"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_top_chapter_ornament_artifact():
    image = np.full((1200, 320, 3), [25, 25, 28], dtype=np.uint8)
    bbox = [70, 70, 250, 155]
    cv2.rectangle(image, (70, 70), (250, 155), (210, 160, 40), -1)
    cv2.putText(image, "114", (126, 128), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3, cv2.LINE_AA)

    assert _looks_like_top_chapter_ornament_artifact(
        {"confidence": 0.08, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_pale_vertical_artifact():
    image = np.full((1200, 320, 3), [225, 232, 238], dtype=np.uint8)
    bbox = [40, 480, 96, 720]
    cv2.line(image, (65, 490), (78, 700), (170, 178, 188), 3, cv2.LINE_AA)

    assert _looks_like_pale_vertical_artifact(
        {"confidence": 0.025, "sfx": {"visual_source": "anime_text_yolo_low_conf"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_low_detail_logo_artifact():
    image = np.full((1200, 320, 3), [232, 232, 232], dtype=np.uint8)
    bbox = [92, 520, 212, 640]
    cv2.circle(image, (152, 580), 42, (178, 181, 182), -1)
    cv2.circle(image, (152, 580), 26, (205, 205, 205), -1)

    assert _looks_like_low_detail_logo_artifact(
        {"confidence": 0.18, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_low_detail_blue_scene_artifact():
    image = np.full((180, 240, 3), [38, 62, 78], dtype=np.uint8)
    bbox = [42, 52, 172, 112]
    cv2.rectangle(image, (42, 52), (172, 112), (42, 76, 92), -1)
    cv2.line(image, (48, 104), (166, 58), (26, 44, 58), 2, cv2.LINE_AA)

    assert _looks_like_low_detail_blue_scene_artifact(
        {
            "confidence": 0.07,
            "sfx": {"visual_source": "comic_text_detector_fallback"},
            "sfx_ocr": {"status": "no_confident_cjk"},
        },
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_low_conf_warm_artifact():
    image = np.full((1200, 320, 3), [230, 232, 236], dtype=np.uint8)
    bbox = [90, 420, 175, 500]
    cv2.rectangle(image, (108, 436), (175, 500), (178, 130, 92), -1)
    cv2.line(image, (96, 490), (166, 426), (70, 58, 52), 2, cv2.LINE_AA)

    assert _looks_like_low_conf_warm_artifact(
        {"confidence": 0.024, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_low_conf_horizontal_texture_artifact():
    image = np.full((1200, 320, 3), [220, 228, 232], dtype=np.uint8)
    bbox = [40, 280, 260, 330]
    cv2.rectangle(image, (40, 280), (260, 330), (122, 138, 144), -1)
    for y in range(286, 326, 8):
        cv2.line(image, (44, y), (256, y + 2), (152, 164, 170), 1, cv2.LINE_AA)

    assert _looks_like_low_conf_horizontal_texture_artifact(
        {"confidence": 0.07, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_overbroad_low_conf_long_page_candidate():
    image = np.full((1200, 320, 3), 245, dtype=np.uint8)

    assert _looks_like_overbroad_low_conf_long_page_artifact(
        {"confidence": 0.012, "sfx": {"visual_source": "anime_text_yolo_low_conf"}},
        image,
        [18, 700, 300, 900],
    )


def test_post_ocr_gate_rejects_white_dialogue_caption_artifact():
    image = np.full((1200, 320, 3), [210, 216, 220], dtype=np.uint8)
    bbox = [30, 520, 250, 670]
    cv2.rectangle(image, (30, 520), (250, 670), (255, 255, 255), -1)
    cv2.rectangle(image, (30, 520), (250, 670), (20, 20, 20), 2)
    cv2.putText(image, "THERE WAS", (62, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(image, "NO PRINCIPLE", (46, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)

    assert _looks_like_white_dialogue_or_caption_artifact(
        {"confidence": 0.12, "sfx": {"visual_source": "anime_text_yolo_low_conf"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_nearly_square_white_logo_caption_artifact():
    image = np.full((1200, 320, 3), [210, 216, 220], dtype=np.uint8)
    bbox = [70, 520, 190, 630]
    cv2.rectangle(image, (70, 520), (190, 630), (255, 255, 255), -1)
    cv2.putText(image, "LOGO", (88, 584), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)

    assert _looks_like_white_dialogue_or_caption_artifact(
        {"confidence": 0.10, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )


def test_post_ocr_gate_keeps_nonwhite_cjk_sfx_shape():
    image = np.full((1200, 320, 3), [40, 50, 60], dtype=np.uint8)
    bbox = [80, 520, 220, 660]
    cv2.rectangle(image, (92, 530), (122, 650), (245, 250, 255), -1)
    cv2.rectangle(image, (92, 620), (214, 650), (245, 250, 255), -1)

    assert not _looks_like_white_dialogue_or_caption_artifact(
        {"confidence": 0.12, "sfx": {"visual_source": "anime_text_yolo_low_conf"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_dark_narration_caption_artifact():
    image = np.full((1200, 320, 3), [35, 45, 52], dtype=np.uint8)
    bbox = [38, 520, 282, 650]
    cv2.rectangle(image, (38, 520), (282, 650), (20, 24, 26), -1)
    cv2.putText(image, "COME THEN", (70, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 245, 240), 2, cv2.LINE_AA)
    cv2.putText(image, "LET'S SEE THIS", (54, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 245, 240), 2, cv2.LINE_AA)

    assert _looks_like_dark_narration_caption_artifact(
        {"confidence": 0.92, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )


def test_post_ocr_gate_keeps_dark_compact_non_caption_sfx():
    image = np.full((1200, 320, 3), [35, 45, 52], dtype=np.uint8)
    bbox = [80, 520, 220, 660]
    cv2.line(image, (100, 530), (145, 650), (245, 250, 255), 12, cv2.LINE_AA)
    cv2.line(image, (165, 530), (120, 650), (245, 250, 255), 12, cv2.LINE_AA)

    assert not _looks_like_dark_narration_caption_artifact(
        {"confidence": 0.92, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        bbox,
    )


def test_post_ocr_gate_rejects_spurious_cjk_recognition_on_latin_caption():
    image = np.full((1200, 320, 3), [35, 45, 52], dtype=np.uint8)
    bbox = [38, 520, 282, 650]
    cv2.rectangle(image, (38, 520), (282, 650), (20, 24, 26), -1)
    cv2.putText(image, "YANG-GAM WILL", (54, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 245, 240), 2)
    cv2.putText(image, "BE PLEASED", (76, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 245, 240), 2)
    candidate = {
        "bbox": bbox,
        "confidence": 0.94,
        "recognized_text": "エAS",
        "sfx": {"visual_source": "comic_text_detector_fallback"},
        "sfx_ocr": {
            "status": "recognized",
            "text": "エAS",
            "attempts": [
                {"text": "YANG-GAM"},
                {"text": "WILL"},
                {"text": "BE"},
                {"text": "PLEASED"},
            ],
        },
    }

    assert _looks_like_spurious_recognized_caption_artifact(candidate, image, bbox)
    assert _should_drop_non_cjk_sfx_artifact(candidate, image)


def test_post_ocr_gate_rejects_broad_low_conf_comic_art_candidate():
    image = np.full((1200, 320, 3), [130, 145, 150], dtype=np.uint8)

    assert _looks_like_overbroad_low_conf_long_page_artifact(
        {"confidence": 0.08, "sfx": {"visual_source": "comic_text_detector_fallback"}},
        image,
        [30, 520, 280, 760],
    )


def test_post_ocr_gate_rejects_single_spurious_glyph_in_white_bubble():
    image = np.full((1200, 320, 3), [210, 216, 220], dtype=np.uint8)
    bbox = [170, 940, 260, 1060]
    cv2.ellipse(image, (215, 1000), (44, 58), 0, 0, 360, (255, 255, 255), -1)
    cv2.putText(image, "!", (205, 1018), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 2, cv2.LINE_AA)
    candidate = {
        "bbox": bbox,
        "recognized_text": "つ",
        "sfx": {"visual_source": "anime_text_yolo_low_conf"},
        "sfx_ocr": {"status": "recognized", "text": "つ", "confidence": 0.78},
    }

    assert _looks_like_single_glyph_white_bubble_artifact(candidate, image, bbox)
    assert _looks_like_spurious_recognized_caption_artifact(candidate, image, bbox)


def test_white_fragment_suppression_keeps_large_sfx_box():
    large = {
        "id": "sfx_visual_001",
        "bbox": [40, 70, 205, 225],
        "confidence": 0.85,
        "sfx": {"visual_source": "white_near_chroma"},
    }
    fragment = {
        "id": "sfx_visual_002",
        "bbox": [185, 95, 237, 193],
        "confidence": 0.82,
        "sfx": {"visual_source": "white_near_chroma"},
    }

    kept = _suppress_color_fragments_near_white_candidates([large, fragment])

    assert kept == [large]


def test_low_conf_text_detector_rescue_keeps_large_sfx_and_drops_small_fragment():
    image = np.full((600, 320, 3), 248, dtype=np.uint8)
    cv2.putText(image, "o!", (82, 520), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (5, 5, 5), 8, cv2.LINE_AA)
    cv2.line(image, (20, 320), (42, 370), (5, 5, 5), 3)

    candidates = text_blocks_to_sfx_candidates(
        image,
        [
            {"bbox": [73, 440, 177, 541], "confidence": 0.01079},
            {"bbox": [14, 316, 45, 366], "confidence": 0.0131},
        ],
        source="anime_text_yolo_low_conf",
        min_confidence=0.0107,
    )

    assert len(candidates) == 1
    assert candidates[0]["bbox"] == [73, 440, 177, 541]
    assert candidates[0]["detector"] == "sfx_text_detector"
    assert "sfx_text_detector_candidate" in candidates[0]["qa_flags"]


def test_merge_sfx_candidates_suppresses_contained_low_conf_duplicate():
    primary = {
        "id": "sfx_visual_001",
        "text_id": "sfx_visual_001",
        "bbox": [189, 222, 309, 380],
        "confidence": 0.268,
        "sfx": {"visual_source": "anime_text_yolo_low_conf"},
    }
    duplicate = {
        "id": "sfx_visual_002",
        "text_id": "sfx_visual_002",
        "bbox": [205, 255, 279, 375],
        "confidence": 0.012,
        "sfx": {"visual_source": "anime_text_yolo_low_conf"},
    }

    kept = merge_sfx_candidates([primary, duplicate])

    assert len(kept) == 1
    assert kept[0]["bbox"] == primary["bbox"]


def test_merge_sfx_candidates_prefers_text_detector_box_over_visual_fragment():
    visual = {
        "id": "sfx_visual_001",
        "text_id": "sfx_visual_001",
        "bbox": [120, 120, 180, 170],
        "text_pixel_bbox": [120, 120, 180, 170],
        "confidence": 0.92,
        "detector": "sfx_visual",
        "qa_flags": ["sfx_visual_candidate"],
        "sfx": {"visual_source": "red_chroma"},
    }
    text_detector = {
        "id": "sfx_visual_002",
        "text_id": "sfx_visual_002",
        "bbox": [70, 60, 190, 190],
        "text_pixel_bbox": [70, 60, 190, 190],
        "confidence": 0.04,
        "detector": "sfx_text_detector",
        "qa_flags": ["sfx_visual_candidate", "sfx_text_detector_candidate"],
        "sfx": {"visual_source": "anime_text_yolo_low_conf"},
    }

    kept = merge_sfx_candidates([visual, text_detector])

    assert len(kept) == 1
    assert kept[0]["bbox"] == text_detector["bbox"]
    assert kept[0]["detector"] == "sfx_text_detector"


def test_sfx_clean_expectations_fixture_matches_probe_summary(tmp_path):
    summary = {
        "pages": [
            {
                "image": "N:\\TraduzAI\\data\\sfx_benchmarks\\sfx_clean_inputs\\Captura de tela 2026-06-05 221235.png",
                "final_candidates": [{"bbox": [64, 165, 132, 262]}],
            }
        ]
    }
    expectations = {
        "min_iou": 0.42,
        "pages": {
            "Captura de tela 2026-06-05 221235.png": {
                "min_final_count": 1,
                "max_final_count": 1,
                "required_boxes": [{"label": "purple floor sfx", "bbox": [64, 165, 132, 262]}],
                "forbidden_boxes": [{"label": "people and floor artifact", "bbox": [262, 7, 454, 199]}],
            }
        },
    }
    path = tmp_path / "expect.json"
    path.write_text(json.dumps(expectations), encoding="utf-8")

    result = _validate_expectations(summary, path)

    assert result["passed"] is True
    assert result["failures"] == []


def test_sfx_expectations_raise_on_missing_or_forbidden_box(tmp_path):
    summary = {
        "pages": [
            {
                "image": "N:\\TraduzAI\\data\\sfx_benchmarks\\sfx_clean_inputs\\sample.png",
                "final_candidates": [{"bbox": [200, 200, 260, 260]}],
            }
        ]
    }
    expectations = {
        "min_iou": 0.42,
        "pages": {
            "sample.png": {
                "required_boxes": [{"label": "missing sfx", "bbox": [10, 10, 60, 60]}],
                "forbidden_boxes": [{"label": "bad art", "bbox": [200, 200, 260, 260]}],
            }
        },
    }
    path = tmp_path / "expect.json"
    path.write_text(json.dumps(expectations), encoding="utf-8")

    try:
        _validate_expectations(summary, path)
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("expected _validate_expectations to fail")

    assert "missing required" in message
    assert "forbidden" in message


def test_validate_expectations_requires_all_pages_when_enabled(tmp_path):
    summary = {
        "pages": [
            {"image": "N:\\TraduzAI\\first.png", "final_candidates": []},
            {"image": "N:\\TraduzAI\\second.png", "final_candidates": []},
        ]
    }
    expectations = {
        "require_all_pages": True,
        "pages": {
            "first.png": {"reviewed": True, "min_final_count": 0, "max_final_count": 0},
        },
    }
    path = tmp_path / "expect.json"
    path.write_text(json.dumps(expectations), encoding="utf-8")

    try:
        _validate_expectations(summary, path)
    except SystemExit as exc:
        assert "second.png: page missing from expectations" in str(exc)
    else:
        raise AssertionError("expected _validate_expectations to fail")


def test_validate_expectations_requires_reviewed_pages_when_enabled(tmp_path):
    summary = {"pages": [{"image": "N:\\TraduzAI\\sample.png", "final_candidates": []}]}
    expectations = {
        "require_reviewed_pages": True,
        "pages": {
            "sample.png": {"min_final_count": 0, "max_final_count": 0},
        },
    }
    path = tmp_path / "expect.json"
    path.write_text(json.dumps(expectations), encoding="utf-8")

    try:
        _validate_expectations(summary, path)
    except SystemExit as exc:
        assert "sample.png: expectation page must set reviewed=true" in str(exc)
    else:
        raise AssertionError("expected _validate_expectations to fail")


def test_final_crops_sheet_renders_review_grid():
    crop = np.full((80, 120, 3), 230, dtype=np.uint8)
    cv2.rectangle(crop, (20, 12), (92, 70), (120, 20, 40), -1)

    sheet = _make_final_crops_sheet([("sample.png #1 10,10,60,60", crop)])

    assert sheet.shape[0] >= 190
    assert sheet.shape[1] >= 220
    assert int(sheet.std()) > 0


def test_validate_sfx_expectations_cli_passes(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "summary.json"
    expect_path = tmp_path / "expect.json"
    summary_path.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "image": "N:\\TraduzAI\\sample.png",
                        "final_candidates": [{"bbox": [10, 10, 60, 60]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    expect_path.write_text(
        json.dumps(
            {
                "pages": {
                    "sample.png": {
                        "min_final_count": 1,
                        "max_final_count": 1,
                        "required_boxes": [{"bbox": [10, 10, 60, 60]}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_sfx_expectations", "--summary", str(summary_path), "--expect", str(expect_path)],
    )

    assert validate_sfx_expectations_main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is True


def test_validate_sfx_expectations_cli_fails(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    expect_path = tmp_path / "expect.json"
    summary_path.write_text(
        json.dumps({"pages": [{"image": "N:\\TraduzAI\\sample.png", "final_candidates": []}]}),
        encoding="utf-8",
    )
    expect_path.write_text(
        json.dumps({"pages": {"sample.png": {"required_boxes": [{"bbox": [10, 10, 60, 60]}]}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_sfx_expectations", "--summary", str(summary_path), "--expect", str(expect_path)],
    )

    try:
        validate_sfx_expectations_main()
    except SystemExit as exc:
        assert "missing required" in str(exc)
    else:
        raise AssertionError("expected validation failure")


def test_short_page_visual_merge_combines_nearby_same_source_fragments():
    first = {
        "id": "sfx_visual_001",
        "text_id": "sfx_visual_001",
        "bbox": [40, 40, 110, 120],
        "text_pixel_bbox": [40, 40, 110, 120],
        "detector": "sfx_visual",
        "confidence": 0.76,
        "qa_flags": ["sfx_visual_candidate"],
        "sfx": {"visual_source": "local_contrast", "qa_flags": ["sfx_visual_candidate"]},
    }
    second = {
        "id": "sfx_visual_002",
        "text_id": "sfx_visual_002",
        "bbox": [102, 78, 172, 150],
        "text_pixel_bbox": [102, 78, 172, 150],
        "detector": "sfx_visual",
        "confidence": 0.72,
        "qa_flags": ["sfx_visual_candidate"],
        "sfx": {"visual_source": "local_contrast", "qa_flags": ["sfx_visual_candidate"]},
    }

    merged = _merge_nearby_short_page_visual_candidates([first, second], (320, 260, 3))

    assert len(merged) == 1
    assert merged[0]["bbox"] == [40, 40, 172, 150]
    assert "sfx_visual_fragment_merged" in merged[0]["qa_flags"]


def test_short_page_visual_merge_keeps_distant_fragments_separate():
    first = {
        "bbox": [20, 20, 80, 90],
        "detector": "sfx_visual",
        "confidence": 0.76,
        "sfx": {"visual_source": "local_contrast"},
    }
    second = {
        "bbox": [190, 210, 250, 280],
        "detector": "sfx_visual",
        "confidence": 0.72,
        "sfx": {"visual_source": "local_contrast"},
    }

    merged = _merge_nearby_short_page_visual_candidates([first, second], (320, 260, 3))

    assert len(merged) == 2
