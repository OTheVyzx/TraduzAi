import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_detect_residual_text_flags_dark_remnants_inside_region():
    from qa.inpaint_residual import detect_residual_text

    before = np.full((32, 48, 3), 245, dtype=np.uint8)
    after_clean = before.copy()
    after_residual = before.copy()
    after_residual[12:15, 16:32] = 48
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[8:20, 10:38] = 255

    clean = detect_residual_text(before, after_clean, mask)
    residual = detect_residual_text(before, after_residual, mask)

    assert clean["has_residual"] is False
    assert clean["score"] == 0.0
    assert residual["has_residual"] is True
    assert residual["score"] > clean["score"]
    assert "dark_residual_pixels" in residual["flags"]


def test_detect_residual_text_uses_absolute_pixel_gate_for_large_regions():
    from qa.inpaint_residual import detect_residual_text

    before = np.full((160, 240, 3), 245, dtype=np.uint8)
    after = before.copy()
    after[72:76, 90:108] = 48
    mask = np.ones((160, 240), dtype=np.uint8) * 255

    residual = detect_residual_text(before, after, mask)

    assert residual["dark_residual_pixels"] >= 32
    assert residual["score"] < 0.01
    assert residual["has_residual"] is True


def test_detect_residual_text_flags_bright_remnants_on_colored_panel():
    from qa.inpaint_residual import detect_residual_text

    before = np.full((52, 120, 3), (72, 136, 214), dtype=np.uint8)
    after_residual = before.copy()
    import cv2

    cv2.putText(
        after_residual,
        "TITLE",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (248, 248, 248),
        2,
        cv2.LINE_AA,
    )
    mask = np.zeros((52, 120), dtype=np.uint8)
    mask[10:42, 12:108] = 255

    residual = detect_residual_text(after_residual, after_residual, mask, include_light_residual=True)

    assert residual["has_residual"] is True
    assert residual["light_residual_pixels"] > 0
    assert "light_residual_pixels" in residual["flags"]


def test_detect_residual_text_does_not_flag_removed_light_text_as_dark_residue():
    from qa.inpaint_residual import detect_residual_text

    before = np.full((70, 160, 3), 42, dtype=np.uint8)
    after = before.copy()
    before[24:32, 34:126] = 245
    before[38:46, 44:116] = 245
    mask = np.zeros((70, 160), dtype=np.uint8)
    mask[18:52, 24:136] = 255

    residual = detect_residual_text(before, after, mask, include_light_residual=True)

    assert residual["has_residual"] is False
    assert residual["dark_residual_pixels"] == 0
    assert residual["light_residual_pixels"] == 0


def test_detect_residual_text_does_not_flag_removed_blue_text_as_dark_residue():
    from qa.inpaint_residual import detect_residual_text

    before = np.full((72, 180, 3), (8, 12, 70), dtype=np.uint8)
    after = before.copy()
    import cv2

    cv2.putText(
        before,
        "SYSTEM",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        (20, 80, 230),
        2,
        cv2.LINE_AA,
    )
    mask = np.zeros((72, 180), dtype=np.uint8)
    mask[12:58, 14:166] = 255

    residual = detect_residual_text(before, after, mask, include_light_residual=True)

    assert residual["has_residual"] is False
    assert residual["dark_residual_pixels"] == 0
    assert residual["colored_residual_pixels"] == 0


def test_detect_residual_text_flags_visible_blue_text_on_dark_panel():
    from qa.inpaint_residual import detect_residual_text

    before = np.full((72, 180, 3), (8, 12, 70), dtype=np.uint8)
    import cv2

    cv2.putText(
        before,
        "SYSTEM",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        (20, 80, 230),
        2,
        cv2.LINE_AA,
    )
    after = before.copy()
    mask = np.zeros((72, 180), dtype=np.uint8)
    mask[12:58, 14:166] = 255

    residual = detect_residual_text(before, after, mask, include_light_residual=True)

    assert residual["has_residual"] is True
    assert residual["colored_residual_pixels"] > 0
    assert "colored_residual_pixels" in residual["flags"]


def test_detect_residual_text_ignores_smooth_dark_inpainted_art_region():
    from qa.inpaint_residual import detect_residual_text

    before = np.full((80, 180, 3), (205, 184, 140), dtype=np.uint8)
    after = np.full((80, 180, 3), (70, 62, 58), dtype=np.uint8)
    mask = np.zeros((80, 180), dtype=np.uint8)
    mask[16:64, 20:160] = 255

    residual = detect_residual_text(before, after, mask, include_light_residual=True)

    assert residual["has_residual"] is False
    assert residual["dark_residual_pixels"] == 0


def test_residual_text_qa_flag_downgrades_low_signal_light_residual():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.007,
            "flags": ["light_residual_pixels"],
            "dark_residual_pixels": 0,
            "light_residual_pixels": 43,
        }
    )

    assert flag == "weak_text_residual_after_inpaint"


def test_residual_text_qa_flag_keeps_high_ratio_residual_blocking():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.06,
            "flags": ["light_residual_pixels", "high_residual_ratio", "dark_panel_fill_applied"],
            "dark_residual_pixels": 0,
            "light_residual_pixels": 1976,
            "light_residual_on_dark_context": True,
        }
    )

    assert flag == "text_residual_after_inpaint"


def test_residual_text_qa_flag_downgrades_low_ratio_dark_panel_fill_artifact():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.0259,
            "flags": ["dark_residual_pixels", "colored_residual_pixels", "dark_panel_fill_applied"],
            "dark_residual_pixels": 3307,
            "light_residual_pixels": 31,
            "colored_residual_pixels": 1590,
            "dark_background_context": True,
        }
    )

    assert flag == "weak_text_residual_after_inpaint"


def test_residual_text_qa_flag_downgrades_medium_dark_panel_fill_artifact_without_text_color():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.066,
            "flags": ["dark_residual_pixels", "high_residual_ratio", "dark_panel_fill_applied"],
            "dark_residual_pixels": 1627,
            "light_residual_pixels": 0,
            "colored_residual_pixels": 0,
            "dark_background_context": False,
        }
    )

    assert flag == "weak_text_residual_after_inpaint"


def test_residual_text_qa_flag_downgrades_small_colored_panel_glow_artifact():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.0369,
            "flags": ["colored_residual_pixels", "dark_panel_fill_applied"],
            "dark_residual_pixels": 0,
            "light_residual_pixels": 0,
            "colored_residual_pixels": 967,
            "light_residual_on_dark_context": True,
            "dark_background_context": True,
        }
    )

    assert flag == "weak_text_residual_after_inpaint"


def test_residual_text_qa_flag_downgrades_light_only_white_region_even_when_dense():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.29,
            "flags": ["light_residual_pixels", "high_residual_ratio"],
            "dark_residual_pixels": 0,
            "light_residual_pixels": 5192,
            "light_residual_on_dark_context": False,
            "dark_background_context": False,
        }
    )

    assert flag == "weak_text_residual_after_inpaint"


def test_residual_text_qa_flag_keeps_dense_dark_residual_blocking():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.006,
            "flags": ["dark_residual_pixels"],
            "dark_residual_pixels": 396,
            "light_residual_pixels": 0,
        }
    )

    assert flag == "text_residual_after_inpaint"


def test_residual_text_qa_flag_downgrades_small_non_high_dark_artifact():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.041447,
            "flags": ["dark_residual_pixels"],
            "dark_residual_pixels": 321,
            "light_residual_pixels": 1,
            "colored_residual_pixels": 0,
            "dark_background_context": False,
            "region_source": "expanded_mask",
        }
    )

    assert flag == "weak_text_residual_after_inpaint"


def test_residual_text_qa_flag_downgrades_tiny_white_balloon_line_art_residual():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.00112,
            "flags": ["dark_residual_pixels"],
            "dark_residual_pixels": 168,
            "light_residual_pixels": 0,
            "region_source": "text_region_white_balloon",
        }
    )

    assert flag == "weak_text_residual_after_inpaint"


def test_residual_text_qa_flag_downgrades_low_ratio_white_balloon_art_pixels():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.005757,
            "flags": ["dark_residual_pixels"],
            "dark_residual_pixels": 1000,
            "light_residual_pixels": 0,
            "colored_residual_pixels": 0,
            "dark_background_context": True,
            "region_source": "text_region_white_balloon",
        }
    )

    assert flag == "weak_text_residual_after_inpaint"


def test_residual_text_qa_flag_downgrades_low_dark_sign_border_residual():
    from inpainter import _residual_text_qa_flag

    flag = _residual_text_qa_flag(
        {
            "has_residual": True,
            "score": 0.022172,
            "flags": ["dark_residual_pixels"],
            "dark_residual_pixels": 138,
            "light_residual_pixels": 0,
            "colored_residual_pixels": 0,
            "dark_background_context": True,
            "region_source": "text_region_white_balloon",
        }
    )

    assert flag == "weak_text_residual_after_inpaint"


def test_inpaint_band_image_records_decision_payload_with_active_debug_recorder(tmp_path):
    from debug_tools import DebugRecorder, bind_recorder
    from inpainter import inpaint_band_image

    image = np.full((80, 120, 3), 245, dtype=np.uint8)
    image[30:42, 45:75] = 10
    ocr_page = {
        "_band_index": 7,
        "_source_page_number": 3,
        "texts": [
            {
                "id": "ocr_007",
                "text": "YES",
                "translated": "SIM",
                "bbox": [45, 30, 75, 42],
                "text_pixel_bbox": [45, 30, 75, 42],
                "tipo": "fala",
                "confidence": 0.95,
                "balloon_type": "white",
            }
        ],
        "_vision_blocks": [{"bbox": [45, 30, 75, 42], "mask": None, "confidence": 0.95}],
    }

    def fake_cleanup(_original, cleaned, _texts, **_kwargs):
        result = cleaned.copy()
        result[0, 0] = [0, 0, 0]
        return result, {"cleanup_changed_outside_limit_mask": 1}

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        with patch.dict(
            os.environ,
            {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            },
        ), patch("vision_stack.runtime._get_inpainter", return_value=object()), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=lambda image_np, payload, inp: image_np.copy(),
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=fake_cleanup,
        ):
            inpaint_band_image(image, ocr_page)
    finally:
        bind_recorder(None)

    decision_path = tmp_path / "debug" / "e2e" / "08_inpaint" / "page_003_band_007" / "inpaint_decision.json"
    payload = json.loads(decision_path.read_text(encoding="utf-8"))

    assert payload["stage"] == "inpaint"
    assert payload["page_id"] == "page_003"
    assert payload["band_id"] == "page_003_band_007"
    assert payload["text_ids"] == ["ocr_007"]
    assert payload["trace_ids"] == ["ocr_007@page_003_band_007"]
    assert payload["trace_ids_in_band"] == ["ocr_007@page_003_band_007"]
    assert payload["skip_inpaint_honored"] is False
    assert payload["used_fast_white_fill"] is False
    assert payload["used_fast_local_fill"] is False
    assert payload["used_real_inpaint"] is True
    assert payload["used_post_cleanup"] is True
    assert payload["changed_outside_expanded_pixels"] > 0
    assert "has_residual" in payload["residual_text"]
