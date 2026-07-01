import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_translator_note_text_mask_fills_with_dark_background_near_white_bubble():
    from inpainter import _try_dark_panel_text_fill

    image = np.full((180, 320, 3), 8, dtype=np.uint8)
    image[20:145, 170:300] = 255
    image[70:74, 22:168] = 245
    image[88:92, 26:168] = 245
    image[106:110, 16:186] = 245
    text = {
        "id": "ocr_tn_dark",
        "bbox": [16, 66, 186, 116],
        "text_pixel_bbox": [16, 66, 186, 116],
        "line_polygons": [
            [[22, 68], [168, 68], [168, 78], [22, 78]],
            [[26, 86], [168, 86], [168, 96], [26, 96]],
            [[16, 104], [186, 104], [186, 114], [16, 114]],
        ],
        "bubble_mask_source": "translator_note_text_mask",
        "background_rgb": [8, 8, 8],
        "qa_flags": ["translator_note_text_only_mask"],
        "route_action": "translate_inpaint_render",
    }

    filled = _try_dark_panel_text_fill(image, text)

    assert filled is not None
    changed = np.any(filled != image, axis=2)
    assert int(np.count_nonzero(changed)) > 0
    assert float(np.mean(filled[changed])) < 40.0


def test_fast_fill_resolved_text_is_suppressed_from_real_inpaint_rebuild():
    from inpainter import _text_suppressed_for_inpaint

    text = {
        "id": "ocr_tn_dark",
        "bubble_mask_source": "translator_note_text_mask",
        "route_action": "translate_inpaint_render",
        "_fast_fill_inpaint_resolved": True,
    }

    assert _text_suppressed_for_inpaint(text) is True


def test_image_white_bubble_mask_with_dark_background_is_not_white_context():
    from inpainter import _text_is_white_balloon_context

    text = {
        "id": "ocr_dark_false_white",
        "bubble_mask_source": "image_white_bubble_mask",
        "background_rgb": [12, 12, 12],
        "style_origin": "auto_dark_panel_glow",
        "layout_profile": "connected_balloon",
        "qa_flags": ["auto_dark_panel_glow_fallback", "mask_outside_balloon_critical"],
    }

    assert _text_is_white_balloon_context(text) is False


def test_image_white_bubble_mask_with_plain_white_background_stays_white_context():
    from inpainter import _text_is_white_balloon_context

    text = {
        "id": "ocr_white_balloon",
        "bubble_mask_source": "image_white_bubble_mask",
        "background_rgb": [248, 248, 248],
        "layout_profile": "speech_balloon",
    }

    assert _text_is_white_balloon_context(text) is True


def test_false_white_dark_bubble_uses_local_dark_fill_when_global_fast_fill_disabled():
    from inpainter import _apply_fast_dark_panel_text_fill

    image = np.full((140, 260, 3), 8, dtype=np.uint8)
    image[46:56, 92:166] = 236
    image[66:76, 84:174] = 236
    text = {
        "id": "ocr_false_white_dark",
        "bbox": [80, 42, 178, 80],
        "text_pixel_bbox": [80, 42, 178, 80],
        "line_polygons": [
            [[92, 46], [166, 46], [166, 56], [92, 56]],
            [[84, 66], [174, 66], [174, 76], [84, 76]],
        ],
        "balloon_bbox": [42, 22, 220, 112],
        "bubble_mask_bbox": [42, 22, 220, 112],
        "bubble_mask_source": "image_white_bubble_mask",
        "route_action": "translate_inpaint_render",
        "qa_flags": ["mask_outside_balloon_critical", "bubble_clip_preserved_raw_text"],
    }
    vision_blocks = [dict(text)]

    filled, remaining, stats = _apply_fast_dark_panel_text_fill(image, {"texts": [text]}, vision_blocks)

    assert stats["dark_panel_fill_count"] == 1
    assert remaining == []
    changed = np.any(filled != image, axis=2)
    assert int(np.count_nonzero(changed)) > 0
    assert float(np.mean(filled[changed])) < 48.0


def test_false_white_dark_bubble_with_preserved_clip_uses_local_dark_fill():
    from inpainter import _apply_fast_dark_panel_text_fill

    image = np.full((140, 300, 3), 7, dtype=np.uint8)
    image[58:68, 118:220] = 238
    image[78:88, 104:238] = 238
    text = {
        "id": "ocr_false_white_clip",
        "bbox": [98, 54, 244, 92],
        "text_pixel_bbox": [98, 54, 244, 92],
        "line_polygons": [
            [[118, 58], [220, 58], [220, 68], [118, 68]],
            [[104, 78], [238, 78], [238, 88], [104, 88]],
        ],
        "balloon_bbox": [76, 34, 270, 122],
        "bubble_mask_bbox": [76, 34, 270, 122],
        "bubble_mask_source": "image_white_bubble_mask",
        "route_action": "translate_inpaint_render",
        "qa_flags": ["bubble_clip_preserved_raw_text"],
    }

    filled, remaining, stats = _apply_fast_dark_panel_text_fill(image, {"texts": [text]}, [dict(text)])

    assert stats["dark_panel_fill_count"] == 1
    assert remaining == []
    changed = np.any(filled != image, axis=2)
    assert int(np.count_nonzero(changed)) > 0
    assert float(np.mean(filled[changed])) < 48.0


def test_broken_image_white_dark_bubble_cleans_bright_text_outside_false_bbox():
    from inpainter import _apply_fast_dark_panel_text_fill

    image = np.full((330, 820, 3), 5, dtype=np.uint8)
    image[86:108, 388:692] = 244
    image[130:152, 382:710] = 244
    image[176:198, 420:668] = 244
    image[222:244, 472:626] = 244
    text = {
        "id": "ocr_partial_dark_bubble",
        "bbox": [452, 184, 655, 226],
        "text_pixel_bbox": [402, 176, 660, 232],
        "line_polygons": [[[452, 184], [655, 184], [655, 226], [452, 226]]],
        "balloon_bbox": [451, 170, 655, 260],
        "bubble_mask_bbox": [451, 170, 655, 260],
        "bubble_mask_source": "image_white_bubble_mask",
        "route_action": "translate_inpaint_render",
        "qa_flags": ["mask_outside_balloon_critical"],
    }

    filled, remaining, stats = _apply_fast_dark_panel_text_fill(image, {"texts": [text]}, [dict(text)])

    assert stats["dark_panel_fill_count"] == 1
    assert remaining == []
    assert float(np.mean(filled[86:108, 388:692])) < 48.0
    assert float(np.mean(filled[130:152, 382:710])) < 48.0
    assert float(np.mean(filled[176:198, 420:668])) < 48.0
    assert float(np.mean(filled[222:244, 472:626])) < 48.0


def test_image_dark_bubble_uses_local_dark_fill_when_global_fast_fill_disabled():
    from inpainter import _apply_fast_dark_panel_text_fill

    image = np.full((150, 320, 3), 6, dtype=np.uint8)
    image[54:66, 92:220] = 232
    image[78:90, 74:236] = 232
    text = {
        "id": "ocr_dark_bubble",
        "bbox": [70, 50, 240, 94],
        "text_pixel_bbox": [70, 50, 240, 94],
        "line_polygons": [
            [[92, 54], [220, 54], [220, 66], [92, 66]],
            [[74, 78], [236, 78], [236, 90], [74, 90]],
        ],
        "balloon_bbox": [42, 20, 278, 126],
        "bubble_mask_bbox": [42, 20, 278, 126],
        "bubble_mask_source": "image_dark_bubble_mask",
        "route_action": "translate_inpaint_render",
    }

    filled, remaining, stats = _apply_fast_dark_panel_text_fill(image, {"texts": [text]}, [dict(text)])

    assert stats["dark_panel_fill_count"] == 1
    assert remaining == []
    changed = np.any(filled != image, axis=2)
    assert int(np.count_nonzero(changed)) > 0
    assert float(np.mean(filled[changed])) < 48.0


def test_image_dark_bubble_overbroad_fill_uses_visual_glyph_mask_and_preserves_outline():
    import cv2

    from inpainter import _apply_fast_dark_panel_text_fill

    image = np.zeros((180, 360, 3), dtype=np.uint8)
    cv2.ellipse(image, (180, 90), (150, 62), 0, 0, 360, (2, 2, 2), -1)
    cv2.ellipse(image, (180, 90), (150, 62), 0, 0, 360, (60, 130, 170), 3)
    cv2.putText(
        image,
        "Quest Completion",
        (90, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Criteria: Level 1",
        (92, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    text = {
        "id": "ocr_dark_overbroad",
        "trace_id": "ocr_dark_overbroad@band",
        "text": "Quest Completion Criteria: Level 1",
        "bbox": [70, 45, 285, 125],
        "text_pixel_bbox": [70, 45, 285, 125],
        "line_polygons": [[[70, 45], [285, 45], [285, 125], [70, 125]]],
        "balloon_bbox": [30, 28, 330, 152],
        "bubble_mask_bbox": [30, 28, 330, 152],
        "bubble_mask_source": "image_dark_bubble_mask",
        "mask_evidence": {
            "kind": "ocr_pixels",
            "raw_mask_pixels": 5000,
            "expanded_mask_pixels": 15000,
            "evidence_score": 1.0,
            "fast_fill_allowed": True,
        },
        "route_action": "translate_inpaint_render",
    }

    filled, remaining, stats = _apply_fast_dark_panel_text_fill(image, {"texts": [text]}, [dict(text)])

    changed = np.any(filled != image, axis=2)
    outline = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(outline, (180, 90), (150, 62), 0, 0, 360, 255, 3)
    metrics = text.get("qa_metrics") or {}
    override = metrics.get("dark_bubble_visual_fill_override") or {}

    assert stats["dark_panel_fill_count"] == 1
    assert remaining == []
    assert override.get("reason") == "overbroad_dark_fill_mask"
    assert int(np.count_nonzero(changed)) < int(override["current_pixels"])
    assert int(np.count_nonzero(changed & (outline > 0))) == 0


def test_dark_bubble_fast_fill_prefers_local_vision_geometry_for_visual_override():
    import cv2

    from inpainter import _apply_fast_dark_panel_text_fill

    image = np.zeros((180, 360, 3), dtype=np.uint8)
    cv2.ellipse(image, (180, 90), (150, 62), 0, 0, 360, (2, 2, 2), -1)
    cv2.ellipse(image, (180, 90), (150, 62), 0, 0, 360, (60, 130, 170), 3)
    cv2.putText(
        image,
        "Quest Completion",
        (90, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Criteria: Level 1",
        (92, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    page_text = {
        "id": "ocr_dark_page_space",
        "trace_id": "ocr_dark_page_space@band",
        "text": "Quest Completion Criteria: Level 1",
        "bbox": [70, 5045, 285, 5125],
        "text_pixel_bbox": [70, 5045, 285, 5125],
        "line_polygons": [[[70, 5045], [285, 5045], [285, 5125], [70, 5125]]],
        "balloon_bbox": [30, 5028, 330, 5152],
        "bubble_mask_bbox": [30, 5028, 330, 5152],
        "bubble_mask_source": "image_dark_bubble_mask",
        "route_action": "translate_inpaint_render",
    }
    local_block = dict(page_text)
    local_block.update(
        {
            "bbox": [70, 45, 285, 125],
            "text_pixel_bbox": [70, 45, 285, 125],
            "line_polygons": [[[70, 45], [285, 45], [285, 125], [70, 125]]],
            "balloon_bbox": [30, 28, 330, 152],
            "bubble_mask_bbox": [30, 28, 330, 152],
            "mask_evidence": {
                "kind": "ocr_pixels",
                "raw_mask_pixels": 5000,
                "expanded_mask_pixels": 15000,
                "evidence_score": 1.0,
                "fast_fill_allowed": True,
            },
        }
    )

    filled, remaining, stats = _apply_fast_dark_panel_text_fill(
        image,
        {"texts": [page_text]},
        [local_block],
    )

    changed = np.any(filled != image, axis=2)
    override = (page_text.get("qa_metrics") or {}).get("dark_bubble_visual_fill_override") or {}

    assert stats["dark_panel_fill_count"] == 1
    assert remaining == []
    assert override.get("reason") == "overbroad_dark_fill_mask"
    assert int(np.count_nonzero(changed)) < 10000


def test_dark_connected_bubble_visual_override_uses_compact_contract_bbox():
    import cv2

    from inpainter import _dark_bubble_visual_fill_override

    image = np.zeros((240, 360, 3), dtype=np.uint8)
    cv2.ellipse(image, (116, 104), (96, 68), 0, 0, 360, (1, 1, 1), -1)
    cv2.ellipse(image, (250, 104), (100, 68), 0, 0, 360, (1, 1, 1), -1)
    cv2.putText(image, "The subspace", (130, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, "retention is only", (130, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, "five minutes.", (130, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)
    current_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    current_mask[30:190, 40:334] = 255
    text = {
        "id": "direct_paddle_reocr_001",
        "bbox": [129, 64, 312, 156],
        "text_pixel_bbox": [129, 64, 312, 156],
        "bubble_mask_bbox": [28, 28, 206, 178],
        "balloon_bbox": [20, 20, 340, 190],
        "bubble_mask_source": "image_dark_bubble_mask",
        "qa_flags": [
            "visual_text_only_inpaint_contract",
            "dark_bubble_lobe_mask_bbox_preferred",
            "dark_connected_bubble_compact_bbox_replaced_aggregate_source",
            "fast_fill_no_glyph_evidence",
        ],
        "qa_metrics": {
            "inpaint_mask_contract_text_bbox_replaced_aggregate_source": {
                "bbox": [122, 57, 319, 163],
                "bbox_pixels": 20882,
            }
        },
    }

    override = _dark_bubble_visual_fill_override(image, text, current_mask)

    assert override is not None
    assert int(np.count_nonzero(override)) < int(np.count_nonzero(current_mask) * 0.60)
    assert int(np.count_nonzero(override[57:163, 122:319])) > 18000
    assert int(np.count_nonzero(override[:, 327:])) == 0
    metrics = text.get("qa_metrics") or {}
    assert (metrics.get("dark_bubble_visual_fill_override") or {}).get("source") == "compact_text_bbox_contract"


def test_rejected_dark_crop_uses_local_image_context_without_background_hint():
    from inpainter import _apply_fast_dark_panel_text_fill

    image = np.full((160, 360, 3), 9, dtype=np.uint8)
    image[46:58, 112:248] = 236
    image[74:86, 98:260] = 236
    text = {
        "id": "ocr_rejected_dark",
        "bbox": [92, 42, 266, 90],
        "text_pixel_bbox": [92, 42, 266, 90],
        "line_polygons": [
            [[112, 46], [248, 46], [248, 58], [112, 58]],
            [[98, 74], [260, 74], [260, 86], [98, 86]],
        ],
        "balloon_bbox": [52, 18, 320, 134],
        "bubble_mask_bbox": [52, 18, 320, 134],
        "bubble_mask_source": "derived_white_crop_rejected",
        "route_action": "translate_inpaint_render",
    }

    with patch.dict(os.environ, {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}):
        filled, remaining, stats = _apply_fast_dark_panel_text_fill(image, {"texts": [text]}, [dict(text)])

    assert stats["dark_panel_fill_count"] == 1
    assert remaining == []
    changed = np.any(filled != image, axis=2)
    assert int(np.count_nonzero(changed)) > 0
    assert float(np.mean(filled[changed])) < 48.0


def test_rejected_dark_crop_with_no_glyph_flag_gets_enough_padding():
    from inpainter import _try_dark_panel_text_fill

    image = np.full((160, 360, 3), 7, dtype=np.uint8)
    image[34:44, 120:234] = 240
    image[82:92, 108:250] = 240
    text = {
        "id": "ocr_rejected_dark_no_glyph",
        "bbox": [120, 54, 236, 72],
        "text_pixel_bbox": [120, 54, 236, 72],
        "balloon_bbox": [54, 18, 318, 132],
        "bubble_mask_bbox": [54, 18, 318, 132],
        "bubble_mask_source": "derived_white_crop_rejected",
        "route_action": "translate_inpaint_render",
        "qa_flags": ["fast_fill_no_glyph_evidence"],
    }

    filled = _try_dark_panel_text_fill(image, text)

    assert filled is not None
    assert float(np.mean(filled[34:44, 120:234])) < 48.0
    assert float(np.mean(filled[82:92, 108:250])) < 48.0


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
