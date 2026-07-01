import numpy as np
import cv2
from copy import deepcopy
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inpainter.mask_builder import (
    bbox_to_octagon_mask,
    build_raw_text_mask_from_image,
    build_inpaint_mask,
    build_mask_regions,
    expand_light_halo_mask_for_dark_card,
    expand_text_mask,
    expand_text_mask_monotonic,
    mask_from_text_geometry,
    polygon_to_mask,
)
from inpainter import (
    _authorized_dark_panel_padding,
    _clip_dark_bubble_fill_mask_to_lobe,
    _expand_strip_real_inpaint_mask,
    _filter_unsafe_auto_inpaint_blocks,
    _sanitize_precomputed_remaining_mask,
    _sanitize_rebuilt_expanded_mask,
    _try_dark_panel_text_fill,
)
from vision_stack.cjk_segmentation_mask import build_manhwa_manhua_roi_segmentation_mask
from vision_stack.runtime import vision_blocks_to_mask


def test_polygon_to_mask_fills_polygon_in_page_space():
    mask = polygon_to_mask([[10, 10], [50, 10], [50, 40], [10, 40]], (80, 90, 3))

    assert mask[20, 20] == 255
    assert mask[5, 20] == 0


def test_build_inpaint_mask_clips_text_to_balloon_interior():
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    block = {
        "bbox": [10, 10, 110, 70],
        "text_pixel_bbox": [42, 32, 78, 45],
        "balloon_polygon": [[10, 10], [110, 10], [110, 70], [10, 70]],
        "balloon_type": "white",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert mask[36, 50] == 255
    assert mask[10, 50] == 0
    assert mask[5, 50] == 0


def test_dark_bubble_inpaint_uses_expanded_glyph_mask_not_safe_box():
    image = np.full((130, 220, 3), 0, dtype=np.uint8)
    cv2.ellipse(image, (110, 65), (92, 48), 0, 0, 360, (0, 0, 0), -1)
    cv2.ellipse(image, (110, 65), (92, 48), 0, 0, 360, (20, 120, 150), 2)
    cv2.putText(image, "POINTS", (70, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    bubble_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(bubble_mask, (110, 65), (92, 48), 0, 0, 360, 255, -1)
    block = {
        "bbox": [18, 17, 202, 113],
        "text_pixel_bbox": [60, 42, 158, 84],
        "safe_text_box": [34, 28, 186, 101],
        "bubble_mask": bubble_mask,
        "bubble_mask_bbox": [18, 17, 202, 113],
        "bubble_mask_source": "image_dark_bubble_mask",
        "layout_profile": "dark_bubble",
        "background_rgb": [0, 0, 0],
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    glyph_area = int(np.count_nonzero(mask))
    bubble_area = int(np.count_nonzero(bubble_mask))
    safe_area = (186 - 34) * (101 - 28)
    assert glyph_area > 100
    assert glyph_area < int(bubble_area * 0.35)
    assert glyph_area < int(safe_area * 0.55)
    assert block["qa_metrics"]["inpaint_mask_contract"]["contract"] == "expanded_text_mask"
    assert block["qa_metrics"]["inpaint_mask_contract"]["area_fallback_used"] is False


def test_dark_bubble_inpaint_replaces_undercovered_line_polygon_with_ocr_bbox_contract():
    image = np.zeros((180, 360, 3), dtype=np.uint8)
    cv2.ellipse(image, (180, 90), (150, 70), 0, 0, 360, (0, 0, 0), -1)
    cv2.ellipse(image, (180, 90), (150, 70), 0, 0, 360, (16, 120, 150), 2)
    cv2.putText(image, "THAT MEANS", (60, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, "YOU ARE TALENTED", (44, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    bubble_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(bubble_mask, (180, 90), (150, 70), 0, 0, 360, 1, -1)
    block = {
        "id": "partial_dark_ocr",
        "text": "That means you're talented!",
        "bbox": [44, 48, 318, 122],
        "text_pixel_bbox": [44, 48, 318, 122],
        "source_text_mask_bbox": [44, 48, 318, 122],
        "bubble_mask": bubble_mask,
        "bubble_mask_bbox": [30, 20, 330, 160],
        "bubble_mask_source": "image_dark_bubble_mask",
        "layout_profile": "dark_bubble",
        "qa_flags": ["visual_text_only_inpaint_contract"],
        "line_polygons": [
            [[222, 88], [318, 88], [318, 122], [222, 122]],
        ],
        "background_rgb": [0, 0, 0],
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    bbox = np.argwhere(mask > 0)
    y1, x1 = bbox.min(axis=0)
    y2, x2 = bbox.max(axis=0) + 1
    assert x1 <= 60
    assert x2 >= 300
    assert mask[68, 76] == 255
    assert mask[105, 292] == 255
    metrics = block.get("qa_metrics", {})
    assert metrics["inpaint_mask_contract"]["source"] == "ocr_bbox_undercoverage_contract"
    assert "text_mask_undercovered_ocr_text" in block.get("qa_flags", [])


def test_dark_connected_bubble_inpaint_uses_text_contract_when_lobe_mask_is_tiny():
    image = np.zeros((700, 800, 3), dtype=np.uint8)
    broken_lobe_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    broken_lobe_mask[537:569, 494:633] = 255
    block = {
        "id": "direct_paddle_reocr_001",
        "text": "You must quickly build a new underworld!",
        "bbox": [448, 443, 681, 571],
        "source_bbox": [448, 443, 681, 571],
        "text_pixel_bbox": [448, 443, 681, 571],
        "balloon_bbox": [397, 405, 732, 609],
        "bubble_mask": broken_lobe_mask,
        "bubble_mask_bbox": [494, 537, 633, 569],
        "bubble_mask_source": "image_dark_bubble_mask",
        "layout_profile": "dark_bubble",
        "block_profile": "dark_bubble",
        "qa_flags": [
            "dark_bubble_lobe_mask_bbox_preferred",
            "dark_bubble_visual_glyph_mask_replaced_geometry",
            "visual_text_only_inpaint_contract",
        ],
        "line_polygons": [
            [[580, 548], [681, 548], [681, 571], [580, 571]],
        ],
        "background_rgb": [0, 0, 0],
    }

    with patch.dict("os.environ", {"TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE": "0"}):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    bbox = np.argwhere(mask > 0)
    y1, x1 = bbox.min(axis=0)
    y2, x2 = bbox.max(axis=0) + 1
    assert x1 <= 452
    assert y1 <= 447
    assert x2 >= 677
    assert y2 >= 567
    assert int(np.count_nonzero(mask)) > 25000
    flags = block.get("qa_flags", [])
    assert "dark_connected_missing_glyph_bbox_contract" in flags
    assert "dark_connected_bubble_mask_undercovered_text_contract" not in flags


def test_dark_connected_mask_region_seed_uses_text_bbox_when_line_polygon_is_partial():
    text = {
        "id": "direct_paddle_reocr_001",
        "text": "You must quickly build a new underworld!",
        "bbox": [448, 443, 681, 571],
        "source_bbox": [448, 443, 681, 571],
        "text_pixel_bbox": [448, 443, 681, 571],
        "balloon_bbox": [397, 405, 732, 609],
        "bubble_mask_bbox": [397, 405, 732, 609],
        "bubble_mask_source": "image_dark_bubble_mask",
        "layout_profile": "dark_bubble",
        "block_profile": "dark_bubble",
        "qa_flags": [
            "dark_bubble_lobe_mask_bbox_preferred",
            "dark_bubble_visual_glyph_mask_replaced_geometry",
        ],
        "line_polygons": [
            [[580, 548], [681, 548], [681, 571], [580, 571]],
        ],
    }

    regions = build_mask_regions([text], (700, 800, 3))

    assert len(regions) == 1
    bbox = regions[0]["bbox"]
    assert bbox[0] <= 452
    assert bbox[1] <= 447
    assert bbox[2] >= 677
    assert bbox[3] >= 567
    assert "dark_connected_mask_region_seed_bbox_replaced" in text.get("qa_flags", [])


def test_vision_blocks_dark_connected_floor_ignores_undercovered_bubble_clip():
    image = np.zeros((700, 800, 3), dtype=np.uint8)
    local_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    local_mask[548:571, 580:681] = 255
    broken_lobe_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    broken_lobe_mask[537:569, 494:633] = 255
    block = {
        "id": "direct_paddle_reocr_001",
        "text": "You must quickly build a new underworld!",
        "bbox": [448, 443, 681, 571],
        "text_pixel_bbox": [448, 443, 681, 571],
        "balloon_bbox": [397, 405, 732, 609],
        "bubble_mask": broken_lobe_mask,
        "bubble_mask_bbox": [494, 537, 633, 569],
        "bubble_mask_source": "image_dark_bubble_mask",
        "layout_profile": "dark_bubble",
        "qa_flags": [
            "candidate_crop_direct_paddle_reocr",
            "dark_bubble_lobe_mask_bbox_preferred",
        ],
        "line_polygons": [
            [[580, 548], [681, 548], [681, 571], [580, 571]],
        ],
    }

    mask = vision_blocks_to_mask(image.shape, [block], image_rgb=image)

    assert mask is not None
    bbox = np.argwhere(mask > 0)
    y1, x1 = bbox.min(axis=0)
    y2, x2 = bbox.max(axis=0) + 1
    assert x1 <= 448
    assert y1 <= 443
    assert x2 >= 681
    assert y2 >= 571
    assert "dark_bubble_recovered_text_bbox_floor" in block.get("qa_flags", [])


def test_dark_bubble_lobe_clip_rejects_bbox_that_misses_text_contract():
    mask = np.zeros((700, 800), dtype=np.uint8)
    mask[435:580, 440:690] = 255
    text = {
        "bbox": [448, 443, 681, 571],
        "text_pixel_bbox": [448, 443, 681, 571],
        "balloon_bbox": [397, 405, 732, 609],
        "bubble_mask_bbox": [494, 537, 633, 569],
        "bubble_mask_source": "image_dark_bubble_mask",
        "qa_flags": ["dark_bubble_lobe_mask_bbox_preferred"],
    }

    clipped = _clip_dark_bubble_fill_mask_to_lobe(text, mask, mask.shape)

    assert int(np.count_nonzero(clipped)) == int(np.count_nonzero(mask))
    assert "dark_bubble_lobe_clip_rejected_undercovered_text" in text.get("qa_flags", [])


def test_dark_panel_inpaint_without_glyph_source_does_not_fill_bbox():
    image = np.full((100, 180, 3), 0, dtype=np.uint8)
    block = {
        "bbox": [20, 20, 160, 78],
        "text_pixel_bbox": [40, 34, 140, 60],
        "safe_text_box": [28, 26, 152, 72],
        "bubble_mask_bbox": [20, 20, 160, 78],
        "bubble_mask_source": "image_dark_panel_mask",
        "layout_profile": "dark_panel",
        "background_rgb": [0, 0, 0],
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is None
    assert block["qa_metrics"]["inpaint_mask_contract"]["rejected_reason"] == "missing_text_glyph_source"


def test_text_geometry_uses_line_polygons_before_full_bbox():
    block = {
        "bbox": [10, 10, 100, 70],
        "line_polygons": [[[30, 30], [70, 30], [70, 40], [30, 40]]],
    }

    mask = mask_from_text_geometry(block, (90, 120, 3))

    assert mask is not None
    assert mask[35, 45] == 255
    assert mask[65, 95] == 0


def test_bbox_fallback_masks_as_octagon_not_full_rectangle():
    mask = bbox_to_octagon_mask(120, 90, [20, 20, 100, 70])

    assert mask[45, 60] == 255
    assert mask[20, 60] == 255
    assert mask[45, 20] == 255
    assert mask[20, 20] == 0
    assert mask[69, 99] == 0


def test_text_geometry_bbox_fallback_uses_octagon_shape():
    block = {
        "bbox": [10, 10, 110, 80],
        "text_pixel_bbox": [25, 25, 95, 60],
        "font_size_px": 16,
    }

    mask = mask_from_text_geometry(block, (100, 130, 3))

    assert mask is not None
    assert mask[42, 60] == 255
    assert mask[22, 60] == 255
    assert mask[22, 22] == 0


def test_expand_text_mask_monotonic_preserves_raw_pixels_outside_limit():
    raw = np.zeros((40, 80), dtype=np.uint8)
    raw[18:22, 10:18] = 255
    raw[18:22, 58:66] = 255
    limit = np.zeros_like(raw)
    limit[10:30, 0:40] = 255

    expanded = expand_text_mask_monotonic(raw, limit_mask=limit, radius=4)

    assert expanded[20, 12] == 255
    assert expanded[20, 62] == 255
    assert int(np.count_nonzero(expanded)) >= int(np.count_nonzero(raw))
    assert expanded[20, 54] == 0


def test_strip_real_inpaint_limit_preserves_raw_floor_on_dark_card():
    raw = np.zeros((48, 90), dtype=np.uint8)
    raw[20:28, 16:74] = 255
    expanded = np.zeros_like(raw)
    expanded[22:26, 34:56] = 255
    image = np.full((48, 90, 3), 18, dtype=np.uint8)
    ocr_page = {}
    texts = [
        {
            "bbox": [34, 22, 56, 26],
            "text_pixel_bbox": [34, 22, 56, 26],
            "line_polygons": [[[34, 22], [56, 22], [56, 26], [34, 26]]],
            "background_rgb": [18, 18, 18],
            "layout_profile": "dark_panel",
        }
    ]

    repaired = _expand_strip_real_inpaint_mask(raw, expanded, ocr_page, texts, image)

    assert int(np.count_nonzero(repaired)) >= int(np.count_nonzero(raw))
    assert np.all(repaired[raw > 0] == 255)
    assert ocr_page["_strip_raw_floor_preserved_after_limit"]["raw_pixels"] == int(np.count_nonzero(raw))


def test_strip_real_inpaint_limit_preserves_raw_floor_when_bubble_clip_preserved_raw_text():
    image = np.full((80, 140, 3), 255, dtype=np.uint8)
    raw = np.zeros((80, 140), dtype=np.uint8)
    raw[20:32, 44:86] = 255
    expanded = raw.copy()
    expanded[32:40, 48:82] = 255
    bubble_mask = np.zeros((80, 140), dtype=np.uint8)
    bubble_mask[4:70, 10:132] = 1
    text = {
        "bbox": [44, 28, 86, 42],
        "text_pixel_bbox": [44, 28, 86, 42],
        "line_polygons": [[[44, 28], [86, 28], [86, 42], [44, 42]]],
        "bubble_mask": bubble_mask,
        "bubble_mask_source": "image_white_bubble_mask",
        "qa_flags": ["bubble_clip_preserved_raw_text"],
    }

    mask = _expand_strip_real_inpaint_mask(raw, expanded, {"texts": [text]}, [text], image)

    assert np.all(mask[raw > 0] == 255)
    assert mask[22, 52] == 255


def test_strip_real_inpaint_preserves_valid_expanded_pixels_inside_real_bubble():
    image = np.full((90, 150, 3), 255, dtype=np.uint8)
    raw = np.zeros((90, 150), dtype=np.uint8)
    raw[42:50, 44:76] = 255
    raw[42:50, 96:118] = 255
    expanded = raw.copy()
    expanded[28:42, 42:82] = 255
    expanded[28:42, 92:122] = 255
    bubble_mask = np.zeros((90, 150), dtype=np.uint8)
    cv2.ellipse(bubble_mask, (76, 45), (58, 34), 0, 0, 360, 1, -1)
    text = {
        "bubble_id": "bubble_001",
        "bbox": [42, 38, 122, 56],
        "source_bbox": [42, 38, 122, 56],
        "text_pixel_bbox": [44, 42, 118, 50],
        "line_polygons": [[[42, 38], [122, 38], [122, 56], [42, 56]]],
        "rotation_deg": 16.0,
        "bubble_mask": bubble_mask,
        "bubble_mask_bbox": [18, 11, 134, 79],
        "bubble_mask_source": "image_white_bubble_mask",
    }

    mask = _expand_strip_real_inpaint_mask(raw, expanded, {"texts": [text]}, [text], image)

    assert np.all(mask[raw > 0] == 255)
    assert np.all(mask[(expanded > 0) & (bubble_mask > 0)] == 255)


def test_strip_real_inpaint_ignores_rejected_fragment_without_glyph_evidence():
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    raw = np.zeros((120, 180), dtype=np.uint8)
    raw[62:96, 8:158] = 255
    valid = np.zeros_like(raw)
    valid[24:52, 108:160] = 255
    raw = np.maximum(raw, valid)
    card_mask = np.zeros_like(raw)
    card_mask[14:66, 100:170] = 1
    valid_text = {
        "bubble_id": "bubble_card",
        "bbox": [108, 24, 160, 52],
        "text_pixel_bbox": [108, 24, 160, 52],
        "source_bbox": [106, 22, 162, 54],
        "line_polygons": [[[108, 24], [160, 24], [160, 52], [108, 52]]],
        "bubble_mask": card_mask,
        "bubble_mask_bbox": [100, 14, 170, 66],
        "bubble_mask_source": "image_contour_bubble_mask",
    }
    rejected_fragment = {
        "bubble_id": "bad_fragment",
        "bbox": [8, 62, 158, 96],
        "text_pixel_bbox": [8, 62, 158, 96],
        "source_bbox": [8, 62, 158, 96],
        "balloon_bbox": [0, 56, 180, 110],
        "bubble_mask_source": "derived_white_crop_rejected",
        "bubble_mask_error": "derived_mask_not_anchored_to_text",
        "qa_flags": [],
    }

    mask = _expand_strip_real_inpaint_mask(
        raw,
        raw,
        {"texts": [valid_text, rejected_fragment]},
        [valid_text, rejected_fragment],
        image,
    )

    assert mask[34, 130] == 255
    assert mask[74, 40] == 0
    assert mask[90, 150] == 0


def test_sanitize_precomputed_remaining_mask_rebuilds_after_rejected_text_filter():
    precomputed = np.zeros((80, 140), dtype=np.uint8)
    precomputed[8:70, 4:130] = 255
    rebuilt = np.zeros_like(precomputed)
    rebuilt[18:42, 88:124] = 255

    sanitized = _sanitize_precomputed_remaining_mask(
        precomputed,
        rebuilt,
        original_text_count=2,
        filtered_text_count=1,
    )

    assert sanitized[28, 100] == 255
    assert sanitized[62, 20] == 0


def test_sanitize_rebuilt_expanded_mask_preserves_plausible_expanded_pixels():
    raw = np.zeros((80, 140), dtype=np.uint8)
    raw[34:46, 54:78] = 255
    rebuilt_expanded = raw.copy()
    rebuilt_expanded[24:34, 50:84] = 255
    rebuilt_expanded[34:48, 78:96] = 255

    sanitized = _sanitize_rebuilt_expanded_mask(raw, rebuilt_expanded, raw)

    assert np.all(sanitized[raw > 0] == 255)
    assert sanitized[26, 60] == 255
    assert sanitized[40, 88] == 255


def test_sanitize_rebuilt_expanded_mask_rejects_overbroad_expansion():
    raw = np.zeros((80, 140), dtype=np.uint8)
    raw[34:46, 54:78] = 255
    rebuilt_expanded = np.zeros_like(raw)
    rebuilt_expanded[4:72, 8:132] = 255

    sanitized = _sanitize_rebuilt_expanded_mask(raw, rebuilt_expanded, raw)

    assert np.all(sanitized[raw > 0] == 255)
    assert sanitized[10, 10] == 0


def test_raw_text_mask_uses_line_polygon_as_search_area_not_filled_rectangle():
    image = np.full((90, 130, 3), 248, dtype=np.uint8)
    image[34:50, 34:40] = 18
    image[34:50, 86:92] = 18
    block = {
        "bbox": [20, 25, 110, 62],
        "line_polygons": [[[20, 25], [110, 25], [110, 62], [20, 62]]],
    }

    mask = build_raw_text_mask_from_image(block, image, image.shape)

    assert mask is not None
    assert mask[40, 36] == 255
    assert mask[40, 88] == 255
    assert mask[40, 62] == 0
    assert mask[26, 22] == 0


def test_build_inpaint_mask_filters_art_components_outside_text_geometry():
    image = np.full((140, 180, 3), 248, dtype=np.uint8)
    image[28:40, 22:38] = [230, 16, 16]
    image[48:60, 42:58] = [230, 16, 16]
    image[90:100, 112:132] = 18
    block = {
        "bbox": [10, 20, 150, 112],
        "source_bbox": [10, 20, 150, 112],
        "text_pixel_bbox": [100, 82, 146, 106],
        "line_polygons": [[[100, 82], [146, 82], [146, 106], [100, 106]]],
        "allow_broad_bbox_text_search": True,
        "font_size": 18,
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert mask[94, 120] == 255
    assert mask[34, 30] == 0
    assert mask[54, 50] == 0
    metrics = block.get("qa_metrics", {}).get("raw_text_component_filter", {})
    assert metrics.get("filtered_pixels", 0) < metrics.get("source_pixels", 0)


def test_inpaint_mask_unions_raw_pixels_with_line_geometry_for_sparse_glyphs():
    image = np.full((90, 130, 3), 248, dtype=np.uint8)
    image[34:50, 34:40] = 18
    block = {
        "bbox": [20, 25, 110, 62],
        "line_polygons": [[[20, 25], [110, 25], [110, 62], [20, 62]]],
        "balloon_polygon": [[10, 10], [120, 10], [120, 80], [10, 80]],
        "balloon_type": "white",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert mask[40, 36] == 255
    assert mask[40, 62] == 0
    assert mask[12, 12] == 0


def test_strip_real_inpaint_expansion_clips_broad_raw_mask_to_text_geometry():
    from inpainter import _expand_strip_real_inpaint_mask

    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    raw = np.zeros((100, 160), dtype=np.uint8)
    raw[20:72, 20:142] = 255
    bubble_mask = np.zeros((100, 160), dtype=np.uint8)
    bubble_mask[10:90, 10:150] = 1
    text = {
        "bubble_id": "bubble_001",
        "bbox": [20, 20, 142, 72],
        "source_bbox": [20, 20, 142, 72],
        "text_pixel_bbox": [60, 42, 104, 54],
        "line_polygons": [[[20, 20], [142, 20], [142, 72], [20, 72]]],
        "bubble_mask": bubble_mask,
        "bubble_mask_bbox": [10, 10, 150, 90],
    }

    mask = _expand_strip_real_inpaint_mask(raw, raw, {"texts": [text]}, [text], image)

    assert mask[46, 68] == 255
    assert mask[24, 24] == 0
    assert mask[68, 136] == 0


def test_strip_real_inpaint_expansion_recovers_moderate_line_extent_around_tight_text_pixels():
    from inpainter import _expand_strip_real_inpaint_mask

    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    raw = np.zeros((100, 160), dtype=np.uint8)
    raw[52:60, 24:34] = 255
    raw[52:60, 82:94] = 255
    bubble_mask = np.zeros((100, 160), dtype=np.uint8)
    bubble_mask[10:90, 10:150] = 1
    text = {
        "bubble_id": "bubble_001",
        "bbox": [20, 40, 100, 70],
        "source_bbox": [20, 40, 100, 70],
        "text_pixel_bbox": [42, 48, 76, 62],
        "line_polygons": [[[22, 44], [98, 44], [98, 68], [22, 68]]],
        "bubble_mask": bubble_mask,
        "bubble_mask_bbox": [10, 10, 150, 90],
    }

    mask = _expand_strip_real_inpaint_mask(raw, raw, {"texts": [text]}, [text], image)

    assert mask[55, 28] == 255
    assert mask[55, 88] == 255
    assert mask[24, 24] == 0


def test_strip_real_inpaint_expansion_recovers_raw_glyph_components_inside_ocr_support():
    from inpainter import _expand_strip_real_inpaint_mask

    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    raw = np.zeros((100, 160), dtype=np.uint8)
    raw[52:60, 14:24] = 255
    raw[52:60, 48:60] = 255
    raw[52:60, 98:108] = 255
    expanded = np.zeros((100, 160), dtype=np.uint8)
    expanded[52:60, 48:60] = 255
    bubble_mask = np.zeros((100, 160), dtype=np.uint8)
    bubble_mask[10:90, 10:150] = 1
    text = {
        "bubble_id": "bubble_001",
        "bbox": [12, 40, 110, 70],
        "source_bbox": [12, 40, 110, 70],
        "text_pixel_bbox": [42, 48, 76, 62],
        "line_polygons": [[[12, 44], [110, 44], [110, 68], [12, 68]]],
        "bubble_mask": bubble_mask,
        "bubble_mask_bbox": [10, 10, 150, 90],
    }

    mask = _expand_strip_real_inpaint_mask(raw, expanded, {"texts": [text]}, [text], image)

    assert mask[55, 18] == 255
    assert mask[55, 54] == 255
    assert mask[55, 102] == 255
    assert mask[24, 24] == 0


def test_strip_real_inpaint_expansion_recovers_raw_glyph_components_just_outside_ocr_support():
    from inpainter import _expand_strip_real_inpaint_mask

    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    raw = np.zeros((100, 180), dtype=np.uint8)
    raw[48:62, 22:30] = 255
    raw[48:62, 52:72] = 255
    raw[48:62, 118:130] = 255
    expanded = np.zeros((100, 180), dtype=np.uint8)
    expanded[48:62, 52:72] = 255
    bubble_mask = np.zeros((100, 180), dtype=np.uint8)
    bubble_mask[10:90, 10:170] = 1
    text = {
        "bubble_id": "bubble_001",
        "bbox": [32, 38, 128, 70],
        "source_bbox": [32, 38, 128, 70],
        "text_pixel_bbox": [52, 46, 76, 62],
        "line_polygons": [[[32, 42], [128, 42], [128, 68], [32, 68]]],
        "bubble_mask": bubble_mask,
        "bubble_mask_bbox": [10, 10, 170, 90],
    }

    mask = _expand_strip_real_inpaint_mask(raw, expanded, {"texts": [text]}, [text], image)

    assert mask[54, 26] == 255
    assert mask[54, 62] == 255
    assert mask[54, 124] == 255
    assert mask[24, 24] == 0


def test_strip_real_inpaint_expansion_recovers_raw_glyphs_when_bubble_mask_is_tight():
    from inpainter import _expand_strip_real_inpaint_mask

    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    raw = np.zeros((100, 180), dtype=np.uint8)
    raw[48:62, 24:34] = 255
    raw[48:62, 52:72] = 255
    raw[48:62, 130:140] = 255
    expanded = np.zeros((100, 180), dtype=np.uint8)
    expanded[48:62, 52:72] = 255
    tight_bubble_mask = np.zeros((100, 180), dtype=np.uint8)
    tight_bubble_mask[24:76, 44:126] = 1
    text = {
        "bubble_id": "bubble_001",
        "bbox": [30, 38, 136, 70],
        "source_bbox": [30, 38, 136, 70],
        "text_pixel_bbox": [52, 46, 76, 62],
        "line_polygons": [[[30, 42], [136, 42], [136, 68], [30, 68]]],
        "bubble_mask": tight_bubble_mask,
        "bubble_mask_bbox": [10, 10, 170, 90],
        "balloon_bbox": [10, 10, 170, 90],
    }

    mask = _expand_strip_real_inpaint_mask(raw, expanded, {"texts": [text]}, [text], image)

    assert mask[54, 28] == 255
    assert mask[54, 62] == 255
    assert mask[54, 134] == 255
    assert mask[24, 24] == 0


def test_strip_real_inpaint_expansion_preserves_bubble_outline_band():
    from inpainter import _expand_strip_real_inpaint_mask

    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    raw = np.zeros((100, 160), dtype=np.uint8)
    raw[44:54, 12:20] = 255
    raw[44:54, 42:58] = 255
    bubble_mask = np.zeros((100, 160), dtype=np.uint8)
    bubble_mask[10:90, 10:150] = 1
    text = {
        "bubble_id": "bubble_001",
        "bbox": [10, 34, 80, 64],
        "source_bbox": [10, 34, 80, 64],
        "text_pixel_bbox": [38, 42, 62, 56],
        "line_polygons": [[[12, 38], [82, 38], [82, 62], [12, 62]]],
        "bubble_mask": bubble_mask,
        "bubble_mask_bbox": [10, 10, 150, 90],
    }

    mask = _expand_strip_real_inpaint_mask(raw, raw, {"texts": [text]}, [text], image)

    assert mask[48, 14] == 0
    assert mask[48, 48] == 255


def test_raw_text_mask_does_not_use_broad_bbox_search_when_text_geometry_exists():
    image = np.full((120, 180, 3), 248, dtype=np.uint8)
    image[24:38, 58:64] = 18
    image[78:92, 108:114] = 18
    block = {
        "bbox": [30, 15, 150, 100],
        "line_polygons": [[[96, 72], [138, 72], [138, 98], [96, 98]]],
    }

    mask = build_raw_text_mask_from_image(block, image, image.shape)

    assert mask is not None
    assert mask[30, 60] == 0
    assert mask[84, 110] == 255
    assert mask[56, 90] == 0


def test_raw_text_mask_can_opt_into_broad_bbox_search_for_partial_lines():
    image = np.full((120, 180, 3), 248, dtype=np.uint8)
    image[24:38, 58:64] = 18
    image[78:92, 108:114] = 18
    block = {
        "bbox": [30, 15, 150, 100],
        "line_polygons": [[[96, 72], [138, 72], [138, 98], [96, 98]]],
        "allow_broad_bbox_text_search": True,
    }

    mask = build_raw_text_mask_from_image(block, image, image.shape)

    assert mask is not None
    assert mask[30, 60] == 255
    assert mask[84, 110] == 255


def test_expand_text_mask_uses_configured_five_pixel_radius():
    raw = np.zeros((40, 40), dtype=np.uint8)
    raw[20, 20] = 255

    expanded = expand_text_mask(raw, expand_px=5)

    assert expanded[20, 20] == 255
    assert expanded[20, 25] == 255
    assert expanded[25, 20] == 255
    assert expanded[20, 26] == 0
    assert expanded[26, 20] == 0


def test_expanded_text_mask_never_loses_raw_text_pixels_inside_limit():
    raw = np.zeros((300, 800), dtype=np.uint8)
    raw[103:168, 453:561] = 255
    limit = np.zeros((300, 800), dtype=np.uint8)
    limit[102:169, 450:565] = 255

    expanded = expand_text_mask_monotonic(raw, limit_mask=limit, radius=2)

    assert int(np.count_nonzero((expanded > 0) & (raw > 0))) == int(np.count_nonzero(raw))
    assert int(np.count_nonzero(expanded)) >= int(np.count_nonzero(raw))


def test_build_inpaint_mask_preserves_raw_pixels_near_white_bubble_edge():
    image = np.full((110, 180, 3), 255, dtype=np.uint8)
    raw = np.zeros((110, 180), dtype=np.uint8)
    raw[18:28, 132:154] = 255
    raw[58:70, 64:112] = 255
    bubble = np.zeros((110, 180), dtype=np.uint8)
    cv2.ellipse(bubble, (90, 55), (70, 42), 0, 0, 360, 255, -1)
    block = {
        "bbox": [58, 16, 156, 72],
        "text_pixel_bbox": [58, 16, 156, 72],
        "line_polygons": [
            [[132, 18], [154, 18], [154, 28], [132, 28]],
            [[64, 58], [112, 58], [112, 70], [64, 70]],
        ],
        "bubble_mask": bubble,
        "bubble_mask_source": "image_white_bubble_mask",
        "layout_profile": "white_balloon",
        "font_size": 16,
    }

    with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    raw_inside_bubble = (raw > 0) & (bubble > 0)
    assert int(np.count_nonzero(raw_inside_bubble & (mask == 0))) == 0
    assert block.get("qa_metrics", {}).get("balloon_clip_raw_preservation") is not None


def test_light_halo_expansion_includes_near_glow_on_dark_card_only():
    image = np.full((64, 96, 3), 25, dtype=np.uint8)
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[28:36, 42:54] = 255
    halo = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
    image[halo > 0] = [135, 122, 66]
    image[mask > 0] = [255, 255, 255]

    expanded = expand_light_halo_mask_for_dark_card(
        mask,
        image,
        {"background_rgb": [30, 25, 15], "block_profile": "standard"},
        radius=7,
    )
    white_context = expand_light_halo_mask_for_dark_card(
        mask,
        np.full_like(image, 255),
        {"background_rgb": [255, 255, 255], "block_profile": "white_balloon"},
        radius=7,
    )

    assert int(np.count_nonzero(expanded)) > int(np.count_nonzero(mask))
    assert int(np.count_nonzero(white_context)) == int(np.count_nonzero(mask))


def test_light_halo_expansion_handles_dark_colored_title_effect_on_black_card():
    image = np.zeros((84, 180, 3), dtype=np.uint8)
    mask = np.zeros((84, 180), dtype=np.uint8)
    mask[36:48, 62:118] = 255
    halo = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    image[halo > 0] = [16, 54, 74]
    image[mask > 0] = [48, 76, 90]

    expanded = expand_light_halo_mask_for_dark_card(mask, image, {}, radius=8)

    assert int(np.count_nonzero(expanded)) > int(np.count_nonzero(mask))
    assert int(np.count_nonzero((expanded > 0) & (halo > 0) & (mask == 0))) > 0


def test_light_halo_expansion_infers_colored_card_without_metadata():
    image = np.full((80, 140, 3), [32, 42, 86], dtype=np.uint8)
    mask = np.zeros((80, 140), dtype=np.uint8)
    mask[32:40, 58:82] = 255
    halo = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1)
    image[halo > 0] = [150, 120, 210]
    image[mask > 0] = [245, 245, 255]

    expanded = expand_light_halo_mask_for_dark_card(mask, image, {}, radius=8)

    assert int(np.count_nonzero(expanded)) > int(np.count_nonzero(mask))


def test_light_halo_expansion_handles_colored_light_text_on_dark_card():
    image = np.full((80, 160, 3), [8, 10, 14], dtype=np.uint8)
    mask = np.zeros((80, 160), dtype=np.uint8)
    mask[34:46, 62:98] = 255
    halo = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1)
    image[halo > 0] = [135, 70, 88]
    image[mask > 0] = [245, 100, 120]

    expanded = expand_light_halo_mask_for_dark_card(
        mask,
        image,
        {"background_rgb": [8, 10, 14], "block_profile": "dark_panel"},
        radius=8,
    )

    assert int(np.count_nonzero(expanded)) > int(np.count_nonzero(mask))
    assert int(np.count_nonzero((expanded > 0) & (halo > 0) & (mask == 0))) > 0


def test_colored_card_fill_authorizes_near_text_glow_outside_line_geometry():
    image = np.full((96, 180, 3), [253, 209, 156], dtype=np.uint8)
    text_mask = np.zeros((96, 180), dtype=np.uint8)
    text_mask[38:52, 64:116] = 255
    glow = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
    image[glow > 0] = [250, 242, 230]
    image[text_mask > 0] = [255, 255, 255]
    block = {
        "bbox": [64, 38, 116, 52],
        "text_pixel_bbox": [64, 38, 116, 52],
        "line_polygons": [[[64, 38], [116, 38], [116, 52], [64, 52]]],
        "target_bbox": [42, 20, 138, 70],
        "balloon_bbox": [42, 20, 138, 70],
        "background_rgb": [253, 209, 156],
        "route_action": "translate_inpaint_render",
    }

    filled = _try_dark_panel_text_fill(image, block)

    assert filled is not None
    changed = np.any(filled != image, axis=2)
    assert int(np.count_nonzero(changed & (glow > 0) & (text_mask == 0))) > 0


def test_colored_textured_card_fill_uses_glyph_mask_not_line_rectangle():
    image = np.full((104, 208, 3), [253, 209, 156], dtype=np.uint8)
    for y in range(18, 86):
        image[y, 34:174, 0] = np.clip(244 + ((y - 18) % 9), 0, 255)
        image[y, 34:174, 1] = np.clip(196 + ((y - 18) % 7), 0, 255)
        image[y, 34:174, 2] = np.clip(150 + ((y - 18) % 5), 0, 255)

    line_rect = np.zeros((104, 208), dtype=np.uint8)
    line_rect[36:68, 58:150] = 255
    glyph = np.zeros((104, 208), dtype=np.uint8)
    glyph[40:47, 70:140] = 255
    glyph[56:63, 82:128] = 255
    halo = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    image[halo > 0] = [246, 231, 215]
    image[glyph > 0] = [255, 255, 255]

    block = {
        "bbox": [58, 36, 150, 68],
        "text_pixel_bbox": [58, 36, 150, 68],
        "line_polygons": [[[58, 36], [150, 36], [150, 68], [58, 68]]],
        "target_bbox": [34, 18, 174, 86],
        "balloon_bbox": [34, 18, 174, 86],
        "background_rgb": [253, 209, 156],
        "layout_profile": "card",
        "route_action": "translate_inpaint_render",
    }

    filled = _try_dark_panel_text_fill(image, block)

    assert filled is not None
    fill_mask = block.get("_dark_panel_fill_mask")
    assert isinstance(fill_mask, np.ndarray)
    line_area = int(np.count_nonzero(line_rect))
    fill_area = int(np.count_nonzero(fill_mask))
    assert fill_area < int(line_area * 1.05)
    assert fill_area > int(np.count_nonzero(glyph))
    changed = np.any(filled != image, axis=2)
    non_glyph_line = (line_rect > 0) & (halo == 0)
    assert int(np.count_nonzero(changed & non_glyph_line)) < int(np.count_nonzero(non_glyph_line) * 0.25)


def test_colored_card_inpaint_mask_refines_stale_background_to_visual_glyphs():
    image = np.full((118, 220, 3), [225, 145, 120], dtype=np.uint8)
    for y in range(22, 96):
        image[y, 32:188, 0] = np.clip(215 + ((y * 3) % 34), 0, 255)
        image[y, 32:188, 1] = np.clip(126 + ((y * 5) % 46), 0, 255)
        image[y, 32:188, 2] = np.clip(118 + ((y * 7) % 38), 0, 255)
    line_rect = np.zeros((118, 220), dtype=np.uint8)
    line_rect[42:78, 58:164] = 255
    glyph = np.zeros((118, 220), dtype=np.uint8)
    cv2.putText(glyph, "HOST", (70, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, 255, 2, cv2.LINE_AA)
    cv2.putText(glyph, "TITLE", (68, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, 255, 2, cv2.LINE_AA)
    halo = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    image[halo > 0] = [242, 235, 224]
    image[glyph > 0] = [255, 255, 255]
    block = {
        "bbox": [58, 42, 164, 78],
        "text_pixel_bbox": [58, 42, 164, 78],
        "line_polygons": [[[58, 42], [164, 42], [164, 78], [58, 78]]],
        "target_bbox": [32, 22, 188, 96],
        "balloon_bbox": [32, 22, 188, 96],
        "background_rgb": [253, 209, 156],
        "bubble_mask_source": "derived_card_panel_mask",
        "route_action": "translate_inpaint_render",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert int(np.count_nonzero(mask)) < int(np.count_nonzero(line_rect) * 0.85)
    assert int(np.count_nonzero((mask > 0) & (glyph > 0))) > 0
    assert block.get("qa_metrics", {}).get("colored_card_visual_glyph_mask_refined") is not None


def test_colored_card_padding_uses_background_chroma_even_without_status_terms():
    padding = _authorized_dark_panel_padding(
        {"background_rgb": [253, 209, 156], "route_action": "translate_inpaint_render"},
        has_line_geometry=True,
        detected_dark_panel=False,
        colored_status_panel=False,
        base_padding=1,
    )

    assert padding >= 5


def test_colored_card_without_bubble_mask_uses_panel_bbox_as_limit():
    image = np.full((140, 240, 3), [42, 38, 56], dtype=np.uint8)
    cv2.rectangle(image, (64, 42), (176, 96), (238, 160, 112), -1)
    text_mask = np.zeros((140, 240), dtype=np.uint8)
    text_mask[60:72, 94:146] = 255
    halo = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=1)
    image[halo > 0] = [180, 90, 112]
    image[text_mask > 0] = [250, 95, 120]
    block = {
        "bbox": [94, 60, 146, 72],
        "text_pixel_bbox": [94, 60, 146, 72],
        "line_polygons": [[[94, 60], [146, 60], [146, 72], [94, 72]]],
        "balloon_bbox": [64, 42, 176, 96],
        "background_rgb": [238, 160, 112],
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert int(np.count_nonzero(mask)) > int(np.count_nonzero(text_mask))
    assert mask[50, 70] == 0
    assert mask[66, 120] == 255
    panel_bbox = block["qa_metrics"]["derived_card_panel_mask"]["mask_bbox"]
    assert panel_bbox[0] <= 64
    assert panel_bbox[1] <= 42
    assert panel_bbox[2] >= 176
    assert panel_bbox[3] >= 96
    assert panel_bbox[0] >= 58
    assert panel_bbox[1] >= 36
    assert panel_bbox[2] <= 182
    assert panel_bbox[3] <= 102


def test_dark_panel_negative_geometry_detects_panel_without_using_inverted_colors():
    image = np.full((452, 800, 3), [2, 2, 2], dtype=np.uint8)
    cv2.rectangle(image, (61, 122), (443, 356), (3, 3, 3), -1)
    cv2.rectangle(image, (61, 122), (443, 356), (160, 155, 142), 2)
    glow = np.zeros((452, 800), dtype=np.uint8)
    cv2.rectangle(glow, (16, 67), (488, 378), 255, -1)
    cv2.rectangle(glow, (57, 118), (447, 360), 0, -1)
    image[glow > 0] = [75, 61, 24]
    cv2.rectangle(image, (61, 122), (443, 356), (3, 3, 3), -1)
    cv2.rectangle(image, (61, 122), (443, 356), (160, 155, 142), 2)
    glyph = np.zeros((452, 800), dtype=np.uint8)
    cv2.putText(glyph, "Quest Introduction:", (118, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.78, 255, 2, cv2.LINE_AA)
    cv2.putText(glyph, "As the new King", (118, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.78, 255, 2, cv2.LINE_AA)
    cv2.putText(glyph, "Yeomra!", (118, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.78, 255, 2, cv2.LINE_AA)
    halo = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    image[halo > 0] = [83, 71, 39]
    image[glyph > 0] = [252, 250, 236]
    block = {
        "text": "Quest Introduction: As the new King Yeomra!",
        "translated": "Introducao da missao: como o novo Rei Yeomra!",
        "bbox": [118, 186, 390, 304],
        "text_pixel_bbox": [118, 186, 390, 304],
        "line_polygons": [
            [[118, 188], [390, 188], [390, 220], [118, 220]],
            [[118, 228], [350, 228], [350, 260], [118, 260]],
            [[118, 268], [230, 268], [230, 300], [118, 300]],
        ],
        "balloon_bbox": [118, 186, 390, 304],
        "bubble_mask_source": "rejected_derived_bubble_mask",
        "bubble_mask_error": "derived_mask_not_anchored_to_text",
        "background_rgb": [2, 2, 2],
        "card_panel_text_context": True,
        "route_action": "translate_inpaint_render",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert block["bubble_mask_source"] == "image_dark_panel_mask"
    assert block["bubble_mask_bbox"][0] <= 64
    assert block["bubble_mask_bbox"][1] <= 124
    assert block["bubble_mask_bbox"][2] >= 440
    assert block["bubble_mask_bbox"][3] >= 354
    metrics = block.get("qa_metrics", {}).get("image_dark_panel_mask")
    assert metrics is not None
    assert metrics["detection_space"] == "negative_geometry"
    assert metrics["color_sample_space"] == "original_image"
    assert metrics["panel_fill_rgb"] == [3, 3, 3]
    assert metrics["text_fill_rgb"][0] >= 245
    assert metrics["text_glow_rgb"][0] > metrics["panel_glow_rgb"][0]
    assert metrics["bad_negative_text_glow_rgb"][2] > metrics["text_glow_rgb"][2]


def test_dark_panel_negative_geometry_rejects_overbroad_art_bbox():
    image = np.full((360, 800, 3), [2, 2, 2], dtype=np.uint8)
    cv2.rectangle(image, (37, 122), (687, 332), (4, 4, 4), -1)
    cv2.rectangle(image, (37, 122), (687, 332), (118, 116, 116), 2)
    cv2.rectangle(image, (40, 124), (330, 270), (3, 3, 3), -1)
    cv2.rectangle(image, (40, 124), (330, 270), (150, 148, 148), 2)
    image[205:332, 285:687] = [38, 44, 64]
    cv2.rectangle(image, (37, 122), (687, 332), (118, 116, 116), 2)
    cv2.rectangle(image, (40, 124), (330, 270), (150, 148, 148), 2)
    glyph = np.zeros((360, 800), dtype=np.uint8)
    cv2.putText(glyph, "Main Quest will", (108, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.72, 255, 2, cv2.LINE_AA)
    cv2.putText(glyph, "be shown shortly.", (108, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.72, 255, 2, cv2.LINE_AA)
    halo = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    image[halo > 0] = [52, 40, 12]
    image[glyph > 0] = [253, 249, 230]
    block = {
        "text": "Main Quest will be shown shortly.",
        "translated": "A missao principal sera mostrada em breve.",
        "bbox": [108, 165, 260, 236],
        "text_pixel_bbox": [108, 165, 260, 236],
        "line_polygons": [
            [[121, 165], [251, 165], [251, 191], [121, 191]],
            [[108, 204], [260, 204], [260, 236], [108, 236]],
        ],
        "balloon_bbox": [75, 144, 293, 257],
        "bubble_mask_source": "rejected_derived_bubble_mask",
        "bubble_mask_error": "derived_mask_not_anchored_to_text",
        "background_rgb": [2, 2, 2],
        "card_panel_text_context": True,
        "route_action": "translate_inpaint_render",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert block["bubble_mask_source"] != "image_dark_panel_mask"
    assert block["bubble_mask_bbox"][2] < 430
    rejected = block.get("qa_metrics", {}).get("image_dark_panel_mask_rejected")
    assert rejected
    assert rejected[-1]["reason"] == "overbroad_against_balloon_bbox"
    assert rejected[-1]["mask_bbox"][2] >= 680


def test_dark_visual_page_cleanup_does_not_paint_art_inside_overbroad_dark_bubble():
    from strip.run import _apply_dark_visual_text_geometry_cleanup

    image = np.full((170, 230, 3), [34, 30, 28], dtype=np.uint8)
    image[54:78, 58:112] = [245, 242, 224]
    image[128:142, 176:196] = [188, 176, 164]
    text = {
        "text_pixel_bbox": [58, 54, 112, 78],
        "source_bbox": [58, 54, 112, 78],
        "line_polygons": [[[58, 54], [112, 54], [112, 78], [58, 78]]],
        "bubble_mask_source": "image_dark_bubble_mask",
        "bubble_mask_bbox": [18, 30, 212, 152],
        "card_panel_text_context": True,
        "dark_panel_effect_colors": {"panel_fill_rgb": [8, 3, 2]},
    }

    result, changed = _apply_dark_visual_text_geometry_cleanup(image, [text])

    assert changed > 0
    assert np.array_equal(result[135, 185], image[135, 185])
    assert np.array_equal(result[62, 82], np.asarray([8, 3, 2], dtype=np.uint8))


def test_dark_oval_bubble_negative_geometry_detects_ellipse_shape():
    image = np.full((300, 800, 3), [2, 2, 2], dtype=np.uint8)
    center = (225, 140)
    axes = (195, 120)
    glow = np.zeros((300, 800), dtype=np.uint8)
    cv2.ellipse(glow, center, (205, 128), 0, 0, 360, 255, 8, cv2.LINE_AA)
    image[glow > 0] = [83, 71, 39]
    cv2.ellipse(image, center, axes, 0, 0, 360, (3, 3, 3), -1, cv2.LINE_AA)
    cv2.ellipse(image, center, axes, 0, 0, 360, (75, 61, 24), 3, cv2.LINE_AA)
    glyph = np.zeros((300, 800), dtype=np.uint8)
    cv2.putText(glyph, "The host", (156, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.82, 255, 2, cv2.LINE_AA)
    cv2.putText(glyph, "is not authorized", (116, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.82, 255, 2, cv2.LINE_AA)
    cv2.putText(glyph, "to know.", (172, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.82, 255, 2, cv2.LINE_AA)
    halo = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    image[halo > 0] = [83, 71, 39]
    image[glyph > 0] = [252, 250, 236]
    block = {
        "text": "The host is not authorized to know.",
        "translated": "O anfitriao nao esta autorizado a saber.",
        "bbox": [116, 94, 350, 204],
        "text_pixel_bbox": [116, 94, 350, 204],
        "line_polygons": [
            [[156, 92], [294, 92], [294, 124], [156, 124]],
            [[116, 132], [350, 132], [350, 164], [116, 164]],
            [[172, 172], [280, 172], [280, 204], [172, 204]],
        ],
        "balloon_bbox": [30, 20, 420, 260],
        "bubble_mask_source": "derived_white_crop_rejected",
        "bubble_mask_error": "derived_mask_not_anchored_to_text",
        "background_rgb": [2, 2, 2],
        "card_panel_text_context": True,
        "route_action": "translate_inpaint_render",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert block["bubble_mask_source"] == "image_dark_bubble_mask"
    metrics = block.get("qa_metrics", {}).get("image_dark_bubble_mask")
    assert metrics is not None
    assert metrics["shape_kind"] == "ellipse"
    assert metrics["detection_space"] == "negative_geometry"
    assert metrics["color_sample_space"] == "original_image"
    assert metrics["mask_bbox"][0] <= 40
    assert metrics["mask_bbox"][1] <= 28
    assert metrics["mask_bbox"][2] >= 410
    assert metrics["mask_bbox"][3] >= 252
    assert mask[140, 225] == 255
    assert mask[20, 720] == 0
    assert metrics["text_fill_rgb"][0] >= 245
    assert metrics["text_glow_rgb"][0] > metrics["panel_glow_rgb"][0]


def test_colored_card_inpaint_mask_uses_visual_glyphs_instead_of_text_rectangle():
    image = np.full((104, 208, 3), [253, 209, 156], dtype=np.uint8)
    for y in range(18, 86):
        image[y, 34:174, 0] = np.clip(244 + ((y - 18) % 9), 0, 255)
        image[y, 34:174, 1] = np.clip(196 + ((y - 18) % 7), 0, 255)
        image[y, 34:174, 2] = np.clip(150 + ((y - 18) % 5), 0, 255)
    line_rect = np.zeros((104, 208), dtype=np.uint8)
    line_rect[36:68, 58:150] = 255
    glyph = np.zeros((104, 208), dtype=np.uint8)
    glyph[40:47, 70:140] = 255
    glyph[56:63, 82:128] = 255
    halo = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    image[halo > 0] = [246, 231, 215]
    image[glyph > 0] = [255, 255, 255]
    block = {
        "bbox": [58, 36, 150, 68],
        "text_pixel_bbox": [58, 36, 150, 68],
        "line_polygons": [[[58, 36], [150, 36], [150, 68], [58, 68]]],
        "balloon_bbox": [34, 18, 174, 86],
        "background_rgb": [253, 209, 156],
        "layout_profile": "card",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    line_area = int(np.count_nonzero(line_rect))
    mask_area = int(np.count_nonzero(mask))
    assert mask_area < int(line_area * 1.35)
    assert mask_area > int(np.count_nonzero(glyph))
    assert int(np.count_nonzero((mask > 0) & (line_rect > 0) & (halo == 0))) < int(line_area * 0.30)


def test_colored_card_mask_allows_small_panel_edge_growth_for_bottom_halo():
    image = np.full((120, 220, 3), [253, 209, 156], dtype=np.uint8)
    text_mask = np.zeros((120, 220), dtype=np.uint8)
    text_mask[68:86, 82:142] = 255
    image[text_mask > 0] = [255, 255, 255]
    image[95:100, 92:138] = [245, 242, 235]
    block = {
        "bbox": [82, 68, 142, 86],
        "text_pixel_bbox": [82, 68, 142, 86],
        "line_polygons": [[[82, 68], [142, 68], [142, 86], [82, 86]]],
        "balloon_bbox": [64, 34, 176, 94],
        "background_rgb": [253, 209, 156],
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert mask[96, 110] == 255
    assert block["qa_metrics"]["derived_card_panel_mask"]["mask_bbox"][3] > 94


def test_non_card_large_art_bbox_does_not_become_panel_limit():
    image = np.full((180, 220, 3), [118, 132, 158], dtype=np.uint8)
    text_mask = np.zeros((180, 220), dtype=np.uint8)
    text_mask[130:146, 48:168] = 255
    image[text_mask > 0] = [245, 245, 245]
    block = {
        "bbox": [48, 130, 168, 146],
        "text_pixel_bbox": [48, 130, 168, 146],
        "line_polygons": [[[48, 130], [168, 130], [168, 146], [48, 146]]],
        "balloon_bbox": [0, 20, 220, 170],
        "background_rgb": [118, 132, 158],
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert block.get("qa_metrics", {}).get("derived_card_panel_mask") is None
    assert int(np.count_nonzero(mask)) < 10000


def test_merged_fragment_without_glyph_evidence_does_not_build_inpaint_mask():
    image = np.full((90, 140, 3), 255, dtype=np.uint8)
    block = {
        "bbox": [10, 45, 130, 75],
        "text_pixel_bbox": [10, 45, 130, 75],
        "line_polygons": [[[10, 45], [130, 45], [130, 75], [10, 75]]],
        "qa_flags": [
            "same_balloon_fragment_merged",
            "raw_text_evidence_missing",
            "fast_fill_no_glyph_evidence",
        ],
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is None
    assert block["mask_evidence"]["kind"] == "none"


def test_merged_fragment_without_glyph_evidence_does_not_widen_mask_region_cluster():
    texts = [
        {
            "id": "ocr_001",
            "bbox": [90, 30, 130, 55],
            "line_polygons": [[[90, 30], [130, 30], [130, 55], [90, 55]]],
        },
        {
            "id": "ocr_002",
            "bbox": [10, 60, 132, 80],
            "line_polygons": [],
            "qa_flags": [
                "same_balloon_fragment_merged",
                "raw_text_evidence_missing",
                "fast_fill_no_glyph_evidence",
            ],
        },
    ]

    regions = build_mask_regions(texts, (100, 160, 3))

    assert len(regions) == 1
    assert regions[0]["bbox"][0] >= 80
    assert regions[0]["bbox"][2] <= 140
    assert [text["id"] for text in regions[0]["texts"]] == ["ocr_001"]


def test_build_inpaint_mask_prefers_irregular_text_pixels_when_image_available():
    image = np.full((100, 140, 3), 248, dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (122, 82), (255, 255, 255), -1)
    image[40:52, 34:40] = 16
    image[40:52, 90:96] = 16
    block = {
        "bbox": [20, 25, 120, 70],
        "line_polygons": [[[20, 25], [120, 25], [120, 70], [20, 70]]],
        "balloon_polygon": [[18, 18], [122, 18], [122, 82], [18, 82]],
        "balloon_type": "white",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert mask[46, 36] == 255
    assert mask[46, 92] == 255
    assert mask[18, 70] == 0


def test_build_inpaint_mask_uses_white_balloon_geometry_floor_when_raw_is_fragmented():
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    raw = np.zeros((120, 180), dtype=np.uint8)
    raw[52:62, 86:126] = 255
    block = {
        "bbox": [30, 34, 150, 88],
        "text_pixel_bbox": [30, 34, 150, 88],
        "line_polygons": [
            [[70, 38], [130, 38], [130, 48], [70, 48]],
            [[50, 55], [145, 55], [145, 65], [50, 65]],
            [[76, 72], [124, 72], [124, 82], [76, 82]],
        ],
        "balloon_bbox": [20, 20, 160, 104],
        "balloon_type": "white",
        "layout_profile": "white_balloon",
    }

    with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert mask[43, 100] == 255
    assert mask[58, 100] == 255
    assert mask[77, 100] == 255
    assert "white_balloon_geometry_floor" in block.get("qa_metrics", {}) or (
        "white_balloon_geometry_floor_after_component_cleaner" in block.get("qa_metrics", {})
    )


def test_build_inpaint_mask_discards_outline_pixels_outside_line_geometry():
    image = np.full((120, 180, 3), 248, dtype=np.uint8)
    image[56:70, 96:102] = 14
    image[56:70, 128:134] = 14
    image[30:39, 28:38] = 14
    block = {
        "bbox": [18, 18, 158, 96],
        "text_pixel_bbox": [18, 18, 158, 96],
        "line_polygons": [[[82, 48], [148, 48], [148, 78], [82, 78]]],
        "balloon_polygon": [[10, 10], [170, 10], [170, 108], [10, 108]],
        "balloon_type": "white",
        "detected_font_size_px": 18,
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert mask[62, 99] == 255
    assert mask[62, 131] == 255
    assert mask[34, 33] == 0


def test_build_inpaint_mask_rejects_large_no_line_art_component():
    image = np.full((220, 320, 3), 236, dtype=np.uint8)
    image[28:38, 90:230] = 18
    image[50:60, 98:222] = 18
    cv2.ellipse(image, (164, 148), (108, 58), 0, 0, 360, (70, 70, 70), -1)
    block = {
        "bbox": [20, 10, 300, 205],
        "text_pixel_bbox": [20, 10, 300, 205],
        "balloon_bbox": [0, 0, 320, 220],
        "balloon_type": "white",
        "layout_profile": "white_balloon",
        "content_class": "narration",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is None
    assert set(block.get("qa_flags", [])) & {"raw_text_evidence_missing", "raw_text_evidence_rejected"}


def test_segment_evidence_validates_source_boxes_without_art():
    image = np.full((300, 400, 3), 242, dtype=np.uint8)
    for y in (42, 72):
        for x in range(72, 326, 18):
            image[y : y + 11, x : x + 7] = 16
    cv2.ellipse(image, (210, 198), (118, 58), 0, 0, 360, (66, 66, 66), -1)
    block = {
        "bbox": [20, 20, 380, 290],
        "text_pixel_bbox": [20, 20, 380, 290],
        "_merged_source_bboxes": [[40, 20, 360, 120], [20, 80, 380, 290]],
        "balloon_bbox": [0, 0, 400, 300],
        "balloon_type": "white",
        "layout_profile": "white_balloon",
        "content_class": "narration",
    }

    mask = build_raw_text_mask_from_image(block, image, image.shape)

    assert mask is not None
    assert block["_validated_text_source_bboxes"] == [[40, 20, 360, 120]]
    assert block["_rejected_text_source_bboxes"] == [[20, 80, 380, 290]]
    assert block["validated_by_segment_mask"] is True
    assert block["_raw_text_evidence_pixels"] > 0
    assert block["_raw_text_evidence_bbox"][1] < 120
    assert mask[46, 74] == 255
    assert mask[198, 210] == 0


def test_vision_blocks_to_mask_does_not_fallback_to_bbox_after_raw_evidence_rejection():
    image = np.full((220, 320, 3), 236, dtype=np.uint8)
    image[28:38, 90:230] = 18
    cv2.ellipse(image, (164, 148), (108, 58), 0, 0, 360, (70, 70, 70), -1)
    block = {
        "bbox": [20, 10, 300, 205],
        "text_pixel_bbox": [20, 10, 300, 205],
        "balloon_bbox": [0, 0, 320, 220],
        "balloon_type": "white",
        "layout_profile": "white_balloon",
        "content_class": "narration",
    }

    mask = vision_blocks_to_mask(image.shape, [deepcopy(block)], image_rgb=image, expand_mask=False)

    assert int(np.count_nonzero(mask)) == 0


def test_build_inpaint_mask_does_not_clip_to_tiny_sanitized_balloon_bbox():
    image = np.full((130, 180, 3), 248, dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (162, 112), (255, 255, 255), -1)
    image[42:56, 52:58] = 16
    image[82:96, 108:114] = 16
    block = {
        "bbox": [20, 20, 160, 110],
        "text_pixel_bbox": [96, 76, 150, 104],
        "balloon_bbox": [96, 76, 150, 104],
        "line_polygons": [[[96, 76], [150, 76], [150, 104], [96, 104]]],
        "balloon_type": "white",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert mask[49, 55] == 0
    assert mask[89, 111] == 255
    assert mask[20, 90] == 0


def test_raw_text_mask_detects_light_text_on_mid_light_background():
    image = np.full((90, 130, 3), 40, dtype=np.uint8)
    image[30:64, 44:86] = [218, 190, 128]
    image[40:54, 58:72] = 252
    block = {
        "bbox": [40, 28, 90, 66],
        "line_polygons": [[[40, 28], [90, 28], [90, 66], [40, 66]]],
    }

    mask = build_raw_text_mask_from_image(block, image, image.shape)

    assert mask is not None
    assert mask[46, 64] == 255
    assert mask[35, 50] == 0
    assert mask[62, 88] == 0


def test_raw_text_mask_searches_just_outside_italic_ocr_polygon():
    image = np.full((90, 140, 3), 245, dtype=np.uint8)
    image[36:50, 94:99] = 10
    block = {
        "line_polygons": [[[40, 30], [90, 30], [90, 55], [40, 55]]],
        "estilo": {"italico": True, "tamanho": 48},
        "layout_profile": "top_narration",
    }

    mask = build_raw_text_mask_from_image(block, image, image.shape)

    assert mask is not None
    assert mask[42, 96] == 255


def test_raw_text_mask_expands_tight_white_balloon_line_polygon_for_punctuation():
    image = np.full((90, 180, 3), 255, dtype=np.uint8)
    image[38:50, 52:56] = 18
    image[38:50, 64:68] = 18
    image[46:50, 52:82] = 18
    image[42:46, 91:95] = 18
    block = {
        "bbox": [40, 28, 120, 62],
        "text_pixel_bbox": [50, 36, 96, 52],
        "line_polygons": [[[50, 36], [86, 36], [86, 52], [50, 52]]],
        "balloon_polygon": [[20, 15], [160, 15], [160, 75], [20, 75]],
        "balloon_type": "white",
        "layout_profile": "white_balloon",
    }

    mask = build_raw_text_mask_from_image(block, image, image.shape)

    assert mask is not None
    assert mask[44, 54] == 255
    assert mask[44, 92] == 255
    assert mask[20, 22] == 0


def test_inpaint_mask_expands_tight_white_balloon_line_polygon_for_punctuation():
    image = np.full((90, 180, 3), 255, dtype=np.uint8)
    image[38:50, 52:56] = 18
    image[38:50, 64:68] = 18
    image[46:50, 52:82] = 18
    image[42:46, 91:95] = 18
    block = {
        "bbox": [40, 28, 120, 62],
        "text_pixel_bbox": [50, 36, 96, 52],
        "line_polygons": [[[50, 36], [86, 36], [86, 52], [50, 52]]],
        "balloon_polygon": [[20, 15], [160, 15], [160, 75], [20, 75]],
        "balloon_type": "white",
        "layout_profile": "white_balloon",
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert mask[44, 54] == 255
    assert mask[44, 92] == 255
    assert mask[20, 22] == 0


def test_cjk_roi_segmentation_mask_remaps_crop_pixels_to_page_space():
    image = np.full((240, 180, 3), 248, dtype=np.uint8)
    block = {"bbox": [40, 80, 100, 130], "text_pixel_bbox": [55, 95, 84, 112]}

    def segmenter(crop):
        local = np.zeros(crop.shape[:2], dtype=np.uint8)
        local[50:62, 48:60] = 255
        return local

    mask = build_manhwa_manhua_roi_segmentation_mask(
        image,
        [block],
        [block],
        segmenter=segmenter,
    )

    assert int(np.count_nonzero(mask)) > 0
    assert mask[90, 40] == 0


def test_fast_fill_only_changes_masked_pixels_inside_same_bubble_id():
    from inpainter import apply_koharu_bubble_fast_fill

    image = np.full((64, 96, 3), 240, dtype=np.uint8)
    image[28:36, 32:64] = 10
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[28:36, 32:64] = 255
    bubble_mask = np.zeros((64, 96), dtype=np.uint8)
    bubble_mask[12:52, 16:80] = 3

    result, remaining, metadata = apply_koharu_bubble_fast_fill(image, mask, bubble_mask)

    assert metadata["filled_pixels"] == int(np.count_nonzero(mask))
    assert int(np.count_nonzero(remaining)) == 0
    assert np.array_equal(result[0:12, :, :], image[0:12, :, :])
    assert np.array_equal(result[:, 0:16, :], image[:, 0:16, :])
    assert np.all(result[28:36, 32:64] == 240)


def test_fast_fill_leaves_bubble_outline_touching_mask_for_aot():
    from inpainter import apply_koharu_bubble_fast_fill

    image = np.full((64, 96, 3), 240, dtype=np.uint8)
    image[12:16, 32:64] = 10
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[12:16, 32:64] = 255
    bubble_mask = np.zeros((64, 96), dtype=np.uint8)
    bubble_mask[12:52, 16:80] = 3

    result, remaining, metadata = apply_koharu_bubble_fast_fill(image, mask, bubble_mask)

    assert metadata["filled_pixels"] == 0
    assert metadata["reason"] == "mask_touches_bubble_outline"
    assert int(np.count_nonzero(remaining)) == int(np.count_nonzero(mask))
    assert np.array_equal(result, image)


def test_fast_fill_leaves_mask_for_aot_when_bubble_mask_is_missing():
    from inpainter import apply_koharu_bubble_fast_fill

    image = np.full((32, 40, 3), 220, dtype=np.uint8)
    mask = np.zeros((32, 40), dtype=np.uint8)
    mask[10:16, 12:22] = 255

    result, remaining, metadata = apply_koharu_bubble_fast_fill(image, mask, None)

    assert metadata["filled_pixels"] == 0
    assert metadata["reason"] == "missing_bubble_mask"
    assert np.array_equal(result, image)
    assert np.array_equal(remaining, mask)


def test_fast_fill_leaves_mask_for_aot_when_bubble_background_varies():
    from inpainter import apply_koharu_bubble_fast_fill

    image = np.full((48, 72, 3), 220, dtype=np.uint8)
    image[:, ::2] = 180
    image[20:28, 24:48] = 12
    mask = np.zeros((48, 72), dtype=np.uint8)
    mask[20:28, 24:48] = 255
    bubble_mask = np.zeros((48, 72), dtype=np.uint8)
    bubble_mask[8:40, 12:60] = 2

    result, remaining, metadata = apply_koharu_bubble_fast_fill(image, mask, bubble_mask)

    assert metadata["filled_pixels"] == 0
    assert metadata["reason"] == "background_variation_high"
    assert np.array_equal(result, image)
    assert np.array_equal(remaining, mask)


def test_inpaint_mask_rounds_diagonal_glyph_tip_and_respects_real_bubble_clip():
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    bubble_mask = np.zeros((80, 80), dtype=np.uint8)
    bubble_mask[2:78, 2:78] = 3
    raw = np.zeros((80, 80), dtype=np.uint8)
    raw[6, 40] = 255
    raw[7, 41] = 255
    raw[8, 42] = 255
    block = {
        "bbox": [4, 34, 72, 48],
        "text_pixel_bbox": [4, 34, 72, 48],
        "line_polygons": [
            [[4, 34], [72, 34], [72, 48], [4, 48]],
        ],
        "bubble_mask": bubble_mask,
        "bubble_id": 3,
        "font_size": 10,
    }

    with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert int(mask[7, 41]) == 255
    assert int(mask[8, 42]) == 255
    assert int(mask[9, 43]) == 255
    assert int(mask[3, 40]) == 0
    assert int(mask[77, 40]) == 0


def test_inpaint_mask_rounds_diagonal_glyph_tip_with_derived_bubble_clip():
    image = np.full((120, 180, 3), 24, dtype=np.uint8)
    cv2.rectangle(image, (22, 18), (150, 102), (255, 255, 255), -1)
    cv2.rectangle(image, (22, 18), (150, 102), (0, 0, 0), 2)
    raw = np.zeros((120, 180), dtype=np.uint8)
    raw[35, 52] = 255
    raw[36, 53] = 255
    raw[37, 54] = 255
    block = {
        "bbox": [50, 34, 120, 66],
        "text_pixel_bbox": [50, 34, 120, 66],
        "balloon_bbox": [20, 16, 152, 108],
        "balloon_type": "white",
        "layout_profile": "white_balloon",
        "font_size": 10,
    }

    with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert block.get("qa_metrics", {}).get("derived_white_bubble_mask", {}).get("source") == "derived_rectangular_balloon"
    assert int(mask[36, 53]) == 255
    assert int(mask[37, 54]) == 255
    assert int(mask[38, 55]) == 255
    assert int(mask[18, 52]) == 0
    assert int(mask[19, 52]) == 0


def test_build_inpaint_mask_uses_image_white_bubble_source_as_geometry_floor():
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    raw = np.zeros((120, 180), dtype=np.uint8)
    raw[38:48, 76:118] = 255
    block = {
        "bbox": [38, 28, 148, 90],
        "text_pixel_bbox": [38, 28, 148, 90],
        "line_polygons": [
            [[44, 30], [140, 30], [140, 42], [44, 42]],
            [[40, 50], [146, 50], [146, 62], [40, 62]],
            [[58, 72], [128, 72], [128, 84], [58, 84]],
        ],
        "bubble_mask_source": "image_white_bubble_mask",
        "font_size": 13,
    }

    with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert int(np.count_nonzero(mask)) > int(np.count_nonzero(raw)) * 2
    metrics = block.get("qa_metrics", {}).get("white_balloon_geometry_floor")
    assert metrics is not None
    assert metrics["coverage"] < 0.68


def test_build_inpaint_mask_keeps_colored_card_line_geometry_when_image_bubble_is_tiny():
    image = np.full((120, 220, 3), [50, 90, 180], dtype=np.uint8)
    raw = np.zeros((120, 220), dtype=np.uint8)
    tiny_bubble = np.zeros((120, 220), dtype=np.uint8)
    tiny_bubble[20:38, 150:170] = 255
    block = {
        "bbox": [72, 42, 150, 78],
        "text_pixel_bbox": [72, 42, 150, 78],
        "line_polygons": [
            [[76, 44], [146, 44], [146, 56], [76, 56]],
            [[82, 62], [140, 62], [140, 74], [82, 74]],
        ],
        "bubble_mask_source": "image_white_bubble_mask",
        "bubble_mask": tiny_bubble,
        "layout_profile": "colored_status_panel",
        "background_rgb": [50, 90, 180],
        "font_size": 14,
    }

    with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert int(np.count_nonzero(mask)) >= 1200
    assert block.get("qa_metrics", {}).get("colored_card_geometry_preserved") is not None


def test_colored_card_overrides_tight_explicit_image_bubble_with_panel_mask():
    image = np.full((140, 240, 3), [253, 194, 150], dtype=np.uint8)
    raw = np.zeros((140, 240), dtype=np.uint8)
    raw[54:88, 78:162] = 255
    tiny_bubble = np.zeros((140, 240), dtype=np.uint8)
    tiny_bubble[52:90, 76:160] = 255
    block = {
        "bbox": [76, 52, 160, 90],
        "source_bbox": [76, 52, 160, 90],
        "text_pixel_bbox": [76, 52, 160, 90],
        "target_bbox": [76, 52, 160, 90],
        "line_polygons": [
            [[78, 54], [158, 54], [158, 68], [78, 68]],
            [[82, 74], [154, 74], [154, 88], [82, 88]],
        ],
        "balloon_bbox": [42, 24, 198, 116],
        "bubble_mask_bbox": [76, 52, 160, 90],
        "bubble_mask_source": "image_white_bubble_mask",
        "bubble_mask": tiny_bubble,
        "background_rgb": [253, 194, 150],
        "font_size": 14,
    }

    with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert block["bubble_mask_source"] == "derived_card_panel_mask"
    assert block["bubble_mask_bbox"] == [36, 18, 204, 122]
    assert block.get("qa_metrics", {}).get("derived_card_panel_mask") is not None
    assert int(np.count_nonzero(mask)) >= int(np.count_nonzero(raw))
    assert int(np.count_nonzero(mask[:24, :])) == 0
    assert int(np.count_nonzero(mask[116:, :])) == 0
    assert int(np.count_nonzero(mask[:, :42])) == 0
    assert int(np.count_nonzero(mask[:, 198:])) == 0


def test_colored_card_rejects_explicit_bubble_mask_smaller_than_raw_text():
    image = np.full((140, 240, 3), [96, 128, 196], dtype=np.uint8)
    cv2.rectangle(image, (42, 24), (198, 116), (230, 238, 255), 2)
    raw = np.zeros((140, 240), dtype=np.uint8)
    raw[50:82, 70:170] = 255
    tiny_bubble = np.zeros((140, 240), dtype=np.uint8)
    tiny_bubble[60:78, 116:150] = 255
    block = {
        "bbox": [70, 50, 170, 82],
        "source_bbox": [70, 50, 170, 82],
        "text_pixel_bbox": [70, 50, 170, 82],
        "line_polygons": [
            [[70, 50], [170, 50], [170, 64], [70, 64]],
            [[78, 68], [164, 68], [164, 82], [78, 82]],
        ],
        "balloon_bbox": [42, 24, 198, 116],
        "bubble_mask_bbox": [116, 60, 150, 78],
        "bubble_mask_source": "image_white_bubble_mask",
        "bubble_mask": tiny_bubble,
        "font_size": 16,
    }

    with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert block["bubble_mask_source"] == "derived_card_panel_mask"
    assert block.get("qa_metrics", {}).get("explicit_bubble_mask_rejected_for_card_panel") is not None
    assert int(np.count_nonzero(mask)) > int(np.count_nonzero(raw))
    assert np.all(mask[raw > 0] > 0)


def test_dark_card_overrides_rejected_tight_bubble_bbox_with_panel_mask():
    image = np.full((256, 800, 3), [118, 98, 49], dtype=np.uint8)
    cv2.rectangle(image, (438, 81), (638, 194), (205, 205, 190), 2)
    raw = np.zeros((256, 800), dtype=np.uint8)
    raw[108:166, 452:642] = 255
    tiny_bubble = np.zeros((256, 800), dtype=np.uint8)
    tiny_bubble[139:169, 514:586] = 255
    block = {
        "bbox": [468, 102, 608, 173],
        "source_bbox": [468, 102, 608, 173],
        "text_pixel_bbox": [451, 107, 643, 168],
        "line_polygons": [
            [[451, 107], [643, 107], [643, 134], [451, 134]],
            [[468, 141], [608, 141], [608, 168], [468, 168]],
        ],
        "balloon_bbox": [438, 81, 638, 194],
        "bubble_mask_bbox": [514, 139, 586, 169],
        "bubble_mask_source": "derived_white_crop_rejected",
        "bubble_mask": tiny_bubble,
        "background_rgb": [118, 98, 49],
        "font_size": 18,
    }

    with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=raw):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert block["bubble_mask_source"] == "derived_card_panel_mask"
    assert block["bubble_mask_bbox"][0] <= 438
    assert block["bubble_mask_bbox"][1] <= 81
    assert block["bubble_mask_bbox"][2] >= 638
    assert block["bubble_mask_bbox"][3] >= 194
    assert block.get("qa_metrics", {}).get("derived_card_panel_mask") is not None
    assert int(np.count_nonzero(mask)) >= int(np.count_nonzero(raw))


def test_dark_card_uses_geometry_when_raw_text_evidence_is_rejected():
    image = np.full((256, 800, 3), [118, 98, 49], dtype=np.uint8)
    cv2.rectangle(image, (438, 81), (638, 194), (205, 205, 190), 2)
    block = {
        "bbox": [468, 102, 608, 173],
        "text_pixel_bbox": [451, 107, 643, 168],
        "line_polygons": [
            [[451, 107], [643, 107], [643, 134], [451, 134]],
            [[468, 141], [608, 141], [608, 168], [468, 168]],
        ],
        "balloon_bbox": [438, 81, 638, 194],
        "bubble_mask_bbox": [514, 139, 586, 169],
        "bubble_mask_source": "rejected_derived_bubble_mask",
        "background_rgb": [118, 98, 49],
        "font_size": 18,
        "qa_flags": ["raw_text_evidence_rejected", "missing_real_bubble_mask"],
    }

    with patch("inpainter.mask_builder.build_raw_text_mask_from_image", return_value=None):
        mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert int(np.count_nonzero(mask)) > 0
    assert block["bubble_mask_source"] == "derived_card_panel_mask"
    assert "raw_text_evidence_rejected" not in block.get("qa_flags", [])
    assert "missing_real_bubble_mask" not in block.get("qa_flags", [])
    assert block.get("qa_metrics", {}).get("raw_text_evidence_card_geometry_fallback") is not None


def test_unsafe_filter_rebuilds_when_reason_is_inherited_from_text():
    image = np.full((256, 800, 3), [118, 98, 49], dtype=np.uint8)
    cv2.rectangle(image, (438, 81), (638, 194), (205, 205, 190), 2)
    text = {
        "id": "ocr_001",
        "bbox": [468, 102, 608, 173],
        "text_pixel_bbox": [451, 107, 643, 168],
        "balloon_bbox": [438, 81, 638, 194],
        "bubble_mask_bbox": [514, 139, 586, 169],
        "bubble_mask_source": "derived_white_crop_rejected",
        "bubble_mask_error": "derived_mask_not_anchored_to_text",
        "qa_flags": ["bubble_clip_preserved_raw_text", "fast_fill_no_glyph_evidence"],
        "route_action": "translate_inpaint_render",
        "line_polygons": [
            [[451, 107], [643, 107], [643, 134], [451, 134]],
            [[468, 141], [608, 141], [608, 168], [468, 168]],
        ],
        "background_rgb": [118, 98, 49],
        "font_size": 18,
    }
    block = dict(text)
    block.update(
        {
            "bubble_mask_bbox": [438, 81, 652, 194],
            "bubble_mask_source": "derived_card_panel_mask",
            "bubble_mask_error": "derived_mask_not_anchored_to_text",
            "mask_evidence": {
                "kind": "none",
                "raw_mask_pixels": 0,
                "expanded_mask_pixels": 0,
                "evidence_score": 0.0,
            },
        }
    )
    ocr_page = {"texts": [text]}

    kept = _filter_unsafe_auto_inpaint_blocks(ocr_page, [block], image)

    assert kept == [block]
    assert ocr_page.get("_strip_unsafe_inpaint_block_count") in (None, 0)
    assert block.get("mask_evidence", {}).get("raw_mask_pixels", 0) > 0


def test_dark_panel_current_mask_clears_inherited_outside_balloon_critical():
    from inpainter import _auto_inpaint_unsafe_reason

    image = np.full((220, 840, 3), [5, 4, 3], dtype=np.uint8)
    cv2.rectangle(image, (563, 98), (797, 150), (8, 4, 2), -1)
    cv2.rectangle(image, (563, 98), (797, 150), (150, 141, 131), 2)
    text = {
        "id": "direct_paddle_reocr_001",
        "trace_id": "direct_paddle_reocr_001@page_002_band_004",
        "bbox": [585, 109, 767, 138],
        "text_pixel_bbox": [585, 109, 767, 138],
        "line_polygons": [[[585, 109], [767, 109], [767, 138], [585, 138]]],
        "balloon_bbox": [545, 97, 800, 150],
        "bubble_mask_bbox": [563, 98, 797, 150],
        "bubble_mask_source": "image_dark_panel_mask",
        "card_panel_text_context": True,
        "block_profile": "dark_panel",
        "background_rgb": [8, 4, 2],
        "route_action": "translate_inpaint_render",
        "qa_flags": ["candidate_crop_direct_paddle_reocr", "mask_outside_balloon_critical"],
        "mask_evidence": {
            "kind": "ocr_pixels",
            "raw_mask_pixels": 1317,
            "expanded_mask_pixels": 8809,
            "evidence_score": 1.0,
            "fast_fill_allowed": True,
        },
    }
    block = dict(text)
    block.update(
        {
            "bubble_mask_bbox": [392, 30, 800, 173],
            "bubble_mask_source": "image_dark_bubble_mask",
        }
    )
    ocr_page = {"texts": [text]}

    assert _auto_inpaint_unsafe_reason(text) == ""
    kept = _filter_unsafe_auto_inpaint_blocks(ocr_page, [block], image)

    assert kept == [block]
    assert block["bubble_mask_source"] == "image_dark_panel_mask"
    assert block["bubble_mask_bbox"] == [563, 98, 797, 150]
    assert ocr_page.get("_strip_unsafe_inpaint_block_count") in (None, 0)
