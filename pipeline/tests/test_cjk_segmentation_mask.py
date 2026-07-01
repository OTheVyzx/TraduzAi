import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_stack.cjk_segmentation_mask import (
    _recover_textlike_strokes_from_candidate,
    build_manga_segmentation_mask,
    build_manhwa_manhua_roi_segmentation_mask,
    expand_cjk_glyph_mask_for_inpaint,
)


def test_manga_segmentation_mask_uses_glyph_pixels_instead_of_full_bbox():
    image = np.full((120, 120, 3), 248, dtype=np.uint8)
    block = {"bbox": [20, 20, 100, 90], "text_pixel_bbox": [35, 35, 80, 58]}
    segmentation = np.zeros((120, 120), dtype=np.uint8)
    segmentation[40:52, 44:50] = 255
    segmentation[40:52, 70:76] = 255

    mask = build_manga_segmentation_mask(image, [block], segmentation)

    assert mask[45, 46] == 255
    assert mask[45, 72] == 255
    assert mask[25, 25] == 0
    assert mask[84, 96] == 0


def test_manga_segmentation_mask_runs_segmenter_on_crop_and_remaps_to_page():
    image = np.full((120, 120, 3), 248, dtype=np.uint8)
    block = {"bbox": [20, 20, 100, 90], "text_pixel_bbox": [35, 35, 80, 58]}
    seen_shapes = []

    def segmenter(crop):
        seen_shapes.append(crop.shape[:2])
        local = np.zeros(crop.shape[:2], dtype=np.uint8)
        local[20:25, 25:28] = 255
        local[20:25, 50:53] = 255
        return local

    mask = build_manga_segmentation_mask(
        image,
        [block],
        None,
        ocr_texts=[block],
        segmenter=segmenter,
    )

    assert seen_shapes == [(70, 80), (120, 120)]
    assert mask[42, 46] == 255
    assert mask[42, 72] == 255
    assert mask[25, 25] == 0
    assert mask[84, 96] == 0


def test_manga_segmentation_mask_recovers_nearby_orphan_segmenter_components():
    image = np.full((120, 160, 3), 248, dtype=np.uint8)
    block = {"bbox": [70, 50, 100, 76], "text_pixel_bbox": [74, 54, 96, 70]}
    seen_shapes = []

    def segmenter(crop):
        seen_shapes.append(crop.shape[:2])
        local = np.zeros(crop.shape[:2], dtype=np.uint8)
        if crop.shape[:2] == image.shape[:2]:
            local[80:90, 94:102] = 255
            local[6:16, 6:16] = 255
        else:
            local[8:14, 10:16] = 255
        return local

    mask = build_manga_segmentation_mask(
        image,
        [block],
        None,
        ocr_texts=[block],
        segmenter=segmenter,
    )

    assert seen_shapes == [(26, 30), (120, 160)]
    assert mask[60, 84] == 255
    assert mask[84, 98] == 255
    assert mask[10, 10] == 0


def test_manga_segmentation_mask_absorbs_dark_core_inside_orphan_outline():
    image = np.full((120, 160, 3), 248, dtype=np.uint8)
    image[84:95, 101:110] = 24
    image[84:95, 40:44] = 24
    block = {"bbox": [70, 50, 100, 76], "text_pixel_bbox": [74, 54, 96, 70]}

    def segmenter(crop):
        local = np.zeros(crop.shape[:2], dtype=np.uint8)
        if crop.shape[:2] == image.shape[:2]:
            local[82:98, 96:100] = 255
            local[82:98, 111:115] = 255
        else:
            local[8:14, 10:16] = 255
        return local

    mask = build_manga_segmentation_mask(
        image,
        [block],
        None,
        ocr_texts=[block],
        segmenter=segmenter,
    )

    assert mask[88, 98] == 255
    assert mask[88, 105] == 255
    assert mask[88, 42] == 0


def test_manga_segmentation_mask_recovers_chained_orphan_components_near_seed():
    image = np.full((190, 190, 3), 248, dtype=np.uint8)
    block = {"bbox": [50, 50, 82, 76], "text_pixel_bbox": [55, 55, 74, 68]}

    def segmenter(crop):
        local = np.zeros(crop.shape[:2], dtype=np.uint8)
        if crop.shape[:2] == image.shape[:2]:
            local[96:106, 82:86] = 255
            local[96:106, 88:92] = 255
            local[132:144, 104:109] = 255
            local[132:144, 112:118] = 255
            local[160:172, 136:142] = 255
            local[160:172, 145:150] = 255
            local[20:32, 165:170] = 255
            local[20:32, 173:178] = 255
        else:
            local[8:14, 10:16] = 255
        return local

    mask = build_manga_segmentation_mask(
        image,
        [block],
        None,
        ocr_texts=[block],
        segmenter=segmenter,
    )

    assert mask[101, 84] == 255
    assert mask[138, 106] == 255
    assert mask[166, 138] == 255
    assert mask[26, 167] == 0


def test_manga_segmentation_mask_skips_preserved_texts_before_segmenting():
    image = np.full((120, 120, 3), 248, dtype=np.uint8)
    block = {"bbox": [20, 20, 100, 90], "text_pixel_bbox": [35, 35, 80, 58]}
    text = {**block, "skip_processing": True, "ignored_reason": "cjk_sfx_preserved"}
    seen_shapes = []

    def segmenter(crop):
        seen_shapes.append(crop.shape[:2])
        return np.full(crop.shape[:2], 255, dtype=np.uint8)

    mask = build_manga_segmentation_mask(
        image,
        [block],
        None,
        ocr_texts=[text],
        segmenter=segmenter,
    )

    assert seen_shapes == []
    assert int(np.count_nonzero(mask)) == 0


def test_manhwa_roi_segmentation_runs_segmenter_on_crop_not_full_tall_page():
    image = np.full((7600, 800, 3), 248, dtype=np.uint8)
    block = {"bbox": [100, 3000, 240, 3120], "text_pixel_bbox": [126, 3032, 190, 3068]}
    seen_shapes = []

    def segmenter(crop):
        seen_shapes.append(crop.shape[:2])
        local = np.zeros(crop.shape[:2], dtype=np.uint8)
        local[52:68, 58:82] = 255
        return local

    mask = build_manhwa_manhua_roi_segmentation_mask(
        image,
        [block],
        [block],
        segmenter=segmenter,
    )

    assert seen_shapes
    assert all(shape[0] < image.shape[0] for shape in seen_shapes)
    assert all(shape[0] <= 1400 for shape in seen_shapes)
    assert int(np.count_nonzero(mask)) > 0
    assert mask[3020, 90] == 0


def test_manhwa_roi_segmentation_skips_preserved_texts_before_segmenting():
    image = np.full((7600, 800, 3), 248, dtype=np.uint8)
    block = {"bbox": [100, 3000, 240, 3120], "text_pixel_bbox": [126, 3032, 190, 3068]}
    text = {**block, "skip_processing": True, "ignored_reason": "cjk_sfx_preserved"}
    seen_shapes = []

    def segmenter(crop):
        seen_shapes.append(crop.shape[:2])
        return np.full(crop.shape[:2], 255, dtype=np.uint8)

    mask = build_manhwa_manhua_roi_segmentation_mask(
        image,
        [block],
        [text],
        segmenter=segmenter,
    )

    assert seen_shapes == []
    assert int(np.count_nonzero(mask)) == 0


def test_manhwa_roi_segmentation_rejects_broad_rectangular_mask_and_falls_back_to_text_geometry():
    image = np.full((7600, 800, 3), 248, dtype=np.uint8)
    block = {
        "bbox": [100, 3000, 260, 3140],
        "text_pixel_bbox": [132, 3032, 198, 3068],
    }

    def segmenter(crop):
        return np.full(crop.shape[:2], 255, dtype=np.uint8)

    mask = build_manhwa_manhua_roi_segmentation_mask(
        image,
        [block],
        [block],
        segmenter=segmenter,
    )

    assert mask[3050, 160] == 255
    assert mask[3010, 110] == 0
    assert mask[3130, 250] == 0


def test_manhwa_roi_segmentation_splits_oversized_tall_regions():
    image = np.full((7600, 800, 3), 248, dtype=np.uint8)
    block = {"bbox": [100, 900, 280, 4200], "text_pixel_bbox": [130, 1000, 210, 1040]}
    seen_shapes = []

    def segmenter(crop):
        seen_shapes.append(crop.shape[:2])
        return np.zeros(crop.shape[:2], dtype=np.uint8)

    build_manhwa_manhua_roi_segmentation_mask(
        image,
        [block],
        [block],
        segmenter=segmenter,
    )

    assert len(seen_shapes) >= 2
    assert all(shape[0] <= 1400 for shape in seen_shapes)


def test_textlike_orphan_recovery_keeps_glyph_strokes_without_long_art_lines():
    image = np.full((180, 260, 3), 248, dtype=np.uint8)
    cv2.putText(image, "A", (42, 92), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (8, 8, 8), 8, cv2.LINE_AA)
    cv2.line(image, (20, 130), (230, 132), (8, 8, 8), 3)
    candidate = np.zeros((180, 260), dtype=np.uint8)
    candidate[35:145, 15:240] = 255

    recovered = _recover_textlike_strokes_from_candidate(candidate, image)

    assert int(np.count_nonzero(recovered[45:105, 35:95])) > 80
    assert int(np.count_nonzero(recovered[124:138, 20:230])) == 0


def test_manga_segmentation_mask_recovers_large_orphan_as_strokes_not_filled_blob():
    image = np.full((180, 260, 3), 248, dtype=np.uint8)
    cv2.putText(image, "GO!", (42, 96), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (170, 20, 20), 8, cv2.LINE_AA)
    block = {"bbox": [34, 42, 150, 112], "text_pixel_bbox": [44, 56, 136, 98]}

    def segmenter(crop):
        local = np.zeros(crop.shape[:2], dtype=np.uint8)
        if crop.shape[:2] == image.shape[:2]:
            local[44:114, 32:154] = 255
        else:
            local[12:58, 10:110] = 255
        return local

    mask = build_manga_segmentation_mask(
        image,
        [block],
        None,
        ocr_texts=[block],
        segmenter=segmenter,
    )

    roi = mask[44:114, 32:154]
    fill_ratio = int(np.count_nonzero(roi)) / float(roi.size)
    assert fill_ratio < 0.55
    assert mask[78, 78] == 255
    assert mask[50, 40] == 0


def test_expand_cjk_glyph_mask_for_inpaint_does_not_fill_between_glyphs():
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:48, 28:36] = 255
    mask[30:48, 78:86] = 255

    expanded = expand_cjk_glyph_mask_for_inpaint(mask, min_radius=2, max_radius=4)

    assert expanded[38, 31] == 255
    assert expanded[38, 82] == 255
    assert expanded[38, 58] == 0


def test_manhwa_roi_fallback_prefers_textlike_strokes_over_filled_bbox():
    image = np.full((120, 220, 3), 236, dtype=np.uint8)
    image[42:50, 42:170] = [170, 40, 70]
    image[62:98, 58:84] = [20, 75, 255]
    image[82:98, 58:144] = [20, 75, 255]
    image[62:98, 126:152] = [20, 75, 255]
    block = {"bbox": [38, 38, 176, 106], "confidence": 0.91}
    text = {"bbox": [38, 38, 176, 106], "text_pixel_bbox": [38, 38, 176, 106], "text": "점화"}

    mask = build_manhwa_manhua_roi_segmentation_mask(
        image,
        [block],
        [text],
        segmenter=None,
    )

    assert mask[88, 70] == 255
    assert mask[88, 136] == 255
    assert mask[100, 42] == 0


def test_manhwa_roi_refines_accepted_broad_segmenter_mask_to_textlike_strokes():
    image = np.full((120, 220, 3), 236, dtype=np.uint8)
    image[42:50, 42:170] = [170, 40, 70]
    image[62:98, 58:84] = [20, 75, 255]
    image[82:98, 58:144] = [20, 75, 255]
    image[62:98, 126:152] = [20, 75, 255]
    block = {"bbox": [38, 38, 176, 106], "confidence": 0.91}
    text = {"bbox": [38, 38, 176, 106], "text_pixel_bbox": [38, 38, 176, 106], "text": "점화"}

    def segmenter(crop):
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        mask[4:68, 2:136] = 255
        return mask

    mask = build_manhwa_manhua_roi_segmentation_mask(
        image,
        [block],
        [text],
        segmenter=segmenter,
    )

    assert mask[88, 70] == 255
    assert mask[88, 136] == 255
    assert mask[100, 42] == 0
