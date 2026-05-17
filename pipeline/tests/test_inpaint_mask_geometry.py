import numpy as np
import cv2

from inpainter.mask_builder import (
    bbox_to_octagon_mask,
    build_raw_text_mask_from_image,
    build_inpaint_mask,
    expand_text_mask,
    mask_from_text_geometry,
    polygon_to_mask,
)
from vision_stack.cjk_segmentation_mask import build_manhwa_manhua_roi_segmentation_mask


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
    assert mask[40, 62] == 255
    assert mask[12, 12] == 0


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
    assert mask[46, 47] == 255
    assert mask[46, 64] == 255
    assert mask[18, 70] == 0


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
