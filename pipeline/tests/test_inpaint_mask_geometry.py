import numpy as np

from inpainter.mask_builder import build_inpaint_mask, mask_from_text_geometry, polygon_to_mask


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
