import numpy as np

from main import _project_inpaint_block_from_vision_block
from vision_stack.runtime import vision_blocks_to_mask
from vision_stack.smart_text_mask import SmartTextMaskEngine


def test_smart_text_mask_builds_local_mask_from_text_geometry():
    image = np.full((48, 64, 3), 255, dtype=np.uint8)
    image[18:26, 20:36] = 0
    block = {
        "bbox": [16, 14, 40, 30],
        "text_pixel_bbox": [18, 16, 38, 28],
        "line_polygons": [[[18, 16], [38, 16], [38, 28], [18, 28]]],
        "confidence": 0.9,
    }

    result = SmartTextMaskEngine().build_mask(image, [block], quality="normal")

    assert result.stats["built"] == 1
    assert result.stats["failed"] == 0
    assert result.entries[0]["mask_source"] == "smart_text_mask"
    assert result.entries[0]["mask_bbox"][0] >= 14
    assert result.entries[0]["mask"].ndim == 2
    assert result.entries[0]["mask"].shape != result.mask.shape
    assert int(np.count_nonzero(result.mask)) > 0


def test_smart_text_mask_preserves_entry_when_mask_fails():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    block = {"confidence": 0.3}

    result = SmartTextMaskEngine().build_mask(image, [block], quality="normal")

    assert result.stats["built"] == 0
    assert result.stats["failed"] == 1
    assert "mask" not in result.entries[0]


def test_vision_blocks_to_mask_honors_local_smart_mask_bbox():
    local_mask = np.full((4, 5), 255, dtype=np.uint8)
    block = {
        "bbox": [0, 0, 20, 20],
        "mask": local_mask,
        "mask_bbox": [7, 8, 12, 12],
    }

    mask = vision_blocks_to_mask((24, 24, 3), [block], expand_mask=False)

    assert int(np.count_nonzero(mask[8:12, 7:12])) == 20
    assert int(np.count_nonzero(mask[0:6, 0:6])) == 0


def test_project_inpaint_block_preserves_smart_mask_metadata():
    block = {
        "bbox": [1, 2, 10, 12],
        "mask_bbox": [2, 3, 9, 11],
        "mask_source": "smart_text_mask",
        "mask_confidence": 0.91,
        "mask_pixels": 42,
        "text_refiner": "ppocrv5_text_refiner",
    }

    project_block = _project_inpaint_block_from_vision_block(block)

    assert project_block["mask_bbox"] == [2, 3, 9, 11]
    assert project_block["mask_source"] == "smart_text_mask"
    assert project_block["mask_confidence"] == 0.91
    assert project_block["mask_pixels"] == 42
    assert project_block["text_refiner"] == "ppocrv5_text_refiner"
