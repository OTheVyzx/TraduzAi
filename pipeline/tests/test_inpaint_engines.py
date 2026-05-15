import numpy as np

from inpainter.engines.base import pixel_lock_composite
from inpainter.engines.composite import CompositeBandInpaintEngine
from inpainter.engines.opencv_fallback import OpenCVFallbackInpaintEngine
from vision_stack.runtime import vision_blocks_to_mask


def test_pixel_lock_composite_preserves_everything_outside_mask():
    original = np.zeros((12, 12, 3), dtype=np.uint8)
    inpainted = np.full((12, 12, 3), 255, dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[4:8, 5:9] = 255

    result = pixel_lock_composite(original, inpainted, mask)

    assert np.all(result[mask == 0] == 0)
    assert np.all(result[mask > 0] == 255)


def test_opencv_fallback_inpaint_keeps_pixels_outside_mask():
    image = np.full((24, 24, 3), 180, dtype=np.uint8)
    image[8:16, 8:16] = 0
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[8:16, 8:16] = 255

    result = OpenCVFallbackInpaintEngine().inpaint(image, mask, quality="normal")

    assert result.shape == image.shape
    assert np.array_equal(result[mask == 0], image[mask == 0])


def test_composite_band_inpaint_uses_vision_block_mask_and_records_metadata():
    image = np.full((24, 24, 3), 180, dtype=np.uint8)
    image[8:16, 8:16] = 0
    local_mask = np.full((8, 8), 255, dtype=np.uint8)
    page = {
        "texts": [{"text": "HELLO", "bbox": [8, 8, 16, 16]}],
        "_vision_blocks": [{"bbox": [8, 8, 16, 16], "mask": local_mask, "mask_bbox": [8, 8, 16, 16]}],
    }

    result = CompositeBandInpaintEngine().inpaint_band_image(image, page)
    final_mask = vision_blocks_to_mask(image.shape, page["_vision_blocks"], image_rgb=image, expand_mask=True)

    assert result.shape == image.shape
    assert np.array_equal(result[final_mask == 0], image[final_mask == 0])
    assert page["_inpaint_engine"] == "lama_onnx_composite"
    assert page["_strip_used_real_inpaint"] is True


def test_composite_band_inpaint_records_residual_validation():
    class CleanValidator:
        name = "residual_validator"

        def validate(self, original_roi, cleaned_roi, mask, text_entries):
            return {
                "status": "clean",
                "residual_bboxes": [],
                "retry_recommended": False,
                "confidence": 0.9,
            }

    image = np.full((24, 24, 3), 180, dtype=np.uint8)
    local_mask = np.full((6, 6), 255, dtype=np.uint8)
    page = {
        "texts": [{"text": "HELLO", "bbox": [9, 9, 15, 15]}],
        "_vision_blocks": [{"bbox": [9, 9, 15, 15], "mask": local_mask, "mask_bbox": [9, 9, 15, 15]}],
    }

    CompositeBandInpaintEngine(validator=CleanValidator()).inpaint_band_image(image, page)

    assert page["_residual_validation"]["status"] == "clean"
