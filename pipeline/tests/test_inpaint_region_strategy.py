import numpy as np
from PIL import Image

from inpainter.region_strategy import (
    MangaCleanerROI,
    debug_output_paths,
    manga_cleaner_roi_from_mask,
    pasteback_masked_pixels,
    plan_inpaint,
)


def _mask(path):
    Image.new("RGBA", (20, 20), (255, 255, 255, 255)).save(path)


def test_classifier_uses_region_type_strategy(tmp_path):
    path = tmp_path / "mask.png"
    _mask(path)

    plan = plan_inpaint({"bbox": [0, 0, 20, 20], "background_type": "textured_background"}, path)

    assert plan["run"] is True
    assert plan["strategy"] == "lama_required"


def test_sfx_is_skipped_without_config(tmp_path):
    path = tmp_path / "mask.png"
    _mask(path)

    plan = plan_inpaint({"bbox": [0, 0, 20, 20], "tipo": "sfx"}, path)

    assert plan["run"] is False
    assert "sfx_preserved" in plan["qa_flags"]


def test_invalid_mask_blocks_inpaint(tmp_path):
    plan = plan_inpaint({"bbox": [0, 0, 20, 20]}, tmp_path / "missing.png")

    assert plan["run"] is False
    assert "mask_missing" in plan["qa_flags"]


def test_debug_outputs_are_declared(tmp_path):
    paths = debug_output_paths(tmp_path, 1)

    assert paths["before"].name == "page_001_before.png"
    assert paths["diff"].parent.exists()


def test_manga_cleaner_roi_from_mask_tracks_source_crop_and_padded_size():
    mask = np.zeros((101, 157), dtype=np.uint8)
    mask[25:64, 41:83] = 255

    roi = manga_cleaner_roi_from_mask(mask, padding=12, multiple=8)

    assert isinstance(roi, MangaCleanerROI)
    assert roi.x1 <= 41
    assert roi.y1 <= 25
    assert roi.x2 > 83
    assert roi.y2 > 64
    assert roi.source_width == roi.x2 - roi.x1
    assert roi.source_height == roi.y2 - roi.y1
    assert roi.padded_width % 8 == 0
    assert roi.padded_height % 8 == 0


def test_pasteback_masked_pixels_changes_only_masked_crop_pixels():
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    crop_output = np.full((20, 20, 3), 200, dtype=np.uint8)
    crop_mask = np.zeros((20, 20), dtype=np.uint8)
    crop_mask[5:10, 5:10] = 255

    result = pasteback_masked_pixels(base, crop_output, crop_mask, [10, 10, 30, 30])

    assert np.all(result[16, 16] == 200)
    assert np.all(result[12, 12] == 0)
    assert np.all(result[0, 0] == 0)
