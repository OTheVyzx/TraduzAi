from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import _external_mask_vision_block, _finalize_reinpaint_output_path


def test_external_mask_vision_block_uses_bbox_crop(tmp_path: Path):
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (12, 10), 0)
    for x in range(4, 8):
        for y in range(3, 6):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)

    block = _external_mask_vision_block({"bbox": [2, 2, 10, 8], "mask_path": str(mask_path)})

    assert block is not None
    assert block["bbox"] == [2, 2, 10, 8]
    assert isinstance(block["mask"], np.ndarray)
    assert block["mask"].shape == (6, 8)
    assert int(np.count_nonzero(block["mask"])) == 12


def test_external_mask_vision_block_accepts_bbox_sized_mask(tmp_path: Path):
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (8, 6), 0)
    for x in range(2, 6):
        for y in range(1, 4):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)

    block = _external_mask_vision_block({"bbox": [20, 30, 28, 36], "mask_path": str(mask_path)})

    assert block is not None
    assert block["bbox"] == [20, 30, 28, 36]
    assert isinstance(block["mask"], np.ndarray)
    assert block["mask"].shape == (6, 8)
    assert int(np.count_nonzero(block["mask"])) == 12


def test_external_mask_vision_block_rejects_empty_mask(tmp_path: Path):
    mask_path = tmp_path / "mask.png"
    Image.new("L", (12, 10), 0).save(mask_path)

    assert _external_mask_vision_block({"bbox": [2, 2, 10, 8], "mask_path": str(mask_path)}) is None


def test_regional_reinpaint_finalizes_into_accumulated_inpaint(tmp_path: Path):
    base_path = tmp_path / "current_inpaint.png"
    fallback_path = tmp_path / "original.png"
    generated_path = tmp_path / "cache" / "current_inpaint.png"
    generated_path.parent.mkdir(parents=True)

    Image.new("RGB", (8, 8), (255, 0, 0)).save(base_path)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(fallback_path)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(generated_path)

    completed = _finalize_reinpaint_output_path(
        outputs=[generated_path],
        is_regional=True,
        base_path=base_path,
        fallback_path=fallback_path,
        bbox=[2, 2, 6, 6],
        default_path=tmp_path / "unused.png",
    )

    assert completed == base_path
    with Image.open(base_path) as merged:
        assert merged.getpixel((1, 1)) == (255, 0, 0)
        assert merged.getpixel((3, 3)) == (0, 0, 255)
