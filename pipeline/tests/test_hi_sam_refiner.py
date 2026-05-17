import numpy as np

from vision_stack.hi_sam_refiner import HiSamTextRefiner


def test_hi_sam_refiner_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("TRADUZAI_HISAM_TEXT_REFINE", "0")
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    seed = np.zeros((80, 120), dtype=np.uint8)

    refiner = HiSamTextRefiner(model=lambda crop, mask: mask)

    assert refiner.refine(image, [10, 20, 50, 60], seed, evidence=[]) is None


def test_hi_sam_refiner_remaps_crop_mask_to_page(monkeypatch):
    monkeypatch.setenv("TRADUZAI_HISAM_TEXT_REFINE", "1")
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    seed = np.zeros((80, 120), dtype=np.uint8)

    def model(crop, mask):
        del mask
        local = np.zeros(crop.shape[:2], dtype=np.uint8)
        local[5:10, 6:12] = 255
        return local

    result = HiSamTextRefiner(model=model).refine(image, [30, 20, 70, 60], seed, evidence=[])

    assert result is not None
    assert result.mask[26, 38] == 255
    assert result.mask[5, 5] == 0
