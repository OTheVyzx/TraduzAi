import numpy as np

from vision_stack.cjk_mask_fusion import fuse_cjk_text_mask
from vision_stack.text_mask_evidence import TextEvidence


def test_fusion_reinforces_undercovered_ocr_text():
    image = np.full((90, 140, 3), 240, dtype=np.uint8)
    image[40:50, 60:72] = 18
    base = np.zeros((90, 140), dtype=np.uint8)
    base[40:45, 60:66] = 255
    evidence = [TextEvidence(bbox=[55, 35, 80, 58], text="TEXT", source="ocr")]

    final = fuse_cjk_text_mask(image, base, evidence)

    assert final[42, 62] == 255
    assert final[48, 70] == 255


def test_fusion_rejects_far_segmenter_only_component():
    image = np.full((90, 140, 3), 240, dtype=np.uint8)
    image[70:80, 110:122] = 15
    base = np.zeros((90, 140), dtype=np.uint8)
    evidence = [TextEvidence(bbox=[20, 20, 50, 40], text="TEXT", source="ocr")]

    final = fuse_cjk_text_mask(image, base, evidence)

    assert final[75, 115] == 0


def test_fusion_accepts_hi_sam_mask_inside_evidence_support():
    image = np.full((90, 140, 3), 240, dtype=np.uint8)
    base = np.zeros((90, 140), dtype=np.uint8)
    hi_sam = np.zeros((90, 140), dtype=np.uint8)
    hi_sam[42:48, 64:76] = 255
    evidence = [TextEvidence(bbox=[55, 35, 82, 58], text="TEXT", source="ocr")]

    final = fuse_cjk_text_mask(image, base, evidence, hi_sam_mask=hi_sam)

    assert final[45, 70] == 255
    assert final[10, 10] == 0


def test_fusion_recovers_detached_compact_punctuation_inside_evidence():
    image = np.full((90, 150, 3), 25, dtype=np.uint8)
    image[36:52, 48:82] = 235
    image[55:60, 92:98] = 235
    base = np.zeros((90, 150), dtype=np.uint8)
    base[36:52, 48:82] = 255
    evidence = [TextEvidence(bbox=[44, 32, 105, 66], text="TEXT...", source="ocr")]

    final = fuse_cjk_text_mask(image, base, evidence)

    assert final[57, 95] == 255


def test_fusion_rejects_large_art_patch_inside_evidence_support():
    image = np.full((100, 160, 3), 235, dtype=np.uint8)
    image[55:80, 92:132] = 20
    base = np.zeros((100, 160), dtype=np.uint8)
    base[34:48, 48:76] = 255
    evidence = [TextEvidence(bbox=[44, 30, 138, 84], text="TEXT", source="ocr")]

    final = fuse_cjk_text_mask(image, base, evidence)

    assert final[66, 110] == 0


def test_fusion_accepts_large_saturated_glyph_inside_ocr_evidence_without_base_touch():
    image = np.full((130, 180, 3), 236, dtype=np.uint8)
    image[42:50, 42:140] = [170, 40, 70]
    image[58:100, 54:76] = [20, 75, 255]
    image[82:100, 54:126] = [20, 75, 255]
    image[58:100, 112:132] = [20, 75, 255]
    base = np.zeros((130, 180), dtype=np.uint8)
    base[42:50, 42:140] = 255
    evidence = [TextEvidence(bbox=[38, 38, 146, 106], text="점화", source="ocr")]

    final = fuse_cjk_text_mask(image, base, evidence)

    assert final[88, 62] == 255
    assert final[90, 120] == 255


def test_fusion_replaces_broad_segmenter_blob_with_local_text_pixels():
    image = np.full((120, 180, 3), [70, 110, 180], dtype=np.uint8)
    image[32:46, 38:140] = [180, 35, 45]
    image[56:92, 54:74] = [180, 35, 45]
    image[76:92, 54:132] = [180, 35, 45]
    image[56:92, 112:132] = [180, 35, 45]
    base = np.zeros((120, 180), dtype=np.uint8)
    base[24:102, 28:152] = 255
    evidence = [TextEvidence(bbox=[24, 20, 158, 108], text="SFX", source="ocr")]

    final = fuse_cjk_text_mask(image, base, evidence)

    assert final[84, 62] == 255
    assert final[84, 120] == 255
    assert final[96, 34] == 0
    assert final[102, 146] == 0
