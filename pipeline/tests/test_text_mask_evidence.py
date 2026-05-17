import numpy as np

from vision_stack.text_mask_evidence import measure_mask_coverage, normalize_text_evidence


def test_normalize_text_evidence_keeps_bbox_and_text():
    page = {
        "texts": [{"text": "ガシャーン", "bbox": [10, 20, 80, 50], "confidence": 0.91}],
        "_vision_blocks": [{"bbox": [8, 18, 84, 54]}],
    }

    evidence = normalize_text_evidence(page, width=100, height=100)

    assert evidence[0].text == "ガシャーン"
    assert evidence[0].bbox == [10, 20, 80, 50]
    assert evidence[0].source == "ocr"


def test_measure_mask_coverage_counts_dark_pixels_inside_evidence():
    image = np.full((80, 120, 3), 240, dtype=np.uint8)
    image[30:40, 50:60] = 20
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:35, 50:60] = 255
    evidence = normalize_text_evidence({"texts": [{"bbox": [45, 25, 70, 50]}]}, width=120, height=80)

    coverage = measure_mask_coverage(mask, image, evidence)

    assert coverage["dark_pixels"] == 100
    assert coverage["dark_inside_mask"] == 50
    assert coverage["dark_outside_mask"] == 50
    assert coverage["coverage_ratio"] == 0.5
