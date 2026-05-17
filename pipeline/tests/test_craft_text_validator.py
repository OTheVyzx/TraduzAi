import numpy as np

from vision_stack.craft_text_validator import measure_craft_coverage


def test_craft_validator_reports_undercovered_character_heatmap():
    mask = np.zeros((80, 120), dtype=np.uint8)
    heatmap = np.zeros((80, 120), dtype=np.float32)
    heatmap[30:40, 50:60] = 0.95

    result = measure_craft_coverage(mask, heatmap)

    assert result["heatmap_pixels"] == 100
    assert result["undercovered_pixels"] == 100
    assert result["coverage_ratio"] == 0.0
