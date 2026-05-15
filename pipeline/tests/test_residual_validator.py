import numpy as np

from vision_stack.residual_validator import ResidualValidationEngine


def test_residual_validator_reports_clean_on_flat_masked_region():
    original = np.full((32, 32, 3), 200, dtype=np.uint8)
    cleaned = original.copy()
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    result = ResidualValidationEngine(quality="ultra").validate(original, cleaned, mask, [])

    assert result["status"] == "clean"
    assert result["retry_recommended"] is False


def test_residual_validator_flags_high_contrast_leftover_text():
    original = np.full((32, 32, 3), 200, dtype=np.uint8)
    cleaned = original.copy()
    cleaned[14:18, 10:22] = 0
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    result = ResidualValidationEngine(quality="ultra").validate(original, cleaned, mask, [])

    assert result["status"] == "residual"
    assert result["retry_recommended"] is True
    assert result["residual_bboxes"]
