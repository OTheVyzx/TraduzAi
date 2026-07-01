import numpy as np

from inpainter.lama_onnx import (
    _write_sfx_roi_debug,
    build_lama_region_jobs,
    build_lama_component_rois,
    pad_to_modulo_reflect,
    prepare_lama_dynamic_inputs,
)


def test_build_component_rois_splits_connected_mask_components():
    image = np.zeros((100, 140, 3), dtype=np.uint8)
    mask = np.zeros((100, 140), dtype=np.uint8)
    mask[10:30, 12:28] = 255
    mask[60:78, 90:112] = 255

    jobs = build_lama_component_rois(image, mask, min_component_area=20, margin_ratio=0.0)

    assert len(jobs) == 2
    bboxes = sorted([tuple(job["bbox"]) for job in jobs])
    assert bboxes[0] == (4, 2, 36, 38)
    assert bboxes[1] == (82, 52, 120, 86)


def test_build_component_rois_applies_margin_clamp_to_at_least_minimum():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[20:24, 30:34] = 255

    jobs = build_lama_component_rois(image, mask, min_component_area=4, margin_ratio=0.0)

    assert len(jobs) == 1
    x1, y1, x2, y2 = jobs[0]["bbox"]
    assert (x1, y1, x2, y2) == (22, 12, 42, 32)


def test_prepare_lama_dynamic_inputs_snaps_to_multiple_of_8_and_preserves_top_left():
    image = np.full((19, 23, 3), 255, dtype=np.uint8)
    mask = np.zeros((19, 23), dtype=np.uint8)
    mask[2:17, 4:18] = 255

    image_input, mask_input, original_size = prepare_lama_dynamic_inputs(image, mask)

    assert original_size == (19, 23)
    assert image_input.shape == (1, 3, 24, 24)
    assert mask_input.shape == (1, 1, 24, 24)
    assert image_input.shape[2] % 8 == 0
    assert image_input.shape[3] % 8 == 0
    np.testing.assert_allclose(image_input[0, :, :19, :23], np.transpose(image / 255.0, (2, 0, 1)))


def test_pad_to_modulo_reflect_uses_reflect_padding():
    image = np.arange(1 * 6 * 6, dtype=np.float32).reshape(1, 6, 6)
    padded = pad_to_modulo_reflect(image, modulo=8)

    expected = np.pad(image, ((0, 0), (0, 2), (0, 2)), mode="reflect")
    assert padded.shape == (1, 8, 8)
    np.testing.assert_array_equal(padded, expected)


def test_build_lama_component_rois_skips_already_occupied_mask():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:22, 10:24] = 255
    mask[28:40, 30:42] = 255

    occupied = np.zeros((64, 64), dtype=np.uint8)
    occupied[12:20, 12:20] = 255

    jobs = build_lama_component_rois(
        image,
        mask,
        min_component_area=4,
        occupied_mask=occupied,
        margin_ratio=0.0,
    )

    assert len(jobs) == 1


def test_safe_sfx_region_jobs_use_glyph_components_not_full_bbox():
    image = np.full((80, 120, 3), 238, dtype=np.uint8)
    image[28:48, 28:38] = 8
    image[28:38, 28:58] = 8
    image[28:48, 72:82] = 8
    image[38:48, 72:98] = 8
    text = {
        "bbox": [12, 12, 108, 64],
        "text_pixel_bbox": [12, 12, 108, 64],
        "text": "\ucff5",
        "raw_ocr": "\ucff5",
        "route_action": "translate_sfx_inpaint_render",
        "content_class": "sfx",
        "script": "hangul",
        "mask_evidence": {"kind": "sfx_glyph_mask"},
        "sfx": {"inpaint_allowed": True},
    }

    jobs = build_lama_region_jobs(image, [text])

    assert jobs
    full_bbox_area = (text["bbox"][2] - text["bbox"][0]) * (text["bbox"][3] - text["bbox"][1])
    mask_area = sum(int(np.count_nonzero(job["mask"])) for job in jobs)
    assert mask_area < int(full_bbox_area * 0.55)
    assert all(job["mask"].ndim == 2 for job in jobs)
    assert all(int(np.count_nonzero(job["mask"])) < job["mask"].size for job in jobs)
    assert all(job["bbox"] != text["bbox"] for job in jobs)


def test_blocked_sfx_region_does_not_fallback_to_full_bbox_jobs():
    image = np.full((80, 120, 3), 238, dtype=np.uint8)
    image[28:48, 28:38] = 8
    text = {
        "bbox": [12, 12, 108, 64],
        "text_pixel_bbox": [12, 12, 108, 64],
        "text": "\ucff5",
        "route_action": "translate_sfx_inpaint_render",
        "content_class": "sfx",
        "script": "hangul",
        "mask_evidence": {"kind": "sfx_glyph_mask"},
        "sfx": {"inpaint_allowed": False, "qa_flags": ["complex_background"]},
    }

    jobs = build_lama_region_jobs(image, [text])

    assert jobs == []


def test_sfx_region_jobs_recheck_gate_flags_even_when_inpaint_allowed_true():
    image = np.full((80, 120, 3), 238, dtype=np.uint8)
    image[28:48, 28:38] = 8
    text = {
        "bbox": [12, 12, 108, 64],
        "text_pixel_bbox": [12, 12, 108, 64],
        "text": "\ucff5",
        "route_action": "translate_sfx_inpaint_render",
        "content_class": "sfx",
        "script": "hangul",
        "mask_evidence": {"kind": "sfx_glyph_mask", "bbox_fill_ratio": 0.12, "expanded_mask_pixels": 120},
        "sfx": {"inpaint_allowed": True, "qa_flags": ["complex_background"]},
    }

    jobs = build_lama_region_jobs(image, [text])

    assert jobs == []
    assert text["sfx_inpaint_gate"]["allow_inpaint"] is False


def test_write_sfx_roi_debug_outputs_mask_before_after_and_diff(tmp_path):
    before = np.full((16, 20, 3), 80, dtype=np.uint8)
    after = before.copy()
    after[6:10, 8:12] = 160
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[6:10, 8:12] = 255
    job = {
        "mask": mask,
        "band_id": "page_001_band_002",
        "sfx_id": "sfx:01",
    }
    ocr_data = {"_debug_root": str(tmp_path)}

    _write_sfx_roi_debug(ocr_data, job, before, after, tmp_path / "out")

    debug_dir = tmp_path / "08_inpaint" / "page_001_band_002"
    assert (debug_dir / "sfx_sfx_01_mask.png").exists()
    assert (debug_dir / "sfx_sfx_01_before.png").exists()
    assert (debug_dir / "sfx_sfx_01_after.png").exists()
    assert (debug_dir / "sfx_sfx_01_diff.png").exists()
