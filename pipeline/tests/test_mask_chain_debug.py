import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools import DebugRecorder, bind_recorder


def _sample_band_masks():
    image = np.full((20, 20, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((20, 20), dtype=np.uint8)
    raw_mask[2:8, 2:8] = 255
    expanded_mask = np.zeros((20, 20), dtype=np.uint8)
    expanded_mask[1:10, 1:10] = 255
    return image, raw_mask, expanded_mask


def test_mask_debug_payload_flags_density_and_outside_balloon_pixels():
    from debug_tools.masks import build_mask_chain_debug_payload

    image, raw_mask, expanded_mask = _sample_band_masks()
    ocr_page = {
        "numero": 1,
        "_band_index": 2,
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_002",
                "text_instance_id": "page_001_band_002_ocr_001",
                "bbox": [2, 2, 8, 8],
                "text_pixel_bbox": [2, 2, 8, 8],
                "line_polygons": [[[2, 2], [8, 2], [8, 8], [2, 8]]],
                "balloon_bbox": [1, 1, 5, 5],
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
        protection_mask=np.zeros_like(raw_mask),
    )

    assert decision["band_id"] == "page_001_band_002"
    assert decision["text_id"] == "ocr_001"
    assert decision["trace_ids"] == ["ocr_001@page_001_band_002"]
    assert decision["text_instance_ids"] == ["page_001_band_002_ocr_001"]
    assert decision["mask_density_in_band"] > 0.12
    assert decision["mask_balloon_ratio"] > 1.0
    assert decision["outside_balloon_pixels"] > 50
    assert decision["outside_balloon_ratio"] >= 0.18
    assert "mask_density_high" in decision["flags"]
    assert "mask_outside_balloon_critical" in decision["flags"]
    assert decision["gates"]["mask_density_high"] is True
    assert decision["gates"]["mask_outside_balloon_critical"] is True
    assert set(images) == {
        "01_glyph_mask.png",
        "02_line_polygon_mask.png",
        "03_detected_text_mask.png",
        "04_balloon_mask.png",
        "05_balloon_inner_mask.png",
        "06_protection_mask.png",
        "07_raw_text_mask.png",
        "08_expanded_text_mask.png",
        "09_final_inpaint_mask.png",
        "10_mask_overlay.jpg",
    }


def test_mask_debug_downgrades_synthetic_tight_balloon_reference():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((30, 30, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((30, 30), dtype=np.uint8)
    raw_mask[8:18, 8:18] = 255
    expanded_mask = np.zeros((30, 30), dtype=np.uint8)
    expanded_mask[6:20, 6:20] = 255
    ocr_page = {
        "numero": 1,
        "_band_index": 2,
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_002",
                "text_instance_id": "page_001_band_002_ocr_001",
                "bbox": [8, 8, 18, 18],
                "text_pixel_bbox": [8, 8, 18, 18],
                "line_polygons": [[[8, 8], [18, 8], [18, 18], [8, 18]]],
                "balloon_bbox": [8, 8, 18, 18],
            }
        ],
    }

    decision, _images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
        protection_mask=np.zeros_like(raw_mask),
    )

    assert decision["synthetic_tight_balloon_reference"] is True
    assert decision["outside_balloon_pixels"] > 50
    assert decision["outside_balloon_ratio"] >= 0.18
    assert "mask_outside_balloon" in decision["flags"]
    assert "mask_outside_balloon_critical" not in decision["flags"]
    assert decision["gates"]["mask_outside_balloon_critical"] is False


def test_mask_debug_does_not_raise_density_for_synthetic_tight_balloon_reference():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((80, 100, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((80, 100), dtype=np.uint8)
    raw_mask[10:42, 30:78] = 255
    expanded_mask = raw_mask.copy()
    ocr_page = {
        "numero": 1,
        "_band_index": 2,
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [36, 12, 76, 40],
                "text_pixel_bbox": [36, 12, 76, 40],
                "line_polygons": [[[30, 10], [78, 10], [78, 42], [30, 42]]],
                "balloon_bbox": [35, 10, 78, 42],
                "balloon_type": "textured",
                "layout_profile": "standard",
            }
        ],
    }

    decision, _images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
        protection_mask=np.zeros_like(raw_mask),
    )

    assert decision["mask_density_in_band"] > 0.12
    assert decision["synthetic_tight_balloon_reference"] is True
    assert "mask_density_high" not in decision["flags"]
    assert decision["gates"]["mask_density_high"] is False


def test_mask_debug_downgrades_tight_text_bbox_balloon_reference_to_review():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[0:34, 0:70] = 255
    expanded_mask = np.zeros((100, 100), dtype=np.uint8)
    expanded_mask[0:40, 0:73] = 255
    ocr_page = {
        "numero": 10,
        "_band_index": 93,
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_010_band_093",
                "text_instance_id": "page_010_band_093_ocr_001",
                "bbox": [0, 0, 70, 34],
                "text_pixel_bbox": [0, 0, 70, 34],
                "balloon_bbox": [0, 0, 70, 34],
                "balloon_type": "white",
            }
        ],
    }

    decision, _images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
        protection_mask=np.zeros_like(raw_mask),
    )

    assert decision["mask_source"] == "text_pixel_bbox"
    assert decision["edge_clipped_text_bbox_reference"] is True
    assert decision["outside_balloon_pixels"] > 50
    assert decision["outside_balloon_ratio"] >= 0.18
    assert decision["mask_balloon_ratio"] <= 1.35
    assert "mask_outside_balloon" in decision["flags"]
    assert "mask_outside_balloon_critical" not in decision["flags"]
    assert decision["gates"]["mask_outside_balloon_critical"] is False


def test_mask_debug_keeps_text_bbox_outside_balloon_critical_for_broad_overreach():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[20:50, 20:50] = 255
    expanded_mask = np.zeros((100, 100), dtype=np.uint8)
    expanded_mask[10:70, 10:70] = 255
    ocr_page = {
        "numero": 1,
        "_band_index": 7,
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [20, 20, 50, 50],
                "text_pixel_bbox": [20, 20, 50, 50],
                "balloon_bbox": [20, 20, 50, 50],
            }
        ],
    }

    decision, _images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
        protection_mask=np.zeros_like(raw_mask),
    )

    assert decision["mask_source"] == "text_pixel_bbox"
    assert decision["edge_clipped_text_bbox_reference"] is False
    assert decision["mask_balloon_ratio"] > 1.35
    assert "mask_outside_balloon_critical" in decision["flags"]
    assert decision["gates"]["mask_outside_balloon_critical"] is True


def test_mask_debug_keeps_text_bbox_outside_balloon_critical_with_bbox_overreach_flag():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[0:34, 0:70] = 255
    expanded_mask = np.zeros((100, 100), dtype=np.uint8)
    expanded_mask[0:40, 0:73] = 255
    ocr_page = {
        "numero": 1,
        "_band_index": 8,
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [0, 0, 70, 34],
                "text_pixel_bbox": [0, 0, 70, 34],
                "balloon_bbox": [0, 0, 70, 34],
                "qa_flags": ["bbox_overreach_critical"],
            }
        ],
    }

    decision, _images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
        protection_mask=np.zeros_like(raw_mask),
    )

    assert decision["edge_clipped_text_bbox_reference"] is False
    assert "mask_outside_balloon_critical" in decision["flags"]
    assert decision["gates"]["mask_outside_balloon_critical"] is True


def test_write_strip_inpaint_debug_exports_mask_chain_with_active_recorder_without_env(tmp_path, monkeypatch):
    from inpainter import _write_strip_inpaint_debug

    image, raw_mask, expanded_mask = _sample_band_masks()
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-mask")
    bind_recorder(recorder)
    monkeypatch.delenv("TRADUZAI_INPAINT_DEBUG_DIR", raising=False)
    ocr_page = {
        "numero": 1,
        "_band_index": 2,
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [2, 2, 8, 8],
                "text_pixel_bbox": [2, 2, 8, 8],
                "line_polygons": [[[2, 2], [8, 2], [8, 8], [2, 8]]],
                "balloon_bbox": [1, 1, 5, 5],
            }
        ],
    }

    try:
        _write_strip_inpaint_debug(
            ocr_page,
            original_rgb=image,
            working_rgb=image.copy(),
            cleaned_rgb=image.copy(),
            vision_blocks=list(ocr_page["texts"]),
            used_real_inpaint=True,
            fast_fill_mask=np.zeros_like(raw_mask),
            raw_mask=raw_mask,
            expanded_mask=expanded_mask,
        )
    finally:
        bind_recorder(None)

    mask_root = tmp_path / "debug" / "e2e" / "06_mask_segmentation" / "page_001_band_002"
    assert (mask_root / "01_glyph_mask.png").exists()
    assert (mask_root / "10_mask_overlay.jpg").exists()
    decision = json.loads((mask_root / "mask_decision.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "debug" / "e2e" / "06_mask_segmentation" / "mask_chain_summary.json").read_text(encoding="utf-8"))

    assert "mask_density_high" in decision["flags"]
    assert "mask_outside_balloon_critical" in decision["flags"]
    assert summary["band_count"] == 1
    assert summary["bands_with_flags"] == 1
    assert summary["flagged_bands"] == ["page_001_band_002"]


def test_mask_debug_ignores_tiny_outside_balloon_ratio():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    raw_mask = np.ones((100, 100), dtype=np.uint8) * 255
    expanded_mask = np.ones((100, 100), dtype=np.uint8) * 255
    ocr_page = {
        "numero": 1,
        "_band_index": 3,
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_003",
                "text_instance_id": "page_001_band_003_ocr_001",
                "bbox": [0, 0, 100, 100],
                "text_pixel_bbox": [0, 0, 100, 100],
                "line_polygons": [[[0, 0], [99, 0], [99, 99], [0, 99]]],
                "balloon_bbox": [0, 0, 99, 99],
            }
        ],
    }

    decision, _ = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
        protection_mask=np.zeros_like(raw_mask),
    )

    assert decision["outside_balloon_pixels"] > 50
    assert decision["outside_balloon_ratio"] < 0.18
    assert decision["outside_balloon_ratio"] < 0.08
    assert "mask_outside_balloon" not in decision["flags"]
    assert "mask_outside_balloon_critical" not in decision["flags"]
    assert decision["gates"]["mask_outside_balloon"] is False
    assert decision["gates"]["mask_outside_balloon_critical"] is False


def test_mask_debug_downgrades_moderate_outside_balloon_ratio_to_review():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    raw_mask = np.ones((100, 100), dtype=np.uint8) * 255
    expanded_mask = np.ones((100, 100), dtype=np.uint8) * 255
    ocr_page = {
        "numero": 1,
        "_band_index": 3,
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_003",
                "text_instance_id": "page_001_band_003_ocr_001",
                "bbox": [0, 0, 100, 100],
                "text_pixel_bbox": [0, 0, 100, 100],
                "line_polygons": [[[0, 0], [99, 0], [99, 99], [0, 99]]],
                "balloon_bbox": [0, 0, 95, 95],
            }
        ],
    }

    decision, _ = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
        protection_mask=np.zeros_like(raw_mask),
    )

    assert decision["outside_balloon_pixels"] > 50
    assert 0.08 <= decision["outside_balloon_ratio"] < 0.18
    assert "mask_outside_balloon" in decision["flags"]
    assert "mask_outside_balloon_critical" not in decision["flags"]
    assert decision["gates"]["mask_outside_balloon"] is True
    assert decision["gates"]["mask_outside_balloon_critical"] is False


def test_mask_debug_uses_effective_final_mask_for_outside_balloon_when_protected():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    raw_mask = np.ones((20, 20), dtype=np.uint8) * 255
    expanded_mask = np.ones((20, 20), dtype=np.uint8) * 255
    protection_mask = np.ones((20, 20), dtype=np.uint8) * 255
    protection_mask[5:15, 5:15] = 0
    ocr_page = {
        "band_id": "page_001_band_001",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [0, 0, 20, 20],
                "balloon_bbox": [5, 5, 15, 15],
            }
        ],
    }

    decision, _ = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
        protection_mask=protection_mask,
    )

    assert decision["used_protection_mask"] is True
    assert decision["outside_balloon_reference"] == "final_mask"
    assert decision["outside_balloon_pixels"] == 0
    assert decision["outside_balloon_ratio"] == 0.0
    assert "mask_outside_balloon" not in decision["flags"]
    assert "mask_outside_balloon_critical" not in decision["flags"]


def test_mask_debug_uses_trace_band_id_as_artifact_identity():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((20, 20, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((20, 20), dtype=np.uint8)
    expanded_mask = np.zeros((20, 20), dtype=np.uint8)
    raw_mask[2:8, 2:8] = 255
    expanded_mask[1:10, 1:10] = 255
    ocr_page = {
        "numero": 1,
        "_band_index": 4,
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_003",
                "bbox": [2, 2, 8, 8],
                "line_polygons": [[[2, 2], [8, 2], [8, 8], [2, 8]]],
                "balloon_bbox": [1, 1, 10, 10],
            }
        ],
    }

    decision, _ = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
    )

    assert decision["band_id"] == "page_001_band_003"


def test_mask_debug_does_not_flag_clean_top_narration_density():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    expanded_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[0:33, :] = 255
    expanded_mask[0:33, :] = 255
    ocr_page = {
        "numero": 1,
        "_band_index": 4,
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_004",
                "bbox": [0, 0, 100, 33],
                "text_pixel_bbox": [0, 0, 100, 33],
                "line_polygons": [[[0, 0], [99, 0], [99, 32], [0, 32]]],
                "balloon_bbox": [0, 0, 100, 40],
                "tipo": "narracao",
                "content_class": "narration",
                "layout_profile": "top_narration",
                "balloon_type": "white",
            }
        ],
    }

    decision, _ = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    assert decision["mask_density_in_band"] > 0.30
    assert decision["outside_balloon_ratio"] == 0.0
    assert decision["expanded_raw_ratio"] == 1.0
    assert "mask_density_high" not in decision["flags"]
    assert decision["gates"]["mask_density_high"] is False


def test_mask_debug_does_not_flag_borderline_clean_dialogue_density():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    expanded_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[0:13, :] = 255
    expanded_mask[0:13, :] = 255
    ocr_page = {
        "numero": 1,
        "_band_index": 5,
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_005",
                "bbox": [0, 0, 100, 13],
                "text_pixel_bbox": [0, 0, 100, 13],
                "line_polygons": [[[0, 0], [99, 0], [99, 12], [0, 12]]],
                "balloon_bbox": [0, 0, 100, 25],
                "tipo": "fala",
                "content_class": "dialogue",
            }
        ],
    }

    decision, _ = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    assert 0.12 < decision["mask_density_in_band"] < 0.15
    assert decision["outside_balloon_ratio"] == 0.0
    assert "mask_density_high" not in decision["flags"]


def test_mask_debug_keeps_connected_group_density_when_source_area_is_broad():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((100, 100, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    expanded_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[0:32, :] = 255
    expanded_mask[0:32, :] = 255
    ocr_page = {
        "numero": 1,
        "_band_index": 6,
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_006",
                "bbox": [0, 0, 100, 100],
                "line_polygons": [[[10, 10], [30, 10], [30, 30], [10, 30]]],
                "balloon_bbox": [0, 0, 100, 100],
                "layout_profile": "connected_balloon",
            },
            {
                "id": "ocr_002",
                "trace_id": "ocr_002@page_001_band_006",
                "bbox": [0, 0, 100, 100],
                "line_polygons": [[[70, 70], [90, 70], [90, 90], [70, 90]]],
                "balloon_bbox": [0, 0, 100, 100],
                "layout_profile": "connected_balloon",
            },
        ],
    }

    decision, _ = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    assert decision["mask_density_in_band"] >= 0.30
    assert decision["source_glyph_area_ratio"] >= 1.5
    assert "mask_density_high" in decision["flags"]
    assert decision["gates"]["mask_density_high"] is True
