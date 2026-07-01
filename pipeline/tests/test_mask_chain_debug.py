import json
import sys
from pathlib import Path

import cv2
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


def test_mask_debug_prefers_real_bubble_mask_over_balloon_bbox():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((40, 40, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((40, 40), dtype=np.uint8)
    raw_mask[16:24, 16:24] = 255
    expanded_mask = raw_mask.copy()
    bubble_mask = np.zeros((40, 40), dtype=np.uint8)
    bubble_mask[8:18, 8:18] = 255
    ocr_page = {
        "band_id": "page_001_band_001",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [16, 16, 24, 24],
                "line_polygons": [[[16, 16], [24, 16], [24, 24], [16, 24]]],
                "balloon_bbox": [0, 0, 40, 40],
                "bubble_mask": bubble_mask,
                "bubble_mask_source": "real_bubble_mask",
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    assert decision["used_real_bubble_mask"] is True
    assert decision["used_balloon_bbox_fallback"] is False
    assert decision["bubble_mask_source"] == "real_bubble_mask"
    assert decision["balloon_mask_pixels"] == int(np.count_nonzero(bubble_mask))
    assert int(np.count_nonzero(images["04_balloon_mask.png"])) == int(np.count_nonzero(bubble_mask))


def test_mask_debug_treats_text_rect_fallback_as_derived_bubble_mask():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((40, 60, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((40, 60), dtype=np.uint8)
    raw_mask[18:28, 18:48] = 255
    bubble_mask = np.ones((18, 40), dtype=np.uint8) * 255
    ocr_page = {
        "band_id": "page_001_band_001",
        "texts": [
            {
                "id": "ocr_note",
                "bbox": [14, 14, 54, 32],
                "line_polygons": [[[18, 18], [48, 18], [48, 28], [18, 28]]],
                "bubble_mask": bubble_mask,
                "bubble_mask_bbox": [10, 12, 50, 30],
                "bubble_mask_source": "text_rect_fallback",
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=raw_mask,
        final_mask=raw_mask,
    )

    assert decision["used_derived_bubble_mask"] is True
    assert decision["bubble_mask_source"] == "text_rect_fallback"
    assert decision["balloon_mask_pixels"] == 18 * 40
    assert int(np.count_nonzero(images["04_balloon_mask.png"])) == 18 * 40


def test_mask_debug_forces_rect_for_dark_panel_with_stale_ellipse_mask():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.zeros((360, 720, 3), dtype=np.uint8)
    raw_mask = np.zeros((360, 720), dtype=np.uint8)
    raw_mask[165:236, 113:259] = 255
    stale_bubble = np.zeros((360, 720), dtype=np.uint8)
    cv2.ellipse(stale_bubble, (194, 227), (156, 105), 0, 0, 360, 255, -1)
    ocr_page = {
        "band_id": "page_002_band_003",
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_002_band_003",
                "bbox": [113, 165, 259, 236],
                "text_pixel_bbox": [113, 165, 259, 236],
                "line_polygons": [[[113, 165], [259, 165], [259, 236], [113, 236]]],
                "bubble_mask": stale_bubble,
                "bubble_mask_bbox": [37, 122, 350, 332],
                "balloon_bbox": [37, 122, 350, 332],
                "bubble_mask_source": "image_dark_bubble_mask",
                "bubble_mask_ellipse": {"center": [193.5, 227.0], "axes": [313.0, 210.0], "angle": 0.0},
                "card_panel_text_context": True,
                "qa_flags": ["dark_bubble_ellipse_bbox_mask"],
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=raw_mask,
        final_mask=raw_mask,
    )

    assert decision["bubble_mask_source"] == "image_dark_panel_mask"
    assert decision["balloon_mask_pixels"] == (302 - 70) * (265 - 136)
    assert int(np.count_nonzero(images["04_balloon_mask.png"])) == decision["balloon_mask_pixels"]
    assert int(images["04_balloon_mask.png"][136, 70]) == 255
    assert int(images["04_balloon_mask.png"][264, 301]) == 255
    assert int(images["04_balloon_mask.png"][122, 37]) == 0
    assert int(images["04_balloon_mask.png"][331, 349]) == 0


def test_mask_debug_does_not_treat_unsourced_bubble_mask_as_real():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((40, 40, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((40, 40), dtype=np.uint8)
    raw_mask[16:24, 16:24] = 255
    bubble_mask = np.zeros((40, 40), dtype=np.uint8)
    bubble_mask[8:30, 8:30] = 255
    ocr_page = {
        "band_id": "page_001_band_001",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [16, 16, 24, 24],
                "line_polygons": [[[16, 16], [24, 16], [24, 24], [16, 24]]],
                "bubble_mask": bubble_mask,
                "bubble_mask_bbox": [8, 8, 30, 30],
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=raw_mask,
        final_mask=raw_mask,
    )

    assert decision["used_real_bubble_mask"] is False
    assert decision["used_image_bubble_mask"] is False
    assert decision["used_derived_bubble_mask"] is False
    assert decision["bubble_mask_source"] == "missing"
    assert int(np.count_nonzero(images["04_balloon_mask.png"])) == 0


def test_mask_debug_reports_accepted_image_bubble_mask_as_image_action_mask_not_real():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((40, 40, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((40, 40), dtype=np.uint8)
    raw_mask[16:24, 16:24] = 255
    expanded_mask = raw_mask.copy()
    bubble_mask = np.zeros((20, 24), dtype=np.uint8)
    bubble_mask[2:18, 3:21] = 255
    ocr_page = {
        "band_id": "page_001_band_001",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [16, 16, 24, 24],
                "line_polygons": [[[16, 16], [24, 16], [24, 24], [16, 24]]],
                "balloon_bbox": [0, 0, 40, 40],
                "bubble_mask": bubble_mask,
                "bubble_mask_bbox": [6, 8, 30, 28],
                "bubble_mask_source": "image_white_bubble_mask",
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    assert decision["used_real_bubble_mask"] is False
    assert decision["used_image_bubble_mask"] is True
    assert decision["used_derived_bubble_mask"] is False
    assert decision["bubble_mask_source"] == "image_white_bubble_mask"
    assert "bbox_fallback_bubble_mask" not in decision["flags"]
    assert decision["balloon_mask_pixels"] == int(np.count_nonzero(bubble_mask))
    assert int(np.count_nonzero(images["04_balloon_mask.png"])) == int(np.count_nonzero(bubble_mask))


def test_mask_debug_reconstructs_dark_bubble_from_bbox_when_ellipse_missing():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((80, 160, 3), 8, dtype=np.uint8)
    raw_mask = np.zeros((80, 160), dtype=np.uint8)
    raw_mask[32:42, 62:100] = 255
    ocr_page = {
        "band_id": "page_002_band_011",
        "texts": [
            {
                "id": "direct_paddle_reocr_001",
                "bbox": [62, 32, 100, 42],
                "line_polygons": [[[62, 32], [100, 32], [100, 42], [62, 42]]],
                "bubble_mask_bbox": [20, 10, 140, 70],
                "bubble_mask_source": "image_dark_bubble_mask",
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=raw_mask,
        final_mask=raw_mask,
    )

    assert decision["used_image_bubble_mask"] is True
    assert decision["bubble_mask_source"] == "image_dark_bubble_mask"
    assert decision["balloon_mask_pixels"] > 0
    assert int(np.count_nonzero(images["04_balloon_mask.png"])) == decision["balloon_mask_pixels"]


def test_mask_debug_recovered_dark_bubble_prefers_wide_balloon_bbox_over_tight_mask_bbox():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((120, 260, 3), 8, dtype=np.uint8)
    raw_mask = np.zeros((120, 260), dtype=np.uint8)
    raw_mask[52:64, 44:216] = 255
    tight_bubble_mask = np.zeros((20, 50), dtype=np.uint8)
    tight_bubble_mask[:, :] = 255
    ocr_page = {
        "band_id": "page_002_band_011",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [42, 48, 216, 68],
                "text_pixel_bbox": [42, 48, 216, 68],
                "line_polygons": [[[42, 48], [216, 48], [216, 68], [42, 68]]],
                "bubble_mask": tight_bubble_mask,
                "bubble_mask_bbox": [104, 48, 154, 68],
                "balloon_bbox": [16, 20, 240, 100],
                "bubble_mask_source": "image_dark_bubble_mask",
                "qa_flags": ["candidate_crop_direct_paddle_reocr", "detected_dark_bubble_without_text_reocr"],
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=raw_mask,
        final_mask=raw_mask,
    )

    assert decision["used_image_bubble_mask"] is True
    assert int(np.count_nonzero(images["04_balloon_mask.png"][:, :80])) > 0
    assert int(np.count_nonzero(images["04_balloon_mask.png"][:, 180:])) > 0


def test_mask_debug_writes_per_text_balloon_masks_for_multi_text_band(tmp_path):
    from debug_tools.masks import write_mask_chain_debug_artifacts

    image = np.full((80, 160, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((80, 160), dtype=np.uint8)
    raw_mask[20:28, 22:44] = 255
    raw_mask[50:58, 108:132] = 255
    expanded_mask = raw_mask.copy()
    first_bubble = np.zeros((80, 160), dtype=np.uint8)
    first_bubble[10:38, 10:58] = 255
    second_bubble = np.zeros((80, 160), dtype=np.uint8)
    second_bubble[40:70, 96:148] = 255
    ocr_page = {
        "band_id": "page_001_band_007",
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_007",
                "bbox": [22, 20, 44, 28],
                "line_polygons": [[[22, 20], [44, 20], [44, 28], [22, 28]]],
                "bubble_mask": first_bubble,
                "bubble_mask_source": "image_white_bubble_mask",
            },
            {
                "id": "ocr_002",
                "trace_id": "ocr_002@page_001_band_007",
                "bbox": [108, 50, 132, 58],
                "line_polygons": [[[108, 50], [132, 50], [132, 58], [108, 58]]],
                "bubble_mask": second_bubble,
                "bubble_mask_source": "image_white_bubble_mask",
            },
        ],
    }
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="test")

    write_mask_chain_debug_artifacts(
        recorder,
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    first_json = tmp_path / "debug" / "e2e" / "06_mask_segmentation" / "page_001_band_007" / "per_text" / "ocr_001" / "mask_decision.json"
    second_json = tmp_path / "debug" / "e2e" / "06_mask_segmentation" / "page_001_band_007" / "per_text" / "ocr_002" / "mask_decision.json"
    assert first_json.exists()
    assert second_json.exists()
    first_decision = json.loads(first_json.read_text(encoding="utf-8"))
    second_decision = json.loads(second_json.read_text(encoding="utf-8"))
    root_decision = json.loads(
        (
            tmp_path
            / "debug"
            / "e2e"
            / "06_mask_segmentation"
            / "page_001_band_007"
            / "mask_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert root_decision["decision_scope"] == "band_aggregate_debug_only"
    assert root_decision["actionable"] is False
    assert root_decision["per_text_decision_paths"] == [
        "per_text/ocr_001/mask_decision.json",
        "per_text/ocr_002/mask_decision.json",
    ]
    assert first_decision["text_id"] == "ocr_001"
    assert second_decision["text_id"] == "ocr_002"
    assert first_decision["text_ids"] == ["ocr_001"]
    assert second_decision["text_ids"] == ["ocr_002"]
    assert first_decision["balloon_mask_pixels"] == int(np.count_nonzero(first_bubble))
    assert second_decision["balloon_mask_pixels"] == int(np.count_nonzero(second_bubble))


def test_mask_debug_rejects_flagged_image_bubble_mask_as_reference():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((40, 40, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((40, 40), dtype=np.uint8)
    raw_mask[16:24, 16:24] = 255
    expanded_mask = raw_mask.copy()
    bubble_mask = np.zeros((20, 24), dtype=np.uint8)
    bubble_mask[2:18, 3:21] = 255
    ocr_page = {
        "band_id": "page_001_band_001",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [16, 16, 24, 24],
                "line_polygons": [[[16, 16], [24, 16], [24, 24], [16, 24]]],
                "balloon_bbox": [0, 0, 40, 40],
                "bubble_mask": bubble_mask,
                "bubble_mask_bbox": [6, 8, 30, 28],
                "bubble_mask_source": "image_white_bubble_mask",
                "qa_flags": ["debug_derived_bubble_mask_rejected"],
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    assert decision["used_real_bubble_mask"] is False
    assert decision["used_image_bubble_mask"] is False
    assert decision["used_balloon_clip"] is False
    assert decision["bubble_mask_source"] == "rejected_derived_bubble_mask"
    assert decision["bubble_mask_rejection_reason"] == "debug_derived_bubble_mask_rejected"
    assert int(np.count_nonzero(images["04_balloon_mask.png"])) == 0


def test_mask_debug_falls_back_to_balloon_bbox_when_real_bubble_mask_missing():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((40, 40, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((40, 40), dtype=np.uint8)
    raw_mask[16:24, 16:24] = 255
    expanded_mask = raw_mask.copy()
    ocr_page = {
        "band_id": "page_001_band_001",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [16, 16, 24, 24],
                "line_polygons": [[[16, 16], [24, 16], [24, 24], [16, 24]]],
                "balloon_bbox": [5, 6, 25, 26],
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    assert decision["used_real_bubble_mask"] is False
    assert decision["used_balloon_bbox_fallback"] is True
    assert decision["bubble_mask_source"] == "balloon_bbox_fallback"
    assert decision["balloon_mask_pixels"] == 400
    assert int(np.count_nonzero(images["04_balloon_mask.png"])) == 400


def test_mask_debug_does_not_fallback_to_rejected_derived_balloon_bbox():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((40, 40, 3), 240, dtype=np.uint8)
    raw_mask = np.zeros((40, 40), dtype=np.uint8)
    raw_mask[16:24, 16:24] = 255
    expanded_mask = raw_mask.copy()
    ocr_page = {
        "band_id": "page_001_band_001",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [16, 16, 24, 24],
                "line_polygons": [[[16, 16], [24, 16], [24, 24], [16, 24]]],
                "balloon_bbox": [0, 0, 40, 40],
                "bubble_mask_source": "derived_white_crop_rejected",
                "bubble_mask_error": "derived_mask_not_anchored_to_text",
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    assert decision["used_real_bubble_mask"] is False
    assert decision["used_balloon_bbox_fallback"] is False
    assert decision["bubble_mask_source"] == "rejected_derived_bubble_mask"
    assert decision["bubble_mask_rejection_reason"] == "derived_mask_not_anchored_to_text"
    assert decision["balloon_mask_pixels"] == 0
    assert int(np.count_nonzero(images["04_balloon_mask.png"])) == 0


def test_mask_debug_rejects_flagged_derived_mask_without_bbox_fallback():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((48, 64, 3), 255, dtype=np.uint8)
    raw_mask = np.zeros((48, 64), dtype=np.uint8)
    raw_mask[20:28, 26:38] = 255
    expanded_mask = raw_mask.copy()
    bad_bubble_mask = np.zeros((40, 54), dtype=np.uint8)
    bad_bubble_mask[:, :] = 255
    ocr_page = {
        "band_id": "page_002_band_008",
        "texts": [
            {
                "id": "ocr_002",
                "bbox": [26, 20, 38, 28],
                "text_pixel_bbox": [26, 20, 38, 28],
                "line_polygons": [[[26, 20], [38, 20], [38, 28], [26, 28]]],
                "balloon_bbox": [5, 4, 59, 44],
                "bubble_mask": bad_bubble_mask,
                "bubble_mask_bbox": [5, 4, 59, 44],
                "bubble_mask_source": "derived_white_crop",
                "qa_flags": ["rejected_derived_bubble_mask"],
            }
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    assert decision["used_real_bubble_mask"] is False
    assert decision["used_derived_bubble_mask"] is False
    assert decision["used_balloon_bbox_fallback"] is False


def test_mask_debug_ignores_rejected_merged_fragment_without_glyph_evidence():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((90, 160, 3), 255, dtype=np.uint8)
    raw_mask = np.zeros((90, 160), dtype=np.uint8)
    raw_mask[20:42, 92:132] = 255
    final_mask = raw_mask.copy()
    bubble_mask = np.zeros((50, 70), dtype=np.uint8)
    bubble_mask[4:46, 6:64] = 255
    ocr_page = {
        "band_id": "page_004_band_055",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [92, 20, 132, 42],
                "text_pixel_bbox": [92, 20, 132, 42],
                "line_polygons": [[[92, 20], [132, 20], [132, 42], [92, 42]]],
                "bubble_mask": bubble_mask,
                "bubble_mask_bbox": [80, 10, 150, 60],
                "bubble_mask_source": "image_contour_bubble_mask",
            },
            {
                "id": "ocr_002",
                "bbox": [0, 58, 150, 88],
                "text_pixel_bbox": [0, 58, 150, 88],
                "bubble_mask_source": "derived_white_crop_rejected",
                "bubble_mask_error": "derived_mask_not_anchored_to_text",
                "qa_flags": [],
            },
        ],
    }

    decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=final_mask,
        final_mask=final_mask,
    )

    assert decision["text_ids"] == ["ocr_001"]
    assert decision["used_real_bubble_mask"] is False
    assert decision["used_image_bubble_mask"] is True
    assert decision["bubble_mask_source"] == "image_contour_bubble_mask"
    assert int(np.count_nonzero(images["04_balloon_mask.png"])) == int(np.count_nonzero(bubble_mask))


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


def test_mask_debug_expanded_and_final_preserve_raw_inside_balloon():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((80, 140, 3), 255, dtype=np.uint8)
    raw_mask = np.zeros((80, 140), dtype=np.uint8)
    raw_mask[14:22, 106:124] = 255
    expanded_mask = np.zeros((80, 140), dtype=np.uint8)
    expanded_mask[34:46, 52:88] = 255
    bubble_mask = np.zeros((80, 140), dtype=np.uint8)
    cv2.ellipse(bubble_mask, (70, 40), (58, 28), 0, 0, 360, 255, -1)
    ocr_page = {
        "numero": 1,
        "_band_index": 6,
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_006",
                "bbox": [48, 12, 126, 48],
                "text_pixel_bbox": [48, 12, 126, 48],
                "line_polygons": [
                    [[106, 14], [124, 14], [124, 22], [106, 22]],
                    [[52, 34], [88, 34], [88, 46], [52, 46]],
                ],
                "bubble_mask": bubble_mask,
                "bubble_mask_bbox": [0, 0, 140, 80],
                "bubble_mask_source": "image_white_bubble_mask",
            }
        ],
    }

    _decision, images = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    raw_inside_bubble = (raw_mask > 0) & (bubble_mask > 0)
    assert int(np.count_nonzero(raw_inside_bubble & (images["08_expanded_text_mask.png"] == 0))) == 0
    assert int(np.count_nonzero(raw_inside_bubble & (images["09_final_inpaint_mask.png"] == 0))) == 0


def test_mask_debug_does_not_require_real_bubble_for_colored_card_text():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((80, 160, 3), [253, 194, 150], dtype=np.uint8)
    raw_mask = np.zeros((80, 160), dtype=np.uint8)
    raw_mask[28:52, 58:112] = 255
    tiny_bubble = np.zeros((80, 160), dtype=np.uint8)
    tiny_bubble[34:48, 80:106] = 255
    ocr_page = {
        "band_id": "page_001_band_010",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [58, 28, 112, 52],
                "text_pixel_bbox": [58, 28, 112, 52],
                "line_polygons": [[[58, 28], [112, 28], [112, 52], [58, 52]]],
                "background_rgb": [253, 194, 150],
                "bubble_mask": tiny_bubble,
                "bubble_mask_source": "image_white_bubble_mask",
            }
        ],
    }

    decision, _ = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=raw_mask,
        final_mask=raw_mask,
    )

    assert decision["outside_balloon_ratio"] > 0
    assert decision["gates"]["mask_outside_balloon_critical"] is False
    assert decision["gates"]["missing_real_bubble_mask"] is False
    assert "mask_outside_balloon_critical" not in decision["flags"]


def test_mask_debug_reports_weak_per_text_image_bubble_reference_even_with_valid_neighbor():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((220, 320, 3), 255, dtype=np.uint8)
    raw_mask = np.zeros((220, 320), dtype=np.uint8)
    raw_mask[74:96, 152:230] = 255
    raw_mask[150:190, 72:160] = 255
    expanded_mask = cv2.dilate(
        raw_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=1,
    )
    contour_bubble = np.zeros((220, 320), dtype=np.uint8)
    cv2.ellipse(contour_bubble, (190, 86), (95, 45), 0, 0, 360, 255, -1)
    ocr_page = {
        "band_id": "page_002_band_002",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [150, 72, 230, 98],
                "text_pixel_bbox": [152, 74, 228, 96],
                "line_polygons": [[[152, 74], [228, 74], [228, 96], [152, 96]]],
                "bubble_mask": contour_bubble,
                "bubble_mask_bbox": [80, 40, 300, 140],
                "bubble_mask_source": "image_contour_bubble_mask",
                "tipo": "fala",
            },
            {
                "id": "ocr_002",
                "bbox": [70, 148, 162, 192],
                "text_pixel_bbox": [72, 150, 160, 190],
                "line_polygons": [[[72, 150], [160, 150], [160, 190], [72, 190]]],
                "balloon_bbox": [68, 146, 164, 194],
                "bubble_mask_bbox": [68, 146, 164, 194],
                "bubble_mask_source": "image_rect_bubble_mask",
                "tipo": "fala",
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

    assert decision["bubble_mask_source"] == "image_contour_bubble_mask"
    assert "weak_image_bubble_mask_reference" in decision["flags"]
    assert decision["weak_image_bubble_mask_text_ids"] == ["ocr_002"]
    assert decision["gates"]["weak_image_bubble_mask_reference"] is True


def test_mask_debug_does_not_require_real_bubble_for_dark_glow_card_text():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((80, 160, 3), [32, 32, 40], dtype=np.uint8)
    raw_mask = np.zeros((80, 160), dtype=np.uint8)
    raw_mask[30:50, 48:118] = 255
    ocr_page = {
        "band_id": "page_001_band_011",
        "texts": [
            {
                "id": "ocr_001",
                "bbox": [48, 30, 118, 50],
                "text_pixel_bbox": [48, 30, 118, 50],
                "line_polygons": [[[48, 30], [118, 30], [118, 50], [48, 50]]],
                "background_rgb": [87, 78, 45],
                "style": {"glow": True, "glow_px": 3},
                "bubble_mask_source": "rejected_derived_bubble_mask",
                "bubble_mask_error": "derived_mask_not_anchored_to_text",
            }
        ],
    }

    decision, _ = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=raw_mask,
        final_mask=raw_mask,
    )

    assert decision["used_balloon_clip"] is False
    assert decision["gates"]["missing_real_bubble_mask"] is False
    assert "missing_real_bubble_mask" not in decision["flags"]


def test_mask_debug_does_not_require_balloon_containment_for_translator_note():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((80, 180, 3), 251, dtype=np.uint8)
    raw_mask = np.zeros((80, 180), dtype=np.uint8)
    raw_mask[24:58, 70:168] = 255
    bubble_mask = np.zeros((80, 180), dtype=np.uint8)
    bubble_mask[0:16, 0:60] = 255
    ocr_page = {
        "band_id": "page_001_band_012",
        "texts": [
            {
                "id": "ocr_001",
                "text": "T/N: HYUNGNIM IS A TERM USED FOR CALLING ONE'S BOSS.",
                "bbox": [70, 24, 168, 58],
                "text_pixel_bbox": [70, 24, 168, 58],
                "line_polygons": [[[70, 24], [168, 24], [168, 58], [70, 58]]],
                "qa_flags": ["translator_note_best_effort_render"],
                "bubble_mask": bubble_mask,
                "bubble_mask_source": "image_contour_bubble_mask",
            }
        ],
    }

    decision, _ = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=raw_mask,
        final_mask=raw_mask,
    )

    assert decision["outside_balloon_ratio"] > 0
    assert decision["gates"]["mask_outside_balloon_critical"] is False
    assert "mask_outside_balloon_critical" not in decision["flags"]


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


def test_mask_debug_flags_critical_source_glyph_ratio_for_derived_white_crop():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    raw_mask = np.zeros((100, 160), dtype=np.uint8)
    expanded_mask = np.zeros((100, 160), dtype=np.uint8)
    raw_mask[50:58, 70:90] = 255
    expanded_mask[48:60, 68:92] = 255
    bubble_mask = np.zeros((100, 160), dtype=np.uint8)
    bubble_mask[5:95, 5:155] = 255
    ocr_page = {
        "band_id": "page_003_band_023",
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_023",
                "bbox": [0, 0, 155, 95],
                "source_bbox": [0, 0, 155, 95],
                "text_pixel_bbox": [70, 50, 90, 58],
                "line_polygons": [[[70, 50], [90, 50], [90, 58], [70, 58]]],
                "bubble_mask": bubble_mask,
                "bubble_mask_source": "derived_white_crop",
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

    assert decision["used_derived_bubble_mask"] is True
    assert decision["source_glyph_area_ratio"] >= 8.0
    assert decision["gates"]["source_glyph_area_ratio_critical"] is True
    assert "source_glyph_area_ratio_critical" in decision["flags"]


def test_mask_debug_accepts_derived_card_panel_mask_with_stale_rejection_flags():
    from debug_tools.masks import build_mask_chain_debug_payload

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    raw_mask = np.zeros((80, 120), dtype=np.uint8)
    expanded_mask = np.zeros((80, 120), dtype=np.uint8)
    raw_mask[34:42, 48:72] = 255
    expanded_mask[31:45, 45:75] = 255
    ocr_page = {
        "band_id": "page_006_band_105",
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_006_band_105",
                "bbox": [48, 34, 72, 42],
                "line_polygons": [[[48, 34], [72, 34], [72, 42], [48, 42]]],
                "bubble_mask_bbox": [20, 18, 100, 62],
                "bubble_mask_source": "derived_card_panel_mask",
                "qa_flags": [
                    "missing_real_bubble_mask",
                    "debug_derived_bubble_mask_rejected",
                ],
            }
        ],
    }

    decision, artifacts = build_mask_chain_debug_payload(
        ocr_page,
        image_rgb=image,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        final_mask=expanded_mask,
    )

    assert decision["used_derived_bubble_mask"] is True
    assert decision["bubble_mask_source"] == "derived_card_panel_mask"
    assert decision["bubble_mask_rejection_reason"] is None
    assert decision["gates"]["bbox_fallback_bubble_mask"] is False
    assert "bbox_fallback_bubble_mask" not in decision["flags"]
    assert artifacts["04_balloon_mask.png"].sum() > 0
