import numpy as np
from pathlib import Path

from inpainter.region_strategy import classify_region, plan_inpaint
from PIL import Image
from inpainter import (
    _enrich_vision_blocks_from_texts_for_inpaint,
    _image_dark_bubble_is_visually_light,
    _promote_visually_light_dark_bubbles_to_white,
)


def _write_opaque_mask(path: Path) -> None:
    mask = np.zeros((20, 30, 4), dtype=np.uint8)
    mask[..., 3] = 255
    Image.fromarray(mask).save(path)


def test_region_strategy_ignores_legacy_text_classification_fields():
    region = {
        "bbox": [10, 20, 80, 60],
        "tipo": "sfx",
        "content_class": "logo",
        "balloon_type": "dark",
        "background_type": "",
        "skip_processing": True,
        "preserve_original": True,
    }

    assert classify_region(region) == "text"


def test_image_dark_bubble_visually_light_blocks_dark_fill():
    image = np.zeros((220, 320, 3), dtype=np.uint8)
    image[20:180, 60:240] = 248
    text = {
        "bubble_mask_source": "image_dark_bubble_mask",
        "bubble_mask_bbox": [60, 20, 240, 180],
        "text_pixel_bbox": [105, 78, 180, 112],
        "qa_flags": ["dark_bubble_oval_reocr"],
    }

    assert _image_dark_bubble_is_visually_light(image, text) is True
    assert "false_light_bubble_dark_fill_blocked" in text["qa_flags"]


def test_visually_light_dark_bubble_promotes_to_white_context():
    image = np.zeros((220, 320, 3), dtype=np.uint8)
    image[20:180, 60:240] = 248
    text = {
        "bubble_mask_source": "image_dark_bubble_mask",
        "bubble_mask_bbox": [0, 0, 320, 220],
        "balloon_bbox": [0, 0, 320, 220],
        "text_pixel_bbox": [105, 78, 180, 112],
        "qa_flags": ["dark_bubble_oval_reocr", "dark_panel_style_grouped"],
        "qa_metrics": {"image_dark_bubble_mask": {"mask_bbox": [60, 20, 240, 180]}},
        "style": {"cor": "#ffffff", "glow": True, "glow_px": 3, "contorno_px": 2},
    }
    ocr_page = {"texts": [text], "_vision_blocks": [text]}

    promoted = _promote_visually_light_dark_bubbles_to_white(image, ocr_page, [text])

    assert promoted == 1
    assert text["bubble_mask_source"] == "image_white_bubble_mask"
    assert text["bubble_mask_bbox"] == [60, 20, 240, 180]
    assert text["balloon_bbox"] == [60, 20, 240, 180]
    assert text["layout_profile"] == "white_balloon"
    assert text["style"]["cor"] == "#000000"
    assert text["style"]["contorno_px"] == 0
    assert "false_light_dark_bubble_promoted_to_white" in text["qa_flags"]
    assert "dark_panel_style_grouped" not in text["qa_flags"]
    assert ocr_page["_strip_false_light_dark_bubble_promoted_count"] == 1


def test_enrich_vision_blocks_uses_layout_bbox_when_text_pixel_bbox_is_stale():
    block = {
        "id": "ocr_001_002",
        "bbox": [373, 75, 698, 455],
        "text_pixel_bbox": [131, 112, 312, 231],
        "bubble_mask_source": "image_dark_bubble_mask",
    }
    text = {
        "id": "ocr_001_002",
        "bbox": [373, 75, 698, 455],
        "layout_bbox": [399, 204, 675, 335],
        "text_pixel_bbox": [131, 112, 312, 231],
        "bubble_mask_source": "image_dark_bubble_mask",
        "qa_flags": ["dark_bubble_oval_reocr"],
    }

    enriched = _enrich_vision_blocks_from_texts_for_inpaint([block], [text], 800, 620)

    assert enriched[0]["text_pixel_bbox"] == [399, 204, 675, 335]


def test_plan_inpaint_does_not_skip_sfx_by_legacy_tipo(tmp_path):
    mask_path = tmp_path / "mask.png"
    _write_opaque_mask(mask_path)

    result = plan_inpaint(
            {
                "bbox": [0, 0, 30, 20],
                "tipo": "sfx",
                "content_class": "noise",
            "skip_processing": True,
            "preserve_original": True,
        },
        str(mask_path),
    )

    assert result["region_type"] == "text"
    assert result["run"] is True
    assert result["strategy"] != "skip_without_explicit_config"
    assert result["strategy"] == "component_roi_snap8"
    assert result["roi_strategy"] == "component_roi_snap8"


def test_caption_box_keeps_preserve_borders_strategy(tmp_path):
    mask_path = tmp_path / "caption_mask.png"
    _write_opaque_mask(mask_path)

    result = plan_inpaint({"bbox": [0, 0, 30, 20], "background_type": "caption_box"}, str(mask_path))

    assert result["run"] is True
    assert result["region_type"] == "caption_box"
    assert result["strategy"] == "preserve_borders"
    assert result["roi_strategy"] is None


def test_plan_inpaint_uses_component_roi_strategy_for_text_regions(tmp_path):
    mask_path = tmp_path / "text_mask.png"
    _write_opaque_mask(mask_path)

    result = plan_inpaint({"bbox": [0, 0, 30, 20], "background_type": "text"}, str(mask_path))

    assert result["run"] is True
    assert result["roi_strategy"] == "component_roi_snap8"
    assert result["strategy"] == "component_roi_snap8"


def test_sfx_inpaint_still_blocked_without_explicit_allow(tmp_path):
    mask_path = tmp_path / "sfx_mask.png"
    _write_opaque_mask(mask_path)

    result = plan_inpaint({"bbox": [10, 10, 30, 20], "background_type": "sfx_text"}, str(mask_path))

    assert result["run"] is False
    assert result["strategy"] == "skip_without_explicit_config"
    assert "sfx_preserved" in result["qa_flags"]
    assert result["region_type"] == "sfx_text"


def test_sfx_inpaint_requires_sfx_gate_even_when_explicitly_allowed(tmp_path):
    mask_path = tmp_path / "sfx_mask.png"
    _write_opaque_mask(mask_path)

    result = plan_inpaint(
        {
            "bbox": [10, 10, 30, 20],
            "background_type": "sfx_text",
            "mask_evidence": {"kind": "ocr_pixels"},
        },
        str(mask_path),
        allow_sfx=True,
    )

    assert result["run"] is False
    assert result["strategy"] == "review_required"
    assert "sfx_mask_evidence_missing" in result["qa_flags"]
    assert result["sfx_inpaint_gate"]["allow_inpaint"] is False


def test_sfx_inpaint_uses_component_roi_when_gate_allows(tmp_path):
    mask_path = tmp_path / "sfx_mask.png"
    _write_opaque_mask(mask_path)

    result = plan_inpaint(
        {
            "bbox": [0, 0, 30, 20],
            "background_type": "sfx_text",
            "mask_evidence": {
                "kind": "sfx_glyph_mask",
                "bbox_fill_ratio": 0.14,
                "expanded_mask_pixels": 80,
            },
        },
        str(mask_path),
        allow_sfx=True,
    )

    assert result["run"] is True
    assert result["strategy"] == "lama_component_roi"
    assert result["roi_strategy"] == "component_roi_snap8"
    assert result["sfx_inpaint_gate"]["allow_inpaint"] is True
