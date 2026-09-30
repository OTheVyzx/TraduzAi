from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from typesetter import renderer as renderer_mod


def test_owner_render_accepts_observable_effect_abstention_when_core_is_valid():
    canvas = np.zeros((120, 240, 3), dtype=np.uint8)
    image = Image.fromarray(canvas.copy(), mode="RGB")
    block = {
        "owner_id": "owner_effect_fallback",
        "translated": "TEXTO LEGIVEL",
        "translated_payload": "TEXTO LEGIVEL",
        "render_safe_polygon_page": [[20, 20], [220, 20], [220, 100], [20, 100]],
        "paint_safe_polygon_page": [[20, 20], [220, 20], [220, 100], [20, 100]],
        "source_font_bounds_px": [20, 20],
        "container_font_bounds_px": [20, 20],
        "_owner_render_mode": True,
    }

    def plan(_candidate):
        return {
            "font_size_bounds_px": [20, 20],
            "safe_text_box": [20, 20, 220, 100],
            "target_bbox": [20, 20, 220, 100],
            "max_width": 200,
            "max_height": 80,
            "font_name": "ComicNeue-Bold.ttf",
            "line_spacing_ratio": 0.2,
            "layout_profile": "dark_bubble",
            "trusted_container": True,
        }

    def render_candidate(trial_image, child, _plan, **_kwargs):
        core = np.zeros(canvas.shape[:2], dtype=np.uint8)
        core[45:75, 70:170] = 255
        rgba = np.zeros((*canvas.shape[:2], 4), dtype=np.uint8)
        rgba[core > 0] = (255, 255, 255, 255)
        ImageDraw.Draw(trial_image).rectangle((70, 45, 169, 74), fill=(255, 255, 255))
        child.update(
            {
                "fit_status": "ok",
                "render_bbox": [70, 45, 170, 75],
                "font_size_final": 20,
                "minimum_legible_font_px": 20,
            }
        )
        return renderer_mod.GlyphRasterResult(
            status="fallback",
            rgba=rgba,
            glyph_core_mask=core,
            effect_mask=np.zeros_like(core),
            glyph_core_envelope=(70, 45, 170, 75),
            effect_envelope=None,
            observed_attributes={"fill": "#FFFFFF"},
            abstained_attributes={"glow": "effect_envelope_outside_safe"},
            metrics={"core_pixels_outside_safe": 0},
            unavailable_attributes={"glow": "effect_envelope_outside_safe"},
        )

    quality = {
        "status": "ok",
        "source_scale_ratio": 1.0,
        "safe_height_occupancy": 0.5,
        "font_size_final": 20,
    }
    with (
        patch("typesetter.renderer.plan_text_layout", side_effect=plan),
        patch("typesetter.renderer._fits_in_box", return_value=True),
        patch("typesetter.renderer._render_single_text_block", side_effect=render_candidate),
        patch("typesetter.renderer._evaluate_rendered_owner_candidate", return_value=quality),
        patch("typesetter.renderer._minimum_legible_font_px", return_value=20),
    ):
        result = renderer_mod._render_single_owner_proportionally(
            image,
            block,
            pre_render_np=None,
        )

    assert result is not None
    assert result.status == "fallback"
    assert result.unavailable_attributes == {
        "glow": "effect_envelope_outside_safe"
    }
    assert block["fit_status"] == "ok"
    assert block["render_completed"] is True
