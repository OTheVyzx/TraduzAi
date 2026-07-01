import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.text_router import route_text
from sfx.inpaint_gate import evaluate_sfx_inpaint_gate
from sfx.mask import build_sfx_glyph_mask
from sfx.renderer import render_sfx_layer


def _synthetic_sfx_page() -> tuple[np.ndarray, dict]:
    page = np.full((140, 220, 3), 238, dtype=np.uint8)
    layer = {
        "id": "sfx_contract",
        "bbox": [24, 24, 190, 110],
        "text": "\ucff5",
        "original": "\ucff5",
        "content_class": "sfx",
        "script": "hangul",
        "route_action": "translate_sfx_inpaint_render",
    }
    cv2.rectangle(page, (46, 42), (60, 88), (15, 15, 15), -1)
    cv2.rectangle(page, (46, 78), (92, 92), (15, 15, 15), -1)
    cv2.rectangle(page, (128, 38), (144, 96), (15, 15, 15), -1)
    cv2.rectangle(page, (128, 38), (172, 52), (15, 15, 15), -1)
    cv2.rectangle(page, (158, 68), (174, 96), (15, 15, 15), -1)
    return page, layer


def test_hangul_sfx_routes_to_sfx_engine():
    routed = route_text("\ucff5", tipo="sfx")

    assert routed["content_class"] == "sfx"
    assert routed["script"] == "hangul"
    assert routed["route_action"] == "translate_sfx_inpaint_render"


def test_synthetic_sfx_mask_does_not_fill_bbox():
    page, layer = _synthetic_sfx_page()

    result = build_sfx_glyph_mask(page, layer)

    assert result.mask is not None
    assert result.evidence["kind"] == "sfx_glyph_mask"
    assert result.evidence["bbox_fill_ratio"] < 0.45


def test_sfx_gate_blocks_complex_art_simulation():
    gate = evaluate_sfx_inpaint_gate(
        {
            "content_class": "sfx",
            "qa_flags": ["complex_background"],
            "mask_evidence": {
                "kind": "sfx_glyph_mask",
                "bbox_fill_ratio": 0.12,
                "expanded_mask_pixels": 120,
            },
        }
    )

    assert gate["allow_inpaint"] is False
    assert gate["strategy"] == "review_required"


def test_sfx_renderer_emits_overlay_for_safe_candidate():
    page, layer = _synthetic_sfx_page()
    layer.update(
        {
            "sfx": {
                "adapted_text": "TUM",
                "inpaint_allowed": True,
                "style": {
                    "fill_color": "#FFFFFF",
                    "stroke_color": "#25334A",
                    "stroke_width_px": 2,
                    "glow_color": "#B8D9FF",
                    "glow_width_px": 3,
                    "rotation_deg": -16,
                },
            }
        }
    )

    rendered = render_sfx_layer(page, layer)

    assert np.count_nonzero(rendered != page) > 0
    assert layer["translated"] == "TUM"
    assert layer["render_bbox"]
