import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfx.renderer import render_sfx_layer


def _sfx_layer(**overrides):
    layer = {
        "id": "sfx_001",
        "content_class": "sfx",
        "route_action": "translate_sfx_inpaint_render",
        "bbox": [30, 30, 150, 95],
        "sfx": {
            "adapted_text": "TUM",
            "kind": "impact",
            "inpaint_allowed": True,
            "style": {
                "fill_color": "#FFFFFF",
                "stroke_color": "#000000",
                "stroke_width_px": 2,
                "glow_color": "#FFEE88",
                "glow_width_px": 4,
                "rotation_deg": -18,
            },
        },
    }
    layer.update(overrides)
    return layer


def test_render_sfx_layer_uses_adapted_text_and_writes_render_bbox():
    page = np.full((140, 200, 3), 230, dtype=np.uint8)
    layer = _sfx_layer()

    rendered = render_sfx_layer(page, layer)

    assert rendered.shape == page.shape
    assert np.count_nonzero(rendered != page) > 0
    assert layer["translated"] == "TUM"
    assert layer["traduzido"] == "TUM"
    assert layer["fit_status"] == "ok"
    assert len(layer["render_bbox"]) == 4


def test_render_sfx_layer_does_not_render_review_required():
    page = np.full((140, 200, 3), 230, dtype=np.uint8)
    layer = _sfx_layer(route_action="review_required")

    rendered = render_sfx_layer(page, layer)

    assert np.array_equal(rendered, page)
    assert "sfx_render_missing" in layer["qa_flags"]


def test_render_sfx_layer_respects_inpaint_block():
    page = np.full((140, 200, 3), 230, dtype=np.uint8)
    layer = _sfx_layer()
    layer["sfx"]["inpaint_allowed"] = False

    rendered = render_sfx_layer(page, layer)

    assert np.array_equal(rendered, page)
    assert "sfx_render_missing" in layer["qa_flags"]
