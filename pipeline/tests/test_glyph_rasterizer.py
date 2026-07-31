"""Unified text/SFX V2 glyph rasterizer contracts."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import cv2
import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.glyph_rasterizer import rasterize_v2_glyph_layers  # noqa: E402


def _glyph(shape=(120, 260)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x in (55, 92, 129, 166):
        cv2.rectangle(mask, (x, 42), (x + 22, 78), 255, -1)
    return mask


def _safe(shape=(120, 260), inset=8) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[inset : shape[0] - inset, inset : shape[1] - inset] = 255
    return mask


def test_renderer_applies_v2_tracking_slant_and_width() -> None:
    core = _glyph()
    result = rasterize_v2_glyph_layers(
        core,
        _safe(),
        {
            "fill": "#202020",
            "tracking_xh": 0.18,
            "slant_tangent": 0.22,
            "width_scale": 0.72,
        },
        source_x_height_px=36.0,
    )

    assert result.status == "applied"
    assert result.applied_attributes["tracking_xh"] == 0.18
    assert result.applied_attributes["slant_tangent"] == 0.22
    assert result.applied_attributes["width_scale"] == 0.72
    assert not np.array_equal(result.glyph_core_mask, core)


def test_effect_pixels_scale_with_x_height() -> None:
    style = {"fill": "#FFFFFF", "stroke": {"color": "#202060", "width_xh": 0.12}}
    small = rasterize_v2_glyph_layers(
        _glyph(), _safe(), style, source_x_height_px=25.0
    )
    large = rasterize_v2_glyph_layers(
        _glyph(), _safe(), style, source_x_height_px=50.0
    )

    assert large.metrics["stroke_width_px"] == 2 * small.metrics["stroke_width_px"]
    assert np.count_nonzero(large.effect_mask) > np.count_nonzero(small.effect_mask)


def test_renderer_applies_multistroke_shadow_glow_and_gradient_layers() -> None:
    result = rasterize_v2_glyph_layers(
        _glyph(),
        _safe(inset=2),
        {
            "fill": "#FFFFFF",
            "gradient": ["#FFE040", "#E04080"],
            "multistroke": [
                {"color": "#FFFFFF", "width_xh": 0.08},
                {"color": "#202060", "width_xh": 0.16},
            ],
            "shadow": {"color": "#101020", "offset_xh": [0.10, 0.12]},
            "glow": {"color": "#80C0FF", "width_xh": 0.10},
        },
        source_x_height_px=32.0,
    )

    assert result.status == "applied"
    assert {"multistroke", "shadow", "glow", "gradient"}.issubset(result.applied_attributes)
    colors = np.unique(result.rgba[result.rgba[:, :, 3] > 0, :3].reshape(-1, 3), axis=0)
    assert len(colors) >= 4


def test_renderer_never_draws_outside_owner_safe_polygon() -> None:
    safe = np.zeros((120, 260), dtype=np.uint8)
    polygon = np.asarray([[35, 15], [225, 15], [205, 105], [55, 105]], dtype=np.int32)
    cv2.fillPoly(safe, [polygon], 255)
    result = rasterize_v2_glyph_layers(
        _glyph(), safe, {"fill": "#202020", "glow": {"color": "#88CCFF", "width_xh": 0.18}},
        source_x_height_px=36.0,
    )

    assert not np.any((result.rgba[:, :, 3] > 0) & (safe == 0))


def test_style_effect_envelope_never_clips_core_or_high_confidence_effects() -> None:
    result = rasterize_v2_glyph_layers(
        _glyph(), _safe(inset=2),
        {"fill": "#FFFFFF", "stroke": {"color": "#000000", "width_xh": 0.10}, "glow": {"color": "#80C0FF", "width_xh": 0.08}},
        source_x_height_px=30.0,
    )

    assert result.status == "applied"
    assert result.metrics["core_pixels_outside_safe"] == 0
    assert result.metrics["effect_pixels_outside_safe"] == 0
    assert result.glyph_core_envelope is not None
    assert result.effect_envelope is not None


def test_unsafe_effect_abstains_without_reflow_or_payload_change() -> None:
    core = _glyph()
    safe = cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=1)
    payload = {"translated": "CORPO INTEIRO", "line_breaks": [5], "font_size": 28}
    before = copy.deepcopy(payload)
    baseline = rasterize_v2_glyph_layers(
        core, safe, {"fill": "#202020"}, source_x_height_px=30.0
    )
    with_unsafe_glow = rasterize_v2_glyph_layers(
        core, safe,
        {"fill": "#202020", "glow": {"color": "#88CCFF", "width_xh": 0.35}},
        source_x_height_px=30.0,
    )

    assert with_unsafe_glow.status == "fallback"
    assert with_unsafe_glow.abstained_attributes["glow"] == "effect_envelope_outside_safe"
    assert np.array_equal(with_unsafe_glow.glyph_core_mask, baseline.glyph_core_mask)
    assert payload == before
