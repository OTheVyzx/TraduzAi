"""Pure proportional render-quality contracts for owner output."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _evaluate(
    *,
    source_heights: tuple[int, ...] = (36,),
    source_x_heights: tuple[float, ...] = (24.0,),
    font_size: int = 28,
    minimum: int = 12,
    line_heights: tuple[int, ...] = (27,),
    safe_bbox: list[int] = [10, 10, 90, 90],
    trusted_container: bool = True,
    source_confidence: float = 0.95,
):
    from typesetter.owner_render_quality import evaluate_owner_render_quality

    safe = np.zeros((100, 100), dtype=np.uint8)
    safe[10:90, 10:90] = 255
    glyph = np.zeros_like(safe)
    y = 12
    for height in line_heights:
        glyph[y : y + height, 25:75] = 255
        y += height + 2
    ys, xs = np.nonzero(glyph)
    render_bbox = (
        [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        if len(xs)
        else [20, 20, 20, 20]
    )
    return evaluate_owner_render_quality(
        render_bbox=render_bbox,
        safe_bbox=safe_bbox,
        safe_mask=safe,
        glyph_core_mask=glyph,
        glyph_pixels=int(np.count_nonzero(glyph)),
        font_size_final=font_size,
        minimum_legible_font_px=minimum,
        source_ink_heights_px=source_heights,
        source_x_heights_px=source_x_heights,
        source_evidence_confidence=source_confidence,
        page_width=100,
        page_height=100,
        translated_text="TEXTO TRADUZIDO",
        layout_profile="dialogue",
        trusted_container=trusted_container,
    )


def test_source_scale_rejects_16px_render_for_36px_source_ink() -> None:
    quality = _evaluate(
        source_heights=(36,),
        source_x_heights=(25.2,),
        font_size=16,
        line_heights=(16,),
    )

    assert quality.status == "under_source_scale"
    assert quality.source_scale_ratio < 0.75


def test_source_scale_rejects_16px_render_for_66px_source_ink() -> None:
    quality = _evaluate(
        source_heights=(66,),
        source_x_heights=(46.2,),
        font_size=16,
        line_heights=(16,),
    )

    assert quality.status == "under_source_scale"
    assert quality.source_scale_ratio < 0.75


def test_trusted_container_rejects_eight_percent_height_occupancy() -> None:
    quality = _evaluate(
        source_heights=(),
        source_x_heights=(),
        font_size=16,
        line_heights=(6,),
        source_confidence=0.0,
    )

    assert quality.safe_height_occupancy < 0.10
    assert quality.status == "underfilled"


def test_proportional_render_accepts_source_scale_between_075_and_135() -> None:
    quality = _evaluate(
        source_heights=(28, 30, 32),
        source_x_heights=(19.0, 20.0, 21.0),
        font_size=30,
        line_heights=(29,),
    )

    assert quality.status == "ok"
    assert 0.75 <= quality.source_scale_ratio <= 1.35


def test_missing_owner_render_quality_metrics_fail_closed() -> None:
    quality = _evaluate(
        source_heights=(),
        source_x_heights=(),
        line_heights=(),
        source_confidence=0.95,
    )

    assert quality.status == "invalid"
    assert "missing_render_ink" in quality.reasons


def test_render_quality_canonical_round_trip_is_immutable() -> None:
    from typesetter.owner_render_quality import OwnerRenderQuality

    quality = _evaluate()
    payload = quality.to_dict()
    restored = OwnerRenderQuality.from_dict(payload)
    payload["reasons"].append("tampered")
    payload["rendered_line_core_heights_px"].append(999)

    assert restored == quality
    assert "tampered" not in restored.reasons
    assert 999 not in restored.rendered_line_core_heights_px
    with pytest.raises((AttributeError, TypeError)):
        restored.reasons += ("mutation",)


def test_multiline_quality_uses_median_core_ink_height_per_line() -> None:
    quality = _evaluate(
        source_heights=(20, 24, 28),
        source_x_heights=(14.0, 16.0, 18.0),
        line_heights=(10, 24, 38),
        font_size=24,
    )

    assert quality.rendered_line_core_heights_px == (10, 24, 38)
    assert quality.render_ink_height_px == 24
    assert quality.source_ink_height_px == 24
    assert quality.source_scale_ratio == pytest.approx(1.0)


def test_render_with_pixels_outside_curved_safe_polygon_is_rejected() -> None:
    from typesetter.owner_render_quality import evaluate_owner_render_quality

    safe = np.zeros((80, 80), dtype=np.uint8)
    yy, xx = np.indices(safe.shape)
    safe[((xx - 40) ** 2 + (yy - 40) ** 2) <= 28**2] = 255
    glyph = np.zeros_like(safe)
    glyph[20:30, 15:35] = 255

    quality = evaluate_owner_render_quality(
        render_bbox=[15, 20, 35, 30],
        safe_bbox=[12, 12, 68, 68],
        safe_mask=safe,
        glyph_core_mask=glyph,
        glyph_pixels=int(np.count_nonzero(glyph)),
        font_size_final=24,
        minimum_legible_font_px=12,
        source_ink_heights_px=(24,),
        source_x_heights_px=(16.0,),
        source_evidence_confidence=0.95,
        page_width=80,
        page_height=80,
        translated_text="FORA DA CURVA",
        layout_profile="dialogue",
        trusted_container=True,
    )

    assert quality.outside_safe_pixels > 0
    assert quality.containment_status == "outside_safe_region"
    assert quality.status == "outside_safe_region"
