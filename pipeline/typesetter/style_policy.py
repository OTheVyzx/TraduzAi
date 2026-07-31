"""Style policy for automatic typesetting output.

The automatic pipeline should produce conservative, readable defaults. Manual
editor choices are handled outside this module and must not be normalized here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Sequence

import numpy as np

from typesetter.style_contract import StyleEvidenceV2


CANONICAL_AUTO_FONT = "ComicNeue-Bold.ttf"
SOURCE_STYLE_CONFIDENCE_THRESHOLD = 0.70
SOURCE_STYLE_SAFE_FIELDS = {
    "fonte",
    "cor",
    "cor_gradiente",
    "contorno",
    "contorno_px",
    "glow",
    "glow_cor",
    "glow_px",
    "sombra",
    "sombra_cor",
    "sombra_offset",
    "curva",
    "curva_direcao",
    "curva_intensidade",
    "rotacao",
}
OWNER_STYLE_FORBIDDEN_FIELDS = frozenset(
    {
        "id",
        "text_id",
        "owner_id",
        "page_id",
        "text",
        "original",
        "raw_ocr",
        "normalized_ocr",
        "normalized_text_final",
        "translated",
        "traduzido",
        "component_ids",
        "observation_ids",
        "selected_observation_ids",
        "semantic_role",
        "source_payload",
        "translated_payload",
        "disposition",
        "state",
        "route_action",
        "execution_tile_id",
        "action_mask_ref",
        "layout_region_ids",
        "layout_regions",
        "coordinate_space",
        "bbox",
        "source_bbox",
        "text_pixel_bbox",
        "layout_bbox",
        "balloon_bbox",
        "bubble_mask_bbox",
        "safe_text_box",
        "layout_safe_bbox",
        "owner_bbox_page",
        "component_geometry_sha256",
        "action_mask",
        "action_mask_sha256",
        "protected_art_mask",
        "protected_art_mask_sha256",
        "changed_mask",
        "changed_mask_sha256",
        "before_sha256",
        "after_sha256",
        "render_safe_polygon_page",
        "render_safe_polygon_sha256",
        "glyph_mask",
        "glyph_mask_sha256",
    }
)
AUTO_VISUAL_STYLE_FIELDS = frozenset(
    SOURCE_STYLE_SAFE_FIELDS
    | {
        "tipo",
        "layout_profile",
        "style_origin",
        "style_confidence",
        "style_source",
        "tamanho",
        "font_family",
        "bold",
        "italico",
        "alinhamento",
        "force_upper",
        "line_spacing_ratio",
        "vertical_bias_px",
        "horizontal_bias_px",
    }
)


def style_evidence_v2_shadow_policy(evidence: StyleEvidenceV2) -> dict[str, object]:
    """Expose the v2 rollout decision without changing renderer behavior in shadow mode."""
    del evidence
    return {
        "apply_to_renderer": False,
        "reason": "shadow_mode_no_runtime_behavior_change",
        "schema_version": 2,
    }


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    def channel(value: int) -> float:
        value = max(0, min(255, int(value))) / 255.0
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def auto_text_color_for_background(background_rgb: tuple[int, int, int]) -> str:
    # Conservative for manga/manhwa: if contrast is ambiguous, use black.
    return "#000000" if relative_luminance(background_rgb) >= 0.25 else "#FFFFFF"


def source_style_copy_allowed(
    origin_or_mapping: str | Mapping[str, object] | None,
    confidence: object | None = None,
) -> bool:
    """Return whether source style is explicitly eligible for visual copying."""

    if isinstance(origin_or_mapping, Mapping):
        origin = origin_or_mapping.get("style_origin")
        confidence_value = (
            origin_or_mapping.get("style_confidence")
            if confidence is None
            else confidence
        )
    else:
        origin = origin_or_mapping
        confidence_value = confidence

    if str(origin or "").strip().lower() != "source_detected":
        return False
    if isinstance(confidence_value, bool):
        return False
    try:
        confidence_number = float(confidence_value)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(confidence_number)
        and SOURCE_STYLE_CONFIDENCE_THRESHOLD <= confidence_number <= 1.0
    )


def _has_confident_source_style(style: dict) -> bool:
    return source_style_copy_allowed(style)


def _force_black_overrides_source_style(style: dict, force_black_text: bool) -> bool:
    if not force_black_text:
        return False
    if style.get("tipo") == "sfx":
        return False
    if style.get("glow") or style.get("contorno") or style.get("contorno_px") or style.get("cor_gradiente"):
        return False

    layout_profile = style.get("layout_profile")
    return layout_profile in (None, "", "white_balloon")


def normalize_auto_typesetting_style(
    style: dict | None,
    background_rgb: tuple[int, int, int],
    *,
    force_black_text: bool = False,
) -> dict:
    normalized = {
        key: deepcopy(value)
        for key, value in dict(style or {}).items()
        if key in AUTO_VISUAL_STYLE_FIELDS
        and key not in OWNER_STYLE_FORBIDDEN_FIELDS
    }
    preserve_source_style = _has_confident_source_style(normalized)
    force_black_overrides_source = _force_black_overrides_source_style(normalized, force_black_text)

    normalized["fonte"] = CANONICAL_AUTO_FONT
    normalized["cor"] = "#000000" if force_black_text else auto_text_color_for_background(background_rgb)
    normalized["cor_gradiente"] = []
    normalized["contorno"] = ""
    normalized["contorno_px"] = 0
    normalized["glow"] = False
    normalized["glow_cor"] = ""
    normalized["glow_px"] = 0
    normalized["sombra"] = False
    normalized["sombra_cor"] = ""
    normalized["sombra_offset"] = [0, 0]
    normalized["curva"] = False
    normalized["curva_direcao"] = ""
    normalized["curva_intensidade"] = 0.0
    normalized["bold"] = True
    normalized.setdefault("italico", False)
    normalized.setdefault("rotacao", 0)
    normalized.setdefault("alinhamento", "center")
    normalized.setdefault("force_upper", False)

    if preserve_source_style:
        source_style = style or {}
        for field in SOURCE_STYLE_SAFE_FIELDS:
            if field == "cor" and force_black_overrides_source:
                continue
            if field in source_style:
                normalized[field] = deepcopy(source_style[field])

    return normalized


def _coerce_bbox(bbox: Sequence[int | float] | None) -> tuple[int, int, int, int] | None:
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def sample_text_background_rgb(
    image_rgb: np.ndarray,
    bbox: Sequence[int | float] | None,
) -> tuple[int, int, int]:
    if image_rgb is None or image_rgb.ndim < 3 or image_rgb.shape[2] < 3:
        return (255, 255, 255)
    coerced = _coerce_bbox(bbox)
    if coerced is None:
        return (255, 255, 255)

    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = coerced
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return (255, 255, 255)

    box_w = x2 - x1
    box_h = y2 - y1
    margin = max(2, int(round(min(box_w, box_h) * 0.08)))
    if box_w > margin * 2 + 2 and box_h > margin * 2 + 2:
        x1 += margin
        x2 -= margin
        y1 += margin
        y2 -= margin

    crop = image_rgb[y1:y2, x1:x2, :3]
    if crop.size == 0:
        return (255, 255, 255)

    pixels = crop.reshape(-1, 3).astype(np.float32)
    if pixels.shape[0] >= 32:
        luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
        low = np.percentile(luminance, 10)
        high = np.percentile(luminance, 95)
        filtered = pixels[(luminance >= low) & (luminance <= high)]
        if filtered.shape[0] >= max(8, pixels.shape[0] // 8):
            pixels = filtered

    rgb = np.median(pixels, axis=0)
    return tuple(int(max(0, min(255, round(float(value))))) for value in rgb)  # type: ignore[return-value]
