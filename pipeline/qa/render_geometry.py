from __future__ import annotations

from typing import Any

import numpy as np

SFX_REVIEW_FLAGS = {
    "sfx_render_missing",
    "sfx_render_outside_source_region",
    "sfx_inpaint_damaged_art_risk",
    "sfx_translation_unknown",
    "sfx_style_low_confidence",
    "sfx_visual_candidate",
    "sfx_script_unknown",
    "sfx_mask_density_high",
}


def _coerce_bbox(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_containment_ratio(inner_bbox: Any, outer_bbox: Any) -> float | None:
    inner = _coerce_bbox(inner_bbox)
    outer = _coerce_bbox(outer_bbox)
    if not inner or not outer:
        return None

    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    overlap_w = max(0, min(ix2, ox2) - max(ix1, ox1))
    overlap_h = max(0, min(iy2, oy2) - max(iy1, oy1))
    return round((overlap_w * overlap_h) / float(inner_area), 4)


def check_render_inside_balloon(
    *,
    render_bbox: Any,
    balloon_bbox: Any,
    threshold: float = 0.85,
) -> dict[str, Any]:
    containment = bbox_containment_ratio(render_bbox, balloon_bbox)
    flags: list[str] = []
    if containment is not None and containment < float(threshold):
        flags.append("render_outside_balloon")
    return {"containment": containment, "flags": flags}


def check_render_background(
    image: Any,
    *,
    render_bbox: Any,
    balloon_bbox: Any,
    balloon_type: str,
    luma_threshold: float = 215.0,
) -> dict[str, Any]:
    render = _coerce_bbox(render_bbox)
    balloon = _coerce_bbox(balloon_bbox)
    if not render or not balloon:
        return {"background_luma": None, "flags": []}

    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    else:
        rgb = arr[:, :, :3]
    height, width = rgb.shape[:2]
    x1 = max(0, min(width, max(render[0], balloon[0])))
    y1 = max(0, min(height, max(render[1], balloon[1])))
    x2 = max(0, min(width, min(render[2], balloon[2])))
    y2 = max(0, min(height, min(render[3], balloon[3])))
    if x2 <= x1 or y2 <= y1:
        return {"background_luma": None, "flags": []}

    crop = rgb[y1:y2, x1:x2].astype(np.float32)
    luma_plane = (0.2126 * crop[:, :, 0]) + (0.7152 * crop[:, :, 1]) + (0.0722 * crop[:, :, 2])
    luma = round(float(np.median(luma_plane)), 2)
    luma_std = round(float(np.std(luma_plane)), 2)

    balloon_crop = rgb[
        max(0, min(height, balloon[1])) : max(0, min(height, balloon[3])),
        max(0, min(width, balloon[0])) : max(0, min(width, balloon[2])),
    ].astype(np.float32)
    balloon_luma = None
    balloon_luma_std = None
    flat_balloon_background = False
    containment = bbox_containment_ratio(render, balloon)
    if balloon_crop.size:
        balloon_luma_plane = (
            (0.2126 * balloon_crop[:, :, 0])
            + (0.7152 * balloon_crop[:, :, 1])
            + (0.0722 * balloon_crop[:, :, 2])
        )
        balloon_luma = round(float(np.median(balloon_luma_plane)), 2)
        balloon_luma_std = round(float(np.std(balloon_luma_plane)), 2)
        flat_balloon_background = (
            containment is not None
            and containment >= 0.98
            and balloon_luma >= 180.0
            and (
                balloon_luma_std <= 8.0
                or (luma >= 180.0 and luma_std <= 6.0)
            )
        )

    flags = []
    if luma < float(luma_threshold) and not flat_balloon_background:
        flags.append("render_on_art_suspected")
    return {
        "background_luma": luma,
        "background_luma_std": luma_std,
        "balloon_background_luma": balloon_luma,
        "balloon_background_luma_std": balloon_luma_std,
        "flat_balloon_background": flat_balloon_background,
        "flags": flags,
    }


def check_sfx_render_geometry(
    layer: dict[str, Any],
    *,
    containment_threshold: float = 0.55,
    style_confidence_threshold: float = 0.55,
) -> dict[str, Any]:
    """Return review-only QA flags for SFX inpaint/render candidates."""

    if not _is_sfx_layer(layer):
        return {"flags": [], "containment": None, "style_confidence": None}

    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    render_bbox = layer.get("render_bbox")
    source_bbox = layer.get("source_bbox") or layer.get("bbox") or layer.get("text_pixel_bbox")
    flags: list[str] = []

    render = _coerce_bbox(render_bbox)
    if render is None:
        flags.append("sfx_render_missing")

    containment = bbox_containment_ratio(render_bbox, source_bbox)
    if containment is not None and containment < float(containment_threshold):
        flags.append("sfx_render_outside_source_region")

    inpaint_allowed = sfx.get("inpaint_allowed")
    gate = sfx.get("inpaint_gate") if isinstance(sfx.get("inpaint_gate"), dict) else {}
    gate_flags = {
        str(flag)
        for flag in (
            sfx.get("inpaint_flags")
            or sfx.get("gate_flags")
            or gate.get("qa_flags")
            or gate.get("flags")
            or []
        )
        if flag
    }
    layer_flags = {str(flag) for flag in (layer.get("qa_flags") or []) if flag}
    sfx_flags = {str(flag) for flag in (sfx.get("qa_flags") or []) if flag}
    all_flags = layer_flags | sfx_flags | gate_flags
    if inpaint_allowed is False or all_flags & {"complex_background", "art_overlap", "mask_too_dense", "sfx_mask_density_high"}:
        flags.append("sfx_inpaint_damaged_art_risk")

    if (
        sfx.get("review_required")
        or str(layer.get("route_action") or "").strip().lower() == "review_required"
        or str(layer.get("script") or "").strip().lower() in {"unknown", "cjk_unknown"}
        or all_flags & {"sfx_script_unknown", "unknown_sfx", "empty_sfx", "non_hangul_sfx"}
        or str(sfx.get("adaptation_status") or "").lower() in {
        "unknown",
        "review",
        "needs_review",
        }
    ):
        flags.append("sfx_translation_unknown")

    style_payload = sfx.get("style") if isinstance(sfx.get("style"), dict) else {}
    style_confidence = _as_float(
        sfx.get("style_confidence")
        or style_payload.get("confidence")
        or layer.get("style_confidence")
    )
    style_flags = {str(flag) for flag in (sfx.get("style_flags") or []) if flag}
    style_flags.update(str(flag) for flag in (style_payload.get("qa_flags") or []) if flag)
    if style_confidence is not None and style_confidence < float(style_confidence_threshold):
        flags.append("sfx_style_low_confidence")
    elif style_flags & {"low_confidence", "style_low_confidence"}:
        flags.append("sfx_style_low_confidence")

    return {
        "flags": _dedupe_flags(flags),
        "containment": containment,
        "style_confidence": style_confidence,
    }


def _is_sfx_layer(layer: dict[str, Any]) -> bool:
    route = str(layer.get("route") or layer.get("translation_route") or "")
    kind = str(layer.get("content_class") or layer.get("tipo") or layer.get("type") or "").lower()
    return (
        kind in {"sfx", "sound_effect", "onomatopoeia"}
        or route == "translate_sfx_inpaint_render"
        or isinstance(layer.get("sfx"), dict)
    )


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _dedupe_flags(flags: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for flag in flags:
        if flag in seen:
            continue
        seen.add(flag)
        result.append(flag)
    return result
