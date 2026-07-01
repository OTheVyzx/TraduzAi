"""Export blocking policy for known P0 render issues."""

from __future__ import annotations

import re
from typing import Any

from qa.translation_qa import severity_for_flag


SOURCE_SCRIPT_RE = re.compile(
    r"[\u1100-\u11FF\u3000-\u303F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]"
)
EXPORT_BLOCKING_REVIEW_FLAGS = {
    "TEXT_CLIPPED",
    "TEXT_OVERFLOW",
    "untranslated_english",
    "gibberish_detected",
}
CONFIRMED_VISUAL_DAMAGE_FLAGS = {
    "TEXT_CLIPPED",
    "TEXT_OVERFLOW",
    "render_outside_balloon",
    "render_bbox_far_from_target_bbox",
    "page_space_rerender_mixed_coordinates",
    "missing_render_bbox",
    "text_residual_after_inpaint",
    "text_residual_after_inpaint_confirmed",
    "fast_fill_unverified_residual",
    "fast_fill_insufficient_coverage",
}
IGNORED_LEGACY_FLAGS = {
    "low_confidence_visual_noise",
    "cover_title_logo",
    "mask_density_high",
}
ROUTE_ACTIONS_REQUIRING_EXPORT_GATE = {
    "review_required",
    "translate_inpaint_render",
    "translate_render_only",
    "translate_sfx_inpaint_render",
}
SCANLATION_CREDIT_RE = re.compile(
    r"\b(?:"
    r"RESET\s*SCANS?|SCANSHOMEMANGA|SCANLATOR|SCANS?|UTOON|HIVE(?:TOON)?|TOON\s*\.?\s*(?:NET|NETS|NETE|net)|"
    r"NEW\s*TOKI?|[A-Z0-9]*TOK[A-Z0-9]*|[A-Z0-9]*IOKI[A-Z0-9]*|NEWTO\w*|NEVTO\w*|NEYTO\w*|NWTOK\w*|WTOK\w*|TOKLJ?G?O|"
    r"(?=[A-Z0-9]*\d)[A-Z0-9]{5,}\s*\.?\s*COM|"
    r"DISCORD|PATREON|PAYPAL|KO-?FI|DEVMAX|SUPPORT\s*US|SPECIAL\s+THANKS|"
    r"CONTACT|INVITE|RECRUITING|TRANSLATORS?|EDITORS?|TYPESETTERS?|READ\s+ON|"
    r"FOR\s+FASTER\s+UPP?ATE|CONTENTS?\s+LAB|"
    r"BRONZE|SILVER|GOLD|DIAMOND|PLATINUM|GOATBEARDS|DRAGENDAVE|SILICONMAGE|GUDPLAYUR|"
    r"ALL\s+COMICS\s+ON\s+THIS\s+WEBSITE|ORIGINAL\s+VERSION|FOR\s+THE\s+ORIGINAL"
    r")\b",
    re.IGNORECASE,
)
SCANLATION_VISUAL_REVIEW_ONLY_FLAGS = {
    "render_on_art_suspected",
    "TEXT_CLIPPED",
    "TEXT_OVERFLOW",
    "render_outside_balloon",
    "render_bbox_far_from_target_bbox",
    "text_residual_after_inpaint",
    "text_residual_after_inpaint_confirmed",
    "fast_fill_unverified_residual",
    "fast_fill_insufficient_coverage",
    "fast_fill_no_glyph_evidence",
    "fit_below_minimum_legible",
    "missing_render_bbox",
    "layout_bbox_coordinate_mismatch",
    "mask_outside_balloon_critical",
    "source_glyph_area_ratio_critical",
    "mask_outside_balloon",
    "weak_text_residual_after_inpaint",
}
SCANLATION_HARMLESS_CONTEXT_FLAGS = {
    "compact_small_text_capacity",
    "connected_lobe_boxes_missing_source_anchor_fallback",
    "ocr_art_fragment_suspected",
    "ocr_partial_low_confidence_fragment",
    "ocr_run_on_suspect",
    "rotated_text_recovery",
    "safe_text_box_recomputed",
    "tiny_bubble_inner_bbox_rejected",
}
SCANLATION_TIER_WORDS = {"BRONZE", "SILVER", "GOLD", "DIAMOND", "PLATINUM"}
SCANLATION_CONTEXTUAL_RE = re.compile(
    r"\b(?:WARNING|NOTICE|READ\s+THIS|OFFICIAL\s+SITE|FASTER\s+UPDATES?)\b",
    re.IGNORECASE,
)


def _page_id_from_identity(identity: str) -> str | None:
    match = re.search(r"(page_\d{3})_band_\d{3}", identity)
    return match.group(1) if match else None


def _band_id_from_identity(identity: str) -> str | None:
    match = re.search(r"(page_\d{3}_band_\d{3})", identity)
    return match.group(1) if match else None


def _trace_id_from_identity(identity: str) -> str | None:
    value = str(identity or "").strip()
    return value if "@" in value else None


def _text_id_from_identity(identity: str) -> str | None:
    value = str(identity or "").strip()
    if "@" in value:
        return value.split("@", 1)[0] or None
    return value or None


def _clean_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _layer_text_blob(layer: dict[str, Any]) -> str:
    return " ".join(
        str(layer.get(key) or "")
        for key in (
            "text",
            "original",
            "raw_ocr",
            "normalized_ocr",
            "normalized_text_final",
            "translated",
            "traduzido",
        )
    )


def _is_strong_scanlation_credit_layer(layer: dict[str, Any]) -> bool:
    blob = _layer_text_blob(layer)
    if not blob.strip():
        return False
    matches = [str(match or "").upper().strip() for match in SCANLATION_CREDIT_RE.findall(blob)]
    if layer.get("_scanlation_credit_context") is True and any(
        match in SCANLATION_TIER_WORDS for match in matches
    ):
        return True
    if layer.get("_scanlation_credit_context") is True and SCANLATION_CONTEXTUAL_RE.search(blob):
        return True
    if any(match and match not in SCANLATION_TIER_WORDS for match in matches):
        return True
    compact = re.sub(r"[^A-Z0-9]+", "", blob.upper())
    return bool(
        "RESETSCAN" in compact
        or "SCANSHOMEMANGA" in compact
        or "HIVETOON" in compact
        or "HIVESCAN" in compact
        or "PATREONCOM" in compact
        or "DISCORDCOM" in compact
        or "PAYPALCOM" in compact
    )


def _is_suppressed_scanlation_credit_layer(layer: dict[str, Any]) -> bool:
    reason = str(layer.get("skip_reason") or layer.get("route_reason") or "").strip().lower()
    if reason == "scanlation_credit_suppressed":
        return True
    return any(
        str(flag or "").strip().lower() == "scanlation_credit_suppressed"
        for flag in layer.get("qa_flags") or []
    )


def _layer_is_export_gate_candidate(layer: dict[str, Any]) -> bool:
    if _is_suppressed_scanlation_credit_layer(layer):
        return False
    route_action = _clean_string(layer.get("route_action"))
    if route_action:
        return (
            route_action.startswith("translate_")
            or route_action in ROUTE_ACTIONS_REQUIRING_EXPORT_GATE
        )
    return bool(
        layer.get("qa_flags")
        or layer.get("text")
        or layer.get("original")
        or layer.get("raw_ocr")
        or layer.get("translated")
    )


def _sfx_inpaint_requires_review(layer: dict[str, Any]) -> bool:
    if _clean_string(layer.get("route_action")) != "translate_sfx_inpaint_render":
        return False
    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    return sfx.get("inpaint_allowed") is False


def _sfx_review_flags(layer: dict[str, Any]) -> list[str]:
    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    flags = [str(flag) for flag in sfx.get("qa_flags") or [] if flag]
    if "sfx_inpaint_review_required" not in flags:
        flags.insert(0, "sfx_inpaint_review_required")
    return list(dict.fromkeys(flags))


def _first_list_string(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    for item in value:
        text = _clean_string(item)
        if text:
            return text
    return None


def _resolve_trace_id(layer: dict[str, Any], text_id: str | None, band_id: str | None) -> str | None:
    for value in (
        layer.get("trace_id"),
        layer.get("text_instance_id"),
        _first_list_string(layer.get("source_trace_ids")),
        _first_list_string(layer.get("_source_trace_ids")),
        _first_list_string(layer.get("trace_ids")),
    ):
        trace_id = _clean_string(value)
        if trace_id and "@" in trace_id:
            return trace_id
    if text_id and band_id:
        return f"{text_id}@{band_id}"
    return None


def _synthetic_band_id(page_id: str, layer_index: int) -> str:
    return f"{page_id}_layer_{layer_index:03d}"


def _stable_rel_path(value: Any) -> str | None:
    if not value:
        return None
    rel_path = str(value).strip().replace("\\", "/")
    if not rel_path or re.match(r"^[A-Za-z]:/", rel_path) or rel_path.startswith("/"):
        return None
    return rel_path


def _translated_page_ref(page: dict[str, Any]) -> str | None:
    for value in (
        page.get("arquivo_traduzido"),
        page.get("translated_path"),
        page.get("output_path"),
    ):
        rel_path = _stable_rel_path(value)
        if rel_path:
            return rel_path
    image_layers = page.get("image_layers")
    if isinstance(image_layers, dict):
        rendered = image_layers.get("rendered")
        if isinstance(rendered, dict):
            return _stable_rel_path(rendered.get("path"))
    return None


def _layer_qa_flags(layer: dict[str, Any]) -> set[str]:
    flags = {str(flag) for flag in layer.get("qa_flags") or [] if flag}
    top_level_render_flags = set(flags & {"TEXT_CLIPPED", "TEXT_OVERFLOW", "render_outside_balloon"})
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    render_fit_flags = {str(flag) for flag in render_fit.get("flags") or [] if flag}
    render_fit_stale = _render_fit_evidence_is_stale(layer)
    if _final_render_text_flags_are_review_only(layer):
        flags.difference_update({"TEXT_CLIPPED", "TEXT_OVERFLOW", "render_outside_balloon"})
    if not render_fit_stale:
        flags.update(render_fit_flags)
    if _final_render_text_flags_are_review_only(layer) and top_level_render_flags:
        flags.difference_update(top_level_render_flags)
    flags.difference_update(IGNORED_LEGACY_FLAGS)
    if layer.get("ocr_repair_status") != "repair_failed":
        flags.discard("ocr_truncated_or_joined")
    return flags


def _bbox4(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _bbox_area(value: list[int] | None) -> int:
    if value is None:
        return 0
    return max(0, value[2] - value[0]) * max(0, value[3] - value[1])


def _bbox_contains(outer: list[int] | None, inner: list[int] | None, margin: int = 4) -> bool:
    if outer is None or inner is None:
        return False
    return bool(
        inner[0] >= outer[0] - margin
        and inner[1] >= outer[1] - margin
        and inner[2] <= outer[2] + margin
        and inner[3] <= outer[3] + margin
    )


def _containment_target_bboxes(layer: dict[str, Any]) -> list[list[int]]:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    dark_bubble_metrics = qa_metrics.get("image_dark_bubble_mask") if isinstance(qa_metrics.get("image_dark_bubble_mask"), dict) else {}
    dark_panel_metrics = qa_metrics.get("image_dark_panel_mask") if isinstance(qa_metrics.get("image_dark_panel_mask"), dict) else {}
    candidates = [
        _bbox4(layer.get("bubble_inner_bbox")),
        _bbox4(layer.get("bubble_mask_bbox")),
        _bbox4(dark_bubble_metrics.get("mask_bbox")),
        _bbox4(dark_panel_metrics.get("mask_bbox")),
        _bbox4(layer.get("target_bbox")),
        _bbox4(layer.get("capacity_bbox")),
        _bbox4(layer.get("layout_bbox")),
        _bbox4(layer.get("balloon_bbox")),
        _bbox4(layer.get("bbox")),
    ]
    seen: set[tuple[int, int, int, int]] = set()
    targets: list[list[int]] = []
    for bbox in candidates:
        if bbox is None:
            continue
        key = tuple(bbox)
        if key in seen:
            continue
        seen.add(key)
        targets.append(bbox)
    return targets

def _bbox_intersection_area(a: list[int] | None, b: list[int] | None) -> int:
    if a is None or b is None:
        return 0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _bbox_center_distance(a: list[int] | None, b: list[int] | None) -> tuple[float, float]:
    if a is None or b is None:
        return (0.0, 0.0)
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return (abs(ax - bx), abs(ay - by))


def _render_fit_evidence_is_stale(layer: dict[str, Any]) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    if not render_fit:
        return False
    current_target = (
        _bbox4(layer.get("target_bbox"))
        or _bbox4(layer.get("balloon_bbox"))
        or _bbox4(layer.get("layout_bbox"))
        or _bbox4(layer.get("capacity_bbox"))
    )
    fit_target = _bbox4(render_fit.get("target_bbox")) or _bbox4(render_fit.get("balloon_bbox"))
    render_bbox = _bbox4(layer.get("render_bbox"))
    if current_target is None or fit_target is None or render_bbox is None:
        return False
    current_area = _bbox_area(current_target)
    fit_area = _bbox_area(fit_target)
    if current_area <= 0 or fit_area <= 0:
        return False
    return bool(
        _bbox_contains(current_target, render_bbox)
        and _bbox_contains(current_target, fit_target)
        and fit_area < int(current_area * 0.40)
    )


def _final_render_text_flags_are_review_only(layer: dict[str, Any]) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    target = (
        _bbox4(layer.get("balloon_bbox"))
        or _bbox4(render_fit.get("balloon_bbox"))
        or _bbox4(layer.get("target_bbox"))
        or _bbox4(render_fit.get("target_bbox"))
        or _bbox4(layer.get("capacity_bbox"))
        or _bbox4(layer.get("layout_bbox"))
        or _bbox4(layer.get("bbox"))
    )
    safe_bbox = _bbox4(layer.get("safe_text_box")) or _bbox4(render_fit.get("safe_text_box"))
    render_bbox = _bbox4(layer.get("render_bbox")) or _bbox4(render_fit.get("render_bbox"))
    if target is None or safe_bbox is None or render_bbox is None:
        return False
    attempts = [item for item in list(layer.get("fit_attempts") or []) if isinstance(item, dict)]
    fit_ok = str(layer.get("fit_status") or "").strip().lower() == "ok" or any(
        str(item.get("status") or "").strip().lower() == "ok" for item in attempts
    )
    if fit_ok and _bbox_contains(safe_bbox, render_bbox, margin=2):
        return True
    if _is_dark_bubble_or_panel_layer(layer) and _bbox_contains(safe_bbox, render_bbox, margin=2):
        return True
    if not _bbox_contains(target, render_bbox, margin=2):
        return False
    sx1, sy1, sx2, sy2 = safe_bbox
    rx1, ry1, rx2, ry2 = render_bbox
    safe_w = max(1, sx2 - sx1)
    safe_h = max(1, sy2 - sy1)
    overhang_px = max(0, sx1 - rx1, rx2 - sx2, sy1 - ry1, ry2 - sy2)
    return overhang_px <= max(8, int(round(min(safe_w, safe_h) * 0.04)))


def _is_dark_bubble_or_panel_layer(layer: dict[str, Any]) -> bool:
    profiles = {
        str(layer.get("layout_profile") or "").strip().lower(),
        str(layer.get("block_profile") or "").strip().lower(),
        str(layer.get("background_type") or "").strip().lower(),
    }
    source = str(layer.get("bubble_mask_source") or layer.get("balloon_mask_source") or "").strip().lower()
    if source in {"image_dark_bubble_mask", "image_dark_panel_mask", "derived_card_panel_mask"}:
        return True
    return bool(profiles & {"dark_bubble", "dark_panel", "colored_status_panel"})


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_real_lobe_assignment_evidence(layer: dict[str, Any]) -> bool:
    confidence = _float_or_none(layer.get("lobe_assignment_confidence"))
    if confidence is not None:
        return confidence < 0.6
    return bool(_clean_string(layer.get("connected_balloon_id")) or _clean_string(layer.get("lobe_id")))


def _warning_flags_blocking_export(flags: set[str], layer: dict[str, Any]) -> set[str]:
    if _scanlation_credit_flags_are_review_only(flags, layer):
        return set()
    blocking = set(flags & EXPORT_BLOCKING_REVIEW_FLAGS)
    if "mask_outside_balloon" in flags and _render_balloon_containment_is_low(layer):
        blocking.add("mask_outside_balloon")
    return blocking


def _layer_has_confirmed_visual_damage(flags: set[str]) -> bool:
    return bool(flags & CONFIRMED_VISUAL_DAMAGE_FLAGS)


def _scanlation_credit_flags_are_review_only(flags: set[str], layer: dict[str, Any]) -> bool:
    if not flags:
        return False
    if not _is_strong_scanlation_credit_layer(layer):
        return False
    decision_flags = flags - SCANLATION_HARMLESS_CONTEXT_FLAGS
    if not decision_flags:
        return True
    return decision_flags.issubset(SCANLATION_VISUAL_REVIEW_ONLY_FLAGS)


def _render_geometry_contained(layer: dict[str, Any]) -> bool:
    render_bbox = _bbox4(layer.get("render_bbox"))
    if render_bbox is None:
        return False
    safe_bbox = _bbox4(layer.get("safe_text_box"))
    for target in _containment_target_bboxes(layer):
        if not _bbox_contains(target, render_bbox, margin=4):
            continue
        if safe_bbox is not None and not _bbox_contains(target, safe_bbox, margin=6):
            continue
        return True
    return False


def _render_inside_safe_box(layer: dict[str, Any]) -> bool:
    render_bbox = _bbox4(layer.get("render_bbox"))
    safe_bbox = _bbox4(layer.get("safe_text_box"))
    return bool(render_bbox is not None and safe_bbox is not None and _bbox_contains(safe_bbox, render_bbox, margin=2))


def _rendered_background_is_white_balloon(layer: dict[str, Any]) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    if bool(qa_metrics.get("render_flat_balloon_background")):
        return True
    luma = _float_or_none(qa_metrics.get("render_balloon_background_luma"))
    std = _float_or_none(qa_metrics.get("render_balloon_background_luma_std"))
    return bool(luma is not None and luma >= 245.0 and (std is None or std <= 24.0))


def _traceability_missing_entry_is_review_only(missing: dict[str, Any]) -> bool:
    if bool(missing.get("is_review_only")):
        return True
    flag = str(missing.get("flag") or "").strip()
    if flag != "fast_fill_no_glyph_evidence":
        return False
    source = str(missing.get("source") or "").strip()
    return source in {"render_plan", "inpaint_decision", "mask_decision"}


def _render_balloon_containment_is_low(layer: dict[str, Any], *, threshold: float = 0.90) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    validated_containment = _float_or_none(qa_metrics.get("render_validated_containment"))
    if validated_containment is not None and validated_containment >= threshold and _render_geometry_contained(layer):
        return False
    if _render_geometry_contained(layer):
        return False
    containment = _float_or_none(qa_metrics.get("render_balloon_containment"))
    if containment is not None:
        return containment < threshold
    target = _bbox4(layer.get("balloon_bbox")) or _bbox4(layer.get("target_bbox"))
    render_bbox = _bbox4(layer.get("render_bbox"))
    if target is None or render_bbox is None:
        return False
    render_area = _bbox_area(render_bbox)
    if render_area <= 0:
        return False
    containment_ratio = _bbox_intersection_area(target, render_bbox) / float(render_area)
    return containment_ratio < threshold


def _render_validated_containment_is_good(layer: dict[str, Any], *, threshold: float = 0.92) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    validated_containment = _float_or_none(qa_metrics.get("render_validated_containment"))
    return bool(validated_containment is not None and validated_containment >= threshold)


def _source_and_render_are_displaced(layer: dict[str, Any]) -> bool:
    render_bbox = _bbox4(layer.get("render_bbox"))
    source_bbox = _best_source_alignment_bbox(layer, render_bbox)
    if source_bbox is None or render_bbox is None:
        return False
    for target in (
        _bbox4(layer.get("bubble_mask_bbox")),
        _bbox4(layer.get("bubble_inner_bbox")),
    ):
        if _bbox_contains(target, source_bbox, margin=8) and _bbox_contains(target, render_bbox, margin=8):
            return False
    source_area = _bbox_area(source_bbox)
    render_area = _bbox_area(render_bbox)
    if source_area <= 0 or render_area <= 0:
        return False
    overlap = _bbox_intersection_area(source_bbox, render_bbox)
    if overlap >= int(min(source_area, render_area) * 0.20):
        return False
    dx, dy = _bbox_center_distance(source_bbox, render_bbox)
    source_w = max(1, source_bbox[2] - source_bbox[0])
    source_h = max(1, source_bbox[3] - source_bbox[1])
    render_w = max(1, render_bbox[2] - render_bbox[0])
    render_h = max(1, render_bbox[3] - render_bbox[1])
    return bool(
        dx > max(24.0, min(source_w, render_w) * 0.45)
        or dy > max(18.0, min(source_h, render_h) * 0.75)
    )


def _best_source_alignment_bbox(layer: dict[str, Any], render_bbox: list[int] | None) -> list[int] | None:
    candidates = [
        _bbox4(layer.get("source_bbox")),
        _bbox4(layer.get("layout_bbox")),
        _bbox4(layer.get("bbox")),
        _bbox4(layer.get("text_pixel_bbox")),
    ]
    candidates = [bbox for bbox in candidates if bbox is not None]
    if not candidates:
        return None
    if render_bbox is None:
        return candidates[0]
    best = max(candidates, key=lambda bbox: _bbox_intersection_area(bbox, render_bbox))
    if _bbox_intersection_area(best, render_bbox) > 0:
        return best
    return candidates[0]


def _is_microtext_layer(layer: dict[str, Any]) -> bool:
    candidates = [
        _bbox4(layer.get("text_pixel_bbox")),
        _bbox4(layer.get("source_bbox")),
        _bbox4(layer.get("bbox")),
        _bbox4(layer.get("layout_bbox")),
    ]
    heights = [bbox[3] - bbox[1] for bbox in candidates if bbox is not None]
    if not heights or min(heights) > 34:
        return False
    target = _bbox4(layer.get("balloon_bbox")) or _bbox4(layer.get("target_bbox"))
    if target is None:
        return True
    target_h = target[3] - target[1]
    target_area = _bbox_area(target)
    return bool(target_h <= 80 or target_area <= 20000)


def _microtext_render_is_upscaled(layer: dict[str, Any]) -> bool:
    render_bbox = _bbox4(layer.get("render_bbox"))
    if render_bbox is None:
        return False
    candidates = [
        _bbox4(layer.get("text_pixel_bbox")),
        _bbox4(layer.get("bbox")),
        _bbox4(layer.get("layout_bbox")),
    ]
    small_heights = [bbox[3] - bbox[1] for bbox in candidates if bbox is not None]
    if not small_heights:
        return False
    source_h = min(small_heights)
    if source_h > 16:
        return False
    render_h = render_bbox[3] - render_bbox[1]
    return render_h > max(18, source_h * 3)


def _render_replaces_source_text_area(layer: dict[str, Any]) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    render_bbox = _bbox4(layer.get("render_bbox")) or _bbox4(render_fit.get("render_bbox"))
    if render_bbox is None:
        return False
    for source_bbox in (
        _bbox4(layer.get("text_pixel_bbox")),
        _bbox4(layer.get("source_bbox")),
        _bbox4(layer.get("layout_bbox")),
        _bbox4(layer.get("bbox")),
    ):
        if source_bbox is None:
            continue
        if _bbox_contains(source_bbox, render_bbox, margin=16):
            return True
        source_area = _bbox_area(source_bbox)
        render_area = _bbox_area(render_bbox)
        if source_area <= 0 or render_area <= 0:
            continue
        overlap = _bbox_intersection_area(source_bbox, render_bbox)
        if overlap >= int(min(source_area, render_area) * 0.55):
            return True
    return False


def _dark_layer_render_is_contained(layer: dict[str, Any]) -> bool:
    if not _is_dark_bubble_or_panel_layer(layer):
        return False
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    render_bbox = _bbox4(layer.get("render_bbox")) or _bbox4(render_fit.get("render_bbox"))
    safe_bbox = _bbox4(layer.get("safe_text_box")) or _bbox4(render_fit.get("safe_text_box"))
    if render_bbox is None:
        return False
    if safe_bbox is not None and not _bbox_contains(safe_bbox, render_bbox, margin=3):
        return False
    for target in _containment_target_bboxes(layer):
        if _bbox_contains(target, render_bbox, margin=8):
            return True
    containment = _float_or_none(qa_metrics.get("render_balloon_containment"))
    return bool(containment is not None and containment >= 0.90)


def _critical_flag_can_be_review_only(flag: str, flags: set[str], layer: dict[str, Any]) -> bool:
    if _scanlation_credit_flags_are_review_only(flags, layer):
        return True
    if flag == "text_residual_after_inpaint" and _contained_dark_residual_is_review_only(layer):
        return True
    if _layer_has_confirmed_visual_damage(flags):
        return False
    if flag == "mask_outside_balloon_critical":
        if _source_and_render_are_displaced(layer):
            return False
        return _render_geometry_contained(layer)
    if flag in {"bbox_overreach_critical", "fit_below_minimum_legible"}:
        if flag == "fit_below_minimum_legible" and _translator_note_fit_is_review_only(layer):
            return True
        if flag == "fit_below_minimum_legible" and "compact_small_text_capacity" in flags:
            return _render_inside_safe_box(layer)
        if flag == "fit_below_minimum_legible" and _rendered_background_is_white_balloon(layer):
            return _render_geometry_contained(layer) and _render_inside_safe_box(layer)
        if _microtext_render_is_upscaled(layer):
            return False
        return _is_microtext_layer(layer) and _render_geometry_contained(layer)
    if flag == "render_on_art_suspected":
        if _dark_layer_render_is_contained(layer):
            return True
        return _render_replaces_source_text_area(layer)
    if flag == "fast_fill_no_glyph_evidence":
        if _dark_layer_render_is_contained(layer):
            return True
        if layer.get("_render_metadata_group_sibling_geometry") and _render_inside_safe_box(layer):
            return True
        return _render_replaces_source_text_area(layer)
    return False


def _translator_note_fit_is_review_only(layer: dict[str, Any]) -> bool:
    text = str(layer.get("translated") or layer.get("text") or "").strip().lower()
    if not (text.startswith("t/n:") or text.startswith("tn:") or text.startswith("n/t:")):
        return False
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    containment = _float_or_none(qa_metrics.get("render_balloon_containment"))
    bg_luma = _float_or_none(qa_metrics.get("render_background_luma"))
    bg_std = _float_or_none(qa_metrics.get("render_background_luma_std"))
    flat_bg = bool(qa_metrics.get("render_flat_balloon_background"))
    if containment is not None and containment < 0.96:
        return False
    if bg_luma is not None and bg_luma < 235.0:
        return False
    if bg_std is not None and bg_std > 8.0:
        return False
    return flat_bg or bg_luma is not None


def _contained_dark_residual_is_review_only(layer: dict[str, Any]) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    luma = _float_or_none(qa_metrics.get("render_background_luma"))
    luma_std = _float_or_none(qa_metrics.get("render_background_luma_std"))
    containment = _float_or_none(qa_metrics.get("render_balloon_containment"))
    if luma is None or luma > 96.0:
        return False
    if luma_std is not None and luma_std > 24.0:
        return False
    if containment is not None and containment < 0.90:
        return False
    return bool(_render_inside_safe_box(layer) or _render_geometry_contained(layer))


def _artifact_links_for_issue(
    flags: set[str],
    *,
    page: dict[str, Any],
    page_id: str | None,
    band_id: str | None,
    trace_id: str | None,
) -> list[str]:
    links: list[str] = ["11_qa_export_gate/qa_issues.jsonl"]
    render_flags = {
        "TEXT_CLIPPED",
        "TEXT_OVERFLOW",
        "render_outside_balloon",
        "render_outside_bubble_mask",
        "render_bbox_far_from_target_bbox",
        "render_on_art_suspected",
        "page_space_rerender_mixed_coordinates",
    }
    geometry_flags = {
        "bbox_overreach_critical",
        "layout_bbox_coordinate_mismatch",
        "bubble_inner_bbox_coordinate_mismatch",
        "source_bbox_assigned_from_balloon",
        "safe_text_box_recomputed",
        "balloon_bbox_collapsed_to_text",
        "balloon_bbox_missing",
    }
    mask_flags = {
        "mask_outside_balloon_critical",
        "source_glyph_area_ratio_critical",
        "mask_outside_balloon",
        "bbox_fallback_bubble_mask",
        "glyph_mask_outside_bubble",
        "missing_real_bubble_mask",
        "fast_fill_insufficient_coverage",
        "fast_fill_unverified_residual",
        "low_inpaint_coverage",
    }
    residual_flags = {
        "weak_text_residual_after_inpaint",
        "text_residual_after_inpaint",
        "text_residual_after_inpaint_confirmed",
        "text_residual_after_inpaint_suspected",
        "fast_fill_unverified_residual",
        "fast_fill_insufficient_coverage",
    }
    translation_flags = {
        "vlm_failure_phrase",
        "translation_fallback_phrase",
        "glossary_violation",
        "forbidden_translation",
        "placeholder_lost",
        "unrestored_placeholder",
        "entity_mistranslated",
        "untranslated_english",
        "empty_translation",
        "mojibake_in_translation",
        "source_script_leak",
        "speech_cjk_preserved_inside_balloon",
    }

    if flags & render_flags:
        translated_ref = _translated_page_ref(page)
        if translated_ref:
            links.append(translated_ref)
        links.extend(
            [
                "09_typeset/render_plan_final.jsonl",
                "05_layout_geometry/layout_blocks.jsonl",
                f"12_contact_sheets/{band_id}.jpg",
            ]
        )

    if flags & geometry_flags:
        links.append("05_layout_geometry/layout_blocks.jsonl")
        if band_id:
            links.append(f"12_contact_sheets/{band_id}.jpg")

    if flags & mask_flags:
        links.append("06_mask_segmentation/mask_chain_summary.json")
        if band_id:
            links.append(f"06_mask_segmentation/{band_id}/mask_overlay.jpg")

    if flags & residual_flags:
        if band_id:
            links.extend(
                [
                    f"08_inpaint/{band_id}/03_inpaint_mask_overlay.jpg",
                    f"08_inpaint/{band_id}/inpaint_decision.json",
                    f"08_inpaint/{band_id}/06_band_after_inpaint.jpg",
                ]
            )

    if flags & translation_flags:
        links.extend(
            [
                "07_translation/translation_inputs.jsonl",
                "07_translation/translation_outputs.jsonl",
            ]
        )

    if trace_id:
        links.append("11_qa_export_gate/export_gate.json")

    existing = set()
    deduped: list[str] = []
    for rel_path in links:
        stable = _stable_rel_path(rel_path)
        if stable and stable not in existing:
            deduped.append(stable)
            existing.add(stable)
    return deduped


def evaluate_export_gate(project: dict[str, Any], *, override: bool = False) -> dict[str, Any]:
    issues = collect_export_blocking_issues(project)
    critical_issues = [issue for issue in issues if issue.get("severity") == "critical"]
    review_issues = [issue for issue in issues if issue.get("severity") == "warning"]
    blocking_issues = [
        issue
        for issue in issues
        if issue.get("severity") == "critical" or bool(issue.get("blocks_export"))
    ]
    critical_flag_count = sum(len(issue.get("flags") or []) for issue in critical_issues)
    review_flag_count = sum(len(issue.get("flags") or []) for issue in review_issues)
    blocking_flag_count = sum(len(issue.get("flags") or []) for issue in blocking_issues)
    status = "PASS"
    if blocking_issues:
        status = "OVERRIDDEN" if override else "BLOCK"
    elif any(issue.get("type") == "sfx_inpaint_review" for issue in review_issues):
        status = "REVIEW"
    return {
        "status": status,
        "allowed": status != "BLOCK",
        "override": bool(override),
        "issue_count": len(issues),
        "blocking_issue_count": len(blocking_issues),
        "blocking_flag_count": blocking_flag_count,
        "critical_issue_count": len(critical_issues),
        "critical_flag_count": critical_flag_count,
        "review_issue_count": len(review_issues),
        "review_flag_count": review_flag_count,
        "needs_review": bool(review_issues),
        "issues": issues,
    }


def collect_export_blocking_issues(project: dict[str, Any]) -> list[dict[str, Any]]:
    source_lang = str(project.get("idioma_origem") or "").lower()
    cjk_source = source_lang in {"ja", "jp", "ko", "kr", "zh", "zh-cn", "zh-tw"}
    issues: list[dict[str, Any]] = []
    for page_index, page in enumerate(project.get("paginas") or [], start=1):
        layers = page.get("text_layers") or page.get("textos") or []
        page_number = int(page.get("numero") or page_index)
        page_id = str(page.get("page_id") or f"page_{page_number:03d}")
        scanlation_credit_bands = {
            (
                _clean_string(candidate.get("band_id"))
                or _band_id_from_identity(_clean_string(candidate.get("trace_id")) or "")
                or ""
            )
            for candidate in layers
            if isinstance(candidate, dict) and _is_strong_scanlation_credit_layer(candidate)
        }
        scanlation_credit_bands.discard("")
        page_has_scanlation_credit = any(
            isinstance(candidate, dict) and _is_strong_scanlation_credit_layer(candidate)
            for candidate in layers
        )
        for layer_index, layer in enumerate(layers, start=1):
            if not isinstance(layer, dict):
                continue
            if not _layer_is_export_gate_candidate(layer):
                continue
            text_id = str(layer.get("text_id") or layer.get("id") or f"t{layer_index}")
            raw_trace_id = _clean_string(layer.get("trace_id") or layer.get("text_instance_id"))
            resolved_page_id = (
                layer.get("page_id")
                or _page_id_from_identity(raw_trace_id or "")
                or page_id
            )
            band_id = (
                _clean_string(layer.get("band_id"))
                or _band_id_from_identity(raw_trace_id or "")
                or _synthetic_band_id(str(resolved_page_id), layer_index)
            )
            policy_layer = layer
            if (
                band_id in scanlation_credit_bands
                and not _is_strong_scanlation_credit_layer(layer)
                and SCANLATION_CREDIT_RE.search(_layer_text_blob(layer))
            ):
                policy_layer = {**layer, "_scanlation_credit_context": True}
            elif (
                page_has_scanlation_credit
                and not _is_strong_scanlation_credit_layer(layer)
                and SCANLATION_CONTEXTUAL_RE.search(_layer_text_blob(layer))
            ):
                policy_layer = {**layer, "_scanlation_credit_context": True}
            trace_id = _resolve_trace_id(layer, text_id, band_id)
            translated = str(layer.get("translated") or layer.get("traduzido") or "")
            flags = _layer_qa_flags(layer)
            if cjk_source and translated and SOURCE_SCRIPT_RE.search(translated):
                flags.add("speech_cjk_preserved_inside_balloon")
            critical_flags = {flag for flag in flags if severity_for_flag(flag) == "critical"}
            demoted_critical_flags = {
                flag
                for flag in critical_flags
                if _critical_flag_can_be_review_only(flag, flags, policy_layer)
            }
            critical_flags -= demoted_critical_flags
            warning_flags = {flag for flag in flags if severity_for_flag(flag) == "high"}
            warning_flags.update(demoted_critical_flags)
            base_issue = {
                "page": page_number,
                "page_id": resolved_page_id,
                "band_id": band_id,
                "layer": layer.get("id") or text_id,
                "text_id": text_id,
                "text_instance_id": layer.get("text_instance_id")
                or (f"{band_id}_{text_id}" if band_id else None),
                "trace_id": trace_id,
                "coordinate_space": layer.get("coordinate_space") or "page",
                "text": translated[:160],
                "bbox": layer.get("bbox") or layer.get("layout_bbox") or layer.get("source_bbox"),
                "source_bbox": layer.get("source_bbox") or layer.get("bbox"),
                "balloon_bbox": layer.get("balloon_bbox"),
                "safe_text_box": layer.get("safe_text_box") or layer.get("_debug_safe_text_box"),
                "render_bbox": layer.get("render_bbox"),
                "qa_metrics": dict(layer.get("qa_metrics") or {}),
            }
            if _sfx_inpaint_requires_review(layer):
                issues.append(
                    {
                        **base_issue,
                        "type": "sfx_inpaint_review",
                        "severity": "warning",
                        "flags": _sfx_review_flags(layer),
                        "blocks_export": False,
                    }
                )
            if critical_flags:
                artifact_links = _artifact_links_for_issue(
                    critical_flags,
                    page=page,
                    page_id=str(base_issue.get("page_id") or ""),
                    band_id=str(band_id or ""),
                    trace_id=str(trace_id or ""),
                )
                issues.append(
                    {
                        **base_issue,
                        "type": "p0_render_blocker",
                        "severity": "critical",
                        "flags": sorted(critical_flags),
                        "blocks_export": True,
                        **({"artifact_links": artifact_links} if artifact_links else {}),
                    }
                )
            if warning_flags:
                blocks_export = bool(_warning_flags_blocking_export(warning_flags, policy_layer))
                artifact_links = _artifact_links_for_issue(
                    warning_flags,
                    page=page,
                    page_id=str(base_issue.get("page_id") or ""),
                    band_id=str(band_id or ""),
                    trace_id=str(trace_id or ""),
                )
                issues.append(
                    {
                        **base_issue,
                        "type": "needs_review",
                        "severity": "warning",
                        "flags": sorted(warning_flags),
                        "blocks_export": blocks_export,
                        **({"artifact_links": artifact_links} if artifact_links else {}),
                    }
                )
    qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
    propagation_audit = qa.get("flag_propagation_audit") if isinstance(qa, dict) else None
    if isinstance(propagation_audit, dict):
        for missing in propagation_audit.get("missing_in_project") or []:
            if not isinstance(missing, dict):
                continue
            identity = str(missing.get("identity") or missing.get("text_id") or "").strip()
            trace_id = _trace_id_from_identity(identity)
            text_id = missing.get("text_id") or _text_id_from_identity(identity)
            page_id = _page_id_from_identity(identity) or "unresolved"
            band_id = _band_id_from_identity(identity) or "unresolved"
            artifact_links = [
                "11_qa_export_gate/qa_flag_propagation_audit.json",
                "11_qa_export_gate/qa_issues.jsonl",
            ]
            if _traceability_missing_entry_is_review_only(missing):
                issues.append(
                    {
                        "page": None,
                        "page_id": page_id,
                        "band_id": band_id,
                        "layer": identity or "unresolved",
                        "text_id": text_id or "unresolved",
                        "text_instance_id": identity or None,
                        "trace_id": trace_id,
                        "coordinate_space": "debug_identity",
                        "text": f"QA flag not propagated from {missing.get('source') or 'debug'}",
                        "type": "needs_review",
                        "issue_scope": "run",
                        "severity": "warning",
                        "flags": ["qa_flag_not_propagated"],
                        "missing_flag": missing.get("flag"),
                        "missing_identity": identity or None,
                        "source": missing.get("source"),
                        "artifact_links": artifact_links,
                        "linked_artifacts": artifact_links,
                        "blocks_export": False,
                    }
                )
                continue
            issues.append(
                {
                    "page": None,
                    "page_id": page_id,
                    "band_id": band_id,
                    "layer": identity or "unresolved",
                    "text_id": text_id or "unresolved",
                    "text_instance_id": identity or None,
                    "trace_id": trace_id,
                    "coordinate_space": "debug_identity",
                    "text": f"QA flag not propagated from {missing.get('source') or 'debug'}",
                    "type": "p0_traceability_blocker",
                    "issue_scope": "run",
                    "severity": "critical",
                    "flags": ["qa_flag_not_propagated"],
                    "missing_flag": missing.get("flag"),
                    "missing_identity": identity or None,
                    "source": missing.get("source"),
                    "artifact_links": artifact_links,
                    "linked_artifacts": artifact_links,
                }
            )
        for candidate in propagation_audit.get("unmatched_detect_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("candidate_id") or "").strip()
            band_id = str(candidate.get("band_id") or _band_id_from_identity(candidate_id) or "unresolved")
            page_id = str(candidate.get("page_id") or _page_id_from_identity(candidate_id) or _page_id_from_identity(band_id) or "unresolved")
            artifact_links = [
                "02_strip_detect/detect_candidates.jsonl",
                "02_strip_detect/candidate_text_matching.jsonl",
                "11_qa_export_gate/qa_flag_propagation_audit.json",
                "11_qa_export_gate/qa_issues.jsonl",
            ]
            issues.append(
                {
                    "page": None,
                    "page_id": page_id,
                    "band_id": band_id,
                    "layer": candidate_id or "unresolved",
                    "text_id": "unresolved",
                    "text_instance_id": candidate_id or None,
                    "trace_id": None,
                    "coordinate_space": "debug_candidate",
                    "text": "Accepted detect candidate has no OCR text layer",
                    "type": "p0_traceability_blocker",
                    "issue_scope": "run",
                    "severity": "critical",
                    "flags": ["detect_candidate_without_ocr_text"],
                    "candidate_id": candidate_id or None,
                    "bbox": candidate.get("bbox_page"),
                    "bbox_strip": candidate.get("bbox_strip"),
                    "match_reason": candidate.get("match_reason"),
                    "artifact_links": artifact_links,
                    "linked_artifacts": artifact_links,
                }
            )
    return issues
