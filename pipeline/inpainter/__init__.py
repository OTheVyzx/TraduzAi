"""Adapter em-memoria do inpainter para o pipeline strip-based."""

from __future__ import annotations

import json
import copy
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

try:
    from .mask_builder import (
        bbox_to_octagon_mask,
        build_inpaint_mask,
        build_raw_text_mask_from_image,
        consolidate_mask_evidence,
        expand_text_mask,
        _close_visual_text_source_mask_gaps,
        _filter_dark_bubble_connected_lobe_line_polygons,
        mask_from_text_geometry,
    )
except ImportError:  # pragma: no cover - supports direct pipeline path imports
    from inpainter.mask_builder import (
        bbox_to_octagon_mask,
        build_inpaint_mask,
        build_raw_text_mask_from_image,
        consolidate_mask_evidence,
        expand_text_mask,
        _close_visual_text_source_mask_gaps,
        _filter_dark_bubble_connected_lobe_line_polygons,
        mask_from_text_geometry,
    )

try:
    from ocr.text_router import ROUTE_ACTIONS, route_action_requires_inpaint
except ImportError:  # pragma: no cover - supports package imports
    from ..ocr.text_router import ROUTE_ACTIONS, route_action_requires_inpaint

FAST_FILL_BLOCKING_QA_FLAGS = {
    "bbox_overreach",
    "bbox_overreach_critical",
    "mask_outside_balloon_critical",
}
FAST_FILL_EVIDENCE_DERIVED_QA_FLAGS: set[str] = set()
OUTSIDE_BALLOON_CRITICAL_PIXELS = 50
OUTSIDE_BALLOON_CRITICAL_RATIO = 0.18
OUTSIDE_BALLOON_WARN_RATIO = 0.08
SUPPRESSED_OCR_ROUTE_REASONS = {
    "english_ocr_gibberish_suppressed",
    "scanlator_text_caption_suppressed",
    "source_language_cjk_text_suppressed",
    "suppressed_duplicate_phrase_fragment",
    "visual_cjk_suppressed",
    "visual_sfx_overlap_suppressed",
}
SUPPRESSED_OCR_QA_FLAGS = set(SUPPRESSED_OCR_ROUTE_REASONS)


def _item_has_current_inpaint_mask_evidence(item: dict | None) -> bool:
    if not isinstance(item, dict):
        return False
    error = str(item.get("bubble_mask_error") or item.get("bubbleMaskError") or "").strip().lower()
    flags = {str(flag).strip() for flag in item.get("qa_flags") or [] if str(flag).strip()}
    source = str(item.get("bubble_mask_source") or item.get("bubbleMaskSource") or "").strip().lower()
    evidence = item.get("mask_evidence") if isinstance(item.get("mask_evidence"), dict) else {}
    kind = str(evidence.get("kind") or "").strip().lower()
    try:
        score = float(evidence.get("evidence_score") or 0.0)
    except Exception:
        score = 0.0
    raw_pixels = int(evidence.get("raw_mask_pixels") or 0)
    expanded_pixels = int(evidence.get("expanded_mask_pixels") or 0)
    if (
        source in _REJECTED_BUBBLE_MASK_SOURCES
        and error == "derived_mask_not_anchored_to_text"
        and kind in {"component_bubble_cleaner", "glyph_segmentation", "ocr_pixels"}
        and score >= 0.75
        and raw_pixels > 0
        and expanded_pixels >= raw_pixels
    ):
        return True
    if error or flags & {
        "debug_derived_bubble_mask_rejected",
        "derived_bubble_mask_rejected",
        "rejected_derived_bubble_mask",
    }:
        return False
    if source not in {
        "image_contour_bubble_mask",
        "image_rect_bubble_mask",
        "image_white_bubble_mask",
        "image_dark_panel_mask",
        "image_dark_bubble_mask",
        "derived_card_panel_mask",
    }:
        return False
    if kind not in {"component_bubble_cleaner", "glyph_segmentation", "ocr_pixels"}:
        return False
    return bool(score >= 0.75 and raw_pixels > 0 and expanded_pixels >= raw_pixels)


def _current_contour_mask_evidence_clears_outside_critical(item: dict | None) -> bool:
    if not _item_has_current_inpaint_mask_evidence(item):
        return False
    if not isinstance(item, dict):
        return False
    source = str(item.get("bubble_mask_source") or item.get("bubbleMaskSource") or "").strip().lower()
    if source != "image_contour_bubble_mask":
        return False
    evidence = item.get("mask_evidence") if isinstance(item.get("mask_evidence"), dict) else {}
    kind = str(evidence.get("kind") or "").strip().lower()
    if kind != "component_bubble_cleaner":
        return False
    reject_reasons = evidence.get("fast_fill_reject_reasons")
    if isinstance(reject_reasons, list) and any(str(reason).strip() for reason in reject_reasons):
        return False
    return True


def _current_dark_panel_mask_evidence_clears_outside_critical(item: dict | None) -> bool:
    if not _item_has_current_inpaint_mask_evidence(item):
        return False
    if not isinstance(item, dict):
        return False
    source = str(item.get("bubble_mask_source") or item.get("bubbleMaskSource") or "").strip().lower()
    if source not in {"image_dark_panel_mask", "image_dark_bubble_mask", "derived_card_panel_mask"}:
        return False
    if not (bool(item.get("card_panel_text_context")) or _is_dark_panel_text_candidate(item)):
        return False
    mask_bbox = _normalize_bbox(item.get("bubble_mask_bbox") or item.get("bubbleMaskBbox"), 10**9, 10**9)
    text_bbox = _normalize_bbox(item.get("text_pixel_bbox") or item.get("bbox"), 10**9, 10**9)
    if mask_bbox is None or text_bbox is None:
        return False
    return bool(_bbox_center_inside(text_bbox, mask_bbox) or _bbox_overlap_ratio(text_bbox, mask_bbox) >= 0.35)


def _is_translator_note_item(item: dict | None) -> bool:
    if not isinstance(item, dict):
        return False
    text = str(
        item.get("translated")
        or item.get("text")
        or item.get("original")
        or ""
    ).strip().lower()
    return text.startswith("t/n:") or text.startswith("tn:") or text.startswith("n/t:")


def _translator_note_has_current_text_mask_evidence(item: dict | None) -> bool:
    return bool(_is_translator_note_item(item) and _item_has_current_inpaint_mask_evidence(item))


def _translator_note_has_text_geometry(item: dict | None) -> bool:
    if not isinstance(item, dict):
        return False
    if not (_is_translator_note_item(item) or _translator_note_text_only_mask(item)):
        return False
    if _item_has_current_inpaint_mask_evidence(item):
        return True
    if _normalize_bbox(item.get("text_pixel_bbox") or item.get("bbox") or item.get("source_bbox"), 10**9, 10**9):
        return True
    polygons = item.get("line_polygons")
    return bool(isinstance(polygons, list) and polygons)


def _route_action_blocks_inpaint(text: dict) -> bool:
    if not isinstance(text, dict):
        return True
    if _text_suppressed_for_inpaint(text):
        return True
    route_action = str(text.get("route_action") or "").strip().lower()
    if route_action in ROUTE_ACTIONS:
        if route_action == "review_required" and _item_has_current_inpaint_mask_evidence(text):
            return False
        return not route_action_requires_inpaint(route_action)
    return False


def _text_suppressed_for_inpaint(text: dict) -> bool:
    if not isinstance(text, dict):
        return True
    if text.get("_fast_fill_inpaint_resolved"):
        return True
    route = str(text.get("route") or "").strip().lower()
    route_reason = str(text.get("route_reason") or "").strip().lower()
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if route == "suppress":
        return True
    if route_reason in SUPPRESSED_OCR_ROUTE_REASONS:
        return True
    if flags & SUPPRESSED_OCR_QA_FLAGS:
        return True
    content_class = str(text.get("content_class") or "").strip().lower()
    tipo = str(text.get("tipo") or "").strip().lower()
    route_action = str(text.get("route_action") or "").strip().lower()
    if content_class == "sfx" or tipo == "sfx":
        sfx = text.get("sfx") if isinstance(text.get("sfx"), dict) else {}
        if route_action != "translate_sfx_inpaint_render" or sfx.get("inpaint_allowed") is not True:
            return True
    return False


def _route_action_allows_local_dark_panel_fill(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    if _text_suppressed_for_inpaint(text):
        return False
    if not _route_action_blocks_inpaint(text):
        return True
    route_reason = str(text.get("route_reason") or "").strip().lower()
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    return bool(
        route_reason in {"mask_outside_balloon_critical", "missing_real_bubble_mask"}
        or flags & {"mask_outside_balloon_critical", "missing_real_bubble_mask"}
    )


def apply_koharu_bubble_fast_fill(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bubble_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not isinstance(bubble_mask, np.ndarray) or bubble_mask.shape[:2] != mask.shape[:2]:
        return image_rgb.copy(), mask.copy(), {"filled_pixels": 0, "reason": "missing_bubble_mask"}

    result = image_rgb.copy()
    remaining = np.where(mask > 0, 255, 0).astype(np.uint8)
    filled_pixels = 0
    reject_reason = ""
    overlap = (remaining > 0) & (bubble_mask > 0)
    bubble_ids = sorted(int(value) for value in np.unique(bubble_mask[overlap]))
    safe_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for bubble_id in bubble_ids:
        inside = bubble_mask == bubble_id
        safe_inside = cv2.erode(inside.astype(np.uint8), safe_kernel, iterations=1) > 0
        if not np.any(safe_inside):
            reject_reason = reject_reason or "bubble_mask_too_small"
            continue
        bubble_remaining = (remaining > 0) & inside
        outline_touching = bubble_remaining & ~safe_inside
        target = bubble_remaining & safe_inside
        if np.any(outline_touching):
            labels_count, labels = cv2.connectedComponents(bubble_remaining.astype(np.uint8), connectivity=8)
            for label in range(1, int(labels_count)):
                component = labels == label
                if np.any(component & outline_touching):
                    target[component] = False
        background = safe_inside & (remaining == 0)
        if np.any(outline_touching):
            reject_reason = reject_reason or "mask_touches_bubble_outline"
        if not np.any(target):
            continue
        if not np.any(background):
            reject_reason = reject_reason or "insufficient_background_sample"
            continue

        samples = image_rgb[background]
        median = np.median(samples, axis=0)
        std = np.std(samples, axis=0)
        if float(np.max(std)) >= 10.0:
            reject_reason = reject_reason or "background_variation_high"
            continue

        result[target] = np.asarray(median, dtype=np.uint8)
        remaining[target] = 0
        filled_pixels += int(np.count_nonzero(target))

    metadata = {"filled_pixels": filled_pixels, "bubble_ids": bubble_ids}
    if filled_pixels <= 0 and reject_reason:
        metadata["reason"] = reject_reason
    return result, remaining, metadata


def _original_text_scale_experiment_enabled() -> bool:
    return str(os.getenv("TRADUZAI_EXPERIMENT_ORIGINAL_TEXT_SCALE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _fallback_dark_bubble_mask_from_block_bbox(block: dict, width: int, height: int) -> np.ndarray | None:
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    flags = {str(flag).strip() for flag in block.get("qa_flags") or [] if str(flag).strip()}
    if source != "image_dark_bubble_mask" and "visual_text_only_inpaint_contract" not in flags:
        return None
    bbox = _normalize_bbox(block.get("balloon_bbox") or block.get("balloonBbox"), width, height)
    if bbox is None:
        bbox = _normalize_bbox(block.get("bubble_mask_bbox") or block.get("bubbleMaskBbox"), width, height)
    if bbox is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def _dark_text_contract_fill_rgb(block: dict) -> np.ndarray | None:
    source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
    flags = {str(flag).strip() for flag in block.get("qa_flags") or [] if str(flag).strip()}
    if source not in {"image_dark_bubble_mask", "image_dark_panel_mask", "derived_card_panel_mask"} and not (
        "visual_text_only_inpaint_contract" in flags
        or "dark_panel_style_grouped" in flags
        or "weak_text_residual_after_inpaint" in flags
    ):
        return None
    metrics = block.get("qa_metrics") if isinstance(block.get("qa_metrics"), dict) else {}
    for metric_key in ("image_dark_bubble_mask", "dark_bubble_visual_glyph_mask"):
        metric = metrics.get(metric_key) if isinstance(metrics, dict) else None
        if isinstance(metric, dict):
            raw = metric.get("panel_fill_rgb") or metric.get("fill_rgb")
            if isinstance(raw, (list, tuple)) and len(raw) >= 3:
                try:
                    return np.asarray([int(max(0, min(255, round(float(v))))) for v in raw[:3]], dtype=np.uint8)
                except Exception:
                    pass
    raw_background = block.get("background_rgb")
    if isinstance(raw_background, (list, tuple)) and len(raw_background) >= 3:
        try:
            rgb = np.asarray([int(max(0, min(255, round(float(v))))) for v in raw_background[:3]], dtype=np.uint8)
            if int(np.max(rgb)) <= 64:
                return rgb
        except Exception:
            pass
    return np.asarray([0, 0, 0], dtype=np.uint8)


def _text_is_preserved_or_sfx(text: dict) -> bool:
    if not isinstance(text, dict):
        return True
    content_class = str(text.get("content_class") or "").strip().lower()
    tipo = str(text.get("tipo") or "").strip().lower()
    route_action = str(text.get("route_action") or "").strip().lower()
    render_policy = str(text.get("render_policy") or "").strip().lower()
    if content_class == "sfx" or tipo == "sfx":
        sfx = text.get("sfx") if isinstance(text.get("sfx"), dict) else {}
        return route_action != "translate_sfx_inpaint_render" or sfx.get("inpaint_allowed") is not True
    return bool(
        text.get("skip_processing")
        or text.get("preserve_original")
        or route_action in {"skip", "preserve_original"}
        or render_policy in {"preserve_original", "merged_into_primary"}
    )


def _trusted_text_bbox_for_contract(text: dict, width: int, height: int) -> list[int] | None:
    candidates = []
    for key in (
        "source_text_anchor_bbox",
        "_source_text_anchor_bbox",
        "source_text_mask_bbox",
        "_source_text_mask_bbox",
        "text_pixel_bbox",
        "ocr_text_bbox",
        "bbox",
    ):
        bbox = _normalize_bbox(text.get(key), width, height)
        if bbox is not None:
            area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            candidates.append((area, key, bbox))
    if not candidates:
        return None
    _area, _key, bbox = min(candidates, key=lambda item: item[0])
    return bbox


def _text_mask_evidence_pixel_bounds(text: dict) -> tuple[int, int]:
    evidence = text.get("mask_evidence") if isinstance(text.get("mask_evidence"), dict) else {}
    raw = 0
    expanded = 0
    for key in ("raw_mask_pixels", "source_pixels"):
        try:
            raw = max(raw, int(evidence.get(key) or 0))
        except Exception:
            pass
    for key in ("expanded_mask_pixels", "expanded_pixels"):
        try:
            expanded = max(expanded, int(evidence.get(key) or 0))
        except Exception:
            pass
    metrics = text.get("qa_metrics") if isinstance(text.get("qa_metrics"), dict) else {}
    for metric_key in (
        "inpaint_mask_contract",
        "dark_bubble_visual_glyph_mask",
        "dark_bubble_visual_glyph_mask_replaced_geometry",
        "colored_card_visual_glyph_mask",
    ):
        metric = metrics.get(metric_key) if isinstance(metrics, dict) else None
        if not isinstance(metric, dict):
            continue
        for key in ("source_pixels", "raw_mask_pixels", "visual_pixels", "mask_pixels"):
            try:
                raw = max(raw, int(metric.get(key) or 0))
            except Exception:
                pass
        for key in ("expanded_pixels", "expanded_mask_pixels", "final_mask_pixels"):
            try:
                expanded = max(expanded, int(metric.get(key) or 0))
            except Exception:
                pass
    if expanded <= 0:
        expanded = raw
    return raw, expanded


def _text_contract_mask_is_plausible(text: dict, mask: np.ndarray, width: int, height: int) -> bool:
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return False
    pixels = int(np.count_nonzero(mask))
    if pixels < 24:
        return False
    bbox = _bbox_from_binary_mask(mask)
    trusted_bbox = _trusted_text_bbox_for_contract(text, width, height)
    if bbox is None or trusted_bbox is None:
        return False
    mask_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    trusted_area = max(1, (trusted_bbox[2] - trusted_bbox[0]) * (trusted_bbox[3] - trusted_bbox[1]))
    raw_evidence, expanded_evidence = _text_mask_evidence_pixel_bounds(text)
    if expanded_evidence > 0 and pixels > max(expanded_evidence + 2048, int(round(expanded_evidence * 2.20))):
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["dark_text_contract_fill_mask_rejected"] = {
                "reason": "mask_exceeds_text_evidence",
                "mask_pixels": int(pixels),
                "expanded_evidence_pixels": int(expanded_evidence),
            }
        _append_text_flag(text, "dark_text_contract_mask_rejected_overbroad")
        return False
    if mask_area > max(trusted_area + 12000, int(round(trusted_area * 2.60))):
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["dark_text_contract_fill_mask_rejected"] = {
                "reason": "mask_bbox_exceeds_text_bbox",
                "mask_bbox": list(bbox),
                "trusted_bbox": list(trusted_bbox),
                "mask_bbox_area": int(mask_area),
                "trusted_bbox_area": int(trusted_area),
                "raw_evidence_pixels": int(raw_evidence),
            }
        _append_text_flag(text, "dark_text_contract_mask_rejected_overbroad")
        return False
    return True


def _dark_fill_mask_is_overbroad_for_text(text: dict, mask: np.ndarray | None) -> bool:
    if not isinstance(text, dict) or not isinstance(mask, np.ndarray) or not np.any(mask):
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source != "image_dark_bubble_mask":
        return False
    pixels = int(np.count_nonzero(mask))
    _raw_evidence, expanded_evidence = _text_mask_evidence_pixel_bounds(text)
    if expanded_evidence > 0 and pixels > max(expanded_evidence + 4096, int(round(expanded_evidence * 2.50))):
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["dark_fill_mask_rejected_overbroad"] = {
                "mask_pixels": int(pixels),
                "expanded_evidence_pixels": int(expanded_evidence),
            }
        _append_text_flag(text, "dark_fill_mask_rejected_overbroad")
        return True
    bbox = _bbox_from_binary_mask(mask)
    if bbox is None:
        return False
    trusted = _trusted_text_bbox_for_contract(text, mask.shape[1], mask.shape[0])
    if trusted is None:
        return False
    mask_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    trusted_area = max(1, (trusted[2] - trusted[0]) * (trusted[3] - trusted[1]))
    if mask_area > max(trusted_area + 24000, int(round(trusted_area * 3.25))):
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["dark_fill_mask_rejected_overbroad"] = {
                "reason": "mask_bbox_exceeds_text_bbox",
                "mask_bbox": list(bbox),
                "trusted_bbox": list(trusted),
                "mask_bbox_area": int(mask_area),
                "trusted_bbox_area": int(trusted_area),
            }
        _append_text_flag(text, "dark_fill_mask_rejected_overbroad")
        return True
    return False


def _dark_text_contract_fill_mask(
    text: dict,
    width: int,
    height: int,
    image_rgb: np.ndarray | None = None,
) -> np.ndarray | None:
    if not isinstance(text, dict):
        return None
    if _text_is_preserved_or_sfx(text):
        return None
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    eligible = bool(
        source in {
            "image_dark_bubble_mask",
            "image_dark_panel_mask",
            "derived_card_panel_mask",
            "translator_note_text_mask",
            "text_rect_fallback",
        }
        or "visual_text_only_inpaint_contract" in flags
        or "weak_text_residual_after_inpaint" in flags
        or "dark_panel_style_grouped" in flags
    )
    if not eligible:
        return None
    mask = None
    used_raw_glyph_mask = False
    contract_source = ""
    if isinstance(image_rgb, np.ndarray) and image_rgb.shape[:2] == (height, width):
        try:
            raw_text_mask = build_raw_text_mask_from_image(text, image_rgb, image_rgb.shape)
        except Exception:
            raw_text_mask = None
        if isinstance(raw_text_mask, np.ndarray) and np.any(raw_text_mask):
            raw_pixels_before_close = int(np.count_nonzero(raw_text_mask))
            raw_text_mask = _close_visual_text_source_mask_gaps(raw_text_mask).astype(np.uint8)
            raw_pixels_after_close = int(np.count_nonzero(raw_text_mask))
            mask = expand_text_mask(raw_text_mask, expand_px=4)
            used_raw_glyph_mask = True
            contract_source = "raw_glyph_mask"
            metrics = text.setdefault("qa_metrics", {})
            if isinstance(metrics, dict):
                raw_bbox = _bbox_from_binary_mask(raw_text_mask)
                mask_bbox = _bbox_from_binary_mask(mask)
                metrics["dark_text_contract_raw_glyph_mask"] = {
                    "raw_bbox": [int(v) for v in raw_bbox] if raw_bbox is not None else None,
                    "mask_bbox": [int(v) for v in mask_bbox] if mask_bbox is not None else None,
                    "raw_mask_pixels": raw_pixels_after_close,
                    "expanded_mask_pixels": int(np.count_nonzero(mask)),
                    "source": "build_raw_text_mask_from_image",
                    "raw_mask_pixels_before_gap_close": raw_pixels_before_close,
                }
        try:
            contract_candidate = build_inpaint_mask(dict(text), image_rgb.shape, image_rgb=image_rgb)
        except Exception:
            contract_candidate = None
        if isinstance(contract_candidate, np.ndarray) and np.any(contract_candidate):
            contract_candidate = _coerce_mask_for_shape(contract_candidate, image_rgb.shape[:2])
            if isinstance(contract_candidate, np.ndarray) and np.any(contract_candidate):
                current_pixels = int(np.count_nonzero(mask)) if isinstance(mask, np.ndarray) else 0
                candidate_pixels = int(np.count_nonzero(contract_candidate))
                added_pixels = (
                    int(np.count_nonzero((contract_candidate > 0) & (mask == 0)))
                    if isinstance(mask, np.ndarray) and mask.shape[:2] == contract_candidate.shape[:2]
                    else candidate_pixels
                )
                if (
                    candidate_pixels > max(current_pixels + 512, int(round(current_pixels * 1.18)))
                    or added_pixels > max(256, int(round(max(1, current_pixels) * 0.05)))
                ):
                    mask = contract_candidate.astype(np.uint8)
                    used_raw_glyph_mask = True
                    contract_source = "build_inpaint_mask_contract"
                    metrics = text.setdefault("qa_metrics", {})
                    if isinstance(metrics, dict):
                        metrics["dark_text_contract_fill_uses_inpaint_contract_mask"] = {
                            "previous_pixels": int(current_pixels),
                            "contract_pixels": int(candidate_pixels),
                            "added_pixels": int(added_pixels),
                            "contract_bbox": [int(v) for v in (_bbox_from_binary_mask(contract_candidate) or [])],
                        }
    if mask is None or not np.any(mask):
        mask = _strict_text_geometry_mask(width, height, text)
    if mask is None or not np.any(mask):
        return None
    if not used_raw_glyph_mask:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    if not _text_contract_mask_is_plausible(text, mask, width, height):
        return None
    bbox = _bbox_from_binary_mask(mask)
    text["_force_solid_dark_text_fill"] = True
    metrics = text.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["dark_text_contract_fill_mask"] = {
            "bbox": [int(v) for v in bbox] if bbox is not None else None,
            "mask_pixels": int(np.count_nonzero(mask)),
            "source": contract_source or ("raw_glyph_mask" if used_raw_glyph_mask else "strict_text_geometry_mask"),
        }
    return mask.astype(np.uint8)


def _dark_panel_visual_contract_fill_mask(
    image_rgb: np.ndarray,
    text: dict,
    contract_mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or not isinstance(text, dict)
        or not isinstance(contract_mask, np.ndarray)
        or not np.any(contract_mask)
    ):
        return None
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source not in {"image_dark_panel_mask", "derived_card_panel_mask"}:
        return None
    if _text_is_preserved_or_sfx(text):
        return None
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    text_only_contract_flags = {
        "visual_text_only_inpaint_contract",
        "translator_note_text_only_mask",
        "text_contract_direct_fill",
    }
    if flags & text_only_contract_flags:
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["dark_panel_visual_contract_fill_mask_rejected"] = {
                "reason": "text_only_contract_requires_strict_text_mask",
                "contract_mask_pixels": int(np.count_nonzero(contract_mask)),
            }
        return None
    if not (
        source == "image_dark_panel_mask"
        or "dark_panel_full_bbox_selected" in flags
        or "dark_panel_rect_from_border_lines" in flags
        or str(text.get("layout_profile") or "").strip().lower() == "dark_panel"
    ):
        return None
    bbox = (
        _normalize_bbox(text.get("source_text_mask_bbox"), width, height)
        or _normalize_bbox(text.get("_source_text_mask_bbox"), width, height)
        or _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
    )
    if bbox is None:
        return None
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    pad = max(6, min(18, int(round(max(bbox_w, bbox_h) * 0.045))))
    limit_bbox = _expanded_bbox(width, height, bbox, padding=pad)
    if limit_bbox is None:
        return None
    limit_mask = _mask_from_bbox(width, height, limit_bbox, padding=0)
    region = limit_mask > 0
    if int(np.count_nonzero(region)) < 48:
        return None
    rgb_i = image_rgb.astype(np.int16)
    rgb_f = image_rgb.astype(np.float32)
    luma = (rgb_f[:, :, 0] * 0.299) + (rgb_f[:, :, 1] * 0.587) + (rgb_f[:, :, 2] * 0.114)
    chroma = np.max(rgb_i, axis=2) - np.min(rgb_i, axis=2)
    bg_region = region & (contract_mask == 0)
    bg_luma = float(np.percentile(luma[bg_region], 35)) if int(np.count_nonzero(bg_region)) >= 64 else 24.0
    threshold = max(72.0, min(190.0, bg_luma + 26.0))
    candidate = region & (
        (luma >= threshold)
        | ((luma >= max(58.0, bg_luma + 14.0)) & (chroma >= 28))
    )
    candidate_pixels = int(np.count_nonzero(candidate))
    if candidate_pixels < 24:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    visual_mask = cv2.dilate(candidate.astype(np.uint8) * 255, kernel, iterations=1)
    visual_mask = np.where(limit_mask > 0, visual_mask, 0).astype(np.uint8)
    combined = np.maximum(contract_mask.astype(np.uint8), visual_mask).astype(np.uint8)
    combined_pixels = int(np.count_nonzero(combined))
    contract_pixels = int(np.count_nonzero(contract_mask))
    if combined_pixels <= contract_pixels:
        return None
    limit_pixels = int(np.count_nonzero(limit_mask))
    max_pixels = max(contract_pixels + 4096, int(round(contract_pixels * 1.85)))
    max_pixels = min(max_pixels, limit_pixels)
    if combined_pixels > max_pixels:
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["dark_panel_visual_contract_fill_mask_rejected"] = {
                "reason": "visual_mask_too_large",
                "combined_pixels": int(combined_pixels),
                "contract_pixels": int(contract_pixels),
                "limit_pixels": int(limit_pixels),
                "max_pixels": int(max_pixels),
            }
        return None
    metrics = text.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["dark_panel_visual_contract_fill_mask"] = {
            "bbox": list(_bbox_from_binary_mask(combined) or limit_bbox),
            "mask_pixels": int(combined_pixels),
            "contract_mask_pixels": int(contract_pixels),
            "candidate_pixels": int(candidate_pixels),
            "source": "visual_glyph_union",
        }
    return combined


def _sample_dark_panel_contract_fill_rgb(
    image_rgb: np.ndarray,
    fill_mask: np.ndarray,
    text: dict,
    width: int,
    height: int,
) -> np.ndarray | None:
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or not isinstance(fill_mask, np.ndarray)
        or not np.any(fill_mask)
        or not isinstance(text, dict)
    ):
        return None
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source not in {"image_dark_panel_mask", "derived_card_panel_mask"}:
        return None
    bbox = (
        _normalize_bbox(text.get("source_text_mask_bbox"), width, height)
        or _normalize_bbox(text.get("_source_text_mask_bbox"), width, height)
        or _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
    )
    if bbox is None:
        return None
    search_bbox = _expanded_bbox(width, height, bbox, padding=24)
    if search_bbox is None:
        return None
    search_mask = _mask_from_bbox(width, height, search_bbox, padding=0) > 0
    mask = _coerce_mask_for_shape(fill_mask, (height, width)) > 0
    rgb = image_rgb.astype(np.float32)
    luma = (rgb[:, :, 0] * 0.299) + (rgb[:, :, 1] * 0.587) + (rgb[:, :, 2] * 0.114)
    candidates = search_mask & ~mask & (luma <= 80.0)
    if int(np.count_nonzero(candidates)) < 32:
        bubble_bbox = (
            _normalize_bbox(text.get("bubble_mask_bbox") or text.get("bubbleMaskBbox"), width, height)
            or _normalize_bbox(text.get("balloon_bbox") or text.get("balloonBbox"), width, height)
        )
        if bubble_bbox is not None:
            bubble_mask = _mask_from_bbox(width, height, bubble_bbox, padding=-6) > 0
            candidates = bubble_mask & ~mask & (luma <= 80.0)
    if int(np.count_nonzero(candidates)) < 32:
        return None
    sample = image_rgb[candidates]
    rgb_med = np.median(sample.astype(np.float32), axis=0)
    if float(np.max(rgb_med)) > 96.0:
        return None
    fill_rgb = np.asarray(
        [int(max(0, min(255, round(float(v))))) for v in rgb_med[:3]],
        dtype=np.uint8,
    )
    metrics = text.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["dark_panel_sampled_contract_fill_rgb"] = {
            "rgb": [int(v) for v in fill_rgb.tolist()],
            "sample_pixels": int(np.count_nonzero(candidates)),
            "source": "local_dark_panel_context",
        }
    return fill_rgb


def _normalize_bbox(raw_bbox, width: int, height: int) -> list[int] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox]
    except Exception:
        return None
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _append_text_flag(text: dict, flag: str) -> None:
    if not isinstance(text, dict) or not flag:
        return
    flags = [str(value) for value in text.get("qa_flags") or [] if str(value).strip()]
    if flag not in flags:
        flags.append(flag)
    text["qa_flags"] = flags


def _text_uses_direct_text_contract_fill(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if "text_contract_direct_fill" in flags:
        return True
    metrics = text.get("qa_metrics") if isinstance(text.get("qa_metrics"), dict) else {}
    return isinstance(metrics.get("text_contract_direct_fill"), dict)


def _image_dark_bubble_is_visually_light(image_rgb: np.ndarray | None, text: dict) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or not isinstance(text, dict):
        return False
    if str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower() != "image_dark_bubble_mask":
        return False
    height, width = image_rgb.shape[:2]
    metrics = text.get("qa_metrics") if isinstance(text.get("qa_metrics"), dict) else {}
    dark_metrics = metrics.get("image_dark_bubble_mask") if isinstance(metrics.get("image_dark_bubble_mask"), dict) else {}
    bbox = (
        _normalize_bbox(dark_metrics.get("mask_bbox"), width, height)
        or _normalize_bbox(text.get("bubble_mask_bbox"), width, height)
        or _normalize_bbox(text.get("balloon_bbox"), width, height)
        or _normalize_bbox(text.get("target_bbox"), width, height)
    )
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw < 32 or bh < 32:
        return False
    ix = max(2, int(round(bw * 0.12)))
    iy = max(2, int(round(bh * 0.12)))
    sx1, sy1, sx2, sy2 = x1 + ix, y1 + iy, x2 - ix, y2 - iy
    if sx2 <= sx1 or sy2 <= sy1:
        return False
    crop = image_rgb[sy1:sy2, sx1:sx2].astype(np.float32)
    if crop.size == 0:
        return False
    luma = (crop[:, :, 0] * 0.299) + (crop[:, :, 1] * 0.587) + (crop[:, :, 2] * 0.114)
    median_luma = float(np.median(luma))
    bright_ratio = float(np.count_nonzero(luma >= 210.0)) / float(max(1, luma.size))
    dark_ratio = float(np.count_nonzero(luma <= 80.0)) / float(max(1, luma.size))
    if median_luma >= 205.0 and bright_ratio >= 0.55 and dark_ratio <= 0.22:
        _append_text_flag(text, "false_light_bubble_dark_fill_blocked")
        return True
    return False


def _promote_visually_light_dark_bubbles_to_white(
    image_rgb: np.ndarray | None,
    ocr_page: dict | None,
    vision_blocks: list[dict] | None = None,
) -> int:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or not isinstance(ocr_page, dict):
        return 0
    height, width = image_rgb.shape[:2]

    def _promote_item(item: dict) -> bool:
        if not isinstance(item, dict):
            return False
        source = str(item.get("bubble_mask_source") or item.get("bubbleMaskSource") or "").strip().lower()
        if source != "image_dark_bubble_mask":
            return False
        if not _image_dark_bubble_is_visually_light(image_rgb, item):
            return False

        metrics = item.get("qa_metrics") if isinstance(item.get("qa_metrics"), dict) else {}
        dark_metrics = metrics.get("image_dark_bubble_mask") if isinstance(metrics.get("image_dark_bubble_mask"), dict) else {}
        metric_bbox = _normalize_bbox(dark_metrics.get("mask_bbox"), width, height)
        current_bbox = _normalize_bbox(item.get("bubble_mask_bbox") or item.get("bubbleMaskBbox"), width, height)
        should_replace_bbox = False
        if metric_bbox is not None:
            if current_bbox is None:
                should_replace_bbox = True
            else:
                current_area = max(1, (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1]))
                metric_area = max(1, (metric_bbox[2] - metric_bbox[0]) * (metric_bbox[3] - metric_bbox[1]))
                image_area = max(1, width * height)
                should_replace_bbox = bool(
                    current_area >= int(image_area * 0.48)
                    or (metric_area <= int(current_area * 0.62) and metric_area >= 64)
                )

        item["bubble_mask_source"] = "image_white_bubble_mask"
        item["bubbleMaskSource"] = "image_white_bubble_mask"
        if should_replace_bbox and metric_bbox is not None:
            item["bubble_mask_bbox"] = list(metric_bbox)
            item["bubbleMaskBbox"] = list(metric_bbox)
            item["balloon_bbox"] = list(metric_bbox)
        item["layout_profile"] = "white_balloon"
        item["block_profile"] = "white_balloon"
        item["background_rgb"] = [245, 245, 245]
        item.pop("_dark_bubble_sibling_texts", None)
        for key in (
            "card_panel_text_context",
            "dark_panel_text_context",
            "original_dark_panel_effect_colors",
        ):
            item.pop(key, None)
        style = item.get("style") if isinstance(item.get("style"), dict) else item.get("estilo")
        if isinstance(style, dict):
            style = dict(style)
            for key in ("glow", "glow_cor", "glow_px"):
                style.pop(key, None)
            style["cor"] = "#000000"
            style["contorno_px"] = 0
            style["style_origin"] = "false_light_dark_bubble_promoted_to_white"
            item["style"] = style
            item["estilo"] = style
        flags = [
            str(flag)
            for flag in item.get("qa_flags") or []
            if str(flag).strip()
            and str(flag).strip()
            not in {
                "auto_dark_panel_glow_fallback",
                "original_dark_panel_effect_colors",
                "dark_panel_style_grouped",
            }
        ]
        for flag in (
            "false_light_bubble_dark_fill_blocked",
            "false_light_dark_bubble_promoted_to_white",
            "false_dark_white_style_neutralized",
        ):
            if flag not in flags:
                flags.append(flag)
        item["qa_flags"] = flags
        return True

    promoted = 0
    seen: set[int] = set()
    for collection in (ocr_page.get("texts"), ocr_page.get("_vision_blocks"), vision_blocks):
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            obj_id = id(item)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            if _promote_item(item):
                promoted += 1
    if promoted:
        ocr_page["_strip_false_light_dark_bubble_promoted_count"] = int(
            ocr_page.get("_strip_false_light_dark_bubble_promoted_count") or 0
        ) + promoted
    return promoted


def _bbox_looks_page_relative(raw_bbox, *, width: int, height: int, band_y_top: int) -> bool:
    if band_y_top <= 0 or not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox]
    except Exception:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0 or x2 > width:
        return False
    fits_local = 0 <= y1 < y2 <= height
    shifted_fits = 0 <= (y1 - band_y_top) < (y2 - band_y_top) <= height
    ambiguous_lower_band = (
        fits_local
        and y1 >= band_y_top
        and band_y_top >= max(96, int(round(height * 0.45)))
    )
    return shifted_fits and (not fits_local or ambiguous_lower_band)


def _shift_bbox_to_band_local(raw_bbox, *, width: int, height: int, band_y_top: int) -> list[int] | None:
    if not _bbox_looks_page_relative(raw_bbox, width=width, height=height, band_y_top=band_y_top):
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox]
    except Exception:
        return None
    return _normalize_bbox([x1, y1 - band_y_top, x2, y2 - band_y_top], width, height)


def _line_polygons_look_page_relative(raw_polygons, *, width: int, height: int, band_y_top: int) -> bool:
    if band_y_top <= 0 or not isinstance(raw_polygons, list) or not raw_polygons:
        return False
    xs: list[int] = []
    ys: list[int] = []
    try:
        for polygon in raw_polygons:
            if not isinstance(polygon, (list, tuple)):
                continue
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                xs.append(int(round(float(point[0]))))
                ys.append(int(round(float(point[1]))))
    except Exception:
        return False
    if not xs or not ys:
        return False
    if min(xs) < 0 or max(xs) > width:
        return False
    fits_local = 0 <= min(ys) <= max(ys) <= height
    shifted_fits = 0 <= min(y - band_y_top for y in ys) <= max(y - band_y_top for y in ys) <= height
    ambiguous_lower_band = (
        fits_local
        and min(ys) >= band_y_top
        and band_y_top >= max(96, int(round(height * 0.45)))
    )
    return shifted_fits and (not fits_local or ambiguous_lower_band)


def _shift_line_polygons_to_band_local(raw_polygons, *, width: int, height: int, band_y_top: int) -> list | None:
    if not _line_polygons_look_page_relative(raw_polygons, width=width, height=height, band_y_top=band_y_top):
        return None
    shifted: list[list[list[int]]] = []
    for polygon in raw_polygons:
        if not isinstance(polygon, (list, tuple)):
            continue
        shifted_polygon: list[list[int]] = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                px = max(0, min(width - 1, int(round(float(point[0])))))
                py = max(0, min(height - 1, int(round(float(point[1]))) - band_y_top))
            except Exception:
                continue
            shifted_polygon.append([px, py])
        if len(shifted_polygon) >= 3:
            shifted.append(shifted_polygon)
    return shifted or None


def _polygon_bbox(polygon) -> list[int] | None:
    if not isinstance(polygon, (list, tuple)):
        return None
    xs: list[int] = []
    ys: list[int] = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            xs.append(int(round(float(point[0]))))
            ys.append(int(round(float(point[1]))))
        except Exception:
            continue
    if not xs or not ys:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _drop_isolated_side_note_line_polygons(text: dict) -> dict:
    """Remove OCR note/SFX polygons that were merged into a dialogue block."""

    if not isinstance(text, dict):
        return text
    polygons = text.get("line_polygons")
    if not isinstance(polygons, list) or len(polygons) < 3:
        return text
    boxes: list[tuple[int, list[int], float, int]] = []
    for index, polygon in enumerate(polygons):
        bbox = _polygon_bbox(polygon)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        boxes.append((index, bbox, (x1 + x2) / 2.0, x2 - x1))
    if len(boxes) < 3:
        return text
    centers = np.asarray([entry[2] for entry in boxes], dtype=np.float32)
    widths = np.asarray([max(1, entry[3]) for entry in boxes], dtype=np.float32)
    median_center = float(np.median(centers))
    main_width = float(np.median(widths))
    kept_indices: set[int] = set()
    removed = []
    for index, bbox, center, poly_width in boxes:
        center_gap = abs(center - median_center)
        short_line = poly_width <= max(92.0, main_width * 0.62)
        far_side = center_gap >= max(120.0, main_width * 0.82)
        if short_line and far_side:
            removed.append({"index": index, "bbox": bbox, "center_gap": round(center_gap, 3)})
            continue
        kept_indices.add(index)
    if not removed or len(kept_indices) < 2:
        return text
    item = dict(text)
    item["line_polygons"] = [polygon for idx, polygon in enumerate(polygons) if idx in kept_indices]
    metrics = item.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["isolated_side_note_line_polygons_removed"] = removed
    return item


def _shift_bbox_list_to_band_local(raw_bboxes, *, width: int, height: int, band_y_top: int) -> list[list[int]] | None:
    shifted: list[list[int]] = []
    for raw_bbox in raw_bboxes or []:
        bbox = _shift_bbox_to_band_local(raw_bbox, width=width, height=height, band_y_top=band_y_top)
        if bbox is not None and bbox not in shifted:
            shifted.append(bbox)
    return shifted or None


def _texts_with_band_local_bboxes(texts: list[dict], *, width: int, height: int, band_y_top: int) -> list[dict]:
    if band_y_top <= 0:
        return texts
    normalized: list[dict] = []
    bbox_fields = (
        "bbox",
        "source_bbox",
        "text_pixel_bbox",
        "balloon_bbox",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "layout_bbox",
        "target_bbox",
    )
    for text in texts:
        if not isinstance(text, dict):
            continue
        item = dict(text)
        shifted_any = False
        for field in bbox_fields:
            shifted = _shift_bbox_to_band_local(item.get(field), width=width, height=height, band_y_top=band_y_top)
            if shifted is not None:
                item[field] = shifted
                shifted_any = True
        shifted_polygons = _shift_line_polygons_to_band_local(
            item.get("line_polygons"),
            width=width,
            height=height,
            band_y_top=band_y_top,
        )
        if shifted_polygons is not None:
            item["line_polygons"] = shifted_polygons
            shifted_any = True
        shifted_source_bboxes = _shift_bbox_list_to_band_local(
            item.get("_merged_source_bboxes") or item.get("merged_source_bboxes"),
            width=width,
            height=height,
            band_y_top=band_y_top,
        )
        if shifted_source_bboxes is not None:
            item["_merged_source_bboxes"] = shifted_source_bboxes
            item["merged_source_bboxes"] = shifted_source_bboxes
            shifted_any = True
        if shifted_any:
            item["_band_local_bbox_normalized"] = True
        normalized.append(item)
    return normalized


def _build_fallback_vision_blocks(ocr_page: dict, width: int, height: int) -> list[dict]:
    blocks: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for txt in ocr_page.get("texts", []):
        if (
            not isinstance(txt, dict)
            or _text_suppressed_for_inpaint(txt)
            or _route_action_blocks_inpaint(txt)
            or _text_has_rejected_bubble_without_glyph_evidence(txt)
        ):
            continue
        bbox = (
            _normalize_bbox(txt.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(txt.get("bbox"), width, height)
        )
        if bbox is None:
            continue
        key = tuple(bbox)
        if key in seen:
            continue
        seen.add(key)
        blocks.append(
            {
                "bbox": bbox,
                "mask": None,
                "confidence": float(txt.get("confidence", txt.get("ocr_confidence", 0.0)) or 0.0),
                "id": txt.get("id"),
                "text_id": txt.get("text_id") or txt.get("id"),
                "page_id": txt.get("page_id"),
                "band_id": txt.get("band_id"),
                "trace_id": txt.get("trace_id"),
                "text_pixel_bbox": txt.get("text_pixel_bbox"),
                "source_bbox": txt.get("source_bbox"),
                "balloon_bbox": txt.get("balloon_bbox"),
                "_merged_source_bboxes": txt.get("_merged_source_bboxes") or txt.get("merged_source_bboxes"),
                "merged_source_bboxes": txt.get("_merged_source_bboxes") or txt.get("merged_source_bboxes"),
                "line_polygons": txt.get("line_polygons"),
                "bubble_id": txt.get("bubble_id") or txt.get("bubbleId"),
                "bubble_mask": txt.get("bubble_mask") if txt.get("bubble_mask") is not None else txt.get("bubbleMask"),
                "bubble_mask_source": txt.get("bubble_mask_source") or txt.get("bubbleMaskSource"),
                "bubbleMaskSource": txt.get("bubbleMaskSource") or txt.get("bubble_mask_source"),
                "bubble_mask_bbox": txt.get("bubble_mask_bbox") or txt.get("bubbleMaskBbox"),
                "bubbleMaskBbox": txt.get("bubbleMaskBbox") or txt.get("bubble_mask_bbox"),
                "bubble_inner_bbox": txt.get("bubble_inner_bbox") or txt.get("bubbleInnerBbox"),
                "bubbleInnerBbox": txt.get("bubbleInnerBbox") or txt.get("bubble_inner_bbox"),
                "bubble_mask_error": txt.get("bubble_mask_error") or txt.get("bubbleMaskError"),
                "bubbleMaskError": txt.get("bubbleMaskError") or txt.get("bubble_mask_error"),
                "balloon_type": txt.get("balloon_type"),
                "block_profile": txt.get("block_profile"),
                "background_rgb": txt.get("background_rgb"),
                "card_panel_text_context": txt.get("card_panel_text_context"),
                "route_action": txt.get("route_action"),
                "route_reason": txt.get("route_reason"),
                "qa_flags": list(txt.get("qa_flags") or []),
                "mask_evidence": copy.deepcopy(txt.get("mask_evidence")) if isinstance(txt.get("mask_evidence"), dict) else txt.get("mask_evidence"),
            }
        )
    return blocks


def _cjk_mask_kwargs_for_strip_page(ocr_page: dict) -> dict:
    engine_meta = ocr_page.get("_engine_preset")
    mask_strategy = ""
    if isinstance(engine_meta, dict):
        mask_strategy = str(engine_meta.get("mask_strategy") or "").strip().lower()
    try:
        from vision_stack.runtime import _get_bubble_segmenter_for_page, _get_text_segmenter_for_page

        text_segmenter = _get_text_segmenter_for_page(ocr_page)
        bubble_segmenter = _get_bubble_segmenter_for_page(ocr_page)
    except Exception:
        text_segmenter = None
        bubble_segmenter = None
    return {
        "mask_strategy": mask_strategy,
        "ocr_texts": [
            text
            for text in list(ocr_page.get("texts", [])) + list(ocr_page.get("_oar_ocr_regions", []))
            if isinstance(text, dict) and not _text_suppressed_for_inpaint(text)
        ],
        "text_segmenter": text_segmenter,
        "bubble_segmenter": bubble_segmenter,
    }


def _fast_white_balloon_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_WHITE_INPAINT", "0").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _fast_solid_balloon_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_SOLID_INPAINT", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _fast_white_post_cleanup_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_WHITE_POST_CLEANUP", "1").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _fast_white_narration_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_WHITE_NARRATION", "0").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _fast_local_balloon_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_LOCAL_INPAINT", "0").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _fast_metadata_background_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_METADATA_FILL", "1").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _fast_dark_panel_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_DARK_PANEL_FILL", "0").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    local_flag = os.getenv("TRADUZAI_STRIP_FAST_LOCAL_INPAINT")
    if local_flag is not None and local_flag.strip().lower() in {"0", "false", "no", "off"}:
        return False
    return True


def _auto_fast_dark_card_fill_allowed(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    if _text_suppressed_for_inpaint(text) or not _route_action_allows_local_dark_panel_fill(text):
        return False
    bubble_mask_source = str(text.get("bubble_mask_source") or "").strip().lower()
    qa_flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    evidence = text.get("mask_evidence")
    has_fast_ocr_evidence = bool(
        isinstance(evidence, dict)
        and evidence.get("fast_fill_allowed") is True
        and str(evidence.get("kind") or "").strip().lower() == "ocr_pixels"
        and int(evidence.get("raw_mask_pixels") or 0) > 0
    )
    false_white_dark_hint = bool(
        bubble_mask_source == "image_white_bubble_mask"
        and (text.get("balloon_bbox") or text.get("bubble_mask_bbox"))
        and (
            qa_flags
            & {
                "mask_outside_balloon",
                "mask_outside_balloon_critical",
                "bubble_clip_preserved_raw_text",
                "balloon_outline_components_removed",
            }
            or has_fast_ocr_evidence
        )
    )
    bbox_fallback_dark_hint = bool(
        bubble_mask_source == "bbox_fallback"
        and str(text.get("bubble_mask_error") or "").strip().lower() == "missing_real_bubble_mask"
        and has_fast_ocr_evidence
        and (text.get("balloon_bbox") or text.get("bubble_mask_bbox"))
        and (text.get("line_polygons") or text.get("text_pixel_bbox"))
    )
    dark_bubble_mask_hint = bool(
        bubble_mask_source == "image_dark_bubble_mask"
        and (text.get("balloon_bbox") or text.get("bubble_mask_bbox"))
        and (text.get("line_polygons") or text.get("text_pixel_bbox"))
    )
    if _text_is_white_balloon_context(text) and not false_white_dark_hint:
        return False
    if _text_has_rejected_bubble_without_glyph_evidence(text):
        return False
    if _text_has_no_glyph_evidence(text) and not _has_fast_fillable_text_mask_evidence(text):
        return False
    translator_note_text_mask = bubble_mask_source == "translator_note_text_mask"
    if translator_note_text_mask:
        return bool(text.get("line_polygons") or text.get("text_pixel_bbox"))
    mask_evidence_reason = _fast_fill_mask_evidence_rejection_reason(text)
    if mask_evidence_reason and not (false_white_dark_hint or dark_bubble_mask_hint):
        return False
    derived_card_panel = bubble_mask_source in {"derived_card_panel_mask", "image_dark_panel_mask", "image_dark_bubble_mask"}
    rejected_visual_card_panel = bubble_mask_source in _REJECTED_BUBBLE_MASK_SOURCES and bool(
        text.get("bubble_mask_bbox") or text.get("balloon_bbox")
    )
    if derived_card_panel and not rejected_visual_card_panel and not dark_bubble_mask_hint:
        return False
    if not (
        translator_note_text_mask
        or _is_dark_or_colored_card_text(text)
        or derived_card_panel
        or rejected_visual_card_panel
        or false_white_dark_hint
        or dark_bubble_mask_hint
        or bbox_fallback_dark_hint
    ):
        return False
    if not (text.get("line_polygons") or text.get("text_pixel_bbox")):
        return False
    background = text.get("background_rgb")
    if isinstance(background, (list, tuple)) and len(background) >= 3:
        return True
    if false_white_dark_hint or dark_bubble_mask_hint or bbox_fallback_dark_hint:
        return True
    if derived_card_panel or rejected_visual_card_panel:
        evidence = text.get("mask_evidence")
        has_mask_pixels = bool(
            isinstance(evidence, dict)
            and (
                int(evidence.get("raw_mask_pixels") or 0) > 0
                or int(evidence.get("expanded_mask_pixels") or 0) > 0
            )
        )
        return bool(has_mask_pixels and (text.get("bubble_mask_bbox") or text.get("balloon_bbox")))
    return False


def _rejected_visual_card_requires_real_inpaint(image_rgb: np.ndarray, text: dict) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or not isinstance(text, dict):
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source not in _REJECTED_BUBBLE_MASK_SOURCES:
        return False
    error = str(text.get("bubble_mask_error") or text.get("bubbleMaskError") or "").strip().lower()
    if error == "derived_mask_not_anchored_to_text" and _has_fast_fillable_text_mask_evidence(text):
        return True
    card_context = bool(text.get("card_panel_text_context")) or _is_dark_or_colored_card_text(text)
    if not card_context:
        return False
    height, width = image_rgb.shape[:2]
    bbox = (
        _normalize_bbox(text.get("balloon_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
        or _normalize_bbox(text.get("text_pixel_bbox"), width, height)
    )
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size < 64:
        return False
    pixels = crop.reshape(-1, crop.shape[-1]).astype(np.float32)
    channel_std = float(np.mean(np.std(pixels[:, :3], axis=0)))
    luma = pixels[:, 0] * 0.299 + pixels[:, 1] * 0.587 + pixels[:, 2] * 0.114
    chroma = np.max(pixels[:, :3], axis=1) - np.min(pixels[:, :3], axis=1)
    luma_spread = float(np.percentile(luma, 95) - np.percentile(luma, 5))
    chroma_spread = float(np.percentile(chroma, 95) - np.percentile(chroma, 5))
    colorful = float(np.percentile(chroma, 75)) >= 32.0
    textured = channel_std >= 24.0 or luma_spread >= 48.0 or chroma_spread >= 36.0
    return bool(colorful and textured)


def _rejected_visual_card_allows_local_text_fill(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source not in _REJECTED_BUBBLE_MASK_SOURCES:
        return False
    error = str(text.get("bubble_mask_error") or text.get("bubbleMaskError") or "").strip().lower()
    if error != "derived_mask_not_anchored_to_text":
        return False
    evidence = text.get("mask_evidence")
    if not isinstance(evidence, dict) or int(evidence.get("raw_mask_pixels") or 0) <= 0:
        return False
    return bool(text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("bbox"))


def _experimental_gpu_image_ops_enabled() -> bool:
    flag = os.getenv("TRADUZAI_EXPERIMENTAL_GPU_IMAGE_OPS", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _gpu_image_ops_backend() -> str:
    return os.getenv("TRADUZAI_GPU_IMAGE_OPS_BACKEND", "auto").strip() or "auto"


def _fill_mask_solid(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: int | tuple[int, int, int] = 255,
) -> np.ndarray:
    if _experimental_gpu_image_ops_enabled():
        try:
            from vision_stack.gpu_image_ops import apply_white_fill

            return apply_white_fill(image_rgb, mask, backend=_gpu_image_ops_backend(), color=color)
        except Exception:
            pass
    result = image_rgb.copy()
    result[mask > 0] = color
    return result


def _text_allows_fast_white_fill(text: dict) -> bool:
    return not _fast_white_rejection_reason(text)


def _fast_fill_blocking_qa_reason(text: dict, *, include_evidence_derived: bool = True) -> str:
    raw_flags = text.get("qa_flags") if isinstance(text, dict) else None
    if not isinstance(raw_flags, (list, tuple, set)):
        return ""
    flags = {str(flag).strip() for flag in raw_flags}
    blocked = sorted(flags & FAST_FILL_BLOCKING_QA_FLAGS)
    if not include_evidence_derived:
        blocked = [flag for flag in blocked if flag not in FAST_FILL_EVIDENCE_DERIVED_QA_FLAGS]
    return f"qa_flag:{blocked[0]}" if blocked else ""


def _can_use_local_card_fill_despite_blocking_qa(text: dict, qa_reason: str) -> bool:
    if not isinstance(text, dict) or not qa_reason:
        return False
    blocked_flag = qa_reason.removeprefix("qa_flag:").strip()
    if blocked_flag not in {"mask_outside_balloon_critical", "mask_outside_balloon"}:
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source not in {"derived_card_panel_mask", "image_dark_panel_mask", "image_dark_bubble_mask"} and source not in _REJECTED_BUBBLE_MASK_SOURCES:
        return False
    if not (text.get("text_pixel_bbox") or text.get("line_polygons")):
        return False
    evidence = text.get("mask_evidence")
    if not isinstance(evidence, dict) or evidence.get("fast_fill_allowed") is not True:
        return False
    return bool(int(evidence.get("raw_mask_pixels") or 0) > 0)


def _fast_fill_mask_evidence_rejection_reason(text: dict) -> str:
    if not isinstance(text, dict):
        return "invalid_text"
    mask_evidence = text.get("mask_evidence")
    if not isinstance(mask_evidence, dict):
        return "mask_evidence:missing"
    if mask_evidence.get("fast_fill_allowed") is not True:
        reasons = mask_evidence.get("fast_fill_reject_reasons")
        if isinstance(reasons, (list, tuple)) and reasons:
            reason = str(reasons[0] or "").strip()
            if reason:
                return f"mask_evidence:{reason}"
        return "mask_evidence:not_allowed"
    return ""


def _mask_to_canvas_for_bbox(mask_value, bbox: list[int] | None, width: int, height: int) -> np.ndarray | None:
    try:
        arr = np.asarray(mask_value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return None
    if arr.shape == (height, width):
        canvas = arr
    else:
        normalized = _normalize_bbox(bbox, width, height) if bbox is not None else None
        if normalized is None:
            return None
        x1, y1, x2, y2 = normalized
        target_h = max(1, y2 - y1)
        target_w = max(1, x2 - x1)
        patch = arr
        if patch.shape != (target_h, target_w):
            patch = cv2.resize(patch.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = patch[:target_h, :target_w]
    binary = np.where(canvas > 0, 255, 0).astype(np.uint8)
    return binary if np.any(binary) else None


def _weak_image_bubble_mask_error(
    item: dict,
    mask: np.ndarray,
    *,
    width: int,
    height: int,
) -> str:
    source = str(item.get("bubble_mask_source") or item.get("bubbleMaskSource") or "").strip().lower()
    if source not in {"image_white_bubble_mask", "image_rect_bubble_mask", "image_contour_bubble_mask"}:
        return ""
    if not isinstance(mask, np.ndarray) or mask.shape[:2] != (height, width) or not np.any(mask):
        return ""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return ""
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    mask_w = max(1, x2 - x1)
    mask_h = max(1, y2 - y1)
    mask_area = int(np.count_nonzero(mask > 0))
    bbox_area = max(1, mask_w * mask_h)
    bbox_fill = mask_area / float(bbox_area)
    edge_hits = 0
    roi = mask[y1:y2, x1:x2] > 0
    if np.count_nonzero(roi[0, :]) / float(max(1, mask_w)) >= 0.92:
        edge_hits += 1
    if np.count_nonzero(roi[-1, :]) / float(max(1, mask_w)) >= 0.92:
        edge_hits += 1
    if np.count_nonzero(roi[:, 0]) / float(max(1, mask_h)) >= 0.92:
        edge_hits += 1
    if np.count_nonzero(roi[:, -1]) / float(max(1, mask_h)) >= 0.92:
        edge_hits += 1
    if source in {"image_white_bubble_mask", "image_rect_bubble_mask"} and mask_area >= 12_000 and bbox_fill >= 0.94 and edge_hits >= 3:
        return "suspicious_rectangular_image_bubble_mask"

    text_bbox = _normalize_bbox(
        item.get("text_pixel_bbox") or item.get("layout_bbox") or item.get("bbox"),
        width,
        height,
    )
    if text_bbox is None:
        return ""
    text_area = max(1, (int(text_bbox[2]) - int(text_bbox[0])) * (int(text_bbox[3]) - int(text_bbox[1])))
    area_ratio = mask_area / float(text_area)
    text_cx = (text_bbox[0] + text_bbox[2]) / 2.0
    text_cy = (text_bbox[1] + text_bbox[3]) / 2.0
    mask_cx = (x1 + x2) / 2.0
    mask_cy = (y1 + y2) / 2.0
    offset_x = abs(text_cx - mask_cx) / float(mask_w)
    offset_y = abs(text_cy - mask_cy) / float(mask_h)
    min_margin_x = min(max(0.0, text_cx - x1), max(0.0, x2 - text_cx)) / float(mask_w)
    min_margin_y = min(max(0.0, text_cy - y1), max(0.0, y2 - text_cy)) / float(mask_h)
    if area_ratio >= 18.0 and (max(offset_x, offset_y) >= 0.28 or min(min_margin_x, min_margin_y) <= 0.08):
        return "derived_mask_not_anchored_to_text"
    return ""


def _weak_image_bubble_mask_error_for_item(item: dict | None) -> str:
    if not isinstance(item, dict):
        return ""
    for mask_key in ("bubble_mask", "bubbleMask", "balloon_mask", "balloonMask", "segmentation_mask", "mask"):
        if mask_key not in item:
            continue
        mask_value = item.get(mask_key)
        try:
            arr = np.asarray(mask_value)
        except Exception:
            continue
        if arr.size == 0:
            continue
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            continue
        bbox = (
            item.get("bubble_mask_bbox")
            or item.get("bubbleMaskBbox")
            or item.get("balloon_bbox")
            or item.get("bbox")
        )
        normalized = None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
                normalized = [max(0, x1), max(0, y1), max(1, x2), max(1, y2)]
            except Exception:
                normalized = None
        if normalized is not None:
            width = max(int(normalized[2]), 1)
            height = max(int(normalized[3]), 1)
            canvas = _mask_to_canvas_for_bbox(arr, normalized, width, height)
        else:
            height, width = arr.shape[:2]
            canvas = np.where(arr > 0, 255, 0).astype(np.uint8)
        if canvas is None or not np.any(canvas):
            continue
        reason = _weak_image_bubble_mask_error(item, canvas, width=width, height=height)
        if reason:
            return reason
    return ""


def _real_bubble_mask_for_text(ocr_page: dict, text: dict, width: int, height: int) -> tuple[np.ndarray | None, str]:
    if not isinstance(text, dict):
        return None, "invalid_text"
    bubble_id = str(text.get("bubble_id") or text.get("bubbleId") or "").strip()
    if not bubble_id:
        return None, "missing_bubble_id"

    candidates: list[dict] = []
    if isinstance(ocr_page, dict):
        for region in ocr_page.get("_bubble_regions") or ocr_page.get("bubble_regions") or []:
            if not isinstance(region, dict):
                continue
            region_id = str(region.get("bubble_id") or region.get("bubbleId") or region.get("id") or "").strip()
            if region_id == bubble_id:
                candidates.append(region)
    candidates.append(text)

    for candidate in candidates:
        for mask_key in ("bubble_mask", "bubbleMask", "balloon_mask", "balloonMask", "segmentation_mask", "mask"):
            if mask_key not in candidate:
                continue
            bbox = (
                candidate.get("bubble_mask_bbox")
                or candidate.get("bubbleMaskBbox")
                or candidate.get("balloon_bbox")
                or candidate.get("bbox")
            )
            mask = _mask_to_canvas_for_bbox(candidate.get(mask_key), bbox, width, height)
            if mask is not None and np.any(mask):
                reason = _weak_image_bubble_mask_error({**text, **candidate}, mask, width=width, height=height)
                if reason:
                    return None, reason
                return mask, ""
    return None, "missing_real_bubble_mask"


def _safe_real_bubble_interior_mask(bubble_mask: np.ndarray, width: int, height: int, erode_px: int = 2) -> np.ndarray:
    mask = _coerce_mask_for_shape(bubble_mask, (height, width))
    if not np.any(mask) or erode_px <= 0:
        return mask
    kernel_size = int(erode_px) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(np.uint8) if np.any(eroded) else mask


def _clip_fast_fill_text_mask_to_real_bubble(
    text_fill_mask: np.ndarray,
    bubble_mask: np.ndarray,
    width: int,
    height: int,
    *,
    min_inside_ratio: float = 0.90,
) -> tuple[np.ndarray | None, str]:
    if not isinstance(text_fill_mask, np.ndarray) or not np.any(text_fill_mask):
        return None, "missing_text_geometry_mask"
    bubble = _coerce_mask_for_shape(bubble_mask, (height, width))
    if not np.any(bubble):
        return None, "missing_real_bubble_mask"
    text_mask = _coerce_mask_for_shape(text_fill_mask, (height, width))
    text_pixels = int(np.count_nonzero(text_mask))
    if text_pixels <= 0:
        return None, "missing_text_geometry_mask"
    inside_pixels = int(np.count_nonzero((text_mask > 0) & (bubble > 0)))
    if inside_pixels <= 0:
        return None, "text_mask_outside_bubble"
    inside_ratio = inside_pixels / float(max(1, text_pixels))
    if inside_ratio < min_inside_ratio:
        return None, "text_mask_outside_bubble"
    safe_bubble = _safe_real_bubble_interior_mask(bubble, width, height)
    clipped = cv2.bitwise_and(text_mask.astype(np.uint8), safe_bubble.astype(np.uint8))
    if not np.any(clipped):
        clipped = cv2.bitwise_and(text_mask.astype(np.uint8), bubble.astype(np.uint8))
    return (clipped.astype(np.uint8), "") if np.any(clipped) else (None, "text_mask_outside_bubble")


def _has_fast_white_text_geometry(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    if text.get("line_polygons") or text.get("text_pixel_bbox"):
        return True
    return False


def _fast_white_can_use_geometry_despite_mask_evidence(text: dict, reason: str) -> bool:
    if not isinstance(text, dict):
        return False
    if reason not in {
        "mask_evidence:missing",
        "mask_evidence:mask_kind_not_fast_fill_allowed",
        "mask_evidence:coverage_too_low",
    }:
        return False
    return _has_fast_white_text_geometry(text)


def _propagate_existing_mask_evidence_decision_flags(ocr_page: dict, text: dict) -> None:
    if not isinstance(text, dict):
        return
    mask_evidence = text.get("mask_evidence")
    if isinstance(mask_evidence, dict):
        _propagate_mask_evidence_decision_flags(ocr_page, text, mask_evidence)


def _fast_white_rejection_reason(text: dict) -> str:
    if not isinstance(text, dict):
        return "invalid_text"
    if _route_action_blocks_inpaint(text):
        return "route_action_no_inpaint"
    qa_reason = _fast_fill_blocking_qa_reason(text, include_evidence_derived=False)
    if qa_reason:
        return qa_reason
    has_fast_white_geometry = _has_fast_white_text_geometry(text)
    mask_evidence_reason = _fast_fill_mask_evidence_rejection_reason(text)
    if (
        mask_evidence_reason
        and has_fast_white_geometry
        and not _fast_white_can_use_geometry_despite_mask_evidence(text, mask_evidence_reason)
    ):
        return mask_evidence_reason
    qa_reason = _fast_fill_blocking_qa_reason(text)
    if qa_reason:
        return qa_reason
    if not has_fast_white_geometry:
        return "missing_text_geometry"

    raw_confidence = text.get("ocr_confidence", text.get("confidence"))
    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
        except Exception:
            confidence = 1.0
        if confidence < 0.85:
            moderate_clean_white = (
                confidence >= 0.75
                and bool(text.get("text_pixel_bbox") or text.get("line_polygons"))
            )
            if not moderate_clean_white:
                return "low_confidence"

    return ""


def _bbox_center_inside(inner: list[int], outer: list[int]) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _bbox_overlap_ratio(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    b_area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(min(a_area, b_area))


def _bbox_union_values(a: list[int] | None, b: list[int] | None) -> list[int] | None:
    if a is None:
        return list(b) if b is not None else None
    if b is None:
        return list(a)
    return [
        min(int(a[0]), int(b[0])),
        min(int(a[1]), int(b[1])),
        max(int(a[2]), int(b[2])),
        max(int(a[3]), int(b[3])),
    ]


def _merge_unique_line_polygons(texts: list[dict]) -> list:
    merged: list = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for text in texts:
        raw_polygons = text.get("line_polygons") if isinstance(text, dict) else None
        if not isinstance(raw_polygons, list):
            continue
        for polygon in raw_polygons:
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                continue
            normalized: list[list[int]] = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    normalized.append([int(round(float(point[0]))), int(round(float(point[1])))])
                except Exception:
                    normalized = []
                    break
            if len(normalized) < 3:
                continue
            key = tuple((point[0], point[1]) for point in normalized)
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged


def _text_bbox_for_inpaint_geometry(item: dict, width: int, height: int) -> list[int] | None:
    text_bbox = _normalize_bbox(item.get("text_pixel_bbox"), width, height)
    layout_bbox = _normalize_bbox(item.get("layout_bbox"), width, height)
    bbox = _normalize_bbox(item.get("bbox"), width, height)
    peer = layout_bbox or bbox
    if text_bbox is not None and peer is not None:
        text_area = max(1, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
        peer_area = max(1, (peer[2] - peer[0]) * (peer[3] - peer[1]))
        inter_x1 = max(text_bbox[0], peer[0])
        inter_y1 = max(text_bbox[1], peer[1])
        inter_x2 = min(text_bbox[2], peer[2])
        inter_y2 = min(text_bbox[3], peer[3])
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter / float(min(text_area, peer_area)) >= 0.20 or _bbox_overlap_ratio(text_bbox, peer) >= 0.20:
            return text_bbox
        return peer
    return text_bbox or peer


def _enrich_vision_blocks_from_texts_for_inpaint(
    vision_blocks: list[dict],
    texts: list[dict],
    width: int,
    height: int,
) -> list[dict]:
    if not vision_blocks or not texts:
        return vision_blocks

    def _identity_values(item: dict) -> set[str]:
        values: set[str] = set()
        for key in ("trace_id", "text_instance_id", "id", "text_id"):
            value = str(item.get(key) or "").strip()
            if value:
                values.add(f"{key}:{value}")
        return values

    enriched_blocks: list[dict] = []
    for block in vision_blocks:
        current = dict(block)
        block_bbox = _normalize_bbox(current.get("bbox"), width, height)
        best_text = None
        best_score = 0.0
        matched_texts: list[dict] = []
        if block_bbox is not None:
            for text in texts:
                if not isinstance(text, dict):
                    continue
                text_bbox = _text_bbox_for_inpaint_geometry(text, width, height)
                if text_bbox is None:
                    continue
                score = _bbox_overlap_ratio(block_bbox, text_bbox)
                if score >= 0.35:
                    matched_texts.append(text)
                if score > best_score:
                    best_score = score
                    best_text = text
        if best_text is None:
            block_identities = _identity_values(current)
            if block_identities:
                for text in texts:
                    if not isinstance(text, dict):
                        continue
                    if block_identities & _identity_values(text):
                        best_text = text
                        best_score = 1.0
                        break
        if best_text is not None and best_score >= 0.35:
            preserve_current_mask_evidence = _item_has_current_inpaint_mask_evidence(current)
            for key in (
                "bbox",
                "rotation_deg",
                "rotation_source",
                "qa_flags",
                "allow_broad_bbox_text_search",
                "balloon_type",
                "block_profile",
                "line_polygons",
                "text_pixel_bbox",
                "source_bbox",
                "balloon_bbox",
                "background_rgb",
                "layout_bbox",
                "_merged_source_bboxes",
                "merged_source_bboxes",
                "content_class",
                "skip_processing",
                "preserve_original",
                "render_policy",
                "route_action",
                "route_reason",
                "bubble_mask_source",
                "bubbleMaskSource",
                "bubble_mask_bbox",
                "bubbleMaskBbox",
                "bubble_inner_bbox",
                "bubbleInnerBbox",
                "bubble_id",
                "bubbleId",
                "bubble_mask_error",
                "bubbleMaskError",
                "card_panel_text_context",
                "mask_evidence",
            ):
                if preserve_current_mask_evidence and key in {
                    "qa_flags",
                    "route_action",
                    "route_reason",
                    "bubble_mask_source",
                    "bubbleMaskSource",
                    "bubble_mask_error",
                    "bubbleMaskError",
                    "mask_evidence",
                }:
                    continue
                if key == "bbox" and block_bbox is not None:
                    continue
                value = best_text.get(key)
                if value not in (None, [], ""):
                    current[key] = copy.deepcopy(value)
            coherent_text_bbox = _text_bbox_for_inpaint_geometry(current, width, height)
            if coherent_text_bbox is not None:
                current["text_pixel_bbox"] = coherent_text_bbox
            if len(matched_texts) > 1:
                merged_polygons = _merge_unique_line_polygons(matched_texts)
                if merged_polygons:
                    current["line_polygons"] = merged_polygons
                merged_text_bbox: list[int] | None = None
                for text in matched_texts:
                    text_bbox = _text_bbox_for_inpaint_geometry(text, width, height)
                    merged_text_bbox = _bbox_union_values(merged_text_bbox, text_bbox)
                if merged_text_bbox is not None:
                    current["text_pixel_bbox"] = merged_text_bbox
        enriched_blocks.append(current)
    return enriched_blocks


def _append_missing_text_inpaint_blocks(
    vision_blocks: list[dict],
    texts: list[dict],
    width: int,
    height: int,
    image_rgb: np.ndarray,
) -> list[dict]:
    if not texts:
        return vision_blocks

    def _identity_values(item: dict) -> set[str]:
        values: set[str] = set()
        for key in ("trace_id", "text_instance_id", "id", "text_id"):
            value = str(item.get(key) or "").strip()
            if value:
                values.add(f"{key}:{value}")
        return values

    existing = [dict(block) for block in vision_blocks if isinstance(block, dict)]
    existing_ids: set[str] = set()
    existing_bboxes: list[list[int]] = []
    for block in existing:
        existing_ids.update(_identity_values(block))
        bbox = _text_bbox_for_inpaint_geometry(block, width, height)
        if bbox is not None:
            existing_bboxes.append(bbox)

    added = 0
    samples: list[dict] = []
    for text in texts:
        if not isinstance(text, dict):
            continue
        text_ids = _identity_values(text)
        if text_ids and text_ids & existing_ids:
            continue
        if (
            _text_suppressed_for_inpaint(text)
            or _route_action_blocks_inpaint(text)
            or _text_has_rejected_bubble_without_glyph_evidence(text)
        ):
            continue
        if _translator_note_text_only_mask(text) and not _translator_note_has_text_geometry(text):
            continue
        text_bbox = _text_bbox_for_inpaint_geometry(text, width, height)
        if text_bbox is None:
            continue
        if any(_bbox_overlap_ratio(text_bbox, bbox) >= 0.72 for bbox in existing_bboxes):
            continue
        candidate = dict(text)
        try:
            candidate_mask = build_inpaint_mask(candidate, image_rgb.shape, image_rgb)
        except Exception:
            candidate_mask = None
        if not isinstance(candidate_mask, np.ndarray) or not np.any(candidate_mask):
            continue
        fallback = _build_fallback_vision_blocks({"texts": [candidate]}, width, height)
        if not fallback:
            continue
        block = dict(fallback[0])
        block["_promoted_missing_text_inpaint_block"] = True
        existing.append(block)
        existing_ids.update(_identity_values(block))
        existing_bboxes.append(text_bbox)
        added += 1
        if len(samples) < 8:
            samples.append(
                {
                    "id": block.get("id") or block.get("text_id"),
                    "trace_id": block.get("trace_id"),
                    "bbox": block.get("bbox"),
                    "bubble_mask_source": block.get("bubble_mask_source"),
                    "mask_pixels": int(np.count_nonzero(candidate_mask)),
                }
            )
    if added:
        # The caller stores this on the page for debug metadata; keep the helper
        # pure with respect to ocr_page so unit tests can exercise it directly.
        for block in existing[-added:]:
            _append_qa_flag_to_item(block, "missing_text_promoted_to_inpaint_block")
    return existing


def _is_rotated_recovery_text(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    flags = {str(flag).strip().upper() for flag in text.get("qa_flags") or []}
    if "ROTATED_TEXT_RECOVERY" in flags:
        return True
    try:
        return abs(float(text.get("rotation_deg") or 0.0)) >= 35.0 and bool(text.get("allow_broad_bbox_text_search"))
    except Exception:
        return False


def _rotated_recovery_line_mask(text: dict, shape: tuple[int, ...]) -> np.ndarray | None:
    height = int(shape[0]) if shape else 0
    width = int(shape[1]) if len(shape) > 1 else 0
    if height <= 0 or width <= 0:
        return None
    polygons = text.get("line_polygons")
    if not isinstance(polygons, list) or not polygons:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    drew = False
    for polygon in polygons:
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
            continue
        points: list[list[int]] = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                points.append([
                    max(0, min(width - 1, int(round(float(point[0]))))),
                    max(0, min(height - 1, int(round(float(point[1]))))),
                ])
            except Exception:
                continue
        if len(points) >= 3:
            cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
            drew = True
    if not drew:
        return None
    return expand_text_mask(mask, expand_px=34)


def _apply_rotated_recovery_residual_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> tuple[np.ndarray, int]:
    if not isinstance(original_rgb, np.ndarray) or not isinstance(cleaned_rgb, np.ndarray):
        return cleaned_rgb, 0
    if original_rgb.shape != cleaned_rgb.shape:
        return cleaned_rgb, 0
    residual_mask = np.zeros(cleaned_rgb.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2HSV)
    bright = (gray >= 210) & (hsv[:, :, 1] <= 120)
    height, width = cleaned_rgb.shape[:2]
    for text in texts:
        if not _is_rotated_recovery_text(text):
            continue
        candidate = _rotated_recovery_line_mask(text, cleaned_rgb.shape)
        if candidate is None or not np.any(candidate):
            continue
        bbox = _normalize_bbox(text.get("text_pixel_bbox") or text.get("bbox"), width, height)
        if bbox is None:
            continue
        candidate_bool = (candidate > 0) & bright
        if not np.any(candidate_bool):
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_bool.astype(np.uint8), connectivity=8)
        box_w = max(1, bbox[2] - bbox[0])
        box_h = max(1, bbox[3] - bbox[1])
        for label in range(1, num_labels):
            x, y, comp_w, comp_h, area = stats[label].tolist()
            area = int(area)
            if area < 12 or area > 4200:
                continue
            if comp_w > max(90, int(box_w * 0.36)) or comp_h > max(140, int(box_h * 0.36)):
                continue
            # Frame/ornament strokes are usually long and thin; residual glyphs are compact.
            slender = max(comp_w, comp_h) / float(max(1, min(comp_w, comp_h)))
            if slender >= 9.0 and area >= 80:
                continue
            residual_mask[labels == label] = 255
    if not np.any(residual_mask):
        return cleaned_rgb, 0
    cleaned = cv2.inpaint(cleaned_rgb, residual_mask, 5, cv2.INPAINT_TELEA)
    return cleaned, int(np.count_nonzero(residual_mask))


def _fast_fill_union_mask_from_bboxes(filled_bboxes: list[list[int]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for filled in filled_bboxes:
        bbox = _normalize_bbox(filled, width, height)
        if bbox is None:
            continue
        mask = np.maximum(mask, _mask_from_bbox(width, height, bbox, padding=0))
    return mask


def _block_fast_fill_geometry_mask(block: dict, width: int, height: int) -> np.ndarray | None:
    geometry = _text_geometry_mask(width, height, block)
    if geometry is not None and np.any(geometry):
        return geometry
    bbox = _normalize_bbox(block.get("bbox"), width, height)
    if bbox is None:
        return None
    return _mask_from_bbox(width, height, bbox, padding=0)


def _block_is_covered_by_fast_fill(
    block: dict,
    filled_bboxes: list[list[int]],
    width: int,
    height: int,
    fast_fill_mask: np.ndarray | None = None,
) -> bool:
    bbox = _normalize_bbox(block.get("bbox"), width, height)
    if bbox is None:
        return False
    has_bbox_candidate = False
    for filled in filled_bboxes:
        if _bbox_center_inside(bbox, filled) or _bbox_overlap_ratio(bbox, filled) >= 0.25:
            has_bbox_candidate = True
            break
    if not has_bbox_candidate:
        return False

    block_mask = _block_fast_fill_geometry_mask(block, width, height)
    if block_mask is None or not np.any(block_mask):
        return False
    if isinstance(fast_fill_mask, np.ndarray) and fast_fill_mask.shape[:2] == (height, width):
        effective_mask = np.where(fast_fill_mask > 0, 255, 0).astype(np.uint8)
    else:
        effective_mask = _fast_fill_union_mask_from_bboxes(filled_bboxes, width, height)
    if not np.any(effective_mask):
        return False

    intersection = int(np.count_nonzero((block_mask > 0) & (effective_mask > 0)))
    if intersection <= 0:
        return False
    block_pixels = int(np.count_nonzero(block_mask))
    required = max(12, min(64, int(round(block_pixels * 0.18))))
    return intersection >= required or (intersection / float(max(1, block_pixels))) >= 0.22


def _is_connected_balloon_text(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    layout_profile = str(text.get("layout_profile") or "").strip().lower()
    if layout_profile == "connected_balloon":
        return True
    for key in ("balloon_subregions", "connected_lobe_bboxes"):
        values = text.get(key)
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            return True
    return False


def _connected_white_geometry_fill_candidate(text: dict, image_rgb: np.ndarray) -> bool:
    if not _is_connected_balloon_text(text):
        return False
    if _route_action_blocks_inpaint(text):
        return False
    if _fast_fill_blocking_qa_reason(text):
        return False
    if _fast_fill_mask_evidence_rejection_reason(text):
        return False
    if not text.get("line_polygons"):
        return False

    try:
        from vision_stack.runtime import _is_white_balloon_region
    except Exception:
        return False

    height, width = image_rgb.shape[:2]
    text_bbox = (
        _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
    )
    if text_bbox is None:
        return False
    text_mask = _mask_from_bbox(width, height, text_bbox, padding=2)
    candidates = [text_bbox]
    for value in text.get("balloon_subregions") or text.get("connected_lobe_bboxes") or []:
        bbox = _normalize_bbox(value, width, height)
        if bbox and _bbox_overlap_ratio(text_bbox, bbox) >= 0.05:
            candidates.append(bbox)
    balloon_bbox = _normalize_bbox(text.get("balloon_bbox"), width, height)
    if balloon_bbox:
        candidates.append(balloon_bbox)

    for bbox in candidates:
        if _is_white_balloon_region(image_rgb, bbox) and not _looks_translucent_or_textured_background(
            image_rgb,
            bbox,
            text_mask,
        ):
            return True
    return False


def _block_matches_any_text(block: dict, text_keys: set[str]) -> bool:
    if not text_keys:
        return False
    for key in ("trace_id", "text_id", "id", "text_instance_id"):
        value = str(block.get(key) or "").strip()
        if value and value in text_keys:
            return True
    for key in ("trace_ids", "trace_ids_in_band", "matched_trace_ids", "text_ids", "matched_text_ids"):
        raw_values = block.get(key)
        if not isinstance(raw_values, (list, tuple, set)):
            continue
        for value in raw_values:
            normalized = str(value or "").strip()
            if normalized and normalized in text_keys:
                return True
    return False


def _fast_fill_id_aliases(raw_id: object) -> set[str]:
    value = str(raw_id or "").strip()
    if not value:
        return set()
    aliases = {value}
    short = value.split("@", 1)[0].strip()
    if short:
        aliases.add(short)
    return aliases


def _mask_evidence_text_aliases(text: dict) -> set[str]:
    aliases: set[str] = set()
    for key in ("trace_id", "text_id", "id", "text_instance_id"):
        aliases.update(_fast_fill_id_aliases(text.get(key)))
    for key in ("trace_ids", "trace_ids_in_band", "matched_trace_ids", "text_ids", "matched_text_ids"):
        values = text.get(key)
        if not isinstance(values, (list, tuple, set)):
            continue
        for value in values:
            aliases.update(_fast_fill_id_aliases(value))
    return aliases


def _find_mask_evidence_text_for_block(
    block: dict,
    texts: list[dict],
    width: int,
    height: int,
) -> dict | None:
    for text in texts:
        if _block_matches_any_text(block, _mask_evidence_text_aliases(text)):
            return text

    block_bbox = _normalize_bbox(
        block.get("text_pixel_bbox") or block.get("bbox"),
        width,
        height,
    )
    if block_bbox is None:
        return None
    best_text = None
    best_score = 0.0
    for text in texts:
        text_bbox = _normalize_bbox(
            text.get("text_pixel_bbox") or text.get("bbox"),
            width,
            height,
        )
        if text_bbox is None:
            continue
        score = _bbox_overlap_ratio(block_bbox, text_bbox)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text if best_score >= 0.35 else None


def _merge_qa_flags_from_mask_evidence_block(text: dict, block: dict) -> None:
    merged = list(text.get("qa_flags") or [])
    for flag in block.get("qa_flags") or []:
        if flag not in merged:
            merged.append(flag)
    if merged:
        text["qa_flags"] = merged


def _append_qa_flag_to_item(item: dict, flag: str) -> None:
    flags = item.setdefault("qa_flags", [])
    if not isinstance(flags, list):
        flags = [str(flags)]
        item["qa_flags"] = flags
    if flag not in flags:
        flags.append(flag)


def _remove_qa_flags_from_item(item: dict, flags_to_remove: set[str]) -> None:
    if not isinstance(item, dict):
        return
    flags = [str(flag) for flag in item.get("qa_flags") or [] if str(flag).strip()]
    kept = [flag for flag in flags if flag not in flags_to_remove]
    if kept:
        item["qa_flags"] = kept
    else:
        item.pop("qa_flags", None)
    route_reason = str(item.get("route_reason") or "").strip()
    if route_reason in flags_to_remove:
        item.pop("route_reason", None)


def _clear_resolved_current_inpaint_flags(
    ocr_page: dict,
    *,
    final_residual_check: dict | None,
    final_clamped_outside: int,
) -> None:
    if not isinstance(ocr_page, dict):
        return
    if int(final_clamped_outside or 0) > 0:
        return
    flags_to_remove = {"mask_outside_balloon", "mask_outside_balloon_critical"}
    if not (isinstance(final_residual_check, dict) and final_residual_check.get("has_residual")):
        flags_to_remove.add("weak_text_residual_after_inpaint")
    changed = False
    for collection_key in ("texts", "_vision_blocks"):
        for item in ocr_page.get(collection_key) or []:
            if not isinstance(item, dict) or not _item_has_current_inpaint_mask_evidence(item):
                continue
            before = list(item.get("qa_flags") or [])
            _remove_qa_flags_from_item(item, flags_to_remove)
            changed = changed or before != list(item.get("qa_flags") or [])
    if not changed:
        return
    decision_flags = [
        str(flag)
        for flag in ocr_page.get("_strip_inpaint_decision_flags") or []
        if str(flag).strip() and str(flag).strip() not in flags_to_remove
    ]
    if not _ocr_page_has_unsafe_auto_inpaint_evidence(ocr_page):
        decision_flags = [flag for flag in decision_flags if flag != "real_inpaint_skipped_unsafe_mask"]
    if decision_flags:
        ocr_page["_strip_inpaint_decision_flags"] = decision_flags
    else:
        ocr_page.pop("_strip_inpaint_decision_flags", None)


def _mark_mask_outside_balloon_before_inpaint(
    ocr_page: dict,
    text: dict,
    block: dict,
    band_rgb: np.ndarray,
) -> None:
    height, width = band_rgb.shape[:2]
    action_mask = _coerce_mask_for_shape(block.get("_precomputed_inpaint_mask"), (height, width))
    if not np.any(action_mask):
        action_mask = _coerce_mask_for_shape(text.get("_precomputed_inpaint_mask"), (height, width))
    if not np.any(action_mask):
        try:
            action_mask = build_inpaint_mask(block, band_rgb.shape, band_rgb)
        except Exception:
            action_mask = None
    if action_mask is None or not np.any(action_mask):
        return
    bubble_mask, _reason = _real_bubble_mask_for_text(ocr_page, block, width, height)
    if bubble_mask is None or not np.any(bubble_mask):
        bubble_mask, _reason = _real_bubble_mask_for_text(ocr_page, text, width, height)
    if bubble_mask is None or not np.any(bubble_mask):
        return
    action = _coerce_mask_for_shape(action_mask, (height, width))
    bubble = _coerce_mask_for_shape(bubble_mask, (height, width))
    action_pixels = int(np.count_nonzero(action))
    if action_pixels <= 0:
        return
    outside_pixels = int(np.count_nonzero((action > 0) & (bubble == 0)))
    outside_ratio = outside_pixels / float(max(1, action_pixels))
    if outside_pixels > 0 and outside_ratio >= OUTSIDE_BALLOON_WARN_RATIO:
        for item in (text, block):
            _append_qa_flag_to_item(item, "mask_outside_balloon")
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["pre_inpaint_outside_balloon_ratio"] = round(float(outside_ratio), 6)
            metrics["pre_inpaint_outside_balloon_pixels"] = int(outside_pixels)
    if outside_pixels > OUTSIDE_BALLOON_CRITICAL_PIXELS and outside_ratio >= OUTSIDE_BALLOON_CRITICAL_RATIO:
        for item in (text, block):
            _append_qa_flag_to_item(item, "mask_outside_balloon_critical")
        _append_inpaint_decision_flag(ocr_page, "mask_outside_balloon_critical")


def _prime_mask_evidence_for_fast_fill(
    ocr_page: dict,
    vision_blocks: list[dict],
    band_rgb: np.ndarray,
) -> None:
    if not isinstance(band_rgb, np.ndarray) or band_rgb.size == 0:
        return
    texts = [text for text in list(ocr_page.get("texts") or []) if isinstance(text, dict)]
    if not texts or not vision_blocks:
        return
    height, width = band_rgb.shape[:2]
    for block in vision_blocks:
        if not isinstance(block, dict):
            continue
        target_text = _find_mask_evidence_text_for_block(block, texts, width, height)
        if _item_has_current_inpaint_mask_evidence(block):
            if target_text is not None:
                for key in (
                    "mask_evidence",
                    "bubble_mask_source",
                    "bubbleMaskSource",
                    "bubble_mask_error",
                    "bubbleMaskError",
                    "bubble_mask_bbox",
                    "bubbleMaskBbox",
                    "bubble_inner_bbox",
                    "bubbleInnerBbox",
                    "card_panel_text_context",
                ):
                    value = block.get(key)
                    if value not in (None, [], ""):
                        target_text[key] = copy.deepcopy(value)
                _merge_qa_flags_from_mask_evidence_block(target_text, block)
                _propagate_mask_evidence_decision_flags(ocr_page, target_text, target_text.get("mask_evidence") or {})
            continue
        if target_text is None:
            continue
        if not isinstance(block.get("mask_evidence"), dict):
            try:
                build_inpaint_mask(block, band_rgb.shape, band_rgb)
            except Exception:
                continue
        evidence = block.get("mask_evidence")
        if not isinstance(evidence, dict):
            continue
        target_text["mask_evidence"] = dict(evidence)
        _merge_qa_flags_from_mask_evidence_block(target_text, block)
        _mark_mask_outside_balloon_before_inpaint(ocr_page, target_text, block, band_rgb)
        _merge_qa_flags_from_mask_evidence_block(target_text, block)
        _propagate_mask_evidence_decision_flags(ocr_page, target_text, evidence)


def _fast_solid_verified_text_ids(ocr_page: dict) -> set[str]:
    verified: set[str] = set()
    for sample in ocr_page.get("_strip_fast_solid_fill_samples") or []:
        if not isinstance(sample, dict) or sample.get("fast_fill_verified") is not True:
            continue
        for alias in _fast_fill_id_aliases(sample.get("text_id")):
            verified.add(alias)
    return verified


def _block_has_verified_fast_solid_fill(block: dict, verified_text_ids: set[str]) -> bool:
    if not verified_text_ids:
        return False
    for key in ("id", "text_id", "trace_id", "text_instance_id"):
        for alias in _fast_fill_id_aliases(block.get(key)):
            if alias in verified_text_ids:
                return True
    for key in ("trace_ids", "trace_ids_in_band", "matched_trace_ids", "text_ids", "matched_text_ids"):
        raw_values = block.get(key)
        if not isinstance(raw_values, (list, tuple, set)):
            continue
        for value in raw_values:
            for alias in _fast_fill_id_aliases(value):
                if alias in verified_text_ids:
                    return True
    return False


def _fast_fill_residual_edge_ratio(
    image_rgb: np.ndarray,
    text_bbox: list[int],
    text_fill_mask: np.ndarray,
) -> float:
    if not isinstance(image_rgb, np.ndarray) or not isinstance(text_fill_mask, np.ndarray):
        return 1.0
    height, width = image_rgb.shape[:2]
    bbox = _normalize_bbox(text_bbox, width, height)
    if bbox is None or text_fill_mask.shape[:2] != (height, width):
        return 1.0
    x1, y1, x2, y2 = bbox
    region = text_fill_mask[y1:y2, x1:x2] > 0
    region_pixels = int(np.count_nonzero(region))
    if region_pixels <= 0:
        return 1.0
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return 1.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    return float(np.count_nonzero((edges > 0) & region) / float(region_pixels))


def _connected_geometry_fill_mask(image_rgb: np.ndarray, blocks: list[dict]) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    geometry_mask = np.zeros((height, width), dtype=np.uint8)
    for block in blocks:
        block_mask = mask_from_text_geometry(block, image_rgb.shape)
        if block_mask is not None and np.any(block_mask):
            geometry_mask = np.maximum(geometry_mask, block_mask.astype(np.uint8))
    if not np.any(geometry_mask):
        return geometry_mask
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dark_text = ((gray <= 215) & (geometry_mask > 0)).astype(np.uint8) * 255
    if not np.any(dark_text):
        return geometry_mask
    repair_mask = cv2.dilate(
        dark_text,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return cv2.bitwise_and(repair_mask, geometry_mask)


def _apply_connected_white_geometry_fill(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    rejection_reasons: dict[str, int] = {}

    def _reject(reason: str) -> None:
        rejection_reasons[reason or "unknown"] = rejection_reasons.get(reason or "unknown", 0) + 1

    def _record(stats: dict) -> dict:
        ocr_page["_strip_connected_white_rejection_reasons"] = dict(rejection_reasons)
        return stats

    if not _fast_white_balloon_fill_enabled():
        return band_rgb, vision_blocks, _record({"connected_white_count": 0, "remaining_blocks": len(vision_blocks)})
    height, width = band_rgb.shape[:2]
    connected_texts = []
    connected_bubble_mask = np.zeros((height, width), dtype=np.uint8)
    for text in ocr_page.get("texts", []):
        if not isinstance(text, dict):
            continue
        mask_evidence_reason = _fast_fill_mask_evidence_rejection_reason(text)
        if mask_evidence_reason and _is_connected_balloon_text(text):
            _propagate_existing_mask_evidence_decision_flags(ocr_page, text)
            _reject(mask_evidence_reason)
            continue
        if _connected_white_geometry_fill_candidate(text, band_rgb):
            real_bubble_mask, bubble_rejection = _real_bubble_mask_for_text(ocr_page, text, width, height)
            if real_bubble_mask is None:
                _reject(bubble_rejection)
                continue
            connected_bubble_mask = np.maximum(
                connected_bubble_mask,
                _safe_real_bubble_interior_mask(real_bubble_mask, width, height),
            )
            connected_texts.append(text)
    if not connected_texts:
        return band_rgb, vision_blocks, _record({"connected_white_count": 0, "remaining_blocks": len(vision_blocks)})

    text_keys: set[str] = set()
    for text in connected_texts:
        for key in ("trace_id", "text_id", "id", "text_instance_id"):
            value = str(text.get(key) or "").strip()
            if value:
                text_keys.add(value)

    selected_blocks = [block for block in vision_blocks if _block_matches_any_text(block, text_keys)]
    if not selected_blocks and len(connected_texts) == 1 and len(vision_blocks) == 1:
        selected_blocks = [vision_blocks[0]]
    if not selected_blocks:
        selected_blocks = [dict(text) for text in connected_texts]

    fill_mask = _connected_geometry_fill_mask(band_rgb, [dict(block) for block in selected_blocks])
    if np.any(fill_mask):
        fill_mask = cv2.bitwise_and(fill_mask.astype(np.uint8), connected_bubble_mask.astype(np.uint8))
    if not np.any(fill_mask):
        return band_rgb, vision_blocks, _record({"connected_white_count": 0, "remaining_blocks": len(vision_blocks)})

    result = band_rgb.copy()
    result[fill_mask > 0] = 255
    remaining_blocks = [
        block
        for block in vision_blocks
        if not _block_matches_any_text(block, text_keys)
    ]
    if len(vision_blocks) == 1 and len(selected_blocks) == 1 and len(remaining_blocks) == 1:
        remaining_blocks = []
    ocr_page["_strip_connected_white_geometry_fill_count"] = len(connected_texts)
    ocr_page["_strip_connected_white_geometry_fill_mask_pixels"] = int(np.count_nonzero(fill_mask))
    ocr_page["_strip_remaining_inpaint_blocks"] = len(remaining_blocks)
    return result, remaining_blocks, _record(
        {
            "connected_white_count": len(connected_texts),
            "remaining_blocks": len(remaining_blocks),
        }
    )


def _koharu_style_fast_white_evidence_rejection_reason(
    image_rgb: np.ndarray,
    text: dict,
    text_fill_mask: np.ndarray,
) -> str:
    """Require real glyph evidence before bbox-derived white fill.

    Koharu's Lama/AOT path expands the CTD segment mask and never treats OCR
    boxes alone as an erase mask. This guard keeps our strip fast path aligned
    with that rule for no-polygon text: a generated line/fill mask must overlap
    a raw text-pixel mask from the actual image, otherwise it may be face/art.
    """

    if not isinstance(text, dict) or not isinstance(text_fill_mask, np.ndarray):
        return "missing_koharu_text_evidence"
    if text.get("line_polygons"):
        return ""
    if image_rgb.size == 0 or text_fill_mask.size == 0 or not np.any(text_fill_mask):
        return "missing_koharu_text_evidence"
    try:
        evidence = build_raw_text_mask_from_image(dict(text), image_rgb, image_rgb.shape)
    except Exception:
        evidence = None
    if evidence is None or not isinstance(evidence, np.ndarray) or not np.any(evidence):
        return "missing_koharu_text_evidence"
    if evidence.shape[:2] != text_fill_mask.shape[:2]:
        return "missing_koharu_text_evidence"

    fill = text_fill_mask > 0
    glyph = evidence > 0
    overlap_pixels = int(np.count_nonzero(fill & glyph))
    glyph_pixels = int(np.count_nonzero(glyph))
    fill_pixels = int(np.count_nonzero(fill))
    min_overlap = max(16, int(round(min(glyph_pixels, fill_pixels) * 0.18)))
    if overlap_pixels < min_overlap:
        return "koharu_text_evidence_mismatch"
    return ""


def _sample_solid_fill_color_for_mask(
    image_rgb: np.ndarray,
    fill_mask: np.ndarray,
    sample_limit_mask: np.ndarray,
) -> tuple[tuple[int, int, int] | None, dict]:
    metadata = {
        "accepted": False,
        "reason": "",
        "color": None,
        "sample_bbox": None,
        "sample_pixels": 0,
        "max_std": None,
        "p95_abs_delta": None,
    }
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        metadata["reason"] = "invalid_image"
        return None, metadata
    if not isinstance(fill_mask, np.ndarray) or not np.any(fill_mask):
        metadata["reason"] = "missing_fill_mask"
        return None, metadata
    if not isinstance(sample_limit_mask, np.ndarray) or not np.any(sample_limit_mask):
        metadata["reason"] = "missing_sample_limit"
        return None, metadata

    shape = image_rgb.shape[:2]
    fill = (fill_mask > 0).astype(np.uint8) * 255
    limit = (sample_limit_mask > 0).astype(np.uint8) * 255
    sample_mask = cv2.dilate(
        fill,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=1,
    )
    sample_mask = cv2.bitwise_and(sample_mask, limit)
    sample_mask = cv2.bitwise_and(sample_mask, cv2.bitwise_not(fill))
    if int(np.count_nonzero(sample_mask)) < 64:
        sample_mask = cv2.bitwise_and(limit, cv2.bitwise_not(fill))
    if int(np.count_nonzero(sample_mask)) < 64:
        metadata["reason"] = "insufficient_sample_pixels"
        return None, metadata

    sample_bbox = _bbox_from_binary_mask(sample_mask)
    sample = image_rgb[sample_mask > 0].astype(np.float32)
    median = np.median(sample, axis=0)
    std = np.sqrt(np.mean(np.square(sample - median[None, :]), axis=0))
    abs_delta = np.max(np.abs(sample - median[None, :]), axis=1)
    max_std = float(np.max(std))
    p95_delta = float(np.percentile(abs_delta, 95))
    raw_max_std = max_std
    raw_p95_delta = p95_delta
    metadata.update(
        {
            "sample_bbox": sample_bbox,
            "sample_pixels": int(sample.shape[0]),
            "max_std": round(max_std, 4),
            "p95_abs_delta": round(p95_delta, 4),
            "raw_max_std": round(raw_max_std, 4),
            "raw_p95_abs_delta": round(raw_p95_delta, 4),
        }
    )
    if max_std > 9.0 or p95_delta > 24.0:
        robust_delta = np.max(np.abs(sample - median[None, :]), axis=1)
        robust_sample = sample[robust_delta <= 24.0]
        if robust_sample.shape[0] >= 64 and robust_sample.shape[0] / float(max(1, sample.shape[0])) >= 0.58:
            median = np.median(robust_sample, axis=0)
            std = np.sqrt(np.mean(np.square(robust_sample - median[None, :]), axis=0))
            abs_delta = np.max(np.abs(robust_sample - median[None, :]), axis=1)
            max_std = float(np.max(std))
            p95_delta = float(np.percentile(abs_delta, 95))
            metadata.update(
                {
                    "sample_pixels": int(robust_sample.shape[0]),
                    "max_std": round(max_std, 4),
                    "p95_abs_delta": round(p95_delta, 4),
                    "robust_dominant_sample": True,
                }
            )
        if max_std > 9.0 or p95_delta > 24.0:
            metadata["reason"] = "non_solid_background"
            return None, metadata

    median_luma = float(np.mean(median))
    median_chroma = float(np.max(median) - np.min(median))
    metadata["median_rgb"] = [int(max(0, min(255, round(float(v))))) for v in median]
    metadata["median_luma"] = round(median_luma, 4)
    metadata["median_chroma"] = round(median_chroma, 4)
    if median_luma >= 252.0 and median_chroma <= 4.0:
        color = (255, 255, 255)
    elif median_luma <= 8.0 and median_chroma <= 6.0:
        color = (0, 0, 0)
    else:
        color = tuple(int(max(0, min(255, round(float(v))))) for v in median)
    metadata["accepted"] = True
    metadata["reason"] = "solid_background_sample"
    metadata["color"] = list(color)
    return color, metadata


def _solid_fill_color_bucket(color: tuple[int, int, int] | list[int] | None) -> str:
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return "unknown"
    try:
        channels = [float(value) for value in color[:3]]
    except Exception:
        return "unknown"
    luma = sum(channels) / 3.0
    chroma = max(channels) - min(channels)
    if luma >= 245.0 and chroma <= 10.0:
        return "white"
    if luma <= 24.0 and chroma <= 16.0:
        return "black"
    return "colored"


def _fast_solid_rejection_reason(text: dict) -> str:
    if not isinstance(text, dict):
        return "invalid_text"
    if _route_action_blocks_inpaint(text):
        return "route_action_no_inpaint"
    qa_reason = _fast_fill_blocking_qa_reason(text)
    if qa_reason:
        return qa_reason
    mask_evidence_reason = _fast_fill_mask_evidence_rejection_reason(text)
    if mask_evidence_reason:
        return mask_evidence_reason
    if text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("bbox"):
        return ""
    return "missing_text_geometry"


def _fast_solid_pre_evidence_rejection_reason(text: dict) -> str:
    if not isinstance(text, dict):
        return "invalid_text"
    if _route_action_blocks_inpaint(text):
        return "route_action_no_inpaint"
    qa_reason = _fast_fill_blocking_qa_reason(text)
    if qa_reason:
        return qa_reason
    if text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("bbox"):
        return ""
    return "missing_text_geometry"


def _propagate_mask_evidence_decision_flags(ocr_page: dict, text: dict, evidence: dict) -> None:
    reasons = set(evidence.get("fast_fill_reject_reasons") or [])
    flags = set(text.get("qa_flags") or [])
    if "raw_mask_pixels_zero" in reasons and "fast_fill_no_glyph_evidence" in flags:
        _append_inpaint_decision_flag(ocr_page, "fast_fill_no_glyph_evidence")


def _solid_fill_limit_bbox(text: dict, width: int, height: int) -> list[int] | None:
    limit_bbox = None
    for key in (
        "balloon_inner_bbox",
        "layout_safe_bbox",
        "layout_bbox",
        "safe_text_box",
        "_visual_rect_inner_bbox",
        "_visual_rect_outer_bbox",
    ):
        bbox = _normalize_bbox(text.get(key), width, height)
        if bbox is not None:
            limit_bbox = bbox
            break
    if text.get("line_polygons"):
        limit_bbox = _expand_solid_fill_limit_for_text_geometry(text, width, height, limit_bbox)
    return limit_bbox


def _fast_solid_line_expand_px() -> int:
    raw_value = os.environ.get("TRADUZAI_FAST_SOLID_LINE_EXPAND_PX", "2")
    try:
        value = int(round(float(raw_value)))
    except Exception:
        value = 2
    return max(0, min(8, value))


def _bbox_area_value(bbox: list[int] | None) -> int:
    if bbox is None:
        return 0
    return max(1, (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1])))


def _line_polygons_bbox(text: dict, width: int, height: int, *, padding: int = 0) -> list[int] | None:
    raw_polygons = text.get("line_polygons") if isinstance(text, dict) else None
    if not isinstance(raw_polygons, list):
        return None
    xs: list[int] = []
    ys: list[int] = []
    for polygon in raw_polygons:
        if not isinstance(polygon, (list, tuple)):
            continue
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                xs.append(max(0, min(width, int(round(float(point[0]))))))
                ys.append(max(0, min(height, int(round(float(point[1]))))))
            except Exception:
                continue
    if not xs or not ys:
        return None
    return _normalize_bbox(
        [min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding],
        width,
        height,
    )


def _text_source_limit_bbox(text: dict, width: int, height: int, *, include_line_bbox: bool = True) -> list[int] | None:
    limit_bbox: list[int] | None = None
    for key in ("source_bbox", "text_pixel_bbox", "bbox"):
        bbox = _normalize_bbox(text.get(key), width, height)
        if bbox is not None:
            limit_bbox = _bbox_union_values(limit_bbox, bbox)
    if include_line_bbox:
        line_bbox = _line_polygons_bbox(text, width, height, padding=0)
        if line_bbox is not None:
            limit_bbox = _bbox_union_values(limit_bbox, line_bbox)
    return _normalize_bbox(limit_bbox, width, height) if limit_bbox is not None else None


def _text_cleanup_limit_bbox(text: dict, width: int, height: int) -> list[int] | None:
    """Limit strip-real inpaint to glyph evidence, with bounded OCR line recovery."""

    try:
        rotation_abs = abs(float(text.get("rotation_deg") or text.get("rotation") or 0.0))
    except Exception:
        rotation_abs = 0.0
    if rotation_abs >= 8.0:
        rotated_source = _text_source_limit_bbox(text, width, height)
        if rotated_source is not None:
            return rotated_source

    pixel_bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height)
    line_bbox = _line_polygons_bbox(text, width, height, padding=1)
    if pixel_bbox is None:
        return line_bbox or _text_source_limit_bbox(text, width, height)
    if line_bbox is None:
        return pixel_bbox

    px1, py1, px2, py2 = pixel_bbox
    lx1, ly1, lx2, ly2 = line_bbox
    pixel_w = max(1, px2 - px1)
    pixel_h = max(1, py2 - py1)
    extra_x = max(14, min(28, int(round(pixel_w * 0.14))))
    extra_y = max(8, min(24, int(round(pixel_h * 0.25))))
    expanded = _normalize_bbox(
        [
            max(lx1, px1 - extra_x),
            max(ly1, py1 - extra_y),
            min(lx2, px2 + extra_x),
            min(ly2, py2 + extra_y),
        ],
        width,
        height,
    )
    return expanded or pixel_bbox


def _text_source_limit_mask(width: int, height: int, text: dict, *, padding: int = 1) -> np.ndarray | None:
    bbox = _text_source_limit_bbox(text, width, height)
    if bbox is None:
        return None
    return _mask_from_bbox(width, height, bbox, padding=padding)


def _source_bbox_raw_text_mask(image_rgb: np.ndarray, text: dict, width: int, height: int) -> np.ndarray | None:
    source_bbox = (
        _normalize_bbox(text.get("source_bbox"), width, height)
        or _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
    )
    if source_bbox is None:
        return None
    candidate = dict(text)
    candidate.pop("line_polygons", None)
    candidate["bbox"] = list(source_bbox)
    candidate["text_pixel_bbox"] = list(source_bbox)
    candidate["source_bbox"] = list(source_bbox)
    try:
        raw = build_raw_text_mask_from_image(candidate, image_rgb, image_rgb.shape)
    except Exception:
        raw = None
    if not isinstance(raw, np.ndarray) or not np.any(raw):
        return None
    limit = _mask_from_bbox(width, height, source_bbox, padding=1)
    raw = cv2.bitwise_and(raw.astype(np.uint8), limit.astype(np.uint8))
    return raw if np.any(raw) else None


def _nearby_raw_text_mask_from_image(image_rgb: np.ndarray, text: dict, width: int, height: int) -> np.ndarray | None:
    base_bbox = _text_source_limit_bbox(text, width, height)
    if base_bbox is None or not isinstance(image_rgb, np.ndarray) or image_rgb.shape[:2] != (height, width):
        return None
    expanded_limit = _resolve_fast_solid_limit_bbox(image_rgb, text, width, height) or base_bbox
    search_bbox = _bbox_union_values(base_bbox, expanded_limit)
    if search_bbox is None:
        return None
    bx1, by1, bx2, by2 = base_bbox
    base_w = max(1, bx2 - bx1)
    base_h = max(1, by2 - by1)
    pad_x = max(12, min(48, int(round(base_w * 0.35))))
    pad_y = max(4, min(18, int(round(base_h * 0.25))))
    sx1 = max(0, int(search_bbox[0]) - pad_x)
    sy1 = max(0, int(search_bbox[1]) - pad_y)
    sx2 = min(width, int(search_bbox[2]) + pad_x)
    sy2 = min(height, int(search_bbox[3]) + pad_y)
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    roi = image_rgb[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    dark = (gray <= 112).astype(np.uint8)
    if not np.any(dark):
        return None

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    extra = np.zeros((height, width), dtype=np.uint8)

    def _overlap(a: list[int], b: list[int]) -> int:
        ox = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        oy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ox * oy

    for label in range(1, labels_count):
        cx, cy, cw, ch, area = [int(v) for v in stats[label]]
        if area < 3 or cw <= 0 or ch <= 0:
            continue
        gx1, gy1, gx2, gy2 = sx1 + cx, sy1 + cy, sx1 + cx + cw, sy1 + cy + ch
        if gx1 <= sx1 or gy1 <= sy1 or gx2 >= sx2 or gy2 >= sy2:
            continue
        comp_bbox = [gx1, gy1, gx2, gy2]
        overlaps_base = _overlap(comp_bbox, base_bbox) > 0
        if overlaps_base:
            continue
        y_overlap = max(0, min(gy2, by2) - max(gy1, by1))
        y_overlap_ratio = y_overlap / float(max(1, min(ch, base_h)))
        horizontal_gap = max(0, max(bx1 - gx2, gx1 - bx2))
        close_to_base = y_overlap_ratio >= 0.35 and horizontal_gap <= max(18, int(round(base_w * 0.30)))
        if not close_to_base:
            continue
        max_extra_w = max(18, min(32, int(round(base_w * 0.22))))
        max_extra_h = max(14, min(48, int(round(base_h * 0.85))))
        if cw > max_extra_w or ch > max_extra_h:
            continue
        ex1, ey1, ex2, ey2 = max(0, cx - 2), max(0, cy - 2), min(dark.shape[1], cx + cw + 2), min(
            dark.shape[0], cy + ch + 2
        )
        support = gray[ey1:ey2, ex1:ex2]
        if support.size == 0 or float(np.median(support)) < 210.0:
            continue
        component = (labels[cy : cy + ch, cx : cx + cw] == label)
        extra[gy1:gy2, gx1:gx2][component] = 255

    return extra if np.any(extra) else None


def _preserve_dark_pixels_outside_source_bbox(
    image_rgb: np.ndarray | None,
    mask: np.ndarray,
    text: dict,
    width: int,
    height: int,
    *,
    allowed_extra_mask: np.ndarray | None = None,
) -> np.ndarray:
    source_bbox = _text_source_limit_bbox(text, width, height, include_line_bbox=False)
    if (
        source_bbox is None
        or not isinstance(image_rgb, np.ndarray)
        or image_rgb.shape[:2] != (height, width)
        or not isinstance(mask, np.ndarray)
        or mask.shape[:2] != (height, width)
    ):
        return mask
    source_core = _mask_from_bbox(width, height, source_bbox, padding=0) > 0
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    allowed_extra = _coerce_mask_for_shape(allowed_extra_mask, (height, width)) > 0
    dark_outside_source = (gray <= 96) & (~source_core) & (~allowed_extra)
    if not np.any(dark_outside_source):
        return mask
    guarded = mask.astype(np.uint8).copy()
    guarded[(guarded > 0) & dark_outside_source] = 0
    return guarded


def _expand_tight_text_limit_to_white_region(
    image_rgb: np.ndarray,
    text: dict,
    limit_bbox: list[int] | None,
    width: int,
    height: int,
) -> list[int] | None:
    limit_bbox = _normalize_bbox(limit_bbox, width, height)
    text_bbox = _text_source_limit_bbox(text, width, height)
    if limit_bbox is None or text_bbox is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return limit_bbox
    limit_area = _bbox_area_value(limit_bbox)
    text_area = _bbox_area_value(text_bbox)
    if limit_area > max(text_area * 2.2, text_area + 4096):
        return limit_bbox

    x1, y1, x2, y2 = limit_bbox
    pad = max(24, min(96, int(round(max(x2 - x1, y2 - y1) * 1.4))))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(width, x2 + pad)
    ry2 = min(height, y2 + pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return limit_bbox

    roi = image_rgb[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return limit_bbox
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    bright = (gray >= 235).astype(np.uint8)
    if int(np.count_nonzero(bright)) < max(64, int(limit_area * 0.35)):
        return limit_bbox

    seed = np.zeros(bright.shape, dtype=np.uint8)
    sx1, sy1, sx2, sy2 = text_bbox
    seed[max(0, sy1 - ry1) : max(0, sy2 - ry1), max(0, sx1 - rx1) : max(0, sx2 - rx1)] = 1
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    best_label = 0
    best_overlap = 0
    for label in range(1, labels_count):
        overlap = int(np.count_nonzero((labels == label) & (seed > 0)))
        if overlap > best_overlap:
            best_label = label
            best_overlap = overlap
    if best_label <= 0 or best_overlap < 8:
        return limit_bbox

    cx, cy, cw, ch, area = [int(v) for v in stats[best_label]]
    candidate = _normalize_bbox([rx1 + cx, ry1 + cy, rx1 + cx + cw, ry1 + cy + ch], width, height)
    if candidate is None:
        return limit_bbox
    candidate_area = _bbox_area_value(candidate)
    if candidate_area < int(limit_area * 1.25):
        return limit_bbox
    if candidate_area > max(limit_area * 12, limit_area + 60_000):
        return limit_bbox
    return _bbox_union_values(limit_bbox, candidate)


def _resolve_fast_solid_limit_bbox(image_rgb: np.ndarray, text: dict, width: int, height: int) -> list[int] | None:
    limit_bbox = _solid_fill_limit_bbox(text, width, height)
    return _expand_tight_text_limit_to_white_region(image_rgb, text, limit_bbox, width, height)


def _expand_solid_fill_limit_for_text_geometry(
    text: dict,
    width: int,
    height: int,
    limit_bbox: list[int] | None,
) -> list[int] | None:
    expand_px = _fast_solid_line_expand_px() + 2
    for bbox in (
        _line_polygons_bbox(text, width, height, padding=expand_px),
        _normalize_bbox(text.get("text_pixel_bbox"), width, height),
        _normalize_bbox(text.get("bbox"), width, height),
    ):
        if bbox is not None:
            limit_bbox = _bbox_union_values(limit_bbox, bbox)
    if limit_bbox is None:
        return None
    return _normalize_bbox(limit_bbox, width, height)


def _fast_solid_line_geometry_mask(
    width: int,
    height: int,
    text: dict,
    image_rgb: np.ndarray | None = None,
    limit_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    mask = _strict_text_geometry_mask(width, height, text)
    if mask is None or not np.any(mask):
        return None
    expand_px = _fast_solid_line_expand_px()
    if expand_px > 0:
        kernel_size = expand_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    source_limit = _text_source_limit_mask(width, height, text, padding=max(1, expand_px))
    if source_limit is not None and np.any(source_limit):
        mask = cv2.bitwise_and(mask.astype(np.uint8), source_limit.astype(np.uint8))
    allowed_extra_mask: np.ndarray | None = None
    if isinstance(image_rgb, np.ndarray) and image_rgb.shape[:2] == (height, width):
        raw_source = _source_bbox_raw_text_mask(image_rgb, text, width, height)
        if raw_source is not None and np.any(raw_source):
            raw_expand_px = max(1, min(3, expand_px))
            raw_source = expand_text_mask(raw_source.astype(np.uint8), expand_px=raw_expand_px)
            if source_limit is not None and np.any(source_limit):
                raw_source = cv2.bitwise_and(raw_source.astype(np.uint8), source_limit.astype(np.uint8))
            mask = np.maximum(mask.astype(np.uint8), raw_source.astype(np.uint8))
        nearby_source = _nearby_raw_text_mask_from_image(image_rgb, text, width, height)
        if nearby_source is not None and np.any(nearby_source):
            raw_expand_px = max(1, min(3, expand_px))
            nearby_source = expand_text_mask(nearby_source.astype(np.uint8), expand_px=raw_expand_px)
            limit_bbox = _resolve_fast_solid_limit_bbox(image_rgb, text, width, height)
            if limit_bbox is not None:
                nearby_source = cv2.bitwise_and(
                    nearby_source.astype(np.uint8),
                    _mask_from_bbox(width, height, limit_bbox, padding=1).astype(np.uint8),
                )
            mask = np.maximum(mask.astype(np.uint8), nearby_source.astype(np.uint8))
            allowed_extra_mask = nearby_source
    mask = _preserve_dark_pixels_outside_source_bbox(
        image_rgb,
        mask,
        text,
        width,
        height,
        allowed_extra_mask=allowed_extra_mask,
    )
    if isinstance(limit_mask, np.ndarray) and limit_mask.shape[:2] == (height, width) and np.any(limit_mask):
        mask = cv2.bitwise_and(mask.astype(np.uint8), limit_mask.astype(np.uint8))
    else:
        limit_bbox = _resolve_fast_solid_limit_bbox(image_rgb, text, width, height) if isinstance(image_rgb, np.ndarray) else _solid_fill_limit_bbox(text, width, height)
        if limit_bbox is not None:
            bbox_limit_mask = _mask_from_bbox(width, height, limit_bbox, padding=1)
            mask = cv2.bitwise_and(mask.astype(np.uint8), bbox_limit_mask.astype(np.uint8))
    return mask if np.any(mask) else None


def _fast_fill_min_coverage(mask_source: str) -> float:
    base = float(os.environ.get("TRADUZAI_FAST_FILL_MIN_COVERAGE", "0.18"))
    if mask_source == "line_geometry":
        line_base = float(os.environ.get("TRADUZAI_FAST_FILL_LINE_MIN_COVERAGE", str(base)))
        return max(base, line_base)
    return base


def _solid_text_fill_mask(
    image_rgb: np.ndarray,
    text: dict,
    width: int,
    height: int,
    limit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str]:
    if text.get("line_polygons"):
        mask = _fast_solid_line_geometry_mask(width, height, text, image_rgb, limit_mask=limit_mask)
        if mask is not None and np.any(mask):
            return mask, "line_geometry"
        return None, "missing_text_geometry_mask"
    try:
        evidence = build_raw_text_mask_from_image(dict(text), image_rgb, image_rgb.shape)
    except Exception:
        evidence = None
    if not isinstance(evidence, np.ndarray) or not np.any(evidence):
        return None, "missing_koharu_text_evidence"
    expanded = expand_text_mask(evidence.astype(np.uint8), expand_px=5)
    if not isinstance(expanded, np.ndarray) or not np.any(expanded):
        return None, "missing_koharu_text_evidence"
    return expanded.astype(np.uint8), "raw_text_evidence"


def _apply_fast_solid_balloon_fill(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    rejection_reasons: dict[str, int] = {}
    fill_samples: list[dict] = []
    color_counts = {"white": 0, "black": 0, "colored": 0}

    def _reject(reason: str) -> None:
        rejection_reasons[reason or "unknown"] = rejection_reasons.get(reason or "unknown", 0) + 1

    def _record(stats: dict) -> dict:
        ocr_page["_strip_fast_solid_balloon_count"] = int(stats["solid_balloon_count"])
        ocr_page["_strip_fast_solid_white_count"] = int(color_counts["white"])
        ocr_page["_strip_fast_solid_black_count"] = int(color_counts["black"])
        ocr_page["_strip_fast_solid_colored_count"] = int(color_counts["colored"])
        ocr_page["_strip_remaining_inpaint_blocks"] = int(stats["remaining_blocks"])
        ocr_page["_strip_fast_solid_rejection_reasons"] = dict(rejection_reasons)
        ocr_page["_strip_fast_solid_fill_reject_reasons"] = dict(rejection_reasons)
        ocr_page["_strip_fast_solid_fill_samples"] = list(fill_samples)
        ocr_page["_strip_used_fast_solid_fill"] = bool(stats["solid_balloon_count"])
        return stats

    text_count = len([text for text in ocr_page.get("texts", []) if isinstance(text, dict)])
    if _fast_local_balloon_fill_enabled():
        rejection_reasons["fast_local_enabled"] = max(1, text_count)
        return band_rgb, vision_blocks, _record({"solid_balloon_count": 0, "remaining_blocks": len(vision_blocks)})
    if _fast_white_balloon_fill_enabled():
        rejection_reasons["fast_white_enabled"] = max(1, text_count)
        return band_rgb, vision_blocks, _record({"solid_balloon_count": 0, "remaining_blocks": len(vision_blocks)})
    if not _fast_solid_balloon_fill_enabled():
        rejection_reasons["disabled"] = max(1, text_count)
        return band_rgb, vision_blocks, _record({"solid_balloon_count": 0, "remaining_blocks": len(vision_blocks)})
    if not isinstance(band_rgb, np.ndarray) or band_rgb.size == 0:
        return band_rgb, vision_blocks, _record({"solid_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    height, width = band_rgb.shape[:2]
    if not vision_blocks:
        for text in ocr_page.get("texts", []) or []:
            rejection_reason = _fast_solid_rejection_reason(text)
            if rejection_reason:
                _propagate_existing_mask_evidence_decision_flags(ocr_page, text)
                _reject(rejection_reason)
        return band_rgb, vision_blocks, _record({"solid_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    result = band_rgb.copy()
    filled_bboxes: list[list[int]] = []
    filled_text_keys: set[str] = set()
    filled_mask = np.zeros((height, width), dtype=np.uint8)

    for text in ocr_page.get("texts", []) or []:
        rejection_reason = _fast_solid_rejection_reason(text)
        if rejection_reason:
            _propagate_existing_mask_evidence_decision_flags(ocr_page, text)
            _reject(rejection_reason)
            continue
        text_bbox = _line_polygons_bbox(text, width, height) or _normalize_bbox(
            text.get("text_pixel_bbox"),
            width,
            height,
        )
        if text_bbox is None:
            _reject("missing_text_bbox")
            continue
        real_bubble_mask, bubble_rejection = _real_bubble_mask_for_text(ocr_page, text, width, height)
        if real_bubble_mask is None:
            _reject(bubble_rejection)
            continue
        balloon_limit = _safe_real_bubble_interior_mask(real_bubble_mask, width, height)
        real_bubble_bbox = _bbox_from_binary_mask(real_bubble_mask)
        if real_bubble_bbox is None:
            _reject("missing_real_bubble_mask")
            continue
        text_fill_mask, mask_source = _solid_text_fill_mask(
            band_rgb,
            text,
            width,
            height,
            limit_mask=balloon_limit,
        )
        if text_fill_mask is None or not np.any(text_fill_mask):
            mask_evidence = consolidate_mask_evidence(
                text,
                kind="none",
                raw_mask_pixels=0,
                expanded_mask_pixels=0,
                evidence_score=0.0,
                fast_fill_reject_reasons=["raw_mask_pixels_zero"],
            )
            _propagate_mask_evidence_decision_flags(ocr_page, text, mask_evidence)
            _reject("raw_mask_pixels_zero")
            continue
        text_fill_mask, clip_rejection = _clip_fast_fill_text_mask_to_real_bubble(
            text_fill_mask,
            real_bubble_mask,
            width,
            height,
        )
        if text_fill_mask is None or not np.any(text_fill_mask):
            mask_evidence = consolidate_mask_evidence(
                text,
                kind="none",
                raw_mask_pixels=0,
                expanded_mask_pixels=0,
                evidence_score=0.0,
                fast_fill_reject_reasons=[clip_rejection or "text_mask_outside_bubble"],
            )
            _propagate_mask_evidence_decision_flags(ocr_page, text, mask_evidence)
            _reject(clip_rejection or "text_mask_outside_bubble")
            continue
        raw_mask_pixels = int(np.count_nonzero(text_fill_mask))
        mask_evidence = consolidate_mask_evidence(
            text,
            kind="glyph_segmentation" if mask_source == "line_geometry" else "ocr_pixels",
            raw_mask_pixels=raw_mask_pixels,
            expanded_mask_pixels=raw_mask_pixels,
            evidence_score=1.0,
        )
        _propagate_mask_evidence_decision_flags(ocr_page, text, mask_evidence)
        rejection_reason = _fast_solid_rejection_reason(text)
        if rejection_reason:
            _reject(rejection_reason)
            continue
        evidence_rejection = _koharu_style_fast_white_evidence_rejection_reason(
            band_rgb,
            text,
            text_fill_mask,
        )
        if evidence_rejection:
            _reject(evidence_rejection)
            continue
        fill_color, metadata = _sample_solid_fill_color_for_mask(band_rgb, text_fill_mask, balloon_limit)
        metadata["text_id"] = str(text.get("id") or text.get("text_id") or text.get("trace_id") or "")
        metadata["mask_source"] = mask_source
        metadata["mask_evidence"] = dict(mask_evidence)
        if not metadata.get("accepted") or fill_color is None:
            _reject(str(metadata.get("reason") or "solid_fill_rejected"))
            continue
        bucket = _solid_fill_color_bucket(fill_color)
        profiles = {
            str(text.get("layout_profile") or "").strip().lower(),
            str(text.get("block_profile") or "").strip().lower(),
            str(text.get("background_type") or "").strip().lower(),
        }
        textured_profile = bool(profiles & {"textured", "textured_background", "standard"})
        explicit_solid_profile = bool(
            profiles
            & {
                "connected_balloon",
                "solid_color",
                "solid_colored",
                "solid_dark",
                "dark_panel",
                "white_balloon",
            }
        )
        if bucket == "colored" and textured_profile and not explicit_solid_profile:
            _reject("textured_background")
            continue
        filled_from_original = _fill_mask_solid(band_rgb, text_fill_mask, color=fill_color)
        changed_mask = np.any(filled_from_original != band_rgb, axis=2)
        if not np.any(changed_mask):
            _reject("no_fast_fill_change")
            continue
        coverage_bbox = text_bbox
        if mask_source == "line_geometry":
            coverage_bbox = _line_polygons_bbox(text, width, height, padding=0) or text_bbox
        coverage_mask = _mask_from_bbox(width, height, coverage_bbox) > 0
        changed_in_text = int(np.count_nonzero(changed_mask & coverage_mask))
        min_changed = max(24, int(round(max(1, int(np.count_nonzero(text_fill_mask))) * 0.06)))
        if changed_in_text < min_changed:
            _append_inpaint_decision_flag(ocr_page, "fast_fill_insufficient_coverage")
            _reject("fast_fill_insufficient_coverage")
            continue
        text_bbox_area = max(
            1,
            (int(coverage_bbox[2]) - int(coverage_bbox[0]))
            * (int(coverage_bbox[3]) - int(coverage_bbox[1])),
        )
        changed_coverage = changed_in_text / float(text_bbox_area)
        min_coverage = _fast_fill_min_coverage(mask_source)
        if changed_coverage < min_coverage:
            _append_inpaint_decision_flag(ocr_page, "fast_fill_insufficient_coverage")
            _reject("fast_fill_insufficient_coverage")
            continue
        residual_ratio = _fast_fill_residual_edge_ratio(filled_from_original, coverage_bbox, text_fill_mask)
        max_residual = float(os.environ.get("TRADUZAI_FAST_FILL_MAX_RESIDUAL_EDGE_RATIO", "0.08"))
        if residual_ratio > max_residual:
            _append_inpaint_decision_flag(ocr_page, "text_residual_after_inpaint_suspected")
            _reject("text_residual_after_inpaint_suspected")
            continue

        result[changed_mask] = filled_from_original[changed_mask]
        filled_mask[changed_mask] = 255
        if bucket in color_counts:
            color_counts[bucket] += 1
        metadata["color_bucket"] = bucket
        metadata["fill_pixels"] = int(np.count_nonzero(changed_mask))
        metadata["fill_bbox"] = _bbox_from_binary_mask(changed_mask.astype(np.uint8))
        metadata["fast_fill_verified"] = True
        metadata["fast_fill_text_bbox_coverage"] = round(changed_coverage, 4)
        metadata["fast_fill_coverage_bbox"] = list(coverage_bbox)
        metadata["fast_fill_min_coverage"] = round(min_coverage, 4)
        if mask_source == "line_geometry":
            metadata["line_geometry_expand_px"] = _fast_solid_line_expand_px()
        metadata["fast_fill_residual_edge_ratio"] = round(residual_ratio, 4)
        fill_samples.append(metadata)
        filled_bboxes.append(_bbox_union_values(real_bubble_bbox, text_bbox) or real_bubble_bbox)

    if not filled_bboxes:
        return band_rgb, vision_blocks, _record({"solid_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    ocr_page["_strip_fast_solid_fill_samples"] = list(fill_samples)
    verified_text_ids = _fast_solid_verified_text_ids(ocr_page)
    remaining_blocks = []
    for block in vision_blocks:
        block_has_id = any(_fast_fill_id_aliases(block.get(key)) for key in ("id", "text_id", "trace_id", "text_instance_id"))
        if block_has_id and not _block_has_verified_fast_solid_fill(block, verified_text_ids):
            remaining_blocks.append(block)
            continue
        if _block_is_covered_by_fast_fill(block, filled_bboxes, width, height, filled_mask):
            continue
        remaining_blocks.append(block)
    return result, remaining_blocks, _record(
        {"solid_balloon_count": len(filled_bboxes), "remaining_blocks": len(remaining_blocks)}
    )


def _solid_fast_fill_override_allowed(text: dict, solid_fill_metadata: dict) -> bool:
    if not isinstance(text, dict) or not isinstance(solid_fill_metadata, dict):
        return False
    if not solid_fill_metadata.get("accepted"):
        return False
    color = solid_fill_metadata.get("color")
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return False
    try:
        luma = sum(float(v) for v in color[:3]) / 3.0
    except Exception:
        return False
    max_std = solid_fill_metadata.get("raw_max_std", solid_fill_metadata.get("max_std"))
    p95_delta = solid_fill_metadata.get("raw_p95_abs_delta", solid_fill_metadata.get("p95_abs_delta"))
    try:
        low_variation = float(max_std) <= 9.0 and float(p95_delta) <= 24.0
    except Exception:
        low_variation = True
    return low_variation and 0.0 <= luma <= 255.0


def _solid_fill_raw_variation_high(solid_fill_metadata: dict) -> bool:
    if not isinstance(solid_fill_metadata, dict):
        return True
    max_std = solid_fill_metadata.get("raw_max_std", solid_fill_metadata.get("max_std"))
    p95_delta = solid_fill_metadata.get("raw_p95_abs_delta", solid_fill_metadata.get("p95_abs_delta"))
    try:
        raw_high = float(max_std) > 9.0 or float(p95_delta) > 24.0
    except Exception:
        return False
    if not raw_high:
        return False
    try:
        median_luma = float(solid_fill_metadata.get("median_luma"))
        median_chroma = float(solid_fill_metadata.get("median_chroma"))
    except Exception:
        median_luma = 0.0
        median_chroma = 255.0
    if solid_fill_metadata.get("robust_dominant_sample") and median_luma >= 220.0 and median_chroma <= 24.0:
        return False
    return True


def _looks_like_solid_dark_fill_region(
    image_rgb: np.ndarray,
    balloon_limit: np.ndarray,
    text_bbox: list[int],
) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return False
    if not isinstance(balloon_limit, np.ndarray) or not np.any(balloon_limit):
        return False
    height, width = image_rgb.shape[:2]
    text_guard = _mask_from_bbox(width, height, text_bbox, padding=8)
    sample_mask = cv2.bitwise_and(
        (balloon_limit > 0).astype(np.uint8) * 255,
        cv2.bitwise_not((text_guard > 0).astype(np.uint8) * 255),
    )
    if int(np.count_nonzero(sample_mask)) < 64:
        return False
    sample = image_rgb[sample_mask > 0].astype(np.float32)
    luma = np.mean(sample, axis=1)
    median_luma = float(np.median(luma))
    p90_luma = float(np.percentile(luma, 90))
    std_luma = float(np.std(luma))
    return median_luma <= 12.0 and p90_luma <= 28.0 and std_luma <= 12.0


def _apply_fast_white_balloon_fill(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    rejection_reasons: dict[str, int] = {}
    evidence_constrained_fill_count = 0
    solid_fill_samples: list[dict] = []
    solid_fill_reject_reasons: dict[str, int] = {}

    def _reject(reason: str) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    def _record(stats: dict) -> dict:
        ocr_page["_strip_fast_white_balloon_count"] = stats["white_balloon_count"]
        ocr_page["_strip_remaining_inpaint_blocks"] = stats["remaining_blocks"]
        ocr_page["_strip_fast_white_rejection_reasons"] = dict(rejection_reasons)
        ocr_page["_strip_fast_white_evidence_constrained_fill_count"] = int(evidence_constrained_fill_count)
        ocr_page["_strip_fast_solid_fill_samples"] = list(solid_fill_samples)
        ocr_page["_strip_fast_solid_fill_reject_reasons"] = dict(solid_fill_reject_reasons)
        return stats

    if not _fast_white_balloon_fill_enabled():
        text_count = len([text for text in ocr_page.get("texts", []) if isinstance(text, dict)])
        rejection_reasons["disabled"] = max(1, text_count)
        return band_rgb, vision_blocks, _record({"white_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    from vision_stack.runtime import _build_white_balloon_text_line_fill_mask

    height, width = band_rgb.shape[:2]
    result = band_rgb.copy()
    filled_bboxes: list[list[int]] = []
    filled_text_keys: set[str] = set()
    filled_mask = np.zeros((height, width), dtype=np.uint8)

    for text in ocr_page.get("texts", []):
        rejection_reason = _fast_white_rejection_reason(text)
        if _is_false_white_card_candidate(band_rgb, text):
            _reject("false_white_card_panel")
            continue
        component_fill_allowed_rejection = rejection_reason in {
            "narration_disabled",
        }
        if rejection_reason and rejection_reason != "textured_or_dark_region" and not component_fill_allowed_rejection:
            if rejection_reason.startswith("mask_evidence:"):
                _propagate_existing_mask_evidence_decision_flags(ocr_page, text)
            _reject(rejection_reason)
            continue
        text_bbox = _line_polygons_bbox(text, width, height) or _normalize_bbox(
            text.get("text_pixel_bbox"),
            width,
            height,
        )
        if text_bbox is None:
            _reject("missing_text_bbox")
            continue
        real_bubble_mask, bubble_rejection = _real_bubble_mask_for_text(ocr_page, text, width, height)
        if real_bubble_mask is None:
            _reject(bubble_rejection)
            continue
        used_component_text_fill = False
        balloon_limit = _safe_real_bubble_interior_mask(real_bubble_mask, width, height)
        real_bubble_bbox = _bbox_from_binary_mask(real_bubble_mask)
        if real_bubble_bbox is None:
            _reject("missing_real_bubble_mask")
            continue
        balloon_bbox = real_bubble_bbox
        if rejection_reason == "textured_or_dark_region" and not _looks_like_solid_dark_fill_region(
            band_rgb,
            balloon_limit,
            text_bbox,
        ):
            _reject(rejection_reason)
            continue
        used_bbox_fallback_fill = False
        text_fill_mask = _text_geometry_mask(width, height, text) if text.get("line_polygons") else None
        if text_fill_mask is None or not np.any(text_fill_mask):
            text_fill_mask = _build_white_balloon_text_line_fill_mask(band_rgb, text)
            used_component_text_fill = text_fill_mask is not None and np.any(text_fill_mask)
        if (text_fill_mask is None or not np.any(text_fill_mask)) and text.get("text_pixel_bbox"):
            text_fill_mask = _text_geometry_mask(width, height, text)
        if text_fill_mask is None or not np.any(text_fill_mask):
            _reject("missing_text_geometry_mask" if not component_fill_allowed_rejection else rejection_reason)
            continue
        text_mask = np.where(text_fill_mask > 0, 255, 0).astype(np.uint8)
        text_fill_mask, clip_rejection = _clip_fast_fill_text_mask_to_real_bubble(
            text_fill_mask,
            real_bubble_mask,
            width,
            height,
        )
        if text_fill_mask is None or not np.any(text_fill_mask):
            _reject(clip_rejection or "text_geometry_outside_balloon")
            continue
        fill_color, solid_fill_metadata = _sample_solid_fill_color_for_mask(
            band_rgb,
            text_fill_mask,
            balloon_limit,
        )
        solid_fill_metadata["text_id"] = str(text.get("id") or text.get("text_id") or text.get("trace_id") or "")
        solid_background_ok = bool(solid_fill_metadata.get("accepted"))
        if solid_background_ok:
            if _solid_fill_raw_variation_high(solid_fill_metadata):
                reason = "background_variation_high"
                solid_fill_reject_reasons[reason] = solid_fill_reject_reasons.get(reason, 0) + 1
                _reject(reason)
                continue
            solid_fill_samples.append(solid_fill_metadata)
        else:
            reason = str(solid_fill_metadata.get("reason") or "solid_fill_rejected")
            solid_fill_reject_reasons[reason] = solid_fill_reject_reasons.get(reason, 0) + 1
            _reject(reason)
            continue
        use_sampled_fill_color = True
        if rejection_reason == "textured_or_dark_region":
            solid_override_ok = _solid_fast_fill_override_allowed(text, solid_fill_metadata)
            if not solid_override_ok:
                _reject(rejection_reason)
                continue
            use_sampled_fill_color = _solid_fast_fill_override_allowed(text, solid_fill_metadata)

        resolved = real_bubble_bbox
        if not used_component_text_fill and _looks_translucent_or_textured_background(band_rgb, balloon_bbox, text_mask):
            _reject("translucent_background")
            continue
        if _looks_saturated_colored_background(band_rgb, balloon_bbox, text_mask) and not solid_background_ok:
            _reject("colored_background")
            continue
        evidence_rejection = _koharu_style_fast_white_evidence_rejection_reason(
            band_rgb,
            text,
            text_fill_mask,
        )
        if evidence_rejection and not used_bbox_fallback_fill:
            _reject(evidence_rejection)
            continue
        used_koharu_evidence_fill = False
        text_area_for_evidence = max(1, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
        image_area_for_evidence = max(1, width * height)
        text_w_ratio = (text_bbox[2] - text_bbox[0]) / float(max(1, width))
        text_h_ratio = (text_bbox[3] - text_bbox[1]) / float(max(1, height))
        large_no_line_support = text_area_for_evidence >= max(45_000, int(image_area_for_evidence * 0.12))
        constrain_to_koharu_evidence = (
            not text.get("line_polygons")
            and (
                text_area_for_evidence >= max(80_000, int(image_area_for_evidence * 0.18))
                or (large_no_line_support and max(text_w_ratio, text_h_ratio) >= 0.60)
                or (large_no_line_support and text_w_ratio >= 0.35 and text_h_ratio >= 0.45)
            )
        )
        if constrain_to_koharu_evidence:
            try:
                evidence_mask = build_raw_text_mask_from_image(dict(text), band_rgb, band_rgb.shape)
            except Exception:
                evidence_mask = None
            if isinstance(evidence_mask, np.ndarray) and np.any(evidence_mask):
                evidence_fill = expand_text_mask(evidence_mask.astype(np.uint8), expand_px=5)
                evidence_bbox = _bbox_from_binary_mask(evidence_mask)
                if evidence_bbox:
                    text_h = max(1, text_bbox[3] - text_bbox[1])
                    ev_x1, ev_y1, ev_x2, ev_y2 = evidence_bbox
                    evidence_h = max(1, ev_y2 - ev_y1)
                    evidence_bottom_ratio = (ev_y2 - text_bbox[1]) / float(text_h)
                    compact_validated_source = (
                        evidence_h <= max(32, int(round(text_h * 0.55)))
                        and evidence_bottom_ratio <= 0.62
                    )
                    if compact_validated_source:
                        source_rects: list[list[int]] = []
                        for key in ("_merged_source_bboxes", "merged_source_bboxes", "source_bboxes"):
                            raw_boxes = text.get(key)
                            if not isinstance(raw_boxes, (list, tuple)):
                                continue
                            for raw_box in raw_boxes:
                                source_box = _normalize_bbox(raw_box, width, height)
                                if source_box is None:
                                    continue
                                sx1, sy1, sx2, sy2 = source_box
                                overlap = int(np.count_nonzero(evidence_mask[sy1:sy2, sx1:sx2] > 0))
                                source_h = max(1, sy2 - sy1)
                                source_bottom_ratio = (sy2 - text_bbox[1]) / float(text_h)
                                if overlap >= 16 and source_h <= max(48, int(round(text_h * 0.55))) and source_bottom_ratio <= 0.66:
                                    source_rects.append(source_box)
                        if not source_rects:
                            source_rects.append(evidence_bbox)
                        rect_mask = np.zeros((height, width), dtype=np.uint8)
                        for source_rect in source_rects:
                            rect_bbox = _expanded_bbox(width, height, source_rect, padding=8)
                            if rect_bbox:
                                rx1, ry1, rx2, ry2 = rect_bbox
                                rect_mask[ry1:ry2, rx1:rx2] = 255
                        if np.any(rect_mask):
                            evidence_fill = np.maximum(evidence_fill.astype(np.uint8), rect_mask)
                evidence_fill = cv2.bitwise_and(evidence_fill.astype(np.uint8), balloon_limit.astype(np.uint8))
                if np.any(evidence_fill):
                    text_fill_mask = evidence_fill
                    used_koharu_evidence_fill = True
                    evidence_constrained_fill_count += 1
                    fill_color, solid_fill_metadata = _sample_solid_fill_color_for_mask(
                        band_rgb,
                        text_fill_mask,
                        balloon_limit,
                    )
                    solid_fill_metadata["text_id"] = str(text.get("id") or text.get("text_id") or text.get("trace_id") or "")
                    if solid_fill_metadata.get("accepted"):
                        solid_fill_samples.append(solid_fill_metadata)
        if used_koharu_evidence_fill and fill_color is not None:
            pass
        elif not use_sampled_fill_color:
            fill_color = (255, 255, 255)
        elif fill_color is None:
            fill_color = (255, 255, 255)
        filled_from_original = _fill_mask_solid(band_rgb, text_fill_mask, color=fill_color)
        changed_mask = np.any(filled_from_original != band_rgb, axis=2)
        if not np.any(changed_mask):
            _reject("no_fast_fill_change")
            continue
        if not text.get("line_polygons"):
            text_area = max(1, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
            if used_koharu_evidence_fill:
                text_area = max(1, int(np.count_nonzero(text_fill_mask)))
            changed_in_text = int(np.count_nonzero(changed_mask & (_mask_from_bbox(width, height, text_bbox) > 0)))
            min_changed = max(48 if used_koharu_evidence_fill else 96, int(round(text_area * (0.08 if used_koharu_evidence_fill else 0.04))))
            if changed_in_text < min_changed:
                _reject("insufficient_fast_fill_coverage")
                continue
        result[changed_mask] = filled_from_original[changed_mask]
        filled_mask[changed_mask] = 255
        filled_bboxes.append(_bbox_union_values(resolved, text_bbox) or resolved)
        for key_name in ("id", "text_id", "trace_id", "text_instance_id"):
            value = text.get(key_name)
            if value is not None:
                filled_text_keys.add(str(value))

    if not filled_bboxes:
        return band_rgb, vision_blocks, _record({"white_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    remaining_blocks = [
        block
        for block in vision_blocks
        if not _block_is_covered_by_fast_fill(block, filled_bboxes, width, height, filled_mask)
    ]
    stats = {
        "white_balloon_count": len(filled_bboxes),
        "remaining_blocks": len(remaining_blocks),
    }
    if filled_text_keys:
        flags_to_remove = {"mask_outside_balloon", "mask_outside_balloon_critical", "real_inpaint_skipped_unsafe_mask"}
        for collection_key in ("texts", "_vision_blocks"):
            for item in ocr_page.get(collection_key) or []:
                if not isinstance(item, dict):
                    continue
                if any(str(item.get(key_name) or "") in filled_text_keys for key_name in ("id", "text_id", "trace_id", "text_instance_id")):
                    _remove_qa_flags_from_item(item, flags_to_remove)
        unsafe_samples = [
            sample
            for sample in ocr_page.get("_strip_unsafe_inpaint_block_samples") or []
            if not isinstance(sample, dict)
            or not any(str(sample.get(key_name) or "") in filled_text_keys for key_name in ("id", "text_id", "trace_id", "text_instance_id"))
        ]
        if unsafe_samples:
            ocr_page["_strip_unsafe_inpaint_block_samples"] = unsafe_samples
            reasons: dict[str, int] = {}
            for sample in unsafe_samples:
                if not isinstance(sample, dict):
                    continue
                reason = str(sample.get("reason") or "").strip()
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
            if reasons:
                ocr_page["_strip_unsafe_inpaint_block_reasons"] = reasons
                ocr_page["_strip_unsafe_inpaint_block_count"] = sum(reasons.values())
            else:
                ocr_page.pop("_strip_unsafe_inpaint_block_reasons", None)
                ocr_page.pop("_strip_unsafe_inpaint_block_count", None)
        else:
            ocr_page.pop("_strip_unsafe_inpaint_block_samples", None)
            ocr_page.pop("_strip_unsafe_inpaint_block_reasons", None)
            ocr_page.pop("_strip_unsafe_inpaint_block_count", None)
        decision_flags = [
            str(flag)
            for flag in ocr_page.get("_strip_inpaint_decision_flags") or []
            if str(flag).strip() not in flags_to_remove
        ]
        if decision_flags:
            ocr_page["_strip_inpaint_decision_flags"] = decision_flags
        else:
            ocr_page.pop("_strip_inpaint_decision_flags", None)
    return result, remaining_blocks, _record(stats)


def _text_allows_fast_local_fill(text: dict) -> bool:
    return not _fast_local_rejection_reason(text)


def _fast_local_rejection_reason(text: dict) -> str:
    if not isinstance(text, dict):
        return "invalid_text"
    if _route_action_blocks_inpaint(text):
        return "route_action_no_inpaint"
    qa_reason = _fast_fill_blocking_qa_reason(text)
    if qa_reason:
        return qa_reason
    mask_evidence_reason = _fast_fill_mask_evidence_rejection_reason(text)
    if mask_evidence_reason:
        return mask_evidence_reason
    has_text_geometry = bool(text.get("line_polygons") or text.get("text_pixel_bbox"))
    if has_text_geometry and _metadata_background_color(text) is not None:
        if text.get("line_polygons") or text.get("text_pixel_bbox"):
            return ""
    if has_text_geometry:
        return ""
    return "missing_text_geometry"


def _mask_from_bbox(width: int, height: int, bbox: list[int], padding: int = 2) -> np.ndarray:
    return bbox_to_octagon_mask(width, height, bbox, padding=padding)


def _bbox_from_binary_mask(mask: np.ndarray) -> list[int] | None:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _expanded_bbox(width: int, height: int, bbox: list[int], padding: int) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    return _normalize_bbox([x1 - padding, y1 - padding, x2 + padding, y2 + padding], width, height)


def _local_context_bbox_for_text(text: dict, width: int, height: int) -> list[int] | None:
    anchor = _line_polygons_bbox(text, width, height) or _normalize_bbox(
        text.get("text_pixel_bbox"),
        width,
        height,
    )
    if anchor is None:
        return None
    box_w = max(1, anchor[2] - anchor[0])
    box_h = max(1, anchor[3] - anchor[1])
    pad = max(10, min(48, int(round(max(box_w, box_h) * 0.45))))
    return _expanded_bbox(width, height, anchor, padding=pad) or anchor


def _looks_translucent_or_textured_background(
    image_rgb: np.ndarray,
    sample_bbox: list[int],
    text_mask: np.ndarray | None = None,
) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return False
    height, width = image_rgb.shape[:2]
    bbox = _normalize_bbox(sample_bbox, width, height)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    sample_mask = np.zeros((height, width), dtype=np.uint8)
    sample_mask[y1:y2, x1:x2] = 255
    if isinstance(text_mask, np.ndarray) and text_mask.shape[:2] == sample_mask.shape:
        exclusion = cv2.dilate(
            (text_mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        sample_mask = cv2.bitwise_and(sample_mask, cv2.bitwise_not(exclusion))
    if int(np.count_nonzero(sample_mask)) < 64:
        return False
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb.astype(np.uint8)
    bright_sample = (sample_mask > 0) & (gray >= 205)
    pixels = gray[bright_sample].astype(np.float32)
    if pixels.size < 64:
        return False
    mean_luma = float(np.mean(pixels))
    if mean_luma < 205.0:
        return False
    spread = float(np.percentile(pixels, 95) - np.percentile(pixels, 5))
    std = float(np.std(pixels))
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)[bright_sample]
    grad_p90 = float(np.percentile(grad, 90)) if grad.size else 0.0
    return spread >= 14.0 or std >= 5.5 or grad_p90 >= 18.0


def _looks_saturated_colored_background(
    image_rgb: np.ndarray,
    sample_bbox: list[int],
    text_mask: np.ndarray | None = None,
) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return False
    height, width = image_rgb.shape[:2]
    bbox = _normalize_bbox(sample_bbox, width, height)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    sample_mask = np.zeros((height, width), dtype=np.uint8)
    sample_mask[y1:y2, x1:x2] = 255
    if isinstance(text_mask, np.ndarray) and text_mask.shape[:2] == sample_mask.shape:
        exclusion = cv2.dilate(
            (text_mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        sample_mask = cv2.bitwise_and(sample_mask, cv2.bitwise_not(exclusion))
    if int(np.count_nonzero(sample_mask)) < 64:
        return False

    sample = image_rgb[sample_mask > 0].astype(np.float32)
    luma = (sample[:, 0] * 0.299) + (sample[:, 1] * 0.587) + (sample[:, 2] * 0.114)
    max_channel = np.maximum(np.max(sample, axis=1), 1.0)
    chroma = np.max(sample, axis=1) - np.min(sample, axis=1)
    saturation = chroma / max_channel
    useful = (luma >= 32.0) & (luma <= 246.0)
    if int(np.count_nonzero(useful)) < 64:
        return False
    median_chroma = float(np.median(chroma[useful]))
    p75_chroma = float(np.percentile(chroma[useful], 75))
    median_saturation = float(np.median(saturation[useful]))
    return (median_chroma >= 16.0 and median_saturation >= 0.08) or p75_chroma >= 28.0


def _try_solid_background_text_fill(
    image_rgb: np.ndarray,
    text_bbox: list[int],
    fill_bbox: list[int],
) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    text_bbox = _normalize_bbox(text_bbox, width, height)
    fill_bbox = _normalize_bbox(fill_bbox, width, height)
    if text_bbox is None or fill_bbox is None:
        return None
    text_mask = _mask_from_bbox(width, height, text_bbox, padding=8)
    if _looks_translucent_or_textured_background(image_rgb, fill_bbox, text_mask):
        return None

    context_bbox = [
        min(fill_bbox[0], text_bbox[0] - 24),
        min(fill_bbox[1], text_bbox[1] - 24),
        max(fill_bbox[2], text_bbox[2] + 24),
        max(fill_bbox[3], text_bbox[3] + 24),
    ]
    context_bbox = _normalize_bbox(context_bbox, width, height)
    if context_bbox is None:
        return None

    cx1, cy1, cx2, cy2 = context_bbox
    local = image_rgb[cy1:cy2, cx1:cx2]
    if local.size == 0:
        return None

    tx1, ty1, tx2, ty2 = text_bbox
    local_text_bbox = [tx1 - cx1, ty1 - cy1, tx2 - cx1, ty2 - cy1]
    local_text_mask = _mask_from_bbox(local.shape[1], local.shape[0], local_text_bbox, padding=8)
    sample = local[local_text_mask == 0]
    if sample.size == 0 or len(sample) < 64:
        return None

    sample_f = sample.astype(np.float32)
    median = np.median(sample_f, axis=0)
    std = np.sqrt(np.mean(np.square(sample_f - median[None, :]), axis=0))
    median_luma = float(np.mean(median))
    text_area = max(1, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
    max_std = max(float(v) for v in std)
    dark_panel_sample = False
    if text_area > 24_000 and median_luma <= 12.0:
        sample_luma = np.mean(sample_f, axis=1)
        dark_panel_sample = (
            max_std <= 16.0
            and float(np.percentile(sample_luma, 90)) <= 28.0
            and float(np.percentile(sample_luma, 98)) <= 80.0
        )
    if max_std > 10.0 and not dark_panel_sample:
        return None
    if median_luma <= 32.0:
        if text_area > 24_000:
            return None
    elif median_luma >= 238.0:
        pass
    else:
        return None

    region = image_rgb[ty1:ty2, tx1:tx2].astype(np.float32)
    if region.size == 0:
        return None
    contrast = float(np.max(np.abs(region - median[None, None, :])))
    if contrast < 32.0:
        return None

    bbox_width = text_bbox[2] - text_bbox[0]
    bbox_height = text_bbox[3] - text_bbox[1]
    fill_padding = max(8, min(24, int(round(max(bbox_width, bbox_height) * 0.08))))
    fill_bbox = _expanded_bbox(width, height, text_bbox, padding=fill_padding)
    if fill_bbox is None:
        return None
    fx1, fy1, fx2, fy2 = fill_bbox
    result = image_rgb.copy()
    fill = np.asarray([int(round(float(v))) for v in median], dtype=np.uint8)
    fill_mask = _mask_from_bbox(width, height, text_bbox, padding=fill_padding)
    if not np.any(fill_mask):
        result[fy1:fy2, fx1:fx2] = fill
    else:
        result[fill_mask > 0] = fill
    return result


def _is_dark_panel_text_candidate(text: dict) -> bool:
    if not isinstance(text, dict) or not _route_action_allows_local_dark_panel_fill(text):
        return False
    profiles = {
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("background_type") or "").strip().lower(),
    }
    background = _rgb_luma_chroma(text.get("background_rgb"))
    background_dark = bool(background is not None and background[0] <= 90.0)
    if not (profiles & {"dark_panel", "solid_dark", "dark"} or background_dark):
        return False
    if not (text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("bbox")):
        return False
    return True


def _text_has_dark_visual_context(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    profiles = {
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("render_profile") or "").strip().lower(),
        str(text.get("background_type") or "").strip().lower(),
        str(text.get("balloon_type") or "").strip().lower(),
    }
    if profiles & {"dark", "dark_panel", "solid_dark", "dark_bubble", "connected_dark_bubble"}:
        return True
    source = str(text.get("bubble_mask_source") or "").strip().lower()
    if source in {"image_dark_panel_mask", "image_dark_bubble_mask", "derived_card_panel_mask"}:
        return True
    style_origin = str(text.get("style_origin") or "").strip().lower()
    if style_origin in {"auto_dark_panel_glow", "source_dark_panel_glow"}:
        return True
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if flags & {"auto_dark_panel_glow_fallback", "original_dark_panel_effect_colors"}:
        return True
    background = _rgb_luma_chroma(text.get("background_rgb"))
    return bool(background is not None and background[0] <= 160.0)


def _text_is_white_balloon_context(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    if _text_has_dark_visual_context(text):
        background = _rgb_luma_chroma(text.get("background_rgb"))
        if background is None or background[0] < 214.0:
            return False
    values = {
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("balloon_type") or "").strip().lower(),
        str(text.get("bubble_mask_source") or "").strip().lower(),
    }
    if values & {
        "white",
        "white_balloon",
        "image_contour_bubble_mask",
        "image_rect_bubble_mask",
        "image_white_bubble_mask",
        "derived_white_bubble_mask",
        "derived_white_crop",
    }:
        return True
    background = _rgb_luma_chroma(text.get("background_rgb"))
    return bool(background is not None and background[0] >= 228.0 and background[1] <= 18.0)


def _unsafe_white_balloon_requires_real_inpaint(text: dict) -> bool:
    if not isinstance(text, dict) or not _text_is_white_balloon_context(text):
        return False
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    unsafe = bool(
        str(text.get("route_reason") or "").strip().lower()
        in {"mask_outside_balloon_critical", "missing_real_bubble_mask"}
        or flags
        & {
            "mask_outside_balloon",
            "mask_outside_balloon_critical",
            "missing_real_bubble_mask",
            "rejected_derived_bubble_mask",
            "debug_derived_bubble_mask_rejected",
        }
    )
    if not unsafe:
        return False
    source = str(text.get("bubble_mask_source") or "").strip().lower()
    if source in {
        "image_contour_bubble_mask",
        "image_rect_bubble_mask",
        "image_white_bubble_mask",
        "derived_white_bubble_mask",
        "derived_white_crop",
    }:
        return True
    return bool(str(text.get("balloon_type") or "").strip().lower() in {"white", "white_balloon", "speech_balloon"})


def _is_false_white_card_candidate(image_rgb: np.ndarray, text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source != "image_white_bubble_mask":
        return False
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    evidence = text.get("mask_evidence")
    has_fast_ocr_evidence = bool(
        isinstance(evidence, dict)
        and evidence.get("fast_fill_allowed") is True
        and str(evidence.get("kind") or "").strip().lower() == "ocr_pixels"
        and int(evidence.get("raw_mask_pixels") or 0) > 0
        and (text.get("balloon_bbox") or text.get("bubble_mask_bbox"))
    )
    if not (
        flags
        & {
            "mask_outside_balloon",
            "mask_outside_balloon_critical",
            "bubble_clip_preserved_raw_text",
            "balloon_outline_components_removed",
        }
        or has_fast_ocr_evidence
    ):
        return False
    return not _text_region_looks_plain_white(image_rgb, text)


def _is_dark_or_colored_card_text(text: dict) -> bool:
    if not isinstance(text, dict) or not _route_action_allows_local_dark_panel_fill(text):
        return False
    profiles = {
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("render_profile") or "").strip().lower(),
        str(text.get("background_type") or "").strip().lower(),
    }
    if profiles & {"card", "title_card", "status_panel", "colored_status_panel", "dark_panel", "solid_dark", "dark"}:
        return True
    background = _rgb_luma_chroma(text.get("background_rgb"))
    if background is None:
        return False
    luma, chroma = background
    return bool(luma < 135.0 or chroma > 45.0)


def _strip_texts_need_raw_floor(texts: list[dict]) -> bool:
    for text in texts or []:
        if _is_dark_or_colored_card_text(text):
            return True
        flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()} if isinstance(text, dict) else set()
        if "bubble_clip_preserved_raw_text" in flags:
            return True
    return False


_REJECTED_BUBBLE_MASK_SOURCES = {
    "derived_white_crop_rejected",
    "rejected_derived_bubble_mask",
}

def _is_rejected_dark_card_fill_candidate(image_rgb: np.ndarray, text: dict) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or not isinstance(text, dict):
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source not in _REJECTED_BUBBLE_MASK_SOURCES:
        return False
    if not (text.get("balloon_bbox") or text.get("bubble_mask_bbox")):
        return False
    if not (text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("bbox")):
        return False
    if _text_has_dark_visual_context(text):
        return True
    height, width = image_rgb.shape[:2]
    bbox = (
        _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
        or _normalize_bbox(text.get("balloon_bbox") or text.get("bubble_mask_bbox"), width, height)
    )
    if bbox is None:
        return False
    fill_bbox = _expanded_bbox(width, height, bbox, padding=20) or bbox
    x1, y1, x2, y2 = fill_bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size < 64:
        return False
    pixels = crop.reshape(-1, crop.shape[-1]).astype(np.float32)
    luma = pixels[:, 0] * 0.299 + pixels[:, 1] * 0.587 + pixels[:, 2] * 0.114
    dark_ratio = float(np.count_nonzero(luma <= 96.0)) / float(max(1, luma.size))
    p25 = float(np.percentile(luma, 25))
    median = float(np.median(luma))
    p90 = float(np.percentile(luma, 90))
    return bool(dark_ratio >= 0.30 and p25 <= 88.0 and median <= 130.0 and p90 <= 252.0)


_UNSAFE_AUTO_INPAINT_QA_FLAGS = {
    "bbox_overreach_critical",
    "mask_outside_balloon_critical",
}


def _text_has_rejected_bubble_mask_source(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    error = str(text.get("bubble_mask_error") or text.get("bubbleMaskError") or "").strip()
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    return bool(
        source in _REJECTED_BUBBLE_MASK_SOURCES
        or bool(error)
        or "debug_derived_bubble_mask_rejected" in flags
        or "derived_bubble_mask_rejected" in flags
        or "rejected_derived_bubble_mask" in flags
    )


def _text_has_rejected_bubble_without_glyph_evidence(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    error = str(text.get("bubble_mask_error") or text.get("bubbleMaskError") or "").strip()
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    rejected = (
        source in _REJECTED_BUBBLE_MASK_SOURCES
        or bool(error)
        or "debug_derived_bubble_mask_rejected" in flags
        or "derived_bubble_mask_rejected" in flags
        or "rejected_derived_bubble_mask" in flags
    )
    if not rejected:
        return False
    has_line_geometry = bool(text.get("line_polygons") or text.get("text_pixel_bbox"))
    missing_glyph_evidence = (
        "raw_text_evidence_missing" in flags
        or "fast_fill_no_glyph_evidence" in flags
        or bool(text.get("raw_text_evidence_missing"))
        or not has_line_geometry
    )
    merged_fragment = "same_balloon_fragment_merged" in flags or bool(text.get("merged_into_trace_id"))
    return bool(missing_glyph_evidence and (merged_fragment or not has_line_geometry))


def _text_has_no_glyph_evidence(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    return bool("fast_fill_no_glyph_evidence" in flags or "raw_text_evidence_missing" in flags or text.get("raw_text_evidence_missing"))


def _has_fast_fillable_text_mask_evidence(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    evidence = text.get("mask_evidence")
    if not isinstance(evidence, dict) or evidence.get("fast_fill_allowed") is not True:
        return False
    kind = str(evidence.get("kind") or "").strip().lower()
    if kind not in {"ocr_pixels", "glyph_segmentation", "component_bubble_cleaner"}:
        return False
    try:
        raw_pixels = int(evidence.get("raw_mask_pixels") or 0)
        expanded_pixels = int(evidence.get("expanded_mask_pixels") or 0)
    except Exception:
        return False
    return bool(raw_pixels > 0 or expanded_pixels > 0)


def _dark_bubble_no_glyph_fast_fillable(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    return bool(
        str(text.get("bubble_mask_source") or "").strip().lower() == "image_dark_bubble_mask"
        and _text_has_no_glyph_evidence(text)
        and _has_fast_fillable_text_mask_evidence(text)
    )


def _has_trustworthy_glyph_action_mask(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    if not (text.get("line_polygons") or text.get("text_pixel_bbox")):
        return False
    return _has_fast_fillable_text_mask_evidence(text)


_COLORED_STATUS_PANEL_TERMS = {
    "APROVADO",
    "APROVADA",
    "BEM",
    "BOMBEIROS",
    "CANDIDATE",
    "CANDIDATO",
    "CANDIDATA",
    "BEAST",
    "COLLECTION",
    "CONFIRM",
    "CONSULTA",
    "ERROR",
    "FIREMAN",
    "GO",
    "INQUIRY",
    "LEVEL",
    "MISSION",
    "NAME",
    "NOME",
    "NEWS",
    "NUMBER",
    "NUMERO",
    "POWER",
    "PROMOTION",
    "PROCURAR",
    "RECORD",
    "RECORDS",
    "RECRUITMENT",
    "RECRUTAMENTO",
    "REQUIRED",
    "REGIONAL",
    "REGISTRO",
    "REGISTRATION",
    "RESIDENT",
    "RESIDENTE",
    "SEARCH",
    "SOURCE",
    "STATUS",
    "SUCCESSFUL",
    "SUCEDIDA",
    "SUCEDIDO",
    "SYSTEM",
    "TAMED",
    "TAMEABLE",
    "TESTE",
    "TRIAL",
}

_PHONE_UI_MONTH_TERMS = {
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
}


def _source_text_has_words(text: dict) -> bool:
    source = " ".join(str(text.get(key) or "") for key in ("text", "original", "translated", "traduzido"))
    return bool(re.search(r"[0-9A-Za-z]{2,}", source))


def _source_text_has_colored_status_terms(text: dict) -> bool:
    source_raw = " ".join(str(text.get(key) or "") for key in ("text", "original", "translated", "traduzido"))
    source = unicodedata.normalize("NFKD", source_raw).encode("ascii", "ignore").decode("ascii").upper()
    terms = set(re.findall(r"[A-Z]+", source))
    if terms & _COLORED_STATUS_PANEL_TERMS:
        return True
    if {"MISSED", "CALL"}.issubset(terms):
        return True
    if (terms & _PHONE_UI_MONTH_TERMS) and re.search(r"\d", source):
        return True
    return False


def _source_text_has_strict_ui_form_signal(text: dict) -> bool:
    source_raw = " ".join(
        str(text.get(key) or "")
        for key in ("text", "original", "translated", "traduzido", "raw_ocr", "normalized_ocr")
    )
    source = unicodedata.normalize("NFKD", source_raw).encode("ascii", "ignore").decode("ascii").upper()
    terms = set(re.findall(r"[A-Z]+", source))
    if re.search(r"\b20\s*\*+", source):
        return True
    if {"SUCCESSFUL", "CANDIDATE"}.issubset(terms) or {"CANDIDATE", "INQUIRY"}.issubset(terms):
        return True
    if {"CONSULTA", "CANDIDATO"}.issubset(terms) or {"CANDIDATO", "SUCEDIDA"}.issubset(terms):
        return True
    if {"FIREMAN", "RECRUITMENT"}.issubset(terms) or {"BOMBEIROS", "RECRUTAMENTO"}.issubset(terms):
        return True
    if "NAME" in terms and terms & {"RESIDENT", "REGISTRATION", "NUMBER"}:
        return True
    if "NOME" in terms and terms & {"RESIDENTE", "REGISTRO", "NUMERO"}:
        return True
    if terms & {"SEARCH", "PROCURAR"}:
        return True
    strong_terms = terms & {
        "CANDIDATE",
        "CANDIDATO",
        "FIREMAN",
        "BOMBEIROS",
        "INQUIRY",
        "CONSULTA",
        "NAME",
        "NOME",
        "RECRUITMENT",
        "RECRUTAMENTO",
        "REGIONAL",
        "REGISTRATION",
        "REGISTRO",
        "SUCCESSFUL",
        "SUCEDIDA",
        "TEST",
        "TESTE",
    }
    return len(strong_terms) >= 3


def _rgb_luma_chroma(rgb: object) -> tuple[float, int] | None:
    if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
        return None
    try:
        channels = [int(round(float(value))) for value in rgb[:3]]
    except Exception:
        return None
    luma = (channels[0] * 0.299) + (channels[1] * 0.587) + (channels[2] * 0.114)
    return float(luma), int(max(channels) - min(channels))


def _sample_ui_panel_background_from_mask(
    image_rgb: np.ndarray,
    panel_mask: np.ndarray,
    text_mask: np.ndarray,
) -> np.ndarray | None:
    sample_region = (panel_mask > 0) & (text_mask == 0)
    if int(np.count_nonzero(sample_region)) < 64:
        sample_region = panel_mask > 0
    if int(np.count_nonzero(sample_region)) < 64:
        return None
    pixels = image_rgb[sample_region].astype(np.float32)
    luma = (pixels[:, 0] * 0.299) + (pixels[:, 1] * 0.587) + (pixels[:, 2] * 0.114)
    chroma = np.max(pixels, axis=1) - np.min(pixels, axis=1)
    non_page = (luma <= 242.0) & ((chroma >= 5.0) | (luma <= 170.0))
    non_page_count = int(np.count_nonzero(non_page))
    if non_page_count < 64:
        return None
    non_page_ratio = non_page_count / float(max(1, int(pixels.shape[0])))
    if non_page_ratio < 0.16:
        return None
    fill = np.median(pixels[non_page], axis=0)
    fill_color = np.asarray([int(max(0, min(255, round(float(v))))) for v in fill], dtype=np.uint8)
    background = _rgb_luma_chroma(fill_color.tolist())
    if background is None:
        return None
    bg_luma, bg_chroma = background
    if bg_luma > 238.0:
        return None
    if bg_chroma < 5 and bg_luma > 170.0:
        return None
    text_region = text_mask > 0
    text_region_count = int(np.count_nonzero(text_region))
    if text_region_count > 0:
        text_pixels = image_rgb[text_region].astype(np.int16)
        delta_to_fill = np.mean(np.abs(text_pixels - fill_color.astype(np.int16)[None, :]), axis=1)
        in_text_panel_ratio = int(np.count_nonzero(delta_to_fill <= 22.0)) / float(text_region_count)
        if in_text_panel_ratio < 0.18:
            return None
    return fill_color


def _sample_colored_panel_background(
    image_rgb: np.ndarray,
    text: dict,
    text_mask: np.ndarray,
) -> np.ndarray | None:
    if not _source_text_has_words(text):
        return None
    height, width = image_rgb.shape[:2]
    text_bbox = _line_polygons_bbox(text, width, height) or _normalize_bbox(text.get("text_pixel_bbox"), width, height)
    if text_bbox is None:
        return None
    line_polygons = text.get("line_polygons") or []
    bubble_mask_source = str(text.get("bubble_mask_source") or "").strip().lower()
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()} if isinstance(text, dict) else set()
    visual_card_candidate = (
        bubble_mask_source in {"derived_card_panel_mask", "image_dark_panel_mask", "image_dark_bubble_mask"}
        or bubble_mask_source in _REJECTED_BUBBLE_MASK_SOURCES
        or (bubble_mask_source == "image_white_bubble_mask" and "mask_outside_balloon_critical" in flags)
        or _is_dark_or_colored_card_text(text)
    )
    if line_polygons and not _source_text_has_colored_status_terms(text) and not visual_card_candidate:
        return None
    fill_bbox = _expanded_bbox(width, height, text_bbox, padding=24) or text_bbox
    text_area = max(1, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
    fill_area = max(1, (fill_bbox[2] - fill_bbox[0]) * (fill_bbox[3] - fill_bbox[1]))
    if fill_area <= int(text_area * 1.12):
        max_side = max(text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        sampled_bbox = _expanded_bbox(width, height, text_bbox, padding=max(18, min(48, int(round(max_side * 0.12)))))
        if sampled_bbox is not None:
            fill_bbox = sampled_bbox
            fill_area = max(1, (fill_bbox[2] - fill_bbox[0]) * (fill_bbox[3] - fill_bbox[1]))
    if fill_area > int(width * height * 0.75):
        return None

    panel_mask = _mask_from_bbox(width, height, fill_bbox, padding=0) > 0
    ui_panel_fallback = (
        _sample_ui_panel_background_from_mask(image_rgb, panel_mask.astype(np.uint8), text_mask.astype(np.uint8))
        if _source_text_has_colored_status_terms(text)
        else None
    )
    inner = cv2.dilate(
        (text_mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ).astype(bool)
    outer = cv2.dilate(
        (text_mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33)),
        iterations=1,
    ).astype(bool)
    sample_region = panel_mask & outer & ~inner
    if int(np.count_nonzero(sample_region)) < 64:
        sample_region = panel_mask & ~inner
    if int(np.count_nonzero(sample_region)) < 64:
        return None

    sample = image_rgb[sample_region].astype(np.float32)
    fill = np.median(sample, axis=0)
    fill_color = np.asarray([int(max(0, min(255, round(float(v))))) for v in fill], dtype=np.uint8)
    background = _rgb_luma_chroma(fill_color.tolist())
    if background is None:
        return ui_panel_fallback
    bg_luma, bg_chroma = background
    if not (36.0 <= bg_luma <= 238.0 and bg_chroma >= 8):
        return ui_panel_fallback

    rgb_i = image_rgb.astype(np.int16)
    delta = np.mean(np.abs(rgb_i - fill_color.astype(np.int16)[None, None, :]), axis=2)
    rgb_f = image_rgb.astype(np.float32)
    luma = (rgb_f[:, :, 0] * 0.299) + (rgb_f[:, :, 1] * 0.587) + (rgb_f[:, :, 2] * 0.114)
    chroma = np.max(rgb_i, axis=2) - np.min(rgb_i, axis=2)
    source_glyph_like = (
        (text_mask > 0)
        & (delta >= 24.0)
        & (
            ((luma >= bg_luma + 18.0) & (luma >= 145.0))
            | ((chroma >= 38) & (delta >= 28.0))
        )
    )
    glyph_pixels = int(np.count_nonzero(source_glyph_like))
    mask_pixels = int(np.count_nonzero(text_mask > 0))
    if glyph_pixels < max(24, min(180, int(round(mask_pixels * 0.006)))):
        return ui_panel_fallback
    if glyph_pixels > int(mask_pixels * 0.72) and text.get("line_polygons"):
        return ui_panel_fallback
    if glyph_pixels > int(mask_pixels * 0.92):
        return ui_panel_fallback
    return fill_color


def _is_colored_status_panel_text_candidate(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    if not (text.get("line_polygons") or text.get("text_pixel_bbox")):
        return False
    background = _rgb_luma_chroma(text.get("background_rgb"))
    if background is None:
        return False
    luma, chroma = background
    if not (45.0 <= luma <= 215.0 and chroma >= 18):
        return False
    if not _source_text_has_colored_status_terms(text):
        return False
    line_count = len(text.get("line_polygons") or [])
    bbox = text.get("text_pixel_bbox") or text.get("bbox")
    bbox_area = 0
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        except Exception:
            bbox_area = 0
    if line_count >= 3 or bbox_area >= 16_000:
        return True
    return line_count >= 1 and bbox_area >= 1_000


def _colored_status_panel_glyph_mask(
    image_rgb: np.ndarray,
    text: dict,
    text_mask: np.ndarray,
    background_rgb: np.ndarray | None = None,
    *,
    dilation_size: int = 13,
) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        return None
    raw_background = background_rgb.tolist() if isinstance(background_rgb, np.ndarray) else text.get("background_rgb")
    background = _rgb_luma_chroma(raw_background)
    if background is None:
        return None
    bg_luma, _ = background
    try:
        bg = np.asarray([int(round(float(v))) for v in raw_background[:3]], dtype=np.int16)
    except Exception:
        return None
    rgb_i = image_rgb.astype(np.int16)
    delta = np.mean(np.abs(rgb_i - bg[None, None, :]), axis=2)
    rgb_f = image_rgb.astype(np.float32)
    luma = (rgb_f[:, :, 0] * 0.299) + (rgb_f[:, :, 1] * 0.587) + (rgb_f[:, :, 2] * 0.114)
    chroma = np.max(rgb_i, axis=2) - np.min(rgb_i, axis=2)
    derived_card_panel = str(text.get("bubble_mask_source") or "").strip().lower() in {"derived_card_panel_mask", "image_dark_panel_mask", "image_dark_bubble_mask"}
    rejected_visual_card_panel = str(text.get("bubble_mask_source") or "").strip().lower() in _REJECTED_BUBBLE_MASK_SOURCES
    if derived_card_panel:
        candidate = (text_mask > 0) & (delta >= 18.0) & (
            ((luma >= float(bg_luma) + 18.0) & (luma >= 145.0))
            | ((luma >= 218.0) & (delta >= 10.0))
            | ((luma >= 198.0) & (chroma <= 58) & (delta >= 14.0))
        )
    else:
        candidate = (text_mask > 0) & (delta >= 24.0) & (
            ((luma >= float(bg_luma) + 18.0) & (luma >= 145.0))
            | ((luma <= float(bg_luma) - 20.0) & (float(bg_luma) >= 88.0))
            | ((np.abs(luma - float(bg_luma)) >= 4.0) & (chroma >= 38) & (delta >= 28.0))
        )
    if int(np.count_nonzero(candidate)) < 24:
        return None
    if not text.get("line_polygons") and not derived_card_panel and not rejected_visual_card_panel:
        height, width = image_rgb.shape[:2]
        text_bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height) or _normalize_bbox(
            text.get("bbox"),
            width,
            height,
        )
        if text_bbox is not None:
            return _mask_from_bbox(width, height, text_bbox, padding=8)
    kernel_side = max(3, int(dilation_size))
    if kernel_side % 2 == 0:
        kernel_side += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_side, kernel_side))
    glyph_mask = cv2.dilate(candidate.astype(np.uint8) * 255, kernel, iterations=1)
    return glyph_mask if np.any(glyph_mask) else None


def _unsafe_card_visual_glyph_mask(
    image_rgb: np.ndarray,
    text_mask: np.ndarray,
    *,
    strict_text_rect: bool = False,
) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        return None
    region = text_mask > 0
    region_pixels = int(np.count_nonzero(region))
    if region_pixels < 16:
        return None
    rgb_f = image_rgb.astype(np.float32)
    luma = (rgb_f[:, :, 0] * 0.299) + (rgb_f[:, :, 1] * 0.587) + (rgb_f[:, :, 2] * 0.114)
    local_luma = luma[region]
    if local_luma.size < 16:
        return None
    p50 = float(np.percentile(local_luma, 50))
    p75 = float(np.percentile(local_luma, 75))
    p85 = float(np.percentile(local_luma, 85))
    local_chroma = None
    if image_rgb.ndim == 3:
        rgb_i = image_rgb.astype(np.int16)
        chroma = np.max(rgb_i, axis=2) - np.min(rgb_i, axis=2)
        local_chroma = chroma[region]
    dense_light_glyph = p75 >= 220.0 and p50 <= 180.0
    tight_glow_glyph = p85 >= 245.0 and p50 <= 105.0
    light_on_colored_card = bool(
        p75 >= 230.0
        and p50 >= 145.0
        and local_chroma is not None
        and local_chroma.size >= 16
        and float(np.percentile(local_chroma, 50)) >= 28.0
    )
    if light_on_colored_card:
        chroma_cutoff = max(55.0, min(105.0, float(np.percentile(local_chroma, 45))))
        threshold = max(212.0, min(242.0, p75 - 8.0))
        candidate = region & (luma >= threshold) & (chroma <= chroma_cutoff)
    else:
        threshold = max(128.0, p50 + 22.0) if dense_light_glyph else max(178.0, p75 + 18.0)
        candidate = region & (luma >= threshold)
    candidate_pixels = int(np.count_nonzero(candidate))
    if candidate_pixels < max(18, min(160, int(round(region_pixels * 0.003)))):
        return None
    max_candidate_ratio = 0.58 if strict_text_rect else (0.88 if (dense_light_glyph or light_on_colored_card) else 0.38)
    if candidate_pixels > int(region_pixels * max_candidate_ratio):
        return None
    kernel_side = 3 if strict_text_rect else 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_side, kernel_side))
    glyph_mask = cv2.dilate(candidate.astype(np.uint8) * 255, kernel, iterations=1)
    glyph_pixels = int(np.count_nonzero(glyph_mask))
    max_glyph_ratio = 0.62 if strict_text_rect else (0.96 if (dense_light_glyph or light_on_colored_card) else (0.72 if tight_glow_glyph else 0.55))
    if glyph_pixels > int(region_pixels * max_glyph_ratio):
        return None
    return glyph_mask if glyph_pixels else None


def _remember_text_fill_mask(text: dict, key: str, mask: np.ndarray | None) -> None:
    if not isinstance(text, dict) or not isinstance(mask, np.ndarray) or not np.any(mask):
        return
    text[key] = np.where(mask > 0, 255, 0).astype(np.uint8)


def _pop_text_fill_mask(text: dict, key: str, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(text, dict):
        return np.zeros(shape, dtype=np.uint8)
    mask = text.pop(key, None)
    if not isinstance(mask, np.ndarray):
        return np.zeros(shape, dtype=np.uint8)
    return _coerce_mask_for_shape(mask, shape)


def _accumulate_page_fill_mask(ocr_page: dict, key: str, mask: np.ndarray | None, shape: tuple[int, int]) -> None:
    if not isinstance(ocr_page, dict) or not isinstance(mask, np.ndarray) or not np.any(mask):
        return
    current = _coerce_mask_for_shape(ocr_page.get(key), shape)
    ocr_page[key] = np.maximum(current, _coerce_mask_for_shape(mask, shape)).astype(np.uint8)


def _page_fill_mask(ocr_page: dict | None, key: str, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(ocr_page, dict):
        return np.zeros(shape, dtype=np.uint8)
    return _coerce_mask_for_shape(ocr_page.get(key), shape)


def _clip_dark_bubble_fill_mask_to_lobe(
    text: dict,
    mask: np.ndarray,
    shape: tuple[int, int],
    *,
    padding: int = 4,
) -> np.ndarray:
    if not isinstance(text, dict) or not isinstance(mask, np.ndarray) or not np.any(mask):
        return mask
    if str(text.get("bubble_mask_source") or "").strip().lower() != "image_dark_bubble_mask":
        return mask
    height, width = shape[:2]
    sibling_clip = _dark_bubble_sibling_lobe_clip_mask(text, width, height, padding=max(0, int(padding)))
    if isinstance(sibling_clip, np.ndarray) and np.any(sibling_clip):
        sibling_clipped = np.where((mask > 0) & (sibling_clip > 0), 255, 0).astype(np.uint8)
        sibling_pixels = int(np.count_nonzero(sibling_clipped))
        original_pixels_for_sibling = int(np.count_nonzero(mask))
        if sibling_pixels >= max(48, int(round(original_pixels_for_sibling * 0.18))):
            if sibling_pixels < original_pixels_for_sibling:
                metrics = text.setdefault("qa_metrics", {})
                if isinstance(metrics, dict):
                    metrics["dark_bubble_sibling_lobe_fill_mask_clipped"] = {
                        "source_pixels": int(original_pixels_for_sibling),
                        "clipped_pixels": int(sibling_pixels),
                    }
            mask = sibling_clipped
    lobe_bbox = _normalize_bbox(text.get("bubble_mask_bbox") or text.get("bubbleMaskBbox"), width, height)
    if lobe_bbox is None:
        return mask
    text_bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height) or _normalize_bbox(text.get("bbox"), width, height)
    if text_bbox is not None:
        tx1, ty1, tx2, ty2 = text_bbox
        lx1, ly1, lx2, ly2 = lobe_bbox
        text_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        inter_w = max(0, min(tx2, lx2) - max(tx1, lx1))
        inter_h = max(0, min(ty2, ly2) - max(ty1, ly1))
        lobe_text_overlap = inter_w * inter_h
        lobe_text_width = max(0, min(tx2, lx2) - max(tx1, lx1))
        text_w = max(1, tx2 - tx1)
        if lobe_text_overlap < int(text_area * 0.42) or lobe_text_width < int(text_w * 0.50):
            metrics = text.setdefault("qa_metrics", {})
            if isinstance(metrics, dict):
                metrics["dark_bubble_lobe_clip_rejected_undercovered_text"] = {
                    "text_bbox": list(text_bbox),
                    "lobe_bbox": list(lobe_bbox),
                    "overlap_pixels": int(lobe_text_overlap),
                    "text_area": int(text_area),
                }
            _append_qa_flag_to_item(text, "dark_bubble_lobe_clip_rejected_undercovered_text")
            return mask
    balloon_bbox = _normalize_bbox(text.get("balloon_bbox"), width, height)
    connected_hint = bool(
        "dark_bubble_connected_lobes_promoted" in {str(flag).strip() for flag in text.get("qa_flags") or []}
        or "dark_bubble_lobe_mask_bbox_preferred" in {str(flag).strip() for flag in text.get("qa_flags") or []}
    )
    if balloon_bbox is not None:
        lobe_area = (lobe_bbox[2] - lobe_bbox[0]) * (lobe_bbox[3] - lobe_bbox[1])
        balloon_area = (balloon_bbox[2] - balloon_bbox[0]) * (balloon_bbox[3] - balloon_bbox[1])
        connected_hint = connected_hint or lobe_area <= int(balloon_area * 0.82)
    if not connected_hint:
        return mask

    clip_bbox = _expanded_bbox(width, height, lobe_bbox, padding=max(0, int(padding))) or lobe_bbox
    x1, y1, x2, y2 = clip_bbox
    clip = np.zeros((height, width), dtype=np.uint8)
    clip[y1:y2, x1:x2] = 255
    clipped = np.where((mask > 0) & (clip > 0), 255, 0).astype(np.uint8)
    clipped_pixels = int(np.count_nonzero(clipped))
    original_pixels = int(np.count_nonzero(mask))
    if clipped_pixels < max(48, int(round(original_pixels * 0.20))):
        return mask
    if clipped_pixels < original_pixels:
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["dark_bubble_lobe_fill_mask_clipped"] = {
                "source_pixels": int(original_pixels),
                "clipped_pixels": int(clipped_pixels),
                "clip_bbox": list(clip_bbox),
            }
    return clipped


def _dark_bubble_sibling_lobe_clip_mask(text: dict, width: int, height: int, *, padding: int = 4) -> np.ndarray | None:
    siblings = text.get("_dark_bubble_sibling_texts")
    if not isinstance(siblings, list) or not siblings:
        return None
    own_flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    has_explicit_connected_lobe = bool(
        "dark_bubble_connected_lobes_promoted" in own_flags
        or "dark_bubble_lobe_mask_bbox_preferred" in own_flags
        or any(
            isinstance(sibling, dict)
            and {
                str(flag).strip()
                for flag in sibling.get("qa_flags") or []
                if str(flag).strip()
            }
            & {"dark_bubble_connected_lobes_promoted", "dark_bubble_lobe_mask_bbox_preferred"}
            for sibling in siblings
        )
    )
    if not has_explicit_connected_lobe:
        return None
    text_bbox = (
        _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
    )
    base_bbox = (
        _normalize_bbox(text.get("bubble_mask_bbox"), width, height)
        or _normalize_bbox(text.get("balloon_bbox"), width, height)
        or text_bbox
    )
    if text_bbox is None or base_bbox is None:
        return None
    tx1, ty1, tx2, ty2 = text_bbox
    tcx = (tx1 + tx2) / 2.0
    tcy = (ty1 + ty2) / 2.0
    clip = np.zeros((height, width), dtype=np.uint8)
    bx1, by1, bx2, by2 = base_bbox
    clip[by1:by2, bx1:bx2] = 255
    changed = False
    own_id = str(text.get("id") or text.get("text_id") or text.get("trace_id") or "").strip()
    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue
        if str(sibling.get("bubble_mask_source") or "").strip().lower() != "image_dark_bubble_mask":
            continue
        sibling_id = str(sibling.get("id") or sibling.get("text_id") or sibling.get("trace_id") or "").strip()
        if own_id and sibling_id and sibling_id == own_id:
            continue
        sibling_flags = {str(flag).strip() for flag in sibling.get("qa_flags") or [] if str(flag).strip()}
        sibling_lobe_bbox = _normalize_bbox(sibling.get("bubble_mask_bbox"), width, height)
        sibling_text_bbox = (
            _normalize_bbox(sibling.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(sibling.get("bbox"), width, height)
        )
        sibling_connected_lobe = bool(
            "dark_bubble_connected_lobes_promoted" in sibling_flags
            or "dark_bubble_lobe_mask_bbox_preferred" in sibling_flags
        )
        sibling_bbox = sibling_lobe_bbox if sibling_connected_lobe and sibling_lobe_bbox is not None else sibling_text_bbox
        if sibling_bbox is None:
            continue
        sx1, sy1, sx2, sy2 = sibling_bbox
        scx = (sx1 + sx2) / 2.0
        scy = (sy1 + sy2) / 2.0
        overlap_x = max(0, min(tx2, sx2) - max(tx1, sx1))
        overlap_y = max(0, min(ty2, sy2) - max(ty1, sy1))
        same_connected_area = bool(
            _bbox_overlap_ratio(base_bbox, _normalize_bbox(sibling.get("balloon_bbox"), width, height) or sibling_bbox) >= 0.18
            or _bbox_overlap_ratio(base_bbox, _normalize_bbox(sibling.get("bubble_mask_bbox"), width, height) or sibling_bbox) >= 0.18
            or overlap_x > 0
            or overlap_y > 0
        )
        if not same_connected_area:
            continue
        if abs(tcx - scx) >= max(24.0, abs(tcy - scy) * 0.72):
            mid = int(round((tcx + scx) / 2.0))
            if tcx < scx:
                boundary = min(mid, sx1 - padding)
                clip[:, max(0, boundary):] = 0
            else:
                boundary = max(mid, sx2 + padding)
                clip[:, :min(width, boundary)] = 0
            changed = True
        elif abs(tcy - scy) >= 24.0:
            mid = int(round((tcy + scy) / 2.0))
            if tcy < scy:
                clip[max(0, mid + padding):, :] = 0
            else:
                clip[:min(height, mid - padding), :] = 0
            changed = True
    if not changed or not np.any(clip):
        return None
    return clip


def _filter_dark_bubble_line_polygons_against_siblings(text: dict, width: int, height: int) -> dict:
    if not isinstance(text, dict):
        return text
    if str(text.get("bubble_mask_source") or "").strip().lower() != "image_dark_bubble_mask":
        return text
    polygons = text.get("line_polygons")
    siblings = text.get("_dark_bubble_sibling_texts")
    if not isinstance(polygons, list) or len(polygons) <= 1 or not isinstance(siblings, list):
        return text
    own_id = str(text.get("id") or text.get("text_id") or text.get("trace_id") or "").strip()
    sibling_boxes: list[list[int]] = []
    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue
        if str(sibling.get("bubble_mask_source") or "").strip().lower() != "image_dark_bubble_mask":
            continue
        sibling_id = str(sibling.get("id") or sibling.get("text_id") or sibling.get("trace_id") or "").strip()
        if own_id and sibling_id and sibling_id == own_id:
            continue
        sibling_bbox = (
            _normalize_bbox(sibling.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(sibling.get("bbox"), width, height)
        )
        if sibling_bbox is not None:
            sibling_boxes.append(sibling_bbox)
    if not sibling_boxes:
        return text

    kept = []
    removed = []
    for index, polygon in enumerate(polygons):
        poly_bbox = _polygon_bbox(polygon)
        if poly_bbox is None:
            kept.append(polygon)
            continue
        px1, py1, px2, py2 = poly_bbox
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        belongs_to_sibling = False
        for sx1, sy1, sx2, sy2 in sibling_boxes:
            if sx1 <= pcx <= sx2 and sy1 <= pcy <= sy2:
                belongs_to_sibling = True
                break
        if belongs_to_sibling:
            removed.append({"index": int(index), "bbox": list(poly_bbox)})
        else:
            kept.append(polygon)
    if not removed or not kept:
        return text
    updated = dict(text)
    updated["line_polygons"] = kept
    metrics = updated.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["dark_bubble_sibling_line_polygons_removed"] = {
            "removed": removed,
            "kept": int(len(kept)),
        }
    flags = updated.setdefault("qa_flags", [])
    if isinstance(flags, list) and "dark_bubble_sibling_line_polygons_removed" not in flags:
        flags.append("dark_bubble_sibling_line_polygons_removed")
    return updated


def _dark_bubble_light_text_visual_mask(
    image_rgb: np.ndarray,
    text: dict,
    *,
    padding: int = 22,
) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or not isinstance(text, dict):
        return None
    if str(text.get("bubble_mask_source") or "").strip().lower() != "image_dark_bubble_mask":
        return None
    height, width = image_rgb.shape[:2]
    text_bbox = _trusted_text_bbox_for_contract(text, width, height) or _normalize_bbox(
        text.get("text_pixel_bbox"), width, height
    ) or _normalize_bbox(text.get("bbox"), width, height)
    clip_bbox = _normalize_bbox(text.get("balloon_bbox") or text.get("bubble_mask_bbox"), width, height)
    if clip_bbox is None:
        clip_bbox = text_bbox
    if clip_bbox is None:
        return None
    if text_bbox is not None:
        clip_bbox = [
            min(clip_bbox[0], text_bbox[0]),
            min(clip_bbox[1], text_bbox[1]),
            max(clip_bbox[2], text_bbox[2]),
            max(clip_bbox[3], text_bbox[3]),
        ]
    clip_bbox = _expanded_bbox(width, height, clip_bbox, padding=max(0, int(padding))) or clip_bbox
    x1, y1, x2, y2 = clip_bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_f = crop.astype(np.float32)
    luma = crop_f[:, :, 0] * 0.299 + crop_f[:, :, 1] * 0.587 + crop_f[:, :, 2] * 0.114
    bg_luma = float(np.percentile(luma, 20)) if luma.size else 0.0
    threshold = max(174.0, bg_luma + 108.0)
    bright = np.where(luma >= threshold, 255, 0).astype(np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(bright, 8)
    local = np.zeros_like(bright)
    text_area = 0
    if text_bbox is not None:
        text_area = max(1, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
    max_area = max(2400, int(round(text_area * 0.90))) if text_area else 2400
    for label in range(1, component_count):
        lx, ly, lw, lh, area = [int(v) for v in stats[label]]
        if area < 10 or area > max_area:
            continue
        component_bbox = [x1 + lx, y1 + ly, x1 + lx + lw, y1 + ly + lh]
        if text_bbox is not None:
            overlap = _bbox_overlap_ratio(component_bbox, text_bbox)
            near_text = bool(
                component_bbox[2] >= text_bbox[0] - 24
                and component_bbox[0] <= text_bbox[2] + 24
                and component_bbox[3] >= text_bbox[1] - 56
                and component_bbox[1] <= text_bbox[3] + 56
            )
            if overlap <= 0.01 and not near_text:
                continue
        if lw > max(320, (x2 - x1) * 0.92) or lh > max(110, (y2 - y1) * 0.80):
            continue
        local[labels == label] = 255
    if not np.any(local):
        return None
    local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = local
    mask = _clip_dark_bubble_fill_mask_to_lobe(text, mask, image_rgb.shape[:2])
    if int(np.count_nonzero(mask)) < 48:
        return None
    return mask


def _dark_connected_compact_contract_fill_mask(
    image_rgb: np.ndarray,
    text: dict,
) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or not isinstance(text, dict):
        return None
    if str(text.get("bubble_mask_source") or "").strip().lower() != "image_dark_bubble_mask":
        return None
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if "dark_connected_bubble_compact_bbox_replaced_aggregate_source" not in flags:
        return None
    metrics = text.get("qa_metrics") if isinstance(text.get("qa_metrics"), dict) else {}
    replacement = metrics.get("inpaint_mask_contract_text_bbox_replaced_aggregate_source")
    if not isinstance(replacement, dict):
        return None
    height, width = image_rgb.shape[:2]
    bbox = _normalize_bbox(replacement.get("bbox"), width, height)
    if bbox is None:
        return None
    mask = _mask_from_bbox(width, height, bbox, padding=7)
    source_pixels = int(np.count_nonzero(mask))
    clipped = _clip_dark_bubble_fill_mask_to_lobe(text, mask, image_rgb.shape[:2], padding=7)
    clipped_pixels = int(np.count_nonzero(clipped)) if isinstance(clipped, np.ndarray) else 0
    if clipped_pixels >= max(64, int(round(source_pixels * 0.65))):
        mask = clipped
    else:
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["dark_connected_compact_bbox_lobe_clip_rejected"] = {
                "source_pixels": int(source_pixels),
                "clipped_pixels": int(clipped_pixels),
            }
    if int(np.count_nonzero(mask)) < 64:
        return None
    return mask.astype(np.uint8)


def _dark_bubble_visual_fill_override(image_rgb: np.ndarray, text: dict, current_mask: np.ndarray) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or not isinstance(text, dict):
        return None
    if str(text.get("bubble_mask_source") or "").strip().lower() != "image_dark_bubble_mask":
        return None
    if not isinstance(current_mask, np.ndarray) or not np.any(current_mask):
        return None
    candidate = dict(text)
    visual_mask = build_inpaint_mask(candidate, image_rgb.shape, image_rgb=image_rgb)
    used_direct_visual_mask = False
    compact_contract_mask = _dark_connected_compact_contract_fill_mask(image_rgb, text)
    if isinstance(visual_mask, np.ndarray) and np.any(visual_mask):
        visual_mask = _coerce_mask_for_shape(visual_mask, image_rgb.shape[:2])
        visual_mask = _clip_dark_bubble_fill_mask_to_lobe(text, visual_mask, image_rgb.shape[:2])
    else:
        visual_mask = _dark_bubble_light_text_visual_mask(image_rgb, text)
        used_direct_visual_mask = isinstance(visual_mask, np.ndarray) and np.any(visual_mask)
    if not isinstance(visual_mask, np.ndarray) or not np.any(visual_mask):
        if isinstance(compact_contract_mask, np.ndarray) and np.any(compact_contract_mask):
            text_metrics = text.setdefault("qa_metrics", {})
            if isinstance(text_metrics, dict):
                text_metrics["dark_bubble_visual_fill_override"] = {
                    "visual_pixels": int(np.count_nonzero(compact_contract_mask)),
                    "current_pixels": int(np.count_nonzero(current_mask)),
                    "reason": "dark_connected_compact_bbox_contract",
                    "source": "compact_text_bbox_contract",
                }
            text["_force_solid_dark_text_fill"] = True
            return compact_contract_mask.astype(np.uint8)
        return None
    visual_pixels = int(np.count_nonzero(visual_mask))
    current_pixels = int(np.count_nonzero(current_mask))
    no_glyph_evidence = _text_has_no_glyph_evidence(text)
    overbroad_current_mask = bool(current_pixels >= max(192, int(round(visual_pixels * 1.35))))
    if visual_pixels < 96 or visual_pixels >= int(current_pixels * 0.88):
        return None
    metrics = candidate.get("qa_metrics") if isinstance(candidate.get("qa_metrics"), dict) else {}
    flags = {str(flag).strip() for flag in candidate.get("qa_flags") or [] if str(flag).strip()}
    visual_evidence = bool(
        isinstance(metrics.get("dark_bubble_visual_glyph_mask"), dict)
        or isinstance(metrics.get("dark_bubble_visual_glyph_mask_replaced_geometry"), dict)
        or "bubble_clip_preserved_raw_text" in flags
        or used_direct_visual_mask
    )
    if not visual_evidence or (not no_glyph_evidence and not overbroad_current_mask):
        return None
    if isinstance(compact_contract_mask, np.ndarray) and np.any(compact_contract_mask):
        compact_pixels = int(np.count_nonzero(compact_contract_mask))
        if (
            compact_pixels >= max(96, int(round(visual_pixels * 1.20)))
            and compact_pixels <= max(visual_pixels + 32000, int(round(current_pixels * 0.75)))
        ):
            text_metrics = text.setdefault("qa_metrics", {})
            if isinstance(text_metrics, dict):
                text_metrics["dark_bubble_visual_fill_override"] = {
                    "visual_pixels": int(compact_pixels),
                    "current_pixels": int(current_pixels),
                    "reason": "dark_connected_compact_bbox_contract",
                    "source": "compact_text_bbox_contract",
                    "raw_visual_pixels": int(visual_pixels),
                }
            text["_force_solid_dark_text_fill"] = True
            return compact_contract_mask.astype(np.uint8)
    text_metrics = text.setdefault("qa_metrics", {})
    if isinstance(text_metrics, dict):
        text_metrics["dark_bubble_visual_fill_override"] = {
            "visual_pixels": int(visual_pixels),
            "current_pixels": int(current_pixels),
            "reason": "no_glyph_evidence" if no_glyph_evidence else "overbroad_dark_fill_mask",
            "source": "direct_light_text" if used_direct_visual_mask else "build_inpaint_mask",
        }
    visual_mask = _expand_dark_bubble_visual_text_fill_mask(
        image_rgb,
        text,
        visual_mask.astype(np.uint8),
        source_pixels=visual_pixels,
    )
    text["_force_solid_dark_text_fill"] = True
    return visual_mask.astype(np.uint8)


def _expand_dark_bubble_visual_text_fill_mask(
    image_rgb: np.ndarray,
    text: dict,
    visual_mask: np.ndarray,
    *,
    source_pixels: int,
) -> np.ndarray:
    if not isinstance(image_rgb, np.ndarray) or not isinstance(text, dict):
        return visual_mask
    if str(text.get("bubble_mask_source") or "").strip().lower() != "image_dark_bubble_mask":
        return visual_mask
    if not isinstance(visual_mask, np.ndarray) or not np.any(visual_mask):
        return visual_mask
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    metrics = text.get("qa_metrics") if isinstance(text.get("qa_metrics"), dict) else {}
    has_visual_glyph = bool(
        isinstance(metrics.get("dark_bubble_visual_glyph_mask"), dict)
        or isinstance(metrics.get("dark_bubble_visual_glyph_mask_replaced_geometry"), dict)
        or "dark_bubble_visual_glyph_mask_replaced_geometry" in flags
        or "fast_fill_no_glyph_evidence" in flags
    )
    if not has_visual_glyph:
        return visual_mask
    glow_px = 0
    estilo = text.get("estilo") if isinstance(text.get("estilo"), dict) else {}
    try:
        glow_px = int(estilo.get("glow_px", 0) or 0) if bool(estilo.get("glow")) else 0
    except Exception:
        glow_px = 0
    no_glyph_evidence = bool("fast_fill_no_glyph_evidence" in flags or _text_has_no_glyph_evidence(text))
    pad = max(3, min(10 if no_glyph_evidence else 8, glow_px + (7 if no_glyph_evidence else 4)))
    kernel_size = max(3, pad * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(visual_mask.astype(np.uint8), kernel, iterations=1)
    expanded = _clip_dark_bubble_fill_mask_to_lobe(text, expanded, image_rgb.shape[:2], padding=max(2, pad))
    expanded_pixels = int(np.count_nonzero(expanded))
    original_pixels = max(1, int(source_pixels))
    if expanded_pixels < max(48, int(round(original_pixels * 1.05))):
        return visual_mask
    if expanded_pixels > int(original_pixels * (3.4 if no_glyph_evidence else 2.8)):
        # This is a text-halo expansion, not permission to flood-fill the lobe.
        limit_kernel_size = 13 if no_glyph_evidence else 9
        limit_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (limit_kernel_size, limit_kernel_size))
        expanded = cv2.dilate(visual_mask.astype(np.uint8), limit_kernel, iterations=1)
        expanded = _clip_dark_bubble_fill_mask_to_lobe(text, expanded, image_rgb.shape[:2], padding=4)
        expanded_pixels = int(np.count_nonzero(expanded))
    text_metrics = text.setdefault("qa_metrics", {})
    if isinstance(text_metrics, dict):
        text_metrics["dark_bubble_visual_fill_expanded"] = {
            "source_pixels": int(source_pixels),
            "expanded_pixels": int(expanded_pixels),
            "pad_px": int(pad),
        }
    return expanded.astype(np.uint8)


def _authorized_dark_panel_padding(
    text: dict,
    *,
    has_line_geometry: bool,
    detected_dark_panel: bool,
    colored_status_panel: bool,
    base_padding: int,
) -> int:
    if not has_line_geometry:
        return base_padding if detected_dark_panel else 1
    if not isinstance(text, dict):
        return 1
    profiles = {
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("background_type") or "").strip().lower(),
    }
    background = _rgb_luma_chroma(text.get("background_rgb"))
    panel_hint = bool(
        detected_dark_panel
        or colored_status_panel
        or (background is not None and background[0] <= 130.0)
        or (background is not None and background[1] > 45.0)
        or (profiles & {"ui_form", "dark_panel", "solid_dark", "dark"})
    )
    if not panel_hint:
        return 1
    padding = max(5, min(8, int(round(max(1, base_padding) * 0.75))))
    return padding


def _sample_dark_bubble_inner_fill_color(
    image_rgb: np.ndarray,
    text: dict,
    text_mask: np.ndarray,
) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or not isinstance(text_mask, np.ndarray):
        return None
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0 or text_mask.shape[:2] != (height, width):
        return None
    local_bbox = (
        _line_polygons_bbox(text, width, height)
        or _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
    )
    if local_bbox is not None:
        lx1, ly1, lx2, ly2 = local_bbox
        local_side = max(lx2 - lx1, ly2 - ly1)
        local_sample_bbox = _expanded_bbox(
            width,
            height,
            local_bbox,
            padding=max(18, min(72, int(round(local_side * 0.18)))),
        )
        if local_sample_bbox is not None:
            sx1, sy1, sx2, sy2 = local_sample_bbox
            local_mask = np.zeros((height, width), dtype=np.uint8)
            local_mask[sy1:sy2, sx1:sx2] = 255
            exclusion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
            text_exclusion = cv2.dilate(text_mask.astype(np.uint8), exclusion_kernel, iterations=1)
            local_region = (local_mask > 0) & (text_exclusion == 0)
            if int(np.count_nonzero(local_region)) >= 80:
                local_pixels = image_rgb[local_region].astype(np.float32)
                local_luma = (
                    local_pixels[:, 0] * 0.299
                    + local_pixels[:, 1] * 0.587
                    + local_pixels[:, 2] * 0.114
                )
                local_chroma = np.max(local_pixels, axis=1) - np.min(local_pixels, axis=1)
                dark_cutoff = min(34.0, max(10.0, float(np.percentile(local_luma, 25)) + 5.0))
                dark_neutral = (local_luma <= dark_cutoff) & (local_chroma <= 32.0)
                if int(np.count_nonzero(dark_neutral)) < 48:
                    dark_neutral = (local_luma <= min(44.0, dark_cutoff + 10.0)) & (local_chroma <= 36.0)
                if int(np.count_nonzero(dark_neutral)) >= 48:
                    fill = np.median(local_pixels[dark_neutral], axis=0)
                    fill_color = np.asarray(
                        [int(max(0, min(255, round(float(v))))) for v in fill],
                        dtype=np.uint8,
                    )
                    fill_luma = float(
                        (fill_color[0] * 0.299)
                        + (fill_color[1] * 0.587)
                        + (fill_color[2] * 0.114)
                    )
                    if fill_luma <= 46.0:
                        return fill_color
    bbox = _normalize_bbox(text.get("balloon_bbox"), width, height)
    if bbox is None:
        candidate = _normalize_bbox(text.get("bubble_mask_bbox"), width, height)
        if candidate is not None:
            x1, y1, x2, y2 = candidate
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(max(1, width * height))
            if area_ratio <= 0.58:
                bbox = candidate
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if bw < 24 or bh < 24:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    inset_x = max(3, int(round(bw * 0.045)))
    inset_y = max(3, int(round(bh * 0.045)))
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, bw // 2 - inset_x), max(1, bh // 2 - inset_y))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    exclusion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    text_exclusion = cv2.dilate(text_mask.astype(np.uint8), exclusion_kernel, iterations=1)
    region = (mask > 0) & (text_exclusion == 0)
    if int(np.count_nonzero(region)) < 80:
        region = mask > 0
    if int(np.count_nonzero(region)) < 80:
        return None

    pixels = image_rgb[region].astype(np.float32)
    luma = (pixels[:, 0] * 0.299) + (pixels[:, 1] * 0.587) + (pixels[:, 2] * 0.114)
    cutoff = max(24.0, min(72.0, float(np.percentile(luma, 35)) + 6.0))
    dark_pixels = pixels[luma <= cutoff]
    if dark_pixels.shape[0] < 48:
        dark_pixels = pixels[luma <= max(82.0, cutoff + 18.0)]
    if dark_pixels.shape[0] < 48:
        return None
    fill = np.median(dark_pixels, axis=0)
    fill_color = np.asarray([int(max(0, min(255, round(float(v))))) for v in fill], dtype=np.uint8)
    fill_luma = float((fill_color[0] * 0.299) + (fill_color[1] * 0.587) + (fill_color[2] * 0.114))
    if fill_luma > 88.0:
        return None
    return fill_color


def _try_dark_panel_text_fill(image_rgb: np.ndarray, text: dict) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray):
        return None
    qa_flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()} if isinstance(text, dict) else set()
    if "short_dark_text_full_panel_bbox_rejected" in qa_flags:
        return None
    if _rejected_visual_card_requires_real_inpaint(image_rgb, text) and not _rejected_visual_card_allows_local_text_fill(text):
        return None
    false_white_card_candidate = _is_false_white_card_candidate(image_rgb, text)
    if _unsafe_white_balloon_requires_real_inpaint(text) and not false_white_card_candidate:
        return None
    white_balloon_context = _text_is_white_balloon_context(text)
    unsafe_mask_fallback = bool(
        qa_flags
        & {
            "mask_outside_balloon_critical",
            "missing_real_bubble_mask",
            "rejected_derived_bubble_mask",
            "debug_derived_bubble_mask_rejected",
        }
    )
    if (
        white_balloon_context
        and not false_white_card_candidate
        and (not unsafe_mask_fallback or _text_region_looks_plain_white(image_rgb, text))
    ):
        return None
    bubble_mask_source = str(text.get("bubble_mask_source") or "").strip().lower()
    if _image_dark_bubble_is_visually_light(image_rgb, text):
        return None
    translator_note_text_mask = bubble_mask_source == "translator_note_text_mask"
    derived_card_panel = bubble_mask_source in {"derived_card_panel_mask", "image_dark_panel_mask", "image_dark_bubble_mask"}
    rejected_visual_card_panel = bubble_mask_source in _REJECTED_BUBBLE_MASK_SOURCES and bool(
        text.get("bubble_mask_bbox") or text.get("balloon_bbox")
    )
    colored_status_panel = _is_colored_status_panel_text_candidate(text)
    ui_panel_text = _source_text_has_colored_status_terms(text)
    colored_card_text = (
        _is_dark_or_colored_card_text(text)
        or derived_card_panel
        or rejected_visual_card_panel
        or false_white_card_candidate
    )
    metadata_candidate = _is_dark_panel_text_candidate(text) or colored_status_panel or colored_card_text or translator_note_text_mask
    unsafe_card_glyph_only = bool(unsafe_mask_fallback and colored_card_text and not (colored_status_panel or ui_panel_text))
    if not metadata_candidate:
        if not isinstance(text, dict) or not _route_action_allows_local_dark_panel_fill(text):
            return None
    if not metadata_candidate and not text.get("text_pixel_bbox") and not text.get("line_polygons"):
        return None
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return None
    if bubble_mask_source == "image_dark_bubble_mask":
        sibling_filtered_text = _filter_dark_bubble_line_polygons_against_siblings(text, width, height)
        if isinstance(sibling_filtered_text, dict) and sibling_filtered_text is not text:
            text.update(sibling_filtered_text)
        filtered_text = _filter_dark_bubble_connected_lobe_line_polygons(text, width, height)
        if isinstance(filtered_text, dict) and filtered_text is not text:
            text.update(filtered_text)
    text_mask = _text_geometry_mask(width, height, text)
    if text_mask is None or not np.any(text_mask):
        return None
    mask_area = int(np.count_nonzero(text_mask))
    sampled_colored_background = None
    if not colored_status_panel:
        sampled_colored_background = _sample_colored_panel_background(image_rgb, text, text_mask)
        if sampled_colored_background is not None:
            colored_status_panel = True
            metadata_candidate = True
    if mask_area > int(width * height * 0.78):
        return None

    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
    inner = cv2.dilate(text_mask, inner_kernel, iterations=1)
    outer = cv2.dilate(text_mask, outer_kernel, iterations=1)
    sample_region = (outer > 0) & (inner == 0)
    if int(np.count_nonzero(sample_region)) < 48:
        return None

    sample = image_rgb[sample_region].astype(np.float32)
    sample_luma = np.mean(sample, axis=1)
    median_luma = float(np.median(sample_luma))
    p25_luma = float(np.percentile(sample_luma, 25))
    p80_luma = float(np.percentile(sample_luma, 80))
    p90_luma = float(np.percentile(sample_luma, 90))
    sample_fill_color = np.asarray(
        [int(max(0, min(255, round(float(v))))) for v in np.median(sample, axis=0)],
        dtype=np.uint8,
    )
    metadata_fill_color = None
    background_hint = None
    raw_background_rgb = text.get("background_rgb")
    if isinstance(raw_background_rgb, (list, tuple)) and len(raw_background_rgb) >= 3:
        try:
            background_hint = _rgb_luma_chroma(raw_background_rgb)
            if (
                translator_note_text_mask
                or colored_status_panel
                or colored_card_text
                or (ui_panel_text and background_hint is not None and background_hint[0] <= 130.0)
            ):
                metadata_fill_color = np.asarray(
                    [int(max(0, min(255, round(float(v))))) for v in raw_background_rgb[:3]],
                    dtype=np.uint8,
                )
        except Exception:
            metadata_fill_color = None
    dark_bubble_inner_fill_color = (
        _sample_dark_bubble_inner_fill_color(image_rgb, text, text_mask)
        if bubble_mask_source == "image_dark_bubble_mask"
        else None
    )
    if bubble_mask_source == "image_dark_bubble_mask" and dark_bubble_inner_fill_color is None:
        return None
    rgb_i = image_rgb.astype(np.int16)
    rgb_f = image_rgb.astype(np.float32)
    luma = (rgb_f[:, :, 0] * 0.299) + (rgb_f[:, :, 1] * 0.587) + (rgb_f[:, :, 2] * 0.114)
    chroma = np.max(rgb_i, axis=2) - np.min(rgb_i, axis=2)
    delta = np.mean(np.abs(rgb_i - sample_fill_color.astype(np.int16)[None, None, :]), axis=2)
    glyph_region = text_mask > 0
    dark_panel_glyph_like = glyph_region & (
        ((luma >= median_luma + 20.0) & (luma >= 82.0))
        | ((chroma >= 24) & (delta >= 18.0) & (luma >= median_luma + 8.0))
    )
    glyph_like_pixels = int(np.count_nonzero(dark_panel_glyph_like))
    min_glyph_like = max(18, min(220, int(round(mask_area * 0.004))))
    glyph_like_ratio = glyph_like_pixels / float(max(1, mask_area))
    detected_dark_panel = (median_luma <= 45.0 and p90_luma <= 95.0) or (
        median_luma <= 58.0
        and p80_luma <= 125.0
        and glyph_like_pixels >= min_glyph_like
        and glyph_like_ratio >= 0.16
    )
    if not metadata_candidate and not detected_dark_panel:
        return None
    max_geometry_ratio = 0.70 if colored_status_panel else (0.52 if detected_dark_panel else 0.22)
    if mask_area > int(width * height * max_geometry_ratio):
        return None
    if colored_status_panel:
        if p90_luma > 245.0 and sampled_colored_background is None and metadata_fill_color is None:
            return None
    else:
        if median_luma > 150.0 and metadata_fill_color is None:
            if not (derived_card_panel or rejected_visual_card_panel or false_white_card_candidate):
                return None
        if p90_luma > 210.0 and metadata_fill_color is None:
            if not (derived_card_panel or rejected_visual_card_panel or false_white_card_candidate):
                return None

    if translator_note_text_mask:
        dark_cutoff = max(36.0, min(96.0, p25_luma + 10.0))
        dark_sample = sample[sample_luma <= dark_cutoff]
        if metadata_fill_color is not None:
            fill = metadata_fill_color.astype(np.float32)
        elif dark_sample.shape[0] >= 24:
            fill = np.median(dark_sample, axis=0)
        else:
            return None
    elif sampled_colored_background is not None:
        fill = sampled_colored_background.astype(np.float32)
    elif dark_bubble_inner_fill_color is not None:
        fill = dark_bubble_inner_fill_color.astype(np.float32)
    elif false_white_card_candidate:
        dark_cutoff = max(28.0, min(86.0, p25_luma + 8.0))
        dark_sample = sample[sample_luma <= dark_cutoff]
        if dark_sample.shape[0] >= 24:
            fill = np.median(dark_sample, axis=0)
        else:
            fill = sample_fill_color.astype(np.float32)
    elif (
        metadata_fill_color is not None
        and background_hint is not None
        and background_hint[0] <= 100.0
        and not colored_status_panel
        and p25_luma <= max(46.0, background_hint[0] - 12.0)
    ):
        dark_cutoff = max(34.0, min(82.0, p25_luma + 8.0))
        dark_sample = sample[sample_luma <= dark_cutoff]
        if dark_sample.shape[0] >= 32:
            fill = np.median(dark_sample, axis=0)
        else:
            fill = metadata_fill_color.astype(np.float32)
    elif metadata_fill_color is not None:
        fill = metadata_fill_color.astype(np.float32)
    else:
        fill = sample_fill_color.astype(np.float32)
    fill_color = np.asarray([int(max(0, min(255, round(float(v))))) for v in fill], dtype=np.uint8)
    fill_luma = float(np.mean(fill_color.astype(np.float32)))
    rejected_dark_local_context = bool(
        rejected_visual_card_panel
        and (
            _text_has_dark_visual_context(text)
            or detected_dark_panel
            or (fill_luma <= 90.0 and p25_luma <= 110.0)
            or (median_luma <= 82.0 and p90_luma <= 145.0)
        )
    )
    broken_image_white_dark_context = bool(
        bubble_mask_source == "image_white_bubble_mask"
        and (
            false_white_card_candidate
            or "mask_outside_balloon_critical" in qa_flags
            or "mask_outside_balloon" in qa_flags
            or "dark_bubble_visual_glyph_mask_replaced_geometry" in qa_flags
        )
        and (
            _text_has_dark_visual_context(text)
            or detected_dark_panel
            or (fill_luma <= 96.0 and p25_luma <= 112.0)
            or (median_luma <= 88.0 and p90_luma <= 150.0)
        )
    )
    text_bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height) or _normalize_bbox(
        text.get("bbox"),
        width,
        height,
    )

    def _visual_light_text_mask_within_balloon() -> np.ndarray | None:
        clip_bbox = _normalize_bbox(text.get("balloon_bbox") or text.get("bubble_mask_bbox"), width, height)
        if clip_bbox is None:
            return None
        loose_dark_visual_source = (
            rejected_visual_card_panel
            or bubble_mask_source == "bbox_fallback"
            or broken_image_white_dark_context
        )
        if loose_dark_visual_source and text_bbox is not None:
            clip_bbox = [
                min(clip_bbox[0], text_bbox[0]),
                min(clip_bbox[1], text_bbox[1]),
                max(clip_bbox[2], text_bbox[2]),
                max(clip_bbox[3], text_bbox[3]),
            ]
        clip_padding = 42 if broken_image_white_dark_context else (64 if loose_dark_visual_source else 18)
        clip_bbox = _expanded_bbox(width, height, clip_bbox, padding=clip_padding) or clip_bbox
        bx1, by1, bx2, by2 = clip_bbox
        crop = image_rgb[by1:by2, bx1:bx2]
        if crop.size == 0:
            return None
        crop_luma = (
            crop[:, :, 0].astype(np.float32) * 0.299
            + crop[:, :, 1].astype(np.float32) * 0.587
            + crop[:, :, 2].astype(np.float32) * 0.114
        )
        if bubble_mask_source == "image_dark_bubble_mask":
            bg_luma = float(np.percentile(crop_luma, 20)) if crop_luma.size else 0.0
            bright_threshold = max(174.0, bg_luma + 110.0)
        else:
            bright_threshold = 95.0
        bright = np.where(crop_luma >= bright_threshold, 255, 0).astype(np.uint8)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(bright, 8)
        local = np.zeros_like(bright)
        for label in range(1, component_count):
            x, y, w, h, area = [int(v) for v in stats[label]]
            if area < 10:
                continue
            component_bbox = [bx1 + x, by1 + y, bx1 + x + w, by1 + y + h]
            if text_bbox is not None:
                overlap = _bbox_overlap_ratio(component_bbox, text_bbox)
                near_text_column = bool(
                    component_bbox[2] >= text_bbox[0] - 24
                    and component_bbox[0] <= text_bbox[2] + 24
                    and component_bbox[3] >= text_bbox[1] - 48
                    and component_bbox[1] <= text_bbox[3] + 48
                )
                nearby_small_text_component = bool(area <= 720 and w <= 44 and h <= 44 and near_text_column)
                same_dark_bubble_component = False
                if overlap <= 0.01 and not nearby_small_text_component and not same_dark_bubble_component:
                    continue
            max_area = 2400
            if text_bbox is not None:
                text_area = max(1, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
                max_area = max(max_area, int(round(text_area * 0.90)))
            if area > max_area:
                continue
            if w > max(260, (bx2 - bx1) * 0.92) or h > max(90, (by2 - by1) * 0.80):
                continue
            local[labels == label] = 255
        if not np.any(local):
            return None
        kernel_size = 5 if bubble_mask_source == "image_dark_bubble_mask" else 19
        local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)), iterations=1)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[by1:by2, bx1:bx2] = local
        return mask

    dark_panel_padding = 1
    fill_mask: np.ndarray
    has_line_geometry = bool(text.get("line_polygons"))

    def _refined_dark_bubble_visual_fill_mask() -> np.ndarray | None:
        if bubble_mask_source != "image_dark_bubble_mask":
            return None
        candidate = dict(text)
        mask = build_inpaint_mask(candidate, image_rgb.shape, image_rgb=image_rgb)
        if not isinstance(mask, np.ndarray) or not np.any(mask):
            return None
        flags = {str(flag).strip() for flag in candidate.get("qa_flags") or [] if str(flag).strip()}
        metrics = candidate.get("qa_metrics") if isinstance(candidate.get("qa_metrics"), dict) else {}
        replaced_geometry = bool(
            "dark_bubble_visual_glyph_mask_replaced_geometry" in flags
            or isinstance(metrics.get("dark_bubble_visual_glyph_mask_replaced_geometry"), dict)
        )
        mask = _coerce_mask_for_shape(mask, image_rgb.shape[:2])
        if not np.any(mask):
            return None
        geometry_pixels = int(np.count_nonzero(text_mask))
        mask_pixels = int(np.count_nonzero(mask))
        relaxed_bbox_only_visual = bool(
            not has_line_geometry
            and _text_has_no_glyph_evidence(text)
            and (
                isinstance(metrics.get("dark_bubble_visual_glyph_mask"), dict)
                or "bubble_clip_preserved_raw_text" in flags
            )
            and mask_pixels >= max(96, int(round(max(1, geometry_pixels) * 0.08)))
            and mask_pixels <= int(max(geometry_pixels * 0.86, geometry_pixels - 96))
        )
        if not (replaced_geometry or relaxed_bbox_only_visual):
            return None
        if geometry_pixels > 0 and mask_pixels > int(geometry_pixels * 0.92):
            return None
        mask = _clip_dark_bubble_fill_mask_to_lobe(text, mask, image_rgb.shape[:2])
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels < 48:
            return None
        text_flags = text.setdefault("qa_flags", [])
        if not isinstance(text_flags, list):
            text_flags = []
            text["qa_flags"] = text_flags
        existing_flags = {str(flag).strip() for flag in text_flags if str(flag).strip()}
        for flag in flags:
            if flag and flag not in existing_flags:
                text_flags.append(flag)
                existing_flags.add(flag)
        candidate_metrics = candidate.get("qa_metrics")
        if isinstance(candidate_metrics, dict):
            text_metrics = text.setdefault("qa_metrics", {})
            if isinstance(text_metrics, dict):
                text_metrics.update(candidate_metrics)
        candidate_evidence = candidate.get("mask_evidence")
        if isinstance(candidate_evidence, dict):
            text["mask_evidence"] = dict(candidate_evidence)
        if relaxed_bbox_only_visual:
            text_metrics = text.setdefault("qa_metrics", {})
            if isinstance(text_metrics, dict):
                text_metrics["dark_bubble_visual_glyph_mask_replaced_bbox_geometry"] = {
                    "visual_pixels": int(mask_pixels),
                    "geometry_pixels": int(geometry_pixels),
                }
        text["_authorize_fast_fill_by_balloon_bbox"] = True
        text["_force_solid_dark_text_fill"] = True
        return mask.astype(np.uint8)

    if translator_note_text_mask:
        glyph_mask = _colored_status_panel_glyph_mask(
            image_rgb,
            text,
            text_mask,
            np.asarray([int(max(0, min(255, round(float(v))))) for v in fill], dtype=np.uint8),
            dilation_size=7,
        )
        fill_mask = glyph_mask if glyph_mask is not None else cv2.dilate(
            text_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
    elif broken_image_white_dark_context and text_bbox is not None:
        visual_mask = _visual_light_text_mask_within_balloon()
        text["_authorize_fast_fill_by_text_bbox"] = True
        text["_authorize_fast_fill_padding"] = 18
        text["_force_solid_dark_text_fill"] = True
        if visual_mask is not None:
            ys, xs = np.where(visual_mask > 0)
            if xs.size and ys.size:
                visual_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                text_area = max(1, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
                visual_area = max(1, (visual_bbox[2] - visual_bbox[0]) * (visual_bbox[3] - visual_bbox[1]))
                visual_overlap = _bbox_overlap_ratio(visual_bbox, text_bbox)
                if visual_area > int(text_area * 2.4) or visual_overlap <= 0.08:
                    fill_mask = _mask_from_bbox(width, height, text_bbox, padding=18)
                    qa_metrics = text.setdefault("qa_metrics", {})
                    if isinstance(qa_metrics, dict):
                        qa_metrics["broken_white_dark_visual_mask_rejected"] = {
                            "visual_bbox": list(visual_bbox),
                            "text_bbox": list(text_bbox),
                            "visual_area": int(visual_area),
                            "text_area": int(text_area),
                            "overlap": round(float(visual_overlap), 6),
                        }
                else:
                    fill_mask = _mask_from_bbox(width, height, visual_bbox, padding=10)
            else:
                fill_mask = visual_mask
        else:
            fill_mask = _mask_from_bbox(width, height, text_bbox, padding=18)
    elif false_white_card_candidate:
        if text_bbox is None:
            return None
        text["_authorize_fast_fill_by_balloon_bbox"] = True
        text["_authorize_fast_fill_padding"] = 58
        fill_mask = _mask_from_bbox(width, height, text_bbox, padding=58)
    elif rejected_dark_local_context and text_bbox is not None:
        visual_mask = _visual_light_text_mask_within_balloon()
        if visual_mask is not None:
            text["_authorize_fast_fill_by_text_bbox"] = True
            text["_authorize_fast_fill_padding"] = 64
            text["_force_solid_dark_text_fill"] = True
            ys, xs = np.where(visual_mask > 0)
            if xs.size and ys.size:
                visual_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                fill_mask = _mask_from_bbox(width, height, visual_bbox, padding=16)
            else:
                fill_mask = visual_mask
        else:
            text["_authorize_fast_fill_by_text_bbox"] = True
            text["_authorize_fast_fill_padding"] = 58
            fill_mask = _mask_from_bbox(width, height, text_bbox, padding=58)
    elif rejected_visual_card_panel and _text_has_no_glyph_evidence(text) and text_bbox is not None:
        text["_authorize_fast_fill_by_text_bbox"] = True
        text["_authorize_fast_fill_padding"] = 38
        fill_mask = _mask_from_bbox(width, height, text_bbox, padding=38)
    elif (
        bubble_mask_source == "bbox_fallback"
        and str(text.get("bubble_mask_error") or "").strip().lower() == "missing_real_bubble_mask"
        and text_bbox is not None
        and (detected_dark_panel or fill_luma <= 90.0 or p25_luma <= 110.0)
    ):
        text["_authorize_fast_fill_by_text_bbox"] = True
        text["_authorize_fast_fill_padding"] = 8
        text["_force_local_inpaint_dark_panel_fill"] = True
        fill_mask = _mask_from_bbox(width, height, text_bbox, padding=8)
    elif bubble_mask_source == "image_dark_bubble_mask" and text_bbox is not None and _has_trustworthy_glyph_action_mask(text):
        raw_action_mask = build_raw_text_mask_from_image(dict(text), image_rgb, image_rgb.shape)
        geometry_action_mask = cv2.dilate(
            text_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        refined_visual_fill = _refined_dark_bubble_visual_fill_mask() if _text_has_no_glyph_evidence(text) else None
        if refined_visual_fill is not None:
            fill_mask = refined_visual_fill
        elif isinstance(raw_action_mask, np.ndarray) and np.any(raw_action_mask):
            fill_mask = raw_action_mask.astype(np.uint8)
            no_glyph_visual_only = _text_has_no_glyph_evidence(text)
            glyph_kernel_size = 7 if no_glyph_visual_only else 9
            glyph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (glyph_kernel_size, glyph_kernel_size))
            fill_mask = cv2.dilate(fill_mask, glyph_kernel, iterations=1)
            if no_glyph_visual_only:
                fill_mask = _clip_dark_bubble_fill_mask_to_lobe(text, fill_mask, image_rgb.shape[:2])
                text["_force_solid_dark_text_fill"] = True
                qa_metrics = text.setdefault("qa_metrics", {})
                if isinstance(qa_metrics, dict):
                    qa_metrics["dark_bubble_raw_visual_mask_replaced_geometry"] = {
                        "raw_pixels": int(np.count_nonzero(raw_action_mask)),
                        "visual_pixels": int(np.count_nonzero(fill_mask)),
                        "geometry_pixels": int(np.count_nonzero(geometry_action_mask)),
                    }
            else:
                fill_mask = np.maximum(fill_mask, geometry_action_mask).astype(np.uint8)
        else:
            fill_mask = geometry_action_mask
        text["_authorize_fast_fill_by_balloon_bbox"] = True
    elif bubble_mask_source == "image_dark_bubble_mask" and text_bbox is not None and not _has_trustworthy_glyph_action_mask(text):
        refined_visual_fill = _refined_dark_bubble_visual_fill_mask()
        if refined_visual_fill is not None:
            fill_mask = refined_visual_fill
        else:
            visual_mask = _visual_light_text_mask_within_balloon()
            if visual_mask is not None:
                text["_authorize_fast_fill_by_balloon_bbox"] = True
                text["_force_solid_dark_text_fill"] = True
                fill_mask = _clip_dark_bubble_fill_mask_to_lobe(text, visual_mask.astype(np.uint8), image_rgb.shape[:2])
                qa_metrics = text.setdefault("qa_metrics", {})
                if isinstance(qa_metrics, dict):
                    qa_metrics["dark_bubble_visual_glyph_fill_mask"] = {
                        "mask_pixels": int(np.count_nonzero(fill_mask)),
                    }
            else:
                text["_authorize_fast_fill_by_balloon_bbox"] = True
                text["_authorize_fast_fill_padding"] = 58
                fill_mask = _mask_from_bbox(width, height, text_bbox, padding=58)
    elif unsafe_card_glyph_only:
        glyph_mask = _unsafe_card_visual_glyph_mask(image_rgb, text_mask)
        if glyph_mask is None:
            return None
        fill_mask = glyph_mask
        if rejected_visual_card_panel and float(np.mean(fill_color.astype(np.float32))) <= 90.0 and text_bbox is not None:
            text["_authorize_fast_fill_by_text_bbox"] = True
            bbox_mask = _mask_from_bbox(width, height, text_bbox, padding=6)
            fill_mask = np.maximum(fill_mask, bbox_mask).astype(np.uint8)
            visual_mask = _visual_light_text_mask_within_balloon()
            if visual_mask is not None:
                text["_authorize_fast_fill_by_balloon_bbox"] = True
                fill_mask = np.maximum(fill_mask, visual_mask).astype(np.uint8)
            visual_mask = _visual_light_text_mask_within_balloon()
            if visual_mask is not None:
                text["_authorize_fast_fill_by_balloon_bbox"] = True
                fill_mask = np.maximum(fill_mask, visual_mask).astype(np.uint8)
    elif detected_dark_panel and text_bbox is not None and not has_line_geometry:
        max_side = max(text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        pad = max(8, min(22, int(round(max_side * 0.06))))
        dark_panel_padding = pad
        fill_bbox = _expanded_bbox(width, height, text_bbox, padding=pad) or text_bbox
        fx1, fy1, fx2, fy2 = fill_bbox
        fill_mask = np.zeros((height, width), dtype=np.uint8)
        fill_mask[fy1:fy2, fx1:fx2] = 255
    elif colored_status_panel and not (derived_card_panel or rejected_visual_card_panel):
        glyph_mask = _colored_status_panel_glyph_mask(image_rgb, text, text_mask, fill_color)
        if glyph_mask is None:
            return None
        fill_mask = glyph_mask
    elif rejected_visual_card_panel and text_bbox is not None and _text_has_dark_visual_context(text):
        text["_authorize_fast_fill_by_text_bbox"] = True
        text["_authorize_fast_fill_padding"] = 38
        fill_mask = _mask_from_bbox(width, height, text_bbox, padding=38)
    elif derived_card_panel and has_line_geometry:
        kernel_size = 15 if bubble_mask_source == "image_dark_bubble_mask" else 11
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        fill_mask = cv2.dilate(text_mask, fill_kernel, iterations=1)
        clip_bbox = _normalize_bbox(text.get("bubble_mask_bbox") or text.get("balloon_bbox"), width, height)
        if clip_bbox is not None:
            clip = np.zeros((height, width), dtype=np.uint8)
            cx1, cy1, cx2, cy2 = clip_bbox
            clip[cy1:cy2, cx1:cx2] = 255
            fill_mask = np.where(clip > 0, fill_mask, 0).astype(np.uint8)
    elif colored_card_text and (not ui_panel_text or derived_card_panel or rejected_visual_card_panel):
        sampled_bg = (
            sample_fill_color
            if (derived_card_panel or rejected_visual_card_panel) and metadata_fill_color is None
            else None
        )
        glyph_mask = _colored_status_panel_glyph_mask(
            image_rgb,
            text,
            text_mask,
            sampled_bg,
            dilation_size=7 if derived_card_panel else 11,
        )
        if (
            glyph_mask is not None
            and (derived_card_panel or rejected_visual_card_panel)
            and int(np.count_nonzero(glyph_mask)) > int(mask_area * 0.62)
        ):
            glyph_mask = None
        if glyph_mask is None:
            glyph_mask = _unsafe_card_visual_glyph_mask(
                image_rgb,
                text_mask,
                strict_text_rect=bool(derived_card_panel or rejected_visual_card_panel),
            )
        if glyph_mask is None:
            return None
        fill_mask = glyph_mask
        if rejected_visual_card_panel and float(np.mean(fill_color.astype(np.float32))) <= 90.0 and text_bbox is not None:
            text["_authorize_fast_fill_by_text_bbox"] = True
            bbox_mask = _mask_from_bbox(width, height, text_bbox, padding=6)
            fill_mask = np.maximum(fill_mask, bbox_mask).astype(np.uint8)
    else:
        kernel_size = 25 if colored_status_panel else 15
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        fill_mask = cv2.dilate(text_mask, fill_kernel, iterations=1)
    contract_fill_mask = _dark_text_contract_fill_mask(text, width, height, image_rgb)
    contract_direct_allowed = bool(
        translator_note_text_mask
        or qa_flags
        & {
            "text_contract_direct_fill",
            "visual_text_only_inpaint_contract",
            "weak_text_residual_after_inpaint",
        }
    )
    if bubble_mask_source == "image_dark_bubble_mask" and not contract_direct_allowed:
        contract_fill_mask = None
    if contract_fill_mask is not None:
        if bubble_mask_source == "image_dark_bubble_mask":
            contract_fill_mask = _clip_dark_bubble_fill_mask_to_lobe(
                text,
                contract_fill_mask.astype(np.uint8),
                image_rgb.shape[:2],
            )
            if not np.any(contract_fill_mask):
                return None
        visual_contract_fill_mask = _dark_panel_visual_contract_fill_mask(
            image_rgb,
            text,
            contract_fill_mask,
            width,
            height,
        )
        fill_mask = (
            visual_contract_fill_mask.astype(np.uint8)
            if visual_contract_fill_mask is not None
            else contract_fill_mask.astype(np.uint8)
        )
    authorized_padding = _authorized_dark_panel_padding(
        text,
        has_line_geometry=has_line_geometry,
        detected_dark_panel=detected_dark_panel,
        colored_status_panel=colored_status_panel,
        base_padding=dark_panel_padding,
    )
    authorized_mask = _authorized_fast_fill_mask(width, height, text, padding=authorized_padding)
    if authorized_mask is None:
        return None
    if contract_fill_mask is not None:
        authorized_mask = np.maximum(authorized_mask.astype(np.uint8), contract_fill_mask.astype(np.uint8))
    fill_mask = cv2.bitwise_and(fill_mask.astype(np.uint8), authorized_mask.astype(np.uint8))
    if bubble_mask_source == "image_dark_bubble_mask" and contract_fill_mask is None:
        fill_mask = _clip_dark_bubble_fill_mask_to_lobe(text, fill_mask, image_rgb.shape[:2])
    if not np.any(fill_mask):
        return None
    if contract_fill_mask is not None:
        result = image_rgb.copy()
        solid_fill = (
            dark_bubble_inner_fill_color.astype(np.uint8)
            if dark_bubble_inner_fill_color is not None
            else fill_color.astype(np.uint8)
        )
        result[fill_mask > 0] = solid_fill
        _remember_text_fill_mask(text, "_dark_panel_fill_mask", fill_mask)
        qa_metrics = text.setdefault("qa_metrics", {})
        if isinstance(qa_metrics, dict):
            qa_metrics["text_contract_direct_fill"] = {
                "mask_pixels": int(np.count_nonzero(fill_mask)),
                "contract_mask_pixels": int(np.count_nonzero(contract_fill_mask)),
                "fill_rgb": [int(v) for v in solid_fill.tolist()],
                "reason": "expanded_text_contract_mask",
            }
        _append_text_flag(text, "text_contract_direct_fill")
        return result
    sample_std = float(np.mean(np.std(sample, axis=0))) if sample.size else 0.0
    textured_panel = bool(
        not translator_note_text_mask
        and
        (unsafe_card_glyph_only or colored_card_text or colored_status_panel or detected_dark_panel)
        and (
            derived_card_panel
            or
            sample_std >= 18.0
            or (p90_luma - p25_luma) >= 42.0
            or (background_hint is not None and background_hint[0] <= 110.0 and p90_luma >= 80.0)
        )
    )
    solid_dark_glyph_fill = bool(
        translator_note_text_mask
        or (
            false_white_card_candidate
            and float(np.mean(fill_color.astype(np.float32))) <= 90.0
            and p25_luma <= 95.0
        )
        or (
            bubble_mask_source == "image_dark_bubble_mask"
            and float(np.mean(fill_color.astype(np.float32))) <= 90.0
            and p25_luma <= 95.0
            and not _has_trustworthy_glyph_action_mask(text)
        )
        or
        (
            unsafe_card_glyph_only
            and background_hint is not None
            and background_hint[0] <= 110.0
            and p25_luma <= 75.0
        )
        or (
            rejected_visual_card_panel
            and float(np.mean(fill_color.astype(np.float32))) <= 90.0
            and p25_luma <= 95.0
        )
    )
    dark_bubble_untrusted_local_inpaint = bool(
        bubble_mask_source == "image_dark_bubble_mask"
        and not translator_note_text_mask
        and (
            _text_has_no_glyph_evidence(text)
            or not _has_trustworthy_glyph_action_mask(text)
        )
        and int(np.count_nonzero(fill_mask)) >= 8
    )
    if dark_bubble_untrusted_local_inpaint:
        inpaint_mask = fill_mask.astype(np.uint8)
        if dark_bubble_inner_fill_color is not None and fill_luma <= 36.0 and p25_luma <= 42.0:
            try:
                solid_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                inpaint_mask = cv2.dilate(inpaint_mask, solid_kernel, iterations=1).astype(np.uint8)
                inpaint_mask = cv2.bitwise_and(inpaint_mask, authorized_mask.astype(np.uint8))
                inpaint_mask = _clip_dark_bubble_fill_mask_to_lobe(text, inpaint_mask, image_rgb.shape[:2])
                if not np.any(inpaint_mask):
                    inpaint_mask = fill_mask.astype(np.uint8)
                filled = image_rgb.copy()
                _remember_text_fill_mask(text, "_dark_panel_fill_mask", inpaint_mask)
                qa_metrics = text.setdefault("qa_metrics", {})
                if isinstance(qa_metrics, dict):
                    qa_metrics["dark_bubble_local_solid_fill"] = {
                        "mask_pixels": int(np.count_nonzero(inpaint_mask)),
                        "reason": "flat_dark_bubble_untrusted_glyph_mask",
                        "fill_rgb": [int(v) for v in dark_bubble_inner_fill_color.tolist()],
                    }
                filled[inpaint_mask > 0] = dark_bubble_inner_fill_color.astype(np.uint8)
                return filled
            except Exception:
                pass
        try:
            _remember_text_fill_mask(text, "_dark_panel_fill_mask", inpaint_mask)
            qa_metrics = text.setdefault("qa_metrics", {})
            if isinstance(qa_metrics, dict):
                qa_metrics["dark_bubble_local_inpaint_fill"] = {
                    "mask_pixels": int(np.count_nonzero(inpaint_mask)),
                    "reason": "untrusted_dark_bubble_glyph_mask",
                }
            return cv2.inpaint(image_rgb, inpaint_mask, 3, cv2.INPAINT_TELEA)
        except Exception:
            pass
    if (
        bubble_mask_source == "image_dark_bubble_mask"
        and dark_bubble_inner_fill_color is not None
        and fill_luma <= 36.0
        and p25_luma <= 42.0
        and int(np.count_nonzero(fill_mask)) >= 8
    ):
        try:
            solid_mask = fill_mask.astype(np.uint8)
            solid_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            solid_mask = cv2.dilate(solid_mask, solid_kernel, iterations=1).astype(np.uint8)
            solid_mask = cv2.bitwise_and(solid_mask, authorized_mask.astype(np.uint8))
            solid_mask = _clip_dark_bubble_fill_mask_to_lobe(text, solid_mask, image_rgb.shape[:2])
            if np.any(solid_mask):
                result = image_rgb.copy()
                _remember_text_fill_mask(text, "_dark_panel_fill_mask", solid_mask)
                qa_metrics = text.setdefault("qa_metrics", {})
                if isinstance(qa_metrics, dict):
                    qa_metrics["dark_bubble_local_solid_fill"] = {
                        "mask_pixels": int(np.count_nonzero(solid_mask)),
                        "reason": "flat_dark_bubble_glyph_mask",
                        "fill_rgb": [int(v) for v in dark_bubble_inner_fill_color.tolist()],
                    }
                result[solid_mask > 0] = dark_bubble_inner_fill_color.astype(np.uint8)
                return result
        except Exception:
            pass
    if bool(text.get("_force_local_inpaint_dark_panel_fill")):
        inpaint_mask = fill_mask.astype(np.uint8)
        if int(np.count_nonzero(inpaint_mask)) >= 8:
            try:
                _remember_text_fill_mask(text, "_dark_panel_fill_mask", inpaint_mask)
                qa_metrics = text.setdefault("qa_metrics", {})
                if isinstance(qa_metrics, dict):
                    metric_key = "dark_panel_bbox_fallback_local_inpaint"
                    metric_payload = {
                        "mask_pixels": int(np.count_nonzero(inpaint_mask)),
                        "reason": "missing_real_bubble_mask_text_bbox_limited",
                    }
                flat_dark_panel = bool(
                    fill_luma <= 42.0
                    and p25_luma <= 50.0
                    and sample_std <= 12.0
                    and not textured_panel
                )
                if flat_dark_panel:
                    result = image_rgb.copy()
                    solid_fill = (
                        dark_bubble_inner_fill_color.astype(np.uint8)
                        if dark_bubble_inner_fill_color is not None
                        else fill_color.astype(np.uint8)
                    )
                    result[inpaint_mask > 0] = solid_fill
                    if isinstance(qa_metrics, dict):
                        qa_metrics["dark_panel_bbox_fallback_solid_fill"] = {
                            **metric_payload,
                            "reason": "missing_real_bubble_mask_flat_dark_text_bbox_limited",
                            "fill_rgb": [int(v) for v in solid_fill.tolist()],
                            "sample_std": round(float(sample_std), 4),
                        }
                    return result
                if isinstance(qa_metrics, dict):
                    qa_metrics[metric_key] = metric_payload
                return cv2.inpaint(image_rgb, inpaint_mask, 3, cv2.INPAINT_TELEA)
            except Exception:
                pass
    if textured_panel and not solid_dark_glyph_fill:
        inpaint_mask = fill_mask.astype(np.uint8)
        if int(np.count_nonzero(inpaint_mask)) >= 8:
            try:
                _remember_text_fill_mask(text, "_dark_panel_fill_mask", inpaint_mask)
                return cv2.inpaint(image_rgb, inpaint_mask, 3, cv2.INPAINT_TELEA)
            except Exception:
                pass

    result = image_rgb.copy()
    if (
        not translator_note_text_mask
        and not bool(text.get("_force_solid_dark_text_fill"))
        and float(np.mean(fill_color.astype(np.float32))) <= 90.0
        and int(np.count_nonzero(fill_mask)) >= 64
    ):
        soft_mask = cv2.GaussianBlur(fill_mask.astype(np.float32) / 255.0, (21, 21), 0)
        core = cv2.erode(fill_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        alpha = np.maximum(soft_mask, (core.astype(np.float32) / 255.0))[:, :, None]
        result = np.clip(
            result.astype(np.float32) * (1.0 - alpha) + fill_color.astype(np.float32)[None, None, :] * alpha,
            0,
            255,
        ).astype(np.uint8)
    else:
        result[fill_mask > 0] = fill_color
    _remember_text_fill_mask(text, "_dark_panel_fill_mask", fill_mask)
    return result


def _unsafe_white_balloon_limit_mask(
    ocr_page: dict,
    text: dict,
    width: int,
    height: int,
) -> np.ndarray | None:
    real_bubble_mask, _reason = _real_bubble_mask_for_text(ocr_page, text, width, height)
    if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask):
        return _safe_real_bubble_interior_mask(real_bubble_mask, width, height, erode_px=2)
    bbox = _normalize_bbox(
        text.get("bubble_mask_bbox")
        or text.get("bubbleMaskBbox")
        or text.get("balloon_bbox")
        or text.get("bbox"),
        width,
        height,
    )
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    mask = np.zeros((height, width), dtype=np.uint8)
    source = str(text.get("bubble_mask_source") or "").strip().lower()
    if source in {"image_contour_bubble_mask", "image_white_bubble_mask", "derived_white_bubble_mask"}:
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    else:
        mask[y1:y2, x1:x2] = 255
    return _safe_real_bubble_interior_mask(mask, width, height, erode_px=3)


def _apply_unsafe_white_balloon_text_fills(image_rgb: np.ndarray, ocr_page: dict) -> tuple[np.ndarray, int]:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or not isinstance(ocr_page, dict):
        return image_rgb, 0
    height, width = image_rgb.shape[:2]
    result = image_rgb.copy()
    fill_count = 0
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    filled_bboxes: list[list[int]] = []
    filled_text_keys: set[str] = set()
    candidates = [
        item
        for item in list(ocr_page.get("texts") or [])
        + list(ocr_page.get("_strip_unsafe_inpaint_block_samples") or [])
        + list(ocr_page.get("_vision_blocks") or [])
        if isinstance(item, dict)
        and _unsafe_white_balloon_requires_real_inpaint(item)
        and not _is_false_white_card_candidate(image_rgb, item)
    ]
    for text in candidates:
        text_mask = _text_geometry_mask(width, height, text)
        if text_mask is None or not np.any(text_mask):
            text_bbox = _normalize_bbox(text.get("text_pixel_bbox") or text.get("bbox"), width, height)
            if text_bbox is None:
                continue
            text_mask = _mask_from_bbox(width, height, text_bbox, padding=1)
        bbox = _bbox_from_binary_mask(text_mask)
        if bbox is None:
            continue
        bbox_key = tuple(int(v) for v in bbox)
        if bbox_key in seen_bboxes:
            continue
        seen_bboxes.add(bbox_key)
        limit_mask = _unsafe_white_balloon_limit_mask(ocr_page, text, width, height)
        if limit_mask is None or not np.any(limit_mask):
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        expanded = cv2.dilate(np.where(text_mask > 0, 255, 0).astype(np.uint8), kernel, iterations=1)
        fill_mask = cv2.bitwise_and(expanded, np.where(limit_mask > 0, 255, 0).astype(np.uint8))
        if not np.any(fill_mask):
            continue
        before = result.copy()
        result[fill_mask > 0] = np.asarray([255, 255, 255], dtype=np.uint8)
        if not np.any(np.any(result != before, axis=2)):
            continue
        fill_count += 1
        filled_bboxes.append(bbox)
        for key_name in ("id", "text_id", "trace_id", "text_instance_id"):
            value = text.get(key_name)
            if value is not None:
                filled_text_keys.add(str(value))
    if fill_count:
        ocr_page["_strip_unsafe_white_balloon_fill_count"] = int(fill_count)
        ocr_page["_strip_used_fast_white_fill"] = True
        flags_to_remove = {"mask_outside_balloon", "mask_outside_balloon_critical", "real_inpaint_skipped_unsafe_mask"}

        def _item_was_filled(item: dict) -> bool:
            if any(str(item.get(key_name) or "") in filled_text_keys for key_name in ("id", "text_id", "trace_id", "text_instance_id")):
                return True
            item_bbox = _normalize_bbox(item.get("text_pixel_bbox") or item.get("bbox"), width, height)
            if item_bbox is None:
                return False
            return any(
                _bbox_overlap_ratio(item_bbox, filled_bbox) >= 0.35
                or _bbox_center_inside(item_bbox, filled_bbox)
                or _bbox_center_inside(filled_bbox, item_bbox)
                for filled_bbox in filled_bboxes
            )

        for collection_key in ("texts", "_vision_blocks"):
            for item in ocr_page.get(collection_key) or []:
                if isinstance(item, dict) and _item_was_filled(item):
                    _remove_qa_flags_from_item(item, flags_to_remove)
        unsafe_samples = [
            sample
            for sample in ocr_page.get("_strip_unsafe_inpaint_block_samples") or []
            if not isinstance(sample, dict) or not _item_was_filled(sample)
        ]
        if unsafe_samples:
            ocr_page["_strip_unsafe_inpaint_block_samples"] = unsafe_samples
            reasons: dict[str, int] = {}
            for sample in unsafe_samples:
                if not isinstance(sample, dict):
                    continue
                reason = str(sample.get("reason") or "").strip()
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
            if reasons:
                ocr_page["_strip_unsafe_inpaint_block_reasons"] = reasons
                ocr_page["_strip_unsafe_inpaint_block_count"] = sum(reasons.values())
            else:
                ocr_page.pop("_strip_unsafe_inpaint_block_reasons", None)
                ocr_page.pop("_strip_unsafe_inpaint_block_count", None)
        else:
            ocr_page.pop("_strip_unsafe_inpaint_block_samples", None)
            ocr_page.pop("_strip_unsafe_inpaint_block_reasons", None)
            ocr_page.pop("_strip_unsafe_inpaint_block_count", None)
        decision_flags = [
            str(flag)
            for flag in ocr_page.get("_strip_inpaint_decision_flags") or []
            if str(flag).strip() not in flags_to_remove
        ]
        if decision_flags:
            ocr_page["_strip_inpaint_decision_flags"] = decision_flags
        else:
            ocr_page.pop("_strip_inpaint_decision_flags", None)
    return result, fill_count


def _is_ui_form_metadata_background_text_candidate(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    if _route_action_blocks_inpaint(text):
        return False
    if not (text.get("line_polygons") or text.get("text_pixel_bbox")):
        return False
    if _metadata_background_color(text) is None:
        return False
    profiles = {
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
    }
    if "ui_form" in profiles:
        return True
    evidence = text.get("ui_layout_evidence")
    if isinstance(evidence, dict) and str(evidence.get("source") or "").strip().lower() in {
        "uied_cv",
        "ui_layout",
        "layout_cv",
    }:
        return True
    return _source_text_has_colored_status_terms(text)


def _try_ui_white_source_bbox_text_fill(image_rgb: np.ndarray, text: dict) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        return None
    if not isinstance(text, dict) or _route_action_blocks_inpaint(text):
        return None
    if not _source_text_has_strict_ui_form_signal(text):
        return None
    color = _metadata_background_color(text)
    if color is None:
        return None
    color_i = color.astype(np.int16)
    if int(color.min()) < 252 or int(color.max()) - int(color.min()) > 8:
        return None
    height, width = image_rgb.shape[:2]
    source_bbox = (
        _normalize_bbox(text.get("source_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
    )
    if source_bbox is None:
        return None
    geometry_bbox = (
        _line_polygons_bbox(text, width, height)
        or _normalize_bbox(text.get("text_pixel_bbox"), width, height)
    )
    source_area = max(1, (source_bbox[2] - source_bbox[0]) * (source_bbox[3] - source_bbox[1]))
    geometry_area = (
        max(1, (geometry_bbox[2] - geometry_bbox[0]) * (geometry_bbox[3] - geometry_bbox[1]))
        if geometry_bbox is not None
        else 0
    )
    if geometry_area and source_area < max(96, int(geometry_area * 1.35)):
        return None
    sx1, sy1, sx2, sy2 = source_bbox
    crop = image_rgb[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return None
    geometry_mask = _text_geometry_mask(width, height, text)
    sample_mask = np.ones(crop.shape[:2], dtype=bool)
    if isinstance(geometry_mask, np.ndarray) and geometry_mask.shape[:2] == (height, width):
        exclusion = cv2.dilate(
            (geometry_mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        sample_mask = exclusion[sy1:sy2, sx1:sx2] == 0
    sample_pixels = crop[sample_mask].astype(np.float32)
    if sample_pixels.shape[0] >= 64:
        bg_luma = float(np.mean(color_i.astype(np.float32)))
        luma = (
            sample_pixels[:, 0] * 0.299
            + sample_pixels[:, 1] * 0.587
            + sample_pixels[:, 2] * 0.114
        )
        chroma = np.max(sample_pixels, axis=1) - np.min(sample_pixels, axis=1)
        panel_like = (
            (luma >= 80.0)
            & (luma <= 245.0)
            & (
                (chroma >= 8.0)
                | (np.abs(luma - bg_luma) >= 28.0)
            )
        )
        if float(np.mean(panel_like)) >= 0.12:
            return None
    delta = np.mean(np.abs(crop.astype(np.int16) - color_i[None, None, :]), axis=2)
    close_ratio = float(np.mean(delta <= 18.0))
    if close_ratio < 0.42:
        return None
    result = image_rgb.copy()
    fill_mask = _mask_from_bbox(width, height, source_bbox, padding=3)
    result[fill_mask > 0] = color.astype(np.uint8)
    return result


def _try_ui_metadata_geometry_text_fill(image_rgb: np.ndarray, text: dict) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        return None
    if not isinstance(text, dict) or _route_action_blocks_inpaint(text):
        return None
    if not _source_text_has_strict_ui_form_signal(text):
        return None
    color = _metadata_background_color(text)
    if color is None:
        return None
    height, width = image_rgb.shape[:2]
    mask = _text_geometry_mask(width, height, text)
    if mask is None or not np.any(mask):
        return None
    mask_area = int(np.count_nonzero(mask > 0))
    if mask_area > int(width * height * 0.35):
        return None
    pixels = image_rgb[mask > 0].astype(np.int16)
    if pixels.size == 0:
        return None
    bg_i = color.astype(np.int16)
    delta = np.mean(np.abs(pixels - bg_i[None, :]), axis=1)
    close_ratio = float(np.mean(delta <= 28.0))
    if close_ratio < 0.30:
        return None
    if float(np.percentile(delta, 90)) < 24.0:
        return None
    result = image_rgb.copy()
    result[mask > 0] = color.astype(np.uint8)
    return result


def _apply_flat_ui_text_prefill_to_blocks(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    if not isinstance(band_rgb, np.ndarray) or band_rgb.ndim != 3 or not vision_blocks:
        return band_rgb, vision_blocks, {"flat_ui_prefill_count": 0, "remaining_blocks": len(vision_blocks)}
    height, width = band_rgb.shape[:2]
    result = band_rgb.copy()
    filled_bboxes: list[list[int]] = []
    filled_mask = np.zeros((height, width), dtype=np.uint8)
    fill_count = 0
    for text in _processable_texts_for_inpaint(ocr_page):
        if not _source_text_has_strict_ui_form_signal(text):
            continue
        before = result
        filled = (
            _try_metadata_background_text_fill(result, text)
            if _is_ui_form_metadata_background_text_candidate(text)
            else None
        )
        if filled is None:
            filled = _try_ui_metadata_geometry_text_fill(result, text)
        if filled is None:
            filled = _try_dark_panel_text_fill(result, text)
        if filled is None:
            filled = _try_ui_white_source_bbox_text_fill(result, text)
        if filled is None:
            continue
        changed_mask = np.any(filled != before, axis=2).astype(np.uint8) * 255
        if not np.any(changed_mask):
            continue
        dark_fill_mask = _pop_text_fill_mask(text, "_dark_panel_fill_mask", changed_mask.shape[:2])
        if np.any(dark_fill_mask):
            changed_mask = np.maximum(changed_mask, dark_fill_mask).astype(np.uint8)
        fill_bbox = (
            _bbox_from_binary_mask(changed_mask)
            or _normalize_bbox(text.get("source_bbox"), width, height)
            or _normalize_bbox(text.get("bbox"), width, height)
            or _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        )
        if fill_bbox is None:
            continue
        result = filled
        filled_bboxes.append(fill_bbox)
        filled_mask = np.maximum(filled_mask, changed_mask)
        _accumulate_page_fill_mask(ocr_page, "_strip_dark_panel_fill_mask", changed_mask, changed_mask.shape[:2])
        fill_count += 1
    if not fill_count:
        return band_rgb, vision_blocks, {"flat_ui_prefill_count": 0, "remaining_blocks": len(vision_blocks)}
    remaining_blocks = [
        block
        for block in vision_blocks
        if not _block_is_covered_by_fast_fill(block, filled_bboxes, width, height, filled_mask)
    ]
    ocr_page["_strip_used_flat_ui_prefill"] = True
    ocr_page["_strip_flat_ui_prefill_count"] = int(fill_count)
    ocr_page["_strip_used_dark_panel_fill"] = True
    previous = int(ocr_page.get("_strip_dark_panel_fill_count") or 0)
    ocr_page["_strip_dark_panel_fill_count"] = previous + int(fill_count)
    ocr_page["_strip_remaining_inpaint_blocks"] = len(remaining_blocks)
    return result, remaining_blocks, {
        "flat_ui_prefill_count": int(fill_count),
        "remaining_blocks": len(remaining_blocks),
    }


def _apply_dark_panel_text_fills(image_rgb: np.ndarray, ocr_page: dict) -> tuple[np.ndarray, int]:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        return image_rgb, 0
    result = image_rgb.copy()
    fill_count = 0
    filled_text_keys: set[str] = set()
    candidate_texts = [
        item
        for item in ocr_page.get("texts", [])
        if isinstance(item, dict)
        and not _text_suppressed_for_inpaint(item)
        and _route_action_allows_local_dark_panel_fill(item)
        and not _text_has_rejected_bubble_without_glyph_evidence(item)
        and (not _text_has_no_glyph_evidence(item) or _dark_bubble_no_glyph_fast_fillable(item))
        and (
            not _unsafe_white_balloon_requires_real_inpaint(item)
            or _is_false_white_card_candidate(image_rgb, item)
        )
    ]
    dark_bubble_siblings = [
        item
        for item in candidate_texts
        if str(item.get("bubble_mask_source") or "").strip().lower() == "image_dark_bubble_mask"
    ]
    for text in candidate_texts:
        if str(text.get("bubble_mask_source") or "").strip().lower() == "image_dark_bubble_mask":
            text["_dark_bubble_sibling_texts"] = dark_bubble_siblings
        ui_metadata_candidate = _is_ui_form_metadata_background_text_candidate(text)
        flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
        false_white_card_candidate = (
            str(text.get("bubble_mask_source") or "").strip().lower() == "image_white_bubble_mask"
            and "mask_outside_balloon_critical" in flags
            and not _text_region_looks_plain_white(image_rgb, text)
        )
        translator_note_text_mask = str(text.get("bubble_mask_source") or "").strip().lower() == "translator_note_text_mask"
        dark_panel_candidate = (
            translator_note_text_mask
            or
            _is_colored_status_panel_text_candidate(text)
            or _is_dark_or_colored_card_text(text)
            or false_white_card_candidate
            or (
                _source_text_has_colored_status_terms(text)
                and bool(text.get("line_polygons") or text.get("text_pixel_bbox"))
            )
            or _is_dark_panel_text_candidate(text)
        )
        if not (ui_metadata_candidate or dark_panel_candidate):
            text.pop("_dark_bubble_sibling_texts", None)
            continue
        filled = _try_metadata_background_text_fill(result, text) if ui_metadata_candidate else None
        if filled is None and dark_panel_candidate:
            filled = _try_dark_panel_text_fill(result, text)
        text.pop("_dark_bubble_sibling_texts", None)
        if filled is None:
            continue
        changed = np.any(filled != result, axis=2)
        if not np.any(changed):
            continue
        dark_fill_mask = _pop_text_fill_mask(text, "_dark_panel_fill_mask", changed.shape[:2])
        if not np.any(dark_fill_mask):
            dark_fill_mask = changed.astype(np.uint8) * 255
        visual_override = _dark_bubble_visual_fill_override(result, text, dark_fill_mask)
        if isinstance(visual_override, np.ndarray) and np.any(visual_override):
            filled = np.where(visual_override[:, :, None] > 0, filled, result)
            changed = np.any(filled != result, axis=2)
            dark_fill_mask = visual_override
        elif _dark_fill_mask_is_overbroad_for_text(text, dark_fill_mask):
            continue
        _accumulate_page_fill_mask(ocr_page, "_strip_dark_panel_fill_mask", dark_fill_mask, changed.shape[:2])
        result = filled
        fill_count += 1
        text["_fast_fill_inpaint_resolved"] = True
        for key_name in ("id", "text_id", "trace_id", "text_instance_id"):
            value = text.get(key_name)
            if value is not None:
                filled_text_keys.add(str(value))
    cleanup_count = _apply_clipped_overlap_fragment_cleanup_fill(result, ocr_page)
    if cleanup_count:
        fill_count += cleanup_count
    if fill_count:
        for collection_key in ("texts", "_vision_blocks"):
            for item in ocr_page.get(collection_key) or []:
                if not isinstance(item, dict):
                    continue
                if any(str(item.get(key_name) or "") in filled_text_keys for key_name in ("id", "text_id", "trace_id", "text_instance_id")):
                    item["_fast_fill_inpaint_resolved"] = True
        unsafe_samples = [
            sample
            for sample in ocr_page.get("_strip_unsafe_inpaint_block_samples") or []
            if not isinstance(sample, dict)
            or not any(str(sample.get(key_name) or "") in filled_text_keys for key_name in ("id", "text_id", "trace_id", "text_instance_id"))
        ]
        if unsafe_samples:
            ocr_page["_strip_unsafe_inpaint_block_samples"] = unsafe_samples
            reasons: dict[str, int] = {}
            for sample in unsafe_samples:
                if not isinstance(sample, dict):
                    continue
                reason = str(sample.get("reason") or "").strip()
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
            if reasons:
                ocr_page["_strip_unsafe_inpaint_block_reasons"] = reasons
                ocr_page["_strip_unsafe_inpaint_block_count"] = sum(reasons.values())
            else:
                ocr_page.pop("_strip_unsafe_inpaint_block_reasons", None)
                ocr_page.pop("_strip_unsafe_inpaint_block_count", None)
        else:
            ocr_page.pop("_strip_unsafe_inpaint_block_samples", None)
            ocr_page.pop("_strip_unsafe_inpaint_block_reasons", None)
            ocr_page.pop("_strip_unsafe_inpaint_block_count", None)
            ocr_page["_strip_inpaint_decision_flags"] = [
                str(flag)
                for flag in ocr_page.get("_strip_inpaint_decision_flags") or []
                if str(flag).strip() not in {"mask_outside_balloon", "mask_outside_balloon_critical", "real_inpaint_skipped_unsafe_mask"}
            ]
            if not ocr_page["_strip_inpaint_decision_flags"]:
                ocr_page.pop("_strip_inpaint_decision_flags", None)
        ocr_page["_strip_used_dark_panel_fill"] = True
        previous = int(ocr_page.get("_strip_dark_panel_fill_count") or 0)
        ocr_page["_strip_dark_panel_fill_count"] = previous + fill_count
    return result, fill_count


def _clipped_overlap_fragment_cleanup_bbox_for_text(text: dict, width: int, height: int) -> list[int] | None:
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    cleanup = text.get("clipped_overlap_fragment_cleanup_bbox")
    metrics = text.get("qa_metrics") if isinstance(text.get("qa_metrics"), dict) else {}
    if not isinstance(cleanup, dict):
        cleanup = metrics.get("clipped_overlap_fragment_cleanup_bbox") if isinstance(metrics, dict) else None
    cleanup_bbox = cleanup.get("bbox") if isinstance(cleanup, dict) else None
    cleanup_bbox = _normalize_bbox(cleanup_bbox, width, height)
    if cleanup_bbox is not None:
        return cleanup_bbox
    if "false_dark_bubble_trailing_clipped_fragment_removed" not in flags:
        return None
    text_bbox = (
        _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
        or _normalize_bbox(text.get("source_bbox"), width, height)
    )
    if text_bbox is None:
        return None
    bubble_bbox = (
        _normalize_bbox(text.get("bubble_mask_bbox"), width, height)
        or _normalize_bbox(text.get("balloon_bbox"), width, height)
        or text_bbox
    )
    text_w = max(1, text_bbox[2] - text_bbox[0])
    text_h = max(1, text_bbox[3] - text_bbox[1])
    x1 = max(0, text_bbox[2] - max(96, int(round(text_w * 0.50))))
    x2 = min(width, text_bbox[2] + max(180, text_w))
    y1 = max(text_bbox[3] + max(18, int(round(text_h * 0.75))), bubble_bbox[3] - max(36, int(round(text_h * 0.50))))
    y2 = min(height, max(y1 + max(32, int(round(text_h * 0.70))), bubble_bbox[3] + max(48, int(round(text_h * 0.80)))))
    return _normalize_bbox([x1, y1, x2, y2], width, height)


def _apply_clipped_overlap_fragment_cleanup_fill(image_rgb: np.ndarray, ocr_page: dict) -> int:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        return 0
    height, width = image_rgb.shape[:2]
    applied = 0
    for text in ocr_page.get("texts", []):
        if not isinstance(text, dict) or _text_suppressed_for_inpaint(text):
            continue
        bbox = _clipped_overlap_fragment_cleanup_bbox_for_text(text, width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        cleanup_mask = _mask_from_bbox(width, height, bbox, padding=0)
        if not np.any(cleanup_mask):
            continue
        before = image_rgb.copy()
        image_rgb[y1:y2, x1:x2] = np.array([0, 0, 0], dtype=np.uint8)
        changed = np.any(image_rgb != before, axis=2).astype(np.uint8) * 255
        if not np.any(changed):
            continue
        changed = np.maximum(changed, cleanup_mask).astype(np.uint8)
        _accumulate_page_fill_mask(ocr_page, "_strip_dark_panel_fill_mask", changed, changed.shape[:2])
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["clipped_overlap_fragment_cleanup_fill"] = {
                "bbox": bbox,
                "changed_pixels": int(np.count_nonzero(changed)),
            }
        applied += 1
    if applied:
        flags = list(ocr_page.get("_strip_inpaint_decision_flags") or [])
        if "clipped_overlap_fragment_cleanup_fill" not in flags:
            flags.append("clipped_overlap_fragment_cleanup_fill")
        ocr_page["_strip_inpaint_decision_flags"] = flags
    return applied


def _apply_fast_dark_panel_text_fill(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    rejection_reasons: dict[str, int] = {}

    def _reject(reason: str) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    def _record(stats: dict) -> dict:
        ocr_page["_strip_fast_dark_panel_fill_count"] = stats["dark_panel_fill_count"]
        ocr_page["_strip_fast_dark_rejection_reasons"] = dict(rejection_reasons)
        return stats

    candidate_texts: list[dict] = []
    seen_candidate_keys: set[str] = set()
    candidate_by_key: dict[str, dict] = {}

    def _mask_evidence_quality(value) -> tuple[float, int, int]:
        if not isinstance(value, dict):
            return (0.0, 0, 0)
        try:
            score = float(value.get("evidence_score") or 0.0)
        except Exception:
            score = 0.0
        return (
            score,
            int(value.get("raw_mask_pixels") or 0),
            int(value.get("expanded_mask_pixels") or 0),
        )

    for source_text in list(ocr_page.get("texts", []) or []) + list(vision_blocks or []):
        if not isinstance(source_text, dict):
            continue
        key = None
        for key_name in ("trace_id", "text_instance_id", "id", "text_id"):
            value = source_text.get(key_name)
            if value is not None:
                key = f"{key_name}:{value}"
                break
        if key is None:
            key = f"bbox:{source_text.get('bbox')}:{source_text.get('text_pixel_bbox')}"
        if key in seen_candidate_keys:
            existing = candidate_by_key.get(key)
            if isinstance(existing, dict):
                existing_quality = _mask_evidence_quality(existing.get("mask_evidence"))
                source_quality = _mask_evidence_quality(source_text.get("mask_evidence"))
                source_has_better_mask = source_quality > existing_quality
                if source_has_better_mask:
                    for field in (
                        "mask_evidence",
                        "bubble_mask_source",
                        "bubbleMaskSource",
                        "bubble_mask_error",
                        "bubbleMaskError",
                        "bubble_mask_bbox",
                        "bubble_inner_bbox",
                        "bbox",
                        "source_bbox",
                        "text_pixel_bbox",
                        "line_polygons",
                        "balloon_bbox",
                        "background_rgb",
                        "route_action",
                        "route_reason",
                    ):
                        value = source_text.get(field)
                        if value not in (None, [], ""):
                            existing[field] = value
                for field in (
                    "text",
                    "translated",
                    "original",
                    "source_text",
                    "balloon_bbox",
                    "background_rgb",
                    "route_action",
                    "route_reason",
                    "layout_profile",
                    "block_profile",
                ):
                    value = source_text.get(field)
                    if value not in (None, [], "") and existing.get(field) in (None, [], ""):
                        existing[field] = value
                source_flags = [str(flag) for flag in source_text.get("qa_flags") or [] if str(flag).strip()]
                if source_flags:
                    merged_flags = list(existing.get("qa_flags") or [])
                    for flag in source_flags:
                        if flag not in merged_flags:
                            merged_flags.append(flag)
                    existing["qa_flags"] = merged_flags
            continue
        seen_candidate_keys.add(key)
        candidate_texts.append(source_text)
        candidate_by_key[key] = source_text

    auto_dark_card_fill = any(_auto_fast_dark_card_fill_allowed(text) for text in candidate_texts)
    if not _fast_dark_panel_fill_enabled() and not auto_dark_card_fill:
        rejection_reasons["disabled"] = max(
            1,
            len(candidate_texts),
        )
        return band_rgb, vision_blocks, _record(
            {"dark_panel_fill_count": 0, "remaining_blocks": len(vision_blocks)}
        )

    if not isinstance(band_rgb, np.ndarray) or band_rgb.size == 0 or not vision_blocks:
        return band_rgb, vision_blocks, _record(
            {"dark_panel_fill_count": 0, "remaining_blocks": len(vision_blocks)}
        )

    height, width = band_rgb.shape[:2]
    result = band_rgb.copy()
    filled_bboxes: list[list[int]] = []
    filled_keys: set[tuple[int, int, int, int]] = set()
    filled_text_keys: set[str] = set()
    filled_mask = np.zeros((height, width), dtype=np.uint8)
    global_fast_dark_enabled = _fast_dark_panel_fill_enabled()
    dark_bubble_siblings = [
        item
        for item in candidate_texts
        if isinstance(item, dict)
        and str(item.get("bubble_mask_source") or "").strip().lower() == "image_dark_bubble_mask"
    ]

    for text in candidate_texts:
        if not isinstance(text, dict):
            _reject("invalid_text")
            continue
        if _translator_note_text_only_mask(text):
            _reject("translator_note_text_mask_requires_real_inpaint")
            continue
        if not global_fast_dark_enabled and not _auto_fast_dark_card_fill_allowed(text):
            _reject("disabled")
            continue
        false_white_card_candidate = _is_false_white_card_candidate(band_rgb, text)
        visually_light_dark_bubble = _image_dark_bubble_is_visually_light(band_rgb, text)
        local_dark_bubble_candidate = bool(
            not visually_light_dark_bubble
            and
            str(text.get("bubble_mask_source") or "").strip().lower() == "image_dark_bubble_mask"
            and (text.get("balloon_bbox") or text.get("bubble_mask_bbox"))
            and (text.get("line_polygons") or text.get("text_pixel_bbox"))
        )
        if visually_light_dark_bubble:
            _reject("false_light_bubble_dark_fill")
            continue
        local_rejected_dark_card_candidate = _is_rejected_dark_card_fill_candidate(band_rgb, text)
        if _unsafe_white_balloon_requires_real_inpaint(text) and not false_white_card_candidate:
            _reject("unsafe_white_balloon_context")
            continue
        if _text_is_white_balloon_context(text) and not false_white_card_candidate:
            _reject("white_balloon_context")
            continue
        if (
            _text_has_no_glyph_evidence(text)
            and not _has_fast_fillable_text_mask_evidence(text)
            and not local_rejected_dark_card_candidate
        ):
            _reject("missing_glyph_evidence")
            continue
        mask_evidence_reason = _fast_fill_mask_evidence_rejection_reason(text)
        if mask_evidence_reason and not (
            false_white_card_candidate
            or local_dark_bubble_candidate
            or local_rejected_dark_card_candidate
        ):
            _propagate_existing_mask_evidence_decision_flags(ocr_page, text)
            _reject(mask_evidence_reason)
            continue
        qa_reason = _fast_fill_blocking_qa_reason(text)
        if qa_reason:
            if not false_white_card_candidate and not _can_use_local_card_fill_despite_blocking_qa(text, qa_reason):
                _reject(qa_reason)
                continue
        if _is_rotated_recovery_text(text):
            _reject("rotated_recovery_real_inpaint_required")
            continue
        if (
            _rejected_visual_card_requires_real_inpaint(band_rgb, text)
            and not _rejected_visual_card_allows_local_text_fill(text)
            and not local_rejected_dark_card_candidate
        ):
            _reject("rejected_visual_card_requires_real_inpaint")
            continue
        text_bbox = (
            _normalize_bbox(text.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(text.get("bbox"), width, height)
        )
        if text_bbox is None:
            _reject("missing_text_bbox")
            continue
        max_side = max(text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        pad = max(10, min(28, int(round(max_side * 0.08))))
        fill_bbox = _expanded_bbox(width, height, text_bbox, padding=pad) or text_bbox
        fill_key = tuple(fill_bbox)
        if fill_key in filled_keys:
            continue
        if str(text.get("bubble_mask_source") or "").strip().lower() == "image_dark_bubble_mask":
            text["_dark_bubble_sibling_texts"] = dark_bubble_siblings
        filled = _try_dark_panel_text_fill(result, text)
        text.pop("_dark_bubble_sibling_texts", None)
        if filled is None:
            _reject("not_solid_dark_panel")
            continue
        changed_mask = np.any(filled != result, axis=2).astype(np.uint8) * 255
        if not np.any(changed_mask):
            _reject("no_fast_fill_change")
            continue
        dark_fill_mask = _pop_text_fill_mask(text, "_dark_panel_fill_mask", changed_mask.shape[:2])
        if np.any(dark_fill_mask):
            changed_mask = np.maximum(changed_mask, dark_fill_mask).astype(np.uint8)
        visual_override = _dark_bubble_visual_fill_override(result, text, changed_mask)
        if isinstance(visual_override, np.ndarray) and np.any(visual_override):
            filled = np.where(visual_override[:, :, None] > 0, filled, result)
            changed_mask = visual_override
        elif _dark_fill_mask_is_overbroad_for_text(text, changed_mask):
            _reject("overbroad_dark_fill_mask")
            continue
        cleanup_bbox = _clipped_overlap_fragment_cleanup_bbox_for_text(text, width, height)
        if cleanup_bbox is not None:
            cx1, cy1, cx2, cy2 = cleanup_bbox
            cleanup_mask = _mask_from_bbox(width, height, cleanup_bbox, padding=0)
            if np.any(cleanup_mask):
                filled = filled.copy()
                filled[cy1:cy2, cx1:cx2] = np.array([0, 0, 0], dtype=np.uint8)
                changed_mask = np.maximum(changed_mask, cleanup_mask).astype(np.uint8)
                metrics = text.setdefault("qa_metrics", {})
                if isinstance(metrics, dict):
                    metrics["clipped_overlap_fragment_cleanup_fill"] = {
                        "bbox": cleanup_bbox,
                        "changed_pixels": int(np.count_nonzero(cleanup_mask)),
                        "source": "fast_dark_panel_text_fill",
                    }
                flags = list(ocr_page.get("_strip_inpaint_decision_flags") or [])
                if "clipped_overlap_fragment_cleanup_fill" not in flags:
                    flags.append("clipped_overlap_fragment_cleanup_fill")
                ocr_page["_strip_inpaint_decision_flags"] = flags
        if not any(
            _block_is_covered_by_fast_fill(block, [fill_bbox], width, height, changed_mask)
            for block in vision_blocks
        ):
            _reject("no_covered_vision_block")
            continue
        result = filled
        filled_mask = np.maximum(filled_mask, changed_mask)
        _accumulate_page_fill_mask(ocr_page, "_strip_dark_panel_fill_mask", changed_mask, changed_mask.shape[:2])
        filled_bboxes.append(fill_bbox)
        filled_keys.add(fill_key)
        for key_name in ("id", "text_id", "trace_id", "text_instance_id"):
            value = text.get(key_name)
            if value is not None:
                filled_text_keys.add(str(value))
        text["_fast_fill_inpaint_resolved"] = True

    if not filled_bboxes:
        return band_rgb, vision_blocks, _record(
            {"dark_panel_fill_count": 0, "remaining_blocks": len(vision_blocks)}
        )

    def _block_matches_filled_text(block: dict) -> bool:
        if not filled_text_keys or not isinstance(block, dict):
            return False
        for key_name in ("id", "text_id", "trace_id", "text_instance_id"):
            value = block.get(key_name)
            if value is not None and str(value) in filled_text_keys:
                return True
        return False

    def _block_allows_fast_fill_coverage_resolution(block: dict) -> bool:
        if not isinstance(block, dict):
            return False
        source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
        if source == "image_dark_bubble_mask":
            return False
        flags = {str(flag).strip().lower() for flag in block.get("qa_flags") or [] if str(flag).strip()}
        if any(flag.startswith("dark_bubble") for flag in flags):
            return False
        return True

    remaining_blocks = [
        block
        for block in vision_blocks
        if not _block_matches_filled_text(block)
        and not (
            _block_allows_fast_fill_coverage_resolution(block)
            and _block_is_covered_by_fast_fill(block, filled_bboxes, width, height, filled_mask)
        )
    ]
    for collection in (ocr_page.get("texts") or [], vision_blocks):
        for item in collection or []:
            if not isinstance(item, dict):
                continue
            if _block_matches_filled_text(item) or (
                _block_allows_fast_fill_coverage_resolution(item)
                and _block_is_covered_by_fast_fill(item, filled_bboxes, width, height, filled_mask)
            ):
                item["_fast_fill_inpaint_resolved"] = True
    ocr_page["_strip_used_dark_panel_fill"] = True
    previous_count = int(ocr_page.get("_strip_dark_panel_fill_count") or 0)
    ocr_page["_strip_dark_panel_fill_count"] = previous_count + len(filled_bboxes)
    return result, remaining_blocks, _record(
        {"dark_panel_fill_count": len(filled_bboxes), "remaining_blocks": len(remaining_blocks)}
    )


def _metadata_background_color(text: dict) -> np.ndarray | None:
    raw_color = text.get("background_rgb")
    if not isinstance(raw_color, (list, tuple)) or len(raw_color) != 3:
        return None
    try:
        color = np.asarray([int(round(float(v))) for v in raw_color], dtype=np.uint8)
    except Exception:
        return None
    luma = float(np.mean(color.astype(np.float32)))
    chroma = int(color.max()) - int(color.min())
    if luma >= 235.0 or luma <= 36.0:
        return color
    if chroma <= 4 and (luma >= 220.0 or luma <= 52.0):
        return color
    return None


def _metadata_background_is_stale_for_current_region(
    image_rgb: np.ndarray,
    text_mask: np.ndarray,
    color: np.ndarray,
) -> bool:
    metadata = _rgb_luma_chroma(color.tolist())
    if metadata is None:
        return False
    metadata_luma, metadata_chroma = metadata
    if metadata_luma < 235.0 or metadata_chroma > 10:
        return False
    if not isinstance(text_mask, np.ndarray) or text_mask.shape[:2] != image_rgb.shape[:2]:
        return False
    region = text_mask > 0
    if int(np.count_nonzero(region)) < 48:
        return False
    pixels = image_rgb[region].astype(np.float32)
    luma = (pixels[:, 0] * 0.299) + (pixels[:, 1] * 0.587) + (pixels[:, 2] * 0.114)
    median_luma = float(np.median(luma))
    dark_ratio = float(np.mean(luma <= 170.0))
    return median_luma <= 170.0 and dark_ratio >= 0.55


def _text_geometry_mask(width: int, height: int, text: dict) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=np.uint8)
    has_polygon = False
    raw_polygons = text.get("line_polygons")
    if isinstance(raw_polygons, list):
        for polygon in raw_polygons:
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                continue
            points: list[list[int]] = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    px = max(0, min(width - 1, int(round(float(point[0])))))
                    py = max(0, min(height - 1, int(round(float(point[1])))))
                except Exception:
                    continue
                points.append([px, py])
            if len(points) >= 3:
                cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
                has_polygon = True

    if not has_polygon:
        bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        if bbox is None:
            return None
        mask = _mask_from_bbox(width, height, bbox, padding=3)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if int(np.count_nonzero(mask)) < 24:
        return None
    return mask


def _strict_text_geometry_mask(width: int, height: int, text: dict) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=np.uint8)
    has_polygon = False
    raw_polygons = text.get("line_polygons") if isinstance(text, dict) else None
    if isinstance(raw_polygons, list):
        for polygon in raw_polygons:
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                continue
            points: list[list[int]] = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    px = max(0, min(width - 1, int(round(float(point[0])))))
                    py = max(0, min(height - 1, int(round(float(point[1])))))
                except Exception:
                    continue
                points.append([px, py])
            if len(points) >= 3:
                cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
                has_polygon = True
    bbox = (
        _normalize_bbox(text.get("source_text_anchor_bbox"), width, height)
        or _normalize_bbox(text.get("_source_text_anchor_bbox"), width, height)
        or _normalize_bbox(text.get("source_text_mask_bbox"), width, height)
        or _normalize_bbox(text.get("_source_text_mask_bbox"), width, height)
        or _trusted_text_bbox_for_contract(text, width, height)
        or _normalize_bbox(text.get("text_pixel_bbox"), width, height)
    )
    if has_polygon and bbox is not None and isinstance(text, dict):
        flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
        no_glyph_or_fragmented = bool(
            "fast_fill_no_glyph_evidence" in flags
            or "dark_connected_lobe_mask_rebuilt_from_glyphs" in flags
            or "dark_bubble_visual_glyph_mask_replaced_geometry" in flags
        )
        polygon_bbox = _bbox_from_binary_mask(mask)
        if no_glyph_or_fragmented and polygon_bbox is not None:
            trusted_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            polygon_area = max(1, (polygon_bbox[2] - polygon_bbox[0]) * (polygon_bbox[3] - polygon_bbox[1]))
            overlap = _bbox_overlap_ratio(polygon_bbox, bbox)
            if polygon_area < int(trusted_area * 0.72) or overlap < 0.55:
                replacement = _mask_from_bbox(width, height, bbox, padding=1)
                if isinstance(replacement, np.ndarray) and np.any(replacement):
                    metrics = text.setdefault("qa_metrics", {})
                    if isinstance(metrics, dict):
                        metrics["strict_text_geometry_polygon_replaced_by_text_bbox"] = {
                            "polygon_bbox": list(polygon_bbox),
                            "trusted_bbox": list(bbox),
                            "polygon_area": int(polygon_area),
                            "trusted_area": int(trusted_area),
                            "overlap": round(float(overlap), 6),
                        }
                    mask = replacement
    if not has_polygon:
        if bbox is None:
            return None
        mask = _mask_from_bbox(width, height, bbox, padding=1)
    if int(np.count_nonzero(mask)) < 12:
        return None
    return mask


def _authorized_fast_fill_mask(width: int, height: int, text: dict, padding: int = 1) -> np.ndarray | None:
    extra_padding = 0
    if isinstance(text, dict):
        try:
            extra_padding = int(text.get("_authorize_fast_fill_padding") or 0)
        except Exception:
            extra_padding = 0
    if isinstance(text, dict) and text.get("_authorize_fast_fill_by_balloon_bbox"):
        bbox = _normalize_bbox(text.get("balloon_bbox") or text.get("bubble_mask_bbox"), width, height)
        if bbox is None:
            return None
        mask = _mask_from_bbox(width, height, bbox, padding=max(18, int(padding), extra_padding))
        return mask if np.any(mask) else None
    if isinstance(text, dict) and text.get("_authorize_fast_fill_by_text_bbox"):
        bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height) or _normalize_bbox(
            text.get("bbox"),
            width,
            height,
        )
        if bbox is None:
            return None
        mask = _mask_from_bbox(width, height, bbox, padding=max(1, int(padding), extra_padding))
        return mask if np.any(mask) else None
    mask = _strict_text_geometry_mask(width, height, text)
    if mask is None or not np.any(mask):
        return None
    if padding > 1:
        kernel_size = max(3, min(47, int(padding) * 2 + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    if not np.any(mask):
        return None
    return mask


def _strip_inpaint_debug_dir(ocr_page: dict) -> Path | None:
    root = str(os.getenv("TRADUZAI_INPAINT_DEBUG_DIR", "") or "").strip()
    if not root:
        return None
    band_id = str(ocr_page.get("_band_id") or ocr_page.get("band_id") or "").strip()
    if re.fullmatch(r"page_\d+_band_\d+", band_id):
        debug_dir = Path(root) / band_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir
    try:
        page_number = int(ocr_page.get("_source_page_number") or ocr_page.get("numero") or 0)
    except Exception:
        page_number = 0
    try:
        band_index = int(ocr_page.get("_band_index") or 0)
    except Exception:
        band_index = 0
    debug_dir = Path(root) / f"page_{page_number:03d}_band_{band_index:03d}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    Image.fromarray(image_rgb.astype(np.uint8)).save(path, quality=92)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), mask.astype(np.uint8))


def _mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, blocks: list[dict]) -> np.ndarray:
    overlay = image_rgb.copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    active = mask > 0
    overlay[active] = (overlay[active].astype(np.float32) * 0.45 + red[active].astype(np.float32) * 0.55).astype(np.uint8)
    for index, block in enumerate(blocks, start=1):
        bbox = _normalize_bbox(block.get("bbox"), image_rgb.shape[1], image_rgb.shape[0])
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 220, 255), 1)
        cv2.putText(overlay, str(index), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 220, 255), 1, cv2.LINE_AA)
    return overlay


def _active_debug_recorder():
    try:
        from debug_tools import get_recorder
    except Exception:
        return None
    recorder = get_recorder()
    if recorder is not None and getattr(recorder, "enabled", False):
        return recorder
    return None


def _strip_band_id(ocr_page: dict) -> str:
    raw_band_id = ocr_page.get("_band_id") or ocr_page.get("band_id")
    if raw_band_id:
        return str(raw_band_id)
    try:
        page_number = int(ocr_page.get("_source_page_number") or ocr_page.get("numero") or 0)
    except Exception:
        page_number = 0
    try:
        band_index = int(ocr_page.get("_band_index") or 0)
    except Exception:
        band_index = 0
    return f"page_{page_number:03d}_band_{band_index:03d}"


def _strip_page_id(ocr_page: dict) -> str | None:
    raw_page_id = ocr_page.get("_page_id") or ocr_page.get("page_id")
    if raw_page_id:
        return str(raw_page_id)
    try:
        page_number = int(ocr_page.get("_source_page_number") or ocr_page.get("numero") or 0)
    except Exception:
        page_number = 0
    if page_number <= 0:
        return None
    return f"page_{page_number:03d}"


def _trace_id_for(text_id: str, band_id: str) -> str:
    return f"{text_id}@{band_id}" if band_id else text_id


def _strip_text_ids(ocr_page: dict) -> list[str]:
    text_ids: list[str] = []
    for index, text in enumerate(ocr_page.get("texts", []), start=1):
        if not isinstance(text, dict):
            continue
        raw_id = text.get("text_id") or text.get("id") or text.get("_id")
        text_ids.append(str(raw_id or f"text_{index:03d}"))
    return text_ids


def _strip_trace_ids(ocr_page: dict) -> list[str]:
    band_id = _strip_band_id(ocr_page)
    trace_ids: list[str] = []
    for index, text in enumerate(ocr_page.get("texts", []), start=1):
        if not isinstance(text, dict):
            continue
        text_id = str(text.get("text_id") or text.get("id") or text.get("_id") or f"text_{index:03d}")
        trace_id = str(text.get("trace_id") or _trace_id_for(text_id, band_id))
        if trace_id and trace_id not in trace_ids:
            trace_ids.append(trace_id)
    for index, block in enumerate(ocr_page.get("_vision_blocks", []), start=1):
        if not isinstance(block, dict):
            continue
        raw_text_id = block.get("text_id") or block.get("id")
        if not raw_text_id:
            continue
        text_id = str(raw_text_id)
        trace_id = str(block.get("trace_id") or _trace_id_for(text_id, str(block.get("band_id") or band_id)))
        if trace_id and trace_id not in trace_ids:
            trace_ids.append(trace_id)
    return trace_ids


def _processable_texts_for_inpaint(ocr_page: dict) -> list[dict]:
    return [
        text
        for text in ocr_page.get("texts", [])
        if isinstance(text, dict)
        and not _text_suppressed_for_inpaint(text)
        and not _route_action_blocks_inpaint(text)
        and (not _translator_note_text_only_mask(text) or _translator_note_has_text_geometry(text))
        and not _text_has_rejected_bubble_without_glyph_evidence(text)
    ]


def _translator_note_text_only_mask(item: dict | None) -> bool:
    if not isinstance(item, dict):
        return False
    source = str(item.get("bubble_mask_source") or item.get("bubbleMaskSource") or "").strip().lower()
    flags = {str(flag).strip() for flag in item.get("qa_flags") or [] if str(flag).strip()}
    return bool(source == "translator_note_text_mask" or "translator_note_text_only_mask" in flags)


def _vision_block_requires_inpaint(block: dict) -> bool:
    if not isinstance(block, dict):
        return False
    if _text_suppressed_for_inpaint(block):
        return False
    if _translator_note_text_only_mask(block) and not _translator_note_has_text_geometry(block):
        return False
    if _text_has_rejected_bubble_without_glyph_evidence(block):
        return False
    route_action = str(block.get("route_action") or "").strip().lower()
    if route_action in ROUTE_ACTIONS:
        if route_action == "review_required" and _item_has_current_inpaint_mask_evidence(block):
            return True
        return route_action_requires_inpaint(route_action)
    return True


def _processable_vision_blocks_for_inpaint(blocks: list[dict]) -> list[dict]:
    return [block for block in blocks if _vision_block_requires_inpaint(block)]


def _qa_flags_for_auto_inpaint(item: dict | None) -> set[str]:
    if not isinstance(item, dict):
        return set()
    return {str(flag).strip() for flag in item.get("qa_flags") or [] if str(flag).strip()}


def _has_current_image_bubble_mask_evidence(item: dict | None) -> bool:
    return _item_has_current_inpaint_mask_evidence(item)


def _unsafe_source_is_covered_by_clean_current_block(source: dict, blocks: list[dict] | None) -> bool:
    if not isinstance(source, dict) or not blocks:
        return False
    source_bbox = _normalize_bbox(source.get("bbox"), 10**9, 10**9)
    if source_bbox is None:
        return False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if not _item_has_current_inpaint_mask_evidence(block):
            continue
        if _auto_inpaint_unsafe_reason(block):
            continue
        block_bbox = _normalize_bbox(block.get("bbox"), 10**9, 10**9)
        if block_bbox is None:
            continue
        if (
            _bbox_overlap_ratio(source_bbox, block_bbox) >= 0.20
            or _bbox_center_inside(source_bbox, block_bbox)
            or _bbox_center_inside(block_bbox, source_bbox)
        ):
            return True
    return False


def _source_matches_active_inpaint_block(source: dict, blocks: list[dict] | None) -> bool:
    if not isinstance(source, dict) or blocks is None:
        return True
    source_keys = {
        str(source.get(key) or "").strip()
        for key in ("trace_id", "text_id", "id", "text_instance_id")
        if str(source.get(key) or "").strip()
    }
    source_bbox = _normalize_bbox(
        source.get("text_pixel_bbox") or source.get("bbox"),
        10**9,
        10**9,
    )
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_keys = {
            str(block.get(key) or "").strip()
            for key in ("trace_id", "text_id", "id", "text_instance_id")
            if str(block.get(key) or "").strip()
        }
        if source_keys and block_keys and source_keys & block_keys:
            return True
        block_bbox = _normalize_bbox(
            block.get("text_pixel_bbox") or block.get("bbox"),
            10**9,
            10**9,
        )
        if source_bbox is not None and block_bbox is not None:
            if _bbox_overlap_ratio(source_bbox, block_bbox) >= 0.20:
                return True
            if _bbox_center_inside(source_bbox, block_bbox) or _bbox_center_inside(block_bbox, source_bbox):
                return True
    return False


def _auto_inpaint_unsafe_reason(item: dict | None) -> str:
    flags = _qa_flags_for_auto_inpaint(item)
    has_current_mask_evidence = _has_current_image_bubble_mask_evidence(item)
    if has_current_mask_evidence:
        flags = {flag for flag in flags if flag != "mask_outside_balloon"}
    if _current_contour_mask_evidence_clears_outside_critical(item):
        flags = {flag for flag in flags if flag != "mask_outside_balloon_critical"}
    if _current_dark_panel_mask_evidence_clears_outside_critical(item):
        flags = {
            flag
            for flag in flags
            if flag not in {"mask_outside_balloon", "mask_outside_balloon_critical"}
        }
    if _translator_note_has_current_text_mask_evidence(item):
        flags = {
            flag
            for flag in flags
            if flag not in {"mask_outside_balloon", "mask_outside_balloon_critical"}
        }
    blocked = sorted(flags & _UNSAFE_AUTO_INPAINT_QA_FLAGS)
    if blocked:
        return blocked[0]
    if not isinstance(item, dict):
        return ""
    source = str(item.get("bubble_mask_source") or item.get("bubbleMaskSource") or "").strip().lower()
    error = str(item.get("bubble_mask_error") or item.get("bubbleMaskError") or "").strip().lower()
    card_panel_text_context = bool(item.get("card_panel_text_context")) or _is_dark_or_colored_card_text(item)
    if (
        source in _REJECTED_BUBBLE_MASK_SOURCES
        and _has_fast_fillable_text_mask_evidence(item)
        and error == "derived_mask_not_anchored_to_text"
    ):
        return ""
    if source in _REJECTED_BUBBLE_MASK_SOURCES and not has_current_mask_evidence:
        return source
    if error in {"derived_mask_not_anchored_to_text", "rejected_rectangular_crop"} and not has_current_mask_evidence:
        return error
    weak_mask_error = _weak_image_bubble_mask_error_for_item(item)
    if weak_mask_error:
        return weak_mask_error
    return ""


def _text_lookup_key(item: dict) -> str:
    for key in ("trace_id", "text_id", "id", "_id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    bbox = _normalize_bbox(item.get("bbox"), 10**9, 10**9)
    return ",".join(str(v) for v in bbox) if bbox is not None else ""


def _unsafe_text_reason_for_block_by_geometry(block: dict, texts: list[dict]) -> str:
    block_bbox = _normalize_bbox(block.get("bbox"), 10**9, 10**9)
    if block_bbox is None:
        return ""
    best_reason = ""
    best_score = 0.0
    center_reason = ""
    for text in texts:
        if not isinstance(text, dict):
            continue
        reason = _auto_inpaint_unsafe_reason(text)
        if not reason:
            continue
        text_bbox = _normalize_bbox(text.get("bbox"), 10**9, 10**9)
        if text_bbox is None:
            continue
        score = _bbox_overlap_ratio(block_bbox, text_bbox)
        if score > best_score:
            best_score = score
            best_reason = reason
        if not center_reason and (
            _bbox_center_inside(block_bbox, text_bbox) or _bbox_center_inside(text_bbox, block_bbox)
        ):
            center_reason = reason
    if best_score >= 0.20:
        return best_reason
    if center_reason:
        return center_reason
    return ""


def _filter_unsafe_auto_inpaint_blocks(
    ocr_page: dict,
    blocks: list[dict],
    band_rgb: np.ndarray | None = None,
) -> list[dict]:
    if not blocks:
        return []
    text_reasons: dict[str, str] = {}
    unsafe_texts: list[dict] = []
    for text in ocr_page.get("texts", []) if isinstance(ocr_page, dict) else []:
        if not isinstance(text, dict):
            continue
        reason = _auto_inpaint_unsafe_reason(text)
        key = _text_lookup_key(text)
        if reason and key:
            text_reasons[key] = reason
        if reason:
            unsafe_texts.append(text)

    safe_blocks: list[dict] = []
    skipped = 0
    reasons: dict[str, int] = {}
    samples: list[dict] = []

    def _sample_block(block: dict, *, reason: str, rebuilt_pixels: int | None = None) -> dict:
        return {
            "id": block.get("id") or block.get("text_id"),
            "trace_id": block.get("trace_id"),
            "band_id": block.get("band_id"),
            "reason": reason,
            "rebuilt_mask_pixels": rebuilt_pixels,
            "bbox": block.get("bbox"),
            "text_pixel_bbox": block.get("text_pixel_bbox"),
            "balloon_bbox": block.get("balloon_bbox"),
            "bubble_mask_bbox": block.get("bubble_mask_bbox"),
            "bubble_mask_source": block.get("bubble_mask_source"),
            "qa_flags": list(block.get("qa_flags") or []),
            "route_action": block.get("route_action"),
            "mask_evidence": block.get("mask_evidence"),
            "_band_local_bbox_normalized": block.get("_band_local_bbox_normalized"),
        }

    def _copy_fast_fillable_text_evidence(block: dict) -> bool:
        block_key = _text_lookup_key(block)
        block_bbox = _normalize_bbox(block.get("text_pixel_bbox") or block.get("bbox"), 10**9, 10**9)
        for text in ocr_page.get("texts", []) if isinstance(ocr_page, dict) else []:
            if not isinstance(text, dict):
                continue
            text_key = _text_lookup_key(text)
            text_bbox = _normalize_bbox(text.get("text_pixel_bbox") or text.get("bbox"), 10**9, 10**9)
            same_identity = bool(block_key and text_key and block_key == text_key)
            same_geometry = bool(
                block_bbox is not None
                and text_bbox is not None
                and (
                    _bbox_overlap_ratio(block_bbox, text_bbox) >= 0.35
                    or _bbox_center_inside(block_bbox, text_bbox)
                    or _bbox_center_inside(text_bbox, block_bbox)
                )
            )
            if not (same_identity or same_geometry):
                continue
            if not (_has_fast_fillable_text_mask_evidence(text) and _auto_fast_dark_card_fill_allowed(text)):
                continue
            for key in (
                "mask_evidence",
                "background_rgb",
                "line_polygons",
                "text_pixel_bbox",
                "balloon_bbox",
                "bubble_mask_source",
                "bubbleMaskSource",
                "bubble_mask_error",
                "bubbleMaskError",
                "bubble_mask_bbox",
                "bubbleMaskBbox",
                "bubble_inner_bbox",
                "bubbleInnerBbox",
                "card_panel_text_context",
                "route_action",
                "route_reason",
                "qa_flags",
            ):
                value = text.get(key)
                if value not in (None, [], ""):
                    block[key] = copy.deepcopy(value)
            return True
        return False

    def _copy_matching_dark_panel_context(block: dict) -> bool:
        block_key = _text_lookup_key(block)
        block_bbox = _normalize_bbox(block.get("text_pixel_bbox") or block.get("bbox"), 10**9, 10**9)
        for text in ocr_page.get("texts", []) if isinstance(ocr_page, dict) else []:
            if not isinstance(text, dict):
                continue
            source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
            if source not in {"image_dark_panel_mask", "image_dark_bubble_mask", "derived_card_panel_mask"}:
                continue
            if not _current_dark_panel_mask_evidence_clears_outside_critical(text):
                continue
            text_key = _text_lookup_key(text)
            text_bbox = _normalize_bbox(text.get("text_pixel_bbox") or text.get("bbox"), 10**9, 10**9)
            same_identity = bool(block_key and text_key and block_key == text_key)
            same_geometry = bool(
                block_bbox is not None
                and text_bbox is not None
                and (
                    _bbox_overlap_ratio(block_bbox, text_bbox) >= 0.35
                    or _bbox_center_inside(block_bbox, text_bbox)
                    or _bbox_center_inside(text_bbox, block_bbox)
                )
            )
            if not (same_identity or same_geometry):
                continue
            for key in (
                "mask_evidence",
                "background_rgb",
                "line_polygons",
                "text_pixel_bbox",
                "balloon_bbox",
                "bubble_mask_source",
                "bubbleMaskSource",
                "bubble_mask_error",
                "bubbleMaskError",
                "bubble_mask_bbox",
                "bubbleMaskBbox",
                "bubble_inner_bbox",
                "bubbleInnerBbox",
                "card_panel_text_context",
                "block_profile",
                "layout_profile",
                "background_type",
                "route_action",
                "route_reason",
                "qa_flags",
            ):
                value = text.get(key)
                if value not in (None, [], ""):
                    block[key] = copy.deepcopy(value)
            return True
        return False

    def _try_rebuild_mask_evidence(block: dict, reason: str) -> tuple[str, int | None]:
        rebuilt_pixels: int | None = None
        if not reason:
            return reason, rebuilt_pixels
        if _copy_matching_dark_panel_context(block):
            reason = _auto_inpaint_unsafe_reason(block)
            if not reason:
                return "", rebuilt_pixels
        if _copy_fast_fillable_text_evidence(block):
            return "", rebuilt_pixels
        if _auto_fast_dark_card_fill_allowed(block) and _has_fast_fillable_text_mask_evidence(block):
            return "", rebuilt_pixels
        if _item_has_current_inpaint_mask_evidence(block):
            return _auto_inpaint_unsafe_reason(block), rebuilt_pixels
        if not isinstance(band_rgb, np.ndarray):
            return reason, rebuilt_pixels
        try:
            rebuilt_mask = build_inpaint_mask(block, band_rgb.shape, band_rgb)
        except Exception:
            rebuilt_mask = None
        if isinstance(rebuilt_mask, np.ndarray):
            rebuilt_pixels = int(np.count_nonzero(rebuilt_mask))
            if rebuilt_pixels > 0:
                reason = _auto_inpaint_unsafe_reason(block)
                if reason and _item_has_current_inpaint_mask_evidence(block):
                    reason = ""
        return reason, rebuilt_pixels

    def _stale_anchor_reason_cleared_by_current_card_mask(block: dict, reason: str) -> bool:
        if reason != "derived_mask_not_anchored_to_text":
            return False
        source = str(block.get("bubble_mask_source") or "").strip().lower()
        if source not in {"derived_card_panel_mask", "image_dark_panel_mask", "image_dark_bubble_mask"}:
            return False
        mask_bbox = _normalize_bbox(block.get("bubble_mask_bbox"), 10**9, 10**9)
        text_bbox = _normalize_bbox(block.get("text_pixel_bbox") or block.get("bbox"), 10**9, 10**9)
        if mask_bbox is None or text_bbox is None:
            return False
        return _bbox_overlap_ratio(text_bbox, mask_bbox) >= 0.72 or _bbox_center_inside(text_bbox, mask_bbox)

    for block in blocks:
        _copy_matching_dark_panel_context(block)
        reason = _auto_inpaint_unsafe_reason(block)
        rebuilt_pixels: int | None = None
        reason, rebuilt_pixels = _try_rebuild_mask_evidence(block, reason)
        if _stale_anchor_reason_cleared_by_current_card_mask(block, reason):
            reason = ""
        if not reason:
            key = _text_lookup_key(block)
            reason = text_reasons.get(key, "")
            if reason:
                reason, rebuilt_pixels = _try_rebuild_mask_evidence(block, reason)
                if _stale_anchor_reason_cleared_by_current_card_mask(block, reason):
                    reason = ""
                if reason and _item_has_current_inpaint_mask_evidence(block):
                    reason = ""
        if not reason and unsafe_texts:
            reason = _unsafe_text_reason_for_block_by_geometry(block, unsafe_texts)
            if reason:
                reason, rebuilt_pixels = _try_rebuild_mask_evidence(block, reason)
                if _stale_anchor_reason_cleared_by_current_card_mask(block, reason):
                    reason = ""
                if reason and _item_has_current_inpaint_mask_evidence(block):
                    reason = ""
        if reason:
            skipped += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            if len(samples) < 8:
                samples.append(_sample_block(block, reason=reason, rebuilt_pixels=rebuilt_pixels))
            continue
        safe_blocks.append(block)
    if skipped:
        ocr_page["_strip_unsafe_inpaint_block_count"] = int(skipped)
        ocr_page["_strip_unsafe_inpaint_block_reasons"] = reasons
        ocr_page["_strip_unsafe_inpaint_block_samples"] = samples
        if not safe_blocks:
            _append_inpaint_decision_flag(ocr_page, "real_inpaint_skipped_unsafe_mask")
    return safe_blocks


def _ocr_page_has_unsafe_auto_inpaint_evidence(
    ocr_page: dict | None,
    active_blocks: list[dict] | None = None,
) -> bool:
    if not isinstance(ocr_page, dict):
        return False
    sources = [
        source
        for source in list(ocr_page.get("texts") or []) + list(ocr_page.get("_vision_blocks") or [])
        if isinstance(source, dict)
    ]
    source_has_current_evidence = any(_item_has_current_inpaint_mask_evidence(source) for source in sources)
    for source in sources:
        if _auto_inpaint_unsafe_reason(source):
            if active_blocks is not None and not _source_matches_active_inpaint_block(source, active_blocks):
                continue
            if _unsafe_source_is_covered_by_clean_current_block(source, active_blocks):
                continue
            return True
    decision_flags = {
        str(flag).strip()
        for flag in ocr_page.get("_strip_inpaint_decision_flags", [])
        if str(flag).strip()
    }
    if decision_flags & _UNSAFE_AUTO_INPAINT_QA_FLAGS and not source_has_current_evidence:
        return True
    return False


def _mask_exceeds_bubble_limit(
    ocr_page: dict,
    mask: np.ndarray | None,
    shape: tuple[int, int],
) -> tuple[bool, int, float]:
    action = _coerce_mask_for_shape(mask, shape)
    action_pixels = int(np.count_nonzero(action))
    if action_pixels <= 0:
        return False, 0, 0.0
    height, width = shape
    bubble_union = np.zeros(shape, dtype=np.uint8)
    for text in _processable_texts_for_inpaint(ocr_page):
        bubble_mask, _reason = _real_bubble_mask_for_text(ocr_page, text, width, height)
        if bubble_mask is None or not np.any(bubble_mask):
            continue
        bubble_union = np.maximum(bubble_union, _coerce_mask_for_shape(bubble_mask, shape))
    if not np.any(bubble_union):
        return False, 0, 0.0
    outside_pixels = int(np.count_nonzero((action > 0) & (bubble_union == 0)))
    outside_ratio = outside_pixels / float(max(1, action_pixels))
    critical = outside_pixels > OUTSIDE_BALLOON_CRITICAL_PIXELS and outside_ratio >= OUTSIDE_BALLOON_CRITICAL_RATIO
    return critical, outside_pixels, outside_ratio


def _rejected_card_action_mask_allows_real_inpaint(
    ocr_page: dict,
    active_blocks: list[dict] | None,
    action_mask: np.ndarray | None,
    shape: tuple[int, int],
) -> bool:
    action = _coerce_mask_for_shape(action_mask, shape)
    action_pixels = int(np.count_nonzero(action))
    if action_pixels <= 0:
        return False
    if action_pixels / float(max(1, int(shape[0]) * int(shape[1]))) > 0.12:
        return False
    blocks = [block for block in (active_blocks or []) if isinstance(block, dict)]
    height, width = shape
    texts = [text for text in list(ocr_page.get("texts") or []) if isinstance(text, dict)]
    candidates = blocks or texts
    if not candidates:
        return False
    matched = 0
    for block in candidates:
        text = _find_mask_evidence_text_for_block(block, texts, width, height)
        merged = dict(text or {})
        merged.update(block)
        source = str(
            merged.get("bubble_mask_source")
            or merged.get("bubbleMaskSource")
            or ""
        ).strip().lower()
        error = str(
            merged.get("bubble_mask_error")
            or merged.get("bubbleMaskError")
            or ""
        ).strip().lower()
        if source not in _REJECTED_BUBBLE_MASK_SOURCES or error != "derived_mask_not_anchored_to_text":
            return False
        has_text_mask_evidence = bool(
            isinstance(merged.get("mask_evidence"), dict)
            and int((merged.get("mask_evidence") or {}).get("raw_mask_pixels") or 0) > 0
        )
        if not (
            merged.get("card_panel_text_context")
            or _is_dark_or_colored_card_text(merged)
            or has_text_mask_evidence
        ):
            return False
        if _text_suppressed_for_inpaint(merged) or _route_action_blocks_inpaint(merged):
            return False
        geometry = _text_geometry_mask(width, height, merged)
        if geometry is None or not np.any(geometry):
            bbox = _normalize_bbox(merged.get("text_pixel_bbox") or merged.get("bbox"), width, height)
            if bbox is None:
                return False
            geometry = _mask_from_bbox(width, height, bbox, padding=2)
        geometry_pixels = int(np.count_nonzero(geometry))
        if geometry_pixels <= 0:
            return False
        overlap_pixels = int(np.count_nonzero((geometry > 0) & (action > 0)))
        if overlap_pixels < max(24, int(round(geometry_pixels * 0.25))):
            return False
        matched += 1
    return matched > 0


def _clear_stale_unsafe_inpaint_flags_for_current_action(ocr_page: dict, active_blocks: list[dict] | None) -> None:
    flags_to_remove = {"mask_outside_balloon", "mask_outside_balloon_critical", "real_inpaint_skipped_unsafe_mask"}
    for collection in (ocr_page.get("texts") or [], ocr_page.get("_vision_blocks") or [], active_blocks or []):
        for item in collection if isinstance(collection, list) else []:
            if isinstance(item, dict):
                _remove_qa_flags_from_item(item, flags_to_remove)
    decision_flags = [
        str(flag)
        for flag in ocr_page.get("_strip_inpaint_decision_flags") or []
        if str(flag).strip() not in flags_to_remove
    ]
    if decision_flags:
        ocr_page["_strip_inpaint_decision_flags"] = decision_flags
    else:
        ocr_page.pop("_strip_inpaint_decision_flags", None)


def _vision_block_for_real_inpaint_payload(block: dict) -> dict:
    payload = dict(block)
    payload.pop("mask_evidence", None)
    payload.pop("validated_by_segment_mask", None)
    runtime_flags = {
        "mask_outside_balloon",
        "mask_outside_balloon_critical",
        "fast_fill_no_glyph_evidence",
    }
    flags = [str(flag) for flag in payload.get("qa_flags") or [] if str(flag).strip()]
    remaining_flags = [flag for flag in flags if flag not in runtime_flags]
    if remaining_flags:
        payload["qa_flags"] = remaining_flags
    else:
        payload.pop("qa_flags", None)
    return payload


def _all_processable_texts_are_white_balloon(texts: list[dict], image_rgb: np.ndarray | None = None) -> bool:
    if not texts:
        return False
    for text in texts:
        if image_rgb is not None:
            if not _text_region_looks_plain_white(image_rgb, text):
                return False
            continue
        color = _metadata_background_color(text)
        if color is None:
            return False
        color_f = color.astype(np.float32)
        luma = float(np.mean(color_f))
        chroma = float(np.max(color_f) - np.min(color_f))
        if luma < 228.0 or chroma > 18.0:
            return False
    return True


def _build_residual_text_region_mask(ocr_page: dict | None, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(ocr_page, dict):
        return mask
    for text in _processable_texts_for_inpaint(ocr_page):
        geometry_mask = None
        if text.get("line_polygons"):
            try:
                geometry_mask = mask_from_text_geometry(text, (height, width))
            except Exception:
                geometry_mask = None
        if geometry_mask is not None and np.any(geometry_mask):
            geometry_mask = cv2.dilate(
                geometry_mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            mask = np.maximum(mask, geometry_mask.astype(np.uint8))
            continue
        bbox = _line_polygons_bbox(text, width, height) or _normalize_bbox(
            text.get("text_pixel_bbox"),
            width,
            height,
        )
        if bbox is None:
            continue
        mask = np.maximum(mask, _mask_from_bbox(width, height, bbox, padding=3))
    return mask


def _coerce_mask_for_shape(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape != shape:
        return np.zeros(shape, dtype=np.uint8)
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _mask_band_density(mask: np.ndarray | None, shape: tuple[int, int]) -> float:
    coerced = _coerce_mask_for_shape(mask, shape)
    return int(np.count_nonzero(coerced)) / float(max(1, int(shape[0]) * int(shape[1])))


def _density_guarded_inpaint_mask(
    raw_mask: np.ndarray | None,
    expanded_mask: np.ndarray | None,
    shape: tuple[int, int],
    *,
    threshold: float = 0.12,
    erode_px: int = 2,
) -> tuple[np.ndarray, str]:
    expanded = _coerce_mask_for_shape(expanded_mask, shape)
    raw = _coerce_mask_for_shape(raw_mask, shape)
    if not np.any(expanded):
        return raw, "raw_mask" if np.any(raw) else "empty"
    if _mask_band_density(expanded, shape) <= threshold:
        return expanded, "expanded_mask"
    if np.any(raw) and _mask_band_density(raw, shape) <= threshold:
        return raw, "raw_mask"
    if erode_px > 0:
        kernel_size = int(erode_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(expanded.astype(np.uint8), kernel, iterations=1)
        if np.any(eroded) and int(np.count_nonzero(eroded)) < int(np.count_nonzero(expanded)):
            return eroded.astype(np.uint8), "eroded_expanded_mask"
    if np.any(raw):
        return raw, "raw_mask"
    return expanded, "expanded_mask"


def _dilate_mask_asymmetric(
    mask: np.ndarray,
    *,
    left_px: int = 4,
    right_px: int = 4,
    up_px: int = 2,
    down_px: int = 8,
) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        if isinstance(mask, np.ndarray):
            return np.zeros(mask.shape[:2], dtype=np.uint8)
        return np.zeros((0, 0), dtype=np.uint8)
    left_px = max(0, int(left_px))
    right_px = max(0, int(right_px))
    up_px = max(0, int(up_px))
    down_px = max(0, int(down_px))
    if left_px == right_px == up_px == down_px == 0:
        return np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel = np.ones((up_px + down_px + 1, left_px + right_px + 1), dtype=np.uint8)
    return cv2.dilate(
        np.where(mask > 0, 255, 0).astype(np.uint8),
        kernel,
        anchor=(right_px, down_px),
        iterations=1,
    ).astype(np.uint8)


def _strip_real_inpaint_expansion_limit_mask(
    ocr_page: dict,
    texts: list[dict],
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    limit = np.zeros((height, width), dtype=np.uint8)
    if height <= 0 or width <= 0:
        return limit
    for text in texts:
        if not isinstance(text, dict) or _route_action_blocks_inpaint(text):
            continue
        if _text_has_rejected_bubble_mask_source(text):
            continue
        bbox = _text_cleanup_limit_bbox(text, width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        box_h = max(1, int(y2) - int(y1))
        pad_x = max(4, min(14, int(round((int(x2) - int(x1)) * 0.08))))
        pad_top = max(2, min(6, int(round(box_h * 0.16))))
        pad_bottom = max(8, min(20, int(round(box_h * 0.55))))
        text_limit = _mask_from_bbox(
            width,
            height,
            [
                max(0, int(x1) - pad_x),
                max(0, int(y1) - pad_top),
                min(width, int(x2) + pad_x),
                min(height, int(y2) + pad_bottom),
            ],
            padding=0,
        )
        real_bubble_mask, _ = _real_bubble_mask_for_text(ocr_page, text, width, height)
        if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask):
            safe_bubble = _safe_real_bubble_interior_mask(real_bubble_mask, width, height, erode_px=2)
            if np.any(safe_bubble):
                text_limit = cv2.bitwise_and(text_limit.astype(np.uint8), safe_bubble.astype(np.uint8))
        limit = np.maximum(limit, text_limit.astype(np.uint8))
    return limit.astype(np.uint8)


def _strip_real_inpaint_raw_component_recovery_mask(
    raw_mask: np.ndarray,
    ocr_page: dict,
    texts: list[dict],
    shape: tuple[int, int],
) -> np.ndarray:
    """Recover glyph-like raw components that tight text_pixel_bbox limits miss."""

    height, width = shape
    raw = _coerce_mask_for_shape(raw_mask, shape)
    recovered = np.zeros((height, width), dtype=np.uint8)
    if height <= 0 or width <= 0 or not np.any(raw):
        return recovered

    for text in texts:
        if not isinstance(text, dict) or _route_action_blocks_inpaint(text):
            continue
        if _text_has_rejected_bubble_mask_source(text):
            continue
        support_bbox = _text_source_limit_bbox(text, width, height, include_line_bbox=True)
        if support_bbox is None:
            continue
        sx1, sy1, sx2, sy2 = support_bbox
        support_w = max(1, int(sx2) - int(sx1))
        support_h = max(1, int(sy2) - int(sy1))
        pad_x = max(6, min(14, int(round(support_w * 0.06))))
        pad_y = max(3, min(8, int(round(support_h * 0.12))))
        support_bbox = _normalize_bbox(
            [
                int(sx1) - pad_x,
                int(sy1) - pad_y,
                int(sx2) + pad_x,
                int(sy2) + pad_y,
            ],
            width,
            height,
        )
        if support_bbox is None:
            continue
        support = _mask_from_bbox(width, height, support_bbox, padding=0)
        bubble_bbox = None
        real_bubble_mask, _ = _real_bubble_mask_for_text(ocr_page, text, width, height)
        if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask):
            bubble_bbox = (
                _normalize_bbox(text.get("bubble_mask_bbox"), width, height)
                or _normalize_bbox(text.get("balloon_bbox"), width, height)
                or _bbox_from_binary_mask(real_bubble_mask)
            )
            safe_bubble = _safe_real_bubble_interior_mask(real_bubble_mask, width, height, erode_px=4)
            if np.any(safe_bubble):
                raw_in_support = cv2.bitwise_and(raw.astype(np.uint8), support.astype(np.uint8))
                raw_pixels = int(np.count_nonzero(raw_in_support))
                safe_support = cv2.bitwise_and(support.astype(np.uint8), safe_bubble.astype(np.uint8))
                safe_raw_pixels = int(np.count_nonzero(cv2.bitwise_and(raw_in_support, safe_bubble.astype(np.uint8))))
                if raw_pixels <= 0 or safe_raw_pixels / float(max(1, raw_pixels)) >= 0.85:
                    support = safe_support
                elif bubble_bbox is not None:
                    bubble_box_mask = _mask_from_bbox(width, height, bubble_bbox, padding=0)
                    support = cv2.bitwise_and(support.astype(np.uint8), bubble_box_mask.astype(np.uint8))
                else:
                    support = safe_support
        if not np.any(support):
            continue

        candidate = cv2.bitwise_and(raw.astype(np.uint8), support.astype(np.uint8))
        if not np.any(candidate):
            continue
        support_pixels = int(np.count_nonzero(support))
        max_component_area = max(96, min(3500, int(round(support_pixels * 0.35))))
        labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (candidate > 0).astype(np.uint8),
            connectivity=8,
        )
        for label in range(1, int(labels_count)):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 0 or area > max_component_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if bubble_bbox is not None:
                bx1, by1, bx2, by2 = [int(v) for v in bubble_bbox]
                min_edge_gap = max(6, min(12, int(round(min(max(1, bx2 - bx1), max(1, by2 - by1)) * 0.08))))
                near_bubble_edge = (
                    x - bx1 < min_edge_gap
                    or y - by1 < min_edge_gap
                    or bx2 - (x + w) < min_edge_gap
                    or by2 - (y + h) < min_edge_gap
                )
                component_aspect = float(w) / float(max(1, h))
                if near_bubble_edge and component_aspect < 0.90:
                    continue
            cx, cy = centroids[label]
            ix, iy = int(round(cx)), int(round(cy))
            if iy < 0 or iy >= height or ix < 0 or ix >= width or support[iy, ix] == 0:
                continue
            component = labels == label
            recovered[component] = 255
    return recovered.astype(np.uint8)


def _strip_real_inpaint_expanded_evidence_floor_mask(
    raw_mask: np.ndarray,
    expanded_mask: np.ndarray,
    ocr_page: dict,
    texts: list[dict],
    shape: tuple[int, int],
) -> np.ndarray:
    """Keep debug-approved expansion pixels near glyphs after geometry clipping."""

    height, width = shape
    raw = _coerce_mask_for_shape(raw_mask, shape)
    expanded = _coerce_mask_for_shape(expanded_mask, shape)
    floor = np.zeros((height, width), dtype=np.uint8)
    if height <= 0 or width <= 0 or not np.any(raw) or not np.any(expanded):
        return floor

    expanded_only = cv2.bitwise_and(expanded, cv2.bitwise_not(np.where(raw > 0, 255, 0).astype(np.uint8)))
    if not np.any(expanded_only):
        return floor

    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 33))
    near_raw = cv2.dilate(np.where(raw > 0, 255, 0).astype(np.uint8), near_kernel, iterations=1)
    candidate = cv2.bitwise_and(expanded_only.astype(np.uint8), near_raw.astype(np.uint8))
    if not np.any(candidate):
        return floor

    bubble_union = np.zeros((height, width), dtype=np.uint8)
    support_union = np.zeros((height, width), dtype=np.uint8)
    for text in texts:
        if not isinstance(text, dict) or _route_action_blocks_inpaint(text):
            continue
        if _text_has_rejected_bubble_mask_source(text):
            continue
        real_bubble_mask, _ = _real_bubble_mask_for_text(ocr_page, text, width, height)
        if isinstance(real_bubble_mask, np.ndarray) and np.any(real_bubble_mask):
            safe_bubble = _safe_real_bubble_interior_mask(real_bubble_mask, width, height, erode_px=1)
            bubble_union = np.maximum(bubble_union, safe_bubble.astype(np.uint8))
        support_bbox = _text_source_limit_bbox(text, width, height, include_line_bbox=True)
        if support_bbox is not None:
            sx1, sy1, sx2, sy2 = support_bbox
            support_w = max(1, int(sx2) - int(sx1))
            support_h = max(1, int(sy2) - int(sy1))
            support = _mask_from_bbox(
                width,
                height,
                [
                    int(sx1) - max(10, min(22, int(round(support_w * 0.12)))),
                    int(sy1) - max(8, min(18, int(round(support_h * 0.70)))),
                    int(sx2) + max(10, min(22, int(round(support_w * 0.12)))),
                    int(sy2) + max(8, min(18, int(round(support_h * 0.70)))),
                ],
                padding=0,
            )
            support_union = np.maximum(support_union, support.astype(np.uint8))

    if np.any(bubble_union):
        candidate = cv2.bitwise_and(candidate, bubble_union)
    if np.any(support_union):
        candidate = cv2.bitwise_and(candidate, support_union)
    return candidate.astype(np.uint8)


def _expand_strip_real_inpaint_mask(
    raw_mask: np.ndarray | None,
    expanded_mask: np.ndarray | None,
    ocr_page: dict,
    texts: list[dict],
    image_rgb: np.ndarray,
) -> np.ndarray:
    shape = image_rgb.shape[:2]
    raw = _coerce_mask_for_shape(raw_mask, shape)
    base = np.maximum(raw, _coerce_mask_for_shape(expanded_mask, shape))
    if not np.any(base):
        return raw

    candidate = _dilate_mask_asymmetric(base)
    limit = _strip_real_inpaint_expansion_limit_mask(ocr_page, texts, shape)
    raw_recovery = _strip_real_inpaint_raw_component_recovery_mask(raw, ocr_page, texts, shape)
    expanded_floor = _strip_real_inpaint_expanded_evidence_floor_mask(raw, base, ocr_page, texts, shape)
    if np.any(limit):
        clipped_base = cv2.bitwise_and(base.astype(np.uint8), limit.astype(np.uint8))
        clipped_candidate = cv2.bitwise_and(candidate.astype(np.uint8), limit.astype(np.uint8))
        raw_pixels = int(np.count_nonzero(raw))
        raw_kept = int(np.count_nonzero(cv2.bitwise_and(raw.astype(np.uint8), limit.astype(np.uint8))))
        if raw_pixels > 0 and _strip_texts_need_raw_floor(texts) and raw_kept < raw_pixels:
            base = np.maximum(clipped_base, raw)
            candidate = np.maximum(clipped_candidate, raw)
            ocr_page["_strip_raw_floor_preserved_after_limit"] = {
                "raw_pixels": raw_pixels,
                "raw_kept_pixels": raw_kept,
            }
        else:
            base = clipped_base
            candidate = clipped_candidate
        if np.any(expanded_floor):
            lost_pixels = int(np.count_nonzero(cv2.bitwise_and(expanded_floor, cv2.bitwise_not(candidate.astype(np.uint8)))))
            if lost_pixels > 0:
                ocr_page["_strip_expanded_floor_preserved_after_limit"] = {
                    "preserved_pixels": lost_pixels,
                }
            candidate = np.maximum(candidate, expanded_floor)
    else:
        candidate = base
    if np.any(raw_recovery):
        candidate = np.maximum(candidate, raw_recovery)
    return np.maximum(base, candidate).astype(np.uint8)


def _sanitize_precomputed_remaining_mask(
    precomputed_mask: np.ndarray | None,
    rebuilt_mask: np.ndarray | None,
    *,
    original_text_count: int,
    filtered_text_count: int,
) -> np.ndarray:
    precomputed = np.where(np.asarray(precomputed_mask) > 0, 255, 0).astype(np.uint8) if precomputed_mask is not None else None
    rebuilt = np.where(np.asarray(rebuilt_mask) > 0, 255, 0).astype(np.uint8) if rebuilt_mask is not None else None
    if precomputed is None:
        return rebuilt if rebuilt is not None else np.zeros((0, 0), dtype=np.uint8)
    if rebuilt is None or rebuilt.shape[:2] != precomputed.shape[:2] or not np.any(rebuilt):
        return precomputed
    if int(filtered_text_count) < int(original_text_count):
        return rebuilt
    return precomputed


def _sanitize_rebuilt_expanded_mask(
    expanded_mask: np.ndarray | None,
    rebuilt_expanded_mask: np.ndarray | None,
    raw_mask: np.ndarray | None,
) -> np.ndarray:
    raw = np.where(np.asarray(raw_mask) > 0, 255, 0).astype(np.uint8) if raw_mask is not None else None
    expanded = (
        np.where(np.asarray(expanded_mask) > 0, 255, 0).astype(np.uint8)
        if expanded_mask is not None
        else None
    )
    rebuilt = (
        np.where(np.asarray(rebuilt_expanded_mask) > 0, 255, 0).astype(np.uint8)
        if rebuilt_expanded_mask is not None
        else None
    )
    if raw is None:
        return rebuilt if rebuilt is not None else (expanded if expanded is not None else np.zeros((0, 0), dtype=np.uint8))
    base = np.maximum(raw, expanded) if expanded is not None and expanded.shape[:2] == raw.shape[:2] else raw.copy()
    if rebuilt is None or rebuilt.shape[:2] != raw.shape[:2] or not np.any(rebuilt):
        return base.astype(np.uint8)

    raw_pixels = int(np.count_nonzero(raw))
    rebuilt_pixels = int(np.count_nonzero(rebuilt))
    if raw_pixels <= 0:
        return np.maximum(base, rebuilt).astype(np.uint8)
    if rebuilt_pixels < int(raw_pixels * 0.92):
        return base.astype(np.uint8)
    if rebuilt_pixels > max(raw_pixels + 4096, int(raw_pixels * 3.5)):
        return base.astype(np.uint8)
    return np.maximum(base, rebuilt).astype(np.uint8)


def _augment_inpaint_masks_from_texts(
    raw_mask: np.ndarray | None,
    expanded_mask: np.ndarray | None,
    texts: list[dict],
    image_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shape = image_rgb.shape[:2]
    raw = _coerce_mask_for_shape(raw_mask, shape)
    expanded = _coerce_mask_for_shape(expanded_mask, shape)
    if image_rgb.size == 0 or not texts:
        return raw, expanded

    for text in texts:
        if not isinstance(text, dict) or _route_action_blocks_inpaint(text):
            continue
        bubble_source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
        qa_flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
        mask_evidence = text.get("mask_evidence") if isinstance(text.get("mask_evidence"), dict) else {}
        has_mask_evidence = int(mask_evidence.get("raw_mask_pixels") or 0) > 0
        rejected_source = _text_has_rejected_bubble_mask_source(text)
        dark_visual_text = bubble_source == "image_dark_bubble_mask" or "dark_bubble_oval_reocr" in qa_flags
        if rejected_source and not has_mask_evidence:
            continue
        if not (text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("source_bbox")):
            continue
        if rejected_source or dark_visual_text:
            bbox = _text_bbox_for_inpaint_geometry(text, shape[1], shape[0])
            if bbox is not None:
                bbox_floor = _mask_from_bbox(shape[1], shape[0], bbox, padding=8)
                if dark_visual_text:
                    clip_bbox = _normalize_bbox(text.get("balloon_bbox"), shape[1], shape[0]) or _normalize_bbox(
                        text.get("bubble_mask_bbox"),
                        shape[1],
                        shape[0],
                    )
                else:
                    clip_bbox = _normalize_bbox(text.get("balloon_bbox") or text.get("bubble_mask_bbox"), shape[1], shape[0])
                if clip_bbox is not None:
                    clip_bbox = _expanded_bbox(shape[1], shape[0], clip_bbox, padding=18) or clip_bbox
                    clip = _mask_from_bbox(shape[1], shape[0], clip_bbox, padding=3)
                    bbox_floor = cv2.bitwise_and(bbox_floor, clip)
                    bx1, by1, bx2, by2 = clip_bbox
                    crop = image_rgb[by1:by2, bx1:bx2]
                    if crop.size:
                        crop_luma = (
                            crop[:, :, 0].astype(np.float32) * 0.299
                            + crop[:, :, 1].astype(np.float32) * 0.587
                            + crop[:, :, 2].astype(np.float32) * 0.114
                        )
                        bright = np.where(crop_luma >= 95.0, 255, 0).astype(np.uint8)
                        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(bright, 8)
                        visual_text = np.zeros_like(bright)
                        for label in range(1, component_count):
                            x, y, w, h, area = [int(v) for v in stats[label]]
                            if area < 10 or area > 2400:
                                continue
                            if w > max(260, (bx2 - bx1) * 0.92) or h > max(90, (by2 - by1) * 0.80):
                                continue
                            visual_text[labels == label] = 255
                        if np.any(visual_text):
                            visual_text = cv2.dilate(
                                visual_text,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                                iterations=1,
                            )
                            visual_mask = np.zeros(shape, dtype=np.uint8)
                            visual_mask[by1:by2, bx1:bx2] = visual_text
                            bbox_floor = np.maximum(bbox_floor, visual_mask)
                if np.any(bbox_floor):
                    raw = np.maximum(raw, bbox_floor)
                    expanded = np.maximum(expanded, bbox_floor)
        evidence_block = _drop_isolated_side_note_line_polygons(dict(text))
        try:
            evidence_mask = build_inpaint_mask(evidence_block, image_rgb.shape, image_rgb=image_rgb)
        except Exception:
            evidence_mask = None
        if evidence_mask is None or not np.any(evidence_mask):
            continue
        evidence = _coerce_mask_for_shape(evidence_mask, shape)
        if not np.any(evidence):
            continue

        expanded_pixels = int(np.count_nonzero(expanded))
        raw_pixels = int(np.count_nonzero(raw))
        evidence_pixels = int(np.count_nonzero(evidence))
        extra_pixels = int(np.count_nonzero((evidence > 0) & (expanded == 0)))
        if extra_pixels <= 0:
            continue
        evidence_metrics = evidence_block.get("qa_metrics") if isinstance(evidence_block.get("qa_metrics"), dict) else {}
        has_geometry_floor_evidence = any(
            key in evidence_metrics
            for key in (
                "white_balloon_geometry_floor",
                "white_balloon_geometry_floor_after_image_derivation",
                "white_balloon_geometry_floor_after_component_cleaner",
            )
        )
        allow_raw_floor_repair = has_geometry_floor_evidence or _is_dark_or_colored_card_text(text) or (
            raw_pixels > 0 and expanded_pixels > 0 and expanded_pixels < int(raw_pixels * 0.92)
        )
        if has_geometry_floor_evidence:
            evidence = np.maximum(
                evidence,
                _dilate_mask_asymmetric(evidence, left_px=3, right_px=3, up_px=2, down_px=10),
            )
            evidence_pixels = int(np.count_nonzero(evidence))
            extra_pixels = int(np.count_nonzero((evidence > 0) & (expanded == 0)))
            if extra_pixels <= 0:
                continue
        if expanded_pixels > 0:
            if not allow_raw_floor_repair and evidence_pixels > max(expanded_pixels + 4096, int(expanded_pixels * 1.45)):
                continue
            if not allow_raw_floor_repair and extra_pixels > max(4096, int(expanded_pixels * 0.45)):
                continue
        qa_metrics = text.setdefault("qa_metrics", {})
        if isinstance(qa_metrics, dict):
            qa_metrics["text_evidence_mask_extra_pixels"] = int(extra_pixels)
            if has_geometry_floor_evidence:
                qa_metrics["text_evidence_mask_geometry_floor_accepted"] = dict(evidence_metrics)
        raw = np.maximum(raw, evidence)
        expanded = np.maximum(expanded, evidence)
    raw = _drop_unprotected_dark_outline_slivers(raw, texts, image_rgb)
    expanded = _drop_unprotected_dark_outline_slivers(expanded, texts, image_rgb)
    return raw.astype(np.uint8), expanded.astype(np.uint8)


def _drop_unprotected_dark_outline_slivers(
    mask: np.ndarray,
    texts: list[dict],
    image_rgb: np.ndarray,
) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return mask
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0 or image_rgb.shape[:2] != mask.shape[:2]:
        return mask
    height, width = mask.shape[:2]
    protected = np.zeros((height, width), dtype=np.uint8)
    has_white_context = False
    for text in texts or []:
        if not isinstance(text, dict) or _is_dark_or_colored_card_text(text):
            continue
        profile = str(text.get("layout_profile") or text.get("block_profile") or "").strip().lower()
        balloon_type = str(text.get("balloon_type") or "").strip().lower()
        bubble_source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
        if not (
            profile == "white_balloon"
            or balloon_type == "white"
            or bubble_source in {"image_contour_bubble_mask", "image_white_bubble_mask", "image_rect_bubble_mask", "text_rect_fallback"}
        ):
            continue
        text_geometry = _text_geometry_mask(width, height, text)
        if isinstance(text_geometry, np.ndarray) and np.any(text_geometry):
            protected = cv2.bitwise_or(protected, text_geometry.astype(np.uint8))
            has_white_context = True
        source_bbox = _normalize_bbox(text.get("source_bbox"), width, height)
        text_bbox = _normalize_bbox(text.get("text_pixel_bbox") or text.get("bbox"), width, height)
        if source_bbox is not None and text_bbox is not None:
            sx1, sy1, sx2, sy2 = source_bbox
            tx1, ty1, tx2, ty2 = text_bbox
            iy1 = max(sy1, ty1)
            iy2 = min(sy2, ty2)
            if sx2 > sx1 and iy2 > iy1:
                source_anchor = np.zeros((height, width), dtype=np.uint8)
                source_anchor[iy1:iy2, sx1:sx2] = 255
                protected = cv2.bitwise_or(protected, source_anchor)
                has_white_context = True
    if not has_white_context or not np.any(protected):
        return mask
    gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dark_pixels = gray < 96
    source = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, connectivity=8)
    kept = np.zeros_like(source, dtype=np.uint8)
    removed = 0
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        protected_overlap = int(np.count_nonzero(component & (protected > 0)))
        dark_overlap = int(np.count_nonzero(component & dark_pixels))
        dark_ratio = dark_overlap / float(max(1, area))
        if area <= 768 and protected_overlap == 0 and dark_overlap >= 8 and dark_ratio >= 0.18:
            removed += area
            continue
        kept[component] = 255
    return kept if removed else source


def _select_residual_check_mask(
    *,
    ocr_page: dict | None,
    shape: tuple[int, int],
    expanded_mask: np.ndarray | None,
    raw_mask: np.ndarray | None = None,
    fast_fill_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str, bool]:
    expanded = _coerce_mask_for_shape(expanded_mask, shape)
    raw = _coerce_mask_for_shape(raw_mask, shape)
    fast = _coerce_mask_for_shape(fast_fill_mask, shape)
    text_region = _build_residual_text_region_mask(ocr_page, shape)
    processable_texts_list = _processable_texts_for_inpaint(ocr_page or {})
    processable_texts = bool(processable_texts_list)

    expanded_pixels = int(np.count_nonzero(expanded))
    raw_pixels = int(np.count_nonzero(raw))
    fallback = np.zeros(shape, dtype=np.uint8)
    fallback_sources: list[str] = []
    if processable_texts and np.any(fast):
        fallback = np.maximum(fallback, fast)
        fallback_sources.append("fast_fill_mask")
    if np.any(text_region):
        fallback = np.maximum(fallback, text_region)
        fallback_sources.append("text_region")
    fallback_pixels = int(np.count_nonzero(fallback))

    if _all_processable_texts_are_white_balloon(processable_texts_list) and np.any(text_region):
        if np.any(fast):
            return np.maximum(text_region, fast), "text_region_white_balloon+fast_fill_mask", False
        return text_region, "text_region_white_balloon", False

    if expanded_pixels:
        insufficient = bool(fallback_pixels and expanded_pixels < max(32, int(fallback_pixels * 0.10)))
        if not insufficient:
            return expanded, "expanded_mask", False
        return np.maximum(expanded, fallback), "expanded_mask+" + "+".join(fallback_sources), True
    if raw_pixels:
        insufficient = bool(fallback_pixels and raw_pixels < max(32, int(fallback_pixels * 0.10)))
        if not insufficient:
            return raw, "raw_mask", False
        return np.maximum(raw, fallback), "raw_mask+" + "+".join(fallback_sources), True
    if fallback_pixels:
        return fallback, "+".join(fallback_sources), True
    return expanded, "empty_region", False


def _append_inpaint_decision_flag(ocr_page: dict, flag: str) -> None:
    flags = ocr_page.setdefault("_strip_inpaint_decision_flags", [])
    if not isinstance(flags, list):
        flags = []
        ocr_page["_strip_inpaint_decision_flags"] = flags
    if flag not in flags:
        flags.append(flag)


def _mark_suspicious_fast_fill_without_raw_mask(
    ocr_page: dict,
    fast_fill_mask: np.ndarray | None,
    raw_mask: np.ndarray | None,
) -> None:
    fast_pixels = int(np.count_nonzero(fast_fill_mask)) if isinstance(fast_fill_mask, np.ndarray) else 0
    raw_pixels = int(np.count_nonzero(raw_mask)) if isinstance(raw_mask, np.ndarray) else 0
    if fast_pixels <= 0 or raw_pixels > 0:
        return
    ocr_page["_strip_fast_fill_without_raw_mask"] = True
    _append_inpaint_decision_flag(ocr_page, "fast_fill_without_raw_mask")


def _text_region_looks_plain_white(image_rgb: np.ndarray, text: dict) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return False
    height, width = image_rgb.shape[:2]
    bbox = _local_context_bbox_for_text(text, width, height)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    sample_mask = np.zeros((height, width), dtype=np.uint8)
    sample_mask[y1:y2, x1:x2] = 255
    text_bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height) or _normalize_bbox(text.get("bbox"), width, height)
    if text_bbox is not None:
        text_mask = _mask_from_bbox(width, height, text_bbox, padding=4)
        sample_mask = cv2.bitwise_and(sample_mask, cv2.bitwise_not(text_mask))
    if int(np.count_nonzero(sample_mask)) < 64:
        return False
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    pixels = gray[sample_mask > 0].astype(np.float32)
    if pixels.size < 64:
        return False
    return float(np.median(pixels)) >= 238.0 and float(np.std(pixels)) <= 8.0


def _page_has_nonwhite_text_for_light_residual(ocr_page: dict | None, image_rgb: np.ndarray | None = None) -> bool:
    if not isinstance(ocr_page, dict):
        return False
    for text in _processable_texts_for_inpaint(ocr_page):
        if image_rgb is not None:
            if _text_region_looks_plain_white(image_rgb, text):
                continue
            if text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("bbox"):
                return True
        color = _metadata_background_color(text)
        if color is None:
            continue
        color_f = color.astype(np.float32)
        luma = float(np.mean(color_f))
        chroma = float(np.max(color_f) - np.min(color_f))
        if luma < 228.0 or chroma > 18.0:
            return True
    return False


def _fill_white_balloon_residual_mask(image_rgb: np.ndarray, mask: np.ndarray, texts: list[dict]) -> np.ndarray:
    if image_rgb.size == 0 or mask.size == 0 or not texts:
        return image_rgb
    for text in texts:
        if not _text_region_looks_plain_white(image_rgb, text):
            return image_rgb
    fill_color = _sample_local_solid_fill_color(image_rgb, mask)
    result = image_rgb.copy()
    result[mask > 0] = fill_color
    return result


def _cleanup_dark_glyph_residuals_in_text_mask(
    image_rgb: np.ndarray,
    residual_mask: np.ndarray | None,
    texts: list[dict],
) -> tuple[np.ndarray, int]:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0 or not texts:
        return image_rgb, 0
    height, width = image_rgb.shape[:2]
    residual = _coerce_mask_for_shape(residual_mask, (height, width)) > 0
    if not np.any(residual):
        return image_rgb, 0

    focus = np.zeros((height, width), dtype=np.uint8)
    focus_rects: list[list[int]] = []
    for text in texts:
        if not isinstance(text, dict):
            continue
        bbox = (
            _line_polygons_bbox(text, width, height, padding=0)
            or _normalize_bbox(text.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(text.get("bbox"), width, height)
        )
        if bbox is None:
            continue
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        pad_x = max(18, min(42, int(round(bw * 0.30))))
        pad_y = max(6, min(18, int(round(bh * 0.35))))
        x1, y1, x2, y2 = bbox
        rect = _normalize_bbox([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)
        if rect is not None:
            focus_rects.append(rect)
            focus = np.maximum(focus, _mask_from_bbox(width, height, rect, padding=0))
    if not np.any(focus):
        return image_rgb, 0

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    candidates = (gray <= 128) & (focus > 0)
    if not np.any(candidates):
        return image_rgb, 0

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)
    cleanup = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, labels_count):
        x, y, comp_w, comp_h, area = [int(v) for v in stats[label]]
        if area < 3 or area > 320:
            continue
        if comp_w > 30 or comp_h > 38:
            continue
        touches_focus_edge = False
        for rx1, ry1, rx2, ry2 in focus_rects:
            if rx1 <= x < rx2 and ry1 <= y < ry2:
                if x <= rx1 + 1 or y <= ry1 + 1 or x + comp_w >= rx2 - 1 or y + comp_h >= ry2 - 1:
                    touches_focus_edge = True
                    break
        if touches_focus_edge:
            continue
        slender = max(comp_w, comp_h) / float(max(1, min(comp_w, comp_h)))
        if slender >= 10.0 and area >= 80:
            continue
        cleanup[labels == label] = 255
    pixels = int(np.count_nonzero(cleanup))
    if pixels <= 0:
        return image_rgb, 0
    cleaned = image_rgb.copy()
    cleaned[cleanup > 0] = _sample_local_solid_fill_color(image_rgb, cleanup)
    return cleaned, pixels


def _sample_local_solid_fill_color(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image_rgb.size == 0 or not isinstance(mask, np.ndarray) or not np.any(mask):
        return np.asarray([255, 255, 255], dtype=np.uint8)
    if mask.shape[:2] != image_rgb.shape[:2]:
        return np.asarray([255, 255, 255], dtype=np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    sample_mask = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    )
    sample_mask = cv2.bitwise_and(sample_mask, cv2.bitwise_not(mask_u8))
    if int(np.count_nonzero(sample_mask)) < 32:
        sample_mask = cv2.bitwise_not(mask_u8)
    sample = image_rgb[sample_mask > 0]
    if sample.size == 0:
        return np.asarray([255, 255, 255], dtype=np.uint8)
    fill = np.median(sample.astype(np.float32), axis=0).clip(0, 255)
    return np.asarray([int(round(float(v))) for v in fill], dtype=np.uint8)


def _detect_inpaint_residual_text(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray | None,
    expanded_mask: np.ndarray | None,
    *,
    raw_mask: np.ndarray | None = None,
    fast_fill_mask: np.ndarray | None = None,
    ocr_page: dict | None = None,
) -> dict:
    if cleaned_rgb is None:
        return {"has_residual": False, "score": 0.0, "flags": ["missing_cleaned_image"]}
    try:
        from qa.inpaint_residual import detect_residual_text

        mask, source, include_unchanged_dark = _select_residual_check_mask(
            ocr_page=ocr_page,
            shape=original_rgb.shape[:2],
            expanded_mask=expanded_mask,
            raw_mask=raw_mask,
            fast_fill_mask=fast_fill_mask,
        )
        result = detect_residual_text(
            original_rgb,
            cleaned_rgb,
            mask,
            include_unchanged_dark=include_unchanged_dark,
            include_light_residual=_page_has_nonwhite_text_for_light_residual(ocr_page, original_rgb),
        )
        result["region_source"] = source
        result["region_pixels"] = int(np.count_nonzero(mask))
        if include_unchanged_dark:
            flags = list(result.get("flags") or [])
            if "fallback_region" not in flags:
                flags.append("fallback_region")
            result["flags"] = flags
        return result
    except Exception as exc:
        return {"has_residual": False, "score": 0.0, "flags": [f"residual_check_failed:{type(exc).__name__}"]}


def _light_residual_contrast(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.float32)
    min_side = min(gray.shape[:2])
    if min_side < 5:
        return np.zeros_like(gray, dtype=np.float32)
    kernel = max(5, min(31, (min_side // 3) | 1))
    if kernel % 2 == 0:
        kernel += 1
    return gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (kernel, kernel), 0)


def _build_light_residual_retry_mask(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    base_mask: np.ndarray | None,
    limit_mask: np.ndarray | None,
) -> np.ndarray | None:
    if cleaned_rgb is None or original_rgb.shape[:2] != cleaned_rgb.shape[:2]:
        return None
    shape = original_rgb.shape[:2]
    base = _coerce_mask_for_shape(base_mask, shape) > 0
    limit = _coerce_mask_for_shape(limit_mask, shape) > 0
    if not np.any(base) or not np.any(limit):
        return None

    before_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    after_gray = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    before_contrast = _light_residual_contrast(before_gray)
    after_contrast = _light_residual_contrast(after_gray)
    candidate = (
        limit
        & (before_gray >= 210.0)
        & (after_gray >= 210.0)
        & (before_contrast >= 10.0)
        & (after_contrast >= 10.0)
    )
    if not np.any(candidate):
        return None

    raw = candidate.astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    extras = np.zeros(shape, dtype=np.uint8)
    height, width = shape
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if 2 <= area <= 1400 and comp_w <= max(18, int(width * 0.24)) and comp_h <= max(10, int(height * 0.45)):
            extras[labels == label] = 255
    if not np.any(extras):
        return None

    extras = cv2.dilate(extras, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    retry = np.maximum(base.astype(np.uint8) * 255, extras)
    retry = cv2.bitwise_and(retry, limit.astype(np.uint8) * 255)
    base_pixels = int(np.count_nonzero(base))
    retry_pixels = int(np.count_nonzero(retry))
    if retry_pixels <= base_pixels:
        return None
    if retry_pixels > max(base_pixels + 1024, int(base_pixels * 1.35)):
        bounded = np.maximum(base.astype(np.uint8) * 255, cv2.bitwise_and(extras, base.astype(np.uint8) * 255))
        bounded = cv2.bitwise_and(bounded, limit.astype(np.uint8) * 255)
        return bounded if int(np.count_nonzero(bounded)) > base_pixels else None
    return retry


def _build_dark_residual_retry_mask(
    base_mask: np.ndarray | None,
    limit_mask: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    base = _coerce_mask_for_shape(base_mask, shape) > 0
    limit = _coerce_mask_for_shape(limit_mask, shape) > 0
    if not np.any(base) or not np.any(limit):
        return None

    expanded = cv2.dilate(
        base.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    retry = cv2.bitwise_and(expanded, limit.astype(np.uint8) * 255)
    base_pixels = int(np.count_nonzero(base))
    retry_pixels = int(np.count_nonzero(retry))
    if retry_pixels <= max(base_pixels + 512, int(base_pixels * 1.08)):
        retry = expanded
        retry_pixels = int(np.count_nonzero(retry))
    if retry_pixels <= base_pixels:
        return None
    if retry_pixels > max(base_pixels + 1600, int(base_pixels * 1.45)):
        bounded = cv2.bitwise_and(
            cv2.dilate(
                base.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ),
            limit.astype(np.uint8) * 255,
        )
        return bounded if int(np.count_nonzero(bounded)) > base_pixels else None
    return retry


def _apply_white_residual_expanded_mask_force_fill(
    cleaned_rgb: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    if cleaned_rgb is None or cleaned_rgb.size == 0:
        return cleaned_rgb
    fill_mask = _coerce_mask_for_shape(mask, cleaned_rgb.shape[:2]) > 0
    if not np.any(fill_mask):
        return cleaned_rgb

    gray = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2HSV)
    sample_mask = cv2.dilate(
        fill_mask.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
        iterations=1,
    ) > 0
    sample_mask &= ~fill_mask
    sample_mask &= gray >= 220
    sample_mask &= hsv[:, :, 1] <= 80
    local_ring = cv2.dilate(
        fill_mask.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    ) > 0
    local_ring &= ~fill_mask
    local_luma = gray[local_ring] if np.any(local_ring) else np.asarray([], dtype=gray.dtype)
    if int(np.count_nonzero(sample_mask)) < 24:
        sample_mask = (~fill_mask) & (gray >= 235) & (hsv[:, :, 1] <= 64)
    if int(np.count_nonzero(sample_mask)) < 24:
        if local_luma.size >= 24 and float(np.percentile(local_luma, 75)) < 150.0:
            local_rgb = cleaned_rgb[local_ring].astype(np.float32)
            fill = np.median(local_rgb, axis=0).clip(0, 255)
            fill_color = np.asarray([int(round(float(v))) for v in fill], dtype=np.uint8)
        else:
            fill_color = np.asarray([255, 255, 255], dtype=np.uint8)
    else:
        fill = np.median(cleaned_rgb[sample_mask].astype(np.float32), axis=0).clip(0, 255)
        fill_color = np.asarray([int(round(float(v))) for v in fill], dtype=np.uint8)

    result = cleaned_rgb.copy()
    result[fill_mask] = fill_color
    return result


def _apply_translator_note_dark_text_contract_fill(
    cleaned_rgb: np.ndarray,
    ocr_page: dict,
    texts_override: list[dict] | None = None,
) -> tuple[np.ndarray, int]:
    if not isinstance(cleaned_rgb, np.ndarray) or cleaned_rgb.ndim != 3 or not isinstance(ocr_page, dict):
        return cleaned_rgb, 0
    height, width = cleaned_rgb.shape[:2]
    result = cleaned_rgb.copy()
    changed_pixels = 0
    try:
        band_y_top = int(ocr_page.get("_band_y_top") or 0)
    except Exception:
        band_y_top = 0
    note_candidates: list[dict] = []
    seen_note_keys: set[str] = set()
    collections = (
        [texts_override]
        if isinstance(texts_override, list)
        else [
            ocr_page.get("_strip_inpaint_local_texts"),
            ocr_page.get("texts"),
            ocr_page.get("text_samples"),
        ]
    )
    for collection in collections:
        if not isinstance(collection, list):
            continue
        for text in collection:
            if not isinstance(text, dict):
                continue
            key = str(text.get("trace_id") or text.get("id") or id(text))
            if key in seen_note_keys:
                continue
            seen_note_keys.add(key)
            note_candidates.append(dict(text))
    local_texts = _texts_with_band_local_bboxes(
        note_candidates,
        width=width,
        height=height,
        band_y_top=band_y_top,
    )
    for raw_text in local_texts:
        if not isinstance(raw_text, dict):
            continue
        source = str(raw_text.get("bubble_mask_source") or raw_text.get("bubbleMaskSource") or "").strip().lower()
        if source != "translator_note_text_mask":
            continue
        if _text_suppressed_for_inpaint(raw_text) or _route_action_blocks_inpaint(raw_text):
            continue
        background = raw_text.get("background_rgb")
        fill_color = np.asarray([0, 0, 0], dtype=np.uint8)
        if isinstance(background, (list, tuple)) and len(background) >= 3:
            try:
                candidate = np.asarray(
                    [int(max(0, min(255, round(float(v))))) for v in background[:3]],
                    dtype=np.uint8,
                )
                if int(np.max(candidate)) <= 96:
                    fill_color = candidate
            except Exception:
                fill_color = np.asarray([0, 0, 0], dtype=np.uint8)
        mask = _dark_text_contract_fill_mask(raw_text, width, height, cleaned_rgb)
        try:
            text_mask = build_inpaint_mask(raw_text, cleaned_rgb.shape, image_rgb=cleaned_rgb)
        except Exception:
            text_mask = None
        if isinstance(text_mask, np.ndarray) and np.any(text_mask):
            text_mask = _coerce_mask_for_shape(text_mask, (height, width))
            mask = np.maximum(mask, text_mask).astype(np.uint8) if isinstance(mask, np.ndarray) else text_mask
        text_bbox = _text_bbox_for_inpaint_geometry(raw_text, width, height)
        if text_bbox is not None:
            support_bbox = _expanded_bbox(width, height, text_bbox, padding=14) or text_bbox
            sx1, sy1, sx2, sy2 = support_bbox
            crop = result[sy1:sy2, sx1:sx2]
            if crop.size:
                crop_luma = (
                    crop[:, :, 0].astype(np.float32) * 0.299
                    + crop[:, :, 1].astype(np.float32) * 0.587
                    + crop[:, :, 2].astype(np.float32) * 0.114
                )
                bright_residual = np.where(crop_luma >= 150.0, 255, 0).astype(np.uint8)
                if np.any(bright_residual):
                    bright_residual = cv2.dilate(
                        bright_residual,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
                        iterations=1,
                    )
                    residual_mask = np.zeros((height, width), dtype=np.uint8)
                    residual_mask[sy1:sy2, sx1:sx2] = bright_residual
                    mask = (
                        np.maximum(mask, residual_mask).astype(np.uint8)
                        if isinstance(mask, np.ndarray)
                        else residual_mask
                    )
        if not isinstance(mask, np.ndarray) or not np.any(mask):
            continue
        before = result.copy()
        result[mask > 0] = fill_color
        delta = np.any(result != before, axis=2)
        changed_pixels += int(np.count_nonzero(delta & (mask > 0)))
        raw_text["_fast_fill_inpaint_resolved"] = True
        _append_text_flag(raw_text, "translator_note_dark_contract_final_fill")
    if changed_pixels:
        ocr_page["_strip_translator_note_dark_contract_final_fill_pixels"] = int(changed_pixels)
    return result, changed_pixels


def _apply_dark_mask_component_bright_residual_fill(
    cleaned_rgb: np.ndarray,
    original_rgb: np.ndarray,
    action_mask: np.ndarray | None,
) -> tuple[np.ndarray, int, int]:
    if (
        not isinstance(cleaned_rgb, np.ndarray)
        or cleaned_rgb.ndim != 3
        or not isinstance(original_rgb, np.ndarray)
        or original_rgb.shape[:2] != cleaned_rgb.shape[:2]
        or not isinstance(action_mask, np.ndarray)
    ):
        return cleaned_rgb, 0, 0
    height, width = cleaned_rgb.shape[:2]
    mask = _coerce_mask_for_shape(action_mask, (height, width))
    if not np.any(mask):
        return cleaned_rgb, 0, 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return cleaned_rgb, 0, 0
    result = cleaned_rgb.copy()
    total_changed = 0
    component_count = 0
    cleaned_luma = (
        cleaned_rgb[:, :, 0].astype(np.float32) * 0.299
        + cleaned_rgb[:, :, 1].astype(np.float32) * 0.587
        + cleaned_rgb[:, :, 2].astype(np.float32) * 0.114
    )
    original_luma = (
        original_rgb[:, :, 0].astype(np.float32) * 0.299
        + original_rgb[:, :, 1].astype(np.float32) * 0.587
        + original_rgb[:, :, 2].astype(np.float32) * 0.114
    )
    for label in range(1, num_labels):
        x, y, comp_w, comp_h, area = [int(v) for v in stats[label].tolist()]
        if area < 80 or comp_w <= 0 or comp_h <= 0:
            continue
        x2 = min(width, x + comp_w)
        y2 = min(height, y + comp_h)
        if x >= x2 or y >= y2:
            continue
        comp = labels[y:y2, x:x2] == label
        comp_orig_luma = original_luma[y:y2, x:x2]
        comp_clean_luma = cleaned_luma[y:y2, x:x2]
        support = cv2.dilate(
            comp.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7)),
            iterations=1,
        ) > 0
        local_orig = original_luma[y:y2, x:x2]
        background_pixels = local_orig[support & ~comp]
        if background_pixels.size < 12:
            background_pixels = comp_orig_luma[~comp] if np.any(~comp) else comp_orig_luma.reshape(-1)
        if background_pixels.size == 0:
            continue
        background_luma = float(np.median(background_pixels))
        component_luma = float(np.median(comp_orig_luma[comp])) if np.any(comp) else 255.0
        if background_luma > 96.0 or component_luma > 170.0:
            continue
        bright_residual = comp & (comp_clean_luma >= 150.0)
        if not np.any(bright_residual):
            continue
        bright_residual = cv2.dilate(
            bright_residual.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
            iterations=1,
        ) > 0
        fill_candidates = bright_residual & comp
        changed = int(np.count_nonzero(fill_candidates))
        if changed <= 0:
            continue
        dark_samples = result[y:y2, x:x2][support & ~comp]
        if dark_samples.size:
            fill_color = np.median(dark_samples.reshape(-1, 3), axis=0)
            if float(np.max(fill_color)) > 96.0:
                fill_color = np.asarray([0, 0, 0], dtype=np.float32)
        else:
            fill_color = np.asarray([0, 0, 0], dtype=np.float32)
        local_result = result[y:y2, x:x2]
        local_result[fill_candidates] = np.asarray(fill_color, dtype=np.uint8)
        result[y:y2, x:x2] = local_result
        total_changed += changed
        component_count += 1
    return result, total_changed, component_count


def _record_inpaint_decision(
    ocr_page: dict,
    *,
    original_rgb: np.ndarray,
    working_rgb: np.ndarray,
    cleaned_rgb: np.ndarray | None,
    vision_blocks: list[dict],
    used_real_inpaint: bool,
    fast_fill_mask: np.ndarray,
    raw_mask: np.ndarray,
    expanded_mask: np.ndarray,
    effective_limit_mask: np.ndarray,
) -> None:
    recorder = _active_debug_recorder()
    if recorder is None:
        return

    changed = np.any(cleaned_rgb != working_rgb, axis=2) if cleaned_rgb is not None else np.zeros(expanded_mask.shape, dtype=bool)
    residual_ocr_page = ocr_page
    residual_texts = ocr_page.get("_strip_residual_texts") if isinstance(ocr_page, dict) else None
    if isinstance(residual_texts, list):
        residual_ocr_page = dict(ocr_page)
        residual_ocr_page["texts"] = [dict(text) for text in residual_texts if isinstance(text, dict)]
    residual = _detect_inpaint_residual_text(
        original_rgb,
        cleaned_rgb,
        expanded_mask,
        raw_mask=raw_mask,
        fast_fill_mask=fast_fill_mask,
        ocr_page=residual_ocr_page,
    )
    if residual.get("has_residual") and ocr_page.get("_strip_fast_fill_residual_real_inpaint"):
        try:
            residual_score = float(residual.get("score") or 0.0)
        except Exception:
            residual_score = 0.0
        if residual_score < 0.02:
            residual = dict(residual)
            residual["has_residual"] = False
            flags = list(residual.get("flags") or [])
            if "low_residual_after_fast_fill_repair" not in flags:
                flags.append("low_residual_after_fast_fill_repair")
            residual["flags"] = flags
    if residual.get("has_residual") and ocr_page.get("_strip_light_residual_retry"):
        try:
            residual_score = float(residual.get("score") or 0.0)
        except Exception:
            residual_score = 0.0
        if residual_score < 0.02:
            residual = dict(residual)
            residual["has_residual"] = False
            flags = list(residual.get("flags") or [])
            if "low_residual_after_light_retry" not in flags:
                flags.append("low_residual_after_light_retry")
            residual["flags"] = flags
    if residual.get("has_residual") and ocr_page.get("_strip_dark_residual_retry"):
        try:
            residual_score = float(residual.get("score") or 0.0)
        except Exception:
            residual_score = 0.0
        if residual_score < 0.02:
            residual = dict(residual)
            residual["has_residual"] = False
            flags = list(residual.get("flags") or [])
            if "low_residual_after_dark_retry" not in flags:
                flags.append("low_residual_after_dark_retry")
            residual["flags"] = flags
    if ocr_page.get("_strip_used_dark_panel_fill"):
        residual = dict(residual)
        flags = list(residual.get("flags") or [])
        if "dark_panel_fill_applied" not in flags:
            flags.append("dark_panel_fill_applied")
        residual["flags"] = flags
    band_id = _strip_band_id(ocr_page)
    trace_ids = _strip_trace_ids(ocr_page)
    engine_preset = ocr_page.get("_engine_preset") if isinstance(ocr_page.get("_engine_preset"), dict) else {}
    validated_source_bboxes = _collect_bboxes_for_debug(
        ocr_page,
        vision_blocks,
        "_validated_text_source_bboxes",
    )
    rejected_source_bboxes = _collect_bboxes_for_debug(
        ocr_page,
        vision_blocks,
        "_rejected_text_source_bboxes",
    )
    decision_flags = [
        str(flag)
        for flag in ocr_page.get("_strip_inpaint_decision_flags", [])
        if flag
    ]
    payload = {
        "page_id": _strip_page_id(ocr_page),
        "band_id": band_id,
        "page_number": int(ocr_page.get("_source_page_number") or ocr_page.get("numero") or 0),
        "band_index": int(ocr_page.get("_band_index") or 0),
        "text_ids": _strip_text_ids(ocr_page),
        "trace_ids": trace_ids,
        "trace_ids_in_band": trace_ids,
        "engine_preset_id": str(engine_preset.get("engine_preset_id") or ocr_page.get("engine_preset_id") or ""),
        "mask_strategy": str(engine_preset.get("mask_strategy") or ""),
        "detector_engine_id": str(engine_preset.get("detector_engine_id") or engine_preset.get("detector") or ""),
        "validated_text_source_bboxes": validated_source_bboxes,
        "rejected_text_source_bboxes": rejected_source_bboxes,
        "skip_inpaint_requested": bool(ocr_page.get("_skip_inpaint_requested") or ocr_page.get("skip_inpaint_requested")),
        "skip_inpaint_honored": bool(ocr_page.get("_skip_inpaint_honored")),
        "used_fast_solid_fill": bool(ocr_page.get("_strip_used_fast_solid_fill")),
        "fast_solid_balloon_count": int(ocr_page.get("_strip_fast_solid_balloon_count") or 0),
        "fast_solid_white_count": int(ocr_page.get("_strip_fast_solid_white_count") or 0),
        "fast_solid_black_count": int(ocr_page.get("_strip_fast_solid_black_count") or 0),
        "fast_solid_colored_count": int(ocr_page.get("_strip_fast_solid_colored_count") or 0),
        "used_fast_white_fill": bool(ocr_page.get("_strip_used_fast_white_fill")),
        "connected_white_geometry_fill_count": int(ocr_page.get("_strip_connected_white_geometry_fill_count") or 0),
        "connected_white_geometry_fill_mask_pixels": int(
            ocr_page.get("_strip_connected_white_geometry_fill_mask_pixels") or 0
        ),
        "used_fast_local_fill": bool(ocr_page.get("_strip_used_fast_local_fill")),
        "used_fast_dark_fill": bool(ocr_page.get("_strip_used_fast_dark_fill")),
        "used_flat_ui_prefill": bool(ocr_page.get("_strip_used_flat_ui_prefill")),
        "flat_ui_prefill_count": int(ocr_page.get("_strip_flat_ui_prefill_count") or 0),
        "used_dark_panel_fill": bool(ocr_page.get("_strip_used_dark_panel_fill")),
        "dark_panel_fill_count": int(ocr_page.get("_strip_dark_panel_fill_count") or 0),
        "used_real_inpaint": bool(used_real_inpaint or ocr_page.get("_strip_used_real_inpaint")),
        "used_post_cleanup": bool(ocr_page.get("_strip_used_post_cleanup")),
        "post_cleanup_skipped_reason": str(ocr_page.get("_strip_post_cleanup_skipped_reason") or ""),
        "remaining_inpaint_blocks": len(vision_blocks),
        "fast_fill_mask_pixels": int(np.count_nonzero(fast_fill_mask)),
        "raw_mask_pixels": int(np.count_nonzero(raw_mask)),
        "expanded_mask_pixels": int(np.count_nonzero(expanded_mask)),
        "fast_fill_without_raw_mask": bool(ocr_page.get("_strip_fast_fill_without_raw_mask")),
        "changed_pixels_total": int(np.count_nonzero(changed)),
        "changed_pixels_outside_expanded": int(np.count_nonzero(changed & (expanded_mask == 0))),
        "changed_outside_expanded_pixels": int(np.count_nonzero(changed & (expanded_mask == 0))),
        "changed_pixels_outside_effective_limit": int(np.count_nonzero(changed & (effective_limit_mask == 0))),
        "raw_changed_outside_limit_mask": int(ocr_page.get("_strip_raw_changed_outside_limit_mask") or 0),
        "cleanup_changed_outside_limit_mask": int(ocr_page.get("cleanup_changed_outside_limit_mask") or 0),
        "residual_text": residual,
        "fast_fill_residual_check": ocr_page.get("_strip_fast_fill_residual_check"),
        "fast_solid_fill_samples": ocr_page.get("_strip_fast_solid_fill_samples") or [],
        "fast_solid_rejection_reasons": ocr_page.get("_strip_fast_solid_rejection_reasons") or {},
        "fast_solid_fill_reject_reasons": ocr_page.get("_strip_fast_solid_fill_reject_reasons") or {},
        "fast_fill_residual_real_inpaint": bool(ocr_page.get("_strip_fast_fill_residual_real_inpaint")),
        "fast_fill_residual_mask_pixels": int(ocr_page.get("_strip_fast_fill_residual_mask_pixels") or 0),
        "light_residual_retry": bool(ocr_page.get("_strip_light_residual_retry")),
        "light_residual_retry_mask_pixels": int(ocr_page.get("_strip_light_residual_retry_mask_pixels") or 0),
        "dark_residual_retry": bool(ocr_page.get("_strip_dark_residual_retry")),
        "dark_residual_retry_mask_pixels": int(ocr_page.get("_strip_dark_residual_retry_mask_pixels") or 0),
        "white_residual_expanded_mask_force_fill": bool(
            ocr_page.get("_strip_white_residual_expanded_mask_force_fill")
        ),
        "white_residual_expanded_mask_force_fill_pixels": int(
            ocr_page.get("_strip_white_residual_expanded_mask_force_fill_pixels") or 0
        ),
        "white_residual_expanded_mask_force_fill_mask_pixels": int(
            ocr_page.get("_strip_white_residual_expanded_mask_force_fill_mask_pixels") or 0
        ),
        "white_residual_expanded_mask_force_fill_source": str(
            ocr_page.get("_strip_white_residual_expanded_mask_force_fill_source") or ""
        ),
        "final_action_mask_white_cleanup_added_pixels": int(
            ocr_page.get("_strip_final_action_mask_white_cleanup_added_pixels") or 0
        ),
        "dark_mask_component_bright_residual_fill_pixels": int(
            ocr_page.get("_strip_dark_mask_component_bright_residual_fill_pixels") or 0
        ),
        "dark_mask_component_bright_residual_fill_count": int(
            ocr_page.get("_strip_dark_mask_component_bright_residual_fill_count") or 0
        ),
    }
    if bool(payload["used_real_inpaint"]) and not residual.get("has_residual"):
        preliminary_fast_fill_flags = {
            "fast_fill_insufficient_coverage",
            "text_residual_after_inpaint_suspected",
        }
        decision_flags = [
            flag for flag in decision_flags if flag not in preliminary_fast_fill_flags
        ]
    residual_qa_flag = _residual_text_qa_flag(residual)
    if residual_qa_flag:
        decision_flags.append(residual_qa_flag)
    payload["flags"] = list(dict.fromkeys(decision_flags))
    recorder.write_json(f"08_inpaint/{payload['band_id']}/inpaint_decision.json", payload)


def _collect_bboxes_for_debug(ocr_page: dict, vision_blocks: list[dict], key: str) -> list[list[int]]:
    collected: list[list[int]] = []
    for source in list(ocr_page.get("texts", []) or []) + list(vision_blocks or []):
        if not isinstance(source, dict):
            continue
        values = source.get(key)
        if not isinstance(values, (list, tuple)):
            continue
        for value in values:
            bbox = _normalize_bbox(value, 10**9, 10**9)
            if bbox is None:
                continue
            if bbox not in collected:
                collected.append(bbox)
    return collected


def _residual_text_qa_flag(residual: dict) -> str:
    """Classify residual evidence without hiding weak but traceable artifacts."""
    if not isinstance(residual, dict) or not residual.get("has_residual"):
        return ""
    flags = {str(flag).strip() for flag in residual.get("flags") or [] if str(flag).strip()}
    try:
        score = float(residual.get("score") or 0.0)
    except Exception:
        score = 0.0
    try:
        dark_pixels = int(residual.get("dark_residual_pixels") or 0)
    except Exception:
        dark_pixels = 0
    try:
        light_pixels = int(residual.get("light_residual_pixels") or 0)
    except Exception:
        light_pixels = 0
    try:
        colored_pixels = int(residual.get("colored_residual_pixels") or 0)
    except Exception:
        colored_pixels = 0
    light_only = light_pixels > 0 and dark_pixels <= 0
    light_on_dark_context = bool("dark_panel_fill_applied" in flags)
    region_source = str(residual.get("region_source") or "")

    if (
        "dark_panel_fill_applied" in flags
        and "high_residual_ratio" not in flags
        and score < 0.035
            and light_pixels < 128
    ):
        return "weak_text_residual_after_inpaint"
    if (
        "dark_panel_fill_applied" in flags
        and dark_pixels < 2500
        and light_pixels < 128
        and colored_pixels < 128
    ):
        return "weak_text_residual_after_inpaint"
    if (
        "dark_panel_fill_applied" in flags
        and dark_pixels <= 0
        and light_pixels <= 0
        and colored_pixels < 1200
        and score < 0.045
    ):
        return "weak_text_residual_after_inpaint"
    if (
        dark_pixels > 0
        and region_source.startswith("text_region_white_balloon")
        and dark_pixels < 256
        and score < 0.003
    ):
        return "weak_text_residual_after_inpaint"
    if (
        dark_pixels > 0
        and region_source.startswith("text_region_white_balloon")
        and "high_residual_ratio" not in flags
        and score < 0.008
        and dark_pixels < 1400
        and light_pixels < 64
        and colored_pixels <= 0
    ):
        return "weak_text_residual_after_inpaint"
    if (
        dark_pixels > 0
        and region_source.startswith("text_region_white_balloon")
        and "high_residual_ratio" not in flags
        and dark_pixels < 160
        and light_pixels < 64
        and colored_pixels <= 0
        and score < 0.03
    ):
        return "weak_text_residual_after_inpaint"
    if (
        dark_pixels > 0
        and dark_pixels < 384
        and light_pixels < 32
        and colored_pixels <= 0
        and "high_residual_ratio" not in flags
        and region_source == "expanded_mask"
        and not bool(residual.get("dark_background_context"))
    ):
        return "weak_text_residual_after_inpaint"
    if dark_pixels >= 160:
        return "text_residual_after_inpaint"
    if light_only and not light_on_dark_context:
        return "weak_text_residual_after_inpaint"
    if "high_residual_ratio" in flags or score >= 0.02:
        return "text_residual_after_inpaint"
    if light_pixels >= 900:
        return "text_residual_after_inpaint"
    return "weak_text_residual_after_inpaint"


def _record_mask_chain_debug(
    ocr_page: dict,
    *,
    image_rgb: np.ndarray,
    raw_mask: np.ndarray,
    expanded_mask: np.ndarray,
    effective_limit_mask: np.ndarray,
) -> None:
    recorder = _active_debug_recorder()
    if recorder is None:
        return
    try:
        from debug_tools.masks import write_mask_chain_debug_artifacts

        protection_mask = np.where((expanded_mask > 0) & (effective_limit_mask == 0), 255, 0).astype(np.uint8)
        write_mask_chain_debug_artifacts(
            recorder,
            ocr_page,
            image_rgb=image_rgb,
            raw_mask=raw_mask,
            expanded_mask=expanded_mask,
            final_mask=expanded_mask,
            protection_mask=protection_mask,
        )
    except Exception as exc:
        try:
            recorder.event(
                "mask_segmentation",
                "mask_chain_debug_failed",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        except Exception:
            pass


def _write_strip_inpaint_debug(
    ocr_page: dict,
    *,
    original_rgb: np.ndarray,
    working_rgb: np.ndarray,
    cleaned_rgb: np.ndarray | None,
    vision_blocks: list[dict],
    used_real_inpaint: bool,
    fast_fill_mask: np.ndarray | None = None,
    raw_mask: np.ndarray | None = None,
    expanded_mask: np.ndarray | None = None,
) -> None:
    debug_dir = _strip_inpaint_debug_dir(ocr_page)
    recorder = _active_debug_recorder()
    if debug_dir is None and recorder is None:
        return
    from vision_stack.runtime import _build_post_cleanup_limit_mask, vision_blocks_to_mask

    mask_kwargs = _cjk_mask_kwargs_for_strip_page(ocr_page)
    if raw_mask is None:
        raw_mask = vision_blocks_to_mask(
            working_rgb.shape,
            vision_blocks,
            image_rgb=working_rgb,
            expand_mask=False,
            **mask_kwargs,
        )
    if expanded_mask is None:
        expanded_mask = vision_blocks_to_mask(
            working_rgb.shape,
            vision_blocks,
            image_rgb=working_rgb,
            expand_mask=True,
            **mask_kwargs,
        )
    if fast_fill_mask is None:
        fast_fill_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
    dark_panel_fill_mask = _page_fill_mask(ocr_page, "_strip_dark_panel_fill_mask", raw_mask.shape[:2])
    if np.any(dark_panel_fill_mask):
        raw_mask = np.maximum(raw_mask, dark_panel_fill_mask).astype(np.uint8)
        expanded_mask = np.maximum(expanded_mask, dark_panel_fill_mask).astype(np.uint8)
    _mark_suspicious_fast_fill_without_raw_mask(ocr_page, fast_fill_mask, raw_mask)
    effective_limit_mask = _build_post_cleanup_limit_mask(
        expanded_mask,
        list(ocr_page.get("texts", [])),
        expanded_mask.shape[:2],
    )
    if effective_limit_mask is None:
        effective_limit_mask = expanded_mask
    _record_mask_chain_debug(
        ocr_page,
        image_rgb=working_rgb,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        effective_limit_mask=effective_limit_mask,
    )
    _record_inpaint_decision(
        ocr_page,
        original_rgb=original_rgb,
        working_rgb=working_rgb,
        cleaned_rgb=cleaned_rgb,
        vision_blocks=vision_blocks,
        used_real_inpaint=used_real_inpaint,
        fast_fill_mask=fast_fill_mask,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
        effective_limit_mask=effective_limit_mask,
    )
    if debug_dir is None:
        return
    _save_rgb(debug_dir / "00_band_original.jpg", original_rgb)
    _save_rgb(debug_dir / "00_band_before_inpaint.jpg", working_rgb)
    _save_mask(debug_dir / "01_fast_fill_changed_mask.png", fast_fill_mask)
    _save_mask(debug_dir / "02_inpaint_mask_raw.png", raw_mask)
    _save_mask(debug_dir / "03_inpaint_mask_expanded.png", expanded_mask)
    _save_mask(debug_dir / "04_real_inpaint_mask_used.png", expanded_mask if used_real_inpaint else np.zeros(raw_mask.shape, dtype=np.uint8))
    _save_rgb(debug_dir / "05_inpaint_mask_overlay.jpg", _mask_overlay(working_rgb, expanded_mask, vision_blocks))
    _save_mask(debug_dir / "07_effective_inpaint_limit_mask.png", effective_limit_mask)
    if cleaned_rgb is not None:
        _save_rgb(debug_dir / "06_band_after_inpaint.jpg", cleaned_rgb)
        _save_rgb(debug_dir / "04_band_after_inpaint.jpg", cleaned_rgb)
        changed = np.any(cleaned_rgb != working_rgb, axis=2)
        changed_outside = changed & (expanded_mask == 0)
        changed_outside_effective = changed & (effective_limit_mask == 0)
        _save_mask(debug_dir / "08_changed_outside_expanded_mask.png", changed_outside.astype(np.uint8) * 255)
        _save_mask(debug_dir / "09_changed_outside_effective_limit_mask.png", changed_outside_effective.astype(np.uint8) * 255)
        if np.any(changed_outside):
            _save_rgb(debug_dir / "10_changed_outside_expanded_overlay.jpg", _mask_overlay(working_rgb, changed_outside.astype(np.uint8) * 255, []))
        if np.any(changed_outside_effective):
            _save_rgb(
                debug_dir / "11_changed_outside_effective_limit_overlay.jpg",
                _mask_overlay(working_rgb, changed_outside_effective.astype(np.uint8) * 255, []),
            )
    _save_mask(debug_dir / "01_inpaint_mask_raw.png", raw_mask)
    _save_mask(debug_dir / "02_inpaint_mask_expanded.png", expanded_mask)
    _save_rgb(debug_dir / "03_inpaint_mask_overlay.jpg", _mask_overlay(working_rgb, expanded_mask, vision_blocks))
    engine_preset = ocr_page.get("_engine_preset") if isinstance(ocr_page.get("_engine_preset"), dict) else {}
    def _debug_text_samples(collection):
        samples = []
        for item in collection or []:
            if not isinstance(item, dict):
                continue
            samples.append(
                {
                    "id": item.get("id") or item.get("text_id"),
                    "trace_id": item.get("trace_id"),
                    "band_id": item.get("band_id"),
                    "bbox": item.get("bbox"),
                    "text_pixel_bbox": item.get("text_pixel_bbox"),
                    "balloon_bbox": item.get("balloon_bbox"),
                    "bubble_mask_bbox": item.get("bubble_mask_bbox"),
                    "bubble_mask_source": item.get("bubble_mask_source"),
                    "bubble_mask_error": item.get("bubble_mask_error"),
                    "qa_flags": list(item.get("qa_flags") or []),
                    "route_action": item.get("route_action"),
                    "mask_evidence": item.get("mask_evidence"),
                    "_band_local_bbox_normalized": item.get("_band_local_bbox_normalized"),
                }
            )
            if len(samples) >= 4:
                break
        return samples

    metadata = {
        "page_number": int(ocr_page.get("_source_page_number") or ocr_page.get("numero") or 0),
        "band_index": int(ocr_page.get("_band_index") or 0),
        "band_id": _strip_band_id(ocr_page),
        "band_y_top": int(ocr_page.get("_band_y_top") or 0),
        "text_count": len([t for t in ocr_page.get("texts", []) if isinstance(t, dict)]),
        "text_samples": _debug_text_samples(ocr_page.get("texts")),
        "vision_block_samples": _debug_text_samples(vision_blocks),
        "pre_recovery_text_samples": ocr_page.get("_strip_pre_recovery_text_samples") or [],
        "initial_empty_recovery_attempts": ocr_page.get("_strip_initial_empty_recovery_attempts") or [],
        "recovered_initial_empty_vision_blocks_from_texts": bool(
            ocr_page.get("_strip_recovered_initial_empty_vision_blocks_from_texts")
        ),
        "remaining_inpaint_blocks": len(vision_blocks),
        "engine_preset_id": str(engine_preset.get("engine_preset_id") or ocr_page.get("engine_preset_id") or ""),
        "mask_strategy": str(engine_preset.get("mask_strategy") or ""),
        "detector_engine_id": str(engine_preset.get("detector_engine_id") or engine_preset.get("detector") or ""),
        "validated_text_source_bboxes": _collect_bboxes_for_debug(
            ocr_page,
            vision_blocks,
            "_validated_text_source_bboxes",
        ),
        "rejected_text_source_bboxes": _collect_bboxes_for_debug(
            ocr_page,
            vision_blocks,
            "_rejected_text_source_bboxes",
        ),
        "fast_fill_mask_pixels": int(np.count_nonzero(fast_fill_mask)),
        "raw_mask_pixels": int(np.count_nonzero(raw_mask)),
        "expanded_mask_pixels": int(np.count_nonzero(expanded_mask)),
        "used_real_inpaint": bool(used_real_inpaint),
        "fast_fill_without_raw_mask": bool(ocr_page.get("_strip_fast_fill_without_raw_mask")),
        "used_fast_solid_fill": bool(ocr_page.get("_strip_used_fast_solid_fill")),
        "fast_solid_balloon_count": int(ocr_page.get("_strip_fast_solid_balloon_count") or 0),
        "fast_solid_white_count": int(ocr_page.get("_strip_fast_solid_white_count") or 0),
        "fast_solid_black_count": int(ocr_page.get("_strip_fast_solid_black_count") or 0),
        "fast_solid_colored_count": int(ocr_page.get("_strip_fast_solid_colored_count") or 0),
        "used_fast_white_fill": bool(ocr_page.get("_strip_used_fast_white_fill")),
        "fast_solid_fill_samples": ocr_page.get("_strip_fast_solid_fill_samples") or [],
        "fast_solid_rejection_reasons": ocr_page.get("_strip_fast_solid_rejection_reasons") or {},
        "fast_solid_fill_reject_reasons": ocr_page.get("_strip_fast_solid_fill_reject_reasons") or {},
        "unsafe_inpaint_block_count": int(ocr_page.get("_strip_unsafe_inpaint_block_count") or 0),
        "unsafe_inpaint_block_reasons": ocr_page.get("_strip_unsafe_inpaint_block_reasons") or {},
        "unsafe_inpaint_block_samples": ocr_page.get("_strip_unsafe_inpaint_block_samples") or [],
        "connected_white_geometry_fill_count": int(ocr_page.get("_strip_connected_white_geometry_fill_count") or 0),
        "connected_white_geometry_fill_mask_pixels": int(
            ocr_page.get("_strip_connected_white_geometry_fill_mask_pixels") or 0
        ),
        "used_fast_local_fill": bool(ocr_page.get("_strip_used_fast_local_fill")),
        "used_fast_dark_fill": bool(ocr_page.get("_strip_used_fast_dark_fill")),
        "used_dark_panel_fill": bool(ocr_page.get("_strip_used_dark_panel_fill")),
        "dark_panel_fill_count": int(ocr_page.get("_strip_dark_panel_fill_count") or 0),
        "changed_pixels_after_inpaint": int(np.count_nonzero(np.any(cleaned_rgb != working_rgb, axis=2))) if cleaned_rgb is not None else 0,
        "post_cleanup_skipped_reason": str(ocr_page.get("_strip_post_cleanup_skipped_reason") or ""),
        "changed_pixels_outside_expanded_mask": int(
            np.count_nonzero(np.any(cleaned_rgb != working_rgb, axis=2) & (expanded_mask == 0))
        )
        if cleaned_rgb is not None
        else 0,
        "changed_pixels_outside_effective_limit_mask": int(
            np.count_nonzero(np.any(cleaned_rgb != working_rgb, axis=2) & (effective_limit_mask == 0))
        )
        if cleaned_rgb is not None
        else 0,
        "raw_changed_outside_limit_mask": int(ocr_page.get("_strip_raw_changed_outside_limit_mask") or 0),
        "cleanup_changed_outside_limit_mask": int(ocr_page.get("cleanup_changed_outside_limit_mask") or 0),
        "final_clamped_outside_expanded_mask_pixels": int(
            ocr_page.get("_strip_final_clamped_outside_expanded_mask_pixels") or 0
        ),
        "final_action_mask_white_cleanup_added_pixels": int(
            ocr_page.get("_strip_final_action_mask_white_cleanup_added_pixels") or 0
        ),
        "flags": list(dict.fromkeys(str(flag) for flag in ocr_page.get("_strip_inpaint_decision_flags", []) if flag)),
    }
    (debug_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    ocr_page["_strip_inpaint_debug_dir"] = str(debug_dir)


def _clamp_final_inpaint_to_expanded_mask(
    working_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    expanded_mask: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    if not isinstance(working_rgb, np.ndarray) or not isinstance(cleaned_rgb, np.ndarray):
        return cleaned_rgb, 0
    if working_rgb.shape != cleaned_rgb.shape:
        return cleaned_rgb, 0
    if not isinstance(expanded_mask, np.ndarray) or expanded_mask.shape[:2] != cleaned_rgb.shape[:2]:
        return cleaned_rgb, 0
    allowed = expanded_mask > 0
    changed_outside = np.any(cleaned_rgb != working_rgb, axis=2) & ~allowed
    outside_count = int(np.count_nonzero(changed_outside))
    if outside_count <= 0:
        return cleaned_rgb, 0
    result = cleaned_rgb.copy()
    result[changed_outside] = working_rgb[changed_outside]
    return result, outside_count


def _extend_final_action_mask_for_white_balloon_cleanup(
    final_action_mask: np.ndarray | None,
    residual_ocr_page: dict | None,
    texts: list[dict],
    image_rgb: np.ndarray,
) -> tuple[np.ndarray, int]:
    shape = image_rgb.shape[:2] if isinstance(image_rgb, np.ndarray) and image_rgb.ndim >= 2 else (0, 0)
    action = _coerce_mask_for_shape(final_action_mask, shape)
    if not shape[0] or not shape[1]:
        return action, 0
    cleanup_texts = [
        text
        for text in texts
        if _text_allows_final_white_cleanup_extension(text, image_rgb)
        and not _translator_note_text_only_mask(text)
    ]
    if not cleanup_texts:
        return action, 0
    has_white_balloon_metadata = _has_white_balloon_cleanup_metadata(cleanup_texts)
    if not has_white_balloon_metadata and not _all_processable_texts_are_white_balloon(cleanup_texts, image_rgb):
        return action, 0
    residual_region = _build_residual_text_region_mask({"texts": cleanup_texts}, shape)
    if not np.any(residual_region):
        return action, 0
    extended = np.maximum(action, residual_region.astype(np.uint8)).astype(np.uint8)
    added = int(np.count_nonzero((extended > 0) & (action == 0)))
    return extended, added


def _text_allows_final_white_cleanup_extension(text: dict, image_rgb: np.ndarray | None = None) -> bool:
    if not isinstance(text, dict):
        return False
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if "short_dark_text_full_panel_bbox_rejected" in flags:
        return False
    if any(flag.startswith("dark_bubble") for flag in flags):
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source in {"image_dark_panel_mask", "image_dark_bubble_mask", "derived_card_panel_mask", "translator_note_text_mask"}:
        return False
    profile = str(text.get("block_profile") or text.get("layout_profile") or text.get("render_profile") or "").strip().lower()
    if profile in {"dark_panel", "dark_bubble", "black_bubble", "colored_status_panel", "status_panel", "card", "title_card"}:
        return False
    if (
        profile == "white_balloon"
        or str(text.get("balloon_type") or "").strip().lower() == "white"
        or source in {"image_white_bubble_mask", "derived_white_bubble_mask", "derived_white_crop"}
    ):
        return True
    if image_rgb is not None:
        return _text_region_looks_plain_white(image_rgb, text)
    return False


def _has_white_balloon_cleanup_metadata(texts: list[dict]) -> bool:
    return any(
        isinstance(text, dict)
        and (
            str(text.get("block_profile") or text.get("layout_profile") or "").strip().lower() == "white_balloon"
            or str(text.get("balloon_type") or "").strip().lower() == "white"
            or str(text.get("bubble_mask_source") or "").strip().lower()
            in {"image_white_bubble_mask", "derived_white_bubble_mask", "derived_white_crop"}
        )
        for text in texts
    )


def _apply_real_inpaint_for_fast_fill_residual(
    *,
    original_rgb: np.ndarray,
    working_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    ocr_page: dict,
    fast_fill_mask: np.ndarray,
) -> tuple[np.ndarray, bool, np.ndarray | None]:
    del working_rgb
    empty_mask = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
    residual = _detect_inpaint_residual_text(
        original_rgb,
        cleaned_rgb,
        empty_mask,
        raw_mask=empty_mask,
        fast_fill_mask=fast_fill_mask,
        ocr_page=ocr_page,
    )
    ocr_page["_strip_fast_fill_residual_check"] = residual
    if not residual.get("has_residual"):
        return cleaned_rgb, False, None
    if _ocr_page_has_unsafe_auto_inpaint_evidence(ocr_page):
        _append_inpaint_decision_flag(ocr_page, "real_inpaint_skipped_unsafe_mask")
        return cleaned_rgb, False, None

    residual_mask, source, _ = _select_residual_check_mask(
        ocr_page=ocr_page,
        shape=original_rgb.shape[:2],
        expanded_mask=empty_mask,
        raw_mask=empty_mask,
        fast_fill_mask=fast_fill_mask,
    )
    texts_for_mask = [dict(text) for text in ocr_page.get("texts", []) if isinstance(text, dict)]
    residual_raw_mask, residual_expanded_mask = _augment_inpaint_masks_from_texts(
        residual_mask,
        residual_mask,
        texts_for_mask,
        original_rgb,
    )
    residual_mask, density_guard_source = _density_guarded_inpaint_mask(
        residual_raw_mask,
        residual_expanded_mask,
        original_rgb.shape[:2],
    )
    if density_guard_source != "expanded_mask":
        ocr_page["_strip_fast_fill_residual_density_guard_source"] = density_guard_source
    mask_pixels = int(np.count_nonzero(residual_mask))
    ocr_page["_strip_fast_fill_residual_mask_pixels"] = mask_pixels
    if mask_pixels <= 0:
        _append_inpaint_decision_flag(ocr_page, "text_residual_after_fast_fill")
        _append_inpaint_decision_flag(ocr_page, "fast_fill_residual_mask_missing")
        return cleaned_rgb, False, None
    critical_outside, outside_pixels, outside_ratio = _mask_exceeds_bubble_limit(
        ocr_page,
        residual_mask,
        original_rgb.shape[:2],
    )
    if critical_outside:
        ocr_page["_strip_fast_fill_residual_outside_balloon_pixels"] = int(outside_pixels)
        ocr_page["_strip_fast_fill_residual_outside_balloon_ratio"] = round(float(outside_ratio), 6)
        _append_inpaint_decision_flag(ocr_page, "mask_outside_balloon_critical")
        _append_inpaint_decision_flag(ocr_page, "real_inpaint_skipped_unsafe_mask")
        return cleaned_rgb, False, residual_mask

    try:
        from vision_stack.runtime import (
            _apply_post_inpaint_cleanup_timed,
            _clamp_image_to_limit_mask,
            _get_inpainter,
        )

        started = time.perf_counter()
        inpainter = _get_inpainter("quality")
        repaired = inpainter.inpaint(original_rgb, residual_mask, batch_size=4, force_no_tiling=True)
        try:
            band_y_top = int(ocr_page.get("_band_y_top") or 0)
        except Exception:
            band_y_top = 0
        texts = _texts_with_band_local_bboxes(
            [dict(text) for text in _processable_texts_for_inpaint(ocr_page)],
            width=original_rgb.shape[1],
            height=original_rgb.shape[0],
            band_y_top=band_y_top,
        )
        repaired, raw_limit_pixels, raw_changed_outside = _clamp_image_to_limit_mask(
            original_rgb,
            repaired,
            residual_mask,
            texts,
        )
        repaired = _fill_white_balloon_residual_mask(repaired, residual_mask, texts)
        ocr_page["_strip_raw_limit_mask_pixels"] = int(raw_limit_pixels)
        ocr_page["_strip_raw_changed_outside_limit_mask"] = int(raw_changed_outside)
        ocr_page["_t_lama_total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        repaired, cleanup_stats = _apply_post_inpaint_cleanup_timed(
            original_rgb,
            repaired,
            texts,
            limit_mask=residual_mask,
        )
        repaired = _fill_white_balloon_residual_mask(repaired, residual_mask, texts)
        repaired, dark_glyph_cleanup_pixels = _cleanup_dark_glyph_residuals_in_text_mask(
            repaired,
            residual_mask,
            texts,
        )
        if dark_glyph_cleanup_pixels:
            ocr_page["_strip_dark_glyph_residual_cleanup_pixels"] = int(dark_glyph_cleanup_pixels)
        ocr_page.update(cleanup_stats)
        ocr_page["_strip_used_real_inpaint"] = True
        ocr_page["_strip_used_post_cleanup"] = True
        ocr_page["_strip_fast_fill_residual_real_inpaint"] = True
        ocr_page["_strip_fast_fill_residual_mask_source"] = source
        if "dark_residual_pixels" in {str(flag) for flag in residual.get("flags") or []}:
            ocr_page["_strip_dark_residual_retry"] = True
            ocr_page["_strip_dark_residual_retry_from_fast_fill_residual"] = True
            ocr_page["_strip_dark_residual_retry_mask_pixels"] = int(mask_pixels)
        return repaired, True, residual_mask
    except Exception as exc:
        _append_inpaint_decision_flag(ocr_page, "text_residual_after_fast_fill")
        _append_inpaint_decision_flag(ocr_page, "real_inpaint_unavailable")
        ocr_page["_strip_fast_fill_residual_real_inpaint_error"] = f"{type(exc).__name__}: {exc}"
        return cleaned_rgb, False, residual_mask


def _try_metadata_background_text_fill(image_rgb: np.ndarray, text: dict) -> np.ndarray | None:
    if not _fast_metadata_background_fill_enabled():
        return None
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return None
    color = _metadata_background_color(text)
    if color is None:
        return None
    mask = _text_geometry_mask(width, height, text)
    if mask is None:
        return None
    if _metadata_background_is_stale_for_current_region(image_rgb, mask, color):
        return None
    sample_bbox = (
        _local_context_bbox_for_text(text, width, height)
        or _normalize_bbox(text.get("layout_bbox"), width, height)
        or _normalize_bbox(text.get("bbox"), width, height)
    )
    if sample_bbox is not None and _looks_translucent_or_textured_background(image_rgb, sample_bbox, mask):
        return None
    mask_area = int(np.count_nonzero(mask))
    if mask_area > int(width * height * 0.35):
        return None

    bg_i = color.astype(np.int16)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    ring = cv2.dilate(mask, kernel, iterations=1)
    ring = ((ring > 0) & (mask == 0))
    if int(np.count_nonzero(ring)) >= 32:
        ring_pixels = image_rgb[ring].astype(np.int16)
        ring_delta = np.mean(np.abs(ring_pixels - bg_i[None, :]), axis=1)
        if float(np.mean(ring_delta <= 28.0)) < 0.35:
            return None

    text_pixels = image_rgb[mask > 0].astype(np.int16)
    if text_pixels.size == 0:
        return None
    text_delta = np.mean(np.abs(text_pixels - bg_i[None, :]), axis=1)
    if float(np.percentile(text_delta, 90)) < 24.0:
        return None

    result = image_rgb.copy()
    result[mask > 0] = color
    return result


def _apply_fast_local_balloon_fill(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    rejection_reasons: dict[str, int] = {}

    def _reject(reason: str) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    def _record(stats: dict) -> dict:
        ocr_page["_strip_fast_local_balloon_count"] = stats["local_balloon_count"]
        ocr_page["_strip_remaining_inpaint_blocks"] = stats["remaining_blocks"]
        ocr_page["_strip_fast_local_rejection_reasons"] = dict(rejection_reasons)
        return stats

    fast_local_enabled = _fast_local_balloon_fill_enabled()
    if not fast_local_enabled or not vision_blocks:
        text_count = len([text for text in ocr_page.get("texts", []) if isinstance(text, dict)])
        rejection_reasons["disabled" if not fast_local_enabled else "no_vision_blocks"] = max(1, text_count)
        return band_rgb, vision_blocks, _record({"local_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    from vision_stack.runtime import _try_koharu_balloon_fill

    height, width = band_rgb.shape[:2]
    result = band_rgb.copy()
    filled_bboxes: list[list[int]] = []
    filled_keys: set[tuple[int, int, int, int]] = set()
    filled_mask = np.zeros((height, width), dtype=np.uint8)

    for text in ocr_page.get("texts", []):
        if _is_rotated_recovery_text(text):
            _reject("rotated_recovery_real_inpaint_required")
            continue
        rejection_reason = _fast_local_rejection_reason(text)
        if rejection_reason:
            if rejection_reason.startswith("mask_evidence:"):
                _propagate_existing_mask_evidence_decision_flags(ocr_page, text)
            _reject(rejection_reason)
            continue
        text_bbox = _line_polygons_bbox(text, width, height) or _normalize_bbox(
            text.get("text_pixel_bbox"),
            width,
            height,
        )
        fill_bbox = None
        if text_bbox is None:
            _reject("missing_text_bbox")
            continue
        real_bubble_mask, bubble_rejection = _real_bubble_mask_for_text(ocr_page, text, width, height)
        if real_bubble_mask is None:
            _reject(bubble_rejection)
            continue
        real_bubble_bbox = _bbox_from_binary_mask(real_bubble_mask)
        if real_bubble_bbox is not None:
            fill_bbox = real_bubble_bbox
        if fill_bbox is None:
            _reject("missing_real_bubble_mask")
            continue
        fill_key = tuple(fill_bbox)
        if fill_key in filled_keys:
            continue
        candidate_mask = _text_geometry_mask(width, height, text)
        if candidate_mask is None:
            candidate_mask = _mask_from_bbox(width, height, text_bbox)
        if not any(
            _block_is_covered_by_fast_fill(block, [fill_bbox], width, height, candidate_mask)
            for block in vision_blocks
        ):
            _reject("no_covered_vision_block")
            continue

        mask = _mask_from_bbox(width, height, text_bbox)
        filled = _try_koharu_balloon_fill(result, mask)
        if filled is None:
            filled = _try_solid_background_text_fill(result, text_bbox, fill_bbox)
        if filled is None:
            filled = _try_metadata_background_text_fill(result, text)
        if filled is None:
            _reject("no_flat_fill")
            continue
        safe_bubble = _safe_real_bubble_interior_mask(real_bubble_mask, width, height)
        text_limited_mask, clip_rejection = _clip_fast_fill_text_mask_to_real_bubble(
            candidate_mask,
            real_bubble_mask,
            width,
            height,
        )
        if text_limited_mask is None or not np.any(text_limited_mask):
            _reject(clip_rejection or "text_mask_outside_bubble")
            continue
        changed_mask = (
            np.any(filled != result, axis=2)
            & (safe_bubble > 0)
            & (text_limited_mask > 0)
        ).astype(np.uint8) * 255
        if not np.any(changed_mask):
            _reject("no_fast_fill_change")
            continue

        clamped = result.copy()
        clamped[changed_mask > 0] = filled[changed_mask > 0]
        result = clamped
        filled_mask = np.maximum(filled_mask, changed_mask)
        filled_bboxes.append(fill_bbox)
        filled_keys.add(fill_key)

    if not filled_bboxes:
        return band_rgb, vision_blocks, _record({"local_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    remaining_blocks = [
        block
        for block in vision_blocks
        if not _block_is_covered_by_fast_fill(block, filled_bboxes, width, height, filled_mask)
    ]
    stats = {
        "local_balloon_count": len(filled_bboxes),
        "remaining_blocks": len(remaining_blocks),
    }
    return result, remaining_blocks, _record(stats)

def _real_bubble_mask_for_koharu_fill(
    ocr_page: dict,
    block: dict,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, str]:
    for key in ("bubble_mask", "bubbleMask", "balloon_mask", "balloonMask"):
        mask = block.get(key)
        if isinstance(mask, np.ndarray) and mask.shape[:2] == (height, width) and np.any(mask):
            return mask.astype(np.uint8), ""

    mask, reason = _real_bubble_mask_for_text(ocr_page, block, width, height)
    if isinstance(mask, np.ndarray) and np.any(mask):
        return _coerce_mask_for_shape(mask, (height, width)), ""
    return None, reason or "missing_real_bubble_mask"


def _apply_koharu_bubble_fast_fill_to_blocks(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray, dict]:
    height, width = band_rgb.shape[:2]
    working_rgb = band_rgb.copy()
    fast_fill_mask = np.zeros((height, width), dtype=np.uint8)
    remaining_mask = np.zeros((height, width), dtype=np.uint8)
    remaining_blocks: list[dict] = []
    samples: list[dict] = []
    rejection_reasons: dict[str, int] = {}
    filled_total = 0

    for block in vision_blocks:
        if not isinstance(block, dict) or _text_suppressed_for_inpaint(block) or _route_action_blocks_inpaint(block):
            continue
        if (
            str(block.get("route_action") or "").strip().lower() == "review_required"
            and _item_has_current_inpaint_mask_evidence(block)
        ):
            remaining_blocks.append(dict(block))
            reason = "review_required_real_inpaint"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            samples.append(
                {
                    "text": block.get("text") or block.get("original"),
                    "bbox": block.get("bbox"),
                    "bubble_id": block.get("bubble_id") or block.get("bubbleId"),
                    "filled_pixels": 0,
                    "remaining_pixels": 0,
                    "reason": reason,
                }
            )
            continue
        text_mask = build_inpaint_mask(block, band_rgb.shape, image_rgb=working_rgb)
        if not isinstance(text_mask, np.ndarray) or not np.any(text_mask):
            reason = "missing_glyph_text_mask"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue

        bubble_mask, bubble_reason = _real_bubble_mask_for_koharu_fill(ocr_page, block, width, height)
        if isinstance(bubble_mask, np.ndarray) and np.any(bubble_mask):
            bubble_mask = _coerce_mask_for_shape(bubble_mask, (height, width))
            text_for_overlap = _coerce_mask_for_shape(text_mask, (height, width))
            overlap_pixels = int(np.count_nonzero((text_for_overlap > 0) & (bubble_mask > 0)))
            text_pixels_for_overlap = int(np.count_nonzero(text_for_overlap))
            if text_pixels_for_overlap > 0 and overlap_pixels < int(text_pixels_for_overlap * 0.72):
                fallback_bubble = _fallback_dark_bubble_mask_from_block_bbox(block, width, height)
                if isinstance(fallback_bubble, np.ndarray) and np.any(fallback_bubble):
                    bubble_mask = fallback_bubble.astype(np.uint8)
                    metrics = block.setdefault("qa_metrics", {})
                    if isinstance(metrics, dict):
                        metrics["koharu_fast_fill_bubble_mask_replaced_for_text_contract"] = {
                            "text_pixels": int(text_pixels_for_overlap),
                            "overlap_pixels": int(overlap_pixels),
                        }
                    _append_qa_flag_to_item(block, "koharu_fast_fill_bubble_mask_replaced_for_text_contract")
            text_mask_for_gate = _coerce_mask_for_shape(text_mask, (height, width))
            text_pixels = int(np.count_nonzero(text_mask_for_gate))
            outside_pixels = int(np.count_nonzero((text_mask_for_gate > 0) & (bubble_mask == 0)))
            outside_ratio = outside_pixels / float(max(1, text_pixels))
            if outside_pixels > OUTSIDE_BALLOON_CRITICAL_PIXELS and outside_ratio >= OUTSIDE_BALLOON_CRITICAL_RATIO:
                _append_qa_flag_to_item(block, "mask_outside_balloon")
                _append_qa_flag_to_item(block, "mask_outside_balloon_critical")
                _append_inpaint_decision_flag(ocr_page, "mask_outside_balloon_critical")
                _append_inpaint_decision_flag(ocr_page, "real_inpaint_skipped_unsafe_mask")
                reason = "mask_outside_balloon_critical"
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                samples.append(
                    {
                        "text": block.get("text") or block.get("original"),
                        "bbox": block.get("bbox"),
                        "bubble_id": block.get("bubble_id") or block.get("bubbleId"),
                        "filled_pixels": 0,
                        "remaining_pixels": 0,
                        "reason": reason,
                    }
                )
                continue
        fast_image, block_remaining, metadata = apply_koharu_bubble_fast_fill(
            working_rgb,
            text_mask,
            bubble_mask,
        )
        metadata_reason = str(metadata.get("reason") or bubble_reason or "").strip().lower()
        block_source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
        if metadata_reason == "missing_bubble_mask" and block_source in {
            "image_dark_panel_mask",
            "derived_card_panel_mask",
        }:
            contract_mask = _dark_text_contract_fill_mask(block, width, height, working_rgb)
            if not (isinstance(contract_mask, np.ndarray) and np.any(contract_mask)):
                contract_bbox = (
                    _normalize_bbox(block.get("source_text_mask_bbox"), width, height)
                    or _normalize_bbox(block.get("_source_text_mask_bbox"), width, height)
                    or _normalize_bbox(block.get("text_pixel_bbox"), width, height)
                    or _normalize_bbox(block.get("bbox"), width, height)
                )
                if contract_bbox is not None:
                    contract_mask = _mask_from_bbox(width, height, contract_bbox, padding=6)
                    metrics = block.setdefault("qa_metrics", {})
                    if isinstance(metrics, dict):
                        metrics["koharu_missing_bubble_bbox_contract_fill_mask"] = {
                            "bbox": [int(v) for v in contract_bbox],
                            "mask_pixels": int(np.count_nonzero(contract_mask)),
                            "source": "text_bbox_fallback",
                        }
            if isinstance(contract_mask, np.ndarray) and np.any(contract_mask):
                visual_contract = _dark_panel_visual_contract_fill_mask(
                    working_rgb,
                    block,
                    contract_mask,
                    width,
                    height,
                )
                block_remaining = (
                    visual_contract.astype(np.uint8)
                    if isinstance(visual_contract, np.ndarray) and np.any(visual_contract)
                    else contract_mask.astype(np.uint8)
                )
                metadata = dict(metadata)
                metadata["reason"] = "visual_contract_missing_bubble_mask"
                metrics = block.setdefault("qa_metrics", {})
                if isinstance(metrics, dict):
                    metrics["koharu_missing_bubble_visual_contract_fill_mask"] = {
                        "mask_pixels": int(np.count_nonzero(block_remaining)),
                        "source": "dark_panel_text_contract",
                    }
        fill_rgb = None
        fill_reason = str(metadata.get("reason") or metadata_reason or "").strip().lower()
        if fill_reason == "visual_contract_missing_bubble_mask" and block_source in {
            "image_dark_panel_mask",
            "derived_card_panel_mask",
        }:
            fill_rgb = _sample_dark_panel_contract_fill_rgb(
                working_rgb,
                block_remaining,
                block,
                width,
                height,
            )
        if fill_rgb is None:
            fill_rgb = _dark_text_contract_fill_rgb(block)
        if fill_rgb is not None:
            unresolved = _coerce_mask_for_shape(block_remaining, (height, width)) > 0
            if np.any(unresolved):
                fast_image = fast_image.copy()
                fast_image[unresolved] = fill_rgb
                metadata = dict(metadata)
                metadata["text_contract_direct_dark_fill_pixels"] = int(np.count_nonzero(unresolved))
                metadata["text_contract_direct_dark_fill_rgb"] = [int(v) for v in fill_rgb.tolist()]
                block_remaining = np.zeros((height, width), dtype=np.uint8)
                _append_qa_flag_to_item(block, "text_contract_direct_dark_fill")
        filled_pixels = int(metadata.get("filled_pixels") or 0)
        direct_contract_pixels = int(metadata.get("text_contract_direct_dark_fill_pixels") or 0)
        if filled_pixels:
            changed = np.any(fast_image != working_rgb, axis=2)
            fast_fill_mask[changed] = 255
            working_rgb = fast_image
            filled_total += filled_pixels
        elif direct_contract_pixels:
            changed = np.any(fast_image != working_rgb, axis=2)
            fast_fill_mask[changed] = 255
            working_rgb = fast_image
            filled_total += direct_contract_pixels
        else:
            reason = str(metadata.get("reason") or bubble_reason or "fast_fill_not_applicable")
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        block_remaining = _coerce_mask_for_shape(block_remaining, (height, width))
        if np.any(block_remaining):
            remaining = dict(block)
            remaining["_koharu_remaining_mask_pixels"] = int(np.count_nonzero(block_remaining))
            remaining_blocks.append(remaining)
            remaining_mask = cv2.bitwise_or(remaining_mask, block_remaining.astype(np.uint8))

        samples.append(
            {
                "text": block.get("text") or block.get("original"),
                "bbox": block.get("bbox"),
                "bubble_id": block.get("bubble_id") or block.get("bubbleId"),
                "filled_pixels": filled_pixels + direct_contract_pixels,
                "remaining_pixels": int(np.count_nonzero(block_remaining)),
                "reason": metadata.get("reason") or bubble_reason or "",
            }
        )

    metadata = {
        "filled_pixels": int(filled_total),
        "remaining_pixels": int(np.count_nonzero(remaining_mask)),
        "samples": samples,
        "rejection_reasons": rejection_reasons,
    }
    return working_rgb, remaining_blocks, fast_fill_mask, remaining_mask, metadata


def prewarm_band_inpainter(profile: str = "quality"):
    """Carrega o inpainter pesado cedo para sobrepor inicializacao com OCR."""
    from vision_stack.runtime import _get_inpainter

    inpainter = _get_inpainter(profile)
    inpaint = getattr(inpainter, "inpaint", None)
    if callable(inpaint):
        dummy_rgb = np.full((128, 128, 3), 255, dtype=np.uint8)
        dummy_mask = np.zeros(dummy_rgb.shape[:2], dtype=np.uint8)
        dummy_mask[56:72, 56:72] = 255
        try:
            inpaint(dummy_rgb, dummy_mask, batch_size=1, force_no_tiling=True)
        except Exception:
            pass
    return inpainter


def inpaint_band_image(band_rgb: np.ndarray, ocr_page: dict) -> np.ndarray:
    """Aplica o mesmo round de inpaint do runtime principal na banda do strip."""
    from vision_stack.runtime import (
        _apply_inpainting_round,
        _apply_white_balloon_residual_force_fill,
        _build_post_cleanup_limit_mask,
        _apply_post_inpaint_cleanup_timed,
        _clamp_image_to_limit_mask,
        _get_inpainter,
        _has_white_balloon_text_residual,
    )

    if band_rgb.size == 0 or not ocr_page.get("texts"):
        return band_rgb.copy()

    height, width = band_rgb.shape[:2]
    try:
        band_y_top = int(ocr_page.get("_band_y_top") or 0)
    except Exception:
        band_y_top = 0
    texts_for_inpaint = _texts_with_band_local_bboxes(
        [dict(text) for text in list(ocr_page.get("texts") or []) if isinstance(text, dict)],
        width=width,
        height=height,
        band_y_top=band_y_top,
    )
    texts_for_inpaint = [
        text for text in texts_for_inpaint if not _text_has_rejected_bubble_without_glyph_evidence(text)
    ]
    if texts_for_inpaint:
        ocr_page["texts"] = texts_for_inpaint
    vision_blocks = _texts_with_band_local_bboxes(
        [dict(block) for block in list(ocr_page.get("_vision_blocks") or []) if isinstance(block, dict)],
        width=width,
        height=height,
        band_y_top=band_y_top,
    )
    if not vision_blocks:
        vision_blocks = _build_fallback_vision_blocks({"texts": texts_for_inpaint}, width, height)
    vision_blocks = _enrich_vision_blocks_from_texts_for_inpaint(
        vision_blocks,
        texts_for_inpaint,
        width,
        height,
    )
    promoted_before = len(vision_blocks)
    vision_blocks = _append_missing_text_inpaint_blocks(
        vision_blocks,
        texts_for_inpaint,
        width,
        height,
        band_rgb,
    )
    promoted_count = max(0, len(vision_blocks) - promoted_before)
    if promoted_count:
        ocr_page["_strip_promoted_missing_text_inpaint_blocks"] = int(promoted_count)
    if vision_blocks:
        processable_vision_blocks = _processable_vision_blocks_for_inpaint(vision_blocks)
        ignored_count = len(vision_blocks) - len(processable_vision_blocks)
        if ignored_count:
            ocr_page["_strip_nonprocessable_remaining_block_count"] = int(ignored_count)
            _append_inpaint_decision_flag(ocr_page, "nonprocessable_remaining_blocks_ignored")
        vision_blocks = processable_vision_blocks
    if not vision_blocks:
        recovered_blocks: list[dict] = []
        recovery_candidates = _texts_with_band_local_bboxes(
            [dict(text) for text in list(ocr_page.get("texts") or []) if isinstance(text, dict)],
            width=width,
            height=height,
            band_y_top=band_y_top,
        )
        pre_recovery_samples: list[dict] = []
        recovery_attempts: list[dict] = []
        for text in recovery_candidates:
            if len(pre_recovery_samples) < 8:
                pre_recovery_samples.append(
                    {
                        "id": text.get("id") or text.get("text_id"),
                        "trace_id": text.get("trace_id"),
                        "band_id": text.get("band_id"),
                        "bbox": text.get("bbox"),
                        "text_pixel_bbox": text.get("text_pixel_bbox"),
                        "balloon_bbox": text.get("balloon_bbox"),
                        "bubble_mask_bbox": text.get("bubble_mask_bbox"),
                        "bubble_mask_source": text.get("bubble_mask_source"),
                        "bubble_mask_error": text.get("bubble_mask_error"),
                        "qa_flags": list(text.get("qa_flags") or []),
                        "route_action": text.get("route_action"),
                        "mask_evidence": text.get("mask_evidence"),
                        "_band_local_bbox_normalized": text.get("_band_local_bbox_normalized"),
                    }
                )
            attempt = {
                "id": text.get("id") or text.get("text_id"),
                "trace_id": text.get("trace_id"),
                "bbox": text.get("bbox"),
                "text_pixel_bbox": text.get("text_pixel_bbox"),
                "bubble_mask_source": text.get("bubble_mask_source"),
                "qa_flags": list(text.get("qa_flags") or []),
                "route_action": text.get("route_action"),
            }
            if _text_suppressed_for_inpaint(text) or _route_action_blocks_inpaint(text):
                attempt["result"] = "skipped_route_or_suppressed"
                if len(recovery_attempts) < 8:
                    recovery_attempts.append(attempt)
                continue
            if _translator_note_text_only_mask(text) and not _translator_note_has_text_geometry(text):
                attempt["result"] = "skipped_translator_note_text_only_without_geometry"
                if len(recovery_attempts) < 8:
                    recovery_attempts.append(attempt)
                continue
            try:
                recovered_mask = build_inpaint_mask(text, band_rgb.shape, band_rgb)
            except Exception as exc:
                recovered_mask = None
                attempt["result"] = "exception"
                attempt["error"] = str(exc)
            if isinstance(recovered_mask, np.ndarray) and np.any(recovered_mask):
                attempt["result"] = "recovered"
                attempt["mask_pixels"] = int(np.count_nonzero(recovered_mask))
                recovered_blocks.append(text)
            elif isinstance(recovered_mask, np.ndarray):
                attempt["result"] = "zero_mask"
                attempt["mask_pixels"] = 0
            else:
                attempt.setdefault("result", "no_mask")
            if len(recovery_attempts) < 8:
                recovery_attempts.append(attempt)
        if pre_recovery_samples:
            ocr_page["_strip_pre_recovery_text_samples"] = pre_recovery_samples
        if recovery_attempts:
            ocr_page["_strip_initial_empty_recovery_attempts"] = recovery_attempts
        if recovered_blocks:
            vision_blocks = recovered_blocks
            ocr_page["_strip_recovered_initial_empty_vision_blocks_from_texts"] = True
            ocr_page["_strip_remaining_inpaint_blocks"] = len(vision_blocks)
        else:
            if _ocr_page_has_unsafe_auto_inpaint_evidence(ocr_page):
                _append_inpaint_decision_flag(ocr_page, "real_inpaint_skipped_unsafe_mask")
            ocr_page["_strip_remaining_inpaint_blocks"] = 0
            ocr_page["_strip_used_real_inpaint"] = False
            cleaned_rgb, _white_fill_count = _apply_unsafe_white_balloon_text_fills(band_rgb, ocr_page)
            cleaned_rgb, _dark_fill_count = _apply_dark_panel_text_fills(cleaned_rgb, ocr_page)
            return cleaned_rgb.copy()

    ocr_page["_strip_used_fast_solid_fill"] = False
    ocr_page["_strip_used_fast_white_fill"] = False
    ocr_page["_strip_used_fast_dark_fill"] = False
    ocr_page["_strip_used_fast_local_fill"] = False
    ocr_page["_strip_used_real_inpaint"] = False
    ocr_page["_strip_used_post_cleanup"] = False
    ocr_page.setdefault("_strip_fast_solid_balloon_count", 0)
    ocr_page.setdefault("_strip_fast_solid_white_count", 0)
    ocr_page.setdefault("_strip_fast_solid_black_count", 0)
    ocr_page.setdefault("_strip_fast_solid_colored_count", 0)
    ocr_page.setdefault("_strip_fast_white_balloon_count", 0)
    ocr_page.setdefault("_strip_fast_local_balloon_count", 0)
    ocr_page.setdefault("_strip_flat_ui_prefill_count", 0)
    ocr_page.setdefault("_strip_remaining_inpaint_blocks", len(vision_blocks))

    _prime_mask_evidence_for_fast_fill(ocr_page, vision_blocks, band_rgb)
    _promote_visually_light_dark_bubbles_to_white(band_rgb, ocr_page, vision_blocks)
    vision_blocks = _filter_unsafe_auto_inpaint_blocks(ocr_page, vision_blocks, band_rgb)
    ocr_page["_strip_remaining_inpaint_blocks"] = len(vision_blocks)
    unresolved_dark_bubble_text_after_prefill = any(
        isinstance(text, dict)
        and not text.get("_fast_fill_inpaint_resolved")
        and not _text_suppressed_for_inpaint(text)
        and not _route_action_blocks_inpaint(text)
        and str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower() == "image_dark_bubble_mask"
        and bool(text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("bbox"))
        for text in ocr_page.get("texts", []) if isinstance(ocr_page, dict)
    )
    skip_empty_rebuild_after_prefill = (
        int(ocr_page.get("_strip_flat_ui_prefill_count") or 0) > 0
        or (
            int(ocr_page.get("_strip_fast_dark_panel_fill_count") or 0) > 0
            and not unresolved_dark_bubble_text_after_prefill
        )
        or bool(ocr_page.get("_strip_used_koharu_fast_fill"))
    )
    if not vision_blocks and not skip_empty_rebuild_after_prefill:
        try:
            band_y_top = int(ocr_page.get("_band_y_top") or 0)
        except Exception:
            band_y_top = 0
        rebuilt_local_texts = _texts_with_band_local_bboxes(
            [dict(text) for text in list(ocr_page.get("texts", [])) if isinstance(text, dict)],
            width=width,
            height=height,
            band_y_top=band_y_top,
        )
        rebuilt_local_texts = [
            text
            for text in rebuilt_local_texts
            if not _text_suppressed_for_inpaint(text)
            and not _route_action_blocks_inpaint(text)
            and not text.get("_fast_fill_inpaint_resolved")
            and (not _translator_note_text_only_mask(text) or _translator_note_has_text_geometry(text))
            and not _text_has_rejected_bubble_without_glyph_evidence(text)
        ]
        if rebuilt_local_texts:
            from vision_stack.runtime import vision_blocks_to_mask

            rebuilt_remaining = vision_blocks_to_mask(
                band_rgb.shape,
                rebuilt_local_texts,
                image_rgb=band_rgb,
                expand_mask=False,
                **_cjk_mask_kwargs_for_strip_page(ocr_page),
            )
            if np.any(rebuilt_remaining):
                vision_blocks = rebuilt_local_texts
                ocr_page["_strip_rebuilt_empty_remaining_blocks_from_local_texts"] = True
                ocr_page["_strip_remaining_inpaint_blocks"] = len(vision_blocks)
                ocr_page["_strip_koharu_remaining_mask_pixels"] = int(np.count_nonzero(rebuilt_remaining))
    if not vision_blocks and not skip_empty_rebuild_after_prefill:
        ocr_page["_strip_used_real_inpaint"] = False
        cleaned_rgb, _white_fill_count = _apply_unsafe_white_balloon_text_fills(band_rgb, ocr_page)
        cleaned_rgb, _dark_fill_count = _apply_dark_panel_text_fills(cleaned_rgb, ocr_page)
        _write_strip_inpaint_debug(
            ocr_page,
            original_rgb=band_rgb,
            working_rgb=band_rgb,
            cleaned_rgb=cleaned_rgb,
            vision_blocks=[],
            used_real_inpaint=False,
            fast_fill_mask=np.zeros((height, width), dtype=np.uint8),
            raw_mask=np.zeros((height, width), dtype=np.uint8),
            expanded_mask=np.zeros((height, width), dtype=np.uint8),
        )
        return cleaned_rgb.copy()

    working_rgb, vision_blocks, flat_ui_meta = _apply_flat_ui_text_prefill_to_blocks(
        band_rgb,
        ocr_page,
        vision_blocks,
    )
    ocr_page["_strip_flat_ui_prefill_count"] = int(flat_ui_meta.get("flat_ui_prefill_count") or 0)
    ocr_page["_strip_remaining_inpaint_blocks"] = len(vision_blocks)

    working_rgb, vision_blocks, dark_fill_meta = _apply_fast_dark_panel_text_fill(
        working_rgb,
        ocr_page,
        vision_blocks,
    )
    ocr_page["_strip_fast_dark_panel_fill_count"] = int(dark_fill_meta.get("dark_panel_fill_count") or 0)
    ocr_page["_strip_remaining_inpaint_blocks"] = len(vision_blocks)
    cleanup_fill_count = _apply_clipped_overlap_fragment_cleanup_fill(working_rgb, ocr_page)
    if cleanup_fill_count:
        previous_count = int(ocr_page.get("_strip_fast_dark_panel_fill_count") or 0)
        ocr_page["_strip_fast_dark_panel_fill_count"] = previous_count + cleanup_fill_count

    working_rgb, vision_blocks, koharu_fast_fill_mask, koharu_remaining_mask, koharu_meta = (
        _apply_koharu_bubble_fast_fill_to_blocks(working_rgb, ocr_page, vision_blocks)
    )
    ocr_page["_strip_used_koharu_fast_fill"] = bool(koharu_meta.get("filled_pixels"))
    ocr_page["_strip_koharu_fast_fill_pixels"] = int(koharu_meta.get("filled_pixels") or 0)
    ocr_page["_strip_koharu_remaining_mask_pixels"] = int(koharu_meta.get("remaining_pixels") or 0)
    ocr_page["_strip_koharu_fast_fill_samples"] = list(koharu_meta.get("samples") or [])
    ocr_page["_strip_koharu_fast_fill_reject_reasons"] = dict(koharu_meta.get("rejection_reasons") or {})
    ocr_page["_strip_remaining_inpaint_blocks"] = len(vision_blocks)
    unresolved_dark_bubble_text_after_prefill = any(
        isinstance(text, dict)
        and not text.get("_fast_fill_inpaint_resolved")
        and not _text_suppressed_for_inpaint(text)
        and not _route_action_blocks_inpaint(text)
        and str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower() == "image_dark_bubble_mask"
        and bool(text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("bbox"))
        for text in ocr_page.get("texts", []) if isinstance(ocr_page, dict)
    )
    skip_empty_rebuild_after_prefill = (
        int(ocr_page.get("_strip_flat_ui_prefill_count") or 0) > 0
        or (
            int(ocr_page.get("_strip_fast_dark_panel_fill_count") or 0) > 0
            and not unresolved_dark_bubble_text_after_prefill
        )
        or bool(ocr_page.get("_strip_used_koharu_fast_fill"))
    )
    if not vision_blocks and not skip_empty_rebuild_after_prefill:
        try:
            band_y_top = int(ocr_page.get("_band_y_top") or 0)
        except Exception:
            band_y_top = 0
        rebuilt_local_texts = _texts_with_band_local_bboxes(
            [dict(text) for text in list(ocr_page.get("texts", [])) if isinstance(text, dict)],
            width=width,
            height=height,
            band_y_top=band_y_top,
        )
        rebuilt_local_texts = [
            text
            for text in rebuilt_local_texts
            if not _text_suppressed_for_inpaint(text)
            and not _route_action_blocks_inpaint(text)
            and not text.get("_fast_fill_inpaint_resolved")
            and (not _translator_note_text_only_mask(text) or _translator_note_has_text_geometry(text))
            and not _text_has_rejected_bubble_without_glyph_evidence(text)
        ]
        if rebuilt_local_texts:
            from vision_stack.runtime import vision_blocks_to_mask

            rebuilt_remaining = vision_blocks_to_mask(
                working_rgb.shape,
                rebuilt_local_texts,
                image_rgb=working_rgb,
                expand_mask=False,
                **_cjk_mask_kwargs_for_strip_page(ocr_page),
            )
            if np.any(rebuilt_remaining):
                vision_blocks = rebuilt_local_texts
                koharu_remaining_mask = rebuilt_remaining
                ocr_page["_strip_rebuilt_empty_remaining_blocks_from_local_texts"] = True
                ocr_page["_strip_remaining_inpaint_blocks"] = len(vision_blocks)
                ocr_page["_strip_koharu_remaining_mask_pixels"] = int(np.count_nonzero(koharu_remaining_mask))

    fast_fill_mask = (np.any(working_rgb != band_rgb, axis=2).astype(np.uint8) * 255)
    if not vision_blocks:
        if int(ocr_page.get("_strip_flat_ui_prefill_count") or 0) > 0:
            ocr_page["_strip_remaining_inpaint_blocks"] = 0
            ocr_page["_strip_used_real_inpaint"] = False
            _write_strip_inpaint_debug(
                ocr_page,
                original_rgb=band_rgb,
                working_rgb=band_rgb,
                cleaned_rgb=working_rgb,
                vision_blocks=[],
                used_real_inpaint=False,
                fast_fill_mask=fast_fill_mask,
                raw_mask=fast_fill_mask,
                expanded_mask=fast_fill_mask,
            )
            return working_rgb.copy()
        if int(ocr_page.get("_strip_fast_dark_panel_fill_count") or 0) > 0:
            ocr_page["_strip_remaining_inpaint_blocks"] = 0
            ocr_page["_strip_used_real_inpaint"] = False
            _write_strip_inpaint_debug(
                ocr_page,
                original_rgb=band_rgb,
                working_rgb=band_rgb,
                cleaned_rgb=working_rgb,
                vision_blocks=[],
                used_real_inpaint=False,
                fast_fill_mask=fast_fill_mask,
                raw_mask=fast_fill_mask,
                expanded_mask=fast_fill_mask,
            )
            return working_rgb.copy()
        if bool(ocr_page.get("_strip_used_koharu_fast_fill")):
            ocr_page["_strip_remaining_inpaint_blocks"] = 0
            ocr_page["_strip_used_real_inpaint"] = False
            return working_rgb.copy()
        _mark_suspicious_fast_fill_without_raw_mask(
            ocr_page,
            fast_fill_mask,
            np.zeros(fast_fill_mask.shape, dtype=np.uint8),
        )
        connected_geometry_fill = int(ocr_page.get("_strip_connected_white_geometry_fill_count") or 0) > 0
        evidence_constrained_fill = int(ocr_page.get("_strip_fast_white_evidence_constrained_fill_count") or 0) > 0
        solid_fast_fill = int(ocr_page.get("_strip_fast_solid_balloon_count") or 0) > 0
        skip_post_cleanup = (
            solid_fast_fill
            or connected_geometry_fill
            or evidence_constrained_fill
            or not _fast_white_post_cleanup_enabled()
        )
        if skip_post_cleanup:
            if solid_fast_fill:
                ocr_page["_strip_post_cleanup_skipped_reason"] = "fast_solid_fill"
            elif connected_geometry_fill:
                ocr_page["_strip_post_cleanup_skipped_reason"] = "connected_white_geometry_fill"
            elif evidence_constrained_fill:
                ocr_page["_strip_post_cleanup_skipped_reason"] = "koharu_evidence_constrained_fast_fill"
            cleaned, used_residual_real_inpaint, residual_real_mask = _apply_real_inpaint_for_fast_fill_residual(
                original_rgb=band_rgb,
                working_rgb=working_rgb,
                cleaned_rgb=working_rgb,
                ocr_page=ocr_page,
                fast_fill_mask=fast_fill_mask,
            )
            cleaned, _ = _apply_dark_panel_text_fills(cleaned, ocr_page)
            _write_strip_inpaint_debug(
                ocr_page,
                original_rgb=band_rgb,
                working_rgb=working_rgb,
                cleaned_rgb=cleaned,
                vision_blocks=[],
                used_real_inpaint=used_residual_real_inpaint,
                fast_fill_mask=fast_fill_mask,
                raw_mask=residual_real_mask,
                expanded_mask=residual_real_mask,
            )
            return cleaned.copy()
        cleaned, cleanup_stats = _apply_post_inpaint_cleanup_timed(
            band_rgb,
            working_rgb,
            list(ocr_page.get("texts", [])),
            limit_mask=fast_fill_mask if np.any(fast_fill_mask) else None,
        )
        ocr_page.update(cleanup_stats)
        ocr_page["_strip_used_post_cleanup"] = True
        cleaned, used_residual_real_inpaint, residual_real_mask = _apply_real_inpaint_for_fast_fill_residual(
            original_rgb=band_rgb,
            working_rgb=working_rgb,
            cleaned_rgb=cleaned,
            ocr_page=ocr_page,
            fast_fill_mask=fast_fill_mask,
        )
        cleaned, _ = _apply_dark_panel_text_fills(cleaned, ocr_page)
        _write_strip_inpaint_debug(
            ocr_page,
            original_rgb=band_rgb,
            working_rgb=working_rgb,
            cleaned_rgb=cleaned,
            vision_blocks=[],
            used_real_inpaint=used_residual_real_inpaint,
            fast_fill_mask=fast_fill_mask,
            raw_mask=residual_real_mask,
            expanded_mask=residual_real_mask,
        )
        return cleaned

    inpaint_payload = dict(ocr_page)
    inpaint_payload["_vision_blocks"] = [_vision_block_for_real_inpaint_payload(block) for block in vision_blocks]
    inpaint_payload["_skip_internal_post_cleanup"] = True
    from vision_stack.runtime import vision_blocks_to_mask
    mask_kwargs = _cjk_mask_kwargs_for_strip_page(ocr_page)
    try:
        band_y_top = int(ocr_page.get("_band_y_top") or 0)
    except Exception:
        band_y_top = 0
    local_texts = _texts_with_band_local_bboxes(
        [dict(text) for text in list(ocr_page.get("texts", [])) if isinstance(text, dict)],
        width=width,
        height=height,
        band_y_top=band_y_top,
    )
    translator_note_fill_texts = [
        dict(text)
        for text in local_texts
        if isinstance(text, dict)
        and _translator_note_text_only_mask(text)
        and _translator_note_has_text_geometry(text)
        and not _text_suppressed_for_inpaint(text)
        and not _route_action_blocks_inpaint(text)
    ]
    original_local_text_count = len(local_texts)
    local_texts = [
        _drop_isolated_side_note_line_polygons(text)
        for text in local_texts
        if not _text_suppressed_for_inpaint(text)
        and not text.get("_fast_fill_inpaint_resolved")
        and (not _translator_note_text_only_mask(text) or _translator_note_has_text_geometry(text))
        and not _text_has_rejected_bubble_without_glyph_evidence(text)
    ]
    ocr_page["_strip_inpaint_local_texts"] = [dict(text) for text in local_texts]
    raw_mask = _coerce_mask_for_shape(koharu_remaining_mask, working_rgb.shape[:2])
    rebuilt_raw_mask = vision_blocks_to_mask(
        working_rgb.shape,
        vision_blocks,
        image_rgb=working_rgb,
        expand_mask=False,
        **mask_kwargs,
    )
    rebuilt_expanded_mask = vision_blocks_to_mask(
        working_rgb.shape,
        vision_blocks,
        image_rgb=working_rgb,
        expand_mask=True,
        **mask_kwargs,
    )
    if not np.any(rebuilt_raw_mask) and local_texts:
        rebuilt_from_texts = vision_blocks_to_mask(
            working_rgb.shape,
            local_texts,
            image_rgb=working_rgb,
            expand_mask=False,
            **mask_kwargs,
        )
        if np.any(rebuilt_from_texts):
            rebuilt_raw_mask = rebuilt_from_texts
            rebuilt_expanded_from_texts = vision_blocks_to_mask(
                working_rgb.shape,
                local_texts,
                image_rgb=working_rgb,
                expand_mask=True,
                **mask_kwargs,
            )
            if np.any(rebuilt_expanded_from_texts):
                rebuilt_expanded_mask = rebuilt_expanded_from_texts
            ocr_page["_strip_rebuilt_raw_mask_from_local_texts"] = True
    raw_mask = _sanitize_precomputed_remaining_mask(
        raw_mask,
        rebuilt_raw_mask,
        original_text_count=original_local_text_count,
        filtered_text_count=len(local_texts),
    )
    if not np.any(raw_mask):
        raw_mask = rebuilt_raw_mask
    expanded_seed_mask = _sanitize_rebuilt_expanded_mask(raw_mask, rebuilt_expanded_mask, raw_mask)
    raw_mask, expanded_mask = _augment_inpaint_masks_from_texts(
        raw_mask,
        expanded_seed_mask,
        local_texts,
        working_rgb,
    )
    expanded_mask = _expand_strip_real_inpaint_mask(
        raw_mask,
        expanded_mask,
        ocr_page,
        local_texts,
        working_rgb,
    )
    late_dark_input = working_rgb
    working_rgb, vision_blocks, late_dark_fill_meta = _apply_fast_dark_panel_text_fill(
        working_rgb,
        ocr_page,
        vision_blocks,
    )
    late_dark_count = int(late_dark_fill_meta.get("dark_panel_fill_count") or 0)
    if late_dark_count:
        previous_late_count = int(ocr_page.get("_strip_late_fast_dark_panel_fill_count") or 0)
        ocr_page["_strip_late_fast_dark_panel_fill_count"] = previous_late_count + late_dark_count
        ocr_page["_strip_fast_dark_panel_fill_count"] = int(
            ocr_page.get("_strip_fast_dark_panel_fill_count") or 0
        )
        ocr_page["_strip_remaining_inpaint_blocks"] = len(vision_blocks)
        late_fast_fill_mask = (np.any(working_rgb != late_dark_input, axis=2).astype(np.uint8) * 255)
        fast_fill_mask = np.maximum(fast_fill_mask, late_fast_fill_mask).astype(np.uint8)
        if not vision_blocks:
            ocr_page["_strip_remaining_inpaint_blocks"] = 0
            ocr_page["_strip_used_real_inpaint"] = False
            _write_strip_inpaint_debug(
                ocr_page,
                original_rgb=band_rgb,
                working_rgb=late_dark_input,
                cleaned_rgb=working_rgb,
                vision_blocks=[],
                used_real_inpaint=False,
                fast_fill_mask=fast_fill_mask,
                raw_mask=fast_fill_mask,
                expanded_mask=fast_fill_mask,
            )
            return working_rgb.copy()

        local_texts = [
            _drop_isolated_side_note_line_polygons(text)
            for text in _texts_with_band_local_bboxes(
                [dict(text) for text in list(ocr_page.get("texts", [])) if isinstance(text, dict)],
                width=width,
                height=height,
                band_y_top=band_y_top,
            )
            if not _text_suppressed_for_inpaint(text)
            and not text.get("_fast_fill_inpaint_resolved")
            and (not _translator_note_text_only_mask(text) or _translator_note_has_text_geometry(text))
            and not _text_has_rejected_bubble_without_glyph_evidence(text)
        ]
        late_note_fill_texts = [
            dict(text)
            for text in local_texts
            if isinstance(text, dict)
            and _translator_note_text_only_mask(text)
            and _translator_note_has_text_geometry(text)
            and not _text_suppressed_for_inpaint(text)
            and not _route_action_blocks_inpaint(text)
        ]
        if late_note_fill_texts:
            translator_note_fill_texts = late_note_fill_texts
        ocr_page["_strip_inpaint_local_texts"] = [dict(text) for text in local_texts]
        raw_mask = vision_blocks_to_mask(
            working_rgb.shape,
            vision_blocks,
            image_rgb=working_rgb,
            expand_mask=False,
            **mask_kwargs,
        )
        rebuilt_expanded_mask = vision_blocks_to_mask(
            working_rgb.shape,
            vision_blocks,
            image_rgb=working_rgb,
            expand_mask=True,
            **mask_kwargs,
        )
        expanded_seed_mask = _sanitize_rebuilt_expanded_mask(raw_mask, rebuilt_expanded_mask, raw_mask)
        raw_mask, expanded_mask = _augment_inpaint_masks_from_texts(
            raw_mask,
            expanded_seed_mask,
            local_texts,
            working_rgb,
        )
    expanded_mask = _expand_strip_real_inpaint_mask(
        raw_mask,
        expanded_mask,
        ocr_page,
        local_texts,
        working_rgb,
    )
    if (
        np.any(expanded_mask)
        and vision_blocks
        and not _ocr_page_has_unsafe_auto_inpaint_evidence(ocr_page, vision_blocks)
    ):
        _clear_stale_unsafe_inpaint_flags_for_current_action(ocr_page, vision_blocks)
    final_action_mask = expanded_mask.copy()
    inpaint_payload["_precomputed_inpaint_mask"] = expanded_mask
    inpainter = _get_inpainter("quality")
    started = time.perf_counter()
    cleaned = _apply_inpainting_round(working_rgb, inpaint_payload, inpainter)
    cleaned, raw_limit_pixels, raw_changed_outside = _clamp_image_to_limit_mask(
        working_rgb,
        cleaned,
        expanded_mask,
        list(ocr_page.get("texts", [])),
        include_text_bboxes=False,
    )
    ocr_page["_strip_raw_limit_mask_pixels"] = int(raw_limit_pixels)
    ocr_page["_strip_raw_changed_outside_limit_mask"] = int(raw_changed_outside)
    ocr_page["_t_lama_total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    round_stats = inpaint_payload.get("_inpaint_round_stats")
    if isinstance(round_stats, dict):
        ocr_page.update(round_stats)
    if (
        np.any(expanded_mask)
        and vision_blocks
        and not _ocr_page_has_unsafe_auto_inpaint_evidence(ocr_page, vision_blocks)
    ):
        _clear_stale_unsafe_inpaint_flags_for_current_action(ocr_page, vision_blocks)
    if _rejected_card_action_mask_allows_real_inpaint(
        ocr_page,
        vision_blocks,
        expanded_mask,
        working_rgb.shape[:2],
    ) or _rejected_card_action_mask_allows_real_inpaint(
        ocr_page,
        None,
        expanded_mask,
        working_rgb.shape[:2],
    ):
        _clear_stale_unsafe_inpaint_flags_for_current_action(ocr_page, vision_blocks)
    unsafe_after_real_inpaint = _ocr_page_has_unsafe_auto_inpaint_evidence(ocr_page, vision_blocks)
    if unsafe_after_real_inpaint and (
        _rejected_card_action_mask_allows_real_inpaint(
            ocr_page,
            vision_blocks,
            expanded_mask,
            working_rgb.shape[:2],
        )
        or _rejected_card_action_mask_allows_real_inpaint(
            ocr_page,
            None,
            expanded_mask,
            working_rgb.shape[:2],
        )
    ):
        _clear_stale_unsafe_inpaint_flags_for_current_action(ocr_page, vision_blocks)
        unsafe_after_real_inpaint = False
    if unsafe_after_real_inpaint:
        _append_inpaint_decision_flag(ocr_page, "real_inpaint_skipped_unsafe_mask")
        cleaned, _white_fill_count = _apply_unsafe_white_balloon_text_fills(working_rgb, ocr_page)
        cleaned, _ = _apply_dark_panel_text_fills(cleaned, ocr_page)
        ocr_page["_strip_used_real_inpaint"] = False
        _write_strip_inpaint_debug(
            ocr_page,
            original_rgb=band_rgb,
            working_rgb=working_rgb,
            cleaned_rgb=cleaned,
            vision_blocks=[],
            used_real_inpaint=False,
            fast_fill_mask=fast_fill_mask,
            raw_mask=raw_mask,
            expanded_mask=expanded_mask,
        )
        return cleaned.copy()
    ocr_page["_strip_used_real_inpaint"] = True
    cleaned, cleanup_stats = _apply_post_inpaint_cleanup_timed(
        band_rgb,
        cleaned,
        list(ocr_page.get("texts", [])),
        limit_mask=expanded_mask,
    )
    ocr_page.update(cleanup_stats)
    ocr_page["_strip_used_post_cleanup"] = True
    texts = local_texts
    residual_ocr_page = dict(ocr_page)
    residual_ocr_page["texts"] = texts
    ocr_page["_strip_residual_texts"] = texts
    cleaned, rotated_residual_pixels = _apply_rotated_recovery_residual_cleanup(band_rgb, cleaned, texts)
    if rotated_residual_pixels:
        ocr_page["_strip_rotated_residual_cleanup_pixels"] = int(rotated_residual_pixels)
    if _has_white_balloon_text_residual(band_rgb, cleaned, texts):
        forced = _apply_white_balloon_residual_force_fill(band_rgb, cleaned, texts)
        forced, force_limit_pixels, force_changed_outside = _clamp_image_to_limit_mask(
            cleaned,
            forced,
            expanded_mask,
            texts,
            include_text_bboxes=False,
        )
        ocr_page["_strip_white_residual_force_fill"] = bool(np.any(forced != cleaned))
        ocr_page["_strip_white_residual_force_fill_limit_pixels"] = int(force_limit_pixels)
        ocr_page["_strip_white_residual_force_fill_changed_outside"] = int(force_changed_outside)
        cleaned = forced
    cleaned, _ = _apply_dark_panel_text_fills(cleaned, ocr_page)
    cleaned, _translator_note_dark_fill_pixels = _apply_translator_note_dark_text_contract_fill(
        cleaned,
        ocr_page,
        texts_override=translator_note_fill_texts,
    )
    residual_check = _detect_inpaint_residual_text(
        band_rgb,
        cleaned,
        expanded_mask,
        raw_mask=raw_mask,
        fast_fill_mask=fast_fill_mask,
        ocr_page=residual_ocr_page,
    )
    if (
        residual_check.get("has_residual")
        and str(residual_check.get("region_source") or "").startswith("text_region_white_balloon")
    ):
        forced = _apply_white_balloon_residual_force_fill(band_rgb, cleaned, texts)
        forced, force_limit_pixels, force_changed_outside = _clamp_image_to_limit_mask(
            cleaned,
            forced,
            expanded_mask,
            texts,
            include_text_bboxes=False,
        )
        force_changed = bool(np.any(forced != cleaned))
        if force_changed:
            ocr_page["_strip_white_residual_force_fill"] = True
            ocr_page["_strip_white_residual_force_fill_from_residual_check"] = True
            ocr_page["_strip_white_residual_force_fill_limit_pixels"] = int(force_limit_pixels)
            ocr_page["_strip_white_residual_force_fill_changed_outside"] = int(force_changed_outside)
            cleaned = forced
            residual_check = _detect_inpaint_residual_text(
                band_rgb,
                cleaned,
                expanded_mask,
                raw_mask=raw_mask,
                fast_fill_mask=fast_fill_mask,
                ocr_page=residual_ocr_page,
            )
    if (
        residual_check.get("has_residual")
        and "dark_residual_pixels" in set(residual_check.get("flags") or [])
        and str(residual_check.get("region_source") or "").startswith("text_region_white_balloon")
        and _all_processable_texts_are_white_balloon(texts, band_rgb)
    ):
        forced = _apply_white_balloon_residual_force_fill(band_rgb, cleaned, texts)
        forced, force_limit_pixels, force_changed_outside = _clamp_image_to_limit_mask(
            cleaned,
            forced,
            expanded_mask,
            texts,
            include_text_bboxes=False,
        )
        force_changed = bool(np.any(forced != cleaned))
        if force_changed:
            ocr_page["_strip_white_residual_force_fill"] = True
            ocr_page["_strip_white_residual_force_fill_from_residual_check"] = True
            ocr_page["_strip_white_residual_force_fill_limit_pixels"] = int(force_limit_pixels)
            ocr_page["_strip_white_residual_force_fill_changed_outside"] = int(force_changed_outside)
            cleaned = forced
            residual_check = _detect_inpaint_residual_text(
                band_rgb,
                cleaned,
                expanded_mask,
                raw_mask=raw_mask,
                fast_fill_mask=fast_fill_mask,
                ocr_page=residual_ocr_page,
            )
    if residual_check.get("has_residual") and "light_residual_pixels" in set(residual_check.get("flags") or []):
        retry_limit = _build_post_cleanup_limit_mask(
            expanded_mask,
            texts,
            cleaned.shape[:2],
        )
        if not np.any(retry_limit):
            retry_limit = expanded_mask
        retry_mask = _build_light_residual_retry_mask(
            band_rgb,
            cleaned,
            expanded_mask,
            retry_limit,
        )
        if retry_mask is not None and np.any(retry_mask):
            retry_started = time.perf_counter()
            retried = inpainter.inpaint(working_rgb, retry_mask, batch_size=4, force_no_tiling=True)
            retried, retry_limit_pixels, retry_changed_outside = _clamp_image_to_limit_mask(
                working_rgb,
                retried,
                retry_mask,
                texts,
            )
            retried, retry_cleanup_stats = _apply_post_inpaint_cleanup_timed(
                band_rgb,
                retried,
                texts,
                limit_mask=retry_mask,
            )
            cleaned = retried
            final_action_mask = np.maximum(final_action_mask, retry_mask.astype(np.uint8))
            ocr_page.update(retry_cleanup_stats)
            ocr_page["_strip_light_residual_retry"] = True
            ocr_page["_strip_light_residual_retry_mask_pixels"] = int(np.count_nonzero(retry_mask))
            ocr_page["_strip_light_residual_retry_limit_pixels"] = int(retry_limit_pixels)
            ocr_page["_strip_light_residual_retry_changed_outside_limit"] = int(retry_changed_outside)
            ocr_page["_t_light_residual_retry_ms"] = round((time.perf_counter() - retry_started) * 1000.0, 3)
    residual_check = _detect_inpaint_residual_text(
        band_rgb,
        cleaned,
        expanded_mask,
        raw_mask=raw_mask,
        fast_fill_mask=fast_fill_mask,
        ocr_page=residual_ocr_page,
    )
    if (
        residual_check.get("has_residual")
        and str(residual_check.get("region_source") or "").startswith("text_region_white_balloon")
    ):
        forced = _apply_white_balloon_residual_force_fill(band_rgb, cleaned, texts)
        forced, force_limit_pixels, force_changed_outside = _clamp_image_to_limit_mask(
            cleaned,
            forced,
            expanded_mask,
            texts,
            include_text_bboxes=False,
        )
        force_changed = bool(np.any(forced != cleaned))
        if force_changed:
            ocr_page["_strip_white_residual_force_fill"] = True
            ocr_page["_strip_white_residual_force_fill_from_residual_check"] = True
            ocr_page["_strip_white_residual_force_fill_limit_pixels"] = int(force_limit_pixels)
            ocr_page["_strip_white_residual_force_fill_changed_outside"] = int(force_changed_outside)
            cleaned = forced
            residual_check = _detect_inpaint_residual_text(
                band_rgb,
                cleaned,
                expanded_mask,
                raw_mask=raw_mask,
                fast_fill_mask=fast_fill_mask,
                ocr_page=residual_ocr_page,
            )
    if (
        residual_check.get("has_residual")
        and "dark_residual_pixels" in set(residual_check.get("flags") or [])
        and str(residual_check.get("region_source") or "") != "expanded_mask"
    ):
        retry_limit = _build_post_cleanup_limit_mask(
            expanded_mask,
            texts,
            cleaned.shape[:2],
        )
        if not np.any(retry_limit):
            retry_limit = expanded_mask
        retry_mask = _build_dark_residual_retry_mask(
            expanded_mask,
            retry_limit,
            cleaned.shape[:2],
        )
        if retry_mask is not None and np.any(retry_mask):
            retry_started = time.perf_counter()
            retried = inpainter.inpaint(working_rgb, retry_mask, batch_size=4, force_no_tiling=True)
            retried, retry_limit_pixels, retry_changed_outside = _clamp_image_to_limit_mask(
                working_rgb,
                retried,
                retry_mask,
                texts,
            )
            retried, retry_cleanup_stats = _apply_post_inpaint_cleanup_timed(
                band_rgb,
                retried,
                texts,
                limit_mask=retry_mask,
            )
            cleaned = retried
            final_action_mask = np.maximum(final_action_mask, retry_mask.astype(np.uint8))
            ocr_page.update(retry_cleanup_stats)
            ocr_page["_strip_dark_residual_retry"] = True
            ocr_page["_strip_dark_residual_retry_mask_pixels"] = int(np.count_nonzero(retry_mask))
            ocr_page["_strip_dark_residual_retry_limit_pixels"] = int(retry_limit_pixels)
            ocr_page["_strip_dark_residual_retry_changed_outside_limit"] = int(retry_changed_outside)
            ocr_page["_t_dark_residual_retry_ms"] = round((time.perf_counter() - retry_started) * 1000.0, 3)
    residual_check = _detect_inpaint_residual_text(
        band_rgb,
        cleaned,
        expanded_mask,
        raw_mask=raw_mask,
        fast_fill_mask=fast_fill_mask,
        ocr_page=residual_ocr_page,
    )
    if (
        residual_check.get("has_residual")
        and str(residual_check.get("region_source") or "").startswith("text_region_white_balloon")
    ):
        residual_mask = _build_residual_text_region_mask(residual_ocr_page, cleaned.shape[:2])
        residual_source = str(residual_check.get("region_source") or "")
        if not np.any(residual_mask):
            residual_mask, residual_source, _ = _select_residual_check_mask(
                ocr_page=residual_ocr_page,
                shape=cleaned.shape[:2],
                expanded_mask=expanded_mask,
                raw_mask=raw_mask,
                fast_fill_mask=fast_fill_mask,
            )
        fallback_mask = np.maximum(
            _coerce_mask_for_shape(expanded_mask, cleaned.shape[:2]),
            _coerce_mask_for_shape(residual_mask, cleaned.shape[:2]),
        ).astype(np.uint8)
        forced = _apply_white_residual_expanded_mask_force_fill(cleaned, fallback_mask)
        forced, force_limit_pixels, force_changed_outside = _clamp_image_to_limit_mask(
            cleaned,
            forced,
            fallback_mask,
            texts,
            include_text_bboxes=False,
        )
        force_changed = bool(np.any(forced != cleaned))
        if force_changed:
            force_changed_pixels = int(np.count_nonzero(np.any(forced != cleaned, axis=2)))
            cleaned = forced
            final_action_mask = np.maximum(final_action_mask, fallback_mask.astype(np.uint8))
            ocr_page["_strip_white_residual_expanded_mask_force_fill"] = True
            ocr_page["_strip_white_residual_expanded_mask_force_fill_pixels"] = int(force_changed_pixels)
            ocr_page["_strip_white_residual_expanded_mask_force_fill_mask_pixels"] = int(
                np.count_nonzero(fallback_mask)
            )
            ocr_page["_strip_white_residual_expanded_mask_force_fill_source"] = str(residual_source or "")
            ocr_page["_strip_white_residual_force_fill_limit_pixels"] = int(force_limit_pixels)
            ocr_page["_strip_white_residual_force_fill_changed_outside"] = int(force_changed_outside)
            residual_check = _detect_inpaint_residual_text(
                band_rgb,
                cleaned,
                expanded_mask,
                raw_mask=raw_mask,
                fast_fill_mask=fast_fill_mask,
                ocr_page=residual_ocr_page,
            )
    cleaned, _ = _apply_dark_panel_text_fills(cleaned, residual_ocr_page)
    if residual_ocr_page.get("_strip_used_dark_panel_fill"):
        ocr_page["_strip_used_dark_panel_fill"] = True
        ocr_page["_strip_dark_panel_fill_count"] = max(
            int(ocr_page.get("_strip_dark_panel_fill_count") or 0),
            int(residual_ocr_page.get("_strip_dark_panel_fill_count") or 0),
        )
    final_residual_check = _detect_inpaint_residual_text(
        band_rgb,
        cleaned,
        final_action_mask,
        raw_mask=raw_mask,
        fast_fill_mask=fast_fill_mask,
        ocr_page=residual_ocr_page,
    )
    if (
        final_residual_check.get("has_residual")
        and str(final_residual_check.get("region_source") or "").startswith("text_region_white_balloon")
    ):
        residual_mask = _build_residual_text_region_mask(residual_ocr_page, cleaned.shape[:2])
        residual_source = str(final_residual_check.get("region_source") or "")
        if not np.any(residual_mask):
            residual_mask, residual_source, _ = _select_residual_check_mask(
                ocr_page=residual_ocr_page,
                shape=cleaned.shape[:2],
                expanded_mask=final_action_mask,
                raw_mask=raw_mask,
                fast_fill_mask=fast_fill_mask,
            )
        fallback_mask = np.maximum(
            _coerce_mask_for_shape(final_action_mask, cleaned.shape[:2]),
            _coerce_mask_for_shape(residual_mask, cleaned.shape[:2]),
        ).astype(np.uint8)
        forced = _apply_white_residual_expanded_mask_force_fill(cleaned, fallback_mask)
        forced, force_limit_pixels, force_changed_outside = _clamp_image_to_limit_mask(
            cleaned,
            forced,
            fallback_mask,
            texts,
            include_text_bboxes=False,
        )
        force_changed = bool(np.any(forced != cleaned))
        if force_changed:
            force_changed_pixels = int(np.count_nonzero(np.any(forced != cleaned, axis=2)))
            cleaned = forced
            final_action_mask = np.maximum(final_action_mask, fallback_mask.astype(np.uint8))
            ocr_page["_strip_white_residual_expanded_mask_force_fill"] = True
            ocr_page["_strip_white_residual_expanded_mask_force_fill_pixels"] = int(force_changed_pixels)
            ocr_page["_strip_white_residual_expanded_mask_force_fill_mask_pixels"] = int(
                np.count_nonzero(fallback_mask)
            )
            ocr_page["_strip_white_residual_expanded_mask_force_fill_source"] = str(residual_source or "")
            ocr_page["_strip_white_residual_force_fill_limit_pixels"] = int(force_limit_pixels)
            ocr_page["_strip_white_residual_force_fill_changed_outside"] = int(force_changed_outside)
    final_action_mask, white_cleanup_added = _extend_final_action_mask_for_white_balloon_cleanup(
        final_action_mask,
        residual_ocr_page,
        texts,
        band_rgb,
    )
    if white_cleanup_added:
        ocr_page["_strip_final_action_mask_white_cleanup_added_pixels"] = int(white_cleanup_added)
        if _has_white_balloon_cleanup_metadata(texts) or _all_processable_texts_are_white_balloon(texts, band_rgb):
            forced = _apply_white_residual_expanded_mask_force_fill(cleaned, final_action_mask)
            forced, force_limit_pixels, force_changed_outside = _clamp_image_to_limit_mask(
                cleaned,
                forced,
                final_action_mask,
                texts,
                include_text_bboxes=False,
            )
            force_changed = bool(np.any(forced != cleaned))
            if force_changed:
                force_changed_pixels = int(np.count_nonzero(np.any(forced != cleaned, axis=2)))
                cleaned = forced
                ocr_page["_strip_final_action_mask_white_cleanup_force_fill"] = True
                ocr_page["_strip_final_action_mask_white_cleanup_force_fill_pixels"] = int(force_changed_pixels)
                ocr_page["_strip_white_residual_force_fill_limit_pixels"] = int(force_limit_pixels)
                ocr_page["_strip_white_residual_force_fill_changed_outside"] = int(force_changed_outside)
    cleaned, dark_component_residual_pixels, dark_component_residual_count = (
        _apply_dark_mask_component_bright_residual_fill(cleaned, working_rgb, final_action_mask)
    )
    if dark_component_residual_pixels:
        ocr_page["_strip_dark_mask_component_bright_residual_fill_pixels"] = int(dark_component_residual_pixels)
        ocr_page["_strip_dark_mask_component_bright_residual_fill_count"] = int(dark_component_residual_count)
    cleaned, final_clamped_outside = _clamp_final_inpaint_to_expanded_mask(
        working_rgb,
        cleaned,
        final_action_mask,
    )
    if final_clamped_outside:
        ocr_page["_strip_final_clamped_outside_expanded_mask_pixels"] = int(final_clamped_outside)
    cleaned, translator_note_final_fill_pixels = _apply_translator_note_dark_text_contract_fill(
        cleaned,
        ocr_page,
        texts_override=translator_note_fill_texts,
    )
    if translator_note_final_fill_pixels:
        final_note_mask = np.zeros(final_action_mask.shape, dtype=np.uint8)
        local_note_texts = _texts_with_band_local_bboxes(
            [dict(text) for text in list(ocr_page.get("texts") or []) if isinstance(text, dict)],
            width=width,
            height=height,
            band_y_top=band_y_top,
        )
        for raw_text in local_note_texts:
            if not isinstance(raw_text, dict):
                continue
            source = str(raw_text.get("bubble_mask_source") or raw_text.get("bubbleMaskSource") or "").strip().lower()
            if source != "translator_note_text_mask":
                continue
            bbox = (
                _normalize_bbox(raw_text.get("source_text_anchor_bbox"), width, height)
                or _normalize_bbox(raw_text.get("_source_text_anchor_bbox"), width, height)
                or _normalize_bbox(raw_text.get("source_text_mask_bbox"), width, height)
                or _normalize_bbox(raw_text.get("_source_text_mask_bbox"), width, height)
                or _normalize_bbox(raw_text.get("text_pixel_bbox"), width, height)
                or _normalize_bbox(raw_text.get("ocr_text_bbox"), width, height)
                or _normalize_bbox(raw_text.get("bbox"), width, height)
            )
            if bbox is not None:
                final_note_mask = np.maximum(final_note_mask, _mask_from_bbox(width, height, bbox, padding=10))
        if np.any(final_note_mask):
            final_action_mask = np.maximum(final_action_mask, final_note_mask.astype(np.uint8))
    _clear_resolved_current_inpaint_flags(
        ocr_page,
        final_residual_check=final_residual_check,
        final_clamped_outside=final_clamped_outside,
    )
    if _rejected_card_action_mask_allows_real_inpaint(
        ocr_page,
        vision_blocks,
        final_action_mask,
        working_rgb.shape[:2],
    ) or _rejected_card_action_mask_allows_real_inpaint(
        ocr_page,
        None,
        final_action_mask,
        working_rgb.shape[:2],
    ):
        _clear_stale_unsafe_inpaint_flags_for_current_action(ocr_page, vision_blocks)
    _write_strip_inpaint_debug(
        ocr_page,
        original_rgb=band_rgb,
        working_rgb=working_rgb,
        cleaned_rgb=cleaned,
        vision_blocks=vision_blocks,
        used_real_inpaint=True,
        fast_fill_mask=fast_fill_mask,
        raw_mask=raw_mask,
        expanded_mask=final_action_mask,
    )
    return cleaned.copy() if cleaned is working_rgb else cleaned
