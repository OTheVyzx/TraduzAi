"""Adapter em-memoria do inpainter para o pipeline strip-based."""

from __future__ import annotations

import json
import copy
import os
import re
import time
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
        mask_from_text_geometry,
    )
except ImportError:  # pragma: no cover - supports direct pipeline path imports
    from inpainter.mask_builder import (
        bbox_to_octagon_mask,
        build_inpaint_mask,
        build_raw_text_mask_from_image,
        consolidate_mask_evidence,
        expand_text_mask,
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


def _route_action_blocks_inpaint(text: dict) -> bool:
    return False


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

    for bubble_id in bubble_ids:
        inside = bubble_mask == bubble_id
        target = (remaining > 0) & inside
        background = inside & (remaining == 0)
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
    bbox_fields = ("bbox", "source_bbox", "text_pixel_bbox", "balloon_bbox", "layout_bbox", "target_bbox")
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
        if not isinstance(txt, dict) or _route_action_blocks_inpaint(txt):
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
                "balloon_type": txt.get("balloon_type"),
                "block_profile": txt.get("block_profile"),
            }
        )
    return blocks


def _cjk_mask_kwargs_for_strip_page(ocr_page: dict) -> dict:
    engine_meta = ocr_page.get("_engine_preset")
    mask_strategy = ""
    if isinstance(engine_meta, dict):
        mask_strategy = str(engine_meta.get("mask_strategy") or "").strip().lower()
    try:
        from vision_stack.runtime import _get_text_segmenter_for_page

        text_segmenter = _get_text_segmenter_for_page(ocr_page)
    except Exception:
        text_segmenter = None
    return {
        "mask_strategy": mask_strategy,
        "ocr_texts": list(ocr_page.get("texts", [])) + list(ocr_page.get("_oar_ocr_regions", [])),
        "text_segmenter": text_segmenter,
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


def _enrich_vision_blocks_from_texts_for_inpaint(
    vision_blocks: list[dict],
    texts: list[dict],
    width: int,
    height: int,
) -> list[dict]:
    if not vision_blocks or not texts:
        return vision_blocks
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
                text_bbox = _normalize_bbox(
                    text.get("text_pixel_bbox") or text.get("bbox"),
                    width,
                    height,
                )
                if text_bbox is None:
                    continue
                score = _bbox_overlap_ratio(block_bbox, text_bbox)
                if score >= 0.35:
                    matched_texts.append(text)
                if score > best_score:
                    best_score = score
                    best_text = text
        if best_text is not None and best_score >= 0.35:
            for key in (
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
                "layout_bbox",
                "_merged_source_bboxes",
                "merged_source_bboxes",
                "content_class",
                "skip_processing",
                "preserve_original",
                "render_policy",
            ):
                value = best_text.get(key)
                if value not in (None, [], ""):
                    current[key] = copy.deepcopy(value)
            if len(matched_texts) > 1:
                merged_polygons = _merge_unique_line_polygons(matched_texts)
                if merged_polygons:
                    current["line_polygons"] = merged_polygons
                merged_text_bbox: list[int] | None = None
                for text in matched_texts:
                    text_bbox = _normalize_bbox(text.get("text_pixel_bbox") or text.get("bbox"), width, height)
                    merged_text_bbox = _bbox_union_values(merged_text_bbox, text_bbox)
                if merged_text_bbox is not None:
                    current["text_pixel_bbox"] = merged_text_bbox
        enriched_blocks.append(current)
    return enriched_blocks


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
    filled_mask = np.zeros((height, width), dtype=np.uint8)

    for text in ocr_page.get("texts", []):
        rejection_reason = _fast_white_rejection_reason(text)
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
    if not isinstance(text, dict) or _route_action_blocks_inpaint(text):
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


_COLORED_STATUS_PANEL_TERMS = {
    "BEAST",
    "COLLECTION",
    "CONFIRM",
    "ERROR",
    "LEVEL",
    "MISSION",
    "NEWS",
    "POWER",
    "PROMOTION",
    "RECORD",
    "RECORDS",
    "REQUIRED",
    "SOURCE",
    "STATUS",
    "SYSTEM",
    "TAMED",
    "TAMEABLE",
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
    source = " ".join(str(text.get(key) or "") for key in ("text", "original", "translated", "traduzido")).upper()
    terms = set(re.findall(r"[A-Z]+", source))
    if terms & _COLORED_STATUS_PANEL_TERMS:
        return True
    if {"MISSED", "CALL"}.issubset(terms):
        return True
    if (terms & _PHONE_UI_MONTH_TERMS) and re.search(r"\d", source):
        return True
    return False


def _rgb_luma_chroma(rgb: object) -> tuple[float, int] | None:
    if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
        return None
    try:
        channels = [int(round(float(value))) for value in rgb[:3]]
    except Exception:
        return None
    luma = (channels[0] * 0.299) + (channels[1] * 0.587) + (channels[2] * 0.114)
    return float(luma), int(max(channels) - min(channels))


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
    if line_polygons and not _source_text_has_colored_status_terms(text):
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
        return None
    bg_luma, bg_chroma = background
    if not (36.0 <= bg_luma <= 238.0 and bg_chroma >= 8):
        return None

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
        return None
    if glyph_pixels > int(mask_pixels * 0.72) and text.get("line_polygons"):
        return None
    if glyph_pixels > int(mask_pixels * 0.92):
        return None
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
    return line_count >= 3 or bbox_area >= 16_000


def _colored_status_panel_glyph_mask(
    image_rgb: np.ndarray,
    text: dict,
    text_mask: np.ndarray,
    background_rgb: np.ndarray | None = None,
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
    candidate = (text_mask > 0) & (delta >= 24.0) & (
        ((luma >= float(bg_luma) + 18.0) & (luma >= 145.0))
        | ((luma <= float(bg_luma) - 20.0) & (float(bg_luma) >= 88.0))
        | ((np.abs(luma - float(bg_luma)) >= 4.0) & (chroma >= 38) & (delta >= 28.0))
    )
    if int(np.count_nonzero(candidate)) < 24:
        return None
    if not text.get("line_polygons"):
        height, width = image_rgb.shape[:2]
        text_bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height) or _normalize_bbox(
            text.get("bbox"),
            width,
            height,
        )
        if text_bbox is not None:
            return _mask_from_bbox(width, height, text_bbox, padding=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    glyph_mask = cv2.dilate(candidate.astype(np.uint8) * 255, kernel, iterations=1)
    return glyph_mask if np.any(glyph_mask) else None


def _try_dark_panel_text_fill(image_rgb: np.ndarray, text: dict) -> np.ndarray | None:
    if not isinstance(image_rgb, np.ndarray):
        return None
    colored_status_panel = _is_colored_status_panel_text_candidate(text)
    metadata_candidate = _is_dark_panel_text_candidate(text) or colored_status_panel
    if not metadata_candidate:
        if not isinstance(text, dict) or _route_action_blocks_inpaint(text):
            return None
    if not metadata_candidate and not text.get("text_pixel_bbox") and not text.get("line_polygons"):
        return None
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return None
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
    p80_luma = float(np.percentile(sample_luma, 80))
    p90_luma = float(np.percentile(sample_luma, 90))
    sample_fill_color = np.asarray(
        [int(max(0, min(255, round(float(v))))) for v in np.median(sample, axis=0)],
        dtype=np.uint8,
    )
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
        if p90_luma > 245.0 and sampled_colored_background is None:
            return None
    else:
        if median_luma > 150.0:
            return None
        if p90_luma > 210.0:
            return None

    fill = sampled_colored_background.astype(np.float32) if sampled_colored_background is not None else sample_fill_color.astype(np.float32)
    fill_color = np.asarray([int(max(0, min(255, round(float(v))))) for v in fill], dtype=np.uint8)
    text_bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height) or _normalize_bbox(
        text.get("bbox"),
        width,
        height,
    )
    fill_mask: np.ndarray
    if detected_dark_panel and text_bbox is not None:
        max_side = max(text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        pad = max(8, min(22, int(round(max_side * 0.06))))
        fill_bbox = _expanded_bbox(width, height, text_bbox, padding=pad) or text_bbox
        fx1, fy1, fx2, fy2 = fill_bbox
        fill_mask = np.zeros((height, width), dtype=np.uint8)
        fill_mask[fy1:fy2, fx1:fx2] = 255
    elif colored_status_panel:
        glyph_mask = _colored_status_panel_glyph_mask(image_rgb, text, text_mask, fill_color)
        if glyph_mask is None:
            return None
        fill_mask = glyph_mask
    else:
        kernel_size = 25 if colored_status_panel else 15
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        fill_mask = cv2.dilate(text_mask, fill_kernel, iterations=1)
    has_line_geometry = bool(text.get("line_polygons"))
    authorized_padding = 1 if has_line_geometry else (pad if detected_dark_panel and text_bbox is not None else 1)
    authorized_mask = _authorized_fast_fill_mask(width, height, text, padding=authorized_padding)
    if authorized_mask is None:
        return None
    fill_mask = cv2.bitwise_and(fill_mask.astype(np.uint8), authorized_mask.astype(np.uint8))
    if not np.any(fill_mask):
        return None
    result = image_rgb.copy()
    result[fill_mask > 0] = fill_color
    return result


def _apply_dark_panel_text_fills(image_rgb: np.ndarray, ocr_page: dict) -> tuple[np.ndarray, int]:
    return image_rgb, 0


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

    if not _fast_dark_panel_fill_enabled():
        rejection_reasons["disabled"] = max(
            1,
            len([text for text in ocr_page.get("texts", []) if isinstance(text, dict)]),
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
    filled_mask = np.zeros((height, width), dtype=np.uint8)

    for text in ocr_page.get("texts", []) or []:
        if not isinstance(text, dict):
            _reject("invalid_text")
            continue
        mask_evidence_reason = _fast_fill_mask_evidence_rejection_reason(text)
        if mask_evidence_reason:
            _propagate_existing_mask_evidence_decision_flags(ocr_page, text)
            _reject(mask_evidence_reason)
            continue
        qa_reason = _fast_fill_blocking_qa_reason(text)
        if qa_reason:
            _reject(qa_reason)
            continue
        if _is_rotated_recovery_text(text):
            _reject("rotated_recovery_real_inpaint_required")
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
        filled = _try_dark_panel_text_fill(result, text)
        if filled is None:
            _reject("not_solid_dark_panel")
            continue
        changed_mask = np.any(filled != result, axis=2).astype(np.uint8) * 255
        if not np.any(changed_mask):
            _reject("no_fast_fill_change")
            continue
        if not any(
            _block_is_covered_by_fast_fill(block, [fill_bbox], width, height, changed_mask)
            for block in vision_blocks
        ):
            _reject("no_covered_vision_block")
            continue
        result = filled
        filled_mask = np.maximum(filled_mask, changed_mask)
        filled_bboxes.append(fill_bbox)
        filled_keys.add(fill_key)

    if not filled_bboxes:
        return band_rgb, vision_blocks, _record(
            {"dark_panel_fill_count": 0, "remaining_blocks": len(vision_blocks)}
        )

    remaining_blocks = [
        block
        for block in vision_blocks
        if not _block_is_covered_by_fast_fill(block, filled_bboxes, width, height, filled_mask)
    ]
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
    if not has_polygon:
        bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height)
        if bbox is None:
            return None
        mask = _mask_from_bbox(width, height, bbox, padding=1)
    if int(np.count_nonzero(mask)) < 12:
        return None
    return mask


def _authorized_fast_fill_mask(width: int, height: int, text: dict, padding: int = 1) -> np.ndarray | None:
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
        and not _route_action_blocks_inpaint(text)
    ]


def _vision_block_requires_inpaint(block: dict) -> bool:
    if not isinstance(block, dict):
        return False
    route_action = str(block.get("route_action") or "").strip().lower()
    if route_action in ROUTE_ACTIONS:
        return route_action_requires_inpaint(route_action)
    return True


def _processable_vision_blocks_for_inpaint(blocks: list[dict]) -> list[dict]:
    return [block for block in blocks if _vision_block_requires_inpaint(block)]


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
        if not (text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("source_bbox")):
            continue
        try:
            evidence_mask = build_inpaint_mask(dict(text), image_rgb.shape, image_rgb=image_rgb)
        except Exception:
            evidence_mask = None
        if evidence_mask is None or not np.any(evidence_mask):
            continue
        evidence = _coerce_mask_for_shape(evidence_mask, shape)
        if not np.any(evidence):
            continue

        expanded_pixels = int(np.count_nonzero(expanded))
        evidence_pixels = int(np.count_nonzero(evidence))
        extra_pixels = int(np.count_nonzero((evidence > 0) & (expanded == 0)))
        if extra_pixels <= 0:
            continue
        if expanded_pixels > 0:
            if evidence_pixels > max(expanded_pixels + 4096, int(expanded_pixels * 1.45)):
                continue
            if extra_pixels > max(4096, int(expanded_pixels * 0.45)):
                continue
        qa_metrics = text.setdefault("qa_metrics", {})
        if isinstance(qa_metrics, dict):
            qa_metrics["text_evidence_mask_extra_pixels"] = int(extra_pixels)
        raw = np.maximum(raw, evidence)
        expanded = np.maximum(expanded, evidence)
    return raw.astype(np.uint8), expanded.astype(np.uint8)


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
    metadata = {
        "page_number": int(ocr_page.get("_source_page_number") or ocr_page.get("numero") or 0),
        "band_index": int(ocr_page.get("_band_index") or 0),
        "text_count": len([t for t in ocr_page.get("texts", []) if isinstance(t, dict)]),
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
        "flags": list(dict.fromkeys(str(flag) for flag in ocr_page.get("_strip_inpaint_decision_flags", []) if flag)),
    }
    (debug_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    ocr_page["_strip_inpaint_debug_dir"] = str(debug_dir)


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
        if not isinstance(block, dict) or _route_action_blocks_inpaint(block):
            continue
        text_mask = build_inpaint_mask(block, band_rgb.shape, image_rgb=working_rgb)
        if not isinstance(text_mask, np.ndarray) or not np.any(text_mask):
            reason = "missing_glyph_text_mask"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue

        bubble_mask, bubble_reason = _real_bubble_mask_for_koharu_fill(ocr_page, block, width, height)
        fast_image, block_remaining, metadata = apply_koharu_bubble_fast_fill(
            working_rgb,
            text_mask,
            bubble_mask,
        )
        filled_pixels = int(metadata.get("filled_pixels") or 0)
        if filled_pixels:
            changed = np.any(fast_image != working_rgb, axis=2)
            fast_fill_mask[changed] = 255
            working_rgb = fast_image
            filled_total += filled_pixels
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
                "filled_pixels": filled_pixels,
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
    if not vision_blocks:
        return band_rgb.copy()

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
    ocr_page.setdefault("_strip_remaining_inpaint_blocks", len(vision_blocks))

    _prime_mask_evidence_for_fast_fill(ocr_page, vision_blocks, band_rgb)

    working_rgb, vision_blocks, koharu_fast_fill_mask, koharu_remaining_mask, koharu_meta = (
        _apply_koharu_bubble_fast_fill_to_blocks(band_rgb, ocr_page, vision_blocks)
    )
    ocr_page["_strip_used_koharu_fast_fill"] = bool(koharu_meta.get("filled_pixels"))
    ocr_page["_strip_koharu_fast_fill_pixels"] = int(koharu_meta.get("filled_pixels") or 0)
    ocr_page["_strip_koharu_remaining_mask_pixels"] = int(koharu_meta.get("remaining_pixels") or 0)
    ocr_page["_strip_koharu_fast_fill_samples"] = list(koharu_meta.get("samples") or [])
    ocr_page["_strip_koharu_fast_fill_reject_reasons"] = dict(koharu_meta.get("rejection_reasons") or {})
    ocr_page["_strip_remaining_inpaint_blocks"] = len(vision_blocks)

    if vision_blocks:
        processable_vision_blocks = _processable_vision_blocks_for_inpaint(vision_blocks)
        ignored_count = len(vision_blocks) - len(processable_vision_blocks)
        if ignored_count:
            ocr_page["_strip_nonprocessable_remaining_block_count"] = int(ignored_count)
            _append_inpaint_decision_flag(ocr_page, "nonprocessable_remaining_blocks_ignored")
        vision_blocks = processable_vision_blocks

    fast_fill_mask = (np.any(working_rgb != band_rgb, axis=2).astype(np.uint8) * 255)
    if not vision_blocks:
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
    raw_mask = _coerce_mask_for_shape(koharu_remaining_mask, working_rgb.shape[:2])
    if not np.any(raw_mask):
        raw_mask = vision_blocks_to_mask(
            working_rgb.shape,
            vision_blocks,
            image_rgb=working_rgb,
            expand_mask=False,
            **mask_kwargs,
        )
    expanded_mask = raw_mask.astype(np.uint8)
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
    ocr_page["_strip_used_real_inpaint"] = True
    cleaned, cleanup_stats = _apply_post_inpaint_cleanup_timed(
        band_rgb,
        cleaned,
        list(ocr_page.get("texts", [])),
        limit_mask=expanded_mask,
    )
    ocr_page.update(cleanup_stats)
    ocr_page["_strip_used_post_cleanup"] = True
    try:
        band_y_top = int(ocr_page.get("_band_y_top") or 0)
    except Exception:
        band_y_top = 0
    texts = _texts_with_band_local_bboxes(
        [dict(text) for text in list(ocr_page.get("texts", [])) if isinstance(text, dict)],
        width=width,
        height=height,
        band_y_top=band_y_top,
    )
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
        and "dark_residual_pixels" in set(residual_check.get("flags") or [])
        and _page_has_nonwhite_text_for_light_residual(residual_ocr_page, band_rgb)
    ):
        retry_limit = _build_post_cleanup_limit_mask(
            expanded_mask,
            texts,
            cleaned.shape[:2],
        )
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
            ocr_page.update(retry_cleanup_stats)
            ocr_page["_strip_dark_residual_retry"] = True
            ocr_page["_strip_dark_residual_retry_mask_pixels"] = int(np.count_nonzero(retry_mask))
            ocr_page["_strip_dark_residual_retry_limit_pixels"] = int(retry_limit_pixels)
            ocr_page["_strip_dark_residual_retry_changed_outside_limit"] = int(retry_changed_outside)
            ocr_page["_t_dark_residual_retry_ms"] = round((time.perf_counter() - retry_started) * 1000.0, 3)
    _write_strip_inpaint_debug(
        ocr_page,
        original_rgb=band_rgb,
        working_rgb=working_rgb,
        cleaned_rgb=cleaned,
        vision_blocks=vision_blocks,
        used_real_inpaint=True,
        fast_fill_mask=fast_fill_mask,
        raw_mask=raw_mask,
        expanded_mask=expanded_mask,
    )
    return cleaned.copy() if cleaned is working_rgb else cleaned
