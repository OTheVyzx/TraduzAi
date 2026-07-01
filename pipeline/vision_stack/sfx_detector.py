from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


def detect_sfx_candidates(
    image_rgb: np.ndarray,
    *,
    existing_texts: list[dict[str, Any]] | None = None,
    existing_blocks: list[dict[str, Any]] | None = None,
    min_confidence: float = 0.45,
) -> list[dict[str, Any]]:
    """Detect visual manhwa SFX candidates before OCR can read them."""

    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return []
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return []
    long_page_restricted = _is_long_page_visual_detector_restricted(height, width)

    text_bboxes = [_coerce_bbox(text.get("bbox") or text.get("text_pixel_bbox")) for text in existing_texts or []]
    text_bboxes = [bbox for bbox in text_bboxes if bbox is not None]
    block_bboxes = [_coerce_bbox(block.get("bbox") or block.get("balloon_bbox")) for block in existing_blocks or []]
    block_bboxes = [bbox for bbox in block_bboxes if bbox is not None]

    mask_sources = _candidate_masks(image_rgb)
    candidates: list[dict[str, Any]] = []
    for source, mask in mask_sources:
        if long_page_restricted and source not in {"red_chroma", "color_chroma"}:
            continue
        for bbox, component_area in _component_bboxes(mask, image_rgb.shape):
            if _suppressed_by_existing_text(bbox, text_bboxes):
                continue
            if _looks_like_speech_balloon_text(bbox, block_bboxes, image_rgb.shape):
                continue
            score = _score_candidate(image_rgb, mask, bbox, component_area, source)
            if score < float(min_confidence):
                continue
            candidates.append(_candidate_payload(len(candidates) + 1, bbox, score, source))

    candidates = _dedupe_candidates(candidates)
    candidates = _merge_nearby_short_page_visual_candidates(candidates, image_rgb.shape)
    return _dedupe_candidates(candidates)


def _is_long_page_visual_detector_restricted(height: int, width: int) -> bool:
    if height / float(max(1, width)) < 3.0:
        return False
    flag = os.getenv("TRADUZAI_SFX_VISUAL_LONG_PAGE", "0")
    return str(flag).strip().lower() not in {"1", "true", "yes", "on"}


def text_blocks_to_sfx_candidates(
    image_rgb: np.ndarray,
    blocks: list[Any],
    *,
    source: str,
    existing_texts: list[dict[str, Any]] | None = None,
    min_confidence: float = 0.01,
    min_area_ratio: float = 0.0015,
    min_low_conf_area_ratio: float = 0.010,
) -> list[dict[str, Any]]:
    """Convert detector blocks into conservative SFX review candidates."""

    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return []
    height, width = image_rgb.shape[:2]
    page_area = max(1, height * width)
    text_bboxes = [_coerce_bbox(text.get("bbox") or text.get("text_pixel_bbox")) for text in existing_texts or []]
    text_bboxes = [bbox for bbox in text_bboxes if bbox is not None]
    candidates: list[dict[str, Any]] = []
    for block in blocks or []:
        bbox = _block_bbox(block)
        if bbox is None:
            continue
        bbox = [
            max(0, min(width, bbox[0])),
            max(0, min(height, bbox[1])),
            max(0, min(width, bbox[2])),
            max(0, min(height, bbox[3])),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        confidence = _block_confidence(block)
        if confidence < float(min_confidence):
            continue
        if _suppressed_by_existing_text(bbox, text_bboxes):
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < int(page_area * float(min_area_ratio)):
            continue
        if confidence < 0.02 and area < int(page_area * float(min_low_conf_area_ratio)):
            continue
        if _looks_like_long_page_scanlator_or_ui_artifact(
            image_rgb,
            bbox,
            source=source,
            confidence=confidence,
        ):
            continue
        if not _has_sfx_text_block_support(image_rgb, bbox, source=source, confidence=confidence):
            continue
        candidates.append(
            _candidate_payload(
                len(candidates) + 1,
                bbox,
                round(float(confidence), 4),
                source,
                detector="sfx_text_detector",
            )
        )
    return _dedupe_candidates(candidates)


def _looks_like_long_page_scanlator_or_ui_artifact(
    image_rgb: np.ndarray,
    bbox: list[int],
    *,
    source: str,
    confidence: float,
) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_ratio = (bw * bh) / float(max(1, width * height))
    y_center = (y1 + y2) / 2.0
    y_center_ratio = y_center / float(max(1, height))
    width_ratio = bw / float(max(1, width))
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    bright_ratio = float(np.mean(gray >= 235))
    dark_ratio = float(np.mean(gray <= 72))
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    aspect = bw / float(max(1, bh))

    # Scanlator/advertisement footer blocks are usually wide, near the bottom,
    # and mostly rectangular text/logo regions. Real manhwa SFX in this data is
    # not presented as a bottom legal/credit banner.
    if y_center_ratio >= 0.88 and width_ratio >= 0.35:
        return True
    if y_center_ratio >= 0.78 and width_ratio >= 0.70 and aspect >= 1.8:
        return True

    # Top-of-chapter promo/social cards and UI labels are also frequent false
    # positives from comic-text-detector on webtoon strips.
    if y_center_ratio <= 0.14 and width_ratio >= 0.32 and aspect >= 1.45:
        return True
    if y_center_ratio <= 0.18 and area_ratio >= 0.006 and bright_ratio >= 0.35 and dark_ratio <= 0.18:
        return True

    # Small horizontal UI/caption labels are not SFX even when detector confidence
    # is high.
    if aspect >= 2.4 and bright_ratio >= 0.28 and edge_ratio <= 0.12 and source == "comic_text_detector_fallback":
        return True
    return False


def merge_sfx_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply final SFX candidate dedupe and stable ids."""

    return _dedupe_candidates([candidate for candidate in candidates if isinstance(candidate, dict)])


def filter_sfx_candidates_after_ocr(
    candidates: list[dict[str, Any]],
    image_rgb: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Drop detector-only SFX candidates that OCR and visual checks agree are art."""

    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return candidates
    long_page = image_rgb.shape[0] / float(max(1, image_rgb.shape[1])) >= 3.0
    cjk_visual_boxes = _recognized_visual_cjk_boxes(candidates, image_rgb) if long_page else []
    kept: list[dict[str, Any]] = []
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        if long_page and _is_restricted_long_page_visual_candidate(candidate):
            bbox = _coerce_bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox"))
            if bbox is not None and _looks_like_long_page_visual_rescue_artifact(candidate, image_rgb, bbox):
                candidate.setdefault("qa_flags", []).append("sfx_artifact_long_page_visual_rescue_rejected")
                continue
            status = _candidate_ocr_status(candidate)
            if status in {"", "no_confident_cjk"}:
                if bbox is not None and _looks_like_unconfirmed_light_color_sfx(candidate, image_rgb, bbox):
                    candidate.setdefault("qa_flags", []).append("sfx_long_page_light_color_rescue")
                elif bbox is None or not _near_any_bbox(bbox, cjk_visual_boxes, max_gap=96):
                    candidate.setdefault("qa_flags", []).append("sfx_artifact_long_page_unconfirmed_visual_rejected")
                    continue
                else:
                    candidate.setdefault("qa_flags", []).append("sfx_long_page_visual_neighbor_rescue")
        if _should_drop_non_cjk_sfx_artifact(candidate, image_rgb):
            continue
        kept.append(candidate)
    if long_page:
        kept = _merge_nearby_long_page_visual_candidates(kept)
    return _dedupe_candidates(kept)


def _recognized_visual_cjk_boxes(candidates: list[dict[str, Any]], image_rgb: np.ndarray) -> list[list[int]]:
    boxes: list[list[int]] = []
    for candidate in candidates or []:
        if not isinstance(candidate, dict) or not _is_restricted_long_page_visual_candidate(candidate):
            continue
        if _candidate_ocr_status(candidate) != "recognized":
            continue
        if not _candidate_has_cjk_ocr(candidate):
            continue
        bbox = _coerce_bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox"))
        if bbox is not None and _looks_like_long_page_visual_rescue_artifact(candidate, image_rgb, bbox):
            continue
        if bbox is not None:
            boxes.append(bbox)
    return boxes


def _is_restricted_long_page_visual_candidate(candidate: dict[str, Any]) -> bool:
    sfx = candidate.get("sfx") if isinstance(candidate.get("sfx"), dict) else {}
    return (
        str(candidate.get("detector") or "") == "sfx_visual"
        and str(sfx.get("visual_source") or "") in {"red_chroma", "color_chroma"}
    )


def _candidate_ocr_status(candidate: dict[str, Any]) -> str:
    sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
    return str(sfx_ocr.get("status") or "").strip().lower()


def _candidate_has_cjk_ocr(candidate: dict[str, Any]) -> bool:
    text = str(candidate.get("recognized_text") or candidate.get("text") or "")
    if _contains_cjk(text):
        return True
    sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
    attempts = sfx_ocr.get("attempts") if isinstance(sfx_ocr.get("attempts"), list) else []
    return any(isinstance(attempt, dict) and _contains_cjk(str(attempt.get("text") or "")) for attempt in attempts[:12])


def _near_any_bbox(bbox: list[int], others: list[list[int]], *, max_gap: int) -> bool:
    return any(_bbox_gap(bbox, other) <= int(max_gap) for other in others)


def _bbox_gap(a: list[int], b: list[int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0, max(by1 - ay2, ay1 - by2))
    return int(max(dx, dy))


def _merge_nearby_long_page_visual_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    used: set[int] = set()
    merged: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if index in used:
            continue
        if not _mergeable_long_page_visual_candidate(candidate):
            merged.append(candidate)
            used.add(index)
            continue
        group = [candidate]
        used.add(index)
        changed = True
        while changed:
            changed = False
            group_boxes = [_coerce_bbox(item.get("bbox") or item.get("text_pixel_bbox")) for item in group]
            group_boxes = [bbox for bbox in group_boxes if bbox is not None]
            for other_index, other in enumerate(candidates):
                if other_index in used or not _mergeable_long_page_visual_candidate(other):
                    continue
                other_bbox = _coerce_bbox(other.get("bbox") or other.get("text_pixel_bbox"))
                if other_bbox is None:
                    continue
                if any(_bbox_gap(other_bbox, group_bbox) <= 96 for group_bbox in group_boxes):
                    group.append(other)
                    used.add(other_index)
                    changed = True
        if len(group) == 1:
            merged.append(candidate)
        else:
            current = group[0]
            current_bbox = _coerce_bbox(current.get("bbox") or current.get("text_pixel_bbox"))
            for other in group[1:]:
                other_bbox = _coerce_bbox(other.get("bbox") or other.get("text_pixel_bbox"))
                if current_bbox is None or other_bbox is None:
                    continue
                current = _merge_visual_candidate_payload(current, other, current_bbox, other_bbox)
                current["qa_flags"] = _merge_unique_flags(current.get("qa_flags"), ["sfx_long_page_visual_cluster_merged"])
                sfx = current.get("sfx") if isinstance(current.get("sfx"), dict) else {}
                sfx["qa_flags"] = _merge_unique_flags(sfx.get("qa_flags"), ["sfx_long_page_visual_cluster_merged"])
                current["sfx"] = sfx
                current_bbox = _coerce_bbox(current.get("bbox") or current.get("text_pixel_bbox"))
            merged.append(current)
    return merged


def _mergeable_long_page_visual_candidate(candidate: dict[str, Any]) -> bool:
    if not _is_restricted_long_page_visual_candidate(candidate):
        return False
    flags = {str(flag) for flag in candidate.get("qa_flags") or []}
    return (
        _candidate_ocr_status(candidate) == "recognized"
        or "sfx_long_page_visual_neighbor_rescue" in flags
        or "sfx_long_page_light_color_rescue" in flags
    )


def _looks_like_long_page_visual_rescue_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return True
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    rgb = rgb_u8.astype(np.float32)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    saturation = hsv[:, :, 1].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.float32)
    edges = cv2.Canny(gray, 55, 150)
    bright_ratio = float(np.mean(gray >= 235))
    dark_ratio = float(np.mean(gray <= 75))
    edge_ratio = float(np.mean(edges > 0))
    color_ratio = float(np.mean((saturation >= 45.0) & (chroma >= 28.0)))
    skin_or_brown = (
        (hue >= 5.0)
        & (hue <= 28.0)
        & (saturation >= 20.0)
        & (saturation <= 170.0)
        & (gray >= 45)
        & (gray <= 235)
    )
    skin_or_brown_ratio = float(np.mean(skin_or_brown))
    source = str((candidate.get("sfx") or {}).get("visual_source") or "")
    if bright_ratio >= 0.72 and dark_ratio <= 0.025 and edge_ratio <= 0.04:
        return True
    if source == "red_chroma" and skin_or_brown_ratio >= 0.55:
        return True
    if source == "color_chroma" and color_ratio <= 0.075 and bright_ratio <= 0.08 and dark_ratio <= 0.08:
        return True
    return False


def _looks_like_unconfirmed_light_color_sfx(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    source = str((candidate.get("sfx") or {}).get("visual_source") or "")
    if source != "color_chroma":
        return False
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    bw = x2 - x1
    bh = y2 - y1
    if bw < 48 or bh < 96:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    rgb = rgb_u8.astype(np.float32)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    saturation = hsv[:, :, 1].astype(np.float32)
    edges = cv2.Canny(gray, 55, 150)
    bright_ratio = float(np.mean(gray >= 235))
    dark_ratio = float(np.mean(gray <= 75))
    edge_ratio = float(np.mean(edges > 0))
    color_ratio = float(np.mean((saturation >= 45.0) & (chroma >= 28.0)))
    if bright_ratio < 0.52 or bright_ratio > 0.82:
        return False
    if dark_ratio > 0.035:
        return False
    if edge_ratio < 0.075:
        return False
    if color_ratio < 0.12:
        return False
    return True


def _candidate_masks(image_rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    rgb_u8 = image_rgb.astype(np.uint8)
    rgb_i = rgb_u8.astype(np.int16)
    luma = (
        rgb_u8[:, :, 0].astype(np.float32) * 0.299
        + rgb_u8[:, :, 1].astype(np.float32) * 0.587
        + rgb_u8[:, :, 2].astype(np.float32) * 0.114
    )
    chroma = (np.max(rgb_i, axis=2) - np.min(rgb_i, axis=2)).astype(np.float32)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    color = ((saturation >= 34.0) & (chroma >= 22.0) & (value <= 242.0)).astype(np.uint8) * 255
    color = cv2.morphologyEx(color, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7)))

    hue = hsv[:, :, 0].astype(np.float32)
    red = (
        (((hue <= 10.0) | (hue >= 165.0))
        & (saturation >= 35.0)
        & (value >= 80.0))
    ).astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))

    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    near_chroma = cv2.dilate(color, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    white_near_chroma = ((gray >= 245) & near_chroma).astype(np.uint8) * 255
    white_near_chroma = cv2.morphologyEx(
        white_near_chroma,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    white_near_chroma = cv2.morphologyEx(
        white_near_chroma,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9)),
    )

    background = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32)
    gray_f = gray.astype(np.float32)
    std = float(np.std(gray_f))
    dark = (background - gray_f) >= max(20.0, std * 0.38)
    light = (gray_f - background) >= max(20.0, std * 0.38)
    not_balloon_black_text = ~((luma <= 82.0) & (chroma <= 18.0))
    contrast = ((dark | light) & not_balloon_black_text).astype(np.uint8) * 255
    contrast = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    contrast = cv2.morphologyEx(contrast, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9)))

    return [
        ("red_chroma", red),
        ("color_chroma", color),
        ("white_near_chroma", white_near_chroma),
        ("local_contrast", contrast),
    ]


def _component_bboxes(mask: np.ndarray, image_shape: tuple[int, int, int]) -> list[tuple[list[int], int]]:
    height, width = image_shape[:2]
    page_area = max(1, height * width)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    items: list[tuple[list[int], int]] = []
    for label in range(1, count):
        x, y, w_box, h_box, area = [int(v) for v in stats[label].tolist()]
        if area < max(80, int(page_area * 0.00035)):
            continue
        if area > int(page_area * 0.18):
            continue
        if w_box < 24 or h_box < 24:
            continue
        bbox = _expand_bbox([x, y, x + w_box, y + h_box], width, height, pad=max(6, min(w_box, h_box) // 8))
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bw * bh > int(page_area * 0.26):
            continue
        aspect = max(bw, bh) / float(max(1, min(bw, bh)))
        if aspect > 7.5:
            continue
        items.append((bbox, area))
    return items


def _score_candidate(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: list[int],
    component_area: int,
    source: str,
) -> float:
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2] > 0
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    page_area = max(1, image_rgb.shape[0] * image_rgb.shape[1])
    area_ratio = bbox_area / float(page_area)
    page_aspect = image_rgb.shape[0] / float(max(1, image_rgb.shape[1]))
    fill_ratio = component_area / float(bbox_area)
    max_fill_ratio = 0.72 if source in {"color_chroma", "red_chroma"} else 0.55
    if crop.size == 0 or fill_ratio < 0.015 or fill_ratio > max_fill_ratio:
        return 0.0
    if page_aspect >= 3.0 and area_ratio >= 0.022:
        return 0.0
    rgb = crop.astype(np.float32)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    mean_chroma = float(np.mean(chroma[mask_crop])) if np.any(mask_crop) else 0.0
    luma = (
        rgb[:, :, 0] * 0.299
        + rgb[:, :, 1] * 0.587
        + rgb[:, :, 2] * 0.114
    )
    luma_std = float(np.std(luma))
    bw = x2 - x1
    bh = y2 - y1
    if _looks_like_latin_balloon_text(crop, source, bw, bh):
        return 0.0
    if _looks_like_character_artifact(crop, source):
        return 0.0
    if _looks_like_large_panel_or_scene_artifact(crop, source, area_ratio=area_ratio, fill_ratio=fill_ratio):
        return 0.0
    if source == "white_near_chroma" and (bw < 85 and bh < 85):
        return 0.0
    size_score = min(1.0, max(bw, bh) / 130.0)
    chroma_score = min(1.0, mean_chroma / 70.0)
    texture_score = min(1.0, luma_std / 58.0)
    fill_target = 0.30 if source in {"color_chroma", "red_chroma"} else 0.18
    fill_score = 1.0 - min(1.0, abs(fill_ratio - fill_target) / 0.42)
    source_bonus = 0.10 if source in {"color_chroma", "red_chroma"} else 0.06 if source == "white_near_chroma" else 0.0
    score = 0.22 + size_score * 0.18 + chroma_score * 0.24 + texture_score * 0.16 + fill_score * 0.20 + source_bonus
    return round(float(min(0.98, max(0.0, score))), 3)


def _looks_like_large_panel_or_scene_artifact(
    crop_rgb: np.ndarray,
    source: str,
    *,
    area_ratio: float,
    fill_ratio: float,
) -> bool:
    """Reject broad panel/art regions before they become SFX candidates."""

    if area_ratio < 0.035:
        return False
    rgb_u8 = crop_rgb.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    rgb = rgb_u8.astype(np.float32)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    saturation = hsv[:, :, 1].astype(np.float32)
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    dark_ratio = float(np.mean(gray <= 75))
    bright_ratio = float(np.mean(gray >= 235))
    color_ratio = float(np.mean((saturation >= 45.0) & (chroma >= 28.0)))
    luma_std = float(np.std(gray.astype(np.float32)))

    if source == "local_contrast":
        if area_ratio >= 0.08 and fill_ratio <= 0.12:
            return True
        if bright_ratio >= 0.34 and dark_ratio <= 0.10 and color_ratio <= 0.025:
            return True
        if edge_ratio <= 0.018 and luma_std <= 34.0:
            return True

    if source == "color_chroma":
        hue = hsv[:, :, 0].astype(np.float32)
        skin_or_brown = (
            (hue >= 5.0)
            & (hue <= 28.0)
            & (saturation >= 25.0)
            & (saturation <= 165.0)
            & (gray >= 45)
            & (gray <= 230)
        )
        skin_ratio = float(np.mean(skin_or_brown))
        if area_ratio >= 0.05 and fill_ratio <= 0.18 and skin_ratio >= 0.10:
            return True
        if area_ratio >= 0.10 and fill_ratio <= 0.10 and edge_ratio <= 0.035:
            return True

    return False


def _block_bbox(block: Any) -> list[int] | None:
    if isinstance(block, dict):
        return _coerce_bbox(block.get("bbox") or block.get("xyxy") or block.get("text_pixel_bbox"))
    xyxy = getattr(block, "xyxy", None)
    return _coerce_bbox(xyxy)


def _block_confidence(block: Any) -> float:
    if isinstance(block, dict):
        return _as_float(block.get("confidence"), 0.0)
    return _as_float(getattr(block, "confidence", 0.0), 0.0)


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _has_sfx_text_block_support(
    image_rgb: np.ndarray,
    bbox: list[int],
    *,
    source: str,
    confidence: float,
) -> bool:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw < 18 or bh < 18:
        return False
    page_area = max(1, height * width)
    area = bw * bh
    max_area_ratio = 0.42 if confidence >= 0.05 else 0.30
    if area > int(page_area * max_area_ratio):
        return False
    aspect = max(bw, bh) / float(max(1, min(bw, bh)))
    if aspect > 6.8:
        return False

    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    rgb_f = rgb_u8.astype(np.float32)
    chroma = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    saturation = hsv[:, :, 1].astype(np.float32)
    dark_ratio = float(np.mean(gray <= 72))
    bright_ratio = float(np.mean(gray >= 235))
    color_ratio = float(np.mean((saturation >= 45.0) & (chroma >= 28.0)))
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    luma_std = float(np.std(gray.astype(np.float32)))
    if _looks_like_plain_dialogue_or_caption_crop(
        bw=bw,
        bh=bh,
        dark_ratio=dark_ratio,
        bright_ratio=bright_ratio,
        color_ratio=color_ratio,
        edge_ratio=edge_ratio,
    ):
        return False
    if source == "anime_text_yolo_low_conf" and confidence >= 0.02:
        return True
    if source == "comic_text_detector_fallback" and confidence >= 0.05:
        return True

    # Low-confidence rescue must be visually text/SFX-like, not just a tiny line or glow.
    if confidence < 0.02:
        if max(bw, bh) < 92:
            return False
        has_ink = dark_ratio >= 0.025 and edge_ratio >= 0.025 and luma_std >= 28.0
        has_colored_stroke = color_ratio >= 0.035 and edge_ratio >= 0.018 and luma_std >= 22.0
        if not (has_ink or has_colored_stroke):
            return False

    if source == "comic_text_detector_fallback":
        return dark_ratio >= 0.015 or color_ratio >= 0.025 or edge_ratio >= 0.035

    return dark_ratio >= 0.012 or color_ratio >= 0.020 or bright_ratio >= 0.10


def _looks_like_plain_dialogue_or_caption_crop(
    *,
    bw: int,
    bh: int,
    dark_ratio: float,
    bright_ratio: float,
    color_ratio: float,
    edge_ratio: float,
) -> bool:
    if bh <= 0:
        return False
    aspect = bw / float(max(1, bh))
    if aspect < 1.55:
        return False
    if bright_ratio < 0.34 or color_ratio >= 0.055:
        return False
    if dark_ratio > 0.34:
        return False
    return edge_ratio <= 0.13


def _should_drop_non_cjk_sfx_artifact(candidate: dict[str, Any], image_rgb: np.ndarray) -> bool:
    sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
    status = str(sfx_ocr.get("status") or "").strip().lower()
    recognized_spurious_caption = False
    recognized_footer_credit = False
    recognized_top_ornament = False
    bbox = _coerce_bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox"))
    if status == "recognized":
        recognized_spurious_caption = bool(
            bbox is not None and (
                _looks_like_spurious_recognized_caption_artifact(candidate, image_rgb, bbox)
                or _looks_like_short_hangul_caption_artifact(candidate, image_rgb, bbox)
            )
        )
        recognized_footer_credit = bool(
            bbox is not None and _looks_like_long_page_footer_credit_artifact(candidate, image_rgb, bbox)
        )
        recognized_top_ornament = bool(
            bbox is not None and _looks_like_top_chapter_ornament_artifact(candidate, image_rgb, bbox)
        )
        if not recognized_spurious_caption and not recognized_footer_credit and not recognized_top_ornament:
            return False
    elif status in {"no_image", "invalid_bbox", "empty_crop"}:
        return False
    if (
        status
        and status != "no_confident_cjk"
        and not recognized_spurious_caption
        and not recognized_footer_credit
        and not recognized_top_ornament
    ):
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    visual_sources = {"local_contrast", "white_near_chroma", "color_chroma", "red_chroma"}
    if source not in {"comic_text_detector_fallback", "anime_text_yolo_low_conf", *visual_sources}:
        return False
    if bbox is None:
        return False
    reason = ""
    if source in visual_sources and _looks_like_unconfirmed_visual_sfx_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_unconfirmed_visual_rejected"
    elif _looks_like_latin_ocr_text_artifact(candidate):
        reason = "sfx_artifact_latin_ocr_text_rejected"
    elif _looks_like_long_page_footer_credit_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_footer_credit_rejected"
    elif _looks_like_top_chapter_ornament_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_top_chapter_ornament_rejected"
    elif _looks_like_short_hangul_caption_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_short_hangul_caption_rejected"
    elif _looks_like_pale_vertical_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_pale_vertical_rejected"
    elif _looks_like_low_detail_logo_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_low_detail_logo_rejected"
    elif _looks_like_low_conf_bottom_color_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_low_conf_bottom_color_rejected"
    elif _looks_like_low_detail_blue_scene_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_low_detail_blue_scene_rejected"
    elif _looks_like_low_conf_warm_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_low_conf_warm_rejected"
    elif _looks_like_low_conf_horizontal_texture_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_low_conf_horizontal_texture_rejected"
    elif _looks_like_grid_or_building_artifact(image_rgb, bbox):
        reason = "sfx_artifact_grid_rejected"
    elif _looks_like_top_credit_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_top_credit_rejected"
    elif _looks_like_scanlator_logo_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_scanlator_logo_rejected"
    elif _looks_like_overbroad_low_conf_long_page_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_overbroad_low_conf_rejected"
    elif _looks_like_white_dialogue_or_caption_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_white_dialogue_rejected"
    elif _looks_like_dark_narration_caption_artifact(candidate, image_rgb, bbox):
        reason = "sfx_artifact_dark_caption_rejected"
    elif recognized_spurious_caption:
        reason = "sfx_artifact_spurious_cjk_caption_rejected"
    if not reason:
        return False
    candidate.setdefault("qa_flags", []).append(reason)
    sfx = candidate.setdefault("sfx", {})
    if isinstance(sfx, dict):
        sfx.setdefault("qa_flags", []).append(reason)
    return True


def _looks_like_short_hangul_caption_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
    recognized = str(candidate.get("recognized_text") or sfx_ocr.get("text") or "").strip()
    hangul_count = sum(1 for char in recognized if _contains_hangul(char))
    if hangul_count <= 0 or hangul_count > 2:
        return False
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source not in {"comic_text_detector_fallback", "anime_text_yolo_low_conf"}:
        return False
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(max(1, bh))
    width_ratio = bw / float(max(1, width))
    if width_ratio < 0.25 or not 1.0 <= aspect <= 3.25:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    rgb = rgb_u8.astype(np.float32)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    bright_ratio = float(np.mean(gray >= 220))
    dark_ratio = float(np.mean(gray <= 80))
    color_ratio = float(np.mean((saturation >= 45.0) & (value >= 45.0) & (chroma >= 28.0)))
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    return bright_ratio >= 0.60 and color_ratio <= 0.05 and 0.055 <= dark_ratio <= 0.30 and edge_ratio <= 0.16


def _looks_like_top_chapter_ornament_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source != "comic_text_detector_fallback":
        return False
    confidence = float(candidate.get("confidence") or 0.0)
    if confidence >= 0.16:
        return False
    x1, y1, x2, y2 = bbox
    y_center_ratio = ((y1 + y2) / 2.0) / float(max(1, height))
    width_ratio = max(1, x2 - x1) / float(max(1, width))
    if y_center_ratio > 0.14 or width_ratio < 0.14:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2HSV)
    dark_ratio = float(np.mean(gray <= 95))
    color_ratio = float(np.mean((hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 45)))
    return dark_ratio >= 0.18 or color_ratio >= 0.12


def _looks_like_pale_vertical_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source != "anime_text_yolo_low_conf":
        return False
    confidence = float(candidate.get("confidence") or 0.0)
    if confidence >= 0.035:
        return False
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(max(1, bh))
    if aspect >= 0.48:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 55, 150)
    bright_ratio = float(np.mean(gray >= 210))
    dark_ratio = float(np.mean(gray <= 85))
    edge_ratio = float(np.mean(edges > 0))
    return bright_ratio >= 0.45 and dark_ratio <= 0.08 and edge_ratio <= 0.08


def _looks_like_low_detail_logo_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source != "comic_text_detector_fallback":
        return False
    confidence = float(candidate.get("confidence") or 0.0)
    if confidence >= 0.50:
        return False
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(max(1, bh))
    if not 0.70 <= aspect <= 1.35:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 55, 150)
    color_ratio = float(np.mean((hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 45)))
    edge_ratio = float(np.mean(edges > 0))
    luma_std = float(np.std(gray.astype(np.float32)))
    return color_ratio <= 0.22 and edge_ratio <= 0.16 and luma_std <= 38.0


def _looks_like_low_conf_bottom_color_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source != "anime_text_yolo_low_conf":
        return False
    confidence = float(candidate.get("confidence") or 0.0)
    if confidence >= 0.025:
        return False
    x1, y1, x2, y2 = bbox
    y_center_ratio = ((y1 + y2) / 2.0) / float(max(1, height))
    if y_center_ratio < 0.82:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    rgb = rgb_u8.astype(np.float32)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    edges = cv2.Canny(gray, 55, 150)
    color_ratio = float(np.mean((hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 45) & (chroma >= 28)))
    edge_ratio = float(np.mean(edges > 0))
    bright_ratio = float(np.mean(gray >= 220))
    dark_ratio = float(np.mean(gray <= 80))
    return color_ratio >= 0.35 and bright_ratio >= 0.20 and dark_ratio >= 0.20 and edge_ratio <= 0.12


def _looks_like_low_detail_blue_scene_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source not in {"comic_text_detector_fallback", "anime_text_yolo_low_conf"}:
        return False
    confidence = float(candidate.get("confidence") or 0.0)
    if confidence >= 0.12:
        return False
    sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
    status = str(sfx_ocr.get("status") or "").strip().lower()
    if status not in {"", "no_confident_cjk"}:
        return False
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    blue_ratio = float(np.mean((hue >= 85.0) & (hue <= 125.0) & (saturation >= 25.0) & (value >= 45.0)))
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    dark_ratio = float(np.mean(gray <= 80))
    luma_std = float(np.std(gray.astype(np.float32)))
    return blue_ratio >= 0.82 and dark_ratio >= 0.45 and edge_ratio <= 0.16 and luma_std <= 38.0


def _looks_like_low_conf_warm_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source != "comic_text_detector_fallback":
        return False
    confidence = float(candidate.get("confidence") or 0.0)
    if confidence >= 0.04:
        return False
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    skin_or_wood = (
        (hue >= 5.0)
        & (hue <= 28.0)
        & (saturation >= 25.0)
        & (saturation <= 165.0)
        & (gray >= 45)
        & (gray <= 230)
    )
    skin_or_wood_ratio = float(np.mean(skin_or_wood))
    bright_ratio = float(np.mean(gray >= 220))
    dark_ratio = float(np.mean(gray <= 80))
    return skin_or_wood_ratio >= 0.14 and bright_ratio >= 0.20 and dark_ratio <= 0.14


def _looks_like_low_conf_horizontal_texture_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source != "comic_text_detector_fallback":
        return False
    confidence = float(candidate.get("confidence") or 0.0)
    if confidence >= 0.10:
        return False
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if bw / float(max(1, bh)) < 3.4:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    rgb = rgb_u8.astype(np.float32)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    blue_ratio = float(np.mean((hue >= 85.0) & (hue <= 125.0) & (saturation >= 25.0) & (value >= 45.0)))
    color_ratio = float(np.mean((saturation >= 45.0) & (value >= 45.0) & (chroma >= 28.0)))
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    return blue_ratio >= 0.45 and color_ratio <= 0.30 and edge_ratio <= 0.16


def _looks_like_latin_ocr_text_artifact(candidate: dict[str, Any]) -> bool:
    sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
    status = str(sfx_ocr.get("status") or "").strip().lower()
    if status not in {"", "no_confident_cjk"}:
        return False
    attempts = sfx_ocr.get("attempts") if isinstance(sfx_ocr.get("attempts"), list) else []
    high_conf_words = 0
    high_conf_chars = 0
    for attempt in attempts[:32]:
        if not isinstance(attempt, dict):
            continue
        text = str(attempt.get("text") or "").strip()
        if len(text) < 2 or not _mostly_latin_text(text):
            continue
        confidence = _as_float(attempt.get("confidence"), 0.0)
        if confidence < 0.82:
            continue
        high_conf_words += 1
        high_conf_chars += len(text)
    return high_conf_words >= 3 and high_conf_chars >= 10


def _looks_like_long_page_footer_credit_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source not in {"comic_text_detector_fallback", "anime_text_yolo_low_conf"}:
        return False
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(max(1, bh))
    y_center_ratio = ((y1 + y2) / 2.0) / float(max(1, height))
    width_ratio = bw / float(max(1, width))
    if y_center_ratio < 0.86 or width_ratio < 0.22 or aspect < 1.75:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    bright_ratio = float(np.mean(gray >= 210))
    dark_ratio = float(np.mean(gray <= 95))
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    return bright_ratio >= 0.34 and dark_ratio >= 0.02 and edge_ratio <= 0.18


def _looks_like_unconfirmed_visual_sfx_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source not in {"local_contrast", "white_near_chroma", "color_chroma", "red_chroma"}:
        return False
    sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
    status = str(sfx_ocr.get("status") or "").strip().lower()
    if status == "recognized":
        return False
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return False
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_ratio = (bw * bh) / float(max(1, height * width))
    width_ratio = bw / float(max(1, width))
    height_ratio = bh / float(max(1, height))
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    rgb = rgb_u8.astype(np.float32)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    dark_ratio = float(np.mean(gray <= 80))
    bright_ratio = float(np.mean(gray >= 220))
    color_ratio = float(np.mean((saturation >= 45.0) & (value >= 45.0) & (chroma >= 28.0)))
    red_ratio = float(np.mean((((hue <= 10.0) | (hue >= 170.0)) & (saturation >= 45.0) & (value >= 45.0))))
    cyan_ratio = float(np.mean((hue >= 75.0) & (hue <= 100.0) & (saturation >= 25.0) & (value >= 80.0)))
    blue_ratio = float(np.mean((hue >= 85.0) & (hue <= 125.0) & (saturation >= 25.0) & (value >= 45.0)))
    purple_ratio = float(np.mean((hue >= 120.0) & (hue <= 165.0) & (saturation >= 25.0) & (value >= 45.0)))
    skin_or_wood = (
        (hue >= 5.0)
        & (hue <= 28.0)
        & (saturation >= 25.0)
        & (saturation <= 165.0)
        & (gray >= 45)
        & (gray <= 230)
    )
    skin_or_wood_ratio = float(np.mean(skin_or_wood))

    if height / float(max(1, width)) < 3.0:
        black_on_white_sfx = (
            source == "local_contrast"
            and bright_ratio >= 0.70
            and dark_ratio >= 0.08
            and color_ratio <= 0.05
            and edge_ratio <= 0.08
        )
        dense_red_sfx = (
            red_ratio >= 0.18
            and (blue_ratio + purple_ratio) <= 0.20
            and skin_or_wood_ratio <= 0.06
            and dark_ratio >= 0.45
        )
        red_chroma_stroke_sfx = (
            source == "red_chroma"
            and skin_or_wood_ratio <= 0.06
            and dark_ratio >= 0.45
            and edge_ratio >= 0.14
        )
        if black_on_white_sfx or dense_red_sfx or red_chroma_stroke_sfx:
            return False
        if source == "red_chroma" and skin_or_wood_ratio >= 0.14:
            return True
        cyan_glow_sfx = (
            source == "local_contrast"
            and cyan_ratio >= 0.14
            and purple_ratio <= 0.08
            and skin_or_wood_ratio <= 0.03
            and edge_ratio >= 0.10
        )
        if cyan_glow_sfx:
            return False
        if skin_or_wood_ratio >= 0.12 and bright_ratio >= 0.25:
            return True
        if source == "white_near_chroma" and (
            area_ratio >= 0.075
            or bright_ratio >= 0.60
            or bw / float(max(1, bh)) >= 2.75
        ):
            return True
        if (blue_ratio + purple_ratio) >= 0.45:
            return True
        if source == "local_contrast" and blue_ratio >= 0.30 and dark_ratio <= 0.16 and bright_ratio >= 0.18:
            return True

    # Local-contrast crops with no CJK confirmation often latch onto hands,
    # floors and clothing folds. Real unconfirmed SFX can be black/white, but
    # should not be dominated by warm material colors.
    if source == "local_contrast" and skin_or_wood_ratio >= 0.45 and bright_ratio <= 0.42:
        return True

    # Large unconfirmed visual regions are usually panel art. Keep the common
    # black-on-white SFX case even when OCR fails.
    looks_like_white_ink_sfx = bright_ratio >= 0.62 and dark_ratio >= 0.08 and color_ratio <= 0.08
    if source == "local_contrast" and area_ratio >= 0.12 and not looks_like_white_ink_sfx:
        return True

    touches_vertical_edge = x1 <= 1 or x2 >= width - 1
    skinny_edge_sliver = width_ratio <= 0.16 and height_ratio >= 0.24
    if touches_vertical_edge and skinny_edge_sliver and source == "local_contrast":
        return True

    low_ink_colored_panel = (
        source == "local_contrast"
        and area_ratio >= 0.08
        and color_ratio >= 0.35
        and bright_ratio <= 0.18
        and edge_ratio <= 0.16
    )
    return low_ink_colored_panel


def _looks_like_spurious_recognized_caption_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
    recognized = str(candidate.get("recognized_text") or sfx_ocr.get("text") or "").strip()
    if _contains_hangul(recognized):
        return False
    if _looks_like_single_glyph_white_bubble_artifact(candidate, image_rgb, bbox):
        return True
    if _looks_like_dark_narration_caption_artifact(candidate, image_rgb, bbox):
        return True
    if _looks_like_white_dialogue_or_caption_artifact(candidate, image_rgb, bbox):
        return True
    attempts = sfx_ocr.get("attempts") if isinstance(sfx_ocr.get("attempts"), list) else []
    latin_attempts = 0
    for attempt in attempts[:24]:
        if not isinstance(attempt, dict):
            continue
        text = str(attempt.get("text") or "").strip()
        if len(text) >= 2 and _mostly_latin_text(text):
            latin_attempts += 1
    if latin_attempts < 4:
        return False
    x1, y1, x2, y2 = bbox
    aspect = max(1, x2 - x1) / float(max(1, y2 - y1))
    return aspect >= 1.65


def _looks_like_dark_narration_caption_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source not in {"comic_text_detector_fallback", "anime_text_yolo_low_conf"}:
        return False
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(max(1, bh))
    if aspect < 1.65:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dark_ratio = float(np.mean(gray <= 88))
    bright_ratio = float(np.mean(gray >= 205))
    color_ratio = float(np.mean((saturation >= 45) & (value >= 45)))
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    if dark_ratio >= 0.62 and bright_ratio >= 0.05 and edge_ratio <= 0.14:
        return True
    return dark_ratio >= 0.50 and bright_ratio >= 0.14 and color_ratio <= 0.08 and edge_ratio <= 0.13


def _looks_like_white_dialogue_or_caption_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source not in {"comic_text_detector_fallback", "anime_text_yolo_low_conf"}:
        return False
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(max(1, bh))
    if not 1.05 <= aspect <= 1.85:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright_ratio = float(np.mean(gray >= 220))
    dark_ratio = float(np.mean(gray <= 80))
    color_ratio = float(np.mean((saturation >= 45) & (value >= 45)))
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    return bright_ratio >= 0.66 and color_ratio <= 0.06 and dark_ratio <= 0.24 and edge_ratio <= 0.14


def _looks_like_single_glyph_white_bubble_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
    recognized = str(candidate.get("recognized_text") or sfx_ocr.get("text") or "").strip()
    if len(recognized) != 1:
        return False
    if _contains_hangul(recognized):
        return False
    confidence = float(sfx_ocr.get("confidence") or 0.0)
    if confidence >= 0.84:
        return False
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    bright_ratio = float(np.mean(gray >= 220))
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    return bright_ratio >= 0.72 and edge_ratio <= 0.055


def _looks_like_top_credit_artifact(candidate: dict[str, Any], image_rgb: np.ndarray, bbox: list[int]) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    confidence = float(candidate.get("confidence") or 0.0)
    if confidence >= 0.16:
        return False
    x1, y1, x2, y2 = bbox
    y_center_ratio = ((y1 + y2) / 2.0) / float(max(1, height))
    if y_center_ratio > 0.075:
        return False
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if bw < 70 or bh < 70:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    bright_ratio = float(np.mean(gray >= 220))
    dark_ratio = float(np.mean(gray <= 80))
    return bright_ratio >= 0.04 and dark_ratio >= 0.025


def _looks_like_scanlator_logo_artifact(candidate: dict[str, Any], image_rgb: np.ndarray, bbox: list[int]) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    confidence = float(candidate.get("confidence") or 0.0)
    if confidence >= 0.22:
        return False
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(max(1, bh))
    if not 0.70 <= aspect <= 1.65:
        return False
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    rgb_u8 = crop.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_ratio = float(np.mean((saturation >= 45) & (value >= 45)))
    dark_ratio = float(np.mean(gray <= 80))
    bright_ratio = float(np.mean(gray >= 220))
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    return color_ratio >= 0.74 and dark_ratio <= 0.18 and bright_ratio <= 0.08 and edge_ratio <= 0.20


def _looks_like_overbroad_low_conf_long_page_artifact(
    candidate: dict[str, Any],
    image_rgb: np.ndarray,
    bbox: list[int],
) -> bool:
    height, width = image_rgb.shape[:2]
    if height / float(max(1, width)) < 3.0:
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "").strip()
    if source not in {"anime_text_yolo_low_conf", "comic_text_detector_fallback"}:
        return False
    confidence = float(candidate.get("confidence") or 0.0)
    max_confidence = 0.02 if source == "anime_text_yolo_low_conf" else 0.12
    if confidence >= max_confidence:
        return False
    x1, y1, x2, y2 = bbox
    area_ratio = max(1, x2 - x1) * max(1, y2 - y1) / float(max(1, width * height))
    width_ratio = max(1, x2 - x1) / float(max(1, width))
    height_ratio = max(1, y2 - y1) / float(max(1, height))
    min_height_ratio = 0.055 if source == "anime_text_yolo_low_conf" else 0.030
    return area_ratio >= 0.018 or (width_ratio >= 0.55 and height_ratio >= min_height_ratio)


def _looks_like_grid_or_building_artifact(image_rgb: np.ndarray, bbox: list[int]) -> bool:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [
        max(0, min(width, int(bbox[0]))),
        max(0, min(height, int(bbox[1]))),
        max(0, min(width, int(bbox[2]))),
        max(0, min(height, int(bbox[3]))),
    ]
    if x2 <= x1 or y2 <= y1:
        return False
    bw = x2 - x1
    bh = y2 - y1
    if bw > 56 or bh > 74:
        return False
    crop = image_rgb[y1:y2, x1:x2].astype(np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    if edge_ratio < 0.15:
        return False
    min_line = max(8, min(bw, bh) // 3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=8,
        minLineLength=min_line,
        maxLineGap=3,
    )
    if lines is None or len(lines) < 8:
        return False
    angles: list[float] = []
    for line in lines[:40]:
        lx1, ly1, lx2, ly2 = [float(value) for value in line[0]]
        if lx1 == lx2 and ly1 == ly2:
            continue
        angle = abs(float(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))))
        if angle > 90.0:
            angle = 180.0 - angle
        angles.append(angle)
    if len(angles) < 8:
        return False
    diagonal = sum(45.0 <= angle <= 82.0 for angle in angles)
    vertical = sum(angle >= 84.0 for angle in angles)
    horizontal = sum(angle <= 16.0 for angle in angles)
    dominant_ratio = max(diagonal, vertical, horizontal) / float(len(angles))
    if diagonal < 6 or dominant_ratio < 0.54:
        return False
    if vertical / float(len(angles)) >= 0.28:
        return False
    dark_mask = gray <= 90
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_mask.astype(np.uint8), connectivity=8)
    component_count = sum(1 for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= 5)
    return component_count <= 3


def _candidate_payload(
    index: int,
    bbox: list[int],
    confidence: float,
    source: str,
    *,
    detector: str = "sfx_visual",
) -> dict[str, Any]:
    qa_flags = ["sfx_visual_candidate", "sfx_script_unknown"]
    if detector != "sfx_visual":
        qa_flags.append("sfx_text_detector_candidate")
    return {
        "id": f"sfx_visual_{index:03d}",
        "text_id": f"sfx_visual_{index:03d}",
        "bbox": list(bbox),
        "text_pixel_bbox": list(bbox),
        "source_bbox": list(bbox),
        "text": "",
        "original": "",
        "content_class": "sfx",
        "tipo": "sfx",
        "detector": detector,
        "confidence": float(confidence),
        "script": "unknown",
        "route_action": "review_required",
        "translate_policy": "review",
        "render_policy": "review_required",
        "qa_flags": qa_flags,
        "sfx": {
            "source_text": "",
            "adapted_text": "",
            "visual_detector": detector,
            "visual_confidence": float(confidence),
            "visual_source": source,
            "inpaint_allowed": False,
            "qa_flags": list(qa_flags),
        },
    }


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=_candidate_dedupe_priority, reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        bbox = candidate["bbox"]
        if any(_overlap_ratio(bbox, existing["bbox"], smaller=True) >= 0.48 for existing in kept):
            continue
        candidate["id"] = f"sfx_visual_{len(kept) + 1:03d}"
        candidate["text_id"] = candidate["id"]
        kept.append(candidate)
    kept = _suppress_color_fragments_near_white_candidates(kept)
    kept.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return kept


def _candidate_dedupe_priority(candidate: dict[str, Any]) -> tuple[int, float]:
    detector = str(candidate.get("detector") or "").strip()
    text_detector_priority = 1 if detector == "sfx_text_detector" else 0
    return (text_detector_priority, float(candidate.get("confidence") or 0.0))


def _merge_nearby_short_page_visual_candidates(
    candidates: list[dict[str, Any]],
    image_shape: tuple[int, int, int],
) -> list[dict[str, Any]]:
    """Merge fragmented visual SFX components on cropped/non-webtoon pages."""

    if not candidates:
        return []
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        return candidates
    if height / float(max(1, width)) >= 3.0:
        return candidates
    page_area = max(1, height * width)
    merged = [dict(candidate) for candidate in candidates]
    changed = True
    while changed:
        changed = False
        next_items: list[dict[str, Any]] = []
        used: set[int] = set()
        for index, candidate in enumerate(merged):
            if index in used:
                continue
            current = dict(candidate)
            current_bbox = _coerce_bbox(current.get("bbox"))
            if current_bbox is None:
                used.add(index)
                next_items.append(current)
                continue
            for other_index in range(index + 1, len(merged)):
                if other_index in used:
                    continue
                other = merged[other_index]
                other_bbox = _coerce_bbox(other.get("bbox"))
                if other_bbox is None:
                    continue
                if not _should_merge_short_page_visual_pair(current, other, current_bbox, other_bbox, page_area):
                    continue
                current = _merge_visual_candidate_payload(current, other, current_bbox, other_bbox)
                current_bbox = _coerce_bbox(current.get("bbox")) or current_bbox
                used.add(other_index)
                changed = True
            used.add(index)
            next_items.append(current)
        merged = next_items
    for index, candidate in enumerate(merged, start=1):
        candidate["id"] = f"sfx_visual_{index:03d}"
        candidate["text_id"] = candidate["id"]
    return merged


def _should_merge_short_page_visual_pair(
    candidate: dict[str, Any],
    other: dict[str, Any],
    bbox: list[int],
    other_bbox: list[int],
    page_area: int,
) -> bool:
    if candidate.get("detector") != "sfx_visual" or other.get("detector") != "sfx_visual":
        return False
    source = str((candidate.get("sfx") or {}).get("visual_source") or "")
    other_source = str((other.get("sfx") or {}).get("visual_source") or "")
    if source != other_source:
        return False
    if source not in {"local_contrast", "white_near_chroma", "color_chroma", "red_chroma"}:
        return False
    union = _bbox_union(bbox, other_bbox)
    union_area = max(1, (union[2] - union[0]) * (union[3] - union[1]))
    if union_area > int(page_area * 0.20):
        return False
    gap_x = max(0, max(bbox[0], other_bbox[0]) - min(bbox[2], other_bbox[2]))
    gap_y = max(0, max(bbox[1], other_bbox[1]) - min(bbox[3], other_bbox[3]))
    overlap_x = max(0, min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0]))
    overlap_y = max(0, min(bbox[3], other_bbox[3]) - max(bbox[1], other_bbox[1]))
    min_w = max(1, min(bbox[2] - bbox[0], other_bbox[2] - other_bbox[0]))
    min_h = max(1, min(bbox[3] - bbox[1], other_bbox[3] - other_bbox[1]))
    close_horizontal = gap_x <= 24 and overlap_y / float(min_h) >= 0.18
    close_vertical = gap_y <= 24 and overlap_x / float(min_w) >= 0.18
    return close_horizontal or close_vertical


def _merge_visual_candidate_payload(
    candidate: dict[str, Any],
    other: dict[str, Any],
    bbox: list[int],
    other_bbox: list[int],
) -> dict[str, Any]:
    merged = dict(candidate)
    union = _bbox_union(bbox, other_bbox)
    confidence = max(float(candidate.get("confidence") or 0.0), float(other.get("confidence") or 0.0))
    merged["bbox"] = union
    merged["text_pixel_bbox"] = union
    merged["source_bbox"] = union
    merged["confidence"] = round(float(confidence), 4)
    merged["qa_flags"] = _merge_unique_flags(
        candidate.get("qa_flags"),
        other.get("qa_flags"),
        ["sfx_visual_fragment_merged"],
    )
    sfx = dict(candidate.get("sfx") if isinstance(candidate.get("sfx"), dict) else {})
    other_sfx = other.get("sfx") if isinstance(other.get("sfx"), dict) else {}
    sfx["visual_confidence"] = round(float(confidence), 4)
    sfx["qa_flags"] = _merge_unique_flags(
        sfx.get("qa_flags"),
        other_sfx.get("qa_flags"),
        ["sfx_visual_fragment_merged"],
    )
    merged["sfx"] = sfx
    return merged


def _merge_unique_flags(*collections: Any) -> list[str]:
    flags: list[str] = []
    for collection in collections:
        if isinstance(collection, str):
            items = [collection]
        elif isinstance(collection, (list, tuple, set)):
            items = list(collection)
        else:
            items = []
        for item in items:
            value = str(item or "").strip()
            if value and value not in flags:
                flags.append(value)
    return flags


def _bbox_union(a: list[int], b: list[int]) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _looks_like_latin_balloon_text(crop_rgb: np.ndarray, source: str, bw: int, bh: int) -> bool:
    if source != "local_contrast" or bh <= 0:
        return False
    aspect = bw / float(max(1, bh))
    if aspect < 2.35:
        return False
    rgb = crop_rgb.astype(np.float32)
    gray = cv2.cvtColor(crop_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    bright_ratio = float(np.mean(gray >= 235))
    dark_ratio = float(np.mean(gray <= 85))
    mean_chroma = float(np.mean(chroma))
    return bright_ratio >= 0.42 and dark_ratio >= 0.08 and mean_chroma <= 8.0


def _suppress_color_fragments_near_white_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    white_candidates = [
        item for item in candidates
        if ((item.get("sfx") or {}).get("visual_source") == "white_near_chroma")
    ]
    if not white_candidates:
        return candidates
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        source = (candidate.get("sfx") or {}).get("visual_source")
        if source == "white_near_chroma" and _is_small_white_fragment(candidate, white_candidates):
            continue
        if source == "color_chroma":
            bbox = candidate.get("bbox") or []
            if any(_is_small_color_fragment_of_white_candidate(bbox, white.get("bbox") or []) for white in white_candidates):
                continue
        kept.append(candidate)
    for index, candidate in enumerate(kept, start=1):
        candidate["id"] = f"sfx_visual_{index:03d}"
        candidate["text_id"] = candidate["id"]
    return kept


def _is_small_white_fragment(candidate: dict[str, Any], white_candidates: list[dict[str, Any]]) -> bool:
    bbox = candidate.get("bbox") or []
    area = max(1, (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1])))
    for other in white_candidates:
        if other is candidate:
            continue
        other_bbox = other.get("bbox") or []
        other_area = max(1, (int(other_bbox[2]) - int(other_bbox[0])) * (int(other_bbox[3]) - int(other_bbox[1])))
        if area >= other_area:
            continue
        if area > int(other_area * 0.35):
            continue
        if _overlap_ratio(bbox, other_bbox, smaller=True) >= 0.25:
            return True
    return False


def _looks_like_character_artifact(crop_rgb: np.ndarray, source: str) -> bool:
    rgb_u8 = crop_rgb.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    if source == "local_contrast":
        rgb = rgb_u8.astype(np.float32)
        chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
        return float(np.mean(gray >= 230)) < 0.08 and float(np.mean(chroma)) < 16.0
    if source != "color_chroma":
        return False
    skin_or_brown = (
        (hue >= 5.0)
        & (hue <= 25.0)
        & (saturation >= 25.0)
        & (saturation <= 150.0)
        & (gray >= 55)
        & (gray <= 225)
    )
    rgb = rgb_u8.astype(np.float32)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    blueish = (blue - red > 12.0) & (blue - green > 0.0) & (saturation > 25.0)
    skin_ratio = float(np.mean(skin_or_brown))
    blue_ratio = float(np.mean(blueish))
    bright_ratio = float(np.mean(gray >= 230))
    if skin_ratio >= 0.28 and blue_ratio < 0.05:
        return True
    return bright_ratio >= 0.30 and 0.08 <= blue_ratio <= 0.22 and skin_ratio < 0.06


def _is_small_color_fragment_of_white_candidate(bbox: list[int], white_bbox: list[int]) -> bool:
    overlap = _overlap_ratio(bbox, white_bbox, smaller=True)
    if overlap < 0.12:
        return False
    area = max(1, (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1])))
    white_area = max(1, (int(white_bbox[2]) - int(white_bbox[0])) * (int(white_bbox[3]) - int(white_bbox[1])))
    return area <= int(white_area * 0.65)


def _suppressed_by_existing_text(bbox: list[int], text_bboxes: list[list[int]]) -> bool:
    return any(_overlap_ratio(bbox, text_bbox, smaller=True) >= 0.60 for text_bbox in text_bboxes)


def _looks_like_speech_balloon_text(
    bbox: list[int],
    block_bboxes: list[list[int]],
    image_shape: tuple[int, int, int],
) -> bool:
    height, width = image_shape[:2]
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw < width * 0.30 and bh < height * 0.16:
        return any(_overlap_ratio(bbox, block_bbox) >= 0.75 for block_bbox in block_bboxes)
    return False


def _overlap_ratio(a: list[int], b: list[int], *, smaller: bool = False) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    denom = min(a_area, b_area) if smaller else a_area
    return inter / float(max(1, denom))


def _contains_hangul(text: str) -> bool:
    return any(
        "\uac00" <= char <= "\ud7af"
        or "\u3130" <= char <= "\u318f"
        or "\u1100" <= char <= "\u11ff"
        for char in str(text or "")
    )


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3040" <= char <= "\u30ff"
        or "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uac00" <= char <= "\ud7af"
        or "\u3130" <= char <= "\u318f"
        or "\u1100" <= char <= "\u11ff"
        for char in str(text or "")
    )


def _mostly_latin_text(text: str) -> bool:
    letters = [char for char in str(text or "") if char.isalpha()]
    if len(letters) < 2:
        return False
    latin = sum(("A" <= char <= "Z") or ("a" <= char <= "z") for char in letters)
    return latin / float(max(1, len(letters))) >= 0.75


def _coerce_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _expand_bbox(bbox: list[int], width: int, height: int, *, pad: int) -> list[int]:
    return [
        max(0, int(bbox[0]) - pad),
        max(0, int(bbox[1]) - pad),
        min(width, int(bbox[2]) + pad),
        min(height, int(bbox[3]) + pad),
    ]
