from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class UiLayoutComponent:
    bbox: list[int]
    component_type: str
    background_rgb: list[int]
    confidence: float
    source: str = "uied_cv"


def _bbox_area(bbox: list[int]) -> int:
    return max(1, int(bbox[2] - bbox[0]) * int(bbox[3] - bbox[1]))


def _bbox_intersection(a: list[int], b: list[int]) -> list[int] | None:
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_intersection_area(a: list[int], b: list[int]) -> int:
    inter = _bbox_intersection(a, b)
    return 0 if inter is None else _bbox_area(inter)


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _coerce_bbox(value: object) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_from_polygons(polygons: object) -> list[int] | None:
    if not isinstance(polygons, list):
        return None
    xs: list[int] = []
    ys: list[int] = []
    for polygon in polygons:
        if not isinstance(polygon, (list, tuple)):
            continue
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
    return [min(xs), min(ys), max(xs), max(ys)]


def _text_anchor_bbox(text: dict) -> list[int] | None:
    return (
        _coerce_bbox(_bbox_from_polygons(text.get("line_polygons")))
        or _coerce_bbox(text.get("text_pixel_bbox"))
        or _coerce_bbox(text.get("source_bbox"))
        or _coerce_bbox(text.get("bbox"))
    )


def _expanded_bbox(bbox: list[int], width: int, height: int, pad_x: int, pad_y: int) -> list[int]:
    return [
        max(0, int(bbox[0]) - pad_x),
        max(0, int(bbox[1]) - pad_y),
        min(width, int(bbox[2]) + pad_x),
        min(height, int(bbox[3]) + pad_y),
    ]


def _median_rgb(image_rgb: np.ndarray, bbox: list[int]) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return [255, 255, 255]
    rgb = np.median(crop.reshape(-1, 3), axis=0)
    return [int(max(0, min(255, round(float(v))))) for v in rgb]


def _masked_median_rgb(image_rgb: np.ndarray, mask: np.ndarray, bbox: list[int]) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = image_rgb[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2] > 0
    if crop.size == 0 or not np.any(mask_crop):
        return _median_rgb(image_rgb, bbox)
    rgb = np.median(crop[mask_crop].reshape(-1, 3), axis=0)
    return [int(max(0, min(255, round(float(v))))) for v in rgb]


def _component_type_from_bbox(bbox: list[int], image_shape: tuple[int, int, int]) -> str:
    height, width = image_shape[:2]
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    aspect = bw / float(bh)
    if aspect >= 6.0 and bw >= int(width * 0.30):
        return "ui_panel"
    if aspect >= 3.0:
        return "ui_input"
    return "ui_component"


def _iter_runs(active: np.ndarray, *, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active.tolist()):
        if bool(value):
            if start is None:
                start = index
            continue
        if start is not None and index - start >= min_len:
            runs.append((start, index))
        start = None
    if start is not None and len(active) - start >= min_len:
        runs.append((start, len(active)))
    return runs


def _component_from_bbox(image_rgb: np.ndarray, mask: np.ndarray, bbox: list[int]) -> UiLayoutComponent | None:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    bw = x2 - x1
    bh = y2 - y1
    if bw < 42 or bh < 6:
        return None
    aspect = bw / float(max(1, bh))
    if aspect < 2.0:
        return None
    rect_area = _bbox_area(bbox)
    area = int(np.count_nonzero(mask[y1:y2, x1:x2] > 0))
    fill_ratio = area / float(rect_area)
    if fill_ratio < 0.34:
        return None
    crop = image_rgb[y1:y2, x1:x2].astype(np.float32)
    if crop.size == 0:
        return None
    channel_std = float(np.mean(np.std(crop.reshape(-1, 3), axis=0)))
    if channel_std > 62.0:
        return None
    confidence = min(0.98, max(0.35, fill_ratio * 0.58 + min(1.0, aspect / 10.0) * 0.24 + 0.12))
    return UiLayoutComponent(
        bbox=[x1, y1, x2, y2],
        component_type=_component_type_from_bbox([x1, y1, x2, y2], (height, width, 3)),
        background_rgb=_median_rgb(image_rgb, [x1, y1, x2, y2]),
        confidence=round(float(confidence), 3),
    )


def _detect_horizontal_band_components(image_rgb: np.ndarray, mask: np.ndarray) -> list[UiLayoutComponent]:
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return []
    row_coverage = np.count_nonzero(mask > 0, axis=1) / float(width)
    active_rows = row_coverage >= 0.10
    components: list[UiLayoutComponent] = []
    for y1, y2 in _iter_runs(active_rows, min_len=6):
        band_h = y2 - y1
        if band_h > max(96, int(height * 0.16)):
            continue
        band_mask = mask[y1:y2, :]
        col_coverage = np.count_nonzero(band_mask > 0, axis=0) / float(max(1, band_h))
        active_cols = col_coverage >= 0.30
        for x1, x2 in _iter_runs(active_cols, min_len=42):
            component = _component_from_bbox(image_rgb, mask, [x1, y1, x2, y2])
            if component is not None:
                components.append(component)
    return components


def _detect_perspective_panel_components(image_rgb: np.ndarray, mask: np.ndarray) -> list[UiLayoutComponent]:
    height, width = image_rgb.shape[:2]
    page_area = max(1, height * width)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[UiLayoutComponent] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(360.0, float(page_area) * 0.0008):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 48 or bh < 14:
            continue
        if area > float(page_area) * 0.35:
            continue
        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = [float(v) for v in rect[1]]
        if rect_w <= 1.0 or rect_h <= 1.0:
            continue
        rotated_aspect = max(rect_w, rect_h) / max(1.0, min(rect_w, rect_h))
        axis_aspect = bw / float(max(1, bh))
        if axis_aspect < 0.85:
            continue
        if axis_aspect >= 2.0 and rotated_aspect / max(1.0, axis_aspect) < 1.15:
            continue
        if rotated_aspect < 2.0 and axis_aspect < 2.0:
            continue
        rect_fill_ratio = area / max(1.0, rect_w * rect_h)
        if rect_fill_ratio < 0.45:
            continue
        crop = image_rgb[y : y + bh, x : x + bw].astype(np.float32)
        mask_crop = mask[y : y + bh, x : x + bw] > 0
        if crop.size == 0 or not np.any(mask_crop):
            continue
        masked_pixels = crop[mask_crop].reshape(-1, 3)
        channel_std = float(np.mean(np.std(masked_pixels, axis=0)))
        if channel_std > 72.0:
            continue
        fill_ratio = float(np.count_nonzero(mask_crop)) / float(max(1, bw * bh))
        confidence = min(
            0.98,
            max(0.38, rect_fill_ratio * 0.36 + min(1.0, rotated_aspect / 8.0) * 0.30 + fill_ratio * 0.20),
        )
        bbox = [int(x), int(y), int(x + bw), int(y + bh)]
        components.append(
            UiLayoutComponent(
                bbox=bbox,
                component_type=_component_type_from_bbox(bbox, (height, width, 3)),
                background_rgb=_masked_median_rgb(image_rgb, mask, bbox),
                confidence=round(float(confidence), 3),
            )
        )
    return components


def _candidate_mask(image_rgb: np.ndarray) -> np.ndarray:
    rgb = image_rgb.astype(np.int16)
    luma = (
        image_rgb[:, :, 0].astype(np.float32) * 0.299
        + image_rgb[:, :, 1].astype(np.float32) * 0.587
        + image_rgb[:, :, 2].astype(np.float32) * 0.114
    )
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    not_page_white = (luma <= 246.0) & ((chroma >= 4) | (luma <= 232.0))
    not_ink_black = luma >= 42.0
    return (not_page_white & not_ink_black).astype(np.uint8) * 255


def detect_uied_like_components(image_rgb: np.ndarray) -> list[UiLayoutComponent]:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
        return []
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return []

    raw_mask = _candidate_mask(image_rgb)
    components: list[UiLayoutComponent] = _detect_perspective_panel_components(image_rgb, raw_mask)
    mask = raw_mask
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components.extend(_detect_horizontal_band_components(image_rgb, mask))
    page_area = max(1, width * height)
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if bw < 42 or bh < 6:
            continue
        aspect = bw / float(max(1, bh))
        if aspect < 2.0:
            continue
        max_area_ratio = 0.35 if aspect >= 3.0 else 0.20
        if area > int(page_area * max_area_ratio):
            continue
        component = _component_from_bbox(image_rgb, mask, [x, y, x + bw, y + bh])
        if component is not None:
            components.append(component)

    components.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.bbox[2] - item.bbox[0]))
    return _dedupe_components(components)


def _dedupe_components(components: list[UiLayoutComponent]) -> list[UiLayoutComponent]:
    kept: list[UiLayoutComponent] = []
    for component in components:
        duplicate = False
        for existing in kept:
            inter = _bbox_intersection_area(component.bbox, existing.bbox)
            smaller = min(_bbox_area(component.bbox), _bbox_area(existing.bbox))
            if inter / float(max(1, smaller)) >= 0.82:
                duplicate = True
                break
        if not duplicate:
            kept.append(component)
    return kept


def _component_payload(component: UiLayoutComponent, role: str) -> dict:
    return {
        "source": component.source,
        "role": role,
        "component_type": component.component_type,
        "component_bbox": list(component.bbox),
        "background_rgb": list(component.background_rgb),
        "confidence": float(component.confidence),
    }


def _matching_component_for_text(text_bbox: list[int], components: Iterable[UiLayoutComponent]) -> UiLayoutComponent | None:
    text_area = _bbox_area(text_bbox)
    best: tuple[float, UiLayoutComponent] | None = None
    cx, cy = _bbox_center(text_bbox)
    for component in components:
        bx1, by1, bx2, by2 = component.bbox
        inside_center = bx1 <= cx <= bx2 and by1 <= cy <= by2
        overlap = _bbox_intersection_area(text_bbox, component.bbox) / float(max(1, text_area))
        score = overlap + (0.35 if inside_center else 0.0)
        if score < 0.18:
            continue
        if best is None or score > best[0]:
            best = (score, component)
    return None if best is None else best[1]


def _nearby_component_count(text_bbox: list[int], components: Iterable[UiLayoutComponent], image_shape: tuple[int, int, int]) -> int:
    height, width = image_shape[:2]
    search = _expanded_bbox(
        text_bbox,
        width,
        height,
        pad_x=max(48, int((text_bbox[2] - text_bbox[0]) * 0.80)),
        pad_y=max(28, int((text_bbox[3] - text_bbox[1]) * 1.20)),
    )
    count = 0
    for component in components:
        if _bbox_intersection_area(search, component.bbox) > 0:
            count += 1
    return count


def attach_uied_layout_evidence(
    image_rgb: np.ndarray,
    texts: list[dict],
    *,
    components: list[UiLayoutComponent] | None = None,
) -> tuple[list[dict], list[dict]]:
    if os.getenv("TRADUZAI_UIED_LAYOUT", "1").strip().lower() in {"0", "false", "no", "off"}:
        return texts, []
    if not isinstance(texts, list) or not texts:
        return texts, []
    detected = components if components is not None else detect_uied_like_components(image_rgb)
    if not detected:
        return texts, []

    for text in texts:
        if not isinstance(text, dict):
            continue
        anchor = _text_anchor_bbox(text)
        if anchor is None:
            continue
        matched = _matching_component_for_text(anchor, detected)
        if matched is not None:
            text["ui_layout_evidence"] = _component_payload(matched, "text_inside_component")
            text["background_rgb"] = list(matched.background_rgb)
            text["layout_profile"] = "ui_form"
            text["block_profile"] = "ui_form"
            text.setdefault("layout_safe_reason", "uied_cv_component")
            continue
        nearby = _nearby_component_count(anchor, detected, image_rgb.shape)
        if nearby >= 2:
            text["ui_layout_evidence"] = {
                "source": "uied_cv",
                "role": "label_near_components",
                "nearby_component_count": int(nearby),
                "text_anchor_bbox": list(anchor),
                "confidence": 0.52,
            }
            text["layout_profile"] = "ui_form"
            text["block_profile"] = "ui_form"
            text.setdefault("layout_safe_reason", "uied_cv_label")

    return texts, [_component_payload(component, "detected_component") for component in detected]
