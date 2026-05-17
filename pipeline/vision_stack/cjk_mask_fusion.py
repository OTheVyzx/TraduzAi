from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

try:
    from .text_mask_evidence import TextEvidence, evidence_support_mask
except ImportError:
    from vision_stack.text_mask_evidence import TextEvidence, evidence_support_mask


def _component_mask(binary: np.ndarray) -> Iterable[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        yield labels == label


def _compact_punctuation_components(binary: np.ndarray) -> np.ndarray:
    recovered = np.zeros_like(binary, dtype=np.uint8)
    for component in _component_mask(binary):
        area = int(np.count_nonzero(component))
        if area < 3 or area > 240:
            continue
        ys, xs = np.where(component)
        width = int(xs.max()) - int(xs.min()) + 1
        height = int(ys.max()) - int(ys.min()) + 1
        if max(width, height) > 36:
            continue
        fill_ratio = area / float(max(1, width * height))
        if fill_ratio < 0.18:
            continue
        recovered[component] = 255
    return recovered


def _local_text_candidates(image_rgb: np.ndarray, support: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        color_image = image_rgb.astype(np.float32)
    else:
        gray = image_rgb.astype(np.uint8)
        saturation = np.zeros_like(gray)
        value = gray
        color_image = np.dstack([gray, gray, gray]).astype(np.float32)

    supported = support > 0
    if not np.any(supported):
        return np.zeros_like(support, dtype=np.uint8)

    local_gray = gray[supported]
    local_color = color_image[supported]
    local_median = float(np.median(local_gray))
    local_rgb_median = np.median(local_color, axis=0)
    color_delta = np.linalg.norm(color_image - local_rgb_median, axis=2)
    color_threshold = max(42.0, float(np.percentile(color_delta[supported], 68)))
    dark_threshold = min(132, int(np.percentile(local_gray, 38)) + 10)
    bright_threshold = max(172, int(np.percentile(local_gray, 82)) - 6)
    dark = gray <= dark_threshold
    bright = (gray >= bright_threshold) if local_median < 190.0 else np.zeros_like(dark, dtype=bool)
    saturated = (saturation >= 70) & (value >= 40) & (color_delta >= color_threshold)

    if local_median < 170.0:
        text_like = bright | saturated
    elif local_median > 205.0:
        text_like = dark | saturated
    else:
        text_like = dark | bright | saturated

    raw_candidates = (text_like & supported).astype(np.uint8) * 255
    candidates = cv2.morphologyEx(
        raw_candidates,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    candidates = np.maximum(candidates, _compact_punctuation_components(raw_candidates))
    return candidates


def _filter_candidates(candidates: np.ndarray, support: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    filtered = np.zeros_like(candidates, dtype=np.uint8)
    base_near = cv2.dilate(
        (base_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    for component in _component_mask(candidates):
        area = int(np.count_nonzero(component))
        if area < 4:
            continue
        ys, xs = np.where(component)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        fill_ratio = area / float(bbox_area)
        aspect = max(x2 - x1, y2 - y1) / float(max(1, min(x2 - x1, y2 - y1)))
        if fill_ratio > 0.92 and area > 256:
            continue
        if aspect > 26 and fill_ratio < 0.18:
            continue
        if not np.any(component & (support > 0)):
            continue
        compact_punctuation = area <= 240 and max(x2 - x1, y2 - y1) <= 36 and fill_ratio >= 0.18
        support_area = int(np.count_nonzero(support[y1:y2, x1:x2]))
        support_ratio = area / float(max(1, support_area))
        textlike_inside_evidence = (
            support_area > 0
            and area <= 14000
            and 0.035 <= support_ratio <= 0.82
            and 0.075 <= fill_ratio <= 0.86
            and aspect <= 18.0
        )
        if (
            np.any(component & (base_near > 0))
            or area <= 96
            or compact_punctuation
            or textlike_inside_evidence
        ):
            filtered[component] = 255
    return filtered


def _refine_broad_base_components(base_mask: np.ndarray, candidates: np.ndarray, support: np.ndarray) -> np.ndarray:
    if not np.any(base_mask) or not np.any(candidates) or not np.any(support):
        return base_mask

    refined = (base_mask > 0).astype(np.uint8) * 255
    candidate_bin = (candidates > 0).astype(np.uint8) * 255
    support_bin = support > 0
    link_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    for component in _component_mask(refined):
        area = int(np.count_nonzero(component))
        if area < 512:
            continue
        ys, xs = np.where(component)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        fill_ratio = area / float(bbox_area)
        support_overlap = int(np.count_nonzero(component & support_bin))
        if support_overlap < int(area * 0.35):
            continue

        local_component = component[y1:y2, x1:x2].astype(np.uint8) * 255
        local_candidates = candidate_bin[y1:y2, x1:x2]
        near_component = cv2.dilate(local_component, link_kernel, iterations=1)
        replacement = cv2.bitwise_and(local_candidates, near_component)
        replacement_area = int(np.count_nonzero(replacement))
        if replacement_area < 48:
            continue
        if area <= int(replacement_area * 1.55):
            continue
        if fill_ratio < 0.18 and area <= int(replacement_area * 2.4):
            continue

        refined[component] = 0
        refined[y1:y2, x1:x2] = np.maximum(refined[y1:y2, x1:x2], replacement)

    return refined.astype(np.uint8)


def fuse_cjk_text_mask(
    image_rgb: np.ndarray,
    base_mask: np.ndarray,
    evidence: list[TextEvidence],
    *,
    hi_sam_mask: np.ndarray | None = None,
    craft_heatmap: np.ndarray | None = None,
    allow_orphan_sfx: bool = True,
) -> np.ndarray:
    if base_mask.ndim == 3:
        base_mask = base_mask[:, :, 0]
    final = (base_mask > 0).astype(np.uint8) * 255
    if image_rgb.shape[:2] != final.shape[:2]:
        return final

    support = evidence_support_mask(final.shape[:2], evidence, pad=8)
    if np.any(support):
        candidates = _local_text_candidates(image_rgb, support)
        filtered_candidates = _filter_candidates(candidates, support, final)
        final = _refine_broad_base_components(final, filtered_candidates, support)
        final = np.maximum(final, filtered_candidates)

    if hi_sam_mask is not None and hi_sam_mask.shape[:2] == final.shape[:2]:
        if np.any(support):
            final = np.maximum(final, cv2.bitwise_and((hi_sam_mask > 0).astype(np.uint8) * 255, support))
        else:
            final = np.maximum(final, (hi_sam_mask > 0).astype(np.uint8) * 255)

    if craft_heatmap is not None and craft_heatmap.shape[:2] == final.shape[:2] and np.any(support):
        craft = ((craft_heatmap > 0.45) & (support > 0)).astype(np.uint8) * 255
        final = np.maximum(final, craft)

    if allow_orphan_sfx and np.any(final):
        final = cv2.morphologyEx(
            final,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    return final.astype(np.uint8)
