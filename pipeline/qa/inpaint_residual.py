from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _as_rgb_array(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("image must be HxW, HxWx3, or HxWx4")
    return arr[:, :, :3].astype(np.uint8, copy=False)


def _as_region_mask(mask: Any | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape != shape:
        raise ValueError("mask shape must match image height and width")
    return arr > 0


def _gray(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    return (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114).astype(np.float32)


def _local_light_contrast(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.float32)
    min_side = min(gray.shape[:2])
    if min_side < 5:
        return np.zeros_like(gray, dtype=np.float32)
    kernel = max(5, min(31, (min_side // 3) | 1))
    if kernel % 2 == 0:
        kernel += 1
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0).astype(np.float32)
    return gray.astype(np.float32) - background


def _channel_spread(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    return (np.max(rgb, axis=2) - np.min(rgb, axis=2)).astype(np.float32)


def detect_residual_text(
    before_rgb: Any,
    after_rgb: Any,
    mask: Any | None = None,
    *,
    dark_threshold: int = 112,
    light_threshold: int = 212,
    min_changed_delta: int = 12,
    min_light_contrast: int = 14,
    min_pixels: int = 8,
    min_ratio: float = 0.01,
    include_unchanged_dark: bool = False,
    include_light_residual: bool = False,
) -> dict:
    """Cheap threshold check for dark or bright text-like remnants after inpaint."""
    before = _as_rgb_array(before_rgb)
    after = _as_rgb_array(after_rgb)
    if before.shape != after.shape:
        raise ValueError("before_rgb and after_rgb must have the same shape")
    if before.size == 0:
        return {"has_residual": False, "score": 0.0, "flags": ["empty_image"]}

    region = _as_region_mask(mask, before.shape[:2])
    region_pixels = int(np.count_nonzero(region))
    if region_pixels <= 0:
        return {"has_residual": False, "score": 0.0, "flags": ["empty_region"]}

    before_gray = _gray(before)
    after_gray = _gray(after)
    before_spread = _channel_spread(before)
    after_spread = _channel_spread(after)
    delta = np.abs(after_gray - before_gray)
    dark_before = before_gray <= float(dark_threshold)
    dark_after = after_gray <= float(dark_threshold)
    changed_dark = delta >= float(min_changed_delta)
    residual_core = changed_dark
    if include_unchanged_dark:
        residual_core = residual_core | dark_before
    nearby_dark_context = cv2.dilate(
        (before_gray <= float(dark_threshold + 48)).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    ).astype(bool)
    region_before = before_gray[region]
    dark_background_context = False
    if region_before.size:
        non_light = region_before[region_before < float(light_threshold)]
        dark_background_context = bool(
            non_light.size >= max(16, int(region_pixels * 0.12))
            and float(np.median(non_light)) <= float(dark_threshold + 64)
        )
    expected_dark_fill = (
        (before_gray >= float(light_threshold))
        & (nearby_dark_context | dark_background_context)
        & dark_after
    )
    after_abs_contrast = np.abs(_local_light_contrast(after_gray))
    source_colored_text_fill = (
        dark_background_context
        & (before_spread >= 52.0)
        & (before_gray <= float(light_threshold))
        & dark_after
        & changed_dark
        & (after_abs_contrast < float(max(min_light_contrast, 18)))
    )
    expected_dark_fill = expected_dark_fill | source_colored_text_fill
    dark_text_like = after_abs_contrast >= float(max(8, min_light_contrast))
    dark_residual = region & dark_after & residual_core & dark_text_like & ~expected_dark_fill
    source_light_text_on_dark = bool(
        include_light_residual
        and dark_background_context
        and (int(np.count_nonzero(region & (before_gray >= float(light_threshold)))) / float(region_pixels)) >= 0.03
    )
    if source_light_text_on_dark:
        dark_residual = np.zeros_like(region, dtype=bool)
    dark_residual_pixels = int(np.count_nonzero(dark_residual))

    if include_light_residual:
        before_light_contrast = _local_light_contrast(before_gray)
        light_contrast = _local_light_contrast(after_gray)
        light_before = before_gray >= float(light_threshold)
        light_after = after_gray >= float(light_threshold)
        light_residual = (
            region
            & light_before
            & light_after
            & (before_light_contrast >= float(min_light_contrast))
            & (light_contrast >= float(min_light_contrast))
        )
        light_residual_pixels = int(np.count_nonzero(light_residual))
        color_delta = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=2)
        colored_residual = (
            region
            & dark_background_context
            & (before_spread >= 52.0)
            & (after_spread >= 52.0)
            & (after_gray <= float(light_threshold))
            & (after_abs_contrast >= float(min_light_contrast))
            & (color_delta <= float(max(min_changed_delta * 2, 24)))
        )
        colored_residual_pixels = int(np.count_nonzero(colored_residual))
    else:
        light_residual = np.zeros_like(region, dtype=bool)
        light_residual_pixels = 0
        colored_residual = np.zeros_like(region, dtype=bool)
        colored_residual_pixels = 0

    residual_pixels = int(np.count_nonzero(dark_residual | light_residual | colored_residual))
    score = round(float(residual_pixels) / float(region_pixels), 6)

    flags: list[str] = []
    absolute_pixel_gate = max(int(min_pixels), 32)
    has_residual = residual_pixels >= int(min_pixels) and (
        score >= float(min_ratio) or residual_pixels >= absolute_pixel_gate
    )
    if has_residual:
        if dark_residual_pixels:
            flags.append("dark_residual_pixels")
        if light_residual_pixels:
            flags.append("light_residual_pixels")
        if colored_residual_pixels:
            flags.append("colored_residual_pixels")
        if score >= 0.05:
            flags.append("high_residual_ratio")

    return {
        "has_residual": bool(has_residual),
        "score": score,
        "flags": flags,
        "dark_residual_pixels": dark_residual_pixels,
        "light_residual_pixels": light_residual_pixels,
        "colored_residual_pixels": colored_residual_pixels,
        "light_residual_on_dark_context": bool(source_light_text_on_dark),
        "dark_background_context": bool(dark_background_context),
    }
