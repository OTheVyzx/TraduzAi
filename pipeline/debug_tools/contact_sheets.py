from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np


def _image_from_attr(item: Any, attr: str) -> np.ndarray | None:
    image = getattr(item, attr, None)
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.size:
        return image
    return None


def _fit_tile(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    tile = np.full((height, width, 3), 245, dtype=np.uint8)
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
        return tile
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return tile
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    tile[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return tile


def _stack_rows(rows: list[np.ndarray], *, width: int, height: int) -> np.ndarray:
    if not rows:
        return np.full((height, width, 3), 245, dtype=np.uint8)
    return np.vstack(rows)


def translated_comparison_sheet(
    original_pages: Iterable[Any],
    translated_pages: Iterable[Any],
    *,
    tile_width: int = 260,
    tile_height: int = 360,
    max_pages: int = 12,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for original_page, translated_page in list(zip(original_pages, translated_pages))[:max_pages]:
        original = _image_from_attr(original_page, "image")
        translated = _image_from_attr(translated_page, "image")
        if original is None and translated is None:
            continue
        rows.append(
            np.hstack(
                [
                    _fit_tile(original if original is not None else translated, width=tile_width, height=tile_height),
                    _fit_tile(translated if translated is not None else original, width=tile_width, height=tile_height),
                ]
            )
        )
    return _stack_rows(rows, width=tile_width * 2, height=tile_height)


def problem_bands_sheet(
    bands: Iterable[Any],
    *,
    tile_width: int = 520,
    tile_height: int = 160,
    max_bands: int = 16,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for band in list(bands)[:max_bands]:
        rendered = _image_from_attr(band, "rendered_slice")
        cleaned = _image_from_attr(band, "cleaned_slice")
        original = _image_from_attr(band, "original_slice")
        if original is None:
            original = _image_from_attr(band, "strip_slice")
        image = rendered if rendered is not None else cleaned
        if image is None:
            image = original
        if image is None:
            continue
        rows.append(_fit_tile(image, width=tile_width, height=tile_height))
    return _stack_rows(rows, width=tile_width, height=tile_height)
