"""Adapter em-memoria do inpainter para o pipeline strip-based."""

from __future__ import annotations

import numpy as np


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


def _build_fallback_vision_blocks(ocr_page: dict, width: int, height: int) -> list[dict]:
    blocks: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for txt in ocr_page.get("texts", []):
        bbox = (
            _normalize_bbox(txt.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(txt.get("bbox"), width, height)
            or _normalize_bbox(txt.get("balloon_bbox"), width, height)
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
            }
        )
    return blocks


def inpaint_band_image(band_rgb: np.ndarray, ocr_page: dict) -> np.ndarray:
    """Aplica o mesmo round de inpaint do runtime principal na banda do strip."""
    from vision_stack.runtime import _apply_inpainting_round, _get_inpainter

    if band_rgb.size == 0 or not ocr_page.get("texts"):
        return band_rgb.copy()

    height, width = band_rgb.shape[:2]
    vision_blocks = list(ocr_page.get("_vision_blocks") or [])
    if not vision_blocks:
        vision_blocks = _build_fallback_vision_blocks(ocr_page, width, height)
    if not vision_blocks:
        return band_rgb.copy()

    inpaint_payload = dict(ocr_page)
    inpaint_payload["_vision_blocks"] = vision_blocks
    inpainter = _get_inpainter("quality")
    cleaned = _apply_inpainting_round(band_rgb, inpaint_payload, inpainter)
    return cleaned.copy() if cleaned is band_rgb else cleaned
