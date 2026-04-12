"""
Fallback OCR pass for low-confidence or suspicious regions.
The goal is to re-read difficult balloons without rescanning the whole page.
"""

from __future__ import annotations

import cv2

from .postprocess import fix_ocr_errors, looks_suspicious

PROFILE_THRESHOLDS = {
    "compat": 0.55,
    "quality": 0.62,
    "max": 0.72,
}

def needs_fallback(text: str, confidence: float, profile: str = "quality") -> bool:
    threshold = PROFILE_THRESHOLDS.get(profile, PROFILE_THRESHOLDS["quality"])
    return confidence < threshold or looks_suspicious(text, confidence)


def run_fallback_recognition(
    reader,
    image_bgr,
    bbox: list[int],
    primary_text: str,
    primary_confidence: float,
    profile: str = "quality",
) -> list[dict]:
    if not needs_fallback(primary_text, primary_confidence, profile):
        return []

    crop = _crop_region(image_bgr, bbox)
    if crop is None or crop.size == 0:
        return []

    variants = {
        "fallback-upscale": _upscale_variant(crop),
        "fallback-threshold": _threshold_variant(crop),
        "fallback-inverted": cv2.bitwise_not(_threshold_variant(crop)),
    }

    candidates = []
    for source, variant in variants.items():
        try:
            results = reader.readtext(
                variant,
                text_threshold=0.3,
                low_text=0.2,
                canvas_size=2048,
            )
        except Exception:
            continue

        best_text, best_confidence = _best_crop_reading(results)
        if not best_text:
            continue

        candidates.append(
            {
                "text": fix_ocr_errors(best_text),
                "confidence": round(float(best_confidence), 3),
                "source": source,
            }
        )

    return candidates


def _crop_region(image_bgr, bbox: list[int]):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    pad_x = max(6, int(width * 0.22))
    pad_y = max(6, int(height * 0.3))
    max_h, max_w = image_bgr.shape[:2]
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(max_w, x2 + pad_x)
    cy2 = min(max_h, y2 + pad_y)
    return image_bgr[cy1:cy2, cx1:cx2]


def _upscale_variant(crop):
    scale = 3.0 if max(crop.shape[:2]) < 128 else 2.0
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _threshold_variant(crop):
    gray = cv2.cvtColor(_upscale_variant(crop), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def _best_crop_reading(results) -> tuple[str, float]:
    best_text = ""
    best_confidence = 0.0
    for _, text, confidence in results:
        cleaned = text.strip()
        if cleaned and float(confidence) > best_confidence:
            best_text = cleaned
            best_confidence = float(confidence)
    return best_text, best_confidence
