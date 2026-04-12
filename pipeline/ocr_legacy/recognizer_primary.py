"""
Primary OCR pass using EasyOCR over the full page.
"""

from __future__ import annotations

import cv2

from .postprocess import merge_ocr_runs


def run_primary_recognition(reader, preprocessed_image) -> list[dict]:
    normal_results = reader.readtext(
        preprocessed_image,
        text_threshold=0.4,
        low_text=0.3,
        canvas_size=2560,
    )
    inverted_results = reader.readtext(
        cv2.bitwise_not(preprocessed_image),
        text_threshold=0.4,
        low_text=0.3,
        canvas_size=2560,
    )

    primary_runs = [
        {
            "bbox_pts": bbox_pts,
            "text": text,
            "confidence": float(confidence),
            "source": "primary",
        }
        for bbox_pts, text, confidence in normal_results
    ]
    inverted_runs = [
        {
            "bbox_pts": bbox_pts,
            "text": text,
            "confidence": float(confidence),
            "source": "primary-inverted",
        }
        for bbox_pts, text, confidence in inverted_results
    ]
    return merge_ocr_runs(primary_runs, inverted_runs)
