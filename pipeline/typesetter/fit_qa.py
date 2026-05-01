"""Typesetting fit QA helpers."""

from __future__ import annotations

from typing import Any, Callable


def assess_fit(layout: dict[str, Any]) -> dict[str, Any]:
    bbox = layout.get("bbox") or [0, 0, 0, 0]
    text_bbox = layout.get("text_bbox") or bbox
    font_size = int(layout.get("font_size", 0) or 0)
    lines = int(layout.get("lines", 1) or 1)
    margin = min(text_bbox[0] - bbox[0], text_bbox[1] - bbox[1], bbox[2] - text_bbox[2], bbox[3] - text_bbox[3])
    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    text_area = max(0, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
    occupancy = text_area / bbox_area
    overflow = margin < 0 or occupancy > 0.82 or font_size < 16 or (bbox_area < 20_000 and lines > 4)
    flags = []
    if margin < 8:
        flags.append("typesetting_margin_low")
    if occupancy > 0.82:
        flags.append("text_overflow")
    if font_size < 16:
        flags.append("font_too_small")
    if bbox_area < 20_000 and lines > 4:
        flags.append("too_many_lines")
    return {"ok": not overflow and margin >= 8, "occupancy": occupancy, "margin": margin, "flags": flags}


def fit_text(
    text: str,
    layout: dict[str, Any],
    shortener: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    current = dict(layout)
    assessment = assess_fit(current)
    if assessment["ok"]:
        return {"text": text, "layout": current, "qa_flags": [], "method": "fits"}
    if current.get("font_size", 0) > 16:
        current["font_size"] = max(16, int(current["font_size"]) - 4)
        assessment = assess_fit(current)
        if assessment["ok"]:
            return {"text": text, "layout": current, "qa_flags": [], "method": "reduced_font"}
    if shortener is not None:
        shortened = shortener(text)
        current["lines"] = max(1, int(current.get("lines", 1) or 1) - 1)
        current["text_bbox"] = _shrink_text_bbox(current.get("text_bbox") or current.get("bbox"))
        assessment = assess_fit(current)
        if assessment["ok"]:
            return {"text": shortened, "layout": current, "qa_flags": [], "method": "shortened"}
    return {"text": text, "layout": current, "qa_flags": ["text_overflow"], "method": "overflow"}


def _shrink_text_bbox(bbox: list[int]) -> list[int]:
    return [bbox[0] + 8, bbox[1] + 8, bbox[2] - 8, bbox[3] - 8]

