"""Simple OCR-position geometry for one-text-to-one-render flows."""

from __future__ import annotations

from typing import Any


CONNECTED_LIST_KEYS = (
    "balloon_subregions",
    "connected_lobe_bboxes",
    "connected_lobe_polygons",
    "connected_position_bboxes",
    "connected_focus_bboxes",
    "connected_text_groups",
)

CONNECTED_CONFIDENCE_KEYS = (
    "connected_detection_confidence",
    "connected_group_confidence",
    "connected_position_confidence",
    "subregion_confidence",
)


def _valid_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def resolve_text_anchor_bbox(text: dict) -> list[int] | None:
    """Return the exact OCR text position preferred for simple rendering."""

    return (
        _valid_bbox(text.get("text_pixel_bbox"))
        or _valid_bbox(text.get("source_bbox"))
        or _valid_bbox(text.get("bbox"))
    )


def _non_connected_profile(value: Any, fallback: str = "standard") -> str:
    profile = str(value or "").strip()
    if not profile or profile == "connected_balloon":
        return fallback
    return profile


def sanitize_simple_text_geometry(text: dict) -> dict:
    """Keep raw OCR geometry while forcing layout/render geometry to one text."""

    updated = dict(text)
    anchor = resolve_text_anchor_bbox(updated)
    if anchor is not None:
        updated["layout_bbox"] = list(anchor)
        updated["balloon_bbox"] = list(anchor)
        updated["text_pixel_bbox"] = list(anchor)
    elif _valid_bbox(updated.get("layout_bbox")) is not None:
        updated["layout_bbox"] = _valid_bbox(updated.get("layout_bbox"))
        updated["balloon_bbox"] = list(updated["layout_bbox"])
    elif _valid_bbox(updated.get("balloon_bbox")) is not None:
        updated["balloon_bbox"] = _valid_bbox(updated.get("balloon_bbox"))
        updated["layout_bbox"] = list(updated["balloon_bbox"])

    source_bbox = _valid_bbox(updated.get("source_bbox"))
    raw_bbox = _valid_bbox(updated.get("bbox"))
    updated["ocr_text_bbox"] = raw_bbox or source_bbox or resolve_text_anchor_bbox(updated) or []

    updated["layout_group_size"] = 1
    for key in CONNECTED_LIST_KEYS:
        updated[key] = []
    for key in CONNECTED_CONFIDENCE_KEYS:
        updated[key] = 0.0

    updated["connected_balloon_orientation"] = ""
    updated["connected_position_reasoner"] = ""
    updated["connected_reasoner_model"] = ""
    updated["connected_reasoner_notes"] = ""
    updated.pop("connected_children", None)
    updated.pop("_connected_slot_index", None)
    updated.pop("_connected_slot_count", None)
    updated.pop("_connected_vertical_bias_ratio", None)
    updated["_is_lobe_subregion"] = False

    fallback_profile = _non_connected_profile(updated.get("block_profile"))
    updated["layout_profile"] = _non_connected_profile(updated.get("layout_profile"), fallback_profile)
    return updated
