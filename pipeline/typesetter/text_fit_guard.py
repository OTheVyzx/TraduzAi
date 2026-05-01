"""
Validates that rendered text ink stays within the safe area of a balloon.
Generates structured flags for the QA pipeline.
"""
from __future__ import annotations

from typing import Optional


def validate_rendered_text_fit(
    *,
    page_width: int,
    page_height: int,
    target_bbox: list[int],
    safe_bbox: list[int],
    ink_bbox: list[int],
    balloon_bbox: Optional[list[int]],
    region_id: str,
    page: int,
) -> dict:
    """
    Check whether ink_bbox is fully contained within safe_bbox.

    Args:
        page_width / page_height: full page dimensions (pixels).
        target_bbox: the layout bbox the renderer aimed for [x1,y1,x2,y2].
        safe_bbox: padded inner area allowed for text [x1,y1,x2,y2].
        ink_bbox: actual bounding box of rendered glyph pixels [x1,y1,x2,y2].
        balloon_bbox: outer balloon bbox, if available.
        region_id: identifier like "p012_r003".
        page: page number (1-indexed).

    Returns:
        {
            "ok": bool,
            "flags": list[dict]   # empty when ok is True
        }
    """
    flags: list[dict] = []

    sx1, sy1, sx2, sy2 = safe_bbox
    ix1, iy1, ix2, iy2 = ink_bbox

    # --- text_clipped: ink leaks outside safe_bbox on any side ---
    if ix1 < sx1 or iy1 < sy1 or ix2 > sx2 or iy2 > sy2:
        evidence: dict = {
            "ink_bbox": ink_bbox,
            "safe_bbox": safe_bbox,
        }
        if balloon_bbox:
            evidence["balloon_bbox"] = balloon_bbox
        flags.append({
            "type": "text_clipped",
            "severity": "critical",
            "page": page,
            "region_id": region_id,
            "evidence": evidence,
        })

    # --- text_near_edge: ink dangerously close to safe_bbox border (<4 px) ---
    near_edge = (
        (ix1 - sx1) < 4
        or (sy2 - iy2) < 4
        or (ix2 - sx1) < 4  # left proximity of right edge
        or (sx2 - ix2) < 4
    )
    if near_edge and not any(f["type"] == "text_clipped" for f in flags):
        flags.append({
            "type": "text_near_edge",
            "severity": "warning",
            "page": page,
            "region_id": region_id,
            "evidence": {
                "ink_bbox": ink_bbox,
                "safe_bbox": safe_bbox,
                "margins": {
                    "left": ix1 - sx1,
                    "right": sx2 - ix2,
                    "top": iy1 - sy1,
                    "bottom": sy2 - iy2,
                },
            },
        })

    # --- layout_bbox_too_small: safe_bbox is <65% of balloon area ---
    if balloon_bbox:
        bx1, by1, bx2, by2 = balloon_bbox
        balloon_area = max(1, (bx2 - bx1) * (by2 - by1))
        safe_area = (sx2 - sx1) * (sy2 - sy1)
        if safe_area < 0.65 * balloon_area:
            flags.append({
                "type": "layout_bbox_too_small",
                "severity": "warning",
                "page": page,
                "region_id": region_id,
                "evidence": {
                    "safe_bbox": safe_bbox,
                    "balloon_bbox": balloon_bbox,
                    "safe_area_pct": round(safe_area / balloon_area * 100, 1),
                },
            })

    return {"ok": len(flags) == 0, "flags": flags}


def blocks_clean_export(fit_result: dict) -> bool:
    """Return True if any critical flag should block a clean export."""
    return any(f["severity"] == "critical" for f in fit_result.get("flags", []))
