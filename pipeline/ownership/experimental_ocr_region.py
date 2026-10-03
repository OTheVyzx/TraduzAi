"""Opt-in virtual layout region from selected OCR; never a cleanup mask."""
from __future__ import annotations

import math
import os

import cv2
import numpy as np


def enabled() -> bool:
    return os.getenv("TRADUZAI_EXPERIMENTAL_OCR_REGION", "0").strip().lower() in {"1", "true", "yes", "on"}


def propose(source_bbox, *, width: int, height: int, protected_art_mask,
            foreign_boxes=()):
    """Return the smallest rounded OCR reserve supported by the art guard."""
    if not enabled():
        return None, "disabled"
    if protected_art_mask is None:
        return None, "missing_protected_art_evidence"
    protected = np.asarray(protected_art_mask)
    if protected.shape != (height,width) or protected.dtype != np.uint8:
        return None, "invalid_protected_art_evidence"
    try:
        sx1,sy1,sx2,sy2 = map(int,source_bbox)
    except (TypeError,ValueError):
        return None, "invalid_ocr_bbox"
    if not (0 <= sx1 < sx2 <= width and 0 <= sy1 < sy2 <= height):
        return None, "invalid_ocr_bbox"
    if (sx2-sx1)*(sy2-sy1) > 0.30*width*height:
        return None, "oversized_ocr_region"
    margin = max(3,min(18,math.ceil(0.30*min(sx2-sx1,sy2-sy1))))
    x1,y1,x2,y2 = sx1-margin,sy1-margin,sx2+margin,sy2+margin
    if not (0 < x1 < x2 < width and 0 < y1 < y2 < height):
        return None, "reserve_reaches_page_edge"
    for foreign in foreign_boxes:
        fx1,fy1,fx2,fy2 = map(int,foreign)
        if min(x2,fx2)>max(x1,fx1) and min(y2,fy2)>max(y1,fy1):
            return None, "foreign_owner_intersection"
    radius = max(2,min(margin,(x2-x1)//6,(y2-y1)//6))
    polygon = ((x1+radius,y1),(x2-radius,y1),(x2,y1+radius),(x2,y2-radius),
               (x2-radius,y2),(x1+radius,y2),(x1,y2-radius),(x1,y1+radius))
    region = np.zeros((height,width),np.uint8)
    cv2.fillPoly(region,[np.asarray(polygon,np.int32)],1)
    # The OCR interior may contain ink, but layout reserve must not overlap
    # independently protected art. Cleanup remains limited to owner glyphs.
    reserve = region.astype(bool)
    reserve[sy1:sy2,sx1:sx2] = False
    if np.any(reserve & (protected>0)):
        return None, "protected_art_in_reserve"
    return ((x1,y1,x2,y2),polygon), "experimental_virtual_ocr_region"
