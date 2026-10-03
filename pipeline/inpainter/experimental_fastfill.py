"""Opt-in solid background engine for a verified owner glyph mask.

The caller's owner executor still validates identity and clamps every changed
pixel to its action mask.  Uncertain backgrounds use the ordinary inpainter.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def enabled() -> bool:
    return os.getenv("TRADUZAI_EXPERIMENTAL_OWNER_FASTFILL", "0").strip().lower() in {"1", "true", "yes", "on"}


class SolidGlyphFill:
    engine_name = "experimental_owner_solid_glyph_fill"

    def __init__(self, rgb: tuple[int, int, int]):
        self.rgb = rgb

    def inpaint(self, image: np.ndarray, mask: np.ndarray, **_kwargs) -> np.ndarray:
        result = np.asarray(image).copy()
        result[np.asarray(mask) > 0] = self.rgb
        return result


def select(image: np.ndarray, plan, record: dict) -> tuple[SolidGlyphFill | None, str]:
    """Select only well sampled uniform white/dark/colored interiors."""
    if not enabled():
        return None, "disabled"
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        return None, "invalid_image"
    if getattr(plan, "component_geometry_verified", False) is not True:
        return None, "unverified_geometry"
    action = np.asarray(getattr(plan, "action_mask", None))
    protected = np.asarray(getattr(plan, "protected_art_mask", None))
    if action.shape != image.shape[:2] or protected.shape != image.shape[:2]:
        return None, "mask_shape_mismatch"
    forbidden = ("translucent", "transparent", "textured", "gradient", "art_background", "sfx")
    profiles = " ".join(str(record.get(key) or "").lower() for key in
                        ("layout_profile", "block_profile", "background_type", "balloon_type", "content_class"))
    if any(word in profiles for word in forbidden):
        return None, "non_uniform_or_art_profile"
    mask = (action > 0) & (protected == 0)
    if int(mask.sum()) < 24:
        return None, "small_action_mask"
    bbox = getattr(plan, "owner_bbox_page", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None, "missing_owner_bbox"
    x1,y1,x2,y2 = map(int,bbox)
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        return None, "invalid_owner_bbox"
    owner = np.zeros(image.shape[:2],bool)
    owner[y1:y2,x1:x2] = True
    if np.any(mask & ~owner):
        return None, "mask_outside_owner"
    action_u8 = mask.astype(np.uint8)
    outer = cv2.dilate(action_u8,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))) > 0
    inner = cv2.dilate(action_u8,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))) > 0
    sample_mask = outer & ~inner & owner & (protected == 0)
    if int(sample_mask.sum()) < 64:
        return None, "insufficient_clean_sample"
    sample = image[sample_mask].astype(np.float32)
    median = np.median(sample,axis=0)
    delta = np.max(np.abs(sample-median),axis=1)
    if float(np.percentile(delta,95)) > 10.0 or float(np.max(np.std(sample,axis=0))) > 7.0:
        return None, "non_uniform_sample"
    color = tuple(int(round(float(v))) for v in median)
    glyph_delta = np.max(np.abs(image[mask].astype(np.float32)-median),axis=1)
    if float(np.mean(glyph_delta >= 25.0)) < 0.02:
        return None, "no_glyph_contrast"
    return SolidGlyphFill(color), "uniform_sampled_interior"
