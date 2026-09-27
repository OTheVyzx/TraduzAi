"""Local two-member speech composition consumed by Consumer Fast recipes.

The caller supplies original chapter order, source identities and a validated
translation. This module has no chapter, page number, OCR phrase or target text
policy. It renders one logical block on a bounded seam canvas and splits only
the resulting physical raster layer.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from consumer_fast_core import RasterLayer
from typesetter.renderer import render_text_block


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_adjacent(chapter_members: list[str], first: str, second: str) -> None:
    """Adjacency comes from the complete source manifest, not selected pages."""
    if chapter_members.count(first) != 1 or chapter_members.count(second) != 1:
        raise ValueError("continuity source member absent or ambiguous")
    if chapter_members.index(second) != chapter_members.index(first) + 1:
        raise ValueError("continuity members are not adjacent in source chapter")


def seam_geometry(first_box, second_box, first_height: int, window: int):
    a = [first_box[0], first_box[1] - first_height + window,
         first_box[2], first_box[3] - first_height + window]
    b = [second_box[0], second_box[1] + window,
         second_box[2], second_box[3] + window]
    if not (0 <= a[1] < window <= b[1] < 2 * window):
        raise ValueError("selected OCR does not straddle seam")
    return [min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3])]


def balloon_region(original: np.ndarray, anchor: list[int], window: int):
    whiteness = (np.min(original, axis=2) >= 245).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(whiteness, 8)
    sx = round((anchor[0] + anchor[2]) / 2)
    sy = max(0, anchor[1] - 30)
    label = int(labels[sy, sx])
    if not 0 < label < count:
        raise ValueError("connected balloon interior absent")
    mask = (labels == label).astype(np.uint8)
    x, y, width, height, area = map(int, stats[label])
    if area < 1000 or not (y < window < y + height):
        raise ValueError("balloon does not cross source seam")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    safe = cv2.erode(filled, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(safe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 2.0, True).reshape(-1, 2).tolist()
    ix, iy = max(8, round(width * .08)), max(8, round(height * .08))
    return safe.astype(bool), polygon, [x + ix, y + iy, x + width - ix, y + height - iy]


def build_block(style_recipe: dict, *, owner_id: str, source: str, target: str,
                anchor: list[int], polygon: list[list[int]], safe_box: list[int]):
    block = copy.deepcopy(style_recipe["layout"]["texts"][0])
    style = copy.deepcopy(block["visual_profile_v2"]["applied_style"])
    for stale in ("visual_profile_v2", "visual_profile_sha256", "style_resolved_intent_v1",
                  "style_group_resolution_v3", "owner_text_execution_authority",
                  "text_execution_authority_sha256", "owner_render_geometry_sha256",
                  "owner_style_capture"):
        block.pop(stale, None)
    block.update(id=owner_id, owner_id=owner_id, page_id="continuity",
                 original=source, source_payload=source, translated=target,
                 translated_payload=target, source_text_anchor_bbox=anchor,
                 _source_text_anchor_bbox=anchor, safe_text_box=safe_box,
                 layout_safe_bbox=safe_box, layout_bbox=safe_box,
                 balloon_bbox=safe_box, bubble_inner_bbox=safe_box,
                 render_safe_polygon_page=polygon, paint_safe_polygon_page=polygon,
                 layout_regions=[], layout_region_ids=[], balloon_subregions=[],
                 connected_lobe_bboxes=[], _owner_mode=True,
                 _owner_render_mode=True, _owner_layout_verified=True,
                 estilo=style, style=style)
    return block


def render_local(local_clean: np.ndarray, local_original: np.ndarray, block: dict):
    block = copy.deepcopy(block)
    canvas = Image.fromarray(local_clean)
    render_text_block(canvas, block, pre_render_np=local_original)
    alignment = block.get("source_center_alignment") or {}
    error = alignment.get("center_error_px") or ()
    if block.get("fit_status") != "ok" or block.get("render_completed") is not True:
        raise ValueError("connected utterance did not render safely")
    if len(error) != 2 or any(abs(float(x)) > 1 for x in error):
        raise ValueError("connected utterance changed source center")
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8).copy(), block


def fragment_layer(clean: np.ndarray, recipe: dict):
    """Render once in seam coordinates, then retain only this physical half."""
    if hashlib.sha256(clean.tobytes()).hexdigest() != recipe["clean_pixels_sha256"]:
        raise ValueError("connected fragment clean base changed")
    rendered, block = render_local(recipe["local_clean"], recipe["local_original"], recipe["block"])
    safe = recipe["safe_mask"]
    window = recipe["window"]
    half = recipe["half"]
    y0 = 0 if half == "first" else window
    mask = safe[y0:y0 + window]
    pixels = rendered[y0:y0 + window]
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("connected fragment has no pixels")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    ly1, ly2 = int(ys.min()), int(ys.max()) + 1
    rgba = np.zeros((ly2 - ly1, x2 - x1, 4), dtype=np.uint8)
    rgba[..., :3] = pixels[ly1:ly2, x1:x2]
    rgba[..., 3] = mask[ly1:ly2, x1:x2].astype(np.uint8) * 255
    page_y = clean.shape[0] - window if half == "first" else 0
    layer = RasterLayer(recipe["page_id"], recipe["owner_id"],
                        (x1, ly1 + page_y, x2, ly2 + page_y), rgba)
    return layer, block
