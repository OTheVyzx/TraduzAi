"""Bounded, reviewed white-field text recipe for the normal Consumer Fast loader.

The caller must supply a source-linked visual adjudication. This module does not
infer narration from the absence of a balloon, translate text, or erase art.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from typesetter.stable_baseline import rasterize, line_advance

from consumer_fast_core import RasterLayer


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def group_line_boxes(lines: list[tuple[str, list[int]]]) -> list[list[str]]:
    """Propose typographic line groups; semantic/container review remains required."""
    ordered = sorted(lines, key=lambda row: (row[1][1], row[1][0]))
    groups: list[list[str]] = []
    previous = None
    for line_id, box in ordered:
        if previous is None:
            groups.append([line_id])
        else:
            old = previous
            gap = box[1] - old[3]
            height = max(old[3] - old[1], box[3] - box[1])
            center_delta = abs((old[0] + old[2]) - (box[0] + box[2])) / 2
            width = max(old[2] - old[0], box[2] - box[0])
            if 0 <= gap <= 0.65 * height and center_delta <= 0.2 * width:
                groups[-1].append(line_id)
            else:
                groups.append([line_id])
        previous = box
    return groups


def _line_vertical_extent(font_path: Path, size: int, text: str) -> tuple[int, int]:
    """Measure the actual Unicode ink, including marks and descenders."""
    mask = rasterize(str(font_path), size, text)
    rows = np.flatnonzero(np.any(mask > 0, axis=1))
    if not len(rows):
        raise ValueError('line produced no ink')
    return int(rows[0]), int(rows[-1])


def line_ink_gaps(font_path: Path, size: int, lines: list[str], advance: int) -> list[int]:
    """Actual empty pixel rows between adjacent ink envelopes."""
    extents = [_line_vertical_extent(font_path, size, line) for line in lines]
    return [advance + next_top - prior_bottom - 1
            for (_, prior_bottom), (next_top, _) in zip(extents, extents[1:])]


def compact_safe_advance(font_path: Path, size: int, lines: list[str]) -> tuple[int, dict]:
    """Opt-in v1 leading: 1.10 target, increased uniformly for actual ink."""
    if size <= 0 or not lines:
        raise ValueError('invalid compact leading inputs')
    minimum_gap = max(2, int(math.ceil(size * 0.07)))
    preferred = int(math.floor(size * 1.10 + 0.5))
    extents = [_line_vertical_extent(font_path, size, line) for line in lines]
    required = max((bottom - next_top + 1 + minimum_gap
                    for (_, bottom), (next_top, _) in zip(extents, extents[1:])),
                   default=0)
    advance = max(preferred, required)
    gaps = line_ink_gaps(font_path, size, lines, advance)
    return advance, dict(schema='compact_safe_leading_v1',
                         target_ratio=1.10, preferred_advance_px=preferred,
                         line_advance_px=advance,
                         minimum_required_gap_px=minimum_gap,
                         minimum_actual_gap_px=min(gaps) if gaps else None,
                         adjacent_ink_gaps_px=gaps)


def wrap_lines(text: str, font_path: Path, size: int, width: int) -> list[str]:
    words = text.split()
    if not words:
        raise ValueError('empty target')
    lines: list[str] = []
    line = ''
    for word in words:
        candidate = (line + ' ' + word).strip()
        if rasterize(str(font_path), size, candidate).shape[1] > width:
            if not line:
                raise ValueError('word exceeds local safe width')
            lines.append(line)
            if rasterize(str(font_path), size, word).shape[1] > width:
                raise ValueError('word exceeds local safe width')
            line = word
        else:
            line = candidate
    lines.append(line)
    return lines


def _ink(text: str, font_path: Path, size: int, width: int, leading: int,
         line_plan: dict | None = None):
    if line_plan is None:
        lines = wrap_lines(text, font_path, size, width)
    else:
        if line_plan.get('schema') != 'explicit_word_breaks_v1':
            raise ValueError('unsupported line plan')
        lines = line_plan.get('lines')
        if (not isinstance(lines, list) or not lines or
                any(not isinstance(line, str) or not line.strip() or line != line.strip()
                    for line in lines) or
                ' '.join(' '.join(lines).split()) != ' '.join(text.split())):
            raise ValueError('line plan changes target words or order')
        if any(rasterize(str(font_path), size, line).shape[1] > width for line in lines):
            raise ValueError('planned line exceeds local safe width')
    cells = [rasterize(str(font_path), size, value) for value in lines]
    cell_height = max(cell.shape[0] for cell in cells)
    canvas = np.zeros((leading * (len(lines) - 1) + cell_height, width), dtype=np.uint8)
    for index, value in enumerate(lines):
        cell = cells[index]
        x = round((width - cell.shape[1]) / 2)
        y = index * leading
        canvas[y:y + cell.shape[0], x:x + cell.shape[1]] = np.maximum(
            canvas[y:y + cell.shape[0], x:x + cell.shape[1]], cell)
    yy, xx = np.nonzero(canvas)
    if not len(xx):
        raise ValueError('target produced no ink')
    return canvas[yy.min():yy.max() + 1, xx.min():xx.max() + 1], lines


def _check_safe_interior(clean: np.ndarray, recipe: dict,
                         bbox: tuple[int, int, int, int], alpha: np.ndarray) -> dict | None:
    """Check ink against the clean page's actual white connected interior.

    This optional contract is source-bound. Its seed comes from the OCR anchor,
    while the contour is recomputed from the authenticated clean canvas on every
    rerender. A rectangle alone cannot authorize crossing a balloon boundary.
    """
    policy = recipe.get('safe_interior_policy')
    if policy is None:
        return None
    if policy.get('schema') != 'connected_white_component_v1':
        raise ValueError('unsupported safe interior policy')
    threshold = int(policy['threshold'])
    dilation = int(policy['barrier_dilation_px'])
    pad = int(policy['roi_pad_px'])
    if not (0 < threshold < 255 and 1 <= dilation <= 15 and dilation % 2 and 8 <= pad <= 256):
        raise ValueError('unsafe safe interior parameters')
    x1, y1, x2, y2 = map(int, recipe['safe_bbox'])
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(clean.shape[1], x2 + pad), min(clean.shape[0], y2 + pad)
    gray = cv2.cvtColor(clean[ry1:ry2, rx1:rx2], cv2.COLOR_RGB2GRAY)
    barrier = cv2.dilate((gray < threshold).astype(np.uint8),
                         np.ones((dilation, dilation), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats((barrier == 0).astype(np.uint8), 8)
    ax1, ay1, ax2, ay2 = map(int, recipe['anchor_bbox'])
    seed_x, seed_y = round((ax1 + ax2) / 2), round((ay1 + ay2) / 2)
    label = int(labels[seed_y - ry1, seed_x - rx1])
    if label == 0:
        raise ValueError('source anchor is on an interior barrier')
    left, top, width, height, area = map(int, stats[label])
    if left <= 0 or top <= 0 or left + width >= gray.shape[1] or top + height >= gray.shape[0]:
        raise ValueError('safe interior leaks through crop boundary')
    px, py, qx, qy = bbox
    interior = labels[py-ry1:qy-ry1, px-rx1:qx-rx1] == label
    if interior.shape != alpha.shape:
        raise ValueError('ink outside safe interior crop')
    unsafe = int(np.count_nonzero((alpha > 0) & ~interior))
    if unsafe:
        raise ValueError(f'text crosses nonrectangular safe interior: {unsafe} alpha pixels')
    return dict(schema='connected_white_component_v1',
                interior_bbox=[left+rx1, top+ry1, left+width+rx1, top+height+ry1],
                interior_area=area, unsafe_alpha_pixels=unsafe,
                threshold=threshold, barrier_dilation_px=dilation, roi_pad_px=pad)


def render_reviewed_text(clean: np.ndarray, recipe: dict,
                         *, prepared_comfort: dict | None = None) -> tuple[RasterLayer, dict]:
    if recipe.get('schema') != 'consumer_focal_text_v1':
        raise ValueError('unsupported focal recipe')
    if sha256_bytes(clean.tobytes()) != recipe['clean_pixels_sha256']:
        raise ValueError('clean canvas changed')
    decision = recipe.get('adjudication') or {}
    if decision.get('human_review') is not False:
        raise ValueError('human review state missing')
    if decision.get('observer') == 'source_geometry_policy_v1':
        if (decision.get('status') != 'translation_ready' or
                decision.get('unit_id') != recipe.get('owner_id') or
                decision.get('source_text_sha256') != sha256_bytes(recipe['source'].encode('utf-8'))):
            raise ValueError('operational geometry decision mismatch')
    elif decision.get('observer') != 'model_visual_review':
        raise ValueError('source-linked visual adjudication required')
    if recipe['adjudication'].get('source_sha256') != recipe['source_sha256']:
        raise ValueError('visual adjudication source mismatch')
    font_path = Path(recipe['font_path'])
    if sha256_bytes(font_path.read_bytes()) != recipe['font_sha256']:
        raise ValueError('font changed')
    x1, y1, x2, y2 = map(int, recipe['safe_bbox'])
    ax1, ay1, ax2, ay2 = map(int, recipe['anchor_bbox'])
    if not (0 <= x1 < ax1 < ax2 < x2 <= clean.shape[1] and
            0 <= y1 < ay1 < ay2 < y2 <= clean.shape[0]):
        raise ValueError('anchor outside local safe area')
    size = int(recipe['font_size'])
    if size < int(recipe['minimum_font_size']):
        raise ValueError('text scale below legibility gate')
    leading = int(recipe['line_advance'])
    profile = recipe.get('line_spacing_profile')
    if profile not in (None, 'compact_safe_leading_v1'):
        raise ValueError('unsupported line spacing profile')
    if profile is None and leading < line_advance(str(font_path), size, 'compact_5_unadorned'):
        raise ValueError('line spacing collision')
    line_width = x2 - x1 - 12
    if recipe.get('max_line_width_px') is not None:
        requested = int(recipe['max_line_width_px'])
        if not 8 <= requested <= line_width:
            raise ValueError('line width outside local safe area')
        line_width = requested
    mask, lines = _ink(recipe['target'], font_path, size, line_width, leading,
                       recipe.get('line_plan'))
    compact_evidence = None
    if profile == 'compact_safe_leading_v1':
        required, compact_evidence = compact_safe_advance(font_path, size, lines)
        if leading != required:
            raise ValueError('compact line spacing differs from measured safe advance')
    height, width = mask.shape
    if width > x2 - x1 - 8 or height > y2 - y1 - 8:
        raise ValueError('text does not fit local safe area')
    center_x = (ax1 + ax2) / 2
    center_y = (ay1 + ay2) / 2
    px = round(center_x - width / 2)
    py = round(center_y - height / 2)
    if px < x1 + 4 or py < y1 + 4 or px + width > x2 - 4 or py + height > y2 - 4:
        raise ValueError('centered text crosses local safe margin')
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = np.array(recipe.get('rgb', [0, 0, 0]), dtype=np.uint8)
    rgba[..., 3] = mask
    error = [(px + width / 2) - center_x, (py + height / 2) - center_y]
    if any(abs(v) > 1 for v in error):
        raise ValueError('source center changed')
    layer = RasterLayer(recipe['page_id'], recipe['owner_id'], (px, py, px + width, py + height), rgba)
    interior = _check_safe_interior(clean, recipe, layer.bbox, mask)
    comfort = None
    if recipe.get('visual_comfort_policy') is not None:
        from consumer_visual_comfort import (measure_prepared_source_white_comfort,
                                             measure_source_white_comfort)
        if prepared_comfort is not None and prepared_comfort['source_sha256'] != recipe['source_sha256']:
            raise ValueError('prepared source comfort hash mismatch')
        comfort = (measure_prepared_source_white_comfort(prepared_comfort, layer.bbox, mask)
                   if prepared_comfort is not None else
                   measure_source_white_comfort(recipe['visual_comfort_policy'],
                                               recipe['source_sha256'], layer.bbox, mask))
        if not comfort['safety_pass']:
            raise ValueError('rendered ink crosses authenticated source contour')
        if not comfort['comfort_minimum_pass']:
            raise ValueError('rendered ink below source-contour comfort minimum')
    value = {'status': 'rendered', 'fit_status': 'ok', 'lines': lines,
                   'font_size': size, 'line_advance': leading,
                   'ink_bbox': list(layer.bbox), 'ink_height': height,
                   'center_error_px': error, 'safe_bbox': list(recipe['safe_bbox'])}
    if interior is not None:
        value['safe_interior'] = interior
        value['max_line_width_px'] = line_width
    if compact_evidence is not None:
        value['line_spacing'] = compact_evidence
    if comfort is not None:
        value['visual_comfort'] = comfort
    return layer, value
