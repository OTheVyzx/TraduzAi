"""Bounded source-centered layout search for a new Consumer Fast focal recipe."""

from __future__ import annotations

import math
import time
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
from typesetter.stable_baseline import rasterize, line_advance

from consumer_focal_text import compact_safe_advance, render_reviewed_text, wrap_lines
from consumer_visual_comfort import prepare_source_white_comfort


POLICY = "source_center_comfort_search_v2"


class LocalRasterMetricCache:
    """Bounded, call-local glyph masks for repeated line-width comparisons."""

    def __init__(self, font: Path, *, enabled: bool = True,
                 max_entries: int = 256, max_bytes: int = 4 * 1024 * 1024):
        self.font = Path(font)
        self.font_sha256 = (hashlib.sha256(self.font.read_bytes()).hexdigest()
                            if self.font.is_file() else "unavailable_font_fixture")
        rasterizer_file = Path(rasterize.__code__.co_filename)
        self.rasterizer_sha256 = (hashlib.sha256(rasterizer_file.read_bytes()).hexdigest()
                                  if rasterizer_file.is_file() else "unavailable_rasterizer_fixture")
        self.enabled = enabled
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.bytes_used = 0
        self.calls = 0
        self.hits = 0
        self.entries: OrderedDict[tuple[str, int, str, str], np.ndarray] = OrderedDict()

    def mask(self, size: int, value: str) -> np.ndarray:
        key = (self.font_sha256, size, value, self.rasterizer_sha256)
        if self.enabled and key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.calls += 1
        mask = rasterize(str(self.font), size, value)
        if self.enabled and mask.nbytes <= self.max_bytes:
            while self.entries and (len(self.entries) >= self.max_entries or
                                    self.bytes_used + mask.nbytes > self.max_bytes):
                _, removed = self.entries.popitem(last=False)
                self.bytes_used -= removed.nbytes
            self.entries[key] = mask
            self.bytes_used += mask.nbytes
        return mask

    def evidence(self) -> dict:
        return dict(schema="local_raster_metric_cache_v1", enabled=self.enabled,
                    font_sha256=self.font_sha256,
                    rasterizer_sha256=self.rasterizer_sha256,
                    unique_raster_calls=self.calls, hits=self.hits,
                    entries=len(self.entries), resident_bytes=self.bytes_used,
                    max_entries=self.max_entries, max_bytes=self.max_bytes)


def alternative_word_breaks(text: str, font: Path, size: int, width: int,
                            *, limit: int = 4,
                            raster_widths: dict[str, int] | None = None,
                            metric_cache: LocalRasterMetricCache | None = None) -> list[list[str]]:
    """Bounded beam of complete word-boundary layouts; never split or omit words."""
    words = text.split()
    if not words or not 1 <= limit <= 8:
        raise ValueError("invalid alternative line search")
    widths: dict[str, int] = raster_widths if raster_widths is not None else {}
    def ink_width(value: str) -> int:
        if metric_cache is not None:
            return metric_cache.mask(size, value).shape[1]
        if value not in widths:
            widths[value] = rasterize(str(font), size, value).shape[1]
        return widths[value]
    # Beam entries hold a cost and an immutable line tuple. Penalize short
    # function-word lines; the real contour check still decides safety.
    beam: dict[int, list[tuple[float, tuple[str, ...]]]] = {0: [(0.0, ())]}
    for start in range(len(words)):
        if start not in beam:
            continue
        for prior_cost, prior in beam[start]:
            for end in range(start + 1, len(words) + 1):
                line = " ".join(words[start:end])
                measured = ink_width(line)
                if measured > width:
                    break
                singleton = len(line.split()) == 1 and line.strip(".,!?;:").casefold() in {
                    "a", "o", "as", "os", "de", "do", "da", "em", "e", "um", "uma",
                    "para", "por", "que", "não", "the", "of", "to", "and"}
                if singleton:
                    continue
                cost = prior_cost + ((width - measured) / width) ** 2
                bucket = beam.setdefault(end, [])
                bucket.append((cost, (*prior, line)))
                bucket.sort(key=lambda item: (item[0], len(item[1]), item[1]))
                del bucket[limit:]
    return [list(lines) for _, lines in beam.get(len(words), [])[:limit]]


def contour_word_breaks(text: str, font: Path, size: int, width: int,
                        prepared: dict, anchor_bbox: list[int], safe_bbox: list[int],
                        *, limit: int = 3,
                        metric_cache: LocalRasterMetricCache | None = None) -> list[list[str]]:
    """Propose complete lines against approximate horizontal room at each row.

    This only orders proposals. The actual raster and contour gate remain final.
    """
    words = text.split()
    if not words:
        return []
    cache: dict[str, int] = {}
    def measure(value: str) -> int:
        if metric_cache is not None:
            return metric_cache.mask(size, value).shape[1]
        if value not in cache:
            cache[value] = rasterize(str(font), size, value).shape[1]
        return cache[value]
    from consumer_focal_text import compact_safe_advance
    visual = _visual_core_height(font, size, "HATO")
    advance, _ = compact_safe_advance(font, size, ["HATO", "HATO"])
    rx1, ry1, rx2, ry2 = prepared["source_roi_bbox"]
    distance = prepared["distance"]
    center_x = round((anchor_bbox[0]+anchor_bbox[2])/2)
    center_y = (anchor_bbox[1]+anchor_bbox[3])/2
    result: list[tuple[float, list[str]]] = []
    for count in range(3, min(len(words), 9)+1):
        top = center_y - (advance*(count-1)+visual)/2
        if top < safe_bbox[1]+4 or top+advance*(count-1)+visual > safe_bbox[3]-4:
            continue
        caps = []
        for index in range(count):
            middle = top + index*advance + visual/2
            rows = [round(middle+offset) for offset in (-visual/4, 0, visual/4)]
            if any(row < ry1 or row >= ry2 for row in rows):
                break
            lo, hi = max(rx1, safe_bbox[0]+4), min(rx2, safe_bbox[2]-4)
            valid = np.ones(hi-lo, dtype=bool)
            for row in rows:
                valid &= distance[row-ry1, lo-rx1:hi-rx1] >= prepared["minimum_px"]
            center = center_x-lo
            if center < 0 or center >= len(valid) or not valid[center]:
                break
            left = center
            right = center
            while left > 0 and valid[left-1]:
                left -= 1
            while right+1 < len(valid) and valid[right+1]:
                right += 1
            caps.append(min(width, 2*min(center-left, right-center)+1))
        if len(caps) != count:
            continue
        beam: dict[tuple[int, int], list[tuple[float, tuple[str, ...]]]] = {(0, 0): [(0.0, ())]}
        for line_index in range(count):
            for start in range(len(words)):
                for prior_cost, prior in beam.get((line_index, start), []):
                    for end in range(start+1, len(words)+1):
                        line = " ".join(words[start:end])
                        ink_width = measure(line)
                        if ink_width > caps[line_index]:
                            break
                        singleton = len(line.split()) == 1 and line.strip(".,!?;:").casefold() in {
                            "a", "o", "as", "os", "de", "do", "da", "em", "e", "um", "uma",
                            "para", "por", "que", "não"}
                        if singleton:
                            continue
                        cost = prior_cost + ((caps[line_index]-ink_width)/max(1, caps[line_index]))**2
                        bucket = beam.setdefault((line_index+1, end), [])
                        bucket.append((cost, (*prior, line)))
                        bucket.sort(key=lambda item: (item[0], item[1]))
                        del bucket[limit:]
        result.extend((cost, list(lines)) for cost, lines in beam.get((count, len(words)), []))
    result.sort(key=lambda item: (item[0], len(item[1]), item[1]))
    return [lines for _, lines in result[:limit]]


def source_line_ink_height(original: np.ndarray, prepared: dict,
                           observation_bboxes: list[list[int]]) -> tuple[float, dict]:
    """Measure the actual dark source lettering core in authenticated line boxes."""
    if original.ndim != 3 or original.shape[2] != 3 or not observation_bboxes:
        raise ValueError("source line evidence missing")
    rx1, ry1, rx2, ry2 = prepared["source_roi_bbox"]
    distance = prepared["distance"]
    heights = []
    counts = []
    for x1, y1, x2, y2 in observation_bboxes:
        if not (rx1 <= x1 < x2 <= rx2 and ry1 <= y1 < y2 <= ry2):
            raise ValueError("source line outside authenticated contour crop")
        patch = original[y1:y2, x1:x2]
        near = distance[y1-ry1:y2-ry1, x1-rx1:x2-rx1]
        chroma = patch.max(2).astype(np.int16)-patch.min(2).astype(np.int16)
        ink = (patch.max(2) < 180) & (chroma <= 10) & (near >= 3)
        rows = np.flatnonzero(np.any(ink, axis=1))
        if len(rows) and int(np.count_nonzero(ink)) >= 12:
            heights.append(int(rows[-1]-rows[0]+1))
            counts.append(int(np.count_nonzero(ink)))
    if len(heights) < max(1, len(observation_bboxes)//2):
        raise ValueError("insufficient source glyph cores")
    height = float(np.median(heights))
    return height, dict(schema="source_line_ink_height_v1", height_px=height,
                        line_heights_px=heights, line_ink_pixels=counts,
                        observations_used=len(heights),
                        observations_total=len(observation_bboxes),
                        confidence="source_dark_ink_contour_gated")


def _visual_core_height(font_path: Path, size: int, sample: str) -> int:
    mask = rasterize(str(font_path), size, sample)
    rows = np.flatnonzero(np.any(mask > 0, axis=1))
    if not len(rows):
        raise ValueError("font sample has no ink")
    return int(rows[-1]-rows[0]+1)


def preferred_size(font_path: Path, source_visual_height_px: float,
                   sample: str = "HATO", *, minimum_size: int = 18,
                   maximum_size: int = 56) -> tuple[int, dict]:
    """Match observed source body height to actual destination-font ink."""
    if source_visual_height_px <= 0:
        raise ValueError("source visual height missing")
    candidates = [(size, _visual_core_height(font_path, size, sample))
                  for size in range(minimum_size, maximum_size+1)]
    selected = min(candidates, key=lambda row: (abs(row[1]-source_visual_height_px), row[0]))
    return selected[0], dict(source_visual_height_px=source_visual_height_px,
                             font_sample=sample, target_core_height_px=selected[1],
                             evidence="selected_source_observation_core_height")


def search_source_centered_layout(clean: np.ndarray, base_recipe: dict,
                                  *, source_visual_height_px: float,
                                  preferred_font_size: int, minimum_font_size: int,
                                  maximum_font_size: int,
                                  width_ratios: tuple[float, ...] =
                                  (1.0, .92, .84, .76, .68, .60),
                                  maximum_candidates: int = 120,
                                  font_sample: str = "HATO",
                                  alternative_breaks: bool = False,
                                  maximum_seconds: float | None = None,
                                  use_metric_cache: bool = True) -> dict:
    """Try reflow at a fixed size before shrinking; return real raster evidence."""
    if not 1 <= minimum_font_size <= preferred_font_size <= maximum_font_size <= 100:
        raise ValueError("invalid visual scale range")
    if not 1 <= maximum_candidates <= 300 or not width_ratios or any(
            not .35 <= ratio <= 1.0 for ratio in width_ratios):
        raise ValueError("invalid bounded layout search")
    if maximum_seconds is not None and not 1 <= maximum_seconds <= 300:
        raise ValueError("invalid layout time limit")
    started = time.monotonic()
    if base_recipe.get("visual_comfort_policy") is None:
        raise ValueError("source contour policy required")
    font = Path(base_recipe["font_path"])
    metrics = LocalRasterMetricCache(font, enabled=use_metric_cache)
    prepared = prepare_source_white_comfort(base_recipe["visual_comfort_policy"],
                                            base_recipe["source_sha256"])
    max_width = base_recipe["safe_bbox"][2]-base_recipe["safe_bbox"][0]-12
    widths = sorted({max(8, int(round(max_width*ratio))) for ratio in width_ratios}, reverse=True)
    attempts = []
    accepted = []
    # Prefer breathing room within a bounded, source-relative readable range.
    # Once this distance is reached, extra empty space earns no reward.
    desired_comfort = max(float(prepared["ideal_px"]), .50 * source_visual_height_px)
    readable_minimum = max(minimum_font_size, math.ceil(.78 * preferred_font_size))
    # Reserve capacity for the full declared scale range. A budget exhausted
    # before reaching smaller candidates would report a false no-solution.
    extra = list(range(preferred_font_size+1, maximum_font_size+1))
    span = preferred_font_size-readable_minimum+1
    size_budget = math.ceil(maximum_candidates/(len(widths)*(5 if alternative_breaks else 1)))
    if span > 1 and size_budget < 2:
        raise ValueError("candidate budget cannot cover declared scale range")
    stride = max(1, math.ceil((span-1)/max(1, size_budget-1)))
    sizes = list(range(preferred_font_size, readable_minimum-1, -stride))
    if sizes[-1] != readable_minimum:
        sizes.append(readable_minimum)
    sizes += extra[:max(0, size_budget-len(sizes))]
    for size in sizes:
        if maximum_seconds is not None and time.monotonic()-started >= maximum_seconds:
            break
        raster_widths: dict[str, int] = {}
        visual = _visual_core_height(font, size, font_sample)
        visual_ratio = visual/source_visual_height_px
        if not .78 <= visual_ratio <= 1.20:
            continue
        for width in widths:
            if len(attempts) >= maximum_candidates or (maximum_seconds is not None and
                    time.monotonic()-started >= maximum_seconds):
                break
            try:
                greedy = wrap_lines(base_recipe["target"], font, size, width)
                plans = [greedy]
                if alternative_breaks:
                    plans += [lines for lines in contour_word_breaks(
                        base_recipe["target"], font, size, width, prepared,
                        base_recipe["anchor_bbox"], base_recipe["safe_bbox"], limit=3,
                        metric_cache=metrics)
                        if lines not in plans]
                    plans += [lines for lines in alternative_word_breaks(
                        base_recipe["target"], font, size, width, limit=1,
                        raster_widths=raster_widths, metric_cache=metrics) if lines not in plans]
            except ValueError as error:
                attempts.append({"font_size": size, "max_line_width_px": width,
                                 "status": "rejected", "reason": str(error)})
                continue
            for lines in plans:
                if len(attempts) >= maximum_candidates or (maximum_seconds is not None and
                        time.monotonic()-started >= maximum_seconds):
                    break
                entry = {"font_size": size, "max_line_width_px": width,
                         "target_visual_core_px": visual, "source_scale_ratio": visual_ratio,
                         "lines": lines, "break_mode": "greedy" if lines == greedy else "explicit"}
                try:
                    if base_recipe.get("line_spacing_profile") == "compact_safe_leading_v1":
                        advance, _ = compact_safe_advance(font, size, lines)
                    else:
                        advance = line_advance(str(font), size, "standard")
                    recipe = dict(base_recipe, font_size=size,
                                  minimum_font_size=minimum_font_size,
                                  max_line_width_px=width, line_advance=advance)
                    if lines != greedy:
                        recipe["line_plan"] = {"schema": "explicit_word_breaks_v1", "lines": lines}
                    layer, evidence = render_reviewed_text(clean, recipe,
                                                           prepared_comfort=prepared)
                    comfort = evidence["visual_comfort"]
                    line_widths = [metrics.mask(size, line).shape[1] for line in lines]
                    balance = float(np.std(line_widths)/max(line_widths)) if len(lines) > 1 else 0.0
                    shortfall = max(0.0, desired_comfort-comfort["minimum_actual_px"])
                    score = (8*shortfall +
                             1.5*(preferred_font_size-size) +
                             2*balance + .03*len(lines))
                    entry.update(status="valid", score=score, lines=lines,
                                 line_advance=advance,
                                 minimum_contour_distance_px=comfort["minimum_actual_px"],
                                 desired_contour_distance_px=desired_comfort,
                                 comfort_shortfall_px=shortfall,
                                 center_error_px=evidence["center_error_px"])
                    accepted.append((score, recipe, layer, evidence, entry))
                except (ValueError, IndexError) as error:
                    entry.update(status="rejected", reason=str(error))
                attempts.append(entry)
        if len(attempts) >= maximum_candidates:
            break
    if not accepted:
        return {"status": "review_required", "policy": POLICY,
                "reason": ("search_time_budget_exhausted" if maximum_seconds is not None and
                           time.monotonic()-started >= maximum_seconds else
                           "no_candidate_satisfies_source_contour_scale_and_center"),
                "attempts": attempts, "candidate_count": len(attempts),
                "metric_cache": metrics.evidence()}
    selected = min(accepted, key=lambda item: (item[0], -item[1]["font_size"],
                                                   -item[1]["max_line_width_px"]))
    score, recipe, layer, evidence, choice = selected
    # Bind the selected decision; the normal loader must rasterize this recipe again.
    recipe["layout_search"] = {"schema": POLICY,
                               "source_visual_height_px": source_visual_height_px,
                               "preferred_font_size": preferred_font_size,
                               "minimum_font_size": minimum_font_size,
                               "readable_minimum_font_size": readable_minimum,
                               "maximum_font_size": maximum_font_size,
                               "desired_contour_distance_px": desired_comfort,
                               "comfort_target_reached": choice["comfort_shortfall_px"] == 0,
                               "chosen_score": score,
                               "chosen_lines": choice["lines"],
                               "alternative_breaks": alternative_breaks,
                               "time_budget_exhausted": (maximum_seconds is not None and
                                                         time.monotonic()-started >= maximum_seconds),
                               "candidate_count": len(attempts),
                               "attempts": attempts}
    return {"status": "rendered_candidate", "policy": POLICY,
            "recipe": recipe, "layer": layer, "evidence": evidence,
            "attempts": attempts, "candidate_count": len(attempts),
            "metric_cache": metrics.evidence()}
