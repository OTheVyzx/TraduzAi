"""Entry-point do pipeline strip-based.

Chamado por `pipeline/main.py::_run_pipeline` após a Fase 6 do switchover.
"""

from __future__ import annotations

import copy
import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import threading
import time

import cv2
import numpy as np

from strip._diagnostics import dump_strip_debug, is_debug_enabled
from strip.bands import attach_band_slices, group_balloons_into_bands
from strip.concat import build_strip
from strip.detect_balloons import _inner_dark_text_evidence, detect_strip_balloons
from strip.process_bands import _band_id_for, _page_id_for, process_band
from strip.reassemble import assemble_output_pages
from strip.types import Band, Balloon, BBox, OutputPage, VerticalStrip


_LEGACY_DECISION_FIELDS = frozenset(
    {
        "skip_processing",
        "skip_reason",
        "preserve_original",
        "tipo",
        "content_class",
        "balloon_type",
    }
)


def _without_legacy_decision_fields(record: dict) -> dict:
    cleaned = copy.deepcopy(record)
    for key in _LEGACY_DECISION_FIELDS:
        cleaned.pop(key, None)
    return cleaned


def _texts_without_legacy_decision_fields(texts) -> list[dict]:
    return [
        _without_legacy_decision_fields(text)
        for text in list(texts or [])
        if isinstance(text, dict)
    ]


def _legacy_compat_key(record: dict, index: int) -> tuple[str, str | int]:
    for key in ("trace_id", "text_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return key, str(value)
    for key in ("bbox", "source_bbox", "text_pixel_bbox"):
        value = record.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                return key, ",".join(str(int(round(float(v)))) for v in value[:4])
            except Exception:
                continue
    return "index", int(index)


def _legacy_decision_fields_by_record(records) -> dict[tuple[str, str | int], dict]:
    payload: dict[tuple[str, str | int], dict] = {}
    for index, record in enumerate(list(records or [])):
        if not isinstance(record, dict):
            continue
        fields = {
            key: copy.deepcopy(record[key])
            for key in _LEGACY_DECISION_FIELDS
            if record.get(key) not in (None, "")
        }
        if fields:
            payload[_legacy_compat_key(record, index)] = fields
    return payload


def _restore_legacy_decision_fields(records, payload: dict[tuple[str, str | int], dict]) -> None:
    if not isinstance(records, list) or not payload:
        return
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        fields = payload.get(_legacy_compat_key(record, index)) or payload.get(("index", index))
        if not fields:
            continue
        for key, value in fields.items():
            if record.get(key) in (None, ""):
                record[key] = copy.deepcopy(value)


def _render_payload_without_legacy_decision_fields(texts, *, coordinate_space: str) -> dict:
    return {
        "texts": _texts_without_legacy_decision_fields(texts),
        "_coordinate_space": coordinate_space,
    }


def _add_timing(telemetry: dict | None, stage: str, seconds: float) -> None:
    if telemetry is None:
        return
    durations = telemetry.setdefault("durations_sec", {})
    durations[stage] = round(float(durations.get(stage, 0.0) or 0.0) + float(seconds), 4)


class _TimingScope:
    def __init__(self, telemetry: dict | None, stage: str):
        self._telemetry = telemetry
        self._stage = stage
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _add_timing(self._telemetry, self._stage, time.perf_counter() - self._start)
        return False


def _timed(telemetry: dict | None, stage: str) -> _TimingScope:
    return _TimingScope(telemetry, stage)


def _strip_band_margin_px(idioma_origem: str = "") -> int:
    raw = os.getenv("TRADUZAI_STRIP_BAND_MARGIN_PX", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    # Dark/black balloons and burst balloons often have low-contrast outlines;
    # their detected bbox can start inside the visual bubble, so a 96px band
    # margin still lets the real balloon/text touch the crop edge.  Keep the
    # split logic unchanged, but use a safer default crop margin so inpaint and
    # mask debug receive the whole balloon.
    return 160


@dataclass(frozen=True)
class OrderedBandContextSnapshot:
    _band_history: tuple[dict, ...]
    _glossario: dict

    def __post_init__(self) -> None:
        object.__setattr__(self, "_band_history", tuple(copy.deepcopy(list(self._band_history))))
        object.__setattr__(self, "_glossario", copy.deepcopy(dict(self._glossario)))

    def to_process_kwargs(self) -> dict:
        return {
            "band_history": [copy.deepcopy(item) for item in self._band_history],
            "glossario": copy.deepcopy(self._glossario),
        }


def _build_ordered_band_context_snapshot(
    running_history: list[dict],
    running_glossary: dict,
    *,
    history_limit: int = 20,
) -> OrderedBandContextSnapshot:
    return OrderedBandContextSnapshot(
        tuple(list(running_history)[-history_limit:]),
        dict(running_glossary or {}),
    )


def _merge_ordered_band_context_after_commit(
    running_history: list[dict],
    running_glossary: dict,
    ocr_result: dict | None,
) -> None:
    if not isinstance(ocr_result, dict):
        return
    result_snapshot = copy.deepcopy(ocr_result)
    running_history.append(result_snapshot)
    additions = result_snapshot.get("_glossary_additions")
    if additions and isinstance(additions, dict):
        running_glossary.update(copy.deepcopy(additions))


def _paste_band_attr_into_image(strip_image, bands: list, attr_name: str):
    result = strip_image.copy()
    strip_height = result.shape[0]
    for band in bands:
        band_slice = getattr(band, attr_name, None)
        if band_slice is None:
            continue
        y0 = max(0, band.y_top)
        y1 = min(strip_height, band.y_bottom)
        h_avail = y1 - y0
        if h_avail <= 0:
            continue
        source = band_slice[:h_avail, :, :]
        original_slice = getattr(band, "original_slice", None)
        if isinstance(original_slice, np.ndarray) and original_slice.shape[:2] == band_slice.shape[:2]:
            original = original_slice[:h_avail, :, :]
            changed = np.any(source != original, axis=2)
            if not np.any(changed):
                continue
            target = result[y0:y1, :, :]
            target[changed] = source[changed]
            result[y0:y1, :, :] = target
            continue
        result[y0:y1, :, :] = source
    return result


def _band_excluded_from_translated_output(band: Band) -> bool:
    result = getattr(band, "ocr_result", None)
    if not isinstance(result, dict):
        return False
    return bool(result.get("excluded_non_story")) or (
        str(result.get("export_policy") or "").strip().lower() == "exclude_from_translated_output"
    )


def _band_text_count(band: Band) -> int:
    result = getattr(band, "ocr_result", None)
    if not isinstance(result, dict):
        return 0
    return len([text for text in list(result.get("texts") or []) if isinstance(text, dict)])


def _band_has_story_text_for_scanlation_tail_guard(band: Band) -> bool:
    result = getattr(band, "ocr_result", None)
    if not isinstance(result, dict):
        return False
    texts = [text for text in list(result.get("texts") or []) if isinstance(text, dict)]
    if not texts:
        return False
    joined = " ".join(str(text.get("text") or "") for text in texts).strip().lower()
    if not joined:
        return False
    scanlation_credit_markers = (
        "discord",
        "iscord",
        "scan",
        "secret",
        "mz",
        "family",
        "support us",
        "recruiting",
        "looking for",
        "proofreader",
        "redrawer",
        "typesetter",
        "translator",
        "leave it blank",
        "message",
        ".gg",
        "http://",
        "https://",
        ".com",
        ".co.kr",
    )
    if any(marker in joined for marker in scanlation_credit_markers):
        return False
    if not re.search(r"[a-z]", joined):
        return False
    return True


def _band_scanlation_promo_reason(band: Band) -> str | None:
    result = getattr(band, "ocr_result", None)
    if not isinstance(result, dict):
        return None
    texts = [text for text in list(result.get("texts") or []) if isinstance(text, dict)]
    if not texts:
        return None
    raw_values = [
        " ".join(
            str(text.get(key) or "")
            for key in ("text", "original", "raw_ocr", "normalized_ocr", "translated", "route_reason", "skip_reason")
        )
        for text in texts
    ]
    joined = " ".join(raw_values).lower()
    compact = re.sub(r"[^a-z0-9]+", "", joined)
    suppressed_count = 0
    for text in texts:
        flags = {str(flag or "").strip().lower() for flag in list(text.get("qa_flags") or [])}
        route_reason = str(text.get("route_reason") or text.get("skip_reason") or "").strip().lower()
        if "scanlation_credit_suppressed" in flags or route_reason == "scanlation_credit_suppressed":
            suppressed_count += 1
    has_discord = "discord" in joined or "iscord" in joined or "discordgg" in compact
    has_url = ".gg" in joined or "discordgg" in compact or "http" in joined
    has_invite = "join" in joined or "support us" in joined

    has_recruitment = any(marker in joined for marker in ("recruiting", "looking for", "hiring", "staff"))
    has_role_or_contact = any(
        marker in joined
        for marker in (
            "translator",
            "proofreader",
            "redrawer",
            "typesetter",
            "leave it blank",
            "message",
        )
    )
    has_scanlation_marker = any(marker in joined for marker in ("secret scans", "secretscans", "scanlator", "scanlation"))
    if has_recruitment and has_role_or_contact and (has_scanlation_marker or suppressed_count >= 2 or has_discord):
        return "scanlation_recruitment_promo"
    if has_discord and has_url and (has_invite or len(texts) <= 4):
        return "scanlation_discord_promo"
    return None


def _band_interval_overlap_ratio(band: Band, intervals: list[tuple[int, int]]) -> float:
    y1 = int(getattr(band, "y_top", 0) or 0)
    y2 = int(getattr(band, "y_bottom", 0) or 0)
    height = max(0, y2 - y1)
    if height <= 0:
        return 0.0
    covered = 0
    for start, end in intervals:
        covered += max(0, min(y2, int(end)) - max(y1, int(start)))
    return covered / float(height)


def _is_scanlation_promo_exclusion_reason(reason: object) -> bool:
    return str(reason or "").strip().lower() in {
        "scanlation_discord_promo",
        "scanlation_recruitment_promo",
        "scanlation_promo",
    }


def _resolve_scanlation_promo_exclusion_owner(
    bands: list[Band],
    source_band: Band,
    source_index: int,
) -> tuple[Band, int]:
    source_y1 = int(getattr(source_band, "y_top", 0) or 0)
    source_y2 = int(getattr(source_band, "y_bottom", 0) or 0)
    source_height = max(0, source_y2 - source_y1)
    if source_height <= 0:
        return source_band, source_index
    source_center = (source_y1 + source_y2) / 2.0
    candidates: list[tuple[int, int, int, Band, int]] = []
    for index, band in enumerate(bands):
        y1 = int(getattr(band, "y_top", 0) or 0)
        y2 = int(getattr(band, "y_bottom", 0) or 0)
        height = max(0, y2 - y1)
        if height <= source_height:
            continue
        if y1 < source_y1:
            continue
        overlap = max(0, min(source_y2, y2) - max(source_y1, y1))
        if overlap <= 0:
            continue
        if not (y1 <= source_center <= y2):
            continue
        if _band_text_count(band) > 0:
            continue
        center_delta = abs(((y1 + y2) / 2.0) - source_center)
        candidates.append((-height, int(center_delta), index, band, index))
    if not candidates:
        return source_band, source_index
    candidates.sort()
    return candidates[0][3], candidates[0][4]


def _page_bounds_for_y(
    source_page_breaks: list[int] | tuple[int, ...] | None,
    strip_height: int | None,
    y: int,
) -> tuple[int, int] | None:
    breaks = sorted({int(value) for value in list(source_page_breaks or []) if int(value) >= 0})
    if strip_height is not None and int(strip_height) > 0:
        breaks.append(int(strip_height))
    breaks = sorted(set(breaks))
    if len(breaks) < 2:
        return None
    for index in range(len(breaks) - 1):
        start = breaks[index]
        end = breaks[index + 1]
        if start <= int(y) < end:
            return start, end
    if int(y) >= breaks[-1]:
        return breaks[-2], breaks[-1]
    return breaks[0], breaks[1]


def _scanlation_promo_tail_page_end(
    bands: list[Band],
    *,
    y_start: int,
    y_end: int,
    source_page_breaks: list[int] | tuple[int, ...] | None,
    strip_height: int | None,
) -> int | None:
    bounds = _page_bounds_for_y(source_page_breaks, strip_height, y_start)
    if bounds is None:
        return None
    page_start, page_end = bounds
    page_height = max(1, page_end - page_start)
    if y_start < page_start + int(page_height * 0.65):
        return None
    for band in bands:
        by1 = int(getattr(band, "y_top", 0) or 0)
        by2 = int(getattr(band, "y_bottom", 0) or 0)
        if by2 <= y_end or by1 >= page_end:
            continue
        if by2 > page_end:
            continue
        if _band_excluded_from_translated_output(band):
            continue
        if _band_has_story_text_for_scanlation_tail_guard(band):
            return None
    return page_end


def _excluded_non_story_intervals(
    bands: list[Band],
    *,
    source_page_breaks: list[int] | tuple[int, ...] | None = None,
    strip_height: int | None = None,
) -> tuple[list[tuple[int, int]], list[dict]]:
    intervals: list[tuple[int, int]] = []
    rows: list[dict] = []
    seen_band_ids: set[str] = set()
    for index, band in enumerate(bands):
        inferred_reason = _band_scanlation_promo_reason(band)
        if not _band_excluded_from_translated_output(band) and not inferred_reason:
            continue
        result = getattr(band, "ocr_result", None) or {}
        source_band_id = _band_debug_id(band, index)
        reason = result.get("exclusion_reason") or inferred_reason or "scanlation_discord_promo"
        owner_band = band
        owner_index = index
        if _is_scanlation_promo_exclusion_reason(reason):
            owner_band, owner_index = _resolve_scanlation_promo_exclusion_owner(bands, band, index)
        y1 = int(getattr(band, "y_top", 0) or 0)
        y2 = int(getattr(band, "y_bottom", 0) or 0)
        if owner_band is not band:
            y1 = int(getattr(owner_band, "y_top", 0) or 0)
            y2 = int(getattr(owner_band, "y_bottom", 0) or 0)
        if _is_scanlation_promo_exclusion_reason(reason):
            source_y1 = int(getattr(band, "y_top", 0) or 0)
            source_y2 = int(getattr(band, "y_bottom", 0) or 0)
            if source_y2 > source_y1:
                y1 = min(y1, source_y1)
                y2 = max(y2, source_y2)
            tail_end = _scanlation_promo_tail_page_end(
                bands,
                y_start=y1,
                y_end=y2,
                source_page_breaks=source_page_breaks,
                strip_height=strip_height,
            )
            if tail_end is not None and tail_end > y2:
                y2 = tail_end
        if y2 <= y1:
            continue
        band_id = _band_debug_id(owner_band, owner_index)
        if band_id in seen_band_ids:
            continue
        seen_band_ids.add(band_id)
        intervals.append((y1, y2))
        row = {
            "band_id": band_id,
            "y_top": y1,
            "y_bottom": y2,
            "content_class": result.get("content_class") or "scanlation_credit",
            "export_policy": "exclude_from_translated_output",
            "translate_policy": result.get("translate_policy") or "skip",
            "inpaint_policy": result.get("inpaint_policy") or "skip",
            "render_policy": result.get("render_policy") or "skip",
            "exclusion_reason": reason,
        }
        if _is_scanlation_promo_exclusion_reason(reason) and y2 != int(getattr(owner_band, "y_bottom", 0) or 0):
            row["extended_to_page_end"] = True
            row["detected_cluster_y_top"] = y1
        if band_id != source_band_id:
            row["detected_band_id"] = source_band_id
        rows.append(row)
    if not intervals:
        return [], rows
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for y1, y2 in intervals:
        if not merged or y1 > merged[-1][1]:
            merged.append((y1, y2))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], y2))
    for index, band in enumerate(bands):
        band_id = _band_debug_id(band, index)
        if band_id in seen_band_ids or _band_excluded_from_translated_output(band):
            continue
        if _band_interval_overlap_ratio(band, merged) < 0.90:
            continue
        if _band_has_story_text_for_scanlation_tail_guard(band):
            continue
        covering_row = None
        by1 = int(getattr(band, "y_top", 0) or 0)
        by2 = int(getattr(band, "y_bottom", 0) or 0)
        for row in rows:
            overlap = max(0, min(by2, int(row.get("y_bottom") or 0)) - max(by1, int(row.get("y_top") or 0)))
            if overlap > 0:
                covering_row = row
                break
        rows.append(
            {
                "band_id": band_id,
                "y_top": by1,
                "y_bottom": by2,
                "content_class": "scanlation_credit",
                "export_policy": "exclude_from_translated_output",
                "translate_policy": "skip",
                "inpaint_policy": "skip",
                "render_policy": "skip",
                "exclusion_reason": (covering_row or {}).get("exclusion_reason") or "scanlation_promo",
                "covered_by_exclusion_band_id": (covering_row or {}).get("band_id"),
                "covered_by_non_story_interval": True,
            }
        )
    return merged, rows


def _remap_y_after_exclusions(y: int, intervals: list[tuple[int, int]]) -> int:
    mapped = int(y)
    for start, end in intervals:
        if y <= start:
            break
        mapped -= max(0, min(int(y), end) - start)
    return max(0, mapped)


def _remove_vertical_intervals(image: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    if not intervals:
        return image
    height = int(image.shape[0])
    keep = np.ones(height, dtype=bool)
    for start, end in intervals:
        keep[max(0, start) : min(height, end)] = False
    if not np.any(keep):
        return image[:0, :, :].copy()
    return image[keep, :, :].copy()


def _remove_band_local_excluded_rows(
    image: np.ndarray | None,
    *,
    band_y_top: int,
    intervals: list[tuple[int, int]],
) -> np.ndarray | None:
    if image is None or not intervals:
        return image
    height = int(image.shape[0])
    keep = np.ones(height, dtype=bool)
    for start, end in intervals:
        local_start = max(0, int(start) - int(band_y_top))
        local_end = min(height, int(end) - int(band_y_top))
        if local_end > local_start:
            keep[local_start:local_end] = False
    if not np.any(keep):
        return image[:0, :, :].copy()
    return image[keep, :, :].copy()


def _remap_bbox_y_after_exclusions(bbox: BBox, intervals: list[tuple[int, int]]) -> BBox:
    return BBox(
        x1=int(bbox.x1),
        y1=_remap_y_after_exclusions(int(bbox.y1), intervals),
        x2=int(bbox.x2),
        y2=_remap_y_after_exclusions(int(bbox.y2), intervals),
    )


def _remap_bands_after_exclusions(bands: list[Band], intervals: list[tuple[int, int]]) -> list[Band]:
    if not intervals:
        return bands
    remapped: list[Band] = []
    for band in bands:
        if _band_excluded_from_translated_output(band):
            continue
        if _band_scanlation_promo_reason(band):
            continue
        if _band_interval_overlap_ratio(band, intervals) >= 0.90 and not _band_has_story_text_for_scanlation_tail_guard(band):
            continue
        old_y_top = int(getattr(band, "y_top", 0) or 0)
        y_top = _remap_y_after_exclusions(int(getattr(band, "y_top", 0) or 0), intervals)
        y_bottom = _remap_y_after_exclusions(int(getattr(band, "y_bottom", 0) or 0), intervals)
        if y_bottom <= y_top:
            continue
        strip_slice = _remove_band_local_excluded_rows(band.strip_slice, band_y_top=old_y_top, intervals=intervals)
        original_slice = _remove_band_local_excluded_rows(band.original_slice, band_y_top=old_y_top, intervals=intervals)
        cleaned_slice = _remove_band_local_excluded_rows(band.cleaned_slice, band_y_top=old_y_top, intervals=intervals)
        rendered_slice = _remove_band_local_excluded_rows(band.rendered_slice, band_y_top=old_y_top, intervals=intervals)
        balloons = [
            Balloon(
                strip_bbox=_remap_bbox_y_after_exclusions(balloon.strip_bbox, intervals),
                confidence=balloon.confidence,
                lobe_count=balloon.lobe_count,
                metadata=copy.deepcopy(balloon.metadata),
            )
            for balloon in list(getattr(band, "balloons", []) or [])
        ]
        remapped.append(
            Band(
                y_top=y_top,
                y_bottom=y_bottom,
                balloons=balloons,
                strip_slice=strip_slice,
                original_slice=original_slice,
                cleaned_slice=cleaned_slice,
                rendered_slice=rendered_slice,
                ocr_result=band.ocr_result,
                perf=dict(getattr(band, "perf", {}) or {}),
            )
        )
    return remapped


def _remap_breaks_after_exclusions(breaks: list[int], intervals: list[tuple[int, int]], new_height: int) -> list[int]:
    if not intervals:
        return breaks
    remapped: list[int] = []
    for value in list(breaks or []):
        mapped = _remap_y_after_exclusions(int(value), intervals)
        if not remapped or mapped > remapped[-1]:
            remapped.append(mapped)
    if not remapped or remapped[0] != 0:
        remapped.insert(0, 0)
    if remapped[-1] != int(new_height):
        remapped.append(int(new_height))
    return remapped


def _write_non_story_exclusions_debug(exclusion_rows: list[dict]) -> None:
    recorder = _get_debug_recorder()
    if recorder is None or not exclusion_rows:
        return
    try:
        recorder.write_json(
            "10_copyback_reassemble/non_story_exclusions.json",
            {
                "excluded_non_story_bands": [row["band_id"] for row in exclusion_rows],
                "excluded_count": len(exclusion_rows),
                "exclusions": exclusion_rows,
            },
        )
    except Exception:
        return


def _shift_bbox_y(value, delta_y: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    return [int(value[0]), int(value[1]) + delta_y, int(value[2]), int(value[3]) + delta_y]


def _bbox4_or_none(value) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _same_bbox(a, b) -> bool:
    left = _bbox4_or_none(a)
    right = _bbox4_or_none(b)
    return left is not None and right is not None and left == right


def _shift_bbox_list_y(values, delta_y: int) -> list[list[int]]:
    shifted: list[list[int]] = []
    for value in values or []:
        bbox = _shift_bbox_y(value, delta_y)
        if bbox is not None:
            shifted.append(bbox)
    return shifted


def _shift_polygons_y(polygons, delta_y: int):
    if not isinstance(polygons, list):
        return polygons
    if polygons and isinstance(polygons[0], (list, tuple)) and len(polygons[0]) >= 2 and not isinstance(polygons[0][0], (list, tuple)):
        return [
            [int(point[0]), int(point[1]) + delta_y]
            if isinstance(point, (list, tuple)) and len(point) >= 2
            else point
            for point in polygons
        ]
    shifted = []
    for polygon in polygons:
        if not isinstance(polygon, list):
            shifted.append(polygon)
            continue
        shifted_polygon = []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                shifted_polygon.append([int(point[0]), int(point[1]) + delta_y])
            else:
                shifted_polygon.append(point)
        shifted.append(shifted_polygon)
    return shifted


def _shift_text_geometry_y(text: dict, delta_y: int) -> dict:
    shifted = dict(text)

    for key in (
        "bbox",
        "source_bbox",
        "balloon_bbox",
        "text_pixel_bbox",
        "layout_bbox",
        "_visual_rect_outer_bbox",
        "_visual_rect_inner_bbox",
        "render_bbox",
        "safe_text_box",
        "_debug_safe_text_box",
        "layout_safe_bbox",
        "position_bbox",
        "capacity_bbox",
        "target_bbox",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "balloon_inner_bbox",
    ):
        bbox = _shift_bbox_y(shifted.get(key), delta_y)
        if bbox is not None:
            shifted[key] = bbox

    for key in (
        "balloon_subregions",
        "connected_lobe_bboxes",
        "connected_text_groups",
        "connected_position_bboxes",
        "connected_focus_bboxes",
        "_merged_source_bboxes",
        "merged_source_bboxes",
    ):
        if key in shifted:
            shifted[key] = _shift_bbox_list_y(shifted.get(key), delta_y)

    for key in ("line_polygons", "connected_lobe_polygons", "balloon_polygon"):
        if key in shifted:
            shifted[key] = _shift_polygons_y(shifted.get(key), delta_y)

    for key in ("qa_metrics", "_render_debug"):
        if isinstance(shifted.get(key), dict):
            shifted[key] = _shift_nested_debug_bboxes_y(shifted[key], delta_y)

    return shifted


def _looks_like_debug_bbox_key(key: str) -> bool:
    key = str(key)
    return key.endswith("bbox") or key.endswith("_bbox") or key.endswith("_bboxes") or key in {
        "safe_text_box",
        "_debug_safe_text_box",
    }


def _shift_nested_debug_bboxes_y(value, delta_y: int):
    if isinstance(value, dict):
        shifted = {}
        for key, item in value.items():
            if _looks_like_debug_bbox_key(str(key)):
                if str(key).endswith("bboxes"):
                    shifted[key] = _shift_bbox_list_y(item, delta_y)
                    continue
                bbox = _shift_bbox_y(item, delta_y)
                if bbox is not None:
                    shifted[key] = bbox
                    continue
            shifted[key] = _shift_nested_debug_bboxes_y(item, delta_y)
        return shifted
    if isinstance(value, list):
        return [_shift_nested_debug_bboxes_y(item, delta_y) for item in value]
    return value


def _inpaint_block_from_vision_block(block: dict) -> dict | None:
    if not isinstance(block, dict):
        return None
    bbox = _shift_bbox_y(block.get("bbox"), 0)
    if bbox is None:
        return None
    out: dict = {"bbox": bbox}
    for key in (
        "confidence",
        "line_polygons",
        "text_pixel_bbox",
        "source_bbox",
        "_merged_source_bboxes",
        "merged_source_bboxes",
        "balloon_bbox",
        "bubble_mask_bbox",
        "bubble_mask_source",
        "bubble_mask_shape",
        "bubble_mask_ellipse",
        "dark_panel_effect_colors",
        "card_panel_text_context",
        "layout_bbox",
        "block_profile",
        "background_type",
        "font_size_px",
        "font_size",
        "rotation_deg",
        "rotation_source",
        "id",
        "text_id",
        "page_id",
        "band_id",
        "trace_id",
        "render_policy",
        "qa_flags",
    ):
        value = block.get(key)
        if value is not None and value != [] and value != "":
            out[key] = copy.deepcopy(value)
    return out


def _enrich_inpaint_block_from_text_layers(block: dict, texts: list[dict]) -> dict:
    if not isinstance(block, dict) or not texts:
        return block
    bbox = _bbox4_or_none(block.get("bbox"))
    if bbox is None:
        return block
    block_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    best_text = None
    best_score = 0.0
    for text in texts:
        if not isinstance(text, dict):
            continue
        text_bbox = _bbox4_or_none(text.get("bbox") or text.get("balloon_bbox") or text.get("text_pixel_bbox"))
        if text_bbox is None:
            continue
        overlap = _bbox_overlap_area(bbox, text_bbox)
        if overlap <= 0:
            continue
        text_area = max(1, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
        score = overlap / float(max(1, min(block_area, text_area)))
        if score > best_score:
            best_score = score
            best_text = text
    if best_text is None or best_score < 0.35:
        return block
    enriched = dict(block)
    for key in (
        "id",
        "text_id",
        "page_id",
        "band_id",
        "trace_id",
        "text_instance_id",
        "source_trace_ids",
        "source_text_ids",
        "_merged_source_bboxes",
        "merged_source_bboxes",
        "rotation_deg",
        "rotation_source",
        "qa_flags",
        "allow_broad_bbox_text_search",
        "block_profile",
        "line_polygons",
        "text_pixel_bbox",
        "bbox",
        "source_bbox",
        "balloon_bbox",
        "bubble_mask_bbox",
        "bubble_mask_source",
        "bubble_mask_shape",
        "bubble_mask_ellipse",
        "dark_panel_effect_colors",
        "card_panel_text_context",
        "layout_bbox",
        "render_policy",
    ):
        value = best_text.get(key)
        if value not in (None, [], ""):
            enriched[key] = copy.deepcopy(value)
    return enriched


def _trace_id_from_record(record: dict, *, fallback_index: int = 0) -> str | None:
    if not isinstance(record, dict):
        return None
    raw_trace_id = record.get("trace_id")
    if raw_trace_id:
        return str(raw_trace_id)
    band_id = str(record.get("band_id") or "")
    text_id = record.get("text_id") or record.get("id")
    if text_id and band_id:
        return f"{text_id}@{band_id}"
    if text_id:
        return str(text_id)
    return None


def _page_id_from_band_id(band_id: str | None) -> str | None:
    if not band_id:
        return None
    match = re.match(r"^(page_\d{3})_band_\d{3}$", str(band_id))
    if match:
        return match.group(1)
    return None


def _trace_ids_by_band_from_page(page: OutputPage) -> dict[str, list[str]]:
    by_band: dict[str, list[str]] = {}
    ocr_result = getattr(page, "ocr_result", {}) if isinstance(getattr(page, "ocr_result", None), dict) else {}
    records = (
        list(ocr_result.get("texts") or [])
        + list(ocr_result.get("_vision_blocks") or [])
        + _page_text_layer_records(page)
    )
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        band_id = str(record.get("band_id") or "")
        if not band_id:
            continue
        trace_id = _trace_id_from_record(record, fallback_index=index)
        if not trace_id:
            continue
        bucket = by_band.setdefault(band_id, [])
        if trace_id not in bucket:
            bucket.append(trace_id)
    return by_band


def _page_text_layer_records(page: OutputPage) -> list[dict]:
    raw_layers = getattr(page, "text_layers", None)
    if isinstance(raw_layers, list):
        return [layer for layer in raw_layers if isinstance(layer, dict)]
    if isinstance(raw_layers, dict):
        for key in ("text_layers", "texts", "layers"):
            value = raw_layers.get(key)
            if isinstance(value, list):
                return [layer for layer in value if isinstance(layer, dict)]
    return []


def _debug_bbox4(value) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in value[:4])
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _debug_bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _debug_bbox_intersection_area(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )


def _record_debug_bboxes(record: dict) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for key in ("bbox", "source_bbox", "balloon_bbox", "layout_bbox", "text_pixel_bbox"):
        bbox = _debug_bbox4(record.get(key))
        if bbox and bbox not in boxes:
            boxes.append(bbox)
    return boxes


def _best_inpaint_trace_source(payload: dict, page: OutputPage, source_blocks: list[dict]) -> dict:
    payload_boxes = _record_debug_bboxes(payload)
    if not payload_boxes:
        return {}
    ocr_result = getattr(page, "ocr_result", {}) if isinstance(getattr(page, "ocr_result", None), dict) else {}
    candidates = (
        list(source_blocks)
        + [text for text in list(ocr_result.get("texts") or []) if isinstance(text, dict)]
        + _page_text_layer_records(page)
    )
    best: tuple[float, dict] = (0.0, {})
    for candidate in candidates:
        candidate_boxes = _record_debug_bboxes(candidate)
        if not candidate_boxes:
            continue
        best_candidate_score = 0.0
        for left in payload_boxes:
            left_area = _debug_bbox_area(left)
            for right in candidate_boxes:
                inter = _debug_bbox_intersection_area(left, right)
                if inter <= 0:
                    continue
                right_area = _debug_bbox_area(right)
                score = inter / float(max(1, min(left_area, right_area)))
                best_candidate_score = max(best_candidate_score, score)
        if best_candidate_score > best[0]:
            best = (best_candidate_score, candidate)
    return best[1] if best[0] >= 0.20 else {}


def _enrich_inpaint_block_debug_payload(
    payload: dict,
    *,
    page: OutputPage,
    page_index: int,
    block_index: int,
    trace_ids_by_band: dict[str, list[str]],
) -> dict:
    ocr_result = getattr(page, "ocr_result", {}) if isinstance(getattr(page, "ocr_result", None), dict) else {}
    source_blocks = [block for block in list(ocr_result.get("_vision_blocks") or []) if isinstance(block, dict)]
    source_block = source_blocks[block_index] if block_index < len(source_blocks) else {}
    if not _trace_id_from_record(source_block, fallback_index=block_index):
        source_block = _best_inpaint_trace_source(payload, page, source_blocks)
    for key in ("id", "text_id", "page_id", "band_id", "trace_id"):
        value = source_block.get(key) if isinstance(source_block, dict) else None
        if value is not None and value != "" and key not in payload:
            payload[key] = copy.deepcopy(value)

    band_id = str(payload.get("band_id") or "")
    page_id = payload.get("page_id") or _page_id_from_band_id(band_id) or _page_id_for(page_index)
    if page_id:
        payload.setdefault("page_id", page_id)
    trace_id = _trace_id_from_record(payload, fallback_index=block_index)
    if trace_id:
        payload.setdefault("trace_id", trace_id)
        payload.setdefault("trace_ids", [trace_id])
    if band_id and trace_ids_by_band.get(band_id):
        payload.setdefault("trace_ids_in_band", list(trace_ids_by_band[band_id]))
    elif trace_id:
        payload.setdefault("trace_ids_in_band", [trace_id])
    return payload


def _ocr_metadata_signature(texts: list[dict], blocks: list[dict]) -> tuple:
    def _bbox(value) -> tuple[int, int, int, int] | tuple:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return tuple()
        return tuple(int(v) for v in value[:4])

    return tuple(
        (
            str(text.get("text") or ""),
            str(text.get("translated") or text.get("traduzido") or ""),
            _bbox(text.get("bbox")),
            _bbox(text.get("text_pixel_bbox")),
            _bbox(text.get("source_bbox")),
            _bbox(text.get("layout_bbox")),
            _bbox(block.get("bbox")) if isinstance(block, dict) else tuple(),
        )
        for text, block in zip(texts, blocks)
        if isinstance(text, dict)
    )


def _finalize_output_page_ocr_metadata(
    page: OutputPage,
    page_number: int,
    total_pages: int | None = None,
) -> bool:
    if not isinstance(getattr(page, "text_layers", None), dict):
        return False
    if not isinstance(getattr(page, "ocr_result", None), dict):
        page.ocr_result = {"_vision_blocks": []}

    texts = [text for text in list(page.text_layers.get("texts") or []) if isinstance(text, dict)]
    blocks = [block for block in list(page.ocr_result.get("_vision_blocks") or []) if isinstance(block, dict)]
    if not texts:
        page.ocr_result["texts"] = []
        page.ocr_result["_vision_blocks"] = []
        return bool(blocks)

    image = getattr(page, "image", None)
    if isinstance(image, np.ndarray) and image.ndim >= 2:
        image_shape = image.shape if image.ndim == 3 else (image.shape[0], image.shape[1], 3)
    else:
        page_height = max(1, int(getattr(page, "y_bottom", 0) or 0) - int(getattr(page, "y_top", 0) or 0))
        page_width = max(1, int(max((text.get("bbox", [0, 0, 1, 1])[2] for text in texts), default=1)))
        image_shape = (page_height, page_width, 3)

    before = _ocr_metadata_signature(texts, blocks)
    text_legacy_fields = _legacy_decision_fields_by_record(texts)
    stage_texts = _texts_without_legacy_decision_fields(texts)
    stage_blocks = [
        _without_legacy_decision_fields(block)
        for block in blocks
        if isinstance(block, dict)
    ]
    try:
        from vision_stack.runtime import _finalize_page_ocr_texts

        final_texts, final_blocks = _finalize_page_ocr_texts(
            stage_texts,
            stage_blocks,
            image_shape,
            page_number=page_number,
            total_pages=total_pages,
        )
    except Exception:
        return False
    _restore_legacy_decision_fields(final_texts, text_legacy_fields)

    for text in final_texts:
        if not isinstance(text, dict):
            continue
        if _same_bbox(text.get("source_bbox"), text.get("balloon_bbox")):
            text_pixel_bbox = _bbox4_or_none(text.get("text_pixel_bbox"))
            if text_pixel_bbox is not None and not _same_bbox(text_pixel_bbox, text.get("balloon_bbox")):
                text["source_bbox"] = text_pixel_bbox
                text["source_bbox_origin"] = "text_pixel_bbox_repaired_from_balloon"

    after = _ocr_metadata_signature(final_texts, final_blocks)
    changed = before != after or len(final_texts) != len(texts) or len(final_blocks) != len(blocks)
    page.text_layers["texts"] = final_texts
    page.ocr_result["texts"] = final_texts
    page.ocr_result["_vision_blocks"] = final_blocks
    return changed



def _source_page_number_for_band(strip: VerticalStrip, band: Band) -> int:
    breaks = list(strip.source_page_breaks or [])
    if len(breaks) < 2:
        return 1

    best_page = 1
    best_overlap = -1
    for index in range(len(breaks) - 1):
        y0 = int(breaks[index])
        y1 = int(breaks[index + 1])
        overlap = max(0, min(int(band.y_bottom), y1) - max(int(band.y_top), y0))
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = index + 1

    return best_page


def _sync_record_page_identity_for_output_page(record: dict, page_index: int) -> dict:
    """Keep final page metadata aligned with the output page that owns the record."""
    final_page_id = _page_id_for(int(page_index) + 1)
    previous_page_id = str(record.get("page_id") or "").strip()
    if previous_page_id and previous_page_id != final_page_id:
        record.setdefault("source_page_id", previous_page_id)
    record["page_id"] = final_page_id
    record["assigned_page_id"] = final_page_id
    return record


def _clamp_record_geometry_to_page(record: dict, *, page_width: int, page_height: int) -> dict:
    if not isinstance(record, dict):
        return record
    max_x = max(1, int(page_width))
    max_y = max(1, int(page_height))

    def _clamp_bbox(value):
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
        except (TypeError, ValueError):
            return None
        x1 = max(0, min(max_x, x1))
        x2 = max(0, min(max_x, x2))
        y1 = max(0, min(max_y, y1))
        y2 = max(0, min(max_y, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    for key in (
        "bbox",
        "source_bbox",
        "text_pixel_bbox",
        "layout_bbox",
        "_raw_text_evidence_bbox",
        "balloon_bbox",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "target_bbox",
        "position_bbox",
        "capacity_bbox",
        "layout_safe_bbox",
        "safe_text_box",
        "_debug_safe_text_box",
        "render_bbox",
    ):
        clamped = _clamp_bbox(record.get(key))
        if clamped is not None:
            record[key] = clamped

    for key in ("balloon_subregions", "connected_lobe_bboxes", "_merged_source_bboxes", "merged_source_bboxes"):
        values = record.get(key)
        if not isinstance(values, list):
            continue
        clamped_values = []
        for value in values:
            clamped = _clamp_bbox(value)
            if clamped is not None:
                clamped_values.append(clamped)
        if clamped_values:
            record[key] = clamped_values
    return record


def _write_strip_detect_debug_artifacts(
    strip: VerticalStrip,
    bands: list[Band],
    *,
    band_margin_px: int,
) -> None:
    try:
        from debug_tools import get_recorder
    except Exception:
        return
    recorder = get_recorder()
    if not recorder or not getattr(recorder, "enabled", False):
        return
    try:
        band_records = []
        for band_index, band in enumerate(bands):
            source_page_number = _source_page_number_for_band(strip, band)
            band_id = _band_id_for(source_page_number, band_index)
            balloon_ids = [
                f"{band_id}_balloon_{balloon_index:02d}"
                for balloon_index, _balloon in enumerate(band.balloons)
            ]
            band_records.append(
                {
                    "band_id": band_id,
                    "source_page_number": int(source_page_number),
                    "band_index": int(band_index),
                    "y_top": int(band.y_top),
                    "y_bottom": int(band.y_bottom),
                    "height": int(band.height),
                    "balloon_count": int(len(band.balloons)),
                    "balloon_ids": balloon_ids,
                }
            )

        recorder.write_json(
            "02_strip_detect/bands_manifest.json",
            {
                "band_count": len(band_records),
                "strip_width": int(strip.width),
                "strip_height": int(strip.height),
                "source_page_breaks": [int(value) for value in list(strip.source_page_breaks or [])],
                "page_x_offsets": [int(value) for value in list(strip.page_x_offsets or [])],
                "band_margin_px": int(band_margin_px),
                "bands": band_records,
            },
        )

        for band_record, band in zip(band_records, bands):
            band_id = band_record["band_id"]
            source_page_number = int(band_record["source_page_number"])
            page_y0, _page_y1 = _source_page_bounds(strip, source_page_number)
            for balloon_index, balloon in enumerate(band.balloons):
                candidate_id = f"{band_id}_cand_{balloon_index:03d}"
                bbox_strip = [
                    int(balloon.strip_bbox.x1),
                    int(balloon.strip_bbox.y1),
                    int(balloon.strip_bbox.x2),
                    int(balloon.strip_bbox.y2),
                ]
                text_evidence = {
                    "has_inner_dark_text": False,
                    "inner_dark_component_count": 0,
                    "inner_dark_area": 0,
                    "significant_component_count": 0,
                    "significant_area": 0,
                    "bright_pixel_ratio": 0.0,
                    "dark_pixel_ratio": 0.0,
                }
                try:
                    text_evidence = _inner_dark_text_evidence(strip.image, balloon.strip_bbox)
                except Exception:
                    pass
                recorder.write_jsonl(
                    "02_strip_detect/detect_candidates.jsonl",
                    {
                        "candidate_id": candidate_id,
                        "band_id": band_id,
                        "bbox_strip": bbox_strip,
                        "bbox_page": [
                            bbox_strip[0],
                            bbox_strip[1] - int(page_y0),
                            bbox_strip[2],
                            bbox_strip[3] - int(page_y0),
                        ],
                        "confidence": round(float(balloon.confidence), 4),
                        "source": "strip_detector",
                        "accepted": True,
                        "reject_reason": None,
                        "matched_text_id": None,
                        **text_evidence,
                    },
                )
    except Exception:
        return


def _bbox_overlap_area(a, b) -> int:
    left = _bbox4_or_none(a)
    right = _bbox4_or_none(b)
    if left is None or right is None:
        return 0
    ix = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    iy = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return ix * iy


def _bbox_area_for_match(bbox) -> int:
    value = _bbox4_or_none(bbox)
    if value is None:
        return 0
    return max(0, value[2] - value[0]) * max(0, value[3] - value[1])


def _bbox_center_inside(outer, inner) -> bool:
    outer_bbox = _bbox4_or_none(outer)
    inner_bbox = _bbox4_or_none(inner)
    if outer_bbox is None or inner_bbox is None:
        return False
    cx = (inner_bbox[0] + inner_bbox[2]) / 2.0
    cy = (inner_bbox[1] + inner_bbox[3]) / 2.0
    return outer_bbox[0] <= cx <= outer_bbox[2] and outer_bbox[1] <= cy <= outer_bbox[3]


def _candidate_matches_band_text_bbox(candidate_bbox: list[int], text: dict) -> bool:
    text_bbox = (
        text.get("text_pixel_bbox")
        or text.get("layout_bbox")
        or text.get("bbox")
        or text.get("balloon_bbox")
    )
    text_area = _bbox_area_for_match(text_bbox)
    if text_area <= 0:
        return False
    overlap = _bbox_overlap_area(candidate_bbox, text_bbox)
    if overlap <= 0:
        return False
    overlap_ratio = overlap / float(text_area)
    if overlap_ratio >= 0.45:
        return True
    return overlap_ratio >= 0.20 and _bbox_center_inside(candidate_bbox, text_bbox)


def _write_strip_detect_text_matching_debug_artifacts(
    strip: VerticalStrip,
    bands: list[Band],
    output_pages: list[OutputPage],
) -> None:
    try:
        from debug_tools import get_recorder
    except Exception:
        return
    recorder = get_recorder()
    if not recorder or not getattr(recorder, "enabled", False):
        return
    try:
        match_rows: list[dict] = []
        texts_by_band: dict[str, list[dict]] = {}
        for page in output_pages:
            text_layers = getattr(page, "text_layers", None)
            if not isinstance(text_layers, dict):
                continue
            for text in list(text_layers.get("texts") or []):
                if not isinstance(text, dict):
                    continue
                band_id = str(text.get("band_id") or "")
                if not band_id:
                    continue
                texts_by_band.setdefault(band_id, []).append(text)

        for band_index, band in enumerate(bands):
            source_page_number = _source_page_number_for_band(strip, band)
            band_id = _band_id_for(source_page_number, band_index)
            page_id = _page_id_for(source_page_number)
            page_y0, _page_y1 = _source_page_bounds(strip, source_page_number)
            band_texts = texts_by_band.get(band_id, [])
            band_trace_ids = [
                str(text.get("trace_id") or _trace_id_from_record(text, fallback_index=index))
                for index, text in enumerate(band_texts)
                if str(text.get("trace_id") or _trace_id_from_record(text, fallback_index=index))
            ]
            for balloon_index, balloon in enumerate(band.balloons):
                candidate_id = f"{band_id}_cand_{balloon_index:03d}"
                bbox_strip = [
                    int(balloon.strip_bbox.x1),
                    int(balloon.strip_bbox.y1),
                    int(balloon.strip_bbox.x2),
                    int(balloon.strip_bbox.y2),
                ]
                bbox_page = [
                    bbox_strip[0],
                    bbox_strip[1] - int(page_y0),
                    bbox_strip[2],
                    bbox_strip[3] - int(page_y0),
                ]
                matched = [
                    text
                    for text in band_texts
                    if _candidate_matches_band_text_bbox(bbox_page, text)
                ]
                match_reason = "same_band_bbox_overlap"
                if not matched and len(band_texts) == 1:
                    matched = list(band_texts)
                    match_reason = "same_band_fallback"
                matched_trace_ids = [
                    str(text.get("trace_id") or _trace_id_from_record(text, fallback_index=index))
                    for index, text in enumerate(matched)
                    if str(text.get("trace_id") or _trace_id_from_record(text, fallback_index=index))
                ]
                matched_text_ids = [
                    str(text.get("text_id") or text.get("id") or "")
                    for text in matched
                    if str(text.get("text_id") or text.get("id") or "")
                ]
                match_method = match_reason if matched_trace_ids else "no_text_in_band"
                row = {
                    "candidate_id": candidate_id,
                    "page_id": page_id,
                    "band_id": band_id,
                    "bbox_page": bbox_page,
                    "matched_text_ids": matched_text_ids,
                    "matched_trace_ids": matched_trace_ids,
                    "band_text_count": len(band_texts),
                    "band_trace_ids": band_trace_ids,
                    "match_count": len(matched_trace_ids),
                    "match_reason": match_method,
                    "match_method": match_method,
                }
                match_rows.append(row)
                recorder.write_jsonl(
                    "02_strip_detect/candidate_text_matching.jsonl",
                    row,
                )
        _enrich_detect_candidates_with_text_matches(recorder, match_rows)
    except Exception:
        return


def _enrich_detect_candidates_with_text_matches(recorder, match_rows: list[dict]) -> None:
    root = getattr(recorder, "_root", None)
    if root is None:
        return
    target = root / "02_strip_detect" / "detect_candidates.jsonl"
    if not target.exists() or not match_rows:
        return
    by_candidate = {
        str(row.get("candidate_id") or ""): row
        for row in match_rows
        if str(row.get("candidate_id") or "")
    }
    rows: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        match = by_candidate.get(str(row.get("candidate_id") or ""))
        if match is not None:
            matched_text_ids = list(match.get("matched_text_ids") or [])
            matched_trace_ids = list(match.get("matched_trace_ids") or [])
            row.update(
                {
                    "page_id": match.get("page_id"),
                    "matched_text_id": matched_text_ids[0] if matched_text_ids else None,
                    "matched_text_ids": matched_text_ids,
                    "matched_trace_ids": matched_trace_ids,
                    "match_count": int(match.get("match_count") or 0),
                    "match_reason": match.get("match_reason"),
                    "match_method": match.get("match_method") or match.get("match_reason"),
                    "band_text_count": int(match.get("band_text_count") or 0),
                    "band_trace_ids": list(match.get("band_trace_ids") or []),
                }
            )
        rows.append(row)
    if not rows:
        return
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    try:
        stage = recorder._stage_from_rel("02_strip_detect/detect_candidates.jsonl")
        recorder.register_artifact(stage=stage, rel_path="02_strip_detect/detect_candidates.jsonl", kind="jsonl")
    except Exception:
        pass


def _build_ocr_confidence_audit(output_pages: list[OutputPage]) -> dict:
    total_blocks = 0
    blocks_with_available_confidence = 0
    blocks_with_confidence_zero = 0
    blocks_with_confidence_lt_05 = 0
    by_band = []
    for page_index, page in enumerate(output_pages, start=1):
        text_layers = getattr(page, "text_layers", None)
        texts = []
        if isinstance(text_layers, dict):
            texts = [text for text in list(text_layers.get("texts") or []) if isinstance(text, dict)]
        total_blocks += len(texts)
        for text in texts:
            raw_available = text.get("confidence_raw") is not None
            try:
                confidence = float(text.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            if raw_available:
                blocks_with_available_confidence += 1
                if confidence == 0.0:
                    blocks_with_confidence_zero += 1
                    by_band.append(
                        {
                            "page_number": int(page_index),
                            "band_id": text.get("band_id"),
                            "text_id": text.get("text_id") or text.get("id"),
                            "confidence_at_accept": text.get("confidence_raw"),
                            "confidence_in_project_json": confidence,
                            "delta": round(confidence - float(text.get("confidence_raw") or 0.0), 4),
                            "lost_between": "_finalize_page_ocr_texts | strip remap",
                        }
                    )
                if confidence < 0.5:
                    blocks_with_confidence_lt_05 += 1
    summary = {
        "total_blocks": int(total_blocks),
        "blocks_with_available_confidence": int(blocks_with_available_confidence),
        "blocks_with_confidence_zero": int(blocks_with_confidence_zero),
        "blocks_with_confidence_lt_05": int(blocks_with_confidence_lt_05),
    }
    if blocks_with_available_confidence and blocks_with_confidence_zero == blocks_with_available_confidence:
        summary["warning"] = "all_blocks_have_confidence_zero_likely_lost_in_metadata_flow"
    return {
        "schema_version": 1,
        "summary": summary,
        "by_band": by_band,
    }


def _write_ocr_confidence_audit(output_pages: list[OutputPage]) -> None:
    try:
        from debug_tools import get_recorder
    except Exception:
        return
    recorder = get_recorder()
    if not recorder or not getattr(recorder, "enabled", False):
        return
    try:
        recorder.write_json("03_ocr/ocr_confidence_audit.json", _build_ocr_confidence_audit(output_pages))
    except Exception:
        return


def _write_pipeline_artifacts_debug(output_pages: list[OutputPage]) -> None:
    recorder = _get_debug_recorder()
    if recorder is None:
        return
    try:
        pages = []
        for page_index, page in enumerate(output_pages, start=1):
            ocr_result = getattr(page, "ocr_result", {}) if isinstance(getattr(page, "ocr_result", {}), dict) else {}
            artifacts = ocr_result.get("_pipeline_artifacts") if isinstance(ocr_result.get("_pipeline_artifacts"), dict) else {}
            engine_preset = ocr_result.get("_engine_preset") if isinstance(ocr_result.get("_engine_preset"), dict) else {}
            pages.append(
                {
                    "page_number": page_index,
                    "engine_preset": engine_preset,
                    "artifacts": artifacts,
                    "by_band": list(ocr_result.get("_pipeline_artifacts_by_band") or []),
                }
            )
        recorder.write_json("00_run/pipeline_artifacts.json", {"schema_version": 1, "pages": pages})
    except Exception:
        return


def _attach_band_pipeline_metadata_to_page(page: OutputPage, band: Band, band_index: int) -> None:
    if not isinstance(getattr(page, "ocr_result", None), dict):
        page.ocr_result = {"_vision_blocks": []}
    ocr_result = band.ocr_result if isinstance(getattr(band, "ocr_result", None), dict) else {}
    if not ocr_result:
        return
    engine_preset = ocr_result.get("_engine_preset") if isinstance(ocr_result.get("_engine_preset"), dict) else {}
    artifacts = (
        ocr_result.get("_pipeline_artifacts")
        if isinstance(ocr_result.get("_pipeline_artifacts"), dict)
        else {}
    )
    if not engine_preset and not artifacts:
        return

    source_page_number = int(ocr_result.get("_source_page_number") or 0) or None
    band_id = str(ocr_result.get("_band_id") or _band_id_for(source_page_number, band_index))
    entry = {
        "band_id": band_id,
        "band_index": int(ocr_result.get("_band_index") or band_index),
        "band_y_top": int(getattr(band, "y_top", 0)),
        "band_y_bottom": int(getattr(band, "y_bottom", 0)),
    }
    if engine_preset:
        entry["engine_preset"] = copy.deepcopy(engine_preset)
        page.ocr_result.setdefault("_engine_preset", copy.deepcopy(engine_preset))
    if artifacts:
        entry["artifacts"] = copy.deepcopy(artifacts)
        page.ocr_result.setdefault("_pipeline_artifacts", copy.deepcopy(artifacts))
    page.ocr_result.setdefault("_pipeline_artifacts_by_band", []).append(entry)


def _mark_pipeline_artifact_status(
    page: OutputPage,
    artifact_name: str,
    status: str,
    *,
    producer: str | None = None,
) -> None:
    if not isinstance(getattr(page, "ocr_result", None), dict):
        return
    artifacts = page.ocr_result.get("_pipeline_artifacts")
    if not isinstance(artifacts, dict):
        return
    artifact = artifacts.get(artifact_name)
    if not isinstance(artifact, dict):
        artifact = {}
        artifacts[artifact_name] = artifact
    if producer and not artifact.get("producer"):
        artifact["producer"] = producer
    artifact["status"] = status
    artifact["updated_by"] = "strip_pipeline"
    for band_entry in page.ocr_result.get("_pipeline_artifacts_by_band") or []:
        if not isinstance(band_entry, dict):
            continue
        band_artifacts = band_entry.get("artifacts")
        if not isinstance(band_artifacts, dict):
            continue
        band_artifact = band_artifacts.get(artifact_name)
        if isinstance(band_artifact, dict):
            if producer and not band_artifact.get("producer"):
                band_artifact["producer"] = producer
            band_artifact["status"] = status
            band_artifact["updated_by"] = "strip_pipeline"
    if isinstance(getattr(page, "page_profile", None), dict):
        page.page_profile["_pipeline_artifacts"] = artifacts


_BUBBLE_IMAGE_FALLBACK_SOURCES = frozenset(
    {
        "image_white_bubble_mask",
        "image_rect_bubble_mask",
        "image_contour_bubble_mask",
    }
)
_BUBBLE_DERIVED_FALLBACK_SOURCES = frozenset(
    {
        "bbox_fallback",
        "balloon_bbox_fallback",
        "derived_white_crop",
        "derived_rectangular_balloon",
        "derived_white_crop_rejected",
        "image_white_region",
        "outline_seeded_contour",
        "rejected_derived_bubble_mask",
    }
)
_BUBBLE_REAL_SOURCES = frozenset(
    {
        "real",
        "real_bubble_mask",
        "worker_bubble_mask",
        "speech-bubble-segmentation",
        "speech_bubble_segmentation",
    }
)


def _contract_value_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return bool(value.size > 0 and np.any(value))
    if isinstance(value, (list, tuple, dict, str, bytes)):
        return bool(value)
    return True


def _bubble_mask_artifact_status_for_texts(texts: list[dict]) -> tuple[str, str]:
    saw_real = False
    saw_image_fallback = False
    saw_fallback = False
    saw_text_needing_bubble = False
    for text in texts:
        if not isinstance(text, dict):
            continue
        has_text_bbox = _contract_value_present(text.get("bbox")) or _contract_value_present(text.get("text_pixel_bbox"))
        has_balloon_hint = (
            _contract_value_present(text.get("balloon_bbox"))
            or _contract_value_present(text.get("bubble_mask_bbox"))
            or _contract_value_present(text.get("balloon_polygon"))
        )
        if has_text_bbox or has_balloon_hint:
            saw_text_needing_bubble = True
        source = str(text.get("bubble_mask_source") or "").strip().lower()
        error = str(text.get("bubble_mask_error") or "").strip().lower()
        has_mask_payload = (
            _contract_value_present(text.get("bubble_mask"))
            or _contract_value_present(text.get("bubble_mask_path"))
            or _contract_value_present(text.get("bubble_mask_layer_path"))
        )
        if source in _BUBBLE_IMAGE_FALLBACK_SOURCES:
            saw_image_fallback = True
            continue
        if source in _BUBBLE_DERIVED_FALLBACK_SOURCES or error:
            saw_fallback = True
            continue
        if source in _BUBBLE_REAL_SOURCES and has_mask_payload:
            saw_real = True
            continue
        if has_mask_payload and source and source not in _BUBBLE_IMAGE_FALLBACK_SOURCES:
            saw_real = True
            continue
        if has_balloon_hint:
            saw_fallback = True
    if saw_image_fallback:
        return "fallback", "image_fallback_mask"
    if saw_fallback:
        return "fallback", "derived_or_bbox_fallback"
    if saw_real:
        return "ok", "real_bubble_mask"
    if saw_text_needing_bubble:
        return "missing", "missing_real_bubble_mask"
    return "missing", "no_text_bubble_evidence"


def _mark_bubble_mask_artifact_from_contract(page: OutputPage) -> None:
    if not isinstance(getattr(page, "ocr_result", None), dict):
        return
    texts = [text for text in page.ocr_result.get("texts") or [] if isinstance(text, dict)]
    status, evidence = _bubble_mask_artifact_status_for_texts(texts)
    _mark_pipeline_artifact_status(page, "BubbleMask", status, producer="speech-bubble-segmentation")
    artifacts = page.ocr_result.get("_pipeline_artifacts")
    if isinstance(artifacts, dict) and isinstance(artifacts.get("BubbleMask"), dict):
        artifacts["BubbleMask"]["evidence"] = evidence
    for band_entry in page.ocr_result.get("_pipeline_artifacts_by_band") or []:
        if not isinstance(band_entry, dict):
            continue
        band_id = str(band_entry.get("band_id") or "")
        band_texts = [text for text in texts if str(text.get("band_id") or "") == band_id] if band_id else texts
        if band_id and not band_texts and not any(str(text.get("band_id") or "") for text in texts):
            band_texts = texts
        band_status, band_evidence = _bubble_mask_artifact_status_for_texts(band_texts)
        band_artifacts = band_entry.get("artifacts")
        if not isinstance(band_artifacts, dict):
            continue
        band_bubble = band_artifacts.get("BubbleMask")
        if not isinstance(band_bubble, dict):
            band_bubble = {"producer": "speech-bubble-segmentation"}
            band_artifacts["BubbleMask"] = band_bubble
        band_bubble["status"] = band_status
        band_bubble["evidence"] = band_evidence
        band_bubble["updated_by"] = "strip_pipeline"


def _mark_pipeline_artifacts_after_render(output_pages: list[OutputPage]) -> None:
    for page in output_pages:
        inpaint_status = "ok" if isinstance(getattr(page, "inpainted_image", None), np.ndarray) else "skipped"
        render_status = "ok" if isinstance(getattr(page, "image", None), np.ndarray) else "pending"
        _mark_pipeline_artifact_status(page, "Inpainted", inpaint_status)
        _mark_pipeline_artifact_status(
            page,
            "FinalRender",
            render_status,
            producer="traduzai-typesetter",
        )
        _mark_bubble_mask_artifact_from_contract(page)


def _get_debug_recorder():
    try:
        from debug_tools import get_recorder
    except Exception:
        return None
    recorder = get_recorder()
    if not recorder or not getattr(recorder, "enabled", False):
        return None
    return recorder


def _write_empty_jsonl_artifact(recorder, rel_path: str) -> None:
    try:
        root = Path(getattr(recorder, "_root"))
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)
        recorder.register_artifact(
            stage=getattr(recorder, "_stage_from_rel")(rel_path),
            rel_path=rel_path,
            kind="jsonl",
        )
    except Exception:
        return


def _write_input_manifest_debug(page_paths: list[Path], strip: VerticalStrip) -> None:
    recorder = _get_debug_recorder()
    if recorder is None:
        return
    try:
        breaks = [int(value) for value in list(strip.source_page_breaks or [])]
        files = []
        for index, path in enumerate(page_paths):
            y_top = breaks[index] if index < len(breaks) else None
            y_bottom = breaks[index + 1] if index + 1 < len(breaks) else None
            files.append(
                {
                    "page_number": index + 1,
                    "path": str(path),
                    "strip_y_top": y_top,
                    "strip_y_bottom": y_bottom,
                    "height": (int(y_bottom) - int(y_top)) if y_top is not None and y_bottom is not None else None,
                }
            )
        recorder.write_json(
            "01_input_extract/input_manifest.json",
            {
                "input_page_count": len(page_paths),
                "strip_width": int(strip.width),
                "strip_height": int(strip.height),
                "source_page_breaks": breaks,
                "files": files,
            },
        )
    except Exception:
        return


def _write_reassemble_manifest_debug(
    output_pages: list[OutputPage],
    original_pages: list[OutputPage],
    clean_pages: list[OutputPage],
    *,
    target_count: int,
) -> None:
    recorder = _get_debug_recorder()
    if recorder is None:
        return
    try:
        pages = []
        for index, page in enumerate(output_pages):
            image = getattr(page, "image", None)
            height = int(image.shape[0]) if isinstance(image, np.ndarray) and image.ndim >= 2 else 0
            width = int(image.shape[1]) if isinstance(image, np.ndarray) and image.ndim >= 2 else 0
            pages.append(
                {
                    "page_number": index + 1,
                    "y_top": int(getattr(page, "y_top", 0) or 0),
                    "y_bottom": int(getattr(page, "y_bottom", 0) or 0),
                    "height": height,
                    "width": width,
                }
            )
        recorder.write_json(
            "10_copyback_reassemble/reassemble_manifest.json",
            {
                "target_count": int(target_count),
                "rendered_page_count": len(output_pages),
                "original_page_count": len(original_pages),
                "clean_page_count": len(clean_pages),
                "pages": pages,
            },
        )
    except Exception:
        return


def _write_inpaint_blocks_debug(output_pages: list[OutputPage]) -> None:
    recorder = _get_debug_recorder()
    if recorder is None:
        return
    rel_path = "08_inpaint/inpaint_blocks.jsonl"
    wrote = False
    try:
        for page_index, page in enumerate(output_pages, start=1):
            trace_ids_by_band = _trace_ids_by_band_from_page(page)
            for block_index, block in enumerate(list(getattr(page, "inpaint_blocks", None) or [])):
                payload = dict(block) if isinstance(block, dict) else {"value": block}
                payload.setdefault("page_number", page_index)
                payload.setdefault("block_index", block_index)
                payload = _enrich_inpaint_block_debug_payload(
                    payload,
                    page=page,
                    page_index=page_index,
                    block_index=block_index,
                    trace_ids_by_band=trace_ids_by_band,
                )
                recorder.write_jsonl(rel_path, payload)
                wrote = True
        if not wrote:
            _write_empty_jsonl_artifact(recorder, rel_path)
    except Exception:
        return


def _write_page_cleanup_breakdown_debug(breakdown: dict[str, float]) -> None:
    recorder = _get_debug_recorder()
    if recorder is None:
        return
    try:
        rounded = {key: round(float(value or 0.0), 4) for key, value in sorted(breakdown.items())}
        cleanup_skipped = bool(rounded.pop("page_cleanup_skipped", 0.0))
        recorder.write_json(
            "10_copyback_reassemble/page_cleanup_breakdown.json",
            {"durations_sec": rounded, "cleanup_skipped": cleanup_skipped},
        )
    except Exception:
        return


def _dark_text_cleanup_loses_visible_ink(
    before: np.ndarray,
    after: np.ndarray,
    page_texts: list[dict],
) -> bool:
    if not isinstance(before, np.ndarray) or not isinstance(after, np.ndarray):
        return False
    if before.shape != after.shape or before.ndim < 3:
        return False

    def _bbox4(value) -> list[int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in value]
        except Exception:
            return None
        h, w = before.shape[:2]
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _bright_ink(image: np.ndarray, bbox: list[int]) -> int:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2, :3]
        if crop.size == 0:
            return 0
        bright = (crop[:, :, 0] >= 170) & (crop[:, :, 1] >= 170) & (crop[:, :, 2] >= 170)
        return int(np.count_nonzero(bright))

    for text in page_texts or []:
        if not isinstance(text, dict):
            continue
        flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
        source = str(text.get("bubble_mask_source") or text.get("balloon_mask_source") or "").strip().lower()
        dark_text = bool(
            source in {"image_dark_bubble_mask", "image_dark_panel_mask"}
            or flags
            & {
                "dark_bubble_ellipse_bbox_mask",
                "dark_bubble_oval_reocr",
                "dark_bubble_visual_glyph_mask_replaced_geometry",
                "auto_dark_panel_glow_fallback",
            }
        )
        if not dark_text:
            continue
        bbox = _bbox4(
            text.get("safe_text_box")
            or text.get("render_bbox")
            or text.get("balloon_bbox")
            or text.get("bubble_mask_bbox")
            or text.get("bbox")
        )
        if bbox is None:
            continue
        before_ink = _bright_ink(before, bbox)
        if before_ink < 160:
            continue
        after_ink = _bright_ink(after, bbox)
        if after_ink < int(before_ink * 0.52):
            return True
    return False


def _write_contact_sheets_debug(
    original_pages: list[OutputPage],
    output_pages: list[OutputPage],
    bands: list[Band],
) -> None:
    recorder = _get_debug_recorder()
    if recorder is None:
        return
    try:
        from debug_tools.contact_sheets import problem_bands_sheet, translated_comparison_sheet

        recorder.write_image(
            "12_contact_sheets/translated_comparison.jpg",
            translated_comparison_sheet(original_pages, output_pages),
        )
        recorder.write_image(
            "12_contact_sheets/problem_bands.jpg",
            problem_bands_sheet(bands),
        )
    except Exception:
        return


def _band_debug_id(band: Band, fallback_index: int) -> str:
    ocr_result = getattr(band, "ocr_result", None)
    if isinstance(ocr_result, dict):
        raw = ocr_result.get("_band_id") or ocr_result.get("band_id")
        if raw:
            return str(raw)
        for text in list(ocr_result.get("texts") or []):
            if isinstance(text, dict) and text.get("band_id"):
                return str(text["band_id"])
        for block in list(ocr_result.get("_vision_blocks") or []):
            if isinstance(block, dict) and block.get("band_id"):
                return str(block["band_id"])
        source_page_number = ocr_result.get("_source_page_number") or ocr_result.get("source_page_number")
        if source_page_number:
            return _band_id_for(int(source_page_number), fallback_index)
    return _band_id_for(1, fallback_index)


def _output_page_crop_for_band(
    output_pages: list[OutputPage],
    band: Band,
) -> tuple[int, OutputPage, np.ndarray, np.ndarray, int, int] | None:
    best_page_index = None
    best_overlap = 0
    band_y_top = int(getattr(band, "y_top", 0) or 0)
    band_y_bottom = int(getattr(band, "y_bottom", 0) or 0)
    for page_index, page in enumerate(output_pages):
        overlap = max(
            0,
            min(band_y_bottom, int(getattr(page, "y_bottom", 0) or 0))
            - max(band_y_top, int(getattr(page, "y_top", 0) or 0)),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_page_index = page_index
    if best_page_index is None or best_overlap <= 0:
        return None

    page = output_pages[best_page_index]
    image = getattr(page, "image", None)
    if not isinstance(image, np.ndarray) or image.ndim < 2 or image.size == 0:
        return None
    page_y_top = int(getattr(page, "y_top", 0) or 0)
    crop_y1 = max(0, band_y_top - page_y_top)
    crop_y2 = min(int(image.shape[0]), band_y_bottom - page_y_top)
    if crop_y2 <= crop_y1:
        return None
    crop = image[crop_y1:crop_y2, :, :]
    if crop.size == 0:
        return None
    return best_page_index, page, image, crop, crop_y1, crop_y2


def _stitch_output_band_crop(output_pages: list[OutputPage], band: Band) -> np.ndarray | None:
    band_y_top = int(getattr(band, "y_top", 0) or 0)
    band_y_bottom = int(getattr(band, "y_bottom", 0) or 0)
    valid_pages: list[tuple[OutputPage, np.ndarray, int, int]] = []
    for page in output_pages:
        image = getattr(page, "image", None)
        if not isinstance(image, np.ndarray) or image.ndim < 2 or image.size == 0:
            continue
        page_y_top = int(getattr(page, "y_top", 0) or 0)
        page_y_bottom = min(
            int(getattr(page, "y_bottom", page_y_top + image.shape[0]) or 0),
            page_y_top + int(image.shape[0]),
        )
        if page_y_bottom > page_y_top:
            valid_pages.append((page, image, page_y_top, page_y_bottom))
    if not valid_pages:
        return None

    visible_y_top = max(band_y_top, min(item[2] for item in valid_pages))
    visible_y_bottom = min(band_y_bottom, max(item[3] for item in valid_pages))
    height = visible_y_bottom - visible_y_top
    if height <= 0:
        return None

    canvas: np.ndarray | None = None
    covered = np.zeros(height, dtype=bool)
    for _page, image, page_y_top, page_y_bottom in valid_pages:
        overlap_y1 = max(visible_y_top, page_y_top)
        overlap_y2 = min(visible_y_bottom, page_y_bottom)
        if overlap_y2 <= overlap_y1:
            continue
        if canvas is None:
            canvas = np.empty((height, *image.shape[1:]), dtype=image.dtype)
        if image.shape[1:] != canvas.shape[1:] or image.dtype != canvas.dtype:
            return None

        source_y1 = overlap_y1 - page_y_top
        source_y2 = overlap_y2 - page_y_top
        target_y1 = overlap_y1 - visible_y_top
        target_y2 = overlap_y2 - visible_y_top
        uncovered = ~covered[target_y1:target_y2]
        if not np.any(uncovered):
            continue
        canvas[target_y1:target_y2][uncovered] = image[source_y1:source_y2][uncovered]
        covered[target_y1:target_y2][uncovered] = True

    if canvas is None or not bool(np.all(covered)):
        return None
    return canvas


def _write_lossless_visual_baseline(output_pages: list[OutputPage], bands: list[Band]) -> None:
    recorder = _get_debug_recorder()
    if recorder is None:
        return
    canonical_enabled = str(
        os.getenv("TRADUZAI_FLAG_VISUAL_BASELINE_LOSSLESS_V2", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not canonical_enabled:
        return
    try:
        page_ids = [f"page_{page_index + 1:03d}" for page_index in range(len(output_pages))]
        band_ids = [_band_debug_id(band, band_index) for band_index, band in enumerate(bands)]
        recorder.set_canonical_expected_coverage(page_ids=page_ids, band_ids=band_ids)
        text_metrics: list[dict] = []
        for page_index, page in enumerate(output_pages):
            image = getattr(page, "image", None)
            if isinstance(image, np.ndarray) and image.ndim >= 2 and image.size:
                recorder.write_canonical_image(
                    "page",
                    image,
                    page_id=f"page_{page_index + 1:03d}",
                    color_space="bgr",
                )
            for text in _page_texts_from_text_layers(getattr(page, "text_layers", None)):
                metric = dict(text)
                metric.setdefault("page_id", f"page_{page_index + 1:03d}")
                text_metrics.append(metric)
        recorder.record_canonical_text_metrics(text_metrics)
        for band_index, band in enumerate(bands):
            crop = _stitch_output_band_crop(output_pages, band)
            if crop is None:
                continue
            band_id = _band_debug_id(band, band_index)
            page_id = (
                band_id.split("_band_", 1)[0]
                if "_band_" in band_id
                else "page_unknown"
            )
            recorder.write_canonical_image(
                "final_band",
                crop,
                page_id=page_id,
                band_id=band_id,
                color_space="bgr",
            )
    except Exception as exc:
        recorder.event(
            "00_run",
            "lossless_visual_baseline_failed",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return


def _write_final_band_crop_debug(output_pages: list[OutputPage], bands: list[Band]) -> None:
    recorder = _get_debug_recorder()
    if recorder is None:
        return
    try:
        trace_ids_by_page = {
            page_index: _trace_ids_by_band_from_page(page)
            for page_index, page in enumerate(output_pages)
        }
        for band_index, band in enumerate(bands):
            band_id = _band_debug_id(band, band_index)
            rendered = getattr(band, "rendered_slice", None)
            rendered_rel = f"09_typeset/rendered_bands/{band_id}.jpg"
            if isinstance(rendered, np.ndarray) and rendered.size:
                recorder.write_image(rendered_rel, rendered, quality=92)

            band_y_top = int(getattr(band, "y_top", 0) or 0)
            band_y_bottom = int(getattr(band, "y_bottom", 0) or 0)
            crop_record = _output_page_crop_for_band(output_pages, band)
            if crop_record is None:
                continue
            best_page_index, page, image, crop, crop_y1, crop_y2 = crop_record
            page_y_top = int(getattr(page, "y_top", 0) or 0)

            final_rel = f"10_copyback_reassemble/final_bands/{band_id}.jpg"
            recorder.write_image(final_rel, crop, quality=92)
            output_name = (
                Path(getattr(page, "path")).name
                if getattr(page, "path", None)
                else f"{best_page_index + 1:03d}.jpg"
            )
            recorder.write_jsonl(
                "10_copyback_reassemble/final_band_crops.jsonl",
                {
                    "band_id": band_id,
                    "translated_output_page": output_name,
                    "output_page_number": int(best_page_index + 1),
                    "output_page_y_top": page_y_top,
                    "output_page_y_bottom": int(getattr(page, "y_bottom", 0) or 0),
                    "band_y_top": band_y_top,
                    "band_y_bottom": band_y_bottom,
                    "crop_bbox_in_translated_page": [0, crop_y1, int(image.shape[1]), crop_y2],
                    "final_crop_path": final_rel,
                    "rendered_band_path": rendered_rel,
                    "trace_ids": list(trace_ids_by_page.get(best_page_index, {}).get(band_id, [])),
                },
            )
    except Exception:
        return


def _debug_skip_page_cleanup_rerender() -> bool:
    rerender = os.getenv("TRADUZAI_PAGE_CLEANUP_RERENDER")
    if rerender is not None:
        return str(rerender).strip().lower() in {"0", "false", "no", "off"}
    raw = os.getenv("TRADUZAI_DEBUG_SKIP_PAGE_CLEANUP_RERENDER", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _strip_final_page_space_typeset_enabled() -> bool:
    raw = os.getenv("TRADUZAI_STRIP_FINAL_PAGE_SPACE_TYPESET", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _strip_inpainter_prewarm_enabled() -> bool:
    raw = os.getenv("TRADUZAI_STRIP_INPAINTER_PREWARM", "1")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _strip_scheduler_executor_enabled() -> bool:
    return bool(_strip_scheduler_executor_mode())


def _strip_scheduler_executor_mode() -> str:
    raw = str(os.getenv("TRADUZAI_STRIP_SCHEDULER_EXECUTOR", "")).strip().lower()
    if raw in {"overlap", "overlap_context_release"}:
        return "overlap_context_release"
    if raw in {"1", "true", "yes", "on"}:
        return "sequential_safe"
    return ""


def _strip_parallel_inpaint_threads() -> int:
    explicit = os.getenv("TRADUZAI_STRIP_PARALLEL_INPAINT_THREADS")
    if explicit is None:
        explicit = os.getenv("TRADUZAI_STRIP_INPAINT_WORKERS")
    if explicit is not None:
        raw = str(explicit).strip().lower()
        if raw in {"", "0", "1", "false", "no", "off"}:
            return 1
        if raw in {"true", "yes", "on"}:
            return 2
        try:
            return max(1, min(4, int(raw)))
        except Exception:
            return 1

    enabled = str(os.getenv("TRADUZAI_STRIP_PARALLEL_INPAINT", "")).strip().lower()
    if enabled in {"1", "true", "yes", "on"}:
        return 2
    return 1


def _start_inpainter_prewarm(inpainter, work_available=True) -> tuple[ThreadPoolExecutor, Future] | None:
    if not work_available or not _strip_inpainter_prewarm_enabled():
        return None
    prewarm = getattr(inpainter, "prewarm_band_inpainter", None)
    if not callable(prewarm):
        return None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="traduzai-inpaint-prewarm")
    future = executor.submit(prewarm)
    return executor, future


def _close_inpainter_prewarm(handle: tuple[ThreadPoolExecutor, Future] | None) -> None:
    if handle is None:
        return
    executor, future = handle
    if future.done():
        try:
            future.result()
        except Exception:
            pass
    executor.shutdown(wait=False, cancel_futures=True)


def _macro_ocr_shadow_enabled() -> bool:
    raw = os.getenv("TRADUZAI_MACRO_OCR_SHADOW", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _macro_ocr_gate_on_fallback_resolved_enabled() -> bool:
    raw = os.getenv("TRADUZAI_MACRO_OCR_GATE_FALLBACK_RESOLVED", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _macro_ocr_real_enabled() -> bool:
    raw = os.getenv("TRADUZAI_MACRO_OCR", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _macro_ocr_precompute_min_blocks() -> int:
    return max(1, _env_int("TRADUZAI_MACRO_OCR_PRECOMPUTE_MIN_BLOCKS", 1))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _image_io_worker_count(page_count: int) -> int:
    if page_count <= 1 or not _env_bool("TRADUZAI_PARALLEL_IMAGE_IO", True):
        return 1
    default_workers = min(4, max(1, os.cpu_count() or 1))
    workers = max(1, _env_int("TRADUZAI_IMAGE_IO_WORKERS", default_workers))
    return min(int(page_count), workers)


def _write_jpeg_timed(path: Path, image: np.ndarray, *, quality: int = 92) -> float:
    started = time.perf_counter()
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise IOError(f"Falha ao gravar imagem: {path}")
    return time.perf_counter() - started


def _write_output_pages_jpegs(output_pages: list[OutputPage], output_dir: Path, *, quality: int = 92) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(output_pages):
        page.path = output_dir / f"{i + 1:03d}.jpg"

    workers = _image_io_worker_count(len(output_pages))
    if workers <= 1:
        return sum(_write_jpeg_timed(page.path, page.image, quality=quality) for page in output_pages)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="traduzai-image-io") as pool:
        futures = [pool.submit(_write_jpeg_timed, page.path, page.image, quality=quality) for page in output_pages]
        return sum(future.result() for future in futures)


def _write_output_pages_after_lossless_debug(
    output_pages: list[OutputPage],
    bands: list[Band],
    output_dir: Path,
) -> float:
    _write_lossless_visual_baseline(output_pages, bands)
    return _write_output_pages_jpegs(output_pages, output_dir)


def _page_final_near_text_cleanup_enabled() -> bool:
    return _env_bool("TRADUZAI_ENABLE_PAGE_FINAL_NEAR_TEXT_CLEANUP", True)


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / float(total), 4)


def _run_macro_ocr_shadow(
    output_pages: list[OutputPage],
    runtime,
    *,
    idioma_origem: str = "en",
) -> dict:
    started_at = time.perf_counter()
    try:
        from ocr.macro_ocr import (
            collect_page_ocr_blocks,
            compare_aligned_macro_ocr_texts,
            estimate_macro_ocr_fallback_cost,
            recognize_macro_ocr_windows,
        )
    except Exception as exc:
        return _macro_ocr_shadow_blocked(started_at, [f"macro OCR helpers unavailable: {exc}"])

    get_ocr = getattr(runtime, "_get_ocr_engine", None)
    ocr_profile = "quality"
    if not callable(get_ocr):
        try:
            from vision_stack.runtime import _get_ocr_engine as get_ocr

            ocr_profile = "max"
        except Exception as exc:
            return _macro_ocr_shadow_blocked(
                started_at,
                [f"runtime has no _get_ocr_engine and fallback import failed: {exc}"],
            )

    try:
        ocr_engine = get_ocr(ocr_profile, lang=idioma_origem)
    except TypeError:
        try:
            ocr_engine = get_ocr(ocr_profile)
        except Exception as exc:
            return _macro_ocr_shadow_blocked(started_at, [f"OCR engine unavailable: {exc}"])
    except Exception as exc:
        return _macro_ocr_shadow_blocked(started_at, [f"OCR engine unavailable: {exc}"])

    totals = {
        "total": 0,
        "missing_count": 0,
        "different_count": 0,
        "fallback_resolved_different_count": 0,
        "material_different_count": 0,
        "line_marker_artifact_count": 0,
        "line_marker_minor_variation_count": 0,
        "minor_ocr_variation_count": 0,
        "numeric_confusable_variation_count": 0,
        "episode_marker_variation_count": 0,
        "numeric_token_change_count": 0,
        "fallback_required_count": 0,
        "acceptable_variation_count": 0,
        "exact_match_count": 0,
        "crop_fallback_attempts": 0,
        "crop_fallback_recovered": 0,
        "macro_window_count": 0,
    }
    blocks_processed = 0
    page_reports: list[dict] = []

    for page_number, page in enumerate(output_pages, start=1):
        image_rgb = getattr(page, "original_image", None)
        if image_rgb is None:
            continue
        page_payload = {
            "numero": page_number,
            "inpaint_blocks": list(page.inpaint_blocks or []),
            "text_layers": list((page.text_layers or {}).get("texts") or []),
        }
        blocks = collect_page_ocr_blocks(page_payload)
        if not blocks:
            continue

        macro_texts, ocr_stats, windows = recognize_macro_ocr_windows(
            ocr_engine,
            image_rgb,
            blocks,
            window_mode="band-groups",
            crop_fallback_max=0,
            window_max_blocks=_env_int("TRADUZAI_MACRO_OCR_WINDOW_MAX_BLOCKS", 2),
            window_merge_gap=_env_int("TRADUZAI_MACRO_OCR_WINDOW_MERGE_GAP", 1000),
            window_padding=_env_int("TRADUZAI_MACRO_OCR_WINDOW_PADDING", 96),
        )
        baseline_texts = list((page.text_layers or {}).get("texts") or [])
        compare = compare_aligned_macro_ocr_texts(baseline_texts, macro_texts)
        totals["total"] += int(compare["total"])
        totals["missing_count"] += int(compare["missing_count"])
        totals["different_count"] += int(compare["different_count"])
        totals["fallback_resolved_different_count"] += int(
            compare.get("fallback_resolved_different_count", 0)
        )
        totals["material_different_count"] += int(compare.get("material_different_count", 0))
        totals["line_marker_artifact_count"] += int(compare.get("line_marker_artifact_count", 0))
        totals["line_marker_minor_variation_count"] += int(
            compare.get("line_marker_minor_variation_count", 0)
        )
        totals["minor_ocr_variation_count"] += int(compare.get("minor_ocr_variation_count", 0))
        totals["numeric_confusable_variation_count"] += int(
            compare.get("numeric_confusable_variation_count", 0)
        )
        totals["episode_marker_variation_count"] += int(
            compare.get("episode_marker_variation_count", 0)
        )
        totals["numeric_token_change_count"] += int(compare.get("numeric_token_change_count", 0))
        totals["fallback_required_count"] += int(compare.get("fallback_required_count", 0))
        totals["acceptable_variation_count"] += int(compare.get("acceptable_variation_count", 0))
        totals["exact_match_count"] += int(compare["exact_match_count"])
        totals["crop_fallback_attempts"] += _env_int_from_value(ocr_stats.get("crop_fallback_attempts"))
        totals["crop_fallback_recovered"] += _env_int_from_value(ocr_stats.get("crop_fallback_recovered"))
        totals["macro_window_count"] += _env_int_from_value(ocr_stats.get("macro_window_count"))
        blocks_processed += len(blocks)
        page_reports.append(
            {
                "page_number": page_number,
                "blocks": len(blocks),
                "macro_window_count": _env_int_from_value(ocr_stats.get("macro_window_count")),
                **compare,
                "ocr_stats": ocr_stats,
                "window_count": len(windows),
            }
        )

    missing_text_rate = _rate(totals["missing_count"], totals["total"])
    different_text_rate = _rate(totals["different_count"], totals["total"])
    fallback_resolved_different_text_rate = _rate(
        totals["fallback_resolved_different_count"], totals["total"]
    )
    gate_on_fallback_resolved_text = _macro_ocr_gate_on_fallback_resolved_enabled()
    text_quality_gate_rate = (
        fallback_resolved_different_text_rate
        if gate_on_fallback_resolved_text
        else different_text_rate
    )
    material_different_text_rate = _rate(totals["material_different_count"], totals["total"])
    fallback_required_text_rate = _rate(totals["fallback_required_count"], totals["total"])
    fallback_rate = _rate(totals["crop_fallback_attempts"], blocks_processed)
    window_reduction_rate = _rate(
        max(0, blocks_processed - totals["macro_window_count"]),
        blocks_processed,
    )
    fallback_cost = estimate_macro_ocr_fallback_cost(
        block_count=blocks_processed,
        macro_window_count=totals["macro_window_count"],
        material_different_count=totals["material_different_count"],
        fallback_required_count=totals["fallback_required_count"],
    )
    status = "PASS"
    reasons = ["macro OCR shadow stayed within thresholds"]
    if missing_text_rate > 0.02 or text_quality_gate_rate > 0.25 or fallback_rate > 0.15:
        status = "FAIL"
        reasons = []
        if missing_text_rate > 0.02:
            reasons.append(f"missing text rate {missing_text_rate:.2%} exceeds 2.00%")
        if text_quality_gate_rate > 0.25:
            text_rate_label = (
                "fallback-resolved different text rate"
                if gate_on_fallback_resolved_text
                else "different text rate"
            )
            reasons.append(f"{text_rate_label} {text_quality_gate_rate:.2%} exceeds 25.00%")
        if fallback_rate > 0.15:
            reasons.append(f"fallback rate {fallback_rate:.2%} exceeds 15.00%")

    return {
        "status": status,
        "reasons": reasons,
        "window_mode": "band-groups",
        "runtime_seconds": round(time.perf_counter() - started_at, 4),
        "pages_processed": len(page_reports),
        "blocks_processed": blocks_processed,
        "text_line_count": totals["total"],
        "missing_count": totals["missing_count"],
        "different_count": totals["different_count"],
        "fallback_resolved_different_count": totals["fallback_resolved_different_count"],
        "material_different_count": totals["material_different_count"],
        "line_marker_artifact_count": totals["line_marker_artifact_count"],
        "line_marker_minor_variation_count": totals["line_marker_minor_variation_count"],
        "minor_ocr_variation_count": totals["minor_ocr_variation_count"],
        "numeric_confusable_variation_count": totals["numeric_confusable_variation_count"],
        "episode_marker_variation_count": totals["episode_marker_variation_count"],
        "numeric_token_change_count": totals["numeric_token_change_count"],
        "fallback_required_count": totals["fallback_required_count"],
        "acceptable_variation_count": totals["acceptable_variation_count"],
        "exact_match_count": totals["exact_match_count"],
        "crop_fallback_attempts": totals["crop_fallback_attempts"],
        "crop_fallback_recovered": totals["crop_fallback_recovered"],
        "macro_window_count": totals["macro_window_count"],
        "fallback_adjusted_ocr_call_count": fallback_cost["effective_ocr_call_count"],
        "fallback_adjusted_window_reduction_rate": fallback_cost[
            "fallback_adjusted_window_reduction_rate"
        ],
        "window_reduction_rate": window_reduction_rate,
        "missing_text_rate": missing_text_rate,
        "different_text_rate": different_text_rate,
        "fallback_resolved_different_text_rate": fallback_resolved_different_text_rate,
        "text_quality_gate_rate": text_quality_gate_rate,
        "gate_on_fallback_resolved_text": gate_on_fallback_resolved_text,
        "material_different_text_rate": material_different_text_rate,
        "fallback_required_text_rate": fallback_required_text_rate,
        "fallback_rate": fallback_rate,
        "page_reports": page_reports,
    }


def _env_int_from_value(value, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _macro_ocr_shadow_blocked(started_at: float, reasons: list[str]) -> dict:
    return {
        "status": "BLOCK",
        "reasons": reasons,
        "window_mode": "band-groups",
        "runtime_seconds": round(time.perf_counter() - started_at, 4),
        "pages_processed": 0,
        "blocks_processed": 0,
        "text_line_count": 0,
        "macro_window_count": 0,
    }


def _get_macro_ocr_engine(runtime, *, idioma_origem: str):
    get_ocr = getattr(runtime, "_get_ocr_engine", None)
    if not callable(get_ocr):
        from vision_stack.runtime import _get_ocr_engine as get_ocr

    try:
        return get_ocr("quality", lang=idioma_origem)
    except TypeError:
        return get_ocr("quality")


def _source_page_bounds(strip: VerticalStrip, page_number: int) -> tuple[int, int]:
    breaks = list(strip.source_page_breaks or [])
    page_index = max(0, int(page_number) - 1)
    if page_index + 1 < len(breaks):
        return int(breaks[page_index]), int(breaks[page_index + 1])
    return 0, int(strip.height)


def _build_scheduler_executor_report(*, band_count: int, page_count: int) -> dict | None:
    mode = _strip_scheduler_executor_mode()
    if not mode:
        return None
    try:
        from strip.scheduler import build_strip_scheduler_plan
    except Exception:
        from pipeline.strip.scheduler import build_strip_scheduler_plan

    plan = build_strip_scheduler_plan(band_count=band_count, page_count=max(1, page_count))
    parallel_inpaint_threads = _strip_parallel_inpaint_threads() if mode == "overlap_context_release" else 1
    overlap_worker_count = max(2, parallel_inpaint_threads + 1) if mode == "overlap_context_release" else 1
    inpaint_lock_mode = "bounded_semaphore" if parallel_inpaint_threads > 1 else "shared_gpu_lock"
    return {
        "enabled": True,
        "mode": mode,
        "processed_band_count": 0,
        "task_count": plan.task_count,
        "cpu_task_count": plan.cpu_task_count,
        "gpu_task_count": plan.gpu_task_count,
        "stage_counts": dict(plan.stage_counts),
        "max_cpu_parallel": plan.max_cpu_parallel,
        "max_gpu_parallel": plan.max_gpu_parallel,
        "parallel_inpaint_threads": parallel_inpaint_threads,
        "overlap_worker_count": overlap_worker_count,
        "ocr_serialized": True,
        "inpaint_lock_mode": inpaint_lock_mode,
        "validation_status": plan.validation.status,
        "validation_reasons": list(plan.validation.reasons),
        "notes": [
            "Experimental flag only: validate produced output against the sequential baseline.",
            "overlap_context_release keeps a single GPU lane and releases ordered context after translate.",
            "TRADUZAI_STRIP_PARALLEL_INPAINT_THREADS enables bounded inpaint overlap only in overlap_context_release.",
            "Use scheduler shadow gate against the produced output before considering a parallel executor.",
        ],
    }


def _band_image_label(source_page_number: int | None) -> str:
    try:
        number = int(source_page_number or 0)
    except Exception:
        return f"band_{source_page_number}"
    if number > 0:
        return f"band_{number:03d}"
    return f"band_{number}"


def _shift_bbox_xy(value, dx: int, dy: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [
            int(round(float(value[0]))) + dx,
            int(round(float(value[1]))) + dy,
            int(round(float(value[2]))) + dx,
            int(round(float(value[3]))) + dy,
        ]
    except Exception:
        return None


def _shift_bbox_list_xy(values, dx: int, dy: int) -> list[list[int]]:
    shifted: list[list[int]] = []
    for value in values or []:
        bbox = _shift_bbox_xy(value, dx, dy)
        if bbox is not None:
            shifted.append(bbox)
    return shifted


def _shift_polygons_xy(polygons, dx: int, dy: int):
    if not isinstance(polygons, list):
        return polygons
    if polygons and isinstance(polygons[0], (list, tuple)) and len(polygons[0]) >= 2 and not isinstance(polygons[0][0], (list, tuple)):
        return [
            [int(point[0]) + dx, int(point[1]) + dy]
            if isinstance(point, (list, tuple)) and len(point) >= 2
            else point
            for point in polygons
        ]
    shifted = []
    for polygon in polygons:
        if not isinstance(polygon, list):
            shifted.append(polygon)
            continue
        shifted_polygon = []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                shifted_polygon.append([int(point[0]) + dx, int(point[1]) + dy])
            else:
                shifted_polygon.append(point)
        shifted.append(shifted_polygon)
    return shifted


def _shift_text_geometry_xy(text: dict, dx: int, dy: int) -> dict:
    shifted = dict(text)
    for key in (
        "bbox",
        "source_bbox",
        "balloon_bbox",
        "text_pixel_bbox",
        "layout_bbox",
        "_visual_rect_outer_bbox",
        "_visual_rect_inner_bbox",
        "render_bbox",
        "safe_text_box",
        "_debug_safe_text_box",
        "layout_safe_bbox",
        "position_bbox",
        "capacity_bbox",
        "target_bbox",
        "sign_bbox",
        "_connected_source_bbox",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "balloon_inner_bbox",
    ):
        bbox = _shift_bbox_xy(shifted.get(key), dx, dy)
        if bbox is not None:
            shifted[key] = bbox
    for key in (
        "balloon_subregions",
        "connected_lobe_bboxes",
        "connected_text_groups",
        "connected_position_bboxes",
        "connected_focus_bboxes",
        "_merged_source_bboxes",
        "merged_source_bboxes",
        "_connected_source_anchor_bboxes",
    ):
        if key in shifted:
            shifted[key] = _shift_bbox_list_xy(shifted.get(key), dx, dy)
    for key in ("line_polygons", "connected_lobe_polygons", "balloon_polygon"):
        if key in shifted:
            shifted[key] = _shift_polygons_xy(shifted.get(key), dx, dy)
    for key in ("qa_metrics", "_render_debug", "_render_debug_candidates", "_render_debug_skipped", "connected_children"):
        if isinstance(shifted.get(key), dict):
            shifted[key] = _shift_nested_debug_bboxes_xy(shifted[key], dx, dy)
        elif isinstance(shifted.get(key), list):
            shifted[key] = _shift_nested_debug_bboxes_xy(shifted[key], dx, dy)
    return shifted


def _shift_nested_debug_bboxes_xy(value, dx: int, dy: int):
    if isinstance(value, dict):
        shifted = {}
        for key, item in value.items():
            if _looks_like_debug_bbox_key(str(key)):
                if str(key).endswith("bboxes"):
                    shifted[key] = _shift_bbox_list_xy(item, dx, dy)
                    continue
                bbox = _shift_bbox_xy(item, dx, dy)
                if bbox is not None:
                    shifted[key] = bbox
                    continue
            shifted[key] = _shift_nested_debug_bboxes_xy(item, dx, dy)
        return shifted
    if isinstance(value, list):
        return [_shift_nested_debug_bboxes_xy(item, dx, dy) for item in value]
    return value


def _bbox_center_y(value) -> float | None:
    bbox = _shift_bbox_xy(value, 0, 0)
    if bbox is None:
        return None
    return (bbox[1] + bbox[3]) / 2.0


def _bbox_overlaps_band(global_bbox: list[int] | None, band: Band) -> bool:
    if global_bbox is None:
        return False
    center_y = _bbox_center_y(global_bbox)
    if center_y is not None and band.y_top <= center_y < band.y_bottom:
        return True
    y1 = max(int(global_bbox[1]), int(band.y_top))
    y2 = min(int(global_bbox[3]), int(band.y_bottom))
    overlap = max(0, y2 - y1)
    height = max(1, int(global_bbox[3]) - int(global_bbox[1]))
    return (overlap / float(height)) >= 0.5


def _shift_block_geometry_xy(block: dict, dx: int, dy: int) -> dict:
    return _shift_text_geometry_xy(block, dx, dy)


def _split_koharu_page_result_into_bands(
    strip: VerticalStrip,
    *,
    page_number: int,
    page_result: dict,
    page_bands: list[tuple[int, Band]],
) -> dict[int, dict]:
    page_y0, _ = _source_page_bounds(strip, page_number)
    page_x_offsets = list(strip.page_x_offsets or [])
    page_x0 = int(page_x_offsets[page_number - 1]) if 0 <= page_number - 1 < len(page_x_offsets) else 0
    raw_texts = [text for text in list(page_result.get("texts") or []) if isinstance(text, dict)]
    raw_blocks = [block for block in list(page_result.get("_vision_blocks") or []) if isinstance(block, dict)]
    mapped: dict[int, dict] = {}

    for band_index, band in page_bands:
        if band.strip_slice is None:
            continue
        local_texts = []
        for text in raw_texts:
            global_bbox = _shift_bbox_xy(text.get("bbox"), page_x0, page_y0)
            if not _bbox_overlaps_band(global_bbox, band):
                continue
            local_texts.append(_shift_text_geometry_xy(text, page_x0, page_y0 - band.y_top))

        if not local_texts:
            continue

        local_blocks = []
        for block in raw_blocks:
            global_bbox = _shift_bbox_xy(block.get("bbox"), page_x0, page_y0)
            if _bbox_overlaps_band(global_bbox, band):
                local_blocks.append(_shift_block_geometry_xy(block, page_x0, page_y0 - band.y_top))

        if not local_blocks:
            for text in local_texts:
                bbox = _shift_bbox_xy(text.get("balloon_bbox") or text.get("bbox"), 0, 0)
                if bbox is not None:
                    local_blocks.append(
                        {
                            "bbox": bbox,
                            "confidence": float(text.get("confidence", text.get("ocr_confidence", 0.9)) or 0.9),
                            "detector": "koharu-text-fallback",
                        }
                    )

        height, width = band.strip_slice.shape[:2]
        mapped[band_index] = {
            "image": _band_image_label(page_number),
            "width": width,
            "height": height,
            "texts": local_texts,
            "_vision_blocks": local_blocks,
            "_vision_backend": page_result.get("_vision_backend", "koharu-http"),
            "_koharu_http": dict(page_result.get("_koharu_http") or {}),
            "_ocr_stats": {
                "koharu_cjk_precompute": True,
                "koharu_cjk_page_number": int(page_number),
                "koharu_cjk_text_count": len(local_texts),
                "koharu_cjk_block_count": len(local_blocks),
            },
        }

    return mapped


def _koharu_cjk_strip_precompute_enabled(idioma_origem: str, models_dir: str = "") -> bool:
    raw = os.getenv("TRADUZAI_KOHARU_CJK_STRIP_OCR", "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "disabled"}:
        return False
    try:
        from vision_stack.runtime import _should_use_koharu_cjk_ocr
        return bool(_should_use_koharu_cjk_ocr(idioma_origem, models_dir))
    except Exception:
        return False


def _koharu_cjk_strip_roi_enabled() -> bool:
    return _env_bool("TRADUZAI_KOHARU_CJK_STRIP_ROI", True)


def _koharu_cjk_selective_enabled() -> bool:
    return _env_bool("TRADUZAI_KOHARU_CJK_SELECTIVE", True)


def _koharu_cjk_roi_padding_px() -> int:
    return max(16, _env_int("TRADUZAI_KOHARU_CJK_ROI_PAD_PX", 96))


def _koharu_cjk_empty_roi_filter_enabled() -> bool:
    return _env_bool("TRADUZAI_KOHARU_CJK_EMPTY_ROI_FILTER", True)


def _koharu_cjk_page_fallback_enabled() -> bool:
    return _env_bool("TRADUZAI_KOHARU_CJK_PAGE_FALLBACK", True)


def _koharu_cjk_page_fallback_max() -> int:
    return max(0, _env_int("TRADUZAI_KOHARU_CJK_PAGE_FALLBACK_MAX", 12))


def _koharu_cjk_ocr_only_enabled() -> bool:
    return _env_bool("TRADUZAI_KOHARU_WORKER_OCR_ONLY", True)


def _koharu_known_bboxes_for_roi(band: Band, crop_bbox: list[int]) -> list[list[int]]:
    crop_x1, crop_y1, crop_x2, crop_y2 = [int(v) for v in crop_bbox]
    known: list[list[int]] = []
    for balloon in band.balloons:
        bbox = [
            int(balloon.strip_bbox.x1) - crop_x1,
            int(balloon.strip_bbox.y1) - crop_y1,
            int(balloon.strip_bbox.x2) - crop_x1,
            int(balloon.strip_bbox.y2) - crop_y1,
        ]
        bbox[0] = max(0, min(crop_x2 - crop_x1, bbox[0]))
        bbox[2] = max(0, min(crop_x2 - crop_x1, bbox[2]))
        bbox[1] = max(0, min(crop_y2 - crop_y1, bbox[1]))
        bbox[3] = max(0, min(crop_y2 - crop_y1, bbox[3]))
        if bbox[2] - bbox[0] < 8 or bbox[3] - bbox[1] < 8:
            continue
        if bbox not in known:
            known.append(bbox)
    return known


def _koharu_roi_has_textlike_content(crop_image) -> tuple[bool, str]:
    try:
        from vision_stack.runtime import _quick_text_presence_details

        return _quick_text_presence_details(crop_image)
    except Exception:
        return True, "unavailable"


def _merge_koharu_worker_batch_telemetry(stats: dict, page_results: list[dict]) -> None:
    for page_result in page_results:
        if not isinstance(page_result, dict):
            continue
        worker_batch = page_result.get("_koharu_worker_batch")
        if not isinstance(worker_batch, dict):
            continue
        summary = dict(stats.get("worker_batch") or {})
        summary["persistent"] = bool(worker_batch.get("persistent"))
        summary["job_count"] = int(worker_batch.get("job_count") or summary.get("job_count") or 0)
        summary["ocr_only_job_count"] = int(
            worker_batch.get("ocr_only_job_count") or summary.get("ocr_only_job_count") or 0
        )
        if worker_batch.get("worker_wall_ms") is not None:
            summary["worker_wall_ms"] = int(worker_batch.get("worker_wall_ms") or 0)
        if worker_batch.get("worker_json_parse_ms") is not None:
            summary["worker_json_parse_ms"] = int(worker_batch.get("worker_json_parse_ms") or 0)
        if worker_batch.get("request_write_ms") is not None:
            summary["request_write_ms"] = int(worker_batch.get("request_write_ms") or 0)
        if worker_batch.get("persistent_error"):
            summary["persistent_error"] = str(worker_batch.get("persistent_error"))[:240]
        if isinstance(worker_batch.get("batch_timings_ms"), dict):
            summary["batch_timings_ms"] = dict(worker_batch.get("batch_timings_ms") or {})
        token_values = worker_batch.get("max_new_tokens")
        if isinstance(token_values, list):
            summary["max_new_tokens"] = [int(v or 0) for v in token_values]
        stats["worker_batch"] = summary
        return


def _text_has_alnum_or_cjk(text: str) -> bool:
    for ch in text:
        if ch.isalnum():
            return True
        code = ord(ch)
        if (
            0x3040 <= code <= 0x30FF
            or 0x3400 <= code <= 0x9FFF
            or 0xAC00 <= code <= 0xD7AF
            or 0x1100 <= code <= 0x11FF
            or 0x3130 <= code <= 0x318F
        ):
            return True
    return False


def _text_requires_final_cleanup(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    translated = str(text.get("translated") or text.get("traduzido") or "").strip()
    if not translated:
        return False
    original = str(text.get("original") or text.get("text") or "").strip()
    if original and translated == original:
        return False
    return True


def _cleanup_text_geometry_mask(text: dict, shape: tuple[int, int]) -> np.ndarray | None:
    try:
        from inpainter.mask_builder import mask_from_text_geometry

        mask = mask_from_text_geometry(text, shape)
        if mask is not None and np.any(mask):
            return mask.astype(np.uint8)
    except Exception:
        pass

    bbox = text.get("text_pixel_bbox") or text.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    h, w = shape
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except Exception:
        return None
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _cleanup_reintroduces_text_residual(
    clean_image: np.ndarray,
    fixed_clean: np.ndarray,
    texts: list[dict],
) -> bool:
    if clean_image is None or fixed_clean is None or clean_image.shape != fixed_clean.shape or clean_image.size == 0:
        return False
    clean_gray = cv2.cvtColor(clean_image, cv2.COLOR_RGB2GRAY)
    fixed_gray = cv2.cvtColor(fixed_clean, cv2.COLOR_RGB2GRAY)
    shape = clean_gray.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for text in texts:
        if not isinstance(text, dict):
            continue
        mask = _cleanup_text_geometry_mask(text, shape)
        if mask is None or not np.any(mask):
            continue
        mask = cv2.dilate(mask, kernel, iterations=1)
        clean_dark = int(np.count_nonzero((clean_gray < 180) & (mask > 0)))
        fixed_dark = int(np.count_nonzero((fixed_gray < 180) & (mask > 0)))
        if fixed_dark >= max(clean_dark + 24, int(clean_dark * 1.45) + 12):
            return True
    return False


def _page_cleanup_rerender_margin_px() -> int:
    return max(0, _env_int("TRADUZAI_PAGE_CLEANUP_RERENDER_MARGIN", 48))


def _page_cleanup_rerender_max_crop_ratio() -> float:
    return max(0.05, min(1.0, _env_float("TRADUZAI_PAGE_CLEANUP_RERENDER_MAX_CROP_RATIO", 0.70)))


def _page_cleanup_rerender_max_total_ratio() -> float:
    return max(0.05, min(2.0, _env_float("TRADUZAI_PAGE_CLEANUP_RERENDER_MAX_TOTAL_RATIO", 0.95)))


def _page_cleanup_rerender_max_crops() -> int:
    return max(1, _env_int("TRADUZAI_PAGE_CLEANUP_RERENDER_MAX_CROPS", 12))


def _page_cleanup_background_delta_enabled() -> bool:
    return _env_bool("TRADUZAI_PAGE_CLEANUP_BACKGROUND_DELTA", True)


def _bbox_area(bbox: list[int] | None) -> int:
    if bbox is None:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _bbox_union(bboxes: list[list[int] | None]) -> list[int] | None:
    valid = [bbox for bbox in bboxes if bbox is not None and _bbox_area(bbox) > 0]
    if not valid:
        return None
    return [
        min(int(bbox[0]) for bbox in valid),
        min(int(bbox[1]) for bbox in valid),
        max(int(bbox[2]) for bbox in valid),
        max(int(bbox[3]) for bbox in valid),
    ]


def _clip_bbox_to_shape(bbox: list[int] | None, shape: tuple[int, int]) -> list[int] | None:
    bbox = _bbox4_or_none(bbox)
    if bbox is None:
        return None
    h, w = int(shape[0]), int(shape[1])
    x1 = max(0, min(w, int(bbox[0])))
    y1 = max(0, min(h, int(bbox[1])))
    x2 = max(0, min(w, int(bbox[2])))
    y2 = max(0, min(h, int(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _expand_bbox_to_shape(bbox: list[int] | None, shape: tuple[int, int], margin: int) -> list[int] | None:
    bbox = _bbox4_or_none(bbox)
    if bbox is None:
        return None
    return _clip_bbox_to_shape(
        [
            int(bbox[0]) - int(margin),
            int(bbox[1]) - int(margin),
            int(bbox[2]) + int(margin),
            int(bbox[3]) + int(margin),
        ],
        shape,
    )


def _changed_pixel_component_bboxes(clean_image: np.ndarray, fixed_clean: np.ndarray) -> list[list[int]]:
    if clean_image is None or fixed_clean is None or clean_image.shape != fixed_clean.shape or clean_image.size == 0:
        return []
    diff = cv2.absdiff(clean_image, fixed_clean)
    if diff.ndim == 3:
        mask = np.any(diff > 0, axis=2).astype(np.uint8)
    else:
        mask = (diff > 0).astype(np.uint8)
    if not np.any(mask):
        return []

    try:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    except Exception:
        ys, xs = np.where(mask > 0)
        return [[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]]

    components: list[list[int]] = []
    for idx in range(1, int(count)):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if w > 0 and h > 0:
            components.append([x, y, x + w, y + h])
    if len(components) > max(1, _env_int("TRADUZAI_PAGE_CLEANUP_MAX_COMPONENTS", 128)):
        union = _bbox_union(components)
        return [union] if union is not None else []
    return components


def _page_cleanup_text_bboxes(text: dict) -> list[list[int]]:
    if not isinstance(text, dict):
        return []
    bboxes: list[list[int]] = []
    for key in (
        "render_bbox",
        "safe_text_box",
        "_debug_safe_text_box",
        "position_bbox",
        "capacity_bbox",
        "target_bbox",
        "balloon_bbox",
        "layout_bbox",
        "text_pixel_bbox",
        "source_bbox",
        "bbox",
        "sign_bbox",
        "_connected_source_bbox",
    ):
        bbox = _shift_bbox_xy(text.get(key), 0, 0)
        if bbox is not None:
            bboxes.append(bbox)
    for key in (
        "balloon_subregions",
        "connected_lobe_bboxes",
        "connected_text_groups",
        "connected_position_bboxes",
        "connected_focus_bboxes",
        "_merged_source_bboxes",
        "merged_source_bboxes",
        "_connected_source_anchor_bboxes",
    ):
        bboxes.extend(_shift_bbox_list_xy(text.get(key), 0, 0))
    return bboxes


def _dedupe_texts_by_identity(texts: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[int] = set()
    for text in texts:
        marker = id(text)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(text)
    return unique


def _merge_page_cleanup_jobs(jobs: list[dict], shape: tuple[int, int]) -> list[dict]:
    merged: list[dict] = []
    merge_margin = max(0, _env_int("TRADUZAI_PAGE_CLEANUP_RERENDER_MERGE_MARGIN", 8))
    for job in jobs:
        bbox = _clip_bbox_to_shape(job.get("bbox"), shape)
        if bbox is None:
            continue
        pending = {"bbox": bbox, "texts": _dedupe_texts_by_identity(list(job.get("texts") or []))}
        did_merge = True
        while did_merge:
            did_merge = False
            for idx, existing in enumerate(list(merged)):
                expanded = _expand_bbox_to_shape(pending["bbox"], shape, merge_margin) or pending["bbox"]
                if not _bbox_intersects(expanded, existing.get("bbox")):
                    continue
                union = _bbox_union([pending["bbox"], existing.get("bbox")])
                if union is None:
                    continue
                pending["bbox"] = _clip_bbox_to_shape(union, shape) or union
                pending["texts"] = _dedupe_texts_by_identity(
                    list(pending.get("texts") or []) + list(existing.get("texts") or [])
                )
                del merged[idx]
                did_merge = True
                break
        merged.append(pending)
    return sorted(merged, key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))


def _assign_page_cleanup_texts_to_jobs(jobs: list[dict], page_texts: list[dict]) -> list[dict]:
    assigned: list[dict] = []
    for job in jobs:
        bbox = job.get("bbox")
        texts: list[dict] = []
        for text in page_texts:
            text_boxes = _page_cleanup_text_bboxes(text)
            if text_boxes and any(_bbox_intersects(box, bbox) for box in text_boxes):
                texts.append(text)
        updated = dict(job)
        updated["texts"] = _dedupe_texts_by_identity(texts)
        assigned.append(updated)
    return assigned


def _page_cleanup_crop_jobs(
    clean_image: np.ndarray,
    fixed_clean: np.ndarray,
    page_texts: list[dict],
) -> list[dict] | None:
    shape = fixed_clean.shape[:2]
    components = _changed_pixel_component_bboxes(clean_image, fixed_clean)
    if not components:
        return []

    margin = _page_cleanup_rerender_margin_px()
    jobs: list[dict] = []
    for component in components:
        selection_bbox = _expand_bbox_to_shape(component, shape, margin)
        if selection_bbox is None:
            continue
        affected_texts: list[dict] = []
        affected_boxes: list[list[int]] = []
        for text in page_texts:
            text_boxes = _page_cleanup_text_bboxes(text)
            if not text_boxes:
                continue
            if any(_bbox_intersects(box, selection_bbox) for box in text_boxes):
                affected_texts.append(text)
                affected_boxes.extend(text_boxes)
        crop_bbox = _bbox_union([component] + affected_boxes) or component
        crop_bbox = _expand_bbox_to_shape(crop_bbox, shape, margin)
        if crop_bbox is not None:
            jobs.append({"bbox": crop_bbox, "texts": affected_texts})

    jobs = _merge_page_cleanup_jobs(jobs, shape)
    jobs = _assign_page_cleanup_texts_to_jobs(jobs, page_texts)
    if len(jobs) > _page_cleanup_rerender_max_crops():
        return None

    page_area = max(1, int(shape[0]) * int(shape[1]))
    max_crop_ratio = _page_cleanup_rerender_max_crop_ratio()
    max_total_ratio = _page_cleanup_rerender_max_total_ratio()
    total_area = 0
    for job in jobs:
        area = _bbox_area(job.get("bbox"))
        total_area += area
        if area / float(page_area) > max_crop_ratio:
            return None
    if total_area / float(page_area) > max_total_ratio:
        return None
    return jobs


def _render_page_cleanup_regions(
    *,
    clean_image: np.ndarray,
    fixed_clean: np.ndarray,
    rendered_image: np.ndarray,
    page_texts: list[dict],
    typesetter,
) -> tuple[np.ndarray, str]:
    if _page_cleanup_background_delta_enabled():
        delta_rendered = _apply_page_cleanup_background_delta(
            clean_image=clean_image,
            fixed_clean=fixed_clean,
            rendered_image=rendered_image,
        )
        if delta_rendered is not None:
            return delta_rendered, "delta"

    jobs = _page_cleanup_crop_jobs(clean_image, fixed_clean, page_texts)
    if jobs is None:
        return (
            typesetter.render_band_image(
                fixed_clean,
                _render_payload_without_legacy_decision_fields(page_texts, coordinate_space="page"),
            ),
            "full",
        )
    if not jobs:
        return rendered_image.copy(), "noop"
    if not isinstance(rendered_image, np.ndarray) or rendered_image.shape != fixed_clean.shape:
        return (
            typesetter.render_band_image(
                fixed_clean,
                _render_payload_without_legacy_decision_fields(page_texts, coordinate_space="page"),
            ),
            "full",
        )

    result = rendered_image.copy()
    for job in jobs:
        bbox = _clip_bbox_to_shape(job.get("bbox"), fixed_clean.shape[:2])
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop_clean = fixed_clean[y1:y2, x1:x2].copy()
        crop_texts = [_shift_text_geometry_xy(copy.deepcopy(text), -x1, -y1) for text in list(job.get("texts") or [])]
        if crop_texts:
            crop_rendered = typesetter.render_band_image(
                crop_clean,
                {
                    "texts": _texts_without_legacy_decision_fields(crop_texts),
                    "_coordinate_space": "page_cleanup_crop",
                    "_page_cleanup_crop_bbox": [x1, y1, x2, y2],
                },
            )
        else:
            crop_rendered = crop_clean
        if not isinstance(crop_rendered, np.ndarray) or crop_rendered.shape[:2] != crop_clean.shape[:2]:
            return (
                typesetter.render_band_image(
                    fixed_clean,
                    _render_payload_without_legacy_decision_fields(page_texts, coordinate_space="page"),
                ),
                "full",
            )
        result[y1:y2, x1:x2] = crop_rendered[: y2 - y1, : x2 - x1]
    return result, "roi"


def _apply_page_cleanup_background_delta(
    *,
    clean_image: np.ndarray,
    fixed_clean: np.ndarray,
    rendered_image: np.ndarray,
) -> np.ndarray | None:
    if (
        not isinstance(clean_image, np.ndarray)
        or not isinstance(fixed_clean, np.ndarray)
        or not isinstance(rendered_image, np.ndarray)
        or clean_image.shape != fixed_clean.shape
        or clean_image.shape != rendered_image.shape
        or clean_image.size == 0
    ):
        return None

    cleanup_diff = cv2.absdiff(clean_image, fixed_clean)
    if cleanup_diff.ndim == 3:
        cleanup_mask = np.any(cleanup_diff > 0, axis=2).astype(np.uint8)
    else:
        cleanup_mask = (cleanup_diff > 0).astype(np.uint8)
    if not np.any(cleanup_mask):
        return rendered_image.copy()

    render_diff = cv2.absdiff(rendered_image, clean_image)
    if render_diff.ndim == 3:
        rendered_text_mask = np.any(render_diff > 8, axis=2).astype(np.uint8)
    else:
        rendered_text_mask = (render_diff > 8).astype(np.uint8)
    if np.any(rendered_text_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rendered_text_mask = cv2.dilate(rendered_text_mask, kernel, iterations=1)

    copy_mask = (cleanup_mask > 0) & (rendered_text_mask == 0)
    result = rendered_image.copy()
    if np.any(copy_mask):
        result[copy_mask] = fixed_clean[copy_mask]
    return result


def _restore_rendered_text_regions_after_page_clamp(
    rendered_image: np.ndarray,
    clamped_rendered: np.ndarray,
    page_texts: list[dict] | None,
) -> np.ndarray:
    if (
        not isinstance(rendered_image, np.ndarray)
        or not isinstance(clamped_rendered, np.ndarray)
        or rendered_image.shape != clamped_rendered.shape
        or rendered_image.size == 0
    ):
        return clamped_rendered
    height, width = rendered_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for text in page_texts or []:
        if not isinstance(text, dict):
            continue
        source = str(text.get("bubble_mask_source") or text.get("balloon_mask_source") or "").strip().lower()
        if source in {"image_dark_bubble_mask", "image_dark_panel_mask", "derived_card_panel_mask"}:
            for key in ("bubble_mask_bbox", "balloon_bbox"):
                bbox = _shift_bbox_y(text.get(key), 0)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(width, int(x1) + 1))
                x2 = max(0, min(width, int(x2) - 1))
                y1 = max(0, min(height, int(y1) + 1))
                y2 = max(0, min(height, int(y2) - 1))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255
                    break
        for key in (
            "render_bbox",
            "_debug_render_bbox",
            "safe_text_box",
            "_debug_safe_text_box",
            "target_bbox",
            "position_bbox",
            "text_pixel_bbox",
            "bbox",
        ):
            bbox = _shift_bbox_y(text.get(key), 0)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            bw = max(1, int(x2) - int(x1))
            bh = max(1, int(y2) - int(y1))
            pad = max(2, min(10, int(round(max(bw, bh) * 0.04))))
            x1 = max(0, min(width, int(x1) - pad))
            x2 = max(0, min(width, int(x2) + pad))
            y1 = max(0, min(height, int(y1) - pad))
            y2 = max(0, min(height, int(y2) + pad))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
            break
    if not np.any(mask):
        return clamped_rendered
    result = clamped_rendered.copy()
    result[mask > 0] = rendered_image[mask > 0]
    return result


def _cleanup_page_inpaint_and_rerender(
    *,
    original_image,
    clean_image,
    page_texts: list[dict],
    rendered_image,
    typesetter,
    breakdown: dict[str, float] | None = None,
) -> tuple[object, object, bool]:
    cleanup_candidates = [
        _without_legacy_decision_fields(text)
        for text in page_texts
        if _text_requires_final_cleanup(text)
    ]
    if not cleanup_candidates:
        return clean_image, rendered_image, False

    def _add_breakdown(stage: str, started_at: float) -> None:
        if breakdown is None:
            return
        breakdown[stage] = float(breakdown.get(stage, 0.0) or 0.0) + (time.perf_counter() - started_at)

    try:
        from vision_stack.runtime import (
            _apply_white_balloon_near_text_residual_cleanup,
            _apply_post_inpaint_cleanup_timed,
            _has_white_balloon_text_residual,
            _white_cleanup_texts,
        )

        cleanup_texts = _white_cleanup_texts(original_image, cleanup_candidates)
        if not cleanup_texts:
            return clean_image, rendered_image, False

        near_text_cleanup_enabled = _page_final_near_text_cleanup_enabled()
        full_cleanup_enabled = _env_bool("TRADUZAI_PAGE_FINAL_FULL_CLEANUP", True)
        if not near_text_cleanup_enabled and not full_cleanup_enabled:
            return clean_image, rendered_image, False

        fixed_clean = clean_image
        if near_text_cleanup_enabled:
            cleanup_stage_started = time.perf_counter()
            fixed_clean = _apply_white_balloon_near_text_residual_cleanup(original_image, fixed_clean, cleanup_texts)
            _add_breakdown("cleanup_inpaint", cleanup_stage_started)
        if full_cleanup_enabled and _has_white_balloon_text_residual(
            original_image,
            fixed_clean,
            cleanup_texts,
        ):
            limit_mask = _build_page_cleanup_limit_mask(original_image, cleanup_texts)
            cleanup_stage_started = time.perf_counter()
            fixed_clean, _stats = _apply_post_inpaint_cleanup_timed(
                original_image,
                fixed_clean,
                cleanup_texts,
                limit_mask=limit_mask,
            )
            _add_breakdown("cleanup_inpaint", cleanup_stage_started)
        if _cleanup_reintroduces_text_residual(clean_image, fixed_clean, cleanup_candidates):
            return clean_image, rendered_image, False
        if not np.any(cv2.absdiff(clean_image, fixed_clean)):
            return clean_image, rendered_image, False
        cleanup_stage_started = time.perf_counter()
        fixed_rendered, render_mode = _render_page_cleanup_regions(
            clean_image=clean_image,
            fixed_clean=fixed_clean,
            rendered_image=rendered_image,
            page_texts=page_texts,
            typesetter=typesetter,
        )
        _add_breakdown("cleanup_typeset", cleanup_stage_started)
        if breakdown is not None:
            breakdown[f"cleanup_{render_mode}_rerender_count"] = float(
                breakdown.get(f"cleanup_{render_mode}_rerender_count", 0.0) or 0.0
            ) + 1.0
        return fixed_clean, fixed_rendered, True
    except Exception:
        return clean_image, rendered_image, False


def _dark_visual_text_cleanup_mask(shape: tuple[int, int], text: dict) -> np.ndarray | None:
    height, width = shape
    if height <= 0 or width <= 0 or not isinstance(text, dict):
        return None
    source = str(text.get("bubble_mask_source") or text.get("balloon_mask_source") or "").strip().lower()
    dark_sources = {"image_dark_bubble_mask", "image_dark_panel_mask", "derived_card_panel_mask"}
    rejected_visual_sources = {"derived_white_crop_rejected", "rejected_derived_bubble_mask"}
    colors = text.get("dark_panel_effect_colors") if isinstance(text.get("dark_panel_effect_colors"), dict) else {}
    has_visual_colors = isinstance(colors, dict) and isinstance(colors.get("panel_fill_rgb"), (list, tuple))
    if source not in dark_sources and not (source in rejected_visual_sources and has_visual_colors):
        return None
    mask = np.zeros((height, width), dtype=np.uint8)

    def _mark_polygon(points) -> None:
        if not isinstance(points, (list, tuple)) or len(points) < 3:
            return
        clipped: list[list[int]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return
            try:
                x = max(0, min(width - 1, int(round(float(point[0])))))
                y = max(0, min(height - 1, int(round(float(point[1])))))
            except Exception:
                return
            clipped.append([x, y])
        if len(clipped) >= 3:
            cv2.fillPoly(mask, [np.asarray(clipped, dtype=np.int32)], 255)

    polygons = text.get("line_polygons")
    if isinstance(polygons, (list, tuple)) and polygons:
        first = polygons[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2 and not (
            first and isinstance(first[0], (list, tuple))
        ):
            _mark_polygon(polygons)
        else:
            for polygon in polygons:
                _mark_polygon(polygon)
    bbox = _shift_bbox_y(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"), 0)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        pad = 18 if source == "image_dark_bubble_mask" else (16 if source in rejected_visual_sources else 12)
        x1 = max(0, min(width, int(x1) - pad))
        x2 = max(0, min(width, int(x2) + pad))
        y1 = max(0, min(height, int(y1) - pad))
        y2 = max(0, min(height, int(y2) + pad))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    if not np.any(mask):
        return None
    kernel_size = 15 if source == "image_dark_bubble_mask" else (13 if source in rejected_visual_sources else 11)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=1)
    clip_source = (text.get("bubble_mask_bbox") or text.get("balloon_bbox")) if source in dark_sources else text.get("balloon_bbox")
    clip_bbox = _shift_bbox_y(clip_source, 0)
    if clip_bbox is not None:
        clip = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = clip_bbox
        x1 = max(0, min(width, int(x1)))
        x2 = max(0, min(width, int(x2)))
        y1 = max(0, min(height, int(y1)))
        y2 = max(0, min(height, int(y2)))
        if x2 > x1 and y2 > y1:
            clip[y1:y2, x1:x2] = 255
            mask = np.where(clip > 0, mask, 0).astype(np.uint8)
    return mask if np.any(mask) else None


def _dark_visual_cleanup_bubble_bbox_overbroad(text: dict, text_bbox: list[int] | None, bubble_bbox: list[int] | None) -> bool:
    if not isinstance(text, dict) or text_bbox is None or bubble_bbox is None:
        return False
    source = str(text.get("bubble_mask_source") or text.get("balloon_mask_source") or "").strip().lower()
    if source not in {"image_dark_bubble_mask", "image_dark_panel_mask", "derived_card_panel_mask"}:
        return False
    text_area = _bbox_area(text_bbox)
    bubble_area = _bbox_area(bubble_bbox)
    if text_area <= 0 or bubble_area <= 0:
        return False
    tw = max(1, int(text_bbox[2]) - int(text_bbox[0]))
    th = max(1, int(text_bbox[3]) - int(text_bbox[1]))
    bw = max(1, int(bubble_bbox[2]) - int(bubble_bbox[0]))
    bh = max(1, int(bubble_bbox[3]) - int(bubble_bbox[1]))
    overflow = max(
        int(text_bbox[0]) - int(bubble_bbox[0]),
        int(bubble_bbox[2]) - int(text_bbox[2]),
        int(text_bbox[1]) - int(bubble_bbox[1]),
        int(bubble_bbox[3]) - int(text_bbox[3]),
    )
    area_ratio = bubble_area / float(text_area)
    width_ratio = bw / float(tw)
    height_ratio = bh / float(th)
    return (
        overflow >= 48
        and area_ratio >= 4.0
        and (width_ratio >= 2.25 or height_ratio >= 2.25)
    )


def _apply_dark_visual_text_geometry_cleanup(clean_image, page_texts: list[dict]) -> tuple[object, int]:
    if not isinstance(clean_image, np.ndarray) or clean_image.ndim != 3:
        return clean_image, 0
    result = clean_image.copy()
    changed_count = 0
    height, width = result.shape[:2]
    for text in page_texts or []:
        if not isinstance(text, dict):
            continue
        mask = _dark_visual_text_cleanup_mask((height, width), text)
        if mask is None or not np.any(mask):
            continue
        text_bbox = _shift_bbox_y(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"), 0)
        bubble_bbox = _shift_bbox_y(text.get("bubble_mask_bbox") or text.get("balloon_bbox"), 0)
        overbroad_bubble_bbox = _dark_visual_cleanup_bubble_bbox_overbroad(text, text_bbox, bubble_bbox)
        if text_bbox is not None and bubble_bbox is not None and not overbroad_bubble_bbox:
            tx1, ty1, tx2, ty2 = text_bbox
            bx1, by1, bx2, by2 = bubble_bbox
            wx1 = max(0, min(width, min(int(tx1), int(bx1)) - 48))
            wx2 = max(0, min(width, max(int(tx2), int(bx2)) + 48))
            wy1 = max(0, min(height, int(ty1) - 48))
            wy2 = max(0, min(height, min(int(by2), int(ty2) + 72)))
            bx1 = max(0, min(width, int(bx1) + 6))
            bx2 = max(0, min(width, int(bx2) - 6))
            by1 = max(0, min(height, int(by1) + 6))
            by2 = max(0, min(height, int(by2) - 6))
            if wx2 > wx1 and wy2 > wy1 and bx2 > bx1 and by2 > by1:
                window = np.zeros((height, width), dtype=np.uint8)
                window[max(wy1, by1) : min(wy2, by2), max(wx1, bx1) : min(wx2, bx2)] = 255
                rgb = result.astype(np.float32)
                luma = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
                bright = np.where((window > 0) & (luma >= 96.0), 255, 0).astype(np.uint8)
                if np.any(bright):
                    bright = cv2.dilate(
                        bright,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                        iterations=1,
                    )
                    mask = np.maximum(mask, bright).astype(np.uint8)
        colors = text.get("dark_panel_effect_colors") if isinstance(text.get("dark_panel_effect_colors"), dict) else {}
        source = str(text.get("bubble_mask_source") or text.get("balloon_mask_source") or "").strip().lower()
        if source in {"derived_white_crop_rejected", "rejected_derived_bubble_mask"}:
            before = result.copy()
            result = cv2.inpaint(result, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
            changed_count += int(np.count_nonzero(np.any(result != before, axis=2)))
            continue
        fill = colors.get("panel_fill_rgb") if isinstance(colors, dict) else None
        if not isinstance(fill, (list, tuple)) or len(fill) < 3:
            fill = [0, 0, 0]
        fill_color = np.asarray(
            [max(0, min(255, int(round(float(channel))))) for channel in fill[:3]],
            dtype=np.uint8,
        )
        before = result.copy()
        result[mask > 0] = fill_color
        changed_count += int(np.count_nonzero(np.any(result != before, axis=2)))
    return result, changed_count


def _cleanup_dark_panel_page_and_rerender(
    *,
    clean_image,
    rendered_image,
    page_texts: list[dict],
    typesetter,
    breakdown: dict[str, float] | None = None,
) -> tuple[object, object, bool]:
    if not page_texts or clean_image is None or not isinstance(clean_image, np.ndarray):
        return clean_image, rendered_image, False
    try:
        from inpainter import _apply_dark_panel_text_fills

        stage_texts = _texts_without_legacy_decision_fields(page_texts)
        started = time.perf_counter()
        fixed_clean, count = _apply_dark_panel_text_fills(clean_image, {"texts": stage_texts})
        if breakdown is not None:
            breakdown["cleanup_inpaint"] = float(breakdown.get("cleanup_inpaint", 0.0) or 0.0) + (
                time.perf_counter() - started
            )
        geometry_clean, geometry_count = _apply_dark_visual_text_geometry_cleanup(fixed_clean, stage_texts)
        if geometry_count:
            fixed_clean = geometry_clean
            count = int(count or 0) + int(geometry_count)
            if breakdown is not None:
                breakdown["cleanup_dark_visual_geometry_pixels"] = float(
                    breakdown.get("cleanup_dark_visual_geometry_pixels", 0.0) or 0.0
                ) + float(geometry_count)
        if not count or not isinstance(fixed_clean, np.ndarray) or not np.any(cv2.absdiff(clean_image, fixed_clean)):
            return clean_image, rendered_image, False
        started = time.perf_counter()
        fixed_rendered = typesetter.render_band_image(
            fixed_clean,
            _render_payload_without_legacy_decision_fields(
                page_texts,
                coordinate_space="page_dark_panel_cleanup",
            ),
        )
        if breakdown is not None:
            breakdown["cleanup_typeset"] = float(breakdown.get("cleanup_typeset", 0.0) or 0.0) + (
                time.perf_counter() - started
            )
            breakdown["cleanup_dark_panel_rerender_count"] = float(
                breakdown.get("cleanup_dark_panel_rerender_count", 0.0) or 0.0
            ) + 1.0
        return fixed_clean, fixed_rendered, True
    except Exception:
        return clean_image, rendered_image, False


def _page_texts_from_text_layers(text_layers) -> list[dict]:
    if isinstance(text_layers, dict):
        return [text for text in list(text_layers.get("texts") or []) if isinstance(text, dict)]
    if isinstance(text_layers, list):
        return [text for text in list(text_layers) if isinstance(text, dict)]
    return []


def _page_requires_page_space_typeset(page_texts: list[dict]) -> bool:
    for text in page_texts or []:
        if not isinstance(text, dict):
            continue
        translated = str(text.get("translated") or text.get("traduzido") or "").strip()
        if not translated:
            continue
        if _shift_bbox_list_xy(text.get("balloon_subregions"), 0, 0) or _shift_bbox_list_xy(
            text.get("connected_lobe_bboxes"),
            0,
            0,
        ):
            return True
        bbox = _shift_bbox_xy(
            text.get("balloon_bbox") or text.get("layout_bbox") or text.get("bbox"),
            0,
            0,
        )
        if bbox is None:
            continue
        width = max(1, int(bbox[2]) - int(bbox[0]))
        height = max(1, int(bbox[3]) - int(bbox[1]))
        if width >= 140 and height >= 55 and width / float(height) >= 1.45:
            return True
    return False


def _build_page_cleanup_limit_mask(image_rgb, texts: list[dict]) -> np.ndarray | None:
    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None
    shape = image_rgb.shape[:2]
    limit = np.zeros(shape, dtype=np.uint8)
    try:
        from inpainter.mask_builder import build_inpaint_mask, mask_from_text_geometry
    except Exception:
        build_inpaint_mask = None
        mask_from_text_geometry = None

    for text in texts:
        if not isinstance(text, dict):
            continue
        mask = None
        if build_inpaint_mask is not None:
            try:
                mask = build_inpaint_mask(text, image_rgb.shape, image_rgb=image_rgb)
            except Exception:
                mask = None
        if (mask is None or not np.any(mask)) and mask_from_text_geometry is not None:
            try:
                mask = mask_from_text_geometry(text, image_rgb.shape)
            except Exception:
                mask = None
        if isinstance(mask, np.ndarray) and mask.shape[:2] == shape and np.any(mask):
            limit = np.maximum(limit, (mask > 0).astype(np.uint8) * 255)
            continue

        bbox = _shift_bbox_y(text.get("text_pixel_bbox") or text.get("bbox"), 0)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        pad = max(8, int(round(max(x2 - x1, y2 - y1) * 0.12)))
        x1 -= pad
        x2 += pad
        y1 -= pad
        y2 += pad
        x1 = max(0, min(shape[1], x1))
        x2 = max(0, min(shape[1], x2))
        y1 = max(0, min(shape[0], y1))
        y2 = max(0, min(shape[0], y2))
        if x2 > x1 and y2 > y1:
            limit[y1:y2, x1:x2] = 255

    return limit if np.any(limit) else None


def _build_page_inpaint_limit_mask(
    image_rgb,
    inpaint_blocks: list[dict] | None,
    page_texts: list[dict] | None,
) -> np.ndarray | None:
    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None
    shape = image_rgb.shape[:2]
    height, width = shape
    records = [dict(block) for block in list(inpaint_blocks or []) if isinstance(block, dict)]
    if not records:
        records = [dict(text) for text in list(page_texts or []) if isinstance(text, dict)]
    if not records:
        return None

    limit = np.zeros(shape, dtype=np.uint8)

    def _clip_point(x, y) -> list[int] | None:
        try:
            px = int(round(float(x)))
            py = int(round(float(y)))
        except Exception:
            return None
        return [max(0, min(width - 1, px)), max(0, min(height - 1, py))]

    def _mark_polygon(points) -> bool:
        if not isinstance(points, (list, tuple)) or len(points) < 3:
            return False
        clipped = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return False
            clipped_point = _clip_point(point[0], point[1])
            if clipped_point is None:
                return False
            clipped.append(clipped_point)
        if len(clipped) < 3:
            return False
        cv2.fillPoly(limit, [np.asarray(clipped, dtype=np.int32)], 255)
        return True

    def _mark_polygons(value) -> bool:
        if not isinstance(value, (list, tuple)) or not value:
            return False
        first = value[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2 and not (
            first and isinstance(first[0], (list, tuple))
        ):
            return _mark_polygon(value)
        marked = False
        for item in value:
            marked = _mark_polygon(item) or marked
        return marked

    def _mark_bbox(value) -> bool:
        bbox = _shift_bbox_y(value, 0)
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return False
        limit[y1:y2, x1:x2] = 255
        return True

    def _bbox_from_polygons(value) -> list[int] | None:
        if not isinstance(value, (list, tuple)) or not value:
            return None
        polygons = value
        first = value[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2 and not (
            first and isinstance(first[0], (list, tuple))
        ):
            polygons = [value]
        xs: list[int] = []
        ys: list[int] = []
        for polygon in polygons:
            if not isinstance(polygon, (list, tuple)):
                continue
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    xs.append(int(round(float(point[0]))))
                    ys.append(int(round(float(point[1]))))
                except Exception:
                    continue
        if not xs or not ys:
            return None
        return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]

    def _polygon_count(value) -> int:
        if not isinstance(value, (list, tuple)) or not value:
            return 0
        first = value[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2 and not (
            first and isinstance(first[0], (list, tuple))
        ):
            return 1
        return sum(1 for item in value if isinstance(item, (list, tuple)) and len(item) >= 3)

    def _bbox_gap(a: list[int], b: list[int]) -> int:
        x_gap = max(0, max(int(a[0]) - int(b[2]), int(b[0]) - int(a[2])))
        y_gap = max(0, max(int(a[1]) - int(b[3]), int(b[1]) - int(a[3])))
        return max(x_gap, y_gap)

    def _tight_reference_bbox(value, geometry_bbox: list[int] | None) -> list[int] | None:
        reference = _shift_bbox_y(value, 0)
        if reference is None or geometry_bbox is None:
            return None
        geometry_area = _bbox_area(geometry_bbox)
        reference_area = _bbox_area(reference)
        if geometry_area <= 0 or reference_area <= 0:
            return None
        geometry_w = max(1, int(geometry_bbox[2]) - int(geometry_bbox[0]))
        geometry_h = max(1, int(geometry_bbox[3]) - int(geometry_bbox[1]))
        reference_w = max(1, int(reference[2]) - int(reference[0]))
        reference_h = max(1, int(reference[3]) - int(reference[1]))
        if _bbox_gap(reference, geometry_bbox) > max(18, int(round(max(geometry_w, geometry_h) * 0.35))):
            return None
        if reference_area > max(geometry_area + 4096, int(round(geometry_area * 2.4))):
            return None
        if reference_w > max(geometry_w + 96, int(round(geometry_w * 2.1))):
            return None
        if reference_h > max(geometry_h + 96, int(round(geometry_h * 2.1))):
            return None
        return reference

    def _mark_tight_reference_bbox(record: dict, geometry_bbox: list[int] | None) -> bool:
        marked = False
        for key in ("source_bbox", "balloon_bbox"):
            reference = _tight_reference_bbox(record.get(key), geometry_bbox)
            if reference is not None:
                marked = _mark_bbox(reference) or marked
        return marked

    def _mark_single_line_text_bbox(record: dict, geometry_bbox: list[int] | None) -> bool:
        if _polygon_count(record.get("line_polygons")) != 1:
            return False
        reference = _tight_reference_bbox(record.get("text_pixel_bbox"), geometry_bbox)
        if reference is None:
            return False
        return _mark_bbox(reference)

    def _rotation_abs(record: dict) -> float:
        try:
            return abs(float(record.get("rotation_deg") or record.get("rotation") or 0.0))
        except Exception:
            return 0.0

    def _mark_rotated_source_bbox(record: dict) -> bool:
        if _rotation_abs(record) < 8.0:
            return False
        for key in ("source_bbox", "bbox", "text_pixel_bbox"):
            if _mark_bbox(record.get(key)):
                return True
        return False

    for record in records:
        if not isinstance(record, dict):
            continue
        mask_source = str(record.get("bubble_mask_source") or record.get("balloon_mask_source") or "").strip().lower()
        if mask_source in {"image_dark_bubble_mask", "image_dark_panel_mask", "derived_card_panel_mask"}:
            marked_visual_mask = False
            for key in ("bubble_mask_bbox", "balloon_bbox", "target_bbox", "position_bbox"):
                marked_visual_mask = _mark_bbox(record.get(key)) or marked_visual_mask
            if marked_visual_mask:
                continue
        geometry_bbox = _bbox_from_polygons(record.get("line_polygons"))
        if geometry_bbox is None:
            geometry_bbox = _shift_bbox_y(record.get("text_pixel_bbox") or record.get("bbox"), 0)
        if _mark_rotated_source_bbox(record):
            _mark_polygons(record.get("line_polygons"))
            continue
        if _mark_polygons(record.get("line_polygons")):
            _mark_tight_reference_bbox(record, geometry_bbox)
            _mark_single_line_text_bbox(record, geometry_bbox)
            continue
        for key in ("text_pixel_bbox", "layout_bbox", "source_bbox", "bbox"):
            if _mark_bbox(record.get(key)):
                break

    if not np.any(limit):
        return None
    pad = max(0, _env_int("TRADUZAI_PAGE_INPAINT_CLAMP_PAD_PX", 8))
    if pad > 0:
        kernel_size = max(3, pad * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        limit = cv2.dilate(limit, kernel, iterations=1)
    return (limit > 0).astype(np.uint8) * 255


def _clamp_page_inpaint_to_mask(
    *,
    original_image,
    clean_image,
    rendered_image,
    page_texts: list[dict],
    inpaint_blocks: list[dict] | None,
    breakdown: dict[str, float] | None = None,
) -> tuple[object, object, bool]:
    if (
        original_image is None
        or clean_image is None
        or rendered_image is None
        or not isinstance(original_image, np.ndarray)
        or not isinstance(clean_image, np.ndarray)
        or not isinstance(rendered_image, np.ndarray)
        or original_image.shape != clean_image.shape
        or clean_image.shape != rendered_image.shape
        or clean_image.size == 0
    ):
        return clean_image, rendered_image, False

    try:
        from vision_stack.runtime import _clamp_image_to_limit_mask, _restore_dark_line_art_outside_text_geometry
    except Exception:
        return clean_image, rendered_image, False

    limit_mask = _build_page_inpaint_limit_mask(original_image, inpaint_blocks, page_texts)
    if limit_mask is None or not np.any(limit_mask):
        return clean_image, rendered_image, False

    clamped_clean, limit_pixels, changed_outside = _clamp_image_to_limit_mask(
        original_image,
        clean_image,
        limit_mask,
        page_texts,
        include_text_bboxes=False,
    )
    restored_clean = _restore_dark_line_art_outside_text_geometry(
        original_image,
        clamped_clean,
        page_texts,
    )
    restored_pixels = int(np.count_nonzero(np.any(restored_clean != clamped_clean, axis=2)))
    if not changed_outside and not restored_pixels:
        return clean_image, rendered_image, False

    clamped_rendered = _apply_page_cleanup_background_delta(
        clean_image=clean_image,
        fixed_clean=restored_clean,
        rendered_image=rendered_image,
    )
    if clamped_rendered is None:
        clamped_rendered = rendered_image.copy()
    clamped_rendered = _restore_rendered_text_regions_after_page_clamp(
        rendered_image,
        clamped_rendered,
        page_texts,
    )

    if breakdown is not None:
        breakdown["page_inpaint_clamp_count"] = float(
            breakdown.get("page_inpaint_clamp_count", 0.0) or 0.0
        ) + 1.0
        breakdown["page_inpaint_clamp_limit_pixels"] = float(
            breakdown.get("page_inpaint_clamp_limit_pixels", 0.0) or 0.0
        ) + float(limit_pixels)
        breakdown["page_inpaint_clamp_changed_outside"] = float(
            breakdown.get("page_inpaint_clamp_changed_outside", 0.0) or 0.0
        ) + float(changed_outside)
        breakdown["page_line_art_restore_pixels"] = float(
            breakdown.get("page_line_art_restore_pixels", 0.0) or 0.0
        ) + float(restored_pixels)
    return restored_clean, clamped_rendered, True


def _is_korean_source_language(idioma_origem: str = "") -> bool:
    normalized = str(idioma_origem or "").strip().lower()
    return normalized in {"ko", "kor", "korean", "kr"}


def _looks_like_korean_source_sfx_noise(raw: str, *, bright_balloon_context: bool) -> bool:
    meaningful = [ch for ch in raw if ch.isalnum()]
    if not meaningful:
        return False
    has_hangul = any(
        0xAC00 <= ord(ch) <= 0xD7AF
        or 0x1100 <= ord(ch) <= 0x11FF
        or 0x3130 <= ord(ch) <= 0x318F
        for ch in meaningful
    )
    if has_hangul:
        return False
    has_latin_or_digit = any(("A" <= ch.upper() <= "Z") or ch.isdigit() for ch in meaningful)
    if has_latin_or_digit:
        latin_core = "".join(ch for ch in raw if ("A" <= ch.upper() <= "Z"))
        mostly_latin = bool(latin_core) and len(latin_core) >= max(1, int(len(meaningful) * 0.75))
        if (
            mostly_latin
            and not bright_balloon_context
            and len(latin_core) <= 6
            and latin_core.islower()
        ):
            return True
        return False
    has_cjk_or_kana = any(
        0x3040 <= ord(ch) <= 0x30FF
        or 0x3400 <= ord(ch) <= 0x9FFF
        for ch in meaningful
    )
    compact_len = len(meaningful)
    if has_cjk_or_kana:
        return compact_len <= 4 or (compact_len <= 8 and not bright_balloon_context)
    return compact_len <= 8


def _has_hangul_text(raw: str) -> bool:
    return any(
        0xAC00 <= ord(ch) <= 0xD7AF
        or 0x1100 <= ord(ch) <= 0x11FF
        or 0x3130 <= ord(ch) <= 0x318F
        for ch in str(raw or "")
    )


def _korean_sfx_should_be_translated(raw: str, confidence: float) -> bool:
    if not _has_hangul_text(raw):
        return False
    if float(confidence or 0.0) < 0.82:
        return False
    text = str(raw or "").strip()
    if re.search(r"[!?！？]", text):
        return True
    hangul_count = sum(1 for ch in text if _has_hangul_text(ch))
    return hangul_count >= 3 and bool(re.search(r"(?<!\.)\.$|。$", text))


def _koharu_text_bright_balloon_context(text: dict) -> tuple[bool, bool]:
    profiles = {
        str(text.get("balloon_type") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("layout_profile") or "").strip().lower(),
    }
    structural = bool(profiles & {"white", "white_balloon", "connected_balloon"})
    bright = structural
    background_rgb = text.get("background_rgb")
    if not bright and isinstance(background_rgb, (list, tuple)) and len(background_rgb) >= 3:
        try:
            bright = sum(float(v) for v in background_rgb[:3]) / 3.0 >= 235.0
        except Exception:
            bright = False
    return bright, structural


def _koharu_cjk_text_is_translatable(text: dict, *, idioma_origem: str = "") -> bool:
    raw = str(text.get("text") or text.get("original") or "").strip()
    if not raw:
        return False
    return _text_has_alnum_or_cjk(raw)


def _koharu_cjk_should_page_fallback(
    *,
    text_count: int,
    filtered_count: int,
    empty_count: int,
    job_count: int,
) -> bool:
    if not _koharu_cjk_page_fallback_enabled() or job_count <= 0:
        return False
    if text_count <= 0:
        return True
    if filtered_count > 0:
        return True
    if empty_count >= job_count and job_count > 0:
        return True
    return False


def _koharu_cjk_page_fallback_priority(page_number: int, page_stat: dict) -> tuple[int, int, int, int]:
    filtered_count = int(page_stat.get("filtered_count", 0) or 0)
    text_count = int(page_stat.get("text_count", 0) or 0)
    empty_count = int(page_stat.get("empty_count", 0) or 0)
    job_count = int(page_stat.get("job_count", 0) or 0)
    textlike_count = int(page_stat.get("textlike_count", 0) or 0)
    balloon_area = int(page_stat.get("balloon_area", 0) or 0)
    if filtered_count > 0:
        class_rank = 4
    elif text_count <= 0 and textlike_count > 0:
        class_rank = 3
    elif empty_count >= job_count and job_count > 0:
        class_rank = 2
    else:
        class_rank = 1
    return (class_rank, filtered_count + textlike_count, balloon_area, -int(page_number))


def _bbox_intersects(a: list[int] | None, b: list[int] | None) -> bool:
    if a is None or b is None:
        return False
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) > 0 and max(
        0,
        min(a[3], b[3]) - max(a[1], b[1]),
    ) > 0


def _filter_koharu_cjk_page_result(
    page_result: dict,
    *,
    selective: bool,
    idioma_origem: str = "",
) -> tuple[dict, int]:
    if not selective:
        return page_result, 0
    texts = [text for text in list(page_result.get("texts") or []) if isinstance(text, dict)]
    kept_texts = [
        text
        for text in texts
        if _koharu_cjk_text_is_translatable(text, idioma_origem=idioma_origem)
    ]
    if len(kept_texts) == len(texts):
        return page_result, 0

    kept_bboxes = [
        _shift_bbox_xy(text.get("text_pixel_bbox") or text.get("bbox"), 0, 0)
        for text in kept_texts
    ]
    kept_bboxes = [bbox for bbox in kept_bboxes if bbox is not None]
    filtered_blocks = []
    for block in list(page_result.get("_vision_blocks") or []):
        if not isinstance(block, dict):
            continue
        block_bbox = _shift_bbox_xy(block.get("bbox"), 0, 0)
        if any(_bbox_intersects(block_bbox, bbox) for bbox in kept_bboxes):
            filtered_blocks.append(block)

    filtered = dict(page_result)
    filtered["texts"] = kept_texts
    filtered["_vision_blocks"] = filtered_blocks if kept_texts else []
    return filtered, len(texts) - len(kept_texts)


def _koharu_roi_bbox_for_band(strip: VerticalStrip, band: Band, page_number: int) -> list[int] | None:
    if not band.balloons:
        return None
    page_y0, page_y1 = _source_page_bounds(strip, page_number)
    pad = _koharu_cjk_roi_padding_px()
    x1 = min(int(balloon.strip_bbox.x1) for balloon in band.balloons) - pad
    y1 = min(int(balloon.strip_bbox.y1) for balloon in band.balloons) - pad
    x2 = max(int(balloon.strip_bbox.x2) for balloon in band.balloons) + pad
    y2 = max(int(balloon.strip_bbox.y2) for balloon in band.balloons) + pad
    x1 = max(0, min(int(strip.width), x1))
    x2 = max(0, min(int(strip.width), x2))
    y1 = max(page_y0, min(page_y1, y1))
    y2 = max(page_y0, min(page_y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _run_koharu_cjk_pages_ocr(
    runtime,
    jobs: list[dict],
    *,
    models_dir: str,
    idioma_origem: str,
    work_title: str = "",
    work_title_user_provided: bool = False,
) -> list[dict]:
    if not jobs:
        return []
    runner = getattr(runtime, "run_koharu_cjk_pages", None)
    if callable(runner):
        return list(
            runner(
                jobs,
                models_dir=models_dir,
                idioma_origem=idioma_origem,
                work_title=work_title,
                work_title_user_provided=work_title_user_provided,
            )
        )
    from vision_stack.runtime import _run_koharu_cjk_http_detect_ocr_batch

    return _run_koharu_cjk_http_detect_ocr_batch(
        jobs,
        models_dir=models_dir,
        profile="max",
        idioma_origem=idioma_origem,
        work_title=work_title,
        work_title_user_provided=work_title_user_provided,
    )


def _empty_koharu_precomputed_page(
    band: Band,
    *,
    page_number: int,
    mode: str,
    filtered_text_count: int = 0,
    backend: str = "koharu-http",
    koharu_worker_batch: dict | None = None,
) -> dict:
    height, width = band.strip_slice.shape[:2] if band.strip_slice is not None else (band.height, 0)
    page = {
        "image": _band_image_label(page_number),
        "width": width,
        "height": height,
        "texts": [],
        "_vision_blocks": [],
        "_vision_backend": backend,
        "_ocr_stats": {
            "koharu_cjk_precompute": True,
            "koharu_cjk_mode": mode,
            "koharu_cjk_page_number": int(page_number),
            "koharu_cjk_text_count": 0,
            "koharu_cjk_block_count": 0,
            "koharu_cjk_filtered_text_count": int(filtered_text_count),
        },
    }
    if koharu_worker_batch:
        page["_koharu_worker_batch"] = dict(koharu_worker_batch)
    return page


def _map_koharu_roi_result_to_band(
    *,
    band: Band,
    page_number: int,
    page_result: dict,
    crop_bbox: list[int],
    filtered_text_count: int,
) -> dict:
    if band.strip_slice is None:
        return _empty_koharu_precomputed_page(
            band,
            page_number=page_number,
            mode="roi",
            filtered_text_count=filtered_text_count,
            backend=page_result.get("_vision_backend", "koharu-http"),
            koharu_worker_batch=page_result.get("_koharu_worker_batch"),
        )

    crop_x0, crop_y0, _, _ = crop_bbox
    dx = int(crop_x0)
    dy = int(crop_y0) - int(band.y_top)
    local_texts = [
        _shift_text_geometry_xy(text, dx, dy)
        for text in list(page_result.get("texts") or [])
        if isinstance(text, dict)
    ]
    local_blocks = [
        _shift_block_geometry_xy(block, dx, dy)
        for block in list(page_result.get("_vision_blocks") or [])
        if isinstance(block, dict)
    ]

    if not local_texts:
        return _empty_koharu_precomputed_page(
            band,
            page_number=page_number,
            mode="roi",
            filtered_text_count=filtered_text_count,
            backend=page_result.get("_vision_backend", "koharu-http"),
            koharu_worker_batch=page_result.get("_koharu_worker_batch"),
        )

    if not local_blocks:
        for text in local_texts:
            bbox = _shift_bbox_xy(text.get("balloon_bbox") or text.get("bbox"), 0, 0)
            if bbox is not None:
                local_blocks.append(
                    {
                        "bbox": bbox,
                        "confidence": float(text.get("confidence", text.get("ocr_confidence", 0.9)) or 0.9),
                        "detector": "koharu-text-fallback",
                    }
                )

    height, width = band.strip_slice.shape[:2]
    page = {
        "image": _band_image_label(page_number),
        "width": width,
        "height": height,
        "texts": local_texts,
        "_vision_blocks": local_blocks,
        "_vision_backend": page_result.get("_vision_backend", "koharu-http"),
        "_koharu_http": dict(page_result.get("_koharu_http") or {}),
        "_ocr_stats": {
            "koharu_cjk_precompute": True,
            "koharu_cjk_mode": "roi",
            "koharu_cjk_page_number": int(page_number),
            "koharu_cjk_text_count": len(local_texts),
            "koharu_cjk_block_count": len(local_blocks),
            "koharu_cjk_filtered_text_count": int(filtered_text_count),
        },
    }
    if page_result.get("engine_preset_id"):
        page["engine_preset_id"] = page_result.get("engine_preset_id")
    if isinstance(page_result.get("engine_preset"), dict):
        page["engine_preset"] = dict(page_result.get("engine_preset") or {})
    if isinstance(page_result.get("_engine_preset"), dict):
        page["_engine_preset"] = dict(page_result.get("_engine_preset") or {})
    if page_result.get("_koharu_worker_batch"):
        page["_koharu_worker_batch"] = dict(page_result.get("_koharu_worker_batch") or {})
    return page


def _run_koharu_cjk_page_ocr(
    runtime,
    *,
    image_rgb,
    image_path: Path,
    models_dir: str,
    idioma_origem: str,
    work_title: str = "",
    work_title_user_provided: bool = False,
) -> dict:
    runner = getattr(runtime, "run_koharu_cjk_page", None)
    if callable(runner):
        try:
            return runner(
                image_rgb,
                str(image_path),
                work_title=work_title,
                work_title_user_provided=work_title_user_provided,
            )
        except TypeError:
            return runner(image_rgb, str(image_path))
    from vision_stack.runtime import _run_koharu_cjk_http_detect_ocr

    return _run_koharu_cjk_http_detect_ocr(
        image_rgb=image_rgb,
        image_label=str(Path(image_path).resolve()),
        models_dir=models_dir,
        profile="max",
        idioma_origem=idioma_origem,
        work_title=work_title,
        work_title_user_provided=work_title_user_provided,
    )


def _build_precomputed_koharu_cjk_pages(
    strip: VerticalStrip,
    bands: list[Band],
    runtime,
    page_paths: list[Path],
    *,
    models_dir: str = "",
    idioma_origem: str = "en",
    obra: str = "",
    work_title_user_provided: bool = False,
    telemetry: dict | None = None,
) -> dict[int, dict]:
    started_at = time.perf_counter()
    stats = telemetry if telemetry is not None else {}
    enabled = bool(bands) and _koharu_cjk_strip_precompute_enabled(idioma_origem, models_dir)
    stats.update(
        {
            "enabled": enabled,
            "seconds": 0.0,
            "page_count": 0,
            "precomputed_band_count": 0,
            "failed_page_count": 0,
            "failures": [],
            "text_count": 0,
        }
    )
    if not enabled:
        return {}

    precomputed: dict[int, dict] = {}
    bands_by_page: dict[int, list[tuple[int, Band]]] = {}
    for index, band in enumerate(bands):
        if band.strip_slice is None or not band.balloons:
            continue
        page_number = _source_page_number_for_band(strip, band)
        bands_by_page.setdefault(page_number, []).append((index, band))

    if _koharu_cjk_strip_roi_enabled():
        stats["batch_mode"] = "roi"
        stats["roi_job_count"] = 0
        stats["roi_candidate_count"] = 0
        stats["roi_quick_skip_count"] = 0
        stats["roi_quick_skip_reasons"] = {}
        stats["filtered_text_count"] = 0
        stats["empty_precomputed_band_count"] = 0
        selective = _koharu_cjk_selective_enabled()
        with tempfile.TemporaryDirectory(prefix="traduzai_koharu_cjk_roi_") as tmpdir:
            tmp_path = Path(tmpdir)
            jobs: list[dict] = []
            page_roi_stats: dict[int, dict[str, int]] = {}
            for page_number, page_bands in bands_by_page.items():
                page_index = page_number - 1
                if page_index < 0 or page_index >= len(page_paths):
                    continue
                for band_index, band in page_bands:
                    crop_bbox = _koharu_roi_bbox_for_band(strip, band, page_number)
                    if crop_bbox is None:
                        continue
                    x1, y1, x2, y2 = crop_bbox
                    crop_image = strip.image[y1:y2, x1:x2, :]
                    if crop_image.size == 0:
                        continue
                    stats["roi_candidate_count"] = int(stats.get("roi_candidate_count", 0) or 0) + 1
                    known_text_bboxes = (
                        _koharu_known_bboxes_for_roi(band, crop_bbox)
                        if _koharu_cjk_ocr_only_enabled()
                        else []
                    )
                    has_textlike = True
                    if _koharu_cjk_empty_roi_filter_enabled():
                        has_textlike, quick_reason = _koharu_roi_has_textlike_content(crop_image)
                        if not has_textlike:
                            stats["roi_quick_skip_count"] = int(stats.get("roi_quick_skip_count", 0) or 0) + 1
                            skip_reasons = dict(stats.get("roi_quick_skip_reasons") or {})
                            skip_reasons[str(quick_reason)] = int(skip_reasons.get(str(quick_reason), 0) or 0) + 1
                            stats["roi_quick_skip_reasons"] = skip_reasons
                            stats["empty_precomputed_band_count"] = int(
                                stats.get("empty_precomputed_band_count", 0) or 0
                            ) + 1
                            precomputed[band_index] = _empty_koharu_precomputed_page(
                                band,
                                page_number=page_number,
                                mode="roi_quick_skip",
                            )
                            continue
                    page_stat = page_roi_stats.setdefault(
                        int(page_number),
                        {
                            "job_count": 0,
                            "text_count": 0,
                            "filtered_count": 0,
                            "empty_count": 0,
                            "textlike_count": 0,
                            "balloon_area": 0,
                        },
                    )
                    if has_textlike:
                        page_stat["textlike_count"] += 1
                    page_stat["balloon_area"] += sum(
                        max(0, int(balloon.strip_bbox.width)) * max(0, int(balloon.strip_bbox.height))
                        for balloon in band.balloons
                    )
                    crop_path = tmp_path / f"p{page_number:04d}_b{band_index:04d}.jpg"
                    cv2.imwrite(str(crop_path), crop_image)
                    job = {
                        "image_path": str(crop_path),
                        "image_rgb": crop_image,
                        "mode": "roi",
                        "page_number": int(page_number),
                        "band_index": int(band_index),
                        "crop_bbox": crop_bbox,
                        "band": band,
                        "work_title": obra,
                        "work_title_user_provided": bool(work_title_user_provided),
                    }
                    if known_text_bboxes:
                        job["known_text_bboxes"] = known_text_bboxes
                    jobs.append(job)
                    page_stat["job_count"] += 1

            stats["roi_job_count"] = len(jobs)
            try:
                page_results = _run_koharu_cjk_pages_ocr(
                    runtime,
                    jobs,
                    models_dir=models_dir,
                    idioma_origem=idioma_origem,
                    work_title=obra,
                    work_title_user_provided=bool(work_title_user_provided),
                )
            except Exception as exc:
                stats["failed_page_count"] = int(stats.get("failed_page_count", 0) or 0) + len(jobs)
                failures = list(stats.get("failures") or [])
                failures.append({"page": "roi_batch", "error": str(exc)[:240]})
                stats["failures"] = failures[-10:]
                if _env_bool("TRADUZAI_KOHARU_CJK_STRICT", False):
                    raise
                stats["seconds"] = round(time.perf_counter() - started_at, 4)
                return {}

            _merge_koharu_worker_batch_telemetry(stats, page_results)
            for job, page_result in zip(jobs, page_results):
                band_index = int(job["band_index"])
                band = job["band"]
                page_number = int(job["page_number"])
                page_stat = page_roi_stats.setdefault(
                    page_number,
                    {
                        "job_count": 0,
                        "text_count": 0,
                        "filtered_count": 0,
                        "empty_count": 0,
                        "textlike_count": 0,
                        "balloon_area": 0,
                    },
                )
                filtered_result, filtered_count = _filter_koharu_cjk_page_result(
                    page_result,
                    selective=selective,
                    idioma_origem=idioma_origem,
                )
                stats["filtered_text_count"] = int(stats.get("filtered_text_count", 0) or 0) + filtered_count
                stats["page_count"] = int(stats.get("page_count", 0) or 0) + 1
                stats["text_count"] = int(stats.get("text_count", 0) or 0) + len(filtered_result.get("texts") or [])
                page_stat["filtered_count"] += int(filtered_count)
                page_stat["text_count"] += len(filtered_result.get("texts") or [])
                mapped_page = _map_koharu_roi_result_to_band(
                    band=band,
                    page_number=page_number,
                    page_result=filtered_result,
                    crop_bbox=list(job["crop_bbox"]),
                    filtered_text_count=filtered_count,
                )
                if not list(mapped_page.get("texts") or []):
                    stats["empty_precomputed_band_count"] = int(
                        stats.get("empty_precomputed_band_count", 0) or 0
                    ) + 1
                    page_stat["empty_count"] += 1
                precomputed[band_index] = mapped_page

            fallback_candidates = [
                (page_number, page_stat)
                for page_number, page_stat in page_roi_stats.items()
                if _koharu_cjk_should_page_fallback(
                    text_count=int(page_stat.get("text_count", 0) or 0),
                    filtered_count=int(page_stat.get("filtered_count", 0) or 0),
                    empty_count=int(page_stat.get("empty_count", 0) or 0),
                    job_count=int(page_stat.get("job_count", 0) or 0),
                )
            ]
            fallback_page_numbers = [
                page_number
                for page_number, _page_stat in sorted(
                    fallback_candidates,
                    key=lambda item: _koharu_cjk_page_fallback_priority(item[0], item[1]),
                    reverse=True,
                )
            ][:_koharu_cjk_page_fallback_max()]
            stats["page_fallback_candidate_count"] = len(fallback_page_numbers)
            stats["page_fallback_text_count"] = 0
            if fallback_page_numbers:
                page_jobs: list[dict] = []
                for page_number in fallback_page_numbers:
                    page_index = page_number - 1
                    if page_index < 0 or page_index >= len(page_paths):
                        continue
                    page_y0, page_y1 = _source_page_bounds(strip, page_number)
                    page_x_offsets = list(strip.page_x_offsets or [])
                    page_x0 = int(page_x_offsets[page_index]) if page_index < len(page_x_offsets) else 0
                    page_image = strip.image[page_y0:page_y1, page_x0:strip.width, :]
                    if page_image.size == 0:
                        continue
                    page_jobs.append(
                        {
                            "image_path": str(page_paths[page_index]),
                            "image_rgb": page_image,
                            "mode": "page_fallback",
                            "page_number": int(page_number),
                            "page_bands": list(bands_by_page.get(page_number) or []),
                            "work_title": obra,
                            "work_title_user_provided": bool(work_title_user_provided),
                        }
                    )
                stats["page_fallback_job_count"] = len(page_jobs)
                try:
                    fallback_results = _run_koharu_cjk_pages_ocr(
                        runtime,
                        page_jobs,
                        models_dir=models_dir,
                        idioma_origem=idioma_origem,
                        work_title=obra,
                        work_title_user_provided=bool(work_title_user_provided),
                    )
                except Exception as exc:
                    failures = list(stats.get("failures") or [])
                    failures.append({"page": "page_fallback_batch", "error": str(exc)[:240]})
                    stats["failures"] = failures[-10:]
                    fallback_results = []
                for job, page_result in zip(page_jobs, fallback_results):
                    filtered_result, filtered_count = _filter_koharu_cjk_page_result(
                        page_result,
                        selective=selective,
                        idioma_origem=idioma_origem,
                    )
                    stats["filtered_text_count"] = int(stats.get("filtered_text_count", 0) or 0) + filtered_count
                    stats["page_fallback_text_count"] = int(stats.get("page_fallback_text_count", 0) or 0) + len(
                        filtered_result.get("texts") or []
                    )
                    precomputed.update(
                        _split_koharu_page_result_into_bands(
                            strip,
                            page_number=int(job["page_number"]),
                            page_result=filtered_result,
                            page_bands=list(job["page_bands"]),
                        )
                    )

        stats["precomputed_band_count"] = len(precomputed)
        stats["seconds"] = round(time.perf_counter() - started_at, 4)
        return precomputed

    stats["batch_mode"] = "page"
    page_jobs: list[dict] = []
    for page_number, page_bands in bands_by_page.items():
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(page_paths):
            continue
        page_y0, page_y1 = _source_page_bounds(strip, page_number)
        page_x_offsets = list(strip.page_x_offsets or [])
        page_x0 = int(page_x_offsets[page_index]) if page_index < len(page_x_offsets) else 0
        page_image = strip.image[page_y0:page_y1, page_x0:strip.width, :]
        if page_image.size == 0:
            continue
        page_jobs.append(
            {
                "image_path": str(page_paths[page_index]),
                "image_rgb": page_image,
                "mode": "page",
                "page_number": int(page_number),
                "page_bands": page_bands,
                "work_title": obra,
                "work_title_user_provided": bool(work_title_user_provided),
            }
        )

    try:
        page_results = _run_koharu_cjk_pages_ocr(
            runtime,
            page_jobs,
            models_dir=models_dir,
            idioma_origem=idioma_origem,
            work_title=obra,
            work_title_user_provided=bool(work_title_user_provided),
        )
    except Exception:
        page_results = []

    if page_results:
        _merge_koharu_worker_batch_telemetry(stats, page_results)
        for job, page_result in zip(page_jobs, page_results):
            page_number = int(job["page_number"])
            page_bands = list(job["page_bands"])
            page_result, filtered_count = _filter_koharu_cjk_page_result(
                page_result,
                selective=_koharu_cjk_selective_enabled(),
                idioma_origem=idioma_origem,
            )
            stats["filtered_text_count"] = int(stats.get("filtered_text_count", 0) or 0) + filtered_count
            stats["page_count"] = int(stats.get("page_count", 0) or 0) + 1
            stats["text_count"] = int(stats.get("text_count", 0) or 0) + len(page_result.get("texts") or [])
            precomputed.update(
                _split_koharu_page_result_into_bands(
                    strip,
                    page_number=page_number,
                    page_result=page_result,
                    page_bands=page_bands,
                )
            )
    else:
        for job in page_jobs:
            page_number = int(job["page_number"])
            page_bands = list(job["page_bands"])
            try:
                page_result = _run_koharu_cjk_page_ocr(
                    runtime,
                    image_rgb=job["image_rgb"],
                    image_path=Path(str(job["image_path"])),
                    models_dir=models_dir,
                    idioma_origem=idioma_origem,
                    work_title=obra,
                    work_title_user_provided=bool(work_title_user_provided),
                )
            except Exception as exc:
                stats["failed_page_count"] = int(stats.get("failed_page_count", 0) or 0) + 1
                failures = list(stats.get("failures") or [])
                failures.append({"page": int(page_number), "error": str(exc)[:240]})
                stats["failures"] = failures[-10:]
                if _env_bool("TRADUZAI_KOHARU_CJK_STRICT", False):
                    raise
                continue

            stats["page_count"] = int(stats.get("page_count", 0) or 0) + 1
            page_result, filtered_count = _filter_koharu_cjk_page_result(
                page_result,
                selective=_koharu_cjk_selective_enabled(),
                idioma_origem=idioma_origem,
            )
            stats["filtered_text_count"] = int(stats.get("filtered_text_count", 0) or 0) + filtered_count
            stats["text_count"] = int(stats.get("text_count", 0) or 0) + len(page_result.get("texts") or [])
            precomputed.update(
                _split_koharu_page_result_into_bands(
                    strip,
                    page_number=page_number,
                    page_result=page_result,
                    page_bands=page_bands,
                )
            )

    stats["precomputed_band_count"] = len(precomputed)
    stats["seconds"] = round(time.perf_counter() - started_at, 4)
    return precomputed


def _macro_text_has_value(value) -> bool:
    if isinstance(value, dict):
        return bool(str(value.get("text") or value.get("translated") or "").strip())
    return bool(str(value or "").strip())


def _macro_ocr_precompute_skip_reason(
    image_rgb,
    page_blocks: list[dict],
    *,
    source_page_number: int,
) -> str | None:
    try:
        from vision_stack.runtime import (
            _looks_like_cover_editorial_band,
            _looks_like_scanlation_credit_band,
            _strip_scanlation_credit_skip_enabled,
        )
    except Exception:
        return None

    runtime_blocks = [
        SimpleNamespace(
            xyxy=tuple(block["bbox"]),
            confidence=float(block.get("confidence", 1.0) or 1.0),
            detector="macro-ocr-precompute",
        )
        for block in page_blocks
        if isinstance(block, dict) and block.get("bbox")
    ]
    if not runtime_blocks:
        return None
    if _strip_scanlation_credit_skip_enabled() and _looks_like_scanlation_credit_band(
        image_rgb,
        runtime_blocks,
    ):
        return "scanlation_credit"
    if _looks_like_cover_editorial_band(image_rgb, runtime_blocks, source_page_number):
        return "cover_editorial"
    return None


def _record_macro_ocr_precompute_skip(stats: dict, reason: str) -> None:
    stats["skipped_page_count"] = int(stats.get("skipped_page_count", 0) or 0) + 1
    skip_reasons = dict(stats.get("skip_reasons") or {})
    skip_reasons[reason] = int(skip_reasons.get(reason, 0) or 0) + 1
    stats["skip_reasons"] = skip_reasons


def _build_precomputed_macro_ocr_pages(
    strip: VerticalStrip,
    bands: list[Band],
    runtime,
    *,
    idioma_origem: str = "en",
    telemetry: dict | None = None,
) -> dict[int, dict]:
    started_at = time.perf_counter()
    stats = telemetry if telemetry is not None else {}
    stats.update(
        {
            "enabled": bool(_macro_ocr_real_enabled() and bands),
            "seconds": 0.0,
            "page_count": 0,
            "precomputed_band_count": 0,
            "skipped_page_count": 0,
            "skip_reasons": {},
            "macro_window_count": 0,
            "macro_ocr_block_count": 0,
        }
    )
    if not _macro_ocr_real_enabled() or not bands:
        return {}

    from ocr.macro_ocr import recognize_macro_ocr_windows
    from vision_stack.runtime import build_page_result

    precomputed: dict[int, dict] = {}
    try:
        ocr_engine = _get_macro_ocr_engine(runtime, idioma_origem=idioma_origem)
        backend_name = getattr(ocr_engine, "_backend", getattr(ocr_engine, "model_name", "vision"))
        bands_by_page: dict[int, list[tuple[int, Band]]] = {}
        for index, band in enumerate(bands):
            if band.strip_slice is None or not band.balloons:
                continue
            page_number = _source_page_number_for_band(strip, band)
            bands_by_page.setdefault(page_number, []).append((index, band))

        min_blocks = _macro_ocr_precompute_min_blocks()
        stats["min_blocks"] = min_blocks
        for page_number, page_bands in bands_by_page.items():
            page_y0, page_y1 = _source_page_bounds(strip, page_number)
            page_image = strip.image[page_y0:page_y1, :, :]
            if page_image.size == 0:
                continue

            page_blocks: list[dict] = []
            refs: list[tuple[int, Band, dict]] = []
            page_h, page_w = page_image.shape[:2]
            for band_index, band in page_bands:
                for balloon in band.balloons:
                    bbox = [
                        int(balloon.strip_bbox.x1),
                        int(balloon.strip_bbox.y1 - page_y0),
                        int(balloon.strip_bbox.x2),
                        int(balloon.strip_bbox.y2 - page_y0),
                    ]
                    bbox[0] = max(0, min(page_w, bbox[0]))
                    bbox[2] = max(0, min(page_w, bbox[2]))
                    bbox[1] = max(0, min(page_h, bbox[1]))
                    bbox[3] = max(0, min(page_h, bbox[3]))
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    block = {"bbox": bbox, "confidence": float(balloon.confidence)}
                    page_blocks.append(block)
                    refs.append((band_index, band, block))

            if not page_blocks:
                continue

            stats["page_count"] = int(stats.get("page_count", 0) or 0) + 1
            if len(page_blocks) < min_blocks:
                _record_macro_ocr_precompute_skip(stats, "below_min_blocks")
                continue
            skip_reason = _macro_ocr_precompute_skip_reason(
                page_image,
                page_blocks,
                source_page_number=page_number,
            )
            if skip_reason:
                _record_macro_ocr_precompute_skip(stats, skip_reason)
                continue

            macro_texts, ocr_stats, windows = recognize_macro_ocr_windows(
                ocr_engine,
                page_image,
                page_blocks,
                window_mode="band-groups",
                crop_fallback_max=_env_int("TRADUZAI_MACRO_OCR_CROP_FALLBACK_MAX", 0),
                window_max_blocks=_env_int("TRADUZAI_MACRO_OCR_WINDOW_MAX_BLOCKS", 2),
                window_merge_gap=_env_int("TRADUZAI_MACRO_OCR_WINDOW_MERGE_GAP", 1000),
                window_padding=_env_int("TRADUZAI_MACRO_OCR_WINDOW_PADDING", 96),
            )
            stats["macro_window_count"] = int(stats.get("macro_window_count", 0) or 0) + int(
                (ocr_stats or {}).get("macro_window_count") or len(windows or [])
            )
            stats["macro_ocr_block_count"] = int(
                stats.get("macro_ocr_block_count", 0) or 0
            ) + len(page_blocks)
            by_band: dict[int, list[tuple[Band, dict, object]]] = {}
            for ref_index, (band_index, band, page_block) in enumerate(refs):
                raw_text = macro_texts[ref_index] if ref_index < len(macro_texts) else ""
                by_band.setdefault(band_index, []).append((band, page_block, raw_text))

            window_owner_band_index = min(by_band) if by_band else None
            for band_index, entries in by_band.items():
                band = entries[0][0]
                if band.strip_slice is None:
                    continue
                local_blocks = []
                band_texts = []
                empty_record_count = 0
                for _, page_block, raw_text in entries:
                    x1, y1, x2, y2 = [int(v) for v in page_block["bbox"]]
                    local_bbox = [x1, y1 + page_y0 - band.y_top, x2, y2 + page_y0 - band.y_top]
                    local_blocks.append(
                        SimpleNamespace(
                            xyxy=tuple(local_bbox),
                            confidence=float(page_block.get("confidence", 1.0) or 1.0),
                            detector="macro-ocr",
                        )
                    )
                    band_texts.append(raw_text)
                    if not _macro_text_has_value(raw_text):
                        empty_record_count += 1

                if not local_blocks:
                    continue

                # Macro OCR is already operating on a band crop.  Do not pass a
                # three-digit pseudo page label here, otherwise OCR postprocess
                # can infer "page 001" and apply cover-opening filters to an
                # ordinary band crop.
                page_result = build_page_result(
                    image_path=f"macro_band_source_page_{int(page_number)}",
                    image_rgb=band.strip_slice,
                    blocks=local_blocks,
                    texts=band_texts,
                    profile="quality",
                    ocr_backend=backend_name,
                    enable_font_detection=True,
                    idioma_origem=idioma_origem,
                )
                ocr_page_stats = dict(page_result.get("_ocr_stats") or {})
                owns_page_windows = band_index == window_owner_band_index
                ocr_page_stats.update(
                    {
                        "macro_ocr_real": True,
                        "macro_window_count": (
                            int((ocr_stats or {}).get("macro_window_count") or 0)
                            if owns_page_windows
                            else 0
                        ),
                        "macro_window_reports": len(windows or []) if owns_page_windows else 0,
                        "macro_ocr_page_window_owner": bool(owns_page_windows),
                        "macro_ocr_page_number": int(page_number),
                        "macro_ocr_block_count": int(len(local_blocks)),
                        "macro_ocr_empty_record_count": int(empty_record_count),
                    }
                )
                page_result["_ocr_stats"] = ocr_page_stats
                if page_result.get("texts"):
                    precomputed[band_index] = page_result
    finally:
        stats["precomputed_band_count"] = len(precomputed)
        stats["seconds"] = round(time.perf_counter() - started_at, 4)

    return precomputed


def _summarize_band_perf(
    bands: list[Band],
    macro_ocr_precompute: dict | None = None,
    koharu_cjk_precompute: dict | None = None,
    scheduler_executor: dict | None = None,
) -> dict:
    totals: dict[str, float] = {}
    entries: list[dict] = []
    text_count = 0
    fast_solid_balloon_count = 0
    fast_solid_white_count = 0
    fast_solid_black_count = 0
    fast_solid_colored_count = 0
    fast_white_balloon_count = 0
    fast_local_balloon_count = 0
    remaining_inpaint_blocks = 0
    fast_solid_band_count = 0
    fast_white_band_count = 0
    fast_local_band_count = 0
    ocr_crop_fallback_attempts = 0
    ocr_crop_fallback_recovered = 0
    ocr_crop_fallback_suppressed = 0
    ocr_full_page_mapped = 0
    ocr_precomputed_page_band_count = 0
    ocr_runtime_skipped_band_count = 0
    ocr_macro_ocr_real_band_count = 0
    ocr_macro_window_count = 0
    ocr_macro_ocr_block_count = 0
    ocr_macro_ocr_empty_record_count = 0
    ocr_quick_skipped_no_text_band_count = 0
    ocr_scanlation_credit_skipped_band_count = 0
    ocr_cover_editorial_skipped_band_count = 0
    unchanged_translation_skip_band_count = 0
    skip_processing_copy_band_count = 0
    smart_skip_shadow_candidate_count = 0
    smart_skip_shadow_not_safe_count = 0
    smart_skip_shadow_category_counts: dict[str, int] = {}
    smart_skip_real_candidate_count = 0
    smart_skip_real_not_safe_count = 0
    smart_skip_real_applied_band_count = 0
    smart_skip_real_category_counts: dict[str, int] = {}
    fast_solid_rejection_reasons: dict[str, int] = {}
    fast_solid_fill_reject_reasons: dict[str, int] = {}
    fast_white_rejection_reasons: dict[str, int] = {}
    fast_local_rejection_reasons: dict[str, int] = {}

    def _merge_counts(target: dict[str, int], source) -> None:
        if not isinstance(source, dict):
            return
        for key, value in source.items():
            try:
                count = int(value or 0)
            except Exception:
                continue
            if count > 0:
                target[str(key)] = target.get(str(key), 0) + count

    for index, band in enumerate(bands):
        perf = getattr(band, "perf", {}) or {}
        durations = perf.get("durations_sec") or {}
        for stage, value in durations.items():
            try:
                totals[str(stage)] = totals.get(str(stage), 0.0) + float(value)
            except Exception:
                continue
        try:
            band_texts = int(perf.get("text_count", perf.get("ocr_text_count", 0)) or 0)
        except Exception:
            band_texts = 0
        text_count += band_texts
        try:
            band_fast_solid = int(perf.get("fast_solid_balloon_count", 0) or 0)
        except Exception:
            band_fast_solid = 0
        try:
            band_fast_solid_white = int(perf.get("fast_solid_white_count", 0) or 0)
        except Exception:
            band_fast_solid_white = 0
        try:
            band_fast_solid_black = int(perf.get("fast_solid_black_count", 0) or 0)
        except Exception:
            band_fast_solid_black = 0
        try:
            band_fast_solid_colored = int(perf.get("fast_solid_colored_count", 0) or 0)
        except Exception:
            band_fast_solid_colored = 0
        try:
            band_fast_white = int(perf.get("fast_white_balloon_count", 0) or 0)
        except Exception:
            band_fast_white = 0
        try:
            band_fast_local = int(perf.get("fast_local_balloon_count", 0) or 0)
        except Exception:
            band_fast_local = 0
        try:
            band_remaining_inpaint = int(perf.get("remaining_inpaint_blocks", 0) or 0)
        except Exception:
            band_remaining_inpaint = 0
        fast_solid_balloon_count += band_fast_solid
        fast_solid_white_count += band_fast_solid_white
        fast_solid_black_count += band_fast_solid_black
        fast_solid_colored_count += band_fast_solid_colored
        fast_white_balloon_count += band_fast_white
        fast_local_balloon_count += band_fast_local
        remaining_inpaint_blocks += band_remaining_inpaint
        _merge_counts(fast_solid_rejection_reasons, perf.get("fast_solid_rejection_reasons"))
        _merge_counts(fast_solid_fill_reject_reasons, perf.get("fast_solid_fill_reject_reasons"))
        _merge_counts(fast_white_rejection_reasons, perf.get("fast_white_rejection_reasons"))
        _merge_counts(fast_local_rejection_reasons, perf.get("fast_local_rejection_reasons"))
        if band_fast_solid > 0:
            fast_solid_band_count += 1
        if band_fast_white > 0:
            fast_white_band_count += 1
        if band_fast_local > 0:
            fast_local_band_count += 1
        try:
            band_ocr_full_page_mapped = int(perf.get("ocr_full_page_mapped", 0) or 0)
        except Exception:
            band_ocr_full_page_mapped = 0
        try:
            band_ocr_fallback_attempts = int(perf.get("ocr_crop_fallback_attempts", 0) or 0)
        except Exception:
            band_ocr_fallback_attempts = 0
        try:
            band_ocr_fallback_recovered = int(perf.get("ocr_crop_fallback_recovered", 0) or 0)
        except Exception:
            band_ocr_fallback_recovered = 0
        try:
            band_ocr_fallback_suppressed = int(perf.get("ocr_crop_fallback_suppressed", 0) or 0)
        except Exception:
            band_ocr_fallback_suppressed = 0
        ocr_full_page_mapped += band_ocr_full_page_mapped
        ocr_crop_fallback_attempts += band_ocr_fallback_attempts
        ocr_crop_fallback_recovered += band_ocr_fallback_recovered
        ocr_crop_fallback_suppressed += band_ocr_fallback_suppressed
        band_ocr_precomputed_page = bool(perf.get("ocr_precomputed_page"))
        band_ocr_runtime_skipped = bool(perf.get("ocr_runtime_skipped"))
        band_ocr_macro_ocr_real = bool(perf.get("ocr_macro_ocr_real"))
        if band_ocr_precomputed_page:
            ocr_precomputed_page_band_count += 1
        if band_ocr_runtime_skipped:
            ocr_runtime_skipped_band_count += 1
        if band_ocr_macro_ocr_real:
            ocr_macro_ocr_real_band_count += 1
        try:
            band_ocr_macro_window_count = int(perf.get("ocr_macro_window_count", 0) or 0)
        except Exception:
            band_ocr_macro_window_count = 0
        try:
            band_ocr_macro_ocr_block_count = int(perf.get("ocr_macro_ocr_block_count", 0) or 0)
        except Exception:
            band_ocr_macro_ocr_block_count = 0
        try:
            band_ocr_macro_ocr_empty_record_count = int(
                perf.get("ocr_macro_ocr_empty_record_count", 0) or 0
            )
        except Exception:
            band_ocr_macro_ocr_empty_record_count = 0
        ocr_macro_window_count += band_ocr_macro_window_count
        ocr_macro_ocr_block_count += band_ocr_macro_ocr_block_count
        ocr_macro_ocr_empty_record_count += band_ocr_macro_ocr_empty_record_count
        band_ocr_quick_skipped = bool(perf.get("ocr_quick_skipped_no_text"))
        if band_ocr_quick_skipped:
            ocr_quick_skipped_no_text_band_count += 1
        band_ocr_scanlation_credit_skipped = bool(perf.get("ocr_scanlation_credit_skipped"))
        if band_ocr_scanlation_credit_skipped:
            ocr_scanlation_credit_skipped_band_count += 1
        band_ocr_cover_editorial_skipped = bool(perf.get("ocr_cover_editorial_skipped"))
        if band_ocr_cover_editorial_skipped:
            ocr_cover_editorial_skipped_band_count += 1
        band_unchanged_translation_skip = bool(perf.get("unchanged_translation_skip"))
        if band_unchanged_translation_skip:
            unchanged_translation_skip_band_count += 1
        band_skip_processing_copy = bool(perf.get("skip_processing_copy"))
        if band_skip_processing_copy:
            skip_processing_copy_band_count += 1
        try:
            band_smart_skip_candidates = int(perf.get("smart_skip_shadow_candidate_count", 0) or 0)
        except Exception:
            band_smart_skip_candidates = 0
        try:
            band_smart_skip_not_safe = int(perf.get("smart_skip_shadow_not_safe_count", 0) or 0)
        except Exception:
            band_smart_skip_not_safe = 0
        smart_skip_shadow_candidate_count += band_smart_skip_candidates
        smart_skip_shadow_not_safe_count += band_smart_skip_not_safe
        _merge_counts(smart_skip_shadow_category_counts, perf.get("smart_skip_shadow_category_counts"))
        try:
            band_smart_skip_real_candidates = int(perf.get("smart_skip_real_candidate_count", 0) or 0)
        except Exception:
            band_smart_skip_real_candidates = 0
        try:
            band_smart_skip_real_not_safe = int(perf.get("smart_skip_real_not_safe_count", 0) or 0)
        except Exception:
            band_smart_skip_real_not_safe = 0
        band_smart_skip_real_applied = bool(perf.get("smart_skip_real_applied"))
        smart_skip_real_candidate_count += band_smart_skip_real_candidates
        smart_skip_real_not_safe_count += band_smart_skip_real_not_safe
        if band_smart_skip_real_applied:
            smart_skip_real_applied_band_count += 1
        _merge_counts(smart_skip_real_category_counts, perf.get("smart_skip_real_category_counts"))
        entries.append(
            {
                "band_index": int(perf.get("band_index", index) or index),
                "y_top": int(perf.get("y_top", getattr(band, "y_top", 0)) or 0),
                "y_bottom": int(perf.get("y_bottom", getattr(band, "y_bottom", 0)) or 0),
                "height": int(perf.get("height", getattr(band, "height", 0)) or 0),
                "balloon_count": int(perf.get("balloon_count", len(getattr(band, "balloons", []))) or 0),
                "text_count": band_texts,
                "fast_solid_balloon_count": band_fast_solid,
                "fast_solid_white_count": band_fast_solid_white,
                "fast_solid_black_count": band_fast_solid_black,
                "fast_solid_colored_count": band_fast_solid_colored,
                "fast_white_balloon_count": band_fast_white,
                "fast_local_balloon_count": band_fast_local,
                "remaining_inpaint_blocks": band_remaining_inpaint,
                "fast_solid_rejection_reasons": dict(perf.get("fast_solid_rejection_reasons") or {}),
                "fast_solid_fill_reject_reasons": dict(perf.get("fast_solid_fill_reject_reasons") or {}),
                "fast_solid_fill_samples": list(perf.get("fast_solid_fill_samples") or []),
                "fast_white_rejection_reasons": dict(perf.get("fast_white_rejection_reasons") or {}),
                "fast_local_rejection_reasons": dict(perf.get("fast_local_rejection_reasons") or {}),
                "ocr_full_page_mapped": band_ocr_full_page_mapped,
                "ocr_crop_fallback_attempts": band_ocr_fallback_attempts,
                "ocr_crop_fallback_recovered": band_ocr_fallback_recovered,
                "ocr_crop_fallback_suppressed": band_ocr_fallback_suppressed,
                "ocr_precomputed_page": band_ocr_precomputed_page,
                "ocr_runtime_skipped": band_ocr_runtime_skipped,
                "ocr_macro_ocr_real": band_ocr_macro_ocr_real,
                "ocr_macro_window_count": band_ocr_macro_window_count,
                "ocr_macro_ocr_block_count": band_ocr_macro_ocr_block_count,
                "ocr_macro_ocr_empty_record_count": band_ocr_macro_ocr_empty_record_count,
                "ocr_quick_skipped_no_text": band_ocr_quick_skipped,
                "ocr_scanlation_credit_skipped": band_ocr_scanlation_credit_skipped,
                "ocr_cover_editorial_skipped": band_ocr_cover_editorial_skipped,
                "unchanged_translation_skip": band_unchanged_translation_skip,
                "skip_processing_copy": band_skip_processing_copy,
                "smart_skip_shadow_candidate_count": band_smart_skip_candidates,
                "smart_skip_shadow_not_safe_count": band_smart_skip_not_safe,
                "smart_skip_shadow_category_counts": dict(perf.get("smart_skip_shadow_category_counts") or {}),
                "smart_skip_real_candidate_count": band_smart_skip_real_candidates,
                "smart_skip_real_not_safe_count": band_smart_skip_real_not_safe,
                "smart_skip_real_applied": band_smart_skip_real_applied,
                "smart_skip_real_category_counts": dict(perf.get("smart_skip_real_category_counts") or {}),
                "durations_sec": {stage: round(float(value), 4) for stage, value in sorted(durations.items())},
                "total_sec": round(float(perf.get("total_sec", 0.0) or 0.0), 4),
            }
        )

    def _top_stage(stage: str) -> list[dict]:
        return sorted(
            entries,
            key=lambda item: float(item.get("durations_sec", {}).get(stage, 0.0) or 0.0),
            reverse=True,
        )[:8]

    summary = {
        "band_count": len(bands),
        "text_count": text_count,
        "fast_solid_balloon_count": fast_solid_balloon_count,
        "fast_solid_white_count": fast_solid_white_count,
        "fast_solid_black_count": fast_solid_black_count,
        "fast_solid_colored_count": fast_solid_colored_count,
        "fast_white_balloon_count": fast_white_balloon_count,
        "fast_local_balloon_count": fast_local_balloon_count,
        "fast_solid_band_count": fast_solid_band_count,
        "fast_white_band_count": fast_white_band_count,
        "fast_local_band_count": fast_local_band_count,
        "remaining_inpaint_blocks": remaining_inpaint_blocks,
        "fast_solid_rejection_reasons": fast_solid_rejection_reasons,
        "fast_solid_fill_reject_reasons": fast_solid_fill_reject_reasons,
        "fast_white_rejection_reasons": fast_white_rejection_reasons,
        "fast_local_rejection_reasons": fast_local_rejection_reasons,
        "ocr_full_page_mapped": ocr_full_page_mapped,
        "ocr_crop_fallback_attempts": ocr_crop_fallback_attempts,
        "ocr_crop_fallback_recovered": ocr_crop_fallback_recovered,
        "ocr_crop_fallback_suppressed": ocr_crop_fallback_suppressed,
        "ocr_precomputed_page_band_count": ocr_precomputed_page_band_count,
        "ocr_runtime_skipped_band_count": ocr_runtime_skipped_band_count,
        "ocr_macro_ocr_real_band_count": ocr_macro_ocr_real_band_count,
        "ocr_macro_window_count": ocr_macro_window_count,
        "ocr_macro_ocr_block_count": ocr_macro_ocr_block_count,
        "ocr_macro_ocr_empty_record_count": ocr_macro_ocr_empty_record_count,
        "ocr_quick_skipped_no_text_band_count": ocr_quick_skipped_no_text_band_count,
        "ocr_scanlation_credit_skipped_band_count": ocr_scanlation_credit_skipped_band_count,
        "ocr_cover_editorial_skipped_band_count": ocr_cover_editorial_skipped_band_count,
        "unchanged_translation_skip_band_count": unchanged_translation_skip_band_count,
        "skip_processing_copy_band_count": skip_processing_copy_band_count,
        "smart_skip_shadow_candidate_count": smart_skip_shadow_candidate_count,
        "smart_skip_shadow_not_safe_count": smart_skip_shadow_not_safe_count,
        "smart_skip_shadow_category_counts": smart_skip_shadow_category_counts,
        "smart_skip_real_candidate_count": smart_skip_real_candidate_count,
        "smart_skip_real_not_safe_count": smart_skip_real_not_safe_count,
        "smart_skip_real_applied_band_count": smart_skip_real_applied_band_count,
        "smart_skip_real_category_counts": smart_skip_real_category_counts,
        "durations_sec": {stage: round(value, 4) for stage, value in sorted(totals.items())},
        "entries": entries,
        "top_bands": sorted(entries, key=lambda item: item["total_sec"], reverse=True)[:8],
        "top_ocr_bands": _top_stage("ocr"),
        "top_inpaint_bands": _top_stage("inpaint"),
        "top_typeset_bands": _top_stage("typeset"),
    }
    if macro_ocr_precompute and macro_ocr_precompute.get("enabled"):
        macro_summary = {
            "enabled": True,
            "seconds": round(float(macro_ocr_precompute.get("seconds", 0.0) or 0.0), 4),
            "page_count": int(macro_ocr_precompute.get("page_count", 0) or 0),
            "precomputed_band_count": int(
                macro_ocr_precompute.get("precomputed_band_count", 0) or 0
            ),
            "skipped_page_count": int(macro_ocr_precompute.get("skipped_page_count", 0) or 0),
            "skip_reasons": dict(macro_ocr_precompute.get("skip_reasons") or {}),
            "min_blocks": int(macro_ocr_precompute.get("min_blocks", 1) or 1),
            "macro_window_count": int(macro_ocr_precompute.get("macro_window_count", 0) or 0),
            "macro_ocr_block_count": int(
                macro_ocr_precompute.get("macro_ocr_block_count", 0) or 0
            ),
        }
        summary["macro_ocr_precompute"] = macro_summary
        summary["durations_sec"]["macro_ocr_precompute"] = macro_summary["seconds"]
    if koharu_cjk_precompute and koharu_cjk_precompute.get("enabled"):
        koharu_summary = {
            "enabled": True,
            "seconds": round(float(koharu_cjk_precompute.get("seconds", 0.0) or 0.0), 4),
            "batch_mode": str(koharu_cjk_precompute.get("batch_mode") or "page"),
            "page_count": int(koharu_cjk_precompute.get("page_count", 0) or 0),
            "roi_candidate_count": int(koharu_cjk_precompute.get("roi_candidate_count", 0) or 0),
            "roi_job_count": int(koharu_cjk_precompute.get("roi_job_count", 0) or 0),
            "roi_quick_skip_count": int(koharu_cjk_precompute.get("roi_quick_skip_count", 0) or 0),
            "roi_quick_skip_reasons": dict(koharu_cjk_precompute.get("roi_quick_skip_reasons") or {}),
            "precomputed_band_count": int(
                koharu_cjk_precompute.get("precomputed_band_count", 0) or 0
            ),
            "empty_precomputed_band_count": int(
                koharu_cjk_precompute.get("empty_precomputed_band_count", 0) or 0
            ),
            "failed_page_count": int(koharu_cjk_precompute.get("failed_page_count", 0) or 0),
            "text_count": int(koharu_cjk_precompute.get("text_count", 0) or 0),
            "filtered_text_count": int(koharu_cjk_precompute.get("filtered_text_count", 0) or 0),
            "failures": list(koharu_cjk_precompute.get("failures") or []),
        }
        if isinstance(koharu_cjk_precompute.get("worker_batch"), dict):
            koharu_summary["worker_batch"] = dict(koharu_cjk_precompute.get("worker_batch") or {})
        summary["koharu_cjk_precompute"] = koharu_summary
        summary["durations_sec"]["koharu_cjk_precompute"] = koharu_summary["seconds"]
    if scheduler_executor and scheduler_executor.get("enabled"):
        summary["scheduler_executor"] = {
            "enabled": True,
            "mode": str(scheduler_executor.get("mode") or "sequential_safe"),
            "processed_band_count": int(scheduler_executor.get("processed_band_count", 0) or 0),
            "task_count": int(scheduler_executor.get("task_count", 0) or 0),
            "cpu_task_count": int(scheduler_executor.get("cpu_task_count", 0) or 0),
            "gpu_task_count": int(scheduler_executor.get("gpu_task_count", 0) or 0),
            "stage_counts": dict(scheduler_executor.get("stage_counts") or {}),
            "max_cpu_parallel": int(scheduler_executor.get("max_cpu_parallel", 0) or 0),
            "max_gpu_parallel": int(scheduler_executor.get("max_gpu_parallel", 0) or 0),
            "parallel_inpaint_threads": int(
                scheduler_executor.get("parallel_inpaint_threads", 1) or 1
            ),
            "overlap_worker_count": int(scheduler_executor.get("overlap_worker_count", 1) or 1),
            "ocr_serialized": bool(scheduler_executor.get("ocr_serialized", True)),
            "inpaint_lock_mode": str(scheduler_executor.get("inpaint_lock_mode") or "shared_gpu_lock"),
            "validation_status": str(scheduler_executor.get("validation_status") or ""),
            "validation_reasons": list(scheduler_executor.get("validation_reasons") or []),
            "notes": list(scheduler_executor.get("notes") or []),
        }
    return summary


def run_chapter(
    image_files: list[Path],
    output_dir: Path,
    target_count: int = 60,
    *,
    detector,
    runtime,
    translator,
    inpainter,
    typesetter,
    context: dict | None = None,
    glossario: dict | None = None,
    idioma_origem: str = "en",
    idioma_destino: str = "pt-BR",
    obra: str = "",
    work_title_user_provided: bool = False,
    connected_reasoner_config: dict | None = None,
    models_dir: str = "",
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "traduzai-translator",
    translation_context: dict | None = None,
    chapter_telemetry: dict | None = None,
    skip_page_cleanup_rerender: bool = False,

    progress_callback=None,
) -> list[OutputPage]:
    """Executa o pipeline strip-based ponta-a-ponta."""
    if not image_files:
        return []

    page_paths = image_files
    run_started = time.perf_counter()
    if chapter_telemetry is not None:
        chapter_telemetry.setdefault("durations_sec", {})
        chapter_telemetry["input_page_count"] = len(page_paths)
    with _timed(chapter_telemetry, "strip_build"):
        strip = build_strip(page_paths, progress_callback=progress_callback)
    _write_input_manifest_debug(page_paths, strip)
    if chapter_telemetry is not None:
        chapter_telemetry["strip_width"] = int(strip.width)
        chapter_telemetry["strip_height"] = int(strip.height)
    with _timed(chapter_telemetry, "strip_copy_original"):
        original_strip_image = strip.image.copy()

    with _timed(chapter_telemetry, "inpainter_prewarm_start"):
        prewarm_handle = _start_inpainter_prewarm(inpainter, page_paths)

    try:
        if progress_callback: progress_callback("detect", 0, 1)
        with _timed(chapter_telemetry, "strip_detect_balloons"):
            balloons = detect_strip_balloons(strip, detector=detector)
        if chapter_telemetry is not None:
            chapter_telemetry["balloon_count"] = len(balloons)

        with _timed(chapter_telemetry, "strip_group_bands"):
            band_margin = _strip_band_margin_px(idioma_origem)
            bands = group_balloons_into_bands(balloons, margin=band_margin)
        if chapter_telemetry is not None:
            chapter_telemetry["band_count"] = len(bands)
            chapter_telemetry["band_margin_px"] = int(band_margin)
        with _timed(chapter_telemetry, "strip_attach_band_slices"):
            attach_band_slices(strip, bands)
        _write_strip_detect_debug_artifacts(strip, bands, band_margin_px=band_margin)

        if is_debug_enabled():
            with _timed(chapter_telemetry, "strip_debug_dump"):
                dump_strip_debug(strip, bands, output_dir.parent / "_strip_debug")

        with _timed(chapter_telemetry, "scheduler_plan"):
            scheduler_executor_report = _build_scheduler_executor_report(
                band_count=len(bands),
                page_count=len(page_paths),
            )
        macro_ocr_precompute_stats: dict = {}
        with _timed(chapter_telemetry, "macro_ocr_precompute_wall"):
            precomputed_macro_ocr_pages = _build_precomputed_macro_ocr_pages(
                strip,
                bands,
                runtime,
                idioma_origem=idioma_origem,
                telemetry=macro_ocr_precompute_stats,
            )
        koharu_cjk_precompute_stats: dict = {}
        with _timed(chapter_telemetry, "koharu_cjk_precompute_wall"):
            precomputed_koharu_cjk_pages = _build_precomputed_koharu_cjk_pages(
                strip,
                bands,
                runtime,
                page_paths,
                models_dir=models_dir,
                idioma_origem=idioma_origem,
                telemetry=koharu_cjk_precompute_stats,
                obra=obra,
                work_title_user_provided=bool(work_title_user_provided),
            )
        precomputed_ocr_pages = {
            **precomputed_macro_ocr_pages,
            **precomputed_koharu_cjk_pages,
        }

        running_glossary: dict = dict(glossario or {})
        running_history: list[dict] = []
        overlap_executor = (
            scheduler_executor_report is not None
            and scheduler_executor_report.get("mode") == "overlap_context_release"
        )
        parallel_inpaint_threads = (
            int(scheduler_executor_report.get("parallel_inpaint_threads", 1) or 1)
            if overlap_executor and scheduler_executor_report is not None
            else 1
        )
        overlap_worker_count = (
            int(scheduler_executor_report.get("overlap_worker_count", 2) or 2)
            if overlap_executor and scheduler_executor_report is not None
            else 2
        )
        if overlap_executor and parallel_inpaint_threads > 1:
            gpu_stage_lock = None
            ocr_stage_lock = threading.Lock()
            inpaint_stage_lock = threading.BoundedSemaphore(parallel_inpaint_threads)
        else:
            gpu_stage_lock = threading.Lock() if overlap_executor else None
            ocr_stage_lock = None
            inpaint_stage_lock = None
        typeset_stage_lock = threading.Lock() if overlap_executor else None

        def _make_ordered_context_callback():
            state = {
                "merged": False,
                "event": threading.Event() if overlap_executor else None,
            }

            def _merge_after_translate(translated_page: dict | None) -> None:
                _merge_ordered_band_context_after_commit(
                    running_history,
                    running_glossary,
                    translated_page,
                )
                state["merged"] = True
                event = state.get("event")
                if event is not None:
                    event.set()

            return state, _merge_after_translate

        debug_recorder_for_workers = _get_debug_recorder()

        def _process_one_band(idx: int, band: Band, ordered_kwargs: dict, callback):
            if debug_recorder_for_workers is not None:
                try:
                    from debug_tools import bind_recorder

                    bind_recorder(debug_recorder_for_workers)
                except Exception:
                    pass
            source_page_number = _source_page_number_for_band(strip, band)
            page_y0, page_y1 = _source_page_bounds(strip, source_page_number)
            layout_page_image_bgr = cv2.cvtColor(strip.image[page_y0:page_y1, :, :], cv2.COLOR_RGB2BGR)
            return process_band(
                band,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
                page_idx=idx,
                context=context,
                glossario=ordered_kwargs["glossario"],
                idioma_origem=idioma_origem,
                idioma_destino=idioma_destino,
                obra=obra,
                work_title_user_provided=work_title_user_provided,
                connected_reasoner_config=connected_reasoner_config,
                band_history=ordered_kwargs["band_history"],
                source_page_number=source_page_number,
                models_dir=models_dir,
                ollama_host=ollama_host,
                ollama_model=ollama_model,
                translation_context=translation_context,
                precomputed_ocr_page=precomputed_ocr_pages.get(idx),
                ordered_context_after_translate_callback=callback,
                layout_page_image_bgr=layout_page_image_bgr,
                layout_page_y_top=page_y0,
                gpu_stage_lock=gpu_stage_lock,
                ocr_stage_lock=ocr_stage_lock,
                inpaint_stage_lock=inpaint_stage_lock,
                typeset_stage_lock=typeset_stage_lock,
            )

        def _merge_fallback_if_needed(state: dict, band: Band) -> None:
            if state.get("merged"):
                return
            _merge_ordered_band_context_after_commit(
                running_history,
                running_glossary,
                band.ocr_result,
            )
            state["merged"] = True
            event = state.get("event")
            if event is not None:
                event.set()

        process_bands_started = time.perf_counter()
        if overlap_executor:
            futures = []
            with ThreadPoolExecutor(
                max_workers=overlap_worker_count,
                thread_name_prefix="traduzai-strip-overlap",
            ) as executor:
                for idx, band in enumerate(bands):
                    if progress_callback: progress_callback("process", idx, len(bands))
                    ordered_context = _build_ordered_band_context_snapshot(
                        running_history,
                        running_glossary,
                    )
                    ordered_kwargs = ordered_context.to_process_kwargs()
                    state, callback = _make_ordered_context_callback()
                    future = executor.submit(_process_one_band, idx, band, ordered_kwargs, callback)
                    futures.append((future, state, band))
                    event = state.get("event")
                    while event is not None and not event.is_set():
                        if future.done():
                            future.result()
                            break
                        event.wait(0.01)
                    _merge_fallback_if_needed(state, band)

                for future, state, band in futures:
                    future.result()
                    _merge_fallback_if_needed(state, band)
                    if scheduler_executor_report is not None:
                        scheduler_executor_report["processed_band_count"] = (
                            int(scheduler_executor_report.get("processed_band_count", 0) or 0) + 1
                        )

        for idx, band in enumerate([] if overlap_executor else bands):
            if progress_callback: progress_callback("process", idx, len(bands))
            ordered_context = _build_ordered_band_context_snapshot(
                running_history,
                running_glossary,
            )
            ordered_kwargs = ordered_context.to_process_kwargs()
            ordered_context_merged = False

            def _merge_after_translate(translated_page: dict | None) -> None:
                nonlocal ordered_context_merged
                _merge_ordered_band_context_after_commit(
                    running_history,
                    running_glossary,
                    translated_page,
                )
                ordered_context_merged = True

            source_page_number = _source_page_number_for_band(strip, band)
            page_y0, page_y1 = _source_page_bounds(strip, source_page_number)
            layout_page_image_bgr = cv2.cvtColor(strip.image[page_y0:page_y1, :, :], cv2.COLOR_RGB2BGR)
            process_band(
                band,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
                page_idx=idx,
                context=context,
                glossario=ordered_kwargs["glossario"],
                idioma_origem=idioma_origem,
                idioma_destino=idioma_destino,
                obra=obra,
                work_title_user_provided=work_title_user_provided,
                connected_reasoner_config=connected_reasoner_config,
                band_history=ordered_kwargs["band_history"],
                source_page_number=source_page_number,
                models_dir=models_dir,
                ollama_host=ollama_host,
                ollama_model=ollama_model,
                translation_context=translation_context,
                precomputed_ocr_page=precomputed_ocr_pages.get(idx),
                ordered_context_after_translate_callback=_merge_after_translate,
                layout_page_image_bgr=layout_page_image_bgr,
                layout_page_y_top=page_y0,
            )
            if scheduler_executor_report is not None:
                scheduler_executor_report["processed_band_count"] = (
                    int(scheduler_executor_report.get("processed_band_count", 0) or 0) + 1
                )
            # Acumular history e mesclar adições ao glossário
            if not ordered_context_merged:
                _merge_ordered_band_context_after_commit(
                    running_history,
                    running_glossary,
                    band.ocr_result,
                )
        _add_timing(chapter_telemetry, "strip_process_bands_total", time.perf_counter() - process_bands_started)
    finally:
        with _timed(chapter_telemetry, "inpainter_prewarm_close"):
            _close_inpainter_prewarm(prewarm_handle)

    exclusion_intervals, exclusion_rows = _excluded_non_story_intervals(
        bands,
        source_page_breaks=list(strip.source_page_breaks or []),
        strip_height=int(strip.image.shape[0]),
    )
    _write_non_story_exclusions_debug(exclusion_rows)
    if exclusion_rows:
        chapter_telemetry["excluded_non_story_bands"] = [row["band_id"] for row in exclusion_rows]
        chapter_telemetry["excluded_non_story_count"] = len(exclusion_rows)

    with _timed(chapter_telemetry, "strip_paste_cleaned"):
        clean_strip_image_full = _paste_band_attr_into_image(original_strip_image, bands, "cleaned_slice")
    with _timed(chapter_telemetry, "strip_paste_rendered"):
        rendered_strip_image_full = _paste_band_attr_into_image(original_strip_image, bands, "rendered_slice")
    if exclusion_intervals:
        with _timed(chapter_telemetry, "strip_remove_non_story_exclusions"):
            output_original_strip_image = _remove_vertical_intervals(original_strip_image, exclusion_intervals)
            clean_strip_image = _remove_vertical_intervals(clean_strip_image_full, exclusion_intervals)
            rendered_strip_image = _remove_vertical_intervals(rendered_strip_image_full, exclusion_intervals)
            output_bands = _remap_bands_after_exclusions(bands, exclusion_intervals)
            output_balloons = [
                balloon
                for band in output_bands
                for balloon in list(getattr(band, "balloons", []) or [])
            ]
            output_breaks = _remap_breaks_after_exclusions(
                list(strip.source_page_breaks or []),
                exclusion_intervals,
                int(rendered_strip_image.shape[0]),
            )
    else:
        output_original_strip_image = original_strip_image
        clean_strip_image = clean_strip_image_full
        rendered_strip_image = rendered_strip_image_full
        output_bands = bands
        output_balloons = balloons
        output_breaks = list(strip.source_page_breaks)
    with _timed(chapter_telemetry, "strip_assign_rendered"):
        if strip.image.shape == rendered_strip_image.shape:
            strip.image[:, :, :] = rendered_strip_image

    with _timed(chapter_telemetry, "assemble_rendered_pages"):
        output_pages = assemble_output_pages(
            VerticalStrip(
                image=rendered_strip_image,
                width=strip.width,
                height=int(rendered_strip_image.shape[0]),
                source_page_breaks=list(output_breaks),
                page_x_offsets=list(strip.page_x_offsets),
            ),
            output_balloons,
            target_count=target_count,
        )
    with _timed(chapter_telemetry, "assemble_original_pages"):
        original_pages = assemble_output_pages(
            VerticalStrip(
                image=output_original_strip_image,
                width=strip.width,
                height=int(output_original_strip_image.shape[0]),
                source_page_breaks=list(output_breaks),
                page_x_offsets=list(strip.page_x_offsets),
            ),
            output_balloons,
            target_count=target_count,
        )
    with _timed(chapter_telemetry, "assemble_clean_pages"):
        clean_pages = assemble_output_pages(
            VerticalStrip(
                image=clean_strip_image,
                width=strip.width,
                height=int(clean_strip_image.shape[0]),
                source_page_breaks=list(output_breaks),
                page_x_offsets=list(strip.page_x_offsets),
            ),
            output_balloons,
            target_count=target_count,
        )
    _write_reassemble_manifest_debug(
        output_pages,
        original_pages,
        clean_pages,
        target_count=target_count,
    )

    # Remapeamento de metadados para project.json
    remap_started = time.perf_counter()
    all_texts: list[dict] = []
    all_vision_blocks: list[dict] = []
    for band in output_bands:
        if not band.ocr_result:
            continue
        b_y = band.y_top
        # Coleta textos e remapa para coordenadas do strip
        for txt in band.ocr_result.get("texts", []):
            new_txt = dict(txt)
            # bbox é OBRIGATÓRIO — pular texto sem bbox para evitar placeholder [0,0,32,32]
            if not new_txt.get("bbox"):
                continue
            new_txt["band_y_top"] = int(b_y)
            new_txt["band_height"] = int(band.y_bottom - band.y_top)
            new_txt.setdefault("source_coordinate_space", "band")
            new_txt = _shift_text_geometry_y(new_txt, b_y)
            all_texts.append(new_txt)

        for vb in band.ocr_result.get("_vision_blocks", []):
            new_vb = dict(vb)
            if not new_vb.get("bbox"):
                continue
            new_vb["band_y_top"] = int(b_y)
            new_vb["band_height"] = int(band.y_bottom - band.y_top)
            new_vb.setdefault("source_coordinate_space", "band")
            new_vb = _shift_text_geometry_y(new_vb, b_y)
            all_vision_blocks.append(new_vb)
    _add_timing(chapter_telemetry, "remap_band_metadata", time.perf_counter() - remap_started)

    def _assign_text_to_page(txt_y1: int, txt_y2: int, pages: list) -> int | None:
        """Retorna índice da página com maior intersecção em y (sem duplicar)."""
        best_idx = None
        best_overlap = 0
        for idx, page in enumerate(pages):
            overlap = max(0, min(txt_y2, page.y_bottom) - max(txt_y1, page.y_top))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        return best_idx

    # Inicializar listas em cada página
    assign_started = time.perf_counter()
    for page in output_pages:
        page.ocr_result = {"_vision_blocks": []}
        page.text_layers = {"texts": []}

    for band_index, band in enumerate(output_bands, start=1):
        if not isinstance(getattr(band, "ocr_result", None), dict):
            continue
        for page in output_pages:
            overlap = max(0, min(int(band.y_bottom), int(page.y_bottom)) - max(int(band.y_top), int(page.y_top)))
            if overlap > 0:
                _attach_band_pipeline_metadata_to_page(page, band, band_index)

    # Distribuir textos para as páginas por máxima intersecção (não centro-y)
    for txt in all_texts:
        tx1, ty1, tx2, ty2 = txt["bbox"]
        pidx = _assign_text_to_page(ty1, ty2, output_pages)
        if pidx is None:
            continue
        page = output_pages[pidx]
        p_y0 = page.y_top
        local_txt = _shift_text_geometry_y(txt, -p_y0)
        if "band_y_top" in local_txt:
            local_txt["strip_band_y_top"] = int(txt.get("band_y_top") or 0)
            local_txt["band_y_top"] = int(local_txt["strip_band_y_top"]) - int(p_y0)
        if "_band_y_top" in local_txt:
            local_txt["_strip_band_y_top"] = int(txt.get("_band_y_top") or 0)
            local_txt["_band_y_top"] = int(local_txt["_strip_band_y_top"]) - int(p_y0)
        local_txt = _sync_record_page_identity_for_output_page(local_txt, pidx)
        local_txt["coordinate_space"] = "page"
        local_txt["source_coordinate_space"] = "page"
        local_txt = _clamp_record_geometry_to_page(
            local_txt,
            page_width=int(page.image.shape[1]) if isinstance(page.image, np.ndarray) and page.image.ndim >= 2 else 0,
            page_height=int(page.image.shape[0]) if isinstance(page.image, np.ndarray) and page.image.ndim >= 2 else int(page.y_bottom - page.y_top),
        )
        page.text_layers["texts"].append(local_txt)

    # Distribuir vision_blocks igualmente
    for vb in all_vision_blocks:
        vx1, vy1, vx2, vy2 = vb["bbox"]
        pidx = _assign_text_to_page(vy1, vy2, output_pages)
        if pidx is None:
            continue
        page = output_pages[pidx]
        p_y0 = page.y_top
        local_vb = _shift_text_geometry_y(vb, -p_y0)
        if "band_y_top" in local_vb:
            local_vb["strip_band_y_top"] = int(vb.get("band_y_top") or 0)
            local_vb["band_y_top"] = int(local_vb["strip_band_y_top"]) - int(p_y0)
        if "_band_y_top" in local_vb:
            local_vb["_strip_band_y_top"] = int(vb.get("_band_y_top") or 0)
            local_vb["_band_y_top"] = int(local_vb["_strip_band_y_top"]) - int(p_y0)
        local_vb["bbox"] = [vx1, vy1 - p_y0, vx2, vy2 - p_y0]
        local_vb = _sync_record_page_identity_for_output_page(local_vb, pidx)
        local_vb["coordinate_space"] = "page"
        local_vb["source_coordinate_space"] = "page"
        local_vb = _clamp_record_geometry_to_page(
            local_vb,
            page_width=int(page.image.shape[1]) if isinstance(page.image, np.ndarray) and page.image.ndim >= 2 else 0,
            page_height=int(page.image.shape[0]) if isinstance(page.image, np.ndarray) and page.image.ndim >= 2 else int(page.y_bottom - page.y_top),
        )
        page.ocr_result["_vision_blocks"].append(local_vb)
    _add_timing(chapter_telemetry, "assign_metadata_to_pages", time.perf_counter() - assign_started)

    finalize_page_metadata_started = time.perf_counter()
    page_metadata_changed = [False for _ in output_pages]
    for page_index, page in enumerate(output_pages):
        page_metadata_changed[page_index] = _finalize_output_page_ocr_metadata(
            page,
            page_index + 1,
            total_pages=len(output_pages),
        )
    _add_timing(
        chapter_telemetry,
        "finalize_page_ocr_metadata",
        time.perf_counter() - finalize_page_metadata_started,
    )
    _write_ocr_confidence_audit(output_pages)
    try:
        from debug_tools.bbox import write_layout_geometry_debug_artifacts

        write_layout_geometry_debug_artifacts(output_pages)
    except Exception:
        pass
    _write_strip_detect_text_matching_debug_artifacts(strip, output_bands, output_pages)

    with _timed(chapter_telemetry, "summarize_band_perf"):
        strip_perf_summary = _summarize_band_perf(
            bands,
            macro_ocr_precompute=macro_ocr_precompute_stats,
            koharu_cjk_precompute=koharu_cjk_precompute_stats,
            scheduler_executor=scheduler_executor_report,
        )
    if chapter_telemetry is not None:
        strip_perf_summary["chapter_stage_durations_sec"] = dict(chapter_telemetry.get("durations_sec", {}))
        strip_perf_summary["chapter_stage_total_sec"] = round(
            sum(float(v) for v in chapter_telemetry.get("durations_sec", {}).values()),
            4,
        )

    # Preencher page_profile e inpaint_blocks em cada página
    attach_profile_started = time.perf_counter()
    for page_index, page in enumerate(output_pages):
        page.page_profile = {
            "width": strip.width,
            "height": page.y_bottom - page.y_top,
            "y_in_strip_top": page.y_top,
            "y_in_strip_bottom": page.y_bottom,
        }
        if page_index == 0:
            page.page_profile["strip_perf_summary"] = strip_perf_summary
        if isinstance(page.ocr_result.get("_pipeline_artifacts"), dict):
            page.page_profile["_pipeline_artifacts"] = page.ocr_result["_pipeline_artifacts"]
        page.ocr_result["page_profile"] = page.page_profile
        page_text_layers = _page_texts_from_text_layers(page.text_layers)
        page.inpaint_blocks = [
            _enrich_inpaint_block_from_text_layers(block, page_text_layers)
            for block in [
                _inpaint_block_from_vision_block(vb)
                for vb in page.ocr_result.get("_vision_blocks", [])
            ]
            if block is not None
        ]
    _write_inpaint_blocks_debug(output_pages)
    _add_timing(chapter_telemetry, "attach_page_profiles", time.perf_counter() - attach_profile_started)

    cleanup_breakdown: dict[str, float] = {
        "cleanup_inpaint": 0.0,
        "cleanup_typeset": 0.0,
        "cleanup_copyback": 0.0,
        "cleanup_save": 0.0,
    }
    cleanup_started = time.perf_counter()
    skip_page_cleanup = bool(skip_page_cleanup_rerender) or _debug_skip_page_cleanup_rerender()
    for page_index, (page, original_page, clean_page) in enumerate(zip(output_pages, original_pages, clean_pages)):
        page_texts = _page_texts_from_text_layers(page.text_layers)
        stage_page_texts = _texts_without_legacy_decision_fields(page_texts)
        page.original_image = original_page.image
        clean_base, rendered_base, _did_page_clamp = _clamp_page_inpaint_to_mask(
            original_image=original_page.image,
            clean_image=clean_page.image,
            rendered_image=page.image,
            page_texts=stage_page_texts,
            inpaint_blocks=page.inpaint_blocks,
            breakdown=cleanup_breakdown,
        )
        if skip_page_cleanup:
            page.inpainted_image = clean_base
            page.image = rendered_base
            continue
        fixed_clean, fixed_rendered, did_fix = _cleanup_page_inpaint_and_rerender(
            original_image=original_page.image,
            clean_image=clean_base,
            page_texts=stage_page_texts,
            rendered_image=rendered_base,
            typesetter=typesetter,
            breakdown=cleanup_breakdown,
        )
        dark_clean, dark_rendered, did_dark_fix = _cleanup_dark_panel_page_and_rerender(
            clean_image=fixed_clean,
            rendered_image=fixed_rendered,
            page_texts=stage_page_texts,
            typesetter=typesetter,
            breakdown=cleanup_breakdown,
        )
        if did_dark_fix:
            fixed_clean = dark_clean
            fixed_rendered = dark_rendered
            did_fix = True
        if _page_requires_page_space_typeset(page_texts):
            try:
                cleanup_stage_started = time.perf_counter()
                fixed_rendered = typesetter.render_band_image(
                    fixed_clean,
                    _render_payload_without_legacy_decision_fields(
                        page_texts,
                        coordinate_space="page_rectframe_connected",
                    ),
                )
                cleanup_breakdown["cleanup_typeset"] += time.perf_counter() - cleanup_stage_started
                cleanup_breakdown["cleanup_page_space_typeset_count"] = float(
                    cleanup_breakdown.get("cleanup_page_space_typeset_count", 0.0) or 0.0
                ) + 1.0
                did_fix = True
            except Exception:
                pass
        if did_fix and _dark_text_cleanup_loses_visible_ink(rendered_base, fixed_rendered, page_texts):
            cleanup_breakdown["cleanup_rerender_rejected_dark_text_ink_loss"] = float(
                cleanup_breakdown.get("cleanup_rerender_rejected_dark_text_ink_loss", 0.0) or 0.0
            ) + 1.0
            fixed_rendered = rendered_base
            did_fix = False
        page.inpainted_image = fixed_clean
        if did_fix:
            page.image = fixed_rendered
        elif page_index < len(page_metadata_changed) and page_metadata_changed[page_index]:
            try:
                cleanup_stage_started = time.perf_counter()
                page.image = typesetter.render_band_image(
                    fixed_clean,
                    _render_payload_without_legacy_decision_fields(page_texts, coordinate_space="page"),
                )
                cleanup_breakdown["cleanup_typeset"] += time.perf_counter() - cleanup_stage_started
            except Exception:
                page.image = fixed_rendered
    cleanup_total = time.perf_counter() - cleanup_started
    cleanup_breakdown["cleanup_total"] = cleanup_total
    cleanup_breakdown["page_cleanup_skipped"] = 1.0 if skip_page_cleanup else 0.0
    _add_timing(chapter_telemetry, "page_cleanup_rerender", cleanup_total)

    if _macro_ocr_shadow_enabled() and output_pages:
        with _timed(chapter_telemetry, "macro_ocr_shadow"):
            macro_shadow = _run_macro_ocr_shadow(
                output_pages,
                runtime,
                idioma_origem=idioma_origem,
            )
        output_pages[0].page_profile["macro_ocr_shadow"] = macro_shadow
        output_pages[0].ocr_result["page_profile"] = output_pages[0].page_profile

    final_page_space_started = time.perf_counter()
    final_page_space_count = 0
    if not skip_page_cleanup and _strip_final_page_space_typeset_enabled():
        for page in output_pages:
            page_texts = _page_texts_from_text_layers(page.text_layers)
            if not _page_requires_page_space_typeset(page_texts):
                continue
            clean_base = page.inpainted_image if isinstance(page.inpainted_image, np.ndarray) else None
            if clean_base is None:
                continue
            try:
                page.image = typesetter.render_band_image(
                    clean_base,
                    _render_payload_without_legacy_decision_fields(
                        page_texts,
                        coordinate_space="final_page_space_typeset",
                    ),
                )
                final_page_space_count += 1
            except Exception:
                continue
    if final_page_space_count:
        cleanup_breakdown["final_page_space_typeset_count"] = float(final_page_space_count)
    _add_timing(chapter_telemetry, "final_page_space_typeset", time.perf_counter() - final_page_space_started)

    _mark_pipeline_artifacts_after_render(output_pages)
    _write_pipeline_artifacts_debug(output_pages)
    _write_contact_sheets_debug(original_pages, output_pages, output_bands)

    with _timed(chapter_telemetry, "write_translated_pages"):
        cleanup_breakdown["cleanup_save"] += _write_output_pages_after_lossless_debug(
            output_pages,
            output_bands,
            output_dir,
        )
    _write_final_band_crop_debug(output_pages, output_bands)

    _write_page_cleanup_breakdown_debug(cleanup_breakdown)

    if chapter_telemetry is not None:
        chapter_telemetry["wall_total_sec"] = round(time.perf_counter() - run_started, 4)
        chapter_telemetry["output_page_count"] = len(output_pages)
        chapter_telemetry["text_count"] = sum(
            len((page.text_layers or {}).get("texts") or [])
            for page in output_pages
            if isinstance(page.text_layers, dict)
        )

    return output_pages
