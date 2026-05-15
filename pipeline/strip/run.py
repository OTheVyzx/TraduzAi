"""Entry-point do pipeline strip-based.

Chamado por `pipeline/main.py::_run_pipeline` após a Fase 6 do switchover.
"""

from __future__ import annotations

import copy
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
from strip.detect_balloons import detect_strip_balloons
from strip.process_bands import process_band
from strip.reassemble import assemble_output_pages
from strip.types import Band, OutputPage, VerticalStrip


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
        result[y0:y1, :, :] = band_slice[:h_avail, :, :]
    return result


def _shift_bbox_y(value, delta_y: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    return [int(value[0]), int(value[1]) + delta_y, int(value[2]), int(value[3]) + delta_y]


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

    for key in ("bbox", "source_bbox", "balloon_bbox", "text_pixel_bbox", "mask_bbox"):
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
    ):
        if key in shifted:
            shifted[key] = _shift_bbox_list_y(shifted.get(key), delta_y)

    for key in ("line_polygons", "connected_lobe_polygons", "balloon_polygon"):
        if key in shifted:
            shifted[key] = _shift_polygons_y(shifted.get(key), delta_y)

    return shifted


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
        "balloon_bbox",
        "balloon_type",
        "block_profile",
        "background_type",
        "tipo",
        "font_size_px",
        "font_size",
    ):
        value = block.get(key)
        if value is not None and value != [] and value != "":
            out[key] = copy.deepcopy(value)
    return out


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
            _bbox(block.get("bbox")) if isinstance(block, dict) else tuple(),
        )
        for text, block in zip(texts, blocks)
        if isinstance(text, dict)
    )


def _finalize_output_page_ocr_metadata(page: OutputPage, page_number: int) -> bool:
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
    try:
        from vision_stack.runtime import _finalize_page_ocr_texts

        final_texts, final_blocks = _finalize_page_ocr_texts(
            texts,
            blocks,
            image_shape,
            page_number=page_number,
        )
    except Exception:
        return False

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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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
        "validation_status": plan.validation.status,
        "validation_reasons": list(plan.validation.reasons),
        "notes": [
            "Experimental flag only: validate produced output against the sequential baseline.",
            "overlap_context_release keeps a single GPU lane and releases ordered context after translate.",
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
    for key in ("bbox", "source_bbox", "balloon_bbox", "text_pixel_bbox"):
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
    ):
        if key in shifted:
            shifted[key] = _shift_bbox_list_xy(shifted.get(key), dx, dy)
    for key in ("line_polygons", "connected_lobe_polygons", "balloon_polygon"):
        if key in shifted:
            shifted[key] = _shift_polygons_xy(shifted.get(key), dx, dy)
    return shifted


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
    if not isinstance(text, dict) or text.get("skip_processing"):
        return False
    translated = str(text.get("translated") or text.get("traduzido") or "").strip()
    if not translated:
        return False
    original = str(text.get("original") or text.get("text") or "").strip()
    if original and translated == original:
        return False
    return True


def _cleanup_page_inpaint_and_rerender(
    *,
    original_image,
    clean_image,
    page_texts: list[dict],
    rendered_image,
    typesetter,
) -> tuple[object, object, bool]:
    cleanup_candidates = [text for text in page_texts if _text_requires_final_cleanup(text)]
    if not cleanup_candidates:
        return clean_image, rendered_image, False
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
            fixed_clean = _apply_white_balloon_near_text_residual_cleanup(original_image, fixed_clean, cleanup_texts)
        if full_cleanup_enabled and _has_white_balloon_text_residual(
            original_image,
            fixed_clean,
            cleanup_texts,
        ):
            limit_mask = _build_page_cleanup_limit_mask(original_image, cleanup_texts)
            fixed_clean, _stats = _apply_post_inpaint_cleanup_timed(
                original_image,
                fixed_clean,
                cleanup_texts,
                limit_mask=limit_mask,
            )
        if not np.any(cv2.absdiff(clean_image, fixed_clean)):
            return clean_image, rendered_image, False
        fixed_rendered = typesetter.render_band_image(fixed_clean, {"texts": page_texts})
        return fixed_clean, fixed_rendered, True
    except Exception:
        return clean_image, rendered_image, False


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
        if not isinstance(text, dict) or text.get("skip_processing"):
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
    structural = bool(profiles & {"white", "white_balloon", "connected_balloon", "top_narration"})
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
    if not _text_has_alnum_or_cjk(raw):
        return False
    tipo = str(text.get("tipo") or "").strip().lower()
    confidence = float(text.get("confidence", text.get("ocr_confidence", 0.0)) or 0.0)
    bright_balloon_context, structural_balloon_context = _koharu_text_bright_balloon_context(text)
    keep_korean_sfx_dialogue = (
        tipo == "sfx"
        and _is_korean_source_language(idioma_origem)
        and _korean_sfx_should_be_translated(raw, confidence)
    )
    keep_structural_korean_sfx_dialogue = (
        tipo == "sfx"
        and _is_korean_source_language(idioma_origem)
        and _has_hangul_text(raw)
        and structural_balloon_context
    )
    if tipo == "sfx" and not keep_korean_sfx_dialogue:
        if not keep_structural_korean_sfx_dialogue:
            return False
    if _is_korean_source_language(idioma_origem) and _looks_like_korean_source_sfx_noise(
        raw,
        bright_balloon_context=bright_balloon_context,
    ):
        return False
    try:
        from ocr.postprocess import is_korean_sfx

        if is_korean_sfx(raw) and not bright_balloon_context and not keep_korean_sfx_dialogue:
            return False
    except Exception:
        pass
    return True


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
            )
        )
    from vision_stack.runtime import _run_koharu_cjk_http_detect_ocr_batch

    return _run_koharu_cjk_http_detect_ocr_batch(
        jobs,
        models_dir=models_dir,
        profile="max",
        idioma_origem=idioma_origem,
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
) -> dict:
    runner = getattr(runtime, "run_koharu_cjk_page", None)
    if callable(runner):
        return runner(image_rgb, str(image_path))
    from vision_stack.runtime import _run_koharu_cjk_http_detect_ocr

    return _run_koharu_cjk_http_detect_ocr(
        image_rgb=image_rgb,
        image_label=str(Path(image_path).resolve()),
        models_dir=models_dir,
        profile="max",
        idioma_origem=idioma_origem,
    )


def _build_precomputed_koharu_cjk_pages(
    strip: VerticalStrip,
    bands: list[Band],
    runtime,
    page_paths: list[Path],
    *,
    models_dir: str = "",
    idioma_origem: str = "en",
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
                        }
                    )
                stats["page_fallback_job_count"] = len(page_jobs)
                try:
                    fallback_results = _run_koharu_cjk_pages_ocr(
                        runtime,
                        page_jobs,
                        models_dir=models_dir,
                        idioma_origem=idioma_origem,
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
            }
        )

    try:
        page_results = _run_koharu_cjk_pages_ocr(
            runtime,
            page_jobs,
            models_dir=models_dir,
            idioma_origem=idioma_origem,
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

                page_result = build_page_result(
                    image_path=_band_image_label(page_number),
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
    fast_white_balloon_count = 0
    fast_local_balloon_count = 0
    remaining_inpaint_blocks = 0
    fast_white_band_count = 0
    fast_local_band_count = 0
    ocr_crop_fallback_attempts = 0
    ocr_crop_fallback_recovered = 0
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
        fast_white_balloon_count += band_fast_white
        fast_local_balloon_count += band_fast_local
        remaining_inpaint_blocks += band_remaining_inpaint
        _merge_counts(fast_white_rejection_reasons, perf.get("fast_white_rejection_reasons"))
        _merge_counts(fast_local_rejection_reasons, perf.get("fast_local_rejection_reasons"))
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
        ocr_full_page_mapped += band_ocr_full_page_mapped
        ocr_crop_fallback_attempts += band_ocr_fallback_attempts
        ocr_crop_fallback_recovered += band_ocr_fallback_recovered
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
                "fast_white_balloon_count": band_fast_white,
                "fast_local_balloon_count": band_fast_local,
                "remaining_inpaint_blocks": band_remaining_inpaint,
                "fast_white_rejection_reasons": dict(perf.get("fast_white_rejection_reasons") or {}),
                "fast_local_rejection_reasons": dict(perf.get("fast_local_rejection_reasons") or {}),
                "ocr_full_page_mapped": band_ocr_full_page_mapped,
                "ocr_crop_fallback_attempts": band_ocr_fallback_attempts,
                "ocr_crop_fallback_recovered": band_ocr_fallback_recovered,
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
        "fast_white_balloon_count": fast_white_balloon_count,
        "fast_local_balloon_count": fast_local_balloon_count,
        "fast_white_band_count": fast_white_band_count,
        "fast_local_band_count": fast_local_band_count,
        "remaining_inpaint_blocks": remaining_inpaint_blocks,
        "fast_white_rejection_reasons": fast_white_rejection_reasons,
        "fast_local_rejection_reasons": fast_local_rejection_reasons,
        "ocr_full_page_mapped": ocr_full_page_mapped,
        "ocr_crop_fallback_attempts": ocr_crop_fallback_attempts,
        "ocr_crop_fallback_recovered": ocr_crop_fallback_recovered,
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
    connected_reasoner_config: dict | None = None,
    models_dir: str = "",
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "traduzai-translator",
    translation_context: dict | None = None,
    chapter_telemetry: dict | None = None,

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
            bands = group_balloons_into_bands(balloons)
        if chapter_telemetry is not None:
            chapter_telemetry["band_count"] = len(bands)
        with _timed(chapter_telemetry, "strip_attach_band_slices"):
            attach_band_slices(strip, bands)

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
        gpu_stage_lock = threading.Lock() if overlap_executor else None
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

        def _process_one_band(idx: int, band: Band, ordered_kwargs: dict, callback):
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
                connected_reasoner_config=connected_reasoner_config,
                band_history=ordered_kwargs["band_history"],
                source_page_number=_source_page_number_for_band(strip, band),
                models_dir=models_dir,
                ollama_host=ollama_host,
                ollama_model=ollama_model,
                translation_context=translation_context,
                precomputed_ocr_page=precomputed_ocr_pages.get(idx),
                ordered_context_after_translate_callback=callback,
                gpu_stage_lock=gpu_stage_lock,
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
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="traduzai-strip-overlap") as executor:
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
                connected_reasoner_config=connected_reasoner_config,
                band_history=ordered_kwargs["band_history"],
                source_page_number=_source_page_number_for_band(strip, band),
                models_dir=models_dir,
                ollama_host=ollama_host,
                ollama_model=ollama_model,
                translation_context=translation_context,
                precomputed_ocr_page=precomputed_ocr_pages.get(idx),
                ordered_context_after_translate_callback=_merge_after_translate,
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

    with _timed(chapter_telemetry, "strip_paste_cleaned"):
        clean_strip_image = _paste_band_attr_into_image(original_strip_image, bands, "cleaned_slice")
    with _timed(chapter_telemetry, "strip_paste_rendered"):
        rendered_strip_image = _paste_band_attr_into_image(original_strip_image, bands, "rendered_slice")
    with _timed(chapter_telemetry, "strip_assign_rendered"):
        strip.image[:, :, :] = rendered_strip_image

    with _timed(chapter_telemetry, "assemble_rendered_pages"):
        output_pages = assemble_output_pages(strip, balloons, target_count=target_count)
    with _timed(chapter_telemetry, "assemble_original_pages"):
        original_pages = assemble_output_pages(
            VerticalStrip(
                image=original_strip_image,
                width=strip.width,
                height=strip.height,
                source_page_breaks=list(strip.source_page_breaks),
                page_x_offsets=list(strip.page_x_offsets),
            ),
            balloons,
            target_count=target_count,
        )
    with _timed(chapter_telemetry, "assemble_clean_pages"):
        clean_pages = assemble_output_pages(
            VerticalStrip(
                image=clean_strip_image,
                width=strip.width,
                height=strip.height,
                source_page_breaks=list(strip.source_page_breaks),
                page_x_offsets=list(strip.page_x_offsets),
            ),
            balloons,
            target_count=target_count,
        )

    # Remapeamento de metadados para project.json
    remap_started = time.perf_counter()
    all_texts: list[dict] = []
    all_vision_blocks: list[dict] = []
    for band in bands:
        if not band.ocr_result:
            continue
        b_y = band.y_top
        # Coleta textos e remapa para coordenadas do strip
        for txt in band.ocr_result.get("texts", []):
            new_txt = dict(txt)
            # bbox é OBRIGATÓRIO — pular texto sem bbox para evitar placeholder [0,0,32,32]
            if not new_txt.get("bbox"):
                continue
            new_txt = _shift_text_geometry_y(new_txt, b_y)
            all_texts.append(new_txt)

        for vb in band.ocr_result.get("_vision_blocks", []):
            new_vb = dict(vb)
            if not new_vb.get("bbox"):
                continue
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

    # Distribuir textos para as páginas por máxima intersecção (não centro-y)
    for txt in all_texts:
        tx1, ty1, tx2, ty2 = txt["bbox"]
        pidx = _assign_text_to_page(ty1, ty2, output_pages)
        if pidx is None:
            continue
        page = output_pages[pidx]
        p_y0 = page.y_top
        local_txt = _shift_text_geometry_y(txt, -p_y0)
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
        local_vb["bbox"] = [vx1, vy1 - p_y0, vx2, vy2 - p_y0]
        page.ocr_result["_vision_blocks"].append(local_vb)
    _add_timing(chapter_telemetry, "assign_metadata_to_pages", time.perf_counter() - assign_started)

    finalize_page_metadata_started = time.perf_counter()
    page_metadata_changed = [False for _ in output_pages]
    for page_index, page in enumerate(output_pages):
        page_metadata_changed[page_index] = _finalize_output_page_ocr_metadata(page, page_index + 1)
    _add_timing(
        chapter_telemetry,
        "finalize_page_ocr_metadata",
        time.perf_counter() - finalize_page_metadata_started,
    )

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
        page.ocr_result["page_profile"] = page.page_profile
        page.inpaint_blocks = [
            block
            for vb in page.ocr_result.get("_vision_blocks", [])
            for block in [_inpaint_block_from_vision_block(vb)]
            if block is not None
        ]
    _add_timing(chapter_telemetry, "attach_page_profiles", time.perf_counter() - attach_profile_started)

    cleanup_started = time.perf_counter()
    for page_index, (page, original_page, clean_page) in enumerate(zip(output_pages, original_pages, clean_pages)):
        page_texts = []
        if isinstance(page.text_layers, dict):
            page_texts = [text for text in list(page.text_layers.get("texts") or []) if isinstance(text, dict)]
        fixed_clean, fixed_rendered, did_fix = _cleanup_page_inpaint_and_rerender(
            original_image=original_page.image,
            clean_image=clean_page.image,
            page_texts=page_texts,
            rendered_image=page.image,
            typesetter=typesetter,
        )
        page.original_image = original_page.image
        page.inpainted_image = fixed_clean
        if did_fix:
            page.image = fixed_rendered
        elif page_index < len(page_metadata_changed) and page_metadata_changed[page_index]:
            try:
                page.image = typesetter.render_band_image(fixed_clean, {"texts": page_texts})
            except Exception:
                page.image = fixed_rendered
    _add_timing(chapter_telemetry, "page_cleanup_rerender", time.perf_counter() - cleanup_started)

    if _macro_ocr_shadow_enabled() and output_pages:
        with _timed(chapter_telemetry, "macro_ocr_shadow"):
            macro_shadow = _run_macro_ocr_shadow(
                output_pages,
                runtime,
                idioma_origem=idioma_origem,
            )
        output_pages[0].page_profile["macro_ocr_shadow"] = macro_shadow
        output_pages[0].ocr_result["page_profile"] = output_pages[0].page_profile

    with _timed(chapter_telemetry, "write_translated_pages"):
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, page in enumerate(output_pages):
            page.path = output_dir / f"{i + 1:03d}.jpg"
            cv2.imwrite(str(page.path), page.image, [cv2.IMWRITE_JPEG_QUALITY, 92])

    if chapter_telemetry is not None:
        chapter_telemetry["wall_total_sec"] = round(time.perf_counter() - run_started, 4)
        chapter_telemetry["output_page_count"] = len(output_pages)
        chapter_telemetry["text_count"] = sum(
            len((page.text_layers or {}).get("texts") or [])
            for page in output_pages
            if isinstance(page.text_layers, dict)
        )

    return output_pages
