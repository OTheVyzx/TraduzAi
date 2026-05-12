"""Helpers for Macro OCR shadow analysis.

This module does not replace the current OCR path. It estimates the mapping
risk of doing OCR at page/macro-window granularity and remapping text lines
back to strip bands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class MacroOcrLine:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float
    page_number: int


@dataclass(frozen=True)
class BandWindow:
    band_index: int
    page_number: int
    y_top: int
    y_bottom: int


@dataclass(frozen=True)
class MacroLineMapping:
    line: MacroOcrLine
    band_index: int | None
    status: str
    overlap_ratio: float


@dataclass(frozen=True)
class MacroOcrShadowReport:
    current_ocr_seconds: float
    estimated_macro_ocr_seconds: float
    estimated_savings_seconds: float
    current_ocr_band_calls: int
    macro_window_count: int
    text_line_count: int
    mapped_line_count: int
    fallback_line_count: int
    missing_line_count: int
    missing_text_rate: float
    fallback_rate: float
    wrong_band_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_band_windows_from_project(project: dict[str, Any]) -> list[BandWindow]:
    pages = list(project.get("paginas") or [])
    page_ranges = []
    for index, page in enumerate(pages):
        profile = page.get("page_profile") or {}
        page_ranges.append(
            {
                "page_number": int(page.get("numero") or index + 1),
                "strip_top": _int(profile.get("y_in_strip_top")),
                "strip_bottom": _int(profile.get("y_in_strip_bottom")),
            }
        )

    summary = _find_strip_perf_summary(project)
    windows: list[BandWindow] = []
    for entry in summary.get("entries") or []:
        y_top = _int(entry.get("y_top"))
        y_bottom = _int(entry.get("y_bottom"))
        page_range = _best_page_range_for_band(page_ranges, y_top, y_bottom)
        if page_range is None:
            continue
        windows.append(
            BandWindow(
                band_index=_int(entry.get("band_index")),
                page_number=page_range["page_number"],
                y_top=max(0, y_top - page_range["strip_top"]),
                y_bottom=max(0, y_bottom - page_range["strip_top"]),
            )
        )
    return windows


def extract_macro_lines_from_project(project: dict[str, Any]) -> list[MacroOcrLine]:
    lines: list[MacroOcrLine] = []
    for page_index, page in enumerate(project.get("paginas") or []):
        page_number = _int(page.get("numero"), page_index + 1)
        for text in page.get("text_layers") or []:
            if not isinstance(text, dict):
                continue
            bbox = text.get("bbox") or text.get("balloon_bbox") or text.get("render_bbox")
            normalized_bbox = _bbox_tuple(bbox)
            if normalized_bbox is None:
                continue
            raw_text = str(text.get("original") or text.get("text") or text.get("translated") or "")
            if not raw_text.strip():
                continue
            lines.append(
                MacroOcrLine(
                    text=raw_text.strip(),
                    bbox=normalized_bbox,
                    confidence=_float(text.get("ocr_confidence", text.get("confidence"))),
                    page_number=page_number,
                )
            )
    return lines


def collect_page_ocr_blocks(page: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    raw_blocks = list(page.get("inpaint_blocks") or [])
    if raw_blocks:
        for block in raw_blocks:
            if not isinstance(block, dict):
                continue
            bbox = _bbox_tuple(block.get("bbox"))
            if bbox is None:
                continue
            blocks.append({"bbox": list(bbox), "confidence": _float(block.get("confidence"), 1.0)})
        return blocks

    for text in page.get("text_layers") or []:
        if not isinstance(text, dict):
            continue
        bbox = _bbox_tuple(text.get("balloon_bbox") or text.get("bbox") or text.get("render_bbox"))
        if bbox is None:
            continue
        blocks.append({"bbox": list(bbox), "confidence": _float(text.get("confidence"), 1.0)})
    return blocks


def map_macro_lines_to_bands(
    lines: list[MacroOcrLine],
    bands: list[BandWindow],
) -> list[MacroLineMapping]:
    mappings: list[MacroLineMapping] = []
    for line in lines:
        same_page = [band for band in bands if band.page_number == line.page_number]
        best_band: BandWindow | None = None
        best_ratio = 0.0
        overlapping_count = 0
        for band in same_page:
            ratio = _vertical_overlap_ratio(line.bbox, band)
            if ratio > 0:
                overlapping_count += 1
            if ratio > best_ratio:
                best_ratio = ratio
                best_band = band

        if best_band is None:
            mappings.append(MacroLineMapping(line, None, "missing", 0.0))
            continue

        center_y = (line.bbox[1] + line.bbox[3]) / 2.0
        center_inside = best_band.y_top <= center_y <= best_band.y_bottom
        crosses_boundary = overlapping_count > 1
        status = "fallback" if crosses_boundary or not center_inside else "mapped"
        mappings.append(
            MacroLineMapping(
                line=line,
                band_index=best_band.band_index,
                status=status,
                overlap_ratio=round(best_ratio, 4),
            )
        )
    return mappings


def estimate_macro_ocr_shadow(project: dict[str, Any]) -> MacroOcrShadowReport:
    summary = _find_strip_perf_summary(project)
    entries = [entry for entry in summary.get("entries") or [] if isinstance(entry, dict)]
    current_ocr_seconds = _float((summary.get("durations_sec") or {}).get("ocr"))
    if current_ocr_seconds <= 0:
        current_ocr_seconds = sum(_float((entry.get("durations_sec") or {}).get("ocr")) for entry in entries)
    current_ocr_band_calls = sum(
        1 for entry in entries if _float((entry.get("durations_sec") or {}).get("ocr")) > 0
    )

    bands = collect_band_windows_from_project(project)
    lines = extract_macro_lines_from_project(project)
    mappings = map_macro_lines_to_bands(lines, bands)

    pages_with_lines = {line.page_number for line in lines}
    pages_with_bands = {band.page_number for band in bands}
    macro_window_count = len(pages_with_lines or pages_with_bands)
    if current_ocr_band_calls > 0 and macro_window_count > 0:
        estimated_macro_ocr_seconds = current_ocr_seconds * (
            macro_window_count / float(current_ocr_band_calls)
        )
    else:
        estimated_macro_ocr_seconds = current_ocr_seconds

    text_line_count = len(lines)
    missing_line_count = sum(1 for item in mappings if item.status == "missing")
    fallback_line_count = sum(1 for item in mappings if item.status == "fallback")
    mapped_line_count = sum(1 for item in mappings if item.status == "mapped")

    return MacroOcrShadowReport(
        current_ocr_seconds=round(current_ocr_seconds, 4),
        estimated_macro_ocr_seconds=round(estimated_macro_ocr_seconds, 4),
        estimated_savings_seconds=round(max(0.0, current_ocr_seconds - estimated_macro_ocr_seconds), 4),
        current_ocr_band_calls=current_ocr_band_calls,
        macro_window_count=macro_window_count,
        text_line_count=text_line_count,
        mapped_line_count=mapped_line_count,
        fallback_line_count=fallback_line_count,
        missing_line_count=missing_line_count,
        missing_text_rate=_rate(missing_line_count, text_line_count),
        fallback_rate=_rate(fallback_line_count, text_line_count),
        wrong_band_rate=0.0,
    )


def compare_aligned_macro_ocr_texts(
    baseline_texts: list[Any],
    macro_texts: list[Any],
) -> dict[str, Any]:
    total = max(len(baseline_texts), len(macro_texts))
    missing_count = 0
    exact_match_count = 0
    different_count = 0
    material_different_count = 0
    fallback_required_count = 0
    difference_kind_counts: dict[str, int] = {}
    acceptable_difference_kinds = {
        "line_marker_artifact",
        "line_marker_minor_variation",
        "minor_ocr_variation",
        "numeric_confusable_variation",
        "episode_marker_variation",
    }
    for index in range(total):
        baseline = _extract_text(baseline_texts[index]) if index < len(baseline_texts) else ""
        macro = _extract_text(macro_texts[index]) if index < len(macro_texts) else ""
        if baseline and not macro:
            missing_count += 1
        if baseline and macro:
            difference_kind = classify_ocr_text_difference(baseline, macro)
            if difference_kind == "exact":
                exact_match_count += 1
            else:
                different_count += 1
                difference_kind_counts[difference_kind] = (
                    difference_kind_counts.get(difference_kind, 0) + 1
                )
                if difference_kind == "material":
                    material_different_count += 1
                if difference_kind in {"material", "numeric_token_change"}:
                    fallback_required_count += 1
    return {
        "total": total,
        "missing_count": missing_count,
        "different_count": different_count,
        "fallback_resolved_different_count": max(0, different_count - fallback_required_count),
        "exact_match_count": exact_match_count,
        "line_marker_artifact_count": difference_kind_counts.get("line_marker_artifact", 0),
        "line_marker_minor_variation_count": difference_kind_counts.get(
            "line_marker_minor_variation", 0
        ),
        "minor_ocr_variation_count": difference_kind_counts.get("minor_ocr_variation", 0),
        "numeric_confusable_variation_count": difference_kind_counts.get(
            "numeric_confusable_variation", 0
        ),
        "episode_marker_variation_count": difference_kind_counts.get(
            "episode_marker_variation", 0
        ),
        "numeric_token_change_count": difference_kind_counts.get("numeric_token_change", 0),
        "material_different_count": material_different_count,
        "fallback_required_count": fallback_required_count,
        "acceptable_variation_count": sum(
            difference_kind_counts.get(kind, 0) for kind in acceptable_difference_kinds
        ),
        "missing_text_rate": _rate(missing_count, total),
        "different_text_rate": _rate(different_count, total),
        "fallback_resolved_different_text_rate": _rate(
            max(0, different_count - fallback_required_count), total
        ),
        "material_different_text_rate": _rate(material_different_count, total),
        "fallback_required_text_rate": _rate(fallback_required_count, total),
        "exact_match_rate": _rate(exact_match_count, total),
        "difference_kind_counts": difference_kind_counts,
    }


def classify_ocr_text_difference(baseline: str, macro: str) -> str:
    if _normalize_for_compare(baseline) == _normalize_for_compare(macro):
        return "exact"
    macro_without_markers = _remove_probable_line_marker_numbers(macro)
    if (
        _normalize_for_compare(baseline) == _normalize_for_compare(macro_without_markers)
        and not _contains_numeric_measurement(macro)
    ):
        return "line_marker_artifact"
    if (
        _normalize_for_compare(macro) != _normalize_for_compare(macro_without_markers)
        and
        _digit_tokens(baseline) == _digit_tokens(macro_without_markers)
        and _is_minor_ocr_variation(baseline, macro_without_markers)
        and not _contains_numeric_measurement(macro)
    ):
        return "line_marker_minor_variation"
    if _is_episode_marker_variation(baseline, macro):
        return "episode_marker_variation"
    if _is_confusable_numeric_variation(baseline, macro):
        return "numeric_confusable_variation"
    if _digit_tokens(baseline) != _digit_tokens(macro):
        return "numeric_token_change"
    if _is_minor_ocr_variation(baseline, macro):
        return "minor_ocr_variation"
    return "material"


def estimate_macro_ocr_fallback_cost(
    *,
    block_count: int,
    macro_window_count: int,
    material_different_count: int,
    fallback_required_count: int | None = None,
) -> dict[str, Any]:
    fallback_source = material_different_count if fallback_required_count is None else fallback_required_count
    fallback_call_count = max(0, int(fallback_source))
    effective_ocr_call_count = max(0, int(macro_window_count)) + fallback_call_count
    total_blocks = max(0, int(block_count))
    return {
        "fallback_call_count": fallback_call_count,
        "effective_ocr_call_count": effective_ocr_call_count,
        "fallback_adjusted_window_reduction_rate": _rate(
            max(0, total_blocks - effective_ocr_call_count),
            total_blocks,
        ),
    }


def recognize_macro_ocr_windows(
    ocr_engine: Any,
    image_rgb: Any,
    blocks: list[dict[str, Any]],
    *,
    window_mode: str = "page",
    crop_fallback_max: int = 0,
    window_max_blocks: int = 8,
    window_merge_gap: int = 180,
    window_padding: int = 48,
) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    if window_mode not in {"page", "band-groups"}:
        raise ValueError(f"Unsupported Macro OCR window mode: {window_mode}")

    if window_mode == "page":
        block_objects = [_block_object(block) for block in blocks]
        macro_records = ocr_engine.recognize_blocks_from_page(
            image_rgb,
            block_objects,
            allow_sparse_mapping=True,
            crop_fallback_max=crop_fallback_max,
        )
        ocr_stats = dict(getattr(ocr_engine, "_last_recognize_blocks_stats", None) or {})
        ocr_stats["macro_window_count"] = 1
        return list(macro_records or []), ocr_stats, [
            {"block_indices": list(range(len(blocks))), "bbox": _image_bbox(image_rgb)}
        ]

    windows = _build_band_group_windows(
        blocks,
        image_rgb,
        max_blocks=window_max_blocks,
        merge_gap=window_merge_gap,
        padding=window_padding,
    )
    macro_texts: list[Any] = [""] * len(blocks)
    aggregate_stats = {
        "block_count": len(blocks),
        "macro_window_count": len(windows),
        "crop_fallback_attempts": 0,
        "crop_fallback_recovered": 0,
        "full_page_mapped": 0,
        "full_page_mapping_failed_windows": 0,
    }
    window_reports: list[dict[str, Any]] = []

    for window in windows:
        x1, y1, x2, y2 = window["bbox"]
        crop = image_rgb[y1:y2, x1:x2]
        window_blocks = [
            _block_object(
                {
                    **blocks[index],
                    "bbox": _translate_bbox(blocks[index]["bbox"], dx=-x1, dy=-y1),
                }
            )
            for index in window["indices"]
        ]
        records = list(
            ocr_engine.recognize_blocks_from_page(
                crop,
                window_blocks,
                allow_sparse_mapping=True,
                crop_fallback_max=crop_fallback_max,
            )
            or []
        )
        stats = dict(getattr(ocr_engine, "_last_recognize_blocks_stats", None) or {})
        aggregate_stats["crop_fallback_attempts"] += _int(stats.get("crop_fallback_attempts"))
        aggregate_stats["crop_fallback_recovered"] += _int(stats.get("crop_fallback_recovered"))
        aggregate_stats["full_page_mapped"] += _int(stats.get("full_page_mapped"))
        if stats.get("full_page_mapping_failed"):
            aggregate_stats["full_page_mapping_failed_windows"] += 1

        for offset, original_index in enumerate(window["indices"]):
            if offset < len(records):
                macro_texts[original_index] = records[offset]
        window_reports.append(
            {
                "block_indices": window["indices"],
                "bbox": list(window["bbox"]),
                "ocr_stats": stats,
            }
        )

    return macro_texts, aggregate_stats, window_reports


def _find_strip_perf_summary(project: dict[str, Any]) -> dict[str, Any]:
    for page in project.get("paginas") or []:
        profile = page.get("page_profile") or {}
        summary = profile.get("strip_perf_summary")
        if isinstance(summary, dict):
            return summary
    return {}


def _best_page_range_for_band(
    page_ranges: list[dict[str, int]],
    y_top: int,
    y_bottom: int,
) -> dict[str, int] | None:
    best = None
    best_overlap = 0
    for page_range in page_ranges:
        overlap = max(
            0,
            min(y_bottom, page_range["strip_bottom"]) - max(y_top, page_range["strip_top"]),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best = page_range
    return best


def _vertical_overlap_ratio(bbox: tuple[int, int, int, int], band: BandWindow) -> float:
    y_top = int(bbox[1])
    y_bottom = int(bbox[3])
    overlap = max(0, min(y_bottom, band.y_bottom) - max(y_top, band.y_top))
    line_height = max(1, y_bottom - y_top)
    return overlap / float(line_height)


def _bbox_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / float(total), 4)


def _extract_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("original") or value.get("translated") or "").strip()
    return str(value or "").strip()


def _normalize_for_compare(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _digit_tokens(value: str) -> list[str]:
    import re

    return re.findall(r"\d+", str(value or ""))


def _normalize_confusable_numeric(value: str) -> str:
    import re

    text = str(value or "").lower().translate(str.maketrans({"o": "0", "i": "1", "l": "1", "s": "5", "z": "2"}))
    return re.sub(r"[^a-z0-9]+", "", text)


def _is_confusable_numeric_variation(baseline: str, macro: str) -> bool:
    return (
        _digit_tokens(baseline) != _digit_tokens(macro)
        and _normalize_confusable_numeric(baseline) == _normalize_confusable_numeric(macro)
    )


def _is_episode_marker_variation(baseline: str, macro: str) -> bool:
    import re

    baseline_text = str(baseline or "").strip()
    macro_text = str(macro or "").strip()
    if not re.match(r"(?i)^ep\b", baseline_text) or not re.match(r"(?i)^ep\.?\s*\d+\b", macro_text):
        return False
    macro_without_episode_number = re.sub(r"(?i)^ep\.?\s*\d+\b", "EP", macro_text, count=1)
    return _normalize_for_compare(baseline_text) == _normalize_for_compare(macro_without_episode_number)


def _contains_numeric_measurement(value: str) -> bool:
    import re

    return bool(
        re.search(
            r"\b\d+(?:[.,:]\d+)*\s*(?:degrees?|celsius|years?|mins?|minutes?|secs?|seconds?|%)\b",
            str(value or ""),
            flags=re.IGNORECASE,
        )
    )


def _is_minor_ocr_variation(baseline: str, macro: str) -> bool:
    import difflib

    baseline_norm = _normalize_for_compare(baseline)
    macro_norm = _normalize_for_compare(macro)
    if not baseline_norm or not macro_norm:
        return False
    ratio = difflib.SequenceMatcher(None, baseline_norm, macro_norm).ratio()
    max_len = max(len(baseline_norm), len(macro_norm))
    len_delta = abs(len(baseline_norm) - len(macro_norm))
    return ratio >= 0.93 and len_delta <= max(3, int(0.08 * max_len))


def _remove_probable_line_marker_numbers(value: str) -> str:
    import re

    # Paddle occasionally inserts detached OCR line ids such as "67" between
    # words in larger windows. Single digits are kept because they often carry
    # chapter/page meaning in manga text.
    return re.sub(r"\b\d{2,}\b", " ", str(value or ""))


def _build_band_group_windows(
    blocks: list[dict[str, Any]],
    image_rgb: Any,
    *,
    max_blocks: int,
    merge_gap: int,
    padding: int,
) -> list[dict[str, Any]]:
    indexed_blocks = sorted(
        [(index, _required_bbox_tuple(block.get("bbox"))) for index, block in enumerate(blocks)],
        key=lambda item: (item[1][1], item[1][0]),
    )
    groups: list[list[int]] = []
    current: list[int] = []
    current_bbox: tuple[int, int, int, int] | None = None
    max_blocks = max(1, int(max_blocks))
    merge_gap = max(0, int(merge_gap))

    for index, bbox in indexed_blocks:
        if not current or current_bbox is None:
            current = [index]
            current_bbox = bbox
            continue
        gap = max(0, bbox[1] - current_bbox[3])
        if len(current) < max_blocks and gap <= merge_gap:
            current.append(index)
            current_bbox = _union_bbox([current_bbox, bbox])
        else:
            groups.append(current)
            current = [index]
            current_bbox = bbox
    if current:
        groups.append(current)

    return [
        {
            "indices": group,
            "bbox": _padded_window_bbox(
                [_required_bbox_tuple(blocks[index]["bbox"]) for index in group],
                image_rgb,
                padding,
            ),
        }
        for group in groups
    ]


def _block_object(block: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        xyxy=tuple(block["bbox"]),
        confidence=float(block.get("confidence", 1.0) or 1.0),
    )


def _required_bbox_tuple(value: Any) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in value]
    return x1, y1, x2, y2


def _translate_bbox(value: Any, *, dx: int, dy: int) -> list[int]:
    x1, y1, x2, y2 = _required_bbox_tuple(value)
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def _union_bbox(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _padded_window_bbox(
    bboxes: list[tuple[int, int, int, int]],
    image_rgb: Any,
    padding: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _union_bbox(bboxes)
    image_h, image_w = _image_size(image_rgb)
    padding = max(0, int(padding))
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_w, x2 + padding),
        min(image_h, y2 + padding),
    )


def _image_size(image_rgb: Any) -> tuple[int, int]:
    shape = getattr(image_rgb, "shape", None)
    if not shape or len(shape) < 2:
        return 0, 0
    return int(shape[0]), int(shape[1])


def _image_bbox(image_rgb: Any) -> list[int]:
    image_h, image_w = _image_size(image_rgb)
    return [0, 0, image_w, image_h]


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
