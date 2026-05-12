"""Run actual Macro OCR shadow against an existing pipeline output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(pipeline_root))

from pipeline.ocr.macro_ocr import (
    collect_page_ocr_blocks,
    compare_aligned_macro_ocr_texts,
    estimate_macro_ocr_fallback_cost,
    recognize_macro_ocr_windows,
)


DEFAULT_MAX_MISSING_TEXT_RATE = 0.02
DEFAULT_MAX_DIFFERENT_TEXT_RATE = 0.25
DEFAULT_MAX_FALLBACK_RATE = 0.15


def evaluate_actual_macro_ocr_shadow(
    output_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    ocr_engine: Any | None = None,
    image_loader: Callable[[Path], Any] | None = None,
    max_pages: int | None = None,
    lang: str = "en",
    max_missing_text_rate: float = DEFAULT_MAX_MISSING_TEXT_RATE,
    max_different_text_rate: float = DEFAULT_MAX_DIFFERENT_TEXT_RATE,
    max_fallback_rate: float = DEFAULT_MAX_FALLBACK_RATE,
    sample_limit: int = 5,
    crop_fallback_max: int = 0,
    window_mode: str = "page",
    window_max_blocks: int = 8,
    window_merge_gap: int = 180,
    window_padding: int = 48,
    min_window_reduction_rate: float = 0.0,
    min_fallback_adjusted_reduction_rate: float = 0.0,
    gate_on_fallback_resolved_text: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    project_path = output_path / "project.json"
    if not project_path.exists():
        return _write_result(_blocked(output_path, ["missing project.json"]), out_dir)

    project = _load_json(project_path)
    pages = list(project.get("paginas") or [])
    if max_pages is not None:
        pages = pages[: max(0, int(max_pages))]
    if not pages:
        return _write_result(_blocked(output_path, ["project has no pages"]), out_dir)

    if ocr_engine is None:
        ocr_engine = _create_ocr_engine(lang=lang)
    if image_loader is None:
        image_loader = _load_image_rgb

    started_at = time.perf_counter()
    pages_processed = 0
    blocks_processed = 0
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
    page_reports = []

    for page in pages:
        blocks = collect_page_ocr_blocks(page)
        if not blocks:
            continue
        image_path = _resolve_page_image_path(output_path, page)
        image_rgb = image_loader(image_path)
        if image_rgb is None:
            page_reports.append(
                {
                    "page_number": page.get("numero"),
                    "status": "missing_image",
                    "path": str(image_path),
                }
            )
            continue

        macro_texts, ocr_stats, window_reports = recognize_macro_ocr_windows(
            ocr_engine,
            image_rgb,
            blocks,
            window_mode=window_mode,
            crop_fallback_max=crop_fallback_max,
            window_max_blocks=window_max_blocks,
            window_merge_gap=window_merge_gap,
            window_padding=window_padding,
        )
        baseline_texts = list(page.get("text_layers") or [])
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
        totals["macro_window_count"] += _int(ocr_stats.get("macro_window_count"))
        pages_processed += 1
        blocks_processed += len(blocks)
        totals["crop_fallback_attempts"] += _int(ocr_stats.get("crop_fallback_attempts"))
        totals["crop_fallback_recovered"] += _int(ocr_stats.get("crop_fallback_recovered"))
        page_reports.append(
            {
                "page_number": page.get("numero"),
                "blocks": len(blocks),
                **compare,
                "ocr_stats": ocr_stats,
                "windows": window_reports,
                "samples": _sample_text_diffs(baseline_texts, macro_texts, limit=sample_limit),
            }
        )

    missing_text_rate = _rate(totals["missing_count"], totals["total"])
    different_text_rate = _rate(totals["different_count"], totals["total"])
    fallback_resolved_different_text_rate = _rate(
        totals["fallback_resolved_different_count"], totals["total"]
    )
    text_quality_gate_rate = (
        fallback_resolved_different_text_rate
        if gate_on_fallback_resolved_text
        else different_text_rate
    )
    material_different_text_rate = _rate(totals["material_different_count"], totals["total"])
    fallback_required_text_rate = _rate(totals["fallback_required_count"], totals["total"])
    exact_match_rate = _rate(totals["exact_match_count"], totals["total"])
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
    status = (
        "PASS"
        if (
            missing_text_rate <= max_missing_text_rate
            and text_quality_gate_rate <= max_different_text_rate
            and fallback_rate <= max_fallback_rate
            and window_reduction_rate >= min_window_reduction_rate
            and fallback_cost["fallback_adjusted_window_reduction_rate"]
            >= min_fallback_adjusted_reduction_rate
        )
        else "FAIL"
    )
    reasons = []
    if status == "PASS":
        reasons.append("actual Macro OCR shadow stayed within text quality thresholds")
    else:
        if missing_text_rate > max_missing_text_rate:
            reasons.append(
                f"missing text rate {missing_text_rate:.2%} exceeds "
                f"{max_missing_text_rate:.2%}"
            )
        if text_quality_gate_rate > max_different_text_rate:
            text_rate_label = (
                "fallback-resolved different text rate"
                if gate_on_fallback_resolved_text
                else "different text rate"
            )
            reasons.append(
                f"{text_rate_label} {text_quality_gate_rate:.2%} exceeds "
                f"{max_different_text_rate:.2%}"
            )
        if fallback_rate > max_fallback_rate:
            reasons.append(
                f"fallback rate {fallback_rate:.2%} exceeds "
                f"{max_fallback_rate:.2%}"
            )
        if window_reduction_rate < min_window_reduction_rate:
            reasons.append(
                f"window reduction rate {window_reduction_rate:.2%} is below "
                f"{min_window_reduction_rate:.2%}"
            )
        if (
            fallback_cost["fallback_adjusted_window_reduction_rate"]
            < min_fallback_adjusted_reduction_rate
        ):
            reasons.append(
                "fallback-adjusted window reduction rate "
                f"{fallback_cost['fallback_adjusted_window_reduction_rate']:.2%} "
                f"is below {min_fallback_adjusted_reduction_rate:.2%}"
            )

    result = {
        "source_path": str(output_path),
        "gate": {
            "name": "macro_ocr_actual_shadow",
            "status": status,
            "reasons": reasons,
            "runtime_seconds": round(time.perf_counter() - started_at, 4),
            "pages_processed": pages_processed,
            "blocks_processed": blocks_processed,
            "text_line_count": totals["total"],
            "missing_count": totals["missing_count"],
            "different_count": totals["different_count"],
            "fallback_resolved_different_count": totals[
                "fallback_resolved_different_count"
            ],
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
            "window_mode": window_mode,
            "window_reduction_rate": window_reduction_rate,
            "min_window_reduction_rate": float(min_window_reduction_rate),
            "min_fallback_adjusted_reduction_rate": float(
                min_fallback_adjusted_reduction_rate
            ),
            "missing_text_rate": missing_text_rate,
            "different_text_rate": different_text_rate,
            "fallback_resolved_different_text_rate": fallback_resolved_different_text_rate,
            "text_quality_gate_rate": text_quality_gate_rate,
            "gate_on_fallback_resolved_text": bool(gate_on_fallback_resolved_text),
            "material_different_text_rate": material_different_text_rate,
            "fallback_required_text_rate": fallback_required_text_rate,
            "exact_match_rate": exact_match_rate,
            "fallback_rate": fallback_rate,
            "max_missing_text_rate": float(max_missing_text_rate),
            "max_different_text_rate": float(max_different_text_rate),
            "max_fallback_rate": float(max_fallback_rate),
            "page_reports": page_reports,
        },
    }
    return _write_result(result, out_dir)


def _create_ocr_engine(*, lang: str):
    from vision_stack.ocr import OCREngine

    return OCREngine(model="paddleocr", device="cuda", half=True, batch_size=8, lang=lang)


def _recognize_page_macro_texts(
    ocr_engine: Any,
    image_rgb: Any,
    blocks: list[dict[str, Any]],
    *,
    window_mode: str,
    crop_fallback_max: int,
    window_max_blocks: int,
    window_merge_gap: int,
    window_padding: int,
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


def _build_band_group_windows(
    blocks: list[dict[str, Any]],
    image_rgb: Any,
    *,
    max_blocks: int,
    merge_gap: int,
    padding: int,
) -> list[dict[str, Any]]:
    indexed_blocks = sorted(
        [(index, _bbox_tuple(block.get("bbox"))) for index, block in enumerate(blocks)],
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
            "bbox": _padded_window_bbox([_bbox_tuple(blocks[index]["bbox"]) for index in group], image_rgb, padding),
        }
        for group in groups
    ]


def _block_object(block: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        xyxy=tuple(block["bbox"]),
        confidence=float(block.get("confidence", 1.0) or 1.0),
    )


def _bbox_tuple(value: Any) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in value]
    return x1, y1, x2, y2


def _translate_bbox(value: Any, *, dx: int, dy: int) -> list[int]:
    x1, y1, x2, y2 = _bbox_tuple(value)
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


def _load_image_rgb(path: Path):
    import cv2

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _resolve_page_image_path(output_path: Path, page: dict[str, Any]) -> Path:
    raw = str(page.get("arquivo_original") or "")
    if raw:
        candidate = output_path / raw
        if candidate.exists():
            return candidate
        absolute = Path(raw)
        if absolute.exists():
            return absolute
    numero = int(page.get("numero") or 0)
    return output_path / "originals" / f"{numero:03d}.jpg"


def _blocked(output_path: Path, reasons: list[str]) -> dict[str, Any]:
    return {
        "source_path": str(output_path),
        "gate": {
            "name": "macro_ocr_actual_shadow",
            "status": "BLOCK",
            "reasons": reasons,
            "runtime_seconds": 0.0,
            "pages_processed": 0,
            "blocks_processed": 0,
            "text_line_count": 0,
            "missing_count": 0,
            "different_count": 0,
            "exact_match_count": 0,
            "crop_fallback_attempts": 0,
            "crop_fallback_recovered": 0,
            "missing_text_rate": 0.0,
            "different_text_rate": 0.0,
            "exact_match_rate": 0.0,
            "fallback_rate": 0.0,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_result(result: dict[str, Any], out_dir: str | Path | None) -> dict[str, Any]:
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / float(total), 4)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sample_text_diffs(
    baseline_texts: list[Any],
    macro_texts: list[Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    total = max(len(baseline_texts), len(macro_texts))
    for index in range(total):
        baseline = _extract_text(baseline_texts[index]) if index < len(baseline_texts) else ""
        macro = _extract_text(macro_texts[index]) if index < len(macro_texts) else ""
        status = ""
        if baseline and not macro:
            status = "missing"
        elif macro and not baseline:
            status = "extra"
        elif baseline and macro and _normalize_for_compare(baseline) != _normalize_for_compare(macro):
            status = "different"
        if not status:
            continue
        samples.append(
            {
                "index": index,
                "status": status,
                "baseline": baseline,
                "macro": macro,
            }
        )
        if len(samples) >= max(0, int(limit)):
            break
    return samples


def _extract_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("original") or value.get("translated") or "").strip()
    return str(value or "").strip()


def _normalize_for_compare(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--max-missing-text-rate", type=float, default=DEFAULT_MAX_MISSING_TEXT_RATE)
    parser.add_argument("--max-different-text-rate", type=float, default=DEFAULT_MAX_DIFFERENT_TEXT_RATE)
    parser.add_argument("--max-fallback-rate", type=float, default=DEFAULT_MAX_FALLBACK_RATE)
    parser.add_argument("--crop-fallback-max", type=int, default=0)
    parser.add_argument("--window-mode", choices=["page", "band-groups"], default="page")
    parser.add_argument("--window-max-blocks", type=int, default=8)
    parser.add_argument("--window-merge-gap", type=int, default=180)
    parser.add_argument("--window-padding", type=int, default=48)
    parser.add_argument("--min-window-reduction-rate", type=float, default=0.0)
    parser.add_argument("--min-fallback-adjusted-reduction-rate", type=float, default=0.0)
    parser.add_argument(
        "--gate-on-fallback-resolved-text",
        action="store_true",
        help=(
            "Use different text rate after required fallbacks are assumed to use "
            "the baseline crop OCR result. Keeps the default conservative gate unchanged."
        ),
    )
    args = parser.parse_args(argv)

    result = evaluate_actual_macro_ocr_shadow(
        args.output_dir,
        args.out,
        max_pages=args.max_pages,
        lang=args.lang,
        max_missing_text_rate=args.max_missing_text_rate,
        max_different_text_rate=args.max_different_text_rate,
        max_fallback_rate=args.max_fallback_rate,
        crop_fallback_max=args.crop_fallback_max,
        window_mode=args.window_mode,
        window_max_blocks=args.window_max_blocks,
        window_merge_gap=args.window_merge_gap,
        window_padding=args.window_padding,
        min_window_reduction_rate=args.min_window_reduction_rate,
        min_fallback_adjusted_reduction_rate=args.min_fallback_adjusted_reduction_rate,
        gate_on_fallback_resolved_text=args.gate_on_fallback_resolved_text,
    )
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
