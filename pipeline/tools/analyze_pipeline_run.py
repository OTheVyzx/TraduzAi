"""Analyze a TraduzAi pipeline output directory.

This module intentionally reads existing artifacts only. It does not import the
heavy OCR/inpaint stack, so it can be used as a cheap performance gate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BandMetric:
    band_index: int
    durations_sec: dict[str, float] = field(default_factory=dict)
    total_sec: float = 0.0
    text_count: int = 0
    remaining_inpaint_blocks: int = 0
    y_top: int | None = None
    y_bottom: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class RunMetrics:
    source_path: str
    run_name: str
    total_seconds: float
    pages: int
    text_count: int
    text_layers: int
    inpaint_blocks_exported: int
    band_count: int
    durations_sec: dict[str, float] = field(default_factory=dict)
    bands: list[BandMetric] = field(default_factory=list)
    skip_candidate_count: int = 0


def load_run_metrics(output_dir: str | Path, run_name: str | None = None) -> RunMetrics:
    """Load metrics from a pipeline output directory or analysis directory."""

    output_path = Path(output_dir)
    if output_path.is_file():
        payload = _load_json(output_path)
        return _metrics_from_analysis_payload(payload, output_path, run_name=run_name)

    project_path = output_path / "project.json"
    if project_path.exists():
        return _metrics_from_project(project_path)

    metrics_path = output_path / "metrics.json"
    if metrics_path.exists():
        payload = _load_json(metrics_path)
        return _metrics_from_analysis_payload(payload, output_path, run_name=run_name)

    raise FileNotFoundError(
        f"Could not find project.json or metrics.json under {output_path}"
    )


def summarize_stages(metrics: RunMetrics) -> dict[str, float]:
    return dict(metrics.durations_sec)


def rank_bands(metrics: RunMetrics, stage: str, limit: int = 10) -> list[BandMetric]:
    ranked = sorted(
        metrics.bands,
        key=lambda band: (band.durations_sec.get(stage, 0.0), band.total_sec),
        reverse=True,
    )
    return ranked[:limit]


def build_summary(metrics: RunMetrics, *, limit: int = 10) -> dict[str, Any]:
    return {
        "source_path": metrics.source_path,
        "run_name": metrics.run_name,
        "total_seconds": metrics.total_seconds,
        "pages": metrics.pages,
        "text_count": metrics.text_count,
        "text_layers": metrics.text_layers,
        "inpaint_blocks_exported": metrics.inpaint_blocks_exported,
        "band_count": metrics.band_count,
        "stage_seconds": summarize_stages(metrics),
        "top_ocr_bands": [asdict(band) for band in rank_bands(metrics, "ocr", limit)],
        "top_inpaint_bands": [
            asdict(band) for band in rank_bands(metrics, "inpaint", limit)
        ],
        "skip_candidate_count": metrics.skip_candidate_count,
    }


def _metrics_from_project(project_path: Path) -> RunMetrics:
    payload = _load_json(project_path)
    pages = payload.get("paginas") or []
    stats = payload.get("estatisticas") or {}
    perf_summary = _find_strip_perf_summary(pages)
    entries = perf_summary.get("entries") if perf_summary else []

    text_layers = sum(len(page.get("text_layers") or []) for page in pages)
    inpaint_blocks = sum(len(page.get("inpaint_blocks") or []) for page in pages)
    text_count = _int_or_default(
        perf_summary.get("text_count") if perf_summary else None,
        _int_or_default(stats.get("total_textos"), text_layers),
    )

    return RunMetrics(
        source_path=str(project_path.parent),
        run_name=project_path.parent.name,
        total_seconds=_float_or_default(stats.get("tempo_processamento_seg"), 0.0),
        pages=_int_or_default(stats.get("total_paginas"), len(pages)),
        text_count=text_count,
        text_layers=text_layers,
        inpaint_blocks_exported=inpaint_blocks,
        band_count=_int_or_default(
            perf_summary.get("band_count") if perf_summary else None,
            len(entries or []),
        ),
        durations_sec=_durations(perf_summary.get("durations_sec") if perf_summary else {}),
        bands=[_band_metric_from_entry(entry) for entry in entries or []],
        skip_candidate_count=_int_or_default(
            perf_summary.get("smart_skip_shadow_candidate_count") if perf_summary else None,
            0,
        ),
    )


def _metrics_from_analysis_payload(
    payload: dict[str, Any], source_path: Path, *, run_name: str | None
) -> RunMetrics:
    runs = payload.get("runs")
    if isinstance(runs, dict):
        selected_name = run_name or _select_run_name(runs, source_path)
        run_payload = runs[selected_name]
    else:
        selected_name = run_name or str(payload.get("name") or source_path.stem)
        run_payload = payload

    entries = run_payload.get("entries") or run_payload.get("top_bands") or []
    return RunMetrics(
        source_path=str(source_path),
        run_name=str(run_payload.get("name") or selected_name),
        total_seconds=_float_or_default(run_payload.get("tempo_processamento_seg"), 0.0),
        pages=_int_or_default(run_payload.get("pages"), 0),
        text_count=_int_or_default(
            run_payload.get("texts"),
            _int_or_default(run_payload.get("summary_text_count"), 0),
        ),
        text_layers=_int_or_default(run_payload.get("text_layers"), 0),
        inpaint_blocks_exported=_int_or_default(
            run_payload.get("inpaint_blocks_exported"), 0
        ),
        band_count=_int_or_default(run_payload.get("band_count"), len(entries)),
        durations_sec=_durations(run_payload.get("durations_sec") or {}),
        bands=[_band_metric_from_entry(entry) for entry in entries],
        skip_candidate_count=_int_or_default(
            run_payload.get("smart_skip_shadow_candidate_count"), 0
        ),
    )


def _find_strip_perf_summary(pages: list[dict[str, Any]]) -> dict[str, Any]:
    for page in pages:
        profile = page.get("page_profile") or {}
        summary = profile.get("strip_perf_summary")
        if isinstance(summary, dict):
            return summary
    return {}


def _band_metric_from_entry(entry: dict[str, Any]) -> BandMetric:
    return BandMetric(
        band_index=_int_or_default(entry.get("band_index"), -1),
        durations_sec=_durations(entry.get("durations_sec") or {}),
        total_sec=_float_or_default(entry.get("total_sec"), 0.0),
        text_count=_int_or_default(entry.get("text_count"), 0),
        remaining_inpaint_blocks=_int_or_default(
            entry.get("remaining_inpaint_blocks"), 0
        ),
        y_top=_optional_int(entry.get("y_top")),
        y_bottom=_optional_int(entry.get("y_bottom")),
        height=_optional_int(entry.get("height")),
    )


def _select_run_name(runs: dict[str, Any], source_path: Path) -> str:
    path_hint = source_path.stem.lower()
    for name in runs:
        if name.lower() in path_hint:
            return name
    if len(runs) == 1:
        return next(iter(runs))
    return sorted(runs)[-1]


def _durations(raw: dict[str, Any]) -> dict[str, float]:
    return {
        str(key): _float_or_default(value, 0.0)
        for key, value in raw.items()
        if isinstance(raw, dict)
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    metrics = load_run_metrics(args.output_dir, run_name=args.run_name)
    summary = build_summary(metrics, limit=args.limit)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
