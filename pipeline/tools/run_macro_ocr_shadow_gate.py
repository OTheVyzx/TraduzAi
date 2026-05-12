"""Evaluate Macro OCR shadow/precheck evidence from pipeline artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.ocr.macro_ocr import estimate_macro_ocr_shadow


DEFAULT_MIN_SAVINGS_SECONDS = 10.0
DEFAULT_MAX_MISSING_TEXT_RATE = 0.02
DEFAULT_MAX_FALLBACK_RATE = 0.15


def evaluate_macro_ocr_shadow_gate(
    output_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    min_savings_seconds: float = DEFAULT_MIN_SAVINGS_SECONDS,
    max_missing_text_rate: float = DEFAULT_MAX_MISSING_TEXT_RATE,
    max_fallback_rate: float = DEFAULT_MAX_FALLBACK_RATE,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    project_path = output_path / "project.json"
    if not project_path.exists():
        return _write_result(_blocked(output_path, ["missing project.json"]), out_dir)

    try:
        project = _load_json(project_path)
    except Exception as exc:
        return _write_result(_blocked(output_path, [f"could not load project.json: {exc}"]), out_dir)

    if not _find_strip_perf_summary(project):
        return _write_result(_blocked(output_path, ["missing strip_perf_summary"]), out_dir)

    report = estimate_macro_ocr_shadow(project)
    reasons: list[str] = []
    status = "PASS"

    if report.estimated_savings_seconds < min_savings_seconds:
        status = "FAIL"
        reasons.append(
            f"estimated OCR savings {report.estimated_savings_seconds:.2f}s "
            f"below threshold {min_savings_seconds:.2f}s"
        )
    if report.missing_text_rate > max_missing_text_rate:
        status = "FAIL"
        reasons.append(
            f"missing text rate {report.missing_text_rate:.2%} exceeds "
            f"{max_missing_text_rate:.2%}"
        )
    if report.fallback_rate > max_fallback_rate:
        status = "FAIL"
        reasons.append(
            f"fallback rate {report.fallback_rate:.2%} exceeds "
            f"{max_fallback_rate:.2%}"
        )
    if not reasons:
        reasons.append(
            "artifact-level Macro OCR shadow mapping passes estimated savings "
            "and risk thresholds"
        )

    result = {
        "source_path": str(output_path),
        "gate": {
            "name": "macro_ocr_shadow",
            "status": status,
            "reasons": reasons,
            "current_ocr_seconds": report.current_ocr_seconds,
            "estimated_macro_ocr_seconds": report.estimated_macro_ocr_seconds,
            "estimated_savings_seconds": report.estimated_savings_seconds,
            "current_ocr_band_calls": report.current_ocr_band_calls,
            "macro_window_count": report.macro_window_count,
            "text_line_count": report.text_line_count,
            "mapped_line_count": report.mapped_line_count,
            "fallback_line_count": report.fallback_line_count,
            "missing_line_count": report.missing_line_count,
            "missing_text_rate": report.missing_text_rate,
            "fallback_rate": report.fallback_rate,
            "wrong_band_rate": report.wrong_band_rate,
            "thresholds": {
                "min_savings_seconds": float(min_savings_seconds),
                "max_missing_text_rate": float(max_missing_text_rate),
                "max_fallback_rate": float(max_fallback_rate),
            },
            "note": (
                "This gate uses existing project artifacts to test remapping risk "
                "and estimate OCR-call reduction; it does not replace the OCR path."
            ),
        },
    }
    return _write_result(result, out_dir)


def _blocked(output_path: Path, reasons: list[str]) -> dict[str, Any]:
    return {
        "source_path": str(output_path),
        "gate": {
            "name": "macro_ocr_shadow",
            "status": "BLOCK",
            "reasons": reasons,
            "current_ocr_seconds": 0.0,
            "estimated_macro_ocr_seconds": 0.0,
            "estimated_savings_seconds": 0.0,
            "current_ocr_band_calls": 0,
            "macro_window_count": 0,
            "text_line_count": 0,
            "mapped_line_count": 0,
            "fallback_line_count": 0,
            "missing_line_count": 0,
            "missing_text_rate": 0.0,
            "fallback_rate": 0.0,
            "wrong_band_rate": 0.0,
        },
    }


def _find_strip_perf_summary(project: dict[str, Any]) -> dict[str, Any]:
    for page in project.get("paginas") or []:
        profile = page.get("page_profile") or {}
        summary = profile.get("strip_perf_summary")
        if isinstance(summary, dict):
            return summary
    return {}


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-savings-seconds", type=float, default=DEFAULT_MIN_SAVINGS_SECONDS)
    parser.add_argument("--max-missing-text-rate", type=float, default=DEFAULT_MAX_MISSING_TEXT_RATE)
    parser.add_argument("--max-fallback-rate", type=float, default=DEFAULT_MAX_FALLBACK_RATE)
    args = parser.parse_args(argv)

    result = evaluate_macro_ocr_shadow_gate(
        args.output_dir,
        args.out,
        min_savings_seconds=args.min_savings_seconds,
        max_missing_text_rate=args.max_missing_text_rate,
        max_fallback_rate=args.max_fallback_rate,
    )
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
