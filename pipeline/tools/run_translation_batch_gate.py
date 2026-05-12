"""Evaluate whether chapter-level translation batching is worth implementing now."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.tools.analyze_pipeline_run import build_summary, load_run_metrics


DEFAULT_MIN_TRANSLATION_SECONDS = 5.0
DEFAULT_MIN_TRANSLATION_SHARE = 0.05


def evaluate_translation_batch_gate(
    output_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    min_translation_seconds: float = DEFAULT_MIN_TRANSLATION_SECONDS,
    min_translation_share: float = DEFAULT_MIN_TRANSLATION_SHARE,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    project_path = output_path / "project.json"
    if not project_path.exists():
        return _write_result(
            _blocked(output_path, ["missing project.json; cannot evaluate translation stage"]),
            out_dir,
        )

    try:
        metrics = load_run_metrics(output_path)
        summary = build_summary(metrics)
    except Exception as exc:
        return _write_result(_blocked(output_path, [f"could not load metrics: {exc}"]), out_dir)

    stage_seconds = summary.get("stage_seconds") or {}
    translation_seconds = round(float(stage_seconds.get("translate", 0.0) or 0.0), 4)
    stage_total_seconds = round(sum(float(value or 0.0) for value in stage_seconds.values()), 4)
    translation_share = round(
        translation_seconds / stage_total_seconds,
        4,
    ) if stage_total_seconds > 0 else 0.0

    reasons: list[str] = []
    status = "PASS"
    if (
        translation_seconds < min_translation_seconds
        and translation_share < min_translation_share
    ):
        status = "FAIL"
        reasons.append(
            "translation stage is below batching threshold "
            f"({translation_seconds:.2f}s, {translation_share:.2%})"
        )
    else:
        reasons.append(
            "translation stage is material enough to justify batching investigation"
        )

    result = {
        "source_path": str(output_path),
        "gate": {
            "name": "translation_batch_candidate",
            "status": status,
            "reasons": reasons,
            "translation_seconds": translation_seconds,
            "stage_total_seconds": stage_total_seconds,
            "translation_share": translation_share,
            "min_translation_seconds": float(min_translation_seconds),
            "min_translation_share": float(min_translation_share),
            "text_count": int(summary.get("text_count") or 0),
        },
    }
    return _write_result(result, out_dir)


def _blocked(output_path: Path, reasons: list[str]) -> dict[str, Any]:
    return {
        "source_path": str(output_path),
        "gate": {
            "name": "translation_batch_candidate",
            "status": "BLOCK",
            "reasons": reasons,
            "translation_seconds": 0.0,
            "stage_total_seconds": 0.0,
            "translation_share": 0.0,
            "min_translation_seconds": DEFAULT_MIN_TRANSLATION_SECONDS,
            "min_translation_share": DEFAULT_MIN_TRANSLATION_SHARE,
            "text_count": 0,
        },
    }


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
    parser.add_argument("--min-translation-seconds", type=float, default=DEFAULT_MIN_TRANSLATION_SECONDS)
    parser.add_argument("--min-translation-share", type=float, default=DEFAULT_MIN_TRANSLATION_SHARE)
    args = parser.parse_args(argv)

    result = evaluate_translation_batch_gate(
        args.output_dir,
        args.out,
        min_translation_seconds=args.min_translation_seconds,
        min_translation_share=args.min_translation_share,
    )
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
