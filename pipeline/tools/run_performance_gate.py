"""Run the first performance gate for a TraduzAi pipeline output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.tools.analyze_pipeline_run import build_summary, load_run_metrics


VISUAL_BOTTLENECK_THRESHOLD = 0.60


def evaluate_performance_gate(
    output_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    reasons: list[str] = []

    if not (output_path / "project.json").exists():
        result = _blocked_result(
            output_path,
            ["missing project.json; cannot validate pipeline output contract"],
        )
        return _write_result(result, out_dir)

    if not _has_final_images(output_path):
        result = _blocked_result(
            output_path,
            ["missing final images under translated/ or images/"],
        )
        return _write_result(result, out_dir)

    try:
        metrics = load_run_metrics(output_path)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        result = _blocked_result(output_path, [f"could not load metrics: {exc}"])
        return _write_result(result, out_dir)

    summary = build_summary(metrics)
    stage_seconds = summary["stage_seconds"]
    stage_total = sum(float(value) for value in stage_seconds.values())
    visual_seconds = float(stage_seconds.get("ocr", 0.0)) + float(
        stage_seconds.get("inpaint", 0.0)
    )
    visual_share = round(visual_seconds / stage_total, 4) if stage_total > 0 else 0.0

    if stage_total <= 0:
        status = "BLOCK"
        reasons.append("stage timings are missing or zero")
    elif visual_share < VISUAL_BOTTLENECK_THRESHOLD:
        status = "FAIL"
        reasons.append(
            "visual bottleneck share "
            f"{visual_share:.2%} is below {VISUAL_BOTTLENECK_THRESHOLD:.0%}"
        )
    else:
        status = "PASS"
        reasons.append(
            "OCR and inpaint dominate measured stage time "
            f"({visual_share:.2%})"
        )

    summary["gate"] = {
        "name": "baseline_visual_bottleneck",
        "status": status,
        "reasons": reasons,
        "visual_seconds": round(visual_seconds, 4),
        "stage_total_seconds": round(stage_total, 4),
        "visual_bottleneck_share": visual_share,
        "threshold": VISUAL_BOTTLENECK_THRESHOLD,
        "required_artifacts": {
            "project_json": True,
            "final_images": True,
        },
    }
    return _write_result(summary, out_dir)


def _blocked_result(output_path: Path, reasons: list[str]) -> dict[str, Any]:
    return {
        "source_path": str(output_path),
        "gate": {
            "name": "baseline_visual_bottleneck",
            "status": "BLOCK",
            "reasons": reasons,
            "visual_seconds": 0.0,
            "stage_total_seconds": 0.0,
            "visual_bottleneck_share": 0.0,
            "threshold": VISUAL_BOTTLENECK_THRESHOLD,
            "required_artifacts": {
                "project_json": (output_path / "project.json").exists(),
                "final_images": _has_final_images(output_path),
            },
        },
    }


def _has_final_images(output_path: Path) -> bool:
    for folder_name in ("translated", "images"):
        folder = output_path / folder_name
        if folder.exists() and any(folder.glob("*.jpg")):
            return True
        if folder.exists() and any(folder.glob("*.png")):
            return True
    return False


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
    args = parser.parse_args(argv)

    result = evaluate_performance_gate(args.output_dir, args.out)
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
