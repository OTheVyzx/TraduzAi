"""Evaluate Smart Skip shadow-mode evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.tools.analyze_pipeline_run import load_run_metrics


DEFAULT_MIN_SAVINGS_SECONDS = 16.75


def evaluate_smart_skip_shadow_gate(
    output_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    baseline_dir: str | Path | None = None,
    min_savings_seconds: float = DEFAULT_MIN_SAVINGS_SECONDS,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    project_path = output_path / "project.json"
    if not project_path.exists():
        return _write_result(
            _result(output_path, "BLOCK", ["missing project.json"]),
            out_dir,
        )

    project = _load_json(project_path)
    summary = _find_strip_perf_summary(project)
    if "smart_skip_shadow_candidate_count" not in summary:
        return _write_result(
            _result(output_path, "BLOCK", ["missing Smart Skip shadow summary"]),
            out_dir,
        )

    candidate_count = _int(summary.get("smart_skip_shadow_candidate_count"))
    category_counts = dict(summary.get("smart_skip_shadow_category_counts") or {})
    unsafe_candidate_count = _unsafe_candidate_count(project)
    estimated_savings = _estimate_savings_from_entries(summary)

    reasons: list[str] = []
    structural = _compare_structural_baseline(output_path, baseline_dir)
    if not structural["ok"]:
        reasons.extend(structural["reasons"])
        status = "FAIL"
    elif unsafe_candidate_count:
        reasons.append(f"{unsafe_candidate_count} unsafe candidate(s) found")
        status = "FAIL"
    elif candidate_count <= 0:
        reasons.append("no Smart Skip shadow candidates found")
        status = "FAIL"
    elif estimated_savings < min_savings_seconds:
        reasons.append(
            f"estimated savings {estimated_savings:.2f}s below threshold "
            f"{min_savings_seconds:.2f}s"
        )
        status = "FAIL"
    else:
        reasons.append(
            f"estimated savings {estimated_savings:.2f}s meets threshold "
            f"{min_savings_seconds:.2f}s"
        )
        status = "PASS"

    result = {
        "source_path": str(output_path),
        "baseline_path": str(Path(baseline_dir)) if baseline_dir else None,
        "gate": {
            "name": "smart_skip_shadow",
            "status": status,
            "reasons": reasons,
            "candidate_count": candidate_count,
            "unsafe_candidate_count": unsafe_candidate_count,
            "category_counts": category_counts,
            "estimated_savings_seconds": round(estimated_savings, 4),
            "min_savings_seconds": float(min_savings_seconds),
            "structural_compare": structural,
        },
    }
    return _write_result(result, out_dir)


def _estimate_savings_from_entries(summary: dict[str, Any]) -> float:
    total = 0.0
    for entry in summary.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        if _int(entry.get("smart_skip_shadow_candidate_count")) <= 0:
            continue
        durations = entry.get("durations_sec") or {}
        total += _float(durations.get("inpaint"))
    return total


def _unsafe_candidate_count(project: dict[str, Any]) -> int:
    count = 0
    for page in project.get("paginas") or []:
        for candidate in ((page.get("_smart_skip_shadow") or {}).get("candidates") or []):
            if candidate.get("category") == "not_safe_to_skip":
                count += 1
    return count


def _compare_structural_baseline(
    output_path: Path,
    baseline_dir: str | Path | None,
) -> dict[str, Any]:
    if baseline_dir is None:
        return {"ok": True, "reasons": ["baseline not provided; structural compare skipped"]}
    try:
        baseline = load_run_metrics(Path(baseline_dir))
        candidate = load_run_metrics(output_path)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        return {"ok": False, "reasons": [f"could not compare baseline: {exc}"]}

    fields = ("pages", "text_count", "text_layers", "inpaint_blocks_exported")
    reasons = []
    for field in fields:
        if getattr(baseline, field) != getattr(candidate, field):
            reasons.append(
                f"{field} changed: baseline={getattr(baseline, field)} "
                f"candidate={getattr(candidate, field)}"
            )
    return {"ok": not reasons, "reasons": reasons or ["structural counts match"]}


def _result(output_path: Path, status: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "source_path": str(output_path),
        "gate": {
            "name": "smart_skip_shadow",
            "status": status,
            "reasons": reasons,
            "candidate_count": 0,
            "unsafe_candidate_count": 0,
            "category_counts": {},
            "estimated_savings_seconds": 0.0,
            "min_savings_seconds": DEFAULT_MIN_SAVINGS_SECONDS,
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


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--min-savings-seconds", type=float, default=DEFAULT_MIN_SAVINGS_SECONDS)
    args = parser.parse_args(argv)

    result = evaluate_smart_skip_shadow_gate(
        args.output_dir,
        args.out,
        baseline_dir=args.baseline,
        min_savings_seconds=args.min_savings_seconds,
    )
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
