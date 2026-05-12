"""Run a shadow gate for the strip scheduler DAG against pipeline outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.strip.scheduler import build_strip_scheduler_plan
from pipeline.tools.compare_pipeline_outputs import evaluate_pipeline_output_compare


def evaluate_scheduler_shadow_gate(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    baseline_path = Path(baseline_dir)
    candidate_path = Path(candidate_dir)
    reasons: list[str] = []

    candidate_project = _load_project(candidate_path / "project.json")
    if candidate_project is None:
        return _write_result(
            _blocked_result(
                baseline_path,
                candidate_path,
                ["candidate missing or invalid project.json"],
            ),
            out_dir,
        )

    page_count = len(_pages(candidate_project))
    band_count = _strip_band_count(candidate_project)
    if page_count <= 0:
        reasons.append("candidate has no pages")
    if band_count <= 0:
        reasons.append("candidate missing strip_perf_summary entries")
    if reasons:
        return _write_result(_blocked_result(baseline_path, candidate_path, reasons), out_dir)

    plan = build_strip_scheduler_plan(band_count=band_count, page_count=page_count)
    compare = evaluate_pipeline_output_compare(baseline_path, candidate_path)
    compare_gate = compare.get("gate") or {}
    status = "PASS"
    if plan.validation.status != "PASS":
        status = "FAIL"
        reasons.append("scheduler plan validation failed")
    if compare_gate.get("status") != "PASS":
        status = "FAIL" if compare_gate.get("status") == "FAIL" else "BLOCK"
        reasons.append("scheduler shadow output compare failed")
    if not reasons:
        reasons.append("scheduler shadow plan is valid and candidate output matches baseline")

    result = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "gate": {
            "name": "strip_scheduler_shadow",
            "status": status,
            "reasons": reasons,
            "band_count": band_count,
            "page_count": page_count,
            "task_count": plan.task_count,
            "cpu_task_count": plan.cpu_task_count,
            "gpu_task_count": plan.gpu_task_count,
            "stage_counts": plan.stage_counts,
            "max_cpu_parallel": plan.max_cpu_parallel,
            "max_gpu_parallel": plan.max_gpu_parallel,
            "scheduler_validation_status": plan.validation.status,
            "scheduler_validation_reasons": plan.validation.reasons,
            "output_compare_status": compare_gate.get("status"),
            "output_compare_reasons": compare_gate.get("reasons", []),
            "notes": [
                "Shadow only: this gate does not execute run_chapter with the DAG.",
                "A PASS means the DAG contract is valid for this output shape and a candidate output remained structurally equivalent.",
            ],
        },
        "plan": plan.to_dict(),
        "output_compare": compare_gate,
    }
    return _write_result(result, out_dir)


def _load_project(project_path: Path) -> dict[str, Any] | None:
    if not project_path.exists():
        return None
    try:
        payload = json.loads(project_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _pages(project: dict[str, Any]) -> list[dict[str, Any]]:
    return [page for page in list(project.get("paginas") or []) if isinstance(page, dict)]


def _strip_band_count(project: dict[str, Any]) -> int:
    for page in _pages(project):
        profile = page.get("page_profile")
        if not isinstance(profile, dict):
            continue
        summary = profile.get("strip_perf_summary")
        if not isinstance(summary, dict):
            continue
        entries = [entry for entry in list(summary.get("entries") or []) if isinstance(entry, dict)]
        if entries:
            return len(entries)
        try:
            return max(0, int(summary.get("band_count") or 0))
        except Exception:
            return 0
    return 0


def _blocked_result(
    baseline_path: Path,
    candidate_path: Path,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "gate": {
            "name": "strip_scheduler_shadow",
            "status": "BLOCK",
            "reasons": reasons,
            "band_count": 0,
            "page_count": 0,
            "task_count": 0,
            "scheduler_validation_status": "BLOCK",
            "output_compare_status": "BLOCK",
        },
        "plan": {},
        "output_compare": {},
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
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = evaluate_scheduler_shadow_gate(args.baseline_dir, args.candidate_dir, args.out)
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
