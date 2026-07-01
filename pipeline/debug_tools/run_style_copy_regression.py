"""Run a focused style-copy regression on a copied TraduzAI chapter run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RUN = Path(
    r"N:\TraduzAI\DEBUGM\runs\2026-05-23_mihon_matrix20_v33\18_regressed_items_ch3"
)
DEFAULT_PAGES = [1, 2, 5, 6, 7, 8]


def _copy_run(source_run: Path, dest_run: Path) -> None:
    if dest_run.exists():
        shutil.rmtree(dest_run)
    dest_run.mkdir(parents=True, exist_ok=True)

    for name in ("project.json", "runner_config.json"):
        source = source_run / name
        if source.exists():
            shutil.copy2(source, dest_run / name)

    for name in ("originals", "images", "translated", "layers"):
        source = source_run / name
        if source.exists():
            shutil.copytree(source, dest_run / name)

    (dest_run / "debug").mkdir(exist_ok=True)
    (dest_run / "logs").mkdir(exist_ok=True)


def _rewrite_project(dest_run: Path) -> None:
    project_path = dest_run / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["_work_dir"] = str(dest_run)
    qa = project.get("qa")
    if isinstance(qa, dict):
        qa.pop("summary", None)
    log = project.get("log")
    if isinstance(log, dict):
        log.pop("summary", None)
    project_path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")


def _rewrite_runner_config(dest_run: Path) -> None:
    config_path = dest_run / "runner_config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    config["work_dir"] = str(dest_run)
    config["logs_dir"] = str(dest_run / "logs")
    config["run_id"] = f"{config.get('run_id') or 'style_copy'}_codex"
    config["job_id"] = f"{config.get('job_id') or 'style_copy'}_codex"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_step(project_path: Path, page_number: int, command: str) -> dict:
    page_index = page_number - 1
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "main.py"),
        command,
        str(project_path),
        str(page_index),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "page": page_number,
        "command": command,
        "returncode": proc.returncode,
        "output_tail": proc.stdout[-5000:],
    }


def _run_style_audit(dest_run: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "debug_tools" / "style_audit_report.py"),
        "--run",
        str(dest_run),
        "--originals",
        str(dest_run / "originals"),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        parsed = json.loads(proc.stdout[proc.stdout.find("{") :])
    except Exception:
        parsed = {}
    parsed["returncode"] = proc.returncode
    parsed["output_tail"] = proc.stdout[-5000:]
    return parsed


def _run_style_score(dest_run: Path) -> dict:
    report_dir = dest_run / "debug" / "codex_style_audit" / "visual_report"
    records_path = report_dir / "style_audit_records.jsonl"
    output_path = report_dir / "style_copy_score.json"
    if not records_path.exists():
        return {"returncode": 1, "error": f"missing records: {records_path}"}
    cmd = [
        sys.executable,
        str(ROOT / "pipeline" / "debug_tools" / "style_copy_score.py"),
        "--records",
        str(records_path),
        "--output",
        str(output_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        parsed = json.loads(proc.stdout[proc.stdout.find("{") :])
    except Exception:
        parsed = {}
    parsed["returncode"] = proc.returncode
    parsed["output_path"] = str(output_path)
    parsed["output_tail"] = proc.stdout[-5000:]
    return parsed


def _parse_pages(value: str) -> list[int]:
    pages: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        pages.append(int(chunk))
    return pages or DEFAULT_PAGES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-run", type=Path)
    parser.add_argument("--pages", default=",".join(str(p) for p in DEFAULT_PAGES))
    parser.add_argument("--skip-inpaint", action="store_true")
    args = parser.parse_args()

    source_run = args.source_run
    if not source_run.exists():
        raise FileNotFoundError(source_run)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest_run = args.output_run or (source_run.parent.parent / f"style_copy_regression_ch3_{timestamp}")
    pages = _parse_pages(args.pages)

    _copy_run(source_run, dest_run)
    _rewrite_project(dest_run)
    _rewrite_runner_config(dest_run)

    project_path = dest_run / "project.json"
    steps: list[dict] = []
    for page in pages:
        for command in ("--detect-page", "--reinpaint-page", "--retypeset"):
            if args.skip_inpaint and command == "--reinpaint-page":
                continue
            result = _run_step(project_path, page, command)
            steps.append(result)
            if result["returncode"] != 0:
                summary = {
                    "source_run": str(source_run),
                    "output_run": str(dest_run),
                    "pages": pages,
                    "steps": steps,
                    "failed": result,
                }
                (dest_run / "debug" / "style_copy_regression_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return result["returncode"]

    audit = _run_style_audit(dest_run)
    score = _run_style_score(dest_run) if audit.get("returncode") == 0 else {}
    summary = {
        "source_run": str(source_run),
        "output_run": str(dest_run),
        "pages": pages,
        "steps": steps,
        "audit": audit,
        "score": score,
        "visual_report": str(dest_run / "debug" / "codex_style_audit" / "visual_report" / "index.html"),
    }
    (dest_run / "debug" / "style_copy_regression_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
