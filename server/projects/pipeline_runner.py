from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from server.projects.workspace import relative_asset


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO_ROOT / "pipeline"
PIPELINE_MAIN = PIPELINE_DIR / "main.py"


def _pipeline_env() -> dict[str, str]:
    env = os.environ.copy()
    python_path = [str(PIPELINE_DIR), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


def _run(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(PIPELINE_MAIN), *args]
    result = subprocess.run(
        command,
        cwd=str(PIPELINE_DIR),
        env=_pipeline_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "pipeline falhou").strip()
        raise HTTPException(status_code=500, detail=details[-4000:])
    return result


def _region_args(region: dict[str, Any] | None) -> list[str]:
    if not region:
        return []
    args: list[str] = []
    bbox = region.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        args.extend(["--region-bbox", ",".join(str(float(value)) for value in bbox[:4])])
    mask_path = region.get("mask_path")
    if isinstance(mask_path, str) and mask_path:
        args.extend(["--external-mask", mask_path])
    return args


def render_preview_page(root: Path, page_index: int, page: dict[str, Any]) -> str:
    project_json = root / "project.json"
    cache_dir = root / "render-cache" / "preview"
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = f"{page_index + 1:03d}-{int(time.time() * 1000)}"
    page_override = cache_dir / f"{token}.json"
    output = cache_dir / f"{token}.png"
    page_override.write_text(json.dumps(page, ensure_ascii=True), encoding="utf-8")
    _run(["--render-preview-page", str(project_json), str(page_index), str(page_override), str(output)])
    return relative_asset(root, output)


def run_page_action(root: Path, page_index: int, action: str, region: dict[str, Any] | None = None) -> list[str]:
    project_json = root / "project.json"
    if action == "retypeset":
        _run(["--retypeset", str(project_json), str(page_index)])
        return ["rendered", "project_json"]
    command_by_action = {
        "detect": "--detect-page",
        "detect_boxes": "--detect-boxes-page",
        "ocr": "--ocr-page",
        "translate": "--translate-page",
        "inpaint": "--reinpaint-page",
    }
    command = command_by_action.get(action)
    if command is None:
        raise HTTPException(status_code=422, detail="acao invalida")
    _run([command, str(project_json), str(page_index), *_region_args(region)])
    if action == "inpaint":
        return ["inpaint", "rendered", "project_json"]
    if action in {"detect", "detect_boxes", "ocr", "translate"}:
        return ["project_json", "rendered"]
    return ["project_json"]


def process_block(root: Path, page_index: int, block_id: str, mode: str) -> list[str]:
    if mode not in {"ocr", "translate", "inpaint"}:
        raise HTTPException(status_code=422, detail="modo invalido")
    project_json = root / "project.json"
    _run(["--process-block", mode, str(project_json), str(page_index), block_id])
    changed = ["project_json"]
    if mode == "inpaint":
        changed.extend(["inpaint", "rendered"])
    if mode in {"translate", "ocr"}:
        changed.append("rendered")
    return changed
