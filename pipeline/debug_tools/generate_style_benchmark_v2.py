"""Build deterministic, isolated Style Atlas v2 benchmark specifications."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools.style_copy_score import (
    DEFAULT_RUNTIME_LOCK,
    _runtime_metadata,
    validate_runtime_contract,
)

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_DIR.parent
FONT_DIR = PIPELINE_DIR.parent / "fonts"
REQUIRED_LEVELS = ("smoke", "combinatorial", "hard-negative", "holdout")
REQUIRED_STYLE_FIELDS = (
    "alignment",
    "container",
    "fill",
    "font_name",
    "font_size_px",
    "font_weight",
    "font_width",
    "gradient",
    "rotation_deg",
    "shadow",
    "stroke",
    "text_a",
    "text_b",
)


def load_benchmark_spec(path: Path) -> dict[str, Any]:
    """Load a versioned specification and reject unavailable font fixtures."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("style benchmark v2 spec must use schema_version 2")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("style benchmark v2 spec must contain non-empty cases")
    levels = {case.get("level") for case in cases if isinstance(case, dict)}
    if levels != set(REQUIRED_LEVELS):
        raise ValueError(f"style benchmark v2 spec levels must be {REQUIRED_LEVELS}")
    for case in cases:
        if not isinstance(case, dict) or not set(REQUIRED_STYLE_FIELDS) <= set(case):
            raise ValueError("style benchmark v2 case is missing required StyleSpec fields")
        font_name = case["font_name"]
        if not isinstance(font_name, str) or not (FONT_DIR / font_name).is_file():
            raise ValueError(f"style benchmark v2 font does not exist: {font_name!r}")
        if case["text_a"] == case["text_b"]:
            raise ValueError("style benchmark v2 round-trip texts must differ")
    return payload


def build_style_specs(spec: dict[str, Any], *, seed: int) -> list[dict[str, Any]]:
    """Normalize the declared cases into a stable seeded StyleSpec sequence."""
    cases = [dict(case) for case in spec["cases"]]
    random.Random(seed).shuffle(cases)
    for index, case in enumerate(cases):
        case["id"] = str(case.get("id") or f"{case['level']}-{index:03d}")
        case["seed"] = int(seed)
        case["schema_version"] = 2
    return cases


def _single_component(value: str, *, label: str) -> str:
    if not value or not value.strip() or value in {".", ".."} or any(char in value for char in "/\\\x00"):
        raise ValueError(f"{label} must be a non-empty single path component")
    return value


def _assert_output_root_is_external(output_root: Path) -> Path:
    resolved = output_root.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError("style benchmark output_root must be outside the Git worktree")


def _rgba(hex_color: str) -> tuple[int, int, int, int]:
    value = str(hex_color).lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected #RRGGBB color, got {hex_color!r}")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4)) + (255,)


def _render_style_image(style: dict[str, Any], text: str, output_path: Path) -> None:
    """Render one case through Agg, avoiding Pillow's broken Windows glyph path."""
    import cv2
    import matplotlib
    import numpy as np
    from matplotlib import patheffects
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties

    matplotlib.use("Agg", force=True)
    container = style["container"]
    size = int(container["width"]), int(container["height"])
    figure = Figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    FigureCanvasAgg(figure)
    figure.patch.set_facecolor(container["background"])
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_facecolor(container["background"])
    axis.set_xlim(0, size[0])
    axis.set_ylim(size[1], 0)
    axis.set_axis_off()
    x, horizontal_alignment = {
        "left": (18, "left"),
        "right": (size[0] - 18, "right"),
    }.get(str(style["alignment"]), (size[0] / 2, "center"))
    y = size[1] / 2
    font = FontProperties(fname=str(FONT_DIR / style["font_name"]), size=int(style["font_size_px"]))
    effects: list[Any] = []
    stroke = style.get("stroke")
    if stroke:
        effects.append(
            patheffects.withStroke(
                linewidth=int(stroke["width_px"]) * 2,
                foreground=stroke["color"],
            )
        )
    shadow = style.get("shadow")
    if shadow:
        offset = tuple(int(value) for value in shadow["offset"])
        if shadow["kind"] == "shadow":
            axis.text(
                x + offset[0], y + offset[1], text,
                color=shadow["color"], fontproperties=font,
                horizontalalignment=horizontal_alignment, rotation=float(style["rotation_deg"]),
                verticalalignment="center", zorder=1,
            )
        else:
            effects.append(
                patheffects.withStroke(
                    linewidth=max(1, int(shadow["blur_px"])), foreground=shadow["color"]
                )
            )
    gradient = style.get("gradient")
    axis.text(
        x, y, text,
        color=gradient["colors"][0] if gradient else style["fill"],
        fontproperties=font, horizontalalignment=horizontal_alignment,
        path_effects=effects, rotation=float(style["rotation_deg"]),
        verticalalignment="center", zorder=2,
    )
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    if gradient:
        # Render the glyph alpha separately so the declared gradient is inside the letters,
        # while the primary canvas keeps the stroke and glow/shadow layers intact.
        mask_figure = Figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        FigureCanvasAgg(mask_figure)
        mask_figure.patch.set_alpha(0.0)
        mask_axis = mask_figure.add_axes((0, 0, 1, 1))
        mask_axis.patch.set_alpha(0.0)
        mask_axis.set_xlim(0, size[0])
        mask_axis.set_ylim(size[1], 0)
        mask_axis.set_axis_off()
        mask_axis.text(
            x, y, text,
            color="#FFFFFF",
            fontproperties=font,
            horizontalalignment=horizontal_alignment,
            rotation=float(style["rotation_deg"]),
            verticalalignment="center",
        )
        mask_figure.canvas.draw()
        alpha = np.asarray(mask_figure.canvas.buffer_rgba())[:, :, 3].astype(np.float32) / 255.0
        start = np.array(_rgba(gradient["colors"][0])[:3], dtype=np.float32)
        end = np.array(_rgba(gradient["colors"][-1])[:3], dtype=np.float32)
        ratios = np.linspace(0.0, 1.0, size[1], dtype=np.float32)[:, None, None]
        gradient_rgb = start[None, None, :] + (end - start)[None, None, :] * ratios
        rgba[:, :, :3] = (
            gradient_rgb * alpha[:, :, None] + rgba[:, :, :3].astype(np.float32) * (1.0 - alpha[:, :, None])
        ).round().astype(np.uint8)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)):
        raise RuntimeError(f"failed to write benchmark image: {output_path}")


def _select_cases(spec: dict[str, Any], *, level: str, seed: int) -> list[dict[str, Any]]:
    cases = build_style_specs(spec, seed=seed)
    if level != "all":
        if level not in REQUIRED_LEVELS:
            raise ValueError(f"unknown style benchmark level: {level}")
        cases = [case for case in cases if case["level"] == level]
    return cases


def _write_child_run(*, staging_run: Path, cases: list[dict[str, Any]], level: str, seed: int) -> None:
    """Render the benchmark payload inside the disposable child-process directory."""
    images_dir = staging_run / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    manifest_cases = []
    for case in cases:
        case_id = _single_component(case["id"], label="case id")
        image_a = Path("images") / f"{case_id}-a.png"
        image_b = Path("images") / f"{case_id}-b.png"
        if case["container"].get("render_text", True):
            _render_style_image(case, case["text_a"], staging_run / image_a)
            _render_style_image(case, case["text_b"], staging_run / image_b)
        else:
            import cv2
            import numpy as np
            background = _rgba(case["container"]["background"])
            size = int(case["container"]["width"]), int(case["container"]["height"])
            image = np.full((size[1], size[0], 3), background[:3][::-1], dtype=np.uint8)
            for image_path in (staging_run / image_a, staging_run / image_b):
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"failed to write hard-negative benchmark image: {image_path}")
        manifest_cases.append({**case, "image_a": image_a.as_posix(), "image_b": image_b.as_posix()})

    manifest = {
        "cases": manifest_cases,
        "level": level,
        "schema_version": 2,
        "seed": int(seed),
    }
    (staging_run / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_child_output(staging_run: Path, cases: list[dict[str, Any]], *, level: str, seed: int) -> None:
    manifest_path = staging_run / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("child generator did not create benchmark_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("level") != level or manifest.get("seed") != int(seed):
        raise RuntimeError("child generator manifest does not match the requested benchmark")
    if len(manifest.get("cases", [])) != len(cases):
        raise RuntimeError("child generator wrote an incomplete benchmark manifest")
    for case in manifest["cases"]:
        for image_key in ("image_a", "image_b"):
            image_path = staging_run / str(case.get(image_key, ""))
            if not image_path.is_file() or image_path.stat().st_size == 0:
                raise RuntimeError(f"child generator did not create {image_key}")


def _write_child_failure_warning(
    output_root: Path,
    *,
    run_id: str,
    completed: subprocess.CompletedProcess[str],
    error: str | None = None,
) -> Path:
    warning_dir = output_root / ".style-benchmark-v2-debug"
    warning_dir.mkdir(parents=True, exist_ok=True)
    warning_path = warning_dir / f"{run_id}-{time.time_ns()}.json"
    warning_path.write_text(
        json.dumps(
            {
                "action": "style_benchmark_child_generator_failed",
                "error": error,
                "returncode": completed.returncode,
                "stderr": completed.stderr,
                "stdout": completed.stdout,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return warning_path


def _run_child_job(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    staging_run = Path(job["staging_run"])
    _write_child_run(
        staging_run=staging_run,
        cases=list(job["cases"]),
        level=str(job["level"]),
        seed=int(job["seed"]),
    )


def generate_benchmark(
    *,
    spec_path: Path,
    level: str,
    output_root: Path,
    run_id: str,
    seed: int,
    runtime_lock_path: Path = DEFAULT_RUNTIME_LOCK,
) -> Path:
    """Generate atomically through a child so native font failures cannot publish partial runs."""
    run_id = _single_component(run_id, label="run_id")
    output_root = _assert_output_root_is_external(Path(output_root))
    validate_runtime_contract(dict(_runtime_metadata()), runtime_lock_path)
    spec = load_benchmark_spec(spec_path)
    cases = _select_cases(spec, level=level, seed=seed)

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"style benchmark run already exists: {run_dir}")

    staging_root = Path(tempfile.mkdtemp(prefix=f".style-benchmark-v2-{run_id}-", dir=output_root))
    staging_run = staging_root / "run"
    job_path = staging_root / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "cases": cases,
                "level": level,
                "seed": int(seed),
                "staging_run": str(staging_run),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child-job", str(job_path)],
            cwd=PIPELINE_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            warning_path = _write_child_failure_warning(output_root, run_id=run_id, completed=completed)
            raise RuntimeError(f"child generator failed; diagnostic written to {warning_path}")
        try:
            _validate_child_output(staging_run, cases, level=level, seed=seed)
        except RuntimeError as error:
            warning_path = _write_child_failure_warning(
                output_root,
                run_id=run_id,
                completed=completed,
                error=str(error),
            )
            raise RuntimeError(f"child generator failed; diagnostic written to {warning_path}") from error
        os.replace(staging_run, run_dir)
        return run_dir
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=PIPELINE_DIR / "tests" / "fixtures" / "style_benchmark_v2" / "benchmark_spec.json",
    )
    parser.add_argument("--level", choices=("all", *REQUIRED_LEVELS), default="smoke")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-lock", type=Path, default=DEFAULT_RUNTIME_LOCK)
    return parser


def _child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal isolated style benchmark renderer")
    parser.add_argument("--child-job", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if argv is not None and "--child-job" in argv:
        _run_child_job(_child_parser().parse_args(argv).child_job)
        return 0
    if argv is None and "--child-job" in sys.argv[1:]:
        _run_child_job(_child_parser().parse_args().child_job)
        return 0
    args = _parser().parse_args(argv)
    print(
        generate_benchmark(
            spec_path=args.spec,
            level=args.level,
            output_root=args.output_root,
            run_id=args.run_id,
            seed=args.seed,
            runtime_lock_path=args.runtime_lock,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
