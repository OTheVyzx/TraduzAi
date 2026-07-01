from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from typesetter import renderer as renderer_mod  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class RenderRun:
    image: Image.Image
    input_image: Image.Image
    texts: list[dict[str, Any]]
    error: str | None = None


@contextmanager
def _renderer_backend_env(backend: str):
    old_backend = os.environ.get("TRADUZAI_RENDERER_BACKEND")
    old_strict = os.environ.get("TRADUZAI_RENDERER_STRICT")
    try:
        if backend == "koharu_rust":
            os.environ["TRADUZAI_RENDERER_BACKEND"] = "koharu_rust"
            os.environ.pop("TRADUZAI_RENDERER_STRICT", None)
        else:
            os.environ.pop("TRADUZAI_RENDERER_BACKEND", None)
            os.environ.pop("TRADUZAI_RENDERER_STRICT", None)
        yield
    finally:
        if old_backend is None:
            os.environ.pop("TRADUZAI_RENDERER_BACKEND", None)
        else:
            os.environ["TRADUZAI_RENDERER_BACKEND"] = old_backend
        if old_strict is None:
            os.environ.pop("TRADUZAI_RENDERER_STRICT", None)
        else:
            os.environ["TRADUZAI_RENDERER_STRICT"] = old_strict


def _case_dirs(fixture_dir: Path) -> Iterable[Path]:
    for child in sorted(fixture_dir.iterdir()):
        if child.is_dir() and (child / "case.json").is_file():
            yield child


def _parse_color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(max(0, min(255, int(v))) for v in value[:3])  # type: ignore[return-value]
        except (TypeError, ValueError):
            return (255, 255, 255)
    if isinstance(value, str):
        raw = value.strip().lstrip("#")
        if len(raw) == 6:
            try:
                return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
            except ValueError:
                return (255, 255, 255)
    return (255, 255, 255)


def _load_case(case_dir: Path) -> dict[str, Any]:
    return json.loads((case_dir / "case.json").read_text(encoding="utf-8"))


def _load_input_image(case_dir: Path, case: dict[str, Any]) -> Image.Image:
    image_name = case.get("image")
    if isinstance(image_name, str) and image_name.strip():
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = case_dir / image_path
        return Image.open(image_path).convert("RGB")
    width = int(case.get("width") or 640)
    height = int(case.get("height") or 480)
    return Image.new("RGB", (width, height), _parse_color(case.get("background")))


def _render_case(case_dir: Path, backend: str) -> RenderRun:
    case = _load_case(case_dir)
    input_image = _load_input_image(case_dir, case)
    image = input_image.copy()
    texts = copy.deepcopy(case.get("texts") or [])
    error: str | None = None
    with _renderer_backend_env(backend):
        for text in texts:
            if not isinstance(text, dict):
                continue
            try:
                renderer_mod.render_text_block(
                    image,
                    text,
                    pre_render_np=np.asarray(input_image.convert("RGB"), dtype=np.uint8),
                )
            except Exception as exc:  # pragma: no cover - kept for CLI diagnostics.
                error = str(exc)
    return RenderRun(image=image.convert("RGB"), input_image=input_image.convert("RGB"), texts=texts, error=error)


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.int16)


def _coverage(image: Image.Image, input_image: Image.Image) -> int:
    current = _rgb_array(image)
    base = _rgb_array(input_image)
    if current.shape != base.shape:
        base = np.asarray(input_image.resize(image.size).convert("RGB"), dtype=np.int16)
    return int(np.count_nonzero(np.any(np.abs(current - base) > 3, axis=2)))


def _alpha_bbox_from_image(image: Image.Image, input_image: Image.Image) -> list[int] | None:
    current = _rgb_array(image)
    base = _rgb_array(input_image)
    if current.shape != base.shape:
        base = np.asarray(input_image.resize(image.size).convert("RGB"), dtype=np.int16)
    changed = np.any(np.abs(current - base) > 3, axis=2)
    ys, xs = np.where(changed)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _text_union_bbox(texts: list[dict[str, Any]]) -> list[int] | None:
    boxes: list[list[int]] = []
    for text in texts:
        value = text.get("render_bbox")
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in value]
            except (TypeError, ValueError):
                continue
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _bbox_delta(a: list[int] | None, b: list[int] | None) -> int | None:
    if a is None or b is None:
        return None
    return int(sum(abs(int(x) - int(y)) for x, y in zip(a, b)))


def _qa_flags(texts: list[dict[str, Any]]) -> list[str]:
    flags: set[str] = set()
    for text in texts:
        for flag in text.get("qa_flags") or []:
            if flag:
                flags.add(str(flag))
        metrics = text.get("qa_metrics")
        if isinstance(metrics, dict):
            render_fit = metrics.get("render_fit")
            if isinstance(render_fit, dict):
                for flag in render_fit.get("flags") or []:
                    if flag:
                        flags.add(str(flag))
    return sorted(flags)


def _font_sizes(texts: list[dict[str, Any]]) -> list[float]:
    sizes: list[float] = []
    for text in texts:
        for candidate in (
            text.get("font_size"),
            (text.get("estilo") or {}).get("tamanho") if isinstance(text.get("estilo"), dict) else None,
            (text.get("style") or {}).get("fontSize") if isinstance(text.get("style"), dict) else None,
            (text.get("style") or {}).get("font_size") if isinstance(text.get("style"), dict) else None,
        ):
            try:
                if candidate is not None:
                    sizes.append(float(candidate))
                    break
            except (TypeError, ValueError):
                continue
    return sizes


def _fallback_occurred(texts: list[dict[str, Any]]) -> bool:
    renderable = [text for text in texts if str(text.get("translated") or "").strip()]
    if not renderable:
        return False
    rust_count = 0
    for text in renderable:
        debug = text.get("_render_debug")
        if isinstance(debug, dict) and debug.get("renderer_backend") == "koharu_rust":
            rust_count += 1
    return rust_count < len(renderable)


def _diff_image(python_image: Image.Image, rust_image: Image.Image) -> Image.Image:
    a = _rgb_array(python_image)
    b = _rgb_array(rust_image)
    if a.shape != b.shape:
        rust_image = rust_image.resize(python_image.size)
        b = _rgb_array(rust_image)
    intensity = np.max(np.abs(a - b), axis=2).clip(0, 255).astype(np.uint8)
    heat = np.zeros((*intensity.shape, 3), dtype=np.uint8)
    heat[..., 0] = intensity
    heat[..., 1] = (intensity // 3).astype(np.uint8)
    heat[..., 2] = 255 - intensity
    return Image.fromarray(heat, "RGB")


def _metrics(input_image: Image.Image, python_run: RenderRun, rust_run: RenderRun) -> dict[str, Any]:
    python_arr = _rgb_array(python_run.image)
    rust_arr = _rgb_array(rust_run.image)
    if python_arr.shape != rust_arr.shape:
        rust_arr = np.asarray(rust_run.image.resize(python_run.image.size).convert("RGB"), dtype=np.int16)
    diff = np.abs(python_arr - rust_arr)
    changed = np.any(diff > 3, axis=2)
    total = max(1, changed.size)
    python_bbox = _text_union_bbox(python_run.texts) or _alpha_bbox_from_image(python_run.image, input_image)
    rust_bbox = _text_union_bbox(rust_run.texts) or _alpha_bbox_from_image(rust_run.image, input_image)
    return {
        "pixel_diff_pct": round(float(np.count_nonzero(changed) / total), 6),
        "mean_abs_diff": round(float(np.mean(diff)), 6),
        "alpha_coverage_python": _coverage(python_run.image, input_image),
        "alpha_coverage_rust": _coverage(rust_run.image, input_image),
        "alpha_coverage_diff": abs(_coverage(python_run.image, input_image) - _coverage(rust_run.image, input_image)),
        "render_bbox_python": python_bbox,
        "render_bbox_rust": rust_bbox,
        "render_bbox_delta": _bbox_delta(python_bbox, rust_bbox),
        "qa_flags_python": _qa_flags(python_run.texts),
        "qa_flags_rust": _qa_flags(rust_run.texts),
        "font_sizes_python": _font_sizes(python_run.texts),
        "font_sizes_rust": _font_sizes(rust_run.texts),
    }


def _draw_label(image: Image.Image, label: str) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + 22), (245, 245, 245))
    out.paste(image.convert("RGB"), (0, 22))
    draw = ImageDraw.Draw(out)
    draw.text((6, 5), label, fill=(20, 20, 20))
    return out


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    out = Image.new("RGB", (width, height), (245, 245, 245))
    src = image.convert("RGB")
    src.thumbnail((width, height))
    out.paste(src, ((width - src.width) // 2, (height - src.height) // 2))
    return out


def _write_contact_sheet(rows: list[dict[str, Any]], out_dir: Path) -> None:
    tiles: list[Image.Image] = []
    for row in rows[:24]:
        case_dir = out_dir / row["name"]
        images = [
            ("input", case_dir / "input.png"),
            ("python", case_dir / "python.png"),
            ("koharu_rust", case_dir / "koharu_rust.png"),
            ("diff", case_dir / "diff.png"),
        ]
        fitted = [_draw_label(_fit(Image.open(path), 220, 160), label) for label, path in images if path.exists()]
        if fitted:
            sheet_row = Image.new("RGB", (sum(img.width for img in fitted), max(img.height for img in fitted)), (245, 245, 245))
            x = 0
            for tile in fitted:
                sheet_row.paste(tile, (x, 0))
                x += tile.width
            tiles.append(sheet_row)
    if not tiles:
        return
    sheet = Image.new("RGB", (max(tile.width for tile in tiles), sum(tile.height for tile in tiles)), (245, 245, 245))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    sheet.save(out_dir / "contact_sheet.png")


def compare_fixture_dir(fixture_dir: Path, out_dir: Path, threshold: float = 0.35) -> dict[str, Any]:
    if not fixture_dir.is_dir():
        raise FileNotFoundError(f"fixture directory not found: {fixture_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case_dir in _case_dirs(fixture_dir):
        case = _load_case(case_dir)
        name = str(case.get("name") or case_dir.name)
        case_out = out_dir / name
        case_out.mkdir(parents=True, exist_ok=True)

        python_run = _render_case(case_dir, "python")
        rust_run = _render_case(case_dir, "koharu_rust")
        diff = _diff_image(python_run.image, rust_run.image)

        python_run.input_image.save(case_out / "input.png")
        python_run.image.save(case_out / "python.png")
        rust_run.image.save(case_out / "koharu_rust.png")
        diff.save(case_out / "diff.png")

        metrics = _metrics(python_run.input_image, python_run, rust_run)
        fallback = _fallback_occurred(rust_run.texts)
        passed = not fallback and metrics["pixel_diff_pct"] <= threshold
        rows.append(
            {
                "name": name,
                "fixture": str(case_dir / "case.json"),
                "passed": bool(passed),
                "fallback_occurred": bool(fallback),
                "metrics": metrics,
                "python_error": python_run.error,
                "rust_error": rust_run.error,
                "artifacts": {
                    "input": str(case_out / "input.png"),
                    "python": str(case_out / "python.png"),
                    "koharu_rust": str(case_out / "koharu_rust.png"),
                    "diff": str(case_out / "diff.png"),
                },
            }
        )
    passed_count = sum(1 for row in rows if row["passed"])
    report = {
        "fixture_dir": str(fixture_dir),
        "out_dir": str(out_dir),
        "threshold": threshold,
        "total": len(rows),
        "passed": passed_count,
        "failed": len(rows) - passed_count,
        "results": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_contact_sheet(rows, out_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Python and Koharu Rust typesetting backends.")
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--strict", action="store_true", help="Return non-zero when any case fails threshold/fallback.")
    args = parser.parse_args(argv)

    report = compare_fixture_dir(args.fixture_dir, args.out_dir, threshold=args.threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.strict and report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
