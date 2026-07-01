from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sfx.ocr_probe import probe_sfx_candidate_ocr
from vision_stack.runtime import _get_detector
from vision_stack.sfx_detector import (
    detect_sfx_candidates,
    filter_sfx_candidates_after_ocr,
    merge_sfx_candidates,
    text_blocks_to_sfx_candidates,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight SFX detection probe with visual overlays.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--anime-conf", type=float, default=0.02)
    parser.add_argument("--comic-conf", type=float, default=0.02)
    parser.add_argument("--profile", default="quality")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ocr-probe", action="store_true", help="Run CJK crop OCR probe before final filtering.")
    parser.add_argument("--expect", default="", help="Optional JSON expectations file for regression validation.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = _collect_images(input_path)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found: {input_path}")

    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    final_crop_rows: list[tuple[str, np.ndarray]] = []
    pages: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        image_rgb = _load_rgb(image_path)
        raw_visual = detect_sfx_candidates(image_rgb)
        raw_text = _detect_text_candidates(
            image_rgb,
            anime_conf=float(args.anime_conf),
            comic_conf=float(args.comic_conf),
            profile=str(args.profile),
        )
        raw_merged = merge_sfx_candidates(raw_visual + raw_text)
        probed = raw_merged
        if args.ocr_probe:
            probed = []
            for candidate in raw_merged:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    probed.append(probe_sfx_candidate_ocr(candidate, image_rgb))
        final = filter_sfx_candidates_after_ocr(probed, image_rgb)

        page_dir = out_dir / f"{index:03d}_{_safe_name(image_path.stem)}"
        page_dir.mkdir(parents=True, exist_ok=True)
        overlay = _draw_overlay(image_rgb, raw_merged, final)
        _save_rgb(page_dir / "overlay.png", overlay)
        _write_crops(page_dir / "crops", image_rgb, final)
        final_crop_rows.extend(_collect_final_crop_rows(image_path.name, image_rgb, final))
        page = {
            "image": str(image_path),
            "overlay": str(page_dir / "overlay.png"),
            "raw_visual_count": len(raw_visual),
            "raw_text_count": len(raw_text),
            "raw_merged_count": len(raw_merged),
            "final_count": len(final),
            "final_candidates": [_summary(item) for item in final],
            "raw_candidates": [_summary(item) for item in raw_merged],
        }
        (page_dir / "summary.json").write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
        pages.append(page)
        rows.append((image_path.name, image_rgb, overlay))

    contact = _make_contact_sheet(rows)
    contact_path = out_dir / "sfx_detection_contact_sheet.png"
    _save_rgb(contact_path, contact)
    final_crops_path = out_dir / "sfx_final_crops_sheet.png"
    if final_crop_rows:
        _save_rgb(final_crops_path, _make_final_crops_sheet(final_crop_rows))
    summary = {
        "input": str(input_path),
        "output": str(out_dir),
        "anime_conf": float(args.anime_conf),
        "comic_conf": float(args.comic_conf),
        "ocr_probe": bool(args.ocr_probe),
        "page_count": len(pages),
        "total_raw_visual": sum(page["raw_visual_count"] for page in pages),
        "total_raw_text": sum(page["raw_text_count"] for page in pages),
        "total_raw_merged": sum(page["raw_merged_count"] for page in pages),
        "total_final": sum(page["final_count"] for page in pages),
        "contact_sheet": str(contact_path),
        "final_crops_sheet": str(final_crops_path) if final_crop_rows else "",
        "pages": pages,
    }
    if args.expect:
        summary["expectations"] = _validate_expectations(summary, Path(args.expect))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _collect_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def _load_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))


def _detect_text_candidates(
    image_rgb: np.ndarray,
    *,
    anime_conf: float,
    comic_conf: float,
    profile: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for model, conf, source, min_area in (
        ("anime-text-yolo-n", anime_conf, "anime_text_yolo_low_conf", 0.010),
        ("comic-text-detector", comic_conf, "comic_text_detector_fallback", 0.0015),
    ):
        detector = _get_detector(profile, model=model)
        blocks = detector.detect(image_rgb, conf_threshold=float(conf))
        candidates.extend(
            text_blocks_to_sfx_candidates(
                image_rgb,
                blocks,
                source=source,
                min_confidence=float(conf),
                min_low_conf_area_ratio=float(min_area),
            )
        )
    return merge_sfx_candidates(candidates)


def _draw_overlay(image_rgb: np.ndarray, raw: list[dict[str, Any]], final: list[dict[str, Any]]) -> np.ndarray:
    rendered = Image.fromarray(image_rgb.astype(np.uint8), "RGB").convert("RGBA")
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    final_boxes = {tuple(_bbox(item) or []) for item in final}
    for index, candidate in enumerate(raw, start=1):
        bbox = _bbox(candidate)
        if bbox is None:
            continue
        is_final = tuple(bbox) in final_boxes
        color = (0, 220, 80, 255) if is_final else (255, 188, 0, 255)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3 if is_final else 2)
        source = str((candidate.get("sfx") or {}).get("visual_source") or "")[:8]
        label = f"{index}:{float(candidate.get('confidence') or 0):.2f} {source}"
        draw.text((x1, max(0, y1 - 12)), label, fill=color, font=font)
    return np.asarray(rendered.convert("RGB"))


def _write_crops(out_dir: Path, image_rgb: np.ndarray, candidates: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates, start=1):
        bbox = _bbox(candidate)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size:
            _save_rgb(out_dir / f"{index:03d}.png", crop)


def _collect_final_crop_rows(name: str, image_rgb: np.ndarray, candidates: list[dict[str, Any]]) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    for index, candidate in enumerate(candidates, start=1):
        bbox = _bbox(candidate)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size:
            label = f"{_short_image_label(name)} #{index} {x1},{y1},{x2},{y2}"
            rows.append((label, crop))
    return rows


def _make_contact_sheet(rows: list[tuple[str, np.ndarray, np.ndarray]]) -> np.ndarray:
    cell_w = 300
    label_h = 18
    row_images = []
    font = ImageFont.load_default()
    for name, original, overlay in rows:
        cells = []
        max_h = 0
        for label, image in (("original", original), ("overlay", overlay)):
            h, w = image.shape[:2]
            scale = min(1.0, cell_w / float(max(1, w)))
            target = (max(1, int(w * scale)), max(1, int(h * scale)))
            resized = np.asarray(Image.fromarray(image).resize(target, Image.Resampling.LANCZOS))
            cells.append((label, resized))
            max_h = max(max_h, resized.shape[0])
        canvas = Image.new("RGB", (cell_w * 2, max_h + label_h * 2), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), name[:80], fill=(0, 0, 0), font=font)
        for idx, (label, image) in enumerate(cells):
            x = idx * cell_w
            draw.text((x + 4, label_h), label, fill=(0, 0, 0), font=font)
            canvas.paste(Image.fromarray(image), (x, label_h * 2))
        row_images.append(np.asarray(canvas))
    width = max(item.shape[1] for item in row_images)
    height = sum(item.shape[0] for item in row_images)
    sheet = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 0
    for image in row_images:
        sheet[y : y + image.shape[0], : image.shape[1]] = image
        y += image.shape[0]
    return sheet


def _make_final_crops_sheet(rows: list[tuple[str, np.ndarray]]) -> np.ndarray:
    cell_w = 220
    cell_h = 190
    label_h = 28
    columns = 4
    font = ImageFont.load_default()
    total_rows = int(np.ceil(len(rows) / float(columns)))
    canvas = Image.new("RGB", (cell_w * columns, cell_h * total_rows), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, (label, crop) in enumerate(rows):
        col = index % columns
        row = index // columns
        x = col * cell_w
        y = row * cell_h
        draw.text((x + 4, y + 3), label[:42], fill=(0, 0, 0), font=font)
        h, w = crop.shape[:2]
        scale = min((cell_w - 12) / float(max(1, w)), (cell_h - label_h - 8) / float(max(1, h)), 1.0)
        target = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = Image.fromarray(crop.astype(np.uint8)).resize(target, Image.Resampling.LANCZOS)
        canvas.paste(resized, (x + 6, y + label_h))
    return np.asarray(canvas)


def _summary(item: dict[str, Any]) -> dict[str, Any]:
    sfx = item.get("sfx") if isinstance(item.get("sfx"), dict) else {}
    sfx_ocr = item.get("sfx_ocr") if isinstance(item.get("sfx_ocr"), dict) else {}
    return {
        "bbox": _bbox(item),
        "confidence": item.get("confidence"),
        "detector": item.get("detector"),
        "visual_source": sfx.get("visual_source"),
        "ocr_status": sfx_ocr.get("status"),
        "recognized_text": item.get("recognized_text") or item.get("text") or "",
        "qa_flags": item.get("qa_flags") or [],
    }


def _validate_expectations(summary: dict[str, Any], expect_path: Path) -> dict[str, Any]:
    expectations = json.loads(expect_path.read_text(encoding="utf-8"))
    page_expectations = expectations.get("pages") if isinstance(expectations, dict) else None
    if not isinstance(page_expectations, dict):
        raise SystemExit(f"Invalid expectations file: {expect_path}")
    min_iou = float(expectations.get("min_iou", 0.45))
    require_all_pages = bool(expectations.get("require_all_pages", False))
    require_reviewed_pages = bool(expectations.get("require_reviewed_pages", False))
    failures: list[str] = []
    pages_by_name = {Path(page["image"]).name: page for page in summary.get("pages", [])}
    if require_all_pages:
        expected_names = set(str(name) for name in page_expectations)
        actual_names = set(pages_by_name)
        for image_name in sorted(actual_names - expected_names):
            failures.append(f"{image_name}: page missing from expectations")
        for image_name in sorted(expected_names - actual_names):
            failures.append(f"{image_name}: page not found in probe output")
    for image_name, expected in page_expectations.items():
        page = pages_by_name.get(str(image_name))
        if not page:
            if not require_all_pages:
                failures.append(f"{image_name}: page not found in probe output")
            continue
        if require_reviewed_pages and expected.get("reviewed") is not True:
            failures.append(f"{image_name}: expectation page must set reviewed=true")
        final = page.get("final_candidates") or []
        final_boxes = [candidate.get("bbox") for candidate in final if candidate.get("bbox")]
        min_count = expected.get("min_final_count")
        max_count = expected.get("max_final_count")
        if min_count is not None and len(final_boxes) < int(min_count):
            failures.append(f"{image_name}: final_count {len(final_boxes)} < min {int(min_count)}")
        if max_count is not None and len(final_boxes) > int(max_count):
            failures.append(f"{image_name}: final_count {len(final_boxes)} > max {int(max_count)}")
        for required in expected.get("required_boxes") or []:
            bbox = required.get("bbox") if isinstance(required, dict) else required
            best = max((_iou(bbox, box) for box in final_boxes), default=0.0)
            if best < min_iou:
                label = required.get("label", bbox) if isinstance(required, dict) else bbox
                failures.append(f"{image_name}: missing required {label} best_iou={best:.3f}")
        for forbidden in expected.get("forbidden_boxes") or []:
            bbox = forbidden.get("bbox") if isinstance(forbidden, dict) else forbidden
            best = max((_iou(bbox, box) for box in final_boxes), default=0.0)
            if best >= min_iou:
                label = forbidden.get("label", bbox) if isinstance(forbidden, dict) else bbox
                failures.append(f"{image_name}: forbidden {label} matched best_iou={best:.3f}")
    result = {
        "file": str(expect_path),
        "min_iou": min_iou,
        "require_all_pages": require_all_pages,
        "require_reviewed_pages": require_reviewed_pages,
        "reviewed_page_count": sum(1 for expected in page_expectations.values() if isinstance(expected, dict) and expected.get("reviewed") is True),
        "expected_page_count": len(page_expectations),
        "actual_page_count": len(pages_by_name),
        "passed": not failures,
        "failures": failures,
    }
    if failures:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _iou(a: Any, b: Any) -> float:
    box_a = _coerce_expect_bbox(a)
    box_b = _coerce_expect_bbox(b)
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1.0, area_a + area_b - inter)


def _coerce_expect_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox(item: dict[str, Any]) -> list[int] | None:
    value = item.get("bbox") or item.get("text_pixel_bbox") or item.get("source_bbox")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)[:80]


def _short_image_label(value: str) -> str:
    stem = Path(value).stem
    if len(stem) > 18:
        stem = stem[-18:]
    return stem + Path(value).suffix


if __name__ == "__main__":
    raise SystemExit(main())
