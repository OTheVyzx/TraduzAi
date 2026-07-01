"""Build a continuity-engine crop pack from a TraduzAI pipeline output."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageDraw


CRITICAL_FLAGS = {
    "mask_outside_balloon_critical",
    "missing_real_bubble_mask",
    "used_real_inpaint",
    "bubble_mask_overreach",
    "inpaint_artifact",
    "outline_damage_high",
}

DEFAULT_MAX_CASES = 400
DEFAULT_PER_TYPE_LIMIT = 80
DEFAULT_PAD = 18
MIN_CROP_SIDE = 24


def build_continuity_crop_pack(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    max_cases: int = DEFAULT_MAX_CASES,
    per_type_limit: int = DEFAULT_PER_TYPE_LIMIT,
    write_zip: bool = True,
    zip_name: str = "continuity_crop_pack_v001.zip",
) -> dict[str, Any]:
    """Export candidate crops from an existing pipeline output directory."""

    root = Path(run_dir)
    target = Path(out_dir)
    project_path = root / "project.json"
    if not project_path.exists():
        result = {
            "status": "BLOCK",
            "reasons": ["missing project.json"],
            "selected_count": 0,
            "output_dir": str(target),
        }
        _write_json(target / "summary.json", result)
        return result

    project = _load_json(project_path)
    pages = project.get("paginas") or project.get("pages") or []
    if not isinstance(pages, list):
        result = {
            "status": "BLOCK",
            "reasons": ["project pages are not a list"],
            "selected_count": 0,
            "output_dir": str(target),
        }
        _write_json(target / "summary.json", result)
        return result

    _prepare_output_dir(target)

    selected: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[int, int, int, int]]] = set()
    type_counts: Counter[str] = Counter()

    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        page_number = _safe_int(page.get("numero"), page_index)
        original_path = _resolve_page_image(root, page, "base", "originals", page_number)
        if original_path is None:
            discarded.append({"page_number": page_number, "reason": "missing_original"})
            continue
        try:
            original = Image.open(original_path).convert("RGB")
        except Exception as exc:
            discarded.append({"page_number": page_number, "reason": f"cannot_open_original: {exc}"})
            continue

        inpaint_path = _resolve_page_image(root, page, "inpaint", "images", page_number)
        inpaint = _open_rgb(inpaint_path)
        bubble_mask_page = _open_mask(_resolve_bubble_mask_path(root, page, None))

        records = _page_records(page)
        for record_index, record in enumerate(records, start=1):
            if len(selected) >= max(1, int(max_cases)):
                break
            candidate = _candidate_from_record(
                root=root,
                page=page,
                page_number=page_number,
                record=record,
                record_index=record_index,
                original=original,
                inpaint=inpaint,
                page_bubble_mask=bubble_mask_page,
            )
            if candidate is None:
                discarded.append(
                    {
                        "page_number": page_number,
                        "record_index": record_index,
                        "reason": "invalid_or_empty_candidate",
                    }
                )
                continue
            dedupe_key = (page_number, tuple(candidate["crop_bbox"]))
            if dedupe_key in seen:
                discarded.append(
                    {
                        "page_number": page_number,
                        "record_index": record_index,
                        "reason": "duplicate_crop_bbox",
                        "crop_bbox": list(candidate["crop_bbox"]),
                    }
                )
                continue
            estimated_type = str(candidate["metadata"]["estimated_type"])
            if estimated_type != "negative" and type_counts[estimated_type] >= max(1, int(per_type_limit)):
                discarded.append(
                    {
                        "page_number": page_number,
                        "record_index": record_index,
                        "reason": "per_type_limit",
                        "estimated_type": estimated_type,
                    }
                )
                continue
            seen.add(dedupe_key)
            type_counts[estimated_type] += 1
            selected.append(candidate)
        if len(selected) >= max(1, int(max_cases)):
            break

    cases = []
    for case_index, candidate in enumerate(selected, start=1):
        case_id = f"case_{case_index:06d}"
        case_dir = target / "candidates" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_case(case_dir, case_id, candidate)
        metadata = dict(candidate["metadata"])
        metadata["case_id"] = case_id
        metadata["files"] = {
            "original_crop": f"candidates/{case_id}/original_crop.png",
            "text_mask": f"candidates/{case_id}/text_mask.png",
            "bubble_mask": f"candidates/{case_id}/bubble_mask.png",
            "overlay_debug": f"candidates/{case_id}/overlay_debug.png",
            "metadata": f"candidates/{case_id}/metadata.json",
        }
        if candidate.get("current_inpaint_crop") is not None:
            metadata["files"]["current_inpaint"] = f"candidates/{case_id}/current_inpaint.png"
        cases.append(metadata)
        bucket = str(metadata.get("bucket") or "")
        if bucket in {"candidates_review", "candidates_negative"}:
            _write_json(target / bucket / f"{case_id}.json", {"case_id": case_id, "metadata": metadata})

    manifest = {
        "schema": "traduzai.continuity_crop_pack.v1",
        "source_run_dir": str(root),
        "selected_count": len(cases),
        "discarded_count": len(discarded),
        "counts_by_type": dict(Counter(str(case["estimated_type"]) for case in cases)),
        "cases": cases,
    }
    _write_json(target / "manifest.json", manifest)
    _write_json(target / "selection_report.json", {"discarded": discarded, "counts_by_type": manifest["counts_by_type"]})
    _write_json(target / "candidates_discarded" / "discarded.json", {"discarded": discarded})
    _write_readme(target / "README.md")

    archive_path = None
    if write_zip and cases:
        archive_path = target / zip_name
        _write_zip(target, archive_path)

    result = {
        "status": "PASS" if cases else "BLOCK",
        "reasons": ["crop pack generated"] if cases else ["no valid candidates"],
        "selected_count": len(cases),
        "discarded_count": len(discarded),
        "output_dir": str(target),
        "manifest": str(target / "manifest.json"),
        "zip": str(archive_path) if archive_path else "",
    }
    _write_json(target / "summary.json", result)
    return result


def _candidate_from_record(
    *,
    root: Path,
    page: dict[str, Any],
    page_number: int,
    record: dict[str, Any],
    record_index: int,
    original: Image.Image,
    inpaint: Image.Image | None,
    page_bubble_mask: Image.Image | None,
) -> dict[str, Any] | None:
    image_w, image_h = original.size
    bboxes = [
        _bbox4(record.get("bubble_mask_bbox")),
        _bbox4(record.get("balloon_bbox")),
        _bbox4(record.get("layout_bbox")),
        _bbox4(record.get("text_pixel_bbox")),
        _bbox4(record.get("source_bbox")),
        _bbox4(record.get("bbox")),
        _polygon_bbox(record.get("line_polygons")),
    ]
    crop_bbox = _expand_bbox(_union_bboxes([bbox for bbox in bboxes if bbox is not None]), image_w, image_h, DEFAULT_PAD)
    if crop_bbox is None:
        return None
    if crop_bbox[2] - crop_bbox[0] < MIN_CROP_SIDE or crop_bbox[3] - crop_bbox[1] < MIN_CROP_SIDE:
        return None

    text_mask = _build_text_mask(record, crop_bbox)
    if text_mask is None or not text_mask.getbbox():
        return None

    bubble_mask = _build_bubble_mask(root, page, record, page_bubble_mask, crop_bbox, original.size)
    original_crop = original.crop(crop_bbox)
    current_inpaint_crop = inpaint.crop(crop_bbox) if inpaint is not None and inpaint.size == original.size else None
    overlay = _build_overlay(original_crop, text_mask, bubble_mask)
    qa_flags = [str(flag) for flag in record.get("qa_flags") or [] if flag]
    estimated_type = _estimate_type(record, qa_flags)
    bucket = _bucket_for(estimated_type, qa_flags)
    metadata = {
        "page_number": page_number,
        "record_index": record_index,
        "record_id": str(record.get("id") or record.get("bubble_id") or f"record-{record_index}"),
        "source_collection": str(record.get("_source_collection") or "unknown"),
        "estimated_type": estimated_type,
        "bucket": bucket,
        "crop_bbox": list(crop_bbox),
        "text_bbox": _bbox4(record.get("text_pixel_bbox") or record.get("source_bbox") or record.get("bbox")),
        "balloon_bbox": _bbox4(record.get("balloon_bbox") or record.get("bubble_mask_bbox")),
        "bubble_mask_source": record.get("bubble_mask_source") or "",
        "bubble_mask_value": _safe_int(record.get("bubble_mask_value"), 0),
        "confidence": _safe_float(record.get("confidence"), 0.0),
        "layout_profile": record.get("layout_profile") or "",
        "block_profile": record.get("block_profile") or "",
        "content_class": record.get("content_class") or "",
        "tipo": record.get("tipo") or "",
        "route_action": record.get("route_action") or "",
        "qa_flags": qa_flags,
        "line_polygons_count": len(record.get("line_polygons") or []) if isinstance(record.get("line_polygons"), list) else 0,
    }
    return {
        "crop_bbox": crop_bbox,
        "original_crop": original_crop,
        "current_inpaint_crop": current_inpaint_crop,
        "text_mask": text_mask,
        "bubble_mask": bubble_mask,
        "overlay": overlay,
        "metadata": metadata,
    }


def _page_records(page: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for collection in ("text_layers", "textos", "inpaint_blocks"):
        value = page.get(collection)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                record = dict(item)
                record["_source_collection"] = collection
                records.append(record)
    return records


def _write_case(case_dir: Path, case_id: str, candidate: dict[str, Any]) -> None:
    candidate["original_crop"].save(case_dir / "original_crop.png")
    if candidate.get("current_inpaint_crop") is not None:
        candidate["current_inpaint_crop"].save(case_dir / "current_inpaint.png")
    candidate["text_mask"].save(case_dir / "text_mask.png")
    candidate["bubble_mask"].save(case_dir / "bubble_mask.png")
    candidate["overlay"].save(case_dir / "overlay_debug.png")
    metadata = dict(candidate["metadata"])
    metadata["case_id"] = case_id
    _write_json(case_dir / "metadata.json", metadata)


def _prepare_output_dir(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for dirname in ("candidates", "candidates_review", "candidates_negative", "candidates_discarded"):
        path = target / dirname
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for filename in (
        "manifest.json",
        "selection_report.json",
        "summary.json",
        "README.md",
        "continuity_crop_pack_v001.zip",
    ):
        path = target / filename
        if path.exists() and path.is_file():
            path.unlink()


def _build_text_mask(record: dict[str, Any], crop_bbox: tuple[int, int, int, int]) -> Image.Image | None:
    w = crop_bbox[2] - crop_bbox[0]
    h = crop_bbox[3] - crop_bbox[1]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    drew = False
    polygons = record.get("line_polygons")
    if isinstance(polygons, list):
        for polygon in polygons:
            points = _normalize_polygon(polygon, crop_bbox)
            if len(points) >= 3:
                draw.polygon(points, fill=255)
                drew = True
    if not drew:
        bbox = _bbox4(record.get("text_pixel_bbox") or record.get("source_bbox") or record.get("bbox"))
        if bbox is not None:
            draw.rectangle(_local_bbox(bbox, crop_bbox), fill=255)
            drew = True
    return mask if drew else None


def _build_bubble_mask(
    root: Path,
    page: dict[str, Any],
    record: dict[str, Any],
    page_bubble_mask: Image.Image | None,
    crop_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> Image.Image:
    mask = _open_mask(_resolve_bubble_mask_path(root, page, record)) or page_bubble_mask
    if mask is not None and mask.size == image_size:
        crop = mask.crop(crop_bbox).convert("L")
        mask_value = _safe_int(record.get("bubble_mask_value"), 0)
        if mask_value > 0:
            crop = crop.point(lambda px: 255 if int(px) == mask_value else 0)
        else:
            crop = crop.point(lambda px: 255 if int(px) > 0 else 0)
        if crop.getbbox():
            return crop

    w = crop_bbox[2] - crop_bbox[0]
    h = crop_bbox[3] - crop_bbox[1]
    fallback = Image.new("L", (w, h), 0)
    bbox = _bbox4(record.get("balloon_bbox") or record.get("bubble_mask_bbox") or record.get("bbox"))
    if bbox is not None:
        ImageDraw.Draw(fallback).rectangle(_local_bbox(bbox, crop_bbox), fill=255)
    return fallback


def _build_overlay(original_crop: Image.Image, text_mask: Image.Image, bubble_mask: Image.Image) -> Image.Image:
    base = original_crop.convert("RGBA")
    bubble = Image.new("RGBA", base.size, (0, 160, 255, 0))
    bubble.putalpha(bubble_mask.point(lambda px: 70 if int(px) > 0 else 0))
    text = Image.new("RGBA", base.size, (255, 40, 40, 0))
    text.putalpha(text_mask.point(lambda px: 160 if int(px) > 0 else 0))
    return Image.alpha_composite(Image.alpha_composite(base, bubble), text).convert("RGB")


def _estimate_type(record: dict[str, Any], qa_flags: list[str]) -> str:
    flag_set = set(qa_flags)
    if flag_set & CRITICAL_FLAGS or record.get("bubble_mask_error"):
        return "negative"
    values = {
        str(record.get("layout_profile") or "").lower(),
        str(record.get("block_profile") or "").lower(),
        str(record.get("balloon_type") or "").lower(),
        str(record.get("background_type") or "").lower(),
    }
    joined = " ".join(values)
    if "connected" in joined or record.get("connected_lobe_bboxes") or record.get("balloon_subregions"):
        return "connected_balloon"
    if "speed" in joined:
        return "speed_line"
    if "panel" in joined or record.get("content_class") == "panel_line":
        return "panel_line"
    if record.get("bubble_mask_source") or record.get("balloon_bbox") or "balloon" in joined or "white" in joined:
        return "balloon_fill"
    if record.get("line_polygons"):
        return "balloon_outline"
    return "candidates_review"


def _bucket_for(estimated_type: str, qa_flags: list[str]) -> str:
    if estimated_type == "negative" or set(qa_flags) & CRITICAL_FLAGS:
        return "candidates_negative"
    if estimated_type == "candidates_review":
        return "candidates_review"
    return "keep_good"


def _resolve_page_image(root: Path, page: dict[str, Any], layer_key: str, fallback_dir: str, page_number: int) -> Path | None:
    image_layers = page.get("image_layers") if isinstance(page.get("image_layers"), dict) else {}
    layer = image_layers.get(layer_key) if isinstance(image_layers, dict) else None
    if isinstance(layer, dict) and isinstance(layer.get("path"), str):
        candidate = root / layer["path"]
        if candidate.exists():
            return candidate
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = root / fallback_dir / f"{page_number:03d}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _resolve_bubble_mask_path(root: Path, page: dict[str, Any], record: dict[str, Any] | None) -> Path | None:
    if record:
        for key in ("bubble_mask_layer_path", "bubble_mask_path"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = root / candidate
                if candidate.exists():
                    return candidate
    image_layers = page.get("image_layers") if isinstance(page.get("image_layers"), dict) else {}
    layer = image_layers.get("bubble_mask") if isinstance(image_layers, dict) else None
    if isinstance(layer, dict) and isinstance(layer.get("path"), str):
        candidate = root / layer["path"]
        if candidate.exists():
            return candidate
    return None


def _open_rgb(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _open_mask(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    try:
        return Image.open(path).convert("L")
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_readme(path: Path) -> None:
    path.write_text(
        "# TraduzAI Continuity Crop Pack\n\n"
        "Each case contains `original_crop.png`, `text_mask.png`, `bubble_mask.png`, "
        "`overlay_debug.png`, optional `current_inpaint.png`, and `metadata.json`.\n\n"
        "This pack is for continuity-engine development only. It does not define inpaint success.\n",
        encoding="utf-8",
    )


def _write_zip(root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path == archive_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(root).as_posix())


def _bbox4(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _polygon_bbox(value: Any) -> tuple[int, int, int, int] | None:
    points: list[tuple[float, float]] = []
    if not isinstance(value, list):
        return None
    for polygon in value:
        if not isinstance(polygon, list):
            continue
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    points.append((float(point[0]), float(point[1])))
                except Exception:
                    continue
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return _bbox4([min(xs), min(ys), max(xs), max(ys)])


def _union_bboxes(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not bboxes:
        return None
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _expand_bbox(
    bbox: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    pad: int,
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad))


def _local_bbox(bbox: tuple[int, int, int, int], crop_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        max(0, bbox[0] - crop_bbox[0]),
        max(0, bbox[1] - crop_bbox[1]),
        max(0, bbox[2] - crop_bbox[0]),
        max(0, bbox[3] - crop_bbox[1]),
    )


def _normalize_polygon(polygon: Any, crop_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    points = []
    if not isinstance(polygon, list):
        return points
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = int(round(float(point[0]))) - crop_bbox[0]
            y = int(round(float(point[1]))) - crop_bbox[1]
        except Exception:
            continue
        points.append((x, y))
    return points


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Pipeline output directory containing project.json")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the crop pack")
    parser.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES)
    parser.add_argument("--per-type-limit", type=int, default=DEFAULT_PER_TYPE_LIMIT)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv)

    result = build_continuity_crop_pack(
        args.run_dir,
        args.out,
        max_cases=args.max_cases,
        per_type_limit=args.per_type_limit,
        write_zip=not args.no_zip,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
