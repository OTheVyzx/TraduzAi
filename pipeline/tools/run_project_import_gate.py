"""Validate whether a TraduzAi pipeline output is importable by the editor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from PIL import Image
except Exception:  # pragma: no cover - defensive CLI boundary
    Image = None  # type: ignore[assignment]


IMAGE_LAYER_KEYS = ("base", "mask", "inpaint", "brush", "recovery", "rendered")
REQUIRED_IMAGE_LAYER_KEYS = ("base", "inpaint", "rendered")
BBOX_KEYS = ("render_bbox", "layout_bbox", "bbox", "source_bbox", "balloon_bbox")


def evaluate_project_import_gate(
    output_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    project_path = _resolve_project_path(output_path)
    project_dir = project_path.parent

    if not project_path.exists():
        return _write_result(
            _blocked_result(output_path, ["missing project.json"]),
            out_dir,
        )

    try:
        project = _load_project(project_path)
    except Exception as exc:
        return _write_result(
            _blocked_result(output_path, [f"could not load project.json: {exc}"]),
            out_dir,
        )

    reasons: list[str] = []
    warnings: list[str] = []
    pages = project.get("paginas")
    if not isinstance(pages, list):
        return _write_result(
            _blocked_result(output_path, ["project.json has no paginas list"]),
            out_dir,
        )
    if not pages:
        reasons.append("project has no pages")

    stats_mismatches = _validate_stats(project, pages)
    reasons.extend(stats_mismatches)

    text_layer_count = 0
    checked_image_count = 0
    missing_image_count = 0
    invalid_bbox_count = 0
    legacy_alias_mismatch_count = 0
    duplicate_text_id_count = 0
    image_dimension_mismatch_count = 0
    sampled_pages: list[dict[str, Any]] = []

    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            reasons.append(f"page {page_index}: page entry is not an object")
            continue

        page_number = _page_number(page, page_index)
        page_reasons, page_warnings, page_counts = _validate_page(
            project_dir,
            page,
            page_number,
        )
        reasons.extend(page_reasons)
        warnings.extend(page_warnings)
        text_layer_count += page_counts["text_layer_count"]
        checked_image_count += page_counts["checked_image_count"]
        missing_image_count += page_counts["missing_image_count"]
        invalid_bbox_count += page_counts["invalid_bbox_count"]
        legacy_alias_mismatch_count += page_counts["legacy_alias_mismatch_count"]
        duplicate_text_id_count += page_counts["duplicate_text_id_count"]
        image_dimension_mismatch_count += page_counts["image_dimension_mismatch_count"]
        if len(sampled_pages) < 10:
            sampled_pages.append(page_counts["sample"])

    status = "PASS" if not reasons else "FAIL"
    if Image is None:
        status = "BLOCK"
        reasons.insert(0, "PIL is unavailable; cannot decode referenced images")

    if not reasons:
        reasons.append("project.json can be loaded and referenced editor assets are present")

    result = {
        "source_path": str(output_path),
        "project_path": str(project_path),
        "gate": {
            "name": "project_import_contract",
            "status": status,
            "reasons": reasons[:50],
            "warnings": warnings[:50],
            "page_count": len(pages),
            "text_layer_count": text_layer_count,
            "checked_image_count": checked_image_count,
            "missing_image_count": missing_image_count,
            "invalid_bbox_count": invalid_bbox_count,
            "legacy_alias_mismatch_count": legacy_alias_mismatch_count,
            "duplicate_text_id_count": duplicate_text_id_count,
            "image_dimension_mismatch_count": image_dimension_mismatch_count,
            "stats_mismatch_count": len(stats_mismatches),
            "sampled_pages": sampled_pages,
        },
    }
    return _write_result(result, out_dir)


def _resolve_project_path(output_path: Path) -> Path:
    if output_path.name.lower() == "project.json":
        return output_path
    return output_path / "project.json"


def _load_project(project_path: Path) -> dict[str, Any]:
    with project_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("root JSON value must be an object")
    return payload


def _validate_stats(project: dict[str, Any], pages: list[Any]) -> list[str]:
    reasons: list[str] = []
    stats = project.get("estatisticas")
    if not isinstance(stats, dict):
        reasons.append("estatisticas is missing or not an object")
        return reasons

    total_pages = _int_or_none(stats.get("total_paginas"))
    if total_pages is not None and total_pages != len(pages):
        reasons.append("estatisticas.total_paginas does not match paginas length")

    actual_texts = sum(len(_text_layers(page)) for page in pages if isinstance(page, dict))
    total_texts = _int_or_none(stats.get("total_textos"))
    if total_texts is not None and total_texts != actual_texts:
        reasons.append("estatisticas.total_textos does not match text_layers length")
    return reasons


def _validate_page(
    project_dir: Path,
    page: dict[str, Any],
    page_number: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    reasons: list[str] = []
    warnings: list[str] = []
    counts = {
        "text_layer_count": 0,
        "checked_image_count": 0,
        "missing_image_count": 0,
        "invalid_bbox_count": 0,
        "legacy_alias_mismatch_count": 0,
        "duplicate_text_id_count": 0,
        "image_dimension_mismatch_count": 0,
        "sample": {
            "page": page_number,
            "text_layers": 0,
            "images": {},
        },
    }

    image_layers = page.get("image_layers")
    if not isinstance(image_layers, dict):
        reasons.append(f"page {page_number}: image_layers is missing or not an object")
        image_layers = {}

    for key in REQUIRED_IMAGE_LAYER_KEYS:
        layer = image_layers.get(key)
        if not isinstance(layer, dict) or not _non_empty_string(layer.get("path")):
            reasons.append(f"page {page_number}: required image layer {key} has no path")

    decoded_sizes: dict[str, tuple[int, int]] = {}
    for key in IMAGE_LAYER_KEYS:
        layer = image_layers.get(key)
        if not isinstance(layer, dict):
            if key in REQUIRED_IMAGE_LAYER_KEYS:
                reasons.append(f"page {page_number}: required image layer {key} is missing")
            continue
        rel_path = layer.get("path")
        if not _non_empty_string(rel_path):
            continue
        image_path = _resolve_asset_path(project_dir, rel_path)
        if not image_path.exists():
            counts["missing_image_count"] += 1
            reasons.append(f"missing referenced image: page {page_number} {key} -> {rel_path}")
            continue
        decoded = _decode_image_size(image_path)
        if decoded is None:
            reasons.append(f"page {page_number}: referenced image is not decodable: {rel_path}")
            continue
        counts["checked_image_count"] += 1
        decoded_sizes[key] = decoded
        counts["sample"]["images"][key] = {
            "path": str(rel_path),
            "width": decoded[0],
            "height": decoded[1],
        }

    base_size = decoded_sizes.get("base")
    for key in ("inpaint", "rendered"):
        current_size = decoded_sizes.get(key)
        if base_size is not None and current_size is not None and current_size != base_size:
            counts["image_dimension_mismatch_count"] += 1
            reasons.append(
                f"page {page_number}: image layer {key} dimensions do not match base"
            )
    for key in ("mask", "brush", "recovery"):
        current_size = decoded_sizes.get(key)
        if (
            base_size is not None
            and current_size is not None
            and current_size != base_size
            and current_size != (1, 1)
        ):
            warnings.append(
                f"page {page_number}: image layer {key} is neither full-size nor 1x1 placeholder"
            )

    for alias_key, layer_key in (("arquivo_original", "base"), ("arquivo_traduzido", "rendered")):
        alias = page.get(alias_key)
        layer = image_layers.get(layer_key)
        layer_path = layer.get("path") if isinstance(layer, dict) else None
        if _non_empty_string(alias) and _non_empty_string(layer_path):
            if _normalize_rel_path(alias) != _normalize_rel_path(layer_path):
                counts["legacy_alias_mismatch_count"] += 1
                reasons.append(f"page {page_number}: {alias_key} does not match image_layers.{layer_key}.path")

    text_layers = _text_layers(page)
    counts["text_layer_count"] = len(text_layers)
    counts["sample"]["text_layers"] = len(text_layers)

    seen_ids: set[str] = set()
    for text_index, layer in enumerate(text_layers, start=1):
        layer_id = layer.get("id")
        if _non_empty_string(layer_id):
            if layer_id in seen_ids:
                counts["duplicate_text_id_count"] += 1
                reasons.append(f"page {page_number}: duplicate text layer id {layer_id}")
            seen_ids.add(layer_id)
        else:
            warnings.append(f"page {page_number} text {text_index}: id will be generated on import")

        if not _first_valid_bbox(layer):
            counts["invalid_bbox_count"] += 1
            reasons.append(f"invalid text layer bbox: page {page_number} text {text_index}")

        if not _has_readable_text(layer):
            warnings.append(f"page {page_number} text {text_index}: no readable text fields")

    return reasons, warnings, counts


def _page_number(page: dict[str, Any], fallback: int) -> int:
    value = _int_or_none(page.get("numero"))
    return value if value is not None else fallback


def _text_layers(page: dict[str, Any]) -> list[dict[str, Any]]:
    layers = page.get("text_layers")
    if isinstance(layers, dict):
        layers = layers.get("texts")
    if isinstance(layers, list):
        return [layer for layer in layers if isinstance(layer, dict)]
    textos = page.get("textos")
    if isinstance(textos, list):
        return [layer for layer in textos if isinstance(layer, dict)]
    return []


def _resolve_asset_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_dir / path


def _decode_image_size(path: Path) -> tuple[int, int] | None:
    if Image is None:
        return None
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _first_valid_bbox(layer: dict[str, Any]) -> list[float] | None:
    for key in BBOX_KEYS:
        value = layer.get(key)
        parsed = _parse_bbox(value)
        if parsed is not None:
            return parsed
    return None


def _parse_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    coords: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        coords.append(float(item))
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        return None
    return coords


def _has_readable_text(layer: dict[str, Any]) -> bool:
    for key in ("translated", "traduzido", "original", "text"):
        value = layer.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_rel_path(value: str) -> str:
    return value.strip().replace("\\", "/").lower()


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _blocked_result(output_path: Path, reasons: list[str]) -> dict[str, Any]:
    return {
        "source_path": str(output_path),
        "project_path": str(_resolve_project_path(output_path)),
        "gate": {
            "name": "project_import_contract",
            "status": "BLOCK",
            "reasons": reasons,
            "warnings": [],
            "page_count": 0,
            "text_layer_count": 0,
            "checked_image_count": 0,
            "missing_image_count": 0,
            "invalid_bbox_count": 0,
            "legacy_alias_mismatch_count": 0,
            "duplicate_text_id_count": 0,
            "image_dimension_mismatch_count": 0,
            "stats_mismatch_count": 0,
            "sampled_pages": [],
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
    args = parser.parse_args(argv)

    result = evaluate_project_import_gate(args.output_dir, args.out)
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
