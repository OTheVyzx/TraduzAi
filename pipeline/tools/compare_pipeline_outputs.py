"""Compare two TraduzAi pipeline outputs for structural regressions."""

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


def evaluate_pipeline_output_compare(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    baseline_path = Path(baseline_dir)
    candidate_path = Path(candidate_dir)
    reasons: list[str] = []

    baseline_project_path = baseline_path / "project.json"
    candidate_project_path = candidate_path / "project.json"
    if not baseline_project_path.exists():
        reasons.append("baseline missing project.json")
    if not candidate_project_path.exists():
        reasons.append("candidate missing project.json")
    if reasons:
        return _write_result(
            _blocked_result(baseline_path, candidate_path, reasons),
            out_dir,
        )

    try:
        baseline_project = _load_project(baseline_project_path)
        candidate_project = _load_project(candidate_project_path)
    except Exception as exc:
        return _write_result(
            _blocked_result(
                baseline_path,
                candidate_path,
                [f"could not load project.json: {exc}"],
            ),
            out_dir,
        )

    baseline_pages = _page_map(baseline_project)
    candidate_pages = _page_map(candidate_project)
    baseline_page_count = len(baseline_pages)
    candidate_page_count = len(candidate_pages)
    baseline_text_count = sum(len(_text_layers(page)) for page in baseline_pages.values())
    candidate_text_count = sum(len(_text_layers(page)) for page in candidate_pages.values())
    baseline_inpaint_count = sum(
        len(_list_field(page, "inpaint_blocks")) for page in baseline_pages.values()
    )
    candidate_inpaint_count = sum(
        len(_list_field(page, "inpaint_blocks")) for page in candidate_pages.values()
    )
    baseline_translated_regions = _translated_region_count(
        baseline_project,
        baseline_text_count,
    )
    candidate_translated_regions = _translated_region_count(
        candidate_project,
        candidate_text_count,
    )
    audited_skip_count = _audited_skip_count(candidate_pages.values())
    macro_shadow_page_count = sum(
        1
        for page in candidate_pages.values()
        if isinstance(page.get("page_profile"), dict)
        and isinstance(page["page_profile"].get("macro_ocr_shadow"), dict)
    )
    dimension_mismatches = _compare_image_dimensions(
        baseline_path,
        candidate_path,
        baseline_pages,
        candidate_pages,
    )

    status = "PASS"
    if baseline_page_count != candidate_page_count:
        status = "FAIL"
        reasons.append("page count changed")
    if sorted(baseline_pages) != sorted(candidate_pages):
        status = "FAIL"
        reasons.append("page numbers changed")
    if baseline_text_count != candidate_text_count:
        missing_texts = max(0, baseline_text_count - candidate_text_count)
        if missing_texts > audited_skip_count:
            status = "FAIL"
            reasons.append("text count changed without matching audited skips")
    if candidate_text_count > baseline_text_count:
        status = "FAIL"
        reasons.append("candidate has extra text layers")
    if baseline_inpaint_count != candidate_inpaint_count and audited_skip_count == 0:
        status = "FAIL"
        reasons.append("inpaint block count changed without audited skips")
    if baseline_translated_regions != candidate_translated_regions and audited_skip_count == 0:
        status = "FAIL"
        reasons.append("translated region count changed without audited skips")
    if dimension_mismatches:
        status = "FAIL"
        reasons.append("final image dimensions changed")

    if not _has_final_images(candidate_path):
        status = "BLOCK"
        reasons.append("candidate missing final images")
    if not _has_final_images(baseline_path):
        status = "BLOCK"
        reasons.append("baseline missing final images")

    if not reasons:
        reasons.append("pipeline output structure and final image dimensions match")

    result = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "gate": {
            "name": "pipeline_output_compare",
            "status": status,
            "reasons": reasons,
            "page_count": {
                "baseline": baseline_page_count,
                "candidate": candidate_page_count,
            },
            "text_count": {
                "baseline": baseline_text_count,
                "candidate": candidate_text_count,
            },
            "inpaint_block_count": {
                "baseline": baseline_inpaint_count,
                "candidate": candidate_inpaint_count,
            },
            "translated_region_count": {
                "baseline": baseline_translated_regions,
                "candidate": candidate_translated_regions,
            },
            "audited_skip_count": audited_skip_count,
            "macro_ocr_shadow_page_count": macro_shadow_page_count,
            "image_dimension_mismatch_count": len(dimension_mismatches),
            "image_dimension_mismatches": dimension_mismatches[:20],
        },
    }
    return _write_result(result, out_dir)


def _blocked_result(
    baseline_path: Path,
    candidate_path: Path,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "gate": {
            "name": "pipeline_output_compare",
            "status": "BLOCK",
            "reasons": reasons,
            "page_count": {"baseline": 0, "candidate": 0},
            "text_count": {"baseline": 0, "candidate": 0},
            "inpaint_block_count": {"baseline": 0, "candidate": 0},
            "translated_region_count": {"baseline": 0, "candidate": 0},
            "audited_skip_count": 0,
            "macro_ocr_shadow_page_count": 0,
            "image_dimension_mismatch_count": 0,
            "image_dimension_mismatches": [],
        },
    }


def _load_project(project_path: Path) -> dict[str, Any]:
    with project_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {project_path}")
    return payload


def _page_map(project: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    for index, page in enumerate(project.get("paginas") or [], start=1):
        if not isinstance(page, dict):
            continue
        number = page.get("numero", index)
        try:
            page_number = int(number)
        except (TypeError, ValueError):
            page_number = index
        pages[page_number] = page
    return pages


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


def _list_field(page: dict[str, Any], key: str) -> list[Any]:
    value = page.get(key)
    return value if isinstance(value, list) else []


def _translated_region_count(project: dict[str, Any], fallback: int) -> int:
    stats = project.get("estatisticas")
    if isinstance(stats, dict):
        for key in ("translated_regions", "regioes_traduzidas", "total_textos"):
            value = stats.get(key)
            if isinstance(value, int):
                return value
    return fallback


def _audited_skip_count(pages: Any) -> int:
    count = 0
    for page in pages:
        for layer in _text_layers(page):
            if not layer.get("skip_processing"):
                continue
            if _skip_has_audit(layer):
                count += 1
    return count


def _skip_has_audit(layer: dict[str, Any]) -> bool:
    for key in (
        "skip_reason",
        "skip_motivo",
        "smart_skip_reason",
        "smart_skip_decision",
        "_smart_skip_shadow",
    ):
        if layer.get(key):
            return True
    return False


def _compare_image_dimensions(
    baseline_path: Path,
    candidate_path: Path,
    baseline_pages: dict[int, dict[str, Any]],
    candidate_pages: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    if Image is None:
        return [
            {
                "page_number": None,
                "reason": "Pillow is not available; cannot compare final image dimensions",
            }
        ]

    for page_number, baseline_page in baseline_pages.items():
        candidate_page = candidate_pages.get(page_number)
        if candidate_page is None:
            mismatches.append({"page_number": page_number, "reason": "candidate page missing"})
            continue
        baseline_image = _final_image_path(baseline_path, baseline_page, page_number)
        candidate_image = _final_image_path(candidate_path, candidate_page, page_number)
        if baseline_image is None:
            mismatches.append(
                {"page_number": page_number, "reason": "baseline final image missing"}
            )
            continue
        if candidate_image is None:
            mismatches.append(
                {"page_number": page_number, "reason": "candidate final image missing"}
            )
            continue
        baseline_size = _image_size(baseline_image)
        candidate_size = _image_size(candidate_image)
        if baseline_size != candidate_size:
            mismatches.append(
                {
                    "page_number": page_number,
                    "baseline": list(baseline_size),
                    "candidate": list(candidate_size),
                    "baseline_image": str(baseline_image),
                    "candidate_image": str(candidate_image),
                }
            )
    return mismatches


def _final_image_path(
    output_path: Path,
    page: dict[str, Any],
    page_number: int,
) -> Path | None:
    candidates: list[Path] = []
    arquivo_traduzido = page.get("arquivo_traduzido")
    if isinstance(arquivo_traduzido, str) and arquivo_traduzido:
        candidates.append(output_path / arquivo_traduzido)
    image_layers = page.get("image_layers")
    if isinstance(image_layers, dict):
        rendered = image_layers.get("rendered")
        if isinstance(rendered, dict):
            rendered_path = rendered.get("path")
            if isinstance(rendered_path, str) and rendered_path:
                candidates.append(output_path / rendered_path)
    for suffix in (".jpg", ".png", ".jpeg"):
        candidates.append(output_path / "translated" / f"{page_number:03d}{suffix}")
        candidates.append(output_path / "images" / f"{page_number:03d}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:  # type: ignore[union-attr]
        return image.size


def _has_final_images(output_path: Path) -> bool:
    for folder_name in ("translated", "images"):
        folder = output_path / folder_name
        if folder.exists() and any(folder.glob("*.jpg")):
            return True
        if folder.exists() and any(folder.glob("*.png")):
            return True
        if folder.exists() and any(folder.glob("*.jpeg")):
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
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = evaluate_pipeline_output_compare(
        args.baseline_dir,
        args.candidate_dir,
        args.out,
    )
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
