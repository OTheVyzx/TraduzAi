"""Export an HTML visual review sheet for two TraduzAi pipeline outputs."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from PIL import Image, ImageChops
except Exception:  # pragma: no cover - defensive CLI boundary
    Image = None  # type: ignore[assignment]
    ImageChops = None  # type: ignore[assignment]


def export_visual_review_sheet(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    out_path: str | Path,
    *,
    max_pages: int = 10,
    max_crops_per_page: int = 6,
) -> dict[str, Any]:
    baseline_path = Path(baseline_dir)
    candidate_path = Path(candidate_dir)
    output_html = Path(out_path)
    reasons: list[str] = []

    baseline_project_path = baseline_path / "project.json"
    candidate_project_path = candidate_path / "project.json"
    if not baseline_project_path.exists():
        reasons.append("baseline missing project.json")
    if not candidate_project_path.exists():
        reasons.append("candidate missing project.json")
    if reasons:
        result = {
            "status": "BLOCK",
            "reasons": reasons,
            "selected_pages": [],
            "output_html": str(output_html),
        }
        _write_summary(output_html, result)
        return result

    try:
        baseline_project = _load_project(baseline_project_path)
        candidate_project = _load_project(candidate_project_path)
    except Exception as exc:
        result = {
            "status": "BLOCK",
            "reasons": [f"could not load project.json: {exc}"],
            "selected_pages": [],
            "output_html": str(output_html),
        }
        _write_summary(output_html, result)
        return result

    baseline_pages = _page_map(baseline_project)
    candidate_pages = _page_map(candidate_project)
    selected_reports = _select_review_pages(candidate_pages, max_pages=max_pages)
    selected_pages = [report["page_number"] for report in selected_reports]

    assets_dir = output_html.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    page_results: list[dict[str, Any]] = []
    for report in selected_reports:
        page_number = int(report["page_number"])
        baseline_image = _final_image_path(
            baseline_path,
            baseline_pages.get(page_number, {}),
            page_number,
        )
        candidate_image = _final_image_path(
            candidate_path,
            candidate_pages.get(page_number, {}),
            page_number,
        )
        baseline_asset = _copy_review_image(
            baseline_image,
            assets_dir / f"baseline_{page_number:03d}.jpg",
        )
        candidate_asset = _copy_review_image(
            candidate_image,
            assets_dir / f"candidate_{page_number:03d}.jpg",
        )
        pixel_diff_rate = _pixel_diff_rate(baseline_image, candidate_image)
        crop_pairs = _export_text_crops(
            baseline_image,
            candidate_image,
            candidate_pages.get(page_number, {}),
            assets_dir,
            page_number,
            max_crops=max_crops_per_page,
        )
        enriched_report = dict(report)
        enriched_report["pixel_diff_rate"] = pixel_diff_rate
        enriched_report["crop_pairs"] = crop_pairs
        page_results.append(
            {
                "page_number": page_number,
                "pixel_diff_rate": pixel_diff_rate,
                "crop_count": len(crop_pairs),
                "different_text_rate": float(report.get("different_text_rate") or 0.0),
                "missing_text_rate": float(report.get("missing_text_rate") or 0.0),
                "fallback_rate": float(report.get("fallback_rate") or 0.0),
            }
        )
        rows.append(
            _render_page_row(
                page_number,
                enriched_report,
                baseline_asset,
                candidate_asset,
            )
        )

    status = "PASS" if selected_pages else "BLOCK"
    if not selected_pages:
        reasons.append("no pages available for visual review")
    else:
        reasons.append("visual review sheet generated")

    html_text = _render_html(rows, baseline_path, candidate_path)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")

    result = {
        "status": status,
        "reasons": reasons,
        "selected_pages": selected_pages,
        "output_html": str(output_html),
        "asset_count": len(selected_pages) * 2
        + sum(int(report["crop_count"]) * 2 for report in page_results),
        "page_reports": page_results,
    }
    _write_summary(output_html, result)
    return result


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
        try:
            page_number = int(page.get("numero", index))
        except (TypeError, ValueError):
            page_number = index
        pages[page_number] = page
    return pages


def _select_review_pages(
    candidate_pages: dict[int, dict[str, Any]],
    *,
    max_pages: int,
) -> list[dict[str, Any]]:
    reports = _macro_page_reports(candidate_pages)
    if not reports:
        reports = [
            {
                "page_number": page_number,
                "different_text_rate": 0.0,
                "missing_text_rate": 0.0,
                "fallback_rate": 0.0,
                "reason": "fallback selection",
            }
            for page_number in sorted(candidate_pages)
        ]

    def sort_key(report: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(report.get("missing_text_rate") or 0.0),
            float(report.get("different_text_rate") or 0.0),
            float(report.get("fallback_rate") or 0.0),
            int(report.get("page_number") or 0),
        )

    sorted_reports = sorted(reports, key=sort_key, reverse=True)
    return sorted_reports[: max(1, int(max_pages))]


def _macro_page_reports(candidate_pages: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for page in candidate_pages.values():
        profile = page.get("page_profile")
        if not isinstance(profile, dict):
            continue
        shadow = profile.get("macro_ocr_shadow")
        if not isinstance(shadow, dict):
            continue
        page_reports = shadow.get("page_reports")
        if not isinstance(page_reports, list):
            continue
        for report in page_reports:
            if isinstance(report, dict) and report.get("page_number") is not None:
                reports.append(report)
    return reports


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


def _copy_review_image(source: Path | None, destination: Path) -> str:
    if source is None:
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination.name


def _export_text_crops(
    baseline_image: Path | None,
    candidate_image: Path | None,
    candidate_page: dict[str, Any],
    assets_dir: Path,
    page_number: int,
    *,
    max_crops: int,
) -> list[dict[str, Any]]:
    if baseline_image is None or candidate_image is None or Image is None:
        return []
    crop_pairs: list[dict[str, Any]] = []
    layers = _text_layers(candidate_page)
    with Image.open(baseline_image) as baseline, Image.open(candidate_image) as candidate:
        baseline_rgb = baseline.convert("RGB")
        candidate_rgb = candidate.convert("RGB")
        for index, layer in enumerate(layers[: max(0, int(max_crops))], start=1):
            bbox = _layer_bbox(layer)
            if bbox is None:
                continue
            baseline_crop = _crop_image(baseline_rgb, bbox, padding=12)
            candidate_crop = _crop_image(candidate_rgb, bbox, padding=12)
            baseline_name = f"baseline_{page_number:03d}_crop_{index:03d}.jpg"
            candidate_name = f"candidate_{page_number:03d}_crop_{index:03d}.jpg"
            baseline_crop.save(assets_dir / baseline_name, quality=92)
            candidate_crop.save(assets_dir / candidate_name, quality=92)
            crop_pairs.append(
                {
                    "index": index,
                    "bbox": bbox,
                    "is_sfx": _is_sfx_layer(layer),
                    "baseline_asset": baseline_name,
                    "candidate_asset": candidate_name,
                    "original": str(layer.get("original") or layer.get("text") or ""),
                    "translated": str(
                        layer.get("translated") or layer.get("traduzido") or ""
                    ),
                    "qa_flags": _layer_flags(layer),
                    "sfx": _sfx_review_metadata(layer),
                }
            )
    return crop_pairs


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


def _layer_bbox(layer: dict[str, Any]) -> list[int] | None:
    bbox = layer.get("bbox") or layer.get("source_bbox") or layer.get("text_pixel_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _is_sfx_layer(layer: dict[str, Any]) -> bool:
    route = str(layer.get("route") or layer.get("translation_route") or "")
    kind = str(layer.get("content_class") or layer.get("tipo") or layer.get("type") or "").lower()
    return (
        kind in {"sfx", "sound_effect", "onomatopoeia"}
        or route == "translate_sfx_inpaint_render"
        or isinstance(layer.get("sfx"), dict)
    )


def _layer_flags(layer: dict[str, Any]) -> list[str]:
    flags = [str(flag) for flag in layer.get("qa_flags") or [] if flag]
    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    for key in ("qa_flags", "style_flags", "inpaint_flags", "gate_flags"):
        flags.extend(str(flag) for flag in sfx.get(key) or [] if flag)
    gate = sfx.get("inpaint_gate") if isinstance(sfx.get("inpaint_gate"), dict) else {}
    flags.extend(str(flag) for flag in gate.get("flags") or [] if flag)
    seen: set[str] = set()
    result: list[str] = []
    for flag in flags:
        if flag in seen:
            continue
        seen.add(flag)
        result.append(flag)
    return result


def _sfx_review_metadata(layer: dict[str, Any]) -> dict[str, Any]:
    if not _is_sfx_layer(layer):
        return {}
    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    style_payload = sfx.get("style") if isinstance(sfx.get("style"), dict) else {}
    return {
        "adapted_text": sfx.get("adapted_text") or layer.get("translated") or layer.get("traduzido"),
        "route": layer.get("route") or layer.get("translation_route"),
        "inpaint_allowed": sfx.get("inpaint_allowed"),
        "review_required": sfx.get("review_required"),
        "style_confidence": sfx.get("style_confidence")
        or style_payload.get("confidence")
        or layer.get("style_confidence"),
    }


def _crop_image(image: Any, bbox: list[int], *, padding: int) -> Any:
    x1, y1, x2, y2 = bbox
    left = max(0, x1 - padding)
    top = max(0, y1 - padding)
    right = min(image.width, x2 + padding)
    bottom = min(image.height, y2 + padding)
    return image.crop((left, top, right, bottom))


def _pixel_diff_rate(baseline_image: Path | None, candidate_image: Path | None) -> float | None:
    if baseline_image is None or candidate_image is None:
        return None
    if Image is None or ImageChops is None:
        return None
    with Image.open(baseline_image) as baseline, Image.open(candidate_image) as candidate:
        if baseline.size != candidate.size:
            return 1.0
        baseline_rgb = baseline.convert("RGB")
        candidate_rgb = candidate.convert("RGB")
        diff = ImageChops.difference(baseline_rgb, candidate_rgb)
        if diff.getbbox() is None:
            return 0.0
        changed_pixels = sum(1 for pixel in diff.getdata() if pixel != (0, 0, 0))
        total_pixels = baseline_rgb.size[0] * baseline_rgb.size[1]
        return round(changed_pixels / total_pixels, 6) if total_pixels else 0.0


def _render_page_row(
    page_number: int,
    report: dict[str, Any],
    baseline_asset: str,
    candidate_asset: str,
) -> str:
    metrics = {
        "different_text_rate": report.get("different_text_rate", 0.0),
        "missing_text_rate": report.get("missing_text_rate", 0.0),
        "fallback_rate": report.get("fallback_rate", 0.0),
        "pixel_diff_rate": report.get("pixel_diff_rate", 0.0),
        "different_count": report.get("different_count", 0),
        "missing_count": report.get("missing_count", 0),
    }
    metrics_html = "".join(
        f"<li><code>{html.escape(key)}</code>: {html.escape(str(value))}</li>"
        for key, value in metrics.items()
    )
    crops_html = _render_crop_pairs(report.get("crop_pairs") or [])
    baseline_img = _image_tag(baseline_asset, f"Baseline pagina {page_number}")
    candidate_img = _image_tag(candidate_asset, f"Candidato pagina {page_number}")
    return f"""
    <section class="page" id="page-{page_number:03d}">
      <h2>Pagina {page_number}</h2>
      <ul class="metrics">{metrics_html}</ul>
      <div class="pair">
        <figure>
          <figcaption>Baseline</figcaption>
          {baseline_img}
        </figure>
        <figure>
          <figcaption>Candidato</figcaption>
          {candidate_img}
        </figure>
      </div>
      {crops_html}
    </section>
    """


def _render_crop_pairs(crop_pairs: list[dict[str, Any]]) -> str:
    if not crop_pairs:
        return '<p class="missing">Crops indisponiveis para esta pagina.</p>'
    rows = []
    sfx_rows = []
    for crop in crop_pairs:
        original = html.escape(str(crop.get("original") or ""))
        translated = html.escape(str(crop.get("translated") or ""))
        baseline_asset = html.escape(str(crop.get("baseline_asset") or ""))
        candidate_asset = html.escape(str(crop.get("candidate_asset") or ""))
        flags = ", ".join(str(flag) for flag in crop.get("qa_flags") or [])
        flags_html = f"<small>Flags: <code>{html.escape(flags)}</code></small>" if flags else ""
        rows.append(
            f"""
            <div class="crop">
              <p><code>{original}</code><br>{translated}</p>
              {flags_html}
              <div class="pair">
                <img src="assets/{baseline_asset}" alt="Crop baseline">
                <img src="assets/{candidate_asset}" alt="Crop candidato">
              </div>
            </div>
            """
        )
        if crop.get("is_sfx"):
            sfx_rows.append(_render_sfx_panel(crop))
    sfx_html = ""
    if sfx_rows:
        sfx_html = f"<h3>SFX</h3><div class=\"crops sfx-crops\">{''.join(sfx_rows)}</div>"
    return f"<h3>Crops</h3><div class=\"crops\">{''.join(rows)}</div>{sfx_html}"


def _render_sfx_panel(crop: dict[str, Any]) -> str:
    original = html.escape(str(crop.get("original") or ""))
    translated = html.escape(str(crop.get("translated") or ""))
    baseline_asset = html.escape(str(crop.get("baseline_asset") or ""))
    candidate_asset = html.escape(str(crop.get("candidate_asset") or ""))
    flags = html.escape(", ".join(str(flag) for flag in crop.get("qa_flags") or []))
    sfx = crop.get("sfx") if isinstance(crop.get("sfx"), dict) else {}
    metadata = html.escape(json.dumps(sfx, ensure_ascii=False, indent=2))
    return f"""
    <div class="crop sfx-panel">
      <p><strong>SFX</strong> <code>{original}</code><br>{translated}</p>
      <p><small>QA: <code>{flags or "sem flags"}</code></small></p>
      <div class="pair">
        <figure><figcaption>Original crop</figcaption><img src="assets/{baseline_asset}" alt="SFX original"></figure>
        <figure><figcaption>Final crop</figcaption><img src="assets/{candidate_asset}" alt="SFX final"></figure>
      </div>
      <pre>{metadata}</pre>
    </div>
    """


def _image_tag(asset_name: str, alt: str) -> str:
    if not asset_name:
        return '<p class="missing">Imagem final ausente</p>'
    return f'<img src="assets/{html.escape(asset_name)}" alt="{html.escape(alt)}">'


def _render_html(rows: list[str], baseline_path: Path, candidate_path: Path) -> str:
    body = "\n".join(rows)
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>TraduzAi Visual Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #15171a; color: #f2f4f7; }}
    code {{ color: #b6d7ff; }}
    .meta {{ color: #b8c0cc; }}
    .page {{ border-top: 1px solid #333842; padding: 20px 0; }}
    .pair {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: #20242b; padding: 12px; border-radius: 6px; }}
    figcaption {{ margin-bottom: 8px; color: #dce3ed; }}
    img {{ max-width: 100%; height: auto; display: block; background: #fff; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 8px 18px; padding-left: 18px; }}
    .crops {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 12px; }}
    .crop {{ background: #20242b; padding: 10px; border-radius: 6px; }}
    .crop .pair {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    .missing {{ color: #ffb8b8; }}
  </style>
</head>
<body>
  <h1>TraduzAi Visual Review</h1>
  <p class="meta">Baseline: <code>{html.escape(str(baseline_path))}</code></p>
  <p class="meta">Candidato: <code>{html.escape(str(candidate_path))}</code></p>
  {body}
</body>
</html>
"""


def _write_summary(output_html: Path, result: dict[str, Any]) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    (output_html.parent / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-crops-per-page", type=int, default=6)
    args = parser.parse_args(argv)

    result = export_visual_review_sheet(
        args.baseline,
        args.candidate,
        args.out,
        max_pages=args.max_pages,
        max_crops_per_page=args.max_crops_per_page,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
