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

from sfx.inpaint_gate import evaluate_sfx_inpaint_gate
from sfx.mask import build_sfx_glyph_mask
from sfx.ocr_probe import probe_sfx_candidate_ocr
from sfx.style import extract_manhwa_sfx_style
from typesetter.renderer import render_band_image, save_typeset_page_image
from vision_stack.runtime import _get_detector, run_inpaint_pages, vision_blocks_to_mask
from vision_stack.sfx_detector import (
    filter_sfx_candidates_after_ocr,
    merge_sfx_candidates,
    text_blocks_to_sfx_candidates,
)


SAMPLE_SFX_TEXTS = ("TUM", "TAC", "VRUM", "ZAS", "CLANG", "WHOOSH")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SFX detect/OCR/inpaint/typeset visual probe.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--anime-conf", type=float, default=0.0107)
    parser.add_argument("--comic-conf", type=float, default=0.05)
    parser.add_argument("--profile", default="quality")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_images(input_path)
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found: {input_path}")

    pages: list[dict[str, Any]] = []
    forced_ocr_pages: list[dict[str, Any]] = []
    forced_image_paths: list[Path] = []

    for image_index, image_path in enumerate(image_paths, start=1):
        image_rgb = _load_rgb(image_path)
        raw_candidates = _detect_text_sfx_candidates(
            image_rgb,
            anime_conf=float(args.anime_conf),
            comic_conf=float(args.comic_conf),
            profile=str(args.profile),
        )
        probed = []
        for candidate in raw_candidates:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                probed.append(probe_sfx_candidate_ocr(candidate, image_rgb))
        candidates = filter_sfx_candidates_after_ocr(probed, image_rgb)
        forced_layers = _build_forced_sfx_layers(image_rgb, candidates)

        page_dir = out_dir / f"{image_index:03d}_{_safe_stem(image_path)}"
        page_dir.mkdir(parents=True, exist_ok=True)
        _save_rgb(page_dir / "00_original.png", image_rgb)
        _save_rgb(page_dir / "01_detect_ocr_overlay.png", _draw_overlay(image_rgb, candidates, forced_layers))

        forced_page = {"texts": forced_layers, "_vision_blocks": _vision_blocks_for_layers(forced_layers)}
        mask = vision_blocks_to_mask(
            image_rgb.shape,
            forced_page["_vision_blocks"],
            image_rgb=image_rgb,
            expand_mask=True,
            ocr_texts=forced_layers,
        )
        _save_mask(page_dir / "02_sfx_mask.png", mask)

        forced_ocr_pages.append(forced_page)
        forced_image_paths.append(image_path)
        pages.append(
            {
                "image": str(image_path),
                "page_dir": str(page_dir),
                "candidate_count": len(candidates),
                "forced_layer_count": len(forced_layers),
                "mask_pixels": int(np.count_nonzero(mask)),
                "candidates": [_candidate_summary(item) for item in candidates],
                "forced_layers": [_layer_summary(item) for item in forced_layers],
            }
        )

    inpaint_dir = out_dir / "inpainted"
    inpainted_paths = run_inpaint_pages(forced_image_paths, forced_ocr_pages, str(inpaint_dir), profile=str(args.profile))

    contact_rows = []
    for page, image_path, forced_page, inpainted_path in zip(pages, forced_image_paths, forced_ocr_pages, inpainted_paths):
        page_dir = Path(page["page_dir"])
        original = _load_rgb(image_path)
        inpainted = _load_rgb(inpainted_path)
        _save_rgb(page_dir / "03_inpaint.png", inpainted)
        rendered = render_band_image(inpainted, forced_page)
        save_typeset_page_image(Image.fromarray(rendered), page_dir / "04_typeset.png", quality=95)
        page["inpaint_path"] = str(page_dir / "03_inpaint.png")
        page["typeset_path"] = str(page_dir / "04_typeset.png")
        page["rendered_layers"] = [_layer_summary(item) for item in forced_page.get("texts", [])]
        contact_rows.append(
            [
                ("original", original),
                ("detect+ocr", _load_rgb(page_dir / "01_detect_ocr_overlay.png")),
                ("mask", _mask_to_rgb(_load_mask(page_dir / "02_sfx_mask.png"))),
                ("inpaint", inpainted),
                ("typeset", rendered),
            ]
        )

    contact = _make_contact_sheet(contact_rows)
    contact_path = out_dir / "sfx_inpaint_type_contact_sheet.png"
    save_typeset_page_image(Image.fromarray(contact), contact_path, quality=95)

    summary = {
        "input": str(input_path),
        "output": str(out_dir),
        "anime_conf": float(args.anime_conf),
        "comic_conf": float(args.comic_conf),
        "page_count": len(pages),
        "total_candidates": sum(int(page["candidate_count"]) for page in pages),
        "total_forced_layers": sum(int(page["forced_layer_count"]) for page in pages),
        "total_mask_pixels": sum(int(page["mask_pixels"]) for page in pages),
        "contact_sheet": str(contact_path),
        "pages": pages,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file()
        and item.suffix.lower() in suffixes
        and not item.stem.lower().startswith(("font_probe", "sfx_inpaint_type", "sfx_post_ocr", "sfx_text_ensemble"))
    )


def _load_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8))


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask.astype(np.uint8)


def _detect_text_sfx_candidates(
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


def _build_forced_sfx_layers(image_rgb: np.ndarray, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        bbox = _bbox(candidate)
        if bbox is None:
            continue
        layer = dict(candidate)
        layer["id"] = f"forced_sfx_{index:03d}"
        layer["text_id"] = layer["id"]
        layer["content_class"] = "sfx"
        layer["tipo"] = "sfx"
        layer["route_action"] = "translate_sfx_inpaint_render"
        layer["render_policy"] = "sfx_style"
        layer["skip_processing"] = False
        layer["preserve_original"] = False
        layer["text"] = layer.get("recognized_text") or layer.get("text") or "쾅"
        layer["original"] = layer["text"]
        layer["translated"] = SAMPLE_SFX_TEXTS[(index - 1) % len(SAMPLE_SFX_TEXTS)]
        layer["traduzido"] = layer["translated"]
        layer["bbox"] = bbox
        layer["text_pixel_bbox"] = bbox
        layer["source_bbox"] = bbox
        layer["balloon_bbox"] = bbox
        layer["sfx"] = dict(layer.get("sfx") or {})
        layer["sfx"]["source_text"] = layer["text"]
        layer["sfx"]["adapted_text"] = layer["translated"]

        mask_result = build_sfx_glyph_mask(image_rgb, layer)
        layer["mask_evidence"] = dict(mask_result.evidence)
        if mask_result.mask is not None and np.any(mask_result.mask):
            layer["mask"] = mask_result.mask
            x1, y1, x2, y2 = bbox
            crop = image_rgb[y1:y2, x1:x2]
            crop_mask = mask_result.mask[y1:y2, x1:x2]
            style = extract_manhwa_sfx_style(crop, crop_mask, layer=layer).to_dict()
        else:
            style = _fallback_style_for_crop(image_rgb, bbox)

        gate = evaluate_sfx_inpaint_gate(layer)
        layer["sfx_inpaint_gate"] = gate
        layer["sfx"]["inpaint_allowed"] = bool(gate.get("allow_inpaint")) and mask_result.mask is not None
        layer["sfx"]["style"] = style
        if layer["sfx"]["inpaint_allowed"]:
            layers.append(layer)
    return layers


def _vision_blocks_for_layers(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = []
    for layer in layers:
        if (layer.get("sfx") or {}).get("inpaint_allowed") is not True:
            continue
        block = dict(layer)
        block["confidence"] = float(layer.get("confidence") or 1.0)
        blocks.append(block)
    return blocks


def _draw_overlay(image_rgb: np.ndarray, candidates: list[dict[str, Any]], forced_layers: list[dict[str, Any]]) -> np.ndarray:
    rendered = Image.fromarray(image_rgb.astype(np.uint8), "RGB").convert("RGBA")
    draw = ImageDraw.Draw(rendered)
    forced_ids = {str(item.get("id") or "") for item in forced_layers}
    forced_bboxes = {tuple(_bbox(item) or []) for item in forced_layers}
    font = ImageFont.load_default()
    for index, candidate in enumerate(candidates, start=1):
        bbox = _bbox(candidate)
        if bbox is None:
            continue
        is_forced = tuple(bbox) in forced_bboxes or str(candidate.get("id") or "") in forced_ids
        color = (0, 230, 80, 255) if is_forced else (255, 188, 0, 255)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
        label = f"{index}:{candidate.get('confidence', 0):.3f} {sfx_ocr.get('status') or ''}"
        draw.text((x1, max(0, y1 - 12)), label, fill=color, font=font)
    return np.asarray(rendered.convert("RGB"))


def _make_contact_sheet(rows: list[list[tuple[str, np.ndarray]]]) -> np.ndarray:
    if not rows:
        return np.full((64, 320, 3), 255, dtype=np.uint8)
    cell_w = 220
    label_h = 18
    row_images = []
    font = ImageFont.load_default()
    for row in rows:
        cells = []
        max_h = 0
        for _label, image in row:
            image = np.asarray(image).astype(np.uint8)
            if image.ndim == 2:
                image = _mask_to_rgb(image)
            if image.ndim != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
                image = np.full((64, cell_w, 3), 255, dtype=np.uint8)
            h, w = image.shape[:2]
            scale = min(1.0, cell_w / float(max(1, w)))
            target_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            resized = np.asarray(Image.fromarray(image).resize(target_size, Image.Resampling.LANCZOS))
            cells.append((_label, resized))
            max_h = max(max_h, resized.shape[0])
        canvas = Image.new("RGB", (cell_w * len(cells), max_h + label_h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        for idx, (label, image) in enumerate(cells):
            x = idx * cell_w
            draw.text((x + 4, 2), label, fill=(0, 0, 0), font=font)
            canvas.paste(Image.fromarray(image), (x, label_h))
        row_images.append(np.asarray(canvas))
    width = max(item.shape[1] for item in row_images)
    height = sum(item.shape[0] for item in row_images)
    sheet = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 0
    for item in row_images:
        sheet[y : y + item.shape[0], : item.shape[1]] = item
        y += item.shape[0]
    return sheet


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat(mask[:, :, None], 3, axis=2)


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


def _fallback_style_for_crop(image_rgb: np.ndarray, bbox: list[int]) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return {
            "fill_color": "#FFFFFF",
            "stroke_color": "#1B2430",
            "stroke_width_px": 2,
            "glow_color": "",
            "glow_width_px": 0,
            "rotation_deg": 0.0,
            "confidence": 0.0,
            "qa_flags": ["sfx_style_empty_crop"],
        }
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    fill = "#FFFFFF" if float(np.mean(gray)) < 150.0 else "#2A1A1A"
    stroke = "#1B2430" if fill == "#FFFFFF" else "#FFFFFF"
    return {
        "fill_color": fill,
        "stroke_color": stroke,
        "stroke_width_px": 2,
        "glow_color": "",
        "glow_width_px": 0,
        "rotation_deg": 0.0,
        "confidence": 0.2,
        "qa_flags": ["sfx_style_fallback"],
    }


def _candidate_summary(item: dict[str, Any]) -> dict[str, Any]:
    sfx = item.get("sfx") if isinstance(item.get("sfx"), dict) else {}
    sfx_ocr = item.get("sfx_ocr") if isinstance(item.get("sfx_ocr"), dict) else {}
    return {
        "bbox": _bbox(item),
        "confidence": item.get("confidence"),
        "visual_source": sfx.get("visual_source"),
        "ocr_status": sfx_ocr.get("status"),
        "recognized_text": item.get("recognized_text") or item.get("text") or "",
        "route_action": item.get("route_action"),
    }


def _layer_summary(item: dict[str, Any]) -> dict[str, Any]:
    sfx = item.get("sfx") if isinstance(item.get("sfx"), dict) else {}
    gate = item.get("sfx_inpaint_gate") if isinstance(item.get("sfx_inpaint_gate"), dict) else {}
    return {
        "id": item.get("id"),
        "bbox": _bbox(item),
        "translated": item.get("translated"),
        "gate_allow_inpaint": gate.get("allow_inpaint"),
        "gate_reason": gate.get("reason"),
        "mask_pixels": int(np.count_nonzero(item.get("mask"))) if isinstance(item.get("mask"), np.ndarray) else 0,
        "fit_status": item.get("fit_status"),
        "render_bbox": item.get("render_bbox"),
        "style": sfx.get("style"),
        "qa_flags": item.get("qa_flags") or [],
    }


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)[:80]


if __name__ == "__main__":
    raise SystemExit(main())
