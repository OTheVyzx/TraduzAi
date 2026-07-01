"""Renderer adapter for translated manhwa SFX layers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


FONT_PRESETS = {
    "impact": "KOMIKAX_.ttf",
    "explosion_impact": "KOMIKAX_.ttf",
    "motion": "Newrotic.ttf",
    "mechanical_click": "CCDaveGibbonsLower W00 Regular.ttf",
    "mechanical": "CCDaveGibbonsLower W00 Regular.ttf",
}

FONT_DIRS = [
    Path(__file__).resolve().parents[2] / "fonts",
    Path.home() / ".traduzai" / "fonts",
    Path.home() / ".mangatl" / "fonts",
    Path("C:/Windows/Fonts"),
]
SAFE_DRAW_FONT_NAMES = {
    "ComicNeue-Bold.ttf",
    "ComicNeue-Regular.ttf",
    "Bangers-Regular.ttf",
    "LuckiestGuy-Regular.ttf",
    "PermanentMarker-Regular.ttf",
    "impact.ttf",
    "arialbd.ttf",
    "arial.ttf",
    "comicbd.ttf",
    "comic.ttf",
}
LATIN_FALLBACK_FONT_NAMES = [
    "impact.ttf",
    "arialbd.ttf",
    "arial.ttf",
    "comicbd.ttf",
    "comic.ttf",
]


def render_sfx_layer(page_rgb: np.ndarray | Image.Image, layer: dict[str, Any]) -> np.ndarray:
    """Render one translated SFX layer onto a page image."""

    image = page_rgb.convert("RGB") if isinstance(page_rgb, Image.Image) else Image.fromarray(page_rgb.astype(np.uint8), "RGB")
    if not _should_render_sfx(layer):
        _append_flag(layer, "sfx_render_missing")
        return np.asarray(image)

    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    text = str(sfx.get("adapted_text") or layer.get("translated") or layer.get("traduzido") or "").strip()
    bbox = _bbox(layer, image.size)
    if not text or bbox is None:
        _append_flag(layer, "sfx_render_missing")
        return np.asarray(image)

    style = sfx.get("style") if isinstance(sfx.get("style"), dict) else {}
    fill = str(style.get("fill_color") or "#000000")
    stroke = str(style.get("stroke_color") or "")
    stroke_width = max(0, int(style.get("stroke_width_px") or 0))
    glow = str(style.get("glow_color") or "")
    glow_width = max(0, int(style.get("glow_width_px") or 0))
    rotation = float(style.get("rotation_deg") or 0.0)
    if _is_latin_sfx_text(text):
        rendered = _render_cv2_latin_sfx(image, layer, text, bbox, fill, stroke, stroke_width, glow, glow_width, rotation)
        if rendered is not None:
            return rendered
    font = _load_font(_font_name_for_sfx(sfx), _fit_font_size(text, bbox, stroke_width), text=text)

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad = max(12, stroke_width * 4 + glow_width * 2)
    overlay = Image.new("RGBA", (width + pad * 2, height + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw = max(1, text_bbox[2] - text_bbox[0])
    th = max(1, text_bbox[3] - text_bbox[1])
    tx = pad + max(0, (width - tw) // 2) - text_bbox[0]
    ty = pad + max(0, (height - th) // 2) - text_bbox[1]

    if glow and glow_width > 0:
        glow_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)
        glow_draw.text((tx, ty), text, font=font, fill=glow, stroke_width=stroke_width + glow_width, stroke_fill=glow)
        overlay.alpha_composite(glow_layer.filter(ImageFilter.GaussianBlur(radius=max(1, glow_width // 2))))

    draw.text(
        (tx, ty),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke or fill,
    )

    if rotation:
        overlay = overlay.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

    px = int(round((x1 + x2 - overlay.width) / 2))
    py = int(round((y1 + y2 - overlay.height) / 2))
    base = image.convert("RGBA")
    base.alpha_composite(overlay, (px, py))
    alpha_bbox = overlay.getchannel("A").getbbox()
    if alpha_bbox:
        render_bbox = [
            max(0, px + alpha_bbox[0]),
            max(0, py + alpha_bbox[1]),
            min(image.width, px + alpha_bbox[2]),
            min(image.height, py + alpha_bbox[3]),
        ]
        layer["render_bbox"] = render_bbox
        layer["fit_status"] = "ok"
        if not _bbox_contains(_expand_bbox(bbox, image.width, image.height, 18), render_bbox):
            _append_flag(layer, "sfx_render_outside_source_region")
    else:
        _append_flag(layer, "sfx_render_missing")
    layer["render_policy"] = "sfx_style"
    layer["translated"] = text
    layer["traduzido"] = text
    return np.asarray(base.convert("RGB"))


def _is_latin_sfx_text(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9!?.,:'\" -]{1,24}", str(text or "").strip()))


def _render_cv2_latin_sfx(
    image: Image.Image,
    layer: dict[str, Any],
    text: str,
    bbox: list[int],
    fill: str,
    stroke: str,
    stroke_width: int,
    glow: str,
    glow_width: int,
    rotation: float,
) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad = max(16, stroke_width * 5 + glow_width * 2)
    canvas_w = width + pad * 2
    canvas_h = height + pad * 2
    overlay = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    font_face = cv2.FONT_HERSHEY_TRIPLEX
    raw_size, raw_baseline = cv2.getTextSize(text, font_face, 1.0, max(1, stroke_width + 1))
    raw_w = max(1, raw_size[0])
    raw_h = max(1, raw_size[1] + raw_baseline)
    scale = max(0.25, min(width / float(raw_w) * 0.88, height / float(raw_h) * 0.88))
    fill_rgba = _hex_to_rgba(fill, (0, 0, 0, 255))
    stroke_rgba = _hex_to_rgba(stroke, fill_rgba)
    glow_rgba = _hex_to_rgba(glow, stroke_rgba)
    thickness = max(1, int(round(max(1.0, scale * 2.2))))
    outline_thickness = max(thickness + 1, int(stroke_width) * 2 + thickness)
    size, baseline = cv2.getTextSize(text, font_face, scale, thickness)
    tx = int(round(pad + (width - size[0]) / 2.0))
    ty = int(round(pad + (height + size[1]) / 2.0 - baseline / 2.0))
    if glow and glow_width > 0:
        glow_layer = np.zeros_like(overlay)
        cv2.putText(glow_layer, text, (tx, ty), font_face, scale, glow_rgba, outline_thickness + int(glow_width), cv2.LINE_AA)
        alpha = glow_layer[:, :, 3]
        if np.any(alpha):
            blurred = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=max(1.0, glow_width / 2.0))
            overlay = _alpha_composite_np(overlay, blurred)
    if stroke_width > 0 or stroke:
        cv2.putText(overlay, text, (tx, ty), font_face, scale, stroke_rgba, outline_thickness, cv2.LINE_AA)
    cv2.putText(overlay, text, (tx, ty), font_face, scale, fill_rgba, thickness, cv2.LINE_AA)
    overlay_img = Image.fromarray(overlay, "RGBA")
    if rotation:
        overlay_img = overlay_img.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    px = int(round((x1 + x2 - overlay_img.width) / 2))
    py = int(round((y1 + y2 - overlay_img.height) / 2))
    base = image.convert("RGBA")
    base.alpha_composite(overlay_img, (px, py))
    alpha_bbox = overlay_img.getchannel("A").getbbox()
    if not alpha_bbox:
        _append_flag(layer, "sfx_render_missing")
        return None
    render_bbox = [
        max(0, px + alpha_bbox[0]),
        max(0, py + alpha_bbox[1]),
        min(image.width, px + alpha_bbox[2]),
        min(image.height, py + alpha_bbox[3]),
    ]
    layer["render_bbox"] = render_bbox
    layer["fit_status"] = "ok"
    layer["render_policy"] = "sfx_style"
    layer["translated"] = text
    layer["traduzido"] = text
    if not _bbox_contains(_expand_bbox(bbox, image.width, image.height, 18), render_bbox):
        _append_flag(layer, "sfx_render_outside_source_region")
    return np.asarray(base.convert("RGB"))


def _hex_to_rgba(value: str, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    raw = str(value or "").strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return fallback
    try:
        return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16), 255)
    except ValueError:
        return fallback


def _alpha_composite_np(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    bottom_f = bottom.astype(np.float32) / 255.0
    top_f = top.astype(np.float32) / 255.0
    top_a = top_f[:, :, 3:4]
    bottom_a = bottom_f[:, :, 3:4]
    out_a = top_a + bottom_a * (1.0 - top_a)
    out_rgb = np.where(out_a > 0, (top_f[:, :, :3] * top_a + bottom_f[:, :, :3] * bottom_a * (1.0 - top_a)) / np.maximum(out_a, 1e-6), 0.0)
    out = np.concatenate([out_rgb, out_a], axis=2)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def _should_render_sfx(layer: dict[str, Any]) -> bool:
    if str(layer.get("content_class") or "").strip().lower() != "sfx":
        return False
    if str(layer.get("route_action") or "").strip().lower() == "review_required":
        return False
    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    return sfx.get("inpaint_allowed") is not False


def _font_name_for_sfx(sfx: dict[str, Any]) -> str:
    kind = str(sfx.get("kind") or "").strip().lower()
    return FONT_PRESETS.get(kind, FONT_PRESETS["impact"])


def _load_font(font_name: str, size: int, *, text: str) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate_names = [font_name]
    candidate_names.extend(name for name in LATIN_FALLBACK_FONT_NAMES if name not in candidate_names)
    candidate_names.append("ComicNeue-Bold.ttf")
    for candidate_name in candidate_names:
        for root in FONT_DIRS:
            candidate = root / candidate_name
            if candidate.exists():
                try:
                    font = ImageFont.truetype(str(candidate), size=max(8, int(size)))
                except Exception:
                    continue
                if _font_draws_text(font, text):
                    return font
    return ImageFont.load_default()


def _font_draws_text(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> bool:
    probe = Image.new("RGBA", (240, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=0)
    if bbox is None:
        return False
    px = max(0, 20 - min(0, bbox[0]))
    py = max(0, 20 - min(0, bbox[1]))
    draw.text((px, py), text, font=font, fill=(255, 255, 255, 255))
    return probe.getchannel("A").getbbox() is not None


def _fit_font_size(text: str, bbox: list[int], stroke_width: int) -> int:
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    per_char = max(1, len(text))
    return max(12, min(96, int(min(height * 0.78, width / per_char * 1.65)) - stroke_width))


def _bbox(layer: dict[str, Any], image_size: tuple[int, int]) -> list[int] | None:
    width, height = image_size
    value = layer.get("source_bbox") or layer.get("bbox") or layer.get("text_pixel_bbox")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _expand_bbox(bbox: list[int], width: int, height: int, pad: int) -> list[int]:
    return [
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(width, bbox[2] + pad),
        min(height, bbox[3] + pad),
    ]


def _bbox_contains(container: list[int], child: list[int]) -> bool:
    return child[0] >= container[0] and child[1] >= container[1] and child[2] <= container[2] and child[3] <= container[3]


def _append_flag(layer: dict[str, Any], flag: str) -> None:
    flags = layer.setdefault("qa_flags", [])
    if flag not in flags:
        flags.append(flag)
