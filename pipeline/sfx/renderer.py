"""Renderer adapter for translated manhwa SFX layers."""

from __future__ import annotations

import copy
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from typesetter.font_matcher import render_source_text_mask
from typesetter.glyph_rasterizer import rasterize_v2_glyph_layers


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
        rendered = _render_project_font_latin_sfx(
            image,
            layer,
            text,
            bbox,
            style,
            fill,
            stroke,
            stroke_width,
            glow,
            glow_width,
            rotation,
        )
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


def _render_project_font_latin_sfx(
    image: Image.Image,
    layer: dict[str, Any],
    text: str,
    bbox: list[int],
    style: dict[str, Any],
    fill: str,
    stroke: str,
    stroke_width: int,
    glow: str,
    glow_width: int,
    rotation: float,
) -> np.ndarray | None:
    """Render Latin SFX with a selected project font and the shared V2 compositor."""

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    font_name = str(style.get("font_name") or _font_name_for_sfx(sfx))
    font_path = _project_font_path(font_name) or _project_font_path("ComicNeue-Bold.ttf")
    if font_path is None:
        _append_flag(layer, "sfx_render_missing")
        return None
    normalized = render_source_text_mask(text, font_path, profile={})
    normalized_bbox = cv2.boundingRect(normalized)
    nx, ny, nw, nh = normalized_bbox
    crop = normalized[ny : ny + nh, nx : nx + nw]
    fit_scale = min(
        max(1.0, width * 0.68) / max(1.0, float(nw)),
        max(1.0, height * 0.62) / max(1.0, float(nh)),
    )
    resized = cv2.resize(
        crop,
        (max(1, int(round(nw * fit_scale))), max(1, int(round(nh * fit_scale)))),
        interpolation=cv2.INTER_NEAREST,
    )
    core = np.zeros((image.height, image.width), dtype=np.uint8)
    px = int(round((x1 + x2 - resized.shape[1]) / 2.0))
    py = int(round((y1 + y2 - resized.shape[0]) / 2.0))
    core[py : py + resized.shape[0], px : px + resized.shape[1]] = resized
    safe = _sfx_safe_mask(layer, image.size, bbox)
    raster_style: dict[str, Any] = {
        "fill": fill,
        "rotation_deg": rotation,
        "width_scale": float(style.get("scale_x") or 1.0),
        "scale_y": float(style.get("scale_y") or 1.0),
        "tracking_xh": float(style.get("tracking_xh") or 0.0),
        "slant_tangent": float(style.get("slant_tangent") or 0.0),
    }
    if stroke or stroke_width > 0:
        raster_style["stroke"] = {"color": stroke or fill, "width_px": stroke_width}
    if glow and glow_width > 0:
        raster_style["glow"] = {"color": glow, "width_px": glow_width}
    if isinstance(style.get("shadow"), dict):
        raster_style["shadow"] = style["shadow"]
    if isinstance(style.get("gradient"), (list, tuple)):
        raster_style["gradient"] = style["gradient"]
    result = rasterize_v2_glyph_layers(
        core,
        safe,
        raster_style,
        source_x_height_px=max(8.0, resized.shape[0] * 0.72),
    )
    if result.status == "review_required" or not np.any(result.rgba[:, :, 3]):
        _append_flag(layer, "sfx_render_outside_source_region")
        return None
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    alpha = result.rgba[:, :, 3:4].astype(np.float32) / 255.0
    base = np.clip(
        result.rgba[:, :, :3].astype(np.float32) * alpha
        + base.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)
    alpha_bbox = cv2.boundingRect(np.where(result.rgba[:, :, 3] > 0, 255, 0).astype(np.uint8))
    rx, ry, rw, rh = alpha_bbox
    render_bbox = [rx, ry, rx + rw, ry + rh]
    layer["render_bbox"] = render_bbox
    layer["fit_status"] = "ok"
    layer["render_policy"] = "sfx_style"
    layer["sfx_font_backend"] = "project_font_textpath"
    layer["render_font_name"] = font_path.name
    layer["style_v2_raster_contract"] = {
        "status": result.status,
        "applied_attributes": copy.deepcopy(result.applied_attributes),
        "abstained_attributes": copy.deepcopy(result.abstained_attributes),
        "glyph_core_envelope": list(result.glyph_core_envelope or ()),
        "effect_envelope": list(result.effect_envelope or ()),
        "metrics": copy.deepcopy(result.metrics),
    }
    layer["translated"] = text
    layer["traduzido"] = text
    for attribute in result.abstained_attributes:
        _append_flag(layer, f"sfx_style_{attribute}_abstained")
    return base


def _project_font_path(font_name: str) -> Path | None:
    for root in FONT_DIRS:
        candidate = root / str(font_name)
        if candidate.is_file():
            return candidate
    return None


def _sfx_safe_mask(
    layer: dict[str, Any],
    image_size: tuple[int, int],
    bbox: list[int],
) -> np.ndarray:
    width, height = image_size
    safe = np.zeros((height, width), dtype=np.uint8)
    polygon = layer.get("render_safe_polygon_page")
    if isinstance(polygon, (list, tuple)) and len(polygon) >= 3:
        try:
            points = np.asarray(
                [[int(round(float(point[0]))), int(round(float(point[1])))] for point in polygon],
                dtype=np.int32,
            )
            cv2.fillPoly(safe, [points], 255)
            return safe
        except (TypeError, ValueError, IndexError):
            pass
    x1, y1, x2, y2 = bbox
    safe[y1:y2, x1:x2] = 255
    return safe


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
