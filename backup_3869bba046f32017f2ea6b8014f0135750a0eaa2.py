"""
Typesetting module - renders translated text onto inpainted manga pages.
Now uses inferred balloon/layout geometry instead of relying only on the raw
OCR bounding box.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from itertools import product
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from matplotlib.ft2font import FT2Font as _FT2Font
from PIL import Image, ImageDraw, ImageFilter, ImageFont


FONT_DIRS = [
    Path(__file__).parent.parent.parent / "fonts",
    Path.home() / ".traduzai" / "fonts",
    Path.home() / ".mangatl" / "fonts",  # legado
    Path("/usr/share/fonts"),
]

DEFAULT_FONTS = {
    "fala":      "ComicNeue-Bold.ttf",
    "narracao":  "ComicNeue-Bold.ttf",
    "sfx":       "ComicNeue-Bold.ttf",
    "pensamento": "ComicNeue-Bold.ttf",
}

SAFE_PATH_FORCE_KEYWORDS = (
    "newrotic",
    "wildwords",
    "blambot",
)

_PUNCT_REPLACEMENTS = {"…": "...", "⋯": "...", "‥": "..", "\u201c": "\"", "\u201d": "\"", "\u2018": "'", "\u2019": "'", "\u2014": "-", "\u2013": "-", "\u2015": "-", "□": ".", "■": ".", "▪": ".", "•": ".", "·": ".", "・": "."}
_MIN_FONT_SIZE = 8
_font_cache: dict[tuple[str, int], object] = {}


class SafeTextPathFont:
    def __init__(self, font_path: str | Path, size: int) -> None:
        self.font_path = Path(font_path)
        self.size = int(size)
        self._bbox_cache: dict[str, tuple[int, int, int, int]] = {}
        self._mask_cache: dict[tuple[str, int], np.ndarray] = {}

    def getbbox(self, text: str) -> tuple[int, int, int, int]:
        if text in self._bbox_cache:
            return self._bbox_cache[text]
        # Mede pela bitmap real da fonte; necessário para fontes estilizadas
        # como Newrotic, cuja largura real é muito maior que a estimativa simples.
        try:
            bitmap = _render_text_with_fallback(self, text)
            if bitmap.size > 0:
                bbox = (0, 0, int(bitmap.shape[1]), int(bitmap.shape[0]))
            else:
                bbox = (0, 0, max(1, int(self.size * 0.5)), max(1, int(self.size * 1.15)))
        except Exception:
            bbox = (0, 0, max(1, int(len(text) * self.size * 0.55)), max(1, int(self.size * 1.15)))
        self._bbox_cache[text] = bbox
        return bbox


_FALLBACK_FONTS = [
    "ComicNeue-Regular.ttf",
    "CCDaveGibbonsLower W00 Regular.ttf",
]


def _font_has_glyph(font_path: str, char: str) -> bool:
    """Verifica se a fonte tem o glyph para um caractere."""
    try:
        ft2 = _FT2Font(font_path)
        glyph_id = ft2.get_char_index(ord(char))
        return glyph_id != 0
    except Exception:
        return False


def _find_fallback_font_path(char: str, original_path: str) -> str | None:
    """Encontra uma fonte fallback que tenha o glyph para o caractere."""
    for font_dir in FONT_DIRS:
        if not font_dir.exists():
            continue
        for fallback_name in _FALLBACK_FONTS:
            fallback_path = font_dir / fallback_name
            if fallback_path.exists() and str(fallback_path) != original_path:
                if _font_has_glyph(str(fallback_path), char):
                    return str(fallback_path)
    return None


def _render_text_with_fallback(font: SafeTextPathFont, text: str) -> np.ndarray:
    """Renderiza texto com fallback automático para caracteres sem glyph na fonte principal.

    Para cada caractere que não existe na fonte principal, usa uma fonte fallback.
    Renderiza char a char apenas quando necessário (quando há chars faltando).
    """
    font_path = str(font.font_path)

    # Verificar se todos os chars existem na fonte principal
    missing_chars = []
    for ch in text:
        if ch.isspace() or not ch.isprintable():
            continue
        if not _font_has_glyph(font_path, ch):
            missing_chars.append(ch)

    # Se não falta nenhum, renderiza tudo de uma vez (mais rápido)
    if not missing_chars:
        ft2 = _FT2Font(font_path)
        ft2.set_size(font.size, 72)
        ft2.set_text(text, 0.0)
        ft2.draw_glyphs_to_bitmap()
        return ft2.get_image()

    # Renderiza caractere a caractere, usando fallback quando necessário
    fallback_cache: dict[str, str | None] = {}
    char_bitmaps: list[tuple[np.ndarray, int]] = []  # (bitmap, y_offset)

    for ch in text:
        if ch == " ":
            # Espaço: renderiza com fonte principal para obter largura correta
            ft2 = _FT2Font(font_path)
            ft2.set_size(font.size, 72)
            ft2.set_text(" I", 0.0)
            ft2.draw_glyphs_to_bitmap()
            space_bitmap = ft2.get_image()
            ft2_single = _FT2Font(font_path)
            ft2_single.set_size(font.size, 72)
            ft2_single.set_text("I", 0.0)
            ft2_single.draw_glyphs_to_bitmap()
            single_bitmap = ft2_single.get_image()
            space_w = max(1, space_bitmap.shape[1] - single_bitmap.shape[1])
            space_img = np.zeros((max(1, int(font.size)), space_w), dtype=np.uint8)
            char_bitmaps.append((space_img, 0))
            continue

        # Determinar qual fonte usar
        use_path = font_path
        if ch in missing_chars:
            if ch not in fallback_cache:
                fallback_cache[ch] = _find_fallback_font_path(ch, font_path)
            fb = fallback_cache[ch]
            if fb:
                use_path = fb

        ft2 = _FT2Font(use_path)
        ft2.set_size(font.size, 72)
        ft2.set_text(ch, 0.0)
        ft2.draw_glyphs_to_bitmap()
        bmp = ft2.get_image()
        if bmp.size == 0:
            continue
        char_bitmaps.append((bmp, 0))

    if not char_bitmaps:
        return np.zeros((1, 1), dtype=np.uint8)

    # Combinar todos os bitmaps lado a lado
    max_h = max(bmp.shape[0] for bmp, _ in char_bitmaps)
    total_w = sum(bmp.shape[1] for bmp, _ in char_bitmaps)
    combined = np.zeros((max_h, total_w), dtype=np.uint8)
    x_cursor = 0
    for bmp, _ in char_bitmaps:
        h, w = bmp.shape
        y_off = max_h - h  # Alinhar por baixo (baseline)
        combined[y_off:y_off + h, x_cursor:x_cursor + w] = np.maximum(
            combined[y_off:y_off + h, x_cursor:x_cursor + w], bmp
        )
        x_cursor += w

    return combined


def _build_textpath_mask(font: SafeTextPathFont, text: str, padding: int = 0) -> np.ndarray:
    if not text or not text.strip():
        return np.zeros((1, 1), dtype=np.uint8)

    cache_key = (text, int(padding))
    cached = font._mask_cache.get(cache_key)
    if cached is not None:
        return cached.copy()

    try:
        bitmap = _render_text_with_fallback(font, text)
    except Exception:
        mask = np.zeros((1, 1), dtype=np.uint8)
        font._mask_cache[cache_key] = mask
        return mask.copy()

    if bitmap.size == 0:
        mask = np.zeros((1, 1), dtype=np.uint8)
        font._mask_cache[cache_key] = mask
        return mask.copy()

    pad = max(0, int(padding))
    if pad > 0:
        padded = np.zeros((bitmap.shape[0] + pad * 2, bitmap.shape[1] + pad * 2), dtype=np.uint8)
        padded[pad:pad + bitmap.shape[0], pad:pad + bitmap.shape[1]] = bitmap
        mask = padded
    else:
        mask = bitmap.copy()

    font._mask_cache[cache_key] = mask
    return mask.copy()


def _blend_mask_into_image(
    image_np: np.ndarray,
    mask: np.ndarray,
    origin_x: int,
    origin_y: int,
    color: str,
) -> None:
    if mask.size == 0:
        return

    x1 = max(0, int(origin_x))
    y1 = max(0, int(origin_y))
    x2 = min(image_np.shape[1], int(origin_x) + mask.shape[1])
    y2 = min(image_np.shape[0], int(origin_y) + mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return

    mx1 = max(0, x1 - int(origin_x))
    my1 = max(0, y1 - int(origin_y))
    mx2 = mx1 + (x2 - x1)
    my2 = my1 + (y2 - y1)
    local_mask = mask[my1:my2, mx1:mx2].astype(np.float32) / 255.0
    if local_mask.size == 0 or float(np.max(local_mask)) <= 0.0:
        return

    target = image_np[y1:y2, x1:x2].astype(np.float32)
    fill = np.array(_parse_hex_color(color), dtype=np.float32)
    alpha = local_mask[..., None]
    blended = target * (1.0 - alpha) + fill * alpha
    image_np[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)


def _render_safe_text_layer(
    image_np: np.ndarray,
    lines: list[str],
    font: SafeTextPathFont,
    positions: list[tuple[int, int]],
    *,
    fill_color: str,
    outline_color: str = "",
    outline_px: int = 0,
) -> None:
    for line, (lx, ly) in zip(lines, positions):
        mask = _build_textpath_mask(font, line, padding=max(0, outline_px))
        if outline_color and outline_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (outline_px * 2 + 1, outline_px * 2 + 1),
            )
            outline_mask = cv2.dilate(mask, kernel, iterations=1)
            _blend_mask_into_image(image_np, outline_mask, lx - outline_px, ly - outline_px, outline_color)
        _blend_mask_into_image(image_np, mask, lx - max(0, outline_px), ly - max(0, outline_px), fill_color)


def _blend_rgb_patch_with_mask(
    image_np: np.ndarray,
    rgb_patch: np.ndarray,
    mask: np.ndarray,
    origin_x: int,
    origin_y: int,
) -> None:
    if mask.size == 0 or rgb_patch.size == 0:
        return

    x1 = max(0, int(origin_x))
    y1 = max(0, int(origin_y))
    x2 = min(image_np.shape[1], int(origin_x) + mask.shape[1])
    y2 = min(image_np.shape[0], int(origin_y) + mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return

    mx1 = max(0, x1 - int(origin_x))
    my1 = max(0, y1 - int(origin_y))
    mx2 = mx1 + (x2 - x1)
    my2 = my1 + (y2 - y1)
    local_mask = mask[my1:my2, mx1:mx2].astype(np.float32) / 255.0
    if local_mask.size == 0 or float(np.max(local_mask)) <= 0.0:
        return

    target = image_np[y1:y2, x1:x2].astype(np.float32)
    patch = rgb_patch[my1:my2, mx1:mx2].astype(np.float32)
    alpha = local_mask[..., None]
    blended = target * (1.0 - alpha) + patch * alpha
    image_np[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)


def _measure_safe_text_block_bbox(
    font: SafeTextPathFont,
    lines: list[str],
    positions: list[tuple[int, int]],
) -> list[int] | None:
    block_x1 = block_y1 = None
    block_x2 = block_y2 = None

    for line, (lx, ly) in zip(lines, positions):
        mask = _build_textpath_mask(font, line, padding=0)
        if mask.size <= 1:
            continue
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue

        line_x1 = int(lx + int(xs.min()))
        line_y1 = int(ly + int(ys.min()))
        line_x2 = int(lx + int(xs.max()) + 1)
        line_y2 = int(ly + int(ys.max()) + 1)

        if block_x1 is None:
            block_x1, block_y1, block_x2, block_y2 = line_x1, line_y1, line_x2, line_y2
            continue

        block_x1 = min(block_x1, line_x1)
        block_y1 = min(block_y1, line_y1)
        block_x2 = max(block_x2, line_x2)
        block_y2 = max(block_y2, line_y2)

    if block_x1 is None or block_y1 is None or block_x2 is None or block_y2 is None:
        return None
    return [int(block_x1), int(block_y1), int(block_x2), int(block_y2)]


def _recenter_safe_text_positions(
    font: SafeTextPathFont,
    lines: list[str],
    positions: list[tuple[int, int]],
    *,
    target_bbox: list[int],
    padding_y: int,
    vertical_anchor: str,
    vertical_bias_px: int = 0,
) -> list[tuple[int, int]]:
    corrected = list(positions)
    if vertical_anchor == "top" or not corrected:
        return corrected

    measured_bbox = _measure_safe_text_block_bbox(font, lines, corrected)
    if not measured_bbox:
        return corrected

    _, glyph_top, _, glyph_bottom = measured_bbox
    x1, y1, x2, y2 = [int(v) for v in target_bbox]
    box_height = max(1, y2 - y1)
    glyph_height = max(1, glyph_bottom - glyph_top)
    safe_padding_y = max(0, int(padding_y))

    ideal_top = y1 + ((box_height - glyph_height) // 2) + int(vertical_bias_px)
    min_top = y1 + safe_padding_y
    max_top = y2 - safe_padding_y - glyph_height
    if max_top >= min_top:
        ideal_top = min(max(ideal_top, min_top), max_top)
    else:
        ideal_top = min_top

    dy = int(ideal_top - glyph_top)
    if dy == 0:
        return corrected
    return [(lx, ly + dy) for lx, ly in corrected]


def _apply_safe_glow(
    image_np: np.ndarray,
    lines: list[str],
    font: SafeTextPathFont,
    positions: list[tuple[int, int]],
    glow_color: str,
    glow_px: int,
) -> None:
    if glow_px <= 0:
        return

    pad = max(2, int(glow_px) * 2)
    sigma = max(1.0, float(glow_px) * 0.9)
    blur_margin = int(sigma * 3) + 2

    # Compute bounding box of all text lines to work on a cropped region
    img_h, img_w = image_np.shape[:2]
    roi_x1, roi_y1 = img_w, img_h
    roi_x2, roi_y2 = 0, 0

    line_masks: list[tuple[np.ndarray, int, int]] = []
    for line, (lx, ly) in zip(lines, positions):
        mask = _build_textpath_mask(font, line, padding=pad)
        gx = lx - pad
        gy = ly - pad
        line_masks.append((mask, int(gx), int(gy)))
        roi_x1 = min(roi_x1, int(gx))
        roi_y1 = min(roi_y1, int(gy))
        roi_x2 = max(roi_x2, int(gx) + mask.shape[1])
        roi_y2 = max(roi_y2, int(gy) + mask.shape[0])

    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return

    # Expand ROI by blur margin and clamp to image bounds
    roi_x1 = max(0, roi_x1 - blur_margin)
    roi_y1 = max(0, roi_y1 - blur_margin)
    roi_x2 = min(img_w, roi_x2 + blur_margin)
    roi_y2 = min(img_h, roi_y2 + blur_margin)

    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1
    glow_layer = np.zeros((roi_h, roi_w), dtype=np.float32)

    for mask, gx, gy in line_masks:
        x1 = max(roi_x1, gx)
        y1 = max(roi_y1, gy)
        x2 = min(roi_x2, gx + mask.shape[1])
        y2 = min(roi_y2, gy + mask.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue
        mx1 = max(0, x1 - gx)
        my1 = max(0, y1 - gy)
        mx2 = mx1 + (x2 - x1)
        my2 = my1 + (y2 - y1)
        lx1 = x1 - roi_x1
        ly1 = y1 - roi_y1
        lx2 = lx1 + (x2 - x1)
        ly2 = ly1 + (y2 - y1)
        glow_layer[ly1:ly2, lx1:lx2] = np.maximum(
            glow_layer[ly1:ly2, lx1:lx2],
            mask[my1:my2, mx1:mx2].astype(np.float32),
        )

    glow_blur = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if float(np.max(glow_blur)) <= 0.0:
        return

    color = np.array(_parse_hex_color(glow_color), dtype=np.float32)
    alpha = np.clip((glow_blur / 255.0) * 0.82, 0.0, 1.0)[..., None]
    roi = image_np[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
    blended = roi * (1.0 - alpha) + color * alpha
    image_np[roi_y1:roi_y2, roi_x1:roi_x2] = np.clip(blended, 0, 255).astype(np.uint8)


def _apply_safe_gradient_text(
    image_np: np.ndarray,
    lines: list[str],
    font: SafeTextPathFont,
    positions: list[tuple[int, int]],
    color_top: str,
    color_bottom: str,
    outline_color: str,
    outline_px: int,
    start_y: int,
    total_height: int,
) -> None:
    ct = np.array(_parse_hex_color(color_top), dtype=np.float32)
    cb = np.array(_parse_hex_color(color_bottom), dtype=np.float32)

    for line, (lx, ly) in zip(lines, positions):
        pad = max(0, outline_px)
        mask = _build_textpath_mask(font, line, padding=pad)
        origin_x = lx - pad
        origin_y = ly - pad

        if outline_color and outline_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (outline_px * 2 + 1, outline_px * 2 + 1),
            )
            outline_mask = cv2.dilate(mask, kernel, iterations=1)
            _blend_mask_into_image(image_np, outline_mask, origin_x, origin_y, outline_color)

        gradient_patch = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for y in range(mask.shape[0]):
            global_y = (origin_y + y) - start_y
            t = float(np.clip(global_y / max(1, total_height), 0.0, 1.0))
            color = (ct * (1.0 - t) + cb * t).clip(0, 255).astype(np.uint8)
            gradient_patch[y, :] = color

        _blend_rgb_patch_with_mask(image_np, gradient_patch, mask, origin_x, origin_y)


def find_font(font_name: str) -> str | None:
    for font_dir in FONT_DIRS:
        if not font_dir.exists():
            continue
        for path in font_dir.rglob("*"):
            if path.name.lower() == font_name.lower():
                return str(path)
            if path.stem.lower() == Path(font_name).stem.lower():
                return str(path)
    return None


def _should_force_safe_text_path(font_name: str) -> bool:
    lowered = Path(str(font_name or "")).name.lower()
    return any(keyword in lowered for keyword in SAFE_PATH_FORCE_KEYWORDS)


def _normalize_render_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    for source, target in _PUNCT_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    only_marks = re.sub(r"\s+", "", normalized)
    if only_marks and re.fullmatch(r"[.\-]+", only_marks):
        return only_marks
    return normalized.strip()


def _rebalance_wrapped_lines(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    if len(lines) < 2:
        return lines

    def _score(candidate: list[str]) -> float:
        widths = [measure_text_width(font, line) for line in candidate if line.strip()]
        if not widths:
            return float("inf")
        mean = sum(widths) / float(len(widths))
        variance = sum(abs(width - mean) for width in widths) / float(len(widths))
        orphan_penalty = sum(18.0 for line in candidate if len(line.split()) == 1 and len(candidate) > 2)
        return variance + orphan_penalty

    best = list(lines)
    best_score = _score(best)

    for _ in range(3):
        changed = False
        for index in range(len(best) - 1):
            left_words = best[index].split()
            right_words = best[index + 1].split()
            candidates = []

            if len(left_words) >= 2:
                moved = left_words[-1]
                cand_left = " ".join(left_words[:-1]).strip()
                cand_right = " ".join([moved] + right_words).strip()
                if cand_left and measure_text_width(font, cand_left) <= max_width and measure_text_width(font, cand_right) <= max_width:
                    candidate = list(best)
                    candidate[index] = cand_left
                    candidate[index + 1] = cand_right
                    candidates.append(candidate)

            if len(right_words) >= 2:
                moved = right_words[0]
                cand_left = " ".join(left_words + [moved]).strip()
                cand_right = " ".join(right_words[1:]).strip()
                if cand_right and measure_text_width(font, cand_left) <= max_width and measure_text_width(font, cand_right) <= max_width:
                    candidate = list(best)
                    candidate[index] = cand_left
                    candidate[index + 1] = cand_right
                    candidates.append(candidate)

            for candidate in candidates:
                score = _score(candidate)
                if score + 1.0 < best_score:
                    best = candidate
                    best_score = score
                    changed = True
                    break
        if not changed:
            break
    return best


def _dedupe_render_blocks(blocks: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for block in blocks:
        bbox = tuple(int(v) for v in (block.get("balloon_bbox") or block.get("bbox") or []))
        text = re.sub(r"\s+", " ", str(block.get("translated", "")).strip())
        key = (block.get("tipo", "fala"), bbox, text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(block)
    return deduped


def _coerce_bbox(raw_bbox) -> list[int] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in raw_bbox]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _resolve_source_bbox(text_data: dict) -> list[int] | None:
    for key in ("_connected_source_bbox", "source_bbox", "ocr_text_bbox", "bbox"):
        bbox = _coerce_bbox(text_data.get(key))
        if bbox is not None:
            return bbox
    return None


def _join_translated_fragments(texts: list[dict], separator: str = "\n") -> str:
    return separator.join(
        str(text.get("translated", "") or "").strip()
        for text in texts
        if str(text.get("translated", "") or "").strip()
    )


def _connected_confidence_score(text_data: dict) -> float:
    candidates = []
    for key in (
        "connected_position_confidence",
        "connected_group_confidence",
        "connected_detection_confidence",
        "subregion_confidence",
    ):
        try:
            candidates.append(float(text_data.get(key, 0.0) or 0.0))
        except Exception:
            continue
    return max(candidates or [0.0])


def _has_reliable_connected_layout(text_data: dict, subregions: list[list[int]]) -> bool:
    if len(subregions) < 2:
        return False

    group_size = max(1, int(text_data.get("layout_group_size", 1) or 1))
    if 1 < group_size <= len(subregions):
        return True

    explicit_children = text_data.get("connected_children") or []
    if len(explicit_children) == len(subregions):
        return True

    orientation = str(text_data.get("connected_balloon_orientation", "") or "").strip().lower()
    confidence = _connected_confidence_score(text_data)

    if group_size == 1:
        if orientation in {"left-right", "diagonal"}:
            return confidence >= 0.55
        if orientation == "top-bottom":
            return confidence >= 0.72
        return confidence >= 0.82

    if orientation in {"left-right", "diagonal"}:
        return confidence >= 0.18 or confidence == 0.0
    if orientation == "top-bottom":
        return confidence >= 0.55
    return confidence >= 0.72


def get_font(font_name: str, size: int):
    key = (font_name, size)
    if key in _font_cache:
        return _font_cache[key]

    font_path = find_font(font_name)
    if font_path:
        font = SafeTextPathFont(font_path, size)
        _font_cache[key] = font
        return font

    import platform

    fallback_paths = []
    system = platform.system()
    if system == "Windows":
        fallback_paths = [
            r"C:\Windows\Fonts\Arial.ttf",
            r"C:\Windows\Fonts\Tahoma.ttf",
            r"C:\Windows\Fonts\Verdana.ttf",
        ]
    elif system == "Darwin":
        fallback_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    else:
        fallback_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

    for fallback in fallback_paths:
        try:
            font = ImageFont.truetype(fallback, size)
            _font_cache[key] = font
            return font
        except Exception:
            continue

    font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def _shift_preview_bbox(bbox: list[int] | tuple[int, int, int, int] | None, dx: int, dy: int) -> list[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return [dx, dy, dx + 32, dy + 32]
    return [
        int(bbox[0]) + dx,
        int(bbox[1]) + dy,
        int(bbox[2]) + dx,
        int(bbox[3]) + dy,
    ]


def _render_text_preview_asset(project_root: Path, text_data: dict) -> None:
    translated = str(text_data.get("translated", "") or "").strip()
    preview_rel = text_data.get("render_preview_path")
    if not translated or not preview_rel:
        text_data["render_bbox"] = None
        return

    target_bbox = text_data.get("balloon_bbox") or text_data.get("layout_bbox") or text_data.get("bbox")
    x1, y1, x2, y2 = [int(v) for v in _shift_preview_bbox(target_bbox, 0, 0)]
    box_w = max(8, x2 - x1)
    box_h = max(8, y2 - y1)
    padding = max(18, int(max(box_w, box_h) * 0.08))
    bg_key = (1, 2, 3)

    preview_canvas = Image.new("RGB", (box_w + padding * 2, box_h + padding * 2), bg_key)
    local_text = dict(text_data)
    dx = padding - x1
    dy = padding - y1
    local_text["bbox"] = _shift_preview_bbox(text_data.get("bbox") or target_bbox, dx, dy)
    local_text["layout_bbox"] = _shift_preview_bbox(text_data.get("layout_bbox") or target_bbox, dx, dy)
    local_text["source_bbox"] = _shift_preview_bbox(text_data.get("source_bbox") or text_data.get("bbox") or target_bbox, dx, dy)
    local_text["balloon_bbox"] = _shift_preview_bbox(target_bbox, dx, dy)
    local_text["balloon_subregions"] = [
        _shift_preview_bbox(subregion, dx, dy)
        for subregion in (text_data.get("balloon_subregions") or [])
        if isinstance(subregion, (list, tuple)) and len(subregion) >= 4
    ]

    try:
        render_text_block(preview_canvas, local_text)
    except Exception:
        text_data["render_bbox"] = None
        return

    preview_np = np.array(preview_canvas)
    alpha = np.any(preview_np != np.array(bg_key, dtype=np.uint8), axis=2).astype(np.uint8) * 255
    ys, xs = np.where(alpha > 0)
    preview_path = project_root / str(preview_rel)
    if xs.size == 0 or ys.size == 0:
        text_data["render_bbox"] = None
        if preview_path.exists():
            preview_path.unlink(missing_ok=True)
        return

    crop_x1 = int(xs.min())
    crop_y1 = int(ys.min())
    crop_x2 = int(xs.max()) + 1
    crop_y2 = int(ys.max()) + 1

    rgba = np.dstack(
        [
            preview_np[crop_y1:crop_y2, crop_x1:crop_x2],
            alpha[crop_y1:crop_y2, crop_x1:crop_x2],
        ],
    )

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(preview_path)
    text_data["render_bbox"] = [
        int(x1 + crop_x1 - padding),
        int(y1 + crop_y1 - padding),
        int(x1 + crop_x2 - padding),
        int(y1 + crop_y2 - padding),
    ]


def _typeset_single_page(args: tuple) -> int:
    """Renderiza uma única página — projetada para rodar em worker process."""
    img_path_str, trans_page, output_dir_str = args
    img_path = Path(img_path_str)
    output_path = Path(output_dir_str)
    project_root = output_path.parent

    img = Image.open(img_path).convert("RGB")
    texts = trans_page.get("texts", [])

    for text_data in build_render_blocks(texts):
        translated_text = text_data.get("translated", "")
        if not translated_text:
            continue
        try:
            render_text_block(img, text_data)
        except Exception as exc:
            sys.stderr.write(
                f"[typeset] WARN: falha ao renderizar bloco "
                f"'{translated_text[:40]}': {type(exc).__name__}: {exc}\n"
            )
            continue

    for text_data in texts:
        if not isinstance(text_data, dict):
            continue
        _render_text_preview_asset(project_root, text_data)

    dest = output_path / img_path.name
    img.save(dest, quality=95)
    return 0


def run_typesetting(
    inpainted_paths: list[Path],
    translated_results: list[dict],
    output_dir: str,
    progress_callback: Callable | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    project_root = output_path.parent
    total = len(inpainted_paths)

    if total == 0:
        return

    # Serial rendering (FreeType not thread-safe), but I/O threaded:
    # prefetch next image + async save of previous result.
    from concurrent.futures import ThreadPoolExecutor

    def _load_img(path):
        return Image.open(str(path)).convert("RGB")

    def _save_img(img, dest):
        img.save(dest, quality=95)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="typeset-io") as io_pool:
        load_future = io_pool.submit(_load_img, inpainted_paths[0])
        pending_save = None

        for index, (img_path, trans_page) in enumerate(zip(inpainted_paths, translated_results)):
            img = load_future.result()

            # Prefetch next image while rendering current
            if index + 1 < total:
                load_future = io_pool.submit(_load_img, inpainted_paths[index + 1])

            texts = trans_page.get("texts", [])
            for text_data in build_render_blocks(texts):
                translated_text = text_data.get("translated", "")
                if not translated_text:
                    continue
                try:
                    render_text_block(img, text_data)
                except Exception as exc:
                    sys.stderr.write(
                        f"[typeset] WARN: falha ao renderizar bloco "
                        f"'{translated_text[:40]}': {type(exc).__name__}: {exc}\n"
                    )
                    continue

            for text_data in texts:
                if not isinstance(text_data, dict):
                    continue
                _render_text_preview_asset(project_root, text_data)

            # Wait for previous save before starting new one
            if pending_save is not None:
                pending_save.result()

            dest = output_path / Path(img_path).name
            pending_save = io_pool.submit(_save_img, img, dest)

            if progress_callback:
                progress_callback(index + 1, total, f"Tipografia {index + 1}/{total}")

        if pending_save is not None:
            pending_save.result()


def _normalize_balloon_subregions(raw) -> list[list[int]]:
    subregions: list[list[int]] = []
    if not raw:
        return subregions
    for bbox in raw:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        subregions.append([x1, y1, x2, y2])
    return subregions


def _infer_connected_orientation_from_subregions(
    subregions: list[list[int]],
    explicit_orientation: str = "",
) -> str:
    if explicit_orientation:
        return explicit_orientation
    if len(subregions) < 2:
        return ""

    centers = [
        ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        for bbox in subregions[:2]
    ]
    dx = abs(centers[0][0] - centers[1][0])
    dy = abs(centers[0][1] - centers[1][1])
    if dx >= dy * 1.1:
        return "left-right"
    if dy >= dx * 1.1:
        return "top-bottom"
    return "diagonal"


def _order_connected_subregions(
    subregions: list[list[int]],
    explicit_orientation: str = "",
) -> list[list[int]]:
    ordered = [list(bbox) for bbox in subregions]
    orientation = _infer_connected_orientation_from_subregions(ordered, explicit_orientation)
    if orientation == "left-right":
        ordered.sort(key=lambda b: (((b[0] + b[2]) / 2.0), ((b[1] + b[3]) / 2.0)))
    elif orientation == "top-bottom":
        ordered.sort(key=lambda b: (((b[1] + b[3]) / 2.0), ((b[0] + b[2]) / 2.0)))
    else:
        ordered.sort(
            key=lambda b: (
                ((b[1] + b[3]) / 2.0) + ((b[0] + b[2]) / 2.0),
                ((b[1] + b[3]) / 2.0),
                ((b[0] + b[2]) / 2.0),
            ),
        )
    return ordered


def _merge_bbox_list(texts: list[dict]) -> list[int] | None:
    boxes = [
        [int(v) for v in (text.get("bbox") or [])]
        for text in texts
        if isinstance(text.get("bbox"), (list, tuple)) and len(text.get("bbox", [])) == 4
    ]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _compute_connected_vertical_bias_ratio(
    source_bbox: list[int] | None,
    subregion: list[int],
) -> float:
    if not source_bbox or len(source_bbox) != 4:
        return 0.0
    try:
        _, sy1, _, sy2 = [int(v) for v in subregion]
        _, by1, _, by2 = [int(v) for v in source_bbox]
    except Exception:
        return 0.0
    sub_height = max(1, sy2 - sy1)
    source_cy = (by1 + by2) / 2.0
    sub_cy = (sy1 + sy2) / 2.0
    raw = (source_cy - sub_cy) / float(sub_height)
    return float(max(-0.22, min(0.22, raw)))


def _default_connected_vertical_bias_ratio(
    slot_index: int,
    slot_count: int,
    orientation: str,
) -> float:
    if orientation != "left-right" or slot_count != 2:
        return 0.0
    return -0.12 if int(slot_index) == 0 else 0.12


def _resolve_connected_area_weights(text_data: dict, ordered_subregions: list[list[int]]) -> list[float]:
    text_groups = [
        [int(v) for v in bbox]
        for bbox in (text_data.get("connected_text_groups") or [])
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
    ]
    if len(text_groups) == len(ordered_subregions):
        areas = [max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) for bbox in text_groups]
    else:
        focus_bboxes = [
            [int(v) for v in bbox]
            for bbox in ((text_data.get("connected_position_bboxes") or []) or (text_data.get("connected_focus_bboxes") or []))
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4
        ]
        if len(focus_bboxes) == len(ordered_subregions):
            areas = [max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) for bbox in focus_bboxes]
        else:
            areas = [max(1, (sub[2] - sub[0]) * (sub[3] - sub[1])) for sub in ordered_subregions]
    total_area = max(1, sum(areas))
    return [area / float(total_area) for area in areas]


def _resolve_connected_position_bbox(text_data: dict, target_bbox: list[int]) -> list[int]:
    if not text_data.get("_is_lobe_subregion"):
        return list(target_bbox)

    orientation = str(text_data.get("connected_balloon_orientation", "") or "")
    raw_slot_index = text_data.get("_connected_slot_index", -1)
    raw_slot_count = text_data.get("_connected_slot_count", 0)
    slot_index = int(-1 if raw_slot_index is None else raw_slot_index)
    slot_count = int(0 if raw_slot_count is None else raw_slot_count)
    if orientation != "left-right" or slot_count != 2 or slot_index not in (0, 1):
        return list(target_bbox)

    explicit_position_bboxes = [
        [int(v) for v in bbox]
        for bbox in ((text_data.get("connected_position_bboxes") or []) or (text_data.get("connected_focus_bboxes") or []))
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
    ]
    if len(explicit_position_bboxes) == slot_count:
        px1, py1, px2, py2 = explicit_position_bboxes[slot_index]
        tx1, ty1, tx2, ty2 = [int(v) for v in target_bbox]
        clamped = [
            max(tx1, min(tx2, px1)),
            max(ty1, min(ty2, py1)),
            max(tx1, min(tx2, px2)),
            max(ty1, min(ty2, py2)),
        ]
        if clamped[2] > clamped[0] and clamped[3] > clamped[1]:
            return clamped

    def _build_border_driven_bbox() -> list[int]:
        x1, y1, x2, y2 = [int(v) for v in target_bbox]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_x = max(6, int(width * 0.05))
        pad_y = max(6, int(height * 0.05))
        seam_margin = max(10, int(width * 0.14))
        top_focus = max(10, int(height * 0.16))
        bottom_focus = max(10, int(height * 0.18))

        if slot_index == 0:
            pos = [
                x1 + pad_x,
                y1 + pad_y,
                x2 - seam_margin,
                y2 - bottom_focus,
            ]
        else:
            pos = [
                x1 + seam_margin,
                y1 + top_focus,
                x2 - pad_x,
                y2 - max(4, pad_y // 2),
            ]

        if pos[2] <= pos[0]:
            pos[0], pos[2] = x1, x2
        if pos[3] <= pos[1]:
            pos[1], pos[3] = y1, y2
        return [int(v) for v in pos]

    def _shift_bbox_inside_target(
        bbox: list[int],
        desired_center_x: float,
        desired_center_y: float,
    ) -> list[int]:
        tx1, ty1, tx2, ty2 = [int(v) for v in target_bbox]
        bw = max(1, int(bbox[2] - bbox[0]))
        bh = max(1, int(bbox[3] - bbox[1]))

        x1 = int(round(desired_center_x - (bw / 2.0)))
        y1 = int(round(desired_center_y - (bh / 2.0)))
        x2 = x1 + bw
        y2 = y1 + bh

        if x1 < tx1:
            x2 += tx1 - x1
            x1 = tx1
        if x2 > tx2:
            x1 -= x2 - tx2
            x2 = tx2
        if y1 < ty1:
            y2 += ty1 - y1
            y1 = ty1
        if y2 > ty2:
            y1 -= y2 - ty2
            y2 = ty2

        x1 = max(tx1, x1)
        y1 = max(ty1, y1)
        x2 = min(tx2, x2)
        y2 = min(ty2, y2)
        if x2 <= x1:
            x1, x2 = tx1, tx2
        if y2 <= y1:
            y1, y2 = ty1, ty2
        return [int(x1), int(y1), int(x2), int(y2)]

    base_bbox = _build_border_driven_bbox()
    text_groups = [
        [int(v) for v in bbox]
        for bbox in (text_data.get("connected_text_groups") or [])
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
    ]
    if len(text_groups) == slot_count:
        gx1, gy1, gx2, gy2 = text_groups[slot_index]
        focus_center_x = (gx1 + gx2) / 2.0
        focus_center_y = (gy1 + gy2) / 2.0
        base_center_x = (base_bbox[0] + base_bbox[2]) / 2.0
        base_center_y = (base_bbox[1] + base_bbox[3]) / 2.0

        if slot_index == 0:
            desired_center_x = min(base_center_x, focus_center_x)
            desired_center_y = min(base_center_y, focus_center_y)
        else:
            desired_center_x = max(base_center_x, focus_center_x)
            desired_center_y = max(base_center_y, focus_center_y)

        shifted = _shift_bbox_inside_target(base_bbox, desired_center_x, desired_center_y)
        if shifted[2] > shifted[0] and shifted[3] > shifted[1]:
            return shifted

    return base_bbox


def _build_connected_group_block(
    group_texts: list[dict],
    ordered_subregions: list[list[int]],
    balloon_bbox: list[int],
    orientation: str,
) -> dict:
    assignments = _assign_texts_to_subregions(group_texts, ordered_subregions)
    assigned_map = {
        tuple(int(v) for v in assigned_sub): text
        for text, assigned_sub in assignments
    }
    group_style = merge_group_style(group_texts)
    ordered_children = []
    for index, assigned_sub in enumerate(ordered_subregions):
        text = assigned_map.get(tuple(int(v) for v in assigned_sub))
        if text is None:
            continue
        child = dict(text)
        child["estilo"] = group_style
        child["bbox"] = list(assigned_sub)
        child["balloon_bbox"] = list(assigned_sub)
        child["balloon_subregions"] = []
        child["layout_shape"] = _infer_layout_shape_from_bbox(assigned_sub, child.get("tipo", "fala"))
        child["layout_align"] = "top" if child.get("tipo") == "narracao" else "center"
        child["layout_group_size"] = 1
        child["_is_lobe_subregion"] = True
        child["_connected_slot_index"] = index
        child["_connected_slot_count"] = len(ordered_subregions)
        child["connected_balloon_orientation"] = orientation
        source_bbox = [int(v) for v in (text.get("bbox") or assigned_sub)]
        child["_connected_source_bbox"] = source_bbox
        child["_connected_vertical_bias_ratio"] = _compute_connected_vertical_bias_ratio(source_bbox, assigned_sub)
        ordered_children.append(child)

    parent = dict(group_texts[0])
    parent["balloon_bbox"] = list(balloon_bbox)
    parent["balloon_subregions"] = [list(sub) for sub in ordered_subregions]
    parent["connected_children"] = ordered_children
    parent["connected_balloon_orientation"] = orientation
    parent["translated"] = "\n\n".join(
        child.get("translated", "").strip()
        for child in ordered_children
        if child.get("translated", "").strip()
    )
    parent["estilo"] = group_style
    parent["layout_group_size"] = len(ordered_children)
    parent["source_text_count"] = len(group_texts)
    return parent


def _build_connected_group_block_from_fragment_groups(
    grouped_texts: list[list[dict]],
    ordered_subregions: list[list[int]],
    balloon_bbox: list[int],
    orientation: str,
) -> dict:
    flattened = [text for group in grouped_texts for text in group]
    group_style = merge_group_style(flattened)
    ordered_children = []
    for index, (texts_for_subregion, subregion) in enumerate(zip(grouped_texts, ordered_subregions)):
        merged_text = _join_translated_fragments(texts_for_subregion, separator="\n")
        if not merged_text:
            continue
        child = dict(texts_for_subregion[0])
        child["translated"] = merged_text
        child["estilo"] = group_style
        child["bbox"] = list(subregion)
        child["balloon_bbox"] = list(subregion)
        child["balloon_subregions"] = []
        child["layout_shape"] = _infer_layout_shape_from_bbox(subregion, child.get("tipo", "fala"))
        child["layout_align"] = "top" if child.get("tipo") == "narracao" else "center"
        child["layout_group_size"] = 1
        child["_is_lobe_subregion"] = True
        child["_connected_slot_index"] = index
        child["_connected_slot_count"] = len(ordered_subregions)
        child["connected_balloon_orientation"] = orientation
        child["source_text_count"] = len(texts_for_subregion)
        source_bbox = _merge_bbox_list(texts_for_subregion) or list(subregion)
        child["_connected_source_bbox"] = source_bbox
        child["_connected_vertical_bias_ratio"] = _compute_connected_vertical_bias_ratio(source_bbox, subregion)
        ordered_children.append(child)

    parent = dict(flattened[0])
    parent["balloon_bbox"] = list(balloon_bbox)
    parent["balloon_subregions"] = [list(sub) for sub in ordered_subregions]
    parent["connected_children"] = ordered_children
    parent["connected_balloon_orientation"] = orientation
    parent["translated"] = "\n\n".join(
        child.get("translated", "").strip()
        for child in ordered_children
        if child.get("translated", "").strip()
    )
    parent["estilo"] = group_style
    parent["layout_group_size"] = len(flattened)
    parent["source_text_count"] = len(flattened)
    return parent


def _pick_subregion_for_text(text_bbox: list[int], subregions: list[list[int]]) -> list[int] | None:
    if not isinstance(text_bbox, (list, tuple)) or len(text_bbox) != 4:
        return None
    try:
        tx1, ty1, tx2, ty2 = [int(v) for v in text_bbox]
    except Exception:
        return None
    if tx2 <= tx1 or ty2 <= ty1:
        return None

    cx = (tx1 + tx2) / 2.0
    cy = (ty1 + ty2) / 2.0
    text_area = max(1, (tx2 - tx1) * (ty2 - ty1))

    scored: list[tuple[float, bool, list[int]]] = []
    for sx1, sy1, sx2, sy2 in subregions:
        inside = sx1 <= cx <= sx2 and sy1 <= cy <= sy2
        ix1 = max(tx1, sx1)
        iy1 = max(ty1, sy1)
        ix2 = min(tx2, sx2)
        iy2 = min(ty2, sy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ratio = inter / float(text_area)
        scored.append((ratio, inside, [sx1, sy1, sx2, sy2]))

    if not scored:
        return None
    scored.sort(key=lambda item: (1 if item[1] else 0, item[0]), reverse=True)
    best_ratio, best_inside, best_bbox = scored[0]
    second_ratio = scored[1][0] if len(scored) >= 2 else 0.0

    # Só fixa em subregion quando está claramente dentro de um único balão.
    if best_ratio >= 0.55 and (best_ratio - second_ratio) >= 0.12:
        return best_bbox
    if best_inside and best_ratio >= 0.38 and (best_ratio - second_ratio) >= 0.10:
        return best_bbox
    return None


def _assign_texts_to_subregions(
    texts: list[dict],
    subregions: list[list[int]],
) -> list[tuple[dict, list[int]]]:
    """Emparelha textos com subregions por menor distância centro-a-centro.

    Usa matching guloso: para cada texto, calcula a distância euclidiana
    do centro do texto ao centro de cada subregion disponível e atribui
    a mais próxima. Isso funciona para splits horizontais, verticais e
    diagonais sem assumir ordenação fixa.
    """
    if not texts or not subregions:
        return []

    def _center(bbox: list[int]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    sub_centers = [_center(s) for s in subregions]
    text_items = [(t, _center(t.get("bbox", [0, 0, 0, 0]))) for t in texts]

    # Matching guloso: atribuir cada texto à subregion mais próxima não usada
    used_subs: set[int] = set()
    assignments: list[tuple[dict, list[int]]] = []

    # Ordenar textos por distância mínima a qualquer sub (atribuir os mais
    # "óbvios" primeiro para evitar que um texto ambíguo roube a sub de outro)
    def _min_dist(item: tuple) -> float:
        _, tc = item
        return min(
            ((tc[0] - sc[0]) ** 2 + (tc[1] - sc[1]) ** 2) ** 0.5
            for sc in sub_centers
        )

    text_items.sort(key=_min_dist)

    for text, tc in text_items:
        best_idx = -1
        best_dist = float("inf")
        for si, sc in enumerate(sub_centers):
            if si in used_subs:
                continue
            d = ((tc[0] - sc[0]) ** 2 + (tc[1] - sc[1]) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = si
        if best_idx >= 0:
            used_subs.add(best_idx)
            assignments.append((text, list(subregions[best_idx])))

    return assignments


def _group_texts_by_subregions(
    texts: list[dict],
    subregions: list[list[int]],
) -> list[list[dict]]:
    if not texts or not subregions:
        return []

    centers = [
        ((sub[0] + sub[2]) / 2.0, (sub[1] + sub[3]) / 2.0)
        for sub in subregions
    ]
    groups: list[list[dict]] = [[] for _ in subregions]
    confident_assignment_count = 0

    for text in texts:
        text_bbox = text.get("bbox", [0, 0, 0, 0])
        chosen = _pick_subregion_for_text(text_bbox, subregions)
        if chosen is not None:
            idx = next((i for i, sub in enumerate(subregions) if sub == chosen), -1)
            if idx >= 0:
                groups[idx].append(text)
                confident_assignment_count += 1
                continue

        if not isinstance(text_bbox, (list, tuple)) or len(text_bbox) != 4:
            return []
        tx1, ty1, tx2, ty2 = [int(v) for v in text_bbox]
        tcx = (tx1 + tx2) / 2.0
        tcy = (ty1 + ty2) / 2.0
        distances = [
            ((tcx - scx) ** 2 + (tcy - scy) ** 2, idx)
            for idx, (scx, scy) in enumerate(centers)
        ]
        distances.sort(key=lambda item: item[0])
        if len(distances) >= 2:
            best_dist = distances[0][0] ** 0.5
            second_dist = distances[1][0] ** 0.5
            if abs(second_dist - best_dist) < 36:
                return []
        groups[distances[0][1]].append(text)

    if any(not group for group in groups):
        return []

    # If almost everything was ambiguous, keep the semantic fallback.
    if confident_assignment_count < max(1, len(texts) // 2):
        return []

    return [
        sorted(group, key=lambda t: (t.get("bbox", [0, 0, 0, 0])[1], t.get("bbox", [0, 0, 0, 0])[0]))
        for group in groups
    ]


def _bbox_center_point(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_axis_gap(first_bbox: list[int], second_bbox: list[int]) -> tuple[int, int]:
    gap_x = max(0, max(first_bbox[0], second_bbox[0]) - min(first_bbox[2], second_bbox[2]))
    gap_y = max(0, max(first_bbox[1], second_bbox[1]) - min(first_bbox[3], second_bbox[3]))
    return int(gap_x), int(gap_y)


def _merge_group_bbox(group_texts: list[dict]) -> list[int] | None:
    boxes = []
    for text in group_texts:
        bbox = _resolve_source_bbox(text)
        if bbox is not None:
            boxes.append(bbox)
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _split_group_into_connected_bodies(group_texts: list[dict], balloon_bbox: list[int]) -> list[list[dict]]:
    ordered = sorted(
        group_texts,
        key=lambda text: (text.get("bbox", [0, 0, 0, 0])[1], text.get("bbox", [0, 0, 0, 0])[0]),
    )
    if len(ordered) < 3:
        return []

    bx1, by1, bx2, by2 = balloon_bbox
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    best_groups: list[list[dict]] = []
    best_score = float("-inf")

    for pivot in range(1, len(ordered)):
        left_group = ordered[:pivot]
        right_group = ordered[pivot:]
        left_bbox = _merge_group_bbox(left_group)
        right_bbox = _merge_group_bbox(right_group)
        prev_bbox = _resolve_source_bbox(left_group[-1])
        next_bbox = _resolve_source_bbox(right_group[0])
        if left_bbox is None or right_bbox is None or prev_bbox is None or next_bbox is None:
            continue

        left_center_x, left_center_y = _bbox_center_point(left_bbox)
        right_center_x, right_center_y = _bbox_center_point(right_bbox)
        prev_center_x, prev_center_y = _bbox_center_point(prev_bbox)
        next_center_x, next_center_y = _bbox_center_point(next_bbox)
        pivot_dx = next_center_x - prev_center_x
        pivot_dy = next_center_y - prev_center_y
        group_dx = right_center_x - left_center_x
        group_dy = right_center_y - left_center_y
        gap_x, gap_y = _bbox_axis_gap(left_bbox, right_bbox)
        left_height = max(1, left_bbox[3] - left_bbox[1])
        right_height = max(1, right_bbox[3] - right_bbox[1])
        height_delta_ratio = abs(left_height - right_height) / float(max(left_height, right_height))

        if pivot_dx < max(72.0, bw * 0.12):
            continue
        if pivot_dy < max(12.0, bh * 0.04):
            continue
        if group_dx < max(96.0, bw * 0.16):
            continue
        if group_dy < max(24.0, bh * 0.06):
            continue
        if gap_x < max(18, int(bw * 0.025)) and height_delta_ratio < 0.16:
            continue

        score = 0.0
        score += pivot_dx / float(max(1, bw))
        score += group_dx / float(max(1, bw))
        score += max(0, gap_x) / float(max(1, bw))
        score += pivot_dy / float(max(1, bh))
        score += height_delta_ratio * 0.7
        score -= abs(len(left_group) - len(right_group)) * 0.08

        if score > best_score:
            best_score = score
            best_groups = [left_group, right_group]

    return best_groups if best_score > 0.0 else []


def _infer_alignment_from_source_position(text_data: dict, position_bbox: list[int]) -> str:
    return "center"


def _looks_like_connected_balloon_pair(group_texts: list[dict], balloon_bbox: list[int]) -> bool:
    if len(group_texts) != 2:
        return False

    source_bboxes = [_resolve_source_bbox(text) for text in group_texts]
    if any(bbox is None for bbox in source_bboxes):
        return False

    first_bbox = source_bboxes[0]
    second_bbox = source_bboxes[1]
    if first_bbox is None or second_bbox is None:
        return False

    bx1, by1, bx2, by2 = balloon_bbox
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    first_center_x, first_center_y = _bbox_center_point(first_bbox)
    second_center_x, second_center_y = _bbox_center_point(second_bbox)
    dx = abs(second_center_x - first_center_x)
    dy = abs(second_center_y - first_center_y)
    gap_x, gap_y = _bbox_axis_gap(first_bbox, second_bbox)
    first_height = max(1, first_bbox[3] - first_bbox[1])
    second_height = max(1, second_bbox[3] - second_bbox[1])
    height_delta_ratio = abs(first_height - second_height) / float(max(first_height, second_height))

    below_and_right = second_center_x > first_center_x and second_center_y > first_center_y
    strong_x_distance = dx >= max(52.0, bw * 0.22) or gap_x >= max(16, int(bw * 0.05))
    strong_y_distance = dy >= max(30.0, bh * 0.12) or gap_y >= max(12, int(bh * 0.04))
    meaningful_height_delta = height_delta_ratio >= 0.18 and dy >= max(20.0, bh * 0.08)

    if below_and_right and strong_x_distance and strong_y_distance:
        return True
    if strong_x_distance and gap_x >= max(20, int(bw * 0.06)):
        return True
    if meaningful_height_delta:
        return True
    return False


def _enforce_min_connected_lobe_size(
    subregions: list[list[int]],
    balloon_bbox: list[int],
) -> list[list[int]]:
    bx1, by1, bx2, by2 = balloon_bbox
    min_ratio = 0.30
    normalized = [
        [max(int(sub[0]), bx1), max(int(sub[1]), by1), min(int(sub[2]), bx2), min(int(sub[3]), by2)]
        for sub in subregions
        if isinstance(sub, (list, tuple)) and len(sub) == 4
    ]
    if len(normalized) != 2:
        return normalized

    first, second = normalized
    first_width = max(1, first[2] - first[0])
    second_width = max(1, second[2] - second[0])
    total_width = first_width + second_width
    if total_width > 0:
        if first_width / float(total_width) < min_ratio:
            needed = int(total_width * min_ratio) - first_width
            first[2] = min(bx2, first[2] + needed)
            second[0] = first[2]
        elif second_width / float(total_width) < min_ratio:
            needed = int(total_width * min_ratio) - second_width
            second[0] = max(bx1, second[0] - needed)
            first[2] = second[0]

    first_height = max(1, first[3] - first[1])
    second_height = max(1, second[3] - second[1])
    total_height = first_height + second_height
    if total_height > 0:
        if first_height / float(total_height) < min_ratio:
            needed = int(total_height * min_ratio) - first_height
            first[3] = min(by2, first[3] + needed)
            second[1] = first[3]
        elif second_height / float(total_height) < min_ratio:
            needed = int(total_height * min_ratio) - second_height
            second[1] = max(by1, second[1] - needed)
            first[3] = second[1]
    return normalized


def _build_connected_subregions_from_group_boxes(
    group_bboxes: list[list[int]],
    balloon_bbox: list[int],
) -> list[list[int]]:
    if len(group_bboxes) < 2:
        return []

    ordered = [list(bbox) for bbox in group_bboxes[:2]]
    bx1, by1, bx2, by2 = balloon_bbox
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    centers = [_bbox_center_point(bbox) for bbox in ordered]
    dx = abs(centers[0][0] - centers[1][0])
    dy = abs(centers[0][1] - centers[1][1])

    if dy >= dx * 1.1:
        top_bbox, bottom_bbox = sorted(ordered, key=lambda bbox: (bbox[1], bbox[0]))
        if top_bbox[3] < bottom_bbox[1]:
            seam_y = int((top_bbox[3] + bottom_bbox[1]) / 2.0)
        else:
            seam_y = int((centers[0][1] + centers[1][1]) / 2.0)
        seam_y = max(by1 + 32, min(by2 - 32, seam_y))
        gap_y = max(4, int(bh * 0.03))
        return _enforce_min_connected_lobe_size(
            [[bx1, by1, bx2, seam_y - gap_y], [bx1, seam_y + gap_y, bx2, by2]],
            balloon_bbox,
        )

    if dx >= dy * 1.1:
        left_bbox, right_bbox = sorted(ordered, key=lambda bbox: (bbox[0], bbox[1]))
        if left_bbox[2] < right_bbox[0]:
            seam_x = int((left_bbox[2] + right_bbox[0]) / 2.0)
        else:
            seam_x = int((centers[0][0] + centers[1][0]) / 2.0)
        seam_x = max(bx1 + 32, min(bx2 - 32, seam_x))
        gap_x = max(4, int(bw * 0.0125))
        return _enforce_min_connected_lobe_size(
            [[bx1, by1, seam_x - gap_x, by2], [seam_x + gap_x, by1, bx2, by2]],
            balloon_bbox,
        )

    first_bbox, second_bbox = sorted(ordered, key=lambda bbox: (bbox[1], bbox[0]))
    first_center = _bbox_center_point(first_bbox)
    second_center = _bbox_center_point(second_bbox)

    if first_bbox[2] < second_bbox[0]:
        seam_x = int((first_bbox[2] + second_bbox[0]) / 2.0)
    else:
        seam_x = int((first_center[0] + second_center[0]) / 2.0)
    if first_bbox[3] < second_bbox[1]:
        seam_y = int((first_bbox[3] + second_bbox[1]) / 2.0)
    else:
        seam_y = int((first_center[1] + second_center[1]) / 2.0)

    seam_x = max(bx1 + 32, min(bx2 - 32, seam_x))
    seam_y = max(by1 + 28, min(by2 - 28, seam_y))
    gap_x = max(6, int(bw * 0.02))
    overlap_y = max(6, int(bh * 0.03))

    if first_center[0] <= second_center[0]:
        subregions = [
            [bx1, by1, min(bx2, seam_x - gap_x), min(by2, seam_y + overlap_y)],
            [max(bx1, seam_x + gap_x), max(by1, seam_y - overlap_y), bx2, by2],
        ]
    else:
        subregions = [
            [max(bx1, seam_x + gap_x), by1, bx2, min(by2, seam_y + overlap_y)],
            [bx1, max(by1, seam_y - overlap_y), min(bx2, seam_x - gap_x), by2],
        ]
    return _enforce_min_connected_lobe_size(subregions, balloon_bbox)


def _synthesize_connected_subregions(group_texts: list[dict], balloon_bbox: list[int]) -> list[list[int]]:
    if not _looks_like_connected_balloon_pair(group_texts, balloon_bbox):
        return []
    source_bboxes = [_resolve_source_bbox(text) for text in group_texts]
    if any(bbox is None for bbox in source_bboxes):
        return []
    return _build_connected_subregions_from_group_boxes(
        [bbox for bbox in source_bboxes if bbox is not None],
        balloon_bbox,
    )


def build_render_blocks(texts: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, tuple[int, int, int, int]], list[dict]] = {}
    passthrough: list[dict] = []

    # Fase 1: Pré-agrupar textos multi-texto que compartilham subregions
    multi_sub_groups: dict[tuple[str, tuple], list[dict]] = {}
    for text in texts:
        balloon_bbox = text.get("balloon_bbox")
        tipo = text.get("tipo", "fala")
        subregions = _normalize_balloon_subregions(text.get("balloon_subregions", []))
        if (
            len(subregions) >= 2
            and balloon_bbox
            and int(text.get("layout_group_size", 1)) > 1
            and _has_reliable_connected_layout(text, subregions)
        ):
            key = (tipo, tuple(int(v) for v in balloon_bbox))
            multi_sub_groups.setdefault(key, []).append(text)

    # Fase 2: Atribuir textos a subregions quando as contagens casam.
    # Se as contagens NÃO casam (ex: 6 textos OCR para 2 subregions), mescla
    # todos em 1 bloco consolidado e mantém balloon_subregions para o renderer
    # dividir semanticamente.
    assigned_ids: set[int] = set()
    for key, group_texts in multi_sub_groups.items():
        if len(group_texts) < 2:
            continue
        subregions = _normalize_balloon_subregions(group_texts[0].get("balloon_subregions", []))
        orientation = str(group_texts[0].get("connected_balloon_orientation", "") or "")
        ordered_subregions = _order_connected_subregions(subregions, orientation)
        if len(group_texts) == len(subregions):
            passthrough.append(
                _build_connected_group_block(
                    group_texts,
                    ordered_subregions,
                    list(key[1]),
                    _infer_connected_orientation_from_subregions(ordered_subregions, orientation),
                ),
            )
            for text in group_texts:
                assigned_ids.add(id(text))
        else:
            grouped_by_lobe = _group_texts_by_subregions(group_texts, ordered_subregions)
            if grouped_by_lobe:
                passthrough.append(
                    _build_connected_group_block_from_fragment_groups(
                        grouped_by_lobe,
                        ordered_subregions,
                        list(key[1]),
                        _infer_connected_orientation_from_subregions(ordered_subregions, orientation),
                    ),
                )
            else:
                # N:M – sem atribuicao geometrica confiavel; manter merge semanticamente.
                ordered = sorted(
                    group_texts,
                    key=lambda t: (t.get("bbox", [0, 0, 0, 0])[1], t.get("bbox", [0, 0, 0, 0])[0]),
                )
                merged = dict(ordered[0])
                merged["translated"] = _join_translated_fragments(ordered, separator="\n")
                merged["estilo"] = merge_group_style(ordered)
                merged["balloon_bbox"] = list(key[1])
                merged["balloon_subregions"] = ordered_subregions
                merged["connected_balloon_orientation"] = _infer_connected_orientation_from_subregions(
                    ordered_subregions,
                    orientation,
                )
                merged["layout_group_size"] = len(ordered)
                merged["source_text_count"] = len(ordered)
                passthrough.append(merged)
            for text in group_texts:
                assigned_ids.add(id(text))

    # Fase 3: Processar textos restantes normalmente
    for text in texts:
        if id(text) in assigned_ids:
            continue

        balloon_bbox = text.get("balloon_bbox")
        tipo = text.get("tipo", "fala")

        subregions = _normalize_balloon_subregions(text.get("balloon_subregions", []))

        if len(subregions) >= 2:
            if _has_reliable_connected_layout(text, subregions):
                passthrough.append(text)
            else:
                text = dict(text)
                text["balloon_subregions"] = []
                text["connected_children"] = []
                text["connected_balloon_orientation"] = ""
                text["connected_detection_confidence"] = 0.0
                text["connected_group_confidence"] = 0.0
                text["connected_position_confidence"] = 0.0
                if (
                    balloon_bbox
                    and int(text.get("layout_group_size", 1)) > 1
                    and tipo in {"fala", "narracao", "pensamento"}
                ):
                    grouped.setdefault((tipo, tuple(int(v) for v in balloon_bbox)), []).append(text)
                else:
                    passthrough.append(text)
            continue

        if balloon_bbox and len(subregions) == 1:
            chosen = _pick_subregion_for_text(text.get("bbox", [0, 0, 0, 0]), subregions)
            if chosen:
                text = dict(text)
                text["balloon_bbox"] = chosen
                text["balloon_subregions"] = []
                text["layout_shape"] = _infer_layout_shape_from_bbox(chosen, tipo)
                text["layout_align"] = "top" if tipo == "narracao" else "center"
                text["_resolved_subregion"] = True
                balloon_bbox = chosen

        if (
            balloon_bbox
            and int(text.get("layout_group_size", 1)) > 1
            and tipo in {"fala", "narracao", "pensamento"}
        ):
            key = (tipo, tuple(int(v) for v in balloon_bbox))
            grouped.setdefault(key, []).append(text)
        else:
            passthrough.append(text)

    blocks = list(passthrough)
    for (tipo, bbox_tuple), group in grouped.items():
        ordered = sorted(
            group,
            key=lambda item: (
                item.get("bbox", [0, 0, 0, 0])[1],
                item.get("bbox", [0, 0, 0, 0])[0],
            ),
        )
        body_groups = _split_group_into_connected_bodies(ordered, list(bbox_tuple))
        if len(body_groups) == 2:
            group_boxes = [_merge_group_bbox(body_group) for body_group in body_groups]
            if all(group_boxes):
                synthetic_subregions = _build_connected_subregions_from_group_boxes(
                    [bbox for bbox in group_boxes if bbox is not None],
                    list(bbox_tuple),
                )
                if len(synthetic_subregions) >= 2:
                    blocks.append(
                        _build_connected_group_block_from_fragment_groups(
                            body_groups,
                            _order_connected_subregions(synthetic_subregions),
                            list(bbox_tuple),
                            _infer_connected_orientation_from_subregions(synthetic_subregions),
                        ),
                    )
                    continue

        synthetic_subregions = _synthesize_connected_subregions(ordered, list(bbox_tuple))
        if len(synthetic_subregions) >= 2:
            blocks.append(
                _build_connected_group_block(
                    ordered,
                    _order_connected_subregions(synthetic_subregions),
                    list(bbox_tuple),
                    _infer_connected_orientation_from_subregions(synthetic_subregions),
                ),
            )
            continue
        combined = dict(ordered[0])
        combined.pop("_resolved_subregion", None)
        combined["balloon_bbox"] = list(bbox_tuple)
        combined["translated"] = _join_translated_fragments(ordered, separator="\n")
        combined["estilo"] = merge_group_style(ordered)
        combined["source_text_count"] = len(ordered)
        combined["layout_group_size"] = len(ordered)
        if combined.get("balloon_subregions") and all(text.get("_resolved_subregion") for text in ordered):
            combined["balloon_subregions"] = []
        blocks.append(combined)

    return _dedupe_render_blocks(blocks)


def merge_group_style(group: list[dict]) -> dict:
    styles = [text.get("estilo", {}) for text in group]
    best_outlined = max(
        styles,
        key=lambda style: (
            int(style.get("contorno_px", 0)),
            1 if style.get("contorno") else 0,
            int(style.get("tamanho", 0)),
        ),
    )
    largest = max(styles, key=lambda style: int(style.get("tamanho", 0)))
    merged = dict(styles[0]) if styles else {}
    merged["tamanho"] = largest.get("tamanho", merged.get("tamanho", 16))
    merged["contorno"] = best_outlined.get("contorno", merged.get("contorno", "#000000"))
    merged["contorno_px"] = best_outlined.get("contorno_px", merged.get("contorno_px", 2))
    merged["cor"] = best_outlined.get("cor", merged.get("cor", "#FFFFFF"))
    merged["alinhamento"] = merged.get("alinhamento", "center")
    # Gradient: take the first style that has one
    merged["cor_gradiente"] = next(
        (s.get("cor_gradiente") for s in styles if s.get("cor_gradiente")), []
    )
    # Glow: use strongest
    merged["glow"] = any(s.get("glow", False) for s in styles)
    merged["glow_cor"] = next((s.get("glow_cor", "") for s in styles if s.get("glow_cor")), "")
    merged["glow_px"] = max((int(s.get("glow_px", 0)) for s in styles), default=0)
    # Shadow: take first that has it
    merged["sombra"] = any(s.get("sombra", False) for s in styles)
    merged["sombra_cor"] = next((s.get("sombra_cor", "") for s in styles if s.get("sombra_cor")), "")
    merged["sombra_offset"] = next(
        (s.get("sombra_offset", [0, 0]) for s in styles if s.get("sombra")), [0, 0]
    )
    return merged


def _detect_balloon_geometry(text_data: dict) -> str:
    """Detecta se o balão é retangular ou elíptico.
    Balões brancos (fala/pensamento) = elíptico.
    Balões texturizados, narração, sfx = retangular."""
    tipo = text_data.get("tipo", "fala")
    if tipo in ("narracao", "sfx"):
        return "rect"
    # Só trata como retangular quando a fonte indica balão texturizado/estilizado.
    estilo = text_data.get("estilo", {})
    fonte = str(estilo.get("fonte", "") or "").lower()
    if any(keyword in fonte for keyword in SAFE_PATH_FORCE_KEYWORDS):
        return "rect"
    return "ellipse"


def plan_text_layout(text_data: dict) -> dict:
    target_bbox = text_data.get("balloon_bbox") or text_data.get("bbox") or [0, 0, 0, 0]
    position_bbox = _resolve_connected_position_bbox(text_data, target_bbox)
    x1, y1, x2, y2 = target_bbox
    px1, py1, px2, py2 = [int(v) for v in position_bbox]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    position_width = max(1, px2 - px1)
    position_height = max(1, py2 - py1)

    tipo = text_data.get("tipo", "fala")
    layout_shape = text_data.get("layout_shape", "square")
    layout_align = text_data.get("layout_align", "center")
    group_size = max(1, int(text_data.get("layout_group_size", 1)))
    estilo = text_data.get("estilo", {})
    corpus_visual = text_data.get("corpus_visual_benchmark", {}) or {}
    corpus_textual = text_data.get("corpus_textual_benchmark", {}) or {}
    balloon_geo = _detect_balloon_geometry(text_data)

    if tipo == "narracao":
        # Retangular — usa mais espaço horizontal
        width_ratio = 0.9 if layout_shape == "wide" else 0.85
        vertical_anchor = "top"
        padding_y = 10
        line_spacing = 0.12
    elif tipo == "sfx":
        # Retangular — SFX compacto
        width_ratio = 0.76 if layout_shape == "tall" else 0.82
        vertical_anchor = "center"
        padding_y = 6
        line_spacing = 0.05
    elif balloon_geo == "ellipse":
        # Speech balloons were being under-used, which forced too many line breaks
        # and collapsed the font size. Be less conservative here.
        if layout_shape == "tall":
            width_ratio = 0.74
            padding_y = max(8, int(box_height * 0.13))
        elif layout_shape == "wide":
            width_ratio = 0.86
            padding_y = max(8, int(box_height * 0.14))
        else:
            width_ratio = 0.82
            padding_y = max(8, int(box_height * 0.14))
        vertical_anchor = "center"
        line_spacing = 0.08
    else:
        # Retangular texturizado — margem de segurança para não ultrapassar
        width_ratio = 0.72
        vertical_anchor = "center"
        padding_y = max(6, int(box_height * 0.10))
        line_spacing = 0.1

    if group_size > 1 and tipo == "fala":
        # Connected-balloon content already gets split; avoid over-shrinking width.
        width_ratio -= 0.02

    # Lobe subregions have a flat seam edge — use "lobe" geometry with
    # scoring targets adapted to the lobe's own aspect ratio.
    if text_data.get("_is_lobe_subregion"):
        balloon_geo = "lobe"
        lobe_aspect = box_width / float(max(1, box_height))
        if lobe_aspect >= 1.4:
            # Wide lobe — seam edge is flat, text can extend closer to it
            width_ratio = 0.95
            padding_y = max(6, int(box_height * 0.07))
        elif lobe_aspect <= 0.7:
            # Tall lobe
            width_ratio = 0.92
            padding_y = max(6, int(box_height * 0.05))
        else:
            # Square-ish lobe — one edge flat (seam), allows wider text
            width_ratio = 0.95
            padding_y = max(6, int(box_height * 0.06))
        line_spacing = 0.04

    width_ratio, target_size_delta, outline_boost = _apply_corpus_layout_hints(
        width_ratio=width_ratio,
        tipo=tipo,
        layout_shape=layout_shape,
        corpus_visual=corpus_visual,
        corpus_textual=corpus_textual,
    )

    target_size = max(10, int(estilo.get("tamanho", 16)) + target_size_delta)
    outline_px = max(int(estilo.get("contorno_px", 2)), outline_boost)
    raw_vertical_bias_ratio = float(text_data.get("_connected_vertical_bias_ratio", 0.0) or 0.0)
    connected_orientation = str(text_data.get("connected_balloon_orientation", "") or "")
    raw_slot_index = text_data.get("_connected_slot_index", -1)
    slot_index = int(-1 if raw_slot_index is None else raw_slot_index)
    vertical_bias_px = 0
    if text_data.get("_is_lobe_subregion") and raw_vertical_bias_ratio:
        max_bias_px = max(10, int(box_height * 0.22))
        vertical_bias_px = int(round(box_height * raw_vertical_bias_ratio))
        vertical_bias_px = max(-max_bias_px, min(max_bias_px, vertical_bias_px))
    if text_data.get("_is_lobe_subregion") and connected_orientation == "left-right":
        if slot_index == 0:
            vertical_bias_px += max(24, int(box_height * 0.095))
        elif slot_index == 1:
            vertical_bias_px += max(10, int(box_height * 0.04))
    return {
        "target_bbox": target_bbox,
        "position_bbox": position_bbox,
        "layout_shape": layout_shape,
        "balloon_geo": balloon_geo,
        "max_width": max(40, int(box_width * width_ratio)),
        "max_height": max(20, box_height - (padding_y * 2)),
        "padding_y": padding_y,
        "vertical_anchor": vertical_anchor if layout_align != "top" else "top",
        "alignment": _infer_alignment_from_source_position(text_data, position_bbox),
        "font_name": estilo.get("fonte", DEFAULT_FONTS.get(tipo, "AnimeAce.ttf")),
        "target_size": target_size + (4 if tipo == "sfx" else 0),
        "text_color": estilo.get("cor", "#FFFFFF"),
        "cor_gradiente": estilo.get("cor_gradiente", []),
        "outline_color": estilo.get("contorno", "#000000"),
        "outline_px": outline_px,
        "glow": estilo.get("glow", False),
        "glow_cor": estilo.get("glow_cor", ""),
        "glow_px": int(estilo.get("glow_px", 0)),
        "sombra": estilo.get("sombra", False),
        "sombra_cor": estilo.get("sombra_cor", ""),
        "sombra_offset": estilo.get("sombra_offset", [0, 0]),
        "line_spacing_ratio": line_spacing,
        "vertical_bias_px": vertical_bias_px,
    }


def _infer_layout_shape_from_bbox(bbox: list[int], tipo: str) -> str:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect = width / float(height)
    if tipo == "narracao":
        return "wide" if aspect >= 1.6 else "square"
    if aspect >= 1.45:
        return "wide"
    if aspect <= 0.9:
        return "tall"
    return "square"


def _split_text_for_connected_balloons(
    text: str,
    count: int,
    area_weights: list[float] | None = None,
) -> list[str]:
    """Split text into `count` chunks for connected balloon subregions.

    Splitting priority:
      1. Explicit newlines (\\n)
      2. Best semantic boundary (sentence → clause) scored by balance + coherence
      3. Word-level split as last resort when no semantic boundary is close enough

    For step 2, all possible split points at clause boundaries are tried and
    scored. The best semantic split (minimum imbalance vs area weights) is used
    as long as it has deviation ≤ 0.25. This avoids blindly breaking in the
    middle of sentences while still getting a reasonable visual balance.

    When `area_weights` is provided (one float per subregion, summing to ~1.0),
    larger subregions receive proportionally more text.
    """
    stripped = text.strip()
    if count <= 1 or not stripped:
        return [stripped]

    # 1. Explicit newlines
    newline_parts = [part.strip() for part in re.split(r"\n+", stripped) if part.strip()]
    if len(newline_parts) >= count:
        return _merge_chunks_to_target_count(newline_parts, count, area_weights)

    # 2. Semantic split: collect clause segments (sentence ends + comma clauses)
    #    then try ALL possible k-way groupings and pick the most balanced one.
    clause_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?…,;])(?:\s+)", stripped)
        if part.strip()
    ]
    if len(clause_parts) >= count:
        best = _best_semantic_split(clause_parts, count, area_weights)
        if best is not None:
            return best

    # 3. Word-level split as final fallback
    return _split_words_weighted(stripped, count, area_weights)


def _best_semantic_split(
    parts: list[str],
    count: int,
    area_weights: list[float] | None,
) -> list[str] | None:
    """Try all consecutive groupings of `parts` into `count` chunks.

    Scores each candidate by its imbalance vs `area_weights`.
    Returns the grouping with the lowest imbalance if it is ≤ 0.25,
    otherwise returns None (caller should fall through to word split).

    Semantic coherence bonus: prefer splits that end at sentence boundaries
    (., !, ?) over those that end at clause boundaries (, ;).
    """
    words_per_part = [len(p.split()) for p in parts]
    total_words = max(1, sum(words_per_part))
    weights = area_weights if (area_weights and len(area_weights) == count) else None
    uniform_w = 1.0 / count

    best_candidate: list[str] | None = None
    best_score = float("inf")

    # For count=2: try every split point k in [1, N-1].
    # For count>2: enumerate via recursive partition (bounded by N choose count).
    # In practice count is always 2 for connected balloons.
    n = len(parts)

    def _try_partition(start: int, slots_left: int, current: list[list[str]]) -> None:
        nonlocal best_candidate, best_score
        if slots_left == 1:
            group = parts[start:]
            if not group:
                return
            candidate = [" ".join(g) for g in current + [group]]
            # Score: sum of |ratio_i - weight_i| across all slots
            chunk_words = [len(c.split()) for c in candidate]
            ratios = [wc / float(total_words) for wc in chunk_words]
            target = [weights[i] if weights else uniform_w for i in range(count)]
            imbalance = max(abs(r - t) for r, t in zip(ratios, target))
            # Semantic coherence bonus: reward splits where each group ends at . ! ?
            # Also check the final group so we don't split mid-sentence anywhere.
            coherence_penalty = 0.0
            all_groups = list(current) + [group]
            for g in all_groups:
                last_part = g[-1] if g else ""
                if not re.search(r"[.!?…]$", last_part):
                    coherence_penalty += 0.10  # stronger penalty for comma/semicolon break
            total = imbalance + coherence_penalty
            if total < best_score:
                best_score = total
                best_candidate = candidate
            return
        for k in range(start + 1, n - slots_left + 2):
            _try_partition(k, slots_left - 1, current + [parts[start:k]])

    _try_partition(0, count, [])

    # Accept if best semantic split has imbalance ≤ 0.25
    if best_candidate is not None and best_score <= 0.25:
        return best_candidate
    return None


def _merge_chunks_to_target_count(
    chunks: list[str],
    count: int,
    area_weights: list[float] | None = None,
) -> list[str]:
    if len(chunks) <= count:
        padded = list(chunks)
        while len(padded) < count:
            padded.append("")
        return padded[:count]

    merged = [chunk.strip() for chunk in chunks if chunk.strip()]

    if area_weights and len(area_weights) == count:
        # Distribute chunks proportionally to area weights
        total_words = sum(len(c.split()) for c in merged)
        targets = [max(1, int(w * total_words + 0.5)) for w in area_weights]
        result: list[str] = []
        cursor = 0
        for slot_idx in range(count):
            target_wc = targets[slot_idx]
            slot_parts: list[str] = []
            slot_wc = 0
            while cursor < len(merged):
                part_wc = len(merged[cursor].split())
                # Always take at least one chunk per slot
                if slot_wc > 0 and slot_wc + part_wc > target_wc * 1.3 and slot_idx < count - 1:
                    break
                slot_parts.append(merged[cursor])
                slot_wc += part_wc
                cursor += 1
            result.append(" ".join(slot_parts).strip())
        # Dump remaining into last slot
        if cursor < len(merged):
            remainder = " ".join(merged[cursor:]).strip()
            if result:
                result[-1] = f"{result[-1]} {remainder}".strip()
            else:
                result.append(remainder)
        return [r for r in result if r] or [stripped for stripped in [text.strip()] if stripped]

    # Fallback: merge smallest adjacent pair until we reach target count
    while len(merged) > count:
        smallest_index = min(range(len(merged) - 1), key=lambda idx: len(merged[idx].split()))
        merged[smallest_index] = f"{merged[smallest_index]} {merged.pop(smallest_index + 1)}".strip()
    return merged


def _split_words_weighted(
    text: str,
    count: int,
    area_weights: list[float] | None = None,
) -> list[str]:
    """Split text by words, distributing proportionally to area weights."""
    words = text.split()
    if not words:
        return [text.strip()]

    total_words = len(words)

    if area_weights and len(area_weights) == count:
        # Proportional word distribution
        targets = [max(1, round(w * total_words)) for w in area_weights]
        # Adjust so targets sum to total_words
        diff = total_words - sum(targets)
        if diff != 0:
            idx = max(range(count), key=lambda i: targets[i])
            targets[idx] += diff
    else:
        base = total_words // count
        remainder = total_words % count
        targets = [base + (1 if i < remainder else 0) for i in range(count)]

    chunks = []
    cursor = 0
    for take in targets:
        if take <= 0:
            continue
        chunk_words = words[cursor:cursor + take]
        cursor += take
        chunks.append(" ".join(chunk_words).strip())
    return [chunk for chunk in chunks if chunk]


def _enumerate_connected_text_candidates(
    text: str,
    count: int,
    area_weights: list[float] | None = None,
) -> list[dict]:
    stripped = text.strip()
    if count <= 1 or not stripped:
        return [{"chunks": [stripped], "semantic_bonus": 0.0, "label": "single"}]

    weight_variants = []
    if area_weights and len(area_weights) == count:
        weight_variants.append(area_weights)
    weight_variants.append(None)

    candidate_map: dict[tuple[str, ...], dict] = {}

    def _register(chunks: list[str], semantic_bonus: float, label: str) -> None:
        cleaned = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
        if len(cleaned) != count:
            return
        key = tuple(cleaned)
        existing = candidate_map.get(key)
        payload = {
            "chunks": cleaned,
            "semantic_bonus": float(semantic_bonus),
            "label": label,
        }
        if existing is None or payload["semantic_bonus"] > existing["semantic_bonus"]:
            candidate_map[key] = payload

    for weights in weight_variants:
        _register(_split_text_for_connected_balloons(stripped, count, weights), 1.2 if weights else 1.0, "semantic")
        _register(_split_words_weighted(stripped, count, weights), 0.3 if weights else 0.15, "words")

    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?…])(?:\s+)", stripped)
        if part.strip()
    ]
    clause_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?…,;:])(?:\s+)", stripped)
        if part.strip()
    ]
    explicit_parts = [part.strip() for part in re.split(r"\n+", stripped) if part.strip()]

    def _register_partitioned(parts: list[str], base_bonus: float, label: str) -> None:
        if len(parts) < count:
            return
        if len(parts) == count:
            _register(parts, base_bonus, label)
            return
        best_local: list[tuple[float, list[str]]] = []
        total_words = max(1, sum(len(part.split()) for part in parts))
        targets = area_weights if (area_weights and len(area_weights) == count) else [1.0 / count] * count

        def _walk(start: int, slots_left: int, groups: list[list[str]]) -> None:
            if slots_left == 1:
                grouped = groups + [parts[start:]]
                chunks = [" ".join(group).strip() for group in grouped]
                chunk_words = [len(chunk.split()) for chunk in chunks]
                ratios = [words / float(total_words) for words in chunk_words]
                imbalance = max(abs(r - t) for r, t in zip(ratios, targets))
                punctuation_bonus = 0.0
                for chunk in chunks:
                    if re.search(r"[.!?…]$", chunk):
                        punctuation_bonus += 0.15
                    elif re.search(r"[,;:]$", chunk):
                        punctuation_bonus += 0.05
                score = imbalance - punctuation_bonus
                best_local.append((score, chunks))
                return
            max_split = len(parts) - slots_left + 1
            for pivot in range(start + 1, max_split + 1):
                _walk(pivot, slots_left - 1, groups + [parts[start:pivot]])

        _walk(0, count, [])
        best_local.sort(key=lambda item: item[0])
        for score, chunks in best_local[:4]:
            _register(chunks, base_bonus - score, label)

    _register_partitioned(explicit_parts, 1.8, "newline")
    _register_partitioned(sentence_parts, 2.2, "sentence")
    _register_partitioned(clause_parts, 1.0, "clause")

    candidates = sorted(
        candidate_map.values(),
        key=lambda item: item["semantic_bonus"],
        reverse=True,
    )
    return candidates[:8] or [{"chunks": [stripped], "semantic_bonus": 0.0, "label": "single"}]


def _score_layout_candidate(
    *,
    block_width: int,
    block_height: int,
    box_width: int,
    box_height: int,
    font_size: int,
    layout_shape: str,
    balloon_geo: str = "ellipse",
) -> float:
    width_ratio = block_width / float(max(1, box_width))
    height_ratio = block_height / float(max(1, box_height))

    if balloon_geo == "lobe":
        # Lobe subregion — adapt targets to lobe shape.
        # Wide lobes (horizontal split) can fill more width.
        # Tall/square lobes (diagonal/vertical split) need less width pressure.
        target_width = {"wide": 0.84, "square": 0.78, "tall": 0.72}.get(layout_shape, 0.78)
        target_height = {"wide": 0.75, "square": 0.72, "tall": 0.68}.get(layout_shape, 0.72)
        overflow_w = {"wide": 0.93, "square": 0.90, "tall": 0.88}.get(layout_shape, 0.90)
        overflow_h = 0.90
    elif balloon_geo == "rect":
        # Retangular (narração/sfx) — pode usar mais espaço
        target_width = {"wide": 0.78, "square": 0.72, "tall": 0.60}.get(layout_shape, 0.72)
        target_height = {"wide": 0.40, "square": 0.50, "tall": 0.62}.get(layout_shape, 0.50)
        overflow_w, overflow_h = 0.92, 0.88
    else:
        # Ellipse was too conservative and favored tiny type. Relax targets so
        # simple speech balloons can actually be filled in a human-looking way.
        target_width = {"wide": 0.68, "square": 0.62, "tall": 0.54}.get(layout_shape, 0.62)
        target_height = {"wide": 0.38, "square": 0.46, "tall": 0.56}.get(layout_shape, 0.46)
        overflow_w, overflow_h = 0.84, 0.78

    min_width = {"wide": 0.42, "square": 0.40, "tall": 0.34}.get(layout_shape, 0.40)
    min_height = {"wide": 0.18, "square": 0.22, "tall": 0.30}.get(layout_shape, 0.22)

    score = float(font_size) * 0.08
    score -= abs(width_ratio - target_width) * 12.0
    score -= abs(height_ratio - target_height) * 11.0
    if width_ratio < min_width:
        score -= (min_width - width_ratio) * 18.0
    if height_ratio < min_height:
        score -= (min_height - height_ratio) * 15.0
    if width_ratio > overflow_w:
        score -= (width_ratio - overflow_w) * 40.0
    if height_ratio > overflow_h:
        score -= (height_ratio - overflow_h) * 40.0
    return score


def _fits_in_box(text: str, font_name: str, size: int, max_width: int, max_height: int, line_spacing_ratio: float) -> bool:
    """Verifica se o texto cabe na caixa com o tamanho de fonte dado."""
    font = get_font(font_name, size)
    wrapped = wrap_text(text, font, max_width)
    line_height = get_line_height(font, size, line_spacing_ratio)
    total_height = line_height * len(wrapped)
    line_widths = [measure_text_width(font, line, size) for line in wrapped]
    block_width = max(line_widths, default=0)
    return block_width <= max_width and total_height <= max_height


def _compute_font_search_upper_bound(plan: dict, text: str) -> int:
    """Allow the renderer to grow beyond OCR seed size when the balloon has room.

    The old logic treated target_size as a hard cap, which is the main reason
    small OCR-estimated sizes stayed tiny even inside large clean balloons.
    """
    x1, y1, x2, y2 = plan["target_bbox"]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    seed = int(plan.get("target_size", 16) or 16)
    max_height = int(plan.get("max_height", box_height) or box_height)
    text_len = len(re.sub(r"\s+", "", text or ""))
    geo = plan.get("balloon_geo", "ellipse")

    if geo == "lobe":
        growth = max(12, int(box_height * 0.20))
    elif geo == "ellipse":
        growth = max(10, int(box_height * 0.18))
    else:
        growth = max(8, int(box_height * 0.14))

    if text_len <= 18:
        growth += 12
    elif text_len <= 32:
        growth += 8
    elif text_len <= 50:
        growth += 4

    explicit_cap = int(plan.get("_font_search_cap", 0) or 0)
    hi = max(seed + 4, seed + growth)
    if explicit_cap > 0:
        hi = min(hi, explicit_cap)
    hi = min(hi, max(12, int(box_height * 0.56)))
    hi = min(hi, max(12, max_height))
    hi = min(hi, 96)
    return max(_MIN_FONT_SIZE, hi)


def _resolve_text_layout(text_data: dict, plan: dict) -> dict:
    text = text_data.get("translated", "")
    x1, y1, x2, y2 = plan["target_bbox"]
    px1, py1, px2, py2 = [int(v) for v in plan.get("position_bbox", plan["target_bbox"])]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    position_width = max(1, px2 - px1)
    position_height = max(1, py2 - py1)
    font_size = min(_compute_font_search_upper_bound(plan, text), max(_MIN_FONT_SIZE, box_height - 4))
    best_candidate = None

    # Binary search: achar o maior tamanho que cabe
    lo, hi = _MIN_FONT_SIZE, font_size
    best_fit = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fits_in_box(text, plan["font_name"], mid, plan["max_width"], plan["max_height"], plan["line_spacing_ratio"]):
            best_fit = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Refinar: testar best_fit e vizinhos (±2, ±1, melhor) para scoring
    floor_bound = int(plan.get("_font_search_floor", _MIN_FONT_SIZE) or _MIN_FONT_SIZE)
    candidate_sizes = sorted(
        {
            size
            for size in (
                best_fit + 2,
                best_fit + 1,
                best_fit,
                best_fit - 1,
                best_fit - 2,
            )
            if floor_bound <= size <= font_size
        },
        reverse=True,
    )
    if not candidate_sizes:
        # floor > best_fit means text doesn't fit at the target size; use the
        # largest actually-fitting size rather than falling back to a rigid floor.
        candidate_sizes = [max(_MIN_FONT_SIZE, best_fit)]

    for attempt_size in candidate_sizes:
        font = get_font(plan["font_name"], attempt_size)
        wrapped = wrap_text(text, font, plan["max_width"])
        line_height = get_line_height(font, attempt_size, plan["line_spacing_ratio"])
        total_text_height = line_height * len(wrapped)
        line_widths = [measure_text_width(font, line, attempt_size) for line in wrapped]
        block_width = max(line_widths, default=0)
        if block_width > plan["max_width"] or total_text_height > plan["max_height"]:
            continue
        start_y = (
            py1 + plan["padding_y"]
            if plan["vertical_anchor"] == "top"
            else py1 + max(plan["padding_y"], (position_height - total_text_height) // 2) + int(plan.get("vertical_bias_px", 0) or 0)
        )
        if plan["vertical_anchor"] != "top":
            min_start_y = py1 + int(plan["padding_y"])
            max_start_y = py2 - int(plan["padding_y"]) - total_text_height
            if max_start_y >= min_start_y:
                start_y = min(max(start_y, min_start_y), max_start_y)
            else:
                start_y = min_start_y
        center_x = px1 + (position_width // 2)
        inner_x1 = center_x - (plan["max_width"] // 2)
        inner_x2 = center_x + (plan["max_width"] // 2)
        positions = [
            (
                _line_x(center_x, inner_x1, inner_x2, plan["alignment"], line_width),
                start_y + index * line_height,
            )
            for index, line_width in enumerate(line_widths)
        ]
        if positions:
            block_x1 = min(px for px, _ in positions)
            block_x2 = max(px + width for (px, _), width in zip(positions, line_widths))
            block_y1 = start_y
            block_y2 = start_y + total_text_height
        else:
            block_x1 = center_x
            block_x2 = center_x
            block_y1 = start_y
            block_y2 = start_y

        candidate = {
            "font": font,
            "lines": wrapped,
            "font_size": attempt_size,
            "line_height": line_height,
            "positions": positions,
            "start_y": start_y,
            "total_text_height": total_text_height,
            "line_widths": line_widths,
            "block_bbox": [block_x1, block_y1, block_x2, block_y2],
            "block_width": max(1, block_x2 - block_x1),
            "block_height": max(1, block_y2 - block_y1),
        }
        candidate["width_ratio"] = candidate["block_width"] / float(max(1, box_width))
        candidate["height_ratio"] = candidate["block_height"] / float(max(1, box_height))
        candidate["score"] = _score_layout_candidate(
            block_width=candidate["block_width"],
            block_height=candidate["block_height"],
            box_width=box_width,
            box_height=box_height,
            font_size=attempt_size,
            layout_shape=plan.get("layout_shape", "square"),
            balloon_geo=plan.get("balloon_geo", "ellipse"),
        )
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate

    if best_candidate is not None:
        return best_candidate

    fallback_size = max(_MIN_FONT_SIZE, best_fit)
    fallback_font = get_font(plan["font_name"], fallback_size)
    fallback_lines = wrap_text(text, fallback_font, plan["max_width"])
    fallback_line_height = get_line_height(fallback_font, fallback_size, plan["line_spacing_ratio"])
    fallback_total_height = fallback_line_height * len(fallback_lines)
    start_y = (
        py1 + plan["padding_y"]
        if plan["vertical_anchor"] == "top"
        else py1 + max(plan["padding_y"], (position_height - fallback_total_height) // 2) + int(plan.get("vertical_bias_px", 0) or 0)
    )
    if plan["vertical_anchor"] != "top":
        min_start_y = py1 + int(plan["padding_y"])
        max_start_y = py2 - int(plan["padding_y"]) - fallback_total_height
        if max_start_y >= min_start_y:
            start_y = min(max(start_y, min_start_y), max_start_y)
        else:
            start_y = min_start_y
    center_x = px1 + (position_width // 2)
    fallback_widths = [measure_text_width(fallback_font, line, fallback_size) for line in fallback_lines]
    inner_x1 = center_x - (plan["max_width"] // 2)
    inner_x2 = center_x + (plan["max_width"] // 2)
    positions = [
        (_line_x(center_x, inner_x1, inner_x2, plan["alignment"], width), start_y + index * fallback_line_height)
        for index, width in enumerate(fallback_widths)
    ]
    block_x1 = min((px for px, _ in positions), default=center_x)
    block_x2 = max((px + width for (px, _), width in zip(positions, fallback_widths)), default=center_x)
    return {
        "font": fallback_font,
        "lines": fallback_lines,
        "font_size": fallback_size,
        "line_height": fallback_line_height,
        "positions": positions,
        "start_y": start_y,
        "total_text_height": fallback_total_height,
        "line_widths": fallback_widths,
        "block_bbox": [block_x1, start_y, block_x2, start_y + fallback_total_height],
        "block_width": max(1, block_x2 - block_x1),
        "block_height": max(1, fallback_total_height),
        "width_ratio": max(1, block_x2 - block_x1) / float(max(1, box_width)),
        "height_ratio": max(1, fallback_total_height) / float(max(1, box_height)),
        "score": -9999.0,
    }


def _apply_corpus_layout_hints(
    width_ratio: float,
    tipo: str,
    layout_shape: str,
    corpus_visual: dict,
    corpus_textual: dict,
) -> tuple[float, int, int]:
    visual_geometry = corpus_visual.get("page_geometry", {}) or {}
    paired_text_stats = corpus_textual.get("paired_text_stats", {}) or {}
    textual_ratio = float(paired_text_stats.get("mean_translation_length_ratio", 1.0) or 1.0)
    median_width = int(visual_geometry.get("median_width", 0) or 0)
    median_aspect_ratio = float(visual_geometry.get("median_aspect_ratio", 0.0) or 0.0)

    target_size_delta = 0
    outline_boost = 2 if tipo in {"fala", "pensamento"} else 1
    adjusted_width_ratio = width_ratio

    if textual_ratio >= 1.12 and tipo in {"fala", "narracao", "pensamento"}:
        target_size_delta -= 2
        adjusted_width_ratio -= 0.04

    if median_width and median_width <= 820 and median_aspect_ratio <= 0.34:
        outline_boost = max(outline_boost, 2)
        if layout_shape == "tall":
            adjusted_width_ratio -= 0.03

    return max(0.58, adjusted_width_ratio), target_size_delta, outline_boost


def _line_x(center_x: int, inner_x1: int, inner_x2: int, alignment: str, line_width: int) -> int:
    if alignment == "right":
        return inner_x2 - line_width
    if alignment == "left":
        return inner_x1
    return center_x - (line_width // 2)


def _parse_hex_color(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    if len(h) < 6:
        return (255, 255, 255)
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _color_luminance(color: str) -> float:
    r, g, b = _parse_hex_color(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _sample_background_color(img: Image.Image, bbox: list[int]) -> tuple[int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.width, x2)
    y2 = min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return (255, 255, 255)
    crop = np.array(img.crop((x1, y1, x2, y2)).convert("RGB"))
    if crop.size == 0:
        return (255, 255, 255)
    return tuple(int(v) for v in np.median(crop.reshape(-1, 3), axis=0))


def _contrast_gap(color_a: str, color_b: str) -> float:
    return abs(_color_luminance(color_a) - _color_luminance(color_b))


def ensure_legible_plan(img: Image.Image, plan: dict) -> dict:
    adjusted = dict(plan)
    bg_rgb = _sample_background_color(img, adjusted["target_bbox"])
    bg_hex = "#{:02X}{:02X}{:02X}".format(*bg_rgb)
    bg_luma = _color_luminance(bg_hex)

    text_color = adjusted.get("text_color", "#FFFFFF")
    outline_color = adjusted.get("outline_color", "") or ""
    outline_px = int(adjusted.get("outline_px", 0) or 0)
    glow_color = adjusted.get("glow_cor", "") or ""

    if bg_luma >= 180:
        if _contrast_gap(text_color, bg_hex) < 110:
            adjusted["text_color"] = "#111111"
        if not outline_color or _contrast_gap(outline_color, bg_hex) < 55:
            adjusted["outline_color"] = "#FFFFFF"
        adjusted["outline_px"] = max(2, outline_px)
        if adjusted.get("glow") and (not glow_color or _contrast_gap(glow_color, bg_hex) < 55):
            adjusted["glow"] = False
            adjusted["glow_px"] = 0
            adjusted["glow_cor"] = ""
    elif bg_luma <= 90:
        if _contrast_gap(text_color, bg_hex) < 110:
            adjusted["text_color"] = "#F5F5F5"
        if not outline_color or _contrast_gap(outline_color, bg_hex) < 55:
            adjusted["outline_color"] = "#000000"
        adjusted["outline_px"] = max(2, outline_px)
    else:
        if _contrast_gap(text_color, bg_hex) < 95:
            adjusted["text_color"] = "#111111" if bg_luma > 128 else "#F5F5F5"
        if not outline_color or _contrast_gap(outline_color, bg_hex) < 45:
            adjusted["outline_color"] = "#FFFFFF" if bg_luma < 128 else "#000000"
        adjusted["outline_px"] = max(2, outline_px)

    gradient = adjusted.get("cor_gradiente", []) or []
    if len(gradient) >= 2:
        top_gap = _contrast_gap(gradient[0], bg_hex)
        bottom_gap = _contrast_gap(gradient[1], bg_hex)
        if min(top_gap, bottom_gap) < 70:
            adjusted["cor_gradiente"] = []

    return adjusted


def _resolve_connected_target_sizes(children: list[dict], plans: list[dict]) -> list[int]:
    """Resolve human-looking font sizes for connected balloon lobes.

    Instead of picking the largest size that merely fits, search a narrow band
    of near-uniform sizes and choose the combination that yields a cleaner comic
    layout: fewer stacked lines, healthier occupancy, and only a tiny lobe-to-
    lobe size variation.
    """
    if not children or not plans:
        return []
    raw_resolved = [_resolve_text_layout(child, plan) for child, plan in zip(children, plans)]
    raw_sizes = [int(item["font_size"]) for item in raw_resolved]
    common_floor = max(_MIN_FONT_SIZE, min(raw_sizes) - 10)

    preferred_caps: list[int] = []
    for child, plan, raw_size in zip(children, plans, raw_sizes):
        text = str(child.get("translated", "") or "").strip()
        word_count = len(text.split())
        if word_count >= 14:
            ideal_max_lines = 5
        elif word_count >= 9:
            ideal_max_lines = 4
        else:
            ideal_max_lines = 3

        cap = int(raw_size)
        for size in range(int(raw_size), common_floor - 1, -1):
            fixed_plan = dict(plan)
            fixed_plan["target_size"] = int(size)
            fixed_plan["_font_search_cap"] = int(size)
            fixed_plan["_font_search_floor"] = int(size)
            resolved = _resolve_text_layout(child, fixed_plan)
            line_count = len(resolved.get("lines", []))
            width_ratio = float(resolved.get("width_ratio", 0.0))
            if line_count <= ideal_max_lines and width_ratio <= 0.90:
                cap = int(size)
                break
        preferred_caps.append(cap)

    def _score_child(child: dict, plan: dict, resolved: dict, raw_size: int) -> float:
        text = str(child.get("translated", "") or "").strip()
        word_count = len(text.split())
        line_count = len(resolved.get("lines", []))
        width_ratio = float(resolved.get("width_ratio", 0.0))
        height_ratio = float(resolved.get("height_ratio", 0.0))
        size = int(resolved.get("font_size", 0) or 0)

        if word_count >= 14:
            ideal_min_lines, ideal_max_lines = 4, 5
        elif word_count >= 9:
            ideal_min_lines, ideal_max_lines = 3, 4
        else:
            ideal_min_lines, ideal_max_lines = 2, 3

        score = 0.0
        score -= abs(width_ratio - 0.80) * 10.0
        score -= abs(height_ratio - 0.52) * 13.0
        if line_count < ideal_min_lines:
            score -= (ideal_min_lines - line_count) * 2.4
        if line_count > ideal_max_lines:
            score -= (line_count - ideal_max_lines) * 12.0
            if word_count >= 9:
                score -= 18.0
        if height_ratio > 0.68:
            score -= (height_ratio - 0.68) * 34.0
        if height_ratio < 0.30:
            score -= (0.30 - height_ratio) * 16.0
        if width_ratio < 0.62:
            score -= (0.62 - width_ratio) * 15.0
        if width_ratio > 0.90:
            score -= (width_ratio - 0.90) * 24.0

        lines = resolved.get("lines", [])
        single_word_lines = sum(1 for line in lines if len(line.split()) <= 1)
        short_lines = sum(1 for line in lines if len(line.split()) <= 2)
        score -= single_word_lines * 3.0
        if line_count >= 4:
            score -= max(0, short_lines - 1) * 1.4

        # Allow shrinking away from the raw max-fit size when it produces a
        # cleaner comic shape, but avoid collapsing excessively.
        if size < raw_size:
            score -= (raw_size - size) * 0.10
        return score

    candidate_ranges: list[list[int]] = []
    for raw_size, preferred_cap in zip(raw_sizes, preferred_caps):
        upper = min(int(raw_size), int(preferred_cap) + 1)
        lower = max(common_floor, upper - 10)
        candidate_ranges.append(list(range(upper, lower - 1, -1)))

    best_sizes = list(raw_sizes)
    best_score = float("-inf")
    found_valid_combo = False
    tried: set[tuple[int, ...]] = set()
    for combo in product(*candidate_ranges):
        combo_key = tuple(int(v) for v in combo)
        if combo_key in tried:
            continue
        tried.add(combo_key)
        if max(combo_key) - min(combo_key) > 2:
            continue
        found_valid_combo = True

        combo_score = 0.0
        for child, plan, raw_size, size in zip(children, plans, raw_sizes, combo_key):
            fixed_plan = dict(plan)
            fixed_plan["target_size"] = int(size)
            fixed_plan["_font_search_cap"] = int(size)
            fixed_plan["_font_search_floor"] = int(size)
            resolved = _resolve_text_layout(child, fixed_plan)
            combo_score += _score_child(child, fixed_plan, resolved, raw_size)

        combo_score -= (max(combo_key) - min(combo_key)) * 1.6
        if combo_score > best_score:
            best_score = combo_score
            best_sizes = list(combo_key)

    if not found_valid_combo:
        floor = min(preferred_caps) if preferred_caps else min(raw_sizes)
        best_sizes = [max(floor, min(int(size), floor + 2)) for size in preferred_caps or raw_sizes]

    if len(best_sizes) >= 2 and max(best_sizes) == min(best_sizes):
        word_counts = [len(str(child.get("translated", "") or "").split()) for child in children]
        boost_order = sorted(range(len(best_sizes)), key=lambda idx: (word_counts[idx], idx))
        for idx in boost_order:
            upper_cap = min(int(raw_sizes[idx]), int(preferred_caps[idx]) + 1)
            current_plan = dict(plans[idx])
            current_plan["target_size"] = int(best_sizes[idx])
            current_plan["_font_search_cap"] = int(best_sizes[idx])
            current_plan["_font_search_floor"] = int(best_sizes[idx])
            current_lines = len(_resolve_text_layout(children[idx], current_plan).get("lines", []))
            for extra in (1,):
                candidate = list(best_sizes)
                candidate[idx] = min(upper_cap, candidate[idx] + extra)
                if max(candidate) - min(candidate) > 2:
                    continue
                fixed_plan = dict(plans[idx])
                fixed_plan["target_size"] = int(candidate[idx])
                fixed_plan["_font_search_cap"] = int(candidate[idx])
                fixed_plan["_font_search_floor"] = int(candidate[idx])
                resolved = _resolve_text_layout(children[idx], fixed_plan)
                if len(resolved.get("lines", [])) <= current_lines:
                    best_sizes = candidate
                    break
            if max(best_sizes) != min(best_sizes):
                break

    return best_sizes


def _score_connected_group_candidate(
    resolved_items: list[dict],
    children: list[dict],
    plans: list[dict],
    semantic_bonus: float = 0.0,
) -> float:
    score = sum(float(item.get("score", 0.0)) for item in resolved_items)
    if not resolved_items:
        return score

    sizes = [int(item.get("font_size", 0)) for item in resolved_items]
    score -= (max(sizes) - min(sizes)) * 1.1

    line_counts = [len(item.get("lines", [])) for item in resolved_items]
    if len(line_counts) >= 2:
        score -= abs(line_counts[0] - line_counts[1]) * 0.9

    for item, child, plan in zip(resolved_items, children, plans):
        lines = item.get("lines", [])
        single_word_lines = sum(1 for line in lines if len(line.split()) <= 1)
        width_ratio = float(item.get("width_ratio", 0.0))
        height_ratio = float(item.get("height_ratio", 0.0))
        target_size = int(plan.get("target_size", item.get("font_size", 0)) or 0)
        size = int(item.get("font_size", 0) or 0)
        score -= single_word_lines * 2.4
        if width_ratio < 0.58:
            score -= (0.58 - width_ratio) * 18.0
        if height_ratio < 0.24:
            score -= (0.24 - height_ratio) * 15.0
        if size < max(_MIN_FONT_SIZE, target_size - 8):
            score -= (max(_MIN_FONT_SIZE, target_size - 8) - size) * 0.85
        if len(lines) >= 3 and all(len(line.split()) <= 2 for line in lines):
            score -= 3.0
        if child.get("translated") and re.search(r"[.!?…]$", child.get("translated", "").strip()):
            score += 0.2

        block_bbox = item.get("block_bbox") or []
        position_bbox = plan.get("position_bbox") or []
        if (
            isinstance(block_bbox, (list, tuple))
            and len(block_bbox) == 4
            and isinstance(position_bbox, (list, tuple))
            and len(position_bbox) == 4
        ):
            bx1, by1, bx2, by2 = [int(v) for v in block_bbox]
            px1, py1, px2, py2 = [int(v) for v in position_bbox]
            block_cx = (bx1 + bx2) / 2.0
            block_cy = (by1 + by2) / 2.0
            pos_cx = (px1 + px2) / 2.0
            pos_cy = (py1 + py2) / 2.0
            pos_w = max(1.0, float(px2 - px1))
            pos_h = max(1.0, float(py2 - py1))
            drift_x = abs(block_cx - pos_cx) / pos_w
            drift_y = abs(block_cy - pos_cy) / pos_h
            score -= drift_x * 6.0
            score -= drift_y * 8.0

    for child in children[:-1]:
        boundary_text = str(child.get("translated", "") or "").strip()
        if not boundary_text:
            score -= 2.0
            continue
        if re.search(r"[.!?â€¦]$", boundary_text):
            score += 4.2
        elif re.search(r"[,;:]$", boundary_text):
            score += 1.2
        else:
            score -= 3.8

    score += semantic_bonus
    return score


def _build_connected_children_candidates(
    text_data: dict,
    text: str,
    subregions: list[list[int]],
) -> list[dict]:
    orientation = _infer_connected_orientation_from_subregions(
        subregions,
        str(text_data.get("connected_balloon_orientation", "") or ""),
    )
    ordered_subregions = _order_connected_subregions(
        subregions,
        orientation,
    )
    connected_children = text_data.get("connected_children") or []
    if connected_children and len(connected_children) == len(ordered_subregions):
        children = []
        for index, (subregion, source_child) in enumerate(zip(ordered_subregions, connected_children)):
            child = dict(source_child)
            child["bbox"] = list(subregion)
            child["balloon_bbox"] = list(subregion)
            child["balloon_subregions"] = []
            child["layout_group_size"] = 1
            child["layout_shape"] = _infer_layout_shape_from_bbox(subregion, child.get("tipo", "fala"))
            child["layout_align"] = "top" if child.get("tipo") == "narracao" else "center"
            child["_is_lobe_subregion"] = True
            child["_connected_slot_index"] = index
            child["_connected_slot_count"] = len(ordered_subregions)
            child["connected_balloon_orientation"] = orientation
            if not float(child.get("_connected_vertical_bias_ratio", 0.0) or 0.0):
                child["_connected_vertical_bias_ratio"] = _default_connected_vertical_bias_ratio(
                    index,
                    len(ordered_subregions),
                    orientation,
                )
            children.append(child)
        return [{"children": children, "semantic_bonus": 1.5, "label": "assigned"}]

    area_weights = _resolve_connected_area_weights(text_data, ordered_subregions)
    candidates = []
    for option in _enumerate_connected_text_candidates(text, len(ordered_subregions), area_weights):
        children = []
        for index, (chunk, subregion) in enumerate(zip(option["chunks"], ordered_subregions)):
            child = dict(text_data)
            child["translated"] = chunk
            child["bbox"] = list(subregion)
            child["balloon_bbox"] = list(subregion)
            child["balloon_subregions"] = []
            child["layout_group_size"] = 1
            child["layout_shape"] = _infer_layout_shape_from_bbox(subregion, child.get("tipo", "fala"))
            child["layout_align"] = "top" if child.get("tipo") == "narracao" else "center"
            child["_is_lobe_subregion"] = True
            child["_connected_slot_index"] = index
            child["_connected_slot_count"] = len(ordered_subregions)
            child["connected_balloon_orientation"] = orientation
            child["_connected_vertical_bias_ratio"] = _default_connected_vertical_bias_ratio(
                index,
                len(ordered_subregions),
                orientation,
            )
            children.append(child)
        candidates.append(
            {
                "children": children,
                "semantic_bonus": float(option.get("semantic_bonus", 0.0)),
                "label": option.get("label", "unknown"),
            },
        )
    return candidates


def _render_connected_subregions(
    img: Image.Image,
    text_data: dict,
    text: str,
    subregions: list[list[int]],
) -> None:
    candidates = _build_connected_children_candidates(text_data, text, subregions)
    if not candidates:
        child = dict(text_data)
        child["balloon_subregions"] = []
        render_text_block(img, child)
        return

    best_candidate = None
    best_score = float("-inf")

    for candidate in candidates:
        children = [dict(child) for child in candidate.get("children", []) if child.get("translated", "").strip()]
        if len(children) != len(subregions):
            continue
        for child in children:
            estilo = child.get("estilo", {})
            if estilo.get("force_upper") or estilo.get("fonte") == "CCDaveGibbonsLower W00 Regular.ttf":
                child["translated"] = child.get("translated", "").upper()

        plans = [ensure_legible_plan(img, plan_text_layout(child)) for child in children]
        target_sizes = _resolve_connected_target_sizes(children, plans)
        resolved_items = []
        final_plans = []
        for child, plan, target_size in zip(children, plans, target_sizes):
            fixed_plan = dict(plan)
            fixed_plan["target_size"] = max(_MIN_FONT_SIZE, int(target_size))
            fixed_plan["_font_search_cap"] = max(_MIN_FONT_SIZE, int(target_size))
            fixed_plan["_font_search_floor"] = max(_MIN_FONT_SIZE, int(target_size))
            fixed_plan["outline_px"] = max(fixed_plan["outline_px"], 2 if target_size <= 22 else 3)
            resolved_items.append(_resolve_text_layout(child, fixed_plan))
            final_plans.append(fixed_plan)

        group_score = _score_connected_group_candidate(
            resolved_items,
            children,
            final_plans,
            semantic_bonus=float(candidate.get("semantic_bonus", 0.0)),
        )
        if group_score > best_score:
            best_score = group_score
            best_candidate = {
                "children": children,
                "plans": final_plans,
            }

    if not best_candidate:
        child = dict(text_data)
        child["balloon_subregions"] = []
        render_text_block(img, child)
        return

    for child, plan in zip(best_candidate["children"], best_candidate["plans"]):
        _render_single_text_block(img, child, plan)


def _render_single_text_block(
    img: Image.Image, text_data: dict, plan: dict,
) -> None:
    """Core rendering logic for a single text block (no subregion recursion)."""
    text = text_data.get("translated", "")
    if not text:
        return

    x1, y1, x2, y2 = plan["target_bbox"]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    if box_width < 10 or box_height < 10:
        return

    resolved = _resolve_text_layout(text_data, plan)
    best_font = resolved["font"]
    best_lines = resolved["lines"]
    best_size = resolved["font_size"]
    line_height = resolved["line_height"]
    total_text_height = resolved["total_text_height"]
    start_y = resolved["start_y"]
    positions = resolved["positions"]

    outline_color = plan["outline_color"]
    outline_px = int(plan["outline_px"])

    if isinstance(best_font, SafeTextPathFont):
        image_np = np.array(img)
        px1, py1, px2, py2 = [int(v) for v in plan.get("position_bbox", plan["target_bbox"])]
        position_width = max(1, px2 - px1)
        center_x = px1 + (position_width // 2)
        inner_x1 = center_x - (plan["max_width"] // 2)
        inner_x2 = center_x + (plan["max_width"] // 2)

        corrected_positions = []
        for line, (_, ly) in zip(best_lines, positions):
            mask = _build_textpath_mask(best_font, line, padding=0)
            real_width = mask.shape[1] if mask.size > 1 else measure_text_width(best_font, line, best_size)
            new_lx = _line_x(center_x, inner_x1, inner_x2, plan["alignment"], real_width)
            corrected_positions.append((new_lx, ly))
        positions = _recenter_safe_text_positions(
            best_font,
            best_lines,
            corrected_positions,
            target_bbox=plan.get("position_bbox", plan["target_bbox"]),
            padding_y=int(plan["padding_y"]),
            vertical_anchor=str(plan["vertical_anchor"]),
            vertical_bias_px=int(plan.get("vertical_bias_px", 0) or 0),
        )

        if plan["sombra"] and plan["sombra_cor"]:
            dx, dy = plan["sombra_offset"]
            shadow_positions = [(lx + int(dx), ly + int(dy)) for lx, ly in positions]
            _render_safe_text_layer(
                image_np, best_lines, best_font, shadow_positions,
                fill_color=plan["sombra_cor"],
            )

        if plan["glow"] and plan["glow_cor"] and int(plan["glow_px"]) > 0:
            _apply_safe_glow(
                image_np, best_lines, best_font, positions,
                plan["glow_cor"], int(plan["glow_px"]),
            )

        gradient = plan["cor_gradiente"]
        if gradient and len(gradient) >= 2:
            _apply_safe_gradient_text(
                image_np, best_lines, best_font, positions,
                gradient[0], gradient[1],
                outline_color, outline_px,
                start_y, total_text_height,
            )
        else:
            _render_safe_text_layer(
                image_np, best_lines, best_font, positions,
                fill_color=plan["text_color"],
                outline_color=outline_color,
                outline_px=outline_px,
            )
        img.paste(Image.fromarray(image_np))
        return

    # PIL path (system fallback fonts)
    if plan["sombra"] and plan["sombra_cor"]:
        draw = ImageDraw.Draw(img)
        dx, dy = plan["sombra_offset"]
        for line, (lx, ly) in zip(best_lines, positions):
            draw.text((lx + dx, ly + dy), line, font=best_font, fill=plan["sombra_cor"])

    if plan["glow"] and plan["glow_cor"] and plan["glow_px"] > 0:
        _apply_glow(img, best_lines, best_font, positions, plan["glow_cor"], plan["glow_px"])

    if outline_color and outline_px > 0:
        draw = ImageDraw.Draw(img)
        for line, (lx, ly) in zip(best_lines, positions):
            for dx in range(-outline_px, outline_px + 1):
                for dy in range(-outline_px, outline_px + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((lx + dx, ly + dy), line, font=best_font, fill=outline_color)

    gradient = plan["cor_gradiente"]
    if gradient and len(gradient) >= 2:
        _apply_gradient_text(
            img, best_lines, best_font, positions,
            gradient[0], gradient[1],
            outline_color, outline_px,
            start_y, total_text_height,
        )
    else:
        draw = ImageDraw.Draw(img)
        for line, (lx, ly) in zip(best_lines, positions):
            draw.text((lx, ly), line, font=best_font, fill=plan["text_color"])


def _apply_glow(
    img: Image.Image,
    lines: list,
    font: ImageFont.FreeTypeFont,
    positions: list,
    glow_color: str,
    glow_px: int,
) -> None:
    """Render a soft Gaussian-blurred halo behind text."""
    gc = _parse_hex_color(glow_color) + (210,)
    glow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    for line, (lx, ly) in zip(lines, positions):
        glow_draw.text((lx, ly), line, font=font, fill=gc)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=max(1, glow_px)))
    r, g, b, a = glow_layer.split()
    img.paste(Image.merge("RGB", (r, g, b)), (0, 0), a)


def _apply_gradient_text(
    img: Image.Image,
    lines: list,
    font: ImageFont.FreeTypeFont,
    positions: list,
    color_top: str,
    color_bottom: str,
    outline_color: str,
    outline_px: int,
    start_y: int,
    total_height: int,
) -> None:
    """Render text with a vertical gradient fill on top of already-drawn outlines."""
    ct = np.array(_parse_hex_color(color_top), dtype=float)
    cb = np.array(_parse_hex_color(color_bottom), dtype=float)

    for line, (lx, ly) in zip(lines, positions):
        try:
            tbbox = font.getbbox(line)
            tw = tbbox[2] - tbbox[0]
            th = tbbox[3] - tbbox[1]
        except Exception:
            continue
        if tw <= 0 or th <= 0:
            continue

        pad = 2
        lw, lh = tw + pad * 2, th + pad * 2

        # Text mask
        mask = Image.new("L", (lw, lh), 0)
        ImageDraw.Draw(mask).text((pad, pad), line, font=font, fill=255)

        # Gradient strip mapped to global vertical position
        gradient = np.zeros((lh, lw, 3), dtype=np.uint8)
        for y in range(lh):
            global_y = (ly + y - pad) - start_y
            t = float(np.clip(global_y / max(1, total_height), 0.0, 1.0))
            color = (ct * (1.0 - t) + cb * t).clip(0, 255).astype(np.uint8)
            gradient[y, :] = color

        img.paste(Image.fromarray(gradient, "RGB"), (lx - pad, ly - pad), mask)


def render_text_block(img: Image.Image, text_data: dict, img_size: tuple = None):
    del img_size
    text = text_data.get("translated", "")
    if not text:
        return
    text = _normalize_render_text(text)

    subregions = [
        [int(v) for v in bbox]
        for bbox in text_data.get("balloon_subregions", []) or []
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
    ]
    if len(subregions) >= 2 and _has_reliable_connected_layout(text_data, subregions):
        text_data = dict(text_data)
        text_data["translated"] = text
        _render_connected_subregions(img, text_data, text, subregions)
        return

    estilo = text_data.get("estilo", {})
    if estilo.get("force_upper") or estilo.get("fonte") == "CCDaveGibbonsLower W00 Regular.ttf":
        text = text.upper()
        text_data = dict(text_data)
        text_data["translated"] = text

    plan = ensure_legible_plan(img, plan_text_layout(text_data))
    _render_single_text_block(img, text_data, plan)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if text is None:
        return [""]

    raw_lines = str(text).split("\n")
    if not raw_lines:
        return [""]

    if len(raw_lines) > 1:
        preserved = [re.sub(r"[ \t]+", " ", raw_line).strip() for raw_line in raw_lines]
        return [line for line in preserved if line] or [""]

    wrapped: list[str] = []
    for raw_line in raw_lines:
        normalized = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not normalized:
            continue

        words = normalized.split()
        if not words:
            continue

        lines: list[str] = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if current_line and measure_text_width(font, test_line) > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line

        if current_line:
            lines.append(current_line)

        balanced = _rebalance_wrapped_lines(lines or [normalized], font, max_width)
        wrapped.extend(balanced or [normalized])

    return wrapped or [str(text).strip()]


def get_line_height(font: ImageFont.FreeTypeFont, font_size: int, spacing_ratio: float) -> int:
    try:
        base = font.getbbox("Ay")[3]
    except Exception:
        base = font_size
    return int(base + max(2, font_size * spacing_ratio))


def measure_text_width(font: ImageFont.FreeTypeFont, text: str, fallback_size: int = 16) -> int:
    try:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]
    except Exception:
        return int(len(text) * fallback_size * 0.6)
