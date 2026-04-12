"""
Typesetting module - renders translated text onto inpainted manga pages.
Now uses inferred balloon/layout geometry instead of relying only on the raw
OCR bounding box.
"""

from __future__ import annotations

import math
import re
import sys
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
    "fala":      "CCDaveGibbonsLower W00 Regular.ttf",
    "narracao":  "CCDaveGibbonsLower W00 Regular.ttf",
    "sfx":       "CCDaveGibbonsLower W00 Regular.ttf",
    "pensamento": "CCDaveGibbonsLower W00 Regular.ttf",
}

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
        # Estimativa sem FreeType — evita centenas de chamadas TextPath durante binary search
        w = max(1, int(len(text) * self.size * 0.55))
        h = max(1, int(self.size * 1.15))
        bbox = (0, 0, w, h)
        self._bbox_cache[text] = bbox
        return bbox


_FALLBACK_FONTS = [
    "ComicNeue-Bold.ttf",
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


def _typeset_single_page(args: tuple) -> int:
    """Renderiza uma única página — projetada para rodar em worker process."""
    img_path_str, trans_page, output_dir_str = args
    img_path = Path(img_path_str)
    output_path = Path(output_dir_str)

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
    if best_ratio >= 0.60 and (best_ratio - second_ratio) >= 0.18:
        return best_bbox
    if best_inside and best_ratio >= 0.42 and (best_ratio - second_ratio) >= 0.15:
        return best_bbox
    return None


def build_render_blocks(texts: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, tuple[int, int, int, int]], list[dict]] = {}
    passthrough: list[dict] = []

    for text in texts:
        balloon_bbox = text.get("balloon_bbox")
        tipo = text.get("tipo", "fala")

        subregions = _normalize_balloon_subregions(text.get("balloon_subregions", []))
        if balloon_bbox and len(subregions) >= 2:
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
        combined = dict(ordered[0])
        combined.pop("_resolved_subregion", None)
        combined["balloon_bbox"] = list(bbox_tuple)
        combined["translated"] = "\n".join(
            text.get("translated", "").strip()
            for text in ordered
            if text.get("translated", "").strip()
        )
        combined["estilo"] = merge_group_style(ordered)
        combined["source_text_count"] = len(ordered)
        combined["layout_group_size"] = len(ordered)
        if combined.get("balloon_subregions") and all(text.get("_resolved_subregion") for text in ordered):
            combined["balloon_subregions"] = []
        blocks.append(combined)

    return blocks


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
    # Balões texturizados usam Newrotic → retangular
    estilo = text_data.get("estilo", {})
    fonte = estilo.get("fonte", "")
    if fonte and fonte != "ComicNeue-Bold.ttf":
        return "rect"
    return "ellipse"


def plan_text_layout(text_data: dict) -> dict:
    target_bbox = text_data.get("balloon_bbox") or text_data.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = target_bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

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
        # Elíptico (fala/pensamento) — inscreve retângulo na elipse (~1/√2 ≈ 0.71)
        if layout_shape == "tall":
            width_ratio = 0.65
            padding_y = max(8, int(box_height * 0.15))
        elif layout_shape == "wide":
            width_ratio = 0.75
            padding_y = max(8, int(box_height * 0.18))
        else:
            width_ratio = 0.70
            padding_y = max(8, int(box_height * 0.16))
        vertical_anchor = "center"
        line_spacing = 0.1
    else:
        # Retangular texturizado — margem de segurança para não ultrapassar
        width_ratio = 0.72
        vertical_anchor = "center"
        padding_y = max(6, int(box_height * 0.10))
        line_spacing = 0.1

    if group_size > 1 and tipo == "fala":
        width_ratio -= 0.04

    width_ratio, target_size_delta, outline_boost = _apply_corpus_layout_hints(
        width_ratio=width_ratio,
        tipo=tipo,
        layout_shape=layout_shape,
        corpus_visual=corpus_visual,
        corpus_textual=corpus_textual,
    )

    target_size = max(10, int(estilo.get("tamanho", 16)) + target_size_delta)
    outline_px = max(int(estilo.get("contorno_px", 2)), outline_boost)
    return {
        "target_bbox": target_bbox,
        "layout_shape": layout_shape,
        "balloon_geo": balloon_geo,
        "max_width": max(40, int(box_width * width_ratio)),
        "max_height": max(20, box_height - (padding_y * 2)),
        "padding_y": padding_y,
        "vertical_anchor": vertical_anchor if layout_align != "top" else "top",
        "alignment": estilo.get("alinhamento", "center"),
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


def _split_text_for_connected_balloons(text: str, count: int) -> list[str]:
    stripped = text.strip()
    if count <= 1 or not stripped:
        return [stripped]

    newline_parts = [part.strip() for part in re.split(r"\n+", stripped) if part.strip()]
    if len(newline_parts) >= count:
        return _merge_chunks_to_target_count(newline_parts, count)

    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?…])\s+", stripped)
        if part.strip()
    ]
    if len(sentence_parts) >= count:
        return _merge_chunks_to_target_count(sentence_parts, count)

    return _split_words_evenly(stripped, count)


def _merge_chunks_to_target_count(chunks: list[str], count: int) -> list[str]:
    if len(chunks) <= count:
        return chunks[:count]

    merged = [chunk.strip() for chunk in chunks if chunk.strip()]
    while len(merged) > count:
        smallest_index = min(range(len(merged) - 1), key=lambda idx: len(merged[idx].split()))
        merged[smallest_index] = f"{merged[smallest_index]} {merged.pop(smallest_index + 1)}".strip()
    return merged


def _split_words_evenly(text: str, count: int) -> list[str]:
    words = text.split()
    if not words:
        return [text.strip()]

    total_words = len(words)
    base = total_words // count
    remainder = total_words % count
    chunks = []
    cursor = 0
    for idx in range(count):
        take = base + (1 if idx < remainder else 0)
        if take <= 0:
            continue
        chunk_words = words[cursor:cursor + take]
        cursor += take
        chunks.append(" ".join(chunk_words).strip())
    return [chunk for chunk in chunks if chunk]


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

    if balloon_geo == "rect":
        # Retangular (narração/sfx) — pode usar mais espaço
        target_width = {"wide": 0.78, "square": 0.72, "tall": 0.60}.get(layout_shape, 0.72)
        target_height = {"wide": 0.40, "square": 0.50, "tall": 0.62}.get(layout_shape, 0.50)
        overflow_w, overflow_h = 0.92, 0.88
    else:
        # Elíptico (fala/pensamento) — área inscrita menor
        target_width = {"wide": 0.62, "square": 0.56, "tall": 0.48}.get(layout_shape, 0.56)
        target_height = {"wide": 0.32, "square": 0.42, "tall": 0.54}.get(layout_shape, 0.42)
        overflow_w, overflow_h = 0.78, 0.72

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


def _resolve_text_layout(text_data: dict, plan: dict) -> dict:
    text = text_data.get("translated", "")
    x1, y1, x2, y2 = plan["target_bbox"]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    font_size = min(plan["target_size"], max(8, box_height - 4))
    best_candidate = None

    # Binary search: achar o maior tamanho que cabe
    lo, hi = 8, font_size
    best_fit = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fits_in_box(text, plan["font_name"], mid, plan["max_width"], plan["max_height"], plan["line_spacing_ratio"]):
            best_fit = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Refinar: testar best_fit e vizinhos (±1) para scoring
    candidate_sizes = sorted(
        {size for size in (best_fit + 1, best_fit, best_fit - 1) if 8 <= size <= font_size},
        reverse=True,
    )
    if not candidate_sizes:
        candidate_sizes = [8]

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
            y1 + plan["padding_y"]
            if plan["vertical_anchor"] == "top"
            else y1 + max(plan["padding_y"], (box_height - total_text_height) // 2)
        )
        center_x = x1 + (box_width // 2)
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

    fallback_size = 8
    fallback_font = get_font(plan["font_name"], fallback_size)
    fallback_lines = wrap_text(text, fallback_font, plan["max_width"])
    fallback_line_height = get_line_height(fallback_font, fallback_size, plan["line_spacing_ratio"])
    fallback_total_height = fallback_line_height * len(fallback_lines)
    start_y = (
        y1 + plan["padding_y"]
        if plan["vertical_anchor"] == "top"
        else y1 + max(plan["padding_y"], (box_height - fallback_total_height) // 2)
    )
    center_x = x1 + (box_width // 2)
    fallback_widths = [measure_text_width(fallback_font, line, fallback_size) for line in fallback_lines]
    positions = [
        (center_x - (width // 2), start_y + index * fallback_line_height)
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

    subregions = [
        [int(v) for v in bbox]
        for bbox in text_data.get("balloon_subregions", []) or []
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
    ]
    if len(subregions) >= 2:
        chunks = _split_text_for_connected_balloons(text, len(subregions))
        if len(chunks) == len(subregions):
            for chunk, subregion in zip(chunks, subregions):
                child = dict(text_data)
                child["translated"] = chunk
                child["bbox"] = subregion
                child["balloon_bbox"] = subregion
                child["balloon_subregions"] = []
                child["layout_group_size"] = 1
                child["layout_shape"] = _infer_layout_shape_from_bbox(subregion, child.get("tipo", "fala"))
                child["layout_align"] = "top" if child.get("tipo") == "narracao" else "center"
                render_text_block(img, child)
            return

    estilo = text_data.get("estilo", {})
    if estilo.get("force_upper") or estilo.get("fonte") == "CCDaveGibbonsLower W00 Regular.ttf":
        text = text.upper()
        text_data = dict(text_data)
        text_data["translated"] = text

    plan = ensure_legible_plan(img, plan_text_layout(text_data))
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
        center_x = x1 + (box_width // 2)

        # Re-centralizar linhas baseado na largura real do mask (não da estimativa)
        corrected_positions = []
        for line, (lx, ly) in zip(best_lines, positions):
            mask = _build_textpath_mask(best_font, line, padding=0)
            real_width = mask.shape[1] if mask.size > 1 else 0
            new_lx = center_x - (real_width // 2)
            corrected_positions.append((new_lx, ly))
        positions = corrected_positions

        # Re-centralizar verticalmente com altura real
        total_real_height = line_height * len(best_lines)
        if plan["vertical_anchor"] != "top" and positions:
            current_top = positions[0][1]
            ideal_top = y1 + max(plan["padding_y"], (box_height - total_real_height) // 2)
            if current_top != ideal_top:
                dy = ideal_top - current_top
                positions = [(lx, ly + dy) for lx, ly in positions]

        if plan["sombra"] and plan["sombra_cor"]:
            dx, dy = plan["sombra_offset"]
            shadow_positions = [(lx + int(dx), ly + int(dy)) for lx, ly in positions]
            _render_safe_text_layer(
                image_np,
                best_lines,
                best_font,
                shadow_positions,
                fill_color=plan["sombra_cor"],
            )

        if plan["glow"] and plan["glow_cor"] and int(plan["glow_px"]) > 0:
            _apply_safe_glow(
                image_np,
                best_lines,
                best_font,
                positions,
                plan["glow_cor"],
                int(plan["glow_px"]),
            )

        gradient = plan["cor_gradiente"]
        if gradient and len(gradient) >= 2:
            _apply_safe_gradient_text(
                image_np,
                best_lines,
                best_font,
                positions,
                gradient[0],
                gradient[1],
                outline_color,
                outline_px,
                start_y,
                total_text_height,
            )
        else:
            _render_safe_text_layer(
                image_np,
                best_lines,
                best_font,
                positions,
                fill_color=plan["text_color"],
                outline_color=outline_color,
                outline_px=outline_px,
            )
        img.paste(Image.fromarray(image_np))
        return

    # ── 1. Shadow ─────────────────────────────────────────────────────────
    if plan["sombra"] and plan["sombra_cor"]:
        draw = ImageDraw.Draw(img)
        dx, dy = plan["sombra_offset"]
        for line, (lx, ly) in zip(best_lines, positions):
            draw.text((lx + dx, ly + dy), line, font=best_font, fill=plan["sombra_cor"])

    # ── 2. Glow ───────────────────────────────────────────────────────────
    if plan["glow"] and plan["glow_cor"] and plan["glow_px"] > 0:
        _apply_glow(img, best_lines, best_font, positions, plan["glow_cor"], plan["glow_px"])

    # ── 3. Outline ────────────────────────────────────────────────────────
    if outline_color and outline_px > 0:
        draw = ImageDraw.Draw(img)
        for line, (lx, ly) in zip(best_lines, positions):
            for dx in range(-outline_px, outline_px + 1):
                for dy in range(-outline_px, outline_px + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((lx + dx, ly + dy), line, font=best_font, fill=outline_color)

    # ── 4. Text fill (gradient or solid) ──────────────────────────────────
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


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]

    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if measure_text_width(font, test_line) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)
    return lines or [text]


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
