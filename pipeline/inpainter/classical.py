"""
Smart inpainting backend.

Pipeline por região:
  1. classify_background() → detecta solid_light / solid_dark / gradient / textured
  2. apply_fill()          → preenche com cor exata ou textura copiada (SEM blur)
  3. is_natural()          → verifica continuidade de cor na borda preenchida
  4. feather_boundary()    → blur fino APENAS na borda externa, só se necessário

Regra central: nunca borre antes de copiar o fundo.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .mask_builder import build_mask_regions, build_region_pixel_mask


# ── Ponto de entrada ──────────────────────────────────────────────────────

def run_classical_inpainting(
    image_files: list[Path],
    ocr_results: list[dict],
    output_dir: str,
    corpus_visual_benchmark: dict | None = None,
    progress_callback: Callable | None = None,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    inpainted_paths = []
    total = len(image_files)

    for index, (img_path, ocr_data) in enumerate(zip(image_files, ocr_results)):
        texts = ocr_data.get("texts", [])
        dest = output_path / img_path.name

        if not texts:
            if dest != img_path:
                shutil.copy2(img_path, dest)
            inpainted_paths.append(dest)
        else:
            img = Image.open(img_path).convert("RGB")
            cleaned = clean_image(img, texts, corpus_visual_benchmark=corpus_visual_benchmark)
            cleaned.save(dest, quality=95)
            inpainted_paths.append(dest)

        if progress_callback:
            progress_callback(index + 1, total, f"Inpainting pagina {index + 1}/{total}")

    return inpainted_paths


# ── Pipeline principal ────────────────────────────────────────────────────

def clean_image(
    img: Image.Image,
    texts: list[dict],
    corpus_visual_benchmark: dict | None = None,
) -> Image.Image:
    img_array = np.array(img)
    regions = build_mask_regions(texts=texts, image_shape=img_array.shape)
    corpus_profile = build_corpus_inpainting_profile(corpus_visual_benchmark or {})

    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        if x2 <= x1 or y2 <= y1:
            continue

        white_balloon = detect_white_balloon_overlay(img_array, region)
        if white_balloon is not None:
            img_array = apply_fill(
                img_array,
                np.zeros(img_array.shape[:2], dtype=np.uint8),
                region["bbox"],
                "balloon_white_overlay",
                white_balloon,
            )
            continue

        mask = build_region_pixel_mask(img_array.shape[:2], region)
        if not np.any(mask):
            continue

        before = img_array.copy()

        # 1. Classificar tipo de fundo
        bg_type, bg_meta = classify_background(img_array, region["bbox"], mask, corpus_profile=corpus_profile)

        textured_overlay = detect_textured_overlay(img_array, region, mask, bg_type, bg_meta)
        if textured_overlay is not None:
            img_array = apply_fill(
                img_array,
                mask,
                region["bbox"],
                "textured_overlay",
                textured_overlay,
            )
            continue

        # 2. Preencher SEM blur
        img_array = apply_fill(img_array, mask, region["bbox"], bg_type, bg_meta)

        # 3. Verificar naturalidade; blur fino só na borda se necessário
        if not is_natural(img_array, before, mask, threshold=corpus_profile["naturality_threshold"]):
            img_array = feather_boundary(img_array, mask, feather_radius=corpus_profile["feather_radius"])

    return Image.fromarray(img_array)


# ── Classificação de fundo ────────────────────────────────────────────────

def classify_background(
    img_array: np.ndarray,
    bbox: list[int],
    mask: np.ndarray,
    corpus_profile: dict | None = None,
) -> tuple[str, dict]:
    """
    Retorna (tipo, meta) onde tipo é um de:
      solid_light, solid_dark, solid_mid, gradient, textured
    """
    profile = corpus_profile or build_corpus_inpainting_profile({})
    ring = _sample_outer_ring(img_array, bbox, ring_width=profile["ring_width"])
    if len(ring) == 0:
        return "solid_light", {"color": (255, 255, 255)}

    gray = np.mean(ring, axis=1)
    mean_br = float(np.mean(gray))
    std_br = float(np.std(gray))

    dominant = tuple(int(c) for c in np.median(ring, axis=0))
    linear_texture = _detect_linear_texture(img_array, bbox, mask)

    # Verificar gradiente vertical: compara metade superior vs inferior
    x1, y1, x2, y2 = bbox
    mid_y = (y1 + y2) // 2
    top_ring = _sample_outer_ring(img_array, [x1, y1, x2, mid_y], ring_width=10)
    bot_ring = _sample_outer_ring(img_array, [x1, mid_y, x2, y2], ring_width=10)
    vertical_diff = 0.0
    if len(top_ring) > 0 and len(bot_ring) > 0:
        vertical_diff = float(
            np.linalg.norm(np.mean(top_ring, axis=0) - np.mean(bot_ring, axis=0))
        )

    if linear_texture:
        return linear_texture, {"hint_color": dominant}

    if std_br < 20:
        if mean_br > 175:
            return "solid_light", {"color": dominant}
        elif mean_br < 80:
            return "solid_dark", {"color": dominant}
        else:
            return "solid_mid", {"color": dominant}

    if vertical_diff > 28 and std_br < 55:
        top_c = tuple(int(c) for c in np.median(top_ring, axis=0)) if len(top_ring) > 0 else dominant
        bot_c = tuple(int(c) for c in np.median(bot_ring, axis=0)) if len(bot_ring) > 0 else dominant
        return "gradient", {"top": top_c, "bottom": bot_c}

    if std_br < 45:
        return "solid_mid", {"color": dominant}

    return "textured", {"hint_color": dominant}


def build_corpus_inpainting_profile(corpus_visual_benchmark: dict) -> dict:
    geometry = (corpus_visual_benchmark or {}).get("page_geometry", {}) or {}
    luminance = (corpus_visual_benchmark or {}).get("luminance_profile", {}) or {}
    median_width = int(geometry.get("median_width", 800) or 800)
    ring_width = 14 if median_width >= 900 else 16

    dark_pages = int(luminance.get("dark_pages", 0) or 0)
    mid_pages = int(luminance.get("mid_pages", 0) or 0)
    light_pages = int(luminance.get("light_pages", 0) or 0)
    if (dark_pages + mid_pages) > light_pages:
        naturality_threshold = 18.0
        feather_radius = 2
    else:
        naturality_threshold = 20.0
        feather_radius = 1

    return {
        "ring_width": ring_width,
        "naturality_threshold": naturality_threshold,
        "feather_radius": feather_radius,
    }


def detect_white_balloon_overlay(
    img_array: np.ndarray,
    region: dict,
) -> dict | None:
    if region.get("tipo") not in {"fala", "narracao"}:
        return None

    bbox = region.get("bbox")
    if not bbox:
        return None

    balloon_mask = _extract_white_balloon_mask(img_array, bbox)
    if balloon_mask is None:
        return None

    overlay_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    height, width = img_array.shape[:2]
    for text in region.get("texts", []):
        text_bbox = text.get("bbox")
        if not text_bbox:
            continue
        ox1, oy1, ox2, oy2 = _expand_overlay_bbox(
            text_bbox,
            image_width=width,
            image_height=height,
            confidence=float(text.get("confidence", 0.0)),
        )
        cv2.rectangle(overlay_mask, (ox1, oy1), (ox2, oy2), 255, thickness=-1)

    overlay_mask = cv2.bitwise_and(overlay_mask, balloon_mask)
    if not np.any(overlay_mask):
        return None

    pixels = img_array[balloon_mask > 0]
    if len(pixels) == 0:
        return None

    color = tuple(int(c) for c in np.median(pixels, axis=0))
    if np.mean(color) < 220:
        return None

    return {
        "color": color,
        "overlay_mask": overlay_mask,
    }


def _extract_white_balloon_mask(
    img_array: np.ndarray,
    bbox: list[int],
) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h, w = img_array.shape[:2]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = max(20, int(box_w * 0.9))
    pad_y = max(20, int(box_h * 1.0))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(w, x2 + pad_x)
    ry2 = min(h, y2 + pad_y)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi = img_array[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 215, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    inner_rect = np.zeros_like(thresh, dtype=np.uint8)
    ix1 = max(0, x1 - rx1)
    iy1 = max(0, y1 - ry1)
    ix2 = min(rx2 - rx1, x2 - rx1)
    iy2 = min(ry2 - ry1, y2 - ry1)
    inner_rect[iy1:iy2, ix1:ix2] = 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    best_label = 0
    best_overlap = 0
    bbox_area = max(1, (x2 - x1) * (y2 - y1))

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < bbox_area * 1.4 or area > bbox_area * 30:
            continue

        component = (labels == label).astype(np.uint8) * 255
        overlap = int(np.count_nonzero((component > 0) & (inner_rect > 0)))
        if overlap <= best_overlap:
            continue

        pixels = roi[component > 0]
        if len(pixels) == 0:
            continue
        if float(np.mean(np.mean(pixels, axis=1))) < 225:
            continue

        best_label = label
        best_overlap = overlap

    if best_label == 0:
        return None

    component = (labels == best_label).astype(np.uint8) * 255
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = component
    return full_mask


def _expand_overlay_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    confidence: float = 1.0,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    margin_x = max(2, int(width * 0.08))
    margin_y = max(2, int(height * 0.18))
    if confidence < 0.7:
        margin_x += 2
        margin_y += 2
    return [
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(image_width, x2 + margin_x),
        min(image_height, y2 + margin_y),
    ]


def detect_textured_overlay(
    img_array: np.ndarray,
    region: dict,
    mask: np.ndarray,
    bg_type: str,
    bg_meta: dict,
) -> dict | None:
    if region.get("tipo") not in {"fala", "narracao"}:
        return None
    if bg_type not in {"textured", "textured_vertical", "textured_horizontal", "solid_mid", "gradient"}:
        return None

    bbox = region.get("bbox")
    if not bbox:
        return None
    balloon_mask = _extract_textured_balloon_mask(img_array, bbox, region)
    if balloon_mask is None:
        return None

    corner_colors = _sample_balloon_corner_colors(img_array, balloon_mask)
    if corner_colors is None:
        return None
    if not _should_use_textured_overlay(corner_colors, bg_type):
        return None

    overlay_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    height, width = img_array.shape[:2]
    for text in region.get("texts", []):
        text_bbox = text.get("bbox")
        if not text_bbox:
            continue
        ox1, oy1, ox2, oy2 = _expand_overlay_bbox(
            text_bbox,
            image_width=width,
            image_height=height,
            confidence=float(text.get("confidence", 0.0)),
        )
        cv2.rectangle(overlay_mask, (ox1, oy1), (ox2, oy2), 255, thickness=-1)

    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    overlay_mask = cv2.morphologyEx(overlay_mask, cv2.MORPH_CLOSE, merge_kernel, iterations=1)
    overlay_mask = cv2.bitwise_and(overlay_mask, balloon_mask)
    if not np.any(overlay_mask):
        return None

    return {
        "color": tuple(int(c) for c in np.median(np.array(list(corner_colors.values())), axis=0)),
        "overlay_mask": overlay_mask,
        "balloon_mask": balloon_mask,
        "corner_colors": corner_colors,
        "feather_px": 9,
    }


def _should_use_textured_overlay(
    sampled_colors: dict[str, tuple[int, int, int]],
    bg_type: str,
) -> bool:
    values = np.array(list(sampled_colors.values()), dtype=np.float32)
    if values.size == 0:
        return False
    spread = float(np.max(values[:, 0]) - np.min(values[:, 0]))
    mean_sat_proxy = float(np.mean(np.max(values, axis=1) - np.min(values, axis=1)))
    if bg_type in {"textured", "textured_vertical", "textured_horizontal"}:
        return True
    return spread >= 12.0 or mean_sat_proxy >= 20.0


def _extract_textured_balloon_mask(
    img_array: np.ndarray,
    bbox: list[int],
    region: dict,
) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h, w = img_array.shape[:2]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = max(24, int(box_w * 1.0))
    pad_y = max(24, int(box_h * 1.2))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(w, x2 + pad_x)
    ry2 = min(h, y2 + pad_y)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi = img_array[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    candidate = np.where((sat > 30) & (val > 18), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if num_labels <= 1:
        return None

    seeds = _cardinal_seed_points([x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1], roi.shape[:2], region)
    component_mask = np.zeros_like(candidate)
    for sx, sy in seeds:
        label = _best_component_near_seed(labels, stats, sat, sx, sy)
        if label > 0:
            component_mask[labels == label] = 255

    if not np.any(component_mask):
        return None

    component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = component_mask
    return full_mask


def _cardinal_seed_points(
    local_bbox: list[int],
    roi_shape: tuple[int, int],
    region: dict,
) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = local_bbox
    roi_h, roi_w = roi_shape[:2]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    inset_x = max(6, int((x2 - x1) * 0.35))
    inset_y = max(6, int((y2 - y1) * 0.45))
    return [
        (int(np.clip(cx, 0, roi_w - 1)), int(np.clip(y1 - inset_y, 0, roi_h - 1))),
        (int(np.clip(cx, 0, roi_w - 1)), int(np.clip(y2 + inset_y, 0, roi_h - 1))),
        (int(np.clip(x1 - inset_x, 0, roi_w - 1)), int(np.clip(cy, 0, roi_h - 1))),
        (int(np.clip(x2 + inset_x, 0, roi_w - 1)), int(np.clip(cy, 0, roi_h - 1))),
    ]


def _best_component_near_seed(
    labels: np.ndarray,
    stats: np.ndarray,
    sat: np.ndarray,
    seed_x: int,
    seed_y: int,
) -> int:
    h, w = labels.shape
    patch_r = 8
    px1 = max(0, seed_x - patch_r)
    py1 = max(0, seed_y - patch_r)
    px2 = min(w, seed_x + patch_r + 1)
    py2 = min(h, seed_y + patch_r + 1)
    patch_labels = labels[py1:py2, px1:px2]
    patch_sat = sat[py1:py2, px1:px2]

    best_label = 0
    best_score = -1.0
    for label in np.unique(patch_labels):
        if label <= 0:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 80:
            continue
        score = float(np.mean(patch_sat[patch_labels == label]))
        if score > best_score:
            best_score = score
            best_label = int(label)
    return best_label


def _sample_balloon_corner_colors(
    img_array: np.ndarray,
    balloon_mask: np.ndarray,
) -> dict[str, tuple[int, int, int]] | None:
    ys, xs = np.where(balloon_mask > 0)
    if len(xs) == 0:
        return None

    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    width = max(1, right - left)
    height = max(1, bottom - top)

    points = {
        "nw": (left + max(4, width // 8), top + max(4, height // 8)),
        "ne": (right - max(4, width // 8), top + max(4, height // 8)),
        "sw": (left + max(4, width // 8), bottom - max(4, height // 8)),
        "se": (right - max(4, width // 8), bottom - max(4, height // 8)),
    }
    sampled: dict[str, tuple[int, int, int]] = {}
    for name, (px, py) in points.items():
        color = _sample_masked_patch_color(img_array, balloon_mask, px, py, radius=10)
        if color is None:
            return None
        sampled[name] = color
    return sampled


def _sample_masked_patch_color(
    img_array: np.ndarray,
    mask: np.ndarray,
    px: int,
    py: int,
    radius: int = 8,
) -> tuple[int, int, int] | None:
    h, w = mask.shape
    x1 = max(0, px - radius)
    y1 = max(0, py - radius)
    x2 = min(w, px + radius + 1)
    y2 = min(h, py + radius + 1)
    patch_mask = mask[y1:y2, x1:x2] > 0
    if not np.any(patch_mask):
        return None
    pixels = img_array[y1:y2, x1:x2][patch_mask]
    if len(pixels) == 0:
        return None
    return tuple(int(c) for c in np.median(pixels, axis=0))


def _detect_linear_texture(
    img_array: np.ndarray,
    bbox: list[int],
    mask: np.ndarray,
) -> str | None:
    x1, y1, x2, y2 = bbox
    h, w = img_array.shape[:2]
    pad = 10
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = img_array[cy1:cy2, cx1:cx2]
    crop_mask = mask[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    valid = crop_mask == 0
    if not np.any(valid):
        valid = np.ones_like(crop_mask, dtype=bool)

    mean_grad_x = float(np.mean(grad_x[valid]))
    mean_grad_y = float(np.mean(grad_y[valid]))
    dominant = max(mean_grad_x, mean_grad_y)
    if dominant < 10.0:
        return None
    if mean_grad_x > mean_grad_y * 1.35:
        return "textured_vertical"
    if mean_grad_y > mean_grad_x * 1.35:
        return "textured_horizontal"
    return None


# ── Preenchimento (sempre sem blur) ──────────────────────────────────────

def apply_fill(
    img_array: np.ndarray,
    mask: np.ndarray,
    bbox: list[int],
    bg_type: str,
    bg_meta: dict,
) -> np.ndarray:
    if bg_type == "balloon_white_overlay":
        return _overlay_fill(img_array, bg_meta["overlay_mask"], bg_meta["color"])

    if bg_type == "textured_overlay":
        return _soft_gradient_overlay_fill(
            img_array,
            bg_meta["overlay_mask"],
            bg_meta["balloon_mask"],
            bg_meta["corner_colors"],
            feather_px=int(bg_meta.get("feather_px", 8)),
        )

    if bg_type in ("solid_light", "solid_dark", "solid_mid"):
        return _flat_fill(img_array, mask, bg_meta["color"])

    if bg_type == "gradient":
        return _gradient_fill(img_array, mask, bbox, bg_meta["top"], bg_meta["bottom"])

    if bg_type == "textured_vertical":
        return _directional_patch_fill(img_array, mask, bbox, axis="vertical")

    if bg_type == "textured_horizontal":
        return _directional_patch_fill(img_array, mask, bbox, axis="horizontal")

    # textured: copia pixels reais do entorno
    return _patch_copy(img_array, mask, bbox)


def _flat_fill(
    img_array: np.ndarray,
    mask: np.ndarray,
    color: tuple,
) -> np.ndarray:
    """Preenche a máscara com uma cor sólida exata."""
    result = img_array.copy()
    result[mask > 0] = np.array(color, dtype=np.uint8)
    return result


def _overlay_fill(
    img_array: np.ndarray,
    overlay_mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    result = img_array.copy()
    result[overlay_mask > 0] = np.array(color, dtype=np.uint8)
    return result


def _soft_overlay_fill(
    img_array: np.ndarray,
    overlay_mask: np.ndarray,
    color: tuple[int, int, int],
    feather_px: int = 4,
) -> np.ndarray:
    result = img_array.copy().astype(np.float32)
    mask_float = overlay_mask.astype(np.float32) / 255.0
    if feather_px > 0:
        ksize = max(3, feather_px * 2 + 1)
        alpha = cv2.GaussianBlur(mask_float, (ksize, ksize), 0)
        alpha = np.clip(alpha * 1.35, 0.0, 1.0)
    else:
        alpha = mask_float

    if not np.any(alpha > 0):
        return img_array

    alpha = alpha[..., None]
    overlay_color = np.array(color, dtype=np.float32)
    result = result * (1.0 - alpha) + overlay_color * alpha
    return result.clip(0, 255).astype(np.uint8)


def _soft_gradient_overlay_fill(
    img_array: np.ndarray,
    overlay_mask: np.ndarray,
    balloon_mask: np.ndarray,
    corner_colors: dict[str, tuple[int, int, int]],
    feather_px: int = 8,
) -> np.ndarray:
    result = img_array.copy().astype(np.float32)
    ys, xs = np.where(balloon_mask > 0)
    if len(xs) == 0:
        return img_array

    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    roi = result[top:bottom + 1, left:right + 1]

    roi_overlay = overlay_mask[top:bottom + 1, left:right + 1].astype(np.float32) / 255.0
    roi_balloon = balloon_mask[top:bottom + 1, left:right + 1].astype(np.float32) / 255.0
    if not np.any(roi_overlay > 0):
        return img_array

    ksize = max(7, feather_px * 2 + 1)
    soft_alpha = cv2.GaussianBlur(roi_overlay, (ksize, ksize), 0)
    alpha = np.maximum(roi_overlay, np.clip(soft_alpha * 1.8, 0.0, 1.0)) * roi_balloon
    if not np.any(alpha > 0):
        return img_array

    height = max(1, roi.shape[0] - 1)
    width = max(1, roi.shape[1] - 1)
    yy, xx = np.indices((roi.shape[0], roi.shape[1]), dtype=np.float32)
    ty = yy / height
    tx = xx / width

    nw = np.array(corner_colors["nw"], dtype=np.float32)
    ne = np.array(corner_colors["ne"], dtype=np.float32)
    sw = np.array(corner_colors["sw"], dtype=np.float32)
    se = np.array(corner_colors["se"], dtype=np.float32)

    top_row = nw[None, None, :] * (1.0 - tx[..., None]) + ne[None, None, :] * tx[..., None]
    bottom_row = sw[None, None, :] * (1.0 - tx[..., None]) + se[None, None, :] * tx[..., None]
    gradient = top_row * (1.0 - ty[..., None]) + bottom_row * ty[..., None]

    alpha = alpha[..., None]
    blended = roi * (1.0 - alpha) + gradient * alpha
    result[top:bottom + 1, left:right + 1] = blended
    return result.clip(0, 255).astype(np.uint8)


def _gradient_fill(
    img_array: np.ndarray,
    mask: np.ndarray,
    bbox: list[int],
    top_color: tuple,
    bottom_color: tuple,
) -> np.ndarray:
    """Preenche a máscara interpolando cores topo→base, sem blur."""
    result = img_array.copy()
    x1, y1, x2, y2 = bbox
    height = max(1, y2 - y1)

    ct = np.array(top_color, dtype=float)
    cb = np.array(bottom_color, dtype=float)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return result

    t = np.clip((ys - y1) / height, 0.0, 1.0)[:, None]
    colors = (ct * (1 - t) + cb * t).clip(0, 255).astype(np.uint8)
    result[ys, xs] = colors
    return result


def _patch_copy(
    img_array: np.ndarray,
    mask: np.ndarray,
    bbox: list[int],
) -> np.ndarray:
    """
    Preenche a máscara copiando os pixels de textura mais próximos do anel
    externo ao texto. Usa vizinho mais próximo vetorizado por chunks.
    Não aplica blur.
    """
    result = img_array.copy()
    h, w = img_array.shape[:2]
    x1, y1, x2, y2 = bbox
    ring_w = max(14, (x2 - x1) // 3, (y2 - y1) // 3)

    ex1 = max(0, x1 - ring_w)
    ey1 = max(0, y1 - ring_w)
    ex2 = min(w, x2 + ring_w)
    ey2 = min(h, y2 + ring_w)

    # Máscara de anel: região expandida excluindo a bbox original
    ring_selection = np.ones((ey2 - ey1, ex2 - ex1), dtype=bool)
    inner_y1 = max(0, y1 - ey1)
    inner_x1 = max(0, x1 - ex1)
    inner_y2 = min(ey2 - ey1, y2 - ey1)
    inner_x2 = min(ex2 - ex1, x2 - ex1)
    ring_selection[inner_y1:inner_y2, inner_x1:inner_x2] = False

    ring_ys_rel, ring_xs_rel = np.where(ring_selection)
    if len(ring_ys_rel) == 0:
        # Fallback: cor mediana
        fallback = _sample_outer_ring(img_array, bbox, ring_width=16)
        if len(fallback) > 0:
            result[mask > 0] = np.median(fallback, axis=0).astype(np.uint8)
        return result

    # Coordenadas absolutas do anel
    ring_ys = ring_ys_rel + ey1
    ring_xs = ring_xs_rel + ex1

    # Subsamplar anel se muito grande (preserva textura representativa)
    max_ring = 4000
    if len(ring_ys) > max_ring:
        step = len(ring_ys) // max_ring
        ring_ys = ring_ys[::step][:max_ring]
        ring_xs = ring_xs[::step][:max_ring]

    # Pixels mascarados que precisam ser preenchidos
    masked_ys, masked_xs = np.where(mask > 0)
    if len(masked_ys) == 0:
        return result

    # Vizinho mais próximo do anel, processado em chunks para evitar OOM
    chunk_size = 3000
    for start in range(0, len(masked_ys), chunk_size):
        end = min(start + chunk_size, len(masked_ys))
        my = masked_ys[start:end, None].astype(np.int32)   # (C, 1)
        mx = masked_xs[start:end, None].astype(np.int32)

        dists = (my - ring_ys[None, :]) ** 2 + (mx - ring_xs[None, :]) ** 2  # (C, R)
        nearest = np.argmin(dists, axis=1)                                      # (C,)
        result[masked_ys[start:end], masked_xs[start:end]] = img_array[ring_ys[nearest], ring_xs[nearest]]

    return result


def _directional_patch_fill(
    img_array: np.ndarray,
    mask: np.ndarray,
    bbox: list[int],
    axis: str,
) -> np.ndarray:
    result = img_array.copy()
    h, w = img_array.shape[:2]
    x1, y1, x2, y2 = bbox

    if axis == "vertical":
        for x in range(max(0, x1), min(w, x2)):
            masked_rows = np.where(mask[:, x] > 0)[0]
            if len(masked_rows) == 0:
                continue
            samples = _collect_column_samples(img_array, mask, x, y1, y2)
            if not samples:
                continue
            sample_rows = np.array([row for row, _ in samples], dtype=np.int32)
            sample_colors = np.array([color for _, color in samples], dtype=np.uint8)
            nearest = np.abs(masked_rows[:, None] - sample_rows[None, :]).argmin(axis=1)
            result[masked_rows, x] = sample_colors[nearest]
        return result

    for y in range(max(0, y1), min(h, y2)):
        masked_cols = np.where(mask[y, :] > 0)[0]
        if len(masked_cols) == 0:
            continue
        samples = _collect_row_samples(img_array, mask, y, x1, x2)
        if not samples:
            continue
        sample_cols = np.array([col for col, _ in samples], dtype=np.int32)
        sample_colors = np.array([color for _, color in samples], dtype=np.uint8)
        nearest = np.abs(masked_cols[:, None] - sample_cols[None, :]).argmin(axis=1)
        result[y, masked_cols] = sample_colors[nearest]
    return result


def _collect_column_samples(
    img_array: np.ndarray,
    mask: np.ndarray,
    column_x: int,
    y1: int,
    y2: int,
) -> list[tuple[int, np.ndarray]]:
    samples: list[tuple[int, np.ndarray]] = []
    h, w = mask.shape
    for offset_x in (0, -1, 1, -2, 2):
        x = column_x + offset_x
        if x < 0 or x >= w:
            continue
        rows = np.where(mask[:, x] == 0)[0]
        rows = rows[(rows >= max(0, y1 - 20)) & (rows < min(h, y2 + 20))]
        if len(rows) == 0:
            continue
        for row in rows[:: max(1, len(rows) // 12)]:
            samples.append((int(row), img_array[row, x]))
        if len(samples) >= 16:
            break
    return samples


def _collect_row_samples(
    img_array: np.ndarray,
    mask: np.ndarray,
    row_y: int,
    x1: int,
    x2: int,
) -> list[tuple[int, np.ndarray]]:
    samples: list[tuple[int, np.ndarray]] = []
    h, w = mask.shape
    for offset_y in (0, -1, 1, -2, 2):
        y = row_y + offset_y
        if y < 0 or y >= h:
            continue
        cols = np.where(mask[y, :] == 0)[0]
        cols = cols[(cols >= max(0, x1 - 20)) & (cols < min(w, x2 + 20))]
        if len(cols) == 0:
            continue
        for col in cols[:: max(1, len(cols) // 12)]:
            samples.append((int(col), img_array[y, col]))
        if len(samples) >= 16:
            break
    return samples


# ── Avaliação e correção de borda ─────────────────────────────────────────

def is_natural(
    filled: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    threshold: float = 20.0,
) -> bool:
    """
    Verifica se a transição na borda da máscara é suave.
    Compara os pixels preenchidos na borda interna da máscara
    com os pixels originais vizinhos fora da máscara.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask, kernel)
    # Borda externa: pixels adjacentes à máscara mas fora dela
    outer_edge = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))

    if not np.any(outer_edge):
        return True

    filled_edge = filled[outer_edge > 0].astype(float)
    orig_edge = original[outer_edge > 0].astype(float)

    if len(filled_edge) == 0:
        return True

    diff = float(np.mean(np.abs(filled_edge - orig_edge)))
    return diff < threshold


def feather_boundary(
    img_array: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 2,
) -> np.ndarray:
    """
    Aplica blur Gaussiano fino SOMENTE na borda externa da máscara.
    O interior preenchido não é tocado. Chamado apenas após apply_fill.
    Usa crop local para evitar blur na imagem inteira.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask, kernel, iterations=feather_radius)
    outer_edge = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))

    if not np.any(outer_edge):
        return img_array

    # Crop to ROI around the edge region instead of blurring full image
    ys, xs = np.where(dilated > 0)
    if len(ys) == 0:
        return img_array
    margin = feather_radius * 2 + 4
    ry1 = max(0, int(ys.min()) - margin)
    ry2 = min(img_array.shape[0], int(ys.max()) + margin + 1)
    rx1 = max(0, int(xs.min()) - margin)
    rx2 = min(img_array.shape[1], int(xs.max()) + margin + 1)

    ksize = feather_radius * 2 + 1
    roi = img_array[ry1:ry2, rx1:rx2]
    blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)

    edge_roi = outer_edge[ry1:ry2, rx1:rx2]
    edge_weight = (edge_roi.astype(float) / 255.0)[..., None]
    result = img_array.copy()
    blended = (roi * (1 - edge_weight) + blurred_roi * edge_weight)
    result[ry1:ry2, rx1:rx2] = blended.clip(0, 255).astype(np.uint8)
    return result


# ── Utilitários ───────────────────────────────────────────────────────────

def _sample_outer_ring(
    img_array: np.ndarray,
    bbox: list[int],
    ring_width: int = 14,
) -> np.ndarray:
    """Retorna array (N, 3) com pixels do anel ao redor da bbox."""
    x1, y1, x2, y2 = bbox
    h, w = img_array.shape[:2]
    rw = max(4, ring_width)

    ex1 = max(0, x1 - rw)
    ey1 = max(0, y1 - rw)
    ex2 = min(w, x2 + rw)
    ey2 = min(h, y2 + rw)

    crop_h, crop_w = ey2 - ey1, ex2 - ex1
    if crop_h <= 0 or crop_w <= 0:
        return np.empty((0, 3), dtype=float)

    crop = img_array[ey1:ey2, ex1:ex2].reshape(-1, 3).astype(float)

    inner = np.zeros((crop_h, crop_w), dtype=bool)
    iy1 = max(0, y1 - ey1)
    ix1 = max(0, x1 - ex1)
    iy2 = min(crop_h, y2 - ey1)
    ix2 = min(crop_w, x2 - ex1)
    inner[iy1:iy2, ix1:ix2] = True
    flat = inner.ravel()

    return crop[~flat]
