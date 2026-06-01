"""Select inpaint strategy by region type."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np

from inpainter.mask_validator import validate_mask


REGION_STRATEGY = {
    "white_balloon": "telea_fast",
    "colored_balloon": "lama_or_patchmatch",
    "caption_box": "preserve_borders",
    "solid_background": "telea_fast",
    "textured_background": "lama_required",
    "gradient_background": "lama_required",
    "dark_background": "lama_required",
    "sfx_text": "skip_without_explicit_config",
    "map_texture": "manual_review",
    "sky_texture": "lama_required",
    "character_overlap": "manual_review",
}


@dataclass(frozen=True)
class MangaCleanerROI:
    x1: int
    y1: int
    x2: int
    y2: int
    source_width: int
    source_height: int
    padded_width: int
    padded_height: int


def _to_int_bbox(raw_bbox: tuple[int, int, int, int] | list[int] | np.ndarray) -> tuple[int, int, int, int]:
    if len(raw_bbox) != 4:
        raise ValueError("bbox inválida")
    x1, y1, x2, y2 = [int(v) for v in raw_bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox invertida")
    return x1, y1, x2, y2


def _ceil_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple deve ser > 0")
    if value <= 0:
        return multiple
    return ((value + multiple - 1) // multiple) * multiple


def manga_cleaner_roi_from_mask(mask: np.ndarray, padding: int = 16, multiple: int = 8) -> MangaCleanerROI:
    if mask.ndim != 2:
        raise ValueError("máscara deve ser 2D")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("máscara sem pixels positivos")

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    height, width = mask.shape[:2]
    pad = max(0, int(padding))
    source_x1 = max(0, x1 - pad)
    source_y1 = max(0, y1 - pad)
    source_x2 = min(width, x2 + pad)
    source_y2 = min(height, y2 + pad)

    source_width = int(source_x2 - source_x1)
    source_height = int(source_y2 - source_y1)
    if source_width <= 0 or source_height <= 0:
        source_x1, source_y1, source_x2, source_y2 = _to_int_bbox((x1, y1, x2, y2))
        source_width = int(source_x2 - source_x1)
        source_height = int(source_y2 - source_y1)

    return MangaCleanerROI(
        x1=source_x1,
        y1=source_y1,
        x2=source_x2,
        y2=source_y2,
        source_width=source_width,
        source_height=source_height,
        padded_width=_ceil_multiple(source_width, multiple),
        padded_height=_ceil_multiple(source_height, multiple),
    )


def reflect_pad_crop_to_multiple(
    crop_image: np.ndarray,
    crop_mask: np.ndarray,
    multiple: int = 8,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if crop_image.size == 0 or crop_mask.size == 0:
        raise ValueError("crop vazio")
    if crop_image.shape[:2] != crop_mask.shape[:2]:
        raise ValueError("imagens com tamanhos diferentes")

    height, width = crop_image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("crop com dimensão inválida")
    if multiple <= 0:
        raise ValueError("multiple deve ser > 0")

    padded_height = _ceil_multiple(height, multiple)
    padded_width = _ceil_multiple(width, multiple)
    pad_h = padded_height - height
    pad_w = padded_width - width
    if pad_h == 0 and pad_w == 0:
        return crop_image.copy(), crop_mask.copy(), (0, 0)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    def _pad_array(
        array: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
    ) -> np.ndarray:
        if array.ndim == 2:
            paddings = ((pad_top, pad_bottom), (pad_left, pad_right))
        elif array.ndim == 3:
            paddings = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            raise ValueError("array inválido para padding")
        try:
            return np.pad(array, paddings, mode="reflect")
        except Exception:
            return np.pad(array, paddings, mode="edge")

    padded_image = _pad_array(crop_image, top, bottom, left, right)
    padded_mask = np.pad(
        crop_mask,
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=0,
    )
    return padded_image, padded_mask, (top, left)


def pasteback_masked_pixels(
    base: np.ndarray,
    crop_output: np.ndarray,
    crop_mask: np.ndarray,
    roi_bbox: list[int] | tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = _to_int_bbox(roi_bbox)
    if base.ndim != 3 or crop_output.ndim != 3 or crop_mask.ndim != 2:
        raise ValueError("formatos inválidos")

    output_h, output_w = crop_output.shape[:2]
    mask_h, mask_w = crop_mask.shape[:2]
    if output_h != mask_h or output_w != mask_w:
        raise ValueError("crop e máscara precisam ter o mesmo tamanho")

    roi_h = y2 - y1
    roi_w = x2 - x1
    if output_h != roi_h or output_w != roi_w:
        raise ValueError("roi e crop com tamanho diferente")

    result = base.copy()
    roi = result[y1:y2, x1:x2]
    if roi.shape[:2] != crop_output.shape[:2]:
        raise ValueError("roi inválida para pasteback")
    paste_mask = crop_mask > 0
    roi[paste_mask] = crop_output[paste_mask]
    result[y1:y2, x1:x2] = roi
    return result


def classify_region(region: dict[str, Any]) -> str:
    if region.get("tipo") == "sfx":
        return "sfx_text"
    if region.get("tipo") == "narracao":
        return "caption_box"
    return region.get("background_type") or region.get("balloon_type") or "white_balloon"


def plan_inpaint(region: dict[str, Any], mask_path: str | Path | None, *, allow_sfx: bool = False) -> dict[str, Any]:
    region_type = classify_region(region)
    if region_type == "sfx_text" and not allow_sfx:
        return {"run": False, "strategy": REGION_STRATEGY[region_type], "qa_flags": ["sfx_preserved"], "region_type": region_type}
    if not mask_path:
        return {"run": False, "strategy": "blocked", "qa_flags": ["mask_missing"], "region_type": region_type}
    mask = validate_mask(mask_path, region.get("bbox"))
    if not mask["valid"]:
        return {"run": False, "strategy": "blocked", "qa_flags": [mask["reason"]], "region_type": region_type}
    return {"run": True, "strategy": REGION_STRATEGY.get(region_type, "manual_review"), "qa_flags": [], "region_type": region_type}


def debug_output_paths(debug_root: str | Path, page: int) -> dict[str, Path]:
    root = Path(debug_root) / "inpaint"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"page_{page:03}"
    return {
        "before": root / f"{stem}_before.png",
        "mask": root / f"{stem}_mask.png",
        "after": root / f"{stem}_after.png",
        "diff": root / f"{stem}_diff.png",
    }

