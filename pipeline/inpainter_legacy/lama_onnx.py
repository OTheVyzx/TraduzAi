"""
ONNX backend for manga-oriented LaMa inpainting.
Uses mayocream/lama-manga-onnx weights when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from .mask_builder import build_mask_regions, build_region_pixel_mask

PRIMARY_MODEL_REPO = "ogkalu/lama-manga-onnx-dynamic"
PRIMARY_MODEL_FILENAME = "lama-manga-dynamic.onnx"
FALLBACK_MODEL_REPO = "mayocream/lama-manga-onnx"
FALLBACK_MODEL_FILENAME = "lama-manga.onnx"

_session = None
_session_path = None


def is_lama_manga_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def ensure_lama_manga_model(models_dir: str | Path = "") -> Path:
    target_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parent.parent / "models" / "lama_manga_onnx_dynamic"
    target_dir.mkdir(parents=True, exist_ok=True)
    primary_path = target_dir / PRIMARY_MODEL_FILENAME
    if primary_path.exists():
        return primary_path

    try:
        downloaded = hf_hub_download(
            repo_id=PRIMARY_MODEL_REPO,
            filename=PRIMARY_MODEL_FILENAME,
            local_dir=str(target_dir),
        )
        return Path(downloaded)
    except Exception:
        fallback_dir = target_dir.parent / "lama_manga_onnx"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / FALLBACK_MODEL_FILENAME
        if fallback_path.exists():
            return fallback_path
        downloaded = hf_hub_download(
            repo_id=FALLBACK_MODEL_REPO,
            filename=FALLBACK_MODEL_FILENAME,
            local_dir=str(fallback_dir),
        )
        return Path(downloaded)


def get_lama_session(models_dir: str | Path = ""):
    global _session, _session_path
    import onnxruntime as ort

    model_path = str(ensure_lama_manga_model(models_dir))
    if _session is None or _session_path != model_path:
        providers = ["CPUExecutionProvider"]
        _session = ort.InferenceSession(model_path, providers=providers)
        _session_path = model_path
    return _session


def prepare_lama_inputs(image_rgb: np.ndarray, mask_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized_image = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
    resized_mask = cv2.resize(mask_gray, (512, 512), interpolation=cv2.INTER_NEAREST)

    image_input = resized_image.astype(np.float32) / 255.0
    image_input = np.transpose(image_input, (2, 0, 1))[None, ...]

    mask_input = (resized_mask > 0).astype(np.float32)[None, None, ...]
    return image_input, mask_input


def pad_to_modulo(chw_image: np.ndarray, modulo: int = 8) -> np.ndarray:
    _, height, width = chw_image.shape
    out_h = ((height + modulo - 1) // modulo) * modulo
    out_w = ((width + modulo - 1) // modulo) * modulo
    pad_h = out_h - height
    pad_w = out_w - width
    if pad_h == 0 and pad_w == 0:
        return chw_image
    return np.pad(chw_image, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")


def prepare_lama_dynamic_inputs(
    image_rgb: np.ndarray,
    mask_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image_chw = np.transpose(image_rgb.astype(np.float32) / 255.0, (2, 0, 1))
    mask_chw = ((mask_gray > 0).astype(np.float32))[None, ...]
    original_size = (image_rgb.shape[0], image_rgb.shape[1])
    image_chw = pad_to_modulo(image_chw, modulo=8)
    mask_chw = pad_to_modulo(mask_chw, modulo=8)
    return image_chw[None, ...], mask_chw[None, ...], original_size


def build_lama_region_jobs(image_rgb: np.ndarray, texts: list[dict]) -> list[dict]:
    regions = build_mask_regions(texts=texts, image_shape=image_rgb.shape)
    jobs: list[dict] = []
    for region in regions:
        mask = build_region_pixel_mask(image_rgb.shape[:2], region)
        if not np.any(mask):
            continue
        x1, y1, x2, y2 = region["bbox"]
        pad_x = max(12, int((x2 - x1) * 0.18))
        pad_y = max(14, int((y2 - y1) * 0.28))
        h, w = image_rgb.shape[:2]
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(w, x2 + pad_x)
        ry2 = min(h, y2 + pad_y)
        crop = image_rgb[ry1:ry2, rx1:rx2].copy()
        crop_mask = mask[ry1:ry2, rx1:rx2].copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        crop_mask = cv2.dilate(crop_mask, kernel, iterations=1)
        balloon_mask = refine_crop_mask_with_balloon_fill(crop, crop_mask)
        text_mask = np.zeros_like(crop_mask)
        for text in region.get("texts", []):
            local_seed = build_local_text_seed(
                bbox=text.get("bbox"),
                crop_bbox=[rx1, ry1, rx2, ry2],
                crop_shape=crop_mask.shape,
            )
            if not np.any(local_seed):
                continue
            local_segment = segment_text_pixels_from_mask(crop, local_seed)
            if np.any(local_segment):
                text_mask = cv2.bitwise_or(text_mask, local_segment)
            else:
                text_mask = cv2.bitwise_or(text_mask, local_seed)
        if not np.any(text_mask):
            text_mask = segment_text_pixels_from_mask(crop, crop_mask)
        if np.any(text_mask):
            if np.any(balloon_mask):
                text_mask = cv2.bitwise_and(text_mask, balloon_mask)
            text_mask = cv2.morphologyEx(
                text_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            text_mask = cv2.dilate(
                text_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            text_area = int(np.count_nonzero(text_mask))
            base_area = int(np.count_nonzero(crop_mask))
            if 24 <= text_area < int(base_area * 0.92):
                crop_mask = text_mask
            elif np.any(balloon_mask):
                crop_mask = balloon_mask
        elif np.any(balloon_mask):
            crop_mask = balloon_mask
        if crop.size == 0 or not np.any(crop_mask):
            continue
        jobs.append(
            {
                "bbox": [rx1, ry1, rx2, ry2],
                "crop": crop,
                "mask": crop_mask,
            }
        )
    return jobs


def refine_crop_mask_with_balloon_fill(crop_rgb: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    if crop_rgb.size == 0 or not np.any(crop_mask):
        return crop_mask

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    seed_y, seed_x = np.argwhere(crop_mask > 0).mean(axis=0).astype(int)
    h, w = crop_mask.shape
    seed_x = int(np.clip(seed_x, 0, w - 1))
    seed_y = int(np.clip(seed_y, 0, h - 1))

    work = gray.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    lo_diff = 14
    up_diff = 14
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    try:
        cv2.floodFill(work, flood_mask, (seed_x, seed_y), 0, (lo_diff,) * 3, (up_diff,) * 3, flags)
    except Exception:
        return crop_mask

    filled = flood_mask[1:-1, 1:-1]
    if not np.any(filled):
        return crop_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)

    original_area = int(np.count_nonzero(crop_mask))
    filled_area = int(np.count_nonzero(filled))
    if filled_area <= original_area:
        return crop_mask
    if filled_area > crop_mask.size * 0.65:
        return crop_mask

    refined = cv2.bitwise_and(filled, filled, mask=(filled > 0).astype(np.uint8) * 255)
    return refined.astype(np.uint8)


def build_local_text_seed(
    bbox: list[int] | None,
    crop_bbox: list[int],
    crop_shape: tuple[int, int],
) -> np.ndarray:
    seed = np.zeros(crop_shape, dtype=np.uint8)
    if not bbox:
        return seed

    x1, y1, x2, y2 = bbox
    rx1, ry1, _, _ = crop_bbox
    h, w = crop_shape
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = max(2, int(width * 0.08))
    pad_y = max(2, int(height * 0.18))
    lx1 = max(0, x1 - rx1 - pad_x)
    ly1 = max(0, y1 - ry1 - pad_y)
    lx2 = min(w, x2 - rx1 + pad_x)
    ly2 = min(h, y2 - ry1 + pad_y)
    if lx2 <= lx1 or ly2 <= ly1:
        return seed

    cv2.rectangle(seed, (lx1, ly1), (lx2, ly2), 255, thickness=-1)
    return seed


def segment_text_pixels_from_mask(crop_rgb: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    if crop_rgb.size == 0 or not np.any(crop_mask):
        return crop_mask

    mask_bin = crop_mask > 0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    outer_ring = cv2.dilate(crop_mask, outer_kernel, iterations=1)
    outer_ring = cv2.subtract(outer_ring, crop_mask)
    outer_pixels = gray[outer_ring > 0]
    if outer_pixels.size < 24:
        outer_pixels = gray[mask_bin]
    if outer_pixels.size == 0:
        return crop_mask

    bg_gray = float(np.median(outer_pixels))
    bg_color = np.median(crop_rgb[outer_ring > 0], axis=0) if np.any(outer_ring) else np.median(crop_rgb[mask_bin], axis=0)

    inside_gray = gray[mask_bin]
    dark_score = bg_gray - float(np.percentile(inside_gray, 15))
    light_score = float(np.percentile(inside_gray, 85)) - bg_gray
    light_on_dark = light_score > dark_score

    gray_delta = gray.astype(np.float32) - bg_gray
    if light_on_dark:
        polarity_mask = gray_delta > max(14.0, np.std(outer_pixels) * 0.65 + 10.0)
    else:
        polarity_mask = (-gray_delta) > max(14.0, np.std(outer_pixels) * 0.65 + 10.0)

    color_delta = np.linalg.norm(crop_rgb.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    inside_color_delta = color_delta[mask_bin]
    nonzero_color_delta = inside_color_delta[inside_color_delta > 2.0]
    if nonzero_color_delta.size:
        color_threshold = max(18.0, float(np.min(nonzero_color_delta) * 0.72))
    else:
        color_threshold = 18.0

    local_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.2)
    local_contrast = cv2.absdiff(gray, local_blur)
    contrast_mask = local_contrast > max(10.0, float(np.percentile(local_contrast[mask_bin], 65)) if np.any(mask_bin) else 10.0)

    segmented = mask_bin & (polarity_mask & ((color_delta >= color_threshold) | contrast_mask))
    segmented = (segmented.astype(np.uint8) * 255)
    segmented = cv2.morphologyEx(
        segmented,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    segmented = cv2.morphologyEx(
        segmented,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    segmented_area = int(np.count_nonzero(segmented))
    base_area = int(np.count_nonzero(crop_mask))
    if segmented_area < 12 or segmented_area > int(base_area * 0.94):
        return crop_mask

    return segmented


def merge_inpainted_crop(
    base_image_rgb: np.ndarray,
    inpainted_crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    bbox: list[int],
) -> np.ndarray:
    result = base_image_rgb.copy()
    x1, y1, x2, y2 = bbox
    region = result[y1:y2, x1:x2]
    alpha = (crop_mask.astype(np.float32) / 255.0)[..., None]
    region[:] = (region.astype(np.float32) * (1.0 - alpha) + inpainted_crop_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return result


def inpaint_region_with_lama(session, crop_rgb: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    input_shape = session.get_inputs()[0].shape
    dynamic_input = any(not isinstance(dim, int) for dim in input_shape[2:])
    if dynamic_input:
        image_input, mask_input, original_size = prepare_lama_dynamic_inputs(crop_rgb, crop_mask)
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {"image": image_input, "mask": mask_input})[0][0]
        output = output[:, : original_size[0], : original_size[1]]
    else:
        image_input, mask_input = prepare_lama_inputs(crop_rgb, crop_mask)
        output = session.run(None, {"image": image_input, "mask": mask_input})[0][0]
    output = np.transpose(output, (1, 2, 0))
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    if output.shape[:2] != crop_rgb.shape[:2]:
        output = cv2.resize(output, (crop_rgb.shape[1], crop_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    return output


def run_lama_manga_inpainting(
    image_files: list[Path],
    ocr_results: list[dict],
    output_dir: str,
    models_dir: str = "",
    progress_callback: Callable | None = None,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    session = get_lama_session(models_dir)
    inpainted_paths: list[Path] = []
    total = len(image_files)

    for index, (img_path, ocr_data) in enumerate(zip(image_files, ocr_results)):
        image = Image.open(img_path).convert("RGB")
        image_rgb = np.array(image)
        jobs = build_lama_region_jobs(image_rgb, ocr_data.get("texts", []))

        result = image_rgb.copy()
        for job in jobs:
            inpainted_crop = inpaint_region_with_lama(session, job["crop"], job["mask"])
            result = merge_inpainted_crop(result, inpainted_crop, job["mask"], job["bbox"])

        dest = output_path / img_path.name
        Image.fromarray(result).save(dest, quality=95)
        inpainted_paths.append(dest)

        if progress_callback:
            progress_callback(index + 1, total, f"Inpainting LaMa pagina {index + 1}/{total}")

    return inpainted_paths
