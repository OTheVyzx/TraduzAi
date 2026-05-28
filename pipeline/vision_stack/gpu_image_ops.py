"""Experimental GPU-backed image operations for strip fast paths.

This module is intentionally isolated from the production pipeline. It exposes
small CPU-equivalent operations that can be benchmarked behind flags before any
automatic inpaint/OCR path depends on them.
"""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cv2_cuda_available() -> bool:
    if not hasattr(cv2, "cuda"):
        return False
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount()) > 0
    except Exception:
        return False


def probe_gpu_image_ops() -> dict[str, Any]:
    """Return runtime capability information for the experimental backends."""

    cv2_cuda = hasattr(cv2, "cuda")
    cuda_device_count = 0
    if cv2_cuda:
        try:
            cuda_device_count = int(cv2.cuda.getCudaEnabledDeviceCount())
        except Exception:
            cuda_device_count = 0
    torch_cuda = _torch_cuda_available()
    cv2_cuda_ready = cv2_cuda and cuda_device_count > 0
    return {
        "torch_cuda": torch_cuda,
        "cv2_cuda": cv2_cuda,
        "cv2_cuda_device_count": cuda_device_count,
        "cv2_cuda_ready": cv2_cuda_ready,
        "cv2_cuda_has_gpumat": bool(cv2_cuda and hasattr(cv2.cuda, "GpuMat")),
        "cv2_cuda_has_resize": bool(cv2_cuda and hasattr(cv2.cuda, "resize")),
        "cv2_cuda_has_threshold": bool(cv2_cuda and hasattr(cv2.cuda, "threshold")),
        "cv2_cuda_has_morphology": bool(cv2_cuda and hasattr(cv2.cuda, "createMorphologyFilter")),
        "cv2_cuda_has_connected_components": bool(cv2_cuda and hasattr(cv2.cuda, "connectedComponents")),
        "selected_backend": "torch" if torch_cuda else ("cv2cuda" if cv2_cuda_ready else "cpu"),
    }


def _normalize_backend(backend: str) -> str:
    value = (backend or "auto").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "cuda": "auto",
        "gpu": "auto",
        "cv2": "cv2cuda",
        "opencvcuda": "cv2cuda",
        "torchcuda": "torch",
    }
    return aliases.get(value, value)


def _mask_u8(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = (arr > 0).astype(np.uint8) * 255
    if shape is not None and arr.shape[:2] != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return arr


def _as_uint8_image(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must be an HxWx3 array")
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _as_uint8_crop(crop: np.ndarray) -> np.ndarray:
    arr = np.asarray(crop)
    if arr.ndim not in {2, 3}:
        raise ValueError("crop must be an HxW or HxWxC array")
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_white_fill(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    backend: str = "auto",
    color: int | tuple[int, int, int] = 255,
) -> np.ndarray:
    """Fill masked pixels with white or a solid RGB color.

    CPU behavior is the reference. GPU backends fall back to CPU if unsupported
    or unavailable so callers can benchmark safely on mixed Windows machines.
    """

    image = _as_uint8_image(image_rgb)
    mask_u8 = _mask_u8(mask, image.shape[:2])
    selected = _normalize_backend(backend)
    if selected == "auto":
        selected = "torch" if _torch_cuda_available() else ("cv2cuda" if _cv2_cuda_available() else "cpu")

    if selected == "torch":
        try:
            return _apply_white_fill_torch(image, mask_u8, color)
        except Exception:
            return _apply_white_fill_cpu(image, mask_u8, color)
    if selected == "cv2cuda":
        try:
            return _apply_white_fill_cv2cuda(image, mask_u8, color)
        except Exception:
            return _apply_white_fill_cpu(image, mask_u8, color)
    return _apply_white_fill_cpu(image, mask_u8, color)


def _rgb_color(color: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError("color tuple must have three channels")
        return tuple(int(max(0, min(255, c))) for c in color)
    value = int(max(0, min(255, color)))
    return value, value, value


def _apply_white_fill_cpu(
    image_rgb: np.ndarray,
    mask_u8: np.ndarray,
    color: int | tuple[int, int, int],
) -> np.ndarray:
    result = image_rgb.copy()
    result[mask_u8 > 0] = _rgb_color(color)
    return result


def _apply_white_fill_torch(
    image_rgb: np.ndarray,
    mask_u8: np.ndarray,
    color: int | tuple[int, int, int],
) -> np.ndarray:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch cuda unavailable")
    image_t = torch.as_tensor(np.ascontiguousarray(image_rgb), device="cuda")
    mask_t = torch.as_tensor(np.ascontiguousarray(mask_u8 > 0), device="cuda")
    color_t = torch.tensor(_rgb_color(color), dtype=image_t.dtype, device=image_t.device)
    result = image_t.clone()
    result[mask_t] = color_t
    return result.cpu().numpy().astype(np.uint8)


def _apply_white_fill_cv2cuda(
    image_rgb: np.ndarray,
    mask_u8: np.ndarray,
    color: int | tuple[int, int, int],
) -> np.ndarray:
    if not _cv2_cuda_available():
        raise RuntimeError("cv2 cuda unavailable")
    gpu = cv2.cuda_GpuMat()
    gpu.upload(image_rgb)
    mask_gpu = cv2.cuda_GpuMat()
    mask_gpu.upload(mask_u8)
    # GpuMat.setTo exists on CUDA-enabled OpenCV builds. It is the cheapest
    # path for simple white-balloon fills when the image already lives on GPU.
    gpu.setTo(_rgb_color(color), mask_gpu)
    return gpu.download().astype(np.uint8)


def expand_mask(
    mask: np.ndarray,
    *,
    kernel_size: int = 3,
    iterations: int = 1,
    backend: str = "auto",
) -> np.ndarray:
    """Dilate a binary mask with an elliptical kernel."""

    mask_u8 = _mask_u8(mask)
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    iterations = max(0, int(iterations))
    if iterations == 0:
        return mask_u8.copy()
    selected = _normalize_backend(backend)
    if selected == "auto":
        selected = "torch" if _torch_cuda_available() else ("cv2cuda" if _cv2_cuda_available() else "cpu")
    if selected == "torch":
        try:
            return _expand_mask_torch(mask_u8, kernel_size, iterations)
        except Exception:
            return _expand_mask_cpu(mask_u8, kernel_size, iterations)
    if selected == "cv2cuda":
        try:
            return _expand_mask_cv2cuda(mask_u8, kernel_size, iterations)
        except Exception:
            return _expand_mask_cpu(mask_u8, kernel_size, iterations)
    return _expand_mask_cpu(mask_u8, kernel_size, iterations)


def _ellipse_kernel(kernel_size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)).astype(np.uint8)


def _expand_mask_cpu(mask_u8: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    return cv2.dilate(mask_u8, _ellipse_kernel(kernel_size), iterations=iterations)


def _expand_mask_torch(mask_u8: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("torch cuda unavailable")
    kernel = torch.as_tensor(_ellipse_kernel(kernel_size), dtype=torch.float32, device="cuda")
    tensor = torch.as_tensor((mask_u8 > 0).astype(np.float32), device="cuda")[None, None, :, :]
    weight = kernel[None, None, :, :]
    padding = kernel_size // 2
    for _ in range(iterations):
        tensor = (F.conv2d(tensor, weight, padding=padding) > 0).to(torch.float32)
    return (tensor[0, 0].cpu().numpy().astype(np.uint8) * 255)


def _expand_mask_cv2cuda(mask_u8: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    if not _cv2_cuda_available() or not hasattr(cv2.cuda, "createMorphologyFilter"):
        raise RuntimeError("cv2 cuda morphology unavailable")
    gpu = cv2.cuda_GpuMat()
    gpu.upload(mask_u8)
    filt = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, _ellipse_kernel(kernel_size))
    result = gpu
    for _ in range(iterations):
        result = filt.apply(result)
    return result.download().astype(np.uint8)


def connected_components_with_stats(
    mask: np.ndarray,
    *,
    min_area: int = 1,
    backend: str = "auto",
) -> list[dict[str, Any]]:
    """Return connected component boxes and stats for a binary mask.

    OpenCV CPU remains the reference implementation. cv2.cuda connected
    components is used only if the local OpenCV build exposes it; most Windows
    wheels do not.
    """

    mask_u8 = _mask_u8(mask)
    selected = _normalize_backend(backend)
    if selected == "auto":
        selected = "cv2cuda" if _cv2_cuda_available() and hasattr(cv2.cuda, "connectedComponents") else "cpu"
    if selected == "cv2cuda":
        try:
            return _connected_components_cv2cuda(mask_u8, min_area=min_area)
        except Exception:
            return _connected_components_cpu(mask_u8, min_area=min_area, backend_used="cpu-fallback")
    return _connected_components_cpu(mask_u8, min_area=min_area, backend_used="cpu")


def _connected_components_cpu(
    mask_u8: np.ndarray,
    *,
    min_area: int,
    backend_used: str,
) -> list[dict[str, Any]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        components.append(
            {
                "label": int(label),
                "bbox": [x, y, x + w, y + h],
                "area": area,
                "centroid": [float(cx), float(cy)],
                "backend": backend_used,
            }
        )
    components.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["bbox"][3], item["bbox"][2]))
    return components


def _connected_components_cv2cuda(mask_u8: np.ndarray, *, min_area: int) -> list[dict[str, Any]]:
    if not _cv2_cuda_available() or not hasattr(cv2.cuda, "connectedComponents"):
        raise RuntimeError("cv2 cuda connected components unavailable")
    gpu = cv2.cuda_GpuMat()
    gpu.upload((mask_u8 > 0).astype(np.uint8))
    labels_gpu = cv2.cuda.connectedComponents(gpu)
    labels = labels_gpu.download().astype(np.int32)
    components: list[dict[str, Any]] = []
    for label in sorted(int(v) for v in np.unique(labels) if int(v) != 0):
        ys, xs = np.where(labels == label)
        area = int(xs.size)
        if area < int(min_area):
            continue
        components.append(
            {
                "label": label,
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                "area": area,
                "centroid": [float(xs.mean()), float(ys.mean())],
                "backend": "cv2cuda",
            }
        )
    components.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["bbox"][3], item["bbox"][2]))
    return components


def resize_crops_batch(
    crops: Iterable[np.ndarray],
    *,
    size: tuple[int, int],
    backend: str = "auto",
    interpolation: int = cv2.INTER_AREA,
) -> list[np.ndarray]:
    """Resize OCR crops while preserving order.

    `size` follows OpenCV convention: (width, height).
    """

    crop_list = [_as_uint8_crop(crop) for crop in crops]
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("size must be positive")
    selected = _normalize_backend(backend)
    if selected == "auto":
        selected = "torch" if _torch_cuda_available() else ("cv2cuda" if _cv2_cuda_available() else "cpu")
    if selected == "torch":
        try:
            return _resize_crops_torch(crop_list, width=width, height=height)
        except Exception:
            return _resize_crops_cpu(crop_list, width=width, height=height, interpolation=interpolation)
    if selected == "cv2cuda":
        try:
            return _resize_crops_cv2cuda(crop_list, width=width, height=height, interpolation=interpolation)
        except Exception:
            return _resize_crops_cpu(crop_list, width=width, height=height, interpolation=interpolation)
    return _resize_crops_cpu(crop_list, width=width, height=height, interpolation=interpolation)


def _resize_crops_cpu(
    crops: list[np.ndarray],
    *,
    width: int,
    height: int,
    interpolation: int,
) -> list[np.ndarray]:
    return [cv2.resize(crop, (width, height), interpolation=interpolation).astype(np.uint8) for crop in crops]


def _resize_crops_torch(crops: list[np.ndarray], *, width: int, height: int) -> list[np.ndarray]:
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("torch cuda unavailable")
    if crops and all(crop.shape == crops[0].shape and crop.ndim == crops[0].ndim for crop in crops):
        first = crops[0]
        batch = torch.as_tensor(np.ascontiguousarray(np.stack(crops, axis=0)), dtype=torch.float32, device="cuda")
        grayscale_batch = first.ndim == 2
        if grayscale_batch:
            batch = batch.unsqueeze(1)
        else:
            batch = batch.permute(0, 3, 1, 2)
        out = F.interpolate(batch, size=(height, width), mode="bilinear", align_corners=False)
        if grayscale_batch:
            arr = out.squeeze(1).clamp(0, 255).byte().cpu().numpy()
            return [arr[index] for index in range(arr.shape[0])]
        arr = out.permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()
        return [arr[index] for index in range(arr.shape[0])]

    resized: list[np.ndarray] = []
    for crop in crops:
        tensor = torch.as_tensor(np.ascontiguousarray(crop), dtype=torch.float32, device="cuda")
        grayscale = tensor.ndim == 2
        if grayscale:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        out = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        if grayscale:
            arr = out.squeeze(0).squeeze(0).clamp(0, 255).byte().cpu().numpy()
        else:
            arr = out.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        resized.append(arr)
    return resized


def _resize_crops_cv2cuda(
    crops: list[np.ndarray],
    *,
    width: int,
    height: int,
    interpolation: int,
) -> list[np.ndarray]:
    if not _cv2_cuda_available() or not hasattr(cv2.cuda, "resize"):
        raise RuntimeError("cv2 cuda resize unavailable")
    resized: list[np.ndarray] = []
    for crop in crops:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(crop)
        out = cv2.cuda.resize(gpu, (width, height), interpolation=interpolation)
        resized.append(out.download().astype(np.uint8))
    return resized
