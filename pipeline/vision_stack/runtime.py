from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from ocr.postprocess import (
    _find_hf_model,
    analyze_style,
    classify_text_type,
    fix_ocr_errors,
    is_editorial_credit,
    is_non_english,
    is_watermark,
    looks_suspicious,
)
from ocr.semantic_reviewer import semantic_refine_text

logger = logging.getLogger(__name__)

_font_detector = None


def _get_font_detector():
    global _font_detector
    if _font_detector is not None:
        return _font_detector
    model_path = _find_hf_model(
        "fffonion/yuzumarker-font-detection",
        "yuzumarker-font-detection.safetensors",
    )
    if model_path is None:
        return None
    fonts_dir = Path(__file__).parent.parent.parent / "fonts"
    try:
        from typesetter.font_detector import FontDetector
        _font_detector = FontDetector(model_path, fonts_dir)
    except Exception as exc:
        logger.warning("FontDetector não carregado: %s", exc)
        return None
    return _font_detector

_detector = None
_ocr_engine = None
_inpainter = None
_configured_models_dir = None


def _emit_stage_progress(progress_callback, stage: str, progress: float, message: str):
    if progress_callback is None:
        return
    try:
        clamped = max(0.0, min(1.0, float(progress)))
    except Exception:
        clamped = 0.0
    progress_callback(stage, clamped, message)


def _quick_text_presence_check(image_rgb: np.ndarray) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return False

    height, width = image_rgb.shape[:2]
    if min(height, width) < 256:
        return True

    max_dim = max(height, width)
    scale = min(1.0, 384.0 / float(max_dim))
    if scale < 1.0:
        resized = cv2.resize(
            image_rgb,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image_rgb

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.2, sigmaY=3.2)
    dark_contrast = cv2.subtract(blur, gray)
    bright_contrast = cv2.subtract(gray, blur)

    dark_mask = (dark_contrast >= 18).astype(np.uint8) * 255
    bright_mask = (bright_contrast >= 18).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    def _has_textlike_components(mask: np.ndarray) -> bool:
        if mask.size == 0 or not np.any(mask):
            return False

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        component_count = 0
        combined_area = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            box_w = int(stats[label, cv2.CC_STAT_WIDTH])
            box_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < 4 or area > 900:
                continue
            if box_w < 2 or box_h < 2:
                continue
            if box_w > 160 or box_h > 80:
                continue
            fill_ratio = area / float(max(1, box_w * box_h))
            aspect_ratio = max(box_w, box_h) / float(max(1, min(box_w, box_h)))
            if fill_ratio < 0.08 or aspect_ratio > 18.0:
                continue
            component_count += 1
            combined_area += area
            if component_count >= 3 or combined_area >= 42:
                return True
        return False

    if _has_textlike_components(dark_mask) or _has_textlike_components(bright_mask):
        return True

    edge_density = float(np.count_nonzero(cv2.Canny(gray, 90, 180))) / float(gray.size)
    gray_std = float(np.std(gray))
    if gray_std >= 18.0 and edge_density >= 0.012:
        return True

    return False


@dataclass
class DebugRunRecorder:
    run_dir: Path
    experiment: str
    image_path: str
    events: list[dict] = field(default_factory=list)
    tile_logs: list[dict] = field(default_factory=list)
    roi_logs: list[dict] = field(default_factory=list)
    seam_cleanup_logs: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload):
        entry = {"event": event, **payload}
        self.events.append(entry)
        if event == "tiled_inpaint":
            self.tile_logs.extend(payload.get("tiles", []))
        elif event == "roi":
            self.roi_logs.append(payload)
        elif event == "seam_cleanup":
            self.seam_cleanup_logs.append(payload)

    def callback(self, payload: dict):
        event = str(payload.get("event", "unknown"))
        rest = {k: v for k, v in payload.items() if k != "event"}
        self.log(event, **rest)

    def save_image(self, name: str, image: np.ndarray):
        path = self.run_dir / name
        if image.ndim == 2:
            cv2.imwrite(str(path), image)
        else:
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def save_json(self, name: str, payload: dict | list):
        path = self.run_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def finalize(self):
        self.save_json(
            "trace.json",
            {
                "experiment": self.experiment,
                "image_path": self.image_path,
                "events": self.events,
                "roi_logs": self.roi_logs,
                "tile_logs": self.tile_logs,
                "seam_cleanup_logs": self.seam_cleanup_logs,
            },
        )


def _profile_to_device(profile: str) -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _profile_to_ocr_model(profile: str) -> str:
    manga_flag = (
        os.getenv("TRADUZAI_ENABLE_MANGA_OCR")
        or os.getenv("MANGATL_ENABLE_MANGA_OCR")
        or ""
    ).strip().lower()
    enable_manga_ocr = manga_flag in {"1", "true", "yes", "on"}
    if enable_manga_ocr and profile not in {"rapida", "compat"}:
        return "manga-ocr"
    return "paddleocr"


def _profile_to_detection_threshold(profile: str) -> float:
    if profile in {"alta", "max"}:
        return 0.42
    if profile in {"rapida", "compat"}:
        return 0.58
    return 0.5


def _configure_model_roots(models_dir: str = ""):
    global _configured_models_dir

    if not models_dir:
        return

    root = Path(models_dir)
    if _configured_models_dir == root:
        return

    from vision_stack import detector as detector_module
    from vision_stack import inpainter as inpainter_module

    detector_module.MODELS_DIR = root
    inpainter_module.MODELS_DIR = root
    _configured_models_dir = root


def _vision_worker_runtime_root(models_dir: str = "") -> str:
    if models_dir:
        try:
            return str(Path(models_dir).resolve().parent)
        except Exception:
            return str(Path(models_dir).parent)
    default = Path("D:/traduzai_data")
    legacy = Path("D:/mangatl_data")
    if not default.exists() and legacy.exists():
        return str(legacy)
    return str(default)


def _find_cuda_toolkit_root() -> Path | None:
    for key in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"):
        value = os.getenv(key, "").strip()
        if value:
            candidate = Path(value)
            nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
            if (candidate / "bin" / nvcc_name).exists():
                return candidate

    if os.name == "nt":
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.exists():
            versions = sorted(
                [
                    path
                    for path in base.iterdir()
                    if path.is_dir() and (path / "bin" / "nvcc.exe").exists()
                ],
                reverse=True,
            )
            if versions:
                return versions[0]
    return None


def _infer_cudarc_cuda_version(cuda_root: Path) -> str | None:
    name = cuda_root.name
    if name.lower().startswith("v"):
        name = name[1:]
    parts = name.split(".")
    if len(parts) < 2:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except ValueError:
        return None
    return f"{major}0{minor}0"


def _build_koharu_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    cuda_root = _find_cuda_toolkit_root()
    if cuda_root is None:
        return env

    cuda_root_str = str(cuda_root)
    env["CUDA_PATH"] = cuda_root_str
    env["CUDA_HOME"] = cuda_root_str
    env["CUDA_ROOT"] = cuda_root_str
    env["CUDA_TOOLKIT_ROOT_DIR"] = cuda_root_str

    cudarc_version = _infer_cudarc_cuda_version(cuda_root)
    if cudarc_version:
        env["CUDARC_CUDA_VERSION"] = cudarc_version

    cuda_bin = cuda_root / "bin"
    if cuda_bin.exists():
        current_path = env.get("PATH", "")
        env["PATH"] = f"{cuda_bin}{os.pathsep}{current_path}" if current_path else str(cuda_bin)

    return env


def _build_koharu_worker_page_result(
    image_rgb: np.ndarray,
    image_label: str,
    worker_payload: dict,
    profile: str = "quality",
    progress_callback=None,
) -> dict:
    worker_text_blocks = list(worker_payload.get("text_blocks") or worker_payload.get("textBlocks") or [])
    worker_bubble_regions = list(worker_payload.get("bubble_regions") or worker_payload.get("bubbleRegions") or [])
    blocks = []
    texts = []
    for item in worker_text_blocks:
        bbox = [int(v) for v in item.get("bbox", [0, 0, 0, 0])]
        blocks.append(
            SimpleNamespace(
                xyxy=tuple(bbox),
                mask=None,
                confidence=float(item.get("confidence", 0.0)),
                detector=item.get("detector"),
                line_polygons=item.get("line_polygons"),
                source_direction=item.get("source_direction"),
            )
        )
        texts.append(str(item.get("text", "")))

    page_result = build_page_result(
        image_path=image_label,
        image_rgb=image_rgb,
        blocks=blocks,
        texts=texts,
        profile=profile,
        ocr_backend="koharu-paddle-ocr-vl-1.5",
        enable_font_detection=True,
        progress_callback=progress_callback,
    )
    page_result["_bubble_regions"] = worker_bubble_regions
    page_result["_vision_backend"] = "koharu"
    return page_result


def _run_koharu_worker_detect_ocr(
    image_rgb: np.ndarray,
    image_label: str,
    vision_worker_path: str,
    models_dir: str = "",
    profile: str = "quality",
    progress_callback=None,
) -> dict:
    worker_path = Path(str(vision_worker_path).strip())
    if not worker_path.exists():
        raise FileNotFoundError(f"Koharu vision worker nao encontrado: {worker_path}")

    _emit_stage_progress(progress_callback, "load_detector", 0.08, "Carregando detector Koharu")
    _emit_stage_progress(progress_callback, "load_ocr_engine", 0.18, "Carregando OCR Koharu")

    runtime_root = _vision_worker_runtime_root(models_dir)
    request_payload = {
        "imagePath": image_label,
        "mode": "page",
        "runtimeRoot": runtime_root,
        "cpu": False,
        "maxNewTokens": 128,
        "detectionThreshold": _profile_to_detection_threshold(profile),
    }

    with tempfile.TemporaryDirectory(prefix="traduzai_koharu_vision_") as tmpdir:
        request_path = Path(tmpdir) / "request.json"
        request_path.write_text(
            json.dumps(request_payload, ensure_ascii=False),
            encoding="utf-8",
        )

        _emit_stage_progress(progress_callback, "detect_text", 0.38, "Detectando blocos com Koharu")
        result = subprocess.run(
            [str(worker_path), "--request-file", str(request_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_build_koharu_worker_env(),
            check=False,
        )

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"codigo {result.returncode}"
        raise RuntimeError(f"Koharu vision worker falhou: {detail}")

    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Koharu vision worker retornou stdout vazio")

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON invalido do Koharu vision worker: {exc}") from exc

    if str(payload.get("status", "")).lower() != "ok":
        raise RuntimeError(str(payload.get("error") or "worker sem status ok"))

    _emit_stage_progress(progress_callback, "recognize_text", 0.62, "Reconhecendo texto com PaddleOCR-VL")
    return _build_koharu_worker_page_result(
        image_rgb=image_rgb,
        image_label=image_label,
        worker_payload=payload,
        profile=profile,
        progress_callback=progress_callback,
    )


def _get_detector(profile: str = "quality"):
    global _detector
    if _detector is None:
        from vision_stack.detector import TextDetector

        _detector = TextDetector(
            model="comic-text-detector",
            device=_profile_to_device(profile),
            half=True,
        )
    return _detector


def _get_ocr_engine(profile: str = "quality", lang: str = "en"):
    global _ocr_engine
    desired_model = _profile_to_ocr_model(profile)
    current_request = getattr(_ocr_engine, "_requested_model", getattr(_ocr_engine, "model_name", ""))
    current_lang = getattr(_ocr_engine, "lang", "en")
    
    if _ocr_engine is None or current_request != desired_model or current_lang != lang:
        from vision_stack.ocr import OCREngine

        _ocr_engine = OCREngine(
            model=desired_model,
            device=_profile_to_device(profile),
            half=True,
            lang=lang,
        )
    return _ocr_engine


def _get_inpainter(profile: str = "quality"):
    global _inpainter
    if _inpainter is None:
        from vision_stack.inpainter import Inpainter

        _inpainter = Inpainter(
            model="lama-manga",
            device=_profile_to_device(profile),
            half=True,
        )
    return _inpainter


def warmup_visual_stack(models_dir: str = "", profile: str = "quality"):
    _configure_model_roots(models_dir)

    detector = _get_detector(profile)
    ocr = _get_ocr_engine(profile)
    font_detector = _get_font_detector()

    sample_image = np.full((256, 256, 3), 255, dtype=np.uint8)
    cv2.putText(
        sample_image,
        "WARM",
        (36, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (18, 18, 18),
        3,
        cv2.LINE_AA,
    )

    try:
        detector.detect(sample_image, conf_threshold=_profile_to_detection_threshold(profile))
    except Exception as exc:
        logger.warning("Warmup do detector falhou: %s", exc)

    sample_crop = sample_image[84:172, 28:228]
    try:
        ocr.recognize_batch([sample_crop])
    except Exception as exc:
        logger.warning("Warmup do OCR falhou: %s", exc)

    if font_detector is not None:
        try:
            font_detector.detect(sample_crop, allow_default=False)
        except Exception as exc:
            logger.warning("Warmup do FontDetector falhou: %s", exc)


def _new_debug_run_root(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir else Path.cwd().parent / "debug_runs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root / f"{stamp}_{uuid4().hex[:8]}"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _save_mask_png(path: Path, mask: np.ndarray):
    cv2.imwrite(str(path), mask.astype(np.uint8))


def _draw_boxes_overlay(image_rgb: np.ndarray, blocks: list[dict]) -> np.ndarray:
    overlay = image_rgb.copy()
    for index, block in enumerate(blocks, start=1):
        bbox = [int(v) for v in block.get("bbox", [0, 0, 0, 0])]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 64, 64), 2)
        cv2.putText(
            overlay,
            str(index),
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _draw_roi_boundaries_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return overlay
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return overlay


def _draw_tile_boundaries_overlay(image_rgb: np.ndarray, tiles: list[dict]) -> np.ndarray:
    overlay = image_rgb.copy()
    for tile in tiles:
        x1, y1, x2, y2 = int(tile["x1"]), int(tile["y1"]), int(tile["x2"]), int(tile["y2"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 1)
    return overlay


def _load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _save_image_rgb(image_rgb: np.ndarray, dest: Path):
    Image.fromarray(image_rgb).save(dest, quality=95)


def _build_diff_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(a, b)
    if diff.ndim == 3:
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        boosted = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(boosted, cv2.COLOR_GRAY2RGB)
    boosted = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(boosted, cv2.COLOR_GRAY2RGB)


def _call_inpainter(
    inpainter,
    image_np: np.ndarray,
    mask: np.ndarray,
    batch_size: int = 4,
    debug: DebugRunRecorder | None = None,
    force_no_tiling: bool = False,
) -> np.ndarray:
    kwargs = {"batch_size": batch_size}
    if debug is not None:
        kwargs["debug"] = debug.callback
    if force_no_tiling:
        kwargs["force_no_tiling"] = True
    try:
        return inpainter.inpaint(image_np, mask, **kwargs)
    except TypeError:
        return inpainter.inpaint(image_np, mask, batch_size=batch_size)


def _serialize_block(block, page_shape: tuple[int, int]) -> dict:
    x1, y1, x2, y2 = [int(round(v)) for v in block.xyxy]
    x1 = max(0, min(page_shape[1], x1))
    x2 = max(0, min(page_shape[1], x2))
    y1 = max(0, min(page_shape[0], y1))
    y2 = max(0, min(page_shape[0], y2))

    local_mask = None
    mask = getattr(block, "mask", None)
    if isinstance(mask, np.ndarray) and mask.size > 0:
        if mask.shape == page_shape:
            local_mask = mask[y1:y2, x1:x2].copy()
        else:
            local_mask = mask.copy()

    return {
        "bbox": [x1, y1, x2, y2],
        "mask": local_mask,
        "confidence": float(getattr(block, "confidence", 0.0)),
    }


def _clone_page_result(page_result: dict) -> dict:
    cloned_texts = [dict(item) for item in page_result.get("texts", [])]
    cloned_blocks = []
    for block in page_result.get("_vision_blocks", []):
        cloned_block = dict(block)
        mask = cloned_block.get("mask")
        if isinstance(mask, np.ndarray):
            cloned_block["mask"] = mask.copy()
        cloned_blocks.append(cloned_block)
    return {
        **page_result,
        "texts": cloned_texts,
        "_vision_blocks": cloned_blocks,
    }


def _normalize_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _tokenize_text(text: str) -> list[str]:
    return [token for token in re.split(r"\s+", str(text or "").strip()) if token]


def _bbox_union(a: list[int], b: list[int]) -> list[int]:
    return [
        min(int(a[0]), int(b[0])),
        min(int(a[1]), int(b[1])),
        max(int(a[2]), int(b[2])),
        max(int(a[3]), int(b[3])),
    ]


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1.0, area_a + area_b - inter)


def _bbox_contains_center(container: list[int], inner: list[int], margin: int = 24) -> bool:
    cx, cy = _bbox_center(inner)
    return (
        float(container[0]) - margin <= cx <= float(container[2]) + margin
        and float(container[1]) - margin <= cy <= float(container[3]) + margin
    )


def _bbox_gaps(a: list[int], b: list[int]) -> tuple[float, float]:
    horiz_gap = max(0.0, max(float(a[0]), float(b[0])) - min(float(a[2]), float(b[2])))
    vert_gap = max(0.0, max(float(a[1]), float(b[1])) - min(float(a[3]), float(b[3])))
    return horiz_gap, vert_gap


def _merge_text_fragments(base_text: str, residual_text: str, base_bbox: list[int], residual_bbox: list[int]) -> str:
    base_norm = _normalize_text_key(base_text)
    residual_norm = _normalize_text_key(residual_text)
    if not residual_norm:
        return base_text
    if residual_norm == base_norm or residual_norm in base_norm:
        return base_text
    if base_norm and base_norm in residual_norm:
        return residual_text

    base_tokens = _tokenize_text(base_text)
    residual_tokens = _tokenize_text(residual_text)
    if not base_tokens:
        return residual_text
    if not residual_tokens:
        return base_text

    merged_tokens = list(base_tokens)
    dedupe_norm = {_normalize_text_key(token) for token in merged_tokens}
    residual_tokens = [token for token in residual_tokens if _normalize_text_key(token) not in dedupe_norm]
    if not residual_tokens:
        return base_text

    width = max(1.0, float(base_bbox[2] - base_bbox[0]))
    residual_cx, _ = _bbox_center(residual_bbox)
    ratio = min(1.0, max(0.0, (residual_cx - float(base_bbox[0])) / width))
    insert_at = min(len(merged_tokens), max(0, int(math.ceil(ratio * len(merged_tokens)))))
    merged_tokens = merged_tokens[:insert_at] + residual_tokens + merged_tokens[insert_at:]

    compacted: list[str] = []
    for token in merged_tokens:
        norm = _normalize_text_key(token)
        if compacted and norm and _normalize_text_key(compacted[-1]) == norm:
            continue
        compacted.append(token)
    return " ".join(compacted)


def _merge_nearby_bboxes(boxes: list[list[int]], gap_x: int = 60, gap_y: int = 40) -> list[list[int]]:
    pending = [list(box) for box in boxes if box and len(box) == 4]
    merged: list[list[int]] = []

    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            next_pending = []
            for other in pending:
                horiz_overlap = min(current[2], other[2]) - max(current[0], other[0])
                horiz_gap = max(0, max(current[0], other[0]) - min(current[2], other[2]))
                vert_overlap = min(current[3], other[3]) - max(current[1], other[1])
                vert_gap = max(0, max(current[1], other[1]) - min(current[3], other[3]))
                same_balloon = (
                    horiz_overlap >= -gap_x and vert_gap <= gap_y
                ) or (
                    vert_overlap >= -gap_y and horiz_gap <= gap_x
                )
                if same_balloon:
                    current = [
                        min(current[0], other[0]),
                        min(current[1], other[1]),
                        max(current[2], other[2]),
                        max(current[3], other[3]),
                    ]
                    changed = True
                else:
                    next_pending.append(other)
            pending = next_pending
        merged.append(current)

    return merged


def _group_text_indices_by_balloon(texts: list[dict], gap_x: int = 90, gap_y: int = 54) -> list[list[int]]:
    clusters: list[list[int]] = []
    for index, text in enumerate(texts):
        if text.get("skip_processing"):
            continue
        bbox = text.get("bbox", [0, 0, 0, 0])
        attached = False
        for cluster in clusters:
            cluster_bbox = texts[cluster[0]].get("_cluster_bbox")
            if cluster_bbox is None:
                cluster_bbox = texts[cluster[0]].get("bbox", [0, 0, 0, 0])
                for cluster_index in cluster[1:]:
                    cluster_bbox = _bbox_union(cluster_bbox, texts[cluster_index].get("bbox", [0, 0, 0, 0]))
                texts[cluster[0]]["_cluster_bbox"] = cluster_bbox
            horiz_gap, vert_gap = _bbox_gaps(cluster_bbox, bbox)
            horiz_overlap = min(cluster_bbox[2], bbox[2]) - max(cluster_bbox[0], bbox[0])
            vert_overlap = min(cluster_bbox[3], bbox[3]) - max(cluster_bbox[1], bbox[1])
            same_cluster = (
                horiz_overlap >= -gap_x and vert_gap <= gap_y
            ) or (
                vert_overlap >= -gap_y and horiz_gap <= gap_x
            )
            if same_cluster:
                cluster.append(index)
                texts[cluster[0]]["_cluster_bbox"] = _bbox_union(cluster_bbox, bbox)
                attached = True
                break
        if not attached:
            clusters.append([index])
            texts[index]["_cluster_bbox"] = bbox
    return clusters


def _expand_bbox(
    bbox: list[int],
    image_shape: tuple[int, int] | tuple[int, int, int],
    pad_x_ratio: float = 0.05,
    pad_y_ratio: float = 0.18,
    min_pad_x: int = 8,
    min_pad_y: int = 14,
) -> list[int]:
    if len(image_shape) == 3:
        height, width = image_shape[:2]
    else:
        height, width = image_shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad_x = max(min_pad_x, int((x2 - x1) * pad_x_ratio))
    pad_y = max(min_pad_y, int((y2 - y1) * pad_y_ratio))
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    ]


def _enlarge_koharu_window(
    bbox: list[int],
    image_width: int,
    image_height: int,
    ratio: float = 1.7,
    aspect_ratio: float = 1.0,
) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = float(max(0, x2 - x1))
    height = float(max(0, y2 - y1))
    if width <= 0.0 or height <= 0.0 or aspect_ratio <= 0.0 or ratio <= 1.0:
        return [x1, y1, x2, y2]

    a = float(aspect_ratio)
    b = width + height * aspect_ratio
    c = (1.0 - ratio) * width * height
    discriminant = max(0.0, b * b - 4.0 * a * c)
    delta = round(((-b + math.sqrt(discriminant)) / (2.0 * a)) / 2.0)
    delta_h = max(0, int(delta))
    delta_w = max(0, int(round(delta * aspect_ratio)))

    delta_w = min(delta_w, x1, max(0, image_width - x2))
    delta_h = min(delta_h, y1, max(0, image_height - y2))

    return [
        max(0, x1 - delta_w),
        max(0, y1 - delta_h),
        min(int(image_width), x2 + delta_w),
        min(int(image_height), y2 + delta_h),
    ]


def _mask_nonzero_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _mask_overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero((left > 0) & (right > 0)))


def _median_channel(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.median(values.astype(np.float32)))


def _median_rgb(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float] | None:
    pixels = image_rgb[mask > 0]
    if pixels.size == 0:
        return None
    medians = np.median(pixels.astype(np.float32), axis=0)
    return (float(medians[0]), float(medians[1]), float(medians[2]))


def _color_stddev(image_rgb: np.ndarray, mask: np.ndarray, median_rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    pixels = image_rgb[mask > 0]
    if pixels.size == 0:
        return (float("inf"), float("inf"), float("inf"))
    diffs = pixels.astype(np.float32) - np.asarray(median_rgb, dtype=np.float32)[None, :]
    std = np.sqrt(np.mean(np.square(diffs), axis=0))
    return (float(std[0]), float(std[1]), float(std[2]))


def _stddev3(values: tuple[float, float, float]) -> float:
    array = np.asarray(values, dtype=np.float32)
    return float(np.std(array))


def _extract_koharu_balloon_masks(image_rgb: np.ndarray, text_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if image_rgb.shape[:2] != text_mask.shape[:2]:
        return None

    text_bbox = _mask_nonzero_bbox(text_mask)
    text_sum = int(np.count_nonzero(text_mask))
    if text_bbox is None or text_sum == 0:
        return None

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
    cannyed = cv2.Canny(blurred, 70.0, 140.0)
    cannyed = cv2.dilate(
        cannyed,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    cannyed[0, :] = 255
    cannyed[-1, :] = 255
    cannyed[:, 0] = 255
    cannyed[:, -1] = 255
    cannyed[text_mask > 0] = 0

    contours, _ = cv2.findContours(cannyed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_mask = None
    best_area = float("inf")
    tx1, ty1, tx2, ty2 = text_bbox

    for contour in contours:
        if contour is None or len(contour) < 3:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        bbox = [int(bx), int(by), int(bx + bw), int(by + bh)]
        if bbox[0] > tx1 or bbox[1] > ty1 or bbox[2] < tx2 or bbox[3] < ty2:
            continue

        candidate = np.zeros(text_mask.shape, dtype=np.uint8)
        cv2.drawContours(candidate, [contour], -1, 255, thickness=-1)
        if _mask_overlap_count(candidate, text_mask) < text_sum:
            continue

        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        if area < best_area:
            best_area = area
            best_mask = candidate

    if best_mask is None:
        return None

    non_text_mask = best_mask.copy()
    non_text_mask[text_mask > 0] = 0
    return best_mask, non_text_mask


def _try_koharu_balloon_fill(image_rgb: np.ndarray, text_mask: np.ndarray) -> np.ndarray | None:
    masks = _extract_koharu_balloon_masks(image_rgb, text_mask)
    if masks is None:
        return None

    balloon_mask, non_text_mask = masks
    average_bg_color = _median_rgb(image_rgb, non_text_mask)
    if average_bg_color is None:
        return None

    std_rgb = _color_stddev(image_rgb, non_text_mask, average_bg_color)
    inpaint_threshold = 7.0 if _stddev3(std_rgb) > 1.0 else 10.0
    if max(std_rgb) >= inpaint_threshold:
        return None

    result = image_rgb.copy()
    fill = np.asarray([int(round(channel)) for channel in average_bg_color], dtype=np.uint8)
    result[balloon_mask > 0] = fill
    return result


def _clear_mask_bbox(mask: np.ndarray, bbox: list[int]) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(mask.shape[1], x1))
    x2 = max(0, min(mask.shape[1], x2))
    y1 = max(0, min(mask.shape[0], y1))
    y2 = max(0, min(mask.shape[0], y2))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 0


def _build_refined_bbox_mask(image_rgb: np.ndarray, bbox: list[int]) -> tuple[int, int, np.ndarray] | None:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = max(4, int(box_w * 0.12))
    pad_y = max(4, int(box_h * 0.22))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(width, x2 + pad_x)
    ry2 = min(height, y2 + pad_y)

    crop = image_rgb[ry1:ry2, rx1:rx2]
    crop_h, crop_w = crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        return None

    seed = np.zeros((crop_h, crop_w), dtype=np.uint8)
    sx1 = max(0, x1 - rx1)
    sy1 = max(0, y1 - ry1)
    sx2 = min(crop_w, x2 - rx1)
    sy2 = min(crop_h, y2 - ry1)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    seed[sy1:sy2, sx1:sx2] = 255

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    outer_ring = cv2.dilate(seed, outer_kernel, iterations=1)
    outer_ring = cv2.subtract(outer_ring, seed)
    outer_pixels = gray[outer_ring > 0]
    if outer_pixels.size < 24:
        outer_pixels = gray[seed > 0]
    if outer_pixels.size == 0:
        return rx1, ry1, seed

    bg_gray = float(np.median(outer_pixels))
    bg_color = (
        np.median(crop[outer_ring > 0], axis=0)
        if np.any(outer_ring)
        else np.median(crop[seed > 0], axis=0)
    )

    inside_mask = seed > 0
    inside_gray = gray[inside_mask]
    dark_score = bg_gray - float(np.percentile(inside_gray, 15))
    light_score = float(np.percentile(inside_gray, 85)) - bg_gray
    light_on_dark = light_score > dark_score

    gray_delta = gray.astype(np.float32) - bg_gray
    deviation = float(np.std(outer_pixels)) if outer_pixels.size else 0.0
    if light_on_dark:
        polarity_mask = gray_delta > max(12.0, deviation * 0.65 + 8.0)
    else:
        polarity_mask = (-gray_delta) > max(12.0, deviation * 0.65 + 8.0)

    color_delta = np.linalg.norm(crop.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    local_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    local_contrast = cv2.absdiff(gray, local_blur)
    contrast_thresh = float(np.percentile(local_contrast[inside_mask], 60)) if np.any(inside_mask) else 8.0
    contrast_mask = local_contrast >= max(8.0, contrast_thresh)

    refined = inside_mask & polarity_mask & ((color_delta >= 14.0) | contrast_mask)
    refined = refined.astype(np.uint8) * 255
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    seed_area = int(np.count_nonzero(seed))
    refined_area = int(np.count_nonzero(refined))
    if refined_area < max(12, int(seed_area * 0.04)):
        refined = seed.copy()

    dilate_w = max(3, min(9, (box_w // 18) * 2 + 1))
    dilate_h = max(3, min(11, (box_h // 10) * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_w, dilate_h))
    refined = cv2.dilate(refined, kernel, iterations=1)
    refined = cv2.bitwise_and(refined, seed)
    return rx1, ry1, refined


def _is_white_balloon_region(image_rgb: np.ndarray, bbox: list[int]) -> bool:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return False

    pad_x = max(8, int((x2 - x1) * 0.25))
    pad_y = max(8, int((y2 - y1) * 0.45))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(width, x2 + pad_x)
    ry2 = min(height, y2 + pad_y)
    crop = image_rgb[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    seed = np.zeros(gray.shape, dtype=np.uint8)
    sx1 = max(0, x1 - rx1)
    sy1 = max(0, y1 - ry1)
    sx2 = min(gray.shape[1], x2 - rx1)
    sy2 = min(gray.shape[0], y2 - ry1)
    seed[sy1:sy2, sx1:sx2] = 255

    bright = (gray >= 222).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    seed_area = max(1, int(np.count_nonzero(seed)))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_edge = x <= 0 or y <= 0 or (x + w) >= bright.shape[1] or (y + h) >= bright.shape[0]
        if touches_edge or area < max(80, int(seed_area * 1.4)):
            continue
        component = labels == label
        if np.any(seed[component] > 0):
            return True

    brightness = float(np.percentile(gray, 75))
    bright_ratio = float(np.mean(gray >= 220))
    if brightness < 236.0 or bright_ratio < 0.55:
        return False

    fill_mask = _extract_white_balloon_fill_mask(image_rgb, bbox)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    fill_area = int(np.count_nonzero(fill_mask))
    if fill_area < int(bbox_area * 0.9):
        return False

    full_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    boundary = cv2.subtract(
        cv2.dilate(fill_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1),
        cv2.erode(fill_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1),
    )
    boundary_pixels = full_gray[boundary > 0]
    if boundary_pixels.size == 0:
        return False
    dark_outline_ratio = float(np.mean(boundary_pixels <= 132))
    if dark_outline_ratio >= 0.01:
        return True

    spread = float(np.std(gray))
    p20 = float(np.percentile(gray, 20))
    return brightness >= 240.0 and spread <= 18.0 and p20 >= 220.0


def _should_use_base_white_balloon_font(image_rgb: np.ndarray, bbox: list[int]) -> bool:
    if _is_white_balloon_region(image_rgb, bbox):
        return True

    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return False

    sample_bbox = _expand_bbox(
        [x1, y1, x2, y2],
        image_rgb.shape,
        pad_x_ratio=0.08,
        pad_y_ratio=0.18,
        min_pad_x=6,
        min_pad_y=8,
    )
    sx1, sy1, sx2, sy2 = sample_bbox
    crop = image_rgb[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    brightness = float(np.percentile(gray, 75))
    bright_ratio = float(np.mean(gray >= 220))
    dark_ratio = float(np.mean(gray <= 110))
    return brightness >= 240.0 and bright_ratio >= 0.58 and dark_ratio <= 0.22


def _fill_internal_mask_holes(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not np.any(mask):
        return mask

    filled = mask.astype(np.uint8).copy()
    binary = (filled > 0).astype(np.uint8)
    inverse = cv2.bitwise_not(binary * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((inverse > 0).astype(np.uint8), connectivity=8)

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_edge = x <= 0 or y <= 0 or (x + w) >= filled.shape[1] or (y + h) >= filled.shape[0]
        if touches_edge:
            continue
        filled[labels == label] = 255

    return filled


def _extract_white_balloon_fill_mask(image_rgb: np.ndarray, bbox: list[int]) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return mask

    pad_x = max(12, int((x2 - x1) * 0.45))
    pad_y = max(12, int((y2 - y1) * 0.9))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(width, x2 + pad_x)
    ry2 = min(height, y2 + pad_y)
    crop = image_rgb[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return mask

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    bright = (gray >= 225).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    seed = np.zeros_like(bright)
    sx1 = max(0, x1 - rx1)
    sy1 = max(0, y1 - ry1)
    sx2 = min(bright.shape[1], x2 - rx1)
    sy2 = min(bright.shape[0], y2 - ry1)
    if sx2 <= sx1 or sy2 <= sy1:
        return mask
    seed[sy1:sy2, sx1:sx2] = 255
    search_seed = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    )
    local_ellipse = np.zeros_like(bright)
    sx1 = max(0, x1 - rx1)
    sy1 = max(0, y1 - ry1)
    sx2 = min(bright.shape[1], x2 - rx1)
    sy2 = min(bright.shape[0], y2 - ry1)
    local_cx = int((sx1 + sx2) / 2)
    local_cy = int((sy1 + sy2) / 2)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    aspect = float(box_w) / float(max(1, box_h))
    if box_w <= 90:
        axis_x = max(46, min(bright.shape[1] // 2, int(box_w * 1.18)))
    else:
        axis_x = max(40, min(bright.shape[1] // 2, int(box_w * 0.76)))
    if aspect >= 2.0:
        if box_h <= 42:
            axis_y = max(28, min(bright.shape[0] // 2, int(box_h * 1.16)))
        else:
            axis_y = max(24, min(bright.shape[0] // 2, int(box_h * 0.82)))
    else:
        axis_y = max(26, min(bright.shape[0] // 2, int(box_h * 0.94)))
    cv2.ellipse(local_ellipse, (local_cx, local_cy), (axis_x, axis_y), 0, 0, 360, 255, -1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    component_mask = np.zeros_like(bright)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_edge = x <= 0 or y <= 0 or (x + w) >= bright.shape[1] or (y + h) >= bright.shape[0]
        if touches_edge or area < 32:
            continue
        component = labels == label
        if np.any(search_seed[component] > 0):
            component_mask[component] = 255

    if not np.any(component_mask):
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            touches_edge = x <= 0 or y <= 0 or (x + w) >= bright.shape[1] or (y + h) >= bright.shape[0]
            if touches_edge or area < 32:
                continue
            component_mask[labels == label] = 255

    legacy_mask = _extract_white_balloon_mask_legacy(image_rgb, bbox)
    legacy_local = None
    if isinstance(legacy_mask, np.ndarray) and np.any(legacy_mask):
        legacy_local = legacy_mask[ry1:ry2, rx1:rx2].copy()
        legacy_local = cv2.morphologyEx(
            legacy_local,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )

    if not np.any(component_mask):
        if legacy_local is not None and np.any(legacy_local):
            mask[ry1:ry2, rx1:rx2] = _fill_internal_mask_holes(legacy_local)
            return mask
        mask[ry1:ry2, rx1:rx2] = _fill_internal_mask_holes(local_ellipse)
        return mask

    component_mask = cv2.morphologyEx(
        component_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    )
    component_mask = cv2.dilate(
        component_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    ellipse_core = cv2.erode(
        local_ellipse,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if np.count_nonzero(component_mask) < int(np.count_nonzero(ellipse_core) * 0.7):
        component_mask = cv2.bitwise_or(component_mask, ellipse_core)
    if legacy_local is not None and np.any(legacy_local):
        guard = cv2.dilate(
            legacy_local,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        complemented = cv2.bitwise_and(component_mask, guard)
        component_mask = cv2.bitwise_or(legacy_local, complemented)
    else:
        component_mask = cv2.bitwise_and(component_mask, local_ellipse)

    mask[ry1:ry2, rx1:rx2] = _fill_internal_mask_holes(component_mask)
    return mask


def _extract_white_balloon_mask_legacy(image_rgb: np.ndarray, bbox: list[int]) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    height, width = image_rgb.shape[:2]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = max(20, int(box_w * 0.9))
    pad_y = max(20, int(box_h * 1.0))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(width, x2 + pad_x)
    ry2 = min(height, y2 + pad_y)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi = image_rgb[ry1:ry2, rx1:rx2]
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
        if area < int(bbox_area * 1.4) or area > int(bbox_area * 30):
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
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = component
    return full_mask


def _apply_white_text_overlay(image_rgb: np.ndarray, bbox: list[int]) -> np.ndarray:
    result = image_rgb.copy()
    height, width = result.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return result

    pad_x = max(4, int((x2 - x1) * 0.08))
    pad_y = max(5, int((y2 - y1) * 0.20))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(width, x2 + pad_x)
    ry2 = min(height, y2 + pad_y)
    patch = result[ry1:ry2, rx1:rx2].copy()
    radius = max(3, min(rx2 - rx1, ry2 - ry1) // 5)
    rounded_mask = _build_rounded_rect_mask(ry2 - ry1, rx2 - rx1, radius)
    patch[rounded_mask > 0] = 255
    result[ry1:ry2, rx1:rx2] = patch
    return result


def _apply_letter_white_boxes(image_rgb: np.ndarray, text_item: dict) -> np.ndarray:
    result = image_rgb.copy()
    bbox = text_item.get("bbox") or [0, 0, 0, 0]
    text = str(text_item.get("text", "") or "")
    if not text.strip():
        return result

    x1, y1, x2, y2 = [int(v) for v in bbox]
    height, width = result.shape[:2]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return result

    char_count = max(1, sum(1 for ch in text if not ch.isspace()))
    region = result[y1:y2, x1:x2]
    if region.size == 0:
        return result
    sample_bbox = _expand_bbox([x1, y1, x2, y2], result.shape, pad_x_ratio=0.22, pad_y_ratio=0.45, min_pad_x=8, min_pad_y=8)
    sx1, sy1, sx2, sy2 = sample_bbox
    sample_region = result[sy1:sy2, sx1:sx2]
    if sample_region.size == 0:
        sample_region = region
    region_gray = cv2.cvtColor(sample_region, cv2.COLOR_RGB2GRAY)
    bright_background_hint = max(
        float(np.mean(region_gray)),
        float(np.percentile(region_gray, 80)),
    )
    if bright_background_hint < 210.0:
        return result

    step = max(1.0, (x2 - x1) / char_count)
    current_x = float(x1)
    for ch in text:
        if ch.isspace():
            current_x += step
            continue
        rx1 = max(0, int(round(current_x - step * 0.08)))
        rx2 = min(width, int(round(current_x + step * 0.88)))
        ry1 = max(0, y1 - 2)
        ry2 = min(height, y2 + 2)
        if rx2 > rx1 and ry2 > ry1:
            patch = result[ry1:ry2, rx1:rx2].copy()
            radius = max(2, min(rx2 - rx1, ry2 - ry1) // 4)
            rounded_mask = _build_rounded_rect_mask(ry2 - ry1, rx2 - rx1, radius)
            patch[rounded_mask > 0] = 255
            result[ry1:ry2, rx1:rx2] = patch
        current_x += step
    return result


def _build_balloon_ellipse_mask(image_shape: tuple[int, int] | tuple[int, int, int], bbox: list[int]) -> np.ndarray:
    if len(image_shape) == 3:
        height, width = image_shape[:2]
    else:
        height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return mask

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    aspect = float(box_w) / float(max(1, box_h))
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    axis_x = max(24, int(box_w * 0.58))
    if aspect >= 2.0:
        axis_y = max(20, int(box_h * 0.72))
    else:
        axis_y = max(24, int(box_h * 0.86))
    cv2.ellipse(mask, (cx, cy), (axis_x, axis_y), 0, 0, 360, 255, -1)
    return mask


def _build_rounded_rect_mask(height: int, width: int, radius: int) -> np.ndarray:
    mask = np.zeros((max(0, height), max(0, width)), dtype=np.uint8)
    if height <= 0 or width <= 0:
        return mask

    radius = max(0, min(int(radius), (width - 1) // 2, (height - 1) // 2))
    if radius <= 0:
        mask[:, :] = 255
        return mask

    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius - 1, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius - 1), radius, 255, -1)
    cv2.circle(mask, (width - radius - 1, height - radius - 1), radius, 255, -1)
    return mask


def _apply_white_balloon_fill(image_rgb: np.ndarray, bbox: list[int]) -> np.ndarray:
    result = image_rgb.copy()
    balloon_mask = _extract_white_balloon_fill_mask(image_rgb, bbox)
    ellipse_mask = _build_balloon_ellipse_mask(result.shape, bbox)
    if np.any(balloon_mask):
        balloon_mask = cv2.bitwise_and(balloon_mask, ellipse_mask)
    else:
        balloon_mask = ellipse_mask
    if not np.any(balloon_mask):
        return _apply_white_text_overlay(result, bbox)
    mask_binary = (balloon_mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    preserve_band = ((distance > 0.0) & (distance <= 2.0)).astype(np.uint8) * 255
    fill_mask = cv2.bitwise_and(balloon_mask, cv2.bitwise_not(preserve_band))
    result[fill_mask > 0] = 255
    if np.any(preserve_band):
        result[preserve_band > 0] = image_rgb[preserve_band > 0]
    return result


def _apply_white_balloon_artifact_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts:
        return result

    text_items = [dict(text) for text in texts]
    clusters = _group_text_indices_by_balloon(text_items, gap_x=84, gap_y=72)
    if not clusters:
        return result

    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    cleaned_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    for cluster in clusters:
        cluster_mask = np.zeros(original_gray.shape, dtype=np.uint8)
        has_white_balloon = False

        for index in cluster:
            bbox = text_items[index].get("bbox") or [0, 0, 0, 0]
            if len(bbox) != 4 or not _is_white_balloon_region(original_rgb, bbox):
                continue
            has_white_balloon = True
            balloon_mask = _extract_white_balloon_fill_mask(original_rgb, bbox)
            if not np.any(balloon_mask):
                legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, bbox)
                if isinstance(legacy_mask, np.ndarray):
                    balloon_mask = legacy_mask
            if np.any(balloon_mask):
                cluster_mask = np.maximum(cluster_mask, balloon_mask.astype(np.uint8))

        if not has_white_balloon or not np.any(cluster_mask):
            continue

        cluster_mask = cv2.morphologyEx(
            cluster_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        balloon_binary = (cluster_mask > 0).astype(np.uint8)
        distance = cv2.distanceTransform(balloon_binary, cv2.DIST_L2, 5)
        interior = (distance > 6.0).astype(np.uint8) * 255
        if not np.any(interior):
            interior = (distance > 3.5).astype(np.uint8) * 255
        if not np.any(interior):
            continue

        original_inside = original_gray[interior > 0]
        if original_inside.size == 0:
            continue

        white_level = float(np.percentile(original_inside, 75))
        dark_threshold = min(210.0, white_level - 22.0)
        if dark_threshold < 150.0:
            dark_threshold = 150.0

        artifact_mask = (
            (cleaned_gray.astype(np.float32) <= dark_threshold)
            & (interior > 0)
        ).astype(np.uint8) * 255
        if not np.any(artifact_mask):
            continue

        artifact_mask = cv2.morphologyEx(
            artifact_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        artifact_mask = cv2.dilate(
            artifact_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        artifact_mask = cv2.bitwise_and(artifact_mask, interior)
        if not np.any(artifact_mask):
            continue

        result = cv2.inpaint(result, artifact_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        cleaned_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    return result


def _restore_textured_balloon_borders(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    expanded_mask: np.ndarray | None,
    texts: list[dict],
) -> np.ndarray:
    """Restaura bordas de balões texturizados que o inpainter corrompeu.

    Para cada balão texturizado, faz blending suave na borda da máscara
    para mesclar o resultado do inpainter (centro) com a imagem original (bordas),
    evitando manchas brancas que o inpainter deixa nas bordas.
    """
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts or expanded_mask is None:
        return result

    for text in texts:
        bbox = text.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4:
            continue

        # Só restaurar bordas de balões texturizados (não brancos)
        if _is_white_balloon_region(original_rgb, bbox):
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        height, width = result.shape[:2]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # Criar máscara local do balão
        local_mask = expanded_mask[y1:y2, x1:x2]
        if not np.any(local_mask):
            continue

        # Erodir a máscara para obter apenas o centro (onde o texto estava)
        box_w = x2 - x1
        box_h = y2 - y1
        erode_size = max(3, min(box_w, box_h) // 6)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        core_mask = cv2.erode(local_mask, erode_kernel, iterations=1)

        # Zona de transição: borda entre core e mask completa
        border_zone = cv2.subtract(local_mask, core_mask)
        if not np.any(border_zone):
            continue

        # Na zona de transição, fazer blend entre inpainter e original
        # Usar distanceTransform para gradiente suave
        dist = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
        max_dist = float(dist.max()) if dist.max() > 0 else 1.0
        # Alpha: 0 na borda externa, 1 no centro
        alpha = np.clip(dist / max(1.0, erode_size * 1.5), 0.0, 1.0)

        # Aplicar blend só na zona de transição
        for c in range(3):
            orig_patch = original_rgb[y1:y2, x1:x2, c].astype(np.float32)
            clean_patch = result[y1:y2, x1:x2, c].astype(np.float32)
            blended = clean_patch * alpha + orig_patch * (1.0 - alpha)
            # Aplicar blend apenas onde a máscara está ativa
            mask_bool = local_mask > 0
            result_patch = result[y1:y2, x1:x2, c].copy()
            result_patch[mask_bool] = blended[mask_bool].astype(np.uint8)
            result[y1:y2, x1:x2, c] = result_patch

    return result


def _extract_textured_balloon_support_mask(
    image_rgb: np.ndarray,
    text_item: dict,
) -> np.ndarray | None:
    seed_bbox = text_item.get("balloon_bbox") or text_item.get("bbox") or [0, 0, 0, 0]
    text_bbox = text_item.get("bbox") or seed_bbox
    if len(seed_bbox) != 4 or len(text_bbox) != 4:
        return None

    try:
        from inpainter.classical import _extract_textured_balloon_mask
    except Exception:
        return None

    region = {
        "bbox": [int(v) for v in seed_bbox],
        "tipo": text_item.get("tipo", "fala"),
        "texts": [
            {
                "bbox": [int(v) for v in text_bbox],
                "confidence": float(text_item.get("confidence", 0.0)),
            }
        ],
    }
    mask = _extract_textured_balloon_mask(image_rgb, region["bbox"], region)
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return None

    box_w = max(1, int(text_bbox[2]) - int(text_bbox[0]))
    box_h = max(1, int(text_bbox[3]) - int(text_bbox[1]))

    outer_pad_x = max(16, int(box_w * 0.28))
    outer_pad_top = max(20, int(box_h * 0.78))
    outer_pad_bottom = max(12, int(box_h * 0.34))
    outer_bbox = [
        max(0, int(text_bbox[0]) - outer_pad_x),
        max(0, int(text_bbox[1]) - outer_pad_top),
        min(image_rgb.shape[1], int(text_bbox[2]) + outer_pad_x),
        min(image_rgb.shape[0], int(text_bbox[3]) + outer_pad_bottom),
    ]
    gx1, gy1, gx2, gy2 = outer_bbox
    if gx2 <= gx1 or gy2 <= gy1:
        return None

    outer_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    outer_w = max(1, gx2 - gx1)
    outer_h = max(1, gy2 - gy1)
    outer_cx = int((gx1 + gx2) / 2)
    outer_cy = int((gy1 + gy2) / 2 - box_h * 0.08)
    outer_axis_x = max(20, int(outer_w * 0.45))
    outer_axis_y = max(18, int(outer_h * 0.50))
    cv2.ellipse(outer_mask, (outer_cx, outer_cy), (outer_axis_x, outer_axis_y), 0, 0, 360, 255, -1)

    inner_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    inner_cx = int((int(text_bbox[0]) + int(text_bbox[2])) / 2)
    inner_cy = int((int(text_bbox[1]) + int(text_bbox[3])) / 2 - box_h * 0.04)
    inner_axis_x = max(12, int(box_w * 0.52))
    inner_axis_y = max(10, int(box_h * 0.85))
    cv2.ellipse(inner_mask, (inner_cx, inner_cy), (inner_axis_x, inner_axis_y), 0, 0, 360, 255, -1)

    mask = cv2.bitwise_and(mask.astype(np.uint8), outer_mask)
    if not np.any(mask):
        return None

    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    if np.any(opened):
        mask = opened

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        filtered = np.zeros_like(mask)
        for label in range(1, num_labels):
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            if not np.any(cv2.bitwise_and(component, inner_mask)):
                continue
            filtered = np.maximum(filtered, component)
        if np.any(filtered):
            mask = filtered

    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    mask = cv2.bitwise_and(mask, outer_mask)
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return mask


def _sample_patch_median_rgb(
    image_rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int] | None:
    height, width = image_rgb.shape[:2]
    x1 = max(0, min(width, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height, int(y1)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    patch = image_rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    pixels = patch.reshape(-1, 3)
    if len(pixels) == 0:
        return None
    return tuple(int(c) for c in np.median(pixels, axis=0))


def _apply_textured_balloon_seam_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts:
        return result

    seam_mask = np.zeros(result.shape[:2], dtype=np.uint8)
    for text in texts:
        if text.get("skip_processing"):
            continue
        bbox = text.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4:
            continue
        if _is_white_balloon_region(original_rgb, bbox):
            continue

        x1, y1, x2, y2 = _expand_bbox(
            [int(v) for v in bbox],
            result.shape,
            pad_x_ratio=0.06,
            pad_y_ratio=0.10,
            min_pad_x=14,
            min_pad_y=12,
        )
        if x2 <= x1 or y2 <= y1:
            continue

        rect_mask = np.zeros(result.shape[:2], dtype=np.uint8)
        rect_mask[y1:y2, x1:x2] = 255
        candidate = _build_mask_boundary_seam_mask(result, rect_mask)
        if not np.any(candidate):
            continue

        support_mask = _extract_textured_balloon_support_mask(original_rgb, text)
        if isinstance(support_mask, np.ndarray) and np.any(support_mask):
            candidate = cv2.bitwise_and(candidate, support_mask.astype(np.uint8))
            if not np.any(candidate):
                continue

        seam_mask = np.maximum(seam_mask, candidate)

    if not np.any(seam_mask):
        return result

    return cv2.inpaint(result, seam_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def _apply_textured_balloon_band_artifact_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts:
        return result

    try:
        from inpainter.classical import _expand_overlay_bbox
    except Exception:
        return result

    height, width = result.shape[:2]
    for text in texts:
        if text.get("skip_processing"):
            continue

        text_bbox = text.get("bbox") or [0, 0, 0, 0]
        balloon_bbox = text.get("balloon_bbox") or [0, 0, 0, 0]
        if len(text_bbox) != 4 or len(balloon_bbox) != 4:
            continue
        if _is_white_balloon_region(original_rgb, text_bbox):
            continue

        bx1, by1, bx2, by2 = [int(v) for v in balloon_bbox]
        bx1 = max(0, min(width, bx1))
        bx2 = max(0, min(width, bx2))
        by1 = max(0, min(height, by1))
        by2 = max(0, min(height, by2))
        if bx2 <= bx1 or by2 <= by1:
            continue

        ox1, oy1, ox2, oy2 = _expand_overlay_bbox(
            [int(v) for v in text_bbox],
            image_width=width,
            image_height=height,
            confidence=float(text.get("confidence", 0.0)),
        )
        if ox2 <= ox1 or oy2 <= oy1:
            continue

        sample_pad = max(12, int((ox2 - ox1) * 0.05))
        sx1 = max(bx1, ox1 + sample_pad)
        sx2 = min(bx2, ox2 - sample_pad)
        top_color = _sample_patch_median_rgb(original_rgb, sx1, by1, sx2, oy1)
        bottom_color = _sample_patch_median_rgb(original_rgb, sx1, oy2, sx2, by2)
        if top_color is None or bottom_color is None:
            continue

        overlay_mask = np.zeros(result.shape[:2], dtype=np.uint8)
        overlay_mask[oy1:oy2, ox1:ox2] = 255

        balloon_core = np.zeros(result.shape[:2], dtype=np.uint8)
        center_x = int((bx1 + bx2) / 2)
        center_y = int((by1 + by2) / 2)
        axis_x = max(20, int((bx2 - bx1) * 0.43))
        axis_y = max(20, int((by2 - by1) * 0.37))
        cv2.ellipse(balloon_core, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)

        repair_mask = cv2.bitwise_and(overlay_mask, balloon_core)
        if not np.any(repair_mask):
            continue

        ys, xs = np.where(repair_mask > 0)
        ry1, ry2 = int(ys.min()), int(ys.max()) + 1
        rx1, rx2 = int(xs.min()), int(xs.max()) + 1
        roi_mask = repair_mask[ry1:ry2, rx1:rx2]
        roi_rgb = result[ry1:ry2, rx1:rx2]
        if roi_rgb.size == 0:
            continue

        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        row_means: list[float] = []
        for row_index in range(roi_gray.shape[0]):
            row_pixels = roi_gray[row_index][roi_mask[row_index] > 0]
            row_means.append(float(np.mean(row_pixels)) if row_pixels.size else 0.0)

        if len(row_means) < 12:
            continue

        row_profile = np.asarray(row_means, dtype=np.float32)
        row_profile = np.convolve(
            row_profile,
            np.array([0.25, 0.5, 0.25], dtype=np.float32),
            mode="same",
        )
        row_diffs = np.diff(row_profile)
        if row_diffs.size == 0:
            continue

        drop_index = int(np.argmin(row_diffs))
        drop_value = float(row_diffs[drop_index])
        if drop_value > -10.0:
            continue
        if drop_index < int(roi_mask.shape[0] * 0.10) or drop_index > int(roi_mask.shape[0] * 0.82):
            continue

        top_band = float(np.mean(row_profile[max(0, drop_index - 4) : drop_index + 1]))
        bottom_band = float(np.mean(row_profile[drop_index + 1 : min(len(row_profile), drop_index + 6)]))
        if (top_band - bottom_band) < 10.0:
            continue

        alpha = roi_mask.astype(np.float32) / 255.0
        soft_alpha = cv2.GaussianBlur(alpha, (31, 31), 0)
        alpha = np.maximum(alpha, np.clip(soft_alpha * 1.15, 0.0, 1.0)) * 0.62

        yy = np.indices((roi_rgb.shape[0], roi_rgb.shape[1]), dtype=np.float32)[0]
        ty = yy / max(1, roi_rgb.shape[0] - 1)
        gradient = (
            np.array(top_color, dtype=np.float32)[None, None, :] * (1.0 - ty[..., None])
            + np.array(bottom_color, dtype=np.float32)[None, None, :] * ty[..., None]
        )
        blended = roi_rgb.astype(np.float32) * (1.0 - alpha[..., None]) + gradient * alpha[..., None]
        result[ry1:ry2, rx1:rx2] = blended.clip(0, 255).astype(np.uint8)

    return result


def _apply_white_balloon_line_artifact_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts:
        return result

    text_items = [dict(text) for text in texts]
    clusters = _group_text_indices_by_balloon(text_items, gap_x=84, gap_y=72)
    if not clusters:
        return result

    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    cleaned_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    for cluster in clusters:
        cluster_mask = np.zeros(cleaned_gray.shape, dtype=np.uint8)
        cluster_bbox = None
        has_white_balloon = False

        for index in cluster:
            bbox = text_items[index].get("bbox") or [0, 0, 0, 0]
            if len(bbox) != 4:
                continue
            if not _is_white_balloon_region(original_rgb, bbox):
                continue
            balloon_mask = _extract_white_balloon_fill_mask(original_rgb, bbox)
            if not np.any(balloon_mask):
                legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, bbox)
                if isinstance(legacy_mask, np.ndarray):
                    balloon_mask = legacy_mask
            if not np.any(balloon_mask):
                continue

            has_white_balloon = True
            cluster_bbox = bbox if cluster_bbox is None else _bbox_union(cluster_bbox, bbox)
            cluster_mask = np.maximum(cluster_mask, balloon_mask.astype(np.uint8))

        if not has_white_balloon or cluster_bbox is None or not np.any(cluster_mask):
            continue

        cluster_mask = cv2.morphologyEx(
            cluster_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        distance = cv2.distanceTransform((cluster_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        interior = (distance > 7.0).astype(np.uint8) * 255
        if not np.any(interior):
            interior = (distance > 4.0).astype(np.uint8) * 255
        if not np.any(interior):
            continue

        x1, y1, x2, y2 = [int(v) for v in cluster_bbox]
        cluster_w = max(1, x2 - x1)
        cluster_h = max(1, y2 - y1)
        local_mean = cv2.blur(cleaned_gray, (31, 31))
        relative_dark = ((local_mean.astype(np.int16) - cleaned_gray.astype(np.int16)) >= 18).astype(np.uint8) * 255
        absolute_dark = (cleaned_gray <= 228).astype(np.uint8) * 255
        candidate = cv2.bitwise_and(cv2.bitwise_and(relative_dark, absolute_dark), interior)

        kernel_w = max(21, int(cluster_w * 0.24))
        if kernel_w % 2 == 0:
            kernel_w += 1
        horizontal = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1)),
            iterations=1,
        )
        if not np.any(horizontal):
            continue

        line_mask = np.zeros_like(horizontal)
        min_width = max(24, int(cluster_w * 0.22))
        max_height = max(7, int(cluster_h * 0.14))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
        for label in range(1, num_labels):
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if width < min_width or height > max_height:
                continue
            line_mask[labels == label] = 255

        if not np.any(line_mask):
            continue

        line_mask = cv2.dilate(
            line_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
            iterations=1,
        )
        line_mask = cv2.bitwise_and(line_mask, interior)
        if not np.any(line_mask):
            continue

        result = cv2.inpaint(result, line_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        cleaned_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    return result


def _merge_component_boxes_into_lines(boxes: list[list[int]], gap_x: int = 10, gap_y: int = 4) -> list[list[int]]:
    pending = [list(box) for box in boxes if box and len(box) == 4]
    merged: list[list[int]] = []

    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            next_pending = []
            for other in pending:
                horiz_gap, vert_gap = _bbox_gaps(current, other)
                horiz_overlap = min(current[2], other[2]) - max(current[0], other[0])
                vert_overlap = min(current[3], other[3]) - max(current[1], other[1])
                same_line = (
                    vert_overlap >= -gap_y and horiz_gap <= gap_x
                ) or (
                    horiz_overlap > 0 and vert_gap <= gap_y
                )
                if same_line:
                    current = _bbox_union(current, other)
                    changed = True
                else:
                    next_pending.append(other)
            pending = next_pending
        merged.append(current)

    return sorted(merged, key=lambda box: (box[1], box[0]))


def _cluster_component_boxes_by_rows(boxes: list[list[int]], gap_y: int = 8) -> list[list[list[int]]]:
    rows: list[list[list[int]]] = []
    for box in sorted([list(box) for box in boxes if box and len(box) == 4], key=lambda item: ((item[1] + item[3]) / 2.0, item[0])):
        box_cy = (float(box[1]) + float(box[3])) / 2.0
        attached = False
        for row in rows:
            row_top = min(item[1] for item in row)
            row_bottom = max(item[3] for item in row)
            row_cy = (float(row_top) + float(row_bottom)) / 2.0
            if abs(box_cy - row_cy) <= float(gap_y) or (box[1] <= row_bottom + gap_y and box[3] >= row_top - gap_y):
                row.append(box)
                attached = True
                break
        if not attached:
            rows.append([box])
    return rows


def _extract_white_balloon_text_boxes(image_rgb: np.ndarray, bbox: list[int]) -> list[list[int]]:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return []

    balloon_mask = _extract_white_balloon_fill_mask(image_rgb, [x1, y1, x2, y2])
    if not np.any(balloon_mask):
        legacy_mask = _extract_white_balloon_mask_legacy(image_rgb, [x1, y1, x2, y2])
        if isinstance(legacy_mask, np.ndarray):
            balloon_mask = legacy_mask
    if not np.any(balloon_mask):
        return []

    expanded = _expand_bbox([x1, y1, x2, y2], image_rgb.shape, pad_x_ratio=0.08, pad_y_ratio=0.16, min_pad_x=4, min_pad_y=4)
    rx1, ry1, rx2, ry2 = expanded
    crop = image_rgb[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return []

    local_balloon = balloon_mask[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    local_mean = cv2.blur(gray, (21, 21))
    search = np.zeros_like(gray, dtype=np.uint8)
    sx1 = max(0, x1 - rx1)
    sy1 = max(0, y1 - ry1)
    sx2 = min(gray.shape[1], x2 - rx1)
    sy2 = min(gray.shape[0], y2 - ry1)
    search[sy1:sy2, sx1:sx2] = 255
    search = cv2.dilate(search, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    relative_dark = ((local_mean.astype(np.int16) - gray.astype(np.int16)) >= 18).astype(np.uint8) * 255
    absolute_dark = (gray <= 212).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(cv2.bitwise_or(relative_dark, absolute_dark), search)
    candidate = cv2.bitwise_and(candidate, local_balloon.astype(np.uint8))
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    if not np.any(candidate):
        return []

    component_boxes: list[list[int]] = []
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    max_component_h = max(18, int(bbox_h * 0.20))
    max_component_area = max(1600, int(bbox_w * bbox_h * 0.028))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    min_area = max(4, int((x2 - x1) * (y2 - y1) * 0.0007))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area or w < 2 or h < 2:
            continue
        if h > max_component_h or area > max_component_area:
            continue
        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        component_boxes.append([rx1 + bx, ry1 + by, rx1 + bx + w, ry1 + by + h])

    if not component_boxes:
        return []

    component_heights = [max(1, box[3] - box[1]) for box in component_boxes]
    median_height = int(np.median(np.asarray(component_heights, dtype=np.int32))) if component_heights else 8
    row_gap_y = max(4, int(median_height * 0.9))
    line_gap_x = max(8, int(median_height * 0.95))
    row_groups = _cluster_component_boxes_by_rows(component_boxes, gap_y=row_gap_y)

    merged_lines: list[list[int]] = []
    for row in row_groups:
        merged_lines.extend(_merge_component_boxes_into_lines(row, gap_x=line_gap_x, gap_y=max(2, int(median_height * 0.35))))
    return sorted(merged_lines, key=lambda box: (box[1], box[0]))


def _apply_white_balloon_text_box_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts:
        return result

    for text in texts:
        if text.get("skip_processing"):
            continue
        bbox = text.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4:
            continue
        if not _is_white_balloon_region(original_rgb, bbox):
            continue
        balloon_mask = _extract_white_balloon_fill_mask(original_rgb, bbox)
        if not np.any(balloon_mask):
            legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, bbox)
            if isinstance(legacy_mask, np.ndarray):
                balloon_mask = legacy_mask
        if np.any(balloon_mask):
            distance = cv2.distanceTransform((balloon_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
            interior = (distance > 3.0).astype(np.uint8) * 255
            if not np.any(interior):
                interior = (distance > 1.5).astype(np.uint8) * 255
        else:
            interior = np.zeros(result.shape[:2], dtype=np.uint8)
        boxes = _extract_white_balloon_text_boxes(original_rgb, bbox)
        for bx1, by1, bx2, by2 in boxes:
            bx1 = max(0, min(result.shape[1], int(bx1)))
            bx2 = max(0, min(result.shape[1], int(bx2)))
            by1 = max(0, min(result.shape[0], int(by1)))
            by2 = max(0, min(result.shape[0], int(by2)))
            if bx2 <= bx1 or by2 <= by1:
                continue
            radius = max(2, min(bx2 - bx1, by2 - by1) // 4)
            rounded_mask = _build_rounded_rect_mask(by2 - by1, bx2 - bx1, radius) > 0
            if np.any(interior):
                clipped = interior[by1:by2, bx1:bx2]
                overlap_mask = (clipped > 0) & rounded_mask
                ys, xs = np.where(overlap_mask)
                if len(xs) == 0:
                    continue
                overlap_area = int(np.count_nonzero(overlap_mask))
                box_area = max(1, (bx2 - bx1) * (by2 - by1))
                if overlap_area < max(4, int(box_area * 0.08)):
                    continue
                patch = result[by1:by2, bx1:bx2].copy()
                patch[overlap_mask] = 255
                result[by1:by2, bx1:bx2] = patch
            else:
                patch = result[by1:by2, bx1:bx2].copy()
                patch[rounded_mask] = 255
                result[by1:by2, bx1:bx2] = patch

    return result


def _apply_white_balloon_micro_artifact_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts:
        return result

    text_items = [dict(text) for text in texts]
    clusters = _group_text_indices_by_balloon(text_items, gap_x=84, gap_y=72)
    if not clusters:
        return result

    cleaned_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    for cluster in clusters:
        cluster_mask = np.zeros(cleaned_gray.shape, dtype=np.uint8)
        cluster_bbox = None

        for index in cluster:
            bbox = text_items[index].get("bbox") or [0, 0, 0, 0]
            if len(bbox) != 4:
                continue
            if not _is_white_balloon_region(original_rgb, bbox):
                continue
            balloon_mask = _extract_white_balloon_fill_mask(original_rgb, bbox)
            if not np.any(balloon_mask):
                legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, bbox)
                if isinstance(legacy_mask, np.ndarray):
                    balloon_mask = legacy_mask
            if not np.any(balloon_mask):
                continue

            cluster_bbox = bbox if cluster_bbox is None else _bbox_union(cluster_bbox, bbox)
            cluster_mask = np.maximum(cluster_mask, balloon_mask.astype(np.uint8))

        if cluster_bbox is None or not np.any(cluster_mask):
            continue

        distance = cv2.distanceTransform((cluster_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        interior = (distance > 8.0).astype(np.uint8) * 255
        if not np.any(interior):
            interior = (distance > 5.0).astype(np.uint8) * 255
        if not np.any(interior):
            continue

        x1, y1, x2, y2 = [int(v) for v in cluster_bbox]
        cluster_w = max(1, x2 - x1)
        cluster_h = max(1, y2 - y1)
        local_mean = cv2.blur(cleaned_gray, (17, 17))
        relative_dark = ((local_mean.astype(np.int16) - cleaned_gray.astype(np.int16)) >= 16).astype(np.uint8) * 255
        absolute_dark = (cleaned_gray <= 218).astype(np.uint8) * 255
        candidate = cv2.bitwise_and(cv2.bitwise_or(relative_dark, absolute_dark), interior)
        if not np.any(candidate):
            continue

        micro_mask = np.zeros_like(candidate)
        max_component_area = max(140, int(cluster_w * cluster_h * 0.006))
        max_component_w = max(18, int(cluster_w * 0.10))
        max_component_h = max(28, int(cluster_h * 0.18))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area <= 0 or area > max_component_area:
                continue
            if width > max_component_w or height > max_component_h:
                continue
            micro_mask[labels == label] = 255

        if not np.any(micro_mask):
            continue

        micro_mask = cv2.dilate(
            micro_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        micro_mask = cv2.bitwise_and(micro_mask, interior)
        if not np.any(micro_mask):
            continue
        result = cv2.inpaint(result, micro_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        cleaned_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    return result


def _build_residual_cleanup_mask(image_rgb: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    if image_rgb.size == 0 or not np.any(base_mask):
        return np.zeros(base_mask.shape, dtype=np.uint8)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    outer = cv2.dilate(
        base_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
        iterations=1,
    )
    inner = cv2.erode(
        base_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    ring = cv2.subtract(outer, inner)
    if not np.any(ring):
        ring = outer

    expanded_core = cv2.dilate(
        base_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=1,
    )
    search_region = cv2.bitwise_or(ring, expanded_core)

    local_mean = cv2.blur(gray, (31, 31))
    dark = ((gray <= 64).astype(np.uint8) * 255)
    relative_dark = (((local_mean.astype(np.int16) - gray.astype(np.int16)) >= 18).astype(np.uint8) * 255)
    candidate_dark = cv2.bitwise_or(dark, relative_dark)
    horizontal = cv2.morphologyEx(
        candidate_dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1)),
        iterations=1,
    )
    vertical = cv2.morphologyEx(
        candidate_dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 13)),
        iterations=1,
    )
    cleanup = cv2.bitwise_or(horizontal, vertical)
    cleanup = cv2.bitwise_and(cleanup, search_region)
    cleanup = cv2.dilate(
        cleanup,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return cleanup


def _build_bright_zone_line_mask(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.size == 0:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    local_mean = cv2.blur(gray, (31, 31))
    dark = ((gray <= 155).astype(np.uint8) * 255)
    bright_zone = ((local_mean >= 205).astype(np.uint8) * 255)
    candidate = cv2.bitwise_and(dark, bright_zone)
    horizontal = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (51, 1)),
        iterations=1,
    )
    horizontal = cv2.dilate(
        horizontal,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        iterations=1,
    )
    return horizontal


def _build_mask_boundary_seam_mask(image_rgb: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    if image_rgb.size == 0 or not np.any(base_mask):
        return np.zeros(base_mask.shape, dtype=np.uint8)

    ys, xs = np.where(base_mask > 0)
    if len(xs) == 0:
        return np.zeros(base_mask.shape, dtype=np.uint8)
    mask_x1, mask_y1, mask_x2, mask_y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    mask_w = max(1, mask_x2 - mask_x1 + 1)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    local_mean = cv2.blur(gray, (41, 41))

    outer = cv2.dilate(
        base_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 9)),
        iterations=1,
    )
    inner = cv2.erode(
        base_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 5)),
        iterations=1,
    )
    boundary_band = cv2.subtract(outer, inner)
    if not np.any(boundary_band):
        return np.zeros(base_mask.shape, dtype=np.uint8)

    relative_dark = (((local_mean.astype(np.int16) - gray.astype(np.int16)) >= 12).astype(np.uint8) * 255)
    absolute_dark = ((gray <= 145).astype(np.uint8) * 255)
    candidate = cv2.bitwise_or(relative_dark, absolute_dark)
    candidate = cv2.bitwise_and(candidate, boundary_band)
    horizontal = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1)),
        iterations=1,
    )
    horizontal = cv2.dilate(
        horizontal,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        iterations=1,
    )
    if not np.any(horizontal):
        return np.zeros(base_mask.shape, dtype=np.uint8)

    seam_mask = np.zeros_like(horizontal)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
    min_width = max(40, int(mask_w * 0.25))
    edge_margin = 24

    for label in range(1, num_labels):
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        if width < min_width:
            continue
        cy = float(centroids[label][1])
        near_top = abs(cy - float(mask_y1)) <= edge_margin
        near_bottom = abs(cy - float(mask_y2)) <= edge_margin
        if near_top or near_bottom:
            seam_mask[labels == label] = 255

    return seam_mask


def _apply_mask_boundary_seam_cleanup(
    image_rgb: np.ndarray,
    base_mask: np.ndarray,
    debug: DebugRunRecorder | None = None,
) -> np.ndarray:
    seam_mask = _build_mask_boundary_seam_mask(image_rgb, base_mask)
    if not np.any(seam_mask):
        if debug is not None:
            debug.log("seam_cleanup", ran=False, seam_coords=[])
        return image_rgb
    ys, xs = np.where(seam_mask > 0)
    seam_coords = []
    if len(xs) > 0:
        seam_coords.append(
            {
                "x1": int(xs.min()),
                "y1": int(ys.min()),
                "x2": int(xs.max()),
                "y2": int(ys.max()),
            }
        )
    if debug is not None:
        debug.log("seam_cleanup", ran=True, seam_coords=seam_coords)
    return cv2.inpaint(image_rgb, seam_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def _apply_bright_zone_line_cleanup(image_rgb: np.ndarray) -> np.ndarray:
    result = image_rgb.copy()
    line_mask = _build_bright_zone_line_mask(image_rgb)
    if not np.any(line_mask):
        return result

    expanded = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        iterations=1,
    )
    feather = cv2.GaussianBlur(expanded.astype(np.float32), (0, 0), sigmaX=2.2, sigmaY=1.0)
    if float(np.max(feather)) <= 0.0:
        return result

    alpha = np.clip(feather / 255.0, 0.0, 1.0)[..., None]
    local_fill = cv2.blur(image_rgb, (41, 41)).astype(np.float32)
    local_fill = np.maximum(local_fill, 245.0)
    blended = result.astype(np.float32) * (1.0 - alpha) + local_fill * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_post_inpaint_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    final = _apply_textured_balloon_seam_cleanup(
        original_rgb,
        cleaned_rgb,
        texts,
    )
    final = _apply_textured_balloon_band_artifact_cleanup(
        original_rgb,
        final,
        texts,
    )
    final = _apply_white_balloon_line_artifact_cleanup(
        original_rgb,
        final,
        texts,
    )
    final = _apply_white_balloon_text_box_cleanup(
        original_rgb,
        final,
        texts,
    )
    return _apply_white_balloon_micro_artifact_cleanup(
        original_rgb,
        final,
        texts,
    )


def _run_koharu_blockwise_inpaint_page(
    image_np: np.ndarray,
    ocr_data: dict,
    inpainter,
) -> np.ndarray:
    height, width = image_np.shape[:2]
    vision_blocks = list(ocr_data.get("_vision_blocks", []))
    if not vision_blocks:
        return image_np.copy()

    working_mask = vision_blocks_to_mask(
        image_np.shape,
        vision_blocks,
        image_rgb=image_np,
        expand_mask=False,
    ).astype(np.uint8)
    if not np.any(working_mask):
        return image_np.copy()

    inpainted = image_np.copy()
    for block in vision_blocks:
        bbox = [int(v) for v in (block.get("bbox") or [0, 0, 0, 0])]
        x1 = max(0, min(width, bbox[0]))
        y1 = max(0, min(height, bbox[1]))
        x2 = max(0, min(width, bbox[2]))
        y2 = max(0, min(height, bbox[3]))
        if x2 <= x1 or y2 <= y1:
            continue

        window = _enlarge_koharu_window([x1, y1, x2, y2], width, height)
        wx1, wy1, wx2, wy2 = window
        if wx2 <= wx1 or wy2 <= wy1:
            continue

        crop_image = inpainted[wy1:wy2, wx1:wx2].copy()
        crop_mask = working_mask[wy1:wy2, wx1:wx2].copy()
        if not np.any(crop_mask):
            _clear_mask_bbox(working_mask, [x1, y1, x2, y2])
            continue

        filled = _try_koharu_balloon_fill(crop_image, crop_mask)
        if filled is not None:
            output = filled
        else:
            inpaint_result = _run_masked_inpaint_passes(
                inpainter,
                crop_image,
                crop_mask,
                batch_size=4,
                force_no_tiling=True,
            )
            output = inpaint_result["final_output"] if isinstance(inpaint_result, dict) else inpaint_result

        inpainted[wy1:wy2, wx1:wx2] = output
        _clear_mask_bbox(working_mask, [x1, y1, x2, y2])

    return _apply_post_inpaint_cleanup(
        image_np,
        inpainted,
        list(ocr_data.get("texts", [])),
    )


def _run_masked_inpaint_passes(
    inpainter,
    image_np: np.ndarray,
    mask: np.ndarray,
    batch_size: int = 4,
    debug: DebugRunRecorder | None = None,
    seam_cleanup: bool = False,
    multi_pass: bool = False,
    force_no_tiling: bool = True,
) -> dict:
    assert mask.shape[:2] == image_np.shape[:2], (
        f"mask/image mismatch before passes: mask={mask.shape[:2]} image={image_np.shape[:2]}"
    )
    expanded = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )
    if debug is not None:
        debug.log(
            "roi",
            x1=0,
            y1=0,
            x2=int(image_np.shape[1]),
            y2=int(image_np.shape[0]),
            width=int(image_np.shape[1]),
            height=int(image_np.shape[0]),
            resize_width=int(image_np.shape[1]),
            resize_height=int(image_np.shape[0]),
            padding={"top": 0, "bottom": 0, "left": 0, "right": 0},
            shape_before_inpaint=list(image_np.shape),
            shape_after_inpaint=list(image_np.shape),
            shape_before_paste=list(image_np.shape),
            shape_after_paste=list(image_np.shape),
            paste_offsets={"x": 0, "y": 0},
            clamped={"left": False, "top": False, "right": False, "bottom": False},
            passes=1 if not multi_pass else 2,
            seam_cleanup=bool(seam_cleanup),
        )
    fallback_to_legacy = False
    fallback_error = ""
    raw_output = None
    after_paste = None
    cleanup_base_mask = expanded

    if not multi_pass:
        try:
            first_pass = _call_inpainter(
                inpainter,
                image_np,
                expanded,
                batch_size=batch_size,
                debug=debug,
                force_no_tiling=force_no_tiling,
            )
            if first_pass.shape[:2] != image_np.shape[:2]:
                raise ValueError(
                    f"single-pass retornou shape {first_pass.shape[:2]} esperado {image_np.shape[:2]}"
                )
            raw_output = first_pass
            after_paste = first_pass.copy()
        except Exception as exc:
            fallback_to_legacy = True
            fallback_error = str(exc)
            multi_pass = True
            force_no_tiling = False
            seam_cleanup = True
            if debug is not None:
                debug.log("single_pass_fallback", reason=fallback_error)

    if multi_pass:
        first_pass = _call_inpainter(
            inpainter,
            image_np,
            expanded,
            batch_size=batch_size,
            debug=debug,
            force_no_tiling=force_no_tiling,
        )
        second_mask = cv2.dilate(
            expanded,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        second_pass = _call_inpainter(
            inpainter,
            first_pass,
            second_mask,
            batch_size=batch_size,
            debug=debug,
            force_no_tiling=force_no_tiling,
        )
        cleanup_mask = _build_residual_cleanup_mask(second_pass, second_mask)
        if np.any(cleanup_mask):
            third_mask = cv2.dilate(
                cleanup_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            second_pass = _call_inpainter(
                inpainter,
                second_pass,
                third_mask,
                batch_size=batch_size,
                debug=debug,
                force_no_tiling=force_no_tiling,
            )
        raw_output = second_pass
        after_paste = second_pass.copy()
        cleanup_base_mask = second_mask

    after_cleanup = (
        _apply_mask_boundary_seam_cleanup(raw_output, cleanup_base_mask, debug=debug)
        if seam_cleanup
        else raw_output.copy()
    )
    if not seam_cleanup and debug is not None:
        debug.log("seam_cleanup", ran=False, seam_coords=[])
    return {
        "expanded_mask": expanded,
        "raw_output": raw_output,
        "after_roi_paste": after_paste,
        "after_seam_cleanup": after_cleanup,
        "final_output": after_cleanup,
        "cleanup_base_mask": cleanup_base_mask,
        "fallback_to_legacy": fallback_to_legacy,
        "fallback_error": fallback_error,
    }


def _apply_inpainting_round(
    image_np: np.ndarray,
    ocr_data: dict,
    inpainter,
    debug: DebugRunRecorder | None = None,
    seam_cleanup: bool = False,
    multi_pass: bool = False,
    force_no_tiling: bool = True,
) -> np.ndarray | dict:
    vision_blocks = ocr_data.get("_vision_blocks", [])
    full_mask = (
        vision_blocks_to_mask(image_np.shape, vision_blocks, image_rgb=image_np)
        if vision_blocks
        else np.zeros(image_np.shape[:2], dtype=np.uint8)
    )
    if np.any(full_mask):
        if debug is None and not seam_cleanup and not multi_pass and force_no_tiling:
            result = _run_masked_inpaint_passes(
                inpainter,
                image_np,
                full_mask,
                batch_size=4,
            )
        else:
            result = _run_masked_inpaint_passes(
                inpainter,
                image_np,
                full_mask,
                batch_size=4,
                debug=debug,
                seam_cleanup=seam_cleanup,
                multi_pass=multi_pass,
                force_no_tiling=force_no_tiling,
            )
        if debug is not None:
            return result
        if isinstance(result, dict):
            final = result["final_output"]

            # Restaurar bordas de balões texturizados — evita mancha branca do inpainter
            final = _apply_textured_balloon_seam_cleanup(
                image_np,
                final,
                list(ocr_data.get("texts", [])),
            )
            final = _apply_textured_balloon_band_artifact_cleanup(
                image_np,
                final,
                list(ocr_data.get("texts", [])),
            )

            line_cleaned = _apply_white_balloon_line_artifact_cleanup(
                image_np,
                final,
                list(ocr_data.get("texts", [])),
            )
            text_box_cleaned = _apply_white_balloon_text_box_cleanup(
                image_np,
                line_cleaned,
                list(ocr_data.get("texts", [])),
            )
            return _apply_white_balloon_micro_artifact_cleanup(
                image_np,
                text_box_cleaned,
                list(ocr_data.get("texts", [])),
            )
        return result
    else:
        return image_np.copy()


def _select_recovery_match(base_texts: list[dict], recovered_text: dict) -> int | None:
    residual_bbox = recovered_text.get("bbox", [0, 0, 0, 0])
    residual_norm = _normalize_text_key(recovered_text.get("text", ""))
    best_index = None
    best_score = -1e9

    for index, base in enumerate(base_texts):
        if base.get("skip_processing"):
            continue
        base_bbox = base.get("bbox", [0, 0, 0, 0])
        base_norm = _normalize_text_key(base.get("text", ""))
        iou = _bbox_iou(base_bbox, residual_bbox)
        horiz_gap, vert_gap = _bbox_gaps(base_bbox, residual_bbox)
        close_geometry = (
            _bbox_contains_center(base_bbox, residual_bbox, margin=28)
            or iou >= 0.04
            or (horiz_gap <= 52.0 and vert_gap <= 46.0)
        )
        if not close_geometry:
            continue
        if residual_norm and base_norm and (residual_norm == base_norm or residual_norm in base_norm or base_norm in residual_norm):
            if iou >= 0.2 or _bbox_contains_center(base_bbox, residual_bbox, margin=18):
                return index

        base_cx, base_cy = _bbox_center(base_bbox)
        res_cx, res_cy = _bbox_center(residual_bbox)
        distance = math.hypot(base_cx - res_cx, base_cy - res_cy)
        score = -distance
        if _bbox_contains_center(base_bbox, residual_bbox, margin=28):
            score += 220.0
        if iou > 0.02:
            score += iou * 160.0
        vertical_overlap = min(base_bbox[3], residual_bbox[3]) - max(base_bbox[1], residual_bbox[1])
        if vertical_overlap > 0:
            score += 55.0
        if base.get("tipo") == recovered_text.get("tipo"):
            score += 18.0
        if score > best_score:
            best_score = score
            best_index = index

    if best_score >= 20.0:
        return best_index

    clusters = _group_text_indices_by_balloon(base_texts, gap_x=96, gap_y=58)
    residual_bbox = recovered_text.get("bbox", [0, 0, 0, 0])
    for cluster in clusters:
        cluster_bbox = base_texts[cluster[0]].get("_cluster_bbox", base_texts[cluster[0]].get("bbox", [0, 0, 0, 0]))
        if not _bbox_contains_center(cluster_bbox, residual_bbox, margin=96):
            continue
        candidate = min(
            cluster,
            key=lambda idx: abs(_bbox_center(base_texts[idx].get("bbox", [0, 0, 0, 0]))[1] - _bbox_center(residual_bbox)[1]),
        )
        return candidate
    return None


def _integrate_recovery_page(base_page: dict, recovered_page: dict) -> tuple[dict, dict]:
    updated_page = _clone_page_result(base_page)
    recovery_by_index: dict[int, tuple[dict, dict]] = {}
    recovered_texts = recovered_page.get("texts", [])
    recovered_blocks = recovered_page.get("_vision_blocks", [])

    for recovered_text, recovered_block in zip(recovered_texts, recovered_blocks):
        if recovered_text.get("skip_processing"):
            continue
        match_index = _select_recovery_match(updated_page["texts"], recovered_text)
        if match_index is None:
            continue

        target = updated_page["texts"][match_index]
        merged_text = _merge_text_fragments(
            target.get("text", ""),
            recovered_text.get("text", ""),
            target.get("bbox", [0, 0, 0, 0]),
            recovered_text.get("bbox", [0, 0, 0, 0]),
        )
        target["text"] = merged_text
        target["bbox"] = _bbox_union(target.get("bbox", [0, 0, 0, 0]), recovered_text.get("bbox", [0, 0, 0, 0]))
        target["confidence"] = max(float(target.get("confidence", 0.0)), float(recovered_text.get("confidence", 0.0)))
        target["ocr_second_pass"] = True
        if match_index < len(updated_page["_vision_blocks"]):
            updated_page["_vision_blocks"][match_index]["bbox"] = _bbox_union(
                updated_page["_vision_blocks"][match_index].get("bbox", [0, 0, 0, 0]),
                recovered_block.get("bbox", [0, 0, 0, 0]),
            )
            updated_page["_vision_blocks"][match_index]["confidence"] = max(
                float(updated_page["_vision_blocks"][match_index].get("confidence", 0.0)),
                float(recovered_block.get("confidence", 0.0)),
            )

        merged_recovery_text = dict(target)
        merged_recovery_block = (
            dict(updated_page["_vision_blocks"][match_index])
            if match_index < len(updated_page["_vision_blocks"])
            else {"bbox": list(target.get("bbox", [0, 0, 0, 0])), "mask": None, "confidence": float(target.get("confidence", 0.0))}
        )
        recovery_by_index[match_index] = (merged_recovery_text, merged_recovery_block)

    ordered_indices = sorted(recovery_by_index.keys())
    recovery_texts = [recovery_by_index[index][0] for index in ordered_indices]
    recovery_blocks = [recovery_by_index[index][1] for index in ordered_indices]

    recovery_page = {
        "image": recovered_page.get("image", base_page.get("image", "")),
        "width": recovered_page.get("width", base_page.get("width", 0)),
        "height": recovered_page.get("height", base_page.get("height", 0)),
        "texts": recovery_texts,
        "_vision_blocks": recovery_blocks,
    }
    return updated_page, recovery_page


def build_page_result(
    image_path: str,
    image_rgb: np.ndarray,
    blocks: list,
    texts: list[str],
    profile: str = "quality",
    ocr_backend: str = "vision",
    enable_font_detection: bool = False,
    progress_callback=None,
    idioma_origem: str = "en",
) -> dict:
    height, width = image_rgb.shape[:2]
    page_texts = []
    vision_blocks = []
    total_blocks = max(1, len(blocks))

    _emit_stage_progress(progress_callback, "build_blocks", 0.74, "Montando blocos OCR")

    for index, (block, raw_text) in enumerate(zip(blocks, texts), start=1):
        bbox = [int(round(v)) for v in block.xyxy]
        bbox[0] = max(0, min(width, bbox[0]))
        bbox[2] = max(0, min(width, bbox[2]))
        bbox[1] = max(0, min(height, bbox[1]))
        bbox[3] = max(0, min(height, bbox[3]))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        confidence = round(float(getattr(block, "confidence", 0.0)), 3)
        cleaned = fix_ocr_errors(str(raw_text or "").strip(), idioma_origem=idioma_origem)
        if not cleaned:
            continue

        if is_watermark(cleaned):
            continue

        if is_editorial_credit(cleaned):
            continue

        # Ignorar textos não-latinos apenas se a origem for inglês.
        # Se a origem for CJK, devemos manter o texto para tradução.
        if idioma_origem == "en" and is_non_english(cleaned):
            continue

        tipo = classify_text_type(cleaned, bbox, width)
        cleaned = semantic_refine_text(cleaned, tipo=tipo, confidence=confidence)
        if is_editorial_credit(cleaned):
            continue
        if looks_suspicious(cleaned, confidence) and confidence < 0.6:
            continue
        estilo = analyze_style(image_rgb, bbox)
        if _should_use_base_white_balloon_font(image_rgb, bbox):
            estilo["fonte"] = "ComicNeue-Bold.ttf"
        else:
            estilo["fonte"] = "Newrotic.ttf"
            estilo["cor"] = "#FFFFFF"
        estilo["force_upper"] = True
        page_texts.append(
            {
                "text": cleaned,
                "bbox": bbox,
                "confidence": confidence,
                "tipo": tipo,
                "estilo": estilo,
                "ocr_source": f"vision-{ocr_backend}",
                "ocr_reviewed": False,
                "ocr_profile": profile,
                "ocr_semantic_reviewed": False,
                "ocr_mode": ocr_backend,
                "skip_processing": False,
            }
        )
        vision_blocks.append(_serialize_block(block, (height, width)))
        finalize_progress = 0.90 + (index / total_blocks) * 0.08
        _emit_stage_progress(progress_callback, "finalize_blocks", finalize_progress, "Finalizando blocos OCR")

    return {
        "image": image_path,
        "width": width,
        "height": height,
        "texts": page_texts,
        "_vision_blocks": vision_blocks,
    }


def _run_detect_ocr_on_image(
    image_rgb: np.ndarray,
    image_label: str,
    profile: str = "quality",
    progress_callback=None,
    idioma_origem: str = "en",
) -> dict:
    _emit_stage_progress(progress_callback, "load_detector", 0.08, "Carregando detector de texto")
    detector = _get_detector(profile)
    _emit_stage_progress(progress_callback, "load_ocr_engine", 0.18, "Carregando motor de OCR")
    ocr = _get_ocr_engine(profile, lang=idioma_origem)
    _emit_stage_progress(progress_callback, "detect_text", 0.38, "Detectando regioes de texto")
    blocks = detector.detect(image_rgb, conf_threshold=_profile_to_detection_threshold(profile))
    backend_name = getattr(ocr, "_backend", getattr(ocr, "model_name", "vision"))
    recognize_message = (
        f"Reconhecendo {len(blocks)} bloco(s) de texto" if blocks else "Nenhum texto detectado"
    )
    _emit_stage_progress(progress_callback, "recognize_text", 0.62, recognize_message)
    # PaddleOCR: por padrão, roda 1 OCR na página inteira e mapeia as linhas aos blocos
    # (bem mais rápido que rodar detecção+OCR dentro de cada crop).
    # Desative com `TRADUZAI_PADDLE_FULL_PAGE=0` se precisar diagnosticar regressões.
    # (Mantém compat com `MANGATL_PADDLE_FULL_PAGE`.)
    paddle_full_page_flag = (
        os.getenv("TRADUZAI_PADDLE_FULL_PAGE")
        or os.getenv("MANGATL_PADDLE_FULL_PAGE")
        or "1"
    )
    enable_paddle_full_page = str(paddle_full_page_flag).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if blocks and backend_name == "paddleocr" and enable_paddle_full_page and hasattr(ocr, "recognize_blocks_from_page"):
        texts = ocr.recognize_blocks_from_page(image_rgb, blocks)
    else:
        crops = [detector.crop(image_rgb, block) for block in blocks]
        texts = ocr.recognize_batch(crops) if crops else []
    return build_page_result(
        image_path=image_label,
        image_rgb=image_rgb,
        blocks=blocks,
        texts=texts,
        profile=profile,
        ocr_backend=backend_name,
        enable_font_detection=True,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
    )


def vision_blocks_to_mask(
    image_shape: tuple[int, int, int] | tuple[int, int],
    vision_blocks: list[dict],
    image_rgb: np.ndarray | None = None,
    expand_mask: bool = True,
) -> np.ndarray:
    if len(image_shape) == 3:
        height, width = image_shape[:2]
    else:
        height, width = image_shape

    mask = np.zeros((height, width), dtype=np.uint8)
    for block in vision_blocks:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        local_mask = block.get("mask")
        if isinstance(local_mask, np.ndarray) and local_mask.size > 0:
            if local_mask.shape == (height, width):
                mask = np.maximum(mask, local_mask.astype(np.uint8))
            else:
                target_h = y2 - y1
                target_w = x2 - x1
                patch = local_mask
                if patch.shape != (target_h, target_w):
                    patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], patch.astype(np.uint8))
        else:
            applied_refined = False
            if image_rgb is not None:
                is_white_balloon = _is_white_balloon_region(image_rgb, [x1, y1, x2, y2])
                balloon_mask = None
                if is_white_balloon:
                    text_boxes = _extract_white_balloon_text_boxes(image_rgb, [x1, y1, x2, y2])
                    bbox_area = max(1, (x2 - x1) * (y2 - y1))
                    text_box_area = sum(
                        max(0, int(bx2) - int(bx1)) * max(0, int(by2) - int(by1))
                        for bx1, by1, bx2, by2 in text_boxes
                    )
                    if text_boxes and text_box_area >= max(64, int(bbox_area * 0.12)):
                        for bx1, by1, bx2, by2 in text_boxes:
                            bx1 = max(0, min(width, int(bx1)))
                            bx2 = max(0, min(width, int(bx2)))
                            by1 = max(0, min(height, int(by1)))
                            by2 = max(0, min(height, int(by2)))
                            if bx2 > bx1 and by2 > by1:
                                mask[by1:by2, bx1:bx2] = 255
                                applied_refined = True

                if not applied_refined:
                    refined = _build_refined_bbox_mask(image_rgb, [x1, y1, x2, y2])
                    if refined is not None:
                        rx1, ry1, patch = refined
                        patch_h, patch_w = patch.shape[:2]
                        bbox_area = max(1, (x2 - x1) * (y2 - y1))
                        refined_area = int(np.count_nonzero(patch))
                        area_ratio = refined_area / float(bbox_area)

                        if 0.035 <= area_ratio <= 0.78:
                            if is_white_balloon:
                                balloon_mask = _extract_white_balloon_fill_mask(image_rgb, [x1, y1, x2, y2])
                                if np.any(balloon_mask):
                                    local_balloon = balloon_mask[ry1 : ry1 + patch_h, rx1 : rx1 + patch_w]
                                    if local_balloon.shape == patch.shape:
                                        patch = cv2.bitwise_and(patch.astype(np.uint8), local_balloon.astype(np.uint8))
                                        patch = cv2.dilate(
                                            patch,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                            iterations=1,
                                        )
                                        refined_area = int(np.count_nonzero(patch))
                                if area_ratio >= 0.70:
                                    guard_bbox = _expand_bbox(
                                        [x1, y1, x2, y2],
                                        image_rgb.shape,
                                        pad_x_ratio=0.04,
                                        pad_y_ratio=0.10,
                                        min_pad_x=4,
                                        min_pad_y=6,
                                    )
                                    gx1, gy1, gx2, gy2 = guard_bbox
                                    guard_patch = np.full((gy2 - gy1, gx2 - gx1), 255, dtype=np.uint8)
                                    if isinstance(balloon_mask, np.ndarray) and np.any(balloon_mask):
                                        local_guard_balloon = balloon_mask[gy1:gy2, gx1:gx2]
                                        if local_guard_balloon.shape == guard_patch.shape:
                                            guard_patch = cv2.bitwise_and(
                                                guard_patch.astype(np.uint8),
                                                local_guard_balloon.astype(np.uint8),
                                            )
                                    if np.any(guard_patch):
                                        mask[gy1:gy2, gx1:gx2] = np.maximum(
                                            mask[gy1:gy2, gx1:gx2],
                                            guard_patch.astype(np.uint8),
                                        )
                                        applied_refined = True
                            else:
                                # Balão texturizado/SFX: expandir a máscara para cobrir contornos
                                # Dilatamos o patch refinado ao invés de clipar ao bbox
                                expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                                patch = cv2.dilate(patch, expand_kernel, iterations=2)
                                refined_area = int(np.count_nonzero(patch))

                            if refined_area >= max(12, int(bbox_area * 0.025)):
                                if not applied_refined:
                                    mask[ry1 : ry1 + patch_h, rx1 : rx1 + patch_w] = np.maximum(
                                        mask[ry1 : ry1 + patch_h, rx1 : rx1 + patch_w],
                                        patch.astype(np.uint8),
                                    )
                                    applied_refined = True

            if not applied_refined:
                mask[y1:y2, x1:x2] = 255

    if expand_mask and np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def run_detect_ocr(
    image_path: str,
    models_dir: str = "",
    profile: str = "quality",
    vision_worker_path: str = "",
    progress_callback=None,
    idioma_origem: str = "en",
) -> dict:

    _configure_model_roots(models_dir)
    _emit_stage_progress(progress_callback, "prepare_image", 0.03, "Preparando imagem para OCR")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        _emit_stage_progress(progress_callback, "complete", 1.0, "Imagem nao encontrada")
        return {"image": image_path, "width": 0, "height": 0, "texts": [], "_vision_blocks": []}

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if not _quick_text_presence_check(image_rgb):
        _emit_stage_progress(progress_callback, "complete", 1.0, "Pagina sem texto detectavel; OCR pulado")
        height, width = image_rgb.shape[:2]
        return {
            "image": image_path,
            "width": width,
            "height": height,
            "texts": [],
            "_vision_blocks": [],
            "quick_skipped_no_text": True,
            "sem_texto_detectado": True,
        }

    use_koharu_worker = bool(str(vision_worker_path or "").strip())
    if use_koharu_worker:
        try:
            page_result = _run_koharu_worker_detect_ocr(
                image_rgb=image_rgb,
                image_label=image_path,
                vision_worker_path=vision_worker_path,
                models_dir=models_dir,
                profile=profile,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            logger.warning("Koharu vision worker falhou em %s; fallback para stack atual: %s", image_path, exc)
            page_result = _run_detect_ocr_on_image(
                image_rgb,
                image_path,
                profile=profile,
                progress_callback=progress_callback,
                idioma_origem=idioma_origem,
            )
    else:
        page_result = _run_detect_ocr_on_image(
            image_rgb,
            image_path,
            profile=profile,
            progress_callback=progress_callback,
            idioma_origem=idioma_origem,
        )
    # Cache image for downstream use (layout enrichment) to avoid re-reading from disk
    page_result["_cached_image_bgr"] = image_bgr

    _emit_stage_progress(
        progress_callback,
        "complete",
        1.0,
        f"OCR concluido com {len(page_result.get('texts', []))} texto(s)",
    )
    return page_result


def run_inpaint_pages(
    image_files: list[Path],
    ocr_results: list[dict],
    output_dir: str,
    models_dir: str = "",
    profile: str = "quality",
    progress_callback=None,
) -> list[Path]:
    _configure_model_roots(models_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inpainter = _get_inpainter(profile)
    outputs: list[Path] = []
    total = len(image_files)

    if not image_files:
        return outputs

    with ThreadPoolExecutor(max_workers=2) as io_pool:
        load_future = io_pool.submit(_load_image_rgb, image_files[0])
        pending_save = None

        for index, (img_path, ocr_data) in enumerate(zip(image_files, ocr_results), start=1):
            image_np = load_future.result()

            if index < total:
                load_future = io_pool.submit(_load_image_rgb, image_files[index])
            else:
                load_future = None

            vision_blocks = list((ocr_data or {}).get("_vision_blocks", []))
            if not vision_blocks:
                ocr_data["sem_texto_detectado"] = True
                cleaned = image_np
            else:
                if "sem_texto_detectado" in ocr_data:
                    ocr_data["sem_texto_detectado"] = False
                try:
                    cleaned = _run_koharu_blockwise_inpaint_page(image_np, ocr_data, inpainter)
                except Exception as exc:
                    logger.warning(
                        "Inpaint blockwise estilo Koharu falhou em %s; fallback para fluxo anterior: %s",
                        img_path,
                        exc,
                    )
                    cleaned = _apply_inpainting_round(image_np, ocr_data, inpainter)

            if pending_save is not None:
                pending_save.result()

            dest = output_path / img_path.name
            pending_save = io_pool.submit(_save_image_rgb, cleaned, dest)
            outputs.append(dest)

            if progress_callback:
                progress_callback(index, total, f"Inpainting pagina {index}/{total}")

        if pending_save is not None:
            pending_save.result()

    return outputs


def run_debug_experiments(
    image_path: str,
    models_dir: str = "",
    profile: str = "quality",
    debug_root: str | Path | None = None,
) -> dict:
    _configure_model_roots(models_dir)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detect_page = _run_detect_ocr_on_image(image_rgb, image_path, profile=profile)
    raw_mask = vision_blocks_to_mask(image_rgb.shape, detect_page.get("_vision_blocks", []), image_rgb=image_rgb, expand_mask=False)
    expanded_mask = vision_blocks_to_mask(
        image_rgb.shape,
        detect_page.get("_vision_blocks", []),
        image_rgb=image_rgb,
        expand_mask=True,
    )

    run_root = _new_debug_run_root(debug_root)
    inpainter = _get_inpainter(profile)
    experiments = [
        {
            "name": "A_normal",
            "seam_cleanup": True,
            "multi_pass": True,
            "force_no_tiling": False,
        },
        {
            "name": "B_no_seam_cleanup",
            "seam_cleanup": False,
            "multi_pass": True,
            "force_no_tiling": False,
        },
        {
            "name": "C_single_pass_full_image",
            "seam_cleanup": False,
            "multi_pass": False,
            "force_no_tiling": True,
        },
    ]

    results = []
    boxes_overlay = _draw_boxes_overlay(image_rgb, detect_page.get("_vision_blocks", []))
    roi_overlay = _draw_roi_boundaries_overlay(image_rgb, expanded_mask)

    for config in experiments:
        recorder = DebugRunRecorder(run_dir=run_root / config["name"], experiment=config["name"], image_path=image_path)
        recorder.save_image("00_original.png", image_rgb)
        recorder.save_image("01_detect_boxes_overlay.png", boxes_overlay)
        _save_mask_png(recorder.run_dir / "02_text_mask_raw.png", raw_mask)
        _save_mask_png(recorder.run_dir / "03_text_mask_after_expand.png", expanded_mask)
        recorder.save_image("04_inpaint_input_image.png", image_rgb)
        _save_mask_png(recorder.run_dir / "05_inpaint_input_mask.png", expanded_mask)
        recorder.save_image("10_roi_boundaries_overlay.png", roi_overlay)

        round_result = _apply_inpainting_round(
            image_rgb,
            detect_page,
            inpainter,
            debug=recorder,
            seam_cleanup=config["seam_cleanup"],
            multi_pass=config["multi_pass"],
            force_no_tiling=config["force_no_tiling"],
        )
        if not isinstance(round_result, dict):
            raise RuntimeError("Modo debug deveria retornar dicionario de artefatos")

        raw_output = round_result["raw_output"]
        after_paste = round_result["after_roi_paste"]
        after_cleanup = round_result["after_seam_cleanup"]
        final_output = round_result["final_output"]

        recorder.save_image("06_inpaint_raw_output.png", raw_output)
        recorder.save_image("07_after_roi_paste.png", after_paste)
        recorder.save_image("08_after_seam_cleanup.png", after_cleanup)
        recorder.save_image("09_final_output.png", final_output)
        recorder.save_image("11_tile_boundaries_overlay.png", _draw_tile_boundaries_overlay(image_rgb, recorder.tile_logs))
        recorder.save_image("12_diff_06_vs_07.png", _build_diff_image(raw_output, after_paste))
        recorder.save_image("13_diff_07_vs_08.png", _build_diff_image(after_paste, after_cleanup))
        recorder.finalize()

        results.append(
            {
                "name": config["name"],
                "run_dir": str(recorder.run_dir),
                "tile_count": len(recorder.tile_logs),
                "seam_cleanup": bool(config["seam_cleanup"]),
                "multi_pass": bool(config["multi_pass"]),
                "force_no_tiling": bool(config["force_no_tiling"]),
            }
        )

    summary = {
        "image_path": image_path,
        "run_root": str(run_root),
        "experiments": results,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
