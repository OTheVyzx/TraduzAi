from __future__ import annotations

import atexit
import copy
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import json
import logging
import math
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    # Hints para o IDE - Ignorar avisos de resolução pois o sys.path é dinâmico
    from ocr.postprocess import ( # type: ignore
        _find_hf_model, analyze_style, classify_content, classify_text_type, fix_ocr_errors,
        infer_block_profile, infer_page_profile,
        is_editorial_credit, is_non_english, is_punctuation_only_noise,
        is_ghost_ocr_noise, is_hallucination, is_short_ornamental_text, is_short_textured_sfx_or_noise,
        is_structured_ocr_payload, is_watermark,
            is_vlm_failure_phrase,
            has_run_on_tokens, looks_suspicious, suspicious_confidence_threshold,
            should_retain_low_confidence_dialogue_ocr,
            is_korean_sfx, should_preserve_cjk_sfx_candidate, split_sfx_inline,
            SCANLATOR_RE, URL_RE,
    )
    from ocr.text_router import ROUTE_ACTIONS, apply_route_action, route_action_requires_inpaint
    from ocr.semantic_reviewer import semantic_refine_text # type: ignore
    from inpainter.classical import _extract_textured_balloon_mask, _expand_overlay_bbox # type: ignore
    from .ocr import _derive_text_pixel_bbox, infer_rotation_deg_from_line_polygons, normalize_paddleocr_language # type: ignore
else:
    # Imports relativos com fallback para garantir portabilidade no runtime
    try:
        from ocr.postprocess import (
            _find_hf_model, analyze_style, classify_content, classify_text_type, fix_ocr_errors,
            infer_block_profile, infer_page_profile,
            is_editorial_credit, is_non_english, is_punctuation_only_noise,
            is_ghost_ocr_noise, is_hallucination, is_short_ornamental_text, is_short_textured_sfx_or_noise,
            is_structured_ocr_payload, is_watermark,
            is_vlm_failure_phrase,
            has_run_on_tokens, looks_suspicious, suspicious_confidence_threshold,
            should_retain_low_confidence_dialogue_ocr,
            is_korean_sfx, should_preserve_cjk_sfx_candidate, split_sfx_inline,
            SCANLATOR_RE, URL_RE,
        )
        from ocr.text_router import ROUTE_ACTIONS, apply_route_action, route_action_requires_inpaint
        from ocr.semantic_reviewer import semantic_refine_text
    except ImportError:
        from ..ocr.postprocess import ( 
            _find_hf_model, analyze_style, classify_content, classify_text_type, fix_ocr_errors,
            infer_block_profile, infer_page_profile,
            is_editorial_credit, is_non_english, is_punctuation_only_noise,
            is_ghost_ocr_noise, is_hallucination, is_short_ornamental_text, is_short_textured_sfx_or_noise,
            is_structured_ocr_payload, is_watermark,
            is_vlm_failure_phrase,
            has_run_on_tokens, looks_suspicious, suspicious_confidence_threshold,
            should_retain_low_confidence_dialogue_ocr,
            is_korean_sfx, should_preserve_cjk_sfx_candidate, split_sfx_inline,
            SCANLATOR_RE, URL_RE,
        )
        from ..ocr.text_router import ROUTE_ACTIONS, apply_route_action, route_action_requires_inpaint
        from ..ocr.semantic_reviewer import semantic_refine_text

    try:
        from inpainter.classical import _extract_textured_balloon_mask, _expand_overlay_bbox
    except ImportError:
        from ..inpainter.classical import _extract_textured_balloon_mask, _expand_overlay_bbox

    from .ocr import _derive_text_pixel_bbox, infer_rotation_deg_from_line_polygons, normalize_paddleocr_language

try:
    from utils.decision_log import infer_page_number, record_decision
except ImportError:
    from ..utils.decision_log import infer_page_number, record_decision

try:
    from typesetter.style_policy import normalize_auto_typesetting_style, sample_text_background_rgb
except ImportError:
    from ..typesetter.style_policy import normalize_auto_typesetting_style, sample_text_background_rgb

try:
    from layout.simple_text_geometry import resolve_text_anchor_bbox
except ImportError:
    from ..layout.simple_text_geometry import resolve_text_anchor_bbox

try:
    from .engine_presets import EnginePreset, engine_steps_for_preset, resolve_engine_preset
except ImportError:
    from vision_stack.engine_presets import EnginePreset, engine_steps_for_preset, resolve_engine_preset

logger = logging.getLogger(__name__)

_font_detector = None
_koharu_http_client = None
_koharu_http_client_lock = threading.Lock()
_koharu_vision_worker_clients: dict[str, "_KoharuVisionWorkerProcess"] = {}
_koharu_vision_worker_lock = threading.Lock()

_KOHARU_CJK_OCR_STEPS = [
    "pp-doclayout-v3",
    "comic-text-detector-seg",
    "speech-bubble-segmentation",
    "paddle-ocr-vl-1.5",
]
_KOHARU_CJK_LANGS = {"japan", "korean", "ch", "chinese_cht"}


def _resolve_runtime_engine_preset(engine_preset_id: str = "", idioma_origem: str = "en") -> EnginePreset:
    config = {
        "engine_preset_id": engine_preset_id,
        "idioma_origem": idioma_origem,
    }
    return resolve_engine_preset(config, idioma_origem=idioma_origem)


def _preserve_cjk_sfx_for_engine_preset(engine_preset: EnginePreset | None) -> bool:
    return False


def _runtime_engine_steps(preset: EnginePreset, *, legacy_default: bool = False) -> list[str]:
    steps = engine_steps_for_preset(preset)
    if steps:
        return steps
    if legacy_default:
        return list(_KOHARU_CJK_OCR_STEPS)
    return []


def _attach_engine_preset_metadata(
    page_result: dict,
    preset: EnginePreset,
    engine_steps: list[str] | None = None,
) -> dict:
    steps = list(engine_steps if engine_steps is not None else _runtime_engine_steps(preset))
    detector_loader = _detector_model_for_preset(preset)
    metadata = {
        "engine_preset_id": preset.id,
        "content_family": preset.content_family,
        "detector": preset.detector,
        "detector_engine_id": preset.detector,
        "detector_loader": detector_loader,
        "segmenter": preset.segmenter,
        "bubble_segmenter": preset.bubble_segmenter,
        "ocr": preset.ocr,
        "inpainter": preset.inpainter,
        "mask_strategy": preset.mask_strategy,
        "engine_steps": steps,
    }
    page_result["engine_preset_id"] = preset.id
    page_result["engine_preset"] = preset.to_dict()
    page_result["_engine_preset"] = metadata
    page_result.setdefault("_pipeline_artifacts", _pipeline_artifacts_for_preset(preset))
    for block in page_result.get("_vision_blocks") or []:
        if isinstance(block, dict):
            _attach_detector_candidate_metadata(block, preset, detector_loader)
    return page_result


def _attach_detector_candidate_metadata(
    block: dict,
    preset: EnginePreset,
    detector_loader: str | None = None,
) -> dict:
    loader = detector_loader or _detector_model_for_preset(preset)
    block["detector_preset_id"] = preset.id
    block["detector_engine_id"] = preset.detector
    block["detector_loader"] = loader
    block.setdefault("candidate_kind", "detector_block")
    block.setdefault("validated_by_segment_mask", False)
    return block


def _pipeline_artifacts_for_preset(preset: EnginePreset) -> dict:
    return {
        "TextBoxes": {"producer": preset.detector, "status": "ok" if preset.detector != "disabled" else "skipped"},
        "SegmentMask": {"producer": preset.segmenter, "status": "ok" if preset.segmenter != "disabled" else "skipped"},
        "BubbleMask": {
            "producer": preset.bubble_segmenter,
            "status": "ok" if preset.bubble_segmenter != "disabled" else "skipped",
        },
        "OcrText": {"producer": preset.ocr, "status": "ok" if preset.ocr != "disabled" else "skipped"},
        "Inpainted": {"producer": preset.inpainter, "status": "pending"},
        "FinalRender": {"producer": "traduzai-typesetter", "status": "pending"},
    }


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        value = int(default)
    else:
        try:
            value = int(str(raw).strip())
        except Exception:
            value = int(default)
    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value


def _white_balloon_whitening_enabled() -> bool:
    return os.getenv("MANGATL_DISABLE_WHITE_BALLOON_WHITENING", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _white_balloon_text_box_cleanup_enabled() -> bool:
    return _env_flag("TRADUZAI_ENABLE_WHITE_BALLOON_TEXT_BOX_CLEANUP", False)


def _cleanup_selective_enabled() -> bool:
    return _env_flag("TRADUZAI_CLEANUP_SELECTIVE", False)


def _inpaint_roi_tighten_enabled() -> bool:
    return _env_flag("TRADUZAI_INPAINT_ROI_TIGHTEN", False)


def _inpaint_clustered_crop_windows_enabled() -> bool:
    return _env_flag("TRADUZAI_INPAINT_CLUSTERED_CROP_WINDOWS", False)


def _koharu_blockwise_inpaint_enabled() -> bool:
    return _env_flag("TRADUZAI_KOHARU_BLOCKWISE_INPAINT", False)


def _ocr_quick_check_2stage_enabled() -> bool:
    return _env_flag("TRADUZAI_OCR_QUICK_CHECK_2STAGE", False)


def _ocr_run_on_guard_enabled() -> bool:
    return _env_flag("TRADUZAI_OCR_RUN_ON_GUARD", True)


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
        from typesetter.font_detector import FontDetector # type: ignore
        _font_detector = FontDetector(model_path, fonts_dir)
    except Exception as exc:
        logger.warning("FontDetector não carregado: %s", exc)
        return None
    return _font_detector

_detector = None
_detector_model = ""
_ocr_engine = None
_inpainter = None
_inpainter_model = ""
_text_segmenter = None
_text_segmenter_model = ""
_detector_lock = threading.Lock()
_ocr_engine_lock = threading.Lock()
_inpainter_lock = threading.Lock()
_text_segmenter_lock = threading.Lock()
_configured_models_dir = None


def _emit_stage_progress(progress_callback, stage: str, progress: float, message: str):
    if progress_callback is None:
        return
    try:
        clamped = max(0.0, min(1.0, float(progress)))
    except Exception:
        clamped = 0.0
    progress_callback(stage, clamped, message)


def _coerce_bbox(raw_bbox) -> list[int] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _resolve_white_balloon_bbox(image_rgb: np.ndarray, text: dict) -> list[int] | None:
    candidates: list[list[int]] = []
    for key in ("balloon_bbox", "bbox", "text_pixel_bbox"):
        bbox = _coerce_bbox(text.get(key))
        if bbox is not None and bbox not in candidates:
            candidates.append(bbox)

    for bbox in candidates:
        if _is_white_balloon_region(image_rgb, bbox):
            return bbox
    return None


def _normalize_line_polygons(raw_line_polygons) -> list[list[list[int]]]:
    normalized: list[list[list[int]]] = []
    for polygon in raw_line_polygons or []:
        if not isinstance(polygon, (list, tuple)):
            continue
        points: list[list[int]] = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                points.append([int(round(float(point[0]))), int(round(float(point[1])))])
            except Exception:
                continue
        if len(points) >= 4:
            normalized.append(points)
    return normalized


def _bbox_from_line_polygons(line_polygons) -> list[int] | None:
    polygons = _normalize_line_polygons(line_polygons)
    if not polygons:
        return None
    xs: list[int] = []
    ys: list[int] = []
    for polygon in polygons:
        for px, py in polygon:
            xs.append(int(px))
            ys.append(int(py))
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


def _split_line_polygons_by_large_vertical_gap(line_polygons) -> list[list[list[list[int]]]]:
    entries: list[tuple[list[int], list[list[int]]]] = []
    for polygon in _normalize_line_polygons(line_polygons):
        bbox = _bbox_from_line_polygons([polygon])
        if bbox is not None:
            entries.append((bbox, polygon))
    if len(entries) < 2:
        return []

    entries.sort(key=lambda item: (item[0][1], item[0][0]))
    heights = [max(1, bbox[3] - bbox[1]) for bbox, _polygon in entries]
    median_height = sorted(heights)[len(heights) // 2]
    all_bbox = _bbox_from_line_polygons([polygon for _bbox, polygon in entries])
    if all_bbox is None:
        return []
    total_height = max(1, all_bbox[3] - all_bbox[1])
    threshold = max(42, int(median_height * 2.4), int(total_height * 0.18))

    groups: list[list[list[int]]] = [[entries[0][1]]]
    for previous, current in zip(entries, entries[1:]):
        previous_bbox = previous[0]
        current_bbox = current[0]
        gap = max(0, current_bbox[1] - previous_bbox[3])
        if gap >= threshold:
            groups.append([])
        groups[-1].append(current[1])

    return [group for group in groups if group]


def _split_white_cleanup_candidate_by_line_gaps(text: dict) -> list[dict]:
    groups = _split_line_polygons_by_large_vertical_gap(text.get("line_polygons") or [])
    if len(groups) < 2:
        return []

    split_items: list[dict] = []
    for index, group in enumerate(groups):
        bbox = _bbox_from_line_polygons(group)
        if bbox is None:
            continue
        if (bbox[2] - bbox[0]) < 16 or (bbox[3] - bbox[1]) < 8:
            continue
        child = dict(text)
        child["bbox"] = list(bbox)
        child["text_pixel_bbox"] = list(bbox)
        child["source_bbox"] = list(bbox)
        child["layout_bbox"] = list(bbox)
        child["line_polygons"] = [polygon for polygon in group]
        child["_white_cleanup_split_parent_bbox"] = list(text.get("bbox") or text.get("text_pixel_bbox") or [])
        child["_white_cleanup_split_index"] = index
        child["_white_cleanup_split_count"] = len(groups)
        split_items.append(child)
    return split_items


def _quick_text_presence_details(image_rgb: np.ndarray) -> tuple[bool, str]:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return False, "fast_skip"

    height, width = image_rgb.shape[:2]
    if min(height, width) < 256:
        return True, "fast_pass"

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

    def _textlike_component_stats(mask: np.ndarray) -> tuple[int, int]:
        if mask.size == 0 or not np.any(mask):
            return 0, 0

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
        return component_count, combined_area

    dark_components, dark_area = _textlike_component_stats(dark_mask)
    bright_components, bright_area = _textlike_component_stats(bright_mask)
    component_count = max(dark_components, bright_components)
    combined_area = max(dark_area, bright_area)
    if component_count >= 3 or combined_area >= 60:
        return True, "fast_pass"

    edge_density = float(np.count_nonzero(cv2.Canny(gray, 90, 180))) / float(gray.size)
    gray_std = float(np.std(gray))
    marginal = component_count in {1, 2} or 30 <= combined_area < 60
    if _ocr_quick_check_2stage_enabled() and marginal:
        center_margin_x = max(0, int(width * 0.18))
        center_margin_y = max(0, int(height * 0.18))
        center = image_rgb[
            center_margin_y : max(center_margin_y + 1, height - center_margin_y),
            center_margin_x : max(center_margin_x + 1, width - center_margin_x),
        ]
        if center.size and center.shape[:2] != image_rgb.shape[:2]:
            center_present, _ = _quick_text_presence_details(center)
            return bool(center_present), "marginal_pass" if center_present else "marginal_skip"
    if gray_std >= 18.0 and edge_density >= 0.012:
        return True, "fast_pass" if not marginal else "marginal_pass"

    return False, "fast_skip"


def _quick_text_presence_check(image_rgb: np.ndarray) -> bool:
    return _quick_text_presence_details(image_rgb)[0]


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


def _strip_paddle_crop_fallback_max() -> int:
    raw = (
        os.getenv("TRADUZAI_STRIP_PADDLE_CROP_FALLBACK_MAX")
        or os.getenv("TRADUZAI_PADDLE_CROP_FALLBACK_MAX")
        or "3"
    )
    try:
        return max(0, int(str(raw).strip()))
    except Exception:
        return 1


def _strip_paddle_sparse_crop_fallback_max() -> int:
    raw = (
        os.getenv("TRADUZAI_STRIP_PADDLE_SPARSE_CROP_FALLBACK_MAX")
        or os.getenv("TRADUZAI_PADDLE_SPARSE_CROP_FALLBACK_MAX")
    )
    if raw is None:
        return _strip_paddle_crop_fallback_max()
    try:
        return max(0, int(str(raw).strip()))
    except Exception:
        return _strip_paddle_crop_fallback_max()


def _strip_quick_text_skip_enabled() -> bool:
    return False


def _strip_scanlation_credit_skip_enabled() -> bool:
    return False


def _looks_like_scanlation_credit_band(image_rgb: np.ndarray, blocks: list) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return False

    height, width = image_rgb.shape[:2]
    if len(blocks) < 10 or min(height, width) < 180:
        return False

    bboxes: list[list[int]] = []
    for block in blocks:
        raw_bbox = getattr(block, "xyxy", None)
        bbox = _coerce_bbox(raw_bbox)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            bboxes.append([x1, y1, x2, y2])

    if len(bboxes) < 10:
        return False

    compact_text_blocks = 0
    for x1, y1, x2, y2 in bboxes:
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        if box_h <= height * 0.16 and box_area <= (width * height) * 0.06:
            compact_text_blocks += 1
    if compact_text_blocks < 10:
        return False

    y_span = max(bbox[3] for bbox in bboxes) - min(bbox[1] for bbox in bboxes)
    if y_span < height * 0.42:
        return False

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if float(np.percentile(gray, 95)) < 160.0:
        return False
    bright_mask = (gray >= 170).astype(np.uint8) * 255
    kernel_width = max(48, int(round(width * 0.12)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 2))
    horizontal = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_horizontal_count = 0
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        del x, y
        if box_w >= width * 0.18 and box_h <= max(18, int(round(height * 0.06))):
            long_horizontal_count += 1

    return long_horizontal_count >= 5


def _looks_like_cover_editorial_band(
    image_rgb: np.ndarray,
    blocks: list,
    source_page_number: int | None,
) -> bool:
    if source_page_number not in {1, 2}:
        return False
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return False

    height, width = image_rgb.shape[:2]
    if len(blocks) < 8 or min(height, width) < 180:
        return False

    bboxes: list[list[int]] = []
    for block in blocks:
        bbox = _coerce_bbox(getattr(block, "xyxy", None))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            bboxes.append([x1, y1, x2, y2])

    if len(bboxes) < 8:
        return False

    compact_text_blocks = 0
    for x1, y1, x2, y2 in bboxes:
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        if box_h <= height * 0.20 and box_area <= (width * height) * 0.08:
            compact_text_blocks += 1
    if compact_text_blocks < 7:
        return False

    y_span = max(bbox[3] for bbox in bboxes) - min(bbox[1] for bbox in bboxes)
    if y_span < height * 0.55:
        return False

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if float(np.percentile(gray, 95)) < 190.0:
        return False

    bright_threshold = max(170.0, min(255.0, float(np.percentile(gray, 95)) + 8.0))
    bright_mask = (gray >= bright_threshold).astype(np.uint8) * 255
    kernel_width = max(48, int(round(width * 0.12)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 2))
    horizontal = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_horizontal_count = 0
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        del x, y
        if box_w >= width * 0.18 and box_h <= max(20, int(round(height * 0.07))):
            long_horizontal_count += 1

    return long_horizontal_count >= 3


def _configure_model_roots(models_dir: str = ""):
    global _configured_models_dir

    if not models_dir:
        return

    root = Path(models_dir)
    if _configured_models_dir == root:
        return

    from . import detector as detector_module
    from . import inpainter as inpainter_module
    from . import manga_text_segmenter as manga_text_segmenter_module

    detector_module.MODELS_DIR = root
    inpainter_module.MODELS_DIR = root
    manga_text_segmenter_module.MODELS_DIR = root
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


def _koharu_worker_persistent_enabled() -> bool:
    return _env_flag("TRADUZAI_KOHARU_WORKER_PERSISTENT", True)


def _koharu_worker_ocr_only_enabled() -> bool:
    return _env_flag("TRADUZAI_KOHARU_WORKER_OCR_ONLY", True)


class _KoharuVisionWorkerProcess:
    def __init__(self, worker_path: Path):
        self.worker_path = worker_path
        self.process: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.unavailable = False

    def stop(self) -> None:
        proc = self.process
        self.process = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            pass

    def _ensure_started(self) -> subprocess.Popen:
        if self.unavailable:
            raise RuntimeError("worker persistente indisponivel")
        if self.process is not None and self.process.poll() is None:
            return self.process
        self.process = subprocess.Popen(
            [str(self.worker_path), "--stdio-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_build_koharu_worker_env(),
        )
        return self.process

    def run_batch(self, request_payload: dict) -> tuple[dict, dict]:
        with self.lock:
            started = time.perf_counter()
            proc = self._ensure_started()
            if proc.stdin is None or proc.stdout is None:
                self.unavailable = True
                raise RuntimeError("worker persistente sem pipes stdio")
            raw = json.dumps(request_payload, ensure_ascii=False)
            try:
                proc.stdin.write(raw + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except Exception:
                self.unavailable = True
                self.stop()
                raise
            if not line:
                self.unavailable = True
                self.stop()
                raise RuntimeError("worker persistente encerrou sem resposta")
            parse_started = time.perf_counter()
            payload = json.loads(line)
            parse_ms = int(round((time.perf_counter() - parse_started) * 1000))
            if str(payload.get("status", "")).lower() == "error":
                raise RuntimeError(str(payload.get("error") or "worker persistente retornou erro"))
            timings = {
                "persistent": True,
                "worker_wall_ms": int(round((time.perf_counter() - started) * 1000)),
                "worker_json_parse_ms": parse_ms,
            }
            return payload, timings


def _get_koharu_vision_worker_client(worker_path: Path) -> _KoharuVisionWorkerProcess:
    key = str(worker_path.resolve())
    with _koharu_vision_worker_lock:
        client = _koharu_vision_worker_clients.get(key)
        if client is None:
            client = _KoharuVisionWorkerProcess(worker_path)
            _koharu_vision_worker_clients[key] = client
        return client


def _shutdown_koharu_vision_workers() -> None:
    with _koharu_vision_worker_lock:
        clients = list(_koharu_vision_worker_clients.values())
        _koharu_vision_worker_clients.clear()
    for client in clients:
        client.stop()


atexit.register(_shutdown_koharu_vision_workers)


def _build_koharu_worker_page_result(
    image_rgb: np.ndarray,
    image_label: str,
    worker_payload: dict,
    profile: str = "quality",
    progress_callback=None,
    idioma_origem: str = "en",
    engine_preset_id: str = "",
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:
    engine_preset = _resolve_runtime_engine_preset(engine_preset_id, idioma_origem)
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
                line_polygons=item.get("line_polygons") or item.get("linePolygons"),
                source_direction=item.get("source_direction") or item.get("sourceDirection"),
                balloon_bbox=item.get("balloon_bbox") or item.get("balloonBBox"),
                balloon_polygon=item.get("balloon_polygon") or item.get("balloonPolygon"),
                balloon_subregions=item.get("balloon_subregions") or item.get("balloonSubregions"),
                connected_lobe_bboxes=item.get("connected_lobe_bboxes") or item.get("connectedLobeBboxes"),
                connected_lobe_polygons=item.get("connected_lobe_polygons") or item.get("connectedLobePolygons"),
            )
        )
        rich_item = dict(item)
        rich_item["text"] = str(rich_item.get("text", "") or "")
        if "line_polygons" not in rich_item and "linePolygons" in rich_item:
            rich_item["line_polygons"] = rich_item.get("linePolygons")
        if "source_direction" not in rich_item and "sourceDirection" in rich_item:
            rich_item["source_direction"] = rich_item.get("sourceDirection")
        texts.append(rich_item)

    page_result = build_page_result(
        image_path=image_label,
        image_rgb=image_rgb,
        blocks=blocks,
        texts=texts,
        profile=profile,
        ocr_backend="koharu-paddle-ocr-vl-1.5",
        enable_font_detection=True,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
        preserve_cjk_sfx=_preserve_cjk_sfx_for_engine_preset(engine_preset),
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    page_result["_bubble_regions"] = worker_bubble_regions
    _attach_worker_bubble_geometry(page_result, worker_bubble_regions)
    page_result["_vision_backend"] = "koharu"
    return page_result


def _rect_polygon_from_bbox_for_geometry(bbox: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]]


def _bbox_overlap_area_for_geometry(a: list[int], b: list[int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _bbox_center_inside_for_geometry(container: list[int], inner: list[int]) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return container[0] <= cx <= container[2] and container[1] <= cy <= container[3]


def _attach_worker_bubble_geometry(page_result: dict, bubble_regions: list) -> None:
    page_width = int(page_result.get("width", 0) or 0)
    page_height = int(page_result.get("height", 0) or 0)

    def _inner_bbox_for_bubble(bbox: list[int]) -> list[int] | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = max(4, int(min(max(1, x2 - x1), max(1, y2 - y1)) * 0.07))
        inner = [
            max(0, x1 + pad),
            max(0, y1 + pad),
            min(page_width or x2, x2 - pad),
            min(page_height or y2, y2 - pad),
        ]
        if inner[2] <= inner[0] or inner[3] <= inner[1]:
            return None
        return inner

    bubbles: list[dict] = []
    for index, region in enumerate(bubble_regions or [], start=1):
        if not isinstance(region, dict):
            continue
        bbox = _coerce_bbox(region.get("bbox") or region.get("box"))
        if bbox is not None:
            bubble_id = str(region.get("bubble_id") or region.get("bubbleId") or region.get("id") or f"bubble_{index:03d}")
            bubble = dict(region)
            bubble["bbox"] = list(bbox)
            bubble["bubble_id"] = bubble_id
            bubble["bubble_mask_bbox"] = _coerce_bbox(
                region.get("bubble_mask_bbox") or region.get("bubbleMaskBbox")
            ) or list(bbox)
            bubble["bubble_inner_bbox"] = _coerce_bbox(
                region.get("bubble_inner_bbox") or region.get("bubbleInnerBbox")
            ) or _inner_bbox_for_bubble(bbox)
            bubbles.append(bubble)
    if not bubbles:
        return
    page_result["_bubble_regions"] = [dict(bubble) for bubble in bubbles]

    def _best_bubble_for_bbox(bbox: list[int] | None) -> dict | None:
        if bbox is None:
            return None
        best = None
        best_score = 0
        for bubble in bubbles:
            bubble_bbox = bubble.get("bbox")
            overlap = _bbox_overlap_area_for_geometry(bubble_bbox, bbox)
            if _bbox_center_inside_for_geometry(bubble_bbox, bbox):
                overlap += max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            if overlap > best_score:
                best = bubble
                best_score = overlap
        return best

    for collection in (page_result.get("texts") or [], page_result.get("_vision_blocks") or []):
        for item in collection:
            if not isinstance(item, dict):
                continue
            bbox = _coerce_bbox(item.get("text_pixel_bbox")) or _coerce_bbox(item.get("bbox"))
            bubble = _best_bubble_for_bbox(bbox)
            if bubble is None:
                continue
            bubble_bbox = list(bubble.get("bbox") or [])
            item.setdefault("balloon_bbox", bubble_bbox)
            item.setdefault("balloon_polygon", _rect_polygon_from_bbox_for_geometry(bubble_bbox))
            for key in ("bubble_id", "bubble_mask_bbox", "bubble_inner_bbox"):
                value = bubble.get(key)
                if value not in (None, [], ""):
                    item.setdefault(key, copy.deepcopy(value))


def _read_koharu_worker_json_payload(result: subprocess.CompletedProcess, context: str) -> dict:
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"codigo {result.returncode}"
        raise RuntimeError(f"{context} falhou: {detail}")

    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{context} retornou stdout vazio")

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON invalido do {context}: {exc}") from exc

    if str(payload.get("status", "")).lower() != "ok":
        raise RuntimeError(str(payload.get("error") or f"{context} sem status ok"))
    return payload


def _run_koharu_worker_detect_ocr(
    image_rgb: np.ndarray,
    image_label: str,
    vision_worker_path: str,
    models_dir: str = "",
    profile: str = "quality",
    progress_callback=None,
    idioma_origem: str = "en",
    engine_preset_id: str = "",
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:
    worker_path = Path(str(vision_worker_path).strip())
    if not worker_path.exists():
        raise FileNotFoundError(f"Koharu vision worker nao encontrado: {worker_path}")

    engine_preset = _resolve_runtime_engine_preset(engine_preset_id, idioma_origem)
    engine_steps = _runtime_engine_steps(engine_preset)
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
        "enginePresetId": engine_preset.id,
        "engineSteps": engine_steps,
        "maskStrategy": engine_preset.mask_strategy,
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

    payload = _read_koharu_worker_json_payload(result, "Koharu vision worker")

    _emit_stage_progress(progress_callback, "recognize_text", 0.62, "Reconhecendo texto com PaddleOCR-VL")
    page_result = _build_koharu_worker_page_result(
        image_rgb=image_rgb,
        image_label=image_label,
        worker_payload=payload,
        profile=profile,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
        engine_preset_id=engine_preset.id,
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    page_result["_koharu_worker"] = {
        "engine_preset_id": engine_preset.id,
        "engine_steps": engine_steps,
        "mask_strategy": engine_preset.mask_strategy,
    }
    return _attach_engine_preset_metadata(page_result, engine_preset, engine_steps)


def _coerce_worker_known_bboxes(raw_bboxes) -> list[list[int]]:
    known: list[list[int]] = []
    for raw_bbox in raw_bboxes or []:
        bbox = _coerce_bbox(raw_bbox)
        if bbox is None:
            continue
        if bbox not in known:
            known.append(bbox)
    return known


def _estimate_koharu_worker_max_new_tokens(job: dict, *, known_bboxes: list[list[int]]) -> int:
    explicit = job.get("max_new_tokens")
    if explicit is not None:
        try:
            return max(16, int(explicit))
        except Exception:
            pass

    image_rgb = job.get("image_rgb")
    height = width = 0
    if isinstance(image_rgb, np.ndarray) and image_rgb.size:
        height, width = image_rgb.shape[:2]
    area = int(max(1, width) * max(1, height))
    known_count = len(known_bboxes)
    min_tokens = max(32, int(os.getenv("TRADUZAI_KOHARU_MIN_NEW_TOKENS", "64") or 64))
    max_tokens = max(min_tokens, int(os.getenv("TRADUZAI_KOHARU_MAX_NEW_TOKENS", "192") or 192))

    if known_count <= 1 and area <= 220_000:
        estimate = 64
    elif known_count <= 2 and area <= 520_000:
        estimate = 96
    elif known_count >= 4 or area >= 1_100_000:
        estimate = 192
    else:
        estimate = 128
    return max(min_tokens, min(max_tokens, estimate))


def _build_koharu_worker_batch_request_payload(
    jobs: list[dict],
    *,
    runtime_root: str,
    threshold: float,
    engine_preset: EnginePreset,
    engine_steps: list[str],
) -> list[dict]:
    request_payloads: list[dict] = []
    for job in jobs:
        region = job.get("region")
        known_bboxes = _coerce_worker_known_bboxes(
            job.get("known_text_bboxes") or job.get("knownTextBBoxes") or job.get("knownTextBboxes")
        )
        use_ocr_only = bool(known_bboxes and _koharu_worker_ocr_only_enabled())
        if use_ocr_only:
            mode = "ocrOnly"
        else:
            mode = "region" if isinstance(region, (list, tuple)) and len(region) >= 4 else "page"
        max_new_tokens = _estimate_koharu_worker_max_new_tokens(job, known_bboxes=known_bboxes)
        payload = {
            "imagePath": str(job.get("image_path")),
            "mode": mode,
            "runtimeRoot": runtime_root,
            "cpu": False,
            "maxNewTokens": max_new_tokens,
            "detectionThreshold": threshold,
            "enginePresetId": engine_preset.id,
            "engineSteps": engine_steps,
            "maskStrategy": engine_preset.mask_strategy,
        }
        if mode == "region":
            payload["region"] = [int(v) for v in list(region)[:4]]
        if use_ocr_only:
            payload["knownTextBBoxes"] = known_bboxes
        request_payloads.append(payload)
    return request_payloads


def _run_koharu_worker_detect_ocr_batch(
    jobs: list[dict],
    vision_worker_path: str,
    models_dir: str = "",
    profile: str = "quality",
    progress_callback=None,
    idioma_origem: str = "en",
    engine_preset_id: str = "",
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> list[dict]:
    worker_path = Path(str(vision_worker_path).strip())
    if not worker_path.exists():
        raise FileNotFoundError(f"Koharu vision worker nao encontrado: {worker_path}")

    clean_jobs = [job for job in jobs if isinstance(job, dict) and job.get("image_path") is not None]
    if not clean_jobs:
        return []

    runtime_root = _vision_worker_runtime_root(models_dir)
    threshold = _profile_to_detection_threshold(profile)
    engine_preset = _resolve_runtime_engine_preset(engine_preset_id, idioma_origem)
    engine_steps = _runtime_engine_steps(engine_preset)
    request_payloads = _build_koharu_worker_batch_request_payload(
        clean_jobs,
        runtime_root=runtime_root,
        threshold=threshold,
        engine_preset=engine_preset,
        engine_steps=engine_steps,
    )

    _emit_stage_progress(progress_callback, "load_detector", 0.08, "Carregando detector Koharu")
    _emit_stage_progress(progress_callback, "load_ocr_engine", 0.18, "Carregando OCR Koharu")

    request_envelope = {"requests": request_payloads}
    batch_transport: dict = {
        "persistent": False,
        "job_count": len(clean_jobs),
        "ocr_only_job_count": sum(1 for item in request_payloads if item.get("mode") == "ocrOnly"),
        "max_new_tokens": [int(item.get("maxNewTokens") or 0) for item in request_payloads],
        "engine_preset_id": engine_preset.id,
        "engine_steps": engine_steps,
        "mask_strategy": engine_preset.mask_strategy,
    }
    payload = None
    if _koharu_worker_persistent_enabled():
        try:
            _emit_stage_progress(progress_callback, "detect_text", 0.38, "Detectando blocos com Koharu persistente")
            client = _get_koharu_vision_worker_client(worker_path)
            payload, persistent_timings = client.run_batch(request_envelope)
            batch_transport.update(persistent_timings)
        except Exception as exc:
            batch_transport["persistent_error"] = str(exc)[:240]
            logger.warning("Koharu worker persistente indisponivel; fallback para batch CLI: %s", exc)

    if payload is None:
        with tempfile.TemporaryDirectory(prefix="traduzai_koharu_vision_batch_") as tmpdir:
            request_path = Path(tmpdir) / "batch_request.json"
            write_started = time.perf_counter()
            request_path.write_text(
                json.dumps(request_envelope, ensure_ascii=False),
                encoding="utf-8",
            )
            batch_transport["request_write_ms"] = int(round((time.perf_counter() - write_started) * 1000))

            _emit_stage_progress(progress_callback, "detect_text", 0.38, "Detectando blocos com Koharu em lote")
            worker_started = time.perf_counter()
            result = subprocess.run(
                [str(worker_path), "--batch-request-file", str(request_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_build_koharu_worker_env(),
                check=False,
            )
            batch_transport["worker_wall_ms"] = int(round((time.perf_counter() - worker_started) * 1000))

        parse_started = time.perf_counter()
        payload = _read_koharu_worker_json_payload(result, "Koharu vision worker batch")
        batch_transport["worker_json_parse_ms"] = int(round((time.perf_counter() - parse_started) * 1000))

    batch_transport["batch_timings_ms"] = payload.get("timings_ms") or payload.get("timingsMs") or {}
    responses = list(payload.get("responses") or [])
    if len(responses) != len(clean_jobs):
        raise RuntimeError(
            f"Koharu vision worker batch retornou {len(responses)} resposta(s) para {len(clean_jobs)} job(s)"
        )

    _emit_stage_progress(progress_callback, "recognize_text", 0.62, "Reconhecendo texto com PaddleOCR-VL em lote")
    page_results: list[dict] = []
    for job, item in zip(clean_jobs, responses):
        image_rgb = job.get("image_rgb")
        if image_rgb is None:
            raise ValueError("job Koharu batch sem image_rgb")
        image_label = str(job.get("image_path"))
        item_status = str(item.get("status", "")).lower()
        response_payload = item.get("response") if isinstance(item, dict) else None
        if item_status != "ok" or not isinstance(response_payload, dict):
            height, width = image_rgb.shape[:2]
            page_results.append(
                _attach_engine_preset_metadata(
                    {
                    "image": image_label,
                    "width": width,
                    "height": height,
                    "texts": [],
                    "_vision_blocks": [],
                    "_vision_backend": "koharu-worker-batch",
                    "_koharu_worker_batch": {
                        "status": "error",
                        "error": str(item.get("error") or "item sem resposta ok")[:240],
                        "index": item.get("index"),
                        **batch_transport,
                    },
                    },
                    engine_preset,
                    engine_steps,
                )
            )
            continue

        page_result = _build_koharu_worker_page_result(
            image_rgb=image_rgb,
            image_label=image_label,
            worker_payload=response_payload,
            profile=profile,
            progress_callback=progress_callback,
            idioma_origem=idioma_origem,
            engine_preset_id=engine_preset.id,
            work_title=work_title or str(job.get("work_title") or ""),
            work_title_aliases=work_title_aliases or job.get("work_title_aliases"),
            work_title_user_provided=bool(work_title_user_provided or job.get("work_title_user_provided")),
        )
        page_result["_vision_backend"] = "koharu-worker-batch"
        page_result["_koharu_worker_batch"] = {
            "status": item_status,
            "index": item.get("index"),
            "batch_size": len(clean_jobs),
            "timings_ms": response_payload.get("timings_ms") or response_payload.get("timingsMs") or {},
            **batch_transport,
        }
        page_results.append(_attach_engine_preset_metadata(page_result, engine_preset, engine_steps))

    return page_results


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_koharu_exe(models_dir: str = "") -> Path | None:
    configured = os.getenv("TRADUZAI_KOHARU_EXE", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    if models_dir:
        base = Path(models_dir)
        candidates.extend([base / "koharu.exe", base / "koharu" / "koharu.exe"])
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parent
    candidates.extend(
        [
            workspace_root / "koharu" / "koharu.exe",
            repo_root / "koharu" / "koharu.exe",
            Path.cwd().parent / "koharu" / "koharu.exe",
        ]
    )
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _should_use_koharu_cjk_ocr(idioma_origem: str, models_dir: str = "") -> bool:
    normalized = normalize_paddleocr_language(idioma_origem)
    if normalized not in _KOHARU_CJK_LANGS:
        return False
    raw = os.getenv("TRADUZAI_KOHARU_CJK_OCR", "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "disabled"}:
        return False
    if raw in {"1", "true", "yes", "on", "auto", ""}:
        return _resolve_koharu_exe(models_dir) is not None
    return False


class _KoharuHttpOcrClient:
    def __init__(self, exe_path: Path):
        self.exe_path = exe_path
        self.port = _find_free_local_port()
        self.base_url = f"http://127.0.0.1:{self.port}/api/v1"
        self.process: subprocess.Popen | None = None
        self.project_ready = False

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self.process = subprocess.Popen(
            [str(self.exe_path), "--headless", "--port", str(self.port)],
            cwd=str(self.exe_path.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self._wait_api_ready()

    def stop(self) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.request_json("DELETE", "/projects/current", timeout=10)
                except Exception:
                    pass
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=15)
        finally:
            self.process = None
            self.project_ready = False

    def request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        timeout: int = 120,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Koharu HTTP {method} {path} falhou com {exc.code}: {body}") from exc
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def _wait_api_ready(self, timeout_sec: int = 240) -> None:
        deadline = time.time() + timeout_sec
        last_error: Exception | None = None
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"Koharu encerrou durante bootstrap com codigo {self.process.returncode}")
            try:
                self.request_json("GET", "/meta", timeout=10)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.75)
        raise TimeoutError(f"Koharu HTTP nao ficou pronto: {last_error}")

    def _ensure_project(self) -> None:
        if self.project_ready:
            return
        self.request_json(
            "POST",
            "/projects",
            {"name": f"traduzai-cjk-ocr-{datetime.now().strftime('%H%M%S')}-{uuid4().hex[:6]}"},
            timeout=60,
        )
        self.project_ready = True

    def _wait_operation(self, operation_id: str, timeout_sec: int = 900) -> dict:
        deadline = time.time() + timeout_sec
        last_seen: dict | None = None
        while time.time() < deadline:
            payload = self.request_json("GET", "/operations", timeout=30)
            for operation in payload.get("operations", []):
                if operation.get("id") == operation_id:
                    last_seen = operation
                    if str(operation.get("status", "")).lower() != "running":
                        return operation
            time.sleep(0.75)
        raise TimeoutError(f"Koharu operation {operation_id} nao terminou: {last_seen}")

    def run_ocr(
        self,
        image_path: str,
        image_rgb: np.ndarray,
        profile: str = "quality",
        progress_callback=None,
        idioma_origem: str = "en",
        engine_preset_id: str = "",
        work_title: str = "",
        work_title_aliases: list[str] | tuple[str, ...] | None = None,
        work_title_user_provided: bool = False,
    ) -> dict:
        engine_preset = _resolve_runtime_engine_preset(engine_preset_id, idioma_origem)
        engine_steps = _runtime_engine_steps(engine_preset, legacy_default=True)
        self.start()
        self._ensure_project()
        source_path = str(Path(image_path).resolve())
        _emit_stage_progress(progress_callback, "koharu_import", 0.12, "Importando pagina no Koharu")
        imported = self.request_json(
            "POST",
            "/pages/from-paths",
            {"paths": [source_path], "replace": True},
            timeout=180,
        )
        page_ids = list(imported.get("pages") or [])
        if not page_ids:
            raise RuntimeError("Koharu nao retornou page id ao importar a pagina")
        page_id = page_ids[0]
        _emit_stage_progress(progress_callback, "koharu_ocr", 0.28, "Rodando PaddleOCR-VL no Koharu")
        operation = self.request_json(
            "POST",
            "/pipelines",
            {"steps": engine_steps, "pages": [page_id]},
            timeout=60,
        )
        finished = self._wait_operation(str(operation.get("operationId") or ""), timeout_sec=900)
        status = str(finished.get("status", "")).lower()
        if status not in {"completed", "completedwitherrors"}:
            raise RuntimeError(f"Koharu OCR falhou: {finished}")
        scene = self.request_json("GET", "/scene.json", timeout=120)
        text_blocks = _extract_koharu_scene_text_blocks(scene, page_id)
        page_result = _build_koharu_worker_page_result(
            image_rgb=image_rgb,
            image_label=image_path,
            worker_payload={"text_blocks": text_blocks, "bubble_regions": []},
            profile=profile,
            progress_callback=progress_callback,
            idioma_origem=idioma_origem,
            engine_preset_id=engine_preset.id,
            work_title=work_title,
            work_title_aliases=work_title_aliases,
            work_title_user_provided=work_title_user_provided,
        )
        page_result["_vision_backend"] = "koharu-http"
        page_result["_koharu_http"] = {
            "engine_preset_id": engine_preset.id,
            "content_family": engine_preset.content_family,
            "mask_strategy": engine_preset.mask_strategy,
            "engine_steps": engine_steps,
            "operation_status": finished.get("status"),
            "text_block_count": len(text_blocks),
        }
        return _attach_engine_preset_metadata(page_result, engine_preset, engine_steps)

    def run_ocr_batch(
        self,
        jobs: list[dict],
        profile: str = "quality",
        progress_callback=None,
        idioma_origem: str = "en",
        engine_preset_id: str = "",
        work_title: str = "",
        work_title_aliases: list[str] | tuple[str, ...] | None = None,
        work_title_user_provided: bool = False,
    ) -> list[dict]:
        clean_jobs = [job for job in jobs if isinstance(job, dict) and job.get("image_path") is not None]
        if not clean_jobs:
            return []

        engine_preset = _resolve_runtime_engine_preset(engine_preset_id, idioma_origem)
        engine_steps = _runtime_engine_steps(engine_preset, legacy_default=True)
        self.start()
        self._ensure_project()
        source_paths = [str(Path(str(job.get("image_path"))).resolve()) for job in clean_jobs]
        _emit_stage_progress(progress_callback, "koharu_import", 0.12, "Importando paginas no Koharu")
        imported = self.request_json(
            "POST",
            "/pages/from-paths",
            {"paths": source_paths, "replace": True},
            timeout=300,
        )
        page_ids = [str(page_id) for page_id in list(imported.get("pages") or [])]
        if len(page_ids) != len(clean_jobs):
            raise RuntimeError(
                f"Koharu retornou {len(page_ids)} page id(s) para {len(clean_jobs)} pagina(s)"
            )

        _emit_stage_progress(progress_callback, "koharu_ocr", 0.28, "Rodando PaddleOCR-VL em lote no Koharu")
        operation = self.request_json(
            "POST",
            "/pipelines",
            {"steps": engine_steps, "pages": page_ids},
            timeout=60,
        )
        finished = self._wait_operation(str(operation.get("operationId") or ""), timeout_sec=1800)
        status = str(finished.get("status", "")).lower()
        if status not in {"completed", "completedwitherrors"}:
            raise RuntimeError(f"Koharu OCR em lote falhou: {finished}")
        scene = self.request_json("GET", "/scene.json", timeout=180)

        results: list[dict] = []
        for job, page_id in zip(clean_jobs, page_ids):
            image_rgb = job.get("image_rgb")
            if not isinstance(image_rgb, np.ndarray):
                image_bgr = cv2.imread(str(job.get("image_path")))
                if image_bgr is None:
                    raise RuntimeError(f"Imagem do batch Koharu nao encontrada: {job.get('image_path')}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_path = str(job.get("image_path"))
            text_blocks = _extract_koharu_scene_text_blocks(scene, page_id)
            page_result = _build_koharu_worker_page_result(
                image_rgb=image_rgb,
                image_label=image_path,
                worker_payload={"text_blocks": text_blocks, "bubble_regions": []},
                profile=profile,
                progress_callback=progress_callback,
                idioma_origem=idioma_origem,
                engine_preset_id=engine_preset.id,
                work_title=work_title or str(job.get("work_title") or ""),
                work_title_aliases=work_title_aliases or job.get("work_title_aliases"),
                work_title_user_provided=bool(work_title_user_provided or job.get("work_title_user_provided")),
            )
            page_result["_vision_backend"] = "koharu-http"
            page_result["_koharu_http"] = {
                "engine_preset_id": engine_preset.id,
                "content_family": engine_preset.content_family,
                "mask_strategy": engine_preset.mask_strategy,
                "engine_steps": engine_steps,
                "operation_status": finished.get("status"),
                "text_block_count": len(text_blocks),
                "batch": True,
                "batch_size": len(clean_jobs),
                "page_id": page_id,
            }
            results.append(_attach_engine_preset_metadata(page_result, engine_preset, engine_steps))
        return results


def _extract_koharu_scene_text_blocks(scene_snapshot: dict, page_id: str) -> list[dict]:
    scene = scene_snapshot.get("scene") if isinstance(scene_snapshot, dict) else {}
    if not isinstance(scene, dict):
        scene = scene_snapshot
    pages = scene.get("pages") if isinstance(scene, dict) else {}
    page = (pages or {}).get(page_id) or {}
    text_blocks: list[dict] = []
    for node_id, node in (page.get("nodes") or {}).items():
        kind = node.get("kind") or {}
        data = kind.get("text") if isinstance(kind, dict) else None
        if not isinstance(data, dict):
            continue
        text = str(data.get("text") or "").strip()
        if not text:
            continue
        transform = node.get("transform") or {}
        try:
            x = float(transform.get("x") or 0)
            y = float(transform.get("y") or 0)
            w = float(transform.get("width") or 0)
            h = float(transform.get("height") or 0)
        except Exception:
            x = y = w = h = 0.0
        bbox = [int(round(x)), int(round(y)), int(round(x + max(1.0, w))), int(round(y + max(1.0, h)))]
        line_polygons = data.get("linePolygons") or data.get("line_polygons") or []
        text_blocks.append(
            {
                "id": str(node_id),
                "bbox": bbox,
                "text_pixel_bbox": bbox,
                "confidence": float(data.get("confidence") or 0.0),
                "text": text,
                "detector": data.get("detector") or "koharu-paddle-ocr-vl-1.5",
                "line_polygons": line_polygons,
                "source_direction": data.get("sourceDirection") or data.get("source_direction"),
            }
        )
    return text_blocks


def _get_koharu_http_client(koharu_exe: Path) -> _KoharuHttpOcrClient:
    global _koharu_http_client
    with _koharu_http_client_lock:
        if (
            _koharu_http_client is None
            or _koharu_http_client.exe_path != koharu_exe
            or (_koharu_http_client.process is not None and _koharu_http_client.process.poll() is not None)
        ):
            if _koharu_http_client is not None:
                _koharu_http_client.stop()
            _koharu_http_client = _KoharuHttpOcrClient(koharu_exe)
        return _koharu_http_client


def _shutdown_koharu_http_client() -> None:
    global _koharu_http_client
    with _koharu_http_client_lock:
        if _koharu_http_client is not None:
            _koharu_http_client.stop()
            _koharu_http_client = None


atexit.register(_shutdown_koharu_http_client)


def _run_koharu_cjk_http_detect_ocr(
    image_rgb: np.ndarray,
    image_label: str,
    models_dir: str = "",
    profile: str = "quality",
    progress_callback=None,
    idioma_origem: str = "en",
    engine_preset_id: str = "",
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:
    koharu_exe = _resolve_koharu_exe(models_dir)
    if koharu_exe is None:
        raise FileNotFoundError("koharu.exe nao encontrado para OCR CJK")
    client = _get_koharu_http_client(koharu_exe)
    return client.run_ocr(
        image_path=image_label,
        image_rgb=image_rgb,
        profile=profile,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
        engine_preset_id=engine_preset_id,
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )


def _run_koharu_cjk_http_detect_ocr_batch(
    jobs: list[dict],
    models_dir: str = "",
    profile: str = "quality",
    progress_callback=None,
    idioma_origem: str = "en",
    engine_preset_id: str = "",
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> list[dict]:
    koharu_exe = _resolve_koharu_exe(models_dir)
    if koharu_exe is None:
        raise FileNotFoundError("koharu.exe nao encontrado para OCR CJK")
    client = _get_koharu_http_client(koharu_exe)
    return client.run_ocr_batch(
        jobs,
        profile=profile,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
        engine_preset_id=engine_preset_id,
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )


def _get_detector(profile: str = "quality", model: str = "comic-text-detector"):
    global _detector, _detector_model
    desired_model = str(model or "comic-text-detector")
    if _detector is None or _detector_model != desired_model:
        with _detector_lock:
            if _detector is None or _detector_model != desired_model:
                from .detector import TextDetector # type: ignore

                _detector = TextDetector(
                    model=desired_model,
                    device=_profile_to_device(profile),
                    half=True,
                )
                _detector_model = desired_model
    return _detector


def _detector_model_for_preset(engine_preset: EnginePreset | None) -> str:
    detector_id = str(getattr(engine_preset, "detector", "") or "").strip().lower()
    if detector_id in {"anime-text-yolo", "anime-text-yolo-n"}:
        return "anime-text-yolo-n"
    if detector_id in {"anime-text-yolo-s", "anime-text-yolo-m", "anime-text-yolo-l", "anime-text-yolo-x"}:
        return detector_id
    return "comic-text-detector"


def _get_ocr_engine(profile: str = "quality", lang: str = "en"):
    global _ocr_engine
    desired_model = _profile_to_ocr_model(profile)
    current_request = getattr(_ocr_engine, "_requested_model", getattr(_ocr_engine, "model_name", ""))
    current_lang = getattr(_ocr_engine, "lang", "en")
    
    if _ocr_engine is None or current_request != desired_model or current_lang != lang:
        with _ocr_engine_lock:
            current_request = getattr(_ocr_engine, "_requested_model", getattr(_ocr_engine, "model_name", ""))
            current_lang = getattr(_ocr_engine, "lang", "en")
            if _ocr_engine is None or current_request != desired_model or current_lang != lang:
                from .ocr import OCREngine # type: ignore

                _ocr_engine = OCREngine(
                    model=desired_model,
                    device=_profile_to_device(profile),
                    half=True,
                    lang=lang,
                )
    return _ocr_engine


def _get_inpainter(profile: str = "quality", model: str = "aot-inpainting"):
    global _inpainter, _inpainter_model
    desired_model = str(model or "aot-inpainting")
    if _inpainter is None or _inpainter_model != desired_model:
        with _inpainter_lock:
            if _inpainter is None or _inpainter_model != desired_model:
                from .inpainter import Inpainter # type: ignore

                _inpainter = Inpainter(
                    model=desired_model,
                    device=_profile_to_device(profile),
                    half=True,
                )
                _inpainter_model = desired_model
    return _inpainter


def _force_koharu_visual_engines(preset: dict | None) -> dict:
    normalized = dict(preset or {})
    normalized["segmenter"] = "comic-text-detector-seg"
    normalized["bubble_segmenter"] = "speech-bubble-segmentation"
    normalized["inpainter"] = "aot-inpainting"
    return normalized


def _page_engine_preset_dict(ocr_data: dict | None) -> dict:
    if not isinstance(ocr_data, dict):
        return {}
    preset = ocr_data.get("engine_preset")
    if isinstance(preset, dict):
        return _force_koharu_visual_engines(preset)
    preset_id = str(ocr_data.get("engine_preset_id") or "").strip()
    if preset_id:
        return _force_koharu_visual_engines(resolve_engine_preset({"engine_preset_id": preset_id}).to_dict())
    internal = ocr_data.get("_engine_preset")
    if isinstance(internal, dict):
        internal_id = str(internal.get("engine_preset_id") or "").strip()
        if internal_id:
            return _force_koharu_visual_engines(resolve_engine_preset({"engine_preset_id": internal_id}).to_dict())
        return _force_koharu_visual_engines(internal)
    return {}


def _inpainter_model_for_page(ocr_data: dict | None) -> str:
    preset = _page_engine_preset_dict(ocr_data)
    inpainter = str(preset.get("inpainter") or "").strip()
    if inpainter == "lama-manga":
        return "lama-manga"
    if inpainter == "aot-inpainting":
        return "aot-inpainting"
    return "aot-inpainting"


def _page_mask_strategy(ocr_data: dict | None) -> str:
    if not isinstance(ocr_data, dict):
        return ""
    preset = _page_engine_preset_dict(ocr_data)
    mask_strategy = str(preset.get("mask_strategy") or "").strip().lower()
    if mask_strategy:
        return mask_strategy
    internal = ocr_data.get("_engine_preset")
    if isinstance(internal, dict):
        return str(internal.get("mask_strategy") or "").strip().lower()
    return ""


def _strict_inpaint_mask_only_for_page(ocr_data: dict | None) -> bool:
    if _inpainter_model_for_page(ocr_data) != "aot-inpainting":
        return False
    return _page_mask_strategy(ocr_data) in {
        "segmentation_assisted",
        "roi_segmentation_assisted",
        "ocr_guided_segmentation",
        "ocr_guided_roi_segmentation",
    }


def _segmenter_model_for_page(ocr_data: dict | None) -> str:
    preset = _page_engine_preset_dict(ocr_data)
    segmenter = str(preset.get("segmenter") or "").strip()
    if segmenter == "manga-text-segmentation-2025":
        return "manga-text-segmentation-2025"
    return ""


def _get_text_segmenter_for_page(ocr_data: dict | None, profile: str = "quality"):
    global _text_segmenter, _text_segmenter_model
    desired_model = _segmenter_model_for_page(ocr_data)
    if desired_model != "manga-text-segmentation-2025":
        return None

    if _text_segmenter is None or _text_segmenter_model != desired_model:
        with _text_segmenter_lock:
            if _text_segmenter is None or _text_segmenter_model != desired_model:
                try:
                    from .manga_text_segmenter import MangaTextSegmenter

                    _text_segmenter = MangaTextSegmenter(
                        device=_profile_to_device(profile),
                        half=True,
                    )
                    _text_segmenter_model = desired_model
                except Exception as exc:
                    logger.warning("Manga-Text-Segmentation-2025 indisponivel; usando fallback geometrico: %s", exc)
                    _text_segmenter = None
                    _text_segmenter_model = ""
                    return None
    return _text_segmenter


def warmup_visual_stack(
    models_dir: str = "",
    profile: str = "quality",
    run_sample: bool = True,
    lang: str = "en",
):
    _configure_model_roots(models_dir)

    detector = _get_detector(profile)
    ocr = _get_ocr_engine(profile, lang=lang)
    font_detector = _get_font_detector()
    if not run_sample:
        return

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


def _block_should_skip_inpaint_mask(block: dict | None) -> bool:
    return not isinstance(block, dict)


def _text_cleanup_kinds(texts: list[dict] | None) -> tuple[bool, bool]:
    has_white = False
    has_textured = False
    for text in texts or []:
        if not isinstance(text, dict):
            continue
        if text.get("line_polygons") or text.get("text_pixel_bbox") or text.get("bubble_mask_bbox"):
            has_white = True
    return has_white, has_textured


def _text_has_nonwhite_cleanup_marker(text: dict) -> bool:
    return False


def _text_has_white_cleanup_marker(text: dict) -> bool:
    return False


def _text_background_looks_translucent_or_textured(image_rgb: np.ndarray, text: dict) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0 or not isinstance(text, dict):
        return False
    height, width = image_rgb.shape[:2]
    bbox = (
        _coerce_bbox(text.get("balloon_bbox"))
        or _coerce_bbox(text.get("layout_bbox"))
        or _coerce_bbox(text.get("bbox"))
    )
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return False

    try:
        sample_mask = _extract_white_balloon_fill_mask(image_rgb, [x1, y1, x2, y2])
    except Exception:
        sample_mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(sample_mask, np.ndarray) or not np.any(sample_mask):
        sample_mask = np.zeros((height, width), dtype=np.uint8)
        sample_mask[y1:y2, x1:x2] = 255
    else:
        safe = cv2.erode(
            sample_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        if np.any(safe):
            sample_mask = safe

    text_bbox = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(text.get("bbox"))
    if text_bbox is not None:
        tx1, ty1, tx2, ty2 = _expand_bbox(
            text_bbox,
            image_rgb.shape,
            pad_x_ratio=0.05,
            pad_y_ratio=0.12,
            min_pad_x=5,
            min_pad_y=5,
        )
        exclusion = np.zeros((height, width), dtype=np.uint8)
        exclusion[ty1:ty2, tx1:tx2] = 255
        exclusion = cv2.dilate(
            exclusion,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        sample_mask = cv2.bitwise_and(sample_mask.astype(np.uint8), cv2.bitwise_not(exclusion))

    if int(np.count_nonzero(sample_mask)) < 64:
        return False
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb.astype(np.uint8)
    bright_sample = (sample_mask > 0) & (gray >= 205)
    pixels = gray[bright_sample].astype(np.float32)
    if pixels.size < 64:
        return False
    mean_luma = float(np.mean(pixels))
    if mean_luma < 205.0:
        return False
    spread = float(np.percentile(pixels, 95) - np.percentile(pixels, 5))
    std = float(np.std(pixels))
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)[bright_sample]
    grad_p90 = float(np.percentile(grad, 90)) if grad.size else 0.0
    return spread >= 14.0 or std >= 5.5 or grad_p90 >= 18.0


def _text_anchor_has_white_cleanup_context(
    image_rgb: np.ndarray,
    text: dict,
    *,
    include_source_bbox: bool = True,
) -> bool:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0 or not isinstance(text, dict):
        return False
    height, width = image_rgb.shape[:2]
    candidates: list[list[int]] = []

    line_polygons = _normalize_line_polygons(text.get("line_polygons") or [])
    if line_polygons:
        xs: list[int] = []
        ys: list[int] = []
        for polygon in line_polygons:
            for px, py in polygon:
                xs.append(int(px))
                ys.append(int(py))
        if xs and ys:
            candidates.append([min(xs), min(ys), max(xs) + 1, max(ys) + 1])

    candidate_keys = ("source_bbox", "text_pixel_bbox", "layout_bbox") if include_source_bbox else ("text_pixel_bbox", "layout_bbox")
    for key in candidate_keys:
        bbox = _coerce_bbox(text.get(key))
        if bbox is not None and bbox not in candidates:
            candidates.append(bbox)

    for bbox in candidates:
        x1, y1, x2, y2 = bbox
        pad_x = max(5, int(round((x2 - x1) * 0.08)))
        pad_y = max(5, int(round((y2 - y1) * 0.20)))
        x1 = max(0, min(width, x1 - pad_x))
        x2 = max(0, min(width, x2 + pad_x))
        y1 = max(0, min(height, y1 - pad_y))
        y2 = max(0, min(height, y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.astype(np.uint8)
        if crop.ndim == 3:
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            bright = (gray >= 220) & (value >= 220) & (saturation <= 70)
        else:
            bright = gray >= 220
        if float(np.mean(bright)) < 0.48:
            continue
        bright_pixels = gray[bright]
        if bright_pixels.size < 24:
            continue
        if float(np.percentile(bright_pixels, 70)) >= 228.0:
            return True
    return False


def _text_is_white_cleanup_safe(image_rgb: np.ndarray, text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    white_marker = _text_has_white_cleanup_marker(text)
    nonwhite_marker = _text_has_nonwhite_cleanup_marker(text)
    anchor_white_context = _text_anchor_has_white_cleanup_context(
        image_rgb,
        text,
        include_source_bbox=False,
    )
    if nonwhite_marker:
        if not _normalize_line_polygons(text.get("line_polygons") or []):
            return False
        return anchor_white_context
    if anchor_white_context and _normalize_line_polygons(text.get("line_polygons") or []):
        return True
    if text.get("_white_cleanup_split_count") and anchor_white_context:
        return True
    if white_marker and anchor_white_context:
        return True
    if anchor_white_context and not _text_background_looks_translucent_or_textured(image_rgb, text):
        return True
    if _text_background_looks_translucent_or_textured(image_rgb, text):
        return False
    if white_marker:
        return True
    for key in ("balloon_bbox", "bbox", "text_pixel_bbox"):
        bbox = _coerce_bbox(text.get(key))
        if bbox is not None and _is_white_balloon_region(image_rgb, bbox):
            return True
    return False


def _white_cleanup_texts(image_rgb: np.ndarray, texts: list[dict] | None) -> list[dict]:
    cleanup_texts: list[dict] = []
    for text in texts or []:
        if not isinstance(text, dict):
            continue
        if _text_is_white_cleanup_safe(image_rgb, text):
            cleanup_texts.append(text)
            continue
        for split_text in _split_white_cleanup_candidate_by_line_gaps(text):
            if _text_is_white_cleanup_safe(image_rgb, split_text):
                cleanup_texts.append(split_text)
    return cleanup_texts


def _build_post_cleanup_limit_mask(
    limit_mask: np.ndarray | None,
    texts: list[dict] | None,
    shape: tuple[int, int],
    *,
    include_text_bboxes: bool = True,
) -> np.ndarray | None:
    if not isinstance(limit_mask, np.ndarray) or limit_mask.shape[:2] != shape:
        return None
    allowed = (limit_mask > 0).astype(np.uint8) * 255
    if not include_text_bboxes:
        return allowed
    height, width = shape
    for text in texts or []:
        if not isinstance(text, dict):
            continue
        bbox = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(text.get("bbox"))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            allowed[y1:y2, x1:x2] = 255
    return allowed


def _clamp_image_to_limit_mask(
    base_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    limit_mask: np.ndarray | None,
    texts: list[dict] | None = None,
    *,
    include_text_bboxes: bool = True,
) -> tuple[np.ndarray, int, int]:
    if base_rgb.shape[:2] != candidate_rgb.shape[:2]:
        return candidate_rgb, 0, 0
    cleanup_limit_mask = _build_post_cleanup_limit_mask(
        limit_mask,
        texts,
        candidate_rgb.shape[:2],
        include_text_bboxes=include_text_bboxes,
    )
    if cleanup_limit_mask is None:
        return candidate_rgb, 0, 0
    allowed = cleanup_limit_mask > 0
    changed_outside = np.any(candidate_rgb != base_rgb, axis=2) & ~allowed
    outside_count = int(np.count_nonzero(changed_outside))
    if not outside_count:
        return candidate_rgb, int(np.count_nonzero(allowed)), 0
    clamped = candidate_rgb.copy()
    clamped[~allowed] = base_rgb[~allowed]
    return clamped, int(np.count_nonzero(allowed)), outside_count


def _select_inpaint_roi(
    mask: np.ndarray,
    image_shape: tuple[int, int, int] | tuple[int, int],
    prefer_roi: bool = True,
    texts: list[dict] | None = None,
) -> tuple[list[int], bool]:
    if len(image_shape) == 3:
        height, width = image_shape[:2]
    else:
        height, width = image_shape

    full_bbox = [0, 0, int(width), int(height)]
    if not prefer_roi:
        return full_bbox, False

    mask_bbox = _mask_nonzero_bbox(mask)
    if mask_bbox is None:
        return full_bbox, False

    x1, y1, x2, y2 = mask_bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    if _inpaint_roi_tighten_enabled():
        has_white, has_textured = _text_cleanup_kinds(texts)
        min_side = max(1, min(box_w, box_h))
        if has_textured and not has_white:
            pad = max(16, int(round(min_side * 0.20)))
            pad_x = pad_y = pad
        elif has_white and box_w < 100 and box_h < 100:
            pad = max(8, int(round(min_side * 0.10)))
            pad_x = pad_y = pad
        else:
            pad = max(16, int(round(min_side * 0.16)))
            pad_x = pad_y = pad
    else:
        pad_x = max(96, int(box_w * 1.0))
        pad_y = max(96, int(box_h * 1.2))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(int(width), x2 + pad_x)
    ry2 = min(int(height), y2 + pad_y)
    if rx2 <= rx1 or ry2 <= ry1:
        return full_bbox, False

    full_area = max(1, int(width) * int(height))
    roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))
    if roi_area >= int(full_area * 0.88):
        return full_bbox, False

    return [rx1, ry1, rx2, ry2], True


def _clip_bbox_to_shape(
    bbox: list[int] | tuple[int, ...] | None,
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    if len(image_shape) == 3:
        height, width = image_shape[:2]
    else:
        height, width = image_shape
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    except Exception:
        return None
    x1 = max(0, min(int(width), x1))
    x2 = max(0, min(int(width), x2))
    y1 = max(0, min(int(height), y1))
    y2 = max(0, min(int(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _expand_bbox_by_pixels(
    bbox: list[int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    margin: int,
) -> list[int] | None:
    if len(image_shape) == 3:
        height, width = image_shape[:2]
    else:
        height, width = image_shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return _clip_bbox_to_shape(
        [x1 - margin, y1 - margin, x2 + margin, y2 + margin],
        (int(height), int(width)),
    )


def _strict_cjk_aot_crop_windows(
    mask: np.ndarray,
    vision_blocks: list[dict],
    image_shape: tuple[int, int, int] | tuple[int, int],
    *,
    margin: int = 128,
    max_contour_windows: int = 80,
) -> list[list[int]]:
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return []

    windows: list[list[int]] = []
    covered = np.zeros(mask.shape[:2], dtype=np.uint8)
    for block in vision_blocks or []:
        if not isinstance(block, dict):
            continue
        bbox = _clip_bbox_to_shape(
            block.get("bbox") or block.get("text_pixel_bbox") or block.get("balloon_bbox"),
            image_shape,
        )
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        local_mask = mask[y1:y2, x1:x2]
        local_bbox = _mask_nonzero_bbox(local_mask)
        if local_bbox is None:
            continue
        lx1, ly1, lx2, ly2 = local_bbox
        mask_bbox = [x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2]
        window = _expand_bbox_by_pixels(mask_bbox, image_shape, margin)
        if window is not None:
            windows.append(window)
            wx1, wy1, wx2, wy2 = window
            covered[wy1:wy2, wx1:wx2] = 255

    uncovered = cv2.bitwise_and((mask > 0).astype(np.uint8) * 255, cv2.bitwise_not(covered))
    if np.any(uncovered):
        binary = uncovered
        contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_boxes: list[tuple[int, list[int]]] = []
        for contour in contours:
            if contour is None or len(contour) == 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            area = max(int(cv2.contourArea(contour)), int(w) * int(h))
            if area < 16:
                continue
            contour_boxes.append((area, [int(x), int(y), int(x + w), int(y + h)]))
        contour_boxes.sort(key=lambda item: (-item[0], item[1][1], item[1][0]))
        for _area, bbox in contour_boxes[:max_contour_windows]:
            window = _expand_bbox_by_pixels(bbox, image_shape, margin)
            if window is not None:
                windows.append(window)

    unique: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for window in windows:
        key = tuple(int(v) for v in window)
        if key in seen:
            continue
        seen.add(key)
        unique.append(window)
    return unique


def _clustered_inpaint_crop_windows(
    mask: np.ndarray,
    image_shape: tuple[int, int, int] | tuple[int, int],
    *,
    texts: list[dict] | None = None,
) -> list[list[int]] | None:
    if not _inpaint_clustered_crop_windows_enabled():
        return None
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return None
    if len(image_shape) == 3:
        height, width = image_shape[:2]
    else:
        height, width = image_shape
    full_area = max(1, int(width) * int(height))
    single_roi, _single_uses_roi = _select_inpaint_roi(mask, image_shape, prefer_roi=True, texts=texts)
    single_roi_area = max(1, _bbox_area_safe(single_roi))

    margin = _env_int("TRADUZAI_INPAINT_CLUSTERED_CROP_MARGIN", 72, min_value=8, max_value=256)
    max_windows = _env_int("TRADUZAI_INPAINT_CLUSTERED_CROP_MAX_WINDOWS", 12, min_value=2, max_value=80)
    min_component_area = _env_int("TRADUZAI_INPAINT_CLUSTERED_CROP_MIN_COMPONENT_AREA", 16, min_value=1)

    binary = (mask > 0).astype(np.uint8)
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    windows: list[list[int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w_box = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w_box <= 0 or h_box <= 0:
            continue
        window = _expand_bbox_by_pixels([x, y, x + w_box, y + h_box], image_shape, margin)
        if window is not None:
            windows.append(window)

    if len(windows) <= 1:
        return None

    merged = _merge_nearby_bboxes(windows, gap_x=0, gap_y=0)
    merged = sorted(
        merged,
        key=lambda box: (
            int(box[1]),
            int(box[0]),
            -_bbox_area_safe(box),
        ),
    )
    if len(merged) <= 1 or len(merged) > max_windows:
        return None

    crop_area = sum(max(1, _bbox_area_safe(window)) for window in merged)
    savings_ratio = crop_area / float(max(1, single_roi_area))
    full_ratio = crop_area / float(full_area)
    max_single_roi_ratio = float(
        _env_int("TRADUZAI_INPAINT_CLUSTERED_CROP_MAX_SINGLE_ROI_PCT", 72, min_value=10, max_value=95)
    ) / 100.0
    max_full_ratio = float(
        _env_int("TRADUZAI_INPAINT_CLUSTERED_CROP_MAX_FULL_PCT", 80, min_value=10, max_value=98)
    ) / 100.0
    if savings_ratio > max_single_roi_ratio or full_ratio > max_full_ratio:
        return None

    return merged


def _call_inpainter_in_roi(
    inpainter,
    image_np: np.ndarray,
    mask: np.ndarray,
    roi_bbox: list[int],
    use_roi: bool,
    batch_size: int = 4,
    debug: DebugRunRecorder | None = None,
    force_no_tiling: bool = False,
) -> np.ndarray:
    if not use_roi:
        return _call_inpainter(
            inpainter,
            image_np,
            mask,
            batch_size=batch_size,
            debug=debug,
            force_no_tiling=force_no_tiling,
        )

    rx1, ry1, rx2, ry2 = roi_bbox
    crop_image = image_np[ry1:ry2, rx1:rx2].copy()
    crop_mask = mask[ry1:ry2, rx1:rx2].copy()
    crop_output = _call_inpainter(
        inpainter,
        crop_image,
        crop_mask,
        batch_size=batch_size,
        debug=debug,
        force_no_tiling=force_no_tiling,
    )
    if crop_output.shape[:2] != crop_image.shape[:2]:
        raise ValueError(
            f"roi inpaint retornou shape {crop_output.shape[:2]} esperado {crop_image.shape[:2]}"
        )

    result = image_np.copy()
    target = result[ry1:ry2, rx1:rx2]
    if _inpaint_roi_tighten_enabled():
        alpha = (crop_mask > 0).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        blended = crop_output.astype(np.float32) * alpha + target.astype(np.float32) * (1.0 - alpha)
        target[:] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        paste_mask = crop_mask > 0
        target[paste_mask] = crop_output[paste_mask]
    result[ry1:ry2, rx1:rx2] = target
    return result


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

    serialized = {
        "bbox": [x1, y1, x2, y2],
        "mask": local_mask,
        "confidence": float(getattr(block, "confidence", 0.0)),
    }
    for key in (
        "balloon_bbox",
        "balloon_polygon",
        "balloon_subregions",
        "connected_lobe_bboxes",
        "connected_lobe_ids",
        "connected_lobe_polygons",
        "bubble_id",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
    ):
        value = getattr(block, key, None)
        if value not in (None, [], ""):
            serialized[key] = value
    return serialized


def _apply_text_geometry_to_serialized_block(serialized_block: dict, text_entry: dict) -> dict:
    enriched = dict(serialized_block)
    anchor_bbox = resolve_text_anchor_bbox(text_entry)
    source_bbox = _coerce_bbox(enriched.get("bbox")) or _coerce_bbox(text_entry.get("bbox"))
    if source_bbox is not None:
        enriched["bbox"] = list(source_bbox)
        enriched.setdefault("source_bbox", list(source_bbox))
    if anchor_bbox is not None:
        enriched["text_pixel_bbox"] = list(anchor_bbox)
    for key in (
        "line_polygons",
        "balloon_bbox",
        "balloon_polygon",
        "balloon_subregions",
        "connected_lobe_bboxes",
        "connected_lobe_ids",
        "connected_lobe_polygons",
        "bubble_id",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "balloon_type",
        "tipo",
        "block_profile",
        "page_profile",
        "_merged_source_bboxes",
        "merged_source_bboxes",
        "text",
        "original",
        "raw_ocr",
        "translated",
        "traduzido",
        "skip_processing",
        "skip_reason",
        "preserve_original",
        "translate_policy",
        "render_policy",
        "route_action",
        "route_reason",
        "content_class",
        "is_watermark",
        "is_non_english",
        "qa_flags",
        "qa_metrics",
        "_validated_text_source_bboxes",
        "_rejected_text_source_bboxes",
        "_raw_text_evidence_bbox",
        "_raw_text_evidence_pixels",
        "validated_by_segment_mask",
        "detector_preset_id",
        "detector_engine_id",
        "detector_loader",
        "candidate_kind",
        "rotation_deg",
        "rotation_source",
    ):
        value = text_entry.get(key)
        if value not in (None, [], ""):
            enriched[key] = value
    return enriched


def _normalize_rotation_metadata_value(value) -> float:
    try:
        numeric = float(value or 0)
    except Exception:
        return 0.0
    normalized = numeric % 360.0
    if normalized > 180.0:
        normalized -= 360.0
    if normalized <= -180.0:
        normalized += 360.0
    if abs(normalized) < 0.01:
        return 0.0
    return round(normalized, 2)


def _rotation_metadata_from_ocr(raw_record: dict, block, line_polygons: list) -> tuple[float, str]:
    candidates = (
        (
            raw_record.get("rotation_deg") if isinstance(raw_record, dict) else None,
            raw_record.get("rotation_source") if isinstance(raw_record, dict) else None,
            "ocr",
        ),
        (
            getattr(block, "rotation_deg", None) if not isinstance(block, dict) else block.get("rotation_deg"),
            getattr(block, "rotation_source", None) if not isinstance(block, dict) else block.get("rotation_source"),
            "detector",
        ),
    )
    for value, source, fallback_source in candidates:
        rotation = _normalize_rotation_metadata_value(value)
        if rotation != 0.0:
            return rotation, str(source or fallback_source)

    inferred = infer_rotation_deg_from_line_polygons(line_polygons)
    if inferred != 0.0:
        return inferred, "line_polygons"
    return 0.0, ""


def _normalize_geometry_polygon(value, page_shape: tuple[int, int]) -> list[list[int]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    height, width = int(page_shape[0]), int(page_shape[1])
    points: list[list[int]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
        except Exception:
            return None
        points.append([max(0, min(width - 1, x)), max(0, min(height - 1, y))])
    return points if len(points) >= 3 else None


def _normalize_geometry_polygons(value, page_shape: tuple[int, int]) -> list[list[list[int]]]:
    if not isinstance(value, (list, tuple)) or not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2 and not (
        first and isinstance(first[0], (list, tuple))
    ):
        polygon = _normalize_geometry_polygon(value, page_shape)
        return [polygon] if polygon else []
    polygons = []
    for item in value:
        polygon = _normalize_geometry_polygon(item, page_shape)
        if polygon:
            polygons.append(polygon)
    return polygons


def _geometry_value(raw_record: dict, block, *keys: str):
    for key in keys:
        if isinstance(raw_record, dict) and raw_record.get(key) not in (None, [], ""):
            return raw_record.get(key)
        value = getattr(block, key, None)
        if value not in (None, [], ""):
            return value
    return None


def _apply_balloon_geometry_to_text_entry(
    text_entry: dict,
    raw_record: dict,
    block,
    page_shape: tuple[int, int],
) -> dict:
    balloon_bbox = _coerce_bbox(
        _geometry_value(raw_record, block, "balloon_bbox", "balloonBBox")
    )
    if balloon_bbox is not None:
        text_entry["balloon_bbox"] = balloon_bbox

    balloon_polygon = _normalize_geometry_polygon(
        _geometry_value(raw_record, block, "balloon_polygon", "balloonPolygon"),
        page_shape,
    )
    if balloon_polygon:
        text_entry["balloon_polygon"] = balloon_polygon

    connected_polygons = _normalize_geometry_polygons(
        _geometry_value(raw_record, block, "connected_lobe_polygons", "connectedLobePolygons"),
        page_shape,
    )
    if connected_polygons:
        text_entry["connected_lobe_polygons"] = connected_polygons

    for key, camel in (
        ("balloon_subregions", "balloonSubregions"),
        ("connected_lobe_bboxes", "connectedLobeBboxes"),
        ("_validated_text_source_bboxes", "validatedTextSourceBboxes"),
        ("_rejected_text_source_bboxes", "rejectedTextSourceBboxes"),
    ):
        values = _geometry_value(raw_record, block, key, camel)
        bboxes = []
        if isinstance(values, list):
            for value in values:
                bbox = _coerce_bbox(value)
                if bbox is not None:
                    bboxes.append(bbox)
        if bboxes:
            text_entry[key] = bboxes

    for key, camel in (
        ("bubble_mask_bbox", "bubbleMaskBbox"),
        ("bubble_inner_bbox", "bubbleInnerBbox"),
    ):
        bbox = _coerce_bbox(_geometry_value(raw_record, block, key, camel))
        if bbox is not None:
            text_entry[key] = bbox
    for key, camel in (
        ("bubble_id", "bubbleId"),
        ("connected_lobe_ids", "connectedLobeIds"),
    ):
        value = _geometry_value(raw_record, block, key, camel)
        if value not in (None, [], ""):
            text_entry[key] = value
    for key in (
        "_raw_text_evidence_bbox",
        "_raw_text_evidence_pixels",
        "validated_by_segment_mask",
        "detector_preset_id",
        "detector_engine_id",
        "detector_loader",
        "candidate_kind",
    ):
        value = _geometry_value(raw_record, block, key)
        if value not in (None, [], ""):
            text_entry[key] = value
    return text_entry


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


def _normalized_bbox_list(values) -> list[list[int]]:
    normalized: list[list[int]] = []
    if not isinstance(values, (list, tuple)):
        return normalized
    for value in values:
        bbox = _coerce_bbox(value)
        if bbox is None:
            continue
        if not any(_bbox_iou(bbox, existing) >= 0.94 for existing in normalized):
            normalized.append(bbox)
    return normalized


def _bbox_union_many(bboxes: list[list[int]]) -> list[int] | None:
    union_bbox = None
    for bbox in bboxes:
        union_bbox = list(bbox) if union_bbox is None else _bbox_union(union_bbox, bbox)
    return union_bbox


def _copy_validated_source_metadata(target: dict, source: dict) -> None:
    for key in (
        "_validated_text_source_bboxes",
        "_rejected_text_source_bboxes",
        "_raw_text_evidence_bbox",
        "_raw_text_evidence_pixels",
        "validated_by_segment_mask",
        "detector_preset_id",
        "detector_engine_id",
        "detector_loader",
        "candidate_kind",
    ):
        value = source.get(key)
        if value not in (None, [], "") and target.get(key) in (None, [], ""):
            target[key] = value


def _validated_source_bboxes_for_text(text: dict, vision_blocks: list[dict]) -> list[list[int]]:
    direct = _normalized_bbox_list(text.get("_validated_text_source_bboxes"))
    if direct:
        return direct
    text_bbox = _coerce_bbox(text.get("source_bbox") or text.get("text_pixel_bbox") or text.get("bbox"))
    if text_bbox is None:
        return []
    best_block = None
    best_score = 0.0
    for block in vision_blocks:
        candidates = _normalized_bbox_list(block.get("_validated_text_source_bboxes"))
        if not candidates:
            continue
        block_bbox = _coerce_bbox(block.get("source_bbox") or block.get("text_pixel_bbox") or block.get("bbox"))
        union_bbox = _bbox_union_many(candidates)
        score = 0.0
        if block_bbox is not None:
            score = max(score, _bbox_inner_overlap_ratio(text_bbox, block_bbox))
            score = max(score, _bbox_inner_overlap_ratio(block_bbox, text_bbox))
        if union_bbox is not None:
            score = max(score, _bbox_inner_overlap_ratio(union_bbox, text_bbox))
        if score > best_score:
            best_score = score
            best_block = block
    if best_block is None or best_score < 0.18:
        return []
    _copy_validated_source_metadata(text, best_block)
    return _normalized_bbox_list(best_block.get("_validated_text_source_bboxes"))


def _validated_sources_are_separated(bboxes: list[list[int]]) -> bool:
    if len(bboxes) < 2:
        return False
    ordered = sorted(bboxes, key=lambda bbox: (bbox[1], bbox[0]))
    for previous, current in zip(ordered, ordered[1:]):
        vertical_gap = int(current[1]) - int(previous[3])
        horizontal_gap = int(current[0]) - int(previous[2])
        min_h = max(1, min(previous[3] - previous[1], current[3] - current[1]))
        min_w = max(1, min(previous[2] - previous[0], current[2] - current[0]))
        if vertical_gap > max(18, int(min_h * 0.70)) or horizontal_gap > max(24, int(min_w * 0.70)):
            return True
    return False


def _split_text_value_for_validated_sources(text: dict, split_count: int) -> list[str] | None:
    for key in ("line_texts", "text_lines", "ocr_lines", "source_lines"):
        raw_lines = text.get(key)
        if not isinstance(raw_lines, (list, tuple)):
            continue
        lines: list[str] = []
        for raw_line in raw_lines:
            if isinstance(raw_line, dict):
                value = raw_line.get("text") or raw_line.get("value") or raw_line.get("content")
            else:
                value = raw_line
            normalized = str(value or "").strip()
            if normalized:
                lines.append(normalized)
        if len(lines) == split_count:
            return lines
    raw_text = str(text.get("text") or text.get("original") or "").strip()
    if raw_text:
        lines = [line.strip() for line in re.split(r"\r?\n+", raw_text) if line.strip()]
        if len(lines) == split_count:
            return lines
    return None


def _line_polygons_by_validated_source(
    line_polygons,
    validated_bboxes: list[list[int]],
) -> list[list[list[list[int]]]] | None:
    polygons = _normalize_line_polygons(line_polygons)
    if len(polygons) < len(validated_bboxes) or len(validated_bboxes) < 2:
        return None
    groups: list[list[list[list[int]]]] = [[] for _ in validated_bboxes]
    for polygon in polygons:
        polygon_bbox = _bbox_from_line_polygons([polygon])
        if polygon_bbox is None:
            return None
        best_index = -1
        best_score = 0.0
        center_x = (polygon_bbox[0] + polygon_bbox[2]) / 2.0
        center_y = (polygon_bbox[1] + polygon_bbox[3]) / 2.0
        for index, source_bbox in enumerate(validated_bboxes):
            score = _bbox_inner_overlap_ratio(polygon_bbox, source_bbox)
            if source_bbox[0] <= center_x <= source_bbox[2] and source_bbox[1] <= center_y <= source_bbox[3]:
                score = max(score, 0.95)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0 or best_score < 0.45:
            return None
        groups[best_index].append(polygon)
    if any(not group for group in groups):
        return None
    return groups


def _split_text_by_validated_sources(text: dict, validated_bboxes: list[list[int]]) -> list[dict]:
    if len(validated_bboxes) < 2 or not _validated_sources_are_separated(validated_bboxes):
        return []
    polygon_groups = _line_polygons_by_validated_source(text.get("line_polygons"), validated_bboxes)
    if not polygon_groups:
        return []
    line_texts = _split_text_value_for_validated_sources(text, len(validated_bboxes))
    if line_texts is None:
        return []
    split_items: list[dict] = []
    parent_id = str(text.get("id") or text.get("text_id") or text.get("trace_id") or "ocr")
    for index, (source_bbox, group, line_text) in enumerate(zip(validated_bboxes, polygon_groups, line_texts), start=1):
        polygon_bbox = _bbox_from_line_polygons(group) or list(source_bbox)
        child = dict(text)
        child["text"] = line_text
        if "original" in child:
            child["original"] = line_text
        child_id = f"{parent_id}_validated_{index:02d}"
        child["id"] = child_id
        child["text_id"] = child_id
        child["bbox"] = list(polygon_bbox)
        child["text_pixel_bbox"] = list(polygon_bbox)
        child["source_bbox"] = list(source_bbox)
        child["layout_bbox"] = list(source_bbox)
        child["line_polygons"] = [polygon for polygon in group]
        child["_validated_text_source_bboxes"] = [list(source_bbox)]
        child["validated_by_segment_mask"] = True
        child["_render_target_source"] = "validated_text_source"
        child["_validated_source_split_parent_id"] = parent_id
        child["_validated_source_split_index"] = index
        child["_validated_source_split_count"] = len(validated_bboxes)
        flags = child.setdefault("qa_flags", [])
        if isinstance(flags, list) and "ocr_split_validated_sources" not in flags:
            flags.append("ocr_split_validated_sources")
        split_items.append(child)
    return split_items


def _reconcile_ocr_with_validated_sources(page_result: dict) -> dict:
    if not isinstance(page_result, dict):
        return page_result
    vision_blocks = [block for block in page_result.get("_vision_blocks", []) if isinstance(block, dict)]
    reconciled_texts: list[dict] = []
    for text in page_result.get("texts", []) or []:
        if not isinstance(text, dict):
            reconciled_texts.append(text)
            continue
        validated_bboxes = _validated_source_bboxes_for_text(text, vision_blocks)
        if not validated_bboxes:
            text.setdefault("validated_by_segment_mask", False)
            reconciled_texts.append(text)
            continue
        union_bbox = _bbox_union_many(validated_bboxes)
        if union_bbox is None:
            reconciled_texts.append(text)
            continue
        split_items = _split_text_by_validated_sources(text, validated_bboxes)
        if split_items:
            reconciled_texts.extend(split_items)
            continue
        text["_validated_text_source_bboxes"] = validated_bboxes
        text["validated_by_segment_mask"] = True
        original_bbox = _coerce_bbox(text.get("ocr_text_bbox") or text.get("bbox") or text.get("text_pixel_bbox"))
        if original_bbox is not None:
            original_area = max(1, _bbox_area_safe(original_bbox))
            union_area = max(1, _bbox_area_safe(union_bbox))
            if original_area >= union_area * 1.85 or _bbox_inner_overlap_ratio(union_bbox, original_bbox) < 0.86:
                flags = text.setdefault("qa_flags", [])
                if isinstance(flags, list) and "ocr_overmerged_validated_sources" not in flags:
                    flags.append("ocr_overmerged_validated_sources")
        if _validated_sources_are_separated(validated_bboxes):
            flags = text.setdefault("qa_flags", [])
            if isinstance(flags, list) and "ocr_multiple_validated_sources" not in flags:
                flags.append("ocr_multiple_validated_sources")
            text["_render_target_source"] = "validated_text_source"
        text["layout_bbox"] = list(union_bbox)
        if not text.get("line_polygons"):
            text["text_pixel_bbox"] = list(union_bbox)
        text.setdefault("ocr_text_bbox", original_bbox or list(union_bbox))
        reconciled_texts.append(text)
    page_result["texts"] = reconciled_texts
    return page_result


def _orientation_recovery_enabled() -> bool:
    value = os.getenv("TRADUZAI_ORIENTATION_RECOVERY", "1")
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _rotate_image_for_orientation(image_rgb: np.ndarray, rotation_deg: int) -> np.ndarray:
    normalized = int(rotation_deg) % 360
    if normalized == 90:
        return np.rot90(image_rgb, k=-1).copy()
    if normalized == 180:
        return np.rot90(image_rgb, k=2).copy()
    if normalized == 270:
        return np.rot90(image_rgb, k=1).copy()
    return image_rgb.copy()


def _map_orientation_point_to_original(
    point: tuple[float, float],
    rotation_deg: int,
    original_shape: tuple[int, int],
) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    original_h, original_w = int(original_shape[0]), int(original_shape[1])
    normalized = int(rotation_deg) % 360
    if normalized == 90:
        return y, original_h - x
    if normalized == 180:
        return original_w - x, original_h - y
    if normalized == 270:
        return original_w - y, x
    return x, y


def _clamp_bbox_to_shape(bbox: list[int], shape: tuple[int, int]) -> list[int]:
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _rotate_bbox_from_view_to_original(
    bbox: list[int] | tuple[int, int, int, int],
    rotation_deg: int,
    original_shape: tuple[int, int],
) -> list[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    points = [
        _map_orientation_point_to_original((x1, y1), rotation_deg, original_shape),
        _map_orientation_point_to_original((x2, y1), rotation_deg, original_shape),
        _map_orientation_point_to_original((x2, y2), rotation_deg, original_shape),
        _map_orientation_point_to_original((x1, y2), rotation_deg, original_shape),
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return _clamp_bbox_to_shape(
        [math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))],
        original_shape,
    )


def _rotate_polygon_from_view_to_original(
    polygon,
    rotation_deg: int,
    original_shape: tuple[int, int],
) -> list[list[int]]:
    mapped = []
    for point in polygon or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x, y = _map_orientation_point_to_original((point[0], point[1]), rotation_deg, original_shape)
        mapped.append([int(round(x)), int(round(y))])
    return mapped


def _full_mask_from_rotated_block(
    mask: np.ndarray,
    bbox: list[int],
    rotated_shape: tuple[int, int],
) -> np.ndarray:
    rotated_h, rotated_w = int(rotated_shape[0]), int(rotated_shape[1])
    if mask.shape[:2] == (rotated_h, rotated_w):
        return mask.astype(np.uint8)
    full = np.zeros((rotated_h, rotated_w), dtype=np.uint8)
    x1, y1, x2, y2 = _clamp_bbox_to_shape(bbox, (rotated_h, rotated_w))
    if x2 <= x1 or y2 <= y1:
        return full
    patch = mask.astype(np.uint8)
    target_h = max(1, y2 - y1)
    target_w = max(1, x2 - x1)
    if patch.shape[:2] != (target_h, target_w):
        patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    full[y1:y2, x1:x2] = np.maximum(full[y1:y2, x1:x2], patch[:target_h, :target_w])
    return full


def _rotate_mask_from_view_to_original(
    mask: np.ndarray,
    bbox: list[int],
    rotation_deg: int,
    original_shape: tuple[int, int],
    rotated_shape: tuple[int, int],
) -> np.ndarray:
    full = _full_mask_from_rotated_block(mask, bbox, rotated_shape)
    normalized = int(rotation_deg) % 360
    if normalized == 90:
        restored = np.rot90(full, k=1)
    elif normalized == 180:
        restored = np.rot90(full, k=2)
    elif normalized == 270:
        restored = np.rot90(full, k=-1)
    else:
        restored = full
    original_h, original_w = int(original_shape[0]), int(original_shape[1])
    if restored.shape[:2] != (original_h, original_w):
        restored = cv2.resize(restored, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return restored.astype(np.uint8)


def _remap_orientation_recovery_page(
    page_result: dict,
    rotation_deg: int,
    original_shape: tuple[int, int],
    rotated_shape: tuple[int, int],
) -> dict:
    remapped = _clone_page_result(page_result)
    original_h, original_w = int(original_shape[0]), int(original_shape[1])
    remapped["width"] = original_w
    remapped["height"] = original_h
    remapped["orientation_recovery_deg"] = int(rotation_deg)
    remapped["orientation_recovered"] = True
    if remapped.get("texts"):
        remapped["sem_texto_detectado"] = False

    bbox_keys = (
        "bbox",
        "source_bbox",
        "layout_bbox",
        "text_pixel_bbox",
        "balloon_bbox",
        "render_bbox",
    )
    list_bbox_keys = (
        "balloon_subregions",
        "connected_lobe_bboxes",
        "connected_text_groups",
        "connected_position_bboxes",
        "connected_focus_bboxes",
        "_validated_text_source_bboxes",
        "_rejected_text_source_bboxes",
    )

    for text in remapped.get("texts", []):
        for key in bbox_keys:
            value = text.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 4:
                text[key] = _rotate_bbox_from_view_to_original(value, rotation_deg, original_shape)
        for key in list_bbox_keys:
            value = text.get(key)
            if isinstance(value, list):
                text[key] = [
                    _rotate_bbox_from_view_to_original(item, rotation_deg, original_shape)
                    for item in value
                    if isinstance(item, (list, tuple)) and len(item) == 4
                ]
        polygons = text.get("line_polygons")
        if isinstance(polygons, list):
            text["line_polygons"] = [
                _rotate_polygon_from_view_to_original(polygon, rotation_deg, original_shape)
                for polygon in polygons
                if isinstance(polygon, list)
            ]
        polygon = text.get("balloon_polygon")
        if isinstance(polygon, list):
            rotated = _rotate_polygon_from_view_to_original(polygon, rotation_deg, original_shape)
            if rotated:
                text["balloon_polygon"] = rotated
        polygons = text.get("connected_lobe_polygons")
        if isinstance(polygons, list):
            text["connected_lobe_polygons"] = [
                _rotate_polygon_from_view_to_original(polygon, rotation_deg, original_shape)
                for polygon in polygons
                if isinstance(polygon, list)
            ]
        text["orientation_recovery_deg"] = int(rotation_deg)

    for block in remapped.get("_vision_blocks", []):
        bbox = block.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            old_bbox = [int(v) for v in bbox]
            block["bbox"] = _rotate_bbox_from_view_to_original(old_bbox, rotation_deg, original_shape)
            mask = block.get("mask")
            if isinstance(mask, np.ndarray) and mask.size > 0:
                block["mask"] = _rotate_mask_from_view_to_original(
                    mask,
                    old_bbox,
                    rotation_deg,
                    original_shape,
                    rotated_shape,
                )
        for key in (
            "source_bbox",
            "layout_bbox",
            "text_pixel_bbox",
            "balloon_bbox",
            "_raw_text_evidence_bbox",
        ):
            value = block.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 4:
                block[key] = _rotate_bbox_from_view_to_original(value, rotation_deg, original_shape)
        for key in (
            "balloon_subregions",
            "connected_lobe_bboxes",
            "_validated_text_source_bboxes",
            "_rejected_text_source_bboxes",
        ):
            value = block.get(key)
            if isinstance(value, list):
                block[key] = [
                    _rotate_bbox_from_view_to_original(item, rotation_deg, original_shape)
                    for item in value
                    if isinstance(item, (list, tuple)) and len(item) == 4
                ]
        polygon = block.get("balloon_polygon")
        if isinstance(polygon, list):
            rotated = _rotate_polygon_from_view_to_original(polygon, rotation_deg, original_shape)
            if rotated:
                block["balloon_polygon"] = rotated
        polygons = block.get("connected_lobe_polygons")
        if isinstance(polygons, list):
            block["connected_lobe_polygons"] = [
                _rotate_polygon_from_view_to_original(polygon, rotation_deg, original_shape)
                for polygon in polygons
                if isinstance(polygon, list)
            ]
        block["orientation_recovery_deg"] = int(rotation_deg)
    return remapped


def _orientation_result_score(page_result: dict) -> tuple[int, int, float, int]:
    texts = page_result.get("texts", []) or []
    non_empty = [str(item.get("text") or item.get("original") or "").strip() for item in texts]
    non_empty = [text for text in non_empty if text]
    char_count = sum(len(text) for text in non_empty)
    confidences = [
        float(item.get("confidence", item.get("ocr_confidence", item.get("confianca_ocr", 0.0))) or 0.0)
        for item in texts
    ]
    avg_confidence = sum(confidences) / float(len(confidences)) if confidences else 0.0
    return (len(non_empty), char_count, avg_confidence, len(page_result.get("_vision_blocks", []) or []))


def _should_try_orientation_recovery(page_result: dict) -> bool:
    if not _orientation_recovery_enabled():
        return False
    accepted, chars, _avg_confidence, blocks = _orientation_result_score(page_result)
    return accepted == 0 or (accepted <= 1 and chars <= 2 and blocks <= 2)


def _normalize_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _text_matches_work_title(text: str, work_title: str = "", aliases: list[str] | tuple[str, ...] | None = None) -> bool:
    text_key = _normalize_text_key(text).replace(" ", "")
    if len(text_key) < 3:
        return False
    candidates = [work_title, *(aliases or [])]
    for candidate in candidates:
        candidate_key = _normalize_text_key(candidate).replace(" ", "")
        if len(candidate_key) < 3:
            continue
        if text_key == candidate_key or text_key in candidate_key or candidate_key in text_key:
            return True
    return False


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


def _compose_ocr_cluster_text(texts: list[dict]) -> str:
    ordered = sorted(
        texts,
        key=lambda item: (
            int(item.get("bbox", [0, 0, 0, 0])[1]),
            int(item.get("bbox", [0, 0, 0, 0])[0]),
        ),
    )
    parts = [str(item.get("text", "") or "").strip() for item in ordered]
    parts = [part for part in parts if part]
    return " ".join(parts).strip()


def _compose_ocr_cluster_field(texts: list[dict], *keys: str) -> str:
    ordered = sorted(
        texts,
        key=lambda item: (
            int(item.get("bbox", [0, 0, 0, 0])[1]),
            int(item.get("bbox", [0, 0, 0, 0])[0]),
        ),
    )
    parts: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        value = ""
        for key in keys:
            value = str(item.get(key, "") or "").strip()
            if value:
                break
        if not value:
            continue
        norm = _normalize_text_key(value)
        compact = norm.replace(" ", "")
        if compact and compact in seen:
            continue
        if parts and norm and (_normalize_text_key(parts[-1]).replace(" ", "") == compact):
            continue
        parts.append(value)
        if compact:
            seen.add(compact)
    return " ".join(parts).strip()


def _merge_local_block_masks(blocks: list[dict], merged_bbox: list[int]) -> np.ndarray | None:
    mx1, my1, mx2, my2 = [int(v) for v in merged_bbox]
    merged_h = max(1, my2 - my1)
    merged_w = max(1, mx2 - mx1)
    merged_mask = np.zeros((merged_h, merged_w), dtype=np.uint8)
    has_mask = False

    for block in blocks:
        bbox = [int(v) for v in block.get("bbox", [0, 0, 0, 0])]
        bx1, by1, bx2, by2 = bbox
        if bx2 <= bx1 or by2 <= by1:
            continue
        local_mask = block.get("mask")
        if not isinstance(local_mask, np.ndarray) or local_mask.size == 0:
            continue

        expected_h = max(1, by2 - by1)
        expected_w = max(1, bx2 - bx1)
        if local_mask.shape[:2] != (expected_h, expected_w):
            local_mask = cv2.resize(
                local_mask,
                (expected_w, expected_h),
                interpolation=cv2.INTER_NEAREST,
            )

        offset_x = max(0, bx1 - mx1)
        offset_y = max(0, by1 - my1)
        paste_h = min(local_mask.shape[0], merged_mask.shape[0] - offset_y)
        paste_w = min(local_mask.shape[1], merged_mask.shape[1] - offset_x)
        if paste_h <= 0 or paste_w <= 0:
            continue
        merged_mask[offset_y:offset_y + paste_h, offset_x:offset_x + paste_w] = np.maximum(
            merged_mask[offset_y:offset_y + paste_h, offset_x:offset_x + paste_w],
            local_mask[:paste_h, :paste_w],
        )
        has_mask = True

    return merged_mask if has_mask else None


def _text_fragment_bbox(text: dict) -> list[int] | None:
    return _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(text.get("bbox"))


def _text_fragment_source_bbox(text: dict) -> list[int] | None:
    source_bbox = _coerce_bbox(text.get("source_bbox"))
    if source_bbox is not None:
        return source_bbox
    raw_bbox = _coerce_bbox(text.get("bbox"))
    text_bbox = _coerce_bbox(text.get("text_pixel_bbox"))
    return text_bbox or raw_bbox


def _text_fragment_has_white_balloon_marker(text: dict) -> bool:
    return bool(text.get("bubble_id") or text.get("bubble_mask_bbox") or text.get("bubble_mask"))


def _text_fragment_can_merge_by_geometry(text: dict) -> bool:
    cleaned = str(text.get("text", "") or "").strip()
    if not cleaned:
        return False
    return True


def _text_fragment_line_height(text: dict, bbox: list[int]) -> float:
    heights: list[float] = []
    for polygon in _normalize_line_polygons(text.get("line_polygons") or []):
        ys = [float(point[1]) for point in polygon]
        if ys:
            heights.append(max(1.0, max(ys) - min(ys)))
    if heights:
        heights.sort()
        return float(heights[len(heights) // 2])
    return float(max(1, int(bbox[3]) - int(bbox[1])))


def _text_fragments_share_source_context(first: dict, second: dict) -> bool:
    first_bbox = _text_fragment_bbox(first)
    second_bbox = _text_fragment_bbox(second)
    if first_bbox is None or second_bbox is None:
        return False
    first_source = _text_fragment_source_bbox(first)
    second_source = _text_fragment_source_bbox(second)
    if first_source is None or second_source is None:
        return True
    if _bbox_iou(first_source, second_source) >= 0.015:
        return True
    if _bbox_contains_center(first_source, second_bbox, margin=28):
        return True
    if _bbox_contains_center(second_source, first_bbox, margin=28):
        return True
    source_gap_x, source_gap_y = _bbox_gaps(first_source, second_source)
    min_source_w = max(1, min(first_source[2] - first_source[0], second_source[2] - second_source[0]))
    min_source_h = max(1, min(first_source[3] - first_source[1], second_source[3] - second_source[1]))
    return source_gap_x <= max(18.0, min_source_w * 0.16) and source_gap_y <= max(28.0, min_source_h * 0.30)


def _text_fragments_have_expanded_source_context(first: dict, second: dict) -> bool:
    first_bbox = _text_fragment_bbox(first)
    second_bbox = _text_fragment_bbox(second)
    first_source = _text_fragment_source_bbox(first)
    second_source = _text_fragment_source_bbox(second)
    if first_bbox is None or second_bbox is None or first_source is None or second_source is None:
        return False

    def _expanded(source: list[int], bbox: list[int]) -> bool:
        horizontal_margin = (
            source[0] <= bbox[0] - 12
            or source[2] >= bbox[2] + 12
        )
        vertical_margin = (
            source[1] <= bbox[1] - 12
            or source[3] >= bbox[3] + 12
        )
        return horizontal_margin and vertical_margin

    if not (_expanded(first_source, first_bbox) or _expanded(second_source, second_bbox)):
        return False

    if _bbox_iou(first_source, second_source) >= 0.015:
        return True
    if _bbox_contains_center(first_source, second_bbox, margin=28):
        return True
    if _bbox_contains_center(second_source, first_bbox, margin=28):
        return True
    return True


def _text_fragments_are_stacked_same_balloon(first: dict, second: dict, region_bbox: list[int]) -> bool:
    first_bbox = _text_fragment_bbox(first)
    second_bbox = _text_fragment_bbox(second)
    if first_bbox is None or second_bbox is None:
        return False

    if first_bbox[1] > second_bbox[1]:
        first, second = second, first
        first_bbox, second_bbox = second_bbox, first_bbox

    first_w = max(1, first_bbox[2] - first_bbox[0])
    second_w = max(1, second_bbox[2] - second_bbox[0])
    first_h = max(1, first_bbox[3] - first_bbox[1])
    second_h = max(1, second_bbox[3] - second_bbox[1])
    min_w = max(1, min(first_w, second_w))
    min_h = max(1, min(first_h, second_h))
    overlap_x = max(0, min(first_bbox[2], second_bbox[2]) - max(first_bbox[0], second_bbox[0]))
    overlap_y = max(0, min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]))
    gap_y = max(0, second_bbox[1] - first_bbox[3])
    center_dx = abs(_bbox_center(first_bbox)[0] - _bbox_center(second_bbox)[0])
    region_h = max(1, int(region_bbox[3]) - int(region_bbox[1]))
    line_h = max(
        1.0,
        min(
            _text_fragment_line_height(first, first_bbox),
            _text_fragment_line_height(second, second_bbox),
        ),
    )

    vertically_close = gap_y <= max(28.0, line_h * 0.85)
    horizontally_aligned = (
        overlap_x >= min_w * 0.35
        or center_dx <= max(first_w, second_w) * 0.24
    )
    not_separate_lobes = (
        overlap_y > 0
        or gap_y <= max(28.0, line_h * 0.85)
        or abs(_bbox_center(first_bbox)[1] - _bbox_center(second_bbox)[1]) <= region_h * 0.58
    )
    return (
        vertically_close
        and horizontally_aligned
        and not_separate_lobes
        and _text_fragments_share_source_context(first, second)
    )


def _should_merge_marker_ocr_cluster(texts: list[dict], region_bbox: list[int]) -> bool:
    if len(texts) < 2:
        return False
    if not all(_text_fragment_can_merge_by_geometry(text) for text in texts):
        return False
    if not any(_text_fragment_has_white_balloon_marker(text) for text in texts):
        return False
    if _mixed_balloon_cluster_has_card_title_veto(texts, region_bbox):
        return False

    ordered = sorted(
        texts,
        key=lambda item: (
            int((_text_fragment_bbox(item) or item.get("bbox", [0, 0, 0, 0]))[1]),
            int((_text_fragment_bbox(item) or item.get("bbox", [0, 0, 0, 0]))[0]),
        ),
    )
    return all(
        _text_fragments_have_expanded_source_context(ordered[index], ordered[index + 1])
        and _text_fragments_are_stacked_same_balloon(ordered[index], ordered[index + 1], region_bbox)
        for index in range(len(ordered) - 1)
    )


_UI_SYSTEM_MESSAGE_TERMS = {
    "ALERT",
    "CANDIDATE",
    "CLEAR",
    "COLLECTION",
    "COMPLETE",
    "COMPLETED",
    "CONFIRM",
    "ERROR",
    "FAILED",
    "HOT",
    "LEVEL",
    "MISSION",
    "NEWS",
    "NOTICE",
    "PLAYER",
    "PREPARE",
    "QUEST",
    "RANK",
    "RECORD",
    "RECORDS",
    "REWARD",
    "SEARCH",
    "STATUS",
    "SUCCESS",
    "SUCCESSFUL",
    "SYSTEM",
    "TITLE",
    "TRIAL",
    "WARNING",
}


def _rgb_luma_value(rgb: object) -> float | None:
    if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
        return None
    try:
        r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    except (TypeError, ValueError):
        return None
    return (r * 0.299) + (g * 0.587) + (b * 0.114)


def _text_fragment_background_luma(text: dict) -> float | None:
    for key in ("background_rgb", "median_rgb", "dominant_rgb"):
        luma = _rgb_luma_value(text.get(key))
        if luma is not None:
            return luma
    return None


def _text_words_upper(text: str) -> list[str]:
    return [word.upper() for word in re.findall(r"[A-Za-z]+", str(text or ""))]


def _text_tokens_upper(text: str) -> list[str]:
    return [token.upper() for token in re.findall(r"[A-Za-z0-9]+", str(text or ""))]


def _looks_like_system_ui_message_text(text: str) -> bool:
    words = _text_words_upper(text)
    if not words:
        return False
    tokens = _text_tokens_upper(text)
    token_set = set(tokens)
    strong_art_label_terms = {"SIDE", "STEREO", "VINYL", "LP", "EP", "SOUL", "LOVE", "YEARS"}
    if token_set & strong_art_label_terms:
        return False
    if {"SYSTEM", "ERROR"}.issubset(set(words)):
        return True
    ui_terms = [word for word in words if word in _UI_SYSTEM_MESSAGE_TERMS]
    has_mixed_code = any(any(char.isalpha() for char in token) and any(char.isdigit() for char in token) for token in tokens)
    if ui_terms:
        if len(words) <= 5:
            return True
        if has_mixed_code or len(ui_terms) >= 2:
            return True
    if any(token in _UI_SYSTEM_MESSAGE_TERMS for token in tokens):
        return len(words) <= 5
    return False


def _looks_like_textured_art_label_text(text: str, line_polygons: list) -> bool:
    tokens = re.findall(r"[A-Za-z0-9]+", str(text or ""))
    if not (4 <= len(tokens) <= 16):
        return False
    mixed_code_tokens = [
        token
        for token in tokens
        if len(token) >= 4
        and any(char.isalpha() for char in token)
        and any(char.isdigit() for char in token)
    ]
    if len(mixed_code_tokens) < 2:
        return False
    alpha_words = [token.upper() for token in tokens if any(char.isalpha() for char in token)]
    label_terms = {"SIDE", "STEREO", "VINYL", "LP", "EP", "SOUL", "LOVE", "YEARS"}
    has_label_term = any(word in label_terms for word in alpha_words)
    dense_multiline_label = len(line_polygons or []) >= 3
    return has_label_term or dense_multiline_label


def _mixed_balloon_cluster_has_card_title_veto(texts: list[dict], region_bbox: list[int]) -> bool:
    if len(texts) < 2:
        return False
    white_items = [text for text in texts if _text_fragment_has_white_balloon_marker(text)]
    non_white_items = [text for text in texts if not _text_fragment_has_white_balloon_marker(text)]
    if not white_items or not non_white_items:
        return False
    region_h = max(1, int(region_bbox[3]) - int(region_bbox[1]))
    for white in white_items:
        white_bbox = _text_fragment_bbox(white) or _coerce_bbox(white.get("bbox"))
        white_luma = _text_fragment_background_luma(white)
        if white_bbox is None:
            continue
        for other in non_white_items:
            other_bbox = _text_fragment_bbox(other) or _coerce_bbox(other.get("bbox"))
            other_luma = _text_fragment_background_luma(other)
            if other_bbox is None:
                continue
            other_text = str(other.get("text") or other.get("original") or "").strip()
            other_words = _text_words_upper(other_text)
            if not other_words or len(other_words) > 5:
                continue
            title_like = (
                _looks_like_system_ui_message_text(other_text)
                or any(word in {"NEWS", "COLLECTION", "HOT", "SYSTEM", "ERROR", "LEVEL", "TRIAL"} for word in other_words)
            )
            if not title_like:
                continue
            background_split = (
                white_luma is not None
                and white_luma >= 220
                and other_luma is not None
                and other_luma <= 205
            )
            gap_x, gap_y = _bbox_gaps(white_bbox, other_bbox)
            stacked_separate = (
                gap_y >= 0
                and gap_y <= max(64.0, region_h * 0.06)
                and other_bbox[1] >= white_bbox[3] - 12
            )
            if background_split and stacked_separate and gap_x <= max(48.0, (white_bbox[2] - white_bbox[0]) * 0.16):
                return True
    return False


def _should_merge_ocr_cluster(texts: list[dict], region_bbox: list[int]) -> bool:
    if len(texts) < 2:
        return False
    if len(texts) >= 3:
        if _ocr_cluster_has_broad_container_with_separate_lower_fragments(texts):
            return False
        ordered = sorted(
            texts,
            key=lambda item: (
                int(item.get("bbox", [0, 0, 0, 0])[1]),
                int(item.get("bbox", [0, 0, 0, 0])[0]),
            ),
        )
        for index in range(len(ordered) - 1):
            if _ocr_cluster_merge_veto_reason([ordered[index], ordered[index + 1]], region_bbox) is not None:
                return False
        return True

    if _ocr_cluster_merge_veto_reason(texts, region_bbox) is not None:
        return False

    first, second = sorted(
        texts,
        key=lambda item: (
            int((_text_fragment_bbox(item) or item.get("bbox", [0, 0, 0, 0]))[1]),
            int((_text_fragment_bbox(item) or item.get("bbox", [0, 0, 0, 0]))[0]),
        ),
    )[:2]
    a = [int(v) for v in (_text_fragment_bbox(first) or first.get("bbox", [0, 0, 0, 0]))]
    b = [int(v) for v in (_text_fragment_bbox(second) or second.get("bbox", [0, 0, 0, 0]))]
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return False

    region_w = max(1, int(region_bbox[2]) - int(region_bbox[0]))
    region_h = max(1, int(region_bbox[3]) - int(region_bbox[1]))
    min_w = max(1, min(a[2] - a[0], b[2] - b[0]))
    min_h = max(1, min(a[3] - a[1], b[3] - b[1]))
    overlap_x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    gap_x, gap_y = _bbox_gaps(a, b)
    first_center = _bbox_center(a)
    second_center = _bbox_center(b)
    dx = abs(first_center[0] - second_center[0])
    dy = abs(first_center[1] - second_center[1])
    if gap_y >= max(18.0, min_h * 0.35) and dx >= min_w * 0.18:
        return False
    if (
        dy >= min_h * 0.55
        and dx >= min_w * 0.45
        and overlap_y <= max(10.0, min_h * 0.16)
    ):
        return False

    stacked_lines = (
        gap_y <= max(44.0, min_h * 1.8)
        and overlap_x >= min_w * 0.35
        and dy <= region_h * 0.42
    )
    touching_stacked_lines = (
        gap_y <= max(4.0, min_h * 0.08)
        and overlap_x >= min_w * 0.55
        and dy <= region_h * 0.50
    )
    same_line_fragments = (
        gap_x <= max(28.0, min_w * 0.20)
        and overlap_y >= min_h * 0.55
        and dx <= region_w * 0.42
    )
    short_fragment_pair = (
        max(
            len(str(first.get("text", "") or "").strip()),
            len(str(second.get("text", "") or "").strip()),
        ) <= 12
        and (gap_x <= 36.0 or gap_y <= 56.0)
    )

    return stacked_lines or touching_stacked_lines or same_line_fragments or short_fragment_pair


def _ocr_cluster_has_broad_container_with_separate_lower_fragments(texts: list[dict]) -> bool:
    if len(texts) < 3:
        return False

    bboxes: list[tuple[dict, list[int], int]] = []
    for text in texts:
        bbox = _coerce_bbox(text.get("bbox")) or _text_fragment_bbox(text)
        if bbox is None:
            continue
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            continue
        bboxes.append((text, [int(v) for v in bbox], int(w * h)))
    if len(bboxes) < 3:
        return False

    for container, container_bbox, container_area in bboxes:
        container_w = max(1, container_bbox[2] - container_bbox[0])
        container_h = max(1, container_bbox[3] - container_bbox[1])
        lower_children: list[tuple[dict, list[int], int]] = []
        for child, child_bbox, child_area in bboxes:
            if child is container:
                continue
            child_w = max(1, child_bbox[2] - child_bbox[0])
            child_h = max(1, child_bbox[3] - child_bbox[1])
            overlap = _bbox_overlap_area_for_geometry(container_bbox, child_bbox)
            child_overlap_ratio = overlap / float(max(1, child_area))
            child_center_y = _bbox_center(child_bbox)[1]
            lower_in_container = child_center_y >= container_bbox[1] + container_h * 0.55
            broad_container = (
                container_area >= child_area * 4.0
                and container_w >= child_w * 1.8
                and container_h >= child_h * 2.0
            )
            if broad_container and child_overlap_ratio >= 0.55 and lower_in_container:
                lower_children.append((child, child_bbox, child_area))
        if len(lower_children) < 2:
            continue

        for index, (_first_child, first_bbox, _first_area) in enumerate(lower_children):
            for _second_child, second_bbox, _second_area in lower_children[index + 1:]:
                min_h = max(1, min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]))
                gap_x, gap_y = _bbox_gaps(first_bbox, second_bbox)
                dx = abs(_bbox_center(first_bbox)[0] - _bbox_center(second_bbox)[0])
                vertically_separated = gap_y >= max(16.0, min_h * 0.35)
                diagonally_separated = gap_y >= 0 and dx >= max(48.0, min(container_w, 220) * 0.20)
                if vertically_separated or (diagonally_separated and gap_x <= container_w * 0.60):
                    return True
    return False


def _ocr_cluster_merge_veto_reason(texts: list[dict], region_bbox: list[int]) -> tuple[str, dict] | None:
    if len(texts) != 2:
        return None
    try:
        first, second = sorted(
            texts,
            key=lambda item: (
                int((_text_fragment_bbox(item) or item.get("bbox", [0, 0, 0, 0]))[1]),
                int((_text_fragment_bbox(item) or item.get("bbox", [0, 0, 0, 0]))[0]),
            ),
        )[:2]
        a = [int(v) for v in (_text_fragment_bbox(first) or first.get("bbox", [0, 0, 0, 0]))]
        b = [int(v) for v in (_text_fragment_bbox(second) or second.get("bbox", [0, 0, 0, 0]))]
    except Exception:
        return None
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return None

    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    min_area = min(area_a, area_b)
    max_area = max(area_a, area_b)
    dominant_ratio = max_area / float(min_area)
    overlap_x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap_area = overlap_x * overlap_y
    small_overlap_ratio = overlap_area / float(min_area)
    min_w = max(1, min(a[2] - a[0], b[2] - b[0]))
    min_h = max(1, min(a[3] - a[1], b[3] - b[1]))
    first_center = _bbox_center(a)
    second_center = _bbox_center(b)
    dx = abs(first_center[0] - second_center[0])
    dy = abs(first_center[1] - second_center[1])
    gap_x, gap_y = _bbox_gaps(a, b)
    details = {
        "bbox_a": a,
        "bbox_b": b,
        "dominant_area_ratio": round(dominant_ratio, 3),
        "small_overlap_ratio": round(small_overlap_ratio, 3),
        "overlap_x": int(overlap_x),
        "overlap_y": int(overlap_y),
        "gap_x": round(gap_x, 3),
        "gap_y": round(gap_y, 3),
        "dx": round(dx, 3),
        "dy": round(dy, 3),
    }

    if (
        dominant_ratio >= 6.0
        and 0.15 <= small_overlap_ratio <= 0.85
        and (dx >= min_w * 0.45 or dy >= min_h * 0.9)
    ):
        return "dominant_partial_overlap", details

    if gap_y > min_h * 2.5 and overlap_x < min_w * 0.5:
        return "text_fragment_gap_too_large", details

    return None


def _raw_region_bbox_for_texts(texts: list[dict]) -> list[int]:
    bbox = _coerce_bbox(texts[0].get("bbox")) or _text_fragment_bbox(texts[0]) or [0, 0, 0, 0]
    for text in texts[1:]:
        next_bbox = _coerce_bbox(text.get("bbox")) or _text_fragment_bbox(text)
        if next_bbox is not None:
            bbox = _bbox_union(bbox, next_bbox)
    return [int(v) for v in bbox]


def _select_merge_subset_from_region(region_texts: list[dict]) -> tuple[list[dict], list[int], bool] | None:
    if len(region_texts) < 3:
        return None
    ordered = sorted(
        region_texts,
        key=lambda item: (
            int((_text_fragment_bbox(item) or item.get("bbox", [0, 0, 0, 0]))[1]),
            int((_text_fragment_bbox(item) or item.get("bbox", [0, 0, 0, 0]))[0]),
        ),
    )
    for index in range(len(ordered) - 1):
        pair = [ordered[index], ordered[index + 1]]
        pair_bbox = _raw_region_bbox_for_texts(pair)
        if _should_merge_ocr_cluster(pair, pair_bbox):
            return pair, pair_bbox, False
    return None


def _add_unique_string(values: list[str], value) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text and text not in values:
        values.append(text)


def _merge_source_ids_for_texts(texts: list[dict]) -> tuple[list[str], list[str]]:
    source_text_ids: list[str] = []
    source_trace_ids: list[str] = []
    for text in texts:
        for value in text.get("source_text_ids") or text.get("_source_text_ids") or []:
            _add_unique_string(source_text_ids, value)
        _add_unique_string(source_text_ids, text.get("text_id") or text.get("id"))

        for value in text.get("source_trace_ids") or text.get("_source_trace_ids") or []:
            _add_unique_string(source_trace_ids, value)
        _add_unique_string(source_trace_ids, text.get("trace_id"))
    return source_text_ids, source_trace_ids


def _qa_flags_for_text(text: dict) -> set[str]:
    return {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}


def _append_qa_flag(text: dict, flag: str) -> None:
    flag = str(flag or "").strip()
    if not flag:
        return
    flags = list(text.get("qa_flags") or [])
    if flag not in flags:
        flags.append(flag)
    text["qa_flags"] = flags


def _mark_partial_ocr_review_required(text: dict, *, reason: str) -> None:
    _append_qa_flag(text, "ocr_partial_low_confidence_fragment")
    text["needs_review"] = False
    apply_route_action(text)


def _bbox_fragment_neighbor(candidate: list[int] | None, other: list[int] | None) -> bool:
    candidate_bbox = _coerce_bbox(candidate)
    other_bbox = _coerce_bbox(other)
    if candidate_bbox is None or other_bbox is None:
        return False
    c_w = max(1, candidate_bbox[2] - candidate_bbox[0])
    c_h = max(1, candidate_bbox[3] - candidate_bbox[1])
    o_w = max(1, other_bbox[2] - other_bbox[0])
    o_h = max(1, other_bbox[3] - other_bbox[1])
    vertical_gap = max(0, max(candidate_bbox[1], other_bbox[1]) - min(candidate_bbox[3], other_bbox[3]))
    if vertical_gap > max(80, int(min(c_h, o_h) * 2.5)):
        return False
    horizontal_overlap = max(0, min(candidate_bbox[2], other_bbox[2]) - max(candidate_bbox[0], other_bbox[0]))
    overlap_ratio = horizontal_overlap / max(1, min(c_w, o_w))
    c_center = (candidate_bbox[0] + candidate_bbox[2]) / 2.0
    o_center = (other_bbox[0] + other_bbox[2]) / 2.0
    center_distance = abs(c_center - o_center)
    return overlap_ratio >= 0.20 or center_distance <= max(120, max(c_w, o_w))


def _propagate_partial_ocr_review_to_neighbors(
    page_texts: list[dict],
    vision_blocks: list[dict],
    page_number: int | None,
) -> tuple[list[dict], list[dict]]:
    if len(page_texts) < 2 or len(page_texts) != len(vision_blocks):
        return page_texts, vision_blocks
    review_indices = [
        index
        for index, text in enumerate(page_texts)
        if "ocr_partial_low_confidence_fragment" in _qa_flags_for_text(text)
    ]
    if not review_indices:
        return page_texts, vision_blocks

    changed = False
    for review_index in review_indices:
        review_text = page_texts[review_index]
        review_bbox = review_text.get("text_pixel_bbox") or review_text.get("bbox")
        for index, text in enumerate(page_texts):
            if index == review_index:
                continue
            if str(text.get("route_action") or "").strip().lower() == "review_required":
                continue
            text_bbox = text.get("text_pixel_bbox") or text.get("bbox")
            if not _bbox_fragment_neighbor(review_bbox, text_bbox):
                continue
            _mark_partial_ocr_review_required(text, reason="ocr_partial_low_confidence_neighbor")
            block = vision_blocks[index]
            block["route_action"] = text.get("route_action")
            block["route_reason"] = text.get("route_reason")
            block["skip_processing"] = text.get("skip_processing")
            block["qa_flags"] = list(text.get("qa_flags") or [])
            changed = True
            record_decision(
                stage="ocr",
                action="flag_block",
                reason="ocr_partial_low_confidence_neighbor",
                page=page_number,
                layer=text.get("id") or text.get("text_id"),
                text=_ocr_text_from_entry(text),
                bbox=text.get("bbox", []),
                details={"review_fragment": _ocr_text_from_entry(review_text)},
            )
    return page_texts, vision_blocks


def _merge_ocr_clusters(
    page_texts: list[dict],
    vision_blocks: list[dict],
    image_shape: tuple[int, int, int],
    page_number: int | None,
) -> tuple[list[dict], list[dict]]:
    if len(page_texts) < 2 or len(page_texts) != len(vision_blocks):
        return page_texts, vision_blocks

    try:
        from inpainter.mask_builder import build_mask_regions
    except ImportError:
        from ..inpainter.mask_builder import build_mask_regions

    regions = build_mask_regions(page_texts, image_shape)
    index_by_identity = {id(text): index for index, text in enumerate(page_texts)}
    merged_indices: set[int] = set()
    merged_pairs: list[tuple[dict, dict]] = []

    for region in regions:
        region_texts = [text for text in region.get("texts", []) if id(text) in index_by_identity]
        if len(region_texts) < 2:
            continue
        region_bbox = [int(v) for v in region.get("bbox", [0, 0, 0, 0])]
        marker_merge = _should_merge_marker_ocr_cluster(region_texts, region_bbox)
        veto = None if marker_merge else _ocr_cluster_merge_veto_reason(region_texts, region_bbox)
        if not marker_merge and (veto is not None or not _should_merge_ocr_cluster(region_texts, region_bbox)):
            reason = veto[0] if veto is not None else "cluster_not_line_merge"
            details = {"count": len(region_texts)}
            if veto is not None:
                details.update(veto[1])
            record_decision(
                stage="ocr",
                action="keep_block_separate",
                reason=reason,
                page=page_number,
                bbox=region_bbox,
                details=details,
            )
            continue

        ordered_indices = sorted(
            {index_by_identity[id(text)] for text in region_texts},
            key=lambda idx: (
                int(page_texts[idx].get("bbox", [0, 0, 0, 0])[1]),
                int(page_texts[idx].get("bbox", [0, 0, 0, 0])[0]),
            ),
        )
        if len(ordered_indices) < 2:
            continue

        ordered_texts = [page_texts[idx] for idx in ordered_indices]
        ordered_blocks = [vision_blocks[idx] for idx in ordered_indices]
        merged_bbox = ordered_texts[0].get("bbox", [0, 0, 0, 0])
        merged_pixel_bbox = ordered_texts[0].get("text_pixel_bbox", merged_bbox)
        merged_source_bbox = (
            _coerce_bbox(ordered_texts[0].get("source_bbox"))
            or _coerce_bbox(ordered_texts[0].get("bbox"))
            or [int(v) for v in merged_bbox]
        )
        merged_line_polygons: list = []
        for item in ordered_texts:
            merged_bbox = _bbox_union(merged_bbox, item.get("bbox", merged_bbox))
            merged_pixel_bbox = _bbox_union(
                merged_pixel_bbox,
                item.get("text_pixel_bbox", item.get("bbox", merged_pixel_bbox)),
            )
            item_source_bbox = _coerce_bbox(item.get("source_bbox")) or _coerce_bbox(item.get("bbox"))
            if item_source_bbox is not None:
                merged_source_bbox = _bbox_union(merged_source_bbox, item_source_bbox)
            merged_line_polygons.extend(item.get("line_polygons") or [])

        dominant = max(
            ordered_texts,
            key=lambda item: (
                float(item.get("confidence", 0.0) or 0.0),
                int(item.get("bbox", [0, 0, 0, 0])[2] - item.get("bbox", [0, 0, 0, 0])[0]),
            ),
        )
        merged_text = dict(dominant)
        merged_confidence = max(float(item.get("confidence", 0.0) or 0.0) for item in ordered_texts)
        merged_text["text"] = semantic_refine_text(
            _compose_ocr_cluster_text(ordered_texts),
            tipo=str(merged_text.get("tipo", "fala") or "fala"),
            confidence=merged_confidence,
        )
        merged_original = _compose_ocr_cluster_field(ordered_texts, "original", "text")
        if merged_original:
            merged_text["original"] = merged_original
        merged_translated = _compose_ocr_cluster_field(ordered_texts, "translated", "traduzido")
        if merged_translated:
            merged_text["translated"] = merged_translated
            merged_text["traduzido"] = merged_translated
        merged_text["bbox"] = [int(v) for v in merged_bbox]
        merged_text["text_pixel_bbox"] = [int(v) for v in merged_pixel_bbox]
        merged_text["source_bbox"] = [int(v) for v in merged_source_bbox]
        merged_text["line_polygons"] = merged_line_polygons
        merged_text["confidence"] = merged_confidence
        source_text_ids, source_trace_ids = _merge_source_ids_for_texts(ordered_texts)
        if source_text_ids:
            merged_text["source_text_ids"] = source_text_ids
        if source_trace_ids:
            merged_text["source_trace_ids"] = source_trace_ids
            merged_text["_source_trace_ids"] = source_trace_ids
        if len(source_text_ids) > 1 or len(source_trace_ids) > 1:
            merged_text["merge_reason"] = "clustered_line_fragments"
        merged_text["qa_flags"] = sorted(
            {
                str(flag)
                for item in ordered_texts
                for flag in (item.get("qa_flags") or [])
                if str(flag).strip()
            }
        )
        if any(str(item.get("route_action") or "").strip().lower() == "review_required" for item in ordered_texts) or (
            "ocr_partial_low_confidence_fragment" in merged_text["qa_flags"]
        ):
            _mark_partial_ocr_review_required(
                merged_text,
                reason="ocr_partial_low_confidence_fragment",
            )
        merged_text["ocr_merged_source_count"] = len(ordered_texts)
        merged_source_bboxes = [
            [int(v) for v in text.get("bbox", [0, 0, 0, 0])]
            for text in ordered_texts
        ]
        merged_text["_merged_source_bboxes"] = merged_source_bboxes
        merged_text["merged_source_bboxes"] = merged_source_bboxes

        if False and is_editorial_credit(str(merged_text.get("text", "") or "")):
            merged_indices.update(ordered_indices)
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="editorial_credit_after_merge",
                page=page_number,
                text=merged_text.get("text", ""),
                bbox=merged_text["bbox"],
                details={"count": len(ordered_texts)},
            )
            continue

        merged_block = {
            "bbox": [int(v) for v in merged_bbox],
            "source_bbox": [int(v) for v in merged_source_bbox],
            "mask": _merge_local_block_masks(ordered_blocks, merged_bbox),
            "confidence": max(float(block.get("confidence", 0.0) or 0.0) for block in ordered_blocks),
            "line_polygons": merged_line_polygons,
            "text_pixel_bbox": [int(v) for v in merged_pixel_bbox],
            "balloon_type": merged_text.get("balloon_type"),
            "tipo": merged_text.get("tipo"),
            "block_profile": merged_text.get("block_profile"),
            "page_profile": merged_text.get("page_profile"),
            "text": merged_text.get("text"),
            "_merged_source_bboxes": merged_source_bboxes,
            "merged_source_bboxes": merged_source_bboxes,
            "route_action": merged_text.get("route_action"),
            "route_reason": merged_text.get("route_reason"),
            "skip_processing": merged_text.get("skip_processing"),
            "qa_flags": list(merged_text.get("qa_flags") or []),
        }
        if source_text_ids:
            merged_block["source_text_ids"] = source_text_ids
        if source_trace_ids:
            merged_block["source_trace_ids"] = source_trace_ids
            merged_block["_source_trace_ids"] = source_trace_ids
        if merged_text.get("merge_reason"):
            merged_block["merge_reason"] = merged_text.get("merge_reason")
        merged_pairs.append((merged_text, merged_block))
        merged_indices.update(ordered_indices)

        record_decision(
            stage="ocr",
            action="merge_blocks",
            reason="clustered_line_fragments",
            page=page_number,
            text=merged_text.get("text", ""),
            bbox=merged_text["bbox"],
            details={"count": len(ordered_texts)},
        )

    if not merged_pairs:
        return page_texts, vision_blocks

    final_pairs: list[tuple[dict, dict]] = []
    for index, (text, block) in enumerate(zip(page_texts, vision_blocks)):
        if index in merged_indices:
            continue
        final_pairs.append((text, block))
    final_pairs.extend(merged_pairs)
    final_pairs.sort(
        key=lambda pair: (
            int(pair[0].get("bbox", [0, 0, 0, 0])[1]),
            int(pair[0].get("bbox", [0, 0, 0, 0])[0]),
        )
    )
    return [pair[0] for pair in final_pairs], [pair[1] for pair in final_pairs]


def _bbox_area_safe(bbox: list[int] | None) -> int:
    if not bbox or len(bbox) != 4:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _bbox_inner_overlap_ratio(inner: list[int], outer: list[int]) -> float:
    ix1, iy1, ix2, iy2 = [float(v) for v in inner]
    ox1, oy1, ox2, oy2 = [float(v) for v in outer]
    inter_w = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    inter_h = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
    return (inter_w * inter_h) / inner_area


def _ocr_duplicate_similarity(first: str, second: str) -> float:
    first_norm = _normalize_text_key(first)
    second_norm = _normalize_text_key(second)
    if not first_norm or not second_norm:
        return 0.0
    if first_norm == second_norm:
        return 1.0
    if first_norm in second_norm or second_norm in first_norm:
        return 0.90
    first_tokens = set(first_norm.split())
    second_tokens = set(second_norm.split())
    overlap = len(first_tokens & second_tokens) / float(max(1, min(len(first_tokens), len(second_tokens))))
    return max(overlap, SequenceMatcher(None, first_norm, second_norm).ratio())


def _drop_contained_duplicate_ocr_texts(
    page_texts: list[dict],
    vision_blocks: list[dict],
    page_number: int | None,
) -> tuple[list[dict], list[dict]]:
    if len(page_texts) < 2 or len(page_texts) != len(vision_blocks):
        return page_texts, vision_blocks

    drop_indices: set[int] = set()
    for first_index, first_text in enumerate(page_texts):
        if first_index in drop_indices:
            continue
        first_bbox = _coerce_bbox(first_text.get("text_pixel_bbox")) or _coerce_bbox(first_text.get("bbox"))
        if first_bbox is None:
            continue
        for second_index in range(first_index + 1, len(page_texts)):
            if second_index in drop_indices:
                continue
            second_text = page_texts[second_index]
            second_bbox = _coerce_bbox(second_text.get("text_pixel_bbox")) or _coerce_bbox(second_text.get("bbox"))
            if second_bbox is None:
                continue

            first_area = _bbox_area_safe(first_bbox)
            second_area = _bbox_area_safe(second_bbox)
            if first_area <= 0 or second_area <= 0:
                continue
            similarity = _ocr_duplicate_similarity(
                str(first_text.get("text", "") or ""),
                str(second_text.get("text", "") or ""),
            )
            if (
                similarity >= 0.96
                and _bbox_iou(first_bbox, second_bbox) >= 0.72
                and len(_normalize_text_key(first_text.get("text", ""))) >= 4
            ):
                first_conf = float(first_text.get("confidence", 0.0) or 0.0)
                second_conf = float(second_text.get("confidence", 0.0) or 0.0)
                drop_index = second_index if first_conf >= second_conf else first_index
                kept_text = first_text if drop_index == second_index else second_text
                dropped_text = second_text if drop_index == second_index else first_text
                drop_indices.add(drop_index)
                record_decision(
                    stage="ocr",
                    action="drop_block",
                    reason="overlapping_duplicate_ocr_block",
                    page=page_number,
                    text=dropped_text.get("text", ""),
                    bbox=dropped_text.get("bbox", first_bbox if drop_index == first_index else second_bbox),
                    details={
                        "kept_text": kept_text.get("text", ""),
                        "kept_bbox": kept_text.get("bbox", []),
                        "similarity": round(float(similarity), 3),
                        "iou": round(float(_bbox_iou(first_bbox, second_bbox)), 3),
                    },
                )
                if drop_index == first_index:
                    break
                continue
            if first_area >= second_area:
                large_index, small_index = first_index, second_index
                large_text, small_text = first_text, second_text
                large_bbox, small_bbox = first_bbox, second_bbox
                large_area, small_area = first_area, second_area
            else:
                large_index, small_index = second_index, first_index
                large_text, small_text = second_text, first_text
                large_bbox, small_bbox = second_bbox, first_bbox
                large_area, small_area = second_area, first_area

            large_source_bbox = (
                _coerce_bbox(large_text.get("source_bbox"))
                or _coerce_bbox(large_text.get("bbox"))
                or large_bbox
            )
            large_source_area = _bbox_area_safe(large_source_bbox)
            if large_area < int(small_area * 1.75):
                if (
                    large_source_area >= int(small_area * 1.75)
                    and (
                        _bbox_contains_center(large_source_bbox, small_bbox, margin=16)
                        or _bbox_inner_overlap_ratio(small_bbox, large_source_bbox) >= 0.62
                    )
                ):
                    large_bbox = large_source_bbox
                    large_area = large_source_area
                else:
                    continue
            if not (
                _bbox_contains_center(large_bbox, small_bbox, margin=16)
                or _bbox_inner_overlap_ratio(small_bbox, large_bbox) >= 0.62
            ):
                continue

            large_key = _ocr_compact_key(str(large_text.get("text", "") or ""))
            small_key = _ocr_compact_key(str(small_text.get("text", "") or ""))
            large_conf = float(large_text.get("confidence", 0.0) or 0.0)
            small_conf = float(small_text.get("confidence", 0.0) or 0.0)
            if (
                small_key
                and large_key
                and small_key in large_key
                and large_key != small_key
                and large_conf <= small_conf + 0.2
            ):
                drop_indices.add(large_index)
                record_decision(
                    stage="ocr",
                    action="drop_block",
                    reason="overmerged_container_ocr_block",
                    page=page_number,
                    text=large_text.get("text", ""),
                    bbox=large_text.get("bbox", large_bbox),
                    details={
                        "kept_text": small_text.get("text", ""),
                        "kept_bbox": small_text.get("bbox", small_bbox),
                        "area_ratio": round(float(large_area) / float(max(1, small_area)), 2),
                        "large_confidence": round(large_conf, 3),
                        "small_confidence": round(small_conf, 3),
                    },
                )
                if drop_index := (large_index == first_index):
                    break
                continue

            if similarity < 0.72:
                continue
            if len(_normalize_text_key(small_text.get("text", ""))) < 4:
                continue

            drop_indices.add(large_index)
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="contained_duplicate_ocr_block",
                page=page_number,
                text=large_text.get("text", ""),
                bbox=large_text.get("bbox", large_bbox),
                details={
                    "kept_text": small_text.get("text", ""),
                    "kept_bbox": small_text.get("bbox", small_bbox),
                    "similarity": round(float(similarity), 3),
                    "area_ratio": round(float(large_area) / float(max(1, small_area)), 2),
                },
            )

    if not drop_indices:
        return page_texts, vision_blocks
    kept_pairs = [
        (text, block)
        for index, (text, block) in enumerate(zip(page_texts, vision_blocks))
        if index not in drop_indices
    ]
    return [pair[0] for pair in kept_pairs], [pair[1] for pair in kept_pairs]


def _ocr_text_from_entry(text: dict) -> str:
    for key in ("text", "original"):
        value = str(text.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _ocr_compact_key(text: str) -> str:
    return _normalize_text_key(text).replace(" ", "")


def _ocr_text_has_line_geometry(text: dict) -> bool:
    return bool(_normalize_line_polygons(text.get("line_polygons") or []))


def _ocr_text_bbox_for_cleanup(text: dict) -> list[int] | None:
    return (
        _coerce_bbox(text.get("text_pixel_bbox"))
        or _coerce_bbox(text.get("bbox"))
        or _coerce_bbox(text.get("source_bbox"))
    )


def _ocr_block_from_text_entry(text: dict) -> dict:
    bbox = (
        _coerce_bbox(text.get("source_bbox"))
        or _coerce_bbox(text.get("bbox"))
        or _coerce_bbox(text.get("text_pixel_bbox"))
        or [0, 0, 0, 0]
    )
    text_pixel_bbox = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(text.get("bbox")) or bbox
    source_bbox = _coerce_bbox(text.get("source_bbox")) or bbox
    return {
        "bbox": [int(v) for v in bbox],
        "source_bbox": [int(v) for v in source_bbox],
        "text_pixel_bbox": [int(v) for v in text_pixel_bbox],
        "line_polygons": list(text.get("line_polygons") or []),
        "confidence": float(text.get("confidence", 0.0) or 0.0),
        "balloon_type": text.get("balloon_type"),
        "tipo": text.get("tipo"),
        "block_profile": text.get("block_profile"),
        "page_profile": text.get("page_profile"),
        "text": _ocr_text_from_entry(text),
        "rotation_deg": text.get("rotation_deg"),
        "rotation_source": text.get("rotation_source"),
    }


def _looks_like_short_art_ocr_noise(text: dict) -> bool:
    return False


def _looks_like_final_project_text_entry(text: dict) -> bool:
    """Return true for OCR entries that already passed translation/typeset flow."""
    translated = str(text.get("translated") or text.get("traduzido") or "").strip()
    if not translated:
        return False
    has_trace = bool(
        text.get("trace_id")
        or text.get("text_instance_id")
        or text.get("source_trace_ids")
        or text.get("_source_trace_ids")
    )
    has_pipeline_anchor = bool(
        text.get("band_id")
        or text.get("render_bbox")
        or text.get("safe_text_box")
        or text.get("layout_safe_bbox")
    )
    return has_trace and has_pipeline_anchor


def _image_height_from_shape(
    image_shape: tuple[int, int, int] | tuple[int, int] | None,
) -> int | None:
    if not image_shape:
        return None
    try:
        return max(1, int(image_shape[0]))
    except Exception:
        return None


def _cover_title_overlay_scope_allows_drop(
    bbox: list[int],
    image_shape: tuple[int, int, int] | tuple[int, int] | None,
    page_number: int | None,
    total_pages: int | None = None,
) -> bool:
    try:
        page_no = int(page_number) if page_number is not None else None
    except Exception:
        page_no = None
    if page_no is None:
        return False

    try:
        total = int(total_pages) if total_pages is not None else None
    except Exception:
        total = None

    # This filter is intentionally narrow: it is for title/credit overlays at
    # chapter boundaries, not for regular narration boxes on middle pages.
    if total and total > 1 and 1 < page_no < total:
        return False
    if not total and page_no > 1:
        return False

    page_height = _image_height_from_shape(image_shape)
    y1 = int(bbox[1])
    y2 = int(bbox[3])

    if page_no <= 1:
        if page_height is None:
            return y1 <= 220
        top_limit = max(180, int(page_height * 0.18))
        bottom_limit = max(top_limit + 1, int(page_height * 0.45))
        return y1 <= top_limit and y2 <= bottom_limit

    if total and page_no >= total:
        if page_height is None:
            return False
        footer_edge = page_height - max(180, int(page_height * 0.16))
        footer_body = int(page_height * 0.55)
        return y2 >= footer_edge and y1 >= footer_body

    return False


def _looks_like_cover_title_overlay_noise(
    text: dict,
    image_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    page_number: int | None = None,
    total_pages: int | None = None,
) -> bool:
    return False


def _ocr_pre_translation_skip_policy(
    text: str,
    bbox: list[int],
    confidence: float,
    *,
    tipo: str,
    page_profile: str,
    block_profile: str,
    is_white_balloon: bool,
    image_shape: tuple[int, int, int] | tuple[int, int],
    line_polygons: list,
    run_on_suspect: bool,
    pre_semantic_run_on: bool,
    source_lang: str,
    background_rgb: list[int] | tuple[int, int, int] | None = None,
    title_rules_enabled: bool = False,
    text_matches_work_title: bool = False,
) -> dict | None:
    stripped = " ".join(str(text or "").split()).strip()
    if not stripped:
        return {
            "skip_reason": "ocr_noise_empty",
            "content_class": "noise",
            "qa_flags": ["ocr_noise_skip"],
            "needs_review": False,
        }

    normalized = _normalize_text_key(stripped)
    compact = normalized.replace(" ", "")
    words = re.findall(r"[A-Za-z]+", stripped)
    unique_words = {word.upper() for word in words}
    has_sentence_punctuation = bool(re.search(r"[?!.,:;]", stripped))
    plain_bright_background = False
    if isinstance(background_rgb, (list, tuple)) and len(background_rgb) >= 3:
        try:
            channels = [int(value) for value in background_rgb[:3]]
            plain_bright_background = min(channels) >= 235 and (max(channels) - min(channels)) <= 18
        except (TypeError, ValueError):
            plain_bright_background = False
    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    bbox_area = _bbox_area_safe(bbox)
    page_area = max(1, image_h * image_w)
    area_ratio = bbox_area / float(page_area)
    box_w = max(1, int(bbox[2]) - int(bbox[0]))
    box_h = max(1, int(bbox[3]) - int(bbox[1]))
    width_ratio = box_w / float(max(1, image_w))
    height_ratio = box_h / float(max(1, image_h))
    line_bbox = _bbox_from_line_polygons(line_polygons)
    compact_line_geometry_in_broad_region = False
    if line_bbox is not None:
        line_area = _bbox_area_safe(line_bbox)
        line_w = max(1, int(line_bbox[2]) - int(line_bbox[0]))
        line_h = max(1, int(line_bbox[3]) - int(line_bbox[1]))
        compact_line_geometry_in_broad_region = bool(
            bbox_area >= max(1, line_area * 8)
            and (line_w / float(box_w) <= 0.45 or line_h / float(box_h) <= 0.24)
            and len(words) >= 2
            and confidence >= 0.50
        )
    profile_clean = str(block_profile or "").strip().lower()
    page_profile_clean = str(page_profile or "").strip().lower()
    normalized_source_lang = str(source_lang or "").strip().lower()
    line_geometry_missing = not bool(line_polygons)
    speech_like_white_balloon = bool(
        is_white_balloon
        and has_sentence_punctuation
        and len(words) >= 4
    )
    sentence_like_plain_dialogue = bool(
        has_sentence_punctuation
        and len(words) >= 3
        and confidence >= 0.55
        and (is_white_balloon or plain_bright_background or profile_clean in {"white_balloon", "standard"})
    )
    short_punctuated_white_balloon = bool(
        is_white_balloon
        and has_sentence_punctuation
        and confidence >= 0.50
        and 1 <= len(words) <= 2
    )
    system_ui_message = bool(
        normalized_source_lang == "en"
        and confidence >= 0.72
        and _looks_like_system_ui_message_text(stripped)
    )

    cover_logo_art_rules_enabled = bool(title_rules_enabled and text_matches_work_title)

    if cover_logo_art_rules_enabled and page_profile_clean == "cover_opening":
        repeated_cover_words = len(words) >= 3 and len(unique_words) < len(words)
        alpha_chars = [char for char in stripped if char.isalpha()]
        uppercase_ratio = (
            sum(1 for char in alpha_chars if char.isupper()) / float(len(alpha_chars))
            if alpha_chars
            else 0.0
        )
        title_case_ratio = sum(1 for word in words if word[:1].isupper()) / float(max(1, len(words)))
        work_title_like_cover_text = bool(
            not system_ui_message
            and (
                repeated_cover_words
                or uppercase_ratio >= 0.58
                or title_case_ratio >= 0.72
                or profile_clean in {"cover_title_logo", "decorative_noise"}
            )
        )
        short_plain_white_balloon_text = bool(
            is_white_balloon
            and line_polygons
            and confidence >= 0.55
            and not repeated_cover_words
            and 2 <= len(words) <= 8
            and any(
                word.upper()
                in {
                    "I",
                    "YOU",
                    "WE",
                    "HE",
                    "SHE",
                    "THEY",
                    "MY",
                    "YOUR",
                    "HIS",
                    "HER",
                    "HAVE",
                    "HAD",
                    "HID",
                    "CAN",
                    "DO",
                    "ARE",
                    "IS",
                    "WAS",
                }
                for word in words
            )
        )
        long_plain_sentence_text = bool(
            len(words) >= 6
            and confidence >= 0.70
            and not repeated_cover_words
            and (is_white_balloon or plain_bright_background)
            and any(word.upper() in {"I", "YOU", "WE", "HE", "SHE", "THE", "MY", "YOUR", "HAVE", "CAN", "DO"} for word in words)
        )
        short_ornamental = (
            2 <= len(compact) <= 6
            and len(words) <= 1
            and not short_punctuated_white_balloon
        )
        logo_like_title = (
            not has_sentence_punctuation
            and not long_plain_sentence_text
            and not short_plain_white_balloon_text
            and work_title_like_cover_text
            and len(compact) >= 8
            and (
                repeated_cover_words
                or area_ratio >= 0.025
                or width_ratio >= 0.30
                or profile_clean in {"cover_title_logo", "decorative_noise"}
            )
        )
        cover_noise = (
            repeated_cover_words
            and not speech_like_white_balloon
            and not sentence_like_plain_dialogue
        ) or short_ornamental or logo_like_title
        if cover_noise:
            reason = "cover_repeated_words_noise" if repeated_cover_words else "cover_logo_or_art_ocr"
            if short_ornamental:
                reason = "cover_short_ornamental_noise"
            return {
                "skip_reason": reason,
                "content_class": "noise",
                "qa_flags": [reason],
                "needs_review": False,
            }

    logo_or_emblem = bool(
        normalized_source_lang == "en"
        and not is_white_balloon
        and not plain_bright_background
        and not system_ui_message
        and not compact_line_geometry_in_broad_region
        and not has_sentence_punctuation
        and 1 <= len(words) <= 4
        and 3 <= len(compact) <= 28
        and not any(char.isdigit() for char in compact)
        and (
            line_geometry_missing
            or confidence < 0.72
            or area_ratio >= 0.015
            or width_ratio >= 0.18
            or height_ratio >= 0.08
        )
        and not any(word.upper() in {"I", "YOU", "WE", "HE", "SHE", "THEY", "WHAT", "WHY", "HOW", "PLEASE"} for word in words)
    )
    if cover_logo_art_rules_enabled and logo_or_emblem:
        return {
            "skip_reason": "logo_or_emblem_preserved",
            "content_class": "logo",
            "qa_flags": ["logo_or_emblem_preserved"],
            "needs_review": False,
            "preserve_original": False,
        }

    textured_art_label = bool(
        normalized_source_lang == "en"
        and not is_white_balloon
        and not plain_bright_background
        and not system_ui_message
        and not re.search(r"[?!.,;]", stripped)
        and confidence >= 0.52
        and (area_ratio >= 0.008 or width_ratio >= 0.16 or height_ratio >= 0.045)
        and _looks_like_textured_art_label_text(stripped, line_polygons)
    )
    if cover_logo_art_rules_enabled and textured_art_label:
        return {
            "skip_reason": "textured_art_label_preserved",
            "content_class": "logo",
            "qa_flags": ["textured_art_label_preserved"],
            "needs_review": False,
            "preserve_original": False,
        }

    large_textured_region = area_ratio >= 0.015 or width_ratio >= 0.30 or height_ratio >= 0.10
    common_short_plain_dialogue = {
        "I",
        "ME",
        "MY",
        "YOU",
        "WE",
        "HE",
        "SHE",
        "THEY",
        "WHAT",
        "WHY",
        "HOW",
        "YES",
        "NO",
        "MOM",
        "DAD",
        "OK",
        "OKAY",
        "SURE",
        "WAIT",
        "HEY",
        "HUH",
        "HMM",
        "FOUR",
        "THREE",
        "TWO",
        "ONE",
    }
    short_low_signal_white_art_ocr = bool(
        normalized_source_lang == "en"
        and is_white_balloon
        and line_geometry_missing
        and confidence < 0.62
        and not has_sentence_punctuation
        and 1 <= len(words) <= 2
        and 2 <= len(compact) <= 8
        and compact.upper() not in common_short_plain_dialogue
        and large_textured_region
        and (not plain_bright_background or height_ratio >= 0.12 or area_ratio >= 0.02)
    )
    if cover_logo_art_rules_enabled and short_low_signal_white_art_ocr:
        return {
            "skip_reason": "suspicious_art_ocr_low_confidence",
            "content_class": "noise",
            "qa_flags": ["suspicious_art_ocr", "ocr_needs_review"],
            "needs_review": True,
            "preserve_original": False,
        }
    joined_word_art_marker = bool(re.search(r"\b(?:VHEN|IGETBACK|TOWORK|DOYOU|MAKINGUS)\b", stripped, re.IGNORECASE))
    run_on_art_suspect = bool(run_on_suspect or pre_semantic_run_on or joined_word_art_marker)
    if (
        cover_logo_art_rules_enabled
        and
        normalized_source_lang == "en"
        and joined_word_art_marker
        and not plain_bright_background
        and confidence < 0.66
        and line_geometry_missing
        and large_textured_region
    ):
        return {
            "skip_reason": "suspicious_art_ocr_low_confidence",
            "content_class": "noise",
            "qa_flags": ["suspicious_art_ocr", "ocr_needs_review"],
            "needs_review": True,
        }
    if (
        cover_logo_art_rules_enabled
        and
        normalized_source_lang == "en"
        and not is_white_balloon
        and not plain_bright_background
        and confidence < 0.66
        and line_geometry_missing
        and large_textured_region
        and (run_on_art_suspect or looks_suspicious(stripped, confidence) or confidence < 0.52)
    ):
        return {
            "skip_reason": "suspicious_art_ocr_low_confidence",
            "content_class": "noise",
            "qa_flags": ["suspicious_art_ocr", "ocr_needs_review"],
            "needs_review": True,
        }

    return None


def _is_scanlation_credit_text_entry(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    raw_text = str(text.get("text") or text.get("original") or "").strip()
    return bool(
        "scanlation_credit" in flags
        or "scanlator_credit" in flags
        or SCANLATOR_RE.search(raw_text)
        or URL_RE.search(raw_text)
        or _looks_like_hyphenated_credit_name_list(raw_text)
    )


def _looks_like_hyphenated_credit_name_list(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    letters = re.findall(r"[A-Za-z]", raw)
    if not letters or any(letter != letter.upper() for letter in letters):
        return False
    normalized = re.sub(r"\s+", " ", raw.upper())
    tokens = re.findall(r"-?[A-Z0-9][A-Z0-9_]{2,}", normalized)
    hyphenated = [token for token in tokens if token.startswith("-")]
    if len(hyphenated) < 2 or not tokens:
        return False
    hyphen_ratio = len(hyphenated) / float(max(1, len(tokens)))
    return hyphen_ratio >= 0.66 or (len(hyphenated) >= 3 and hyphen_ratio >= 0.50)


def _looks_like_scanlation_promotional_title(text: dict, image_shape: tuple[int, int, int] | tuple[int, int]) -> bool:
    if not isinstance(text, dict):
        return False
    raw_text = str(text.get("text") or "").strip()
    if not raw_text:
        return False
    if re.search(r"[?!:;]", raw_text):
        return False
    words = re.findall(r"[A-Za-z]+", raw_text)
    compact = _normalize_text_key(raw_text).replace(" ", "")
    if not (1 <= len(words) <= 7 and 3 <= len(compact) <= 64):
        return False
    bbox = _coerce_bbox(text.get("bbox")) or _coerce_bbox(text.get("text_pixel_bbox"))
    if bbox is None:
        return False
    height = max(1, int(image_shape[0]))
    width = max(1, int(image_shape[1]))
    box_w = max(1, int(bbox[2]) - int(bbox[0]))
    box_h = max(1, int(bbox[3]) - int(bbox[1]))
    return bool(
        box_h <= max(72, int(height * 0.20))
        and box_w <= max(260, int(width * 0.55))
    )


def _scanlation_title_strip_is_coherent(entries: list[dict], image_shape: tuple[int, int, int] | tuple[int, int]) -> bool:
    if len(entries) < 3:
        return False
    bboxes = [_coerce_bbox(entry.get("bbox")) or _coerce_bbox(entry.get("text_pixel_bbox")) for entry in entries]
    bboxes = [bbox for bbox in bboxes if bbox is not None]
    if len(bboxes) < 3:
        return False
    height = max(1, int(image_shape[0]))
    y_span = max(bbox[3] for bbox in bboxes) - min(bbox[1] for bbox in bboxes)
    center_span = max((bbox[1] + bbox[3]) / 2.0 for bbox in bboxes) - min((bbox[1] + bbox[3]) / 2.0 for bbox in bboxes)
    return bool(y_span <= max(96, int(height * 0.35)) and center_span <= max(48, int(height * 0.18)))


def _mark_scanlation_promotional_text(text: dict, reason: str) -> None:
    # Compatibility hook only. Scanlation/title proximity can no longer rewrite
    # normal OCR into preserve/skip routes in the automatic pipeline.
    text["skip_processing"] = False
    text["preserve_original"] = False
    text["skip_reason"] = reason
    text["content_class"] = "text"
    text["translate_policy"] = "translate"
    text["render_policy"] = "normal"
    text["route_action"] = "translate_inpaint_render"
    text["route_reason"] = reason
    text["needs_review"] = bool(text.get("needs_review", False))
    flags = list(text.get("qa_flags") or [])
    for flag in ("scanlation_credit", reason):
        if flag not in flags:
            flags.append(flag)
    text["qa_flags"] = flags


def _propagate_scanlation_credit_band_policy(
    page_texts: list[dict],
    vision_blocks: list[dict],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[list[dict], list[dict]]:
    del image_shape
    # Scanlation/credit/title proximity is no longer allowed to rewrite normal
    # text into a preserve/skip route. Keep the function as a compatibility
    # hook for older callers, but leave text and blocks unchanged.
    return page_texts, vision_blocks


def _ocr_partial_duplicate_similarity(first: str, second: str) -> float:
    first_compact = _ocr_compact_key(first)
    second_compact = _ocr_compact_key(second)
    if not first_compact or not second_compact:
        return 0.0
    if first_compact == second_compact:
        return 1.0
    if first_compact in second_compact or second_compact in first_compact:
        return 0.95
    return _ocr_duplicate_similarity(first, second)


def _is_partial_duplicate_ocr_fragment(candidate: dict, other: dict) -> bool:
    if not isinstance(candidate, dict) or not isinstance(other, dict):
        return False
    if _ocr_text_has_line_geometry(candidate):
        return False

    candidate_text = _ocr_text_from_entry(candidate)
    other_text = _ocr_text_from_entry(other)
    candidate_compact = _ocr_compact_key(candidate_text)
    other_compact = _ocr_compact_key(other_text)
    if len(candidate_compact) < 4 or len(other_compact) <= len(candidate_compact):
        return False

    candidate_bbox = _ocr_text_bbox_for_cleanup(candidate)
    other_bbox = _ocr_text_bbox_for_cleanup(other)
    if candidate_bbox is None or other_bbox is None:
        return False

    same_area = (
        _bbox_contains_center(other_bbox, candidate_bbox, margin=18)
        or _bbox_inner_overlap_ratio(candidate_bbox, other_bbox) >= 0.58
        or _bbox_iou(candidate_bbox, other_bbox) >= 0.30
    )
    if not same_area:
        return False

    similarity = _ocr_partial_duplicate_similarity(candidate_text, other_text)
    if similarity >= 0.82:
        return True

    if _ocr_text_has_line_geometry(other) and candidate_compact in other_compact:
        return True
    return False


def _filter_page_ocr_noise(
    page_texts: list[dict],
    vision_blocks: list[dict],
    page_number: int | None,
    image_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    total_pages: int | None = None,
) -> tuple[list[dict], list[dict]]:
    if len(page_texts) < 2 or len(page_texts) != len(vision_blocks):
        return page_texts, vision_blocks

    drop_indices: set[int] = set()
    for index, text in enumerate(page_texts):
        drop_reason = ""
        if _looks_like_cover_title_overlay_noise(text, image_shape, page_number, total_pages):
            drop_reason = "cover_title_overlay_noise"
        elif _looks_like_short_art_ocr_noise(text):
            drop_reason = "short_art_ocr_noise"
        if drop_reason:
            drop_indices.add(index)
            record_decision(
                stage="ocr",
                action="drop_block",
                reason=drop_reason,
                page=page_number,
                text=_ocr_text_from_entry(text),
                bbox=text.get("bbox", _ocr_text_bbox_for_cleanup(text) or []),
            )

    for candidate_index, candidate in enumerate(page_texts):
        if candidate_index in drop_indices:
            continue
        for other_index, other in enumerate(page_texts):
            if candidate_index == other_index or other_index in drop_indices:
                continue
            if _is_partial_duplicate_ocr_fragment(candidate, other):
                drop_indices.add(candidate_index)
                record_decision(
                    stage="ocr",
                    action="drop_block",
                    reason="partial_duplicate_ocr_fragment",
                    page=page_number,
                    text=_ocr_text_from_entry(candidate),
                    bbox=candidate.get("bbox", _ocr_text_bbox_for_cleanup(candidate) or []),
                    details={"kept_text": _ocr_text_from_entry(other), "kept_bbox": other.get("bbox", [])},
                )
                break

    if not drop_indices:
        return page_texts, vision_blocks

    kept_pairs = [
        (text, block)
        for index, (text, block) in enumerate(zip(page_texts, vision_blocks))
        if index not in drop_indices
    ]
    return [pair[0] for pair in kept_pairs], [pair[1] for pair in kept_pairs]


def _finalize_page_ocr_texts(
    page_texts: list[dict],
    vision_blocks: list[dict],
    image_shape: tuple[int, int, int],
    page_number: int | None,
    total_pages: int | None = None,
) -> tuple[list[dict], list[dict]]:
    texts = [dict(text) for text in page_texts if isinstance(text, dict)]
    blocks = [dict(block) for block in vision_blocks if isinstance(block, dict)]
    if not texts:
        return [], []
    if len(blocks) != len(texts):
        blocks = [_ocr_block_from_text_entry(text) for text in texts]

    texts, blocks = _filter_page_ocr_noise(texts, blocks, page_number, image_shape, total_pages)
    texts, blocks = _drop_contained_duplicate_ocr_texts(texts, blocks, page_number)
    texts, blocks = _merge_ocr_clusters(texts, blocks, image_shape, page_number)
    texts, blocks = _drop_contained_duplicate_ocr_texts(texts, blocks, page_number)
    texts, blocks = _propagate_partial_ocr_review_to_neighbors(texts, blocks, page_number)
    final_pairs = list(zip(texts, blocks))
    final_pairs.sort(
        key=lambda pair: (
            int(pair[0].get("bbox", [0, 0, 0, 0])[1]),
            int(pair[0].get("bbox", [0, 0, 0, 0])[0]),
        )
    )
    return [pair[0] for pair in final_pairs], [pair[1] for pair in final_pairs]


def _is_ambiguous_single_editorial_role_text(text: str) -> bool:
    words = [
        re.sub(r"[^A-Z]", "", token.upper())
        for token in re.findall(r"[A-Za-z][A-Za-z0-9._-]*", str(text or ""))
    ]
    words = [word for word in words if word]
    return len(words) == 1 and words[0] in {"RAW", "STAFF"}


def _drop_ambiguous_editorial_roles_on_credit_page(
    page_texts: list[dict],
    vision_blocks: list[dict],
    *,
    page_number: int,
) -> tuple[list[dict], list[dict]]:
    filtered_texts: list[dict] = []
    filtered_blocks: list[dict] = []
    for text, block in zip(page_texts, vision_blocks):
        if _is_ambiguous_single_editorial_role_text(str(text.get("text", "") or "")):
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="ambiguous_editorial_role_on_credit_page",
                page=page_number,
                text=text.get("text", ""),
                bbox=text.get("bbox", [0, 0, 0, 0]),
            )
            continue
        filtered_texts.append(text)
        filtered_blocks.append(block)
    return filtered_texts, filtered_blocks


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
    if not _white_balloon_whitening_enabled():
        return None
    masks = _extract_koharu_balloon_masks(image_rgb, text_mask)
    if masks is None:
        return None

    balloon_mask, non_text_mask = masks
    average_bg_color = _median_rgb(image_rgb, non_text_mask)
    if average_bg_color is None:
        return None

    std_rgb = _color_stddev(image_rgb, non_text_mask, average_bg_color)
    # Se houver qualquer variação cromática significante, não usamos preenchimento sólido (preserva gradientes/texturas)
    inpaint_threshold = 3.5 if _stddev3(std_rgb) > 0.5 else 5.0
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
    clip_mask = seed
    if light_on_dark:
        expand_w = max(dilate_w, min(17, dilate_w + 4))
        expand_h = max(dilate_h, min(19, dilate_h + 6))
        expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_w, expand_h))
        clip_mask = cv2.dilate(seed, expand_kernel, iterations=1)
    refined = cv2.bitwise_and(refined, clip_mask)
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


def _looks_like_cjk_dialogue_for_expanded_balloon_search(text: str, source_lang: str) -> bool:
    lang = normalize_paddleocr_language(source_lang)
    if lang not in {"ko", "korean", "ja", "japan", "zh", "ch", "chinese", "chinese_cht"}:
        return False
    stripped = " ".join((text or "").split()).strip()
    if not stripped:
        return False
    has_cjk = any(
        0x3040 <= ord(ch) <= 0x30FF
        or 0x3400 <= ord(ch) <= 0x9FFF
        or 0xAC00 <= ord(ch) <= 0xD7AF
        or 0x1100 <= ord(ch) <= 0x11FF
        or 0x3130 <= ord(ch) <= 0x318F
        for ch in stripped
    )
    if not has_cjk:
        return False
    if lang == "ko" and is_korean_sfx(stripped):
        return False
    if re.search(r"[.!?！？。…]|\.{2,}", stripped):
        return True
    compact = re.sub(r"\s+", "", stripped)
    return len(compact) >= 5


def _is_white_balloon_context_for_text(
    image_rgb: np.ndarray,
    bbox: list[int],
    text: str,
    *,
    source_lang: str,
    raw_record: dict | None = None,
    block=None,
) -> bool:
    if _is_white_balloon_region(image_rgb, bbox):
        return True

    geometry_candidates = []
    if isinstance(raw_record, dict):
        geometry_candidates.extend([raw_record.get("balloon_bbox"), raw_record.get("layout_bbox")])
    geometry_candidates.extend([getattr(block, "balloon_bbox", None), getattr(block, "layout_bbox", None)])
    for candidate_value in geometry_candidates:
        candidate = _coerce_bbox(candidate_value)
        if candidate is not None and _is_white_balloon_region(image_rgb, candidate):
            return True

    if not _looks_like_cjk_dialogue_for_expanded_balloon_search(text, source_lang):
        return False

    for pad_x_ratio, pad_y_ratio, min_pad_x, min_pad_y in (
        (0.70, 1.00, 42, 46),
        (1.00, 1.40, 60, 70),
    ):
        expanded = _expand_bbox(
            bbox,
            image_rgb.shape,
            pad_x_ratio=pad_x_ratio,
            pad_y_ratio=pad_y_ratio,
            min_pad_x=min_pad_x,
            min_pad_y=min_pad_y,
        )
        if _is_white_balloon_region(image_rgb, expanded):
            return True
    return False


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
    if not _white_balloon_whitening_enabled():
        return image_rgb.copy()
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
    if not _white_balloon_whitening_enabled():
        return image_rgb.copy()
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


def _apply_white_balloon_fill(
    image_rgb: np.ndarray,
    bbox: list[int],
    *,
    text_bbox: list[int] | None = None,
) -> np.ndarray:
    if not _white_balloon_whitening_enabled():
        return image_rgb.copy()
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
    preserve_band = ((distance > 0.0) & (distance <= 4.0)).astype(np.uint8) * 255
    fill_mask = cv2.bitwise_and(balloon_mask, cv2.bitwise_not(preserve_band))
    if text_bbox is not None:
        tx1, ty1, tx2, ty2 = [int(v) for v in text_bbox[:4]]
        height, width = result.shape[:2]
        tx1 = max(0, min(width, tx1))
        tx2 = max(0, min(width, tx2))
        ty1 = max(0, min(height, ty1))
        ty2 = max(0, min(height, ty2))
        if tx2 > tx1 and ty2 > ty1:
            text_zone = np.zeros(result.shape[:2], dtype=np.uint8)
            pad_x = max(5, int(round((tx2 - tx1) * 0.06)))
            pad_y = max(5, int(round((ty2 - ty1) * 0.12)))
            text_zone[
                max(0, ty1 - pad_y) : min(height, ty2 + pad_y),
                max(0, tx1 - pad_x) : min(width, tx2 + pad_x),
            ] = 255
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            outline_sample = gray[(balloon_mask > 0) & (text_zone == 0)]
            if outline_sample.size:
                white_level = float(np.percentile(outline_sample, 80))
                line_threshold = min(220.0, max(150.0, white_level - 28.0))
            else:
                line_threshold = 150.0
            dark_line_art = ((gray <= line_threshold) & (balloon_mask > 0) & (text_zone == 0)).astype(np.uint8) * 255
            if np.any(dark_line_art):
                line_guard = cv2.dilate(
                    dark_line_art,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
                fill_mask = cv2.bitwise_and(fill_mask, cv2.bitwise_not(line_guard))
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
        if not isinstance(text, dict):
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

    height, width = result.shape[:2]
    for text in texts:
        if not isinstance(text, dict):
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


def _apply_textured_light_text_residual_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts:
        return result

    try:
        from inpainter.mask_builder import build_inpaint_mask
        from qa.inpaint_residual import detect_residual_text
    except ImportError:
        from ..inpainter.mask_builder import build_inpaint_mask
        from ..qa.inpaint_residual import detect_residual_text

    height, width = result.shape[:2]
    max_mask_pixels = int(height * width * 0.18)
    for text in texts:
        if not isinstance(text, dict):
            continue
        mask = build_inpaint_mask(text, (height, width), image_rgb=original_rgb)
        if mask is None or not np.any(mask):
            continue
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels < 8 or mask_pixels > max_mask_pixels:
            continue

        residual = detect_residual_text(
            original_rgb,
            result,
            mask,
            include_light_residual=True,
            min_pixels=max(8, int(mask_pixels * 0.004)),
            min_ratio=0.004,
        )
        if not residual.get("has_residual") or "light_residual_pixels" not in set(residual.get("flags") or []):
            continue

        cleanup_mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        result = cv2.inpaint(result, cleanup_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

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
    min_area = max(3, min(6, int((x2 - x1) * (y2 - y1) * 0.00008)))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area or w < 2 or h < 2:
            continue
        slender = max(w, h) / float(max(1, min(w, h)))
        horizontal_text_stroke = w > h and h <= 16 and area <= max_component_area
        if slender >= 9.0 and area >= 45 and not horizontal_text_stroke:
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


def _scan_uncovered_white_text_line_blocks(
    image_rgb: np.ndarray,
    blocks: list,
    existing_bboxes: list[list[int]],
) -> list:
    """Find missed dark text lines that sit inside white speech/narration regions."""
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return []

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright = ((gray >= 225) & (value >= 225) & (saturation <= 76)).astype(np.float32)
    bright_ratio = cv2.blur(bright, (35, 35))
    local_mean = cv2.blur(gray, (21, 21))
    relative_dark = (local_mean.astype(np.int16) - gray.astype(np.int16)) >= 34
    absolute_dark = gray <= 158
    candidate = ((relative_dark | absolute_dark) & (gray <= 214) & (bright_ratio >= 0.52)).astype(np.uint8) * 255

    if existing_bboxes:
        existing_mask = np.zeros((height, width), dtype=np.uint8)
        for bbox in existing_bboxes:
            expanded = _expand_bbox(
                list(bbox),
                image_rgb.shape,
                pad_x_ratio=0.12,
                pad_y_ratio=0.40,
                min_pad_x=8,
                min_pad_y=8,
            )
            x1, y1, x2, y2 = expanded
            existing_mask[y1:y2, x1:x2] = 255
        candidate[existing_mask > 0] = 0

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    if not np.any(candidate):
        return []

    component_boxes: list[list[int]] = []
    component_area_by_box: dict[tuple[int, int, int, int], int] = {}
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > 420:
            continue
        if w < 2 or h < 3 or w > 48 or h > 38:
            continue
        if w > max(18, h * 4):
            continue
        box = [x, y, x + w, y + h]
        component_boxes.append(box)
        component_area_by_box[tuple(box)] = area

    if not component_boxes:
        return []

    median_h = int(np.median([max(1, box[3] - box[1]) for box in component_boxes]))
    rows = _cluster_component_boxes_by_rows(component_boxes, gap_y=max(5, int(median_h * 0.85)))
    added: list = []
    added_bboxes = [list(bbox) for bbox in existing_bboxes]
    for row in rows:
        if len(row) < 3:
            continue
        row_bbox = row[0]
        for box in row[1:]:
            row_bbox = _bbox_union(row_bbox, box)
        line_w = row_bbox[2] - row_bbox[0]
        line_h = row_bbox[3] - row_bbox[1]
        row_area = sum(component_area_by_box.get(tuple(box), 0) for box in row)
        if len(row) < 5 or row_area < 240:
            continue
        if line_w < 32 or line_h < 6 or line_h > 42:
            continue
        if line_w > int(width * 0.56):
            continue
        aspect = line_w / float(max(1, line_h))
        if aspect < 1.4 or aspect > 20.0:
            continue
        expanded = _expand_bbox(
            row_bbox,
            image_rgb.shape,
            pad_x_ratio=0.10,
            pad_y_ratio=0.30,
            min_pad_x=5,
            min_pad_y=5,
        )
        if any(
            _bbox_contains_center(existing, expanded, margin=14)
            or _bbox_contains_center(expanded, existing, margin=14)
            or _bbox_iou(expanded, existing) >= 0.08
            for existing in added_bboxes
        ):
            continue
        x1, y1, x2, y2 = expanded
        crop_bright = bright_ratio[y1:y2, x1:x2]
        if crop_bright.size == 0 or float(np.mean(crop_bright >= 0.52)) < 0.45:
            continue
        added.append(
            SimpleNamespace(
                xyxy=tuple(float(v) for v in expanded),
                mask=None,
                confidence=0.54,
                detector="white_text_line_orphan_scan",
                line_polygons=None,
                source_direction=None,
            )
        )
        added_bboxes.append(expanded)

    if added:
        logger.info("_scan_uncovered_white_text_line_blocks: adicionou %d linha(s) OCR", len(added))
    return added


def _apply_white_balloon_text_box_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts:
        return result

    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    def _expanded_cleanup_bbox(text: dict, fallback_bbox: list[int]) -> list[int]:
        focus = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(fallback_bbox) or fallback_bbox
        x1, y1, x2, y2 = [int(v) for v in focus]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        top_factor = 0.95
        pad_x = max(14, int(round(box_w * 0.32)))
        pad_top = max(18, int(round(box_h * top_factor)))
        pad_bottom = max(12, int(round(box_h * 0.45)))
        return [
            max(0, x1 - pad_x),
            max(0, y1 - pad_top),
            min(result.shape[1], x2 + pad_x),
            min(result.shape[0], y2 + pad_bottom),
        ]

    for text in texts:
        if not isinstance(text, dict):
            continue
        bbox = text.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4:
            continue
        balloon_bbox = _resolve_white_balloon_bbox(original_rgb, text)
        if balloon_bbox is None:
            continue
        balloon_mask = _extract_white_balloon_fill_mask(original_rgb, balloon_bbox)
        if not np.any(balloon_mask):
            legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, balloon_bbox)
            if isinstance(legacy_mask, np.ndarray):
                balloon_mask = legacy_mask
        if np.any(balloon_mask):
            distance = cv2.distanceTransform((balloon_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
            interior = (distance > 3.0).astype(np.uint8) * 255
            if not np.any(interior):
                interior = (distance > 1.5).astype(np.uint8) * 255
        else:
            interior = np.zeros(result.shape[:2], dtype=np.uint8)
        search_bbox = _expanded_cleanup_bbox(text, bbox)
        boxes = _extract_white_balloon_text_boxes(original_rgb, search_bbox)
        for bx1, by1, bx2, by2 in boxes:
            box_pad = 2
            bx1 -= box_pad
            by1 -= box_pad
            bx2 += box_pad
            by2 += box_pad
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

        focus_bbox = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(bbox)
        if focus_bbox is None:
            continue

        fx1, fy1, fx2, fy2 = focus_bbox
        focus_mask = np.zeros(result.shape[:2], dtype=np.uint8)
        focus_mask[fy1:fy2, fx1:fx2] = 255
        focus_area = int(np.count_nonzero(focus_mask))
        if np.any(balloon_mask):
            focus_mask = cv2.bitwise_and(focus_mask, balloon_mask.astype(np.uint8))
        elif np.any(interior):
            focus_mask = cv2.bitwise_and(focus_mask, interior)
        if focus_area > 0 and int(np.count_nonzero(focus_mask)) < max(12, int(focus_area * 0.28)):
            focus_region = original_gray[fy1:fy2, fx1:fx2]
            if focus_region.size:
                bright_ratio = float(np.mean(focus_region >= 210))
                p75 = float(np.percentile(focus_region, 75))
                if bright_ratio >= 0.42 or p75 >= 224.0:
                    focus_mask = np.zeros(result.shape[:2], dtype=np.uint8)
                    focus_mask[fy1:fy2, fx1:fx2] = 255
        if not np.any(focus_mask):
            continue

        balloon_pixels = original_gray[(balloon_mask > 0) if np.any(balloon_mask) else (focus_mask > 0)]
        if balloon_pixels.size == 0:
            continue
        white_level = float(np.percentile(balloon_pixels, 75))
        dark_threshold = min(220.0, white_level - 18.0)
        if dark_threshold < 165.0:
            dark_threshold = 165.0

        cleaned_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        residual_mask = (
            (cleaned_gray.astype(np.float32) <= dark_threshold)
            & (focus_mask > 0)
        ).astype(np.uint8) * 255
        if not np.any(residual_mask):
            continue

        residual_mask = cv2.dilate(
            residual_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        residual_mask = cv2.bitwise_and(residual_mask, focus_mask)
        if np.any(residual_mask):
            result[residual_mask > 0] = 255

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
            balloon_bbox = _resolve_white_balloon_bbox(original_rgb, text_items[index])
            if balloon_bbox is None:
                continue
            balloon_mask = _extract_white_balloon_fill_mask(original_rgb, balloon_bbox)
            if not np.any(balloon_mask):
                legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, balloon_bbox)
                if isinstance(legacy_mask, np.ndarray):
                    balloon_mask = legacy_mask
            if not np.any(balloon_mask):
                continue

            cluster_bbox = [int(v) for v in balloon_bbox] if cluster_bbox is None else _bbox_union(cluster_bbox, balloon_bbox)
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


def _apply_white_balloon_near_text_residual_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts:
        return result

    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    cleaned_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    height, width = cleaned_gray.shape[:2]

    for text in texts:
        if not isinstance(text, dict):
            continue
        focus = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(text.get("bbox"))
        if focus is None:
            continue
        fx1, fy1, fx2, fy2 = focus
        pad = max(8, int(round(max(fx2 - fx1, fy2 - fy1) * 0.12)))
        sx1 = max(0, fx1 - pad)
        sy1 = max(0, fy1 - pad)
        sx2 = min(width, fx2 + pad)
        sy2 = min(height, fy2 + pad)
        if sx2 <= sx1 or sy2 <= sy1:
            continue

        search = np.ones((sy2 - sy1, sx2 - sx1), dtype=np.uint8) * 255

        balloon_bbox = _resolve_white_balloon_bbox(original_rgb, text)
        if balloon_bbox is not None:
            balloon_mask = _extract_white_balloon_fill_mask(original_rgb, balloon_bbox)
            if not np.any(balloon_mask):
                legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, balloon_bbox)
                if isinstance(legacy_mask, np.ndarray):
                    balloon_mask = legacy_mask
            if np.any(balloon_mask):
                balloon_roi = (balloon_mask[sy1:sy2, sx1:sx2] > 0).astype(np.uint8)
                if np.any(balloon_roi):
                    distance = cv2.distanceTransform(balloon_roi, cv2.DIST_L2, 5)
                    interior = (distance > 2.0).astype(np.uint8) * 255
                    clip_mask = interior if np.any(interior) else (balloon_roi * 255)
                    search = cv2.bitwise_and(search, clip_mask)

        original_roi_gray = original_gray[sy1:sy2, sx1:sx2].astype(np.float32)
        cleaned_roi_gray = cleaned_gray[sy1:sy2, sx1:sx2].astype(np.float32)
        candidate = (
            (
                ((original_roi_gray < 165.0) & (cleaned_roi_gray < 185.0))
                | (cleaned_roi_gray < 145.0)
            )
            & (search > 0)
        ).astype(np.uint8) * 255
        if not np.any(candidate):
            continue

        cleanup = np.zeros_like(candidate)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
            comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area <= 0 or area > 360:
                continue
            if comp_w > 52 or comp_h > 42:
                continue
            if area <= 18 and comp_w <= 8 and comp_h <= 8:
                cleanup[labels == label] = 255
                continue
            thin_horizontal = comp_h <= max(3, int(comp_w * 0.16))
            thin_vertical = comp_w <= max(3, int(comp_h * 0.16))
            if thin_horizontal or thin_vertical:
                continue
            cleanup[labels == label] = 255
        if not np.any(cleanup):
            continue

        cleanup = cv2.dilate(
            cleanup,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        cleanup = cv2.bitwise_and(cleanup, search)
        sample_mask = cv2.dilate(search, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        sample_mask = cv2.bitwise_and(sample_mask, cv2.bitwise_not(cleanup))
        result_roi = result[sy1:sy2, sx1:sx2]
        sample = result_roi[sample_mask > 0]
        fill = (
            np.median(sample.astype(np.float32), axis=0).clip(0, 255).astype(np.uint8)
            if sample.size
            else np.array([255, 255, 255], dtype=np.uint8)
        )
        result_roi[cleanup > 0] = fill
        result[sy1:sy2, sx1:sx2] = result_roi
        cleaned_gray[sy1:sy2, sx1:sx2] = cv2.cvtColor(result_roi, cv2.COLOR_RGB2GRAY)

    return result


def _bbox_gap_pixels(a: list[int], b: list[int]) -> int:
    x_gap = max(0, max(int(a[0]) - int(b[2]), int(b[0]) - int(a[2])))
    y_gap = max(0, max(int(a[1]) - int(b[3]), int(b[1]) - int(a[3])))
    return max(x_gap, y_gap)


def _tight_reference_bbox_for_text_geometry(text: dict, width: int, height: int) -> list[int] | None:
    geometry_bbox = _bbox_from_line_polygons(text.get("line_polygons") or [])
    if geometry_bbox is None:
        geometry_bbox = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(text.get("bbox"))
    if geometry_bbox is None:
        return None

    geometry_area = _bbox_area_safe(geometry_bbox)
    if geometry_area <= 0:
        return None
    geometry_w = max(1, int(geometry_bbox[2]) - int(geometry_bbox[0]))
    geometry_h = max(1, int(geometry_bbox[3]) - int(geometry_bbox[1]))

    for key in ("source_bbox", "balloon_bbox"):
        reference = _coerce_bbox(text.get(key))
        if reference is None:
            continue
        reference_area = _bbox_area_safe(reference)
        if reference_area <= 0:
            continue
        reference_w = max(1, int(reference[2]) - int(reference[0]))
        reference_h = max(1, int(reference[3]) - int(reference[1]))
        if _bbox_gap_pixels(reference, geometry_bbox) > max(18, int(round(max(geometry_w, geometry_h) * 0.35))):
            continue
        if reference_area > max(geometry_area + 4096, int(round(geometry_area * 2.4))):
            continue
        if reference_w > max(geometry_w + 96, int(round(geometry_w * 2.1))):
            continue
        if reference_h > max(geometry_h + 96, int(round(geometry_h * 2.1))):
            continue
        x1 = max(0, min(width, int(reference[0])))
        y1 = max(0, min(height, int(reference[1])))
        x2 = max(0, min(width, int(reference[2])))
        y2 = max(0, min(height, int(reference[3])))
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]
    return None


def _restore_dark_line_art_outside_text_geometry(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb.copy()
    if result.size == 0 or not texts or original_rgb.shape[:2] != result.shape[:2]:
        return result

    height, width = result.shape[:2]
    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    cleaned_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    global_text_halo: np.ndarray | None = None
    global_guard = np.zeros((height, width), dtype=np.uint8)
    for text in texts:
        if not isinstance(text, dict):
            continue
        has_line_polygons = bool(_normalize_line_polygons(text.get("line_polygons") or []))
        guard = _build_text_geometry_guard_mask(
            text,
            height,
            width,
            include_text_bbox=not has_line_polygons,
        )
        if guard is not None and np.any(guard):
            global_guard = np.maximum(global_guard, guard)
        reference_bbox = _tight_reference_bbox_for_text_geometry(text, width, height)
        if reference_bbox is not None:
            rx1, ry1, rx2, ry2 = reference_bbox
            global_guard[ry1:ry2, rx1:rx2] = 255
    if np.any(global_guard):
        global_text_halo = cv2.dilate(
            global_guard,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33)),
            iterations=1,
        )

    for text in texts:
        if not isinstance(text, dict):
            continue
        focus = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(text.get("bbox"))
        if focus is None:
            continue
        fx1, fy1, fx2, fy2 = focus
        pad = max(24, int(round(max(fx2 - fx1, fy2 - fy1) * 0.45)))
        sx1 = max(0, fx1 - pad)
        sy1 = max(0, fy1 - pad)
        sx2 = min(width, fx2 + pad)
        sy2 = min(height, fy2 + pad)
        if sx2 <= sx1 or sy2 <= sy1:
            continue

        roi_text = dict(text)
        for key in ("bbox", "source_bbox", "balloon_bbox", "text_pixel_bbox", "layout_bbox"):
            bbox = _coerce_bbox(roi_text.get(key))
            if bbox is not None:
                roi_text[key] = [bbox[0] - sx1, bbox[1] - sy1, bbox[2] - sx1, bbox[3] - sy1]
        polygons = _normalize_line_polygons(roi_text.get("line_polygons") or [])
        if polygons:
            roi_text["line_polygons"] = [
                [[int(px) - sx1, int(py) - sy1] for px, py in polygon]
                for polygon in polygons
            ]
            polygons = roi_text["line_polygons"]

        roi_h = sy2 - sy1
        roi_w = sx2 - sx1
        guard = _build_text_geometry_guard_mask(
            roi_text,
            roi_h,
            roi_w,
            include_text_bbox=not bool(polygons),
        )
        if guard is None or not np.any(guard):
            continue
        line_band_guard = np.zeros((roi_h, roi_w), dtype=np.uint8)
        for polygon in polygons:
            if not polygon:
                continue
            xs = [int(point[0]) for point in polygon if len(point) >= 2]
            ys = [int(point[1]) for point in polygon if len(point) >= 2]
            if not xs or not ys:
                continue
            lx1 = max(0, min(xs) - 12)
            ly1 = max(0, min(ys) - 6)
            lx2 = min(roi_w, max(xs) + 12)
            ly2 = min(roi_h, max(ys) + 6)
            if lx2 > lx1 and ly2 > ly1:
                cv2.rectangle(line_band_guard, (lx1, ly1), (lx2, ly2), 255, -1)
        focus_w = max(1, fx2 - fx1)
        focus_h = max(1, fy2 - fy1)
        halo_px = max(16, min(36, int(round(max(focus_w, focus_h) * 0.12))))
        text_halo = cv2.dilate(
            guard,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_px * 2 + 1, halo_px * 2 + 1)),
            iterations=1,
        )
        if global_text_halo is not None:
            text_halo = np.maximum(text_halo, global_text_halo[sy1:sy2, sx1:sx2])
        global_guard_roi = global_guard[sy1:sy2, sx1:sx2]
        candidate = (
            (original_gray[sy1:sy2, sx1:sx2].astype(np.float32) < 245.0)
            & (guard == 0)
            & (global_guard_roi == 0)
        ).astype(np.uint8) * 255
        if not np.any(candidate):
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        restore_mask = np.zeros_like(candidate)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
            comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < 4:
                continue
            component = labels == label
            outside_halo = bool(np.any(component & (text_halo == 0)))
            overlaps_text_area = bool(np.any(component & (text_halo > 0)))
            thin_horizontal = comp_h <= max(4, int(comp_w * 0.14))
            thin_vertical = comp_w <= max(4, int(comp_h * 0.14))
            component_box_area = max(1, comp_w * comp_h)
            elongated_sparse = (
                max(comp_w, comp_h) >= 28
                and (
                    min(comp_w, comp_h) <= max(8, int(max(comp_w, comp_h) * 0.30))
                    or area <= int(component_box_area * 0.48)
                )
            )
            large_line_art = area > 1200 or comp_w > 140 or comp_h > 70
            line_art_like = thin_horizontal or thin_vertical or elongated_sparse or large_line_art
            near_text_line_band = bool(np.any(component & (line_band_guard > 0)))
            near_text_line_band_ratio = (
                float(np.count_nonzero(component & (line_band_guard > 0))) / float(max(1, area))
                if near_text_line_band
                else 0.0
            )
            if near_text_line_band_ratio >= 0.30 and not large_line_art:
                continue
            if overlaps_text_area:
                if line_art_like:
                    restore_mask[component] = 255
                elif outside_halo and large_line_art:
                    restore_mask[component & (text_halo == 0)] = 255
                continue
            if outside_halo or line_art_like:
                restore_mask[component] = 255
        if np.any(restore_mask):
            result_roi = result[sy1:sy2, sx1:sx2]
            result_roi[restore_mask > 0] = original_rgb[sy1:sy2, sx1:sx2][restore_mask > 0]
            result[sy1:sy2, sx1:sx2] = result_roi
            cleaned_gray[sy1:sy2, sx1:sx2] = cv2.cvtColor(result_roi, cv2.COLOR_RGB2GRAY)
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


def _apply_cjk_mask_residual_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    if mask is None or not isinstance(mask, np.ndarray) or not np.any(mask):
        return cleaned_rgb
    if original_rgb.shape[:2] != cleaned_rgb.shape[:2] or mask.shape[:2] != cleaned_rgb.shape[:2]:
        return cleaned_rgb
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    local_gray = cv2.blur(original_gray, (31, 31))
    hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    warm_or_purple = (hue <= 24) | (hue >= 145) | ((hue >= 25) & (hue <= 45) & (saturation >= 105))
    original_textlike = (
        ((original_gray <= 96) & (local_gray >= 132))
        | ((saturation >= 70) & (value >= 42) & (value <= 252) & warm_or_purple)
        | ((original_gray >= 218) & (local_gray <= 190))
    )

    diff = np.mean(cv2.absdiff(original_rgb, cleaned_rgb), axis=2)
    unchanged_text = ((mask > 0) & original_textlike & (diff <= 18.0)).astype(np.uint8) * 255
    if not np.any(unchanged_text):
        return cleaned_rgb

    count, labels, stats, _ = cv2.connectedComponentsWithStats(unchanged_text, connectivity=8)
    residual = np.zeros_like(unchanged_text, dtype=np.uint8)
    image_area = max(1, int(mask.shape[0]) * int(mask.shape[1]))
    mask_area = max(1, int(np.count_nonzero(mask)))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3 or area > max(6000, int(image_area * 0.008)):
            continue
        w_box = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_area = max(1, w_box * h_box)
        fill_ratio = area / float(bbox_area)
        aspect_ratio = max(w_box, h_box) / float(max(1, min(w_box, h_box)))
        if fill_ratio > 0.94 and area > 512:
            continue
        if aspect_ratio > 28.0 and min(w_box, h_box) <= 10:
            continue
        residual[labels == label] = 255

    if not np.any(residual) or int(np.count_nonzero(residual)) > int(mask_area * 0.35):
        return cleaned_rgb

    residual = cv2.dilate(
        residual,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    residual = cv2.bitwise_and(residual, (mask > 0).astype(np.uint8) * 255)
    if not np.any(residual):
        return cleaned_rgb
    return cv2.inpaint(cleaned_rgb, residual, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


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
    final, _stats = _apply_post_inpaint_cleanup_timed(original_rgb, cleaned_rgb, texts)
    return final


def _fill_dark_text_pixels_from_bright_context(
    image_rgb: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    if image_rgb.size == 0 or not isinstance(target_mask, np.ndarray):
        return image_rgb
    if target_mask.shape[:2] != image_rgb.shape[:2]:
        return image_rgb

    mask = (target_mask > 0).astype(np.uint8)
    if not np.any(mask):
        return image_rgb

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    masked = mask > 0
    masked_gray = gray[masked]
    if masked_gray.size == 0:
        return image_rgb

    dark_residual = ((gray < 236) | (saturation > 64)) & masked
    if int(np.count_nonzero(dark_residual)) < 6:
        return image_rgb

    clean_mask = cv2.dilate(
        dark_residual.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    clean_mask = ((clean_mask > 0) & masked).astype(np.uint8) * 255

    outer = cv2.dilate(
        mask * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    )
    inner = cv2.dilate(
        mask * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    ring = (outer > 0) & (inner == 0)
    sample_mask = ring & (gray >= 214) & (saturation <= 88)
    if int(np.count_nonzero(sample_mask)) < 24:
        sample_mask = ring & (gray >= 202)
    if int(np.count_nonzero(sample_mask)) < 12:
        return image_rgb

    fill_rgb = np.median(image_rgb[sample_mask], axis=0)
    if float(np.mean(fill_rgb)) < 202.0:
        return image_rgb

    result = image_rgb.copy()
    result[clean_mask > 0] = np.clip(fill_rgb, 0, 255).astype(np.uint8)
    return result


def _build_glyph_residual_cleanup_mask(
    original_rgb: np.ndarray,
    text: dict,
    shape: tuple[int, int],
    *,
    balloon_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    if original_rgb.size == 0 or not isinstance(text, dict):
        return None
    try:
        try:
            from inpainter.mask_builder import build_raw_text_mask_from_image, expand_text_mask
        except ImportError:
            from ..inpainter.mask_builder import build_raw_text_mask_from_image, expand_text_mask
    except Exception:
        return None

    try:
        raw_mask = build_raw_text_mask_from_image(text, original_rgb, original_rgb.shape)
    except Exception:
        raw_mask = None
    if raw_mask is None or not np.any(raw_mask):
        return None
    line_guard = _build_text_geometry_guard_mask(
        text,
        shape[0],
        shape[1],
        include_text_bbox=False,
    )
    if line_guard is not None and np.any(line_guard):
        raw_mask = cv2.bitwise_and(raw_mask.astype(np.uint8), line_guard)
        if not np.any(raw_mask):
            return None
    focus_bbox = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(text.get("bbox"))
    if focus_bbox is not None:
        fx1, fy1, fx2, fy2 = focus_bbox
        focus_area = max(1, (fx2 - fx1) * (fy2 - fy1))
        raw_area = int(np.count_nonzero(raw_mask))
        has_line_geometry = bool(_normalize_line_polygons(text.get("line_polygons") or []))
        if not has_line_geometry and raw_area > int(focus_area * 0.35):
            return None
    glyph_mask = expand_text_mask(raw_mask.astype(np.uint8), expand_px=5)
    if glyph_mask is None or not np.any(glyph_mask):
        return None
    if glyph_mask.shape[:2] != shape:
        return None

    if isinstance(balloon_mask, np.ndarray) and balloon_mask.shape[:2] == shape and np.any(balloon_mask):
        distance = cv2.distanceTransform((balloon_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        interior = (distance > 1.5).astype(np.uint8) * 255
        clip_mask = interior if np.any(interior) else balloon_mask.astype(np.uint8)
        clipped = cv2.bitwise_and(glyph_mask.astype(np.uint8), clip_mask)
        if np.any(clipped):
            glyph_mask = clipped

    return glyph_mask.astype(np.uint8)


def _resolve_glyph_cleanup_clip_mask(
    original_rgb: np.ndarray,
    text: dict,
    shape: tuple[int, int],
) -> np.ndarray | None:
    best_mask: np.ndarray | None = None
    best_area = 0
    for key in ("source_bbox", "balloon_bbox", "bbox", "text_pixel_bbox"):
        bbox = _coerce_bbox(text.get(key))
        if bbox is None:
            continue
        fill_mask = _extract_white_balloon_fill_mask(original_rgb, bbox)
        if not np.any(fill_mask):
            legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, bbox)
            if isinstance(legacy_mask, np.ndarray):
                fill_mask = legacy_mask
        if not isinstance(fill_mask, np.ndarray) or fill_mask.shape[:2] != shape or not np.any(fill_mask):
            continue
        area = int(np.count_nonzero(fill_mask))
        if area > best_area:
            best_mask = fill_mask.astype(np.uint8)
            best_area = area
    return best_mask


def _apply_glyph_residual_cleanup_for_texts(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    result = cleaned_rgb
    if result.size == 0 or not texts:
        return result
    shape = result.shape[:2]
    for text in texts:
        if not isinstance(text, dict):
            continue
        if not _text_is_white_cleanup_safe(original_rgb, text):
            continue
        clip_mask = _resolve_glyph_cleanup_clip_mask(original_rgb, text, shape)
        glyph_mask = _build_glyph_residual_cleanup_mask(
            original_rgb,
            text,
            shape,
            balloon_mask=clip_mask,
        )
        if glyph_mask is None or not np.any(glyph_mask):
            continue
        result = _fill_dark_text_pixels_from_bright_context(result, glyph_mask)
    return result


def _glyph_cleanup_limit_mask(
    original_rgb: np.ndarray,
    limit_mask: np.ndarray | None,
    texts: list[dict],
    shape: tuple[int, int],
) -> np.ndarray | None:
    if not isinstance(limit_mask, np.ndarray) or limit_mask.shape[:2] != shape:
        return limit_mask
    allowed = (limit_mask > 0).astype(np.uint8) * 255
    if original_rgb.size == 0 or not texts:
        return allowed
    for text in texts:
        if not isinstance(text, dict):
            continue
        if not _text_is_white_cleanup_safe(original_rgb, text):
            continue
        clip_mask = _resolve_glyph_cleanup_clip_mask(original_rgb, text, shape)
        glyph_mask = _build_glyph_residual_cleanup_mask(
            original_rgb,
            text,
            shape,
            balloon_mask=clip_mask,
        )
        if glyph_mask is not None and np.any(glyph_mask):
            allowed = np.maximum(allowed, glyph_mask.astype(np.uint8))
    return allowed


def _apply_geometry_white_balloon_cleanup(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    if cleaned_rgb.size == 0 or not texts:
        return cleaned_rgb
    try:
        try:
            from inpainter.fill_normalization import normalize_white_balloon_fill
            from inpainter.mask_builder import balloon_mask_from_block, build_inpaint_mask
            from inpainter.residual_cleanup import cleanup_white_balloon_residuals
        except ImportError:
            from ..inpainter.fill_normalization import normalize_white_balloon_fill
            from ..inpainter.mask_builder import balloon_mask_from_block, build_inpaint_mask
            from ..inpainter.residual_cleanup import cleanup_white_balloon_residuals
    except Exception:
        return cleaned_rgb

    result = cleaned_rgb
    shape = result.shape[:2]

    def _candidate_bboxes(text: dict) -> list[list[int]]:
        candidates: list[list[int]] = []
        for key in ("balloon_bbox", "bbox", "text_pixel_bbox", "source_bbox"):
            bbox = _coerce_bbox(text.get(key))
            if bbox is not None and bbox not in candidates:
                candidates.append(bbox)
                expanded = _expand_bbox(
                    bbox,
                    original_rgb.shape,
                    pad_x_ratio=0.30,
                    pad_y_ratio=0.80,
                    min_pad_x=24,
                    min_pad_y=28,
                )
                if expanded not in candidates:
                    candidates.append(expanded)
        return candidates

    def _explicit_white_context(text: dict) -> bool:
        if _text_has_nonwhite_cleanup_marker(text):
            return False
        if _text_has_white_cleanup_marker(text):
            return True
        return False

    def _bbox_has_bright_context(bbox: list[int]) -> bool:
        height, width = original_rgb.shape[:2]
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
        crop = original_rgb[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return float(np.percentile(gray, 75)) >= 236.0 and float(np.mean(gray >= 220)) >= 0.55

    def _best_white_fill_mask(text: dict, candidates: list[list[int]], *, white_context: bool) -> np.ndarray | None:
        if not white_context:
            return None

        best_mask: np.ndarray | None = None
        best_area = 0
        for bbox in candidates:
            fill_mask = _extract_white_balloon_fill_mask(original_rgb, bbox)
            if not np.any(fill_mask):
                legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, bbox)
                if isinstance(legacy_mask, np.ndarray):
                    fill_mask = legacy_mask
            area = int(np.count_nonzero(fill_mask))
            if area > best_area:
                best_mask = fill_mask
                best_area = area
        return best_mask if best_mask is not None and best_area > 0 else None

    def _text_geometry_protected_line_mask(text: dict) -> np.ndarray | None:
        guard = _build_text_geometry_guard_mask(
            text,
            shape[0],
            shape[1],
            include_text_bbox=False,
        )
        if guard is None or not np.any(guard):
            guard = _build_text_geometry_guard_mask(text, shape[0], shape[1])
        if guard is None or not np.any(guard):
            return None
        gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
        dark_outside_text = ((gray < 150) & (guard == 0)).astype(np.uint8) * 255
        if not np.any(dark_outside_text):
            return None
        return cv2.dilate(
            dark_outside_text,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

    for text in texts:
        if not isinstance(text, dict):
            continue
        if not _text_is_white_cleanup_safe(original_rgb, text):
            continue
        candidates = _candidate_bboxes(text)
        explicit_context = _explicit_white_context(text)
        white_context = (
            explicit_context
            or _text_anchor_has_white_cleanup_context(original_rgb, text)
            or any(_bbox_has_bright_context(bbox) for bbox in candidates)
        )
        if not white_context:
            continue

        text_mask = build_inpaint_mask(text, shape, image_rgb=original_rgb)
        if text_mask is not None and np.any(text_mask):
            result = normalize_white_balloon_fill(result, text_mask, text)
            result = _fill_dark_text_pixels_from_bright_context(result, text_mask)

        balloon_mask = balloon_mask_from_block(text, shape)
        fill_mask = _best_white_fill_mask(text, candidates, white_context=white_context)
        if fill_mask is not None and np.any(fill_mask):
            if balloon_mask is None or not np.any(balloon_mask):
                balloon_mask = fill_mask
            else:
                block_area = int(np.count_nonzero(balloon_mask))
                fill_area = int(np.count_nonzero(fill_mask))
                if fill_area > int(block_area * 1.15):
                    balloon_mask = fill_mask
        if balloon_mask is None or not np.any(balloon_mask):
            balloon_bbox = _coerce_bbox(text.get("balloon_bbox")) or _resolve_white_balloon_bbox(original_rgb, text)
            if balloon_bbox is not None:
                balloon_mask = _extract_white_balloon_fill_mask(original_rgb, balloon_bbox)
                if not np.any(balloon_mask):
                    legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, balloon_bbox)
                    if isinstance(legacy_mask, np.ndarray):
                        balloon_mask = legacy_mask
        if balloon_mask is not None and np.any(balloon_mask):
            glyph_mask = _build_glyph_residual_cleanup_mask(
                original_rgb,
                text,
                shape,
                balloon_mask=balloon_mask,
            )
            if glyph_mask is not None and np.any(glyph_mask):
                result = _fill_dark_text_pixels_from_bright_context(result, glyph_mask)
            line_cleanup_mask = _build_text_geometry_guard_mask(
                text,
                shape[0],
                shape[1],
                include_text_bbox=False,
            )
            if line_cleanup_mask is not None and np.any(line_cleanup_mask):
                if line_cleanup_mask.shape[:2] == balloon_mask.shape[:2]:
                    line_cleanup_mask = cv2.bitwise_and(
                        line_cleanup_mask.astype(np.uint8),
                        (balloon_mask > 0).astype(np.uint8) * 255,
                    )
                if np.any(line_cleanup_mask):
                    result = _fill_dark_text_pixels_from_bright_context(result, line_cleanup_mask)
            protected_mask = _text_geometry_protected_line_mask(text)
            if (
                protected_mask is not None
                and protected_mask.shape[:2] == balloon_mask.shape[:2]
                and not text.get("line_polygons")
            ):
                balloon_interior = cv2.erode(
                    (balloon_mask > 0).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                    iterations=1,
                )
                protected_mask = cv2.bitwise_and(protected_mask, cv2.bitwise_not(balloon_interior))
            result = cleanup_white_balloon_residuals(
                result,
                balloon_mask,
                protected_mask=protected_mask,
            )
    return result


def _apply_post_inpaint_cleanup_timed(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
    *,
    selective: bool | None = None,
    limit_mask: np.ndarray | None = None,
    include_text_bboxes_in_limit: bool = True,
) -> tuple[np.ndarray, dict]:
    if selective is None:
        selective = _cleanup_selective_enabled()

    has_white, has_textured = _text_cleanup_kinds(texts)
    white_texts = _white_cleanup_texts(original_rgb, texts)
    has_white_cleanup = bool(white_texts)
    if has_white and has_textured:
        cleanup_reason = "mixed"
    elif has_textured:
        cleanup_reason = "textured_only"
    elif has_white:
        cleanup_reason = "white_only"
    elif selective:
        cleanup_reason = "micro_only"
    else:
        cleanup_reason = "full"

    stats = {
        "_t_cleanup_seam_ms": 0.0,
        "_t_cleanup_band_artifact_ms": 0.0,
        "_t_cleanup_textured_light_residual_ms": 0.0,
        "_t_cleanup_white_line_ms": 0.0,
        "_t_cleanup_white_box_ms": 0.0,
        "_t_cleanup_geometry_white_ms": 0.0,
        "_t_cleanup_near_text_residual_ms": 0.0,
        "_t_cleanup_micro_ms": 0.0,
        "cleanup_skipped_seam": False,
        "cleanup_skipped_band_artifact": False,
        "cleanup_skipped_textured_light_residual": False,
        "cleanup_skipped_white_line": False,
        "cleanup_skipped_white_box": False,
        "cleanup_skipped_geometry_white": False,
        "cleanup_skipped_near_text_residual": False,
        "cleanup_reason": cleanup_reason,
        "cleanup_limit_mask_pixels": 0,
        "cleanup_changed_outside_limit_mask": 0,
    }
    total_start = time.perf_counter()
    final = cleaned_rgb

    def _run_step(key: str, callback):
        started = time.perf_counter()
        result = callback()
        stats[key] = round((time.perf_counter() - started) * 1000.0, 3)
        return result

    if (not selective) or has_textured:
        final = _run_step(
            "_t_cleanup_seam_ms",
            lambda: _apply_textured_balloon_seam_cleanup(original_rgb, final, texts),
        )
        final = _run_step(
            "_t_cleanup_band_artifact_ms",
            lambda: _apply_textured_balloon_band_artifact_cleanup(original_rgb, final, texts),
        )
        final = _run_step(
            "_t_cleanup_textured_light_residual_ms",
            lambda: _apply_textured_light_text_residual_cleanup(original_rgb, final, texts),
        )
    else:
        stats["cleanup_skipped_seam"] = True
        stats["cleanup_skipped_band_artifact"] = True
        stats["cleanup_skipped_textured_light_residual"] = True

    if ((not selective) or has_white) and has_white_cleanup:
        final = _run_step(
            "_t_cleanup_white_line_ms",
            lambda: _apply_white_balloon_line_artifact_cleanup(original_rgb, final, white_texts),
        )
        if _white_balloon_text_box_cleanup_enabled():
            final = _run_step(
                "_t_cleanup_white_box_ms",
                lambda: _apply_white_balloon_text_box_cleanup(original_rgb, final, white_texts),
            )
        else:
            stats["cleanup_skipped_white_box"] = True
        final = _run_step(
            "_t_cleanup_geometry_white_ms",
            lambda: _apply_geometry_white_balloon_cleanup(original_rgb, final, white_texts),
        )
        final = _restore_dark_line_art_outside_text_geometry(original_rgb, final, white_texts)
        final = _run_step(
            "_t_cleanup_near_text_residual_ms",
            lambda: _apply_white_balloon_near_text_residual_cleanup(original_rgb, final, white_texts),
        )
    else:
        stats["cleanup_skipped_white_line"] = True
        stats["cleanup_skipped_white_box"] = True
        stats["cleanup_skipped_geometry_white"] = True
        stats["cleanup_skipped_near_text_residual"] = True

    micro_texts = white_texts if has_white_cleanup else (texts if selective else [])
    if micro_texts:
        final = _run_step(
            "_t_cleanup_micro_ms",
            lambda: _apply_white_balloon_micro_artifact_cleanup(original_rgb, final, micro_texts),
        )
        if has_white_cleanup:
            final = _restore_dark_line_art_outside_text_geometry(original_rgb, final, white_texts)
    final, limit_pixels, changed_outside = _clamp_image_to_limit_mask(
        cleaned_rgb,
        final,
        limit_mask,
        texts,
        include_text_bboxes=include_text_bboxes_in_limit,
    )
    stats["cleanup_limit_mask_pixels"] = limit_pixels
    stats["cleanup_changed_outside_limit_mask"] = changed_outside
    if has_white_cleanup:
        final = _restore_dark_line_art_outside_text_geometry(original_rgb, final, white_texts)
        final = _apply_glyph_residual_cleanup_for_texts(original_rgb, final, white_texts)
        glyph_limit_mask = _glyph_cleanup_limit_mask(
            original_rgb,
            limit_mask,
            white_texts,
            final.shape[:2],
        )
        final, final_limit_pixels, final_changed_outside = _clamp_image_to_limit_mask(
            cleaned_rgb,
            final,
            glyph_limit_mask,
            texts,
            include_text_bboxes=include_text_bboxes_in_limit,
        )
        stats["cleanup_limit_mask_pixels"] = max(stats["cleanup_limit_mask_pixels"], final_limit_pixels)
        stats["cleanup_changed_outside_limit_mask"] += final_changed_outside
        final = _restore_dark_line_art_outside_text_geometry(original_rgb, final, white_texts)
    stats["_t_cleanup_total_ms"] = round((time.perf_counter() - total_start) * 1000.0, 3)
    return final, stats


def _has_white_balloon_text_residual(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> bool:
    if cleaned_rgb.size == 0 or not texts:
        return False

    cleaned_gray = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2GRAY)
    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    height, width = cleaned_gray.shape[:2]

    def _normalize_focus_bbox(candidate) -> list[int] | None:
        if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
            return None
        x1, y1, x2, y2 = [int(v) for v in candidate]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _expanded_focus_bbox(text: dict, fallback_bbox: list[int]) -> list[int] | None:
        focus = _normalize_focus_bbox(text.get("text_pixel_bbox")) or _normalize_focus_bbox(fallback_bbox)
        if focus is None:
            return None
        x1, y1, x2, y2 = focus
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        profile = str(text.get("block_profile") or text.get("layout_profile") or "").strip().lower()
        top_factor = 0.95
        pad_x = max(14, int(round(box_w * 0.32)))
        pad_top = max(18, int(round(box_h * top_factor)))
        pad_bottom = max(12, int(round(box_h * 0.45)))
        return [
            max(0, x1 - pad_x),
            max(0, y1 - pad_top),
            min(width, x2 + pad_x),
            min(height, y2 + pad_bottom),
        ]

    for text in texts:
        if not isinstance(text, dict):
            continue

        bbox = _normalize_focus_bbox(text.get("bbox"))
        if bbox is None:
            continue

        resolved_balloon_bbox = _resolve_white_balloon_bbox(original_rgb, text)
        balloon_bbox = _normalize_focus_bbox(resolved_balloon_bbox) or bbox
        focus_candidates = [
            _normalize_focus_bbox(text.get("text_pixel_bbox")),
            bbox,
        ]

        if resolved_balloon_bbox is None:
            continue

        balloon_mask = _extract_white_balloon_fill_mask(original_rgb, balloon_bbox)
        if not np.any(balloon_mask):
            legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, balloon_bbox)
            if isinstance(legacy_mask, np.ndarray):
                balloon_mask = legacy_mask
        if not np.any(balloon_mask):
            continue

        distance = cv2.distanceTransform((balloon_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        interior = (distance > 3.0).astype(np.uint8) * 255
        if not np.any(interior):
            interior = (distance > 1.5).astype(np.uint8) * 255
        if not np.any(interior):
            continue

        target_mask = None
        target_area = 0
        fallback_focus_bbox = None
        expanded_candidate = _expanded_focus_bbox(text, bbox)
        if expanded_candidate is not None:
            focus_candidates.append(expanded_candidate)

        for focus_bbox in focus_candidates:
            if focus_bbox is None:
                continue
            if fallback_focus_bbox is None:
                fallback_focus_bbox = focus_bbox
            x1, y1, x2, y2 = focus_bbox
            candidate_mask = np.zeros((height, width), dtype=np.uint8)
            candidate_mask[y1:y2, x1:x2] = 255
            candidate_mask = cv2.bitwise_and(candidate_mask, interior)
            candidate_area = int(np.count_nonzero(candidate_mask))
            if candidate_area >= 12 and candidate_area > target_area:
                target_mask = candidate_mask
                target_area = candidate_area

        if fallback_focus_bbox is not None:
            x1, y1, x2, y2 = fallback_focus_bbox
            focus_area = max(1, (x2 - x1) * (y2 - y1))
            if target_mask is None or target_area < int(focus_area * 0.28):
                focus_region = original_gray[y1:y2, x1:x2]
                if focus_region.size:
                    bright_ratio = float(np.mean(focus_region >= 210))
                    p75 = float(np.percentile(focus_region, 75))
                    if bright_ratio >= 0.42 or p75 >= 224.0:
                        fallback_mask = np.zeros((height, width), dtype=np.uint8)
                        fallback_mask[y1:y2, x1:x2] = 255
                        target_mask = fallback_mask
                        target_area = int(np.count_nonzero(fallback_mask))

        if target_mask is None or target_area < 12:
            continue

        balloon_pixels = original_gray[interior > 0]
        if balloon_pixels.size == 0:
            continue

        white_level = float(np.percentile(balloon_pixels, 75))
        dark_threshold = min(220.0, white_level - 18.0)
        if dark_threshold < 170.0:
            dark_threshold = 170.0

        residual_pixels = (
            (cleaned_gray.astype(np.float32) <= dark_threshold)
            & (target_mask > 0)
        )
        residual_count = int(np.count_nonzero(residual_pixels))
        if residual_count >= max(18, int(target_area * 0.004)):
            return True

    return False


def _build_white_balloon_residual_line_box_mask(
    original_rgb: np.ndarray,
    text: dict,
    interior: np.ndarray,
) -> np.ndarray | None:
    height, width = original_rgb.shape[:2]
    focus = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(text.get("bbox"))
    if focus is None:
        return None
    focus = _expand_bbox(
        focus,
        original_rgb.shape,
        pad_x_ratio=0.03,
        pad_y_ratio=0.08,
        min_pad_x=3,
        min_pad_y=3,
    )
    fx1, fy1, fx2, fy2 = [int(v) for v in focus]
    if fx2 <= fx1 or fy2 <= fy1:
        return None

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    local_mean = cv2.blur(gray, (21, 21))
    search = np.zeros((height, width), dtype=np.uint8)
    search[fy1:fy2, fx1:fx2] = 255
    search = cv2.bitwise_and(search, interior.astype(np.uint8))
    if not np.any(search):
        return None

    candidate = (
        (((local_mean.astype(np.int16) - gray.astype(np.int16)) >= 18) | (gray <= 190))
        & (search > 0)
    ).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    if not np.any(candidate):
        return None

    component_boxes: list[list[int]] = []
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    focus_w = max(1, fx2 - fx1)
    focus_h = max(1, fy2 - fy1)
    max_component_area = max(1800, int(focus_w * focus_h * 0.08))
    max_component_h = max(64, int(focus_h * 0.70))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 4 or comp_w < 2 or comp_h < 2:
            continue
        if area > max_component_area or comp_h > max_component_h:
            continue
        slender = max(comp_w, comp_h) / float(max(1, min(comp_w, comp_h)))
        if slender >= 9.0 and area >= 45:
            continue
        component_boxes.append([x, y, x + comp_w, y + comp_h])

    if not component_boxes:
        return None

    median_h = int(np.median([max(1, box[3] - box[1]) for box in component_boxes]))
    row_gap_y = max(4, int(median_h * 0.90))
    line_gap_x = max(8, int(median_h * 1.10))
    rows = _cluster_component_boxes_by_rows(component_boxes, gap_y=row_gap_y)
    line_mask = np.zeros((height, width), dtype=np.uint8)
    pad_x = max(3, int(round(median_h * 0.20)))
    pad_y = max(3, int(round(median_h * 0.24)))
    for row in rows:
        for box in _merge_component_boxes_into_lines(row, gap_x=line_gap_x, gap_y=max(2, int(median_h * 0.35))):
            x1, y1, x2, y2 = _expand_bbox(
                box,
                original_rgb.shape,
                pad_x_ratio=0.0,
                pad_y_ratio=0.0,
                min_pad_x=pad_x,
                min_pad_y=pad_y,
            )
            line_mask[y1:y2, x1:x2] = 255

    line_mask = cv2.bitwise_and(line_mask, interior.astype(np.uint8))
    line_pixels = int(np.count_nonzero(line_mask))
    interior_pixels = int(np.count_nonzero(interior))
    if line_pixels < 8 or line_pixels > int(max(1, interior_pixels) * 0.48):
        return None
    return line_mask


def _merged_source_text_boxes_for_white_fill(
    text: dict,
    shape: tuple[int, ...],
    focus: list[int] | None,
) -> list[list[int]]:
    height = int(shape[0]) if shape else 0
    width = int(shape[1]) if len(shape) > 1 else 0
    if height <= 0 or width <= 0 or not isinstance(text, dict):
        return []

    focus_area = _bbox_area_safe(focus) if focus is not None else 0
    raw_boxes: list = []
    for key in ("_merged_source_bboxes", "merged_source_bboxes"):
        values = text.get(key)
        if isinstance(values, list):
            raw_boxes.extend(values)

    source_boxes: list[list[int]] = []
    for raw_box in raw_boxes:
        box = _coerce_bbox(raw_box)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        normalized = [x1, y1, x2, y2]
        if focus_area > 0 and _bbox_area_safe(normalized) >= int(focus_area * 0.92):
            continue
        if normalized not in source_boxes:
            source_boxes.append(normalized)
    return source_boxes


def _line_mask_from_white_source_boxes(
    original_rgb: np.ndarray,
    source_boxes: list[list[int]],
    interior: np.ndarray,
) -> np.ndarray | None:
    if original_rgb.size == 0 or not source_boxes:
        return None
    line_mask = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
    heights = [max(1, int(box[3]) - int(box[1])) for box in source_boxes]
    median_height = int(np.median(np.asarray(heights, dtype=np.int32))) if heights else 12
    pad_x = max(4, int(round(median_height * 0.18)))
    pad_y = max(4, int(round(median_height * 0.18)))
    for box in source_boxes:
        x1, y1, x2, y2 = _expand_bbox(
            box,
            original_rgb.shape,
            pad_x_ratio=0.0,
            pad_y_ratio=0.0,
            min_pad_x=pad_x,
            min_pad_y=pad_y,
        )
        line_mask[y1:y2, x1:x2] = 255

    line_mask = cv2.bitwise_and(line_mask, interior.astype(np.uint8))
    line_pixels = int(np.count_nonzero(line_mask))
    interior_pixels = int(np.count_nonzero(interior))
    if line_pixels < 24 or line_pixels > int(max(1, interior_pixels) * 0.42):
        return None
    return line_mask


def _build_white_balloon_text_line_fill_mask(original_rgb: np.ndarray, text: dict) -> np.ndarray | None:
    if original_rgb.size == 0 or not isinstance(text, dict):
        return None
    resolved_bbox = _resolve_white_balloon_bbox(original_rgb, text)
    raw_balloon_bbox = _coerce_bbox(text.get("balloon_bbox"))
    candidate_bboxes = [bbox for bbox in (resolved_bbox, raw_balloon_bbox) if bbox is not None]
    if not candidate_bboxes:
        return None
    balloon_mask = None
    balloon_pixels = -1
    for candidate_bbox in candidate_bboxes:
        candidate_mask = _extract_white_balloon_fill_mask(original_rgb, candidate_bbox)
        if not np.any(candidate_mask):
            legacy_mask = _extract_white_balloon_mask_legacy(original_rgb, candidate_bbox)
            if isinstance(legacy_mask, np.ndarray):
                candidate_mask = legacy_mask
        candidate_pixels = int(np.count_nonzero(candidate_mask))
        if candidate_pixels > balloon_pixels:
            balloon_pixels = candidate_pixels
            balloon_mask = candidate_mask
    if balloon_mask is None:
        return None
    if not np.any(balloon_mask):
        return None
    distance = cv2.distanceTransform((balloon_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    interior = (distance > 2.0).astype(np.uint8) * 255
    if not np.any(interior):
        interior = (distance > 1.0).astype(np.uint8) * 255
    if not np.any(interior):
        return None
    focus = _coerce_bbox(text.get("text_pixel_bbox")) or _coerce_bbox(text.get("bbox")) or raw_balloon_bbox
    source_line_mask = _line_mask_from_white_source_boxes(
        original_rgb,
        _merged_source_text_boxes_for_white_fill(text, original_rgb.shape, focus),
        interior,
    )
    if source_line_mask is not None:
        return source_line_mask
    extracted_boxes = _extract_white_balloon_text_boxes(original_rgb, focus) if focus is not None else []
    if extracted_boxes:
        line_mask = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
        for box in extracted_boxes:
            x1, y1, x2, y2 = _expand_bbox(
                box,
                original_rgb.shape,
                pad_x_ratio=0.0,
                pad_y_ratio=0.0,
                min_pad_x=4,
                min_pad_y=4,
            )
            line_mask[y1:y2, x1:x2] = 255
        line_mask = cv2.bitwise_and(line_mask, interior.astype(np.uint8))
        line_pixels = int(np.count_nonzero(line_mask))
        interior_pixels = int(np.count_nonzero(interior))
        min_line_pixels = max(64, int(_bbox_area_safe(focus) * 0.006))
        if min_line_pixels <= line_pixels <= int(max(1, interior_pixels) * 0.48):
            return line_mask
    return _build_white_balloon_residual_line_box_mask(original_rgb, text, interior)


def _sample_white_balloon_fill_color(
    original_rgb: np.ndarray,
    interior_mask: np.ndarray,
    exclude_mask: np.ndarray | None = None,
    sample_bbox: list[int] | None = None,
) -> np.ndarray:
    if original_rgb.size == 0 or not isinstance(interior_mask, np.ndarray) or not np.any(interior_mask):
        return np.asarray([255, 255, 255], dtype=np.uint8)
    if interior_mask.shape[:2] != original_rgb.shape[:2]:
        return np.asarray([255, 255, 255], dtype=np.uint8)

    sample_mask = interior_mask > 0
    if sample_bbox is not None:
        height, width = original_rgb.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in sample_bbox]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            local_mask = np.zeros(interior_mask.shape[:2], dtype=bool)
            local_mask[y1:y2, x1:x2] = True
            local_sample = sample_mask & local_mask
            if int(np.count_nonzero(local_sample)) >= 24:
                sample_mask = local_sample
    if isinstance(exclude_mask, np.ndarray) and exclude_mask.shape[:2] == interior_mask.shape[:2]:
        sample_mask &= exclude_mask == 0
    if not np.any(sample_mask):
        sample_mask = interior_mask > 0

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    bright_non_text = sample_mask & (gray >= 190) & (hsv[:, :, 2] >= 190) & (hsv[:, :, 1] <= 96)
    if int(np.count_nonzero(bright_non_text)) >= 24:
        sample_mask = bright_non_text
    elif int(np.count_nonzero(sample_mask)) < 24:
        return np.asarray([255, 255, 255], dtype=np.uint8)

    fill = np.median(original_rgb[sample_mask].astype(np.float32), axis=0).clip(0, 255)
    return np.asarray([int(round(float(v))) for v in fill], dtype=np.uint8)


def _apply_white_balloon_residual_force_fill(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
    texts: list[dict],
) -> np.ndarray:
    return cleaned_rgb.copy()


def _run_koharu_blockwise_inpaint_page(
    image_np: np.ndarray,
    ocr_data: dict,
    inpainter,
) -> np.ndarray:
    height, width = image_np.shape[:2]
    vision_blocks = list(ocr_data.get("_vision_blocks", []))
    if not vision_blocks:
        return image_np.copy()
    text_segmenter = _get_text_segmenter_for_page(ocr_data)

    working_mask = vision_blocks_to_mask(
        image_np.shape,
        vision_blocks,
        image_rgb=image_np,
        expand_mask=False,
        mask_strategy=((ocr_data.get("_engine_preset") or {}).get("mask_strategy") if isinstance(ocr_data.get("_engine_preset"), dict) else ""),
        ocr_texts=list(ocr_data.get("texts", [])) + list(ocr_data.get("_oar_ocr_regions", [])),
        text_segmenter=text_segmenter,
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

    cleaned = _apply_post_inpaint_cleanup(
        image_np,
        inpainted,
        list(ocr_data.get("texts", [])),
    )
    if _has_white_balloon_text_residual(image_np, cleaned, list(ocr_data.get("texts", []))):
        logger.info("Resíduo persistente em balão branco após inpaint blockwise; usando fallback full-page")
        return _apply_inpainting_round(image_np, ocr_data, inpainter)
    return cleaned


def _run_masked_inpaint_passes(
    inpainter,
    image_np: np.ndarray,
    mask: np.ndarray,
    batch_size: int = 4,
    debug: DebugRunRecorder | None = None,
    seam_cleanup: bool = False,
    multi_pass: bool = False,
    force_no_tiling: bool = True,
    prefer_roi: bool = True,
    texts: list[dict] | None = None,
    expand_mask: bool = True,
    crop_windows: list[list[int]] | None = None,
) -> dict:
    assert mask.shape[:2] == image_np.shape[:2], (
        f"mask/image mismatch before passes: mask={mask.shape[:2]} image={image_np.shape[:2]}"
    )
    if expand_mask:
        expanded = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
    else:
        expanded = mask.astype(np.uint8, copy=True)

    if (
        crop_windows is None
        and prefer_roi
        and not multi_pass
        and not seam_cleanup
        and force_no_tiling
    ):
        crop_windows = _clustered_inpaint_crop_windows(
            expanded,
            image_np.shape,
            texts=texts,
        )

    if crop_windows:
        crop_started = time.perf_counter()
        working_mask = expanded.copy()
        output = image_np.copy()
        windows_used = 0
        crop_area = 0
        lama_ms = 0.0
        full_area = max(1, int(image_np.shape[0]) * int(image_np.shape[1]))
        for raw_window in crop_windows:
            window = _clip_bbox_to_shape(raw_window, image_np.shape)
            if window is None:
                continue
            wx1, wy1, wx2, wy2 = window
            crop_mask = working_mask[wy1:wy2, wx1:wx2].copy()
            if not np.any(crop_mask):
                continue
            crop_image = output[wy1:wy2, wx1:wx2].copy()
            lama_started = time.perf_counter()
            crop_output = _call_inpainter(
                inpainter,
                crop_image,
                crop_mask,
                batch_size=batch_size,
                debug=debug,
                force_no_tiling=force_no_tiling,
            )
            lama_ms += (time.perf_counter() - lama_started) * 1000.0
            if crop_output.shape[:2] != crop_image.shape[:2]:
                raise ValueError(
                    f"crop inpaint retornou shape {crop_output.shape[:2]} esperado {crop_image.shape[:2]}"
                )
            target = output[wy1:wy2, wx1:wx2]
            paste_mask = crop_mask > 0
            target[paste_mask] = crop_output[paste_mask]
            output[wy1:wy2, wx1:wx2] = target
            window_mask = working_mask[wy1:wy2, wx1:wx2]
            window_mask[paste_mask] = 0
            windows_used += 1
            crop_area += int((wx2 - wx1) * (wy2 - wy1))

        roi_select_ms = round((time.perf_counter() - crop_started) * 1000.0, 3)
        roi_area_ratio = round(min(1.0, crop_area / float(full_area)), 6)
        if debug is not None:
            debug.log(
                "crop_windows",
                count=int(windows_used),
                requested=int(len(crop_windows)),
                mask_expanded=bool(expand_mask),
                roi_area_ratio=roi_area_ratio,
            )
        return {
            "expanded_mask": expanded,
            "raw_output": output.copy(),
            "after_roi_paste": output.copy(),
            "after_seam_cleanup": output.copy(),
            "final_output": output,
            "cleanup_base_mask": expanded,
            "fallback_to_legacy": False,
            "fallback_error": "",
            "_t_roi_select_ms": roi_select_ms,
            "_t_lama_ms": round(lama_ms, 3),
            "used_roi_crop": bool(windows_used and roi_area_ratio < 1.0),
            "roi_area_ratio": roi_area_ratio,
            "crop_windows_used": int(windows_used),
        }

    roi_started = time.perf_counter()
    first_roi, first_uses_roi = _select_inpaint_roi(
        expanded,
        image_np.shape,
        prefer_roi=prefer_roi,
        texts=texts,
    )
    roi_select_ms = round((time.perf_counter() - roi_started) * 1000.0, 3)
    rx1, ry1, rx2, ry2 = first_roi
    full_area = max(1, int(image_np.shape[0]) * int(image_np.shape[1]))
    roi_area_ratio = round(((rx2 - rx1) * (ry2 - ry1)) / float(full_area), 6)
    if debug is not None:
        debug.log(
            "roi",
            x1=int(rx1),
            y1=int(ry1),
            x2=int(rx2),
            y2=int(ry2),
            width=int(rx2 - rx1),
            height=int(ry2 - ry1),
            resize_width=int(rx2 - rx1),
            resize_height=int(ry2 - ry1),
            padding={"top": 0, "bottom": 0, "left": 0, "right": 0},
            shape_before_inpaint=list(image_np[ry1:ry2, rx1:rx2].shape),
            shape_after_inpaint=list(image_np[ry1:ry2, rx1:rx2].shape),
            shape_before_paste=list(image_np[ry1:ry2, rx1:rx2].shape),
            shape_after_paste=list(image_np.shape),
            paste_offsets={"x": int(rx1), "y": int(ry1)},
            clamped={
                "left": int(rx1) == 0,
                "top": int(ry1) == 0,
                "right": int(rx2) == int(image_np.shape[1]),
                "bottom": int(ry2) == int(image_np.shape[0]),
            },
            passes=1 if not multi_pass else 2,
            seam_cleanup=bool(seam_cleanup),
            cropped=bool(first_uses_roi),
            mask_expanded=bool(expand_mask),
        )
    fallback_to_legacy = False
    fallback_error = ""
    raw_output = None
    after_paste = None
    cleanup_base_mask = expanded
    lama_ms = 0.0

    if not multi_pass:
        try:
            lama_started = time.perf_counter()
            first_pass = _call_inpainter_in_roi(
                inpainter,
                image_np,
                expanded,
                first_roi,
                first_uses_roi,
                batch_size=batch_size,
                debug=debug,
                force_no_tiling=force_no_tiling,
            )
            lama_ms += (time.perf_counter() - lama_started) * 1000.0
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
            prefer_roi = False
            first_roi, first_uses_roi = _select_inpaint_roi(
                expanded,
                image_np.shape,
                prefer_roi=False,
                texts=texts,
            )
            if debug is not None:
                debug.log("single_pass_fallback", reason=fallback_error)

    if multi_pass:
        lama_started = time.perf_counter()
        first_pass = _call_inpainter_in_roi(
            inpainter,
            image_np,
            expanded,
            first_roi,
            first_uses_roi,
            batch_size=batch_size,
            debug=debug,
            force_no_tiling=force_no_tiling,
        )
        lama_ms += (time.perf_counter() - lama_started) * 1000.0
        second_mask = cv2.dilate(
            expanded,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        second_roi, second_uses_roi = _select_inpaint_roi(
            second_mask,
            image_np.shape,
            prefer_roi=prefer_roi,
            texts=texts,
        )
        lama_started = time.perf_counter()
        second_pass = _call_inpainter_in_roi(
            inpainter,
            first_pass,
            second_mask,
            second_roi,
            second_uses_roi,
            batch_size=batch_size,
            debug=debug,
            force_no_tiling=force_no_tiling,
        )
        lama_ms += (time.perf_counter() - lama_started) * 1000.0
        cleanup_mask = _build_residual_cleanup_mask(second_pass, second_mask)
        residual_ratio = float(np.count_nonzero(cleanup_mask)) / float(max(1, np.count_nonzero(second_mask)))
        if np.any(cleanup_mask) and (not _inpaint_roi_tighten_enabled() or residual_ratio > 0.05):
            third_mask = cv2.dilate(
                cleanup_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            third_roi, third_uses_roi = _select_inpaint_roi(
                third_mask,
                image_np.shape,
                prefer_roi=prefer_roi,
                texts=texts,
            )
            lama_started = time.perf_counter()
            second_pass = _call_inpainter_in_roi(
                inpainter,
                second_pass,
                third_mask,
                third_roi,
                third_uses_roi,
                batch_size=batch_size,
                debug=debug,
                force_no_tiling=force_no_tiling,
            )
            lama_ms += (time.perf_counter() - lama_started) * 1000.0
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
        "_t_roi_select_ms": roi_select_ms,
        "_t_lama_ms": round(lama_ms, 3),
        "used_roi_crop": bool(first_uses_roi),
        "roi_area_ratio": roi_area_ratio,
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
    texts = list(ocr_data.get("texts", [])) + list(ocr_data.get("_oar_ocr_regions", []))
    text_segmenter = _get_text_segmenter_for_page(ocr_data) if vision_blocks else None
    precomputed_mask = ocr_data.get("_precomputed_inpaint_mask") if isinstance(ocr_data, dict) else None
    if isinstance(precomputed_mask, np.ndarray) and precomputed_mask.shape[:2] == image_np.shape[:2]:
        full_mask = np.where(precomputed_mask > 0, 255, 0).astype(np.uint8)
    else:
        full_mask = (
            vision_blocks_to_mask(
                image_np.shape,
                vision_blocks,
                image_rgb=image_np,
                mask_strategy=((ocr_data.get("_engine_preset") or {}).get("mask_strategy") if isinstance(ocr_data.get("_engine_preset"), dict) else ""),
                ocr_texts=texts,
                text_segmenter=text_segmenter,
            )
            if vision_blocks
            else np.zeros(image_np.shape[:2], dtype=np.uint8)
        )
    if np.any(full_mask):
        strict_mask_only = _strict_inpaint_mask_only_for_page(ocr_data)
        strict_crop_windows = (
            _strict_cjk_aot_crop_windows(full_mask, vision_blocks, image_np.shape)
            if strict_mask_only
            else None
        )
        if debug is None and not seam_cleanup and not multi_pass and force_no_tiling:
            result = _run_masked_inpaint_passes(
                inpainter,
                image_np,
                full_mask,
                batch_size=4,
                texts=texts,
                expand_mask=not strict_mask_only,
                crop_windows=strict_crop_windows,
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
                texts=texts,
                expand_mask=not strict_mask_only,
                crop_windows=strict_crop_windows,
            )
        if debug is not None:
            return result
        if isinstance(result, dict):
            if strict_mask_only:
                result["final_output"] = _apply_cjk_mask_residual_cleanup(
                    image_np,
                    result["final_output"],
                    result.get("expanded_mask"),
                )
            stats = {
                key: result[key]
                for key in ("_t_roi_select_ms", "_t_lama_ms", "used_roi_crop", "roi_area_ratio", "crop_windows_used")
                if key in result
            }
            if isinstance(ocr_data, dict):
                ocr_data["_inpaint_round_stats"] = stats
            limited_raw, raw_limit_pixels, raw_changed_outside = _clamp_image_to_limit_mask(
                image_np,
                result["final_output"],
                result.get("expanded_mask"),
                texts,
                include_text_bboxes=not strict_mask_only,
            )
            result["final_output"] = limited_raw
            if isinstance(ocr_data, dict):
                ocr_data["_inpaint_raw_limit_mask_pixels"] = raw_limit_pixels
                ocr_data["_inpaint_raw_changed_outside_limit_mask"] = raw_changed_outside
            if ocr_data.get("_skip_internal_post_cleanup"):
                return limited_raw
            cleaned, cleanup_limit_stats = _apply_post_inpaint_cleanup_timed(
                image_np,
                result["final_output"],
                texts,
                limit_mask=result.get("expanded_mask"),
                include_text_bboxes_in_limit=not strict_mask_only,
            )
            if isinstance(ocr_data, dict):
                ocr_data.update(cleanup_limit_stats)
            if _has_white_balloon_text_residual(image_np, cleaned, texts):
                if strict_mask_only:
                    if isinstance(ocr_data, dict):
                        ocr_data["_inpaint_white_residual_force_fill"] = False
                        ocr_data["_inpaint_white_residual_force_fill_skipped"] = "strict_cjk_aot"
                    return cleaned
                forced = _apply_white_balloon_residual_force_fill(image_np, cleaned, texts)
                limit_mask = result.get("expanded_mask")
                cleanup_limit_mask = _build_post_cleanup_limit_mask(
                    limit_mask,
                    texts,
                    forced.shape[:2],
                    include_text_bboxes=not strict_mask_only,
                )
                if cleanup_limit_mask is not None:
                    allowed = cleanup_limit_mask > 0
                    if np.any(allowed):
                        limited_forced = forced.copy()
                        limited_forced[~allowed] = cleaned[~allowed]
                        forced = limited_forced
                forced = _restore_dark_line_art_outside_text_geometry(image_np, forced, _white_cleanup_texts(image_np, texts))
                if isinstance(ocr_data, dict):
                    ocr_data["_inpaint_white_residual_force_fill"] = bool(np.any(forced != cleaned))
                return forced
            return cleaned
        return result
    else:
        return image_np.copy()


def _select_recovery_match(base_texts: list[dict], recovered_text: dict) -> int | None:
    residual_bbox = recovered_text.get("bbox", [0, 0, 0, 0])
    residual_norm = _normalize_text_key(recovered_text.get("text", ""))
    best_index = None
    best_score = -1e9

    for index, base in enumerate(base_texts):
        if not isinstance(base, dict):
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
        if not isinstance(recovered_text, dict):
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
        for key in ("rotation_deg", "rotation_source", "line_polygons", "text_pixel_bbox"):
            if recovered_text.get(key) not in (None, [], "") and target.get(key) in (None, [], "", 0, 0.0):
                target[key] = recovered_text[key]
        if match_index < len(updated_page["_vision_blocks"]):
            updated_page["_vision_blocks"][match_index]["bbox"] = _bbox_union(
                updated_page["_vision_blocks"][match_index].get("bbox", [0, 0, 0, 0]),
                recovered_block.get("bbox", [0, 0, 0, 0]),
            )
            updated_page["_vision_blocks"][match_index]["confidence"] = max(
                float(updated_page["_vision_blocks"][match_index].get("confidence", 0.0)),
                float(recovered_block.get("confidence", 0.0)),
            )
            for key in ("rotation_deg", "rotation_source", "line_polygons", "text_pixel_bbox"):
                if (
                    recovered_block.get(key) not in (None, [], "")
                    and updated_page["_vision_blocks"][match_index].get(key) in (None, [], "", 0, 0.0)
                ):
                    updated_page["_vision_blocks"][match_index][key] = recovered_block[key]

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
    preserve_cjk_sfx: bool = True,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:
    height, width = image_rgb.shape[:2]
    page_texts = []
    vision_blocks = []
    total_blocks = max(1, len(blocks))
    normalized_source_lang = normalize_paddleocr_language(idioma_origem)
    page_number = infer_page_number(image_path)
    page_profile = infer_page_profile(page_number, image_rgb.shape, len(blocks))
    editorial_credit_drop_count = 0
    run_on_suspect_count = 0
    run_on_resolved_count = 0

    _emit_stage_progress(progress_callback, "build_blocks", 0.74, "Montando blocos OCR")
    record_decision(
        stage="ocr",
        action="classify_page_profile",
        reason=page_profile,
        page=page_number,
        details={"block_count": len(blocks), "image_path": image_path},
    )

    for index, (block, raw_text) in enumerate(zip(blocks, texts), start=1):
        layer_ref = f"ocr_{index:03d}"
        bbox = [int(round(v)) for v in block.xyxy]
        bbox[0] = max(0, min(width, bbox[0]))
        bbox[2] = max(0, min(width, bbox[2]))
        bbox[1] = max(0, min(height, bbox[1]))
        bbox[3] = max(0, min(height, bbox[3]))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="invalid_bbox",
                page=page_number,
                layer=layer_ref,
                bbox=bbox,
            )
            continue

        raw_record = raw_text if isinstance(raw_text, dict) else {}
        raw_text_value = raw_record.get("text") or raw_record.get("translated") or raw_text
        if isinstance(raw_text, dict) and not (raw_record.get("text") or raw_record.get("translated")):
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="structured_payload",
                page=page_number,
                layer=layer_ref,
                bbox=bbox,
                details={"raw_kind": "dict_without_text"},
            )
            continue
        confidence = round(float(getattr(block, "confidence", 0.0)), 3)
        raw_text_str = str(raw_text_value or "").strip()
        if is_structured_ocr_payload(raw_text_str):
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="structured_payload",
                page=page_number,
                layer=layer_ref,
                text=raw_text_str,
                bbox=bbox,
            )
            continue
        cleaned = fix_ocr_errors(raw_text_str, idioma_origem=idioma_origem)
        if not cleaned:
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="empty_after_cleanup",
                page=page_number,
                layer=layer_ref,
                bbox=bbox,
            )
            continue
        informative_qa_flags: list[str] = []

        if False and is_watermark(cleaned):
            line_polygons: list = []
            text_pixel_bbox = _derive_text_pixel_bbox(image_rgb, raw_record.get("bbox") or bbox, line_polygons)
            if text_pixel_bbox is None:
                text_pixel_bbox = bbox
            text_entry = {
                "id": layer_ref,
                "text_id": layer_ref,
                "text": cleaned,
                "bbox": bbox,
                "confidence": confidence,
                "confidence_raw": confidence,
                "tipo": "text",
                "estilo": analyze_style(image_rgb, bbox),
                "style_origin": "auto",
                "ocr_source": f"vision-{ocr_backend}",
                "ocr_reviewed": False,
                "ocr_profile": profile,
                "ocr_semantic_reviewed": False,
                "ocr_mode": ocr_backend,
                "skip_processing": False,
                "skip_reason": None,
                "preserve_original": False,
                "content_class": "text",
                "is_watermark": False,
                "is_non_english": False,
                "translate_policy": "translate",
                "render_policy": "normal",
                "route_action": "translate_inpaint_render",
                "route_reason": "watermark_detected",
                "needs_review": False,
                "line_polygons": line_polygons,
                "text_pixel_bbox": text_pixel_bbox,
                "balloon_type": "white",
                "page_profile": page_profile,
                "block_profile": "watermark",
                "qa_flags": [],
            }
            apply_route_action(
                text_entry,
                route_action=text_entry["route_action"],
                route_reason=text_entry["route_reason"],
            )
            _apply_balloon_geometry_to_text_entry(text_entry, raw_record, block, (height, width))
            page_texts.append(text_entry)
            serialized_block = _apply_text_geometry_to_serialized_block(
                _serialize_block(block, (height, width)),
                text_entry,
            )
            serialized_block["text_id"] = layer_ref
            serialized_block["confidence_raw"] = confidence
            serialized_block["content_class"] = text_entry["content_class"]
            serialized_block["is_watermark"] = True
            serialized_block["route_action"] = text_entry["route_action"]
            serialized_block["route_reason"] = text_entry["route_reason"]
            serialized_block["skip_processing"] = text_entry["skip_processing"]
            vision_blocks.append(serialized_block)
            record_decision(
                stage="ocr",
                action="route_block",
                reason="watermark_detected",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
            )
            continue

        if False and is_editorial_credit(cleaned):
            editorial_credit_drop_count += 1
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="editorial_credit",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
            )
            continue

        if is_punctuation_only_noise(cleaned):
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="punctuation_only",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
            )
            continue

        # Ignorar textos não-latinos apenas se a origem for inglês.
        # Se a origem for CJK, devemos manter o texto para tradução.
        if False and normalized_source_lang == "en" and is_non_english(cleaned):
            line_polygons: list = []
            text_pixel_bbox = _derive_text_pixel_bbox(image_rgb, raw_record.get("bbox") or bbox, line_polygons)
            if text_pixel_bbox is None:
                text_pixel_bbox = bbox
            korean_sfx = is_korean_sfx(cleaned)
            text_entry = {
                "id": layer_ref,
                "text_id": layer_ref,
                "text": cleaned,
                "bbox": bbox,
                "confidence": confidence,
                "confidence_raw": confidence,
                "tipo": "sfx" if korean_sfx else "preserved_text",
                "estilo": analyze_style(image_rgb, bbox),
                "style_origin": "auto",
                "ocr_source": f"vision-{ocr_backend}",
                "ocr_reviewed": False,
                "ocr_profile": profile,
                "ocr_semantic_reviewed": False,
                "ocr_mode": ocr_backend,
                "skip_processing": False,
                "skip_reason": None,
                "preserve_original": False,
                "ignored_reason": "cjk_sfx_preserved" if korean_sfx else "non_english_text_preserved",
                "content_class": "sfx" if korean_sfx else "preserved_non_english",
                "is_non_english": True,
                "translate_policy": "translate",
                "render_policy": "normal",
                "route_action": "translate_inpaint_render",
                "route_reason": "korean_sfx_preserved_by_default" if korean_sfx else "non_english_text_preserved",
                "needs_review": False,
                "line_polygons": line_polygons,
                "text_pixel_bbox": text_pixel_bbox,
                "balloon_type": "textured",
                "page_profile": page_profile,
                "block_profile": "non_english_text",
                "qa_flags": ["sfx_preserved"] if korean_sfx else ["non_english_preserved"],
            }
            apply_route_action(
                text_entry,
                route_action=text_entry["route_action"],
                route_reason=text_entry["route_reason"],
            )
            _apply_balloon_geometry_to_text_entry(text_entry, raw_record, block, (height, width))
            page_texts.append(text_entry)
            serialized_block = _apply_text_geometry_to_serialized_block(
                _serialize_block(block, (height, width)),
                text_entry,
            )
            serialized_block["text_id"] = layer_ref
            serialized_block["confidence_raw"] = confidence
            vision_blocks.append(serialized_block)
            record_decision(
                stage="ocr",
                action="preserve_block",
                reason=text_entry["route_reason"],
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
            )
            continue

        if is_hallucination(cleaned, bbox, confidence):
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="vlm_failure_phrase" if is_vlm_failure_phrase(cleaned) else "ocr_hallucination",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
                details={"confidence": confidence},
            )
            continue

        early_white_balloon_context = _is_white_balloon_context_for_text(
            image_rgb,
            bbox,
            cleaned,
            source_lang=normalized_source_lang,
            raw_record=raw_record,
            block=block,
        )
        tipo = "text"
        content_class_value = "text"
        credit_name_list = False
        sfx_split_off = None
        skip_translation_content = False
        pre_semantic_run_on = _ocr_run_on_guard_enabled() and has_run_on_tokens(cleaned)
        original_cleaned = cleaned
        cleaned = semantic_refine_text(cleaned, tipo=tipo, confidence=confidence)
        run_on_suspect = _ocr_run_on_guard_enabled() and has_run_on_tokens(cleaned)
        if pre_semantic_run_on and not run_on_suspect:
            run_on_resolved_count += 1
            record_decision(
                stage="ocr",
                action="repair_block",
                reason="ocr_run_on_resolved",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
                details={"confidence": confidence, "original_text": original_cleaned},
            )
        if run_on_suspect:
            run_on_suspect_count += 1
            record_decision(
                stage="ocr",
                action="flag_block",
                reason="ocr_run_on_suspect",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
                details={"confidence": confidence},
            )
        if False and is_editorial_credit(cleaned):
            editorial_credit_drop_count += 1
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="editorial_credit",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
                details={"phase": "semantic_review"},
            )
            continue
        is_white_balloon = early_white_balloon_context
        if is_ghost_ocr_noise(
            cleaned,
            bbox,
            confidence,
            is_white_balloon=is_white_balloon,
            image_shape=image_rgb.shape,
        ):
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="ghost_ocr_noise",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
                details={"confidence": confidence},
            )
            continue
        block_profile = infer_block_profile(
            cleaned,
            bbox,
            tipo,
            image_rgb.shape,
            page_profile=page_profile,
            is_white_balloon=is_white_balloon,
        )
        title_rules_enabled = bool(work_title_user_provided and str(work_title or "").strip())
        text_matches_work_title = bool(
            title_rules_enabled
            and _text_matches_work_title(cleaned, work_title, work_title_aliases)
        )
        if block_profile == "cover_title_logo" and not text_matches_work_title:
            block_profile = "standard"
        elif (
            text_matches_work_title
            and page_profile == "cover_opening"
            and not is_white_balloon
        ):
            block_profile = "cover_title_logo"
        if block_profile != "standard":
            record_decision(
                stage="ocr",
                action="classify_block_profile",
                reason=block_profile,
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
                details={"confidence": confidence, "tipo": tipo, "page_profile": page_profile},
            )
        suspicious_threshold = suspicious_confidence_threshold(block_profile, page_profile)
        force_review_low_confidence_fragment = False
        if looks_suspicious(cleaned, confidence) and confidence < suspicious_threshold:
            retain_for_review = bool(
                is_white_balloon
                and block_profile == "white_balloon"
                and should_retain_low_confidence_dialogue_ocr(
                    {
                        "text": cleaned,
                        "confidence": confidence,
                        "bbox": bbox,
                        "balloon_bbox": raw_record.get("balloon_bbox") or bbox,
                        "tipo": tipo,
                        "content_class": content_class_value,
                    }
                )
            )
            if retain_for_review:
                for flag in ("low_ocr_confidence", "ocr_partial_low_confidence_fragment"):
                    if flag not in informative_qa_flags:
                        informative_qa_flags.append(flag)
                record_decision(
                    stage="ocr",
                    action="flag_block",
                    reason="ocr_partial_low_confidence_fragment",
                    page=page_number,
                    layer=layer_ref,
                    text=cleaned,
                    bbox=bbox,
                    details={
                        "confidence": confidence,
                        "threshold": suspicious_threshold,
                        "block_profile": block_profile,
                        "page_profile": page_profile,
                    },
                )
            else:
                for flag in ("low_ocr_confidence", "suspicious_low_confidence"):
                    if flag not in informative_qa_flags:
                        informative_qa_flags.append(flag)
                record_decision(
                    stage="ocr",
                    action="flag_block",
                    reason="suspicious_low_confidence",
                    page=page_number,
                    layer=layer_ref,
                    text=cleaned,
                    bbox=bbox,
                    details={
                        "confidence": confidence,
                        "threshold": suspicious_threshold,
                        "block_profile": block_profile,
                        "page_profile": page_profile,
                    },
                )
        estilo = analyze_style(image_rgb, bbox)
        if False and is_short_textured_sfx_or_noise(
            cleaned,
            bbox,
            confidence,
            is_white_balloon,
        ):
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="textured_sfx_or_noise",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
                details={
                    "confidence": confidence,
                    "tipo": tipo,
                    "page_profile": page_profile,
                    "block_profile": block_profile,
                },
            )
            continue
        line_polygons = _normalize_line_polygons(
            raw_record.get("line_polygons")
            or getattr(block, "line_polygons", None)
            or []
        )
        text_pixel_bbox = _coerce_bbox(raw_record.get("text_pixel_bbox"))
        if text_pixel_bbox is None:
            text_pixel_bbox = _derive_text_pixel_bbox(image_rgb, raw_record.get("bbox") or bbox, line_polygons)
        if text_pixel_bbox is None:
            text_pixel_bbox = bbox
        rotation_deg, rotation_source = _rotation_metadata_from_ocr(raw_record, block, line_polygons)
        if False and preserve_cjk_sfx and should_preserve_cjk_sfx_candidate(
            cleaned,
            bbox,
            confidence,
            is_white_balloon=is_white_balloon,
            source_lang=normalized_source_lang,
            image_shape=image_rgb.shape,
            block_profile=block_profile,
        ):
            text_entry = {
                "id": layer_ref,
                "text_id": layer_ref,
                "text": cleaned,
                "bbox": bbox,
                "confidence": confidence,
                "confidence_raw": confidence,
                "tipo": "sfx",
                "estilo": analyze_style(image_rgb, bbox),
                "style_origin": "auto",
                "ocr_source": f"vision-{ocr_backend}",
                "ocr_reviewed": False,
                "ocr_profile": profile,
                "ocr_semantic_reviewed": False,
                "ocr_mode": ocr_backend,
                "skip_processing": False,
                "preserve_original": False,
                "ignored_reason": "cjk_sfx_preserved",
                "content_class": "sfx",
                "is_non_english": True,
                "translate_policy": "translate",
                "render_policy": "normal",
                "route_action": "translate_inpaint_render",
                "route_reason": "korean_sfx_preserved_by_default",
                "line_polygons": line_polygons,
                "text_pixel_bbox": text_pixel_bbox,
                "balloon_type": "white" if is_white_balloon else "textured",
                "page_profile": page_profile,
                "block_profile": block_profile,
                "qa_flags": ["sfx_preserved"],
            }
            apply_route_action(
                text_entry,
                route_action=text_entry["route_action"],
                route_reason=text_entry["route_reason"],
            )
            if rotation_deg != 0.0:
                text_entry["rotation_deg"] = rotation_deg
                text_entry["rotation_source"] = rotation_source
            _apply_balloon_geometry_to_text_entry(text_entry, raw_record, block, (height, width))
            page_texts.append(text_entry)
            record_decision(
                stage="ocr",
                action="preserve_block",
                reason="cjk_sfx_candidate",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
                details={
                    "confidence": confidence,
                    "balloon_type": "white" if is_white_balloon else "textured",
                    "page_profile": page_profile,
                    "block_profile": block_profile,
                },
            )
            continue
        style_bbox = raw_record.get("balloon_bbox") or raw_record.get("layout_bbox") or bbox
        background_rgb = sample_text_background_rgb(image_rgb, style_bbox)
        pre_translation_skip = None
        if False and not credit_name_list:
            pre_translation_skip = _ocr_pre_translation_skip_policy(
                cleaned,
                bbox,
                confidence,
                tipo=tipo,
                page_profile=page_profile,
                block_profile=block_profile,
                is_white_balloon=is_white_balloon,
                image_shape=image_rgb.shape,
                line_polygons=line_polygons,
                run_on_suspect=run_on_suspect,
                pre_semantic_run_on=pre_semantic_run_on,
                source_lang=normalized_source_lang,
                background_rgb=background_rgb,
                title_rules_enabled=title_rules_enabled,
                text_matches_work_title=text_matches_work_title,
            )
        if False and is_short_ornamental_text(
            cleaned,
            confidence,
            bbox,
            image_rgb.shape,
            tipo,
            is_white_balloon,
            page_profile=page_profile,
        ):
            record_decision(
                stage="ocr",
                action="drop_block",
                reason="ornamental_cover_noise",
                page=page_number,
                layer=layer_ref,
                text=cleaned,
                bbox=bbox,
                details={
                    "confidence": confidence,
                    "tipo": tipo,
                    "page_profile": page_profile,
                    "block_profile": block_profile,
                },
            )
            continue
        
        # Regra do Usuário: Balões quadrados e textos sem balão (narração) usam KOMIKAX
        # Classificamos como 'square' inicialmente, mas serah refinado no layout.
        # Mantemos fontes base deterministicamente no OCR e deixamos ajustes mais finos
        # para o layout/typesetter. Isso evita custo extra e ruído de detector neste estágio.
        estilo = normalize_auto_typesetting_style(
            estilo,
            background_rgb=background_rgb,
            force_black_text=is_white_balloon,
        )
        estilo["force_upper"] = True
        qa_flags = [block_profile] if block_profile == "decorative_noise" else []
        for flag in informative_qa_flags:
            if flag in {"low_confidence_visual_noise", "low_ocr_confidence", "suspicious_low_confidence", "ocr_partial_low_confidence_fragment"}:
                continue
            if flag not in qa_flags:
                qa_flags.append(flag)
        if run_on_suspect and "ocr_run_on_suspect" not in qa_flags:
            qa_flags.append("ocr_run_on_suspect")
        if pre_translation_skip:
            for flag in pre_translation_skip.get("qa_flags", []):
                if flag and flag not in qa_flags:
                    qa_flags.append(str(flag))
        skip_reason = None
        content_class_value = "text"
        needs_review = False
        translate_policy = "translate"
        render_policy = "normal"
        if pre_translation_skip:
            needs_review = bool(pre_translation_skip.get("needs_review", False))
        force_review_low_confidence_fragment = False
        text_entry = {
            "id": layer_ref,
            "text_id": layer_ref,
            "text": cleaned,
            "bbox": bbox,
            "confidence": confidence,
            "confidence_raw": confidence,
            "tipo": "text",
            "estilo": estilo,
            "style_origin": "auto",
            "background_rgb": list(background_rgb),
            "ocr_source": f"vision-{ocr_backend}",
            "ocr_reviewed": False,
            "ocr_profile": profile,
            "ocr_semantic_reviewed": False,
            "ocr_mode": ocr_backend,
            "skip_processing": False,
            "skip_reason": skip_reason,
            "preserve_original": False,
            "content_class": content_class_value,
            "is_watermark": False,
            "is_non_english": is_non_english(cleaned),
            "translate_policy": translate_policy,
            "render_policy": render_policy,
            "needs_review": needs_review,
            "line_polygons": line_polygons,
            "text_pixel_bbox": text_pixel_bbox,
            "balloon_type": "",
            "page_profile": page_profile,
            "block_profile": block_profile,
            "qa_flags": qa_flags,
        }
        if False and force_review_low_confidence_fragment and not credit_name_list:
            apply_route_action(
                text_entry,
                route_action="review_required",
                route_reason="ocr_partial_low_confidence_fragment",
            )
        else:
            apply_route_action(text_entry)
        if rotation_deg != 0.0:
            text_entry["rotation_deg"] = rotation_deg
            text_entry["rotation_source"] = rotation_source
        if sfx_split_off:
            text_entry["_sfx_split_off"] = sfx_split_off
        _apply_balloon_geometry_to_text_entry(text_entry, raw_record, block, (height, width))
        page_texts.append(text_entry)
        record_decision(
            stage="ocr",
            action="accept_block",
            reason="ready_for_layout",
            page=page_number,
            layer=layer_ref,
            text=cleaned,
            bbox=bbox,
            details={
                "confidence": confidence,
                "tipo": "text",
                "balloon_type": "",
                "page_profile": page_profile,
                "block_profile": block_profile,
                "skip_reason": skip_reason,
            },
        )
        serialized_block = _apply_text_geometry_to_serialized_block(
            _serialize_block(block, (height, width)),
            text_entry,
        )
        serialized_block["text_id"] = layer_ref
        serialized_block["confidence_raw"] = confidence
        vision_blocks.append(serialized_block)
        finalize_progress = 0.90 + (index / total_blocks) * 0.08
        _emit_stage_progress(progress_callback, "finalize_blocks", finalize_progress, "Finalizando blocos OCR")

    if editorial_credit_drop_count >= 2:
        page_texts, vision_blocks = _drop_ambiguous_editorial_roles_on_credit_page(
            page_texts,
            vision_blocks,
            page_number=page_number,
        )
    page_texts, vision_blocks = _propagate_scanlation_credit_band_policy(
        page_texts,
        vision_blocks,
        image_rgb.shape,
    )
    if run_on_suspect_count:
        record_decision(
            stage="ocr",
            action="flag_page",
            reason="ocr_run_on_suspect",
            page=page_number,
            details={"count": int(run_on_suspect_count)},
        )

    page_texts, vision_blocks = _finalize_page_ocr_texts(
        page_texts,
        vision_blocks,
        image_rgb.shape,
        page_number,
    )

    return {
        "image": image_path,
        "width": width,
        "height": height,
        "texts": page_texts,
        "_vision_blocks": vision_blocks,
        "page_profile": page_profile,
        "_ocr_stats": {
            "ocr_run_on_suspect_count": int(run_on_suspect_count),
            "ocr_run_on_resolved_count": int(run_on_resolved_count),
        },
    }


def _scan_orphan_lobe_blocks(
    image_rgb: np.ndarray,
    blocks: list,
    ocr,
) -> list:
    """Scan each detected block for connected-balloon lobes missed by the detector.

    For each block whose surrounding white balloon is significantly larger than
    the text area, we run lobe detection and OCR any orphan lobe that has no
    existing block covering it.  Returns the extended blocks list.
    """
    try:
        from layout.balloon_layout import _detect_connected_lobes_from_outline  # type: ignore
    except ImportError:
        try:
            from ..layout.balloon_layout import _detect_connected_lobes_from_outline  # type: ignore
        except ImportError:
            return blocks

    img_h, img_w = image_rgb.shape[:2]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    added: list = []

    for block in blocks:
        x1, y1, x2, y2 = [int(v) for v in block.xyxy]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        text_area = bw * bh

        # Search region: generous expansion to see the full balloon
        pad_x = max(20, int(bw * 0.35))
        pad_y = max(20, int(bh * 0.35))
        sx1 = max(0, x1 - pad_x)
        sy1 = max(0, y1 - pad_y)
        sx2 = min(img_w, x2 + pad_x)
        sy2 = min(img_h, y2 + pad_y)
        balloon_search = [sx1, sy1, sx2, sy2]

        # Only attempt for white-looking balloon regions
        if not _is_white_balloon_region(image_rgb, balloon_search):
            continue

        lobes = _detect_connected_lobes_from_outline(
            image_bgr, balloon_search, [x1, y1, x2, y2],
        )
        if len(lobes) < 2:
            continue

        # Check that balloon is significantly bigger than the existing text block
        balloon_area = (sx2 - sx1) * (sy2 - sy1)
        if balloon_area < text_area * 1.8:
            continue

        for lobe in lobes:
            lbox = lobe["bbox"]  # [x1,y1,x2,y2] global
            lx1, ly1, lx2, ly2 = [int(v) for v in lbox]

            # Skip if any existing block covers this lobe adequately
            covered = False
            for existing in list(blocks) + added:
                ex1, ey1, ex2, ey2 = [int(v) for v in existing.xyxy]
                ix1 = max(lx1, ex1)
                iy1 = max(ly1, ey1)
                ix2 = min(lx2, ex2)
                iy2 = min(ly2, ey2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    lobe_area = max(1, (lx2 - lx1) * (ly2 - ly1))
                    if inter / lobe_area > 0.25:
                        covered = True
                        break
            if covered:
                continue

            # OCR the orphan lobe crop
            crop = image_rgb[max(0,ly1):min(img_h,ly2), max(0,lx1):min(img_w,lx2)]
            if crop.size == 0:
                continue
            try:
                recognized = ocr.recognize_batch([crop])
            except Exception:
                continue
            if not recognized or not recognized[0].get("text", "").strip():
                continue
            if float(recognized[0].get("confidence", 0.0)) < 0.40:
                continue

            new_block = SimpleNamespace(
                xyxy=(float(lx1), float(ly1), float(lx2), float(ly2)),
                mask=None,
                confidence=float(recognized[0].get("confidence", 0.55)),
                detector="orphan_lobe_scan",
                line_polygons=None,
                source_direction=None,
            )
            added.append(new_block)
            logger.info(
                "_scan_orphan_lobe_blocks: lobo orfao detectado em [%d,%d,%d,%d] texto=%r",
                lx1, ly1, lx2, ly2, recognized[0].get("text", "")[:40],
            )

    return list(blocks) + added


def _block_xyxy(block) -> list[int] | None:
    raw = getattr(block, "xyxy", None)
    if raw is None and isinstance(block, dict):
        raw = block.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in raw[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _scan_orphan_white_balloon_blocks(image_rgb: np.ndarray, blocks: list) -> list:
    """Add tight text boxes for white speech balloons missed by strip detection."""
    if image_rgb.size == 0:
        return blocks

    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return blocks

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright = ((gray >= 228) & (value >= 228) & (saturation <= 48)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    existing_bboxes = [bbox for bbox in (_block_xyxy(block) for block in blocks) if bbox is not None]
    added = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    image_area = max(1, width * height)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 1800 or area > int(image_area * 0.18):
            continue
        if w < 48 or h < 28:
            continue
        aspect = w / float(max(1, h))
        if aspect < 0.45 or aspect > 5.8:
            continue
        touches_side_edge = x <= 1 or (x + w) >= width - 1
        touches_vertical_edge = y <= 1 or (y + h) >= height - 1
        if touches_side_edge:
            continue

        candidate_bbox = [x, y, x + w, y + h]
        component_has_existing = any(
            _bbox_contains_center(candidate_bbox, existing, margin=12)
            or _bbox_iou(candidate_bbox, existing) >= 0.06
            for existing in existing_bboxes
        )
        if not component_has_existing and (area < 3500 or w < 75 or h < 35 or aspect < 0.70):
            continue
        text_boxes = _extract_white_balloon_text_boxes(image_rgb, candidate_bbox)
        if not text_boxes:
            continue
        text_union = None
        uncovered_count = 0
        for box in text_boxes:
            expanded_text_box = _expand_bbox(
                list(box),
                image_rgb.shape,
                pad_x_ratio=0.10,
                pad_y_ratio=0.22,
                min_pad_x=5,
                min_pad_y=6,
            )
            if any(
                _bbox_contains_center(existing, expanded_text_box, margin=14)
                or _bbox_contains_center(expanded_text_box, existing, margin=14)
                or _bbox_iou(expanded_text_box, existing) >= 0.08
                for existing in existing_bboxes
            ):
                continue
            uncovered_count += 1
            text_union = expanded_text_box if text_union is None else _bbox_union(text_union, expanded_text_box)
        if text_union is None:
            continue
        if touches_vertical_edge and uncovered_count < 2:
            continue
        text_area = max(1, (text_union[2] - text_union[0]) * (text_union[3] - text_union[1]))
        if text_area < 40:
            continue
        text_bbox = text_union

        new_block = SimpleNamespace(
            xyxy=tuple(float(v) for v in text_bbox),
            mask=None,
            confidence=0.56,
            detector="white_balloon_orphan_scan",
            line_polygons=None,
            source_direction=None,
        )
        added.append(new_block)
        existing_bboxes.append(text_bbox)
        logger.info(
            "_scan_orphan_white_balloon_blocks: balao branco sem texto detectado em %s; texto=%s",
            candidate_bbox,
            text_bbox,
        )

    added.extend(_scan_uncovered_white_text_line_blocks(image_rgb, blocks, existing_bboxes))

    if not added:
        return blocks
    def _sort_key(block) -> tuple[int, int]:
        bbox = _block_xyxy(block) or [0, 0, 0, 0]
        return bbox[1], bbox[0]

    return sorted(list(blocks) + added, key=_sort_key)


def _run_detect_ocr_on_image(
    image_rgb: np.ndarray,
    image_label: str,
    profile: str = "quality",
    progress_callback=None,
    idioma_origem: str = "en",
    engine_preset: EnginePreset | None = None,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:
    _emit_stage_progress(progress_callback, "load_detector", 0.08, "Carregando detector de texto")
    detector = _get_detector(profile, model=_detector_model_for_preset(engine_preset))
    _emit_stage_progress(progress_callback, "load_ocr_engine", 0.18, "Carregando motor de OCR")
    ocr = _get_ocr_engine(profile, lang=idioma_origem)
    _emit_stage_progress(progress_callback, "detect_text", 0.38, "Detectando regioes de texto")
    blocks = detector.detect(image_rgb, conf_threshold=_profile_to_detection_threshold(profile))
    blocks = _scan_orphan_lobe_blocks(image_rgb, blocks, ocr)
    detector_backend = str(getattr(detector, "_backend", "") or "")
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
    } and detector_backend != "anime-text-yolo"
    if blocks and backend_name == "paddleocr" and enable_paddle_full_page and hasattr(ocr, "recognize_blocks_from_page"):
        texts = ocr.recognize_blocks_from_page(image_rgb, blocks)
    else:
        crops = [detector.crop(image_rgb, block) for block in blocks]
        texts = ocr.recognize_batch(crops) if crops else []
    page_result = build_page_result(
        image_path=image_label,
        image_rgb=image_rgb,
        blocks=blocks,
        texts=texts,
        profile=profile,
        ocr_backend=backend_name,
        enable_font_detection=True,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
        preserve_cjk_sfx=_preserve_cjk_sfx_for_engine_preset(engine_preset),
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    if _should_run_rotated_text_recovery(page_result, blocks, backend_name, ocr):
        page_result = _run_rotated_text_recovery_pass(
            image_rgb=image_rgb,
            image_label=image_label,
            page_result=page_result,
            ocr=ocr,
            profile=profile,
            backend_name=backend_name,
            idioma_origem=idioma_origem,
            progress_callback=progress_callback,
            engine_preset=engine_preset,
            work_title=work_title,
            work_title_aliases=work_title_aliases,
            work_title_user_provided=work_title_user_provided,
        )
    if _should_run_sparse_page_recovery(page_result, blocks, backend_name):
        recovery_page = _run_sparse_page_recovery_pass(
            image_rgb=image_rgb,
            image_label=image_label,
            ocr=ocr,
            profile=profile,
            idioma_origem=idioma_origem,
            progress_callback=progress_callback,
            engine_preset=engine_preset,
            work_title=work_title,
            work_title_aliases=work_title_aliases,
            work_title_user_provided=work_title_user_provided,
        )
        if recovery_page and recovery_page.get("texts"):
            if page_result.get("texts"):
                page_result, _ = _integrate_recovery_page(page_result, recovery_page)
            else:
                page_result = recovery_page
    page_result = _apply_adaptive_cjk_reocr(
        image_rgb=image_rgb,
        image_label=image_label,
        page_result=page_result,
        blocks=blocks,
        ocr=ocr,
        profile=profile,
        backend_name=backend_name,
        idioma_origem=idioma_origem,
        progress_callback=progress_callback,
        preserve_cjk_sfx=_preserve_cjk_sfx_for_engine_preset(engine_preset),
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    page_result = _reconcile_ocr_with_validated_sources(page_result)
    return page_result


def _run_orientation_recovery(
    image_rgb: np.ndarray,
    image_label: str,
    baseline_page: dict,
    profile: str = "quality",
    progress_callback=None,
    idioma_origem: str = "en",
    engine_preset: EnginePreset | None = None,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict | None:
    if not _should_try_orientation_recovery(baseline_page):
        return None

    original_shape = image_rgb.shape[:2]
    best_page: dict | None = None
    best_score = _orientation_result_score(baseline_page)
    for rotation_deg in (90, 180, 270):
        rotated = _rotate_image_for_orientation(image_rgb, rotation_deg)
        _emit_stage_progress(
            progress_callback,
            "orientation_recovery",
            0.72,
            f"Testando OCR com orientacao {rotation_deg} graus",
        )
        try:
            candidate = _run_detect_ocr_on_image(
                rotated,
                f"{image_label}#rot{rotation_deg}",
                profile=profile,
                progress_callback=progress_callback,
                idioma_origem=idioma_origem,
                engine_preset=engine_preset,
                work_title=work_title,
                work_title_aliases=work_title_aliases,
                work_title_user_provided=work_title_user_provided,
            )
        except Exception as exc:
            logger.warning("Orientation recovery %s falhou em %s: %s", rotation_deg, image_label, exc)
            continue

        remapped = _remap_orientation_recovery_page(
            candidate,
            rotation_deg=rotation_deg,
            original_shape=original_shape,
            rotated_shape=rotated.shape[:2],
        )
        remapped["image"] = image_label
        score = _orientation_result_score(remapped)
        if score > best_score:
            best_page = remapped
            best_score = score

    return best_page


def _should_run_sparse_page_recovery(page_result: dict, blocks: list, backend_name: str) -> bool:
    if backend_name != "paddleocr":
        return False
    if not blocks:
        return False
    accepted = len(page_result.get("texts", []))
    detected = len(blocks)
    return accepted == 0 and detected <= 4


def _adaptive_cjk_bbox_reocr_enabled(source_lang: str) -> bool:
    normalized = str(source_lang or "").strip().lower()
    if normalized not in {"ja", "jp", "ko", "kr", "zh", "zh-cn", "zh-tw"}:
        return False
    flag = os.getenv("TRADUZAI_CJK_BBOX_EXPANDED_REOCR", "1")
    return str(flag).strip().lower() not in {"0", "false", "no", "off"}


def _cjk_page_detect_auto_enabled() -> bool:
    flag = os.getenv("TRADUZAI_CJK_PAGE_DETECT_AUTO", "0")
    return str(flag).strip().lower() in {"1", "true", "yes", "on"}


def _apply_adaptive_cjk_reocr(
    *,
    image_rgb: np.ndarray,
    image_label: str,
    page_result: dict,
    blocks: list,
    ocr,
    profile: str,
    backend_name: str,
    idioma_origem: str,
    progress_callback=None,
    preserve_cjk_sfx: bool = True,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:
    try:
        from qa.page_quality import evaluate_page_quality
    except Exception:
        return page_result

    quality = evaluate_page_quality(page_result, source_lang=idioma_origem)
    route_history = list(page_result.get("route_history") or [])
    route_history.append(
        {
            "stage": "page_quality",
            "route": "shadow",
            "should_try_bbox_expanded_reocr": bool(quality.get("should_try_bbox_expanded_reocr")),
            "should_try_page_detect": bool(quality.get("should_try_page_detect")),
            "issue_count": len(quality.get("issues") or []),
        }
    )
    page_result["page_quality"] = quality
    page_result["route_history"] = route_history

    if not quality.get("should_try_bbox_expanded_reocr"):
        return page_result
    if not _adaptive_cjk_bbox_reocr_enabled(idioma_origem):
        route_history.append({"stage": "bbox_expanded_reocr", "route": "skipped", "reason": "feature_disabled"})
        return page_result
    if not blocks or not hasattr(ocr, "recognize_batch"):
        route_history.append({"stage": "bbox_expanded_reocr", "route": "skipped", "reason": "ocr_batch_unavailable"})
        return page_result

    ratio = _adaptive_reocr_expansion_ratio(quality)
    expanded_blocks = [_expanded_namespace_block(block, image_rgb.shape, ratio=ratio) for block in blocks]
    crops = []
    height, width = image_rgb.shape[:2]
    for block in expanded_blocks:
        x1, y1, x2, y2 = [int(v) for v in block.xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            crops.append(image_rgb[y1:y2, x1:x2])
        else:
            crops.append(np.zeros((32, 32, 3), dtype=np.uint8))

    _emit_stage_progress(progress_callback, "bbox_expanded_reocr", 0.66, "Re-OCR com bbox expandido")
    try:
        expanded_texts = ocr.recognize_batch(crops)
    except Exception as exc:
        route_history.append(
            {"stage": "bbox_expanded_reocr", "route": "failed", "reason": str(exc)}
        )
        return page_result

    recovery_page = build_page_result(
        image_path=f"{image_label}#bbox-expanded",
        image_rgb=image_rgb,
        blocks=expanded_blocks,
        texts=expanded_texts,
        profile=profile,
        ocr_backend=f"{backend_name}-bbox_expanded",
        enable_font_detection=True,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
        preserve_cjk_sfx=preserve_cjk_sfx,
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    if recovery_page.get("texts"):
        if page_result.get("texts"):
            page_result, _ = _integrate_recovery_page(page_result, recovery_page)
        else:
            page_result = recovery_page
        for text in page_result.get("texts", []) or []:
            text.setdefault("qa_flags", [])
    updated_quality = evaluate_page_quality(
        page_result,
        source_lang=idioma_origem,
        expanded_reocr_attempted=True,
    )
    route_history = list(page_result.get("route_history") or route_history)
    route_history.append(
        {
            "stage": "bbox_expanded_reocr",
            "route": "attempted",
            "expansion_ratio": ratio,
            "recovered_text_count": len(recovery_page.get("texts") or []),
            "remaining_issue_count": len(updated_quality.get("issues") or []),
        }
    )
    if updated_quality.get("should_try_page_detect"):
        route_history.append(
            {
                "stage": "page_detect",
                "route": "candidate" if not _cjk_page_detect_auto_enabled() else "auto_requested",
                "reason": "qa_still_requires_expensive_fallback",
                "auto_allowed": _cjk_page_detect_auto_enabled(),
            }
        )
    page_result["page_quality"] = updated_quality
    page_result["route_history"] = route_history
    return page_result


def _adaptive_reocr_expansion_ratio(quality: dict) -> float:
    issue_types = {str(issue.get("type")) for issue in quality.get("issues") or []}
    if issue_types & {"partial_multiline_ocr", "known_speech_balloon_without_ocr"}:
        return 0.50
    return 0.30


def _expanded_namespace_block(block, image_shape: tuple[int, ...], *, ratio: float) -> SimpleNamespace:
    bbox = _block_xyxy(block) or [0, 0, 0, 0]
    expanded = _expand_bbox(
        bbox,
        image_shape,
        pad_x_ratio=ratio,
        pad_y_ratio=ratio,
        min_pad_x=12,
        min_pad_y=12,
    )
    return SimpleNamespace(
        xyxy=tuple(float(value) for value in expanded),
        confidence=float(getattr(block, "confidence", 1.0) if not isinstance(block, dict) else block.get("confidence", 1.0)),
        mask=getattr(block, "mask", None) if not isinstance(block, dict) else block.get("mask"),
        detector="bbox_expanded_reocr",
        line_polygons=getattr(block, "line_polygons", None) if not isinstance(block, dict) else block.get("line_polygons"),
        source_direction=getattr(block, "source_direction", None) if not isinstance(block, dict) else block.get("source_direction"),
        balloon_bbox=getattr(block, "balloon_bbox", None) if not isinstance(block, dict) else block.get("balloon_bbox"),
        balloon_polygon=getattr(block, "balloon_polygon", None) if not isinstance(block, dict) else block.get("balloon_polygon"),
        balloon_subregions=getattr(block, "balloon_subregions", None) if not isinstance(block, dict) else block.get("balloon_subregions"),
        connected_lobe_bboxes=getattr(block, "connected_lobe_bboxes", None) if not isinstance(block, dict) else block.get("connected_lobe_bboxes"),
        connected_lobe_polygons=getattr(block, "connected_lobe_polygons", None) if not isinstance(block, dict) else block.get("connected_lobe_polygons"),
        rotation_deg=getattr(block, "rotation_deg", None) if not isinstance(block, dict) else block.get("rotation_deg"),
        rotation_source=getattr(block, "rotation_source", None) if not isinstance(block, dict) else block.get("rotation_source"),
    )


def _run_sparse_page_recovery_pass(
    image_rgb: np.ndarray,
    image_label: str,
    ocr,
    profile: str,
    idioma_origem: str,
    progress_callback=None,
    engine_preset: EnginePreset | None = None,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict | None:
    if not hasattr(ocr, "recognize_full_page_lines"):
        return None
    _emit_stage_progress(progress_callback, "recover_text", 0.68, "Recuperando texto em pagina esparsa")
    line_records = ocr.recognize_full_page_lines(image_rgb)
    if not line_records:
        return None

    recovery_blocks = []
    recovery_texts = []
    for record in line_records:
        bbox = record.get("source_bbox") or record.get("bbox") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        bbox = [int(v) for v in bbox]
        recovery_blocks.append(
            SimpleNamespace(
                xyxy=tuple(bbox),
                mask=None,
                confidence=float(record.get("confidence", 0.0) or 0.0),
                detector="full_page_recovery",
                line_polygons=record.get("line_polygons"),
                source_direction=None,
                rotation_deg=record.get("rotation_deg"),
                rotation_source=record.get("rotation_source"),
            )
        )
        recovery_texts.append(dict(record))

    if not recovery_blocks:
        return None

    return build_page_result(
        image_path=image_label,
        image_rgb=image_rgb,
        blocks=recovery_blocks,
        texts=recovery_texts,
        profile=profile,
        ocr_backend=getattr(ocr, "_backend", getattr(ocr, "model_name", "vision")),
        enable_font_detection=True,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
        preserve_cjk_sfx=_preserve_cjk_sfx_for_engine_preset(engine_preset),
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )


def _rotated_text_recovery_enabled() -> bool:
    raw = os.getenv("TRADUZAI_ROTATED_TEXT_RECOVERY", "1")
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _bbox_intersection_fraction(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def _expanded_bbox_for_recovery(bbox: list[int], margin: int = 24) -> list[int]:
    return [
        int(bbox[0]) - margin,
        int(bbox[1]) - margin,
        int(bbox[2]) + margin,
        int(bbox[3]) + margin,
    ]


def _rotated_record_overlaps_existing(record: dict, existing_bboxes: list[list[int]]) -> bool:
    bbox = _coerce_bbox(record.get("source_bbox") or record.get("bbox"))
    if bbox is None:
        return True
    expanded = _expanded_bbox_for_recovery(bbox, margin=12)
    for existing in existing_bboxes:
        if _bbox_intersection_fraction(expanded, _expanded_bbox_for_recovery(existing, margin=20)) >= 0.20:
            return True
    return False


def _rotated_text_needs_recovery(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    rotation = text.get("rotation_deg")
    try:
        rotation_abs = abs(float(rotation or 0.0))
    except (TypeError, ValueError):
        rotation_abs = 0.0
    if rotation_abs < 35.0:
        rotation_abs = abs(float(infer_rotation_deg_from_line_polygons(text.get("line_polygons")) or 0.0))
    if rotation_abs < 35.0:
        return False
    flags = {str(flag).strip().upper() for flag in (text.get("qa_flags") or [])}
    try:
        confidence = float(text.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return "TEXT_CLIPPED" in flags or confidence < 0.62


def _text_len_for_recovery(text: dict) -> int:
    value = text.get("text") or text.get("raw_text") or text.get("texto_original") or ""
    return len(str(value).strip())


def _bbox_area_for_recovery(bbox: list[int] | None) -> int:
    if not bbox:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _best_rotated_overlap_index(record: dict, texts: list[dict]) -> int | None:
    bbox = _coerce_bbox(record.get("source_bbox") or record.get("bbox"))
    if bbox is None:
        return None
    expanded = _expanded_bbox_for_recovery(bbox, margin=16)
    best_index: int | None = None
    best_score = 0.0
    for index, text in enumerate(texts):
        if not isinstance(text, dict):
            continue
        existing = _coerce_bbox(text.get("source_bbox") or text.get("text_pixel_bbox") or text.get("bbox"))
        if existing is None:
            continue
        score = max(
            _bbox_intersection_fraction(expanded, _expanded_bbox_for_recovery(existing, margin=20)),
            _bbox_intersection_fraction(_expanded_bbox_for_recovery(existing, margin=20), expanded),
        )
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score >= 0.18 else None


def _rotated_recovery_record_is_better(record: dict, existing: dict) -> bool:
    if not _rotated_text_needs_recovery(existing):
        return False
    record_len = _text_len_for_recovery(record)
    existing_len = _text_len_for_recovery(existing)
    if record_len >= max(existing_len + 12, int(existing_len * 1.25)):
        return True
    record_polygons = record.get("line_polygons") or []
    existing_polygons = existing.get("line_polygons") or []
    if len(record_polygons) > len(existing_polygons):
        return True
    record_bbox = _coerce_bbox(record.get("source_bbox") or record.get("bbox"))
    existing_bbox = _coerce_bbox(existing.get("source_bbox") or existing.get("text_pixel_bbox") or existing.get("bbox"))
    if _bbox_area_for_recovery(record_bbox) >= int(_bbox_area_for_recovery(existing_bbox) * 1.35):
        return True
    return False


def _merge_rotated_recovery_metadata(target: dict, source: dict, *, text_entry: bool) -> None:
    preserved_id = target.get("id")
    preserved_text_id = target.get("text_id")
    for key in (
        "text",
        "raw_text",
        "normalized_text",
        "bbox",
        "source_bbox",
        "text_pixel_bbox",
        "line_polygons",
        "confidence",
        "rotation_deg",
        "rotation_source",
        "balloon_type",
        "source_direction",
        "detector",
    ):
        value = source.get(key)
        if value not in (None, [], ""):
            target[key] = copy.deepcopy(value)
    if preserved_id not in (None, ""):
        target["id"] = preserved_id
    if preserved_text_id not in (None, ""):
        target["text_id"] = preserved_text_id
    flags = list(target.get("qa_flags") or [])
    for flag in list(source.get("qa_flags") or []) + ["rotated_text_recovery"]:
        if flag not in flags:
            flags.append(flag)
    target["qa_flags"] = flags
    target["allow_broad_bbox_text_search"] = True
    if not text_entry:
        target["detector"] = target.get("detector") or "rotated_full_page_recovery"


def _expand_rotated_recovery_bbox(bbox: list[int], image_shape: tuple[int, ...]) -> list[int]:
    height = int(image_shape[0]) if image_shape else 0
    width = int(image_shape[1]) if len(image_shape) > 1 else 0
    box_w = max(1, int(bbox[2]) - int(bbox[0]))
    box_h = max(1, int(bbox[3]) - int(bbox[1]))
    pad = max(48, min(96, int(max(box_w, box_h) * 0.16)))
    return [
        max(0, int(bbox[0]) - pad),
        max(0, int(bbox[1]) - pad),
        min(width, int(bbox[2]) + pad),
        min(height, int(bbox[3]) + pad),
    ]


def _should_run_rotated_text_recovery(page_result: dict, blocks: list, backend_name: str, ocr) -> bool:
    if not _rotated_text_recovery_enabled():
        return False
    if backend_name != "paddleocr":
        return False
    if not hasattr(ocr, "recognize_rotated_full_page_lines"):
        return False
    if not blocks:
        return False
    accepted = len(page_result.get("texts") or [])
    detected = len(blocks)
    if accepted < detected:
        return True
    return any(_rotated_text_needs_recovery(text) for text in list(page_result.get("texts") or []))


def _append_rotated_recovery_page(base_page: dict, recovered_page: dict) -> tuple[dict, int]:
    updated_page = _clone_page_result(base_page)
    existing_bboxes = [
        _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox"))
        for text in list(updated_page.get("texts") or [])
        if isinstance(text, dict)
    ]
    existing_bboxes = [bbox for bbox in existing_bboxes if bbox is not None]
    appended = 0
    recovered_texts = [text for text in list(recovered_page.get("texts") or []) if isinstance(text, dict)]
    recovered_blocks = [block for block in list(recovered_page.get("_vision_blocks") or []) if isinstance(block, dict)]
    for recovered_text, recovered_block in zip(recovered_texts, recovered_blocks):
        bbox = _coerce_bbox(recovered_text.get("text_pixel_bbox") or recovered_text.get("bbox"))
        if bbox is None:
            continue
        target_index = _best_rotated_overlap_index(recovered_text, list(updated_page.get("texts") or []))
        if target_index is not None:
            existing_texts = list(updated_page.get("texts") or [])
            existing_text = existing_texts[target_index]
            if _rotated_recovery_record_is_better(recovered_text, existing_text):
                _merge_rotated_recovery_metadata(existing_text, recovered_text, text_entry=True)
                existing_blocks = list(updated_page.get("_vision_blocks") or [])
                if target_index < len(existing_blocks) and isinstance(existing_blocks[target_index], dict):
                    _merge_rotated_recovery_metadata(existing_blocks[target_index], recovered_block, text_entry=False)
                    for key in (
                        "rotation_deg",
                        "rotation_source",
                        "qa_flags",
                        "balloon_type",
                        "line_polygons",
                        "text_pixel_bbox",
                        "bbox",
                        "source_bbox",
                        "allow_broad_bbox_text_search",
                    ):
                        value = existing_text.get(key)
                        if value not in (None, [], ""):
                            existing_blocks[target_index][key] = copy.deepcopy(value)
                appended += 1
                existing_bboxes.append(bbox)
            continue
        appended += 1
        new_id = f"rotocr_{len(updated_page.get('texts') or []) + appended:03d}"
        text_entry = dict(recovered_text)
        text_entry["id"] = new_id
        text_entry["text_id"] = new_id
        text_entry["ocr_second_pass"] = True
        flags = list(text_entry.get("qa_flags") or [])
        if "rotated_text_recovery" not in flags:
            flags.append("rotated_text_recovery")
        text_entry["qa_flags"] = flags
        text_entry["allow_broad_bbox_text_search"] = True

        block_entry = dict(recovered_block)
        block_entry["text_id"] = new_id
        block_entry["detector"] = block_entry.get("detector") or "rotated_full_page_recovery"
        block_entry["allow_broad_bbox_text_search"] = True
        for key in (
            "rotation_deg",
            "rotation_source",
            "qa_flags",
            "balloon_type",
            "line_polygons",
            "text_pixel_bbox",
            "bbox",
            "source_bbox",
        ):
            value = text_entry.get(key)
            if value not in (None, [], ""):
                block_entry[key] = copy.deepcopy(value)
        updated_page.setdefault("texts", []).append(text_entry)
        updated_page.setdefault("_vision_blocks", []).append(block_entry)
        existing_bboxes.append(bbox)

    if appended:
        stats = dict(updated_page.get("_ocr_stats") or {})
        stats["rotated_text_recovery_count"] = int(appended)
        updated_page["_ocr_stats"] = stats
    return updated_page, appended


def _run_rotated_text_recovery_pass(
    image_rgb: np.ndarray,
    image_label: str,
    page_result: dict,
    ocr,
    profile: str,
    backend_name: str,
    idioma_origem: str,
    progress_callback=None,
    engine_preset: EnginePreset | None = None,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:
    _emit_stage_progress(progress_callback, "recover_rotated_text", 0.69, "Recuperando texto rotacionado")
    records = ocr.recognize_rotated_full_page_lines(image_rgb)
    if not records:
        return page_result
    existing_bboxes = [
        _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox"))
        for text in list(page_result.get("texts") or [])
        if isinstance(text, dict)
    ]
    existing_bboxes = [bbox for bbox in existing_bboxes if bbox is not None]
    filtered_records = [
        record
        for record in records
        if not _rotated_record_overlaps_existing(record, existing_bboxes)
        or (
            (index := _best_rotated_overlap_index(record, list(page_result.get("texts") or []))) is not None
            and _rotated_recovery_record_is_better(record, list(page_result.get("texts") or [])[index])
        )
    ]
    if not filtered_records:
        return page_result

    recovery_blocks = []
    for record in filtered_records:
        bbox = _coerce_bbox(record.get("source_bbox") or record.get("bbox"))
        if bbox is None:
            continue
        expanded_bbox = _expand_rotated_recovery_bbox(bbox, image_rgb.shape)
        recovery_blocks.append(
            SimpleNamespace(
                xyxy=tuple(expanded_bbox),
                mask=None,
                confidence=float(record.get("confidence", 0.0) or 0.0),
                detector="rotated_full_page_recovery",
                line_polygons=record.get("line_polygons"),
                source_direction=None,
                rotation_deg=record.get("rotation_deg"),
                rotation_source=record.get("rotation_source"),
            )
        )
    if not recovery_blocks:
        return page_result

    recovery_page = build_page_result(
        image_path=image_label,
        image_rgb=image_rgb,
        blocks=recovery_blocks,
        texts=filtered_records,
        profile=profile,
        ocr_backend=f"{backend_name}-rotated",
        enable_font_detection=True,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
        preserve_cjk_sfx=_preserve_cjk_sfx_for_engine_preset(engine_preset),
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    if not recovery_page.get("texts"):
        return page_result
    updated_page, appended = _append_rotated_recovery_page(page_result, recovery_page)
    if appended:
        record_decision(
            stage="ocr",
            action="recover_block",
            reason="rotated_text_recovery",
            page=infer_page_number(image_label),
            details={"recovered_text_count": int(appended)},
        )
    return updated_page


def run_ocr_stage(
    image_rgb: np.ndarray,
    page_dict: dict,
    profile: str = "quality",
    progress_callback=None,
    idioma_origem: str = "en",
    engine_preset_id: str = "",
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:
    """Roda OCR em blocos já detectados (para o pipeline strip-based)."""
    # Converter dicionários de blocos para SimpleNamespace (formato que build_page_result espera)
    engine_preset = _resolve_runtime_engine_preset(engine_preset_id, idioma_origem)

    def _with_engine_preset(result: dict) -> dict:
        return _attach_engine_preset_metadata(result, engine_preset)

    def _band_image_label() -> str:
        raw_number = page_dict.get("_source_page_number", page_dict.get("numero", 0))
        try:
            number = int(raw_number)
        except Exception:
            return f"band_{raw_number}"
        if number > 0:
            return f"band_{number:03d}"
        return f"band_{number}"

    blocks = []
    for b in page_dict.get("_vision_blocks", []):
        blocks.append(
            SimpleNamespace(
                xyxy=tuple(b["bbox"]),
                confidence=float(b.get("confidence", 1.0)),
                mask=b.get("mask"),
                detector=b.get("detector", "strip-detector"),
                line_polygons=b.get("line_polygons"),
                source_direction=b.get("source_direction"),
                balloon_bbox=b.get("balloon_bbox"),
                balloon_polygon=b.get("balloon_polygon"),
                balloon_subregions=b.get("balloon_subregions"),
                connected_lobe_bboxes=b.get("connected_lobe_bboxes"),
                connected_lobe_ids=b.get("connected_lobe_ids"),
                connected_lobe_polygons=b.get("connected_lobe_polygons"),
                bubble_id=b.get("bubble_id"),
                bubble_mask_bbox=b.get("bubble_mask_bbox"),
                bubble_inner_bbox=b.get("bubble_inner_bbox"),
                rotation_deg=b.get("rotation_deg"),
                rotation_source=b.get("rotation_source"),
            )
        )

    height, width = image_rgb.shape[:2]
    raw_source_page_number = page_dict.get("_source_page_number", page_dict.get("numero"))
    try:
        source_page_number = int(raw_source_page_number)
    except Exception:
        source_page_number = None
    quick_text_check_stage = ""
    if (
        blocks
        and _strip_scanlation_credit_skip_enabled()
        and _looks_like_scanlation_credit_band(image_rgb, blocks)
    ):
        return _with_engine_preset({
            "image": _band_image_label(),
            "width": width,
            "height": height,
            "texts": [],
            "_vision_blocks": list(page_dict.get("_vision_blocks", [])),
            "_bubble_regions": list(page_dict.get("_bubble_regions", [])),
            "scanlation_credit_skipped": True,
            "sem_texto_detectado": True,
            "_ocr_stats": {
                "block_count": len(blocks),
                "quick_skipped_no_text": False,
                "scanlation_credit_skipped": True,
                "full_page_mapped": 0,
                "crop_fallback_max": 0,
                "crop_fallback_attempts": 0,
                "crop_fallback_recovered": 0,
            },
        })
    if blocks and _strip_quick_text_skip_enabled():
        has_quick_text, quick_text_check_stage = _quick_text_presence_details(image_rgb)
    if blocks and _strip_quick_text_skip_enabled() and not has_quick_text:
        return _with_engine_preset({
            "image": _band_image_label(),
            "width": width,
            "height": height,
            "texts": [],
            "_vision_blocks": list(page_dict.get("_vision_blocks", [])),
            "_bubble_regions": list(page_dict.get("_bubble_regions", [])),
            "quick_skipped_no_text": True,
            "sem_texto_detectado": True,
            "_ocr_stats": {
                "block_count": len(blocks),
                "quick_skipped_no_text": True,
                "full_page_mapped": 0,
                "crop_fallback_max": 0,
                "crop_fallback_attempts": 0,
                "crop_fallback_recovered": 0,
                "quick_text_check_stage": quick_text_check_stage or "fast_skip",
            },
        })
    if blocks and _looks_like_cover_editorial_band(image_rgb, blocks, source_page_number):
        return _with_engine_preset({
            "image": _band_image_label(),
            "width": width,
            "height": height,
            "texts": [],
            "_vision_blocks": list(page_dict.get("_vision_blocks", [])),
            "_bubble_regions": list(page_dict.get("_bubble_regions", [])),
            "cover_editorial_skipped": True,
            "sem_texto_detectado": True,
            "_ocr_stats": {
                "block_count": len(blocks),
                "quick_skipped_no_text": False,
                "cover_editorial_skipped": True,
                "full_page_mapped": 0,
                "crop_fallback_max": 0,
                "crop_fallback_attempts": 0,
                "crop_fallback_recovered": 0,
            },
        })

    _emit_stage_progress(progress_callback, "load_ocr_engine", 0.10, "Carregando motor de OCR")
    ocr = _get_ocr_engine(profile, lang=idioma_origem)

    orphan_lobe_flag = (
        page_dict.get("_enable_orphan_lobe_scan")
        if "_enable_orphan_lobe_scan" in page_dict
        else os.getenv("TRADUZAI_STRIP_ORPHAN_LOBE_SCAN", "0")
    )
    enable_orphan_lobe_scan = str(orphan_lobe_flag).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enable_orphan_lobe_scan:
        # Full-page OCR still runs this unconditionally.  In strip mode it
        # can trigger many extra crop OCR calls and is kept opt-in.
        blocks = _scan_orphan_lobe_blocks(image_rgb, blocks, ocr)

    white_orphan_flag = (
        page_dict.get("_enable_white_balloon_orphan_scan")
        if "_enable_white_balloon_orphan_scan" in page_dict
        else os.getenv("TRADUZAI_STRIP_WHITE_BALLOON_ORPHAN_SCAN", "1")
    )
    if str(white_orphan_flag).strip().lower() in {"1", "true", "yes", "on"}:
        blocks = _scan_orphan_white_balloon_blocks(image_rgb, blocks)

    recognize_message = f"Reconhecendo {len(blocks)} bloco(s) de texto"
    _emit_stage_progress(progress_callback, "recognize_text", 0.30, recognize_message)

    backend_name = getattr(ocr, "_backend", getattr(ocr, "model_name", "vision"))

    paddle_full_page_flag = os.getenv("TRADUZAI_PADDLE_FULL_PAGE", "1")
    enable_paddle_full_page = str(paddle_full_page_flag).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    if (
        blocks
        and backend_name == "paddleocr"
        and enable_paddle_full_page
        and hasattr(ocr, "recognize_blocks_from_page")
    ):
        allow_sparse_mapping = not bool(page_dict.get("_disable_sparse_ocr_mapping"))
        try:
            texts = ocr.recognize_blocks_from_page(
                image_rgb,
                blocks,
                allow_sparse_mapping=allow_sparse_mapping,
                crop_fallback_max=_strip_paddle_crop_fallback_max(),
                sparse_crop_fallback_max=_strip_paddle_sparse_crop_fallback_max(),
            )
        except TypeError:
            texts = ocr.recognize_blocks_from_page(image_rgb, blocks)
    else:
        # Fallback para crop por crop
        crops = []
        height, width = image_rgb.shape[:2]
        for block in blocks:
            x1, y1, x2, y2 = [int(v) for v in block.xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                crops.append(image_rgb[y1:y2, x1:x2])
            else:
                crops.append(np.zeros((32, 32, 3), dtype=np.uint8))
        texts = ocr.recognize_batch(crops) if crops else []

    page_result = build_page_result(
        image_path=_band_image_label(),
        image_rgb=image_rgb,
        blocks=blocks,
        texts=texts,
        profile=profile,
        ocr_backend=backend_name,
        enable_font_detection=True,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
        preserve_cjk_sfx=_preserve_cjk_sfx_for_engine_preset(engine_preset),
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    ocr_stats = getattr(ocr, "_last_recognize_blocks_stats", None)
    existing_stats = page_result.get("_ocr_stats")
    if isinstance(existing_stats, dict):
        page_result["_ocr_stats"] = dict(existing_stats)
    else:
        page_result["_ocr_stats"] = {}
    if isinstance(ocr_stats, dict):
        page_result["_ocr_stats"].update(ocr_stats)
    batch_cache_stats = getattr(ocr, "_last_batch_cache_stats", None)
    if isinstance(batch_cache_stats, dict):
        page_result["_ocr_stats"].update(batch_cache_stats)
    if quick_text_check_stage:
        page_result["_ocr_stats"]["quick_text_check_stage"] = quick_text_check_stage
    if page_dict.get("_bubble_regions") and not page_result.get("_bubble_regions"):
        page_result["_bubble_regions"] = [dict(item) for item in page_dict.get("_bubble_regions", []) if isinstance(item, dict)]
    if _should_run_rotated_text_recovery(page_result, blocks, backend_name, ocr):
        page_result = _run_rotated_text_recovery_pass(
            image_rgb=image_rgb,
            image_label=_band_image_label(),
            page_result=page_result,
            ocr=ocr,
            profile=profile,
            backend_name=backend_name,
            idioma_origem=idioma_origem,
            progress_callback=progress_callback,
            engine_preset=engine_preset,
            work_title=work_title,
            work_title_aliases=work_title_aliases,
            work_title_user_provided=work_title_user_provided,
        )
    page_result = _apply_adaptive_cjk_reocr(
        image_rgb=image_rgb,
        image_label=_band_image_label(),
        page_result=page_result,
        blocks=blocks,
        ocr=ocr,
        profile=profile,
        backend_name=backend_name,
        idioma_origem=idioma_origem,
        progress_callback=progress_callback,
        preserve_cjk_sfx=_preserve_cjk_sfx_for_engine_preset(engine_preset),
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    page_result = _reconcile_ocr_with_validated_sources(page_result)
    return _with_engine_preset(page_result)


def _build_text_geometry_block_mask(block: dict, height: int, width: int) -> np.ndarray | None:
    bbox = _coerce_bbox(block.get("bbox"))
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    block_mask = np.zeros((height, width), dtype=np.uint8)
    polygons = _normalize_line_polygons(block.get("line_polygons") or [])
    if polygons:
        for polygon in polygons:
            points = np.array(
                [
                    [max(0, min(width - 1, int(px))), max(0, min(height - 1, int(py)))]
                    for px, py in polygon
                ],
                dtype=np.int32,
            )
            if points.shape[0] >= 4:
                cv2.fillPoly(block_mask, [points], 255)
        if np.any(block_mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            block_mask = cv2.dilate(block_mask, kernel, iterations=2)
            clip = np.zeros_like(block_mask)
            cx1, cy1, cx2, cy2 = _expand_bbox(
                [x1, y1, x2, y2],
                (height, width),
                pad_x_ratio=0.03,
                pad_y_ratio=0.08,
                min_pad_x=3,
                min_pad_y=4,
            )
            clip[cy1:cy2, cx1:cx2] = 255
            return cv2.bitwise_and(block_mask, clip)

    text_bbox = _coerce_bbox(block.get("text_pixel_bbox"))
    if text_bbox is None:
        return None
    tx1, ty1, tx2, ty2 = _expand_bbox(
        text_bbox,
        (height, width),
        pad_x_ratio=0.03,
        pad_y_ratio=0.12,
        min_pad_x=4,
        min_pad_y=6,
    )
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    text_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    if text_area >= int(bbox_area * 0.92):
        return None
    try:
        try:
            from inpainter.mask_builder import bbox_to_octagon_mask
        except ImportError:
            from ..inpainter.mask_builder import bbox_to_octagon_mask

        block_mask = bbox_to_octagon_mask(width, height, [tx1, ty1, tx2, ty2])
    except Exception:
        block_mask[ty1:ty2, tx1:tx2] = 255
    return block_mask if np.any(block_mask) else None


def _build_text_geometry_guard_mask(
    block: dict,
    height: int,
    width: int,
    *,
    include_text_bbox: bool = True,
) -> np.ndarray | None:
    if not isinstance(block, dict):
        return None
    guard = np.zeros((height, width), dtype=np.uint8)
    polygons = _normalize_line_polygons(block.get("line_polygons") or [])
    for polygon in polygons:
        points = np.array(
            [
                [max(0, min(width - 1, int(px))), max(0, min(height - 1, int(py)))]
                for px, py in polygon
            ],
            dtype=np.int32,
        )
        if points.shape[0] >= 4:
            cv2.fillPoly(guard, [points], 255)

    text_bbox = _coerce_bbox(block.get("text_pixel_bbox")) if include_text_bbox else None
    if text_bbox is not None:
        x1, y1, x2, y2 = text_bbox
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            guard[y1:y2, x1:x2] = 255

    if not np.any(guard):
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(guard, kernel, iterations=1)


def _protect_dark_line_art_outside_text_geometry(
    mask: np.ndarray,
    vision_blocks: list[dict],
    image_rgb: np.ndarray | None,
) -> np.ndarray:
    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.shape[:2] != mask.shape[:2]:
        return mask
    if not any(isinstance(block, dict) and (block.get("line_polygons") or block.get("text_pixel_bbox")) for block in vision_blocks):
        return mask

    height, width = mask.shape[:2]
    text_guard = np.zeros((height, width), dtype=np.uint8)
    for block in vision_blocks:
        guard = _build_text_geometry_guard_mask(block, height, width)
        if guard is not None:
            text_guard = np.maximum(text_guard, guard)
        qa_metrics = block.get("qa_metrics") if isinstance(block, dict) else None
        allow_tight_reference_guard = isinstance(qa_metrics, dict) and bool(
            qa_metrics.get("tight_reference_geometry_extra_pixels")
        )
        if not allow_tight_reference_guard:
            try:
                try:
                    from inpainter.mask_builder import _balloon_bbox_is_tight_text_anchor
                except ImportError:
                    from ..inpainter.mask_builder import _balloon_bbox_is_tight_text_anchor

                allow_tight_reference_guard = _balloon_bbox_is_tight_text_anchor(block, (height, width))
            except Exception:
                allow_tight_reference_guard = False
        if allow_tight_reference_guard:
            try:
                try:
                    from inpainter.mask_builder import build_inpaint_mask
                except ImportError:
                    from ..inpainter.mask_builder import build_inpaint_mask

                tight_guard = build_inpaint_mask(dict(block), (height, width), image_rgb=image_rgb)
            except Exception:
                tight_guard = None
            if isinstance(tight_guard, np.ndarray) and np.any(tight_guard):
                guard_area = int(np.count_nonzero(guard)) if isinstance(guard, np.ndarray) else 0
                tight_area = int(np.count_nonzero(tight_guard))
                if guard_area <= 0 or tight_area <= max(guard_area + 4096, int(guard_area * 1.45)):
                    text_guard = np.maximum(text_guard, tight_guard.astype(np.uint8))
    if not np.any(text_guard):
        return mask

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb.astype(np.uint8)
    text_halo = cv2.dilate(
        text_guard,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    dark_outside_text = ((gray < 170) & (text_guard == 0)).astype(np.uint8) * 255
    if not np.any(dark_outside_text):
        return mask
    protected = np.zeros_like(dark_outside_text)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_outside_text, connectivity=8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area <= 0:
            continue
        component = labels == label
        outside_halo = bool(np.any(component & (text_halo == 0)))
        tiny_near_text_residual = (
            not outside_halo
            and area <= 18
            and comp_w <= 8
            and comp_h <= 8
        )
        if not tiny_near_text_residual:
            protected[component] = 255
    if not np.any(protected):
        return mask
    protected = cv2.dilate(
        protected,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return cv2.bitwise_and(mask, cv2.bitwise_not(protected))


def vision_blocks_to_mask(
    image_shape: tuple[int, int, int] | tuple[int, int],
    vision_blocks: list[dict],
    image_rgb: np.ndarray | None = None,
    expand_mask: bool = True,
    mask_strategy: str = "",
    ocr_texts: list[dict] | None = None,
    text_segmenter=None,
    bubble_segmenter=None,
) -> np.ndarray:
    if len(image_shape) == 3:
        height, width = image_shape[:2]
    else:
        height, width = image_shape

    vision_blocks = [block for block in list(vision_blocks or []) if not _block_should_skip_inpaint_mask(block)]
    ocr_texts = [text for text in list(ocr_texts or []) if not _block_should_skip_inpaint_mask(text)]
    mask = np.zeros((height, width), dtype=np.uint8)
    if not vision_blocks:
        return mask
    strategy = str(mask_strategy or "").strip().lower()
    cjk_strategies = {
        "segmentation_assisted",
        "roi_segmentation_assisted",
        "ocr_guided_segmentation",
        "ocr_guided_roi_segmentation",
    }
    if strategy in cjk_strategies and isinstance(image_rgb, np.ndarray):
        try:
            try:
                from .cjk_segmentation_mask import (
                    _absorb_dark_text_core,
                    build_manga_segmentation_mask,
                    build_manhwa_manhua_roi_segmentation_mask,
                    expand_cjk_glyph_mask_for_inpaint,
                )
                from .cjk_mask_fusion import fuse_cjk_text_mask
                from .text_mask_evidence import evidence_support_mask, normalize_text_evidence
            except ImportError:
                from vision_stack.cjk_segmentation_mask import (
                    _absorb_dark_text_core,
                    build_manga_segmentation_mask,
                    build_manhwa_manhua_roi_segmentation_mask,
                    expand_cjk_glyph_mask_for_inpaint,
                )
                from vision_stack.cjk_mask_fusion import fuse_cjk_text_mask
                from vision_stack.text_mask_evidence import evidence_support_mask, normalize_text_evidence

            if strategy in {"segmentation_assisted", "ocr_guided_segmentation"}:
                mask = build_manga_segmentation_mask(
                    image_rgb,
                    vision_blocks,
                    None,
                    ocr_texts=ocr_texts,
                    segmenter=text_segmenter,
                    bubble_segmenter=bubble_segmenter,
                )
            else:
                mask = build_manhwa_manhua_roi_segmentation_mask(
                    image_rgb,
                    vision_blocks,
                    ocr_texts,
                    segmenter=text_segmenter,
                    bubble_segmenter=bubble_segmenter,
                )
            if strategy in {"ocr_guided_segmentation", "ocr_guided_roi_segmentation"}:
                evidence = normalize_text_evidence(
                    {"texts": ocr_texts, "_vision_blocks": vision_blocks},
                    width,
                    height,
                )
                mask = fuse_cjk_text_mask(image_rgb, mask, evidence)
            if np.any(mask):
                if expand_mask:
                    mask = expand_cjk_glyph_mask_for_inpaint(mask, max_radius=4, component_ratio=0.10)
                    support = None
                    if strategy in {"ocr_guided_segmentation", "ocr_guided_roi_segmentation"}:
                        evidence = normalize_text_evidence(
                            {"texts": ocr_texts, "_vision_blocks": vision_blocks},
                            width,
                            height,
                        )
                        support = evidence_support_mask(mask.shape[:2], evidence, pad=14)
                        local_mask_support = cv2.dilate(
                            (mask > 0).astype(np.uint8) * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                            iterations=1,
                        )
                        support = cv2.bitwise_or(support, local_mask_support)
                    absorbed = _absorb_dark_text_core(
                        mask,
                        image_rgb,
                        support,
                        aggressive=strategy in {"ocr_guided_segmentation", "ocr_guided_roi_segmentation"},
                    )
                    if absorbed is not None and np.any(absorbed):
                        mask = absorbed
                    # The CJK segmenters are the text geometry for this path.
                    # Running the generic dark-line-art guard here removes orphan SFX cores
                    # because those glyphs often sit outside OCR/detector boxes.
                return mask
        except Exception as exc:
            logger.warning("Mascara CJK preset=%s falhou; usando mascara padrao: %s", strategy, exc)

    def _bbox_fill_mask(bbox_value: list[int]) -> np.ndarray:
        try:
            try:
                from inpainter.mask_builder import bbox_to_octagon_mask
            except ImportError:
                from ..inpainter.mask_builder import bbox_to_octagon_mask

            bbox_mask = bbox_to_octagon_mask(width, height, bbox_value)
        except Exception:
            bx1, by1, bx2, by2 = bbox_value
            bbox_mask = np.zeros((height, width), dtype=np.uint8)
            if bx2 > bx1 and by2 > by1:
                bbox_mask[by1:by2, bx1:bx2] = 255
        return bbox_mask

    def _explicit_geometry_mask_for_block(block: dict, bbox_area: int) -> np.ndarray | None:
        if not (block.get("line_polygons") or block.get("text_pixel_bbox")):
            return None
        try:
            try:
                from inpainter.mask_builder import build_inpaint_mask
            except ImportError:
                from ..inpainter.mask_builder import build_inpaint_mask

            geometry_mask = build_inpaint_mask(dict(block), (height, width), image_rgb=image_rgb)
        except Exception:
            return None
        if geometry_mask is None or not np.any(geometry_mask):
            return None
        geometry_area = int(np.count_nonzero(geometry_mask))
        if geometry_area < max(8, int(bbox_area * 0.006)):
            return None
        return geometry_mask.astype(np.uint8)

    def _local_mask_to_canvas(local_mask: np.ndarray, bbox_value: list[int]) -> np.ndarray | None:
        candidate = local_mask
        if candidate.ndim == 3:
            candidate = candidate[:, :, 0]
        if candidate.shape == (height, width):
            return candidate.astype(np.uint8)
        bx1, by1, bx2, by2 = bbox_value
        target_h = by2 - by1
        target_w = bx2 - bx1
        if target_h <= 0 or target_w <= 0:
            return None
        patch = candidate
        if patch.shape != (target_h, target_w):
            patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[by1:by2, bx1:bx2] = patch.astype(np.uint8)
        return canvas

    def _should_prefer_geometry_mask(local_candidate: np.ndarray, geometry_mask: np.ndarray, bbox_area: int) -> bool:
        local_area = int(np.count_nonzero(local_candidate))
        geometry_area = int(np.count_nonzero(geometry_mask))
        if local_area <= 0 or geometry_area <= 0:
            return False
        extra_area = int(np.count_nonzero((local_candidate > 0) & (geometry_mask == 0)))
        extra_ratio = extra_area / float(max(1, local_area))
        area_limit = max(int(geometry_area * 2.8), geometry_area + max(256, int(bbox_area * 0.06)))
        return local_area >= area_limit and extra_ratio >= 0.45

    def _merge_missing_geometry_into_local_mask(
        block: dict,
        local_candidate: np.ndarray,
        geometry_mask: np.ndarray | None,
    ) -> np.ndarray:
        if geometry_mask is None or not np.any(geometry_mask):
            return local_candidate
        if not _normalize_line_polygons(block.get("line_polygons") or []):
            return local_candidate
        try:
            try:
                from inpainter.mask_builder import _merge_missing_geometry_components
            except ImportError:
                from ..inpainter.mask_builder import _merge_missing_geometry_components
        except Exception:
            return local_candidate
        tight_reference_extra_pixels = 0
        try:
            try:
                from inpainter.mask_builder import _balloon_bbox_is_tight_text_anchor
            except ImportError:
                from ..inpainter.mask_builder import _balloon_bbox_is_tight_text_anchor

            if _balloon_bbox_is_tight_text_anchor(block, (height, width)):
                local_area = int(np.count_nonzero(local_candidate))
                geometry_area = int(np.count_nonzero(geometry_mask))
                extra_pixels = int(np.count_nonzero((geometry_mask > 0) & (local_candidate == 0)))
                if (
                    extra_pixels > 0
                    and geometry_area <= max(local_area + 4096, int(local_area * 1.35))
                    and extra_pixels <= max(4096, int(local_area * 0.45))
                ):
                    tight_reference_extra_pixels = extra_pixels
        except Exception:
            tight_reference_extra_pixels = 0
        try:
            merged, added_components = _merge_missing_geometry_components(
                local_candidate.astype(np.uint8),
                geometry_mask.astype(np.uint8),
            )
        except Exception:
            return local_candidate
        if added_components:
            qa_metrics = block.setdefault("qa_metrics", {})
            if isinstance(qa_metrics, dict):
                qa_metrics["local_mask_missing_geometry_components"] = added_components
        if tight_reference_extra_pixels > 0:
            qa_metrics = block.setdefault("qa_metrics", {})
            if isinstance(qa_metrics, dict):
                qa_metrics["tight_reference_geometry_extra_pixels"] = int(tight_reference_extra_pixels)
            remaining_extra = int(np.count_nonzero((geometry_mask > 0) & (merged == 0)))
            if remaining_extra > 0:
                merged = np.maximum(merged.astype(np.uint8), geometry_mask.astype(np.uint8))
        return merged.astype(np.uint8)

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
            local_candidate = _local_mask_to_canvas(local_mask, [x1, y1, x2, y2])
            if local_candidate is None:
                continue
            bbox_area = max(1, (x2 - x1) * (y2 - y1))
            geometry_mask = _explicit_geometry_mask_for_block(block, bbox_area)
            if geometry_mask is not None and _should_prefer_geometry_mask(local_candidate, geometry_mask, bbox_area):
                mask = np.maximum(mask, geometry_mask)
            else:
                local_candidate = _merge_missing_geometry_into_local_mask(block, local_candidate, geometry_mask)
                mask = np.maximum(mask, local_candidate)
        else:
            applied_refined = False
            has_explicit_text_geometry = bool(block.get("line_polygons") or block.get("text_pixel_bbox"))
            raw_text_mask_rejected = False
            try:
                try:
                    from inpainter.mask_builder import build_inpaint_mask, raw_text_evidence_rejected
                except ImportError:
                    from ..inpainter.mask_builder import build_inpaint_mask, raw_text_evidence_rejected

                geometry_mask = build_inpaint_mask(block, (height, width), image_rgb=image_rgb)
                raw_text_mask_rejected = raw_text_evidence_rejected(block)
            except Exception:
                geometry_mask = None
            if raw_text_mask_rejected:
                continue
            if geometry_mask is not None and np.any(geometry_mask):
                bbox_area = max(1, (x2 - x1) * (y2 - y1))
                geometry_area = int(np.count_nonzero(geometry_mask))
                allow_geometry_mask = has_explicit_text_geometry or image_rgb is None
                geometry_area_ok = geometry_area >= max(8, int(bbox_area * 0.006))
                if not has_explicit_text_geometry:
                    geometry_area_ok = geometry_area_ok and geometry_area <= int(bbox_area * 0.38)
                if allow_geometry_mask and geometry_area_ok:
                    mask = np.maximum(mask, geometry_mask.astype(np.uint8))
                    applied_refined = True
                    if has_explicit_text_geometry:
                        continue
            is_white_balloon = False
            if image_rgb is not None:
                is_white_balloon = _is_white_balloon_region(image_rgb, [x1, y1, x2, y2])
                balloon_mask = None
                if not is_white_balloon:
                    geometry_mask = _build_text_geometry_block_mask(block, height, width)
                    if geometry_mask is not None and np.any(geometry_mask):
                        mask = np.maximum(mask, geometry_mask.astype(np.uint8))
                        applied_refined = True
                if is_white_balloon:
                    text_boxes = _extract_white_balloon_text_boxes(image_rgb, [x1, y1, x2, y2])
                    bbox_area = max(1, (x2 - x1) * (y2 - y1))
                    text_box_area = sum(
                        max(0, int(bx2) - int(bx1)) * max(0, int(by2) - int(by1))
                        for bx1, by1, bx2, by2 in text_boxes
                    )
                    text_union = None
                    for bx1, by1, bx2, by2 in text_boxes:
                        candidate = [int(bx1), int(by1), int(bx2), int(by2)]
                        text_union = candidate if text_union is None else _bbox_union(text_union, candidate)
                    union_width_ratio = 0.0
                    union_height_ratio = 0.0
                    if text_union is not None:
                        union_width_ratio = max(0.0, min(1.0, (text_union[2] - text_union[0]) / float(max(1, x2 - x1))))
                        union_height_ratio = max(0.0, min(1.0, (text_union[3] - text_union[1]) / float(max(1, y2 - y1))))
                    exact_boxes_are_representative = (
                        text_boxes
                        and text_box_area >= max(64, int(bbox_area * 0.12))
                        and union_width_ratio >= 0.38
                        and union_height_ratio >= 0.52
                    )
                    if exact_boxes_are_representative:
                        for bx1, by1, bx2, by2 in text_boxes:
                            bx1 = max(0, min(width, int(bx1)))
                            bx2 = max(0, min(width, int(bx2)))
                            by1 = max(0, min(height, int(by1)))
                            by2 = max(0, min(height, int(by2)))
                            if bx2 > bx1 and by2 > by1:
                                mask = np.maximum(mask, _bbox_fill_mask([bx1, by1, bx2, by2]))
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
                                        # Expansão mais agressiva para cobrir glows e sombras de texto (melhora inpaint)
                                        patch = cv2.dilate(
                                            patch,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                                            iterations=2,
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

            elif not is_white_balloon:
                geometry_mask = _build_text_geometry_block_mask(block, height, width)
                if geometry_mask is not None and np.any(geometry_mask):
                    mask = np.maximum(mask, geometry_mask.astype(np.uint8))
                    applied_refined = True

            if not applied_refined:
                mask = np.maximum(mask, _bbox_fill_mask([x1, y1, x2, y2]))

    if expand_mask and np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = _protect_dark_line_art_outside_text_geometry(mask, vision_blocks, image_rgb)
    return mask


def run_detect_ocr(
    image_path: str,
    models_dir: str = "",
    profile: str = "quality",
    vision_worker_path: str = "",
    progress_callback=None,
    idioma_origem: str = "en",
    engine_preset_id: str = "",
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:

    engine_preset = _resolve_runtime_engine_preset(engine_preset_id, idioma_origem)
    engine_steps = _runtime_engine_steps(engine_preset)

    _configure_model_roots(models_dir)
    _emit_stage_progress(progress_callback, "prepare_image", 0.03, "Preparando imagem para OCR")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        _emit_stage_progress(progress_callback, "complete", 1.0, "Imagem nao encontrada")
        return _attach_engine_preset_metadata(
            {"image": image_path, "width": 0, "height": 0, "texts": [], "_vision_blocks": []},
            engine_preset,
            engine_steps,
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    use_koharu_worker = bool(str(vision_worker_path or "").strip())
    use_koharu_cjk_http = (
        (not use_koharu_worker)
        and engine_preset.detector != "anime-text-yolo-n"
        and _should_use_koharu_cjk_ocr(idioma_origem, models_dir)
    )
    if use_koharu_worker:
        try:
            page_result = _run_koharu_worker_detect_ocr(
                image_rgb=image_rgb,
                image_label=image_path,
                vision_worker_path=vision_worker_path,
                models_dir=models_dir,
                profile=profile,
                progress_callback=progress_callback,
                idioma_origem=idioma_origem,
                engine_preset_id=engine_preset.id,
                work_title=work_title,
                work_title_aliases=work_title_aliases,
                work_title_user_provided=work_title_user_provided,
            )
        except Exception as exc:
            logger.warning("Koharu vision worker falhou em %s; fallback para stack atual: %s", image_path, exc)
            page_result = _run_detect_ocr_on_image(
                image_rgb,
                image_path,
                profile=profile,
                progress_callback=progress_callback,
                idioma_origem=idioma_origem,
                engine_preset=engine_preset,
                work_title=work_title,
                work_title_aliases=work_title_aliases,
                work_title_user_provided=work_title_user_provided,
            )
    elif use_koharu_cjk_http:
        try:
            page_result = _run_koharu_cjk_http_detect_ocr(
                image_rgb=image_rgb,
                image_label=image_path,
                models_dir=models_dir,
                profile=profile,
                progress_callback=progress_callback,
                idioma_origem=idioma_origem,
                engine_preset_id=engine_preset.id,
                work_title=work_title,
                work_title_aliases=work_title_aliases,
                work_title_user_provided=work_title_user_provided,
            )
        except Exception as exc:
            logger.warning("Koharu HTTP OCR CJK falhou em %s; fallback para stack atual: %s", image_path, exc)
            if not _quick_text_presence_check(image_rgb):
                _emit_stage_progress(progress_callback, "complete", 1.0, "Pagina sem texto detectavel; OCR pulado")
                height, width = image_rgb.shape[:2]
                return _attach_engine_preset_metadata({
                    "image": image_path,
                    "width": width,
                    "height": height,
                    "texts": [],
                    "_vision_blocks": [],
                    "quick_skipped_no_text": True,
                    "sem_texto_detectado": True,
                    "koharu_cjk_fallback": "quick_skip",
                }, engine_preset, engine_steps)
            page_result = _run_detect_ocr_on_image(
                image_rgb,
                image_path,
                profile=profile,
                progress_callback=progress_callback,
                idioma_origem=idioma_origem,
                engine_preset=engine_preset,
                work_title=work_title,
                work_title_aliases=work_title_aliases,
                work_title_user_provided=work_title_user_provided,
            )
    else:
        if not _quick_text_presence_check(image_rgb):
            _emit_stage_progress(progress_callback, "complete", 1.0, "Pagina sem texto detectavel; OCR pulado")
            height, width = image_rgb.shape[:2]
            return _attach_engine_preset_metadata({
                "image": image_path,
                "width": width,
                "height": height,
                "texts": [],
                "_vision_blocks": [],
                "quick_skipped_no_text": True,
                "sem_texto_detectado": True,
            }, engine_preset, engine_steps)
        page_result = _run_detect_ocr_on_image(
            image_rgb,
            image_path,
            profile=profile,
            progress_callback=progress_callback,
            idioma_origem=idioma_origem,
            engine_preset=engine_preset,
            work_title=work_title,
            work_title_aliases=work_title_aliases,
            work_title_user_provided=work_title_user_provided,
        )
    recovered_page = _run_orientation_recovery(
        image_rgb=image_rgb,
        image_label=image_path,
        baseline_page=page_result,
        profile=profile,
        progress_callback=progress_callback,
        idioma_origem=idioma_origem,
        engine_preset=engine_preset,
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    if recovered_page is not None:
        logger.info(
            "Orientation recovery aplicado em %s: %s graus",
            image_path,
            recovered_page.get("orientation_recovery_deg"),
        )
        page_result = recovered_page
    # Cache image for downstream use (layout enrichment) to avoid re-reading from disk
    page_result["_cached_image_bgr"] = image_bgr
    try:
        try:
            from .oar_ocr_adapter import load_oar_ocr_regions
        except ImportError:
            from vision_stack.oar_ocr_adapter import load_oar_ocr_regions

        height, width = image_rgb.shape[:2]
        oar_regions = load_oar_ocr_regions(image_path, width=width, height=height)
        if oar_regions:
            page_result["_oar_ocr_regions"] = oar_regions
    except Exception as exc:
        logger.warning("oar-ocr auxiliar falhou em %s: %s", image_path, exc)
    _attach_engine_preset_metadata(page_result, engine_preset, engine_steps)

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

    outputs: list[Path] = []
    total = len(image_files)

    if not image_files:
        return outputs

    with ThreadPoolExecutor(max_workers=2) as io_pool:
        load_future = io_pool.submit(_load_image_rgb, image_files[0])
        pending_save = None

        for index, (img_path, ocr_data) in enumerate(zip(image_files, ocr_results), start=1):
            if ocr_data is None:
                ocr_data = {}
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
                    inpainter_model = _inpainter_model_for_page(ocr_data)
                    ocr_data["_inpaint_engine"] = inpainter_model
                    inpainter = _get_inpainter(profile, model=inpainter_model)
                    if _koharu_blockwise_inpaint_enabled():
                        cleaned = _run_koharu_blockwise_inpaint_page(image_np, ocr_data, inpainter)
                    else:
                        cleaned = _apply_inpainting_round(image_np, ocr_data, inpainter)
                except Exception as exc:
                    logger.warning(
                        "Inpaint full-page falhou em %s; sem fallback silencioso: %s",
                        img_path,
                        exc,
                    )
                    raise

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
