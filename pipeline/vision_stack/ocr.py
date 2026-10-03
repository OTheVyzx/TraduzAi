"""
OCR Engine — Reconhecimento de texto em manga
Suporta manga-ocr (japonês/inglês) e PaddleOCR (multilingual)
Batching para máxima performance na GPU
"""

import copy
from dataclasses import replace
import logging
import math
import os
import hashlib
import re
import threading
from collections import OrderedDict
from difflib import SequenceMatcher
from typing import Callable, Optional, Union
import cv2
import numpy as np
import torch
from PIL import Image

from ownership.hash_contract import canonical_page_sha256, canonical_json_sha256, sha256_text
from ownership.ocr_contract import (
    OCRAttempt,
    OCRBlock,
    OCRDiagnostics,
    OCRInputPixelIdentityError,
    OCRInvocationResult,
    OCRObservationRecord,
    OCRRequest,
    OCRTransformOperation,
    OCRTransformSpec,
    normalize_ocr_payload_text,
)

logger = logging.getLogger(__name__)


def _normalize_ocr_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TypeError("OCR provider input must use uint8 samples")
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("OCR provider input must be HxW, HxWx1, HxWx3 or HxWx4")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("OCR provider input cannot be empty")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _iter_provider_text_lines(raw_result):
    if isinstance(raw_result, str):
        yield raw_result, 0.0, ()
        return
    payload = raw_result
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], list):
        payload = payload[0]
    if not isinstance(payload, (list, tuple)):
        return
    for item in payload:
        if isinstance(item, str):
            yield item, 0.0, ()
            continue
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        polygon_raw, metadata = item[0], item[1]
        if not isinstance(metadata, (list, tuple)) or not metadata:
            continue
        text = str(metadata[0] or "")
        try:
            confidence = float(metadata[1]) if len(metadata) >= 2 else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        polygon: list[tuple[int, int]] = []
        if isinstance(polygon_raw, (list, tuple)):
            for point in polygon_raw:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    polygon.append((int(round(float(point[0]))), int(round(float(point[1])))))
        yield text, max(0.0, min(1.0, confidence)), tuple(polygon)


def _legacy_local_ocr_request(
    pixels: np.ndarray,
    *,
    variant_id: str,
    provider_family: str,
) -> OCRRequest:
    pixel_sha256 = canonical_page_sha256(_normalize_ocr_rgb(pixels))
    return OCRRequest(
        run_id=f"legacy-local-{pixel_sha256[:20]}",
        origin_execution_id=f"legacy-execution-{pixel_sha256[:20]}",
        page_id=f"legacy-page-{pixel_sha256[:20]}",
        page_source_sha256=pixel_sha256,
        root_input_pixel_sha256=pixel_sha256,
        invocation_id=f"legacy:{variant_id}:{pixel_sha256[:20]}",
        provider_family=provider_family,
    )


def _identity_transform_spec() -> OCRTransformSpec:
    return OCRTransformSpec.build((OCRTransformOperation(kind="identity"),))


class _ProviderRecordBatch(tuple):
    """Transient records plus raw provider output for legacy parsing only."""

    def __new__(cls, records, raw_result):
        instance = super().__new__(cls, records)
        instance.raw_result = raw_result
        return instance


def execute_hash_bound_provider_attempt(
    *,
    request: OCRRequest,
    root_input_rgb: np.ndarray,
    actual_input_rgb: np.ndarray,
    transform_spec: OCRTransformSpec,
    variant_id: str,
    provider,
    expected_input_pixel_sha256: str | None = None,
    parent_input_pixel_sha256: str | None = None,
    input_bbox_page: tuple[int, int, int, int] | None = None,
    input_kind: str | None = None,
    provider_kwargs: dict | None = None,
    provider_mode: str = "callable",
    attempt_ordinal: int = 1,
    source: str | None = None,
    execution_recorder: Callable[[object], None] | None = None,
) -> tuple[OCRAttempt, tuple[OCRObservationRecord, ...]]:
    """Execute one physical OCR attempt after verifying its pixel identity."""

    root_rgb = _normalize_ocr_rgb(root_input_rgb)
    physical_rgb = _normalize_ocr_rgb(actual_input_rgb)
    root_hash = canonical_page_sha256(root_rgb)
    if root_hash != request.root_input_pixel_sha256:
        raise OCRInputPixelIdentityError("OCR request root hash does not match root_input_rgb")
    replayed = transform_spec.replay(root_rgb)
    if not np.array_equal(replayed, physical_rgb):
        raise OCRInputPixelIdentityError("OCR transform replay differs from physical provider input")
    input_hash = canonical_page_sha256(physical_rgb)
    if expected_input_pixel_sha256 is not None and input_hash != expected_input_pixel_sha256:
        raise OCRInputPixelIdentityError("declared OCR input hash differs from physical provider input")
    parent_hash = parent_input_pixel_sha256 or root_hash
    attempt_seed = canonical_json_sha256(
        {
            "attempt_ordinal": int(attempt_ordinal),
            "input_pixel_sha256": input_hash,
            "request_identity": list(request.identity),
            "transform_spec_sha256": transform_spec.sha256,
            "variant_id": str(variant_id),
        }
    )
    attempt_id = f"ocr_attempt_{attempt_seed[:20]}"

    if provider_mode == "manga_ocr":
        processor = provider["processor"]
        model = provider["model"]
        tokenizer = provider["tokenizer"]
        pixel_values = processor(
            images=[Image.fromarray(physical_rgb, "RGB")],
            return_tensors="pt",
        ).pixel_values.to(provider["device"])
        if provider.get("half"):
            pixel_values = pixel_values.half()
        with torch.inference_mode():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=300,
                num_beams=1,
                do_sample=False,
            )
        raw_result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    elif provider_mode == "callable":
        raw_result = provider(physical_rgb, **dict(provider_kwargs or {}))
    else:
        raise ValueError(f"unsupported OCR provider mode: {provider_mode}")
    if execution_recorder is not None:
        execution_recorder(raw_result)
    attempt = OCRAttempt(
        attempt_id=attempt_id,
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        root_input_pixel_sha256=request.root_input_pixel_sha256,
        invocation_id=request.invocation_id,
        provider_family=request.provider_family,
        variant_id=str(variant_id),
        input_pixel_sha256=input_hash,
        parent_input_pixel_sha256=parent_hash,
        input_bbox_page=input_bbox_page,
        input_kind=str(input_kind or variant_id),
        transform_spec=transform_spec,
        input_width=int(physical_rgb.shape[1]),
        input_height=int(physical_rgb.shape[0]),
        input_mode="RGB",
        provider_called=True,
        cache_hit=False,
    )
    records: list[OCRObservationRecord] = []
    for index, (text, confidence, polygon) in enumerate(_iter_provider_text_lines(raw_result), 1):
        if polygon:
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        elif input_bbox_page is not None:
            bbox = input_bbox_page
        else:
            bbox = (0, 0, int(physical_rgb.shape[1]), int(physical_rgb.shape[0]))
        normalized_text = normalize_ocr_payload_text(text)
        observation_seed = sha256_text(f"{attempt_id}|{index}|{normalized_text}|{bbox}")
        records.append(
            OCRObservationRecord(
                observation_id=f"ocr_observation_{observation_seed[:20]}",
                attempt_id=attempt_id,
                run_id=request.run_id,
                origin_execution_id=request.origin_execution_id,
                page_id=request.page_id,
                page_source_sha256=request.page_source_sha256,
                root_input_pixel_sha256=request.root_input_pixel_sha256,
                input_pixel_sha256=input_hash,
                invocation_id=request.invocation_id,
                provider_family=request.provider_family,
                variant_id=str(variant_id),
                payload_sha256=sha256_text(normalized_text),
                text=normalized_text,
                confidence=confidence,
                bbox_page=bbox,
                polygon_page=polygon,
                source=str(source or request.provider_family),
            )
        )
    return attempt, _ProviderRecordBatch(records, raw_result)


def _records_in_page_space(
    records: tuple[OCRObservationRecord, ...],
    *,
    bbox_page: tuple[int, int, int, int],
    input_width: int,
    input_height: int,
) -> tuple[OCRObservationRecord, ...]:
    """Project provider-local geometry back into the authoritative page."""

    x1, y1, x2, y2 = bbox_page
    scale_x = (x2 - x1) / float(max(1, input_width))
    scale_y = (y2 - y1) / float(max(1, input_height))

    def point(value: tuple[int, int]) -> tuple[int, int]:
        return (
            int(round(x1 + value[0] * scale_x)),
            int(round(y1 + value[1] * scale_y)),
        )

    projected: list[OCRObservationRecord] = []
    for record in records:
        polygon = tuple(point(value) for value in record.polygon_page)
        if polygon:
            xs = [value[0] for value in polygon]
            ys = [value[1] for value in polygon]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            bbox = bbox_page
        projected.append(replace(record, bbox_page=bbox, polygon_page=polygon))
    return tuple(projected)


class OcrBackendUnavailable(RuntimeError):
    """Raised when the required OCR backend cannot be initialized."""


try:
    from ocr.postprocess import normalize_rotated_text_metadata
except ImportError:  # pragma: no cover - supports package imports
    from ..ocr.postprocess import normalize_rotated_text_metadata

PADDLE_DIRECT_LANGUAGE_CODES = {"ch", "en", "korean", "japan", "chinese_cht", "ta", "te", "ka"}
PADDLE_LATIN_LANGUAGE_CODES = {
    "af", "az", "bs", "cs", "cy", "da", "de", "es", "et", "fr", "ga", "hr", "hu",
    "id", "is", "it", "ku", "la", "lt", "lv", "mi", "ms", "mt", "nl", "no", "oc",
    "pi", "pl", "pt", "ro", "rs_latin", "sk", "sl", "sq", "sv", "sw", "tl", "tr",
    "uz", "vi", "french", "german",
}
PADDLE_ARABIC_LANGUAGE_CODES = {"ar", "fa", "ug", "ur"}
PADDLE_CYRILLIC_LANGUAGE_CODES = {
    "ru", "rs_cyrillic", "be", "bg", "uk", "mn", "abq", "ady", "kbd", "ava", "dar",
    "inh", "che", "lbe", "lez", "tab",
}
PADDLE_DEVANAGARI_LANGUAGE_CODES = {
    "hi", "mr", "ne", "bh", "mai", "ang", "bho", "mah", "sck", "new", "gom", "sa", "bgc",
}
SKEWED_TEXT_MIN_ROTATION_DEG = 12.0
SKEWED_TEXT_MAX_DESKEW_DEG = 44.0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _ocr_fallback_shadow_enabled() -> bool:
    return _env_bool("TRADUZAI_OCR_FALLBACK_SHADOW", False)


def _ocr_fallback_shadow_limit(default: int = 1) -> int:
    return max(0, _env_int("TRADUZAI_OCR_FALLBACK_SHADOW_MAX", default))


def _ocr_assignment_audit_enabled() -> bool:
    return _env_bool("TRADUZAI_FLAG_OCR_ASSIGNMENT_AUDIT_V2", False)


def _bbox_union(bboxes: list[list[int]]) -> list[int] | None:
    if not bboxes:
        return None
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def _bbox_area(bbox: list[int] | None) -> int:
    if not bbox or len(bbox) != 4:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _build_ocr_assignment_audit(block_bbox: list[int], lines: list[dict]) -> dict:
    assigned_lines = []
    line_bboxes = []
    for entry in lines:
        bbox = entry.get("line_bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        normalized_bbox = [int(value) for value in bbox]
        line_bboxes.append(normalized_bbox)
        assigned_lines.append(
            {
                "line_bbox": normalized_bbox,
                "text": str(entry.get("text") or ""),
                "confidence": float(entry.get("confidence") or 0.0),
                "assignment_score": round(float(entry.get("assignment_score") or 0.0), 6),
            }
        )

    line_heights = [max(1, bbox[3] - bbox[1]) for bbox in line_bboxes]
    vertical_gaps = [
        max(0, line_bboxes[index + 1][1] - line_bboxes[index][3])
        for index in range(max(0, len(line_bboxes) - 1))
    ]
    median_height = float(np.median(line_heights)) if line_heights else 0.0
    max_gap = max(vertical_gaps, default=0)
    max_gap_ratio = max_gap / max(1.0, median_height)
    union_bbox = _bbox_union(line_bboxes)
    block_area = _bbox_area(block_bbox)
    union_area = _bbox_area(union_bbox)
    union_to_block_area_ratio = union_area / max(1, block_area)

    suspicious = False
    reason = "spatially_coherent"
    if len(line_bboxes) >= 2 and max_gap_ratio >= 4.0:
        suspicious = True
        reason = "large_vertical_gap"
    elif union_to_block_area_ratio > 1.15:
        suspicious = True
        reason = "line_union_exceeds_detector_block"

    return {
        "assignment_mode": "full_page_lines",
        "block_bbox": [int(value) for value in block_bbox],
        "assigned_line_count": len(assigned_lines),
        "assigned_lines": assigned_lines,
        "line_union_bbox": union_bbox,
        "median_line_height": round(median_height, 3),
        "vertical_gaps": vertical_gaps,
        "max_vertical_gap": max_gap,
        "max_vertical_gap_over_median_height": round(max_gap_ratio, 3),
        "line_union_to_block_area_ratio": round(union_to_block_area_ratio, 3),
        "suspicious": suspicious,
        "reason": reason,
    }


def _raw_ocr_output_is_empty(value) -> bool:
    if value is None:
        return True
    if torch.is_tensor(value):
        return value.numel() == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (str, bytes)):
        return len(value) == 0
    if isinstance(value, dict):
        return not value or all(_raw_ocr_output_is_empty(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return not value or all(_raw_ocr_output_is_empty(item) for item in value)
    return False


def _attach_rotated_text_metadata(record: dict) -> dict:
    return normalize_rotated_text_metadata(record)


def _experimental_gpu_ocr_preprocess_enabled() -> bool:
    return _env_bool("TRADUZAI_EXPERIMENTAL_GPU_OCR_PREPROCESS", False)


def _gpu_image_ops_backend() -> str:
    return os.getenv("TRADUZAI_GPU_IMAGE_OPS_BACKEND", "auto").strip() or "auto"


def _resize_for_ocr_preprocess(
    image: np.ndarray,
    size: tuple[int, int],
    *,
    interpolation: int,
) -> np.ndarray:
    if _experimental_gpu_ocr_preprocess_enabled():
        try:
            from vision_stack.gpu_image_ops import resize_crops_batch

            return resize_crops_batch(
                [image],
                size=size,
                backend=_gpu_image_ops_backend(),
                interpolation=interpolation,
            )[0]
        except Exception:
            pass
    return cv2.resize(image, size, interpolation=interpolation)


PADDLE_LANGUAGE_ALIASES = {
    "en-gb": "en",
    "en-us": "en",
    "pt-br": "pt",
    "pt-pt": "pt",
    "zh": "ch",
    "zh-cn": "ch",
    "zh-hans": "ch",
    "zh-tw": "chinese_cht",
    "zh-hant": "chinese_cht",
    "ja": "japan",
    "ko": "korean",
}

EASYOCR_LANGUAGE_ALIASES = {
    "en-gb": ["en"],
    "en-us": ["en"],
    "pt-br": ["pt", "en"],
    "pt-pt": ["pt", "en"],
    "zh": ["ch_sim", "en"],
    "zh-cn": ["ch_sim", "en"],
    "zh-hans": ["ch_sim", "en"],
    "zh-tw": ["ch_tra", "en"],
    "zh-hant": ["ch_tra", "en"],
    "ja": ["ja", "en"],
    "ko": ["ko", "en"],
    "ru": ["ru", "en"],
    "ar": ["ar", "en"],
}


def normalize_paddleocr_language(lang: str) -> str:
    normalized = (lang or "en").strip().replace("_", "-").lower()
    if normalized in PADDLE_LANGUAGE_ALIASES:
        return PADDLE_LANGUAGE_ALIASES[normalized]

    base = normalized.split("-", 1)[0]
    if base in PADDLE_LANGUAGE_ALIASES:
        return PADDLE_LANGUAGE_ALIASES[base]
    if base in PADDLE_DIRECT_LANGUAGE_CODES:
        return base
    if base in PADDLE_LATIN_LANGUAGE_CODES:
        return base
    if base in PADDLE_ARABIC_LANGUAGE_CODES:
        return base
    if base in PADDLE_CYRILLIC_LANGUAGE_CODES:
        return base
    if base in PADDLE_DEVANAGARI_LANGUAGE_CODES:
        return base

    return "latin"


def normalize_easyocr_languages(lang: str) -> list[str]:
    normalized = (lang or "en").strip().replace("_", "-").lower()
    if normalized in EASYOCR_LANGUAGE_ALIASES:
        return EASYOCR_LANGUAGE_ALIASES[normalized]

    base = normalized.split("-", 1)[0]
    if base in EASYOCR_LANGUAGE_ALIASES:
        return EASYOCR_LANGUAGE_ALIASES[base]
    if base in {"es", "de", "fr", "it", "pt", "nl"}:
        return [base, "en"]
    if base == "en":
        return ["en"]

    return ["en"]


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


def _bbox_from_polygons(line_polygons) -> list[int] | None:
    points: list[tuple[float, float]] = []
    for polygon in line_polygons or []:
        if not isinstance(polygon, (list, tuple)):
            continue
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                points.append((float(point[0]), float(point[1])))
            except Exception:
                continue
    if not points:
        return None
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    return [
        int(np.floor(min(xs))),
        int(np.floor(min(ys))),
        int(np.ceil(max(xs))),
        int(np.ceil(max(ys))),
    ]


def normalize_rotation_deg(value) -> float:
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


def _normalize_source_edge_angle(dx: float, dy: float) -> float:
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return 0.0
    angle = math.degrees(math.atan2(dy, dx))
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    if abs(angle) < 0.01:
        return 0.0
    return angle


def infer_rotation_deg_from_line_polygons(line_polygons) -> float:
    weighted_angles: list[tuple[float, float]] = []
    for polygon in line_polygons or []:
        points: list[tuple[float, float]] = []
        for point in polygon or []:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                points.append((float(point[0]), float(point[1])))
            except Exception:
                continue
        if len(points) < 4:
            continue

        edges: list[tuple[float, float, float]] = []
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length >= 8.0:
                edges.append((length, dx, dy))
        if not edges:
            continue

        length, dx, dy = max(edges, key=lambda item: item[0])
        angle = _normalize_source_edge_angle(dx, dy)
        if abs(angle) < SKEWED_TEXT_MIN_ROTATION_DEG:
            continue
        weighted_angles.append((angle, length))

    if not weighted_angles:
        return 0.0
    total_weight = sum(weight for _angle, weight in weighted_angles)
    if total_weight <= 0.0:
        return 0.0
    average = sum(angle * weight for angle, weight in weighted_angles) / total_weight
    if abs(average) < SKEWED_TEXT_MIN_ROTATION_DEG:
        return 0.0
    if abs(abs(average) - 90.0) <= 5.0:
        return 90.0 if average >= 0.0 else -90.0
    return normalize_rotation_deg(average)


def _select_text_mask(gray_crop: np.ndarray) -> np.ndarray:
    if gray_crop.size == 0:
        return np.zeros(gray_crop.shape, dtype=np.uint8)

    _, mask_normal = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_inverted = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = mask_normal if int(np.count_nonzero(mask_normal)) <= int(np.count_nonzero(mask_inverted)) else mask_inverted
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return mask


def _derive_text_pixel_bbox(
    page_rgb: np.ndarray,
    seed_bbox,
    line_polygons: list[list[list[int]]] | None = None,
) -> list[int] | None:
    bbox = _coerce_bbox(seed_bbox)
    polygon_bbox = _bbox_from_polygons(line_polygons or [])
    if polygon_bbox is not None:
        if bbox is None:
            bbox = polygon_bbox
        else:
            bbox = [
                min(bbox[0], polygon_bbox[0]),
                min(bbox[1], polygon_bbox[1]),
                max(bbox[2], polygon_bbox[2]),
                max(bbox[3], polygon_bbox[3]),
            ]
    if bbox is None or not isinstance(page_rgb, np.ndarray) or page_rgb.size == 0:
        return bbox

    height, width = page_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return bbox

    crop = page_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return bbox

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mask = _select_text_mask(gray)
    if not np.any(mask):
        return polygon_bbox or bbox

    row_counts = np.count_nonzero(mask > 0, axis=1)
    col_counts = np.count_nonzero(mask > 0, axis=0)
    for top_count, bottom_count, side_count in ((2, 3, 3), (3, 4, 4), (4, 5, 5), (5, 6, 6)):
        top_rows = np.where(row_counts >= top_count)[0]
        bottom_rows = np.where(row_counts >= bottom_count)[0]
        cols = np.where(col_counts >= side_count)[0]
        if top_rows.size < 4 or bottom_rows.size < 4 or cols.size < 4:
            continue
        left = int(cols[0])
        right = int(cols[-1]) + 1
        top = int(top_rows[0])
        bottom = int(bottom_rows[-1]) + 1
        if bottom > top and right > left:
            return [x1 + left, y1 + top, x1 + right, y1 + bottom]

    return polygon_bbox or bbox


def _paddle_full_page_max_side() -> int:
    raw = os.getenv("TRADUZAI_PADDLE_FULL_PAGE_MAX_SIDE", "0")
    try:
        value = int(str(raw).strip())
    except Exception:
        return 0
    return max(0, value)


def _scale_bbox(bbox: list[int], scale_x: float, scale_y: float) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    ]


def _scale_polygon_points(points: list[list[int]], scale_x: float, scale_y: float) -> list[list[int]]:
    scaled: list[list[int]] = []
    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        scaled.append([
            int(round(float(point[0]) * scale_x)),
            int(round(float(point[1]) * scale_y)),
        ])
    return scaled


def _rotate_orthogonal(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    normalized = int(rotation_deg) % 360
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _unrotate_orthogonal_point(
    point: list[int] | tuple[int, int],
    *,
    original_width: int,
    original_height: int,
    rotation_deg: int,
) -> list[int]:
    x_rot = float(point[0])
    y_rot = float(point[1])
    normalized = int(rotation_deg) % 360
    if normalized == 90:
        x = y_rot
        y = float(original_height - 1) - x_rot
    elif normalized == 180:
        x = float(original_width - 1) - x_rot
        y = float(original_height - 1) - y_rot
    elif normalized == 270:
        x = float(original_width - 1) - y_rot
        y = x_rot
    else:
        x = x_rot
        y = y_rot
    x = max(0.0, min(float(max(0, original_width - 1)), x))
    y = max(0.0, min(float(max(0, original_height - 1)), y))
    return [int(round(x)), int(round(y))]


def _unrotate_orthogonal_polygon(
    polygon: list[list[int]],
    *,
    original_width: int,
    original_height: int,
    rotation_deg: int,
) -> list[list[int]]:
    return [
        _unrotate_orthogonal_point(
            point,
            original_width=original_width,
            original_height=original_height,
            rotation_deg=rotation_deg,
        )
        for point in polygon or []
    ]


def _bbox_union_for_ocr(a: list[int], b: list[int]) -> list[int]:
    return [
        min(int(a[0]), int(b[0])),
        min(int(a[1]), int(b[1])),
        max(int(a[2]), int(b[2])),
        max(int(a[3]), int(b[3])),
    ]


def _expanded_bboxes_touch(a: list[int], b: list[int], margin: int = 48) -> bool:
    return not (
        int(a[2]) + margin < int(b[0])
        or int(b[2]) + margin < int(a[0])
        or int(a[3]) + margin < int(b[1])
        or int(b[3]) + margin < int(a[1])
    )


def _group_rotated_ocr_records(records: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for record in records:
        bbox = _coerce_bbox(record.get("source_bbox"))
        if bbox is None:
            continue
        rotation = normalize_rotation_deg(record.get("rotation_deg"))
        matched_group = None
        for group in groups:
            if abs(normalize_rotation_deg(group.get("rotation_deg")) - rotation) > 20.0:
                continue
            if _expanded_bboxes_touch(group["bbox"], bbox, margin=64):
                matched_group = group
                break
        if matched_group is None:
            groups.append(
                {
                    "bbox": bbox,
                    "rotation_deg": rotation,
                    "records": [record],
                }
            )
            continue
        matched_group["bbox"] = _bbox_union_for_ocr(matched_group["bbox"], bbox)
        matched_group["records"].append(record)

    grouped: list[dict] = []
    for group in groups:
        lines = list(group.get("records") or [])
        lines.sort(
            key=lambda item: (
                int(item.get("_rotated_ocr_angle", 0) or 0),
                (item.get("_rotated_line_bbox") or [0, 0, 0, 0])[1],
                (item.get("_rotated_line_bbox") or [0, 0, 0, 0])[0],
            )
        )
        text = " ".join(str(item.get("text") or "").strip() for item in lines).strip()
        if not text:
            continue
        line_polygons = [
            polygon
            for item in lines
            for polygon in (item.get("line_polygons") or [])
            if polygon
        ]
        bbox = group["bbox"]
        confidence_values = []
        for item in lines:
            try:
                confidence_values.append(float(item.get("confidence") or 0.0))
            except Exception:
                pass
        confidence = sum(confidence_values) / max(1, len(confidence_values))
        rotation_deg = infer_rotation_deg_from_line_polygons(line_polygons)
        if rotation_deg == 0.0:
            rotation_deg = normalize_rotation_deg(group.get("rotation_deg"))
        grouped.append(
            _attach_rotated_text_metadata({
                "text": text,
                "source_bbox": bbox,
                "bbox": bbox,
                "line_polygons": line_polygons,
                "text_pixel_bbox": _bbox_from_polygons(line_polygons) or bbox,
                "confidence": round(float(confidence), 3),
                "rotation_deg": rotation_deg,
                "rotation_source": "rotated_page_ocr",
                "detector": "rotated_full_page_recovery",
                "_rotated_ocr_angle": lines[0].get("_rotated_ocr_angle") if lines else None,
            })
        )
    return grouped


def _repair_rotated_ocr_edge_clipping(text: str) -> str:
    repaired = str(text or "").strip()
    repaired = re.sub(r"(?i)^ppoint\b", "Appoint", repaired)
    if repaired.count("]") > repaired.count("[") and not repaired.lstrip().startswith("["):
        repaired = repaired.replace("]", "", 1)
    return repaired.strip()


class OCREngine:
    """
    Motor de OCR com suporte a batching.
    
    Backends:
        - "manga-ocr": TrOCR fine-tuned para manga (kha-white/manga-ocr)
          Melhor para japonês, excelente para inglês em manga
        - "paddleocr": PaddleOCR multilingual
          Já está no seu stack — usado como fallback ou alternativa
    """

    def __init__(
        self,
        model: str = "paddleocr",
        device: str = "cuda",
        half: bool = True,
        batch_size: int = 8,
        lang: str = "en",
    ):
        self._requested_model = model
        self.model_name = model
        self.device = self._resolve_device(device)
        self.half = half and self.device.type == "cuda"
        if _env_bool("TRADUZAI_OCR_ADAPTIVE_BATCH", False):
            self.batch_size = 16 if self.device.type == "cuda" else 4
        else:
            self.batch_size = batch_size
        self.lang = lang
        self._model = None
        self._processor = None
        self._ocr_cache: OrderedDict[str, str] = OrderedDict()
        self._observation_records_local = threading.local()
        self._load_model()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        if self.model_name == "manga-ocr":
            self._load_manga_ocr()
        elif self.model_name == "paddleocr":
            self._load_paddle_ocr()
        elif self.model_name == "easyocr":
            raise OcrBackendUnavailable("EasyOCR esta desativado; PaddleOCR e obrigatorio.")
        else:
            raise ValueError(f"OCR backend '{self.model_name}' não suportado")

    def _load_manga_ocr(self):
        """manga-ocr usa VisionEncoderDecoder (TrOCR) da HuggingFace."""
        try:
            from transformers import AutoFeatureExtractor, VisionEncoderDecoderModel, AutoTokenizer
            
            model_id = "kha-white/manga-ocr-base"
            logger.info(f"Carregando manga-ocr de {model_id}...")
            
            self._processor = AutoFeatureExtractor.from_pretrained(model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = VisionEncoderDecoderModel.from_pretrained(model_id)
            self._model.to(self.device)
            
            if self.half:
                self._model = self._model.half()
            
            self._model.eval()
            self._backend = "manga-ocr"
            logger.info(f"manga-ocr carregado ({self.device})")

        except ImportError:
            logger.warning("transformers não instalado, usando PaddleOCR como fallback")
            self.model_name = "paddleocr"
            self._load_paddle_ocr()
        except Exception as exc:
            logger.warning("manga-ocr não carregou (%s); usando PaddleOCR como fallback", exc)
            self.model_name = "paddleocr"
            self._load_paddle_ocr()

    def _load_paddle_ocr(self):
        """PaddleOCR — já presente no TraduzAi."""
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        try:
            import sys
            import paddle

            require_gpu = _env_bool("TRADUZAI_REQUIRE_GPU", False)
            paddle_has_cuda = bool(getattr(getattr(paddle, "device", None), "is_compiled_with_cuda", lambda: False)())
            if require_gpu and self.device.type != "cuda" and paddle_has_cuda:
                logger.warning("PaddleOCR recebeu device CPU em ambiente GPU; forcando gpu:0")
                self.device = torch.device("cuda")
                self.half = True
            if sys.version_info >= (3, 12) and self.device.type != "cuda":
                raise ImportError("PaddleOCR incompativel com Python 3.12 via CPU; backend PaddleOCR indisponivel.")
            from paddleocr import PaddleOCR
            from vision_stack.paddle_compat import create_paddle_ocr
            import paddle.base.libpaddle as libpaddle
            if hasattr(libpaddle, 'AnalysisConfig') and not hasattr(libpaddle.AnalysisConfig, 'set_optimization_level'):
                libpaddle.AnalysisConfig.set_optimization_level = lambda *args, **kwargs: None
        except Exception as exc:
            raise OcrBackendUnavailable(
                f"PaddleOCR indisponivel; EasyOCR fallback removido: {exc}"
            ) from exc
        
        mapped_lang = normalize_paddleocr_language(self.lang)
        
        use_gpu = self.device.type == "cuda"
        show_log = _env_bool("TRADUZAI_PADDLE_SHOW_LOG", False)
        use_angle_cls = False
        self._paddle_use_angle_cls = use_angle_cls
        self._model = create_paddle_ocr(
            PaddleOCR,
            use_angle_cls=use_angle_cls,
            lang=mapped_lang,
            use_gpu=use_gpu,
            enable_mkldnn=not use_gpu,  # MKL-DNN acelera CPU
            show_log=show_log,
        )
        self._backend = "paddleocr"
        logger.info(f"PaddleOCR carregado (lang={mapped_lang}, gpu={use_gpu})")

    def _load_easyocr(self):
        raise OcrBackendUnavailable("EasyOCR esta desativado; PaddleOCR e obrigatorio.")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def _record_raw_model_execution(self, raw_output) -> None:
        from qa.runtime_fingerprint import record_engine_event

        requested_engine = getattr(
            self,
            "_requested_model",
            getattr(self, "model_name", ""),
        )
        resolved_engine = getattr(self, "model_name", requested_engine)
        fallback_used = resolved_engine != requested_engine
        record_engine_event(
            stage="ocr",
            requested_engine=requested_engine,
            resolved_engine=resolved_engine,
            backend=self,
            execution_status="succeeded",
            result_status=(
                "empty" if _raw_ocr_output_is_empty(raw_output) else "accepted"
            ),
            fallback_used=fallback_used,
            fallback_reason=(
                "resolved_engine_differs_from_request" if fallback_used else ""
            ),
            execution_context="chapter",
        )

    def _observation_state(self):
        state = getattr(self, "_observation_records_local", None)
        if state is None:
            state = threading.local()
            self._observation_records_local = state
        if not hasattr(state, "records"):
            state.records = []
            state.next_attempt = 1
            state.stats = {}
            state.full_page_lines = []
        return state

    def _reset_observation_records(self) -> None:
        state = self._observation_state()
        state.records = []
        state.next_attempt = 1
        state.stats = {}
        state.full_page_lines = []

    def _append_observation_record(
        self,
        *,
        provider: str,
        text: str = "",
        raw_text: str | None = None,
        confidence: float = 0.0,
        bbox=None,
        line_polygons=None,
        accepted: bool,
        rejection_reason: str | None,
        variant: str,
        **metadata,
    ) -> str:
        state = self._observation_state()
        attempt_id = f"ocr_attempt_{int(state.next_attempt):04d}"
        state.next_attempt = int(state.next_attempt) + 1
        normalized_polygons = _normalize_line_polygons(line_polygons or [])
        normalized_bbox = _coerce_bbox(bbox)
        if normalized_bbox is None and normalized_polygons:
            normalized_bbox = _bbox_from_polygons(normalized_polygons)
        try:
            normalized_confidence = float(confidence or 0.0)
        except (TypeError, ValueError):
            normalized_confidence = 0.0
        cleaned_text = str(text or "")
        record = {
            "provider": str(provider or "ocr"),
            "attempt_id": attempt_id,
            "variant": str(variant or "native"),
            "text": cleaned_text,
            "raw_text": str(cleaned_text if raw_text is None else raw_text),
            "confidence": normalized_confidence,
            "bbox": list(normalized_bbox or []),
            "line_polygons": normalized_polygons,
            "accepted": bool(accepted),
            "rejection_reason": str(rejection_reason) if rejection_reason else None,
        }
        for key, value in metadata.items():
            if value is not None:
                record[str(key)] = copy.deepcopy(value)
        state.records.append(record)
        return attempt_id

    def _crop_observation_provider(self) -> str:
        backend = str(getattr(self, "_backend", getattr(self, "model_name", "ocr")) or "ocr")
        if backend == "paddleocr":
            return "paddle_crop"
        if backend == "manga-ocr":
            return "manga_ocr_crop"
        return "ocr_crop"

    def recognize_page_with_evidence(
        self,
        page_rgb: np.ndarray,
        blocks: list,
        *,
        request: OCRRequest,
        **options,
    ) -> OCRInvocationResult:
        """Recognize one page and return all evidence in one immutable value."""

        root_rgb = _normalize_ocr_rgb(page_rgb)
        if request.root_input_pixel_sha256 != canonical_page_sha256(root_rgb):
            raise OCRInputPixelIdentityError("OCR request is not bound to the supplied page")
        if getattr(self, "_backend", "") != "paddleocr":
            raise OcrBackendUnavailable("atomic OCR evidence currently requires PaddleOCR")
        model = getattr(self, "_model", None)
        provider = getattr(model, "ocr", None)
        if not callable(provider):
            raise OcrBackendUnavailable("PaddleOCR provider is unavailable")
        if (blocks and _env_bool("TRADUZAI_OCR_CROP_FIRST", False)
                and not options.get("force_full_page")
                and self._crop_first_indices(blocks)):
            return self._recognize_page_crop_first(root_rgb, blocks, request, provider)
        operations = [OCRTransformOperation(kind="identity")]
        if bool(options.get("force_downscale")):
            operations.append(
                OCRTransformOperation(
                    kind="resize",
                    output_size=(
                        max(1, root_rgb.shape[1] // 2),
                        max(1, root_rgb.shape[0] // 2),
                    ),
                    interpolation="area",
                )
            )
        transform_spec = OCRTransformSpec.build(tuple(operations))
        provider_rgb = transform_spec.replay(root_rgb)
        attempt, full_page_lines = execute_hash_bound_provider_attempt(
            request=request,
            root_input_rgb=root_rgb,
            actual_input_rgb=provider_rgb,
            transform_spec=transform_spec,
            variant_id="full_page",
            provider=provider,
            expected_input_pixel_sha256=canonical_page_sha256(provider_rgb),
            input_kind="full_page",
            provider_kwargs={"det": True, "rec": True, "cls": False},
            source="paddle_full_page",
            execution_recorder=self._record_raw_model_execution,
        )
        if provider_rgb.shape[:2] != root_rgb.shape[:2]:
            full_page_lines = _records_in_page_space(
                full_page_lines,
                bbox_page=(0, 0, root_rgb.shape[1], root_rgb.shape[0]),
                input_width=provider_rgb.shape[1],
                input_height=provider_rgb.shape[0],
            )

        output_blocks: list[OCRBlock] = []
        if blocks:
            for index, block in enumerate(blocks):
                raw_bbox = getattr(block, "xyxy", (0, 0, root_rgb.shape[1], root_rgb.shape[0]))
                bbox = tuple(int(round(float(value))) for value in raw_bbox)
                bx1, by1, bx2, by2 = bbox
                matching: list[OCRObservationRecord] = []
                for record in full_page_lines:
                    rx1, ry1, rx2, ry2 = record.bbox_page
                    intersection = max(0, min(bx2, rx2) - max(bx1, rx1)) * max(0, min(by2, ry2) - max(by1, ry1))
                    record_area = max(1, (rx2 - rx1) * (ry2 - ry1))
                    if intersection / record_area >= 0.18:
                        matching.append(record)
                text = " ".join(record.text for record in matching if record.text).strip()
                confidence = max((record.confidence for record in matching), default=0.0)
                polygons = tuple(point for record in matching for point in record.polygon_page)
                block_id = str(
                    getattr(block, "region_id", "")
                    or getattr(block, "component_id", "")
                    or f"ocr_block_{index + 1:04d}"
                )
                output_blocks.append(OCRBlock(block_id, text, confidence, bbox, polygons))
        else:
            for index, record in enumerate(full_page_lines):
                output_blocks.append(
                    OCRBlock(
                        f"ocr_block_{index + 1:04d}",
                        record.text,
                        record.confidence,
                        record.bbox_page,
                        record.polygon_page,
                    )
                )

        diagnostics = OCRDiagnostics(
            provider=request.provider_family,
            extras={
                "block_count": len(blocks),
                "full_page_line_count": len(full_page_lines),
                "full_page_mapped": sum(1 for block in output_blocks if block.text),
            },
        )
        return OCRInvocationResult.build(
            request=request,
            blocks=tuple(output_blocks),
            observations=full_page_lines,
            full_page_lines=full_page_lines,
            attempts=(attempt,),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _prefer_detected_crop(block) -> bool:
        """Large tall regions lose detail when the entire page is resized by OCR."""
        try:
            x1, y1, x2, y2 = (float(value) for value in block.xyxy)
        except (AttributeError, TypeError, ValueError):
            return False
        return x2 - x1 >= 280 and y2 - y1 >= 220

    @classmethod
    def _crop_first_indices(cls, blocks: list) -> tuple[int, ...]:
        """Do not independently crop overlapping detections with ambiguous owners."""
        selected = []
        for index, block in enumerate(blocks):
            if not cls._prefer_detected_crop(block):
                continue
            x1, y1, x2, y2 = (float(value) for value in block.xyxy)
            if any(
                other_index != index
                and max(0.0, min(x2, float(other.xyxy[2])) - max(x1, float(other.xyxy[0])))
                * max(0.0, min(y2, float(other.xyxy[3])) - max(y1, float(other.xyxy[1]))) > 0
                for other_index, other in enumerate(blocks)
            ):
                continue
            selected.append(index)
        return tuple(selected)

    @staticmethod
    def _crop_text_skew_angle(crop: np.ndarray) -> float | None:
        """Estimate a dominant text baseline only when Hough support is unambiguous."""
        import math

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30,
            minLineLength=max(35, int(crop.shape[1] * 0.2)), maxLineGap=8,
        )
        if lines is None:
            return None
        candidates = []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = (int(value) for value in line)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) <= 40:
                candidates.append((angle, math.hypot(x2 - x1, y2 - y1)))
        if len(candidates) < 6:
            return None
        best = max(
            range(-40, 41),
            key=lambda center: sum(length for angle, length in candidates if abs(angle - center) <= 3),
        )
        inliers = sorted((angle, length) for angle, length in candidates if abs(angle - best) <= 3)
        total = sum(length for _, length in candidates)
        supported = sum(length for _, length in inliers)
        if supported < 0.65 * total:
            return None
        cumulative = 0.0
        for angle, length in inliers:
            cumulative += length
            # Slanted glyph edges tend to extend below the baseline; the upper
            # supported quantile avoids over-rotating the small end punctuation.
            if cumulative >= supported * 0.70:
                quantized = round(angle * 2) / 2
                return quantized if 12 <= abs(quantized) <= 40 else None
        return None

    @staticmethod
    def _project_deskewed_records(records, box, matrix, crop_shape):
        inverse = cv2.invertAffineTransform(matrix)
        width, height = crop_shape[1], crop_shape[0]
        projected = []
        for record in records:
            points = []
            for x, y in record.polygon_page:
                px, py = inverse @ np.asarray((x, y, 1.0), dtype=np.float64)
                points.append((
                    box[0] + int(round(np.clip(px, 0, width - 1))),
                    box[1] + int(round(np.clip(py, 0, height - 1))),
                ))
            if not points:
                continue
            xs, ys = zip(*points)
            projected.append(replace(
                record, polygon_page=tuple(points),
                bbox_page=(min(xs), min(ys), max(xs), max(ys)),
            ))
        return tuple(projected)

    @staticmethod
    def _terminal_ink_outside_line(image: np.ndarray, polygon) -> bool:
        """Find a small isolated terminal mark just beyond a detected line."""
        if len(polygon) != 4:
            return False
        xs, ys = zip(*polygon)
        right, bottom = max(xs), max(ys)
        height, width = image.shape[:2]
        if right + 2 >= width:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, _, stats, _ = cv2.connectedComponentsWithStats((gray < 120).astype(np.uint8))
        for x, y, w, h, area in stats[1:]:
            if (
                2 <= area <= 12 and w <= 4 and h <= 4
                and right < x <= min(width - 1, right + 8)
                and bottom - 8 <= y <= bottom + 3
            ):
                return True
        return False

    def _recognize_page_crop_first(
        self,
        root_rgb: np.ndarray,
        blocks: list,
        request: OCRRequest,
        provider,
    ) -> OCRInvocationResult:
        """Read detected regions first, then only the unassigned page pixels."""

        height, width = root_rgb.shape[:2]
        crop_boxes = []
        crop_indices = self._crop_first_indices(blocks)
        for index in crop_indices:
            block = blocks[index]
            try:
                x1, y1, x2, y2 = (float(value) for value in block.xyxy)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("crop-first OCR requires a valid detector box") from exc
            box = (
                max(0, int(x1)),
                max(0, int(y1)),
                min(width, int(x2)),
                min(height, int(y2)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("crop-first OCR detector box has no pixels")
            crop_boxes.append(box)

        attempts = []
        observations = []
        crop_lines_by_index = {}
        preserved_boxes = []
        for ordinal, (index, box) in enumerate(zip(crop_indices, crop_boxes), 1):
            native = root_rgb[box[1]:box[3], box[0]:box[2]]
            angle = self._crop_text_skew_angle(native)
            matrix = None
            operations = [OCRTransformOperation(kind="crop", bbox_page=box)]
            if angle is not None:
                crop_height, crop_width = native.shape[:2]
                matrix = cv2.getRotationMatrix2D((crop_width / 2, crop_height / 2), angle, 1.0)
                cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
                new_width = int(crop_height * sine + crop_width * cosine)
                new_height = int(crop_height * cosine + crop_width * sine)
                matrix[0, 2] += new_width / 2 - crop_width / 2
                matrix[1, 2] += new_height / 2 - crop_height / 2
                fixed = tuple(int(round(value * 1_000_000)) for value in matrix.reshape(-1))
                matrix = np.asarray(fixed, dtype=np.float64).reshape(2, 3) / 1_000_000
                operations.append(OCRTransformOperation(
                    kind="deskew_affine", output_size=(new_width, new_height),
                    interpolation="linear", border_mode="constant", border_value_rgb=(255, 255, 255),
                    affine_matrix_fixed_1e6=fixed, algorithm_id="hough_text_baseline_v1",
                ))
            spec = OCRTransformSpec.build(tuple(operations))
            crop = spec.replay(root_rgb)
            attempt, lines = execute_hash_bound_provider_attempt(
                request=request, root_input_rgb=root_rgb, actual_input_rgb=crop,
                transform_spec=spec, variant_id=f"detected_crop_{ordinal:04d}" + ("_deskew" if angle is not None else ""), provider=provider,
                expected_input_pixel_sha256=canonical_page_sha256(crop),
                input_bbox_page=box, input_kind="detected_crop",
                provider_kwargs={"det": True, "rec": True, "cls": False},
                attempt_ordinal=ordinal, source="paddle_detected_crop",
                execution_recorder=self._record_raw_model_execution,
            )
            projected = (
                self._project_deskewed_records(lines, box, matrix, native.shape)
                if matrix is not None else _records_in_page_space(
                    lines, bbox_page=box, input_width=crop.shape[1], input_height=crop.shape[0],
                )
            )
            attempts.append(attempt)
            if matrix is not None:
                refined = list(projected)
                for line_number, raw_line in enumerate(lines):
                    if not raw_line.polygon_page or not self._terminal_ink_outside_line(crop, raw_line.polygon_page):
                        continue
                    xs, ys = zip(*raw_line.polygon_page)
                    left = max(0, min(xs) - 6)
                    top = max(0, min(ys) - 6)
                    right = min(crop.shape[1], max(xs) + 10)
                    bottom = min(crop.shape[0], max(ys) + 7)
                    if right <= left or bottom <= top:
                        continue
                    terminal_spec = OCRTransformSpec.build((*operations, OCRTransformOperation(
                        kind="crop", bbox_page=(left, top, right, bottom),
                    )))
                    terminal_input = terminal_spec.replay(root_rgb)

                    def recognition_only(image, **_kwargs):
                        raw = provider(image, det=False, rec=True, cls=False)
                        candidate = raw[0][0] if isinstance(raw, list) and raw and raw[0] else None
                        if not isinstance(candidate, (tuple, list)) or len(candidate) < 2:
                            return []
                        width, height = image.shape[1], image.shape[0]
                        polygon = ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1))
                        return [[(polygon, (candidate[0], candidate[1]))]]

                    retry_attempt, retry_lines = execute_hash_bound_provider_attempt(
                        request=request, root_input_rgb=root_rgb, actual_input_rgb=terminal_input,
                        transform_spec=terminal_spec,
                        variant_id=f"detected_crop_{ordinal:04d}_terminal_{line_number + 1:02d}",
                        provider=recognition_only,
                        expected_input_pixel_sha256=canonical_page_sha256(terminal_input),
                        input_bbox_page=box, input_kind="terminal_recrop",
                        attempt_ordinal=len(attempts) + 1, source="paddle_terminal_recrop",
                        execution_recorder=self._record_raw_model_execution,
                    )
                    attempts.append(retry_attempt)
                    if len(retry_lines) != 1:
                        continue
                    retry = retry_lines[0]
                    previous = raw_line.text.rstrip()
                    if retry.text not in {previous + mark for mark in (".", ",", "!", "?", ";", ":")}:
                        continue
                    local_polygon = tuple((x + left, y + top) for x, y in retry.polygon_page)
                    local_record = replace(retry, polygon_page=local_polygon)
                    refined[line_number] = self._project_deskewed_records(
                        (local_record,), box, matrix, native.shape,
                    )[0]
                projected = tuple(refined)
            # A short fragment from a large region is insufficient evidence
            # to hide its pixels from the ordinary page read (notably SFX).
            recognized = " ".join(line.text.strip() for line in projected if line.text.strip())
            if sum(char.isalnum() for char in recognized) >= 24:
                observations.extend(projected)
                crop_lines_by_index[index] = projected
                preserved_boxes.append(box)

        # A masked sweep retains the historical coverage of text outside
        # detector boxes without reading the detected text a second time.
        mask_spec = OCRTransformSpec.build((
            OCRTransformOperation(kind="mask_rectangles", mask_bboxes=tuple(preserved_boxes))
            if preserved_boxes else OCRTransformOperation(kind="identity"),
        ))
        residual = mask_spec.replay(root_rgb)
        residual_attempt, residual_lines = execute_hash_bound_provider_attempt(
            request=request, root_input_rgb=root_rgb, actual_input_rgb=residual,
            transform_spec=mask_spec, variant_id="unassigned_page", provider=provider,
            expected_input_pixel_sha256=canonical_page_sha256(residual),
            input_kind="unassigned_page", provider_kwargs={"det": True, "rec": True, "cls": False},
            attempt_ordinal=len(attempts) + 1, source="paddle_unassigned_page",
            execution_recorder=self._record_raw_model_execution,
        )
        attempts.append(residual_attempt)
        for line in residual_lines:
            x1, y1, x2, y2 = line.bbox_page
            if any(max(0, min(x2, bx2) - max(x1, bx1)) * max(0, min(y2, by2) - max(y1, by1)) > 0
                   for bx1, by1, bx2, by2 in preserved_boxes):
                continue
            observations.append(line)
        output_blocks = []
        for index, block in enumerate(blocks):
            raw_bbox = tuple(int(round(float(value))) for value in block.xyxy)
            if index in crop_lines_by_index:
                matching = crop_lines_by_index[index]
            else:
                bx1, by1, bx2, by2 = raw_bbox
                matching = tuple(record for record in residual_lines if (
                    max(0, min(bx2, record.bbox_page[2]) - max(bx1, record.bbox_page[0]))
                    * max(0, min(by2, record.bbox_page[3]) - max(by1, record.bbox_page[1]))
                    / max(1, (record.bbox_page[2] - record.bbox_page[0]) * (record.bbox_page[3] - record.bbox_page[1]))
                ) >= 0.18)
            block_id = str(getattr(block, "region_id", "") or getattr(block, "component_id", "") or f"ocr_block_{index + 1:04d}")
            output_blocks.append(OCRBlock(
                block_id, " ".join(line.text for line in matching if line.text).strip(),
                max((line.confidence for line in matching), default=0.0), raw_bbox,
                tuple(point for line in matching for point in line.polygon_page),
            ))
        diagnostics = OCRDiagnostics(request.provider_family, {
            "block_count": len(blocks),
            "full_page_line_count": len(observations),
            "full_page_mapped": sum(bool(block.text) for block in output_blocks),
            "crop_first_calls": len(attempts),
        })
        return OCRInvocationResult.build(
            request=request, blocks=tuple(output_blocks), observations=tuple(observations),
            full_page_lines=tuple(observations), attempts=tuple(attempts), diagnostics=diagnostics,
        )

    def recognize_region_with_evidence(
        self,
        page_rgb: np.ndarray,
        *,
        bbox_page: tuple[int, int, int, int],
        request: OCRRequest,
        variants: tuple[str, ...] = ("anchored_crop",),
        stop_on_first_text: bool = False,
    ) -> OCRInvocationResult:
        """Run hash-bound OCR variants for one component-anchored page crop."""

        root_rgb = _normalize_ocr_rgb(page_rgb)
        if request.root_input_pixel_sha256 != canonical_page_sha256(root_rgb):
            raise OCRInputPixelIdentityError("OCR request is not bound to the supplied page")
        x1, y1, x2, y2 = tuple(int(value) for value in bbox_page)
        if x1 < 0 or y1 < 0 or x2 > root_rgb.shape[1] or y2 > root_rgb.shape[0] or x2 <= x1 or y2 <= y1:
            raise ValueError("anchored OCR bbox is outside the supplied page")
        if getattr(self, "_backend", "") != "paddleocr":
            raise OcrBackendUnavailable("atomic anchored OCR currently requires PaddleOCR")
        provider = getattr(getattr(self, "_model", None), "ocr", None)
        if not callable(provider):
            raise OcrBackendUnavailable("PaddleOCR provider is unavailable")

        attempts: list[OCRAttempt] = []
        observations: list[OCRObservationRecord] = []
        for ordinal, variant in enumerate(tuple(variants), 1):
            operations: list[OCRTransformOperation] = [
                OCRTransformOperation(kind="crop", bbox_page=(x1, y1, x2, y2))
            ]
            if variant == "gray":
                operations.append(OCRTransformOperation(kind="grayscale_to_rgb"))
            elif variant == "inverted":
                operations.append(OCRTransformOperation(kind="invert"))
            elif variant == "scale_2x":
                operations.append(
                    OCRTransformOperation(
                        kind="resize",
                        output_size=((x2 - x1) * 2, (y2 - y1) * 2),
                        interpolation="cubic",
                    )
                )
            elif variant not in {"native", "anchored_crop"}:
                raise ValueError(f"unsupported anchored OCR variant: {variant}")
            transform_spec = OCRTransformSpec.build(tuple(operations))
            provider_rgb = transform_spec.replay(root_rgb)
            attempt, records = execute_hash_bound_provider_attempt(
                request=request,
                root_input_rgb=root_rgb,
                actual_input_rgb=provider_rgb,
                transform_spec=transform_spec,
                variant_id=variant,
                provider=provider,
                expected_input_pixel_sha256=canonical_page_sha256(provider_rgb),
                input_bbox_page=(x1, y1, x2, y2),
                input_kind="anchored_crop",
                provider_kwargs={"det": True, "rec": True, "cls": False},
                attempt_ordinal=ordinal,
                source=f"paddle_anchored_{variant}",
            )
            attempts.append(attempt)
            observations.extend(
                _records_in_page_space(
                    records,
                    bbox_page=(x1, y1, x2, y2),
                    input_width=provider_rgb.shape[1],
                    input_height=provider_rgb.shape[0],
                )
            )
            if stop_on_first_text and records:
                break

        blocks = tuple(
            OCRBlock(
                f"anchored_block_{index:04d}",
                record.text,
                record.confidence,
                record.bbox_page,
                record.polygon_page,
            )
            for index, record in enumerate(observations, 1)
        )
        return OCRInvocationResult.build(
            request=request,
            blocks=blocks,
            observations=tuple(observations),
            attempts=tuple(attempts),
            diagnostics=OCRDiagnostics(
                request.provider_family,
                {"bbox_page": list(bbox_page), "variants": list(variants)},
            ),
        )

    def recognize_batch(self, crops: list[np.ndarray]) -> list[str]:
        """
        Reconhece texto em múltiplas imagens recortadas.
        Processa em batches para máxima eficiência GPU.
        """
        self._reset_observation_records()
        if not crops:
            return []

        cache_enabled = _env_bool("TRADUZAI_OCR_CACHE", False)
        if not hasattr(self, "_ocr_cache") or not isinstance(getattr(self, "_ocr_cache", None), OrderedDict):
            self._ocr_cache = OrderedDict()

        hits = 0
        misses = 0
        results: list[str | None] = [None] * len(crops)
        pending: list[np.ndarray] = []
        pending_indices: list[int] = []
        pending_keys: list[str] = []

        for index, crop in enumerate(crops):
            cache_key = self._crop_cache_key(crop) if cache_enabled else ""
            if cache_enabled and cache_key in self._ocr_cache:
                hits += 1
                self._ocr_cache.move_to_end(cache_key)
                results[index] = self._ocr_cache[cache_key]
                cached_text = str(results[index] or "")
                crop_height, crop_width = (
                    crop.shape[:2]
                    if isinstance(crop, np.ndarray) and crop.ndim >= 2
                    else (0, 0)
                )
                self._append_observation_record(
                    provider=self._crop_observation_provider(),
                    text=cached_text,
                    confidence=0.0,
                    bbox=[0, 0, int(crop_width), int(crop_height)],
                    line_polygons=[],
                    accepted=bool(cached_text.strip()),
                    rejection_reason=None if cached_text.strip() else "empty_result",
                    variant="cache_hit",
                    coordinate_space="crop",
                    crop_index=index,
                )
                continue
            misses += 1
            pending.append(crop)
            pending_indices.append(index)
            pending_keys.append(cache_key)

        for i in range(0, len(pending), self.batch_size):
            batch = pending[i : i + self.batch_size]
            state = self._observation_state()
            state.active_crop_indices = pending_indices[i : i + len(batch)]
            try:
                batch_results = self._recognize_batch_impl(batch)
            finally:
                state.active_crop_indices = None
            for offset, text in enumerate(batch_results):
                result_index = pending_indices[i + offset]
                results[result_index] = text
                if self._crop_observation_provider() != "paddle_crop":
                    crop = batch[offset]
                    crop_height, crop_width = (
                        crop.shape[:2]
                        if isinstance(crop, np.ndarray) and crop.ndim >= 2
                        else (0, 0)
                    )
                    cleaned_text = str(text or "")
                    self._append_observation_record(
                        provider=self._crop_observation_provider(),
                        text=cleaned_text,
                        confidence=0.0,
                        bbox=[0, 0, int(crop_width), int(crop_height)],
                        line_polygons=[],
                        accepted=bool(cleaned_text.strip()),
                        rejection_reason=None if cleaned_text.strip() else "empty_result",
                        variant="native",
                        coordinate_space="crop",
                        crop_index=result_index,
                    )
                key = pending_keys[i + offset]
                if cache_enabled and key:
                    self._ocr_cache[key] = text
                    self._ocr_cache.move_to_end(key)
                    while len(self._ocr_cache) > 256:
                        self._ocr_cache.popitem(last=False)

        self._observation_state().stats = {
            "ocr_cache_hits": int(hits),
            "ocr_cache_misses": int(misses),
        }
        return [str(text or "") for text in results]

    @staticmethod
    def _crop_cache_key(crop: np.ndarray) -> str:
        if not isinstance(crop, np.ndarray) or crop.size == 0:
            return "empty"
        try:
            resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        except Exception:
            resized = np.zeros((64, 64, 3), dtype=np.uint8)
        return hashlib.blake2b(resized.tobytes(), digest_size=8).hexdigest()

    def recognize_blocks_from_page(
        self,
        page_rgb: np.ndarray,
        blocks: list,
        allow_sparse_mapping: bool = False,
        crop_fallback_max: int | None = None,
        sparse_crop_fallback_max: int | None = None,
    ) -> list[str | dict]:
        """Reconhece texto para cada bloco detectado, alinhando o resultado ao `blocks`.

        Otimiza o backend PaddleOCR: evita rodar detecção repetidamente por crop.
        Faz 1 pass de OCR na página inteira e associa as linhas reconhecidas aos blocos.
        """
        self._reset_observation_records()
        if not blocks:
            self._observation_state().stats = {
                "block_count": 0,
                "full_page_mapped": 0,
                "crop_fallback_max": 0,
                "crop_fallback_attempts": 0,
                "crop_fallback_recovered": 0,
            }
            return []

        if not isinstance(page_rgb, np.ndarray) or page_rgb.size == 0:
            self._observation_state().stats = {
                "block_count": len(blocks),
                "full_page_mapped": 0,
                "crop_fallback_max": 0,
                "crop_fallback_attempts": 0,
                "crop_fallback_recovered": 0,
            }
            return [""] * len(blocks)

        if getattr(self, "_backend", "") != "paddleocr":
            raise ValueError("recognize_blocks_from_page disponível apenas para PaddleOCR")

        page_bgr = page_rgb
        if len(page_rgb.shape) == 3 and page_rgb.shape[2] >= 3:
            try:
                page_bgr = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                page_bgr = page_rgb

        texts = self._paddle_ocr_full_page_to_blocks(
            page_bgr,
            blocks,
            allow_sparse_mapping=allow_sparse_mapping,
        )
        if crop_fallback_max is None:
            crop_fallback_max = _env_int("TRADUZAI_PADDLE_CROP_FALLBACK_MAX", 3)
        max_fallback = max(0, int(crop_fallback_max))
        if sparse_crop_fallback_max is None:
            sparse_crop_fallback_max = max_fallback
        max_sparse_fallback = max(0, int(sparse_crop_fallback_max))
        shadow_enabled = _ocr_fallback_shadow_enabled()
        shadow_limit = _ocr_fallback_shadow_limit()
        if texts is None:
            recovered_by_crop: list[str] = [""] * len(blocks)
            attempts = 0
            recovered_count = 0
            shadow_would_skip = 0
            shadow_recovered_after_limit = 0
            for index, block in enumerate(blocks):
                if attempts >= max_fallback:
                    break
                try:
                    block_confidence = float(getattr(block, "confidence", 1.0) or 0.0)
                except Exception:
                    block_confidence = 1.0
                if block_confidence < 0.45:
                    continue
                crop = self._crop_block_from_page(page_rgb, block)
                if not self._crop_might_have_text(crop):
                    continue
                attempts += 1
                recovered = self._recognize_single_paddle_with_retry(
                    crop,
                    observation_bbox=self._block_bbox(block),
                    observation_line_polygons=getattr(block, "line_polygons", None),
                    observation_coordinate_space="page",
                    crop_index=index,
                )
                recovered_by_crop[index] = recovered
                if str(recovered or "").strip():
                    recovered_count += 1
                if shadow_enabled and attempts > shadow_limit:
                    shadow_would_skip += 1
                    if str(recovered or "").strip():
                        shadow_recovered_after_limit += 1
            self._observation_state().stats = {
                "block_count": len(blocks),
                "full_page_mapping_failed": True,
                "full_page_mapped": 0,
                "crop_fallback_max": int(max_fallback),
                "sparse_crop_fallback_max": int(max_sparse_fallback),
                "crop_fallback_attempts": int(attempts),
                "crop_fallback_recovered": int(recovered_count),
                "crop_fallback_suppressed": 0,
            }
            if shadow_enabled:
                self._observation_state().stats.update(
                    {
                        "fallback_shadow_attempt_limit": int(shadow_limit),
                        "fallback_shadow_attempts_saved_or_would_skip": int(shadow_would_skip),
                        "fallback_shadow_recovered_after_limit": int(shadow_recovered_after_limit),
                        "fallback_shadow_full_page_already_resolved_count": 0,
                    }
                )
            if _env_bool("TRADUZAI_OCR_DEDUP", False):
                self._observation_state().stats["ocr_dedup_removed"] = int(
                    self._dedupe_ocr_records_in_place(recovered_by_crop, blocks)
                )
            else:
                self._observation_state().stats["ocr_dedup_removed"] = 0
            return recovered_by_crop

        # Fallback por crop apenas para casos prováveis, evitando custo alto em falsos positivos.
        full_page_mapped = sum(
            1
            for text in texts
            if str(text.get("text", "") if isinstance(text, dict) else text or "").strip()
        )
        stats = {
            "block_count": len(blocks),
            "full_page_mapping_failed": False,
            "full_page_mapped": int(full_page_mapped),
            "crop_fallback_max": int(max_fallback),
            "sparse_crop_fallback_max": int(max_sparse_fallback),
            "crop_fallback_attempts": 0,
            "crop_fallback_recovered": 0,
            "crop_fallback_suppressed": 0,
        }
        shadow_would_skip = 0
        shadow_recovered_after_limit = 0
        attempted = 0
        for index, text in enumerate(texts):
            if attempted >= max_sparse_fallback:
                break
            current_text = text.get("text", "") if isinstance(text, dict) else text
            if str(current_text or "").strip():
                continue
            try:
                block_confidence = float(getattr(blocks[index], "confidence", 1.0) or 0.0)
            except Exception:
                block_confidence = 1.0
            if block_confidence < 0.45:
                continue
            crop = self._crop_block_from_page(page_rgb, blocks[index])
            if not self._crop_might_have_text(crop):
                continue
            attempted += 1
            stats["crop_fallback_attempts"] += 1
            recovered = self._recognize_single_paddle_with_retry(
                crop,
                observation_bbox=self._block_bbox(blocks[index]),
                observation_line_polygons=getattr(blocks[index], "line_polygons", None),
                observation_coordinate_space="page",
                crop_index=index,
            )
            if str(recovered or "").strip():
                stats["crop_fallback_recovered"] += 1
            if shadow_enabled and attempted > shadow_limit:
                shadow_would_skip += 1
                if str(recovered or "").strip():
                    shadow_recovered_after_limit += 1
            if isinstance(texts[index], dict):
                updated = dict(texts[index])
                updated["text"] = recovered
                if _ocr_assignment_audit_enabled() and str(recovered or "").strip():
                    updated["_ocr_assignment_audit"] = {
                        "assignment_mode": "crop_fallback",
                        "block_bbox": list(self._block_bbox(blocks[index])),
                        "assigned_line_count": 0,
                        "assigned_lines": [],
                        "line_union_bbox": None,
                        "suspicious": False,
                        "reason": "crop_fallback_recovered",
                    }
                texts[index] = updated
            else:
                texts[index] = recovered
        if max_sparse_fallback <= 0:
            suppressed = 0
            for index, text in enumerate(texts):
                current_text = text.get("text", "") if isinstance(text, dict) else text
                if str(current_text or "").strip():
                    continue
                try:
                    block_confidence = float(getattr(blocks[index], "confidence", 1.0) or 0.0)
                except Exception:
                    block_confidence = 1.0
                if block_confidence >= 0.45:
                    crop = self._crop_block_from_page(page_rgb, blocks[index])
                    if self._crop_might_have_text(crop):
                        suppressed += 1
            stats["crop_fallback_suppressed"] = int(suppressed)
        if shadow_enabled:
            stats.update(
                {
                    "fallback_shadow_attempt_limit": int(shadow_limit),
                    "fallback_shadow_attempts_saved_or_would_skip": int(shadow_would_skip),
                    "fallback_shadow_recovered_after_limit": int(shadow_recovered_after_limit),
                    "fallback_shadow_full_page_already_resolved_count": int(full_page_mapped),
                }
            )

        if _env_bool("TRADUZAI_OCR_DEDUP", False):
            stats["ocr_dedup_removed"] = int(self._dedupe_ocr_records_in_place(texts, blocks))
        else:
            stats["ocr_dedup_removed"] = 0
        self._observation_state().stats = stats
        return texts

    @staticmethod
    def _record_text(record) -> str:
        if isinstance(record, dict):
            return str(record.get("text") or "")
        return str(record or "")

    @staticmethod
    def _clear_record_text(record):
        if isinstance(record, dict):
            updated = dict(record)
            updated["text"] = ""
            return updated
        return ""

    @staticmethod
    def _block_bbox(block) -> tuple[int, int, int, int]:
        try:
            xyxy = getattr(block, "xyxy")
            return tuple(int(v) for v in xyxy[:4])  # type: ignore[index]
        except Exception:
            return (
                int(getattr(block, "x1", 0)),
                int(getattr(block, "y1", 0)),
                int(getattr(block, "x2", 0)),
                int(getattr(block, "y2", 0)),
            )

    @staticmethod
    def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / float(area_a + area_b - inter)

    def _dedupe_ocr_records_in_place(self, records: list, blocks: list) -> int:
        removed = 0
        kept: list[int] = []
        for index, record in enumerate(records):
            text = self._record_text(record).strip()
            if not text:
                continue
            bbox = self._block_bbox(blocks[index])
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            duplicate_of: int | None = None
            for kept_index in kept:
                kept_text = self._record_text(records[kept_index]).strip()
                if not kept_text:
                    continue
                kept_bbox = self._block_bbox(blocks[kept_index])
                kept_cx = (kept_bbox[0] + kept_bbox[2]) / 2.0
                kept_cy = (kept_bbox[1] + kept_bbox[3]) / 2.0
                center_distance = ((cx - kept_cx) ** 2 + (cy - kept_cy) ** 2) ** 0.5
                similar = SequenceMatcher(None, text.upper(), kept_text.upper()).ratio()
                if (self._bbox_iou(bbox, kept_bbox) >= 0.85 or center_distance < 8.0) and similar >= 0.9:
                    duplicate_of = kept_index
                    break
            if duplicate_of is None:
                kept.append(index)
                continue
            try:
                current_conf = float(getattr(blocks[index], "confidence", 0.0) or 0.0)
                kept_conf = float(getattr(blocks[duplicate_of], "confidence", 0.0) or 0.0)
            except Exception:
                current_conf = 0.0
                kept_conf = 0.0
            if current_conf > kept_conf:
                dropped_index = duplicate_of
                dropped_record = records[duplicate_of]
                dropped_confidence = kept_conf
                records[duplicate_of] = self._clear_record_text(records[duplicate_of])
                kept.remove(duplicate_of)
                kept.append(index)
            else:
                dropped_index = index
                dropped_record = record
                dropped_confidence = current_conf
                records[index] = self._clear_record_text(record)
            dropped_text = self._record_text(dropped_record).strip()
            dropped_polygons = dropped_record.get("line_polygons") if isinstance(dropped_record, dict) else []
            self._append_observation_record(
                provider="ocr_dedupe",
                text=dropped_text,
                confidence=dropped_confidence,
                bbox=self._block_bbox(blocks[dropped_index]),
                line_polygons=dropped_polygons or [],
                accepted=False,
                rejection_reason="duplicate_ocr_record",
                variant="dedupe",
                coordinate_space="page",
                duplicate_of_index=int(index if dropped_index == duplicate_of else duplicate_of),
            )
            removed += 1
        return removed

    def recognize_single(self, crop: np.ndarray) -> str:
        """Reconhece texto em uma única imagem."""
        results = self.recognize_batch([crop])
        return results[0] if results else ""

    def _recognize_batch_impl(self, crops: list[np.ndarray]) -> list[str]:
        if self._backend == "manga-ocr":
            return self._manga_ocr_batch(crops)
        if self._backend == "easyocr":
            raise OcrBackendUnavailable("EasyOCR esta desativado; PaddleOCR e obrigatorio.")
        else:
            crop_indices = getattr(self._observation_state(), "active_crop_indices", None)
            return self._paddle_ocr_batch(crops, crop_indices=crop_indices)

    def _easyocr_batch(self, crops: list[np.ndarray]) -> list[str]:
        del crops
        raise OcrBackendUnavailable("EasyOCR esta desativado; PaddleOCR e obrigatorio.")

    def _manga_ocr_batch(self, crops: list[np.ndarray]) -> list[str]:
        """Inferência batched com manga-ocr."""
        texts: list[str] = []
        for crop in crops:
            if isinstance(crop, np.ndarray):
                img = Image.fromarray(crop).convert("RGB")
            else:
                img = crop.convert("RGB")
            
            # manga-ocr funciona melhor com imagens quadradas
            img = self._pad_to_square(img)
            physical_rgb = np.asarray(img, dtype=np.uint8)
            request = _legacy_local_ocr_request(
                physical_rgb,
                variant_id="manga_native",
                provider_family="manga-ocr",
            )
            _attempt, records = execute_hash_bound_provider_attempt(
                request=request,
                root_input_rgb=physical_rgb,
                actual_input_rgb=physical_rgb,
                transform_spec=_identity_transform_spec(),
                variant_id="manga_native",
                provider={
                    "processor": self._processor,
                    "model": self._model,
                    "tokenizer": self._tokenizer,
                    "device": self.device,
                    "half": self.half,
                },
                provider_mode="manga_ocr",
                expected_input_pixel_sha256=request.root_input_pixel_sha256,
                source="manga_ocr_crop",
                execution_recorder=self._record_raw_model_execution,
            )
            texts.append(" ".join(record.text for record in records if record.text).strip())

        return texts

    def _paddle_ocr_batch(
        self,
        crops: list[np.ndarray],
        *,
        crop_indices: list[int] | None = None,
    ) -> list[str]:
        """PaddleOCR processa uma imagem por vez (não tem batch nativo)."""
        texts = []
        for local_index, crop in enumerate(crops):
            crop_index = (
                int(crop_indices[local_index])
                if crop_indices is not None and local_index < len(crop_indices)
                else int(local_index)
            )
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                texts.append("")
                crop_height, crop_width = crop.shape[:2] if crop.ndim >= 2 else (0, 0)
                self._append_observation_record(
                    provider="paddle_crop",
                    text="",
                    confidence=0.0,
                    bbox=[0, 0, int(crop_width), int(crop_height)],
                    line_polygons=[],
                    accepted=False,
                    rejection_reason="invalid_crop",
                    variant="native",
                    coordinate_space="crop",
                    crop_index=crop_index,
                )
                continue
            texts.append(self._recognize_single_paddle_with_retry(crop, crop_index=crop_index))
        return texts

    def _recognize_single_paddle(self, crop: np.ndarray, *, cls: bool = False) -> str:
        try:
            physical_rgb = _normalize_ocr_rgb(crop)
            request = _legacy_local_ocr_request(
                physical_rgb,
                variant_id="crop_native",
                provider_family="paddleocr",
            )
            _attempt, records = execute_hash_bound_provider_attempt(
                request=request,
                root_input_rgb=physical_rgb,
                actual_input_rgb=physical_rgb,
                transform_spec=_identity_transform_spec(),
                variant_id="crop_native",
                provider=self._model.ocr,
                provider_kwargs={"det": True, "rec": True, "cls": bool(cls)},
                expected_input_pixel_sha256=request.root_input_pixel_sha256,
                source="paddle_crop",
                execution_recorder=self._record_raw_model_execution,
            )
            return " ".join(record.text for record in records if record.text).strip()
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            return ""

    @staticmethod
    def _score_ocr_candidate(text: str) -> tuple[int, int, int]:
        cleaned = str(text or "").strip()
        alnum = sum(ch.isalnum() for ch in cleaned)
        alpha = sum(ch.isalpha() for ch in cleaned)
        return (alnum, alpha, len(cleaned))

    def _build_paddle_retry_variants(self, crop: np.ndarray) -> list[np.ndarray]:
        variants: list[np.ndarray] = []
        up2 = _resize_for_ocr_preprocess(
            crop,
            (max(1, int(round(crop.shape[1] * 2.0))), max(1, int(round(crop.shape[0] * 2.0)))),
            interpolation=cv2.INTER_CUBIC,
        )
        variants.append(up2)

        # Reuse up2 for grayscale base (avoid redundant resize)
        gray_up2 = cv2.cvtColor(up2, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray_up2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB))

        blur = cv2.GaussianBlur(gray_up2, (0, 0), sigmaX=1.2)
        sharp = cv2.addWeighted(gray_up2, 1.8, blur, -0.8, 0)
        sharp_up15 = _resize_for_ocr_preprocess(
            sharp,
            (max(1, int(round(sharp.shape[1] * 1.5))), max(1, int(round(sharp.shape[0] * 1.5)))),
            interpolation=cv2.INTER_CUBIC,
        )
        variants.append(cv2.cvtColor(sharp_up15, cv2.COLOR_GRAY2RGB))
        return variants

    @staticmethod
    def _detect_dot_run_fallback(crop: np.ndarray) -> str:
        if crop.size == 0:
            return ""

        height, width = crop.shape[:2]
        if height < 10 or width < 18:
            return ""

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        components: list[dict[str, float]] = []
        max_component_area = max(18, int(height * width * 0.08))
        for index in range(1, component_count):
            x, y, w, h, area = stats[index].tolist()
            if area < 6 or area > max_component_area:
                continue
            if w <= 0 or h <= 0:
                continue
            ratio = max(w / float(h), h / float(w))
            if ratio > 1.8:
                continue
            components.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "area": float(area),
                    "cx": float(centroids[index][0]),
                    "cy": float(centroids[index][1]),
                }
            )

        if not 3 <= len(components) <= 8:
            return ""

        components.sort(key=lambda item: item["cx"])
        ys = [item["cy"] for item in components]
        if max(ys) - min(ys) > max(4.0, height * 0.16):
            return ""

        widths = [item["w"] for item in components]
        heights = [item["h"] for item in components]
        if max(widths) > min(widths) * 1.8 or max(heights) > min(heights) * 1.8:
            return ""

        gaps = [
            components[index + 1]["cx"] - components[index]["cx"]
            for index in range(len(components) - 1)
        ]
        if any(gap <= 0 for gap in gaps):
            return ""
        if max(gaps) > max(16.0, min(gaps) * 2.2):
            return ""

        total_area = sum(item["area"] for item in components)
        if total_area > height * width * 0.22:
            return ""

        return "." * len(components)

    def _recognize_single_paddle_with_retry(
        self,
        crop: np.ndarray,
        *,
        observation_bbox=None,
        observation_line_polygons=None,
        observation_coordinate_space: str = "crop",
        crop_index: int | None = None,
    ) -> str:
        crop_height, crop_width = (
            crop.shape[:2]
            if isinstance(crop, np.ndarray) and crop.ndim >= 2
            else (0, 0)
        )
        record_bbox = _coerce_bbox(observation_bbox) or [0, 0, int(crop_width), int(crop_height)]
        attempts: list[tuple[str, str]] = []

        def _record_attempts(selected_index: int | None = None) -> None:
            for attempt_index, (variant_name, attempt_text) in enumerate(attempts):
                cleaned_attempt = str(attempt_text or "").strip()
                is_selected = selected_index is not None and attempt_index == selected_index
                if is_selected:
                    rejection_reason = None
                elif not cleaned_attempt:
                    rejection_reason = "empty_result"
                else:
                    rejection_reason = "lower_score"
                self._append_observation_record(
                    provider="paddle_crop",
                    text=cleaned_attempt,
                    confidence=0.0,
                    bbox=record_bbox,
                    line_polygons=observation_line_polygons or [],
                    accepted=is_selected,
                    rejection_reason=rejection_reason,
                    variant=variant_name,
                    coordinate_space=observation_coordinate_space,
                    crop_index=crop_index,
                )

        use_angle_cls = bool(getattr(self, "_paddle_use_angle_cls", False))
        text = self._recognize_single_paddle(crop, cls=use_angle_cls)
        attempts.append(("native", str(text or "")))
        if text:
            _record_attempts(selected_index=0)
            return text

        best_text = ""
        best_score = self._score_ocr_candidate("")
        best_attempt_index: int | None = None
        for variant_index, variant in enumerate(self._build_paddle_retry_variants(crop), start=1):
            candidate = self._recognize_single_paddle(variant, cls=use_angle_cls)
            attempts.append((f"retry_{variant_index}", str(candidate or "")))
            score = self._score_ocr_candidate(candidate)
            if score > best_score:
                best_score = score
                best_text = candidate
                best_attempt_index = len(attempts) - 1
            # Early exit: good enough result, skip remaining variants
            if score >= (3, 3, 4):
                break
        if best_text.strip():
            _record_attempts(selected_index=best_attempt_index)
            return best_text.strip()
        _record_attempts(selected_index=None)
        dot_fallback = self._detect_dot_run_fallback(crop)
        self._append_observation_record(
            provider="paddle_crop",
            text=dot_fallback,
            confidence=0.0,
            bbox=record_bbox,
            line_polygons=observation_line_polygons or [],
            accepted=bool(str(dot_fallback or "").strip()),
            rejection_reason=None if str(dot_fallback or "").strip() else "empty_result",
            variant="dot_run",
            coordinate_space=observation_coordinate_space,
            crop_index=crop_index,
        )
        return dot_fallback

    @staticmethod
    def _crop_block_from_page(page_rgb: np.ndarray, block, padding: int = 4) -> np.ndarray:
        height, width = page_rgb.shape[:2]
        try:
            x1 = int(getattr(block, "x1"))
            y1 = int(getattr(block, "y1"))
            x2 = int(getattr(block, "x2"))
            y2 = int(getattr(block, "y2"))
        except Exception:
            xyxy = getattr(block, "xyxy", (0, 0, 0, 0))
            x1, y1, x2, y2 = [int(v) for v in xyxy]

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        if _env_bool("TRADUZAI_OCR_ADAPTIVE_CROP_PADDING", False):
            pad = max(2, int(round(min(box_w, box_h) * 0.04)))
        else:
            pad = int(padding)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return page_rgb[y1:y2, x1:x2]

    @staticmethod
    def _crop_might_have_text(crop_rgb: np.ndarray) -> bool:
        if not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0:
            return False

        height, width = crop_rgb.shape[:2]
        if height < 10 or width < 10:
            return False

        if len(crop_rgb.shape) == 3 and crop_rgb.shape[2] >= 3:
            gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop_rgb.astype(np.uint8, copy=False)

        max_dim = max(height, width)
        if max_dim > 256:
            scale = 256.0 / float(max_dim)
            resized = _resize_for_ocr_preprocess(
                gray,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = gray

        std = float(np.std(resized))
        edges = cv2.Canny(resized, 80, 160)
        edge_density = float(np.count_nonzero(edges)) / float(max(1, resized.size))
        dark_ratio = float(np.mean(resized < 150))

        if std < 9.0 and edge_density < 0.006:
            return False
        if dark_ratio < 0.015 and edge_density < 0.004:
            return False
        return True

    @staticmethod
    def _bbox_intersection_area(a: list[int], b: list[int]) -> int:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0
        return int((ix2 - ix1) * (iy2 - iy1))

    @staticmethod
    def _rotate_bound_with_inverse(image: np.ndarray, rotation_deg: float) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if not isinstance(image, np.ndarray) or image.size == 0:
            return None, None
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return None, None
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(rotation_deg), 1.0)
        cos = abs(float(matrix[0, 0]))
        sin = abs(float(matrix[0, 1]))
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        if new_width <= 0 or new_height <= 0:
            return None, None
        matrix[0, 2] += (new_width / 2.0) - (width / 2.0)
        matrix[1, 2] += (new_height / 2.0) - (height / 2.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderValue=(255, 255, 255),
        )
        return rotated, cv2.invertAffineTransform(matrix)

    @staticmethod
    def _map_rotated_box_to_original_polygon(box, inverse_matrix: np.ndarray, offset_x: int, offset_y: int) -> list[list[int]]:
        polygon: list[list[int]] = []
        if inverse_matrix is None:
            return polygon
        for point in box or []:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except Exception:
                continue
            mapped_x = (float(inverse_matrix[0, 0]) * x) + (float(inverse_matrix[0, 1]) * y) + float(inverse_matrix[0, 2])
            mapped_y = (float(inverse_matrix[1, 0]) * x) + (float(inverse_matrix[1, 1]) * y) + float(inverse_matrix[1, 2])
            polygon.append([int(round(mapped_x + offset_x)), int(round(mapped_y + offset_y))])
        return polygon

    @staticmethod
    def _join_line_entries(lines: list[dict]) -> str:
        return " ".join(entry["text"] for entry in lines if str(entry.get("text", "")).strip()).strip()

    @staticmethod
    def _ocr_text_gain_score(text: str) -> tuple[int, int, int]:
        cleaned = str(text or "").strip()
        alnum = sum(ch.isalnum() for ch in cleaned)
        words = len(re.findall(r"[A-Za-z0-9']+", cleaned))
        return alnum, words, len(cleaned)

    def _recognize_skewed_block_lines(
        self,
        page_bgr: np.ndarray,
        block_bbox: list[int],
        rotation_deg: float,
    ) -> list[dict]:
        if abs(float(rotation_deg or 0.0)) < SKEWED_TEXT_MIN_ROTATION_DEG:
            return []
        if abs(float(rotation_deg or 0.0)) > SKEWED_TEXT_MAX_DESKEW_DEG:
            return []
        if not isinstance(page_bgr, np.ndarray) or page_bgr.size == 0:
            return []
        page_h, page_w = page_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in block_bbox]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad = max(12, int(round(min(box_w, box_h) * 0.08)))
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(page_w, x2 + pad)
        cy2 = min(page_h, y2 + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            return []
        crop = page_bgr[cy1:cy2, cx1:cx2]
        rotated, inverse_matrix = self._rotate_bound_with_inverse(crop, float(rotation_deg))
        if rotated is None or inverse_matrix is None:
            return []
        try:
            physical_rgb = _normalize_ocr_rgb(rotated)
            request = _legacy_local_ocr_request(
                physical_rgb,
                variant_id=f"deskew_{float(rotation_deg):.2f}",
                provider_family="paddleocr",
            )
            _attempt, provider_records = execute_hash_bound_provider_attempt(
                request=request,
                root_input_rgb=physical_rgb,
                actual_input_rgb=physical_rgb,
                transform_spec=_identity_transform_spec(),
                variant_id=f"deskew_{float(rotation_deg):.2f}",
                provider=self._model.ocr,
                provider_kwargs={"det": True, "rec": True, "cls": False},
                expected_input_pixel_sha256=request.root_input_pixel_sha256,
                source="paddle_skewed_recovery",
                execution_recorder=self._record_raw_model_execution,
            )
        except Exception as exc:
            logger.debug("OCR deskew para texto inclinado falhou: %s", exc)
            return []
        provider_raw = provider_records.raw_result
        raw_lines = provider_raw[0] if isinstance(provider_raw, list) and provider_raw else []
        recovered: list[dict] = []
        for item in raw_lines or []:
            if not item or len(item) < 2:
                self._append_observation_record(
                    provider="paddle_skewed_recovery",
                    text="",
                    confidence=0.0,
                    bbox=block_bbox,
                    line_polygons=[],
                    accepted=False,
                    rejection_reason="invalid_provider_item",
                    variant=f"deskew_{float(rotation_deg):.2f}",
                    coordinate_space="page",
                )
                continue
            box = item[0]
            meta = item[1]
            provider_text = meta[0] if isinstance(meta, (list, tuple)) and len(meta) >= 1 else ""
            text = str(provider_text or "").strip()
            if not text:
                self._append_observation_record(
                    provider="paddle_skewed_recovery",
                    text="",
                    raw_text=str(provider_text or ""),
                    confidence=0.0,
                    bbox=block_bbox,
                    line_polygons=[],
                    accepted=False,
                    rejection_reason="empty_text",
                    variant=f"deskew_{float(rotation_deg):.2f}",
                    coordinate_space="page",
                )
                continue
            confidence = float(meta[1]) if isinstance(meta, (list, tuple)) and len(meta) >= 2 else 0.0
            if confidence < 0.45:
                self._append_observation_record(
                    provider="paddle_skewed_recovery",
                    text=text,
                    raw_text=str(provider_text or ""),
                    confidence=confidence,
                    bbox=block_bbox,
                    line_polygons=[],
                    accepted=False,
                    rejection_reason="below_min_confidence",
                    variant=f"deskew_{float(rotation_deg):.2f}",
                    coordinate_space="page",
                )
                continue
            rotated_polygon = _normalize_line_polygons([box])
            rotated_line_bbox = _bbox_from_polygons(rotated_polygon) or []
            polygon = self._map_rotated_box_to_original_polygon(box, inverse_matrix, cx1, cy1)
            polygon = _normalize_line_polygons([polygon])
            normalized_polygon = polygon[0] if polygon else []
            line_bbox = _bbox_from_polygons([normalized_polygon]) if normalized_polygon else None
            if not line_bbox:
                self._append_observation_record(
                    provider="paddle_skewed_recovery",
                    text=text,
                    raw_text=str(provider_text or ""),
                    confidence=confidence,
                    bbox=block_bbox,
                    line_polygons=[],
                    accepted=False,
                    rejection_reason="invalid_polygon",
                    variant=f"deskew_{float(rotation_deg):.2f}",
                    coordinate_space="page",
                )
                continue
            self._append_observation_record(
                provider="paddle_skewed_recovery",
                text=text,
                raw_text=str(provider_text or ""),
                confidence=confidence,
                bbox=line_bbox,
                line_polygons=[normalized_polygon],
                accepted=True,
                rejection_reason=None,
                variant=f"deskew_{float(rotation_deg):.2f}",
                coordinate_space="page",
            )
            recovered.append(
                {
                    "line_bbox": line_bbox,
                    "text": text,
                    "line_polygon": normalized_polygon,
                    "confidence": confidence,
                    "_rotated_line_bbox": rotated_line_bbox,
                }
            )
        recovered.sort(
            key=lambda entry: (
                (entry.get("_rotated_line_bbox") or [0, 0, 0, 0])[1],
                (entry.get("_rotated_line_bbox") or [0, 0, 0, 0])[0],
            )
        )
        return recovered

    def _should_replace_with_skewed_recovery(self, current_lines: list[dict], recovered_lines: list[dict]) -> bool:
        if len(recovered_lines) < 2:
            return False
        current_text = self._join_line_entries(current_lines)
        recovered_text = self._join_line_entries(recovered_lines)
        if not recovered_text.strip():
            return False
        current_score = self._ocr_text_gain_score(current_text)
        recovered_score = self._ocr_text_gain_score(recovered_text)
        if recovered_score[0] >= current_score[0] + max(8, int(current_score[0] * 0.12)):
            return True
        if len(recovered_lines) > len(current_lines) and recovered_score[0] >= int(current_score[0] * 0.95):
            return True
        return False

    def _paddle_ocr_full_page_to_blocks(
        self,
        page_bgr: np.ndarray,
        blocks: list,
        allow_sparse_mapping: bool = False,
    ) -> list[dict] | None:
        model_input = page_bgr
        input_h, input_w = page_bgr.shape[:2]
        scale_x = 1.0
        scale_y = 1.0
        max_side = _paddle_full_page_max_side()
        longest_side = max(input_h, input_w)
        if max_side > 0 and longest_side > max_side:
            downscale = max_side / float(longest_side)
            scaled_w = max(1, int(round(input_w * downscale)))
            scaled_h = max(1, int(round(input_h * downscale)))
            if scaled_w < input_w or scaled_h < input_h:
                model_input = _resize_for_ocr_preprocess(
                    page_bgr,
                    (scaled_w, scaled_h),
                    interpolation=cv2.INTER_AREA,
                )
                scale_x = scaled_w / float(max(1, input_w))
                scale_y = scaled_h / float(max(1, input_h))
        self._observation_state().full_page_lines = []
        try:
            root_rgb = _normalize_ocr_rgb(page_bgr)
            physical_rgb = _normalize_ocr_rgb(model_input)
            if physical_rgb.shape[:2] == root_rgb.shape[:2]:
                transform_spec = _identity_transform_spec()
            else:
                transform_spec = OCRTransformSpec.build((
                    OCRTransformOperation(
                        kind="resize",
                        output_size=(int(physical_rgb.shape[1]), int(physical_rgb.shape[0])),
                        interpolation="area",
                    ),
                ))
            request = _legacy_local_ocr_request(
                root_rgb,
                variant_id="full_page",
                provider_family="paddleocr",
            )
            _attempt, provider_records = execute_hash_bound_provider_attempt(
                request=request,
                root_input_rgb=root_rgb,
                actual_input_rgb=physical_rgb,
                transform_spec=transform_spec,
                variant_id="full_page",
                provider=self._model.ocr,
                provider_kwargs={"det": True, "rec": True, "cls": False},
                expected_input_pixel_sha256=canonical_page_sha256(physical_rgb),
                source="paddle_full_page",
                execution_recorder=self._record_raw_model_execution,
            )
        except Exception as exc:
            logger.warning("PaddleOCR full-page falhou; fallback por crop: %s", exc)
            self._append_observation_record(
                provider="paddle_full_page",
                text="",
                confidence=0.0,
                bbox=[],
                line_polygons=[],
                accepted=False,
                rejection_reason="provider_error",
                variant="native",
                coordinate_space="page",
                error=str(exc),
            )
            return None

        provider_raw = provider_records.raw_result
        raw_lines = provider_raw[0] if isinstance(provider_raw, list) and provider_raw else []
        if not raw_lines:
            self._append_observation_record(
                provider="paddle_full_page",
                text="",
                confidence=0.0,
                bbox=[],
                line_polygons=[],
                accepted=False,
                rejection_reason="no_output",
                variant="native",
                coordinate_space="page",
            )
            return None

        original_block_bboxes: list[list[int]] = []
        for block in blocks:
            try:
                original_block_bboxes.append(
                    [
                        int(getattr(block, "x1")),
                        int(getattr(block, "y1")),
                        int(getattr(block, "x2")),
                        int(getattr(block, "y2")),
                    ]
                )
            except Exception:
                xyxy = getattr(block, "xyxy", (0, 0, 0, 0))
                original_block_bboxes.append([int(v) for v in xyxy])
        block_bboxes: list[list[int]] = [list(bbox) for bbox in original_block_bboxes]
        if scale_x != 1.0 or scale_y != 1.0:
            block_bboxes = [_scale_bbox(bbox, scale_x, scale_y) for bbox in block_bboxes]

        page_rgb = page_bgr
        if len(page_bgr.shape) == 3 and page_bgr.shape[2] >= 3:
            try:
                page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                page_rgb = page_bgr

        assigned: list[list[dict]] = [[] for _ in blocks]
        full_page_line_records: list[dict] = []

        for item in raw_lines:
            if not item or len(item) < 2:
                self._append_observation_record(
                    provider="paddle_full_page",
                    text="",
                    confidence=0.0,
                    bbox=[],
                    line_polygons=[],
                    accepted=False,
                    rejection_reason="invalid_provider_item",
                    variant="native",
                    coordinate_space="page",
                )
                continue
            box = item[0]
            meta = item[1]
            text = meta[0] if isinstance(meta, (list, tuple)) and len(meta) >= 1 else ""
            try:
                observation_confidence = (
                    float(meta[1])
                    if isinstance(meta, (list, tuple)) and len(meta) >= 2
                    else 0.0
                )
            except (TypeError, ValueError):
                observation_confidence = 0.0
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                self._append_observation_record(
                    provider="paddle_full_page",
                    text=str(text or "").strip(),
                    raw_text=str(text or ""),
                    confidence=observation_confidence,
                    bbox=[],
                    line_polygons=[],
                    accepted=False,
                    rejection_reason="invalid_polygon",
                    variant="native",
                    coordinate_space="page",
                )
                continue

            xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
            ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
            if not xs or not ys:
                self._append_observation_record(
                    provider="paddle_full_page",
                    text=str(text or "").strip(),
                    raw_text=str(text or ""),
                    confidence=observation_confidence,
                    bbox=[],
                    line_polygons=[],
                    accepted=False,
                    rejection_reason="invalid_polygon",
                    variant="native",
                    coordinate_space="page",
                )
                continue

            scaled_line_bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            score_line_bbox = scaled_line_bbox
            line_bbox = scaled_line_bbox
            if scale_x != 1.0 or scale_y != 1.0:
                line_bbox = _scale_bbox(scaled_line_bbox, 1.0 / scale_x, 1.0 / scale_y)
            lx1, ly1, lx2, ly2 = score_line_bbox
            line_area = max(1, (lx2 - lx1) * (ly2 - ly1))
            line_polygon = _normalize_line_polygons([box])
            normalized_polygon = line_polygon[0] if line_polygon else []
            if normalized_polygon and (scale_x != 1.0 or scale_y != 1.0):
                normalized_polygon = _scale_polygon_points(normalized_polygon, 1.0 / scale_x, 1.0 / scale_y)

            if not str(text or "").strip():
                self._append_observation_record(
                    provider="paddle_full_page",
                    text="",
                    raw_text=str(text or ""),
                    confidence=observation_confidence,
                    bbox=line_bbox,
                    line_polygons=[normalized_polygon] if normalized_polygon else [],
                    accepted=False,
                    rejection_reason="empty_text",
                    variant="native",
                    coordinate_space="page",
                )
                continue

            confidence = float(meta[1]) if isinstance(meta, (list, tuple)) and len(meta) >= 2 else 0.0
            full_page_line_records.append(
                {
                    "text": str(text).strip(),
                    "source_bbox": list(line_bbox),
                    "bbox": list(line_bbox),
                    "text_pixel_bbox": _derive_text_pixel_bbox(
                        page_rgb,
                        line_bbox,
                        [normalized_polygon] if normalized_polygon else [],
                    ) or list(line_bbox),
                    "line_polygons": [normalized_polygon] if normalized_polygon else [],
                    "confidence": confidence,
                    "detector": "paddle_full_page_line",
                }
            )

            best_index = None
            best_score = 0.0
            for idx, block_bbox in enumerate(block_bboxes):
                bx1, by1, bx2, by2 = block_bbox
                inter = self._bbox_intersection_area(score_line_bbox, block_bbox)
                if inter <= 0:
                    continue
                block_area = max(1, (bx2 - bx1) * (by2 - by1))
                score = inter / float(max(1, min(line_area, block_area)))
                if score > best_score:
                    best_score = score
                    best_index = idx

            if best_index is not None and best_score >= 0.18:
                assigned[best_index].append(
                    {
                        "line_bbox": line_bbox,
                        "text": str(text).strip(),
                        "line_polygon": normalized_polygon,
                        "confidence": confidence,
                        "assignment_score": best_score,
                    }
                )
            mapping_accepted = best_index is not None and best_score >= 0.18
            self._append_observation_record(
                provider="paddle_full_page",
                text=str(text).strip(),
                raw_text=str(text),
                confidence=confidence,
                bbox=line_bbox,
                line_polygons=[normalized_polygon] if normalized_polygon else [],
                accepted=mapping_accepted,
                rejection_reason=None if mapping_accepted else "unmapped_to_detector_block",
                variant="native",
                coordinate_space="page",
                mapped_block_index=int(best_index) if mapping_accepted else None,
                mapping_score=round(float(best_score), 6),
            )

        texts: list[dict] = []
        non_empty = 0
        for block_index, lines in enumerate(assigned):
            lines.sort(key=lambda entry: (entry["line_bbox"][1], entry["line_bbox"][0]))
            assignment_audit = None
            if _ocr_assignment_audit_enabled():
                assignment_audit = _build_ocr_assignment_audit(
                    original_block_bboxes[block_index],
                    lines,
                )
            deskew_recovered = False
            pre_recovery_polygons = [entry["line_polygon"] for entry in lines if entry.get("line_polygon")]
            pre_recovery_rotation = infer_rotation_deg_from_line_polygons(pre_recovery_polygons)
            if abs(pre_recovery_rotation) >= SKEWED_TEXT_MIN_ROTATION_DEG and abs(pre_recovery_rotation) <= SKEWED_TEXT_MAX_DESKEW_DEG:
                recovered_lines = self._recognize_skewed_block_lines(
                    page_bgr,
                    original_block_bboxes[block_index] if block_index < len(original_block_bboxes) else block_bboxes[block_index],
                    pre_recovery_rotation,
                )
                if self._should_replace_with_skewed_recovery(lines, recovered_lines):
                    lines = recovered_lines
                    deskew_recovered = True
            joined = " ".join(entry["text"] for entry in lines if str(entry.get("text", "")).strip()).strip()
            if joined:
                non_empty += 1
            combined_polygons = [entry["line_polygon"] for entry in lines if entry.get("line_polygon")]
            combined_line_bboxes = [entry["line_bbox"] for entry in lines if entry.get("line_bbox")]
            if combined_line_bboxes:
                source_bbox = [
                    min(box[0] for box in combined_line_bboxes),
                    min(box[1] for box in combined_line_bboxes),
                    max(box[2] for box in combined_line_bboxes),
                    max(box[3] for box in combined_line_bboxes),
                ]
            else:
                source_bbox = []
            rotation_deg = infer_rotation_deg_from_line_polygons(combined_polygons)
            record = {
                "text": joined,
                "line_texts": [entry["text"] for entry in lines if str(entry.get("text", "")).strip()],
                "source_bbox": source_bbox,
                "line_polygons": combined_polygons,
                "text_pixel_bbox": _derive_text_pixel_bbox(page_rgb, source_bbox, combined_polygons) or source_bbox,
            }
            if rotation_deg != 0.0:
                record["rotation_deg"] = rotation_deg
                record["rotation_source"] = "line_polygons"
            _attach_rotated_text_metadata(record)
            if deskew_recovered:
                record["ocr_recovery"] = "skewed_block_deskew"
                record["qa_flags"] = ["skewed_text_deskew_recovery"]
            if assignment_audit is not None:
                record["_ocr_assignment_audit"] = assignment_audit
            texts.append(
                record
            )

        self._observation_state().full_page_lines = full_page_line_records
        if non_empty == 0:
            return None

        # Se associou texto em poucos blocos, o mapeamento falhou e devemos preservar qualidade
        # voltando ao caminho antigo por crop.
        if (
            not allow_sparse_mapping
            and len(blocks) >= 3
            and non_empty / max(1, len(blocks)) < 0.5
        ):
            return None

        return texts

    def recognize_rotated_full_page_lines(
        self,
        page_rgb: np.ndarray,
        rotations: tuple[int, ...] = (90, 270),
        min_confidence: float = 0.80,
    ) -> list[dict]:
        self._reset_observation_records()
        if getattr(self, "_backend", "") != "paddleocr":
            return []
        if not isinstance(page_rgb, np.ndarray) or page_rgb.size == 0:
            return []

        original_height, original_width = page_rgb.shape[:2]
        records: list[dict] = []
        for rotation_deg in rotations:
            rotated = _rotate_orthogonal(page_rgb, int(rotation_deg))
            try:
                physical_rgb = _normalize_ocr_rgb(rotated)
                request = _legacy_local_ocr_request(
                    physical_rgb,
                    variant_id=f"rotation_{int(rotation_deg) % 360}",
                    provider_family="paddleocr",
                )
                _attempt, provider_records = execute_hash_bound_provider_attempt(
                    request=request,
                    root_input_rgb=physical_rgb,
                    actual_input_rgb=physical_rgb,
                    transform_spec=_identity_transform_spec(),
                    variant_id=f"rotation_{int(rotation_deg) % 360}",
                    provider=self._model.ocr,
                    provider_kwargs={"det": True, "rec": True, "cls": False},
                    expected_input_pixel_sha256=request.root_input_pixel_sha256,
                    source="paddle_rotated_full_page",
                    execution_recorder=self._record_raw_model_execution,
                )
            except Exception as exc:
                logger.debug("PaddleOCR rotated-page recovery falhou (%s): %s", rotation_deg, exc)
                continue
            provider_raw = provider_records.raw_result
            raw_lines = provider_raw[0] if isinstance(provider_raw, list) and provider_raw else []
            if raw_lines is None:
                raw_lines = []
            for item in raw_lines:
                if not item or len(item) < 2:
                    self._append_observation_record(
                        provider="paddle_rotated_full_page",
                        text="",
                        confidence=0.0,
                        bbox=[],
                        line_polygons=[],
                        accepted=False,
                        rejection_reason="invalid_provider_item",
                        variant=f"rotation_{int(rotation_deg) % 360}",
                        coordinate_space="page",
                    )
                    continue
                box = item[0]
                meta = item[1]
                provider_text = meta[0] if isinstance(meta, (list, tuple)) and len(meta) >= 1 else ""
                text = _repair_rotated_ocr_edge_clipping(str(provider_text or "").strip())
                try:
                    confidence = float(meta[1]) if isinstance(meta, (list, tuple)) and len(meta) >= 2 else 0.0
                except Exception:
                    confidence = 0.0
                rotated_polygons = _normalize_line_polygons([box])
                if not rotated_polygons:
                    self._append_observation_record(
                        provider="paddle_rotated_full_page",
                        text=text,
                        raw_text=str(provider_text or ""),
                        confidence=confidence,
                        bbox=[],
                        line_polygons=[],
                        accepted=False,
                        rejection_reason="invalid_polygon",
                        variant=f"rotation_{int(rotation_deg) % 360}",
                        coordinate_space="page",
                    )
                    continue
                polygon = _unrotate_orthogonal_polygon(
                    rotated_polygons[0],
                    original_width=original_width,
                    original_height=original_height,
                    rotation_deg=int(rotation_deg),
                )
                if len(polygon) < 4:
                    self._append_observation_record(
                        provider="paddle_rotated_full_page",
                        text=text,
                        raw_text=str(provider_text or ""),
                        confidence=confidence,
                        bbox=[],
                        line_polygons=[],
                        accepted=False,
                        rejection_reason="invalid_polygon",
                        variant=f"rotation_{int(rotation_deg) % 360}",
                        coordinate_space="page",
                    )
                    continue
                line_polygons = [polygon]
                source_bbox = _bbox_from_polygons(line_polygons)
                if source_bbox is None:
                    self._append_observation_record(
                        provider="paddle_rotated_full_page",
                        text=text,
                        raw_text=str(provider_text or ""),
                        confidence=confidence,
                        bbox=[],
                        line_polygons=line_polygons,
                        accepted=False,
                        rejection_reason="invalid_bbox",
                        variant=f"rotation_{int(rotation_deg) % 360}",
                        coordinate_space="page",
                    )
                    continue
                rotation = infer_rotation_deg_from_line_polygons(line_polygons)
                rejection_reason = None
                if not text:
                    rejection_reason = "empty_text"
                elif confidence < float(min_confidence):
                    rejection_reason = "below_min_confidence"
                elif abs(rotation) < 35.0:
                    rejection_reason = "not_rotated_enough"
                self._append_observation_record(
                    provider="paddle_rotated_full_page",
                    text=text,
                    raw_text=str(provider_text or ""),
                    confidence=confidence,
                    bbox=source_bbox,
                    line_polygons=line_polygons,
                    accepted=rejection_reason is None,
                    rejection_reason=rejection_reason,
                    variant=f"rotation_{int(rotation_deg) % 360}",
                    coordinate_space="page",
                    rotation_deg=rotation,
                )
                if rejection_reason is not None:
                    continue
                xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                rotated_line_bbox = [
                    int(min(xs)),
                    int(min(ys)),
                    int(max(xs)),
                    int(max(ys)),
                ] if xs and ys else [0, 0, 0, 0]
                records.append(
                    _attach_rotated_text_metadata({
                        "text": text,
                        "source_bbox": source_bbox,
                        "bbox": source_bbox,
                        "line_polygons": line_polygons,
                        "text_pixel_bbox": _derive_text_pixel_bbox(page_rgb, source_bbox, line_polygons) or source_bbox,
                        "confidence": round(confidence, 3),
                        "rotation_deg": rotation,
                        "rotation_source": "rotated_page_ocr",
                        "detector": "rotated_full_page_recovery",
                        "_rotated_ocr_angle": int(rotation_deg) % 360,
                        "_rotated_line_bbox": rotated_line_bbox,
                    })
                )

        if not records:
            return []
        return _group_rotated_ocr_records(records)

    @staticmethod
    def _pad_to_square(img: Image.Image, size: int = 224) -> Image.Image:
        """Adiciona padding para tornar quadrado (melhora manga-ocr)."""
        w, h = img.size
        max_dim = max(w, h, 1)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        
        new_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        new_img.paste(img, (pad_w, pad_h))
        return new_img.resize((size, size), Image.LANCZOS)

    def unload(self):
        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
