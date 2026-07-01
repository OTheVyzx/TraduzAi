from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Callable

import cv2
import numpy as np


SFX_OCR_LANGUAGES = ("ko", "ja", "zh")
HANGUL_RE = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
CJK_RE = re.compile(r"[\u3000-\u303F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")


Recognizer = Callable[[np.ndarray, str], list[dict[str, Any]]]
_DEFAULT_RECOGNIZER_UNAVAILABLE: str | None = None


def probe_sfx_candidate_ocr(
    candidate: dict[str, Any],
    image_rgb: np.ndarray | None,
    *,
    recognizer: Recognizer | None = None,
    languages: tuple[str, ...] = SFX_OCR_LANGUAGES,
    min_confidence: float = 0.78,
) -> dict[str, Any]:
    """Run focused CJK OCR on a visual SFX candidate crop when possible."""

    result = deepcopy(candidate)
    if str(result.get("recognized_text") or result.get("text") or result.get("original") or "").strip():
        return result
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return _attach_sfx_ocr_status(result, status="no_image", languages=languages)
    bbox = _coerce_bbox(result.get("bbox") or result.get("text_pixel_bbox"))
    if bbox is None:
        return _attach_sfx_ocr_status(result, status="invalid_bbox", languages=languages)

    crop_rgb = build_sfx_ocr_crop(result, image_rgb)
    if crop_rgb.size == 0:
        return _attach_sfx_ocr_status(result, status="empty_crop", languages=languages)

    read_fn = recognizer or _default_cjk_recognizer
    attempts: list[dict[str, Any]] = []
    for variant_name, variant_crop in build_sfx_ocr_crop_variants(result, image_rgb):
        if variant_crop.size == 0:
            continue
        for lang in languages:
            try:
                records = read_fn(variant_crop, lang)
            except Exception as exc:
                attempts.append({"lang": lang, "variant": variant_name, "status": "error", "error": str(exc)})
                continue
            normalized = [_normalize_record(record, lang, variant=variant_name) for record in records or []]
            normalized = [record for record in normalized if record["text"]]
            attempts.extend(normalized)

    best = _select_best_cjk_attempt(attempts)
    if best is None or float(best.get("confidence") or 0.0) < float(min_confidence):
        return _attach_sfx_ocr_status(result, status="no_confident_cjk", languages=languages, attempts=attempts)

    text = str(best.get("text") or "").strip()
    result["recognized_text"] = text
    result["text"] = text
    result["original"] = text
    result["ocr_confidence"] = float(best.get("confidence") or 0.0)
    result["sfx_ocr"] = {
        "status": "recognized",
        "source": "sfx_cjk_crop_probe",
        "lang": best.get("lang"),
        "confidence": float(best.get("confidence") or 0.0),
        "text": text,
        "languages": list(languages),
        "attempts": _summarize_attempts(attempts),
    }
    sfx = result.get("sfx") if isinstance(result.get("sfx"), dict) else {}
    result["sfx"] = {
        **sfx,
        "ocr_probe": "sfx_cjk_crop_probe",
        "ocr_lang": best.get("lang"),
        "ocr_confidence": float(best.get("confidence") or 0.0),
    }
    return result


def _default_cjk_recognizer(crop_rgb: np.ndarray, lang: str) -> list[dict[str, Any]]:
    global _DEFAULT_RECOGNIZER_UNAVAILABLE
    if _DEFAULT_RECOGNIZER_UNAVAILABLE:
        raise RuntimeError(_DEFAULT_RECOGNIZER_UNAVAILABLE)
    try:
        from ocr_legacy.detector import _check_gpu
        from ocr_legacy.recognizer_paddle import run_paddle_primary_recognition
        from vision_stack.ocr import normalize_paddleocr_language
    except Exception as exc:
        _DEFAULT_RECOGNIZER_UNAVAILABLE = f"PaddleOCR unavailable for SFX OCR probe: {exc}"
        raise RuntimeError(_DEFAULT_RECOGNIZER_UNAVAILABLE) from exc

    crop_bgr = cv2.cvtColor(_prepare_crop_for_ocr(crop_rgb), cv2.COLOR_RGB2BGR)
    try:
        return run_paddle_primary_recognition(crop_bgr, use_gpu=_check_gpu(), lang=normalize_paddleocr_language(lang))
    except Exception as exc:
        message = str(exc)
        if "indisponivel" in message.lower() or "unavailable" in message.lower() or "python 3.12" in message.lower():
            _DEFAULT_RECOGNIZER_UNAVAILABLE = f"PaddleOCR unavailable for SFX OCR probe: {exc}"
        raise


def _prepare_crop_for_ocr(crop_rgb: np.ndarray) -> np.ndarray:
    height, width = crop_rgb.shape[:2]
    max_dim = max(height, width)
    if max_dim < 320:
        scale = 320.0 / float(max(1, max_dim))
        crop_rgb = cv2.resize(
            crop_rgb,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    return crop_rgb


def build_sfx_ocr_crop(candidate: dict[str, Any], image_rgb: np.ndarray) -> np.ndarray:
    bbox = _coerce_bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox"))
    if bbox is None:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    crop = _crop_candidate(image_rgb, bbox)
    if crop.size == 0:
        return crop
    tight = _tighten_sfx_crop(crop, str((candidate.get("sfx") or {}).get("visual_source") or ""))
    return tight if tight.size else crop


def build_sfx_ocr_crop_variants(candidate: dict[str, Any], image_rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build OCR crop variants for stylized SFX without changing detector geometry."""

    base = build_sfx_ocr_crop(candidate, image_rgb)
    if base.size == 0:
        return []
    variants: list[tuple[str, np.ndarray]] = [("tight_rgb", _prepare_crop_for_ocr(base))]
    gray = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    variants.append(("contrast_rgb", _prepare_crop_for_ocr(cv2.cvtColor(contrast, cv2.COLOR_GRAY2RGB))))
    binary = _adaptive_binary_for_sfx_ocr(contrast)
    variants.append(("binary_dark_rgb", _prepare_crop_for_ocr(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))))
    variants.append(("binary_light_rgb", _prepare_crop_for_ocr(cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2RGB))))
    rotation = _candidate_rotation(candidate)
    if abs(rotation) >= 8.0:
        variants.append(("deskew_rgb", _prepare_crop_for_ocr(_rotate_crop(base, -rotation))))
    return _dedupe_crop_variants(variants)


def _adaptive_binary_for_sfx_ocr(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray.astype(np.uint8)
    window = max(15, min(41, ((min(gray.shape[:2]) // 5) | 1)))
    if window % 2 == 0:
        window += 1
    return cv2.adaptiveThreshold(
        gray.astype(np.uint8),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        window,
        5,
    )


def _candidate_rotation(candidate: dict[str, Any]) -> float:
    sfx = candidate.get("sfx") if isinstance(candidate.get("sfx"), dict) else {}
    style = sfx.get("style") if isinstance(sfx.get("style"), dict) else {}
    for value in (style.get("rotation_deg"), candidate.get("rotation_deg")):
        try:
            rotation = float(value)
        except (TypeError, ValueError):
            continue
        if -90.0 <= rotation <= 90.0:
            return rotation
    return 0.0


def _rotate_crop(crop_rgb: np.ndarray, angle: float) -> np.ndarray:
    height, width = crop_rgb.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(round(height * sin + width * cos))
    new_h = int(round(height * cos + width * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(
        crop_rgb.astype(np.uint8),
        matrix,
        (max(1, new_w), max(1, new_h)),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _dedupe_crop_variants(variants: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    deduped: list[tuple[str, np.ndarray]] = []
    signatures: set[tuple[int, int, int, int]] = set()
    for name, crop in variants:
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.astype(np.uint8)
        signature = (
            int(gray.shape[0]),
            int(gray.shape[1]),
            int(np.mean(gray)),
            int(np.std(gray)),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        deduped.append((name, crop))
    return deduped


def _crop_candidate(image_rgb: np.ndarray, bbox: list[int]) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = max(4, int(round(min(x2 - x1, y2 - y1) * 0.08)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    return image_rgb[y1:y2, x1:x2, :3]


def _tighten_sfx_crop(crop_rgb: np.ndarray, visual_source: str) -> np.ndarray:
    if not isinstance(crop_rgb, np.ndarray) or crop_rgb.ndim != 3 or crop_rgb.size == 0:
        return crop_rgb
    rgb_u8 = crop_rgb.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    rgb_i = rgb_u8.astype(np.int16)
    chroma = (np.max(rgb_i, axis=2) - np.min(rgb_i, axis=2)).astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)

    color = ((saturation >= 24.0) & (chroma >= 14.0)).astype(np.uint8) * 255
    edges = cv2.Canny(gray, 55, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    mask = cv2.bitwise_or(color, edges)

    if visual_source == "white_near_chroma":
        near_color = cv2.dilate(color, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
        white = ((gray >= 238) & near_color).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, white)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7)))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return crop_rgb
    height, width = crop_rgb.shape[:2]
    min_area = max(20, int(height * width * 0.003))
    boxes = []
    for label in range(1, count):
        x, y, w_box, h_box, area = [int(v) for v in stats[label].tolist()]
        if area < min_area:
            continue
        boxes.append((x, y, x + w_box, y + h_box))
    if not boxes:
        return crop_rgb
    x1 = max(0, min(box[0] for box in boxes) - 5)
    y1 = max(0, min(box[1] for box in boxes) - 5)
    x2 = min(width, max(box[2] for box in boxes) + 5)
    y2 = min(height, max(box[3] for box in boxes) + 5)
    if x2 <= x1 or y2 <= y1:
        return crop_rgb
    tight = crop_rgb[y1:y2, x1:x2, :3]
    if tight.shape[0] * tight.shape[1] < int(height * width * 0.08):
        return crop_rgb
    return tight


def _select_best_cjk_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [attempt for attempt in attempts if CJK_RE.search(str(attempt.get("text") or ""))]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            1 if HANGUL_RE.search(str(item.get("text") or "")) else 0,
            float(item.get("confidence") or 0.0),
            len(str(item.get("text") or "")),
        ),
    )


def _normalize_record(record: Any, lang: str, *, variant: str = "") -> dict[str, Any]:
    if isinstance(record, dict):
        text = str(record.get("text") or record.get("original") or "").strip()
        confidence = _as_float(record.get("confidence", record.get("ocr_confidence", 0.0)))
        return {"lang": lang, "variant": variant, "text": text, "confidence": confidence}
    if isinstance(record, (list, tuple)) and record:
        text = str(record[0] or "").strip()
        confidence = _as_float(record[1] if len(record) > 1 else 0.0)
        return {"lang": lang, "variant": variant, "text": text, "confidence": confidence}
    return {"lang": lang, "variant": variant, "text": "", "confidence": 0.0}


def _attach_sfx_ocr_status(
    candidate: dict[str, Any],
    *,
    status: str,
    languages: tuple[str, ...],
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = deepcopy(candidate)
    result["sfx_ocr"] = {
        "status": status,
        "source": "sfx_cjk_crop_probe",
        "languages": list(languages),
    }
    if attempts is not None:
        result["sfx_ocr"]["attempt_count"] = len(attempts)
        result["sfx_ocr"]["attempts"] = _summarize_attempts(attempts)
    return result


def _summarize_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        item = {
            "lang": attempt.get("lang"),
            "variant": attempt.get("variant"),
            "text": str(attempt.get("text") or "")[:80],
            "confidence": round(_as_float(attempt.get("confidence")), 4),
        }
        if attempt.get("status"):
            item["status"] = str(attempt.get("status"))
        if attempt.get("error"):
            item["error"] = str(attempt.get("error"))[:160]
        summarized.append(item)
    return summarized


def _coerce_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
